"""Public-only joint image/text training with selectable sequential surrogates.

No memory pool, formal queries or evaluation rank is accepted by this module.
All surrogate image graphs (including the DDIM path) are released one at a time.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.paths import MODEL_ROOT
from src.visual.encoders.backbone import OpenCLIPSurrogate
from src.visual.encoders.joint import ChunkedClipEncoder
from src.visual.robust_objectives import compensated_targets, softmin, transformed_batch
from src.visual._lucid.objective import local_ot_similarity_from_centers, token_kmeans
from src.visual.training_surrogates import (
    HELD_OUT_MODELS, select_training_surrogates, selection_from_config, spec_identity,
    validate_training_specs, validate_loaded_surrogates,
)

TRAINING_OBJECTIVE_KIND = 'sequential_public_joint_text_fg_ot_v3'


def extract_foreground_patch_tokens(
    patch_tokens: torch.Tensor | None,
    fg_mask: torch.Tensor | object | None,
    *,
    threshold: float = 0.50,
    min_patches: int = 1,
) -> torch.Tensor | None:
    """将前景掩膜对齐 ViT/SigLIP 空间网格并剔除 [CLS] 标记，只提取前景内部 Patch Tokens。"""
    if patch_tokens is None or not torch.is_tensor(patch_tokens) or patch_tokens.ndim != 3:
        return None
    n_tok = int(patch_tokens.shape[1])
    if n_tok < 4:
        return None
    s_no_cls = math.isqrt(n_tok - 1)
    s_raw = math.isqrt(n_tok)
    if s_no_cls * s_no_cls == n_tok - 1:
        spatial = patch_tokens[:, 1:, :]
        side = s_no_cls
    elif s_raw * s_raw == n_tok:
        spatial = patch_tokens
        side = s_raw
    else:
        return None

    if fg_mask is None:
        return F.normalize(spatial.float(), dim=-1)
    mask_t = torch.as_tensor(fg_mask, dtype=torch.float32, device=patch_tokens.device)
    if mask_t.ndim == 2:
        mask_t = mask_t[None, None]
    elif mask_t.ndim == 3:
        mask_t = mask_t[:, None]
    if float(mask_t.sum().detach()) <= 0.0:
        return F.normalize(spatial.float(), dim=-1)

    pooled = F.adaptive_avg_pool2d(mask_t, (side, side)).reshape(-1)
    sel = torch.nonzero(pooled >= float(threshold), as_tuple=False).reshape(-1)
    k_need = min(max(1, int(min_patches)), side * side)
    if int(sel.numel()) < k_need:
        sel = torch.sort(torch.topk(pooled, k=k_need).indices).values
    return F.normalize(spatial[:, sel, :].float(), dim=-1)


def load_training_surrogates(device='cuda', config_path=None, *, surrogate_set=None, selection=None):
    """Load only declared local models; dependency/model failures propagate."""
    selection = selection or select_training_surrogates(
        surrogate_set or ('custom' if config_path is not None else 'lucid'), config_path)
    if selection.profile == 'lucid':
        # Use Lucid's exact weight resolution, loader and preprocessing path.
        from src.visual._lucid.config import resolve_optimization
        from src.visual._lucid.surrogates import load_surrogates
        lucid = resolve_optimization(selection.source)
        if tuple(map(spec_identity, lucid.surrogates)) != tuple(map(spec_identity, selection.specs)):
            raise ValueError('原始像素代理配置在加载期间发生变化')
        models = load_surrogates(lucid.surrogates, device=device, precision=lucid.attack_precision,
            allow_downloads=False, allow_device_fallback=False,
            require_true_local_tokens=lucid.runtime.require_true_local_tokens,
            hash_weights=lucid.runtime.hash_model_weights)
        validate_loaded_surrogates(models, selection)
        if hashlib.sha256(selection.source.read_bytes()).hexdigest() != selection.source_sha256:
            raise ValueError('原始像素代理配置在加载期间发生变化')
        return models
    models = []
    for spec in selection.specs:
        # All relative paths use the relocatable model root, not the shell CWD.
        weight = Path(spec.weight_path) if spec.weight_path else None
        tokenizer = Path(spec.tokenizer_path) if spec.tokenizer_path else None
        weight = (MODEL_ROOT / weight if weight and not weight.is_absolute() else weight)
        if weight is None:
            from src.model_registry import resolve_model
            directory = Path(resolve_model(spec.hf_repo)) if spec.hf_repo else None
            weight = next((directory / name for name in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors')
                           if directory and (directory / name).is_file()), None)
        elif not weight.is_file() and weight.name in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors'):
            weight = next((weight.parent / name for name in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors')
                           if (weight.parent / name).is_file()), weight)
        if weight is None or not weight.is_file():
            raise FileNotFoundError('所选训练代理缺少本地权重：' + spec.name + '/' + spec.pretrained)
        spec = replace(spec, weight_path=str(weight),
            tokenizer_path=str(MODEL_ROOT / tokenizer) if tokenizer and not tokenizer.is_absolute() else spec.tokenizer_path)
        models.append(OpenCLIPSurrogate(spec, device=device, precision='fp32',
            allow_downloads=False, allow_device_fallback=False, require_true_local_tokens=False))
    validate_loaded_surrogates(models, selection)
    if hashlib.sha256(selection.source.read_bytes()).hexdigest() != selection.source_sha256:
        raise ValueError('训练代理配置在加载期间发生变化')
    return models


class PublicFullKeyObjective:
    """匹配源商品公开图文全键至联合目标，并辅以独立文本对齐与前景局部特征最优传输。"""
    def __init__(
        self,
        surrogate,
        target_feature,
        source_public_key,
        training_text,
        config,
        *,
        target_patch_tokens=None,
        source_fg_mask=None,
        target_fg_mask=None,
    ):
        for label, value in (('公开商品键', source_public_key), ('目标特点', training_text)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(label + '必须为非空公开正文')
            if any(marker in value.casefold() for marker in
                   ('actions and observations:', 'observed purchase:', 'action: think[')):
                raise ValueError(label + '不能包含私人购物轨迹')
        self.config = config
        encoder = ChunkedClipEncoder(SimpleNamespace(surrogate=surrogate,
            device=surrogate.device, snapshot=surrogate.snapshot()))
        own = encoder.encode_text(source_public_key)
        own_audit = dict(encoder.last_encoding_audit)
        text = encoder.encode_text(training_text)
        text_audit = dict(encoder.last_encoding_audit)
        self.own = torch.as_tensor(own, dtype=torch.float32, device=surrogate.device).reshape(1, -1)
        self.target_text = torch.as_tensor(text, dtype=torch.float32, device=surrogate.device).reshape(1, -1).detach()
        raw_target = getattr(target_feature, 'global_embedding', target_feature)
        self.target = F.normalize(raw_target.detach().float(), dim=-1)
        self.joint_target = F.normalize(self.target + self.target_text, dim=-1)

        comp_weight = float(getattr(config, 'compensation_weight', 0.35))
        if not math.isfinite(comp_weight) or not (0.0 <= comp_weight < 1.0):
            raise ValueError('逆向补偿目标权重 compensation_weight 必须位于 [0.0, 1.0)')
        self.compensation_weight = comp_weight
        comp_vec, comp_valid = compensated_targets(self.joint_target, self.own)
        self.compensated_target = comp_vec.detach()
        self.compensated_valid = bool(comp_valid.all())

        text_weight = float(getattr(config, 'text_weight', 0.0))
        if not math.isfinite(text_weight) or text_weight < 0.0:
            raise ValueError('独立文本对齐权重 text_weight 必须为非负有限实数')
        self.text_weight = text_weight
        self.text_key_mix = float(getattr(config, 'text_key_mix', 0.60))
        text_comp_vec, text_comp_valid = compensated_targets(self.target_text, self.own)
        self.text_compensated_target = text_comp_vec.detach()
        self.text_compensated_valid = bool(text_comp_valid.all())

        self.local_ot_weight = float(getattr(config, 'local_ot_weight', 0.0))
        self.local_ot_clusters = int(getattr(config, 'local_ot_clusters', 4))
        self.local_ot_iterations = int(getattr(config, 'local_ot_iterations', 50))
        self.local_ot_sinkhorn_regularization = float(getattr(config, 'local_ot_sinkhorn_regularization', 0.10))
        self.local_ot_fg_threshold = float(getattr(config, 'local_ot_fg_threshold', 0.50))
        self.source_fg_mask = source_fg_mask
        self.target_fg_centers = None
        tgt_patches = target_patch_tokens if target_patch_tokens is not None else getattr(target_feature, 'patch_tokens', None)
        if self.local_ot_weight > 0.0 and tgt_patches is not None:
            tgt_fg = extract_foreground_patch_tokens(
                tgt_patches.to(surrogate.device),
                target_fg_mask,
                threshold=self.local_ot_fg_threshold,
                min_patches=self.local_ot_clusters,
            )
            if tgt_fg is not None:
                self.target_fg_centers = token_kmeans(
                    tgt_fg, clusters=self.local_ot_clusters, iterations=self.local_ot_iterations
                ).detach()

        query_modes = ['multimodal_equal_weight']
        if self.text_weight > 0.0:
            query_modes.append('independent_target_text')
        self.audit = {'source_public_key_sha256': hashlib.sha256(source_public_key.encode('utf-8')).hexdigest(),
            'training_text_sha256': hashlib.sha256(training_text.encode('utf-8')).hexdigest(),
            'source_text_encoding': own_audit, 'target_text_encoding': text_audit,
            'query_modes': query_modes,
            'query_image_weight': 0.5, 'query_text_weight': 0.5,
            'memory_key_image_weight': 0.5,
            'compensated_target_enabled': bool(self.compensation_weight > 0.0 and self.compensated_valid),
            'compensated_valid': self.compensated_valid,
            'compensation_weight': self.compensation_weight,
            'text_weight': self.text_weight,
            'text_key_mix': self.text_key_mix,
            'text_compensated_valid': self.text_compensated_valid,
            'local_ot_weight': self.local_ot_weight,
            'local_ot_clusters': self.local_ot_clusters,
            'local_ot_fg_threshold': self.local_ot_fg_threshold,
            'local_ot_active': bool(self.local_ot_weight > 0.0 and self.target_fg_centers is not None),
            'victim_memory_used': False,
            'formal_queries_used': False, 'formal_ranking_used': False}

    def scores(self, features):
        img_norm = F.normalize(features.float(), dim=-1)
        key = F.normalize(img_norm + self.own, dim=-1)
        joint_scores = key @ self.joint_target.T
        if self.compensation_weight > 0.0 and self.compensated_valid:
            comp_scores = img_norm @ self.compensated_target.T
            return (1.0 - self.compensation_weight) * joint_scores + self.compensation_weight * comp_scores
        return joint_scores

    def text_scores(self, features):
        """适应 AmbiLAGE 的独立文本对齐打分（兼容非正半球安全退化与解析文本补偿）。"""
        img_norm = F.normalize(features.float(), dim=-1)
        key = F.normalize(img_norm + self.own, dim=-1)
        s_key = key @ self.target_text.T
        if self.text_compensated_valid and self.text_key_mix < 1.0:
            s_comp = img_norm @ self.text_compensated_target.T
            return self.text_key_mix * s_key + (1.0 - self.text_key_mix) * s_comp
        return s_key

    def local_ot_scores(self, patch_tokens, fg_mask=None):
        """仅在源前景掩膜与目标前景语义聚类中心间计算 Sinkhorn 最优传输相似度。"""
        if patch_tokens is None or self.target_fg_centers is None:
            return None
        mask = fg_mask if fg_mask is not None else self.source_fg_mask
        src_fg = extract_foreground_patch_tokens(
            patch_tokens,
            mask,
            threshold=self.local_ot_fg_threshold,
            min_patches=self.local_ot_clusters,
        )
        if src_fg is None:
            return None
        return local_ot_similarity_from_centers(
            src_fg,
            self.target_fg_centers.expand(src_fg.shape[0], -1, -1),
            sinkhorn_iterations=self.local_ot_iterations,
            sinkhorn_regularization=self.local_ot_sinkhorn_regularization,
            detach_transport_plan=True,
        ).reshape(-1, 1)

    def value(self, features, patch_tokens=None):
        scores = self.scores(features)
        joint_weight = float(getattr(self.config, 'joint_weight', 1.0))
        if not math.isfinite(joint_weight) or joint_weight <= 0:
            raise ValueError('图文联合目标权重必须为有限正数')
        temperature = float(self.config.temperature)
        total = joint_weight * (0.5 * scores.mean() + 0.5 * softmin(scores, temperature))
        if self.text_weight > 0.0:
            t_scores = self.text_scores(features)
            u_text = 0.5 * t_scores.mean() + 0.5 * softmin(t_scores, temperature)
            total = total + self.text_weight * u_text
        if self.local_ot_weight > 0.0 and patch_tokens is not None:
            ot_scores = self.local_ot_scores(patch_tokens)
            if ot_scores is not None:
                u_ot = 0.5 * ot_scores.mean() + 0.5 * softmin(ot_scores, temperature)
                total = total + self.local_ot_weight * u_ot
        return total


class SequentialEnsembleObjective:
    """Baseline-calibrated weakest-proxy weights; exact per-proxy image gradients.

    Recomputing each transformation/purification per proxy bounds graph memory.
    The same seed is reused within one step for scoring and differentiation.
    """
    def __init__(self, surrogates, target_tensor, source_public_key, training_text, config,
                 purifier=None, *, selection=None, source_fg_mask=None, target_fg_mask=None):
        self.surrogates = list(surrogates)
        if not self.surrogates:
            raise ValueError('训练代理不能为空')
        self.config, self.purifier = config, purifier
        self.selection = selection or selection_from_config(config)
        snapshots = validate_loaded_surrogates(self.surrogates, self.selection,
            allow_fixture=bool(getattr(config, 'smoke_test', False)))
        self.diffusion_weight = float(getattr(config, 'diffusion_weight', 0.08))
        if not torch.isfinite(torch.tensor(self.diffusion_weight)) or self.diffusion_weight <= 0:
            raise ValueError('完整新方法必须启用正权重扩散净化目标')
        self.ensemble_temperature = float(getattr(config, 'ensemble_temperature', 0.20))
        self.ensemble_uniform_mix = float(getattr(config, 'ensemble_uniform_mix', 0.45))
        if not math.isfinite(self.ensemble_temperature) or self.ensemble_temperature <= 0:
            raise ValueError('集成权重温度 ensemble_temperature 必须为有限正数')
        if not math.isfinite(self.ensemble_uniform_mix) or not (0.0 <= self.ensemble_uniform_mix <= 1.0):
            raise ValueError('均匀保底混合比例 ensemble_uniform_mix 必须位于 [0.0, 1.0]')
        self._baseline_values: torch.Tensor | None = None
        if purifier is None:
            from src.visual.diffusion_purification import LatentDiffusionPurifier, PurificationConfig
            from src.model_registry import resolve_model
            self.purifier = LatentDiffusionPurifier(PurificationConfig(
                model_path=(getattr(config, 'purifier_model_path', None)
                            or getattr(config, 'purifier_model_dir', None) or resolve_model('purifier')),
                working_size=int(getattr(config, 'purifier_working_size', 256)),
                reverse_steps=int(getattr(config, 'purifier_steps', 10)),
                gradient_checkpointing=bool(getattr(config, 'purifier_gradient_checkpointing', True))),
                device=str(self.surrogates[0].device))
        if getattr(getattr(self.purifier, 'config', None), 'reverse_steps', 10) != 10:
            raise ValueError('完整新方法固定十步扩散净化')
        self.objectives = []
        for surrogate in self.surrogates:
            with torch.no_grad():
                target_out = surrogate.encode_image(target_tensor.to(surrogate.device))
            target = target_out.global_embedding
            target_patches = getattr(target_out, 'patch_tokens', None)
            self.objectives.append(PublicFullKeyObjective(
                surrogate, target, source_public_key, training_text, config,
                target_patch_tokens=target_patches,
                source_fg_mask=source_fg_mask,
                target_fg_mask=target_fg_mask,
            ))
        self.audit = {'kind': TRAINING_OBJECTIVE_KIND,
            'surrogates': snapshots, 'surrogate_selection': self.selection.audit(),
            'public_objectives': [o.audit for o in self.objectives],
            'held_out_models': list(HELD_OUT_MODELS),
            'gradient_execution': 'one_proxy_graph_at_a_time_recompute_purifier',
            'weak_proxy_weighting': 'baseline_calibrated_relative_gain_softmin_with_uniform_floor',
            'ensemble_temperature': self.ensemble_temperature,
            'ensemble_uniform_mix': self.ensemble_uniform_mix,
            'eot_enabled': bool(getattr(config, 'eot_enabled', True)),
            'eot_paths': ['identity', 'pillow_blur_bpda', 'jpeg_exact_forward_ste', 'resize'],
            'diffusion_weight': self.diffusion_weight, 'diffusion_gradient': 'full',
            'purifier': self.purifier.snapshot(),
            'victim_memory_used': False, 'formal_queries_used': False}

    def _seed(self, step):
        return (int(getattr(self.config, 'seed', 42)) + max(0, int(step))) % (2**63)

    def _one_value(self, index, image, step, gradient):
        surrogate, objective = self.surrogates[index], self.objectives[index]
        image = image.to(surrogate.device)
        views = transformed_batch(image, step=max(1, int(step))) if getattr(self.config, 'eot_enabled', True) else image
        views_out = surrogate.encode_image(views)
        feature = views_out.global_embedding
        views_patches = getattr(views_out, 'patch_tokens', None)
        score = objective.scores(feature)
        eot = objective.value(feature, patch_tokens=views_patches)
        purified = self.purifier(image, seed=self._seed(step), gradient=gradient)
        purified_out = surrogate.encode_image(purified)
        purified_feature = purified_out.global_embedding
        purified_patches = getattr(purified_out, 'patch_tokens', None)
        pvalue = objective.value(purified_feature, patch_tokens=purified_patches)
        utility = (eot + self.diffusion_weight * pvalue) / (1.0 + self.diffusion_weight)
        return utility, score.mean(0), {'eot': float(eot.detach()), 'purified': float(pvalue.detach())}

    @torch.no_grad()
    def _values(self, image, step):
        values, scores, rows = [], [], []
        for i in range(len(self.surrogates)):
            value, score, row = self._one_value(i, image, step, 'none')
            values.append(value.to(image.device))
            scores.append(score.to(image.device))
            rows.append(row)
        values = torch.stack(values)
        if self._baseline_values is None or self._baseline_values.shape != values.shape:
            self._baseline_values = values.detach().clone()
        gains = (values - self._baseline_values.to(values.device)).detach()
        adaptive = torch.softmax(-gains / max(float(self.ensemble_temperature), 1e-4), dim=0)
        uni_mix = min(max(float(self.ensemble_uniform_mix), 0.0), 1.0)
        uniform = torch.full_like(adaptive, 1.0 / len(self.surrogates))
        weights = ((1.0 - uni_mix) * adaptive + uni_mix * uniform).detach()
        return values, torch.stack(scores), weights, rows, gains

    @torch.no_grad()
    def evaluate(self, image, step=0):
        values, scores, weights, _, _ = self._values(image, step)
        return (weights * values).sum(), scores

    def image_gradient(self, image, step):
        values, scores, weights, rows, gains = self._values(image.detach(), step)
        gradient = torch.zeros_like(image)
        for i in range(len(self.surrogates)):
            leaf = image.detach().clone().requires_grad_(True)
            value, _, _ = self._one_value(i, leaf, step, 'full')
            grad = torch.autograd.grad(value, leaf, retain_graph=False)[0]
            if not torch.isfinite(grad).all():
                raise FloatingPointError('代理目标梯度含非有限值')
            gradient.add_(grad.detach(), alpha=float(weights[i]))
            del leaf, value, grad
        utility = (weights * values).sum().detach()
        return utility, scores.detach(), gradient, {'model_weights': weights.cpu().tolist(),
            'model_utilities': values.cpu().tolist(), 'model_relative_gains': gains.cpu().tolist(),
            'paths': rows, 'purifier_gradient': 'full'}
