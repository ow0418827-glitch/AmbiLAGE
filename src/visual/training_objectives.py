"""Public-only joint image/text training with selectable sequential surrogates.

No memory pool, formal queries or evaluation rank is accepted by this module.
All surrogate image graphs (including the DDIM path) are released one at a time.
"""
from __future__ import annotations

from dataclasses import replace
import gc
import hashlib
import math
from pathlib import Path
import re
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.paths import MODEL_ROOT, PROJECT
from src.visual.encoders.backbone import ImageFeatures, OpenCLIPSurrogate
from src.visual.encoders.joint import ChunkedClipEncoder
from src.visual.hardware import get_device_vram_gb
from src.visual.robust_objectives import compensated_targets, softmin, transformed_batch
from src.visual._lucid.objective import local_ot_similarity_from_centers, token_kmeans
from src.visual.training_surrogates import (
    HELD_OUT_MODELS,
    HELD_OUT_PURIFIERS,
    is_held_out_purifier_reference,
    select_training_surrogates,
    selection_from_config,
    spec_identity,
    validate_loaded_surrogates,
    validate_training_specs,
)

LEGACY_V3_OBJECTIVE_KIND = 'sequential_public_joint_text_fg_ot_v3'
TRAINING_OBJECTIVE_KIND = 'sequential_public_joint_text_fg_ot_psi_v4'
SUPPORTED_TRAINING_OBJECTIVE_KINDS = (LEGACY_V3_OBJECTIVE_KIND, TRAINING_OBJECTIVE_KIND)


def decompose_training_text_phrases(training_text: str, max_phrases: int = 6) -> list[str]:
    """将冻结公开特点正文确定性地拆解为多粒度语义子句簇（完整句 + 品类意图 + 细粒度属性）。"""
    cleaned = ' '.join(training_text.strip().split())
    if not cleaned or max_phrases <= 1:
        return [cleaned] if cleaned else ['product']

    phrases: list[str] = [cleaned]
    seen = {cleaned.casefold()}

    def _add(candidate: str) -> None:
        if len(phrases) >= max_phrases:
            return
        norm = ' '.join(candidate.strip().strip('.,;!?，。；！？').split())
        if not norm:
            return
        has_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in norm)
        if not has_cjk and len(norm.split()) < 2:
            return
        if has_cjk and len(norm) < 3:
            return
        key = norm.casefold()
        if key not in seen:
            seen.add(key)
            phrases.append(norm)

    pattern = re.compile(
        r'^(?P<prefix>i want to buy|i would like to purchase|i am looking for|looking to buy)\s+'
        r'(?P<cat>.+?)\s+(?:with|featuring)\s+(?P<feats>.+?)[.!?]*$',
        flags=re.IGNORECASE,
    )
    match = pattern.match(cleaned)
    if match:
        prefix = match.group('prefix').strip()
        cat = match.group('cat').strip()
        feats_raw = match.group('feats').strip()
        _add(f'{prefix} {cat}')
        sub_feats = [
            f.strip()
            for f in re.split(r'\s+and\s+|[;,]+', feats_raw, flags=re.IGNORECASE)
            if f.strip()
        ]
        for feat in sub_feats:
            _add(f'{cat} with {feat}')
        for feat in sub_feats:
            _add(feat)
        return phrases[:max_phrases]

    parts = re.split(r'[.;,，。；!?\n]+|\band\b|\bwith\b|\bfeaturing\b', cleaned, flags=re.IGNORECASE)
    for part in parts:
        _add(part)
        if len(phrases) >= max_phrases:
            break
    return phrases[:max_phrases]


def worst_case_spherical_queries(
    anchor_queries: torch.Tensor,
    target_direction: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """单位超球面帽 {q in S^{D-1} : ||q - q0||_2 <= r} 上最小化 <target_direction, q> 的闭式最坏查询。

    内层对图像与键方向完全截断梯度 (detached)，外层仅优化图像特征，零语言模型调用。
    """
    r = float(radius)
    if r <= 0.0 or anchor_queries.ndim != 2 or anchor_queries.shape[1] <= 1:
        return F.normalize(anchor_queries.detach(), dim=-1)
    r = min(r, 2.0)
    q0 = F.normalize(anchor_queries.detach().to(dtype=torch.float64), dim=-1)
    v = F.normalize(target_direction.detach().to(device=q0.device, dtype=torch.float64), dim=-1)
    if v.shape[0] == 1 and q0.shape[0] > 1:
        v = v.expand(q0.shape[0], -1)
    cos_max = max(-1.0, min(1.0, 1.0 - 0.5 * (r * r)))
    sin_max = math.sqrt(max(0.0, 1.0 - cos_max * cos_max))
    proj = (v * q0).sum(dim=-1, keepdim=True)
    tangent = v - proj * q0
    t_norm = tangent.norm(dim=-1, keepdim=True)
    unit_tangent = tangent / t_norm.clamp_min(1e-15)
    boundary = cos_max * q0 - sin_max * unit_tangent
    anti_v = -v
    use_anti = ((anti_v * q0).sum(dim=-1, keepdim=True) >= cos_max)
    chosen = torch.where(use_anti, anti_v, boundary)
    chosen = torch.where(t_norm > 1e-12, chosen, q0)
    return F.normalize(chosen, dim=-1).to(dtype=anchor_queries.dtype, device=anchor_queries.device).detach()


def pcgrad_merge_gradients(
    grads: list[torch.Tensor],
    weights: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """范数均衡且保持物理量纲的多代理正交梯度手术 (Norm-Balanced Scale-Preserving PCGrad)。

    1. 消除不同分块架构（如 ViT-B-16 与 ViT-B-32，或 ViT-L-14 与 SigLIP）之间的模长失衡；
    2. 对任意负内积冲突梯度执行正交投影（按集成权重降序优先保护薄弱代理）；
    3. 按原加权平均范数恢复合成梯度的物理尺度，确保与空间平滑正则项量纲严格对齐。
    """
    if not grads:
        raise ValueError('待融合代理梯度列表不能为空')
    if len(grads) == 1:
        return grads[0] * float(weights[0])

    flat_grads = [g.reshape(-1).to(dtype=torch.float64) for g in grads]
    w64 = weights.detach().to(device=flat_grads[0].device, dtype=torch.float64)
    w64 = w64 / w64.sum().clamp_min(eps)

    norms = torch.stack([fg.norm() for fg in flat_grads])
    target_scale = float((w64 * norms).sum().item())
    if target_scale <= eps:
        return torch.zeros_like(grads[0])

    unit_grads = [
        fg / norms[i].clamp_min(eps) if float(norms[i]) > eps else torch.zeros_like(fg)
        for i, fg in enumerate(flat_grads)
    ]
    order = torch.argsort(w64, descending=True).tolist()
    proj_grads = [ug.clone() for ug in unit_grads]

    for i in range(len(proj_grads)):
        if float(norms[i]) <= eps:
            continue
        for j in order:
            if i == j or float(norms[j]) <= eps:
                continue
            ref = unit_grads[j]
            dot = torch.dot(proj_grads[i], ref)
            if float(dot) < 0.0:
                denom = torch.dot(ref, ref).clamp_min(eps)
                proj_grads[i] = proj_grads[i] - (dot / denom) * ref

    combined = torch.zeros_like(proj_grads[0])
    for i, pg in enumerate(proj_grads):
        combined.add_(pg, alpha=float(w64[i]))
    merged = (combined * target_scale).to(dtype=grads[0].dtype).reshape_as(grads[0])
    return merged


def psi_cross_modal_weights(
    config,
    step: int | None = None,
    *,
    progressive: bool = False,
) -> tuple[float, float, float]:
    """计算 PSI 三段式渐进跨模态权重或恒定评估标尺权重。

    各目标项独立按阶段调制，互不耦合分母，且终端总权重与基础配置总和同量纲（约 1.45），
    严格保持与空间全变分及光晕正则项之间的物理梯度比例。当 text_weight == 0 时不衰减联合项。
    """
    j_base = float(getattr(config, 'joint_weight', 1.0))
    t_base = float(getattr(config, 'text_weight', 0.0))
    ot_base = float(getattr(config, 'local_ot_weight', 0.0))
    if not math.isfinite(j_base) or j_base <= 0.0:
        raise ValueError('图文联合目标权重必须为有限正数')

    if not bool(getattr(config, 'psi_progressive_schedule', False)) or not progressive or step is None or int(step) <= 0:
        return j_base, t_base, ot_base

    total_steps = max(int(getattr(config, 'steps', 250)), 2)
    p1 = max(1, int(round(float(getattr(config, 'psi_phase1_ratio', 0.24)) * total_steps)))
    p2 = max(p1 + 1, int(round(float(getattr(config, 'psi_phase2_ratio', 0.72)) * total_steps)))
    cur = int(step)

    j_start = j_base
    j_end = j_base * float(getattr(config, 'psi_joint_end_scale', 0.45)) if t_base > 0.0 else j_base
    t_start = 0.75 * t_base
    t_peak = t_base * float(getattr(config, 'psi_text_peak_multiplier', 2.40))
    ot_start = 0.60 * ot_base
    ot_peak = ot_base * float(getattr(config, 'psi_ot_peak_multiplier', 1.60))

    if cur <= p1:
        return j_start, t_start, ot_start
    if cur <= p2:
        prog = (cur - p1) / max(1, p2 - p1)
        alpha = 0.5 * (1.0 - math.cos(math.pi * min(max(prog, 0.0), 1.0)))
        wj = j_start + alpha * (j_end - j_start)
        wt = t_start + alpha * (t_peak - t_start)
        wot = ot_start + alpha * (ot_peak - ot_start)
        return wj, wt, wot
    return j_end, t_peak, ot_peak


def extract_foreground_patch_tokens(
    patch_tokens: torch.Tensor | None,
    fg_mask: torch.Tensor | object | None,
    *,
    threshold: float = 0.50,
    min_patches: int = 1,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
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
        norm_tok = F.normalize(spatial.float(), dim=-1)
        if return_weights:
            uni = torch.full((norm_tok.shape[1],), 1.0 / norm_tok.shape[1], device=norm_tok.device, dtype=torch.float32)
            return norm_tok, uni
        return norm_tok
    mask_t = torch.as_tensor(fg_mask, dtype=torch.float32, device=patch_tokens.device)
    if mask_t.ndim == 2:
        mask_t = mask_t[None, None]
    elif mask_t.ndim == 3:
        mask_t = mask_t[:, None]
    if float(mask_t.sum().detach()) <= 0.0:
        norm_tok = F.normalize(spatial.float(), dim=-1)
        if return_weights:
            uni = torch.full((norm_tok.shape[1],), 1.0 / norm_tok.shape[1], device=norm_tok.device, dtype=torch.float32)
            return norm_tok, uni
        return norm_tok

    pooled = F.adaptive_avg_pool2d(mask_t, (side, side)).reshape(-1)
    sel = torch.nonzero(pooled >= float(threshold), as_tuple=False).reshape(-1)
    k_need = min(max(1, int(min_patches)), side * side)
    if int(sel.numel()) < k_need:
        sel = torch.sort(torch.topk(pooled, k=k_need).indices).values
    norm_tok = F.normalize(spatial[:, sel, :].float(), dim=-1)
    if return_weights:
        raw_w = pooled[sel].clamp_min(1e-4)
        weights = (raw_w / raw_w.sum().clamp_min(1e-8)).detach()
        return norm_tok, weights
    return norm_tok


def _encode_surrogate_image_microbatched(
    surrogate,
    images: torch.Tensor,
    *,
    microbatch_size: int | None = None,
) -> ImageFeatures:
    """对大分辨率/超大参数代理模型分微批提取特征，数学前反向等价且显著压低 L4 激活显存峰值。"""
    total = int(images.shape[0])
    if microbatch_size is None or microbatch_size <= 0 or total <= microbatch_size:
        return surrogate.encode_image(images)
    globals_list, patches_list = [], []
    has_patches = True
    for start in range(0, total, microbatch_size):
        out = surrogate.encode_image(images[start : start + microbatch_size])
        globals_list.append(out.global_embedding)
        pt = getattr(out, 'patch_tokens', None)
        if pt is None:
            has_patches = False
        else:
            patches_list.append(pt)
    global_cat = torch.cat(globals_list, dim=0)
    patch_cat = torch.cat(patches_list, dim=0) if (has_patches and patches_list) else None
    return ImageFeatures(global_embedding=global_cat, patch_tokens=patch_cat)


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
    """匹配源商品公开图文全键至联合目标，并融合三路多粒度文本对齐、球面查询增强与文本加权前景局部 OT。"""
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
        source_clean_feature=None,
        public_retrieval_state=None,
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
        self.own = torch.as_tensor(own, dtype=torch.float32, device=surrogate.device).reshape(1, -1).detach()
        self.target_text = torch.as_tensor(text, dtype=torch.float32, device=surrogate.device).reshape(1, -1).detach()
        raw_target = getattr(target_feature, 'global_embedding', target_feature)
        self.target = F.normalize(raw_target.detach().float(), dim=-1)
        self.joint_target = F.normalize(self.target + self.target_text, dim=-1)

        if source_clean_feature is not None:
            raw_src = getattr(source_clean_feature, 'global_embedding', source_clean_feature)
            src_norm = F.normalize(raw_src.detach().float().to(surrogate.device), dim=-1)
            self.source_clean_key = F.normalize(src_norm + self.own, dim=-1).detach()
        else:
            self.source_clean_key = None

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
        self.text_pure_image_mix = float(getattr(config, 'text_pure_image_mix', 0.0))
        text_comp_vec, text_comp_valid = compensated_targets(self.target_text, self.own)
        self.text_compensated_target = text_comp_vec.detach()
        self.text_compensated_valid = bool(text_comp_valid.all())

        self.phrase_decomposition = bool(getattr(config, 'text_phrase_decomposition', False)) and self.text_weight > 0.0
        max_phrases = int(getattr(config, 'max_text_phrases', 6))
        if self.phrase_decomposition:
            self.phrase_texts = decompose_training_text_phrases(training_text, max_phrases=max_phrases)
        else:
            self.phrase_texts = [training_text]
        phrase_vecs = [self.target_text]
        for extra_phrase in self.phrase_texts[1:]:
            p_vec = encoder.encode_text(extra_phrase)
            phrase_vecs.append(
                F.normalize(torch.as_tensor(p_vec, dtype=torch.float32, device=surrogate.device).reshape(1, -1), dim=-1).detach()
            )
        self.phrase_features = F.normalize(torch.cat(phrase_vecs, dim=0), dim=-1).detach()
        p_comp, p_valid = compensated_targets(self.phrase_features, self.own)
        self.phrase_compensated_features = p_comp.detach()
        self.phrase_compensated_valid = p_valid.detach()

        self.spherical_augmentation = bool(getattr(config, 'text_spherical_augmentation', False)) and self.text_weight > 0.0
        self.spherical_radius = float(getattr(config, 'text_spherical_radius', 0.05))
        self.spherical_mix = float(getattr(config, 'text_spherical_mix', 0.50))
        self.source_repulsion_weight = float(getattr(config, 'source_repulsion_weight', 0.0))
        self.source_repulsion_margin = float(getattr(config, 'source_repulsion_margin', 0.08))

        self.local_ot_weight = float(getattr(config, 'local_ot_weight', 0.0))
        self.local_ot_clusters = int(getattr(config, 'local_ot_clusters', 4))
        self.local_ot_iterations = int(getattr(config, 'local_ot_iterations', 50))
        self.local_ot_sinkhorn_regularization = float(getattr(config, 'local_ot_sinkhorn_regularization', 0.10))
        self.local_ot_fg_threshold = float(getattr(config, 'local_ot_fg_threshold', 0.50))
        self.local_ot_text_weighted = bool(getattr(config, 'local_ot_text_weighted', False))
        self.local_ot_text_temperature = float(getattr(config, 'local_ot_text_temperature', 0.10))
        self.source_fg_mask = source_fg_mask
        self.target_fg_centers = None
        self.target_fg_center_weights = None
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
                if self.local_ot_text_weighted:
                    center_text_logits = (
                        (self.target_fg_centers[0] @ self.target_text.T).squeeze(-1)
                        / max(self.local_ot_text_temperature, 1e-4)
                    )
                    self.target_fg_center_weights = torch.softmax(center_text_logits, dim=0).detach()

        self.public_retrieval_state = public_retrieval_state
        self.public_retrieval_weight = (
            float(getattr(config, 'public_retrieval_weight', 0.0))
            if public_retrieval_state is not None
            else 0.0
        )

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
            'text_pure_image_mix': self.text_pure_image_mix,
            'text_compensated_valid': self.text_compensated_valid,
            'phrase_decomposition_enabled': self.phrase_decomposition,
            'phrase_count': len(self.phrase_texts),
            'spherical_query_augmentation': self.spherical_augmentation,
            'local_ot_weight': self.local_ot_weight,
            'local_ot_clusters': self.local_ot_clusters,
            'local_ot_fg_threshold': self.local_ot_fg_threshold,
            'local_ot_text_weighted': bool(self.local_ot_text_weighted and self.target_fg_center_weights is not None),
            'local_ot_active': bool(self.local_ot_weight > 0.0 and self.target_fg_centers is not None),
            'public_bundle_retrieval_active': bool(self.public_retrieval_weight > 0.0),
            'victim_memory_used': False,
            'formal_queries_used': False, 'formal_ranking_used': False}

    def scores(self, features):
        img_norm = F.normalize(features.float(), dim=-1)
        own = self.own.to(img_norm.device)
        joint_tgt = self.joint_target.to(img_norm.device)
        key = F.normalize(img_norm + own, dim=-1)
        joint_scores = key @ joint_tgt.T
        if self.compensation_weight > 0.0 and self.compensated_valid:
            comp_scores = img_norm @ self.compensated_target.to(img_norm.device).T
            return (1.0 - self.compensation_weight) * joint_scores + self.compensation_weight * comp_scores
        return joint_scores

    def text_scores(self, features):
        """适应 AmbiLAGE 的主句子独立文本对齐打分（兼容非正半球安全退化与解析文本补偿）。"""
        img_norm = F.normalize(features.float(), dim=-1)
        own = self.own.to(img_norm.device)
        tgt_txt = self.target_text.to(img_norm.device)
        key = F.normalize(img_norm + own, dim=-1)
        s_key = key @ tgt_txt.T
        if self.text_compensated_valid and self.text_key_mix < 1.0:
            s_comp = img_norm @ self.text_compensated_target.to(img_norm.device).T
            return self.text_key_mix * s_key + (1.0 - self.text_key_mix) * s_comp
        return s_key

    def _score_query_bank(
        self,
        img_norm: torch.Tensor,
        key: torch.Tensor,
        queries: torch.Tensor,
        comp_queries: torch.Tensor | None = None,
        comp_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """三路协同文本打分：同时覆盖 joint 图文记忆键、源标题解析补偿与 image 纯图记忆键。"""
        q = queries.to(img_norm.device)
        alpha_img = min(max(self.text_pure_image_mix, 0.0), 0.80)
        rem = 1.0 - alpha_img
        s_key = key @ q.T
        s_img = img_norm @ q.T
        if comp_queries is not None and comp_valid_mask is not None and self.text_key_mix < 1.0:
            c_q = comp_queries.to(img_norm.device)
            valid = comp_valid_mask.to(device=img_norm.device, dtype=img_norm.dtype).reshape(1, -1)
            s_comp = img_norm @ c_q.T
            mixed_joint = self.text_key_mix * s_key + (1.0 - self.text_key_mix) * (
                valid * s_comp + (1.0 - valid) * s_key
            )
        else:
            mixed_joint = s_key
        return rem * mixed_joint + alpha_img * s_img

    def enhanced_text_utility(self, features: torch.Tensor, temperature: float, *, progressive: bool = False) -> torch.Tensor:
        """多粒度子句簇 + 闭式球面最坏查询增强 + 源干净键对比排斥效用。"""
        has_v4_text = (
            self.phrase_decomposition
            or self.text_pure_image_mix > 0.0
            or self.spherical_augmentation
            or (self.source_repulsion_weight > 0.0 and self.source_clean_key is not None)
        )
        if not has_v4_text:
            t_scores = self.text_scores(features)
            return 0.5 * t_scores.mean() + 0.5 * softmin(t_scores, temperature)

        img_norm = F.normalize(features.float(), dim=-1)
        own = self.own.to(img_norm.device)
        key = F.normalize(img_norm + own, dim=-1)
        phrases = self.phrase_features.to(img_norm.device)
        p_comp = self.phrase_compensated_features.to(img_norm.device)
        p_valid = self.phrase_compensated_valid.to(img_norm.device)

        clean_mat = self._score_query_bank(img_norm, key, phrases, p_comp, p_valid)
        if progressive and self.spherical_augmentation and self.spherical_radius > 0.0 and self.spherical_mix > 0.0:
            alpha_img = min(max(self.text_pure_image_mix, 0.0), 0.80)
            v_dir = (1.0 - alpha_img) * key.detach().mean(dim=0, keepdim=True) + alpha_img * img_norm.detach().mean(dim=0, keepdim=True)
            adv_phrases = worst_case_spherical_queries(phrases, v_dir, self.spherical_radius)
            adv_comp, adv_valid = compensated_targets(adv_phrases, own)
            adv_mat = self._score_query_bank(img_norm, key, adv_phrases, adv_comp.detach(), adv_valid.detach())
            score_mat = (1.0 - self.spherical_mix) * clean_mat + self.spherical_mix * adv_mat
        else:
            score_mat = clean_mat

        u_phrases = 0.5 * score_mat.mean() + 0.5 * softmin(score_mat.flatten(), max(temperature, 0.08))

        if self.source_repulsion_weight > 0.0 and self.source_clean_key is not None:
            clean_key = self.source_clean_key.to(img_norm.device)
            s_pos = key @ phrases.T
            s_src = (clean_key @ phrases.T).detach()
            tau_nce = 0.07
            logits_gap = (s_src - (s_pos - self.source_repulsion_margin)) / tau_nce
            u_repel = (-tau_nce * F.softplus(logits_gap)).mean()
            u_phrases = u_phrases + self.source_repulsion_weight * u_repel
        return u_phrases

    def public_bundle_retrieval_utility(
        self,
        features: torch.Tensor,
        query_indices: list[int] | None,
        *,
        progressive: bool = False,
    ) -> torch.Tensor | None:
        """可选同源公开代理库 (90 候选 + 100 查询) 双键兼容困难负例对比检索效用。"""
        if self.public_retrieval_state is None or self.public_retrieval_weight <= 0.0:
            return None
        from src.visual._visual_inception.negative_sampling import select_hard_negatives

        img_norm = F.normalize(features.float(), dim=-1)
        own = self.own.to(img_norm.device)
        key = F.normalize(img_norm + own, dim=-1)
        q_all = self.public_retrieval_state['query_features'].to(img_norm.device)
        negatives = self.public_retrieval_state['negative_features'].to(img_norm.device)
        policy = self.public_retrieval_state['negative_policy']

        q_batch = q_all if query_indices is None else q_all[query_indices]
        q_batch = F.normalize(q_batch.detach(), dim=-1)
        indices, _ = select_hard_negatives(q_batch, negatives.detach(), policy, self.public_retrieval_state['neg_config'])
        selected_neg = negatives[indices.to(device=img_norm.device, dtype=torch.long)]  # [Q, K, D]

        tau = max(float(getattr(self.config, 'public_retrieval_temperature', 0.07)), 1e-4)
        margin = float(getattr(self.config, 'public_contrastive_margin', 0.30)) if progressive else 0.0
        neg_logits = torch.einsum('qd,qkd->qk', q_batch, selected_neg) / tau  # [Q, K]

        # 同时激励纯图片键 (image) 与图文融合键 (joint) 在公开负例竞争中胜出
        pos_img = ((q_batch @ img_norm.T) - margin) / tau   # [Q, B]
        pos_key = ((q_batch @ key.T) - margin) / tau        # [Q, B]

        losses_img = []
        losses_key = []
        for b_idx in range(img_norm.shape[0]):
            p_i = pos_img[:, b_idx : b_idx + 1]
            p_k = pos_key[:, b_idx : b_idx + 1]
            losses_img.append(torch.logsumexp(torch.cat([p_i, neg_logits], dim=1), dim=1) - p_i[:, 0])
            losses_key.append(torch.logsumexp(torch.cat([p_k, neg_logits], dim=1), dim=1) - p_k[:, 0])
        loss_total = 0.5 * torch.stack(losses_img).mean() + 0.5 * torch.stack(losses_key).mean()
        # 转换为与余弦效用量纲匹配的有界最大化效用 (-tau * loss)
        return -tau * loss_total

    def local_ot_scores(self, patch_tokens, fg_mask=None):
        """仅在源前景掩膜与目标前景语义聚类中心间计算文本相关性加权的 Sinkhorn 最优传输相似度。"""
        if patch_tokens is None or self.target_fg_centers is None:
            return None
        mask = fg_mask if fg_mask is not None else self.source_fg_mask
        extracted = extract_foreground_patch_tokens(
            patch_tokens,
            mask,
            threshold=self.local_ot_fg_threshold,
            min_patches=self.local_ot_clusters,
            return_weights=True,
        )
        if extracted is None:
            return None
        src_fg, src_weights = extracted
        centers = self.target_fg_centers.to(src_fg.device).expand(src_fg.shape[0], -1, -1)
        col_w = (
            self.target_fg_center_weights.to(src_fg.device)
            if (self.local_ot_text_weighted and self.target_fg_center_weights is not None)
            else None
        )
        return local_ot_similarity_from_centers(
            src_fg,
            centers,
            sinkhorn_iterations=self.local_ot_iterations,
            sinkhorn_regularization=self.local_ot_sinkhorn_regularization,
            detach_transport_plan=True,
            row_weights=src_weights,
            col_weights=col_w,
        ).reshape(-1, 1)

    def value(
        self,
        features,
        patch_tokens=None,
        *,
        step: int | None = None,
        progressive: bool = False,
        public_query_indices: list[int] | None = None,
    ):
        scores = self.scores(features)
        w_joint, w_text, w_ot = psi_cross_modal_weights(self.config, step=step, progressive=progressive)
        temperature = float(self.config.temperature)
        total = w_joint * (0.5 * scores.mean() + 0.5 * softmin(scores, temperature))
        if w_text > 0.0:
            u_text = self.enhanced_text_utility(features, temperature, progressive=progressive)
            total = total + w_text * u_text
        if w_ot > 0.0 and patch_tokens is not None:
            ot_scores = self.local_ot_scores(patch_tokens)
            if ot_scores is not None:
                u_ot = 0.5 * ot_scores.mean() + 0.5 * softmin(ot_scores, temperature)
                total = total + w_ot * u_ot
        if self.public_retrieval_weight > 0.0:
            u_pub = self.public_bundle_retrieval_utility(features, public_query_indices, progressive=progressive)
            if u_pub is not None:
                total = total + self.public_retrieval_weight * u_pub
        return total


class SequentialEnsembleObjective:
    """Baseline-calibrated weakest-proxy weights; PCGrad orthogonalized image gradients.

    Recomputing each transformation/purification per proxy bounds graph memory, and
    adaptive L4 24GB VRAM management offloads idle text towers and streams ultra-giant
    surrogate backbones cleanly without altering mathematical gradients.
    """
    def __init__(
        self,
        surrogates,
        target_tensor,
        source_public_key,
        training_text,
        config,
        purifier=None,
        *,
        selection=None,
        source_fg_mask=None,
        target_fg_mask=None,
        source_tensor=None,
        training_bundle=None,
        candidate_cache_dir=None,
    ):
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
        self.pcgrad_enabled = bool(getattr(config, 'pcgrad_enabled', False))
        self._baseline_values: torch.Tensor | None = None

        explicit_purifier_path = (
            getattr(config, 'purifier_model_path', None)
            or getattr(config, 'purifier_model_dir', None)
        )
        if is_held_out_purifier_reference(explicit_purifier_path):
            raise ValueError('留出评估净化器 SD 2.1-base 禁止用于训练阶段')
        if purifier is None:
            from src.visual.diffusion_purification import LatentDiffusionPurifier, PurificationConfig
            from src.model_registry import resolve_model
            resolved_purifier = explicit_purifier_path or resolve_model('purifier')
            if is_held_out_purifier_reference(resolved_purifier):
                raise ValueError('留出评估净化器 SD 2.1-base 禁止用于训练阶段')
            self.purifier = LatentDiffusionPurifier(PurificationConfig(
                model_path=resolved_purifier,
                working_size=int(getattr(config, 'purifier_working_size', 256)),
                reverse_steps=int(getattr(config, 'purifier_steps', 10)),
                gradient_checkpointing=bool(getattr(config, 'purifier_gradient_checkpointing', True))),
                device=str(self.surrogates[0].device))
        if getattr(getattr(self.purifier, 'config', None), 'reverse_steps', 10) != 10:
            raise ValueError('完整新方法固定十步扩散净化')
        if is_held_out_purifier_reference(getattr(getattr(self.purifier, 'config', None), 'model_path', None)):
            raise ValueError('留出评估净化器 SD 2.1-base 禁止用于训练阶段')

        self._public_sampling = None
        public_states = [None] * len(self.surrogates)
        cache_records = []
        if training_bundle is not None and float(getattr(config, 'public_retrieval_weight', 0.0)) > 0.0:
            from src.visual._visual_inception.retrieval_cache import encode_candidate_cache
            from src.visual._visual_inception.negative_sampling import build_negative_policy
            from src.visual._visual_inception.objective import sample_query_indices

            fixture_flag = bool(getattr(config, 'smoke_test', False))
            cache_root = Path(candidate_cache_dir) if candidate_cache_dir else (PROJECT / 'output/datasets/public/ambilage_candidate_cache')
            neg_cfg = SimpleNamespace(
                hard_negative_count=32,
                hard_negative_same_category_count=8,
                hard_negative_proxy_memory_count=8,
                hard_negative_popularity_count=8,
            )
            batch_sz = int(getattr(config, 'public_query_batch_size', 16))
            steps_total = int(getattr(config, 'steps', 250))
            seed_val = int(getattr(config, 'seed', 42))
            self._public_sampling = {
                'pool_size': len(training_bundle['queries']),
                'batch_size': batch_sz,
                'step_indices': [
                    sample_query_indices(len(training_bundle['queries']), batch_sz, seed_val, s)
                    for s in range(1, steps_total + 1)
                ],
            }
            for idx, surrogate in enumerate(self.surrogates):
                enc = ChunkedClipEncoder(SimpleNamespace(
                    surrogate=surrogate, device=surrogate.device, snapshot=surrogate.snapshot()
                ))
                q_rows = [
                    torch.as_tensor(enc.encode_text(q_txt), device=surrogate.device, dtype=torch.float32).reshape(1, -1)
                    for q_txt in training_bundle['queries']
                ]
                q_mat = F.normalize(torch.cat(q_rows, dim=0), dim=-1).detach()
                cand_vecs, rec = encode_candidate_cache(
                    surrogate,
                    training_bundle['candidates'],
                    cache_root,
                    bundle_sha256=training_bundle['bundle_sha256'],
                    fixture=fixture_flag,
                )
                cache_records.append(rec)
                neg_pol = build_negative_policy(
                    training_bundle['candidate_metadata'],
                    cand_vecs,
                    category=training_bundle['audit']['category'],
                )
                public_states[idx] = {
                    'query_features': q_mat,
                    'negative_features': cand_vecs.detach(),
                    'negative_policy': neg_pol,
                    'neg_config': neg_cfg,
                }

        self.objectives = []
        for idx, surrogate in enumerate(self.surrogates):
            with torch.no_grad():
                target_out = surrogate.encode_image(target_tensor.to(surrogate.device))
                src_clean_out = (
                    surrogate.encode_image(source_tensor.to(surrogate.device))
                    if source_tensor is not None
                    else None
                )
            target = target_out.global_embedding
            target_patches = getattr(target_out, 'patch_tokens', None)
            self.objectives.append(PublicFullKeyObjective(
                surrogate, target, source_public_key, training_text, config,
                target_patch_tokens=target_patches,
                source_fg_mask=source_fg_mask,
                target_fg_mask=target_fg_mask,
                source_clean_feature=src_clean_out,
                public_retrieval_state=public_states[idx],
            ))

        # L4 24GB 显存自适应治理：预编码完成后卸载闲置文本塔，并对大视觉塔启用梯度检查点与按需流式驻留
        primary_dev = torch.device(self.surrogates[0].device)
        vram_gb = get_device_vram_gb(primary_dev) if primary_dev.type == 'cuda' else 0.0
        low_vram_thresh = float(getattr(config, 'low_vram_offload_threshold_gb', 28.0))
        self._low_vram_mode = bool(primary_dev.type == 'cuda' and 0.0 < vram_gb <= low_vram_thresh)
        self._stream_surrogates_on_gpu = False
        self._view_microbatch_sizes: list[int | None] = [None] * len(self.surrogates)
        offloaded_text_bytes = 0
        vis_ckpt_count = 0

        if self._low_vram_mode:
            total_vis_bytes = 0
            for idx, surrogate in enumerate(self.surrogates):
                if callable(getattr(surrogate, 'offload_text_tower', None)):
                    offloaded_text_bytes += int(surrogate.offload_text_tower())
                if callable(getattr(surrogate, 'enable_visual_grad_checkpointing', None)):
                    if surrogate.enable_visual_grad_checkpointing(True):
                        vis_ckpt_count += 1
                v_bytes = int(surrogate.visual_parameter_bytes()) if callable(getattr(surrogate, 'visual_parameter_bytes', None)) else 0
                total_vis_bytes += v_bytes
                img_sz = int(getattr(surrogate, 'image_size', 224))
                if v_bytes >= int(1.0 * (1024 ** 3)) or img_sz >= 336:
                    self._view_microbatch_sizes[idx] = 2
            if total_vis_bytes > int(4.2 * (1024 ** 3)):
                self._stream_surrogates_on_gpu = True
                for surrogate in self.surrogates:
                    if callable(getattr(surrogate, 'move_visual_to', None)):
                        surrogate.move_visual_to('cpu')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.audit = {'kind': TRAINING_OBJECTIVE_KIND,
            'surrogates': snapshots, 'surrogate_selection': self.selection.audit(),
            'public_objectives': [o.audit for o in self.objectives],
            'held_out_models': list(HELD_OUT_MODELS),
            'held_out_purifiers': list(HELD_OUT_PURIFIERS),
            'gradient_execution': 'shared_views_and_purified_pcgrad_single_backward' if self.pcgrad_enabled else 'shared_views_and_purified_single_backward',
            'pcgrad_enabled': self.pcgrad_enabled,
            'psi_progressive_schedule': bool(getattr(config, 'psi_progressive_schedule', False)),
            'weak_proxy_weighting': 'baseline_calibrated_relative_gain_softmin_with_uniform_floor',
            'ensemble_temperature': self.ensemble_temperature,
            'ensemble_uniform_mix': self.ensemble_uniform_mix,
            'eot_enabled': bool(getattr(config, 'eot_enabled', True)),
            'eot_paths': ['identity', 'pillow_blur_bpda', 'jpeg_exact_forward_ste', 'resize'],
            'diffusion_weight': self.diffusion_weight, 'diffusion_gradient': 'full',
            'purifier': self.purifier.snapshot(),
            'vram_governance': {
                'low_vram_mode': self._low_vram_mode,
                'offloaded_text_bytes': offloaded_text_bytes,
                'visual_grad_checkpointing_models': vis_ckpt_count,
                'sequential_gpu_streaming': self._stream_surrogates_on_gpu,
            },
            'victim_memory_used': False, 'formal_queries_used': False}
        if training_bundle is not None and float(getattr(config, 'public_retrieval_weight', 0.0)) > 0.0:
            self.audit.update(
                training_manifest_sha256=training_bundle['manifest_sha256'],
                training_bundle_sha256=training_bundle['bundle_sha256'],
                public_retrieval_weight=float(getattr(config, 'public_retrieval_weight', 0.0)),
                negative_candidate_caches=cache_records,
            )

    def _seed(self, step):
        return (int(getattr(self.config, 'seed', 42)) + max(0, int(step))) % (2**63)

    def _public_indices_for_step(self, step: int | None, *, progressive: bool) -> list[int] | None:
        if self._public_sampling is None:
            return None
        if not progressive or step is None or int(step) <= 0:
            return None
        idx = (int(step) - 1) % len(self._public_sampling['step_indices'])
        return self._public_sampling['step_indices'][idx]

    def _shared_inputs(self, image, step, *, gradient):
        base_device = getattr(self.purifier, 'device', self.surrogates[0].device)
        base_img = image.to(base_device)
        eot_on = bool(getattr(self.config, 'eot_enabled', True))
        views = transformed_batch(base_img, step=max(1, int(step))) if eot_on else base_img
        purified = self.purifier(base_img, seed=self._seed(step), gradient=gradient)
        return views, purified, eot_on

    def _evaluate_proxy_from_inputs(
        self,
        index,
        views,
        purified,
        *,
        step: int | None = None,
        progressive: bool = False,
    ):
        surrogate, objective = self.surrogates[index], self.objectives[index]
        if self._stream_surrogates_on_gpu and callable(getattr(surrogate, 'move_visual_to', None)):
            surrogate.move_visual_to(surrogate.device)
        mb_size = self._view_microbatch_sizes[index]
        views_out = _encode_surrogate_image_microbatched(
            surrogate, views.to(surrogate.device), microbatch_size=mb_size
        )
        feature = views_out.global_embedding
        views_patches = getattr(views_out, 'patch_tokens', None)
        score = objective.scores(feature)
        pub_idx = self._public_indices_for_step(step, progressive=progressive)
        eot = objective.value(
            feature,
            patch_tokens=views_patches,
            step=step,
            progressive=progressive,
            public_query_indices=pub_idx,
        )
        purified_out = surrogate.encode_image(purified.to(surrogate.device))
        purified_feature = purified_out.global_embedding
        purified_patches = getattr(purified_out, 'patch_tokens', None)
        pvalue = objective.value(
            purified_feature,
            patch_tokens=purified_patches,
            step=step,
            progressive=progressive,
            public_query_indices=pub_idx,
        )
        utility = (eot + self.diffusion_weight * pvalue) / (1.0 + self.diffusion_weight)

        # 同时计算恒定终端评估标尺值，供 _baseline_values 与退火控制器跨阶段一致比较
        if progressive and bool(getattr(self.config, 'psi_progressive_schedule', False)):
            with torch.no_grad():
                stat_eot = objective.value(feature.detach(), patch_tokens=views_patches.detach() if views_patches is not None else None, progressive=False)
                stat_p = objective.value(purified_feature.detach(), patch_tokens=purified_patches.detach() if purified_patches is not None else None, progressive=False)
                stat_util = float(((stat_eot + self.diffusion_weight * stat_p) / (1.0 + self.diffusion_weight)).detach())
        else:
            stat_util = float(utility.detach())

        return utility, score.mean(0), {
            'eot': float(eot.detach()),
            'purified': float(pvalue.detach()),
            'stationary_utility': stat_util,
        }

    def _maybe_offload_streamed_proxy(self, index: int) -> None:
        if self._stream_surrogates_on_gpu:
            surrogate = self.surrogates[index]
            if callable(getattr(surrogate, 'move_visual_to', None)):
                surrogate.move_visual_to('cpu')

    def _weights_from_values(self, values):
        if self._baseline_values is None or self._baseline_values.shape != values.shape:
            self._baseline_values = values.detach().clone()
        gains = (values - self._baseline_values.to(values.device)).detach()
        adaptive = torch.softmax(-gains / max(float(self.ensemble_temperature), 1e-4), dim=0)
        uni_mix = min(max(float(self.ensemble_uniform_mix), 0.0), 1.0)
        uniform = torch.full_like(adaptive, 1.0 / len(self.surrogates))
        weights = ((1.0 - uni_mix) * adaptive + uni_mix * uniform).detach()
        return weights, gains

    def _one_value(self, index, image, step, gradient):
        views, purified, _ = self._shared_inputs(image, step, gradient=gradient)
        try:
            return self._evaluate_proxy_from_inputs(index, views, purified, step=step, progressive=False)
        finally:
            self._maybe_offload_streamed_proxy(index)

    @torch.no_grad()
    def _values(self, image, step):
        views, purified, _ = self._shared_inputs(image, step, gradient='none')
        values, scores, rows = [], [], []
        for i in range(len(self.surrogates)):
            try:
                value, score, row = self._evaluate_proxy_from_inputs(i, views, purified, step=step, progressive=False)
            finally:
                self._maybe_offload_streamed_proxy(i)
            values.append(value.to(image.device))
            scores.append(score.to(image.device))
            rows.append(row)
        values = torch.stack(values)
        weights, gains = self._weights_from_values(values)
        return values, torch.stack(scores), weights, rows, gains

    @torch.no_grad()
    def evaluate(self, image, step=0):
        values, scores, weights, _, _ = self._values(image, step)
        return (weights * values).sum(), scores

    def image_gradient(self, image, step):
        base_device = getattr(self.purifier, 'device', self.surrogates[0].device)
        leaf_img = image.detach().to(base_device).clone().requires_grad_(True)
        views, purified, eot_on = self._shared_inputs(leaf_img, step, gradient='full')
        v_leaf = views.detach().requires_grad_(True)
        p_leaf = purified.detach().requires_grad_(True)
        stat_values, scores, rows, gv_list, gp_list = [], [], [], [], []
        use_prog = bool(getattr(self.config, 'psi_progressive_schedule', False))
        for i in range(len(self.surrogates)):
            try:
                value, score, row = self._evaluate_proxy_from_inputs(
                    i, v_leaf, p_leaf, step=step, progressive=use_prog
                )
                gv, gp = torch.autograd.grad(value, (v_leaf, p_leaf), retain_graph=False)
            finally:
                self._maybe_offload_streamed_proxy(i)
            stat_val = torch.tensor(row['stationary_utility'], device=image.device, dtype=torch.float32)
            stat_values.append(stat_val)
            scores.append(score.detach().to(image.device))
            rows.append(row)
            gv_list.append(gv.detach())
            gp_list.append(gp.detach())
            del value, score, gv, gp
        values = torch.stack(stat_values)
        scores = torch.stack(scores)
        weights, gains = self._weights_from_values(values)
        if self.pcgrad_enabled and len(self.surrogates) > 1:
            grad_views = pcgrad_merge_gradients(gv_list, weights)
            grad_purified = pcgrad_merge_gradients(gp_list, weights)
        else:
            grad_views = torch.zeros_like(v_leaf)
            grad_purified = torch.zeros_like(p_leaf)
            for i in range(len(self.surrogates)):
                w_i = float(weights[i])
                grad_views.add_(gv_list[i], alpha=w_i)
                grad_purified.add_(gp_list[i], alpha=w_i)
        if eot_on:
            gradient = torch.autograd.grad((views, purified), leaf_img, grad_outputs=(grad_views, grad_purified))[0]
        else:
            gradient = grad_views + torch.autograd.grad(purified, leaf_img, grad_outputs=grad_purified)[0]
        gradient = gradient.detach().to(image.device)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('代理目标梯度含非有限值')
        utility = (weights * values).sum().detach()
        return utility, scores.detach(), gradient, {'model_weights': weights.cpu().tolist(),
            'model_utilities': values.cpu().tolist(), 'model_relative_gains': gains.cpu().tolist(),
            'paths': rows, 'purifier_gradient': 'full', 'pcgrad_applied': bool(self.pcgrad_enabled and len(self.surrogates) > 1)}
