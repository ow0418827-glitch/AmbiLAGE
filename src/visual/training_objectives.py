"""Public-only joint image/text training and sequential four-surrogate gradients.

No memory pool, formal queries or evaluation rank is accepted by this module.
All surrogate image graphs (including the DDIM path) are released one at a time.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from src.paths import MODEL_ROOT, VISUAL_CONFIGS
from src.visual.encoders.backbone import BackboneConfig, OpenCLIPSurrogate
from src.visual.encoders.joint import ChunkedClipEncoder
from src.visual.robust_objectives import softmin, transformed_batch

TRAINING_FAMILIES = ('ViT-B-16', 'ViT-L-14', 'ViT-SO400M-14-SigLIP-384', 'ViT-H-14')
HELD_OUT_MODELS = ('ViT-g-14', 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct')
TRAINING_OBJECTIVE_KIND = 'four_proxy_sequential_public_joint_goal_v1'


def validate_training_specs(specs):
    names = tuple(spec.name for spec in specs)
    if names != TRAINING_FAMILIES:
        raise ValueError('新方法须按顺序配置四个冻结训练代理，留出模型不得参与训练')
    return specs


def load_training_surrogates(device='cuda', config_path=None):
    """Load only declared local models; dependency/model failures propagate."""
    path = Path(config_path or VISUAL_CONFIGS / 'ambilage_surrogates.json')
    raw = json.loads(path.read_text(encoding='utf-8'))
    specs = validate_training_specs(tuple(BackboneConfig.from_mapping(s) for s in raw['surrogates']))
    models = []
    for spec in specs:
        # All relative paths use the relocatable model root, not the shell CWD.
        weight = Path(spec.weight_path) if spec.weight_path else None
        tokenizer = Path(spec.tokenizer_path) if spec.tokenizer_path else None
        spec = replace(spec,
            weight_path=str(MODEL_ROOT / weight) if weight and not weight.is_absolute() else spec.weight_path,
            tokenizer_path=str(MODEL_ROOT / tokenizer) if tokenizer and not tokenizer.is_absolute() else spec.tokenizer_path)
        models.append(OpenCLIPSurrogate(spec, device=device, precision='fp32',
            allow_downloads=False, allow_device_fallback=False, require_true_local_tokens=False))
    return models


class PublicFullKeyObjective:
    """Match the source's public image/text key to one joint public target."""
    def __init__(self, surrogate, target_feature, source_public_key, training_text, config):
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
        target_text = torch.as_tensor(text, dtype=torch.float32, device=surrogate.device).reshape(1, -1)
        self.target = F.normalize(target_feature.detach().float(), dim=-1)
        self.joint_target = F.normalize(self.target + target_text, dim=-1)
        self.audit = {'source_public_key_sha256': hashlib.sha256(source_public_key.encode('utf-8')).hexdigest(),
            'training_text_sha256': hashlib.sha256(training_text.encode('utf-8')).hexdigest(),
            'source_text_encoding': own_audit, 'target_text_encoding': text_audit,
            'query_modes': ['multimodal_equal_weight'],
            'query_image_weight': 0.5, 'query_text_weight': 0.5,
            'memory_key_image_weight': 0.5, 'victim_memory_used': False,
            'formal_queries_used': False, 'formal_ranking_used': False}

    def scores(self, features):
        key = F.normalize(F.normalize(features.float(), dim=-1) + self.own, dim=-1)
        return key @ self.joint_target.T

    def value(self, features):
        scores = self.scores(features)
        joint_weight = float(getattr(self.config, 'joint_weight', 1.0))
        if not math.isfinite(joint_weight) or joint_weight <= 0:
            raise ValueError('图文联合目标权重必须为有限正数')
        temperature = float(self.config.temperature)
        return joint_weight * (0.5 * scores.mean() + 0.5 * softmin(scores, temperature))


class SequentialEnsembleObjective:
    """Detached weakest-proxy weights; exact per-proxy image gradients.

    Recomputing each transformation/purification per proxy bounds graph memory.
    The same seed is reused within one step for scoring and differentiation.
    """
    def __init__(self, surrogates, target_tensor, source_public_key, training_text, config,
                 purifier=None):
        self.surrogates = list(surrogates)
        if not self.surrogates:
            raise ValueError('训练代理不能为空')
        self.config, self.purifier = config, purifier
        self.diffusion_weight = float(getattr(config, 'diffusion_weight', 0.25))
        if not torch.isfinite(torch.tensor(self.diffusion_weight)) or self.diffusion_weight <= 0:
            raise ValueError('完整新方法必须启用正权重扩散净化目标')
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
            snapshot = surrogate.snapshot()
            model_name = str(snapshot.get('name', snapshot.get('architecture', '')))
            if 'vit-g-14' in model_name.casefold() or 'gme' in model_name.casefold():
                raise ValueError('留出评估编码器禁止参与训练')
            with torch.no_grad():
                target = surrogate.encode_image(target_tensor.to(surrogate.device)).global_embedding
            self.objectives.append(PublicFullKeyObjective(surrogate, target, source_public_key, training_text, config))
        self.audit = {'kind': TRAINING_OBJECTIVE_KIND,
            'surrogates': [s.snapshot() for s in self.surrogates],
            'public_objectives': [o.audit for o in self.objectives],
            'held_out_models': list(HELD_OUT_MODELS),
            'gradient_execution': 'one_proxy_graph_at_a_time_recompute_purifier',
            'weak_proxy_weighting': 'detached_softmax_negative_current_utility',
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
        feature = surrogate.encode_image(views).global_embedding
        score = objective.scores(feature)
        eot = objective.value(feature)
        purified = self.purifier(image, seed=self._seed(step), gradient=gradient)
        purified_feature = surrogate.encode_image(purified).global_embedding
        pvalue = objective.value(purified_feature)
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
        weights = torch.softmax(-values / float(self.config.temperature), dim=0).detach()
        return values, torch.stack(scores), weights, rows

    @torch.no_grad()
    def evaluate(self, image, step=0):
        values, scores, weights, _ = self._values(image, step)
        return (weights * values).sum(), scores

    def image_gradient(self, image, step):
        values, scores, weights, rows = self._values(image.detach(), step)
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
            'model_utilities': values.cpu().tolist(), 'paths': rows, 'purifier_gradient': 'full'}
