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


def resolve_effective_category_mode(config, training_bundle) -> str:
    raw = str(getattr(config, 'category_data_mode', 'auto') or 'auto')
    if raw == 'auto':
        return 'category_commonalities_abstraction' if training_bundle is not None else 'legacy_target_product'
    return raw


def resolve_effective_key_mode(config, category_mode: str, training_bundle) -> str:
    raw = str(getattr(config, 'memory_key_mode', 'auto') or 'auto')
    if raw == 'auto':
        if training_bundle is not None or category_mode != 'legacy_target_product':
            return 'image'
        return 'joint'
    return raw


def single_model_dominant_pcgrad(
    g_primary: torch.Tensor,
    g_aux: torch.Tensor | None,
    *,
    aux_cap: float = 0.80,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, bool]:
    """将单代理辅助视觉/净化梯度投影至与主类别检索梯度非冲突的半空间，避免局部纹理冲毁大类方向。"""
    if g_aux is None:
        return g_primary, False
    p64 = g_primary.reshape(-1).to(dtype=torch.float64)
    a64 = g_aux.reshape(-1).to(dtype=torch.float64)
    p_norm = float(p64.norm().item())
    a_norm = float(a64.norm().item())
    if a_norm <= eps:
        return g_primary, False
    if p_norm <= eps:
        return g_aux, False
    dot = float(torch.dot(p64, a64).item())
    conflicted = dot < 0.0
    if conflicted:
        a64 = a64 - (dot / max(p_norm * p_norm, eps * eps)) * p64
    a_proj_norm = float(a64.norm().item())
    max_norm = float(aux_cap) * p_norm
    if a_proj_norm > max_norm > 0.0:
        a64 = a64 * (max_norm / a_proj_norm)
    merged = (p64 + a64).to(dtype=g_primary.dtype).reshape_as(g_primary)
    return merged, conflicted


def multiscale_grid_breaking_views(
    image: torch.Tensor,
    *,
    step: int,
    base_views: torch.Tensor,
) -> torch.Tensor:
    """为单代理训练追加确定性多尺度裁剪缩放视图，打破 CLIP-ViT-L/14 固定 14x14 分块边界过拟合。"""
    from src.visual.image_utils import quantize_ste
    x = quantize_ste(image)
    _, _, h, w = x.shape
    scales = (0.91, 0.96, 0.93, 1.05)
    s1 = scales[(int(step) - 1) % 4]
    s2 = scales[(int(step) + 1) % 4]
    extra = []
    for idx, scale in enumerate((s1, s2)):
        if scale < 1.0:
            ch = max(4, min(h, int(round(h * scale))))
            cw = max(4, min(w, int(round(w * scale))))
            max_y = max(0, h - ch)
            max_x = max(0, w - cw)
            top = ((int(step) * 7 + idx * 13) % (max_y + 1)) if max_y > 0 else 0
            left = ((int(step) * 11 + idx * 5) % (max_x + 1)) if max_x > 0 else 0
            cropped = x[:, :, top : top + ch, left : left + cw]
            resized = F.interpolate(
                cropped, size=(h, w), mode='bicubic', align_corners=False, antialias=True
            ).clamp(0, 1)
        else:
            up_h = max(h + 2, int(round(h * scale)))
            up_w = max(w + 2, int(round(w * scale)))
            zoomed = F.interpolate(
                x, size=(up_h, up_w), mode='bicubic', align_corners=False, antialias=True
            )
            top = (up_h - h) // 2
            left = (up_w - w) // 2
            resized = zoomed[:, :, top : top + h, left : left + w].clamp(0, 1)
        extra.append(quantize_ste(resized))
    return torch.cat([base_views, *extra], dim=0)


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


def _project_centers_to_text_space(
    surrogate,
    centers: torch.Tensor,
    target_dim: int,
) -> torch.Tensor | None:
    """将尚未经过视觉线性投影层的前景聚类中心映射至图文联合嵌入空间；不兼容时返回 None。"""
    if centers.shape[-1] == target_dim:
        return F.normalize(centers.float(), dim=-1)
    visual = getattr(getattr(surrogate, 'model', None), 'visual', None)
    if visual is None:
        visual = getattr(surrogate, 'visual', None)
    if visual is None:
        return None
    proj = getattr(visual, 'proj', None)
    if proj is None:
        return None
    ln_post = getattr(visual, 'ln_post', None)
    with torch.no_grad():
        if isinstance(proj, torch.Tensor):
            if proj.ndim != 2 or tuple(proj.shape) != (int(centers.shape[-1]), int(target_dim)):
                return None
            x = centers.to(device=proj.device, dtype=proj.dtype)
            if callable(ln_post):
                x = ln_post(x)
            projected = x @ proj
            return F.normalize(projected.to(device=centers.device, dtype=torch.float32), dim=-1)
        if callable(proj):
            ref_param = next(proj.parameters(), None) if hasattr(proj, 'parameters') else None
            x = centers.to(device=ref_param.device, dtype=ref_param.dtype) if ref_param is not None else centers.float()
            if callable(ln_post):
                x = ln_post(x)
            projected = proj(x)
            if torch.is_tensor(projected) and int(projected.shape[-1]) == int(target_dim):
                return F.normalize(projected.to(device=centers.device, dtype=torch.float32), dim=-1)
    return None


class PublicFullKeyObjective:
    """匹配源商品公开图文全键至联合目标，并融合纯图类别键、大类共性抽象与文本加权前景局部 OT。"""
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
        memory_key_mode='joint',
        category_data_mode='legacy_target_product',
        category_abstraction=None,
        bundle_concept_feature=None,
    ):
        for label, value in (('公开商品键', source_public_key), ('目标特点', training_text)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(label + '必须为非空公开正文')
            if any(marker in value.casefold() for marker in
                   ('actions and observations:', 'observed purchase:', 'action: think[')):
                raise ValueError(label + '不能包含私人购物轨迹')
        self.config = config
        self.effective_key_mode = str(memory_key_mode or 'joint')
        self.effective_category_mode = str(category_data_mode or 'legacy_target_product')
        self.category_abstraction = category_abstraction
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

        if bundle_concept_feature is not None:
            self.bundle_concept_feature = F.normalize(
                bundle_concept_feature.detach().float().to(surrogate.device).reshape(1, -1), dim=-1
            ).detach()
        else:
            self.bundle_concept_feature = None

        if category_abstraction is not None:
            self.mu_cat = F.normalize(
                category_abstraction['mu_cat'].detach().float().to(surrogate.device).reshape(1, -1), dim=-1
            ).detach()
            self.visual_prototype = F.normalize(
                category_abstraction['visual_category_prototype'].detach().float().to(surrogate.device).reshape(1, -1),
                dim=-1,
            ).detach()
            self.category_basis = category_abstraction['basis'].detach().float().to(surrogate.device).detach()
            self.category_anchor_bank = F.normalize(
                category_abstraction['pure_semantic_bank'].detach().float().to(surrogate.device), dim=-1
            ).detach()
            self.visual_bridge_target = F.normalize(
                0.65 * self.visual_prototype + 0.35 * self.mu_cat, dim=-1
            ).detach()
        else:
            self.mu_cat = None
            self.visual_prototype = None
            self.category_basis = None
            self.category_anchor_bank = None
            if self.effective_category_mode == 'direct_vi_bundle' and self.bundle_concept_feature is not None:
                self.visual_bridge_target = F.normalize(
                    0.65 * self.target + 0.35 * self.bundle_concept_feature, dim=-1
                ).detach()
            else:
                self.visual_bridge_target = self.joint_target

        if source_clean_feature is not None:
            raw_src = getattr(source_clean_feature, 'global_embedding', source_clean_feature)
            src_norm = F.normalize(raw_src.detach().float().to(surrogate.device), dim=-1)
            self.source_clean_img = src_norm.detach()
            self.source_clean_key = F.normalize(src_norm + self.own, dim=-1).detach()
        else:
            self.source_clean_img = None
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
        joint_key_aux_mix = float(getattr(config, 'joint_key_aux_mix', 0.0))
        if not math.isfinite(joint_key_aux_mix) or not (0.0 <= joint_key_aux_mix <= 1.0):
            raise ValueError('辅助联合记忆键混合比例 joint_key_aux_mix 必须位于 [0.0, 1.0]')
        self.joint_key_aux_mix = joint_key_aux_mix
        self.joint_grad_protect = bool(getattr(config, 'joint_grad_protect', False))
        self.joint_grad_cap = float(getattr(config, 'joint_grad_cap', 0.35))
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
        ot_ref_text = self.mu_cat if self.mu_cat is not None else self.target_text
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
                    proj_centers = _project_centers_to_text_space(
                        surrogate, self.target_fg_centers[0], int(ot_ref_text.shape[-1])
                    )
                    if proj_centers is not None:
                        center_text_logits = (
                            (proj_centers @ ot_ref_text.to(device=proj_centers.device, dtype=proj_centers.dtype).T).squeeze(-1)
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
        if self.effective_key_mode == 'image':
            query_modes.append('image_only_memory_key_alignment')
        if self.effective_category_mode != 'legacy_target_product':
            query_modes.append('category_level_' + self.effective_category_mode)
        mem_key_img_w = 1.0 if self.effective_key_mode == 'image' else (0.75 if self.effective_key_mode == 'hybrid' else 0.5)
        self.audit = {'source_public_key_sha256': hashlib.sha256(source_public_key.encode('utf-8')).hexdigest(),
            'training_text_sha256': hashlib.sha256(training_text.encode('utf-8')).hexdigest(),
            'source_text_encoding': own_audit, 'target_text_encoding': text_audit,
            'query_modes': query_modes,
            'query_image_weight': 0.5, 'query_text_weight': 0.5,
            'memory_key_image_weight': mem_key_img_w,
            'memory_key_mode': self.effective_key_mode,
            'category_data_mode': self.effective_category_mode,
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
            'joint_key_aux_mix': self.joint_key_aux_mix,
            'joint_grad_protect': self.joint_grad_protect,
            'joint_grad_cap': self.joint_grad_cap,
            'local_ot_clusters': self.local_ot_clusters,
            'local_ot_fg_threshold': self.local_ot_fg_threshold,
            'local_ot_text_weighted': bool(self.local_ot_text_weighted and self.target_fg_center_weights is not None),
            'local_ot_active': bool(self.local_ot_weight > 0.0 and self.target_fg_centers is not None),
            'public_bundle_retrieval_active': bool(self.public_retrieval_weight > 0.0),
            'victim_memory_used': False,
            'formal_queries_used': False, 'formal_ranking_used': False}
        if category_abstraction is not None:
            self.audit['category_abstraction'] = category_abstraction['audit']

    def scores(self, features):
        img_norm = F.normalize(features.float(), dim=-1)
        if self.effective_key_mode == 'image':
            bridge = self.visual_bridge_target.to(img_norm.device)
            pure_scores = img_norm @ bridge.T
            if self.joint_key_aux_mix <= 0.0:
                return pure_scores
        own = self.own.to(img_norm.device)
        joint_tgt = self.joint_target.to(img_norm.device)
        key = F.normalize(img_norm + own, dim=-1)
        joint_scores = key @ joint_tgt.T
        if self.compensation_weight > 0.0 and self.compensated_valid:
            comp_scores = img_norm @ self.compensated_target.to(img_norm.device).T
            legacy_scores = (1.0 - self.compensation_weight) * joint_scores + self.compensation_weight * comp_scores
        else:
            legacy_scores = joint_scores
        if self.effective_key_mode == 'image':
            return (1.0 - self.joint_key_aux_mix) * pure_scores + self.joint_key_aux_mix * legacy_scores
        if self.effective_key_mode == 'hybrid':
            pure_scores = img_norm @ self.visual_bridge_target.to(img_norm.device).T
            return 0.5 * pure_scores + 0.5 * legacy_scores
        return legacy_scores

    def text_scores(self, features):
        """适应 AmbiLAGE 的主句子独立文本对齐打分（支持 image 纯图键与兼容 joint 键）。"""
        img_norm = F.normalize(features.float(), dim=-1)
        tgt_txt = (self.mu_cat if self.mu_cat is not None else self.target_text).to(img_norm.device)
        if self.effective_key_mode == 'image':
            return img_norm @ tgt_txt.T
        own = self.own.to(img_norm.device)
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

    def _pure_image_text_utility(self, img_norm: torch.Tensor, temperature: float, *, progressive: bool) -> torch.Tensor:
        if (
            self.effective_category_mode in ('category_commonalities_abstraction', 'category_subspace_hybrid')
            and self.category_anchor_bank is not None
        ):
            phrases = self.category_anchor_bank.to(img_norm.device)
        elif self.bundle_concept_feature is not None:
            phrases = F.normalize(
                torch.cat([self.bundle_concept_feature.to(img_norm.device), self.phrase_features.to(img_norm.device)], dim=0),
                dim=-1,
            )
        else:
            phrases = self.phrase_features.to(img_norm.device)

        clean_mat = img_norm @ phrases.T
        if progressive and self.spherical_augmentation and self.spherical_radius > 0.0 and self.spherical_mix > 0.0:
            v_dir = img_norm.detach().mean(dim=0, keepdim=True)
            adv_phrases = worst_case_spherical_queries(phrases, v_dir, self.spherical_radius)
            adv_mat = img_norm @ adv_phrases.T
            score_mat = (1.0 - self.spherical_mix) * clean_mat + self.spherical_mix * adv_mat
        else:
            score_mat = clean_mat

        u_phrases = 0.5 * score_mat.mean() + 0.5 * softmin(score_mat.flatten(), max(temperature, 0.08))
        if self.mu_cat is not None:
            u_mu = (img_norm @ self.mu_cat.to(img_norm.device).T).mean()
            u_phrases = 0.65 * u_phrases + 0.35 * u_mu
        if self.source_repulsion_weight > 0.0 and self.source_clean_img is not None:
            s_pos = img_norm @ phrases.T
            s_src = (self.source_clean_img.to(img_norm.device) @ phrases.T).detach()
            tau_nce = 0.07
            u_repel = (-tau_nce * F.softplus((s_src - (s_pos - self.source_repulsion_margin)) / tau_nce)).mean()
            u_phrases = u_phrases + self.source_repulsion_weight * u_repel
        return u_phrases

    def enhanced_text_utility(self, features: torch.Tensor, temperature: float, *, progressive: bool = False) -> torch.Tensor:
        """多粒度子句簇/共性原型 + 闭式球面最坏查询增强 + 源干净特征对比排斥效用。"""
        img_norm = F.normalize(features.float(), dim=-1)
        if self.effective_key_mode == 'image':
            return self._pure_image_text_utility(img_norm, temperature, progressive=progressive)

        has_v4_text = (
            self.phrase_decomposition
            or self.text_pure_image_mix > 0.0
            or self.spherical_augmentation
            or (self.source_repulsion_weight > 0.0 and self.source_clean_key is not None)
        )
        if not has_v4_text:
            t_scores = self.text_scores(features)
            u_legacy = 0.5 * t_scores.mean() + 0.5 * softmin(t_scores, temperature)
        else:
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

            u_legacy = 0.5 * score_mat.mean() + 0.5 * softmin(score_mat.flatten(), max(temperature, 0.08))

            if self.source_repulsion_weight > 0.0 and self.source_clean_key is not None:
                clean_key = self.source_clean_key.to(img_norm.device)
                s_pos = key @ phrases.T
                s_src = (clean_key @ phrases.T).detach()
                tau_nce = 0.07
                logits_gap = (s_src - (s_pos - self.source_repulsion_margin)) / tau_nce
                u_repel = (-tau_nce * F.softplus(logits_gap)).mean()
                u_legacy = u_legacy + self.source_repulsion_weight * u_repel
        if self.effective_key_mode == 'hybrid':
            u_pure = self._pure_image_text_utility(img_norm, temperature, progressive=progressive)
            return 0.5 * u_pure + 0.5 * u_legacy
        return u_legacy

    def public_bundle_retrieval_utility(
        self,
        features: torch.Tensor,
        query_indices: list[int] | None,
        *,
        progressive: bool = False,
    ) -> torch.Tensor | None:
        """温度尺度化全库 InfoNCE + Top-5 困难竞品软间隔对决对比检索效用（支持直接/纯共性/子空间混合接口）。"""
        if self.public_retrieval_state is None or self.public_retrieval_weight <= 0.0:
            return None
        from src.visual._visual_inception.negative_sampling import select_hard_negatives

        img_norm = F.normalize(features.float(), dim=-1)
        state = self.public_retrieval_state
        negatives = state['negative_features'].to(img_norm.device)
        policy = state['negative_policy']

        if self.effective_category_mode == 'category_commonalities_abstraction' and 'abstracted_retrieval_queries' in state:
            q_batch = state['abstracted_retrieval_queries'].to(img_norm.device)
        elif self.effective_category_mode == 'category_subspace_hybrid' and 'denoised_queries_hybrid' in state:
            q_pool = state['denoised_queries_hybrid'].to(img_norm.device)
            q_slice = q_pool if query_indices is None else q_pool[query_indices]
            if progressive and self.mu_cat is not None:
                q_batch = torch.cat([self.mu_cat.to(img_norm.device), q_slice], dim=0)
            else:
                q_batch = q_slice
        else:
            q_all = state['query_features'].to(img_norm.device)
            q_batch = q_all if query_indices is None else q_all[query_indices]
        q_batch = F.normalize(q_batch.detach(), dim=-1)

        if self.effective_key_mode == 'image':
            eval_keys = [img_norm]
        elif self.effective_key_mode == 'hybrid':
            own = self.own.to(img_norm.device)
            eval_keys = [img_norm, F.normalize(img_norm + own, dim=-1)]
        else:
            own = self.own.to(img_norm.device)
            eval_keys = [img_norm, F.normalize(img_norm + own, dim=-1)]

        tau = max(float(getattr(self.config, 'public_retrieval_temperature', 0.07)), 1e-4)
        margin = float(getattr(self.config, 'public_contrastive_margin', 0.30)) if progressive else 0.0
        indices, _ = select_hard_negatives(q_batch, negatives.detach(), policy, state['neg_config'], include_per_query=False)
        indices = indices.to(device=img_norm.device, dtype=torch.long)

        def _loss_for_queries(q_vecs: torch.Tensor, q_idx: torch.Tensor) -> torch.Tensor:
            all_neg_cos = q_vecs @ negatives.detach().T  # [Q, N_bank]
            boost = torch.zeros_like(all_neg_cos)
            if margin > 0.0:
                boost.scatter_(1, q_idx, margin)
            neg_logits = all_neg_cos / tau + boost
            selected_cos = torch.gather(all_neg_cos, 1, q_idx)
            top_r = min(5, selected_cos.shape[1])
            comp_logit = torch.logsumexp(selected_cos.topk(top_r, dim=1).values / tau, dim=1) - math.log(float(top_r))
            key_losses = []
            for k_mat in eval_keys:
                pos_logits = (q_vecs @ k_mat.T) / tau  # [Q, B]
                view_losses = []
                for b_idx in range(k_mat.shape[0]):
                    p_b = pos_logits[:, b_idx : b_idx + 1]
                    bank_l = torch.logsumexp(torch.cat([p_b, neg_logits], dim=1), dim=1) - p_b[:, 0]
                    comp_l = F.softplus(2.0 * (comp_logit - p_b[:, 0] + margin)) / 2.0
                    view_losses.append(0.3 * bank_l + 0.7 * comp_l)
                key_losses.append(torch.stack(view_losses).mean())
            return torch.stack(key_losses).mean()

        loss_total = _loss_for_queries(q_batch, indices)
        adv_radius = float(getattr(self.config, 'public_query_adv_radius', 0.0))
        adv_steps = int(getattr(self.config, 'public_query_adv_steps', 0))
        adv_mix = float(getattr(self.config, 'public_query_adv_mix', 0.0))
        if progressive and adv_radius > 0.0 and adv_steps > 0 and adv_mix > 0.0:
            basis = self.category_basis.to(img_norm.device) if self.category_basis is not None else None
            key_anchor = F.normalize(eval_keys[0].detach().mean(dim=0, keepdim=True), dim=-1)
            q_cur = q_batch.clone()
            step_sz = max(adv_radius / max(1, adv_steps), 1e-3)
            for _ in range(adv_steps):
                # Minimize <key_anchor, q> within text subspace & sphere cap (hardest worst-case query)
                tan = -key_anchor.expand_as(q_cur)
                if basis is not None and basis.shape[0] > 0:
                    tan = (tan @ basis.T) @ basis
                tan = tan - (tan * q_cur).sum(dim=-1, keepdim=True) * q_cur
                t_norm = tan.norm(dim=-1, keepdim=True)
                direction = torch.where(t_norm > 1e-12, tan / t_norm.clamp_min(1e-12), torch.zeros_like(tan))
                q_cand = F.normalize(q_cur + step_sz * direction, dim=-1)
                # Project onto L2 cap around q_batch on S^{D-1}
                dot_orig = (q_cand * q_batch).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
                min_cos = max(-1.0, 1.0 - 0.5 * adv_radius * adv_radius)
                orth = q_cand - dot_orig * q_batch
                o_norm = orth.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cap_proj = F.normalize(min_cos * q_batch + math.sqrt(max(0.0, 1.0 - min_cos * min_cos)) * (orth / o_norm), dim=-1)
                q_cur = torch.where(dot_orig >= min_cos, q_cand, cap_proj).detach()
            idx_adv, _ = select_hard_negatives(q_cur, negatives.detach(), policy, state['neg_config'], include_per_query=False)
            loss_adv = _loss_for_queries(q_cur, idx_adv.to(device=img_norm.device, dtype=torch.long))
            loss_total = (1.0 - adv_mix) * loss_total + adv_mix * loss_adv

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

    def split_three_component_values(
        self,
        features,
        patch_tokens=None,
        *,
        step: int | None = None,
        progressive: bool = False,
        public_query_indices: list[int] | None = None,
        text_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """分解返回：(u_cat, u_joint_weighted, u_ot_weighted, total, u_joint, u_text)。"""
        scores = self.scores(features)
        w_joint, w_text, w_ot = psi_cross_modal_weights(self.config, step=step, progressive=progressive)
        if (
            not progressive
            and bool(getattr(self.config, "text_aware_proxy_weighting", False))
            and w_text > 0.0
        ):
            parity_mix = float(getattr(self.config, "selection_text_parity_mix", 0.0))
            if parity_mix > 0.0:
                w_text = max(w_text, parity_mix * w_joint)
        w_text = w_text * float(text_multiplier)
        temperature = float(self.config.temperature)
        u_joint = 0.5 * scores.mean() + 0.5 * softmin(scores, temperature)
        u_joint_weighted = w_joint * u_joint
        u_ot_weighted = u_joint_weighted.new_zeros(())
        u_cat = u_joint_weighted.new_zeros(())
        if w_text > 0.0:
            u_text = self.enhanced_text_utility(features, temperature, progressive=progressive)
            u_cat = u_cat + w_text * u_text
        else:
            u_text = u_joint_weighted.new_zeros(())
        if w_ot > 0.0 and patch_tokens is not None:
            ot_scores = self.local_ot_scores(patch_tokens)
            if ot_scores is not None:
                u_ot = 0.5 * ot_scores.mean() + 0.5 * softmin(ot_scores, temperature)
                u_ot_weighted = u_ot_weighted + w_ot * u_ot
        if self.public_retrieval_weight > 0.0:
            u_pub = self.public_bundle_retrieval_utility(features, public_query_indices, progressive=progressive)
            if u_pub is not None:
                u_cat = u_cat + self.public_retrieval_weight * u_pub
        total = u_cat + u_joint_weighted + u_ot_weighted
        return u_cat, u_joint_weighted, u_ot_weighted, total, u_joint, u_text

    def split_component_values(
        self,
        features,
        patch_tokens=None,
        *,
        step: int | None = None,
        progressive: bool = False,
        public_query_indices: list[int] | None = None,
        text_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """分解返回：(u_cat 主类别效用, u_vis 视觉/局部OT辅助效用, total, u_joint, u_text)。"""
        u_cat, u_joint_w, u_ot_w, total, u_joint, u_text = self.split_three_component_values(
            features,
            patch_tokens=patch_tokens,
            step=step,
            progressive=progressive,
            public_query_indices=public_query_indices,
            text_multiplier=text_multiplier,
        )
        return u_cat, u_joint_w + u_ot_w, total, u_joint, u_text

    def component_values(
        self,
        features,
        patch_tokens=None,
        *,
        step: int | None = None,
        progressive: bool = False,
        public_query_indices: list[int] | None = None,
        text_multiplier: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, total, u_joint, u_text = self.split_component_values(
            features,
            patch_tokens=patch_tokens,
            step=step,
            progressive=progressive,
            public_query_indices=public_query_indices,
            text_multiplier=text_multiplier,
        )
        return total, u_joint, u_text

    def value(
        self,
        features,
        patch_tokens=None,
        *,
        step: int | None = None,
        progressive: bool = False,
        public_query_indices: list[int] | None = None,
        text_multiplier: float = 1.0,
    ):
        total, _, _ = self.component_values(
            features,
            patch_tokens=patch_tokens,
            step=step,
            progressive=progressive,
            public_query_indices=public_query_indices,
            text_multiplier=text_multiplier,
        )
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
        eot_on = bool(getattr(config, 'eot_enabled', False))
        raw_diff_eot = getattr(config, 'diffusion_eot_enabled', None)
        if raw_diff_eot is None:
            self.diffusion_eot_enabled = bool(eot_on or purifier is not None)
        else:
            self.diffusion_eot_enabled = bool(raw_diff_eot)
        self.diffusion_weight = float(getattr(config, 'diffusion_weight', 0.08))
        if not torch.isfinite(torch.tensor(self.diffusion_weight)) or (
            self.diffusion_weight <= 0 if self.diffusion_eot_enabled else self.diffusion_weight < 0
        ):
            raise ValueError('扩散净化目标权重无效')
        self.ensemble_temperature = float(getattr(config, 'ensemble_temperature', 0.20))
        self.ensemble_uniform_mix = float(getattr(config, 'ensemble_uniform_mix', 0.45))
        if not math.isfinite(self.ensemble_temperature) or self.ensemble_temperature <= 0:
            raise ValueError('集成权重温度 ensemble_temperature 必须为有限正数')
        if not math.isfinite(self.ensemble_uniform_mix) or not (0.0 <= self.ensemble_uniform_mix <= 1.0):
            raise ValueError('均匀保底混合比例 ensemble_uniform_mix 必须位于 [0.0, 1.0]')
        self.pcgrad_enabled = bool(getattr(config, 'pcgrad_enabled', False))
        self.text_aware_proxy_weighting = (
            bool(getattr(config, 'text_aware_proxy_weighting', False))
            and float(getattr(config, 'text_weight', 0.0)) > 0.0
        )
        self.proxy_text_bottleneck_mix = min(
            max(float(getattr(config, 'proxy_text_bottleneck_mix', 0.55)), 0.0), 1.0
        )
        self._baseline_values: torch.Tensor | None = None
        self._baseline_joint_values: torch.Tensor | None = None
        self._baseline_text_values: torch.Tensor | None = None
        self.last_eval_diagnostics: dict[str, Any] = {}

        explicit_purifier_path = (
            getattr(config, 'purifier_model_path', None)
            or getattr(config, 'purifier_model_dir', None)
        )
        if is_held_out_purifier_reference(explicit_purifier_path):
            raise ValueError('留出评估净化器 SD 2.1-base 禁止用于训练阶段')
        if self.diffusion_eot_enabled:
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
                raise ValueError('开启扩散净化期望变换时固定十步扩散净化')
            if is_held_out_purifier_reference(getattr(getattr(self.purifier, 'config', None), 'model_path', None)):
                raise ValueError('留出评估净化器 SD 2.1-base 禁止用于训练阶段')
        else:
            self.diffusion_weight = 0.0
            self.purifier = None

        self.effective_category_mode = resolve_effective_category_mode(config, training_bundle)
        self.effective_key_mode = resolve_effective_key_mode(config, self.effective_category_mode, training_bundle)
        self._public_sampling = None
        public_states: list[dict[str, Any] | None] = [None] * len(self.surrogates)
        cache_records = []
        abstraction_audits: list[dict[str, Any]] = []
        bundle_concept_features: list[torch.Tensor | None] = [None] * len(self.surrogates)
        has_public_retrieval = bool(
            training_bundle is not None and float(getattr(config, 'public_retrieval_weight', 0.0)) > 0.0
        )
        need_bundle_preparation = bool(
            training_bundle is not None
            and (has_public_retrieval or self.effective_category_mode != 'legacy_target_product')
        )
        if has_public_retrieval:
            from src.visual._visual_inception.objective import sample_query_indices

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
        fixture_flag = bool(getattr(config, 'smoke_test', False))
        cache_root = Path(candidate_cache_dir) if candidate_cache_dir else (PROJECT / 'output/datasets/public/ambilage_candidate_cache')
        neg_cfg = SimpleNamespace(
            hard_negative_count=32,
            hard_negative_same_category_count=8,
            hard_negative_proxy_memory_count=8,
            hard_negative_popularity_count=8,
        )

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
            cat_abstraction = None

            if need_bundle_preparation:
                from src.visual._visual_inception.retrieval_cache import encode_candidate_cache
                from src.visual._visual_inception.negative_sampling import build_negative_policy

                enc = ChunkedClipEncoder(SimpleNamespace(
                    surrogate=surrogate, device=surrogate.device, snapshot=surrogate.snapshot()
                ))
                concept_txt = str(training_bundle.get('target_concept_text') or '').strip()
                if concept_txt:
                    bundle_concept_features[idx] = torch.as_tensor(
                        enc.encode_text(concept_txt), device=surrogate.device, dtype=torch.float32
                    ).reshape(1, -1).detach()
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
                pub_state: dict[str, Any] = {
                    'query_features': q_mat,
                    'negative_features': cand_vecs.detach(),
                    'negative_policy': neg_pol,
                    'neg_config': neg_cfg,
                }
                if self.effective_category_mode in ('category_commonalities_abstraction', 'category_subspace_hybrid'):
                    from src.visual.category_abstraction import build_category_abstraction

                    cat_abstraction = build_category_abstraction(
                        training_bundle,
                        lambda txt: torch.as_tensor(enc.encode_text(txt)).detach().cpu(),
                        cand_vecs,
                        target,
                        same_category_indices=neg_pol['role_indices'].get('same_category'),
                        subspace_rank=int(getattr(config, 'category_subspace_rank', 6)),
                        medoid_count=int(getattr(config, 'category_medoid_count', 12)),
                        residual_shrinkage=float(getattr(config, 'category_residual_shrinkage', 0.30)),
                        visual_prototype_mix=float(getattr(config, 'category_visual_prototype_mix', 0.65)),
                        precomputed_query_vecs=q_mat,
                    )
                    pub_state['abstracted_retrieval_queries'] = cat_abstraction['pure_semantic_bank'].detach()
                    pub_state['denoised_queries_hybrid'] = cat_abstraction['denoised_queries_hybrid'].detach()
                    abstraction_audits.append(cat_abstraction['audit'])
                public_states[idx] = pub_state

            self.objectives.append(PublicFullKeyObjective(
                surrogate, target, source_public_key, training_text, config,
                target_patch_tokens=target_patches,
                source_fg_mask=source_fg_mask,
                target_fg_mask=target_fg_mask,
                source_clean_feature=src_clean_out,
                public_retrieval_state=public_states[idx],
                memory_key_mode=self.effective_key_mode,
                category_data_mode=self.effective_category_mode,
                category_abstraction=cat_abstraction,
                bundle_concept_feature=bundle_concept_features[idx],
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
                if len(self.surrogates) > 1 or v_bytes >= int(2.5 * (1024 ** 3)) or img_sz >= 336:
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

        single_task_pcgrad = bool(
            len(self.surrogates) == 1 and bool(getattr(config, 'single_model_task_pcgrad', False))
        )
        eot_on = bool(getattr(config, 'eot_enabled', False))
        raw_ms_eot = getattr(config, 'multiscale_eot_enabled', None)
        if raw_ms_eot is not None:
            self.multiscale_eot_enabled = bool(raw_ms_eot)
        else:
            self.multiscale_eot_enabled = bool(
                len(self.surrogates) == 1
                and eot_on
                and bool(getattr(config, 'single_model_multiscale_eot', False))
            )
        eot_paths = ['identity', 'pillow_blur_bpda', 'jpeg_exact_forward_ste', 'resize'] if eot_on else ['identity']
        if self.multiscale_eot_enabled:
            eot_paths.append('multiscale_grid_breaking_crop_zoom')
        grad_exec_mode = (
            'single_surrogate_task_pcgrad_single_backward'
            if single_task_pcgrad
            else (
                'shared_views_and_purified_pcgrad_single_backward'
                if self.pcgrad_enabled
                else 'shared_views_and_purified_single_backward'
            )
        )
        self.audit = {'kind': TRAINING_OBJECTIVE_KIND,
            'surrogates': snapshots, 'surrogate_selection': self.selection.audit(),
            'public_objectives': [o.audit for o in self.objectives],
            'held_out_models': list(HELD_OUT_MODELS),
            'held_out_purifiers': list(HELD_OUT_PURIFIERS),
            'single_surrogate_mode': bool(len(self.surrogates) == 1),
            'single_model_task_pcgrad': single_task_pcgrad,
            'single_model_multiscale_eot': self.multiscale_eot_enabled,
            'multiscale_eot_enabled': self.multiscale_eot_enabled,
            'defense_eot_enabled': eot_on,
            'effective_key_mode': self.effective_key_mode,
            'effective_category_mode': self.effective_category_mode,
            'gradient_execution': grad_exec_mode,
            'pcgrad_enabled': self.pcgrad_enabled,
            'psi_progressive_schedule': bool(getattr(config, 'psi_progressive_schedule', False)),
            'text_aware_proxy_weighting': self.text_aware_proxy_weighting,
            'weak_proxy_weighting': (
                'text_bottleneck_relative_gain_softmin_with_uniform_floor'
                if self.text_aware_proxy_weighting
                else 'baseline_calibrated_relative_gain_softmin_with_uniform_floor'
            ),
            'ensemble_temperature': self.ensemble_temperature,
            'ensemble_uniform_mix': self.ensemble_uniform_mix,
            'eot_enabled': eot_on,
            'diffusion_eot_enabled': self.diffusion_eot_enabled,
            'eot_paths': eot_paths,
            'diffusion_weight': self.diffusion_weight,
            'diffusion_gradient': 'full' if self.diffusion_eot_enabled else 'disabled',
            'purifier': self.purifier.snapshot() if self.purifier is not None else {'kind': 'disabled', 'enabled': False},
            'vram_governance': {
                'low_vram_mode': self._low_vram_mode,
                'offloaded_text_bytes': offloaded_text_bytes,
                'visual_grad_checkpointing_models': vis_ckpt_count,
                'sequential_gpu_streaming': self._stream_surrogates_on_gpu,
            },
            'victim_memory_used': False, 'formal_queries_used': False}
        if need_bundle_preparation and training_bundle is not None:
            self.audit.update(
                training_manifest_sha256=training_bundle['manifest_sha256'],
                training_bundle_sha256=training_bundle['bundle_sha256'],
                public_retrieval_weight=float(getattr(config, 'public_retrieval_weight', 0.0)),
                negative_candidate_caches=cache_records,
            )
            if abstraction_audits:
                self.audit['category_abstraction'] = abstraction_audits[0]

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
        base_device = getattr(self.purifier, 'device', self.surrogates[0].device) if self.purifier is not None else self.surrogates[0].device
        base_img = image.to(base_device)
        eot_on = bool(getattr(self.config, 'eot_enabled', False))
        views = transformed_batch(base_img, step=max(1, int(step))) if eot_on else base_img
        if self.multiscale_eot_enabled:
            views = multiscale_grid_breaking_views(base_img, step=max(1, int(step)), base_views=views)
        views_transformed = bool(eot_on or self.multiscale_eot_enabled)
        purified = (
            self.purifier(base_img, seed=self._seed(step), gradient=gradient)
            if (self.diffusion_eot_enabled and self.purifier is not None)
            else None
        )
        return views, purified, views_transformed

    def _evaluate_proxy_split_core(
        self,
        index,
        views,
        purified,
        *,
        step: int | None = None,
        progressive: bool = False,
        text_multiplier: float | None = None,
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

        has_purified = bool(self.diffusion_eot_enabled and purified is not None and self.diffusion_weight > 0.0)
        if has_purified:
            purified_out = surrogate.encode_image(purified.to(surrogate.device))
            purified_feature = purified_out.global_embedding
            purified_patches = getattr(purified_out, 'patch_tokens', None)
        else:
            purified_feature = None
            purified_patches = None

        precomputed_stationary = None
        resolved_mult = 1.0 if text_multiplier is None else float(text_multiplier)
        if progressive and self.text_aware_proxy_weighting:
            with torch.no_grad():
                stat_eot, sj_eot, st_eot = objective.component_values(
                    feature.detach(),
                    patch_tokens=views_patches.detach() if views_patches is not None else None,
                    progressive=False,
                )
                if has_purified:
                    stat_p, sj_p, st_p = objective.component_values(
                        purified_feature.detach(),
                        patch_tokens=purified_patches.detach() if purified_patches is not None else None,
                        progressive=False,
                    )
                    denom = 1.0 + self.diffusion_weight
                    stat_u = float(((stat_eot + self.diffusion_weight * stat_p) / denom).detach())
                    stat_j = float(((sj_eot + self.diffusion_weight * sj_p) / denom).detach())
                    stat_t = float(((st_eot + self.diffusion_weight * st_p) / denom).detach())
                else:
                    stat_u = float(stat_eot.detach())
                    stat_j = float(sj_eot.detach())
                    stat_t = float(st_eot.detach())
                precomputed_stationary = (stat_u, stat_j, stat_t)
            if (
                text_multiplier is None
                and self._baseline_joint_values is not None
                and self._baseline_text_values is not None
                and index < len(self._baseline_joint_values)
            ):
                g_joint_i = stat_j - float(self._baseline_joint_values[index])
                g_text_i = stat_t - float(self._baseline_text_values[index])
                lag_i = g_joint_i - g_text_i
                if lag_i > 0.005:
                    resolved_mult = min(1.85, 1.0 + 4.0 * lag_i)

        eot_cat, eot_joint_w, eot_ot_w, eot, eot_joint, eot_text = objective.split_three_component_values(
            feature,
            patch_tokens=views_patches,
            step=step,
            progressive=progressive,
            public_query_indices=pub_idx,
            text_multiplier=resolved_mult,
        )
        if has_purified:
            p_cat, p_joint_w, p_ot_w, pvalue, p_joint, p_text = objective.split_three_component_values(
                purified_feature,
                patch_tokens=purified_patches,
                step=step,
                progressive=progressive,
                public_query_indices=pub_idx,
                text_multiplier=resolved_mult,
            )
            denom = 1.0 + self.diffusion_weight
            val_cat = (eot_cat + self.diffusion_weight * p_cat) / denom
            val_joint = (eot_joint_w + self.diffusion_weight * p_joint_w) / denom
            val_ot = (eot_ot_w + self.diffusion_weight * p_ot_w) / denom
        else:
            denom = 1.0
            val_cat, val_joint, val_ot = eot_cat, eot_joint_w, eot_ot_w
            pvalue, p_joint, p_text = eot, eot_joint, eot_text
        utility = val_cat + val_joint + val_ot

        # 同时计算恒定终端评估标尺值，供 _baseline_values 与退火控制器跨阶段一致比较
        if precomputed_stationary is not None:
            stat_util, stat_joint, stat_text = precomputed_stationary
        elif progressive and bool(getattr(self.config, 'psi_progressive_schedule', False)):
            with torch.no_grad():
                stat_eot, sj_eot, st_eot = objective.component_values(
                    feature.detach(),
                    patch_tokens=views_patches.detach() if views_patches is not None else None,
                    progressive=False,
                )
                if has_purified:
                    stat_p, sj_p, st_p = objective.component_values(
                        purified_feature.detach(),
                        patch_tokens=purified_patches.detach() if purified_patches is not None else None,
                        progressive=False,
                    )
                    stat_util = float(((stat_eot + self.diffusion_weight * stat_p) / denom).detach())
                    stat_joint = float(((sj_eot + self.diffusion_weight * sj_p) / denom).detach())
                    stat_text = float(((st_eot + self.diffusion_weight * st_p) / denom).detach())
                else:
                    stat_util = float(stat_eot.detach())
                    stat_joint = float(sj_eot.detach())
                    stat_text = float(st_eot.detach())
        else:
            stat_util = float(utility.detach())
            stat_joint = (
                float(((eot_joint + self.diffusion_weight * p_joint) / denom).detach())
                if has_purified
                else float(eot_joint.detach())
            )
            stat_text = (
                float(((eot_text + self.diffusion_weight * p_text) / denom).detach())
                if has_purified
                else float(eot_text.detach())
            )

        row = {
            'eot': float(eot.detach()),
            'purified': float(pvalue.detach()),
            'stationary_utility': stat_util,
            'stationary_joint_utility': stat_joint,
            'stationary_text_utility': stat_text,
            'text_multiplier': float(resolved_mult),
            'category_component': float(val_cat.detach()),
            'visual_component': float((val_joint + val_ot).detach()),
            'joint_component': float(val_joint.detach()),
            'ot_component': float(val_ot.detach()),
        }
        return val_cat, val_joint, val_ot, utility, score.mean(0), row

    def _evaluate_proxy_components(
        self,
        index,
        views,
        purified,
        *,
        step: int | None = None,
        progressive: bool = False,
        text_multiplier: float | None = None,
    ):
        val_cat, val_joint, val_ot, utility, score_mean, row = self._evaluate_proxy_split_core(
            index,
            views,
            purified,
            step=step,
            progressive=progressive,
            text_multiplier=text_multiplier,
        )
        return val_cat, val_joint + val_ot, utility, score_mean, row

    def _evaluate_proxy_from_inputs(
        self,
        index,
        views,
        purified,
        *,
        step: int | None = None,
        progressive: bool = False,
        text_multiplier: float | None = None,
    ):
        _, _, utility, score_mean, row = self._evaluate_proxy_components(
            index,
            views,
            purified,
            step=step,
            progressive=progressive,
            text_multiplier=text_multiplier,
        )
        return utility, score_mean, row

    def _maybe_offload_streamed_proxy(self, index: int) -> None:
        if self._stream_surrogates_on_gpu:
            surrogate = self.surrogates[index]
            if callable(getattr(surrogate, 'move_visual_to', None)):
                surrogate.move_visual_to('cpu')

    def _weights_from_values(self, values, *, joint_values=None, text_values=None):
        if self._baseline_values is None or self._baseline_values.shape != values.shape:
            self._baseline_values = values.detach().clone()
        gains = (values - self._baseline_values.to(values.device)).detach()
        effective_gains = gains
        if self.text_aware_proxy_weighting and joint_values is not None and text_values is not None:
            if self._baseline_joint_values is None or self._baseline_joint_values.shape != joint_values.shape:
                self._baseline_joint_values = joint_values.detach().clone()
            if self._baseline_text_values is None or self._baseline_text_values.shape != text_values.shape:
                self._baseline_text_values = text_values.detach().clone()
            g_joint = (joint_values.to(values.device) - self._baseline_joint_values.to(values.device)).detach()
            g_text = (text_values.to(values.device) - self._baseline_text_values.to(values.device)).detach()
            mix = self.proxy_text_bottleneck_mix
            effective_gains = (1.0 - mix) * gains + mix * torch.minimum(g_joint, g_text)
        adaptive = torch.softmax(-effective_gains / max(float(self.ensemble_temperature), 1e-4), dim=0)
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
    def _values(self, image, step=0):
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
        joint_values = torch.tensor(
            [r['stationary_joint_utility'] for r in rows], device=image.device, dtype=values.dtype
        )
        text_values = torch.tensor(
            [r['stationary_text_utility'] for r in rows], device=image.device, dtype=values.dtype
        )
        weights, gains = self._weights_from_values(values, joint_values=joint_values, text_values=text_values)
        return values, torch.stack(scores), weights, rows, gains

    @torch.no_grad()
    def _values_with_components(self, image, step=0):
        values, _, weights, rows, gains = self._values(image, step)
        joint_values = torch.tensor(
            [r['stationary_joint_utility'] for r in rows], device=image.device, dtype=values.dtype
        )
        text_values = torch.tensor(
            [r['stationary_text_utility'] for r in rows], device=image.device, dtype=values.dtype
        )
        return values, joint_values, text_values, weights, gains

    @torch.no_grad()
    def evaluate(self, image, step=0):
        values, scores, weights, rows, gains = self._values(image, step)
        diag: dict[str, Any] = {
            'model_weights': weights.cpu().tolist(),
            'model_utilities': values.cpu().tolist(),
            'model_relative_gains': gains.cpu().tolist(),
        }
        if self.text_aware_proxy_weighting and self._baseline_joint_values is not None and self._baseline_text_values is not None:
            j_vals = torch.tensor([r['stationary_joint_utility'] for r in rows], device=image.device, dtype=values.dtype)
            t_vals = torch.tensor([r['stationary_text_utility'] for r in rows], device=image.device, dtype=values.dtype)
            diag['model_joint_gains'] = (j_vals - self._baseline_joint_values.to(image.device)).cpu().tolist()
            diag['model_text_gains'] = (t_vals - self._baseline_text_values.to(image.device)).cpu().tolist()
        self.last_eval_diagnostics = diag
        return (weights * values).sum(), scores

    def _single_surrogate_image_gradient(self, image, step=1):
        """单代理主导任务 PCGrad 与双分支（EoT vs 扩散净化）正交合并梯度。"""
        base_device = getattr(self.purifier, 'device', self.surrogates[0].device) if self.purifier is not None else self.surrogates[0].device
        leaf_img = image.detach().to(base_device).clone().requires_grad_(True)
        views, purified, views_transformed = self._shared_inputs(leaf_img, step, gradient='full')
        has_purified = bool(self.diffusion_eot_enabled and purified is not None and self.diffusion_weight > 0.0)
        v_leaf = views.detach().requires_grad_(True)
        p_leaf = purified.detach().requires_grad_(True) if has_purified else None
        grad_targets = (v_leaf, p_leaf) if has_purified else (v_leaf,)
        use_prog = bool(getattr(self.config, 'psi_progressive_schedule', False))
        joint_protect = bool(getattr(self.config, 'joint_grad_protect', False))
        vis_cap = float(getattr(self.config, 'single_model_vis_grad_cap', 0.80))
        joint_cap = float(getattr(self.config, 'joint_grad_cap', 0.35))
        conflict_p = False
        conflict_jv = False
        conflict_jp = False
        try:
            if joint_protect:
                val_cat, val_joint, val_ot, utility_tensor, score, row = self._evaluate_proxy_split_core(
                    0, v_leaf, p_leaf, step=step, progressive=use_prog
                )
                active_terms = [
                    (name, tval)
                    for name, tval in (('cat', val_cat), ('ot', val_ot), ('joint', val_joint))
                    if bool(tval.requires_grad)
                ]
                grads_map = {}
                for idx_t, (tname, tval) in enumerate(active_terms):
                    grads_map[tname] = torch.autograd.grad(
                        tval, grad_targets, retain_graph=(idx_t < len(active_terms) - 1)
                    )
                gv_cat = grads_map['cat'][0] if 'cat' in grads_map else torch.zeros_like(v_leaf)
                gv_ot = grads_map['ot'][0] if 'ot' in grads_map else torch.zeros_like(v_leaf)
                gv_j = grads_map['joint'][0] if 'joint' in grads_map else torch.zeros_like(v_leaf)
                gv_cat_ot, conflict_v = single_model_dominant_pcgrad(gv_cat.detach(), gv_ot.detach(), aux_cap=vis_cap)
                gv_j_merged, conflict_jv = single_model_dominant_pcgrad(gv_cat.detach(), gv_j.detach(), aux_cap=joint_cap)
                grad_views = gv_cat_ot + (gv_j_merged - gv_cat.detach())
                if has_purified:
                    gp_cat = grads_map['cat'][1] if 'cat' in grads_map else torch.zeros_like(p_leaf)
                    gp_ot = grads_map['ot'][1] if 'ot' in grads_map else torch.zeros_like(p_leaf)
                    gp_j = grads_map['joint'][1] if 'joint' in grads_map else torch.zeros_like(p_leaf)
                    gp_cat_ot, conflict_p = single_model_dominant_pcgrad(gp_cat.detach(), gp_ot.detach(), aux_cap=vis_cap)
                    gp_j_merged, conflict_jp = single_model_dominant_pcgrad(gp_cat.detach(), gp_j.detach(), aux_cap=joint_cap)
                    grad_purified = gp_cat_ot + (gp_j_merged - gp_cat.detach())
                    del gp_cat, gp_ot, gp_j
                else:
                    grad_purified = None
                del val_cat, val_joint, val_ot, utility_tensor, grads_map, gv_cat, gv_ot, gv_j
            else:
                val_cat, val_vis, utility_tensor, score, row = self._evaluate_proxy_components(
                    0, v_leaf, p_leaf, step=step, progressive=use_prog
                )
                if bool(val_cat.requires_grad) and bool(val_vis.requires_grad):
                    g_cat = torch.autograd.grad(val_cat, grad_targets, retain_graph=True)
                    g_vis = torch.autograd.grad(val_vis, grad_targets, retain_graph=False)
                elif bool(val_cat.requires_grad):
                    g_cat = torch.autograd.grad(val_cat, grad_targets, retain_graph=False)
                    g_vis = tuple(torch.zeros_like(t) for t in grad_targets)
                elif bool(val_vis.requires_grad):
                    g_cat = tuple(torch.zeros_like(t) for t in grad_targets)
                    g_vis = torch.autograd.grad(val_vis, grad_targets, retain_graph=False)
                else:
                    g_cat = tuple(torch.zeros_like(t) for t in grad_targets)
                    g_vis = tuple(torch.zeros_like(t) for t in grad_targets)
                grad_views, conflict_v = single_model_dominant_pcgrad(g_cat[0].detach(), g_vis[0].detach(), aux_cap=vis_cap)
                if has_purified:
                    grad_purified, conflict_p = single_model_dominant_pcgrad(g_cat[1].detach(), g_vis[1].detach(), aux_cap=vis_cap)
                else:
                    grad_purified = None
                del val_cat, val_vis, utility_tensor, g_cat, g_vis
        finally:
            self._maybe_offload_streamed_proxy(0)

        if views_transformed:
            g_img_views = torch.autograd.grad(views, leaf_img, grad_outputs=grad_views, retain_graph=False)[0]
        else:
            g_img_views = grad_views
        if has_purified:
            g_img_pur = torch.autograd.grad(purified, leaf_img, grad_outputs=grad_purified, retain_graph=False)[0]
            gradient, conflict_branch = single_model_dominant_pcgrad(g_img_views, g_img_pur, aux_cap=1.25)
        else:
            gradient, conflict_branch = g_img_views, False
        gradient = gradient.detach().to(image.device)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('单代理任务 PCGrad 目标梯度含非有限值')

        values = torch.tensor([row['stationary_utility']], device=image.device, dtype=torch.float32)
        j_tensor = torch.tensor([row['stationary_joint_utility']], device=image.device, dtype=torch.float32)
        t_tensor = torch.tensor([row['stationary_text_utility']], device=image.device, dtype=torch.float32)
        scores = score.detach().to(image.device).unsqueeze(0)
        weights, gains = self._weights_from_values(values, joint_values=j_tensor, text_values=t_tensor)
        utility = (weights * values).sum().detach()
        eot_on = bool(getattr(self.config, 'eot_enabled', False))
        info: dict[str, Any] = {
            'model_weights': weights.cpu().tolist(),
            'model_utilities': values.cpu().tolist(),
            'model_relative_gains': gains.cpu().tolist(),
            'paths': [row],
            'eot_enabled': eot_on,
            'defense_eot_enabled': eot_on,
            'multiscale_eot_enabled': self.multiscale_eot_enabled,
            'diffusion_eot_enabled': self.diffusion_eot_enabled,
            'purifier_gradient': 'full' if has_purified else 'disabled',
            'pcgrad_applied': bool(conflict_v or conflict_p or conflict_jv or conflict_jp or conflict_branch),
            'single_model_task_pcgrad': True,
            'joint_grad_protect': joint_protect,
            'task_pcgrad_conflicts': {
                'views_task_conflict': bool(conflict_v),
                'purified_task_conflict': bool(conflict_p),
                'views_joint_conflict': bool(conflict_jv),
                'purified_joint_conflict': bool(conflict_jp),
                'branch_conflict': bool(conflict_branch),
            },
        }
        if self.text_aware_proxy_weighting and self._baseline_joint_values is not None and self._baseline_text_values is not None:
            info['model_joint_gains'] = (j_tensor - self._baseline_joint_values.to(image.device)).cpu().tolist()
            info['model_text_gains'] = (t_tensor - self._baseline_text_values.to(image.device)).cpu().tolist()
        return utility, scores.detach(), gradient, info

    def image_gradient(self, image, step=1):
        if len(self.surrogates) == 1 and bool(getattr(self.config, 'single_model_task_pcgrad', False)):
            return self._single_surrogate_image_gradient(image, step=step)

        base_device = getattr(self.purifier, 'device', self.surrogates[0].device) if self.purifier is not None else self.surrogates[0].device
        leaf_img = image.detach().to(base_device).clone().requires_grad_(True)
        views, purified, views_transformed = self._shared_inputs(leaf_img, step, gradient='full')
        has_purified = bool(self.diffusion_eot_enabled and purified is not None and self.diffusion_weight > 0.0)
        v_leaf = views.detach().requires_grad_(True)
        p_leaf = purified.detach().requires_grad_(True) if has_purified else None
        grad_targets = (v_leaf, p_leaf) if has_purified else (v_leaf,)
        stat_values, joint_values, text_values, scores, rows, gv_list, gp_list = [], [], [], [], [], [], []
        use_prog = bool(getattr(self.config, 'psi_progressive_schedule', False))
        for i in range(len(self.surrogates)):
            try:
                value, score, row = self._evaluate_proxy_from_inputs(
                    i, v_leaf, p_leaf, step=step, progressive=use_prog
                )
                grads = torch.autograd.grad(value, grad_targets, retain_graph=False)
            finally:
                self._maybe_offload_streamed_proxy(i)
            stat_val = torch.tensor(row['stationary_utility'], device=image.device, dtype=torch.float32)
            stat_values.append(stat_val)
            joint_values.append(torch.tensor(row['stationary_joint_utility'], device=image.device, dtype=torch.float32))
            text_values.append(torch.tensor(row['stationary_text_utility'], device=image.device, dtype=torch.float32))
            scores.append(score.detach().to(image.device))
            rows.append(row)
            gv_list.append(grads[0].detach())
            if has_purified:
                gp_list.append(grads[1].detach())
            del value, score, grads
        values = torch.stack(stat_values)
        j_tensor = torch.stack(joint_values)
        t_tensor = torch.stack(text_values)
        scores = torch.stack(scores)
        weights, gains = self._weights_from_values(values, joint_values=j_tensor, text_values=t_tensor)
        if self.pcgrad_enabled and len(self.surrogates) > 1:
            grad_views = pcgrad_merge_gradients(gv_list, weights)
            grad_purified = pcgrad_merge_gradients(gp_list, weights) if has_purified else None
        else:
            grad_views = torch.zeros_like(v_leaf)
            grad_purified = torch.zeros_like(p_leaf) if has_purified else None
            for i in range(len(self.surrogates)):
                w_i = float(weights[i])
                grad_views.add_(gv_list[i], alpha=w_i)
                if has_purified:
                    grad_purified.add_(gp_list[i], alpha=w_i)
        if has_purified:
            if views_transformed:
                gradient = torch.autograd.grad((views, purified), leaf_img, grad_outputs=(grad_views, grad_purified))[0]
            else:
                gradient = grad_views + torch.autograd.grad(purified, leaf_img, grad_outputs=grad_purified)[0]
        else:
            if views_transformed:
                gradient = torch.autograd.grad(views, leaf_img, grad_outputs=grad_views)[0]
            else:
                gradient = grad_views
        gradient = gradient.detach().to(image.device)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError('代理目标梯度含非有限值')
        utility = (weights * values).sum().detach()
        eot_on = bool(getattr(self.config, 'eot_enabled', False))
        info: dict[str, Any] = {
            'model_weights': weights.cpu().tolist(),
            'model_utilities': values.cpu().tolist(),
            'model_relative_gains': gains.cpu().tolist(),
            'paths': rows,
            'eot_enabled': eot_on,
            'defense_eot_enabled': eot_on,
            'multiscale_eot_enabled': self.multiscale_eot_enabled,
            'diffusion_eot_enabled': self.diffusion_eot_enabled,
            'purifier_gradient': 'full' if has_purified else 'disabled',
            'pcgrad_applied': bool(self.pcgrad_enabled and len(self.surrogates) > 1),
        }
        if self.text_aware_proxy_weighting and self._baseline_joint_values is not None and self._baseline_text_values is not None:
            info['model_joint_gains'] = (j_tensor - self._baseline_joint_values.to(image.device)).cpu().tolist()
            info['model_text_gains'] = (t_tensor - self._baseline_text_values.to(image.device)).cpu().tolist()
        return utility, scores.detach(), gradient, info
