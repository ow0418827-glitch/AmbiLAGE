"""Stage 4 优化目标函数、空间方差正则化与切空间投影守门 (Objectives & Tangent Projection)。

实现内容：
1. 多模态联合检索效用函数与解析补偿目标 (PublicFullKeyObjective)；
2. 空间方差正则化 (Spatial Variance Regularization)：
   - 主体微光影：TV 平滑与 L2 惩罚；
   - 背景环境光：仅惩罚围绕均值的空间方差 Var(R_bg) 与空间/色度全变分 TV，
     绝不惩罚整体色调均值平移 (NOT penalizing mean shift)，从而自然允许全局环境光色调生成；
3. 正交切空间投影 (Tangent Space Projection)：
   - 在高维潜空间中，将梯度更新步长投影到 SSIM 与 MAE 约束超平面的可行切空间；
4. 回退守门机制 (Backtracking Line Search Gate)：
   - 保证 SSIM >= 0.78, MAE <= 20.0 严格单调守门。
"""
from __future__ import annotations

import copy
import hashlib
import itertools
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.visual.encoders.joint import ChunkedClipEncoder
from src.visual.robust_objectives import compensated_targets, softmin
from .config import AmbiLAGEConfig


def srgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """可微 sRGB -> OKLab 色彩空间转换（输入值域 [0, 1]，形状 [B, 3, H, W] 或 [3, H, W]）。

    在极暗区间 (lms <= 1e-6) 采用 C0 连续线性延拓，确保 0 像素反传梯度有限无 NaN/Inf，
    正常亮度区与标准 Björn Ottosson OKLab 往返误差 <= 1e-5。
    """
    squeeze = rgb.ndim == 3
    x = rgb.unsqueeze(0) if squeeze else rgb
    lin = torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x.clamp_min(0.04045) + 0.055) / 1.055).pow(2.4),
    )
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    def _safe_cbrt(v: torch.Tensor) -> torch.Tensor:
        return torch.where(v > 1e-6, v.clamp_min(1e-6).pow(1.0 / 3.0), v * 1e4)

    l_ = _safe_cbrt(l)
    m_ = _safe_cbrt(m)
    s_ = _safe_cbrt(s)

    big_l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    comp_a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    comp_b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    out = torch.cat([big_l, comp_a, comp_b], dim=1)
    return out.squeeze(0) if squeeze else out


def oklab_to_srgb(
    lab: torch.Tensor,
    clamp: bool = True,
    gamut_preserve_luma: bool = False,
) -> torch.Tensor:
    """可微 OKLab -> sRGB 色彩空间逆变换（形状 [B, 3, H, W] 或 [3, H, W]）。"""
    squeeze = lab.ndim == 3
    x = lab.unsqueeze(0) if squeeze else lab
    big_l, comp_a, comp_b = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    l_ = big_l + 0.3963377774 * comp_a + 0.2158037573 * comp_b
    m_ = big_l - 0.1055613458 * comp_a - 0.0638541728 * comp_b
    s_ = big_l - 0.0894841775 * comp_a - 1.2914855480 * comp_b

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    lin = torch.cat([r, g, b], dim=1)

    if gamut_preserve_luma:
        # 若固定亮度换色导致线性 RGB 越出色域 [0, 1]，沿等亮度中性灰点 L^3 软压缩色度，防止截断偏暗
        lin_neutral = big_l.clamp(0.0, 1.0).pow(3)
        t_low = torch.where(
            lin < lin_neutral,
            lin_neutral / (lin_neutral - lin).clamp_min(1e-6),
            torch.ones_like(lin),
        )
        t_high = torch.where(
            lin > lin_neutral,
            (1.0 - lin_neutral) / (lin - lin_neutral).clamp_min(1e-6),
            torch.ones_like(lin),
        )
        t = torch.minimum(t_low, t_high).min(dim=1, keepdim=True).values.clamp(0.0, 1.0)
        lin = lin_neutral + t * (lin - lin_neutral)

    srgb = torch.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * (lin.clamp_min(0.0031308).pow(1.0 / 2.4)) - 0.055,
    )
    if clamp:
        srgb = srgb.clamp(0.0, 1.0)
    return srgb.squeeze(0) if squeeze else srgb


def extract_oklab_luma(image: torch.Tensor) -> torch.Tensor:
    """提取图像的 OKLab 感知亮度通道 L（值域近似 [0, 1]，形状 [B, 1, H, W]）。"""
    if image.ndim == 4 and image.shape[1] == 1:
        return image
    if image.ndim == 3 and image.shape[0] == 1:
        return image
    lab = srgb_to_oklab(image)
    return lab[:, 0:1] if lab.ndim == 4 else lab[0:1]


def source_luma_mae_levels(image: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """计算 OKLab 感知亮度通道 L 的平均绝对误差（折算 [0, 255] 等价量纲）。"""
    luma_img = extract_oklab_luma(image)
    luma_src = extract_oklab_luma(source)
    return (luma_img - luma_src).abs().mean() * 255.0


def compute_fg_chroma_tv_levels(
    image: torch.Tensor,
    reference: torch.Tensor,
    fg_mask_t: torch.Tensor,
) -> torch.Tensor:
    """计算前景内部相对参考图的 OKLab 色度 (a, b) 变化全变分平滑度（[0, 255] 等价量纲）。"""
    if float(fg_mask_t.sum().detach()) <= 0.0:
        return image.sum() * 0.0
    ab_diff = (srgb_to_oklab(image)[:, 1:3] - srgb_to_oklab(reference)[:, 1:3]) * 255.0
    mask_h = fg_mask_t[..., 1:, :] * fg_mask_t[..., :-1, :]
    mask_w = fg_mask_t[..., :, 1:] * fg_mask_t[..., :, :-1]
    denom_h = mask_h.sum().clamp_min(1.0)
    denom_w = mask_w.sum().clamp_min(1.0)
    tv_h = ((ab_diff[..., 1:, :] - ab_diff[..., :-1, :]).abs() * mask_h).sum() / (2.0 * denom_h)
    tv_w = ((ab_diff[..., :, 1:] - ab_diff[..., :, :-1]).abs() * mask_w).sum() / (2.0 * denom_w)
    return tv_h + tv_w


def source_ssim(image: torch.Tensor, source: torch.Tensor, luma_only: bool = True) -> torch.Tensor:
    """七像素均匀窗口、无偏样本协方差的结构相似度 (SSIM)，默认仅在 OKLab 亮度 L 上考核。"""
    img_in = extract_oklab_luma(image) if luma_only else image
    src_in = extract_oklab_luma(source) if luma_only else source
    img_d = img_in.double()
    src_d = src_in.double()
    mu_x = F.avg_pool2d(img_d, 7, stride=1)
    mu_y = F.avg_pool2d(src_d, 7, stride=1)
    cov_scale = 49.0 / 48.0
    vx = cov_scale * (F.avg_pool2d(img_d.square(), 7, stride=1) - mu_x.square())
    vy = cov_scale * (F.avg_pool2d(src_d.square(), 7, stride=1) - mu_y.square())
    vxy = cov_scale * (F.avg_pool2d(img_d * src_d, 7, stride=1) - mu_x * mu_y)
    
    num = (2 * mu_x * mu_y + 0.01 ** 2) * (2 * vxy + 0.03 ** 2)
    den = (mu_x.square() + mu_y.square() + 0.01 ** 2) * (vx + vy + 0.03 ** 2)
    return (num / den).mean()


def tangent_projection(
    direction: torch.Tensor,
    normals: Sequence[torch.Tensor],
    slacks: Sequence[float],
) -> torch.Tensor:
    """质量线性约束的活跃集正交切空间投影。
    
    求解 min ||d - p||^2 / 2 满足 a_i · d >= -slack_i / 2。
    使用双精度 Gram 矩阵求解，遇到退化情况返回安全零步长。
    """
    if not normals:
        return direction
    shape = direction.shape
    a = torch.stack([n.detach().flatten() for n in normals])
    norms = a.norm(dim=-1)
    valid = norms > 1e-12
    if not valid.any():
        return direction
    a = a[valid] / norms[valid, None]
    slacks_t = direction.new_tensor(slacks)[valid].clamp_min(0)
    lower = -0.5 * slacks_t / norms[valid]

    gram = (a.double() @ a.double().T).cpu()
    current = (a.double() @ direction.flatten().double()).cpu()
    rhs = lower.double().cpu()

    best_cost = float("inf")
    best = None
    num_constraints = len(a)

    for count in range(num_constraints + 1):
        for active in itertools.combinations(range(num_constraints), count):
            if not active:
                coeffs = torch.zeros(num_constraints, dtype=torch.float64)
            else:
                idx = list(active)
                g = gram[idx][:, idx]
                value = torch.linalg.pinv(g, rtol=1e-10) @ (rhs - current)[idx]
                if (value < -1e-8).any():
                    continue
                coeffs = torch.zeros(num_constraints, dtype=torch.float64)
                coeffs[idx] = value.clamp_min(0.0)
            if (current + gram @ coeffs < rhs - 1e-8).any():
                continue
            cost = float(coeffs @ gram @ coeffs)
            if cost < best_cost:
                best_cost, best = cost, coeffs

    if best is None:
        return torch.zeros_like(direction)
    result = direction.flatten() + best.to(a) @ a
    return result.reshape(shape)


class PublicFullKeyObjective:
    """自身公开检索键与图像各半，查询为目标图与目标特点各半的多模态检索效用目标。"""

    def __init__(
        self,
        surrogate: Any,
        target_feature: torch.Tensor,
        source_public_key: str,
        training_queries: Sequence[str],
    ):
        if len(training_queries) != 1 or not isinstance(training_queries[0], str) or not training_queries[0].strip():
            raise ValueError("AmbiLAGE 优化目标严格绑定一条目标特点训练正文")
        
        self.surrogate = surrogate
        self.device = surrogate.device

        # 使用适配器编码公开文本键与目标特点正文
        from types import SimpleNamespace
        encoder = ChunkedClipEncoder(SimpleNamespace(
            surrogate=surrogate, device=surrogate.device, snapshot=surrogate.snapshot()
        ))
        own_vec = encoder.encode_text(source_public_key)
        own_audit = copy.deepcopy(encoder.last_encoding_audit)
        text_vec = encoder.encode_text(training_queries[0])
        text_audit = copy.deepcopy(encoder.last_encoding_audit)

        self.title_feature = torch.as_tensor(own_vec, device=self.device, dtype=torch.float32).reshape(1, -1).detach()
        self.training_text_feature = torch.as_tensor(text_vec, device=self.device, dtype=torch.float32).reshape(1, -1).detach()
        self.target_feature = F.normalize(target_feature.detach().float(), dim=-1)

        # 联合查询向量: 目标图像特征与目标特点文本特征各半归一化
        self.query_features = F.normalize(self.target_feature + self.training_text_feature, dim=-1).detach()

        # 解析补偿教师向量
        teachers, valid = compensated_targets(self.query_features, self.title_feature)
        if not bool(valid.all()):
            raise ValueError("训练联合查询与自身公开全文键不在正半球，解析补偿教师计算无效")
        self.compensated_features = teachers.detach()

        self.audit = {
            "algorithm": "ambilage_public_fullkey_joint_and_analytic_compensated_image_v1",
            "source_public_key_text": source_public_key,
            "source_public_key_sha256": hashlib.sha256(source_public_key.encode("utf-8")).hexdigest(),
            "source_public_key_audit": own_audit,
            "training_query_audit": text_audit,
            "training_queries": list(training_queries),
            "query_image_text_weight": [0.5, 0.5],
            "key_image_text_weight": [0.5, 0.5],
            "joint_compensated_objective_weight": [0.5, 0.5],
            "private_gallery_used": False,
            "victim_memory_used": False,
            "formal_queries_used": False,
        }

    def scores(self, features: torch.Tensor) -> torch.Tensor:
        """联合检索余弦打分：图像与自身公开文本键归一化求和后与查询点乘。"""
        key = F.normalize(F.normalize(features.float(), dim=-1) + self.title_feature, dim=-1)
        return key @ self.query_features.T

    def compensated_scores(self, features: torch.Tensor) -> torch.Tensor:
        """解析补偿特征余弦打分。"""
        return F.normalize(features.float(), dim=-1) @ self.compensated_features.T

    def compute_utility(
        self,
        features: torch.Tensor,
        config: AmbiLAGEConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算检索效用 (结合 Softmin 保障最低召回条件)。"""
        s_joint = self.scores(features)
        u_joint = 0.5 * s_joint.mean() + 0.5 * softmin(s_joint.flatten(), config.temperature)
        s_comp = self.compensated_scores(features)
        u_comp = 0.5 * s_comp.mean() + 0.5 * softmin(s_comp.flatten(), config.temperature)
        total_utility = config.joint_weight * u_joint + config.compensated_weight * u_comp
        return total_utility, s_joint

    def diagnostics(self, features: torch.Tensor) -> dict[str, Any]:
        feat_norm = F.normalize(features.float(), dim=-1)
        return {
            "joint_scores": self.scores(feat_norm).detach().cpu().tolist(),
            "compensated_image_cosine": self.compensated_scores(feat_norm).detach().cpu().tolist(),
            "image_to_target_cosine": (feat_norm @ self.target_feature.T).detach().cpu().tolist(),
            "image_to_training_text_cosine": (feat_norm @ self.training_text_feature.T).detach().cpu().tolist(),
            "image_to_query_cosine": (feat_norm @ self.query_features.T).detach().cpu().tolist(),
        }


def differentiable_halo(
    image: torch.Tensor,
    reference: torch.Tensor,
    fg_mask_t: torch.Tensor,
    bg_mask_t: torch.Tensor,
    luma_only: bool = True,
) -> torch.Tensor:
    """可微前景轮廓边界光晕惩罚：度量前景外/内邻域环带上的局部亮度阶跃幅度。

    默认在 OKLab 亮度 L 上度量轮廓光晕，避免前景合法换色与背景色温差产生虚假 RGB 阶跃；
    全图均匀平移时局部差分严格为零。
    """
    if fg_mask_t.sum() <= 0 or bg_mask_t.sum() <= 0:
        return image.sum() * 0.0
    outer_ring = (F.max_pool2d(fg_mask_t, 5, stride=1, padding=2) - fg_mask_t).clamp(0.0, 1.0) * bg_mask_t
    inner_ring = (fg_mask_t - (1.0 - F.max_pool2d(1.0 - fg_mask_t, 5, stride=1, padding=2))).clamp(0.0, 1.0) * fg_mask_t
    ring_mask = (outer_ring + inner_ring).clamp(0.0, 1.0)
    ring_count = ring_mask.sum()
    if float(ring_count.detach()) <= 0.0:
        return image.sum() * 0.0
    if luma_only:
        diff = (extract_oklab_luma(image) - extract_oklab_luma(reference)) * 255.0
    else:
        diff = (image - reference) * 255.0
    # 与 5x5 局部邻域均值的偏差即为高频边缘光晕阶跃，均匀平移时该项严格为 0
    local_mean = F.avg_pool2d(diff, 5, stride=1, padding=2)
    halo_jump = (diff - local_mean) * ring_mask
    return torch.sqrt(halo_jump.square().sum(dim=1, keepdim=True) + 1e-8).sum() / ring_count.clamp_min(1.0)


def compute_boundary_halo_error(
    source_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    fg_mask: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
) -> float:
    """与可微光晕一致的数学门禁边界光晕误差度量（[0, 255] 灰度级）。"""
    src = np.asarray(source_rgb, dtype=np.float32)
    cand = np.asarray(candidate_rgb, dtype=np.float32)
    if src.shape != cand.shape or src.ndim != 3:
        raise ValueError("光晕检测输入图像尺寸不一致")
    h, w = src.shape[:2]
    if fg_mask is None:
        return 0.0
    fg = np.asarray(fg_mask, dtype=bool)
    if roi_mask is not None:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape == fg.shape:
            fg = fg | roi
    bg = ~fg
    if not fg.any() or not bg.any():
        return 0.0
    fg_t = torch.as_tensor(fg, dtype=torch.float32)[None, None]
    bg_t = torch.as_tensor(bg, dtype=torch.float32)[None, None]
    cand_t = torch.as_tensor(cand.transpose(2, 0, 1).copy(), dtype=torch.float32)[None] / 255.0
    src_t = torch.as_tensor(src.transpose(2, 0, 1).copy(), dtype=torch.float32)[None] / 255.0
    with torch.no_grad():
        val = differentiable_halo(cand_t, src_t, fg_t, bg_t)
    return float(val.item())


def compute_spatial_variance_penalty(
    image: torch.Tensor,
    anchor: torch.Tensor,
    raw_image: torch.Tensor | None,
    fg_mask_t: torch.Tensor,
    bg_mask_t: torch.Tensor,
    config: AmbiLAGEConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算空间方差与平滑正则项损失。
    
    关键数理特性：
    背景正则项通过 (bg_unit - bg_mean) 仅惩罚背景像素偏离自身均值的空间波动方差，
    绝不惩罚整体背景环境光均值向目标色温平移！
    """
    # 1. 主体亮度 L 微光影平滑与幅度惩罚 + 色度 (a, b) 变化平滑 TV 惩罚（不惩罚合法主体均匀换色）
    fg_luma_scale = max(float(getattr(config, "fg_luma_scale", 1.0)), 1e-4)
    fg_denom = max(float(config.latent_levels) * fg_luma_scale, 1.0) / 255.0
    fg_luma_diff = extract_oklab_luma(image) - extract_oklab_luma(anchor)
    fg_unit = (fg_luma_diff * fg_mask_t) / fg_denom
    fg_tv = (
        (fg_unit[..., 1:, :] - fg_unit[..., :-1, :]).abs().mean()
        + (fg_unit[..., :, 1:] - fg_unit[..., :, :-1]).abs().mean()
    )
    fg_l2 = fg_unit.square().sum() / fg_mask_t.sum().clamp_min(1.0)
    fg_chroma_denom = max(float(getattr(config, "fg_chroma_levels", 64)), 1.0)
    fg_chroma_tv = compute_fg_chroma_tv_levels(image, anchor, fg_mask_t) / fg_chroma_denom

    # 2. 背景空间平滑与色度平滑 (仅惩罚方差与 TV，不惩罚整体均值平移)
    bg_denom = max(config.background_levels, 1) / 255.0
    bg_unit = ((image - anchor) * bg_mask_t) / bg_denom
    bg_mean = bg_unit.sum(dim=(-2, -1), keepdim=True) / bg_mask_t.sum().clamp_min(1.0)
    bg_dev = (bg_unit - bg_mean) * bg_mask_t
    bg_tv = (
        (bg_unit[..., 1:, :] - bg_unit[..., :-1, :]).abs().mean()
        + (bg_unit[..., :, 1:] - bg_unit[..., :, :-1]).abs().mean()
    )
    bg_chroma = bg_dev - bg_dev.mean(dim=1, keepdim=True)
    bg_chroma_tv = (
        (bg_chroma[..., 1:, :] - bg_chroma[..., :-1, :]).abs().mean()
        + (bg_chroma[..., :, 1:] - bg_chroma[..., :, :-1]).abs().mean()
    )
    bg_l2 = bg_dev.square().sum() / bg_mask_t.sum().clamp_min(1.0) / 3.0

    # 3. 边界光晕可微惩罚与越界惩罚
    halo_loss = differentiable_halo(image, anchor, fg_mask_t, bg_mask_t)
    if raw_image is not None:
        overflow = F.relu(raw_image - 1.0).square().mean() + F.relu(-raw_image).square().mean()
    else:
        overflow = image.sum() * 0.0

    halo_w = float(getattr(config, "halo_weight", 0.0))
    fg_chroma_tv_w = float(getattr(config, "fg_chroma_tv_weight", 0.0))
    penalty = (
        config.fg_l2_weight * fg_l2
        + config.fg_tv_weight * fg_tv
        + fg_chroma_tv_w * fg_chroma_tv
        + config.bg_l2_weight * bg_l2
        + config.bg_tv_weight * bg_tv
        + config.bg_chroma_tv_weight * bg_chroma_tv
        + halo_w * halo_loss
        + config.overflow_weight * overflow
    )

    metrics = {
        "fg_l2": float(fg_l2.detach()),
        "fg_tv": float(fg_tv.detach()),
        "fg_chroma_tv": float(fg_chroma_tv.detach()),
        "bg_l2": float(bg_l2.detach()),
        "bg_tv": float(bg_tv.detach()),
        "bg_chroma_tv": float(bg_chroma_tv.detach()),
        "halo_loss": float(halo_loss.detach()),
        "overflow": float(overflow.detach()),
        "total_penalty": float(penalty.detach()),
    }
    return penalty, metrics
