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


def source_ssim(image: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """七像素均匀窗口、无偏样本协方差的结构相似度 (SSIM)。"""
    img_d = image.double()
    src_d = source.double()
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
    # 1. 主体微光影平滑与幅度惩罚
    fg_denom = max(config.latent_levels, 1) / 255.0
    fg_unit = ((image - anchor) * fg_mask_t) / fg_denom
    fg_tv = (
        (fg_unit[..., 1:, :] - fg_unit[..., :-1, :]).abs().mean()
        + (fg_unit[..., :, 1:] - fg_unit[..., :, :-1]).abs().mean()
    )
    fg_l2 = fg_unit.square().sum() / fg_mask_t.sum().clamp_min(1.0) / 3.0

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

    # 3. 越界惩罚
    if raw_image is not None:
        overflow = F.relu(raw_image - 1.0).square().mean() + F.relu(-raw_image).square().mean()
    else:
        overflow = image.sum() * 0.0

    penalty = (
        config.fg_l2_weight * fg_l2
        + config.fg_tv_weight * fg_tv
        + config.bg_l2_weight * bg_l2
        + config.bg_tv_weight * bg_tv
        + config.bg_chroma_tv_weight * bg_chroma_tv
        + config.overflow_weight * overflow
    )

    metrics = {
        "fg_l2": float(fg_l2.detach()),
        "fg_tv": float(fg_tv.detach()),
        "bg_l2": float(bg_l2.detach()),
        "bg_tv": float(bg_tv.detach()),
        "bg_chroma_tv": float(bg_chroma_tv.detach()),
        "overflow": float(overflow.detach()),
        "total_penalty": float(penalty.detach()),
    }
    return penalty, metrics
