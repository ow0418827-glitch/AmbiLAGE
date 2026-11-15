"""Stage 4 数理核心：环境光多尺度协同光场与主体微光影建模 (Ambient Cooperative Field)。

数学原理与创新保证：
1. 零中心可行无截断边界投影 [r_min, r_max]：
   r_min(x, y) = max(-eps, -I_anchor(x, y))
   r_max(x, y) = min(eps, 1.0 - I_anchor(x, y))
   raw_res(x, y) = eps * tanh(drive / tau)
   res(x, y) = clamp(raw_res, r_min, r_max)
   零驱动严格对应零残差（避免纯白背景发灰、暗部泛白及轮廓断层暗环），
   任何驱动值均严格收敛于 [r_min, r_max] 内部，数理保证无 255 截断越界。
2. 全局环境底色 a_global ∈ R^(1x3x1x1) 与低频漫射网格 a_grid ∈ R^(1x3x8x8)（大核平滑 sigma=14.0）；
3. 主体微光影：eps_fg <= 6/255，保留 16 通道细腻微光影质感；背景明度 sigma=12.0、色度 sigma=16.0、色度保留率 0.10 锁死高频杂色；
4. 印刷墨迹核心绝对锁定：change[roi] == 0。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AmbiLAGEConfig
from .objectives import oklab_to_srgb, srgb_to_oklab


def make_sep_gaussian(
    sigma: float,
    device: torch.device | str,
    channels: int,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """构建能量守恒且可分离的 1D 水平与垂直高斯平滑滤波核 (截断半径 3*sigma)。"""
    if sigma <= 0.0:
        return None
    radius = int(np.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    kh = g.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kv = g.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    return kh, kv, radius


def smooth_sep(
    x: torch.Tensor,
    kernel_tuple: tuple[torch.Tensor, torch.Tensor, int] | None,
    groups: int,
) -> torch.Tensor:
    """使用反射填充 (reflect pad) 消除边界伪影的可分离高斯卷积。
    自适应小尺寸测试图像，当半径大于等于图像维度时使用 replicate 填充防止越界。
    """
    if kernel_tuple is None:
        return x
    kh, kv, radius = kernel_tuple
    h, w = x.shape[-2:]
    if radius >= min(h, w):
        padded = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    else:
        padded = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    out = F.conv2d(padded, kh, groups=groups)
    return F.conv2d(out, kv, groups=groups)


class AmbientCoopField(nn.Module):
    """环境光协同背景调整与主体微光影参数化光场。"""

    def __init__(
        self,
        anchor: torch.Tensor,
        eps_fg_map: torch.Tensor,
        eps_bg_map: torch.Tensor,
        w_bg_mod: torch.Tensor,
        fg_mask_t: torch.Tensor,
        bg_mask_t: torch.Tensor,
        support_mask: torch.Tensor,
        config: AmbiLAGEConfig,
        w_fg_recolor: torch.Tensor | None = None,
    ):
        super().__init__()
        self.anchor = anchor
        self.eps_fg_map = eps_fg_map
        self.eps_bg_map = eps_bg_map
        self.w_bg_mod = w_bg_mod
        self.fg_mask_t = fg_mask_t
        self.bg_mask_t = bg_mask_t
        self.support_mask = support_mask
        self.config = config
        self.w_fg_recolor = w_fg_recolor
        self.device = anchor.device

        # 1. 计算背景与主体无截断可行区间 [r_min, r_max]
        self.r_min = torch.maximum(-eps_bg_map, -anchor)
        self.r_max = torch.minimum(eps_bg_map, 1.0 - anchor)
        self.bg_mid = (0.5 * (self.r_min + self.r_max)).detach()
        self.bg_half = (0.5 * (self.r_max - self.r_min)).detach()
        self.fg_min = torch.maximum(-eps_fg_map, -anchor).detach()
        self.fg_max = torch.minimum(eps_fg_map, 1.0 - anchor).detach()
        self.fg_mid = ((self.fg_min + self.fg_max) * 0.5).detach()
        self.fg_half = ((self.fg_max - self.fg_min) * 0.5).detach()
        self.tau_fg = max(float(getattr(config, "tau_fg", 0.20)), 1e-4)
        self.tau_bg = max(float(getattr(config, "tau_bg", 0.35)), 1e-4)
        self.fg_luma_scale = max(float(getattr(config, "fg_luma_scale", 0.40)), 0.0)
        self.fg_chroma_limit = max(float(getattr(config, "fg_chroma_limit", 0.16)), 0.0)

        with torch.no_grad():
            self.anchor_lab = srgb_to_oklab(self.anchor).detach()
            self.anchor_oklab_rebuilt = oklab_to_srgb(
                self.anchor_lab, clamp=True, gamut_preserve_luma=True
            ).detach()
        fg_chroma_levels = float(getattr(config, "fg_chroma_levels", 64))
        if w_fg_recolor is not None:
            self.fg_chroma_eps_map = (fg_chroma_levels / 255.0) * w_fg_recolor
        else:
            self.fg_chroma_eps_map = torch.zeros_like(eps_fg_map)

        # 2. 预计算常驻 GPU 的可分离高斯平滑核与低频径向暗角轮廓
        self.fg_luma_kernel = make_sep_gaussian(config.fg_luma_smooth_sigma, self.device, 1)
        self.fg_chroma_kernel = make_sep_gaussian(config.fg_chroma_smooth_sigma, self.device, 3)
        self.bg_luma_kernel = make_sep_gaussian(config.bg_luma_smooth_sigma, self.device, 1)
        self.bg_chroma_kernel = make_sep_gaussian(config.bg_chroma_smooth_sigma, self.device, 3)
        self.ambient_kernel = make_sep_gaussian(config.ambient_smooth_sigma, self.device, 3)

        h, w = anchor.shape[-2:]
        yy = torch.linspace(-1.0, 1.0, h, device=self.device, dtype=torch.float32)
        xx = torch.linspace(-1.0, 1.0, w, device=self.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        r2_norm = (gy * gy + gx * gx) * 0.5
        self.radial_profile = (1.0 - r2_norm).clamp(0.0, 1.0)[None, None].detach()

        # 3. 可学习参数：全局三维背景色向量 bg_color_shift ([1,3,1,1]) + 单标量径向暗角系数 ([1,1,1,1]) + 前景全局 OKLab 色度向量 ([1,2,1,1])
        init_tau = 0.5 * (config.background_levels / 255.0) * config.tau_scale
        init_val = float(config.ambient_init_bias * init_tau)
        self.ambient_global = nn.Parameter(
            torch.full((1, 3, 1, 1), init_val, device=self.device, dtype=torch.float32)
        )
        self.bg_color_shift = self.ambient_global
        self.ambient_grid = nn.Parameter(
            torch.zeros((1, 1, 1, 1), device=self.device, dtype=torch.float32)
        )
        self.fg_chroma_shift = nn.Parameter(
            torch.zeros((1, 2, 1, 1), device=self.device, dtype=torch.float32)
        )

    def ambient_field(self) -> torch.Tensor:
        """合成全局三维背景颜色与极低频 1 自由度径向暗角光场（无空间杂色自由度）。"""
        vig_scale = float(getattr(self.config, "bg_radial_vignette_scale", 0.05))
        return self.bg_color_shift + vig_scale * self.ambient_grid * self.radial_profile

    @staticmethod
    def _feasible_bounded_residual(
        drive: torch.Tensor,
        eps_map: torch.Tensor,
        r_min: torch.Tensor,
        r_max: torch.Tensor,
    ) -> torch.Tensor:
        """零中心平滑映射后再做 [r_min, r_max] 无截断约束：drive==0 时严格返回 0。"""
        raw_res = eps_map * torch.tanh(drive)
        return torch.clamp(raw_res, min=r_min, max=r_max)

    def compute_residual(self, diff: torch.Tensor) -> torch.Tensor:
        """从 VAE 潜差分或特征差分场合成复合残差。
        
        输入 diff: [1, 3, H, W] 连续解码差分场
        输出: [1, 3, H, W] 复合加性残差 (严格保证文字保护区为 0，且无 255 截断越界)
        """
        diff_luma = diff.mean(dim=1, keepdim=True)
        diff_chroma = diff - diff_luma

        # 1. 商品主体分支：
        #    - 若提供双重掩膜 w_fg_recolor：在 OKLab 空间执行“锁亮度 L、放色度 (a, b)”换色；
        #    - 兼容传统 8 参数调用 (w_fg_recolor=None)：保持受 eps_fg_map 严格约束的微光影分支。
        if self.w_fg_recolor is not None and int(getattr(self.config, "fg_chroma_levels", 0)) > 0:
            if self.config.latent_levels > 0 and self.fg_luma_scale > 0.0:
                fg_luma_drive = smooth_sep(diff_luma, self.fg_luma_kernel, groups=1)
                eps_luma = self.eps_fg_map * self.fg_luma_scale
                d_luma = eps_luma * torch.tanh(fg_luma_drive / self.tau_fg)
            else:
                d_luma = torch.zeros_like(diff_luma)

            fg_chroma_rgb = smooth_sep(diff_chroma, self.fg_chroma_kernel, groups=3)
            vae_a = 0.22 * (fg_chroma_rgb[:, 0:1] - fg_chroma_rgb[:, 1:2])
            vae_b = 0.22 * (0.5 * (fg_chroma_rgb[:, 0:1] + fg_chroma_rgb[:, 1:2]) - fg_chroma_rgb[:, 2:3])
            ab_drive = self.fg_chroma_shift + torch.cat([vae_a, vae_b], dim=1)
            d_ab = self.fg_chroma_eps_map * torch.tanh(ab_drive / 0.12)

            lab_new = torch.cat(
                [self.anchor_lab[:, 0:1] + d_luma, self.anchor_lab[:, 1:3] + d_ab],
                dim=1,
            )
            recolored = oklab_to_srgb(lab_new, clamp=True, gamut_preserve_luma=True)
            res_fg_raw = recolored - self.anchor_oklab_rebuilt
            res_fg = torch.clamp(res_fg_raw, min=-self.anchor, max=1.0 - self.anchor) * self.fg_mask_t
        elif self.config.latent_levels > 0:
            fg_luma = smooth_sep(diff_luma, self.fg_luma_kernel, groups=1)
            fg_chroma = diff_chroma * self.config.fg_chroma_keep
            fg_chroma = smooth_sep(fg_chroma, self.fg_chroma_kernel, groups=3)
            fg_field = fg_luma + fg_chroma
            res_fg = self._feasible_bounded_residual(
                fg_field / self.tau_fg, self.eps_fg_map, self.fg_min, self.fg_max
            ) * self.fg_mask_t
        else:
            res_fg = torch.zeros_like(self.anchor)

        # 2. 背景全局三维颜色 + 单自由度径向暗角分支：彻底屏蔽 VAE 潜空间局部差分 diff (v_bg)
        a_bg = self.ambient_field()
        res_bg = self._feasible_bounded_residual(
            a_bg / self.tau_bg, self.eps_bg_map, self.r_min, self.r_max
        ) * self.bg_mask_t

        # 3. 印刷文字区域绝对清零 (change[roi] == 0)
        return (res_fg + res_bg) * self.support_mask
