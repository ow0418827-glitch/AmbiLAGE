"""Stage 4 数理核心：环境光多尺度协同光场与主体微光影建模 (Ambient Cooperative Field)。

数学原理与创新保证：
1. 可行无截断区间中点对称展开 [r_min, r_max]：
   r_min(x, y) = max(-eps_bg, -I_anchor(x, y))
   r_max(x, y) = min(eps_bg, 1.0 - I_anchor(x, y))
   mid_bg(x, y) = (r_min + r_max) / 2
   half_bg(x, y) = (r_max - r_min) / 2
   tau_bg(x, y) = max(half_bg, 1/255) * tau_scale
   任何背景驱动值经 tanh 变换后，严格收敛于 (r_min, r_max) 内部，
   数理上只保证可行区间内的幅度约束；自然斑块与光晕仍须由视觉门禁实际核验。
2. 全局环境底色 a_global ∈ R^(1x3x1x1)，初始偏置 0.30 * tau_bg，处于最大梯度响应区；
3. 低频漫射网格 a_grid ∈ R^(1x3x8x8)，经双三次插值上采样并施加大核高斯平滑 (sigma=14.0)；
4. 主体微光影：eps_fg <= 6/255，超大尺度高斯平滑 (明度 sigma=5.0, 色度 sigma=9.0, 色度保留率 0.30)，
   维持干净自然的主体质感；
5. 印刷文字区域绝对锁定：change[roi] == 0。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AmbiLAGEConfig


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
        self.device = anchor.device

        # 1. 计算背景无截断可行区间 [r_min, r_max] 及其中心 mid 与半宽 half
        self.r_min = torch.maximum(-eps_bg_map, -anchor)
        self.r_max = torch.minimum(eps_bg_map, 1.0 - anchor)
        self.bg_mid = (0.5 * (self.r_min + self.r_max)).detach()
        self.bg_half = (0.5 * (self.r_max - self.r_min)).detach()
        self.tau_bg = (self.bg_half.clamp_min(1.0 / 255.0) * config.tau_scale).detach()
        fg_min = torch.maximum(-eps_fg_map, -anchor)
        fg_max = torch.minimum(eps_fg_map, 1.0 - anchor)
        self.fg_mid = (fg_min + fg_max) * .5
        self.fg_half = (fg_max - fg_min) * .5
        self.tau_fg = self.fg_half.clamp_min(1. / 255.) * config.tau_scale

        # 2. 预计算常驻 GPU 的可分离高斯平滑核
        self.fg_luma_kernel = make_sep_gaussian(config.fg_luma_smooth_sigma, self.device, 1)
        self.fg_chroma_kernel = make_sep_gaussian(config.fg_chroma_smooth_sigma, self.device, 3)
        self.bg_luma_kernel = make_sep_gaussian(config.bg_luma_smooth_sigma, self.device, 1)
        self.bg_chroma_kernel = make_sep_gaussian(config.bg_chroma_smooth_sigma, self.device, 3)
        self.ambient_kernel = make_sep_gaussian(config.ambient_smooth_sigma, self.device, 3)

        # 3. 可学习环境光变量初始化
        init_tau = 0.5 * (config.background_levels / 255.0) * config.tau_scale
        init_val = float(config.ambient_init_bias * init_tau)
        self.ambient_global = nn.Parameter(
            torch.full((1, 3, 1, 1), init_val, device=self.device, dtype=torch.float32)
        )
        gsize = config.ambient_grid_size
        self.ambient_grid = nn.Parameter(
            torch.zeros((1, 3, gsize, gsize), device=self.device, dtype=torch.float32)
        )

    def ambient_field(self) -> torch.Tensor:
        """合成全局底色与低频漫射网格环境光场。"""
        grid_up = F.interpolate(
            self.ambient_grid,
            size=self.anchor.shape[-2:],
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        if self.ambient_kernel is not None:
            grid_up = smooth_sep(grid_up, self.ambient_kernel, groups=3)
        return self.ambient_global + grid_up

    def compute_residual(self, diff: torch.Tensor) -> torch.Tensor:
        """从 VAE 潜差分或特征差分场合成复合残差。
        
        输入 diff: [1, 3, H, W] 连续解码差分场
        输出: [1, 3, H, W] 复合加性残差 (严格保证文字保护区为 0，且无 255 截断越界)
        """
        diff_luma = diff.mean(dim=1, keepdim=True)
        diff_chroma = diff - diff_luma

        # 1. 商品主体微光影分支：大核平滑 + 低色度保留率 (0.30)
        if self.config.latent_levels > 0:
            fg_luma = smooth_sep(diff_luma, self.fg_luma_kernel, groups=1)
            fg_chroma = diff_chroma * self.config.fg_chroma_keep
            fg_chroma = smooth_sep(fg_chroma, self.fg_chroma_kernel, groups=3)
            fg_field = fg_luma + fg_chroma
            res_fg = (self.fg_mid + self.fg_half * torch.tanh(fg_field / self.tau_fg)) * self.fg_mask_t
        else:
            res_fg = torch.zeros_like(self.anchor)

        # 2. 背景多尺度环境光协同分支：中点对称无截断展开
        bg_luma = smooth_sep(diff_luma, self.bg_luma_kernel, groups=1)
        bg_chroma = diff_chroma * self.config.bg_chroma_keep
        bg_chroma = smooth_sep(bg_chroma, self.bg_chroma_kernel, groups=3)
        v_bg = bg_luma + bg_chroma
        a_bg = self.ambient_field()
        bg_drive = a_bg + self.w_bg_mod * v_bg
        res_bg = (self.bg_mid + self.bg_half * torch.tanh(bg_drive / self.tau_bg)) * self.bg_mask_t

        # 3. 印刷文字区域绝对清零 (change[roi] == 0)
        return (res_fg + res_bg) * self.support_mask
