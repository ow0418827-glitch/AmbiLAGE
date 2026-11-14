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

        # 1. 计算背景与主体无截断可行区间 [r_min, r_max]（保留兼容属性，内部使用零中心可行投影）
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

        # 1. 商品主体微光影分支：大核平滑 + 低色度保留率 + 零中心可行投影
        if self.config.latent_levels > 0:
            fg_luma = smooth_sep(diff_luma, self.fg_luma_kernel, groups=1)
            fg_chroma = diff_chroma * self.config.fg_chroma_keep
            fg_chroma = smooth_sep(fg_chroma, self.fg_chroma_kernel, groups=3)
            fg_field = fg_luma + fg_chroma
            res_fg = self._feasible_bounded_residual(
                fg_field / self.tau_fg, self.eps_fg_map, self.fg_min, self.fg_max
            ) * self.fg_mask_t
        else:
            res_fg = torch.zeros_like(self.anchor)

        # 2. 背景多尺度环境光协同分支：大核平滑 + 极低色度保留率 + 零中心可行投影
        bg_luma = smooth_sep(diff_luma, self.bg_luma_kernel, groups=1)
        bg_chroma = diff_chroma * self.config.bg_chroma_keep
        bg_chroma = smooth_sep(bg_chroma, self.bg_chroma_kernel, groups=3)
        v_bg = bg_luma + bg_chroma
        a_bg = self.ambient_field()
        bg_drive = a_bg + self.w_bg_mod * v_bg
        res_bg = self._feasible_bounded_residual(
            bg_drive / self.tau_bg, self.eps_bg_map, self.r_min, self.r_max
        ) * self.bg_mask_t

        # 3. 印刷文字区域绝对清零 (change[roi] == 0)
        return (res_fg + res_bg) * self.support_mask
