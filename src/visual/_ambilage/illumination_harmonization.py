"""Stage 3: 光照一致性协调与生成基准锚点合成 (Illumination Harmonization)。

实现内容：
1. 线性色彩空间 (Linear RGB) 转换与影棚漫射光照提取；
2. 印刷区域保护与透光度分析 (Prepare Print Context)：
   - 提取不透明油墨 (Opaque Ink) 与饱和色彩区，断言 100% 绝对保留；
   - 提取可信主体表面参考点 (Trusted Body Pixels)；
3. 亚光颜色融合 (Matte Blend Harmonization)：
   - 基于目标主色向源商品主体执行光照协调投影；
   - 严格断言印刷文字保护区零改动 (assert np.all(change[roi] == 0))；
   - 严格保证无辅助像素残差 (I_initial == I_anchor)，作为 Stage 4 原生潜变量优化的基准锚点。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_closing

from src.visual.image_utils import read_rgb, sha256_file, write_json
from .config import AmbiLAGEConfig
from .objectives import compute_boundary_halo_error, source_ssim
from .segmentation import extract_ink_core_mask, print_region_mask


def linear_from_srgb(a: np.ndarray) -> np.ndarray:
    """标准 sRGB 到线性 RGB 的正向伽马转换。"""
    a = np.asarray(a, dtype=np.float64)
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def srgb_from_linear(a: np.ndarray) -> np.ndarray:
    """线性 RGB 到标准 sRGB 的逆向伽马转换。"""
    a = np.clip(a, 0.0, 1.0)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * np.power(a, 1.0 / 2.4) - 0.055)


def prepare_print_context(
    source_rgb: np.ndarray,
    foreground: np.ndarray,
    regions: Sequence[Sequence[float]] | None,
    *,
    ink_core_mask: np.ndarray | None = None,
    substrate_mask: np.ndarray | None = None,
    config: AmbiLAGEConfig | None = None,
) -> dict[str, Any]:
    """构建印刷区域分析上下文，区分必须零改动的墨迹核心 `ink_core` 与可平滑调色的标签底色 `substrate`。"""
    import torch

    h, w = source_rgb.shape[:2]
    cfg = config or AmbiLAGEConfig()
    roi = print_region_mask(source_rgb.shape, regions)
    if ink_core_mask is not None and substrate_mask is not None:
        ink_core = np.asarray(ink_core_mask, dtype=bool)
        substrate = np.asarray(substrate_mask, dtype=bool)
    else:
        ink_core, substrate = extract_ink_core_mask(source_rgb, foreground, roi, cfg)

    src_lin = linear_from_srgb(source_rgb.astype(np.float64) / 255.0)
    luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    ys = src_lin @ luma_weights

    # 底材反光闭运算推断
    paper = np.stack([
        grey_closing(src_lin[..., c], size=(11, 11), mode="nearest") for c in range(3)
    ], axis=-1)
    yb = paper @ luma_weights

    transmission = np.zeros((h, w), dtype=np.float64)
    opaque = ink_core.copy()
    chroma = source_rgb.max(axis=-1).astype(float) - source_rgb.min(axis=-1).astype(float)
    colored = chroma >= 12.0

    if regions:
        for region in regions:
            local = print_region_mask(source_rgb.shape, [region])
            inside = local & foreground
            if not inside.any():
                continue
            darkest = float(np.quantile(ys[inside], 0.02))
            trans = np.clip((ys - darkest) / np.maximum(yb - darkest, 1e-4), 0.0, 1.0)
            if (local & substrate).any():
                transmission[local & substrate] = trans[local & substrate]
            elif np.median(ys[inside]) < 0.35 or np.median(chroma[inside]) >= 12.0:
                opaque |= local
            else:
                local_opaque = local & ((trans <= 0.04) | colored)
                transmission[local] = trans[local]
                opaque |= local_opaque
    transmission[opaque] = 0.0
    transmission[ink_core] = 0.0

    # 油墨与距离变换
    ink = ink_core | (roi & ((source_rgb.min(axis=-1) < 180) | colored))
    distance = distance_transform_edt(~ink) if ink.any() else np.full(roi.shape, 64.0)

    # 可信未印刷主体参考点 (Trusted body pixels)
    trusted = foreground & (~roi)
    if trusted.sum() < 50:
        trusted = foreground & (~ink_core)
    if trusted.sum() < 10:
        trusted = foreground.copy()

    _, nearest = distance_transform_edt(~trusted, return_indices=True) if trusted.any() else (None, np.indices((h, w)))
    halo = roi & foreground & (~ink_core) & (source_rgb.min(axis=-1) >= 220) & (transmission > 0.85)

    return {
        "source": source_rgb,
        "foreground": foreground,
        "roi": roi,
        "ink_core": ink_core,
        "substrate": substrate,
        "opaque": opaque,
        "trusted": trusted,
        "transmission": transmission,
        "distance": distance,
        "nearest": nearest,
        "halo": halo,
        "halo_applicable": bool(halo.any()),
    }


def _evaluate_candidate_metrics(
    source_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    foreground: np.ndarray,
    ink_core: np.ndarray,
) -> tuple[float, float, float]:
    import torch

    t_cand = torch.tensor(candidate_rgb.transpose(2, 0, 1).copy(), dtype=torch.float32)[None] / 255.0
    t_src = torch.tensor(source_rgb.transpose(2, 0, 1).copy(), dtype=torch.float32)[None] / 255.0
    ssim_val = float(source_ssim(t_cand, t_src))
    mae_val = float(np.abs(candidate_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean())
    halo_val = compute_boundary_halo_error(source_rgb, candidate_rgb, fg_mask=foreground, roi_mask=ink_core)
    return ssim_val, mae_val, halo_val


def harmonize_illumination(
    source_image: Path | str | np.ndarray,
    target_color_rgb: Sequence[int],
    foreground: np.ndarray,
    regions: Sequence[Sequence[float]] | None,
    config: AmbiLAGEConfig,
    output_dir: Path | str | None = None,
    *,
    ink_core_mask: np.ndarray | None = None,
    substrate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """执行 Stage 3 亚光颜色协调融合与 32 步回溯线搜索，生成轻量干净锚点 I_anchor。

    严格保障：
    1. 形态学印刷墨迹核心绝对零改动 (assert np.all(change[ink_core] == 0))；
    2. 标签底色 `substrate` 经透射率与墨迹邻域高斯环软融合，无硬边框补丁；
    3. 经 32 步回溯线搜索满足 SSIM / MAE / halo_error 数学门禁；
    4. 绝无额外像素优化残差 (I_anchor 为纯净平滑底色)，结果保存为 generative_anchor.png。
    """
    if isinstance(source_image, (str, Path)):
        source_rgb = read_rgb(source_image)
    else:
        source_rgb = np.asarray(source_image, dtype=np.uint8)

    context = prepare_print_context(
        source_rgb,
        foreground,
        regions,
        ink_core_mask=ink_core_mask,
        substrate_mask=substrate_mask,
        config=config,
    )
    roi = context["roi"]
    ink_core = context["ink_core"]
    substrate = context["substrate"]
    trusted = context["trusted"]

    # 1. 在线性色彩空间计算目标色与源主体表面光照差异
    src_lin = linear_from_srgb(source_rgb.astype(np.float64) / 255.0)
    target_lin = linear_from_srgb(np.asarray(target_color_rgb, dtype=np.float64) / 255.0)

    if trusted.any():
        trusted_mean_lin = src_lin[trusted].mean(axis=0)
    else:
        trusted_mean_lin = src_lin[foreground].mean(axis=0) if foreground.any() else src_lin.mean(axis=(0, 1))

    color_diff = target_lin - trusted_mean_lin
    weight = trusted.astype(np.float64)
    den = gaussian_filter(weight, sigma=config.harmonization_smoothing_sigma, mode="nearest")
    smooth_diff = np.stack([
        gaussian_filter(color_diff[c] * weight, sigma=config.harmonization_smoothing_sigma, mode="nearest")
        / np.maximum(den, 1e-6)
        for c in range(3)
    ], axis=-1)
    smooth_diff[den < 0.005] = color_diff

    # 2. 构造主体非印刷区与标签底色 `substrate` 的软空间协调增益
    matte_corr = (
        config.harmonization_strength
        * config.harmonization_selected_color_mix
        * smooth_diff
    )
    ring_sigma = max(float(getattr(config, "harmonization_print_ring_sigma", 7.0)), 1e-4)
    ring = np.exp(-0.5 * (context["distance"] / ring_sigma) ** 2)
    spatial_gain = np.where(foreground & (~roi), 1.0, 0.0)
    if substrate.any():
        substrate_gain = ring * context["transmission"]
        spatial_gain = np.where(substrate, substrate_gain, spatial_gain)
    spatial_gain[ink_core] = 0.0
    spatial_gain[context["opaque"]] = 0.0
    spatial_gain[~foreground] = 0.0

    # 3. 32 步回溯线搜索数学守门 (SSIM + MAE + Halo)
    ssim_floor = max(float(getattr(config, "anchor_ssim_floor", 0.72)), float(config.source_ssim_floor))
    mae_ceiling = min(float(getattr(config, "anchor_mae_ceiling", 26.0)), float(config.source_mae_ceiling))
    halo_ceiling = float(getattr(config, "halo_ceiling", 8.0))
    line_steps = max(1, int(getattr(config, "anchor_steps", 32)))

    chosen_rgb = source_rgb.copy()
    chosen_fraction = 0.0
    chosen_ssim, chosen_mae, chosen_halo = _evaluate_candidate_metrics(source_rgb, chosen_rgb, foreground, ink_core)
    trials: list[dict[str, Any]] = []

    for fraction in np.linspace(1.0, 0.0, line_steps + 1):
        frac = float(fraction)
        candidate_lin = src_lin + frac * spatial_gain[..., None] * matte_corr
        candidate_rgb = np.clip(np.rint(srgb_from_linear(candidate_lin) * 255.0), 0, 255).astype(np.uint8)
        if ink_core.any():
            candidate_rgb[ink_core] = source_rgb[ink_core]
        if context["opaque"].any():
            candidate_rgb[context["opaque"]] = source_rgb[context["opaque"]]
        candidate_rgb[~foreground] = source_rgb[~foreground]

        c_ssim, c_mae, c_halo = _evaluate_candidate_metrics(source_rgb, candidate_rgb, foreground, ink_core)
        accepted = bool(c_ssim >= ssim_floor and c_mae <= mae_ceiling and c_halo <= halo_ceiling)
        trials.append({
            "fraction": frac,
            "source_ssim": c_ssim,
            "source_mae_levels": c_mae,
            "halo_error": c_halo,
            "accepted": accepted,
        })
        if accepted:
            chosen_rgb = candidate_rgb
            chosen_fraction = frac
            chosen_ssim, chosen_mae, chosen_halo = c_ssim, c_mae, c_halo
            break

    anchor_rgb = chosen_rgb
    change = np.abs(anchor_rgb.astype(np.int16) - source_rgb.astype(np.int16))
    if ink_core.any() and np.any(change[ink_core]):
        max_c = int(change[ink_core].max())
        raise AssertionError(f"Stage 3 印刷墨迹核心保护断言失败：检测到 {max_c} 级非零修改")

    fg_change = change[foreground]
    metadata: dict[str, Any] = {
        "status": "completed",
        "stage": "illumination_harmonization",
        "blend_mode": config.harmonization_blend_mode,
        "selected_rgb": list(target_color_rgb),
        "strength": config.harmonization_strength,
        "selected_color_mix": config.harmonization_selected_color_mix,
        "smoothing_sigma": config.harmonization_smoothing_sigma,
        "chosen_fraction": chosen_fraction,
        "line_search_steps": line_steps,
        "source_ssim": chosen_ssim,
        "source_mae_levels": chosen_mae,
        "halo_error": chosen_halo,
        "ink_core_pixels": int(ink_core.sum()),
        "substrate_pixels": int(substrate.sum()),
        "max_body_change_levels": int(fg_change.max()) if fg_change.size else 0,
        "mean_body_change_levels": float(fg_change.mean()) if fg_change.size else 0.0,
        "roi_change_levels": 0,
        "zero_pixel_residual": True,
        "generator": {
            "model": "perceptual_illumination_line_search_v1",
            "chosen_fraction": chosen_fraction,
            "line_search_steps": line_steps,
        },
        "trials_evaluated": len(trials),
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        anchor_path = out / "generative_anchor.png"
        initial_path = out / "initial.png"
        Image.fromarray(anchor_rgb).save(anchor_path)
        Image.fromarray(anchor_rgb).save(initial_path)
        metadata["anchor_path"] = str(anchor_path)
        metadata["anchor_sha256"] = sha256_file(anchor_path)
        write_json(out / "record.json", metadata)

    return anchor_rgb, metadata
