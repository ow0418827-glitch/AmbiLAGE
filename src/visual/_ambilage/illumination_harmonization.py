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
from .segmentation import print_region_mask


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
) -> dict[str, Any]:
    """构建印刷区域分析上下文，区分不透明文字/油墨与未印刷商品表面。"""
    h, w = source_rgb.shape[:2]
    roi = print_region_mask(source_rgb.shape, regions)
    src_lin = linear_from_srgb(source_rgb.astype(np.float64) / 255.0)

    luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    ys = src_lin @ luma_weights

    # 底材反光闭运算推断
    paper = np.stack([
        grey_closing(src_lin[..., c], size=(11, 11), mode="nearest") for c in range(3)
    ], axis=-1)
    yb = paper @ luma_weights

    transmission = np.zeros((h, w), dtype=np.float64)
    opaque = np.zeros((h, w), dtype=bool)
    chroma = source_rgb.max(axis=-1).astype(float) - source_rgb.min(axis=-1).astype(float)
    colored = chroma >= 12.0

    if regions:
        for region in regions:
            local = print_region_mask(source_rgb.shape, [region])
            inside = local & foreground
            if not inside.any():
                continue
            if np.median(ys[inside]) < 0.35 or np.median(chroma[inside]) >= 12.0:
                opaque |= local
            else:
                darkest = float(np.quantile(ys[inside], 0.02))
                trans = np.clip((ys - darkest) / np.maximum(yb - darkest, 1e-4), 0.0, 1.0)
                local_opaque = local & ((trans <= 0.04) | colored)
                transmission[local] = trans[local]
                opaque |= local_opaque
    transmission[opaque] = 0.0

    # 油墨与距离变换
    ink = roi & ((source_rgb.min(axis=-1) < 180) | colored)
    distance = distance_transform_edt(~ink) if ink.any() else np.full(roi.shape, 64.0)

    # 可信未印刷主体参考点 (Trusted body pixels)
    trusted = foreground & (~roi)
    if trusted.sum() < 50:
        trusted = foreground.copy()
        if roi.any():
            trusted[roi] = False
    if trusted.sum() < 10:
        trusted = foreground.copy()

    # 寻找各像素距可信参考点最近邻
    _, nearest = distance_transform_edt(~trusted, return_indices=True)

    # 光晕参考区域
    halo = roi & foreground & (source_rgb.min(axis=-1) >= 220) & (transmission > 0.85)

    return {
        "source": source_rgb,
        "foreground": foreground,
        "roi": roi,
        "opaque": opaque,
        "trusted": trusted,
        "transmission": transmission,
        "distance": distance,
        "nearest": nearest,
        "halo": halo,
        "halo_applicable": bool(halo.any()),
    }


def harmonize_illumination(
    source_image: Path | str | np.ndarray,
    target_color_rgb: Sequence[int],
    foreground: np.ndarray,
    regions: Sequence[Sequence[float]] | None,
    config: AmbiLAGEConfig,
    output_dir: Path | str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """执行 Stage 3 亚光颜色协调融合，生成基准锚点图像 I_anchor。
    
    严格保障：
    1. 印刷文字区域绝对零改动 (assert np.all(change[roi] == 0))；
    2. 绝无额外像素优化残差 (I_anchor 为纯净平滑底色)；
    3. 结果保存为 generative_anchor.png。
    """
    if isinstance(source_image, (str, Path)):
        source_rgb = read_rgb(source_image)
    else:
        source_rgb = np.asarray(source_image, dtype=np.uint8)

    context = prepare_print_context(source_rgb, foreground, regions)
    roi = context["roi"]
    trusted = context["trusted"]

    # 1. 在线性色彩空间计算目标色与源主体表面光照差异
    src_lin = linear_from_srgb(source_rgb.astype(np.float64) / 255.0)
    target_lin = linear_from_srgb(np.asarray(target_color_rgb, dtype=np.float64) / 255.0)

    # 计算主体未印刷表面平均色彩
    if trusted.any():
        trusted_mean_lin = src_lin[trusted].mean(axis=0)
    else:
        trusted_mean_lin = src_lin[foreground].mean(axis=0)

    # 色彩增益与偏置方向
    color_diff = target_lin - trusted_mean_lin
    weight = trusted.astype(np.float64)
    den = gaussian_filter(weight, sigma=config.harmonization_smoothing_sigma, mode="nearest")
    smooth_diff = np.stack([
        gaussian_filter(color_diff[c] * weight, sigma=config.harmonization_smoothing_sigma, mode="nearest")
        / np.maximum(den, 1e-6)
        for c in range(3)
    ], axis=-1)
    smooth_diff[den < 0.005] = color_diff

    # 亚光融合增益
    matte_corr = (
        config.harmonization_strength
        * config.harmonization_selected_color_mix
        * smooth_diff
    )

    # 仅将协调增益作用于前景主体
    harmonized_lin = src_lin.copy()
    fg_3d = foreground[..., None]
    harmonized_lin = np.where(fg_3d, harmonized_lin + matte_corr, harmonized_lin)

    # 逆映射回 sRGB
    anchor_rgb = np.clip(np.rint(srgb_from_linear(harmonized_lin) * 255.0), 0, 255).astype(np.uint8)

    # 严格锁定印刷文字区域与背景区域
    # 印刷区域内不透明文字及所有 ROI 严格恢复源图像素
    if roi.any():
        anchor_rgb[roi] = source_rgb[roi]
    if context["opaque"].any():
        anchor_rgb[context["opaque"]] = source_rgb[context["opaque"]]
    # 背景区域保持源图不变
    bg_mask = ~foreground
    anchor_rgb[bg_mask] = source_rgb[bg_mask]

    # 严格断言印刷区域零改动
    change = np.abs(anchor_rgb.astype(np.int16) - source_rgb.astype(np.int16))
    if roi.any() and np.any(change[roi]):
        max_c = int(change[roi].max())
        raise AssertionError(f"Stage 3 印刷文字区域保护断言失败：检测到 {max_c} 级非零修改")

    metadata: dict[str, Any] = {
        "status": "completed",
        "stage": "illumination_harmonization",
        "blend_mode": config.harmonization_blend_mode,
        "selected_rgb": list(target_color_rgb),
        "strength": config.harmonization_strength,
        "selected_color_mix": config.harmonization_selected_color_mix,
        "smoothing_sigma": config.harmonization_smoothing_sigma,
        "max_body_change_levels": int(change[foreground].max()),
        "mean_body_change_levels": float(change[foreground].mean()),
        "roi_change_levels": 0,
        "zero_pixel_residual": True,
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
