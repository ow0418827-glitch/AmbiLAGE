"""Stage 2: 目标商品主体色彩统计与环境光基准提取 (Statistical Color Selection)。

实现内容：
1. 提取目标商品 (Target) 主体有效色块，排除边框相连背景色；
2. 剔除高光反光 (Specular Highlights) 与暗部阴影 (0.20 ~ 0.90 亮度分位截断)；
3. 计算 Lab 空间核心色彩均值与协方差，逆变换为标准 sRGB 目标主色；
4. 输出结构化色彩凭证与用于后续光照融合的环境光偏置基准。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, binary_propagation
from skimage.color import lab2rgb, rgb2lab

from src.visual.image_utils import read_rgb, write_json


def extract_foreground_statistics(
    target: Path | str | np.ndarray,
    target_foreground: np.ndarray,
    output_dir: Path | str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """提取目标商品主体色彩统计分布，生成主色调与环境光基准。"""
    if isinstance(target, (str, Path)):
        target_array = read_rgb(target)
    else:
        target_array = np.asarray(target, dtype=np.uint8)

    lab = rgb2lab(target_array.astype(np.float32) / 255.0)

    # 1. 提取图像四周边框推断背景底色
    border = np.concatenate([
        target_array[0], target_array[-1], target_array[:, 0], target_array[:, -1]
    ], axis=0)
    bg_color = np.median(border, axis=0)

    # 2. 沿边缘连通扩散排除背景渗入
    potential_bg = np.linalg.norm(target_array.astype(np.float32) - bg_color.astype(np.float32), axis=-1) <= 10.0
    seed = np.zeros(potential_bg.shape, dtype=bool)
    seed[[0, -1], :] = potential_bg[[0, -1], :]
    seed[:, [0, -1]] = potential_bg[:, [0, -1]]
    connected_bg = binary_propagation(seed, mask=potential_bg)

    # 3. 前景腐蚀去除轮廓混色带
    fg_clean = binary_erosion(target_foreground.astype(bool) & (~connected_bg), iterations=2)
    if fg_clean.sum() < 50:
        fg_clean = target_foreground.astype(bool) & (~connected_bg)
    if fg_clean.sum() < 10:
        fg_clean = target_foreground.astype(bool)

    # 4. 亮度分位数裁剪 (20% ~ 90%) 剔除暗部阴影与高光反光
    values = lab[fg_clean]
    if len(values) >= 10:
        low, high = np.quantile(values[:, 0], [0.20, 0.90])
        body_mask = fg_clean & (lab[..., 0] >= low) & (lab[..., 0] <= high)
    else:
        low, high = 0.0, 100.0
        body_mask = fg_clean

    if body_mask.sum() < 10:
        body_mask = fg_clean

    distribution = lab[body_mask]
    mean_lab = distribution.mean(axis=0)
    cov_lab = np.cov(distribution.T) if len(distribution) > 3 else np.eye(3)

    # 5. 逆映射为 sRGB 颜色
    rgb_floats = lab2rgb(mean_lab.reshape(1, 1, 3))[0, 0]
    rgb_u8 = np.clip(np.rint(rgb_floats * 255.0), 0, 255).astype(int).tolist()

    metadata: dict[str, Any] = {
        "color_scheme": "statistics",
        "rgb": rgb_u8,
        "lab_mean": mean_lab.tolist(),
        "lab_covariance": cov_lab.tolist(),
        "luminance_quantiles": [float(low), float(high)],
        "target_foreground_pixels": int(fg_clean.sum()),
        "body_pixels": int(body_mask.sum()),
        "photo_border_background_rgb": bg_color.tolist(),
        "border_connected_background_pixels": int(connected_bg.sum()),
        "covariance_diagnostic_only": True,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "record.json", metadata)

    return metadata, body_mask
