"""Stage 1: 多区域精细分割与四重空间余弦平滑包络构建 (Segmentation & Spatial Envelope)。

实现内容：
1. 商品主体前景分割 (SAM2 / ViT 引导，支持本地纯净无模型降级分割)；
2. 印刷文字与品牌保护区检测与掩码提取 (ROI Detection)；
3. 基于欧氏距离变换 (EDT) 的四重空间余弦平滑包络张量计算：
   - w_bg_budget: 轮廓外 0~3px 抗锯齿平滑带 (余弦 0.70 -> 1.0，抹除纯白光晕环)；
   - w_bg_mod: 轮廓外 0~12px 纹理抑制带 (余弦 0.25 -> 1.0，抑制近轮廓高频波动)；
   - w_fg: 轮廓向内 0~8px 主体边距内缩带 (余弦 0.0 -> 1.0)；
   - w_text: 印刷文字保护区向外 6px 缓冲平滑带 (余弦平滑过渡，保护印刷文字绝对零改动)。
"""
from __future__ import annotations

import gc
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_fill_holes
import torch

from src.paths import model_path
from src.visual.image_utils import read_rgb, sha256_file, write_json
from .config import AmbiLAGEConfig


def print_region_mask(shape: tuple[int, ...], regions: Sequence[Sequence[float]] | None) -> np.ndarray:
    """根据归一化边界框 [x0, y0, x1, y1] 生成印刷文字保护区二值掩码 [H, W]。"""
    h, w = shape[:2]
    result = np.zeros((h, w), dtype=bool)
    if not regions:
        return result
    for x0, y0, x1, y1 in regions:
        if not (0.0 <= x0 < x1 <= 1.0) or not (0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"印刷区域归一化坐标无效: {[x0, y0, x1, y1]}")
        r_y0, r_y1 = int(np.floor(y0*h)), int(np.ceil(y1*h))
        r_x0, r_x1 = int(np.floor(x0*w)), int(np.ceil(x1*w))
        result[r_y0:r_y1, r_x0:r_x1] = True
    return result


def _heuristic_segmentation(image_array: np.ndarray) -> np.ndarray:
    """当无深度分割权重时的稳健启发式主体前景分割。
    
    只供显式合成夹具使用；生产分割不会调用此函数。
    """
    h, w = image_array.shape[:2]
    # 提取四周边框像素推断背景基色
    border = np.concatenate([
        image_array[0], image_array[-1], image_array[:, 0], image_array[:, -1]
    ], axis=0)
    bg_color = np.median(border, axis=0)
    
    diff = np.linalg.norm(image_array.astype(np.float32) - bg_color.astype(np.float32), axis=-1)
    # 差值大于阈值者视为主体候选
    fg_candidate = diff > 15.0
    
    # 填充前景内部孔洞
    fg_filled = binary_fill_holes(fg_candidate)
    
    # 若主体极小则退化为中央区域
    if fg_filled.sum() < (h * w * 0.02):
        cy, cx = h // 2, w // 2
        ry, rx = int(h * 0.35), int(w * 0.35)
        fg_filled[cy - ry : cy + ry, cx - rx : cx + rx] = True
    return fg_filled


def segment_image(
    image_path: Path | str,
    boxes: Sequence[Sequence[float]] | None = None,
    output_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    device: str = "cuda",
) -> tuple[np.ndarray, dict[str, Any]]:
    """真实大模型分割；归一化提示框转换为原图像素，失败即停止。"""
    from transformers import Sam2Processor, Sam2Model
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    normalized = boxes or [[0., 0., 1., 1.]]
    pixels = []
    for box in normalized:
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("主体提示框必须是有效归一化坐标")
        pixels.append([x0*w, y0*h, x1*w, y1*h])
    path = Path(model_dir or model_path("sam2"))
    if not path.is_dir():
        raise FileNotFoundError(f"缺少真实分割权重: {path}")
    processor = Sam2Processor.from_pretrained(str(path), local_files_only=True)
    model = Sam2Model.from_pretrained(str(path), local_files_only=True).eval().to(device)
    try:
        inputs = processor(images=image, input_boxes=[pixels], return_tensors="pt").to(device)
        with torch.inference_mode():
            pred = model(**inputs, multimask_output=True)
        masks = processor.post_process_masks(pred.pred_masks.cpu(), inputs["original_sizes"])[0].numpy().astype(bool)
        scores = pred.iou_scores[0].detach().cpu().numpy()
        selected = [masks[i, int(np.argmax(scores[i]))] for i in range(len(pixels))]
        mask = np.logical_or.reduce(selected)
        ratio = float(mask.mean())
        if not .01 < ratio < .99:
            raise ValueError(f"主体分割面积异常，停止生成: {ratio:.4f}")
        record = dict(image_path=str(image_path), image_sha256=sha256_file(image_path),
                      boxes=normalized, segmenter="sam2.1-hiera-large", model=str(path),
                      scores=scores.tolist(), foreground_ratio=ratio)
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if output_dir:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8)*255).save(out / "foreground.png")
        write_json(out / "record.json", record)
    return mask, record


def compute_spatial_envelopes(
    foreground: np.ndarray,
    roi: np.ndarray,
    config: AmbiLAGEConfig,
) -> dict[str, Any]:
    """计算基于欧氏距离变换 (EDT) 的四重空间余弦平滑包络张量与扰动预算映射。
    
    1. w_bg_budget:
       外侧欧氏距离变换 dist_out = distance_transform_edt(~foreground)。
       在轮廓外侧 0~3px (1~3px) 抗锯齿带内从 0.70 平滑升至 1.0，抹除白边与纯白光晕环。
    2. w_bg_mod:
       在轮廓外侧 edge_feather_pixels (12px) 内从 0.25 余弦平滑升至 1.0，抑制紧邻边缘局部高频波动。
    3. w_fg:
       主体内部欧氏距离变换 dist_in = distance_transform_edt(foreground)。
       轮廓向内 fg_edge_feather_pixels (8px) 内从 0.0 余弦平滑升至 1.0，使边缘质感紧密锚定。
    4. w_text:
       印刷文字保护区向外缓冲 roi_feather_pixels (6px) 余弦平滑，确保文字区域绝对零改动且过渡平滑。
    """
    fg_bool = foreground.astype(bool)
    bg_bool = ~fg_bool
    roi_bool = roi.astype(bool)

    # 1. 背景空间包络 (w_bg_budget & w_bg_mod)
    dist_out = distance_transform_edt(bg_bool)
    t_amb = np.clip((dist_out - 1.0) / max(config.ambient_feather_pixels, 1.0), 0.0, 1.0)
    w_bg_budget = (0.70 + 0.30 * 0.5 * (1.0 - np.cos(np.pi * t_amb))) * bg_bool

    t_out = np.clip(dist_out / max(config.edge_feather_pixels, 1.0), 0.0, 1.0)
    if config.edge_feather_mode == "cosine":
        w_bg_mod = (0.25 + 0.75 * 0.5 * (1.0 - np.cos(np.pi * t_out))) * bg_bool
    else:
        w_bg_mod = (0.25 + 0.75 * t_out) * bg_bool

    # 2. 主体微光影空间包络 (w_fg)
    dist_in = distance_transform_edt(fg_bool)
    t_in = np.clip(dist_in / max(config.fg_edge_feather_pixels, 1.0), 0.0, 1.0)
    if config.edge_feather_mode == "cosine":
        w_fg = 0.5 * (1.0 - np.cos(np.pi * t_in)) * fg_bool
    else:
        w_fg = t_in * fg_bool

    # 3. 印刷文字向外余弦缓冲过渡 (w_text)
    if config.roi_feather_pixels > 0.0 and roi_bool.any():
        dist_text = distance_transform_edt(~roi_bool)
        t_text = np.clip(dist_text / config.roi_feather_pixels, 0.0, 1.0)
        w_text = 0.5 * (1.0 - np.cos(np.pi * t_text))
        w_fg_body = w_fg * w_text
    else:
        w_text = np.ones_like(w_fg)
        w_fg_body = w_fg * (~roi_bool)

    # 4. 空间扰动预算分配（文字区强制为 0）
    fg_level_array = (config.latent_levels * w_fg_body) * (~roi_bool)
    bg_level_array = (config.background_levels * w_bg_budget) * (~roi_bool)
    level_array = fg_level_array + bg_level_array
    max_level = float(max(config.background_levels, config.latent_levels, 1))
    macro_array = level_array / max_level

    return {
        "w_bg_budget": w_bg_budget.astype(np.float32),
        "w_bg_mod": w_bg_mod.astype(np.float32),
        "w_fg": w_fg.astype(np.float32),
        "w_text": w_text.astype(np.float32),
        "w_fg_body": w_fg_body.astype(np.float32),
        "fg_level_array": fg_level_array.astype(np.float32),
        "bg_level_array": bg_level_array.astype(np.float32),
        "level_array": level_array.astype(np.float32),
        "macro_array": macro_array.astype(np.float32),
        "foreground": fg_bool,
        "background": bg_bool,
        "roi": roi_bool,
        "support_mask": (~roi_bool).astype(np.float32),
    }
