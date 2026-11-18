"""前景裁剪与背景抑制视觉外形特征模块（用于杂乱室内场景选图，不修改整图基线）。"""
from __future__ import annotations

from src.paths import ProjectPath as Path
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    gaussian_filter,
    label,
)

from src.visual.audit import digest, freeze_json, model_files, read_json
from src.visual.image_utils import sha256_file
from src.visual.shape_features import LocalShapeBackend, PREPROCESSING

PREPROCESSING_FG_CROP = dict(
    policy="foreground_crop_gray_letterbox224_global_cls_v1",
    size=224,
    grayscale="Pillow_L_then_three_channels",
    resize="bicubic_preserve_aspect",
    padding=127,
    foreground_segmentation=True,
    text_detection=False,
    pilot_edit=False,
)

VALID_SHAPE_POLICIES = (
    PREPROCESSING["policy"],
    PREPROCESSING_FG_CROP["policy"],
)


def _estimate_foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """估计前景商品掩码（[H, W] bool），兼顾纯色棚拍与杂乱室内/高噪背景。"""
    h, w = rgb.shape[:2]
    max_side = max(h, w)
    if max_side > 128:
        scale = 128.0 / float(max_side)
        work_w = max(16, int(round(w * scale)))
        work_h = max(16, int(round(h * scale)))
        small = np.asarray(
            Image.fromarray(rgb).resize((work_w, work_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    else:
        work_h, work_w = h, w
        small = rgb.astype(np.float32)

    bw = max(2, min(work_h, work_w) // 10)
    border_mask = np.zeros((work_h, work_w), dtype=bool)
    border_mask[:bw, :] = True
    border_mask[-bw:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True

    cy0, cy1 = max(bw, work_h // 5), min(work_h - bw, work_h - work_h // 5)
    cx0, cx1 = max(bw, work_w // 5), min(work_w - bw, work_w - work_w // 5)
    center_mask = np.zeros((work_h, work_w), dtype=bool)
    if cy1 > cy0 and cx1 > cx0:
        center_mask[cy0:cy1, cx0:cx1] = True
    else:
        center_mask = ~border_mask

    border_vals = small[border_mask]
    bg_med = np.median(border_vals, axis=0)
    border_dist = np.linalg.norm(border_vals - bg_med, axis=-1)
    p90_border_dist = float(np.percentile(border_dist, 90))

    if p90_border_dist < 16.0:
        dist_all = np.linalg.norm(small - bg_med, axis=-1)
        raw_fg = dist_all > max(18.0, p90_border_dist + 8.0)
    else:
        # 杂乱/纹理/随机噪声背景：使用平滑三维色彩直方图中心-边缘似然比与中心空间先验
        bins = 12
        quant = np.clip((small * (bins / 256.0)).astype(np.int32), 0, bins - 1)
        flat_idx = (quant[..., 0] * bins + quant[..., 1]) * bins + quant[..., 2]

        fg_counts = np.bincount(flat_idx[center_mask].ravel(), minlength=bins**3).astype(np.float32)
        bg_counts = np.bincount(flat_idx[border_mask].ravel(), minlength=bins**3).astype(np.float32)
        fg_grid = gaussian_filter(fg_counts.reshape(bins, bins, bins), sigma=1.0, mode="nearest")
        bg_grid = gaussian_filter(bg_counts.reshape(bins, bins, bins), sigma=1.0, mode="nearest")

        p_fg = fg_grid / max(float(fg_grid.sum()), 1e-6)
        p_bg = bg_grid / max(float(bg_grid.sum()), 1e-6)
        likelihood = p_fg.ravel()[flat_idx] / (p_fg.ravel()[flat_idx] + p_bg.ravel()[flat_idx] + 1e-8)

        yy = np.linspace(-1.0, 1.0, work_h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, work_w, dtype=np.float32)[None, :]
        spatial_prior = np.exp(-0.5 * ((yy / 0.90) ** 2 + (xx / 0.90) ** 2))
        smooth_like = gaussian_filter(likelihood.astype(np.float32), sigma=1.0, mode="nearest")
        raw_fg = (smooth_like * (0.45 + 0.55 * spatial_prior)) > 0.46

    cleaned = binary_fill_holes(raw_fg)
    cleaned = binary_opening(cleaned, iterations=1)
    cleaned = binary_closing(cleaned, iterations=1)
    cleaned = binary_fill_holes(cleaned)

    labeled, num_features = label(cleaned)
    if num_features > 0:
        min_area = max(4, int(0.015 * work_h * work_w))
        kept = np.zeros_like(cleaned, dtype=bool)
        central_box = np.zeros_like(cleaned, dtype=bool)
        central_box[work_h // 4 : (3 * work_h + 3) // 4, work_w // 4 : (3 * work_w + 3) // 4] = True
        for cid in range(1, num_features + 1):
            comp = labeled == cid
            if int(comp.sum()) >= min_area and bool(np.any(comp & central_box)):
                kept |= comp
        if kept.any():
            cleaned = kept

    area_ratio = float(cleaned.mean())
    if area_ratio < 0.02 or area_ratio > 0.96:
        # 退化全匀或极端图像时保留中央主体区域，避免空裁剪
        fallback = np.zeros((work_h, work_w), dtype=bool)
        fallback[work_h // 10 : work_h - work_h // 10, work_w // 10 : work_w - work_w // 10] = True
        cleaned = fallback if fallback.any() else np.ones((work_h, work_w), dtype=bool)

    if (work_h, work_w) != (h, w):
        mask_img = Image.fromarray(cleaned.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)
        full_mask = np.asarray(mask_img) > 127
    else:
        full_mask = cleaned

    return binary_dilation(full_mask, iterations=1)


def foreground_crop_shape_input(path: str | Path, *, margin_ratio: float = 0.08) -> Image.Image:
    """提取前景轮廓并把背景抑制为中性灰 (127)，按前景边界框裁剪后转为 224x224 灰度图。"""
    with Image.open(path) as opened:
        rgb_img = ImageOps.exif_transpose(opened).convert("RGB")
        rgb = np.asarray(rgb_img, dtype=np.uint8).copy()

    h, w = rgb.shape[:2]
    fg_mask = _estimate_foreground_mask(rgb)
    ys, xs = np.where(fg_mask)
    if ys.size > 0 and xs.size > 0:
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        pad = max(1, int(round(margin_ratio * max(y1 - y0, x1 - x0))))
        y0 = max(0, y0 - pad)
        y1 = min(h, y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(w, x1 + pad)
        suppressed = rgb.copy()
        suppressed[~fg_mask] = 127
        cropped = Image.fromarray(suppressed[y0:y1, x0:x1], mode="RGB")
    else:
        cropped = rgb_img

    gray = cropped.convert("L")
    contained = ImageOps.contain(gray, (224, 224), Image.Resampling.BICUBIC)
    canvas = Image.new("L", (224, 224), 127)
    canvas.paste(contained, ((224 - contained.width) // 2, (224 - contained.height) // 2))
    return canvas.convert("RGB")


class ForegroundCropShapeBackend:
    """前景裁剪+背景抑制 DINO 外形后端，对外接口与 LocalShapeBackend 完全一致。"""

    def __init__(self, settings, output, device="cuda"):
        if set(settings) - {"dino_model", "batch_size"}:
            raise ValueError("新选图器只接受一个冻结视觉模型及批量大小")
        from src.paths import PROJECT
        from src.model_registry import resolve_model
        from transformers import AutoImageProcessor, AutoModel
        import PIL
        import torch
        import transformers

        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = settings.get("batch_size", 16)
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("批量大小须为正整数")
        path = Path(settings["dino_model"]).expanduser() if settings.get("dino_model") else Path(resolve_model("selection"))
        path = (path if path.is_absolute() else PROJECT / path).resolve()
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).float().eval().requires_grad_(False).to(device)
        self.snapshot = dict(
            dino=model_files(path),
            preprocessing=PREPROCESSING_FG_CROP,
            precision="float32",
            libraries=dict(
                transformers=transformers.__version__,
                torch=torch.__version__,
                pillow=PIL.__version__,
                numpy=np.__version__,
            ),
            implementation_sha256=sha256_file(__file__),
        )
        self.identity = digest(self.snapshot)
        self.cache = {}

    def features_many(self, products):
        import torch

        pending = {}
        for product in products:
            actual = sha256_file(product["image"])
            if actual != product["image_sha256"]:
                raise ValueError("公开图片内容与候选记录不一致")
            identity = dict(
                image_sha256=actual,
                model_sha256=self.identity,
                preprocessing=PREPROCESSING_FG_CROP,
            )
            key = digest(identity)
            path = self.output / (key + ".json")
            if key in self.cache:
                continue
            if path.exists():
                saved = read_json(path)
                if saved["identity"] != identity or saved["vector_sha256"] != digest(saved["vector"]):
                    raise ValueError("外形特征缓存身份或内容改变")
                vector = np.asarray(saved["vector"], dtype=np.float64)
                if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    raise ValueError("外形特征缓存无效")
                self.cache[key] = dict(shape=vector)
            else:
                pending[key] = (product, identity, path)
        items = list(pending.items())
        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]
            images = [foreground_crop_shape_input(value[0]["image"]) for _, value in batch]
            inputs = self.processor(
                images=images,
                do_resize=False,
                do_center_crop=False,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                vectors = self.model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
            for (key, (_, identity, path)), vector in zip(batch, vectors):
                values = vector.astype(np.float64).tolist()
                freeze_json(path, dict(identity=identity, vector=values, vector_sha256=digest(values)))
                self.cache[key] = dict(shape=np.asarray(values))

    def features(self, product):
        self.features_many([product])
        key = digest(
            dict(
                image_sha256=product["image_sha256"],
                model_sha256=self.identity,
                preprocessing=PREPROCESSING_FG_CROP,
            )
        )
        return self.cache[key]

    def close(self):
        self.model = None
        self.cache.clear()
        import gc
        import torch

        gc.collect()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_shape_backend(settings, output, *, device="cuda", preprocessing=None):
    """按预处理策略分派外形特征后端；未指定时保持原整图策略不变。"""
    if preprocessing is None or preprocessing == PREPROCESSING["policy"]:
        return LocalShapeBackend(settings, output, device=device)
    if preprocessing == PREPROCESSING_FG_CROP["policy"]:
        return ForegroundCropShapeBackend(settings, output, device=device)
    raise ValueError(f"不支持的选图外形预处理策略：{preprocessing}")
