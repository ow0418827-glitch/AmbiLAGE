"""真实阶段后端。没有权重或依赖时停止，不生成替代生产结果。"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from src.paths import model_path, PROJECT
from .config import AmbiLAGEConfig


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def local_model(explicit: str | None, key: str) -> str:
    path = Path(explicit or model_path(key)).expanduser()
    path = (path if path.is_absolute() else PROJECT / path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f'缺少真实模型目录 {path}，请先运行 tools/prepare_models.py；生产模式禁止模型替代')
    return str(path)


def _strip_json_comments_and_trailing_commas(text: str) -> str:
    """移除 JSON 字符串外的 `//`/`/* */` 注释与尾随逗号。"""
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    escaped = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif ch == '/' and i + 1 < n and text[i + 1] == '/':
            i += 2
            while i < n and text[i] not in '\r\n':
                i += 1
        elif ch == '/' and i + 1 < n and text[i + 1] == '*':
            end = text.find('*/', i + 2)
            i = n if end == -1 else end + 2
        else:
            out.append(ch)
            i += 1
    cleaned = ''.join(out)
    # 去除对象或数组闭括号前多余逗号
    result: list[str] = []
    in_string = False
    escaped = False
    for idx, ch in enumerate(cleaned):
        if in_string:
            result.append(ch)
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            result.append(ch)
        elif ch == ',':
            j = idx + 1
            while j < len(cleaned) and cleaned[j].isspace():
                j += 1
            if j < len(cleaned) and cleaned[j] in ']}':
                continue
            result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def parse_json_response(text: str) -> dict:
    value = text.strip()
    if '```' in value:
        parts = value.split('```')
        fenced = [parts[k].split('\n', 1)[1] if '\n' in parts[k] else parts[k] for k in range(1, len(parts), 2)]
        if fenced:
            value = max(fenced, key=len).strip()
    first, last = value.find('{'), value.rfind('}')
    if first != -1 and last > first:
        value = value[first : last + 1]
    result = json.loads(_strip_json_comments_and_trailing_commas(value))
    if not isinstance(result, dict):
        raise ValueError('视觉规划响应必须为对象')
    return result


def _normalize_planner_boxes(
    boxes: Any,
    width: int,
    height: int,
    *,
    allow_empty: bool,
) -> list[list[float]]:
    """确保视觉规划返回的框全部落在有效归一化 `[0, 1]` 区间内。"""
    if not isinstance(boxes, list):
        return [] if allow_empty else [[0.0, 0.0, 1.0, 1.0]]
    parsed: list[list[float]] = []
    for item in boxes:
        if isinstance(item, (list, tuple)) and len(item) == 4:
            try:
                parsed.append([float(v) for v in item])
            except (TypeError, ValueError):
                continue
    if not parsed:
        return [] if allow_empty else [[0.0, 0.0, 1.0, 1.0]]
    need_scale = any(max(b) > 1.0 for b in parsed)
    if need_scale:
        sx = max(float(width), *(b[2] for b in parsed if b[2] > 1.0))
        sy = max(float(height), *(b[3] for b in parsed if b[3] > 1.0))
    else:
        sx = sy = 1.0
    normalized: list[list[float]] = []
    for x0, y0, x1, y1 in parsed:
        nx0 = max(0.0, min(1.0, x0 / sx))
        ny0 = max(0.0, min(1.0, y0 / sy))
        nx1 = max(0.0, min(1.0, x1 / sx))
        ny1 = max(0.0, min(1.0, y1 / sy))
        if 0.0 <= nx0 < nx1 <= 1.0 and 0.0 <= ny0 < ny1 <= 1.0:
            normalized.append([nx0, ny0, nx1, ny1])
    if not normalized and not allow_empty:
        return [[0.0, 0.0, 1.0, 1.0]]
    return normalized


_DEFAULT_FLUX_KONTEXT_RESOLUTIONS = (
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
)


def _flux_kontext_resolutions() -> tuple[tuple[int, int], ...]:
    """返回 FLUX.1-Kontext 原生支持的 (width, height) 分辨率桶。"""
    try:
        from diffusers.pipelines.flux.pipeline_flux_kontext import PREFERRED_KONTEXT_RESOLUTIONS
        return tuple((int(w), int(h)) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
    except Exception:
        return _DEFAULT_FLUX_KONTEXT_RESOLUTIONS


def _prepare_flux_canvas(initial: np.ndarray) -> tuple[np.ndarray, int, int, int, int, int, int]:
    """将极细长/非原生比例图像对称留白到最近的 FLUX.1-Kontext 分辨率桶比例。"""
    h, w = initial.shape[:2]
    aspect = w / max(h, 1)
    kw, kh = min(_flux_kontext_resolutions(), key=lambda wh: abs(aspect - wh[0] / wh[1]))
    target_ar = kw / kh
    if abs(aspect - target_ar) <= 0.015:
        return initial, 0, 0, h, w, kw, kh
    pad_h = max(h, int(round(w / target_ar)))
    pad_w = max(w, int(round(h * target_ar)))
    border = np.concatenate([initial[0], initial[-1], initial[:, 0], initial[:, -1]], axis=0)
    fill = np.rint(np.median(border, axis=0)).astype(np.uint8)
    canvas = np.full((pad_h, pad_w, 3), fill, dtype=np.uint8)
    y0 = (pad_h - h) // 2
    x0 = (pad_w - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = initial
    return canvas, y0, x0, pad_h, pad_w, kw, kh


def _measure_anchor_alignment(candidate: np.ndarray, initial: np.ndarray) -> tuple[float, float]:
    """评估生成锚点与第 3 阶段底图之间的结构相似度 (SSIM) 与平均像素漂移 (MAE)。"""
    from .objectives import source_ssim
    t_cand = torch.tensor(candidate.transpose(2, 0, 1).copy(), dtype=torch.float32)[None] / 255.0
    t_init = torch.tensor(initial.transpose(2, 0, 1).copy(), dtype=torch.float32)[None] / 255.0
    ss = float(source_ssim(t_cand, t_init))
    mae = float(np.abs(candidate.astype(np.float32) - initial.astype(np.float32)).mean())
    return ss, mae


class ProductionStages:
    test_fixture = False

    def __init__(self, config: AmbiLAGEConfig, device):
        self.config, self.device = config, torch.device(device)
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('生产自然图生成要求可用显卡；本地轻量验证须显式指定 smoke_test')
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('生产阶段要求原生脑浮点十六位支持')
        self.planner = self.processor = self.sam = self.sam_processor = self.editor = None

    def close(self):
        self.planner = self.processor = self.sam = self.sam_processor = self.editor = None
        release_memory()

    def _ask(self, images: list[Image.Image], prompt: str) -> tuple[dict, str]:
        if self.planner is None:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            path = local_model(self.config.planner_model_dir, 'planner')
            self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
            self.planner = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.bfloat16, local_files_only=True,
                attn_implementation='sdpa').eval().requires_grad_(False).to(self.device)
        messages = [{'role': 'user', 'content': [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[text], images=images, padding=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            generated = self.planner.generate(**batch, do_sample=False, max_new_tokens=self.config.planner_max_new_tokens)
        response = self.processor.batch_decode(generated[:, batch.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return parse_json_response(response), response

    def plan(self, source: Path, category: str) -> dict:
        # 输入仅包含源公开图片及公开品类；不含目标图、正式请求或记忆。
        image = Image.open(source).convert('RGB')
        w, h = image.size
        prompt = ('Inspect this public product photograph. Public category: ' + category +
            '. Return strict valid JSON only (no comments, no markdown prose) with '
            'source_boxes (list of normalized [x0, y0, x1, y1] float boxes in [0.0, 1.0] covering the product), '
            'print_regions (list of normalized [x0, y0, x1, y1] float boxes in [0.0, 1.0] covering ALL printed text, brand logos and labels), '
            'edit_prompt (a subtle photographic studio relighting/background instruction preserving the exact camera framing, '
            'product scale, pixel alignment, identity, shape, label and readable text; do not re-center, crop or add brands or objects).')
        plan, raw = self._ask([image], prompt)
        if not isinstance(plan.get('source_boxes'), list) or not plan['source_boxes']:
            raise ValueError('视觉规划未返回主体提示框')
        if not isinstance(plan.get('print_regions'), list) or not isinstance(plan.get('edit_prompt'), str) or not plan['edit_prompt'].strip():
            raise ValueError('视觉规划缺少印刷区声明或编辑指令')
        plan['source_boxes'] = _normalize_planner_boxes(plan['source_boxes'], w, h, allow_empty=False)
        plan['print_regions'] = _normalize_planner_boxes(plan['print_regions'], w, h, allow_empty=True)
        plan.update(model='Qwen2.5-VL-32B-Instruct', precision='bfloat16', raw_response=raw,
                    inputs=['source_public_image', 'source_public_category'])
        return plan

    def detect_print(self, image: Path) -> tuple[list, dict]:
        from rapidocr_onnxruntime import RapidOCR
        engine = RapidOCR()
        result, elapsed = engine(str(image))
        height, width = np.asarray(Image.open(image)).shape[:2]
        regions, confidences = [], []
        for row in result or []:
            points = np.asarray(row[0])
            lo, hi = points.min(0), points.max(0)
            region = [max(0., float(lo[0])/width), max(0., float(lo[1])/height),
                      min(1., float(hi[0])/width), min(1., float(hi[1])/height)]
            if region[0] < region[2] and region[1] < region[3]:
                regions.append(region)
                confidences.append(float(row[2]))
        return regions, {'backend': 'rapidocr-onnxruntime', 'regions': regions, 'confidence': confidences}

    def segment(self, image: Path, boxes: list | None) -> tuple[np.ndarray, dict]:
        from .segmentation import segment_image
        return segment_image(image, boxes=boxes, model_dir=local_model(self.config.sam_model_dir, 'sam2'), device=str(self.device))

    @staticmethod
    def _crop_editor_output(
        image: Image.Image, pad_w: int, pad_h: int, x0: int, y0: int, w: int, h: int,
    ) -> np.ndarray:
        canvas_rgb = np.asarray(image.convert('RGB').resize((pad_w, pad_h), Image.Resampling.LANCZOS))
        return canvas_rgb[y0:y0 + h, x0:x0 + w].copy()

    def _flux_sdedit_refine(
        self, canvas: np.ndarray, prompt: str, seed: int, kw: int, kh: int,
        x0: int, y0: int, pad_w: int, pad_h: int, w: int, h: int, strength: float = 0.45,
    ) -> tuple[np.ndarray, int]:
        full_sigmas = np.linspace(1.0, 1.0 / self.config.anchor_steps, self.config.anchor_steps).tolist()
        sub_sigmas = [float(s) for s in full_sigmas if s <= strength] or [float(strength)]
        sigma0 = sub_sigmas[0]
        img_pil = Image.fromarray(canvas)
        resized = self.editor.image_processor.resize(img_pil, kh, kw)
        preprocessed = self.editor.image_processor.preprocess(resized, kh, kw)
        latent_dtype = (
            next(self.editor.transformer.parameters()).dtype
            if getattr(self.editor, 'transformer', None) is not None
            else torch.bfloat16
        )
        in_channels = (
            self.editor.transformer.config.in_channels // 4
            if getattr(self.editor, 'transformer', None) is not None
            else 16
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise_lat, img_lat, _, _ = self.editor.prepare_latents(
            preprocessed, 1, in_channels, kh, kw, latent_dtype, self.device, generator, None,
        )
        init_lat = sigma0 * noise_lat + (1.0 - sigma0) * img_lat
        with torch.inference_mode():
            refined = self.editor(
                image=img_pil, prompt=prompt,
                num_inference_steps=len(sub_sigmas), sigmas=sub_sigmas,
                guidance_scale=self.config.anchor_guidance_scale,
                height=kh, width=kw, max_area=kh * kw, _auto_resize=False,
                latents=init_lat, generator=generator,
            ).images[0]
        return self._crop_editor_output(refined, pad_w, pad_h, x0, y0, w, h), len(sub_sigmas)

    def generate_anchor(self, initial: np.ndarray, prompt: str, seed: int) -> tuple[np.ndarray, dict]:
        from diffusers import FluxKontextPipeline, QwenImageEditPipeline
        path = local_model(self.config.editor_model_dir, 'editor' if self.config.editor_kind == 'flux' else 'editor_qwen')
        if self.editor is None:
            cls = FluxKontextPipeline if self.config.editor_kind == 'flux' else QwenImageEditPipeline
            self.editor = cls.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True).to(self.device)
        h, w = initial.shape[:2]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if self.config.editor_kind == 'flux':
            canvas, y0, x0, pad_h, pad_w, kw, kh = _prepare_flux_canvas(initial)
            with torch.inference_mode():
                raw_pil = self.editor(
                    image=Image.fromarray(canvas), prompt=prompt,
                    num_inference_steps=self.config.anchor_steps, guidance_scale=self.config.anchor_guidance_scale,
                    height=kh, width=kw, max_area=kh * kw, _auto_resize=False, generator=generator,
                ).images[0]
            output = self._crop_editor_output(raw_pil, pad_w, pad_h, x0, y0, w, h)
            init_ssim, init_mae = _measure_anchor_alignment(output, initial)
            audit: dict[str, Any] = {
                'backend': self.config.editor_kind, 'model': path, 'precision': 'bfloat16',
                'steps': self.config.anchor_steps, 'prompt': prompt, 'seed': seed,
                'bucket_size': [kw, kh], 'canvas_size': [pad_w, pad_h],
                'initial_pass_ssim': init_ssim, 'initial_pass_mae': init_mae,
                'alignment_mode': 'full_kontext',
            }
            if (init_ssim < 0.85 or init_mae > 15.0) and hasattr(self.editor, 'prepare_latents'):
                output, sdedit_steps = self._flux_sdedit_refine(
                    canvas, prompt, seed, kw, kh, x0, y0, pad_w, pad_h, w, h, strength=0.45,
                )
                ref_ssim, ref_mae = _measure_anchor_alignment(output, initial)
                audit.update(
                    steps=sdedit_steps, alignment_mode='flow_matching_sdedit',
                    sdedit_strength=0.45, refined_ssim=ref_ssim, refined_mae=ref_mae,
                )
            return output, audit
        # 非 FLUX 编辑器保持原有尺寸映射；回到原始分辨率后再回写印刷核心。
        ratio = self.config.latent_working_size / max(h, w)
        height = max(64, round(h * ratio / 16) * 16)
        width = max(64, round(w * ratio / 16) * 16)
        with torch.inference_mode():
            raw_pil = self.editor(image=Image.fromarray(initial), prompt=prompt,
                num_inference_steps=self.config.anchor_steps, guidance_scale=self.config.anchor_guidance_scale,
                height=height, width=width, generator=generator).images[0]
        output = np.asarray(raw_pil.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)).copy()
        return output, {'backend': self.config.editor_kind, 'model': path, 'precision': 'bfloat16',
                        'steps': self.config.anchor_steps, 'prompt': prompt, 'seed': seed}

    def gate(self, source: Path, candidate: np.ndarray, category: str) -> dict:
        prompt = ('Compare public source product photograph (first) and edited photograph (second). Category: '+category+
            '. Judge physical product identity and shape preservation, intact printed labels, realistic smooth lighting, '
            'absence of patchy pale islands, seams, halos, distortions or invented objects. '
            'Return JSON only with passed (boolean) and reasons (list of strings). Do not judge retrieval or attractiveness.')
        result, raw = self._ask([Image.open(source).convert('RGB'), Image.fromarray(candidate)], prompt)
        if type(result.get('passed')) is not bool or not isinstance(result.get('reasons'), list):
            raise ValueError('自然度门禁未返回有效的布尔判断及原因')
        return dict(result, raw_response=raw, model='Qwen2.5-VL-32B-Instruct', precision='bfloat16')
