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


def parse_json_response(text: str) -> dict:
    value = text.strip()
    if value.startswith('```'):
        value = value.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError('视觉规划响应必须为对象')
    return result


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
        prompt = ('Inspect this public product photograph. Public category: ' + category +
            '. Return JSON only with source_boxes (list of normalized xyxy boxes covering the product), '
            'print_regions (normalized xyxy boxes covering ALL printed text, brand logos and labels), '
            'edit_prompt (a photographic relighting/background instruction preserving the exact product identity, '
            'shape, label and readable text; do not add brands or objects).')
        plan, raw = self._ask([Image.open(source).convert('RGB')], prompt)
        if not isinstance(plan.get('source_boxes'), list) or not plan['source_boxes']:
            raise ValueError('视觉规划未返回主体提示框')
        if not isinstance(plan.get('print_regions'), list) or not isinstance(plan.get('edit_prompt'), str) or not plan['edit_prompt'].strip():
            raise ValueError('视觉规划缺少印刷区声明或编辑指令')
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

    def generate_anchor(self, initial: np.ndarray, prompt: str, seed: int) -> tuple[np.ndarray, dict]:
        from diffusers import FluxKontextPipeline, QwenImageEditPipeline
        path = local_model(self.config.editor_model_dir, 'editor' if self.config.editor_kind == 'flux' else 'editor_qwen')
        cls = FluxKontextPipeline if self.config.editor_kind == 'flux' else QwenImageEditPipeline
        self.editor = cls.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True).to(self.device)
        h, w = initial.shape[:2]
        # 固定工作尺寸；回到原始分辨率后再回写印刷核心。
        ratio = self.config.latent_working_size / max(h, w)
        height = max(64, round(h * ratio / 16) * 16)
        width = max(64, round(w * ratio / 16) * 16)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            output = self.editor(image=Image.fromarray(initial), prompt=prompt,
                num_inference_steps=self.config.anchor_steps, guidance_scale=self.config.anchor_guidance_scale,
                height=height, width=width, generator=generator).images[0]
        output = np.asarray(output.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)).copy()
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
