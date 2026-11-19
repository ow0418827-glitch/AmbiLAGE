"""Standalone, decoupled CLIP/SigLIP evaluation encoder with batched image inference.

This module intentionally does NOT import or instantiate `OpenCLIPSurrogate`
(`src/visual/encoders/backbone.py`) and does NOT call `forward_intermediates`
or compute spatial patch tokens used only during attack optimization.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.metadata
import json
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.experiments import shared_evaluation
from src.paths import MODEL_ROOT, PROJECT, PRIVATE, VISUAL_CONFIGS, ProjectPath as Path
from src.visual.audit import digest, read_json
from src.visual.encoders.backbone import (
    BackboneConfig,
    _infer_image_size,
    _revision_from_weight_path,
    load_local_clip_model,
    resolve_device,
    resolve_hf_cached_weight,
    tokenizer_identity,
)
from src.visual.encoders.joint import _text_digest, normalized, validated_image_weight
from src.visual.hardware import configure_tf32
from src.visual.image_utils import sha256_file


def _decode_rgb_uint8(path: str | Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.array(opened.convert('RGB'), dtype=np.uint8, copy=True)


class StandaloneEvalClipEncoder:
    """Evaluation-only CLIP/OpenCLIP/SigLIP encoder decoupled from attack surrogates."""

    supports_image_weight = True

    def __init__(
        self,
        spec: dict | BackboneConfig,
        *,
        device: str = 'cuda',
        batch_size: int = 16,
        num_workers: int = 4,
        weight_sha256: str | None = None,
        evaluation_protocol: dict | None = None,
    ) -> None:
        cfg = spec if isinstance(spec, BackboneConfig) else BackboneConfig.from_mapping(spec)
        if str(device).startswith('cuda'):
            configure_tf32(False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.device = resolve_device(device, allow_fallback=False)
        if not cfg.weight_path:
            raise ValueError('评估独立编码器必须显式指定本地权重路径')
        weight_path = Path(cfg.weight_path).resolve()
        if not weight_path.is_file():
            raise FileNotFoundError(f'缺少评估检索器的本地权重：{weight_path}')

        import open_clip

        self.model, self.loading = load_local_clip_model(cfg.name, cfg.pretrained, weight_path)
        self.model.eval().to(self.device, dtype=torch.float32)
        for param in self.model.parameters():
            param.requires_grad_(False)

        text_cfg = self.loading['architecture'].get('text_cfg', {})
        if text_cfg.get('hf_tokenizer_name'):
            from open_clip.tokenizer import HFTokenizer

            self.tokenizer = HFTokenizer(
                cfg.tokenizer_path or text_cfg['hf_tokenizer_name'],
                context_length=text_cfg.get('context_length', 77),
                local_files_only=True,
                **text_cfg.get('tokenizer_kwargs', {}),
            )
        else:
            self.tokenizer = open_clip.get_tokenizer(cfg.name)
        self.tokenizer_audit = tokenizer_identity(self.tokenizer)

        self.context_length = int(getattr(self.tokenizer, 'context_length', 77))
        self._backend_tokenizer = getattr(self.tokenizer, 'tokenizer', None)
        self._encode_tokens = getattr(self.tokenizer, 'encode', None)
        if not callable(self._encode_tokens) and self._backend_tokenizer is not None:
            self._encode_tokens = lambda text: self._backend_tokenizer.encode(
                getattr(self.tokenizer, 'clean_fn', lambda value: value)(text),
                add_special_tokens=False,
            )
        self.sot_token = getattr(self.tokenizer, 'sot_token_id', None)
        if self.sot_token is None and self._backend_tokenizer is not None:
            self.sot_token = getattr(self._backend_tokenizer, 'bos_token_id', None)
        self.eot_token = getattr(self.tokenizer, 'eot_token_id', None)
        if self.eot_token is None and self._backend_tokenizer is not None:
            self.eot_token = self._backend_tokenizer.eos_token_id
        self.pad_token = getattr(self.tokenizer, 'pad_token_id', None)
        if self.pad_token is None and self._backend_tokenizer is not None:
            self.pad_token = getattr(self._backend_tokenizer, 'pad_token_id', None)
        self.pad_token = 0 if self.pad_token is None else int(self.pad_token)
        self.special_tokens = int(self.sot_token is not None) + int(self.eot_token is not None)
        self.payload_tokens = self.context_length - self.special_tokens
        if self.payload_tokens <= 0 or not callable(self._encode_tokens):
            raise ValueError('独立评估编码器需要完整的非截断 tokenizer.encode')

        self.image_size = _infer_image_size(self.model)
        self.mean = torch.tensor(self.loading['mean'], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(self.loading['std'], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        resolved_weight_sha = weight_sha256 or sha256_file(weight_path)
        self.weight_sha256 = resolved_weight_sha
        self.batch_size = max(1, int(batch_size))
        self.num_workers = max(1, int(num_workers))
        self.closed = False
        self.last_encoding_audit = None

        self.snapshot = {
            'provider': 'clip',
            'loader': 'standalone_eval_clip',
            'role': 'evaluation_only_encoder',
            'decoupled_from_training_surrogate': True,
            'uses_forward_intermediates': False,
            'batch_size': self.batch_size,
            'name': cfg.name,
            'pretrained': cfg.pretrained,
            'hf_repo': cfg.hf_repo,
            'tokenizer_path': cfg.tokenizer_path,
            'weight_path': str(weight_path),
            'weight_sha256': resolved_weight_sha,
            'weight_revision': _revision_from_weight_path(weight_path),
            'device': str(self.device),
            'precision': 'float32',
            'image_size': self.image_size,
            'preprocess': {
                'resize': 'bicubic_direct_square',
                'align_corners': False,
                'antialias': True,
                'mean': list(self.loading['mean']),
                'std': list(self.loading['std']),
            },
            'open_clip_torch_version': importlib.metadata.version('open-clip-torch'),
            'loading': self.loading,
            'tokenizer_audit': self.tokenizer_audit,
            'fusion': 'equal_unit_image_plus_unit_full_text',
            'weighted_fusion': {
                'algorithm': 'unit_alpha_image_plus_one_minus_alpha_full_text_v1',
                'default_image_weight': 0.5,
                'equal_weight_path': 'exact_legacy_image_plus_text',
                'empty_text': 'image_only_except_text_only_weight_rejected',
            },
            'text_policy': {
                'algorithm': 'contiguous_nonoverlapping_raw_token_chunks_v1',
                'context_length': self.context_length,
                'payload_tokens_per_chunk': self.payload_tokens,
                'special_tokens_per_chunk': self.special_tokens,
                'coverage': 'all_token_ids_no_truncation',
                'aggregation': 'unit_token_count_weighted_mean_then_unit',
                'tokenizer': type(self.tokenizer).__name__,
            },
        }
        if evaluation_protocol is not None:
            self.snapshot['evaluation_protocol'] = evaluation_protocol

    def _assert_open(self) -> None:
        if self.closed:
            raise RuntimeError('StandaloneEvalClipEncoder is closed')

    def _preprocess_uint8(self, arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode='bicubic',
                align_corners=False,
                antialias=True,
            )
        return (tensor - self.mean) / self.std

    @torch.no_grad()
    def encode_batch(self, paths: Iterable[str | Path], *, batch_size: int | None = None) -> list[np.ndarray]:
        """Encode arbitrary mixed-resolution images in bounded GPU batches using CUDA bicubic preprocessing."""
        self._assert_open()
        path_list = [Path(p) for p in paths]
        if not path_list:
            return []
        bs = max(1, int(self.batch_size if batch_size is None else batch_size))
        outputs: list[np.ndarray] = []

        if self.num_workers > 1 and len(path_list) > 1:
            if bs == 1:
                window = min(32, max(8, self.num_workers * 4))
                with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                    for start in range(0, len(path_list), window):
                        chunk_paths = path_list[start:start + window]
                        for arr in pool.map(_decode_rgb_uint8, chunk_paths):
                            outputs.extend(self._forward_uint8_chunk([arr]))
            else:
                with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                    for start in range(0, len(path_list), bs):
                        chunk_paths = path_list[start:start + bs]
                        arrays = list(pool.map(_decode_rgb_uint8, chunk_paths))
                        outputs.extend(self._forward_uint8_chunk(arrays))
        else:
            for start in range(0, len(path_list), bs):
                chunk_paths = path_list[start:start + bs]
                arrays = [_decode_rgb_uint8(p) for p in chunk_paths]
                outputs.extend(self._forward_uint8_chunk(arrays))
        return outputs

    def _forward_uint8_chunk(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        prepped = [self._preprocess_uint8(arr) for arr in arrays]
        batch_tensor = prepped[0] if len(prepped) == 1 else torch.cat(prepped, dim=0)
        raw = self.model.encode_image(batch_tensor)
        vecs = F.normalize(raw.float(), dim=-1).cpu().numpy()
        result = [normalized(row) for row in vecs]
        del prepped, batch_tensor, raw, vecs
        return result

    @torch.no_grad()
    def encode(self, path: str | Path) -> np.ndarray:
        return self.encode_batch([path], batch_size=1)[0]

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        self._assert_open()
        text = str(text or ' ')
        ids = list(self._encode_tokens(text))
        chunks = [ids[start:start + self.payload_tokens] for start in range(0, len(ids), self.payload_tokens)] or [[]]
        weighted = None
        weights = []
        for start in range(0, len(chunks), 16):
            batch = chunks[start:start + 16]
            tokens = torch.full((len(batch), self.context_length), self.pad_token, dtype=torch.long, device=self.device)
            for row, chunk in enumerate(batch):
                body = ([self.sot_token] if self.sot_token is not None else []) + chunk
                if self.eot_token is not None:
                    body.append(self.eot_token)
                tokens[row, :len(body)] = torch.tensor(body, dtype=torch.long, device=self.device)
            features = self.model.encode_text(tokens).float()
            features = F.normalize(features, dim=-1).cpu().numpy()
            for chunk, vector in zip(batch, features):
                weight = max(1, len(chunk))
                weighted = vector * weight if weighted is None else weighted + vector * weight
                weights.append(weight)
        self.last_encoding_audit = {
            'text_sha256': _text_digest(text),
            'input_tokens': len(ids),
            'encoded_tokens': sum(len(chunk) for chunk in chunks),
            'chunks': len(chunks),
            'payload_tokens_per_chunk': self.payload_tokens,
            'chunk_token_counts': [len(chunk) for chunk in chunks],
            'truncated': False,
        }
        return normalized(weighted / sum(weights))

    def encode_joint(self, text: str, path: str | Path, image_weight: float = 0.5) -> np.ndarray:
        self._assert_open()
        weight = validated_image_weight(image_weight)
        if not str(text or '').strip():
            if weight == 0:
                raise ValueError('image_weight=0 requires nonempty text')
            self.last_encoding_audit = {
                'text_sha256': _text_digest(str(text or '')),
                'input_tokens': 0,
                'encoded_tokens': 0,
                'chunks': 0,
                'truncated': False,
            }
            return self.encode(path)
        if weight == 0:
            return self.encode_text(text)
        image = self.encode(path)
        if weight == 1:
            self.last_encoding_audit = {
                'text_sha256': _text_digest(str(text)),
                'text_skipped': True,
                'reason': 'image_weight_is_one',
                'encoded_tokens': 0,
                'chunks': 0,
                'truncated': False,
            }
            return image
        text_vector = self.encode_text(text)
        if weight == 0.5:
            return normalized(image + text_vector)
        return normalized(weight * image + (1 - weight) * text_vector)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BatchedFreshCachedEncoder(shared_evaluation.InProcessFreshCachedEncoder):
    """Cached fresh-encoder wrapper with deterministic batched pre-warming of unique images."""

    def prewarm_images(
        self,
        path_sha_pairs: Iterable[tuple[str | Path, str]],
        *,
        batch_size: int | None = None,
    ) -> int:
        unique_missing: dict[str, Path] = {}
        for raw_path, img_sha in path_sha_pairs:
            if img_sha not in self._image_cache and img_sha not in unique_missing:
                unique_missing[img_sha] = Path(raw_path).resolve()
        if not unique_missing:
            return 0

        ordered = sorted(unique_missing.items(), key=lambda kv: kv[0])
        ordered_paths = [path for _, path in ordered]
        if callable(getattr(self._base, 'encode_batch', None)):
            vectors = self._base.encode_batch(ordered_paths, batch_size=batch_size)
        else:
            vectors = [self._base.encode(p) for p in ordered_paths]

        for (img_sha, path), vec in zip(ordered, vectors):
            self._image_cache[img_sha] = vec
            st = path.stat()
            self._stat_hashes[(str(path), st.st_mtime_ns, st.st_size)] = img_sha
        return len(ordered)


def make_standalone_eval_encoder(
    device: str = 'cuda',
    config_path: str | Path | None = None,
    *,
    training_surrogates=None,
    batch_size: int = 16,
    num_workers: int = 4,
) -> StandaloneEvalClipEncoder:
    """Create StandaloneEvalClipEncoder with full anti-leakage and identity verification."""
    if str(device).startswith('cuda'):
        configure_tf32(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    private = PRIVATE / 'evaluator.json'
    settings_path = Path(
        config_path or (private if private.is_file() else VISUAL_CONFIGS / 'evaluator.example.json')
    ).resolve()
    settings = read_json(settings_path)
    setting = settings.get('setting')
    if setting not in ('held_out_public_encoder', 'matched_public_encoder'):
        raise ValueError('评估编码器须显式标明独立留出或同代理评估设置')
    if setting == 'matched_public_encoder' and config_path is None:
        raise ValueError('同代理评估只能由本次实验显式配置启用')
    if settings.get('provider') == 'gme':
        raise ValueError('StandaloneEvalClipEncoder 仅支持 CLIP/OpenCLIP/SigLIP 编码器')

    actual_context = training_surrogates is not None
    training_specs = (
        shared_evaluation._actual_training_identities(training_surrogates)
        if actual_context
        else shared_evaluation._builtin_training_identities()
    )
    identity_scope = 'actual_frozen_surrogates' if actual_context else 'all_builtin_surrogate_configurations'
    identity_audit = dict(
        training_identity_scope=identity_scope,
        training_surrogates_sha256=digest(training_specs),
    )

    spec = dict(settings['surrogate'])
    matching = [
        i for i, m in enumerate(training_specs)
        if (m['name'], m['pretrained']) == (spec['name'], spec['pretrained'])
    ]
    if matching and setting == 'held_out_public_encoder':
        raise ValueError('正式留出检索器与训练代理相同，不能作为黑盒迁移评估')
    if not matching and setting == 'matched_public_encoder':
        raise ValueError('同代理评估配置与实际训练代理身份不符')

    if spec.get('weight_path'):
        raw_weight = Path(spec['weight_path'])
        if raw_weight.is_absolute():
            weight = raw_weight
        else:
            rel_settings = settings_path.parent / raw_weight
            rel_model_root = MODEL_ROOT / raw_weight
            weight = rel_settings if rel_settings.is_file() else rel_model_root
        if not weight.is_file() and weight.name in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors'):
            weight = next(
                (weight.parent / name for name in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors')
                 if (weight.parent / name).is_file()),
                weight,
            )
    else:
        from src.model_registry import resolve_model

        try:
            directory = Path(resolve_model('eval_g14')) if spec['name'] == 'ViT-g-14' else None
        except FileNotFoundError:
            directory = None
        weight = next(
            (directory / fn for fn in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors',
                                       'open_clip_pytorch_model.bin')
             if directory is not None and (directory / fn).is_file()),
            None,
        )
        if weight is None:
            weight = next(
                (p for fn in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors',
                              'open_clip_pytorch_model.bin')
                 if (p := resolve_hf_cached_weight(spec['hf_repo'], filename=fn)) is not None),
                None,
            )
        if weight is None and spec.get('hf_repo'):
            try:
                repo_dir = Path(resolve_model(spec['hf_repo']))
            except (FileNotFoundError, KeyError, ValueError):
                repo_dir = None
            weight = next(
                (repo_dir / fn for fn in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors',
                                          'open_clip_pytorch_model.bin')
                 if repo_dir is not None and (repo_dir / fn).is_file()),
                None,
            )

    if weight is None or not weight.is_file():
        raise FileNotFoundError('缺少评估检索器的本地权重，请先显式准备')

    chosen_hash = sha256_file(weight) if (actual_context or setting == 'matched_public_encoder') else None
    if actual_context or setting == 'matched_public_encoder':
        weight_matches = (
            [i for i, row in enumerate(training_specs) if row.get('weight_sha256') == chosen_hash]
            if actual_context
            else []
        )
        if weight_matches and setting == 'held_out_public_encoder':
            raise ValueError('正式留出检索器与训练代理使用相同权重，不能作为黑盒迁移评估')
        matching = [
            i for i in matching
            if (training_specs[i]['weight_sha256'] if actual_context
                else shared_evaluation._builtin_weight_sha256(training_specs[i])) == chosen_hash
        ]
        if setting == 'matched_public_encoder' and not matching:
            raise ValueError('同代理评估的实际权重与训练代理不符')

    spec['weight_path'] = str(weight.resolve())
    if spec.get('tokenizer_path') and not Path(spec['tokenizer_path']).is_absolute():
        rel_settings = (settings_path.parent / spec['tokenizer_path']).resolve()
        rel_model_root = (MODEL_ROOT / spec['tokenizer_path']).resolve()
        spec['tokenizer_path'] = str(rel_settings if rel_settings.exists() else rel_model_root)
    if not spec.get('tokenizer_path') and (weight.parent / 'tokenizer_config.json').is_file():
        spec['tokenizer_path'] = str(weight.parent)

    evaluation_protocol = dict(
        setting=setting,
        matches_training_surrogate=bool(matching),
        matching_training_surrogate_indices=matching,
        black_box_transfer_claimed=setting == 'held_out_public_encoder',
        tf32_enabled=False,
        precision='float32',
        **identity_audit,
    )
    return StandaloneEvalClipEncoder(
        spec,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        weight_sha256=chosen_hash,
        evaluation_protocol=evaluation_protocol,
    )
