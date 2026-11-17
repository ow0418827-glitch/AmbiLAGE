from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import hashlib
import json
import math
import os
from src.paths import ProjectPath as Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from .spec import BackboneConfig
from src.visual.io import sha256_file, stable_key


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class ImageFeatures:
    global_embedding: torch.Tensor
    patch_tokens: torch.Tensor


def resolve_device(requested: str, *, allow_fallback: bool = False) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        if allow_fallback:
            return torch.device("cpu")
        raise RuntimeError("配置要求使用图形处理器，但当前张量框架未检测到可用设备。")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            if allow_fallback:
                return torch.device("cpu")
            raise RuntimeError(f"配置要求的图形处理器序号不存在：{device.index}")
    return device


def _huggingface_hub_roots(cache_dir: str | Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if cache_dir:
        candidate = Path(cache_dir)
        roots.append(candidate if candidate.name == "hub" else candidate / "hub")
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(variable):
            roots.append(Path(os.environ[variable]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("TRANSFORMERS_CACHE"):
        roots.append(Path(os.environ["TRANSFORMERS_CACHE"]))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def resolve_hf_cached_weight(
    hf_repo: str | None,
    filename: str = "open_clip_model.safetensors",
    cache_dir: str | Path | None = None,
) -> Path | None:
    if not hf_repo:
        return None
    repository_name = f"models--{hf_repo.replace('/', '--')}"
    filenames = (("open_clip_model.safetensors", "open_clip_pytorch_model.safetensors")
                 if filename == "open_clip_model.safetensors" else (filename,))
    matches: list[Path] = []
    for hub_root in _huggingface_hub_roots(cache_dir):
        repository = hub_root / repository_name
        if not repository.exists():
            continue
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            for asset_name in filenames:
                candidate = repository / "snapshots" / revision / asset_name
                if candidate.is_file():
                    return candidate
        snapshots = repository / "snapshots"
        if snapshots.exists():
            for asset_name in filenames:
                matches.extend(path for path in snapshots.glob(f"*/{asset_name}") if path.is_file())
    return sorted(matches, key=str)[-1] if matches else None


def resolve_hf_snapshot(hf_repo: str, cache_dir: str | Path | None = None) -> Path | None:
    repository_name = f"models--{hf_repo.replace('/', '--')}"
    candidates: list[Path] = []
    for hub_root in _huggingface_hub_roots(cache_dir):
        repository = hub_root / repository_name
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            snapshot = repository / "snapshots" / revision
            if snapshot.is_dir():
                return snapshot.resolve()
        snapshots = repository / "snapshots"
        if snapshots.is_dir():
            candidates.extend(path.resolve() for path in snapshots.iterdir() if path.is_dir())
    return sorted(candidates, key=str)[-1] if candidates else None


def _infer_image_size(model: torch.nn.Module) -> int:
    image_size = getattr(getattr(model, "visual", None), "image_size", 224)
    if isinstance(image_size, Iterable) and not isinstance(image_size, (str, bytes)):
        return int(tuple(image_size)[0])
    return int(image_size)


def _revision_from_weight_path(path: Path | None) -> str | None:
    if path is None:
        return None
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None

def _is_safetensors_file(path: str | Path | None) -> bool:
    if not path:
        return False
    path_obj = Path(path)
    if path_obj.name.endswith(".safetensors") or str(path).endswith(".safetensors"):
        return True
    try:
        if path_obj.is_file():
            with open(path_obj, "rb") as f:
                header = f.read(9)
            if len(header) >= 9:
                import struct
                header_size = struct.unpack("<Q", header[:8])[0]
                if 0 < header_size < 100 * 1024 * 1024 and header[8:9] == b"{":
                    return True
    except Exception:
        pass
    return False


def pretrained_settings(model_name, pretrained):
    """本地权重文件也须继承其声明的预训练结构，不能只继承模型名。"""
    import open_clip
    from open_clip.pretrained import get_pretrained_cfg
    architecture = open_clip.get_model_config(model_name)
    if architecture is None:
        raise ValueError('未知视觉编码器结构：' + model_name)
    registered = get_pretrained_cfg(model_name, pretrained)
    mean = tuple(float(v) for v in (registered.get('mean', CLIP_MEAN) if registered else CLIP_MEAN))
    std = tuple(float(v) for v in (registered.get('std', CLIP_STD) if registered else CLIP_STD))
    if len(mean) != 3 or len(std) != 3 or not all(math.isfinite(v) for v in (*mean, *std)) or min(std) <= 0:
        raise ValueError('预训练图像归一化参数无效')
    # Activation has no trainable tensor: strict loading alone cannot detect a mismatch.
    # OpenAI tags require QuickGELU; LAION/SigLIP retain their native architecture.
    quick_gelu = bool(registered.get('quick_gelu', architecture.get('quick_gelu', False))
                      if registered else architecture.get('quick_gelu', False))
    return dict(architecture=architecture, quick_gelu=quick_gelu,
                activation='QuickGELU' if quick_gelu else 'native', mean=mean, std=std,
                metadata_source='registered_pretrained_tag' if registered else 'explicit_configuration',
                registered_preprocess={k: registered[k] for k in ('interpolation', 'resize_mode') if registered and k in registered})


def load_local_clip_model(model_name, pretrained, weight_path):
    """只创建结构并严格载入本地参数；拒绝缺失参数、额外参数和不安全回退。"""
    import open_clip
    settings = pretrained_settings(model_name, pretrained)
    path = Path(weight_path)
    if not path.is_file():
        raise FileNotFoundError('本地视觉编码器权重不存在：' + str(path))
    if _is_safetensors_file(path):
        from safetensors.torch import load_file
        state = load_file(str(path), device='cpu')
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v) for k,v in state.items()):
        raise ValueError('视觉编码器权重必须是非空参数字典')
    state = {k.removeprefix('module.'): v for k,v in state.items()}
    model = open_clip.create_model(model_name, pretrained=None,
                                  force_quick_gelu=settings['quick_gelu'],
                                  pretrained_image=False, pretrained_text=False)
    model.load_state_dict(state, strict=True)
    actual = {name: sum(type(m).__name__ == name for m in model.modules())
              for name in ('QuickGELU', 'GELU')}
    if settings['quick_gelu'] and actual['QuickGELU'] == 0:
        raise RuntimeError('预训练标签要求快速激活，但实际模型未采用该激活')
    if not settings['quick_gelu'] and actual['QuickGELU']:
        raise RuntimeError('原生普通激活权重被错误替换为快速激活')
    settings.update(strict_state_dict=True, actual_activation_modules=actual,
                    loader_version=3, loader_sha256=sha256_file(__file__))
    return model, settings


def tokenizer_identity(tokenizer):
    """Bind actual vocabulary/merges and tokenizer normalization, not its path."""
    backend = getattr(tokenizer, 'tokenizer', None)
    fast = getattr(backend, 'backend_tokenizer', None)
    if fast is not None and callable(getattr(fast, 'to_str', None)):
        payload = fast.to_str()
    elif hasattr(tokenizer, 'encoder') and hasattr(tokenizer, 'bpe_ranks'):
        payload = json.dumps({'vocabulary': sorted(tokenizer.encoder.items()),
            'merges': sorted((list(k), v) for k, v in tokenizer.bpe_ranks.items())}, ensure_ascii=False)
    elif backend is not None and callable(getattr(backend, 'get_vocab', None)):
        payload = json.dumps({'vocabulary': sorted(backend.get_vocab().items()),
            'settings': getattr(backend, 'init_kwargs', {})}, sort_keys=True, default=str)
    else:
        raise ValueError('无法核对实际分词器词表，拒绝建立不完整缓存身份')
    special = {name: getattr(tokenizer, name, None)
               for name in ('sot_token_id', 'eot_token_id', 'pad_token_id', 'context_length')}
    return {'type': type(tokenizer).__name__, 'special_tokens': special,
        'content_sha256': hashlib.sha256(payload.encode('utf-8')).hexdigest()}


class OpenCLIPSurrogate:
    """参考项目代理模型的独立兼容包装，并显式拒绝局部特征静默回退。"""

    def __init__(
        self,
        config: BackboneConfig,
        *,
        device: str = "cuda",
        precision: str = "fp32",
        allow_downloads: bool = False,
        allow_device_fallback: bool = False,
        require_true_local_tokens: bool = True,
        hash_weight: bool = True,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                f"无法导入 open_clip_torch 或其运行依赖：{exc}"
            ) from exc
        if precision != "fp32":
            raise ValueError("首轮攻击代理只允许 fp32 精度。")
        self.config = config
        self.device = resolve_device(device, allow_fallback=allow_device_fallback)
        self.precision = precision
        self.require_true_local_tokens = bool(require_true_local_tokens)

        weight_path: Path | None = None
        if config.weight_path:
            weight_path = Path(config.weight_path)
            if not weight_path.is_file():
                raise FileNotFoundError(f"显式指定的代理权重不存在：{weight_path}")
        else:
            weight_path = resolve_hf_cached_weight(config.hf_repo)
        if weight_path is None:
            if not allow_downloads:
                raise FileNotFoundError(
                    f"没有找到 {config.name}/{config.pretrained} 的本地冻结权重；"
                    "当前配置禁止自动下载。请先显式准备权重。")
            from open_clip.pretrained import download_pretrained, get_pretrained_cfg
            registered = get_pretrained_cfg(config.name, config.pretrained)
            if not registered:
                raise ValueError('无法下载未知预训练标签')
            weight_path = Path(download_pretrained(registered))
        self.model, self.loading = load_local_clip_model(
            config.name, config.pretrained, weight_path)
        self.model.eval().to(self.device, dtype=torch.float32)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        text_cfg = self.loading['architecture'].get('text_cfg', {})
        if text_cfg.get('hf_tokenizer_name'):
            # A local weight file does not imply a local tokenizer. Do not access
            # the network behind the caller's allow_downloads=False contract.
            from open_clip.tokenizer import HFTokenizer
            self.tokenizer = HFTokenizer(
                config.tokenizer_path or text_cfg['hf_tokenizer_name'],
                context_length=text_cfg.get('context_length', 77),
                local_files_only=not allow_downloads,
                **text_cfg.get('tokenizer_kwargs', {}))
        else:
            self.tokenizer = open_clip.get_tokenizer(config.name)
        self.tokenizer_audit = tokenizer_identity(self.tokenizer)
        self.image_size = _infer_image_size(self.model)
        self.mean = torch.tensor(
            self.loading['mean'], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            self.loading['std'], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self.weight_path = weight_path
        self.weight_sha256 = sha256_file(weight_path) if weight_path and hash_weight else None
        self.weight_revision = _revision_from_weight_path(weight_path)
        self._open_clip_version = importlib.metadata.version("open-clip-torch")
        self._cache_identity = stable_key(
            (
                config.name,
                config.pretrained,
                config.hf_repo or "",
                str(weight_path or "downloaded-by-open-clip"),
                self.weight_sha256 or "hash-not-recorded",
                self.weight_revision or "revision-unavailable",
                precision,
                str(self.image_size),
                json.dumps(self.loading, sort_keys=True),
                json.dumps(self.tokenizer_audit, sort_keys=True),
                self._open_clip_version,
            )
        )

    @property
    def cache_identity(self) -> str:
        return self._cache_identity

    def visual_parameter_bytes(self) -> int:
        visual = getattr(self.model, "visual", None)
        if visual is None or not hasattr(visual, "parameters"):
            return 0
        return sum(p.numel() * p.element_size() for p in visual.parameters())

    def enable_visual_grad_checkpointing(self, enabled: bool = True) -> bool:
        visual = getattr(self.model, "visual", None)
        for module in (visual, self.model):
            if module is not None and callable(getattr(module, "set_grad_checkpointing", None)):
                try:
                    module.set_grad_checkpointing(bool(enabled))
                    return True
                except Exception:
                    continue
        return False

    def offload_text_tower(self) -> int:
        """在文本特征全部预编码冻结后，将不参与图像前/反向传播的文本塔移至 CPU，释放 L4 显存。"""
        if self.device.type != "cuda" or not hasattr(self.model, "named_children"):
            return 0
        freed = 0
        for name, child in self.model.named_children():
            if name == "visual":
                continue
            freed += sum(p.numel() * p.element_size() for p in child.parameters() if p.device.type == "cuda")
            child.to("cpu")
        for pname, param in self.model.named_parameters(recurse=False):
            if param.device.type == "cuda":
                freed += param.numel() * param.element_size()
                param.data = param.data.to("cpu")
        for bname, buf in self.model.named_buffers(recurse=False):
            if buf is not None and buf.device.type == "cuda":
                freed += buf.numel() * buf.element_size()
                buf.data = buf.data.to("cpu")
        self._text_offloaded = True
        return freed

    def restore_text_tower(self) -> None:
        if not getattr(self, "_text_offloaded", False):
            return
        for name, child in self.model.named_children():
            if name != "visual":
                child.to(self.device)
        for _, param in self.model.named_parameters(recurse=False):
            param.data = param.data.to(self.device)
        for _, buf in self.model.named_buffers(recurse=False):
            if buf is not None:
                buf.data = buf.data.to(self.device)
        self._text_offloaded = False

    def move_visual_to(self, target_device: torch.device | str) -> None:
        dev = torch.device(target_device)
        visual = getattr(self.model, "visual", None)
        if visual is not None:
            visual.to(dev)
        self.mean = self.mean.to(dev)
        self.std = self.std.to(dev)
        self._active_visual_device = dev

    def snapshot(self) -> dict[str, Any]:
        return {
            "role": "attack_surrogate",
            "name": self.config.name,
            "pretrained": self.config.pretrained,
            "hf_repo": self.config.hf_repo,
            "tokenizer_path": self.config.tokenizer_path,
            "weight_path": str(self.weight_path) if self.weight_path else None,
            "weight_sha256": self.weight_sha256,
            "weight_revision": self.weight_revision,
            "device": str(self.device),
            "precision": self.precision,
            "image_size": self.image_size,
            "preprocess": {
                "resize": "bicubic_direct_square",
                "align_corners": False,
                "antialias": True,
                "mean": list(self.loading['mean']),
                "std": list(self.loading['std']),
            },
            "open_clip_torch_version": self._open_clip_version,
            "cache_identity": self.cache_identity,
            "requires_true_local_tokens": self.require_true_local_tokens,
            "loading": self.loading,
            "tokenizer": self.tokenizer_audit,
        }

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("代理模型输入必须具有 [批, 3, 高, 宽] 形状。")
        active_dev = getattr(self, "_active_visual_device", self.device)
        image = images.to(active_dev, dtype=torch.float32)
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = F.interpolate(
                image,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (image - self.mean.to(active_dev)) / self.std.to(active_dev)

    def encode_image(self, images: torch.Tensor) -> ImageFeatures:
        normalized = self.preprocess(images)
        visual = self.model.visual
        if not hasattr(visual, "forward_intermediates"):
            if self.require_true_local_tokens:
                raise RuntimeError(
                    f"代理 {self.config.name}/{self.config.pretrained} 没有局部特征接口，兼容实验已中止。"
                )
            global_embedding = F.normalize(self.model.encode_image(normalized), dim=-1)
            return ImageFeatures(global_embedding, global_embedding[:, None, :])

        try:
            output = visual.forward_intermediates(
                normalized,
                indices=[-1],
                output_fmt="NLC",
                output_extra_tokens=True,
            )
            if not isinstance(output, dict):
                raise RuntimeError("当前 open_clip_torch 返回了未知的局部特征结构。")
            raw_feat = output.get("image_features")
            if raw_feat is None and not self.require_true_local_tokens:
                raw_feat = self.model.encode_image(normalized)
            if raw_feat is None or "image_intermediates" not in output:
                raise KeyError("缺少 image_features 或 image_intermediates")
            global_embedding = F.normalize(raw_feat, dim=-1)
            patch_tokens = F.normalize(output["image_intermediates"][-1], dim=-1)
        except (KeyError, IndexError, TypeError, RuntimeError) as exc:
            if self.require_true_local_tokens:
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError("代理模型局部特征接口缺少预期字段。") from exc
            global_embedding = F.normalize(self.model.encode_image(normalized), dim=-1)
            return ImageFeatures(global_embedding, global_embedding[:, None, :])
        if patch_tokens.ndim != 3:
            raise RuntimeError(f"代理模型局部特征形状异常：{tuple(patch_tokens.shape)}")
        if self.require_true_local_tokens and patch_tokens.shape[1] <= 1:
            raise RuntimeError("代理模型只返回了一个全局向量，不能冒充局部图块特征。")
        return ImageFeatures(global_embedding, patch_tokens)

    @torch.no_grad()
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        self.restore_text_tower()
        texts = [text] if isinstance(text, str) else text
        context = int(getattr(self.tokenizer, 'context_length', 77))
        encode = getattr(self.tokenizer, 'encode', None)
        if not callable(encode):
            encode = lambda value: self.tokenizer.tokenizer.encode(
                getattr(self.tokenizer, 'clean_fn', lambda text: text)(value), add_special_tokens=False)
        special = int(getattr(self.tokenizer, 'sot_token_id', None) is not None) + 1
        if any(len(encode(t)) + special > context for t in texts):
            raise ValueError('文字目标超过模型容量，禁止静默截断')
        tokens = self.tokenizer(texts).to(self.device)
        return F.normalize(self.model.encode_text(tokens), dim=-1)

