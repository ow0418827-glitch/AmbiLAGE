"""统一视觉生成接口与结果规约 (Unified Visual Generation Pipeline)。
对外暴露统一的 GenerationResult 与 generate() 函数，无缝调度 ambilage 与 lucid 两种对等方法。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.visual.image_utils import sha256_file
from src.visual.optimization import generate_attack, normalize_method


@dataclass(frozen=True)
class GenerationResult:
    image_path: Path
    ready_path: Path
    metadata_path: Path
    sha256: str
    method: str
    metadata: dict[str, Any]


def generate(
    method: str,
    source: Path | str,
    target: Path | str,
    output_dir: Path | str,
    *,
    steps: int | None = None,
    device: str = "cuda",
    target_text: str | None = None,
    epsilon_levels: int = 16,
    text_weight: float = 1.0,
    allow_tf32: bool = True,
    seed: int = 42,
    config: Any = None,
    ambilage_config: Any = None,
    optimization_config: Any = None,
    surrogate_set: str | None = None,
    surrogate_config: str | Path | None = None,
    **kwargs: Any,
) -> GenerationResult:
    """统一视觉生成接口。
    
    参数:
        method: 'ambilage' 或 'lucid'
        source: 源商品图像路径
        target: 目标商品图像路径
        output_dir: 产物落盘目录
        steps: 优化步数 (ambilage 默认 350, lucid 默认 1000)
        device: 'cuda' 或 'cpu'
        target_text: 目标特点描述文本
        epsilon_levels: 扰动预算 (仅 lucid 使用，默认 16)
        text_weight: 文字对齐损失权重
        allow_tf32: 是否开启 A100 TensorFloat-32 原生加速
        seed: 随机种子
        config: 通用配置对象 (根据 method 自动注入)
    返回:
        GenerationResult(image_path, ready_path, metadata_path, sha256, method, metadata)
    """
    m = normalize_method(method)
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    amb_cfg = ambilage_config or (config if m == "ambilage" else None)
    opt_cfg = optimization_config or (config if m == "lucid" else None)

    img_path, meta = generate_attack(
        source=source,
        target=target,
        output_dir=out_dir,
        method=m,
        steps=steps,
        epsilon_levels=epsilon_levels,
        device=device,
        optimization_config=opt_cfg,
        ambilage_config=amb_cfg,
        target_text=target_text,
        text_weight=text_weight,
        allow_tf32=allow_tf32,
        seed=seed,**kwargs,
        surrogate_set=surrogate_set,surrogate_config=surrogate_config,
    )

    img_p = Path(img_path).resolve()
    ready_p = out_dir / "ready.json"
    metadata_p = out_dir / "metadata.json"
    if not metadata_p.exists():
        metadata_p = out_dir / "attack_metrics.json"

    digest = sha256_file(img_p) or meta.get("output_image_sha256", "")

    return GenerationResult(
        image_path=img_p,
        ready_path=ready_p,
        metadata_path=metadata_p,
        sha256=digest,
        method=m,
        metadata=meta,
    )
