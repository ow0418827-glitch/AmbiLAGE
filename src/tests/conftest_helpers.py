"""测试辅助工具模块。"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def make_test_image(
    path: Path | str,
    size: tuple[int, int] = (32, 32),
    color: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """创建用于测试的合成纯色图片。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(p, format="PNG")
    return p
