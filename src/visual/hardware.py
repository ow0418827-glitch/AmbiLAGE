"""Hardware configuration and measured device facts, without performance guarantees."""
from __future__ import annotations

import os
import torch
from typing import Any


def configure_tf32(enabled: bool = True) -> bool:
    """配置 PyTorch 的 TF32 模式。
    
    在 Ampere 及更新架构（如 A100）上，TF32 保持 8 位指数（与 FP32 相同动态范围），
    采用较短尾数，可能加速矩阵乘法；实际速度与数值差异须在远程实测。
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
        torch.backends.cudnn.allow_tf32 = bool(enabled)
        return bool(enabled)
    return False


def get_device_vram_bytes(device: torch.device | str | None = None) -> int:
    """获取指定 CUDA 设备的物理显存总量（字节数）。非 CUDA 设备返回 0。"""
    if not torch.cuda.is_available():
        return 0
    dev = torch.device(device if device is not None else "cuda")
    if dev.type != "cuda":
        return 0
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        return torch.cuda.get_device_properties(index).total_memory
    except Exception:
        return 0


def get_device_vram_gb(device: torch.device | str | None = None) -> float:
    """获取指定 CUDA 设备的物理显存总量（单位 GB）。"""
    return get_device_vram_bytes(device) / (1024.0 ** 3)


def should_enable_gradient_checkpointing(
    device: torch.device | str | None = None,
    threshold_gb: float = 32.0,
    override: bool | None = None,
) -> bool:
    """判断当前环境是否需要启用梯度检查点 (Gradient Checkpointing)。
    
    大显存设备默认关闭检查点；具体峰值显存仍取决于模型、工作尺寸及反向图。
    小显存设备默认启用检查点。此启发式不能保证避免显存不足。
    若传入显式 override，则优先遵循 override。
    """
    if override is not None:
        return bool(override)
    
    if not torch.cuda.is_available():
        return False
    
    dev = torch.device(device if device is not None else "cuda")
    if dev.type != "cuda":
        return False
    
    vram_bytes = get_device_vram_bytes(dev)
    threshold_bytes = int(threshold_gb * (1024 ** 3))
    
    # 显存大于门槛（如 32GB）则无需开启梯度检查点
    if vram_bytes > threshold_bytes:
        return False
    return True


def apply_hardware_profile(
    device: torch.device | str | None = None,
    *,
    allow_tf32: bool = True,
    seed: int | None = None,
) -> dict[str, Any]:
    """应用全局硬件画像与加速设置。"""
    profile: dict[str, Any] = {
        "tf32_enabled": False,
        "device": str(device or "cpu"),
        "vram_gb": 0.0,
        "gradient_checkpointing_recommended": False,
    }
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        profile["seed"] = seed
    
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    profile["device"] = str(dev)
    
    if dev.type == "cuda" and torch.cuda.is_available():
        tf32_ok = configure_tf32(allow_tf32)
        profile["tf32_enabled"] = tf32_ok
        vram_gb = get_device_vram_gb(dev)
        profile["vram_gb"] = round(vram_gb, 2)
        profile["gradient_checkpointing_recommended"] = should_enable_gradient_checkpointing(dev)
    
    return profile
