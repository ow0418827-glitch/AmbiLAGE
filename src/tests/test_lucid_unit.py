"""Lucid 模块规范、配置校验与 A100 硬件画像单元测试。"""
from __future__ import annotations

import pytest
import torch

from src.visual._lucid import (
    AttackConfig,
    LucidConfig,
    ProjectConfig,
    generate_attack,
    generate_lucid,
    run_attack,
)
from src.visual.hardware import (
    apply_hardware_profile,
    configure_tf32,
    get_device_vram_gb,
    should_enable_gradient_checkpointing,
)


def test_lucid_config_and_aliases():
    """测试 Lucid 配置模型与其别名支持。"""
    attack_cfg = AttackConfig(epsilon=8 / 255, reference_steps=100)
    attack_cfg.validate()
    assert attack_cfg.epsilon == 8 / 255
    assert attack_cfg.reference_steps == 100

    # 验证 LucidConfig 与 ProjectConfig 等价别名
    assert LucidConfig is ProjectConfig

    # 验证 allow_tf32 属性存在
    proj_cfg = ProjectConfig(attack=attack_cfg, surrogates=(), allow_tf32=True)
    assert proj_cfg.allow_tf32 is True

    # 验证导出入口
    assert callable(generate_lucid)
    assert callable(generate_attack)
    assert callable(run_attack)


def test_hardware_profile_tf32_and_vram():
    """测试 A100 硬件画像配置 (TF32, 梯度检查点)。"""
    # 1. TF32 配置测试
    tf32_status = configure_tf32(True)
    if torch.cuda.is_available():
        assert tf32_status is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    else:
        assert tf32_status is False

    # 2. VRAM 梯度检查点自适应测试
    # 显式 override 优先
    assert should_enable_gradient_checkpointing("cpu", override=True) is True
    assert should_enable_gradient_checkpointing("cpu", override=False) is False

    # CPU 环境不启用检查点
    assert should_enable_gradient_checkpointing("cpu") is False

    # 3. 完整硬件画像应用测试
    profile = apply_hardware_profile("cpu", allow_tf32=True, seed=123)
    assert profile["seed"] == 123
    assert profile["device"] == "cpu"
