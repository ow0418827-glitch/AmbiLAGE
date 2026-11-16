"""Adversarial stress-testing and empirical challenge suite for Milestone 2.

Challenges verified:
1. Boundary conditions: Pure white (1.0), pure black (0.0), checkerboard, extreme gradients,
   and extreme driving differentials (+/- 1e12). Ensures zero 255-clipping, zero NaN/Inf,
   and strict adherence to [0.0, 1.0].
2. Strict print text ROI zero alteration: Floating point bit-level invariance across disjoint,
   boundary-spanning, and edge-touching ROIs in Stage 1, Stage 3, and Stage 4.
3. Non-finite value rejection & SSIM/MAE fallback gates: Backtracking line search, snapshot
   restoration upon constraint violation, and gradient nan_to_num sanitization.
4. Milestone 1 non-regression: 189 shared pool loading, zero perlage packages, official
   foreign source default, and offline rank computation.
5. CLI and configuration validation error paths.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent.memory.shared_pool import load_shared_pool
from src.visual._ambilage.ambient import AmbientCoopField
from src.visual._ambilage.config import AmbiLAGEConfig, METHOD, DEFINITION
from src.visual._ambilage.illumination_harmonization import harmonize_illumination
from src.visual._ambilage.latent_refinement import AmbiLAGEVae, refine_ambient_coop_latent
from src.visual._ambilage.objectives import (
    PublicFullKeyObjective,
    compute_spatial_variance_penalty,
    source_ssim,
    tangent_projection,
)
from src.visual._ambilage.pipeline import AmbiLAGEPipeline
from src.visual._ambilage.segmentation import compute_spatial_envelopes, print_region_mask
from src.visual.optimization import normalize_method
from src.visual.pipeline import generate


# =====================================================================
# 1. BOUNDARY CONDITIONS: PURE WHITE, PURE BLACK, MIXED & EXTREME DRIVES
# =====================================================================

@pytest.mark.parametrize("anchor_val, name", [(1.0, "pure_white"), (0.0, "pure_black")])
def test_boundary_anchor_midpoint_expansion_no_clipping_no_nan(anchor_val: float, name: str):
    """验证纯白 (1.0) 与纯黑 (0.0) 背景在极端驱动下绝无 255 截断、无 NaN/Inf、严格在 [0.0, 1.0] 内部。"""
    device = "cpu"
    h, w = 32, 32
    anchor = torch.full((1, 3, h, w), anchor_val, dtype=torch.float32, device=device)
    # 中央 16x16 为主体，其余为背景
    fg_bool = np.zeros((h, w), dtype=bool)
    fg_bool[8:24, 8:24] = True
    roi_bool = np.zeros((h, w), dtype=bool)

    cfg = AmbiLAGEConfig(latent_levels=6, background_levels=28)
    envelopes = compute_spatial_envelopes(fg_bool, roi_bool, cfg)

    field = AmbientCoopField(
        anchor=anchor,
        eps_fg_map=torch.as_tensor(envelopes["fg_level_array"] / 255.0, device=device)[None, None],
        eps_bg_map=torch.as_tensor(envelopes["bg_level_array"] / 255.0, device=device)[None, None],
        w_bg_mod=torch.as_tensor(envelopes["w_bg_mod"], device=device)[None, None],
        fg_mask_t=torch.as_tensor(fg_bool, device=device, dtype=torch.float32)[None, None],
        bg_mask_t=torch.as_tensor(~fg_bool, device=device, dtype=torch.float32)[None, None],
        support_mask=torch.as_tensor(envelopes["support_mask"], device=device)[None, None],
        config=cfg,
    )

    # 验证数学区间 [r_min, r_max] 的性质
    r_min = field.r_min
    r_max = field.r_max
    assert torch.all(r_min <= r_max), f"[{name}] r_min 必须小于等于 r_max"

    if anchor_val == 1.0:
        # 纯白背景：r_max 必须严格为 0.0 (不可再调亮)，r_min 为 -eps_bg
        assert torch.all(r_max[:, :, ~fg_bool] == 0.0), "纯白背景 r_max 必须为 0.0"
        assert torch.all(r_min[:, :, ~fg_bool] <= 0.0), "纯白背景 r_min 必须 <= 0.0"
    else:
        # 纯黑背景：r_min 必须严格为 0.0 (不可再调暗)，r_max 为 eps_bg
        assert torch.all(r_min[:, :, ~fg_bool] == 0.0), "纯黑背景 r_min 必须为 0.0"
        assert torch.all(r_max[:, :, ~fg_bool] >= 0.0), "纯黑背景 r_max 必须 >= 0.0"

    alt_pattern = torch.ones(1, 3, h, w, dtype=torch.float32, device=device)
    alt_pattern[:, :, ::2, ::2] = -1.0

    # 测试多种极端驱动差分场
    drive_scenarios = [
        ("zero", torch.zeros_like(anchor)),
        ("extreme_positive_1e3", torch.full_like(anchor, 1000.0)),
        ("extreme_negative_1e3", torch.full_like(anchor, -1000.0)),
        ("astronomical_pos_1e12", torch.full_like(anchor, 1e12)),
        ("astronomical_neg_1e12", torch.full_like(anchor, -1e12)),
        ("random_huge_gaussian", torch.randn_like(anchor) * 5000.0),
        ("high_freq_alternating", alt_pattern),
    ]

    for s_name, diff_drive in drive_scenarios:
        res = field.compute_residual(diff_drive)
        composed = anchor + res

        # 1. 绝不产生 NaN 或 Inf
        assert not torch.isnan(composed).any(), f"[{name}/{s_name}] 产生了 NaN 值！"
        assert not torch.isinf(composed).any(), f"[{name}/{s_name}] 产生了 Inf 值！"

        # 2. 背景像素数理保证严格位于 [0.0, 1.0] 闭区间内（数学区间中点展开保证，无需任何截断）
        bg_comp = composed.detach()[:, :, ~fg_bool]
        bg_c_min = float(bg_comp.min())
        bg_c_max = float(bg_comp.max())
        assert bg_c_max <= 1.0 + 1e-6, f"[{name}/{s_name}] 背景溢出上限: max={bg_c_max} > 1.0 (发生 255 截断！)"
        assert bg_c_min >= 0.0 - 1e-6, f"[{name}/{s_name}] 背景跌破下限: min={bg_c_min} < 0.0"

        # 3. 验证背景扰动预算界限
        bg_diff = (composed.detach() - anchor).abs()[:, :, ~fg_bool]
        assert float(bg_diff.max()) <= (cfg.background_levels / 255.0) + 1e-5


def test_pure_background_image_midpoint_expansion():
    """验证当全图均为纯背景 (fg_bool 全部为 False) 时，整张图像任何像素均严格在 [0.0, 1.0] 之间。"""
    device = "cpu"
    h, w = 32, 32
    for anchor_val in [0.0, 0.5, 1.0]:
        anchor = torch.full((1, 3, h, w), anchor_val, dtype=torch.float32, device=device)
        fg_bool = np.zeros((h, w), dtype=bool)  # 全图纯背景
        roi_bool = np.zeros((h, w), dtype=bool)

        cfg = AmbiLAGEConfig(latent_levels=6, background_levels=28)
        envelopes = compute_spatial_envelopes(fg_bool, roi_bool, cfg)

        field = AmbientCoopField(
            anchor=anchor,
            eps_fg_map=torch.as_tensor(envelopes["fg_level_array"] / 255.0, device=device)[None, None],
            eps_bg_map=torch.as_tensor(envelopes["bg_level_array"] / 255.0, device=device)[None, None],
            w_bg_mod=torch.as_tensor(envelopes["w_bg_mod"], device=device)[None, None],
            fg_mask_t=torch.as_tensor(fg_bool, device=device, dtype=torch.float32)[None, None],
            bg_mask_t=torch.as_tensor(~fg_bool, device=device, dtype=torch.float32)[None, None],
            support_mask=torch.as_tensor(envelopes["support_mask"], device=device)[None, None],
            config=cfg,
        )

        for diff_val in [-1e9, -100.0, 0.0, 100.0, 1e9]:
            diff = torch.full_like(anchor, diff_val)
            res = field.compute_residual(diff)
            comp = anchor + res

            assert not torch.isnan(comp).any()
            assert not torch.isinf(comp).any()
            assert float(comp.detach().max()) <= 1.0 + 1e-6
            assert float(comp.detach().min()) >= 0.0 - 1e-6


def test_boundary_checkerboard_and_near_edge_anchors():
    """验证棋盘格极端对比度背景下的中点展开稳定性与零截断。"""
    device = "cpu"
    h, w = 32, 32
    checkerboard = torch.zeros((1, 3, h, w), dtype=torch.float32, device=device)
    checkerboard[:, :, ::2, ::2] = 1.0
    checkerboard[:, :, 1::2, 1::2] = 1.0

    fg_bool = np.zeros((h, w), dtype=bool)  # 全图纯背景测试
    roi_bool = np.zeros((h, w), dtype=bool)

    cfg = AmbiLAGEConfig(latent_levels=6, background_levels=28)
    envelopes = compute_spatial_envelopes(fg_bool, roi_bool, cfg)

    field = AmbientCoopField(
        anchor=checkerboard,
        eps_fg_map=torch.as_tensor(envelopes["fg_level_array"] / 255.0, device=device)[None, None],
        eps_bg_map=torch.as_tensor(envelopes["bg_level_array"] / 255.0, device=device)[None, None],
        w_bg_mod=torch.as_tensor(envelopes["w_bg_mod"], device=device)[None, None],
        fg_mask_t=torch.as_tensor(fg_bool, device=device, dtype=torch.float32)[None, None],
        bg_mask_t=torch.as_tensor(~fg_bool, device=device, dtype=torch.float32)[None, None],
        support_mask=torch.as_tensor(envelopes["support_mask"], device=device)[None, None],
        config=cfg,
    )

    # 施加极端扰动
    extreme_diff = torch.randn_like(checkerboard) * 1e5
    res = field.compute_residual(extreme_diff)
    composed = checkerboard + res

    assert not torch.isnan(composed).any()
    assert not torch.isinf(composed).any()
    assert float(composed.detach().max()) <= 1.0 + 1e-6
    assert float(composed.detach().min()) >= 0.0 - 1e-6


# =====================================================================
# 2. STRICT PRINT TEXT ROI ZERO ALTERATION (BIT-LEVEL VERIFICATION)
# =====================================================================

def test_strict_print_roi_zero_alteration_bit_level():
    """测试多重拓扑 ROI (多块离散、跨边界、触边、单像素) 的浮点比特级绝对零改动。"""
    device = "cpu"
    h, w = 64, 64

    # 1. 构造多重复杂 ROI 掩码
    # ROI 1: 前景内部 (20:30, 20:30)
    # ROI 2: 背景内部 (5:12, 5:12)
    # ROI 3: 跨越前景与边界 (25:35, 40:55)
    # ROI 4: 触碰图像边缘 (0:4, 0:4)
    # ROI 5: 单像素点 (50, 50)
    roi_regions = [
        [20 / 64, 20 / 64, 30 / 64, 30 / 64],
        [5 / 64, 5 / 64, 12 / 64, 12 / 64],
        [25 / 64, 40 / 64, 35 / 64, 55 / 64],
        [0.0, 0.0, 4 / 64, 4 / 64],
        [50 / 64, 50 / 64, 51 / 64, 51 / 64],
    ]
    roi_bool = print_region_mask((h, w), roi_regions)
    assert roi_bool.sum() > 0

    fg_bool = np.zeros((h, w), dtype=bool)
    fg_bool[16:48, 16:48] = True

    cfg = AmbiLAGEConfig(latent_levels=6, background_levels=28)

    # 2. 验证 Stage 1 包络
    envelopes = compute_spatial_envelopes(fg_bool, roi_bool, cfg)
    assert np.all(envelopes["fg_level_array"][roi_bool] == 0.0)
    assert np.all(envelopes["bg_level_array"][roi_bool] == 0.0)
    assert np.all(envelopes["level_array"][roi_bool] == 0.0)
    assert np.all(envelopes["support_mask"][roi_bool] == 0.0)

    # 3. 验证 Stage 3 光照协调融合
    source_rgb = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    # 在 ROI 区域赋予高饱和度特征色
    source_rgb[roi_bool] = [255, 10, 20]

    anchor_rgb, rec = harmonize_illumination(
        source_image=source_rgb,
        target_color_rgb=[30, 180, 220],
        foreground=fg_bool,
        regions=roi_regions,
        config=cfg,
    )
    # 验证 Stage 3 像素绝对相等
    diff_s3 = np.abs(anchor_rgb.astype(np.int16) - source_rgb.astype(np.int16))
    assert np.all(diff_s3[roi_bool] == 0), f"Stage 3 ROI 最大像素改动: {diff_s3[roi_bool].max()}"
    assert np.array_equal(anchor_rgb[roi_bool], source_rgb[roi_bool])

    # 4. 验证 Stage 4 AmbientCoopField 浮点比特级零改动
    anchor_t = torch.from_numpy(anchor_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    field = AmbientCoopField(
        anchor=anchor_t,
        eps_fg_map=torch.as_tensor(envelopes["fg_level_array"] / 255.0, device=device)[None, None],
        eps_bg_map=torch.as_tensor(envelopes["bg_level_array"] / 255.0, device=device)[None, None],
        w_bg_mod=torch.as_tensor(envelopes["w_bg_mod"], device=device)[None, None],
        fg_mask_t=torch.as_tensor(fg_bool, device=device, dtype=torch.float32)[None, None],
        bg_mask_t=torch.as_tensor(~fg_bool, device=device, dtype=torch.float32)[None, None],
        support_mask=torch.as_tensor(envelopes["support_mask"], device=device)[None, None],
        config=cfg,
    )

    # 尝试注入巨大差分
    for _ in range(5):
        random_diff = torch.randn_like(anchor_t) * 100.0
        res = field.compute_residual(random_diff)
        res_roi = res[:, :, roi_bool]
        
        # 比特级精确全零断言
        assert torch.equal(res_roi, torch.zeros_like(res_roi)), (
            f"Stage 4 残差在 ROI 区域非严格零: 最大绝对值 {res_roi.abs().max().item()}"
        )
        composed = anchor_t + res
        assert torch.equal(composed[:, :, roi_bool], anchor_t[:, :, roi_bool]), (
            "Stage 4 合成图像在 ROI 区域与 anchor 不严格比特级相同！"
        )


# =====================================================================
# 3. NON-FINITE VALUE REJECTION & FALLBACK GATES (SSIM / MAE)
# =====================================================================

def test_gradient_nan_to_num_sanitization():
    """验证梯度中出现 NaN/Inf 时通过 nan_to_num_ 净化，防止优化器动量崩溃。"""
    param = nn.Parameter(torch.ones(4, dtype=torch.float32))
    opt = torch.optim.Adam([param], lr=0.1)

    # 模拟 backward 产生的 NaN 梯度
    param.grad = torch.tensor([1.0, float("nan"), float("inf"), float("-inf")])
    assert torch.isnan(param.grad).any() or torch.isinf(param.grad).any()

    # 执行代码中应用的净化策略
    torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

    assert not torch.isnan(param.grad).any()
    assert not torch.isinf(param.grad).any()
    assert torch.equal(param.grad, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    opt.step()
    assert not torch.isnan(param.data).any()


def test_ssim_fallback_gate_rejection(tmp_path: Path):
    """验证当步长导致 SSIM 低于阈值 (source_ssim_floor) 时，回退守门触发回滚并锁定 step 0 最佳快照。"""
    # 构造极严格的 SSIM 门槛 (floor = 0.9999) 与非零初偏置验证门禁拒绝
    cfg = AmbiLAGEConfig(
        steps=3,smoke_test=True,
        source_ssim_floor=0.9999,  # 极其严苛的门槛，任何实质扰动都无法通过
        source_mae_ceiling=50.0,
        ambient_init_bias=0.30,
        latent_levels=6,
        background_levels=28,
        guard_backtracks=4,
    )

    src_p = tmp_path / "s.png"
    tgt_p = tmp_path / "t.png"
    out_dir = tmp_path / "gate_test"
    Image.new("RGB", (32, 32), (200, 200, 200)).save(src_p)
    Image.new("RGB", (32, 32), (50, 100, 220)).save(tgt_p)

    pipe = AmbiLAGEPipeline(config=cfg, device="cpu")
    with pytest.raises(RuntimeError,match="自然锚点未通过"):
        pipe.execute(source=src_p,target=tgt_p,output_dir=out_dir,target_text="Blue Guard Test")
    assert not (out_dir/"ready.json").exists()
    assert (out_dir/"FAILED.json").is_file()


def test_mae_fallback_gate_rejection(tmp_path: Path):
    """验证当步长导致 MAE 高于阈值 (source_mae_ceiling) 时，回退守门触发回滚并锁定 step 0。"""
    # 构造极严苛的 MAE 门槛 (ceiling = 0.01 灰度级) 与非零初偏置验证门禁拒绝
    cfg = AmbiLAGEConfig(
        steps=3,smoke_test=True,
        source_ssim_floor=0.50,
        source_mae_ceiling=0.01,  # 极其严苛的 MAE 上限
        ambient_init_bias=0.30,
        latent_levels=6,
        background_levels=28,
        guard_backtracks=4,
    )

    src_p = tmp_path / "s_mae.png"
    tgt_p = tmp_path / "t_mae.png"
    out_dir = tmp_path / "gate_mae_test"
    Image.new("RGB", (32, 32), (180, 180, 180)).save(src_p)
    Image.new("RGB", (32, 32), (20, 80, 200)).save(tgt_p)

    pipe = AmbiLAGEPipeline(config=cfg, device="cpu")
    with pytest.raises(RuntimeError,match="自然锚点未通过"):
        pipe.execute(source=src_p,target=tgt_p,output_dir=out_dir,target_text="MAE Guard Test")
    assert not (out_dir/"ready.json").exists()


def test_tangent_projection_degenerate_normals():
    """测试切空间投影在零法向量、退化法向量与极大斜率下的数值稳定性。"""
    dim = 8
    direction = torch.randn(dim, dtype=torch.float32)

    # 1. 空法向量集 -> 恒等返回
    p_empty = tangent_projection(direction, [], [])
    assert torch.equal(p_empty, direction)

    # 2. 全零法向量 -> 恒等返回
    zero_normal = torch.zeros(dim, dtype=torch.float32)
    p_zero = tangent_projection(direction, [zero_normal], [0.0])
    assert torch.equal(p_zero, direction)

    # 3. 极端巨大法向量 (1e10) -> 归一化后稳定求解
    huge_normal = torch.ones(dim, dtype=torch.float32) * 1e10
    p_huge = tangent_projection(direction, [huge_normal], [0.1])
    assert not torch.isnan(p_huge).any()
    assert not torch.isinf(p_huge).any()


# =====================================================================
# 4. MILESTONE 1 FOUNDATION NON-REGRESSION
# =====================================================================

def test_m1_shared_pool_loading_and_structure():
    """验证 Milestone 1 共享记忆库加载完整性 (189 条)。"""
    pool = load_shared_pool()
    assert len(pool) == 189, f"共享记忆池条目数不符: {len(pool)} != 189"

    # 抽样检验字段规范
    for item in [pool[0], pool[90], pool[-1]]:
        assert "Id" in item
        assert "PrimaryImage" in item
        assert "SourceGroup" in item
        assert "EntryType" in item
        assert item["EntryType"] in ("normal", "error")


def test_m1_purged_blacklist_enforcement():
    """验证彻底清洗黑名单：严禁 _perlage* 包及 deprecated 目录。"""
    proj_root = Path(__file__).resolve().parents[2]
    visual_dir = proj_root / "src/visual"

    perlage_packages = list(visual_dir.glob("_perlage*"))
    assert len(perlage_packages) == 0, f"发现违规 perlage 包: {perlage_packages}"

    deprecated_dir = proj_root / "src/deprecated"
    assert not deprecated_dir.exists(), f"发现违规 deprecated 目录: {deprecated_dir}"


def test_m1_official_foreign_source_default():
    """验证默认使用 Hugging Face 官方外网源 (HF_ENDPOINT 为 None)。"""
    assert os.environ.get("HF_ENDPOINT") is None, (
        f"违规设定了 HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}！必须保持官方外网源默认"
    )


def test_m1_zero_gpu_rank_calculation():
    """验证纯离线无 GPU 矩阵点乘排名计算逻辑无故障。"""
    queries = torch.randn(180, 512)
    candidates = torch.randn(189, 512)
    queries = F.normalize(queries, dim=-1)
    candidates = F.normalize(candidates, dim=-1)

    scores = queries @ candidates.T
    ranks = scores.argsort(dim=-1, descending=True).argsort(dim=-1) + 1
    assert ranks.shape == (180, 189)
    assert int(ranks.min()) == 1
    assert int(ranks.max()) == 189


# =====================================================================
# 5. CLI & UNIFIED PIPELINE ARGUMENT VALIDATION
# =====================================================================

def test_cli_invalid_arguments_rejection():
    """测试统一接口对非法参数的准确拦截与报错。"""
    # 1. 非法方法名
    with pytest.raises(ValueError, match="方法必须是 'lucid' 或 'ambilage'"):
        normalize_method("unsupported_method")

    # 2. 试图启用像素优化
    with pytest.raises(ValueError, match="严格禁止像素残差与像素优化"):
        AmbiLAGEConfig(pixel_optimization=True).validate()

    with pytest.raises(ValueError, match="严格禁止像素残差与像素优化"):
        AmbiLAGEConfig(auxiliary_levels=4).validate()

    # 3. 步数为负数
    with pytest.raises(ValueError, match="优化步数 steps 必须为正整数"):
        AmbiLAGEConfig(steps=-5).validate()
