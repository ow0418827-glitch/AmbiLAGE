"""AmbiLAGE 原生四阶段算法、数理核心与空间包络单元测试。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.visual._ambilage.ambient import AmbientCoopField
from src.visual._ambilage.color_selection import extract_foreground_statistics
from src.visual._ambilage.config import AmbiLAGEConfig, METHOD, DEFINITION
from src.visual._ambilage.illumination_harmonization import harmonize_illumination
from src.tests.conftest_helpers import make_test_image
from src.visual._ambilage.objectives import (
    compute_spatial_variance_penalty,
    source_ssim,
    tangent_projection,
)
from src.visual._ambilage.pipeline import AmbiLAGEPipeline
from src.visual._ambilage.segmentation import (
    _resolve_box_coordinates,
    compute_spatial_envelopes,
    print_region_mask,
    segment_image,
)


def test_ambilage_config_validation():
    """测试配置校验与非法参数拦截。"""
    # 正常配置校验通过
    cfg = AmbiLAGEConfig()
    cfg.validate()
    assert cfg.latent_levels == 6
    assert cfg.background_levels == 28
    assert cfg.auxiliary_levels == 0
    assert cfg.pixel_optimization is False
    assert cfg.joint_weight == 1.0
    assert cfg.text_weight == 0.35
    assert cfg.local_ot_weight == 0.10
    assert cfg.fg_chroma_levels == 0
    assert cfg.fg_source_mae_ceiling == 10.0
    assert cfg.ambient_init_bias == 0.0
    assert cfg.halo_weight == 0.002
    assert cfg.bg_l2_weight == 0.005
    assert cfg.harmonization_selected_color_mix == 0.12
    assert cfg.harmonization_chroma_transfer_strength == 0.0
    assert cfg.harmonization_bg_strength == 0.85

    with pytest.raises(ValueError, match="背景棚拍底色融合强度"):
        AmbiLAGEConfig(harmonization_bg_strength=1.5).validate()

    # 试图启用像素优化或像素残差 -> 抛出异常
    with pytest.raises(ValueError, match="严格禁止像素残差与像素优化"):
        AmbiLAGEConfig(auxiliary_levels=2).validate()

    with pytest.raises(ValueError, match="严格禁止像素残差与像素优化"):
        AmbiLAGEConfig(pixel_optimization=True).validate()

    # 步数无效
    with pytest.raises(ValueError, match="优化步数 steps 必须为正整数"):
        AmbiLAGEConfig(steps=0).validate()

    # 边界模式无效
    with pytest.raises(ValueError, match="未知的边界过渡模式"):
        AmbiLAGEConfig(edge_feather_mode="invalid_mode").validate()


@pytest.mark.parametrize('kwargs', [
    {'text_weight': -0.1},
    {'text_weight': 2.5},
    {'text_key_mix': 1.2},
    {'local_ot_weight': -0.1},
    {'local_ot_clusters': 0},
    {'local_ot_clusters': 12},
    {'fg_source_mae_ceiling': 0.0},
    {'fg_chroma_levels': 80},
])
def test_adapted_text_and_local_ot_config_bounds(kwargs):
    with pytest.raises(ValueError):
        AmbiLAGEConfig.from_dict(kwargs)


@pytest.mark.parametrize('weight', [0., -1., float('inf'), float('nan')])
def test_joint_training_weight_must_be_finite_and_positive(weight):
    with pytest.raises(ValueError):
        AmbiLAGEConfig.from_dict({'joint_weight': weight})


def test_spatial_envelopes():
    """测试 Stage 1: 欧氏距离变换 (EDT) 与四重空间余弦平滑包络。"""
    h, w = 64, 64
    fg = np.zeros((h, w), dtype=bool)
    fg[16:48, 16:48] = True  # 中央 32x32 前景

    # 印刷区域定义在前景中央 8x8 区域
    roi_regions = [[28 / 64, 28 / 64, 36 / 64, 36 / 64]]
    roi = print_region_mask((h, w), roi_regions)
    assert roi.sum() == 8 * 8

    cfg = AmbiLAGEConfig(
        fg_edge_feather_pixels=8.0,
        edge_feather_pixels=12.0,
        ambient_feather_pixels=3.0,
        roi_feather_pixels=6.0,
    )
    envelopes = compute_spatial_envelopes(fg, roi, cfg)

    w_bg_budget = envelopes["w_bg_budget"]
    w_bg_mod = envelopes["w_bg_mod"]
    w_fg = envelopes["w_fg"]
    w_text = envelopes["w_text"]
    fg_levels = envelopes["fg_level_array"]
    bg_levels = envelopes["bg_level_array"]

    # 1. 前景区域 w_bg_budget 与 w_bg_mod 必为 0
    assert np.all(w_bg_budget[fg] == 0.0)
    assert np.all(w_bg_mod[fg] == 0.0)

    # 2. 紧邻轮廓外侧 w_bg_budget >= 0.70，远离轮廓处趋向 1.0
    assert np.all((w_bg_budget[~fg] >= 0.70) & (w_bg_budget[~fg] <= 1.0))
    # 紧邻轮廓外侧 w_bg_mod >= 0.25，远离轮廓处趋向 1.0
    assert np.all((w_bg_mod[~fg] >= 0.25) & (w_bg_mod[~fg] <= 1.0))

    # 3. 轮廓向内 w_fg 从 0.0 余弦过渡到 1.0
    assert np.all((w_fg[fg] >= 0.0) & (w_fg[fg] <= 1.0))
    assert np.all(w_fg[~fg] == 0.0)

    # 4. 印刷区域保护：ROI 内部扰动预算必须严格为 0
    assert np.all(fg_levels[roi] == 0.0)
    assert np.all(bg_levels[roi] == 0.0)
    assert np.all(envelopes["level_array"][roi] == 0.0)

    # 5. 主体非 ROI 区域预算受控在 latent_levels (6 级) 之内
    assert fg_levels.max() <= cfg.latent_levels + 1e-5
    # 背景区域预算受控在 background_levels (28 级) 之内
    assert bg_levels.max() <= cfg.background_levels + 1e-5


def test_statistical_color_selection(tmp_path: Path):
    """测试 Stage 2: 目标商品主体色彩统计与高光阴影排除。"""
    h, w = 64, 64
    # 构造含白色边框与纯蓝色主体的图像
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    # 蓝色主体主体
    img[16:48, 16:48] = [30, 80, 220]
    # 添加少量高光与阴影测试分位截断
    img[20, 20] = [255, 255, 255] # 高光
    img[21, 21] = [0, 0, 0]       # 阴影

    fg = np.zeros((h, w), dtype=bool)
    fg[16:48, 16:48] = True

    stats, body_mask = extract_foreground_statistics(img, fg, output_dir=tmp_path)
    assert stats["color_scheme"] == "statistics"
    rgb = stats["rgb"]
    assert len(rgb) == 3
    # 提取出的颜色应明显偏向蓝色 (B > R)
    assert rgb[2] > rgb[0]
    assert body_mask.sum() > 0
    assert (tmp_path / "record.json").is_file()


def test_illumination_harmonization_strict_roi_zero(tmp_path: Path):
    """测试 Stage 3: 亚光颜色融合与印刷文字保护区严格零改动断言。"""
    h, w = 64, 64
    source = np.full((h, w, 3), 240, dtype=np.uint8)
    fg = np.zeros((h, w), dtype=bool)
    fg[10:54, 10:54] = True
    source[fg] = [200, 200, 200]

    # 在中央设置印刷文字红色 ROI
    roi_regions = [[26 / 64, 26 / 64, 38 / 64, 38 / 64]]
    roi = print_region_mask((h, w), roi_regions)
    source[roi] = [255, 0, 0]  # 印刷红字

    cfg = AmbiLAGEConfig()
    target_color = [50, 150, 220]  # 目标淡蓝色

    anchor_rgb, record = harmonize_illumination(
        source_image=source,
        target_color_rgb=target_color,
        foreground=fg,
        regions=roi_regions,
        config=cfg,
        output_dir=tmp_path,
    )

    # 1. 验证印刷文字区严格零改动 (assert change[roi] == 0)
    change = np.abs(anchor_rgb.astype(np.int16) - source.astype(np.int16))
    assert np.all(change[roi] == 0), "印刷文字区发生改动！"
    assert np.array_equal(anchor_rgb[roi], source[roi])

    # 2. 验证默认配置下背景平滑过渡为温和浅暖棚拍底色 (摆脱 255 饱和死区)
    assert record["mean_bg_change_levels"] > 0.0
    assert float(anchor_rgb[~fg].mean()) <= 245.0
    assert float(anchor_rgb[~fg].mean()) >= 225.0

    # 3. 验证显式关闭背景调色 (harmonization_bg_strength=0.0) 时背景严格保持源图不变
    cfg_locked = AmbiLAGEConfig(harmonization_bg_strength=0.0)
    anchor_locked, rec_locked = harmonize_illumination(
        source_image=source,
        target_color_rgb=target_color,
        foreground=fg,
        regions=roi_regions,
        config=cfg_locked,
    )
    assert np.array_equal(anchor_locked[~fg], source[~fg])
    assert rec_locked["mean_bg_change_levels"] == 0.0

    # 4. 验证产物文件已落盘
    assert (tmp_path / "generative_anchor.png").is_file()
    assert (tmp_path / "initial.png").is_file()
    assert record["zero_pixel_residual"] is True


def test_ambient_field_no_clipping():
    """测试 Stage 4: 可行无截断中点对称展开 [r_min, r_max]，验证零 255 截断特性。"""
    device = "cpu"
    h, w = 32, 32
    # 模拟纯白背景电商图 (I_anchor == 1.0)
    anchor = torch.ones((1, 3, h, w), dtype=torch.float32, device=device)
    # 中央部分为主体 (0.5)
    anchor[:, :, 8:24, 8:24] = 0.5

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

    # 施加极端巨大的随机差分驱动值 (+1000.0, -1000.0)
    huge_diff = torch.randn_like(anchor) * 1000.0
    res = field.compute_residual(huge_diff)
    composed = anchor + res

    # 验证关键数学断言：
    # 1. 即使驱动值极大，合成结果也必须严格位于 [0.0, 1.0] 之间，绝无 255 截断！
    assert float(composed.detach().max()) <= 1.0 + 1e-6, f"越界超出 1.0: {composed.max()}"
    assert float(composed.detach().min()) >= 0.0 - 1e-6, f"越界低于 0.0: {composed.min()}"

    # 2. 背景变动幅度严格 <= background_levels / 255.0
    bg_t = torch.as_tensor(~fg_bool, device=device)
    bg_change = (composed.detach() - anchor).abs()[:, :, bg_t]
    assert float(bg_change.max()) <= (cfg.background_levels / 255.0) + 1e-5

    # 3. 主体变动幅度严格 <= latent_levels / 255.0
    fg_t = torch.as_tensor(fg_bool, device=device)
    fg_change = (composed.detach() - anchor).abs()[:, :, fg_t]
    assert float(fg_change.max()) <= (cfg.latent_levels / 255.0) + 1e-5


def test_spatial_variance_penalty_allows_mean_shift():
    """测试空间方差正则化：仅惩罚背景方差与 TV，绝不惩罚整体均值平移！"""
    device = "cpu"
    h, w = 32, 32
    anchor = torch.zeros((1, 3, h, w), dtype=torch.float32, device=device)
    bg_mask_t = torch.ones((1, 1, h, w), dtype=torch.float32, device=device)
    fg_mask_t = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)

    cfg = AmbiLAGEConfig(
        bg_l2_weight=1.0, bg_tv_weight=1.0, bg_chroma_tv_weight=1.0,
        fg_l2_weight=0.0, fg_tv_weight=0.0, overflow_weight=0.0,
    )

    # 情况 A: 整体平移图像 (全图均匀偏移 10/255)
    uniform_shift = anchor + (10.0 / 255.0)
    pen_uniform, met_uniform = compute_spatial_variance_penalty(
        uniform_shift, anchor, None, fg_mask_t, bg_mask_t, cfg
    )
    # 因为全图均匀无空间方差、无 TV，所以 bg_l2 和 bg_tv 严格为 0！
    assert met_uniform["bg_l2"] == pytest.approx(0.0, abs=1e-7)
    assert met_uniform["bg_tv"] == pytest.approx(0.0, abs=1e-7)
    assert met_uniform["bg_chroma_tv"] == pytest.approx(0.0, abs=1e-7)
    assert pen_uniform.item() == pytest.approx(0.0, abs=1e-7)

    # 情况 B: 存在高频波动的图像 (存在方差与 TV)
    noisy_img = anchor.clone()
    noisy_img[:, :, ::2, ::2] = 10.0 / 255.0
    pen_noisy, met_noisy = compute_spatial_variance_penalty(
        noisy_img, anchor, None, fg_mask_t, bg_mask_t, cfg
    )
    assert met_noisy["bg_l2"] > 0.0
    assert met_noisy["bg_tv"] > 0.0
    assert pen_noisy.item() > 0.0


def test_tangent_projection():
    """测试正交切空间投影。"""
    dim = 16
    direction = torch.ones(dim, dtype=torch.float32)
    # 构造法向量与裕度
    normal = torch.zeros(dim, dtype=torch.float32)
    normal[0] = 1.0  # 要求第一维方向保持正向
    slack = 0.5
    projected = tangent_projection(direction, [normal], [slack])
    assert projected.shape == direction.shape
    assert projected[0].item() >= -1e-6


def test_ambilage_single_pass_pipeline_end_to_end(tmp_path: Path):
    """测试 AmbiLAGEPipeline 单通四阶段端到端运行 (轻量 32x32 仿真测试)。"""
    src_p = tmp_path / "source.png"
    tgt_p = tmp_path / "target.png"
    out_dir = tmp_path / "output_run"

    make_test_image(src_p, (32, 32), color=(230, 230, 230))
    make_test_image(tgt_p, (32, 32), color=(40, 100, 200))

    cfg = AmbiLAGEConfig(
        steps=2,smoke_test=True,
        check_every=1,
        log_every=1,
        latent_levels=6,
        background_levels=28,
        source_ssim_floor=0.50,
        source_mae_ceiling=50.0,
    )
    pipeline = AmbiLAGEPipeline(config=cfg, device="cpu")

    adv_path, ready_doc = pipeline.execute(
        source=src_p,
        target=tgt_p,
        output_dir=out_dir,
        source_title="Test Shampoo Clean",
        target_text="Blue Herbal Essence",
        print_regions=[[0.4, 0.4, 0.6, 0.6]],
    )

    # 验证标准交付物齐全
    assert adv_path.is_file()
    assert (out_dir / "adversarial.png").is_file()
    assert (out_dir / "generative_anchor.png").is_file()
    assert (out_dir / "ready.json").is_file()
    assert (out_dir / "metadata.json").is_file()
    assert (out_dir / "latent_edit_mask.png").is_file()

    # 验证 ready.json 规范与四阶段完整记录
    assert ready_doc["status"] == "completed"
    assert ready_doc["method"] == METHOD
    assert ready_doc["definition"] == DEFINITION
    assert ready_doc["pipeline"] == "ambilage_native_4stage_single_pass"
    assert set(ready_doc["stages"].keys()) == {
        "foreground_segmentation",
        "foreground_statistics",
        "color_harmonization",
        "latent_generation_refinement",
    }
    assert ready_doc["pixel_optimization"] is False
    assert ready_doc["auxiliary_max_levels"] == 0

    # 验证印刷文字绝对零改动
    adv_rgb = np.asarray(Image.open(adv_path))
    anchor_rgb = np.asarray(Image.open(out_dir / "generative_anchor.png"))
    roi = print_region_mask((32, 32), [[0.4, 0.4, 0.6, 0.6]])
    diff = np.abs(adv_rgb.astype(np.int16) - anchor_rgb.astype(np.int16))
    assert np.all(diff[roi] == 0), "端到端生成中印刷区域文字发生改动！"


def test_resolve_box_coordinates_accepts_normalized_and_pixel_boxes():
    """验证分割提示框同时兼容归一化 [0, 1] 与原图像素坐标，并拦截越界框。"""
    norm_default, px_default = _resolve_box_coordinates(None, 181, 500)
    assert norm_default == [[0.0, 0.0, 1.0, 1.0]]
    assert px_default == [[0.0, 0.0, 181.0, 500.0]]

    norm_n, px_n = _resolve_box_coordinates([[0.1, 0.2, 0.9, 0.8]], 100, 200)
    assert norm_n == [[0.1, 0.2, 0.9, 0.8]]
    assert px_n == [[10.0, 40.0, 90.0, 160.0]]

    norm_p, px_p = _resolve_box_coordinates([[1, 1, 180, 497]], 181, 500)
    assert px_p == [[1.0, 1.0, 180.0, 497.0]]
    assert np.allclose(norm_p, [[1.0 / 181.0, 1.0 / 500.0, 180.0 / 181.0, 497.0 / 500.0]])

    with pytest.raises(ValueError, match="主体提示框"):
        _resolve_box_coordinates([[10, 10, 200, 497]], 181, 500)
    with pytest.raises(ValueError, match="主体提示框"):
        _resolve_box_coordinates([[100, 10, 50, 400]], 181, 500)


def test_planner_json_parser_and_box_normalization():
    """验证规划 JSON 解析兼容 // 注释与尾逗号，并将像素输出归一化到 [0, 1]。"""
    from src.visual._ambilage.backends import _normalize_planner_boxes, parse_json_response

    raw = '''Here is the plan:
```json
{
  "source_boxes": [[1, 1, 180, 497]], // product box
  "print_regions": [
    [38, 109, 115, 150], // "label // keep"
  ],
  "edit_prompt": "Preserve https://example.com label lighting",
}
```'''
    parsed = parse_json_response(raw)
    assert parsed["edit_prompt"] == "Preserve https://example.com label lighting"
    norm_src = _normalize_planner_boxes(parsed["source_boxes"], 181, 500, allow_empty=False)
    norm_roi = _normalize_planner_boxes(parsed["print_regions"], 181, 500, allow_empty=True)
    assert all(0.0 <= b[0] < b[2] <= 1.0 and 0.0 <= b[1] < b[3] <= 1.0 for b in norm_src + norm_roi)
    assert _normalize_planner_boxes([], 181, 500, allow_empty=True) == []
    assert _normalize_planner_boxes([], 181, 500, allow_empty=False) == [[0.0, 0.0, 1.0, 1.0]]


def test_flux_canvas_and_sdedit_alignment_fallback(monkeypatch):
    """验证窄高比商品图会按上下文原生桶对称留白，并在全噪漂移时触发部分去噪回正。"""
    from types import SimpleNamespace
    from PIL import Image
    from src.visual._ambilage.backends import ProductionStages, _prepare_flux_canvas

    square = np.full((64, 64, 3), 240, dtype=np.uint8)
    sq_canvas, sy0, sx0, sph, spw, skw, skh = _prepare_flux_canvas(square)
    assert (sy0, sx0, sph, spw, skw, skh) == (0, 0, 64, 64, 1024, 1024)
    assert np.array_equal(sq_canvas, square)

    narrow = np.full((500, 181, 3), 255, dtype=np.uint8)
    narrow[50:450, 20:160] = 120
    canvas, y0, x0, pad_h, pad_w, kw, kh = _prepare_flux_canvas(narrow)
    assert (kw, kh) == (672, 1568)
    assert pad_h == 500 and pad_w > 181 and y0 == 0 and x0 > 0
    assert np.array_equal(canvas[y0:y0 + 500, x0:x0 + 181], narrow)
    assert np.all(canvas[:, :x0] == 255)

    monkeypatch.setattr("src.visual._ambilage.backends.local_model", lambda *args: "/fake/editor")

    class _FakeProc:
        def resize(self, img, h, w):
            return img.resize((w, h))

        def preprocess(self, img, h, w):
            arr = np.asarray(img).transpose(2, 0, 1).copy()
            return torch.tensor(arr, dtype=torch.float32)[None] / 127.5 - 1.0

    class _FakeEditor:
        def __init__(self, drift_full: bool):
            self.drift_full = drift_full
            self.image_processor = _FakeProc()
            self.transformer = SimpleNamespace(
                parameters=lambda: iter([torch.zeros(1, dtype=torch.float32)]),
                config=SimpleNamespace(in_channels=64),
            )
            self.calls = []

        def prepare_latents(self, pre, b, c, h, w, dtype, dev, gen, lat):
            base = torch.ones((1, 4, 8), dtype=dtype)
            return torch.zeros_like(base), base, None, None

        def __call__(self, *, image, prompt, num_inference_steps, guidance_scale, height, width,
                     max_area=None, _auto_resize=True, generator=None, sigmas=None, latents=None):
            self.calls.append({"sigmas": sigmas, "height": height, "width": width, "_auto_resize": _auto_resize})
            if latents is None and self.drift_full:
                return SimpleNamespace(images=[Image.new("RGB", (width, height), (30, 30, 30))])
            return SimpleNamespace(images=[image.resize((width, height))])

    st = ProductionStages.__new__(ProductionStages)
    st.config = AmbiLAGEConfig()
    st.device = torch.device("cpu")

    st.editor = _FakeEditor(drift_full=False)
    out_ok, audit_ok = st.generate_anchor(square, "relight", 42)
    assert out_ok.shape == square.shape
    assert audit_ok["alignment_mode"] == "full_kontext"
    assert len(st.editor.calls) == 1

    st.editor = _FakeEditor(drift_full=True)
    out_fb, audit_fb = st.generate_anchor(narrow, "relight", 42)
    assert out_fb.shape == narrow.shape
    assert audit_fb["alignment_mode"] == "flow_matching_sdedit"
    assert audit_fb["sdedit_strength"] == 0.45
    assert len(st.editor.calls) == 2
    assert st.editor.calls[1]["sigmas"] is not None and max(st.editor.calls[1]["sigmas"]) <= 0.45


def test_ambient_coop_field_zero_centered_and_strict_pure_white_zero_shift():
    import torch
    from src.visual._ambilage.ambient import AmbientCoopField
    from src.visual._ambilage.config import AmbiLAGEConfig

    white = torch.ones((1, 3, 32, 32), dtype=torch.float32)
    fg = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    fg[:, :, 8:24, 8:24] = 1.0
    bg = 1.0 - fg
    eps_fg = fg * (6.0 / 255.0)
    eps_bg = bg * (28.0 / 255.0)
    field_white = AmbientCoopField(white, eps_fg, eps_bg, bg, fg, bg, bg, AmbiLAGEConfig(ambient_init_bias=0.30))
    residual_white = field_white.compute_residual(torch.zeros_like(white))
    assert torch.max(torch.abs(residual_white)).item() == 0.0

    gray = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    field_zero_init = AmbientCoopField(gray, eps_fg, eps_bg, bg, fg, bg, bg, AmbiLAGEConfig(ambient_init_bias=0.0))
    residual_zero = field_zero_init.compute_residual(torch.zeros_like(gray))
    assert torch.max(torch.abs(residual_zero)).item() < 1e-7


def test_extract_ink_core_mask_releases_substrate_on_hd_and_falls_back_on_tiny():
    import numpy as np
    from src.visual._ambilage.config import AmbiLAGEConfig
    from src.visual._ambilage.segmentation import extract_ink_core_mask

    cfg = AmbiLAGEConfig()
    img = np.full((160, 160, 3), 245, dtype=np.uint8)
    fg = np.zeros((160, 160), dtype=bool)
    fg[20:140, 20:140] = True
    rect_roi = np.zeros((160, 160), dtype=bool)
    rect_roi[40:120, 40:120] = True
    # High-contrast dark ink stroke inside label ROI.
    img[75:85, 65:95] = 20

    ink_core, substrate = extract_ink_core_mask(img, fg, rect_roi, cfg)
    assert np.all(ink_core[75:85, 65:95])
    assert bool(ink_core[74, 65])
    assert bool(substrate[45, 45])
    assert not bool(ink_core[45, 45])

    tiny_img = np.full((48, 48, 3), 240, dtype=np.uint8)
    tiny_fg = np.ones((48, 48), dtype=bool)
    tiny_roi = np.zeros((48, 48), dtype=bool)
    tiny_roi[10:30, 10:30] = True
    tiny_img[15:20, 15:25] = 20
    tiny_ink, tiny_sub = extract_ink_core_mask(tiny_img, tiny_fg, tiny_roi, cfg)
    assert np.array_equal(tiny_ink, tiny_roi)
    assert int(tiny_sub.sum()) == 0


def test_production_stages_explicit_plan_anchor_and_mathematical_gate(tmp_path):
    import numpy as np
    import torch
    from PIL import Image
    from src.visual._ambilage.backends import ProductionStages
    from src.visual._ambilage.config import AmbiLAGEConfig

    stages = ProductionStages.__new__(ProductionStages)
    stages.config = AmbiLAGEConfig()
    stages.device = torch.device('cpu')
    stages.planner = stages.processor = stages.sam = stages.sam_processor = stages.editor = None

    src_img = np.full((128, 128, 3), 220, dtype=np.uint8)
    src_img[32:96, 32:96] = [180, 175, 170]
    src_path = tmp_path / 'source.png'
    Image.fromarray(src_img).save(src_path)

    plan = stages.plan(src_path, 'shampoo')
    assert plan['edit_prompt']
    assert len(plan['source_boxes']) == 1
    assert plan.get('backend') == 'explicit_public_recipe_v1' or 'planner_model' in plan

    initial = src_img.copy()
    initial[:16, :16] = 205
    anchor, anchor_audit = stages.generate_anchor(initial, plan['edit_prompt'], 42)
    assert np.array_equal(anchor, initial)
    assert anchor_audit['backend'] == 'perceptual_illumination_line_search_v1'

    roi = np.zeros((128, 128), dtype=bool)
    roi[48:80, 48:80] = True
    fg = np.zeros((128, 128), dtype=bool)
    fg[32:96, 32:96] = True

    good = src_img.copy()
    good[:24, :24] = 212
    passed = stages.gate(src_path, good, 'shampoo', roi_mask=roi, fg_mask=fg)
    assert passed['passed'] is True
    assert passed['metrics']['roi_max_abs_diff'] == 0.0

    bad_roi = src_img.copy()
    bad_roi[50:60, 50:60] = 10
    failed_roi = stages.gate(src_path, bad_roi, 'shampoo', roi_mask=roi, fg_mask=fg)
    assert failed_roi['passed'] is False

    bad_heavy = src_img.copy()
    bad_heavy[~roi] = 30
    failed_heavy = stages.gate(src_path, bad_heavy, 'shampoo', roi_mask=roi, fg_mask=fg)
    assert failed_heavy['passed'] is False


def test_ambilage_config_annealing_and_checkpoint_defaults_and_validation():
    """验证 180~250 步自适应退火与断点续跑默认配置及参数校验。"""
    cfg = AmbiLAGEConfig()
    cfg.validate()
    assert cfg.steps == 250
    assert cfg.adaptive_annealing is True
    assert cfg.anneal_min_steps == 180
    assert cfg.checkpoint_every == 5
    assert cfg.resume_from_checkpoint is True

    bad_cases = [
        {"anneal_min_steps": 0},
        {"anneal_min_lr_ratio": 0.0},
        {"anneal_min_lr_ratio": 1.2},
        {"anneal_warmup_steps": -1},
        {"anneal_plateau_factor": 0.05},
        {"anneal_plateau_patience": 0},
        {"anneal_backtrack_factor": 1.1},
        {"anneal_improve_tol": -1e-4},
        {"anneal_early_stop_patience": -1},
        {"checkpoint_every": 0},
    ]
    for kwargs in bad_cases:
        with pytest.raises(ValueError):
            AmbiLAGEConfig(**kwargs).validate()


def test_adaptive_annealing_controller_cooling_and_bounded_early_stop():
    """验证自适应退火在 [180, 250] 区间约束降温与早停行为。"""
    from src.visual._ambilage.latent_refinement import AdaptiveAnnealingController

    cfg = AmbiLAGEConfig(steps=250, anneal_min_steps=180, anneal_early_stop_patience=4, anneal_plateau_patience=2)
    ctrl = AdaptiveAnnealingController(cfg, init_utility=0.50)
    assert ctrl.lr_scale(1) < ctrl.lr_scale(cfg.anneal_warmup_steps)
    peak_lr = ctrl.lr_scale(cfg.anneal_warmup_steps)
    assert peak_lr == pytest.approx(1.0)

    # 回退和拒步应压低自适应温度系数
    ctrl.on_step_end(accepted=True, used_backtracks=1)
    assert ctrl.adaptive_scale == pytest.approx(cfg.anneal_backtrack_factor)
    ctrl.on_step_end(accepted=False, used_backtracks=0)
    assert ctrl.adaptive_scale == pytest.approx(cfg.anneal_backtrack_factor ** 3)

    # 平台期未到达 min_steps (180) 前绝不触发早停
    for step in range(140, 180, 10):
        stop, reason = ctrl.on_check(step, 0.50, True, current_lr_scale=ctrl.lr_scale(step), best_quality_ok=True)
        assert stop is False and reason is None

    # 达到 min_steps (180) 且停滞次数满足耐心值时收敛早停，且状态可序列化恢复
    state = ctrl.state_dict()
    ctrl_restored = AdaptiveAnnealingController(cfg, init_utility=0.50)
    ctrl_restored.load_state_dict(state)
    stop, reason = ctrl_restored.on_check(180, 0.50, True, current_lr_scale=ctrl_restored.lr_scale(180), best_quality_ok=True)
    assert stop is True
    assert reason == "adaptive_annealing_converged"


def test_latent_refinement_checkpoint_resume_equivalence_and_signature_guard(tmp_path: Path):
    """验证 Stage 4 断点续跑数值等价性及不匹配签名的严格拦截。"""
    from src.visual._ambilage.latent_refinement import AmbiLAGEVae, refine_ambient_coop_latent
    from src.visual._ambilage.smoke import TinyAutoencoder

    class _DetObjective:
        def __init__(self, fail_on_step: int | None = None):
            self.fail_on_step = fail_on_step
            self._baseline_values = None

        def evaluate(self, img: torch.Tensor, step: int = 0):
            score = (img.mean() * 0.5).reshape(1, 1)
            if self._baseline_values is None:
                self._baseline_values = torch.tensor([0.25], device=img.device)
            return img.mean(), score

        def image_gradient(self, img: torch.Tensor, step: int):
            if self.fail_on_step is not None and step == self.fail_on_step:
                raise OSError("[Errno 5] Simulated PTY disconnect")
            if self._baseline_values is None:
                self._baseline_values = torch.tensor([0.25], device=img.device)
            leaf = img.detach().clone().requires_grad_(True)
            util = leaf[:, 2].mean() - 0.1 * leaf[:, 0].mean()
            grad, = torch.autograd.grad(util, leaf)
            return util.detach(), util.detach().reshape(1, 1), grad.detach(), {}

    h, w = 32, 32
    fg = np.zeros((h, w), dtype=bool)
    fg[6:26, 6:26] = True
    roi = np.zeros((h, w), dtype=bool)
    roi[12:20, 12:20] = True
    cfg = AmbiLAGEConfig(
        steps=6, check_every=2, log_every=10, checkpoint_every=2,
        anneal_min_steps=6, anneal_early_stop_patience=0,
        source_ssim_floor=0.5, source_mae_ceiling=50.0, smoke_test=True,
    )
    envelopes = compute_spatial_envelopes(fg, roi, cfg)
    anchor = torch.full((1, 3, h, w), 0.75, dtype=torch.float32)
    orig = torch.full((1, 3, h, w), 0.78, dtype=torch.float32)
    vae = AmbiLAGEVae("cpu", model=TinyAutoencoder(), test_fixture=True)

    torch.manual_seed(123)
    full_img, full_meta = refine_ambient_coop_latent(
        anchor, orig, envelopes, _DetObjective(), None, vae, cfg, output_dir=tmp_path / "full",
    )

    # 先跑到第 4 步保存断点并在第 5 步模拟崩溃，再从第 4 步断点续跑到第 6 步
    ckpt_dir = tmp_path / "resumed"
    torch.manual_seed(123)
    with pytest.raises(OSError, match="Simulated PTY disconnect"):
        refine_ambient_coop_latent(
            anchor, orig, envelopes, _DetObjective(fail_on_step=5), None, vae, cfg, output_dir=ckpt_dir,
        )
    assert (ckpt_dir / "checkpoint_latest.pt").is_file()

    resumed_img, resumed_meta = refine_ambient_coop_latent(
        anchor, orig, envelopes, _DetObjective(), None, vae, cfg, output_dir=ckpt_dir,
    )
    assert torch.allclose(full_img, resumed_img, atol=1e-6)
    assert full_meta["best_step"] == resumed_meta["best_step"]
    assert full_meta["history"] == resumed_meta["history"]
    assert resumed_meta["checkpoint_resume"]["resumed"] is True
    assert resumed_meta["checkpoint_resume"]["resumed_from_step"] == 4

    # 更改配置后复用不匹配检查点必须报错拦截
    mismatched_cfg = AmbiLAGEConfig(
        steps=7, check_every=2, log_every=10, checkpoint_every=2,
        anneal_min_steps=7, anneal_early_stop_patience=0,
        source_ssim_floor=0.5, source_mae_ceiling=50.0, smoke_test=True,
    )
    with pytest.raises(ValueError, match="断点检查点与当前配置或输入不匹配"):
        refine_ambient_coop_latent(
            anchor, orig, envelopes, _DetObjective(), None, vae, mismatched_cfg, output_dir=ckpt_dir,
        )


def test_pipeline_stage3_and_stage4_resume_and_early_stop_validation(tmp_path: Path, monkeypatch):
    """验证完整管线从 Stage 3/4 断点续跑跳过前面阶段、自适应退火收敛早停与凭证门禁。"""
    from src.visual._ambilage.evidence import validate_ready
    from src.visual._ambilage.smoke import SmokeStages

    class _CountingSmokeStages(SmokeStages):
        def __init__(self, config, device):
            self.config = config
            self.calls = {"plan": 0, "detect": 0, "segment": 0, "anchor": 0}

        def plan(self, source, category):
            self.calls["plan"] += 1
            return super().plan(source, category)

        def detect_print(self, source):
            self.calls["detect"] += 1
            return super().detect_print(source)

        def segment(self, image, boxes):
            self.calls["segment"] += 1
            return super().segment(image, boxes)

        def generate_anchor(self, initial, prompt, seed):
            self.calls["anchor"] += 1
            return super().generate_anchor(initial, prompt, seed)

    src_p = tmp_path / "source.png"
    tgt_p = tmp_path / "target.png"
    out_dir = tmp_path / "pipe_resume"
    make_test_image(src_p, (32, 32), color=(225, 225, 225))
    make_test_image(tgt_p, (32, 32), color=(40, 100, 200))

    cfg = AmbiLAGEConfig(
        steps=8, anneal_min_steps=6, check_every=1, log_every=1,
        anneal_plateau_patience=1, anneal_early_stop_patience=2,
        anneal_improve_tol=1.0,  # 触发平台期早停以测试在 [6, 8] 区间收敛停止
        checkpoint_every=2, resume_from_checkpoint=True,
        smoke_test=True, source_ssim_floor=0.50, source_mae_ceiling=50.0,
    )

    from src.visual.training_objectives import SequentialEnsembleObjective
    orig_img_grad = SequentialEnsembleObjective.image_gradient
    calls = {"n": 0}

    def _crash_on_step5(self, image, step):
        calls["n"] += 1
        if step == 5:
            raise OSError("[Errno 5] Simulated terminal disconnect")
        return orig_img_grad(self, image, step)

    monkeypatch.setattr(SequentialEnsembleObjective, "image_gradient", _crash_on_step5)
    b1 = _CountingSmokeStages(cfg, "cpu")
    with pytest.raises(OSError, match="Simulated terminal disconnect"):
        AmbiLAGEPipeline(config=cfg, device="cpu", stage_backend=b1).execute(
            source=src_p, target=tgt_p, output_dir=out_dir,
            source_title="Test", target_text="Blue Herbal", print_regions=[[0.4, 0.4, 0.6, 0.6]],
        )
    assert (out_dir / "FAILED.json").is_file()
    assert (out_dir / "stage3_state.json").is_file()
    assert (out_dir / "checkpoint_latest.pt").is_file()
    assert not (out_dir / "ready.json").exists()
    assert b1.calls["plan"] == 1 and b1.calls["anchor"] == 1

    # 移除故障注入后重跑同一输出目录：Stage 1~3 不应重复调用，并能在 [6, 8] 区间收敛完成
    monkeypatch.setattr(SequentialEnsembleObjective, "image_gradient", orig_img_grad)
    b2 = _CountingSmokeStages(cfg, "cpu")
    adv_path, ready_doc = AmbiLAGEPipeline(config=cfg, device="cpu", stage_backend=b2).execute(
        source=src_p, target=tgt_p, output_dir=out_dir,
        source_title="Test", target_text="Blue Herbal", print_regions=[[0.4, 0.4, 0.6, 0.6]],
    )
    assert b2.calls == {"plan": 0, "detect": 0, "segment": 0, "anchor": 0}
    assert adv_path.is_file()
    assert not (out_dir / "FAILED.json").exists()
    assert not (out_dir / "stage3_state.json").exists()
    assert not (out_dir / "checkpoint_latest.pt").exists()
    assert ready_doc["refinement"]["checkpoint_resume"]["resumed"] is True
    assert ready_doc["refinement"]["checkpoint_resume"]["stage3_resumed"] is True
    assert ready_doc["refinement"]["checkpoint_resume"]["resumed_from_step"] == 4
    assert 6 <= ready_doc["steps_completed"] <= 8
    assert ready_doc["stop_reason"] == "adaptive_annealing_converged"
    assert validate_ready(out_dir, require_production=False)["status"] == "completed"

    # 若伪造早停或越过 anneal_min_steps 下限，validate_ready 必须拒绝
    tampered = json.loads((out_dir / "ready.json").read_text(encoding="utf-8"))
    tampered["steps_completed"] = 2
    tampered["stages"]["latent_generation_refinement"]["steps_completed"] = 2
    tampered["refinement"]["steps_completed"] = 2
    tampered["refinement"]["adaptive_annealing"]["steps_executed"] = 2
    (out_dir / "ready.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="潜空间步骤未完整完成"):
        validate_ready(out_dir, require_production=False)


def test_oklab_roundtrip_and_differentiability():
    """验证可微 OKLab 往返精度、边界零值有限梯度及纯色度迁移下的亮度 SSIM/MAE 门禁特性。"""
    import torch
    from src.visual._ambilage.objectives import (
        oklab_to_srgb,
        source_luma_mae_levels,
        source_ssim,
        srgb_to_oklab,
    )

    torch.manual_seed(0)
    rgb = torch.rand((1, 3, 24, 24), dtype=torch.float32)
    rebuilt = oklab_to_srgb(srgb_to_oklab(rgb), clamp=True)
    assert float((rgb - rebuilt).abs().max()) <= 1e-4

    pts = torch.tensor([[[[0.0]], [[1e-6]], [[0.5]], [[1.0]]]], dtype=torch.float32).repeat(1, 3, 1, 1)
    pts.requires_grad_(True)
    loss = oklab_to_srgb(srgb_to_oklab(pts), clamp=False).sum()
    loss.backward()
    assert pts.grad is not None
    assert torch.isfinite(pts.grad).all()

    pink = torch.tensor([0.90, 0.45, 0.58], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, 32, 32).clone()
    pink_lab = srgb_to_oklab(pink)
    blue_lab = torch.cat([pink_lab[:, 0:1], pink_lab[:, 1:3] - 0.16], dim=1)
    recolored_blue = oklab_to_srgb(blue_lab, clamp=True, gamut_preserve_luma=True)
    rgb_mae = float((pink - recolored_blue).abs().mean() * 255.0)
    luma_mae = float(source_luma_mae_levels(recolored_blue, pink))
    luma_ssim = float(source_ssim(recolored_blue, pink, luma_only=True))
    rgb_ssim = float(source_ssim(recolored_blue, pink, luma_only=False))
    assert rgb_mae > 20.0
    assert luma_mae < 5.0
    assert luma_ssim > 0.90
    assert rgb_ssim < luma_ssim


def test_ambient_field_global_background_zero_spatial_variance():
    """验证背景彻底脱离 16 通道 VAE 潜差分，强局部噪声 diff 下左右背景变化完全一致。"""
    import torch
    from src.visual._ambilage.ambient import AmbientCoopField
    from src.visual._ambilage.config import AmbiLAGEConfig
    from src.visual._ambilage.segmentation import compute_spatial_envelopes

    h, w = 48, 48
    anchor = torch.full((1, 3, h, w), 0.85, dtype=torch.float32)
    fg = np.zeros((h, w), dtype=bool)
    fg[16:32, 16:32] = True
    roi = np.zeros((h, w), dtype=bool)
    cfg = AmbiLAGEConfig(bg_radial_vignette_scale=0.05)
    env = compute_spatial_envelopes(fg, roi, cfg, source_rgb=np.full((h, w, 3), 217, dtype=np.uint8))

    field = AmbientCoopField(
        anchor=anchor,
        eps_fg_map=torch.as_tensor(env["fg_level_array"] / 255.0, dtype=torch.float32)[None, None],
        eps_bg_map=torch.as_tensor(env["bg_level_array"] / 255.0, dtype=torch.float32)[None, None],
        w_bg_mod=torch.as_tensor(env["w_bg_mod"], dtype=torch.float32)[None, None],
        fg_mask_t=torch.as_tensor(fg, dtype=torch.float32)[None, None],
        bg_mask_t=torch.as_tensor(~fg, dtype=torch.float32)[None, None],
        support_mask=torch.as_tensor(env["support_mask"], dtype=torch.float32)[None, None],
        config=cfg,
        w_fg_recolor=torch.as_tensor(env["w_fg_recolor"], dtype=torch.float32)[None, None],
    )
    with torch.no_grad():
        field.bg_color_shift.copy_(torch.tensor([[[[0.05]], [[-0.02]], [[-0.06]]]], dtype=torch.float32))

    # 仅在左半背景注入极大空间异质噪声 diff，右半完全为零
    heterogeneous_diff = torch.zeros_like(anchor)
    heterogeneous_diff[:, :, :, : w // 2] = torch.randn((1, 3, h, w // 2)) * 500.0
    res = field.compute_residual(heterogeneous_diff)

    # 远离前景边界的稳定背景区（w_bg_budget == 1.0），左右两侧残差均值与逐通道空间标准差必须完全相等且零斑块
    outer_bg = torch.ones((h, w), dtype=bool)
    outer_bg[8:40, 8:40] = False
    left_outer = res[:, :, outer_bg & (torch.arange(w)[None, :] < w // 2)].detach()
    right_outer = res[:, :, outer_bg & (torch.arange(w)[None, :] >= w // 2)].detach()
    assert float((left_outer.mean(dim=-1) - right_outer.mean(dim=-1)).abs().max()) < 1e-6
    assert float(left_outer.std(dim=-1).max()) < 1e-6
    assert float(right_outer.std(dim=-1).max()) < 1e-6


def test_dual_naturalness_gate_blocks_chroma_blotches_and_protects_highlights(tmp_path):
    """验证高光/深阴影衰减掩膜与 RGB+OKLab 双硬门禁可拦截严重色偏并放行 Choice 1 稳健微光影。"""
    import numpy as np
    from PIL import Image
    import torch
    from src.visual._ambilage.ambient import AmbientCoopField
    from src.visual._ambilage.backends import ProductionStages
    from src.visual._ambilage.config import AmbiLAGEConfig
    from src.visual._ambilage.illumination_harmonization import harmonize_illumination
    from src.visual._ambilage.segmentation import (
        compute_recolor_attenuation_mask,
        compute_spatial_envelopes,
        print_region_mask,
    )

    h, w = 160, 160
    src = np.full((h, w, 3), 255, dtype=np.uint8)
    fg = np.zeros((h, w), dtype=bool)
    fg[20:140, 20:140] = True
    src[fg] = [235, 115, 145]  # 粉橘色商品本体
    src[30:45, 30:45] = [252, 252, 252]  # 高光反光点
    src[115:130, 30:45] = [12, 12, 12]   # 深黑阴影点
    roi_boxes = [[0.40, 0.40, 0.60, 0.60]]
    roi = print_region_mask((h, w), roi_boxes)
    src[roi] = [20, 20, 20]

    cfg = AmbiLAGEConfig()
    assert cfg.fg_chroma_levels == 0
    att = compute_recolor_attenuation_mask(src, fg, roi, cfg)
    assert float(att[32:42, 32:42].mean()) < 0.15
    assert float(att[118:128, 32:42].mean()) < 0.15
    assert float(att[25:35, 100:120].mean()) > 0.70

    # 1. Choice 1 稳健锚点应保主体本色并通过双硬门禁
    anchor_rgb, _ = harmonize_illumination(
        source_image=src,
        target_color_rgb=[40, 130, 225],
        foreground=fg,
        regions=roi_boxes,
        config=cfg,
    )
    assert np.array_equal(anchor_rgb[roi], src[roi])

    src_path = tmp_path / "src.png"
    Image.fromarray(src).save(src_path)
    stages = ProductionStages.__new__(ProductionStages)
    stages.config = cfg
    stages.device = torch.device("cpu")
    stages.planner = stages.processor = stages.sam = stages.sam_processor = stages.editor = None
    pass_res = stages.gate(
        src_path, anchor_rgb, category="toothbrush", roi_mask=roi, fg_mask=fg
    )
    assert pass_res["passed"] is True
    assert pass_res["metrics"]["roi_max_abs_diff"] == 0.0
    assert pass_res["metrics"]["fg_rgb_source_mae_levels"] <= cfg.fg_source_mae_ceiling

    # 2. 注入严重色度偏色/花斑时，即使亮度一致也必须被全彩双硬门禁拦截
    bad_cfg = AmbiLAGEConfig(fg_chroma_levels=64)
    env = compute_spatial_envelopes(fg, roi, bad_cfg, source_rgb=anchor_rgb)
    anchor_t = torch.from_numpy(anchor_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    field = AmbientCoopField(
        anchor=anchor_t,
        eps_fg_map=torch.as_tensor(env["fg_level_array"] / 255.0)[None, None],
        eps_bg_map=torch.as_tensor(env["bg_level_array"] / 255.0)[None, None],
        w_bg_mod=torch.as_tensor(env["w_bg_mod"])[None, None],
        fg_mask_t=torch.as_tensor(fg, dtype=torch.float32)[None, None],
        bg_mask_t=torch.as_tensor(~fg, dtype=torch.float32)[None, None],
        support_mask=torch.as_tensor(env["support_mask"])[None, None],
        config=bad_cfg,
        w_fg_recolor=torch.as_tensor(env["w_fg_recolor"])[None, None],
    )
    with torch.no_grad():
        field.fg_chroma_shift.copy_(torch.tensor([[[[-0.14]], [[-0.12]]]], dtype=torch.float32))
        res = field.compute_residual(torch.zeros_like(anchor_t))
        blotched_rgb = np.clip(np.rint((anchor_t + res)[0].permute(1, 2, 0).numpy() * 255.0), 0, 255).astype(np.uint8)

    fail_res = stages.gate(
        src_path, blotched_rgb, category="toothbrush", roi_mask=roi, fg_mask=fg
    )
    assert fail_res["metrics"]["fg_rgb_source_mae_levels"] > cfg.fg_source_mae_ceiling
    assert fail_res["passed"] is False
    assert fail_res["natural"] is False


def test_extract_ink_core_mask_releases_colored_bottle_substrate():
    """验证局部中值对比提取仅锁死彩色瓶身中的文字笔画而释放彩色底色基底。"""
    import numpy as np
    from src.visual._ambilage.config import AmbiLAGEConfig
    from src.visual._ambilage.segmentation import extract_ink_core_mask

    cfg = AmbiLAGEConfig()
    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    fg = np.zeros((160, 160), dtype=bool)
    fg[16:144, 24:136] = True
    # 彩色渐变/中等饱和度蓝色瓶身 (RGB chroma = 120 >= 12)
    img[fg] = [75, 125, 195]
    rect_roi = np.zeros((160, 160), dtype=bool)
    rect_roi[50:100, 40:120] = True
    # 在印刷框中央添加白字细笔画 (与蓝色底色对比 >= 30)
    img[72:76, 55:105] = [245, 245, 245]

    ink_core, substrate = extract_ink_core_mask(img, fg, rect_roi, cfg)
    assert np.all(ink_core[72:76, 55:105])
    assert int(substrate.sum()) > int(rect_roi.sum() * 0.5)
    assert bool(substrate[54, 44])
    assert not bool(ink_core[54, 44])


def test_stage3_and_stage4_enforce_fg_chroma_tv_against_source(tmp_path: Path):
    """验证 Stage 3 与 Stage 4 均以原图 source 为基准约束前景 OKLab 色度 TV，防止自然锚点越界后在终检门禁失败。"""
    from PIL import Image
    from src.visual._ambilage.backends import ProductionStages
    from src.visual._ambilage.config import AmbiLAGEConfig
    from src.visual._ambilage.illumination_harmonization import harmonize_illumination
    from src.visual._ambilage.latent_refinement import AmbiLAGEVae, refine_ambient_coop_latent
    from src.visual._ambilage.segmentation import compute_spatial_envelopes
    from src.visual._ambilage.smoke import TinyAutoencoder

    # 1. 高对比度冷暖交替前景条纹：若不回溯将突破 fg_chroma_tv 上限，Stage 3 应回溯至 <= 0.60 * ceiling 并通过终检门禁
    h, w = 96, 96
    src = np.full((h, w, 3), 255, dtype=np.uint8)
    fg = np.zeros((h, w), dtype=bool)
    fg[12:84, 12:84] = True
    yy, xx = np.indices((h, w))
    stripe = ((yy + xx) % 2) == 0
    src[fg & stripe] = [245, 195, 70]
    src[fg & (~stripe)] = [18, 18, 22]

    cfg = AmbiLAGEConfig(harmonization_selected_color_mix=0.30)
    anchor_rgb, rec = harmonize_illumination(
        source_image=src,
        target_color_rgb=[35, 95, 230],
        foreground=fg,
        regions=[],
        config=cfg,
    )
    assert rec["fg_chroma_tv_levels"] <= 0.60 * cfg.fg_chroma_tv_ceiling + 1e-6
    assert rec["chosen_fraction"] < 1.0

    src_path = tmp_path / "striped_src.png"
    Image.fromarray(src).save(src_path)
    stages = ProductionStages.__new__(ProductionStages)
    stages.config = cfg
    stages.device = torch.device("cpu")
    stages.planner = stages.processor = stages.sam = stages.sam_processor = stages.editor = None
    gate_res = stages.gate(src_path, anchor_rgb, category="hair extensions", roi_mask=None, fg_mask=fg)
    assert gate_res["passed"] is True
    assert gate_res["metrics"]["fg_chroma_tv_levels"] <= cfg.fg_chroma_tv_ceiling

    # 2. 若外部传入与原图前景 OKLab 色度 TV 超限的坏锚点，Stage 4 evaluate_quality 必须相对 original_tensor 立即在 step 0 拦截
    h4, w4 = 32, 32
    fg4 = np.zeros((h4, w4), dtype=bool)
    fg4[4:28, 4:28] = True
    roi4 = np.zeros((h4, w4), dtype=bool)
    cfg4 = AmbiLAGEConfig(
        steps=2, check_every=1, log_every=10, anneal_min_steps=2, anneal_early_stop_patience=0,
        source_ssim_floor=0.2, source_mae_ceiling=50.0, fg_chroma_tv_ceiling=2.5, smoke_test=True,
    )
    env4 = compute_spatial_envelopes(fg4, roi4, cfg4)
    orig4 = torch.full((1, 3, h4, w4), 0.5, dtype=torch.float32)
    anchor4 = orig4.clone()
    y4, x4 = torch.meshgrid(torch.arange(h4), torch.arange(w4), indexing="ij")
    checker4 = ((y4 + x4) % 2 == 0) & torch.from_numpy(fg4)
    anti4 = ((y4 + x4) % 2 == 1) & torch.from_numpy(fg4)
    anchor4[0, 0, checker4] += 0.04
    anchor4[0, 1, checker4] -= 0.04
    anchor4[0, 0, anti4] -= 0.04
    anchor4[0, 1, anti4] += 0.04

    class _DummyObjective:
        def evaluate(self, img: torch.Tensor, step: int = 0):
            return img.mean(), img.mean().reshape(1, 1)

    vae = AmbiLAGEVae("cpu", model=TinyAutoencoder(), test_fixture=True)
    with pytest.raises(RuntimeError, match="自然锚点未通过结构/误差门禁"):
        refine_ambient_coop_latent(anchor4, orig4, env4, _DummyObjective(), None, vae, cfg4)
