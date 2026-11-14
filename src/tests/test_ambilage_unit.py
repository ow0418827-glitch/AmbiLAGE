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
    assert 'text_weight' not in cfg.to_dict()

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


@pytest.mark.parametrize('weight', [0., .5, 1.])
def test_removed_text_training_weight_cannot_be_reenabled(weight):
    with pytest.raises(ValueError, match='未知方法配置字段: text_weight'):
        AmbiLAGEConfig.from_dict({'text_weight': weight})


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

    # 2. 验证背景区保持源图不变
    assert np.array_equal(anchor_rgb[~fg], source[~fg])

    # 3. 验证产物文件已落盘
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
