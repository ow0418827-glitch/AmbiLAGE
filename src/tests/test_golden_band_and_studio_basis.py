"""黄金先验区间选图、前景裁剪外形特征、Stage 4 光场空间基与文本瓶颈感知加权单元测试（纯 CPU 微型夹具）。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from src.paths import PROJECT
from src.experiments.public_pair_selection import (
    compute_public_prior_ranks,
    select_groups,
    validate_config,
)
from src.visual._ambilage.ambient import AmbientCoopField
from src.visual._ambilage.config import AmbiLAGEConfig
from src.visual._ambilage.latent_refinement import AmbiLAGEVae, refine_ambient_coop_latent
from src.visual._ambilage.segmentation import compute_spatial_envelopes
from src.visual._ambilage.smoke import TinyAutoencoder
from src.visual.audit import digest, freeze_json
from src.visual.foreground_shape_features import (
    PREPROCESSING_FG_CROP,
    foreground_crop_shape_input,
)
from src.visual.image_utils import sha256_file
from src.visual.pair_selection import (
    GOLDEN_BAND_MEDIAN_POLICY,
    SelectionConfig,
    choose_pair,
    matches_subtype_filter,
    score_pair,
    validate_public_rank_band,
    verify_selection,
)
from src.visual.shape_calibration import calibrate
from src.visual.shape_features import shape_input
from src.visual.selection.webshop_public import extract_public_rank_peers
from src.visual.training_objectives import PublicFullKeyObjective, SequentialEnsembleObjective
from src.visual.training_surrogates import HELD_OUT_MODELS, select_training_surrogates


EXPECTED_SHAPE_FEATURES_SHA256 = "c6c3a6a1a2e38f4c40b9bec57e2c9486f398cfe6f7220e458a374cf02643d9b0"


def _write_png(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path, format="PNG")
    return sha256_file(path)


def _norm_vec(arr: np.ndarray) -> np.ndarray:
    v = arr.astype(np.float64).ravel()
    return v / max(float(np.linalg.norm(v)), 1e-12)


def test_shape_features_untouched_and_foreground_crop_strips_clutter(tmp_path: Path):
    """验证 shape_features.py 哈希绝对未变，且前景裁剪+背景抑制有效消除杂乱背景干扰。"""
    actual_sha = hashlib.sha256((PROJECT / "src/visual/shape_features.py").read_bytes()).hexdigest()
    assert actual_sha == EXPECTED_SHAPE_FEATURES_SHA256

    h, w = 64, 64
    clean = np.full((h, w, 3), 248, dtype=np.uint8)
    clutter = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if (x // 6 + y // 6) % 2 == 0:
                clutter[y, x] = (20, 185, 45)
            else:
                clutter[y, x] = (175, 25, 165)
            if y < 12 or x > 52:
                clutter[y, x] = (25, 25, 25)

    # 居中相同商品轮廓（瓶身 + 瓶盖）
    for canvas in (clean, clutter):
        canvas[22:48, 22:42] = (35, 95, 195)
        canvas[16:22, 27:37] = (210, 55, 50)

    clean_path = tmp_path / "clean.png"
    clutter_path = tmp_path / "clutter.png"
    _write_png(clean_path, clean)
    _write_png(clutter_path, clutter)

    whole_clean = np.asarray(shape_input(clean_path), dtype=np.float64)
    whole_clutter = np.asarray(shape_input(clutter_path), dtype=np.float64)
    fg_clean_img = foreground_crop_shape_input(clean_path)
    fg_clutter_img = foreground_crop_shape_input(clutter_path)

    assert fg_clean_img.size == (224, 224) and fg_clean_img.mode == "RGB"
    assert fg_clutter_img.size == (224, 224) and fg_clutter_img.mode == "RGB"

    fg_clean = np.asarray(fg_clean_img, dtype=np.float64)
    fg_clutter = np.asarray(fg_clutter_img, dtype=np.float64)

    # 四角外缘背景被抑制为中性灰 127
    assert int(fg_clutter[4, 4, 0]) == 127
    assert int(fg_clutter[219, 219, 0]) == 127

    whole_l1 = float(np.mean(np.abs(whole_clean - whole_clutter)))
    fg_l1 = float(np.mean(np.abs(fg_clean - fg_clutter)))
    whole_cos = float(_norm_vec(whole_clean) @ _norm_vec(whole_clutter))
    fg_cos = float(_norm_vec(fg_clean) @ _norm_vec(fg_clutter))

    assert fg_l1 < whole_l1 * 0.65
    assert fg_cos > whole_cos + 0.08


class _FakeFgBackend:
    """合成外形后端，仅用于验证黄金先验区间与校准绑定。"""

    def __init__(self, features_by_id: dict[str, np.ndarray], *, preprocess: dict):
        self._features_by_id = features_by_id
        self.snapshot = {
            "dino": {"path": "fake_offline_fixture", "files": {}},
            "preprocessing": preprocess,
            "precision": "float32",
        }

    def features_many(self, products):
        return None

    def features(self, product):
        return {"shape": self._features_by_id[product["id"]].copy()}

    def close(self):
        return None


def test_source_exclude_limit_1024_and_public_rank_band_gate(tmp_path: Path):
    """验证 source_exclude 仅过滤原商品、1024 预算、公开对照池黄金名次带及留出模型拦截。"""
    # 1) 验证配置文件合法性与 1024 上限
    cfg_path = PROJECT / "src/experiments/config/selection_groups9_golden_band.example.json"
    cfg_doc = validate_config(json.loads(cfg_path.read_text(encoding="utf-8")))
    assert cfg_doc["expand_candidate_limits"][-1] == 1024
    assert validate_config(dict(cfg_doc, candidate_limit=1024, expand_candidate_limits=[1024]))["candidate_limit"] == 1024
    with pytest.raises(ValueError):
        validate_config(dict(cfg_doc, candidate_limit=1025))

    # 2) 验证严禁在公开先验打分器中传入严格留出评估模型
    for held_out in ("eval_g14", "eval_gme", "ViT-g-14", "gme-Qwen2-VL-2B-Instruct"):
        with pytest.raises(ValueError, match="严禁使用严格留出评估模型"):
            validate_public_rank_band({"scorer": held_out})

    # 3) 验证 source_exclude 仅排除原商品，保留目标自身
    cookie_filter = cfg_doc["subtype_filters"]["03_cookies"]
    assert matches_subtype_filter("Edible Raw Cookie Dough 2 Pints", cookie_filter, role="target") is True
    assert matches_subtype_filter("Edible Raw Cookie Dough 2 Pints", cookie_filter, role="source") is False
    assert matches_subtype_filter("Classic Chocolate Chip Cookies Pack", cookie_filter, role="source") is True

    # 4) 构造微型开发集与黄金名次带选图测试目录（公开根目录下）
    pub_root = tmp_path / "public_data"
    img_dir = pub_root / "images"
    dev_dir = pub_root / "dev"
    img_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)

    feat_map: dict[str, np.ndarray] = {}

    def make_prod(pid: str, title: str, color: tuple[int, int, int], shape_vec: list[float], *, owned=True) -> dict:
        img_p = img_dir / f"{pid}.png"
        arr = np.full((64, 64, 3), 245, dtype=np.uint8)
        arr[16:48, 16:48] = color
        sha = _write_png(img_p, arr)
        feat_map[pid] = np.asarray(shape_vec, dtype=np.float64)
        return {
            "id": pid,
            "title": title,
            "image": str(img_p),
            "image_sha256": sha,
            "source_url": f"https://example.invalid/{pid}",
            "owned": owned,
            "source_note": "合成测试夹具",
        }

    # 开发校准集：1 正例（高外形分）、1 负例（低外形分）
    dev_l1 = make_prod("D000000001", "Chair Dev A", (60, 60, 180), [1.0, 0.0, 0.0], owned=False)
    dev_r1 = make_prod("D000000002", "Chair Dev B", (65, 65, 185), [0.99, 0.14, 0.0], owned=False)
    dev_l0 = make_prod("D000000003", "Chair Dev C", (180, 60, 60), [0.0, 1.0, 0.0], owned=False)
    dev_r0 = make_prod("D000000004", "Chair Dev D", (60, 180, 60), [0.0, 0.0, 1.0], owned=False)
    dev_manifest = {
        "schema_version": 1,
        "reviewer": "offline_unit_fixture",
        "pairs": [
            {"left": dev_l1, "right": dev_r1, "label": 1, "review_note": "同类相似外形"},
            {"left": dev_l0, "right": dev_r0, "label": 0, "review_note": "不同外形负例"},
        ],
    }
    dev_path = dev_dir / "dev_labels.json"
    freeze_json(dev_path, dev_manifest)
    backend = _FakeFgBackend(feat_map, preprocess=PREPROCESSING_FG_CROP)
    cal_path = pub_root / "threshold.json"
    calibrate(dev_path, cal_path, backend, public_root=pub_root)

    # 目标商品：Barber Swivel Chair
    target = make_prod("T000000001", "Swivel Hydraulic Barber Chair Black", (40, 40, 200), [1.0, 0.0, 0.0], owned=False)

    # 8 个不同视觉分数的公开对照池（用于划分高分前排 vs 黄金中游 vs 死区）
    peers = [
        make_prod(f"P{i:09d}", f"Salon Barber Chair Peer {i}" if i <= 4 else f"Dining Chair Peer {i}", (30 + 15 * i, 40, 190 - 10 * i),
                  [max(0.2, 0.98 - 0.08 * i), 0.2 + 0.05 * i, 0.0], owned=False)
        for i in range(1, 9)
    ]

    # 候选商品：
    # - C_SOURCE_ATTR: 包含 source_exclude ("Barber") -> source_has_target_differentiating_attribute
    # - C_CEILING: 高视觉相似 + 标题几乎重合 -> rank=1 < 4 -> public_prior_rank_too_high_ceiling
    # - C_BAND_LOW / C_BAND_HIGH: 落在 [4, 7] 内 -> 合格
    # - C_DEADZONE: 低外观+低标题重叠 -> rank=9 > 7 -> public_prior_rank_too_low_deadzone
    c_attr = make_prod("C000000001", "Hydraulic Barber Chair Source A", (55, 55, 190), [0.95, 0.2, 0.0])
    c_ceil = make_prod("C000000002", "Swivel Hydraulic Chair Black Style", (42, 42, 202), [0.99, 0.05, 0.0])
    c_band1 = make_prod("C000000003", "Modern Office Desk Chair", (90, 90, 150), [0.76, 0.55, 0.0])
    c_band2 = make_prod("C000000004", "Classic Wood Accent Chair", (110, 95, 140), [0.71, 0.62, 0.0])
    c_dead = make_prod("C000000005", "Foldable Patio Stool Green", (190, 190, 50), [0.62, 0.78, 0.0])

    candidates = [c_attr, c_ceil, c_band1, c_band2, c_dead]
    ranks = compute_public_prior_ranks(target, candidates, peers, backend, scorer="dino_lexical_hybrid")
    assert set(ranks) == {c["id"] for c in candidates}
    assert ranks[c_ceil["id"]] <= 2
    assert ranks[c_dead["id"]] > 7

    chair_filter = {
        "policy": "chair_test",
        "include": r"\b(?:chairs?|stools?)\b",
        "exclude": r"\bcushion\b",
        "source_exclude": r"\bbarber\b",
    }
    rank_band = {"enabled": True, "peer_count": len(peers), "rank_min": 4, "rank_max": 7, "scorer": "dino_lexical_hybrid"}
    sel_cfg = SelectionConfig(seed=42, calibration_path=str(cal_path), strategy="median")

    result = choose_pair(
        target,
        candidates,
        backend,
        sel_cfg,
        public_root=pub_root,
        subtype_filter=chair_filter,
        public_rank_band=rank_band,
        public_prior_ranks=ranks,
    )
    assert result["protocol"] == GOLDEN_BAND_MEDIAN_POLICY
    assert result["status"] == "selected"

    by_id = {r["source_id"]: r for r in result["candidates"]}
    assert "source_has_target_differentiating_attribute" in by_id[c_attr["id"]]["rejected_reasons"]
    assert "public_prior_rank_too_high_ceiling" in by_id[c_ceil["id"]]["rejected_reasons"]
    assert "public_prior_rank_too_low_deadzone" in by_id[c_dead["id"]]["rejected_reasons"]
    assert result["selected_source_id"] in {c_band1["id"], c_band2["id"]}

    # 补充 source/target 字段后 verify_selection 必须验证通过
    result["source"] = next(c for c in candidates if c["id"] == result["selected_source_id"])
    result["target"] = target
    verify_selection(result)

    # 5) 验证 extract_public_rank_peers 在微型公共 SQLite 索引上严格隔离候选/目标/开发集
    catalog_db = tmp_path / "tiny_public_catalog.sqlite"
    with sqlite3.connect(catalog_db) as conn:
        conn.execute("CREATE TABLE products(ordinal INTEGER PRIMARY KEY, asin TEXT, data TEXT)")
        all_rows = [dev_l1, dev_r1, dev_l0, dev_r0, target] + candidates + peers
        for idx, item in enumerate(all_rows):
            conn.execute(
                "INSERT INTO products VALUES (?, ?, ?)",
                (
                    idx,
                    item["id"],
                    json.dumps({"name": item["title"], "images": [f"https://m.media-amazon.com/images/I/{item['id']}.jpg"]}),
                ),
            )
        conn.commit()
    peer_out = tmp_path / "peer_extract"
    peer_doc = extract_public_rank_peers(
        catalog_db,
        [{"key": "02_chair", "target_id": target["id"]}],
        peer_out,
        seed=42,
        peer_count=6,
        index_path=tmp_path / "pub_idx.sqlite",
        excluded_ids=[dev_l1["id"], dev_r1["id"], dev_l0["id"], dev_r0["id"]],
        candidate_ids_by_group={"02_chair": [c["id"] for c in candidates]},
        subtype_filters={"02_chair": chair_filter},
    )
    extracted_peer_ids = {p["id"] for p in peer_doc["groups"][0]["peers"]}
    assert len(extracted_peer_ids) == 6
    assert extracted_peer_ids.isdisjoint({target["id"], dev_l1["id"], dev_r1["id"], dev_l0["id"], dev_r0["id"]})
    assert extracted_peer_ids.isdisjoint({c["id"] for c in candidates})


def test_studio_lighting_basis_zero_init_smoothness_and_checkpoint_resume(tmp_path: Path):
    """验证 7 模态光滑影棚光场在零初始化时严格退化为基线，非零时平滑且断点续跑数值等价。"""
    h, w = 48, 48
    fg = np.zeros((h, w), dtype=bool)
    fg[12:36, 12:36] = True
    roi = np.zeros((h, w), dtype=bool)
    roi[18:26, 18:30] = True
    anchor = torch.full((1, 3, h, w), 0.72, dtype=torch.float32)
    orig = torch.full((1, 3, h, w), 0.74, dtype=torch.float32)
    diff = torch.zeros((1, 3, h, w), dtype=torch.float32)

    cfg_off = AmbiLAGEConfig(bg_studio_basis_enabled=False, smoke_test=True)
    cfg_on = AmbiLAGEConfig(bg_studio_basis_enabled=True, bg_studio_basis_scale=0.08, smoke_test=True)
    env = compute_spatial_envelopes(fg, roi, cfg_on)
    eps_fg_map = torch.from_numpy((env["fg_level_array"] / 255.0).astype(np.float32))[None, None]
    eps_bg_map = torch.from_numpy((env["bg_level_array"] / 255.0).astype(np.float32))[None, None]
    w_bg_mod = torch.from_numpy(env["w_bg_mod"].astype(np.float32))[None, None]
    fg_mask_t = torch.from_numpy(env["foreground"].astype(np.float32))[None, None]
    bg_mask_t = torch.from_numpy(env["background"].astype(np.float32))[None, None]
    support_mask = torch.from_numpy(env["support_mask"].astype(np.float32))[None, None]

    field_off = AmbientCoopField(
        anchor, eps_fg_map, eps_bg_map, w_bg_mod, fg_mask_t, bg_mask_t, support_mask, cfg_off
    )
    field_on = AmbientCoopField(
        anchor, eps_fg_map, eps_bg_map, w_bg_mod, fg_mask_t, bg_mask_t, support_mask, cfg_on
    )

    assert field_on.studio_basis_maps.shape == (1, 7, h, w)
    assert field_on.ambient_basis.shape == (1, 3, 7)
    assert torch.allclose(field_off.ambient_field(), field_on.ambient_field(), atol=1e-7)
    assert torch.allclose(field_off.compute_residual(diff), field_on.compute_residual(diff), atol=1e-7)

    with torch.no_grad():
        field_on.ambient_basis[0, 0, 0] = 0.8
        field_on.ambient_basis[0, 1, 5] = -0.6
    bg_field = field_on.ambient_field().detach()
    assert float(bg_field[:, :, ~torch.from_numpy(fg)].std()) > 1e-3
    res = field_on.compute_residual(diff).detach()
    roi_t = torch.from_numpy(roi)
    assert float(res[:, :, roi_t].abs().max()) == 0.0
    # 背景偏移受 eps_bg_map 与像素 [0, 1] 边界双重约束
    assert torch.all(res.abs() <= eps_bg_map + 1e-6)
    assert float((anchor + res).min()) >= 0.0 and float((anchor + res).max()) <= 1.0

    # 验证开启 bg_studio_basis_enabled 时的断点续跑完全等价性（含 ambient_basis 与双基线状态）
    class _BasisObjective:
        def __init__(self, fail_step: int | None = None):
            self.fail_step = fail_step
            self._baseline_values = None
            self._baseline_joint_values = None
            self._baseline_text_values = None

        def evaluate(self, img: torch.Tensor, step: int = 0):
            if self._baseline_values is None:
                self._baseline_values = torch.tensor([0.20], device=img.device)
                self._baseline_joint_values = torch.tensor([0.15], device=img.device)
                self._baseline_text_values = torch.tensor([0.05], device=img.device)
            xx = torch.linspace(-1.0, 1.0, img.shape[-1], device=img.device).view(1, 1, 1, -1)
            util = (img[:, 0:1] * xx).mean()
            return util, util.reshape(1, 1)

        def image_gradient(self, img: torch.Tensor, step: int):
            if self.fail_step is not None and step == self.fail_step:
                raise OSError("[Errno 5] Simulated interrupt")
            leaf = img.detach().clone().requires_grad_(True)
            xx = torch.linspace(-1.0, 1.0, leaf.shape[-1], device=leaf.device).view(1, 1, 1, -1)
            util = (leaf[:, 0:1] * xx).mean() + 0.1 * leaf[:, 2].mean()
            grad, = torch.autograd.grad(util, leaf)
            return util.detach(), util.detach().reshape(1, 1), grad.detach(), {}

    cfg_run = AmbiLAGEConfig(
        steps=6,
        check_every=2,
        log_every=10,
        checkpoint_every=2,
        anneal_min_steps=6,
        anneal_early_stop_patience=0,
        bg_studio_basis_enabled=True,
        bg_studio_basis_scale=0.08,
        source_ssim_floor=0.5,
        source_mae_ceiling=50.0,
        smoke_test=True,
    )
    vae = AmbiLAGEVae("cpu", model=TinyAutoencoder(), test_fixture=True)

    torch.manual_seed(77)
    full_img, full_meta = refine_ambient_coop_latent(
        anchor, orig, env, _BasisObjective(), None, vae, cfg_run, output_dir=tmp_path / "full"
    )
    ckpt_dir = tmp_path / "resumed"
    torch.manual_seed(77)
    with pytest.raises(OSError, match="Simulated interrupt"):
        refine_ambient_coop_latent(
            anchor, orig, env, _BasisObjective(fail_step=5), None, vae, cfg_run, output_dir=ckpt_dir
        )
    resumed_img, resumed_meta = refine_ambient_coop_latent(
        anchor, orig, env, _BasisObjective(), None, vae, cfg_run, output_dir=ckpt_dir
    )
    assert torch.allclose(full_img, resumed_img, atol=1e-6)
    assert full_meta["best_step"] == resumed_meta["best_step"]
    assert full_meta["history"] == resumed_meta["history"]
    assert resumed_meta["checkpoint_resume"]["resumed"] is True


def test_text_aware_proxy_weighting_penalizes_joint_only_gain_and_boosts_lagging_text():
    """验证文本瓶颈感知加权能识别只涨 joint 不涨 text 的虚高代理，上调其权重并加强文本梯度。"""

    class _4DTok:
        context_length = 8
        sot_token_id = 1
        eot_token_id = 2
        pad_token_id = 0

        def encode(self, text: str) -> list[int]:
            low = text.lower()
            if "target" in low:
                return [10]
            return [20]

    class _4DTextModel:
        def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
            out = torch.zeros((tokens.shape[0], 4), dtype=torch.float32, device=tokens.device)
            is_target = (tokens[:, 1] == 10)
            out[is_target, 1] = 1.0  # 目标文本向量 = [0, 1, 0, 0]
            out[~is_target, 2] = 1.0  # 源商品文本向量 = [0, 0, 1, 0]
            return out

    class _DualBehaviorSurrogate:
        device = torch.device("cpu")
        tokenizer = _4DTok()
        model = _4DTextModel()

        def __init__(self, mode: str):
            self.mode = mode

        def snapshot(self):
            return {"name": f"offline_dual_{self.mode}", "test_fixture": True}

        def encode_image(self, image: torch.Tensor):
            # 通道均值映射到 4D 单位向量：
            # - mode='joint_only': 仅推高图像第一维（贴近 target image [1,0,0,0]），text 维（第二维）零增益甚至略降
            # - mode='balanced': 同时响应第二维（贴近 target text [0,1,0,0]），joint 与 text 同步均衡小幅上涨
            c0 = image[:, 0].mean(dim=(-2, -1))
            c1 = image[:, 1].mean(dim=(-2, -1))
            if self.mode == "joint_only":
                raw = torch.stack([0.15 + 3.0 * c0, 0.35 - 0.15 * c0, 0.25 * torch.ones_like(c0), 1.0 - c0], dim=-1)
            else:
                raw = torch.stack([0.20 + 0.50 * c0, 0.10 + 1.80 * c1, 0.25 * torch.ones_like(c0), 1.0 - 0.3 * c1], dim=-1)
            emb = F.normalize(raw, dim=-1)
            return SimpleNamespace(global_embedding=emb, patch_tokens=None)

    class _LinearPurifier:
        config = SimpleNamespace(reverse_steps=10)

        def snapshot(self):
            return {"kind": "linear_identity_fixture"}

        def __call__(self, image: torch.Tensor, *, seed, gradient):
            return image

    target_img = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    target_img[:, 0] = 0.85
    target_img[:, 1] = 0.15
    base_img = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    base_img[:, 0] = 0.10
    base_img[:, 1] = 0.10

    # 对 test_img：c0 大幅上升（使 proxy0 joint 大涨、text 滞后），c1 适度上升（使 proxy1 balanced 均衡小涨）
    test_img = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    test_img[:, 0] = 0.80
    test_img[:, 1] = 0.35

    common = dict(
        steps=100,
        joint_weight=1.0,
        text_weight=0.35,
        local_ot_weight=0.0,
        temperature=0.07,
        eot_enabled=False,
        diffusion_weight=0.25,
        ensemble_temperature=0.05,
        ensemble_uniform_mix=0.20,
        psi_progressive_schedule=True,
        seed=42,
        smoke_test=True,
        surrogate_set="lucid",
        surrogate_config=None,
    )
    cfg_plain = AmbiLAGEConfig(**common, text_aware_proxy_weighting=False, selection_text_parity_mix=0.0)
    cfg_aware = AmbiLAGEConfig(
        **common,
        text_aware_proxy_weighting=True,
        selection_text_parity_mix=0.75,
        proxy_text_bottleneck_mix=0.80,
    )

    surrogates = [
        _DualBehaviorSurrogate("joint_only"),
        _DualBehaviorSurrogate("balanced"),
        _DualBehaviorSurrogate("balanced"),
    ]
    ens_plain = SequentialEnsembleObjective(
        surrogates, target_img, "source catalog text", "target feature text", cfg_plain, purifier=_LinearPurifier()
    )
    ens_aware = SequentialEnsembleObjective(
        surrogates, target_img, "source catalog text", "target feature text", cfg_aware, purifier=_LinearPurifier()
    )

    # 先以 base_img 初始化基线
    ens_plain.evaluate(base_img, step=0)
    ens_aware.evaluate(base_img, step=0)

    # 验证稳态选优 selection_text_parity_mix 提高对文本维度的约束贡献
    feat_probe = F.normalize(torch.tensor([[0.90, 0.05, 0.10, 0.10]], dtype=torch.float32), dim=-1)
    stat_plain, _, u_text_plain = ens_plain.objectives[0].component_values(feat_probe, progressive=False)
    stat_aware, _, u_text_aware = ens_aware.objectives[0].component_values(feat_probe, progressive=False)
    assert float(u_text_plain) == pytest.approx(float(u_text_aware), abs=1e-6)
    assert float(stat_aware) != pytest.approx(float(stat_plain), abs=1e-4)

    # 在 test_img 上执行梯度评估：
    # plain 模式被 proxy0 虚高 joint 欺骗，认为 proxy0 无需加大权重；
    # text_aware 模式看到 proxy0 g_text 严重滞后于 g_joint，显著上调 proxy0 权重与文本梯度倍增系数
    _, _, _, info_plain = ens_plain.image_gradient(test_img, step=50)
    _, _, _, info_aware = ens_aware.image_gradient(test_img, step=50)

    assert "model_joint_gains" in info_aware and "model_text_gains" in info_aware
    assert info_aware["model_joint_gains"][0] > info_aware["model_text_gains"][0] + 0.005
    assert info_aware["paths"][0]["text_multiplier"] > 1.0
    assert info_aware["paths"][1]["text_multiplier"] == pytest.approx(1.0, abs=1e-6)
    assert info_aware["model_weights"][0] > info_plain["model_weights"][0] + 0.05


def test_paper_surrogates_example_config_validates_and_selects_14x14_triad():
    """验证 14x14 论文代理示例配置能通过 AmbiLAGEConfig 校验，且完全隔离严控留出模型。"""
    cfg_path = PROJECT / "src/visual/config/ambilage.paper_surrogates.example.json"
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = AmbiLAGEConfig.from_dict(raw)

    assert cfg.surrogate_set == "visual_inception_paper"
    assert cfg.bg_studio_basis_enabled is True
    assert cfg.text_aware_proxy_weighting is True
    assert cfg.selection_text_parity_mix == pytest.approx(0.75)
    assert cfg.proxy_text_bottleneck_mix == pytest.approx(0.55)

    selection = select_training_surrogates(cfg.surrogate_set, cfg.surrogate_config)
    names = tuple(spec.name for spec in selection.specs)
    assert names == ("ViT-L-14", "ViT-SO400M-14-SigLIP-384", "ViT-bigG-14")
    assert set(names).isdisjoint(HELD_OUT_MODELS)
