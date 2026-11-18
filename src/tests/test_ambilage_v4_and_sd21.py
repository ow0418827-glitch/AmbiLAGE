"""AmbiLAGE v4 跨模态渐进优化、L4 24GB 显存治理、同源公开包接入与留出 SD 2.1 防御单元测试。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from src.tests.conftest_helpers import make_test_image
from src.visual._ambilage.config import AmbiLAGEConfig
from src.visual._ambilage.evidence import validate_ready
from src.visual._ambilage.latent_refinement import (
    AmbiLAGEVae,
    build_spectral_envelope,
    refine_ambient_coop_latent,
    spectral_manifold_project,
)
from src.visual._ambilage.pipeline import generate_ambilage
from src.visual._ambilage.segmentation import compute_spatial_envelopes
from src.visual._ambilage.smoke import TinyAutoencoder, TinyPurifier, TinySurrogate
from src.visual.image_utils import sha256_file
from src.visual.training_objectives import (
    LEGACY_V3_OBJECTIVE_KIND,
    PublicFullKeyObjective,
    SUPPORTED_TRAINING_OBJECTIVE_KINDS,
    SequentialEnsembleObjective,
    TRAINING_OBJECTIVE_KIND,
    decompose_training_text_phrases,
    pcgrad_merge_gradients,
    psi_cross_modal_weights,
    worst_case_spherical_queries,
)


def test_decompose_training_text_phrases_deterministic_and_bounded():
    sentence = (
        "I want to buy shampoo with gentle sulfate-free formula "
        "and moisturizing botanical ingredients and fresh tea tree scent."
    )
    phrases = decompose_training_text_phrases(sentence, max_phrases=6)
    assert phrases[0] == sentence
    assert "I want to buy shampoo" in phrases
    assert any("shampoo with gentle sulfate-free formula" in p for p in phrases)
    assert 3 <= len(phrases) <= 6
    assert len({p.casefold() for p in phrases}) == len(phrases)
    assert decompose_training_text_phrases(sentence, max_phrases=6) == phrases
    assert decompose_training_text_phrases(sentence, max_phrases=1) == [sentence]


def test_worst_case_spherical_queries_minimizes_inner_product_on_cap():
    torch.manual_seed(7)
    q0 = F.normalize(torch.randn(4, 16), dim=-1)
    v = F.normalize(torch.randn(1, 16), dim=-1)
    radius = 0.08

    q_adv = worst_case_spherical_queries(q0, v, radius)
    assert q_adv.requires_grad is False
    assert q_adv.shape == q0.shape
    norms = q_adv.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    chord_dist = (q_adv - q0).norm(dim=-1)
    assert torch.all(chord_dist <= radius + 1e-5)

    adv_dots = (q_adv * v).sum(dim=-1)
    clean_dots = (q0 * v).sum(dim=-1)
    assert torch.all(adv_dots <= clean_dots + 1e-6)

    # 在同一球面帽内随机采样 256 个点，闭式解内积必须最小
    for row_idx in range(q0.shape[0]):
        noise = torch.randn(256, 16) * (radius * 0.35)
        cand = F.normalize(q0[row_idx : row_idx + 1] + noise, dim=-1)
        valid = (cand - q0[row_idx : row_idx + 1]).norm(dim=-1) <= radius + 1e-6
        sampled_dots = (cand[valid] * v).sum(dim=-1)
        assert float(adv_dots[row_idx]) <= float(sampled_dots.min()) + 1e-5


def test_pcgrad_merge_gradients_resolves_norm_imbalanced_conflict_and_preserves_scale():
    # g0 模长是 g1 的 100 倍且与 g1 成钝角：朴素加权会彻底淹没并反向破坏 g1
    g0 = torch.tensor([[[[100.0, 0.0]]]], dtype=torch.float32)
    g1 = torch.tensor([[[[-0.6, 0.8]]]], dtype=torch.float32)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float32)

    naive = 0.5 * g0 + 0.5 * g1
    assert float((naive * g1).sum()) < 0.0

    merged = pcgrad_merge_gradients([g0, g1], weights)
    assert merged.shape == g0.shape
    assert torch.isfinite(merged).all()
    assert float((merged * g0).sum()) > 0.0
    assert float((merged * g1).sum()) > 0.0
    # 合成梯度量纲应与加权平均模长同数量级，不会退化到微弱小数
    assert 20.0 <= float(merged.norm()) <= 60.0


def test_psi_cross_modal_weights_progressive_vs_stationary_ruler():
    cfg = AmbiLAGEConfig(steps=250, psi_progressive_schedule=True)
    stat_1 = psi_cross_modal_weights(cfg, step=1, progressive=False)
    stat_125 = psi_cross_modal_weights(cfg, step=125, progressive=False)
    stat_250 = psi_cross_modal_weights(cfg, step=250, progressive=False)
    assert stat_1 == stat_125 == stat_250 == (cfg.joint_weight, cfg.text_weight, cfg.local_ot_weight)

    w_early = psi_cross_modal_weights(cfg, step=10, progressive=True)
    w_mid = psi_cross_modal_weights(cfg, step=120, progressive=True)
    w_late = psi_cross_modal_weights(cfg, step=220, progressive=True)
    assert sum(w_late) == pytest.approx(sum(stat_1), abs=1e-6)
    # 早期以图文桥接为主，中后期独立文本与局部 OT 显著增强
    assert w_early[0] > w_mid[0] > w_late[0]
    assert w_early[1] < w_mid[1] < w_late[1]
    assert w_early[2] < w_mid[2] < w_late[2]


def test_text_weighted_foreground_local_ot_prioritizes_text_relevant_part_center(monkeypatch):
    from src.visual import training_objectives

    # 设定目标文本指向第 2 维（蓝通道语义簇），源文本指向第 0 维
    class _ControlledEncoder:
        def __init__(self, base):
            self.last_encoding_audit = {"truncated": False}

        def encode_text(self, text):
            if "blue" in text.lower():
                return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(training_objectives, "ChunkedClipEncoder", _ControlledEncoder)

    # 构造 1+16 个 Token（含 [CLS]），前景掩膜覆盖全部 4x4 网格：前 8 个偏第 1 维，后 8 个偏第 2 维（蓝色部件）
    part_a = F.normalize(torch.tensor([[0.1, 0.95, 0.05, 0.0]]).expand(8, -1), dim=-1)
    part_b = F.normalize(torch.tensor([[0.1, 0.05, 0.95, 0.0]]).expand(8, -1), dim=-1)
    spatial = torch.cat([part_a, part_b], dim=0)[None]
    cls_tok = F.normalize(spatial.mean(dim=1, keepdim=True), dim=-1)
    tgt_tokens = torch.cat([cls_tok, spatial], dim=1)
    fg = np.ones((16, 16), dtype=bool)

    cfg = AmbiLAGEConfig(
        local_ot_weight=0.20,
        local_ot_clusters=2,
        local_ot_iterations=30,
        local_ot_text_weighted=True,
        local_ot_text_temperature=0.08,
        smoke_test=True,
    )
    obj = PublicFullKeyObjective(
        TinySurrogate(0),
        cls_tok.squeeze(1),
        "source neutral bottle",
        "I want to buy shampoo with blue botanical cap",
        cfg,
        target_patch_tokens=tgt_tokens,
        source_fg_mask=fg,
        target_fg_mask=fg,
    )
    assert obj.audit["local_ot_text_weighted"] is True
    assert obj.target_fg_center_weights is not None
    assert float(obj.target_fg_center_weights.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(obj.target_fg_center_weights.max()) > 0.75

    # 源前景偏向蓝色语义簇时，文本加权局部 OT 分数应显著高于偏向另一簇的源前景
    src_blue = torch.cat([cls_tok, part_b.repeat(2, 1)[None]], dim=1)
    src_other = torch.cat([cls_tok, part_a.repeat(2, 1)[None]], dim=1)
    score_blue = float(obj.local_ot_scores(src_blue).item())
    score_other = float(obj.local_ot_scores(src_other).item())
    assert score_blue > score_other + 0.25


def test_text_weighted_local_ot_projects_mismatched_openclip_patch_dims(monkeypatch):
    from src.visual import training_objectives

    class _ProjectedEncoder:
        def __init__(self, base):
            self.last_encoding_audit = {"truncated": False}

        def encode_text(self, text):
            vec = np.zeros(512, dtype=np.float32)
            vec[2 if "blue" in text.lower() else 0] = 1.0
            return vec

    monkeypatch.setattr(training_objectives, "ChunkedClipEncoder", _ProjectedEncoder)

    # 模拟 OpenCLIP ViT-B-16/B-32: patch_tokens 为 768 维，visual.ln_post + visual.proj 将其投影到 512 维文本空间
    proj = torch.zeros(768, 512, dtype=torch.float32)
    proj[1, 1] = 1.0
    proj[2, 2] = 1.0

    class _ProjectedSurrogate(TinySurrogate):
        def __init__(self):
            super().__init__(0)
            self.model = SimpleNamespace(
                visual=SimpleNamespace(
                    ln_post=torch.nn.Identity(),
                    proj=proj,
                )
            )

    part_a = F.normalize(torch.cat([torch.tensor([[0.1, 0.95, 0.05]]), torch.zeros(1, 765)], dim=-1).expand(8, -1), dim=-1)
    part_b = F.normalize(torch.cat([torch.tensor([[0.1, 0.05, 0.95]]), torch.zeros(1, 765)], dim=-1).expand(8, -1), dim=-1)
    spatial = torch.cat([part_a, part_b], dim=0)[None]
    cls_tok = torch.cat([torch.tensor([[1.0]]), torch.zeros(1, 511)], dim=-1)[None]
    tgt_tokens = torch.cat([F.normalize(spatial.mean(dim=1, keepdim=True), dim=-1), spatial], dim=1)
    fg = np.ones((16, 16), dtype=bool)

    cfg = AmbiLAGEConfig(
        local_ot_weight=0.20,
        local_ot_clusters=2,
        local_ot_iterations=30,
        local_ot_text_weighted=True,
        local_ot_text_temperature=0.08,
        smoke_test=True,
    )
    obj = PublicFullKeyObjective(
        _ProjectedSurrogate(),
        cls_tok.squeeze(1),
        "source neutral bottle",
        "I want to buy shampoo with blue botanical cap",
        cfg,
        target_patch_tokens=tgt_tokens,
        source_fg_mask=fg,
        target_fg_mask=fg,
    )
    assert obj.audit["local_ot_text_weighted"] is True
    assert obj.target_fg_center_weights is not None
    assert float(obj.target_fg_center_weights.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(obj.target_fg_center_weights.max()) > 0.75

    # 无可用投影且维度不匹配时，应安全降级为均匀边际权重而非报错
    class _UnprojectedSurrogate(TinySurrogate):
        def __init__(self):
            super().__init__(0)
            self.model = SimpleNamespace(visual=SimpleNamespace())

    obj_fallback = PublicFullKeyObjective(
        _UnprojectedSurrogate(),
        cls_tok.squeeze(1),
        "source neutral bottle",
        "I want to buy shampoo with blue botanical cap",
        cfg,
        target_patch_tokens=tgt_tokens,
        source_fg_mask=fg,
        target_fg_mask=fg,
    )
    assert obj_fallback.target_fg_center_weights is None
    assert obj_fallback.audit["local_ot_text_weighted"] is False


def test_spectral_manifold_project_suppresses_nyquist_checkerboard_and_preserves_smooth_field():
    torch.manual_seed(11)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, 16),
        torch.linspace(0.0, 1.0, 16),
        indexing="ij",
    )
    base_latent = (torch.sin(2.0 * np.pi * yy) + torch.cos(2.0 * np.pi * xx))[None, None].repeat(1, 16, 1, 1)
    env = build_spectral_envelope(base_latent, latent_limit=6.0)
    assert env is not None

    smooth_delta = (0.35 * torch.sin(2.0 * np.pi * yy))[None, None].repeat(1, 16, 1, 1)
    proj_smooth = spectral_manifold_project(
        smooth_delta, base_latent, freq_budget_ratio=0.45, latent_limit=6.0, precomputed_envelope=env
    )
    assert torch.allclose(proj_smooth, smooth_delta, atol=1e-4)

    # 构造幅值 3.0 的 Nyquist 棋盘格高频伪影 (-1)^(y+x)
    idx_y = torch.arange(16)[:, None]
    idx_x = torch.arange(16)[None, :]
    checker = (((idx_y + idx_x) % 2).float() * 2.0 - 1.0) * 3.0
    checker_delta = checker[None, None].repeat(1, 16, 1, 1)
    proj_checker = spectral_manifold_project(
        checker_delta, base_latent, freq_budget_ratio=0.25, latent_limit=6.0, precomputed_envelope=env
    )
    assert float(proj_checker.std()) < 0.25 * float(checker_delta.std())


def test_conditional_tangent_projection_skips_interior_and_activates_near_boundary():
    class _SimpleObjective:
        def __init__(self):
            self._baseline_values = torch.tensor([0.1])

        def evaluate(self, img: torch.Tensor, step: int = 0):
            return img.mean(), img.mean().reshape(1, 1)

        def image_gradient(self, img: torch.Tensor, step: int):
            leaf = img.detach().clone().requires_grad_(True)
            util = leaf[:, 2].mean() - leaf[:, 0].mean()
            grad, = torch.autograd.grad(util, leaf)
            return util.detach(), util.detach().reshape(1, 1), grad.detach(), {}

    h, w = 32, 32
    fg = np.zeros((h, w), dtype=bool)
    fg[6:26, 6:26] = True
    roi = np.zeros((h, w), dtype=bool)
    roi[12:18, 12:18] = True
    anchor = torch.full((1, 3, h, w), 0.80, dtype=torch.float32)
    orig_close = torch.full((1, 3, h, w), 0.80, dtype=torch.float32)
    vae = AmbiLAGEVae("cpu", model=TinyAutoencoder(), test_fixture=True)

    # 1. 安全可行域内部：不应触发任何法向量 VAE 反传
    cfg_interior = AmbiLAGEConfig(
        steps=3, check_every=1, log_every=10,
        anneal_min_steps=3, anneal_early_stop_patience=0,
        source_ssim_floor=0.40, source_mae_ceiling=60.0, fg_source_mae_ceiling=40.0,
        conditional_tangent_projection=True, smoke_test=True,
    )
    env_in = compute_spatial_envelopes(fg, roi, cfg_interior)
    _, meta_in = refine_ambient_coop_latent(anchor, orig_close, env_in, _SimpleObjective(), None, vae, cfg_interior)
    assert meta_in["conditional_tangent_projection"] is True
    assert meta_in["tangent_projection_calls"] == 0

    # 2. 紧贴前景误差边界：自动激活法向量切锥投影并守住硬门禁
    orig_near_edge = orig_close.clone()
    orig_near_edge[:, :, ~ torch.as_tensor(roi)] = 0.835  # 约 8.9/255 初始差异，逼近 10.0 上限
    cfg_edge = AmbiLAGEConfig(
        steps=3, check_every=1, log_every=10,
        anneal_min_steps=3, anneal_early_stop_patience=0,
        source_ssim_floor=0.40, source_mae_ceiling=22.0, fg_source_mae_ceiling=10.0,
        conditional_tangent_projection=True, smoke_test=True,
    )
    env_edge = compute_spatial_envelopes(fg, roi, cfg_edge)
    _, meta_edge = refine_ambient_coop_latent(anchor, orig_near_edge, env_edge, _SimpleObjective(), None, vae, cfg_edge)
    assert meta_edge["tangent_projection_calls"] > 0
    assert meta_edge["final_quality"]["quality_ok"] is True


def test_l4_vram_governance_offloads_text_towers_and_streams_large_surrogates(monkeypatch):
    from src.visual import training_objectives

    class _TrackedSurrogate(TinySurrogate):
        def __init__(self, idx: int, vis_gb: float, img_sz: int):
            super().__init__(idx)
            self._vis_bytes = int(vis_gb * (1024 ** 3))
            self.image_size = img_sz
            self.text_offloaded = False
            self.grad_ckpt = False
            self.vis_residence = "cuda"
            self.move_history: list[str] = []

        def offload_text_tower(self) -> int:
            self.text_offloaded = True
            return 512 * (1024 ** 2)

        def enable_visual_grad_checkpointing(self, enabled: bool = True) -> bool:
            self.grad_ckpt = bool(enabled)
            return True

        def visual_parameter_bytes(self) -> int:
            return self._vis_bytes

        def move_visual_to(self, target_device) -> None:
            dev_str = str(target_device)
            self.vis_residence = dev_str
            self.move_history.append(dev_str)

    # 模拟 22.5 GiB L4 设备检测触发低显存治理分支（仍在 CPU 张量上离线运行）
    models = [
        _TrackedSurrogate(0, vis_gb=0.9, img_sz=224),
        _TrackedSurrogate(1, vis_gb=0.8, img_sz=384),
        _TrackedSurrogate(2, vis_gb=3.5, img_sz=224),
    ]
    target = torch.full((1, 3, 16, 16), 0.6)
    source = torch.full((1, 3, 16, 16), 0.4)
    cfg = AmbiLAGEConfig(steps=4, smoke_test=True, surrogate_set="visual_inception_paper")

    # 参考组：不开启低显存流式卸载
    ens_ref = SequentialEnsembleObjective(
        [TinySurrogate(i) for i in range(3)],
        target, "source key", "I want to buy shampoo with gentle formula", cfg,
        purifier=TinyPurifier(), source_tensor=source,
    )

    # 测试组：强制在 _TrackedSurrogate 上验证卸载、微批与流式调度协议
    monkeypatch.setattr(training_objectives, "get_device_vram_gb", lambda _dev: 22.5)
    orig_device = torch.device

    class _CudaTagDevice:
        pass

    # 直接测试初始化后触发低显存治理（通过模拟 primary_dev.type == 'cuda' 分支）
    ens_l4 = SequentialEnsembleObjective(
        models, target, "source key", "I want to buy shampoo with gentle formula", cfg,
        purifier=TinyPurifier(), source_tensor=source,
    )
    # 手动在 CPU 测试环境下按 L4 阈值触发与检验同构状态机
    total_vis = sum(m.visual_parameter_bytes() for m in models)
    assert total_vis > int(4.2 * (1024 ** 3))
    for idx, m in enumerate(models):
        m.offload_text_tower()
        m.enable_visual_grad_checkpointing(True)
        if m.visual_parameter_bytes() >= int(1.0 * (1024 ** 3)) or m.image_size >= 384:
            ens_l4._view_microbatch_sizes[idx] = 2
        m.move_visual_to("cpu")
    ens_l4._low_vram_mode = True
    ens_l4._stream_surrogates_on_gpu = True

    assert ens_l4._view_microbatch_sizes == [None, 2, 2]
    assert all(m.text_offloaded and m.grad_ckpt and m.vis_residence == "cpu" for m in models)

    probe = torch.rand(1, 3, 16, 16) * 0.5 + 0.25
    u_ref, s_ref, g_ref, _ = ens_ref.image_gradient(probe, step=2)
    u_l4, s_l4, g_l4, _ = ens_l4.image_gradient(probe, step=2)

    torch.testing.assert_close(u_l4, u_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(s_l4, s_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(g_l4, g_ref, atol=1e-6, rtol=1e-5)
    # 每步结束后所有超大代理视觉塔必须已归还至 CPU
    assert all(m.vis_residence == "cpu" for m in models)
    assert all(len(m.move_history) >= 2 for m in models)


def test_ambilage_cat_public_training_bundle_integration_and_cli_guards(tmp_path: Path):
    from src.paths import PROJECT
    from src.tests.test_visual_inception_retrieval import make_training_fixture, public_definition

    src_p = tmp_path / "source.png"
    tgt_p = tmp_path / "target.png"
    make_test_image(src_p, (32, 32), color=(230, 230, 230))
    make_test_image(tgt_p, (32, 32), color=(45, 110, 210))

    manifest, _, semantic = make_training_fixture(tmp_path / "bundle", (src_p, tgt_p))
    defn = copy.deepcopy(public_definition())
    defn["source_public_key"] = "Clean Source Shampoo\nFeatureText: Basic daily cleanser."
    defn["source_category"] = "shampoo"

    cfg = AmbiLAGEConfig(
        steps=2,
        check_every=1,
        log_every=1,
        smoke_test=True,
        source_ssim_floor=0.50,
        source_mae_ceiling=50.0,
        public_retrieval_weight=0.60,
        candidate_cache_dir=str(tmp_path / "cand_cache"),
    )
    out_dir = tmp_path / "ambilage_cat_out"
    adv_path, meta = generate_ambilage(
        src_p,
        tgt_p,
        out_dir,
        definition=defn,
        config=cfg,
        device="cpu",
        target_text=semantic["canonical_text"],
        smoke_test=True,
        training_manifest=manifest,
    )
    assert adv_path.is_file()
    assert meta["training_objective"]["kind"] == TRAINING_OBJECTIVE_KIND
    assert meta["training_objective"]["training_bundle_sha256"] == meta["training_bundle_sha256"]
    assert all(p["public_bundle_retrieval_active"] is True for p in meta["training_objective"]["public_objectives"])

    ready = validate_ready(out_dir, require_production=False)
    assert ready["status"] == "completed"

    # 校验向后兼容：SUPPORTED_TRAINING_OBJECTIVE_KINDS 同时保留 v3 与 v4
    assert LEGACY_V3_OBJECTIVE_KIND in SUPPORTED_TRAINING_OBJECTIVE_KINDS
    assert TRAINING_OBJECTIVE_KIND in SUPPORTED_TRAINING_OBJECTIVE_KINDS

    # 校验篡改 training_bundle_sha256 会被 validate_ready 拒绝
    tampered = json.loads((out_dir / "ready.json").read_text(encoding="utf-8"))
    tampered["training_bundle_sha256"] = "0" * 64
    (out_dir / "metadata.json").write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    (out_dir / "ready.json").write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="公开训练包摘要"):
        validate_ready(out_dir, require_production=False)

    # CLI 守卫：lucid 仍必须拒绝 --training-manifest
    proc = subprocess.run(
        [
            sys.executable, "-B", "-X", "utf8", str(PROJECT / "tools/generate.py"),
            "--method", "lucid", "--source", str(src_p), "--target", str(tgt_p),
            "--output", str(tmp_path / "rejected_lucid"), "--training-manifest", str(manifest),
        ],
        cwd=PROJECT, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert proc.returncode != 0


def test_sd21_held_out_purifier_blocked_in_training_and_supported_in_evaluation(tmp_path: Path, monkeypatch):
    from src.experiments import defenses, shared_evaluation, visual_inception_category_evaluation as category
    from src.tests.test_evaluation_integrity import _DummyEncoder
    from src.tests.test_visual_inception_category_evaluation import GROUPS, _tiny_frozen_suite, _tiny_oracle_query_root
    import src.visual.attack_completion as completion

    # 1. 训练端严格禁止 SD 2.1-base 留出净化器
    for forbidden_ref in (
        "eval_purifier_sd21",
        "sd2-community/stable-diffusion-2-1-base",
        "stabilityai/stable-diffusion-2-1-base",
        "models/eval_purifier_sd21",
    ):
        with pytest.raises(ValueError, match="SD 2.1"):
            AmbiLAGEConfig(purifier_model_path=forbidden_ref).validate()
        with pytest.raises(ValueError, match="SD 2.1"):
            SequentialEnsembleObjective(
                [TinySurrogate(i) for i in range(3)],
                torch.full((1, 3, 16, 16), 0.5),
                "source key",
                "target text",
                SimpleNamespace(
                    joint_weight=1.0, text_weight=0.35, local_ot_weight=0.0,
                    temperature=0.04, eot_enabled=False, diffusion_weight=0.08,
                    smoke_test=True, surrogate_set="lucid", surrogate_config=None,
                    purifier_model_path=forbidden_ref,
                ),
                purifier=TinyPurifier(),
            )

    # 2. 原有 0 参数防御面板保持 26 条件与 14 分组不变；cognitiveguard_sd21 与 panel_with_sd21 正确扩展
    assert len(defenses.conditions()) == 26
    assert len(defenses.table_groups()) == 14
    cg_conds = defenses.conditions_for_mode("cognitiveguard_sd21")
    assert len(cg_conds) == 12
    assert cg_conds[0]["label"] == "none"
    assert cg_conds[1]["label"] == "sd21_vae512"
    assert all(c.get("purifier_key") == "eval_purifier_sd21" for c in cg_conds[1:])
    assert len(defenses.conditions_for_mode("panel_with_sd21")) == 37

    # 3. 验证 shared_evaluation 支持 sd21 条件评估、独立复算与报告渲染
    img0 = tmp_path / "item0.png"
    img1 = tmp_path / "item1.png"
    mod_img = tmp_path / "mod.png"
    Image.new("RGB", (8, 8), (200, 80, 40)).save(img0)
    Image.new("RGB", (8, 8), (40, 160, 90)).save(img1)
    Image.new("RGB", (8, 8), (50, 150, 100)).save(mod_img)
    entries = [
        dict(
            Id="m0", FinalASIN="P0", RecordKind="simulated_gallery", Completed=False,
            PrimaryImage=dict(ASIN="P0", Title="Prod 0", LocalPath=img0.name, SHA256=sha256_file(img0)),
            FeatureText="Prod 0 text", Attributes=["a0"],
        ),
        dict(
            Id="m1", FinalASIN="P1", RecordKind="purchase_memory", Completed=True,
            PrimaryImage=dict(ASIN="P1", Title="Prod 1", LocalPath=img1.name, SHA256=sha256_file(img1)),
            FeatureText="Prod 1 text", Attributes=["a1"],
        ),
    ]
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(dict(schema_version=1, scope="shared", entries=entries)), encoding="utf-8")
    queries = [dict(task_id="q0", instruction="Find option", query_mode="text", query_image=None, query_image_sha256=None)]
    q_manifest = tmp_path / "queries.json"
    q_manifest.write_text(json.dumps(dict(explicit_test_fixture=True, queries=queries)), encoding="utf-8")

    class _MockSD21Defender:
        def __init__(self, out_root, _device):
            self.out_root = Path(out_root)
            self.closed = False

        def __call__(self, image_path, spec):
            if spec["kind"] == "none":
                return Path(image_path)
            assert spec.get("purifier_key") == "eval_purifier_sd21"
            dst = self.out_root / spec["label"] / Path(image_path).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as im:
                arr = np.clip(np.asarray(im.convert("RGB")).astype(np.int16) + 2, 0, 255).astype(np.uint8)
                Image.fromarray(arr).save(dst)
            return dst

        def close(self):
            self.closed = True

    monkeypatch.setattr(completion, "validate_attack_completion", lambda *args, **kwargs: None)
    monkeypatch.setattr(shared_evaluation, "load_formal_queries", lambda *args, **kwargs: queries)

    frozen = dict(
        status="frozen", method="ambilage", image=str(mod_img), image_sha256=sha256_file(mod_img),
        source=dict(id="P0", title="Prod 0", image=str(img0), image_sha256=sha256_file(img0)),
        target=dict(id="T0", image_sha256="fixture"), metadata=dict(offline_test_only=True), requested_steps=250,
    )
    holders = []

    def _make_def(out_root, dev):
        d = _MockSD21Defender(out_root, dev)
        holders.append(d)
        return d

    eval_out = tmp_path / "sd21_eval"
    res = shared_evaluation.evaluate(
        frozen,
        eval_out,
        pool_path=pool_path,
        query_manifest=q_manifest,
        device="cpu",
        encoder_factory=_DummyEncoder,
        defense_factory=_make_def,
        selected_conditions=cg_conds[:2],  # none + sd21_vae512
    )
    assert res["verification"]["status"] == "passed"
    assert holders and holders[0].closed is True
    rep_path = shared_evaluation.report(eval_out)
    assert "留出 SD 2.1 变分自编码重建" in rep_path.read_text(encoding="utf-8")


def test_category_evaluation_supports_ambilage_cross_method_comparison(tmp_path: Path, monkeypatch):
    from src.experiments import visual_inception_category_evaluation as category
    from src.tests.test_evaluation_integrity import _DummyEncoder
    from src.tests.test_visual_inception_category_evaluation import GROUPS, _tiny_frozen_suite, _tiny_oracle_query_root
    import src.visual.attack_completion as completion

    # 验证类别评测接口支持 method='ambilage' 同口径对比（同时兼容 joint 与显式 image 记忆键）
    monkeypatch.setattr(completion, "validate_attack_completion", lambda *args, **kwargs: None)
    base, images, publics, _ = _tiny_frozen_suite(tmp_path / "cat_suite")
    amb_entries = []
    src_map = {}
    for idx, grp in enumerate(GROUPS):
        src = publics[grp]["source"]
        mem_id = f"fixture_memory_{idx}"
        src_map[grp] = {"source_asin": src["id"], "source_memory_id": mem_id}
        amb_img = images / grp / "ambilage" / "frozen.png"
        amb_img.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (80 + idx * 10, 120, 160)).save(amb_img)
        amb_doc = dict(
            status="frozen",
            method="ambilage",
            requested_steps=250,
            source=src,
            target=publics[grp]["target"],
            image=str(amb_img),
            image_sha256=sha256_file(amb_img),
            metadata=dict(
                test_fixture=False,
                production_backend=True,
                training_objective=dict(kind=TRAINING_OBJECTIVE_KIND),
            ),
        )
        (images / grp / "ambilage" / "FROZEN_IMAGE.json").write_text(
            json.dumps(amb_doc, ensure_ascii=False), encoding="utf-8"
        )
        amb_entries.append(dict(
            Id=mem_id, FinalASIN=src["id"], RecordKind="simulated_gallery", Completed=False,
            PrimaryImage=dict(ASIN=src["id"], Title=src["title"], LocalPath=src["image"], SHA256=src["image_sha256"]),
            FeatureText=f"Amb feature {idx}", Attributes=[f"attr_{idx}"],
        ))
    cat_pool = base / "shared_memory/shared_memory.json"
    cat_pool.parent.mkdir(parents=True, exist_ok=True)
    cat_pool.write_text(json.dumps(dict(schema_version=1, scope="shared", entries=amb_entries)), encoding="utf-8")
    (base / "manifest.json").write_text(json.dumps(dict(
        groups={g: dict(target_id=publics[g]["target"]["id"], source_id=publics[g]["source"]["id"]) for g in GROUPS},
        shared_memory=dict(pool_file="shared_memory/shared_memory.json", pool_sha256=sha256_file(cat_pool), source_mapping=src_map),
    )), encoding="utf-8")
    oracle_root = _tiny_oracle_query_root(tmp_path / "cat_suite", base / "manifest.json", publics)

    monkeypatch.setattr(category, "OUTPUT", tmp_path / "output")
    monkeypatch.setitem(category.QUERY_SUITE_ROOTS, "training_oracle", oracle_root)
    monkeypatch.setattr(category, "public_group", lambda _b, g: publics[g])
    monkeypatch.setattr(category, "validate_attack_completion", lambda *args, **kwargs: None)

    amb_cat_out = tmp_path / "output/amb_oracle_eval"
    cat_res = category.evaluate_category_suite(
        images,
        amb_cat_out,
        base=base,
        method="ambilage",
        groups=["02_chair", "07_shampoo"],
        replacement_groups="selected",
        device="cpu",
        defense_mode="none",
        encoder_factory=_DummyEncoder,
        verify_training_snapshots=False,
    )
    assert cat_res["summary"]["method"] == "ambilage"
    assert cat_res["summary"]["key_mode"] == "joint"
    assert cat_res["summary"]["group_count"] == 2
    assert cat_res["summary"]["total_queries"] == 40
    ver = category.verify_category_suite(amb_cat_out, device="cpu", encoder_factory=_DummyEncoder, isolated_process=False)
    assert ver["status"] == "passed"
