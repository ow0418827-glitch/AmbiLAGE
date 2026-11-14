"""Stage 4: 原生环境光背景协同与微光影潜空间优化 (Latent Refinement)。

实现内容：
1. AutoencoderKL 冻结自编码器接入，具备 A100 显存自适应梯度检查点管理；
2. 联合优化潜差分 (delta_z) 与多尺度环境光场 (a_global, a_grid)；
3. 可行无截断中点对称展开 [r_min, r_max]，杜绝 255 截断白岛与边缘光晕；
4. 检索效用与空间方差正则化联合导引；
5. 正交切空间投影与回退守门 (SSIM >= 0.78, MAE <= 20.0)；
6. 512GB 磁盘防护策略：严禁中间 step PNG 刷盘，仅在内存维护快照。
"""
from __future__ import annotations

import copy
import gc
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.paths import model_path
from src.visual.hardware import should_enable_gradient_checkpointing
from src.visual.image_utils import quantize_ste, read_rgb
from .ambient import AmbientCoopField
from .config import AmbiLAGEConfig
from .objectives import (
    PublicFullKeyObjective,
    compute_spatial_variance_penalty,
    source_ssim,
    tangent_projection,
)


class AmbiLAGEVae:
    """原生十六通道模型封装；严格加载且支持缩放、平移及可微解码。"""
    def __init__(self, device, custom_model_dir=None, enable_checkpointing=None, *, model=None, test_fixture=False):
        self.device = torch.device(device)
        self.test_fixture = bool(test_fixture)
        if model is None:
            from diffusers import AutoencoderKL
            from .backends import local_model
            directory = local_model(custom_model_dir, "editor")
            kwargs = {"subfolder": "vae"} if (Path(directory)/"vae").is_dir() else {}
            model = AutoencoderKL.from_pretrained(directory, local_files_only=True, torch_dtype=torch.float32, **kwargs)
        if not test_fixture and getattr(model.config, "latent_channels", None) != 16:
            raise ValueError("正式潜空间细化必须使用十六通道变分自编码器")
        self.vae = model.eval().requires_grad_(False).to(self.device)
        use_ckpt = should_enable_gradient_checkpointing(self.device, override=enable_checkpointing)
        if use_ckpt:
            self.vae.enable_gradient_checkpointing()
        self.identity = dict(kind="test_fixture" if test_fixture else "flux_16_channel_vae", precision="float32",
                             posterior="mode", latent_channels=getattr(model.config,"latent_channels",None),
                             gradient_checkpointing=use_ckpt)

    def encode(self, image):
        config = self.vae.config
        return (self.vae.encode(image*2.-1.).latent_dist.mode().float() - getattr(config,"shift_factor",0.)) * config.scaling_factor

    def decode_raw(self, latent, target_shape):
        config = self.vae.config
        unscaled = latent / config.scaling_factor + getattr(config,"shift_factor",0.)
        decoded = self.vae.decode(unscaled).sample.float()/2. + .5
        return F.interpolate(decoded, size=target_shape[-2:], mode="bicubic", align_corners=False, antialias=True)


def refine_ambient_coop_latent(
    anchor_tensor: torch.Tensor,
    original_tensor: torch.Tensor,
    envelopes: dict[str, Any],
    objective: PublicFullKeyObjective,
    surrogate: Any,
    vae: AmbiLAGEVae,
    config: AmbiLAGEConfig,
    output_dir: Path | str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Stage 4: 执行原生环境光协同背景调整与主体微光影 350 步潜空间优化。"""
    device = anchor_tensor.device
    shape = anchor_tensor.shape
    h, w = shape[-2:]

    # 1. 提取空间掩码与包络张量
    fg_mask_t = torch.as_tensor(envelopes["foreground"] & (~envelopes["roi"]), device=device, dtype=torch.float32)[None, None]
    bg_mask_t = torch.as_tensor(envelopes["background"] & (~envelopes["roi"]), device=device, dtype=torch.float32)[None, None]
    support_mask = torch.as_tensor(envelopes["support_mask"], device=device, dtype=torch.float32)[None, None]
    eps_fg_map = torch.as_tensor(envelopes["fg_level_array"] / 255.0, device=device, dtype=torch.float32)[None, None]
    eps_bg_map = torch.as_tensor(envelopes["bg_level_array"] / 255.0, device=device, dtype=torch.float32)[None, None]
    w_bg_mod = torch.as_tensor(envelopes["w_bg_mod"], device=device, dtype=torch.float32)[None, None]
    fg_bool = torch.as_tensor(envelopes["foreground"], device=device, dtype=torch.bool)
    bg_bool = ~fg_bool
    roi_bool = torch.as_tensor(envelopes["roi"], device=device, dtype=torch.bool)

    # 2. 构造环境光协同光场模块
    ambient_field = AmbientCoopField(
        anchor=anchor_tensor,
        eps_fg_map=eps_fg_map,
        eps_bg_map=eps_bg_map,
        w_bg_mod=w_bg_mod,
        fg_mask_t=fg_mask_t,
        bg_mask_t=bg_mask_t,
        support_mask=support_mask,
        config=config,
    )

    # 3. 计算 VAE 潜空间基准 latent
    with torch.no_grad():
        w_size = config.latent_working_size
        ratio = w_size / max(h, w)
        work_h = max(64, round(h * ratio / 8) * 8)
        work_w = max(64, round(w * ratio / 8) * 8)
        work_img = F.interpolate(
            anchor_tensor, size=(work_h, work_w), mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
        base_latent = vae.encode(work_img).detach()
        reference_decoded = vae.decode_raw(base_latent, shape).detach()

    # 4. 初始化可学习潜差分与优化器
    delta = nn.Parameter(torch.zeros_like(base_latent))
    optimizer = torch.optim.Adam([
        {"params": [delta], "lr": config.latent_learning_rate},
        {"params": [ambient_field.ambient_global, ambient_field.ambient_grid], "lr": config.ambient_learning_rate},
    ])

    def snapshot_params() -> dict[str, torch.Tensor]:
        return {
            "delta": delta.detach().clone(),
            "ambient_global": ambient_field.ambient_global.detach().clone(),
            "ambient_grid": ambient_field.ambient_grid.detach().clone(),
        }

    def load_params(snap: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            delta.copy_(snap["delta"])
            ambient_field.ambient_global.copy_(snap["ambient_global"])
            ambient_field.ambient_grid.copy_(snap["ambient_grid"])

    def compose_with_raw() -> tuple[torch.Tensor, torch.Tensor]:
        decoded = vae.decode_raw(base_latent + delta, shape)
        diff = decoded - reference_decoded
        residual = ambient_field.compute_residual(diff)
        raw = anchor_tensor + residual
        quantized = quantize_ste(raw.clamp(0.0, 1.0))
        return quantized, raw

    def compose() -> torch.Tensor:
        q, _ = compose_with_raw()
        return q

    @torch.no_grad()
    def evaluate_quality(image: torch.Tensor) -> dict[str, Any]:
        ss = float(source_ssim(image, original_tensor))
        diff_orig = (image - original_tensor).abs() * 255.0
        diff_anchor = (image - anchor_tensor).abs() * 255.0
        mae = float(diff_orig.mean())
        fg_mae = float(diff_orig[:, :, fg_bool].mean()) if fg_bool.any() else 0.0
        bg_mae = float(diff_orig[:, :, bg_bool].mean()) if bg_bool.any() else 0.0
        anchor_mae = float(diff_anchor.mean())
        return {
            "source_ssim": ss,
            "source_mae_levels": mae,
            "fg_source_mae_levels": fg_mae,
            "bg_source_mae_levels": bg_mae,
            "anchor_mae_levels": anchor_mae,
            "quality_ok": bool(ss >= config.source_ssim_floor and mae <= config.source_mae_ceiling),
        }

    history: list[dict[str, Any]] = []
    tick = time.perf_counter()
    accepted = 0
    rejected = 0

    best_snap = snapshot_params()
    best_row = evaluate_quality(compose())
    initial_img = compose()
    if not best_row["quality_ok"]:
        raise RuntimeError("自然锚点未通过结构/误差门禁，不能标记为完整成图")
    if hasattr(objective, "evaluate"):
        init_utility, init_scores = objective.evaluate(initial_img)
    else:
        init_features = surrogate.encode_image(initial_img).global_embedding
        init_utility, init_scores = objective.compute_utility(init_features, config)
    init_penalty, _ = compute_spatial_variance_penalty(initial_img, anchor_tensor, None, fg_mask_t, bg_mask_t, config)
    best_row.update({
        "step": 0,
        "training_utility": float(init_utility.detach()),
        "selection_utility": float(init_utility.detach()) - float(init_penalty.detach()),
        "joint_scores": init_scores.detach().cpu().tolist(),
    })
    history.append(best_row)

    # 5. 优化循环 (原生 4 阶段第 4 阶段细化)
    for step in range(1, config.steps + 1):
        # 5.1 计算 SSIM 与 MAE 关于图像的法向量约束，准备切空间投影
        image_eval = compose().detach().requires_grad_(True)
        normals = []
        slacks = []

        structure = source_ssim(image_eval, original_tensor)
        n_ssim, = torch.autograd.grad(structure, image_eval)
        normals.append(n_ssim.detach())
        slacks.append(float(structure.detach()) - config.source_ssim_floor)

        mae_metric = (image_eval - original_tensor).abs().mean()
        n_mae, = torch.autograd.grad(-mae_metric, image_eval)
        normals.append(n_mae.detach())
        slacks.append(config.source_mae_ceiling / 255.0 - float(mae_metric.detach()))

        old_snap = snapshot_params()
        old_opt_state = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)

        # 5.2 前向合成与切空间梯度反向传播
        decoded_img, raw_img = compose_with_raw()

        latent_normals = []
        for n_img in normals:
            d_latent, = torch.autograd.grad(decoded_img, delta, grad_outputs=n_img, retain_graph=True)
            latent_normals.append(d_latent.detach())

        if hasattr(objective, "image_gradient"):
            utility, scores, image_gradient, objective_metrics = objective.image_gradient(decoded_img.detach(), step)
        else:
            feat = surrogate.encode_image(decoded_img).global_embedding
            utility, scores = objective.compute_utility(feat, config)
            image_gradient = None
        penalty, reg_metrics = compute_spatial_variance_penalty(
            decoded_img, anchor_tensor, raw_img, fg_mask_t, bg_mask_t, config
        )
        if image_gradient is not None:
            torch.autograd.backward((decoded_img, penalty), (-image_gradient, torch.ones_like(penalty)))
        else:
            loss = -utility + penalty
            loss.backward()

        for p in (delta, ambient_field.ambient_global, ambient_field.ambient_grid):
            if p.grad is not None:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

        # 学习率动态衰减
        decay = 1.0 - 0.75 * (step / max(config.steps, 1))
        optimizer.param_groups[0]["lr"] = config.latent_learning_rate * decay
        optimizer.param_groups[1]["lr"] = config.ambient_learning_rate * decay
        optimizer.step()

        # 5.3 切空间投影与回退守门 (Tangent Space Projection & Backtracking Gate)
        with torch.no_grad():
            delta.clamp_(-config.latent_limit, config.latent_limit)
            if config.latent_smoothing_mix > 0.0:
                smooth = F.avg_pool2d(F.pad(delta, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
                delta.lerp_(smooth, config.latent_smoothing_mix)

            step_vector = delta - old_snap["delta"]
            projected = tangent_projection(step_vector, latent_normals, slacks)
            proposed_delta = (old_snap["delta"] + projected).clamp(-config.latent_limit, config.latent_limit)
            proposed_global = ambient_field.ambient_global.clamp(-config.ambient_limit, config.ambient_limit).clone()
            proposed_grid = ambient_field.ambient_grid.clamp(-config.ambient_limit, config.ambient_limit).clone()

            step_accepted = False
            for backtrack in range(config.guard_backtracks + 1):
                scale = 0.5 ** backtrack
                delta.copy_(old_snap["delta"] + scale * (proposed_delta - old_snap["delta"]))
                ambient_field.ambient_global.copy_(
                    old_snap["ambient_global"] + scale * (proposed_global - old_snap["ambient_global"])
                )
                ambient_field.ambient_grid.copy_(
                    old_snap["ambient_grid"] + scale * (proposed_grid - old_snap["ambient_grid"])
                )
                q = evaluate_quality(compose())
                if q["quality_ok"]:
                    step_accepted = True
                    break

            if step_accepted:
                accepted += 1
            else:
                load_params(old_snap)
                optimizer.load_state_dict(old_opt_state)
                rejected += 1

        # 5.4 进度记录与检查 (512GB 磁盘防护：零 step PNG 刷盘)
        if step % config.check_every == 0 or step == config.steps:
            with torch.no_grad():
                cur_img = compose()
                q_row = evaluate_quality(cur_img)
                if hasattr(objective, "evaluate"):
                    u_val, s_val = objective.evaluate(cur_img, step=0)
                else:
                    c_feat = surrogate.encode_image(cur_img).global_embedding
                    u_val, s_val = objective.compute_utility(c_feat, config)
                _, current_reg = compute_spatial_variance_penalty(cur_img, anchor_tensor, None, fg_mask_t, bg_mask_t, config)
                sel_utility = float(u_val) - current_reg["total_penalty"]
                q_row.update({
                    "step": step,
                    "training_utility": float(u_val),
                    "selection_utility": sel_utility,
                    "regularization": current_reg["total_penalty"],
                    "joint_scores": s_val.detach().cpu().tolist(),
                    "accepted": accepted,
                    "rejected": rejected,
                })
                history.append(q_row)

                if q_row["quality_ok"] and (sel_utility > best_row["selection_utility"] or not best_row.get("quality_ok")):
                    best_row = q_row
                    best_snap = snapshot_params()

                if step % config.log_every == 0 or step == config.steps:
                    print(
                        f"AmbiLAGE 细化 [{step}/{config.steps}] "
                        f"Utility={u_val:.4f} SSIM={q_row['source_ssim']:.4f} "
                        f"MAE={q_row['source_mae_levels']:.2f} "
                        f"(Accepted={accepted}, Rejected={rejected})",
                        flush=True,
                    )

    # 6. 加载最佳快照并生成最终成图
    load_params(best_snap)
    final_tensor = compose().detach()

    # 7. 严格质量与文字区域零改动断言
    final_rgb = np.rint(final_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    anchor_rgb = np.rint(anchor_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    fg_np = envelopes["foreground"]
    bg_np = envelopes["background"]
    roi_np = envelopes["roi"]

    diff_anchor = np.abs(final_rgb.astype(np.int16) - anchor_rgb.astype(np.int16))
    if roi_np.any() and np.any(diff_anchor[roi_np]):
        max_roi_diff = int(diff_anchor[roi_np].max())
        raise AssertionError(f"印刷文字区域保护断言失败：检测到 {max_roi_diff} 级改动 (change[roi] == 0 失败)")

    fg_max = int(diff_anchor[fg_np].max()) if fg_np.any() else 0
    bg_max = int(diff_anchor[bg_np].max()) if bg_np.any() else 0
    if fg_max > config.latent_levels:
        raise AssertionError(f"商品主体微光影超出预算: {fg_max} > {config.latent_levels}")
    if bg_max > config.background_levels:
        raise AssertionError(f"背景光场超出预算: {bg_max} > {config.background_levels}")

    metadata: dict[str, Any] = {
        "status": "completed",
        "stage": "latent_generation_refinement",
        "steps_requested": config.steps,
        "steps_completed": config.steps,
        "best_step": best_row["step"],
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "final_quality": best_row,
        "fg_latent_max_levels": fg_max,
        "bg_latent_max_levels": bg_max,
        "roi_max_levels": 0,
        "elapsed_seconds": float(time.perf_counter() - tick),
        "zero_pixel_optimization": True,
        "vae": vae.identity,
        "history": history,
        "auxiliary_levels": 0,
    }

    return final_tensor, metadata
