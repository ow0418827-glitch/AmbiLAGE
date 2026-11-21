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
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import threading
import time
from typing import Any, Mapping
import uuid

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
    compute_fg_chroma_tv_levels,
    compute_spatial_variance_penalty,
    extract_oklab_luma,
    source_luma_mae_levels,
    source_ssim,
    srgb_to_oklab,
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


STAGE4_CHECKPOINT_FILENAME = "checkpoint_latest.pt"


def build_spectral_envelope(
    base_latent: torch.Tensor,
    latent_limit: float = 6.0,
) -> torch.Tensor | None:
    """预计算十六通道潜空间的 1/f 自然频谱幅度参考包络（满足二维 Parseval 能量归一化）。"""
    h_lat, w_lat = int(base_latent.shape[-2]), int(base_latent.shape[-1])
    if h_lat < 4 or w_lat < 4:
        return None
    device = base_latent.device
    fy = torch.fft.fftfreq(h_lat, device=device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(w_lat, device=device, dtype=torch.float32)
    fx_full = torch.fft.fftfreq(w_lat, device=device, dtype=torch.float32)
    norm_diag = math.sqrt(0.5)
    rho = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / norm_diag
    rho_full = torch.sqrt(fy[:, None] ** 2 + fx_full[None, :] ** 2) / norm_diag

    pink_full = (1.0 + 6.0 * rho_full) ** (-1.35)
    pink_rms = torch.sqrt((pink_full ** 2).mean().clamp_min(1e-12))
    pink_rfft = (((1.0 + 6.0 * rho) ** (-1.35)) / pink_rms)[None, None]

    base_f = base_latent.detach().float()
    base_centered = base_f - base_f.mean(dim=(-2, -1), keepdim=True)
    f_base = torch.fft.rfft2(base_centered, norm="ortho")
    base_amp = torch.sqrt((f_base.abs() ** 2).mean(dim=1, keepdim=True) + 1e-12)
    if min(base_amp.shape[-2], base_amp.shape[-1]) >= 3:
        # 沿 fy 维采用圆周延拓以保持实信号共轭对称，沿 fx 维复制边界
        pad_h = torch.cat([base_amp[:, :, -1:, :], base_amp, base_amp[:, :, :1, :]], dim=-2)
        pad_hw = F.pad(pad_h, (1, 1, 0, 0), mode="replicate")
        base_amp = F.avg_pool2d(pad_hw, 3, stride=1)

    sigma_spatial = max(float(base_centered.std().item()), 0.50 * float(latent_limit))
    envelope = torch.maximum(base_amp, sigma_spatial * pink_rfft)
    dc_floor = float(latent_limit) * math.sqrt(float(h_lat * w_lat))
    envelope = envelope.clone()
    envelope[..., 0, 0] = torch.maximum(envelope[..., 0, 0], torch.full_like(envelope[..., 0, 0], dc_floor))
    return envelope.detach()


def spectral_manifold_project(
    delta: torch.Tensor,
    base_latent: torch.Tensor,
    freq_budget_ratio: float = 0.35,
    latent_limit: float = 6.0,
    *,
    precomputed_envelope: torch.Tensor | None = None,
) -> torch.Tensor:
    """将潜空间残差投影到 1/f 自然频谱流形球内，抑制棋盘格高频伪影并抵御扩散净化。"""
    ratio = float(freq_budget_ratio)
    h_lat, w_lat = int(delta.shape[-2]), int(delta.shape[-1])
    if ratio <= 0.0 or h_lat < 4 or w_lat < 4:
        return delta
    envelope = precomputed_envelope if precomputed_envelope is not None else build_spectral_envelope(base_latent, latent_limit)
    if envelope is None:
        return delta
    f_delta = torch.fft.rfft2(delta.float(), norm="ortho")
    cap = ratio * envelope.to(device=delta.device, dtype=torch.float32)
    mag = f_delta.abs()
    scale = torch.clamp(cap / mag.clamp_min(1e-12), max=1.0)
    projected = torch.fft.irfft2(f_delta * scale, s=(h_lat, w_lat), norm="ortho")
    return projected.to(dtype=delta.dtype)


class AdaptiveAnnealingController:
    """Stage 4 自适应退火调度器：余弦包络 + 约束回退降温 + 平台期降温与区间早停收敛。"""

    def __init__(self, config: AmbiLAGEConfig, init_utility: float) -> None:
        self.config = config
        self.effective_min_steps = min(int(config.anneal_min_steps), int(config.steps))
        self.warmup = min(int(config.anneal_warmup_steps), int(config.steps) // 5)
        self.adaptive_scale = 1.0
        self.best_utility = float(init_utility)
        self.stagnation_checks = 0
        self.cool_events = 0
        self.backtrack_events = 0
        self.reject_events = 0

    def lr_scale(self, step: int) -> float:
        if not self.config.adaptive_annealing:
            return max(0.05, 1.0 - 0.75 * (step / max(self.config.steps, 1)))
        min_r = float(self.config.anneal_min_lr_ratio)
        if self.warmup > 0 and step <= self.warmup:
            cosine = 0.35 + 0.65 * (step / self.warmup)
        else:
            span = max(self.config.steps - self.warmup, 1)
            progress = min(max((step - self.warmup) / span, 0.0), 1.0)
            cosine = min_r + 0.5 * (1.0 - min_r) * (1.0 + math.cos(math.pi * progress))
        return max(min_r, min(1.0, cosine * self.adaptive_scale))

    def on_step_end(self, accepted: bool, used_backtracks: int) -> None:
        if not self.config.adaptive_annealing:
            return
        min_r = float(self.config.anneal_min_lr_ratio)
        factor = float(self.config.anneal_backtrack_factor)
        if not accepted:
            self.reject_events += 1
            self.adaptive_scale = max(min_r, self.adaptive_scale * (factor * factor))
        elif used_backtracks > 0:
            self.backtrack_events += 1
            self.adaptive_scale = max(min_r, self.adaptive_scale * factor)

    def on_check(
        self,
        step: int,
        sel_utility: float,
        quality_ok: bool,
        *,
        current_lr_scale: float,
        best_quality_ok: bool,
    ) -> tuple[bool, str | None]:
        if not self.config.adaptive_annealing:
            return False, None
        min_r = float(self.config.anneal_min_lr_ratio)
        if quality_ok and sel_utility > self.best_utility + float(self.config.anneal_improve_tol):
            self.best_utility = float(sel_utility)
            self.stagnation_checks = 0
        else:
            self.stagnation_checks += 1
            if self.stagnation_checks % self.config.anneal_plateau_patience == 0:
                self.adaptive_scale = max(min_r, self.adaptive_scale * float(self.config.anneal_plateau_factor))
                self.cool_events += 1
        patience = int(self.config.anneal_early_stop_patience)
        if patience > 0 and best_quality_ok and step >= self.effective_min_steps and step < self.config.steps:
            if self.stagnation_checks >= patience or (
                current_lr_scale <= min_r * 1.25 and self.stagnation_checks >= max(2, patience // 2)
            ):
                return True, "adaptive_annealing_converged"
        return False, None

    def state_dict(self) -> dict[str, Any]:
        return {
            "adaptive_scale": float(self.adaptive_scale),
            "best_utility": float(self.best_utility),
            "stagnation_checks": int(self.stagnation_checks),
            "cool_events": int(self.cool_events),
            "backtrack_events": int(self.backtrack_events),
            "reject_events": int(self.reject_events),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.adaptive_scale = float(state["adaptive_scale"])
        self.best_utility = float(state["best_utility"])
        self.stagnation_checks = int(state["stagnation_checks"])
        self.cool_events = int(state["cool_events"])
        self.backtrack_events = int(state["backtrack_events"])
        self.reject_events = int(state["reject_events"])

    def summary(self, executed_steps: int, stop_reason: str, final_lr_scale: float, best_step: int) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.adaptive_annealing),
            "schedule": "cosine_with_backtrack_and_plateau_cooling" if self.config.adaptive_annealing else "legacy_linear",
            "min_steps": int(self.effective_min_steps),
            "max_steps": int(self.config.steps),
            "steps_executed": int(executed_steps),
            "stop_reason": stop_reason,
            "early_stopped": bool(stop_reason == "adaptive_annealing_converged"),
            "final_lr_scale": float(final_lr_scale),
            "final_adaptive_scale": float(self.adaptive_scale),
            "stagnation_checks": int(self.stagnation_checks),
            "cool_events": int(self.cool_events),
            "backtrack_events": int(self.backtrack_events),
            "reject_events": int(self.reject_events),
            "best_step": int(best_step),
        }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _stage4_signature(
    anchor_tensor: torch.Tensor,
    original_tensor: torch.Tensor,
    config: AmbiLAGEConfig,
    vae_identity: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any] | None = None,
) -> str:
    cfg_dict = config.to_dict()
    if cfg_dict.get("diffusion_eot_enabled") is None:
        cfg_dict.pop("diffusion_eot_enabled", None)
    if cfg_dict.get("multiscale_eot_enabled") is None:
        cfg_dict.pop("multiscale_eot_enabled", None)
    for runtime_key in ("log_every", "checkpoint_every", "resume_from_checkpoint", "save_intermediate_pngs", "device"):
        cfg_dict.pop(runtime_key, None)
    payload = {
        "anchor_sha256": _tensor_sha256(anchor_tensor),
        "original_sha256": _tensor_sha256(original_tensor),
        "config": cfg_dict,
        "vae": dict(vae_identity),
        "binding": dict(checkpoint_binding or {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optimizer_state_to_cpu(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(dict(state_dict))
    for param_state in copied.get("state", {}).values():
        if isinstance(param_state, dict):
            for key, val in list(param_state.items()):
                if torch.is_tensor(val):
                    param_state[key] = val.detach().cpu().clone()
    return copied


def _optimizer_state_to_device(state_dict: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    copied = copy.deepcopy(dict(state_dict))
    for param_state in copied.get("state", {}).values():
        if isinstance(param_state, dict):
            for key, val in list(param_state.items()):
                if torch.is_tensor(val):
                    # PyTorch Adam 要求标量 step 张量保留在 CPU，其余动量张量放回参数设备
                    target_device = torch.device("cpu") if key == "step" else device
                    param_state[key] = val.detach().clone().to(target_device)
    return copied


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(rng_state: Mapping[str, Any] | None, device: torch.device) -> None:
    if not rng_state:
        return
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])
    if rng_state.get("torch_cpu") is not None:
        torch.set_rng_state(rng_state["torch_cpu"])
    cuda_state = rng_state.get("torch_cuda")
    if cuda_state is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state, device)


def _save_atomic_checkpoint(ckpt_path: Path, payload: Mapping[str, Any]) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_name(f".ckpt-{uuid.uuid4().hex[:12]}.tmp")
    try:
        torch.save(dict(payload), tmp_path)
        os.replace(tmp_path, ckpt_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _load_compatible_checkpoint(ckpt_path: Path, expected_signature: str) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("signature") != expected_signature:
        raise ValueError("Stage 4 断点检查点与当前配置或输入不匹配，请清理旧目录或关闭断点续跑后重试")
    return ckpt


def refine_ambient_coop_latent(
    anchor_tensor: torch.Tensor,
    original_tensor: torch.Tensor,
    envelopes: dict[str, Any],
    objective: PublicFullKeyObjective,
    surrogate: Any,
    vae: AmbiLAGEVae,
    config: AmbiLAGEConfig,
    output_dir: Path | str | None = None,
    checkpoint_binding: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Stage 4: 执行原生环境光协同背景调整与主体微光影潜空间优化 (支持自适应退火与断点续跑)。"""
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
    w_fg_recolor_np = envelopes.get("w_fg_recolor", envelopes["w_fg_body"])
    w_fg_recolor_t = torch.as_tensor(w_fg_recolor_np, device=device, dtype=torch.float32)[None, None]
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
        w_fg_recolor=w_fg_recolor_t,
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
        spectral_envelope = (
            build_spectral_envelope(base_latent, config.latent_limit)
            if bool(getattr(config, "spectral_manifold_projection", False))
            else None
        )

    # 4. 初始化可学习潜差分与优化器
    delta = nn.Parameter(torch.zeros_like(base_latent))
    optimizer = torch.optim.Adam([
        {"params": [delta], "lr": config.latent_learning_rate},
        {
            "params": [
                ambient_field.ambient_global,
                ambient_field.ambient_grid,
                ambient_field.ambient_basis,
                ambient_field.fg_chroma_shift,
            ],
            "lr": config.ambient_learning_rate,
        },
    ])

    def snapshot_params() -> dict[str, torch.Tensor]:
        return {
            "delta": delta.detach().clone(),
            "ambient_global": ambient_field.ambient_global.detach().clone(),
            "ambient_grid": ambient_field.ambient_grid.detach().clone(),
            "ambient_basis": ambient_field.ambient_basis.detach().clone(),
            "fg_chroma_shift": ambient_field.fg_chroma_shift.detach().clone(),
        }

    def load_params(snap: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            delta.copy_(snap["delta"])
            ambient_field.ambient_global.copy_(snap["ambient_global"])
            ambient_field.ambient_grid.copy_(snap["ambient_grid"])
            if "ambient_basis" in snap:
                ambient_field.ambient_basis.copy_(snap["ambient_basis"])
            if "fg_chroma_shift" in snap:
                ambient_field.fg_chroma_shift.copy_(snap["fg_chroma_shift"])

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
        rgb_ssim = float(source_ssim(image, original_tensor, luma_only=False))
        luma_ssim = float(source_ssim(image, original_tensor, luma_only=True))
        ss = min(rgb_ssim, luma_ssim)
        luma_mae = float(source_luma_mae_levels(image, original_tensor))
        luma_diff = (extract_oklab_luma(image) - extract_oklab_luma(original_tensor)).abs() * 255.0
        fg_luma_mae = float(luma_diff[:, :, fg_bool].mean()) if fg_bool.any() else 0.0
        bg_luma_mae = float(luma_diff[:, :, bg_bool].mean()) if bg_bool.any() else 0.0
        fg_chroma_tv = float(compute_fg_chroma_tv_levels(image, original_tensor, fg_mask_t))
        diff_orig = (image - original_tensor).abs() * 255.0
        diff_anchor = (image - anchor_tensor).abs() * 255.0
        rgb_mae = float(diff_orig.mean())
        fg_rgb_mae = float(diff_orig[:, :, fg_bool].mean()) if fg_bool.any() else 0.0
        bg_rgb_mae = float(diff_orig[:, :, bg_bool].mean()) if bg_bool.any() else 0.0
        anchor_mae = float(diff_anchor.mean())
        effective_mae = max(rgb_mae, luma_mae)
        fg_mae_ceiling = max(
            float(getattr(config, "fg_source_mae_ceiling", 10.0)),
            float(config.source_mae_ceiling) if config.source_mae_ceiling > 30.0 else 0.0,
        )
        chroma_tv_ceiling = float(getattr(config, "fg_chroma_tv_ceiling", 2.5))
        return {
            "source_ssim": ss,
            "rgb_source_ssim": rgb_ssim,
            "luma_source_ssim": luma_ssim,
            "source_mae_levels": effective_mae,
            "luma_source_mae_levels": luma_mae,
            "fg_source_mae_levels": fg_luma_mae,
            "bg_source_mae_levels": bg_luma_mae,
            "rgb_source_mae_levels": rgb_mae,
            "fg_rgb_source_mae_levels": fg_rgb_mae,
            "bg_rgb_source_mae_levels": bg_rgb_mae,
            "fg_chroma_tv_levels": fg_chroma_tv,
            "anchor_mae_levels": anchor_mae,
            "quality_ok": bool(
                ss >= config.source_ssim_floor
                and effective_mae <= config.source_mae_ceiling
                and fg_rgb_mae <= fg_mae_ceiling
                and fg_chroma_tv <= chroma_tv_ceiling
            ),
        }

    tick = time.perf_counter()
    ckpt_path = (Path(output_dir) / STAGE4_CHECKPOINT_FILENAME) if output_dir is not None else None
    expected_sig = (
        _stage4_signature(anchor_tensor, original_tensor, config, vae.identity, checkpoint_binding)
        if ckpt_path is not None
        else ""
    )

    if ckpt_path is not None and config.resume_from_checkpoint and ckpt_path.is_file():
        ckpt = _load_compatible_checkpoint(ckpt_path, expected_sig)
        saved_step = int(ckpt["step"])
        if saved_step < 0 or saved_step > config.steps or ckpt["delta"].shape != base_latent.shape:
            raise ValueError("Stage 4 断点检查点状态越界或张量尺寸不匹配")
        with torch.no_grad():
            delta.copy_(ckpt["delta"].to(device=device, dtype=delta.dtype))
            ambient_field.ambient_global.copy_(
                ckpt["ambient_global"].to(device=device, dtype=ambient_field.ambient_global.dtype)
            )
            ambient_field.ambient_grid.copy_(
                ckpt["ambient_grid"].to(device=device, dtype=ambient_field.ambient_grid.dtype)
            )
            if "ambient_basis" in ckpt:
                ambient_field.ambient_basis.copy_(
                    ckpt["ambient_basis"].to(device=device, dtype=ambient_field.ambient_basis.dtype)
                )
            if "fg_chroma_shift" in ckpt:
                ambient_field.fg_chroma_shift.copy_(
                    ckpt["fg_chroma_shift"].to(device=device, dtype=ambient_field.fg_chroma_shift.dtype)
                )
        optimizer.load_state_dict(_optimizer_state_to_device(ckpt["optimizer_state_dict"], device))
        best_snap = {k: v.detach().clone().to(device) for k, v in ckpt["best_snap"].items()}
        best_row = copy.deepcopy(ckpt["best_row"])
        history = copy.deepcopy(ckpt["history"])
        accepted = int(ckpt["accepted"])
        rejected = int(ckpt["rejected"])
        annealer = AdaptiveAnnealingController(config, best_row["selection_utility"])
        annealer.load_state_dict(ckpt["annealer_state"])
        baseline_vals = ckpt.get("objective_baseline_values")
        if baseline_vals is not None and hasattr(objective, "_baseline_values"):
            objective._baseline_values = baseline_vals.detach().clone().to(device)
        baseline_joint = ckpt.get("objective_baseline_joint_values")
        if baseline_joint is not None and hasattr(objective, "_baseline_joint_values"):
            objective._baseline_joint_values = baseline_joint.detach().clone().to(device)
        baseline_text = ckpt.get("objective_baseline_text_values")
        if baseline_text is not None and hasattr(objective, "_baseline_text_values"):
            objective._baseline_text_values = baseline_text.detach().clone().to(device)
        _restore_rng_state(ckpt.get("rng_state"), device)
        start_step = saved_step + 1
        prior_elapsed = float(ckpt.get("accumulated_elapsed_seconds", 0.0))
        resumed_from_step: int | None = saved_step
        resume_count = int(ckpt.get("resume_count", 0)) + 1
        checkpoints_saved = int(ckpt.get("checkpoints_saved", 0))
        executed_steps = int(ckpt.get("executed_steps", saved_step))
        stop_reason = str(ckpt.get("stop_reason", "requested_steps_completed"))
        last_lr_scale = float(ckpt.get("last_lr_scale", annealer.lr_scale(max(1, saved_step))))
        tangent_proj_calls = int(ckpt.get("tangent_projection_calls", 0))
        print(
            f"AmbiLAGE 断点续跑恢复 [step={saved_step}/{config.steps}] "
            f"(Accepted={accepted}, Rejected={rejected}, adaptive_scale={annealer.adaptive_scale:.4f})",
            flush=True,
        )
    else:
        history = []
        accepted = 0
        rejected = 0
        tangent_proj_calls = 0
        best_snap = snapshot_params()
        with torch.no_grad():
            initial_img = compose()
        best_row = evaluate_quality(initial_img)
        if not best_row["quality_ok"]:
            raise RuntimeError("自然锚点未通过结构/误差门禁，不能标记为完整成图")
        if hasattr(objective, "evaluate"):
            init_utility, init_scores = objective.evaluate(initial_img)
        else:
            init_features = surrogate.encode_image(initial_img).global_embedding
            init_utility, init_scores = objective.compute_utility(init_features, config)
        init_penalty, _ = compute_spatial_variance_penalty(initial_img, anchor_tensor, None, fg_mask_t, bg_mask_t, config)
        init_sel_utility = float(init_utility.detach()) - float(init_penalty.detach())
        annealer = AdaptiveAnnealingController(config, init_sel_utility)
        best_row.update({
            "step": 0,
            "training_utility": float(init_utility.detach()),
            "selection_utility": init_sel_utility,
            "joint_scores": init_scores.detach().cpu().tolist(),
            "lr_scale": float(annealer.lr_scale(1)),
            "adaptive_scale": float(annealer.adaptive_scale),
            "stagnation_checks": 0,
        })
        history.append(best_row)
        start_step = 1
        prior_elapsed = 0.0
        resumed_from_step = None
        resume_count = 0
        checkpoints_saved = 0
        executed_steps = 0
        stop_reason = "requested_steps_completed"
        last_lr_scale = float(annealer.lr_scale(1))

    interrupt_requested: dict[str, str | None] = {"name": None}
    previous_handlers: dict[int, Any] = {}
    if ckpt_path is not None and threading.current_thread() is threading.main_thread():
        def _on_signal(signum: int, _frame: Any) -> None:
            interrupt_requested["name"] = signal.Signals(signum).name

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass

    # 5. 优化循环 (原生 4 阶段第 4 阶段细化，支持自适应退火与断点续跑)
    try:
        for step in range(start_step, config.steps + 1):
            old_snap = snapshot_params()
            old_opt_state = copy.deepcopy(optimizer.state_dict())
            optimizer.zero_grad(set_to_none=True)

            # 5.1/5.2 前向合成一次，并按需在边界邻域内触发法向量反向传播（节省 L4 24GB 双重 FLUX VAE 反传开销）
            decoded_img, raw_img = compose_with_raw()
            normals = []
            slacks = []

            with torch.no_grad():
                img_det = decoded_img.detach()
                rgb_ssim_val = float(source_ssim(img_det, original_tensor, luma_only=False))
                luma_ssim_val = float(source_ssim(img_det, original_tensor, luma_only=True))
                min_ssim_val = min(rgb_ssim_val, luma_ssim_val)
                ssim_slack = min_ssim_val - float(config.source_ssim_floor)

                diff_abs = (img_det - original_tensor).abs()
                rgb_mae_norm = float(diff_abs.mean())
                luma_mae_norm = float(source_luma_mae_levels(img_det, original_tensor)) / 255.0
                eff_mae_norm = max(rgb_mae_norm, luma_mae_norm)
                mae_ceiling_norm = float(config.source_mae_ceiling) / 255.0
                mae_slack = mae_ceiling_norm - eff_mae_norm

                fg_mae_ceiling = max(
                    float(getattr(config, "fg_source_mae_ceiling", 10.0)),
                    float(config.source_mae_ceiling) if config.source_mae_ceiling > 30.0 else 0.0,
                )
                fg_mae_ceiling_norm = fg_mae_ceiling / 255.0
                fg_mae_norm = float(diff_abs[:, :, fg_bool].mean()) if fg_bool.any() else 0.0
                fg_mae_slack = fg_mae_ceiling_norm - fg_mae_norm

            if bool(getattr(config, "conditional_tangent_projection", False)):
                ssim_margin = float(getattr(config, "tangent_ssim_slack_margin", 0.03))
                mae_ratio = float(getattr(config, "tangent_mae_slack_ratio", 0.20))
                need_ssim = ssim_slack < ssim_margin
                need_mae = mae_slack < mae_ratio * mae_ceiling_norm
                need_fg_mae = bool(fg_bool.any()) and (fg_mae_slack < mae_ratio * fg_mae_ceiling_norm)
            else:
                need_ssim = True
                need_mae = True
                need_fg_mae = False

            if need_ssim or need_mae or need_fg_mae:
                image_eval = decoded_img.detach().requires_grad_(True)
                if need_ssim:
                    use_luma_ssim = luma_ssim_val < rgb_ssim_val
                    structure = source_ssim(image_eval, original_tensor, luma_only=use_luma_ssim)
                    n_ssim, = torch.autograd.grad(structure, image_eval)
                    normals.append(n_ssim.detach())
                    slacks.append(ssim_slack)
                    del structure
                if need_mae:
                    if luma_mae_norm > rgb_mae_norm:
                        mae_metric = (extract_oklab_luma(image_eval) - extract_oklab_luma(original_tensor)).abs().mean()
                    else:
                        mae_metric = (image_eval - original_tensor).abs().mean()
                    n_mae, = torch.autograd.grad(-mae_metric, image_eval)
                    normals.append(n_mae.detach())
                    slacks.append(mae_slack)
                    del mae_metric
                if need_fg_mae:
                    fg_metric = (image_eval - original_tensor)[:, :, fg_bool].abs().mean()
                    n_fg, = torch.autograd.grad(-fg_metric, image_eval)
                    normals.append(n_fg.detach())
                    slacks.append(fg_mae_slack)
                    del fg_metric
                del image_eval

            latent_normals = []
            for n_img in normals:
                d_latent, = torch.autograd.grad(decoded_img, delta, grad_outputs=n_img, retain_graph=True)
                latent_normals.append(d_latent.detach())
                tangent_proj_calls += 1

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
            del decoded_img, raw_img

            for p in (
                delta,
                ambient_field.ambient_global,
                ambient_field.ambient_grid,
                ambient_field.ambient_basis,
                ambient_field.fg_chroma_shift,
            ):
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)

            # 自适应退火学习率更新
            lr_scale = annealer.lr_scale(step)
            last_lr_scale = lr_scale
            optimizer.param_groups[0]["lr"] = config.latent_learning_rate * lr_scale
            optimizer.param_groups[1]["lr"] = config.ambient_learning_rate * lr_scale
            optimizer.step()

            # 5.3 切空间投影与回退守门 (Tangent Space Projection & Backtracking Gate)
            with torch.no_grad():
                delta.clamp_(-config.latent_limit, config.latent_limit)
                if config.latent_smoothing_mix > 0.0:
                    smooth = F.avg_pool2d(F.pad(delta, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
                    delta.lerp_(smooth, config.latent_smoothing_mix)
                if bool(getattr(config, "spectral_manifold_projection", False)):
                    prog_spec = (step - 1) / max(1, config.steps - 1)
                    b_start = float(getattr(config, "spectral_budget_start", 0.22))
                    b_end = float(getattr(config, "spectral_budget_end", 0.45))
                    freq_ratio = b_start + (b_end - b_start) * min(max(prog_spec, 0.0), 1.0)
                    delta.copy_(
                        spectral_manifold_project(
                            delta,
                            base_latent,
                            freq_budget_ratio=freq_ratio,
                            latent_limit=config.latent_limit,
                            precomputed_envelope=spectral_envelope,
                        )
                    )

                step_vector = delta - old_snap["delta"]
                projected = tangent_projection(step_vector, latent_normals, slacks)
                proposed_delta = (old_snap["delta"] + projected).clamp(-config.latent_limit, config.latent_limit)
                proposed_global = ambient_field.ambient_global.clamp(-config.ambient_limit, config.ambient_limit).clone()
                proposed_grid = ambient_field.ambient_grid.clamp(-1.0, 1.0).clone()
                proposed_basis = ambient_field.ambient_basis.clamp(-1.0, 1.0).clone()
                fg_chroma_lim = float(getattr(config, "fg_chroma_limit", 0.16))
                proposed_fg_chroma = ambient_field.fg_chroma_shift.clamp(-fg_chroma_lim, fg_chroma_lim).clone()

                step_accepted = False
                used_backtracks = 0
                accepted_img: torch.Tensor | None = None
                accepted_q: dict[str, Any] | None = None
                for backtrack in range(config.guard_backtracks + 1):
                    scale = 0.5 ** backtrack
                    delta.copy_(old_snap["delta"] + scale * (proposed_delta - old_snap["delta"]))
                    ambient_field.ambient_global.copy_(
                        old_snap["ambient_global"] + scale * (proposed_global - old_snap["ambient_global"])
                    )
                    ambient_field.ambient_grid.copy_(
                        old_snap["ambient_grid"] + scale * (proposed_grid - old_snap["ambient_grid"])
                    )
                    ambient_field.ambient_basis.copy_(
                        old_snap["ambient_basis"] + scale * (proposed_basis - old_snap["ambient_basis"])
                    )
                    ambient_field.fg_chroma_shift.copy_(
                        old_snap["fg_chroma_shift"] + scale * (proposed_fg_chroma - old_snap["fg_chroma_shift"])
                    )
                    cand_img = compose()
                    q = evaluate_quality(cand_img)
                    if q["quality_ok"]:
                        step_accepted = True
                        used_backtracks = backtrack
                        accepted_img = cand_img
                        accepted_q = q
                        break

                if step_accepted:
                    accepted += 1
                else:
                    load_params(old_snap)
                    optimizer.load_state_dict(old_opt_state)
                    rejected += 1

            annealer.on_step_end(step_accepted, used_backtracks)

            # 5.4 进度记录、自适应退火检查与收敛决策
            should_stop = False
            candidate_stop_reason: str | None = None
            if step % config.check_every == 0 or step == config.steps:
                with torch.no_grad():
                    if step_accepted and accepted_img is not None and accepted_q is not None:
                        cur_img = accepted_img
                        q_row = dict(accepted_q)
                    else:
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

                    if q_row["quality_ok"] and (sel_utility > best_row["selection_utility"] or not best_row.get("quality_ok")):
                        best_row = q_row
                        best_snap = snapshot_params()

                    should_stop, candidate_stop_reason = annealer.on_check(
                        step,
                        sel_utility,
                        q_row["quality_ok"],
                        current_lr_scale=lr_scale,
                        best_quality_ok=bool(best_row.get("quality_ok")),
                    )
                    q_row.update({
                        "lr_scale": float(lr_scale),
                        "adaptive_scale": float(annealer.adaptive_scale),
                        "stagnation_checks": int(annealer.stagnation_checks),
                    })
                    history.append(q_row)

                    if step % config.log_every == 0 or step == config.steps or should_stop:
                        print(
                            f"AmbiLAGE 细化 [{step}/{config.steps}] "
                            f"Utility={u_val:.4f} SSIM={q_row['source_ssim']:.4f} "
                            f"MAE={q_row['source_mae_levels']:.2f} LR={lr_scale:.3f} "
                            f"(Accepted={accepted}, Rejected={rejected})",
                            flush=True,
                        )
                    if should_stop:
                        print(
                            f"AmbiLAGE 自适应退火收敛早停 [{step}/{config.steps}] "
                            f"(min_steps={annealer.effective_min_steps}, lr_scale={lr_scale:.4f}, "
                            f"best_step={best_row['step']}, best_sel_utility={best_row['selection_utility']:.4f})",
                            flush=True,
                        )

            executed_steps = step
            if should_stop and candidate_stop_reason:
                stop_reason = candidate_stop_reason

            if ckpt_path is not None and (
                step % config.checkpoint_every == 0
                or step == config.steps
                or should_stop
                or interrupt_requested["name"] is not None
            ):
                baseline_tensor = getattr(objective, "_baseline_values", None)
                baseline_joint_tensor = getattr(objective, "_baseline_joint_values", None)
                baseline_text_tensor = getattr(objective, "_baseline_text_values", None)
                checkpoints_saved += 1
                _save_atomic_checkpoint(ckpt_path, {
                    "signature": expected_sig,
                    "step": int(step),
                    "executed_steps": int(executed_steps),
                    "stop_reason": stop_reason,
                    "last_lr_scale": float(last_lr_scale),
                    "delta": delta.detach().cpu().clone(),
                    "ambient_global": ambient_field.ambient_global.detach().cpu().clone(),
                    "ambient_grid": ambient_field.ambient_grid.detach().cpu().clone(),
                    "ambient_basis": ambient_field.ambient_basis.detach().cpu().clone(),
                    "fg_chroma_shift": ambient_field.fg_chroma_shift.detach().cpu().clone(),
                    "optimizer_state_dict": _optimizer_state_to_cpu(optimizer.state_dict()),
                    "best_snap": {k: v.detach().cpu().clone() for k, v in best_snap.items()},
                    "best_row": copy.deepcopy(best_row),
                    "history": copy.deepcopy(history),
                    "accepted": int(accepted),
                    "rejected": int(rejected),
                    "annealer_state": annealer.state_dict(),
                    "objective_baseline_values": (
                        baseline_tensor.detach().cpu().clone() if torch.is_tensor(baseline_tensor) else None
                    ),
                    "objective_baseline_joint_values": (
                        baseline_joint_tensor.detach().cpu().clone()
                        if torch.is_tensor(baseline_joint_tensor)
                        else None
                    ),
                    "objective_baseline_text_values": (
                        baseline_text_tensor.detach().cpu().clone()
                        if torch.is_tensor(baseline_text_tensor)
                        else None
                    ),
                    "rng_state": _capture_rng_state(device),
                    "accumulated_elapsed_seconds": float(prior_elapsed + (time.perf_counter() - tick)),
                    "checkpoints_saved": int(checkpoints_saved),
                    "resume_count": int(resume_count),
                    "tangent_projection_calls": int(tangent_proj_calls),
                })

            if interrupt_requested["name"] is not None:
                raise InterruptedError(f"收到中断信号 ({interrupt_requested['name']})，已保存完整断点至第 {step} 步")
            if should_stop:
                break
    finally:
        for sig, prev in previous_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass

    # 6. 加载最佳快照并生成最终成图
    load_params(best_snap)
    with torch.no_grad():
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

    fg_rgb_max = int(diff_anchor[fg_np].max()) if fg_np.any() else 0
    bg_max = int(diff_anchor[bg_np].max()) if bg_np.any() else 0
    fg_chroma_budget = int(getattr(config, "fg_chroma_levels", 0))
    with torch.no_grad():
        final_u8_t = torch.from_numpy((final_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1).copy())[None]
        anchor_u8_t = torch.from_numpy((anchor_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1).copy())[None]
        final_lab = srgb_to_oklab(final_u8_t)
        anchor_lab = srgb_to_oklab(anchor_u8_t)
        fg_luma_diff_np = ((final_lab[:, 0] - anchor_lab[:, 0]).abs() * 255.0)[0].cpu().numpy()
        fg_chroma_diff_np = (torch.maximum(
            (final_lab[:, 1] - anchor_lab[:, 1]).abs(),
            (final_lab[:, 2] - anchor_lab[:, 2]).abs(),
        ) * 255.0)[0].cpu().numpy()
    fg_luma_max = float(fg_luma_diff_np[fg_np].max()) if fg_np.any() else 0.0
    fg_chroma_max = float(fg_chroma_diff_np[fg_np].max()) if fg_np.any() else 0.0

    if fg_chroma_budget > 0:
        if fg_luma_max > float(config.latent_levels) + 1.0:
            raise AssertionError(
                f"商品主体 OKLab 亮度 L 超出锁亮度预算: {fg_luma_max:.2f} > {config.latent_levels + 1}"
            )
        if fg_chroma_max > float(fg_chroma_budget) + 2.0:
            raise AssertionError(
                f"商品主体 OKLab 色度 (a, b) 超出预算: {fg_chroma_max:.2f} > {fg_chroma_budget + 2}"
            )
    elif fg_rgb_max > config.latent_levels:
        raise AssertionError(f"商品主体微光影超出预算: {fg_rgb_max} > {config.latent_levels}")
    if bg_max > config.background_levels:
        raise AssertionError(f"背景光场超出预算: {bg_max} > {config.background_levels}")

    metadata: dict[str, Any] = {
        "status": "completed",
        "stage": "latent_generation_refinement",
        "steps_requested": config.steps,
        "steps_completed": executed_steps,
        "stop_reason": stop_reason,
        "best_step": best_row["step"],
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "final_quality": best_row,
        "fg_latent_max_levels": int(round(fg_luma_max)) if fg_chroma_budget > 0 else fg_rgb_max,
        "fg_chroma_max_levels": float(fg_chroma_max),
        "fg_rgb_max_levels": fg_rgb_max,
        "bg_latent_max_levels": bg_max,
        "roi_max_levels": 0,
        "elapsed_seconds": float(prior_elapsed + (time.perf_counter() - tick)),
        "zero_pixel_optimization": True,
        "vae": vae.identity,
        "spectral_manifold_projection": bool(getattr(config, "spectral_manifold_projection", False)),
        "conditional_tangent_projection": bool(getattr(config, "conditional_tangent_projection", False)),
        "tangent_projection_calls": int(tangent_proj_calls),
        "adaptive_annealing": annealer.summary(executed_steps, stop_reason, last_lr_scale, best_row["step"]),
        "checkpoint_resume": {
            "checkpoint_every": int(config.checkpoint_every),
            "checkpoints_saved": int(checkpoints_saved),
            "resumed": resumed_from_step is not None,
            "resumed_from_step": resumed_from_step,
            "resume_count": int(resume_count),
        },
        "history": history,
        "auxiliary_levels": 0,
    }

    return final_tensor, metadata
