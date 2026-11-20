#!/usr/bin/env python3
"""汇总 Stage 3 稳妥梯度保护方案三组试点结果，并计算自动决策门结论。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.perceptual import load_perceptual

EXP_DIR = Path('/content/AmbiLAGE/output/exp_20261120_grad_protect')
BASE_GEN = Path('/content/AmbiLAGE/output/ambilage_single_clip_l14_9groups_20261118')
PILOT_GEN = EXP_DIR / 'gen'
BASE_ROOT = Path('/content/AmbiLAGE/resources/evaluation_base/groups')
SELECTED_GROUPS = ('02_chair', '05_hair_extension', '09_toothbrush')
ENCODERS = (
    ('01_clip_l14', 'CLIP-L/14 (白盒)'),
    ('02_openclip_bigG14', 'OpenCLIP-bigG-14 (黑盒)'),
    ('03_siglip_so400m', 'SigLIP-SO400M (黑盒)'),
)
STAGE1_SUITES = ('training_oracle_v4', 'overlap_30_v4', 'zero_overlap_v4')


def pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def load_nat(gen_root: Path, lpips_fn) -> list[dict]:
    rows = []
    for g in SELECTED_GROUPS:
        fi = json.loads((gen_root / g / 'ambilage' / 'FROZEN_IMAGE.json').read_text(encoding='utf-8'))
        fn = fi['metadata']['final_naturalness']
        m = fn['metrics']
        src_img = Image.open(BASE_ROOT / g / 'source.jpg').convert('RGB')
        mod_img = Image.open(fi['image']).convert('RGB')
        a_src = np.asarray(src_img, dtype=np.float32)
        a_mod = np.asarray(mod_img, dtype=np.float32)
        mse = float(np.mean((a_src - a_mod) ** 2))
        psnr = 99.0 if mse < 1e-10 else 10.0 * math.log10((255.0 ** 2) / mse)
        t_src = torch.from_numpy(a_src / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        t_mod = torch.from_numpy(a_mod / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            lp = float(lpips_fn(t_mod, t_src).item())
        rows.append(dict(
            group=g,
            passed=bool(fn.get('passed')),
            ssim=float(m['source_ssim']),
            lpips=lp,
            psnr=psnr,
            fg_mae=float(m['fg_rgb_source_mae_levels']),
            roi_max_abs_diff=float(m['roi_max_abs_diff']),
            halo_error=float(m['halo_error']),
        ))
    return rows


def main() -> None:
    vi_cfg = VisualInceptionConfig()
    lpips_fn = load_perceptual(vi_cfg, torch.device('cuda'))

    lines = [
        '# AmbiLAGE 梯度保护稳妥方案三组试点报告（Stage 3）',
        '',
        '- 试点分组：`02_chair`、`05_hair_extension`、`09_toothbrush`（全部通过 `--replacement-groups selected` 同库同步替换）。',
        '- 配置对比：基线 `ambilage.clip_l14.example.json`（`joint_weight=0.15`, `joint_grad_protect=False`）vs 试点（`joint_weight=0.45`, `joint_key_aux_mix=1.0`, `psi_joint_end_scale=1.0`, `joint_grad_protect=True`, `joint_grad_cap=0.65`, `text_weight=1.50`, 绑定 `training_bundle.json`）。',
        '',
        '## 一、Env4（`joint 0.5/0.5`）三编码器对比',
        '',
        '| 方案 | 编码器 | 原图 Top-3 / Top-5 | 修改后 Top-1 / Top-3 / Top-5 | 平均名次(原->改) |',
        '|---|---|---:|---:|---:|',
    ]

    env4_data: dict[tuple[str, str], dict] = {}
    for slug, elabel in ENCODERS:
        for var, vlabel in (('baseline', '基线 (jw=0.15)'), ('gp_pilot3', '稳妥保护 (jw=0.45+PCGrad)')):
            p = EXP_DIR / 'eval_env4' / var / slug / 'env4_summary.json'
            if p.is_file():
                s = json.loads(p.read_text(encoding='utf-8'))
                env4_data[(var, slug)] = s
                lines.append(
                    f"| {vlabel} | {elabel} | {pct(s['clean_top3'])} / {pct(s['clean_top5'])} | "
                    f"{pct(s['modified_top1'])} / {pct(s['modified_top3'])} / {pct(s['modified_top5'])} | "
                    f"{s['clean_mean_rank']:.2f} -> {s['modified_mean_rank']:.2f} |"
                )

    lines += [
        '',
        '## 二、Stage 1 纯图键检索对比（`key_mode=image`, `defense=none`）',
        '',
        '| 方案 | 评测集 | CLIP-L/14 T3 / T5 / 均名 | OpenCLIP-bigG-14 T3 / T5 / 均名 | SigLIP-SO400M T3 / T5 / 均名 | Mean BB T3 / T5 |',
        '|---|---|---:|---:|---:|---:|',
    ]

    st1_data: dict[tuple[str, str], dict] = {}
    for suite in STAGE1_SUITES:
        for var, vlabel in (('baseline', '基线 (jw=0.15)'), ('gp_pilot3', '稳妥保护 (jw=0.45+PCGrad)')):
            p = EXP_DIR / 'eval_stage1' / f'{var}_{suite}' / 'transfer_panel_summary.json'
            if not p.is_file():
                continue
            d = json.loads(p.read_text(encoding='utf-8'))
            st1_data[(var, suite)] = d
            cells = []
            bb_t3, bb_t5 = [], []
            for slug, _ in ENCODERS:
                ov = d['encoders'][slug]['overall']
                t3, t5 = ov['top3']['modified_rate'], ov['top5']['modified_rate']
                if slug != '01_clip_l14':
                    bb_t3.append(t3)
                    bb_t5.append(t5)
                cells.append(f"{pct(t3)} / {pct(t5)} / {ov['mean_rank_modified']:.2f}")
            mean_bb = f"{pct(sum(bb_t3)/len(bb_t3))} / {pct(sum(bb_t5)/len(bb_t5))}"
            lines.append(f'| {vlabel} | `{suite}` | ' + ' | '.join(cells) + f' | {mean_bb} |')

    nat_base = load_nat(BASE_GEN, lpips_fn)
    nat_pilot = load_nat(PILOT_GEN, lpips_fn)
    lines += [
        '',
        '## 三、自然度与前缀预算门禁检验',
        '',
        '| 方案 | 组别 | Gate 通过 | SSIM | LPIPS | PSNR (dB) | FG RGB MAE (levels) | ROI Max Diff | Halo Error |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for vlabel, rows in (('基线 (jw=0.15)', nat_base), ('稳妥保护 (jw=0.45+PCGrad)', nat_pilot)):
        for r in rows:
            lines.append(
                f"| {vlabel} | `{r['group']}` | {'PASS' if r['passed'] else 'FAIL'} | "
                f"{r['ssim']:.4f} | {r['lpips']:.4f} | {r['psnr']:.2f} | "
                f"{r['fg_mae']:.2f} | {r['roi_max_abs_diff']:.4f} | {r['halo_error']:.2f} |"
            )

    # Decision gate checks
    e4_b = env4_data[('baseline', '01_clip_l14')]
    e4_p = env4_data[('gp_pilot3', '01_clip_l14')]
    env4_wb_t3_delta = (e4_p['modified_top3'] - e4_b['modified_top3']) * 100.0
    env4_wb_t5_delta = (e4_p['modified_top5'] - e4_b['modified_top5']) * 100.0

    stage1_deltas = {}
    for suite in STAGE1_SUITES:
        b_enc = st1_data[('baseline', suite)]['encoders']
        p_enc = st1_data[('gp_pilot3', suite)]['encoders']
        wb_t5_d = (p_enc['01_clip_l14']['overall']['top5']['modified_rate']
                   - b_enc['01_clip_l14']['overall']['top5']['modified_rate']) * 100.0
        bb_t5_b = 0.5 * (b_enc['02_openclip_bigG14']['overall']['top5']['modified_rate']
                         + b_enc['03_siglip_so400m']['overall']['top5']['modified_rate'])
        bb_t5_p = 0.5 * (p_enc['02_openclip_bigG14']['overall']['top5']['modified_rate']
                         + p_enc['03_siglip_so400m']['overall']['top5']['modified_rate'])
        bb_t5_d = (bb_t5_p - bb_t5_b) * 100.0
        stage1_deltas[suite] = dict(wb_t5_delta_pp=wb_t5_d, bb_t5_delta_pp=bb_t5_d)

    nat_all_pass = all(r['passed'] and r['roi_max_abs_diff'] == 0.0 and r['ssim'] >= 0.85 and r['fg_mae'] <= 10.0 for r in nat_pilot)
    stage1_ok = all(
        d['wb_t5_delta_pp'] >= -2.0 and d['bb_t5_delta_pp'] >= -2.0
        for d in stage1_deltas.values()
    )
    gate_pass = bool(env4_wb_t3_delta >= 8.0 and env4_wb_t5_delta >= 0.0 and stage1_ok and nat_all_pass)

    lines += [
        '',
        '## 四、自动决策门评估结果',
        '',
        f"- Env4 CLIP-L/14 Top-3 增量：`{env4_wb_t3_delta:+.2f} pp`（门槛 `>= +8.00 pp`）；Top-5 增量：`{env4_wb_t5_delta:+.2f} pp`（门槛 `>= 0.00 pp`）",
        *[
            f"- Stage 1 `{suite}` 变化：WB Top-5 `{stage1_deltas[suite]['wb_t5_delta_pp']:+.2f} pp`，Mean BB Top-5 `{stage1_deltas[suite]['bb_t5_delta_pp']:+.2f} pp`（门槛 `>= -2.00 pp`）"
            for suite in STAGE1_SUITES
        ],
        f"- 自然度与印刷区红线全部通过：`{nat_all_pass}`",
        f"- **最终决策**：**{'PROMOTE（更新 ambilage.clip_l14.example.json）' if gate_pass else 'HOLD（不修改 ambilage.clip_l14.example.json，保持 joint_weight=0.15）'}**",
    ]

    (EXP_DIR / 'GRAD_PROTECT_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    gate_payload = dict(
        gate_pass=gate_pass,
        env4_wb_t3_delta_pp=env4_wb_t3_delta,
        env4_wb_t5_delta_pp=env4_wb_t5_delta,
        stage1_deltas=stage1_deltas,
        nat_all_pass=nat_all_pass,
        naturalness=dict(baseline=nat_base, gp_pilot3=nat_pilot),
    )
    (EXP_DIR / 'GRAD_PROTECT_GATE.json').write_text(
        json.dumps(gate_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(gate_payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
