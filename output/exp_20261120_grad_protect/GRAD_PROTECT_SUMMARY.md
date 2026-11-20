# AmbiLAGE 梯度保护稳妥方案三组试点报告（Stage 3）

- 试点分组：`02_chair`、`05_hair_extension`、`09_toothbrush`（全部通过 `--replacement-groups selected` 同库同步替换）。
- 配置对比：基线 `ambilage.clip_l14.example.json`（`joint_weight=0.15`, `joint_grad_protect=False`）vs 试点（`joint_weight=0.45`, `joint_key_aux_mix=1.0`, `psi_joint_end_scale=1.0`, `joint_grad_protect=True`, `joint_grad_cap=0.65`, `text_weight=1.50`, 绑定 `training_bundle.json`）。

## 一、Env4（`joint 0.5/0.5`）三编码器对比

| 方案 | 编码器 | 原图 Top-3 / Top-5 | 修改后 Top-1 / Top-3 / Top-5 | 平均名次(原->改) |
|---|---|---:|---:|---:|
| 基线 (jw=0.15) | CLIP-L/14 (白盒) | 75.0% / 93.3% | 0.0% / 25.0% / 85.0% | 2.88 -> 4.18 |
| 稳妥保护 (jw=0.45+PCGrad) | CLIP-L/14 (白盒) | 75.0% / 93.3% | 1.7% / 45.0% / 85.0% | 2.88 -> 3.78 |
| 基线 (jw=0.15) | OpenCLIP-bigG-14 (黑盒) | 33.3% / 90.0% | 33.3% / 33.3% / 91.7% | 3.45 -> 3.35 |
| 稳妥保护 (jw=0.45+PCGrad) | OpenCLIP-bigG-14 (黑盒) | 33.3% / 90.0% | 33.3% / 33.3% / 95.0% | 3.45 -> 3.23 |
| 基线 (jw=0.15) | SigLIP-SO400M (黑盒) | 78.3% / 100.0% | 41.7% / 66.7% / 100.0% | 2.18 -> 2.27 |
| 稳妥保护 (jw=0.45+PCGrad) | SigLIP-SO400M (黑盒) | 78.3% / 100.0% | 41.7% / 66.7% / 100.0% | 2.18 -> 2.27 |

## 二、Stage 1 纯图键检索对比（`key_mode=image`, `defense=none`）

| 方案 | 评测集 | CLIP-L/14 T3 / T5 / 均名 | OpenCLIP-bigG-14 T3 / T5 / 均名 | SigLIP-SO400M T3 / T5 / 均名 | Mean BB T3 / T5 |
|---|---|---:|---:|---:|---:|
| 基线 (jw=0.15) | `training_oracle_v4` | 100.0% / 100.0% / 1.00 | 86.7% / 100.0% / 2.13 | 95.0% / 98.3% / 1.58 | 90.8% / 99.2% |
| 稳妥保护 (jw=0.45+PCGrad) | `training_oracle_v4` | 100.0% / 100.0% / 1.00 | 81.7% / 100.0% / 2.13 | 90.0% / 98.3% / 1.70 | 85.8% / 99.2% |
| 基线 (jw=0.15) | `overlap_30_v4` | 98.3% / 100.0% / 1.12 | 61.7% / 88.3% / 3.52 | 68.3% / 78.3% / 3.37 | 65.0% / 83.3% |
| 稳妥保护 (jw=0.45+PCGrad) | `overlap_30_v4` | 100.0% / 100.0% / 1.07 | 56.7% / 83.3% / 3.77 | 66.7% / 80.0% / 3.40 | 61.7% / 81.7% |
| 基线 (jw=0.15) | `zero_overlap_v4` | 95.0% / 100.0% / 1.28 | 50.0% / 70.0% / 6.60 | 48.3% / 58.3% / 5.35 | 49.2% / 64.2% |
| 稳妥保护 (jw=0.45+PCGrad) | `zero_overlap_v4` | 98.3% / 100.0% / 1.17 | 48.3% / 66.7% / 7.08 | 48.3% / 60.0% / 5.60 | 48.3% / 63.3% |

## 三、自然度与前缀预算门禁检验

| 方案 | 组别 | Gate 通过 | SSIM | LPIPS | PSNR (dB) | FG RGB MAE (levels) | ROI Max Diff | Halo Error |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 基线 (jw=0.15) | `02_chair` | PASS | 0.9420 | 0.0811 | 28.45 | 5.14 | 0.0000 | 2.14 |
| 基线 (jw=0.15) | `05_hair_extension` | PASS | 0.9000 | 0.0751 | 28.07 | 7.52 | 0.0000 | 2.84 |
| 基线 (jw=0.15) | `09_toothbrush` | PASS | 0.9762 | 0.0206 | 25.09 | 7.17 | 0.0000 | 3.70 |
| 稳妥保护 (jw=0.45+PCGrad) | `02_chair` | PASS | 0.9416 | 0.0792 | 28.94 | 5.18 | 0.0000 | 1.87 |
| 稳妥保护 (jw=0.45+PCGrad) | `05_hair_extension` | PASS | 0.9005 | 0.0747 | 27.95 | 7.69 | 0.0000 | 2.55 |
| 稳妥保护 (jw=0.45+PCGrad) | `09_toothbrush` | PASS | 0.9756 | 0.0204 | 25.17 | 7.13 | 0.0000 | 3.78 |

## 四、自动决策门评估结果

- Env4 CLIP-L/14 Top-3 增量：`+20.00 pp`（门槛 `>= +8.00 pp`）；Top-5 增量：`+0.00 pp`（门槛 `>= 0.00 pp`）
- Stage 1 `training_oracle_v4` 变化：WB Top-5 `+0.00 pp`，Mean BB Top-5 `+0.00 pp`（门槛 `>= -2.00 pp`）
- Stage 1 `overlap_30_v4` 变化：WB Top-5 `+0.00 pp`，Mean BB Top-5 `-1.67 pp`（门槛 `>= -2.00 pp`）
- Stage 1 `zero_overlap_v4` 变化：WB Top-5 `+0.00 pp`，Mean BB Top-5 `-0.83 pp`（门槛 `>= -2.00 pp`）
- 自然度与印刷区红线全部通过：`True`
- **最终决策**：**PROMOTE（更新 ambilage.clip_l14.example.json）**
