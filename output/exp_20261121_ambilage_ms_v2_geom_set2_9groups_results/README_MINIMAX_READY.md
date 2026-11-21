# AmbiLAGE 默认接口 (`ms_v2_geom`) Set 2 九组完整产物与 MiniMax-M3 测试指南

## 目录结构（独立目录）

- `precomputed_eval_vectors/`：完整的三编码器（`01_clip_l14`、`02_openclip_bigG14`、`03_siglip_so400m`）+ `sbert_ghr_targets.json` 预计算向量库（涵盖全部 `golden_band_9groups` 基础库、9 组默认 `ms_v2_geom` 成图、以及 3 套评估查询集）。
- `encoder_vectors_9groups/`：按编码器独立导出的 9 组 `modified_vectors [9, D]` 与 `clean_vectors [9, D]`（`.npz` + `manifest.json`）。
- `images/` 与 `images_index.json`：Set 2 全部 9 组的原图、修改图本地副本及 SHA-256 索引（原始冻结目录为 `/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_golden_band_9groups`）。
- `eval/`：完整 9 组同时替库（`--group all --replacement-groups all`）在 3 套查询集 × 3 编码器上的 `scores.npz` 与 `evaluation.json`（均已通过 `max_score_error <= 1e-7` 独立核验）。
- `SUMMARY_9G_MS_V2_GEOM.md` / `.json`：9 组全量评测对比汇总报告。

## 一键启动 MiniMax-M3 零 GPU 下游测试命令示例

方式一（使用独立目录内置完整预计算向量库，无需加载任何检索编码器、纯 CPU）：

```bash
cd /content/AmbiLAGE
/content/AmbiLAGE/.venv/bin/python -u tools/evaluate_downstream_recommendation.py \
  --base golden_band_9groups \
  --spec "ambilage_ms_v2_geom_set2:ambilage:/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_golden_band_9groups" \
  --precomputed-dir /content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_set2_9groups_results/precomputed_eval_vectors \
  --suites zero_overlap_v4 \
  --encoder-slugs 02_openclip_bigG14 03_siglip_so400m 01_clip_l14 \
  --output-dir output/exp_20261121_downstream_minimax_m3_set2_ms_v2_geom
```

方式二（直接复用本目录内已冻结的 `eval/` 检索排名与 `scores.npz`）：

```bash
cd /content/AmbiLAGE
/content/AmbiLAGE/.venv/bin/python -u tools/evaluate_downstream_recommendation.py \
  --base golden_band_9groups \
  --eval-root /content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_set2_9groups_results/eval \
  --tags ambilage_ms_v2_geom_set2 \
  --suites zero_overlap_v4 \
  --encoder-slugs 02_openclip_bigG14 03_siglip_so400m 01_clip_l14 \
  --output-dir output/exp_20261121_downstream_minimax_m3_set2_ms_v2_geom
```
