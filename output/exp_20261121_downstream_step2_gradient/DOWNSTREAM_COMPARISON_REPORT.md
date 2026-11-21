# 下游多模态推荐评测对比总表

## 一、全局指标对比总表

| 运行标签 / 方法 | 检索编码器 | 测试集 | 被测 MLLM | Prompt / 记忆文本 / Top-k | CoT 配置 (均字符) | Top-k 召回率 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | Null 拒答率 | P(严格推荐\|Top-k) | P(GHR@0.75\|Top-k) |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `training_oracle_v4` | `MiniMax-M3` | `personalized` / `title_brand_only` / `k=5` | `disabled` / `default` (0) | 180/180 (100.0%) | **82/180 (45.6%)** / 85/180 (47.2%) | **60/180 (33.3%)** | 0.0% | 33.3% | 45.6% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `overlap_30_v4` | `MiniMax-M3` | `personalized` / `title_brand_only` / `k=5` | `disabled` / `default` (0) | 180/180 (100.0%) | **88/180 (48.9%)** / 94/180 (52.2%) | **62/180 (34.4%)** | 0.0% | 34.4% | 48.9% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `MiniMax-M3` | `personalized` / `title_brand_only` / `k=5` | `disabled` / `default` (0) | 179/180 (99.4%) | **82/180 (45.6%)** / 86/180 (47.8%) | **57/180 (31.7%)** | 0.0% | 31.8% | 45.8% |
| `vi_10_00_set2` (`pilot_aligned_tb_k1_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `MiniMax-M3` | `personalized` / `title_brand_only` / `k=1` | `disabled` / `default` (0) | 52/54 (96.3%) | **54/54 (100.0%)** / 54/54 (100.0%) | **52/54 (96.3%)** | 0.0% | 100.0% | 100.0% |
| `vi_10_00_set2` (`pilot_aligned_tb_k3_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `MiniMax-M3` | `personalized` / `title_brand_only` / `k=3` | `disabled` / `default` (0) | 53/54 (98.1%) | **29/54 (53.7%)** / 31/54 (57.4%) | **24/54 (44.4%)** | 0.0% | 45.3% | 54.7% |
| `vi_10_00_set2` (`pilot_aligned_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `MiniMax-M3` | `personalized` / `title_brand_only` / `k=5` | `disabled` / `default` (0) | 53/54 (98.1%) | **25/54 (46.3%)** / 28/54 (51.9%) | **22/54 (40.7%)** | 0.0% | 41.5% | 47.2% |

## 二、按 `query_family` 分层对比总表

| 运行标签 / 方法 | 检索编码器 | 测试集 | 查询家族 | 查询数 | Top-k 召回 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | Null 拒答率 | P(严格推荐\|Top-k) | P(GHR@0.75\|Top-k) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `training_oracle_v4` | `heldout_search_query` | 36 | 36/36 (100.0%) | 20/36 (55.6%) / 20/36 (55.6%) | 14/36 (38.9%) | 0.0% | 38.9% | 55.6% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `training_oracle_v4` | `model_paraphrase` | 54 | 54/54 (100.0%) | 21/54 (38.9%) / 21/54 (38.9%) | 15/54 (27.8%) | 0.0% | 27.8% | 38.9% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `training_oracle_v4` | `category_description` | 90 | 90/90 (100.0%) | 41/90 (45.6%) / 44/90 (48.9%) | 31/90 (34.4%) | 0.0% | 34.4% | 45.6% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `overlap_30_v4` | `heldout_search_query` | 36 | 36/36 (100.0%) | 21/36 (58.3%) / 22/36 (61.1%) | 14/36 (38.9%) | 0.0% | 38.9% | 58.3% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `overlap_30_v4` | `model_paraphrase` | 54 | 54/54 (100.0%) | 19/54 (35.2%) / 21/54 (38.9%) | 17/54 (31.5%) | 0.0% | 31.5% | 35.2% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `overlap_30_v4` | `category_description` | 90 | 90/90 (100.0%) | 48/90 (53.3%) / 51/90 (56.7%) | 31/90 (34.4%) | 0.0% | 34.4% | 53.3% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `heldout_search_query` | 36 | 36/36 (100.0%) | 20/36 (55.6%) / 21/36 (58.3%) | 15/36 (41.7%) | 0.0% | 41.7% | 55.6% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `model_paraphrase` | 54 | 53/54 (98.1%) | 19/54 (35.2%) / 20/54 (37.0%) | 14/54 (25.9%) | 0.0% | 26.4% | 35.8% |
| `vi_10_00_set2` (`aligned_tb_k5_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `category_description` | 90 | 90/90 (100.0%) | 43/90 (47.8%) / 45/90 (50.0%) | 28/90 (31.1%) | 0.0% | 31.1% | 47.8% |
| `vi_10_00_set2` (`pilot_aligned_tb_k1_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `heldout_search_query` | 18 | 17/18 (94.4%) | 18/18 (100.0%) / 18/18 (100.0%) | 17/18 (94.4%) | 0.0% | 100.0% | 100.0% |
| `vi_10_00_set2` (`pilot_aligned_tb_k1_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `model_paraphrase` | 18 | 17/18 (94.4%) | 18/18 (100.0%) / 18/18 (100.0%) | 17/18 (94.4%) | 0.0% | 100.0% | 100.0% |
| `vi_10_00_set2` (`pilot_aligned_tb_k1_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `category_description` | 18 | 18/18 (100.0%) | 18/18 (100.0%) / 18/18 (100.0%) | 18/18 (100.0%) | 0.0% | 100.0% | 100.0% |
| `vi_10_00_set2` (`pilot_aligned_tb_k3_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `heldout_search_query` | 18 | 18/18 (100.0%) | 11/18 (61.1%) / 11/18 (61.1%) | 8/18 (44.4%) | 0.0% | 44.4% | 61.1% |
| `vi_10_00_set2` (`pilot_aligned_tb_k3_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `model_paraphrase` | 18 | 17/18 (94.4%) | 8/18 (44.4%) / 9/18 (50.0%) | 7/18 (38.9%) | 0.0% | 41.2% | 47.1% |
| `vi_10_00_set2` (`pilot_aligned_tb_k3_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `category_description` | 18 | 18/18 (100.0%) | 10/18 (55.6%) / 11/18 (61.1%) | 9/18 (50.0%) | 0.0% | 50.0% | 55.6% |
| `vi_10_00_set2` (`pilot_aligned_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `heldout_search_query` | 18 | 18/18 (100.0%) | 13/18 (72.2%) / 14/18 (77.8%) | 12/18 (66.7%) | 0.0% | 66.7% | 72.2% |
| `vi_10_00_set2` (`pilot_aligned_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `model_paraphrase` | 18 | 17/18 (94.4%) | 7/18 (38.9%) / 7/18 (38.9%) | 5/18 (27.8%) | 0.0% | 29.4% | 41.2% |
| `vi_10_00_set2` (`pilot_aligned_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `category_description` | 18 | 18/18 (100.0%) | 5/18 (27.8%) / 7/18 (38.9%) | 5/18 (27.8%) | 0.0% | 27.8% | 27.8% |
