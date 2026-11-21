# 下游多模态推荐评测对比总表

## 一、全局指标对比总表

| 运行标签 / 方法 | 检索编码器 | 测试集 | 被测 MLLM | Prompt / 记忆文本 / Top-k | CoT 配置 (均字符) | Top-k 召回率 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | Null 拒答率 | P(严格推荐\|Top-k) | P(GHR@0.75\|Top-k) |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `vi_10_00_set2` | `01_clip_l14` | `zero_overlap_v4` | `MiniMax-M3` | `skeptical` / `full` / `k=5` | `enabled` / `default` (2659) | 179/180 (99.4%) | **16/180 (8.9%)** / 25/180 (13.9%) | **18/180 (10.0%)** | 30.6% | 10.1% | 8.9% |
| `vi_10_00_set2` (`pilot_aligned_cot_enabled`) | `01_clip_l14` | `zero_overlap_v4` | `MiniMax-M3` | `personalized` / `title_brand_only` / `k=5` | `enabled` / `default` (1920) | 53/54 (98.1%) | **23/54 (42.6%)** / 26/54 (48.1%) | **18/54 (33.3%)** | 0.0% | 34.0% | 41.5% |
| `vi_10_00_set2` (`pilot_aligned_cot_low`) | `01_clip_l14` | `zero_overlap_v4` | `MiniMax-M3` | `personalized` / `title_brand_only` / `k=5` | `enabled` / `low` (1812) | 53/54 (98.1%) | **23/54 (42.6%)** / 25/54 (46.3%) | **18/54 (33.3%)** | 0.0% | 34.0% | 43.4% |
| `vi_10_00_set2` (`pilot_aligned_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `MiniMax-M3` | `personalized` / `title_brand_only` / `k=5` | `disabled` / `default` (0) | 53/54 (98.1%) | **25/54 (46.3%)** / 28/54 (51.9%) | **22/54 (40.7%)** | 0.0% | 41.5% | 47.2% |

## 二、按 `query_family` 分层对比总表

| 运行标签 / 方法 | 检索编码器 | 测试集 | 查询家族 | 查询数 | Top-k 召回 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | Null 拒答率 | P(严格推荐\|Top-k) | P(GHR@0.75\|Top-k) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `vi_10_00_set2` | `01_clip_l14` | `zero_overlap_v4` | `heldout_search_query` | 36 | 36/36 (100.0%) | 3/36 (8.3%) / 5/36 (13.9%) | 3/36 (8.3%) | 41.7% | 8.3% | 8.3% |
| `vi_10_00_set2` | `01_clip_l14` | `zero_overlap_v4` | `model_paraphrase` | 54 | 53/54 (98.1%) | 2/54 (3.7%) / 2/54 (3.7%) | 4/54 (7.4%) | 31.5% | 7.5% | 3.8% |
| `vi_10_00_set2` | `01_clip_l14` | `zero_overlap_v4` | `category_description` | 90 | 90/90 (100.0%) | 11/90 (12.2%) / 18/90 (20.0%) | 11/90 (12.2%) | 25.6% | 12.2% | 12.2% |
| `vi_10_00_set2` (`pilot_aligned_cot_enabled`) | `01_clip_l14` | `zero_overlap_v4` | `heldout_search_query` | 18 | 18/18 (100.0%) | 11/18 (61.1%) / 11/18 (61.1%) | 8/18 (44.4%) | 0.0% | 44.4% | 61.1% |
| `vi_10_00_set2` (`pilot_aligned_cot_enabled`) | `01_clip_l14` | `zero_overlap_v4` | `model_paraphrase` | 18 | 17/18 (94.4%) | 6/18 (33.3%) / 7/18 (38.9%) | 5/18 (27.8%) | 0.0% | 29.4% | 29.4% |
| `vi_10_00_set2` (`pilot_aligned_cot_enabled`) | `01_clip_l14` | `zero_overlap_v4` | `category_description` | 18 | 18/18 (100.0%) | 6/18 (33.3%) / 8/18 (44.4%) | 5/18 (27.8%) | 0.0% | 27.8% | 33.3% |
| `vi_10_00_set2` (`pilot_aligned_cot_low`) | `01_clip_l14` | `zero_overlap_v4` | `heldout_search_query` | 18 | 18/18 (100.0%) | 12/18 (66.7%) / 13/18 (72.2%) | 11/18 (61.1%) | 0.0% | 61.1% | 66.7% |
| `vi_10_00_set2` (`pilot_aligned_cot_low`) | `01_clip_l14` | `zero_overlap_v4` | `model_paraphrase` | 18 | 17/18 (94.4%) | 6/18 (33.3%) / 6/18 (33.3%) | 5/18 (27.8%) | 0.0% | 29.4% | 35.3% |
| `vi_10_00_set2` (`pilot_aligned_cot_low`) | `01_clip_l14` | `zero_overlap_v4` | `category_description` | 18 | 18/18 (100.0%) | 5/18 (27.8%) / 6/18 (33.3%) | 2/18 (11.1%) | 0.0% | 11.1% | 27.8% |
| `vi_10_00_set2` (`pilot_aligned_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `heldout_search_query` | 18 | 18/18 (100.0%) | 13/18 (72.2%) / 14/18 (77.8%) | 12/18 (66.7%) | 0.0% | 66.7% | 72.2% |
| `vi_10_00_set2` (`pilot_aligned_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `model_paraphrase` | 18 | 17/18 (94.4%) | 7/18 (38.9%) / 7/18 (38.9%) | 5/18 (27.8%) | 0.0% | 29.4% | 41.2% |
| `vi_10_00_set2` (`pilot_aligned_cot_disabled`) | `01_clip_l14` | `zero_overlap_v4` | `category_description` | 18 | 18/18 (100.0%) | 5/18 (27.8%) / 7/18 (38.9%) | 5/18 (27.8%) | 0.0% | 27.8% | 27.8% |
