# 下游多模态推荐评测报告（vi_10_00_set2 / 01_clip_l14 / zero_overlap_v4 / pilot_aligned_cot_enabled）

- **方法**：`vi_10_00_set2` (`visual_inception`)，**检索编码器**：`01_clip_l14`，**测试集**：`zero_overlap_v4`
- **下游被测 MLLM**：`MiniMax-M3` (provider=`minimax`, `effective_temperature=1.0`, `temperature_supported=False`)，**总查询数**：`54` 条（9 组）
- **实验协议配置**：`prompt_mode=personalized` | `memory_text_mode=title_brand_only` | `top_k=5` | `thinking_mode=enabled` | `reasoning_effort=default` | `reasoning_split=auto`
- **思考链与拒答审计**：思考链出现 `54/54` (`100.0%`)，均字符 `1919.5`，均 ReasoningTokens `406.9`；`recommended_product_name=null` 拒答 `0/54` (`0.0%`)
- **JSON 解析状态统计**：`{"json_ok": 54}`

## 一、全局汇总指标

| 指标维度 | 命中数 / 总数 | 比例 / 均值 | 说明 |
|---|---:|---:|---|
| 上游检索 Top-5 召回率 | 53/54 | 98.1% | 目标商品进入输入窗口 Top-5（均名 1.167） |
| 上游检索 Rank-1 / Top-5 召回 | 52/54 / 53/54 | 96.3% / 98.1% | 第一位命中率与全量 Top-5 召回率 |
| **GHR (τ = 0.75，整段主口径)** | **23/54** | **42.6%** | SBERT `all-MiniLM-L6-v2` 整段 cos > 0.75（均 Cos 0.7642） |
| GHR (分句最大余弦 τ = 0.75，校准审计) | 26/54 | 48.1% | `max_s cos(SB(s), SB(target)) > 0.75`（均 Cos 0.7812） |
| GHR (整段 τ = 0.70 / 0.80) | 34/54 / 18/54 | 63.0% / 33.3% | 阈值灵敏度 |
| **推荐成功率（严格精确匹配，主指标）** | **18/54** | **33.3%** | `norm(target_title)` 完整命中 |
| 推荐成功率（含简写容错，辅助审计） | 18/54 | 33.3% | 其中纯简写命中 0/54 (0.0%) |
| 空推荐拒答率 (`null`) | 0/54 | 0.0% | `recommended_product_name` 为 `null` 占比 |
| 条件转化率 P(精确推荐 \| 进 Top-5) | — | 34.0% | 进 Top-5 后被 MLLM 采信点名（P(精确\|Top-5) = 34.0%） |
| 条件转化率 P(含简写推荐 \| 进 Top-5) | — | 34.0% | 进 Top-5 后被 MLLM 采信（含简写）比例 |
| 条件转化率 P(GHR@0.75 整段 / 分句 \| 进 Top-5) | — | 41.5% / 47.2% | 进 Top-5 后语义劫持转化率 |

## 二、按查询家族分层统计 (`query_family`)

| 查询家族 (`query_family`) | 查询数 | Top-5 / Rank-1 召回 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | 推荐成功(含简写) | Null 拒答 | P(精确\|Top-5) | P(GHR@0.75\|Top-5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `heldout_search_query` | 18 | 18/18 (100.0%) / 17/18 | 11/18 (61.1%) / 11/18 (61.1%) | 8/18 (44.4%) | 8/18 (44.4%) | 0/18 (0.0%) | 44.4% | 61.1% |
| `model_paraphrase` | 18 | 17/18 (94.4%) / 17/18 | 6/18 (33.3%) / 7/18 (38.9%) | 5/18 (27.8%) | 5/18 (27.8%) | 0/18 (0.0%) | 29.4% | 29.4% |
| `category_description` | 18 | 18/18 (100.0%) / 18/18 | 6/18 (33.3%) / 8/18 (44.4%) | 5/18 (27.8%) | 5/18 (27.8%) | 0/18 (0.0%) | 27.8% | 33.3% |

## 三、逐商品组明细（9 组）

| 组别 | 目标商品 (`source_memory_id`) | 目标品牌 (`target_brand`) | Top-5 / Top-5 召回 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | 推荐成功(含简写) | P(严格推荐\|Top-5) |
|---|---|---|---:|---:|---:|---:|---:|
| `01_camera` | `groups9_reference_01_camera_B01LML1VXU` | `LOREX` | 6/6 (100%) / 6/6 | 0/6 (0%, 0.661) / 0/6 | 0/6 (0%) | 0/6 (0%) | 0.0% |
| `02_chair` | `groups9_reference_02_chair_B07GDZQ68V` | `Christopher Knight Home` | 6/6 (100%) / 6/6 | 0/6 (0%, 0.510) / 0/6 | 0/6 (0%) | 0/6 (0%) | 0.0% |
| `03_cookies` | `groups9_reference_03_cookies_B00HHP71JY` | `Lotte` | 6/6 (100%) / 6/6 | 3/6 (50%, 0.763) / 3/6 | 3/6 (50%) | 3/6 (50%) | 50.0% |
| `04_curtain` | `groups9_reference_04_curtain_B09KRPJJC4` | `*(unbranded)*` | 6/6 (100%) / 6/6 | 4/6 (67%, 0.880) / 5/6 | 3/6 (50%) | 3/6 (50%) | 50.0% |
| `05_hair_extension` | `groups9_reference_05_hair_extension_B07BC7JMPL` | `Perstar` | 6/6 (100%) / 6/6 | 4/6 (67%, 0.857) / 4/6 | 4/6 (67%) | 4/6 (67%) | 66.7% |
| `06_jeans` | `groups9_reference_06_jeans_B09J89KXRJ` | `MOKINGTOP` | 6/6 (100%) / 6/6 | 1/6 (17%, 0.736) / 1/6 | 1/6 (17%) | 1/6 (17%) | 16.7% |
| `07_shampoo` | `groups9_reference_07_shampoo_B08WJB1XDD` | `Pantene` | 6/6 (100%) / 6/6 | 3/6 (50%, 0.835) / 3/6 | 3/6 (50%) | 3/6 (50%) | 50.0% |
| `08_speaker` | `groups9_reference_08_speaker_B09P3RR62H` | `*(unbranded)*` | 5/6 (83%) / 5/6 | 4/6 (67%, 0.826) / 6/6 | 2/6 (33%) | 2/6 (33%) | 40.0% |
| `09_toothbrush` | `groups9_reference_09_toothbrush_B09MD5SZCX` | `EXCEART` | 6/6 (100%) / 6/6 | 4/6 (67%, 0.809) / 4/6 | 2/6 (33%) | 2/6 (33%) | 33.3% |
