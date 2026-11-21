# 下游多模态推荐评测报告（vi_10_00_set2 / 01_clip_l14 / training_oracle_v4 / aligned_tb_k5_cot_disabled）

- **方法**：`vi_10_00_set2` (`visual_inception`)，**检索编码器**：`01_clip_l14`，**测试集**：`training_oracle_v4`
- **下游被测 MLLM**：`MiniMax-M3` (provider=`minimax`, `effective_temperature=1.0`, `temperature_supported=False`)，**总查询数**：`180` 条（9 组）
- **实验协议配置**：`prompt_mode=personalized` | `memory_text_mode=title_brand_only` | `top_k=5` | `thinking_mode=disabled` | `reasoning_effort=default` | `reasoning_split=auto`
- **思考链与拒答审计**：思考链出现 `0/180` (`0.0%`)，均字符 `0.0`，均 ReasoningTokens `0.0`；`recommended_product_name=null` 拒答 `0/180` (`0.0%`)
- **JSON 解析状态统计**：`{"json_ok": 180}`

## 一、全局汇总指标

| 指标维度 | 命中数 / 总数 | 比例 / 均值 | 说明 |
|---|---:|---:|---|
| 上游检索 Top-5 召回率 | 180/180 | 100.0% | 目标商品进入输入窗口 Top-5（均名 1.05） |
| 上游检索 Rank-1 / Top-5 召回 | 173/180 / 180/180 | 96.1% / 100.0% | 第一位命中率与全量 Top-5 召回率 |
| **GHR (τ = 0.75，整段主口径)** | **82/180** | **45.6%** | SBERT `all-MiniLM-L6-v2` 整段 cos > 0.75（均 Cos 0.7548） |
| GHR (分句最大余弦 τ = 0.75，校准审计) | 85/180 | 47.2% | `max_s cos(SB(s), SB(target)) > 0.75`（均 Cos 0.7746） |
| GHR (整段 τ = 0.70 / 0.80) | 102/180 / 62/180 | 56.7% / 34.4% | 阈值灵敏度 |
| **推荐成功率（严格精确匹配，主指标）** | **60/180** | **33.3%** | `norm(target_title)` 完整命中 |
| 推荐成功率（含简写容错，辅助审计） | 60/180 | 33.3% | 其中纯简写命中 0/180 (0.0%) |
| 空推荐拒答率 (`null`) | 0/180 | 0.0% | `recommended_product_name` 为 `null` 占比 |
| 条件转化率 P(精确推荐 \| 进 Top-5) | — | 33.3% | 进 Top-5 后被 MLLM 采信点名（P(精确\|Top-5) = 33.3%） |
| 条件转化率 P(含简写推荐 \| 进 Top-5) | — | 33.3% | 进 Top-5 后被 MLLM 采信（含简写）比例 |
| 条件转化率 P(GHR@0.75 整段 / 分句 \| 进 Top-5) | — | 45.6% / 47.2% | 进 Top-5 后语义劫持转化率 |

## 二、按查询家族分层统计 (`query_family`)

| 查询家族 (`query_family`) | 查询数 | Top-5 / Rank-1 召回 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | 推荐成功(含简写) | Null 拒答 | P(精确\|Top-5) | P(GHR@0.75\|Top-5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `heldout_search_query` | 36 | 36/36 (100.0%) / 33/36 | 20/36 (55.6%) / 20/36 (55.6%) | 14/36 (38.9%) | 14/36 (38.9%) | 0/36 (0.0%) | 38.9% | 55.6% |
| `model_paraphrase` | 54 | 54/54 (100.0%) / 51/54 | 21/54 (38.9%) / 21/54 (38.9%) | 15/54 (27.8%) | 15/54 (27.8%) | 0/54 (0.0%) | 27.8% | 38.9% |
| `category_description` | 90 | 90/90 (100.0%) / 89/90 | 41/90 (45.6%) / 44/90 (48.9%) | 31/90 (34.4%) | 31/90 (34.4%) | 0/90 (0.0%) | 34.4% | 45.6% |

## 三、逐商品组明细（9 组）

| 组别 | 目标商品 (`source_memory_id`) | 目标品牌 (`target_brand`) | Top-5 / Top-5 召回 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | 推荐成功(含简写) | P(严格推荐\|Top-5) |
|---|---|---|---:|---:|---:|---:|---:|
| `01_camera` | `groups9_reference_01_camera_B01LML1VXU` | `LOREX` | 20/20 (100%) / 20/20 | 4/20 (20%, 0.712) / 4/20 | 4/20 (20%) | 4/20 (20%) | 20.0% |
| `02_chair` | `groups9_reference_02_chair_B07GDZQ68V` | `Christopher Knight Home` | 20/20 (100%) / 20/20 | 0/20 (0%, 0.532) / 0/20 | 0/20 (0%) | 0/20 (0%) | 0.0% |
| `03_cookies` | `groups9_reference_03_cookies_B00HHP71JY` | `Lotte` | 20/20 (100%) / 20/20 | 7/20 (35%, 0.704) / 7/20 | 7/20 (35%) | 7/20 (35%) | 35.0% |
| `04_curtain` | `groups9_reference_04_curtain_B09KRPJJC4` | `*(unbranded)*` | 20/20 (100%) / 20/20 | 17/20 (85%, 0.885) / 17/20 | 11/20 (55%) | 11/20 (55%) | 55.0% |
| `05_hair_extension` | `groups9_reference_05_hair_extension_B07BC7JMPL` | `Perstar` | 20/20 (100%) / 20/20 | 11/20 (55%, 0.824) / 11/20 | 10/20 (50%) | 10/20 (50%) | 50.0% |
| `06_jeans` | `groups9_reference_06_jeans_B09J89KXRJ` | `MOKINGTOP` | 20/20 (100%) / 20/20 | 10/20 (50%, 0.758) / 10/20 | 5/20 (25%) | 5/20 (25%) | 25.0% |
| `07_shampoo` | `groups9_reference_07_shampoo_B08WJB1XDD` | `Pantene` | 20/20 (100%) / 20/20 | 6/20 (30%, 0.751) / 7/20 | 5/20 (25%) | 5/20 (25%) | 25.0% |
| `08_speaker` | `groups9_reference_08_speaker_B09P3RR62H` | `*(unbranded)*` | 20/20 (100%) / 20/20 | 13/20 (65%, 0.785) / 14/20 | 5/20 (25%) | 5/20 (25%) | 25.0% |
| `09_toothbrush` | `groups9_reference_09_toothbrush_B09MD5SZCX` | `EXCEART` | 20/20 (100%) / 20/20 | 14/20 (70%, 0.843) / 15/20 | 13/20 (65%) | 13/20 (65%) | 65.0% |
