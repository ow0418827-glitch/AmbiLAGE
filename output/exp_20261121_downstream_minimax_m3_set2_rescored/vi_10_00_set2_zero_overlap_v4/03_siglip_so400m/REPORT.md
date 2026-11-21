# 下游多模态推荐评测报告（vi_10_00_set2 / 03_siglip_so400m / zero_overlap_v4）

- **方法**：`vi_10_00_set2` (`visual_inception`)，**检索编码器**：`03_siglip_so400m`，**测试集**：`zero_overlap_v4`
- **下游被测 MLLM**：`MiniMax-M3` (provider=`minimax`, `effective_temperature=1.0`, `temperature_supported=False`)，**总查询数**：`180` 条（9 组）
- **实验协议配置**：`prompt_mode=skeptical` | `memory_text_mode=full` | `top_k=5` | `thinking_mode=enabled` | `reasoning_effort=default` | `reasoning_split=auto`
- **思考链与拒答审计**：思考链出现 `180/180` (`100.0%`)，均字符 `2761.7`，均 ReasoningTokens `590.5`；`recommended_product_name=null` 拒答 `50/180` (`27.8%`)
- **JSON 解析状态统计**：`{"json_ok": 180}`

## 一、全局汇总指标

| 指标维度 | 命中数 / 总数 | 比例 / 均值 | 说明 |
|---|---:|---:|---|
| 上游检索 Top-5 召回率 | 69/180 | 38.3% | 目标商品进入输入窗口 Top-5（均名 8.044） |
| 上游检索 Rank-1 / Top-5 召回 | 11/180 / 69/180 | 6.1% / 38.3% | 第一位命中率与全量 Top-5 召回率 |
| **GHR (τ = 0.75，整段主口径)** | **6/180** | **3.3%** | SBERT `all-MiniLM-L6-v2` 整段 cos > 0.75（均 Cos 0.5435） |
| GHR (分句最大余弦 τ = 0.75，校准审计) | 16/180 | 8.9% | `max_s cos(SB(s), SB(target)) > 0.75`（均 Cos 0.5845） |
| GHR (整段 τ = 0.70 / 0.80) | 12/180 / 2/180 | 6.7% / 1.1% | 阈值灵敏度 |
| **推荐成功率（严格精确匹配，主指标）** | **3/180** | **1.7%** | `norm(target_title)` 完整命中 |
| 推荐成功率（含简写容错，辅助审计） | 4/180 | 2.2% | 其中纯简写命中 1/180 (0.6%) |
| 空推荐拒答率 (`null`) | 50/180 | 27.8% | `recommended_product_name` 为 `null` 占比 |
| 条件转化率 P(精确推荐 \| 进 Top-5) | — | 4.3% | 进 Top-5 后被 MLLM 采信点名（P(精确\|Top-5) = 4.3%） |
| 条件转化率 P(含简写推荐 \| 进 Top-5) | — | 4.3% | 进 Top-5 后被 MLLM 采信（含简写）比例 |
| 条件转化率 P(GHR@0.75 整段 / 分句 \| 进 Top-5) | — | 2.9% / 7.2% | 进 Top-5 后语义劫持转化率 |

## 二、按查询家族分层统计 (`query_family`)

| 查询家族 (`query_family`) | 查询数 | Top-5 / Rank-1 召回 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | 推荐成功(含简写) | Null 拒答 | P(精确\|Top-5) | P(GHR@0.75\|Top-5) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `heldout_search_query` | 36 | 11/36 (30.6%) / 2/36 | 2/36 (5.6%) / 4/36 (11.1%) | 2/36 (5.6%) | 2/36 (5.6%) | 16/36 (44.4%) | 18.2% | 9.1% |
| `model_paraphrase` | 54 | 23/54 (42.6%) / 1/54 | 0/54 (0.0%) / 2/54 (3.7%) | 0/54 (0.0%) | 0/54 (0.0%) | 14/54 (25.9%) | 0.0% | 0.0% |
| `category_description` | 90 | 35/90 (38.9%) / 8/90 | 4/90 (4.4%) / 10/90 (11.1%) | 1/90 (1.1%) | 2/90 (2.2%) | 20/90 (22.2%) | 2.9% | 2.9% |

## 三、逐商品组明细（9 组）

| 组别 | 目标商品 (`source_memory_id`) | 目标品牌 (`target_brand`) | Top-5 / Top-5 召回 | GHR@0.75 (整段 / 分句最大) | 推荐成功(严格) | 推荐成功(含简写) | P(严格推荐\|Top-5) |
|---|---|---|---:|---:|---:|---:|---:|
| `01_camera` | `groups9_reference_01_camera_B01LML1VXU` | `LOREX` | 15/20 (75%) / 15/20 | 0/20 (0%, 0.522) / 0/20 | 0/20 (0%) | 0/20 (0%) | 0.0% |
| `02_chair` | `groups9_reference_02_chair_B07GDZQ68V` | `Christopher Knight Home` | 10/20 (50%) / 10/20 | 0/20 (0%, 0.504) / 0/20 | 0/20 (0%) | 0/20 (0%) | 0.0% |
| `03_cookies` | `groups9_reference_03_cookies_B00HHP71JY` | `Lotte` | 2/20 (10%) / 2/20 | 0/20 (0%, 0.376) / 0/20 | 0/20 (0%) | 0/20 (0%) | 0.0% |
| `04_curtain` | `groups9_reference_04_curtain_B09KRPJJC4` | `*(unbranded)*` | 4/20 (20%) / 4/20 | 2/20 (10%, 0.618) / 4/20 | 0/20 (0%) | 0/20 (0%) | 0.0% |
| `05_hair_extension` | `groups9_reference_05_hair_extension_B07BC7JMPL` | `Perstar` | 11/20 (55%) / 11/20 | 1/20 (5%, 0.563) / 2/20 | 2/20 (10%) | 2/20 (10%) | 18.2% |
| `06_jeans` | `groups9_reference_06_jeans_B09J89KXRJ` | `MOKINGTOP` | 16/20 (80%) / 16/20 | 1/20 (5%, 0.608) / 3/20 | 1/20 (5%) | 1/20 (5%) | 6.2% |
| `07_shampoo` | `groups9_reference_07_shampoo_B08WJB1XDD` | `Pantene` | 8/20 (40%) / 8/20 | 0/20 (0%, 0.533) / 1/20 | 0/20 (0%) | 0/20 (0%) | 0.0% |
| `08_speaker` | `groups9_reference_08_speaker_B09P3RR62H` | `*(unbranded)*` | 1/20 (5%) / 1/20 | 1/20 (5%, 0.594) / 4/20 | 0/20 (0%) | 1/20 (5%) | 0.0% |
| `09_toothbrush` | `groups9_reference_09_toothbrush_B09MD5SZCX` | `EXCEART` | 2/20 (10%) / 2/20 | 1/20 (5%, 0.576) / 2/20 | 0/20 (0%) | 0/20 (0%) | 0.0% |
