# 第二阶段第四版纯文字评测集

每组二十个配对位置。三档均包含十条公开商品原文、六条改写和四条短搜索词。训练原题逐字对应冻结训练包。三成档固定复用三条原描述、两条原改写和一条原搜索词。其余位置使用零重合档的对应记录。

| 查询环境 | 每组命中训练原文 | 九组命中训练原文 |
| --- | ---: | ---: |
| `training_oracle_v4` | 20/20 | 180/180 |
| `overlap_30_v4` | 6/20 | 54/180 |
| `zero_overlap_v4` | 0/20 | 0/180 |

[配对及来源清单](../resources/visual_inception_evaluation/exact_overlap_v4/pairs.json)绑定训练原文标识、留出商品原字段或公开搜索记录，以及改写所用的输入字段。规范化统一字符形式、大小写和空白。留出选择也排除标点变化、重排序和长文本近重复。曲奇面团补充材料来自旧商品的未用字段片段，均核对其正文未作为字段或字段片段进入训练。

## 纯规则筛选与去重协议

1. **双池同标去噪过滤（`acceptable`）**：对训练查询池（`training_bundle.json`）与 7.1 GB 公开商品总库（`catalog.sqlite`）统一执行：
   - `BOILERPLATE`（售后/退换货/质保/网页标签）与 `CATEGORY_TEXT_EXCLUSIONS`（非核心主体的跨品类附件说明）过滤；
   - `NOISE_HEAD` 开头噪声过滤（剔除 `SIZE CHART` 尺码表、纯规格参数头、纯面料成分头、`Suitable for daily/work...` 泛场景清单、店铺欢迎语等）；
   - `CATEGORY_ANCHOR_PATTERNS` 正向品类锚词校验（前 250 字符内须包含目标品类核心词，防止错品类或缺失核心主体的文本混入）。
2. **商品与视觉资产严格去重**：留出池按商品编号（ASIN）、去除颜色/尺寸变体词后的规范化标题键（`title_key`）及亚马逊主图资产编号（`images`）与训练使用商品严格隔离；结合 `DuplicateIndex` 校验零字面重合、零词序重排、零长片段复用（`03_cookies` 允许使用旧商品未进入训练的独立字段并核验零片段包含）。
3. **固定种子无泄露抽样、改写长度对齐与严格嵌套（`SEED = 20261120`）**：完全基于纯文本/ID 规则与固定随机种子抽样配对，不使用任何被测方法或评估编码器测试名次挑选题目；`model_paraphrase` 留出改写支持本地纯规则长度对齐生成（`generate-rule`，按组对齐训练集改写字符数分布并保留外接 MiniMax/OpenAI 模型 `generate` 接口）；`overlap_30_v4` 严格嵌套为 6 条 `training_oracle_v4`（3+2+1）+ 14 条 `zero_overlap_v4`（7+4+3）。

**理发椅组的四条留出短搜索是临时生成补位**，标为 `synthetic_search_fallback`。其余八组共三十二条留出搜索都有公开原始记录。三档实际含临时补位分别为零、三、四条，校验结果单列数量。九组均纳入评测。

## 校验及服务器使用

```bash
for suite in training_oracle_v4 overlap_30_v4 zero_overlap_v4; do
  python -B -X utf8 tools/evaluate_visual_inception_category.py validate-queries --query-suite "$suite" || exit
done
```

服务器类别评估及三编码器高速评估通过 `--query-suite` 或 `tools/fast_evaluate_panel.py eval-suites --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4` 选择档位，每档使用独立输出目录。

## 重建

1. 本地公开商品库和上游搜索快照就绪后，运行 `tools/build_visual_inception_eval_v4_suites.py prepare`（本地若无 Parquet 上游搜索库时，可通过 `--reuse-searches-from` 复用已审计 `pairs.json` 中的独立真实公开搜索记录）。
2. 留出改写支持两种方式：
   - **纯规则长度对齐生成（默认无外网依赖）**：运行 `tools/build_visual_inception_eval_v4_suites.py generate-rule --prepared <prepared.json> --output <改写文件>`。
   - **外接大模型生成（MiniMax/OpenAI 接口保留）**：通过 `tools/build_visual_inception_eval_v4_suites.py generate --provider minimax --model <模型名> ...` 调用或复用已有冻结改写与补位文件。
3. 运行 `tools/build_visual_inception_eval_v4_suites.py freeze --rewrites <改写文件> --search-fallbacks <补位文件>` 自动冻结并校验 `pairs.json` 及三档评测集。
