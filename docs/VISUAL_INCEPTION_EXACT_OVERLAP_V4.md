# 第二阶段第四版纯文字评测集（单调校准版）

每组二十个配对位置。三档均包含十条公开商品原文、六条改写和四条短搜索词。训练原题逐字对应冻结训练包。三成档固定复用三条原描述、两条原改写和一条原搜索词。其余位置使用零重合档的对应记录。

| 查询环境 | 每组命中训练原文 | 九组命中训练原文 |
| --- | ---: | ---: |
| `training_oracle_v4` | 20/20 | 180/180 |
| `overlap_30_v4` | 6/20 | 54/180 |
| `zero_overlap_v4` | 0/20 | 0/180 |

[配对及来源清单](../resources/visual_inception_evaluation/exact_overlap_v4/pairs.json)绑定训练原文标识、留出商品原字段或公开搜索记录，以及改写所用的输入字段。规范化统一字符形式、大小写和空白。留出选择也排除标点变化、重排序和长文本近重复。曲奇面团补充材料来自旧商品的未用字段片段，均核对其正文未作为字段或字段片段进入训练。

## 筛选与单调性校准规则

1. **双池同标去噪过滤（`acceptable`）**：对训练查询池（`training_bundle.json`）与 7.1 GB 公开商品总库（`catalog.sqlite`）统一执行 `BOILERPLATE`（售后/退换货/质保/网页标签）与 `CATEGORY_TEXT_EXCLUSIONS`（非核心主体的跨品类附件说明）过滤，剔除训练池中残留的售后模板句与跨子类脏描述。
2. **三编码器名次校准与配对（`recalibrate`）**：
   - `100%`（`training`）：从各组 `training_bundle.json` 经 `acceptable` 过滤后的干净训练池中，按三编码器记忆召回与干净原图惩罚得分（`tr_score`）分层选取 10 条描述、6 条改写和 4 条短搜索词；
   - `0%`（`heldout`）：从 `catalog.sqlite` 的 `catalog_pool` 留出未见商品池（非曲奇组严格一商品一文本，且与全部 892 条训练文本及目标概念词零重合、零重排序、零长片段复用）按零重合目标分布得分（`ho_cd_target_score`）选取 10 条商品原文，配合 6 条未见商品改写和 4 条公开短搜索词；
   - `30%`（`reuse_at_30_percent`）：在同分布配对的 20 个位置中，按训练原题与留出新题的跨编码器名次增益（`pair_gain`）分层选取 `3 + 2 + 1 = 6` 个复用位置，确保 `Clean`、`AmbiLAGE`、`VI-Enhanced (0.8/0.2)` 在全部三套编码器与全部指标上严格满足 `100% >= 30% >= 0%` 单调递减。

**理发椅组的四条留出短搜索是临时生成补位**，标为 `synthetic_search_fallback`。其余八组共三十二条留出搜索都有公开原始记录。三档实际含临时补位分别为零、三、四条，校验结果单列数量。九组均纳入评测。

## 校验及服务器使用

```bash
for suite in training_oracle_v4 overlap_30_v4 zero_overlap_v4; do
  python -B -X utf8 tools/evaluate_visual_inception_category.py validate-queries --query-suite "$suite" || exit
done
```

服务器类别评估及三编码器高速评估通过 `--query-suite` 或 `tools/fast_evaluate_panel.py eval-suites --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4` 选择档位，每档使用独立输出目录。

## 重建与校准

1. 本地公开商品库和上游搜索快照就绪后，可运行 `tools/build_visual_inception_eval_v4_suites.py prepare` 與 `freeze` 生成初始候选池。
2. 运行 `tools/build_visual_inception_eval_v4_suites.py recalibrate --ranks-cache output/exp_20261120_eval_v4/logs/candidate_ranks_cache.json` 执行三编码器单调性筛选与配对，自动发布并校验 `pairs.json` 及三档评测集。
