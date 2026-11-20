# 第二阶段第四版纯文字评测集

每组二十个配对位置。三档均包含十条公开商品原文、六条改写和四条短搜索词。训练原题逐字对应冻结训练包。三成档固定复用三条原描述、两条原改写和一条原搜索词。其余位置使用零重合档的对应记录。

| 查询环境 | 每组命中训练原文 | 九组命中训练原文 |
| --- | ---: | ---: |
| `training_oracle_v4` | 20/20 | 180/180 |
| `overlap_30_v4` | 6/20 | 54/180 |
| `zero_overlap_v4` | 0/20 | 0/180 |

[配对及来源清单](../resources/visual_inception_evaluation/exact_overlap_v4/pairs.json)绑定训练原文标识、留出商品原字段或公开搜索记录，以及改写所用的输入字段。规范化统一字符形式、大小写和空白。留出选择也排除标点变化、重排序和长文本近重复。曲奇面团补充材料来自旧商品的未用字段片段，均核对其正文未作为字段或字段片段进入训练。

**理发椅组的四条留出短搜索是临时生成补位**，标为 `synthetic_search_fallback`。其余八组共三十二条留出搜索都有公开原始记录。三档实际含临时补位分别为零、三、四条，校验结果单列数量。九组均纳入评测。

## 校验及服务器使用

```bash
for suite in training_oracle_v4 overlap_30_v4 zero_overlap_v4; do
  python -B -X utf8 tools/evaluate_visual_inception_category.py validate-queries --query-suite "$suite" || exit
done
```

服务器类别评估及三编码器评估通过 `--query-suite` 选择档位，每档使用独立输出目录。

## 重建

1. 本地公开商品库和上游搜索快照就绪后，运行 `tools/build_visual_inception_eval_v4_suites.py prepare`。准备过程需要 PyArrow(列式数据读取库)。
2. 将改写写成按组组织的对象，每组数组包含 `input_row_id` 和 `text`。外部模型接口由同一工具的 `generate` 子命令调用。可把各组输出合并，然后冻结清单。
3. 运行 `tools/build_visual_inception_eval_v4_suites.py freeze --rewrites <改写文件>`。用 `--search-fallbacks` 显式导入理发椅短搜索补位，输入为以组名为键、短文本数组为值的对象。真实来源补齐时，使用新版准备清单和新输出目录。来源还有缺额时，`--save-preparation` 仅保存准备材料。
