# 正式询问接口

两种公开方法共用同组冻结的二十条特点文字，训练只使用公开目标特点正文，不读取本清单。正式模式为 text（纯文字）或 joint（文字加独立照片）；运行参数必须与清单模式一致，缺图不能回退。

现有基础包已附九组纯文字正式清单，可直接由分阶段实验入口读取。每组二十条围绕公开目标品类及前三项特点，保留与训练正文重合的实际计数。新请求接口支持从指定基础包和组键构建同样格式：

```bash
python -B -X utf8 -m src.experiments.formal_queries --mode feature-text --base-package resources/evaluation_base --group 07_shampoo --output output/queries/shampoo_text.json
```

若要把冻结纯文字清单转成图文模式，先独立生成无明确商品名称、品牌及标志的查询照片，完成真实视觉核对，提供内容摘要及审核记录，然后运行：

```bash
python -B -X utf8 -m src.experiments.formal_queries --mode feature-joint --text-manifest output/queries/shampoo_text.json --image output/queries/shampoo_photo.png --image-review output/queries/shampoo_review.json --output output/queries/shampoo_joint.json
python -B -X utf8 tools/run_experiment.py evaluate --group 07_shampoo --method ambilage --output output/remote_one_group --query-mode joint --query-manifest output/queries/shampoo_joint.json --evaluation-name g14_joint --defense-mode none
```

这里的照片及审核文件需要实际准备，示例命令不会生成图片。多组运行不能共用一份目标清单，应分别提供本组输入并按同一固定协议运行。两方法必须复用同组相同照片和文字，不能根据名次更换输入。

评估输出按组、方法和评估名称分别保存。不同编码器、查询模式或防御范围使用新评估名称；图片仍复用已经冻结的同一批成图。正式评估输出包括逐请求名次、全部候选得分归档、全库独立复算凭证和原图与修改图的配对报告。
