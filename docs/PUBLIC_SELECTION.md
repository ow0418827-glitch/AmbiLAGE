# 轻量公开商品选图入口

当前入口为 `tools/select_public_pairs.py`（公开选图工具），配置使用第二版。默认保留两组公开目标，粗类型候选预算32张，使用公开开发集门槛和固定种子均匀抽样。完整数学定义见[选图协议](PAIR_SELECTION.md)。以下命令均在项目根目录、已激活项目运行环境后执行，同样适用于远程类优尼克斯系统；复用现有九组图对不会启动重新选图。

## 准备本地模型与真实开发标签

```bash
python -X utf8 -B tools/prepare_models.py --scope selection --download
```

选图阶段只需要一个冻结的小型视觉模型，准备工具优先使用中国大陆镜像。正式生成的分割、文字识别与生成模型仍在相应阶段准备。

把 `src/visual/config/shape_development.example.json`（开发标签模板）复制到公开开发校准目录，填入实际公开图片、来源、核对者、逐对说明及零或一标签。模板中的空标签不能用于校准。正负样本须等量；开发商品及其图片不能作为正式目标或候选。若需隔离同系列变体，可额外登记开发保留商品。

```bash
python -X utf8 -B tools/select_public_pairs.py calibrate --development output/datasets/public/calibration/development.json
```

证书写到配置指定的公开校准路径，包含实际逐对分数、阈值、拟合误差及模型身份。没有真实校准证书时停止，不将旧阈值或合成测试标签当作正式开发结果。更换模型、预处理或标签须另建证书与运行目录。

## 从本项目商品库取公开候选并抽图

先将仓库中的商品数据恢复块还原为本项目自己的数据库和原搜索索引，恢复不需要启动商城，也不调用购物模型：

```bash
python -B -X utf8 -m src.webshop.bundle restore
```

```bash
python -X utf8 -B tools/select_public_pairs.py prepare --products 'resources/webshop/catalog/catalog.sqlite'
python -X utf8 -B tools/select_public_pairs.py select
```


数据库已经恢复后，可直接执行上面的准备命令；默认商品路径也是这个本项目数据库。第一次建立公开字段索引时，数据库查询只返回商品编号、公开名称和第一张主图，不把指令、动作、私有类目、价格或奖励传入选图流程。后续不同目标复用本项目公开索引。数据库保存的是既有商城规范化商品记录，不能把它与未经规范化的原始大文件宣称为完全相同的候选集合；新选图须记录实际数据库摘要和投影协议，不改变历史冻结九组。`--verify-source`（重验原始文件内容参数）会额外重新计算原文件摘要；正常复用检查源文件元数据与类型规则身份。

未识别或多义类型必须提供配置中的公开类型核对清单。每条核对包括粗类型、核对者、依据及当前公开名称摘要。程序不把未知类型默认为兼容。

候选按粗类型和既定模拟可控划分截取，不计算标题相似度。下载失败保留证据，不自动补足到预算或改变种子。外形门槛之后，从合格集合抽样。空集合保存逐候选拒绝原因，返回状态码三，停止本组。

## 只标注选中的图对并发布

```bash
python -X utf8 -B tools/select_public_pairs.py annotate
```

此阶段才调用主体与文字标注模型，仅处理实际抽中的源图与目标图；不会给全部三十二张候选做分割和文字区域标注。

查看运行目录中的公开核对图，将实际核对结果写入公开资产目录的 `selected_visual_reviews.json`（选中图对视觉核对文件）。格式沿用逐图审核：顶层为实际核对者与当前标注清单摘要，记录中包含图片摘要、提议摘要、接受状态、必要的主体框及印刷区修正。不合格则停止，不能按正式名次重选。程序不生成自动通过的视觉声明。

```bash
python -X utf8 -B tools/select_public_pairs.py publish
python -X utf8 -B tools/select_public_pairs.py report
```

也可以运行一次完整选图命令：

```bash
python -X utf8 -B tools/select_public_pairs.py all --products 'resources/webshop/catalog/catalog.sqlite'
```

未填写真实视觉核对记录时，完整选图命令在标注后返回状态码二，等待核对，再使用发布阶段继续。这是公开标注输入缺失，不是额外审批流程。

## 正式文件与两种方法

默认运行目录为 `output/public_pair_selection/shape_v2/`（选图输出）；候选、下载与标注默认保存在该次运行的数据子目录。正式分组目录如下。

```text
output/public_pair_selection/shape_v2/data/groups/02_chair/
  source.jpg
  target.jpg
  pair.json
  pipeline.json
  metadata/
```

图片保留原始格式与字节，实际扩展名可能为其他受支持格式。不同实验使用不同运行目录，既有原商品图、目标商品图与生成图保持冻结。根数据目录暂不填入这些开发图对，待最终九组确认后再显式发布。

生成时使用该目录中的完整流程配置，两种方法共同读取相同 source 与 target；生成准备须已有实际核对的主体和印刷区。新配置的正式模糊查询路径也按运行版本隔离，避免自动复用已经观察过的查询。所有组仍指向一个共享长期记忆池。

本工具不修改共享记忆、不生成方法最终图、不进行正式检索或购物。旧多指标选择只保留历史文件读取，第一版配置不能启动新的旧评分实验。
