# 训练代理选择与比较口径

新方法默认采用 lucid（原始像素方法）当前配置中的同一组三代理，并直接调用同一权重解析器、装载器和预处理路径。模型身份同时包含架构、预训练标签及公开仓库；不能只比较架构名称。

| 默认模型 | 预训练权重 | 配置来源 |
|---|---|---|
| ViT-B-16（基础型十六像素图像块视觉变换器） | openai（开放人工智能公司公开权重） | 原始像素方法配置 |
| ViT-B-32（基础型三十二像素图像块视觉变换器） | 同上 | 原始像素方法配置 |
| 同一三十二像素图像块架构 | laion2b_s34b_b79k（二十亿图文数据预训练版本） | 原始像素方法配置 |

后两项是同架构的不同权重，分别参加训练。默认直接读取 `src/visual/config/optimization.json`（原始像素方法配置），不另复制一份容易偏离的默认列表。

## 命令参数

`--surrogate-set`（训练代理方案）可以选择默认三代理或 `heterogeneous4`（四代理异构组合）。`--surrogate-config`（自定义代理配置文件）读取项目内的公开模型列表，两种参数互斥。命令参数优先于方法配置；未传参数时沿用方法配置，方法配置也未指定时默认三代理。

默认方案可以显式写出，也可以省略参数：

```bash
python -B -X utf8 tools/prepare_models.py --scope all --surrogate-set lucid --download
python -B -X utf8 tools/preflight.py --require-models --surrogate-set lucid
python -B -X utf8 tools/run_experiment.py generate --group all --method ambilage --device cuda --surrogate-set lucid --output output/groups9_lucid_proxies
```

原来的四代理组合保留为显式备选，模型为十六像素基础模型、十四像素大型模型、四亿规模视觉语言模型及十四像素超大型模型。其配置位于 `src/visual/config/surrogates.heterogeneous4.json`（四代理组合配置）：

```bash
python -B -X utf8 tools/prepare_models.py --scope all --surrogate-set heterogeneous4 --download
python -B -X utf8 tools/preflight.py --require-models --surrogate-set heterogeneous4
python -B -X utf8 tools/run_experiment.py generate --group all --method ambilage --device cuda --surrogate-set heterogeneous4 --output output/groups9_heterogeneous4
```

自定义示例使用两代理，文件中的顺序就是训练顺序：

```bash
python -B -X utf8 tools/prepare_models.py --scope all --surrogate-config src/visual/config/training_surrogates.custom.example.json --download
python -B -X utf8 tools/preflight.py --require-models --surrogate-config src/visual/config/training_surrogates.custom.example.json
python -B -X utf8 tools/run_experiment.py generate --group all --method ambilage --device cuda --surrogate-config src/visual/config/training_surrogates.custom.example.json --output output/groups9_custom_proxies
```

单图入口 `tools/generate.py`（统一图片生成工具）及 `run_pipeline.sh`（统一阶段入口）支持同样的两个参数。评估阶段读取成图已冻结的代理记录；不能在评估命令中指定另一训练方案来改变图片身份。固定九组原始像素对照继续按历史记录复用，这两个新参数不用于修改该固定对照。

## 自定义文件及本地权重

配置文件须放在项目内，包含非空 `surrogates`（代理列表）；每项明确提供模型名、预训练标签和公开仓库，可再提供 `weight_path`（权重文件路径）及 `tokenizer_path`（分词器路径）。文件相对路径按项目根目录解释，权重及分词器相对路径按模型根目录解释。示例文件可复制后编辑并随仓库保存，真实权重继续不提交。

模型准备工具可以按公开仓库找到资源清单中已登记的模型；清单外的自定义模型需要显式准备本地权重，其他组件可用 `--models`（显式模型编号）单独下载。训练过程禁止自动下载、改用其他权重或回退到其他设备。配置不允许混入令牌等无关字段。

不同预训练来源允许共用相同架构，但相同模型身份不能重复。两个留出评估编码器始终禁止进入训练，包括通过自定义名称、仓库或权重目录指向它们的情形。自定义组合不改变此前收口的唯一图文联合训练目标。

## 冻结、恢复与审计

生成前冻结所选方案、顺序、配置摘要及模型身份；生成目录中的 `training_surrogates.json`（冻结代理选择记录）纳入成图文件摘要。实际模型记录继续保留权重摘要、预训练标签、激活模块、分词器和预处理信息。完成校验按所选代理数核对，不再固定要求四个。

切换组合必须使用新的输出目录；公开运行绑定、完成记录及冻结代理文件须一致。自定义原配置的选择结果也保存于生成目录，冻结结果验证不依赖后来被移动的原配置文件。内置组合定义改变时，旧图不能冒充新定义下的生成结果。

比较时默认对齐的是当前原始像素方法的模型配置及装载器。导入九张历史原始像素成图的记录显示三代理均使用快速激活，而当前共享装载器按预训练原生配置决定激活。因此模型名称及权重来源一致，不等于历史成图已经在当前装载器下重训；正式报告须核对实际权重与激活审计，不能宣称完整数值实现已自动对齐。本轮保留既有对照及其历史证据，不重新训练或改写。
