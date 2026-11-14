# 远程部署与验收

本地已经完成迁移修复及轻量验证。此文给出远程执行顺序，未执行的远程步骤不能记为通过。目标机器为 A100（八十吉字节显存计算卡）、约一百六十七吉字节内存和五百一十二吉字节磁盘；这些容量不能替代实际峰值测量。

## 同步与环境

远程服务器先克隆项目，再进入项目目录执行后续命令。Python（编程语言）需要三点十一版或兼容更新版本，显卡驱动须满足所选计算库。

```bash
git clone --depth 1 https://github.com/ow0418827-glitch/AmbiLAGE.git
cd AmbiLAGE
```

用户于二〇二六年十一月十四日明确授权创建公开 GitHub（代码托管平台）仓库并推送项目，覆盖此前私有仓库约定。真实模型密钥继续禁止上传。服务器从仓库克隆或更新后恢复源码、无密钥配置、九组公开图对、正式请求、一百八十九条共享记忆、一百九十八张共享主图、已冻结成图、实验结果及恢复断点。仓库内购买轨迹是本地商城实验记录，未购买参考保持原标签。公开地址为 https://github.com/ow0418827-glitch/AmbiLAGE 。

真实密钥配置 src/agent/config/llm.local.json（本地模型服务配置文件）仅在本机保留，被忽略规则排除，没有凭据例外。仓库只保存无密钥连接模板；服务器通过 MINIMAX_API_KEY（模型密钥环境变量）或被忽略的本地文件提供真实值，具体见[密钥配置说明](CREDENTIALS.md)。同步检查只输出路径和核验状态，不输出密钥值。权重、运行环境、可重建临时缓存、迁移旧档、测试临时目录及重复副本不参与同步。正式分数、必要向量和优化器断点不因二进制扩展名而被一概排除。

本地提交前运行：

```bash
python -B -X utf8 tools/check_github_sync.py
python -B -X utf8 tools/preflight.py --webshop --output output/deployment/local_preflight.json
python -B -X utf8 tools/plan_initial_sync.py --output-dir output/github_sync_check/initial_sync
```

同步检查使用版本管理工具实际列出的候选；未建立项目仓库时只使用输出目录中的独立临时索引，不修改项目主索引。它检查必要输入、结果与商城块可被纳入，并阻止真实密钥和超限文件。初次同步规划只写入被忽略的检查目录，不改变候选集合。普通完整预检仍会严格核对恢复的资源与内容摘要；商城选项额外核对源码摘要、恢复块结构及恢复副本完整性，不重复读取多吉字节文件做摘要。仅代码预检可用于安装阶段，不能代替正式资源核验。

仓库属性明确禁止对摘要绑定的资源、文本清单和历史证据自动转换换行，避免跨系统克隆后原始字节变化；服务器命令脚本单独固定为类统一系统换行。不要用编辑器批量格式化已冻结的资料。

```bash
nvidia-smi
python3.11 --version
bash tools/setup_runtime.sh
```

安装脚本使用独立虚拟环境，先从官方兼容轮子源安装 CUDA（显卡计算平台）版张量库，再安装普通依赖。普通包默认优先使用清华镜像；可设置 `PIP_INDEX_URL`（普通包索引地址）和 `TORCH_INDEX_URL`（张量库轮子地址）。驱动必须支持所选轮子版本；如服务器已有匹配环境，可显式设置 `AMBILAGE_PYTHON`（启动解释器路径），不必覆盖既有环境。依赖范围用于兼容安装，远程验收后应保存实际安装版本清单。

```bash
.venv/bin/python -m pip freeze > output/deployment/requirements-resolved.txt
.venv/bin/python -X utf8 tools/preflight.py --code-only --hardware --output output/deployment/remote_preflight_before_models.json
```

仅代码预检检查源码语法、依赖信息、模型准备状态及可选设备属性，主动跳过私有评估资源；不下载权重、不运行编码器。它报告缺少的包和权重。模型配置与新方法生产调用仍需要后续真实运行验证。

## 公开模型准备

先检查磁盘剩余容量并设置模型目录。下载服务令牌及模型调用密钥按服务器环境单独配置，不纳入仓库。受许可限制的模型需先在发布方页面接受条件。

```bash
export AMBILAGE_MODEL_ROOT="$PWD/models"
.venv/bin/python -X utf8 tools/prepare_models.py --scope all
.venv/bin/python -X utf8 tools/prepare_models.py --scope all --download
.venv/bin/python -X utf8 tools/preflight.py --code-only --hardware --require-models --output output/deployment/remote_models_preflight.json
```

从私有仓库恢复共享评估目录后，必须运行不带仅代码参数的完整预检。它核对一百八十九条记忆、一百九十八张共享主图、九组图对、一百八十条正式文字及绑定摘要；缺文件或摘要不符会停止，不回退到旧项目。

```bash
.venv/bin/python -X utf8 tools/preflight.py --hardware --require-models --output output/deployment/remote_preflight.json
```

默认下载端点为国内模型镜像，可显式追加 `--endpoint https://huggingface.co` 使用官方源。下载工具先解析实际仓库提交，再固定该提交下载；本地准备清单记录仓库、实际提交和逐文件内容摘要。它不下载量化文件或同模型的传统二进制重复权重。仓库内 resources/model_lock.json（模型恢复锁定清单）会保存这些提交与文件摘要；第一次真实下载后须将该清单一起提交。服务器清空后，准备工具优先按清单中的同一提交重新下载并逐文件比对，不能静默改用最新权重。本机尚未下载真实模型，因此当前锁定清单为空，这是实际状态。

默认生产组件是 SAM2.1-Hiera-Large（大型主体分割模型）、RapidOCR（快速文字识别工具）、Qwen2.5-VL-32B-Instruct（视觉语言规划与自然度检查模型）、FLUX.1-Kontext-dev（自然锚点编辑模型及十六通道变分自编码器）。复杂中文包装可显式准备 `editor_qwen`（中文包装编辑备选）；较小规划模型可显式准备 `planner_7b`（七十亿参数规划备选）。备选不会自动替换默认模型。四个训练代理和两个留出评估模型的完整身份见模型清单。另有 DINOv2-small（小型自监督视觉编码器）作为公开选图的单一冻结外形代理，目录键为 `selection`（公开选择模型）；可用 `--scope selection` 单独准备，完整准备范围也包含它。公开选图标注复用同一大型分割目录，并显式遵循选图命令的设备参数。既有九组复用不触发重新选图；新的门槛校准必须提供真实公开开发集逐对标签，示例夹具不能替代。

官方接口参考：[分割模型](https://huggingface.co/facebook/sam2.1-hiera-large)、[规划模型](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)、[编辑模型及许可](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)、[第二留出模型](https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct)。

## 真模型验收顺序

1. 运行离线单元测试与模型预检，确认全部接口和冻结输入摘要。
2. 用一组公开图对、固定种子运行真实生产生成，保存阶段状态、模型身份、印刷区域零变动核验和最终冻结证据。不要使用 `--smoke-test`（显式微型测试模式）。
3. 先完成全部拟比较图片并冻结，再开放共享池与正式查询的读取。独立进程重新编码及复算，不能复用旧编码器向量。
4. 确认一组的真实显存和磁盘占用后，再按冻结配置运行完整组数与防御矩阵。若出现技术错误，保存失败证据停止；不得改种子或重采样有效结果。

先在一组上验收新方法真实生产生成；原始像素方法只登记已导入成图，不再训练。正式检索必须等待对应方法全部九张图冻结，不能在仅完成一组时提前评价。下方完整组数流程用于正式比较。

原始像素对照已有完整成图，因此相关生成命令只会复用。下面示例不启用购物，不缩减默认训练日程；评估先选无防御，确认调用链与独立复算后再另建评估条件运行防御面板。

```bash
.venv/bin/python -X utf8 tools/run_experiment.py generate --group 07_shampoo --method lucid --output output/remote_one_group --device cuda
.venv/bin/python -X utf8 tools/run_experiment.py generate --group 07_shampoo --method ambilage --output output/remote_one_group --device cuda
```

一组真实模型验收完成后，再使用新的输出目录运行九组。仍先完成两方法全部生成并冻结，再分别评估和独立复算。

```bash
.venv/bin/python -X utf8 tools/run_experiment.py generate --group all --method lucid --output output/remote_groups9 --device cuda
.venv/bin/python -X utf8 tools/run_experiment.py generate --group all --method ambilage --output output/remote_groups9 --device cuda
.venv/bin/python -X utf8 tools/run_experiment.py evaluate --group all --method lucid --output output/remote_groups9 --device cuda --query-mode text --defense-mode panel --evaluation-name g14
.venv/bin/python -X utf8 tools/run_experiment.py verify --group all --method lucid --output output/remote_groups9 --device cuda --evaluation-name g14
.venv/bin/python -X utf8 tools/run_experiment.py evaluate --group all --method ambilage --output output/remote_groups9 --device cuda --query-mode text --defense-mode panel --evaluation-name g14
.venv/bin/python -X utf8 tools/run_experiment.py verify --group all --method ambilage --output output/remote_groups9 --device cuda --evaluation-name g14
```

默认留出配置使用巨型十四像素图像块编码器；第二留出配置见 `src/visual/config/evaluator.gme.example.json`（第二独立留出编码器示例）。完成默认留出评估后，可复用同一批冻结成图运行第二编码器：

```bash
.venv/bin/python -X utf8 tools/run_experiment.py evaluate --group all --method lucid --output output/remote_groups9 --device cuda --query-mode text --defense-mode panel --encoder-config src/visual/config/evaluator.gme.example.json --evaluation-name gme
.venv/bin/python -X utf8 tools/run_experiment.py verify --group all --method lucid --output output/remote_groups9 --device cuda --evaluation-name gme
.venv/bin/python -X utf8 tools/run_experiment.py evaluate --group all --method ambilage --output output/remote_groups9 --device cuda --query-mode text --defense-mode panel --encoder-config src/visual/config/evaluator.gme.example.json --evaluation-name gme
.venv/bin/python -X utf8 tools/run_experiment.py verify --group all --method ambilage --output output/remote_groups9 --device cuda --evaluation-name gme
```

不同编码器和防御条件须使用独立评估名称或输出目录，不能覆盖已有冻结评估计划。`--evaluation-name`（评估名称参数）把对应结果置于每组方法目录的独立评估子目录；复算和报告要使用同一个名称。图文联合模式必须显式提供逐组独立查询图片的合规清单；当前基础包只附正式文字，不能把商品原图当作缺失查询图的自动替代。

```bash
.venv/bin/python -X utf8 tools/run_experiment.py --help
bash run_pipeline.sh doctor --device cuda
```

统一生成入口只接受两个方法身份。完整实验入口的具体配置应指向本次已冻结的图对和查询；不带配置直接运行全阶段会使用示例路径，示例路径缺数据时应停止。正式运行时保留配置、代码摘要、安装版本、驱动及设备报告。

## 实验结果备份与清空后恢复

首次同步包含约二点七零吉二进制字节的商城压缩数据。GitHub 单次推送最大二吉二进制字节，普通单文件不得超过一百兆字节，因此数据已拆成最多六十四兆二进制字节的块，并提供每批最多五百一十二兆二进制字节的只读规划。首次提交必须按规划逐批暂存、提交并推送，推送成功后再继续下一批；不能先生成一个包含所有数据的大提交，也不能累积全部批次后只推送一次。规划命令不会自动执行任何版本管理写入。依据见[官方推送限制](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit)和[仓库限制](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)。

```bash
python -B -X utf8 tools/plan_initial_sync.py --output-dir output/github_sync_check/initial_sync
```

初次分批同步完成后，忽略规则仍只决定可提交范围，不会自动推送结果。实验阶段结束并确认记录落盘后，小批量更新可使用既有仓库流程实际提交与推送；再次运行同步检查以核对已暂存候选。不要在正在运行的实验中覆盖冻结输入或修改代码。大批新增结果仍须核对单文件和推送容量。

```bash
python -B -X utf8 tools/check_github_sync.py
git add -A
python -B -X utf8 tools/check_github_sync.py
git commit -m "保存实验输入、结果与恢复状态"
git push
```

上述命令用于初次分批同步完成后的结果更新。服务器清空后，重新克隆项目，重建运行环境，按模型恢复锁定清单重新准备权重，重新配置不在仓库内的真实密钥，再执行完整资源预检和所需的模型摘要核验。活动输入及历史对照导入器按当前项目根目录解析，克隆到不同目录也不读取来源项目；既有正式运行的冻结证据可能包含其当时绝对路径，续跑旧正式目录时仍需核对并显式迁移路径，保留原始证据，不能以重新采样替代恢复。

必要结果包括正式输入、冻结图、生成审计、原始策略响应、逐请求结果、候选分数、独立复算凭证、向量证据及恢复断点。不能把旧迁移归档和本地测试临时目录混入正式实验结果。模型下载产生的恢复锁定清单也须及时提交，否则服务器清空后无法仅凭模型名称保证恢复同一权重版本。

## 本地与远程证据边界

本地模拟测试只能证明相应逻辑、梯度连接、约束和调用契约。它不能证明真实模型的自然度、鲁棒性、黑盒迁移率、吞吐或八十吉字节显存适配。加载各阶段真实模型后，还需核验模型接口版本、峰值显存、实际图像质量及完整检索结果。四训练代理与两个评估留出模型身份不同，也不能单凭名称隔离就宣称对任意现实系统有效。

购物默认关闭。必要商城源码、完整商品目录和原搜索索引现已包含在项目恢复资料内，不再借用相邻目录。现有九组生成与检索不需要这些可选服务。重新选商品前恢复数据库；最终购物前还需安装 Java（二十一版或更新爪哇运行时）、固定版本搜索接口及大型英语模型：

```bash
.venv/bin/python -B -X utf8 -m src.webshop.bundle restore
.venv/bin/python -B -X utf8 tools/preflight.py --webshop --output output/deployment/webshop_resources.json
bash tools/setup_webshop.sh
.venv/bin/python -B -X utf8 tools/check_webshop_runtime.py --require --output output/deployment/webshop_runtime.json
```

恢复将仓库内202个数据块解码为项目内九十二份运行文件；数据库和原搜索索引逐文件摘要核验，元数据中的旧原始文本路径明确标为历史构建来源。恢复工具使用进程文件锁，失败后可接续，不重新构建索引或改变搜索排序。约需额外八点零八吉二进制字节保存恢复副本，安装依赖及模型另计。目录包含全商城商品图片网址；当前九组图片本体已经本地保存，未来新商品图片仍需联网。详细步骤见[商城运行依赖](WEBSHOP_RUNTIME.md)。本轮没有启动真实商城或购物。

当前九组源图、完整共享库及固定原始像素对照的复用规则、九图同时替换和增量向量更新见 [共享池复用说明](SHARED_POOL_REUSE.md)。原始像素生成阶段不再训练；新编码器的对照排名仍按当前统一库重新计算。
