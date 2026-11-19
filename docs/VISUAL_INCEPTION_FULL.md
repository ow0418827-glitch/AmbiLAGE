# 视觉记忆植入完整图片生成接口

本版本按用户于二〇二六年十一月十六日选定的方案恢复检索损失、独立目标概念、五十／三十／二十查询来源及三项联合优化。默认方法仍是 `visual_inception`(视觉记忆植入方法标识)，默认目标模式为 `retrieval_semantic_perceptual`(检索、语义与感知联合目标)。早期无检索版本保留为显式模式，二者不能混用完成凭证。

二〇二六年十一月十七日补入困难负例、零点三对比间隔和对抗查询增强，并于十一月十八日修订升级，新配置默认使用增强第三版。下文无间隔的检索公式用于说明历史第二版；增强版在完整保留全库候选分母的同时，于温度缩放后的对数几率空间对三角色困难负例及前五强敌施加零点三平滑对比间隔，并在冻结文本流形子空间内执行球面有界对抗查询增强后与干净查询损失各半混合。完整公式、修正原因和作者未公开细节边界见[增强第三版说明](VISUAL_INCEPTION_ENHANCED.md)。历史版本继续保留原目标身份，不能用旧图片补签增强版凭证。

初次交付为算法、数据导入、改写服务连接和冻结校验接口。二〇二六年十一月十七日已按用户授权实际准备独立公开商品图片、公开描述、稀宇模型改写和真实搜索查询。具体资料、完整组与缺额组见[公开训练资料说明](VISUAL_INCEPTION_PUBLIC_DATA.md)。本次没有执行真实代理模型的图片扰动训练或正式效果评估。

## 联合目标与数据角色

目标概念文字由冻结公开品类和核心特点构成一段正文，独立保存为 `target_concept_text`(目标概念文字)。一百条未来查询保存为独立列表，只用于检索损失，不能代替单个目标概念，也不使用目标商品图片作为额外优化目标。

对每个代理，设当前图片与查询的相似度为 \(s_+\)，其余公开候选的相似度为 \(s_j\)：

\[
L_{\mathrm{ret}}=\frac{1}{|B|}\sum_{q\in B}
\left[\log\left(e^{s_+(q)/\tau}+\sum_j e^{s_j(q)/\tau}\right)-s_+(q)/\tau\right],
\quad \tau=0.07.
\]

\[
L_{\mathrm{sem}}=1-\cos(E(I+\delta),E(T)),\qquad
L=\sum_iw_i(L_{\mathrm{ret},i}+L_{\mathrm{sem},i})+0.1L_{\mathrm{per}}.
\]

分母包含当前对抗图片一次；没有另放一份干净源图片作为固定竞争候选。其他候选图片向量全程冻结。候选键使用图片向量，查询使用文字向量，不借用环境光方法的图文融合定义。动态正例约定是用户针对原文未公开细节选定的实现规则，不能声称作者公开了相同的完整代码。

默认三代理依次为大型图文对比编码器、缩放图文对齐编码器、大写巨型图文对比编码器，配置见 [代理配置](../src/visual/config/surrogates.visual_inception.json)。原始像素方法和环境光协同方法默认代理保持原样。

默认执行一百步、步长二除以二百五十五、最大绝对像素扰动八除以二百五十五。十个随机起点只用于估计“检索加语义”输入梯度的方差及代理权重；随后从干净原图执行一次完整优化，保存最终一步。感知项使用已有真实预训练感知度量；本地微型模式使用明确测试夹具，不能作为正式结果。

## 查询池和公开代理库

每个训练包固定一百条互不重复的文字，来源严格为：

| 来源 | 数量 | 必须保留的证据 |
|---|---:|---|
| 同类公开商品描述 | 50 | 公开来源、版本、原始行、品类和文字摘要 |
| 模型改写 | 30 | 输入描述行、实际服务商、实际模型、参数支持情况和输出摘要 |
| 独立公开搜索查询 | 20 | 真实公开搜索语料、版本、品类和预留划分 |

用户最新要求使用 MiniMax(稀宇模型服务)生成三十条改写，因此采用 `declared_model`(如实声明实际模型的适配协议)，不把结果标成论文指定模型的改写。独立公开搜索查询可来自公开商品搜索语料，但不能把合成文字、商品类别字段或本项目正式询问冒充查询日志。来源数据不足、重复或缺少出处时停止。

优化每一步从固定一百条池中无放回抽取十六条，跨步可重复；池内比例保持五十／三十／二十，不要求每个十六条小批量都精确满足比例。校准批次与每步索引均按种子冻结，并写入完成证据。

接口要求候选库至少包含两张不同的公开图片；本次用户明确选择九类各十个商品，合并成同一个九十商品代理库供九组训练共用。选取排除整个公开实验快照中的二百零七个商品及全部公开源图、目标图，不读取正式共享记忆内容。每个包必须位于项目公开目录之一：`data/public`(公开数据目录)、`resources/public_training`(公开训练资源目录)或 `output/datasets/public`(公开数据准备目录)。来源快照和候选图片都在包内，禁止越界引用。完整包的推荐布局是：

```text
data/public/visual_inception/07_shampoo/
  training_bundle.json
  sources/
  images/
```

其中 `training_bundle.json`(冻结训练包清单)绑定源图、目标图、冻结规范句、单个目标概念、一百条查询、来源快照和候选图片的摘要。训练阶段只消费清单，模型改写放在准备阶段执行。

## 准备接口

准备工具为 `tools/prepare_visual_inception_data.py`(视觉记忆植入数据准备工具)。输入来源快照使用 JSONL(逐行对象表示格式)，每行具有唯一编号；来源信息文件使用 JSON(JavaScript 对象表示法)。

| 输入 | 最小记录内容 |
|---|---|
| 描述快照 | 编号、描述文字、公开品类 |
| 改写快照 | 编号、改写文字、品类、输入描述编号、实际服务商与模型 |
| 预留搜索查询快照 | 编号、查询文字、品类、明确的预留划分 |
| 候选商品快照 | 编号、图片路径、图片摘要、公开品类 |
| 来源信息文件 | 四类数据的类型、划分、公开语料地址、版本及是否合成 |

来源信息可从 [无密钥来源配置示例](../src/visual/config/visual_inception_sources.example.json)复制后填写。示例中的语料名称、版本和地址是占位内容，必须替换为实际数据来源；改写来源的模型和参数支持信息须与生成工具输出一致。该示例本身不是已准备好的真实训练数据。

先检查命令帮助，下面命令不发起服务调用：

```bash
python -B -X utf8 tools/prepare_visual_inception_data.py --help
python -B -X utf8 tools/prepare_visual_inception_data.py paraphrases --help
python -B -X utf8 tools/prepare_visual_inception_data.py freeze --help
```

改写准备的默认行为只打印计划；明确增加生成开关后才从 `MINIMAX_API_KEY`(稀宇服务密钥环境变量)读取密钥并请求服务：

```bash
python -B -X utf8 tools/prepare_visual_inception_data.py paraphrases \
  --descriptions data/public/visual_inception_inputs/descriptions.jsonl \
  --provider minimax --model MiniMax-M3 \
  --base-url https://api.minimax.cn/v1 \
  --api-key-env MINIMAX_API_KEY \
  --output data/public/visual_inception_inputs/paraphrases
```

实际执行改写时，在上述命令追加 `--generate-paraphrases`(执行模型改写开关)。缺少密钥、返回模型身份不符、改写数量不足或文字重复时停止，不保存为合格训练数据。

`MiniMax-M3`(稀宇第三代模型)和三点一代快速模型按[官方接口说明](https://platform.minimax.cn/docs/api-reference/text-openai-api)不支持自定义采样温度。遵照用户最新要求，这些模型的请求不发送温度参数，也不保留“请求温度零点八”的假设记录；只记录参数未发送及已知的服务默认温度。支持温度设置的模型才使用零点八。接口不能将未生效的参数标成原文设置已复现。

冻结包示例：

```bash
python -B -X utf8 tools/prepare_visual_inception_data.py freeze \
  --group 07_shampoo --base resources/evaluation_base \
  --descriptions data/public/visual_inception_inputs/descriptions.jsonl \
  --paraphrases data/public/visual_inception_inputs/paraphrases/model_paraphrases.jsonl \
  --heldout-queries data/public/visual_inception_inputs/heldout_search.jsonl \
  --candidates data/public/visual_inception_inputs/candidates.jsonl \
  --source-metadata data/public/visual_inception_inputs/sources.json \
  --paraphrase-policy declared_model \
  --output data/public/visual_inception/07_shampoo
```

改写输出另有生成来源文件，来源信息中的模型及参数支持声明须与之相同。冻结前核验文字数目、唯一性、品类、来源行、模型身份、划分、图片字节与文件路径；失败不会产生可训练完成包。真实凭据不进入包、错误正文或日志。

## 生成、缓存与恢复

单组生成显式传入对应训练清单：

```bash
python -B -X utf8 tools/run_experiment.py generate \
  --group 07_shampoo --method visual_inception \
  --training-manifest data/public/visual_inception/07_shampoo/training_bundle.json \
  --visual-inception-config src/visual/config/visual_inception.example.json \
  --output output/visual_inception_full --device cuda
```

九组各自准备并绑定训练包后，可指定共同的包根目录：

```bash
python -B -X utf8 tools/run_experiment.py generate \
  --group all --method visual_inception \
  --training-bundles-root data/public/visual_inception \
  --visual-inception-config src/visual/config/visual_inception.example.json \
  --output output/visual_inception_full --device cuda
```

远程长任务按[远程部署规定](REMOTE_DEPLOYMENT.md)脱离伪终端、断开标准输入并持久化日志。全部九张成图完成并冻结后，才使用现有评估流程；本轮没有恢复购物。

固定候选向量按每个代理分别缓存。新版固定缓存身份包含模型快照、真实权重、候选图片字节及顺序、编码及像素预处理实现、设备与数值运行时；组别训练包摘要记录在每次请求的外层证据中。因此在指定同一个缓存目录时，候选及编码身份相同的不同组可以共用向量，同时各自保留查询包绑定。旧缓存按旧版本核验，不改名复用。文件本身另验摘要与单位范数；当前对抗图片不进入固定缓存，优化每步重新计算。换模型、图片、预处理或数值运行时仍会生成新缓存键。

增强配置示例将共同缓存目录设为 `output/cache/visual_inception_fixed_public`(固定公开候选共用缓存目录)，上面的多组命令据此跨组复用。未指定该字段或显式设为空时仍使用各成图目录内的独立缓存，保留历史运行路径；这类独立目录之间不会自动共享文件。

完成凭证绑定联合目标版本、完整查询包、候选缓存、采样索引、实际完成步数及最终图片。已有完成目录禁止覆盖；默认完整目标缺训练包时在装载大模型前停止。早期模式须显式使用 [无检索配置](../src/visual/config/visual_inception.semantic_only.example.json)，旧证据按原身份核验，不能重签为完整版本。

## 专用检索测试接口（支持论文训练抽样默认与零重合可选）

为对齐原论文主实验与跨表述泛化两类评测口径，本方法提供独立的检索评估入口 `tools/evaluate_visual_inception_category.py`（对应模块 `src.experiments.visual_inception_category_evaluation`），支持通过 `--query-suite`（测试环境选择参数）切换以下两套每组二十条、九组共一百八十条的纯文字英文查询集：

1. **`training_oracle`（默认测试环境，对应原论文 Table 1 主实验及 Table 6 `100% Oracle` 口径）**：
   - 冻结位置：`resources/visual_inception_evaluation/training_queries_oracle_v1/`。
   - 构成规则：从每组训练查询池按确定性哈希排序（固定种子 `42`）依照原论文 `50% / 30% / 20%` 比例分层抽取二十条原句：`category_description`（同类商品公开描述）十条、`model_paraphrase`（模型改写查询）六条、`heldout_search_query`（独立真实搜索查询）四条。

2. **`zero_overlap`（可选测试环境，0% 字面重合跨表述泛化测试）**：
   - 冻结位置：`resources/visual_inception_evaluation/category_queries_v1/`。
   - 构成规则：每组二十条与全部训练文本（八百九十一条）及既有特征评估查询规范化精确重合数为零的同分布查询，均匀包含五类问法各四条：

| 问法标识 | 中文说明 | 每组条数 |
|---|---|---:|
| `category_direct` | 类别直接询问 | 4 |
| `purchase_intent` | 购买意图 | 4 |
| `use_case` | 使用场景 | 4 |
| `functional_need` | 功能需求 | 4 |
| `selection_comparison` | 选择比较 | 4 |

两套测试资产在优化与生成期间均处于训练访问隔离边界内，训练过程不可读取；每个组目录下的 `queries.zh.json`（中文对照清单）仅供人工审阅，不参与编码器检索。

运行前提与评估口径：

1. 须先完成全部九组同一完整联合目标版本的图片生成并冻结；历史第二版或增强第三版分别评估，禁止一次九组评估混用目标版本。缺组、测试夹具或早期无检索版本均会被拒绝。
2. 沿用 `resources/evaluation_base/shared_memory/shared_memory.json` 一百八十九条完整共享长期记忆库，并将九组冻结源图同时替换进同一个修改库副本，不建按组隔离小库。
3. 环境光协同 `ambilage`（方法标识）与视觉记忆植入 `visual_inception`（方法标识）默认使用 `image`（纯图像键，图像权重为 1.0）。原始像素方法 `lucid`（方法标识）在通用及单评估器入口默认使用 `joint`（图文各半融合，图像权重为 0.5）。三编码器迁移面板统一默认使用纯图像键。完整流程可在配置文件中设置 `evaluation.key_mode`（评估检索键模式），命令行传入 `--key-mode image`（纯图像键）或 `--key-mode joint`（图文融合键）时以命令行为准。
4. 汇总输出写入 `category_retrieval_summary.json`（检索汇总）与 `REPORT.md`（评估报告），同时报告总体（一百八十条）、按协议分类（`training_oracle` 按 10/6/4 三类训练来源；`zero_overlap` 按五类问法各 4 条）及逐组（每组二十条）的原图与修改图 Top-1（首位）、Top-3（前三）、Top-5（前五）命中数、增量百分点与平均名次改善，并对每组全库候选分数执行独立复算。
5. 明确区分检索覆盖与下游推荐：本接口仅反映检索阶段覆盖情况，不启动购物流程或策略模型；最终商品推荐或购买行为是否受到影响，须在下游推荐/购物阶段单独验证。

常用命令示例（默认 `--query-suite training_oracle` 与 `--key-mode image`；传入 `--query-suite zero_overlap` 可切换零字面重合测试环境，传入 `--key-mode joint` 可切换图文联合记忆键；评估器配置默认来自基础包 `evaluation_encoder.json`，也可通过 `--encoder-config` 显式指定）：

```bash
python -B -X utf8 tools/evaluate_visual_inception_category.py validate-queries
python -B -X utf8 tools/evaluate_visual_inception_category.py validate-queries --query-suite zero_overlap
python -B -X utf8 tools/evaluate_visual_inception_category.py evaluate --images-root output/visual_inception_full --output output/visual_inception_oracle_eval --device cuda --defense-mode none
python -B -X utf8 tools/evaluate_visual_inception_category.py evaluate --key-mode joint --images-root output/visual_inception_full --output output/visual_inception_oracle_joint_eval --device cuda --defense-mode none
python -B -X utf8 tools/evaluate_visual_inception_category.py evaluate --query-suite zero_overlap --images-root output/visual_inception_full --output output/visual_inception_category_eval --device cuda --defense-mode none
python -B -X utf8 tools/evaluate_visual_inception_category.py verify --output output/visual_inception_oracle_eval --device cuda
python -B -X utf8 tools/evaluate_visual_inception_category.py report --output output/visual_inception_oracle_eval
```

## 本地验证范围

本地只使用微型合成输入及显式夹具，验证动态分母的数值与梯度、单个概念和查询分离、配额及来源、缓存失效、接口不泄露密钥、缺配置不调用服务，以及新旧完成凭证。测试通过不能说明真实模型的显存、速度、自然度或攻击效果。
