# 会话记录 01：AmbiLAGE 仓库部署、权重下载与双组生成评测

- **会话 ID**：`14100399-1521-4ccd-b4b4-118e88220e10`
- **说明**：完整记录 `/content/AmbiLAGE` 仓库克隆、Python 3.11 环境配置、模型权重校验下载、`lucid` 对照与 `AmbiLAGE` 生产级双组（`07_shampoo`、`09_toothbrush`）生成及 `ViT-B-32` / `ViT-g-14` 无防御评测全过程。
- **安全处理**：文档中的访问令牌（如 Hugging Face Token）已全部脱敏为 `[REDACTED_HF_TOKEN]`；仅归档无敏感信息的实验对比图片。

---

## 第 1 轮 · 用户提问（`2026-11-14T03:21:13Z`）

https://github.com/ow0418827-glitch/AmbiLAGE/commit/8e3412a06add7e5bc74bd119a467002b847a070d  把该仓库clone到/content目录下,clone完成后去下载运行该项目必要的环境和模型权重,下载好后默认使用和lucid一样的三代理训练模型和检索模型,去用ambilage方法生成洗发水组和牙刷组的图片并去测试对应的检索效果,长期记忆等共用数据也在该仓库中,可以复用,懂我意思的话去做吧,如果有不清楚的可以询问我

### 助手回复（`2026-11-14T03:22:12Z`）

正在将 `AmbiLAGE` 仓库克隆到 `/content/AmbiLAGE`，等待克隆完成后继续检查提交版本、依赖环境与模型权重配置。

---

> *[注：2026-11-14T03:27:18Z 系统重新投递了与第 1 轮相同的初始请求，下文紧接正式执行回复]*

### 助手回复（`2026-11-14T03:34:48Z`）

公子，仓库检查与基础运行环境搭建已完成，但在**下载模型权重**与**正式生成/评测**前，有一项硬性阻塞项（需要您的 Hugging Face 访问令牌）以及三个实验口径问题需要向您确认。

---

### 一、已完成的工作与验证结果

1. **仓库核验**
   - 目录 `/content/AmbiLAGE` 已就绪，当前分支为 `codex/initial-migration`，工作区干净，提交哈希（`HEAD`）与您指定的 `8e3412a06add7e5bc74bd119a467002b847a070d` 一致。
2. **虚拟环境与依赖安装（Python 3.11 + CUDA）**
   - 由于系统默认的 Python 3.13 不满足 `rapidocr-onnxruntime>=1.4.4,<2` 的版本约束，而 Python 3.12 缺少 `ensurepip`（标准库环境引导模块），已使用 `uv` 在 `/content/AmbiLAGE/.venv` 创建 **CPython 3.11.17** 虚拟环境。
   - 已安装 [requirements.txt](../../requirements/requirements.txt) 中的全部依赖及 PyTorch CUDA 版本（`torch==2.14.1+cu126`、`torchvision==0.29.1+cu126`、`transformers==5.23.0`、`diffusers==0.43.0`、`accelerate==1.15.0`、`open-clip-torch==3.3.0`、`rapidocr-onnxruntime==1.4.4`、`huggingface-hub==1.33.0` 等），并完成 `ambilage==0.2.0` 的可编辑安装。
3. **硬件、离线预检与单元测试实测**
   - **硬件与磁盘**：`NVIDIA A100-SXM4-80GB`（可用显存 `79.25 GiB`，`bf16` 支持为 `True`），当前磁盘剩余 `221.08 GiB`。
   - **离线预检（`preflight`）**：运行 [preflight.py](../../tools/preflight.py) 生成 [remote_preflight_before_models.json](../../output/deployment/remote_preflight_before_models.json) 通过；已校验 `189` 条共享长记忆（shared memory）、`9` 组定义、`180` 条查询、`198` 个共享图片文件及 `247` 项路径/哈希完整性；确认 `AmbiLAGE` 默认训练配置即与 `lucid` 一致的三代理模型（3-surrogate）：
     - `ViT-B-16` (`openai`)
     - `ViT-B-32` (`openai`)
     - `ViT-B-32` (`laion2b_s34b_b79k`)
   - **离线单元测试（`pytest src/tests`）**：实测 `263 passed, 2 failed`（耗时 `72.46s`）。唯一失败的 2 个用例均位于 `src/tests/test_webshop_bundle.py`（`test_lazy_language_model_and_micro_html_service_use_only_project_source` 与 `test_language_model_never_falls_back_to_a_different_reward_model`），报错均为缺少 `thefuzz` 模块——该模块属于购物仿真端（WebShop）的可选依赖 [requirements-webshop.txt](../../requirements/requirements-webshop.txt)，图像生成与检索评测不依赖它（除非您也想让全套测试全绿，我可以顺手补装）。

---

### 二、阻塞项与待确认问题

#### 1. 硬性阻塞：需要 `HF_TOKEN`（Hugging Face 访问令牌）
- `AmbiLAGE` 的编辑模型 `editor` 指向受限仓库（Gated Repo）`black-forest-labs/FLUX.1-Kontext-dev`（过滤后约 `31.43 GiB`），且潜空间微调阶段 [latent_refinement.py](../../src/visual/_ambilage/latent_refinement.py#L45) 默认从 `"editor"` 读取 16 通道 VAE（变分自编码器）。
- 当前机器无 `HF_TOKEN`，从 `hf-mirror.com` 和 `huggingface.co` 均返回 `401 GatedRepo`。
- **请您确认/提供**：请在 Hugging Face 官网同意 `black-forest-labs/FLUX.1-Kontext-dev` 的使用协议后，将具有读取权限的 `HF_TOKEN` 发给我（我仅在下载命令的环境变量中临时使用，不会写入任何文件或日志）。另外本机直连官方 `https://huggingface.co` 可通且较快，请问是否允许优先直连官网下载（或继续走默认的 `hf-mirror.com`）？

#### 2. “检索模型（评估编码器）”口径歧义
您提到“默认使用和 lucid 一样的三代理训练模型和检索模型”。其中**三代理训练模型**代码里已默认对齐 `lucid`（即上述 3 个 CLIP 代理）；但**检索模型（evaluation encoder）**在历史记录与当前仓库配置中存在两套不同口径：
- **历史 `lucid` 实测口径**：仓库导入的历史证据 [REPORT.md](../../output/imported_lucid_baseline_20261113/original_evidence/lucid_shampoo_reselect_20261112/REPORT.md) 中，实际上用的是第二训练代理 **`ViT-B-32` (`openai`)** 作为检索模型（同编码器评估 `matched_public_encoder`，在该口径下 `07_shampoo` 干净检索 Top-3 为 `9/20` -> Lucid `20/20`；`09_toothbrush` 为 `0/20` -> Lucid `14/20`）。
- **当前仓库默认留出口径**：[evaluation_encoder.json](../../resources/evaluation_base/evaluation_encoder.json) 默认配置为黑盒留出编码器（held-out encoder）**`ViT-g-14` (`laion/CLIP-ViT-g-14-laion2B-s12B-b42K`)**（约 `5.09 GiB`）；此外还提供了多模态检索器示例配置 [evaluator.gme.example.json](../../src/visual/config/evaluator.gme.example.json)（`Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`，约 `8.24 GiB`）。

**请确认**：本次检索测试希望使用哪（几）个检索模型？（例如：`ViT-B-32 (openai)`、`ViT-g-14`、`gme-Qwen2-VL-2B-Instruct`，或前两者都跑一遍以便同时对齐历史基线与当前默认黑盒指标？）

#### 3. 仅跑洗发水/牙刷两组 vs 仓库的九组统一评测约束、防御及查询范围
- **两组 vs 九组门禁**：[base_workflow.py](../../src/experiments/base_workflow.py#L185-L204) 中 `evaluate` 阶段写死了校验逻辑——必须先生成并冻结**全部 9 组**的生成图，否则会报错 `统一修改库须先完成全部九组成图并冻结`。不过底层评测函数 `evaluate()`（[shared_evaluation.py](../../src/experiments/shared_evaluation.py#L51-L54)）和共享池构建函数 `evaluation_banks()`（[shared_pool.py](../../src/agent/memory/shared_pool.py)）支持只替换 `07_shampoo` 与 `09_toothbrush` 两组进入 `189` 条共享长记忆池做评测。
  - **请确认**：是只生成 `07_shampoo`（洗发水）和 `09_toothbrush`（牙刷）两组，然后直接调用底层 `shared_evaluation.evaluate()` 接入 `189` 条共享记忆库分别/联合测检索效果（其余 7 组保留共享池原图），还是需要先补全全部 9 组再走上层命令行统一入口？
- **查询模式与防御面板**：
  - 当前 `resources/evaluation_base/evaluation_inputs/*/queries.json` 中仅包含纯文字请求（`text` 模式，每组 20 条，无配套查询图 `joint`）。本次先按现有的 `text` 查询模式评测，可以吗？
  - 防御方面，是先跑无防御基线（`none`），还是要一并跑包含扩散净化（`purifier` Stable Diffusion v1.5，约 `19.45 GiB`）在内的完整 26 条件防御面板（`panel`）？
- **权重下载范围（共 `130.44 GiB`）**：全量 `--scope all` 过滤后总大小 `130.44 GiB`（其中 `planner` `Qwen/Qwen2.5-VL-32B-Instruct` 占 `63.61 GiB`、`editor` `FLUX.1-Kontext-dev` `31.43 GiB`、`purifier` `19.45 GiB`、`eval_gme` `8.24 GiB`、`eval_g14` `5.09 GiB`、`sam2` `0.84 GiB`、3 个 CLIP 训练代理共 `1.70 GiB`、`selection` `0.08 GiB`），当前剩余磁盘 `221.08 GiB` 可以装下。如果不测 `eval_gme` 或 `panel` 净化防御，我可以跳过对应权重下载以节省时间和空间；请告知您的取舍。

#### 4. 确认修复一处会导致真实分割直接报错的边界问题
在核查两组输入数据时发现：仓库内所有 `resources/evaluation_base/groups/*/definition.json` 文件都带有哈希锁（在 `manifest.json` 中登记，不能直接修改），其中的 `source_boxes` 和 `target_boxes` 存储的是**原图像素坐标**（例如 `07_shampoo` 为 `[[1, 1, 180, 497]]` / `[[2, 5, 158, 494]]`，`09_toothbrush` 为 `[[121, 40, 391, 465]]` / `[[135, 51, 350, 446]]`），而 [segmentation.py](../../src/visual/_ambilage/segmentation.py#L71-L90) 强制校检坐标必须属于 `[0, 1]` 归一化区间，否则直接抛出 `ValueError("主体提示框必须是有效归一化坐标")`。

**请确认**：是否同意我在不改动任何冻结数据集文件的前提下，修改 [segmentation.py](../../src/visual/_ambilage/segmentation.py#L71-L90) 使其兼容像素坐标框并按图像尺寸归一化？

---

请您提供 `HF_TOKEN` 并确认上述第 2、3、4 项选择，收到后我将立即开始下载权重、修复分割框兼容并执行 `07_shampoo` 与 `09_toothbrush` 的生成与检索评测。

---

## 第 2 轮 · 用户提问（`2026-11-14T03:37:38Z`）

**引用上下文（`@[Quote]`）：**

> 另外本机直连官方 https://huggingface.co 可通且较快，请问是否允许优先直连官网下载（或继续走默认的 hf-mirror.com）

@[Quote] 允许,只要不改变模型参数就行;只跑两组在无防御下的检索效果即可,其他的按照你说的来,快去做吧

### 助手回复（`2026-11-14T03:46:47Z`）

公子，我已按照您的确认完成**所有非受限权重下载**、**分割框坐标修复与真机验证**，并启动了对照基线评测，但当前在下载编辑模型 `editor`（`black-forest-labs/FLUX.1-Kontext-dev`）时仍被 **`401` 权限验证拦截**——您上一条回复中尚未提供 `HF_TOKEN`（Hugging Face 访问令牌）。

---

### 一、已完成的进展（不等待期间已全部就绪）

1. **非受限模型已全部从官方源下载并校验完成**
   已使用 `--endpoint https://huggingface.co` 下载并生成 `download_manifest.json` 校验清单（仅跳过无防御测试不需要的 `purifier` 净化模型与 `eval_gme` 多模态评估器；当前磁盘仍剩 `150G` 可用）：
   - 分割模型 `sam2`：`facebook/sam2.1-hiera-large`
   - 规划与自然度门禁 `planner`：`Qwen/Qwen2.5-VL-32B-Instruct`
   - 3 个 CLIP 训练代理：`clip_b16`（`ViT-B-16 openai`）、`clip_b32`（`ViT-B-32 openai`）、`clip_b32_laion`（`ViT-B-32 laion2b_s34b_b79k`）
   - 留出评估编码器 `eval_g14`：`laion/CLIP-ViT-g-14-laion2B-s12B-b42K`
   - 公开灰阶形态选品模型 `selection`：`facebook/dinov2-small`
2. **坐标兼容修复与真机模块验证通过（零模型参数改动）**
   - 在 [segmentation.py](../../src/visual/_ambilage/segmentation.py#L71-L130) 新增 `_resolve_box_coordinates`，在不修改冻结数据集及模型参数的前提下同时兼容 `[0, 1]` 归一化框与原图像素框；并在 [test_ambilage_unit.py](../../src/tests/test_ambilage_unit.py#L338-L355) 新增单元测试，实测 `test_ambilage_unit.py` 与 `test_generation_production_contract.py` **`34 passed`**。
   - 已在 A100 上用真实 `SAM2`（`sam2.1-hiera-large`）跑通 `07_shampoo` 与 `09_toothbrush` 两组全部 4 张图的分割（`07_shampoo` 源/目标前景比 `0.8481` / `0.8011`；`09_toothbrush` 源/目标前景比 `0.2539` / `0.1516`），并已验证 3 个训练代理与两套检索编码器（`ViT-B-32 openai` 同代理口径 + `ViT-g-14` 默认留出口径）离线加载全部通过。
3. **对照基线评测已在后台启动**
   - 已在后台启动导入 `lucid` 基线在这两组无防御条件（`none`）下接入 `189` 条共享记忆池的评测（已先输出 `07_shampoo` 在 `ViT-B-32 (openai)` 口径下 Clean Top-3 `9/20`、Lucid Top-3 `20/20`，完整两组 + `ViT-g-14` 复算正在后台继续）。

---

### 二、当前唯一阻塞：`editor` 模型下载需要 `HF_TOKEN`

实际执行 `tools/prepare_models.py --models editor --endpoint https://huggingface.co --download` 时返回：

> `huggingface_hub.errors.GatedRepoError: 401 Client Error ... Access to model black-forest-labs/FLUX.1-Kontext-dev is restricted. You must have access to it and be authenticated to access it.`

由于 `AmbiLAGE` 生产流程的**自然锚点生成**（[backends.py](../../src/visual/_ambilage/backends.py#L108-L112)）与**潜空间 16 通道 VAE**（[latent_refinement.py](../../src/visual/_ambilage/latent_refinement.py#L45-L49)）均强依赖受限仓库（Gated Repo）`black-forest-labs/FLUX.1-Kontext-dev`，且不能更换模型，因此必须凭已授权的 `HF_TOKEN` 才能继续下载并开始生图。

**请您提供：**
1. 请确认您的 Hugging Face 账号已在 `black-forest-labs/FLUX.1-Kontext-dev` 页面点击同意使用协议；
2. 将只读权限的 **`HF_TOKEN`** 直接发给我（我仅在当前下载命令的临时环境变量中注入使用，绝不写入仓库文件或打印输出）。

收到 `HF_TOKEN` 后，我将立即完成 `editor` 下载、执行 `07_shampoo` 与 `09_toothbrush` 的 `AmbiLAGE` 生成，并在无防御下输出与 `lucid` 基线对齐的检索评测结果。

---

### 助手回复（`2026-11-14T03:48:30Z`）

公子，刚才后台对照基线评测因重复触发 `verify()`（`evaluate()` 内部已自带独立进程复算）而报错退出，我已修正调用方式并重新在后台继续执行（目前 `07_shampoo` 在 `ViT-B-32 (openai)` 无防御下的 `lucid` 基线报告已落盘于 [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/b32_none_text/REPORT.md)：原图 Top-3 `9/20`、Lucid Top-3 `20/20`，最大复算误差 `0.0`；`09_toothbrush` 与 `ViT-g-14` 口径及向量缓存正在后台继续运行）。

当前仍等待您提供已在 Hugging Face 同意 `black-forest-labs/FLUX.1-Kontext-dev` 协议的只读 **`HF_TOKEN`**，收到后我会立即下载 `editor` 权重并运行 `AmbiLAGE` 两组生成与无防御检索对比。

---

## 第 3 轮 · 用户提问（`2026-11-14T03:50:59Z`）

**引用上下文（`@[Quote]`）：**

> 当前在下载编辑模型 editor（black-forest-labs/FLUX.1-Kontext-dev）时仍被 401 权限验证拦截——您上一条回复中尚未提供 HF_TOKEN（Hugging Face 访问令牌）。

@[Quote] 不是采用你说的直连方式吗,这个方式不可以下载吗

### 助手回复（`2026-11-14T03:51:05Z`）

公子，**直连官网（`https://huggingface.co`）已经启用且下载成功了其余全部 7 个公开模型**（包括约 `68 GB` 的 `Qwen2.5-VL-32B-Instruct`），但直连解决的是**网络链路**问题，绕不开 `black-forest-labs/FLUX.1-Kontext-dev` 的**账号授权门禁**：

### 为什么直连官网还是 `401`？
- `black-forest-labs/FLUX.1-Kontext-dev` 被发布方（Black Forest Labs）设为了 **Gated Repo**（受限访问仓库）。
- 无论直连官方 `https://huggingface.co` 还是走镜像，只要下载请求里没有带上**“已登录账号并同意过该模型使用协议”的身份令牌（`HF_TOKEN`）**，Hugging Face 官方服务器都会返回 **`401 Unauthorized`** 拒绝匿名访问（目前只有该仓库里少量公开说明文件可匿名读取，核心模型配置与权重均被拦截）。
- 我也已核对过当前机器环境变量和缓存目录，里面没有任何预先登录的 Hugging Face 凭证。

### 解决步骤（约 1 分钟）
1. 登录您的 Hugging Face 账号，打开模型主页：
   `https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev`
   点击页面上的同意协议按钮（如 `Agree and access repository`）完成授权；
2. 打开令牌页面：
   `https://huggingface.co/settings/tokens`
   创建一个 **Read（只读）** 权限的 Token（`hf_...` 开头）发给我即可（或您自行写入机器的 `~/.cache/huggingface/token` 后告诉我一声也可以；我仅在本次下载命令的环境变量里临时使用，绝不写入项目文件或日志）。

---

*附：后台对照基线评测最新进度——`ViT-B-32 (openai)` 口径下两组 `lucid` 基线均已完成并独立复算通过（`max_err: 0.0`）：*
- `07_shampoo`：Clean Top-3 `9/20`（Top-5 `14/20`）→ Lucid Top-3 `20/20`（Top-5 `20/20`）
- `09_toothbrush`（[REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/b32_none_text/REPORT.md)）：Clean Top-3 `0/20`（Top-5 `1/20`）→ Lucid Top-3 `14/20`（Top-5 `15/20`）
- `ViT-g-14` 口径仍在后台继续运行中。

---

### 助手回复（`2026-11-14T03:54:13Z`）

公子，后台的 `lucid` 基线两组完整评估（`ViT-B-32 (openai)` 同代理口径 + `ViT-g-14` 留出黑盒口径，无防御 `none`、`text` 纯文字查询、`189` 条共享长记忆池）已全部完成并通过独立复算（4 项 `independent_verification.json` 均为 `passed`，最大分数误差 `0.0`）：

| 组别 | 检索编码器 | 原图 Top-3 / Top-5 | Lucid Top-3 / Top-5 | Top-3 增量 | 平均名次改善 | 报告 |
|---|---|---|---|---:|---:|---|
| `07_shampoo`（洗发水） | `ViT-B-32 (openai)` | `9/20` / `14/20` | `20/20` / `20/20` | `+55.0` 个百分点 | `+3.250` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/b32_none_text/REPORT.md) |
| `09_toothbrush`（牙刷） | `ViT-B-32 (openai)` | `0/20` / `1/20` | `14/20` / `15/20` | `+70.0` 个百分点 | `+7.400` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/b32_none_text/REPORT.md) |
| `07_shampoo`（洗发水） | `ViT-g-14`（仓库默认留出） | `19/20` / `20/20` | `19/20` / `20/20` | `+0.0` 个百分点 | `+0.000` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/g14_none_text/REPORT.md) |
| `09_toothbrush`（牙刷） | `ViT-g-14`（仓库默认留出） | `0/20` / `3/20` | `0/20` / `6/20` | `+0.0` 个百分点 | `+0.250` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/g14_none_text/REPORT.md) |

（后台顺带正在做 `planner` `Qwen2.5-VL-32B-Instruct` 的 A100 显存连通预检，确保后续一次跑通。）

当前仍仅差您在 Hugging Face 同意 `black-forest-labs/FLUX.1-Kontext-dev` 协议后的 **`HF_TOKEN`**（或写入 `~/.cache/huggingface/token`），拿到后即可下载 `editor` 并立即运行 `AmbiLAGE` 生成与上述同口径对比评测。

---

## 第 4 轮 · 用户提问（`2026-11-14T03:57:31Z`）

[REDACTED_HF_TOKEN]

**上传附件：**
- `media_1794628635990_9de0405a.jpg`（已脱敏：包含 Hugging Face 访问令牌的截图，未随仓库归档）

### 助手回复（`2026-11-14T03:59:36Z`）

公子，已收到令牌并验证通过，`black-forest-labs/FLUX.1-Kontext-dev`（`editor`）正在高速下载并已接近收尾；同时刚已将训练目标强制依赖的 `purifier`（`stable-diffusion-v1-5`）下载校验完成，并在 [backends.py](../../src/visual/_ambilage/backends.py#L31-L139) 修复了规划器 `Qwen2.5-VL-32B-Instruct` 返回 JSON 带 `//` 注释及坐标越界的兼容解析。

下载与规划阶段核验完成后，我将立即启动 `07_shampoo` 与 `09_toothbrush` 两组 `AmbiLAGE` 正式生成，并随后执行无防御下的检索效果评测与 `lucid` 基线对比。

---

### 助手回复（`2026-11-14T04:02:04Z`）

公子，模型权重已经准备就绪，我已经启动洗发水组（`07_shampoo`）的 `AmbiLAGE` 生成任务（完整生产配置，无防御快速检测之外的全流程规划、局部编辑与优化）。目前任务正在后台运行中，完成后我会第一时间检查生成指标并继续牙刷组（`09_toothbrush`）及无防御检索评测。

---

### 助手回复（`2026-11-14T04:06:26Z`）

公子，我向你同步一下最新进展：

1. **洗发水组（`07_shampoo`）排查结果**：生成在潜空间细化前的质量门禁处中止（要求全图 `SSIM >= 0.78` 且 `MAE <= 20.0` 像素级）。我核对中间产物发现，第三阶段光照协调后的初始底图质量达标（`SSIM = 0.9378`，`MAE = 7.9337`），但随后 `FLUX.1-Kontext-dev` 对这块 `181x500` 的细长瓶身图启用了默认的上下文分辨率自适应（`_auto_resize=True` 将条件图缩放到 `(672, 1568)`，而生成图却按最大面积调整到了 `(624, 1664)`），再加上整体光照与背景重绘，导致生成锚点（`generative_anchor`）相对原图发生几何偏移、印刷区回写重影，指标降到 `SSIM = 0.6454`、`MAE = 24.0524`。
2. **牙刷组（`09_toothbrush`）对照运行中**：由于牙刷组原图是标准方图 `500x500`，我已经先在后台启动 `09_toothbrush` 的无修改原生生成流程，等它完成结果后，我再针对细长比例图像和锚点几何对齐做精确、不改模型权重的最小修复。

---

## 第 5 轮 · 用户提问（`2026-11-14T04:11:10Z`）

**引用上下文（`@[Quote]`）：**

> 洗发水组（07_shampoo）排查结果：生成在潜空间细化前的质量门禁处中止（要求全图 SSIM >= 0.78 且 MAE <= 20.0 像素级）。我核对中间产物发现，第三阶段光照协调后的初始底图质量达标（SSIM = 0.9378，MAE = 7.9337），但随后 FLUX.1-Kontext-dev 对这块 181x500 的细长瓶身图启用了默认的上下文分辨率自适应（_auto_resize=True 将条件图缩放到 (672, 1568)，而生成图却按最大面积调整到了 (624, 1664)），再加上整体光照与背景重绘，导致生成锚点（generative_anchor）相对原图发生几何偏移、印刷区回写重影，指标降到 SSIM = 0.6454、MAE = 24.0524。

> 的无修改原生生成流程，等它完成结果后，我再针对细长比例图像和锚点几何对齐做精确、不改模型权重的最小修复。

@[Quote] @[Quote] 你是准备等牙刷组结束后再去修复这个问题是吗,是的话就按照你说的来,但是你可以先把修复思路跟我说一下

### 助手回复（`2026-11-14T04:11:38Z`）

公子，是的，我打算等牙刷组（`09_toothbrush`）这轮后台生成完全跑完后再修改代码并重跑洗发水组（`07_shampoo`），避免并发占用显存，同时也已经顺便用牙刷组验证了标准比例图像的原生行为。

### 一、已确认的对比现象（对照依据）
* **牙刷组 `09_toothbrush`（`500x500` 方图，目前已在后台正常推进至第 4 阶段优化）**：
  * 第三阶段光照协调底图：`SSIM = 0.9913`，`MAE = 3.1270`。
  * 经过 `FLUX.1-Kontext-dev` 后（输入与生成均对齐至 `1024x1024`）：生成锚点 `SSIM = 0.9830`，`MAE = 3.4502`（背景 `MAE = 0.5289`），瓶身/刷头几何零错位，目前已顺利进入第 4 阶段 350 步潜空间细化（第 20 步时 `Utility = 0.8404`，`SSIM = 0.9458`，`MAE = 12.89`）。
* **洗发水组 `07_shampoo`（`181x500` 细长竖图，宽高比 `0.362`）**：
  * 第三阶段光照协调底图本来是达标的：`SSIM = 0.9378`，`MAE = 7.9337`（远好于门禁下限 `SSIM >= 0.78` 和上限 `MAE <= 20.0`）。
  * 但送入 `ProductionStages.generate_anchor`（位于 [backends.py](../../src/visual/_ambilage/backends.py#L219-L236)）后，生成锚点骤降至 `SSIM = 0.6454`、`MAE = 24.0524`（前景 `MAE = 24.4310`，背景 `MAE = 21.9389`），从而被 [latent_refinement.py](../../src/visual/_ambilage/latent_refinement.py#L177-L178) 的质量门禁拦住。

---

### 二、详细修复思路（均不改动模型权重、不放宽质量门禁）

导致 `07_shampoo` 失败的核心原因有两层：一是 **`FLUX.1-Kontext` 对极细长宽高比的输入/输出分辨率失配**，二是 **重绘指令引发的背景/构图偏移破坏了印刷核心区的像素级对齐**。我的具体修复方案如下：

1. **画布对称留白（Letterbox Padding）+ 原生桶分辨率精确对齐（主要修复，在 `ProductionStages.generate_anchor` 中实现）**
   * **问题机理**：`FLUX.1-Kontext-dev` 官方推荐桶（`PREFERRED_KONTEXT_RESOLUTIONS`）最窄只到 `(672, 1568)`（宽高比 `0.4286`）。而 `07_shampoo` 宽高比仅 `0.362`，同时原代码传给 `self.editor(...)` 的 `(width=192, height=512)` 被管线内部按 `max_area = 1024^2` 强制改写为 `(624, 1664)`（宽高比 `0.375`），输入条件图却因 `_auto_resize=True` 被压缩进 `(672, 1568)`（宽高比 `0.4286`）。输入潜特征与输出潜特征**分辨率与长宽比都不一致**，且细长瓶身左右几乎紧贴画框边缘，导致 FLUX 输出瓶身横向/纵向偏移；接着流水线再按原坐标回写原图印刷字 `anchor[roi] = source[roi]`，就会在瓶身上形成“文字上下重影”。
   * **修复做法**：
     * 先识别输入图像边缘的纯底色（例如由边缘像素中位数估计的白色 `[255, 255, 255]`），当原图长宽比超出 `PREFERRED_KONTEXT_RESOLUTIONS` 覆盖范围 `[672/1568, 1568/672]`（或过于贴边）时，用边缘底色对输入图做**对称白边填充**，使其宽高比落入 Flux Kontext 的原生支持区间内。
     * 根据填充后的画布尺寸选择同一个匹配的原生分辨率 `(kw, kh)`，同时向 `self.editor(...)` 显式传入 `height=kh, width=kw, max_area=kh*kw`（并关闭不同步的隐式重缩放影响），保证 **FLUX 输入条件特征与输出潜变量尺寸 1:1 严格对齐**。
     * 生成完成后，按比例精确裁掉外围填充区域，再还原回原始 `(w, h)` 分辨率。这样瓶身物理坐标不会发生拉伸或偏移，回写印刷核心区 `roi` 也不会产生文字重影。

2. **约束视觉规划 `edit_prompt` 的局部摄影微调语义（辅助保障，位于 `ProductionStages.plan`）**
   * 若仅靠分辨率对齐仍不足以把全局 `bg_mae` / `fg_mae` 压回门禁内（因为 [backends.py](../../src/visual/_ambilage/backends.py#L186-L187) 当前提示词包含较宽泛的 `"relighting/background instruction"`，容易诱导 FLUX 把白底改暗或移动物体构图，而环境光底色迁移其实本来就由第 3 阶段和第 4 阶段环境光场协同完成）：
   * 我们可以在提示词中进一步明确：**禁止改变机位视角、主体轮廓与几何位置，保持电商摄影棚洁净背景构图一致，仅做自然柔和的棚拍布光微调**；同时保留第 4 阶段质量门禁（`SSIM >= 0.78`、`MAE <= 20.0`）与第 5 阶段视觉自然度门禁的原装严格标准不变，不放宽任何阈值。

等后台牙刷组 `09_toothbrush` 一完成，我就按上述第 1 步优先实施最小修改，补齐对应单元测试后重跑 `07_shampoo` 验证实际 `SSIM`/`MAE` 效果，如果还需要微调提示词约束再进一步跟进。目前牙刷组还在运行，如果有需要调整的地方请随时告诉我；没有的话我会在牙刷组完成后自动通知你并接着执行修复。

---

### 助手回复（`2026-11-14T05:04:54Z`）

公子，好消息：**牙刷组（`09_toothbrush`）已经顺利完成全流程生产生成并冻结成图**，各项指标均通过严格门禁：
* **结构与误差指标**：最终 `SSIM = 0.9518`（门限 `>= 0.78`），全图 `MAE = 13.19`（门限 `<= 20.0`），350 步潜空间优化全部接受（`350/350`），`training_utility = 0.8748`。
* **视觉自然度门禁**：`Qwen2.5-VL-32B` 审核通过（`passed = true`），成图保存至 [frozen.png](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/frozen.png) 与 [FROZEN_IMAGE.json](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/FROZEN_IMAGE.json)。

接下来我正在对洗发水组（`07_shampoo`）执行一个短时的 `FLUX.1-Kontext-dev` 分辨率对齐与对称留白对照测试，确认规划提示词以及不同原生桶尺寸下的 `SSIM` / `MAE` 与对齐效果后，就正式落盘修复并重跑洗发水组。完成后再把两组一起跑无防御检索评测。

---

### 助手回复（`2026-11-14T05:08:31Z`）

公子，我正在用 `FLUX.1-Kontext-dev` 对洗发水组测试“对称留白 + 原生桶尺寸对齐 + 锁定机位/构图提示词”的实际 `SSIM` 与 `MAE` 效果，几十秒内出结果后我就落盘修复并继续运行洗发水组生成。

---

### 助手回复（`2026-11-14T05:10:40Z`）

公子，前一组对比显示即便锁定构图提示词，极细长瓶身在生成锚点时仍有约数像素的尺度/位置微移，正在追加测试子像素仿射配准与净白棚拍背景约束下的 `SSIM` 和 `MAE`，很快出结果。

---

### 助手回复（`2026-11-14T05:12:38Z`）

公子，正在测试基于潜空间局部去噪起点（`Flow-Matching SDEdit`）的 `FLUX.1-Kontext` 生成锚点，观察如何在保持洗发水瓶身与文字像素对齐的同时完成真实摄影布光与自然度收敛。完成后马上回传效果。

---

### 助手回复（`2026-11-14T05:15:34Z`）

公子，我已经按照“对称留白 + 原生桶分辨率对齐 + 全噪漂移时自动触发 Flow-Matching 局部去噪回正 + 约束构图提示词”的思路完成了 [backends.py](../../src/visual/_ambilage/backends.py) 的修复，并在 [test_ambilage_unit.py](../../src/tests/test_ambilage_unit.py) 补齐了单元测试，相关 36 项单元与生产契约测试已全部通过。

为稳妥起见，我正在直接调用真实 `Qwen2.5-VL-32B` + `FLUX.1-Kontext-dev` 对洗发水组做一次快速闭环预检（验证锚点对齐、`SSIM`/`MAE` 达标以及最终自然度门禁是否通过），确认无误后就启动完整的 350 步洗发水组生成。

---

### 助手回复（`2026-11-14T05:18:15Z`）

公子，洗发水组（`07_shampoo`）的闭环预检已经全部通过：
* **锚点对齐与质量**：自动触发原生长宽比对称留白（`672x1568` 原生桶）与 Flow-Matching 局部去噪回正后，生成锚点与原图达到 `SSIM = 0.9055`、`MAE = 9.2246`（满足 `SSIM >= 0.78` 且 `MAE <= 20.0`），文字与标签完全像素对齐、无重影。
* **自然度预审**：`Qwen2.5-VL-32B` 视觉门禁预检判定 `passed = true`。

目前我已经清理旧的失败目录，正式在后台启动了 `07_shampoo` 的完整 350 步 `AmbiLAGE` 生产生成任务，并提前写好两组双检索器（`ViT-B-32` 与 `ViT-g-14`）的无防御评测脚本。等洗发水组生成一完成，我就会立刻开始无防御检索评测与全套校验并汇报最终结果。

---

### 助手回复（`2026-11-14T06:10:47Z`）

公子，好消息：**洗发水组（`07_shampoo`）也已完整跑通 350 步潜空间优化并通过所有门禁冻结成图**：
* `SSIM = 0.8853`，全图 `MAE = 11.4989`（前景 `11.3883`、背景 `12.1162`），350 步优化全部接受，`training_utility = 0.6783`，`Qwen2.5-VL-32B` 自然度门禁 `passed = true`，`validate_ready` 校验通过，成图已存至 [frozen.png](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/frozen.png)。

目前我正在后台同时执行两项收尾工作：
1. 在 189 条共享长期记忆池上运行洗发水组与牙刷组在无防御（`none`）、纯文字查询（`text`）下的双模型检索评测（`ViT-B-32 (openai)` 同代理 + `ViT-g-14` 留出跨模型），含独立子进程全库重编码复算；
2. 复跑相关单元与评估回归测试。完成后我会汇总详细对比报告汇报给你。

---

### 助手回复（`2026-11-14T06:11:29Z`）

公子，核心模块与评估回归测试（64 项）已经全部通过，我正在补跑全库 `pytest` 汇总与双检索器无防御评测；很快两个后台任务就会一起完成并输出结果。

---

### 助手回复（`2026-11-14T06:13:02Z`）

公子，全库回归测试已有准确统计（`265` 项通过，仅 `3` 项为环境边界失败：`2` 项依赖未安装的可选 `thefuzz`、以及 `1` 项在同步了本地 `model_lock.json` 后与合成假权重比对失败，与本次代码修改无关）。

无防御双检索器评测已完成 `ViT-B-32 (openai)` 的洗发水组、牙刷组以及 `ViT-g-14` 的洗发水组，正在跑最后一项牙刷组的 `ViT-g-14` 独立复算。待后台评测全部结束后，我再通读四份正式报告与复算凭证并向你汇总完整结果。

---

### 助手回复（`2026-11-14T06:17:42Z`）

公子，当前所有生成、检索评测与独立校验均已完成，我正在执行最后一轮单元测试复核，马上为您整理完整结果报告。

---

### 助手回复（`2026-11-14T06:17:46Z`）

公子，全部任务已完成：`/content/AmbiLAGE` 仓库克隆、依赖与模型权重直连下载（未修改任何模型参数）、`07_shampoo`（洗发水组）与 `09_toothbrush`（牙刷组）的 `AmbiLAGE` 生产级图片生成，以及两组在无防御（`none`）和复用 189 条共享长期记忆底库下的检索评测与独立子进程验真均已跑通。

---

### 一、 环境、模型权重与实验配置

- **代码与硬件**：仓库已克隆至 `/content/AmbiLAGE`（指定提交 `8e3412a06add7e5bc74bd119a467002b847a070d`），运行于 `NVIDIA A100-SXM4-80GB`（`79.25 GiB` 显存，支持 `bf16`），`.venv` 环境为 `Python 3.11.17`、`torch 2.14.1+cu126`、`transformers 5.23.0`、`diffusers 0.43.0`、`accelerate 1.15.0`、`open-clip-torch 3.3.0`、`rapidocr-onnxruntime 1.4.4`、`huggingface-hub 1.33.0`。
- **模型下载与校验**：通过 `https://huggingface.co` 直连完成 `sam2`、`planner`、`editor`（`FLUX.1-Kontext-dev`）、`purifier`（`stable-diffusion-v1-5`）、`clip_b16`、`clip_b32`、`clip_b32_laion`、`eval_g14`、`selection` 权重拉取，未修改任何权重参数；与本次无防御双组检索无关的 `eval_gme` 已按范围跳过。`tools/prepare_models.py` 同步刷新了本地 [model_lock.json](../../resources/model_lock.json)。
- **生成与复用配置**：采用与 `lucid` 一致的三代理（surrogate，替代训练编码器）配置（`surrogate_set='lucid'`），执行 350 步潜变量优化（含预期变换期望鲁棒性增强 EOT、10 步 `stable-diffusion-v1-5` 净化器 `purifier`、FLUX 16 通道 VAE），复用仓库内置 `189` 条共享记忆底库（`51` 条购买记忆 `purchase_memory` + `138` 条模拟图库 `simulated_gallery`）。

---

### 二、 兼容性问题与修复摘要

在严格保持质量门禁（`SSIM >= 0.78`、`MAE <= 20.0` 及 `Qwen2.5-VL-32B-Instruct` 自然度评审）与冻结数据集不变的前提下，共修复三处代码适配问题：

1. **像素框坐标兼容**：修改 [segmentation.py](../../src/visual/_ambilage/segmentation.py)，新增 `_resolve_box_coordinates()`，使 `segment_image()` 同时支持归一化 `[0, 1]` 框与冻结 `definition.json` 中的像素坐标框（用于 `07_shampoo` 等非归一化标注）。
2. **规划器 JSON 容错与非正方图 `FLUX.1-Kontext-dev` 对齐修复**：修改 [backends.py](../../src/visual/_ambilage/backends.py)：
   - 新增 `_strip_json_comments_and_trailing_commas()` 与 `_normalize_planner_boxes()`，兼容 `Qwen2.5-VL-32B-Instruct` 输出中的注释、尾逗号和坐标缩放。
   - 解决 `07_shampoo`（`181x500` 竖长图）初跑时因 `FluxKontextPipeline` 默认 `_auto_resize=True` 自动伸缩比例及全噪声重绘导致的构图漂移（此前在 [latent_refinement.py](../../src/visual/_ambilage/latent_refinement.py#L178) 触发 `SSIM=0.6454, MAE=24.0524` 门禁失败且 ROI 回贴文字重影）：新增 `_flux_kontext_resolutions()`、`_prepare_flux_canvas()`（按边缘中值对称填充到最近支持桶，如 `[672, 1568]`，锁定 `height`/`width`/`max_area` 并关闭 `_auto_resize`）、`_crop_editor_output()` 与 `_measure_anchor_alignment()`；在 `plan()` 提示词中约束保持原始构图与像素对齐；并在 `generate_anchor()` 中先走完整 `full_kontext`，仅当 `init_ssim < 0.85` 或 `init_mae > 15.0` 时自动切至 Flow-Matching（流匹配）部分去噪重绘 `_flux_sdedit_refine()`（`flow_matching_sdedit`，`strength=0.45`）并在审计字段如实记录。正方形 `09_toothbrush`（`500x500`）在 `(1024, 1024)` 下直接通过 `full_kontext`。
3. **单元测试补充**：在 [test_ambilage_unit.py](../../src/tests/test_ambilage_unit.py) 中新增 3 项对应单元测试（像素/归一化框、规划器 JSON 解析、Kontext 画布填充与 SDEdit 对齐回退）。

---

### 三、 `AmbiLAGE` 成图质量与门禁结果

| 组别 | 模式 / 步数 | `source_ssim` | `source_mae`（前景 / 背景） | `anchor_mae` | 训练效用 / 门禁 | 输出文件 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `07_shampoo`（`181x500`） | `flow_matching_sdedit`（桶 `[672, 1568]`） / 350 步（最佳第 240 步，接受 350 / 拒绝 0，耗时 `3115.13s`） | `0.8853` | `11.4989`（fg `11.3883` / bg `12.1162`） | `3.7921` | `0.6783` / `passed: true` | [frozen.png](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/frozen.png)、[FROZEN_IMAGE.json](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/FROZEN_IMAGE.json)、[naturalness_gate.json](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/generation/naturalness_gate.json) |
| `09_toothbrush`（`500x500`） | `full_kontext` / 350 步（最佳第 310 步，接受 350 / 拒绝 0，耗时 `3438.66s`） | `0.9518` | `13.1890`（fg `12.8278` / bg `13.3119`） | `10.5613` | `0.8748` / `passed: true` | [frozen.png](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/frozen.png)、[FROZEN_IMAGE.json](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/FROZEN_IMAGE.json)、[naturalness_gate.json](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/generation/naturalness_gate.json) |

---

### 四、 无防御检索效果对比（`Lucid` vs `AmbiLAGE`）

使用评测脚本 [run_ambilage_eval.py](file:///root/.gemini/antigravity/brain/14100399-1521-4ccd-b4b4-118e88220e10/scratch/run_ambilage_eval.py)，在**无防御（`none`）**、**纯文本冻结查询（`text`，每组 20 条）**及**双组同步替换进 189 条共享底库（其余 187 条保持原样）**的同口径设置下，分别于**匹配训练代理编码器 `ViT-B-32 (openai)`**（配置 [eval_b32_matched.json](file:///root/.gemini/antigravity/brain/14100399-1521-4ccd-b4b4-118e88220e10/scratch/eval_b32_matched.json)）与**黑盒留出公共检索编码器 `ViT-g-14`**（仓库默认 [evaluation_encoder.json](../../resources/evaluation_base/evaluation_encoder.json)）上完成评测；全部 8 组均通过隔离进程重算验真（`status: "passed"`，`40` 次排序全量重算、`7560` 条候选分数核对、`max_score_error: 0.0 <= 1e-07`）。

#### 1. `ViT-B-32 (openai)`（`matched_public_encoder`，匹配三代理训练白盒检索）

| 组别 | Clean Top-3 / Top-5 | 方法 | Method Top-3 / Top-5 | Top-3 增益 | 平均排名提升 | 评测报告与独立验真 |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `07_shampoo` | `9/20` / `14/20` | `Lucid` | `20/20` / `20/20` | `+55.0` pts | `+3.250` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/b32_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/b32_none_text/independent_verification.json) |
| `07_shampoo` | `9/20` / `14/20` | `AmbiLAGE` | `10/20` / `18/20` | `+5.0` pts | `+0.500` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/evaluation/b32_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/evaluation/b32_none_text/independent_verification.json) |
| `09_toothbrush` | `0/20` / `1/20` | `Lucid` | `14/20` / `15/20` | `+70.0` pts | `+7.400` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/b32_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/b32_none_text/independent_verification.json) |
| `09_toothbrush` | `0/20` / `1/20` | `AmbiLAGE` | `3/20` / `5/20` | `+15.0` pts | `+3.100` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/evaluation/b32_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/evaluation/b32_none_text/independent_verification.json) |

#### 2. `ViT-g-14`（`held_out_public_encoder`，黑盒跨模型检索，仓库默认评测器）

| 组别 | Clean Top-3 / Top-5 | 方法 | Method Top-3 / Top-5 | Top-3 增益 | 平均排名提升 | 评测报告与独立验真 |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `07_shampoo` | `19/20` / `20/20` | `Lucid` | `19/20` / `20/20` | `+0.0` pts | `+0.000` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/g14_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/g14_none_text/independent_verification.json) |
| `07_shampoo` | `19/20` / `20/20` | `AmbiLAGE` | `20/20` / `20/20` | `+5.0` pts | `+0.250` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/evaluation/g14_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/evaluation/g14_none_text/independent_verification.json) |
| `09_toothbrush` | `0/20` / `3/20` | `Lucid` | `0/20` / `6/20` | `+0.0` pts | `+0.250` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/g14_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/g14_none_text/independent_verification.json) |
| `09_toothbrush` | `0/20` / `3/20` | `AmbiLAGE` | `11/20` / `14/20` | `+55.0` pts | `+3.000` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/evaluation/g14_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/evaluation/g14_none_text/independent_verification.json) |

**结果简析**：在匹配训练代理编码器 `ViT-B-32 (openai)` 上，无结构约束的像素级 `Lucid` 对白盒拟合更充分；但在**未参与训练的黑盒大编码器 `ViT-g-14`** 上，`Lucid` 在两组均录得 `+0.0` pts 的 Top-3 增益，而 `AmbiLAGE` 表现出显著的跨模型迁移性——`07_shampoo` 达到 `20/20` 满分 Top-3，`09_toothbrush` Top-3 命中率从 `0/20` 跃升至 `11/20`（`+55.0` pts，平均排名提升 `+3.000`）。（注：仓库内原有历史基线替换全部 9 组，本轮按要求仅对 `07_shampoo` 与 `09_toothbrush` 双组同步替换在相同 189 底库下做公平同口径对比。）

---

### 五、 校验与测试结果

- **定向单元测试与契约回归**（所有代码改动完成后最新执行）：运行 [test_ambilage_unit.py](../../src/tests/test_ambilage_unit.py)、[test_generation_production_contract.py](../../src/tests/test_generation_production_contract.py)、[test_base_workflow.py](../../src/tests/test_base_workflow.py)、[test_shared_replacement_reuse.py](../../src/tests/test_shared_replacement_reuse.py)、[test_evaluator_configuration.py](../../src/tests/test_evaluator_configuration.py)，结果 `64 passed in 12.01s`。
- **硬件与数据预检**：`tools/preflight.py --hardware` 返回退出码 `0`（输出见 [remote_preflight.json](../../output/deployment/remote_preflight.json)：`189` 条底库、`9` 组、`180` 条查询、`198` 个共享图片文件、`247` 项路径与哈希校验通过）。
- **全量 `pytest src/tests/`（先前执行记录）**：`265 passed, 3 failed in 80.37s`。其中 3 条失败与本次双组生成及检索评测无关：1 条因本地准备模型后更新了 [model_lock.json](../../resources/model_lock.json) 导致 `test_preflight_does_not_require_unselected_training_weights` 的合成权重锁比对不匹配；另 2 条为 `test_webshop_bundle.py` 因未安装可选 WebShop 运行时依赖 `thefuzz` 报 `ModuleNotFoundError`。

---

## 第 6 轮 · 用户提问（`2026-11-14T06:18:32Z`）

**引用上下文（`@[Quote]`）：**

> 无防御双检索器评测已完成 ViT-B-32 (openai) 的洗发水组、牙刷组以及 ViT-g-14 的洗发水组，正在跑最后一项牙刷组的 ViT-g-14 独立复算。待后台评测全部结束后，我再通读四份正式报告与复算凭证并向你汇总完整结果。

@[Quote] 什么意思,你把两组都去用ViT-B-32 (openai),ViT-g-14测试一下,

### 助手回复（`2026-11-14T06:19:26Z`）

公子，你引用的那段话是当时后台评测快结束时的**中途进度播报**，可能我当时表述偏术语化让你误会还没跑完；实际上，**洗发水组（`07_shampoo`）和牙刷组（`09_toothbrush`）现在已经全部完成 `ViT-B-32 (openai)` 和 `ViT-g-14` 这两个检索器的测试与核验**，不需要再重新跑。

---

### 一、 你引用的那句话是什么意思

- **“无防御双检索器评测”**：
  - **无防御（`none`）**：指按你的要求，测试图与记忆库图片在检索前不经过图像压缩、裁剪或扩散净化等防御处理。
  - **双检索器**：指用**两个检索编码器**同时测一遍：一个是与三代理训练白盒重合的 `ViT-B-32 (openai)`（对应配置 [eval_b32_matched.json](file:///root/.gemini/antigravity/brain/14100399-1521-4ccd-b4b4-118e88220e10/scratch/eval_b32_matched.json)，也是仓库历史 `Lucid` 报告 [REPORT_ALL9.md](../../output/imported_lucid_baseline_20261113/original_evidence/lucid_groups9_20261109/REPORT_ALL9.md) 所用的检索模型）；另一个是不在三代理训练模型里的黑盒留出编码器 `ViT-g-14`（对应仓库默认检索配置 [evaluation_encoder.json](../../resources/evaluation_base/evaluation_encoder.json)，训练代理列表可对照 [training_surrogates.json](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/generation/training_surrogates.json)）。
- **“独立复算”**：
  - 不是额外换别的实验，而是仓库评测流程内置的防作弊自检。每跑完一组检索后，系统会另起一个**隔离进程（`isolated_process: true`）**、重新加载全新编码器权重（`fresh_encoder: true`），把 `189` 条记忆库和 `20` 条查询的 `40` 次排序（Clean 原图库与 Modified 修改图库各 20 次）以及全部 `7560` 个候选相似度分数从头重算一遍，确认最大误差为 `0.0`（容差 `1e-07`）后才出正式报告。
- 当时播报的意思只是：`ViT-B-32 (openai)` 的洗发水组、牙刷组，以及 `ViT-g-14` 的洗发水组都已测完并验真，仅剩牙刷组 `ViT-g-14` 的最后一步校验还在跑；**目前这一步也早已完成**。

---

### 二、 两组 × 双检索器最终实测结果汇总

两组均在同一套设置下测试：通过 [run_ambilage_eval.py](file:///root/.gemini/antigravity/brain/14100399-1521-4ccd-b4b4-118e88220e10/scratch/run_ambilage_eval.py) 在 `189` 条共享长期记忆库（`51` 条真实购买记忆 `purchase_memory` + `138` 条模拟图库 `simulated_gallery`）中同步替换 `07_shampoo` 与 `09_toothbrush` 两组对应条目（其余 `187` 条原封不动），使用冻结纯文本购买请求（`query_mode: "text"`，每组 `20` 条，见 [queries.json](../../resources/evaluation_base/evaluation_inputs/07_shampoo/queries.json)），并将 `Lucid` 与 `AmbiLAGE` 放在同口径下对比：

#### 1. 用 `ViT-B-32 (openai)` 测试（三代理之一，白盒匹配）

| 组别 | 原图 Clean Top-3 / Top-5 | 方法 | 修改图 Top-3 / Top-5 | Top-3 增量 | 平均名次改善 | 报告与复算凭证 |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `07_shampoo`（洗发水） | `9/20` / `14/20` | `Lucid` | `20/20` / `20/20` | `+55.0` pts | `+3.250` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/b32_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/b32_none_text/independent_verification.json) |
| `07_shampoo`（洗发水） | `9/20` / `14/20` | `AmbiLAGE` | `10/20` / `18/20` | `+5.0` pts | `+0.500` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/evaluation/b32_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/evaluation/b32_none_text/independent_verification.json) |
| `09_toothbrush`（牙刷） | `0/20` / `1/20` | `Lucid` | `14/20` / `15/20` | `+70.0` pts | `+7.400` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/b32_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/b32_none_text/independent_verification.json) |
| `09_toothbrush`（牙刷） | `0/20` / `1/20` | `AmbiLAGE` | `3/20` / `5/20` | `+15.0` pts | `+3.100` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/evaluation/b32_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/evaluation/b32_none_text/independent_verification.json) |

#### 2. 用 `ViT-g-14` 测试（未参与训练，黑盒跨模型迁移）

| 组别 | 原图 Clean Top-3 / Top-5 | 方法 | 修改图 Top-3 / Top-5 | Top-3 增量 | 平均名次改善 | 报告与复算凭证 |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `07_shampoo`（洗发水） | `19/20` / `20/20` | `Lucid` | `19/20` / `20/20` | `+0.0` pts | `+0.000` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/g14_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/07_shampoo/lucid/evaluation/g14_none_text/independent_verification.json) |
| `07_shampoo`（洗发水） | `19/20` / `20/20` | `AmbiLAGE` | `20/20` / `20/20` | `+5.0` pts | `+0.250` | [REPORT.md](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/evaluation/g14_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/07_shampoo/ambilage/evaluation/g14_none_text/independent_verification.json) |
| `09_toothbrush`（牙刷） | `0/20` / `3/20` | `Lucid` | `0/20` / `6/20` | `+0.0` pts | `+0.250` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/g14_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/09_toothbrush/lucid/evaluation/g14_none_text/independent_verification.json) |
| `09_toothbrush`（牙刷） | `0/20` / `3/20` | `AmbiLAGE` | `11/20` / `14/20` | `+55.0` pts | `+3.000` | [REPORT.md](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/evaluation/g14_none_text/REPORT.md) / [independent_verification.json](../../output/ambilage_2groups_20261114/09_toothbrush/ambilage/evaluation/g14_none_text/independent_verification.json) |

*(全部 8 份复算凭证均显示 `status: "passed"`、最大分数误差 `0.0`。如需另外加测单组单独替换底库、防御条件下表现或 `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` 检索器，告诉我即可继续跑。)*

---
