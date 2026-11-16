# 当前方法及比较边界

仅公开 lucid（原始像素优化方法）与 ambilage（环境光背景协同调整及主体微光影方法）。二者复用同一份冻结公开图对和核心特点正文，生成不读取共享池、正式请求、正式名次或留出模型。

新方法四阶段为：

1. 根据源商品公开品类和图像亮度统计，使用确定性显式环境光配方生成光影协调计划（不再依赖大参数视觉语言模型）；`SAM2` 主体分割与多尺度/反相增强 `RapidOCR` 识别（含 `definition.json` 已核对文字区域）联合局部中值滤波墨迹核心提取（`extract_ink_core_mask`，`ink_margin_pixels=2`），避免把彩色瓶身或金属漆误锁为墨迹，同时捕获深底白色小商标与抗锯齿笔画边缘并严格锁死高对比印刷文字核心，生成空间包络（`w_fg_body` 与 `w_bg_budget`）。
2. 从目标商品公开图片前景提取颜色统计（均值及分位数），准备背景谐调与平滑约束参数。
3. 在墨迹核心精确保护下，执行 Choice 1 稳健线性影棚微调与 32 步回溯线性搜索（`harmonization_selected_color_mix=0.12`，`harmonization_chroma_transfer_strength=0.0`，不强制大跨度主体换色），同时将纯白背景平滑过渡至温和浅暖棚拍底色（`harmonization_bg_luma_target=238.0`），结合全彩 RGB 与 OKLab 亮度双硬自然度门禁（SSIM、MAE、前景 RGB MAE 与轮廓光晕误差约束）生成自然锚点（不再加载 34GB 扩散编辑全量模型），并将源原图印刷墨迹核心逐像素回写。
4. 在原生十六通道变分自编码器（FLUX `AutoencoderKL` `vae` 子目录）潜空间中联合优化：背景分支彻底取消逐像素 16 通道潜差分与 `8x8` 空间网格，替换为唯一可学习全局三维背景色向量 `bg_color_shift`（`[1, 3, 1, 1]`）叠加单自由度径向暗角轮廓，从数学结构上消除背景局部斑点；前景分支默认关闭高自由度 OKLab 换色（`fg_chroma_levels=0`，严禁前景逐像素 VAE 色度差值制造花斑），回归受 `latent_levels=6` 与紧缩 `fg_chroma_keep=0.15` 约束的主体微光影优化，并辅以强化色度全变分惩罚（`fg_chroma_tv_weight=0.15`）与全彩边缘光晕抑制（`differentiable_halo(..., luma_only=False)`）。按显式方案选择训练代理，逐一计算图像梯度再回传潜变量；联合优化图文联合检索目标、适应性独立文本对齐目标（`text_weight=0.35`）与仅限前景掩膜内部的局部特征 Sinkhorn 最优传输目标（`local_ot_weight=0.10`、`clusters=4`），覆盖模糊、压缩、缩放和十步全梯度扩散净化。最终通过基于全彩 RGB 与 OKLab 亮度双硬自然度门禁（`source_ssim_floor=0.85`、`source_mae_ceiling=22.0`、`fg_source_mae_ceiling=10.0`、`fg_chroma_tv_ceiling=2.5`、边缘光晕与印刷核心零改动）；失败时只保存证据，禁止写完成凭证。全流程峰值显存约 10–12 GB，兼容 T4（16GB）、L4（24GB）与 A100（40GB）显卡。

按二〇二六年十一月十六日用户最新确认，在严格保留图文联合训练查询与源公开检索键的基础上，解决 Modality Gap（跨模态鸿沟）与跨模型局部对齐问题。设归一化运算为 $N$，目标商品图片向量为 $I_t$，冻结目标特点文字向量为 $T_t$，修改图片向量为 $I_x$，源商品自身真实公开文字向量为 $T_s$，联合查询目标与候选键分别为：

$$q=N(I_t+T_t),\qquad k=N(N(I_x)+T_s).$$

训练效用由三部分协同构成：1）图文联合目标（权重 `joint_weight=1.0`，并在 $q\cdot T_s > 10^{-4}$ 时以权重 `compensation_weight=0.35` 混入解析方向补偿 $c = N(2(q\cdot T_s)q - T_s)$）；2）适应性独立特点文字对齐目标（权重 `text_weight=0.35`，以比例 `text_key_mix=0.60` 混合候选全键对齐 $k\cdot T_t$ 与仅在 $T_t\cdot T_s > 10^{-4}$ 时启用的解析文本补偿方向 $I_x \cdot c_{\text{text}}$，突破图像主导造成的 `Env 1~3` 涨分瓶颈）；3）仅限前景掩膜内部的空间局部特征最优传输（权重 `local_ot_weight=0.10`，剔除 ViT `[CLS]` 标记并按前景掩膜对特征图网格自适应筛选前景 Patch Tokens，用 Sinkhorn 传输匹配目标前景 `K=4` 语义中心，提升黑盒编码器跨模型泛化）。每个代理继续通过初始锚点基线校准的相对增益软最小权重（`ensemble_temperature=0.20`、`ensemble_uniform_mix=0.45`）均衡优化幅度。

当前方法版本标识为 `sequential_public_joint_text_fg_ot_v3`，配置与冻结审计均严格绑定 `text_weight` 与 `local_ot_weight` 实际参数；旧双目标（`v1`）及纯联合目标（`v2`）历史完成凭证不能作为当前版本成图复用，历史文件保留原记录。

训练只用上述公开图片、前景掩膜和冻结特点正文。正式评估仍可选择纯文字或图文查询，多组评估遵循同批次同时替换同一完整共享库并覆盖标准非目标环境；评估模式不会改变训练目标，也不允许训练读取正式请求。

指定生产模型、训练代理和两个严格留出评估模型见资源目录的模型清单（`editor` 仅按 `vae/*` 模式拉取 16 通道 VAE 权重，`planner` 标记为可选 `optional`）。公开下载需要由服务器显式执行；运行阶段只加载本地权重。加载失败不能切换成随机网络、启发式模型或模拟编码器。

代理方案、命令参数及历史对照边界见 [训练代理选择说明](TRAINING_SURROGATES.md)。默认方案直接读取原始像素方法配置，不能只按架构名忽略同架构下不同预训练权重。

印刷核心零改动是相对源原图的实际像素约束。主体在 Stage 4 中默认保持商品本色并通过 `latent_levels=6` 与低色度保留率限制柔和微光影（`fg_chroma_levels=0`，如需显式全局色度微调消融仅作用于统一全局通道且严格受全彩 RGB 门禁核查）；源图到自然锚点同样在 RGB 与 OKLab 亮度双数学门禁下进行照明与色调协调。背景仅受全局三维颜色参数与一维径向暗角控制并受限于 `background_levels`。

对称可行区间、大核背景平滑、边缘光晕正则与空间包络用于减少截断及杂色伪影，最终外观由显式线性搜索锚点与确定性数学门禁共同把关；须保留失败案例，不根据正式排名重采样或改图。

扩散净化使用冻结噪声网络及确定性隐式逆过程，是项目中的指定净化条件，不冒充其他完整防御论文的精确复现。压缩期望变换保留实际压缩前向，反向使用明确记录的可微代理。两者具体路径均记入生成审计。

官方接口核对来源：[FLUX 16通道 VAE 模型来源](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)、[可选历史规划模型（默认停用）](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)、[代理模型实现](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py)、[异构训练代理说明](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384)。这些来源用于核对接口，不是本项目真实运行效果证据。
