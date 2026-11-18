# 当前方法及比较边界

公开登记 `lucid`（原始像素优化方法）、`ambilage`（环境光背景协同调整及主体微光影方法）与 `visual_inception`（视觉记忆植入图片扰动基线）三种方法身份。各方法复用同一份冻结公开图对和核心特点正文，生成阶段绝不读取共享长期记忆池、正式请求、正式名次、严格留出编码器或严格留出净化器。

`AmbiLAGE`（当前定义 `ambilage_native_4stage_ambient_coop_psi_v4`）完整四阶段为：

1. 根据源商品公开品类和图像亮度统计，使用确定性显式环境光配方生成光影协调计划（不再依赖大参数视觉语言模型）；`SAM2` 主体分割与多尺度/反相增强 `RapidOCR` 识别（含 `definition.json` 已核对文字区域）联合局部中值滤波墨迹核心提取（`extract_ink_core_mask`，`ink_margin_pixels=2`），避免把彩色瓶身或金属漆误锁为墨迹，同时捕获深底白色小商标与抗锯齿笔画边缘并严格锁死高对比印刷文字核心，生成空间包络（`w_fg_body` 与 `w_bg_budget`）。
2. 从目标商品公开图片前景提取颜色统计（均值及分位数），准备背景谐调与平滑约束参数。
3. 在墨迹核心精确保护下，执行 Choice 1 稳健线性影棚微调与 32 步回溯线性搜索（`harmonization_selected_color_mix=0.12`，`harmonization_chroma_transfer_strength=0.0`，不强制大跨度主体换色），同时将纯白背景平滑过渡至温和浅暖棚拍底色（`harmonization_bg_luma_target=238.0`），结合全彩 RGB 与 OKLab 亮度双硬自然度门禁（SSIM、MAE、前景 RGB MAE 与轮廓光晕误差约束）生成自然锚点（不再加载 34GB 扩散编辑全量模型），并将源原图印刷墨迹核心逐像素回写。
4. 在原生十六通道变分自编码器（FLUX `AutoencoderKL` `vae` 子目录）潜空间中联合优化：背景分支在基础全局三维背景色向量 `bg_color_shift`（`[1, 3, 1, 1]`）叠加单自由度径向暗角轮廓上，支持显式启用七模态光滑 \(C^\infty\) 影棚照明空间基（`bg_studio_basis_enabled=True`，`ambient_basis` 形状 `[1, 3, 7]`，零初始化并在合理幅值 `bg_studio_basis_scale=0.08`、全局 `background_levels` 与印刷核绝对零改动硬约束内学习左右/上下/对角渐变及柔光光斑），在不引入高频噪斑的前提下补足 \(14\times 14\) 空间特征图所需的跨分块语义引导；前景分支默认保持 Choice 1 商品本色微光影（`fg_chroma_levels=0`，`latent_levels=6`，`fg_chroma_keep=0.15`），并在每步优化中对十六通道潜残差施加二维实傅里叶 $1/f$ 自然频谱流形投影（`spectral_manifold_projection=True`，抑制棋盘格高频伪影）以及按需边界切空间投影（`conditional_tangent_projection=True`，内部安全区零额外 VAE 反传，逼近 `SSIM`/`MAE`/`fg_rgb_mae` 边界时沿切锥滑动）。跨代理采用范数均衡且保持物理量纲的正交梯度融合（`PCGrad`），覆盖模糊、压缩、缩放和十步 `SD 1.5` 全梯度扩散净化。最终通过全彩 RGB 与 OKLab 亮度双硬自然度门禁（`source_ssim_floor=0.85`、`source_mae_ceiling=22.0`、`fg_source_mae_ceiling=10.0`、`fg_chroma_tv_ceiling=2.5`、边缘光晕与印刷核心零改动）；失败时只保存证据，禁止写完成凭证。配合针对 NVIDIA L4（24GB 显存）的闲置文本塔卸载、视觉梯度检查点、双视图微批与超大模型流式驻留机制，默认三代理峰值显存约 8.5–10.5 GB，即使切换包含 `ViT-bigG-14` 的巨型三代理方案峰值显存亦控制在约 11–13 GB。

在保留图文联合训练查询与源公开检索键的基础上，`AmbiLAGE v4` 通过渐进跨模态课程与双键文本语义蒸馏弥合 Modality Gap（跨模态鸿沟）。支持启用文本瓶颈感知的多代理动态加权与选优稳态均衡（`text_aware_proxy_weighting=True`）：`SequentialEnsembleObjective` 显式追踪各代理的独立文本相对增益 \(g_{\text{text}}\) 与联合相对增益 \(g_{\text{joint}}\)，以 `(1 - proxy_text_bottleneck_mix) * g + proxy_text_bottleneck_mix * min(g_joint, g_text)` 抑制仅靠表面视觉偏置推高 `joint` 而忽略 `text` 的虚高代理，并在 `psi_progressive_schedule=True` 且 \(g_{\text{joint}, i} - g_{\text{text}, i} > 0.005\) 时确定性倍增该代理的文本对齐梯度系数（最高 `1.85x`），同时按 `selection_text_parity_mix` 对齐稳态选优标尺。对与 `visual_inception` 基线进行 `14×14` 公平对比的实验配置（`surrogate_set="visual_inception_paper"`，即 `ViT-L-14` + `ViT-SO400M-14-SigLIP-384` + `ViT-bigG-14`），提供示例配置 `src/visual/config/ambilage.paper_surrogates.example.json`（上述增强默认保持 `False` 以严格维持既有默认训练行为与向后兼容性）。设归一化运算为 $N$，目标商品图片向量为 $I_t$，冻结目标特点文字向量为 $T_t$，修改图片向量为 $I_x$，源商品自身真实公开文字向量为 $T_s$，联合查询目标与候选键分别为：

$$q=N(I_t+T_t),\qquad k=N(N(I_x)+T_s).$$

训练效用采用 **PSI 三段式渐进跨模态课程**（视觉桥接期 $\to$ 文本主导与部件对齐期 $\to$ 稳态巩固期；快照选优与退火控制器统一使用恒定终端标尺 `progressive=False` 避免跨阶段尺度跳变），由四部分协同构成：
1. 图文联合目标（基础权重 `joint_weight=1.0`，并在 $q\cdot T_s > 10^{-4}$ 时以权重 `compensation_weight=0.35` 混入解析方向补偿 $c = N(2(q\cdot T_s)q - T_s)$）；
2. 三路多粒度独立文本对齐与闭式球面最坏查询增强（基础权重 `text_weight=0.35`、课程峰值 `2.4x`：将冻结公开特点正文确定性拆解为完整句、品类意图与独立属性子句簇，同时覆盖 `joint` 图文记忆键、`compensated` 源标题解析补偿与 `pure_image` 纯图记忆键 `text_pure_image_mix=0.25`，叠加零语言模型调用的单位球面帽闭式最坏查询扰动 `text_spherical_radius=0.05` 与干净源记忆键平滑对比排斥 `source_repulsion_weight=0.15`）；
3. 文本相关性加权的前景局部特征 Sinkhorn 最优传输（基础权重 `local_ot_weight=0.10`、课程峰值 `1.6x`：剔除视觉变换器 `[CLS]` 标记并筛选前景 Patch Tokens，按目标文本与 $K=4$ 前景语义聚类中心的相似度分配传输边际质量，引导潜空间微光影强化与目标文本高度相关的关键视觉部件）；
4. 可选同源公开训练包困难负例对比检索（`AmbiLAGE-Cat` 模式，通过 `--training-manifest` 或 `--training-bundles-root` 显式挂载冻结的 90 公开候选 + 100 独立公开查询训练包，同时激励 `image` 与 `joint` 记忆键在 32 个困难负例与 `0.30` 对比间隔下胜出）。

当前训练目标版本标识为 `sequential_public_joint_text_fg_ot_psi_v4`（评估端向后兼容已冻结的 `sequential_public_joint_text_fg_ot_v3` 产物以便历史对照，拒绝 `v1`/`v2` 旧产物）。

评估端除原有的 `SD 1.5` 扩散净化外，新增独立留出的 `eval_purifier_sd21`（`sd2-community/stable-diffusion-2-1-base`，对应 `--defense-mode cognitiveguard_sd21` 与 `panel_with_sd21`）以模拟异构扩散主干的 CognitiveGuard 系统一净化防御；`SD 2.1-base` 严格禁止进入任何生成或训练配置。

训练只用上述公开图片、前景掩膜和冻结特点正文。正式评估仍可选择纯文字或图文查询，多组评估遵循同批次同时替换同一完整共享库并覆盖标准非目标环境；评估模式不会改变训练目标，也不允许训练读取正式请求。

指定生产模型、训练代理和两个严格留出评估模型见资源目录的模型清单（`editor` 仅按 `vae/*` 模式拉取 16 通道 VAE 权重，`planner` 标记为可选 `optional`）。公开下载需要由服务器显式执行；运行阶段只加载本地权重。加载失败不能切换成随机网络、启发式模型或模拟编码器。

代理方案、命令参数及历史对照边界见 [训练代理选择说明](TRAINING_SURROGATES.md)。默认方案直接读取原始像素方法配置，不能只按架构名忽略同架构下不同预训练权重。

印刷核心零改动是相对源原图的实际像素约束。主体在 Stage 4 中默认保持商品本色并通过 `latent_levels=6` 与低色度保留率限制柔和微光影（`fg_chroma_levels=0`，如需显式全局色度微调消融仅作用于统一全局通道且严格受全彩 RGB 门禁核查）；源图到自然锚点同样在 RGB 与 OKLab 亮度双数学门禁下进行照明与色调协调。背景仅受全局三维颜色参数与一维径向暗角控制并受限于 `background_levels`。

对称可行区间、大核背景平滑、边缘光晕正则与空间包络用于减少截断及杂色伪影，最终外观由显式线性搜索锚点与确定性数学门禁共同把关；须保留失败案例，不根据正式排名重采样或改图。

扩散净化使用冻结噪声网络及确定性隐式逆过程，是项目中的指定净化条件，不冒充其他完整防御论文的精确复现。压缩期望变换保留实际压缩前向，反向使用明确记录的可微代理。两者具体路径均记入生成审计。

官方接口核对来源：[FLUX 16通道 VAE 模型来源](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)、[可选历史规划模型（默认停用）](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)、[代理模型实现](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py)、[异构训练代理说明](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384)。这些来源用于核对接口，不是本项目真实运行效果证据。
