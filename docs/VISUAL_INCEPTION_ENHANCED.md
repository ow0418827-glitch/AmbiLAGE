# 视觉记忆植入机制实现说明（`paper_minimal_margin`）

用户于二〇二六年十一月十七日至十一月二十一日先后授权并收敛 [Visual Inception 原论文（arXiv:2604.16966v1）第 3.2 节](https://arxiv.org/html/2604.16966v1#S3.SS2) 中三处未给出唯一数学公式的工程细节。此前内部试验过的 `enhanced_v3`（动态前 32 困难负例重排 + 文本正交子空间 3 步球面帽查询对抗优化，存在与 AmbiLAGE 自研方法机制混同的问题）及无 `+0.3` 间隔的过渡版 `paper_minimal` 接口已全部移除，仅保留历史 `legacy_v2` 与当前主用单层基线 **`paper_minimal_margin`**。

## 当前严格单层基线 `paper_minimal_margin`

基础公式与超参数严格保持原论文 Eq. (2)–(3) 及 Appendix A.2（`sign-PGD`、`steps=100`、`epsilon_levels=8`、步长 `2/255`、$\lambda_1=1.0$、$\lambda_2=0.1$、检索温度 $\tau=0.07$、100 条训练查询池、`query_batch_size=16`），三项未给公式细节均采用最朴素、无额外技巧的单层实现（不宣称与作者未公开源码逐行一致，`author_exact_implementation_claimed=False`）：

1. **静态三来源随机负例库 $\mathcal{M}$（不做每步动态挖掘与品类轮询）**：
   - 优化开始前，根据冻结公开候选元数据与固定向量从三类独立公开候选池中各**不放回均匀随机抽取固定配额**（默认各 `10` 个，合计 `30` 个：`negative_same_category_count=10`、`negative_popularity_count=10`、`negative_proxy_memory_count=10`）：
     - `same_category`：同品类公开非目标商品；
     - `popular`：优先采用有公开来源凭证的外部热度字段；缺少真实销量字段时，按全库全局平均余弦中心性排序取前三分之一（无品类 round-robin 轮询中心性）；
     - `proxy_user_memory`：使用跨品类独立公开候选作为代理记忆（仅在单品类微型测试夹具回退全库），严禁读取正式用户记忆或正式测试查询。
   - 静态池 $\mathcal{M}$ 直接固定代入 Eq. (3) InfoNCE 分母，每步不再动态重排 Top-32 困难负例。

2. **无梯度各向同性高斯查询扰动（严格维持单层多目标优化）**：
   - 不构建文本正交子空间、不执行内层 PGD/FGSM 梯度上升循环；
   - 对每个 `batch` 内随机抽取的 `25%` 查询向量（`query_noise_fraction=0.25`，即 `16` 条中随机 `4` 条，其余保持原样）叠加各向同性高斯噪声并重新单位球面归一化：
     $$q' = \operatorname{normalize}(q + \varepsilon_q), \qquad \varepsilon_q \sim \mathcal{N}(0, \sigma^2 I), \quad \sigma = 0.03$$
   - 随机种子由基础 `seed`、阶段标识（`calibration` / `step` / `final`）、步号及代理序号确定性派生，外层仅对图像像素求导。

3. **Eq. (3) 静态负例统一 `+0.3` 对数几率间隔（不新增第四项损失）**：
   - 在保持原论文三项损失结构不变的前提下，将 `contrastive_margin=0.3` 统一加在 Eq. (3) 分母中所有静态负例的相似度对数几率上：
     $$z_+ = \frac{\cos(E(I_{\text{adv}}), E(q'))}{\tau}, \qquad z_j = \frac{\cos(E(n_j), E(q'))}{\tau} + 0.3 \quad (n_j \in \mathcal{M})$$
     $$\mathcal{L}_{\text{ret}}(q') = \log\!\left(e^{z_+} + \sum_{j \in \mathcal{M}} e^{z_j}\right) - z_+$$
   - 不新增第四项 Hinge Margin 损失，也不对原始余弦直接减 `0.3` 后再除以 `0.07`（避免 $0.3 / 0.07 \approx 4.29$ 的数值梯度坍塌）。

## 版本与配置说明

- `mechanism_profile` 默认及推荐值为 `paper_minimal_margin`（配置示例见 [visual_inception.example.json](../src/visual/config/visual_inception.example.json) 与 [visual_inception.paper_surrogates.example.json](../src/visual/config/visual_inception.paper_surrogates.example.json)）；历史第二版保留为 `legacy_v2`。
- 训练目标身份标识为 `visual_inception_retrieval_semantic_perceptual_paper_minimal_margin_v4`。已移除的 `enhanced_v3` 与 `paper_minimal` 配置会被配置校验直接拒绝；历史与旧版产物禁止跨版本重签或混评。
- 公开代理负例模块 [negative_sampling.py](../src/visual/_visual_inception/negative_sampling.py) 继续作为 `AmbiLAGE` 训练目标依赖保留，不受本次 Visual Inception 基线收口影响。
