"""AmbiLAGE (环境光协同背景调整＋主体微光影优化) 原生配置模型与参数校验。
包含：
1. 四阶段显式规划与形态学墨迹保护参数；
2. Stage 3 标签底色透射融合与 32 步回溯线搜索数学守门参数；
3. Stage 4 十六通道变分自编码器潜空间与零中心可行无截断环境光场参数；
4. 严格零像素扰动约束 (auxiliary_levels == 0, pixel_optimization is False)；
5. 印刷墨迹核心绝对零改动 (change[roi] == 0) 与结尾严格数学自然度门禁。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping


METHOD = "ambilage"
DEFINITION = "ambilage_native_4stage_ambient_coop_v1"
MASK_POLICY = "ambilage_subject_micro_light_background_range_centered_ambient_smooth_v1"


@dataclass(frozen=True)
class AmbiLAGEConfig:
    # 优化迭代步数、自适应退火区间 [180, 250] 与断点续跑配置
    steps: int = 250
    check_every: int = 10
    log_every: int = 10
    adaptive_annealing: bool = True
    anneal_min_steps: int = 180
    anneal_min_lr_ratio: float = 0.08
    anneal_warmup_steps: int = 10
    anneal_plateau_factor: float = 0.75
    anneal_plateau_patience: int = 2
    anneal_backtrack_factor: float = 0.90
    anneal_improve_tol: float = 5e-4
    anneal_early_stop_patience: int = 4
    checkpoint_every: int = 5
    resume_from_checkpoint: bool = True

    # 主体与背景扰动预算 (0~255 整数级)
    latent_levels: int = 6              # 主体微光影预算 (<= 6 级)
    background_levels: int = 28         # 背景环境光场预算 (<= 28 级)
    auxiliary_levels: int = 0           # 严格禁止像素残差 (0 级)
    pixel_optimization: bool = False    # 严格禁止像素优化

    # Stage 1: 空间包络与欧氏距离变换 (EDT) 四重余弦平滑参数
    fg_edge_feather_pixels: float = 8.0 # 主体轮廓向内余弦过渡宽度 (w_fg: 0~8px, 0.0 -> 1.0)
    edge_feather_pixels: float = 12.0   # 背景局部 VAE 波动向外余弦过渡宽度 (w_bg_mod: 0~12px, 0.25 -> 1.0)
    ambient_feather_pixels: float = 3.0 # 背景总预算抗锯齿带微距平滑宽度 (w_bg_budget: 0~3px, 0.70 -> 1.0)
    roi_feather_pixels: float = 6.0     # 文字边缘余弦缓冲过渡宽度 (w_text: 6px buffer)
    edge_feather_mode: str = "cosine"   # 'cosine' | 'linear'

    # Stage 4: 大核高斯平滑滤波与色度保留率参数（锁死背景高频杂色，保留16通道主体细腻光影）
    fg_luma_smooth_sigma: float = 5.0   # 主体明度大核高斯平滑标准差
    fg_chroma_smooth_sigma: float = 9.0 # 主体色度大核高斯平滑标准差
    fg_chroma_keep: float = 0.30        # 主体色度保留率 (抑制杂色斑块，仅保留柔光影)
    bg_luma_smooth_sigma: float = 12.0  # 背景明度超大核高斯平滑标准差 (锁死高频斑块)
    bg_chroma_smooth_sigma: float = 16.0 # 背景色度超大核高斯平滑标准差
    bg_chroma_keep: float = 0.10        # 背景色度保留率 (锁死背景杂色通道)
    ambient_smooth_sigma: float = 14.0  # 背景环境漫射光高斯平滑标准差
    ambient_grid_size: int = 8          # 背景低频环境光网格尺寸 (8x8)
    ambient_init_bias: float = 0.0      # 初始环境光偏置系数 (零偏置从 Stage 3 浅暖棚拍底色中心起步)
    tau_scale: float = 0.75             # 兼容缩放系数
    tau_fg: float = 0.20                # 主体双曲正切温度尺度 (防止过早饱和)
    tau_bg: float = 0.35                # 背景双曲正切温度尺度 (防止背景饱和成块)

    # 潜空间与环境光优化参数
    latent_working_size: int = 512
    latent_limit: float = 6.0
    latent_learning_rate: float = 0.06
    ambient_learning_rate: float = 0.004
    ambient_limit: float = 0.15
    latent_smoothing_mix: float = 0.05
    temperature: float = 0.04

    # 目标函数与正则化权重
    joint_weight: float = 1.0               # 图文联合检索效用整体权重
    compensation_weight: float = 0.35       # 源商品完整公开文本逆向补偿目标权重
    ensemble_temperature: float = 0.20      # 跨代理相对增益软最小集成温度
    ensemble_uniform_mix: float = 0.45      # 多代理均匀保底混合比例
    surrogate_set: str = 'lucid'            # 默认与原始像素方法共用三代理
    surrogate_config: str | None = None
    compensated_weight: float = 0.5
    fg_l2_weight: float = 0.0001            # 主体微光影 L2 惩罚
    fg_tv_weight: float = 0.0004            # 主体空间全变分 (TV) 惩罚
    fg_chroma_tv_weight: float = 0.02       # 主体 OKLab 色度平滑 (TV) 惩罚
    bg_l2_weight: float = 0.005             # 背景围绕均值偏移的空间方差惩罚 (释放柔和光影同时不惩罚整体平移)
    bg_tv_weight: float = 0.02              # 背景空间全变分 (TV) 惩罚
    bg_chroma_tv_weight: float = 0.08       # 背景色度全变分惩罚
    overflow_weight: float = 0.05           # 越界惩罚
    halo_weight: float = 0.002              # 边界光晕可微惩罚权重 (对齐 0~255 量纲，避免惩罚项反超相似度触发误回滚)
    halo_soft_weight: float = 0.0001        # 兼容旧光晕软约束权重
    halo_floor: float = 12.0
    halo_ceiling: float = 10.0              # 严格数学门禁光晕误差上限
    fg_chroma_tv_ceiling: float = 12.0      # 主体 OKLab 色度 TV 门禁上限
    halo_parent_multiplier: float = 1.0
    guard_backtracks: int = 8               # 回退守门最大步数

    # Stage 3 & 4 前景 OKLab 锁亮度放色度换色与背景全局三维颜色参数
    fg_chroma_levels: int = 64              # 主体 OKLab 色度通道自由度折算上限 (等价 0~255 量纲)
    fg_chroma_limit: float = 0.16           # 主体 OKLab 全局色度平移 [1,2,1,1] 最大幅值
    fg_luma_scale: float = 0.40             # 主体 OKLab 明度 L 锁死折算系数 (|dL| <= latent_levels/255 * scale)
    fg_neutral_recolor_base: float = 0.45   # 中性哑光非印刷表面基础换色增益 (高饱和塑料接近 1.0)
    fg_highlight_luma_knee: float = 0.88    # 高光反光中性区保护起衰减明度阈值 (OKLab L)
    fg_shadow_luma_knee: float = 0.25       # 深阴影/黑瞳孔五官特征区保护起衰减明度阈值 (OKLab L)
    bg_radial_vignette_scale: float = 0.05  # 背景低频径向暗角/光晕最大调制幅度

    # 质量守门门槛 (SSIM & MAE & Halo)
    source_ssim_floor: float = 0.78         # 最终成图结构相似度下限 (基于 OKLab 亮度 L)
    source_mae_ceiling: float = 28.0        # 最终成图亮度平均绝对误差上限 (基于 OKLab 亮度 L)
    anchor_ssim_floor: float = 0.72         # Stage 3 锚点线搜索结构相似度参考下限 (基于 OKLab 亮度 L)
    anchor_mae_ceiling: float = 30.0        # Stage 3 锚点线搜索亮度平均误差参考上限

    # Stage 3: 光照一致性协调、温和浅暖棚拍底色与形态学墨迹保护参数
    harmonization_smoothing_sigma: float = 12.0
    harmonization_print_ring_sigma: float = 7.0
    harmonization_selected_color_mix: float = 0.12
    harmonization_chroma_transfer_strength: float = 0.55
    harmonization_strength: float = 1.0
    harmonization_blend_mode: str = "matte"
    harmonization_fg_feather_pixels: float = 4.0
    harmonization_bg_strength: float = 0.85
    harmonization_bg_luma_target: float = 238.0
    harmonization_bg_warmth: float = 3.0
    harmonization_bg_tint_mix: float = 0.20
    harmonization_bg_feather_pixels: float = 6.0
    ink_margin_pixels: int = 1
    ink_closing_pixels: int = 9
    ink_local_contrast: float = 3.0
    ink_dark_luminance: float = 180.0
    ink_colored_chroma: float = 12.0

    # 硬件加速、轻量模型路径与扩散净化画像
    allow_tf32: bool = True                 # 启用 TensorFloat-32 原生张量核矩阵乘法加速
    enable_gradient_checkpointing: bool | None = None
    save_intermediate_pngs: bool = False
    device: str = "cuda"
    seed: int = 42
    # 生产后端只加载真实权重；轻量模式必须由调用者显式声明。
    smoke_test: bool = False
    sam_model_dir: str | None = None
    planner_model_dir: str | None = None    # 兼容保留，Choice 1 显式规划不再加载 Qwen
    editor_model_dir: str | None = None     # 兼容保留，Choice 1 仅使用其中 16 通道 vae 子目录
    vae_model_dir: str | None = None
    vae_model_path: str | None = None
    purifier_model_path: str | None = None
    purifier_working_size: int = 256
    purifier_gradient_checkpointing: bool = True
    editor_kind: str = "flux"
    anchor_steps: int = 32                  # Stage 3 32 步回溯线搜索步数
    anchor_guidance_scale: float = 2.5
    planner_max_new_tokens: int = 768
    eot_enabled: bool = True
    diffusion_weight: float = 0.08          # 压低扩散净化梯度权重以消除潜空间棋盘格伪影
    purifier_steps: int = 10

    def validate(self) -> None:
        """严格校验配置参数有效性，杜绝越界或无效数值。"""
        from src.visual.training_surrogates import SURROGATE_SETS
        if self.surrogate_set not in SURROGATE_SETS:
            raise ValueError('未知训练代理方案：' + str(self.surrogate_set))
        if self.surrogate_set == 'custom':
            if not isinstance(self.surrogate_config, str) or not self.surrogate_config.strip():
                raise ValueError('自定义训练代理必须提供配置文件')
        elif self.surrogate_config is not None:
            raise ValueError('内置训练代理方案不能同时指定自定义代理配置')
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("优化步数 steps 必须为正整数")
        if type(self.check_every) is not int or self.check_every < 1:
            raise ValueError("检查周期 check_every 必须为正整数")
        if type(self.latent_levels) is not int or not (0 <= self.latent_levels <= 6):
            raise ValueError("主体微光影预算 latent_levels 必须位于 [0, 6]")
        if type(self.background_levels) is not int or not (1 <= self.background_levels <= 64):
            raise ValueError("背景光场预算 background_levels 必须位于 [1, 64]")
        if self.latent_levels == 0 and self.background_levels == 0:
            raise ValueError("主体与背景预算不能同时为零")
        
        # 强制零像素与零残差断言
        if self.auxiliary_levels != 0 or self.pixel_optimization is not False:
            raise ValueError("AmbiLAGE 严格禁止像素残差与像素优化 (auxiliary_levels=0, pixel_optimization=False)")

        # 空间平滑过渡参数校验
        if not (1.0 <= self.fg_edge_feather_pixels <= 32.0):
            raise ValueError("主体内缩过渡宽度 fg_edge_feather_pixels 无效")
        if not (2.0 <= self.edge_feather_pixels <= 32.0):
            raise ValueError("背景局部纹理过渡宽度 edge_feather_pixels 无效")
        if not (1.0 <= self.ambient_feather_pixels <= 16.0):
            raise ValueError("背景总预算抗锯齿平滑宽度 ambient_feather_pixels 无效")
        if not (0.0 <= self.roi_feather_pixels <= 16.0):
            raise ValueError("印刷文字缓冲平滑宽度 roi_feather_pixels 无效")
        if self.edge_feather_mode not in ("cosine", "linear"):
            raise ValueError(f"未知的边界过渡模式: {self.edge_feather_mode}")

        # 高斯滤波标准差与色度保留率校验
        sigmas = [
            self.fg_luma_smooth_sigma, self.fg_chroma_smooth_sigma,
            self.bg_luma_smooth_sigma, self.bg_chroma_smooth_sigma,
            self.ambient_smooth_sigma, self.harmonization_smoothing_sigma,
            self.harmonization_print_ring_sigma,
        ]
        if any(s < 0.0 for s in sigmas):
            raise ValueError("高斯平滑滤波核标准差不能为负数")
        if not (0.05 <= self.fg_chroma_keep <= 1.0) or not (0.05 <= self.bg_chroma_keep <= 1.0):
            raise ValueError("色度保留率必须位于 [0.05, 1.0]")
        if not (0.2 <= self.tau_scale <= 1.0):
            raise ValueError("平滑缩放系数 tau_scale 必须位于 [0.2, 1.0]")
        if not (0.001 <= self.tau_fg <= 5.0) or not (0.001 <= self.tau_bg <= 5.0):
            raise ValueError("双曲正切温度尺度 tau_fg/tau_bg 无效")

        # 学习率与优化参数校验
        if not (0.0 < self.latent_learning_rate <= 0.2):
            raise ValueError("潜变量学习率无效")
        if not (0.0 < self.ambient_learning_rate <= 0.05):
            raise ValueError("环境光学习率无效")
        if not (0.01 <= self.ambient_limit <= 1.0):
            raise ValueError("环境光更新上限 ambient_limit 无效")
        if type(self.ambient_grid_size) is not int or not (2 <= self.ambient_grid_size <= 32):
            raise ValueError("环境光网格尺寸 ambient_grid_size 无效")
        if not (0.0 < self.temperature <= 1.0):
            raise ValueError("检索效用温度参数无效")
        if not (0.0 <= self.compensation_weight < 1.0):
            raise ValueError("逆向补偿目标权重 compensation_weight 必须位于 [0.0, 1.0)")
        if not (0.0 < self.ensemble_temperature <= 5.0) or not (0.0 <= self.ensemble_uniform_mix <= 1.0):
            raise ValueError("多代理集成权重参数无效")

        # 质量守门与正则化权重校验
        if not (0.25 <= self.source_ssim_floor < 1.0) or not (0.0 < self.source_mae_ceiling <= 128.0):
            raise ValueError("质量守门门槛无效")
        if not (0.25 <= self.anchor_ssim_floor < 1.0) or not (0.0 < self.anchor_mae_ceiling <= 128.0):
            raise ValueError("锚点线搜索质量门槛无效")
        if self.halo_ceiling < 0.0:
            raise ValueError("边界光晕门禁上限 halo_ceiling 不能为负数")
        if self.fg_chroma_tv_ceiling < 0.0:
            raise ValueError("前景 OKLab 色度 TV 门禁上限 fg_chroma_tv_ceiling 不能为负数")
        if type(self.fg_chroma_levels) is not int or not (0 <= self.fg_chroma_levels <= 128):
            raise ValueError("前景 OKLab 色度预算 fg_chroma_levels 必须位于 [0, 128]")
        if not (0.0 <= self.fg_chroma_limit <= 0.5):
            raise ValueError("前景 OKLab 全局色度偏移上限 fg_chroma_limit 无效")
        if not (0.0 <= self.fg_luma_scale <= 2.0):
            raise ValueError("前景 OKLab 明度约束比例 fg_luma_scale 无效")
        if not (0.0 <= self.fg_neutral_recolor_base <= 1.0):
            raise ValueError("中性材质换色基础增益 fg_neutral_recolor_base 必须位于 [0.0, 1.0]")
        if not (0.5 <= self.fg_highlight_luma_knee <= 1.0) or not (0.0 <= self.fg_shadow_luma_knee <= 0.5):
            raise ValueError("前景高低光保护阈值无效")
        if not (0.0 <= self.bg_radial_vignette_scale <= 0.25):
            raise ValueError("背景径向暗角系数 bg_radial_vignette_scale 必须位于 [0.0, 0.25]")
        if not (0.0 <= self.harmonization_chroma_transfer_strength <= 1.0):
            raise ValueError("Stage 3 OKLab 色度迁移强度 harmonization_chroma_transfer_strength 必须位于 [0.0, 1.0]")
        if not (0.0 <= self.harmonization_bg_strength <= 1.0):
            raise ValueError("背景棚拍底色融合强度 harmonization_bg_strength 必须位于 [0.0, 1.0]")
        if not (210.0 <= self.harmonization_bg_luma_target <= 255.0):
            raise ValueError("背景棚拍目标明度 harmonization_bg_luma_target 必须位于 [210.0, 255.0]")
        if not (-15.0 <= self.harmonization_bg_warmth <= 15.0):
            raise ValueError("背景棚拍暖调偏置 harmonization_bg_warmth 必须位于 [-15.0, 15.0]")
        if not (0.0 <= self.harmonization_bg_tint_mix <= 0.5):
            raise ValueError("背景目标色温混入比例 harmonization_bg_tint_mix 必须位于 [0.0, 0.5]")
        if self.harmonization_fg_feather_pixels < 0.0 or self.harmonization_bg_feather_pixels < 0.0:
            raise ValueError("Stage 3 边界羽化宽度不能为负数")
        weights = [
            self.fg_l2_weight, self.fg_tv_weight, self.fg_chroma_tv_weight,
            self.bg_l2_weight, self.bg_tv_weight, self.bg_chroma_tv_weight,
            self.overflow_weight, self.halo_soft_weight, self.halo_weight,
        ]
        if any(w < 0.0 for w in weights):
            raise ValueError("正则化权重不能为负数")
        if type(self.guard_backtracks) is not int or not (0 <= self.guard_backtracks <= 16):
            raise ValueError("回退守门步数无效")
        if type(self.log_every) is not int or self.log_every < 1:
            raise ValueError("日志周期必须为正整数")
        if type(self.adaptive_annealing) is not bool:
            raise ValueError("自适应退火开关 adaptive_annealing 必须为布尔值")
        if type(self.anneal_min_steps) is not int or self.anneal_min_steps < 1:
            raise ValueError("自适应退火最小步数 anneal_min_steps 必须为正整数")
        if not (0.0 < self.anneal_min_lr_ratio <= 1.0):
            raise ValueError("自适应退火最小学习率比例 anneal_min_lr_ratio 必须位于 (0.0, 1.0]")
        if type(self.anneal_warmup_steps) is not int or self.anneal_warmup_steps < 0:
            raise ValueError("退火预热步数 anneal_warmup_steps 不能为负数")
        if not (0.1 <= self.anneal_plateau_factor <= 1.0):
            raise ValueError("平台期降温系数 anneal_plateau_factor 必须位于 [0.1, 1.0]")
        if type(self.anneal_plateau_patience) is not int or self.anneal_plateau_patience < 1:
            raise ValueError("平台期耐心周期 anneal_plateau_patience 必须为正整数")
        if not (0.1 <= self.anneal_backtrack_factor <= 1.0):
            raise ValueError("回退降温系数 anneal_backtrack_factor 必须位于 [0.1, 1.0]")
        if self.anneal_improve_tol < 0.0:
            raise ValueError("有效提升容差 anneal_improve_tol 不能为负数")
        if type(self.anneal_early_stop_patience) is not int or self.anneal_early_stop_patience < 0:
            raise ValueError("退火早停耐心次数 anneal_early_stop_patience 不能为负数")
        if type(self.checkpoint_every) is not int or self.checkpoint_every < 1:
            raise ValueError("断点保存周期 checkpoint_every 必须为正整数")
        if type(self.resume_from_checkpoint) is not bool:
            raise ValueError("断点续跑开关 resume_from_checkpoint 必须为布尔值")
        if type(self.smoke_test) is not bool:
            raise ValueError("轻量验证开关必须为布尔值")
        if self.editor_kind not in ("flux", "qwen"):
            raise ValueError("锚点生成后端必须为 flux 或 qwen")
        if type(self.anchor_steps) is not int or self.anchor_steps < 1:
            raise ValueError("自然锚点生成步数必须为正整数")
        if type(self.latent_working_size) is not int or self.latent_working_size < 64 or self.latent_working_size % 8:
            raise ValueError("潜空间工作尺寸必须至少六十四且为八的倍数")
        if not self.eot_enabled or self.diffusion_weight <= 0 or self.purifier_steps != 10:
            raise ValueError("正式方法须开启期望变换和正权重十步扩散净化")
        if self.joint_weight <= 0:
            raise ValueError("图文联合目标权重必须为正")

        # 检查有限实数
        for key, val in asdict(self).items():
            if isinstance(val, float) and not math.isfinite(val):
                raise ValueError(f"配置参数 {key} 必须为有限实数: {val}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AmbiLAGEConfig:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError("未知方法配置字段: " + ", ".join(sorted(unknown)))
        instance = cls(**data)
        instance.validate()
        return instance
