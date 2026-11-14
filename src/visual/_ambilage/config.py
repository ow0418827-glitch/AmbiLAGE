"""AmbiLAGE (环境光协同背景调整＋主体微光影优化) 原生配置模型与参数校验。
面向 A100-SXM4-80GB 原生加速调优，包含：
1. 四阶段空间参数与平滑滤波标准差；
2. 背景无截断可行区间对称中点展开与网格参数；
3. 严格零像素扰动约束 (auxiliary_levels == 0, pixel_optimization is False)；
4. 印刷文字区域绝对保护 (change[roi] == 0)；
5. A100 硬件画像参数 (TF32, 显存自适应检查点, 512GB 磁盘防护)。
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
    # 优化迭代步数与检查周期
    steps: int = 350
    check_every: int = 10
    log_every: int = 10

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

    # Stage 4: 大核高斯平滑滤波与色度保留率参数
    fg_luma_smooth_sigma: float = 5.0   # 主体明度大核高斯平滑标准差
    fg_chroma_smooth_sigma: float = 9.0 # 主体色度大核高斯平滑标准差
    fg_chroma_keep: float = 0.30        # 主体色度保留率 (抑制杂色斑块，仅保留柔光影)
    bg_luma_smooth_sigma: float = 2.2   # 背景 VAE 明度高斯平滑标准差
    bg_chroma_smooth_sigma: float = 4.5 # 背景 VAE 色度高斯平滑标准差
    bg_chroma_keep: float = 0.75        # 背景 VAE 色度保留率
    ambient_smooth_sigma: float = 14.0  # 背景环境漫射光高斯平滑标准差
    ambient_grid_size: int = 8          # 背景低频环境光网格尺寸 (8x8)
    ambient_init_bias: float = 0.30     # 初始环境光偏置系数 (0.30 * tau)
    tau_scale: float = 0.75             # 双曲正切平滑缩放系数 (tau_bg = max(half_bg, 1/255) * tau_scale)

    # 潜空间与环境光优化参数
    latent_working_size: int = 512
    latent_limit: float = 6.0
    latent_learning_rate: float = 0.06
    ambient_learning_rate: float = 0.004
    ambient_limit: float = 0.15
    latent_smoothing_mix: float = 0.05
    temperature: float = 0.04

    # 目标函数与正则化权重
    joint_weight: float = 1.0          # 唯一图文联合检索效用的整体权重
    compensated_weight: float = 0.5
    fg_l2_weight: float = 0.0001        # 主体微光影 L2 惩罚
    fg_tv_weight: float = 0.0004        # 主体空间全变分 (TV) 惩罚
    bg_l2_weight: float = 0.00005       # 背景围绕均值偏移的方差惩罚 (不惩罚整体色调平移)
    bg_tv_weight: float = 0.0012        # 背景空间全变分 (TV) 惩罚
    bg_chroma_tv_weight: float = 0.0020 # 背景色度全变分惩罚
    overflow_weight: float = 0.05       # 越界惩罚
    halo_soft_weight: float = 0.0001    # 光晕软约束惩罚
    halo_floor: float = 12.0
    halo_parent_multiplier: float = 1.0
    guard_backtracks: int = 8           # 回退守门最大步数

    # 质量守门门槛 (SSIM & MAE)
    source_ssim_floor: float = 0.78     # 全图结构相似度下限
    source_mae_ceiling: float = 20.0    # 全图平均绝对误差上限

    # Stage 3: 光照一致性协调参数
    harmonization_smoothing_sigma: float = 12.0
    harmonization_print_ring_sigma: float = 7.0
    harmonization_selected_color_mix: float = 0.25
    harmonization_strength: float = 1.0
    harmonization_blend_mode: str = "matte"

    # A100 硬件加速与资源防护画像
    allow_tf32: bool = True             # 启用 TensorFloat-32 原生张量核矩阵乘法加速
    enable_gradient_checkpointing: bool | None = None  # None: 显存>32GB 时自动禁用
    save_intermediate_pngs: bool = False # 512GB 磁盘防护：严禁每步落盘 PNG，仅存最终成图
    device: str = "cuda"
    seed: int = 42
    # 生产后端只加载真实权重；轻量模式必须由调用者显式声明。
    smoke_test: bool = False
    sam_model_dir: str | None = None
    planner_model_dir: str | None = None
    editor_model_dir: str | None = None
    vae_model_dir: str | None = None
    purifier_model_path: str | None = None
    purifier_working_size: int = 256
    purifier_gradient_checkpointing: bool = True
    editor_kind: str = "flux"
    anchor_steps: int = 30
    anchor_guidance_scale: float = 2.5
    planner_max_new_tokens: int = 768
    eot_enabled: bool = True
    diffusion_weight: float = 0.25
    purifier_steps: int = 10

    def validate(self) -> None:
        """严格校验配置参数有效性，杜绝越界或无效数值。"""
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

        # 质量守门与正则化权重校验
        if not (0.25 <= self.source_ssim_floor < 1.0) or not (0.0 < self.source_mae_ceiling <= 128.0):
            raise ValueError("质量守门门槛无效")
        weights = [
            self.fg_l2_weight, self.fg_tv_weight, self.bg_l2_weight,
            self.bg_tv_weight, self.bg_chroma_tv_weight, self.overflow_weight,
            self.halo_soft_weight,
        ]
        if any(w < 0.0 for w in weights):
            raise ValueError("正则化权重不能为负数")
        if type(self.guard_backtracks) is not int or not (0 <= self.guard_backtracks <= 16):
            raise ValueError("回退守门步数无效")
        if type(self.log_every) is not int or self.log_every < 1:
            raise ValueError("日志周期必须为正整数")
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
