"""Frozen configuration for the semantic/perceptual adaptation."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class VisualInceptionConfig:
    steps: int = 100
    epsilon_levels: int = 8
    step_size_levels: float = 2
    semantic_weight: float = 1.0
    perceptual_weight: float = 0.1
    gradient_restarts: int = 10
    seed: int = 42
    surrogate_set: str = 'visual_inception_paper'
    surrogate_config: str | None = None
    lpips_backbone_weights: str | None = None
    lpips_linear_weights: str | None = None
    smoke_test: bool = False
    objective_mode: str = 'retrieval_semantic_perceptual'
    training_manifest: str | None = None
    retrieval_weight: float = 1.0
    retrieval_temperature: float = 0.07
    query_batch_size: int = 16
    candidate_cache_dir: str | None = None

    def validate(self):
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError('视觉记忆植入优化步数必须为正整数')
        if type(self.epsilon_levels) is not int or not 1 <= self.epsilon_levels <= 255:
            raise ValueError('像素扰动预算必须为一至二百五十五之间的整数')
        for name in ('step_size_levels', 'semantic_weight', 'perceptual_weight', 'retrieval_weight', 'retrieval_temperature'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + ' 必须为有限正数')
        if type(self.gradient_restarts) is not int or self.gradient_restarts < 2:
            raise ValueError('梯度方差估计至少需要两个独立随机起点')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('随机种子必须为有效非负整数')
        if type(self.smoke_test) is not bool:
            raise ValueError('轻量夹具开关必须是布尔值')
        if self.smoke_test and self.steps > 3:
            raise ValueError('轻量夹具最多执行三步')
        if self.objective_mode not in ('semantic_perceptual', 'retrieval_semantic_perceptual'):
            raise ValueError('未知视觉记忆植入训练目标模式')
        if type(self.query_batch_size) is not int or not 1 <= self.query_batch_size <= 100:
            raise ValueError('训练查询批量必须为一至一百之间的整数')
        if self.surrogate_set not in ('lucid', 'heterogeneous4', 'visual_inception_paper', 'custom'):
            raise ValueError('未知训练代理方案')
        if bool(self.surrogate_config) != (self.surrogate_set == 'custom'):
            raise ValueError('自定义代理方案与配置文件必须同时指定')
        for name in ('surrogate_config', 'lpips_backbone_weights', 'lpips_linear_weights',
                     'training_manifest', 'candidate_cache_dir'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(name + ' 必须为非空路径字符串')
        return self

    @classmethod
    def from_dict(cls, value):
        if value is None:
            return cls().validate()
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError('视觉记忆植入配置含未知字段')
        return cls(**value).validate()

    def to_dict(self):
        return asdict(self)
