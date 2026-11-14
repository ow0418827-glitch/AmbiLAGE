"""AmbiLAGE: 原生环境光背景协同调整与主体微光影生成编辑核心包。"""
from .config import AmbiLAGEConfig, METHOD, DEFINITION, MASK_POLICY
from .pipeline import AmbiLAGEPipeline, generate_ambilage, run_attack

__all__ = [
    "AmbiLAGEConfig",
    "AmbiLAGEPipeline",
    "generate_ambilage",
    "run_attack",
    "METHOD",
    "DEFINITION",
    "MASK_POLICY",
]
