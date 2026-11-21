"""Project-local paths; no dependency on the migration source or its outputs."""
from __future__ import annotations
import json
import os
from pathlib import Path as _Path

PROJECT = _Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "src"
OUTPUT = PROJECT / "output"
DATA = PROJECT / "data"
RESOURCES = PROJECT / "resources"
EVALUATION_BASE = RESOURCES / "evaluation_base"
CONFIGS = SOURCE / "agent/config"
PROMPTS = SOURCE / "agent/prompts/templates"
VISUAL_CONFIGS = SOURCE / "visual/config"
EXPERIMENT_CONFIGS = SOURCE / "experiments/config"
EVALUATION_BASES_CONFIG = EXPERIMENT_CONFIGS / "evaluation_bases.json"
PUBLIC = OUTPUT / "datasets/public"
PRIVATE = OUTPUT / "datasets/private"
SHARED_MEMORY = EVALUATION_BASE / "shared_memory/shared_memory.json"
EXPERIMENT_BASE_GROUPS9 = EVALUATION_BASE
_model_root = _Path(os.environ.get("AMBILAGE_MODEL_ROOT", "models")).expanduser()
MODEL_ROOT = (_model_root if _model_root.is_absolute() else PROJECT / _model_root).resolve()
ProjectPath = _Path


def model_path(name):
    return str(MODEL_ROOT / name)


def relocate_project_path(value):
    return os.fspath(value) if isinstance(value, (str, os.PathLike)) else value


def relocate_io_path(value):
    return relocate_project_path(value)


def load_evaluation_bases(config_path=None):
    path = ProjectPath(config_path).expanduser().resolve() if config_path is not None else EVALUATION_BASES_CONFIG
    if not path.is_file():
        return {"schema_version": 1, "default_base": "original_9groups", "bases": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_evaluation_base(base=None, *, config_path=None):
    """将配置文件名、环境变量 AMBILAGE_EVALUATION_BASE 或路径动态解析为实验基础包目录。"""
    doc = load_evaluation_bases(config_path)
    bases = doc.get("bases") or {}
    raw = base
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        env_val = os.environ.get("AMBILAGE_EVALUATION_BASE", "").strip()
        raw = env_val if env_val else doc.get("default_base")
        if not raw:
            return EVALUATION_BASE.resolve()
    key = os.fspath(raw).strip() if isinstance(raw, (str, os.PathLike)) else None
    if key and key in bases:
        target = ProjectPath(bases[key]["path"]).expanduser()
        return (target if target.is_absolute() else PROJECT / target).resolve()
    candidate = ProjectPath(raw).expanduser()
    if not candidate.is_absolute() and not candidate.exists() and (PROJECT / candidate).exists():
        return (PROJECT / candidate).resolve()
    return candidate.resolve()


def public_root_for(value, *, extra_roots=()):
    path = ProjectPath(value).resolve()
    for root in (*extra_roots, DATA, RESOURCES, EVALUATION_BASE, PUBLIC):
        root = ProjectPath(root).resolve()
        if path.is_relative_to(root):
            return root
    if path.is_relative_to(OUTPUT):
        parts = path.relative_to(OUTPUT).parts
        if "data" in parts:
            return OUTPUT / _Path(*parts[:parts.index("data") + 1])
    raise ValueError("Public input must be inside the project data or run data directory: " + str(value))
