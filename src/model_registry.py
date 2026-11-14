"""Model identities shared by download preparation and production loaders."""
from __future__ import annotations
import json
from pathlib import Path
from src.paths import MODEL_ROOT, RESOURCES, PROJECT


def model_specs():
    return json.loads((RESOURCES / "model_manifest.json").read_text(encoding="utf-8"))["models"]


def resolve_model(key_or_repo: str, *, allow_downloads: bool = False) -> str:
    specs = model_specs()
    key = key_or_repo if key_or_repo in specs else next(
        (name for name, spec in specs.items() if spec["repo_id"] == key_or_repo), None)
    path = MODEL_ROOT / key if key is not None else Path(key_or_repo).expanduser()
    if not path.is_absolute():
        path = PROJECT / path
    if path.exists():
        return str(path.resolve())
    if allow_downloads and key is not None:
        return specs[key]["repo_id"]
    if allow_downloads and "/" in key_or_repo:
        return key_or_repo
    raise FileNotFoundError(
        f"Missing local model {key_or_repo!r} at {path}; prepare it with tools/prepare_models.py on the server.")
