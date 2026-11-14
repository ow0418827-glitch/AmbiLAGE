"""Offline tests for deployment safety and portable resource binding."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.paths import PROJECT
from tools.preflight import bounded_path, check_resources
from tools.package_source import source_files


def test_private_resource_bindings_after_sync():
    if not (PROJECT / "resources/evaluation_base/shared_memory/shared_memory.json").is_file():
        pytest.skip("Private evaluation resources have not been synchronized")
    result = check_resources()
    assert (result["entries"], result["groups"], result["queries"], result["shared_image_files"]) == (189, 9, 180, 198)
    assert result["model_encoding_performed"] is False
    assert result["selection_exceptions"] == ["04_curtain", "08_speaker"]


def test_resource_paths_reject_escape_and_foreign_drive(tmp_path):
    (tmp_path / "inside.json").write_text("{}", encoding="utf-8")
    assert bounded_path(tmp_path, "inside.json").is_file()
    for value in ("../outside.json", "D:/old-project/input.json"):
        with pytest.raises(ValueError):
            bounded_path(tmp_path, value)


def test_relative_model_root_does_not_depend_on_working_directory(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(PROJECT), AMBILAGE_MODEL_ROOT="models/test")
    output = subprocess.check_output([sys.executable, "-X", "utf8", "-c", "from src.paths import MODEL_ROOT; print(MODEL_ROOT)"], cwd=tmp_path, env=env, text=True, encoding="utf-8")
    assert Path(output.strip()) == PROJECT / "models/test"


def test_archive_excludes_private_and_runtime_state(tmp_path):
    examples = {
        "README.md": "portable source",
        "src/module.py": "value = 1",
        "src/config/llm.local.json": "{}",
        "src/__pycache__/module.pyc": "cache",
        "resources/input.json": "{}",
        "resources/model.safetensors": "weights",
        "output/result.json": "private runtime",
        ".env": "private",
    }
    for relative, content in examples.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    assert {p.relative_to(tmp_path).as_posix() for p in source_files(tmp_path)} == {"README.md", "src/module.py", "resources/input.json"}


def test_archive_blocks_embedded_credentials_without_echoing_them(tmp_path):
    path = tmp_path / "src/config.json"
    path.parent.mkdir()
    token = "sk-" + "examplecredentialvalue123456"
    path.write_text(json.dumps({"api_key": token}), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        source_files(tmp_path)
    assert token not in str(error.value)


def test_model_plan_is_explicit_and_does_not_import_downloader():
    code = "import runpy,sys; sys.argv=['prepare_models.py','--scope','all']; runpy.run_path('tools/prepare_models.py',run_name='__main__')"
    output = subprocess.check_output([sys.executable, "-X", "utf8", "-c", code], cwd=PROJECT, text=True, encoding="utf-8")
    plan = json.loads(output)
    assert plan["download"] is False
    by_key = {item["key"]: item for item in plan["models"]}
    keys = set(by_key)
    assert {"sam2", "editor", "purifier", "eval_g14", "eval_gme"}.issubset(keys)
    assert not {"planner", "planner_7b", "editor_qwen"} & keys
    assert by_key["editor"]["role"] == "stage4_16_channel_flux_vae"
    assert by_key["editor"]["allow_patterns"] == ["vae/*", "model_index.json", "*.md", "LICENSE*"]
    assert {"clip_b16", "clip_b32", "clip_b32_laion"}.issubset(keys)
    assert not {"clip_l14", "siglip_so400m", "clip_h14"} & keys


def test_choice1_requirements_exclude_deprecated_vlm_and_unused_packages():
    import re
    def _pkg_names(path: Path) -> set[str]:
        names = set()
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("-r "):
                continue
            names.add(re.split(r"[<>=!~;\s\[]", line, maxsplit=1)[0].lower())
        return names

    base_pkgs = _pkg_names(PROJECT / "requirements/requirements.txt")
    optional_pkgs = _pkg_names(PROJECT / "requirements/requirements-optional-vlm.txt")
    assert not {"qwen-vl-utils", "lpips", "matplotlib", "psutil"} & base_pkgs
    assert {"diffusers", "rapidocr-onnxruntime", "open-clip-torch", "scikit-image", "scipy", "sentence-transformers"} <= base_pkgs
    assert "qwen-vl-utils" in optional_pkgs


def test_doctor_marks_optional_models_and_editor_vae_subset(capsys):
    from src.experiments.runtime_check import doctor
    report = doctor(device="cpu")
    capsys.readouterr()
    assert report["local_models"]["planner"]["optional"] is True
    assert report["local_models"]["planner"]["required"] is False
    assert report["local_models"]["editor"]["optional"] is False
    assert report["local_models"]["editor"]["allow_patterns"] == ["vae/*", "model_index.json", "*.md", "LICENSE*"]
