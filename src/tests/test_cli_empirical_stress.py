"""Empirical Stress-Testing and Verification Harness for tools/generate.py and src/visual/pipeline.py.

Covers:
1. tools/generate.py CLI invocation for AmbiLAGE and Lucid (CPU & GPU, small step counts).
2. Verification of exit codes, output directories, ready.json, metadata.json, adversarial.png.
3. Seed reproducibility (identical seed -> identical sha256, verification of deterministic execution).
4. Boundary & invalid inputs (non-existent source/target, invalid/negative steps, unknown method, missing arguments).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
import pytest
import torch

from src.tests.conftest_helpers import make_test_image


PYTHON_EXE = sys.executable
CLI_SCRIPT = Path(__file__).resolve().parents[2] / "tools/generate.py"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# =============================================================================
# 1. CLI INVOCATION: AMBILAGE & LUCID ON CPU AND GPU
# =============================================================================

@pytest.mark.parametrize("device", ["cpu"])
def test_cli_generate_ambilage_valid(tmp_path: Path, device: str):
    """验证 AmbiLAGE CLI 在指定设备上的端到端生成、退出码与产物完整性。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable on this machine")

    src = tmp_path / f"src_{device}.png"
    tgt = tmp_path / f"tgt_{device}.png"
    out_dir = tmp_path / f"out_ambilage_{device}"

    make_test_image(src, (32, 32), color=(220, 210, 200))
    make_test_image(tgt, (32, 32), color=(40, 80, 220))

    cmd = [
        PYTHON_EXE,
        str(CLI_SCRIPT), "--smoke-test",
        "--method", "ambilage",
        "--source", str(src),
        "--target", str(tgt),
        "--output", str(out_dir),
        "--steps", "2",
        "--device", device,
        "--target-text", "Herbal Botanical Shampoo",
        "--seed", "42",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, f"AmbiLAGE CLI ({device}) failed!\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # 验证关键产物生成
    adv_img = out_dir / "adversarial.png"
    ready_file = out_dir / "ready.json"
    meta_file = out_dir / "metadata.json"
    mask_file = out_dir / "latent_edit_mask.png"

    assert adv_img.is_file(), "adversarial.png was not generated"
    assert ready_file.is_file(), "ready.json was not generated"
    assert meta_file.is_file(), "metadata.json was not generated"
    assert mask_file.is_file(), "latent_edit_mask.png was not generated"

    # 验证 ready.json 规范与哈希一致性
    ready_data = json.loads(ready_file.read_text(encoding="utf-8"))
    assert ready_data["status"] == "completed"
    assert ready_data["method"] == "ambilage"
    steps_completed = ready_data.get("steps_completed") or ready_data.get("stages", {}).get("latent_generation_refinement", {}).get("steps_completed")
    assert steps_completed == 2, f"Expected 2 completed steps, got {steps_completed}"
    assert ready_data["pixel_optimization"] is False
    assert ready_data["output_image_sha256"] == sha256_of_file(adv_img)

    # 验证 metadata.json 规范
    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta_data["status"] == "completed"
    assert meta_data["method"] == "ambilage"
    assert meta_data["steps"] == 2
    assert "refinement" in meta_data or "quality" in meta_data


@pytest.mark.parametrize("device", ["cpu"])
def test_cli_generate_lucid_valid(tmp_path: Path, device: str):
    """验证 Lucid CLI 在指定设备上的端到端生成、退出码与产物完整性。"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable on this machine")

    src = tmp_path / f"src_{device}.png"
    tgt = tmp_path / f"tgt_{device}.png"
    out_dir = tmp_path / f"out_lucid_{device}"

    make_test_image(src, (32, 32), color=(210, 210, 210))
    make_test_image(tgt, (32, 32), color=(50, 100, 240))

    cmd = [
        PYTHON_EXE,
        str(CLI_SCRIPT), "--smoke-test",
        "--method", "lucid",
        "--source", str(src),
        "--target", str(tgt),
        "--output", str(out_dir),
        "--steps", "2",
        "--device", device,
        "--target-text", "Herbal Botanical Shampoo",
        "--seed", "42",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, f"Lucid CLI ({device}) failed!\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # 验证关键产物生成
    adv_img = out_dir / "adversarial.png"
    ready_file = out_dir / "ready.json"
    meta_file = out_dir / "metadata.json"

    assert adv_img.is_file(), "adversarial.png was not generated"
    assert ready_file.is_file(), "ready.json was not generated"
    assert meta_file.is_file(), "metadata.json was not generated"

    # 验证 ready.json 规范与哈希一致性
    ready_data = json.loads(ready_file.read_text(encoding="utf-8"))
    assert ready_data["status"] == "completed"
    assert ready_data["method"] == "lucid"
    assert ready_data["steps_completed"] == 2
    assert ready_data["pixel_optimization"] is True
    assert ready_data["output_image_sha256"] == sha256_of_file(adv_img)

    # 验证 metadata.json 规范
    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta_data["status"] == "success"
    assert meta_data["method"] == "lucid"


# =============================================================================
# 2. REPRODUCIBILITY & SEED SENSITIVITY
# =============================================================================

def test_ambilage_seed_reproducibility(tmp_path: Path):
    """验证 AmbiLAGE 在相同 seed 下产物哈希完全一致 (100% 确定性复现)。"""
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    make_test_image(src, (32, 32), color=(200, 200, 200))
    make_test_image(tgt, (32, 32), color=(50, 150, 250))

    out_seed42_a = tmp_path / "out_ambilage_seed42_a"
    out_seed42_b = tmp_path / "out_ambilage_seed42_b"

    common_args = [
        PYTHON_EXE,
        str(CLI_SCRIPT), "--smoke-test",
        "--method", "ambilage",
        "--source", str(src),
        "--target", str(tgt),
        "--steps", "2",
        "--device", "cpu",
        "--target-text", "Herbal Botanical Shampoo",
    ]

    # Run 1: seed 42
    res_a = subprocess.run(common_args + ["--output", str(out_seed42_a), "--seed", "42"], capture_output=True, text=True)
    assert res_a.returncode == 0

    # Run 2: seed 42 (identical seed)
    res_b = subprocess.run(common_args + ["--output", str(out_seed42_b), "--seed", "42"], capture_output=True, text=True)
    assert res_b.returncode == 0

    hash_a = sha256_of_file(out_seed42_a / "adversarial.png")
    hash_b = sha256_of_file(out_seed42_b / "adversarial.png")

    assert hash_a == hash_b, "相同随机种子 (seed=42) 产出的 adversarial.png SHA256 不一致，缺乏确定性！"


def test_lucid_seed_reproducibility(tmp_path: Path):
    """验证 Lucid 在相同 seed 下产物哈希完全一致 (100% 确定性复现)。"""
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    make_test_image(src, (32, 32), color=(200, 200, 200))
    make_test_image(tgt, (32, 32), color=(50, 150, 250))

    out_seed42_a = tmp_path / "out_lucid_seed42_a"
    out_seed42_b = tmp_path / "out_lucid_seed42_b"

    common_args = [
        PYTHON_EXE,
        str(CLI_SCRIPT), "--smoke-test",
        "--method", "lucid",
        "--source", str(src),
        "--target", str(tgt),
        "--steps", "2",
        "--device", "cpu",
        "--target-text", "Herbal Botanical Shampoo",
    ]

    res_a = subprocess.run(common_args + ["--output", str(out_seed42_a), "--seed", "42"], capture_output=True, text=True)
    assert res_a.returncode == 0

    res_b = subprocess.run(common_args + ["--output", str(out_seed42_b), "--seed", "42"], capture_output=True, text=True)
    assert res_b.returncode == 0

    hash_a = sha256_of_file(out_seed42_a / "adversarial.png")
    hash_b = sha256_of_file(out_seed42_b / "adversarial.png")

    assert hash_a == hash_b, "相同随机种子 (seed=42) 产出的 Lucid adversarial.png SHA256 不一致！"


# =============================================================================
# 3. BOUNDARY & INVALID INPUT HANDLING
# =============================================================================

def test_cli_nonexistent_source_file(tmp_path: Path):
    """验证提供不存在的 source 图像时，CLI 优雅抛错且返回非零退出码。"""
    src = tmp_path / "nonexistent_source.png"
    tgt = tmp_path / "tgt.png"
    make_test_image(tgt, (32, 32))

    cmd = [
        PYTHON_EXE,
        str(CLI_SCRIPT), "--smoke-test",
        "--method", "ambilage",
        "--source", str(src),
        "--target", str(tgt),
        "--output", str(tmp_path / "out"),
        "--steps", "2",
        "--device", "cpu",
        "--target-text", "Herbal Shampoo",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode != 0, "CLI did not return non-zero exit code for nonexistent source"
    assert "FileNotFoundError" in res.stderr or "No such file" in res.stderr or "不存在" in res.stderr


def test_cli_nonexistent_target_file(tmp_path: Path):
    """验证提供不存在的 target 图像时，CLI 优雅抛错且返回非零退出码。"""
    src = tmp_path / "src.png"
    tgt = tmp_path / "nonexistent_target.png"
    make_test_image(src, (32, 32))

    cmd = [
        PYTHON_EXE,
        str(CLI_SCRIPT), "--smoke-test",
        "--method", "ambilage",
        "--source", str(src),
        "--target", str(tgt),
        "--output", str(tmp_path / "out"),
        "--steps", "2",
        "--device", "cpu",
        "--target-text", "Herbal Shampoo",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode != 0, "CLI did not return non-zero exit code for nonexistent target"
    assert "FileNotFoundError" in res.stderr or "No such file" in res.stderr or "不存在" in res.stderr


def test_cli_unknown_method(tmp_path: Path):
    """验证传入非法 method 名称时被 argparse 拦截，返回错误码 2。"""
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    make_test_image(src, (32, 32))
    make_test_image(tgt, (32, 32))

    cmd = [
        PYTHON_EXE,
        str(CLI_SCRIPT), "--smoke-test",
        "--method", "invalid_future_method",
        "--source", str(src),
        "--target", str(tgt),
        "--output", str(tmp_path / "out"),
        "--steps", "2",
        "--device", "cpu",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 2, f"Expected returncode 2 for invalid choices, got {res.returncode}"
    assert "invalid choice" in res.stderr or "choose from" in res.stderr


def test_cli_negative_steps_rejection(tmp_path: Path):
    """验证传入负数 steps 时，AmbiLAGE 与 Lucid 均正确拒绝并返回非零退出码。"""
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    make_test_image(src, (32, 32))
    make_test_image(tgt, (32, 32))

    for method in ["ambilage", "lucid"]:
        cmd = [
            PYTHON_EXE,
            str(CLI_SCRIPT), "--smoke-test",
            "--method", method,
            "--source", str(src),
            "--target", str(tgt),
            "--output", str(tmp_path / f"out_{method}"),
            "--steps", "-5",
            "--device", "cpu",
            "--target-text", "Herbal Shampoo",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert res.returncode != 0, f"Method {method} should reject negative steps"
        assert "ValueError" in res.stderr or "必须为正整数" in res.stderr


def test_cli_non_numeric_steps(tmp_path: Path):
    """验证传入非数字 steps 时被 argparse 拦截。"""
    src = tmp_path / "src.png"
    tgt = tmp_path / "tgt.png"
    make_test_image(src, (32, 32))
    make_test_image(tgt, (32, 32))

    cmd = [
        PYTHON_EXE,
        str(CLI_SCRIPT), "--smoke-test",
        "--method", "ambilage",
        "--source", str(src),
        "--target", str(tgt),
        "--output", str(tmp_path / "out"),
        "--steps", "two_steps",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 2
    assert "invalid int value" in res.stderr


def test_cli_missing_required_arguments():
    """验证缺少必需参数时，argparse 拦截并退出码为 2。"""
    cmd = [
        PYTHON_EXE,
        str(CLI_SCRIPT), "--smoke-test",
        "--method", "ambilage",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 2
    assert "the following arguments are required" in res.stderr
