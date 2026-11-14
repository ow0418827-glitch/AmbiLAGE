"""统一视觉生成入口与 CLI 工具集成测试。"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import pytest

from src.visual.optimization import METHODS, normalize_method
from src.visual.pipeline import GenerationResult, generate
from src.tests.conftest_helpers import make_test_image


def test_normalize_method():
    """测试方法名称规范化与非法方法拦截。"""
    assert normalize_method("ambilage") == "ambilage"
    assert normalize_method("AMBILAGE") == "ambilage"
    assert normalize_method("lucid") == "lucid"
    assert normalize_method("LUCID") == "lucid"

    with pytest.raises(ValueError, match="方法必须是 'lucid' 或 'ambilage'"):
        normalize_method("unknown_method")

    with pytest.raises(ValueError, match="方法名称必须为字符串"):
        normalize_method(None)


def test_unified_generate_ambilage(tmp_path: Path):
    """测试统一 generate() 接口生成 ambilage 对抗样本。"""
    src_p = tmp_path / "source.png"
    tgt_p = tmp_path / "target.png"
    out_dir = tmp_path / "unified_run"

    make_test_image(src_p, (32, 32), color=(200, 200, 200))
    make_test_image(tgt_p, (32, 32), color=(100, 150, 250))

    result = generate(
        method="ambilage",
        source=src_p,
        target=tgt_p,
        output_dir=out_dir,
        steps=2,
        device="cpu",
        target_text="Blue Herbal Shampoo",smoke_test=True,
    )

    # 验证返回对象规范
    assert isinstance(result, GenerationResult)
    assert result.method == "ambilage"
    assert result.image_path.is_file()
    assert result.ready_path.is_file()
    assert result.metadata_path.is_file()
    assert len(result.sha256) == 64
    assert result.metadata["status"] == "success"


def test_cli_generate_subprocess(tmp_path: Path):
    """测试 tools/generate.py 命令行端到端调用。"""
    src_p = tmp_path / "cli_src.png"
    tgt_p = tmp_path / "cli_tgt.png"
    out_dir = tmp_path / "cli_output"

    make_test_image(src_p, (32, 32), color=(220, 220, 220))
    make_test_image(tgt_p, (32, 32), color=(50, 80, 210))

    cli_script = Path(__file__).resolve().parents[2] / "tools/generate.py"

    cmd = [
        sys.executable,
        str(cli_script), "--smoke-test",
        "--method", "ambilage",
        "--source", str(src_p),
        "--target", str(tgt_p),
        "--output", str(out_dir),
        "--steps", "2",
        "--device", "cpu",
        "--target-text", "CLI Herbal Test",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert res.returncode == 0, f"CLI 执行失败: STDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
    assert (out_dir / "adversarial.png").is_file()
    assert (out_dir / "ready.json").is_file()
    assert (out_dir / "metadata.json").is_file()
