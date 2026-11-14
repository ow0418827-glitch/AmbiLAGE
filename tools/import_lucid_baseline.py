"""将已完成的原始像素基线导入当前项目，不运行模型或检索。"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT
from src.experiments.lucid_baseline_import import import_baselines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=PROJECT / 'output/imported_lucid_baseline_20261113')
    args = parser.parse_args(argv)
    manifest = import_baselines(args.source_project, PROJECT, args.output)
    print(f"已校验并导入 {len(manifest['groups'])} 组完整基线，复制 {len(manifest['files'])} 份原字节文件；没有训练、编码或检索。")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
