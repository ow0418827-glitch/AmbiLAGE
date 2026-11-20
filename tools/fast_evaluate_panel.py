"""通用高速检索迁移评估入口（支持多方法 × 多查询套单模型加载批量预热）。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.defenses import DEFENSE_MODES
from src.experiments.fast_stage2_panel import VERIFY_MODES, main as fast_panel_main, run_encoder_grouped_panels
from src.experiments.visual_inception_category_evaluation import (
    DEFAULT_KEY_MODE,
    KEY_MODES,
    QUERY_SUITE_ROOTS,
    TRANSFER_PANEL_ENCODERS,
)

ENCODER_SLUGS = tuple(row[0] for row in TRANSFER_PANEL_ENCODERS)
from src.paths import EVALUATION_BASE, OUTPUT
from src.visual.audit import read_json
from src.visual.optimization import METHODS, normalize_method


def _parse_spec(raw: str) -> tuple[str, str, Path]:
    parts = raw.split(':', 2)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise argparse.ArgumentTypeError(f'规格必须为 TAG:METHOD:IMAGES_ROOT：{raw}')
    tag, method, root = parts[0].strip(), normalize_method(parts[1].strip()), Path(parts[2].strip()).resolve()
    if method not in METHODS:
        raise argparse.ArgumentTypeError(f'不支持的方法 {method}（可选：{METHODS}）')
    if not root.is_dir():
        raise argparse.ArgumentTypeError(f'成图根目录不存在：{root}')
    return tag, method, root


def _eval_suites_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='fast_evaluate_panel.py eval-suites',
        description='按编码器外循环批量评估多方法 × 多查询套（支持 --verify-mode in_process / subprocess）。',
    )
    p.add_argument(
        '--spec',
        action='append',
        required=True,
        help='可重复指定待评测对象，格式为 TAG:METHOD:IMAGES_ROOT',
    )
    p.add_argument(
        '--suites',
        nargs='+',
        choices=tuple(QUERY_SUITE_ROOTS),
        required=True,
        help='一个或多个冻结查询套（如 training_oracle_v4 overlap_30_v4 zero_overlap_v4）',
    )
    p.add_argument('--output-dir', type=Path, required=True, help='各面板子目录的输出根目录（须位于 output/ 内）')
    p.add_argument('--base', type=Path, default=EVALUATION_BASE)
    p.add_argument('--key-mode', choices=KEY_MODES, default=DEFAULT_KEY_MODE)
    p.add_argument('--defense-mode', choices=DEFENSE_MODES, default='none')
    p.add_argument('--defended-cache-dir', type=Path, default=None)
    p.add_argument('--memory-vectors-dir', type=Path, default=None)
    p.add_argument('--group', default='all', help='all 或逗号分隔分组（如 02_chair,05_hair_extension）')
    p.add_argument('--replacement-groups', choices=('all', 'selected'), default='all')
    p.add_argument('--batch-size', type=int, default=16, help='默认图像批量前向大小')
    p.add_argument('--batch-size-map', type=str, default=None, help='可选按编码器覆盖的 JSON（覆盖 --batch-size）')
    p.add_argument('--text-batch-size', type=int, default=64, help='文本批量前向大小')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--verify-mode', choices=VERIFY_MODES, default='in_process')
    p.add_argument('--encoder-slugs', nargs='+', default=None)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--skip-training-snapshots', action='store_true')
    p.add_argument('--query-root', type=Path, default=None, help='仅在单 suite 时显式覆盖 query_root')
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.update(
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',
    )
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] in ('-h', '--help'):
        print(
            '用法:\n'
            '  fast_evaluate_panel.py eval-suites [选项]   # 通用多方法 × 多查询套高速评测（默认 in_process 全量复算）\n'
            '  fast_evaluate_panel.py run [选项]           # fast_stage2_panel 作业表模式\n'
            '  fast_evaluate_panel.py finalize [选项]      # 仅汇总生成报告\n'
            '使用 `eval-suites --help` 查看详细参数。'
        )
        return 0

    if args_list[0] != 'eval-suites':
        return int(fast_panel_main(args_list) or 0)

    args = _eval_suites_parser().parse_args(args_list[1:])
    specs = [_parse_spec(raw) for raw in args.spec]
    out_dir = Path(args.output_dir).resolve()
    if not out_dir.is_relative_to(Path(OUTPUT).resolve()):
        raise ValueError('评估输出根目录必须位于当前项目 output 目录内')
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_groups = None if args.group == 'all' else [g.strip() for g in str(args.group).split(',') if g.strip()]
    if args.query_root is not None and len(args.suites) != 1:
        raise ValueError('显式指定 --query-root 时只能传一个 suite')

    jobs = []
    for suite in args.suites:
        for tag, method, images_root in specs:
            name = f'{tag}_{suite}'
            job = dict(
                name=name,
                images_root=str(images_root),
                method=method,
                query_suite=suite,
                output=str(out_dir / name),
                groups=selected_groups,
                replacement_groups=args.replacement_groups,
            )
            if args.query_root is not None:
                job['query_root'] = str(args.query_root.resolve())
            jobs.append(job)

    batch_sizes = {slug: int(args.batch_size) for slug in ENCODER_SLUGS}
    if args.batch_size_map:
        override = read_json(args.batch_size_map) if Path(args.batch_size_map).is_file() else json.loads(args.batch_size_map)
        batch_sizes.update({str(k): int(v) for k, v in override.items()})

    defended_dir = Path(args.defended_cache_dir).resolve() if args.defended_cache_dir else (out_dir / '_defended_cache')
    vec_dir = Path(args.memory_vectors_dir).resolve() if args.memory_vectors_dir else (out_dir / '_memory_vectors')

    results = run_encoder_grouped_panels(
        jobs,
        base=args.base,
        key_mode=args.key_mode,
        defense_mode=args.defense_mode,
        defended_cache_dir=defended_dir,
        memory_vector_cache_dir=vec_dir,
        batch_sizes=batch_sizes,
        text_batch_size=args.text_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        verify_training_snapshots=not args.skip_training_snapshots,
        verify_mode=args.verify_mode,
        encoder_slugs=args.encoder_slugs,
    )
    summary_index = {
        name: {
            slug: enc_data['overall']
            for slug, enc_data in panel['encoders'].items()
        }
        for name, panel in results.items()
    }
    print(json.dumps(summary_index, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
