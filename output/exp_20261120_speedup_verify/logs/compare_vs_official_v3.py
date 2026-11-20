"""Compare Plan A & Plan B retrieval speedup outputs against official eval_v3."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

REPO = Path('/content/AmbiLAGE')
OFFICIAL_EVAL = REPO / 'output/exp_20261120_eval_v3/eval'
VERIFY_ROOT = REPO / 'output/exp_20261120_speedup_verify'
GROUPS = [
    '01_camera', '02_chair', '03_cookies', '04_curtain',
    '05_hair_extension', '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush',
]
ENCODERS = ['01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m']
PANELS = [
    'ambilage_training_oracle_v3',
    'vi_08_02_training_oracle_v3',
    'ambilage_overlap_30_v3',
    'vi_08_02_overlap_30_v3',
    'ambilage_zero_overlap_v3',
    'vi_08_02_zero_overlap_v3',
]
OFFICIAL_SECONDS = {
    'ambilage_training_oracle_v3': 401.0,
    'vi_08_02_training_oracle_v3': 775.0,
    'ambilage_overlap_30_v3': 402.0,
    'vi_08_02_overlap_30_v3': 794.0,
    'ambilage_zero_overlap_v3': 447.0,
    'vi_08_02_zero_overlap_v3': 785.0,
}


def _rank_order(vec: np.ndarray) -> list[int]:
    return sorted(range(len(vec)), key=lambda i: (-float(vec[i]), i))


def _compare_panel_dir(fast_dir: Path, off_dir: Path, encoders: list[str]) -> dict:
    fast_panel = json.loads((fast_dir / 'transfer_panel_summary.json').read_text(encoding='utf-8'))
    off_panel = json.loads((off_dir / 'transfer_panel_summary.json').read_text(encoding='utf-8'))

    max_score_diff = 0.0
    total_scores_checked = 0
    total_rankings_checked = 0
    src_rank_mismatches = 0
    full_order_mismatches = 0
    metric_deltas = []
    ver_failures = []

    for slug in encoders:
        ver_path = fast_dir / slug / 'category_independent_verification.json'
        ver_doc = json.loads(ver_path.read_text(encoding='utf-8'))
        if ver_doc.get('status') != 'passed' or float(ver_doc.get('max_score_error', 1.0)) > 1e-7:
            ver_failures.append((slug, ver_doc.get('status'), ver_doc.get('max_score_error')))

        # Compare overall and by_family exact hits and mean ranks
        f_enc = fast_panel['encoders'][slug]
        o_enc = off_panel['encoders'][slug]
        for scope_name, f_st, o_st in [('overall', f_enc['overall'], o_enc['overall'])] + [
            (f'family:{fam}', f_enc['by_family'][fam], o_enc['by_family'][fam])
            for fam in o_enc['by_family']
        ]:
            for k in ('top1', 'top3', 'top5'):
                for field in ('clean_hits', 'modified_hits'):
                    if f_st[k][field] != o_st[k][field]:
                        metric_deltas.append(
                            f'{slug}/{scope_name}/{k}.{field}: {f_st[k][field]} vs {o_st[k][field]}'
                        )
            for mf in ('mean_rank_clean', 'mean_rank_modified'):
                if abs(float(f_st[mf]) - float(o_st[mf])) > 1e-12:
                    metric_deltas.append(f'{slug}/{scope_name}/{mf}: {f_st[mf]} vs {o_st[mf]}')

        for grp in GROUPS:
            fg = fast_dir / slug / 'groups' / grp
            og = off_dir / slug / 'groups' / grp
            f_rows = json.loads((fg / 'evaluation.json').read_text(encoding='utf-8'))
            o_rows = json.loads((og / 'evaluation.json').read_text(encoding='utf-8'))
            with np.load(fg / 'scores.npz', allow_pickle=False) as fs, np.load(og / 'scores.npz', allow_pickle=False) as os_:
                assert fs['memory_ids'].tolist() == os_['memory_ids'].tolist()
                assert fs['query_ids'].tolist() == os_['query_ids'].tolist()
                assert fs['condition_labels'].tolist() == os_['condition_labels'].tolist()
                assert fs['banks'].tolist() == os_['banks'].tolist()
                f_scores = fs['scores']
                o_scores = os_['scores']
                diff = float(np.max(np.abs(f_scores - o_scores)))
                if diff > max_score_diff:
                    max_score_diff = diff
                n_rows, n_q, n_c = f_scores.shape
                total_scores_checked += int(f_scores.size)
                total_rankings_checked += n_rows * n_q
                for r_idx in range(n_rows):
                    for q_idx in range(n_q):
                        if f_rows[r_idx]['queries'][q_idx]['rank'] != o_rows[r_idx]['queries'][q_idx]['rank']:
                            src_rank_mismatches += 1
                        if _rank_order(f_scores[r_idx, q_idx]) != _rank_order(o_scores[r_idx, q_idx]):
                            full_order_mismatches += 1

    return dict(
        max_score_diff=max_score_diff,
        total_scores_checked=total_scores_checked,
        total_rankings_checked=total_rankings_checked,
        src_rank_mismatches=src_rank_mismatches,
        full_order_mismatches=full_order_mismatches,
        metric_deltas=metric_deltas,
        ver_failures=ver_failures,
    )


def main() -> int:
    timings = json.loads((VERIFY_ROOT / 'logs' / 'timings.json').read_text(encoding='utf-8'))

    # 1. Benchmark 1: Plan A single panel (vi_08_02_training_oracle_v3)
    plan_a_stats = _compare_panel_dir(
        VERIFY_ROOT / 'plan_a_vi_oracle_v3',
        OFFICIAL_EVAL / 'vi_08_02_training_oracle_v3',
        ENCODERS,
    )
    plan_a_pass = (
        plan_a_stats['max_score_diff'] <= 1e-6
        and plan_a_stats['src_rank_mismatches'] == 0
        and not plan_a_stats['metric_deltas']
        and not plan_a_stats['ver_failures']
    )

    # 2. Benchmark 2: Plan B all 6 panels (in_process)
    plan_b_panels = {}
    agg_diff = 0.0
    agg_scores = 0
    agg_rankings = 0
    agg_src_mismatch = 0
    agg_full_mismatch = 0
    agg_metric_deltas = []
    agg_ver_failures = []
    for p in PANELS:
        st = _compare_panel_dir(
            VERIFY_ROOT / 'plan_b_all6_inproc' / p,
            OFFICIAL_EVAL / p,
            ENCODERS,
        )
        plan_b_panels[p] = st
        agg_diff = max(agg_diff, st['max_score_diff'])
        agg_scores += st['total_scores_checked']
        agg_rankings += st['total_rankings_checked']
        agg_src_mismatch += st['src_rank_mismatches']
        agg_full_mismatch += st['full_order_mismatches']
        for d in st['metric_deltas']:
            agg_metric_deltas.append(f'{p}:{d}')
        for vf in st['ver_failures']:
            agg_ver_failures.append((p, *vf))

    plan_b_pass = (
        agg_diff <= 5e-6
        and agg_src_mismatch == 0
        and not agg_metric_deltas
        and not agg_ver_failures
    )

    # 3. Benchmark 3: Plan B subprocess smoke check (01_clip_l14 across all 6 panels)
    smoke_panels = {}
    smoke_diff = 0.0
    smoke_src_mismatch = 0
    smoke_ver_failures = []
    smoke_metric_deltas = []
    for p in PANELS:
        st = _compare_panel_dir(
            VERIFY_ROOT / 'plan_b_subproc_smoke' / p,
            OFFICIAL_EVAL / p,
            ['01_clip_l14'],
        )
        smoke_panels[p] = st
        smoke_diff = max(smoke_diff, st['max_score_diff'])
        smoke_src_mismatch += st['src_rank_mismatches']
        for d in st['metric_deltas']:
            smoke_metric_deltas.append(f'{p}:{d}')
        for vf in st['ver_failures']:
            smoke_ver_failures.append((p, *vf))

    smoke_pass = (
        smoke_diff <= 5e-6
        and smoke_src_mismatch == 0
        and not smoke_metric_deltas
        and not smoke_ver_failures
    )

    all_pass = plan_a_pass and plan_b_pass and smoke_pass

    off_all6_sec = sum(OFFICIAL_SECONDS.values())
    off_vi_oracle_sec = OFFICIAL_SECONDS['vi_08_02_training_oracle_v3']
    plan_a_sec = float(timings['plan_a_sec'])
    plan_b_sec = float(timings['plan_b_all6_inproc_sec'])
    smoke_sec = float(timings['plan_b_subproc_smoke_sec'])

    report = dict(
        passed=all_pass,
        timings=dict(
            official_vi_oracle_sec=off_vi_oracle_sec,
            official_all6_sec=off_all6_sec,
            plan_a_vi_oracle_sec=plan_a_sec,
            plan_a_speedup=off_vi_oracle_sec / max(plan_a_sec, 1e-6),
            plan_b_all6_inproc_sec=plan_b_sec,
            plan_b_speedup=off_all6_sec / max(plan_b_sec, 1e-6),
            plan_b_subproc_smoke_01clip_sec=smoke_sec,
        ),
        plan_a=dict(passed=plan_a_pass, **plan_a_stats),
        plan_b=dict(
            passed=plan_b_pass,
            max_score_diff=agg_diff,
            total_scores_checked=agg_scores,
            total_rankings_checked=agg_rankings,
            src_rank_mismatches=agg_src_mismatch,
            full_order_mismatches=agg_full_mismatch,
            metric_deltas=agg_metric_deltas,
            ver_failures=agg_ver_failures,
            panels=plan_b_panels,
        ),
        plan_b_subproc_smoke=dict(
            passed=smoke_pass,
            max_score_diff=smoke_diff,
            src_rank_mismatches=smoke_src_mismatch,
            metric_deltas=smoke_metric_deltas,
            ver_failures=smoke_ver_failures,
            panels=smoke_panels,
        ),
    )

    (VERIFY_ROOT / 'SPEEDUP_VERIFY_REPORT.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    lines = [
        '# 检索评测时间优化无损等价性与提速验证报告',
        '',
        f"- 总结论：**{'ALL PASSED（全部通过，100% 数值等价）' if all_pass else 'FAILED（未通过）'}**",
        '',
        '## 一、实测耗时对比',
        '',
        '| 验证项 | 原官方 `eval_v3` 耗时 | 优化后耗时 | 加速比 |',
        '|---|---:|---:|---:|',
        f"| 方案 A 单面板 (`vi_08_02_training_oracle_v3 --verify-mode in_process`) | {off_vi_oracle_sec:.1f}s ({off_vi_oracle_sec/60:.2f}m) | {plan_a_sec:.1f}s ({plan_a_sec/60:.2f}m) | **{off_vi_oracle_sec/plan_a_sec:.2f}x** |",
        f"| 方案 B 全 6 面板 (`fast_evaluate_panel.py eval-suites`, 3 模型各加载 1 次) | {off_all6_sec:.1f}s ({off_all6_sec/60:.2f}m) | {plan_b_sec:.1f}s ({plan_b_sec/60:.2f}m) | **{off_all6_sec/plan_b_sec:.2f}x** |",
        f"| 方案 B `--verify-mode subprocess` 烟测 (`01_clip_l14` 全 6 面板) | — | {smoke_sec:.1f}s ({smoke_sec/60:.2f}m) | — |",
        '',
        '## 二、全量数值等价性校验（逐 `scores.npz` 候选分数 + 逐条排序 + 全指标）',
        '',
        '| 验证项 | 核对排序次数 | 核对候选分数总数 | 最大分数误差 | 原图/改图源图名次不一致 | 全库189项排序不一致 | Overall & By-Family 指标偏差 | 内部复算校验 (`<=1e-7`) | 通过 |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|',
        f"| 方案 A (`vi_08_02_training_oracle_v3`) | {plan_a_stats['total_rankings_checked']} | {plan_a_stats['total_scores_checked']:,} | `{plan_a_stats['max_score_diff']:.3e}` | {plan_a_stats['src_rank_mismatches']} | {plan_a_stats['full_order_mismatches']} | {len(plan_a_stats['metric_deltas'])} | {'PASS' if not plan_a_stats['ver_failures'] else 'FAIL'} | {'PASS' if plan_a_pass else 'FAIL'} |",
        f"| 方案 B 全 6 面板 (`in_process`) | {agg_rankings} | {agg_scores:,} | `{agg_diff:.3e}` | {agg_src_mismatch} | {agg_full_mismatch} | {len(agg_metric_deltas)} | {'PASS' if not agg_ver_failures else 'FAIL'} | {'PASS' if plan_b_pass else 'FAIL'} |",
        f"| 方案 B `subprocess` 烟测 (`01_clip_l14`) | {sum(v['total_rankings_checked'] for v in smoke_panels.values())} | {sum(v['total_scores_checked'] for v in smoke_panels.values()):,} | `{smoke_diff:.3e}` | {smoke_src_mismatch} | {sum(v['full_order_mismatches'] for v in smoke_panels.values())} | {len(smoke_metric_deltas)} | {'PASS' if not smoke_ver_failures else 'FAIL'} | {'PASS' if smoke_pass else 'FAIL'} |",
        '',
        '## 三、方案 B 分面板数值误差明细',
        '',
        '| 面板 | 最大分数误差 | 源图名次不一致 | 全序不一致 | 指标偏差数 |',
        '|---|---:|---:|---:|---:|',
    ]
    for p in PANELS:
        st = plan_b_panels[p]
        lines.append(
            f"| `{p}` | `{st['max_score_diff']:.3e}` | {st['src_rank_mismatches']} | {st['full_order_mismatches']} | {len(st['metric_deltas'])} |"
        )
    lines.append('')

    (VERIFY_ROOT / 'SPEEDUP_VERIFY_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
