#!/usr/bin/env python3
"""汇总 Stage 2.3 九组 v2 评测结果：干净基线齐平度、单调衰减、三编码器主表及子家族 Top-3/Top-5 明细。"""
from __future__ import annotations

import json
from pathlib import Path

EXP_DIR = Path('/content/AmbiLAGE/output/exp_20261120_eval_v2')
SUITES = ('training_oracle_v2', 'overlap_30_v2', 'zero_overlap_v2')
ENCODERS = (
    ('01_clip_l14', 'CLIP-L/14 (白盒)'),
    ('02_openclip_bigG14', 'OpenCLIP-bigG-14 (黑盒)'),
    ('03_siglip_so400m', 'SigLIP-SO400M (黑盒)'),
)
METHODS = (
    ('ambilage', 'AmbiLAGE (基线)'),
    ('vi_08_02', 'VI-Enhanced (0.8/0.2)'),
)


def pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def main() -> None:
    loaded: dict[tuple[str, str], dict] = {}
    for m_key, _ in METHODS:
        for suite in SUITES:
            p = EXP_DIR / 'eval' / f'{m_key}_{suite}' / 'transfer_panel_summary.json'
            if p.is_file():
                loaded[(m_key, suite)] = json.loads(p.read_text(encoding='utf-8'))

    lines = [
        '# Visual Inception v2 评测集 9 组全面对比报告（Stage 2.3）',
        '',
        '## 一、干净原图基线对比（验证难度倒挂消除）',
        '',
        '| 评测集 | CLIP-L/14 Clean T1 / T3 / T5 / 均名 | OpenCLIP-bigG-14 Clean T1 / T3 / T5 / 均名 | SigLIP-SO400M Clean T1 / T3 / T5 / 均名 |',
        '|---|---:|---:|---:|',
    ]
    ref_method = 'vi_08_02' if ('vi_08_02', SUITES[0]) in loaded else 'ambilage'
    for suite in SUITES:
        panel = loaded.get((ref_method, suite))
        if panel is None:
            continue
        cells = []
        for eslug, _ in ENCODERS:
            ov = panel['encoders'][eslug]['overall']
            cells.append(
                f"{pct(ov['top1']['clean_rate'])} / {pct(ov['top3']['clean_rate'])} / "
                f"{pct(ov['top5']['clean_rate'])} / {ov['mean_rank_clean']:.2f}"
            )
        lines.append(f'| `{suite}` | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '## 二、修改图整体命中率对比（Top-1 / Top-3 / Top-5 / 平均名次）',
        '',
        '| 方法 | 评测集 | CLIP-L/14 (白盒) T1 / T3 / T5 / 均名 | OpenCLIP-bigG-14 (黑盒) T1 / T3 / T5 / 均名 | SigLIP-SO400M (黑盒) T1 / T3 / T5 / 均名 | Mean BB T3 / T5 |',
        '|---|---|---:|---:|---:|---:|',
    ]

    mono_rows = []
    for m_key, m_label in METHODS:
        suite_t3 = {e[0]: [] for e in ENCODERS}
        suite_t5 = {e[0]: [] for e in ENCODERS}
        for suite in SUITES:
            panel = loaded.get((m_key, suite))
            if panel is None:
                continue
            cells = []
            bb_t3, bb_t5 = [], []
            for eslug, _ in ENCODERS:
                ov = panel['encoders'][eslug]['overall']
                t1, t3, t5 = ov['top1']['modified_rate'], ov['top3']['modified_rate'], ov['top5']['modified_rate']
                suite_t3[eslug].append(t3)
                suite_t5[eslug].append(t5)
                if eslug != '01_clip_l14':
                    bb_t3.append(t3)
                    bb_t5.append(t5)
                cells.append(f"{pct(t1)} / {pct(t3)} / {pct(t5)} / {ov['mean_rank_modified']:.2f}")
            mean_bb = f"{pct(sum(bb_t3)/len(bb_t3))} / {pct(sum(bb_t5)/len(bb_t5))}"
            lines.append(f'| {m_label} | `{suite}` | ' + ' | '.join(cells) + f' | {mean_bb} |')
        for eslug, elabel in ENCODERS:
            t5s = suite_t5[eslug]
            t3s = suite_t3[eslug]
            if len(t5s) == 3:
                mono_t5 = t5s[0] >= t5s[1] >= t5s[2]
                mono_t3 = t3s[0] >= t3s[1] >= t3s[2]
                mono_rows.append(
                    f'| {m_label} | {elabel} | '
                    f"{pct(t3s[0])} -> {pct(t3s[1])} -> {pct(t3s[2])} ({'是' if mono_t3 else '否'}) | "
                    f"{pct(t5s[0])} -> {pct(t5s[1])} -> {pct(t5s[2])} ({'是' if mono_t5 else '否'}) |"
                )

    lines += [
        '',
        '## 三、难度单调性检验（`training_oracle_v2 >= overlap_30_v2 >= zero_overlap_v2`）',
        '',
        '| 方法 | 编码器 | Top-3 序列 (是否单调) | Top-5 序列 (是否单调) |',
        '|---|---|---|---|',
        *mono_rows,
        '',
        '## 四、按查询子家族细分（Top-3 / Top-5 / 均名）',
    ]

    for suite in SUITES:
        panel_ref = loaded.get(('vi_08_02', suite)) or loaded.get(('ambilage', suite))
        if panel_ref is None:
            continue
        fams = list(panel_ref['encoders']['01_clip_l14']['by_family'])
        lines += [
            '',
            f'### `{suite}`',
            '',
            '| 方法 | 子家族 | CLIP-L/14 T3 / T5 / 均名 | OpenCLIP-bigG-14 T3 / T5 / 均名 | SigLIP-SO400M T3 / T5 / 均名 |',
            '|---|---|---:|---:|---:|',
        ]
        for m_key, m_label in METHODS:
            panel = loaded.get((m_key, suite))
            if panel is None:
                continue
            for fam in fams:
                zh = panel['encoders']['01_clip_l14']['by_family'][fam].get('query_family_zh', fam)
                cells = []
                for eslug, _ in ENCODERS:
                    st = panel['encoders'][eslug]['by_family'][fam]
                    cells.append(
                        f"{pct(st['top3']['modified_rate'])} / {pct(st['top5']['modified_rate'])} / {st['mean_rank_modified']:.2f}"
                    )
                lines.append(f'| {m_label} | {zh} (`{fam}`) | ' + ' | '.join(cells) + ' |')

    out_md = EXP_DIR / 'EVAL_V2_SUMMARY.md'
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    summary_json = {
        f'{m}_{s}': {
            es: {
                'overall': loaded[(m, s)]['encoders'][es]['overall'],
                'by_family': loaded[(m, s)]['encoders'][es]['by_family'],
            }
            for es, _ in ENCODERS
        }
        for m, _ in METHODS
        for s in SUITES
        if (m, s) in loaded
    }
    (EXP_DIR / 'EVAL_V2_SUMMARY.json').write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'[SUMMARY WRITTEN] {out_md}')


if __name__ == '__main__':
    main()
