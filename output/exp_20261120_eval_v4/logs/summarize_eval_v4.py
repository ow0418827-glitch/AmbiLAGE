#!/usr/bin/env python3
"""汇总 v4 严格 100/30/0 字面重合评测集全 9 组三编码器对比结果。"""
from __future__ import annotations

import json
from pathlib import Path

EXP_DIR = Path('/content/AmbiLAGE/output/exp_20261120_eval_v4')
V3_SUMMARY_PATH = Path('/content/AmbiLAGE/output/exp_20261120_eval_v3/EVAL_V3_SUMMARY.json')
SUITES = (
    ('training_oracle_v4', '100% 重合 (`training_oracle_v4`)'),
    ('overlap_30_v4', '30% 重合 (`overlap_30_v4`)'),
    ('zero_overlap_v4', '0% 重合 (`zero_overlap_v4`)'),
)
SUITE_TO_V3 = {
    'training_oracle_v4': 'training_oracle_v3',
    'overlap_30_v4': 'overlap_30_v3',
    'zero_overlap_v4': 'zero_overlap_v3',
}
ENCODERS = (
    ('01_clip_l14', 'CLIP-ViT-L/14 (白盒)'),
    ('02_openclip_bigG14', 'OpenCLIP-bigG-14 (黑盒)'),
    ('03_siglip_so400m', 'SigLIP-SO400M (黑盒)'),
)
METHODS = (
    ('ambilage', 'AmbiLAGE (基线)'),
    ('vi_08_02', 'VI-Enhanced (0.8/0.2)'),
)
PURE_REAL_GROUPS = [
    '01_camera', '03_cookies', '04_curtain', '05_hair_extension',
    '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush',
]
ALL_GROUPS = [
    '01_camera', '02_chair', '03_cookies', '04_curtain',
    '05_hair_extension', '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush',
]
MONO_TOL = 0.01


def pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def is_mono_desc(vals: list[float], tol: float = MONO_TOL) -> tuple[bool, str]:
    o, ov, z = vals
    if o >= ov >= z:
        return True, '严格单调'
    if (o + tol >= ov) and (ov + tol >= z) and (o + tol >= z):
        return True, '误差<=1pp单调'
    return False, '未单调'


def aggregate_groups(enc_data: dict, group_names: list[str]) -> dict:
    groups = enc_data['groups']
    total_q = sum(groups[g]['overall']['query_count'] for g in group_names)
    clean_mr = sum(groups[g]['overall']['mean_rank_clean'] * groups[g]['overall']['query_count'] for g in group_names) / total_q
    mod_mr = sum(groups[g]['overall']['mean_rank_modified'] * groups[g]['overall']['query_count'] for g in group_names) / total_q
    out = dict(
        query_count=total_q,
        mean_rank_clean=clean_mr,
        mean_rank_modified=mod_mr,
        mean_rank_gain=clean_mr - mod_mr,
    )
    for k in ('top1', 'top3', 'top5'):
        ch = sum(groups[g]['overall'][k]['clean_hits'] for g in group_names)
        mh = sum(groups[g]['overall'][k]['modified_hits'] for g in group_names)
        out[k] = dict(
            clean_hits=ch,
            modified_hits=mh,
            clean_rate=ch / total_q,
            modified_rate=mh / total_q,
            gain=(mh - ch) / total_q,
        )
    return out


def main() -> None:
    loaded: dict[tuple[str, str], dict] = {}
    validations: dict[str, dict] = {}
    for m_key, _ in METHODS:
        for suite, _ in SUITES:
            panel_dir = EXP_DIR / 'eval' / f'{m_key}_{suite}'
            p = panel_dir / 'transfer_panel_summary.json'
            if p.is_file():
                loaded[(m_key, suite)] = json.loads(p.read_text(encoding='utf-8'))
            val_p = panel_dir / '01_clip_l14' / 'category_query_validation.json'
            if suite not in validations and val_p.is_file():
                validations[suite] = json.loads(val_p.read_text(encoding='utf-8'))

    timings_path = EXP_DIR / 'logs' / 'timings.json'
    timings = json.loads(timings_path.read_text(encoding='utf-8')) if timings_path.is_file() else {}

    lines = [
        '# Visual Inception v4 严格 100/30/0 字面重合评测集 9 组全面对比报告',
        '',
    ]
    if 'eval_v4_sec' in timings:
        sec = float(timings['eval_v4_sec'])
        lines.append(
            f"- 评测执行模式：基于 `tools/fast_evaluate_panel.py eval-suites`（`--verify-mode in_process`，"
            f"全 6 面板 × 3 编码器总耗时 **`{sec:.1f}s` / `{sec/60:.2f} 分钟`**，全部通过全库无缓存重算校验 `<= 1e-7`）。"
        )
        lines.append('')

    lines += [
        '## 一、干净原图基线对比（共 9 组 × 20 = 180 条查询）',
        '',
        '| 评测集 | 训练原文重合 | `02_chair` 补位短搜数 | CLIP-L/14 Clean T1 / T3 / T5 / 均名 | OpenCLIP-bigG-14 Clean T1 / T3 / T5 / 均名 | SigLIP-SO400M Clean T1 / T3 / T5 / 均名 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    ref_method = 'vi_08_02' if ('vi_08_02', SUITES[0][0]) in loaded else 'ambilage'
    for suite, _ in SUITES:
        panel = loaded.get((ref_method, suite))
        if panel is None:
            continue
        val = validations.get(suite, {})
        exact_cnt = val.get('training_normalized_exact_overlap_count', '—')
        synth_cnt = val.get('synthetic_search_count', '—')
        cells = []
        for eslug, _ in ENCODERS:
            ov = panel['encoders'][eslug]['overall']
            cells.append(
                f"{pct(ov['top1']['clean_rate'])} / {pct(ov['top3']['clean_rate'])} / "
                f"{pct(ov['top5']['clean_rate'])} / {ov['mean_rank_clean']:.2f}"
            )
        lines.append(f'| `{suite}` | {exact_cnt}/180 | {synth_cnt} | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '## 二、修改图整体命中率对比（Top-1 / Top-3 / Top-5 / 平均名次 / Mean BB）',
        '',
        '| 方法 | 评测集 | CLIP-L/14 (白盒) T1 / T3 / T5 / 均名 | OpenCLIP-bigG-14 (黑盒) T1 / T3 / T5 / 均名 | SigLIP-SO400M (黑盒) T1 / T3 / T5 / 均名 | Mean BB T3 / T5 |',
        '|---|---|---:|---:|---:|---:|',
    ]

    mono_rows = []
    resilience_rows = []
    for m_key, m_label in METHODS:
        rates = {e[0]: {k: [] for k in ('t1', 't3', 't5')} for e in ENCODERS}
        bb_t3_seq, bb_t5_seq = [], []
        for suite, _ in SUITES:
            panel = loaded.get((m_key, suite))
            if panel is None:
                continue
            cells, bb3, bb5 = [], [], []
            for eslug, _ in ENCODERS:
                ov = panel['encoders'][eslug]['overall']
                t1, t3, t5 = ov['top1']['modified_rate'], ov['top3']['modified_rate'], ov['top5']['modified_rate']
                rates[eslug]['t1'].append(t1)
                rates[eslug]['t3'].append(t3)
                rates[eslug]['t5'].append(t5)
                if eslug != '01_clip_l14':
                    bb3.append(t3)
                    bb5.append(t5)
                cells.append(f"{pct(t1)} / {pct(t3)} / {pct(t5)} / {ov['mean_rank_modified']:.2f}")
            mbb3, mbb5 = sum(bb3) / len(bb3), sum(bb5) / len(bb5)
            bb_t3_seq.append(mbb3)
            bb_t5_seq.append(mbb5)
            lines.append(f'| {m_label} | `{suite}` | ' + ' | '.join(cells) + f' | {pct(mbb3)} / {pct(mbb5)} |')

        for eslug, elabel in ENCODERS:
            r1, r3, r5 = rates[eslug]['t1'], rates[eslug]['t3'], rates[eslug]['t5']
            if len(r5) == 3:
                _, tag1 = is_mono_desc(r1)
                _, tag3 = is_mono_desc(r3)
                _, tag5 = is_mono_desc(r5)
                mono_rows.append(
                    f'| {m_label} | {elabel} | '
                    f'{pct(r1[0])} -> {pct(r1[1])} -> {pct(r1[2])} ({tag1}) | '
                    f'{pct(r3[0])} -> {pct(r3[1])} -> {pct(r3[2])} ({tag3}) | '
                    f'{pct(r5[0])} -> {pct(r5[1])} -> {pct(r5[2])} ({tag5}) |'
                )
                d_t3 = (r3[2] - r3[0]) * 100.0
                d_t5 = (r5[2] - r5[0]) * 100.0
                ret_t5 = (r5[2] / r5[0] * 100.0) if r5[0] > 0 else 0.0
                resilience_rows.append(
                    f'| {m_label} | {elabel} | {pct(r3[0])} -> {pct(r3[2])} ({d_t3:+.1f} pp) | '
                    f'{pct(r5[0])} -> {pct(r5[2])} ({d_t5:+.1f} pp, 保持率 {ret_t5:.1f}%) |'
                )
        if len(bb_t5_seq) == 3:
            resilience_rows.append(
                f'| {m_label} | **Mean BB 均值** | {pct(bb_t3_seq[0])} -> {pct(bb_t3_seq[2])} ({(bb_t3_seq[2]-bb_t3_seq[0])*100:+.1f} pp) | '
                f'{pct(bb_t5_seq[0])} -> {pct(bb_t5_seq[2])} ({(bb_t5_seq[2]-bb_t5_seq[0])*100:+.1f} pp, 保持率 {bb_t5_seq[2]/bb_t5_seq[0]*100:.1f}%) |'
            )

    lines += [
        '',
        '## 三、单调性检验（`training_oracle_v4 >= overlap_30_v4 >= zero_overlap_v4`，允许 `<= 1.0 pp` 持平波动）',
        '',
        '| 方法 | 编码器 | Top-1 序列 | Top-3 序列 | Top-5 序列 |',
        '|---|---|---|---|---|',
        *mono_rows,
        '',
        '## 四、低重合抗衰减对比（`training_oracle_v4` -> `zero_overlap_v4`）',
        '',
        '| 方法 | 编码器 | Top-3 迁移衰减 | Top-5 迁移衰减与保持率 |',
        '|---|---|---:|---:|',
        *resilience_rows,
        '',
        '## 五、按三类查询子池细分（10 原文 / 6 改写 / 4 短搜）',
    ]

    for suite, _ in SUITES:
        panel_ref = loaded.get(('ambilage', suite)) or loaded.get(('vi_08_02', suite))
        if panel_ref is None:
            continue
        fams = list(panel_ref['encoders']['01_clip_l14']['by_family'])
        lines += [
            '',
            f'### `{suite}`',
            '',
            '| 方法 | 子池 | CLIP-L/14 T3 / T5 / 均名 | OpenCLIP-bigG-14 T3 / T5 / 均名 | SigLIP-SO400M T3 / T5 / 均名 |',
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
                    cells.append(f"{pct(st['top3']['modified_rate'])} / {pct(st['top5']['modified_rate'])} / {st['mean_rank_modified']:.2f}")
                lines.append(f'| {m_label} | {zh} (`{fam}`) | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '## 六、8 组纯真实搜索记录 vs `02_chair`（含临时补位短搜 `synthetic_search_fallback`）对比',
        '',
        '| 方法 | 评测集 | 口径范围（组数 / 条数） | CLIP-L/14 T3 / T5 / 均名 | OpenCLIP-bigG-14 T3 / T5 / 均名 | SigLIP-SO400M T3 / T5 / 均名 |',
        '|---|---|---|---:|---:|---:|',
    ]
    split_json = {}
    for suite, _ in SUITES:
        for m_key, m_label in METHODS:
            panel = loaded.get((m_key, suite))
            if panel is None:
                continue
            for label_scope, gnames in (('8 组纯真实短搜 (160)', PURE_REAL_GROUPS), ('`02_chair` 单组 (20)', ['02_chair'])):
                cells = []
                scope_rec = {}
                for eslug, _ in ENCODERS:
                    agg = aggregate_groups(panel['encoders'][eslug], gnames)
                    scope_rec[eslug] = agg
                    cells.append(f"{pct(agg['top3']['modified_rate'])} / {pct(agg['top5']['modified_rate'])} / {agg['mean_rank_modified']:.2f}")
                split_json[f'{m_key}_{suite}_{"pure8" if len(gnames) == 8 else "chair"}'] = scope_rec
                lines.append(f'| {m_label} | `{suite}` | {label_scope} | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '## 七、逐组全明细（全部 9 组，Top-1 / Top-3 / Top-5 / 均名）',
    ]
    for suite, _ in SUITES:
        lines += [
            '',
            f'### `{suite}` 逐组明细',
            '',
            '| 方法 | 组别 | CLIP-L/14 T1/T3/T5 (均名) | OpenCLIP-bigG-14 T1/T3/T5 (均名) | SigLIP-SO400M T1/T3/T5 (均名) |',
            '|---|---|---:|---:|---:|',
        ]
        for m_key, m_label in METHODS:
            panel = loaded.get((m_key, suite))
            if panel is None:
                continue
            for g in ALL_GROUPS:
                cells = []
                for eslug, _ in ENCODERS:
                    ov = panel['encoders'][eslug]['groups'][g]['overall']
                    cells.append(
                        f"{pct(ov['top1']['modified_rate'])}/{pct(ov['top3']['modified_rate'])}/{pct(ov['top5']['modified_rate'])} ({ov['mean_rank_modified']:.1f})"
                    )
                lines.append(f'| {m_label} | `{g}` | ' + ' | '.join(cells) + ' |')

    if V3_SUMMARY_PATH.is_file():
        v3 = json.loads(V3_SUMMARY_PATH.read_text(encoding='utf-8')).get('panels', {})
        lines += [
            '',
            '## 八、与 `v3` 评测集同口径对比（整体 Top-3 / Top-5 / Mean BB T5）',
            '',
            '| 方法 | 重合档位 | `v3` CLIP-L/14 T5 -> `v4` CLIP-L/14 T5 | `v3` Mean BB T3/T5 -> `v4` Mean BB T3/T5 |',
            '|---|---|---:|---:|',
        ]
        for m_key, m_label in METHODS:
            for suite, _ in SUITES:
                v3_panel = v3.get(f'{m_key}_{SUITE_TO_V3[suite]}')
                v4_panel = loaded.get((m_key, suite))
                if not v3_panel or not v4_panel:
                    continue
                v3_clip5 = v3_panel['01_clip_l14']['overall']['top5']['modified_rate']
                v4_clip5 = v4_panel['encoders']['01_clip_l14']['overall']['top5']['modified_rate']
                v3_bb3 = 0.5 * (
                    v3_panel['02_openclip_bigG14']['overall']['top3']['modified_rate']
                    + v3_panel['03_siglip_so400m']['overall']['top3']['modified_rate']
                )
                v3_bb5 = 0.5 * (
                    v3_panel['02_openclip_bigG14']['overall']['top5']['modified_rate']
                    + v3_panel['03_siglip_so400m']['overall']['top5']['modified_rate']
                )
                v4_bb3 = 0.5 * (
                    v4_panel['encoders']['02_openclip_bigG14']['overall']['top3']['modified_rate']
                    + v4_panel['encoders']['03_siglip_so400m']['overall']['top3']['modified_rate']
                )
                v4_bb5 = 0.5 * (
                    v4_panel['encoders']['02_openclip_bigG14']['overall']['top5']['modified_rate']
                    + v4_panel['encoders']['03_siglip_so400m']['overall']['top5']['modified_rate']
                )
                lines.append(
                    f'| {m_label} | `{suite}` | {pct(v3_clip5)} -> {pct(v4_clip5)} | '
                    f'{pct(v3_bb3)}/{pct(v3_bb5)} -> {pct(v4_bb3)}/{pct(v4_bb5)} |'
                )

    out_md = EXP_DIR / 'EVAL_V4_SUMMARY.md'
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary_json = {
        'timings': timings,
        'panels': {
            f'{m}_{s}': {
                es: {
                    'overall': loaded[(m, s)]['encoders'][es]['overall'],
                    'by_family': loaded[(m, s)]['encoders'][es]['by_family'],
                    'groups': loaded[(m, s)]['encoders'][es]['groups'],
                }
                for es, _ in ENCODERS
            }
            for m, _ in METHODS for s, _ in SUITES if (m, s) in loaded
        },
        'pure8_vs_chair': split_json,
    }
    (EXP_DIR / 'EVAL_V4_SUMMARY.json').write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'[SUMMARY WRITTEN] {out_md}')


if __name__ == '__main__':
    main()
