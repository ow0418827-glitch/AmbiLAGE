#!/usr/bin/env python3
"""汇总重建无偏 exact_overlap_v4 100/30/0 评测集结果（支持单方法先测与双方法增量追加）。"""
from __future__ import annotations

import json
from pathlib import Path

EXP_DIR = Path('/content/AmbiLAGE/output/exp_20261121_eval_v4_unbiased')
CALIBRATED_V4_SUMMARY_PATH = Path('/content/AmbiLAGE/output/exp_20261120_eval_v4/EVAL_V4_SUMMARY.json')
SUITES = (
    ('training_oracle_v4', '100% 重合 (`training_oracle_v4`)'),
    ('overlap_30_v4', '30% 重合 (`overlap_30_v4`)'),
    ('zero_overlap_v4', '0% 重合 (`zero_overlap_v4`)'),
)
ENCODERS = (
    ('01_clip_l14', 'CLIP-ViT-L/14 (白盒)'),
    ('02_openclip_bigG14', 'OpenCLIP-bigG-14 (黑盒)'),
    ('03_siglip_so400m', 'SigLIP-SO400M (黑盒)'),
)
METHODS = (
    ('vi_08_02', 'Visual Inception (0.8/0.2)'),
    ('ambilage', 'AmbiLAGE (jw=0.15, tw=1.50)'),
)
PURE_REAL_GROUPS = [
    '01_camera', '03_cookies', '04_curtain', '05_hair_extension',
    '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush',
]
ALL_GROUPS = [
    '01_camera', '02_chair', '03_cookies', '04_curtain',
    '05_hair_extension', '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush',
]
MONO_RATE_TOL = 0.01
MONO_RANK_TOL = 0.1


def pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def is_mono_desc(vals: list[float], tol: float = MONO_RATE_TOL) -> tuple[bool, str]:
    o, ov, z = vals
    if o >= ov >= z:
        return True, '严格单调递减'
    if (o + tol >= ov) and (ov + tol >= z) and (o + tol >= z):
        return True, '误差<=1pp单调'
    return False, '未单调'


def is_mono_rank_worse(vals: list[float], tol: float = MONO_RANK_TOL) -> tuple[bool, str]:
    """Check whether mean rank worsens (number increases: oracle <= overlap_30 <= zero_overlap)."""
    o, ov, z = vals
    if o <= ov <= z:
        return True, '严格单调变差(名次数值递增)'
    if (o - tol <= ov) and (ov - tol <= z) and (o - tol <= z):
        return True, '误差<=0.1单调变差'
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
    active_methods = []
    for m_key, m_label in METHODS:
        found_any = False
        for suite, _ in SUITES:
            panel_dir = EXP_DIR / 'eval' / f'{m_key}_{suite}'
            p = panel_dir / 'transfer_panel_summary.json'
            if p.is_file():
                loaded[(m_key, suite)] = json.loads(p.read_text(encoding='utf-8'))
                found_any = True
            val_p = panel_dir / '01_clip_l14' / 'category_query_validation.json'
            if suite not in validations and val_p.is_file():
                validations[suite] = json.loads(val_p.read_text(encoding='utf-8'))
        if found_any:
            active_methods.append((m_key, m_label))

    timings_path = EXP_DIR / 'logs' / 'timings.json'
    timings = json.loads(timings_path.read_text(encoding='utf-8')) if timings_path.is_file() else {}

    lines = [
        '# Visual Inception v4 重建无偏 100/30/0 字面重合评测集对比报告（commit fd16e7dd）',
        '',
    ]
    if timings:
        timing_parts = [f'`{k}={float(v):.1f}s ({float(v)/60:.2f} 分钟)`' for k, v in timings.items()]
        lines.append(
            f"- 评测执行模式：基于 `tools/fast_evaluate_panel.py eval-suites`（`--verify-mode in_process`，"
            f"已测方法：`{', '.join(m[0] for m in active_methods)}`，耗时：{'，'.join(timing_parts)}，全库无缓存复算校验 `<= 1e-7`）。"
        )
        lines.append('')

    lines += [
        '## 一、干净原图基线对比（共 9 组 × 20 = 180 条查询）',
        '',
        '| 评测集 | 训练原文重合 | `02_chair` 补位短搜数 | CLIP-L/14 Clean T1 / T3 / T5 / 均名 | OpenCLIP-bigG-14 Clean T1 / T3 / T5 / 均名 | SigLIP-SO400M Clean T1 / T3 / T5 / 均名 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    ref_method = active_methods[0][0] if active_methods else 'vi_08_02'
    clean_ranks: dict[str, list[float]] = {e[0]: [] for e in ENCODERS}
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
            clean_ranks[eslug].append(float(ov['mean_rank_clean']))
            cells.append(
                f"{pct(ov['top1']['clean_rate'])} / {pct(ov['top3']['clean_rate'])} / "
                f"{pct(ov['top5']['clean_rate'])} / {ov['mean_rank_clean']:.2f}"
            )
        lines.append(f'| `{suite}` | {exact_cnt}/180 | {synth_cnt} | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '## 二、修改图整体命中率与平均名次对比（Top-1 / Top-3 / Top-5 / 均名 / 名次提升 / Mean BB）',
        '',
        '| 方法 | 评测集 | CLIP-L/14 (白盒) T1 / T3 / T5 / 均名 (提升) | OpenCLIP-bigG-14 (黑盒) T1 / T3 / T5 / 均名 (提升) | SigLIP-SO400M (黑盒) T1 / T3 / T5 / 均名 (提升) | Mean BB T3 / T5 / 均名 |',
        '|---|---|---:|---:|---:|---:|',
    ]

    mono_rows = []
    resilience_rows = []
    for m_key, m_label in active_methods:
        rates = {e[0]: {k: [] for k in ('t1', 't3', 't5', 'mr')} for e in ENCODERS}
        bb_t3_seq, bb_t5_seq, bb_mr_seq = [], [], []
        for suite, _ in SUITES:
            panel = loaded.get((m_key, suite))
            if panel is None:
                continue
            cells, bb3, bb5, bbmr = [], [], [], []
            for eslug, _ in ENCODERS:
                ov = panel['encoders'][eslug]['overall']
                t1, t3, t5 = ov['top1']['modified_rate'], ov['top3']['modified_rate'], ov['top5']['modified_rate']
                mr = float(ov['mean_rank_modified'])
                mr_gain = float(ov.get('mean_rank_gain', float(ov['mean_rank_clean']) - mr))
                rates[eslug]['t1'].append(t1)
                rates[eslug]['t3'].append(t3)
                rates[eslug]['t5'].append(t5)
                rates[eslug]['mr'].append(mr)
                if eslug != '01_clip_l14':
                    bb3.append(t3)
                    bb5.append(t5)
                    bbmr.append(mr)
                cells.append(f"{pct(t1)} / {pct(t3)} / {pct(t5)} / {mr:.2f} ({mr_gain:+.2f})")
            mbb3, mbb5, mbbmr = sum(bb3) / len(bb3), sum(bb5) / len(bb5), sum(bbmr) / len(bbmr)
            bb_t3_seq.append(mbb3)
            bb_t5_seq.append(mbb5)
            bb_mr_seq.append(mbbmr)
            lines.append(f'| {m_label} | `{suite}` | ' + ' | '.join(cells) + f' | {pct(mbb3)} / {pct(mbb5)} / {mbbmr:.2f} |')

        for eslug, elabel in ENCODERS:
            r1, r3, r5, rmr = rates[eslug]['t1'], rates[eslug]['t3'], rates[eslug]['t5'], rates[eslug]['mr']
            if len(r5) == 3:
                _, tag_mr = is_mono_rank_worse(rmr)
                _, tag1 = is_mono_desc(r1)
                _, tag3 = is_mono_desc(r3)
                _, tag5 = is_mono_desc(r5)
                mono_rows.append(
                    f'| {m_label} | {elabel} | '
                    f'{rmr[0]:.2f} -> {rmr[1]:.2f} -> {rmr[2]:.2f} ({tag_mr}) | '
                    f'{pct(r1[0])} -> {pct(r1[1])} -> {pct(r1[2])} ({tag1}) | '
                    f'{pct(r3[0])} -> {pct(r3[1])} -> {pct(r3[2])} ({tag3}) | '
                    f'{pct(r5[0])} -> {pct(r5[1])} -> {pct(r5[2])} ({tag5}) |'
                )
                d_t3 = (r3[2] - r3[0]) * 100.0
                d_t5 = (r5[2] - r5[0]) * 100.0
                d_mr = rmr[2] - rmr[0]
                ret_t5 = (r5[2] / r5[0] * 100.0) if r5[0] > 0 else 0.0
                resilience_rows.append(
                    f'| {m_label} | {elabel} | {rmr[0]:.2f} -> {rmr[2]:.2f} ({d_mr:+.2f}) | '
                    f'{pct(r3[0])} -> {pct(r3[2])} ({d_t3:+.1f} pp) | '
                    f'{pct(r5[0])} -> {pct(r5[2])} ({d_t5:+.1f} pp, 保持率 {ret_t5:.1f}%) |'
                )
        if len(bb_t5_seq) == 3:
            _, tag_bbmr = is_mono_rank_worse(bb_mr_seq)
            _, tag_bb3 = is_mono_desc(bb_t3_seq)
            _, tag_bb5 = is_mono_desc(bb_t5_seq)
            mono_rows.append(
                f'| {m_label} | **Mean BB 均值** | '
                f'{bb_mr_seq[0]:.2f} -> {bb_mr_seq[1]:.2f} -> {bb_mr_seq[2]:.2f} ({tag_bbmr}) | '
                f'— | '
                f'{pct(bb_t3_seq[0])} -> {pct(bb_t3_seq[1])} -> {pct(bb_t3_seq[2])} ({tag_bb3}) | '
                f'{pct(bb_t5_seq[0])} -> {pct(bb_t5_seq[1])} -> {pct(bb_t5_seq[2])} ({tag_bb5}) |'
            )
            resilience_rows.append(
                f'| {m_label} | **Mean BB 均值** | {bb_mr_seq[0]:.2f} -> {bb_mr_seq[2]:.2f} ({bb_mr_seq[2]-bb_mr_seq[0]:+.2f}) | '
                f'{pct(bb_t3_seq[0])} -> {pct(bb_t3_seq[2])} ({(bb_t3_seq[2]-bb_t3_seq[0])*100:+.1f} pp) | '
                f'{pct(bb_t5_seq[0])} -> {pct(bb_t5_seq[2])} ({(bb_t5_seq[2]-bb_t5_seq[0])*100:+.1f} pp, 保持率 {bb_t5_seq[2]/bb_t5_seq[0]*100:.1f}%) |'
            )

    clean_mono_rows = []
    for eslug, elabel in ENCODERS:
        cmr = clean_ranks[eslug]
        if len(cmr) == 3:
            _, tag_cmr = is_mono_rank_worse(cmr)
            clean_mono_rows.append(
                f'| Clean 原图基线 | {elabel} | {cmr[0]:.2f} -> {cmr[1]:.2f} -> {cmr[2]:.2f} ({tag_cmr}) | — | — | — |'
            )

    lines += [
        '',
        '## 三、单调性检验（随着重复率从 100% -> 30% -> 0% 降低，平均名次数值递增变差且命中率递减）',
        '',
        '| 方法 | 编码器 | 平均名次序列 (`100% -> 30% -> 0%`) | Top-1 序列 | Top-3 序列 | Top-5 序列 |',
        '|---|---|---|---|---|---|',
        *clean_mono_rows,
        *mono_rows,
        '',
        '## 四、低重合抗衰减对比（`training_oracle_v4` -> `zero_overlap_v4`）',
        '',
        '| 方法 | 编码器 | 平均名次变化 | Top-3 迁移衰减 | Top-5 迁移衰减与保持率 |',
        '|---|---|---:|---:|---:|',
        *resilience_rows,
        '',
        '## 五、按三类查询子池细分（10 原文 / 6 改写 / 4 短搜）',
    ]

    for suite, _ in SUITES:
        panel_ref = loaded.get((ref_method, suite))
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
        for m_key, m_label in active_methods:
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
        for m_key, m_label in active_methods:
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
        for m_key, m_label in active_methods:
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

    if CALIBRATED_V4_SUMMARY_PATH.is_file():
        old_v4 = json.loads(CALIBRATED_V4_SUMMARY_PATH.read_text(encoding='utf-8')).get('panels', {})
        lines += [
            '',
            '## 八、与原单调校准版 `v4` (`exp_20261120_eval_v4`) 同口径对比（去除名次筛选前后差异）',
            '',
            '| 方法 | 重合档位 | CLIP-L/14 (校准->无偏) T3 / T5 / 均名 | OpenCLIP-bigG-14 (校准->无偏) T3 / T5 / 均名 | SigLIP-SO400M (校准->无偏) T3 / T5 / 均名 | Mean BB T3/T5 (校准->无偏) |',
            '|---|---|---|---|---|---|',
        ]
        for m_key, m_label in active_methods:
            for suite, _ in SUITES:
                old_panel = old_v4.get(f'{m_key}_{suite}')
                new_panel = loaded.get((m_key, suite))
                if not old_panel or not new_panel:
                    continue
                enc_cells = []
                for eslug, _ in ENCODERS:
                    oov = old_panel[eslug]['overall']
                    nov = new_panel['encoders'][eslug]['overall']
                    enc_cells.append(
                        f"{pct(oov['top3']['modified_rate'])}/{pct(oov['top5']['modified_rate'])} ({oov['mean_rank_modified']:.2f})"
                        f" -> {pct(nov['top3']['modified_rate'])}/{pct(nov['top5']['modified_rate'])} ({nov['mean_rank_modified']:.2f})"
                    )
                obb3 = 0.5 * (old_panel['02_openclip_bigG14']['overall']['top3']['modified_rate'] + old_panel['03_siglip_so400m']['overall']['top3']['modified_rate'])
                obb5 = 0.5 * (old_panel['02_openclip_bigG14']['overall']['top5']['modified_rate'] + old_panel['03_siglip_so400m']['overall']['top5']['modified_rate'])
                nbb3 = 0.5 * (new_panel['encoders']['02_openclip_bigG14']['overall']['top3']['modified_rate'] + new_panel['encoders']['03_siglip_so400m']['overall']['top3']['modified_rate'])
                nbb5 = 0.5 * (new_panel['encoders']['02_openclip_bigG14']['overall']['top5']['modified_rate'] + new_panel['encoders']['03_siglip_so400m']['overall']['top5']['modified_rate'])
                lines.append(
                    f'| {m_label} | `{suite}` | ' + ' | '.join(enc_cells) +
                    f' | {pct(obb3)}/{pct(obb5)} -> {pct(nbb3)}/{pct(nbb5)} |'
                )

    out_md = EXP_DIR / 'EVAL_V4_UNBIASED_SUMMARY.md'
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary_json = {
        'timings': timings,
        'active_methods': [m[0] for m in active_methods],
        'panels': {
            f'{m}_{s}': {
                es: {
                    'overall': loaded[(m, s)]['encoders'][es]['overall'],
                    'by_family': loaded[(m, s)]['encoders'][es]['by_family'],
                    'groups': loaded[(m, s)]['encoders'][es]['groups'],
                }
                for es, _ in ENCODERS
            }
            for m, _ in active_methods for s, _ in SUITES if (m, s) in loaded
        },
        'pure8_vs_chair': split_json,
    }
    (EXP_DIR / 'EVAL_V4_UNBIASED_SUMMARY.json').write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'[SUMMARY WRITTEN] {out_md}')


if __name__ == '__main__':
    main()
