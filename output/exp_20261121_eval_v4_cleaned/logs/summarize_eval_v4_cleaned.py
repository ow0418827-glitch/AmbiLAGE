#!/usr/bin/env python3
"""汇总清洗后无偏 exact_overlap_v4 100/30/0 评测集结果（含中位数、MRR、余弦增量及前后对比）。"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path('/content/AmbiLAGE')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.visual_inception_category_evaluation import _paired_summary

EXP_DIR = Path('/content/AmbiLAGE/output/exp_20261121_eval_v4_cleaned')
PREV_UNBIASED_PATH = Path('/content/AmbiLAGE/output/exp_20261121_eval_v4_unbiased/EVAL_V4_UNBIASED_SUMMARY.json')
QUERY_BASE = Path('/content/AmbiLAGE/resources/visual_inception_evaluation/exact_overlap_v4')
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
    ('vi_10_00', 'Visual Inception (1.0/0.0)'),
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
MONO_MRR_TOL = 0.005


def pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def is_mono_desc(vals: list[float], tol: float = MONO_RATE_TOL) -> tuple[bool, str]:
    o, ov, z = vals
    if o >= ov >= z:
        return True, '严格单调递减'
    if (o + tol >= ov) and (ov + tol >= z) and (o + tol >= z):
        return True, '误差容限内单调'
    return False, '未单调'


def is_mono_rank_worse(vals: list[float], tol: float = MONO_RANK_TOL) -> tuple[bool, str]:
    o, ov, z = vals
    if o <= ov <= z:
        return True, '严格单调变差'
    if (o - tol <= ov) and (ov - tol <= z) and (o - tol <= z):
        return True, '误差<=0.1单调'
    return False, '未单调'


def compute_rank_score_stats(panel_dir: Path, eslug: str) -> dict:
    clean_ranks, mod_ranks, clean_scores, mod_scores = [], [], [], []
    for g in ALL_GROUPS:
        ev_path = panel_dir / eslug / 'groups' / g / 'evaluation.json'
        if not ev_path.is_file():
            continue
        ev = json.loads(ev_path.read_text(encoding='utf-8'))
        cq = next(x for x in ev if x['bank'] == 'clean')['queries']
        mq = next(x for x in ev if x['bank'] == 'modified')['queries']
        for c, m in zip(cq, mq):
            clean_ranks.append(float(c['rank']))
            mod_ranks.append(float(m['rank']))
            clean_scores.append(float(c['score']))
            mod_scores.append(float(m['score']))
    clean_mrr = sum(1.0 / r for r in clean_ranks) / len(clean_ranks)
    mod_mrr = sum(1.0 / r for r in mod_ranks) / len(mod_ranks)
    return dict(
        clean_median_rank=float(statistics.median(clean_ranks)),
        mod_median_rank=float(statistics.median(mod_ranks)),
        clean_mrr=float(clean_mrr),
        mod_mrr=float(mod_mrr),
        mrr_gain=float(mod_mrr - clean_mrr),
        clean_mean_score=float(sum(clean_scores) / len(clean_scores)),
        mod_mean_score=float(sum(mod_scores) / len(mod_scores)),
        score_gain=float(sum(m - c for c, m in zip(clean_scores, mod_scores)) / len(clean_scores)),
    )


def fmt_opt_pct(x: float | None, *, signed: bool = False) -> str:
    if x is None:
        return '—'
    return f'{x * 100:+.1f}%' if signed else f'{x * 100:.1f}%'


def compute_headroom_and_conditional_stats(panel_dir: Path, eslug: str, panel_enc_data: dict) -> dict:
    clean_ranks: dict[tuple[str, str], int] = {}
    mod_ranks: dict[tuple[str, str], int] = {}
    per_group_headroom: dict[str, dict] = {}
    miss_c5, miss_m5, hit_c5, hit_m5 = [], [], [], []

    for g in ALL_GROUPS:
        ev_path = panel_dir / eslug / 'groups' / g / 'evaluation.json'
        ev = json.loads(ev_path.read_text(encoding='utf-8'))
        cq = next(x for x in ev if x['bank'] == 'clean')['queries']
        mq = next(x for x in ev if x['bank'] == 'modified')['queries']
        g_clean, g_mod = {}, {}
        for c, m in zip(cq, mq):
            tid = c['task_id']
            cr, mr = int(c['rank']), int(m['rank'])
            clean_ranks[(g, tid)] = cr
            mod_ranks[(g, tid)] = mr
            g_clean[tid] = cr
            g_mod[tid] = mr
            if cr > 5:
                miss_c5.append(cr)
                miss_m5.append(mr)
            else:
                hit_c5.append(cr)
                hit_m5.append(mr)
        per_group_headroom[g] = _paired_summary(g_clean, g_mod)

    paired = _paired_summary(clean_ranks, mod_ranks)
    ov = panel_enc_data['overall']
    assert abs(paired['mean_rank_clean'] - ov['mean_rank_clean']) < 1e-12
    assert abs(paired['mean_rank_modified'] - ov['mean_rank_modified']) < 1e-12
    for k in ('top1', 'top3', 'top5'):
        for fld in ('clean_hits', 'modified_hits', 'new_hits', 'lost_hits'):
            assert paired[k][fld] == ov[k][fld], (eslug, k, fld, paired[k][fld], ov[k][fld])

    lost_breakdown = {}
    for k in ('top1', 'top3', 'top5'):
        lost_breakdown[k] = [
            dict(
                group=g,
                clean_hits=per_group_headroom[g][k]['clean_hits'],
                clean_miss=per_group_headroom[g][k]['clean_miss_count'],
                new_hits=per_group_headroom[g][k]['new_hits'],
                lost_hits=per_group_headroom[g][k]['lost_hits'],
            )
            for g in ALL_GROUPS
            if per_group_headroom[g][k]['lost_hits'] > 0
        ]

    cond_t5 = dict(
        miss_count=len(miss_c5),
        miss_mean_clean=(sum(miss_c5) / len(miss_c5)) if miss_c5 else None,
        miss_mean_modified=(sum(miss_m5) / len(miss_m5)) if miss_m5 else None,
        hit_count=len(hit_c5),
        hit_mean_clean=(sum(hit_c5) / len(hit_c5)) if hit_c5 else None,
        hit_mean_modified=(sum(hit_m5) / len(hit_m5)) if hit_m5 else None,
    )
    return dict(
        overall=paired,
        per_group=per_group_headroom,
        lost_hits_breakdown=lost_breakdown,
        conditional_rank_t5=cond_t5,
    )


def compute_suite_length_stats(suite: str) -> dict:
    by_fam: dict[str, list[int]] = {}
    for g in ALL_GROUPS:
        qpath = QUERY_BASE / suite / g / 'queries.json'
        if not qpath.is_file():
            continue
        doc = json.loads(qpath.read_text(encoding='utf-8'))
        for rec in doc['query_records']:
            by_fam.setdefault(rec['query_family'], []).append(len(rec['instruction']))
    return {
        fam: dict(
            count=len(lens),
            mean_chars=sum(lens) / len(lens),
            median_chars=float(statistics.median(lens)),
            min_chars=min(lens),
            max_chars=max(lens),
        )
        for fam, lens in by_fam.items()
    }


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
    extra_stats: dict[tuple[str, str, str], dict] = {}
    headroom_stats: dict[tuple[str, str, str], dict] = {}
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
                for eslug, _ in ENCODERS:
                    extra_stats[(m_key, suite, eslug)] = compute_rank_score_stats(panel_dir, eslug)
                    headroom_stats[(m_key, suite, eslug)] = compute_headroom_and_conditional_stats(
                        panel_dir, eslug, loaded[(m_key, suite)]['encoders'][eslug]
                    )
            val_p = panel_dir / '01_clip_l14' / 'category_query_validation.json'
            if suite not in validations and val_p.is_file():
                validations[suite] = json.loads(val_p.read_text(encoding='utf-8'))
        if found_any:
            active_methods.append((m_key, m_label))

    timings_path = EXP_DIR / 'logs' / 'timings.json'
    timings = json.loads(timings_path.read_text(encoding='utf-8')) if timings_path.is_file() else {}
    length_stats = {suite: compute_suite_length_stats(suite) for suite, _ in SUITES}

    lines = [
        '# Visual Inception v4 清洗对齐后 100/30/0 无偏评测集实验报告',
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
        '## 零、三档查询文本长度审计（验证改写长度对齐）',
        '',
        '| 评测集 | 商品原文 (`category_description`) 均长/中位 | 改写查询 (`model_paraphrase`) 均长/中位 | 短搜索 (`heldout_search_query`) 均长/中位 |',
        '|---|---:|---:|---:|',
    ]
    for suite, _ in SUITES:
        ls = length_stats[suite]
        lines.append(
            f"| `{suite}` | {ls['category_description']['mean_chars']:.1f} / {ls['category_description']['median_chars']:.1f} "
            f"| {ls['model_paraphrase']['mean_chars']:.1f} / {ls['model_paraphrase']['median_chars']:.1f} "
            f"| {ls['heldout_search_query']['mean_chars']:.1f} / {ls['heldout_search_query']['median_chars']:.1f} |"
        )

    lines += [
        '',
        '## 一、干净原图基线对比（共 9 组 × 20 = 180 条查询）',
        '',
        '| 评测集 | 训练原文重合 | `02_chair` 补位短搜 | CLIP-L/14 Clean T1/T3/T5 / 均名 (中位, MRR) | OpenCLIP-bigG-14 Clean T1/T3/T5 / 均名 (中位, MRR) | SigLIP-SO400M Clean T1/T3/T5 / 均名 (中位, MRR) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    ref_method = active_methods[0][0] if active_methods else 'vi_08_02'
    clean_ranks: dict[str, list[float]] = {e[0]: [] for e in ENCODERS}
    clean_meds: dict[str, list[float]] = {e[0]: [] for e in ENCODERS}
    clean_mrrs: dict[str, list[float]] = {e[0]: [] for e in ENCODERS}
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
            ex = extra_stats[(ref_method, suite, eslug)]
            clean_ranks[eslug].append(float(ov['mean_rank_clean']))
            clean_meds[eslug].append(ex['clean_median_rank'])
            clean_mrrs[eslug].append(ex['clean_mrr'])
            cells.append(
                f"{pct(ov['top1']['clean_rate'])}/{pct(ov['top3']['clean_rate'])}/{pct(ov['top5']['clean_rate'])} / "
                f"{ov['mean_rank_clean']:.2f} ({ex['clean_median_rank']:.1f}, {ex['clean_mrr']:.3f})"
            )
        lines.append(f'| `{suite}` | {exact_cnt}/180 | {synth_cnt} | ' + ' | '.join(cells) + ' |')

    lines += [
        '',
        '## 二、修改图整体命中率、名次与相似度增量对比',
        '',
        '| 方法 | 评测集 | CLIP-L/14 (白盒) T1/T3/T5 / 均名(中位, MRR, ΔCos) | OpenCLIP-bigG-14 (黑盒) T1/T3/T5 / 均名(中位, MRR, ΔCos) | SigLIP-SO400M (黑盒) T1/T3/T5 / 均名(中位, MRR, ΔCos) | Mean BB T3 / T5 / 均名 |',
        '|---|---|---:|---:|---:|---:|',
    ]

    mono_rows = []
    resilience_rows = []
    for m_key, m_label in active_methods:
        rates = {e[0]: {k: [] for k in ('t1', 't3', 't5', 'mr', 'med', 'mrr')} for e in ENCODERS}
        bb_t3_seq, bb_t5_seq, bb_mr_seq = [], [], []
        for suite, _ in SUITES:
            panel = loaded.get((m_key, suite))
            if panel is None:
                continue
            cells, bb3, bb5, bbmr = [], [], [], []
            for eslug, _ in ENCODERS:
                ov = panel['encoders'][eslug]['overall']
                ex = extra_stats[(m_key, suite, eslug)]
                t1, t3, t5 = ov['top1']['modified_rate'], ov['top3']['modified_rate'], ov['top5']['modified_rate']
                mr = float(ov['mean_rank_modified'])
                rates[eslug]['t1'].append(t1)
                rates[eslug]['t3'].append(t3)
                rates[eslug]['t5'].append(t5)
                rates[eslug]['mr'].append(mr)
                rates[eslug]['med'].append(ex['mod_median_rank'])
                rates[eslug]['mrr'].append(ex['mod_mrr'])
                if eslug != '01_clip_l14':
                    bb3.append(t3)
                    bb5.append(t5)
                    bbmr.append(mr)
                cells.append(
                    f"{pct(t1)}/{pct(t3)}/{pct(t5)} / {mr:.2f} "
                    f"({ex['mod_median_rank']:.1f}, {ex['mod_mrr']:.3f}, {ex['score_gain']:+.4f})"
                )
            mbb3, mbb5, mbbmr = sum(bb3) / len(bb3), sum(bb5) / len(bb5), sum(bbmr) / len(bbmr)
            bb_t3_seq.append(mbb3)
            bb_t5_seq.append(mbb5)
            bb_mr_seq.append(mbbmr)
            lines.append(f'| {m_label} | `{suite}` | ' + ' | '.join(cells) + f' | {pct(mbb3)} / {pct(mbb5)} / {mbbmr:.2f} |')

        for eslug, elabel in ENCODERS:
            r1, r3, r5 = rates[eslug]['t1'], rates[eslug]['t3'], rates[eslug]['t5']
            rmr, rmed, rmrr = rates[eslug]['mr'], rates[eslug]['med'], rates[eslug]['mrr']
            if len(r5) == 3:
                _, tag_mr = is_mono_rank_worse(rmr)
                _, tag_med = is_mono_rank_worse(rmed)
                _, tag_mrr = is_mono_desc(rmrr, tol=MONO_MRR_TOL)
                _, tag1 = is_mono_desc(r1)
                _, tag3 = is_mono_desc(r3)
                _, tag5 = is_mono_desc(r5)
                mono_rows.append(
                    f'| {m_label} | {elabel} | '
                    f'{rmr[0]:.2f}->{rmr[1]:.2f}->{rmr[2]:.2f} ({tag_mr}) | '
                    f'{rmed[0]:.1f}->{rmed[1]:.1f}->{rmed[2]:.1f} ({tag_med}) | '
                    f'{rmrr[0]:.3f}->{rmrr[1]:.3f}->{rmrr[2]:.3f} ({tag_mrr}) | '
                    f'{pct(r1[0])}->{pct(r1[1])}->{pct(r1[2])} ({tag1}) | '
                    f'{pct(r3[0])}->{pct(r3[1])}->{pct(r3[2])} ({tag3}) | '
                    f'{pct(r5[0])}->{pct(r5[1])}->{pct(r5[2])} ({tag5}) |'
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
                f'{bb_mr_seq[0]:.2f}->{bb_mr_seq[1]:.2f}->{bb_mr_seq[2]:.2f} ({tag_bbmr}) | '
                f'— | — | — | '
                f'{pct(bb_t3_seq[0])}->{pct(bb_t3_seq[1])}->{pct(bb_t3_seq[2])} ({tag_bb3}) | '
                f'{pct(bb_t5_seq[0])}->{pct(bb_t5_seq[1])}->{pct(bb_t5_seq[2])} ({tag_bb5}) |'
            )
            resilience_rows.append(
                f'| {m_label} | **Mean BB 均值** | {bb_mr_seq[0]:.2f} -> {bb_mr_seq[2]:.2f} ({bb_mr_seq[2]-bb_mr_seq[0]:+.2f}) | '
                f'{pct(bb_t3_seq[0])} -> {pct(bb_t3_seq[2])} ({(bb_t3_seq[2]-bb_t3_seq[0])*100:+.1f} pp) | '
                f'{pct(bb_t5_seq[0])} -> {pct(bb_t5_seq[2])} ({(bb_t5_seq[2]-bb_t5_seq[0])*100:+.1f} pp, 保持率 {bb_t5_seq[2]/bb_t5_seq[0]*100:.1f}%) |'
            )

    clean_mono_rows = []
    for eslug, elabel in ENCODERS:
        cmr, cmed, cmrr = clean_ranks[eslug], clean_meds[eslug], clean_mrrs[eslug]
        if len(cmr) == 3:
            _, tag_cmr = is_mono_rank_worse(cmr)
            _, tag_cmed = is_mono_rank_worse(cmed)
            _, tag_cmrr = is_mono_desc(cmrr, tol=MONO_MRR_TOL)
            clean_mono_rows.append(
                f'| Clean 原图基线 | {elabel} | {cmr[0]:.2f}->{cmr[1]:.2f}->{cmr[2]:.2f} ({tag_cmr}) | '
                f'{cmed[0]:.1f}->{cmed[1]:.1f}->{cmed[2]:.1f} ({tag_cmed}) | '
                f'{cmrr[0]:.3f}->{cmrr[1]:.3f}->{cmrr[2]:.3f} ({tag_cmrr}) | — | — | — |'
            )

    lines += [
        '',
        '## 三、单调性检验（100% -> 30% -> 0%：均名/中位数变大、MRR 与命中率递减）',
        '',
        '| 方法 | 编码器 | 平均名次 | 中位名次 | MRR | Top-1 | Top-3 | Top-5 |',
        '|---|---|---|---|---|---|---|---|',
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
        '## 六、8 组纯真实搜索记录 vs `02_chair` 对比',
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

    if PREV_UNBIASED_PATH.is_file():
        prev_panels = json.loads(PREV_UNBIASED_PATH.read_text(encoding='utf-8')).get('panels', {})
        lines += [
            '',
            '## 八、与修复前无偏版 (`exp_20261121_eval_v4_unbiased`) 前后直接对比',
            '',
            '| 方法 | 重合档位 | 指标对比 | CLIP-L/14 (前->后) | OpenCLIP-bigG-14 (前->后) | SigLIP-SO400M (前->后) | Mean BB (前->后) |',
            '|---|---|---|---|---|---|---|',
        ]
        for m_key, m_label in active_methods:
            for suite, _ in SUITES:
                op = prev_panels.get(f'{m_key}_{suite}')
                np_cur = loaded.get((m_key, suite))
                if not op or not np_cur:
                    continue
                clean_cells, mod_cells = [], []
                for eslug, _ in ENCODERS:
                    oov = op[eslug]['overall']
                    nov = np_cur['encoders'][eslug]['overall']
                    clean_cells.append(
                        f"{pct(oov['top3']['clean_rate'])}/{pct(oov['top5']['clean_rate'])} ({oov['mean_rank_clean']:.2f})"
                        f" -> {pct(nov['top3']['clean_rate'])}/{pct(nov['top5']['clean_rate'])} ({nov['mean_rank_clean']:.2f})"
                    )
                    mod_cells.append(
                        f"{pct(oov['top3']['modified_rate'])}/{pct(oov['top5']['modified_rate'])} ({oov['mean_rank_modified']:.2f})"
                        f" -> {pct(nov['top3']['modified_rate'])}/{pct(nov['top5']['modified_rate'])} ({nov['mean_rank_modified']:.2f})"
                    )
                obb_mr = 0.5 * (op['02_openclip_bigG14']['overall']['mean_rank_modified'] + op['03_siglip_so400m']['overall']['mean_rank_modified'])
                nbb_mr = 0.5 * (np_cur['encoders']['02_openclip_bigG14']['overall']['mean_rank_modified'] + np_cur['encoders']['03_siglip_so400m']['overall']['mean_rank_modified'])
                obb_t5 = 0.5 * (op['02_openclip_bigG14']['overall']['top5']['modified_rate'] + op['03_siglip_so400m']['overall']['top5']['modified_rate'])
                nbb_t5 = 0.5 * (np_cur['encoders']['02_openclip_bigG14']['overall']['top5']['modified_rate'] + np_cur['encoders']['03_siglip_so400m']['overall']['top5']['modified_rate'])
                obb_c_mr = 0.5 * (op['02_openclip_bigG14']['overall']['mean_rank_clean'] + op['03_siglip_so400m']['overall']['mean_rank_clean'])
                nbb_c_mr = 0.5 * (np_cur['encoders']['02_openclip_bigG14']['overall']['mean_rank_clean'] + np_cur['encoders']['03_siglip_so400m']['overall']['mean_rank_clean'])
                lines.append(f'| {m_label} | `{suite}` | Clean T3/T5(均名) | ' + ' | '.join(clean_cells) + f' | 均名 {obb_c_mr:.2f} -> {nbb_c_mr:.2f} |')
                lines.append(f'| {m_label} | `{suite}` | Modified T3/T5(均名) | ' + ' | '.join(mod_cells) + f' | T5 {pct(obb_t5)}->{pct(nbb_t5)} / 均名 {obb_mr:.2f}->{nbb_mr:.2f} |')

    lines += [
        '',
        '## 九、扣除原图已命中后的净剩余空间、未命中转化率与原命中保持率（新默认补充口径）',
        '',
        '- **统计口径说明**：保留全量绝对 Top-K（`modified_hits / 180`）与净增量（`gain = (modified_hits - clean_hits) / 180`）作为主报告口径；本节补充统计：',
        '  - `未命中转化率 (ConvRate)`：`new_hits / (180 - clean_hits)`（仅考察原图未命中的剩余样本中有多少被推入 Top-K，**无法惩罚倒退**）；',
        '  - `原命中保持率 (Retain)`：`(clean_hits - lost_hits) / clean_hits`（直接暴露多少原本排在 Top-K 内的查询在修改后跌出 Top-K）；',
        '  - `净剩余空间捕获率 (NetCap)`：`(modified_hits - clean_hits) / (180 - clean_hits) = (new_hits - lost_hits) / (180 - clean_hits)`（剔除原图已命中分母，同时完整扣减 `lost_hits` 倒退惩罚）。',
        '- 全量汇总仅按全部 180 条查询微平均（不做逐组宏平均，因为 `02_chair` 与 `03_cookies` 在 Top-5 下 `clean_miss_count` 仅 0–2 条）。',
        '',
        '### Mean Black-Box (OpenCLIP-bigG-14 与 SigLIP-SO400M 黑盒均值)',
        '',
        '| Top-K | 方法 | 评测集 | 原命中 / 总查询 | 绝对命中率 | 全量净增量 | 原未命中转化 Conv | 原命中保持 Retain | 净剩余空间捕获 NetCap | 黑盒累计倒退 (`lost/clean`) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for k in ('top1', 'top3', 'top5'):
        k_label = k.upper().replace('TOP', 'Top-')
        for m_key, m_label in active_methods:
            for suite, _ in SUITES:
                if (m_key, suite, '02_openclip_bigG14') not in headroom_stats:
                    continue
                bg = headroom_stats[(m_key, suite, '02_openclip_bigG14')]['overall'][k]
                sg = headroom_stats[(m_key, suite, '03_siglip_so400m')]['overall'][k]
                cl_rate = 0.5 * (bg['clean_rate'] + sg['clean_rate'])
                abs_rate = 0.5 * (bg['modified_rate'] + sg['modified_rate'])
                net_gain = 0.5 * (bg['gain'] + sg['gain'])
                conv = 0.5 * (bg['clean_miss_conversion_rate'] + sg['clean_miss_conversion_rate'])
                ret = 0.5 * (bg['clean_hit_retention_rate'] + sg['clean_hit_retention_rate'])
                netcap = 0.5 * (bg['net_headroom_capture'] + sg['net_headroom_capture'])
                lost_sum = bg['lost_hits'] + sg['lost_hits']
                clean_sum = bg['clean_hits'] + sg['clean_hits']
                lines.append(
                    f"| {k_label} | {m_label} | `{suite}` | {clean_sum}/360 | {abs_rate * 100:.1f}% | "
                    f"{net_gain * 100:+.2f} pp | {conv * 100:.2f}% | {ret * 100:.2f}% | "
                    f"**{netcap * 100:+.2f}%** | {lost_sum}/{clean_sum} |"
                )

    for eslug, elabel in ENCODERS:
        lines += [
            '',
            f'### {elabel} 扣除原图已命中补充分表',
            '',
            '| Top-K | 方法 | 评测集 | 原命中 -> 修改命中 (+新/-退) | 绝对命中率 | 原未命中分母 | Conv | Retain | NetCap |',
            '|---|---|---|---:|---:|---:|---:|---:|---:|',
        ]
        for k in ('top1', 'top3', 'top5'):
            k_label = k.upper().replace('TOP', 'Top-')
            for m_key, m_label in active_methods:
                for suite, _ in SUITES:
                    if (m_key, suite, eslug) not in headroom_stats:
                        continue
                    t = headroom_stats[(m_key, suite, eslug)]['overall'][k]
                    lines.append(
                        f"| {k_label} | {m_label} | `{suite}` | "
                        f"{t['clean_hits']} -> {t['modified_hits']} (+{t['new_hits']}/-{t['lost_hits']}) | "
                        f"{t['modified_rate'] * 100:.1f}% | {t['clean_miss_count']}/180 | "
                        f"{fmt_opt_pct(t['clean_miss_conversion_rate'])} | "
                        f"{fmt_opt_pct(t['clean_hit_retention_rate'])} | "
                        f"{fmt_opt_pct(t['net_headroom_capture'], signed=True)} |"
                    )

    lines += [
        '',
        '### `OpenCLIP-bigG-14` Top-3 / Top-5 逐组倒退明细 (`lost_hits > 0`)',
        '',
        '| 评测集 | 方法 | Top-K | 总倒退 (`lost/clean`) | 发生倒退的分组明细（原命中 / 倒退 / 新增） |',
        '|---|---|---|---:|---|',
    ]
    for suite, _ in SUITES:
        for m_key, m_label in active_methods:
            hd = headroom_stats.get((m_key, suite, '02_openclip_bigG14'))
            if not hd:
                continue
            for k in ('top3', 'top5'):
                ov_t = hd['overall'][k]
                bk = hd['lost_hits_breakdown'][k]
                detail = '；'.join(
                    f"`{r['group']}` ({r['clean_hits']}/20中跌出{r['lost_hits']}条，另新进{r['new_hits']}条)"
                    for r in bk
                ) if bk else '无倒退'
                lines.append(
                    f"| `{suite}` | {m_label} | {k.upper().replace('TOP', 'Top-')} | "
                    f"{ov_t['lost_hits']}/{ov_t['clean_hits']} | {detail} |"
                )

    lines += [
        '',
        '### 条件平均名次拆分（原图 Top-5 未命中 `clean_rank > 5` vs 原图已在 Top-5 内 `clean_rank <= 5`）',
        '',
        '| 编码器 | 方法 | `training_oracle_v4` (未命中均名 / 已命中均名) | `overlap_30_v4` (未命中均名 / 已命中均名) | `zero_overlap_v4` (未命中均名 / 已命中均名) |',
        '|---|---|---|---|---|',
    ]
    for eslug, elabel in ENCODERS:
        for m_key, m_label in active_methods:
            c_cells = []
            for suite, _ in SUITES:
                cr = headroom_stats[(m_key, suite, eslug)]['conditional_rank_t5']
                c_cells.append(
                    f"[{cr['miss_count']}] {cr['miss_mean_clean']:.1f}->{cr['miss_mean_modified']:.1f} / "
                    f"[{cr['hit_count']}] {cr['hit_mean_clean']:.2f}->{cr['hit_mean_modified']:.2f}"
                )
            lines.append(f'| {elabel} | {m_label} | ' + ' | '.join(c_cells) + ' |')

    out_md = EXP_DIR / 'EVAL_V4_CLEANED_SUMMARY.md'
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary_json = {
        'timings': timings,
        'active_methods': [m[0] for m in active_methods],
        'length_stats': length_stats,
        'extra_stats': {f'{m}_{s}_{e}': v for (m, s, e), v in extra_stats.items()},
        'headroom_metrics': {
            f'{m}_{s}_{e}': {
                'top1': v['overall']['top1'],
                'top3': v['overall']['top3'],
                'top5': v['overall']['top5'],
                'conditional_rank_t5': v['conditional_rank_t5'],
                'lost_hits_breakdown': v['lost_hits_breakdown'],
            }
            for (m, s, e), v in headroom_stats.items()
        },
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
    (EXP_DIR / 'EVAL_V4_CLEANED_SUMMARY.json').write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'[SUMMARY WRITTEN] {out_md}')


if __name__ == '__main__':
    main()
