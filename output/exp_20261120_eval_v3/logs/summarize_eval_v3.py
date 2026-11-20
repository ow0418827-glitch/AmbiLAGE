#!/usr/bin/env python3
"""汇总 Stage 2 v3 九组严格嵌套评测结果：基线校准、单调性校验、低重合抗衰减对比与子家族/显隐拆解。"""
from __future__ import annotations

import json
from pathlib import Path

EXP_DIR = Path('/content/AmbiLAGE/output/exp_20261120_eval_v3')
QUERY_DIRS = {
    'training_oracle_v3': Path('/content/AmbiLAGE/resources/visual_inception_evaluation/training_queries_oracle_v3'),
    'overlap_30_v3': Path('/content/AmbiLAGE/resources/visual_inception_evaluation/category_query_overlap_30_v3'),
    'zero_overlap_v3': Path('/content/AmbiLAGE/resources/visual_inception_evaluation/category_queries_v3'),
}
SUITES = ('training_oracle_v3', 'overlap_30_v3', 'zero_overlap_v3')
ENCODERS = (
    ('01_clip_l14', 'CLIP-ViT-L/14 (白盒)'),
    ('02_openclip_bigG14', 'OpenCLIP-bigG-14 (黑盒)'),
    ('03_siglip_so400m', 'SigLIP-SO400M (黑盒)'),
)
METHODS = (
    ('ambilage', 'AmbiLAGE (基线)'),
    ('vi_08_02', 'VI-Enhanced (0.8/0.2)'),
)
MONO_TOL = 0.01  # allow <= 1.0 pp statistical fluctuation per spec


def pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def is_mono_desc(vals: list[float], tol: float = MONO_TOL) -> tuple[bool, str]:
    o, ov, z = vals
    strict = o >= ov >= z
    within_tol = (o + tol >= ov) and (ov + tol >= z) and (o + tol >= z)
    if strict:
        return True, '严格单调'
    if within_tol:
        return True, '误差<=1pp单调'
    return False, '未单调'


def eval_explicit_implicit(m_key: str, suite: str, eslug: str) -> dict[str, dict]:
    ev_root = EXP_DIR / 'eval' / f'{m_key}_{suite}' / eslug / 'groups'
    q_root = QUERY_DIRS[suite]
    buckets = {
        'explicit': {'c1': 0, 'c3': 0, 'c5': 0, 'm1': 0, 'm3': 0, 'm5': 0, 'cr': 0, 'mr': 0, 'n': 0},
        'implicit': {'c1': 0, 'c3': 0, 'c5': 0, 'm1': 0, 'm3': 0, 'm5': 0, 'cr': 0, 'mr': 0, 'n': 0},
    }
    for gd in sorted(q_root.glob('0*')):
        g = gd.name
        recs = json.loads((gd / 'queries.json').read_text(encoding='utf-8'))['query_records']
        rows = json.loads((ev_root / g / 'evaluation.json').read_text(encoding='utf-8'))
        by_bank = {r['bank']: r['queries'] for r in rows}
        for rec, cq, mq in zip(recs, by_bank['clean'], by_bank['modified']):
            b = buckets['implicit' if rec.get('implicit_anchor_free') else 'explicit']
            cr, mr = cq['rank'], mq['rank']
            b['n'] += 1
            b['cr'] += cr
            b['mr'] += mr
            if cr == 1: b['c1'] += 1
            if cr <= 3: b['c3'] += 1
            if cr <= 5: b['c5'] += 1
            if mr == 1: b['m1'] += 1
            if mr <= 3: b['m3'] += 1
            if mr <= 5: b['m5'] += 1
    out = {}
    for k, b in buckets.items():
        if b['n'] == 0:
            continue
        n = b['n']
        out[k] = dict(
            n=n,
            clean_t1=b['c1'] / n, clean_t3=b['c3'] / n, clean_t5=b['c5'] / n, clean_mr=b['cr'] / n,
            mod_t1=b['m1'] / n, mod_t3=b['m3'] / n, mod_t5=b['m5'] / n, mod_mr=b['mr'] / n,
        )
    return out


def main() -> None:
    loaded: dict[tuple[str, str], dict] = {}
    for m_key, _ in METHODS:
        for suite in SUITES:
            p = EXP_DIR / 'eval' / f'{m_key}_{suite}' / 'transfer_panel_summary.json'
            if p.is_file():
                loaded[(m_key, suite)] = json.loads(p.read_text(encoding='utf-8'))

    lines = [
        '# Visual Inception v3 严格嵌套评测集 9 组全面对比报告',
        '',
        '## 一、干净原图基线对比（Gate 3 验证，共 9 组 × 20 = 180 条查询）',
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
        for suite in SUITES:
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
        '## 三、单调性检验（`training_oracle_v3 >= overlap_30_v3 >= zero_overlap_v3`，允许 `<= 1.0 pp` 持平波动）',
        '',
        '| 方法 | 编码器 | Top-1 序列 | Top-3 序列 | Top-5 序列 |',
        '|---|---|---|---|---|',
        *mono_rows,
        '',
        '## 四、低重合抗衰减对比（`oracle_v3` -> `zero_overlap_v3`）',
        '',
        '| 方法 | 编码器 | Top-3 迁移衰减 | Top-5 迁移衰减与保持率 |',
        '|---|---|---:|---:|',
        *resilience_rows,
        '',
        '## 五、同构三子池（`10 描述/摘要` / `6 改写` / `4 短搜词`）Top-3 / Top-5 / 均名细分',
    ]

    for suite in SUITES:
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
        '## 六、显式查询 vs 无核心锚词隐晦查询拆解（`overlap_30_v3` 与 `zero_overlap_v3`）',
        '',
        '| 方法 | 评测集 | 子集（条数） | CLIP-L/14 Clean->Mod T5 (Mod T3) | OpenCLIP-bigG-14 Clean->Mod T5 (Mod T3) | SigLIP-SO400M Clean->Mod T5 (Mod T3) |',
        '|---|---|---|---:|---:|---:|',
    ]
    exp_imp_json = {}
    for suite in ('overlap_30_v3', 'zero_overlap_v3'):
        for m_key, m_label in METHODS:
            split_by_enc = {es: eval_explicit_implicit(m_key, suite, es) for es, _ in ENCODERS}
            exp_imp_json[f'{m_key}_{suite}'] = split_by_enc
            for bucket, blabel in (('explicit', '显式含品类锚词'), ('implicit', '隐晦无品类锚词')):
                if bucket not in split_by_enc['01_clip_l14']:
                    continue
                n_b = split_by_enc['01_clip_l14'][bucket]['n']
                cells = []
                for eslug, _ in ENCODERS:
                    b = split_by_enc[eslug][bucket]
                    cells.append(f"{pct(b['clean_t5'])}->{pct(b['mod_t5'])} ({pct(b['mod_t3'])})")
                lines.append(f'| {m_label} | `{suite}` | {blabel} ({n_b}) | ' + ' | '.join(cells) + ' |')

    out_md = EXP_DIR / 'EVAL_V3_SUMMARY.md'
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary_json = {
        'panels': {
            f'{m}_{s}': {
                es: {'overall': loaded[(m, s)]['encoders'][es]['overall'], 'by_family': loaded[(m, s)]['encoders'][es]['by_family']}
                for es, _ in ENCODERS
            }
            for m, _ in METHODS for s in SUITES if (m, s) in loaded
        },
        'explicit_vs_implicit': exp_imp_json,
    }
    (EXP_DIR / 'EVAL_V3_SUMMARY.json').write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[SUMMARY WRITTEN] {out_md}')


if __name__ == '__main__':
    main()
