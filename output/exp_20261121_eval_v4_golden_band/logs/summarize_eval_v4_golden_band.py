#!/usr/bin/env python3
"""汇总黄金先验区间新选 9 组 (Set 2: golden_band_9groups) 的 VI (1.0/0.0) 与 AmbiLAGE (jw=0.15, tw=1.50) 评测结果。
展示层仅保留核心三要素：原图检索排名/命中、修改图检索排名/命中、净提升（不统计 Top-1）；
底层 EVAL_V4_GOLDEN_BAND_SUMMARY.json 完整保留 Top-1、MRR、中位名次、ΔCos、Conv/Retain/NetCap 等全量指标供后续实验分析。
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path('/content/AmbiLAGE')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.visual_inception_category_evaluation import _compute_headroom_fields, _paired_summary

EXP_DIR = Path('/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band')
SET2_BASE = PROJECT_ROOT / 'resources/evaluation_base_golden_band'
SET1_BASE = PROJECT_ROOT / 'resources/evaluation_base'
SET1_SUMMARY_JSON = Path('/content/AmbiLAGE/output/exp_20261121_eval_v4_cleaned/EVAL_V4_CLEANED_SUMMARY.json')
METHODS = (
    ('vi_10_00_set2', 'Visual Inception (默认 1.0/0.0)', 'vi_10_00'),
    ('ambilage_set2', 'AmbiLAGE (jw=0.15, tw=1.50, EoT开启)', 'ambilage'),
    ('ambilage_noeot_set2', 'AmbiLAGE (jw=0.15, tw=1.50, EoT关闭)', 'ambilage'),
)
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
PURE_REAL_GROUPS = [
    '01_camera', '03_cookies', '04_curtain', '05_hair_extension',
    '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush',
]
ALL_GROUPS = [
    '01_camera', '02_chair', '03_cookies', '04_curtain',
    '05_hair_extension', '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush',
]


def pct(x: float | None) -> str:
    if x is None:
        return '—'
    return f'{x * 100:.1f}%'


def signed_pp(x: float | None) -> str:
    if x is None:
        return '—'
    return f'{x * 100:+.1f}%'


def compute_rank_score_stats(panel_dir: Path, eslug: str) -> dict:
    clean_ranks, mod_ranks, clean_scores, mod_scores = [], [], [], []
    clean_rank_map, mod_rank_map = {}, {}
    for g in ALL_GROUPS:
        ev = json.loads((panel_dir / eslug / 'groups' / g / 'evaluation.json').read_text(encoding='utf-8'))
        cq = next(x for x in ev if x['bank'] == 'clean')['queries']
        mq = next(x for x in ev if x['bank'] == 'modified')['queries']
        for c, m in zip(cq, mq):
            tid = c['task_id']
            cr, mr = int(c['rank']), int(m['rank'])
            clean_ranks.append(float(cr))
            mod_ranks.append(float(mr))
            clean_scores.append(float(c['score']))
            mod_scores.append(float(m['score']))
            clean_rank_map[(g, tid)] = cr
            mod_rank_map[(g, tid)] = mr
    clean_mrr = sum(1.0 / r for r in clean_ranks) / len(clean_ranks)
    mod_mrr = sum(1.0 / r for r in mod_ranks) / len(mod_ranks)
    paired = _paired_summary(clean_rank_map, mod_rank_map)
    return dict(
        clean_median_rank=float(statistics.median(clean_ranks)),
        mod_median_rank=float(statistics.median(mod_ranks)),
        clean_mrr=float(clean_mrr),
        mod_mrr=float(mod_mrr),
        mrr_gain=float(mod_mrr - clean_mrr),
        clean_mean_score=float(sum(clean_scores) / len(clean_scores)),
        mod_mean_score=float(sum(mod_scores) / len(mod_scores)),
        score_gain=float(sum(m - c for c, m in zip(clean_scores, mod_scores)) / len(clean_scores)),
        paired=paired,
    )


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


def load_selection_metadata() -> dict[str, dict]:
    m2 = json.loads((SET2_BASE / 'manifest.json').read_text(encoding='utf-8'))
    m1 = json.loads((SET1_BASE / 'manifest.json').read_text(encoding='utf-8'))
    out = {}
    for g in ALL_GROUPS:
        s2 = m2['groups'][g]
        s1 = m1['groups'][g]
        ev_path = SET2_BASE / 'groups' / g / 'selection_evidence.json'
        ev = json.loads(ev_path.read_text(encoding='utf-8')) if ev_path.is_file() else {}
        draw = ev.get('draw') or {}
        row = next((c for c in ev.get('candidates', []) if c.get('source_id') == s2['source_id']), {})
        out[g] = dict(
            set1_source_id=s1['source_id'],
            set2_source_id=s2['source_id'],
            target_id=s2['target_id'],
            shape_similarity=s2.get('shape_similarity'),
            calibrated_threshold=s2.get('calibrated_threshold'),
            passes_calibrated_threshold=bool(s2.get('passes_calibrated_threshold', True)),
            shape_threshold_fallback_used=bool(s2.get('shape_threshold_fallback_used', ev.get('shape_threshold_fallback_used', False))),
            public_prior_rank=s2.get('public_prior_rank', row.get('public_prior_rank')),
            eligible_count=draw.get('eligible_count'),
            candidate_count=len(ev.get('candidates', [])),
        )
    return out


def _append_method_tables(lines: list[str], mtag: str, mlabel: str, loaded_m: dict[str, dict], sel_meta: dict[str, dict]) -> None:
    lines += [
        '',
        f'## {mlabel} —— Set 2 核心评测结果（原图 vs 修改图 vs 净提升，每套 180 条查询）',
    ]
    for eslug, elabel in ENCODERS:
        lines += [
            '',
            f'### {mlabel} | {elabel}（总查询数 = 180 条）',
            '',
            '| 评测集（重复率） | 指标维度 | 原图检索结果 | 修改图检索结果 | 净提升（修改 - 原图） | 纯新增进榜（扣除都进榜后） |',
            '|---|---|---:|---:|---:|---:|',
        ]
        for suite, slabel in SUITES:
            ov = loaded_m[suite]['encoders'][eslug]['overall']
            t3, t5 = ov['top3'], ov['top5']
            c_mr, m_mr = float(ov['mean_rank_clean']), float(ov['mean_rank_modified'])
            g_mr = c_mr - m_mr
            kept3 = t3['clean_hits'] - t3['lost_hits']
            kept5 = t5['clean_hits'] - t5['lost_hits']
            net3 = t3['modified_hits'] - t3['clean_hits']
            net5 = t5['modified_hits'] - t5['clean_hits']
            lines.append(
                f"| {slabel} | **平均检索名次** | 第 {c_mr:.2f} 名 | **第 {m_mr:.2f} 名** | **提升 {g_mr:+.2f} 名** | — |"
            )
            lines.append(
                f"| {slabel} | **Top-3 命中** | {t3['clean_hits']}/180 ({pct(t3['clean_rate'])}) | "
                f"**{t3['modified_hits']}/180 ({pct(t3['modified_rate'])})** | "
                f"**{net3:+d} 条 ({signed_pp(t3['gain'])})** `[+{t3['new_hits']}/-{t3['lost_hits']}]` | "
                f"+{t3['new_hits']} 条 (都进榜 {kept3} 条) |"
            )
            lines.append(
                f"| {slabel} | **Top-5 命中** | {t5['clean_hits']}/180 ({pct(t5['clean_rate'])}) | "
                f"**{t5['modified_hits']}/180 ({pct(t5['modified_rate'])})** | "
                f"**{net5:+d} 条 ({signed_pp(t5['gain'])})** `[+{t5['new_hits']}/-{t5['lost_hits']}]` | "
                f"+{t5['new_hits']} 条 (都进榜 {kept5} 条) |"
            )

    lines += [
        '',
        f'### {mlabel} | 黑盒均值 Mean Black-Box（`OpenCLIP-bigG-14` + `SigLIP-SO400M` 合计 360 条查询）',
        '',
        '| 评测集（重复率） | 指标维度 | 原图检索结果 | 修改图检索结果 | 净提升（修改 - 原图） | 纯新增进榜（扣除都进榜后） |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for suite, slabel in SUITES:
        b_ov = loaded_m[suite]['encoders']['02_openclip_bigG14']['overall']
        s_ov = loaded_m[suite]['encoders']['03_siglip_so400m']['overall']
        c_mr = 0.5 * (b_ov['mean_rank_clean'] + s_ov['mean_rank_clean'])
        m_mr = 0.5 * (b_ov['mean_rank_modified'] + s_ov['mean_rank_modified'])
        g_mr = c_mr - m_mr
        lines.append(
            f"| {slabel} | **平均检索名次** | 第 {c_mr:.2f} 名 | **第 {m_mr:.2f} 名** | **提升 {g_mr:+.2f} 名** | — |"
        )
        for k, klabel in (('top3', 'Top-3 命中'), ('top5', 'Top-5 命中')):
            ch = b_ov[k]['clean_hits'] + s_ov[k]['clean_hits']
            mh = b_ov[k]['modified_hits'] + s_ov[k]['modified_hits']
            nh = b_ov[k]['new_hits'] + s_ov[k]['new_hits']
            lh = b_ov[k]['lost_hits'] + s_ov[k]['lost_hits']
            kept = ch - lh
            net = mh - ch
            cr = ch / 360.0
            mr = mh / 360.0
            gr = net / 360.0
            lines.append(
                f"| {slabel} | **{klabel}** | {ch}/360 ({pct(cr)}) | **{mh}/360 ({pct(mr)})** | "
                f"**{net:+d} 条 ({signed_pp(gr)})** `[+{nh}/-{lh}]` | +{nh} 条 (都进榜 {kept} 条) |"
            )

    lines += [
        '',
        f'### {mlabel} | 9 组逐组明细（每组 20 条查询，原图 -> 修改图 (净提升): 均名 \\| T3 \\| T5）',
    ]
    for suite, slabel in SUITES:
        panel = loaded_m[suite]
        lines += [
            '',
            f'#### {mlabel} - {slabel}',
            '',
            '| 组别 | Set 2 原图 | CLIP-L/14 (均名 \\| T3 \\| T5) | OpenCLIP-bigG-14 (均名 \\| T3 \\| T5) | SigLIP-SO400M (均名 \\| T3 \\| T5) |',
            '|---|---|---|---|---|',
        ]
        for g in ALL_GROUPS:
            cells = []
            for eslug, _ in ENCODERS:
                ov = panel['encoders'][eslug]['groups'][g]['overall']
                t3, t5 = ov['top3'], ov['top5']
                c_mr, m_mr = ov['mean_rank_clean'], ov['mean_rank_modified']
                net3 = t3['modified_hits'] - t3['clean_hits']
                net5 = t5['modified_hits'] - t5['clean_hits']
                cells.append(
                    f"{c_mr:.1f}->{m_mr:.1f} ({c_mr-m_mr:+.1f}) \\| "
                    f"{t3['clean_hits']}->{t3['modified_hits']} ({net3:+d}) \\| "
                    f"{t5['clean_hits']}->{t5['modified_hits']} ({net5:+d})"
                )
            lines.append(f"| `{g}` | `{sel_meta[g]['set2_source_id']}` | " + ' | '.join(cells) + ' |')


def main() -> None:
    loaded: dict[str, dict[str, dict]] = {}
    extra_stats: dict[str, dict] = {}
    active_methods = []
    for mtag, mlabel, s1_prefix in METHODS:
        has_all = all(
            (EXP_DIR / 'eval' / f'{mtag}_{suite}' / 'transfer_panel_summary.json').is_file()
            for suite, _ in SUITES
        )
        if not has_all:
            continue
        active_methods.append((mtag, mlabel, s1_prefix))
        loaded[mtag] = {}
        for suite, _ in SUITES:
            panel_dir = EXP_DIR / 'eval' / f'{mtag}_{suite}'
            loaded[mtag][suite] = json.loads((panel_dir / 'transfer_panel_summary.json').read_text(encoding='utf-8'))
            for eslug, _ in ENCODERS:
                extra_stats[f'{mtag}_{suite}_{eslug}'] = compute_rank_score_stats(panel_dir, eslug)

    if not active_methods:
        raise FileNotFoundError('未找到任何已完成评测的 Set 2 方法汇总文件')

    timings_path = EXP_DIR / 'logs' / 'timings.json'
    timings = json.loads(timings_path.read_text(encoding='utf-8')) if timings_path.is_file() else {}
    sel_meta = load_selection_metadata()

    lines = [
        '# 黄金先验区间新选 9 组 (`golden_band_9groups`, Set 2) 核心检索评测报告',
        '',
        '> **统计口径说明**：',
        '> 1. 本报告不统计 Top-1，仅聚焦三大核心数据：**原图检索排名/命中**、**修改图检索排名/命中**、**净提升**。',
        '> 2. **净提升（修改 - 原图）**：等于 `修改图进入 Top-K 的数量 - 原图进入 Top-K 的数量`（其中原图与修改图**都**进入排名的重叠部分自动抵消，即 `净提升 = 新增进榜数 - 退榜数`）；同时列出**纯新增进榜数**（即从修改图进榜数中仅扣除“原图和修改图都进入排名”的数量 `+新进`）与**平均名次净提升**（`原图均名 - 修改图均名`，正值代表名次前移提升）。',
        '> 3. 其余细粒度实验分析指标（Top-1、中位名次、MRR、余弦相似度变化、Conv/Retain/NetCap 等）均已完整保存在同目录 `EVAL_V4_GOLDEN_BAND_SUMMARY.json` 中。',
        '',
        '## 一、Set 2 (`golden_band_9groups`) 9 组新选原图明细',
        '',
        '| 组别 | 目标商品 | Set 1 原图 | Set 2 新选原图 | 公开先验名次 (`/20`) | 前景外形分 / 门槛 | 门槛状态 |',
        '|---|---|---|---|---:|---:|---|',
    ]
    for g in ALL_GROUPS:
        sm = sel_meta[g]
        gate_status = '通过' if sm['passes_calibrated_threshold'] and not sm['shape_threshold_fallback_used'] else '带内回退(未过门槛)'
        lines.append(
            f"| `{g}` | `{sm['target_id']}` | `{sm['set1_source_id']}` | `{sm['set2_source_id']}` | "
            f"{sm['public_prior_rank']} | {sm['shape_similarity']:.4f} / {sm['calibrated_threshold']:.4f} | {gate_status} |"
        )

    for cmp_tag, cmp_title in (
        ('ambilage_noeot_set2', 'AmbiLAGE (`jw=0.15, tw=1.50, EoT关闭`)'),
        ('ambilage_set2', 'AmbiLAGE (`jw=0.15, tw=1.50, EoT开启`)'),
    ):
        if 'vi_10_00_set2' not in loaded or cmp_tag not in loaded:
            continue
        lines += [
            '',
            f'## 二、Set 2 上 {cmp_title} vs Visual Inception (`1.0/0.0`) 同场直接对比',
            '',
            f'| 评测集（重复率） | 编码器 / 口径 | 原图检索结果 (均名 \\| T3 \\| T5) | VI (1.0/0.0) 修改图 (净提升) | **{cmp_title} 修改图 (净提升)** | **AmbiLAGE 领先 VI 幅度** |',
            '|---|---|---:|---:|---:|---:|',
        ]
        for suite, slabel in SUITES:
            for eslug, elabel in ENCODERS:
                v_ov = loaded['vi_10_00_set2'][suite]['encoders'][eslug]['overall']
                a_ov = loaded[cmp_tag][suite]['encoders'][eslug]['overall']
                c_mr = v_ov['mean_rank_clean']
                v_mr, a_mr = v_ov['mean_rank_modified'], a_ov['mean_rank_modified']
                vt3, vt5 = v_ov['top3'], v_ov['top5']
                at3, at5 = a_ov['top3'], a_ov['top5']
                vnet3, vnet5 = vt3['modified_hits'] - vt3['clean_hits'], vt5['modified_hits'] - vt5['clean_hits']
                anet3, anet5 = at3['modified_hits'] - at3['clean_hits'], at5['modified_hits'] - at5['clean_hits']
                d3 = at3['modified_hits'] - vt3['modified_hits']
                d5 = at5['modified_hits'] - vt5['modified_hits']
                dmr = v_mr - a_mr
                lines.append(
                    f"| {slabel} | {elabel} | "
                    f"均名 {c_mr:.2f}<br>T3: {vt3['clean_hits']}/180 ({pct(vt3['clean_rate'])})<br>T5: {vt5['clean_hits']}/180 ({pct(vt5['clean_rate'])}) | "
                    f"均名 {v_mr:.2f} ({c_mr-v_mr:+.2f})<br>T3: {vt3['modified_hits']}/180 ({vnet3:+d}, {signed_pp(vt3['gain'])})<br>T5: {vt5['modified_hits']}/180 ({vnet5:+d}, {signed_pp(vt5['gain'])}) | "
                    f"**均名 {a_mr:.2f} ({c_mr-a_mr:+.2f})**<br>**T3: {at3['modified_hits']}/180 ({anet3:+d}, {signed_pp(at3['gain'])})**<br>**T5: {at5['modified_hits']}/180 ({anet5:+d}, {signed_pp(at5['gain'])})** | "
                    f"**均名 {dmr:+.2f} 名**<br>**T3: {d3:+d} 条 ({signed_pp(d3/180.0)})**<br>**T5: {d5:+d} 条 ({signed_pp(d5/180.0)})** |"
                )
            vb = loaded['vi_10_00_set2'][suite]['encoders']['02_openclip_bigG14']['overall']
            vs = loaded['vi_10_00_set2'][suite]['encoders']['03_siglip_so400m']['overall']
            ab = loaded[cmp_tag][suite]['encoders']['02_openclip_bigG14']['overall']
            as_ = loaded[cmp_tag][suite]['encoders']['03_siglip_so400m']['overall']
            c_mr = 0.5 * (vb['mean_rank_clean'] + vs['mean_rank_clean'])
            v_mr = 0.5 * (vb['mean_rank_modified'] + vs['mean_rank_modified'])
            a_mr = 0.5 * (ab['mean_rank_modified'] + as_['mean_rank_modified'])
            c3 = vb['top3']['clean_hits'] + vs['top3']['clean_hits']
            c5 = vb['top5']['clean_hits'] + vs['top5']['clean_hits']
            vm3 = vb['top3']['modified_hits'] + vs['top3']['modified_hits']
            vm5 = vb['top5']['modified_hits'] + vs['top5']['modified_hits']
            am3 = ab['top3']['modified_hits'] + as_['top3']['modified_hits']
            am5 = ab['top5']['modified_hits'] + as_['top5']['modified_hits']
            lines.append(
                f"| {slabel} | **黑盒均值 Mean BB (360条)** | "
                f"均名 {c_mr:.2f}<br>T3: {c3}/360 ({pct(c3/360.0)})<br>T5: {c5}/360 ({pct(c5/360.0)}) | "
                f"均名 {v_mr:.2f} ({c_mr-v_mr:+.2f})<br>T3: {vm3}/360 ({vm3-c3:+d}, {signed_pp((vm3-c3)/360.0)})<br>T5: {vm5}/360 ({vm5-c5:+d}, {signed_pp((vm5-c5)/360.0)}) | "
                f"**均名 {a_mr:.2f} ({c_mr-a_mr:+.2f})**<br>**T3: {am3}/360 ({am3-c3:+d}, {signed_pp((am3-c3)/360.0)})**<br>**T5: {am5}/360 ({am5-c5:+d}, {signed_pp((am5-c5)/360.0)})** | "
                f"**均名 {v_mr-a_mr:+.2f} 名**<br>**T3: {am3-vm3:+d} 条 ({signed_pp((am3-vm3)/360.0)})**<br>**T5: {am5-vm5:+d} 条 ({signed_pp((am5-vm5)/360.0)})** |"
            )

    for mtag, mlabel, _ in reversed(active_methods):
        _append_method_tables(lines, mtag, mlabel, loaded[mtag], sel_meta)

    set1_comparison = {}
    if SET1_SUMMARY_JSON.is_file():
        s1_doc = json.loads(SET1_SUMMARY_JSON.read_text(encoding='utf-8'))
        s1_panels = s1_doc.get('panels', {})
        s1_headroom = s1_doc.get('headroom_metrics', {})
        lines += [
            '',
            '## Set 2 (`golden_band_9groups`) vs Set 1 (`original_9groups`) 跨数据集核心对比',
            '',
            '| 方法 | 评测集 | 数据集批次 | CLIP-L/14 (原图 -> 修改图 \\| 净提升) | OpenCLIP-bigG-14 (原图 -> 修改图 \\| 净提升) | SigLIP-SO400M (原图 -> 修改图 \\| 净提升) | 黑盒均值 Mean BB (原图 -> 修改图 \\| 净提升) |',
            '|---|---|---|---|---|---|---|',
        ]
        for mtag, mlabel, s1_prefix in reversed(active_methods):
            for suite, slabel in SUITES:
                s1p = s1_panels.get(f'{s1_prefix}_{suite}')
                s2p = loaded[mtag][suite]['encoders']
                if not s1p:
                    continue
                for batch_label, src_dict in (('Set 1 (原始9组)', s1p), ('**Set 2 (新选9组)**', s2p)):
                    row_cells = []
                    bb_c_mr = bb_m_mr = 0.0
                    bb_c3 = bb_m3 = bb_c5 = bb_m5 = 0
                    for eslug, _ in ENCODERS:
                        ov = src_dict[eslug]['overall']
                        c_mr, m_mr = ov['mean_rank_clean'], ov['mean_rank_modified']
                        t3, t5 = ov['top3'], ov['top5']
                        net3 = t3['modified_hits'] - t3['clean_hits']
                        net5 = t5['modified_hits'] - t5['clean_hits']
                        row_cells.append(
                            f"均名 {c_mr:.2f}->{m_mr:.2f} ({c_mr-m_mr:+.2f})<br>"
                            f"T3: {t3['clean_hits']}->{t3['modified_hits']} ({net3:+d}, {signed_pp(t3['gain'])})<br>"
                            f"T5: {t5['clean_hits']}->{t5['modified_hits']} ({net5:+d}, {signed_pp(t5['gain'])})"
                        )
                        if eslug != '01_clip_l14':
                            bb_c_mr += 0.5 * c_mr
                            bb_m_mr += 0.5 * m_mr
                            bb_c3 += t3['clean_hits']
                            bb_m3 += t3['modified_hits']
                            bb_c5 += t5['clean_hits']
                            bb_m5 += t5['modified_hits']
                    bb_cell = (
                        f"均名 {bb_c_mr:.2f}->{bb_m_mr:.2f} ({bb_c_mr-bb_m_mr:+.2f})<br>"
                        f"T3: {bb_c3}/360->{bb_m3}/360 ({bb_m3-bb_c3:+d}, {signed_pp((bb_m3-bb_c3)/360.0)})<br>"
                        f"T5: {bb_c5}/360->{bb_m5}/360 ({bb_m5-bb_c5:+d}, {signed_pp((bb_m5-bb_c5)/360.0)})"
                    )
                    lines.append(f'| {mlabel} | {slabel} | {batch_label} | ' + ' | '.join(row_cells) + f' | {bb_cell} |')

                s1_bb_nc5 = s2_bb_nc5 = 0.0
                for eslug in ('02_openclip_bigG14', '03_siglip_so400m'):
                    o1 = s1p[eslug]['overall']
                    o2 = s2p[eslug]['overall']
                    h1 = s1_headroom.get(f'{s1_prefix}_{suite}_{eslug}')
                    o1_t5 = h1['top5'] if h1 else _compute_headroom_fields(
                        o1['top5']['clean_hits'], o1['top5']['modified_hits'],
                        o1['top5']['new_hits'], o1['top5']['lost_hits'], o1['query_count']
                    )
                    s1_bb_nc5 += 0.5 * o1_t5['net_headroom_capture']
                    s2_bb_nc5 += 0.5 * o2['top5']['net_headroom_capture']
                set1_comparison[f'{mtag}_{suite}'] = dict(s1_bb_nc5=s1_bb_nc5, s2_bb_nc5=s2_bb_nc5)

    split_json = {}
    for mtag, _, _ in active_methods:
        for suite, _ in SUITES:
            panel = loaded[mtag][suite]
            for gnames in (PURE_REAL_GROUPS, ['02_chair']):
                scope_rec = {eslug: aggregate_groups(panel['encoders'][eslug], gnames) for eslug, _ in ENCODERS}
                split_json[f'{mtag}_{suite}_{"pure8" if len(gnames) == 8 else "chair"}'] = scope_rec

    out_md = EXP_DIR / 'EVAL_V4_GOLDEN_BAND_SUMMARY.md'
    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary_json = {
        'timings': timings,
        'active_methods': [m[0] for m in active_methods],
        'selection_metadata': sel_meta,
        'extra_stats': extra_stats,
        'panels': {
            f'{mtag}_{suite}': {
                es: {
                    'overall': loaded[mtag][suite]['encoders'][es]['overall'],
                    'by_family': loaded[mtag][suite]['encoders'][es]['by_family'],
                    'groups': loaded[mtag][suite]['encoders'][es]['groups'],
                }
                for es, _ in ENCODERS
            }
            for mtag, _, _ in active_methods
            for suite, _ in SUITES
        },
        'pure8_vs_chair': split_json,
        'set1_vs_set2': set1_comparison,
    }
    (EXP_DIR / 'EVAL_V4_GOLDEN_BAND_SUMMARY.json').write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'[SUMMARY WRITTEN] {out_md}')


if __name__ == '__main__':
    main()
