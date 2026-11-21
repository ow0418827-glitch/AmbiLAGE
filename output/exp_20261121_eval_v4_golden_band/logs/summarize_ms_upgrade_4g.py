#!/usr/bin/env python3
"""Set 2 四组多尺度破网格消融全对照汇总 (`01_camera`, `02_chair`, `06_jeans`, `07_shampoo`)。

对比对象：
- 仅多尺度 v1 (原版): `ambilage_multiscale_only_set2_4g`
- 多尺度 v2 (纯几何+退火): `ambilage_ms_v2_geom_set2_4g`
- 多尺度 v3 (几何+轻量光度): `ambilage_ms_v3_geom_photo_set2_4g`
- EoT=ON (含防御EoT): `ambilage_eot_on_set2_4g`
- 新版 VI (paper_minimal_margin): `vi_paper_minimal_margin_set2_4g`

展示层严格遵循用户要求：
- 表格仅对比 Top-3 与 Top-5 命中结果（不展示 Top-1，也不展示平均名次）；
- 正常检索命中/命中率与净提升（修改 - 原图）、纯新增进榜（扣除都进榜后）并列展示；
- 底层 MS_UPGRADE_4G_SUMMARY.json 完整保留 Top-1、平均名次、中位名次、MRR 与余弦得分变化供深层诊断。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

EXP_DIR = Path('/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band')
EVAL_ROOT = EXP_DIR / 'eval_eot_ablation_4g'

GROUPS = ('01_camera', '02_chair', '06_jeans', '07_shampoo')
SPECS = (
    ('ambilage_multiscale_only_set2_4g', '仅多尺度 v1 (原版)'),
    ('ambilage_ms_v2_geom_set2_4g', '多尺度 v2 (纯几何+退火)'),
    ('ambilage_ms_v3_geom_photo_set2_4g', '多尺度 v3 (几何+轻量光度)'),
    ('ambilage_eot_on_set2_4g', 'EoT=ON (含防御EoT)'),
    ('vi_paper_minimal_margin_set2_4g', '新版 VI (paper_minimal_margin)'),
)
SUITES = (
    ('zero_overlap_v4', '0% 重合 (`zero_overlap_v4`)'),
    ('overlap_30_v4', '30% 重合 (`overlap_30_v4`)'),
    ('training_oracle_v4', '100% 重合 (`training_oracle_v4`)'),
)
ENCODERS = (
    ('01_clip_l14', 'CLIP-ViT-L/14 (白盒)'),
    ('02_openclip_bigG14', 'OpenCLIP-bigG-14 (黑盒)'),
    ('03_siglip_so400m', 'SigLIP-SO400M (黑盒)'),
)
BB_ENCODERS = ('02_openclip_bigG14', '03_siglip_so400m')


def pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def signed_pp(x: float) -> str:
    return f'{x * 100:+.1f}%'


def compute_metrics_for_pairs(pairs: list[tuple[int, int, float, float]]) -> dict:
    n = len(pairs)
    assert n > 0
    clean_ranks = [float(cr) for cr, _, _, _ in pairs]
    mod_ranks = [float(mr) for _, mr, _, _ in pairs]
    clean_scores = [float(cs) for _, _, cs, _ in pairs]
    mod_scores = [float(ms) for _, _, _, ms in pairs]
    out: dict = {
        'query_count': n,
        'mean_rank_clean': sum(clean_ranks) / n,
        'mean_rank_modified': sum(mod_ranks) / n,
        'mean_rank_gain': (sum(clean_ranks) - sum(mod_ranks)) / n,
        'median_rank_clean': float(statistics.median(clean_ranks)),
        'median_rank_modified': float(statistics.median(mod_ranks)),
        'mrr_clean': sum(1.0 / r for r in clean_ranks) / n,
        'mrr_modified': sum(1.0 / r for r in mod_ranks) / n,
        'mean_score_clean': sum(clean_scores) / n,
        'mean_score_modified': sum(mod_scores) / n,
        'mean_score_gain': (sum(mod_scores) - sum(clean_scores)) / n,
    }
    for k in (1, 3, 5):
        ch = sum(1 for cr, _, _, _ in pairs if cr <= k)
        mh = sum(1 for _, mr, _, _ in pairs if mr <= k)
        bh = sum(1 for cr, mr, _, _ in pairs if cr <= k and mr <= k)
        nh = sum(1 for cr, mr, _, _ in pairs if cr > k and mr <= k)
        lh = sum(1 for cr, mr, _, _ in pairs if cr <= k and mr > k)
        net = mh - ch
        out[f'top{k}'] = {
            'clean_hits': ch,
            'modified_hits': mh,
            'net_gain': net,
            'both_hits': bh,
            'new_hits': nh,
            'lost_hits': lh,
            'clean_rate': ch / n,
            'modified_rate': mh / n,
            'net_gain_rate': net / n,
        }
    return out


def load_panel_pairs(spec_tag: str, suite: str, eslug: str) -> tuple[dict[str, dict], dict]:
    by_group: dict[str, dict] = {}
    all_pairs: list[tuple[int, int, float, float]] = []
    panel_dir = EVAL_ROOT / f'{spec_tag}_{suite}' / eslug
    for g in GROUPS:
        ev = json.loads((panel_dir / 'groups' / g / 'evaluation.json').read_text(encoding='utf-8'))
        cq = next(x for x in ev if x['bank'] == 'clean')['queries']
        mq = next(x for x in ev if x['bank'] == 'modified')['queries']
        assert len(cq) == len(mq) == 20, f'{spec_tag}/{suite}/{eslug}/{g} query count {len(cq)}/{len(mq)}'
        pairs = [
            (int(c['rank']), int(m['rank']), float(c['score']), float(m['score']))
            for c, m in zip(cq, mq)
        ]
        for c, m in zip(cq, mq):
            assert c['task_id'] == m['task_id']
        by_group[g] = compute_metrics_for_pairs(pairs)
        all_pairs.extend(pairs)
    return by_group, compute_metrics_for_pairs(all_pairs)


def fmt_hits(m: dict, kkey: str) -> str:
    d = m[kkey]
    n = m['query_count']
    return f"{d['modified_hits']}/{n} ({pct(d['modified_rate'])})"


def fmt_clean(m: dict, kkey: str) -> str:
    d = m[kkey]
    n = m['query_count']
    return f"{d['clean_hits']}/{n} ({pct(d['clean_rate'])})"


def fmt_gain(m: dict, kkey: str) -> str:
    d = m[kkey]
    return (
        f"{d['net_gain']:+d} ({signed_pp(d['net_gain_rate'])}) "
        f"/ +{d['new_hits']} ({d['both_hits']}, -{d['lost_hits']})"
    )


def main() -> int:
    summary: dict = {
        'groups': list(GROUPS),
        'specs': {tag: label for tag, label in SPECS},
        'suites': [s for s, _ in SUITES],
        'encoders': [e for e, _ in ENCODERS],
        'panels': {},
        'mean_blackbox': {},
        'naturalness': {},
    }

    from src.paths import resolve_evaluation_base
    from src.experiments.visual_inception_category_evaluation import load_frozen_suite
    base = resolve_evaluation_base('golden_band_9groups')

    out_roots = {
        'ambilage_multiscale_only_set2_4g': Path('/content/AmbiLAGE/output/exp_20261121_ambilage_multiscale_only_golden_band_9groups'),
        'ambilage_ms_v2_geom_set2_4g': Path('/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_golden_band_9groups'),
        'ambilage_ms_v3_geom_photo_set2_4g': Path('/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v3_geom_photo_golden_band_9groups'),
    }
    for tag, root in out_roots.items():
        summary['naturalness'][tag] = {}
        suite_items = load_frozen_suite(root, method='ambilage', base=base, selected_groups=list(GROUPS))
        for item in suite_items:
            g = item['group']
            meta = item['metadata']
            gate = meta['final_naturalness']['metrics']
            summary['naturalness'][tag][g] = {
                'passed': meta['final_naturalness'].get('passed'),
                'source_ssim': gate.get('source_ssim'),
                'source_mae': gate.get('source_mae_levels'),
                'fg_chroma_tv_levels': gate.get('fg_chroma_tv_levels'),
                'eot_paths': meta['training_objective'].get('eot_paths'),
            }

    for spec_tag, _ in SPECS:
        summary['panels'][spec_tag] = {}
        summary['mean_blackbox'][spec_tag] = {}
        for suite, _ in SUITES:
            summary['panels'][spec_tag][suite] = {}
            bb_pairs: list[tuple[int, int, float, float]] = []
            for eslug, _ in ENCODERS:
                by_g, ov = load_panel_pairs(spec_tag, suite, eslug)
                summary['panels'][spec_tag][suite][eslug] = {
                    'overall': ov,
                    'groups': by_g,
                }
                if eslug in BB_ENCODERS:
                    panel_dir = EVAL_ROOT / f'{spec_tag}_{suite}' / eslug
                    for g in GROUPS:
                        ev = json.loads((panel_dir / 'groups' / g / 'evaluation.json').read_text(encoding='utf-8'))
                        cq = next(x for x in ev if x['bank'] == 'clean')['queries']
                        mq = next(x for x in ev if x['bank'] == 'modified')['queries']
                        for c, m in zip(cq, mq):
                            bb_pairs.append((int(c['rank']), int(m['rank']), float(c['score']), float(m['score'])))
            summary['mean_blackbox'][spec_tag][suite] = compute_metrics_for_pairs(bb_pairs)

    out_json = EXP_DIR / 'MS_UPGRADE_4G_SUMMARY.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    cols = ['测试集', '指标', '原图正常命中']
    for _, label in SPECS:
        cols += [f'**{label} 修改命中**', f'{label} 净提升 / 纯新增']
    hdr = '| ' + ' | '.join(cols) + ' |'
    sep = '|---|---' + '|---:' * (len(cols) - 2) + '|'

    def _build_rows(get_m):
        rows = []
        for suite, slabel in SUITES:
            base_m = get_m(SPECS[0][0], suite)
            for kkey, klabel in (('top3', 'Top-3'), ('top5', 'Top-5')):
                cells = [slabel, f'**{klabel}**', fmt_clean(base_m, kkey)]
                for tag, _ in SPECS:
                    m = get_m(tag, suite)
                    cells += [f'**{fmt_hits(m, kkey)}**', fmt_gain(m, kkey)]
                rows.append('| ' + ' | '.join(cells) + ' |')
        return rows

    lines: list[str] = [
        '# Set 2 四组多尺度破网格演进对比报告（`01_camera` / `02_chair` / `06_jeans` / `07_shampoo`）',
        '',
        '> **对照说明**：所有参评方法均使用完全相同的 4 张 Set 2 原图及 `--replacement-groups selected` 同库替换口径同场评测（全部独立重算误差 `max_score_error = 0.0`）。',
        '> **指标说明**：表格严格仅对比 **Top-3 与 Top-5**，并列展示**原图/修改图正常检索命中**与**净提升（修改 − 原图） / +纯新增 (都进榜, -退榜)**。',
        '',
        '## 一、白盒守恒检查（`CLIP-ViT-L/14`，80 条查询/套）',
        '',
        hdr,
        sep,
    ] + _build_rows(lambda tag, s: summary['panels'][tag][s]['01_clip_l14']['overall'])

    lines += [
        '',
        '## 二、黑盒均值对比（`OpenCLIP-bigG-14` + `SigLIP-SO400M`，合计 160 条查询/套）',
        '',
        hdr,
        sep,
    ] + _build_rows(lambda tag, s: summary['mean_blackbox'][tag][s])

    lines += ['', '## 三、黑盒单编码器分项（80 条查询/套）']
    for eslug, elabel in ENCODERS:
        if eslug == '01_clip_l14':
            continue
        lines += ['', f'### {elabel}（80 条查询）', '', hdr, sep]
        lines += _build_rows(lambda tag, s, e=eslug: summary['panels'][tag][s][e]['overall'])

    lines += ['', '## 四、四组逐组对照明细（每组 20 条查询，格式：`T3: 命中(净提升) / T5: 命中(净提升)`）']
    gcols = ['组别', '编码器', '原图 (T3 / T5)'] + [f'{label} (T3 / T5)' for _, label in SPECS]
    ghdr = '| ' + ' | '.join(gcols) + ' |'
    gsep = '|' + '---|' * len(gcols)
    for suite, slabel in SUITES:
        lines += ['', f'### {slabel}', '', ghdr, gsep]
        for g in GROUPS:
            for eslug, elabel in ENCODERS:
                base_g = summary['panels'][SPECS[0][0]][suite][eslug]['groups'][g]
                c_cell = f"{base_g['top3']['clean_hits']} / {base_g['top5']['clean_hits']}"
                row = [f'`{g}`', elabel, c_cell]
                for tag, _ in SPECS:
                    gm = summary['panels'][tag][suite][eslug]['groups'][g]
                    row.append(
                        f"{gm['top3']['modified_hits']} ({gm['top3']['net_gain']:+d}) / "
                        f"{gm['top5']['modified_hits']} ({gm['top5']['net_gain']:+d})"
                    )
                lines.append('| ' + ' | '.join(row) + ' |')

    lines += ['', '## 五、成图自然度门控指标（v1 vs v2 vs v3）', '', '| 组别 | 档位 | Passed | Source SSIM | Source MAE(255) | Foreground Chroma TV | EoT Paths |', '|---|---|---|---:|---:|---:|---|']
    for g in GROUPS:
        for tag, label in SPECS[:3]:
            nt = summary['naturalness'][tag][g]
            lines.append(
                f"| `{g}` | {label} | `{nt['passed']}` | {nt['source_ssim']:.4f} | {nt['source_mae']:.2f} | {nt['fg_chroma_tv_levels']:.4f} | `{' + '.join(nt['eot_paths'])}` |"
            )

    md_text = '\n'.join(lines) + '\n'
    assert 'Top-1' not in md_text and '平均名次' not in md_text and '均名' not in md_text
    out_md = EXP_DIR / 'MS_UPGRADE_4G_SUMMARY.md'
    out_md.write_text(md_text, encoding='utf-8')
    print(f'[WRITTEN] {out_md}')
    print(f'[WRITTEN] {out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
