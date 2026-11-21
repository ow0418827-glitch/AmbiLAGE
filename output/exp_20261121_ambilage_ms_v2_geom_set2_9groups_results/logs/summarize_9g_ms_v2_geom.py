#!/usr/bin/env python3
"""Set 2 完整九组 `ms_v2_geom`（AmbiLAGE 默认接口）全量评测对照汇总。

对比对象（均使用 `--group all --replacement-groups all` 九组同库全替口径）：
- `ambilage_ms_v2_geom_set2`: AmbiLAGE 默认 `ms_v2_geom`（九组）
- `ambilage_noeot_set2`: AmbiLAGE EoT=OFF（无任何EoT，九组）
- `vi_paper_minimal_margin_set2`: 新版 VI (`paper_minimal_margin`，九组)
- `vi_10_00_set2`: 旧版 VI (1.0/0.0，九组)

展示层严格遵循用户要求：
- Markdown 表格仅对比 Top-3 与 Top-5 命中结果（不展示 Top-1，也不展示平均名次）；
- 正常检索命中/命中率与净提升（修改 - 原图）、纯新增进榜（扣除都进榜后）并列展示；
- 底层 SUMMARY_9G_MS_V2_GEOM.json 完整保留 Top-1、平均名次、MRR 与余弦得分增益。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.experiments.visual_inception_category_evaluation import load_frozen_suite
from src.paths import resolve_evaluation_base

RESULTS_DIR = Path('/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_set2_9groups_results')
BASE_EVAL_ROOT = Path('/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band/eval')
IMAGES_ROOT = Path('/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_golden_band_9groups')

GROUPS = (
    '01_camera',
    '02_chair',
    '03_cookies',
    '04_curtain',
    '05_hair_extension',
    '06_jeans',
    '07_shampoo',
    '08_speaker',
    '09_toothbrush',
)
SPECS = (
    ('ambilage_ms_v2_geom_set2', 'AmbiLAGE 默认 (ms_v2_geom)', RESULTS_DIR / 'eval'),
    ('ambilage_noeot_set2', 'AmbiLAGE EoT=OFF (无EoT)', BASE_EVAL_ROOT),
    ('vi_paper_minimal_margin_set2', '新版 VI (paper_minimal_margin)', BASE_EVAL_ROOT),
    ('vi_10_00_set2', '旧版 VI (1.0/0.0)', BASE_EVAL_ROOT),
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


def _spec_ready(tag: str, eval_root: Path) -> bool:
    for suite, _ in SUITES:
        for eslug, _ in ENCODERS:
            for g in GROUPS:
                if not (eval_root / f'{tag}_{suite}' / eslug / 'groups' / g / 'evaluation.json').is_file():
                    return False
    return True


def load_panel_pairs(tag: str, eval_root: Path, suite: str, eslug: str) -> tuple[dict[str, dict], dict]:
    by_group: dict[str, dict] = {}
    all_pairs: list[tuple[int, int, float, float]] = []
    panel_dir = eval_root / f'{tag}_{suite}' / eslug
    for g in GROUPS:
        ev = json.loads((panel_dir / 'groups' / g / 'evaluation.json').read_text(encoding='utf-8'))
        cq = next(x for x in ev if x['bank'] == 'clean')['queries']
        mq = next(x for x in ev if x['bank'] == 'modified')['queries']
        assert len(cq) == len(mq) == 20, f'{tag}/{suite}/{eslug}/{g}: {len(cq)}/{len(mq)}'
        pairs = []
        for c, m in zip(cq, mq):
            assert c['task_id'] == m['task_id']
            pairs.append((int(c['rank']), int(m['rank']), float(c['score']), float(m['score'])))
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
    active_specs = [(tag, label, root) for tag, label, root in SPECS if _spec_ready(tag, root)]
    assert active_specs and active_specs[0][0] == 'ambilage_ms_v2_geom_set2'

    base = resolve_evaluation_base('golden_band_9groups')
    suite_items = load_frozen_suite(IMAGES_ROOT, method='ambilage', base=base, selected_groups=list(GROUPS))
    nat_info = {}
    for item in suite_items:
        g = item['group']
        meta = item['metadata']
        m = meta['final_naturalness']['metrics']
        nat_info[g] = {
            'passed': meta['final_naturalness'].get('passed'),
            'source_ssim': m.get('source_ssim'),
            'source_mae_levels': m.get('source_mae_levels'),
            'fg_rgb_source_mae_levels': m.get('fg_rgb_source_mae_levels'),
            'fg_chroma_tv_levels': m.get('fg_chroma_tv_levels'),
            'eot_paths': meta['training_objective'].get('eot_paths'),
        }

    summary: dict = {
        'groups': list(GROUPS),
        'specs': {tag: label for tag, label, _ in active_specs},
        'suites': [s for s, _ in SUITES],
        'encoders': [e for e, _ in ENCODERS],
        'panels': {},
        'mean_blackbox': {},
        'naturalness': nat_info,
    }

    for tag, _, eval_root in active_specs:
        summary['panels'][tag] = {}
        summary['mean_blackbox'][tag] = {}
        for suite, _ in SUITES:
            summary['panels'][tag][suite] = {}
            bb_pairs: list[tuple[int, int, float, float]] = []
            for eslug, _ in ENCODERS:
                by_g, ov = load_panel_pairs(tag, eval_root, suite, eslug)
                summary['panels'][tag][suite][eslug] = {'overall': ov, 'groups': by_g}
                if eslug in BB_ENCODERS:
                    panel_dir = eval_root / f'{tag}_{suite}' / eslug
                    for g in GROUPS:
                        ev = json.loads((panel_dir / 'groups' / g / 'evaluation.json').read_text(encoding='utf-8'))
                        cq = next(x for x in ev if x['bank'] == 'clean')['queries']
                        mq = next(x for x in ev if x['bank'] == 'modified')['queries']
                        for c, m in zip(cq, mq):
                            bb_pairs.append((int(c['rank']), int(m['rank']), float(c['score']), float(m['score'])))
            summary['mean_blackbox'][tag][suite] = compute_metrics_for_pairs(bb_pairs)

    out_json = RESULTS_DIR / 'SUMMARY_9G_MS_V2_GEOM.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    cols = ['测试集', '指标', '原图正常命中']
    for _, label, _ in active_specs:
        cols += [f'**{label} 修改命中**', f'{label} 净提升 / 纯新增']
    hdr = '| ' + ' | '.join(cols) + ' |'
    sep = '|---|---' + '|---:' * (len(cols) - 2) + '|'

    def _build_rows(get_m):
        rows = []
        for suite, slabel in SUITES:
            base_m = get_m(active_specs[0][0], suite)
            for kkey, klabel in (('top3', 'Top-3'), ('top5', 'Top-5')):
                cells = [slabel, f'**{klabel}**', fmt_clean(base_m, kkey)]
                for tag, _, _ in active_specs:
                    m = get_m(tag, suite)
                    cells += [f'**{fmt_hits(m, kkey)}**', fmt_gain(m, kkey)]
                rows.append('| ' + ' | '.join(cells) + ' |')
        return rows

    lines = [
        '# Set 2 完整九组评测对比报告（AmbiLAGE 默认 `ms_v2_geom`）',
        '',
        '> **对照说明**：所有方法均为 Set 2 (`golden_band_9groups`) 完整 9 组同时替库（`--group all --replacement-groups all`，每编码器 180 条查询/套，黑盒均值 360 条/套）。',
        '> **指标说明**：表格严格仅展示 **Top-3 与 Top-5**，并列给出**原图/修改图正常检索命中**与**净提升（修改 − 原图） / +纯新增 (都进榜, -退榜)**。',
        '',
        '## 一、黑盒均值对比（`OpenCLIP-bigG-14` + `SigLIP-SO400M`，合计 360 条查询/套）',
        '',
        hdr,
        sep,
    ] + _build_rows(lambda tag, s: summary['mean_blackbox'][tag][s])

    lines += ['', '## 二、单编码器分项结果（180 条查询/套）']
    for eslug, elabel in ENCODERS:
        lines += ['', f'### {elabel}（180 条查询）', '', hdr, sep]
        lines += _build_rows(lambda tag, s, e=eslug: summary['panels'][tag][s][e]['overall'])

    lines += ['', '## 三、9 组逐组对照明细（每组 20 条查询，格式：`T3: 命中(净提升) / T5: 命中(净提升)`）']
    gcols = ['组别', '编码器', '原图 (T3 / T5)'] + [f'{label} (T3 / T5)' for _, label, _ in active_specs]
    ghdr = '| ' + ' | '.join(gcols) + ' |'
    gsep = '|' + '---|' * len(gcols)
    for suite, slabel in SUITES:
        lines += ['', f'### {slabel}', '', ghdr, gsep]
        for g in GROUPS:
            for eslug, elabel in ENCODERS:
                base_g = summary['panels'][active_specs[0][0]][suite][eslug]['groups'][g]
                c_cell = f"{base_g['top3']['clean_hits']} / {base_g['top5']['clean_hits']}"
                row = [f'`{g}`', elabel, c_cell]
                for tag, _, _ in active_specs:
                    gm = summary['panels'][tag][suite][eslug]['groups'][g]
                    row.append(
                        f"{gm['top3']['modified_hits']} ({gm['top3']['net_gain']:+d}) / "
                        f"{gm['top5']['modified_hits']} ({gm['top5']['net_gain']:+d})"
                    )
                lines.append('| ' + ' | '.join(row) + ' |')

    lines += [
        '',
        '## 四、9 组成图自然度门控与配置验证',
        '',
        '| 组别 | Passed | Source SSIM | Source MAE(255) | Foreground MAE(255) | Foreground Chroma TV | EoT Paths |',
        '|---|---|---:|---:|---:|---:|---|',
    ]
    for g in GROUPS:
        nt = nat_info[g]
        lines.append(
            f"| `{g}` | `{nt['passed']}` | {nt['source_ssim']:.4f} | {nt['source_mae_levels']:.2f} | "
            f"{nt['fg_rgb_source_mae_levels']:.2f} | {nt['fg_chroma_tv_levels']:.4f} | `{' + '.join(nt['eot_paths'])}` |"
        )

    md_text = '\n'.join(lines) + '\n'
    assert 'Top-1' not in md_text and '平均名次' not in md_text and '均名' not in md_text
    out_md = RESULTS_DIR / 'SUMMARY_9G_MS_V2_GEOM.md'
    out_md.write_text(md_text, encoding='utf-8')
    print(f'[WRITTEN] {out_md}')
    print(f'[WRITTEN] {out_json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
