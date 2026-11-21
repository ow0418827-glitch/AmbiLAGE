#!/usr/bin/env python3
"""Set 2 四组受控 EoT 消融实验汇总 (`01_camera`, `02_chair`, `06_jeans`, `07_shampoo`)。

支持自动检测并纳入 `ambilage_multiscale_only_set2_4g`（零防御算子、仅多尺度破网格）对照结果。
展示层严格遵循用户要求：
- 仅对比 Top-3 与 Top-5 命中结果（不展示 Top-1，也不展示平均名次）；
- 正常检索命中/命中率与净提升（修改 - 原图）、纯新增进榜（扣除都进榜后）并列展示；
- 底层 EOT_ABLATION_4G_SUMMARY.json 完整保留 Top-1、平均名次、中位名次、MRR 与余弦得分变化供分析。
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

EXP_DIR = Path('/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band')
EVAL_ROOT = EXP_DIR / 'eval_eot_ablation_4g'

GROUPS = ('01_camera', '02_chair', '06_jeans', '07_shampoo')
BASE_SPECS = (
    ('ambilage_eot_on_set2_4g', 'EoT=ON (含防御EoT)'),
    ('ambilage_eot_off_set2_4g', 'EoT=OFF (无任何EoT)'),
    ('vi_10_00_set2_4g', '基线 VI (1.0/0.0)'),
)
MULTISCALE_SPEC = ('ambilage_multiscale_only_set2_4g', '仅多尺度破网格 (零防御EoT)')
SUITES = (
    ('zero_overlap_v4', '0% 重合 (`zero_overlap_v4`)'),
    ('overlap_30_v4', '30% 重合 (`overlap_30_v4`)'),
    ('training_oracle_v4', '100% 重合 (`training_oracle_v4`)'),
)
ENCODERS = (
    ('02_openclip_bigG14', 'OpenCLIP-bigG-14 (黑盒)'),
    ('03_siglip_so400m', 'SigLIP-SO400M (黑盒)'),
    ('01_clip_l14', 'CLIP-ViT-L/14 (白盒)'),
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


def _multiscale_ready() -> bool:
    tag = MULTISCALE_SPEC[0]
    for suite, _ in SUITES:
        for eslug, _ in ENCODERS:
            for g in GROUPS:
                if not (EVAL_ROOT / f'{tag}_{suite}' / eslug / 'groups' / g / 'evaluation.json').is_file():
                    return False
    return True


def load_panel_pairs(spec_tag: str, suite: str, eslug: str) -> tuple[dict[str, dict], dict]:
    by_group: dict[str, dict] = {}
    all_pairs: list[tuple[int, int, float, float]] = []
    panel_dir = EVAL_ROOT / f'{spec_tag}_{suite}' / eslug
    for g in GROUPS:
        ev = json.loads((panel_dir / 'groups' / g / 'evaluation.json').read_text(encoding='utf-8'))
        cq = next(x for x in ev if x['bank'] == 'clean')['queries']
        mq = next(x for x in ev if x['bank'] == 'modified')['queries']
        assert len(cq) == len(mq) == 20, f'{spec_tag}/{suite}/{eslug}/{g} query count {len(cq)}/{len(mq)}'
        pairs: list[tuple[int, int, float, float]] = []
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


def fmt_delta(a_m: dict, b_m: dict, kkey: str) -> str:
    n = a_m['query_count']
    diff = a_m[kkey]['modified_hits'] - b_m[kkey]['modified_hits']
    return f"{diff:+d} 条 ({signed_pp(diff / n)})"


def main() -> int:
    has_ms = _multiscale_ready()
    active_specs = (
        ('ambilage_eot_on_set2_4g', 'EoT=ON (含防御EoT)'),
        MULTISCALE_SPEC,
        ('ambilage_eot_off_set2_4g', 'EoT=OFF (无任何EoT)'),
        ('vi_10_00_set2_4g', '基线 VI (1.0/0.0)'),
    ) if has_ms else BASE_SPECS

    summary: dict = {
        'groups': list(GROUPS),
        'specs': {tag: label for tag, label in active_specs},
        'has_multiscale_only': has_ms,
        'suites': [s for s, _ in SUITES],
        'encoders': [e for e, _ in ENCODERS],
        'panels': {},
        'mean_blackbox': {},
    }

    for spec_tag, _ in active_specs:
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

    out_json = EXP_DIR / 'EOT_ABLATION_4G_SUMMARY.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    ms_tag = MULTISCALE_SPEC[0]
    lines: list[str] = [
        '# Set 2 四组受控 EoT 消融实验报告（`01_camera` / `02_chair` / `06_jeans` / `07_shampoo`）',
        '',
        '> **对照说明**：所有参评方法均使用完全相同的 4 张 Set 2 原图（`B01LML1VXU`、`B07GDZQ68V`、`B09J89KXRJ`、`B08WJB1XDD`）及 `--replacement-groups selected` 同库替换口径同场评测。',
        '> **指标说明**：按要求仅看 **Top-3 与 Top-5**；列中分别给出**正常检索命中**与**净提升（修改 − 原图） / +纯新增 (都进榜, -退榜)**。',
        '',
        '## 一、黑盒均值（`OpenCLIP-bigG-14` + `SigLIP-SO400M`，合计 160 条查询）',
        '',
    ]

    if has_ms:
        hdr = '| 测试集 | 指标 | 原图正常命中 | EoT=ON 修改命中 | EoT=ON 净提升 / 纯新增 | **仅多尺度破网格 修改命中** | **仅多尺度 净提升 / 纯新增** | **多尺度 − OFF** | EoT=OFF 修改命中 | EoT=OFF 净提升 / 纯新增 | **ON − OFF** | VI 修改命中 | VI 净提升 / 纯新增 |'
        sep = '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
    else:
        hdr = '| 测试集 | 指标 | 原图正常命中 | EoT=ON 修改命中 | EoT=ON 净提升 / 纯新增(都进榜,-退榜) | EoT=OFF 修改命中 | EoT=OFF 净提升 / 纯新增(都进榜,-退榜) | **EoT=ON − OFF** | VI (1.0/0.0) 修改命中 | VI 净提升 / 纯新增(都进榜,-退榜) |'
        sep = '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'

    lines += [hdr, sep]
    for suite, slabel in SUITES:
        on_m = summary['mean_blackbox']['ambilage_eot_on_set2_4g'][suite]
        off_m = summary['mean_blackbox']['ambilage_eot_off_set2_4g'][suite]
        vi_m = summary['mean_blackbox']['vi_10_00_set2_4g'][suite]
        ms_m = summary['mean_blackbox'][ms_tag][suite] if has_ms else None
        for kkey, klabel in (('top3', 'Top-3'), ('top5', 'Top-5')):
            if has_ms and ms_m is not None:
                lines.append(
                    f"| {slabel} | **{klabel}** | {fmt_clean(on_m, kkey)} | "
                    f"**{fmt_hits(on_m, kkey)}** | {fmt_gain(on_m, kkey)} | "
                    f"**{fmt_hits(ms_m, kkey)}** | {fmt_gain(ms_m, kkey)} | **{fmt_delta(ms_m, off_m, kkey)}** | "
                    f"{fmt_hits(off_m, kkey)} | {fmt_gain(off_m, kkey)} | **{fmt_delta(on_m, off_m, kkey)}** | "
                    f"{fmt_hits(vi_m, kkey)} | {fmt_gain(vi_m, kkey)} |"
                )
            else:
                lines.append(
                    f"| {slabel} | **{klabel}** | {fmt_clean(on_m, kkey)} | "
                    f"**{fmt_hits(on_m, kkey)}** | {fmt_gain(on_m, kkey)} | "
                    f"{fmt_hits(off_m, kkey)} | {fmt_gain(off_m, kkey)} | "
                    f"**{fmt_delta(on_m, off_m, kkey)}** | "
                    f"{fmt_hits(vi_m, kkey)} | {fmt_gain(vi_m, kkey)} |"
                )

    lines += [
        '',
        '## 二、单编码器分项结果（每套 80 条查询）',
    ]
    for eslug, elabel in ENCODERS:
        lines += ['', f'### {elabel}（80 条查询）', '', hdr, sep]
        for suite, slabel in SUITES:
            on_m = summary['panels']['ambilage_eot_on_set2_4g'][suite][eslug]['overall']
            off_m = summary['panels']['ambilage_eot_off_set2_4g'][suite][eslug]['overall']
            vi_m = summary['panels']['vi_10_00_set2_4g'][suite][eslug]['overall']
            ms_m = summary['panels'][ms_tag][suite][eslug]['overall'] if has_ms else None
            for kkey, klabel in (('top3', 'Top-3'), ('top5', 'Top-5')):
                if has_ms and ms_m is not None:
                    lines.append(
                        f"| {slabel} | **{klabel}** | {fmt_clean(on_m, kkey)} | "
                        f"**{fmt_hits(on_m, kkey)}** | {fmt_gain(on_m, kkey)} | "
                        f"**{fmt_hits(ms_m, kkey)}** | {fmt_gain(ms_m, kkey)} | **{fmt_delta(ms_m, off_m, kkey)}** | "
                        f"{fmt_hits(off_m, kkey)} | {fmt_gain(off_m, kkey)} | **{fmt_delta(on_m, off_m, kkey)}** | "
                        f"{fmt_hits(vi_m, kkey)} | {fmt_gain(vi_m, kkey)} |"
                    )
                else:
                    lines.append(
                        f"| {slabel} | **{klabel}** | {fmt_clean(on_m, kkey)} | "
                        f"**{fmt_hits(on_m, kkey)}** | {fmt_gain(on_m, kkey)} | "
                        f"{fmt_hits(off_m, kkey)} | {fmt_gain(off_m, kkey)} | "
                        f"**{fmt_delta(on_m, off_m, kkey)}** | "
                        f"{fmt_hits(vi_m, kkey)} | {fmt_gain(vi_m, kkey)} |"
                    )

    lines += [
        '',
        '## 三、四组逐组对照明细（每组 20 条查询，格式：`T3: 命中(净提升) / T5: 命中(净提升)`）',
    ]
    for suite, slabel in SUITES:
        if has_ms:
            ghdr = '| 组别 | 编码器 | 原图 (T3 / T5) | EoT=ON (T3 / T5) | **仅多尺度破网格 (T3 / T5)** | EoT=OFF (T3 / T5) | **ON−OFF (T3 / T5)** | **多尺度−OFF (T3 / T5)** | 基线 VI (T3 / T5) |'
            gsep = '|---|---|---|---|---|---|---|---|---|'
        else:
            ghdr = '| 组别 | 编码器 | 原图 (T3 / T5) | EoT=ON (T3 / T5) | EoT=OFF (T3 / T5) | **EoT=ON − OFF (T3 / T5)** | 基线 VI (T3 / T5) |'
            gsep = '|---|---|---|---|---|---|---|'
        lines += ['', f'### {slabel}', '', ghdr, gsep]
        for g in GROUPS:
            for eslug, elabel in ENCODERS:
                on_g = summary['panels']['ambilage_eot_on_set2_4g'][suite][eslug]['groups'][g]
                off_g = summary['panels']['ambilage_eot_off_set2_4g'][suite][eslug]['groups'][g]
                vi_g = summary['panels']['vi_10_00_set2_4g'][suite][eslug]['groups'][g]
                c_cell = f"{on_g['top3']['clean_hits']} / {on_g['top5']['clean_hits']}"
                on_cell = f"{on_g['top3']['modified_hits']} ({on_g['top3']['net_gain']:+d}) / {on_g['top5']['modified_hits']} ({on_g['top5']['net_gain']:+d})"
                off_cell = f"{off_g['top3']['modified_hits']} ({off_g['top3']['net_gain']:+d}) / {off_g['top5']['modified_hits']} ({off_g['top5']['net_gain']:+d})"
                d3 = on_g['top3']['modified_hits'] - off_g['top3']['modified_hits']
                d5 = on_g['top5']['modified_hits'] - off_g['top5']['modified_hits']
                d_cell = f"**{d3:+d} / {d5:+d}**"
                vi_cell = f"{vi_g['top3']['modified_hits']} ({vi_g['top3']['net_gain']:+d}) / {vi_g['top5']['modified_hits']} ({vi_g['top5']['net_gain']:+d})"
                if has_ms:
                    ms_g = summary['panels'][ms_tag][suite][eslug]['groups'][g]
                    ms_cell = f"{ms_g['top3']['modified_hits']} ({ms_g['top3']['net_gain']:+d}) / {ms_g['top5']['modified_hits']} ({ms_g['top5']['net_gain']:+d})"
                    m3 = ms_g['top3']['modified_hits'] - off_g['top3']['modified_hits']
                    m5 = ms_g['top5']['modified_hits'] - off_g['top5']['modified_hits']
                    m_cell = f"**{m3:+d} / {m5:+d}**"
                    lines.append(f"| `{g}` | {elabel} | {c_cell} | {on_cell} | {ms_cell} | {off_cell} | {d_cell} | {m_cell} | {vi_cell} |")
                else:
                    lines.append(f"| `{g}` | {elabel} | {c_cell} | {on_cell} | {off_cell} | {d_cell} | {vi_cell} |")

    md_text = '\n'.join(lines) + '\n'
    assert 'Top-1' not in md_text and '平均名次' not in md_text and '均名' not in md_text
    out_md = EXP_DIR / 'EOT_ABLATION_4G_SUMMARY.md'
    out_md.write_text(md_text, encoding='utf-8')
    print(f'[SUMMARY WRITTEN] {out_md} (has_multiscale_only={has_ms})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
