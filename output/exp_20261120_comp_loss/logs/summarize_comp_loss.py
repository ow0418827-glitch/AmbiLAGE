"""汇总 Stage 1.2 Visual Inception comp_loss 权重对比结果并按黑盒最强选出优胜配置。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/content/AmbiLAGE/output/exp_20261120_comp_loss')
VARIANTS = (
    ('blend_03_07', '0.3 / 0.7 (Current enhanced_v3)', 0.3, 0.7),
    ('blend_08_02', '0.8 / 0.2 (Low comp_loss)', 0.8, 0.2),
    ('blend_10_00', '1.0 / 0.0 (Pure bank_loss)', 1.0, 0.0),
)
SUITES = ('training_oracle', 'zero_overlap')
ENCODERS = ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m')


def main() -> None:
    records = {}
    ranking_rows = []
    lines = [
        '# Stage 1.2 Visual Inception `comp_loss` 权重消融汇总（3 组：`04_curtain, 07_shampoo, 09_toothbrush`，60 条查询）',
        '',
        '| 配置 (`bank/comp`) | 测试环境 (`v1`) | 编码器 | 角色 | 原图 Top-3 / Top-5 | 修改后 Top-3 | 修改后 Top-5 | 均名(原->改) |',
        '|---|---|---|---|---:|---:|---:|---:|',
    ]
    for slug, label, b_w, c_w in VARIANTS:
        wb_t5 = []
        bb_t5 = []
        bb_t3 = []
        bb_mr = []
        for suite in SUITES:
            panel_path = ROOT / 'eval' / f'{slug}_{suite}' / 'transfer_panel_summary.json'
            panel = json.loads(panel_path.read_text(encoding='utf-8'))
            for enc in ENCODERS:
                e = panel['encoders'][enc]
                ov = e['overall']
                c3 = ov['top3']['clean_rate'] * 100
                c5 = ov['top5']['clean_rate'] * 100
                m3 = ov['top3']['modified_rate'] * 100
                m5 = ov['top5']['modified_rate'] * 100
                mrc = ov['mean_rank_clean']
                mrm = ov['mean_rank_modified']
                lines.append(
                    f"| `{b_w}/{c_w}` | `{suite}` | {e['label']} | {e['role_zh']} | "
                    f"{c3:.1f}% / {c5:.1f}% | {m3:.1f}% ({ov['top3']['modified_hits']}/60) | "
                    f"{m5:.1f}% ({ov['top5']['modified_hits']}/60) | {mrc:.2f}->{mrm:.2f} |"
                )
                records[f'{slug}__{suite}__{enc}'] = ov
                if enc == '01_clip_l14':
                    wb_t5.append(ov['top5']['modified_rate'])
                else:
                    bb_t5.append(ov['top5']['modified_rate'])
                    bb_t3.append(ov['top3']['modified_rate'])
                    bb_mr.append(ov['mean_rank_modified'])
        min_wb_t5 = min(wb_t5)
        mean_bb_t5 = sum(bb_t5) / len(bb_t5)
        mean_bb_t3 = sum(bb_t3) / len(bb_t3)
        mean_bb_mr = sum(bb_mr) / len(bb_mr)
        ranking_rows.append(dict(
            slug=slug,
            label=label,
            bank_loss_blend=b_w,
            hard_margin_loss_blend=c_w,
            min_whitebox_top5=min_wb_t5,
            mean_blackbox_top5=mean_bb_t5,
            mean_blackbox_top3=mean_bb_t3,
            mean_blackbox_mean_rank=mean_bb_mr,
            passes_whitebox_guard=(min_wb_t5 >= 0.90),
        ))

    eligible = [r for r in ranking_rows if r['passes_whitebox_guard']] or ranking_rows
    winner = max(
        eligible,
        key=lambda r: (r['mean_blackbox_top5'], r['mean_blackbox_top3'], -r['mean_blackbox_mean_rank']),
    )

    lines += [
        '',
        '## 黑盒综合评分对比（跨 `OpenCLIP-bigG-14` & `SigLIP-SO400M` × `training_oracle` & `zero_overlap` 均值）',
        '',
        '| 配置 (`bank/comp`) | 最低白盒 Top-5 | 黑盒均值 Top-5 | 黑盒均值 Top-3 | 黑盒均值均名 | 满足白盒 >=90% |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for r in ranking_rows:
        lines.append(
            f"| `{r['bank_loss_blend']}/{r['hard_margin_loss_blend']}` | "
            f"{r['min_whitebox_top5']*100:.1f}% | **{r['mean_blackbox_top5']*100:.2f}%** | "
            f"{r['mean_blackbox_top3']*100:.2f}% | {r['mean_blackbox_mean_rank']:.2f} | "
            f"{'YES' if r['passes_whitebox_guard'] else 'NO'} |"
        )
    lines += [
        '',
        f"**Selected Winner**: `{winner['bank_loss_blend']} / {winner['hard_margin_loss_blend']}` "
        f"(Blackbox Mean Top-5 = {winner['mean_blackbox_top5']*100:.2f}%, "
        f"Blackbox Mean Top-3 = {winner['mean_blackbox_top3']*100:.2f}%, "
        f"Min Whitebox Top-5 = {winner['min_whitebox_top5']*100:.1f}%).",
        '',
    ]
    (ROOT / 'COMP_LOSS_SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    (ROOT / 'COMP_LOSS_SUMMARY.json').write_text(
        json.dumps(dict(ranking=ranking_rows, winner=winner, records=records), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print('WROTE:', ROOT / 'COMP_LOSS_SUMMARY.md')
    print('WINNER:', json.dumps(winner, ensure_ascii=False))


if __name__ == '__main__':
    main()
