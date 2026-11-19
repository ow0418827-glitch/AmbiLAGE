"""汇总 Stage 2（JPEG / Blur / Resize / SD1.5 / 留出 SD2.1 净化防御）三环境 × 两方法评测结果。

包含：
1. 全部三套查询环境（training_oracle / overlap_30 / zero_overlap）× 两方法在白盒 CLIP-ViT-L/14（01_clip_l14）上的完整 17 组防御对比。
2. 已完整跑完的首组 ambilage_training_oracle 全三编码器（CLIP-ViT-L/14、OpenCLIP-bigG-14、SigLIP-SO400M）对照留存结果。
（其余 5 组按用户指示在白盒 01_clip_l14 完成并独立复算通过后收尾，跳过黑盒模型评测。）
"""
from __future__ import annotations

import json
from src.experiments.defenses import all_table_groups
from src.paths import OUTPUT
from src.visual.audit import read_json

STAGE2_DIR = OUTPUT / 'eval_9groups_20261119' / 'stage2_defenses'
SUITES = ('training_oracle', 'overlap_30', 'zero_overlap')
METHODS = ('ambilage', 'visual_inception')
METHOD_PREFIX = {'ambilage': 'ambilage', 'visual_inception': 'vi'}


def _fmt_cell(st: dict) -> str:
    t1, t3, t5 = st['top1'], st['top3'], st['top5']
    return (
        f"{t1['modified_rate']*100:.1f}% / {t3['modified_rate']*100:.1f}% / "
        f"{t5['modified_rate']*100:.1f}% ({st['mean_rank_clean']:.2f}->{st['mean_rank_modified']:.2f})"
    )


def main():
    records = {}
    table_titles = [title for title, _ in all_table_groups()]

    lines = [
        '# Stage 2 净化防御三环境对比汇总（9 组同时替换，纯图片检索键 `key_mode=image`）',
        '',
        '> 口径说明：`ambilage_training_oracle` 已完整运行全部三编码器（白盒 `CLIP-ViT-L/14` + 黑盒 `OpenCLIP-bigG-14` / `SigLIP-SO400M`）；其余 5 个面板按指示在白盒 `CLIP-ViT-L/14` 全部完成并通过独立进程全候选复算（最大误差 `0.00e+00`）后即收尾，不再继续测试黑盒模型。',
        '',
        '## 一、白盒 `CLIP-ViT-L/14` 全三套查询环境 × 全 17 类防御对比（Top-1 / Top-3 / Top-5 / 均名）',
        '',
    ]

    for suite in SUITES:
        lines.append(f'### 查询环境：`{suite}`')
        lines.append('')
        lines.append('| 防御条件 | `ambilage` (前1/前3/前5/均名) | `visual_inception` (前1/前3/前5/均名) |')
        lines.append('|---|---|---|')

        docs = {}
        for method in METHODS:
            run_name = f'{METHOD_PREFIX[method]}_{suite}'
            path = STAGE2_DIR / run_name / 'transfer_panel_summary.json'
            doc = read_json(path)
            docs[method] = doc
            enc_map = {}
            for slug in doc['encoder_order']:
                item = doc['encoders'][slug]
                ver = item['verification']
                assert ver.get('status') == 'passed', f'{run_name}/{slug} verification failed: {ver}'
                assert float(ver.get('max_score_error', 1.0)) <= 1e-7, f'{run_name}/{slug} score error too high'
                enc_map[slug] = dict(
                    label=item['label'],
                    role=item['role_zh'],
                    overall=item['overall'],
                    by_table_group=item['by_table_group'],
                    verification=ver,
                )
            assert '01_clip_l14' in enc_map, f'{run_name} missing 01_clip_l14'
            records[f'{method}__{suite}'] = dict(
                source_summary=str(path),
                query_suite=doc['query_suite'],
                protocol=doc['protocol'],
                key_mode=doc['key_mode'],
                defense_mode=doc['defense_mode'],
                encoder_order=doc['encoder_order'],
                blackbox_skipped_per_user_instruction=(len(doc['encoder_order']) == 1),
                encoders=enc_map,
            )

        for title in table_titles:
            amb_st = docs['ambilage']['encoders']['01_clip_l14']['by_table_group'][title]['overall']
            vi_st = docs['visual_inception']['encoders']['01_clip_l14']['by_table_group'][title]['overall']
            lines.append(f'| {title} | {_fmt_cell(amb_st)} | {_fmt_cell(vi_st)} |')
        lines.append('')

    # Preserve completed 3-encoder defense table for ambilage_training_oracle.
    amb_oracle = read_json(STAGE2_DIR / 'ambilage_training_oracle' / 'transfer_panel_summary.json')
    if len(amb_oracle['encoder_order']) == 3:
        lines.extend([
            '## 二、已完成首组 `ambilage_training_oracle` 三编码器全量留存汇总',
            '',
            '| 防御条件 | CLIP-ViT-L/14 (白盒) | OpenCLIP-bigG-14 (黑盒) | SigLIP-SO400M (黑盒) |',
            '|---|---|---|---|',
        ])
        for title in table_titles:
            cells = [
                _fmt_cell(amb_oracle['encoders'][slug]['by_table_group'][title]['overall'])
                for slug in amb_oracle['encoder_order']
            ]
            lines.append(f'| {title} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    (STAGE2_DIR / 'STAGE2_SUMMARY.json').write_text(
        json.dumps(dict(schema_version=1, stage='stage2_defenses', records=records), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (STAGE2_DIR / 'STAGE2_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('WROTE:', STAGE2_DIR / 'STAGE2_SUMMARY.md')


if __name__ == '__main__':
    main()
