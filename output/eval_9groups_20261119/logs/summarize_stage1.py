"""汇总 Stage 1（无防御 none / 纯图片检索键 image）两方法在三套查询环境上的三编码器检索结果。"""
from __future__ import annotations

import json
from pathlib import Path

from src.paths import OUTPUT
from src.visual.audit import read_json

STAGE1_DIR = OUTPUT / 'eval_9groups_20261119' / 'stage1_none'

SOURCES = {
    ('ambilage', 'training_oracle'): OUTPUT / 'ambilage_single_clip_l14_9groups_20261118' / 'transfer_panel_9groups' / 'transfer_panel_summary.json',
    ('visual_inception', 'training_oracle'): OUTPUT / 'vi_single_clip_l14_9groups_20261118' / 'transfer_panel_9groups' / 'transfer_panel_summary.json',
    ('ambilage', 'overlap_30'): STAGE1_DIR / 'ambilage_overlap_30' / 'transfer_panel_summary.json',
    ('visual_inception', 'overlap_30'): STAGE1_DIR / 'vi_overlap_30' / 'transfer_panel_summary.json',
    ('ambilage', 'zero_overlap'): STAGE1_DIR / 'ambilage_zero_overlap' / 'transfer_panel_summary.json',
    ('visual_inception', 'zero_overlap'): STAGE1_DIR / 'vi_zero_overlap' / 'transfer_panel_summary.json',
}


def main():
    records = {}
    lines = [
        '# Stage 1 无防御三环境对比汇总（9 组同时替换，纯图片检索键 `key_mode=image`）',
        '',
        '| 查询环境 | 方法 | 评估编码器 | 角色 | 原图 Top-1 | 修改后 Top-1 | Top-3 | Top-5 | 平均名次(原->改) | 独立复算状态 / 最大误差 |',
        '|---|---|---|---|---:|---:|---:|---:|---:|---|',
    ]
    for suite in ('training_oracle', 'overlap_30', 'zero_overlap'):
        for method in ('ambilage', 'visual_inception'):
            path = SOURCES[(method, suite)]
            doc = read_json(path)
            enc_map = {}
            for slug in doc['encoder_order']:
                item = doc['encoders'][slug]
                ov = item['overall']
                ver = item['verification']
                t1, t3, t5 = ov['top1'], ov['top3'], ov['top5']
                assert ver.get('status') == 'passed', f'{method}/{suite}/{slug} verification failed'
                enc_map[slug] = dict(
                    label=item['label'],
                    role=item['role_zh'],
                    top1_clean=t1['clean_rate'],
                    top1_modified=t1['modified_rate'],
                    top1_hits=f"{t1['modified_hits']}/{ov['query_count']}",
                    top3_modified=t3['modified_rate'],
                    top3_hits=f"{t3['modified_hits']}/{ov['query_count']}",
                    top5_modified=t5['modified_rate'],
                    top5_hits=f"{t5['modified_hits']}/{ov['query_count']}",
                    mean_rank_clean=ov['mean_rank_clean'],
                    mean_rank_modified=ov['mean_rank_modified'],
                    max_score_error=ver['max_score_error'],
                    verification_status=ver['status'],
                )
                lines.append(
                    f"| `{suite}` | `{method}` | {item['label']} | {item['role_zh']} | "
                    f"{t1['clean_rate']*100:.1f}% ({t1['clean_hits']}/{ov['query_count']}) | "
                    f"{t1['modified_rate']*100:.1f}% ({t1['modified_hits']}/{ov['query_count']}) | "
                    f"{t3['modified_rate']*100:.1f}% ({t3['modified_hits']}/{ov['query_count']}) | "
                    f"{t5['modified_rate']*100:.1f}% ({t5['modified_hits']}/{ov['query_count']}) | "
                    f"{ov['mean_rank_clean']:.2f}->{ov['mean_rank_modified']:.2f} | "
                    f"{ver['status']} ({ver['max_score_error']:.2e}) |"
                )
            records[f'{method}__{suite}'] = dict(
                source_summary=str(path),
                query_suite=doc['query_suite'],
                protocol=doc['protocol'],
                key_mode=doc['key_mode'],
                defense_mode=doc['defense_mode'],
                encoders=enc_map,
            )

    summary_doc = dict(schema_version=1, stage='stage1_none', records=records)
    (STAGE1_DIR / 'STAGE1_SUMMARY.json').write_text(
        json.dumps(summary_doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    (STAGE1_DIR / 'STAGE1_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('WROTE:', STAGE1_DIR / 'STAGE1_SUMMARY.md')


if __name__ == '__main__':
    main()
