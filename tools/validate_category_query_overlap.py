"""核验九组纯文字查询的整条训练文本重合率。"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.visual_inception_category_evaluation import (
    TRAINING_ORACLE_QUERY_ROOT,
    _load_training_snapshot_texts,
    normalize_query_text,
)
from src.paths import RESOURCES
from src.visual.audit import read_json
from src.visual.image_utils import sha256_file


PROTOCOL = 'category_query_overlap_30_v1'
MANIFEST = RESOURCES / 'visual_inception_evaluation' / f'{PROTOCOL}.json'
REUSED_BY_FAMILY = {'category_description': 3, 'model_paraphrase': 2, 'heldout_search_query': 1}
NEW_QUERY_COUNT = 14


def validate_query_overlap(manifest_path=MANIFEST):
    manifest = read_json(manifest_path)
    if (manifest.get('protocol') != PROTOCOL or manifest.get('language') != 'en'
            or manifest.get('new_query_source') != 'synthetic_model_authored'):
        raise ValueError('查询协议、语言或新问法来源标记错误。')

    oracle_path = TRAINING_ORACLE_QUERY_ROOT / 'manifest.json'
    if sha256_file(oracle_path) != manifest['training_oracle_manifest_sha256']:
        raise ValueError('训练抽样清单摘要不符。')
    oracle = read_json(oracle_path)
    audit = oracle['training_overlap_audit']
    training_texts = _load_training_snapshot_texts(audit['training_source_snapshots'])
    if not training_texts or len(training_texts) != audit['unique_training_text_count']:
        raise ValueError('训练快照的去重文本数不符。')

    groups = manifest['groups']
    if len(groups) != 9 or groups.keys() != oracle['groups'].keys():
        raise ValueError('查询清单应与基准清单覆盖相同的九个品类。')

    results = {}
    all_new_texts = set()
    for group, reference in oracle['groups'].items():
        group_doc = groups[group]
        if group_doc['category'] != reference['category']:
            raise ValueError(f'{group} 查询品类与基准不一致。')

        reference_path = TRAINING_ORACLE_QUERY_ROOT / reference['path']
        if sha256_file(reference_path) != reference['sha256']:
            raise ValueError(f'{group} 训练抽样查询文件摘要不符。')
        records = read_json(reference_path)['query_records']
        reused_texts = []
        for family, count in REUSED_BY_FAMILY.items():
            matches = [row['instruction'] for row in records if row['query_family'] == family]
            if len(matches) < count:
                raise ValueError(f'{group} 可复用原文不足，来源为 {family}。')
            reused_texts.extend(normalize_query_text(text) for text in matches[:count])
        if len(set(reused_texts)) != len(reused_texts) or not set(reused_texts) <= training_texts:
            raise ValueError(f'{group} 复用原文有重复或未命中训练快照。')

        new_queries = group_doc['new_queries']
        if not isinstance(new_queries, list) or len(new_queries) != NEW_QUERY_COUNT:
            raise ValueError(f'{group} 必须提供 {NEW_QUERY_COUNT} 条新问法。')
        new_texts = {normalize_query_text(text) for text in new_queries}
        if len(new_texts) != NEW_QUERY_COUNT or new_texts & all_new_texts:
            raise ValueError(f'{group} 存在规范化后重复的新问法。')
        if new_texts & training_texts:
            raise ValueError(f'{group} 新问法与训练快照存在整条精确重合。')
        all_new_texts.update(new_texts)

        overlap = len(reused_texts)
        total = overlap + len(new_texts)
        results[group] = dict(category=reference['category'], total=total, overlap=overlap,
                              non_overlap=len(new_texts), overlap_ratio=overlap / total)

    total = sum(group['total'] for group in results.values())
    overlap = sum(group['overlap'] for group in results.values())
    return dict(status='passed', protocol=PROTOCOL, manifest_sha256=sha256_file(manifest_path),
                unique_training_text_count=len(training_texts), total=total, overlap=overlap,
                non_overlap=total - overlap, overlap_ratio=overlap / total, groups=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=MANIFEST, help='待核验的查询清单')
    args = parser.parse_args(argv)
    print(json.dumps(validate_query_overlap(args.manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
