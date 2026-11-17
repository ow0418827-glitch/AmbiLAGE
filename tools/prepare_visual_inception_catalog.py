"""Prepare an independently sampled public catalogue; do not load models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visual._visual_inception.public_catalog import (load_excluded_ids, prepare_public_catalog,
    prepare_public_description_item_group)


def main(argv=None):
    parser = argparse.ArgumentParser(description='按固定随机种子准备公开独立候选与真实商品描述，不下载图片、不加载模型')
    parser.add_argument('--catalog', required=True, type=Path)
    parser.add_argument('--base', type=Path, default=Path('resources/evaluation_base'))
    parser.add_argument('--groups-json', type=Path, help='可选公开商品组对象，格式与 public_group 返回值一致')
    parser.add_argument('--groups', nargs='+', help='默认使用基础包所有公开商品组')
    parser.add_argument('--excluded-ids', required=True, type=Path, help='仅含商品编号数组或 excluded_ids 的文件')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--description-count', type=int, default=50)
    parser.add_argument('--reserve-count', type=int, default=150)
    parser.add_argument('--max-read-per-group', type=int, default=2500)
    parser.add_argument('--max-bucket-ids', type=int, default=25000)
    parser.add_argument('--catalog-sha256', help='上游清单中已经核验的目录内容摘要；不在此重复散列大型目录')
    parser.add_argument('--public-description-items', action='store_true', help='仅曲奇面团组：从真实公开字段原文中选取五十条，每商品最多五条')
    parser.add_argument('--catalog-identity-manifest', type=Path, help='原文条目模式必须使用已有目录抽样根清单中的源身份')
    args = parser.parse_args(argv)
    if args.groups_json:
        groups = json.loads(args.groups_json.read_text(encoding='utf-8'))
        if not isinstance(groups, dict):
            raise ValueError('公开商品组必须为对象')
        if args.groups:
            groups = {name: groups[name] for name in args.groups}
    else:
        from src.experiments.base_workflow import public_group
        document = json.loads((args.base / 'manifest.json').read_text(encoding='utf-8'))
        names = args.groups or sorted(document['groups'])
        groups = {name: public_group(args.base, name) for name in names}
    if args.public_description_items:
        if len(groups) != 1 or not args.catalog_identity_manifest:
            raise ValueError('公开原文条目模式必须只提供曲奇面团组及冻结目录身份清单')
        name, group = next(iter(groups.items()))
        identity = json.loads(args.catalog_identity_manifest.read_text(encoding='utf-8'))['catalog_identity']
        audit = prepare_public_description_item_group(args.catalog, name, group, args.output,
            excluded_ids=load_excluded_ids(args.excluded_ids), catalog_identity=identity,
            seed=args.seed, count=args.count, description_count=args.description_count,
            max_bucket_ids=args.max_bucket_ids)
        print(json.dumps(dict(status='public_original_description_items_frozen', group=name,
            strict_eligible_product_count=audit['strict_eligible_product_count'],
            description_count=audit['description_count'], output=str(args.output.resolve())), ensure_ascii=False))
        return
    manifest = prepare_public_catalog(args.catalog, groups, args.output,
        excluded_ids=load_excluded_ids(args.excluded_ids), seed=args.seed,
        count=args.count, description_count=args.description_count,
        reserve_count=args.reserve_count, max_read_per_group=args.max_read_per_group,
        max_bucket_ids=args.max_bucket_ids, catalog_sha256=args.catalog_sha256)
    print(json.dumps(dict(status='public_catalog_sample_frozen_images_not_downloaded',
        manifest=str((args.output / 'manifest.json').resolve()), groups=len(manifest['groups']),
        candidate_count_per_group=args.count, description_count_per_group=args.description_count),
        ensure_ascii=False))


if __name__ == '__main__':
    main()
