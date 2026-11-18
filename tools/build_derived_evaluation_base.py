"""根据公开中位数选图发布产物构建独立的 --base 实验基础目录；不改动原始九组基础包。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import EVALUATION_BASE, OUTPUT, PROJECT, ProjectPath
from src.visual.audit import digest, read_json
from src.visual.image_utils import sha256_file
from src.visual.pair_assets import load_pair
from src.visual.pair_selection import GOLDEN_BAND_MEDIAN_POLICY, MEDIAN_SELECTION_POLICY, verify_selection
from src.visual.target_features import extract_product_features


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _copy_checked(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    digest_val = sha256_file(dst)
    if sha256_file(src) != digest_val:
        raise ValueError(f'复制文件摘要不符: {src} -> {dst}')
    return digest_val


def _public_catalog_record(asin: str, catalog_sqlite: Path) -> dict:
    with sqlite3.connect(f'file:{catalog_sqlite.resolve().as_posix()}?mode=ro', uri=True) as conn:
        row = conn.execute('SELECT data, category, query FROM products WHERE asin=?', (asin,)).fetchone()
    if row is None:
        raise ValueError(f'公开商品库缺少源商品记录: {asin}')
    raw = json.loads(row[0])
    title = str(raw.get('Title') or raw.get('name') or '').strip()
    if not title:
        raise ValueError(f'公开商品缺少标题: {asin}')
    record = {
        'asin': asin,
        'Title': title,
        'name': str(raw.get('name') or title),
        'Attributes': list(raw.get('Attributes') or []),
        'BulletPoints': list(raw.get('BulletPoints') or raw.get('small_description') or []),
        'Description': str(raw.get('Description') or raw.get('full_description') or ''),
        'product_category': str(raw.get('product_category') or row[1] or ''),
        'category': str(raw.get('category') or row[1] or ''),
        'query': str(raw.get('query') or row[2] or ''),
        'MainImage': raw.get('MainImage') or (raw.get('images') or [''])[0],
        'images': list(raw.get('images') or ([raw['MainImage']] if raw.get('MainImage') else [])),
    }
    if raw.get('Brand') or raw.get('brand'):
        record['Brand'] = str(raw.get('Brand') or raw.get('brand'))
    return record


def build_derived_base(
    selection_root: Path,
    output_base: Path,
    *,
    source_base: Path = EVALUATION_BASE,
    catalog_sqlite: Path | None = None,
) -> dict:
    selection_root = Path(selection_root).resolve()
    source_base = Path(source_base).resolve()
    output_base = Path(output_base).resolve()
    if output_base == EVALUATION_BASE.resolve() or output_base == source_base:
        raise ValueError('禁止覆盖原始九组基础包；请指定新的输出基础目录')
    if output_base.exists() and any(output_base.iterdir()):
        raise FileExistsError('目标基础目录已存在且非空，请指定新目录或先清理：' + str(output_base))
    if catalog_sqlite is None:
        from src.webshop.config import CATALOG
        catalog_sqlite = (CATALOG / 'catalog.sqlite').resolve()
    else:
        catalog_sqlite = Path(catalog_sqlite).resolve()

    src_manifest = read_json(source_base / 'manifest.json')
    src_pool = read_json(source_base / src_manifest['shared_memory']['pool_file'])
    named_pairs = read_json(selection_root / 'NAMED_PAIRS.json')
    if named_pairs.get('status') != 'published':
        raise ValueError('公开选图结果尚未完成标注审核与发布（NAMED_PAIRS.json）')
    bundle = read_json(selection_root / 'PAIR_SELECTION_BUNDLE.json')
    reviews = read_json(selection_root / 'data/selected_visual_reviews.json')

    by_group = {g['key']: g for g in named_pairs['groups']}
    if list(src_manifest['groups']) != [g['key'] for g in named_pairs['groups']] and set(by_group) != set(src_manifest['groups']):
        raise ValueError('公开选图组键与九组基础包不一致')

    # 复制静态评估资产（正式纯文字/图文询问清单、查询图片与评估编码器配置）
    output_base.mkdir(parents=True, exist_ok=True)
    for subdir in ('evaluation_inputs', 'evaluation_inputs_joint', 'evaluation_inputs_vague_joint', 'query_images'):
        src_dir = source_base / subdir
        if src_dir.is_dir():
            shutil.copytree(src_dir, output_base / subdir)
    shutil.copy2(source_base / 'evaluation_encoder.json', output_base / 'evaluation_encoder.json')

    # 读取并扩展公开元数据（仅补充新选出的 9 个 source 商品公开记录）
    meta_doc = copy.deepcopy(read_json(source_base / src_manifest['public_metadata']['path']))
    review_doc = copy.deepcopy(read_json(source_base / src_manifest['public_metadata']['review_path']))

    normal_entries = [copy.deepcopy(e) for e in src_pool['entries'] if e.get('EntryType') == 'normal']
    if len(normal_entries) != 180:
        raise ValueError(f'期望保留 180 条正常背景记忆，实际为 {len(normal_entries)}')

    images_dir = output_base / 'shared_memory/images'
    images_dir.mkdir(parents=True, exist_ok=True)
    copied_images = set()
    for entry in normal_entries:
        rel = entry['PrimaryImage']['LocalPath']
        src_img = (source_base / 'shared_memory' / rel).resolve()
        dst_img = (output_base / 'shared_memory' / rel).resolve()
        if str(dst_img) not in copied_images:
            _copy_checked(src_img, dst_img)
            copied_images.add(str(dst_img))

    # 先收集 9 组公开图对与新源商品元数据
    new_ref_entries = []
    source_mapping = {}
    new_groups_manifest = {}

    for group_key, src_group_spec in src_manifest['groups'].items():
        sel_path = selection_root / 'groups' / group_key / 'selection/SELECTED_PAIR.json'
        selection = read_json(sel_path)
        verify_selection(selection)
        pair_dir = selection_root / 'data/groups' / group_key
        loaded_pair = load_pair(pair_dir / 'pair.json', selection_root / 'data', verify_models=False)
        src_def = read_json(source_base / src_group_spec['definition'])
        src_feat_doc = read_json(source_base / src_group_spec['features'])

        target_info = loaded_pair['target']
        source_info = loaded_pair['source']
        if target_info['id'] != src_group_spec['target_id']:
            raise ValueError(f'组 {group_key} 目标编号改变: {target_info["id"]} != {src_group_spec["target_id"]}')

        source_id = source_info['id']
        if source_id not in meta_doc['products']:
            meta_doc['products'][source_id] = _public_catalog_record(source_id, catalog_sqlite)

    meta_path = output_base / 'public_metadata/product_metadata.json'
    review_path = output_base / 'public_metadata/product_feature_review.json'
    _write_json(meta_path, meta_doc)
    _write_json(review_path, review_doc)

    # 提取新源商品特点并构建分组目录 + 新共享参考条目
    for group_key, src_group_spec in src_manifest['groups'].items():
        sel_path = selection_root / 'groups' / group_key / 'selection/SELECTED_PAIR.json'
        selection = read_json(sel_path)
        pair_dir = selection_root / 'data/groups' / group_key
        loaded_pair = load_pair(pair_dir / 'pair.json', selection_root / 'data', verify_models=False)
        src_def = read_json(source_base / src_group_spec['definition'])
        src_feat_doc = read_json(source_base / src_group_spec['features'])

        target_info = loaded_pair['target']
        source_info = loaded_pair['source']
        source_id = source_info['id']

        target_feat = extract_product_features(target_info['id'], db_path=meta_path, review_path=review_path)
        source_feat = extract_product_features(source_id, db_path=meta_path, review_path=review_path)
        if target_feat['canonical_sentence'] != src_group_spec['target_feature_text']:
            raise ValueError(f'组 {group_key} 目标训练正文不一致')

        out_group_dir = output_base / 'groups' / group_key
        out_group_dir.mkdir(parents=True, exist_ok=True)
        src_img_path = Path(source_info['image'])
        tgt_img_path = Path(target_info['image'])
        src_ext = src_img_path.suffix.lower() or '.jpg'
        tgt_ext = tgt_img_path.suffix.lower() or '.jpg'
        src_filename = 'source' + src_ext
        tgt_filename = 'target' + tgt_ext

        src_sha = _copy_checked(src_img_path, out_group_dir / src_filename)
        tgt_sha = _copy_checked(tgt_img_path, out_group_dir / tgt_filename)
        if src_sha != source_info['image_sha256'] or tgt_sha != target_info['image_sha256']:
            raise ValueError(f'组 {group_key} 源/目标图像摘要不符')

        # 前景与核对叠图
        _copy_checked(Path(source_info['foreground_mask']), out_group_dir / 'source_foreground.png')
        _copy_checked(Path(target_info['foreground_mask']), out_group_dir / 'target_foreground.png')
        ann_dir = selection_root / 'data/selected_annotations/annotations'
        if (ann_dir / source_id / 'review_overlay.png').is_file():
            _copy_checked(ann_dir / source_id / 'review_overlay.png', out_group_dir / 'review_overlay_source.png')
        if (ann_dir / target_info['id'] / 'review_overlay.png').is_file():
            _copy_checked(ann_dir / target_info['id'] / 'review_overlay.png', out_group_dir / 'review_overlay_target.png')
        review_sheet = selection_root / 'public_review' / f'{group_key}_01.jpg'
        if review_sheet.is_file():
            _copy_checked(review_sheet, out_group_dir / 'contact_sheet.jpg')
        _copy_checked(sel_path, out_group_dir / 'selection_evidence.json')
        _write_json(out_group_dir / 'selected_visual_reviews.json', reviews)

        # 共享池图片与参考条目
        pool_img_rel = f'images/{src_sha}{src_ext}'
        _copy_checked(out_group_dir / src_filename, output_base / 'shared_memory' / pool_img_rel)

        mem_id = f'groups9_reference_{group_key}_{source_id}'
        source_mapping[group_key] = {'source_asin': source_id, 'source_memory_id': mem_id}
        new_ref_entries.append({
            'Id': mem_id,
            'SourceGroup': group_key,
            'EntryType': 'error',
            'RecordKind': 'simulated_gallery',
            'Completed': False,
            'Success': False,
            'Actions': [],
            'Instruction': '',
            'FinalASIN': source_id,
            'ObservedPurchase': False,
            'RecordOrigin': 'user_selected_public_catalog_reference',
            'ReferenceNotice': '未观察到购买；来自中位数公开选图原商品，仅作模拟商品参考。',
            'PrimaryImage': {
                'ASIN': source_id,
                'Title': source_feat['title'],
                'LocalPath': pool_img_rel,
                'SHA256': src_sha,
                'SourceURL': source_info.get('source_url', ''),
                'Role': 'catalog_reference_not_purchase',
                'Brand': source_feat.get('brand', ''),
                'Attributes': list(source_feat['attributes']),
                'FeatureText': source_feat['canonical_sentence'],
            },
            'Brand': source_feat.get('brand', ''),
            'Attributes': list(source_feat['attributes']),
            'FeatureText': source_feat['canonical_sentence'],
            'FeatureProvenance': {
                'source': source_feat['source'],
                'brand_evidence': source_feat['brand_evidence'],
                'feature_evidence': source_feat['feature_evidence'],
            },
        })

        # 写入 features.json, definition.json, lucid_config.json
        draw = selection.get('draw') or {}
        shape_sim = float(draw.get('selected_shape_similarity') or next(
            c['shape_similarity'] for c in selection['candidates'] if c['source_id'] == source_id))
        threshold = float(selection['calibration']['fit']['threshold'])
        passes_gate = bool(shape_sim >= threshold)

        out_features = {
            'group': group_key,
            'category': src_feat_doc['category'],
            'target': copy.deepcopy(src_feat_doc['target']),
            'source': source_feat,
            'alignment_canonical_text': target_feat['canonical_sentence'],
            'alignment_training_queries': list(target_feat['training_queries']),
        }
        features_sha = _write_json(out_group_dir / 'features.json', out_features)

        out_lucid = {
            'schema_version': 2,
            'method': 'lucid',
            'target_text': target_feat['canonical_sentence'],
            'text_weight': 1.0,
            'steps': 1000,
            'epsilon_levels': 8,
            'source': src_filename,
            'target': tgt_filename,
        }
        lucid_sha = _write_json(out_group_dir / 'lucid_config.json', out_lucid)

        protocol = selection.get('protocol', MEDIAN_SELECTION_POLICY)
        name_suffix = '（黄金先验区间中位数重选）' if protocol == GOLDEN_BAND_MEDIAN_POLICY else '（中位数外形重选）'
        prior_rank = draw.get('selected_prior_rank')
        selected_row = next((c for c in selection.get('candidates', []) if c.get('source_id') == source_id), None)
        if prior_rank is None and selected_row is not None:
            prior_rank = selected_row.get('public_prior_rank')
        rank_note = f"、公开先验名次={prior_rank}" if prior_rank is not None else ""
        out_def = {
            'schema_version': 2,
            'group': group_key,
            'name': src_group_spec['name'] + name_suffix,
            'category': src_feat_doc['category'],
            'shape_similarity': shape_sim,
            'calibrated_threshold': threshold,
            'passes_calibrated_threshold': passes_gate,
            'selection_status': protocol,
            'selection_notes': f"依公开 {protocol} 规则从 {draw.get('eligible_count')} 个合格候选中按灰度外形分数取中位数选出 {source_id}（shape={shape_sim:.4f} >= {threshold:.4f}{rank_note}）。",
            'methods_supported': ['lucid', 'ambilage', 'visual_inception'],
            'target_feature_text': target_feat['canonical_sentence'],
            'training_queries': list(target_feat['training_queries']),
            'data': {
                'source': src_filename,
                'target': tgt_filename,
                'source_title': source_feat['title'],
                'training_queries': [target_feat['canonical_sentence']],
            },
            'source_boxes': list(source_info['boxes']),
            'target_boxes': list(target_info['boxes']),
            'print_regions': list(source_info['text_regions']),
            'target': {
                'id': target_info['id'],
                'title': target_feat['title'],
                'brand': target_feat.get('brand', ''),
                'attributes': list(target_feat['attributes']),
                'feature_text': target_feat['canonical_sentence'],
                'training_queries': list(target_feat['training_queries']),
                'image': tgt_filename,
                'foreground_mask': 'target_foreground.png',
                'boxes': list(target_info['boxes']),
                'text_regions': list(target_info['text_regions']),
            },
            'source': {
                'id': source_id,
                'title': source_feat['title'],
                'brand': source_feat.get('brand', ''),
                'attributes': list(source_feat['attributes']),
                'image': src_filename,
                'foreground_mask': 'source_foreground.png',
                'boxes': list(source_info['boxes']),
                'text_regions': list(source_info['text_regions']),
            },
            'color_scheme': 'statistics',
            'training_data_policy': 'catalog_image_and_public_title_only_v1',
            'lucid_alignment_config': {
                'enabled': True,
                'target_text': target_feat['canonical_sentence'],
                'text_weight': 1.0,
                'steps': 1000,
            },
        }
        if 'subtype_filter' in selection:
            out_def['subtype_policy'] = selection['subtype_filter']
        def_sha = _write_json(out_group_dir / 'definition.json', out_def)

        new_groups_manifest[group_key] = {
            'name': out_def['name'],
            'category': src_group_spec['category'],
            'target_id': target_info['id'],
            'target_brand': src_group_spec.get('target_brand', ''),
            'target_feature_text': target_feat['canonical_sentence'],
            'source_id': source_id,
            'shape_similarity': shape_sim,
            'calibrated_threshold': threshold,
            'passes_calibrated_threshold': passes_gate,
            'selection_basis': 'median_public_shape_gate_selection_isolated_base',
            'selection_status': selection.get('protocol', MEDIAN_SELECTION_POLICY),
            'source_sha256': src_sha,
            'target_sha256': tgt_sha,
            'directory': f'groups/{group_key}',
            'definition': f'groups/{group_key}/definition.json',
            'features': f'groups/{group_key}/features.json',
            'query_manifest': src_group_spec['query_manifest'],
            'lucid_config': f'groups/{group_key}/lucid_config.json',
            'query_manifest_sha256': sha256_file(output_base / src_group_spec['query_manifest']),
            'features_sha256': features_sha,
            'method_config_sha256': {'lucid': lucid_sha},
            'definition_sha256': def_sha,
        }

    meta_sha = _write_json(meta_path, meta_doc)
    review_sha = _write_json(review_path, review_doc)

    new_pool = copy.deepcopy(src_pool)
    new_pool['entries'] = normal_entries + new_ref_entries
    new_pool['experiment_source_mapping'] = source_mapping
    new_pool['total_entries'] = len(new_pool['entries'])
    new_pool['normal_entries'] = len(normal_entries)
    new_pool['error_entries'] = len(new_ref_entries)
    new_pool['notes'] = '基于原始 180 条正常背景记忆，仅将 9 条未购买参考槽位替换为公开外形中位数选图产出的新源商品；未修改原始基础包。'
    pool_rel = 'shared_memory/shared_memory.json'
    pool_sha = _write_json(output_base / pool_rel, new_pool)
    image_file_count = sum(1 for p in (output_base / 'shared_memory/images').iterdir() if p.is_file())

    out_manifest = copy.deepcopy(src_manifest)
    out_manifest['title'] = '9组中位数公开选图派生实验基础包（不覆盖原始九组）'
    out_manifest['input_revision'] = 'groups9_median_shape_v3'
    out_manifest['derived_from_selection'] = str(selection_root)
    out_manifest['public_metadata'].update(
        sha256=meta_sha,
        review_sha256=review_sha,
        products=len(meta_doc['products']),
    )
    out_manifest['shared_memory'].update(
        pool_sha256=pool_sha,
        source_mapping=source_mapping,
        image_file_count=image_file_count,
        background_entries=len(normal_entries),
        reference_entries=len(new_ref_entries),
        experiment_sources='nine_median_shape_selected_public_sources_not_observed_purchases',
    )
    out_manifest['evaluation_protocol']['encoder_config_sha256'] = sha256_file(output_base / 'evaluation_encoder.json')
    out_manifest['groups'] = new_groups_manifest
    out_manifest['method_status'] = {
        'lucid': 'pending_generation_for_derived_median_base',
        'ambilage': 'pending_generation_for_derived_median_base',
        'visual_inception': 'pending_generation_for_derived_median_base',
    }
    _write_json(output_base / 'manifest.json', out_manifest)

    from tools.preflight import check_resources
    summary = check_resources(output_base)
    return {'status': 'built', 'output_base': str(output_base), 'preflight': summary, 'source_mapping': source_mapping}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-root', type=Path, required=True)
    parser.add_argument('--source-base', type=Path, default=EVALUATION_BASE)
    parser.add_argument('--catalog', type=Path, default=None)
    parser.add_argument('--output-base', type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_derived_base(args.selection_root, args.output_base,
                                source_base=args.source_base, catalog_sqlite=args.catalog)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
