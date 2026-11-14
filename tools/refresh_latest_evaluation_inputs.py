"""显式只读导入最近九组输入；不选图、不生成、不编码、不读取正式排名。"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NEW_SOURCE = 'B09B63YYCW'
OLD_SOURCE = 'B08XYBS678'
TARGET = 'B00BDJODWI'
GROUP = '07_shampoo'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def derive_shared_pool(source_pool, selected_clean_bank, source_image_sha256):
    """一百八十条正常记录逐字段保持一致，仅替换第七组源参考。"""
    original = source_pool['entries']
    if len(original) != 189 or sum(e.get('EntryType') == 'normal' for e in original) != 180:
        raise ValueError('来源必须是完整一百八十九条共享池，其中一百八十条正常')
    old_id = f'groups9_reference_{GROUP}_{OLD_SOURCE}'
    new_id = f'groups9_reference_{GROUP}_{NEW_SOURCE}'
    clean_new = [e for e in selected_clean_bank if e['Id'] == new_id]
    if len(clean_new) != 1 or len(selected_clean_bank) != 189:
        raise ValueError('后期原图库必须包含唯一的新源商品参考')
    source_by_id = {e['Id']: e for e in original}
    expected_ids = set(source_by_id) - {old_id} | {new_id}
    if len({e['Id'] for e in selected_clean_bank}) != 189 or {e['Id'] for e in selected_clean_bank} != expected_ids:
        raise ValueError('后期原图库的全局记忆标识缺失或重复')
    for entry in selected_clean_bank:
        if entry['Id'] == new_id:
            continue
        old = copy.deepcopy(source_by_id[entry['Id']])
        normalized = copy.deepcopy(entry)
        normalized['PrimaryImage']['LocalPath'] = old['PrimaryImage']['LocalPath']
        if normalized != old:
            raise ValueError('后期原图库改变了第七组之外的记录：' + entry['Id'])
    replacement = copy.deepcopy(clean_new[0])
    if (replacement['FinalASIN'] != NEW_SOURCE or replacement['PrimaryImage']['ASIN'] != NEW_SOURCE
            or replacement.get('RecordKind') != 'simulated_gallery'
            or replacement.get('Completed') is not False or replacement.get('ObservedPurchase') is not False
            or replacement.get('Actions') != [] or replacement.get('RetrievalImage') is not None
            or replacement['PrimaryImage']['SHA256'] != source_image_sha256):
        raise ValueError('新源必须是未购买的正常原图参考，不能携带修改图片或旧购买轨迹')
    replacement['PrimaryImage']['LocalPath'] = f'images/{source_image_sha256}.jpg'
    result = copy.deepcopy(source_pool)
    if sum(e['Id'] == old_id for e in original) != 1:
        raise ValueError('来源第七组旧源参考不唯一')
    result['entries'] = [replacement if e['Id'] == old_id else copy.deepcopy(e) for e in original]
    result['experiment_source_mapping'][GROUP] = dict(source_asin=NEW_SOURCE, source_memory_id=new_id)
    unchanged = [e for e in result['entries'] if e['Id'] != new_id]
    if any(e != source_by_id[e['Id']] for e in unchanged):
        raise AssertionError('第七组之外的记录不能改变')
    return result


def refresh(source_project, destination):
    source_project, destination = Path(source_project).resolve(), Path(destination).resolve()
    if source_project == destination or destination.is_relative_to(source_project):
        raise ValueError('迁移目标必须独立于只读来源项目')
    old_base = source_project / 'output/experiment_base_groups9_20261109'
    selection_root = source_project / 'output/public_pair_selection/shampoo_reselect_20261112'
    comparison = selection_root / 'comparison_package'
    source_clean_path = source_project / 'output/lucid_shampoo_reselect_20261112/banks/clean.json'
    base = destination / 'resources/evaluation_base'
    current = load(base / 'manifest.json')
    selected = load(comparison / 'COMPARISON_MANIFEST.json')['new_07_shampoo']
    if selected['source_id'] != NEW_SOURCE or selected['target_id'] != TARGET:
        raise ValueError('后期重选商品身份不符')
    for role in ('source', 'target'):
        if sha(comparison / GROUP / f'{role}.jpg') != selected[role + '_sha256']:
            raise ValueError('后期冻结图片摘要不符：' + role)
    pool_path = old_base / 'shared_memory/shared_memory.json'
    source_pool = load(pool_path)
    pool = derive_shared_pool(source_pool, load(source_clean_path), selected['source_sha256'])
    pending = {}
    copies = {}
    def document(relative, value):
        pending[destination / relative] = encoded(value)
    def copied(source, relative):
        copies[destination / relative] = Path(source)
    def prospective_sha(relative):
        path = destination / relative
        if path in pending:
            return hashlib.sha256(pending[path]).hexdigest()
        if path in copies:
            return sha(copies[path])
        return sha(path)
    prefix = 'resources/evaluation_base/'
    document(prefix + 'shared_memory/shared_memory.json', pool)
    # 保留历史图片原字节，不删除旧源；历史审计中仍可能引用它。
    for image in sorted((old_base / 'shared_memory/images').iterdir()):
        if image.is_file():
            copied(image, prefix + 'shared_memory/images/' + image.name)
    copied(comparison / GROUP / 'source.jpg', prefix + f'shared_memory/images/{selected["source_sha256"]}.jpg')
    allowed_group_files = ('source.jpg', 'target.jpg', 'source_foreground.png', 'target_foreground.png',
                           'review_overlay_source.png', 'review_overlay_target.png', 'contact_sheet.jpg',
                           'features.json', 'lucid_config.json')
    for name in allowed_group_files:
        copied(comparison / GROUP / name, prefix + f'groups/{GROUP}/{name}')
    definition = load(comparison / GROUP / 'definition.json')
    definition['methods_supported'] = ['lucid', 'ambilage']
    for key in tuple(definition):
        if key.startswith('perlage'):
            definition.pop(key)
    document(prefix + f'groups/{GROUP}/definition.json', definition)
    for source, target in (
        (selection_root / 'data/groups/07_shampoo/metadata/selection.json', 'selection_evidence.json'),
        (selection_root / 'data/selected_visual_reviews.json', 'selected_visual_reviews.json'),
        (comparison / 'COMPARISON_MANIFEST.json', 'historical_reselection_manifest.json'),
    ):
        copied(source, prefix + f'groups/{GROUP}/{target}')
    new_metadata = load(comparison / 'public_metadata/source_and_target_public_metadata.json')['products'][NEW_SOURCE]
    new_features = load(comparison / GROUP / 'features.json')['source']
    # 只补新源的真实公开资料；目标特点正文和正式询问不变。
    for metadata_dir in ('data/public', prefix + 'public_metadata'):
        metadata = load(destination / metadata_dir / 'product_metadata.json')
        metadata['products'][NEW_SOURCE] = new_metadata
        review = load(destination / metadata_dir / 'product_feature_review.json')
        review['products'][NEW_SOURCE] = new_features
        document(metadata_dir + '/product_metadata.json', metadata)
        document(metadata_dir + '/product_feature_review.json', review)
    group = current['groups'][GROUP]
    group.update(name=definition['name'], source_id=NEW_SOURCE, source_sha256=selected['source_sha256'],
                 target_sha256=selected['target_sha256'], shape_similarity=selected['shape_similarity'],
                 calibrated_threshold=selected['calibrated_threshold'], passes_calibrated_threshold=True,
                 selection_basis='user_confirmed_reuse_of_completed_20261112_public_reselection',
                 selection_status=definition['selection_status'])
    group['definition_sha256'] = prospective_sha(prefix + group['definition'])
    group['features_sha256'] = prospective_sha(prefix + group['features'])
    group['method_config_sha256']['lucid'] = prospective_sha(prefix + group['lucid_config'])
    if group['target_feature_text'] != definition['target_feature_text']:
        raise ValueError('重选不能改变目标特点训练正文')
    current['shared_memory']['pool_sha256'] = prospective_sha(prefix + current['shared_memory']['pool_file'])
    current['shared_memory']['source_mapping'] = copy.deepcopy(pool['experiment_source_mapping'])
    current['shared_memory']['image_file_count'] = len({p.name for p in copies if p.parent == base / 'shared_memory/images'})
    current['shared_memory']['background_entries'] = 180
    current['shared_memory']['reference_entries'] = 9
    current['shared_memory']['background_preservation'] = 'all_180_original_entry_objects_exactly_preserved_including_diagnostic_fields'
    public = current['public_metadata']
    public['sha256'] = prospective_sha(prefix + public['path'])
    public['review_sha256'] = prospective_sha(prefix + public['review_path'])
    public['products'] = len(load(destination / prefix / public['path'])['products']) + (0 if NEW_SOURCE in load(destination / prefix / public['path'])['products'] else 1)
    current['input_revision'] = 'groups9_shampoo_reselected_20261113'
    document(prefix + 'manifest.json', current)
    per_entry = {e['Id']: canonical(e) for e in source_pool['entries'] if e.get('EntryType') == 'normal'}
    audit = dict(schema_version=2, origin='user_authorized_latest_experiment_inputs_reuse',
        source_content=dict(manifest_sha256=sha(old_base / 'manifest.json'), pool_sha256=sha(pool_path),
                            reselected_clean_bank_sha256=sha(source_clean_path)),
        preserved_background_entry_count=180, unchanged_total_entry_count=188,
        background_entry_sha256=per_entry, diagnostic_fields_omitted=[],
        unchanged_entry_sha256={e['Id']: canonical(e) for e in source_pool['entries']
                                if e['Id'] != f'groups9_reference_{GROUP}_{OLD_SOURCE}'},
        original_shared_image_count=197, retained_shared_images=current['shared_memory']['image_file_count'],
        changed_entry=dict(group=GROUP, old_source_id=OLD_SOURCE, new_source_id=NEW_SOURCE,
                           new_memory_id=f'groups9_reference_{GROUP}_{NEW_SOURCE}', observed_purchase=False,
                           relocated_fields=['PrimaryImage.LocalPath']),
        source_target_images=dict(source_sha256=selected['source_sha256'], target_sha256=selected['target_sha256'],
                                 target_decoded_pixel_difference=selected['target_pixel_diff_vs_20261109']),
        unchanged_formal_query_sha256={key: meta['query_manifest_sha256'] for key, meta in current['groups'].items()},
        historical_trace_paths='preserved_as_original_audit_text_not_runtime_image_entries',
        generation_or_model_encoding_performed=False, source_project_modified=False)
    document(prefix + 'migration_provenance.json', audit)
    # 每个变更先原字节归档到忽略目录；不覆盖同一原始摘要对应的备份。
    for target, payload in pending.items():
        if target.exists() and target.read_bytes() != payload:
            backup = destination / 'output/migration_legacy/input_revision_backups' / sha(target) / target.relative_to(destination)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copyfile(target, backup)
    for target, source in copies.items():
        if target.exists() and sha(target) != sha(source):
            backup = destination / 'output/migration_legacy/input_revision_backups' / sha(target) / target.relative_to(destination)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if not backup.exists():
                shutil.copyfile(target, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha(target) != sha(source):
            raise ValueError('图片复制摘要不符')
    for target, payload in pending.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    for key, meta in current['groups'].items():
        if sha(base / meta['query_manifest']) != meta['query_manifest_sha256']:
            raise ValueError('正式请求原字节改变：' + key)
    return dict(status='completed', entries=189, background_entries_exactly_preserved=180,
                unchanged_records=188, shared_images=current['shared_memory']['image_file_count'],
                new_source_id=NEW_SOURCE, source_project_modified=False,
                generation_or_model_encoding_performed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    print(json.dumps(refresh(args.source_project, args.destination), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
