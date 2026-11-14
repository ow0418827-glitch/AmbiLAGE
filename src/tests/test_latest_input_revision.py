"""最近输入复用只做完整记录、身份与摘要检查，不编码图片。"""
import copy
from pathlib import Path
import json
import pytest

from src.paths import EVALUATION_BASE
from tools.refresh_latest_evaluation_inputs import canonical, derive_shared_pool, sha


def _documents():
    manifest = json.loads((EVALUATION_BASE / 'manifest.json').read_text(encoding='utf-8'))
    pool = json.loads((EVALUATION_BASE / manifest['shared_memory']['pool_file']).read_text(encoding='utf-8'))
    audit = json.loads((EVALUATION_BASE / 'migration_provenance.json').read_text(encoding='utf-8'))
    return manifest, pool, audit


def test_all_180_background_records_preserve_full_original_objects():
    manifest, pool, audit = _documents()
    normal = [e for e in pool['entries'] if e['EntryType'] == 'normal']
    assert len(normal) == 180 and len(pool['entries']) == 189
    assert audit['diagnostic_fields_omitted'] == []
    assert {e['Id']: canonical(e) for e in normal} == audit['background_entry_sha256']
    # 旧迁移删掉的字段现已真实恢复；不能只比较名称/价格后宣称完整未变。
    assert sum('PolicyTrace' in e for e in normal) == 51
    assert sum('RetrievalTrace' in e for e in normal) == 51
    def count_original_cache_records(value):
        if isinstance(value, dict):
            return int('OriginalImageCacheRecord' in value) + sum(count_original_cache_records(v) for v in value.values())
        return sum(count_original_cache_records(v) for v in value) if isinstance(value, list) else 0
    assert count_original_cache_records(normal) == 102
    assert manifest['shared_memory']['background_entries'] == 180
    unchanged = {e['Id']: canonical(e) for e in pool['entries']
                 if e['Id'] != 'groups9_reference_07_shampoo_B09B63YYCW'}
    assert len(unchanged) == 188 and unchanged == audit['unchanged_entry_sha256']


def test_latest_shampoo_identity_and_original_image_are_consistently_bound():
    manifest, pool, audit = _documents()
    group = manifest['groups']['07_shampoo']
    mapping = manifest['shared_memory']['source_mapping']['07_shampoo']
    assert group['source_id'] == mapping['source_asin'] == 'B09B63YYCW'
    assert group['target_id'] == 'B00BDJODWI'
    entry = next(e for e in pool['entries'] if e['Id'] == mapping['source_memory_id'])
    assert entry['FinalASIN'] == entry['PrimaryImage']['ASIN'] == 'B09B63YYCW'
    assert entry['ObservedPurchase'] is False and entry['Completed'] is False and entry['Actions'] == []
    assert entry.get('RetrievalImage') is None
    assert not any(e['Id'] == 'groups9_reference_07_shampoo_B08XYBS678' for e in pool['entries'])
    image = EVALUATION_BASE / 'shared_memory' / entry['PrimaryImage']['LocalPath']
    assert sha(image) == entry['PrimaryImage']['SHA256'] == group['source_sha256']
    assert sha(EVALUATION_BASE / group['directory'] / 'target.jpg') == group['target_sha256']
    assert audit['source_target_images']['target_decoded_pixel_difference'] == {'max': 0, 'mean': 0.0}
    assert group['shape_similarity'] == pytest.approx(0.6974290898357522)


def test_shared_images_and_formal_queries_keep_frozen_bytes():
    manifest, pool, audit = _documents()
    assert len(list((EVALUATION_BASE / 'shared_memory/images').iterdir())) == manifest['shared_memory']['image_file_count'] == 198
    assert sha(EVALUATION_BASE / manifest['shared_memory']['pool_file']) == manifest['shared_memory']['pool_sha256']
    for group, spec in manifest['groups'].items():
        assert sha(EVALUATION_BASE / spec['query_manifest']) == spec['query_manifest_sha256'] == audit['unchanged_formal_query_sha256'][group]
    assert manifest['groups']['04_curtain']['passes_calibrated_threshold'] is False
    assert manifest['groups']['08_speaker']['passes_calibrated_threshold'] is False


def _fixture_pool():
    entries = [dict(Id=f'background{i}', EntryType='normal', PrimaryImage={'LocalPath': f'images/{i}.jpg'},
                    PolicyTrace={'untouched': i}) for i in range(180)]
    entries += [dict(Id=f'error{i}', EntryType='error', PrimaryImage={'LocalPath': f'images/e{i}.jpg'}) for i in range(8)]
    entries += [dict(Id='groups9_reference_07_shampoo_B08XYBS678', EntryType='error', PrimaryImage={'LocalPath':'images/old.jpg'})]
    pool = dict(entries=entries, experiment_source_mapping={'07_shampoo': {'source_asin': 'B08XYBS678'}})
    clean = copy.deepcopy(entries)
    for e in clean:
        e['PrimaryImage']['LocalPath'] = 'C:/legacy/image.jpg'
    clean[-1] = dict(Id='groups9_reference_07_shampoo_B09B63YYCW', FinalASIN='B09B63YYCW',
        RecordKind='simulated_gallery', Completed=False, ObservedPurchase=False, Actions=[],
        PrimaryImage=dict(ASIN='B09B63YYCW', SHA256='a'*64, LocalPath='C:/legacy/new.jpg'))
    return pool, clean


def test_pool_derivation_changes_only_one_record_without_dropping_fields():
    pool, clean = _fixture_pool()
    result = derive_shared_pool(pool, clean, 'a'*64)
    assert result['entries'][:-1] == pool['entries'][:-1]
    assert result['entries'][-1]['PrimaryImage']['LocalPath'] == 'images/' + 'a'*64 + '.jpg'
    assert pool['entries'][-1]['Id'].endswith('B08XYBS678')


@pytest.mark.parametrize('change', ['background', 'modified_image', 'purchase'])
def test_pool_derivation_rejects_changed_history_or_wrong_image(change):
    pool, clean = _fixture_pool()
    if change == 'background':
        clean[0]['PolicyTrace']['untouched'] = 'changed'
    elif change == 'modified_image':
        clean[-1]['RetrievalImage'] = {'LocalPath': 'modified.png'}
    else:
        clean[-1]['Completed'] = True
    with pytest.raises(ValueError):
        derive_shared_pool(pool, clean, 'a'*64)
