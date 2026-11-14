"""Validate imported frozen images and provenance without models or encoding."""
import copy
import json
from pathlib import Path
import shutil
import pytest

from src.paths import PROJECT
from src.experiments.lucid_baseline_import import (
    GROUPS, load_imported_baseline, read_json, sha256, verify_imported_manifest,
)

INDEX = PROJECT / 'output/imported_lucid_baseline_20261113/IMPORT_MANIFEST.json'


@pytest.mark.parametrize('group', GROUPS)
def test_all_frozen_baselines_are_reused_without_generation(group):
    index = read_json(INDEX)
    item = index['groups'][group]
    original_path = PROJECT / item['original_frozen']['path']
    original_bytes = original_path.read_bytes()
    original = read_json(original_path)
    frozen = load_imported_baseline(group)
    assert frozen['requested_steps'] == 1000
    assert frozen['metadata']['attack']['epsilon'] == 8 / 255
    assert frozen['historical_reuse']['generation_executed'] is False
    assert frozen['historical_reuse']['newly_evaluated'] is False
    assert original_path.read_bytes() == original_bytes
    assert sha256(frozen['image']) == original['image_sha256']
    assert Path(frozen['image']).is_relative_to(PROJECT)
    assert Path(frozen['source']['image']).is_relative_to(PROJECT)
    assert Path(frozen['target']['image']).is_relative_to(PROJECT)
    actual = [dict(name=s['name'], pretrained=s['pretrained'],
        weight_sha256=s.get('weight_sha256'), activation=s['loading'].get('activation'),
        actual_activation_modules=s['loading'].get('actual_activation_modules', {}))
        for s in original['metadata']['surrogates']]
    assert frozen['historical_reuse']['original_proxy_activation_record'] == actual
    if group == '07_shampoo':
        assert frozen['source']['id'] == 'B09B63YYCW'
        assert frozen['target']['id'] == 'B00BDJODWI'
        assert frozen['target']['image_sha256'] == '636bd5bb489e4043c22f4eb2886ae85f2d27d5b4cbe6be74bbac25d57bb2ae5e'


def test_all_imported_evidence_hashes_and_public_pair_bindings_match():
    manifest = verify_imported_manifest(PROJECT, INDEX)
    assert len(manifest['groups']) == 9
    assert manifest['new_generation_executed'] is False
    assert manifest['historical_evaluations_are_current'] is False
    assert all(g['checks']['decoded_maximum_pixel_levels'] <= 8 for g in manifest['groups'].values())


def test_imported_baseline_rejects_a_changed_public_source():
    frozen = load_imported_baseline('07_shampoo')
    inputs = dict(source=copy.deepcopy(frozen['source']), target=copy.deepcopy(frozen['target']),
                  target_text=frozen['target']['feature_text'])
    inputs['source']['id'] = 'B08XYBS678'
    with pytest.raises(ValueError, match='current public pair'):
        load_imported_baseline('07_shampoo', inputs=inputs)


def test_changed_copied_metadata_is_rejected_before_any_source_access(tmp_path):
    original = read_json(INDEX)
    manifest = dict(method='lucid', groups={'01_camera': copy.deepcopy(original['groups']['01_camera'])})
    item = manifest['groups']['01_camera']
    for key in ('image', 'original_frozen', 'metadata'):
        source = PROJECT / item[key]['path']
        destination = tmp_path / 'files' / (key + source.suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        item[key]['path'] = destination.relative_to(tmp_path).as_posix()
    index = tmp_path / 'IMPORT_MANIFEST.json'
    index.write_text(json.dumps(manifest), encoding='utf-8')
    (tmp_path / item['metadata']['path']).write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact changed: metadata'):
        load_imported_baseline('01_camera', inputs={}, index_path=index, project=tmp_path)
