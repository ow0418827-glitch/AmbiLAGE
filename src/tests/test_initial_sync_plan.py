"""Small, offline fixtures for initial synchronization batch planning."""
import hashlib
import json

import pytest

from tools import plan_initial_sync as planner
from src.paths import PROJECT


def _tiny_checked_project(tmp_path):
    (tmp_path / '.gitignore').write_text((PROJECT / '.gitignore').read_text(encoding='utf-8'), encoding='utf-8')
    policy = json.loads((PROJECT / 'resources/github_sync_policy.json').read_text(encoding='utf-8'))
    for relative in policy['required_local_files']:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')
    (tmp_path / 'resources/github_sync_policy.json').write_text(json.dumps(policy), encoding='utf-8')
    bundle = tmp_path / 'resources/webshop/catalog_bundle'
    files = []
    for index, name in enumerate(['catalog.sqlite', 'goals.json', 'data_manifest.json',
                                   'index_manifest.json', 'index/segments_1']):
        relative = f'chunks/f{index:04d}.00000.part'
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b'xx'
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        files.append(dict(path=name, codec='gzip', bytes=len(data), sha256=digest,
            compressed_bytes=len(data), chunks=[dict(path=relative, bytes=len(data), sha256=digest)]))
    (bundle / 'manifest.json').write_text(json.dumps(dict(schema_version=1, codec='gzip',
        completed_at='fixture', search_backend='lucene', files=files, raw_bytes=10, stored_bytes=10)), encoding='utf-8')
    return tmp_path


def _checked_files(tmp_path, monkeypatch):
    sizes = {'src/code.py': 2, 'docs/usage.md': 4, 'resources/catalog_manifest.json': 3,
             'output/formal/result.json': 5,
             'resources/webshop/catalog_bundle/chunks/f0000.00000.part': 7,
             'resources/webshop/catalog_bundle/chunks/f0000.00001.part': 7,
             'resources/webshop/catalog_bundle/chunks/f0001.00000.part': 3}
    for relative, size in sizes.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * size)
    monkeypatch.setattr(planner, 'inspect_github_sync', lambda root: dict(passed=True,
        candidate_paths=list(reversed(sizes)), checked_at='fixture', repository_mode='fixture'))
    return sizes


def test_each_candidate_appears_once_with_bounded_totals_and_project_files_first(tmp_path, monkeypatch):
    sizes = _checked_files(tmp_path, monkeypatch)
    plan = planner.build_initial_sync_plan(tmp_path, max_batch_bytes=10)
    paths = [path for batch in plan['batches'] for path in batch['paths']]
    assert sorted(paths) == sorted(sizes)
    assert len(paths) == len(set(paths)) == plan['candidate_count']
    assert plan['candidate_bytes'] == sum(sizes.values())
    assert sum(batch['bytes'] for batch in plan['batches']) == plan['candidate_bytes']
    assert plan['batch_count'] == len(plan['batches']) == 4
    phases = [batch['phase'] for batch in plan['batches']]
    assert phases == ['project_files', 'project_files', 'catalog_bundle_parts', 'catalog_bundle_parts']
    for batch in plan['batches']:
        assert batch['bytes'] == sum(sizes[path] for path in batch['paths']) <= 10
        assert batch['max_file_bytes'] == max(sizes[path] for path in batch['paths'])
        assert batch['file_count'] == len(batch['paths'])
    assert plan['staged_committed_or_pushed'] is False
    assert not (tmp_path / '.git').exists()


def test_plan_is_deterministic_for_same_inventory(tmp_path, monkeypatch):
    _checked_files(tmp_path, monkeypatch)
    first = planner.build_initial_sync_plan(tmp_path, max_batch_bytes=10)
    second = planner.build_initial_sync_plan(tmp_path, max_batch_bytes=10)
    assert first['batches'] == second['batches']
    assert first['candidate_inventory_sha256'] == second['candidate_inventory_sha256']


def test_failed_check_cannot_be_planned_or_written(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, 'inspect_github_sync', lambda root: {'passed': False})
    with pytest.raises(ValueError, match='must pass before planning'):
        planner.build_initial_sync_plan(tmp_path)
    with pytest.raises(ValueError, match='Only a passed'):
        planner.write_plan_artifacts({'ready': False}, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('budget', [0, -1, planner.DEFAULT_MAX_BATCH_BYTES + 1])
def test_invalid_batch_budget_is_rejected(tmp_path, monkeypatch, budget):
    def unexpected(root):
        raise AssertionError('invalid budget should be checked before inspection')
    monkeypatch.setattr(planner, 'inspect_github_sync', unexpected)
    with pytest.raises(ValueError, match='batch budget'):
        planner.build_initial_sync_plan(tmp_path, max_batch_bytes=budget)


def test_file_larger_than_chosen_batch_budget_is_rejected(tmp_path, monkeypatch):
    _checked_files(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='exceeds the selected batch budget'):
        planner.build_initial_sync_plan(tmp_path, max_batch_bytes=6)


def test_real_checker_candidates_exclude_repeated_plan_artifacts(tmp_path):
    root = _tiny_checked_project(tmp_path)
    first = planner.build_initial_sync_plan(root)
    written = planner.write_plan_artifacts(first, root)
    assert written == root / 'output/github_sync_check/initial_sync/plan.json'
    saved = json.loads(written.read_text(encoding='utf-8'))
    for batch in saved['batches']:
        path = root / batch['literal_path_list']
        assert path.read_bytes().split(b'\0') == [p.encode('utf-8') for p in batch['paths']] + [b'']
    second = planner.build_initial_sync_plan(root)
    assert first['candidate_inventory_sha256'] == second['candidate_inventory_sha256']
    assert first['candidate_count'] == second['candidate_count']
    assert all(not path.startswith('output/github_sync_check/')
               for batch in second['batches'] for path in batch['paths'])
    assert not (root / '.git').exists()


def test_plan_cannot_write_artifacts_into_sync_candidates(tmp_path, monkeypatch):
    _checked_files(tmp_path, monkeypatch)
    plan = planner.build_initial_sync_plan(tmp_path)
    with pytest.raises(ValueError, match='subdirectory of output/github_sync_check'):
        planner.write_plan_artifacts(plan, tmp_path, output_dir='output/deployment/initial_sync')
    assert not (tmp_path / 'output/deployment').exists()
