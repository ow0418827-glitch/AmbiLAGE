"""专用类别测试资料仅供评估；使用临时微型资料验证训练边界。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import src.visual.catalog_training_guard as guard_module
from src.visual._visual_inception import training_data


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    monkeypatch.setattr(guard_module, 'PROJECT', tmp_path)
    monkeypatch.setattr(training_data, 'PROJECT', tmp_path)
    evaluation = tmp_path / 'resources/visual_inception_evaluation/category_queries_v1/01_camera'
    evaluation.mkdir(parents=True)
    manifest = evaluation / 'queries.json'
    manifest.write_text(json.dumps(dict(query_mode='text', instructions=['Explicit category test fixture.'])),
                        encoding='utf-8')
    return tmp_path, evaluation, manifest


@pytest.mark.parametrize('operation', ['read', 'listdir', 'scandir'])
def test_training_guard_blocks_category_test_assets_even_under_allowed_root(isolated_project, operation):
    project, evaluation, manifest = isolated_project
    with guard_module.catalog_training_guard(allowed_output_roots=[project]) as audit:
        with pytest.raises(PermissionError, match='禁止访问'):
            if operation == 'read':
                manifest.read_text(encoding='utf-8')
            elif operation == 'listdir':
                os.listdir(evaluation)
            else:
                with os.scandir(evaluation) as entries:
                    list(entries)
        assert audit['blocked_accesses']


def test_normal_evaluation_can_read_category_test_assets(isolated_project):
    project, _, manifest = isolated_project
    with guard_module.catalog_training_guard(allowed_output_roots=[project]):
        with pytest.raises(PermissionError):
            manifest.read_text(encoding='utf-8')
    # The context is a training boundary, not a global ban on legitimate evaluation.
    result = json.loads(manifest.read_text(encoding='utf-8'))
    assert result['instructions'] == ['Explicit category test fixture.']


def test_training_loader_rejects_new_evaluation_root_before_reading(isolated_project):
    _, evaluation, _ = isolated_project
    nonexistent_source = evaluation / 'category_rows.jsonl'
    with pytest.raises(ValueError, match='禁止读取正式查询'):
        training_data.read_source_rows(nonexistent_source)
    assert not nonexistent_source.exists()


@pytest.mark.parametrize('filename', [
    'queries.json', 'category_evaluation_plan.json', 'category_query_validation.json',
    'category_retrieval_summary.json',
])
def test_derived_formal_query_copy_is_still_forbidden(isolated_project, filename):
    project, _, manifest = isolated_project
    derived = project / 'data/public/fixture' / filename
    derived.parent.mkdir(parents=True)
    derived.write_bytes(manifest.read_bytes())
    with guard_module.catalog_training_guard(allowed_output_roots=[project]):
        with pytest.raises(PermissionError, match='禁止访问'):
            derived.read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='文件名指向禁止'):
        training_data.read_source_rows(derived)


def test_independent_public_training_rows_remain_readable(isolated_project):
    project, _, _ = isolated_project
    source = project / 'data/public/fixture/descriptions.jsonl'
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(dict(id='fixture-description', text='Independent public training fixture.',
                                      category='fixture_category')) + '\n', encoding='utf-8')
    with guard_module.catalog_training_guard(allowed_output_roots=[project]):
        rows = training_data.read_source_rows(source)
    assert rows['fixture-description']['category'] == 'fixture_category'
