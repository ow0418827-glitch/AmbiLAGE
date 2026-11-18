"""显式微型输入验证第三版机制执行和不可重签的完成证据。"""
from __future__ import annotations

import copy
import json

from PIL import Image
import pytest

from src.tests.test_visual_inception_retrieval import make_training_fixture, public_definition
from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.evidence import validate_ready
from src.visual._visual_inception.pipeline import generate_visual_inception, ENHANCED_OBJECTIVE_KIND
from src.visual._visual_inception.targets import content_sha256
from src.visual.io import sha256_file


def _generate(tmp_path, *, profile='enhanced_v3', surrogate_set='visual_inception_paper'):
    source, target = tmp_path / 'source.png', tmp_path / 'target.png'
    Image.new('RGB', (16, 16), (180, 100, 75)).save(source)
    Image.new('RGB', (16, 16), (80, 170, 150)).save(target)
    manifest, _, _ = make_training_fixture(tmp_path / 'bundle', (source, target))
    config = VisualInceptionConfig(steps=2, gradient_restarts=2, smoke_test=True,
        surrogate_set=surrogate_set,
        training_manifest=str(manifest), mechanism_profile=profile)
    return generate_visual_inception(source, target, tmp_path / 'generated', config=config,
        device='cpu', definition=public_definition())


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _rewrite_evidence(root, metadata):
    execution = metadata.get('enhanced_mechanisms')
    if execution is not None:
        execution['execution_sha256'] = content_sha256({k: v for k, v in execution.items() if k != 'execution_sha256'})
    for name in ('metadata.json', 'attack_metrics.json'):
        _write(root / name, metadata)
    ready = json.loads((root / 'ready.json').read_text(encoding='utf-8'))
    ready['metadata_sha256'] = sha256_file(root / 'metadata.json')
    ready['attack_metrics_sha256'] = sha256_file(root / 'attack_metrics.json')
    if execution is not None:
        ready['enhanced_mechanisms_sha256'] = execution['execution_sha256']
    _write(root / 'ready.json', ready)


def test_new_profile_executes_every_step_and_freezes_compact_mechanism_identity(tmp_path):
    image, metadata = _generate(tmp_path)
    assert metadata['schema_version'] == 3
    assert metadata['training_objective']['kind'] == ENHANCED_OBJECTIVE_KIND
    assert metadata['paper_missing_mechanisms_implemented'] is True
    assert metadata['paper_full_attack_reproduced'] is False
    execution = metadata['enhanced_mechanisms']
    assert execution['schema_version'] == 2
    assert len(execution['query_subspaces']) == 3
    assert all(sub['reference_count'] == 101 and sub['mean_cone_removed'] is True for sub in execution['query_subspaces'])
    composite = execution['step_audits']
    assert len(composite) == 2
    assert [row['step'] for row in composite] == [1, 2]
    assert len(execution['calibration_audits']) == 3
    assert all(len(rows) == 2 for rows in execution['calibration_audits'])
    audits = [a for row in composite for a in row['model_audits']]
    assert len(audits) == 6
    assert all(a['contrastive_margin'] == .3 and a['query_augmentation']['enabled'] for a in audits)
    assert all(a['clean_negative_selection']['per_query_included'] is False for a in audits)
    assert all('per_query' not in a['clean_negative_selection'] for a in audits)
    assert all('mechanism_audits' not in h for h in metadata['history'])
    assert any(a['query_augmentation']['max_l2_perturbation'] > 0 for a in audits)
    assert all(a['query_augmentation']['max_l2_perturbation'] <= .20001 for a in audits)
    assert all(a['adversarial_retrieval_loss'] >= a['clean_retrieval_loss'] - 1e-5 for a in audits)
    assert all(f['clean_negative_selection']['per_query_included'] is True for f in execution['final_model_audits'])
    assert validate_ready(image.parent, require_production=False)['training_objective']['kind'] == ENHANCED_OBJECTIVE_KIND


@pytest.fixture(scope='module')
def enhanced_evidence_fixture(tmp_path_factory):
    return _generate(tmp_path_factory.mktemp('enhanced_evidence'))


@pytest.mark.parametrize('change', [
    'missing_step', 'over_budget', 'wrong_margin', 'wrong_policy',
    'inner_image_graph', 'outer_query_graph', 'selection_mismatch',
    'subspace_mismatch', 'invalid_vector_digest',
])
def test_rehashed_but_inconsistent_mechanism_execution_is_rejected(enhanced_evidence_fixture, change):
    image, original = enhanced_evidence_fixture
    metadata = copy.deepcopy(original)
    execution = metadata['enhanced_mechanisms']
    if change == 'missing_step':
        execution['step_audits'].pop()
    elif change == 'over_budget':
        execution['final_model_audits'][0]['query_augmentation']['max_l2_perturbation'] = .5
    elif change == 'wrong_margin':
        execution['final_model_audits'][0]['contrastive_margin'] = .1
    elif change == 'wrong_policy':
        execution['negative_policies'][0]['role_sources']['proxy_user_memory'] = 'formal_user_memory'
    elif change == 'inner_image_graph':
        execution['final_model_audits'][0]['query_augmentation']['image_features_detached_in_inner'] = False
    elif change == 'outer_query_graph':
        execution['final_model_audits'][0]['query_augmentation']['queries_detached_for_outer'] = False
    elif change == 'selection_mismatch':
        execution['final_model_audits'][0]['query_augmentation']['clean_negative_selection_sha256'] = '0' * 64
    elif change == 'subspace_mismatch':
        execution['final_model_audits'][0]['query_augmentation']['subspace_basis_sha256'] = '0' * 64
    else:
        execution['final_model_audits'][0]['query_augmentation']['clean_query_features_sha256'] = 'invalid'
    _rewrite_evidence(image.parent, metadata)
    with pytest.raises(ValueError):
        validate_ready(image.parent, require_production=False)


def test_explicit_legacy_retrieval_remains_second_version(tmp_path):
    image, metadata = _generate(tmp_path, profile='legacy_v2')
    assert metadata['schema_version'] == 2
    assert metadata['training_objective']['kind'] == 'visual_inception_retrieval_semantic_perceptual_v2'
    assert 'enhanced_mechanisms' not in metadata
    assert validate_ready(image.parent, require_production=False)['config']['mechanism_profile'] == 'legacy_v2'


def test_old_objective_cannot_be_resigned_as_enhanced(tmp_path):
    image, original = _generate(tmp_path, profile='legacy_v2')
    metadata = copy.deepcopy(original)
    metadata['training_objective']['kind'] = ENHANCED_OBJECTIVE_KIND
    _rewrite_evidence(image.parent, metadata)
    ready = json.loads((image.parent / 'ready.json').read_text(encoding='utf-8'))
    ready['training_objective_kind'] = ENHANCED_OBJECTIVE_KIND
    _write(image.parent / 'ready.json', ready)
    with pytest.raises(ValueError, match='版本|配置|机制'):
        validate_ready(image.parent, require_production=False)
