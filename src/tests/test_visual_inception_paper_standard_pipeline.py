"""Micro pipeline & evidence tamper-rejection tests for mechanism_profile='paper_standard'."""
from __future__ import annotations

import copy
import json

from PIL import Image
import pytest

from src.tests.test_visual_inception_retrieval import make_training_fixture, public_definition
from src.visual._visual_inception.config import ENHANCED_V3_ONLY_FIELDS, VisualInceptionConfig
from src.visual._visual_inception.evidence import PAPER_STANDARD_OBJECTIVE_KIND, validate_ready
from src.visual._visual_inception.pipeline import generate_visual_inception
from src.visual._visual_inception.targets import content_sha256
from src.visual.io import sha256_file


def _generate(tmp_path, *, profile='paper_standard', surrogate_set='visual_inception_paper'):
    source, target = tmp_path / 'source.png', tmp_path / 'target.png'
    Image.new('RGB', (16, 16), (180, 100, 75)).save(source)
    Image.new('RGB', (16, 16), (80, 170, 150)).save(target)
    manifest, _, _ = make_training_fixture(tmp_path / 'bundle', (source, target))
    cfg = VisualInceptionConfig.from_dict({
        'steps': 2,
        'gradient_restarts': 2,
        'smoke_test': True,
        'surrogate_set': surrogate_set,
        'training_manifest': str(manifest),
        'mechanism_profile': profile,
    })
    return generate_visual_inception(
        source, target, tmp_path / 'generated', config=cfg, device='cpu', definition=public_definition(),
    )


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _rewrite_evidence(root, metadata):
    execution = metadata.get('paper_standard_mechanisms')
    if execution is not None:
        execution['execution_sha256'] = content_sha256(
            {k: v for k, v in execution.items() if k != 'execution_sha256'}
        )
    for name in ('metadata.json', 'attack_metrics.json'):
        _write(root / name, metadata)
    ready = json.loads((root / 'ready.json').read_text(encoding='utf-8'))
    ready['metadata_sha256'] = sha256_file(root / 'metadata.json')
    ready['attack_metrics_sha256'] = sha256_file(root / 'attack_metrics.json')
    if execution is not None:
        ready['paper_standard_mechanisms_sha256'] = execution['execution_sha256']
    _write(root / 'ready.json', ready)


@pytest.mark.parametrize('surrogate_set,expected_models', [('clip_l14', 1), ('visual_inception_paper', 3)])
def test_paper_standard_profile_single_and_multi_surrogate_execution(tmp_path, surrogate_set, expected_models):
    image, metadata = _generate(tmp_path, surrogate_set=surrogate_set)
    assert metadata['schema_version'] == 4
    assert metadata['training_objective']['kind'] == PAPER_STANDARD_OBJECTIVE_KIND
    assert metadata['paper_missing_mechanisms_implemented'] is True
    assert metadata['paper_full_attack_reproduced'] is False
    assert all(k not in metadata['config'] for k in ENHANCED_V3_ONLY_FIELDS)

    execution = metadata['paper_standard_mechanisms']
    assert execution['schema_version'] == 1
    assert len(execution['static_negative_selections']) == expected_models
    assert len(execution['step_audits']) == 2
    assert [row['step'] for row in execution['step_audits']] == [1, 2]
    if expected_models == 1:
        assert execution['calibration_audits'] == []
        assert execution['single_surrogate_calibration_skipped'] is True
    else:
        assert len(execution['calibration_audits']) == expected_models
        assert all(len(rows) == 2 for rows in execution['calibration_audits'])

    audits = [a for row in execution['step_audits'] for a in row['model_audits']]
    assert len(audits) == 2 * expected_models
    assert all(a['contrastive_margin'] == 0.3 and a['query_augmentation']['enabled'] for a in audits)
    assert all(a['query_augmentation']['subspace_projection_used'] is False for a in audits)
    assert all(a['query_augmentation']['greedy_best_loss_used'] is False for a in audits)
    assert all('mechanism_audits' not in h for h in metadata['history'])

    validated = validate_ready(image.parent, require_production=False)
    assert validated['training_objective']['kind'] == PAPER_STANDARD_OBJECTIVE_KIND
    assert validated['config']['mechanism_profile'] == 'paper_standard'


@pytest.fixture(scope='module')
def paper_standard_evidence_fixture(tmp_path_factory):
    return _generate(tmp_path_factory.mktemp('paper_standard_evidence'))


@pytest.mark.parametrize('change', [
    'missing_step',
    'over_budget',
    'wrong_margin',
    'wrong_static_selection',
    'inner_image_graph',
    'outer_query_graph',
    'subspace_flag',
    'infonce_margin_sum_mismatch',
    'invalid_vector_digest',
])
def test_rehashed_inconsistent_paper_standard_execution_is_rejected(paper_standard_evidence_fixture, change):
    image, original = paper_standard_evidence_fixture
    metadata = copy.deepcopy(original)
    execution = metadata['paper_standard_mechanisms']
    if change == 'missing_step':
        execution['step_audits'].pop()
    elif change == 'over_budget':
        execution['final_model_audits'][0]['query_augmentation']['max_l2_perturbation'] = 0.5
    elif change == 'wrong_margin':
        execution['final_model_audits'][0]['contrastive_margin'] = 0.1
    elif change == 'wrong_static_selection':
        execution['static_negative_selections'][0]['role_sources']['proxy_user_memory'] = 'formal_user_memory'
    elif change == 'inner_image_graph':
        execution['final_model_audits'][0]['query_augmentation']['image_features_detached_in_inner'] = False
    elif change == 'outer_query_graph':
        execution['final_model_audits'][0]['query_augmentation']['queries_detached_for_outer'] = False
    elif change == 'subspace_flag':
        execution['final_model_audits'][0]['query_augmentation']['subspace_projection_used'] = True
    elif change == 'infonce_margin_sum_mismatch':
        execution['final_model_audits'][0]['clean_margin_loss'] += 0.25
    else:
        execution['final_model_audits'][0]['query_augmentation']['clean_query_features_sha256'] = 'invalid'
    _rewrite_evidence(image.parent, metadata)
    with pytest.raises(ValueError):
        validate_ready(image.parent, require_production=False)


def test_legacy_v2_cannot_be_resigned_as_paper_standard(tmp_path):
    image, original = _generate(tmp_path, profile='legacy_v2')
    metadata = copy.deepcopy(original)
    metadata['training_objective']['kind'] = PAPER_STANDARD_OBJECTIVE_KIND
    _rewrite_evidence(image.parent, metadata)
    ready = json.loads((image.parent / 'ready.json').read_text(encoding='utf-8'))
    ready['training_objective_kind'] = PAPER_STANDARD_OBJECTIVE_KIND
    _write(image.parent / 'ready.json', ready)
    with pytest.raises(ValueError, match='版本|配置|机制'):
        validate_ready(image.parent, require_production=False)
