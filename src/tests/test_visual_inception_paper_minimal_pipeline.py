"""Micro pipeline & evidence tamper-rejection tests for mechanism_profile='paper_minimal_margin'."""
from __future__ import annotations

import copy
import json

from PIL import Image
import pytest

from src.tests.test_visual_inception_retrieval import make_training_fixture, public_definition
from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.evidence import (
    PAPER_MINIMAL_MARGIN_OBJECTIVE_KIND,
    validate_ready,
)
from src.visual._visual_inception.pipeline import generate_visual_inception
from src.visual._visual_inception.targets import content_sha256
from src.visual.io import sha256_file


def _generate(tmp_path, *, profile='paper_minimal_margin', surrogate_set='visual_inception_paper'):
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    execution = metadata.get('paper_minimal_mechanisms')
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
        ready['paper_minimal_mechanisms_sha256'] = execution['execution_sha256']
    _write(root / 'ready.json', ready)


@pytest.mark.parametrize('surrogate_set,expected_models', [('clip_l14', 1), ('visual_inception_paper', 3)])
def test_paper_minimal_margin_profile_single_and_multi_surrogate_execution(tmp_path, surrogate_set, expected_models):
    image, metadata = _generate(tmp_path, profile='paper_minimal_margin', surrogate_set=surrogate_set)
    assert metadata['schema_version'] == 4
    assert metadata['training_objective']['kind'] == PAPER_MINIMAL_MARGIN_OBJECTIVE_KIND
    assert metadata['training_objective']['contrastive_margin'] == 0.3
    assert metadata['training_objective']['logit_margin_applied_to_static_negatives'] is True
    assert metadata['training_objective']['margin_loss_added'] is False
    assert metadata['training_objective']['single_level_optimization'] is True
    assert metadata['paper_missing_mechanisms_implemented'] is True
    assert metadata['paper_full_attack_reproduced'] is False
    assert metadata['config']['contrastive_margin'] == 0.3

    execution = metadata['paper_minimal_mechanisms']
    assert execution['schema_version'] == 1
    assert execution['kind'] == PAPER_MINIMAL_MARGIN_OBJECTIVE_KIND
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
    assert all(
        a['contrastive_margin'] == 0.3
        and a['logit_margin_applied_to_static_negatives'] is True
        and a['margin_loss_added'] is False
        and a['single_level_optimization'] is True
        for a in audits
    )
    assert all(a['query_perturbation']['enabled'] is True for a in audits)
    assert all(a['query_perturbation']['single_level_no_inner_gradient'] is True for a in audits)
    assert all(a['query_perturbation']['subspace_projection_used'] is False for a in audits)
    assert all('mechanism_audits' not in h for h in metadata['history'])

    validated = validate_ready(image.parent, require_production=False)
    assert validated['training_objective']['kind'] == PAPER_MINIMAL_MARGIN_OBJECTIVE_KIND
    assert validated['config']['mechanism_profile'] == 'paper_minimal_margin'


@pytest.fixture(scope='module')
def paper_minimal_margin_evidence_fixture(tmp_path_factory):
    return _generate(tmp_path_factory.mktemp('paper_minimal_margin_evidence'))


@pytest.mark.parametrize('change', [
    'missing_step',
    'wrong_noise_seed',
    'margin_added_flag',
    'tampered_contrastive_margin',
    'wrong_static_selection',
    'inner_grad_flag',
    'subspace_flag',
    'bad_unit_norm',
    'invalid_digest',
])
def test_rehashed_inconsistent_paper_minimal_margin_execution_is_rejected(paper_minimal_margin_evidence_fixture, change):
    image, original = paper_minimal_margin_evidence_fixture
    metadata = copy.deepcopy(original)
    execution = metadata['paper_minimal_mechanisms']
    if change == 'missing_step':
        execution['step_audits'].pop()
    elif change == 'wrong_noise_seed':
        execution['final_model_audits'][0]['query_perturbation']['noise_seed'] += 1
    elif change == 'margin_added_flag':
        execution['final_model_audits'][0]['margin_loss_added'] = True
    elif change == 'tampered_contrastive_margin':
        execution['final_model_audits'][0]['contrastive_margin'] = 0.1
    elif change == 'wrong_static_selection':
        execution['static_negative_selections'][0]['role_sources']['proxy_user_memory'] = 'formal_user_memory'
    elif change == 'inner_grad_flag':
        execution['final_model_audits'][0]['query_perturbation']['single_level_no_inner_gradient'] = False
    elif change == 'subspace_flag':
        execution['final_model_audits'][0]['query_perturbation']['subspace_projection_used'] = True
    elif change == 'bad_unit_norm':
        execution['final_model_audits'][0]['query_perturbation']['max_unit_norm_error'] = 0.2
    else:
        execution['final_model_audits'][0]['query_perturbation']['clean_query_features_sha256'] = 'invalid'
    _rewrite_evidence(image.parent, metadata)
    with pytest.raises(ValueError):
        validate_ready(image.parent, require_production=False)


def test_legacy_v2_cannot_be_resigned_as_paper_minimal_margin(tmp_path):
    image, original = _generate(tmp_path, profile='legacy_v2')
    metadata = copy.deepcopy(original)
    metadata['training_objective']['kind'] = PAPER_MINIMAL_MARGIN_OBJECTIVE_KIND
    _rewrite_evidence(image.parent, metadata)
    ready = json.loads((image.parent / 'ready.json').read_text(encoding='utf-8'))
    ready['training_objective_kind'] = PAPER_MINIMAL_MARGIN_OBJECTIVE_KIND
    _write(image.parent / 'ready.json', ready)
    with pytest.raises(ValueError, match='版本|配置|机制'):
        validate_ready(image.parent, require_production=False)
