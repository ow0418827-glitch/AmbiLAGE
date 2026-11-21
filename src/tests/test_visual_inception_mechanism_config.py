"""Version interpretation and validation without model or evaluation access."""
from dataclasses import replace
import json

import pytest

from src.paths import PROJECT
from src.visual._visual_inception.config import VisualInceptionConfig


def test_new_defaults_and_partial_overrides_enable_paper_minimal_margin():
    for raw in (None, {}, {'training_manifest': 'explicit-public-fixture.json'}):
        config = VisualInceptionConfig.from_dict(raw)
        assert config.mechanism_profile == 'paper_minimal_margin'
        assert config.contrastive_margin == .3
        assert config.negative_same_category_count == 10
        assert config.negative_popularity_count == 10
        assert config.negative_proxy_memory_count == 10
        assert config.query_noise_std == .03
        assert config.query_noise_fraction == .25


@pytest.mark.parametrize('mode', ['semantic_perceptual', 'retrieval_semantic_perceptual'])
def test_historical_objective_configs_keep_legacy_identity(mode):
    assert VisualInceptionConfig.from_dict({'objective_mode': mode}).mechanism_profile == 'legacy_v2'
    explicit = VisualInceptionConfig.from_dict({'objective_mode': mode, 'mechanism_profile': 'paper_minimal_margin'})
    assert explicit.mechanism_profile == 'paper_minimal_margin'


@pytest.mark.parametrize('removed_key', [
    'hard_negative_count',
    'hard_negative_same_category_count',
    'hard_negative_proxy_memory_count',
    'hard_negative_popularity_count',
    'query_adversarial_radius',
    'query_adversarial_steps',
    'query_adversarial_step_size',
    'query_adversarial_mix',
    'bank_loss_blend',
    'hard_margin_loss_blend',
])
def test_removed_enhanced_v3_fields_are_rejected(removed_key):
    with pytest.raises(ValueError):
        VisualInceptionConfig.from_dict({removed_key: 1})


@pytest.mark.parametrize('values', [
    {'mechanism_profile': 'unknown'},
    {'mechanism_profile': 'enhanced_v3'},
    {'mechanism_profile': 'paper_minimal'},
    {'contrastive_margin': -0.1},
    {'contrastive_margin': 2.5},
    {'contrastive_margin': True},
    {'contrastive_margin': float('nan')},
    {'negative_same_category_count': -1},
    {'negative_proxy_memory_count': True},
    {'negative_popularity_count': 1.5},
    {'negative_same_category_count': 0, 'negative_proxy_memory_count': 0, 'negative_popularity_count': 0},
    {'query_noise_std': -0.1},
    {'query_noise_std': 1.1},
    {'query_noise_std': float('inf')},
    {'query_noise_fraction': -0.1},
    {'query_noise_fraction': 1.1},
])
def test_invalid_mechanism_settings_fail_before_any_model_work(values):
    with pytest.raises(ValueError):
        replace(VisualInceptionConfig(), **values).validate()


def test_example_configs_use_paper_minimal_margin_and_validate():
    for filename in ('visual_inception.example.json', 'visual_inception.paper_surrogates.example.json'):
        raw = json.loads((PROJECT / 'src/visual/config' / filename).read_text(encoding='utf-8'))
        cfg = VisualInceptionConfig.from_dict(raw)
        assert cfg.mechanism_profile == 'paper_minimal_margin'
        assert cfg.contrastive_margin == .3
        assert cfg.to_dict() == raw
