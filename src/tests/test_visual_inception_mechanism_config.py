"""Version interpretation and validation without model or evaluation access."""
from dataclasses import replace

import pytest

from src.visual._visual_inception.config import VisualInceptionConfig


def test_new_defaults_and_partial_overrides_enable_enhanced_mechanisms():
    for raw in (None, {}, {'training_manifest': 'explicit-public-fixture.json'}):
        config = VisualInceptionConfig.from_dict(raw)
        assert config.mechanism_profile == 'enhanced_v3'
        assert config.contrastive_margin == .3
        assert config.hard_negative_same_category_count == 10
        assert config.hard_negative_popularity_count == 10
        assert config.hard_negative_proxy_memory_count == 10
        assert config.query_adversarial_radius == .2
        assert config.query_adversarial_step_size == .1


@pytest.mark.parametrize('mode', ['semantic_perceptual', 'retrieval_semantic_perceptual'])
def test_historical_objective_configs_keep_legacy_identity(mode):
    assert VisualInceptionConfig.from_dict({'objective_mode': mode}).mechanism_profile == 'legacy_v2'
    explicit = VisualInceptionConfig.from_dict({'objective_mode': mode, 'mechanism_profile': 'enhanced_v3'})
    assert explicit.mechanism_profile == 'enhanced_v3'


@pytest.mark.parametrize('values', [
    {'mechanism_profile': 'unknown'}, {'contrastive_margin': -0.1},
    {'contrastive_margin': float('nan')}, {'query_adversarial_radius': float('inf')},
    {'hard_negative_count': 0}, {'hard_negative_count': True},
    {'hard_negative_same_category_count': -1}, {'hard_negative_proxy_memory_count': True},
    {'hard_negative_popularity_count': 1.5}, {'query_adversarial_steps': 0},
    {'query_adversarial_step_size': 0}, {'query_adversarial_mix': 1.1},
])
def test_invalid_mechanism_settings_fail_before_any_model_work(values):
    with pytest.raises(ValueError):
        replace(VisualInceptionConfig(), **values).validate()
