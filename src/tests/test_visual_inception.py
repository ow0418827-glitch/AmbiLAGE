"""视觉记忆植入基线的公开目标、真实梯度与微型端到端验证；不装载真实权重。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.evidence import validate_ready
from src.visual._visual_inception.objective import inverse_variance_weights, semantic_loss
from src.visual._visual_inception.pipeline import generate_visual_inception
from src.visual._visual_inception.targets import prepare_semantic_targets
from src.visual.image_utils import sha256_file


def public_definition():
    return {
        'target_feature_text': 'I want to buy shampoo with gentle formula and moisturizing ingredients.',
        'target_product_features': {
            'category': 'shampoo',
            'core_attributes': ['gentle formula', 'moisturizing ingredients', 'fresh scent'],
            'canonical_sentence': 'I want to buy shampoo with gentle formula and moisturizing ingredients.',
        },
    }


@pytest.fixture
def tiny_pair(tmp_path):
    source, target = tmp_path / 'source.png', tmp_path / 'target.png'
    # 非恒定输入能暴露通道交换、错误广播和落盘预算问题。
    yy, xx = np.mgrid[:16, :16]
    rgb = np.stack([70 + xx * 5, 100 + yy * 3, 150 - xx * 2], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(source)
    Image.new('RGB', (16, 16), (70, 130, 180)).save(target)
    return source, target


def run_tiny(tiny_pair, output, **kwargs):
    source, target = tiny_pair
    options = dict(steps=2, device='cpu', smoke_test=True, definition=public_definition(),
        config=VisualInceptionConfig(objective_mode='semantic_perceptual'))
    options.update(kwargs)
    return generate_visual_inception(source, target, output, **options)


def test_default_configuration_is_the_declared_retrieval_baseline():
    cfg = VisualInceptionConfig()
    assert cfg.steps == 100
    assert cfg.epsilon_levels == 8
    assert cfg.step_size_levels == 2
    assert cfg.semantic_weight == 1 and cfg.perceptual_weight == .1
    assert cfg.gradient_restarts == 10
    assert cfg.seed == 42 and cfg.surrogate_set == 'visual_inception_paper'
    assert cfg.smoke_test is False
    assert cfg.objective_mode == 'retrieval_semantic_perceptual'
    assert cfg.retrieval_weight == 1 and cfg.retrieval_temperature == .07
    assert cfg.query_batch_size == 16 and cfg.training_manifest is None
    cfg.validate()


@pytest.mark.parametrize('values', [
    {'steps': 0}, {'steps': -1}, {'steps': True},
    {'epsilon_levels': 0}, {'epsilon_levels': 256},
    {'gradient_restarts': 0}, {'gradient_restarts': True},
    {'step_size_levels': float('nan')}, {'perceptual_weight': 0},
    {'semantic_weight': float('inf')}, {'seed': -1},
    {'surrogate_set': 'missing_profile'},
])
def test_invalid_configuration_fails_before_model_loading(values):
    with pytest.raises((ValueError, TypeError)):
        VisualInceptionConfig(**values).validate()


class ColorSurrogate:
    device = torch.device('cpu')

    def encode_image(self, image):
        return SimpleNamespace(global_embedding=image.mean(dim=(-2, -1)))


def test_public_templates_are_rebuilt_from_frozen_features(tiny_pair):
    definition = public_definition()
    definition['training_queries'] = ['DO NOT OPTIMIZE THIS EMBEDDED QUERY POOL']
    result = prepare_semantic_targets(*tiny_pair, definition=definition)
    assert result['semantic_texts'] == [
        'I want to buy shampoo with gentle formula and moisturizing ingredients.',
        'I would like to purchase shampoo featuring gentle formula and fresh scent.',
        'Looking to buy shampoo with moisturizing ingredients and fresh scent.',
    ]
    assert result['canonical_text'] == definition['target_feature_text']
    assert result['retrieval_loss_omitted'] is True
    assert result['paper_query_distribution_reproduced'] is False
    assert len(result['semantic_text_sha256']) == 3


@pytest.mark.parametrize('change', ['missing_attributes', 'inconsistent_text', 'wrong_target_image'])
def test_public_target_identity_and_frozen_text_must_agree(tiny_pair, change):
    definition = public_definition()
    if change == 'missing_attributes':
        definition['target_product_features'].pop('core_attributes')
    elif change == 'inconsistent_text':
        definition['target_feature_text'] = 'I want to buy a different public product.'
    else:
        definition['target'] = {'image_sha256': '0' * 64}
    with pytest.raises(ValueError):
        prepare_semantic_targets(*tiny_pair, definition=definition)


def test_semantic_loss_averages_all_templates_and_descends_in_image_space():
    surrogate = ColorSurrogate()
    image = torch.tensor([.7, .2, .4]).reshape(1, 3, 1, 1).expand(1, 3, 4, 4).clone().requires_grad_(True)
    texts = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    loss = semantic_loss(surrogate, image, texts)
    expected = (1 - F.normalize(image.mean((-2, -1)), dim=-1) @ F.normalize(texts, dim=-1).T).mean()
    assert torch.allclose(loss, expected, atol=1e-7)
    gradient, = torch.autograd.grad(loss, image)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    next_image = (image.detach() - .01 * gradient.sign()).clamp(0, 1)
    assert semantic_loss(surrogate, next_image, texts) < loss.detach()


def test_variance_weighting_uses_inverse_square_and_a_finite_floor():
    got = torch.as_tensor(inverse_variance_weights([.5, 2.]))
    assert torch.allclose(got, torch.tensor([16 / 17, 1 / 17]), atol=1e-6)
    equal = torch.as_tensor(inverse_variance_weights([0., 0., 0.]))
    assert torch.isfinite(equal).all()
    assert torch.allclose(equal, torch.full((3,), 1 / 3), atol=1e-6)


def test_variance_weights_are_invariant_to_gradient_scale():
    original = torch.as_tensor(inverse_variance_weights([.5, 2., 4.]), dtype=torch.float64)
    rescaled = torch.as_tensor(inverse_variance_weights([.5e-20, 2e-20, 4e-20]), dtype=torch.float64)
    assert torch.allclose(original, rescaled, atol=1e-12, rtol=1e-12)


def test_frozen_capitalization_is_preserved_without_changing_template_semantics(tiny_pair):
    definition = {
        'target_feature_text': 'I want to buy a Bluetooth speaker with water resistance and stereo sound.',
        'target_product_features': {
            'category': 'Bluetooth speaker',
            'core_attributes': ['water resistance', 'stereo sound', 'portable design'],
            'canonical_sentence': 'I want to buy a Bluetooth speaker with water resistance and stereo sound.',
        },
    }
    result = prepare_semantic_targets(*tiny_pair, definition=definition)
    assert result['canonical_text'] == definition['target_feature_text']
    assert result['semantic_texts'][0].casefold() == result['canonical_text'].casefold()
    assert len(result['semantic_texts']) == 3


def test_sequential_gradient_equals_joint_semantic_and_perceptual_gradient():
    from src.visual._visual_inception.pipeline import _gradient
    image = torch.tensor([.7, .2, .4]).reshape(1, 3, 1, 1).expand(1, 3, 4, 4).clone()
    source = image * .8
    targets = [torch.eye(3), torch.tensor([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])]
    weights = [.25, .75]
    config = VisualInceptionConfig(objective_mode='semantic_perceptual', semantic_weight=1.7, perceptual_weight=.3)
    distance = lambda origin, changed: (origin - changed).square().mean()
    gradient, record = _gradient(image, source, [ColorSurrogate(), ColorSurrogate()],
        targets, weights, distance, config)
    joint_image = image.detach().requires_grad_(True)
    normalized = F.normalize(joint_image.mean((-2, -1)), dim=-1)
    semantic_parts = [(1 - normalized @ F.normalize(text, dim=-1).T).mean() for text in targets]
    semantic = weights[0] * semantic_parts[0] + weights[1] * semantic_parts[1]
    perceptual = distance(source, joint_image)
    joint_loss = config.semantic_weight * semantic + config.perceptual_weight * perceptual
    expected_gradient, = torch.autograd.grad(joint_loss, joint_image)
    assert torch.allclose(gradient, expected_gradient, atol=1e-7)
    assert record['semantic_loss'] == pytest.approx(float(semantic.detach()))
    assert record['perceptual_loss'] == pytest.approx(float(perceptual.detach()))
    assert record['loss'] == pytest.approx(float(joint_loss.detach()))


def test_semantic_targets_require_public_provenance_instead_of_formal_queries(tiny_pair):
    source, target = tiny_pair
    with pytest.raises(ValueError):
        prepare_semantic_targets(source, target,
            definition={'training_queries': ['Looking to buy a private target product.']})


def test_saved_perturbation_is_nonzero_and_respects_source_pixel_budget(tiny_pair, tmp_path):
    image, metadata = run_tiny(tiny_pair, tmp_path / 'out')
    original = np.asarray(Image.open(tiny_pair[0])).astype(np.int16)
    edited = np.asarray(Image.open(image)).astype(np.int16)
    assert original.shape == edited.shape
    difference = np.abs(edited - original)
    assert 0 < difference.max() <= 8
    assert metadata['method'] == 'visual_inception'
    assert metadata['test_fixture'] is True and metadata['production_backend'] is False
    assert metadata['paper_encoder_profile_selected'] is True
    assert metadata['paper_encoder_architectures_aligned'] is False
    assert metadata['paper_encoder_ensemble_reproduced'] is False
    assert metadata['paper_encoder_checkpoint_status'] == 'public_checkpoint_choices_paper_exact_revisions_unspecified'
    assert metadata['training_objective']['kind'] == 'visual_inception_semantic_perceptual_no_retrieval_v1'
    assert metadata['output_image_sha256'] == sha256_file(image)
    assert validate_ready(image.parent, require_production=False)['method'] == 'visual_inception'
    with pytest.raises(ValueError):
        validate_ready(image.parent)


def test_training_guard_allows_public_baseline_and_blocks_private_queries(tiny_pair, tmp_path):
    from src.visual.catalog_training_guard import catalog_training_guard
    private = tmp_path / 'private_queries.json'
    private.write_text('{"instructions":["forbidden formal query"]}', encoding='utf-8')
    with catalog_training_guard(allowed_output_roots=[tmp_path], private_paths=[private]) as audit:
        run_tiny(tiny_pair, tmp_path / 'public_run', steps=1)
        assert audit['blocked_accesses'] == []
        with pytest.raises(PermissionError):
            private.read_text(encoding='utf-8')
    assert audit['blocked_accesses'] == [str(private.resolve())]


def test_same_seed_produces_identical_pixels_and_fixed_semantic_targets(tiny_pair, tmp_path):
    first, first_meta = run_tiny(tiny_pair, tmp_path / 'first', seed=87)
    second, second_meta = run_tiny(tiny_pair, tmp_path / 'second', seed=87)
    assert sha256_file(first) == sha256_file(second)
    assert first_meta['training_objective'] == second_meta['training_objective']


@pytest.mark.parametrize('options', [{'steps': 4}, {'device': 'cuda'}])
def test_smoke_mode_rejects_non_micro_workloads(tiny_pair, tmp_path, options):
    with pytest.raises((ValueError, RuntimeError)):
        run_tiny(tiny_pair, tmp_path / 'out', **options)
    assert not (tmp_path / 'out/ready.json').exists()


def test_smoke_mode_rejects_images_larger_than_64(tiny_pair, tmp_path):
    Image.new('RGB', (65, 16), (110, 140, 170)).save(tiny_pair[0])
    with pytest.raises(ValueError):
        run_tiny(tiny_pair, tmp_path / 'out')
    assert not (tmp_path / 'out/ready.json').exists()


def test_fixture_cannot_be_disguised_as_production(tiny_pair, tmp_path):
    from src.visual._visual_inception.targets import content_sha256
    image, _ = run_tiny(tiny_pair, tmp_path / 'out')
    root = image.parent
    metadata = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    metadata['config']['smoke_test'] = False
    metadata['config_sha256'] = content_sha256(metadata['config'])
    metadata['test_fixture'] = False
    metadata['production_backend'] = True
    for filename in ('metadata.json', 'attack_metrics.json'):
        (root / filename).write_text(json.dumps(metadata), encoding='utf-8')
    ready = json.loads((root / 'ready.json').read_text(encoding='utf-8'))
    ready.update(test_fixture=False, production_backend=True, config_sha256=metadata['config_sha256'],
        metadata_sha256=sha256_file(root / 'metadata.json'),
        attack_metrics_sha256=sha256_file(root / 'attack_metrics.json'))
    (root / 'ready.json').write_text(json.dumps(ready), encoding='utf-8')
    with pytest.raises(ValueError, match='代理|预训练感知'):
        validate_ready(image.parent)


def test_completed_output_is_not_overwritten(tiny_pair, tmp_path):
    image, _ = run_tiny(tiny_pair, tmp_path / 'out')
    image_before = image.read_bytes()
    ready_before = (image.parent / 'ready.json').read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        run_tiny(tiny_pair, image.parent, seed=99)
    assert image.read_bytes() == image_before
    assert (image.parent / 'ready.json').read_bytes() == ready_before


def test_evidence_rejects_output_pixel_tampering(tiny_pair, tmp_path):
    image, _ = run_tiny(tiny_pair, tmp_path / 'out')
    Image.new('RGB', (16, 16), (0, 0, 0)).save(image)
    with pytest.raises(ValueError):
        validate_ready(image.parent, require_production=False)


def test_missing_real_perceptual_weights_stops_before_loading_surrogates(tiny_pair, tmp_path, monkeypatch):
    from src.visual import training_objectives
    monkeypatch.setattr(training_objectives, 'load_training_surrogates',
        lambda *args, **kwargs: pytest.fail('缺少真实感知权重时不得装载代理模型'))
    for path in tiny_pair:
        with Image.open(path) as image:
            image.resize((32, 32)).save(path)
    config = VisualInceptionConfig(objective_mode='semantic_perceptual', lpips_backbone_weights=str(tmp_path / 'missing_backbone.pth'),
        lpips_linear_weights=str(tmp_path / 'missing_linear.pth'))
    with pytest.raises(FileNotFoundError):
        generate_visual_inception(*tiny_pair, tmp_path / 'production_missing_weights',
            config=config, device='cpu', definition=public_definition())
    assert not (tmp_path / 'production_missing_weights/ready.json').exists()


@pytest.mark.parametrize('group', [
    '01_camera', '02_chair', '03_cookies', '04_curtain', '05_hair_extension',
    '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush',
])
def test_nine_group_dispatch_binds_baseline_defaults_and_real_public_category(tmp_path, monkeypatch, group):
    from src.paths import EVALUATION_BASE
    from src.experiments import base_workflow, pipeline
    from src.visual import optimization
    expected = base_workflow.public_group(EVALUATION_BASE, group)
    output = tmp_path / 'output'
    monkeypatch.setattr(base_workflow, 'OUTPUT', output)
    monkeypatch.setattr(pipeline, 'source_snapshot', lambda: {'implementation': 'explicit_micro_dispatch_fixture'})
    seen = {}

    class StoppedBeforeWeights(RuntimeError):
        pass

    def capture(source, target, destination, **kwargs):
        seen.update(kwargs)
        semantic = prepare_semantic_targets(source, target, target_text=kwargs['target_text'],
            definition=kwargs['definition'])
        assert len(semantic['semantic_texts']) == 3
        raise StoppedBeforeWeights()

    monkeypatch.setattr(optimization, 'generate_attack', capture)
    root = output / group
    training_manifest = tmp_path / 'training_bundle.json'
    training_manifest.write_text('{}', encoding='utf-8')
    with pytest.raises(StoppedBeforeWeights):
        base_workflow.generate_group(EVALUATION_BASE, group, 'visual_inception', root, device='cpu',
            visual_inception_config={'training_manifest': str(training_manifest)})
    config = seen['visual_inception_config']
    assert isinstance(config, VisualInceptionConfig)
    assert (config.steps, config.epsilon_levels, config.step_size_levels) == (100, 8, 2)
    assert config.objective_mode == 'retrieval_semantic_perceptual'
    assert config.training_manifest == str(training_manifest)
    assert 'ambilage_config' not in seen and seen['method'] == 'visual_inception'
    assert seen['target_text'] == expected['target_text']
    assert seen['definition']['target_product_features'] == expected['target_features']
    binding = json.loads((root / 'PUBLIC_RUN.json').read_text(encoding='utf-8'))
    assert binding['config']['steps'] == 100 and binding['epsilon_levels'] == 8
    assert binding['training_surrogates']['profile'] == 'visual_inception_paper'
    assert not (root / 'FROZEN_IMAGE.json').exists()


@pytest.mark.parametrize('options,expected', [
    ({}, (100, 8, 2)),
    ({'steps': 7, 'epsilon_levels': 4, 'step_size_levels': 1}, (7, 4, 1)),
])
def test_full_pipeline_uses_independent_baseline_configuration(tiny_pair, tmp_path, monkeypatch, options, expected):
    from src.experiments import pipeline
    monkeypatch.setattr(pipeline, 'source_snapshot', lambda: {'implementation': 'explicit_micro_dispatch_fixture'})
    root = tmp_path / 'full_pipeline'
    selection_dir = root / 'selection'
    selection_dir.mkdir(parents=True)
    text = public_definition()['target_feature_text']
    selection = {
        'source': {'id': 'S', 'title': 'source fixture', 'image': str(tiny_pair[0]),
            'image_sha256': sha256_file(tiny_pair[0])},
        'target': {'id': 'T', 'title': 'target fixture', 'image': str(tiny_pair[1]),
            'image_sha256': sha256_file(tiny_pair[1]), 'feature_text': text},
        'target_feature_text': text,
        'target_product_features': public_definition()['target_product_features'],
    }
    (selection_dir / 'SELECTED_PAIR.json').write_text(json.dumps(selection), encoding='utf-8')
    seen = {}

    class StoppedBeforeWeights(RuntimeError):
        pass

    def capture(source, target, destination, **kwargs):
        seen.update(kwargs)
        raise StoppedBeforeWeights()

    training_manifest = tmp_path / 'training_bundle.json'
    training_manifest.write_text('{}', encoding='utf-8')
    config = {'generation': {'visual_inception': dict(options, training_manifest=str(training_manifest))},
        'evaluation': {'query_mode': 'text'}}
    with pytest.raises(StoppedBeforeWeights):
        pipeline.generate(config, root, selection, 'visual_inception', device='cpu', generator=capture)
    actual = seen['visual_inception_config']
    assert (actual['steps'], actual['epsilon_levels'], actual['step_size_levels']) == expected
    assert seen['method'] == 'visual_inception' and seen['target_text'] == text
    assert seen['definition']['target_product_features']['category'] == 'shampoo'
    assert 'ambilage_config' not in seen and 'optimization_config' not in seen
    assert 'steps' not in seen  # 日程由独立配置传递，不能落入原始像素的一千步默认。


def test_unified_generation_preserves_method_identity(tiny_pair, tmp_path):
    from src.visual.pipeline import generate
    result = generate('visual_inception', *tiny_pair, tmp_path / 'unified', steps=1,
        device='cpu', smoke_test=True, definition=public_definition(),
        visual_inception_config={'objective_mode': 'semantic_perceptual'})
    assert result.method == 'visual_inception'
    assert result.image_path.is_file() and result.ready_path.is_file() and result.metadata_path.is_file()
    assert result.sha256 == sha256_file(result.image_path)


def test_unified_cli_runs_explicit_micro_baseline(tiny_pair, tmp_path):
    from src.paths import PROJECT
    definition_path = tmp_path / 'definition.json'
    definition_path.write_text(json.dumps(public_definition()), encoding='utf-8')
    configuration = tmp_path / 'legacy_configuration.json'
    configuration.write_text(json.dumps({'objective_mode': 'semantic_perceptual'}), encoding='utf-8')
    output = tmp_path / 'cli_output'
    result = subprocess.run([
        sys.executable, '-B', '-X', 'utf8', str(PROJECT / 'tools/generate.py'),
        '--method', 'visual_inception', '--source', str(tiny_pair[0]), '--target', str(tiny_pair[1]),
        '--definition', str(definition_path), '--output', str(output),
        '--visual-inception-config', str(configuration),
        '--device', 'cpu', '--steps', '1', '--smoke-test',
    ], cwd=PROJECT, capture_output=True, text=True, encoding='utf-8', timeout=45)
    assert result.returncode == 0, result.stderr + '\n' + result.stdout
    ready = validate_ready(output, require_production=False)
    assert ready['method'] == 'visual_inception' and ready['test_fixture'] is True
