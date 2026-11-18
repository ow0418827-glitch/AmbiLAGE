"""论文代理独立选择、准备及冻结证据验证；仅公开元数据和微型输入。"""
from __future__ import annotations

import hashlib
import json

from PIL import Image
import pytest

from src.paths import VISUAL_CONFIGS
from src.visual import training_surrogates as choices
from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.evidence import validate_ready
from src.visual._visual_inception.pipeline import generate_visual_inception
from src.visual.encoders.spec import BackboneConfig
from src.visual.io import sha256_file


PAPER_PROFILE = 'visual_inception_paper'
PAPER_IDENTITIES = [
    ('ViT-L-14', 'openai', 'timm/vit_large_patch14_clip_224.openai'),
    ('ViT-SO400M-14-SigLIP-384', 'webli', 'timm/ViT-SO400M-14-SigLIP-384'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k', 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'),
]
PAPER_KEYS = {'clip_l14', 'siglip_so400m', 'clip_bigG14'}
LUCID_KEYS = {'clip_b16', 'clip_b32', 'clip_b32_laion'}
LUCID_CONFIG_SHA256 = '41bd088cfd50df5f92e033a04ab23060a2daf080811254bcf998f0f23c6876d9'


def public_definition():
    text = 'I want to buy shampoo with gentle formula and moisturizing ingredients.'
    return dict(target_feature_text=text, target_product_features=dict(category='shampoo',
        core_attributes=['gentle formula', 'moisturizing ingredients', 'fresh scent'],
        canonical_sentence=text))


@pytest.fixture
def tiny_pair(tmp_path):
    source, target = tmp_path / 'source.png', tmp_path / 'target.png'
    Image.new('RGB', (16, 16), (160, 120, 90)).save(source)
    Image.new('RGB', (16, 16), (80, 150, 180)).save(target)
    return source, target


def test_paper_default_is_independent_and_lucid_configuration_bytes_are_preserved():
    assert VisualInceptionConfig().surrogate_set == 'clip_l14'
    assert choices.select_training_surrogates().profile == 'lucid'
    assert hashlib.sha256((VISUAL_CONFIGS / 'optimization.json').read_bytes()).hexdigest() == LUCID_CONFIG_SHA256
    single = choices.selection_from_config(VisualInceptionConfig())
    assert len(single.specs) == 1 and choices.spec_identity(single.specs[0]) == PAPER_IDENTITIES[0]
    paper = choices.selection_from_config(VisualInceptionConfig(surrogate_set=PAPER_PROFILE))
    assert [choices.spec_identity(s) for s in paper.specs] == PAPER_IDENTITIES
    lucid = choices.select_training_surrogates('lucid')
    assert [(s.name, s.pretrained) for s in lucid.specs] == [
        ('ViT-B-16', 'openai'), ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k')]
    assert paper.audit()['specs_sha256'] != lucid.audit()['specs_sha256']


def test_paper_profile_freezes_order_paths_and_pretrained_identity():
    selected = choices.select_training_surrogates(PAPER_PROFILE)
    audit = selected.audit()
    assert audit['profile'] == PAPER_PROFILE and audit['model_count'] == 3
    assert audit['downloads_during_training'] is False
    assert selected.specs[2].weight_path == 'clip_bigG14/open_clip_model.safetensors'
    assert selected.specs[1].tokenizer_path == 'siglip_so400m'
    restored = choices.validate_selection_audit(audit, VisualInceptionConfig(surrogate_set=PAPER_PROFILE).to_dict())
    assert tuple(map(choices.spec_identity, restored.specs)) == tuple(PAPER_IDENTITIES)
    bad = json.loads(json.dumps(audit))
    bad['specs'].reverse()
    bad['specs_sha256'] = choices._digest(bad['specs'])
    with pytest.raises(ValueError, match='改变|冒充'):
        choices.validate_selection_audit(bad, VisualInceptionConfig(surrogate_set=PAPER_PROFILE).to_dict())


@pytest.mark.parametrize('held_out', [
    BackboneConfig('ViT-g-14', 'laion2b_s34b_b88k', 'laion/CLIP-ViT-g-14-laion2B-s34B-b88K'),
    BackboneConfig('ViT-G-14', 'laion2b_s34b_b88k', 'laion/CLIP-ViT-g-14-laion2B-s34B-b88K'),
    BackboneConfig('ViT-bigG-14', 'laion2b_s39b_b160k', 'laion/CLIP-ViT-g-14-laion2B-s34B-b88K'),
    BackboneConfig('ViT-bigG-14', 'laion2b_s39b_b160k', PAPER_IDENTITIES[2][2],
                   weight_path='eval_g14/open_clip_model.safetensors'),
])
def test_giant_training_identity_does_not_let_held_out_encoder_enter_training(held_out):
    selected = choices.select_training_surrogates(PAPER_PROFILE)
    choices.validate_training_specs(selected.specs)
    with pytest.raises(ValueError, match='留出'):
        choices.validate_training_specs([held_out])


@pytest.mark.parametrize('profile,expected', [(PAPER_PROFILE, PAPER_KEYS), ('lucid', LUCID_KEYS)])
def test_preparation_plan_contains_only_the_selected_training_encoders(profile, expected, capsys):
    from tools.prepare_models import main
    assert main(['--scope', 'encoders', '--surrogate-set', profile]) == 0
    report = json.loads(capsys.readouterr().out)
    assert {row['key'] for row in report['models']} == expected
    assert report['training_surrogates']['profile'] == profile
    assert report['download'] is False


@pytest.mark.parametrize('profile,expected', [(PAPER_PROFILE, PAPER_KEYS), ('lucid', LUCID_KEYS)])
def test_preflight_model_requirements_follow_selected_profile(profile, expected, tmp_path, monkeypatch, capsys):
    from tools import preflight
    from src.model_registry import model_specs
    registry = {key: spec for key, spec in model_specs().items()
                if key in PAPER_KEYS | LUCID_KEYS}
    monkeypatch.setattr(preflight, 'MODEL_ROOT', tmp_path / 'absent_models')
    monkeypatch.setattr(preflight, 'RESOURCES', tmp_path / 'resources')
    monkeypatch.setattr(preflight, 'model_specs', lambda: registry)
    monkeypatch.setattr(preflight, 'check_code', lambda: {})
    monkeypatch.setattr(preflight, 'inspect_runtime', lambda include_hardware: {})
    monkeypatch.setattr(preflight, 'check_resources', lambda: pytest.fail('代码预检不得读取正式资源'))
    assert preflight.main(['--code-only', '--surrogate-set', profile]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report['models']) == expected
    assert report['training_surrogates']['profile'] == profile
    assert report['remote_model_execution_verified'] is False


def test_runtime_check_marks_paper_encoders_required_and_lucid_encoders_optional(tmp_path, monkeypatch, capsys):
    from src.experiments import runtime_check
    from src.model_registry import model_specs
    import torch
    registry = {key: spec for key, spec in model_specs().items() if key in PAPER_KEYS | LUCID_KEYS}
    monkeypatch.setattr(runtime_check, 'MODEL_ROOT', tmp_path / 'absent_models')
    monkeypatch.setattr(runtime_check, 'model_specs', lambda: registry)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    report = runtime_check.doctor('cpu', surrogate_set=PAPER_PROFILE)
    capsys.readouterr()
    assert {key for key, row in report['local_models'].items() if row['required']} == PAPER_KEYS
    assert all(row['loaded'] is False for row in report['local_models'].values())
    assert report['real_model_execution_verified'] is False and report['external_model_calls'] == 0


def test_explicit_previous_profile_keeps_frozen_identity_and_cannot_be_relabelled(tiny_pair, tmp_path):
    cfg = VisualInceptionConfig(objective_mode='semantic_perceptual', steps=1, gradient_restarts=2, smoke_test=True, surrogate_set='lucid')
    output, metadata = generate_visual_inception(*tiny_pair, tmp_path / 'old_profile', config=cfg,
        device='cpu', definition=public_definition())
    frozen = validate_ready(output.parent, require_production=False)
    assert frozen['config']['surrogate_set'] == 'lucid'
    assert frozen['surrogate_selection']['profile'] == 'lucid'
    assert [(row['name'], row['pretrained']) for row in frozen['surrogate_selection']['specs']] == [
        ('ViT-B-16', 'openai'), ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k')]
    paper = choices.select_training_surrogates(PAPER_PROFILE).audit()
    metadata['surrogate_selection'] = paper
    metadata['training_objective']['surrogate_selection'] = paper
    for name in ('metadata.json', 'attack_metrics.json'):
        (output.parent / name).write_text(json.dumps(metadata), encoding='utf-8')
    ready_path = output.parent / 'ready.json'
    ready = json.loads(ready_path.read_text(encoding='utf-8'))
    ready.update(metadata_sha256=sha256_file(output.parent / 'metadata.json'),
        attack_metrics_sha256=sha256_file(output.parent / 'attack_metrics.json'))
    ready_path.write_text(json.dumps(ready), encoding='utf-8')
    with pytest.raises(ValueError, match='配置不一致|代理|论文'):
        validate_ready(output.parent, require_production=False)


def test_changing_proxy_profile_never_overwrites_existing_generated_image(tiny_pair, tmp_path):
    previous = VisualInceptionConfig(objective_mode='semantic_perceptual', steps=1, gradient_restarts=2, smoke_test=True, surrogate_set='lucid')
    root = tmp_path / 'frozen_before_switch'
    output, _ = generate_visual_inception(*tiny_pair, root, config=previous,
        device='cpu', definition=public_definition())
    before = {name: (root / name).read_bytes() for name in ('adversarial.png', 'metadata.json', 'ready.json')}
    current = VisualInceptionConfig(objective_mode='semantic_perceptual', steps=1, gradient_restarts=2, smoke_test=True)
    assert current.surrogate_set == 'clip_l14'
    with pytest.raises(FileExistsError, match='禁止覆盖'):
        generate_visual_inception(*tiny_pair, root, config=current, device='cpu', definition=public_definition())
    assert output.is_file()
    assert before == {name: (root / name).read_bytes() for name in before}


@pytest.mark.parametrize('arguments,expected', [
    ([], 'clip_l14'),
    (['--method', 'ambilage'], 'clip_l14'),
    (['--method', 'ambilage', '--surrogate-set', 'lucid'], 'lucid'),
    (['--method', 'lucid'], 'lucid'),
    (['--method', 'visual_inception'], 'clip_l14'),
    (['--method', 'visual_inception', '--surrogate-set', 'lucid'], 'lucid'),
])
def test_full_pipeline_doctor_selects_method_default_without_reading_run_inputs(arguments, expected, monkeypatch):
    from src.experiments import pipeline, runtime_check
    seen = []
    monkeypatch.setattr(runtime_check, 'doctor', lambda device, **kwargs: seen.append((device, kwargs)))
    monkeypatch.setattr(pipeline, 'load_config', lambda *_: pytest.fail('诊断不得读取运行清单或正式输入'))
    assert pipeline.main(['doctor', '--device', 'cpu', *arguments]) == 0
    assert seen == [('cpu', dict(surrogate_set=expected, surrogate_config=None))]
