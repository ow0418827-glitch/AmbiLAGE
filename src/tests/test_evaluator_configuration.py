"""留出配置解析只使用模拟构造器，不加载模型。"""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from src.visual.audit import save_json
from src.visual.image_utils import sha256_file


def test_g14_does_not_require_training_weights(tmp_path, monkeypatch):
    import src.visual.encoders.clip as clip
    import src.visual.encoders.joint as joint
    from src.experiments.shared_evaluation import make_eval_encoder
    weight = tmp_path / 'open_clip_pytorch_model.safetensors'
    weight.write_bytes(b'explicit_mock_weight_bytes')
    settings = tmp_path / 'evaluator.json'
    save_json(settings, dict(setting='held_out_public_encoder', surrogate=dict(
        name='ViT-g-14', pretrained='laion2b_s12b_b42k', weight_path=weight.name)))
    received = []

    def constructor(config, root):
        received.append(config)
        return SimpleNamespace(snapshot={'explicit_test_fixture': True})

    monkeypatch.setattr(clip, 'MatchedClipEncoder', constructor)
    monkeypatch.setattr(joint, 'ChunkedClipEncoder', lambda base: base)
    encoder = make_eval_encoder('cpu', settings)
    assert received[0]['surrogate']['weight_path'] == str(weight.resolve())
    assert encoder.snapshot['evaluation_protocol']['black_box_transfer_claimed']


@pytest.mark.parametrize('name,pretrained', [('ViT-B-16', 'openai'), ('ViT-L-14', 'openai'),
    ('ViT-SO400M-14-SigLIP-384', 'webli'), ('ViT-H-14', 'laion2b_s32b_b79k'),
    ('ViT-bigG-14', 'laion2b_s39b_b160k')])
def test_any_training_proxy_cannot_be_named_held_out(tmp_path, name, pretrained):
    from src.experiments.shared_evaluation import make_eval_encoder
    settings = tmp_path / 'encoder.json'
    save_json(settings, dict(setting='held_out_public_encoder', surrogate=dict(name=name, pretrained=pretrained)))
    with pytest.raises(ValueError, match='训练代理相同'):
        make_eval_encoder('cpu', settings)


def test_gme_uses_separate_text_image_equal_fusion(tmp_path, monkeypatch):
    import src.visual.encoders.joint as joint
    from src.experiments.shared_evaluation import make_eval_encoder
    class DummyGME:
        snapshot = {'provider': 'gme', 'explicit_test_fixture': True}
        def __init__(self, **kwargs):
            assert kwargs['model'] == str(tmp_path)
        def encode(self, path): return np.array([1., 0.])
        def encode_text(self, text): return np.array([0., 1.])
        def close(self): pass
    monkeypatch.setattr(joint, 'GMEProcessEncoder', DummyGME)
    settings = tmp_path / 'gme.json'
    save_json(settings, dict(setting='held_out_public_encoder', provider='gme', model_path=str(tmp_path)))
    encoder = make_eval_encoder('cpu', settings)
    assert np.allclose(encoder.encode_joint('完整文字', tmp_path / 'image.png'), [2**-0.5, 2**-0.5])
    assert encoder.supports_image_weight


def test_evaluation_numeric_mode_does_not_inherit_training_tf32(tmp_path, monkeypatch):
    import src.visual.encoders.clip as clip
    import src.visual.encoders.joint as joint
    import src.visual.hardware as hardware
    from src.experiments.shared_evaluation import make_eval_encoder
    calls = []
    monkeypatch.setattr(hardware, 'configure_tf32', lambda value: calls.append(value))
    monkeypatch.setattr(clip, 'MatchedClipEncoder', lambda *a: SimpleNamespace(snapshot={}))
    monkeypatch.setattr(joint, 'ChunkedClipEncoder', lambda value: value)
    weight = tmp_path / 'mock.safetensors'
    weight.write_bytes(b'explicit_mock_weight')
    path = tmp_path / 'encoder.json'
    save_json(path, dict(setting='held_out_public_encoder', surrogate=dict(
        name='ViT-g-14', pretrained='laion2b_s12b_b42k', weight_path=str(weight))))
    result = make_eval_encoder('cuda', path)
    assert calls == [False]
    assert result.snapshot['evaluation_protocol']['tf32_enabled'] is False


@pytest.fixture
def mocked_clip(monkeypatch):
    import src.visual.encoders.clip as clip
    import src.visual.encoders.joint as joint
    calls = []
    def constructor(config, root):
        calls.append(config)
        return SimpleNamespace(snapshot={})
    monkeypatch.setattr(clip, 'MatchedClipEncoder', constructor)
    monkeypatch.setattr(joint, 'ChunkedClipEncoder', lambda base: base)
    return calls


def _frozen_proxy(name='custom-public-proxy', pretrained='public-v1', weight_sha256='a' * 64):
    return dict(name=name, pretrained=pretrained, hf_repo='public/independent-model',
                weight_sha256=weight_sha256)


def _encoder_config(tmp_path, setting, name, pretrained, data=b'explicit_fixture_weight_bytes'):
    weight = tmp_path / 'fixture.safetensors'
    weight.write_bytes(data)
    settings = tmp_path / 'evaluator.json'
    save_json(settings, dict(setting=setting, surrogate=dict(
        name=name, pretrained=pretrained, weight_path=str(weight))))
    return settings, weight


def test_frozen_custom_proxy_is_blocked_before_loading_weights(tmp_path):
    from src.experiments.shared_evaluation import make_eval_encoder
    settings = tmp_path / 'evaluator.json'
    save_json(settings, dict(setting='held_out_public_encoder', surrogate=dict(
        name='custom-public-proxy', pretrained='public-v1')))
    with pytest.raises(ValueError, match='训练代理相同'):
        make_eval_encoder('cpu', settings, training_surrogates=[_frozen_proxy()])


def test_actual_context_does_not_mislabel_an_unused_builtin_as_matched(tmp_path):
    from src.experiments.shared_evaluation import make_eval_encoder
    settings, _ = _encoder_config(tmp_path, 'matched_public_encoder', 'ViT-L-14', 'openai')
    with pytest.raises(ValueError, match='实际训练代理身份不符'):
        make_eval_encoder('cpu', settings, training_surrogates=[_frozen_proxy()])


def test_matched_custom_encoder_requires_actual_frozen_weight_hash(tmp_path, mocked_clip):
    from src.experiments.shared_evaluation import make_eval_encoder
    settings, weight = _encoder_config(tmp_path, 'matched_public_encoder', 'custom-public-proxy', 'public-v1')
    actual = [_frozen_proxy(weight_sha256=sha256_file(weight))]
    result = make_eval_encoder('cpu', settings, training_surrogates=actual)
    protocol = result.snapshot['evaluation_protocol']
    assert protocol['matches_training_surrogate'] is True
    assert protocol['black_box_transfer_claimed'] is False
    assert protocol['training_identity_scope'] == 'actual_frozen_surrogates'
    assert len(mocked_clip) == 1
    with pytest.raises(ValueError, match='实际权重与训练代理不符'):
        make_eval_encoder('cpu', settings, training_surrogates=[_frozen_proxy()])
    assert len(mocked_clip) == 1


def test_renamed_training_weight_cannot_claim_held_out(tmp_path, mocked_clip):
    from src.experiments.shared_evaluation import make_eval_encoder
    settings, weight = _encoder_config(tmp_path, 'held_out_public_encoder', 'renamed-proxy', 'alias-v1')
    with pytest.raises(ValueError, match='相同权重'):
        make_eval_encoder('cpu', settings,
                          training_surrogates=[_frozen_proxy(weight_sha256=sha256_file(weight))])
    assert mocked_clip == []


def test_big_g_training_and_lowercase_g_holdout_remain_distinct(tmp_path, mocked_clip):
    from src.experiments.shared_evaluation import make_eval_encoder
    settings, _ = _encoder_config(tmp_path, 'held_out_public_encoder', 'ViT-g-14', 'laion2b_s12b_b42k')
    actual = [_frozen_proxy('ViT-bigG-14', 'laion2b_s39b_b160k')]
    result = make_eval_encoder('cpu', settings, training_surrogates=actual)
    assert result.snapshot['evaluation_protocol']['black_box_transfer_claimed'] is True


def test_default_isolation_reads_the_current_builtin_selections(tmp_path, monkeypatch):
    import src.visual.training_surrogates as selection
    from src.experiments.shared_evaluation import make_eval_encoder
    original = selection.select_training_surrogates
    def updated(profile, *args, **kwargs):
        if profile == 'heterogeneous4':
            return SimpleNamespace(source=tmp_path / 'public-proxy-config.json',
                specs=[SimpleNamespace(name='new-public-proxy', pretrained='new-v1',
                    hf_repo='public/new-model', weight_path=None)])
        return original(profile, *args, **kwargs)
    monkeypatch.setattr(selection, 'select_training_surrogates', updated)
    settings = tmp_path / 'evaluator.json'
    save_json(settings, dict(setting='held_out_public_encoder', surrogate=dict(
        name='new-public-proxy', pretrained='new-v1')))
    with pytest.raises(ValueError, match='训练代理相同'):
        make_eval_encoder('cpu', settings)


def test_multireplacement_actual_identity_is_frozen_and_revalidated(tmp_path, monkeypatch):
    """Reuse the three-image fixture, with two distinct actual training records."""
    import copy
    from src.experiments import shared_evaluation as evaluation
    from src.tests.test_evaluation_integrity import micro_run
    from src.visual.audit import read_json
    original = evaluation.evaluate
    first_proxy, second_proxy = _frozen_proxy(), _frozen_proxy('second-public-proxy', weight_sha256='b' * 64)
    def run_with_actual_records(frozen, *args, **kwargs):
        frozen = copy.deepcopy(frozen)
        frozen['metadata']['surrogates'] = [first_proxy]
        second = copy.deepcopy(frozen)
        second['source']['id'] = 'P1'
        second['source']['title'] = '公开商品1'
        second['source']['image'] = str(tmp_path / 'item1.png')
        second['source']['image_sha256'] = sha256_file(tmp_path / 'item1.png')
        second['source_memory_id'] = 'm1'
        second['metadata'].pop('surrogates')
        second['metadata']['training_objective'] = dict(surrogates=[second_proxy])
        return original(frozen, *args, replacement_frozens=[frozen, second], **kwargs)
    monkeypatch.setattr(evaluation, 'evaluate', run_with_actual_records)
    output, result, _ = micro_run.__wrapped__(tmp_path, monkeypatch)
    plan = read_json(output / 'evaluation_plan.json')
    context = plan['training_surrogate_context']
    assert plan['training_identity_policy'] == 'actual_frozen_surrogates_v1'
    assert len(context['images']) == 2
    assert evaluation._context_training_identities(context) == [first_proxy, second_proxy]
    assert result['verification']['status'] == 'passed'
    override = tmp_path / 'different-evaluator.json'
    save_json(override, {'setting': 'held_out_public_encoder'})
    with pytest.raises(ValueError, match='编码器配置与冻结评估不一致'):
        evaluation.verify(output, encoder_factory=lambda: None, encoder_config=override)
    shortened = copy.deepcopy(plan)
    shortened['training_surrogate_context']['images'].pop()
    body = {k: v for k, v in shortened['training_surrogate_context'].items() if k != 'context_sha256'}
    shortened['training_surrogate_context']['context_sha256'] = evaluation.digest(body)
    with pytest.raises(ValueError, match='未覆盖本次全部冻结替换图片'):
        evaluation._plan_training_identities(shortened)
    plan['training_surrogate_context']['images'][1]['surrogates'][0]['weight_sha256'] = 'c' * 64
    save_json(output / 'evaluation_plan.json', plan)
    with pytest.raises(ValueError, match='训练代理上下文或摘要不一致'):
        evaluation.verify(output, encoder_factory=lambda: None)
