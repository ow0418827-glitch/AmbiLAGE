"""留出配置解析只使用模拟构造器，不加载模型。"""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from src.visual.audit import save_json


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
    ('ViT-SO400M-14-SigLIP-384', 'webli'), ('ViT-H-14', 'laion2b_s32b_b79k')])
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
