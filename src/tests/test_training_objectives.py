"""Tiny offline checks: no downloaded weights or full-library evaluation."""
from dataclasses import dataclass
import math
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.visual.encoders.backbone import pretrained_settings, load_local_clip_model, BackboneConfig
from src.visual.encoders.joint import ChunkedClipEncoder
from src.visual.training_objectives import (
    PublicFullKeyObjective, SequentialEnsembleObjective, validate_training_specs, TRAINING_OBJECTIVE_KIND,
)
from src.visual.robust_objectives import transformed_batch
from src.visual.diffusion_purification import LatentDiffusionPurifier, PurificationConfig


@pytest.fixture(autouse=True)
def small_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class TinyTokenizer:
    context_length = 8
    sot_token_id = 62
    eot_token_id = 63
    pad_token_id = 0
    def encode(self, text):
        return [ord(c) % 60 + 1 for c in text]


class TinyTextModel:
    def encode_text(self, tokens):
        value = tokens.float()
        return torch.stack((value.mean(1), value.sin().sum(1), value.cos().sum(1)), dim=1)


class TinySurrogate:
    device = torch.device('cpu')
    tokenizer = TinyTokenizer()
    model = TinyTextModel()
    def __init__(self, index=0):
        self.index = index
    def snapshot(self):
        return {'name': 'tiny_offline_fixture_' + str(self.index), 'test_fixture': True}
    def encode_image(self, image):
        feature = image.mean((-2, -1)) + image.square().mean((-2, -1)) * (self.index + 1) * .1
        return SimpleNamespace(global_embedding=F.normalize(feature, dim=-1))


class TinyPurifier:
    config = SimpleNamespace(reverse_steps=10)
    def __init__(self):
        self.calls = []
    def snapshot(self):
        return {'kind': 'linear_offline_test_fixture'}
    def __call__(self, image, *, seed, gradient):
        self.calls.append((seed, gradient))
        return image * .8 + .1


def config(**kw):
    defaults = dict(joint_weight=1., temperature=.04, eot_enabled=False,
                    diffusion_weight=.25, seed=42,smoke_test=True,surrogate_set='lucid',surrogate_config=None)
    return SimpleNamespace(**(defaults | kw))


def test_native_activation_comes_from_tag(monkeypatch, tmp_path):
    class QuickGELU(torch.nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(1.702 * x)
    created = []
    def factory(name, pretrained, force_quick_gelu, **kw):
        created.append(force_quick_gelu)
        return torch.nn.Sequential(QuickGELU() if force_quick_gelu else torch.nn.GELU(), torch.nn.Linear(2, 2))
    monkeypatch.setitem(sys.modules, 'open_clip', SimpleNamespace(
        get_model_config=lambda _: {'quick_gelu': False}, create_model=factory))
    monkeypatch.setitem(sys.modules, 'open_clip.pretrained', SimpleNamespace(
        get_pretrained_cfg=lambda _, tag: {'quick_gelu': tag == 'openai', 'mean': (.5,)*3, 'std': (.5,)*3}))
    assert pretrained_settings('tiny', 'openai')['quick_gelu'] is True
    assert pretrained_settings('tiny', 'webli')['quick_gelu'] is False
    weight = tmp_path / 'tiny.pt'
    torch.save(factory('tiny', None, False).state_dict(), weight)
    _, audit = load_local_clip_model('tiny', 'webli', weight)
    assert created[-1] is False
    assert audit['actual_activation_modules'] == {'QuickGELU': 0, 'GELU': 1}


def test_siglip_eos_only_native_padding_and_full_coverage():
    class SigTokenizer(TinyTokenizer):
        sot_token_id = None
        eot_token_id = 1
        pad_token_id = 1
    model = TinySurrogate()
    model.tokenizer = SigTokenizer()
    captured = []
    class CapturingModel(TinyTextModel):
        def encode_text(self, tokens):
            captured.append(tokens.clone())
            return super().encode_text(tokens)
    model.model = CapturingModel()
    encoder = ChunkedClipEncoder(SimpleNamespace(surrogate=model, device=model.device, snapshot={}))
    result = encoder.encode_text('abcdefghijklmno')
    assert np.isfinite(result).all()
    assert encoder.last_encoding_audit['input_tokens'] == encoder.last_encoding_audit['encoded_tokens'] == 15
    assert encoder.last_encoding_audit['chunks'] == 3
    assert captured[0][-1, -1] == 1
    assert encoder.snapshot['text_policy']['special_tokens_per_chunk'] == 1


def test_joint_is_the_only_retrieval_target_and_matches_its_reference_gradient(monkeypatch):
    from src.visual import training_objectives
    texts = {'source public features': np.array([0., 1., 0.], dtype=np.float32),
             'target public features': np.array([0., 0., 1.], dtype=np.float32)}
    class ControlledTextEncoder:
        def __init__(self, base):
            self.last_encoding_audit = {'truncated': False}
        def encode_text(self, text):
            return texts[text]
    monkeypatch.setattr(training_objectives, 'ChunkedClipEncoder', ControlledTextEncoder)
    source = TinySurrogate()
    obj = PublicFullKeyObjective(source, torch.tensor([[1., 0., 0.]]),
                                 'source public features', 'target public features', config())
    f = torch.tensor([[.3, .7, .2], [.8, .1, .6]], requires_grad=True)
    scores = obj.scores(f)
    query = F.normalize(torch.tensor([[1., 0., 1.]]), dim=-1)
    keys = F.normalize(F.normalize(f, dim=-1) + torch.tensor([[0., 1., 0.]]), dim=-1)
    expected_scores = keys @ query.T
    expected = .5 * expected_scores.mean() - .5 * .04 * (
        torch.logsumexp(-expected_scores.flatten() / .04, dim=0) - math.log(len(f)))
    actual = obj.value(f)
    assert scores.shape == (2, 1)
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, f, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, f, retain_graph=True)[0]
    torch.testing.assert_close(gradient, expected_gradient)
    # A separate text target would contribute a different, unwanted gradient.
    text_score = keys @ torch.tensor([[0., 0., 1.]]).T
    text_gradient = torch.autograd.grad(text_score.mean(), f)[0]
    assert not torch.allclose(gradient, .5 * (expected_gradient + text_gradient))
    assert obj.audit['query_modes'] == ['multimodal_equal_weight']
    assert obj.audit['query_image_weight'] == obj.audit['query_text_weight'] == .5
    assert obj.audit['formal_queries_used'] is False
    assert obj.audit['source_text_encoding']['truncated'] is False


@pytest.mark.parametrize('eot_enabled', [False, True])
@pytest.mark.parametrize('profile,count',[('lucid',3),('heterogeneous4',4)])
def test_sequential_proxy_gradient_matches_fixed_weight_autograd(eot_enabled,profile,count):
    models, purifier = [TinySurrogate(i) for i in range(count)], TinyPurifier()
    image = torch.rand(1, 3, 16, 16) * .6 + .2
    target = torch.rand_like(image)
    ensemble = SequentialEnsembleObjective(models, target, 'own public key', 'target public features',
                                           config(eot_enabled=eot_enabled,surrogate_set=profile), purifier)
    utility, scores, gradient, metrics = ensemble.image_gradient(image, 3)
    leaf = image.clone().requires_grad_(True)
    weights = torch.tensor(metrics['model_weights'])
    values = torch.stack([ensemble._one_value(i, leaf, 3, 'full')[0] for i in range(count)])
    expected = torch.autograd.grad((values * weights).sum(), leaf)[0]
    torch.testing.assert_close(gradient, expected, atol=1e-7, rtol=1e-5)
    assert gradient.abs().sum() > 0
    assert scores.shape == (count, 1) and torch.isfinite(utility)
    assert ensemble.audit['kind'] == TRAINING_OBJECTIVE_KIND
    assert all(row['query_modes'] == ['multimodal_equal_weight'] for row in ensemble.audit['public_objectives'])
    assert {mode for _, mode in purifier.calls} == {'none', 'full'}
    assert all(seed == 45 for seed, _ in purifier.calls)
    with pytest.raises(ValueError):
        SequentialEnsembleObjective(models, target, 'source', 'target',
            config(diffusion_weight=0,surrogate_set=profile), purifier)


def test_training_steps_cover_all_eot_paths_with_gradients():
    image = torch.rand(1, 3, 16, 16, requires_grad=True)
    views = transformed_batch(image, step=2)
    assert views.shape == (4, 3, 16, 16)
    gradient = torch.autograd.grad(views.sum(), image)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_heldout_cannot_enter_declared_training_proxy_list():
    specs = [BackboneConfig(name, 'test') for name in
             ('ViT-B-16', 'ViT-L-14', 'ViT-g-14', 'ViT-H-14')]
    with pytest.raises(ValueError):
        validate_training_specs(specs)


def test_real_ddim_implementation_keeps_full_image_gradient():
    class VAE(torch.nn.Module):
        config = SimpleNamespace(scaling_factor=.5)
        def encode(self, image):
            latent = F.avg_pool2d(image, 8)
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))
        def decode(self, latent):
            return SimpleNamespace(sample=F.interpolate(latent, scale_factor=8, mode='bilinear', align_corners=False))
    class UNet(torch.nn.Module):
        dtype = torch.float32
        def forward(self, latent, timestep, encoder_hidden_states):
            return SimpleNamespace(sample=latent * .1)
    components = SimpleNamespace(vae=VAE(), unet=UNet(),
        scheduler=SimpleNamespace(config=SimpleNamespace(prediction_type='epsilon'),
                                  alphas_cumprod=torch.linspace(.999, .05, 1000)),
        text_embeddings=torch.zeros(1, 1, 1), manifest={'test_fixture': True})
    purifier = LatentDiffusionPurifier(PurificationConfig(working_size=64, reverse_steps=10,
        gradient_checkpointing=True), device='cpu', components=components)
    image = torch.full((1, 3, 16, 16), .5, requires_grad=True)
    actual = purifier(image, seed=7, gradient='full')
    expected = purifier(image, seed=7, gradient='none')
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    gradient = torch.autograd.grad(actual.mean(), image)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert not expected.requires_grad
    assert purifier.forward_noise_calls >= 20


def test_gme_text_only_worker_preserves_full_text():
    from src.visual.encoders.worker import FullTextGME
    class Processor:
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: list(text))
        def __call__(self, text, **kw):
            assert kw.get('truncation') is False
            return {'input_ids': torch.ones(1, len(text) + 510, dtype=torch.long)}
    class Base:
        snapshot = {'test_fixture': True}
        def __init__(self):
            self.model = SimpleNamespace(max_length=520, processor=Processor())
            self.parts = []
        def encode_text(self, text):
            self.model.processor(text=text)
            self.parts.append(text)
            return np.array([len(text), .5, 1.], dtype=np.float32)
    base = Base()
    encoder = FullTextGME(base)
    vector, audit = encoder.encode_text('abcdefghijklmnopqrstuvwxyz')
    assert ''.join(base.parts) == 'abcdefghijklmnopqrstuvwxyz'
    assert audit['input_characters'] == audit['encoded_characters'] == 26
    assert audit['mode'] == 'text_only' and audit['truncated'] is False
    assert np.isclose(np.linalg.norm(vector), 1)


def test_gme_process_text_op_updates_audit():
    from src.visual.encoders.joint import GMEProcessEncoder
    encoder = GMEProcessEncoder.__new__(GMEProcessEncoder)
    requests = []
    def request(value):
        requests.append(value)
        return {'vector': [1., 2., 3.], 'audit': {'truncated': False}}
    encoder._request = request
    assert np.isclose(np.linalg.norm(encoder.encode_text('完整公开特点')), 1)
    assert requests == [{'op': 'text', 'text': '完整公开特点'}]
    assert encoder.last_encoding_audit == {'truncated': False}


def test_cached_openclip_weight_accepts_official_giant_filename(tmp_path, monkeypatch):
    from src.visual.encoders import backbone
    repository = tmp_path / 'models--official--giant'
    snapshot = repository / 'snapshots' / 'revision'
    snapshot.mkdir(parents=True)
    (repository / 'refs').mkdir()
    (repository / 'refs' / 'main').write_text('revision', encoding='utf-8')
    weight = snapshot / 'open_clip_pytorch_model.safetensors'
    weight.write_bytes(b'only filename resolution; no real weight')
    monkeypatch.setattr(backbone, '_huggingface_hub_roots', lambda cache_dir=None: [tmp_path])
    assert backbone.resolve_hf_cached_weight('official/giant') == weight


def test_old_openclip_hf_wrapper_uses_backend_pad_and_eos():
    class BackendTokenizer:
        bos_token_id = None
        eos_token_id = 1
        pad_token_id = 7
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [ord(c) % 60 + 8 for c in text]
    surrogate = TinySurrogate()
    surrogate.tokenizer = SimpleNamespace(context_length=8, tokenizer=BackendTokenizer(), clean_fn=lambda x: x)
    captured = []
    class Capturing(TinyTextModel):
        def encode_text(self, tokens):
            captured.append(tokens.clone())
            return super().encode_text(tokens)
    surrogate.model = Capturing()
    adapter = ChunkedClipEncoder(SimpleNamespace(surrogate=surrogate, device=surrogate.device, snapshot={}))
    adapter.encode_text('abc')
    assert captured[0][0, 3] == 1
    assert captured[0][0, 4:].tolist() == [7, 7, 7, 7]
    assert adapter.last_encoding_audit['encoded_tokens'] == 3


def test_purifier_default_matches_prepared_model_directory():
    from pathlib import Path
    from src.paths import MODEL_ROOT
    assert Path(PurificationConfig().model_path) == MODEL_ROOT / 'purifier'


def test_purification_loader_uses_regular_safetensors_without_fp16_variant(tmp_path, monkeypatch):
    from src.visual.diffusion_components import load_diffusion_components
    calls = []
    class Module:
        def eval(self): return self
        def requires_grad_(self, value): return self
        def to(self, device): return self
        def __call__(self, ids): return (torch.ones(ids.shape[0], ids.shape[1], 3),)
        @classmethod
        def from_pretrained(cls, root, **kw):
            calls.append(dict(kw))
            return cls()
    class Tokenizer:
        model_max_length = 77
        @classmethod
        def from_pretrained(cls, root, **kw): return cls()
        def __call__(self, prompts, **kw):
            return SimpleNamespace(input_ids=torch.zeros(len(prompts), 77, dtype=torch.long))
    class Scheduler:
        config = SimpleNamespace(prediction_type='epsilon')
        @classmethod
        def from_pretrained(cls, root, **kw): return cls()
        @classmethod
        def from_config(cls, cfg): return cls()
    monkeypatch.setitem(sys.modules, 'diffusers', SimpleNamespace(AutoencoderKL=Module,
        UNet2DConditionModel=Module, DDIMScheduler=Scheduler, DDIMInverseScheduler=Scheduler))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(CLIPTextModel=Module, CLIPTokenizer=Tokenizer))
    components = load_diffusion_components({'model_path':str(tmp_path), 'variant':None,
        'prompt':'', 'prompt_guidance_scale':1., 'attention_slicing':False}, torch.device('cpu'))
    assert len(calls) == 3
    assert all('variant' not in call and call['use_safetensors'] is True and call['local_files_only'] is True for call in calls)
    assert components.manifest['variant'] is None
    assert all(call['torch_dtype'] == torch.float32 for call in calls)
