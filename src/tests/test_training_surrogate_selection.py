"""Offline selections, dispatch and frozen evidence; no real weights are loaded."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

from src.visual import training_surrogates as choices
from src.visual._ambilage.config import AmbiLAGEConfig
from src.visual.encoders.backbone import BackboneConfig


def custom_file(tmp_path, monkeypatch, rows=None):
    monkeypatch.setattr(choices, 'PROJECT', tmp_path)
    rows = rows or [asdict(s) for s in choices.select_training_surrogates().specs[:1]]
    path = tmp_path / 'custom.json'
    path.write_text(json.dumps(dict(surrogates=rows, allow_downloads=False)), encoding='utf-8')
    return path


def real_snapshots(selection):
    return [SimpleNamespace(snapshot=lambda s=s: dict(name=s.name, pretrained=s.pretrained,
        hf_repo=s.hf_repo, weight_sha256='1' * 64)) for s in selection.specs]


def test_default_uses_exact_lucid_model_and_pretraining_configuration():
    from src.visual._lucid.config import load_config
    selected = choices.selection_from_config(AmbiLAGEConfig())
    assert selected.profile == 'lucid'
    assert selected.specs == load_config().surrogates
    assert [(s.name, s.pretrained) for s in selected.specs] == [
        ('ViT-B-16', 'openai'), ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k')]
    assert choices.validate_selection_audit(selected.audit(), AmbiLAGEConfig().to_dict()).specs == selected.specs


def test_default_loader_delegates_to_lucid_resolver_and_loader(monkeypatch):
    from src.visual import training_objectives
    from src.visual._lucid import config, surrogates
    selection = choices.select_training_surrogates()
    base = config.load_config()
    resolved = replace(base, surrogates=tuple(replace(s, weight_path='/explicit/' + str(i)) for i, s in enumerate(base.surrogates)))
    seen = {}
    def resolve(path):
        seen['source'] = path
        return resolved
    def load(specs, **kw):
        seen.update(specs=specs, options=kw)
        return real_snapshots(selection)
    monkeypatch.setattr(config, 'resolve_optimization', resolve)
    monkeypatch.setattr(surrogates, 'load_surrogates', load)
    models = training_objectives.load_training_surrogates('cpu')
    assert len(models) == 3 and seen['source'] == selection.source
    assert seen['specs'] == resolved.surrogates
    assert seen['options'] == dict(device='cpu', precision=base.attack_precision, allow_downloads=False,
        allow_device_fallback=False, require_true_local_tokens=base.runtime.require_true_local_tokens,
        hash_weights=base.runtime.hash_model_weights)


def test_heterogeneous_four_remains_an_explicit_alternative():
    selected = choices.select_training_surrogates('heterogeneous4')
    assert [s.name for s in selected.specs] == ['ViT-B-16', 'ViT-L-14', 'ViT-SO400M-14-SigLIP-384', 'ViT-H-14']
    assert selected.audit()['model_count'] == 4


def test_custom_config_is_project_relative_and_independent_of_cwd(tmp_path, monkeypatch):
    source = custom_file(tmp_path, monkeypatch)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    selected = choices.select_training_surrogates('custom', 'custom.json')
    assert selected.source == source and len(selected.specs) == 1
    assert selected.audit()['config_source'] == 'custom.json'


@pytest.mark.parametrize('row', [
    {'name': 'ViT-g-14', 'pretrained': 'laion2b_s34b_b88k'},
    {'name': 'ViT-B-16', 'pretrained': 'openai', 'hf_repo': 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct'},
    {'name': 'ViT-B-16', 'pretrained': 'openai', 'weight_path': 'eval_g14/model.safetensors'},
])
def test_custom_selection_cannot_import_heldout_encoders(tmp_path, monkeypatch, row):
    path = custom_file(tmp_path, monkeypatch, [row])
    with pytest.raises(ValueError, match='留出评估编码器'):
        choices.select_training_surrogates('custom', path)


def test_same_architecture_different_weights_are_distinct_but_duplicates_fail():
    selected = choices.select_training_surrogates()
    assert selected.specs[1].name == selected.specs[2].name
    choices.validate_training_specs(selected.specs)
    with pytest.raises(ValueError, match='不能重复'):
        choices.validate_training_specs((selected.specs[0], selected.specs[0]))


def test_custom_config_rejects_unknown_or_implicit_weight_fields(tmp_path, monkeypatch):
    path = custom_file(tmp_path, monkeypatch, [{'name': 'ViT-B-16'}])
    with pytest.raises(ValueError, match='预训练身份'):
        choices.select_training_surrogates('custom', path)


def test_loaded_pretrained_identity_is_checked_even_when_architecture_matches():
    selected = choices.select_training_surrogates()
    models = real_snapshots(selected)
    models[2] = SimpleNamespace(snapshot=lambda: dict(name='ViT-B-32', pretrained='openai',
        hf_repo=selected.specs[2].hf_repo, weight_sha256='1' * 64))
    with pytest.raises(ValueError, match='预训练权重身份'):
        choices.validate_loaded_surrogates(models, selected)


@pytest.mark.parametrize('kwargs', [
    {'surrogate_set': 'unknown'}, {'surrogate_set': 'custom'},
    {'surrogate_set': 'lucid', 'surrogate_config': 'custom.json'},
])
def test_invalid_selection_configuration_fails_before_loading_models(kwargs):
    with pytest.raises(ValueError):
        AmbiLAGEConfig(**kwargs).validate()


def test_explicit_preset_clears_previous_custom_file():
    config = AmbiLAGEConfig(surrogate_set='custom', surrogate_config='unused.json')
    chosen = choices.apply_surrogate_overrides(config, 'lucid')
    assert chosen.surrogate_set == 'lucid' and chosen.surrogate_config is None


def test_custom_loader_uses_declared_local_weights_and_does_not_hide_missing_file(tmp_path,monkeypatch):
    from src.visual import training_objectives
    from src import model_registry
    row=dict(name='ViT-B-16',pretrained='openai',hf_repo='timm/vit_base_patch16_clip_224.openai',
             weight_path='chosen/explicit.safetensors')
    path=custom_file(tmp_path,monkeypatch,[row])
    models=tmp_path/'models'
    weight=models/row['weight_path']
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'offline_explicit_weight_path_fixture')
    monkeypatch.setattr(training_objectives,'MODEL_ROOT',models)
    seen={}
    def factory(spec,**kw):
        seen.update(spec=spec,options=kw)
        return SimpleNamespace(snapshot=lambda:dict(name=spec.name,pretrained=spec.pretrained,
            hf_repo=spec.hf_repo,weight_sha256='1'*64))
    monkeypatch.setattr(training_objectives,'OpenCLIPSurrogate',factory)
    monkeypatch.setattr(model_registry,'resolve_model',lambda *a,**k:pytest.fail('explicit weights must not fall back'))
    training_objectives.load_training_surrogates('cpu',path)
    assert Path(seen['spec'].weight_path)==weight
    assert seen['options']['allow_downloads'] is False
    assert seen['options']['allow_device_fallback'] is False
    weight.unlink()
    with pytest.raises(FileNotFoundError,match='所选训练代理'):
        training_objectives.load_training_surrogates('cpu',path)


@pytest.mark.parametrize('profile,count', [('lucid', 3), ('heterogeneous4', 4), ('custom', 2)])
def test_shared_generation_api_freezes_selected_proxy_count(tmp_path, monkeypatch, profile, count):
    from src.visual.optimization import generate_attack
    from src.visual._ambilage.evidence import validate_ready
    kwargs = dict(surrogate_set=profile)
    if profile == 'custom':
        rows = [asdict(s) for s in choices.select_training_surrogates().specs[:2]]
        path = custom_file(tmp_path, monkeypatch, rows)
        kwargs = dict(surrogate_config=path)
    source, target = tmp_path / 'source.png', tmp_path / 'target.png'
    Image.new('RGB', (24, 24), (180, 170, 160)).save(source)
    Image.new('RGB', (24, 24), (80, 100, 140)).save(target)
    image, meta = generate_attack(source, target, tmp_path / 'out', method='ambilage',
        steps=1, device='cpu', target_text='合成公开特点', smoke_test=True, **kwargs)
    training = meta['training_objective']
    assert training['surrogate_selection']['profile'] == profile
    assert len(training['surrogates']) == len(training['public_objectives']) == count
    assert json.loads((image.parent / 'training_surrogates.json').read_text(encoding='utf-8')) == training['surrogate_selection']
    assert validate_ready(image.parent, require_production=False)['status'] == 'completed'
    # A frozen custom selection is self-contained; validation needs its saved copy.
    if profile == 'custom':
        path.unlink()
        validate_ready(image.parent, require_production=False)


def test_generate_cli_enforces_selection_flags_and_method_scope():
    from tools.generate import parse_args
    base = ['--source', 'source.png', '--target', 'target.png', '--output', 'out']
    assert parse_args(base).surrogate_set is None
    assert parse_args(base + ['--surrogate-set', 'heterogeneous4']).surrogate_set == 'heterogeneous4'
    assert parse_args(base + ['--surrogate-config', 'custom.json']).surrogate_config == Path('custom.json')
    with pytest.raises(SystemExit):
        parse_args(base + ['--surrogate-set', 'lucid', '--surrogate-config', 'custom.json'])
    with pytest.raises(SystemExit):
        parse_args(base + ['--method', 'lucid', '--surrogate-set', 'heterogeneous4'])


@pytest.mark.parametrize('profile,keys', [
    ('lucid', {'clip_b16', 'clip_b32', 'clip_b32_laion'}),
    ('heterogeneous4', {'clip_b16', 'clip_l14', 'siglip_so400m', 'clip_h14'}),
])
def test_model_preparation_matches_the_selected_training_set(profile, keys, capsys):
    from tools.prepare_models import main
    assert main(['--scope', 'encoders', '--surrogate-set', profile]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert {row['key'] for row in plan['models']} == keys
    assert plan['download'] is False


@pytest.mark.parametrize('script,args',[
    ('tools/prepare_models.py',['--scope','encoders']),
    ('tools/preflight.py',['--code-only']),
])
def test_bootstrap_selection_tools_do_not_require_model_runtime_packages(script,args):
    from src.paths import PROJECT
    code="""import builtins,sys,runpy
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','diffusers','huggingface_hub'}:
        raise RuntimeError('Bootstrap tools must not import model runtime packages')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=ARGS
runpy.run_path(SCRIPT,run_name='__main__')
""".replace('ARGS',repr([script]+args)).replace('SCRIPT',repr(script))
    data=json.loads(subprocess.check_output([sys.executable,'-B','-X','utf8','-c',code],
        cwd=PROJECT,text=True,encoding='utf-8'))
    assert data['training_surrogates']['profile']=='lucid'


def test_preflight_does_not_require_unselected_training_weights(tmp_path, monkeypatch, capsys):
    from tools import preflight
    from src.model_registry import model_specs
    registry = {key: value for key, value in model_specs().items()
                if key in {'clip_b16', 'clip_b32', 'clip_b32_laion', 'clip_l14'}}
    models = tmp_path / 'models'
    for key in ('clip_b16', 'clip_b32', 'clip_b32_laion'):
        folder = models / key
        folder.mkdir(parents=True)
        (folder / 'fixture.safetensors').write_bytes(b'explicit_offline_fixture')
        (folder / 'download_manifest.json').write_text(json.dumps(dict(repo_id=registry[key]['repo_id'],
            revision='fixture', files={'fixture.safetensors': 'unused_hash_without_verification'})), encoding='utf-8')
    monkeypatch.setattr(preflight, 'MODEL_ROOT', models)
    monkeypatch.setattr(preflight, 'RESOURCES', tmp_path / 'resources')
    monkeypatch.setattr(preflight, 'model_specs', lambda: registry)
    monkeypatch.setattr(preflight, 'check_code', lambda: {})
    monkeypatch.setattr(preflight, 'inspect_runtime', lambda hardware: {})
    assert preflight.main(['--code-only', '--require-models']) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report['models']) == {'clip_b16', 'clip_b32', 'clip_b32_laion'}


def test_experiment_cli_passes_the_explicit_proxy_selection(tmp_path, monkeypatch):
    from src.experiments import base_workflow
    output = tmp_path / 'runs'
    monkeypatch.setattr(base_workflow, 'OUTPUT', output)
    calls = []
    monkeypatch.setattr(base_workflow, 'generate_group', lambda *args, **kw: calls.append(kw))
    assert base_workflow.main(['generate', '--method', 'ambilage', '--group', '07_shampoo',
        '--output', str(output / 'test'), '--surrogate-set', 'heterogeneous4']) == 0
    assert calls[0]['surrogate_set'] == 'heterogeneous4'


def test_unified_doctor_reports_the_requested_proxy_selection(monkeypatch):
    from src.experiments import pipeline,runtime_check
    seen={}
    monkeypatch.setattr(runtime_check,'doctor',lambda device,**kw:seen.update(device=device,**kw))
    assert pipeline.main(['doctor','--device','cpu','--surrogate-set','heterogeneous4'])==0
    assert seen['surrogate_set']=='heterogeneous4'


def test_generation_binding_records_and_freezes_proxy_choice(tmp_path, monkeypatch):
    from src.experiments import base_workflow, pipeline
    from src.visual import optimization
    from src.paths import EVALUATION_BASE
    output = tmp_path / 'runs'
    monkeypatch.setattr(base_workflow, 'OUTPUT', output)
    monkeypatch.setattr(pipeline, 'source_snapshot', lambda: {'fixture': 'public_code'})
    def stop(*args, **kw):
        assert kw['ambilage_config'].surrogate_set == 'heterogeneous4'
        raise RuntimeError('stopped_before_loading')
    monkeypatch.setattr(optimization, 'generate_attack', stop)
    with pytest.raises(RuntimeError, match='stopped_before_loading'):
        base_workflow.generate_group(EVALUATION_BASE, '07_shampoo', 'ambilage', output / 'trial',
            surrogate_set='heterogeneous4')
    binding = json.loads((output / 'trial/PUBLIC_RUN.json').read_text(encoding='utf-8'))
    assert binding['training_surrogates']['profile'] == 'heterogeneous4'
    assert binding['config']['surrogate_set'] == 'heterogeneous4'
    with pytest.raises(ValueError):
        base_workflow.generate_group(EVALUATION_BASE, '07_shampoo', 'ambilage', output / 'trial', surrogate_set='lucid')
