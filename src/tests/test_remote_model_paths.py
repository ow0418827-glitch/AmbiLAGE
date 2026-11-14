"""项目内模型目录解析，不加载真实权重或访问网络。"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

from src.paths import VISUAL_CONFIGS


def _complete(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'config.json').write_text('{}', encoding='utf-8')
    (folder / 'model.safetensors').write_bytes(b'explicit-small-path-fixture')
    return folder


def test_registry_explicit_relative_path_uses_project_not_working_directory(tmp_path, monkeypatch):
    from src import model_registry
    project, working = tmp_path / 'project', tmp_path / 'working'
    project.mkdir(); working.mkdir()
    chosen = _complete(project / 'models/custom')
    _complete(working / 'models/custom')
    monkeypatch.setattr(model_registry, 'PROJECT', project)
    monkeypatch.chdir(working)
    assert Path(model_registry.resolve_model('models/custom')) == chosen


def test_lucid_resolver_prefers_prepared_project_model_directories(tmp_path, monkeypatch):
    from src import model_registry
    from src.visual._lucid.config import resolve_optimization
    from src.visual.encoders import backbone
    raw = json.loads((VISUAL_CONFIGS / 'optimization.json').read_text(encoding='utf-8'))
    path = tmp_path / 'optimization.json'
    path.write_text(json.dumps(raw), encoding='utf-8')
    directories = {}
    for index, spec in enumerate(raw['surrogates']):
        folder = tmp_path / f'model{index}'
        folder.mkdir()
        (folder / 'open_clip_pytorch_model.safetensors').write_bytes(b'explicit-fixture')
        directories[spec['hf_repo']] = folder
    monkeypatch.setattr(model_registry, 'resolve_model', lambda repo: str(directories[repo]))
    monkeypatch.setattr(backbone, 'resolve_hf_cached_weight', lambda *a, **k: pytest.fail('must not depend on global cache'))
    config = resolve_optimization(path)
    assert [Path(s.weight_path) for s in config.surrogates] == [
        directories[s['hf_repo']] / 'open_clip_pytorch_model.safetensors' for s in raw['surrogates']]


def test_lucid_incomplete_prepared_directory_does_not_hide_failure_with_cache(tmp_path, monkeypatch):
    from src import model_registry
    from src.visual._lucid.config import resolve_optimization
    from src.visual.encoders import backbone
    folder = tmp_path / 'incomplete'
    folder.mkdir()
    monkeypatch.setattr(model_registry, 'resolve_model', lambda repo: str(folder))
    monkeypatch.setattr(backbone, 'resolve_hf_cached_weight', lambda *a, **k: pytest.fail('incomplete project model must stop'))
    with pytest.raises(FileNotFoundError, match='项目内代理目录'):
        resolve_optimization()


def _isolate_model_caches(monkeypatch, tmp_path):
    from src.visual.encoders import backbone
    monkeypatch.setattr(backbone, '_huggingface_hub_roots', lambda: [])
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'empty-cache'))
    monkeypatch.setitem(sys.modules, 'transformers.utils.hub', SimpleNamespace(
        cached_file=lambda *a, **k: None, TRANSFORMERS_CACHE=None))


def test_memory_encoder_does_not_discover_parent_or_working_directory(tmp_path, monkeypatch):
    from src.visual.encoders.model_paths import resolve_memory_model
    base, working = tmp_path / 'project', tmp_path / 'working'
    base.mkdir(); working.mkdir()
    _complete(tmp_path / 'unregistered-path-fixture')
    _complete(working / 'unregistered-path-fixture')
    _isolate_model_caches(monkeypatch, tmp_path)
    monkeypatch.chdir(working)
    with pytest.raises(FileNotFoundError):
        resolve_memory_model({'model': 'unregistered-path-fixture'}, base)
    assert resolve_memory_model({'model': 'unregistered-path-fixture', 'search_roots': [str(tmp_path)]}, base) == tmp_path / 'unregistered-path-fixture'


def test_memory_encoder_prefers_registered_project_model(tmp_path, monkeypatch):
    from src import model_registry
    from src.visual.encoders.model_paths import resolve_memory_model
    prepared = _complete(tmp_path / 'models/eval_gme')
    _isolate_model_caches(monkeypatch, tmp_path)
    monkeypatch.setattr(model_registry, 'model_specs', lambda: {'eval_gme': {'repo_id': 'fixture/model'}})
    monkeypatch.setattr(model_registry, 'resolve_model', lambda value: str(prepared))
    assert resolve_memory_model({'model': 'fixture/model'}, tmp_path) == prepared


def test_generation_model_override_relative_to_project(tmp_path, monkeypatch):
    from src.visual._ambilage import backends
    project, working = tmp_path / 'project', tmp_path / 'working'
    desired = _complete(project / 'models/planner_alt')
    _complete(working / 'models/planner_alt')
    monkeypatch.setattr(backends, 'PROJECT', project)
    monkeypatch.chdir(working)
    assert Path(backends.local_model('models/planner_alt', 'planner')) == desired
