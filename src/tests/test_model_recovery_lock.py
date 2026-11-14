"""微型模型文件验证版本恢复逻辑，不联网、不加载权重。"""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest


def _setup(tmp_path, monkeypatch):
    from tools import prepare_models as prepare
    monkeypatch.setattr(prepare, 'MODEL_ROOT', tmp_path / 'models')
    monkeypatch.setattr(prepare, 'MODEL_LOCK_PATH', tmp_path / 'resources/model_lock.json')
    monkeypatch.setattr(prepare, 'model_specs', lambda: {
        'clip_b16': {'repo_id': 'fixture/public-model', 'scope': 'encoders', 'role': 'fixture'}})
    calls = []
    class API:
        def __init__(self, **kwargs): pass
        def model_info(self, repo, revision=None):
            calls.append(revision)
            return SimpleNamespace(sha=revision or 'frozen-fixture-commit')
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=API))
    def download(repo, destination, **kwargs):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'open_clip_model.safetensors').write_bytes(b'explicit-tiny-fixture')
        (destination / 'config.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(prepare, 'download_model', download)
    return prepare, calls


def test_recovery_redownloads_frozen_commit_and_checks_same_files(tmp_path, monkeypatch, capsys):
    prepare, calls = _setup(tmp_path, monkeypatch)
    args = ['--models', 'clip_b16', '--download', '--endpoint', 'https://example.invalid']
    assert prepare.main(args) == 0
    original_lock = prepare.MODEL_LOCK_PATH.read_bytes()
    locked = json.loads(original_lock)['models']['clip_b16']
    assert locked['revision'] == 'frozen-fixture-commit'
    assert 'open_clip_model.safetensors' in locked['files']
    monkeypatch.setattr(prepare, 'MODEL_ROOT', tmp_path / 'fresh-server/models')
    assert prepare.main(args) == 0
    assert calls == [None, 'frozen-fixture-commit']
    assert prepare.MODEL_LOCK_PATH.read_bytes() == original_lock
    capsys.readouterr()


def test_recovery_rejects_changed_download_even_at_same_commit(tmp_path, monkeypatch, capsys):
    prepare, _ = _setup(tmp_path, monkeypatch)
    args = ['--models', 'clip_b16', '--download', '--endpoint', 'https://example.invalid']
    assert prepare.main(args) == 0
    original_lock = prepare.MODEL_LOCK_PATH.read_bytes()
    monkeypatch.setattr(prepare, 'MODEL_ROOT', tmp_path / 'fresh-server/models')
    def changed(repo, destination, **kwargs):
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'open_clip_model.safetensors').write_bytes(b'changed-fixture')
        (destination / 'config.json').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(prepare, 'download_model', changed)
    with pytest.raises(ValueError, match='recovery lock'):
        prepare.main(args)
    assert prepare.MODEL_LOCK_PATH.read_bytes() == original_lock
    capsys.readouterr()
