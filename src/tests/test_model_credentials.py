"""Credential precedence uses disposable values and never contacts a model provider."""
import json
import pytest
from src.agent.config import llm_config


def _write(path,key=''):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'model':{'provider':'minimax','model':'frozen-model',
        'base_url':'https://example.invalid/v1/','api_key':key,'api_key_env':'MINIMAX_API_KEY'}}),encoding='utf-8')
    return path


def test_environment_overrides_ignored_local_key(tmp_path,monkeypatch):
    path=_write(tmp_path/'llm.local.json','local-disposable-secret')
    monkeypatch.setenv('MINIMAX_API_KEY','environment-disposable-secret')
    config=llm_config.load_model_config(path)
    assert config.api_key=='environment-disposable-secret'
    assert config.public()=={'provider':'minimax','model':'frozen-model','base_url':'https://example.invalid/v1'}
    assert config.api_key not in repr(config)


def test_ignored_local_key_works_without_environment(tmp_path,monkeypatch):
    path=_write(tmp_path/'llm.local.json','local-disposable-secret')
    monkeypatch.delenv('MINIMAX_API_KEY',raising=False)
    assert llm_config.load_model_config(path).api_key=='local-disposable-secret'


def test_clean_clone_uses_keyless_template_and_environment(tmp_path,monkeypatch):
    local=tmp_path/'llm.local.json'
    template=_write(tmp_path/'llm.example.json')
    monkeypatch.setattr(llm_config,'DEFAULT_CONFIG',local)
    monkeypatch.setattr(llm_config,'CONNECTION_TEMPLATE',template)
    monkeypatch.setenv('MINIMAX_API_KEY','environment-disposable-secret')
    assert llm_config.load_model_config(None).api_key=='environment-disposable-secret'
    assert not local.exists()


def test_clean_clone_missing_key_fails_without_printing_secrets(tmp_path,monkeypatch):
    local=tmp_path/'llm.local.json'
    template=_write(tmp_path/'llm.example.json')
    monkeypatch.setattr(llm_config,'DEFAULT_CONFIG',local)
    monkeypatch.setattr(llm_config,'CONNECTION_TEMPLATE',template)
    monkeypatch.delenv('MINIMAX_API_KEY',raising=False)
    with pytest.raises(ValueError,match='Missing model credential'):
        llm_config.load_model_config(None)


def test_committed_template_cannot_contain_a_key_even_with_environment(tmp_path,monkeypatch):
    template=_write(tmp_path/'llm.example.json','accidental-template-secret')
    monkeypatch.setattr(llm_config,'CONNECTION_TEMPLATE',template)
    monkeypatch.setenv('MINIMAX_API_KEY','environment-disposable-secret')
    with pytest.raises(ValueError,match='must not contain a credential') as error:
        llm_config.load_model_config(template)
    assert 'accidental-template-secret' not in str(error.value)


def test_invalid_environment_variable_name_is_rejected(tmp_path):
    path=_write(tmp_path/'llm.local.json','local-disposable-secret')
    value=json.loads(path.read_text(encoding='utf-8'))
    value['model']['api_key_env']='not a valid name'
    path.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='api_key_env'):
        llm_config.load_model_config(path)
