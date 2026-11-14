"""Optional shop environment checks use metadata and tiny subprocess stubs only."""
from types import SimpleNamespace
import importlib.metadata
import pytest

from tools import check_webshop_runtime as runtime


@pytest.mark.parametrize('text,expected', [
    ('openjdk version "21.0.7" 2025-04-15', 21),
    ('openjdk 24.0.1 2025-04-15', 24),
    ('java version "1.8.0_361"', 8),
])
def test_java_version_detection_without_starting_a_shop(monkeypatch, text, expected):
    monkeypatch.delenv('JAVA_HOME', raising=False)
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: '/fixture/bin/java')
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stderr=text, stdout=''))
    result = runtime.java_runtime()
    assert result['major'] == expected
    assert result['compatible'] == (expected >= 21)


def test_missing_java_is_reported_without_installing(monkeypatch):
    monkeypatch.delenv('JAVA_HOME', raising=False)
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: None)
    assert runtime.java_runtime()['compatible'] is False


def test_invalid_explicit_java_home_does_not_silently_use_path_java(tmp_path, monkeypatch):
    monkeypatch.setenv('JAVA_HOME', str(tmp_path / 'missing-java'))
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: pytest.fail('invalid explicit Java home must stop'))
    assert runtime.java_runtime()['compatible'] is False


def test_pinned_packages_and_large_english_model_are_explicit(monkeypatch):
    monkeypatch.delenv('AMBILAGE_WEBSHOP_SPACY_MODEL', raising=False)
    monkeypatch.setattr(runtime, 'java_runtime', lambda: dict(compatible=True))
    monkeypatch.setattr(runtime.importlib.metadata, 'version', lambda name: runtime.EXPECTED_PACKAGES[name])
    result = runtime.inspect_webshop_runtime()
    assert result['ready'] is True
    assert result['packages']['en-core-web-lg']['expected'] == '3.8.0'
    assert result['models_loaded'] is False and result['shop_started'] is False


def test_small_english_model_cannot_silently_replace_historical_large_one(monkeypatch):
    monkeypatch.delenv('AMBILAGE_WEBSHOP_SPACY_MODEL', raising=False)
    monkeypatch.setattr(runtime, 'java_runtime', lambda: dict(compatible=True))
    def version(name):
        if name == 'en-core-web-lg':
            raise importlib.metadata.PackageNotFoundError(name)
        return runtime.EXPECTED_PACKAGES[name]
    monkeypatch.setattr(runtime.importlib.metadata, 'version', version)
    result = runtime.inspect_webshop_runtime()
    assert result['ready'] is False
    assert result['packages']['en-core-web-lg']['installed'] is None


def test_java_only_probe_does_not_read_model_distribution(monkeypatch):
    monkeypatch.setattr(runtime, 'java_runtime', lambda: dict(compatible=True))
    monkeypatch.setattr(runtime.importlib.metadata, 'version', lambda name: pytest.fail('must not inspect models'))
    result = runtime.inspect_webshop_runtime(java_only=True)
    assert result['ready'] is True and result['packages'] == {}


def test_explicit_model_directory_requires_same_large_model_metadata(tmp_path, monkeypatch):
    import json
    (tmp_path / 'meta.json').write_text(json.dumps(dict(lang='en', name='core_web_lg', version='3.8.0')), encoding='utf-8')
    monkeypatch.setenv('AMBILAGE_WEBSHOP_SPACY_MODEL', str(tmp_path))
    assert runtime.english_model_metadata()['compatible'] is True
    (tmp_path / 'meta.json').write_text(json.dumps(dict(lang='en', name='core_web_sm', version='3.8.0')), encoding='utf-8')
    assert runtime.english_model_metadata()['compatible'] is False
