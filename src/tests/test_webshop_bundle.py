"""Micro catalog recovery and vendored HTML routing; no Java/model/network calls."""
import copy
import importlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import types

import pytest

from src.paths import PROJECT
from src.webshop import bundle
from src.webshop.disk_catalog import DiskCatalog, fingerprint


def _source(tmp_path):
    source = tmp_path / 'source_catalog'
    source.mkdir()
    goals = [dict(asin='TEST000001', category='fixture', query='fixture', name='Fixture product 1',
                  product_category='fixture', instruction_text='Find a fixture product',
                  attributes=['fixture'], price_upper=100, goal_options={}, weight=1)]
    with sqlite3.connect(source / 'catalog.sqlite') as db:
        db.execute('CREATE TABLE products(asin TEXT PRIMARY KEY,ordinal INTEGER,data TEXT,price REAL,category TEXT,query TEXT)')
        db.execute('CREATE TABLE attribute_products(attribute TEXT,asin TEXT)')
        db.execute('CREATE VIRTUAL TABLE products_fts USING fts5(asin UNINDEXED, contents)')
        for i in range(1, 4):
            asin = f'TEST{i:06d}'
            p = dict(asin=asin, name=f'Fixture product {i}', Title=f'Fixture product {i}', Description='fixture',
                     BulletPoints=['fixture'], Attributes=['fixture'], category='fixture', query='fixture',
                     product_category='fixture', MainImage='https://example.invalid/fixture.jpg',
                     images=['https://example.invalid/fixture.jpg'], Price='$5', pricing=[5.0],
                     options={}, option_to_image={}, Reviews=[], Rating='N.A.')
            db.execute('INSERT INTO products VALUES(?,?,?,?,?,?)', (asin, i - 1, json.dumps(p), 5, 'fixture', 'fixture'))
            db.execute('INSERT INTO products_fts VALUES(?,?)', (asin, p['Title']))
            db.execute('INSERT INTO attribute_products VALUES(?,?)', ('fixture', asin))
    (source / 'goals.json').write_text(json.dumps(goals), encoding='utf-8')
    (source / 'data_manifest.json').write_text(json.dumps(dict(bundle_id='fixture-bundle', products=3,
        scope='explicit_test_fixture', goals_sha256=fingerprint(goals),
        source_files={'items': {'path': 'D:/historical/raw-items.json'}})), encoding='utf-8')
    (source / 'index_manifest.json').write_text(json.dumps(dict(bundle_id='fixture-bundle', backend='sqlite_fts5',
        documents=3, empty_product_asins=[], path='catalog.sqlite')), encoding='utf-8')
    return source


def _packed(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, 'PROJECT', tmp_path)
    source = _source(tmp_path)
    destination = tmp_path / 'bundle'
    manifest = bundle.pack_catalog(source, destination, chunk_bytes=128)
    return source, destination, manifest


def test_chunked_restore_preserves_database_and_goals_bytes(tmp_path, monkeypatch):
    source, packed, manifest = _packed(tmp_path, monkeypatch)
    assert all(p['bytes'] <= 128 for f in manifest['files'] for p in f['chunks'])
    target = tmp_path / 'materialized'
    report = bundle.restore_catalog(packed, target)
    assert report['database_and_index_original_bytes'] and not report['search_ranking_changed']
    assert bundle.sha256(source / 'catalog.sqlite') == bundle.sha256(target / 'catalog.sqlite')
    assert (source / 'goals.json').read_bytes() == (target / 'goals.json').read_bytes()
    provenance = json.loads((target / 'data_manifest.json').read_text(encoding='utf-8'))
    assert provenance['source_files']['items'] == {'historical_build_input_path': 'D:/historical/raw-items.json'}
    assert provenance['historical_build_inputs_required_at_runtime'] is False
    with DiskCatalog.open(target) as catalog:
        assert catalog.count == 3
        assert catalog.product('TEST000001')['Title'] == 'Fixture product 1'
        assert len(catalog.search(['fixture'])) == 3


def test_leftover_lock_file_does_not_block_recovery(tmp_path, monkeypatch):
    _, packed, _ = _packed(tmp_path, monkeypatch)
    target = tmp_path / 'restored'
    target.mkdir()
    (target / '.restore.lock').write_text('terminated previous process', encoding='utf-8')
    assert bundle.restore_catalog(packed, target)['files'] == 4


def test_active_restore_lock_prevents_concurrent_restore(tmp_path, monkeypatch):
    _, packed, _ = _packed(tmp_path, monkeypatch)
    target = tmp_path / 'restored'
    target.mkdir()
    handle = bundle._acquire_restore_lock(target / '.restore.lock')
    try:
        with pytest.raises(RuntimeError, match='actively restoring'):
            bundle.restore_catalog(packed, target)
    finally:
        bundle._release_restore_lock(handle)


def test_changed_chunk_is_rejected_before_catalog_materialization(tmp_path, monkeypatch):
    _, packed, manifest = _packed(tmp_path, monkeypatch)
    path = packed / manifest['files'][0]['chunks'][0]['path']
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match='checksum'):
        bundle.restore_catalog(packed, tmp_path / 'restored')
    assert not (tmp_path / 'restored/catalog.sqlite').exists()


@pytest.mark.parametrize('path', ['../escape', '/outside', '', 'C:/outside', 'bad\\path'])
def test_chunk_paths_cannot_escape_bundle(tmp_path, path):
    with pytest.raises(ValueError):
        bundle._inside(tmp_path, path)


def test_completed_pack_cannot_be_reused_for_another_source(tmp_path, monkeypatch):
    source, packed, _ = _packed(tmp_path, monkeypatch)
    other = tmp_path / 'other_source'
    shutil.copytree(source, other)
    with pytest.raises(ValueError, match='different source'):
        bundle.pack_catalog(other, packed, chunk_bytes=128)


def test_ensure_catalog_repairs_missing_materialized_files(tmp_path, monkeypatch):
    _, packed, _ = _packed(tmp_path, monkeypatch)
    target = tmp_path / 'restored'
    bundle.restore_catalog(packed, target)
    (target / 'goals.json').unlink()
    assert bundle.ensure_catalog(target, packed) == target
    assert (target / 'goals.json').is_file()


def test_vendored_source_and_license_have_frozen_content():
    source = PROJECT / 'resources/webshop/source'
    manifest = json.loads((source / 'VENDOR_MANIFEST.json').read_text(encoding='utf-8'))
    assert (source / 'LICENSE.md').is_file()
    assert len(manifest['files']) == 18
    assert manifest['nlp_protocol']['package'] == 'en_core_web_lg'
    for record in manifest['files']:
        assert bundle.sha256(source / record['path']) == record['vendored_sha256']
    assert not (source / 'web_agent_site/envs/chromedriver').exists()


def _fake_imports(monkeypatch, source, *, language_meta=None):
    for name in list(sys.modules):
        if name == 'web_agent_site' or name.startswith('web_agent_site.'):
            monkeypatch.delitem(sys.modules, name)
    calls = []
    class Language:
        meta = language_meta or dict(lang='en', name='core_web_lg', version='3.8.0')
        def __call__(self, value): return []
    fake_spacy = types.ModuleType('spacy')
    def load(name):
        calls.append(name)
        return Language()
    fake_spacy.load = load
    monkeypatch.setitem(sys.modules, 'spacy', fake_spacy)
    package, search, lucene = (types.ModuleType(name) for name in ('pyserini', 'pyserini.search', 'pyserini.search.lucene'))
    package.__path__ = search.__path__ = []
    def forbid(*args, **kwargs):
        raise AssertionError('A micro SQLite fixture must not initialize Java/Lucene')
    lucene.LuceneSearcher = forbid
    for module in (package, search, lucene):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.delenv('AMBILAGE_WEBSHOP_SPACY_MODEL', raising=False)
    return calls


def test_lazy_language_model_and_micro_html_service_use_only_project_source(tmp_path, monkeypatch):
    from src.webshop import full_webshop, local_webshop
    _, packed, _ = _packed(tmp_path, monkeypatch)
    catalog = tmp_path / 'catalog'
    bundle.restore_catalog(packed, catalog)
    source = tmp_path / 'resources/webshop/source'
    shutil.copytree(PROJECT / 'resources/webshop/source', source)
    calls = _fake_imports(monkeypatch, source)
    goal = importlib.import_module('web_agent_site.engine.goal')
    assert calls == [] and goal.nlp is None
    monkeypatch.setattr(full_webshop, 'PROJECT', tmp_path)
    monkeypatch.setattr(local_webshop, 'configure_java', lambda: None)
    site, metadata = full_webshop.make_site('catalog', 'resources/webshop/source',
                                          task_registry={'fixed_fixture': 0})
    try:
        assert calls == ['en_core_web_lg']
        assert Path(metadata['webshop_source']).is_relative_to(tmp_path)
        assert metadata['products'] == 3 and metadata['search_backend'] == 'sqlite_fts5'
        client = site.app.test_client()
        assert client.get('/fixed_fixture').status_code == 200
        response = client.get("/search_results/fixed_fixture/['fixture']/1")
        assert response.status_code == 200 and b'Fixture product 1' in response.data
        assert client.get('/ambilage-disk/health').get_json()['products'] == 3
        assert not site.user_sessions['fixed_fixture']['done']
    finally:
        site.app._ambilage_disk_catalog.close()
        for name in list(sys.modules):
            if name == 'web_agent_site' or name.startswith('web_agent_site.'):
                monkeypatch.delitem(sys.modules, name)


def test_language_model_never_falls_back_to_a_different_reward_model(monkeypatch):
    source = PROJECT / 'resources/webshop/source'
    calls = _fake_imports(monkeypatch, source, language_meta=dict(lang='en', name='core_web_sm', version='3.8.0'))
    goal = importlib.import_module('web_agent_site.engine.goal')
    with pytest.raises(ValueError, match='en_core_web_lg'):
        goal.initialize_language_model()
    assert calls == ['en_core_web_lg']
    for name in list(sys.modules):
        if name == 'web_agent_site' or name.startswith('web_agent_site.'):
            monkeypatch.delitem(sys.modules, name)
