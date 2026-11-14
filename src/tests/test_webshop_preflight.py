"""Deployment checks reject incomplete project-local resources without model calls."""
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools.preflight import check_webshop_resources


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def shop_root(tmp_path):
    base = tmp_path / 'resources/webshop'
    source, bundle, catalog = (base / name for name in ('source', 'catalog_bundle', 'catalog'))
    vendor = []
    for name in ('LICENSE.md', 'web_agent_site/app.py', 'web_agent_site/engine/goal.py'):
        content = b'# synthetic offline fixture\n'
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        vendor.append(dict(path=name, vendored_bytes=len(content), vendored_sha256=hashlib.sha256(content).hexdigest()))
    write_json(source / 'VENDOR_MANIFEST.json', dict(files=vendor))
    data = dict(bundle_id='synthetic', products=2, historical_build_inputs_required_at_runtime=False)
    index = dict(backend='lucene', path='index')
    files = []
    for ordinal, (name, content) in enumerate({
        'catalog.sqlite': b'synthetic database bytes', 'goals.json': b'[]',
        'data_manifest.json': json.dumps(data).encode(), 'index_manifest.json': json.dumps(index).encode(),
        'index/segments_1': b'synthetic index bytes',
    }.items()):
        path = catalog / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        encoded = gzip.compress(content, mtime=0)
        part = 'chunks/file-' + str(ordinal) + '.part'
        (bundle / part).parent.mkdir(parents=True, exist_ok=True)
        (bundle / part).write_bytes(encoded)
        files.append(dict(path=name, bytes=len(content), codec='gzip', sha256=hashlib.sha256(content).hexdigest(),
                          compressed_bytes=len(encoded), chunks=[dict(path=part, bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())]))
    manifest = dict(schema_version=1, codec='gzip', products=2, bundle_id='synthetic', search_backend='lucene',
                    files=files, raw_bytes=sum(item['bytes'] for item in files),
                    stored_bytes=sum(item['compressed_bytes'] for item in files))
    write_json(bundle / 'manifest.json', manifest)
    write_json(catalog / 'RESTORED.json', dict(catalog_bundle_manifest_sha256=hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest()))
    return tmp_path


def test_preflight_ready_without_model_or_shop_execution(shop_root):
    report = check_webshop_resources(shop_root)
    assert report['status'] == 'ready'
    assert report['products'] == 2 and report['chunks'] == 5
    assert report['shopping_model_calls'] == 0 and report['models_loaded'] is False


def test_preflight_detects_lost_runtime_index(shop_root):
    (shop_root / 'resources/webshop/catalog/index/segments_1').unlink()
    with pytest.raises(FileNotFoundError):
        check_webshop_resources(shop_root)


def test_preflight_detects_changed_vendored_source(shop_root):
    (shop_root / 'resources/webshop/source/web_agent_site/app.py').write_text('raise RuntimeError()', encoding='utf-8')
    with pytest.raises(ValueError, match='source content changed'):
        check_webshop_resources(shop_root)


def test_preflight_detects_orphaned_chunks(shop_root):
    (shop_root / 'resources/webshop/catalog_bundle/chunks/orphan.part').write_bytes(b'bad')
    with pytest.raises(ValueError, match='orphaned'):
        check_webshop_resources(shop_root)
