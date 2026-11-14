"""Only tiny synthetic catalogs; no full-corpus scan or model loading."""
import hashlib
import json
import sqlite3

import pytest

from src.visual.selection.webshop_public import (
    build_public_index, product_source_kind, sqlite_public_products,
)


def catalog(path):
    rows = [
        dict(asin='PUBLIC0002', name='Blue shampoo bottle', images=['https://m.media-amazon.com/two.jpg']),
        dict(asin='PUBLIC0001', name='White shampoo bottle', images=['https://m.media-amazon.com/one.jpg']),
        dict(asin='PUBLIC0003', name='Invalid image host', images=['https://private.invalid/no.jpg']),
    ]
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE products(asin TEXT PRIMARY KEY,ordinal INTEGER,data TEXT)')
        for ordinal, row in enumerate(rows):
            data = dict(row, instructions=['PRIVATE_INSTRUCTION_SENTINEL'], price=999,
                        shopping_actions=['PRIVATE_ACTION_SENTINEL'])
            db.execute('INSERT INTO products VALUES(?,?,?)', (row['asin'], ordinal, json.dumps(data)))
    return rows


def test_sql_projection_only_returns_public_fields_and_preserves_source(tmp_path):
    source = tmp_path / 'catalog.sqlite'
    rows = catalog(source)
    before = source.read_bytes()
    assert product_source_kind(source) == 'normalized_catalog_sqlite'
    projected = list(sqlite_public_products(source))
    assert projected == rows
    assert all(set(item) == {'asin', 'name', 'images'} for item in projected)
    assert 'PRIVATE_' not in json.dumps(projected)
    assert source.read_bytes() == before


def test_sql_and_json_public_indexes_agree_and_record_real_provenance(tmp_path):
    source = tmp_path / 'catalog.sqlite'
    rows = catalog(source)
    raw = tmp_path / 'products.json'
    raw.write_text(json.dumps(rows), encoding='utf-8')
    sql_index, json_index = tmp_path / 'sql-index.sqlite', tmp_path / 'json-index.sqlite'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    sql_manifest = build_public_index(source, sql_index)
    json_manifest = build_public_index(raw, json_index)
    with sqlite3.connect(sql_index) as left, sqlite3.connect(json_index) as right:
        assert left.execute('SELECT * FROM products ORDER BY id').fetchall() == right.execute('SELECT * FROM products ORDER BY id').fetchall()
        assert left.execute('SELECT count(*) FROM products').fetchone()[0] == 2
    assert sql_manifest['scanned_products'] == 3
    assert sql_manifest['private_fields_read'] is False
    assert sql_manifest['normalization_may_differ_from_original_raw_catalog'] is True
    assert json_manifest['normalization_may_differ_from_original_raw_catalog'] is False
    assert sql_manifest['source_sha256'] == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert build_public_index(source, sql_index, verify_source=True) == sql_manifest


def test_sql_projection_rejects_unsupported_schema(tmp_path):
    source = tmp_path / 'unsupported.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE unrelated(value TEXT)')
    with pytest.raises(ValueError, match='products'):
        list(sqlite_public_products(source))
