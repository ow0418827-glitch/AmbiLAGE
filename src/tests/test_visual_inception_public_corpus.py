"""Tiny explicit fixtures exercise public selection; no real catalog or network."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from collections import Counter

import pytest

from tools import prepare_visual_inception_queries as query_tool
from src.visual._visual_inception import public_catalog
from tools import build_visual_inception_public_training as build_tool


def _catalog_fixture(tmp_path, count=20):
    path = tmp_path / 'explicit_fixture_catalog.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE products(asin TEXT PRIMARY KEY,data TEXT,category TEXT,query TEXT)')
        db.execute('CREATE INDEX product_query_lookup ON products(query)')
        for index in list(range(count)) + [900, 901, 902]:
            fields = dict(name=f'Gentle shampoo fixture product {index}',
                Description=f'An explicit synthetic public description for shampoo fixture {index}.',
                images=[f'https://m.media-amazon.com/images/I/fixture{index}.jpg'],
                instructions='PRIVATE_FIXTURE_SENTINEL_SHOULD_NEVER_BE_PROJECTED')
            if index == 902:
                fields['name'] = 'Shampoo brush fixture accessory'
            db.execute('INSERT INTO products VALUES(?,?,?,?)',
                (f'B{index:09d}', json.dumps(fields), 'beauty', 'scalp treatments'))
    groups = {'07_shampoo': dict(source={'id': 'B000000900'}, target={'id': 'B000000901'},
                                 target_features={'category': 'shampoo'})}
    return path, groups


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_catalog_sampling_is_seeded_public_and_excludes_ids_and_accessories(tmp_path):
    catalog, groups = _catalog_fixture(tmp_path)
    kwargs = dict(excluded_ids={'B000000000'}, count=3, description_count=5,
                  reserve_count=5, max_read_per_group=100)
    first = public_catalog.prepare_public_catalog(catalog, groups, tmp_path / 'first_catalog', seed=42, **kwargs)
    public_catalog.prepare_public_catalog(catalog, groups, tmp_path / 'second_catalog', seed=42, **kwargs)
    public_catalog.prepare_public_catalog(catalog, groups, tmp_path / 'third_catalog', seed=43, **kwargs)
    relative = Path('07_shampoo/candidate_reserve.jsonl')
    first_bytes = (tmp_path / 'first_catalog' / relative).read_bytes()
    assert first_bytes == (tmp_path / 'second_catalog' / relative).read_bytes()
    assert first_bytes != (tmp_path / 'third_catalog' / relative).read_bytes()
    rows = _read_rows(tmp_path / 'first_catalog' / relative)
    assert len(rows) == 8 and len({row['id'] for row in rows}) == 8
    assert not {'B000000000', 'B000000900', 'B000000901', 'B000000902'} & {row['id'] for row in rows}
    assert [row['sampling_rank'] for row in rows] == list(range(1, 9))
    assert all('image_url' in row and 'image' not in row for row in rows)
    descriptions = _read_rows(tmp_path / 'first_catalog/07_shampoo/descriptions.jsonl')
    assert len(descriptions) == 5
    assert all(row['public_source']['description_field'] == 'Description'
               and row['text'] == row['public_source']['description_original_value'] for row in descriptions)
    assert first['formal_memory_opened'] is False and first['private_fields_read'] is False
    assert b'PRIVATE_FIXTURE_SENTINEL' not in first_bytes


def test_catalog_shortage_stops_before_creating_frozen_output(tmp_path):
    catalog, groups = _catalog_fixture(tmp_path, count=3)
    output = tmp_path / 'incomplete_catalog'
    with pytest.raises(ValueError, match='少于所需'):
        public_catalog.prepare_public_catalog(catalog, groups, output, excluded_ids=(),
            count=3, description_count=5, reserve_count=2, max_read_per_group=100)
    assert not output.exists()


def test_description_fallback_retains_original_field_and_formatting_evidence():
    bullets = ['First explicit public feature.', 'Second explicit public feature.']
    description = public_catalog.extract_public_description(dict(Description='', BulletPoints=bullets))
    assert description['text'] == '\n'.join(bullets)
    assert description['field'] == 'BulletPoints' and description['original_value'] == bullets
    assert description['formatting_operation'] == 'public_array_items_joined_by_newline'
    assert public_catalog.extract_public_description(dict(Description='DUMMY_ATTR')) is None


def test_original_description_items_keep_field_indices_and_exact_offsets():
    prose = '  First complete public sentence. Second complete public sentence.  '
    bullets = ['short fixture', 'An original public bullet with enough characters.']
    items = public_catalog.extract_public_description_items(dict(Description=prose, BulletPoints=bullets))
    assert all(len(row['text']) >= 16 for row in items)
    for row in items:
        assert row['original_item'].strip() == row['text']
        if row['field'] == 'Description':
            assert row['original_value'] == prose
            assert prose[row['start_offset']:row['end_offset']] == row['original_item']
        else:
            assert row['field'] == 'BulletPoints' and row['item_index'] == 1
            assert row['original_value'][row['item_index']] == row['original_item']


def test_scarce_class_uses_verbatim_items_with_per_product_cap(tmp_path):
    catalog, _ = _catalog_fixture(tmp_path, count=12)
    with sqlite3.connect(catalog) as db:
        for asin, body in db.execute('SELECT asin,data FROM products').fetchall():
            fields = json.loads(body)
            if asin != 'B000000902':
                fields['name'] = f'Cookie dough fixture product {asin}'
                fields['Description'] = ' '.join(
                    f'Original cookie dough public sentence {index} for fixture {asin}.' for index in range(7))
            db.execute('UPDATE products SET data=?,query=? WHERE asin=?', (json.dumps(fields), 'frozen', asin))
    stat = catalog.stat()
    identity = dict(byte_size=stat.st_size, mtime_ns=stat.st_mtime_ns, snapshot_id='0' * 64)
    group = dict(source={'id': 'B000000900'}, target={'id': 'B000000901'},
                 target_features={'category': 'cookie dough'})
    output = tmp_path / 'scarce_class'
    public_catalog.prepare_public_description_item_group(catalog, '03_cookies', group, output,
        excluded_ids=(), catalog_identity=identity, count=10, description_count=50, max_per_product=5)
    rows = _read_rows(output / 'descriptions.jsonl')
    assert len(rows) == 50 and len({row['text'] for row in rows}) == 50
    assert max(Counter(row['product_id'] for row in rows).values()) <= 5
    assert not {'B000000900', 'B000000901', 'B000000902'} & {row['product_id'] for row in rows}
    for row in rows:
        evidence = row['public_source']
        original = evidence['description_original_value']
        assert original[evidence['description_start_offset']:evidence['description_end_offset']].strip() == row['text']
        assert evidence['description_original_item'].strip() == row['text']
    with pytest.raises(ValueError, match='最多五条'):
        public_catalog.prepare_public_description_item_group(catalog, '03_cookies', group,
            tmp_path / 'invalid_cap', excluded_ids=(), catalog_identity=identity, max_per_product=6)


def _query_snapshot(tmp_path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'explicit_fixture_queries.parquet'
    records = [dict(example_id=index, query=text, query_id=query_id,
                    product_id=product, product_locale=locale, split=split)
               for index, (text, query_id, product, locale, split) in enumerate(rows)]
    pq.write_table(pa.Table.from_pylist(records), path)
    return path


def test_real_query_selector_preserves_original_and_excludes_other_partitions(tmp_path):
    original = '  Gentle SHAMPOO for dry hair  '
    rows = [(original, 1, 'PUBLIC0001', 'us', 'train'),
            ('gentle shampoo for dry hair', 1, 'PUBLIC0002', 'us', 'train'),
            ('dog shampoo', 2, 'PUBLIC0003', 'us', 'train'),
            ('excluded shampoo', 3, 'EXCLUDED01', 'us', 'train'),
            ('test partition shampoo', 4, 'PUBLIC0004', 'us', 'test'),
            ('foreign locale shampoo', 5, 'PUBLIC0005', 'es', 'train')]
    pools, counts = query_tool.collect_queries(_query_snapshot(tmp_path, rows),
                                               excluded_ids=['EXCLUDED01'])
    selected = list(pools['07_shampoo'].values())
    assert len(selected) == 1
    assert selected[0]['text'] == original
    assert [row['product_id'] for row in selected[0]['evidence']] == ['PUBLIC0001', 'PUBLIC0002']
    assert counts['rows_seen'] == 6
    assert counts['public_us_train_rows'] == 4


def test_query_freeze_is_seeded_and_freezes_original_source_identity(tmp_path, monkeypatch):
    from src.experiments import base_workflow

    monkeypatch.setattr(query_tool, 'RULES', {'07_shampoo': query_tool.RULES['07_shampoo']})
    monkeypatch.setattr(query_tool, '_inside_project', lambda path: Path(path).resolve())
    monkeypatch.setattr(base_workflow, 'public_group', lambda *args: {'target_features': {'category': 'shampoo'}})
    snapshot = _query_snapshot(tmp_path, [
        (f'gentle shampoo number {index}', index, f'P{index:09}', 'us', 'train') for index in range(27)])
    first = query_tool.freeze_queries(snapshot, tmp_path / 'first', base=tmp_path, seed=42)
    second = query_tool.freeze_queries(snapshot, tmp_path / 'second', base=tmp_path, seed=42)
    third = query_tool.freeze_queries(snapshot, tmp_path / 'third', base=tmp_path, seed=43)
    group = '07_shampoo'
    assert first['complete'] is True and first['groups'][group]['selected_count'] == 20
    assert first['groups'][group]['sha256'] == second['groups'][group]['sha256']
    assert first['groups'][group]['sha256'] != third['groups'][group]['sha256']
    frozen = [json.loads(line) for line in (tmp_path / 'first' / group / 'heldout_queries.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len({row['text'] for row in frozen}) == 20
    assert all(row['original_split'] == 'train' and row['original_locale'] == 'us'
               and row['split'] == 'query_reserve' and row['source_evidence'] for row in frozen)
    source = json.loads((tmp_path / 'first' / group / 'heldout_source_metadata.json').read_text(encoding='utf-8'))
    assert source['origin_type'] == 'public_real_search_queries'
    assert source['original_snapshot_sha256'] == query_tool.SNAPSHOT_SHA256
    assert source['upstream']['revision'] == query_tool.REVISION
    with pytest.raises(FileExistsError, match='冻结'):
        query_tool.freeze_queries(snapshot, tmp_path / 'first', base=tmp_path, seed=43)


def test_query_shortage_stays_incomplete_and_does_not_duplicate_rows(tmp_path, monkeypatch):
    from src.experiments import base_workflow

    monkeypatch.setattr(query_tool, 'RULES', {'07_shampoo': query_tool.RULES['07_shampoo']})
    monkeypatch.setattr(query_tool, '_inside_project', lambda path: Path(path).resolve())
    monkeypatch.setattr(base_workflow, 'public_group', lambda *args: {'target_features': {'category': 'shampoo'}})
    snapshot = _query_snapshot(tmp_path, [
        (f'gentle shampoo number {index}', index, f'P{index:09}', 'us', 'train') for index in range(19)])
    result = query_tool.freeze_queries(snapshot, tmp_path / 'short', base=tmp_path)
    assert result['complete'] is False
    assert result['groups']['07_shampoo']['selected_count'] == 19
    assert result['groups']['07_shampoo']['complete'] is False
    with pytest.raises(ValueError, match='二十'):
        query_tool.freeze_queries(snapshot, tmp_path / 'other', base=tmp_path, count=19)


def _write_fixture_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def _unrestricted_fixture_rows(path):
    return {str(row['id']): row for row in _read_rows(Path(path))}


def _write_catalog_audit(catalog, group):
    directory = catalog / group
    reserve = directory / 'candidate_reserve.jsonl'
    if not reserve.exists():
        _write_fixture_rows(reserve, [dict(id='explicit_fixture_unused_reserve')])
    (directory / 'sampling_manifest.json').write_text(json.dumps(dict(
        catalog_identity={'sha256': '0' * 64},
        descriptions_sha256=build_tool.sha256_file(directory / 'descriptions.jsonl'),
        candidate_reserve_sha256=build_tool.sha256_file(reserve))), encoding='utf-8')


@pytest.mark.parametrize('existing_environment', [None, 'explicit-fixture-previous-value'])
@pytest.mark.parametrize('service_failure', [False, True])
def test_local_service_key_is_temporary_and_never_saved(tmp_path, monkeypatch, capsys,
                                                       existing_environment, service_failure):
    group = '07_shampoo'
    key = 'explicit-test-fixture-key-never-a-real-credential'
    name = 'VISUAL_INCEPTION_PARAPHRASE_SERVICE_KEY'
    if existing_environment is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, existing_environment)
    monkeypatch.setattr(build_tool, 'PROJECT', tmp_path)
    monkeypatch.setattr(build_tool, 'validate_public_bundle_root', lambda p: Path(p).resolve())
    monkeypatch.setattr(build_tool, 'read_source_rows', _unrestricted_fixture_rows)
    config = tmp_path / 'src/agent/config/fixture.local.json'
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({'model': dict(provider='minimax', model='MiniMax-M3',
        base_url='https://api.minimax.cn/v1', api_key=key)}), encoding='utf-8')
    catalog = tmp_path / 'fixture_catalog'
    _write_fixture_rows(catalog / group / 'descriptions.jsonl', [
        dict(id=f'fixture_{i}', text=f'Fixture public shampoo description {i}', category='shampoo')
        for i in range(50)])
    _write_catalog_audit(catalog, group)
    calls = []

    def generate(rows, **kwargs):
        calls.append(kwargs)
        assert len(rows) == 50 and kwargs['generate'] is True
        assert os.environ[kwargs['api_key_env']] == key
        assert 'api_key' not in kwargs
        if service_failure:
            raise RuntimeError('explicit fixture service failure')
        return dict(rows=[dict(id=f'fixture_query_{i}', text=f'Fixture query {i}') for i in range(30)],
                    generation=dict(model='MiniMax-M3', requested_model='MiniMax-M3', provider='minimax',
                        endpoint='https://api.minimax.cn/v1/chat/completions',
                        input_descriptions_sha256=build_tool.content_sha256(rows)))

    monkeypatch.setattr(build_tool, 'generate_paraphrases', generate)
    inputs = tmp_path / 'fixture_inputs'
    if service_failure:
        with pytest.raises(RuntimeError, match='fixture service failure'):
            build_tool.prepare_paraphrases(group, catalog_root=catalog, inputs_root=inputs,
                                          local_service_config=config)
        assert not (inputs / group / 'paraphrases/model_paraphrases.jsonl').exists()
    else:
        result = build_tool.prepare_paraphrases(group, catalog_root=catalog, inputs_root=inputs,
                                               local_service_config=config)
        assert result == dict(status='generated', count=30, model='MiniMax-M3')
        files = list(inputs.rglob('*'))
        assert not any(key in p.read_text(encoding='utf-8') for p in files if p.is_file())
        reused = build_tool.prepare_paraphrases(group, catalog_root=catalog, inputs_root=inputs,
                                                local_service_config=config)
        assert reused['status'] == 'reused'
        changed = json.loads(config.read_text(encoding='utf-8'))
        changed['model']['model'] = 'MiniMax-M3.1'
        config.write_text(json.dumps(changed), encoding='utf-8')
        with pytest.raises(ValueError, match='模型|地址'):
            build_tool.prepare_paraphrases(group, catalog_root=catalog, inputs_root=inputs,
                                          local_service_config=config)
    assert len(calls) == 1
    assert os.environ.get(name) == existing_environment
    assert key not in capsys.readouterr().out


def test_download_sequence_skips_source_pixels_and_duplicate_images(tmp_path, monkeypatch):
    from PIL import Image

    group = '07_shampoo'
    source, target = tmp_path / 'fixture_source.png', tmp_path / 'fixture_target.png'
    Image.new('RGB', (8, 8), 'black').save(source)
    Image.new('RGB', (8, 8), 'white').save(target)
    monkeypatch.setattr(build_tool, 'validate_public_bundle_root', lambda p: Path(p).resolve())
    monkeypatch.setattr(build_tool, 'read_source_rows', _unrestricted_fixture_rows)
    monkeypatch.setattr(build_tool, '_public', lambda *args: (
        dict(source={'image': str(source)}, target={'image': str(target)}), {'category': 'shampoo'}))
    catalog, inputs = tmp_path / 'fixture_catalog', tmp_path / 'fixture_inputs'
    reserve = [dict(id=f'fixture_{i}', title=f'Public fixture shampoo {i}',
        image_url=f'https://m.media-amazon.com/images/I/fixture{i}.jpg') for i in range(1, 6)]
    _write_fixture_rows(catalog / group / 'candidate_reserve.jsonl', reserve)
    _write_fixture_rows(catalog / group / 'descriptions.jsonl', [dict(id='explicit_fixture_description')])
    _write_catalog_audit(catalog, group)
    calls = []

    def download(product, root):
        calls.append(product['id'])
        colors = dict(fixture_1='black', fixture_2='red', fixture_3='red', fixture_4='blue')
        path = root / 'images' / (product['id'] + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), colors[product['id']]).save(path)
        return dict(image=str(path), image_sha256=build_tool.sha256_file(path))

    monkeypatch.setattr(build_tool, 'download_public_image', download)
    result = build_tool.prepare_images(group, base=tmp_path, catalog_root=catalog, inputs_root=inputs, count=2)
    assert result == dict(status='prepared', count=2, rejected=2)
    assert calls == ['fixture_1', 'fixture_2', 'fixture_3', 'fixture_4']
    selected = _read_rows(inputs / group / 'candidates.jsonl')
    assert [r['id'] for r in selected] == ['fixture_2', 'fixture_4']
    assert [r['reserve_rank'] for r in selected] == [2, 4]
    audit = json.loads((inputs / group / 'candidate_download_audit.json').read_text(encoding='utf-8'))
    assert [r['reason'] for r in audit['failure_records']] == ['duplicate_decoded_image'] * 2
    assert build_tool.prepare_images(group, base=tmp_path, catalog_root=catalog,
                                     inputs_root=inputs, count=2)['status'] == 'reused'
    assert calls == ['fixture_1', 'fixture_2', 'fixture_3', 'fixture_4']
    # The audit itself remains untouched: a changed reserve must not be accepted
    # merely because cached image bytes still match their old manifest.
    reserve[0]['title'] = 'Changed fixture source record'
    (catalog / group / 'candidate_reserve.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in reserve), encoding='utf-8')
    with pytest.raises(ValueError, match='摘要|来源'):
        build_tool.prepare_images(group, base=tmp_path, catalog_root=catalog, inputs_root=inputs, count=2)


def test_freeze_shortage_stops_before_creating_bundle(tmp_path, monkeypatch):
    from src.visual._visual_inception import data_preparation

    group = '07_shampoo'
    catalog, inputs, bundles = [tmp_path / name for name in ('fixture_catalog', 'fixture_inputs', 'fixture_bundles')]
    monkeypatch.setattr(build_tool, 'validate_public_bundle_root', lambda p: Path(p).resolve())
    monkeypatch.setattr(data_preparation, 'validate_public_bundle_root', lambda p, **kw: Path(p).resolve())
    monkeypatch.setattr(data_preparation, 'read_source_rows', _unrestricted_fixture_rows)
    monkeypatch.setattr(build_tool, 'read_source_rows', _unrestricted_fixture_rows)
    monkeypatch.setattr(build_tool, '_verify_candidates', lambda *args: [dict(id=f'c_{i}') for i in range(10)])
    monkeypatch.setattr(build_tool, '_image_binding', lambda *args: {})
    monkeypatch.setattr(build_tool, 'prepare_proxy_memory', lambda **kwargs: dict(status='reused', count=90, group_count=9))
    monkeypatch.setattr(build_tool, '_public', lambda *args: (
        dict(source={'image': 'missing-fixture-source.png'}, target={'image': 'missing-fixture-target.png'}),
        {'category': 'shampoo'}))
    _write_fixture_rows(catalog / group / 'descriptions.jsonl', [dict(id=f'd_{i}') for i in range(49)])
    _write_fixture_rows(inputs / group / 'paraphrases/model_paraphrases.jsonl', [dict(id=f'p_{i}') for i in range(30)])
    _write_fixture_rows(inputs / group / 'heldout_queries.jsonl', [dict(id=f'h_{i}') for i in range(20)])
    _write_fixture_rows(inputs / group / 'candidates.jsonl', [dict(id=f'c_{i}') for i in range(2)])
    _write_catalog_audit(catalog, group)
    (inputs / group / 'paraphrases/generation_provenance.json').write_text(json.dumps(
        dict(prompt_version='explicit_fixture', provider='minimax', model='MiniMax-M3',
            input_descriptions_sha256=build_tool.content_sha256(list(_unrestricted_fixture_rows(
                catalog / group / 'descriptions.jsonl').values())),
            output_sha256=build_tool.sha256_file(inputs / group / 'paraphrases/model_paraphrases.jsonl'))), encoding='utf-8')
    (inputs / group / 'heldout_source_metadata.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='配额不足'):
        build_tool.freeze_group(group, base=tmp_path, catalog_root=catalog, inputs_root=inputs, bundles_root=bundles)
    assert not (bundles / group).exists()


def _shared_proxy_fixture(tmp_path, monkeypatch):
    from PIL import Image

    base, catalog, inputs = [tmp_path / name for name in ('fixture_base', 'fixture_catalog', 'fixture_inputs')]
    base.mkdir()
    groups = ['01_fixture', '02_fixture']
    (base / 'manifest.json').write_text(json.dumps({'groups': dict.fromkeys(groups, {})}), encoding='utf-8')
    public = {}
    monkeypatch.setattr(build_tool, 'validate_public_bundle_root', lambda p: Path(p).resolve())
    monkeypatch.setattr(build_tool, 'read_source_rows', _unrestricted_fixture_rows)
    for index, group in enumerate(groups):
        images = {}
        for role, offset in [('source', 0), ('target', 1)]:
            path = base / f'{group}_{role}.png'
            Image.new('RGB', (8, 8), (index * 40 + offset * 10, 2, 3)).save(path)
            images[role] = dict(id=f'fixture_{group}_{role}', image=str(path))
        public[group] = images
        reserve = [dict(id=f'fixture_{group}_{j}', title=f'Fixture category {index} product {j}',
            image_url=f'https://m.media-amazon.com/images/I/fixture{index}_{j}.jpg') for j in range(2)]
        _write_fixture_rows(catalog / group / 'candidate_reserve.jsonl', reserve)
        _write_fixture_rows(catalog / group / 'descriptions.jsonl', [dict(id=f'fixture_description_{index}')])
        _write_catalog_audit(catalog, group)
        rows = []
        for item, candidate in enumerate(reserve):
            path = inputs / group / 'images' / f'fixture_{item}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (8, 8), (100 + index * 30, 60 + item * 20, 99)).save(path)
            rows.append(dict(candidate, image=path.relative_to(inputs / group).as_posix(),
                image_sha256=build_tool.sha256_file(path),
                decoded_image_sha256=build_tool.decoded_image_sha256(path), category=f'fixture_category_{index}',
                reserve_rank=item + 1))
        _write_fixture_rows(inputs / group / 'candidates.jsonl', rows)
        binding = build_tool._image_binding(images, group, catalog, 2)
        (inputs / group / 'candidate_download_audit.json').write_text(json.dumps(dict(
            input_binding=binding, candidates_sha256=build_tool.sha256_file(inputs / group / 'candidates.jsonl'))),
            encoding='utf-8')
    monkeypatch.setattr(build_tool, '_public', lambda group, _base: (public[group], {'category': group}))
    return base, catalog, inputs, public


def test_proxy_memory_merges_all_categories_and_reuses_one_shared_bank(tmp_path, monkeypatch):
    base, catalog, inputs, _ = _shared_proxy_fixture(tmp_path, monkeypatch)
    result = build_tool.prepare_proxy_memory(base=base, catalog_root=catalog, inputs_root=inputs, count_per_group=2)
    assert result == dict(status='prepared', count=4, group_count=2)
    root = inputs / 'proxy_memory'
    rows = _read_rows(root / 'candidates.jsonl')
    assert Counter(row['gallery_group'] for row in rows) == {'01_fixture': 2, '02_fixture': 2}
    assert len({row['id'] for row in rows}) == 4
    assert len({build_tool.decoded_image_sha256(root / row['image']) for row in rows}) == 4
    assert all(build_tool.sha256_file(root / row['image']) == row['image_sha256'] for row in rows)
    manifest = json.loads((root / 'proxy_memory_manifest.json').read_text(encoding='utf-8'))
    assert manifest['candidate_count'] == 4 and manifest['count_per_category'] == 2
    assert manifest['representation'] == 'frozen_image_only_candidates_shared_across_all_training_groups'
    assert build_tool.prepare_proxy_memory(base=base, catalog_root=catalog, inputs_root=inputs,
        count_per_group=2) == dict(status='reused', count=4, group_count=2)


@pytest.mark.parametrize('duplicate_kind', ['other_candidate', 'other_source', 'other_target'])
def test_proxy_memory_rejects_cross_category_duplicate_or_forbidden_pixels(tmp_path, monkeypatch, duplicate_kind):
    base, catalog, inputs, public = _shared_proxy_fixture(tmp_path, monkeypatch)
    source_group, changed_group = '01_fixture', '02_fixture'
    if duplicate_kind == 'other_candidate':
        duplicate = inputs / source_group / 'images/fixture_0.png'
    else:
        role = 'source' if duplicate_kind == 'other_source' else 'target'
        duplicate = Path(public[source_group][role]['image'])
    changed_image = inputs / changed_group / 'images/fixture_0.png'
    changed_image.write_bytes(duplicate.read_bytes())
    candidate_path = inputs / changed_group / 'candidates.jsonl'
    rows = _read_rows(candidate_path)
    rows[0]['image_sha256'] = build_tool.sha256_file(changed_image)
    rows[0]['decoded_image_sha256'] = build_tool.decoded_image_sha256(changed_image)
    candidate_path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    audit_path = inputs / changed_group / 'candidate_download_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit['candidates_sha256'] = build_tool.sha256_file(candidate_path)
    audit_path.write_text(json.dumps(audit), encoding='utf-8')
    with pytest.raises(ValueError, match='重复|源图|目标图'):
        build_tool.prepare_proxy_memory(base=base, catalog_root=catalog, inputs_root=inputs, count_per_group=2)
    assert not (inputs / 'proxy_memory/proxy_memory_manifest.json').exists()


def test_proxy_memory_does_not_reuse_old_bank_after_source_changes(tmp_path, monkeypatch):
    base, catalog, inputs, _ = _shared_proxy_fixture(tmp_path, monkeypatch)
    build_tool.prepare_proxy_memory(base=base, catalog_root=catalog, inputs_root=inputs, count_per_group=2)
    path = inputs / '01_fixture/candidates.jsonl'
    rows = _read_rows(path)
    rows[0]['title'] = 'Changed fixture public source title'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    audit_path = inputs / '01_fixture/candidate_download_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit['candidates_sha256'] = build_tool.sha256_file(path)
    audit_path.write_text(json.dumps(audit), encoding='utf-8')
    with pytest.raises(ValueError, match='来源改变|旧库'):
        build_tool.prepare_proxy_memory(base=base, catalog_root=catalog, inputs_root=inputs, count_per_group=2)
