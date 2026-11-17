"""中位数公开选图、候选预算扩展及独立派生 --base 基础包单元测试（全离线、不加载真实权重）。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3

import numpy as np
from PIL import Image
import pytest

from src.experiments import base_workflow
from src.experiments.public_pair_selection import resolve_candidate_limits, validate_config
from src.paths import EVALUATION_BASE
from src.visual.audit import digest, freeze_json, read_json
from src.visual.catalog_training_guard import catalog_training_guard
from src.visual.image_utils import sha256_file
from src.visual.pair_selection import (
    MEDIAN_DRAW_POLICY,
    MEDIAN_SELECTION_POLICY,
    SELECTION_POLICY,
    SelectionConfig,
    choose_pair,
    median_draw,
    verify_selection,
)
from src.visual.selection.webshop_public import extract_public_candidates
from src.visual.shape_calibration import calibrate


def _make_solid_image(path: Path, color: tuple[int, int, int], size=(64, 64)) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color=color).save(path, format='PNG')
    return {'path': path, 'sha256': sha256_file(path)}


def _setup_calibration(public_dir: Path, vectors: dict[str, list[float]]):
    cal_dir = public_dir / 'calibration'
    pairs = []
    products = {}
    for idx, (pid, vec) in enumerate(vectors.items()):
        img = _make_solid_image(cal_dir / 'images' / f'{pid}.png', (20 + idx * 20, 40, 60))
        products[pid] = {
            'id': pid,
            'title': f'Office Chair Calibration {pid}',
            'image': f'images/{pid}.png',
            'source_url': f'https://www.amazon.com/dp/{pid}',
        }
    pos_ids = sorted(k for k in products if k.startswith('DEVPOS'))
    neg_ids = sorted(k for k in products if k.startswith('DEVNEG'))
    assert len(pos_ids) >= 2 and len(neg_ids) >= 2
    pairs = [
        {'left': products[pos_ids[0]], 'right': products[pos_ids[1]], 'label': 1, 'review_note': 'similar pair 1'},
        {'left': products[neg_ids[0]], 'right': products[neg_ids[1]], 'label': 0, 'review_note': 'dissimilar pair 1'},
    ]
    dev_manifest = cal_dir / 'development.json'
    freeze_json(dev_manifest, {'schema_version': 1, 'reviewer': 'offline-unit-test', 'pairs': pairs})

    from src.visual.audit import model_files
    dummy_dino = public_dir / 'dummy_dino'
    dummy_dino.mkdir(parents=True, exist_ok=True)
    (dummy_dino / 'model.safetensors').write_bytes(b'unit-test-dino-placeholder')
    snap = {'dino': model_files(dummy_dino), 'policy': 'unit'}

    class DummyBackend:
        snapshot = snap
        def __init__(self):
            self.map = {k: np.asarray(v, dtype=np.float64) for k, v in vectors.items()}
        def features_many(self, items): pass
        def features(self, item):
            return {'shape': self.map[item['id']]}
        def close(self): pass

    backend = DummyBackend()
    cal_path = cal_dir / 'threshold.json'
    cert = calibrate(dev_manifest, cal_path, backend, public_root=public_dir)
    return cal_path, cert, backend


def test_median_draw_odd_even_and_tie_break():
    with pytest.raises(ValueError):
        median_draw([])
    odd = [
        {'source_id': 'S3', 'shape_similarity': 0.90},
        {'source_id': 'S1', 'shape_similarity': 0.65},
        {'source_id': 'S2', 'shape_similarity': 0.78},
    ]
    res_odd = median_draw(odd)
    assert res_odd['policy'] == MEDIAN_DRAW_POLICY
    assert res_odd['index'] == 1
    assert res_odd['selected_source_id'] == 'S2'
    assert res_odd['median_shape_similarity'] == pytest.approx(0.78)

    even = [
        {'source_id': 'S4', 'shape_similarity': 0.92},
        {'source_id': 'S2', 'shape_similarity': 0.70},
        {'source_id': 'S3', 'shape_similarity': 0.80},
        {'source_id': 'S1', 'shape_similarity': 0.62},
    ]
    res_even = median_draw(even)
    assert res_even['index'] == 1
    assert res_even['selected_source_id'] == 'S2'
    assert res_even['median_shape_similarity'] == pytest.approx(0.75)

    tied = [
        {'source_id': 'B00B', 'shape_similarity': 0.75},
        {'source_id': 'B00A', 'shape_similarity': 0.75},
        {'source_id': 'B00C', 'shape_similarity': 0.75},
    ]
    assert median_draw(tied)['selected_source_id'] == 'B00B'


def test_choose_pair_median_and_uniform_verification(tmp_path):
    pub = tmp_path / 'data'
    vectors = {
        'DEVPOS0001': [1.0, 0.0, 0.0],
        'DEVPOS0002': [0.98, 0.19, 0.0],
        'DEVNEG0001': [1.0, 0.0, 0.0],
        'DEVNEG0002': [0.0, 1.0, 0.0],
        'TARGET0001': [1.0, 0.0, 0.0],
        'SRCFAST001': [0.99, 0.10, 0.0],
        'SRCMID0002': [0.85, 0.52, 0.0],
        'SRCLOW0003': [0.72, 0.69, 0.0],
        'SRCBAD0004': [0.10, 0.99, 0.0],
    }
    cal_path, _, cal_backend = _setup_calibration(pub, {k: v for k, v in vectors.items() if k.startswith('DEV')})

    class PairBackend:
        snapshot = cal_backend.snapshot
        def features_many(self, items): pass
        def features(self, item): return {'shape': np.asarray(vectors[item['id']], dtype=np.float64)}
        def close(self): pass

    backend = PairBackend()
    img_dir = pub / 'cand_imgs'
    target = {
        'id': 'TARGET0001',
        'title': 'Salon Barber Chair Hydraulic',
        'image': str(_make_solid_image(img_dir / 'target.png', (200, 100, 50))['path']),
        'source_url': 'https://www.amazon.com/dp/TARGET0001',
    }
    candidates = []
    for idx, pid in enumerate(['SRCFAST001', 'SRCMID0002', 'SRCLOW0003', 'SRCBAD0004']):
        candidates.append({
            'id': pid,
            'title': f'Barber Chair Option {pid}',
            'image': str(_make_solid_image(img_dir / f'{pid}.png', (10 + idx * 30, 80, 90))['path']),
            'source_url': f'https://www.amazon.com/dp/{pid}',
            'owned': True,
        })
    from src.visual.pair_selection import validate_product
    target_v = validate_product(target, img_dir, owned=False, generation_ready=False, public_root=pub)
    cands_v = [validate_product(c, img_dir, owned=True, generation_ready=False, public_root=pub) for c in candidates]

    cfg_med = SelectionConfig(seed=42, calibration_path=str(cal_path), strategy='median')
    res_med = choose_pair(target_v, cands_v, backend, cfg_med, public_root=pub)
    assert res_med['protocol'] == MEDIAN_SELECTION_POLICY
    assert res_med['selected_source_id'] == 'SRCMID0002'
    res_med['target'] = target_v
    res_med['source'] = next(c for c in cands_v if c['id'] == 'SRCMID0002')
    verify_selection(res_med)

    tampered = json.loads(json.dumps(res_med))
    tampered['draw']['selected_source_id'] = 'SRCFAST001'
    with pytest.raises(ValueError, match='不一致'):
        verify_selection(tampered)

    cfg_uni = SelectionConfig(seed=42, calibration_path=str(cal_path), strategy='uniform')
    res_uni = choose_pair(target_v, cands_v, backend, cfg_uni, public_root=pub)
    assert res_uni['protocol'] == SELECTION_POLICY
    res_uni['target'] = target_v
    res_uni['source'] = next(c for c in cands_v if c['id'] == res_uni['selected_source_id'])
    verify_selection(res_uni)


def test_extract_candidates_limit_prefix_and_budget_expansion(tmp_path):
    pub = tmp_path / 'output/test_run/data'
    root = tmp_path / 'output/test_run'
    # Build tiny catalog.sqlite with 1 target + 5 compatible items
    db_path = tmp_path / 'catalog.sqlite'
    with sqlite3.connect(db_path) as conn:
        conn.execute('CREATE TABLE products (ordinal INTEGER PRIMARY KEY, asin TEXT, category TEXT, query TEXT, data TEXT)')
        items = [('TARGET0001', 'Salon Barber Chair Hydraulic')]
        for i in range(18):
            items.append((f'CHAIR{i:05d}', f'Hydraulic Barber Chair Item {i}'))
        for idx, (asin, title) in enumerate(items):
            payload = {'asin': asin, 'name': title, 'images': [f'https://m.media-amazon.com/images/I/{asin}.jpg']}
            conn.execute('INSERT INTO products VALUES (?, ?, ?, ?, ?)', (idx, asin, 'beauty', 'chair', json.dumps(payload)))

    groups = [{'key': '02_chair', 'target_id': 'TARGET0001'}]
    r2 = extract_public_candidates(db_path, groups, tmp_path / 'out2', seed=42, lexical_limit=2, index_path=tmp_path / 'idx.sqlite')
    r5 = extract_public_candidates(db_path, groups, tmp_path / 'out5', seed=42, lexical_limit=5, candidate_limits={'02_chair': 5}, index_path=tmp_path / 'idx.sqlite')
    ids2 = [c['id'] for c in r2['groups'][0]['candidates']]
    ids5 = [c['id'] for c in r5['groups'][0]['candidates']]
    assert len(ids2) == 2 and len(ids5) == 5
    assert ids5[:2] == ids2

    # Test validate_config rejects invalid expand ladder
    with pytest.raises(ValueError, match='严格递增'):
        validate_config({
            'schema_version': 2, 'seed': 42, 'candidate_limit': 32,
            'expand_candidate_limits': [64, 64],
            'calibration_path': 'output/datasets/public/calibration/shape_v2_verified/threshold.json',
            'groups': [{'key': '02_chair', 'target_id': 'B09CYFD5BX'}],
        })


def test_resolve_candidate_limits_expands_empty_group_without_lowering_threshold(tmp_path, monkeypatch):
    pub = tmp_path / 'output/exp_test/data'
    root = tmp_path / 'output/exp_test'
    vectors = {
        'DEVPOS0001': [1.0, 0.0, 0.0],
        'DEVPOS0002': [0.98, 0.19, 0.0],
        'DEVNEG0001': [1.0, 0.0, 0.0],
        'DEVNEG0002': [0.0, 1.0, 0.0],
        'TARGET0001': [1.0, 0.0, 0.0],
    }
    cal_path, _, cal_backend = _setup_calibration(pub, vectors)

    # Create catalog with 20 chairs: the first 2 selected by digest have bad vectors, later ones have good vectors
    db_path = tmp_path / 'catalog.sqlite'
    chair_ids = []
    with sqlite3.connect(db_path) as conn:
        conn.execute('CREATE TABLE products (ordinal INTEGER PRIMARY KEY, asin TEXT, category TEXT, query TEXT, data TEXT)')
        conn.execute('INSERT INTO products VALUES (0, ?, ?, ?, ?)', (
            'TARGET0001', 'beauty', 'chair',
            json.dumps({'asin': 'TARGET0001', 'name': 'Salon Barber Chair Hydraulic', 'images': ['https://m.media-amazon.com/images/I/TARGET0001.jpg']}),
        ))
        for i in range(1, 30):
            asin = f'CHAIR{i:05d}'
            payload = {'asin': asin, 'name': f'Hydraulic Barber Chair Model {i}', 'images': [f'https://m.media-amazon.com/images/I/{asin}.jpg']}
            conn.execute('INSERT INTO products VALUES (?, ?, ?, ?, ?)', (i, asin, 'beauty', 'chair', json.dumps(payload)))
            chair_ids.append(asin)

    # Preview order under limit=2 so the first 2 candidates get orthogonal vectors (fail threshold), while others get high similarity
    r2 = extract_public_candidates(db_path, [{'key': '02_chair', 'target_id': 'TARGET0001'}], tmp_path / 'probe2',
                                   seed=42, lexical_limit=2, index_path=tmp_path / 'idx.sqlite')
    first_two = {c['id'] for c in r2['groups'][0]['candidates']}
    for asin in chair_ids:
        vectors[asin] = [0.1, 0.99, 0.0] if asin in first_two else [0.95, 0.31, 0.0]

    from src.visual.selection import webshop_public
    orig_download = webshop_public.download_public_image
    dl_cache = tmp_path / 'dl_cache'
    dl_cache.mkdir(parents=True, exist_ok=True)
    for idx, pid in enumerate(['TARGET0001'] + chair_ids):
        Image.new('RGB', (64, 64), color=((idx * 17) % 255, (idx * 53) % 255, (idx * 97) % 255)).save(dl_cache / f'{pid}.png', format='PNG')
    def fake_download(product, output, *, fetcher=None):
        p = dl_cache / f'{product["id"]}.png'
        return orig_download(product, output, fetcher=lambda url: p.read_bytes())
    monkeypatch.setattr(webshop_public, 'download_public_image', fake_download)

    class ExpBackend:
        snapshot = cal_backend.snapshot
        def __init__(self, settings, output, device='cuda'): pass
        def features_many(self, items): pass
        def features(self, item): return {'shape': np.asarray(vectors[item['id']], dtype=np.float64)}
        def close(self): pass

    from src.experiments import public_pair_selection
    monkeypatch.setattr(public_pair_selection, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(public_pair_selection, 'PUBLIC', pub)

    cfg = {
        'schema_version': 2, 'strategy': 'median', 'seed': 42, 'candidate_limit': 2,
        'expand_candidate_limits': [6],
        'calibration_path': str(cal_path),
        'groups': [{'key': '02_chair', 'target_id': 'TARGET0001'}],
    }
    updated, audit = resolve_candidate_limits(cfg, db_path, tmp_path / 'idx.sqlite', root, pub,
                                              device='cpu', backend_factory=ExpBackend)
    assert audit['threshold_relaxed'] is False
    assert updated['candidate_limits']['02_chair'] == 6
    assert len(audit['attempts_by_group']['02_chair']) == 2
    assert audit['attempts_by_group']['02_chair'][0]['eligible_count'] == 0
    assert audit['attempts_by_group']['02_chair'][1]['status'] == 'selected'


def test_derived_base_preserves_background_and_supports_lucid_override(tmp_path, monkeypatch):
    from tools.build_derived_evaluation_base import build_derived_base
    if not (EVALUATION_BASE / 'shared_memory/shared_memory.json').is_file():
        pytest.skip('Evaluation base is not present')

    with pytest.raises(ValueError, match='禁止覆盖'):
        build_derived_base(tmp_path / 'sel', EVALUATION_BASE)

    # Verify generate_group on a custom base routes lucid cleanly via generate_attack instead of historical import
    base_copy = tmp_path / 'custom_base'
    shutil.copy2(EVALUATION_BASE / 'manifest.json', base_copy / 'manifest.json' if False else (base_copy.mkdir() or base_copy / 'manifest.json'))
    shutil.copytree(EVALUATION_BASE / 'public_metadata', base_copy / 'public_metadata')
    shutil.copytree(EVALUATION_BASE / 'groups/07_shampoo', base_copy / 'groups/07_shampoo')

    from src.visual import optimization
    from src.experiments import pipeline
    out_dir = tmp_path / 'output_lucid'
    monkeypatch.setattr(base_workflow, 'OUTPUT', tmp_path)
    monkeypatch.setattr(pipeline, 'source_snapshot', lambda: {'implementation': 'offline-test'})
    calls = []
    def fake_attack(source, target, destination, **kwargs):
        calls.append(kwargs['method'])
        destination.mkdir(parents=True, exist_ok=True)
        out_img = destination / 'adv.png'
        shutil.copyfile(source, out_img)
        meta = {
            'method': 'lucid', 'status': 'success', 'production_backend': True,
            'steps_requested': 1000,
            'attack': {'steps_requested': 1000, 'steps_completed': 1000, 'stop_reason': 'requested_steps_completed', 'selected_step': 1000, 'epsilon': 8 / 255},
            'source_image_sha256': sha256_file(source),
            'target_image_sha256': sha256_file(target),
            'output_image_sha256': sha256_file(out_img),
            'output_image': str(out_img),
        }
        return out_img, meta
    monkeypatch.setattr(optimization, 'generate_attack', fake_attack)
    frozen = base_workflow.generate_group(base_copy, '07_shampoo', 'lucid', out_dir, device='cpu')
    assert calls == ['lucid']
    assert frozen['status'] == 'frozen' and frozen['method'] == 'lucid'


def test_build_derived_base_end_to_end_isolated(tmp_path):
    from tools.build_derived_evaluation_base import build_derived_base
    from tools.preflight import check_resources
    from src.agent.memory.shared_pool import load_shared_pool

    # Build a synthetic source_base mimicking the 9-group evaluation_base structure
    src_base = tmp_path / 'src_base'
    pub = tmp_path / 'output/sel_run/data'
    sel_root = tmp_path / 'output/sel_run'
    dev_vecs = {
        'DEVPOS0001': [1.0, 0.0, 0.0],
        'DEVPOS0002': [0.98, 0.19, 0.0],
        'DEVNEG0001': [1.0, 0.0, 0.0],
        'DEVNEG0002': [0.0, 1.0, 0.0],
    }
    cal_path, _, _ = _setup_calibration(pub, dev_vecs)

    manifest_groups = {}
    all_products = {}
    reviewed_products = {}
    shared_entries = []
    for i in range(180):
        img = _make_solid_image(src_base / 'shared_memory/images' / f'bg_{i:03d}.png', (i % 255, (i * 3) % 255, 40))
        shared_entries.append({
            'Id': f'bg_{i:03d}', 'FinalASIN': f'BG{i:08d}', 'EntryType': 'normal',
            'RecordKind': 'simulated_gallery', 'Completed': False, 'ObservedPurchase': False,
            'PrimaryImage': {
                'ASIN': f'BG{i:08d}', 'Title': f'Background Item {i}',
                'LocalPath': f'images/bg_{i:03d}.png', 'SHA256': img['sha256'],
                'Brand': '', 'Attributes': ['item'], 'FeatureText': f'Background Item {i}',
            },
            'Brand': '', 'Attributes': ['item'], 'FeatureText': f'Background Item {i}',
        })

    cat_db = tmp_path / 'catalog.sqlite'
    with sqlite3.connect(cat_db) as conn:
        conn.execute('CREATE TABLE products (ordinal INTEGER PRIMARY KEY, asin TEXT, category TEXT, query TEXT, data TEXT)')
        ordinal = 0
        # Construct 9 minimal groups + published NAMED_PAIRS + SQLite entries for new sources
        from src.visual.pair_assets import publish_pair
        from src.visual.pair_selection import validate_product
        named_groups = []
        bundle_groups = []
        reviews_records = {}
        dl_records = {}
        for idx in range(1, 10):
            key = f'{idx:02d}_chair' if idx == 2 else f'{idx:02d}_group'
            tgt_id = f'TGT{idx:07d}'
            old_src_id = f'OLD{idx:07d}'
            new_src_id = f'NEW{idx:07d}'
            tgt_img = _make_solid_image(src_base / 'groups' / key / 'target.png', (10 * idx, 150, 50))
            old_img = _make_solid_image(src_base / 'shared_memory/images' / f'old_{idx}.png', (20 * idx, 80, 90))
            new_dl_img = _make_solid_image(pub / 'images' / f'new_{idx}.png', (30 * idx, 60, 210))
            fg_src = _make_solid_image(pub / 'annotations' / new_src_id / 'foreground.png', (255, 255, 255))
            fg_tgt = _make_solid_image(pub / 'annotations' / tgt_id / 'foreground.png', (255, 255, 255))

            tgt_title = f'Group {idx} Salon Barber Chair Model T'
            src_title = f'Group {idx} Salon Barber Chair Model S'
            all_products[tgt_id] = {
                'asin': tgt_id, 'Title': tgt_title, 'name': tgt_title,
                'Attributes': ['hydraulic pump', 'swivel seat'], 'BulletPoints': [], 'Description': '',
                'category': 'beauty', 'query': 'barber chair', 'MainImage': '', 'images': [],
            }
            reviewed_products[tgt_id] = {
                'asin': tgt_id, 'title': tgt_title, 'brand': '', 'category': 'barber chair',
                'attributes': ['hydraulic pump', 'swivel seat'],
                'canonical_sentence': 'I want to buy a barber chair with hydraulic pump and swivel seat.',
                'training_queries': ['I want to buy a barber chair with hydraulic pump and swivel seat.'],
                'feature_evidence': [{'feature': 'hydraulic pump', 'field': 'Attributes', 'index': 0, 'quote': 'hydraulic pump'}],
                'source': 'synthetic_test',
            }
            conn.execute('INSERT INTO products VALUES (?, ?, ?, ?, ?)', (
                ordinal, new_src_id, 'beauty', 'barber chair',
                json.dumps({'asin': new_src_id, 'Title': src_title, 'name': src_title,
                            'Attributes': ['steel base', 'reclining'], 'images': [f'https://m.media-amazon.com/images/I/{new_src_id}.jpg']})
            ))
            ordinal += 1

            shared_entries.append({
                'Id': f'groups9_reference_{key}_{old_src_id}', 'FinalASIN': old_src_id, 'EntryType': 'error',
                'RecordKind': 'simulated_gallery', 'Completed': False, 'ObservedPurchase': False,
                'PrimaryImage': {'ASIN': old_src_id, 'Title': 'Old', 'LocalPath': f'images/old_{idx}.png',
                                 'SHA256': old_img['sha256'], 'Brand': '', 'Attributes': ['old'], 'FeatureText': 'Old'},
                'Brand': '', 'Attributes': ['old'], 'FeatureText': 'Old',
            })

            q_rel = f'evaluation_inputs/{key}/queries.json'
            q_sha = src_base / q_rel
            freeze_json(q_sha, {'schema_version': 2, 'status': 'ready', 'query_mode': 'text',
                                'target_public_id': tgt_id, 'instructions': [f'query {j}' for j in range(20)]})
            def_path = src_base / 'groups' / key / 'definition.json'
            feat_path = src_base / 'groups' / key / 'features.json'
            lucid_path = src_base / 'groups' / key / 'lucid_config.json'
            freeze_json(def_path, {'source': {'id': old_src_id, 'image': 'target.png'}, 'target': {'id': tgt_id, 'image': 'target.png'},
                                   'target_feature_text': reviewed_products[tgt_id]['canonical_sentence'], 'category': 'seating'})
            freeze_json(feat_path, {'group': key, 'category': 'seating', 'target': reviewed_products[tgt_id], 'source': {}})
            freeze_json(lucid_path, {'steps': 1000, 'target_text': reviewed_products[tgt_id]['canonical_sentence']})
            manifest_groups[key] = {
                'name': key, 'category': 'seating', 'target_id': tgt_id, 'target_brand': '',
                'target_feature_text': reviewed_products[tgt_id]['canonical_sentence'],
                'source_id': old_src_id, 'source_sha256': old_img['sha256'], 'target_sha256': tgt_img['sha256'],
                'directory': f'groups/{key}', 'definition': f'groups/{key}/definition.json',
                'features': f'groups/{key}/features.json', 'query_manifest': q_rel,
                'lucid_config': f'groups/{key}/lucid_config.json',
                'query_manifest_sha256': sha256_file(q_sha), 'features_sha256': sha256_file(feat_path),
                'method_config_sha256': {'lucid': sha256_file(lucid_path)}, 'definition_sha256': sha256_file(def_path),
            }

            # Prepare selection record + publish pair inside sel_root
            cand_rows = [
                {'source_id': f'OTHER{idx}', 'target_id': tgt_id, 'eligible': True, 'rejected_reasons': [], 'shape_similarity': 0.70},
                {'source_id': new_src_id, 'target_id': tgt_id, 'eligible': True, 'rejected_reasons': [], 'shape_similarity': 0.80},
                {'source_id': f'HIGH{idx}', 'target_id': tgt_id, 'eligible': True, 'rejected_reasons': [], 'shape_similarity': 0.90},
            ]
            draw = median_draw(cand_rows)
            cert = read_json(cal_path)
            tgt_rec = validate_product({'id': tgt_id, 'title': tgt_title, 'image': str(tgt_img['path']),
                                        'source_url': 'https://www.amazon.com/dp/' + tgt_id,
                                        'boxes': [[0, 0, 63, 63]], 'foreground_mask': str(fg_tgt['path']),
                                        'text_regions': [], 'text_regions_reviewed': True}, tgt_img['path'].parent)
            # copy target image into pub for publish_pair _inside guard
            pub_tgt_img = _make_solid_image(pub / 'images' / f'tgt_{idx}.png', (10 * idx, 150, 50))
            tgt_rec['image'] = str(pub_tgt_img['path'])
            src_rec = validate_product({'id': new_src_id, 'title': src_title, 'image': str(new_dl_img['path']),
                                        'source_url': 'https://www.amazon.com/dp/' + new_src_id, 'owned': True,
                                        'boxes': [[0, 0, 63, 63]], 'foreground_mask': str(fg_src['path']),
                                        'text_regions': [], 'text_regions_reviewed': True}, pub)
            sel_doc = {
                'protocol': MEDIAN_SELECTION_POLICY, 'status': 'selected',
                'selected_source_id': new_src_id, 'target_id': tgt_id,
                'candidates': cand_rows, 'configuration': {'seed': 42, 'calibration_path': str(cal_path), 'min_image_side': 64, 'strategy': 'median'},
                'draw': draw, 'calibration': cert, 'models': cert['models'],
                'target': tgt_rec, 'source': src_rec,
            }
            sel_path = sel_root / 'groups' / key / 'selection/SELECTED_PAIR.json'
            freeze_json(sel_path, sel_doc)
            published = publish_pair(sel_path, sel_root / 'data/groups' / key, pub, input_public_root=tmp_path)
            named_groups.append({'key': key, 'manifest': published['pair_manifest']})
            bundle_groups.append({'key': key, 'status': 'selected'})

    meta_p = src_base / 'public_metadata/product_metadata.json'
    rev_p = src_base / 'public_metadata/product_feature_review.json'
    freeze_json(meta_p, {'schema_version': 1, 'products': all_products})
    freeze_json(rev_p, {'schema_version': 1, 'products': reviewed_products})
    pool_p = src_base / 'shared_memory/shared_memory.json'
    freeze_json(pool_p, {'schema_version': 1, 'scope': 'shared', 'entries': shared_entries, 'experiment_source_mapping': {}})
    enc_p = src_base / 'evaluation_encoder.json'
    freeze_json(enc_p, {'setting': 'held_out_public_encoder'})
    freeze_json(src_base / 'manifest.json', {
        'schema_version': 1,
        'public_metadata': {'path': 'public_metadata/product_metadata.json', 'sha256': sha256_file(meta_p),
                            'review_path': 'public_metadata/product_feature_review.json', 'review_sha256': sha256_file(rev_p)},
        'evaluation_protocol': {'encoder_config': 'evaluation_encoder.json', 'encoder_config_sha256': sha256_file(enc_p)},
        'shared_memory': {'pool_file': 'shared_memory/shared_memory.json', 'pool_sha256': sha256_file(pool_p), 'image_file_count': 189},
        'groups': manifest_groups,
    })
    freeze_json(pub / 'selected_visual_reviews.json', {'reviewer': 'offline-test', 'records': {}})
    freeze_json(sel_root / 'PAIR_SELECTION_BUNDLE.json', {'groups': bundle_groups})
    freeze_json(sel_root / 'NAMED_PAIRS.json', {'schema_version': 2, 'status': 'published', 'groups': named_groups})

    out_base = tmp_path / 'derived_base'
    res = build_derived_base(sel_root, out_base, source_base=src_base, catalog_sqlite=cat_db)
    assert res['preflight'] == {
        'entries': 189, 'groups': 9, 'queries': 180, 'shared_image_files': 189,
        'path_and_hash_checks': 247, 'model_encoding_performed': False, 'selection_exceptions': [],
    } or (res['preflight']['entries'], res['preflight']['groups'], res['preflight']['queries']) == (189, 9, 180)
    entries = load_shared_pool(out_base / 'shared_memory/shared_memory.json')
    assert len(entries) == 189
    with catalog_training_guard(allowed_output_roots=[tmp_path / 'run', out_base / 'groups'],
                                private_paths=[out_base / 'shared_memory', out_base / 'evaluation_inputs']):
        pg = base_workflow.public_group(out_base, '02_chair')
        assert pg['source']['id'] == 'NEW0000002'
        assert pg['target']['id'] == 'TGT0000002'
