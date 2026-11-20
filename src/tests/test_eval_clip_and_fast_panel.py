"""Unit tests for StandaloneEvalClipEncoder, BatchedFreshCachedEncoder, and fast_stage2_panel."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.visual.image_utils import sha256_file


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def test_eval_clip_is_decoupled_from_openclip_surrogate():
    source_path = Path('/content/AmbiLAGE/src/visual/encoders/eval_clip.py')
    text = source_path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    imported_names = set()
    attr_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.Attribute):
            attr_names.add(node.attr)
    assert 'OpenCLIPSurrogate' not in imported_names
    assert 'forward_intermediates' not in attr_names


def test_batched_fresh_cached_encoder_prewarm_and_mixed_sizes(tmp_path):
    from src.visual.encoders.eval_clip import BatchedFreshCachedEncoder

    img_a = tmp_path / 'a_rect.png'
    img_b = tmp_path / 'b_square.png'
    img_a_dup = tmp_path / 'a_rect_copy.png'
    Image.new('RGB', (14, 9), (200, 30, 50)).save(img_a)
    Image.new('RGB', (10, 10), (30, 180, 70)).save(img_b)
    Image.new('RGB', (14, 9), (200, 30, 50)).save(img_a_dup)

    class _StubBatchEncoder:
        def __init__(self):
            self.snapshot = dict(provider='clip', loader='stub_eval', precision='float32')
            self.supports_image_weight = True
            self.batch_calls = []
            self.single_calls = []

        def _vec(self, p):
            with Image.open(p) as im:
                arr = np.asarray(im.convert('RGB'), dtype=np.float32)
            v = np.asarray([arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean(), float(arr.shape[0])], dtype=np.float32)
            return v / np.linalg.norm(v)

        def encode(self, path):
            self.single_calls.append(str(Path(path).resolve()))
            return self._vec(path)

        def encode_batch(self, paths, *, batch_size=None):
            resolved = [str(Path(p).resolve()) for p in paths]
            self.batch_calls.append((batch_size, resolved))
            return [self._vec(p) for p in resolved]

        def encode_text(self, text):
            v = np.asarray([1.0, 0.5, 0.25, 0.0], dtype=np.float32)
            return v / np.linalg.norm(v)

        def close(self):
            return None

    stub = _StubBatchEncoder()
    wrapper = BatchedFreshCachedEncoder(stub)
    pairs = [
        (img_a, sha256_file(img_a)),
        (img_b, sha256_file(img_b)),
        (img_a_dup, sha256_file(img_a_dup)),
    ]
    count = wrapper.prewarm_images(pairs, batch_size=2)
    assert count == 2
    assert len(stub.batch_calls) == 1
    assert stub.batch_calls[0][0] == 2
    assert stub.single_calls == []

    # Subsequent encode() must hit RAM cache with zero extra encoder calls and match single encode() bit-for-bit.
    for p in (img_a, img_b, img_a_dup):
        got = wrapper.encode(p)
        expected = stub._vec(p)
        assert np.allclose(got, expected, atol=1e-12)
    assert stub.single_calls == []


def test_fast_stage2_panel_end_to_end_and_skip_completed(tmp_path, monkeypatch):
    from src.experiments import fast_stage2_panel as fast_panel
    from src.experiments import shared_evaluation
    from src.experiments import visual_inception_category_evaluation as category
    from src.tests.test_evaluation_integrity import _DummyEncoder
    from src.tests.test_visual_inception_category_evaluation import (
        GROUPS,
        _tiny_frozen_suite,
        _tiny_oracle_query_root,
    )
    import src.visual.attack_completion as completion

    base, images, publics, _ = _tiny_frozen_suite(tmp_path / 'work')
    entries = []
    source_mapping = {}
    for index, group in enumerate(GROUPS):
        src = publics[group]['source']
        mem_id = f'fixture_memory_{index}'
        source_mapping[group] = {'source_asin': src['id'], 'source_memory_id': mem_id}
        entries.append(dict(
            Id=mem_id,
            FinalASIN=src['id'],
            RecordKind='simulated_gallery',
            Completed=False,
            PrimaryImage=dict(
                ASIN=src['id'],
                Title=src['title'],
                LocalPath=src['image'],
                SHA256=src['image_sha256'],
            ),
            FeatureText=f'Fixture feature {index}',
            Attributes=[f'attr_{index}'],
        ))
    pool_path = base / 'shared_memory/shared_memory.json'
    _write(pool_path, dict(schema_version=1, scope='shared', entries=entries))
    _write(base / 'manifest.json', dict(
        groups={g: dict(target_id=publics[g]['target']['id'], source_id=publics[g]['source']['id']) for g in GROUPS},
        shared_memory=dict(
            pool_file='shared_memory/shared_memory.json',
            pool_sha256=sha256_file(pool_path),
            source_mapping=source_mapping,
        ),
    ))
    oracle_root = _tiny_oracle_query_root(tmp_path / 'work', base / 'manifest.json', publics)

    # Create a tiny fake VisualEncoder config file so encoder_config paths resolve on disk.
    fake_cfg_dir = tmp_path / 'work' / 'configs'
    for _, _, _, m_cfg, h_cfg in category.TRANSFER_PANEL_ENCODERS:
        for fn in (m_cfg, h_cfg):
            _write(fake_cfg_dir / fn, dict(setting='held_out_public_encoder', surrogate=dict(name='dummy', pretrained='dummy')))

    out_root = tmp_path / 'work' / 'output'
    monkeypatch.setattr(category, 'OUTPUT', out_root)
    monkeypatch.setattr(fast_panel, 'OUTPUT', out_root)
    monkeypatch.setattr(category, 'VISUAL_CONFIGS', fake_cfg_dir)
    monkeypatch.setitem(category.QUERY_SUITE_ROOTS, 'training_oracle', oracle_root)
    monkeypatch.setattr(category, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    monkeypatch.setattr(category, 'validate_attack_completion', lambda *args, **kwargs: None)

    done_dir = out_root / 'already_done'
    sentinel = dict(preserved=True, schema_version=1)
    _write(done_dir / 'transfer_panel_summary.json', sentinel)

    pending_dir = out_root / 'vi_oracle_fast'
    jobs = [
        dict(name='already_done', images_root=str(images), method='visual_inception',
             query_suite='training_oracle', output=str(done_dir)),
        dict(name='vi_oracle_fast', images_root=str(images), method='visual_inception',
             query_suite='training_oracle', output=str(pending_dir)),
    ]

    result = fast_panel.run_encoder_grouped_panels(
        jobs,
        base=base,
        key_mode='image',
        defense_mode='none',
        defended_cache_dir=out_root / 'defended_cache',
        memory_vector_cache_dir=out_root / 'memory_vectors',
        batch_sizes={'01_clip_l14': 2, '02_openclip_bigG14': 2, '03_siglip_so400m': 2},
        device='cpu',
        verify_training_snapshots=False,
        encoder_factory=_DummyEncoder,
        isolated_process=False,
    )

    assert json.loads((done_dir / 'transfer_panel_summary.json').read_text(encoding='utf-8')) == sentinel
    summary = result['vi_oracle_fast']
    assert summary['query_suite'] == 'training_oracle'
    assert summary['defense_mode'] == 'none'
    assert len(summary['encoder_order']) == 3
    for slug in summary['encoder_order']:
        ver = summary['encoders'][slug]['verification']
        assert ver['status'] == 'passed'
        assert ver['all_candidate_scores_checked'] is True
        assert float(ver['max_score_error']) <= shared_evaluation.SCORE_TOLERANCE
    assert (pending_dir / 'TRANSFER_PANEL_REPORT.md').is_file()


def test_batched_fresh_cached_encoder_prewarm_texts():
    from src.visual.encoders.eval_clip import BatchedFreshCachedEncoder

    class _StubTextBatchEncoder:
        def __init__(self):
            self.snapshot = dict(provider='clip', loader='stub_eval', precision='float32')
            self.supports_image_weight = True
            self.text_batch_calls = []
            self.single_text_calls = []

        def _vec(self, text):
            digest = hashlib.sha256(str(text or ' ').encode('utf-8')).digest()
            v = np.asarray(list(digest[:8]), dtype=np.float32) + 1.0
            return v / np.linalg.norm(v)

        def encode(self, path):
            return self._vec(str(path))

        def encode_text(self, text):
            self.single_text_calls.append(text)
            self.last_encoding_audit = {'text': text, 'chunks': 1}
            return self._vec(text)

        def encode_text_batch(self, texts, *, batch_size=64):
            text_list = list(texts)
            self.text_batch_calls.append((batch_size, text_list))
            return [(self._vec(t), {'text': t, 'chunks': 1}) for t in text_list]

        def close(self):
            return None

    stub = _StubTextBatchEncoder()
    wrapper = BatchedFreshCachedEncoder(stub)
    count = wrapper.prewarm_texts(['beta query', 'alpha query', 'beta query', ''], batch_size=2)
    assert count == 3
    assert len(stub.text_batch_calls) == 1
    assert stub.text_batch_calls[0] == (2, [' ', 'alpha query', 'beta query'])
    assert stub.single_text_calls == []

    for q in ('beta query', 'alpha query', ''):
        got = wrapper.encode_text(q)
        expected = stub._vec(q)
        assert np.allclose(got, expected, atol=1e-12)
        assert wrapper.last_encoding_audit == {'text': str(q or ' '), 'chunks': 1}
    assert stub.single_text_calls == []


def test_sha256_file_cache_hits_and_invalidates_on_disk_change(tmp_path, monkeypatch):
    import src.visual.io as visual_io
    from src.visual.image_utils import sha256_file as utils_sha256_file

    visual_io._SHA256_FILE_CACHE.clear()
    big_file = tmp_path / 'large_asset.bin'
    payload_a = b'A' * 70000
    payload_b = b'B' * 70000
    big_file.write_bytes(payload_a)

    expected_a = hashlib.sha256(payload_a).hexdigest()
    expected_b = hashlib.sha256(payload_b).hexdigest()

    sha256_calls = 0
    real_sha256 = hashlib.sha256

    def _counting_sha256(*args, **kwargs):
        nonlocal sha256_calls
        sha256_calls += 1
        return real_sha256(*args, **kwargs)

    monkeypatch.setattr(visual_io.hashlib, 'sha256', _counting_sha256)

    assert visual_io.sha256_file(big_file) == expected_a
    assert sha256_calls == 1
    # Repeated calls via both io.sha256_file and image_utils.sha256_file must hit the stat-bound cache.
    assert visual_io.sha256_file(big_file) == expected_a
    assert utils_sha256_file(big_file) == expected_a
    assert sha256_calls == 1

    # Sleep briefly past kernel mtime tick so overwriting same-size file updates stat cleanly.
    time.sleep(0.02)
    big_file.write_bytes(payload_b)
    assert visual_io.sha256_file(big_file) == expected_b
    assert utils_sha256_file(big_file) == expected_b
    assert sha256_calls == 2


def test_run_encoder_grouped_panels_in_process_selected_groups(tmp_path, monkeypatch):
    from src.experiments import fast_stage2_panel as fast_panel
    from src.experiments import shared_evaluation
    from src.experiments import visual_inception_category_evaluation as category
    from src.tests.test_evaluation_integrity import _DummyEncoder
    from src.tests.test_visual_inception_category_evaluation import (
        GROUPS,
        _tiny_frozen_suite,
        _tiny_oracle_query_root,
    )
    import src.visual.attack_completion as completion

    base, images, publics, _ = _tiny_frozen_suite(tmp_path / 'work')
    entries = []
    source_mapping = {}
    for index, group in enumerate(GROUPS):
        src = publics[group]['source']
        mem_id = f'fixture_memory_{index}'
        source_mapping[group] = {'source_asin': src['id'], 'source_memory_id': mem_id}
        entries.append(dict(
            Id=mem_id,
            FinalASIN=src['id'],
            RecordKind='simulated_gallery',
            Completed=False,
            PrimaryImage=dict(
                ASIN=src['id'],
                Title=src['title'],
                LocalPath=src['image'],
                SHA256=src['image_sha256'],
            ),
            FeatureText=f'Fixture feature {index}',
            Attributes=[f'attr_{index}'],
        ))
    pool_path = base / 'shared_memory/shared_memory.json'
    _write(pool_path, dict(schema_version=1, scope='shared', entries=entries))
    _write(base / 'manifest.json', dict(
        groups={g: dict(target_id=publics[g]['target']['id'], source_id=publics[g]['source']['id']) for g in GROUPS},
        shared_memory=dict(
            pool_file='shared_memory/shared_memory.json',
            pool_sha256=sha256_file(pool_path),
            source_mapping=source_mapping,
        ),
    ))
    oracle_root = _tiny_oracle_query_root(tmp_path / 'work', base / 'manifest.json', publics)

    fake_cfg_dir = tmp_path / 'work' / 'configs'
    for _, _, _, m_cfg, h_cfg in category.TRANSFER_PANEL_ENCODERS:
        for fn in (m_cfg, h_cfg):
            _write(fake_cfg_dir / fn, dict(setting='held_out_public_encoder', surrogate=dict(name='dummy', pretrained='dummy')))

    out_root = tmp_path / 'work' / 'output'
    monkeypatch.setattr(category, 'OUTPUT', out_root)
    monkeypatch.setattr(fast_panel, 'OUTPUT', out_root)
    monkeypatch.setattr(category, 'VISUAL_CONFIGS', fake_cfg_dir)
    monkeypatch.setitem(category.QUERY_SUITE_ROOTS, 'training_oracle', oracle_root)
    monkeypatch.setattr(category, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    monkeypatch.setattr(category, 'validate_attack_completion', lambda *args, **kwargs: None)

    encoder_creations = 0

    def _counting_factory():
        nonlocal encoder_creations
        encoder_creations += 1
        return _DummyEncoder()

    selected_out = out_root / 'vi_selected_inproc'
    jobs = [
        dict(
            name='vi_selected_inproc',
            images_root=str(images),
            method='visual_inception',
            query_suite='training_oracle',
            output=str(selected_out),
            groups=['02_chair', '05_hair_extension'],
            replacement_groups='selected',
        )
    ]
    result = fast_panel.run_encoder_grouped_panels(
        jobs,
        base=base,
        key_mode='image',
        defense_mode='none',
        defended_cache_dir=out_root / 'defended_cache',
        memory_vector_cache_dir=out_root / 'memory_vectors',
        batch_sizes={'01_clip_l14': 2, '02_openclip_bigG14': 2, '03_siglip_so400m': 2},
        text_batch_size=8,
        device='cpu',
        verify_training_snapshots=False,
        encoder_factory=_counting_factory,
        verify_mode='in_process',
    )
    summary = result['vi_selected_inproc']
    assert summary['group_count'] == 2
    assert summary['simultaneous_replacement_count'] == 2
    # Stub encoder_factory cannot keep base_encoder alive across eval->verify (2 loads per slug = 6).
    assert encoder_creations == 6
    report_text = (selected_out / 'TRANSFER_PANEL_REPORT.md').read_text(encoding='utf-8')
    assert '按查询子家族三编码器对比汇总' in report_text
