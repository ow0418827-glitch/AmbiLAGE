"""三条微型记忆验证统一修改库与仅变图条目的向量更新。"""
import copy
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from src.visual.audit import read_json, save_json
from src.visual.image_utils import sha256_file
from src.visual.visual_retrieval import VisualMemoryRetriever, populate_retriever_cache_from_base
from src.tests.test_evaluation_integrity import _DummyEncoder


def _inputs(tmp_path):
    entries, frozen = [], []
    for i, color in enumerate([(190, 70, 20), (20, 180, 70), (40, 70, 180)]):
        path = tmp_path / f'item{i}.png'
        Image.new('RGB', (8, 8), color).save(path)
        entries.append(dict(Id=f'm{i}', FinalASIN=f'P{i}', RecordKind='simulated_gallery',
            Completed=False, Actions=[], FeatureText=f'公开特点{i}',
            PrimaryImage=dict(ASIN=f'P{i}', Title=f'公开商品{i}', LocalPath=path.name, SHA256=sha256_file(path))))
        if i:
            changed = tmp_path / f'changed{i}.png'
            Image.new('RGB', (8, 8), (110 + i * 20, 140, 60)).save(changed)
            frozen.append(dict(status='frozen', method='lucid', requested_steps=1000,
                source=dict(id=f'P{i}', title=f'公开商品{i}', image=str(path), image_sha256=sha256_file(path)),
                target=dict(id=f'T{i}', image_sha256='explicit-offline-fixture'),
                image=str(changed), image_sha256=sha256_file(changed),
                metadata=dict(explicit_offline_fixture=True), source_memory_id=f'm{i}'))
    pool = tmp_path / 'shared_memory.json'
    save_json(pool, dict(schema_version=1, scope='shared', entries=entries))
    return pool, entries, frozen


def test_multiple_replacements_leave_normal_entry_and_base_bytes_unchanged(tmp_path):
    from src.agent.memory.shared_pool import evaluation_banks
    pool, entries, frozen = _inputs(tmp_path)
    before = pool.read_bytes()
    banks, selected = evaluation_banks(entries, pool, frozen[0]['source'], frozen[0]['image'],
        replacements=[dict(source=f['source'], image=f['image'], source_memory_id=f['source_memory_id']) for f in frozen])
    assert selected == 'm1' and pool.read_bytes() == before
    assert banks['clean'] == entries
    assert banks['modified'][0] == entries[0]
    assert sum('RetrievalImage' in e for e in banks['modified']) == 2
    assert [{k: v for k, v in e.items() if k != 'RetrievalImage'} for e in banks['modified']] == entries


def test_queries_share_one_modified_bank_and_reuse_only_compatible_vectors(tmp_path, monkeypatch):
    import src.visual.attack_completion as completion
    from src.experiments import shared_evaluation as evaluation
    from src.experiments.defenses import conditions
    pool, entries, frozen = _inputs(tmp_path)
    before = pool.read_bytes()
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *a, **k: None)
    queries = [dict(task_id=f'q{i}', instruction=t, query_mode='text', query_image=None,
                    query_image_sha256=None) for i, t in enumerate(['寻找相似商品', '购买柔和颜色'])]
    monkeypatch.setattr(evaluation, 'load_formal_queries', lambda *a, **k: queries)
    manifest = tmp_path / 'queries.json'
    save_json(manifest, dict(explicit_test_fixture=True))
    outputs = []
    for i, item in enumerate(frozen):
        output = tmp_path / f'evaluation{i}'
        result = evaluation.evaluate(item, output, pool_path=pool, query_manifest=manifest,
            source_memory_id=item['source_memory_id'], device='cpu', encoder_factory=_DummyEncoder,
            selected_conditions=conditions()[:1], replacement_frozens=frozen,
            bank_root=tmp_path / 'shared_banks', memory_vector_cache_dir=tmp_path / 'memory_vectors')
        assert result['verification']['max_score_error'] == 0
        outputs.append(read_json(output / 'evaluation.json'))
    first, second = outputs
    assert first[1]['bank_path'] == second[1]['bank_path']
    assert first[1]['bank_sha256'] == second[1]['bank_sha256']
    assert first[1]['retrieval_protocol']['memory_vector_reuse'] == dict(hits=1, misses=2, writes=2, rejected=0)
    assert second[0]['retrieval_protocol']['memory_vector_reuse'] == dict(hits=3, misses=0, writes=0, rejected=0)
    assert second[1]['retrieval_protocol']['memory_vector_reuse'] == dict(hits=3, misses=0, writes=0, rejected=0)
    bank = read_json(first[1]['bank_path'])
    assert bank[0] == entries[0] and sum('RetrievalImage' in e for e in bank) == 2
    assert pool.read_bytes() == before


def test_legacy_vector_remapping_rejects_different_encoder(tmp_path):
    pool, entries, _ = _inputs(tmp_path)
    ranker = VisualMemoryRetriever(_DummyEncoder(), pool, query_mode='text', cache_dir=None)
    identity = ranker.index_identity(entries)
    identity['model_sha256'] = 'different-model'
    with pytest.raises(ValueError, match='不兼容'):
        populate_retriever_cache_from_base(ranker, entries, dict(schema_version=4, identity=identity, vectors={}))


def test_formal_evaluation_waits_for_all_nine_frozen_images(tmp_path, monkeypatch):
    from src.experiments import base_workflow
    output = tmp_path / 'output'
    monkeypatch.setattr(base_workflow, 'OUTPUT', output)
    with pytest.raises(ValueError, match='全部九组成图'):
        base_workflow.main(['evaluate', '--group', '07_shampoo', '--method', 'ambilage',
                            '--output', str(output), '--device', 'cpu'])
