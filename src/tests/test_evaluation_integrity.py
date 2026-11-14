"""三条合成记忆、三条查询，完整面板不接触真实库或模型。"""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
import pytest

from src.visual.audit import read_json, save_json
from src.visual.image_utils import sha256_file


class _DummyEncoder:
    supports_image_weight = True

    def __init__(self):
        self.snapshot = {'provider': 'explicit_offline_test_fixture', 'version': 1}

    def encode(self, path):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'), dtype=np.float32).mean(axis=(0, 1)) / 255
        return np.concatenate([rgb, [1.0, 0.25, 0.5]])

    def encode_text(self, text):
        return np.asarray(list(hashlib.sha256(text.encode('utf-8')).digest()[:6]), dtype=np.float32) + 1

    def encode_joint(self, text, path, image_weight=0.5):
        image, words = self.encode(path), self.encode_text(text)
        return image_weight * image / np.linalg.norm(image) + (1 - image_weight) * words / np.linalg.norm(words)

    def close(self):
        pass


class _TinyPurifier:
    calls = []

    def __init__(self, config, device):
        self.config, self.device = config, device

    def snapshot(self):
        return {'provider': 'explicit_offline_purifier_fixture'}

    def reconstruct(self, image):
        self.calls.append(('vae', None))
        return image

    def __call__(self, image, seed, start_timestep):
        self.calls.append((start_timestep, seed))
        return image


@pytest.fixture
def micro_run(tmp_path, monkeypatch):
    from src.experiments import shared_evaluation as evaluation
    import src.visual.attack_completion as completion
    import src.visual.diffusion_purification as diffusion
    entries = []
    for index, color in enumerate([(200, 90, 30), (30, 180, 80), (40, 60, 190)]):
        image = tmp_path / f'item{index}.png'
        Image.new('RGB', (16, 16), color).save(image)
        entries.append(dict(Id=f'm{index}', FinalASIN=f'P{index}', RecordKind='simulated_gallery',
            Completed=False, PrimaryImage=dict(ASIN=f'P{index}', Title=f'公开商品{index}',
                LocalPath=image.name, SHA256=sha256_file(image)),
            FeatureText=f'商品{index}自己的真实公开特点', Attributes=['合成夹具']))
    pool = tmp_path / 'shared_memory.json'
    save_json(pool, dict(schema_version=1, scope='shared', entries=entries))
    queries = [dict(task_id=f'q{i}', instruction=t, query_mode='text', query_image=None,
                    query_image_sha256=None) for i, t in enumerate(['购买红色商品', '寻找柔和表面', '选择相似外形'])]
    manifest = tmp_path / 'queries.json'
    save_json(manifest, {'offline_fixture_only': True, 'queries': queries})
    modified = tmp_path / 'modified.png'
    Image.new('RGB', (16, 16), (140, 145, 35)).save(modified)
    source = dict(id='P0', title='公开商品0', image=str(tmp_path / 'item0.png'),
                  image_sha256=entries[0]['PrimaryImage']['SHA256'])
    frozen = dict(status='frozen', method='lucid', image=str(modified), image_sha256=sha256_file(modified),
            source=source, target={'id': 'T', 'image_sha256': 'offline_fixture'},
            metadata={'offline_test_only': True}, requested_steps=1)
    # 仅测试内显式替换正式完成检查和二十条请求协议，生产入口不提供此旁路。
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation, 'load_formal_queries', lambda *args, **kwargs: queries)
    monkeypatch.setattr(diffusion, 'LatentDiffusionPurifier', _TinyPurifier)
    _TinyPurifier.calls = []
    output = tmp_path / 'evaluation'
    created = []

    def factory():
        created.append(_DummyEncoder())
        return created[-1]

    result = evaluation.evaluate(frozen, output, pool_path=pool, query_manifest=manifest,
                                 device='cpu', encoder_factory=factory)
    return output, result, created


def test_all_26_conditions_14_rows_and_all_scores(micro_run):
    from src.experiments import shared_evaluation as evaluation
    from src.experiments.defenses import conditions, table_groups
    output, result, created = micro_run
    assert len(conditions()) == 26 and len(table_groups()) == 14
    assert result['rows'] == 52 and result['rankings'] == 156
    assert result['verification']['candidate_scores_recomputed'] == 468
    assert result['verification']['max_score_error'] == 0
    assert len(created) == 2 and created[0] is not created[1]
    with np.load(output / 'scores.npz', allow_pickle=False) as saved:
        assert saved['scores'].shape == (52, 3, 3)
    assert len({c for c in _TinyPurifier.calls if isinstance(c[0], int)}) == 15
    report = evaluation.report(output).read_text(encoding='utf-8')
    assert all('| ' + title + ' |' in report for title, _ in table_groups())
    assert '468' in report


def test_non_source_score_tamper_is_rejected(micro_run):
    from src.experiments.shared_evaluation import verify, report
    output, _, _ = micro_run
    with np.load(output / 'scores.npz', allow_pickle=False) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    data['scores'][0, 0, 1] += 0.01
    np.savez_compressed(output / 'scores.npz', **data)
    with pytest.raises(ValueError, match='全库候选分数'):
        verify(output, encoder_factory=_DummyEncoder)
    with pytest.raises(ValueError, match='复算证据'):
        report(output)


def test_production_refuses_test_fixture_before_reading_pool(tmp_path):
    from src.experiments.shared_evaluation import evaluate
    frozen = dict(status='frozen', metadata={'method': 'ambilage', 'test_fixture': True}, requested_steps=1)
    with pytest.raises(ValueError, match='测试夹具'):
        evaluate(frozen, tmp_path, pool_path=tmp_path / 'not_read.json', query_manifest=tmp_path / 'no.json')


def test_formal_query_count_not_relaxed_for_micro_tests(tmp_path):
    from src.experiments.query_contract import load_formal_queries
    path = tmp_path / 'queries.json'
    save_json(path, dict(target_public_id='T', status='ready', query_mode='text', instructions=['甲', '乙', '丙']))
    with pytest.raises(ValueError, match='二十条'):
        load_formal_queries(path, {'id': 'T'}, query_mode='text')


def test_guard_allows_public_group_but_blocks_shared_pool():
    from src.paths import EVALUATION_BASE, SHARED_MEMORY
    from src.visual.catalog_training_guard import catalog_training_guard
    public = EVALUATION_BASE / 'groups/07_shampoo/source.jpg'
    with catalog_training_guard(allowed_output_roots=[public.parent]):
        assert public.open('rb').read(2)
        for blocked in (
            SHARED_MEMORY,
            EVALUATION_BASE / 'query_images/07_shampoo/query.png',
            EVALUATION_BASE / 'evaluation_inputs_joint/07_shampoo/queries.json',
            EVALUATION_BASE / 'evaluation_inputs_vague_joint/07_shampoo/queries.json',
        ):
            with pytest.raises(PermissionError):
                blocked.open('rb').read(2)


def test_frozen_nine_groups_query_modes_and_vague_images():
    import re
    from src.paths import EVALUATION_BASE
    from src.experiments.formal_queries import VAGUE_INSTRUCTIONS
    from src.experiments.query_contract import load_formal_queries, verify_query_input

    manifest = read_json(EVALUATION_BASE / 'manifest.json')
    frozen_index = read_json(EVALUATION_BASE / 'query_images/QUERY_IMAGES_FROZEN.json')
    by_group = {item['group']: item for item in frozen_index['groups']}
    assert len(by_group) == 9

    for group, spec in manifest['groups'].items():
        target = {'id': spec['target_id'], 'image_sha256': spec['target_sha256']}
        text_path = EVALUATION_BASE / spec['query_manifest']
        joint_path = EVALUATION_BASE / 'evaluation_inputs_joint' / group / 'queries.json'
        vague_joint_path = EVALUATION_BASE / 'evaluation_inputs_vague_joint' / group / 'queries.json'

        for p in (joint_path, vague_joint_path):
            raw = read_json(p)
            assert isinstance(raw.get('image'), str) and not re.match(r'^[A-Za-z]:[\\/]', raw['image']) and not Path(raw['image']).is_absolute()

        q_text = load_formal_queries(text_path, target, query_mode='text')
        q_joint = load_formal_queries(joint_path, target, query_mode='joint')
        q_vague = load_formal_queries(vague_joint_path, target, query_mode='joint')

        assert len(q_text) == len(q_joint) == len(q_vague) == 20
        assert [q['task_id'] for q in q_joint] == [f'feature_{i:02d}' for i in range(1, 21)]
        assert [q['task_id'] for q in q_vague] == [f'vague_{i:02d}' for i in range(1, 21)]
        assert [q['instruction'] for q in q_joint] == [q['instruction'] for q in q_text]
        assert [q['instruction'] for q in q_vague] == list(VAGUE_INSTRUCTIONS)

        expected_sha = by_group[group]['image_sha256']
        resolved_img = (EVALUATION_BASE / 'query_images' / by_group[group]['image']).resolve()
        assert resolved_img.is_file() and sha256_file(resolved_img) == expected_sha
        for item in q_joint + q_vague:
            verify_query_input(item)
            assert Path(item['query_image']).resolve() == resolved_img
            assert item['query_image_sha256'] == expected_sha
            assert item['query_image_sha256'] != spec['target_sha256']
