"""Dedicated category evaluation uses tiny explicit fixtures, never real ranks."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from src.visual.image_utils import sha256_file


GROUPS = ('01_camera', '02_chair', '03_cookies', '04_curtain',
          '05_hair_extension', '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush')


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def _tiny_frozen_suite(tmp_path):
    base = tmp_path / 'fixture_base'
    images = tmp_path / 'fixture_images'
    publics, documents = {}, {}
    for index, group in enumerate(GROUPS):
        pair = {}
        for role, offset in (('source', 0), ('target', 1)):
            path = base / 'groups' / group / (role + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (8, 8), (20 + index * 9 + offset, 30, 40)).save(path)
            pair[role] = dict(id=f'fixture_{role}_{index}', title=f'Fixture {role} {index}',
                              image=str(path), image_sha256=sha256_file(path))
        publics[group] = copy.deepcopy(pair)
        image = images / group / 'visual_inception' / 'frozen.png'
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), (150 + index, 60, 70)).save(image)
        documents[group] = dict(status='frozen', method='visual_inception', requested_steps=100,
            source=pair['source'], target=pair['target'], image=str(image), image_sha256=sha256_file(image),
            metadata=dict(test_fixture=False, production_backend=True,
                training_objective=dict(kind='visual_inception_retrieval_semantic_perceptual_v2')))
        _write(images / group / 'visual_inception' / 'FROZEN_IMAGE.json', documents[group])
    _write(base / 'manifest.json', dict(groups=dict.fromkeys(GROUPS, {}), shared_memory=dict(
        source_mapping={group: {'source_memory_id': f'fixture_memory_{index}'}
                        for index, group in enumerate(GROUPS)})))
    return base, images, publics, documents


def test_frozen_suite_requires_all_nine_before_evaluation(tmp_path, monkeypatch):
    from src.experiments import visual_inception_category_evaluation as category
    import src.visual.attack_completion as completion
    base, images, publics, _ = _tiny_frozen_suite(tmp_path)
    monkeypatch.setattr(category, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    if hasattr(category, 'validate_attack_completion'):
        monkeypatch.setattr(category, 'validate_attack_completion', lambda *args, **kwargs: None)
    (images / GROUPS[-1] / 'visual_inception/FROZEN_IMAGE.json').unlink()
    with pytest.raises((ValueError, FileNotFoundError), match='冻结|成图|尚缺|FROZEN'):
        category.load_frozen_suite(images, base=base)


@pytest.mark.parametrize('mixed', [False, True])
def test_enhanced_suite_accepts_one_version_and_rejects_mixed_versions(tmp_path, monkeypatch, mixed):
    from src.experiments import visual_inception_category_evaluation as category
    base, images, publics, documents = _tiny_frozen_suite(tmp_path)
    monkeypatch.setattr(category, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(category, 'validate_attack_completion', lambda *args, **kwargs: None)
    for group in GROUPS[:-1] if mixed else GROUPS:
        documents[group]['metadata']['training_objective']['kind'] = category.ENHANCED_OBJECTIVE_KIND
        _write(images / group / 'visual_inception/FROZEN_IMAGE.json', documents[group])
    if mixed:
        with pytest.raises(ValueError, match='混用'):
            category.load_frozen_suite(images, base=base)
    else:
        suite = category.load_frozen_suite(images, base=base)
        assert len(suite) == 9
        assert {item['metadata']['training_objective']['kind'] for item in suite} == {category.ENHANCED_OBJECTIVE_KIND}


@pytest.mark.parametrize('change', ('method', 'fixture', 'early_mode', 'image_bytes', 'source_binding'))
def test_frozen_suite_rejects_wrong_identity_or_changed_image(tmp_path, monkeypatch, change):
    from src.experiments import visual_inception_category_evaluation as category
    import src.visual.attack_completion as completion
    base, images, publics, documents = _tiny_frozen_suite(tmp_path)
    monkeypatch.setattr(category, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    if hasattr(category, 'validate_attack_completion'):
        monkeypatch.setattr(category, 'validate_attack_completion', lambda *args, **kwargs: None)
    group = GROUPS[0]
    item = documents[group]
    if change == 'method':
        item['method'] = 'lucid'
    elif change == 'fixture':
        item['metadata']['test_fixture'] = True
    elif change == 'early_mode':
        item['metadata']['training_objective']['kind'] = 'visual_inception_semantic_perceptual_no_retrieval_v1'
    elif change == 'image_bytes':
        Image.new('RGB', (8, 8), 'white').save(item['image'])
    else:
        item['source']['id'] = 'different_fixture_source'
    _write(images / group / 'visual_inception/FROZEN_IMAGE.json', item)
    with pytest.raises(ValueError):
        category.load_frozen_suite(images, base=base)


def test_frozen_category_query_suite_v1_integrity():
    from src.experiments import visual_inception_category_evaluation as category
    from src.paths import EVALUATION_BASE

    result = category.validate_category_query_suite(
        category.CATEGORY_QUERY_ROOT, base=EVALUATION_BASE, verify_training_snapshots=True
    )
    assert result['status'] == 'passed'
    assert result['group_count'] == 9
    assert result['queries_per_group'] == 20
    assert result['total_queries'] == 180
    assert result['queries_per_family'] == 4
    assert result['training_overlap_checked'] is True
    assert result['training_snapshot_count'] == 27
    assert result['unique_training_text_count'] == 891
    assert result['training_normalized_exact_overlap_count'] == 0
    assert result['existing_formal_query_overlap_count'] == 0

    semantic_anchors = {
        '01_camera': ('zoom', 'motion', 'night', 'waterproof', 'weather-resistant', 'ip66', 'outdoor', '1080p', '2k', 'audio', 'voice', 'sd', 'wifi', 'battery', 'mini', 'dome'),
        '02_chair': ('hydraulic', '360', 'swivel', 'height', 'foam', 'sponge', 'cushion', 'steel', 'heavy-duty', 'heavy duty', 'leather', 'pvc', 'reclining', 'footrest'),
        '03_cookies': ('chocolate', 'butter', 'sugar', 'oatmeal', 'frozen', 'refrigerated', 'bake', 'egg', 'gluten', 'vegan', 'edible', 'natural', 'ready-to-eat'),
        '04_curtain': ('blackout', 'darkening', 'thermal', 'insulated', 'privacy', 'noise', 'polyester', 'velvet', 'cotton-linen', 'grommet', 'rod pocket', 'valance', 'wood'),
        '05_hair_extension': ('human hair', 'real hair', 'synthetic', 'crochet', 'kanekalon', 'clip-in', 'tape-in', 'wig', 'weft', 'ponytail', 'halo', 'wavy', 'curly'),
        '06_jeans': ('high waist', 'high-waist', 'high-waisted', 'high-rise', 'mid-rise', 'wide-leg', 'straight-leg', 'straight', 'baggy', 'boyfriend', 'flare', 'bootcut', 'stretch', 'spandex', 'relaxed-fit', 'y2k', 'pocket'),
        '07_shampoo': ('scalp', 'moisturizing', 'hydrating', 'repairing', 'volumizing', 'thickening', 'dandruff', 'sulfate-free', 'tea tree', 'botanical', 'plant', 'aloe', 'purple', 'vegan'),
        '08_speaker': ('waterproof', 'splashproof', 'ipx7', 'portable', 'stereo', 'bass', 'battery', 'playback', 'hour', 'mic', 'hands-free', 'fm', 'microsd', 'tf card', 'solar', 'bluetooth 5.0', 'range'),
        '09_toothbrush': ('u-shaped', 'u-shape', 'silicone', 'soft', 'ultra-soft', 'nano-bristle', 'electric', 'sonic', 'manual', 'kids', 'toddler', 'gum', 'whitening'),
    }
    rare_niche_ceilings = {
        '01_camera': (('optical zoom', 3),),
        '03_cookies': (('egg', 4),),
        '04_curtain': (('wood', 3),),
        '05_hair_extension': (('double weft', 3), ('double-weft', 3)),
        '06_jeans': (('flap', 3),),
        '07_shampoo': (('plant', 3),),
        '08_speaker': (('solar', 3),),
        '09_toothbrush': (('u-shape', 9),),
    }
    banned_meta = ('what should i consider', 'how should i choose', 'how do i choose',
                   'worth putting on a shortlist', 'worth adding to', 'useful shopping shortlist', 'first purchase')
    banned_brands = ('hxview', 'lthdd', 'big dipper', 's-noilite', 'aveda', 'touch of eco', 'buytvproducts4less')

    for group in GROUPS:
        assert result['groups'][group]['query_family_counts'] == {
            family: 4 for family in category.QUERY_FAMILIES
        }
        q_doc = json.loads((category.CATEGORY_QUERY_ROOT / group / 'queries.json').read_text(encoding='utf-8'))
        anchors = semantic_anchors[group]
        lows = [category.normalize_query_text(text) for text in q_doc['instructions']]
        for text, low in zip(q_doc['instructions'], lows):
            assert any(anchor in low for anchor in anchors), (group, text)
            assert not any(meta in low for meta in banned_meta), (group, text)
            assert not any(brand in low for brand in banned_brands), (group, text)
        for token, max_hits in rare_niche_ceilings.get(group, ()):
            assert sum(token in low for low in lows) <= max_hits, (group, token)


def _tiny_category_query_root(tmp_path, base_manifest, publics):
    import hashlib
    from src.experiments import visual_inception_category_evaluation as category

    qroot = tmp_path / 'fixture_category_queries'
    groups_meta = {}
    for index, group in enumerate(GROUPS):
        target_id = publics[group]['target']['id']
        instructions, records, zh_records = [], [], []
        q_num = 0
        for family in category.QUERY_FAMILIES:
            for item_idx in range(4):
                q_num += 1
                qid = f'{group}_category_{q_num:02d}'
                text = f'Fixture {group} {family} query {item_idx + 1} for target {index}'
                text_zh = f'夹具 {group} {family} 第{item_idx + 1}条'
                instructions.append(text)
                records.append(dict(id=qid, query_family=family, instruction=text, instruction_zh=text_zh))
                zh_records.append(dict(
                    id=qid,
                    query_family=family,
                    query_family_zh=category.QUERY_FAMILY_LABELS_ZH[family],
                    instruction=text_zh,
                ))
        instr_sha = hashlib.sha256(
            json.dumps(instructions, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        q_rel = f'{group}/queries.json'
        zh_rel = f'{group}/queries.zh.json'
        q_doc = dict(
            schema_version=2,
            status='ready',
            method='visual_inception',
            evaluation_protocol=category.PROTOCOL,
            group=group,
            query_mode='text',
            target_public_id=target_id,
            category=f'fixture_category_{index}',
            source_type='curated_evaluation',
            language='en',
            instruction_policy='category_diversity_5_families_x4_v1',
            instructions=instructions,
            instructions_sha256=instr_sha,
            query_records=records,
            query_family_counts={f: 4 for f in category.QUERY_FAMILIES},
            query_image_required=False,
            query_images_generated=False,
            is_real_search_log=False,
            training_use_allowed=False,
        )
        zh_doc = dict(
            schema_version=1,
            language='zh',
            purpose='仅供人工阅读，不传给检索编码器',
            group=group,
            category=f'fixture_category_{index}',
            query_records=zh_records,
        )
        _write(qroot / q_rel, q_doc)
        _write(qroot / zh_rel, zh_doc)
        groups_meta[group] = dict(
            path=q_rel,
            sha256=sha256_file(qroot / q_rel),
            chinese_review_path=zh_rel,
            chinese_review_sha256=sha256_file(qroot / zh_rel),
            target_public_id=target_id,
            category=f'fixture_category_{index}',
            query_count=20,
            instructions_sha256=instr_sha,
            query_family_counts={f: 4 for f in category.QUERY_FAMILIES},
        )
    _write(qroot / 'manifest.json', dict(
        schema_version=1,
        protocol=category.PROTOCOL,
        method='visual_inception',
        query_mode='text',
        source_type='curated_evaluation',
        language='en',
        group_count=9,
        queries_per_group=20,
        total_queries=180,
        query_families=list(category.QUERY_FAMILIES),
        queries_per_family=4,
        optimization_may_read=False,
        is_real_search_log=False,
        outcome_selection_used=False,
        base_manifest_sha256=sha256_file(base_manifest),
        groups=groups_meta,
    ))
    return qroot


def test_category_evaluation_end_to_end_micro_suite(tmp_path, monkeypatch):
    from src.experiments import visual_inception_category_evaluation as category
    from src.tests.test_evaluation_integrity import _DummyEncoder
    import src.visual.attack_completion as completion

    base, images, publics, _ = _tiny_frozen_suite(tmp_path)
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
    for bg in range(2):
        bg_path = base / 'shared_memory' / f'bg_{bg}.png'
        bg_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (8, 8), (5 + bg * 30, 120 + bg * 20, 200 - bg * 30)).save(bg_path)
        entries.append(dict(
            Id=f'fixture_bg_{bg}',
            FinalASIN=f'BG_{bg}',
            RecordKind='purchase_memory',
            Completed=True,
            PrimaryImage=dict(
                ASIN=f'BG_{bg}',
                Title=f'Background product {bg}',
                LocalPath=f'bg_{bg}.png',
                SHA256=sha256_file(bg_path),
            ),
            FeatureText=f'Background feature {bg}',
            Attributes=[f'bg_attr_{bg}'],
        ))

    pool_path = base / 'shared_memory/shared_memory.json'
    _write(pool_path, dict(schema_version=1, scope='shared', entries=entries))
    pool_before = pool_path.read_bytes()

    base_manifest = dict(
        groups={
            group: dict(target_id=publics[group]['target']['id'], source_id=publics[group]['source']['id'])
            for group in GROUPS
        },
        shared_memory=dict(
            pool_file='shared_memory/shared_memory.json',
            pool_sha256=sha256_file(pool_path),
            source_mapping=source_mapping,
        ),
    )
    _write(base / 'manifest.json', base_manifest)
    qroot = _tiny_category_query_root(tmp_path, base / 'manifest.json', publics)

    eval_out = tmp_path / 'output/vi_category_eval'
    monkeypatch.setattr(category, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(category, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    monkeypatch.setattr(category, 'validate_attack_completion', lambda *args, **kwargs: None)

    result = category.evaluate_category_suite(
        images,
        eval_out,
        base=base,
        query_root=qroot,
        device='cpu',
        defense_mode='none',
        encoder_factory=_DummyEncoder,
        verify_training_snapshots=False,
    )
    assert pool_path.read_bytes() == pool_before
    for name in (
        'category_evaluation_plan.json',
        'category_query_validation.json',
        'category_retrieval_summary.json',
        'REPORT.md',
    ):
        assert (eval_out / name).is_file()

    verified = category.verify_category_suite(
        eval_out, device='cpu', encoder_factory=_DummyEncoder, isolated_process=False
    )
    assert verified['status'] == 'passed'
    assert verified['groups_verified'] == 9
    assert verified['rankings_recomputed'] == 360
    assert verified['candidate_scores_recomputed'] == 360 * len(entries)
    assert (eval_out / 'category_independent_verification.json').is_file()

    # Verify all nine groups evaluated against the same simultaneously replaced modified bank.
    mod_bank_hashes = set()
    for group in GROUPS:
        rows = json.loads((eval_out / 'groups' / group / 'evaluation.json').read_text(encoding='utf-8'))
        mod_row = next(r for r in rows if r['bank'] == 'modified')
        mod_bank_hashes.add(mod_row['bank_sha256'])
    assert len(mod_bank_hashes) == 1
    shared_mod_bank = json.loads((eval_out / 'shared_banks/banks/none/modified.json').read_text(encoding='utf-8'))
    assert len(shared_mod_bank) == len(entries)
    assert sum('RetrievalImage' in e for e in shared_mod_bank) == 9
    assert [{k: v for k, v in e.items() if k != 'RetrievalImage'} for e in shared_mod_bank] == entries

    summary = result['summary']
    assert summary['total_queries'] == 180
    assert summary['overall']['query_count'] == 180
    for family in category.QUERY_FAMILIES:
        assert summary['by_family'][family]['query_count'] == 36

    # Reconstruct top1/top3/top5 counts directly from per-group query ranks.
    all_details = [
        q
        for group in GROUPS
        for q in summary['groups'][group]['by_condition']['none']['queries']
    ]
    assert len(all_details) == 180
    for k in (1, 3, 5):
        assert summary['overall'][f'top{k}']['clean_hits'] == sum(q['clean_rank'] <= k for q in all_details)
        assert summary['overall'][f'top{k}']['modified_hits'] == sum(q['modified_rank'] <= k for q in all_details)
    for family in category.QUERY_FAMILIES:
        fam_details = [q for q in all_details if q['query_family'] == family]
        for k in (1, 3, 5):
            assert summary['by_family'][family][f'top{k}']['clean_hits'] == sum(q['clean_rank'] <= k for q in fam_details)
            assert summary['by_family'][family][f'top{k}']['modified_hits'] == sum(q['modified_rank'] <= k for q in fam_details)

    assert summary['scope_notice']['retrieval_coverage_only'] is True
    assert summary['scope_notice']['recommendation_or_shopping_executed'] is False

    report_text = (eval_out / 'REPORT.md').read_text(encoding='utf-8')
    assert '按问法类别汇总' in report_text
    assert '类别直接询问' in report_text
    assert '检索覆盖与下游推荐区分' in report_text
    assert 'shopping_allowed=false' in report_text


def test_frozen_training_oracle_query_suite_v1_integrity():
    from src.experiments import visual_inception_category_evaluation as category
    from src.paths import EVALUATION_BASE

    assert category.DEFAULT_QUERY_SUITE == 'training_oracle'
    result = category.validate_category_query_suite(base=EVALUATION_BASE, verify_training_snapshots=True)
    assert result['status'] == 'passed'
    assert result['protocol'] == category.ORACLE_PROTOCOL
    assert result['query_suite'] == 'training_oracle'
    assert result['training_overlap_mode'] == 'oracle_100pct'
    assert result['group_count'] == 9
    assert result['queries_per_group'] == 20
    assert result['total_queries'] == 180
    assert result['queries_per_family'] == {'category_description': 10, 'model_paraphrase': 6, 'heldout_search_query': 4}
    assert result['training_overlap_checked'] is True
    assert result['training_snapshot_count'] == 27
    assert result['unique_training_text_count'] == 891
    assert result['training_normalized_exact_overlap_count'] == 180
    for group in GROUPS:
        assert result['groups'][group]['query_family_counts'] == {
            'category_description': 10,
            'model_paraphrase': 6,
            'heldout_search_query': 4,
        }

    zero_res = category.validate_category_query_suite(
        query_suite='zero_overlap', base=EVALUATION_BASE, verify_training_snapshots=True
    )
    assert zero_res['protocol'] == category.PROTOCOL
    assert zero_res['query_suite'] == 'zero_overlap'
    assert zero_res['training_normalized_exact_overlap_count'] == 0


def _tiny_oracle_query_root(tmp_path, base_manifest, publics):
    import hashlib
    from src.experiments import visual_inception_category_evaluation as category

    qroot = tmp_path / 'fixture_oracle_queries'
    groups_meta = {}
    quotas = {'category_description': 10, 'model_paraphrase': 6, 'heldout_search_query': 4}
    for index, group in enumerate(GROUPS):
        target_id = publics[group]['target']['id']
        instructions, records, zh_records = [], [], []
        q_num = 0
        for family, quota in quotas.items():
            for item_idx in range(quota):
                q_num += 1
                qid = f'{group}_oracle_{q_num:02d}'
                text = f'Oracle fixture {group} {family} item {item_idx + 1} target {index}'
                text_zh = f'训练抽样夹具 {group} {family} 第{item_idx + 1}条'
                instructions.append(text)
                records.append(dict(
                    id=qid,
                    query_family=family,
                    instruction=text,
                    instruction_zh=text_zh,
                    source_row_id=f'row_{q_num:02d}',
                    source_file=f'fixture/{group}/{family}.jsonl',
                ))
                zh_records.append(dict(
                    id=qid,
                    query_family=family,
                    query_family_zh=category.ORACLE_FAMILY_LABELS_ZH[family],
                    instruction=text_zh,
                ))
        instr_sha = hashlib.sha256(
            json.dumps(instructions, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        q_rel = f'{group}/queries.json'
        zh_rel = f'{group}/queries.zh.json'
        q_doc = dict(
            schema_version=2,
            status='ready',
            method='visual_inception',
            evaluation_protocol=category.ORACLE_PROTOCOL,
            group=group,
            query_mode='text',
            target_public_id=target_id,
            category=f'fixture_category_{index}',
            source_type='training_oracle_stratified_sample',
            language='en',
            instruction_policy='training_oracle_50_30_20_stratified_20_v1',
            instructions=instructions,
            instructions_sha256=instr_sha,
            query_records=records,
            query_family_counts=dict(quotas),
            query_image_required=False,
            query_images_generated=False,
            is_real_search_log=False,
            training_use_allowed=False,
        )
        zh_doc = dict(
            schema_version=1,
            language='zh',
            purpose='仅供人工阅读，不传给检索编码器',
            group=group,
            category=f'fixture_category_{index}',
            query_records=zh_records,
        )
        _write(qroot / q_rel, q_doc)
        _write(qroot / zh_rel, zh_doc)
        groups_meta[group] = dict(
            path=q_rel,
            sha256=sha256_file(qroot / q_rel),
            chinese_review_path=zh_rel,
            chinese_review_sha256=sha256_file(qroot / zh_rel),
            target_public_id=target_id,
            category=f'fixture_category_{index}',
            query_count=20,
            instructions_sha256=instr_sha,
            query_family_counts=dict(quotas),
        )
    _write(qroot / 'manifest.json', dict(
        schema_version=1,
        protocol=category.ORACLE_PROTOCOL,
        method='visual_inception',
        query_mode='text',
        source_type='training_oracle_stratified_sample',
        language='en',
        group_count=9,
        queries_per_group=20,
        total_queries=180,
        query_families=list(category.ORACLE_FAMILIES),
        queries_per_family=dict(quotas),
        optimization_may_read=False,
        is_real_search_log=False,
        outcome_selection_used=False,
        base_manifest_sha256=sha256_file(base_manifest),
        groups=groups_meta,
    ))
    return qroot


def test_oracle_query_suite_is_default_in_evaluation(tmp_path, monkeypatch):
    from src.experiments import visual_inception_category_evaluation as category
    from src.tests.test_evaluation_integrity import _DummyEncoder
    import src.visual.attack_completion as completion

    base, images, publics, _ = _tiny_frozen_suite(tmp_path)
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
        groups={
            group: dict(target_id=publics[group]['target']['id'], source_id=publics[group]['source']['id'])
            for group in GROUPS
        },
        shared_memory=dict(
            pool_file='shared_memory/shared_memory.json',
            pool_sha256=sha256_file(pool_path),
            source_mapping=source_mapping,
        ),
    ))
    oracle_root = _tiny_oracle_query_root(tmp_path, base / 'manifest.json', publics)

    eval_out = tmp_path / 'output/vi_oracle_eval'
    monkeypatch.setattr(category, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setitem(category.QUERY_SUITE_ROOTS, 'training_oracle', oracle_root)
    monkeypatch.setattr(category, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    monkeypatch.setattr(category, 'validate_attack_completion', lambda *args, **kwargs: None)

    result = category.evaluate_category_suite(
        images,
        eval_out,
        base=base,
        device='cpu',
        defense_mode='none',
        encoder_factory=_DummyEncoder,
        verify_training_snapshots=False,
    )
    summary = result['summary']
    assert summary['protocol'] == category.ORACLE_PROTOCOL
    assert summary['query_suite'] == 'training_oracle'
    assert summary['key_mode'] == 'image'
    assert summary['key_image_weight'] == 1.0
    assert summary['by_family']['category_description']['query_count'] == 90
    assert summary['by_family']['model_paraphrase']['query_count'] == 54
    assert summary['by_family']['heldout_search_query']['query_count'] == 36
    report_text = (eval_out / 'REPORT.md').read_text(encoding='utf-8')
    assert '100% Oracle' in report_text
    assert '同类商品公开描述(50%)' in report_text
    assert '纯图片向量 `image`' in report_text


def _micro_eval_pool(tmp_path, *, text_variant='original'):
    entries = []
    for index, color in enumerate([(200, 90, 30), (30, 180, 80), (40, 60, 190)]):
        image = tmp_path / f'item{index}.png'
        if not image.exists():
            Image.new('RGB', (8, 8), color).save(image)
        title = f'Fixture Product {index}' if index == 0 else f'Fixture Product {index} {text_variant}'
        entries.append(dict(
            Id=f'm{index}',
            FinalASIN=f'P{index}',
            RecordKind='simulated_gallery',
            Completed=False,
            PrimaryImage=dict(
                ASIN=f'P{index}',
                Title=title,
                LocalPath=image.name,
                SHA256=sha256_file(image),
            ),
            FeatureText=f'Product {index} description {text_variant}',
            Attributes=[f'attr_{index}_{text_variant}'],
        ))
    pool = tmp_path / f'shared_memory_{text_variant}.json'
    _write(pool, dict(schema_version=1, scope='shared', entries=entries))
    return pool, entries


def test_visual_inception_defaults_to_image_key_mode_and_supports_joint_switch(tmp_path, monkeypatch):
    import numpy as np
    from src.experiments import visual_inception_category_evaluation as category
    from src.experiments.defenses import conditions
    from src.experiments import shared_evaluation
    from src.tests.test_evaluation_integrity import _DummyEncoder
    import src.visual.attack_completion as completion

    base, images, publics, _ = _tiny_frozen_suite(tmp_path)
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
            FeatureText=f'Fixture feature alpha {index}',
            Attributes=[f'attr_alpha_{index}'],
        ))
    pool_path = base / 'shared_memory/shared_memory.json'
    _write(pool_path, dict(schema_version=1, scope='shared', entries=entries))
    _write(base / 'manifest.json', dict(
        groups={
            group: dict(target_id=publics[group]['target']['id'], source_id=publics[group]['source']['id'])
            for group in GROUPS
        },
        shared_memory=dict(
            pool_file='shared_memory/shared_memory.json',
            pool_sha256=sha256_file(pool_path),
            source_mapping=source_mapping,
        ),
    ))
    oracle_root = _tiny_oracle_query_root(tmp_path, base / 'manifest.json', publics)

    monkeypatch.setattr(category, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setitem(category.QUERY_SUITE_ROOTS, 'training_oracle', oracle_root)
    monkeypatch.setattr(category, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    monkeypatch.setattr(category, 'validate_attack_completion', lambda *args, **kwargs: None)

    out_img = tmp_path / 'output/vi_default_image'
    res_img = category.evaluate_category_suite(
        images, out_img, base=base, device='cpu', defense_mode='none',
        encoder_factory=_DummyEncoder, verify_training_snapshots=False,
    )
    assert res_img['summary']['key_mode'] == 'image'
    assert res_img['summary']['key_image_weight'] == 1.0
    g0_plan = json.loads((out_img / 'groups' / GROUPS[0] / 'evaluation_plan.json').read_text(encoding='utf-8'))
    assert g0_plan['key_mode'] == 'image'
    assert g0_plan['key_image_weight'] == 1.0
    with np.load(out_img / 'groups' / GROUPS[0] / 'scores.npz', allow_pickle=False) as saved:
        img_scores_alpha = saved['scores'].copy()

    # Alter memory text only while keeping identical images: image-only scores must remain identical.
    entries_beta = copy.deepcopy(entries)
    for idx, item in enumerate(entries_beta):
        item['FeatureText'] = f'Completely rewritten feature beta wording {idx}'
        item['Attributes'] = [f'attr_beta_variant_{idx}']
    pool_beta = base / 'shared_memory/shared_memory_beta.json'
    _write(pool_beta, dict(schema_version=1, scope='shared', entries=entries_beta))
    _write(base / 'manifest.json', dict(
        groups={
            group: dict(target_id=publics[group]['target']['id'], source_id=publics[group]['source']['id'])
            for group in GROUPS
        },
        shared_memory=dict(
            pool_file='shared_memory/shared_memory_beta.json',
            pool_sha256=sha256_file(pool_beta),
            source_mapping=source_mapping,
        ),
    ))
    oracle_manifest = json.loads((oracle_root / 'manifest.json').read_text(encoding='utf-8'))
    oracle_manifest['base_manifest_sha256'] = sha256_file(base / 'manifest.json')
    _write(oracle_root / 'manifest.json', oracle_manifest)
    out_img_beta = tmp_path / 'output/vi_default_image_beta'
    category.evaluate_category_suite(
        images, out_img_beta, base=base, device='cpu', defense_mode='none',
        encoder_factory=_DummyEncoder, verify_training_snapshots=False,
    )
    with np.load(out_img_beta / 'groups' / GROUPS[0] / 'scores.npz', allow_pickle=False) as saved:
        img_scores_beta = saved['scores'].copy()
    assert np.allclose(img_scores_alpha, img_scores_beta, atol=1e-12)

    # Explicit switch to joint must change scores when text changes and record key_mode=joint.
    out_joint_beta = tmp_path / 'output/vi_explicit_joint_beta'
    res_joint = category.evaluate_category_suite(
        images, out_joint_beta, base=base, key_mode='joint', device='cpu', defense_mode='none',
        encoder_factory=_DummyEncoder, verify_training_snapshots=False,
    )
    assert res_joint['summary']['key_mode'] == 'joint'
    assert res_joint['summary']['key_image_weight'] == 0.5
    with np.load(out_joint_beta / 'groups' / GROUPS[0] / 'scores.npz', allow_pickle=False) as saved:
        joint_scores_beta = saved['scores'].copy()
    assert not np.allclose(img_scores_beta, joint_scores_beta, atol=1e-6)
    assert '图文各半融合向量 `joint`' in (out_joint_beta / 'REPORT.md').read_text(encoding='utf-8')
    verified = category.verify_category_suite(
        out_joint_beta, device='cpu', encoder_factory=_DummyEncoder, isolated_process=False,
    )
    assert verified['status'] == 'passed'
    with pytest.raises(ValueError, match='记忆键模式与冻结评估不一致'):
        shared_evaluation.verify(
            out_joint_beta / 'groups' / GROUPS[0],
            encoder_factory=_DummyEncoder,
            key_mode='image',
            isolated_process=False,
        )
    assert conditions()[0]['kind'] == 'none'


@pytest.mark.parametrize('method,expected_default_key_mode,expected_weight', [
    ('lucid', 'joint', 0.5),
    ('ambilage', 'image', 1.0),
    ('visual_inception', 'image', 1.0),
])
def test_default_key_mode_by_method_in_shared_evaluation(
    tmp_path, monkeypatch, method, expected_default_key_mode, expected_weight
):
    from src.experiments import shared_evaluation
    from src.experiments.defenses import conditions
    from src.tests.test_evaluation_integrity import _DummyEncoder
    import src.visual.attack_completion as completion

    pool_alpha, entries_alpha = _micro_eval_pool(tmp_path, text_variant='alpha')
    pool_beta, _ = _micro_eval_pool(tmp_path, text_variant='beta')
    queries = [
        dict(task_id=f'q{i}', instruction=t, query_mode='text', query_image=None, query_image_sha256=None)
        for i, t in enumerate(['Find similar product', 'Smooth color option'])
    ]
    manifest = tmp_path / 'queries.json'
    _write(manifest, dict(explicit_test_fixture=True, queries=queries))
    modified = tmp_path / 'modified.png'
    Image.new('RGB', (8, 8), (120, 120, 40)).save(modified)
    frozen = dict(
        status='frozen',
        method=method,
        image=str(modified),
        image_sha256=sha256_file(modified),
        source=dict(
            id='P0',
            title='Fixture Product 0',
            image=str(tmp_path / 'item0.png'),
            image_sha256=entries_alpha[0]['PrimaryImage']['SHA256'],
        ),
        target=dict(id='T', image_sha256='offline_fixture'),
        metadata=dict(offline_test_only=True),
        requested_steps=1,
    )
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    monkeypatch.setattr(shared_evaluation, 'load_formal_queries', lambda *args, **kwargs: queries)

    out_default = tmp_path / f'eval_{method}_default'
    res = shared_evaluation.evaluate(
        frozen, out_default, pool_path=pool_alpha, query_manifest=manifest,
        device='cpu', encoder_factory=_DummyEncoder, selected_conditions=conditions()[:1],
    )
    assert res['verification']['status'] == 'passed'
    plan = json.loads((out_default / 'evaluation_plan.json').read_text(encoding='utf-8'))
    rows = json.loads((out_default / 'evaluation.json').read_text(encoding='utf-8'))
    assert plan['key_mode'] == expected_default_key_mode
    assert plan['key_image_weight'] == expected_weight
    assert all(r['retrieval_protocol']['key_mode'] == expected_default_key_mode for r in rows)
    assert all(r['retrieval_protocol']['key_image_weight'] == expected_weight for r in rows)

    alt_mode = 'image' if expected_default_key_mode == 'joint' else 'joint'
    out_alt = tmp_path / f'eval_{method}_alt'
    shared_evaluation.evaluate(
        frozen, out_alt, pool_path=pool_beta, query_manifest=manifest,
        device='cpu', key_mode=alt_mode, encoder_factory=_DummyEncoder, selected_conditions=conditions()[:1],
    )
    alt_plan = json.loads((out_alt / 'evaluation_plan.json').read_text(encoding='utf-8'))
    assert alt_plan['key_mode'] == alt_mode
    assert alt_plan['key_image_weight'] == (1.0 if alt_mode == 'image' else 0.5)

    with pytest.raises(ValueError, match='记忆键模式与冻结评估不一致'):
        shared_evaluation.verify(out_alt, encoder_factory=_DummyEncoder, key_mode=expected_default_key_mode)


def test_pipeline_load_config_validates_key_mode(tmp_path):
    from src.experiments import pipeline

    cfg_path = tmp_path / 'pipeline.json'
    cfg = dict(
        schema_version=1,
        method='ambilage',
        public_pair='data/public/example_pair.json',
        evaluation=dict(query_mode='text', key_mode='joint'),
    )
    _write(cfg_path, cfg)
    loaded = pipeline.load_config(cfg_path)
    assert loaded['evaluation']['key_mode'] == 'joint'

    bad_cfg = copy.deepcopy(cfg)
    bad_cfg['evaluation']['key_mode'] = 'text'
    _write(cfg_path, bad_cfg)
    with pytest.raises(ValueError, match='记忆库检索键模式'):
        pipeline.load_config(cfg_path)


def test_frozen_category_query_overlap_30_v1_integrity():
    from src.experiments import visual_inception_category_evaluation as category

    report = category.validate_category_query_suite(query_suite='overlap_30')
    assert report['status'] == 'passed'
    assert report['protocol'] == 'category_query_overlap_30_v1'
    assert report['total_queries'] == 180
    assert report['training_normalized_exact_overlap_count'] == 54
    assert report['queries_per_family'] == {
        'category_description': 3,
        'model_paraphrase': 2,
        'heldout_search_query': 1,
        'synthetic_non_overlap': 14,
    }


def test_defender_disk_snapshot_cache_hit_skips_purifier_init(tmp_path, monkeypatch):
    import PIL
    import src.visual.diffusion_purification as dp
    from src.experiments.defenses import Defender, conditions_for_mode
    from src.visual.audit import digest

    spec = next(s for s in conditions_for_mode('panel_with_sd21') if s['label'] == 'diff512_t99_s1001')
    src_img = tmp_path / 'src.png'
    Image.new('RGB', (16, 16), (30, 80, 140)).save(src_img)
    fake_snap = dict(kind='disk_only_purifier_fixture', working_size=512)
    monkeypatch.setattr(dp, 'purifier_snapshot', lambda *args, **kwargs: copy.deepcopy(fake_snap))

    def _fail_init(*args, **kwargs):
        raise AssertionError('LatentDiffusionPurifier.__init__ must not run on a cache hit')

    monkeypatch.setattr(dp.LatentDiffusionPurifier, '__init__', _fail_init)
    defended_dir = tmp_path / 'defended'
    defender = Defender(defended_dir, device='cpu')
    try:
        identity = dict(
            input_sha256=sha256_file(src_img),
            condition=spec,
            implementation_sha256=defender._impl_sha256,
            pillow=PIL.__version__,
            purifier=fake_snap,
        )
        out_img = defended_dir / spec['label'] / (digest(identity) + '.png')
        out_img.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (16, 16), (40, 90, 150)).save(out_img)
        _write(out_img.with_suffix('.json'), dict(identity=identity, output_sha256=sha256_file(out_img)))

        returned = defender(src_img, spec)
        assert returned == out_img
    finally:
        defender.close()


def test_freeze_memory_matches_unfrozen_and_shared_bank_memo_category_suite(tmp_path, monkeypatch):
    import numpy as np
    from src.experiments import shared_evaluation
    from src.experiments import visual_inception_category_evaluation as category
    from src.tests.test_evaluation_integrity import _DummyEncoder
    from src.visual.visual_retrieval import VisualMemoryRetriever
    import src.visual.attack_completion as completion

    pool_path, entries = _micro_eval_pool(tmp_path, text_variant='alpha')
    encoder = _DummyEncoder()
    try:
        unfrozen = VisualMemoryRetriever(
            encoder, pool_path, key_mode='image', query_mode='text',
            cache_dir=None, image_base_path=pool_path, freeze_memory=False,
        )
        frozen = VisualMemoryRetriever(
            encoder, pool_path, key_mode='image', query_mode='text',
            cache_dir=None, image_base_path=pool_path, freeze_memory=True,
        )
        for query in ('Find similar product', 'Smooth color option'):
            r1 = unfrozen.rank(entries, query, None)
            r2 = frozen.rank(entries, query, None)
            assert [(x['Id'], x['rank'], pytest.approx(x['score'], abs=1e-12)) for x in r1] == [
                (y['Id'], y['rank'], pytest.approx(y['score'], abs=1e-12)) for y in r2
            ]
    finally:
        encoder.close()

    base, images, publics, _ = _tiny_frozen_suite(tmp_path / 'suite_case')
    suite_entries = []
    source_mapping = {}
    for index, group in enumerate(GROUPS):
        src = publics[group]['source']
        mem_id = f'fixture_memory_{index}'
        source_mapping[group] = {'source_asin': src['id'], 'source_memory_id': mem_id}
        suite_entries.append(dict(
            Id=mem_id, FinalASIN=src['id'], RecordKind='simulated_gallery', Completed=False,
            PrimaryImage=dict(ASIN=src['id'], Title=src['title'], LocalPath=src['image'], SHA256=src['image_sha256']),
            FeatureText=f'Fixture feature {index}', Attributes=[f'attr_{index}'],
        ))
    suite_pool = base / 'shared_memory/shared_memory.json'
    _write(suite_pool, dict(schema_version=1, scope='shared', entries=suite_entries))
    _write(base / 'manifest.json', dict(
        groups={g: dict(target_id=publics[g]['target']['id'], source_id=publics[g]['source']['id']) for g in GROUPS},
        shared_memory=dict(pool_file='shared_memory/shared_memory.json', pool_sha256=sha256_file(suite_pool),
                           source_mapping=source_mapping),
    ))
    import hashlib
    overlap_root = tmp_path / 'overlap_queries'
    base_sha = sha256_file(base / 'manifest.json')
    groups_meta = {}
    quotas = dict(category.OVERLAP_30_FAMILY_COUNTS)
    for index, group in enumerate(GROUPS):
        target_id = publics[group]['target']['id']
        instructions, records, zh_records = [], [], []
        q_num = 0
        for fam, quota in quotas.items():
            for item_idx in range(quota):
                q_num += 1
                qid = f'fixture_{group}_o30_{q_num:02d}'
                text = f'Overlap30 fixture {group} {fam} item {item_idx + 1} target {index}'
                text_zh = f'三成重合夹具 {group} {fam} 第{item_idx + 1}条'
                instructions.append(text)
                rec = dict(
                    id=qid,
                    query_family=fam,
                    instruction=text,
                    instruction_zh=text_zh,
                )
                if fam in category.ORACLE_FAMILIES:
                    rec['source_row_id'] = f'row_{q_num:02d}'
                    rec['source_file'] = f'fixture/{group}/{fam}.jsonl'
                records.append(rec)
                zh_records.append(dict(
                    id=qid,
                    query_family=fam,
                    query_family_zh=category.OVERLAP_30_FAMILY_LABELS_ZH[fam],
                    instruction=text_zh,
                ))
        instr_sha = hashlib.sha256(
            json.dumps(instructions, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        q_rel = f'{group}/queries.json'
        zh_rel = f'{group}/queries.zh.json'
        _write(overlap_root / q_rel, dict(
            schema_version=2, status='ready', method='visual_inception',
            evaluation_protocol=category.OVERLAP_30_PROTOCOL, group=group,
            query_mode='text', target_public_id=target_id, category=f'fixture_category_{index}',
            language='en', instructions=instructions, instructions_sha256=instr_sha,
            query_records=records, query_family_counts=quotas,
            query_image_required=False, query_images_generated=False,
            is_real_search_log=False, training_use_allowed=False,
        ))
        _write(overlap_root / zh_rel, dict(
            schema_version=1, language='zh', purpose='仅供人工阅读，不传给检索编码器',
            group=group, category=f'fixture_category_{index}', query_records=zh_records,
        ))
        groups_meta[group] = dict(
            path=q_rel,
            sha256=sha256_file(overlap_root / q_rel),
            chinese_review_path=zh_rel,
            chinese_review_sha256=sha256_file(overlap_root / zh_rel),
            target_public_id=target_id,
            category=f'fixture_category_{index}',
            query_count=20,
            instructions_sha256=instr_sha,
            query_family_counts=quotas,
        )
    _write(overlap_root / 'manifest.json', dict(
        schema_version=1, protocol=category.OVERLAP_30_PROTOCOL, method='visual_inception',
        query_mode='text', language='en', group_count=9, queries_per_group=20, total_queries=180,
        query_families=list(category.OVERLAP_30_FAMILIES), queries_per_family=quotas,
        optimization_may_read=False, is_real_search_log=False, outcome_selection_used=False,
        base_manifest_sha256=base_sha, groups=groups_meta,
    ))
    monkeypatch.setattr(category, 'OUTPUT', tmp_path / 'suite_out')
    monkeypatch.setitem(category.QUERY_SUITE_ROOTS, 'overlap_30', overlap_root)
    monkeypatch.setattr(category, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    monkeypatch.setattr(category, 'validate_attack_completion', lambda *args, **kwargs: None)

    out_dir = tmp_path / 'suite_out/vi_o30_fast'
    res = category.evaluate_category_suite(
        images, out_dir, base=base, query_suite='overlap_30', device='cpu',
        defense_mode='none', encoder_factory=_DummyEncoder, verify_training_snapshots=False,
    )
    assert res['summary']['query_suite'] == 'overlap_30'
    assert res['summary']['overall']['query_count'] == 180
    verified = category.verify_category_suite(out_dir, device='cpu', encoder_factory=_DummyEncoder, isolated_process=False)
    assert verified['status'] == 'passed'
    assert verified['all_candidate_scores_checked'] is True
    assert float(verified['max_score_error']) <= shared_evaluation.SCORE_TOLERANCE




