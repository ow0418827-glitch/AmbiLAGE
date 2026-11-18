"""Unit and offline integration tests for single-surrogate (CLIP-ViT-L/14),
Visual Inception single-encoder Eq. (2), category abstraction, AmbiLAGE single-model
gradient adaptation, and three-encoder transfer panel evaluation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from src.experiments import visual_inception_category_evaluation as category_eval
from src.tests.test_evaluation_integrity import _DummyEncoder
from src.visual import training_surrogates as choices
from src.visual._ambilage.config import AmbiLAGEConfig
from src.visual._ambilage.evidence import validate_ready as validate_ambilage_ready
from src.visual._ambilage.pipeline import AmbiLAGEPipeline
from src.visual._ambilage.smoke import SmokeStages, TinyPurifier, TinySurrogate
from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.data_preparation import _write_rows, prepare_training_bundle
from src.visual._visual_inception.evidence import validate_ready as validate_vi_ready
from src.visual._visual_inception.objective import single_surrogate_loss_and_gradient
from src.visual._visual_inception.pipeline import generate_visual_inception
from src.visual._visual_inception.targets import prepare_semantic_targets, text_sha256
from src.visual._visual_inception.training_data import load_training_bundle
from src.visual.category_abstraction import (
    build_category_abstraction,
    build_contrastive_category_subspace,
    compute_contrastive_salience,
    denoise_queries_james_stein,
    lazy_greedy_facility_location,
    spherical_karcher_mean,
)
from src.visual.io import sha256_file
from src.visual.training_objectives import (
    SequentialEnsembleObjective,
    multiscale_grid_breaking_views,
    single_model_dominant_pcgrad,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def _public_shampoo_definition() -> dict:
    text = 'I want to buy shampoo with gentle moisturizing formula and fresh herbal scent.'
    return dict(
        source_category='shampoo',
        source_public_key='Gentle Cleansing Shampoo with Botanical Extract',
        target_feature_text=text,
        target_product_features=dict(
            category='shampoo',
            core_attributes=['gentle moisturizing formula', 'fresh herbal scent'],
            canonical_sentence=text,
        ),
    )


def _make_rich_bundle_fixture(root: Path, pair: tuple[Path, Path], *, heldout_count: int = 20) -> Path:
    """Create a multi-category public bundle fixture (with brands/titles for salience & same-category peers)."""
    source, target = pair
    semantic = prepare_semantic_targets(source, target, definition=_public_shampoo_definition())
    upstream = {
        'dataset': 'synthetic_single_model_fixture',
        'revision': 'fixture-v1',
        'url': 'https://example.invalid/synthetic-fixture',
    }
    sources = {}
    queries = []
    heldout_ids = []

    desc_rows = []
    for idx in range(50):
        rid = f'desc-{idx:03d}'
        brand_token = f'uniquebrandtoken{idx:03d}'
        txt = (
            f'{brand_token} gentle moisturizing herbal shampoo for dry scalp '
            f'daily cleansing formula bottle variant {idx:03d}.'
        )
        desc_rows.append(dict(id=rid, text=txt, category='shampoo'))
        queries.append(dict(
            id=rid,
            text=txt,
            source_type='category_description',
            provenance=dict(source_id='category_description', row_id=rid, text_sha256=text_sha256(txt)),
        ))
    desc_path = root / 'sources' / 'category_description.jsonl'
    _write_rows(desc_path, desc_rows)
    sources['category_description'] = dict(
        path=desc_path.relative_to(root).as_posix(),
        sha256=sha256_file(desc_path),
        kind='category_description',
        split='public_training',
        upstream=upstream,
        test_fixture=True,
        is_synthetic=True,
    )

    para_rows = []
    for idx in range(30):
        rid = f'para-{idx:03d}'
        txt = f'Need moisturizing herbal shampoo for gentle cleansing option {idx:03d}.'
        para_rows.append(dict(
            id=rid,
            text=txt,
            category='shampoo',
            provider='minimax',
            model='MiniMax-M3',
            effective_temperature=1.0,
            temperature_supported=False,
            temperature_parameter_sent=False,
            input_row_id=f'desc-{idx:03d}',
        ))
        queries.append(dict(
            id=rid,
            text=txt,
            source_type='model_paraphrase',
            provenance=dict(source_id='model_paraphrase', row_id=rid, text_sha256=text_sha256(txt)),
        ))
    para_path = root / 'sources' / 'model_paraphrase.jsonl'
    _write_rows(para_path, para_rows)
    sources['model_paraphrase'] = dict(
        path=para_path.relative_to(root).as_posix(),
        sha256=sha256_file(para_path),
        kind='model_paraphrase',
        split='public_training',
        upstream=upstream,
        test_fixture=True,
        is_synthetic=True,
        provider='minimax',
        model='MiniMax-M3',
        effective_temperature=1.0,
        temperature_supported=False,
        temperature_parameter_sent=False,
    )

    held_rows = []
    for idx in range(heldout_count):
        rid = f'held-{idx:03d}'
        txt = f'best gentle moisturizing shampoo for daily hair care {idx:03d}'
        held_rows.append(dict(id=rid, text=txt, category='shampoo', split='query_reserve'))
        heldout_ids.append(rid)
        queries.append(dict(
            id=rid,
            text=txt,
            source_type='heldout_search_query',
            provenance=dict(source_id='heldout_search_query', row_id=rid, text_sha256=text_sha256(txt)),
        ))
    held_path = root / 'sources' / 'heldout_search_query.jsonl'
    _write_rows(held_path, held_rows)
    held_src = dict(
        path=held_path.relative_to(root).as_posix(),
        sha256=sha256_file(held_path),
        kind='heldout_search_query',
        split='query_reserve',
        upstream=upstream,
        test_fixture=True,
        is_synthetic=True,
    )
    if heldout_count < 20:
        held_src.update(complete=False, actual_count=heldout_count)
    sources['heldout_search_query'] = held_src

    candidates, catalog_rows = [], []
    for idx in range(20):
        cat = 'shampoo' if idx < 10 else 'camera'
        r = (20 + idx * 9) % 250
        g = (40 + idx * 17) % 250
        b = (190 - idx * 7) % 250
        img_path = root / 'images' / f'cand-{idx:02d}.png'
        img_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (16, 16), (r, g, b)).save(img_path)
        sha = sha256_file(img_path)
        cid = f'cand-{idx:02d}'
        title = (
            f'Herbal Gentle Shampoo Care {idx}'
            if cat == 'shampoo'
            else f'Compact Digital Action Camera Lens {idx}'
        )
        catalog_rows.append(dict(id=cid, image_sha256=sha, category=cat, title=title))
        candidates.append(dict(
            id=cid,
            image=img_path.relative_to(root).as_posix(),
            image_sha256=sha,
            provenance=dict(source_id='candidate_catalog', row_id=cid),
        ))
    cat_path = root / 'sources' / 'candidate_catalog.jsonl'
    _write_rows(cat_path, catalog_rows)
    sources['candidate_catalog'] = dict(
        path=cat_path.relative_to(root).as_posix(),
        sha256=sha256_file(cat_path),
        kind='candidate_catalog',
        split='public_training',
        upstream=upstream,
        test_fixture=True,
        is_synthetic=True,
    )

    doc = dict(
        schema_version=1,
        test_fixture=True,
        category='shampoo',
        paraphrase_policy='declared_model',
        target_binding=dict(
            source_image_sha256=sha256_file(source),
            target_image_sha256=sha256_file(target),
            canonical_text_sha256=semantic['canonical_text_sha256'],
        ),
        target_concept_text='shampoo: gentle moisturizing formula; fresh herbal scent',
        queries=queries,
        candidates=candidates,
        sources=sources,
        formal_queries_used=False,
        victim_memory_used=False,
        split_manifest=dict(
            policy='independent_public_training_query_reserve_v1',
            evaluation_partition_excluded=True,
            formal_query_sources_used=False,
            heldout_query_ids=heldout_ids,
        ),
    )
    if heldout_count < 20:
        doc['declared_query_shortfall'] = {
            'heldout_search_query': {'requested': 20, 'actual': heldout_count, 'complete': False}
        }
    manifest = root / 'training_bundle.json'
    _write_json(manifest, doc)
    return manifest


def test_clip_l14_single_and_three_surrogate_profiles_are_preserved():
    clip_sel = choices.select_training_surrogates('clip_l14')
    assert clip_sel.profile == 'clip_l14'
    assert len(clip_sel.specs) == 1
    assert clip_sel.specs[0].name == 'ViT-L-14'
    assert clip_sel.specs[0].pretrained == 'openai'

    vi_paper = choices.select_training_surrogates('visual_inception_paper')
    lucid = choices.select_training_surrogates('lucid')
    assert vi_paper.profile == 'visual_inception_paper' and len(vi_paper.specs) == 3
    assert lucid.profile == 'lucid' and len(lucid.specs) == 3
    assert VisualInceptionConfig().surrogate_set == 'clip_l14'
    assert AmbiLAGEConfig().surrogate_set == 'clip_l14'


def test_visual_inception_single_surrogate_eq2_and_declared_shortfall(tmp_path):
    # 1. Direct single_surrogate_loss_and_gradient test
    source_t = torch.full((1, 3, 16, 16), 0.4, dtype=torch.float32)
    adv_t = torch.full((1, 3, 16, 16), 0.5, dtype=torch.float32)
    tiny_model = TinySurrogate(0)
    target_feat = torch.ones((1, 12), dtype=torch.float32)
    cfg_unit = VisualInceptionConfig(
        steps=2,
        semantic_weight=1.0,
        retrieval_weight=1.0,
        perceptual_weight=0.1,
        surrogate_set='clip_l14',
        smoke_test=True,
    )

    def _dummy_proxy_fn(model, leaf, tgt, *, retrieval_state, model_index, query_indices, temperature, config, return_audit=True, include_per_query_audit=False):
        feat = model.encode_image(leaf).global_embedding
        sem = (1.0 - feat @ F.normalize(tgt, dim=-1).T).mean()
        ret = (leaf.square().mean() + 0.05)
        return sem, ret, {'model_index': model_index, 'query_count': len(query_indices or [])}

    grad, rec = single_surrogate_loss_and_gradient(
        adv_t,
        source_t,
        tiny_model,
        target_feat,
        lambda s, a: (s - a).square().mean(),
        cfg_unit,
        proxy_fn=_dummy_proxy_fn,
        retrieval_state=None,
        query_indices=[0, 1],
    )
    assert grad.shape == adv_t.shape and torch.isfinite(grad).all()
    assert len(rec['semantic_losses']) == 1 and len(rec['retrieval_losses']) == 1
    assert abs(rec['loss'] - (rec['semantic_loss'] + rec['retrieval_loss'] + 0.1 * rec['perceptual_loss'])) < 1e-6

    # 2. End-to-end single-surrogate vs 3-surrogate VI generation
    src_img, tgt_img = tmp_path / 'src.png', tmp_path / 'tgt.png'
    Image.new('RGB', (16, 16), (170, 110, 80)).save(src_img)
    Image.new('RGB', (16, 16), (60, 150, 180)).save(tgt_img)
    manifest_100 = _make_rich_bundle_fixture(tmp_path / 'bundle_100', (src_img, tgt_img), heldout_count=20)

    cfg_1 = VisualInceptionConfig(
        steps=2,
        surrogate_set='clip_l14',
        training_manifest=str(manifest_100),
        candidate_cache_dir=str(tmp_path / 'cache_1'),
        smoke_test=True,
    )
    out1, meta1 = generate_visual_inception(
        src_img, tgt_img, tmp_path / 'vi_single', config=cfg_1, device='cpu', definition=_public_shampoo_definition()
    )
    assert len(meta1['surrogates']) == 1
    assert meta1['attack']['ensemble_weights'] == [1.0]
    assert meta1['attack']['gradient_restarts_executed'] == 0
    assert validate_vi_ready(out1.parent, require_production=False)['attack']['ensemble_weights'] == [1.0]

    cfg_3 = VisualInceptionConfig(
        steps=2,
        gradient_restarts=2,
        surrogate_set='visual_inception_paper',
        training_manifest=str(manifest_100),
        candidate_cache_dir=str(tmp_path / 'cache_3'),
        smoke_test=True,
    )
    _, meta3 = generate_visual_inception(
        src_img, tgt_img, tmp_path / 'vi_triple', config=cfg_3, device='cpu', definition=_public_shampoo_definition()
    )
    assert len(meta3['surrogates']) == 3
    assert len(meta3['attack']['ensemble_weights']) == 3

    # 3. Declared shortfall gate (strict rejection by default vs opt-in)
    prep_dir = tmp_path / 'prep_shortfall'
    prep_src = prep_dir / 'inputs'
    prep_src.mkdir(parents=True)
    desc_p = prep_src / 'desc.jsonl'
    para_p = prep_src / 'para.jsonl'
    held_p = prep_src / 'held.jsonl'
    cand_p = prep_src / 'cand.jsonl'
    c_img0 = prep_src / 'c0.png'
    c_img1 = prep_src / 'c1.png'
    Image.new('RGB', (16, 16), (10, 20, 30)).save(c_img0)
    Image.new('RGB', (16, 16), (200, 180, 90)).save(c_img1)
    _write_rows(desc_p, [{'id': f'd{i}', 'text': f'shampoo description row {i}', 'category': 'shampoo'} for i in range(50)])
    _write_rows(para_p, [{
        'id': f'p{i}', 'text': f'shampoo paraphrase row {i}', 'category': 'shampoo',
        'provider': 'minimax', 'model': 'MiniMax-M3', 'effective_temperature': 1.0,
        'temperature_supported': False, 'temperature_parameter_sent': False, 'input_row_id': f'd{i}',
    } for i in range(30)])
    _write_rows(held_p, [{'id': f'h{i}', 'text': f'shampoo search query {i}', 'category': 'shampoo', 'split': 'query_reserve'} for i in range(12)])
    _write_rows(cand_p, [
        {'id': 'c0', 'image': 'c0.png', 'image_sha256': sha256_file(c_img0), 'category': 'shampoo'},
        {'id': 'c1', 'image': 'c1.png', 'image_sha256': sha256_file(c_img1), 'category': 'camera'},
    ])
    up = {'dataset': 'synthetic_shortfall_fixture', 'revision': 'v1', 'url': 'https://example.invalid/fixture'}
    src_meta = {
        'category_description': {'kind': 'category_description', 'split': 'public_training', 'upstream': up, 'test_fixture': True, 'is_synthetic': True},
        'model_paraphrase': {
            'kind': 'model_paraphrase', 'split': 'public_training', 'upstream': up, 'test_fixture': True, 'is_synthetic': True,
            'provider': 'minimax', 'model': 'MiniMax-M3', 'effective_temperature': 1.0, 'temperature_supported': False, 'temperature_parameter_sent': False,
        },
        'heldout_search_query': {
            'kind': 'heldout_search_query', 'split': 'query_reserve', 'upstream': up, 'test_fixture': True, 'is_synthetic': True,
            'complete': False, 'actual_count': 12,
        },
        'candidate_catalog': {'kind': 'candidate_catalog', 'split': 'public_training', 'upstream': up, 'test_fixture': True, 'is_synthetic': True},
    }
    sem = prepare_semantic_targets(src_img, tgt_img, definition=_public_shampoo_definition())
    with pytest.raises(ValueError, match='配额不足'):
        prepare_training_bundle(
            source=src_img, target=tgt_img, public_semantic=sem, descriptions=desc_p, paraphrases=para_p,
            heldout_queries=held_p, candidates=cand_p, source_metadata=src_meta,
            output_dir=prep_dir / 'strict_fail', test_fixture=True, allow_declared_shortfall=False,
        )
    shortfall_manifest = prepare_training_bundle(
        source=src_img, target=tgt_img, public_semantic=sem, descriptions=desc_p, paraphrases=para_p,
        heldout_queries=held_p, candidates=cand_p, source_metadata=src_meta,
        output_dir=prep_dir / 'allowed_ok', test_fixture=True, allow_declared_shortfall=True,
    )
    loaded = load_training_bundle(
        shortfall_manifest['manifest_path'],
        source=src_img,
        target=tgt_img,
        public_semantic=sem,
        require_production=False,
    )
    assert len(loaded['queries']) == 92


def test_category_abstraction_mathematical_invariants():
    torch.manual_seed(7)
    dim = 16
    raw = F.normalize(torch.randn(24, dim, dtype=torch.float64), dim=-1)
    weights = torch.ones(24, dtype=torch.float64) / 24.0

    mu_cat = spherical_karcher_mean(raw, weights, steps=25)
    assert mu_cat.shape == (1, dim)
    assert abs(float(mu_cat.norm().item()) - 1.0) < 1e-8

    neg = F.normalize(torch.randn(16, dim, dtype=torch.float64), dim=-1)
    basis, proj_vars, anchors = build_contrastive_category_subspace(
        raw, weights, mu_cat, neg_vecs=neg, subspace_rank=5, contrastive_beta=0.35
    )
    k = int(basis.shape[0])
    assert 1 <= k <= 5
    eye = torch.eye(k, dtype=torch.float64)
    assert torch.allclose(basis @ basis.T, eye, atol=1e-8)
    assert torch.allclose(basis @ mu_cat.T, torch.zeros((k, 1), dtype=torch.float64), atol=1e-8)
    assert anchors.shape == (2 * k, dim)
    assert torch.allclose(anchors.norm(dim=-1), torch.ones(2 * k, dtype=torch.float64), atol=1e-8)

    den_id = denoise_queries_james_stein(raw, mu_cat, basis, 1.0)
    assert torch.allclose(den_id, F.normalize(raw, dim=-1), atol=1e-6)
    den_pure = denoise_queries_james_stein(raw, mu_cat, basis, 0.0)
    span = torch.cat([mu_cat, basis], dim=0)
    recon = (den_pure.to(torch.float64) @ span.T) @ span
    assert torch.allclose(den_pure.to(torch.float64), F.normalize(recon, dim=-1), atol=1e-8)

    sel1, score1 = lazy_greedy_facility_location(raw, weights, raw[:10], k=2)
    sel2, score2 = lazy_greedy_facility_location(raw, weights, raw[:10], k=5)
    assert len(sel1) == 2 and len(sel2) == 5 and len(set(sel2)) == 5
    assert score2 >= score1 - 1e-9

    sal_scores, _ = compute_contrastive_salience(
        [
            'BrandAlpha herbal moisturizing shampoo gentle clean',
            'BrandBeta herbal moisturizing shampoo fresh clean',
            'BrandGamma herbal moisturizing shampoo silk finish',
        ],
        ['BrandAlpha digital camera zoom lens'],
    )
    # Single-document brand token should be suppressed (df_cat=1 weight=0.05) relative to shared tokens (df_cat>=2).
    assert sal_scores['moisturizing'] > sal_scores['brandbeta']


def test_ambilage_single_model_pcgrad_multiscale_and_category_modes(tmp_path):
    # 1. Dominant Task-PCGrad non-conflict guarantee
    g_p = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float32)
    g_a = torch.tensor([-20.0, 10.0, 5.0], dtype=torch.float32)
    merged, conflicted = single_model_dominant_pcgrad(g_p, g_a, aux_cap=0.50)
    assert conflicted is True
    assert float(torch.dot(merged, g_p).item()) >= float(torch.dot(g_p, g_p).item()) - 1e-5
    orth_component = merged - (torch.dot(merged, g_p) / torch.dot(g_p, g_p)) * g_p
    assert float(orth_component.norm().item()) <= 0.50 * float(g_p.norm().item()) + 1e-5

    # 2. Multi-scale EoT view expansion (6 views on 1 surrogate vs 4 on 3)
    src_img, tgt_img = tmp_path / 'src.png', tmp_path / 'tgt.png'
    Image.new('RGB', (32, 32), (165, 115, 90)).save(src_img)
    Image.new('RGB', (32, 32), (70, 160, 180)).save(tgt_img)
    base_img = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    views_multi = multiscale_grid_breaking_views(base_img, step=1, base_views=torch.cat([base_img] * 4, dim=0))
    assert views_multi.shape[0] == 6

    cfg_1 = AmbiLAGEConfig(steps=1, surrogate_set='clip_l14', single_model_multiscale_eot=True, smoke_test=True)
    obj_1 = SequentialEnsembleObjective(
        [TinySurrogate(0)], base_img, 'source key', 'target text', cfg_1, purifier=TinyPurifier()
    )
    views_1, _, _ = obj_1._shared_inputs(base_img, step=1, gradient='none')
    assert views_1.shape[0] == 6

    cfg_3 = AmbiLAGEConfig(steps=1, surrogate_set='visual_inception_paper', single_model_multiscale_eot=True, smoke_test=True)
    obj_3 = SequentialEnsembleObjective(
        [TinySurrogate(i) for i in range(3)], base_img, 'source key', 'target text', cfg_3, purifier=TinyPurifier()
    )
    views_3, _, _ = obj_3._shared_inputs(base_img, step=1, gradient='none')
    assert views_3.shape[0] == 4

    # 3. End-to-end AmbiLAGEPipeline across all 3 category data modes
    manifest = _make_rich_bundle_fixture(tmp_path / 'bundle', (src_img, tgt_img), heldout_count=20)
    definition = _public_shampoo_definition()
    for mode in ('direct_vi_bundle', 'category_commonalities_abstraction', 'category_subspace_hybrid'):
        cfg = AmbiLAGEConfig(
            steps=2,
            surrogate_set='clip_l14',
            training_manifest=str(manifest),
            memory_key_mode='image',
            category_data_mode=mode,
            single_model_task_pcgrad=True,
            single_model_multiscale_eot=True,
            smoke_test=True,
        )
        run_dir = tmp_path / f'ambilage_{mode}'
        pipe = AmbiLAGEPipeline(cfg, device='cpu')
        adv_path, metadata = pipe.execute(
            src_img,
            tgt_img,
            run_dir,
            target_text=definition['target_feature_text'],
            source_public_key=definition['source_public_key'],
            source_category=definition['source_category'],
            definition=definition,
        )
        assert adv_path.is_file()
        to = metadata['training_objective']
        assert to['single_surrogate_mode'] is True
        assert to['single_model_task_pcgrad'] is True
        assert to['single_model_multiscale_eot'] is True
        assert to['effective_key_mode'] == 'image'
        assert to['effective_category_mode'] == mode
        if mode != 'direct_vi_bundle':
            assert 'category_abstraction' in to
        else:
            assert 'category_abstraction' not in to
        assert validate_ambilage_ready(run_dir, require_production=False)['status'] == 'completed'


def test_evaluate_transfer_panel_runs_three_encoders_sequentially(tmp_path, monkeypatch):
    from src.tests.test_visual_inception_category_evaluation import (
        GROUPS,
        _tiny_category_query_root,
        _tiny_frozen_suite,
        _write,
    )
    import src.visual.attack_completion as completion

    base, images, publics, documents = _tiny_frozen_suite(tmp_path)
    for group in GROUPS:
        documents[group]['metadata']['training_objective']['surrogates'] = [
            {'name': 'ViT-L-14', 'pretrained': 'openai', 'weight_sha256': '1' * 64}
        ]
        _write(images / group / 'visual_inception/FROZEN_IMAGE.json', documents[group])

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

    panel_out = tmp_path / 'output/transfer_panel'
    monkeypatch.setattr(category_eval, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(category_eval, 'public_group', lambda _base, group: publics[group])
    monkeypatch.setattr(completion, 'validate_attack_completion', lambda *args, **kwargs: None)
    monkeypatch.setattr(category_eval, 'validate_attack_completion', lambda *args, **kwargs: None)

    panel = category_eval.evaluate_transfer_panel(
        images,
        panel_out,
        base=base,
        query_root=qroot,
        method='visual_inception',
        key_mode='image',
        groups=list(GROUPS[:2]),
        replacement_groups='selected',
        device='cpu',
        defense_mode='none',
        encoder_factory=_DummyEncoder,
        verify_training_snapshots=False,
    )
    assert panel['encoder_order'] == ['01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m']
    assert panel['encoders']['01_clip_l14']['role'] == 'white_box_matched'
    assert panel['encoders']['02_openclip_bigG14']['role'] == 'black_box_heldout'
    assert panel['encoders']['03_siglip_so400m']['role'] == 'black_box_heldout'
    assert (panel_out / 'transfer_panel_summary.json').is_file()
    assert (panel_out / 'TRANSFER_PANEL_REPORT.md').is_file()
