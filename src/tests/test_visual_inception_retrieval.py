"""恢复检索目标的解析梯度、独立查询及微型成图验证。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
import torch
import torch.nn.functional as F

from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.objective import (
    retrieval_loss, retrieval_semantic_losses, sample_query_indices,
)
from src.visual._visual_inception.targets import prepare_semantic_targets, text_sha256
from src.visual.io import sha256_file


def public_definition():
    text = 'I want to buy shampoo with gentle formula and moisturizing ingredients.'
    return dict(target_feature_text=text, target_product_features=dict(category='shampoo',
        core_attributes=['gentle formula', 'moisturizing ingredients', 'fresh scent'],
        canonical_sentence=text))


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


@pytest.fixture
def tiny_pair(tmp_path):
    source, target = tmp_path / 'source.png', tmp_path / 'target.png'
    Image.new('RGB', (16, 16), (160, 120, 90)).save(source)
    Image.new('RGB', (16, 16), (80, 150, 180)).save(target)
    return source, target


def make_training_fixture(root, pair):
    """Every byte is synthetic and marked as a fixture, never as real search logs."""
    source, target = pair
    semantic = prepare_semantic_targets(source, target, definition=public_definition())
    sources, queries = {}, []
    kind_counts = [('category_description', 50), ('model_paraphrase', 30), ('heldout_search_query', 20)]
    heldout_ids = []
    for kind, count in kind_counts:
        rows = []
        for index in range(count):
            row_id = f'{kind}-{index:03d}'
            row = dict(id=row_id, text=f'Shampoo synthetic {kind} fixture wording {index:03d}.', category='shampoo')
            if kind == 'model_paraphrase':
                row.update(provider='minimax', model='MiniMax-M3',
                    effective_temperature=1.0, temperature_supported=False, temperature_parameter_sent=False,
                    input_row_id=f'category_description-{index:03d}')
            if kind == 'heldout_search_query':
                row['split'] = 'query_reserve'
                heldout_ids.append(row_id)
            rows.append(row)
            queries.append(dict(id=row_id, text=row['text'], source_type=kind,
                provenance=dict(source_id=kind, row_id=row_id, text_sha256=text_sha256(row['text']))))
        path = root / 'sources' / f'{kind}.jsonl'
        _write_rows(path, rows)
        record = dict(path=path.relative_to(root).as_posix(), sha256=sha256_file(path), kind=kind,
            split='query_reserve' if kind == 'heldout_search_query' else 'public_training',
            upstream=dict(dataset='synthetic_unit_fixture_not_real_queries', revision='fixture-v1',
                          url='https://example.invalid/synthetic-fixture'),
            test_fixture=True, is_synthetic=True)
        if kind == 'model_paraphrase':
            record.update(provider='minimax', model='MiniMax-M3',
                effective_temperature=1.0, temperature_supported=False, temperature_parameter_sent=False)
        sources[kind] = record
    candidates, catalog_rows = [], []
    for index, color in enumerate(((25, 70, 100), (220, 180, 100))):
        image = root / 'images' / f'negative-{index}.png'
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (16, 16), color).save(image)
        row_id = f'negative-{index}'
        image_sha = sha256_file(image)
        catalog_rows.append(dict(id=row_id, image_sha256=image_sha, category='shampoo'))
        candidates.append(dict(id=row_id, image=image.relative_to(root).as_posix(), image_sha256=image_sha,
            provenance=dict(source_id='catalog', row_id=row_id)))
    catalog = root / 'sources' / 'candidate_catalog.jsonl'
    _write_rows(catalog, catalog_rows)
    sources['catalog'] = dict(path=catalog.relative_to(root).as_posix(), sha256=sha256_file(catalog),
        kind='candidate_catalog', split='public_training', test_fixture=True, is_synthetic=True,
        upstream=dict(dataset='synthetic_unit_fixture_not_real_queries', revision='fixture-v1',
                      url='https://example.invalid/synthetic-fixture'))
    document = dict(schema_version=1, test_fixture=True, category='shampoo', paraphrase_policy='declared_model',
        target_binding=dict(source_image_sha256=sha256_file(source), target_image_sha256=sha256_file(target),
                            canonical_text_sha256=semantic['canonical_text_sha256']),
        target_concept_text='shampoo: gentle formula; moisturizing ingredients; fresh scent',
        queries=queries, candidates=candidates, sources=sources,
        formal_queries_used=False, victim_memory_used=False,
        split_manifest=dict(policy='independent_public_training_query_reserve_v1',
            evaluation_partition_excluded=True, formal_query_sources_used=False, heldout_query_ids=heldout_ids))
    manifest = root / 'training_bundle.json'
    _write_json(manifest, document)
    return manifest, document, semantic


def test_dynamic_denominator_counts_current_image_once_and_matches_closed_form():
    image = torch.tensor([[.8, .6]], dtype=torch.float64, requires_grad=True)
    query = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    negatives = torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64, requires_grad=True)
    temperature = .4
    positive = F.normalize(image, dim=-1) @ F.normalize(query, dim=-1).T
    negative_scores = F.normalize(query, dim=-1) @ F.normalize(negatives.detach(), dim=-1).T
    expected = torch.log1p(torch.exp((negative_scores - positive.T) / temperature).sum(-1)).mean()
    actual = retrieval_loss(image, query, negatives, temperature=temperature)
    assert actual.item() >= 0
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    grad_image, grad_negative = torch.autograd.grad(actual, (image, negatives), allow_unused=True)
    assert torch.isfinite(grad_image).all() and grad_image.norm() > 0
    assert grad_negative is None


def test_retrieval_gradient_matches_central_finite_differences():
    value = torch.tensor([[.8, .6]], dtype=torch.float64, requires_grad=True)
    query = torch.tensor([[1., 0.], [.2, .98]], dtype=torch.float64)
    negative = torch.tensor([[.4, .9], [-.8, -.3]], dtype=torch.float64)
    loss = retrieval_loss(value, query, negative, temperature=.3)
    analytic, = torch.autograd.grad(loss, value)
    epsilon = 1e-6
    numeric = torch.zeros_like(value)
    for coordinate in range(2):
        direction = torch.zeros_like(value)
        direction[0, coordinate] = epsilon
        positive = retrieval_loss(value.detach() + direction, query, negative, temperature=.3)
        negative_loss = retrieval_loss(value.detach() - direction, query, negative, temperature=.3)
        numeric[0, coordinate] = (positive - negative_loss) / (2 * epsilon)
    assert torch.allclose(analytic, numeric, atol=2e-7, rtol=2e-6)


def test_fixed_candidates_change_relative_query_gradient_weights():
    query = torch.eye(2, dtype=torch.float64)
    gradients = []
    for candidate in (torch.tensor([[1., 0.]], dtype=torch.float64), torch.tensor([[0., 1.]], dtype=torch.float64)):
        image = torch.tensor([[.7, .7]], dtype=torch.float64, requires_grad=True)
        loss = retrieval_loss(image, query, candidate, temperature=.15)
        gradient, = torch.autograd.grad(loss, image)
        gradients.append(gradient)
    assert not torch.allclose(*gradients, atol=1e-7)
    assert gradients[0][0, 0] < gradients[0][0, 1]
    assert gradients[1][0, 1] < gradients[1][0, 0]


def test_single_concept_and_queries_have_separate_loss_roles():
    image = torch.tensor([[.8, .6]])
    concept = torch.tensor([[1., 0.]])
    query = torch.tensor([[0., 1.], [-1., 0.]])
    negative = torch.tensor([[.2, .9]])
    semantic, retrieval = retrieval_semantic_losses(image, concept, query, negative, temperature=.2)
    expected_semantic = 1 - F.cosine_similarity(image, concept).mean()
    assert torch.allclose(semantic, expected_semantic)
    other_semantic, other_retrieval = retrieval_semantic_losses(image, concept, -query, negative, temperature=.2)
    assert torch.equal(semantic, other_semantic)
    assert not torch.allclose(retrieval, other_retrieval)
    with pytest.raises(ValueError):
        retrieval_semantic_losses(image, torch.cat((concept, concept)), query, negative)


def test_query_sampling_is_seeded_without_replacement_inside_each_step():
    first = sample_query_indices(100, 16, seed=42, step=1)
    assert first == sample_query_indices(100, 16, seed=42, step=1)
    assert len(first) == len(set(first)) == 16 and min(first) >= 0 and max(first) < 100
    assert first != sample_query_indices(100, 16, seed=42, step=2)
    assert first != sample_query_indices(100, 16, seed=43, step=1)


class ColorSurrogate:
    device = torch.device('cpu')

    def encode_image(self, image):
        return SimpleNamespace(global_embedding=image.mean((-2, -1)))


def test_sequential_retrieval_semantic_perceptual_gradient_matches_joint_graph():
    from src.visual._visual_inception.pipeline import _gradient
    image = torch.tensor([.7, .2, .4]).reshape(1, 3, 1, 1).expand(1, 3, 4, 4).clone()
    source = .8 * image
    concepts = [torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 1., 0.]])]
    queries = [torch.eye(3), torch.tensor([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]])]
    negatives = [torch.tensor([[.1, .8, .1], [.9, .1, .1]]), torch.tensor([[.2, .1, .8], [.2, .8, .1]])]
    weights, indices = [.25, .75], [0, 2]
    config = VisualInceptionConfig(semantic_weight=1.7, retrieval_weight=.8,
        retrieval_temperature=.3, perceptual_weight=.3)
    state = dict(concept_features=concepts, query_features=queries, negative_features=negatives)
    distance = lambda origin, changed: (origin - changed).square().mean()
    gradient, record = _gradient(image, source, [ColorSurrogate(), ColorSurrogate()],
        concepts, weights, distance, config, retrieval_state=state, query_indices=indices)
    joint_image = image.detach().requires_grad_(True)
    feature = F.normalize(joint_image.mean((-2, -1)), dim=-1)
    loss = 0
    for weight, concept, query, negative in zip(weights, concepts, queries, negatives):
        semantic = 1 - (feature @ F.normalize(concept, dim=-1).T).mean()
        positive = feature @ F.normalize(query[indices], dim=-1).T
        negative_logits = F.normalize(query[indices], dim=-1) @ F.normalize(negative, dim=-1).T
        logits = torch.cat((positive.T, negative_logits), dim=-1) / config.retrieval_temperature
        retrieval = F.cross_entropy(logits, torch.zeros(len(indices), dtype=torch.long))
        loss = loss + weight * (config.semantic_weight * semantic + config.retrieval_weight * retrieval)
    loss = loss + config.perceptual_weight * distance(source, joint_image)
    expected, = torch.autograd.grad(loss, joint_image)
    assert torch.allclose(gradient, expected, atol=2e-7, rtol=2e-6)
    assert record['loss'] == pytest.approx(float(loss.detach()), rel=2e-6)


def test_missing_training_manifest_stops_before_model_loading(tiny_pair, tmp_path, monkeypatch):
    from src.visual import training_objectives
    from src.visual._visual_inception.pipeline import generate_visual_inception
    monkeypatch.setattr(training_objectives, 'load_training_surrogates',
        lambda *args, **kwargs: pytest.fail('缺少训练清单时不能装载真实模型'))
    with pytest.raises((ValueError, FileNotFoundError), match='清单|manifest|训练'):
        generate_visual_inception(*tiny_pair, tmp_path / 'no_manifest', device='cpu',
            definition=public_definition(), steps=1, smoke_test=True)
    assert not (tmp_path / 'no_manifest' / 'ready.json').exists()


def _load_fixture(manifest, pair, semantic):
    from src.visual._visual_inception.training_data import load_training_bundle
    return load_training_bundle(manifest, source=pair[0], target=pair[1],
        public_semantic=semantic, require_production=False)


def test_declared_paraphrase_model_is_recorded_without_claiming_paper_generator(tiny_pair, tmp_path):
    manifest, document, semantic = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    bundle = _load_fixture(manifest, tiny_pair, semantic)
    assert len(bundle['queries']) == len(set(bundle['queries'])) == 100
    assert bundle['target_concept_text'] == document['target_concept_text']
    assert bundle['target_concept_text'] not in bundle['queries']
    assert len(bundle['candidates']) == 2
    assert bundle['audit']['paper_query_generator_reproduced'] is False
    from src.visual._visual_inception.training_data import load_training_bundle
    with pytest.raises(ValueError, match='夹具|fixture|生产'):
        load_training_bundle(manifest, source=tiny_pair[0], target=tiny_pair[1],
            public_semantic=semantic, require_production=True)


@pytest.mark.parametrize('change', [
    'missing_query', 'duplicate_query', 'wrong_target_binding', 'buyer_intent_as_concept',
    'source_as_negative', 'heldout_source_split', 'missing_provenance_row', 'strict_gpt4_with_minimax',
])
def test_training_manifest_rejects_incomplete_or_misrepresented_inputs(tiny_pair, tmp_path, change):
    manifest, document, semantic = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    if change == 'missing_query':
        document['queries'].pop()
    elif change == 'duplicate_query':
        document['queries'][-1]['text'] = document['queries'][-2]['text']
        document['queries'][-1]['provenance']['text_sha256'] = text_sha256(document['queries'][-1]['text'])
    elif change == 'wrong_target_binding':
        document['target_binding']['target_image_sha256'] = '0' * 64
    elif change == 'buyer_intent_as_concept':
        document['target_concept_text'] = semantic['canonical_text']
    elif change == 'source_as_negative':
        candidate = document['candidates'][0]
        image = manifest.parent / candidate['image']
        image.write_bytes(tiny_pair[0].read_bytes())
        candidate['image_sha256'] = sha256_file(image)
        catalog = manifest.parent / document['sources']['catalog']['path']
        catalog_rows = [json.loads(line) for line in catalog.read_text(encoding='utf-8').splitlines()]
        catalog_rows[0]['image_sha256'] = candidate['image_sha256']
        _write_rows(catalog, catalog_rows)
        document['sources']['catalog']['sha256'] = sha256_file(catalog)
    elif change == 'heldout_source_split':
        document['sources']['heldout_search_query']['split'] = 'public_training'
    elif change == 'missing_provenance_row':
        document['queries'][0]['provenance']['row_id'] = 'does-not-exist'
    elif change == 'strict_gpt4_with_minimax':
        document['paraphrase_policy'] = 'paper_gpt4'
    _write_json(manifest, document)
    with pytest.raises((ValueError, FileNotFoundError)):
        _load_fixture(manifest, tiny_pair, semantic)


def test_provenance_source_bytes_and_candidate_image_bytes_are_verified(tiny_pair, tmp_path):
    manifest, document, semantic = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    candidate = manifest.parent / document['candidates'][0]['image']
    before = candidate.read_bytes()
    Image.new('RGB', (16, 16), (0, 0, 0)).save(candidate)
    with pytest.raises(ValueError, match='摘要|hash|内容'):
        _load_fixture(manifest, tiny_pair, semantic)
    candidate.write_bytes(before)
    description = manifest.parent / document['sources']['category_description']['path']
    description.write_text(description.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='摘要|hash|来源'):
        _load_fixture(manifest, tiny_pair, semantic)


@pytest.mark.parametrize('duplicate', ['source', 'another_negative'])
def test_differently_encoded_duplicate_pixels_do_not_add_a_clean_positive_slot(tiny_pair, tmp_path, duplicate):
    manifest, document, semantic = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    candidate = document['candidates'][0]
    path = manifest.parent / candidate['image']
    original = tiny_pair[0] if duplicate == 'source' else manifest.parent / document['candidates'][1]['image']
    with Image.open(original) as image:
        image.convert('RGB').save(path, compress_level=0)
    assert sha256_file(path) != sha256_file(original)
    candidate['image_sha256'] = sha256_file(path)
    catalog = manifest.parent / document['sources']['catalog']['path']
    catalog_rows = [json.loads(line) for line in catalog.read_text(encoding='utf-8').splitlines()]
    catalog_rows[0]['image_sha256'] = candidate['image_sha256']
    _write_rows(catalog, catalog_rows)
    document['sources']['catalog']['sha256'] = sha256_file(catalog)
    _write_json(manifest, document)
    with pytest.raises(ValueError):
        _load_fixture(manifest, tiny_pair, semantic)


def test_paper_policy_rejects_unsupported_temperature_even_if_model_label_changes(tiny_pair, tmp_path):
    manifest, document, semantic = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    document['paraphrase_policy'] = 'paper_gpt4'
    provenance = document['sources']['model_paraphrase']
    provenance.update(provider='openai', model='gpt-4-0613')
    path = manifest.parent / provenance['path']
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    for row in rows:
        row.update(provider='openai', model='gpt-4-0613')
    _write_rows(path, rows)
    provenance['sha256'] = sha256_file(path)
    _write_json(manifest, document)
    with pytest.raises(ValueError):
        _load_fixture(manifest, tiny_pair, semantic)


class CachedColorSurrogate(ColorSurrogate):
    def __init__(self, weight='a', preprocessing='unit_pixels'):
        self.weight, self.preprocessing, self.calls = weight, preprocessing, 0

    def snapshot(self):
        return dict(name='explicit_test_color_encoder', test_fixture=True,
            weight_sha256=self.weight * 64, preprocess=self.preprocessing)

    def encode_image(self, image):
        self.calls += 1
        return super().encode_image(image)


def test_candidate_cache_reuses_identical_identity_and_rebuilds_for_real_changes(tiny_pair, tmp_path):
    from src.visual._visual_inception.retrieval_cache import encode_candidate_cache
    manifest, document, semantic = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    bundle = _load_fixture(manifest, tiny_pair, semantic)
    candidates, root = bundle['candidates'], tmp_path / 'cache'
    model = CachedColorSurrogate()
    vectors, first = encode_candidate_cache(model, candidates, root,
        bundle_sha256=bundle['bundle_sha256'], fixture=True)
    assert model.calls == len(candidates) and first['cache_hit'] is False
    again, second = encode_candidate_cache(model, candidates, root,
        bundle_sha256=bundle['bundle_sha256'], fixture=True)
    assert model.calls == len(candidates) and second['cache_hit'] is True
    assert first['cache_key'] == second['cache_key'] and torch.equal(vectors, again)
    assert not vectors.requires_grad
    changed_weight = CachedColorSurrogate(weight='b')
    _, weight_record = encode_candidate_cache(changed_weight, candidates, root,
        bundle_sha256=bundle['bundle_sha256'], fixture=True)
    assert weight_record['cache_hit'] is False and weight_record['cache_key'] != first['cache_key']
    changed_preprocessing = CachedColorSurrogate(preprocessing='another_explicit_preprocess')
    _, preprocess_record = encode_candidate_cache(changed_preprocessing, candidates, root,
        bundle_sha256=bundle['bundle_sha256'], fixture=True)
    assert preprocess_record['cache_hit'] is False and preprocess_record['cache_key'] != first['cache_key']
    altered = copy.deepcopy(candidates)
    image = Path(altered[0]['image_path'])
    Image.new('RGB', (16, 16), (110, 20, 60)).save(image)
    altered[0]['image_sha256'] = sha256_file(image)
    _, image_record = encode_candidate_cache(model, altered, root,
        bundle_sha256=bundle['bundle_sha256'], fixture=True)
    assert image_record['cache_hit'] is False and image_record['cache_key'] != first['cache_key']


def test_candidate_cache_rejects_vector_tampering_and_wrong_expected_encoder(tiny_pair, tmp_path):
    from src.visual._visual_inception.retrieval_cache import (
        encode_candidate_cache, candidate_cache_identity, validate_candidate_cache,
    )
    manifest, _, semantic = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    bundle = _load_fixture(manifest, tiny_pair, semantic)
    model = CachedColorSurrogate()
    _, record = encode_candidate_cache(model, bundle['candidates'], tmp_path / 'cache',
        bundle_sha256=bundle['bundle_sha256'], fixture=True)
    wrong_identity = candidate_cache_identity(CachedColorSurrogate(weight='b'), bundle['candidates'],
        bundle_sha256=bundle['bundle_sha256'], fixture=True)
    with pytest.raises(ValueError):
        validate_candidate_cache(record, wrong_identity)
    path = Path(record['vector_path'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='摘要|hash|改变'):
        validate_candidate_cache(record)


def test_full_micro_generation_has_separate_target_queries_and_live_positive(tiny_pair, tmp_path):
    from src.visual._visual_inception.pipeline import generate_visual_inception
    from src.visual._visual_inception.evidence import validate_ready
    manifest, document, semantic = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    config = VisualInceptionConfig(steps=2, gradient_restarts=2, smoke_test=True,
        training_manifest=str(manifest), candidate_cache_dir=str(tmp_path / 'candidate_cache'))
    output, metadata = generate_visual_inception(*tiny_pair, tmp_path / 'full', config=config,
        device='cpu', definition=public_definition())
    assert output.is_file()
    assert metadata['training_objective']['kind'] == 'visual_inception_retrieval_semantic_perceptual_v2'
    assert metadata['retrieval_loss_omitted'] is False
    sampling = metadata['query_sampling']
    assert sampling['pool_size'] == 100 and sampling['batch_size'] == 16
    assert sampling['calibration_indices'] == sample_query_indices(100, 16, config.seed, 0)
    assert sampling['step_indices'] == [sample_query_indices(100, 16, config.seed, step) for step in (1, 2)]
    assert len(metadata['negative_candidate_caches']) == 3
    assert metadata['decoded_image_audit']['within_budget'] is True
    assert metadata['victim_memory_used'] is metadata['formal_queries_used'] is False
    assert validate_ready(output.parent, require_production=False)['retrieval_loss_omitted'] is False
    with pytest.raises(ValueError):
        validate_ready(output.parent)


def test_same_full_training_inputs_reuse_only_frozen_candidate_vectors(tiny_pair, tmp_path):
    from src.visual._visual_inception.pipeline import generate_visual_inception
    manifest, _, _ = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    config = VisualInceptionConfig(steps=1, gradient_restarts=2, smoke_test=True,
        training_manifest=str(manifest), candidate_cache_dir=str(tmp_path / 'shared_fixed_candidate_cache'))
    first, first_metadata = generate_visual_inception(*tiny_pair, tmp_path / 'first', config=config,
        device='cpu', definition=public_definition())
    second, second_metadata = generate_visual_inception(*tiny_pair, tmp_path / 'second', config=config,
        device='cpu', definition=public_definition())
    assert sha256_file(first) == sha256_file(second)
    assert all(not row['cache_hit'] for row in first_metadata['negative_candidate_caches'])
    assert all(row['cache_hit'] for row in second_metadata['negative_candidate_caches'])
    assert first_metadata['query_sampling'] == second_metadata['query_sampling']


def test_full_ready_rejects_changed_training_manifest_after_image_generation(tiny_pair, tmp_path):
    from src.visual._visual_inception.pipeline import generate_visual_inception
    from src.visual._visual_inception.evidence import validate_ready
    manifest, document, _ = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    config = VisualInceptionConfig(steps=1, gradient_restarts=2, smoke_test=True, training_manifest=str(manifest))
    output, _ = generate_visual_inception(*tiny_pair, tmp_path / 'full', config=config,
        device='cpu', definition=public_definition())
    document['queries'][0]['text'] += ' changed'
    _write_json(manifest, document)
    with pytest.raises(ValueError):
        validate_ready(output.parent, require_production=False)


def test_legacy_evidence_without_new_configuration_fields_still_validates(tiny_pair, tmp_path):
    from src.visual._visual_inception.pipeline import generate_visual_inception
    from src.visual._visual_inception.evidence import validate_ready
    from src.visual._visual_inception.targets import content_sha256
    config = VisualInceptionConfig(objective_mode='semantic_perceptual', steps=1,
        gradient_restarts=2, smoke_test=True)
    output, metadata = generate_visual_inception(*tiny_pair, tmp_path / 'legacy', config=config,
        device='cpu', definition=public_definition())
    for key in ('objective_mode', 'training_manifest', 'retrieval_weight', 'retrieval_temperature',
                'query_batch_size', 'candidate_cache_dir'):
        metadata['config'].pop(key)
    metadata['config_sha256'] = content_sha256(metadata['config'])
    root = output.parent
    _write_json(root / 'metadata.json', metadata)
    _write_json(root / 'attack_metrics.json', metadata)
    ready = json.loads((root / 'ready.json').read_text(encoding='utf-8'))
    ready.update(config_sha256=metadata['config_sha256'], metadata_sha256=sha256_file(root / 'metadata.json'),
        attack_metrics_sha256=sha256_file(root / 'attack_metrics.json'))
    _write_json(root / 'ready.json', ready)
    assert validate_ready(root, require_production=False)['training_objective']['kind'].endswith('no_retrieval_v1')


def import_fixture_arguments(root, tiny_pair, output):
    manifest, document, semantic = make_training_fixture(root, tiny_pair)
    catalog_path = manifest.parent / document['sources']['catalog']['path']
    catalog_rows = [json.loads(line) for line in catalog_path.read_text(encoding='utf-8').splitlines()]
    for row, candidate in zip(catalog_rows, document['candidates']):
        row['image'] = '../' + candidate['image']
    _write_rows(catalog_path, catalog_rows)
    kinds = {record['kind']: record for record in document['sources'].values()}
    paths = {kind: manifest.parent / record['path'] for kind, record in kinds.items()}
    metadata = {kind: {key: value for key, value in record.items() if key not in ('path', 'sha256')}
                for kind, record in kinds.items()}
    return dict(source=tiny_pair[0], target=tiny_pair[1], public_semantic=semantic,
        descriptions=paths['category_description'], paraphrases=paths['model_paraphrase'],
        heldout_queries=paths['heldout_search_query'], candidates=paths['candidate_catalog'],
        source_metadata=metadata, output_dir=output, test_fixture=True)


def test_public_data_import_freezes_a_self_contained_bundle_without_overwriting(tiny_pair, tmp_path):
    from src.visual._visual_inception.data_preparation import prepare_training_bundle
    kwargs = import_fixture_arguments(tmp_path / 'import_sources', tiny_pair, tmp_path / 'frozen_import')
    bundle = prepare_training_bundle(**kwargs)
    assert len(bundle['queries']) == 100 and len(bundle['candidates']) == 2
    frozen = Path(bundle['manifest_path'])
    assert all(Path(row['image_path']).is_relative_to(frozen.parent) for row in bundle['candidates'])
    before = frozen.read_bytes()
    with pytest.raises(ValueError, match='覆盖|非空'):
        prepare_training_bundle(**kwargs)
    assert frozen.read_bytes() == before


def test_import_rejects_private_original_candidate_before_hash_or_copy(tiny_pair, tmp_path, monkeypatch):
    from src.visual._visual_inception import training_data, data_preparation
    private_scope = tmp_path / 'explicit_private_scope'
    private_image = private_scope / 'datasets' / 'private' / 'private_fixture.png'
    private_image.parent.mkdir(parents=True)
    Image.new('RGB', (16, 16), (1, 2, 3)).save(private_image)
    kwargs = import_fixture_arguments(tmp_path / 'import_sources', tiny_pair, tmp_path / 'rejected_import')
    catalog = kwargs['candidates']
    rows = [json.loads(line) for line in catalog.read_text(encoding='utf-8').splitlines()]
    rows[0].update(image=str(private_image), image_sha256=sha256_file(private_image))
    _write_rows(catalog, rows)
    monkeypatch.setattr(training_data, 'OUTPUT', private_scope)
    original_open = Path.open
    original_copy = data_preparation.shutil.copyfile
    touched = []

    def tracked_open(path, *args, **kwargs):
        if path.resolve() == private_image.resolve():
            touched.append('opened')
            pytest.fail('不允许打开私有候选原图')
        return original_open(path, *args, **kwargs)

    def tracked_copy(source, destination, *args, **kwargs):
        if Path(source).resolve() == private_image.resolve():
            touched.append('copied')
            pytest.fail('不允许复制私有候选原图')
        return original_copy(source, destination, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', tracked_open)
    monkeypatch.setattr(data_preparation.shutil, 'copyfile', tracked_copy)
    with pytest.raises(ValueError, match='私有|私人|正式'):
        data_preparation.prepare_training_bundle(**kwargs)
    assert touched == []
    assert not (kwargs['output_dir'] / 'images').exists()


def test_full_command_line_passes_training_manifest_into_micro_generation(tiny_pair, tmp_path):
    import subprocess
    import sys
    from src.paths import PROJECT
    from src.visual._visual_inception.evidence import validate_ready
    manifest, _, _ = make_training_fixture(tmp_path / 'bundle', tiny_pair)
    definition = tmp_path / 'definition.json'
    _write_json(definition, public_definition())
    configuration = tmp_path / 'configuration.json'
    _write_json(configuration, dict(gradient_restarts=2))
    output = tmp_path / 'full_cli'
    completed = subprocess.run([
        sys.executable, '-B', '-X', 'utf8', str(PROJECT / 'tools/generate.py'),
        '--method', 'visual_inception', '--source', str(tiny_pair[0]), '--target', str(tiny_pair[1]),
        '--definition', str(definition), '--visual-inception-config', str(configuration),
        '--training-manifest', str(manifest), '--output', str(output),
        '--device', 'cpu', '--steps', '1', '--smoke-test',
    ], cwd=PROJECT, capture_output=True, text=True, encoding='utf-8', timeout=45)
    assert completed.returncode == 0, completed.stdout + '\n' + completed.stderr
    metadata = validate_ready(output, require_production=False)
    assert metadata['config']['objective_mode'] == 'retrieval_semantic_perceptual'
    assert Path(metadata['config']['training_manifest']).resolve() == manifest.resolve()
    assert metadata['training_objective']['kind'] == 'visual_inception_retrieval_semantic_perceptual_v2'
    assert metadata['training_objective']['retrieval_positive_count'] == 1
    assert metadata['query_sampling']['pool_size'] == 100



