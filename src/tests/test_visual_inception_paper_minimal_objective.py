"""Micro vector tests for paper_minimal_margin: static 3-source negatives, Eq. (3) InfoNCE (+0.3 margin), and Gaussian query noise."""
from dataclasses import replace
import json

import pytest
import torch
import torch.nn.functional as F

from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.paper_minimal_objective import (
    PAPER_MINIMAL_MARGIN_RETRIEVAL_KIND,
    PAPER_MINIMAL_QUERY_PERTURBATION_KIND,
    STATIC_POLICY_KIND,
    STATIC_ROLE_ORDER,
    build_static_negative_selection,
    paper_minimal_noise_seed,
    paper_minimal_retrieval_semantic_losses,
    perturb_queries_gaussian,
)


def _config(**changes):
    base = VisualInceptionConfig.from_dict({
        'mechanism_profile': 'paper_minimal_margin',
        'contrastive_margin': 0.3,
        'negative_same_category_count': 2,
        'negative_proxy_memory_count': 2,
        'negative_popularity_count': 2,
    })
    return replace(base, **changes)


def _inputs(*, gradients=False):
    image = torch.tensor([[0.8, 0.6, 0.2]], dtype=torch.float64, requires_grad=gradients)
    concept = torch.tensor([[0.9, 0.2, 0.1]], dtype=torch.float64, requires_grad=gradients)
    queries = torch.tensor([[0.7, 0.2, 0.5], [0.1, 0.9, 0.3]], dtype=torch.float64, requires_grad=gradients)
    negatives = torch.tensor([
        [1.0, 0.1, 0.2],
        [0.1, 1.0, 0.3],
        [0.2, 0.4, 1.0],
        [-0.5, 0.3, 0.7],
        [0.4, -0.8, 0.2],
        [-0.3, -0.2, 1.0],
        [0.6, 0.6, 0.1],
        [-0.7, 0.5, -0.2],
        [0.3, -0.4, 0.8],
    ], dtype=torch.float64, requires_grad=gradients)
    metadata = [
        dict(id=f'fixture_{i}', category='shampoo' if i < 3 else 'camera' if i < 6 else 'jeans')
        for i in range(9)
    ]
    cfg = _config()
    static_sel = build_static_negative_selection(metadata, negatives, 'shampoo', config=cfg)
    return image, concept, queries, negatives, static_sel, cfg


def test_static_negative_sampling_determinism_disjointness_and_global_popularity():
    vectors = torch.tensor([
        [0.0, 1.0, 0.0], [0.0, 0.99, 0.05], [0.0, 0.99, -0.05],
        [1.0, 0.0, 0.0], [0.99, 0.0, 0.08], [0.99, 0.0, -0.08],
        [0.98, 0.0, 0.12], [0.98, 0.0, -0.12], [0.97, 0.0, 0.16],
    ])
    metadata = [dict(id=f'item_{i}', category='shampoo' if i < 3 else 'jeans') for i in range(9)]
    cfg = _config(seed=42)
    sel_a = build_static_negative_selection(metadata, vectors, 'SHAMPOO', config=cfg)
    sel_b = build_static_negative_selection(metadata, vectors.double(), 'shampoo', config=cfg)

    assert sel_a == sel_b
    assert sel_a['kind'] == STATIC_POLICY_KIND
    assert sel_a['role_order'] == list(STATIC_ROLE_ORDER)
    assert sel_a['popularity']['method'] == 'query_independent_global_cosine_centrality_proxy'
    assert sel_a['popularity']['true_public_popularity_used'] is False
    assert sel_a['role_indices']['popular'] == [3, 4, 6]
    assert set(sel_a['role_indices']['popular']).isdisjoint({0, 1, 2})
    assert sel_a['role_indices']['same_category'] == [0, 1, 2]
    assert sel_a['role_indices']['proxy_user_memory'] == [3, 4, 5, 6, 7, 8]
    assert sel_a['cross_category_proxy_disjoint_from_same_category'] is True
    assert sel_a['selected_count'] == 6
    assert len(set(sel_a['selected_indices'])) == 6
    assert sel_a['role_selected_counts'] == {'same_category': 2, 'popular': 2, 'proxy_user_memory': 2}

    sel_diff_seed = build_static_negative_selection(metadata, vectors, 'shampoo', config=_config(seed=123))
    assert sel_diff_seed['selected_indices'] != sel_a['selected_indices']


def test_static_negative_sampling_real_public_popularity_and_shortfall_without_duplication():
    metadata = [dict(id=f'item_{i}', category='shampoo' if i < 3 else 'jeans') for i in range(9)]
    for i, row in enumerate(metadata):
        row['public_popularity'] = dict(
            value=i, metric='public_review_count', source='https://example.org/public-records',
        )
    sel = build_static_negative_selection(metadata, torch.ones(9, 2), 'shampoo', config=_config())
    assert sel['popularity']['true_public_popularity_used'] is True
    assert sel['role_indices']['popular'] == [8, 7, 6]

    small_meta = [dict(id='a', category='shampoo'), dict(id='b', category='jeans')]
    small_cfg = _config(
        negative_same_category_count=10,
        negative_popularity_count=10,
        negative_proxy_memory_count=10,
    )
    small_sel = build_static_negative_selection(small_meta, torch.eye(2), 'shampoo', config=small_cfg)
    assert small_sel['selected_count'] == 2
    assert len(set(small_sel['selected_indices'])) == 2
    assert any(small_sel['role_shortfall'].values())


@pytest.mark.parametrize('batch_size,expected_count', [(16, 4), (100, 25)])
def test_gaussian_query_perturbation_subset_determinism_and_disable(batch_size, expected_count):
    gen = torch.Generator(device='cpu').manual_seed(99)
    raw_queries = torch.randn((batch_size, 32), generator=gen, dtype=torch.float64)
    unit_queries = F.normalize(raw_queries, dim=-1)

    seed = paper_minimal_noise_seed(42, stage='step', step_index=3, model_index=0)
    p1, a1 = perturb_queries_gaussian(raw_queries, std=0.03, fraction=0.25, seed=seed)
    p2, a2 = perturb_queries_gaussian(raw_queries, std=0.03, fraction=0.25, seed=seed)

    assert torch.equal(p1, p2)
    assert a1 == a2
    assert a1['algorithm'] == PAPER_MINIMAL_QUERY_PERTURBATION_KIND
    assert a1['enabled'] is True
    assert a1['single_level_no_inner_gradient'] is True
    assert a1['subspace_projection_used'] is False
    assert a1['queries_detached_for_outer'] is True
    assert not p1.requires_grad
    assert a1['perturbed_query_count'] == expected_count
    assert len(a1['perturbed_row_indices']) == expected_count
    assert torch.allclose(p1.norm(dim=-1), torch.ones(batch_size, dtype=torch.float64), atol=1e-12)

    perturbed_set = set(a1['perturbed_row_indices'])
    unperturbed = [i for i in range(batch_size) if i not in perturbed_set]
    assert torch.equal(p1[unperturbed], unit_queries[unperturbed])
    assert not torch.allclose(p1[a1['perturbed_row_indices']], unit_queries[a1['perturbed_row_indices']])

    p_diff, _ = perturb_queries_gaussian(raw_queries, std=0.03, fraction=0.25, seed=seed + 1)
    assert not torch.equal(p1, p_diff)

    p_zero_std, a_zero_std = perturb_queries_gaussian(raw_queries, std=0.0, fraction=0.25, seed=seed)
    assert a_zero_std['enabled'] is False and a_zero_std['perturbed_query_count'] == 0
    assert torch.equal(p_zero_std, unit_queries)

    p_zero_frac, a_zero_frac = perturb_queries_gaussian(raw_queries, std=0.03, fraction=0.0, seed=seed)
    assert a_zero_frac['enabled'] is False and a_zero_frac['perturbed_query_count'] == 0
    assert torch.equal(p_zero_frac, unit_queries)


def test_eq3_infonce_with_uniform_logit_margin_closed_form_and_finite_differences():
    image, concept, queries, negatives, static_sel, cfg = _inputs(gradients=True)
    cfg_margin = replace(cfg, contrastive_margin=0.3, query_noise_std=0.0)
    cfg_zero = replace(cfg_margin, contrastive_margin=0.0)
    seed = paper_minimal_noise_seed(42, stage='step', step_index=1, model_index=0)

    semantic, loss, audit = paper_minimal_retrieval_semantic_losses(
        image, concept, queries, negatives, static_selection=static_sel,
        config=cfg_margin, noise_seed=seed, mechanism_profile='paper_minimal_margin',
    )
    _, zero_loss, _ = paper_minimal_retrieval_semantic_losses(
        image, concept, queries, negatives, static_selection=static_sel,
        config=cfg_zero, noise_seed=seed, mechanism_profile='paper_minimal_margin',
    )
    q = F.normalize(queries, dim=-1)
    im = F.normalize(image, dim=-1)
    static_n = F.normalize(negatives[static_sel['selected_indices']], dim=-1)
    tau, margin = 0.07, 0.3
    pos_logit = (q @ im.T)[:, 0] / tau
    neg_logits = (q @ static_n.T) / tau + margin
    expected = (torch.logsumexp(torch.cat((pos_logit[:, None], neg_logits), dim=1), dim=1) - pos_logit).mean()

    assert torch.allclose(loss, expected, atol=1e-12)
    assert torch.allclose(semantic, (1.0 - im @ F.normalize(concept, dim=-1).T).mean(), atol=1e-12)
    assert loss > zero_loss
    assert audit['mechanism_profile'] == 'paper_minimal_margin'
    assert audit['retrieval_algorithm'] == PAPER_MINIMAL_MARGIN_RETRIEVAL_KIND
    assert audit['margin_loss_added'] is False
    assert audit['single_level_optimization'] is True
    assert audit['full_bank_denominator_preserved'] is False
    assert audit['contrastive_margin'] == 0.3
    assert audit['logit_margin_applied_to_static_negatives'] is True
    assert audit['query_perturbation']['enabled'] is False
    json.dumps(audit, allow_nan=False)

    def objective(val):
        sem, ret, _ = paper_minimal_retrieval_semantic_losses(
            val, concept, queries, negatives, static_selection=static_sel,
            config=cfg_margin, noise_seed=seed, mechanism_profile='paper_minimal_margin',
        )
        return sem + ret

    total = objective(image)
    gradient, = torch.autograd.grad(total, image)
    for col in range(image.shape[1]):
        delta = torch.zeros_like(image)
        delta[0, col] = 1e-6
        numerical = (objective(image.detach() + delta) - objective(image.detach() - delta)) / 2e-6
        assert gradient[0, col] == pytest.approx(float(numerical), abs=2e-7, rel=2e-7)
    assert concept.grad is queries.grad is negatives.grad is None

    # With Gaussian query noise active under torch.no_grad(), outer stays detached and valid
    with torch.no_grad():
        sem_ng, ret_ng, audit_ng = paper_minimal_retrieval_semantic_losses(
            image, concept, queries, negatives, static_selection=static_sel, config=cfg, noise_seed=seed,
        )
    assert not sem_ng.requires_grad and not ret_ng.requires_grad
    assert audit_ng['query_perturbation']['enabled'] is True
    assert audit_ng['query_perturbation']['max_l2_perturbation'] > 0


@pytest.mark.parametrize('rejected_profile', ['paper_minimal', 'enhanced_v3'])
def test_removed_mechanism_profiles_are_rejected_by_objective(rejected_profile):
    image, concept, queries, negatives, static_sel, cfg = _inputs(gradients=False)
    seed = paper_minimal_noise_seed(42, stage='step', step_index=1, model_index=0)
    with pytest.raises(ValueError):
        paper_minimal_retrieval_semantic_losses(
            image, concept, queries, negatives, static_selection=static_sel,
            config=cfg, noise_seed=seed, mechanism_profile=rejected_profile,
        )
