"""Micro vector tests for paper_standard: static 3-source random negatives, cosine hinge margin, and 1-step FGSM."""
from dataclasses import replace
import json
import math

import pytest
import torch
import torch.nn.functional as F

from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.paper_standard_objective import (
    PAPER_STANDARD_MARGIN_KIND,
    PAPER_STANDARD_QUERY_AUGMENTATION_KIND,
    STATIC_POLICY_KIND,
    STATIC_ROLE_ORDER,
    _paper_standard_per_query_loss,
    build_static_negative_selection,
    paper_standard_adversarial_queries,
    paper_standard_retrieval_semantic_losses,
)


def _config(**changes):
    base = VisualInceptionConfig.from_dict({
        'mechanism_profile': 'paper_standard',
        'hard_negative_same_category_count': 2,
        'hard_negative_proxy_memory_count': 2,
        'hard_negative_popularity_count': 2,
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
    # 6 jeans tightly clustered around [1, 0, 0] vs 3 shampoos around [0, 1, 0]:
    # Plain global mean cosine centrality (without category round-robin) ranks jeans as top-3.
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
        hard_negative_same_category_count=10,
        hard_negative_popularity_count=10,
        hard_negative_proxy_memory_count=10,
    )
    small_sel = build_static_negative_selection(small_meta, torch.eye(2), 'shampoo', config=small_cfg)
    assert small_sel['selected_count'] == 2
    assert len(set(small_sel['selected_indices'])) == 2
    assert any(small_sel['role_shortfall'].values())


def test_paper_standard_loss_closed_form_and_outer_gradient_finite_differences():
    image, concept, queries, negatives, static_sel, cfg = _inputs(gradients=True)
    cfg_no_adv = replace(cfg, query_adversarial_mix=0.0)
    semantic, loss, audit = paper_standard_retrieval_semantic_losses(
        image, concept, queries, negatives, static_selection=static_sel, config=cfg_no_adv,
    )

    q = F.normalize(queries, dim=-1)
    im = F.normalize(image, dim=-1)
    static_n = F.normalize(negatives[static_sel['selected_indices']], dim=-1)
    tau, margin = 0.07, 0.3
    pos_cos = (q @ im.T)[:, 0]
    neg_cos = q @ static_n.T
    pos_logit = pos_cos / tau
    neg_logits = neg_cos / tau
    infonce = torch.logsumexp(torch.cat((pos_logit[:, None], neg_logits), dim=1), dim=1) - pos_logit
    hinge = F.relu(neg_cos - pos_cos[:, None] + margin).mean(dim=1)
    expected = (infonce + hinge).mean()

    assert torch.allclose(loss, expected, atol=1e-12)
    assert torch.allclose(semantic, (1.0 - im @ F.normalize(concept, dim=-1).T).mean(), atol=1e-12)
    assert audit['margin_algorithm'] == PAPER_STANDARD_MARGIN_KIND
    assert audit['full_bank_denominator_preserved'] is False
    assert audit['query_augmentation']['enabled'] is False
    json.dumps(audit, allow_nan=False)

    def objective(val):
        sem, ret, _ = paper_standard_retrieval_semantic_losses(
            val, concept, queries, negatives, static_selection=static_sel, config=cfg_no_adv,
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


def test_single_step_fgsm_query_perturbation_matches_exact_formula_and_no_subspace():
    # Ambient dim 6: queries lie in coords 0..2 while image/negatives have weight in coords 3..5.
    # Without SVD subspace projection, single-step FGSM updates ambient coords 3..5 as well.
    queries = F.normalize(torch.tensor([
        [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        [1.0, -0.2, 0.1, 0.0, 0.0, 0.0],
    ], dtype=torch.float64), dim=-1).detach().requires_grad_(True)
    image = F.normalize(torch.tensor([[0.25, 0.10, 0.05, 0.90, 0.20, 0.10]], dtype=torch.float64), dim=-1)
    image_leaf = image.detach().requires_grad_(True)
    concept = F.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64), dim=-1)
    negatives = F.normalize(torch.tensor([
        [0.25, -0.15, 0.15, 0.85, 0.35, 0.10],
        [0.24, -0.10, 0.20, 0.80, 0.40, 0.20],
        [0.20, 0.05, -0.10, 0.88, -0.20, 0.30],
        [0.18, 0.02, -0.12, 0.86, -0.30, 0.25],
    ], dtype=torch.float64), dim=-1)
    meta = [dict(id=str(i), category='shampoo' if i < 2 else 'other') for i in range(4)]
    cfg = _config(
        hard_negative_same_category_count=1,
        hard_negative_proxy_memory_count=1,
        hard_negative_popularity_count=1,
        query_adversarial_step_size=0.05,
        query_adversarial_mix=0.25,
    )
    static_sel = build_static_negative_selection(meta, negatives, 'shampoo', config=cfg)
    static_n = negatives[static_sel['selected_indices']]

    adv_q, q_audit = paper_standard_adversarial_queries(image_leaf, queries, static_n, config=cfg)

    # Exact closed-form single-step embedding L2 FGSM:
    q_leaf = queries.detach().requires_grad_(True)
    losses, _, _ = _paper_standard_per_query_loss(
        image_leaf.detach(), q_leaf, static_n, temperature=0.07, margin=0.3,
    )
    grad, = torch.autograd.grad(losses.sum(), q_leaf)
    unit_grad = grad / grad.norm(dim=-1, keepdim=True)
    expected_adv = F.normalize(q_leaf.detach() + 0.05 * unit_grad, dim=-1)

    assert torch.allclose(adv_q, expected_adv, atol=1e-12)
    assert adv_q[:, 3:].abs().max().item() > 1e-3
    assert q_audit['algorithm'] == PAPER_STANDARD_QUERY_AUGMENTATION_KIND
    assert q_audit['subspace_projection_used'] is False
    assert q_audit['greedy_best_loss_used'] is False
    assert torch.allclose(adv_q.norm(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-12)
    max_chord = math.sqrt(2.0 * (1.0 - math.sqrt(1.0 - 0.05 ** 2)))
    assert q_audit['max_l2_perturbation'] <= max_chord + 1e-10
    assert image_leaf.grad is None and queries.grad is None

    # Test execution under torch.no_grad() and outer gradient flow
    with torch.no_grad():
        sem_ng, ret_ng, audit_ng = paper_standard_retrieval_semantic_losses(
            image_leaf, concept, queries, negatives, static_selection=static_sel, config=cfg,
        )
    assert not sem_ng.requires_grad and not ret_ng.requires_grad
    assert audit_ng['query_augmentation']['enabled'] is True
    assert audit_ng['query_augmentation']['max_l2_perturbation'] > 0

    sem, ret, outer = paper_standard_retrieval_semantic_losses(
        image_leaf, concept, queries, negatives, static_selection=static_sel, config=cfg,
    )
    (sem + ret).backward()
    assert torch.isfinite(image_leaf.grad).all() and image_leaf.grad.norm() > 0
    assert float(ret.detach()) == pytest.approx(
        0.75 * outer['clean_retrieval_loss'] + 0.25 * outer['adversarial_retrieval_loss'],
    )
    json.dumps(outer, allow_nan=False)
