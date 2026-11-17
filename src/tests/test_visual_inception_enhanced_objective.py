"""Micro vector tests for declared margin, hard negatives and query ascent."""
from dataclasses import replace
import json

import pytest
import torch
import torch.nn.functional as F

from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.enhanced_objective import (
    adversarial_query_features, enhanced_retrieval_semantic_losses, project_spherical_queries,
)
from src.visual._visual_inception.negative_sampling import build_negative_policy
from src.visual._visual_inception.objective import retrieval_loss


def inputs(*, gradients=False):
    image = torch.tensor([[.8, .6, .2]], dtype=torch.float64, requires_grad=gradients)
    concept = torch.tensor([[.9, .2, .1]], dtype=torch.float64, requires_grad=gradients)
    queries = torch.tensor([[.7, .2, .5], [.1, .9, .3]], dtype=torch.float64, requires_grad=gradients)
    negatives = torch.tensor([[1., .1, .2], [.1, 1., .3], [.2, .4, 1.],
                              [-.5, .3, .7], [.4, -.8, .2], [-.3, -.2, 1.]],
                             dtype=torch.float64, requires_grad=gradients)
    metadata = [dict(id=f'fixture_{i}', category='shampoo' if i < 3 else 'other') for i in range(6)]
    policy = build_negative_policy(metadata, negatives, 'shampoo')
    return image, concept, queries, negatives, policy


def config(**changes):
    base = VisualInceptionConfig(hard_negative_count=6,
        hard_negative_same_category_count=0, hard_negative_proxy_memory_count=0,
        hard_negative_popularity_count=0)
    return replace(base, **changes)


def test_cosine_margin_precedes_temperature_and_matches_closed_form():
    image, concept, queries, negatives, policy = inputs()
    cfg = config(query_adversarial_radius=0)
    semantic, loss, audit = enhanced_retrieval_semantic_losses(image, concept, queries, negatives,
        negative_policy=policy, config=cfg)
    q, n, im = F.normalize(queries, dim=-1), F.normalize(negatives, dim=-1), F.normalize(image, dim=-1)
    positive = ((q @ im.T)[:, 0] - .3) / .07
    logits = torch.cat((positive[:, None], q @ n.T / .07), dim=1)
    expected = (torch.logsumexp(logits, dim=1) - positive).mean()
    assert torch.allclose(loss, expected, atol=1e-12)
    assert loss > retrieval_loss(image, queries, negatives)
    assert torch.allclose(semantic, (1-im @ F.normalize(concept, dim=-1).T).mean())
    assert audit['margin_algorithm'] == 'positive_cosine_additive_margin_before_temperature_v1'
    json.dumps(audit, allow_nan=False)


def test_outer_image_gradient_matches_finite_differences_with_frozen_queries():
    image, concept, queries, negatives, policy = inputs(gradients=True)
    cfg = config(query_adversarial_radius=0)

    def objective(value):
        semantic, retrieval, _ = enhanced_retrieval_semantic_losses(value, concept, queries, negatives,
            negative_policy=policy, config=cfg)
        return semantic + retrieval

    loss = objective(image)
    gradient, = torch.autograd.grad(loss, image)
    for column in range(image.shape[1]):
        delta = torch.zeros_like(image)
        delta[0, column] = 1e-6
        numerical = (objective(image.detach()+delta)-objective(image.detach()-delta)) / 2e-6
        assert gradient[0, column] == pytest.approx(float(numerical), abs=2e-7, rel=2e-7)
    assert concept.grad is queries.grad is negatives.grad is None


def test_query_ascent_respects_sphere_budget_best_tracking_and_gradient_isolation():
    image, concept, queries, negatives, policy = inputs(gradients=True)
    cfg = config(hard_negative_count=4, hard_negative_same_category_count=1,
        hard_negative_proxy_memory_count=1, hard_negative_popularity_count=1)
    adversarial, audit = adversarial_query_features(image, queries, negatives, negative_policy=policy, config=cfg)
    original = F.normalize(queries.detach(), dim=-1)
    assert not adversarial.requires_grad
    assert torch.allclose(adversarial.norm(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-12)
    assert float((adversarial-original).norm(dim=-1).max()) <= .05 + 1e-10
    assert audit['max_l2_perturbation'] > 0
    assert all(a >= c-1e-12 for a,c in zip(audit['adversarial_losses'], audit['clean_losses']))
    assert image.grad is queries.grad is negatives.grad is None
    semantic, retrieval, outer = enhanced_retrieval_semantic_losses(image, concept, queries, negatives,
        negative_policy=policy, config=cfg)
    (semantic+retrieval).backward()
    assert torch.isfinite(image.grad).all() and image.grad.norm() > 0
    assert concept.grad is queries.grad is negatives.grad is None
    assert float(retrieval.detach()) == pytest.approx(.5*outer['clean_retrieval_loss']+.5*outer['adversarial_retrieval_loss'])
    json.dumps(outer, allow_nan=False)


@pytest.mark.parametrize('radius', [0., .05, .5, 2.])
def test_spherical_projection_handles_zero_antipodal_and_arbitrary_steps(radius):
    original = torch.tensor([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]], dtype=torch.float64)
    value = torch.tensor([[0.,0.,0.], [0.,-1.,0.], [.3,-.2,1.5]], dtype=torch.float64)
    projected = project_spherical_queries(value, original, radius)
    assert torch.isfinite(projected).all()
    assert torch.allclose(projected.norm(dim=-1), torch.ones(3, dtype=torch.float64), atol=1e-12)
    assert bool(((projected-original).norm(dim=-1) <= radius+1e-10).all())


@pytest.mark.parametrize('radius', [0., .05])
def test_projection_normalizes_nonunit_anchor_for_every_radius(radius):
    original = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
    projected = project_spherical_queries(torch.tensor([[0., 1., 0.]], dtype=torch.float64), original, radius)
    anchor = F.normalize(original, dim=-1)
    assert projected.norm().item() == pytest.approx(1., abs=1e-12)
    assert (projected-anchor).norm().item() <= radius+1e-10


def test_zero_radius_fallback_and_zero_margin_match_legacy_formula():
    image, concept, queries, negatives, policy = inputs()
    cfg = config(query_adversarial_radius=0, contrastive_margin=0)
    adversarial, audit = adversarial_query_features(image, queries, negatives, negative_policy=policy, config=cfg)
    assert torch.equal(adversarial, F.normalize(queries, dim=-1))
    assert audit['enabled'] is False and audit['max_l2_perturbation'] == 0
    _, actual, _ = enhanced_retrieval_semantic_losses(image, concept, queries, negatives,
        negative_policy=policy, config=cfg)
    assert torch.allclose(actual, retrieval_loss(image, queries, negatives), atol=1e-12)


def test_final_diagnostics_under_no_grad_still_run_inner_ascent():
    image, concept, queries, negatives, policy = inputs(gradients=True)
    with torch.no_grad():
        semantic, retrieval, audit = enhanced_retrieval_semantic_losses(image, concept, queries, negatives,
            negative_policy=policy, config=config())
    assert not semantic.requires_grad and not retrieval.requires_grad
    assert audit['query_augmentation']['enabled'] is True
    assert audit['query_augmentation']['max_l2_perturbation'] > 0
    assert image.grad is concept.grad is queries.grad is negatives.grad is None
