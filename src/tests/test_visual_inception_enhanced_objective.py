"""Micro vector tests for temperature-scaled margin, full-bank + top-hard duel, and text-subspace query ascent."""
from dataclasses import replace
import json
import math

import pytest
import torch
import torch.nn.functional as F

from src.visual._visual_inception.config import VisualInceptionConfig
from src.visual._visual_inception.enhanced_objective import (
    BANK_LOSS_BLEND, HARD_MARGIN_LOSS_BLEND, HARD_MARGIN_SHARPNESS, MARGIN_KIND,
    QUERY_AUGMENTATION_KIND, TOP_HARD_COMPETITOR_LIMIT,
    adversarial_query_features, build_query_subspace, enhanced_retrieval_semantic_losses,
    project_spherical_queries,
)
from src.visual._visual_inception.negative_sampling import build_negative_policy


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
    base = VisualInceptionConfig(hard_negative_count=4,
        hard_negative_same_category_count=1, hard_negative_proxy_memory_count=1,
        hard_negative_popularity_count=1)
    return replace(base, **changes)


def test_temperature_scaled_full_bank_and_top_hard_margin_matches_closed_form():
    image, concept, queries, negatives, policy = inputs()
    cfg = config(query_adversarial_radius=0, hard_negative_count=6,
                 hard_negative_same_category_count=0, hard_negative_proxy_memory_count=0,
                 hard_negative_popularity_count=0)
    semantic, loss, audit = enhanced_retrieval_semantic_losses(
        image, concept, queries, negatives, negative_policy=policy, config=cfg)
    _, zero_margin_loss, _ = enhanced_retrieval_semantic_losses(
        image, concept, queries, negatives, negative_policy=policy, config=replace(cfg, contrastive_margin=0.0))

    q = F.normalize(queries, dim=-1)
    n = F.normalize(negatives, dim=-1)
    im = F.normalize(image, dim=-1)
    tau, margin = 0.07, 0.3
    pos_logit = (q @ im.T)[:, 0] / tau
    all_neg_cos = q @ n.T
    neg_logits = all_neg_cos / tau + margin
    bank_loss = torch.logsumexp(torch.cat((pos_logit[:, None], neg_logits), dim=1), dim=1) - pos_logit
    top_r = min(TOP_HARD_COMPETITOR_LIMIT, n.shape[0])
    top_cos = all_neg_cos.topk(top_r, dim=1).values
    comp_logit = torch.logsumexp(top_cos / tau, dim=1) - math.log(float(top_r))
    comp_loss = F.softplus(HARD_MARGIN_SHARPNESS * (comp_logit - pos_logit + margin)) / HARD_MARGIN_SHARPNESS
    expected = (BANK_LOSS_BLEND * bank_loss + HARD_MARGIN_LOSS_BLEND * comp_loss).mean()

    assert torch.allclose(loss, expected, atol=1e-12)
    assert loss > zero_margin_loss
    assert torch.allclose(semantic, (1 - im @ F.normalize(concept, dim=-1).T).mean())
    assert audit['margin_algorithm'] == MARGIN_KIND
    assert audit['full_bank_denominator_preserved'] is True
    json.dumps(audit, allow_nan=False)


def test_cross_modal_modality_gap_regime_preserves_adaptive_query_gradients():
    # In realistic CLIP/SigLIP cosine range (0.10..0.35), losing queries must receive
    # substantially stronger positive-similarity gradient than already-won queries.
    im_easy = torch.tensor([[0.36, 0.932952303175248]], dtype=torch.float64, requires_grad=True)
    im_hard = torch.tensor([[0.18, 0.983666610188635]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    concept = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    negatives = torch.tensor([
        [0.24, 0.9707728879609277],
        [0.22, 0.9754998718605759],
        [0.14, 0.990151503558925],
        [0.12, 0.9927738916792686],
    ], dtype=torch.float64)
    meta = [dict(id=str(i), category='shampoo' if i < 2 else 'other') for i in range(4)]
    policy = build_negative_policy(meta, negatives, 'shampoo')
    cfg = config(query_adversarial_radius=0, hard_negative_count=3)

    _, l_easy, _ = enhanced_retrieval_semantic_losses(im_easy, concept, q, negatives, negative_policy=policy, config=cfg)
    _, l_hard, _ = enhanced_retrieval_semantic_losses(im_hard, concept, q, negatives, negative_policy=policy, config=cfg)
    g_easy, = torch.autograd.grad(l_easy, im_easy)
    g_hard, = torch.autograd.grad(l_hard, im_hard)
    assert g_hard.norm().item() > 2.5 * g_easy.norm().item()


def test_outer_image_gradient_matches_finite_differences_with_frozen_queries():
    image, concept, queries, negatives, policy = inputs(gradients=True)
    cfg = config(query_adversarial_radius=0)

    def objective(value):
        semantic, retrieval, _ = enhanced_retrieval_semantic_losses(
            value, concept, queries, negatives, negative_policy=policy, config=cfg)
        return semantic + retrieval

    loss = objective(image)
    gradient, = torch.autograd.grad(loss, image)
    for column in range(image.shape[1]):
        delta = torch.zeros_like(image)
        delta[0, column] = 1e-6
        numerical = (objective(image.detach() + delta) - objective(image.detach() - delta)) / 2e-6
        assert gradient[0, column] == pytest.approx(float(numerical), abs=2e-7, rel=2e-7)
    assert concept.grad is queries.grad is negatives.grad is None


def test_text_subspace_blocks_modality_gap_escape_into_image_cone():
    # Ambient dim 6: coords 0..2 span text cone, coords 3..5 are pure image-cone modality offset!
    q_pool = F.normalize(torch.tensor([
        [1.0, 0.2, 0.0, 0.0, 0.0, 0.0],
        [1.0, -0.2, 0.1, 0.0, 0.0, 0.0],
        [1.0, 0.1, -0.2, 0.0, 0.0, 0.0],
        [1.0, -0.1, 0.25, 0.0, 0.0, 0.0],
    ], dtype=torch.float64), dim=-1)
    concept = F.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64), dim=-1)
    # Image and negatives sit heavily in image-only coordinates 3..5:
    image = F.normalize(torch.tensor([[0.25, 0.10, 0.05, 0.90, 0.20, 0.10]], dtype=torch.float64), dim=-1)
    negatives = F.normalize(torch.tensor([
        [0.25, -0.15, 0.15, 0.85, 0.35, 0.10],
        [0.24, -0.10, 0.20, 0.80, 0.40, 0.20],
        [0.20, 0.05, -0.10, 0.88, -0.20, 0.30],
        [0.18, 0.02, -0.12, 0.86, -0.30, 0.25],
    ], dtype=torch.float64), dim=-1)
    meta = [dict(id=str(i), category='shampoo' if i < 2 else 'other') for i in range(4)]
    policy = build_negative_policy(meta, negatives, 'shampoo')
    subspace = build_query_subspace(q_pool, concept_features=concept)
    cfg = config(hard_negative_count=4, query_adversarial_radius=0.20, query_adversarial_step_size=0.10)

    adv_q, audit = adversarial_query_features(
        image, q_pool[:2], negatives, negative_policy=policy, config=cfg, query_subspace=subspace)
    # Perturbed text queries MUST remain strictly zero in pure image-cone dims 3..5!
    assert torch.allclose(adv_q[:, 3:], torch.zeros_like(adv_q[:, 3:]), atol=1e-12)
    assert audit['algorithm'] == QUERY_AUGMENTATION_KIND
    assert audit['subspace_rank'] == 2
    assert audit['tangent_fallback_steps'] == 0
    assert audit['max_l2_perturbation'] > 0.05


def test_query_ascent_respects_sphere_budget_best_tracking_and_gradient_isolation():
    image, concept, queries, negatives, policy = inputs(gradients=True)
    cfg = config()
    adversarial, audit = adversarial_query_features(
        image, queries, negatives, negative_policy=policy, config=cfg, concept_features=concept)
    original = F.normalize(queries.detach(), dim=-1)
    assert not adversarial.requires_grad
    assert torch.allclose(adversarial.norm(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-12)
    assert float((adversarial - original).norm(dim=-1).max()) <= 0.20 + 1e-10
    assert audit['max_l2_perturbation'] > 0
    assert all(a >= c - 1e-12 for a, c in zip(audit['adversarial_losses'], audit['clean_losses']))
    assert image.grad is queries.grad is negatives.grad is None
    semantic, retrieval, outer = enhanced_retrieval_semantic_losses(
        image, concept, queries, negatives, negative_policy=policy, config=cfg)
    (semantic + retrieval).backward()
    assert torch.isfinite(image.grad).all() and image.grad.norm() > 0
    assert concept.grad is queries.grad is negatives.grad is None
    assert float(retrieval.detach()) == pytest.approx(
        0.5 * outer['clean_retrieval_loss'] + 0.5 * outer['adversarial_retrieval_loss'])
    json.dumps(outer, allow_nan=False)


@pytest.mark.parametrize('radius', [0., .05, .2, .5, 2.])
def test_spherical_projection_handles_zero_antipodal_and_arbitrary_steps(radius):
    original = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float64)
    value = torch.tensor([[0., 0., 0.], [0., -1., 0.], [.3, -.2, 1.5]], dtype=torch.float64)
    projected = project_spherical_queries(value, original, radius)
    assert torch.isfinite(projected).all()
    assert torch.allclose(projected.norm(dim=-1), torch.ones(3, dtype=torch.float64), atol=1e-12)
    assert bool(((projected - original).norm(dim=-1) <= radius + 1e-10).all())


@pytest.mark.parametrize('radius', [0., .20])
def test_projection_normalizes_nonunit_anchor_for_every_radius(radius):
    original = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
    projected = project_spherical_queries(torch.tensor([[0., 1., 0.]], dtype=torch.float64), original, radius)
    anchor = F.normalize(original, dim=-1)
    assert projected.norm().item() == pytest.approx(1., abs=1e-12)
    assert (projected - anchor).norm().item() <= radius + 1e-10


def test_zero_radius_disables_augmentation_cleanly():
    image, concept, queries, negatives, policy = inputs()
    cfg = config(query_adversarial_radius=0)
    adversarial, audit = adversarial_query_features(
        image, queries, negatives, negative_policy=policy, config=cfg, concept_features=concept)
    assert torch.equal(adversarial, F.normalize(queries, dim=-1))
    assert audit['enabled'] is False and audit['max_l2_perturbation'] == 0
    _, _, outer = enhanced_retrieval_semantic_losses(
        image, concept, queries, negatives, negative_policy=policy, config=cfg)
    assert outer['clean_retrieval_loss'] == pytest.approx(outer['adversarial_retrieval_loss'], abs=1e-12)


def test_final_diagnostics_under_no_grad_still_run_inner_ascent():
    image, concept, queries, negatives, policy = inputs(gradients=True)
    with torch.no_grad():
        semantic, retrieval, audit = enhanced_retrieval_semantic_losses(
            image, concept, queries, negatives, negative_policy=policy, config=config())
    assert not semantic.requires_grad and not retrieval.requires_grad
    assert audit['query_augmentation']['enabled'] is True
    assert audit['query_augmentation']['max_l2_perturbation'] > 0
    assert image.grad is concept.grad is queries.grad is negatives.grad is None


@pytest.mark.parametrize('bank_blend,comp_blend', [(1.0, 0.0), (0.8, 0.2), (0.0, 1.0)])
def test_configurable_comp_loss_blend_matches_closed_form(bank_blend, comp_blend):
    image, concept, queries, negatives, policy = inputs(gradients=True)
    cfg = config(query_adversarial_radius=0, hard_negative_count=6,
                 hard_negative_same_category_count=0, hard_negative_proxy_memory_count=0,
                 hard_negative_popularity_count=0,
                 bank_loss_blend=bank_blend, hard_margin_loss_blend=comp_blend)
    _, loss, audit = enhanced_retrieval_semantic_losses(
        image, concept, queries, negatives, negative_policy=policy, config=cfg)

    q = F.normalize(queries, dim=-1)
    n = F.normalize(negatives, dim=-1)
    im = F.normalize(image, dim=-1)
    tau, margin = 0.07, 0.3
    pos_logit = (q @ im.T)[:, 0] / tau
    all_neg_cos = q @ n.T
    neg_logits = all_neg_cos / tau + margin
    bank_loss = torch.logsumexp(torch.cat((pos_logit[:, None], neg_logits), dim=1), dim=1) - pos_logit
    top_r = min(TOP_HARD_COMPETITOR_LIMIT, n.shape[0])
    top_cos = all_neg_cos.topk(top_r, dim=1).values
    comp_logit = torch.logsumexp(top_cos / tau, dim=1) - math.log(float(top_r))
    comp_loss = F.softplus(HARD_MARGIN_SHARPNESS * (comp_logit - pos_logit + margin)) / HARD_MARGIN_SHARPNESS
    expected = (bank_blend * bank_loss + comp_blend * comp_loss).mean()

    assert torch.allclose(loss, expected, atol=1e-12)
    assert audit['bank_loss_blend'] == bank_blend
    assert audit['hard_margin_loss_blend'] == comp_blend
    loss.backward()
    assert torch.isfinite(image.grad).all() and image.grad.norm() > 0


@pytest.mark.parametrize('bank_blend,comp_blend', [(-0.1, 1.1), (0.5, 0.4), (0.8, 0.8), (True, False)])
def test_invalid_comp_loss_blend_rejected(bank_blend, comp_blend):
    with pytest.raises(ValueError):
        VisualInceptionConfig(bank_loss_blend=bank_blend, hard_margin_loss_blend=comp_blend).validate()


def test_custom_blend_pipeline_and_legacy_frozen_evidence_validation(tmp_path):
    from PIL import Image
    from src.paths import PROJECT
    from src.tests.test_visual_inception_retrieval import make_training_fixture, public_definition
    from src.visual._visual_inception.evidence import validate_ready
    from src.visual._visual_inception.pipeline import generate_visual_inception

    source, target = tmp_path / 'source.png', tmp_path / 'target.png'
    Image.new('RGB', (16, 16), (180, 100, 75)).save(source)
    Image.new('RGB', (16, 16), (80, 170, 150)).save(target)
    manifest, _, _ = make_training_fixture(tmp_path / 'bundle', (source, target))
    cfg = VisualInceptionConfig(
        steps=2, gradient_restarts=2, smoke_test=True, surrogate_set='clip_l14',
        training_manifest=str(manifest), bank_loss_blend=1.0, hard_margin_loss_blend=0.0,
    )
    out_img, metadata = generate_visual_inception(
        source, target, tmp_path / 'generated_pure_bank', config=cfg,
        device='cpu', definition=public_definition(),
    )
    assert metadata['enhanced_mechanisms']['parameters']['bank_loss_blend'] == 1.0
    assert metadata['enhanced_mechanisms']['parameters']['hard_margin_loss_blend'] == 0.0
    validated = validate_ready(out_img.parent, require_production=False)
    assert validated['config']['bank_loss_blend'] == 1.0

    legacy_dir = PROJECT / 'output/vi_single_clip_l14_9groups_20261118/07_shampoo/visual_inception/generation'
    if legacy_dir.is_dir():
        legacy_meta = validate_ready(legacy_dir, require_production=True)
        assert legacy_meta['method'] == 'visual_inception'


def test_visual_inception_defaults_to_pure_bank_weight_10_00():
    from src.paths import PROJECT

    default_cfg = VisualInceptionConfig().validate()
    assert BANK_LOSS_BLEND == 1.0
    assert HARD_MARGIN_LOSS_BLEND == 0.0
    assert default_cfg.bank_loss_blend == 1.0
    assert default_cfg.hard_margin_loss_blend == 0.0
    example = json.loads((PROJECT / 'src/visual/config/visual_inception.example.json').read_text(encoding='utf-8'))
    assert example['bank_loss_blend'] == 1.0
    assert example['hard_margin_loss_blend'] == 0.0
