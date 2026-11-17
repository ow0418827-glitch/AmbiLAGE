"""Declared engineering choices for the paper's underspecified mechanisms.

The positive cosine receives an additive margin before temperature scaling.
Queries undergo per-row L2 ascent on the unit sphere; this is a feature-space
augmentation, not a claim that the authors published this exact algorithm.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F


QUERY_AUGMENTATION_KIND = 'spherical_l2_pgd_best_per_query_v1'
MARGIN_KIND = 'positive_cosine_additive_margin_before_temperature_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _parameters(config):
    values = dict(temperature=config.retrieval_temperature,
        margin=config.contrastive_margin, radius=config.query_adversarial_radius,
        steps=config.query_adversarial_steps, step_size=config.query_adversarial_step_size,
        mix=config.query_adversarial_mix)
    for key in ('temperature', 'margin', 'radius', 'step_size', 'mix'):
        value = values[key]
        _require(not isinstance(value, bool) and isinstance(value, (float, int))
                 and math.isfinite(value), '增强目标参数必须为有限实数：' + key)
    _require(values['temperature'] > 0, '增强检索温度必须为正数')
    _require(0 <= values['margin'] <= 2, '余弦对比间隔必须在零至二之间')
    _require(0 <= values['radius'] <= 2, '单位球面查询扰动半径必须在零至二之间')
    _require(0 <= values['mix'] <= 1, '对抗查询混合权重必须在零至一之间')
    _require(type(values['steps']) is int and values['steps'] >= 0,
             '对抗查询步数必须为非负整数')
    _require(values['step_size'] >= 0, '对抗查询步长不能为负数')
    return values


def _features(image_features, query_features, negative_features):
    _require(image_features.ndim == 2 and image_features.shape[0] == 1
             and query_features.ndim == negative_features.ndim == 2
             and query_features.shape[0] > 0 and negative_features.shape[0] > 0
             and image_features.shape[1] == query_features.shape[1] == negative_features.shape[1],
             '增强检索的图片、查询与负例向量形状不一致')
    dtype = torch.float64 if image_features.dtype == torch.float64 else torch.float32
    device = image_features.device
    for feature in (image_features, query_features, negative_features):
        _require(bool(torch.isfinite(feature).all())
                 and bool((feature.detach().norm(dim=-1) > 0).all()),
                 '增强检索向量必须有限且非零')
    return (F.normalize(image_features.to(dtype=dtype), dim=-1),
            F.normalize(query_features.detach().to(device=device, dtype=dtype), dim=-1),
            F.normalize(negative_features.detach().to(device=device, dtype=dtype), dim=-1))


def _selected_loss(image, queries, negatives, negative_policy, config, parameters):
    from .negative_sampling import select_hard_negatives
    query = F.normalize(queries, dim=-1)
    indices, audit = select_hard_negatives(query.detach(), negatives.detach(), negative_policy, config)
    _require(isinstance(indices, torch.Tensor) and indices.ndim == 2
             and indices.shape[0] == query.shape[0] and indices.shape[1] > 0,
             '困难负例选择必须为每条查询返回非空编号矩阵')
    indices = indices.to(device=query.device, dtype=torch.long)
    _require(bool((indices >= 0).all()) and bool((indices < negatives.shape[0]).all())
             and all(row.unique().numel() == row.numel() for row in indices),
             '困难负例编号越界或重复')
    selected = negatives[indices]
    positive = ((query @ image.T)[:, 0] - parameters['margin']) / parameters['temperature']
    negative = torch.einsum('qd,qkd->qk', query, selected) / parameters['temperature']
    loss = torch.logsumexp(torch.cat((positive[:, None], negative), dim=1), dim=1) - positive
    _require(bool(torch.isfinite(loss).all()), '增强检索损失包含非有限值')
    return loss, audit


def project_spherical_queries(value, original, radius):
    """Nearest unit-sphere cap projection, up to floating-point tolerance.

    For unit vectors, ||q-q0||^2 = 2-2*cos(q,q0). Projection onto that cap
    therefore enforces the requested L2 budget without normalize-after-clamp
    enlarging the perturbation again.
    """
    _require(0 <= radius <= 2 and math.isfinite(radius), '查询投影半径无效')
    _require(value.ndim == 2 and value.shape == original.shape
             and bool(torch.isfinite(value).all()) and bool(torch.isfinite(original).all())
             and bool((original.norm(dim=-1) > 0).all()), '查询投影输入形状或数值无效')
    if radius == 0:
        return F.normalize(original.detach(), dim=-1)
    # Projection is tiny compared with model encoding; double precision avoids
    # cancellation at the small default cap angle even for float32 features.
    anchor = F.normalize(original.detach().to(dtype=torch.float64), dim=-1)
    work = value.to(dtype=torch.float64)
    direction = F.normalize(work, dim=-1)
    direction = torch.where(work.norm(dim=-1, keepdim=True) > 0, direction, anchor)
    cosine = (direction * anchor).sum(dim=-1, keepdim=True).clamp(-1, 1)
    minimum_cosine = 1.0 - float(radius) ** 2 / 2.0
    tangent = direction - cosine * anchor
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    tangent_unit = tangent / tangent_norm.clamp_min(torch.finfo(torch.float64).eps)
    if anchor.shape[1] > 1:
        basis = torch.zeros_like(anchor)
        basis.scatter_(1, anchor.abs().argmin(dim=-1, keepdim=True), 1)
        fallback = F.normalize(basis - (basis*anchor).sum(dim=-1, keepdim=True)*anchor, dim=-1)
        tangent_unit = torch.where(tangent_norm > 1e-14, tangent_unit, fallback)
    boundary = minimum_cosine * anchor + math.sqrt(max(0.0, 1 - minimum_cosine ** 2)) * tangent_unit
    if anchor.shape[1] == 1:
        boundary = anchor
    projected = torch.where(cosine >= minimum_cosine, direction, boundary)
    return F.normalize(projected, dim=-1).to(dtype=original.dtype, device=original.device).detach()


def _vector_digest(value):
    cpu = value.detach().cpu().contiguous()
    payload = (str(cpu.dtype) + ':' + str(list(cpu.shape)) + ':').encode('ascii')
    return hashlib.sha256(payload + cpu.numpy().tobytes()).hexdigest()


def adversarial_query_features(image_features, query_features, negative_features, *, negative_policy, config):
    """Return detached best queries; never differentiate through inner ascent."""
    parameters = _parameters(config)
    image, original, negatives = _features(image_features.detach(), query_features, negative_features)
    image, original, negatives = image.detach(), original.detach(), negatives.detach()
    with torch.no_grad():
        clean_loss, clean_selection = _selected_loss(image, original, negatives, negative_policy, config, parameters)
    best, best_loss = original.clone(), clean_loss.clone()
    current = original.clone()
    enabled = (parameters['radius'] > 0 and parameters['steps'] > 0
               and parameters['step_size'] > 0 and parameters['mix'] > 0)
    if enabled:
        for _ in range(parameters['steps']):
            # The caller may be computing final diagnostics inside no_grad.
            with torch.enable_grad():
                leaf = current.detach().requires_grad_(True)
                losses, _ = _selected_loss(image, leaf, negatives, negative_policy, config, parameters)
                gradient, = torch.autograd.grad(losses.sum(), leaf)
            _require(bool(torch.isfinite(gradient).all()), '对抗查询梯度包含非有限值')
            direction = gradient.detach() / gradient.detach().norm(dim=-1, keepdim=True).clamp_min(1e-20)
            current = project_spherical_queries(current + parameters['step_size'] * direction,
                                                original, parameters['radius'])
            with torch.no_grad():
                candidate_loss, _ = _selected_loss(image, current, negatives, negative_policy, config, parameters)
                improved = candidate_loss > best_loss
                best = torch.where(improved[:, None], current, best).detach()
                best_loss = torch.maximum(best_loss, candidate_loss).detach()
    with torch.no_grad():
        final_loss, final_selection = _selected_loss(image, best, negatives, negative_policy, config, parameters)
        # Re-evaluate after normalization and hard selection; any rounding-induced
        # decrease gets the exact clean feature back on that individual row.
        worse = final_loss < clean_loss
        if bool(worse.any()):
            best = torch.where(worse[:, None], original, best).detach()
            final_loss, final_selection = _selected_loss(image, best, negatives, negative_policy, config, parameters)
        distances = (best - original).norm(dim=-1)
        norms = best.norm(dim=-1)
        tolerance = max(1e-10, 32 * torch.finfo(best.dtype).eps)
        _require(bool((distances <= parameters['radius'] + tolerance).all())
                 and bool((norms - 1).abs().le(tolerance).all()), '对抗查询投影超过球面或扰动预算')
        _require(bool((final_loss >= clean_loss - tolerance).all()), '对抗查询逐条最坏损失降低')
    audit = dict(algorithm=QUERY_AUGMENTATION_KIND, enabled=enabled,
        space='normalized_text_embedding', optimization='per_query_l2_normalized_gradient_ascent',
        projection='unit_sphere_cap_intersection_with_original_l2_ball',
        initialization='clean_query', image_features_detached_in_inner=True,
        negatives_detached=True, queries_detached_for_outer=True,
        radius=parameters['radius'], steps=parameters['steps'], step_size=parameters['step_size'],
        mix=parameters['mix'], numerical_tolerance=tolerance,
        max_l2_perturbation=float(distances.max()), max_unit_norm_error=float((norms-1).abs().max()),
        clean_losses=clean_loss.tolist(), adversarial_losses=final_loss.tolist(),
        minimum_loss_gain=float((final_loss-clean_loss).min()),
        clean_query_features_sha256=_vector_digest(original), adversarial_query_features_sha256=_vector_digest(best),
        clean_negative_selection=clean_selection, adversarial_negative_selection=final_selection,
        paper_exact_query_augmentation_reproduced=False)
    return best.detach(), audit


def enhanced_retrieval_semantic_losses(image_features, concept_features, query_features,
                                     negative_features, *, negative_policy, config):
    """Single concept + clean/adversarial margin retrieval; image keeps graph."""
    parameters = _parameters(config)
    _require(concept_features.ndim == 2 and concept_features.shape[0] == 1
             and concept_features.shape[1] == image_features.shape[1],
             '增强语义损失须使用一个独立目标概念')
    image, clean_queries, negatives = _features(image_features, query_features, negative_features)
    concept = concept_features.detach().to(device=image.device, dtype=image.dtype)
    _require(bool(torch.isfinite(concept).all()) and bool((concept.norm(dim=-1) > 0).all()),
             '增强语义概念必须为有限非零向量')
    concept = F.normalize(concept, dim=-1)
    semantic = (1 - image @ concept.T).mean()
    adversarial, query_audit = adversarial_query_features(image_features, query_features, negative_features,
        negative_policy=negative_policy, config=config)
    clean_loss, clean_selection = _selected_loss(image, clean_queries, negatives, negative_policy, config, parameters)
    adversarial_loss, adversarial_selection = _selected_loss(image, adversarial, negatives, negative_policy, config, parameters)
    retrieval = ((1-parameters['mix']) * clean_loss + parameters['mix'] * adversarial_loss).mean()
    audit = dict(schema_version=1, mechanism_profile='enhanced_v3',
        margin_algorithm=MARGIN_KIND, contrastive_margin=parameters['margin'],
        retrieval_temperature=parameters['temperature'],
        reduction='query_mean_of_clean_adversarial_weighted_losses',
        clean_retrieval_loss=float(clean_loss.detach().mean()),
        adversarial_retrieval_loss=float(adversarial_loss.detach().mean()),
        retrieval_loss=float(retrieval.detach()), semantic_loss=float(semantic.detach()),
        clean_negative_selection=clean_selection, adversarial_negative_selection=adversarial_selection,
        query_augmentation=query_audit, paper_exact_missing_mechanisms_reproduced=False)
    return semantic, retrieval, audit
