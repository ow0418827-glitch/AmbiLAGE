"""Principled mathematical realization of the paper's underspecified mechanisms.

1. Full-Bank + Top-Hard-Competitor Margin Retrieval Loss:
   Cross-modal CLIP/SigLIP cosines suffer from the modality gap (typical cosine
   0.24..0.37 in CLIP and 0.07..0.12 in SigLIP). Subtracting 0.3 directly from
   raw cosine before dividing by tau=0.07 penalizes positive logits by 4.29 and
   collapses p_+ -> 0 on all queries, making hard negatives invisible to outer
   gradients. Instead, the 0.3 margin operates in temperature-scaled logit space
   (equivalent cosine lead 0.3 * tau = 0.021, matching 3-5 rank spacings across
   both CLIP and SigLIP):
   - Full-bank InfoNCE retains all N candidates in the denominator while adding
     +margin to the K role-mined hard negatives;
   - A sharpened smooth margin duel against the top-5 hard competitors focuses
     perturbation budget on losing/borderline queries and relaxes once won.

2. Text-Manifold Subspace Spherical PGD Query Augmentation:
   Unconstrained gradient ascent in R^D pushes text queries across the modality
   gap into the image cone toward negative images (while -v cancels on T_v S^{D-1}).
   Projecting inner gradients onto the orthonormal semantic-variation subspace
   spanned by frozen reference text embeddings (with the mean text-cone axis
   removed) keeps perturbed queries strictly on the encoder's text manifold.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F


QUERY_AUGMENTATION_KIND = 'text_subspace_spherical_l2_pgd_best_per_query_v2'
MARGIN_KIND = 'temperature_scaled_full_bank_and_top_hard_softplus_margin_v2'
TOP_HARD_COMPETITOR_LIMIT = 5
HARD_MARGIN_SHARPNESS = 2.0
BANK_LOSS_BLEND = 0.3
HARD_MARGIN_LOSS_BLEND = 0.7


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
    _require(0 <= values['margin'] <= 2, '对比间隔必须在零至二之间')
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


def _vector_digest(value):
    cpu = value.detach().cpu().contiguous()
    payload = (str(cpu.dtype) + ':' + str(list(cpu.shape)) + ':').encode('ascii')
    return hashlib.sha256(payload + cpu.numpy().tobytes()).hexdigest()


def build_query_subspace(reference_queries, concept_features=None):
    """Extract an orthonormal basis of semantic text variations orthogonal to the cone mean."""
    _require(torch.is_tensor(reference_queries) and reference_queries.ndim == 2
             and reference_queries.shape[0] > 0 and reference_queries.shape[1] > 0
             and bool(torch.isfinite(reference_queries).all())
             and bool((reference_queries.detach().norm(dim=-1) > 0).all()),
             '查询语义子空间参考向量必须为有限非零二维张量')
    rows = [F.normalize(reference_queries.detach().to(device='cpu', dtype=torch.float64), dim=-1)]
    if concept_features is not None:
        _require(torch.is_tensor(concept_features) and concept_features.ndim == 2
                 and concept_features.shape[0] > 0 and concept_features.shape[1] == reference_queries.shape[1]
                 and bool(torch.isfinite(concept_features).all())
                 and bool((concept_features.detach().norm(dim=-1) > 0).all()),
                 '目标概念参考向量无效或维度不一致')
        rows.append(F.normalize(concept_features.detach().to(device='cpu', dtype=torch.float64), dim=-1))
    matrix = torch.cat(rows, dim=0)
    mean_vec = matrix.mean(dim=0, keepdim=True)
    mean_norm = float(mean_vec.norm())
    if mean_norm > 1e-12:
        mean_dir = mean_vec / mean_norm
        centered = matrix - (matrix @ mean_dir.T) * mean_dir
    else:
        centered = matrix
    if matrix.shape[0] >= 2 and float(centered.norm()) > 1e-12:
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        cutoff = max(1e-8, float(singular_values[0]) * 1e-6)
        basis = vh[singular_values > cutoff].contiguous().detach()
    else:
        basis = torch.zeros((0, matrix.shape[1]), dtype=torch.float64)
    return dict(basis=basis, rank=int(basis.shape[0]), ambient_dim=int(matrix.shape[1]),
                reference_count=int(matrix.shape[0]), mean_cone_removed=True,
                basis_sha256=_vector_digest(basis))


def _validate_query_subspace(subspace, dim):
    _require(isinstance(subspace, dict) and torch.is_tensor(subspace.get('basis'))
             and subspace['basis'].ndim == 2 and subspace['basis'].shape[1] == dim
             and subspace.get('rank') == int(subspace['basis'].shape[0])
             and subspace.get('ambient_dim') == dim
             and type(subspace.get('reference_count')) is int and subspace['reference_count'] >= 1
             and subspace.get('mean_cone_removed') is True
             and subspace.get('basis_sha256') == _vector_digest(subspace['basis']),
             '查询文本流形子空间对象无效或维度不符')
    return subspace


def _selected_loss(image, queries, negatives, negative_policy, config, parameters, *, include_per_query=True):
    from .negative_sampling import select_hard_negatives
    query = F.normalize(queries, dim=-1)
    indices, audit = select_hard_negatives(query.detach(), negatives.detach(), negative_policy, config,
                                           include_per_query=include_per_query)
    _require(isinstance(indices, torch.Tensor) and indices.ndim == 2
             and indices.shape[0] == query.shape[0] and indices.shape[1] > 0,
             '困难负例选择必须为每条查询返回非空编号矩阵')
    indices = indices.to(device=query.device, dtype=torch.long)
    _require(bool((indices >= 0).all()) and bool((indices < negatives.shape[0]).all())
             and all(row.unique().numel() == row.numel() for row in indices),
             '困难负例编号越界或重复')

    tau = parameters['temperature']
    margin = parameters['margin']
    pos_cos = (query @ image.T)[:, 0]
    all_neg_cos = query @ negatives.T
    pos_logit = pos_cos / tau

    # 1. Full-bank retrieval term preserving all N candidates, with +margin on mined hard negatives
    boost = torch.zeros_like(all_neg_cos)
    boost.scatter_(1, indices, margin)
    neg_logits = all_neg_cos / tau + boost
    bank_loss = torch.logsumexp(torch.cat((pos_logit[:, None], neg_logits), dim=1), dim=1) - pos_logit

    # 2. Top-5 hard-competitor smooth margin duel in temperature-scaled logit space
    selected_cos = torch.gather(all_neg_cos, 1, indices)
    top_r = min(TOP_HARD_COMPETITOR_LIMIT, selected_cos.shape[1])
    top_hard_cos = selected_cos.topk(top_r, dim=1).values
    comp_logit = torch.logsumexp(top_hard_cos / tau, dim=1) - math.log(float(top_r))
    comp_loss = F.softplus(HARD_MARGIN_SHARPNESS * (comp_logit - pos_logit + margin)) / HARD_MARGIN_SHARPNESS

    loss = BANK_LOSS_BLEND * bank_loss + HARD_MARGIN_LOSS_BLEND * comp_loss
    _require(bool(torch.isfinite(loss).all()), '增强检索损失包含非有限值')
    return loss, audit


def project_spherical_queries(value, original, radius):
    """Nearest unit-sphere cap projection, up to floating-point tolerance."""
    _require(0 <= radius <= 2 and math.isfinite(radius), '查询投影半径无效')
    _require(value.ndim == 2 and value.shape == original.shape
             and bool(torch.isfinite(value).all()) and bool(torch.isfinite(original).all())
             and bool((original.norm(dim=-1) > 0).all()), '查询投影输入形状或数值无效')
    if radius == 0:
        return F.normalize(original.detach(), dim=-1)
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
        fallback = F.normalize(basis - (basis * anchor).sum(dim=-1, keepdim=True) * anchor, dim=-1)
        tangent_unit = torch.where(tangent_norm > 1e-14, tangent_unit, fallback)
    boundary = minimum_cosine * anchor + math.sqrt(max(0.0, 1 - minimum_cosine ** 2)) * tangent_unit
    if anchor.shape[1] == 1:
        boundary = anchor
    projected = torch.where(cosine >= minimum_cosine, direction, boundary)
    return F.normalize(projected, dim=-1).to(dtype=original.dtype, device=original.device).detach()


def adversarial_query_features(image_features, query_features, negative_features, *,
                               negative_policy, config, query_subspace=None,
                               concept_features=None, include_per_query_audit=True):
    """Return detached worst-case queries on the text-manifold spherical cap."""
    parameters = _parameters(config)
    image, original, negatives = _features(image_features.detach(), query_features, negative_features)
    image, original, negatives = image.detach(), original.detach(), negatives.detach()
    subspace = (_validate_query_subspace(query_subspace, original.shape[1]) if query_subspace is not None
                else build_query_subspace(query_features, concept_features=concept_features))
    basis = subspace['basis'].to(device=original.device, dtype=original.dtype)

    with torch.no_grad():
        clean_loss, clean_selection = _selected_loss(
            image, original, negatives, negative_policy, config, parameters,
            include_per_query=include_per_query_audit)
    best, best_loss = original.clone(), clean_loss.clone()
    current = original.clone()
    enabled = (parameters['radius'] > 0 and parameters['steps'] > 0
               and parameters['step_size'] > 0 and parameters['mix'] > 0)
    fallback_rows = 0
    if enabled:
        for _ in range(parameters['steps']):
            with torch.enable_grad():
                leaf = current.detach().requires_grad_(True)
                losses, _ = _selected_loss(
                    image, leaf, negatives, negative_policy, config, parameters, include_per_query=False)
                gradient, = torch.autograd.grad(losses.sum(), leaf)
            _require(bool(torch.isfinite(gradient).all()), '对抗查询梯度包含非有限值')
            raw_grad = gradient.detach()
            raw_tan = raw_grad - (raw_grad * current).sum(dim=-1, keepdim=True) * current
            if basis.shape[0] > 0:
                sub_grad = (raw_grad @ basis.T) @ basis
                sub_tan = sub_grad - (sub_grad * current).sum(dim=-1, keepdim=True) * current
                sub_norm = sub_tan.norm(dim=-1, keepdim=True)
                fallback_rows += int((sub_norm <= 1e-14).sum().item())
                chosen_tan = torch.where(sub_norm > 1e-14, sub_tan, raw_tan)
            else:
                fallback_rows += int(current.shape[0])
                chosen_tan = raw_tan
            direction = chosen_tan / chosen_tan.norm(dim=-1, keepdim=True).clamp_min(1e-20)
            current = project_spherical_queries(current + parameters['step_size'] * direction,
                                                original, parameters['radius'])
            with torch.no_grad():
                candidate_loss, _ = _selected_loss(
                    image, current, negatives, negative_policy, config, parameters, include_per_query=False)
                improved = candidate_loss > best_loss
                best = torch.where(improved[:, None], current, best).detach()
                best_loss = torch.maximum(best_loss, candidate_loss).detach()
    with torch.no_grad():
        final_loss, final_selection = _selected_loss(
            image, best, negatives, negative_policy, config, parameters,
            include_per_query=include_per_query_audit)
        worse = final_loss < clean_loss
        if bool(worse.any()):
            best = torch.where(worse[:, None], original, best).detach()
            final_loss, final_selection = _selected_loss(
                image, best, negatives, negative_policy, config, parameters,
                include_per_query=include_per_query_audit)
        distances = (best - original).norm(dim=-1)
        norms = best.norm(dim=-1)
        tolerance = max(1e-10, 32 * torch.finfo(best.dtype).eps)
        _require(bool((distances <= parameters['radius'] + tolerance).all())
                 and bool((norms - 1).abs().le(tolerance).all()), '对抗查询投影超过球面或扰动预算')
        _require(bool((final_loss >= clean_loss - tolerance).all()), '对抗查询逐条最坏损失降低')
    audit = dict(algorithm=QUERY_AUGMENTATION_KIND, enabled=enabled,
        space='frozen_text_query_subspace_on_unit_sphere',
        optimization='per_query_text_subspace_projected_spherical_gradient_ascent',
        projection='unit_sphere_cap_intersection_with_original_l2_ball',
        initialization='clean_query', image_features_detached_in_inner=True,
        negatives_detached=True, queries_detached_for_outer=True,
        subspace_rank=subspace['rank'], subspace_ambient_dim=subspace['ambient_dim'],
        subspace_reference_count=subspace['reference_count'],
        subspace_mean_cone_removed=True, subspace_basis_sha256=subspace['basis_sha256'],
        tangent_fallback_steps=fallback_rows,
        radius=parameters['radius'], steps=parameters['steps'], step_size=parameters['step_size'],
        mix=parameters['mix'], numerical_tolerance=tolerance,
        max_l2_perturbation=float(distances.max()), max_unit_norm_error=float((norms - 1).abs().max()),
        clean_losses=clean_loss.tolist(), adversarial_losses=final_loss.tolist(),
        minimum_loss_gain=float((final_loss - clean_loss).min()),
        clean_query_features_sha256=_vector_digest(original),
        adversarial_query_features_sha256=_vector_digest(best),
        clean_negative_selection_sha256=clean_selection['selection_sha256'],
        adversarial_negative_selection_sha256=final_selection['selection_sha256'],
        paper_exact_query_augmentation_reproduced=False)
    return best.detach(), audit


def enhanced_retrieval_semantic_losses(image_features, concept_features, query_features,
                                       negative_features, *, negative_policy, config,
                                       query_subspace=None, include_per_query_audit=True):
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
    adversarial, query_audit = adversarial_query_features(
        image_features, query_features, negative_features,
        negative_policy=negative_policy, config=config, query_subspace=query_subspace,
        concept_features=concept_features, include_per_query_audit=include_per_query_audit)
    clean_loss, clean_selection = _selected_loss(
        image, clean_queries, negatives, negative_policy, config, parameters,
        include_per_query=include_per_query_audit)
    adversarial_loss, adversarial_selection = _selected_loss(
        image, adversarial, negatives, negative_policy, config, parameters,
        include_per_query=include_per_query_audit)
    retrieval = ((1 - parameters['mix']) * clean_loss + parameters['mix'] * adversarial_loss).mean()
    audit = dict(schema_version=2, mechanism_profile='enhanced_v3',
        margin_algorithm=MARGIN_KIND, contrastive_margin=parameters['margin'],
        retrieval_temperature=parameters['temperature'],
        top_hard_competitor_limit=TOP_HARD_COMPETITOR_LIMIT,
        hard_margin_sharpness=HARD_MARGIN_SHARPNESS,
        bank_loss_blend=BANK_LOSS_BLEND,
        hard_margin_loss_blend=HARD_MARGIN_LOSS_BLEND,
        full_bank_denominator_preserved=True,
        reduction='query_mean_of_clean_adversarial_weighted_losses',
        clean_retrieval_loss=float(clean_loss.detach().mean()),
        adversarial_retrieval_loss=float(adversarial_loss.detach().mean()),
        retrieval_loss=float(retrieval.detach()), semantic_loss=float(semantic.detach()),
        clean_negative_selection=clean_selection,
        adversarial_negative_selection=adversarial_selection,
        query_augmentation=query_audit, paper_exact_missing_mechanisms_reproduced=False)
    return semantic, retrieval, audit
