"""Standard (non-AmbiLAGE-coupled) realizations of Visual Inception's three underspecified mechanisms.

1. Static Three-Source Uniform Random Negative Sampling:
   Before optimization, uniformly sample fixed quotas at random (without replacement)
   from three independent public pools: (1) same-category non-target items,
   (2) cross-category proxy user memory items, and (3) top-1/3 popular items
   (using sourced public metrics if declared, otherwise plain global mean cosine
   centrality without category round-robin interleaving).

2. Standard InfoNCE + Cosine-Space Hinge Margin (m=0.3):
   Applies standard InfoNCE over the dynamic positive plus the static negative subset
   and adds mean(max(0, cos(E(n), E(Q)) - cos(E(I_adv), E(Q)) + 0.3)).

3. Single-Step Embedding-Space FGSM Query Perturbation:
   Computes one L2-normalized gradient ascent step on q in ambient embedding space
   followed by unit-norm renormalization, without text-manifold SVD subspace projection
   or per-query greedy best-loss selection.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.nn.functional as F

from .negative_sampling import _category, _public_popularity, _validated_features
from .targets import content_sha256


STATIC_POLICY_KIND = 'independent_public_three_role_static_random_v1'
STATIC_ROLE_ORDER = ('same_category', 'popular', 'proxy_user_memory')
PAPER_STANDARD_MARGIN_KIND = 'standard_infonce_plus_cosine_hinge_margin_v1'
PAPER_STANDARD_QUERY_AUGMENTATION_KIND = 'single_step_embedding_l2_fgsm_v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector_digest(value):
    cpu = value.detach().cpu().contiguous()
    payload = (str(cpu.dtype) + ':' + str(list(cpu.shape)) + ':').encode('ascii')
    return hashlib.sha256(payload + cpu.numpy().tobytes()).hexdigest()


def _parameters(config):
    values = dict(
        temperature=config.retrieval_temperature,
        margin=config.contrastive_margin,
        steps=config.query_adversarial_steps,
        step_size=config.query_adversarial_step_size,
        mix=config.query_adversarial_mix,
    )
    for key in ('temperature', 'margin', 'step_size', 'mix'):
        value = values[key]
        _require(
            not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value),
            '标准复现目标参数必须为有限实数：' + key,
        )
    _require(values['temperature'] > 0, '标准复现检索温度必须为正数')
    _require(0 <= values['margin'] <= 2, '标准复现对比间隔必须在零至二之间')
    _require(0 <= values['mix'] <= 1, '标准复现对抗查询混合权重必须在零至一之间')
    _require(type(values['steps']) is int and values['steps'] == 1, '标准复现对抗查询步数必须为 1')
    _require(values['step_size'] >= 0, '标准复现对抗查询步长不能为负数')
    return {
        'temperature': float(values['temperature']),
        'margin': float(values['margin']),
        'steps': int(values['steps']),
        'step_size': float(values['step_size']),
        'mix': float(values['mix']),
    }


def _features(image_features, query_features, negative_features):
    _require(
        image_features.ndim == 2
        and image_features.shape[0] == 1
        and query_features.ndim == negative_features.ndim == 2
        and query_features.shape[0] > 0
        and negative_features.shape[0] > 0
        and image_features.shape[1] == query_features.shape[1] == negative_features.shape[1],
        '标准复现检索的图片、查询与负例向量形状不一致',
    )
    dtype = torch.float64 if image_features.dtype == torch.float64 else torch.float32
    device = image_features.device
    for feature in (image_features, query_features, negative_features):
        _require(
            bool(torch.isfinite(feature).all()) and bool((feature.detach().norm(dim=-1) > 0).all()),
            '标准复现检索向量必须有限且非零',
        )
    return (
        F.normalize(image_features.to(dtype=dtype), dim=-1),
        F.normalize(query_features.detach().to(device=device, dtype=dtype), dim=-1),
        F.normalize(negative_features.detach().to(device=device, dtype=dtype), dim=-1),
    )


def build_static_negative_selection(candidate_metadata, negative_features, category, *, config):
    """Freeze static three-source random negative subset using public metadata only."""
    _require(
        isinstance(candidate_metadata, list)
        and candidate_metadata
        and all(isinstance(r, dict) for r in candidate_metadata),
        '公开候选元数据无效',
    )
    vectors = _validated_features(negative_features, len(candidate_metadata))
    target_category, count = _category(category), len(candidate_metadata)
    categories = [_category(row.get('category')) for row in candidate_metadata]
    ids = [str(row.get('candidate_id') or row.get('id') or '') for row in candidate_metadata]
    _require(all(ids) and len(set(ids)) == count, '公开负例候选编号为空或重复')

    public = [_public_popularity(row) for row in candidate_metadata]
    available = [i for i, value in enumerate(public) if value is not None]
    pool_size = max(1, math.ceil(count / 3))
    if available:
        metrics = {(public[i]['metric'], public[i]['higher_is_more_popular']) for i in available}
        _require(len(metrics) == 1, '真实公开热度的指标或方向不一致，不能混排')
        metric, higher = next(iter(metrics))
        scores = [None if value is None else round(value['value'], 12) for value in public]
        ordered = sorted(available, key=lambda i: (-scores[i] if higher else scores[i], i))
        popularity = dict(
            method='declared_sourced_public_metric',
            metric=metric,
            higher_is_more_popular=higher,
            true_public_popularity_used=True,
            sales_popularity_claimed=False,
            score_sources=public,
            scored_candidate_count=len(available),
            scores=scores,
        )
    else:
        if count == 1:
            scores, ordered = [0.0], [0]
        else:
            sim = vectors @ vectors.T
            global_means = ((sim.sum(dim=1) - 1.0) / (count - 1)).tolist()
            scores = [round(float(v), 12) for v in global_means]
            ordered = sorted(range(count), key=lambda i: (-scores[i], i))
        popularity = dict(
            method='query_independent_global_cosine_centrality_proxy',
            metric='global_mean_cosine_without_category_round_robin',
            higher_is_more_popular=True,
            true_public_popularity_used=False,
            sales_popularity_claimed=False,
            score_sources=None,
            scored_candidate_count=count,
            scores=scores,
        )
    popular = ordered[:pool_size]
    popularity.update(
        pool_fraction='top_one_third_rounded_up_of_all_candidates',
        requested_pool_size=pool_size,
        actual_pool_size=len(popular),
    )

    same_category = [i for i, value in enumerate(categories) if value == target_category]
    cross_category = [i for i, value in enumerate(categories) if value != target_category]
    proxy_memory = cross_category if cross_category else list(range(count))
    proxy_source = (
        'cross_category_independent_public_candidates_as_proxy_not_formal_user_memory'
        if cross_category
        else 'all_independent_public_candidates_single_category_fallback_not_formal_user_memory'
    )
    role_indices = dict(same_category=same_category, proxy_user_memory=proxy_memory, popular=popular)

    quotas = dict(
        same_category=config.hard_negative_same_category_count,
        popular=config.hard_negative_popularity_count,
        proxy_user_memory=config.hard_negative_proxy_memory_count,
    )
    _require(
        all(type(v) is int and v >= 0 for v in quotas.values()) and sum(quotas.values()) >= 1,
        '静态三来源负例配额必须为非负整数且至少为一',
    )
    seed = config.seed
    _require(type(seed) is int and 0 <= seed < 2**63, '静态负例采样种子无效')
    rng = torch.Generator(device='cpu').manual_seed((seed + 7919) % 2**63)

    selected, seen = [], set()
    role_selected_indices, role_shortfall = {}, {}
    for role in STATIC_ROLE_ORDER:
        eligible = [i for i in role_indices[role] if i not in seen]
        take_count = min(quotas[role], len(eligible))
        if take_count > 0:
            perm = torch.randperm(len(eligible), generator=rng)[:take_count].tolist()
            take = [eligible[p] for p in perm]
        else:
            take = []
        role_selected_indices[role] = take
        role_shortfall[role] = max(0, quotas[role] - len(take))
        selected.extend(take)
        seen.update(take)
    _require(len(selected) >= 1 and len(selected) == len(set(selected)), '静态负例采样结果为空或含重复编号')

    record = dict(
        schema_version=1,
        kind=STATIC_POLICY_KIND,
        category=target_category,
        seed=seed,
        candidate_count=count,
        candidate_ids=ids,
        candidate_categories=categories,
        candidate_metadata_sha256=content_sha256(candidate_metadata),
        role_indices=role_indices,
        role_sources=dict(
            same_category='frozen_same_category_public_non_target_products',
            proxy_user_memory=proxy_source,
            popular=popularity['method'],
        ),
        popularity=popularity,
        role_order=list(STATIC_ROLE_ORDER),
        requested_role_quotas=quotas,
        selected_indices=selected,
        selected_candidate_ids=[ids[i] for i in selected],
        selected_count=len(selected),
        role_selected_indices=role_selected_indices,
        role_selected_counts={r: len(role_selected_indices[r]) for r in STATIC_ROLE_ORDER},
        role_shortfall=role_shortfall,
        formal_user_memory_read=False,
        formal_queries_read=False,
        roles_may_overlap=True,
        cross_category_proxy_disjoint_from_same_category=bool(cross_category),
        paper_exact_negative_sampling_reproduced=False,
    )
    record['selection_sha256'] = content_sha256(record)
    return record


def _validate_static_selection(static_selection, candidate_count):
    _require(
        isinstance(static_selection, dict)
        and static_selection.get('kind') == STATIC_POLICY_KIND
        and static_selection.get('schema_version') == 1
        and static_selection.get('candidate_count') == candidate_count,
        '静态三来源负例选择记录与当前固定候选不一致',
    )
    body = {k: v for k, v in static_selection.items() if k != 'selection_sha256'}
    _require(static_selection.get('selection_sha256') == content_sha256(body), '静态三来源负例选择摘要不符')
    selected = static_selection.get('selected_indices')
    _require(
        isinstance(selected, list)
        and len(selected) >= 1
        and len(selected) == len(set(selected))
        and static_selection.get('selected_count') == len(selected)
        and all(type(i) is int and 0 <= i < candidate_count for i in selected),
        '静态三来源负例索引无效或重复',
    )
    return selected


def _paper_standard_per_query_loss(image, queries, static_negatives, *, temperature, margin):
    pos_cos = (queries @ image.T)[:, 0]
    neg_cos = queries @ static_negatives.T
    pos_logit = pos_cos / temperature
    neg_logits = neg_cos / temperature
    infonce = torch.logsumexp(torch.cat((pos_logit[:, None], neg_logits), dim=1), dim=1) - pos_logit
    hinge_margin = F.relu(neg_cos - pos_cos[:, None] + margin).mean(dim=1)
    total = infonce + hinge_margin
    _require(bool(torch.isfinite(total).all()), '标准复现检索损失包含非有限值')
    return total, infonce, hinge_margin


def paper_standard_adversarial_queries(image_features, query_features, static_negative_features, *, config):
    """Return detached queries perturbed by a single ambient embedding FGSM step."""
    parameters = _parameters(config)
    image, original, negatives = _features(image_features.detach(), query_features, static_negative_features)
    image, original, negatives = image.detach(), original.detach(), negatives.detach()
    tau, margin = parameters['temperature'], parameters['margin']
    steps, step_size, mix = parameters['steps'], parameters['step_size'], parameters['mix']

    with torch.no_grad():
        clean_loss, _, _ = _paper_standard_per_query_loss(
            image, original, negatives, temperature=tau, margin=margin,
        )

    enabled = (steps > 0 and step_size > 0 and mix > 0)
    if enabled:
        with torch.enable_grad():
            leaf = original.detach().requires_grad_(True)
            losses, _, _ = _paper_standard_per_query_loss(
                image, leaf, negatives, temperature=tau, margin=margin,
            )
            gradient, = torch.autograd.grad(losses.sum(), leaf)
        _require(bool(torch.isfinite(gradient).all()), '标准复现对抗查询梯度包含非有限值')
        raw_grad = gradient.detach()
        grad_norm = raw_grad.norm(dim=-1, keepdim=True)
        unit_grad = torch.where(
            grad_norm > 1e-12,
            raw_grad / grad_norm.clamp_min(1e-12),
            torch.zeros_like(raw_grad),
        )
        adversarial = F.normalize(original + step_size * unit_grad, dim=-1).detach()
    else:
        adversarial = original.clone()

    with torch.no_grad():
        adv_loss, _, _ = _paper_standard_per_query_loss(
            image, adversarial, negatives, temperature=tau, margin=margin,
        )
        distances = (adversarial - original).norm(dim=-1)
        norms = adversarial.norm(dim=-1)
        tolerance = max(1e-10, 32 * torch.finfo(adversarial.dtype).eps)
        max_theoretical_chord = (
            (math.sqrt(2.0 * (1.0 - math.sqrt(max(0.0, 1.0 - step_size ** 2)))) if step_size < 1.0 else 2.0)
            if enabled
            else 0.0
        )
        _require(
            bool((distances <= max_theoretical_chord + tolerance).all())
            and bool((norms - 1).abs().le(tolerance).all()),
            '标准复现对抗查询超出单位球面或扰动界',
        )

    query_audit = dict(
        algorithm=PAPER_STANDARD_QUERY_AUGMENTATION_KIND,
        enabled=enabled,
        space='ambient_embedding_unit_sphere_no_subspace',
        optimization='single_step_l2_normalized_fgsm',
        subspace_projection_used=False,
        greedy_best_loss_used=False,
        image_features_detached_in_inner=True,
        negatives_detached=True,
        queries_detached_for_outer=True,
        steps=steps,
        step_size=step_size,
        mix=mix,
        numerical_tolerance=tolerance,
        max_theoretical_l2_perturbation=max_theoretical_chord,
        max_l2_perturbation=float(distances.max()),
        max_unit_norm_error=float((norms - 1).abs().max()),
        clean_losses=clean_loss.tolist(),
        adversarial_losses=adv_loss.tolist(),
        clean_query_features_sha256=_vector_digest(original),
        adversarial_query_features_sha256=_vector_digest(adversarial),
        paper_exact_query_augmentation_reproduced=False,
    )
    return adversarial.detach(), query_audit


def paper_standard_retrieval_semantic_losses(
    image_features,
    concept_features,
    query_features,
    negative_features,
    *,
    static_selection,
    config,
):
    """Single concept + static-negative InfoNCE + cosine hinge margin + 1-step FGSM query."""
    parameters = _parameters(config)
    _require(
        concept_features.ndim == 2
        and concept_features.shape[0] == 1
        and concept_features.shape[1] == image_features.shape[1],
        '标准复现语义损失须使用一个独立目标概念',
    )
    image, clean_queries, all_negatives = _features(image_features, query_features, negative_features)
    selected_indices = _validate_static_selection(static_selection, all_negatives.shape[0])
    static_negatives = all_negatives[
        torch.tensor(selected_indices, device=all_negatives.device, dtype=torch.long)
    ]

    concept = concept_features.detach().to(device=image.device, dtype=image.dtype)
    _require(
        bool(torch.isfinite(concept).all()) and bool((concept.norm(dim=-1) > 0).all()),
        '标准复现语义概念必须为有限非零向量',
    )
    concept = F.normalize(concept, dim=-1)
    semantic = (1.0 - image @ concept.T).mean()

    adversarial, query_audit = paper_standard_adversarial_queries(
        image_features, query_features, static_negatives, config=config,
    )
    tau, margin, mix = parameters['temperature'], parameters['margin'], parameters['mix']
    clean_loss, clean_infonce, clean_margin = _paper_standard_per_query_loss(
        image, clean_queries, static_negatives, temperature=tau, margin=margin,
    )
    adv_loss, adv_infonce, adv_margin = _paper_standard_per_query_loss(
        image, adversarial, static_negatives, temperature=tau, margin=margin,
    )
    retrieval = ((1.0 - mix) * clean_loss + mix * adv_loss).mean()

    audit = dict(
        schema_version=1,
        mechanism_profile='paper_standard',
        margin_algorithm=PAPER_STANDARD_MARGIN_KIND,
        contrastive_margin=margin,
        retrieval_temperature=tau,
        static_negative_count=len(selected_indices),
        static_negative_selection_sha256=static_selection['selection_sha256'],
        full_bank_denominator_preserved=False,
        reduction='query_mean_of_clean_adversarial_weighted_losses',
        clean_infonce_loss=float(clean_infonce.detach().mean()),
        clean_margin_loss=float(clean_margin.detach().mean()),
        clean_retrieval_loss=float(clean_loss.detach().mean()),
        adversarial_infonce_loss=float(adv_infonce.detach().mean()),
        adversarial_margin_loss=float(adv_margin.detach().mean()),
        adversarial_retrieval_loss=float(adv_loss.detach().mean()),
        retrieval_loss=float(retrieval.detach()),
        semantic_loss=float(semantic.detach()),
        query_augmentation=query_audit,
        paper_exact_missing_mechanisms_reproduced=False,
    )
    return semantic, retrieval, audit
