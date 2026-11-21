"""Strictly single-level minimal reproduction of Visual Inception Eq. (2)-(3) (`paper_minimal_margin`).

1. Static Three-Source Uniform Random Negative Pool:
   Before optimization, uniformly sample fixed quotas at random (without replacement)
   from three independent public pools: (1) same-category non-target items,
   (2) cross-category proxy user memory items, and (3) top-1/3 popular items
   (using sourced public metrics if declared, otherwise plain global mean cosine
   centrality without category round-robin interleaving).

2. Strictly Single-Level Eq. (3) InfoNCE with Uniform Contrastive Margin (+0.3):
   Applies standard Eq. (3) InfoNCE over the dynamic positive plus the static
   negative pool with a uniform logit offset `+contrastive_margin` on static
   negatives (`cos(E(m), E(q')) / tau + 0.3`), without adding a fourth Hinge
   Margin loss term or dynamic Top-K hard negative mining.

3. Single-Level Gradient-Free Isotropic Gaussian Query Perturbation:
   Randomly perturbs a subset of batch query embeddings by isotropic Gaussian noise
   and re-normalizes onto the unit hypersphere:
     q' = normalize(q + eps_q),  eps_q ~ N(0, sigma^2 I)
   without any inner gradient loop or subspace projection.
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
PAPER_MINIMAL_MARGIN_RETRIEVAL_KIND = 'standard_eq3_infonce_static_three_source_negatives_logit_margin_v1'
PAPER_MINIMAL_QUERY_PERTURBATION_KIND = 'single_level_isotropic_gaussian_query_noise_v1'
_STAGE_OFFSETS = {
    'calibration': 10007,
    'step': 20011,
    'final': 30013,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector_digest(value):
    cpu = value.detach().cpu().contiguous()
    payload = (str(cpu.dtype) + ':' + str(list(cpu.shape)) + ':').encode('ascii')
    return hashlib.sha256(payload + cpu.numpy().tobytes()).hexdigest()


def paper_minimal_noise_seed(base_seed, *, stage, step_index, model_index):
    """Deterministic seed for single-level Gaussian query noise."""
    _require(stage in _STAGE_OFFSETS, '未知的查询随机扰动阶段：' + str(stage))
    _require(
        type(base_seed) is int and 0 <= base_seed < 2**63
        and type(step_index) is int and step_index >= 0
        and type(model_index) is int and model_index >= 0,
        '查询随机扰动种子参数无效',
    )
    return (
        int(base_seed)
        + _STAGE_OFFSETS[stage]
        + 1000003 * int(step_index)
        + 97 * int(model_index)
    ) % 2**63


def _parameters(config):
    values = dict(
        temperature=config.retrieval_temperature,
        noise_std=config.query_noise_std,
        noise_fraction=config.query_noise_fraction,
        logit_margin=config.contrastive_margin,
    )
    for key, value in values.items():
        _require(
            not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value),
            '最小单层复现目标参数必须为有限实数：' + key,
        )
    _require(values['temperature'] > 0, '最小单层复现检索温度必须为正数')
    _require(0 <= values['noise_std'] <= 1.0, '查询高斯噪声标准差必须在零至一之间')
    _require(0 <= values['noise_fraction'] <= 1.0, '查询随机扰动比例必须在零至一之间')
    _require(0 <= values['logit_margin'] <= 2.0, '最小单层复现对比间隔必须在零至二之间')
    return {
        'temperature': float(values['temperature']),
        'noise_std': float(values['noise_std']),
        'noise_fraction': float(values['noise_fraction']),
        'logit_margin': float(values['logit_margin']),
    }


def _features(image_features, query_features, negative_features):
    _require(
        image_features.ndim == 2
        and image_features.shape[0] == 1
        and query_features.ndim == negative_features.ndim == 2
        and query_features.shape[0] > 0
        and negative_features.shape[0] > 0
        and image_features.shape[1] == query_features.shape[1] == negative_features.shape[1],
        '最小单层复现检索的图片、查询与负例向量形状不一致',
    )
    dtype = torch.float64 if image_features.dtype == torch.float64 else torch.float32
    device = image_features.device
    for feature in (image_features, query_features, negative_features):
        _require(
            bool(torch.isfinite(feature).all()) and bool((feature.detach().norm(dim=-1) > 0).all()),
            '最小单层复现检索向量必须有限且非零',
        )
    return (
        F.normalize(image_features.to(dtype=dtype), dim=-1),
        F.normalize(query_features.detach().to(device=device, dtype=dtype), dim=-1),
        F.normalize(negative_features.detach().to(device=device, dtype=dtype), dim=-1),
    )


def build_static_negative_selection(candidate_metadata, negative_features, category, *, config):
    """Freeze static three-source random negative pool using public metadata only."""
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
        same_category=config.negative_same_category_count,
        popular=config.negative_popularity_count,
        proxy_user_memory=config.negative_proxy_memory_count,
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


def perturb_queries_gaussian(query_features, *, std, fraction, seed):
    """Gradient-free isotropic Gaussian noise on a random subset of unit queries."""
    _require(
        torch.is_tensor(query_features)
        and query_features.ndim == 2
        and query_features.shape[0] > 0
        and query_features.shape[1] > 0
        and bool(torch.isfinite(query_features).all())
        and bool((query_features.detach().norm(dim=-1) > 0).all()),
        '查询向量必须为有限非零二维张量',
    )
    _require(
        not isinstance(std, bool) and isinstance(std, (float, int)) and math.isfinite(std) and 0 <= std <= 1.0,
        '查询高斯噪声标准差无效',
    )
    _require(
        not isinstance(fraction, bool) and isinstance(fraction, (float, int))
        and math.isfinite(fraction) and 0 <= fraction <= 1.0,
        '查询随机扰动比例无效',
    )
    _require(type(seed) is int and 0 <= seed, '查询随机扰动种子无效')

    std = float(std)
    fraction = float(fraction)
    effective_seed = int(seed) % 2**63
    original = F.normalize(query_features.detach(), dim=-1)
    count, dim = int(original.shape[0]), int(original.shape[1])
    enabled = (std > 0.0 and fraction > 0.0)

    if enabled:
        perturbed_count = min(count, max(1, int(round(fraction * count))))
        rng = torch.Generator(device='cpu').manual_seed(effective_seed)
        selected_rows = sorted(torch.randperm(count, generator=rng)[:perturbed_count].tolist())
        noise = (torch.randn((perturbed_count, dim), generator=rng, dtype=torch.float64) * std).to(
            device=original.device, dtype=original.dtype,
        )
        perturbed = original.clone()
        perturbed[selected_rows] = F.normalize(original[selected_rows] + noise, dim=-1)
        perturbed = perturbed.detach()
    else:
        perturbed_count = 0
        selected_rows = []
        perturbed = original.clone().detach()

    with torch.no_grad():
        norms = perturbed.norm(dim=-1)
        tolerance = max(1e-10, 32 * torch.finfo(perturbed.dtype).eps)
        _require(
            bool(torch.isfinite(perturbed).all()) and bool((norms - 1).abs().le(tolerance).all()),
            '高斯随机扰动查询未保持有效单位球面归一化',
        )
        distances = (perturbed - original).norm(dim=-1)
        cosines = (perturbed * original).sum(dim=-1).clamp(-1.0, 1.0)

    query_audit = dict(
        algorithm=PAPER_MINIMAL_QUERY_PERTURBATION_KIND,
        enabled=enabled,
        single_level_no_inner_gradient=True,
        subspace_projection_used=False,
        queries_detached_for_outer=True,
        noise_std=std,
        noise_fraction=fraction,
        noise_seed=effective_seed,
        batch_size=count,
        perturbed_query_count=perturbed_count,
        perturbed_row_indices=selected_rows,
        numerical_tolerance=tolerance,
        max_l2_perturbation=float(distances.max()),
        min_query_cosine=float(cosines.min()),
        max_unit_norm_error=float((norms - 1).abs().max()),
        clean_query_features_sha256=_vector_digest(original),
        perturbed_query_features_sha256=_vector_digest(perturbed),
        paper_exact_query_augmentation_reproduced=False,
    )
    return perturbed.detach(), query_audit


def paper_minimal_retrieval_semantic_losses(
    image_features,
    concept_features,
    query_features,
    negative_features,
    *,
    static_selection,
    config,
    noise_seed,
    mechanism_profile='paper_minimal_margin',
):
    """Single concept + static-negative Eq. (3) InfoNCE (+0.3 margin) + gradient-free Gaussian query noise."""
    _require(
        mechanism_profile == 'paper_minimal_margin',
        '未知最小单层复现机制配置：' + str(mechanism_profile),
    )
    parameters = _parameters(config)
    _require(
        concept_features.ndim == 2
        and concept_features.shape[0] == 1
        and concept_features.shape[1] == image_features.shape[1],
        '最小单层复现语义损失须使用一个独立目标概念',
    )
    image, clean_queries, all_negatives = _features(image_features, query_features, negative_features)
    selected_indices = _validate_static_selection(static_selection, all_negatives.shape[0])
    static_negatives = all_negatives[
        torch.tensor(selected_indices, device=all_negatives.device, dtype=torch.long)
    ]

    concept = concept_features.detach().to(device=image.device, dtype=image.dtype)
    _require(
        bool(torch.isfinite(concept).all()) and bool((concept.norm(dim=-1) > 0).all()),
        '最小单层复现语义概念必须为有限非零向量',
    )
    concept = F.normalize(concept, dim=-1)
    semantic = (1.0 - image @ concept.T).mean()

    perturbed_queries, query_audit = perturb_queries_gaussian(
        clean_queries,
        std=parameters['noise_std'],
        fraction=parameters['noise_fraction'],
        seed=noise_seed,
    )

    tau = parameters['temperature']
    pos_logit = (perturbed_queries @ image.T)[:, 0] / tau
    neg_logits = (perturbed_queries @ static_negatives.T) / tau + parameters['logit_margin']
    per_query_loss = torch.logsumexp(torch.cat((pos_logit[:, None], neg_logits), dim=1), dim=1) - pos_logit
    _require(bool(torch.isfinite(per_query_loss).all()), '最小单层复现检索损失包含非有限值')
    retrieval = per_query_loss.mean()

    audit = dict(
        schema_version=1,
        mechanism_profile=mechanism_profile,
        retrieval_algorithm=PAPER_MINIMAL_MARGIN_RETRIEVAL_KIND,
        margin_loss_added=False,
        retrieval_temperature=tau,
        contrastive_margin=parameters['logit_margin'],
        logit_margin_applied_to_static_negatives=True,
        static_negative_count=len(selected_indices),
        static_negative_selection_sha256=static_selection['selection_sha256'],
        full_bank_denominator_preserved=False,
        single_level_optimization=True,
        reduction='query_mean_eq3_infonce',
        retrieval_loss=float(retrieval.detach()),
        semantic_loss=float(semantic.detach()),
        query_perturbation=query_audit,
        paper_exact_missing_mechanisms_reproduced=False,
    )
    return semantic, retrieval, audit
