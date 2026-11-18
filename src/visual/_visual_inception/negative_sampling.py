"""Declared public substitutes for three negative roles and stable hard mining.

The paper names three negative sources—(1) same-category non-target products,
(2) user's existing memory items, and (3) popular catalog items—without giving
an explicit sampling formula. This module defines a strict public convention:
- formal user memory and formal queries are never accessed;
- proxy_user_memory uses cross-category independent public items (falling back
  to all public items only in single-category micro test fixtures) so it is not
  a duplicate of same_category or global_fill;
- popularity uses sourced public metrics when declared, otherwise an equal
  blend of within-category prototypicality and global cosine centrality so no
  single category background monopolizes the popular pool.
"""
from __future__ import annotations

import math
from urllib.parse import urlsplit

import torch
import torch.nn.functional as F

from .targets import content_sha256


POLICY_KIND = 'independent_public_three_role_negatives_v2'
SELECTION_KIND = 'stable_query_cosine_role_quotas_then_global_fill_v2'
ROLE_ORDER = ('same_category', 'popular', 'proxy_user_memory')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _category(value):
    _require(isinstance(value, str) and value.strip(), '公开负例缺少有效品类')
    return ' '.join(value.split()).casefold()


def _public_popularity(row):
    """Accept explicitly sourced public scores, never rank/reserve positions."""
    raw = row.get('public_popularity')
    if isinstance(raw, dict):
        value, source = raw.get('value'), raw.get('source')
        metric = raw.get('metric', 'public_popularity_score')
        higher = raw.get('higher_is_more_popular', True)
    elif 'popularity_score' in row:
        value = row.get('popularity_score')
        source = row.get('popularity_source') or row.get('popularity_provenance')
        metric = row.get('popularity_metric', 'public_popularity_score')
        higher = row.get('popularity_higher_is_more_popular', True)
    else:
        return None
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or type(higher) is not bool
            or not isinstance(metric, str) or not metric.strip()):
        return None
    url = source if isinstance(source, str) else source.get('url') if isinstance(source, dict) else None
    if not isinstance(url, str):
        return None
    parsed = urlsplit(url)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        return None
    return dict(value=float(value), metric=metric.strip(), source=source,
                higher_is_more_popular=higher)


def _validated_features(features, count=None):
    _require(torch.is_tensor(features) and features.ndim == 2 and features.shape[0] > 0
             and features.shape[1] > 0 and torch.isfinite(features).all().item(),
             '负例向量必须为有限的非空二维张量')
    _require(count is None or features.shape[0] == count, '公开候选元数据与向量数量不同')
    features = features.detach().to(device='cpu', dtype=torch.float64)
    _require(bool((torch.linalg.vector_norm(features, dim=-1) > 0).all()), '公开候选向量不能为零')
    return F.normalize(features, dim=-1)


def _category_balanced_centrality(vectors, categories):
    count = vectors.shape[0]
    if count == 1:
        return [0.0], [0]
    sim = vectors @ vectors.T
    global_means = ((sim.sum(dim=1) - 1.0) / (count - 1)).tolist()
    members_by_category = {}
    category_order = []
    for index, cat in enumerate(categories):
        if cat not in members_by_category:
            members_by_category[cat] = []
            category_order.append(cat)
        members_by_category[cat].append(index)
    scores = []
    for index, cat in enumerate(categories):
        peers = [peer for peer in members_by_category[cat] if peer != index]
        within = float(sim[index, peers].mean()) if peers else global_means[index]
        scores.append(round(0.5 * within + 0.5 * global_means[index], 12))
    # Order categories by their strongest prototype, then interleave round-robin so
    # no large background cluster can monopolize the top-1/3 popularity proxy pool.
    ranked_by_cat = {
        cat: sorted(indices, key=lambda i: (-scores[i], i))
        for cat, indices in members_by_category.items()
    }
    ordered_cats = sorted(category_order, key=lambda c: (-scores[ranked_by_cat[c][0]], ranked_by_cat[c][0]))
    interleaved = []
    depth = max(len(v) for v in ranked_by_cat.values())
    for level in range(depth):
        for cat in ordered_cats:
            if level < len(ranked_by_cat[cat]):
                interleaved.append(ranked_by_cat[cat][level])
    return scores, interleaved


def build_negative_policy(candidate_metadata, negative_features, category):
    """Freeze query-independent role membership using public metadata only."""
    _require(isinstance(candidate_metadata, list) and candidate_metadata
             and all(isinstance(r, dict) for r in candidate_metadata), '公开候选元数据无效')
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
        popularity = dict(method='declared_sourced_public_metric', metric=metric,
            higher_is_more_popular=higher, true_public_popularity_used=True,
            sales_popularity_claimed=False, score_sources=public,
            scored_candidate_count=len(available), scores=scores)
    else:
        scores, ordered = _category_balanced_centrality(vectors, categories)
        popularity = dict(method='query_independent_category_balanced_cosine_centrality_proxy',
            metric='equal_blend_within_category_and_global_mean_cosine_round_robin',
            higher_is_more_popular=True, true_public_popularity_used=False,
            sales_popularity_claimed=False, score_sources=None,
            scored_candidate_count=count, scores=scores)
    popular = ordered[:pool_size]
    popularity.update(pool_fraction='top_one_third_rounded_up_of_all_candidates',
                      requested_pool_size=pool_size, actual_pool_size=len(popular))
    same_category = [i for i, value in enumerate(categories) if value == target_category]
    cross_category = [i for i, value in enumerate(categories) if value != target_category]
    proxy_memory = cross_category if cross_category else list(range(count))
    proxy_source = ('cross_category_independent_public_candidates_as_proxy_not_formal_user_memory'
        if cross_category else 'all_independent_public_candidates_single_category_fallback_not_formal_user_memory')
    policy = dict(schema_version=2, kind=POLICY_KIND, category=target_category,
        candidate_count=count, candidate_ids=ids, candidate_categories=categories,
        candidate_metadata_sha256=content_sha256(candidate_metadata),
        role_indices=dict(same_category=same_category, proxy_user_memory=proxy_memory, popular=popular),
        role_sources=dict(same_category='frozen_same_category_public_non_target_products',
            proxy_user_memory=proxy_source, popular=popularity['method']),
        popularity=popularity, formal_user_memory_read=False, formal_queries_read=False,
        roles_may_overlap=True, cross_category_proxy_disjoint_from_same_category=bool(cross_category),
        paper_exact_negative_sampling_reproduced=False)
    policy['policy_sha256'] = content_sha256(policy)
    return policy


def _validate_policy(policy, candidate_count):
    _require(isinstance(policy, dict) and policy.get('kind') == POLICY_KIND
             and policy.get('schema_version') == 2 and policy.get('candidate_count') == candidate_count,
             '负例角色策略与当前固定候选不一致')
    _require(policy.get('policy_sha256') == content_sha256({k: v for k, v in policy.items() if k != 'policy_sha256'}),
             '负例角色策略摘要不符')
    roles = policy.get('role_indices')
    _require(isinstance(roles, dict) and set(roles) == set(ROLE_ORDER), '负例角色不完整')
    for indices in roles.values():
        _require(isinstance(indices, list) and len(indices) == len(set(indices))
                 and all(type(i) is int and 0 <= i < candidate_count for i in indices), '负例角色索引无效')
    return roles


def _selection_digest(selected_rows, role_counts, role_shortfalls, fill_counts):
    return content_sha256(dict(selected=selected_rows, role_counts=role_counts,
        role_shortfalls=role_shortfalls, global_fill_counts=fill_counts))


def select_hard_negatives(queries, negatives, policy, config, *, include_per_query=True):
    """Select distinct negatives per query; selection has no autograd graph."""
    _require(torch.is_tensor(negatives) and negatives.ndim == 2 and negatives.shape[0] > 0,
             '负例向量必须为非空二维张量')
    _require(torch.is_tensor(queries) and queries.ndim == 2 and queries.shape[0] > 0
             and queries.shape[1] == negatives.shape[1] and torch.isfinite(queries).all().item(),
             '查询向量无效或与负例维度不同')
    _require(bool((torch.linalg.vector_norm(queries.detach(), dim=-1) > 0).all()), '查询向量不能为零')
    vectors = _validated_features(negatives)
    count = vectors.shape[0]
    roles = _validate_policy(policy, count)
    requested = getattr(config, 'hard_negative_count', 32)
    quotas = dict(same_category=getattr(config, 'hard_negative_same_category_count', 10),
        proxy_user_memory=getattr(config, 'hard_negative_proxy_memory_count', 10),
        popular=getattr(config, 'hard_negative_popularity_count', 10))
    _require(type(requested) is int and requested > 0 and all(type(v) is int and v >= 0 for v in quotas.values())
             and sum(quotas.values()) <= requested, '困难负例数量或三来源配额无效')
    effective = min(requested, count)
    query_vectors = F.normalize(queries.detach().to(device='cpu', dtype=torch.float64), dim=-1)
    similarities = query_vectors @ vectors.T
    result, query_audits = [], []
    role_counts_rows, role_shortfall_rows, fill_counts = [], [], []
    role_sets = {name: set(indices) for name, indices in roles.items()}
    for scores in similarities.tolist():
        order = sorted(range(count), key=lambda i: (-scores[i], i))
        selected, seen, selected_by_role = [], set(), {}
        for role in ROLE_ORDER:
            eligible = [i for i in order if i in role_sets[role] and i not in seen]
            take = eligible[:min(quotas[role], effective - len(selected))]
            selected_by_role[role] = take
            selected.extend(take)
            seen.update(take)
        fill = [i for i in order if i not in seen][:effective - len(selected)]
        selected.extend(fill)
        result.append(selected)
        role_counts = {r: len(selected_by_role[r]) for r in ROLE_ORDER}
        role_shortfall = {r: max(0, quotas[r] - len(selected_by_role[r])) for r in ROLE_ORDER}
        role_counts_rows.append(role_counts)
        role_shortfall_rows.append(role_shortfall)
        fill_counts.append(len(fill))
        if include_per_query:
            query_audits.append(dict(selected_indices=selected, role_selected_indices=selected_by_role,
                role_selected_counts=role_counts, role_shortfall=role_shortfall,
                global_fill_indices=fill, global_fill_count=len(fill)))
    indices = torch.tensor(result, device=queries.device, dtype=torch.long)
    audit = dict(kind=SELECTION_KIND, policy_sha256=policy['policy_sha256'],
        candidate_count=count, query_count=queries.shape[0], requested_count=requested,
        effective_count=effective, role_order=list(ROLE_ORDER), requested_role_quotas=quotas,
        available_role_counts={r: len(v) for r, v in roles.items()},
        candidate_shortfall=max(0, requested - count),
        selection_sha256=_selection_digest(result, role_counts_rows, role_shortfall_rows, fill_counts),
        per_query_included=bool(include_per_query),
        total_role_selected_counts={r: sum(row[r] for row in role_counts_rows) for r in ROLE_ORDER},
        total_role_shortfall={r: sum(row[r] for row in role_shortfall_rows) for r in ROLE_ORDER},
        total_global_fill_count=sum(fill_counts),
        stable_tie_break='original_candidate_index', selection_uses_detached_query=True)
    if include_per_query:
        audit['per_query'] = query_audits
    return indices, audit


def validate_negative_selection_audit(audit, policy, config, query_count, *, require_per_query=True):
    """Check frozen role/quota/index evidence without querying any model."""
    count = policy.get('candidate_count')
    roles = _validate_policy(policy, count)
    requested = config.hard_negative_count
    quotas = dict(same_category=config.hard_negative_same_category_count,
        proxy_user_memory=config.hard_negative_proxy_memory_count,
        popular=config.hard_negative_popularity_count)
    effective = min(requested, count)
    _require(isinstance(audit, dict) and audit.get('kind') == SELECTION_KIND
             and audit.get('policy_sha256') == policy['policy_sha256']
             and audit.get('candidate_count') == count and audit.get('query_count') == query_count
             and audit.get('requested_count') == requested and audit.get('effective_count') == effective
             and audit.get('role_order') == list(ROLE_ORDER) and audit.get('requested_role_quotas') == quotas
             and audit.get('available_role_counts') == {r: len(v) for r, v in roles.items()}
             and audit.get('candidate_shortfall') == max(0, requested - count)
             and audit.get('stable_tie_break') == 'original_candidate_index'
             and audit.get('selection_uses_detached_query') is True,
             '困难负例选择审计的策略或配额不符')
    digest = audit.get('selection_sha256')
    _require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
             '困难负例选择摘要无效')
    per_query_included = audit.get('per_query_included', True)
    _require(type(per_query_included) is bool, '困难负例逐查询标记无效')
    if require_per_query:
        _require(per_query_included is True, '缺少逐查询困难负例审计明细')
    totals = audit.get('total_role_selected_counts')
    shortfalls = audit.get('total_role_shortfall')
    fill_total = audit.get('total_global_fill_count')
    _require(isinstance(totals, dict) and set(totals) == set(ROLE_ORDER)
             and all(type(v) is int and 0 <= v <= query_count * quotas[r] for r, v in totals.items())
             and isinstance(shortfalls, dict) and set(shortfalls) == set(ROLE_ORDER)
             and all(type(v) is int and v == query_count * quotas[r] - totals[r] for r, v in shortfalls.items())
             and type(fill_total) is int and fill_total >= 0
             and sum(totals.values()) + fill_total == query_count * effective,
             '困难负例汇总计数或补位总数不符')
    if not per_query_included:
        _require('per_query' not in audit, '紧凑模式困难负例审计不应夹带逐查询列表')
        return True
    rows = audit.get('per_query')
    _require(isinstance(rows, list) and len(rows) == query_count, '困难负例逐查询审计数量不符')
    selected_rows, role_counts_rows, role_shortfall_rows, fill_counts = [], [], [], []
    for row in rows:
        _require(isinstance(row, dict), '困难负例逐查询记录无效')
        selected, assigned = row.get('selected_indices'), row.get('role_selected_indices')
        _require(isinstance(selected, list) and len(selected) == effective and len(set(selected)) == effective
                 and all(type(i) is int and 0 <= i < count for i in selected)
                 and isinstance(assigned, dict) and set(assigned) == set(ROLE_ORDER),
                 '困难负例索引重复、越界或来源缺失')
        combined, seen = [], set()
        for role in ROLE_ORDER:
            indices = assigned[role]
            available = set(roles[role]) - seen
            size = min(quotas[role], len(available), effective - len(combined))
            _require(isinstance(indices, list) and len(indices) == size and len(set(indices)) == size
                     and all(type(i) is int for i in indices)
                     and set(indices) <= available, '困难负例来源配额或归属不符')
            combined.extend(indices)
            seen.update(indices)
        fill = row.get('global_fill_indices')
        role_counts = {r: len(v) for r in ROLE_ORDER for v in (assigned[r],)}
        role_shortfall = {r: max(0, quotas[r] - len(assigned[r])) for r in ROLE_ORDER}
        _require(isinstance(fill, list) and len(fill) == effective - len(combined)
                 and len(set(fill)) == len(fill) and not set(fill) & seen
                 and all(type(i) is int and 0 <= i < count for i in fill)
                 and combined + fill == selected and row.get('global_fill_count') == len(fill)
                 and row.get('role_selected_counts') == role_counts
                 and row.get('role_shortfall') == role_shortfall,
                 '困难负例补位、顺序或短缺审计不符')
        selected_rows.append(selected)
        role_counts_rows.append(role_counts)
        role_shortfall_rows.append(role_shortfall)
        fill_counts.append(len(fill))
    _require(digest == _selection_digest(selected_rows, role_counts_rows, role_shortfall_rows, fill_counts)
             and totals == {r: sum(row[r] for row in role_counts_rows) for r in ROLE_ORDER}
             and shortfalls == {r: sum(row[r] for row in role_shortfall_rows) for r in ROLE_ORDER}
             and fill_total == sum(fill_counts), '困难负例逐查询明细与汇总摘要不一致')
    return True
