"""Declared public substitutes for three negative roles and stable hard mining.

The paper names the roles but not their sampling algorithm. This module is an
explicit engineering convention: no formal user memory is accessed, and visual
centrality is never represented as sales or observed popularity.
"""
from __future__ import annotations

import math
from urllib.parse import urlsplit

import torch
import torch.nn.functional as F

from .targets import content_sha256


POLICY_KIND = 'independent_public_three_role_negatives_v1'
SELECTION_KIND = 'stable_query_cosine_role_quotas_then_global_fill_v1'
ROLE_ORDER = ('same_category', 'popular', 'proxy_user_memory')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _category(value):
    _require(isinstance(value, str) and value.strip(), '公开负例缺少有效品类')
    return ' '.join(value.split()).casefold()


def _public_popularity(row):
    """Accept explicitly sourced public scores, never rank/reserve positions.

    Preferred schema: public_popularity={value,metric,source,higher_is_more_popular}.
    A flat popularity_score + popularity_source pair is also supported. Source
    may be an HTTPS URL or an object containing url/revision/field evidence.
    These remain imported declarations, not independently verified sales data.
    """
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
        if count == 1:
            values = torch.zeros(1, dtype=torch.float64)
        else:
            values = ((vectors @ vectors.T).sum(dim=1) - 1.0) / (count - 1)
        scores = [round(float(value), 12) for value in values]
        ordered = sorted(range(count), key=lambda i: (-scores[i], i))
        popularity = dict(method='query_independent_cosine_centrality_proxy',
            metric='mean_cosine_to_other_frozen_public_candidates',
            higher_is_more_popular=True, true_public_popularity_used=False,
            sales_popularity_claimed=False, score_sources=None,
            scored_candidate_count=count, scores=scores)
    popular = ordered[:pool_size]
    popularity.update(pool_fraction='top_one_third_rounded_up_of_all_candidates',
                      requested_pool_size=pool_size, actual_pool_size=len(popular))
    policy = dict(schema_version=1, kind=POLICY_KIND, category=target_category,
        candidate_count=count, candidate_ids=ids, candidate_categories=categories,
        candidate_metadata_sha256=content_sha256(candidate_metadata),
        role_indices=dict(same_category=[i for i, value in enumerate(categories) if value == target_category],
            proxy_user_memory=list(range(count)), popular=popular),
        role_sources=dict(same_category='frozen_same_category_public_non_target_products',
            proxy_user_memory='all_independent_public_candidates_as_proxy_not_formal_user_memory',
            popular=popularity['method']), popularity=popularity,
        formal_user_memory_read=False, formal_queries_read=False,
        roles_may_overlap=True, paper_exact_negative_sampling_reproduced=False)
    policy['policy_sha256'] = content_sha256(policy)
    return policy


def _validate_policy(policy, candidate_count):
    _require(isinstance(policy, dict) and policy.get('kind') == POLICY_KIND
             and policy.get('schema_version') == 1 and policy.get('candidate_count') == candidate_count,
             '负例角色策略与当前固定候选不一致')
    _require(policy.get('policy_sha256') == content_sha256({k:v for k,v in policy.items() if k != 'policy_sha256'}),
             '负例角色策略摘要不符')
    roles = policy.get('role_indices')
    _require(isinstance(roles, dict) and set(roles) == set(ROLE_ORDER), '负例角色不完整')
    for indices in roles.values():
        _require(isinstance(indices, list) and len(indices) == len(set(indices))
                 and all(type(i) is int and 0 <= i < candidate_count for i in indices), '负例角色索引无效')
    return roles


def select_hard_negatives(queries, negatives, policy, config):
    """Select distinct negatives per query; selection has no autograd graph.

    Within each frozen role pool choose current query-cosine hardest candidates.
    Preserve scarce same-category/popularity quotas before the all-public proxy
    role, then fill from all candidates. Ties retain original candidate order.
    """
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
    quotas = dict(same_category=getattr(config, 'hard_negative_same_category_count', 8),
        proxy_user_memory=getattr(config, 'hard_negative_proxy_memory_count', 8),
        popular=getattr(config, 'hard_negative_popularity_count', 8))
    _require(type(requested) is int and requested > 0 and all(type(v) is int and v >= 0 for v in quotas.values())
             and sum(quotas.values()) <= requested, '困难负例数量或三来源配额无效')
    effective = min(requested, count)
    query_vectors = F.normalize(queries.detach().to(device='cpu', dtype=torch.float64), dim=-1)
    similarities = query_vectors @ vectors.T
    result, query_audits = [], []
    role_sets = {name:set(indices) for name,indices in roles.items()}
    for scores in similarities.tolist():
        order = sorted(range(count), key=lambda i: (-scores[i], i))
        selected, seen, selected_by_role = [], set(), {}
        for role in ROLE_ORDER:
            eligible = [i for i in order if i in role_sets[role] and i not in seen]
            take = eligible[:min(quotas[role], effective - len(selected))]
            selected_by_role[role] = take
            selected.extend(take); seen.update(take)
        fill = [i for i in order if i not in seen][:effective-len(selected)]
        selected.extend(fill)
        result.append(selected)
        query_audits.append(dict(selected_indices=selected, role_selected_indices=selected_by_role,
            role_selected_counts={r:len(v) for r,v in selected_by_role.items()},
            role_shortfall={r:max(0,quotas[r]-len(selected_by_role[r])) for r in ROLE_ORDER},
            global_fill_indices=fill, global_fill_count=len(fill)))
    indices = torch.tensor(result, device=queries.device, dtype=torch.long)
    audit = dict(kind=SELECTION_KIND, policy_sha256=policy['policy_sha256'],
        candidate_count=count, query_count=queries.shape[0], requested_count=requested,
        effective_count=effective, role_order=list(ROLE_ORDER), requested_role_quotas=quotas,
        available_role_counts={r:len(v) for r,v in roles.items()},
        candidate_shortfall=max(0,requested-count), per_query=query_audits,
        stable_tie_break='original_candidate_index', selection_uses_detached_query=True)
    return indices, audit


def validate_negative_selection_audit(audit, policy, config, query_count):
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
             and audit.get('available_role_counts') == {r:len(v) for r,v in roles.items()}
             and audit.get('candidate_shortfall') == max(0,requested-count)
             and audit.get('stable_tie_break') == 'original_candidate_index'
             and audit.get('selection_uses_detached_query') is True,
             '困难负例选择审计的策略或配额不符')
    rows = audit.get('per_query')
    _require(isinstance(rows, list) and len(rows) == query_count, '困难负例逐查询审计数量不符')
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
            size = min(quotas[role],len(available),effective-len(combined))
            _require(isinstance(indices,list) and len(indices) == size and len(set(indices)) == size
                     and all(type(i) is int for i in indices)
                     and set(indices) <= available, '困难负例来源配额或归属不符')
            combined.extend(indices); seen.update(indices)
        fill = row.get('global_fill_indices')
        _require(isinstance(fill,list) and len(fill) == effective-len(combined)
                 and len(set(fill)) == len(fill) and not set(fill) & seen
                 and all(type(i) is int and 0 <= i < count for i in fill)
                 and combined + fill == selected and row.get('global_fill_count') == len(fill)
                 and row.get('role_selected_counts') == {r:len(v) for r,v in assigned.items()}
                 and row.get('role_shortfall') == {r:max(0,quotas[r]-len(assigned[r])) for r in ROLE_ORDER},
                 '困难负例补位、顺序或短缺审计不符')
    return True
