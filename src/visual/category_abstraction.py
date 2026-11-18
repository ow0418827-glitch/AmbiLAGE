"""Deterministic mathematical category-commonality abstraction for AmbiLAGE.

Constructs category-generic textual and visual representations strictly from a
frozen public training bundle (never touching victim memory or formal test
queries):
  1. Contrastive lexical salience (c-TF-IDF with df_cat >= 2 brand/SKU
     suppression) + Lazy Greedy Submodular Facility Location clause selection;
  2. Stratum-balanced Riemannian Spherical Karcher Mean on S^{D-1};
  3. Contrastive Principal Subspace (c-PCA: Sigma_+ - beta * Sigma_-) &
     spherical principal semantic anchors;
  4. James-Stein subspace shrinkage denoising of noisy product queries;
  5. In-category public visual Karcher prototype over same-category candidates.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import heapq
import math
import re
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


ABSTRACTION_SCHEMA_VERSION = 1
ABSTRACTION_KIND = 'contrastive_submodular_spherical_karcher_cpca_v1'

_STOPWORDS = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'best', 'brand', 'buy', 'by',
    'can', 'click', 'color', 'count', 'design', 'do', 'does', 'easy', 'feature',
    'features', 'featuring', 'fl', 'for', 'free', 'from', 'great', 'guarantee',
    'has', 'have', 'how', 'i', 'ideal', 'in', 'inch', 'include', 'includes',
    'including', 'into', 'is', 'it', 'item', 'its', 'like', 'look', 'looking',
    'made', 'make', 'makes', 'ml', 'model', 'more', 'need', 'new', 'of', 'on',
    'one', 'or', 'our', 'out', 'oz', 'pack', 'perfect', 'piece', 'pieces',
    'product', 'purchase', 'quality', 'satisfaction', 'set', 'shipping', 'size',
    'sku', 'style', 'that', 'the', 'their', 'these', 'this', 'those', 'to',
    'up', 'use', 'used', 'using', 'want', 'warranty', 'was', 'we', 'what',
    'when', 'which', 'who', 'will', 'with', 'would', 'you', 'your',
})

_WORD_RE = re.compile(r'[a-z0-9]+')
_CLAUSE_SPLIT_RE = re.compile(r'[.!?\n;]+')


def _norm_cat(value: str | None) -> str:
    return ' '.join(str(value or '').split()).casefold()


def _tokenize_clean(text: str) -> list[str]:
    tokens = [w for w in _WORD_RE.findall(text.casefold()) if len(w) >= 2 and w not in _STOPWORDS and not w.isdigit()]
    return tokens


def _terms_for_text(text: str) -> set[str]:
    words = _tokenize_clean(text)
    terms = set(words)
    for a, b in zip(words, words[1:]):
        if a != b:
            terms.add(f'{a} {b}')
    return terms


def _extract_candidate_clauses(
    queries: Sequence[str],
    strata: Sequence[str],
) -> list[tuple[str, str]]:
    """Extract clean, deduplicated clauses from queries across strata."""
    seen: set[str] = set()
    clauses: list[tuple[str, str]] = []
    for text, stratum in zip(queries, strata):
        cleaned = ' '.join(str(text or '').split())
        if not cleaned:
            continue
        parts = _CLAUSE_SPLIT_RE.split(cleaned) if stratum == 'category_description' else [cleaned]
        for raw in parts:
            clause = ' '.join(raw.strip().strip('"\'.,;:-–—').split())
            if not clause:
                continue
            words = clause.split()
            if not (2 <= len(words) <= 28):
                continue
            norm = clause.casefold()
            if norm not in seen:
                seen.add(norm)
                clauses.append((clause, stratum))
    if not clauses:
        for text, stratum in zip(queries, strata):
            c = ' '.join(str(text or '').split())
            if c and c.casefold() not in seen:
                seen.add(c.casefold())
                clauses.append((c, stratum))
    return clauses


def compute_contrastive_salience(
    queries: Sequence[str],
    out_of_category_titles: Sequence[str],
) -> tuple[dict[str, float], list[str]]:
    """Contrastive TF-IDF salience suppressing single-product brand/SKU tokens."""
    df_cat: Counter[str] = Counter()
    for text in queries:
        df_cat.update(_terms_for_text(text))

    df_out: Counter[str] = Counter()
    for title in out_of_category_titles:
        df_out.update(_terms_for_text(title))

    n_out = max(1, len(out_of_category_titles))
    max_df = max(df_cat.values(), default=0)
    scores: dict[str, float] = {}
    for term, count in df_cat.items():
        if count <= 0:
            continue
        # Brand/SKU words almost always appear in only a single item description.
        repeat_weight = 1.0 if (count >= 2 or max_df < 2) else 0.05
        bigram_bonus = 1.35 if ' ' in term else 1.0
        idf_out = math.log((1.0 + n_out) / (1.0 + df_out.get(term, 0)))
        val = math.log1p(count) * max(0.05, idf_out) * repeat_weight * bigram_bonus
        scores[term] = float(val)

    ranked_terms = [
        term for term, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return scores, ranked_terms


def _build_canonical_commonality_prompts(
    category: str,
    target_concept_text: str,
    ranked_terms: Sequence[str],
) -> list[str]:
    cat_clean = ' '.join(str(category or '').split()) or 'product'
    cat_words = set(_tokenize_clean(cat_clean))
    chosen: list[str] = []
    used_tokens: set[str] = set()
    for term in ranked_terms:
        t_words = set(term.split())
        if t_words <= cat_words:
            continue
        if len(t_words & used_tokens) >= len(t_words):
            continue
        chosen.append(term)
        used_tokens.update(t_words)
        if len(chosen) >= 8:
            break

    prompts = []
    seen: set[str] = set()
    candidates = [str(target_concept_text).strip()]
    if chosen:
        candidates.append(f"A {cat_clean} with {', '.join(chosen[:5])}.")
        candidates.append(f"I want to buy {cat_clean} featuring {', '.join(chosen[:6])}.")
    else:
        candidates.append(f"A {cat_clean} designed for everyday use.")
        candidates.append(f"I want to buy {cat_clean} with reliable performance.")
    for item in candidates:
        norm = ' '.join(item.split())
        if norm and norm.casefold() not in seen:
            seen.add(norm.casefold())
            prompts.append(norm)
    return prompts


def _clause_salience_score(clause: str, salience: dict[str, float]) -> float:
    terms = _terms_for_text(clause)
    if not terms:
        return 0.0
    total = sum(salience.get(t, 0.0) for t in terms)
    # Mild proper-noun penalty for heavily capitalized marketing copy.
    words = clause.split()
    cap_ratio = sum(1 for w in words[1:] if len(w) > 1 and w[:1].isupper() and w[1:].islower()) / max(1, len(words) - 1)
    return float(total / math.sqrt(max(1.0, len(terms)))) * (1.0 - 0.25 * cap_ratio)


def lazy_greedy_facility_location(
    query_vecs: torch.Tensor,
    query_weights: torch.Tensor,
    cand_vecs: torch.Tensor,
    k: int,
) -> tuple[list[int], float]:
    """Select <= k candidate indices maximizing weighted facility location on S^{D-1}."""
    n_q = int(query_vecs.shape[0])
    n_c = int(cand_vecs.shape[0])
    k = min(max(0, int(k)), n_c)
    if k == 0 or n_q == 0 or n_c == 0:
        return [], 0.0

    q64 = F.normalize(query_vecs.to(device='cpu', dtype=torch.float64), dim=-1)
    c64 = F.normalize(cand_vecs.to(device='cpu', dtype=torch.float64), dim=-1)
    w64 = query_weights.to(device='cpu', dtype=torch.float64).reshape(-1)
    w64 = w64 / w64.sum().clamp_min(1e-12)

    sim = q64 @ c64.T  # [N_q, N_c]
    current_best = torch.full((n_q,), -2.0, dtype=torch.float64)

    # First element exact scan
    first_gains = (w64[:, None] * torch.maximum(current_best[:, None], sim)).sum(dim=0)
    best_first = int(torch.argmax(first_gains).item())
    selected = [best_first]
    selected_set = {best_first}
    current_best = torch.maximum(current_best, sim[:, best_first])
    total_val = float((w64 * current_best).sum().item())

    if k == 1 or len(selected) >= n_c:
        return selected, total_val

    # Lazy greedy heap (negative gain, tie-break index) for submodular steps 2..k
    heap: list[tuple[float, int]] = []
    for idx in range(n_c):
        if idx in selected_set:
            continue
        gain = float((w64 * (torch.maximum(current_best, sim[:, idx]) - current_best)).sum().item())
        heapq.heappush(heap, (-gain, idx))

    while len(selected) < k and heap:
        _, cand_idx = heapq.heappop(heap)
        if cand_idx in selected_set:
            continue
        exact_gain = float((w64 * (torch.maximum(current_best, sim[:, cand_idx]) - current_best)).sum().item())
        if not heap or exact_gain >= -heap[0][0] - 1e-12:
            selected.append(cand_idx)
            selected_set.add(cand_idx)
            current_best = torch.maximum(current_best, sim[:, cand_idx])
            total_val += max(0.0, exact_gain)
        else:
            heapq.heappush(heap, (-exact_gain, cand_idx))

    return selected, float((w64 * current_best).sum().item())


def spherical_karcher_mean(
    vectors: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    steps: int = 25,
    tol: float = 1e-10,
) -> torch.Tensor:
    """Intrinsic Riemannian Fréchet/Karcher mean on the unit hypersphere S^{D-1}."""
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError('Karcher 均值输入必须为非空二维张量')
    V = F.normalize(vectors.detach().to(device='cpu', dtype=torch.float64), dim=-1)
    if V.shape[0] == 1:
        return V.clone()
    if weights is None:
        w = torch.full((V.shape[0],), 1.0 / V.shape[0], dtype=torch.float64)
    else:
        w = weights.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
        w = w.clamp_min(0.0)
        w = w / w.sum().clamp_min(1e-15)

    ext_sum = (w[:, None] * V).sum(dim=0, keepdim=True)
    if float(ext_sum.norm().item()) > 1e-8:
        mu = F.normalize(ext_sum, dim=-1)
    else:
        best_idx = int(torch.argmax(w).item())
        mu = V[best_idx : best_idx + 1].clone()

    def _obj(point: torch.Tensor) -> float:
        dots = (V @ point.T)[:, 0].clamp(-1.0 + 1e-12, 1.0 - 1e-12)
        angles = torch.arccos(dots)
        return float((w * angles.square()).sum().item())

    cur_obj = _obj(mu)
    for _ in range(int(steps)):
        dots = (V @ mu.T)[:, 0].clamp(-1.0 + 1e-12, 1.0 - 1e-12)
        thetas = torch.arccos(dots)
        sins = torch.sqrt((1.0 - dots * dots).clamp_min(1e-24))
        ratios = torch.where(thetas > 1e-6, thetas / sins, torch.ones_like(thetas))
        logs = ratios[:, None] * (V - dots[:, None] * mu)
        delta = (w[:, None] * logs).sum(dim=0, keepdim=True)
        # Re-project to exact tangent space to eliminate numerical drift.
        delta = delta - (delta @ mu.T) * mu
        d_norm = float(delta.norm().item())
        if d_norm <= tol:
            break
        unit_dir = delta / d_norm
        step_size = min(1.0, math.pi * 0.5 / max(d_norm, 1e-12))
        accepted = False
        for _bt in range(4):
            ang = d_norm * step_size
            cand = F.normalize(math.cos(ang) * mu + math.sin(ang) * unit_dir, dim=-1)
            cand_obj = _obj(cand)
            if cand_obj <= cur_obj + 1e-13:
                mu = cand
                cur_obj = cand_obj
                accepted = True
                break
            step_size *= 0.5
        if not accepted or d_norm * step_size <= tol:
            break
    return F.normalize(mu, dim=-1)


def build_contrastive_category_subspace(
    pos_vecs: torch.Tensor,
    pos_weights: torch.Tensor,
    mu_cat: torch.Tensor,
    *,
    neg_vecs: torch.Tensor | None = None,
    subspace_rank: int = 6,
    contrastive_beta: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute orthonormal Contrastive PCA basis U_K in T_{mu_cat} S^{D-1} and 2K anchors."""
    q64 = F.normalize(pos_vecs.detach().to(device='cpu', dtype=torch.float64), dim=-1)
    w64 = pos_weights.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
    w64 = w64 / w64.sum().clamp_min(1e-15)
    mu64 = F.normalize(mu_cat.detach().to(device='cpu', dtype=torch.float64), dim=-1).reshape(1, -1)
    dim = int(q64.shape[1])
    rank_cap = max(0, min(int(subspace_rank), dim - 1, int(q64.shape[0]) - 1))
    if rank_cap == 0 or dim <= 1:
        return (
            torch.zeros((0, dim), dtype=torch.float64),
            torch.zeros((0,), dtype=torch.float64),
            torch.zeros((0, dim), dtype=torch.float64),
        )

    v_pos = q64 - (q64 @ mu64.T) * mu64
    sw_pos = v_pos * torch.sqrt(w64)[:, None]
    sigma_pos = sw_pos.T @ sw_pos
    tr_pos = float(torch.trace(sigma_pos).item())
    if tr_pos <= 1e-14:
        return (
            torch.zeros((0, dim), dtype=torch.float64),
            torch.zeros((0,), dtype=torch.float64),
            torch.zeros((0, dim), dtype=torch.float64),
        )

    op = sigma_pos
    if neg_vecs is not None and int(neg_vecs.shape[0]) > 0 and contrastive_beta > 0.0:
        n64 = F.normalize(neg_vecs.detach().to(device='cpu', dtype=torch.float64), dim=-1)
        v_neg = n64 - (n64 @ mu64.T) * mu64
        sigma_neg = (v_neg.T @ v_neg) / float(max(1, n64.shape[0]))
        tr_neg = float(torch.trace(sigma_neg).item())
        if tr_neg > 1e-14:
            sigma_neg_scaled = sigma_neg * (tr_pos / tr_neg)
            op = sigma_pos - float(contrastive_beta) * sigma_neg_scaled

    # Project symmetric operator onto mu_cat^perp.
    proj_mu = mu64.T @ mu64
    op = op - proj_mu @ op - op @ proj_mu + proj_mu @ op @ proj_mu
    op = 0.5 * (op + op.T)

    # Wait: if dim is large and number of vectors is small (e.g. D <= 1024 works instantly in <10ms on CPU),
    # guard with exact SVD in tangent span if D > 1536 for speed; otherwise direct eigh is exact.
    if dim <= 1536:
        eigvals, eigvecs = torch.linalg.eigh(op)
        order = torch.argsort(eigvals, descending=True)
        pos_mask = eigvals[order] > max(1e-10, 1e-6 * float(eigvals[order[0]].item()))
        chosen_idx = order[pos_mask][:rank_cap]
        raw_basis = eigvecs[:, chosen_idx].T  # [K, D]
    else:
        _, svals, vh = torch.linalg.svd(sw_pos, full_matrices=False)
        keep = svals > max(1e-10, 1e-6 * float(svals[0].item()))
        raw_basis = vh[keep][:rank_cap]

    if int(raw_basis.shape[0]) == 0:
        return (
            torch.zeros((0, dim), dtype=torch.float64),
            torch.zeros((0,), dtype=torch.float64),
            torch.zeros((0, dim), dtype=torch.float64),
        )

    # Re-orthogonalize cleanly against mu_cat via QR to guarantee U_K mu_cat^T = 0 and U_K U_K^T = I_K.
    ortho = raw_basis - (raw_basis @ mu64.T) * mu64
    q_mat, _ = torch.linalg.qr(ortho.T, mode='reduced')
    basis_cols: list[torch.Tensor] = []
    for col_idx in range(q_mat.shape[1]):
        col = q_mat[:, col_idx : col_idx + 1].T
        col = col - (col @ mu64.T) * mu64
        for prev in basis_cols:
            col = col - (col @ prev.T) * prev
        if float(col.norm().item()) > 1e-8:
            basis_cols.append(F.normalize(col, dim=-1))
    if not basis_cols:
        return (
            torch.zeros((0, dim), dtype=torch.float64),
            torch.zeros((0,), dtype=torch.float64),
            torch.zeros((0, dim), dtype=torch.float64),
        )
    basis = torch.cat(basis_cols, dim=0)
    proj_vars = torch.einsum('kd,df,kf->k', basis, sigma_pos, basis).clamp_min(0.0)

    anchors = []
    for k_idx in range(basis.shape[0]):
        std_k = math.sqrt(float(proj_vars[k_idx].item()))
        theta_k = min(0.32, max(0.05, std_k))
        u_k = basis[k_idx : k_idx + 1]
        anchors.append(F.normalize(math.cos(theta_k) * mu64 + math.sin(theta_k) * u_k, dim=-1))
        anchors.append(F.normalize(math.cos(theta_k) * mu64 - math.sin(theta_k) * u_k, dim=-1))
    anchor_tensor = torch.cat(anchors, dim=0)
    return basis.detach(), proj_vars.detach(), anchor_tensor.detach()


def denoise_queries_james_stein(
    queries: torch.Tensor,
    mu_cat: torch.Tensor,
    basis: torch.Tensor,
    residual_shrinkage: float,
) -> torch.Tensor:
    """James-Stein shrinkage denoising of category queries onto span(mu_cat, U_K).

    gamma = 1.0 leaves queries unchanged; gamma = 0.0 projects strictly onto the
    contrastive category commonality manifold span(mu_cat, U_K) on S^{D-1}.
    """
    gamma = min(max(float(residual_shrinkage), 0.0), 1.0)
    q64 = F.normalize(queries.detach().to(dtype=torch.float64), dim=-1)
    if gamma >= 1.0 - 1e-12:
        return q64.to(dtype=queries.dtype, device=queries.device)
    mu = F.normalize(mu_cat.detach().to(device=q64.device, dtype=torch.float64), dim=-1).reshape(1, -1)
    alpha_mu = (q64 @ mu.T) * mu
    v = q64 - alpha_mu
    if basis is not None and int(basis.shape[0]) > 0:
        b = basis.detach().to(device=q64.device, dtype=torch.float64)
        v_par = (v @ b.T) @ b
    else:
        v_par = torch.zeros_like(v)
    v_perp = v - v_par
    denoised = F.normalize(alpha_mu + v_par + gamma * v_perp, dim=-1)
    return denoised.to(dtype=queries.dtype, device=queries.device).detach()


def compute_visual_category_prototype(
    target_image_vec: torch.Tensor,
    candidate_vectors: torch.Tensor,
    same_category_indices: Sequence[int],
    *,
    visual_prototype_mix: float = 0.65,
) -> torch.Tensor:
    """Riemannian Karcher visual prototype mixing target image with same-category peers."""
    tgt = F.normalize(target_image_vec.detach().to(device='cpu', dtype=torch.float64).reshape(1, -1), dim=-1)
    mix = min(max(float(visual_prototype_mix), 0.0), 1.0)
    if not same_category_indices or mix <= 0.0:
        return tgt.to(dtype=target_image_vec.dtype, device=target_image_vec.device)
    peers = F.normalize(
        candidate_vectors[list(same_category_indices)].detach().to(device='cpu', dtype=torch.float64),
        dim=-1,
    )
    if peers.shape[0] == 0:
        return tgt.to(dtype=target_image_vec.dtype, device=target_image_vec.device)
    peer_w = torch.full((peers.shape[0],), mix / float(peers.shape[0]), dtype=torch.float64)
    tgt_w = torch.tensor([1.0 - mix], dtype=torch.float64)
    all_vecs = torch.cat([tgt, peers], dim=0)
    all_weights = torch.cat([tgt_w, peer_w], dim=0)
    proto = spherical_karcher_mean(all_vecs, all_weights, steps=25)
    return proto.to(dtype=target_image_vec.dtype, device=target_image_vec.device).detach()


def build_category_abstraction(
    training_bundle: dict,
    text_encode_fn: Callable[[str], object],
    candidate_vectors: torch.Tensor,
    target_image_vec: torch.Tensor,
    *,
    same_category_indices: Sequence[int] | None = None,
    subspace_rank: int = 6,
    medoid_count: int = 12,
    residual_shrinkage: float = 0.30,
    visual_prototype_mix: float = 0.65,
    contrastive_beta: float = 0.35,
    precomputed_query_vecs: torch.Tensor | None = None,
) -> dict:
    """Construct complete category-commonality abstraction strictly from public bundle data."""
    queries = list(training_bundle['queries'])
    if not queries:
        raise ValueError('公开训练包查询为空，无法构造类别共性抽象')
    category = str(training_bundle['audit']['category'])
    target_concept = str(training_bundle['target_concept_text'])
    provenance = training_bundle['audit'].get('query_provenance', [])
    if len(provenance) == len(queries):
        strata = [str(row.get('source_type', 'category_description')) for row in provenance]
    else:
        strata = ['category_description'] * len(queries)

    # Equal total weight across present query strata so 12-20 concise search queries
    # carry equal weight to 50 long product descriptions.
    stratum_counts = Counter(strata)
    n_strata = max(1, len(stratum_counts))
    query_weights = torch.tensor(
        [1.0 / (n_strata * stratum_counts[s]) for s in strata],
        dtype=torch.float64,
    )
    query_weights = query_weights / query_weights.sum().clamp_min(1e-12)

    out_titles: list[str] = []
    seen_out: set[str] = set()
    cat_norm = _norm_cat(category)
    for row in training_bundle.get('candidate_metadata', []):
        r_cat = _norm_cat(row.get('category'))
        title = ' '.join(str(row.get('title') or '').split())
        if title and r_cat != cat_norm and title.casefold() not in seen_out:
            seen_out.add(title.casefold())
            out_titles.append(title)

    if precomputed_query_vecs is not None:
        q_mat = F.normalize(precomputed_query_vecs.detach().to(device='cpu', dtype=torch.float64), dim=-1)
    else:
        q_rows = [
            torch.as_tensor(text_encode_fn(q), dtype=torch.float64).reshape(1, -1)
            for q in queries
        ]
        q_mat = F.normalize(torch.cat(q_rows, dim=0), dim=-1)
    dim = int(q_mat.shape[1])

    # 1. Contrastive TF-IDF salience + Canonical commonality prompts + Submodular medoids
    salience, ranked_terms = compute_contrastive_salience(queries, out_titles)
    canonical_prompts = _build_canonical_commonality_prompts(category, target_concept, ranked_terms)
    raw_clauses = _extract_candidate_clauses(queries, strata)
    scored_clauses = sorted(
        raw_clauses,
        key=lambda item: (-_clause_salience_score(item[0], salience), item[0]),
    )
    pool_limit = min(len(scored_clauses), max(48, 4 * int(medoid_count)))
    clause_pool = scored_clauses[:pool_limit]
    clause_texts = [c for c, _ in clause_pool]
    c_rows = [
        torch.as_tensor(text_encode_fn(c), dtype=torch.float64).reshape(1, -1)
        for c in clause_texts
    ]
    c_mat = F.normalize(torch.cat(c_rows, dim=0), dim=-1) if c_rows else torch.zeros((0, dim), dtype=torch.float64)
    medoid_budget = max(0, int(medoid_count) - len(canonical_prompts))
    selected_idx, coverage_score = lazy_greedy_facility_location(q_mat, query_weights, c_mat, medoid_budget)

    abstracted_texts = list(canonical_prompts) + [clause_texts[i] for i in selected_idx]
    abstracted_vecs_list = [
        torch.as_tensor(text_encode_fn(t), dtype=torch.float64).reshape(1, -1)
        for t in abstracted_texts
    ]
    abstracted_vecs = F.normalize(torch.cat(abstracted_vecs_list, dim=0), dim=-1)

    # 2. Stratum-balanced Riemannian Spherical Karcher Mean
    mu_cat = spherical_karcher_mean(q_mat, query_weights, steps=25)

    # 3. Contrastive Principal Subspace & Principal Anchors
    out_sample_titles = out_titles[:40]
    if out_sample_titles:
        neg_title_rows = [
            torch.as_tensor(text_encode_fn(t), dtype=torch.float64).reshape(1, -1)
            for t in out_sample_titles
        ]
        neg_text_vecs = F.normalize(torch.cat(neg_title_rows, dim=0), dim=-1)
    else:
        neg_text_vecs = None

    basis, proj_vars, principal_anchors = build_contrastive_category_subspace(
        q_mat,
        query_weights,
        mu_cat,
        neg_vecs=neg_text_vecs,
        subspace_rank=subspace_rank,
        contrastive_beta=contrastive_beta,
    )

    # 4. James-Stein Subspace-Denoised Query Bank (both pure gamma=0 and hybrid gamma=residual_shrinkage)
    denoised_hybrid = denoise_queries_james_stein(q_mat, mu_cat, basis, residual_shrinkage)
    denoised_pure = denoise_queries_james_stein(q_mat, mu_cat, basis, 0.0)

    # 5. In-category Public Visual Karcher Prototype
    if same_category_indices is None:
        same_category_indices = [
            idx for idx, row in enumerate(training_bundle.get('candidate_metadata', []))
            if _norm_cat(row.get('category')) == cat_norm
        ]
    visual_prototype = compute_visual_category_prototype(
        target_image_vec,
        candidate_vectors,
        same_category_indices,
        visual_prototype_mix=visual_prototype_mix,
    )

    # Pure abstracted semantic anchor bank = [mu_cat] + principal_anchors + abstracted_text_vecs
    pure_semantic_bank = F.normalize(
        torch.cat([mu_cat, principal_anchors, abstracted_vecs], dim=0),
        dim=-1,
    )

    dev = target_image_vec.device
    dt = torch.float32
    summary_payload = {
        'schema_version': ABSTRACTION_SCHEMA_VERSION,
        'kind': ABSTRACTION_KIND,
        'category': category,
        'query_count': len(queries),
        'out_of_category_titles_used': len(out_sample_titles),
        'same_category_visual_peers_used': len(list(same_category_indices)),
        'top_salient_terms': ranked_terms[:12],
        'canonical_prompts': canonical_prompts,
        'abstracted_texts': abstracted_texts,
        'submodular_coverage_score': round(float(coverage_score), 8),
        'subspace_rank': int(basis.shape[0]),
        'ambient_dim': dim,
        'principal_anchor_count': int(principal_anchors.shape[0]),
        'residual_shrinkage': float(residual_shrinkage),
        'visual_prototype_mix': float(visual_prototype_mix),
        'contrastive_beta': float(contrastive_beta),
    }
    summary_sha = hashlib.sha256(repr(sorted(summary_payload.items())).encode('utf-8')).hexdigest()
    summary_payload['abstraction_sha256'] = summary_sha

    return {
        'mu_cat': mu_cat.to(device=dev, dtype=dt).detach(),
        'basis': basis.to(device=dev, dtype=dt).detach(),
        'principal_anchors': principal_anchors.to(device=dev, dtype=dt).detach(),
        'abstracted_texts': abstracted_texts,
        'abstracted_text_vecs': abstracted_vecs.to(device=dev, dtype=dt).detach(),
        'pure_semantic_bank': pure_semantic_bank.to(device=dev, dtype=dt).detach(),
        'denoised_queries_hybrid': denoised_hybrid.to(device=dev, dtype=dt).detach(),
        'denoised_queries_pure': denoised_pure.to(device=dev, dtype=dt).detach(),
        'raw_queries': q_mat.to(device=dev, dtype=dt).detach(),
        'visual_category_prototype': visual_prototype.to(device=dev, dtype=dt).detach(),
        'audit': summary_payload,
    }
