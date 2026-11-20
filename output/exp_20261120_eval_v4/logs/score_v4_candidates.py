#!/usr/bin/env python3
"""Score all training_bundle queries and catalog_pool heldout candidates across the 3 encoders using cached bank vectors."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

REPO = Path('/content/AmbiLAGE')
sys.path.insert(0, str(REPO))

from src.experiments import shared_evaluation
from src.experiments.exact_overlap_queries import (
    ROOT,
    TRAINING_ROOT,
    DuplicateIndex,
    source_exclusions,
    tokens,
    training_index,
)
from src.experiments.visual_inception_category_evaluation import (
     _resolve_transfer_panel_encoders,
    load_frozen_suite,
)
from src.paths import EVALUATION_BASE
from src.visual.encoders.eval_clip import make_standalone_eval_encoder
from src.visual.visual_retrieval import unit
from tools.build_visual_inception_eval_v4_suites import (
    CATALOG,
    CATALOG_INFO,
    SEED,
    acceptable,
    catalog_pool,
)

VEC_DIR = REPO / 'output/exp_20261120_eval_v4/memory_vectors'
EVAL_DIR = REPO / 'output/exp_20261120_eval_v4/eval'
OUT_PATH = REPO / 'output/exp_20261120_eval_v4/logs/candidate_ranks_cache.json'
ENCODERS = ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m')


def load_cached_bank_matrices() -> dict:
    by_model_img = {}
    for p in VEC_DIR.glob('*.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        ident = d['identity']
        by_model_img[(ident['model_sha256'], ident['image_sha256'])] = unit(np.asarray(d['vector'], dtype=np.float32))

    matrices = {}
    for slug in ENCODERS:
        matrices[slug] = {}
        for m_key in ('ambilage', 'vi_08_02'):
            ev_dir = EVAL_DIR / f'{m_key}_training_oracle_v4' / slug
            g0_rows = json.loads((ev_dir / 'groups/01_camera/evaluation.json').read_text(encoding='utf-8'))
            m_sha = g0_rows[0]['retrieval_protocol']['model_sha256']
            for row in g0_rows:
                bname = row['bank']
                key = 'clean' if bname == 'clean' else m_key
                if key in matrices[slug]:
                    continue
                bank = json.loads(Path(row['bank_path']).read_text(encoding='utf-8'))
                vecs = []
                for entry in bank:
                    rec = entry.get('RetrievalImage') if entry.get('RetrievalImage') is not None else entry.get('PrimaryImage')
                    vecs.append(by_model_img[(m_sha, rec['SHA256'])])
                matrices[slug][key] = np.vstack(vecs).astype(np.float32, copy=False)

    # Also load source_idx per group
    src_indices = {}
    g0_plan = json.loads((EVAL_DIR / 'ambilage_training_oracle_v4/01_clip_l14/groups/01_camera/evaluation_plan.json').read_text(encoding='utf-8'))
    pool = shared_evaluation.load_shared_pool(g0_plan['pool_path'])
    ids = [e['Id'] for e in pool]
    for g_dir in sorted((EVAL_DIR / 'ambilage_training_oracle_v4/01_clip_l14/groups').glob('0*')):
        gp = json.loads((g_dir / 'evaluation_plan.json').read_text(encoding='utf-8'))
        src_indices[g_dir.name] = ids.index(gp['source_memory_id'])

    return matrices, src_indices


def compute_rank(scores: np.ndarray, src_idx: int) -> int:
    s_src = float(scores[src_idx])
    # Deterministic tie-breaking: (-score, index)
    better = int(np.sum(scores > s_src))
    tied_before = int(np.sum(scores[:src_idx] == s_src))
    return 1 + better + tied_before


def main() -> None:
    t0 = time.time()
    matrices, src_indices = load_cached_bank_matrices()
    specs, bundles, training, texts, concepts = training_index(EVALUATION_BASE, TRAINING_ROOT)
    used_searches, products = source_exclusions(bundles, TRAINING_ROOT, specs)
    dupes = DuplicateIndex(texts | concepts)
    pairs_doc = json.loads((ROOT / 'pairs.json').read_text(encoding='utf-8'))['groups']

    catalog_manifest = json.loads(CATALOG_INFO.read_text(encoding='utf-8'))
    entry = next(row for row in catalog_manifest['files'] if row['path'] == 'catalog.sqlite')

    cat_candidates = {}
    with closing(sqlite3.connect(CATALOG.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        for g, spec in specs.items():
            pool = catalog_pool(db, g, spec, products, dupes, entry['sha256'], SEED)
            # Keep up to 120 distinct product items (or all for 03_cookies)
            selected = []
            seen_p = set()
            for r in pool:
                if g != '03_cookies' and r['product_id'] in seen_p and len(selected) >= 60:
                    continue
                selected.append(r)
                seen_p.add(r['product_id'])
                if len(selected) >= 120:
                    break
            cat_candidates[g] = selected

    # Collect all unique texts to score across all 9 groups
    all_texts_set = set()
    for g in specs:
        for q in bundles[g]['queries']:
            all_texts_set.add(q['text'])
        for p in pairs_doc[g]:
            all_texts_set.add(p['training']['text'])
            all_texts_set.add(p['heldout']['text'])
        for r in cat_candidates[g]:
            all_texts_set.add(r['original_text'])

    unique_texts = sorted(all_texts_set)
    print(f'Collected {len(unique_texts)} unique candidate texts in {time.time()-t0:.2f}s', flush=True)

    frozen_suite = load_frozen_suite(REPO / 'output/ambilage_single_clip_l14_9groups_20261118', method='ambilage')
    panel_encoders, _ = _resolve_transfer_panel_encoders(frozen_suite)
    training_ctx = shared_evaluation._frozen_training_context(frozen_suite[0], frozen_suite, required=True)
    actual_surrogates = shared_evaluation._context_training_identities(training_ctx)

    text_vecs_by_enc = {}
    for enc_spec in panel_encoders:
        slug = enc_spec['slug']
        t_enc = time.time()
        enc = make_standalone_eval_encoder(
            device='cuda',
            config_path=enc_spec['encoder_config'],
            training_surrogates=actual_surrogates,
            batch_size=16,
            num_workers=4,
        )
        try:
            pairs_out = enc.encode_text_batch(unique_texts, batch_size=64)
            text_vecs_by_enc[slug] = {t: unit(vec) for t, (vec, _audit) in zip(unique_texts, pairs_out)}
        finally:
            enc.close()
        print(f'Encoded {len(unique_texts)} texts on {slug} in {time.time()-t_enc:.2f}s', flush=True)

    def score_text_for_group(g: str, text: str) -> dict:
        src_idx = src_indices[g]
        res = {}
        for slug in ENCODERS:
            q_vec = text_vecs_by_enc[slug][text]
            res[slug] = {
                bkey: compute_rank(matrices[slug][bkey] @ q_vec, src_idx)
                for bkey in ('clean', 'ambilage', 'vi_08_02')
            }
        return res

    output_data = {'groups': {}}
    for g in specs:
        tr_rows = []
        for q in bundles[g]['queries']:
            tr_rows.append(dict(
                id=q['id'],
                source_type=q['source_type'],
                text=q['text'],
                acceptable=bool(acceptable(g, q['text'])),
                token_count=len(tokens(q['text'])),
                ranks=score_text_for_group(g, q['text']),
            ))
        ho_cat_rows = []
        for r in cat_candidates[g]:
            ho_cat_rows.append(dict(
                record=r,
                token_count=len(tokens(r['original_text'])),
                ranks=score_text_for_group(g, r['original_text']),
            ))
        cur_pairs = []
        for p in pairs_doc[g]:
            cur_pairs.append(dict(
                id=p['id'],
                family=p['family'],
                reuse_at_30_percent=p['reuse_at_30_percent'],
                training=p['training'],
                training_ranks=score_text_for_group(g, p['training']['text']),
                heldout=p['heldout'],
                heldout_ranks=score_text_for_group(g, p['heldout']['text']),
            ))
        output_data['groups'][g] = dict(
            training_pool=tr_rows,
            catalog_candidates=ho_cat_rows,
            current_pairs=cur_pairs,
        )

    OUT_PATH.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved {OUT_PATH} in {time.time()-t0:.2f}s total', flush=True)


if __name__ == '__main__':
    main()
