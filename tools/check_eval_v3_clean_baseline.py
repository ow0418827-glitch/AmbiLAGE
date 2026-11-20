#!/usr/bin/env python3
"""Gate 3: 在跑全量攻击前，用干净 source.jpg 对三套 v3 评测集与三编码器做一次快速基线校准。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.agent.memory.shared_pool import evaluation_banks, load_shared_pool
from src.experiments import shared_evaluation
from src.experiments.base_workflow import public_group
from src.experiments.visual_inception_category_evaluation import (
    QUERY_SUITE_ROOTS,
    TRANSFER_PANEL_ENCODERS,
    load_frozen_suite,
    validate_category_query_suite,
)
from src.paths import EVALUATION_BASE, OUTPUT, VISUAL_CONFIGS
from src.visual.audit import freeze_json, read_json
from src.visual.visual_retrieval import VisualMemoryRetriever, detect_index_text_policy

SUITES = ('training_oracle_v3', 'overlap_30_v3', 'zero_overlap_v3')
VEC_DIR = OUTPUT / 'eval_9groups_20261119' / 'stage1_none' / 'memory_vectors'
OUT_DIR = OUTPUT / 'exp_20261120_eval_v3' / 'clean_sanity'


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VEC_DIR.mkdir(parents=True, exist_ok=True)

    base = EVALUATION_BASE.resolve()
    base_manifest = read_json(base / 'manifest.json')
    groups = list(base_manifest['groups'])
    pool_rel = base_manifest['shared_memory']['pool_file']
    pool_path = (base / pool_rel).resolve()
    entries = load_shared_pool(pool_path)
    image_base_path = pool_path
    index_policy = detect_index_text_policy(entries)
    source_mapping = base_manifest['shared_memory']['source_mapping']

    # Construct a pure clean frozen-like descriptor list (no modified images needed)
    clean_items = []
    for g in groups:
        pub = public_group(base, g)
        clean_items.append(dict(
            group=g,
            source=pub['source'],
            image=pub['source']['image'],
            source_memory_id=source_mapping[g]['source_memory_id'],
        ))

    first_src = clean_items[0]
    banks, _ = evaluation_banks(
        entries, pool_path, first_src['source'], first_src['image'],
        source_memory_id=first_src['source_memory_id'], replacements=clean_items,
    )
    clean_bank_file = OUT_DIR / 'clean_shared_bank.json'
    if not clean_bank_file.is_file():
        freeze_json(clean_bank_file, banks['clean'])

    validations = {s: validate_category_query_suite(query_suite=s) for s in SUITES}
    query_cache = {
        (s, g): read_json(Path(validations[s]['groups'][g]['query_manifest']))['query_records']
        for s in SUITES
        for g in groups
    }

    frozen_suite = load_frozen_suite(OUTPUT / 'exp_20261120_vi_08_02_9groups', method='visual_inception')
    training_context = shared_evaluation._frozen_training_context(frozen_suite[0], frozen_suite, required=True)
    actual_surrogates = shared_evaluation._context_training_identities(training_context)

    results = {}
    for slug, label, _pair, matched_cfg, heldout_cfg in TRANSFER_PANEL_ENCODERS:
        cfg_name = matched_cfg if slug == '01_clip_l14' else heldout_cfg
        enc_cfg = VISUAL_CONFIGS / cfg_name
        base_enc = shared_evaluation.make_eval_encoder(
            'cuda', str(enc_cfg), training_surrogates=actual_surrogates
        )
        shared_enc = shared_evaluation.InProcessFreshCachedEncoder(base_enc)
        bank_digest_cache, path_sha_cache = {}, {}
        try:
            ranker = VisualMemoryRetriever(
                shared_enc,
                clean_bank_file,
                key_mode='image',
                query_mode='text',
                key_image_weight=1.0,
                query_image_weight=0.0,
                index_text=index_policy,
                cache_dir=None,
                entry_cache_dir=VEC_DIR,
                image_base_path=image_base_path,
                freeze_memory=True,
                path_sha_cache=path_sha_cache,
                bank_digest_cache=bank_digest_cache,
            )
            bank_data = read_json(clean_bank_file)
            results[slug] = dict(label=label, suites={})
            for suite in SUITES:
                ranks_all = []
                by_fam = {}
                for item in clean_items:
                    g = item['group']
                    sid = item['source_memory_id']
                    for rec in query_cache[(suite, g)]:
                        ranking = ranker.rank(bank_data, rec['instruction'], None)
                        rank = next(r['rank'] for r in ranking if r['Id'] == sid)
                        ranks_all.append(rank)
                        by_fam.setdefault(rec['query_family'], []).append(rank)
                n = len(ranks_all)
                results[slug]['suites'][suite] = dict(
                    top1=sum(r == 1 for r in ranks_all) / n,
                    top3=sum(r <= 3 for r in ranks_all) / n,
                    top5=sum(r <= 5 for r in ranks_all) / n,
                    mean_rank=sum(ranks_all) / n,
                    by_family={
                        fam: dict(
                            n=len(rs),
                            top1=sum(r == 1 for r in rs) / len(rs),
                            top3=sum(r <= 3 for r in rs) / len(rs),
                            top5=sum(r <= 5 for r in rs) / len(rs),
                            mean_rank=sum(rs) / len(rs),
                        )
                        for fam, rs in by_fam.items()
                    },
                )
        finally:
            base_enc.close()

    # Print formatted Clean Sanity Check Report & check Gate 3
    lines = [
        '# Stage 2 v3 干净原图基准校准（Gate 3）',
        '',
        '| 编码器 | 评测集 | Clean T1 | Clean T3 | Clean T5 | Clean 均名 | 子家族 T5 (desc / para / search) |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    gate_inverted = []
    for slug, _, _, _, _ in TRANSFER_PANEL_ENCODERS:
        label = results[slug]['label']
        for suite in SUITES:
            st = results[slug]['suites'][suite]
            bf = st['by_family']
            sub = (
                f"{bf['category_description']['top5']*100:.1f}% / "
                f"{bf['model_paraphrase']['top5']*100:.1f}% / "
                f"{bf['heldout_search_query']['top5']*100:.1f}%"
            )
            lines.append(
                f"| {label} | `{suite}` | {st['top1']*100:.1f}% | {st['top3']*100:.1f}% | "
                f"{st['top5']*100:.1f}% | {st['mean_rank']:.2f} | {sub} |"
            )
        o3 = results[slug]['suites']['training_oracle_v3']
        z3 = results[slug]['suites']['zero_overlap_v3']
        if z3['top5'] > o3['top5'] + 0.01 or z3['top3'] > o3['top3'] + 0.01:
            gate_inverted.append(
                f"{label}: zero_T3={z3['top3']*100:.1f}% vs oracle_T3={o3['top3']*100:.1f}%, "
                f"zero_T5={z3['top5']*100:.1f}% vs oracle_T5={o3['top5']*100:.1f}%"
            )

    lines += ['', f"- Gate 3 (clean zero_overlap_v3 <= oracle_v3 + 1.0 pp): **{'FAILED: ' + '; '.join(gate_inverted) if gate_inverted else 'PASSED'}**"]
    md = '\n'.join(lines) + '\n'
    (OUT_DIR / 'CLEAN_SANITY_CHECK.md').write_text(md, encoding='utf-8')
    (OUT_DIR / 'clean_sanity_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(md)
    return 1 if gate_inverted else 0


if __name__ == '__main__':
    raise SystemExit(main())
