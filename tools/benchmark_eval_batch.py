"""Gate 1 micro-benchmark: validate StandaloneEvalClipEncoder equivalence, VRAM, and batch reproducibility."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.experiments import shared_evaluation
from src.paths import OUTPUT
from src.visual.audit import read_json, save_json
from src.visual.encoders.eval_clip import make_standalone_eval_encoder
from src.visual.image_utils import sha256_file
from src.visual.visual_retrieval import resolve_image, unit

BENCH_CONDITIONS = ('none', 'blur2', 'jpeg50', 'diff512_t199_s1001', 'sd21_diff512_t199_s1001')
BENCH_GROUPS = ('01_camera', '07_shampoo')
CANDIDATE_BATCH_SIZES = (32, 16, 8)
ENCODER_SLUGS = ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m')


def _full_rank_order(scores: np.ndarray) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def _evaluate_subset(enc, vec_map: dict[str, np.ndarray], group_records: dict[str, dict]) -> dict:
    """Recompute all candidate scores & metrics for selected (condition, bank, query) rows."""
    max_score_diff = 0.0
    rank_mismatches = 0
    src_rank_mismatches = 0
    total_scores = 0

    clean_ranks, mod_ranks = [], []
    leg_clean_ranks, leg_mod_ranks = [], []

    for group, rec in group_records.items():
        ids = rec['ids']
        src_idx = rec['src_idx']
        plan = rec['plan']
        pool_path = plan['pool_path']
        queries = plan['queries']
        q_vecs = [unit(enc.encode_text(q['instruction'])) for q in queries]

        cond_labels = rec['condition_labels']
        banks = rec['banks']
        legacy_scores = rec['scores']

        for row_idx, (cond, bank_name) in enumerate(zip(cond_labels, banks)):
            if cond not in BENCH_CONDITIONS:
                continue
            bank = rec['loaded_banks'][(cond, bank_name)]
            bank_matrix = np.asarray(
                [unit(vec_map[rec['bank_shas'][(cond, bank_name)][i]]) for i in range(len(bank))],
                dtype=np.float32,
            )
            for q_idx, q_vec in enumerate(q_vecs):
                # Match VisualMemoryRetriever.rank: float(vector @ query) per entry.
                fresh = np.asarray([float(v @ q_vec) for v in bank_matrix], dtype=np.float64)
                leg = legacy_scores[row_idx, q_idx]
                diff = float(np.max(np.abs(fresh - leg)))
                if diff > max_score_diff:
                    max_score_diff = diff
                total_scores += len(ids)

                fresh_order = _full_rank_order(fresh)
                leg_order = _full_rank_order(leg)
                if fresh_order != leg_order:
                    rank_mismatches += 1
                fresh_src_rank = fresh_order.index(src_idx) + 1
                leg_src_rank = leg_order.index(src_idx) + 1
                if fresh_src_rank != leg_src_rank:
                    src_rank_mismatches += 1

                if bank_name == 'clean':
                    clean_ranks.append(fresh_src_rank)
                    leg_clean_ranks.append(leg_src_rank)
                else:
                    mod_ranks.append(fresh_src_rank)
                    leg_mod_ranks.append(leg_src_rank)

    assert len(mod_ranks) == len(leg_mod_ranks) > 0
    n = len(mod_ranks)
    top1 = sum(r == 1 for r in mod_ranks)
    leg_top1 = sum(r == 1 for r in leg_mod_ranks)
    top3 = sum(r <= 3 for r in mod_ranks)
    leg_top3 = sum(r <= 3 for r in leg_mod_ranks)
    top5 = sum(r <= 5 for r in mod_ranks)
    leg_top5 = sum(r <= 5 for r in leg_mod_ranks)
    mean_rank = sum(mod_ranks) / n
    leg_mean_rank = sum(leg_mod_ranks) / n

    return dict(
        total_scores=total_scores,
        max_score_diff_vs_legacy_b1=max_score_diff,
        rank_mismatches_vs_legacy_b1=rank_mismatches,
        source_rank_mismatches_vs_legacy_b1=src_rank_mismatches,
        top1_delta=top1 - leg_top1,
        top3_delta=top3 - leg_top3,
        top5_delta=top5 - leg_top5,
        mean_rank_delta=mean_rank - leg_mean_rank,
    )


def _benchmark_encoder(run_dir: Path, slug: str, device: str, *, b1_only: bool = False) -> dict:
    enc_dir = run_dir / slug
    group_records = {}
    unique_assets: dict[str, Path] = {}
    sample_plan = None

    for group in BENCH_GROUPS:
        gdir = enc_dir / 'groups' / group
        plan = read_json(gdir / 'evaluation_plan.json')
        sample_plan = plan
        rows = read_json(gdir / 'evaluation.json')
        row_by_key = {(r['condition']['label'], r['bank']): r for r in rows}

        with np.load(gdir / 'scores.npz', allow_pickle=False) as saved:
            scores = saved['scores'].copy()
            memory_ids = saved['memory_ids'].tolist()
            condition_labels = saved['condition_labels'].tolist()
            banks = saved['banks'].tolist()
            src_id = str(saved['source_memory_id'].item())

        loaded_banks = {}
        bank_shas = {}
        pool_path = plan['pool_path']
        for cond in BENCH_CONDITIONS:
            for bname in ('clean', 'modified'):
                row = row_by_key[(cond, bname)]
                bank = read_json(row['bank_path'])
                loaded_banks[(cond, bname)] = bank
                shas = []
                for entry in bank:
                    img_path = resolve_image(entry, pool_path)
                    rec = entry.get('RetrievalImage') if entry.get('RetrievalImage') is not None else entry.get('PrimaryImage')
                    sha = rec.get('SHA256') or sha256_file(img_path)
                    shas.append(sha)
                    if sha not in unique_assets:
                        unique_assets[sha] = img_path
                bank_shas[(cond, bname)] = shas

        group_records[group] = dict(
            plan=plan,
            ids=memory_ids,
            src_idx=memory_ids.index(src_id),
            condition_labels=condition_labels,
            banks=banks,
            scores=scores,
            loaded_banks=loaded_banks,
            bank_shas=bank_shas,
        )

    ordered = sorted(unique_assets.items(), key=lambda kv: kv[0])
    ordered_shas = [sha for sha, _ in ordered]
    ordered_paths = [path for _, path in ordered]
    print(f'\n=== [{slug}] unique_assets={len(ordered)} groups={BENCH_GROUPS} ===', flush=True)

    actual_surrogates = shared_evaluation._plan_training_identities(sample_plan)
    t_load0 = time.perf_counter()
    enc = make_standalone_eval_encoder(
        device,
        sample_plan.get('encoder_config'),
        training_surrogates=actual_surrogates,
        batch_size=1,
        num_workers=4,
    )
    load_sec = time.perf_counter() - t_load0
    print(f'[{slug}] model loaded in {load_sec:.2f}s', flush=True)

    try:
        # Step 1: Standalone batch_size=1 vs legacy MatchedClipEncoder scores.npz
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        b1_list = enc.encode_batch(ordered_paths, batch_size=1)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        b1_sec = time.perf_counter() - t0
        b1_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.startswith('cuda') else 0.0
        b1_map = dict(zip(ordered_shas, b1_list))
        b1_eval = _evaluate_subset(enc, b1_map, group_records)
        print(
            f"[{slug}] batch=1: time={b1_sec:.2f}s ({len(ordered)/b1_sec:.1f} img/s) "
            f"vram={b1_vram:.2f}GiB score_diff={b1_eval['max_score_diff_vs_legacy_b1']:.2e} "
            f"rank_mismatch={b1_eval['rank_mismatches_vs_legacy_b1']} "
            f"src_rank_mismatch={b1_eval['source_rank_mismatches_vs_legacy_b1']} "
            f"top1_d={b1_eval['top1_delta']} top3_d={b1_eval['top3_delta']} "
            f"top5_d={b1_eval['top5_delta']} mean_rank_d={b1_eval['mean_rank_delta']:.4f}",
            flush=True,
        )
        if (
            b1_eval['max_score_diff_vs_legacy_b1'] > 2.5e-7
            or b1_eval['rank_mismatches_vs_legacy_b1'] != 0
            or b1_eval['source_rank_mismatches_vs_legacy_b1'] != 0
            or b1_eval['top1_delta'] != 0
            or b1_eval['top3_delta'] != 0
            or b1_eval['top5_delta'] != 0
            or b1_eval['mean_rank_delta'] != 0.0
        ):
            raise RuntimeError(
                f"Re-plan condition hit: {slug} StandaloneEvalClipEncoder(batch=1) deviated from legacy b=1: {b1_eval}"
            )

        candidates = {}
        recommended = 1
        batch_sizes_to_test = () if b1_only else CANDIDATE_BATCH_SIZES
        for bs in batch_sizes_to_test:
            gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            try:
                t1 = time.perf_counter()
                bs_list = enc.encode_batch(ordered_paths, batch_size=bs)
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t1
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3) if device.startswith('cuda') else 0.0

                vec_map = dict(zip(ordered_shas, bs_list))
                max_vec_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(bs_list, b1_list))
                eval_stats = _evaluate_subset(enc, vec_map, group_records)

                # Same-order exact reproducibility
                bs_same = enc.encode_batch(ordered_paths, batch_size=bs)
                same_order_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(bs_list, bs_same))

                # Reversed-order invariance
                rev_paths = list(reversed(ordered_paths))
                bs_rev = list(reversed(enc.encode_batch(rev_paths, batch_size=bs)))
                rev_order_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(bs_list, bs_rev))

                passed = (
                    peak_vram <= 18.0
                    and eval_stats['rank_mismatches_vs_legacy_b1'] == 0
                    and eval_stats['top1_delta'] == 0
                    and eval_stats['top3_delta'] == 0
                    and eval_stats['top5_delta'] == 0
                    and eval_stats['mean_rank_delta'] == 0.0
                    and eval_stats['max_score_diff_vs_legacy_b1'] <= 5e-6
                    and same_order_diff <= 1e-7
                )
                if passed and bs > recommended:
                    recommended = bs

                rec = dict(
                    batch_size=bs,
                    oom=False,
                    elapsed_sec=elapsed,
                    imgs_per_sec=len(ordered) / elapsed,
                    peak_vram_gib=peak_vram,
                    max_vec_diff_vs_b1=max_vec_diff,
                    same_order_repro_max_diff=same_order_diff,
                    reversed_order_max_diff=rev_order_diff,
                    passed_gate=passed,
                    **eval_stats,
                )
                print(
                    f"[{slug}] batch={bs}: time={elapsed:.2f}s ({rec['imgs_per_sec']:.1f} img/s) "
                    f"vram={peak_vram:.2f}GiB score_diff={rec['max_score_diff_vs_legacy_b1']:.2e} "
                    f"rank_mismatch={rec['rank_mismatches_vs_legacy_b1']} same_repro={same_order_diff:.2e} "
                    f"rev_diff={rev_order_diff:.2e} gate={'PASS' if passed else 'FAIL'}",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError as exc:
                gc.collect()
                torch.cuda.empty_cache()
                rec = dict(batch_size=bs, oom=True, error=str(exc), passed_gate=False)
                print(f"[{slug}] batch={bs}: OOM -> skipped", flush=True)
            candidates[str(bs)] = rec

        return dict(
            encoder_slug=slug,
            unique_assets=len(ordered),
            model_load_sec=load_sec,
            recommended_batch_size=recommended,
            batch_1=dict(
                elapsed_sec=b1_sec,
                imgs_per_sec=len(ordered) / b1_sec,
                peak_vram_gib=b1_vram,
                **b1_eval,
            ),
            candidates=candidates,
        )
    finally:
        enc.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-dir',
        type=Path,
        default=OUTPUT / 'eval_9groups_20261119' / 'stage2_defenses' / 'ambilage_training_oracle',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=OUTPUT / 'eval_9groups_20261119' / 'stage2_defenses' / 'logs' / 'batch_micro_benchmark.json',
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--b1-only', action='store_true', help='Verify batch=1 equivalence only without testing batch>1')
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    encoders = {}
    recommended_map = {}
    for slug in ENCODER_SLUGS:
        res = _benchmark_encoder(run_dir, slug, args.device, b1_only=args.b1_only)
        encoders[slug] = res
        recommended_map[slug] = res['recommended_batch_size']

    payload = dict(
        schema_version=1,
        source_run_dir=str(run_dir),
        b1_only=bool(args.b1_only),
        conditions=list(BENCH_CONDITIONS),
        groups=list(BENCH_GROUPS),
        recommended_batch_sizes=recommended_map,
        encoders=encoders,
    )
    save_json(args.output, payload)
    print('\n=== RECOMMENDED BATCH SIZES ===', json.dumps(recommended_map), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
