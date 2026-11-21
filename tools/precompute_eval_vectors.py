"""提前计算并持久化记忆库与测试文字集在三个检索编码模型（及 SBERT GHR 目标语句）上的编码向量。

支持命令：
  1. `build`（默认）：扫描项目基础记忆库（original_9groups, golden_band_9groups 等）、
     全部测试文字查询集（training_oracle_v4, overlap_30_v4, zero_overlap_v4, evaluation_inputs）
     以及冻结攻击成图（FROZEN_IMAGE.json），在三个编码模型（01_clip_l14, 02_openclip_bigG14,
     03_siglip_so400m）上使用高速批量前向一次性预计算并落盘存储，后续检索与下游测试免模型加载、零 GPU。
  2. `verify`：在纯 CPU 零 GPU 模式下读取预计算向量库，与已归档的 `scores.npz` 逐元素比对，
     验证离线内积与原 GPU 检索得分误差 `<= 1e-7`。
  3. `status`：查看当前已落盘预计算向量库的覆盖统计。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.shared_evaluation import SCORE_TOLERANCE
from src.paths import OUTPUT
from src.visual.audit import read_json
from src.visual.precomputed_eval_store import (
    DEFAULT_PRECOMPUTED_STORE_DIR,
    ENCODER_SPEC_BY_SLUG,
    PrecomputedEncoderStore,
    precompute_all_eval_vectors,
)


CROSS_BATCH_FLOAT32_TOLERANCE = 1e-5


def _verify_against_eval_root(eval_root: Path, store_root: Path) -> dict:
    eval_root = eval_root.resolve()
    score_files = sorted(eval_root.glob("**/scores.npz"))
    if not score_files:
        raise FileNotFoundError(f"未在 {eval_root} 下找到任何 scores.npz 归档")

    stores = {
        slug: PrecomputedEncoderStore(slug, store_root)
        for slug in ENCODER_SPEC_BY_SLUG
    }

    checked_groups = 0
    checked_queries = 0
    checked_scores = 0
    max_abs_err = 0.0
    rank_mismatches = 0
    top5_mismatches = 0

    for spath in score_files:
        group_dir = spath.parent
        plan_path = group_dir / "evaluation_plan.json"
        eval_path = group_dir / "evaluation.json"
        if not plan_path.is_file() or not eval_path.is_file():
            continue
        parts = spath.parts
        slug = next((p for p in parts if p in ENCODER_SPEC_BY_SLUG), None)
        if slug is None:
            continue
        store = stores[slug]
        plan = read_json(plan_path)
        rows = read_json(eval_path)
        key_mode = plan.get("key_mode") or "image"
        key_weight = float(plan.get("key_image_weight", 1.0 if key_mode == "image" else 0.5))
        index_text = plan.get("index_text") or "product_feature"
        pool_path = Path(plan["pool_path"])
        queries = plan["queries"]

        with np.load(spath, allow_pickle=False) as arc:
            saved_scores = arc["scores"]

        for row_idx, row in enumerate(rows):
            bank = read_json(row["bank_path"])
            ranked_res = store.rank_bank_queries(
                bank,
                queries,
                pool_path=pool_path,
                key_mode=key_mode,
                key_image_weight=key_weight,
                index_text=index_text,
                source_memory_id=plan["source_memory_id"],
            )
            offline_scores = ranked_res["scores"]
            err = float(np.max(np.abs(offline_scores - saved_scores[row_idx])))
            if err > CROSS_BATCH_FLOAT32_TOLERANCE:
                raise ValueError(
                    f"离线预计算向量得分与 {spath} (row={row_idx}) 误差超限: {err:.3e} > {CROSS_BATCH_FLOAT32_TOLERANCE}"
                )
            for q_idx, q_res in enumerate(ranked_res["queries"]):
                checked_queries += 1
                saved_order = np.argsort(-saved_scores[row_idx, q_idx], kind="stable").tolist()
                if q_res["top_indices"] != saved_order[:5]:
                    top5_mismatches += 1
                if q_res["source_rank"] != row["queries"][q_idx]["rank"]:
                    rank_mismatches += 1
            max_abs_err = max(max_abs_err, err)
            checked_scores += int(offline_scores.size)
        checked_groups += 1

    if rank_mismatches > 0 or top5_mismatches > 0:
        raise ValueError(f"发现排名不一致: rank_mismatches={rank_mismatches}, top5_mismatches={top5_mismatches}")

    return {
        "status": "passed",
        "eval_root": str(eval_root),
        "store_root": str(store_root),
        "checked_group_archives": checked_groups,
        "checked_queries": checked_queries,
        "checked_candidate_scores": checked_scores,
        "max_score_error": max_abs_err,
        "rank_mismatches": rank_mismatches,
        "top5_mismatches": top5_mismatches,
        "gpu_used": False,
        "models_loaded": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    p_build = sub.add_parser("build", help="批量预计算并保存记忆库与测试文字集在三编码器上的向量")
    p_build.add_argument("--store-dir", type=Path, default=DEFAULT_PRECOMPUTED_STORE_DIR)
    p_build.add_argument("--encoder-slugs", nargs="+", choices=tuple(ENCODER_SPEC_BY_SLUG), default=None)
    p_build.add_argument("--bases", nargs="+", default=None, help="指定基础包名称或路径（默认全部已注册基础包）")
    p_build.add_argument("--suites", nargs="+", default=None, help="指定测试查询集（默认全部 v4 查询集）")
    p_build.add_argument("--images-root", action="append", default=None, help="额外扫描的冻结成图目录")
    p_build.add_argument("--no-scan-output", action="store_true", help="不自动扫描 output/ 下的全部 FROZEN_IMAGE.json")
    p_build.add_argument("--batch-size", type=int, default=16)
    p_build.add_argument("--text-batch-size", type=int, default=64)
    p_build.add_argument("--num-workers", type=int, default=4)
    p_build.add_argument("--device", default="cuda")
    p_build.add_argument("--skip-sbert", action="store_true")

    p_verify = sub.add_parser("verify", help="零 GPU 验证预计算向量与已有 scores.npz 的逐元素等价性")
    p_verify.add_argument("--store-dir", type=Path, default=DEFAULT_PRECOMPUTED_STORE_DIR)
    p_verify.add_argument(
        "--eval-root",
        type=Path,
        default=OUTPUT / "exp_20261121_eval_v4_cleaned" / "eval",
        help="包含已评测 scores.npz 的目录",
    )

    p_status = sub.add_parser("status", help="查看预计算向量库状态")
    p_status.add_argument("--store-dir", type=Path, default=DEFAULT_PRECOMPUTED_STORE_DIR)

    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] not in ("build", "verify", "status", "-h", "--help"):
        args_list = ["build", *args_list]
    args = parser.parse_args(args_list)

    if args.command == "status":
        manifest_file = Path(args.store_dir).resolve() / "store_manifest.json"
        if not manifest_file.is_file():
            print(json.dumps({"exists": False, "store_dir": str(args.store_dir)}, ensure_ascii=False, indent=2))
            return 1
        print(json.dumps(read_json(manifest_file), ensure_ascii=False, indent=2))
        return 0

    if args.command == "verify":
        t0 = time.perf_counter()
        res = _verify_against_eval_root(args.eval_root, args.store_dir)
        res["elapsed_seconds"] = round(time.perf_counter() - t0, 4)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    t0 = time.perf_counter()
    batch_sizes = {slug: int(args.batch_size) for slug in ENCODER_SPEC_BY_SLUG}
    manifest = precompute_all_eval_vectors(
        args.store_dir,
        encoder_slugs=args.encoder_slugs,
        bases=args.bases,
        query_suites=args.suites,
        extra_image_roots=args.images_root,
        include_all_output_frozen=not args.no_scan_output,
        batch_sizes=batch_sizes,
        text_batch_size=args.text_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        precompute_sbert=not args.skip_sbert,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"\n[PRECOMPUTE COMPLETED in {elapsed}s] Store saved to: {Path(args.store_dir).resolve()}", flush=True)
    print(json.dumps(manifest["encoders"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
