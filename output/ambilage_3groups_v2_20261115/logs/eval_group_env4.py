"""Evaluate specified group(s) under Env 4 (feature text + target image) and none defense.

Runs per group against the full 189-entry shared memory pool (replacing only that group's
source_memory_id), on both ViT-g-14 (held-out default evaluator) and ViT-B-32 (matched
training surrogate), with full isolated-subprocess candidate score verification.
"""
from __future__ import annotations

import argparse
import copy
import math
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/content/AmbiLAGE")
sys.path.insert(0, str(REPO_ROOT))

from src.agent.memory.shared_pool import evaluation_banks, load_shared_pool  # noqa: E402
from src.experiments import shared_evaluation  # noqa: E402
from src.experiments.paired_metrics import paired_rank_metrics  # noqa: E402
from src.paths import EVALUATION_BASE, SHARED_MEMORY  # noqa: E402
from src.visual.attack_completion import validate_attack_completion  # noqa: E402
from src.visual.audit import digest, freeze_json, read_json, save_json  # noqa: E402
from src.visual.image_utils import sha256_file  # noqa: E402
from src.visual.visual_retrieval import VisualMemoryRetriever  # noqa: E402

SCORE_TOLERANCE = shared_evaluation.SCORE_TOLERANCE


def _is_verified_passed(eval_dir: Path) -> bool:
    iv = eval_dir / "independent_verification.json"
    rep = eval_dir / "REPORT.md"
    if not iv.is_file() or not rep.is_file():
        return False
    try:
        data = read_json(iv)
        return (
            data.get("status") == "passed"
            and data.get("all_candidate_scores_checked") is True
            and not (eval_dir / "VERIFICATION_FAILED.json").exists()
        )
    except Exception:
        return False


def _clear_partial_dir(eval_dir: Path) -> None:
    if eval_dir.exists() and not _is_verified_passed(eval_dir):
        shutil.rmtree(eval_dir)


def _score_vector(ranking, ids):
    lookup = {r["Id"]: r for r in ranking}
    if len(lookup) != len(ranking) or set(lookup) != set(ids):
        raise ValueError("Candidate ranking mismatch against pool ids")
    scores = np.asarray([lookup[i]["score"] for i in ids], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite candidate scores")
    expected = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
    if [r["Id"] for r in ranking] != [ids[i] for i in expected]:
        raise ValueError("Non-deterministic score/tie ranking")
    return scores


def _save_scores(output: Path, values, ids, queries, rows, source_id: str) -> Path:
    path = output / "scores.npz"
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            scores=np.asarray(values, dtype=np.float64),
            memory_ids=np.asarray(ids, dtype=str),
            query_ids=np.asarray([q["task_id"] for q in queries], dtype=str),
            condition_labels=np.asarray([r["condition"]["label"] for r in rows], dtype=str),
            banks=np.asarray([r["bank"] for r in rows], dtype=str),
            source_memory_id=np.asarray(source_id, dtype=str),
        )
    temporary.replace(path)
    return path


def verify_env4(output: Path, device: str = "cuda") -> dict:
    output = Path(output)
    plan = read_json(output / "evaluation_plan.json")
    if sha256_file(plan["pool_path"]) != plan["pool_sha256"]:
        raise ValueError("Shared pool changed after Env 4 evaluation")
    if sha256_file(plan["encoder_config"]) != plan["encoder_config_sha256"]:
        raise ValueError("Encoder config changed after Env 4 evaluation")
    if sha256_file(plan["target_image_path"]) != plan["target_image_sha256"]:
        raise ValueError("Target query image changed after Env 4 evaluation")

    entries = load_shared_pool(plan["pool_path"])
    ids = [e["Id"] for e in entries]
    rows = read_json(output / "evaluation.json")
    with np.load(output / "scores.npz", allow_pickle=False) as saved:
        scores = saved["scores"].copy()
        if (
            saved["memory_ids"].tolist() != ids
            or saved["query_ids"].tolist() != [q["task_id"] for q in plan["queries"]]
            or saved["condition_labels"].tolist() != [r["condition"]["label"] for r in rows]
            or saved["banks"].tolist() != [r["bank"] for r in rows]
            or saved["source_memory_id"].item() != plan["source_memory_id"]
        ):
            raise ValueError("Env 4 scores.npz metadata mismatch")

    encoder = shared_evaluation.make_eval_encoder(device, plan["encoder_config"])
    max_error, count, score_count = 0.0, 0, 0
    try:
        for row_index, row in enumerate(rows):
            if sha256_file(row["bank_path"]) != row["bank_sha256"]:
                raise ValueError("Bank file changed")
            bank = read_json(row["bank_path"])
            if len(bank) != plan["pool_count"] or [e["Id"] for e in bank] != ids:
                raise ValueError("Pool ids mismatch")
            ranker = VisualMemoryRetriever(
                encoder,
                row["bank_path"],
                key_mode="joint",
                query_mode="joint",
                index_text="auto",
                key_image_weight=0.5,
                query_image_weight=0.5,
                cache_dir=None,
                entry_cache_dir=None,
                image_base_path=plan["pool_path"],
            )
            if digest(encoder.snapshot) != row["retrieval_protocol"]["model_sha256"]:
                raise ValueError("Fresh verification encoder identity changed")
            for query_index, (actual_q, q) in enumerate(zip(row["queries"], plan["queries"])):
                if sha256_file(q["query_image"]) != q["query_image_sha256"]:
                    raise ValueError("Target query image hash mismatch")
                ranking = ranker.rank(bank, q["instruction"], q["query_image"])
                fresh_scores = _score_vector(ranking, ids)
                archived_scores = scores[row_index, query_index]
                full_err = float(np.max(np.abs(fresh_scores - archived_scores)))
                if full_err > SCORE_TOLERANCE:
                    raise ValueError(f"Env 4 verification score error {full_err} > {SCORE_TOLERANCE}")
                score_count += len(ids)
                max_error = max(max_error, full_err)
                hit = next(r for r in ranking if r["Id"] == plan["source_memory_id"])
                archived_order = np.argsort(-archived_scores, kind="stable").tolist()
                archived_rank = archived_order.index(ids.index(plan["source_memory_id"])) + 1
                if (
                    actual_q["task_id"] != q["task_id"]
                    or actual_q["rank"] != hit["rank"]
                    or actual_q["rank"] != archived_rank
                    or not math.isfinite(actual_q["score"])
                ):
                    raise ValueError("Env 4 rank verification mismatch")
                err = abs(actual_q["score"] - hit["score"])
                if err > SCORE_TOLERANCE:
                    raise ValueError(f"Env 4 hit score mismatch: {err}")
                max_error = max(max_error, err)
                count += 1
    finally:
        encoder.close()

    result = dict(
        status="passed",
        environment="env4_feature_text_plus_target_image",
        group=plan["group"],
        rows=len(rows),
        rankings_recomputed=count,
        max_score_error=max_error,
        evaluated_file_sha256=sha256_file(output / "evaluation.json"),
        evaluation_plan_sha256=sha256_file(output / "evaluation_plan.json"),
        scores_sha256=sha256_file(output / "scores.npz"),
        candidate_scores_recomputed=score_count,
        score_tolerance=SCORE_TOLERANCE,
        all_candidate_scores_checked=True,
        isolated_process=True,
        fresh_encoder=True,
        recomputed_embeddings_and_rankings=True,
    )
    freeze_json(output / "independent_verification.json", result)
    return result


def write_env4_report(output: Path) -> dict:
    plan = read_json(output / "evaluation_plan.json")
    rows = read_json(output / "evaluation.json")
    verified = read_json(output / "independent_verification.json")
    if verified.get("status") != "passed" or not verified.get("all_candidate_scores_checked"):
        raise ValueError("Env 4 missing passed verification")
    group = plan["group"]
    lookup = {(r["condition"]["label"], r["bank"]): r for r in rows}
    clean_row = lookup[("none", "clean")]
    mod_row = lookup[("none", "modified")]
    delta = paired_rank_metrics([clean_row], [mod_row])
    c_ranks = [q["rank"] for q in clean_row["queries"]]
    m_ranks = [q["rank"] for q in mod_row["queries"]]
    c_scores = [q["score"] for q in clean_row["queries"]]
    m_scores = [q["score"] for q in mod_row["queries"]]
    c_top1 = sum(r <= 1 for r in c_ranks)
    m_top1 = sum(r <= 1 for r in m_ranks)
    c_top3 = sum(r <= 3 for r in c_ranks)
    m_top3 = sum(r <= 3 for r in m_ranks)
    c_top5 = sum(r <= 5 for r in c_ranks)
    m_top5 = sum(r <= 5 for r in m_ranks)
    n_q = len(c_ranks)
    c_mean = sum(c_ranks) / n_q
    m_mean = sum(m_ranks) / n_q
    c_score_mean = sum(c_scores) / n_q
    m_score_mean = sum(m_scores) / n_q
    lines = [
        "# 共享长期记忆检索评估（环境四：特征文字 + 目标图片）",
        "",
        f"组别：{group}；方法：{plan['method']}。共享池 {plan['pool_count']} 条（仅替换 {group} 源记忆 `{plan['source_memory_id']}`）；"
        "测试环境为无防御（`none`）+ 目标商品原图 `target.jpg` + 二十条冻结特征文字，图文各半融合。",
        "",
        "| 条件 | 原图 Top-1 | 修改图 Top-1 | 原图前三 | 修改图前三 | 原图前五 | 修改图前五 | 前三增量（百分点） | 平均名次（原图 -> 修改图） | 平均名次改善 | 名次区间（原图 -> 修改图） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        f"| 无防御 | {c_top1}/{n_q} | {m_top1}/{n_q} | {c_top3}/{n_q} | {m_top3}/{n_q} | {c_top5}/{n_q} | {m_top5}/{n_q} | {delta['top3_gain']*100:+.1f} | {c_mean:.2f} -> {m_mean:.2f} | {delta['mean_rank_gain']:+.3f} | [{min(c_ranks)},{max(c_ranks)}] -> [{min(m_ranks)},{max(m_ranks)}] |",
        "",
        f"平均余弦相似度分数：{c_score_mean:.4f} -> {m_score_mean:.4f}（{m_score_mean - c_score_mean:+.4f}）。",
        f"独立重新编码并复算 {verified['rankings_recomputed']} 次排名及 {verified['candidate_scores_recomputed']} 个候选分数，最大分数误差 {verified['max_score_error']}，容差 {SCORE_TOLERANCE}。",
        "",
    ]
    path = output / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "clean_top1": c_top1,
        "mod_top1": m_top1,
        "clean_top3": c_top3,
        "mod_top3": m_top3,
        "clean_top5": c_top5,
        "mod_top5": m_top5,
        "n_queries": n_q,
        "clean_mean_rank": c_mean,
        "mod_mean_rank": m_mean,
        "top3_gain_pp": delta["top3_gain"] * 100.0,
        "mean_rank_gain": delta["mean_rank_gain"],
        "clean_range": [min(c_ranks), max(c_ranks)],
        "mod_range": [min(m_ranks), max(m_ranks)],
        "clean_mean_score": c_score_mean,
        "mod_mean_score": m_score_mean,
        "score_delta": m_score_mean - c_score_mean,
        "clean_ranks": c_ranks,
        "mod_ranks": m_ranks,
    }


def run_env4(
    *,
    group: str,
    spec: dict,
    frozen_path: Path,
    output: Path,
    encoder_config: Path,
    vector_cache_dir: Path,
    device: str = "cuda",
) -> dict:
    if _is_verified_passed(output):
        print(f"SKIP verified {output}", flush=True)
        return write_env4_report(output)
    _clear_partial_dir(output)
    output.mkdir(parents=True, exist_ok=True)

    frozen = read_json(frozen_path)
    if frozen.get("status") != "frozen":
        raise ValueError(f"Frozen image not ready: {frozen_path}")
    validate_attack_completion(
        frozen["metadata"],
        frozen["requested_steps"],
        dict(
            method=frozen["method"],
            source_image_sha256=frozen["source"]["image_sha256"],
            target_image_sha256=frozen["target"]["image_sha256"],
        ),
    )
    if sha256_file(frozen["image"]) != frozen["image_sha256"]:
        raise ValueError("Frozen image sha256 mismatch")

    entries = load_shared_pool(SHARED_MEMORY)
    ids = [e["Id"] for e in entries]
    source_id = spec["source_memory_id"]
    banks_dict, chosen_id = evaluation_banks(
        entries,
        SHARED_MEMORY,
        frozen["source"],
        frozen["image"],
        source_memory_id=source_id,
    )
    if chosen_id != source_id:
        raise ValueError(f"Source memory id mismatch: {chosen_id} != {source_id}")

    env4_banks = output / "banks" / "none"
    env4_banks.mkdir(parents=True, exist_ok=True)
    for bname in ("clean", "modified"):
        freeze_json(env4_banks / f"{bname}.json", banks_dict[bname])

    text_manifest = read_json(EVALUATION_BASE / "evaluation_inputs" / group / "queries.json")
    target_img = (EVALUATION_BASE / "groups" / group / "target.jpg").resolve()
    target_sha = sha256_file(target_img)
    if target_sha != spec["target_sha256"]:
        raise ValueError(f"Target image digest mismatch for {group}")

    queries = [
        dict(
            task_id=f"feature_{i:02d}",
            instruction=txt,
            query_mode="joint",
            query_image=str(target_img),
            query_image_sha256=target_sha,
        )
        for i, txt in enumerate(text_manifest["instructions"], 1)
    ]

    cond = dict(label="none", kind="none", size=512)
    encoder = shared_evaluation.make_eval_encoder(device, encoder_config)
    rows, score_rows = [], []
    try:
        for bank_name in ("clean", "modified"):
            bank_file = env4_banks / f"{bank_name}.json"
            bank = read_json(bank_file)
            ranker = VisualMemoryRetriever(
                encoder,
                bank_file,
                key_mode="joint",
                query_mode="joint",
                index_text="auto",
                key_image_weight=0.5,
                query_image_weight=0.5,
                cache_dir=None,
                entry_cache_dir=vector_cache_dir,
                image_base_path=SHARED_MEMORY,
            )
            observations, candidate_scores = [], []
            for q in queries:
                ranking = ranker.rank(bank, q["instruction"], q["query_image"])
                if len(ranking) != len(entries):
                    raise ValueError("Env 4 ranking incomplete")
                chosen = next(r for r in ranking if r["Id"] == source_id)
                candidate_scores.append(_score_vector(ranking, ids))
                observations.append(dict(task_id=q["task_id"], rank=chosen["rank"], score=chosen["score"]))
            row = dict(
                condition=cond,
                bank=bank_name,
                bank_path=str(bank_file.resolve()),
                bank_sha256=sha256_file(bank_file),
                queries=observations,
                top3=sum(q["rank"] <= 3 for q in observations),
                top5=sum(q["rank"] <= 5 for q in observations),
                query_count=len(queries),
                score_archive_index=len(rows),
                retrieval_protocol=copy.deepcopy(ranker.snapshot),
            )
            rows.append(row)
            score_rows.append(candidate_scores)
            ranks_str = [q["rank"] for q in observations]
            print(
                f"{group} env4 none {bank_name}: Top-3={row['top3']}/{len(queries)}, "
                f"Top-5={row['top5']}/{len(queries)}, mean={sum(ranks_str)/len(ranks_str):.2f}, ranks={ranks_str}",
                flush=True,
            )
    finally:
        encoder.close()

    plan = dict(
        schema_version=1,
        scope="shared",
        group=group,
        diagnostic_environment="env4_feature_text_plus_target_image",
        query_image_is_target=True,
        pool_path=str(SHARED_MEMORY.resolve()),
        pool_sha256=sha256_file(SHARED_MEMORY),
        pool_count=len(entries),
        source_memory_id=source_id,
        frozen_image_sha256=frozen["image_sha256"],
        method=frozen["method"],
        queries=queries,
        conditions=[cond],
        encoder_config=str(encoder_config.resolve()),
        encoder_config_sha256=sha256_file(encoder_config),
        key_image_weight=0.5,
        query_image_weight=0.5,
        query_mode="joint",
        target_image_path=str(target_img),
        target_image_sha256=target_sha,
    )
    freeze_json(output / "evaluation_plan.json", plan)
    _save_scores(output, score_rows, ids, queries, rows, source_id)
    freeze_json(output / "evaluation.json", rows)

    proc = subprocess.run(
        [sys.executable, __file__, "--verify-env4", str(output), "--device", device],
        check=False,
        capture_output=True,
        text=True,
    )
    (output / "independent_verification.log").write_text(
        proc.stdout + proc.stderr, encoding="utf-8"
    )
    if proc.returncode != 0:
        save_json(output / "VERIFICATION_FAILED.json", dict(returncode=proc.returncode))
        raise RuntimeError(f"Env 4 verification failed for {output}:\n{proc.stdout}\n{proc.stderr}")
    return write_env4_report(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--groups", nargs="+", default=["06_jeans", "07_shampoo", "08_speaker"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--verify-env4", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_env4 is not None:
        print(verify_env4(args.verify_env4, device=args.device), flush=True)
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --verify-env4 is provided")

    out_root = args.output_root.resolve()
    vector_cache_dir = out_root / "memory_vector_cache"
    manifest = read_json(EVALUATION_BASE / "manifest.json")

    b32_config = out_root / "logs" / "eval_b32_matched.json"
    g14_config = EVALUATION_BASE / "evaluation_encoder.json"
    encoders = [("g14", g14_config), ("b32", b32_config)]

    summary_path = out_root / "logs" / "env4_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}

    for group in args.groups:
        source_id = manifest["shared_memory"]["source_mapping"][group]["source_memory_id"]
        group_spec = dict(manifest["groups"][group], source_memory_id=source_id)
        frozen_path = out_root / group / "ambilage" / "FROZEN_IMAGE.json"
        if not frozen_path.is_file():
            raise FileNotFoundError(f"Frozen image manifest not found for {group}: {frozen_path}")

        group_summary = {}
        for enc_key, enc_cfg in encoders:
            suite_name = f"{enc_key}_none_target_joint"
            eval_dir = out_root / group / "ambilage" / "evaluation" / suite_name
            print(f"=== [EVAL] {group}/ambilage/{suite_name} ===", flush=True)
            stats = run_env4(
                group=group,
                spec=group_spec,
                frozen_path=frozen_path,
                output=eval_dir,
                encoder_config=enc_cfg,
                vector_cache_dir=vector_cache_dir,
                device=args.device,
            )
            group_summary[enc_key] = stats
        summary[group] = group_summary
        save_json(summary_path, summary)

    print("ENV4 EVALUATION COMPLETED AND VERIFIED:", summary, flush=True)


if __name__ == "__main__":
    main()
