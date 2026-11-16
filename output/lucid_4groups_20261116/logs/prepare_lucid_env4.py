"""Prepare LUCID 4-group Env 4 workspace reusing verified 07_shampoo & 09_toothbrush runs."""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path("/content/AmbiLAGE")
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.paired_metrics import paired_rank_metrics  # noqa: E402
from src.visual.attack_completion import validate_attack_completion  # noqa: E402
from src.visual.audit import read_json, save_json  # noqa: E402
from src.visual.image_utils import sha256_file  # noqa: E402

LUCID_ROOT = REPO_ROOT / "output" / "lucid_4groups_20261116"
IMPORT_MANIFEST = REPO_ROOT / "output" / "imported_lucid_baseline_20261113" / "IMPORT_MANIFEST.json"

REUSE_SOURCES = {
    "07_shampoo": {
        "eval_root": REPO_ROOT / "output" / "ambilage_choice1_2groups_20261114" / "07_shampoo" / "lucid_reuse" / "evaluation",
        "frozen_json": REPO_ROOT / "output" / "ambilage_2groups_20261114" / "07_shampoo" / "lucid" / "FROZEN_IMAGE.json",
    },
    "09_toothbrush": {
        "eval_root": REPO_ROOT / "output" / "ambilage_toothbrush_20261115" / "09_toothbrush" / "lucid_reuse" / "evaluation",
        "frozen_json": REPO_ROOT / "output" / "ambilage_2groups_20261114" / "09_toothbrush" / "lucid" / "FROZEN_IMAGE.json",
    },
}


def compute_env4_metrics(eval_dir: Path) -> dict:
    iv = read_json(eval_dir / "independent_verification.json")
    if iv.get("status") != "passed" or iv.get("all_candidate_scores_checked") is not True:
        raise RuntimeError(f"Reused Env 4 directory did not pass verification: {eval_dir}")
    rows = read_json(eval_dir / "evaluation.json")
    lookup = {(r["condition"]["label"], r["bank"]): r for r in rows}
    clean_row = lookup[("none", "clean")]
    mod_row = lookup[("none", "modified")]
    delta = paired_rank_metrics([clean_row], [mod_row])
    c_ranks = [q["rank"] for q in clean_row["queries"]]
    m_ranks = [q["rank"] for q in mod_row["queries"]]
    c_scores = [q["score"] for q in clean_row["queries"]]
    m_scores = [q["score"] for q in mod_row["queries"]]
    n_q = len(c_ranks)
    c_mean = sum(c_ranks) / n_q
    m_mean = sum(m_ranks) / n_q
    c_score_mean = sum(c_scores) / n_q
    m_score_mean = sum(m_scores) / n_q
    return {
        "clean_top1": sum(r <= 1 for r in c_ranks),
        "mod_top1": sum(r <= 1 for r in m_ranks),
        "clean_top3": sum(r <= 3 for r in c_ranks),
        "mod_top3": sum(r <= 3 for r in m_ranks),
        "clean_top5": sum(r <= 5 for r in c_ranks),
        "mod_top5": sum(r <= 5 for r in m_ranks),
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


def prepare_frozen_from_import(group: str, manifest: dict) -> Path:
    info = manifest["groups"][group]
    orig_frozen_path = (REPO_ROOT / info["original_frozen"]["path"]).resolve()
    orig_meta_path = (REPO_ROOT / info["metadata"]["path"]).resolve()
    local_img = (REPO_ROOT / info["image"]["path"]).resolve()
    orig = read_json(orig_frozen_path)

    if sha256_file(local_img) != orig["image_sha256"]:
        raise RuntimeError(f"Imported frozen PNG digest mismatch for {group}: {local_img}")

    src_img = (REPO_ROOT / "resources" / "evaluation_base" / "groups" / group / "source.jpg").resolve()
    tgt_img = (REPO_ROOT / "resources" / "evaluation_base" / "groups" / group / "target.jpg").resolve()
    if sha256_file(src_img) != orig["source"]["image_sha256"]:
        raise RuntimeError(f"Source image digest mismatch for {group}")
    if sha256_file(tgt_img) != orig["target"]["image_sha256"]:
        raise RuntimeError(f"Target image digest mismatch for {group}")

    wrapped = copy.deepcopy(orig)
    wrapped["image"] = str(local_img)
    wrapped["source"]["image"] = str(src_img)
    wrapped["target"]["image"] = str(tgt_img)
    wrapped["metadata"]["output_image"] = str(local_img)
    wrapped["metadata"]["output_metadata"] = str(orig_meta_path)
    wrapped["historical_reuse"] = {
        "kind": "historical_completed_baseline_reuse",
        "generation_executed": False,
        "original_evidence_kept_unchanged": True,
        "import_manifest_sha256": sha256_file(IMPORT_MANIFEST),
        "original_frozen_sha256": sha256_file(orig_frozen_path),
        "original_metadata_sha256": sha256_file(orig_meta_path),
    }

    validate_attack_completion(
        wrapped["metadata"],
        wrapped["requested_steps"],
        {
            "method": wrapped["method"],
            "source_image_sha256": wrapped["source"]["image_sha256"],
            "target_image_sha256": wrapped["target"]["image_sha256"],
        },
    )

    out_path = LUCID_ROOT / group / "lucid" / "FROZEN_IMAGE.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, wrapped)
    return out_path


def main() -> None:
    manifest = read_json(IMPORT_MANIFEST)
    summary_path = LUCID_ROOT / "logs" / "env4_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}

    for group, src in REUSE_SOURCES.items():
        dst_group_dir = LUCID_ROOT / group / "lucid"
        dst_eval_root = dst_group_dir / "evaluation"
        dst_eval_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src["frozen_json"], dst_group_dir / "FROZEN_IMAGE.json")
        group_summary = {}
        for enc_key in ("g14", "b32"):
            suite = f"{enc_key}_none_target_joint"
            src_suite = src["eval_root"] / suite
            dst_suite = dst_eval_root / suite
            if dst_suite.exists():
                shutil.rmtree(dst_suite)
            shutil.copytree(src_suite, dst_suite)
            group_summary[enc_key] = compute_env4_metrics(dst_suite)
        summary[group] = group_summary
        print(f"[REUSED] {group}: g14 mod_mean={group_summary['g14']['mod_mean_rank']:.2f}, b32 mod_mean={group_summary['b32']['mod_mean_rank']:.2f}")

    # Cross-check reused 09_toothbrush numbers against ambilage_toothbrush_20261115 summary
    tb_ref = read_json(REPO_ROOT / "output" / "ambilage_toothbrush_20261115" / "logs" / "env4_summary.json")["lucid_reuse"]
    for enc_key in ("g14", "b32"):
        for k in ("clean_top1", "mod_top1", "clean_top3", "mod_top3", "clean_top5", "mod_top5", "clean_mean_rank", "mod_mean_rank"):
            assert summary["09_toothbrush"][enc_key][k] == tb_ref[enc_key][k], f"Mismatch on 09_toothbrush {enc_key} {k}"

    for group in ("06_jeans", "08_speaker"):
        frozen_out = prepare_frozen_from_import(group, manifest)
        print(f"[PREPARED] {group}: {frozen_out}")

    save_json(summary_path, summary)
    print(f"Saved initial reused summary to {summary_path}")


if __name__ == "__main__":
    main()
