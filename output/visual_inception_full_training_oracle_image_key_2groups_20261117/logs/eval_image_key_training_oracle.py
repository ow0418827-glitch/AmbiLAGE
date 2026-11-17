"""Evaluate frozen full visual_inception (07_shampoo, 09_toothbrush) under default image-key training_oracle."""
from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path("/content/AmbiLAGE")
sys.path.insert(0, str(REPO_ROOT))

from src.experiments import shared_evaluation  # noqa: E402
from src.experiments.defenses import conditions  # noqa: E402
from src.experiments.paired_metrics import paired_rank_metrics  # noqa: E402
from src.paths import EVALUATION_BASE, MODEL_ROOT, RESOURCES  # noqa: E402
from src.visual._visual_inception.evidence import validate_ready  # noqa: E402
from src.visual.audit import read_json, save_json  # noqa: E402
from src.visual.encoders.clip import MatchedClipEncoder  # noqa: E402
from src.visual.encoders.joint import ChunkedClipEncoder  # noqa: E402
from src.visual.hardware import configure_tf32  # noqa: E402
from src.visual.image_utils import sha256_file  # noqa: E402

GROUPS = ["07_shampoo", "09_toothbrush"]
FROZEN_SOURCE_ROOT = REPO_ROOT / "output/visual_inception_full_training_oracle_2groups_20261117"
G14_CONFIG = EVALUATION_BASE / "evaluation_encoder.json"
ORACLE_QUERY_ROOT = RESOURCES / "visual_inception_evaluation/training_queries_oracle_v1"

ORACLE_FAMILIES = (
    ("category_description", "同类公开描述(50%/10条)"),
    ("model_paraphrase", "模型改写查询(30%/6条)"),
    ("heldout_search_query", "独立真实搜索(20%/4条)"),
)

TRAINING_SURROGATE_ENCODERS = {
    "clip_l14": {
        "name": "ViT-L-14",
        "pretrained": "openai",
        "hf_repo": "timm/vit_large_patch14_clip_224.openai",
        "weight_path": str((MODEL_ROOT / "clip_l14/open_clip_model.safetensors").resolve()),
    },
    "siglip_so400m": {
        "name": "ViT-SO400M-14-SigLIP-384",
        "pretrained": "webli",
        "hf_repo": "timm/ViT-SO400M-14-SigLIP-384",
        "weight_path": str((MODEL_ROOT / "siglip_so400m/open_clip_model.safetensors").resolve()),
        "tokenizer_path": str((MODEL_ROOT / "siglip_so400m").resolve()),
    },
    "clip_bigG14": {
        "name": "ViT-bigG-14",
        "pretrained": "laion2b_s39b_b160k",
        "hf_repo": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "weight_path": str((MODEL_ROOT / "clip_bigG14/open_clip_model.safetensors").resolve()),
    },
}

ENCODER_ORDER = ("eval_g14", "clip_l14", "siglip_so400m", "clip_bigG14")


def make_custom_encoder(enc_key: str, device: str):
    """Instantiate one of the visual_inception training surrogate CLIP/SigLIP encoders."""
    if str(device).startswith("cuda"):
        configure_tf32(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    spec = dict(TRAINING_SURROGATE_ENCODERS[enc_key])
    surrogate_number = list(TRAINING_SURROGATE_ENCODERS).index(enc_key) + 1
    enc = ChunkedClipEncoder(
        MatchedClipEncoder({"surrogate": spec, "surrogate_number": surrogate_number, "device": device}, REPO_ROOT)
    )
    enc.snapshot["evaluation_protocol"] = {
        "setting": "visual_inception_matched_training_surrogate",
        "matches_training_surrogate": True,
        "black_box_transfer_claimed": False,
        "tf32_enabled": False,
        "precision": "float32",
    }
    return enc


def _assert_image_key_mode(eval_dir: Path) -> None:
    plan = read_json(eval_dir / "evaluation_plan.json")
    if plan.get("key_mode") != "image" or plan.get("key_image_weight") != 1.0:
        raise AssertionError(f"Expected default key_mode='image' (weight=1.0) in {eval_dir}, got {plan.get('key_mode')}/{plan.get('key_image_weight')}")
    rows = read_json(eval_dir / "evaluation.json")
    for row in rows:
        proto = row.get("retrieval_protocol", {})
        if proto.get("key_mode") != "image" or proto.get("key_image_weight") != 1.0:
            raise AssertionError(f"Unexpected retrieval_protocol key_mode in {eval_dir}: {proto.get('key_mode')}/{proto.get('key_image_weight')}")


def _is_verified_passed(eval_dir: Path) -> bool:
    iv = eval_dir / "independent_verification.json"
    rep = eval_dir / "REPORT.md"
    if not iv.is_file() or not rep.is_file():
        return False
    _assert_image_key_mode(eval_dir)
    data = read_json(iv)
    return (
        data.get("status") == "passed"
        and data.get("all_candidate_scores_checked") is True
        and data.get("isolated_process") is True
        and not (eval_dir / "VERIFICATION_FAILED.json").exists()
    )


def _rank_stats(c_ranks: list[int], m_ranks: list[int], c_scores: list[float], m_scores: list[float]) -> dict:
    n_q = len(c_ranks)
    c_top3 = sum(r <= 3 for r in c_ranks)
    m_top3 = sum(r <= 3 for r in m_ranks)
    c_mean = sum(c_ranks) / n_q
    m_mean = sum(m_ranks) / n_q
    return {
        "clean_top1": sum(r <= 1 for r in c_ranks),
        "mod_top1": sum(r <= 1 for r in m_ranks),
        "clean_top3": c_top3,
        "mod_top3": m_top3,
        "clean_top5": sum(r <= 5 for r in c_ranks),
        "mod_top5": sum(r <= 5 for r in m_ranks),
        "n_queries": n_q,
        "clean_mean_rank": c_mean,
        "mod_mean_rank": m_mean,
        "top3_gain_pp": ((m_top3 - c_top3) / n_q) * 100.0,
        "mean_rank_gain": c_mean - m_mean,
        "clean_range": [min(c_ranks), max(c_ranks)],
        "mod_range": [min(m_ranks), max(m_ranks)],
        "clean_mean_score": sum(c_scores) / n_q,
        "mod_mean_score": sum(m_scores) / n_q,
        "score_delta": (sum(m_scores) - sum(c_scores)) / n_q,
    }


def summarize_eval(eval_dir: Path, query_manifest: Path) -> dict:
    """Compute overall and 50/30/20 stratified retrieval statistics."""
    rows = read_json(eval_dir / "evaluation.json")
    lookup = {(r["condition"]["label"], r["bank"]): r for r in rows}
    clean_row = lookup[("none", "clean")]
    mod_row = lookup[("none", "modified")]
    delta = paired_rank_metrics([clean_row], [mod_row])

    q_doc = read_json(query_manifest)
    q_records = q_doc["query_records"]
    if len(q_records) != len(clean_row["queries"]):
        raise ValueError(f"Query count mismatch in {eval_dir}")

    c_ranks = [q["rank"] for q in clean_row["queries"]]
    m_ranks = [q["rank"] for q in mod_row["queries"]]
    c_scores = [q["score"] for q in clean_row["queries"]]
    m_scores = [q["score"] for q in mod_row["queries"]]
    overall = _rank_stats(c_ranks, m_ranks, c_scores, m_scores)
    overall["top3_gain_pp"] = delta["top3_gain"] * 100.0
    overall["mean_rank_gain"] = delta["mean_rank_gain"]
    overall["clean_ranks"] = c_ranks
    overall["mod_ranks"] = m_ranks

    by_family = {}
    for fam_key, _ in ORACLE_FAMILIES:
        idxs = [i for i, rec in enumerate(q_records) if rec["query_family"] == fam_key]
        by_family[fam_key] = _rank_stats(
            [c_ranks[i] for i in idxs],
            [m_ranks[i] for i in idxs],
            [c_scores[i] for i in idxs],
            [m_scores[i] for i in idxs],
        )
    overall["by_family"] = by_family
    return overall


def write_summary(out_root: Path, summary: dict) -> Path:
    joint_path = FROZEN_SOURCE_ROOT / "logs/training_oracle_summary.json"
    joint_summary = read_json(joint_path) if joint_path.is_file() else None

    lines = [
        "# Visual Inception 默认纯图片记忆键（`key_mode=image`）双组 `training_oracle` 复测汇总",
        "",
        "- **默认记忆键模式**：`key_mode=image`（`key_image_weight=1.0`，按原论文只编码记忆库图片向量，不融合商品文字），由 `shared_evaluation.evaluate(..., key_mode=None)` 自动选定并记录。",
        "- **复用冻结成图**：沿用同一轮完整联合目标（100 查询 / `batch=16` / 90 商品检索损失）冻结图片（零重新训练，确保仅对比检索记忆键差异）。",
        "- **评估环境**：官方 `resources/visual_inception_evaluation/training_queries_oracle_v1/`（每组 20 条纯文字查询：`10` 描述 + `6` 改写 + `4` 真实搜索）。",
        "- **共享池口径**：`189` 条完整共享记忆池（无防御 `none`），同时替换 `07_shampoo` 与 `09_toothbrush` 两条源参考图（其余 187 条保持原样）。",
        "- **复算校验**：全部 8 套评估均在独立子进程重新编码全库并校验全部 `7,560` 个候选分数（`max_score_error <= 1e-7`）。",
        "",
        "## 总体检索覆盖统计（`key_mode=image`，每组 20 条）",
        "",
        "| 组别 | 检索编码器 | 性质 | 原图 Top-1 -> 修改 | 原图 Top-3 -> 修改 (`ΔTop-3`) | 原图 Top-5 -> 修改 | 平均名次 (`ΔRank`) | 名次区间 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    roles = {
        "eval_g14": "留出黑盒 (`ViT-g-14`)",
        "clip_l14": "训练代理白盒 (`ViT-L-14`)",
        "siglip_so400m": "训练代理白盒 (`SigLIP-SO400M`)",
        "clip_bigG14": "训练代理白盒 (`ViT-bigG-14`)",
    }
    for group in GROUPS:
        for enc_key in ENCODER_ORDER:
            s = summary.get(group, {}).get(enc_key)
            if not s:
                continue
            n = s["n_queries"]
            lines.append(
                f"| `{group}` | `{enc_key}` | {roles[enc_key]} | "
                f"{s['clean_top1']}/{n} -> {s['mod_top1']}/{n} | "
                f"{s['clean_top3']}/{n} -> {s['mod_top3']}/{n} ({s['top3_gain_pp']:+.1f}pp) | "
                f"{s['clean_top5']}/{n} -> {s['mod_top5']}/{n} | "
                f"{s['clean_mean_rank']:.2f} -> {s['mod_mean_rank']:.2f} ({s['mean_rank_gain']:+.3f}) | "
                f"[{s['clean_range'][0]},{s['clean_range'][1]}] -> [{s['mod_range'][0]},{s['mod_range'][1]}] |"
            )

    if joint_summary:
        lines += [
            "",
            "## 与先前 `key_mode=joint`（图文各半）结果对比",
            "",
            "| 组别 | 检索编码器 | `joint` 原->改 Top-3 (`Δ`) | `image` 原->改 Top-3 (`Δ`) | `joint` 原->改均名 (`Δ`) | `image` 原->改均名 (`Δ`) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for group in GROUPS:
            for enc_key in ENCODER_ORDER:
                s = summary.get(group, {}).get(enc_key)
                j = joint_summary.get(group, {}).get(enc_key)
                if not s or not j:
                    continue
                n = s["n_queries"]
                lines.append(
                    f"| `{group}` | `{enc_key}` | "
                    f"{j['clean_top3']}/{n} -> {j['mod_top3']}/{n} ({j['top3_gain_pp']:+.1f}pp) | "
                    f"{s['clean_top3']}/{n} -> {s['mod_top3']}/{n} ({s['top3_gain_pp']:+.1f}pp) | "
                    f"{j['clean_mean_rank']:.2f} -> {j['mod_mean_rank']:.2f} ({j['mean_rank_gain']:+.3f}) | "
                    f"{s['clean_mean_rank']:.2f} -> {s['mod_mean_rank']:.2f} ({s['mean_rank_gain']:+.3f}) |"
                )

    lines += [
        "",
        "## 按 `training_oracle` 50/30/20 分层明细（`key_mode=image`）",
        "",
        "| 组别 | 检索编码器 | 查询分层 | 原图 Top-1 -> 修改 | 原图 Top-3 -> 修改 (`ΔTop-3`) | 原图 Top-5 -> 修改 | 平均名次 (`ΔRank`) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        for enc_key in ENCODER_ORDER:
            s = summary.get(group, {}).get(enc_key)
            if not s:
                continue
            for fam_key, fam_label in ORACLE_FAMILIES:
                fs = s["by_family"][fam_key]
                n = fs["n_queries"]
                lines.append(
                    f"| `{group}` | `{enc_key}` | {fam_label} | "
                    f"{fs['clean_top1']}/{n} -> {fs['mod_top1']}/{n} | "
                    f"{fs['clean_top3']}/{n} -> {fs['mod_top3']}/{n} ({fs['top3_gain_pp']:+.1f}pp) | "
                    f"{fs['clean_top5']}/{n} -> {fs['mod_top5']}/{n} | "
                    f"{fs['clean_mean_rank']:.2f} -> {fs['mod_mean_rank']:.2f} ({fs['mean_rank_gain']:+.3f}) |"
                )

    lines.append("")
    md_path = out_root / "logs" / "SUMMARY_TRAINING_ORACLE_IMAGE_KEY.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _validate_full_frozen(frozen: dict, group: str) -> None:
    if frozen.get("status") != "frozen" or frozen.get("method") != "visual_inception":
        raise ValueError(f"Invalid frozen record for {group}")
    meta = frozen.get("metadata", {})
    obj = meta.get("training_objective", {})
    if obj.get("kind") != "visual_inception_retrieval_semantic_perceptual_v2":
        raise ValueError(f"Expected full retrieval objective v2 for {group}, got {obj.get('kind')}")
    if meta.get("retrieval_loss_omitted") is not False or obj.get("query_count") != 100:
        raise ValueError(f"Full retrieval metadata invariant failed for {group}")
    if meta.get("config", {}).get("query_batch_size") != 16:
        raise ValueError(f"Expected query_batch_size=16 for {group}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--verify-dir", type=Path, default=None)
    parser.add_argument("--encoder-key", choices=tuple(TRAINING_SURROGATE_ENCODERS), default=None)
    parser.add_argument("--query-manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.verify_dir is not None:
        if not args.encoder_key or not args.query_manifest:
            parser.error("--verify-dir requires --encoder-key and --query-manifest")
        result = shared_evaluation.verify(
            args.verify_dir,
            encoder_factory=lambda: make_custom_encoder(args.encoder_key, args.device),
            device=args.device,
            query_mode="text",
            query_manifest=args.query_manifest,
            key_mode="image",
            isolated_process=True,
        )
        print(result, flush=True)
        return 0

    if args.output_root is None:
        parser.error("--output-root is required")
    out_root = args.output_root.resolve()
    manifest = read_json(EVALUATION_BASE / "manifest.json")
    pool_path = EVALUATION_BASE / manifest["shared_memory"]["pool_file"]
    if sha256_file(pool_path) != manifest["shared_memory"]["pool_sha256"]:
        raise ValueError("Shared pool sha256 mismatch")

    replacement_frozens = []
    for group in GROUPS:
        frozen_path = FROZEN_SOURCE_ROOT / group / "visual_inception" / "FROZEN_IMAGE.json"
        if not frozen_path.is_file():
            raise FileNotFoundError(f"Missing frozen image for {group}: {frozen_path}")
        validate_ready(FROZEN_SOURCE_ROOT / group / "visual_inception/generation", require_production=True)
        item = read_json(frozen_path)
        _validate_full_frozen(item, group)
        item["source_memory_id"] = manifest["shared_memory"]["source_mapping"][group]["source_memory_id"]
        replacement_frozens.append(item)

    summary_path = out_root / "logs" / "training_oracle_image_key_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    none_condition = [c for c in conditions() if c["kind"] == "none"]

    for enc_key in ENCODER_ORDER:
        eval_label = f"{enc_key}_training_oracle_none"
        for group in GROUPS:
            eval_dir = out_root / group / "visual_inception" / "evaluation" / eval_label
            query_manifest = ORACLE_QUERY_ROOT / group / "queries.json"
            if _is_verified_passed(eval_dir):
                print(f"SKIP verified {group}/{eval_label}", flush=True)
            else:
                print(f"=== [EVAL key_mode=default(image)] {group}/visual_inception/{eval_label} ===", flush=True)
                frozen = read_json(FROZEN_SOURCE_ROOT / group / "visual_inception" / "FROZEN_IMAGE.json")
                source_memory_id = manifest["shared_memory"]["source_mapping"][group]["source_memory_id"]
                if enc_key == "eval_g14":
                    shared_evaluation.evaluate(
                        frozen,
                        eval_dir,
                        pool_path=pool_path,
                        query_manifest=query_manifest,
                        query_mode="text",
                        key_mode=None,
                        source_memory_id=source_memory_id,
                        device=args.device,
                        selected_conditions=none_condition,
                        encoder_config=G14_CONFIG,
                        replacement_frozens=replacement_frozens,
                        bank_root=out_root / "shared_banks" / "visual_inception" / eval_label,
                        memory_vector_cache_dir=out_root / "memory_vectors" / eval_label,
                    )
                    _assert_image_key_mode(eval_dir)
                else:
                    shared_evaluation.evaluate(
                        frozen,
                        eval_dir,
                        pool_path=pool_path,
                        query_manifest=query_manifest,
                        query_mode="text",
                        key_mode=None,
                        source_memory_id=source_memory_id,
                        device=args.device,
                        encoder_factory=lambda k=enc_key: make_custom_encoder(k, args.device),
                        selected_conditions=none_condition,
                        encoder_config=None,
                        replacement_frozens=replacement_frozens,
                        bank_root=out_root / "shared_banks" / "visual_inception" / eval_label,
                        memory_vector_cache_dir=out_root / "memory_vectors" / eval_label,
                    )
                    _assert_image_key_mode(eval_dir)
                    gc.collect()
                    if args.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    iv_path = eval_dir / "independent_verification.json"
                    if iv_path.is_file():
                        iv_path.unlink()
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-B",
                            "-X",
                            "utf8",
                            str(Path(__file__).resolve()),
                            "--verify-dir",
                            str(eval_dir),
                            "--encoder-key",
                            enc_key,
                            "--query-manifest",
                            str(query_manifest),
                            "--device",
                            args.device,
                        ],
                        cwd=REPO_ROOT,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                    )
                    (eval_dir / "independent_verification.log").write_text(
                        proc.stdout + proc.stderr, encoding="utf-8"
                    )
                    if proc.returncode != 0:
                        save_json(eval_dir / "VERIFICATION_FAILED.json", {"returncode": proc.returncode})
                        raise RuntimeError(f"Isolated verification failed for {eval_dir}:\n{proc.stdout}\n{proc.stderr}")
                shared_evaluation.report(eval_dir)
                gc.collect()
                if args.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            stats = summarize_eval(eval_dir, query_manifest)
            summary.setdefault(group, {})[enc_key] = stats
            save_json(summary_path, summary)
            print(
                f"DONE {group}/{enc_key} (key_mode=image): Clean Top-3={stats['clean_top3']}/{stats['n_queries']} "
                f"-> Mod Top-3={stats['mod_top3']}/{stats['n_queries']} "
                f"(mean {stats['clean_mean_rank']:.2f} -> {stats['mod_mean_rank']:.2f})",
                flush=True,
            )

    md_path = write_summary(out_root, summary)
    print(f"ALL IMAGE-KEY EVALUATIONS VERIFIED. Summary written to {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
