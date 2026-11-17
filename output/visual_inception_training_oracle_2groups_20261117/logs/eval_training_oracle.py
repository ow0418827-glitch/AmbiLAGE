"""Evaluate 07_shampoo and 09_toothbrush (visual_inception) under training_oracle (50/30/20 pure text) with simultaneous 2-group shared-pool replacement."""
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
from src.paths import EVALUATION_BASE, MODEL_ROOT, SHARED_MEMORY  # noqa: E402
from src.visual.audit import read_json, save_json  # noqa: E402
from src.visual.encoders.clip import MatchedClipEncoder  # noqa: E402
from src.visual.encoders.joint import ChunkedClipEncoder  # noqa: E402
from src.visual.hardware import configure_tf32  # noqa: E402
from src.visual.image_utils import sha256_file  # noqa: E402

GROUPS = ["07_shampoo", "09_toothbrush"]
G14_CONFIG = EVALUATION_BASE / "evaluation_encoder.json"

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


def _is_verified_passed(eval_dir: Path) -> bool:
    iv = eval_dir / "independent_verification.json"
    rep = eval_dir / "REPORT.md"
    if not iv.is_file() or not rep.is_file():
        return False
    data = read_json(iv)
    return (
        data.get("status") == "passed"
        and data.get("all_candidate_scores_checked") is True
        and data.get("isolated_process") is True
        and not (eval_dir / "VERIFICATION_FAILED.json").exists()
    )


def summarize_eval(eval_dir: Path) -> dict:
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
    return {
        "clean_top1": sum(r <= 1 for r in c_ranks),
        "mod_top1": sum(r <= 1 for r in m_ranks),
        "clean_top3": sum(r <= 3 for r in c_ranks),
        "mod_top3": sum(r <= 3 for r in m_ranks),
        "clean_top5": sum(r <= 5 for r in c_ranks),
        "mod_top5": sum(r <= 5 for r in m_ranks),
        "n_queries": n_q,
        "clean_mean_rank": sum(c_ranks) / n_q,
        "mod_mean_rank": sum(m_ranks) / n_q,
        "top3_gain_pp": delta["top3_gain"] * 100.0,
        "mean_rank_gain": delta["mean_rank_gain"],
        "clean_range": [min(c_ranks), max(c_ranks)],
        "mod_range": [min(m_ranks), max(m_ranks)],
        "clean_mean_score": sum(c_scores) / n_q,
        "mod_mean_score": sum(m_scores) / n_q,
        "score_delta": (sum(m_scores) - sum(c_scores)) / n_q,
        "clean_ranks": c_ranks,
        "mod_ranks": m_ranks,
    }


def write_summary(out_root: Path, summary: dict) -> Path:
    lines = [
        "# Visual Inception 双组（洗发水 + 牙刷）`training_oracle` 论文抽样纯文字评测汇总",
        "",
        "- **评估环境**：`training_oracle`（每组 20 条纯文字查询：`50%` 公开描述/标题/卖点 + `30%` 特征改写 + `20%` 购买意向，无查询图）。",
        "- **共享池口径**：`189` 条完整共享记忆池（无防御 `none`），同时替换 `07_shampoo` 与 `09_toothbrush` 两条源参考图（其余 187 条保持原样）。",
        "- **复算校验**：全部套组均在独立子进程关闭继承缓存后重新编码全库并校验 `7,560` 个候选分数（`max_score_error <= 1e-7`）。",
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
            if enc_key not in summary.get(group, {}):
                continue
            s = summary[group][enc_key]
            n = s["n_queries"]
            lines.append(
                f"| `{group}` | `{enc_key}` | {roles[enc_key]} | "
                f"{s['clean_top1']}/{n} -> {s['mod_top1']}/{n} | "
                f"{s['clean_top3']}/{n} -> {s['mod_top3']}/{n} ({s['top3_gain_pp']:+.1f}pp) | "
                f"{s['clean_top5']}/{n} -> {s['mod_top5']}/{n} | "
                f"{s['clean_mean_rank']:.2f} -> {s['mod_mean_rank']:.2f} ({s['mean_rank_gain']:+.3f}) | "
                f"[{s['clean_range'][0]},{s['clean_range'][1]}] -> [{s['mod_range'][0]},{s['mod_range'][1]}] |"
            )
    lines.append("")
    md_path = out_root / "logs" / "SUMMARY_TRAINING_ORACLE.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


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
        frozen_path = out_root / group / "visual_inception" / "FROZEN_IMAGE.json"
        if not frozen_path.is_file():
            raise FileNotFoundError(f"Missing frozen image for {group}: {frozen_path}")
        item = read_json(frozen_path)
        item["source_memory_id"] = manifest["shared_memory"]["source_mapping"][group]["source_memory_id"]
        replacement_frozens.append(item)

    summary_path = out_root / "logs" / "training_oracle_summary.json"
    summary = read_json(summary_path) if summary_path.is_file() else {}
    none_condition = [c for c in conditions() if c["kind"] == "none"]

    for enc_key in ENCODER_ORDER:
        eval_label = f"{enc_key}_training_oracle_none"
        for group in GROUPS:
            eval_dir = out_root / group / "visual_inception" / "evaluation" / eval_label
            query_manifest = out_root / "logs" / "queries" / f"{group}_training_oracle.json"
            if _is_verified_passed(eval_dir):
                print(f"SKIP verified {group}/{eval_label}", flush=True)
            else:
                print(f"=== [EVAL] {group}/visual_inception/{eval_label} ===", flush=True)
                frozen = read_json(out_root / group / "visual_inception" / "FROZEN_IMAGE.json")
                source_memory_id = manifest["shared_memory"]["source_mapping"][group]["source_memory_id"]
                if enc_key == "eval_g14":
                    shared_evaluation.evaluate(
                        frozen,
                        eval_dir,
                        pool_path=pool_path,
                        query_manifest=query_manifest,
                        query_mode="text",
                        source_memory_id=source_memory_id,
                        device=args.device,
                        selected_conditions=none_condition,
                        encoder_config=G14_CONFIG,
                        replacement_frozens=replacement_frozens,
                        bank_root=out_root / "shared_banks" / "visual_inception" / eval_label,
                        memory_vector_cache_dir=out_root / "memory_vectors" / eval_label,
                    )
                else:
                    shared_evaluation.evaluate(
                        frozen,
                        eval_dir,
                        pool_path=pool_path,
                        query_manifest=query_manifest,
                        query_mode="text",
                        source_memory_id=source_memory_id,
                        device=args.device,
                        encoder_factory=lambda k=enc_key: make_custom_encoder(k, args.device),
                        selected_conditions=none_condition,
                        encoder_config=None,
                        replacement_frozens=replacement_frozens,
                        bank_root=out_root / "shared_banks" / "visual_inception" / eval_label,
                        memory_vector_cache_dir=out_root / "memory_vectors" / eval_label,
                    )
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
            stats = summarize_eval(eval_dir)
            summary.setdefault(group, {})[enc_key] = stats
            save_json(summary_path, summary)
            print(
                f"DONE {group}/{enc_key}: Clean Top-3={stats['clean_top3']}/{stats['n_queries']} "
                f"-> Mod Top-3={stats['mod_top3']}/{stats['n_queries']} "
                f"(mean {stats['clean_mean_rank']:.2f} -> {stats['mod_mean_rank']:.2f})",
                flush=True,
            )

    md_path = write_summary(out_root, summary)
    print(f"ALL EVALUATIONS VERIFIED. Summary written to {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
