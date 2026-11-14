from __future__ import annotations

import json
from pathlib import Path

from src.experiments.defenses import conditions
from src.experiments.shared_evaluation import evaluate, report
from src.visual.audit import read_json

REPO_ROOT = Path("/content/AmbiLAGE")
EVALUATION_BASE = REPO_ROOT / "resources" / "evaluation_base"
OUT_ROOT = REPO_ROOT / "output" / "ambilage_choice1_2groups_20261114"

GROUPS = ["07_shampoo", "09_toothbrush"]
METHODS = ["ambilage"]
EVAL_SUITES = [
    ("g14_none_text", EVALUATION_BASE / "evaluation_encoder.json"),
    (
        "b32_none_text",
        Path(
            "/root/.gemini/antigravity/brain/e0388b53-0e9a-492d-97fa-c6472a1cb191/scratch/eval_b32_matched.json"
        ),
    ),
]


def main() -> None:
    manifest = read_json(EVALUATION_BASE / "manifest.json")
    pool_path = EVALUATION_BASE / manifest["shared_memory"]["pool_file"]
    source_mapping = manifest["shared_memory"]["source_mapping"]
    none_condition = conditions()[:1]

    summary = []
    for method in METHODS:
        replacement_frozens = []
        for group in GROUPS:
            frozen = dict(read_json(OUT_ROOT / group / method / "FROZEN_IMAGE.json"))
            frozen["source_memory_id"] = source_mapping[group]["source_memory_id"]
            replacement_frozens.append(frozen)

        bank_tag = f"{method}_2groups"
        for eval_name, encoder_config in EVAL_SUITES:
            cache_dir = OUT_ROOT / "memory_vectors" / eval_name
            bank_root = OUT_ROOT / "shared_banks" / bank_tag / eval_name
            for group, frozen in zip(GROUPS, replacement_frozens):
                eval_root = OUT_ROOT / group / method / "evaluation" / eval_name
                query_manifest = EVALUATION_BASE / manifest["groups"][group]["query_manifest"]
                result = evaluate(
                    frozen,
                    eval_root,
                    pool_path=pool_path,
                    query_manifest=query_manifest,
                    source_memory_id=frozen["source_memory_id"],
                    device="cuda",
                    selected_conditions=none_condition,
                    encoder_config=encoder_config,
                    query_mode="text",
                    replacement_frozens=replacement_frozens,
                    bank_root=bank_root,
                    memory_vector_cache_dir=cache_dir,
                )
                report_path = report(eval_root)
                summary.append(
                    {
                        "method": method,
                        "group": group,
                        "eval_name": eval_name,
                        "verification": result.get("verification"),
                        "report_path": str(report_path),
                    }
                )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
