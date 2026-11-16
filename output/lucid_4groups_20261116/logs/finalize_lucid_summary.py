"""Finalize LUCID 4-group Env 4 env4_summary.json and markdown summary matching the Quote format."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path("/content/AmbiLAGE")
LUCID_ROOT = REPO_ROOT / "output" / "lucid_4groups_20261116"
LOGS_DIR = LUCID_ROOT / "logs"
GROUPS = ["07_shampoo", "09_toothbrush", "06_jeans", "08_speaker"]

PROVENANCE = {
    "07_shampoo": "复用 `ambilage_choice1_2groups_20261114/07_shampoo/lucid_reuse`（`modified` 库同期替换 `07_shampoo` 与 `09_toothbrush`）",
    "09_toothbrush": "复用 `ambilage_toothbrush_20261115/09_toothbrush/lucid_reuse`（严格单组替换）",
    "06_jeans": "本轮新测（复用 `imported_lucid_baseline_20261113/groups/06_jeans/frozen.png`，严格单组替换）",
    "08_speaker": "本轮新测（复用 `imported_lucid_baseline_20261113/groups/08_speaker/frozen.png`，严格单组替换）",
}


def main() -> None:
    summary_path = LOGS_DIR / "env4_summary.json"
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    ordered = {}
    for g in GROUPS:
        if g not in raw:
            raise KeyError(f"Missing group in env4_summary.json: {g}")
        for enc in ("g14", "b32"):
            iv_path = LUCID_ROOT / g / "lucid" / "evaluation" / f"{enc}_none_target_joint" / "independent_verification.json"
            iv = json.loads(iv_path.read_text(encoding="utf-8"))
            if iv.get("status") != "passed" or iv.get("all_candidate_scores_checked") is not True:
                raise RuntimeError(f"Verification not passed for {g}/{enc}: {iv_path}")
        ordered[g] = raw[g]
    summary_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "### `LUCID` 四组 Env 4 评测总表（`none` 防御 `size=512`，汇总见 [env4_summary.json](file:///content/AmbiLAGE/output/lucid_4groups_20261116/logs/env4_summary.json)）",
        "",
        "| 组别 | `ViT-g-14` `clean → mod` 均名（提升 `gain` / `Δscore`） | `ViT-g-14` `Top-1 / Top-3 / Top-5` | `ViT-B-32` `clean → mod` 均名（提升 `gain` / `Δscore`） | `ViT-B-32` `Top-1 / Top-3 / Top-5` | 结果来源 |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for g in GROUPS:
        g14 = ordered[g]["g14"]
        b32 = ordered[g]["b32"]
        lines.append(
            f"| **`{g}`** | "
            f"`{g14['clean_mean_rank']:.2f} → {g14['mod_mean_rank']:.2f}`（`{g14['mean_rank_gain']:+.2f}` / `{g14['score_delta']:+.4f}`） | "
            f"`{g14['mod_top1']}/{g14['n_queries']}` / `{g14['mod_top3']}/{g14['n_queries']}` / `{g14['mod_top5']}/{g14['n_queries']}` | "
            f"`{b32['clean_mean_rank']:.2f} → {b32['mod_mean_rank']:.2f}`（`{b32['mean_rank_gain']:+.2f}` / `{b32['score_delta']:+.4f}`） | "
            f"`{b32['mod_top1']}/{b32['n_queries']}` / `{b32['mod_top3']}/{b32['n_queries']}` / `{b32['mod_top5']}/{b32['n_queries']}` | "
            f"{PROVENANCE[g]} |"
        )
    lines.append("")
    md_path = LOGS_DIR / "SUMMARY_ENV4.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path} and verified {summary_path}")


if __name__ == "__main__":
    main()
