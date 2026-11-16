### `LUCID` 四组 Env 4 评测总表（`none` 防御 `size=512`，汇总见 [env4_summary.json](file:///content/AmbiLAGE/output/lucid_4groups_20261116/logs/env4_summary.json)）

| 组别 | `ViT-g-14` `clean → mod` 均名（提升 `gain` / `Δscore`） | `ViT-g-14` `Top-1 / Top-3 / Top-5` | `ViT-B-32` `clean → mod` 均名（提升 `gain` / `Δscore`） | `ViT-B-32` `Top-1 / Top-3 / Top-5` | 结果来源 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`07_shampoo`** | `2.80 → 4.40`（`-1.60` / `-0.0172`） | `0/20` / `3/20` / `19/20` | `2.25 → 1.00`（`+1.25` / `+0.0990`） | `20/20` / `20/20` / `20/20` | 复用 `ambilage_choice1_2groups_20261114/07_shampoo/lucid_reuse`（`modified` 库同期替换 `07_shampoo` 与 `09_toothbrush`） |
| **`09_toothbrush`** | `5.10 → 5.40`（`-0.30` / `-0.0030`） | `0/20` / `0/20` / `16/20` | `4.05 → 2.15`（`+1.90` / `+0.0220`） | `8/20` / `20/20` / `20/20` | 复用 `ambilage_toothbrush_20261115/09_toothbrush/lucid_reuse`（严格单组替换） |
| **`06_jeans`** | `19.85 → 18.95`（`+0.90` / `+0.0084`） | `0/20` / `0/20` / `0/20` | `20.95 → 9.65`（`+11.30` / `+0.0547`） | `0/20` / `0/20` / `0/20` | 本轮新测（复用 `imported_lucid_baseline_20261113/groups/06_jeans/frozen.png`，严格单组替换） |
| **`08_speaker`** | `14.70 → 14.25`（`+0.45` / `+0.0098`） | `0/20` / `0/20` / `0/20` | `12.15 → 2.15`（`+10.00` / `+0.1037`） | `0/20` / `20/20` / `20/20` | 本轮新测（复用 `imported_lucid_baseline_20261113/groups/08_speaker/frozen.png`，严格单组替换） |
