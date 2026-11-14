# 本地会话记录归档 (`docs/sessions`)

本目录保存了 2026-11-14 在实验服务器完成 `AmbiLAGE` 部署、双组生成评测与 `perLage` 对比分析期间的完整会话文档（已做访问令牌脱敏处理）：

| 文件 | 对应会话 | 会话 ID | 核心内容 |
|---|---|---|---|
| [01_ambilage_repository_setup_and_run.md](./01_ambilage_repository_setup_and_run.md) | AmbiLAGE Repository Setup And Experiment Run | `14100399-1521-4ccd-b4b4-118e88220e10` | 克隆仓库、搭建环境、直连下载全部模型与锁校验、修复 SAM2 坐标与 FLUX 细长图对齐、`lucid` 基线与 `AmbiLAGE` 双组生成及 `ViT-B-32`/`ViT-g-14` 无防御检索评测 |
| [02_perlage_and_ambilage_method_comparison.md](./02_perlage_and_ambilage_method_comparison.md) | PerLage And Ambilage Method Comparison | `0a8fbf4c-83b1-45fd-9983-8d890beab73d` | 克隆并对比 `perLage`（含 `_perlage_ambient_coop`）与 `AmbiLAGE` 代码差异，定位背景光斑与洗发水文字矩形框成因，对比跨模型检索成绩并给出综合改进思路 |

> 注：按用户明确指示，另一本地会话 `实验进展`（`363c8776-862b-4ee3-a3b6-60db4e2e9f8e`）未做保存与推送。
