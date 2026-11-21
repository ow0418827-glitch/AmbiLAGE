"""MiniMax M3 下游多模态商品推荐评测 CLI（零 GPU / 免检索编码器加载）。

支持两种零 GPU 上游 Top-5 记忆获取模式：
  1. 预计算向量直算模式（推荐）：通过 `--spec TAG:METHOD:IMAGES_ROOT` 指定待测方法成图目录，
     直接从 `output/cache/precomputed_eval_vectors` 读取三编码器预计算向量，在纯 CPU 上毫秒级完成
     189 条记忆库检索排序与 Top-5 提取，无需加载任何 CLIP/OpenCLIP/SigLIP 模型、零 GPU 占用。
  2. 冻结评估目录复用模式：通过 `--eval-root output/exp_20261121_eval_v4_cleaned/eval` 与 `--tags ambilage vi_08_02`
     直接读取已归档的 `scores.npz` + `modified.json` 复原 Top-5 记忆。

下游评测流程：
  - 对 Top-5 记忆做严格白名单脱敏（剔除 Id / EntryType / FinalASIN / ImageSubstitution 等内部审计字段）
  - 组装 5 组“记忆文本块 + Base64 图像块”交替多模态消息
  - 单轮调用 MiniMax M3（开启 `reasoning_split=True`，`temperature=0.0`，支持 JSONL 断点续跑）
  - 离线在 CPU 上同步结算原论文口径 GHR（SBERT `all-MiniLM-L6-v2`，τ = 0.70 / 0.75 / 0.80）
    与目标商品推荐成功率（严格精确匹配 + 简写容错匹配）及条件转化率 P(推荐成功 | 目标进 Top-5)。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Enforce strict zero-GPU isolation before importing torch / sentence_transformers / src.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.downstream_recommendation import (
    SBERTGHRCalculator,
    check_minimax_connectivity,
    evaluate_downstream_job,
    resolve_minimax_client,
    write_comparison_report,
)
from src.experiments.visual_inception_category_evaluation import (
    DEFAULT_KEY_MODE,
    KEY_MODES,
    QUERY_SUITE_ROOTS,
)
from src.paths import OUTPUT, resolve_evaluation_base
from src.visual.optimization import METHODS, normalize_method
from src.visual.precomputed_eval_store import (
    DEFAULT_PRECOMPUTED_STORE_DIR,
    ENCODER_SPEC_BY_SLUG,
)


def _parse_spec(raw: str) -> tuple[str, str, Path]:
    parts = raw.split(":", 2)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        raise argparse.ArgumentTypeError(f"规格必须为 TAG:METHOD:IMAGES_ROOT：{raw}")
    tag, method, root = parts[0].strip(), normalize_method(parts[1].strip()), Path(parts[2].strip()).resolve()
    if method not in METHODS:
        raise argparse.ArgumentTypeError(f"不支持的方法 {method}（可选：{METHODS}）")
    if not root.is_dir():
        raise argparse.ArgumentTypeError(f"成图根目录不存在：{root}")
    return tag, method, root


def _infer_method_from_tag(tag: str) -> str:
    low = tag.lower()
    if "ambilage" in low:
        return "ambilage"
    if "lucid" in low:
        return "lucid"
    return "visual_inception"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_downstream_recommendation.py",
        description=__doc__,
    )
    p.add_argument(
        "--spec",
        action="append",
        default=None,
        help="预计算向量模式：可重复指定 TAG:METHOD:IMAGES_ROOT（如 ambilage:ambilage:output/ambilage_single_clip_l14_9groups_20261118）",
    )
    p.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="冻结 scores.npz 模式：指定已有 eval 目录（如 output/exp_20261121_eval_v4_cleaned/eval）",
    )
    p.add_argument(
        "--tags",
        nargs="+",
        default=["ambilage", "vi_08_02"],
        help="配合 --eval-root 使用的方法标签前缀（默认 ambilage vi_08_02）",
    )
    p.add_argument(
        "--suites",
        nargs="+",
        choices=tuple(QUERY_SUITE_ROOTS),
        default=["zero_overlap_v4"],
        help="测试查询集列表（默认 zero_overlap_v4）",
    )
    p.add_argument(
        "--encoder-slugs",
        nargs="+",
        choices=tuple(ENCODER_SPEC_BY_SLUG),
        default=["02_openclip_bigG14"],
        help="检索编码器列表（默认优先跑 02_openclip_bigG14）",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="下游评测输出根目录（须位于 output/ 内）")
    p.add_argument("--base", type=str, default=None, help="实验基础包名称或路径（如 original_9groups 或 golden_band_9groups）")
    p.add_argument("--key-mode", choices=KEY_MODES, default=DEFAULT_KEY_MODE)
    p.add_argument("--bank", choices=("modified", "clean"), default="modified")
    p.add_argument("--condition", default="none")
    p.add_argument("--group", default="all", help="all 或逗号分隔分组（如 02_chair,09_toothbrush）")
    p.add_argument("--replacement-groups", choices=("all", "selected"), default="all")
    p.add_argument("--precomputed-dir", type=Path, default=DEFAULT_PRECOMPUTED_STORE_DIR)
    p.add_argument("--provider", default="minimax", choices=("minimax", "openai"))
    p.add_argument("--model", default="MiniMax-M3", help="下游被测多模态模型名称（默认 MiniMax-M3）")
    p.add_argument("--api-base", default=None, help="OpenAI 兼容网关地址（默认读取 llm.local.json / llm.example.json，即 https://api.minimax.cn/v1）")
    p.add_argument("--api-key-env", default=None, help="API 密钥环境变量名（默认 MINIMAX_API_KEY）")
    p.add_argument("--api-key-file", default=None, help="可选本地 API 密钥文件路径")
    p.add_argument("--llm-config", type=Path, default=None, help="可选 llm.local.json 配置文件路径")
    p.add_argument("--temperature", type=float, default=None, help="解码温度（MiniMax-M3 默认不发送温度字段；非 M3 模型默认 0.0）")
    p.add_argument("--max-tokens", type=int, default=2048, help="单次请求初始最大 Token 预算（遇 length 自动翻倍至 8192）")
    p.add_argument("--max-retries", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=8, help="并发请求线程数（默认 8）")
    p.add_argument("--check-api", action="store_true", help="发送单条极简多模态 JSON 探针以测试 MiniMax API 连通性与鉴权")
    p.add_argument("--dry-run", action="store_true", help="仅执行零 GPU Top-5 检索、脱敏、多模态载荷组装与校验，不发起 API 请求")
    p.add_argument("--score-only", action="store_true", help="仅对已落盘的 responses.jsonl 离线重算 GHR 与推荐成功率并刷新报告")
    return p


def main(argv: list[str] | None = None) -> int:
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
    )
    args = build_parser().parse_args(argv)
    if args.check_api:
        probe = check_minimax_connectivity(
            provider=args.provider,
            model_name=args.model,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            api_key_file=args.api_key_file,
            llm_config_path=args.llm_config,
            temperature=args.temperature,
        )
        print(json.dumps(probe, ensure_ascii=False, indent=2))
        return 0

    if args.output_dir is None:
        raise ValueError("必须指定 --output-dir（下游评测输出根目录）")
    out_dir = Path(args.output_dir).resolve()
    if not out_dir.is_relative_to(Path(OUTPUT).resolve()):
        raise ValueError("下游评测输出目录必须位于当前项目 output/ 目录内")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_groups = (
        None if args.group == "all" else [g.strip() for g in str(args.group).split(",") if g.strip()]
    )

    if not args.spec and args.eval_root is None and not args.score_only:
        raise ValueError("请指定 --spec TAG:METHOD:IMAGES_ROOT（预计算向量直算模式）或 --eval-root（冻结评估归档模式）")

    client = None
    effective_model = args.model
    effective_provider = args.provider
    if not args.dry_run and not args.score_only:
        client, effective_model, effective_provider, _ = resolve_minimax_client(
            provider=args.provider,
            model_name=args.model,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            api_key_file=args.api_key_file,
            llm_config_path=args.llm_config,
        )

    sbert = None if args.dry_run else SBERTGHRCalculator(store_root=args.precomputed_dir)
    resolved_base = resolve_evaluation_base(args.base)
    summaries = []

    if args.spec:
        specs = [_parse_spec(raw) for raw in args.spec]
        for suite in args.suites:
            for enc_slug in args.encoder_slugs:
                for tag, method, images_root in specs:
                    job_out = out_dir / f"{tag}_{suite}" / enc_slug
                    summary = evaluate_downstream_job(
                        method_tag=tag,
                        method=method,
                        query_suite=suite,
                        encoder_slug=enc_slug,
                        output_dir=job_out,
                        images_root=images_root,
                        base=resolved_base,
                        key_mode=args.key_mode,
                        condition_label=args.condition,
                        bank_name=args.bank,
                        selected_groups=selected_groups,
                        replacement_groups=args.replacement_groups,
                        store_root=args.precomputed_dir,
                        client=client,
                        model_name=effective_model,
                        provider=effective_provider,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        max_retries=args.max_retries,
                        concurrency=args.concurrency,
                        sbert_calculator=sbert,
                        dry_run=args.dry_run,
                        score_only=args.score_only,
                    )
                    summaries.append(summary)
    else:
        eval_root = Path(args.eval_root).resolve() if args.eval_root else None
        for suite in args.suites:
            for enc_slug in args.encoder_slugs:
                for tag in args.tags:
                    panel_name = f"{tag}_{suite}"
                    panel_dir = (eval_root / panel_name) if eval_root is not None else None
                    method = _infer_method_from_tag(tag)
                    job_out = out_dir / panel_name / enc_slug
                    summary = evaluate_downstream_job(
                        method_tag=tag,
                        method=method,
                        query_suite=suite,
                        encoder_slug=enc_slug,
                        output_dir=job_out,
                        eval_panel_dir=panel_dir,
                        base=resolved_base,
                        key_mode=args.key_mode,
                        condition_label=args.condition,
                        bank_name=args.bank,
                        selected_groups=selected_groups,
                        replacement_groups=args.replacement_groups,
                        store_root=args.precomputed_dir,
                        client=client,
                        model_name=effective_model,
                        provider=effective_provider,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        max_retries=args.max_retries,
                        concurrency=args.concurrency,
                        sbert_calculator=sbert,
                        dry_run=args.dry_run,
                        score_only=args.score_only,
                    )
                    summaries.append(summary)

    if not args.dry_run and summaries:
        write_comparison_report(summaries, out_dir)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
