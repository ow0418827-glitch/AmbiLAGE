"""多模态下游商品推荐评测 CLI（零 GPU / 免检索编码器加载）。

支持两种零 GPU 上游 Top-k 记忆获取模式：
  1. 预计算向量直算模式（推荐）：通过 `--spec TAG:METHOD:IMAGES_ROOT` 指定待测方法成图目录，
     直接从 `output/cache/precomputed_eval_vectors` 读取三编码器预计算向量，在纯 CPU 上毫秒级完成
     189 条记忆库检索排序与 Top-k 提取，无需加载任何 CLIP/OpenCLIP/SigLIP 模型、零 GPU 占用。
  2. 冻结评估目录复用模式：通过 `--eval-root output/exp_20261121_eval_v4_golden_band/eval` 与 `--tags vi_10_00_set2`
     直接读取已归档的 `scores.npz` + `modified.json` 复原 Top-k 记忆。

支持的协议对齐与 CoT 剥离消融参数：
  - `--prompt-mode`：`personalized`（对齐后默认无防御基线：信任长期记忆、必选 1 个首选、首句直推）或
    `skeptical`（Visual Inception Appendix B.17 Skeptical Prompting 防御对照）
  - `--memory-text-mode`：`title_brand_only`（默认：仅保留 Title + Brand + Image，剥离竞品属性与摘要噪声）、
    `title_only`（仅 Title + Image）或 `full`（完整原始文本块）
  - `--top-k`：送入下游 MLLM 的召回记忆窗口大小（支持 `1 / 3 / 5`，默认 `5`）
  - `--thinking-mode` / `--reasoning-effort` / `--reasoning-split` / `--extra-body-json`：
    支持关闭或压缩 MiniMax-M3 思考链（`thinking_mode=disabled` 或 `reasoning_effort=low`），
    以及接入非 CoT OpenAI 兼容多模态模型（`--provider openai` / `openai_compatible`）进行混杂因素剥离对照。
  - 离线在 CPU 上同步结算原论文口径 GHR（SBERT `all-MiniLM-L6-v2` 整段 + 分句最大余弦，τ = 0.70 / 0.75 / 0.80）、
    目标商品推荐成功率（严格精确匹配 + 简写容错匹配）、`query_family` 分层指标及条件转化率 P(推荐成功 | Top-k)。
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
    DEFAULT_MEMORY_TEXT_MODE,
    DEFAULT_PROMPT_MODE,
    MEMORY_TEXT_MODES,
    PROMPT_MODES,
    QUERY_FAMILY_ORDER,
    REASONING_EFFORT_CHOICES,
    REASONING_SPLIT_MODES,
    THINKING_MODES,
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
from src.visual.audit import read_json
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


def _parse_extra_body_json(raw: str | None) -> dict | None:
    if raw is None or not str(raw).strip():
        return None
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"--extra-body-json 必须为合法 JSON 对象：{exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--extra-body-json 顶层必须为 JSON 对象（dict）")
    return parsed


def _parse_query_families(raw: str | None) -> list[str] | None:
    if raw is None or str(raw).strip().lower() == "all":
        return None
    families = [f.strip() for f in str(raw).split(",") if f.strip()]
    invalid = [f for f in families if f not in QUERY_FAMILY_ORDER]
    if invalid:
        raise ValueError(f"不支持的 --query-families {invalid}；可选：{QUERY_FAMILY_ORDER}")
    return families or None


def _validate_run_label(raw: str | None) -> str | None:
    if raw is None:
        return None
    label = str(raw).strip()
    if not label:
        return None
    if "/" in label or "\\" in label or ".." in label:
        raise ValueError(f"--run-label 不能包含路径分隔符或 '..'：{raw!r}")
    return label


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate_downstream_recommendation.py",
        description=__doc__,
    )
    p.add_argument(
        "--spec",
        action="append",
        default=None,
        help="预计算向量模式：可重复指定 TAG:METHOD:IMAGES_ROOT（如 vi_10_00_set2:visual_inception:output/exp_20261121_vi_10_00_golden_band_9groups）",
    )
    p.add_argument(
        "--eval-root",
        type=Path,
        default=None,
        help="冻结 scores.npz 模式：指定已有 eval 目录（如 output/exp_20261121_eval_v4_golden_band/eval）",
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

    # 下游协议对齐与上下文消融参数（维度一 / 二 / 四）
    p.add_argument(
        "--prompt-mode",
        choices=PROMPT_MODES,
        default=DEFAULT_PROMPT_MODE,
        help="系统提示词协议：personalized（默认无防御基线）或 skeptical（Appendix B.17 怀疑审慎防御对照）",
    )
    p.add_argument(
        "--memory-text-mode",
        choices=MEMORY_TEXT_MODES,
        default=DEFAULT_MEMORY_TEXT_MODE,
        help="记忆文本格式：title_brand_only（默认，去属性/摘要）、title_only 或 full",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="传入下游 MLLM 的记忆条数窗口（支持 1 / 3 / 5，默认 5）",
    )
    p.add_argument(
        "--query-families",
        default="all",
        help=f"all 或逗号分隔查询家族筛选（可选：{','.join(QUERY_FAMILY_ORDER)}）",
    )
    p.add_argument(
        "--max-queries-per-group",
        type=int,
        default=None,
        help="每组按 query_family 轮转分层抽样上限（用于低成本小规模试点验证，默认全量 20 条/组）",
    )
    p.add_argument(
        "--run-label",
        default=None,
        help="可选运行子目录标签（追加到 output_dir/<tag>_<suite>_<run_label>/<encoder> 以区分多项消融）",
    )

    # 模型连接与 CoT / Reasoning 剥离控制参数
    p.add_argument("--provider", default="minimax", choices=("minimax", "openai", "openai_compatible"))
    p.add_argument("--model", default="MiniMax-M3", help="下游被测多模态模型名称（默认 MiniMax-M3）")
    p.add_argument("--api-base", default=None, help="OpenAI 兼容网关地址（默认读取 llm.local.json / llm.example.json）")
    p.add_argument("--api-key-env", default=None, help="API 密钥环境变量名（默认 MINIMAX_API_KEY / OPENAI_API_KEY）")
    p.add_argument("--api-key-file", default=None, help="可选本地 API 密钥文件路径")
    p.add_argument("--llm-config", type=Path, default=None, help="可选 llm.local.json 配置文件路径")
    p.add_argument("--temperature", type=float, default=None, help="解码温度（MiniMax-M3 系列不支持且不发送该字段，固定 1.0；非 CoT 模型默认 0.0）")
    p.add_argument(
        "--thinking-mode",
        choices=THINKING_MODES,
        default="enabled",
        help="CoT 思考开关：enabled（默认）或 disabled（MiniMax-M3 通过 thinking.type=disabled 剥离思考链）",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default="default",
        help="思考深度预算：default（不传参）或 low / medium / high / xhigh / max（MiniMax-M3.1 压缩 CoT 请用 low）",
    )
    p.add_argument(
        "--reasoning-split",
        choices=REASONING_SPLIT_MODES,
        default="auto",
        help="MiniMax reasoning_split 开关：auto（enabled 时 true，disabled 时不传）、true 或 false",
    )
    p.add_argument(
        "--extra-body-json",
        default=None,
        help="可选附加 JSON 对象并入请求 extra_body（例如本地非 CoT 网关自定义开关）",
    )
    p.add_argument("--max-tokens", type=int, default=2048, help="单次请求初始最大 Token 预算（遇 length 自动翻倍至 8192）")
    p.add_argument("--max-retries", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=8, help="并发请求线程数（默认 8）")

    # 探针 / 干跑 / 离线重算与跨实验汇总工具模式
    p.add_argument("--check-api", action="store_true", help="发送单条极简多模态 JSON 探针以测试连通性、鉴权与 CoT 参数")
    p.add_argument("--dry-run", action="store_true", help="仅执行零 GPU Top-k 检索、脱敏、多模态载荷组装与校验，不发起 API 请求")
    p.add_argument("--score-only", action="store_true", help="仅对已落盘的 responses.jsonl 离线重算 GHR（含分句最大）与推荐成功率并刷新报告")
    p.add_argument(
        "--compare-summaries",
        nargs="+",
        type=Path,
        default=None,
        help="离线汇总模式：传入多个已有 summary.json 路径，在 --output-dir 生成合并对比报告",
    )
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
    if args.top_k < 1:
        raise ValueError(f"--top-k 必须 >= 1，当前为 {args.top_k}")
    if args.max_queries_per_group is not None and args.max_queries_per_group < 1:
        raise ValueError(f"--max-queries-per-group 必须 >= 1，当前为 {args.max_queries_per_group}")
    extra_body = _parse_extra_body_json(args.extra_body_json)
    query_families = _parse_query_families(args.query_families)
    run_label = _validate_run_label(args.run_label)
    reasoning_effort = None if args.reasoning_effort == "default" else args.reasoning_effort

    if args.check_api:
        probe = check_minimax_connectivity(
            provider=args.provider,
            model_name=args.model,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            api_key_file=args.api_key_file,
            llm_config_path=args.llm_config,
            temperature=args.temperature,
            prompt_mode=args.prompt_mode,
            thinking_mode=args.thinking_mode,
            reasoning_effort=reasoning_effort,
            reasoning_split=args.reasoning_split,
            extra_body=extra_body,
            max_tokens=args.max_tokens,
        )
        print(json.dumps(probe, ensure_ascii=False, indent=2))
        return 0

    if args.output_dir is None:
        raise ValueError("必须指定 --output-dir（下游评测输出根目录）")
    out_dir = Path(args.output_dir).resolve()
    if not out_dir.is_relative_to(Path(OUTPUT).resolve()):
        raise ValueError("下游评测输出目录必须位于当前项目 output/ 目录内")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_summaries:
        loaded_summaries = []
        for s_path in args.compare_summaries:
            sp = Path(s_path).resolve()
            if not sp.is_file():
                raise FileNotFoundError(f"找不到 summary.json：{sp}")
            doc = read_json(sp)
            if "overall" not in doc:
                raise ValueError(f"{sp} 缺少 overall 字段，不是有效的 downstream summary.json")
            loaded_summaries.append(doc)
        report_p = write_comparison_report(loaded_summaries, out_dir)
        print(json.dumps({"comparison_report": str(report_p), "run_count": len(loaded_summaries)}, ensure_ascii=False, indent=2))
        return 0

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
                    run_subdir = f"{tag}_{suite}_{run_label}" if run_label else f"{tag}_{suite}"
                    job_out = out_dir / run_subdir / enc_slug
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
                        prompt_mode=args.prompt_mode,
                        memory_text_mode=args.memory_text_mode,
                        top_k=args.top_k,
                        thinking_mode=args.thinking_mode,
                        reasoning_effort=reasoning_effort,
                        reasoning_split=args.reasoning_split,
                        extra_body=extra_body,
                        query_families=query_families,
                        max_queries_per_group=args.max_queries_per_group,
                        run_label=run_label,
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
                    run_subdir = f"{panel_name}_{run_label}" if run_label else panel_name
                    job_out = out_dir / run_subdir / enc_slug
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
                        prompt_mode=args.prompt_mode,
                        memory_text_mode=args.memory_text_mode,
                        top_k=args.top_k,
                        thinking_mode=args.thinking_mode,
                        reasoning_effort=reasoning_effort,
                        reasoning_split=args.reasoning_split,
                        extra_body=extra_body,
                        query_families=query_families,
                        max_queries_per_group=args.max_queries_per_group,
                        run_label=run_label,
                    )
                    summaries.append(summary)

    if not args.dry_run and summaries:
        write_comparison_report(summaries, out_dir)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
