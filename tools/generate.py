"""统一图像生成 CLI 命令行入口 (Unified Generation CLI Tool)。
支持 ambilage 与 lucid 双方法，原生适配 A100 硬件画像参数。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visual.optimization import METHODS
from src.visual.pipeline import generate


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AmbiLAGE 统一对抗样本图像生成工具 (AmbiLAGE & Lucid)"
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="ambilage",
        help="生成方法身份：'ambilage' (环境光协同) 或 'lucid' (原始像素)",
    )
    parser.add_argument(
        "--source",
        "--source-image",
        dest="source",
        required=True,
        help="源商品原图像文件路径 (Source Image)",
    )
    parser.add_argument(
        "--target",
        "--target-image",
        dest="target",
        required=True,
        help="目标商品图像文件路径 (Target Image)",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        required=True,
        help="成图输出落盘目录 (Output Directory)",
    )
    parser.add_argument(
        "--target-text",
        default=None,
        help="目标商品特点描述文本 (若未提供则尝试从同目录 definition.json / features.json 解析)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="优化步数 (ambilage 默认 350, lucid 默认 1000)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="计算设备 (例如 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--epsilon-levels",
        type=int,
        default=16,
        help="像素预算扰动级别 (仅用于 lucid 方法，默认 16)",
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=1.0,
        help="文字对齐监督权重",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        default=True,
        help="是否开启 A100 TensorFloat-32 原生加速 (默认启用)",
    )
    parser.add_argument(
        "--no-tf32",
        dest="allow_tf32",
        action="store_false",
        help="禁用 TensorFloat-32 原生加速",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="全局随机种子 (默认 42)",
    )
    parser.add_argument(
        "--ambilage-config",
        default=None,
        help="AmbiLAGE 自定义配置 JSON 路径或 JSON 字符串",
    )
    parser.add_argument(
        "--optimization-config",
        default=None,
        help="Lucid 优化配置 JSON 路径",
    )
    parser.add_argument('--definition',help='冻结公开图对定义及源商品完整公开检索键')
    parser.add_argument('--smoke-test',action='store_true',help='显式微型夹具，仅中央处理器/六十四像素/至多三步，禁止作为正式实验输入')
    from src.visual.training_surrogates import add_surrogate_arguments
    add_surrogate_arguments(parser)
    args=parser.parse_args(argv)
    if args.method!='ambilage' and (args.surrogate_set is not None or args.surrogate_config is not None):
        parser.error('训练代理选择参数仅用于新方法；固定原始像素方法使用自己的配置')
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    
    amb_cfg = None
    if args.ambilage_config:
        cfg_p = Path(args.ambilage_config)
        if cfg_p.is_file():
            amb_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        else:
            amb_cfg = json.loads(args.ambilage_config)

    print("=" * 60)
    print(f"AmbiLAGE 统一图像生成启动")
    print(f"方法 (Method):       {args.method}")
    print(f"源图 (Source):       {args.source}")
    print(f"目标图 (Target):     {args.target}")
    print(f"输出目录 (Output):   {args.output}")
    print(f"设备 (Device):       {args.device}")
    print(f"TF32 加速 (TF32):    {args.allow_tf32}")
    print(f"随机种子 (Seed):     {args.seed}")
    print("=" * 60)

    result = generate(
        method=args.method,
        source=args.source,
        target=args.target,
        output_dir=args.output,
        steps=args.steps,
        device=args.device,
        target_text=args.target_text,
        epsilon_levels=args.epsilon_levels,
        text_weight=args.text_weight,
        allow_tf32=args.allow_tf32,
        seed=args.seed,
        ambilage_config=amb_cfg,
        optimization_config=args.optimization_config,
        definition=json.loads(Path(args.definition).read_text(encoding='utf-8')) if args.definition else None,
        smoke_test=args.smoke_test,
        surrogate_set=args.surrogate_set,surrogate_config=args.surrogate_config,
    )

    print("\n" + "=" * 60)
    print("生成成功完成！")
    print(f"生成成图路径: {result.image_path}")
    print(f"成图 SHA256:  {result.sha256}")
    print(f"完成记录凭证: {result.ready_path}")
    print(f"详细元数据:   {result.metadata_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
