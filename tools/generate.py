"""统一图像生成 CLI 命令行入口 (Unified Generation CLI Tool)。
支持登记的三种方法，兼容显卡与中央处理器测试环境。
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
        description='统一对抗样本图像生成工具'
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default="ambilage",
        help='生成方法：环境光协同、原始像素或视觉记忆植入',
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
        help='优化步数；环境光协同默认二百五十，原始像素默认一千，视觉记忆植入默认一百',
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="计算设备 (例如 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--epsilon-levels",
        type=int,
        default=None,
        help='像素扰动预算；原始像素默认十六级，视觉记忆植入默认八级',
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=1.0,
        help="文字对齐监督权重（lucid 默认 1.0；ambilage 通过 --ambilage-config 设置，默认 0.35）",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        default=True,
        help="是否开启 Ampere 及更新架构显卡 (L4/A100) 的 TensorFloat-32 加速 (T4/CPU 自动跳过，默认启用)",
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
        "--enable-eot",
        dest="eot_enabled",
        action="store_true",
        default=None,
        help="显式开启 AmbiLAGE 优化阶段 EoT 变换（默认不开启）",
    )
    parser.add_argument(
        "--disable-eot",
        dest="eot_enabled",
        action="store_false",
        help="显式关闭 AmbiLAGE 优化阶段 EoT 变换",
    )
    parser.add_argument(
        "--enable-diffusion-eot",
        dest="diffusion_eot_enabled",
        action="store_true",
        default=None,
        help="显式开启 AmbiLAGE 优化阶段扩散净化 EoT",
    )
    parser.add_argument(
        "--disable-diffusion-eot",
        dest="diffusion_eot_enabled",
        action="store_false",
        help="显式关闭 AmbiLAGE 优化阶段扩散净化 EoT",
    )
    parser.add_argument(
        "--enable-multiscale-eot",
        dest="multiscale_eot_enabled",
        action="store_true",
        default=None,
        help="显式开启纯多尺度破网格视角变换（可独立于防御 EoT 开启）",
    )
    parser.add_argument(
        "--disable-multiscale-eot",
        dest="multiscale_eot_enabled",
        action="store_false",
        help="显式关闭多尺度破网格视角变换",
    )
    parser.add_argument(
        "--multiscale-eot-profile",
        choices=("legacy", "geom_v2", "geom_photo_v3"),
        default=None,
        help="多尺度破网格档位（默认 geom_v2；保留 geom_photo_v3 几何+轻量光度与 legacy 旧版）",
    )
    parser.add_argument(
        "--multiscale-identity-weight-end",
        type=float,
        default=None,
        help="多尺度破网格恒等视图后期退火权重 [0.05, 1.0]（默认 0.35）",
    )
    parser.add_argument(
        "--optimization-config",
        default=None,
        help="Lucid 优化配置 JSON 路径",
    )
    parser.add_argument('--definition',help='冻结公开图对定义及源商品完整公开检索键')
    parser.add_argument('--visual-inception-config', help='视觉记忆植入基线的配置文件路径')
    parser.add_argument('--training-manifest', type=Path, help='环境光协同或视觉记忆植入的独立公开查询及候选库冻结清单')
    parser.add_argument('--smoke-test',action='store_true',help='显式微型夹具，仅中央处理器/六十四像素/至多三步，禁止作为正式实验输入')
    from src.visual.training_surrogates import add_surrogate_arguments
    add_surrogate_arguments(parser)
    args=parser.parse_args(argv)
    if args.method=='lucid' and (args.surrogate_set is not None or args.surrogate_config is not None):
        parser.error('训练代理选择参数仅用于新方法；固定原始像素方法使用自己的配置')
    if args.visual_inception_config and args.method != 'visual_inception':
        parser.error('视觉记忆植入配置只能用于对应方法')
    if args.training_manifest and args.method not in ('visual_inception', 'ambilage'):
        parser.error('公开训练清单仅用于环境光协同或视觉记忆植入方法')
    if args.ambilage_config and args.method != 'ambilage':
        parser.error('环境光协同配置只能用于对应方法')
    if (
        args.eot_enabled is not None
        or args.diffusion_eot_enabled is not None
        or args.multiscale_eot_enabled is not None
        or args.multiscale_eot_profile is not None
        or args.multiscale_identity_weight_end is not None
    ) and args.method != 'ambilage':
        parser.error('EoT 接口开关与多尺度档位仅用于环境光协同（ambilage）方法')
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
    if (
        args.eot_enabled is not None
        or args.diffusion_eot_enabled is not None
        or args.multiscale_eot_enabled is not None
        or args.multiscale_eot_profile is not None
        or args.multiscale_identity_weight_end is not None
    ):
        amb_cfg = dict(amb_cfg or {})
        if args.eot_enabled is not None:
            amb_cfg['eot_enabled'] = bool(args.eot_enabled)
        if args.diffusion_eot_enabled is not None:
            amb_cfg['diffusion_eot_enabled'] = bool(args.diffusion_eot_enabled)
        if args.multiscale_eot_enabled is not None:
            amb_cfg['multiscale_eot_enabled'] = bool(args.multiscale_eot_enabled)
        if args.multiscale_eot_profile is not None:
            amb_cfg['multiscale_eot_profile'] = str(args.multiscale_eot_profile)
        if args.multiscale_identity_weight_end is not None:
            amb_cfg['multiscale_identity_weight_end'] = float(args.multiscale_identity_weight_end)

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
        visual_inception_config=args.visual_inception_config,
        optimization_config=args.optimization_config,
        definition=json.loads(Path(args.definition).read_text(encoding='utf-8')) if args.definition else None,
        smoke_test=args.smoke_test,
        surrogate_set=args.surrogate_set,surrogate_config=args.surrogate_config,
        training_manifest=args.training_manifest,
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
