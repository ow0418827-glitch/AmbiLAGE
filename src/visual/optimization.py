"""公开生成方法的统一接口；完整实验先冻结公开图对及特点。"""
import copy
import json
from src.paths import ProjectPath as Path

METHODS = ('lucid', 'ambilage', 'visual_inception')
NATURAL_METHODS = ('ambilage',)


def normalize_method(method):
    if not isinstance(method, str):
        raise ValueError('方法名称必须为字符串')
    m = method.lower()
    if m not in METHODS:
        raise ValueError("方法必须是 'lucid' 或 'ambilage' 或 'visual_inception'")
    return m


def add_method_arguments(parser):
    parser.add_argument('--method', choices=METHODS, default='ambilage',
                        help='公开生成方法身份：原始像素、环境光协同或视觉记忆植入')
    parser.add_argument('--ambilage-config', help='AmbiLAGE 环境光协同优化的公开训练配置')
    parser.add_argument('--visual-inception-config', help='视觉记忆植入图片扰动基线的公开配置')


def options_from_args(args, group=None):
    if group is not None:
        raise ValueError('已移除按组分派配置和记忆的机制')
    method = normalize_method(args.method)
    config = getattr(args, 'ambilage_config', None)
    if config and method != 'ambilage':
        raise ValueError('ambilage 配置只能用于 ambilage 方法')
    inception = getattr(args, 'visual_inception_config', None)
    if inception and method != 'visual_inception':
        raise ValueError('视觉记忆植入配置只能用于对应方法')
    return ({'ambilage_config': config} if config else {}) | ({'visual_inception_config': inception} if inception else {})


def _adjacent_feature_text(target):
    """读取已冻结的特点正文；文件损坏或两份正文冲突时不能静默关闭监督。"""
    texts = []
    directory = Path(target).resolve().parent
    for name in ('definition.json', 'features.json'):
        path = directory / name
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding='utf-8'))
        text = (document.get('target_feature_text') or document.get('alignment_canonical_text')
                or document.get('canonical_sentence') or document.get('feature_text')
                or document.get('target', {}).get('feature_text'))
        if text is not None:
            if not isinstance(text, str) or not text.strip():
                raise ValueError('冻结的目标商品特点文字必须为非空字符串')
            texts.append(text)
    if len(set(texts)) > 1:
        raise ValueError('图对定义与特点文件中的目标文字不一致')
    return texts[0] if texts else None


def generate_attack(source, target, output_dir, *, method='ambilage', steps=None,
                    epsilon_levels=None, device='cuda', optimization_config=None,
                    ambilage_config=None, visual_inception_config=None, target_text=None, text_weight=1.0,
                    allow_tf32=True, seed=42, definition=None, smoke_test=False,
                    surrogate_set=None,surrogate_config=None):
    method = normalize_method(method)
    if visual_inception_config is not None and method != 'visual_inception':
        raise ValueError('视觉记忆植入配置只能用于对应方法')
    if method == 'lucid':
        if surrogate_set is not None or surrogate_config is not None:
            raise ValueError('训练代理选择参数用于新方法；原始像素方法使用自己的冻结配置')
        if ambilage_config:
            raise ValueError('像素方法不能读取 ambilage 方法配置')
        frozen_text = _adjacent_feature_text(target)
        if target_text is None:
            target_text = frozen_text
        if target_text is None:
            raise ValueError('像素方法默认文字对齐需要选图阶段冻结的目标商品特点文字')
        if frozen_text is not None and target_text != frozen_text:
            raise ValueError('像素方法的文字与同一图对冻结的目标特点不一致')
        from src.visual._lucid.pipeline import generate_attack as generate_lucid
        image, metadata = generate_lucid(
            source, target, output_dir,
            optimization_config=optimization_config, steps=1000 if steps is None else steps,
            epsilon_levels=16 if epsilon_levels is None else epsilon_levels, device=device,
            target_text=target_text, text_weight=text_weight,
            allow_tf32=allow_tf32,seed=seed,smoke_test=smoke_test
        )
        from src.visual.image_utils import sha256_file
        metadata = dict(metadata, method='lucid', source_image_sha256=sha256_file(source),
                        target_image_sha256=sha256_file(target))
        if metadata.get('text_alignment'):
            metadata['text_alignment']['source'] = 'target_product_features'
    elif method == 'visual_inception':
        if ambilage_config is not None or optimization_config is not None or text_weight != 1.0:
            raise ValueError('视觉记忆植入基线须使用自己的配置和损失权重')
        from src.visual._visual_inception.pipeline import generate_visual_inception
        image, metadata = generate_visual_inception(source, target, output_dir,
            config=visual_inception_config, steps=steps, epsilon_levels=epsilon_levels,
            device=device, target_text=target_text, definition=definition,
            allow_tf32=allow_tf32, seed=seed, smoke_test=smoke_test,
            surrogate_set=surrogate_set, surrogate_config=surrogate_config)
    elif method == 'ambilage':
        from src.visual._ambilage.pipeline import run_attack as run_ambilage_attack
        frozen_text = _adjacent_feature_text(target)
        if target_text is None:
            target_text = frozen_text
        if frozen_text is not None and target_text != frozen_text:
            raise ValueError('方法文字与同一图对冻结的目标特点不一致')
        if definition is None:
            adjacent = Path(source).resolve().parent / 'definition.json'
            if adjacent.is_file():
                definition=json.loads(adjacent.read_text(encoding='utf-8'))
        image, metadata = run_ambilage_attack(
            source, target, output_dir,
            target_text=target_text,
            steps=steps,
            device=device,
            ambilage_config=ambilage_config,
            allow_tf32=allow_tf32,
            seed=seed,definition=definition,smoke_test=smoke_test,
            surrogate_set=surrogate_set,surrogate_config=surrogate_config,
        )
        metadata=dict(metadata,status='success',source_image_sha256=metadata['source_sha256'],
                      target_image_sha256=metadata['target_sha256'],ambilage_ready=copy.deepcopy(metadata))
    else:
        raise ValueError(f'未知方法: {method}')

    from src.visual.image_utils import write_json
    write_json(Path(output_dir) / 'generation_result.json', metadata)
    return image, metadata
