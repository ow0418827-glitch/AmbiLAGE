"""自包含基础包的分阶段串行流程；生成与正式评估隔离。"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import shutil

from src.paths import EVALUATION_BASE, OUTPUT
from src.visual.audit import digest, freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file
from src.visual.optimization import METHODS


def public_group(base, group):
    """只读取冻结公开图对和公开特点，不读取共享池、正式询问或旧向量。"""
    from src.visual.target_features import extract_product_features, format_retrieval_text
    base = Path(base).resolve()
    manifest = read_json(base / 'manifest.json')
    if group not in manifest['groups']:
        raise ValueError('基础包没有指定组：' + group)
    binding = manifest['groups'][group]
    directory = (base / binding['directory']).resolve()
    if not directory.is_relative_to(base / 'groups'):
        raise ValueError('基础包公开图对目录越界')
    definition_path = (base / binding['definition']).resolve()
    if not definition_path.is_relative_to(directory):
        raise ValueError('公开图对定义不在对应组目录内')
    if sha256_file(definition_path) != binding['definition_sha256']:
        raise ValueError('公开图对定义摘要不符')
    definition = read_json(definition_path)
    source, target = [(directory / definition[role]['image']).resolve() for role in ('source', 'target')]
    if any(not path.is_relative_to(directory) for path in (source, target)):
        raise ValueError('公开商品图片路径越界')
    for role in ('source', 'target'):
        if definition[role]['id'] != binding[role + '_id']:
            raise ValueError('公开图对定义的商品编号与基础包不符')
    for role, path in [('source', source), ('target', target)]:
        if sha256_file(path) != binding[role + '_sha256']:
            raise ValueError('公开商品图像摘要不符：' + role)
    public = manifest['public_metadata']
    metadata_path = (base / public['path']).resolve()
    review_path = (base / public['review_path']).resolve()
    for path, expected in ((metadata_path, public['sha256']), (review_path, public['review_sha256'])):
        if not path.is_relative_to(base / 'public_metadata'):
            raise ValueError('公开商品特点资源路径越界')
        if sha256_file(path) != expected:
            raise ValueError('公开商品特点资源摘要不符')
    feature_path = (base / binding['features']).resolve()
    if not feature_path.is_relative_to(directory) or sha256_file(feature_path) != binding['features_sha256']:
        raise ValueError('冻结公开特点文件路径或摘要不符')
    source_features = extract_product_features(binding['source_id'], db_path=metadata_path, review_path=review_path)
    target_features = extract_product_features(binding['target_id'], db_path=metadata_path, review_path=review_path)
    feature_text = binding['target_feature_text']
    if feature_text != definition['target_feature_text'] or feature_text != target_features['canonical_sentence']:
        raise ValueError('图对与公开特点训练正文不一致')
    source_key = format_retrieval_text(source_features['title'], source_features.get('brand', ''),
                                       source_features.get('attributes', []))
    source_key += '\nFeatureText: ' + source_features['canonical_sentence']
    definition.update(source_public_key=source_key,
                      source_category=source_features.get('category') or definition['category'])
    return dict(source=dict(id=binding['source_id'], title=source_features['title'],
                            image=str(source), image_sha256=binding['source_sha256']),
                target=dict(id=binding['target_id'], title=target_features['title'],
                            image=str(target), image_sha256=binding['target_sha256']),
                target_text=feature_text, definition=definition,
                selection=dict(basis=binding['selection_basis'],
                               passes_calibrated_threshold=binding['passes_calibrated_threshold'],
                               shape_similarity=binding['shape_similarity']),
                public_binding=dict(definition_sha256=binding['definition_sha256'],
                                    features_sha256=binding['features_sha256'],
                                    public_metadata_sha256=public['sha256'], public_review_sha256=public['review_sha256'],
                                    source_sha256=binding['source_sha256'], target_sha256=binding['target_sha256']))


def generate_group(base, group, method, output, *, device='cuda', steps=None,
                   epsilon_levels=8, ambilage_config=None,surrogate_set=None,surrogate_config=None):
    from src.visual.catalog_training_guard import catalog_training_guard
    from src.visual.optimization import generate_attack
    from src.visual.attack_completion import validate_attack_completion
    from src.experiments.pipeline import source_snapshot
    base, root = Path(base).resolve(), Path(output).resolve()
    if not root.is_relative_to(OUTPUT.resolve()):
        raise ValueError('运行结果必须保存在当前项目输出目录')
    frozen_path = root / 'FROZEN_IMAGE.json'
    if method == 'lucid':
        if surrogate_set is not None or surrogate_config is not None:
            raise ValueError('训练代理选择参数用于新方法；固定原始像素对照按原记录复用')
        if steps not in (None, 1000) or epsilon_levels != 8 or ambilage_config is not None:
            raise ValueError('冻结对照只复用八除以二百五十五、完整千步的原始像素成图')
        from src.experiments.lucid_baseline_import import load_imported_baseline
        inputs = public_group(base, group)
        frozen = load_imported_baseline(group, inputs=inputs)
        freeze_json(root / 'PUBLIC_RUN.json', dict(method='lucid', group=group,
            execution='historical_frozen_image_reuse', public=inputs['public_binding'],
            target_text=inputs['target_text'], parameters=dict(steps=1000, epsilon_levels=8),
            image_sha256=frozen['image_sha256'], provenance=frozen.get('historical_reuse')))
        freeze_json(frozen_path, frozen)
        return frozen
    protected = [base / 'shared_memory', base / 'evaluation_inputs',
                 base / 'evaluation_inputs_joint', base / 'evaluation_inputs_vague_joint',
                 base / 'query_images', base / 'evaluation_encoder.json']
    with catalog_training_guard(allowed_output_roots=[root, base / 'groups'], private_paths=protected):
        inputs = public_group(base, group)
        from src.visual._ambilage.config import AmbiLAGEConfig
        from src.visual.training_surrogates import apply_surrogate_overrides,selection_from_config
        if isinstance(ambilage_config,(str,Path)):
            ambilage_config=read_json(ambilage_config)
        effective_config=AmbiLAGEConfig.from_dict(ambilage_config) if isinstance(ambilage_config,dict) else (ambilage_config or AmbiLAGEConfig())
        effective_config=apply_surrogate_overrides(effective_config,surrogate_set,surrogate_config)
        binding = dict(method=method, group=group, public=inputs['public_binding'],
                       target_text=inputs['target_text'], steps=steps,
                       epsilon_levels=epsilon_levels if method == 'lucid' else None,
                       config=effective_config.to_dict(),training_surrogates=selection_from_config(effective_config).audit())
        binding['source_files'] = source_snapshot()
        freeze_json(root / 'PUBLIC_RUN.json', binding)
        if frozen_path.exists():
            frozen = read_json(frozen_path)
            if frozen['public_binding_sha256'] != digest(binding):
                raise ValueError('成图已经冻结，不能更改输入或设置')
            if sha256_file(frozen['image']) != frozen['image_sha256']:
                raise ValueError('冻结图片摘要改变')
            validate_attack_completion(frozen['metadata'], frozen['requested_steps'])
            return frozen
        options = dict(method=method, device=device, target_text=inputs['target_text'])
        if steps is not None:
            options['steps'] = steps
        if method == 'ambilage':
            options.update(ambilage_config=effective_config, definition=inputs['definition'])
        else:
            options['epsilon_levels'] = epsilon_levels
        image, metadata = generate_attack(inputs['source']['image'], inputs['target']['image'],
                                         root / 'generation', **options)
        count = metadata.get('steps_requested', metadata.get('attack', {}).get('steps_requested'))
        if metadata.get('training_objective',{}).get('surrogate_selection')!=binding['training_surrogates']:
            raise ValueError('实际训练代理与本次冻结选择不一致')
        validate_attack_completion(metadata, count, dict(method=method,
            source_image_sha256=inputs['source']['image_sha256'], target_image_sha256=inputs['target']['image_sha256']))
        final = root / 'frozen.png'
        shutil.copyfile(image, final)
        frozen = dict(schema_version=1, status='frozen', method=method, group=group,
                      image=str(final), image_sha256=sha256_file(final), metadata=metadata,
                      requested_steps=count, source=inputs['source'], target=inputs['target'],
                      selection=inputs['selection'], public_binding_sha256=digest(binding))
        freeze_json(frozen_path, frozen)
        return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('generate', 'evaluate', 'verify', 'report', 'all'))
    parser.add_argument('--base', type=Path, default=EVALUATION_BASE)
    parser.add_argument('--group', required=True, help='冻结组键，如 07_shampoo；all 表示九组')
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--epsilon-levels', type=int, default=8)
    parser.add_argument('--ambilage-config', type=Path)
    from src.visual.training_surrogates import add_surrogate_arguments
    add_surrogate_arguments(parser)
    parser.add_argument('--query-mode', choices=('text', 'joint'), default='text')
    parser.add_argument('--query-manifest', type=Path)
    parser.add_argument('--encoder-config', type=Path)
    parser.add_argument('--evaluation-name', default='default',
                        help='独立评估目录名；不同查询模式或留出编码器使用不同名称')
    parser.add_argument('--defense-mode', choices=('none', 'panel'), default='panel')
    args = parser.parse_args(argv)
    if args.surrogate_set is not None or args.surrogate_config is not None:
        if args.method!='ambilage' or args.stage not in ('generate','all'):
            parser.error('训练代理选择参数仅用于新方法的生成或全阶段命令')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.evaluation_name):
        parser.error('评估名称只允许字母、数字、横线和下划线')
    base, output = args.base.resolve(), args.output.resolve()
    if not output.is_relative_to(OUTPUT.resolve()):
        parser.error('输出目录必须位于当前项目 output 内')
    manifest = read_json(base / 'manifest.json')
    groups = list(manifest['groups']) if args.group == 'all' else [args.group]
    if args.query_manifest is not None and len(groups) != 1:
        parser.error('多组运行不能共用一份正式询问清单')
    config = read_json(args.ambilage_config) if args.ambilage_config else None
    # 所有生成先完成并冻结，随后才进入任何一组正式评估。
    if args.stage in ('generate', 'all'):
        for group in groups:
            generate_group(base, group, args.method, output / group / args.method, device=args.device,
                           steps=args.steps, epsilon_levels=args.epsilon_levels, ambilage_config=config,
                           surrogate_set=args.surrogate_set,surrogate_config=args.surrogate_config)
    replacement_frozens = None
    if args.stage in ('evaluate', 'all'):
        replacement_frozens = []
        for key in manifest['groups']:
            method_root = output / key / args.method
            path = method_root / 'FROZEN_IMAGE.json'
            if not path.is_file() and args.method == 'lucid':
                generate_group(base, key, 'lucid', method_root, device=args.device)
            if not path.is_file():
                raise ValueError('统一修改库须先完成全部九组成图并冻结，尚缺：' + key)
            item = read_json(path)
            public = public_group(base, key)
            if (any(item['source'].get(k) != public['source'][k] for k in ('id', 'title', 'image_sha256'))
                    or any(item['target'].get(k) != public['target'][k] for k in ('id', 'title'))):
                raise ValueError('冻结图与当前基础包源商品或目标不一致：' + key)
            if item['target']['image_sha256'] != public['target']['image_sha256']:
                raise ValueError('冻结目标图片与当前基础包不一致：' + key)
            item['source_memory_id'] = manifest['shared_memory']['source_mapping'][key]['source_memory_id']
            replacement_frozens.append(item)
    for group in groups:
        root = output / group / args.method
        evaluation_root = root / 'evaluation' / args.evaluation_name
        try:
            if args.stage in ('evaluate', 'all'):
                from src.experiments.shared_evaluation import evaluate, report
                from src.experiments.defenses import conditions
                binding = manifest['groups'][group]
                query_path = args.query_manifest or base / binding['query_manifest']
                if args.query_manifest is None and sha256_file(query_path) != binding['query_manifest_sha256']:
                    raise ValueError('基础包正式询问清单摘要不符')
                pool = base / manifest['shared_memory']['pool_file']
                if sha256_file(pool) != manifest['shared_memory']['pool_sha256']:
                    raise ValueError('基础包共享池摘要不符')
                evaluate(read_json(root / 'FROZEN_IMAGE.json'), evaluation_root, pool_path=pool,
                    query_manifest=query_path, query_mode=args.query_mode, device=args.device,
                    source_memory_id=manifest['shared_memory']['source_mapping'][group]['source_memory_id'],
                    selected_conditions=conditions() if args.defense_mode == 'panel' else conditions()[:1],
                    encoder_config=args.encoder_config or base / 'evaluation_encoder.json',
                    replacement_frozens=replacement_frozens,
                    bank_root=output / 'shared_banks' / args.method / args.evaluation_name,
                    memory_vector_cache_dir=output / 'memory_vectors' / args.evaluation_name)
                report(evaluation_root)
            elif args.stage == 'verify':
                from src.experiments.shared_evaluation import verify
                verify(evaluation_root, device=args.device, isolated_process=True)
            elif args.stage == 'report':
                from src.experiments.shared_evaluation import report
                report(evaluation_root)
        except Exception as error:
            save_json(root / 'FAILED.json', dict(stage=args.stage, error_type=type(error).__name__, reason=str(error)))
            raise
    return 0
