"""Prepare real public images, declared-service paraphrases, then frozen bundles.

Each stage is explicit and resumable. This tool never optimizes an image, reads
formal queries, or opens shared-memory contents. Local service credentials stay
in memory and never enter any public source snapshot or console output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.experiments.base_workflow import public_group
from src.visual.io import save_json, sha256_file
from src.visual._visual_inception.data_preparation import (
    _write_rows, completion_endpoint, generate_paraphrases, prepare_training_bundle,
)
from src.visual._visual_inception.targets import content_sha256, prepare_semantic_targets
from src.visual._visual_inception.training_data import (
    decoded_image_sha256, load_training_bundle, read_source_rows,
    validate_public_bundle_root,
)
from src.visual.selection.webshop_public import download_public_image


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _public(group, base):
    public = public_group(base, group)
    semantic = prepare_semantic_targets(public['source']['image'], public['target']['image'],
        target_text=public['target_text'],
        definition=dict(public['definition'], target_product_features=public['target_features']))
    return public, semantic


def _catalog_snapshot(group, catalog_root):
    root = validate_public_bundle_root(catalog_root / group)
    audit = _json(root / 'sampling_manifest.json')
    for name, key in (('descriptions.jsonl', 'descriptions_sha256'),
                      ('candidate_reserve.jsonl', 'candidate_reserve_sha256')):
        if audit.get(key) != sha256_file(root / name):
            raise ValueError('公开抽样快照与冻结摘要不符')
    return audit


def _image_binding(public, group, catalog_root, count):
    _catalog_snapshot(group, catalog_root)
    return dict(sampling_manifest_sha256=sha256_file(catalog_root / group / 'sampling_manifest.json'),
        candidate_reserve_sha256=sha256_file(catalog_root / group / 'candidate_reserve.jsonl'),
        source_image_sha256=sha256_file(public['source']['image']),
        target_image_sha256=sha256_file(public['target']['image']), candidate_count=count)


def _verify_candidates(root, catalog_root, group, binding):
    final = root / 'candidates.jsonl'
    rows = list(read_source_rows(final).values())
    audit = _json(root / 'candidate_download_audit.json')
    if audit.get('input_binding') != binding or audit.get('candidates_sha256') != sha256_file(final):
        raise ValueError('候选清单或公开图对来源改变，必须使用新的准备目录')
    reserve = list(read_source_rows(catalog_root / group / 'candidate_reserve.jsonl').values())
    if len(rows) != binding['candidate_count']:
        raise ValueError('现有候选数量不符')
    prior_rank, seen = 0, set()
    for row in rows:
        rank = row.get('reserve_rank')
        if (type(rank) is not int or not prior_rank < rank <= len(reserve)
                or row['id'] != reserve[rank-1]['id'] or row['image_url'] != reserve[rank-1]['image_url']):
            raise ValueError('候选编号或顺序与冻结随机候补清单不一致')
        image = (root / row['image']).resolve()
        if not image.is_relative_to(root) or sha256_file(image) != row['image_sha256']:
            raise ValueError('候选图片路径或摘要不符')
        pixels = decoded_image_sha256(image)
        if pixels != row['decoded_image_sha256'] or pixels in seen:
            raise ValueError('候选解码像素摘要不符或重复')
        seen.add(pixels); prior_rank = rank
    return rows


def _local_service_config(path, *, require_key):
    config_path = Path(path).resolve()
    allowed = (PROJECT / 'src/agent/config').resolve()
    if not config_path.is_relative_to(allowed) or not config_path.name.endswith('.local.json'):
        raise ValueError('凭据只允许从项目本地服务配置读取')
    config = _json(config_path).get('model', {})
    required = ('model', 'base_url', 'api_key') if require_key else ('model', 'base_url')
    if config.get('provider') != 'minimax' or not all(isinstance(config.get(k), str) and config[k].strip() for k in required):
        raise ValueError('本地服务配置缺少有效的稀宇模型连接信息')
    identity = dict(provider=config['provider'], model=config['model'], endpoint=completion_endpoint(config['base_url']))
    return config, identity


def prepare_images(group, *, base, catalog_root, inputs_root, count=10):
    root = validate_public_bundle_root(inputs_root / group)
    final = root / 'candidates.jsonl'
    public, semantic = _public(group, base)
    binding = _image_binding(public, group, catalog_root, count)
    if final.exists():
        rows = _verify_candidates(root, catalog_root, group, binding)
        return dict(status='reused', count=len(rows))
    reserve = list(read_source_rows(catalog_root / group / 'candidate_reserve.jsonl').values())
    forbidden = {decoded_image_sha256(public[r]['image']) for r in ('source', 'target')}
    seen, rows, failures = set(), [], []
    root.mkdir(parents=True, exist_ok=True)
    for rank, candidate in enumerate(reserve, 1):
        try:
            product = dict(candidate, source_url=candidate.get('source_url') or
                           'https://www.amazon.com/dp/' + candidate['id'])
            record = download_public_image(product, root)
            pixels = decoded_image_sha256(record['image'])
            if pixels in forbidden or pixels in seen:
                failures.append(dict(id=candidate['id'], reason='duplicate_decoded_image', reserve_rank=rank))
                continue
            seen.add(pixels)
            rows.append(dict(id=candidate['id'], category=semantic['category'],
                title=candidate['title'], image=Path(record['image']).relative_to(root).as_posix(),
                image_sha256=record['image_sha256'], decoded_image_sha256=pixels,
                image_url=candidate['image_url'], source_url=product['source_url'], reserve_rank=rank))
            print(json.dumps(dict(stage='images', group=group, accepted=len(rows), required=count), ensure_ascii=False), flush=True)
        except Exception as exc:
            # Record only a public identifier and error class, never an exception body.
            failures.append(dict(id=candidate['id'], reason=type(exc).__name__, reserve_rank=rank))
        if len(rows) == count:
            break
    audit = dict(group=group, requested_count=count,
        accepted_count=len(rows), failure_records=failures,
        selection='first_valid_in_frozen_random_reserve_order',
        sampling_manifest_sha256=sha256_file(catalog_root / group / 'sampling_manifest.json'),
        input_binding=binding, formal_memory_contents_read=False, formal_queries_read=False)
    save_json(root / 'candidate_download_audit.json', audit)
    if len(rows) != count:
        raise ValueError('合格公开候选图片不足；保留下载审计，不冻结不完整清单')
    digest = _write_rows(final, rows)
    save_json(root / 'candidate_download_audit.json', dict(audit, candidates_sha256=digest))
    return dict(status='prepared', count=len(rows), rejected=len(failures))


def prepare_paraphrases(group, *, catalog_root, inputs_root, local_service_config):
    root = validate_public_bundle_root(inputs_root / group / 'paraphrases')
    final = root / 'model_paraphrases.jsonl'
    _catalog_snapshot(group, catalog_root)
    descriptions = list(read_source_rows(catalog_root / group / 'descriptions.jsonl').values())
    config, identity = _local_service_config(local_service_config, require_key=not final.exists())
    if final.exists():
        config.clear()
        provenance = _json(root / 'generation_provenance.json')
        rows = read_source_rows(final)
        if len(rows) != 30 or sha256_file(final) != provenance['output_sha256']:
            raise ValueError('已有改写数量或摘要不符')
        if provenance['input_descriptions_sha256'] != content_sha256(descriptions[:50]):
            raise ValueError('描述来源改变，必须重新生成改写并使用新目录')
        actual_identity = {key: provenance.get(key) for key in identity}
        if actual_identity != identity or provenance.get('requested_model') != identity['model']:
            raise ValueError('声明的模型或服务地址改变，不能复用旧改写')
        return dict(status='reused', count=30, model=provenance['model'])
    env_name = 'VISUAL_INCEPTION_PARAPHRASE_SERVICE_KEY'
    previous = os.environ.get(env_name)
    try:
        os.environ[env_name] = config['api_key']
        result = generate_paraphrases(
            descriptions,
            provider=config['provider'], model=config['model'], base_url=config['base_url'],
            api_key_env=env_name, generate=True, timeout=240)
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous
        config.clear()
    root.mkdir(parents=True, exist_ok=True)
    digest = _write_rows(final, result['rows'])
    save_json(root / 'generation_provenance.json', dict(result['generation'],
        output_file=final.name, output_sha256=digest, output_count=30))
    return dict(status='generated', count=30, model=result['generation']['model'])


def prepare_proxy_memory(*, base, catalog_root, inputs_root, count_per_group=10):
    """Combine each category's random gallery into one common public bank."""
    root = validate_public_bundle_root(inputs_root / 'proxy_memory')
    groups = list(_json(base / 'manifest.json')['groups'])
    all_rows, forbidden_pixels, excluded_ids, input_binding = [], set(), set(), {}
    for group in groups:
        public, _ = _public(group, base)
        for role in ('source', 'target'):
            forbidden_pixels.add(decoded_image_sha256(public[role]['image']))
            excluded_ids.add(public[role].get('id'))
        group_root = validate_public_bundle_root(inputs_root / group)
        binding = _image_binding(public, group, catalog_root, count_per_group)
        rows = _verify_candidates(group_root, catalog_root, group, binding)
        input_binding[group] = dict(images=binding, candidates_sha256=sha256_file(group_root / 'candidates.jsonl'))
        all_rows.extend((group, group_root, row) for row in rows)
    ids, pixels = set(), set()
    for group, group_root, row in all_rows:
        decoded = decoded_image_sha256(group_root / row['image'])
        if row['id'] in ids or row['id'] in excluded_ids or decoded in pixels or decoded in forbidden_pixels:
            raise ValueError('共用代理库含重复商品、重复像素或公开源图和目标图')
        ids.add(row['id']); pixels.add(decoded)
    manifest_path = root / 'proxy_memory_manifest.json'
    final = root / 'candidates.jsonl'
    if manifest_path.exists():
        manifest = _json(manifest_path)
        if manifest.get('input_binding') != input_binding or manifest.get('candidates_sha256') != sha256_file(final):
            raise ValueError('共用代理库来源改变，禁止复用旧库')
        frozen = list(read_source_rows(final).values())
        if len(frozen) != len(all_rows) or {r['id'] for r in frozen} != ids:
            raise ValueError('共用代理库数量或商品编号不符')
        for row in frozen:
            image = (root / row['image']).resolve()
            if not image.is_relative_to(root) or sha256_file(image) != row['image_sha256']:
                raise ValueError('共用代理库图片路径或摘要不符')
        return dict(status='reused', count=len(frozen), group_count=len(groups))
    if root.exists() and any(root.iterdir()):
        raise ValueError('共用代理库目录非空但未完成冻结，请核查后使用新目录')
    (root / 'images').mkdir(parents=True, exist_ok=True)
    combined = []
    for group, group_root, row in all_rows:
        image = group_root / row['image']
        relative = 'images/' + row['image_sha256'] + image.suffix.casefold()
        shutil.copyfile(image, root / relative)
        combined.append(dict(row, image=relative, gallery_group=group))
    digest = _write_rows(final, combined)
    save_json(manifest_path, dict(schema_version=1, policy='shared_independent_public_nine_categories_v1',
        candidate_count=len(combined), count_per_category=count_per_group, category_groups=groups,
        input_binding=input_binding, candidates_sha256=digest,
        representation='frozen_image_only_candidates_shared_across_all_training_groups',
        formal_memory_contents_read=False, formal_queries_read=False, test_fixture=False))
    return dict(status='prepared', count=len(combined), group_count=len(groups))


def freeze_group(group, *, base, catalog_root, inputs_root, bundles_root):
    public, semantic = _public(group, base)
    root = validate_public_bundle_root(bundles_root / group)
    _catalog_snapshot(group, catalog_root)
    prepare_proxy_memory(base=base, catalog_root=catalog_root, inputs_root=inputs_root)
    generation = _json(inputs_root / group / 'paraphrases/generation_provenance.json')
    descriptions = list(read_source_rows(catalog_root / group / 'descriptions.jsonl').values())
    if generation.get('input_descriptions_sha256') != content_sha256(descriptions[:50]):
        raise ValueError('改写没有绑定当前五十条公开描述')
    paths = dict(category_description=catalog_root / group / 'descriptions.jsonl',
        model_paraphrase=inputs_root / group / 'paraphrases/model_paraphrases.jsonl',
        heldout_search_query=inputs_root / group / 'heldout_queries.jsonl',
        candidate_catalog=inputs_root / 'proxy_memory/candidates.jsonl')
    if generation.get('output_sha256') != sha256_file(paths['model_paraphrase']):
        raise ValueError('改写输出与实际生成记录不一致')
    for kind, count in (('category_description', 50), ('model_paraphrase', 30), ('heldout_search_query', 20)):
        if len(read_source_rows(paths[kind])) != count:
            raise ValueError('公开训练来源配额不足：' + kind)
    preparation_binding = dict(source_files={kind:sha256_file(path) for kind, path in paths.items()},
        sampling_manifest_sha256=sha256_file(catalog_root / group / 'sampling_manifest.json'),
        query_metadata_sha256=sha256_file(inputs_root / group / 'heldout_source_metadata.json'),
        generation_provenance_sha256=sha256_file(inputs_root / group / 'paraphrases/generation_provenance.json'),
        shared_proxy_memory_sha256=sha256_file(inputs_root / 'proxy_memory/proxy_memory_manifest.json'))
    if (root / 'training_bundle.json').exists():
        if _json(root / 'preparation_binding.json') != preparation_binding:
            raise ValueError('准备来源发生改变，不能把旧训练包当作新输入结果')
        bundle = load_training_bundle(root / 'training_bundle.json', source=public['source']['image'],
            target=public['target']['image'], public_semantic=semantic)
        return dict(status='reused', queries=100, candidates=len(bundle['candidates']))
    upstream = dict(dataset='WebShop public Amazon product catalog',
        revision=_json(catalog_root / group / 'sampling_manifest.json')['catalog_identity']['sha256'],
        url='https://github.com/princeton-nlp/WebShop')
    common = dict(test_fixture=False, is_synthetic=False, split='public_training', upstream=upstream)
    metadata = dict(category_description=dict(common), candidate_catalog=dict(common),
        model_paraphrase=dict(generation, test_fixture=False, is_synthetic=True,
            split='public_training', upstream=dict(dataset='MiniMax public-description query paraphrases',
                revision=generation['prompt_version'], url='https://platform.minimax.cn')),
        heldout_search_query=_json(inputs_root / group / 'heldout_source_metadata.json'))
    bundle = prepare_training_bundle(source=public['source']['image'], target=public['target']['image'],
        public_semantic=semantic, descriptions=catalog_root / group / 'descriptions.jsonl',
        paraphrases=inputs_root / group / 'paraphrases/model_paraphrases.jsonl',
        heldout_queries=inputs_root / group / 'heldout_queries.jsonl',
        candidates=paths['candidate_catalog'], source_metadata=metadata,
        output_dir=root, paraphrase_policy='declared_model')
    save_json(root / 'preparation_binding.json', preparation_binding)
    return dict(status='frozen', queries=100, candidates=len(bundle['candidates']),
                manifest_sha256=bundle['manifest_sha256'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('images', 'proxy-memory', 'paraphrases', 'freeze'))
    parser.add_argument('--group', default='all')
    parser.add_argument('--base', type=Path, default=PROJECT / 'resources/evaluation_base')
    parser.add_argument('--catalog-root', type=Path, default=PROJECT / 'data/public/visual_inception_inputs/catalog_v5')
    parser.add_argument('--inputs-root', type=Path, default=PROJECT / 'data/public/visual_inception_inputs_v5')
    parser.add_argument('--bundles-root', type=Path, default=PROJECT / 'data/public/visual_inception')
    parser.add_argument('--local-service-config', type=Path, default=PROJECT / 'src/agent/config/llm.local.json')
    args = parser.parse_args(argv)
    if args.stage == 'proxy-memory':
        result = prepare_proxy_memory(base=args.base, catalog_root=args.catalog_root, inputs_root=args.inputs_root)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    groups = list(_json(args.base / 'manifest.json')['groups']) if args.group == 'all' else args.group.split(',')
    results = {}
    for group in groups:
        try:
            if args.stage == 'images':
                result = prepare_images(group, base=args.base, catalog_root=args.catalog_root, inputs_root=args.inputs_root)
            elif args.stage == 'paraphrases':
                result = prepare_paraphrases(group, catalog_root=args.catalog_root, inputs_root=args.inputs_root,
                                            local_service_config=args.local_service_config)
            else:
                result = freeze_group(group, base=args.base, catalog_root=args.catalog_root,
                    inputs_root=args.inputs_root, bundles_root=args.bundles_root)
        except Exception as exc:
            # Only fixed local errors from our adapter are printed, without HTTP bodies.
            result = dict(status='failed', error_type=type(exc).__name__, reason=str(exc))
        results[group] = result
        print(json.dumps(dict(stage=args.stage, group=group, **result), ensure_ascii=False), flush=True)
        args.inputs_root.mkdir(parents=True, exist_ok=True)
        save_json(args.inputs_root / (args.stage + '_status.json'), results)
    return int(any(r['status'] == 'failed' for r in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())
