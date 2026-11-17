"""Strict, self-contained public training bundles, independent of evaluation.

The loader verifies frozen source rows; declarations about the upstream corpus
remain explicit provenance, not a claim that arbitrary imported text is genuine
user traffic. Synthetic fixtures are accepted only through the explicit test
gate. No service calls or credential reads occur in this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from src.paths import PROJECT, DATA, OUTPUT, RESOURCES
from src.visual.io import sha256_file
from .targets import content_sha256, text_sha256

SCHEMA_VERSION = 1
SPLIT_POLICY = 'independent_public_training_query_reserve_v1'
QUERY_COUNTS = {'category_description': 50, 'model_paraphrase': 30, 'heldout_search_query': 20}
SOURCE_TYPES = frozenset((*QUERY_COUNTS, 'candidate_catalog'))
PARAPHRASE_POLICIES = ('declared_model', 'paper_gpt4')
GPT4_MODELS = frozenset(('gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-32k',
    'gpt-4-32k-0314', 'gpt-4-32k-0613', 'gpt-4-1106-preview', 'gpt-4-0125-preview',
    'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4-turbo-2024-04-09'))
MAX_SOURCE_BYTES = 64 * 1024 * 1024
FORBIDDEN_PRIVATE_FIELDS = frozenset(('api_key', 'apikey', 'authorization', 'access_token',
    'refresh_token', 'password', 'secret', 'request_headers', 'reasoning_content'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_text(text):
    _require(isinstance(text, str) and bool(text.strip()), '训练文字必须为非空字符串')
    return ' '.join(text.split()).casefold()


def reject_private_fields(value):
    """Reject obvious credential/hidden-response fields without showing values."""
    if isinstance(value, dict):
        _require(not any(str(key).casefold().replace('-', '_') in FORBIDDEN_PRIVATE_FIELDS for key in value),
                 '公开训练输入不得包含凭据或模型思考字段')
        for item in value.values():
            reject_private_fields(item)
    elif isinstance(value, list):
        for item in value:
            reject_private_fields(item)


def canonical_source_type(value):
    return 'model_paraphrase' if value == 'gpt4_paraphrase' else value


def build_target_concept_text(public_semantic):
    """One category/attribute concept; never an average of future queries."""
    category = public_semantic.get('category')
    attributes = public_semantic.get('core_attributes')
    _require(isinstance(category, str) and bool(category.strip())
             and isinstance(attributes, list) and bool(attributes)
             and all(isinstance(v, str) and bool(v.strip()) for v in attributes),
             '目标概念必须由冻结的公开品类和核心特点构造')
    return category.strip() + ': ' + '; '.join(v.strip() for v in attributes)


def _assert_public_path(path):
    forbidden_roots = (PROJECT / 'resources/evaluation_base/shared_memory',
        PROJECT / 'resources/evaluation_base/evaluation_inputs',
        PROJECT / 'resources/evaluation_base/evaluation_inputs_joint',
        PROJECT / 'resources/evaluation_base/evaluation_inputs_vague_joint',
        PROJECT / 'resources/evaluation_base/query_images',
        PROJECT / 'resources/visual_inception_evaluation', OUTPUT / 'datasets/private')
    _require(not any(path == p.resolve() or path.is_relative_to(p.resolve()) for p in forbidden_roots),
             '训练包禁止读取正式查询、共享记忆或私人数据')
    name = path.name.casefold()
    _require(name not in ('llm.local.json', 'queries.json', 'evaluation.json', 'results.json',
                          'test_results.json', 'formal_queries.py', 'category_evaluation_plan.json',
                          'category_query_validation.json', 'category_retrieval_summary.json')
             and not (path.suffix.casefold() in ('.json', '.jsonl')
                      and name.startswith(('memory', 'history', 'shared_memory', 'clean_memory', 'adversarial_memory'))),
             '训练包文件名指向禁止的正式或私有输入')


def validate_public_bundle_root(root, *, require_production=True):
    root = Path(root).expanduser().resolve()
    _assert_public_path(root)
    if require_production:
        public_roots = (DATA / 'public', RESOURCES / 'public_training', OUTPUT / 'datasets/public')
        _require(any(root == p.resolve() or root.is_relative_to(p.resolve()) for p in public_roots),
                 '正式训练包必须放在本项目独立公开训练目录')
    return root


def bundle_path(root, value, label='训练包文件'):
    _require(isinstance(value, str) and bool(value.strip()), label + '缺少相对路径')
    relative = Path(value)
    _require(not relative.is_absolute() and not relative.drive and '..' not in relative.parts,
             label + '必须为训练包内部相对路径')
    path = (root / relative).resolve()
    _require(path != root and path.is_relative_to(root), label + '路径越界')
    _assert_public_path(path)
    _require(path.is_file(), label + '不存在')
    return path


def read_source_rows(path):
    path = Path(path).resolve()
    _assert_public_path(path)
    _require(path.suffix.casefold() == '.jsonl', '训练来源快照必须使用逐行对象文件')
    _require(path.stat().st_size <= MAX_SOURCE_BYTES, '训练来源快照过大；请先导出有限公开子集')
    rows = {}
    with path.open('r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            _require(line_number <= 100000, '训练来源快照记录过多')
            try:
                row = json.loads(line)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError('训练来源快照含无效对象，行号：' + str(line_number)) from error
            _require(isinstance(row, dict) and isinstance(row.get('id'), (str, int)),
                     '来源行必须提供明确记录编号')
            reject_private_fields(row)
            identifier = str(row['id'])
            _require(bool(identifier.strip()) and identifier not in rows, '来源行编号为空或重复')
            rows[identifier] = row
    _require(bool(rows), '训练来源快照为空')
    return rows


def _source_registry(manifest, root, fixture):
    sources = manifest.get('sources')
    _require(isinstance(sources, dict) and bool(sources), '训练包缺少来源登记')
    loaded, audit = {}, []
    for source_id, source in sources.items():
        _require(isinstance(source_id, str) and source_id.strip() and isinstance(source, dict), '来源登记无效')
        kind = canonical_source_type(source.get('kind'))
        _require(kind in SOURCE_TYPES, '训练来源类型无效')
        _require(source.get('test_fixture') is fixture, '来源夹具标记与训练包不一致')
        expected_split = 'query_reserve' if kind == 'heldout_search_query' else 'public_training'
        _require(source.get('split') == expected_split, '训练数据和预留查询的划分无效')
        upstream = source.get('upstream', {})
        _require(isinstance(upstream, dict) and all(isinstance(upstream.get(k), str) and upstream[k].strip()
                    for k in ('dataset', 'revision', 'url')), '来源必须登记公开语料、版本和地址')
        url = urlsplit(upstream['url'])
        _require(url.scheme == 'https' and bool(url.hostname) and not url.username and not url.password
                 and not url.query and not url.fragment, '公开来源地址不可包含凭据或查询参数')
        if not fixture and kind != 'model_paraphrase':
            _require(source.get('is_synthetic') is False, '正式描述、候选或预留搜索查询不能来自合成数据')
        if not fixture and kind == 'heldout_search_query':
            _require(source.get('origin_type') == 'public_real_search_queries',
                     '预留查询必须明确来自独立公开真实搜索语料，不能冒称查询日志')
        path = bundle_path(root, source.get('path'), '来源快照')
        digest = sha256_file(path)
        _require(digest == source.get('sha256'), '训练来源快照摘要不符')
        rows = read_source_rows(path)
        _require(sha256_file(path) == digest, '读取训练来源时文件发生变化')
        loaded[source_id] = (dict(source, kind=kind), rows)
        audit.append(dict(source_id=source_id, path=str(path), sha256=digest, kind=kind,
            split=expected_split, upstream=upstream, test_fixture=fixture,
            is_synthetic=source.get('is_synthetic')))
    return loaded, audit


def _provenance_row(item, loaded, expected_kind):
    provenance = item.get('provenance', {})
    _require(isinstance(provenance, dict), '训练记录来源必须为对象')
    source_id, row_id = provenance.get('source_id'), provenance.get('row_id')
    _require(source_id in loaded and isinstance(row_id, (str, int)), '训练记录缺少可核验来源行')
    source, rows = loaded[source_id]
    _require(source['kind'] == expected_kind and str(row_id) in rows, '训练记录来源类型或行编号不符')
    return source_id, source, rows[str(row_id)], provenance


def _generation_identity(source, row, policy, *, fixture=False):
    provider, model = source.get('provider'), source.get('model')
    _require(isinstance(provider, str) and provider.strip() and isinstance(model, str) and model.strip(),
             '改写来源必须记录实际服务商和模型名称')
    _require(provider == row.get('provider') and model == row.get('model'), '改写来源和实际返回模型身份不一致')
    supported, sent = source.get('temperature_supported'), source.get('temperature_parameter_sent')
    effective = source.get('effective_temperature')
    _require(type(supported) is bool and type(sent) is bool and not isinstance(effective, bool)
             and isinstance(effective, (int, float)) and math.isfinite(effective) and 0 < effective <= 2
             and row.get('temperature_supported') is supported
             and row.get('temperature_parameter_sent') is sent and row.get('effective_temperature') == effective,
             '改写实际温度或服务支持声明不一致')
    if supported:
        _require(sent and source.get('temperature') == 0.8 and row.get('temperature') == 0.8 and effective == 0.8,
                 '支持温度的模型必须明确使用零点八')
    else:
        _require(not sent and all(key not in source and key not in row for key in ('temperature', 'requested_temperature')),
                 '不支持温度的模型必须取消该参数及其请求记录')
    if provider.casefold() == 'minimax' and model.casefold().startswith('minimax-m3'):
        _require(supported is False and sent is False and effective == 1.0,
                 '该系列模型固定温度为一，须省略不支持的温度参数')
    if policy == 'paper_gpt4':
        _require(provider.casefold() == 'openai' and model in GPT4_MODELS and supported and effective == 0.8,
                 '论文指定改写模式只接受明确的第四代模型身份，其他模型不能冒名')
    identity = dict(provider=provider, model=model, effective_temperature=effective,
                    temperature_supported=supported, temperature_parameter_sent=sent)
    if supported:
        identity['temperature'] = 0.8
    return identity


def decoded_image_sha256(path):
    """Content identity excludes container compression and metadata changes."""
    from PIL import Image
    with Image.open(path) as opened:
        rgb = opened.convert('RGB')
        digest = hashlib.sha256((str(rgb.width) + ',' + str(rgb.height) + ',RGB\0').encode('ascii'))
        digest.update(rgb.tobytes())
    return digest.hexdigest()


def load_training_bundle(manifest_path, *, source, target, public_semantic, require_production=True):
    """Verify a frozen 50/30/20 query set and image-only public negative pool.

    Source snapshots and candidate images must resolve beneath the package
    directory; therefore a caller may grant precisely that one read root.
    """
    path = Path(manifest_path).expanduser().resolve()
    _assert_public_path(path)
    _require(path.is_file(), '缺少冻结公开训练包清单')
    raw_sha = sha256_file(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    _require(isinstance(manifest, dict) and manifest.get('schema_version') == SCHEMA_VERSION, '训练包版本无效')
    reject_private_fields(manifest)
    fixture = manifest.get('test_fixture')
    _require(type(fixture) is bool, '训练包必须显式声明是否为测试夹具')
    _require(not require_production or not fixture, '微型合成训练包不能用于生产成图')
    root = validate_public_bundle_root(path.parent, require_production=require_production)
    _require(manifest.get('formal_queries_used') is False and manifest.get('victim_memory_used') is False,
             '训练包不得使用正式询问或受害者共享记忆')
    category = public_semantic.get('category')
    _require(isinstance(category, str) and normalized_text(manifest.get('category')) == normalized_text(category),
             '训练包与冻结目标的公开品类不一致')
    binding = dict(source_image_sha256=sha256_file(source), target_image_sha256=sha256_file(target),
        canonical_text_sha256=text_sha256(public_semantic['canonical_text']))
    _require(manifest.get('target_binding') == binding, '训练包未绑定本次冻结公开图对和目标文字')
    concept = build_target_concept_text(public_semantic)
    _require(manifest.get('target_concept_text') == concept, '目标概念必须是独立的公开品类特点文字，不能用未来查询替代')
    policy = manifest.get('paraphrase_policy', 'declared_model')
    _require(policy in PARAPHRASE_POLICIES, '未知改写模型身份策略')
    queries = manifest.get('queries')
    _require(isinstance(queries, list) and len(queries) == 100, '每个训练包必须包含一百条查询')
    _require(all(isinstance(q, dict) for q in queries), '查询必须附带来源记录')
    counts = Counter(canonical_source_type(q.get('source_type')) for q in queries)
    _require(dict(counts) == QUERY_COUNTS, '查询来源配额必须为五十条描述、三十条模型改写和二十条预留搜索查询')
    normalized = [normalized_text(q.get('text')) for q in queries]
    _require(len(set(normalized)) == 100, '一百条查询必须互不重复，不能重复模板凑配额')
    query_ids = [q.get('id') for q in queries]
    _require(all(isinstance(v, str) and v.strip() for v in query_ids) and len(set(query_ids)) == 100,
             '查询编号必须为一百个不同的非空字符串')
    split = manifest.get('split_manifest', {})
    heldout_ids = [q['id'] for q in queries if canonical_source_type(q['source_type']) == 'heldout_search_query']
    _require(isinstance(split, dict) and split.get('policy') == SPLIT_POLICY
             and split.get('evaluation_partition_excluded') is True and split.get('formal_query_sources_used') is False
             and isinstance(split.get('heldout_query_ids'), list)
             and len(split['heldout_query_ids']) == 20 and set(split['heldout_query_ids']) == set(heldout_ids),
             '训练查询、预留公开搜索语料与正式评估的隔离声明无效')
    loaded, source_audit = _source_registry(manifest, root, fixture)
    descriptions = set()
    for query in queries:
        if canonical_source_type(query['source_type']) == 'category_description':
            provenance = query.get('provenance', {})
            descriptions.add(str(provenance.get('row_id')))
    generators, provenance_audit = [], []
    for query in queries:
        kind = canonical_source_type(query['source_type'])
        source_id, source_record, row, provenance = _provenance_row(query, loaded, kind)
        text = query['text']
        _require(row.get('text') == text and provenance.get('text_sha256') == text_sha256(text),
                 '查询文字与冻结来源行或文字摘要不一致')
        if kind in QUERY_COUNTS:
            _require(normalized_text(row.get('category')) == normalized_text(category), '查询描述或改写与目标品类不符')
        if kind == 'model_paraphrase':
            identity = _generation_identity(source_record, row, policy, fixture=fixture)
            _require(str(row.get('input_row_id')) in descriptions, '模型改写未引用本次公开描述来源')
            if identity not in generators:
                generators.append(identity)
        if kind == 'heldout_search_query':
            _require(row.get('split') == 'query_reserve', '预留搜索查询行的划分身份无效')
        provenance_audit.append(dict(id=query['id'], source_type=kind, source_id=source_id,
            row_id=str(provenance['row_id']), text_sha256=text_sha256(text)))
    candidates = manifest.get('candidates')
    _require(isinstance(candidates, list) and len(candidates) >= 2, '训练包必须提供至少两张独立公开候选图片')
    source_pixels, target_pixels = decoded_image_sha256(source), decoded_image_sha256(target)
    candidate_ids, candidate_hashes, candidate_pixels, resolved_candidates = set(), set(), set(), []
    for candidate in candidates:
        _require(isinstance(candidate, dict), '候选图片条目必须为对象')
        identifier = candidate.get('id')
        _require(isinstance(identifier, str) and bool(identifier.strip()) and identifier not in candidate_ids,
                 '候选商品编号为空或重复')
        candidate_ids.add(identifier)
        _, _, row, _ = _provenance_row(candidate, loaded, 'candidate_catalog')
        image = bundle_path(root, candidate.get('image'), '候选图片')
        image_hash = sha256_file(image)
        _require(image_hash == candidate.get('image_sha256') == row.get('image_sha256'), '候选图片与公开来源行摘要不一致')
        _require(image_hash not in (binding['source_image_sha256'], binding['target_image_sha256'])
                 and image_hash not in candidate_hashes, '固定候选不能重复，不能包含当前源图或目标商品图')
        candidate_hashes.add(image_hash)
        pixels = decoded_image_sha256(image)
        _require(pixels not in (source_pixels, target_pixels) and pixels not in candidate_pixels,
                 '候选解码像素重复，不能通过重新压缩把源图或目标图作为负例')
        candidate_pixels.add(pixels)
        resolved_candidates.append(dict(id=identifier, image_path=str(image), image_sha256=image_hash,
                                       decoded_image_sha256=pixels))
    _require(sha256_file(path) == raw_sha, '读取训练包时清单发生变化')
    bundle_sha = content_sha256(manifest)
    audit = dict(schema_version=1, kind='visual_inception_independent_public_training_bundle_v1',
        manifest_path=str(path), manifest_sha256=raw_sha, bundle_sha256=bundle_sha,
        target_binding=binding, target_concept_text_sha256=text_sha256(concept),
        target_concept_format='category_colon_public_core_attributes_v1', category=category,
        source_counts=dict(counts), query_source_counts=dict(counts), query_count=100, query_batch_size=16,
        query_texts_sha256=content_sha256([q['text'] for q in queries]), query_provenance=provenance_audit,
        source_files=source_audit, candidate_count=len(resolved_candidates),
        candidate_entries_sha256=content_sha256(resolved_candidates), split_manifest=split,
        paraphrase_policy=policy, paraphrase_generators=generators,
        paper_query_generator_reproduced=not fixture and policy == 'paper_gpt4',
        paper_query_distribution_reproduced=not fixture and policy == 'paper_gpt4',
        test_fixture=fixture, formal_queries_used=False, victim_memory_used=False,
        candidate_representation='image_only_independent_public_candidates',
        target_image_used_for_optimization=False,
        log_origin_evidence='frozen_upstream_declaration_and_source_rows_not_independent_authenticity_verification')
    audit['audit_sha256'] = content_sha256(audit)
    return dict(target_concept_text=concept, queries=[q['text'] for q in queries],
        candidates=resolved_candidates, audit=audit, bundle_sha256=bundle_sha,
        manifest_path=str(path), manifest_sha256=raw_sha, allowed_roots=[str(root)])
