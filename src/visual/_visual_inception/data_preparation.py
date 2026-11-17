"""Explicit offline imports and an opt-in, credential-safe paraphrase adapter.

MiniMax compatibility reference:
https://platform.minimax.cn/docs/api-reference/text-openai-api
M3-family compatibility endpoints ignore temperature. The unsupported parameter
is omitted entirely; only the service's fixed effective value is recorded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from urllib import request, error
from urllib.parse import urlsplit

from src.visual.io import save_json, sha256_file
from .targets import content_sha256, text_sha256
from .training_data import (QUERY_COUNTS, SPLIT_POLICY, build_target_concept_text,
    canonical_source_type, load_training_bundle, normalized_text, read_source_rows, reject_private_fields,
    _assert_public_path, validate_public_bundle_root)

DEFAULT_BASE_URL = 'https://api.minimax.cn/v1'
PARAPHRASE_PROMPT_VERSION = 'public_description_query_paraphrases_v2_one_per_selected_input'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def completion_endpoint(base_url):
    _require(isinstance(base_url, str), '模型服务地址必须为字符串')
    parsed = urlsplit(base_url)
    _require(parsed.scheme == 'https' and bool(parsed.hostname) and not parsed.username
             and not parsed.password and not parsed.query and not parsed.fragment,
             '模型地址必须使用不含凭据或查询参数的加密连接')
    clean = base_url.rstrip('/')
    return clean if clean.endswith('/chat/completions') else clean + '/chat/completions'


def _post_json(url, payload, api_key, timeout):
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    http_request = request.Request(url, data=data, headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key}, method='POST')
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            body = response.read(2 * 1024 * 1024 + 1)
    except (error.URLError, error.HTTPError, TimeoutError, OSError):
        raise RuntimeError('模型改写请求失败；请检查服务地址、模型身份和本机凭据配置') from None
    _require(len(body) <= 2 * 1024 * 1024, '模型响应超出允许大小')
    try:
        result = json.loads(body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError('模型服务没有返回有效结构化响应') from None
    _require(isinstance(result, dict), '模型服务响应必须为对象')
    return result


def generate_paraphrases(descriptions, *, provider='minimax', model, base_url=DEFAULT_BASE_URL,
        api_key_env='MINIMAX_API_KEY', generate=False, count=30, temperature=0.8,
        request_fn=None, timeout=120):
    """Return a plan by default; only explicit generate=True reads a key/calls.

    A test may inject request_fn(url, payload, key, timeout) without any network.
    The returned data never include the credential or HTTP authorization header.
    """
    _require(provider in ('minimax', 'openai'), '仅支持明确登记的模型服务商')
    _require(isinstance(model, str) and bool(model.strip()), '必须显式指定实际改写模型，不自动替换')
    _require(temperature == 0.8 and not isinstance(temperature, bool), '改写请求温度固定为零点八')
    _require(type(count) is int and count == 30, '查询配额要求三十条模型改写')
    _require(isinstance(api_key_env, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', api_key_env),
             '模型凭据环境变量名称无效')
    if isinstance(descriptions, (str, Path)):
        descriptions = list(read_source_rows(descriptions).values())
    _require(isinstance(descriptions, list) and len(descriptions) >= 50, '先提供至少五十条公开描述')
    selected = descriptions[:50]
    _require(all(isinstance(row, dict) and row.get('id') is not None for row in selected), '描述缺少来源编号')
    ids = {str(row['id']) for row in selected}
    _require(len(ids) == 50, '公开描述来源编号必须不同')
    normalized = [normalized_text(row.get('text')) for row in selected]
    _require(len(set(normalized)) == 50, '公开描述文字必须不同')
    categories = {normalized_text(row.get('category')) for row in selected}
    _require(len(categories) == 1, '改写输入必须来自同一目标品类')
    # The descriptions already have a frozen randomized order. Sending exactly
    # thirty selected inputs avoids the model rewriting all fifty pool entries
    # despite a thirty-output instruction. All fifty remain retrieval queries.
    rewrite_inputs = selected[:count]
    rewrite_ids = {str(row['id']) for row in rewrite_inputs}
    endpoint = completion_endpoint(base_url)
    fixed_temperature = provider == 'minimax' and model.casefold().startswith('minimax-m3')
    temperature_identity = dict(effective_temperature=1.0 if fixed_temperature else 0.8,
        temperature_supported=not fixed_temperature, temperature_parameter_sent=not fixed_temperature)
    if not fixed_temperature:
        temperature_identity['temperature'] = 0.8
    plan = dict(status='planned', network_request_performed=False, provider=provider,
        requested_model=model, endpoint=endpoint, count=30, **temperature_identity,
        input_descriptions_sha256=content_sha256(selected),
        paraphrase_input_row_ids=[str(row['id']) for row in rewrite_inputs],
        prompt_version=PARAPHRASE_PROMPT_VERSION, api_key_environment_variable=api_key_env)
    if not generate:
        return plan
    api_key = os.environ.get(api_key_env)
    _require(isinstance(api_key, str) and bool(api_key.strip()), '缺少模型凭据环境变量；尚未发送请求')
    payload = dict(model=model, stream=False, max_completion_tokens=8192,
        messages=[dict(role='system', content=(
            'Rewrite each of the 30 supplied public product descriptions as ONE natural future shopping query. '
            'Return exactly one query for each input, in the same order: exactly 30 objects total. '
            'Use only supported product attributes. Do not invent brands, facts, or purchase history. '
            'Return ONLY a JSON array of exactly 30 objects with keys input_row_id and text. '
            'Each input_row_id must refer to one supplied description. Do not repeat input text. '
            'No markdown, commentary, or reasoning in the final answer.')),
            dict(role='user', content=json.dumps([dict(input_row_id=str(row['id']), text=row['text'])
                for row in rewrite_inputs], ensure_ascii=False))])
    if not fixed_temperature:
        payload['temperature'] = 0.8
    # This structured rewrite task does not need a reasoning transcript. M3
    # explicitly supports disabling it; M3.1 requires adaptive thinking, so do
    # not apply this setting to the rest of the family.
    if provider == 'minimax' and model.casefold() == 'minimax-m3':
        payload['thinking'] = {'type': 'disabled'}
    try:
        response = (request_fn or _post_json)(endpoint, payload, api_key, timeout)
    except Exception:
        raise RuntimeError('模型改写请求失败；未保存服务错误正文或凭据') from None
    _require(isinstance(response, dict), '模型服务响应无效')
    base_response = response.get('base_resp', {})
    _require(isinstance(base_response, dict) and not response.get('error') and base_response.get('status_code', 0) == 0,
             '模型服务返回失败状态；不会保存错误正文')
    actual_model = response.get('model')
    _require(isinstance(actual_model, str) and bool(actual_model.strip()), '服务未返回实际模型身份')
    if provider == 'minimax':
        _require(actual_model.casefold() == model.casefold(),
                 '服务实际模型与指定模型不同；不自动替换模型或冒称请求参数生效')
    choices = response.get('choices')
    _require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
             and choices[0].get('finish_reason') == 'stop', '模型改写响应不完整，不能冻结截断结果')
    message = choices[0].get('message', {})
    content = message.get('content') if isinstance(message, dict) else None
    _require(isinstance(content, str) and bool(content.strip()), '模型没有返回可用改写正文')
    _require(api_key not in content, '模型改写正文包含私有凭据，拒绝保存')
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError('模型改写正文必须为三十条结构化记录，不能包含思考或解释') from None
    _require(isinstance(result, list) and len(result) == 30,
             '模型改写数量不等于三十，实际：' + (str(len(result)) if isinstance(result, list) else '非列表'))
    seen, rows = set(normalized), []
    for index, row in enumerate(result, 1):
        _require(isinstance(row, dict) and str(row.get('input_row_id')) in rewrite_ids, '模型改写缺少有效输入来源编号')
        norm = normalized_text(row.get('text'))
        _require(norm not in seen, '模型改写重复或照抄描述，不能凑配额')
        seen.add(norm)
        rows.append(dict(id='paraphrase_' + str(index).zfill(3), text=row['text'],
            category=selected[0]['category'], input_row_id=str(row['input_row_id']),
            provider=provider, model=actual_model, requested_model=model, **temperature_identity))
    response_id = response.get('id')
    if (not isinstance(response_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}', response_id)
            or api_key in response_id):
        response_id = None
    public_identity = dict(provider=provider, model=actual_model, requested_model=model,
        **temperature_identity,
        endpoint=endpoint, response_id=response_id, prompt_version=PARAPHRASE_PROMPT_VERSION,
        input_descriptions_sha256=plan['input_descriptions_sha256'],
        paraphrase_input_row_ids=plan['paraphrase_input_row_ids'],
        generated_at=datetime.now(timezone.utc).isoformat(),
        paper_query_generator_reproduced=False)
    if 'thinking' in payload:
        public_identity['thinking_mode'] = payload['thinking']['type']
    return dict(status='generated', network_request_performed=True, rows=rows,
                generation=public_identity)


def _write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    return sha256_file(path)


def prepare_training_bundle(*, source, target, public_semantic, descriptions, paraphrases,
        heldout_queries, candidates, source_metadata, output_dir, paraphrase_policy='declared_model',
        test_fixture=False):
    """Copy finite public snapshots/assets into a new immutable bundle.

    Inputs are JSONL paths. source_metadata maps the four canonical source kinds
    to their metadata dicts; all data origin statements must be explicit imports.
    Candidate image paths are relative to the candidate-catalog input file.
    """
    _require(type(test_fixture) is bool, '训练包夹具开关必须为布尔值')
    root = validate_public_bundle_root(output_dir, require_production=not test_fixture)
    _require(not root.exists() or not any(root.iterdir()), '训练包目录非空，禁止覆盖冻结数据')
    paths = {'category_description': Path(descriptions), 'model_paraphrase': Path(paraphrases),
             'heldout_search_query': Path(heldout_queries), 'candidate_catalog': Path(candidates)}
    loaded = {kind: list(read_source_rows(path).values()) for kind, path in paths.items()}
    for kind, count in QUERY_COUNTS.items():
        _require(len(loaded[kind]) >= count, '训练来源配额不足：' + kind)
        loaded[kind] = loaded[kind][:count]
    _require(bool(loaded['candidate_catalog']), '公开候选目录为空')
    _require(isinstance(source_metadata, dict) and set(source_metadata) == set(paths), '必须为四种来源提供完整来源元数据')
    reject_private_fields(source_metadata)
    root.mkdir(parents=True, exist_ok=True)
    records, manifest_sources, query_entries, candidate_entries = {}, {}, [], []
    for kind, rows in loaded.items():
        source_id = kind
        metadata = dict(source_metadata[kind])
        _require(canonical_source_type(metadata.get('kind', kind)) == kind, '来源元数据类型与输入文件不一致')
        _require(metadata.get('test_fixture') is test_fixture, '导入来源的测试夹具身份不可重贴标签')
        if kind == 'candidate_catalog':
            copied_rows = []
            for index, row in enumerate(rows, 1):
                image = Path(row.get('image', ''))
                image = (image if image.is_absolute() else paths[kind].resolve().parent / image).resolve()
                _assert_public_path(image)
                _require(image.is_file(), '公开候选来源图片不存在')
                digest = sha256_file(image)
                _require(row.get('image_sha256') == digest, '候选输入摘要不符')
                suffix = image.suffix.casefold()
                _require(suffix in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'), '候选图片格式无效')
                relative = 'images/' + str(index).zfill(5) + suffix
                (root / 'images').mkdir(exist_ok=True)
                shutil.copyfile(image, root / relative)
                copied_rows.append(dict(row, image=relative))
                candidate_entries.append(dict(id=str(row['id']), image=relative, image_sha256=digest,
                    provenance=dict(source_id=source_id, row_id=str(row['id']))))
            rows = copied_rows
        relative = 'sources/' + kind + '.jsonl'
        digest = _write_rows(root / relative, rows)
        metadata.update(path=relative, sha256=digest, kind=kind)
        manifest_sources[source_id] = metadata
        records[kind] = rows
        if kind in QUERY_COUNTS:
            for index, row in enumerate(rows, 1):
                query_entries.append(dict(id=kind + '_' + str(index).zfill(3), text=row['text'],
                    source_type=kind, provenance=dict(source_id=source_id, row_id=str(row['id']),
                        text_sha256=text_sha256(row['text']))))
    manifest = dict(schema_version=1, test_fixture=test_fixture, category=public_semantic['category'],
        target_binding=dict(source_image_sha256=sha256_file(source), target_image_sha256=sha256_file(target),
                            canonical_text_sha256=text_sha256(public_semantic['canonical_text'])),
        target_concept_text=build_target_concept_text(public_semantic),
        paraphrase_policy=paraphrase_policy, queries=query_entries, candidates=candidate_entries,
        sources=manifest_sources, formal_queries_used=False, victim_memory_used=False,
        split_manifest=dict(policy=SPLIT_POLICY, evaluation_partition_excluded=True,
            formal_query_sources_used=False,
            heldout_query_ids=[q['id'] for q in query_entries if q['source_type'] == 'heldout_search_query']))
    manifest_path = root / 'training_bundle.json'
    save_json(manifest_path, manifest)
    bundle = load_training_bundle(manifest_path, source=source, target=target,
        public_semantic=public_semantic, require_production=not test_fixture)
    return bundle
