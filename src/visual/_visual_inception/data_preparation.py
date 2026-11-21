"""Explicit offline imports and an opt-in, credential-safe paraphrase adapter.

MiniMax compatibility reference:
https://platform.minimax.cn/docs/api-reference/text-openai-api
M3-family compatibility endpoints ignore temperature. The unsupported parameter
is omitted entirely; only the service's fixed effective value is recorded.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from urllib import request, error
from urllib.parse import urlsplit

from src.experiments.exact_overlap_queries import DuplicateIndex, normalize as overlap_normalize
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
    _require(type(count) is int and count == 30, '查询配额要求三十条模型改写')
    return _generate_paraphrases(descriptions, provider=provider, model=model, base_url=base_url,
        api_key_env=api_key_env, generate=generate, count=count, input_count=50,
        prompt_version=PARAPHRASE_PROMPT_VERSION, temperature=temperature,
        request_fn=request_fn, timeout=timeout)


def generate_evaluation_paraphrases(descriptions, *, model, provider='minimax', base_url=DEFAULT_BASE_URL,
        api_key_env='MINIMAX_API_KEY', generate=False, request_fn=None, timeout=120,
        target_char_range=None):
    """Use six independently selected evaluation rows with the public paraphrase adapter."""
    low, high = target_char_range if target_char_range is not None else (110, 190)
    guidance = (
        f' Explicit category naming and concrete factual attributes are required; aim for {low}-{high} '
        'characters matching detailed multi-attribute shopping queries without inventing brands or facts.'
    )
    return _generate_paraphrases(descriptions, provider=provider, model=model, base_url=base_url,
        api_key_env=api_key_env, generate=generate, count=6, input_count=6,
        prompt_version='public_description_query_paraphrases_eval_v4', temperature=0.8,
        request_fn=request_fn, timeout=timeout, extra_guidance=guidance)


def _generate_paraphrases(descriptions, *, provider, model, base_url, api_key_env, generate,
        count, input_count, prompt_version, temperature, request_fn, timeout, extra_guidance=''):
    _require(provider in ('minimax', 'openai'), '仅支持明确登记的模型服务商')
    _require(isinstance(model, str) and bool(model.strip()), '必须显式指定实际改写模型，不自动替换')
    _require(temperature == 0.8 and not isinstance(temperature, bool), '改写请求温度固定为零点八')
    _require(isinstance(api_key_env, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', api_key_env),
             '模型凭据环境变量名称无效')
    if isinstance(descriptions, (str, Path)):
        descriptions = list(read_source_rows(descriptions).values())
    _require(isinstance(descriptions, list) and len(descriptions) >= input_count, f'先提供至少 {input_count} 条公开描述')
    selected = descriptions[:input_count]
    _require(all(isinstance(row, dict) and row.get('id') is not None for row in selected), '描述缺少来源编号')
    ids = {str(row['id']) for row in selected}
    _require(len(ids) == input_count, '公开描述来源编号必须不同')
    normalized = [normalized_text(row.get('text')) for row in selected]
    _require(len(set(normalized)) == input_count, '公开描述文字必须不同')
    categories = {normalized_text(row.get('category')) for row in selected}
    _require(len(categories) == 1, '改写输入必须来自同一目标品类')
    # Preserve the frozen input order and submit exactly the requested number of rows.
    rewrite_inputs = selected[:count]
    rewrite_ids = {str(row['id']) for row in rewrite_inputs}
    endpoint = completion_endpoint(base_url)
    fixed_temperature = provider == 'minimax' and model.casefold().startswith('minimax-m3')
    temperature_identity = dict(effective_temperature=1.0 if fixed_temperature else 0.8,
        temperature_supported=not fixed_temperature, temperature_parameter_sent=not fixed_temperature)
    if not fixed_temperature:
        temperature_identity['temperature'] = 0.8
    plan = dict(status='planned', network_request_performed=False, provider=provider,
        requested_model=model, endpoint=endpoint, count=count, **temperature_identity,
        input_descriptions_sha256=content_sha256(selected),
        paraphrase_input_row_ids=[str(row['id']) for row in rewrite_inputs],
        prompt_version=prompt_version, api_key_environment_variable=api_key_env)
    if not generate:
        return plan
    api_key = os.environ.get(api_key_env)
    _require(isinstance(api_key, str) and bool(api_key.strip()), '缺少模型凭据环境变量；尚未发送请求')
    payload = dict(model=model, stream=False, max_completion_tokens=8192,
        messages=[dict(role='system', content=(
            f'Rewrite each of the {count} supplied public product descriptions as ONE natural future shopping query. '
            f'Return exactly one query for each input, in the same order: exactly {count} objects total. '
            'Use only supported product attributes. Do not invent brands, facts, or purchase history.'
            f'{extra_guidance} '
            f'Return ONLY a JSON array of exactly {count} objects with keys input_row_id and text. '
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
        raise ValueError(f'模型改写正文必须为 {count} 条结构化记录，不能包含思考或解释') from None
    _require(isinstance(result, list) and len(result) == count,
             f'模型改写数量不等于 {count}，实际：' + (str(len(result)) if isinstance(result, list) else '非列表'))
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
    if count != 30:
        _require({row['input_row_id'] for row in rows} == rewrite_ids, '测试改写必须逐条对应不同的留出描述')
    public_identity = dict(provider=provider, model=actual_model, requested_model=model,
        **temperature_identity,
        endpoint=endpoint, response_id=response_id, prompt_version=prompt_version,
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
        test_fixture=False, allow_declared_shortfall=False):
    """Copy finite public snapshots/assets into a new immutable bundle.

    Inputs are JSONL paths. source_metadata maps the four canonical source kinds
    to their metadata dicts; all data origin statements must be explicit imports.
    Candidate image paths are relative to the candidate-catalog input file.
    """
    _require(type(test_fixture) is bool, '训练包夹具开关必须为布尔值')
    _require(type(allow_declared_shortfall) is bool, '声明欠额开关必须为布尔值')
    root = validate_public_bundle_root(output_dir, require_production=not test_fixture)
    _require(not root.exists() or not any(root.iterdir()), '训练包目录非空，禁止覆盖冻结数据')
    _require(isinstance(source_metadata, dict) and set(source_metadata) == {
        'category_description', 'model_paraphrase', 'heldout_search_query', 'candidate_catalog'},
        '必须为四种来源提供完整来源元数据')
    reject_private_fields(source_metadata)
    paths = {'category_description': Path(descriptions), 'model_paraphrase': Path(paraphrases),
             'heldout_search_query': Path(heldout_queries), 'candidate_catalog': Path(candidates)}
    loaded = {kind: list(read_source_rows(path).values()) for kind, path in paths.items()}
    declared_shortfall = {}
    for kind, count in QUERY_COUNTS.items():
        if kind == 'heldout_search_query' and allow_declared_shortfall and len(loaded[kind]) < count:
            held_meta = source_metadata.get('heldout_search_query', {})
            _require(held_meta.get('complete') is False
                     and type(held_meta.get('actual_count')) is int
                     and held_meta['actual_count'] == len(loaded[kind]) >= 1,
                     '允许部分预留搜索查询时须在来源元数据中如实声明 complete=false 且 actual_count 与真实条数一致')
            declared_shortfall[kind] = dict(requested=count, actual=len(loaded[kind]), complete=False)
            loaded[kind] = list(loaded[kind])
        else:
            _require(len(loaded[kind]) >= count, '训练来源配额不足：' + kind)
            loaded[kind] = loaded[kind][:count]
    _require(bool(loaded['candidate_catalog']), '公开候选目录为空')
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
    if declared_shortfall:
        manifest['declared_query_shortfall'] = declared_shortfall
    manifest_path = root / 'training_bundle.json'
    save_json(manifest_path, manifest)
    bundle = load_training_bundle(manifest_path, source=source, target=target,
        public_semantic=public_semantic, require_production=not test_fixture)
    return bundle


RULE_PARAPHRASE_PROMPT_VERSION = 'rule_based_length_aligned_eval_v4_1'

_RULE_GROUP_NOUNS = {
    '01_camera': ('security camera', True),
    '02_chair': ('barber chair', True),
    '03_cookies': ('cookie dough', False),
    '04_curtain': ('curtain panels', False),
    '05_hair_extension': ('hair extensions', False),
    '06_jeans': ('denim jeans', False),
    '07_shampoo': ('shampoo', False),
    '08_speaker': ('wireless speaker', True),
    '09_toothbrush': ('toothbrush', True),
}

_RULE_GROUP_MODIFIERS = {
    '01_camera': ('outdoor', 'indoor', 'wireless', 'wifi', 'wi-fi', 'hd', '1080p', '2mp', '3mp', '4mp', '2k', 'night-vision', 'night vision', 'dome', 'ptz', 'pan-tilt', 'waterproof', 'bullet'),
    '02_chair': ('hydraulic', 'reclining', 'swivel', 'heavy-duty', 'heavy duty', 'adjustable', 'vintage', 'classic', 'salon', 'styling'),
    '03_cookies': ('ready-to-bake', 'edible', 'chocolate chip', 'oatmeal', 'sugar', 'butterscotch', 'refrigerated', 'gourmet', 'homemade'),
    '04_curtain': ('blackout', 'thermal', 'insulated', 'sheer', 'grommet', 'rod pocket', 'printed', 'noise-reducing', 'room-darkening'),
    '05_hair_extension': ('human', 'clip-in', 'tape-in', 'silky straight', 'curly', 'wavy', 'water wave', 'braiding', 'remy', 'seamless'),
    '06_jeans': ('high-waisted', 'high-waist', 'mid-rise', 'low-waist', 'straight-leg', 'wide-leg', 'skinny', 'flare', 'bell-bottom', 'boyfriend', 'stretch', 'ripped', 'distressed', 'baggy'),
    '07_shampoo': ('sulfate-free', 'moisturizing', 'volumizing', 'clarifying', 'anti-dandruff', 'color-care', 'gentle', 'daily', 'repairing', 'curly'),
    '08_speaker': ('bluetooth', 'portable', 'waterproof', 'rechargeable', 'mini', 'outdoor', 'compact', 'pocket-sized', 'high-fidelity'),
    '09_toothbrush': ('electric', 'sonic', 'manual', 'u-shaped', 'soft-bristle', 'soft', 'silicone', 'bamboo', 'waterproof', 'kids', 'toddler', 'rechargeable'),
}

_RULE_MUTUALLY_EXCLUSIVE_MODS = (
    {'electric', 'manual'},
    {'indoor', 'outdoor'},
    {'high-waisted', 'high-waist', 'mid-rise', 'low-waist'},
    {'straight-leg', 'wide-leg', 'skinny', 'flare', 'bell-bottom', 'boyfriend'},
)

_RULE_TRAILING_DANGLING = {
    'and', 'or', 'of', 'with', 'for', 'to', 'in', 'on', 'the', 'a', 'an', 'so',
    'that', 'which', 'by', 'from', 'as', 'at', 'into', 'including', 'featuring', 'letting', 'you',
    'is', 'are', 'was', 'were', 'be', 'has', 'have', 'had', 'when',
}

_RULE_PRESERVE_UPPER = {
    'HD', 'FHD', 'UHD', '1080P', '720P', '2K', '4K', '2MP', '3MP', '4MP', '5MP', '8MP',
    'IP65', 'IP66', 'IP67', 'LED', 'IR', 'UV', 'PU', 'PVC', 'PP', 'TPR', 'ABS', 'USB',
    'WIFI', 'WI-FI', 'AHD', 'CVI', 'TVI', 'CVBS', 'AC', 'DC', 'LED', 'MP3', 'QI', 'APP',
    'PTZ', 'NVR', 'DVR', '360°', 'U-SHAPED', '3D', 'PH', 'BPA',
}

_RULE_FLUFF_RE = re.compile(
    r'\b(?:'
    r'leaving nothing to the imagination|welcome to\b|buy with confidence|100% satisfaction|'
    r'satisfaction guarantee|perfect gift|best gift|great gift|makes a great|ideal gift|'
    r'if you have any questions|feel free to contact|after.?sales|customer service|'
    r'business address|for shipping|valid phone|please be sure|shipped via|before delivery|kindly note|'
    r'exquisite design|durable structure|lazy sofa|related style|'
    r'made of high quality materials?|high quality material|brand new and high quality|'
    r'too good to be true|wear-with-everything|you can use for longer|order now|add to cart|'
    r'definitely be your best partner|excellent value|just start using'
    r')\b',
    re.I,
)

_RULE_OPENERS_SINGULAR = (
    'Looking for a {noun}',
    'I need a {noun}',
    'Find a {noun}',
    'I want a {noun}',
    'I am looking for a {noun}',
    'Searching for a {noun}',
)

_RULE_OPENERS_PLURAL = (
    'Looking for {noun}',
    'I need {noun}',
    'Find {noun}',
    'I want {noun}',
    'I am looking for {noun}',
    'Searching for {noun}',
)


def _rule_clean_text(text):
    cleaned = re.sub(r'【[^】]{1,80}】|\[[^\]]{1,80}\]\s*:?', ' ', str(text))
    cleaned = re.sub(r'â˜…|â€“|â€”|â€™|â€œ|â€[?\u009d]?', ' ', cleaned)
    cleaned = re.sub(r'[➤❁★☆✿❀❤❥✤✔✓•▪●◆◇▶►]+', ' ', cleaned)
    cleaned = re.sub(
        r'(?:^|\n)\s*(?:features?|specifications?|versatile usages?|perfect details?|'
        r'description|about nursing|occasion|product features?|package included|warm tips?|note)\s*[:\-–—]?\s*',
        ' ',
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(r'(?:^|\n)\s*[A-Za-z][A-Za-z0-9 &\-\/]{2,36}:\s*', ' ', cleaned)
    cleaned = re.sub(r'^\s*[:\-–—]+\s*', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _rule_lowercase_clause(clause):
    words = clause.split()
    if not words:
        return ''
    out = []
    for idx, tok in enumerate(words):
        bare = tok.strip('.,;:!?()"\'')
        if bare.upper() in _RULE_PRESERVE_UPPER or (len(bare) >= 2 and bare.isupper() and any(ch.isalpha() for ch in bare)):
            out.append(tok)
        elif idx == 0:
            if bare.casefold() in ('bluetooth', 'wifi', 'wi-fi', 'iphone', 'android', 'alexa', 'google'):
                out.append(tok)
            else:
                out.append(tok[0].lower() + tok[1:] if len(tok) > 1 else tok.lower())
        else:
            out.append(tok)
    return ' '.join(out)


def _rule_normalize_clause(raw_clause):
    c = raw_clause.strip(' \t\r\n,.;:-–—()*')
    if c.count('(') != c.count(')'):
        c = c.replace('(', '').replace(')', '')
    c = re.sub(r'^[\*\d\.\s]+(?=[A-Za-z])', '', c)
    c = re.sub(r'^(?:and|or|also|plus|with|featuring|including|that|which|it|this|these|our|the|can)\s+', '', c, flags=re.I)
    c = re.sub(r'\b(?:our|this)\s+(?:new\s+)?(?:product|item|unit|camera|chair|curtains?|shampoo|speaker|toothbrush|jeans)\b', 'it', c, flags=re.I)
    c = re.sub(r'\byou\s+can\s+(?:easily\s+)?', 'letting you ', c, flags=re.I)
    c = re.sub(r'\blets?\s+you\s+', 'allowing you to ', c, flags=re.I)
    c = re.sub(r'\bprovides?\s+(?:you\s+with\s+)?', '', c, flags=re.I)
    c = re.sub(r'^(?:is|are|can be|was|were)\s+', '', c, flags=re.I)
    c = re.sub(r'^(?:it\s+(?:is|has|features?|includes?|offers?|can)\s+)', '', c, flags=re.I)
    c = re.sub(r'\s+', ' ', c).strip(' \t\r\n,.;:-–—*')
    return _rule_lowercase_clause(c)


def _rule_extract_clauses(text):
    cleaned = _rule_clean_text(text)
    raw_parts = re.split(r'(?<=[.!?])(?:\s+|(?=[A-Z]))|\s*[;\n]+\s*|,\s+(?=(?:and |with |for |featuring |including |designed |made |suitable ))', cleaned)
    clauses, seen = [], set()
    for part in raw_parts:
        subparts = re.split(r'\s+---\s+|\s+--\s+|\s+\|\s+', part)
        for sub in subparts:
            norm = _rule_normalize_clause(sub)
            words = re.findall(r"[^\W_]+", norm)
            if len(words) < 3 or len(norm) < 14 or _RULE_FLUFF_RE.search(norm):
                continue
            if norm.casefold() in seen:
                continue
            seen.add(norm.casefold())
            clauses.append(norm)
    return clauses


def _rule_noun_phrase(group, source_text):
    base_noun, is_singular = _RULE_GROUP_NOUNS.get(group, ('product', True))
    if group == '08_speaker' and re.search(r'\bbluetooth\b', source_text, re.I):
        base_noun = 'Bluetooth speaker'
    lower_src = source_text.casefold()
    chosen = []
    for mod in _RULE_GROUP_MODIFIERS.get(group, ()):
        if len(chosen) >= 2:
            break
        if re.search(r'\b' + re.escape(mod) + r'\b', lower_src):
            if mod in base_noun.casefold():
                continue
            if any(mod in prev.casefold() or prev.casefold() in mod for prev in chosen):
                continue
            if any(mod in grp and any(p.casefold() in grp for p in chosen) for grp in _RULE_MUTUALLY_EXCLUSIVE_MODS):
                continue
            chosen.append('Wi-Fi' if mod in ('wifi', 'wi-fi') else ('Bluetooth' if mod == 'bluetooth' else mod))
    phrase = (' '.join(chosen) + ' ' + base_noun).strip()
    return phrase, is_singular


def _rule_trim_dangling(clause):
    words = clause.split()
    while words and words[-1].strip('.,;:!?').casefold() in _RULE_TRAILING_DANGLING:
        words.pop()
    return ' '.join(words).rstrip(',;:-–—')


def _rule_assemble_candidate(opener_tpl, noun_phrase, is_singular, clauses, target_len, min_len, max_len, *, order_offset=0):
    if is_singular and '{noun}' in opener_tpl:
        lower_np = noun_phrase.casefold()
        use_an = lower_np[0] in 'aeio' or (lower_np[0] == 'u' and not lower_np.startswith(('u-', 'uv')))
        prefix = opener_tpl.replace('a {noun}', ('an ' if use_an else 'a ') + '{noun}').format(noun=noun_phrase)
    else:
        prefix = opener_tpl.format(noun=noun_phrase)

    ordered = clauses[order_offset:] + clauses[:order_offset] if clauses else []
    best = None
    for idx in range(len(ordered)):
        picked = []
        for cl in ordered[idx:] + ordered[:idx]:
            clause_cap = min(105, max(45, max_len - len(prefix) - 13)) if not picked else 105
            if len(cl) > clause_cap:
                words = cl.split()
                trimmed = []
                limit = min(clause_cap, 88) if clause_cap >= 88 else clause_cap
                for w in words:
                    if sum(len(x) + 1 for x in trimmed) + len(w) > limit:
                        break
                    trimmed.append(w)
                cl = _rule_trim_dangling(' '.join(trimmed))
                if len(cl.split()) < 3:
                    continue
            else:
                cl = _rule_trim_dangling(cl)
                if len(cl.split()) < 3:
                    continue
            extra_join = 11 if not picked else 6
            projected = len(prefix) + extra_join + len(', '.join(picked + [cl])) + 1
            if projected > max_len:
                if not picked:
                    continue
                break
            picked.append(cl)
            cur_len = len(prefix) + len(' featuring ') + len(', '.join(picked)) + 1
            if cur_len >= max(min_len, int(round(target_len * 0.90))):
                break
        if not picked:
            continue
        if len(picked) >= 2:
            body = ', '.join(picked[:-1]) + ', and ' + picked[-1]
        else:
            body = picked[0]
        first_cl = picked[0].casefold()
        if first_cl.startswith(('suitable for ', 'ideal for ', 'great for ', 'perfect for ', 'designed for ', 'made of ', 'made from ', 'crafted from ', 'built with ')):
            joiner = ' '
        elif first_cl.startswith(('letting you ', 'allowing you to ')):
            joiner = ', '
        else:
            joiner = ' featuring '
        end_punct = '?' if prefix.startswith('Where can I buy') else '.'
        cand = f'{prefix}{joiner}{body}{end_punct}'
        cand = re.sub(r'\s+', ' ', cand).strip()
        if min_len <= len(cand) <= max_len:
            if best is None or abs(len(cand) - target_len) < abs(len(best) - target_len):
                best = cand
            if abs(len(cand) - target_len) <= 18:
                return cand
    return best


def generate_rule_evaluation_paraphrases(
    descriptions,
    *,
    group,
    category,
    target_lengths=None,
    forbidden_texts=(),
    seed=20261120,
    acceptable_fn=None,
):
    """Deterministically generate 6 grounded, length-aligned shopping queries without calling an LLM."""
    _require(isinstance(descriptions, list) and len(descriptions) >= 6, f'{group} 至少需要 6 条改写候选描述')
    min_len, max_len = (55, 125) if group == '03_cookies' else (58, 225)
    if target_lengths:
        clipped = sorted(max(min_len, min(max_len, int(x))) for x in target_lengths[:6])
        while len(clipped) < 6:
            clipped.append(clipped[-1])
    else:
        default_len = 85 if group == '03_cookies' else 155
        clipped = [default_len] * 6

    dupes = DuplicateIndex(list(forbidden_texts) + [row['text'] for row in descriptions])
    candidates_sorted = sorted(descriptions, key=lambda r: (len(r['text']), str(r['id'])))
    if len(candidates_sorted) > 6:
        primary = candidates_sorted[:6]
        extras = candidates_sorted[6:]
    else:
        primary = candidates_sorted
        extras = []

    ordered_inputs = []
    for row in primary:
        ordered_inputs.append(row)

    def _opener_pool(is_singular, grp):
        pool = list(_RULE_OPENERS_SINGULAR if is_singular else _RULE_OPENERS_PLURAL)
        if grp == '03_cookies':
            pool.append('Where can I buy {noun}')
        return pool

    selected_rows = []
    seen_norms = {overlap_normalize(t) for t in forbidden_texts} | {overlap_normalize(r['text']) for r in descriptions}
    pool_queue = list(ordered_inputs) + list(extras)
    used_input_ids = set()

    for slot_idx, target_len in enumerate(clipped):
        placed = False
        for row in pool_queue:
            rid = str(row['id'])
            if rid in used_input_ids:
                continue
            src_text = row['text']
            clauses = _rule_extract_clauses(src_text)
            noun_phrase, is_singular = _rule_noun_phrase(group, src_text)
            openers = _opener_pool(is_singular, group)
            start_op = int(hashlib.sha256(f'{seed}|{group}|{rid}'.encode('utf-8')).hexdigest(), 16) % len(openers)
            ordered_openers = openers[start_op:] + openers[:start_op]

            chosen_text = None
            for op in ordered_openers:
                for off in range(max(1, min(len(clauses), 4))):
                    cand = _rule_assemble_candidate(
                        op, noun_phrase, is_singular, clauses, target_len, min_len, max_len, order_offset=off
                    )
                    if not cand:
                        continue
                    norm_cand = overlap_normalize(cand)
                    if norm_cand in seen_norms or cand in dupes:
                        continue
                    if acceptable_fn is not None and not acceptable_fn(group, cand):
                        continue
                    chosen_text = cand
                    break
                if chosen_text is not None:
                    break

            if chosen_text is not None:
                seen_norms.add(overlap_normalize(chosen_text))
                dupes.add(chosen_text)
                used_input_ids.add(rid)
                selected_rows.append(
                    dict(
                        id='paraphrase_' + str(slot_idx + 1).zfill(3),
                        text=chosen_text,
                        category=category,
                        input_row_id=rid,
                        provider='local_rule',
                        model='rule_paraphrase_v1',
                        requested_model='rule_paraphrase_v1',
                    )
                )
                placed = True
                break
        _require(placed, f'{group} 规则改写未能生成第 {slot_idx + 1} 条合规且去重的查询')

    generation_meta = dict(
        mode='rule_based_length_aligned',
        provider='local_rule',
        model='rule_paraphrase_v1',
        requested_model='rule_paraphrase_v1',
        prompt_version=RULE_PARAPHRASE_PROMPT_VERSION,
        input_descriptions_sha256=content_sha256(descriptions),
        paraphrase_input_row_ids=[r['input_row_id'] for r in selected_rows],
        target_lengths=clipped,
        generated_lengths=[len(r['text']) for r in selected_rows],
        generated_at=datetime.now(timezone.utc).isoformat(),
        paper_query_generator_reproduced=False,
    )
    return dict(
        status='generated',
        network_request_performed=False,
        rows=selected_rows,
        generation=generation_meta,
    )
