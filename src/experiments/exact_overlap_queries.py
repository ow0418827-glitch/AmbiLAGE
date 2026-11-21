"""Frozen, source-audited 100/30/0 text-overlap evaluation sets."""
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from src.paths import DATA, EVALUATION_BASE, RESOURCES, resolve_evaluation_base
from src.visual.audit import freeze_json, read_json

ROOT = RESOURCES / 'visual_inception_evaluation/exact_overlap_v4'
TRAINING_ROOT = DATA / 'public/visual_inception'
FAMILIES = {'category_description': 10, 'model_paraphrase': 6, 'heldout_search_query': 4}
REPEATED = {'category_description': 3, 'model_paraphrase': 2, 'heldout_search_query': 1}
SUITES = {'training_oracle_v4': 20, 'overlap_30_v4': 6, 'zero_overlap_v4': 0}
PROTOCOLS = {f'visual_inception_{suite}_exact': suite for suite in SUITES}
LABELS = {'category_description': '同类商品原文', 'model_paraphrase': '改写查询',
          'heldout_search_query': '短搜索查询'}
SYNTHETIC_SEARCH = 'synthetic_search_fallback'
SHA256 = re.compile(r'[0-9a-f]{64}')
WORD = re.compile(r"[^\W_]+", re.UNICODE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize(text):
    require(isinstance(text, str) and text.strip(), '查询正文必须为非空字符串')
    return ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()


def tokens(text):
    return WORD.findall(normalize(text))


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def instructions_sha(texts):
    return text_sha(json.dumps(texts, ensure_ascii=False, separators=(',', ':')))


def training_index(base, training_root):
    """Only read the public frozen query bundles and the base's manifest."""
    groups = read_json(Path(base) / 'manifest.json')['groups']
    bundles = {g: read_json(Path(training_root) / g / 'training_bundle.json') for g in groups}
    rows = {g: {row['id']: row for row in bundle['queries']} for g, bundle in bundles.items()}
    for g, bundle in bundles.items():
        require(len(rows[g]) == len(bundle['queries']) and not bundle.get('test_fixture')
                and bundle.get('formal_queries_used') is False and bundle.get('victim_memory_used') is False,
                f'{g} 训练查询身份或隔离状态无效')
    texts = {normalize(row['text']) for group in rows.values() for row in group.values()}
    concepts = {normalize(bundle['target_concept_text']) for bundle in bundles.values()}
    return groups, bundles, rows, texts, concepts


class DuplicateIndex:
    """Check exact text, punctuation/reordering variants, and long near-copies."""

    def __init__(self, texts=()):
        self.by_words = {}
        self.texts = set()
        for text in texts:
            self.add(text)

    def add(self, text):
        self.texts.add(normalize(text))
        words = tokens(text)
        self.by_words.setdefault(tuple(sorted(words)), words)

    def contains_fragment(self, text):
        normalized = normalize(text)
        return any(normalized in old for old in self.texts)

    def __contains__(self, text):
        words = tokens(text)
        if tuple(sorted(words)) in self.by_words:
            return True
        if len(words) < 6:
            return False
        word_set = set(words)
        for old in self.by_words.values():
            if min(len(words), len(old)) < 0.9 * max(len(words), len(old)):
                continue
            if len(word_set.intersection(old)) < 0.75 * len(word_set):
                continue
            if SequenceMatcher(None, old, words, autojunk=False).ratio() >= 0.9:
                return True
        return False


def _group_document(group, spec, pairs, suite):
    protocol = next(key for key, value in PROTOCOLS.items() if value == suite)
    records = []
    for pair in pairs:
        use_training = suite == 'training_oracle_v4' or (
            suite == 'overlap_30_v4' and pair['reuse_at_30_percent'])
        text = pair['training']['text'] if use_training else pair['heldout']['text']
        record = dict(id=pair['id'], query_family=pair['family'], instruction=text,
                      overlap_source='training' if use_training else 'heldout')
        if not use_training and pair['heldout']['source'].get('origin') == SYNTHETIC_SEARCH:
            record['search_provenance'] = SYNTHETIC_SEARCH
        records.append(record)
    instructions = [row['instruction'] for row in records]
    return dict(schema_version=2, status='ready', method='visual_inception',
                evaluation_protocol=protocol, group=group, query_mode='text',
                target_public_id=spec['target_id'], category=spec['category'],
                language='en', training_use_allowed=False, is_real_search_log=False,
                query_image_required=False, query_images_generated=False,
                instructions=instructions, instructions_sha256=instructions_sha(instructions),
                query_records=records, query_family_counts=FAMILIES)


def _check_pairs(group, pairs, own, all_texts, dupes, search_index, product_ids):
    require(len(pairs) == 20 and Counter(p['family'] for p in pairs) == FAMILIES
            and len({p['id'] for p in pairs}) == 20
            and all(type(p['reuse_at_30_percent']) is bool for p in pairs)
            and Counter(p['family'] for p in pairs if p['reuse_at_30_percent']) == REPEATED,
            f'{group} 配对位置或三成配额错误')
    used_old, used_new = set(), DuplicateIndex()
    ids = set()
    for pair in pairs:
        family, old, new = pair['family'], pair['training'], pair['heldout']
        bound = own.get(old['id'], {})
        require(bound.get('text') == old['text'] and bound.get('source_type') == family
                and old['id'] not in used_old, f'{group} 训练原题不匹配')
        used_old.add(old['id'])
        text, source = new['text'], new['source']
        require(normalize(text) not in all_texts and text not in dupes and text not in used_new,
                f'{group} 留出查询命中训练或测试去重规则')
        used_new.add(text)
        require(source['id'] not in ids and text_sha(text) == new['text_sha256'],
                f'{group} 留出来源编号重复或正文摘要不符')
        ids.add(source['id'])
        if family == 'heldout_search_query':
            if source.get('origin') == SYNTHETIC_SEARCH:
                require(group == '02_chair' and source['id'].startswith(f'{group}:synthetic:')
                        and set(source) == {'id', 'origin'} and 2 <= len(tokens(text)) <= 8,
                        f'{group} 临时搜索补位格式或组别无效')
                continue
            dataset, query_id = source['dataset'], source['query_id']
            require(dataset in ('esci', 'orcas', 'msmarco_websearch', 'wands')
                    and (dataset, query_id) not in search_index
                    and SHA256.fullmatch(source['snapshot_sha256']) and source['url'].startswith('https://')
                    and source['id'] == f'{dataset}:{query_id}'
                    and str(source['evidence']['query_id']) == query_id
                    and source['original_text'] == text,
                    f'{group} 搜索词缺少独立的公开原文证据')
            continue
        product = source if family == 'category_description' else source['input']
        require(product['origin'] in ('unseen_product', 'unused_field')
                and ((product['product_id'] in product_ids) == (product['origin'] == 'unused_field'))
                and (product['origin'] != 'unused_field' or group == '03_cookies')
                and SHA256.fullmatch(product['snapshot_sha256']) and product['field']
                and product['original_item'].strip() == product['original_text']
                and not (family == 'category_description' and product['original_text'] != text),
                f'{group} 商品文字缺少原文或独立来源证据')
        require(product['original_text'] not in dupes and not (
                    product['origin'] == 'unused_field' and dupes.contains_fragment(product['original_text'])),
                f'{group} 留出原文与训练文本近重复或已作为训练字段片段使用')


def source_exclusions(bundles, training_root, groups):
    used_search, products = set(), set()
    for group, bundle in bundles.items():
        products.update(groups[group][key] for key in ('source_id', 'target_id'))
        products.update(row['id'] for row in bundle['candidates'])
        group_root = Path(training_root) / group / 'sources'
        descriptions = group_root / 'category_description.jsonl'
        searches = group_root / 'heldout_search_query.jsonl'
        for path in (descriptions, searches):
            require(file_sha(path) == bundle['sources'][path.stem]['sha256'],
                    f'{group} 公开训练来源文件摘要不符：{path.name}')
        for line in descriptions.read_text(encoding='utf-8').splitlines():
            if line.strip():
                products.add(json.loads(line)['product_id'])
        for line in searches.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                used_search.add((row['source_dataset'], str(row['query_id'])))
                used_search.update((item.get('source_dataset', row['source_dataset']), str(item['query_id']))
                                   for item in row.get('source_evidence', []))
    return used_search, products


def _checked_index(base, training_root):
    specs, bundles, old, trained_texts, concepts = training_index(base, training_root)
    search_index, product_ids = source_exclusions(bundles, training_root, specs)
    return specs, old, trained_texts, DuplicateIndex(trained_texts | concepts), search_index, product_ids


def publish(root, base, training_root, pairs_by_group, excluded, seed):
    specs, old, training, dupes, searches, products = _checked_index(base, training_root)
    require(bool(pairs_by_group) and set(pairs_by_group).isdisjoint(excluded)
            and set(pairs_by_group) | set(excluded) == set(specs), '配对集组别覆盖错误')
    for group, pairs in pairs_by_group.items():
        _check_pairs(group, pairs, old[group], training, dupes, searches, products)

    frozen = dict(schema_version=1, seed=seed, missing_groups=excluded, groups=pairs_by_group,
                  training_sha256={g: file_sha(Path(training_root) / g / 'training_bundle.json') for g in specs})
    if excluded:
        freeze_json(Path(root) / 'prepared_pairs.json', dict(frozen, status='pending'), newline='\n')
        return
    pair_path = Path(root) / 'pairs.json'
    freeze_json(pair_path, dict(frozen, status='ready'), newline='\n')
    for suite in SUITES:
        output = Path(root) / suite
        group_files = {}
        for group, pairs in pairs_by_group.items():
            path = output / group / 'queries.json'
            freeze_json(path, _group_document(group, specs[group], pairs, suite), newline='\n')
            group_files[group] = file_sha(path)
        freeze_json(output / 'manifest.json', dict(schema_version=1, suite=suite,
            pairs_sha256=file_sha(pair_path), base_sha256=file_sha(Path(base) / 'manifest.json'),
            groups=group_files), newline='\n')
        validate(output, base=base, training_root=training_root)


def _validate_population(frozen, base, training_root, *, status):
    specs, old, training, dupes, searches, products = _checked_index(base, training_root)
    require(frozen['training_sha256'] ==
            {g: file_sha(Path(training_root) / g / 'training_bundle.json') for g in specs},
            '冻结训练包摘要不符')
    groups, missing = frozen['groups'], frozen['missing_groups']
    require(frozen['status'] == status and set(groups).isdisjoint(missing)
            and set(groups) | set(missing) == set(specs)
            and bool(missing) == (status == 'pending'),
            '评测集组别覆盖错误')
    for group, pairs in groups.items():
        _check_pairs(group, pairs, old[group], training, dupes, searches, products)
    return specs, old, training


def validate_preparation(root=ROOT, *, base=None, training_root=TRAINING_ROOT):
    base = resolve_evaluation_base(base)
    prepared = read_json(Path(root) / 'prepared_pairs.json')
    _validate_population(prepared, base, training_root, status='pending')
    for group, shortage in prepared['missing_groups'].items():
        counts = shortage['counts']
        require(set(counts) == set(FAMILIES) and all(type(counts[f]) is int and 0 <= counts[f] <= n
                for f, n in FAMILIES.items()), f'{group} 准备数量无效')
        missing = {f: n - counts[f] for f, n in FAMILIES.items() if counts[f] < n}
        require(missing and shortage['missing'] == missing, f'{group} 缺额声明有误')
    return prepared


def validate(query_root, *, base=None, training_root=TRAINING_ROOT):
    root = Path(query_root).resolve()
    base = resolve_evaluation_base(base)
    manifest_path = root / 'manifest.json'
    if not manifest_path.is_file() and (root.parent / 'prepared_pairs.json').is_file():
        prepared = validate_preparation(root.parent, base=base, training_root=training_root)
        raise ValueError('第四版三档待补齐九组：' + ', '.join(prepared['missing_groups']))
    manifest = read_json(manifest_path)
    suite = manifest['suite']
    require(suite in SUITES and root.name == suite and manifest['schema_version'] == 1,
            '评测集目录、名称或版本不符')
    pair_path = root.parent / 'pairs.json'
    base_manifest_path = Path(base) / 'manifest.json'
    base_manifest = read_json(base_manifest_path)
    base_matches = (
        file_sha(base_manifest_path) == manifest['base_sha256']
        or base_manifest.get('derived_from_base_manifest_sha256') == manifest['base_sha256']
    )
    require(file_sha(pair_path) == manifest['pairs_sha256'] and base_matches, '配对或基础清单摘要不符')
    frozen = read_json(pair_path)
    specs, old, training = _validate_population(frozen, base, training_root, status='ready')
    require(list(manifest['groups']) == list(frozen['groups']) and manifest['groups'],
            '评测集组别覆盖错误')
    audits = {}
    for group, pairs in frozen['groups'].items():
        path = root / group / 'queries.json'
        digest = file_sha(path)
        doc = read_json(path)
        require(digest == manifest['groups'][group] and doc == _group_document(group, specs[group], pairs, suite),
                f'{group} 查询与配对位置不一致')
        norms = {normalize(text) for text in doc['instructions']}
        exact = len(norms & {normalize(row['text']) for row in old[group].values()})
        require(len(norms) == 20 and exact == SUITES[suite], f'{group} 实际重复数量不符')
        audits[group] = dict(target_public_id=doc['target_public_id'], category=doc['category'],
            query_manifest=str(path), query_manifest_sha256=digest,
            instructions_sha256=doc['instructions_sha256'], query_count=20,
            query_family_counts=FAMILIES, training_normalized_exact_overlap_count=exact,
            synthetic_search_count=sum(r.get('search_provenance') == SYNTHETIC_SEARCH
                                       for r in doc['query_records']))
    return dict(schema_version=1, status='passed', query_suite=suite,
        protocol=next(key for key, value in PROTOCOLS.items() if value == suite),
        training_overlap_mode=suite, query_root=str(root), manifest_sha256=file_sha(manifest_path),
        base_path=str(Path(base).resolve()), base_manifest_sha256=manifest['base_sha256'],
        group_count=len(audits), queries_per_group=20, total_queries=20 * len(audits),
        query_families=list(FAMILIES), query_family_labels_zh=LABELS, queries_per_family=FAMILIES,
        training_overlap_checked=True, training_snapshot_count=len(specs),
        unique_training_text_count=len(training),
        training_normalized_exact_overlap_count=SUITES[suite] * len(audits),
        synthetic_search_count=sum(group['synthetic_search_count'] for group in audits.values()),
        groups=audits)
