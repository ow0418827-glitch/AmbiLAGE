"""Freeze genuine public search queries; never generate or read formal queries.

The Amazon Science ESCI examples snapshot is pinned by commit and Git-LFS hash.
Only English public train rows are eligible. Category membership is a transparent
lexical selection rule on the original customer query, not a model/rank score.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.visual.io import save_json, sha256_file

REVISION = '7916cdf6ab75a462e77f20ab40428a10923998d5'
REPOSITORY = 'https://github.com/amazon-science/esci-data'
SNAPSHOT_NAME = 'shopping_queries_dataset_examples.parquet'
SNAPSHOT_SHA256 = '4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a'
SNAPSHOT_BYTES = 51286808
SNAPSHOT_URL = f'https://media.githubusercontent.com/media/amazon-science/esci-data/{REVISION}/shopping_queries_dataset/{SNAPSHOT_NAME}'
SELECTION_POLICY = 'independent_public_esci_orcas_product_intent_hash_reserve_v5'
ORCAS_REVISION = '21b4ece9f4fc10fa4cec2028bb74287629945df6'
ORCAS_URL = 'https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas-doctrain-queries.tsv.gz'
ORCAS_NAME = 'orcas-doctrain-queries.tsv.gz'
ORCAS_BYTES = 104209356
ORCAS_MD5 = '519c5f522294406e3b0574d7d53cf233'
ORCAS_SHA256 = '8e69679a83f2d9767916e377f142e7c3fe110fa1cca9f52dc644c705ad975692'
WEBSEARCH_REVISION = '7b5a408c5b8b3edd30f21acd7e126996759994fa'
WEBSEARCH_URL = 'https://msmarco.z22.web.core.windows.net/msmarcowebsearch/100M_queries/queries_train.tsv'
WEBSEARCH_SHA256 = 'abeab999bb73dec9445706f240e37d8b3c6dc39bcfafd6b3409622e48520b024'
WEBSEARCH_SUBSET_SHA256 = '885b66208bc0fad8b0ea422f7904b0402b5d4cdd8e1553adec03ea262e1599a6'
PRODUCT_INTENT_EXCLUDE = r'\b(?:recipe|recipes|how|why|can|should|does|do|is|are|was|were|what|when|where|ingredients|salmonella|eating|making|make|homemade|business|start|freeze|freezing|baking|diy|methods?|restoration|rentals?|rent|tax|agreement|near me)\b'

# All selected text remains exactly as written in the public query snapshot.
# These rules are fixed before model use; exclusions remove accessories/different
# object types rather than filtering on downstream retrieval/attack outcomes.
RULES = {
    '01_camera': (r'\b(?:(?:security|surveillance|ip|wifi|wi-fi|wireless|outdoor|indoor|home)\s+(?:\w+\s+){0,2}cameras?|cameras?\s+(?:\w+\s+){0,2}(?:security|surveillance))\b', r'\b(?:mount|mounts|bracket|brackets|case|cases|bag|bags|charger|chargers|adapter|adapters|cable|cables|lens|lenses|skin|skins|cover|covers|housing|power supply|battery|batteries|game|trail|hunting|back ?up|underwater|endoscope|action|gopro|dash|drone|webcam|dslr|rear ?view|reversing|vehicles?|cars?|trucks?)\b'),
    '02_chair': (r'\b(?:(?:barber|salon|styling|hairdresser|hairdressing|hair\s+dressing)\s+(?:\w+\s+){0,2}chairs?)\b', r'\b(?:cover|covers|mat|mats|pump|pumps|base|bases|parts|replacement|cushion|cushions|booster|boosters)\b'),
    '03_cookies': (r'\b(?:cookie\s+dough|edible\s+(?:cookie\s+)?dough)\b', r'\b(?:scoop|scoops|cutter|cutters|roller|rollers|perfume|candle|candles|soap|protein|bars?|halo top|ice cream|icecream|dispenser|mixer|mixers)\b'),
    '04_curtain': (r'\b(?:curtains?|drapes?|draperies)\b', r'\b(?:rod|rods|hook|hooks|ring|rings|clip|clips|tieback|tiebacks|holdback|holdbacks|bracket|brackets|track|tracks|shower|bathroom|bead|beaded|led|light|lights)\b'),
    '05_hair_extension': (r'\b(?:hair\s+extensions?|(?:human\s+)?hair\s+weaves?|clip[- ]?in\s+(?:human\s+)?hair)\b', r'\b(?:remover|pliers|tool|tools|glue|adhesive|tape roll|beads|thread|needle|needles|stand|stands|storage|shampoo|tinsel)\b'),
    '06_jeans': (r'\b(?:jeans|denim\s+(?:pants|trousers))\b', r'\b(?:patch|patches|button|buttons|jacket|jackets|repair|spray|fabric|thread|sewing|skirt|skirts|purse|purses|shorts)\b'),
    '07_shampoo': (r'\bshampoos?\b', r'\b(?:carpet|carpets|dog|dogs|cat|cats|pet|pets|car|cars|vacuum|machine|machines|upholstery|dispenser|dispensers|bottle pump|rug|rugs|horse|horses|cape|capes|basin|basins|bowl|bowls)\b'),
    '08_speaker': (r'\b(?:(?:bluetooth|wireless)\s+(?:\w+\s+){0,3}speakers?|speakers?\s+(?:\w+\s+){0,3}(?:bluetooth|wireless))\b', r'\b(?:stand|stands|mount|mounts|adapter|adapters|cable|cables|case|cases|cover|covers|battery|batteries|charger|chargers|microphone|microphones|receiver|receivers|transmitter|transmitters)\b|\b(?:without|no)\s+bluetooth\b'),
    '09_toothbrush': (r'\btooth\s*brush(?:es)?\b', r'\b(?:head|heads|holder|holders|case|cases|cover|covers|charger|chargers|stand|stands|replacement|travel box|sanitizer|sanitizers|organizer|organizers|storage|dog|dogs|cat|cats|pet|pets)\b'),
}


def _inside_project(path):
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(PROJECT):
        raise ValueError('公开查询资料只能写入当前项目')
    return path


def _normalize(text):
    return ' '.join(text.split()).casefold()


def matches_category(group, text):
    include, exclude = RULES[group]
    return bool(re.search(include, text, flags=re.I)) and not re.search(exclude, text, flags=re.I)


def _intent_exclude(group):
    return PRODUCT_INTENT_EXCLUDE + (r'|\bsalons?\b' if group == '05_hair_extension' else '')


def prepare_snapshot(root, *, download=False):
    root = _inside_project(root)
    destination = root / SNAPSHOT_NAME
    if destination.exists():
        if destination.stat().st_size != SNAPSHOT_BYTES or sha256_file(destination) != SNAPSHOT_SHA256:
            raise ValueError('公开查询来源快照与固定发布版本不符')
        return destination
    if not download:
        raise FileNotFoundError('缺少独立公开查询快照；显式添加下载开关才能获取')
    import requests
    root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.parquet.part')
    if temporary.exists():
        raise FileExistsError('存在未完成下载，请先核查或使用新语料目录')
    digest, size = hashlib.sha256(), 0
    try:
        with requests.get(SNAPSHOT_URL, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            with temporary.open('xb') as stream:
                for block in response.iter_content(1024 * 1024):
                    if not block:
                        continue
                    size += len(block)
                    if size > SNAPSHOT_BYTES:
                        raise ValueError('独立查询快照超过固定发布字节数')
                    digest.update(block); stream.write(block)
        if size != SNAPSHOT_BYTES or digest.hexdigest() != SNAPSHOT_SHA256:
            raise ValueError('独立公开查询快照未通过发布摘要校验')
        temporary.replace(destination)
    except BaseException:
        # Keep any partial bytes for an explicit audit; never call them a source.
        raise
    save_json(root / 'snapshot_manifest.json', dict(schema_version=1, dataset='Amazon Science Shopping Queries Dataset / ESCI',
        repository=REPOSITORY, revision=REVISION, url=SNAPSHOT_URL, sha256=SNAPSHOT_SHA256,
        bytes=SNAPSHOT_BYTES, origin_type='public_real_search_queries',
        test_fixture=False, is_synthetic=False, downloaded_file=destination.name))
    return destination


def collect_queries(snapshot, *, excluded_ids=()):
    import pyarrow.parquet as pq
    excluded = set(excluded_ids)
    pools = {group: {} for group in RULES}
    counts = Counter()
    category_cache = {}
    patterns = {group: (re.compile(include, re.I), re.compile(exclude, re.I))
                for group, (include, exclude) in RULES.items()}
    columns = ['example_id', 'query', 'query_id', 'product_id', 'product_locale', 'split']
    for batch in pq.ParquetFile(snapshot).iter_batches(batch_size=32768, columns=columns):
        data = batch.to_pydict()
        for index, text in enumerate(data['query']):
            counts['rows_seen'] += 1
            if data['product_locale'][index] != 'us' or data['split'][index] != 'train':
                continue
            counts['public_us_train_rows'] += 1
            if not isinstance(text, str) or not text.strip() or data['product_id'][index] in excluded:
                continue
            normalized = _normalize(text)
            if re.search(PRODUCT_INTENT_EXCLUDE, normalized, re.I):
                continue
            query_id = str(data['query_id'][index])
            if normalized not in category_cache:
                category_cache[normalized] = tuple(group for group, (include, exclude) in patterns.items()
                    if include.search(normalized) and not exclude.search(normalized))
            for group in category_cache[normalized]:
                if re.search(_intent_exclude(group), normalized, re.I):
                    continue
                prior = pools[group].get(normalized)
                evidence = dict(query_id=query_id, example_id=str(data['example_id'][index]),
                                product_id=str(data['product_id'][index]))
                if prior is None:
                    pools[group][normalized] = dict(text=text, evidence=[evidence],
                        source_dataset='esci', original_split='train', original_locale='us')
                elif len(prior['evidence']) < 3 and evidence not in prior['evidence']:
                    prior['evidence'].append(evidence)
    return pools, dict(counts)


def collect_orcas_queries(snapshot, *, groups):
    snapshot = Path(snapshot)
    if snapshot.stat().st_size != ORCAS_BYTES or sha256_file(snapshot) != ORCAS_SHA256:
        raise ValueError('独立公开搜索语料摘要不符')
    pools = {group: {} for group in groups}
    patterns = {group: (re.compile(RULES[group][0], re.I), re.compile(RULES[group][1], re.I),
                        re.compile(_intent_exclude(group), re.I))
                for group in groups}
    any_include = re.compile('|'.join(f'(?:{RULES[group][0]})' for group in groups), re.I)
    rows_seen = 0
    with gzip.open(snapshot, 'rt', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            rows_seen += 1
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 2:
                raise ValueError('官方公开查询快照出现意外列数')
            query_id, text = fields
            if not any_include.search(text):
                continue
            normalized = _normalize(text)
            for group, (include, exclude, intent_exclude) in patterns.items():
                if not include.search(normalized) or exclude.search(normalized) or intent_exclude.search(normalized):
                    continue
                pools[group].setdefault(normalized, dict(text=text,
                    source_dataset='orcas', original_split='orcas-doctrain',
                    original_locale='English query as publicly released',
                    evidence=[dict(query_id=query_id, source_line=line_number)]))
    return pools, dict(rows_seen=rows_seen)


def collect_websearch_chair_queries(subset):
    """Read original records retained from the verified complete official stream."""
    subset = Path(subset)
    if sha256_file(subset) != WEBSEARCH_SUBSET_SHA256:
        raise ValueError('公开网页搜索候选快照摘要不符')
    manifest = json.loads((subset.parent / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    if (manifest.get('original_sha256') != WEBSEARCH_SHA256 or
            manifest.get('revision') != WEBSEARCH_REVISION or
            manifest.get('original_bytes') != 711316457 or
            manifest.get('original_rows') != 9206475):
        raise ValueError('公开网页搜索原始来源身份不符')
    pool = {}
    chair_include = r"\b(?:(?:barber(?:'s)?|salon|styling|hairdresser|hairdressing|hair\s+dressing)\s+(?:\w+\s+){0,2}chairs?|chairs?\s+(?:\w+\s+){0,2}(?:salon|barber|hairdressing))\b"
    for line in subset.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        fields = row['raw_source_record'].split('\t')
        if fields != [row['query_id'], row['text'], row['languages']]:
            raise ValueError('保留的公开原始查询记录不一致')
        text = row['text']; normalized = _normalize(text)
        if (not re.search(chair_include, normalized, re.I) or
                re.search(RULES['02_chair'][1], normalized, re.I) or
                re.search(r'\b(?:shampoo|washing|dental|office)\b', normalized, re.I) or
                re.search(_intent_exclude('02_chair'), normalized, re.I)):
            continue
        pool.setdefault(normalized, dict(text=text, source_dataset='msmarco_websearch',
            original_split='train', original_locale=row['languages'],
            source_category_regex=chair_include,
            evidence=[dict(query_id=row['query_id'], source_line=row['source_line'],
                           source_record_sha256=row['source_record_sha256'])]))
    return pool


def rebuild_websearch_subset(root):
    """Explicit online reconstruction; the complete original stream is not kept."""
    import requests
    root = _inside_project(root); root.mkdir(parents=True, exist_ok=True)
    selector = re.compile(rb'barber|salon|styling|hydraulic|hairdress|beauty|shampoo|haircut|saloon|takara|koken|belmont|pibbs|meishida', re.I)
    digest = hashlib.sha256(); size = 0; lines_seen = 0; buffer = b''; records = []
    with requests.get(WEBSEARCH_URL, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        headers = {key: response.headers.get(key) for key in ['ETag', 'Last-Modified', 'Content-Length']}
        for chunk in response.iter_content(2 * 1024 * 1024):
            digest.update(chunk); size += len(chunk); buffer += chunk
            if size > 711316457:
                raise ValueError('公开原流超过冻结大小')
            lines = buffer.split(b'\n'); buffer = lines.pop()
            for raw_line in lines:
                lines_seen += 1
                if not selector.search(raw_line):
                    continue
                original = raw_line.rstrip(b'\r').decode('utf-8')
                fields = original.split('\t')
                if len(fields) != 3:
                    continue
                query_id, text, languages = fields
                if (not re.search(r'\ben-[A-Za-z]{2}\b', languages) or
                        not re.search(r'\bchairs?\b|barberchair', text, re.I)):
                    continue
                records.append(dict(query_id=query_id, text=text, languages=languages,
                    source_line=lines_seen, raw_source_record=original,
                    source_record_sha256=hashlib.sha256(raw_line).hexdigest()))
    if (buffer or size != 711316457 or lines_seen != 9206475 or digest.hexdigest() != WEBSEARCH_SHA256):
        raise ValueError('公开原流大小、行数或摘要不符')
    encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != WEBSEARCH_SUBSET_SHA256:
        raise ValueError('重建公开候选摘录摘要不符')
    destination = root / 'english_salon_chair_candidate_records.jsonl'
    if destination.exists() and destination.read_bytes() != encoded:
        raise FileExistsError('不会覆盖已有公开候选摘录')
    destination.write_bytes(encoded)
    save_json(root / 'snapshot_manifest.json', dict(dataset='Microsoft MS MARCO Web Search',
        repository='https://github.com/microsoft/MS-MARCO-Web-Search', revision=WEBSEARCH_REVISION,
        url=WEBSEARCH_URL, original_sha256=WEBSEARCH_SHA256, original_bytes=size,
        original_rows=lines_seen, headers=headers, original_split='train',
        subset_path=destination.name, subset_sha256=WEBSEARCH_SUBSET_SHA256,
        subset_records=len(records), original_full_stream_saved=False,
        offline_full_source_verification_available=False, test_fixture=False,
        is_synthetic=False, origin_type='public_real_search_queries'))
    return destination


def freeze_queries(snapshot, output, *, base, count=20, seed=42, excluded_ids=(), orcas_snapshot=None, websearch_subset=None):
    if count != 20:
        raise ValueError('本协议每组需要二十条独立公开搜索查询')
    from src.experiments.base_workflow import public_group
    base = Path(base).resolve(); output = _inside_project(output)
    pools, counts = collect_queries(snapshot, excluded_ids=excluded_ids)
    orcas_counts = None
    if orcas_snapshot is not None:
        deficient = [group for group, pool in pools.items() if len(pool) < count]
        supplements, orcas_counts = collect_orcas_queries(orcas_snapshot, groups=deficient)
        for group in deficient:
            if len(supplements[group]) > len(pools[group]):
                pools[group] = supplements[group]
    if websearch_subset is not None and len(pools['02_chair']) < count:
        chairs = collect_websearch_chair_queries(websearch_subset)
        if len(chairs) > len(pools['02_chair']):
            pools['02_chair'] = chairs
    summary = dict(schema_version=1, selection_policy=SELECTION_POLICY, seed=seed,
        source=dict(dataset='Amazon Science Shopping Queries Dataset / ESCI', repository=REPOSITORY,
            revision=REVISION, url=SNAPSHOT_URL, sha256=SNAPSHOT_SHA256),
        source_partition='English US / original train', query_use_partition='query_reserve',
        formal_queries_read=False, victim_memory_read=False, test_fixture=False,
        is_synthetic=False, excluded_product_ids_sha256=hashlib.sha256('\n'.join(sorted(excluded_ids)).encode()).hexdigest(),
        scanning=counts, orcas_scanning=orcas_counts, groups={})
    for group, pool in pools.items():
        category = public_group(base, group)['target_features']['category']
        ordered = sorted(pool.items(), key=lambda item: (hashlib.sha256(
            f'{SELECTION_POLICY}|{seed}|{group}|{item[0]}'.encode('utf-8')).hexdigest(), item[0]))
        selected = ordered[:count]
        directory = output / group
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'heldout_queries.jsonl'
        rows = [dict(id=f'{item["source_dataset"]}_{item["original_split"]}_{item["evidence"][0]["query_id"]}', text=item['text'],
            category=category, split='query_reserve', query_id=item['evidence'][0]['query_id'],
            original_split=item['original_split'], original_locale=item['original_locale'],
            source_dataset=item['source_dataset'], source_evidence=item['evidence'],
            category_assignment=dict(policy=SELECTION_POLICY, include_regex=item.get('source_category_regex', RULES[group][0]),
                                     exclude_regex=RULES[group][1],
                                     product_intent_exclude_regex=_intent_exclude(group))) for _, item in selected]
        serialized = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        if path.exists() and path.read_text(encoding='utf-8') != serialized:
            raise FileExistsError('预留公开查询已冻结，规则改变须使用新目录')
        if not path.exists():
            path.write_text(serialized, encoding='utf-8', newline='\n')
        from_orcas = bool(selected and selected[0][1]['source_dataset'] == 'orcas')
        from_websearch = bool(selected and selected[0][1]['source_dataset'] == 'msmarco_websearch')
        upstream = (dict(dataset='Microsoft ORCAS query-only', revision=ORCAS_REVISION,
                         url='https://github.com/microsoft/msmarco') if from_orcas else
                    dict(dataset='Amazon Science Shopping Queries Dataset / ESCI', revision=REVISION, url=REPOSITORY))
        original_sha256 = ORCAS_SHA256 if from_orcas else SNAPSHOT_SHA256
        original_url = ORCAS_URL if from_orcas else SNAPSHOT_URL
        original_partition = 'orcas-doctrain' if from_orcas else 'train/us'
        if from_websearch:
            upstream = dict(dataset='Microsoft MS MARCO Web Search', revision=WEBSEARCH_REVISION,
                            url='https://github.com/microsoft/MS-MARCO-Web-Search')
            original_sha256 = WEBSEARCH_SHA256; original_url = WEBSEARCH_URL
            original_partition = 'train / English-source query locales'
        source = dict(kind='heldout_search_query', split='query_reserve', test_fixture=False,
            is_synthetic=False, origin_type='public_real_search_queries',
            upstream=upstream, original_snapshot_url=original_url,
            original_snapshot_sha256=original_sha256, original_partition=original_partition,
            selection_policy=SELECTION_POLICY,
            category_rule=[rows[0]['category_assignment']['include_regex'], RULES[group][1]] if rows else list(RULES[group]),
            product_intent_exclude_regex=_intent_exclude(group),
            scope='Original public product-related search strings; Microsoft sources are general web search, not shopping logs')
        save_json(directory / 'heldout_source_metadata.json', source)
        summary['groups'][group] = dict(category=category, eligible_distinct_queries=len(pool),
            source_dataset='msmarco_websearch' if from_websearch else ('orcas' if from_orcas else 'esci'),
            requested_count=count, selected_count=len(rows), complete=len(rows) == count,
            path=str(path), sha256=sha256_file(path))
    summary['complete'] = all(row['complete'] for row in summary['groups'].values())
    save_json(output / 'heldout_query_preparation.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-dir', type=Path, default=PROJECT / 'data/public/visual_inception_corpus/esci')
    parser.add_argument('--output', type=Path, default=PROJECT / 'data/public/visual_inception_inputs')
    parser.add_argument('--base', type=Path, default=PROJECT / 'resources/evaluation_base')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-products', type=Path)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--orcas-snapshot', type=Path)
    parser.add_argument('--websearch-subset', type=Path)
    parser.add_argument('--rebuild-websearch-subset', type=Path,
        help='显式在线重建并核验公开原流，随后退出；不会下载标签或文档')
    args = parser.parse_args(argv)
    if args.rebuild_websearch_subset is not None:
        print(str(rebuild_websearch_subset(args.rebuild_websearch_subset)))
        return 0
    excluded = []
    if args.exclude_products:
        document = json.loads(args.exclude_products.read_text(encoding='utf-8'))
        excluded = document.get('ids', document.get('product_ids', [])) if isinstance(document, dict) else document
        if not isinstance(excluded, list) or not all(isinstance(v, str) for v in excluded):
            raise ValueError('排除清单必须为公开商品编号列表')
    snapshot = prepare_snapshot(args.corpus_dir, download=args.download)
    result = freeze_queries(snapshot, args.output, base=args.base, seed=args.seed,
        excluded_ids=excluded, orcas_snapshot=args.orcas_snapshot, websearch_subset=args.websearch_subset)
    print(json.dumps(dict(complete=result['complete'], groups=result['groups']), ensure_ascii=False, indent=2))
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
