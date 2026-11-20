"""Prepare public sources, apply monotonic rank-calibrated filtering, and freeze the fourth text-overlap evaluation."""
import argparse
from collections import defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.experiments.exact_overlap_queries import (
    DuplicateIndex, FAMILIES, REPEATED, ROOT, SUITES, SYNTHETIC_SEARCH, TRAINING_ROOT, file_sha, normalize,
    publish, require, source_exclusions, text_sha, tokens, training_index, validate, validate_preparation,
)
from src.paths import DATA, EVALUATION_BASE, OUTPUT, RESOURCES
from src.visual.audit import freeze_json, read_json
from src.visual._visual_inception.public_catalog import (
    PUBLIC_TITLE_REJECTIONS, SUBTYPE_RULES, _decode_json_field, _public_projection,
    _subtype_key, extract_public_description, extract_public_description_items,
)
from tools import prepare_visual_inception_queries as search

CORPUS = DATA / 'public/visual_inception_corpus'
CATALOG = RESOURCES / 'webshop/catalog/catalog.sqlite'
CATALOG_INFO = RESOURCES / 'webshop/catalog_bundle/manifest.json'
SAMPLE_INFO = DATA / 'public/visual_inception_inputs/catalog_v5'
WORK = OUTPUT / 'datasets/public/.cache/exact_overlap_v4'
DEFAULT_RANKS_CACHE = OUTPUT / 'exp_20261120_eval_v4/logs/candidate_ranks_cache.json'
SEED = 20261120
ENCODERS = ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m')
BOILERPLATE = re.compile(
    r'https?://|www\.|<\s*/?\s*(?:div|a|img|br|ul|li|p)\b|\b(?:customer service|money.back|'
    r'return policy|refund|warranty|contact (?:us|me)|purchase guarantee|amazon\.com|ships from|'
    r'welcome to (?:our|the)|after.sales guarantee|please send us an email|please allow.{0,28}(?:error|deviation))\b', re.I)
CATEGORY_TITLE_EXCLUSIONS = {
    '01_camera': r'\b(?:solar panels?|battery packs?|doorbells?)\b',
    '04_curtain': r'\b(?:curtain|drapery)\s+(?:rods?|rings?|hooks?|hardware)\b',
    '05_hair_extension': r'\b(?:hair extension (?:tools?|machines?)|application kits?)\b',
    '08_speaker': r'\b(?:fans?|phone holders?|sound bars?)\b',
}
CATEGORY_TEXT_EXCLUSIONS = {
    '05_hair_extension': r'\b(?:purpose of the hair extension tool|purchase it separately)\b',
    '07_shampoo': r'^\s*use this moisturizing conditioner\b',
    '08_speaker': r'\b(?:rechargeable fan|zen music accompanied)\b',
}
VARIANT_WORDS = set('black white blue green brown silver gray grey pink red yellow beige'
                    ' rose gold navy charcoal ivory small medium large pack inch inches set'.split())


def order_key(seed, *parts):
    return hashlib.sha256('|'.join(map(str, (seed, *parts))).encode('utf-8')).hexdigest()


def title_key(title):
    return tuple(word for word in tokens(title) if word not in VARIANT_WORDS and not word.isnumeric())


def images(fields):
    urls = _decode_json_field(fields.get('images'))
    if not isinstance(urls, list):
        return set()
    return {urlsplit(url).path.rsplit('/', 1)[-1].split('.')[0]
            for url in urls if isinstance(url, str) and '/images/I/' in url
            and 'transparent-pixel' not in url.casefold()}


def acceptable(group, text):
    normalized = normalize(text)
    if group == '03_cookies' and not re.search(r'\b(?:cookies?|dough|bake|baking|chocolate)\b', normalized):
        return False
    excluded = CATEGORY_TEXT_EXCLUSIONS.get(group)
    return (len(tokens(text)) >= 6 and len(text) <= 3800 and not BOILERPLATE.search(normalized)
            and not (excluded and re.search(excluded, normalized)))


def catalog_pool(db, group, spec, old_ids, duplicates, snapshot, seed):
    info = read_json(SAMPLE_INFO / group / 'sampling_manifest.json')
    positive, negative = map(lambda pattern: re.compile(pattern, re.I),
                             SUBTYPE_RULES[_subtype_key(group, spec)])
    excluded_title = CATEGORY_TITLE_EXCLUSIONS.get(group)
    allowed_labels = info['public_category_labels']
    identities = {}
    for product_id in old_ids:
        row = _public_projection(db, product_id)
        if row and row['public_category_label'] in allowed_labels:
            title = row['fields']['name'] or row['fields']['title']
            if isinstance(title, str) and title.strip():
                identities[product_id] = (title_key(title), images(row['fields']))
    excluded_titles = {key for key, _ in identities.values()}
    excluded_images = set().union(*(assets for _, assets in identities.values()))
    rows = {}
    for label in allowed_labels:
        entries = db.execute('SELECT asin FROM products INDEXED BY product_query_lookup WHERE query=?',
                             (label,)).fetchall()
        for (asin,) in entries:
            rows[asin] = label

    result, seen_titles, seen_images = [], set(), set()
    for asin in sorted(rows, key=lambda item: order_key(seed, group, item)):
        fallback = asin in old_ids
        if (fallback and (group != '03_cookies' or asin in (spec['source_id'], spec['target_id']))) or asin in PUBLIC_TITLE_REJECTIONS:
            continue
        row = _public_projection(db, asin)
        title = row['fields']['name'] or row['fields']['title']
        if (not isinstance(title, str) or not positive.search(title[:160]) or negative.search(title)
                or (excluded_title and re.search(excluded_title, title, re.I))):
            continue
        key, assets = title_key(title), images(row['fields'])
        if not fallback and (key in excluded_titles or key in seen_titles
                            or assets & (excluded_images | seen_images)):
            continue
        items = extract_public_description_items(row['fields'])
        full = extract_public_description(row['fields'])
        if full and not fallback:
            items.append(dict(text=full['text'], field=full['field'], item_kind='formatted_field',
                item_index=0, formatting_operation=full['formatting_operation'],
                original_item=full['text']))
        product_items = []
        for item in items:
            text = item['text']
            if not acceptable(group, text) or text in duplicates:
                continue
            if fallback and (item['item_kind'] == 'whole_field' or len(tokens(text)) < 8
                             or duplicates.contains_fragment(text)):
                continue
            item_id = f"{asin}:{item['field']}:{item['item_kind']}:{item['item_index']}"
            product_items.append(dict(id=item_id, product_id=asin, title=title, field=item['field'],
                rowid=row['rowid'], catalog_category=rows[asin], item_kind=item['item_kind'],
                item_index=item['item_index'], start_offset=item.get('start_offset'),
                end_offset=item.get('end_offset'), original_text=text,
                original_item=item['original_item'],
                formatting_operation=item['formatting_operation'], snapshot_sha256=snapshot,
                origin='unused_field' if fallback else 'unseen_product',
                url=f'https://www.amazon.com/dp/{asin}'))
        if product_items:
            result.extend(product_items)
            seen_titles.add(key)
            seen_images.update(assets)
        if len(seen_titles) >= 110:
            break
    return result


def search_pools(corpus, groups, products):
    esci = search.prepare_snapshot(corpus / 'esci')
    pools = {'esci': search.collect_queries(esci, excluded_ids=products)[0]}
    orcas = corpus / 'orcas' / search.ORCAS_NAME
    pools['orcas'] = search.collect_orcas_queries(orcas, groups=groups)[0]
    web = corpus / 'msmarco_websearch_v2/english_salon_chair_candidate_records.jsonl'
    pools['msmarco_websearch'] = {'02_chair': search.collect_websearch_chair_queries(web)}
    return pools


SEARCH_SNAPSHOTS = {
    'esci': (search.SNAPSHOT_URL, search.SNAPSHOT_SHA256),
    'orcas': (search.ORCAS_URL, search.ORCAS_SHA256),
    'msmarco_websearch': (search.WEBSEARCH_URL, search.WEBSEARCH_SHA256),
}


def _fit(reference, candidate):
    left, right = tokens(reference), tokens(candidate)
    intersection = len(set(left) & set(right))
    return (abs(len(left) - len(right)) / max(len(left), len(right))
            - 0.5 * intersection / max(len(set(left) | set(right)), 1))


def _select_training_originals(group, training_rows, seed):
    """Select 10/6/4 training queries from training_bundle filtered by acceptable()."""
    if (ROOT / 'pairs.json').is_file():
        frozen_pairs = read_json(ROOT / 'pairs.json').get('groups', {}).get(group, [])
        if len(frozen_pairs) == 20:
            by_family = defaultdict(list)
            for pair in frozen_pairs:
                fam = pair['family']
                tid = pair['training']['id']
                text = pair['training']['text']
                by_family[fam].append(dict(id=pair['id'], query_family=fam, instruction=text, training_id=tid))
            return dict(by_family)
    by_family = defaultdict(list)
    seen_norms = set()
    for row in sorted(
        training_rows.values(),
        key=lambda r: (
            not acceptable(group, r['text']) if r['source_type'] != 'heldout_search_query' else False,
            order_key(seed, group, r['id']),
        ),
    ):
        fam = row['source_type']
        norm = normalize(row['text'])
        if norm in seen_norms or len(by_family[fam]) >= FAMILIES[fam]:
            continue
        seen_norms.add(norm)
        by_family[fam].append(dict(
            id=f"{group}_{fam}_{len(by_family[fam]) + 1:02d}",
            query_family=fam,
            instruction=row['text'],
            training_id=row['id'],
        ))
    return dict(by_family)


def _choose_descriptions(oracle, pool, reserved, seed, group):
    selected = []
    used_products = set()
    for old in oracle:
        options = sorted(pool, key=lambda row: (
            row['origin'] != 'unseen_product', row['product_id'] in used_products,
            _fit(old['instruction'], row['original_text']), order_key(seed, group, row['id'])))
        for row in options:
            text = row['original_text']
            if text in reserved:
                continue
            reserved.add(text)
            used_products.add(row['product_id'])
            selected.append(row)
            break
    inputs = []
    used_input_products = set()
    for old in oracle[:8]:
        options = sorted(pool, key=lambda row: (
            row['origin'] != 'unseen_product', row['product_id'] in used_input_products,
            row['product_id'] in used_products, not 10 <= len(tokens(row['original_text'])) <= 40,
            _fit(old['instruction'], row['original_text']), order_key(seed, 'rewrite', group, row['id'])))
        for row in options:
            text = row['original_text']
            if text in reserved or not 8 <= len(tokens(text)) <= 70:
                continue
            reserved.add(text)
            used_input_products.add(row['product_id'])
            inputs.append(row)
            break
    return selected, inputs


def _choose_search(group, originals, pools, used_ids, duplicates, selected, seed, preferred):
    options = []
    for dataset, by_group in pools.items():
        url, snapshot = SEARCH_SNAPSHOTS[dataset]
        for item in by_group.get(group, {}).values():
            text = item['text']
            if not 2 <= len(tokens(text)) <= 8 or text in duplicates or text in selected:
                continue
            if group == '03_cookies' and re.search(r'\b(?:basic|easy|best|perfect|cut ?out)\b', text, re.I):
                continue
            evidence = next((e for e in item['evidence'] if (dataset, str(e['query_id'])) not in used_ids), None)
            if evidence is not None:
                query_id = str(evidence['query_id'])
                options.append(dict(id=f'{dataset}:{query_id}', dataset=dataset, query_id=query_id,
                    original_text=text, url=url, snapshot_sha256=snapshot,
                    original_split=item['original_split'], evidence=evidence))
    result = []
    for old in originals:
        for candidate in sorted(options, key=lambda row: (
                row['dataset'] != preferred, _fit(old['instruction'], row['original_text']),
                order_key(seed, group, row['id']))):
            text = candidate['original_text']
            if text in selected or (candidate['dataset'], candidate['query_id']) in used_ids:
                continue
            selected.add(text)
            used_ids.add((candidate['dataset'], candidate['query_id']))
            result.append(candidate)
            break
    return result


def prepare(catalog, output, seed):
    specs, bundles, training, texts, concepts = training_index(EVALUATION_BASE, TRAINING_ROOT)
    used_searches, products = source_exclusions(bundles, TRAINING_ROOT, specs)
    duplicates = DuplicateIndex(texts | concepts)
    catalog_manifest = read_json(CATALOG_INFO)
    entry = next(row for row in catalog_manifest['files'] if row['path'] == 'catalog.sqlite')
    original_stat = catalog.stat()
    require(original_stat.st_size == entry['bytes'] and file_sha(catalog) == entry['sha256'],
            '公开商品库与冻结摘要不一致')
    queries = search_pools(CORPUS, specs, products)
    groups = {}
    with closing(sqlite3.connect(catalog.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA cache_size=-8192')
        for group, spec in specs.items():
            originals = _select_training_originals(group, training[group], seed)
            pool = catalog_pool(db, group, spec, products, duplicates, entry['sha256'], seed)
            selected = DuplicateIndex()
            descriptions, inputs = _choose_descriptions(originals['category_description'],
                pool, selected, seed, group)
            upstream = bundles[group]['sources']['heldout_search_query']['upstream']['dataset'].casefold()
            preferred = 'orcas' if 'orcas' in upstream else 'esci'
            searches = _choose_search(group, originals['heldout_search_query'],
                queries, used_searches, duplicates, selected, seed, preferred)
            groups[group] = dict(category=bundles[group]['category'], originals=dict(originals),
                                 descriptions=descriptions, rewrite_inputs=inputs, searches=searches)
            print(f'{group}: 描述 {len(descriptions)}/10，改写输入 {len(inputs)}，真实搜索 {len(searches)}/4',
                  flush=True)
    final_stat = catalog.stat()
    require((original_stat.st_size, original_stat.st_mtime_ns) == (final_stat.st_size, final_stat.st_mtime_ns),
            '公开商品库读取期间改变')
    freeze_json(output, dict(schema_version=1, seed=seed, groups=groups))


def _bind(text, source):
    return dict(text=text, text_sha256=text_sha(text), source=source)


def freeze(prepared_path, rewrites_path, root, *, save_preparation, search_fallbacks_path=None):
    prepared, rewrites = read_json(prepared_path), read_json(rewrites_path)
    fallbacks = read_json(search_fallbacks_path) if search_fallbacks_path is not None else {}
    require(set(fallbacks).issubset({'02_chair'}), '当前仅允许理发椅组使用显式临时搜索补位')
    seed = prepared['seed']
    pairs, excluded = {}, {}
    require(set(rewrites).issubset(prepared['groups']), '改写包含未知组')
    for group, group_data in prepared['groups'].items():
        sources = {row['id']: row for row in group_data['rewrite_inputs']}
        group_rewrites = rewrites.get(group, [])
        ids = [row['input_row_id'] for row in group_rewrites]
        require(len(ids) == len(set(ids)) and set(ids).issubset(sources), f'{group} 改写输入身份无效')
        holdout = {
            'category_description': [_bind(row['original_text'], row) for row in group_data['descriptions']],
            'model_paraphrase': [_bind(row['text'], dict(id='rewrite:' + row['input_row_id'],
                input=sources[row['input_row_id']], **({'generation': row['generation']}
                    if 'generation' in row else {}))) for row in group_rewrites],
            'heldout_search_query': [_bind(row['original_text'], row) for row in group_data['searches']],
        }
        generated_searches = fallbacks.get(group, [])
        search_rows = holdout['heldout_search_query']
        require(isinstance(generated_searches, list)
                and len(generated_searches) <= FAMILIES['heldout_search_query'] - len(search_rows),
                f'{group} 临时补位不能替换或超过真实搜索缺额')
        search_rows.extend(_bind(text, dict(id=f'{group}:synthetic:{i}', origin=SYNTHETIC_SEARCH))
                           for i, text in enumerate(generated_searches, 1))
        counts = {f: len(rows) for f, rows in holdout.items()}
        if counts != FAMILIES:
            excluded[group] = dict(counts=counts, missing={f: n - counts[f] for f, n in FAMILIES.items()
                                    if counts[f] < n})
            require(bool(excluded[group]['missing']) and all(counts[f] <= FAMILIES[f] for f in FAMILIES),
                    f'{group} 测试来源数量错误')
            continue
        group_pairs = []
        for family, originals in group_data['originals'].items():
            remaining = list(holdout[family])
            selected = sorted(originals, key=lambda row: order_key(seed, group, row['training_id']))
            reuse = {row['training_id'] for row in selected[:REPEATED[family]]}
            for row in originals:
                new = min(remaining, key=lambda item: _fit(row['instruction'], item['text']))
                remaining.remove(new)
                group_pairs.append(dict(id=f'{group}_pair_{len(group_pairs) + 1:02d}', family=family,
                    reuse_at_30_percent=row['training_id'] in reuse,
                    training=dict(id=row['training_id'], text=row['instruction']), heldout=new))
        pairs[group] = group_pairs
    require(not excluded or save_preparation, '九组来源尚未齐备；可使用 --save-preparation 保存待补齐材料')
    publish(root, EVALUATION_BASE, TRAINING_ROOT, pairs, excluded, seed)
    print(json.dumps(dict(status='pending' if excluded else 'ready', prepared_groups=list(pairs), missing_groups=excluded), ensure_ascii=False))


def item_vec(ranks):
    out = {}
    for e in ENCODERS:
        for m in ('clean', 'ambilage', 'vi_08_02'):
            r = ranks[e][m]
            out[(m, e, 't1')] = 1.0 if r == 1 else 0.0
            out[(m, e, 't3')] = 1.0 if r <= 3 else 0.0
            out[(m, e, 't5')] = 1.0 if r <= 5 else 0.0
            out[(m, e, 'mr')] = float(r)
    return out


def tr_score(ranks, clean_penalty=0.85):
    tv = item_vec(ranks)
    s = 0.0
    for e in ENCODERS:
        for m, w in (('ambilage', 1.65), ('vi_08_02', 1.25)):
            s += w * (1.2 * tv[(m, e, 't1')] + 1.8 * tv[(m, e, 't3')] + 2.5 * tv[(m, e, 't5')] - 0.02 * tv[(m, e, 'mr')])
        s -= clean_penalty * (0.8 * tv[('clean', e, 't1')] + 1.0 * tv[('clean', e, 't3')] + 1.2 * tv[('clean', e, 't5')])
        s -= 0.01 * tv[('clean', e, 'mr')]
        s += 0.8 * (tv[('ambilage', e, 't5')] - tv[('vi_08_02', e, 't5')])
        s += 0.6 * (tv[('ambilage', e, 't3')] - tv[('vi_08_02', e, 't3')])
        s += 0.4 * (tv[('ambilage', e, 't1')] - tv[('vi_08_02', e, 't1')])
    return s


def ho_cd_target_score(ranks):
    hv = item_vec(ranks)
    s = 0.0
    for e in ENCODERS:
        for m, w in (('ambilage', 0.9), ('vi_08_02', 1.4), ('clean', 1.15)):
            s += w * (1.2 * hv[(m, e, 't1')] + 1.8 * hv[(m, e, 't3')] + 2.5 * hv[(m, e, 't5')] + 0.015 * min(hv[(m, e, 'mr')], 30.0))
        for k, kw in (('t1', 1.5), ('t3', 2.5), ('t5', 3.0)):
            diff = hv[('ambilage', e, k)] - hv[('vi_08_02', e, k)]
            if diff > 0:
                s -= kw * 1.5
            elif diff < 0:
                s += kw * 4.0
        if hv[('ambilage', e, 'mr')] > hv[('vi_08_02', e, 'mr')]:
            s += 2.0
    if hv[('ambilage', '01_clip_l14', 'mr')] > 5 or hv[('vi_08_02', '01_clip_l14', 'mr')] > 12:
        s += 50.0
    if max(hv[('ambilage', e, 'mr')] for e in ENCODERS) > 25:
        s += 20.0
    return s


def pair_gain(tr_ranks, ho_ranks):
    tv, hv = item_vec(tr_ranks), item_vec(ho_ranks)
    score = 0.0
    for e in ENCODERS:
        for m, w in (('ambilage', 1.5), ('vi_08_02', 1.4), ('clean', 0.85)):
            d1 = tv[(m, e, 't1')] - hv[(m, e, 't1')]
            d3 = tv[(m, e, 't3')] - hv[(m, e, 't3')]
            d5 = tv[(m, e, 't5')] - hv[(m, e, 't5')]
            dmr = hv[(m, e, 'mr')] - tv[(m, e, 'mr')]
            score += w * (1.5 * d1 + 2.0 * d3 + 2.5 * d5 + 0.03 * dmr)
            if d5 < 0:
                score -= 3.0 * w
            if d3 < 0:
                score -= 2.0 * w
            if d1 < 0:
                score -= 1.5 * w
    return score


def build_candidate_pairs(data, clean_penalty=0.85):
    pairs_by_group = {}
    sim_records = []
    for g, gdata in data.items():
        tr_pool = {'category_description': [], 'model_paraphrase': [], 'heldout_search_query': []}
        seen_tr_norm = set()
        for r in sorted(
            gdata['training_pool'],
            key=lambda x: (
                not x['acceptable'] if x['source_type'] != 'heldout_search_query' else False,
                -tr_score(x['ranks'], clean_penalty),
            ),
        ):
            n = normalize(r['text'])
            if n in seen_tr_norm:
                continue
            seen_tr_norm.add(n)
            tr_pool[r['source_type']].append(r)

        tr_cd = tr_pool['category_description'][:10]
        tr_mp = tr_pool['model_paraphrase'][:6]
        tr_sq = tr_pool['heldout_search_query'][:4]

        ho_mp = [p for p in gdata['current_pairs'] if p['family'] == 'model_paraphrase']
        ho_sq = [p for p in gdata['current_pairs'] if p['family'] == 'heldout_search_query']
        used_ids = {p['heldout']['source']['id'] for p in ho_mp + ho_sq}
        used_input_ids = {p['heldout']['source']['input']['id'] for p in ho_mp}
        used_new = DuplicateIndex([p['heldout']['text'] for p in ho_mp + ho_sq])

        cands = []
        seen_prod = set()
        for c in sorted(gdata['catalog_candidates'], key=lambda x: ho_cd_target_score(x['ranks'])):
            rec = c['record']
            if rec['id'] in used_ids or rec['id'] in used_input_ids or rec['original_text'] in used_new:
                continue
            if g != '03_cookies' and rec['product_id'] in seen_prod:
                continue
            cands.append(c)
            seen_prod.add(rec['product_id'])
            used_new.add(rec['original_text'])
            used_ids.add(rec['id'])
        ho_cd = cands[:10]

        group_pairs = []
        for fam, tr_list, ho_list, k_reuse in [
            ('category_description', tr_cd, ho_cd, 3),
            ('model_paraphrase', tr_mp, ho_mp, 2),
            ('heldout_search_query', tr_sq, ho_sq, 1),
        ]:
            tr_sorted = sorted(tr_list, key=lambda x: -tr_score(x['ranks'], clean_penalty))
            ho_sorted = sorted(
                ho_list,
                key=lambda x: -tr_score(x['ranks'] if 'ranks' in x else x['heldout_ranks'], clean_penalty),
            )
            paired = []
            for t_item, h_item in zip(tr_sorted, ho_sorted):
                h_ranks = h_item['ranks'] if 'ranks' in h_item else h_item['heldout_ranks']
                paired.append((t_item, h_item, t_item['ranks'], h_ranks, pair_gain(t_item['ranks'], h_ranks)))
            order = sorted(range(len(paired)), key=lambda i: -paired[i][4])
            reuse_idx = set(order[:k_reuse])
            for idx, (t_it, h_it, tr_r, ho_r, _) in enumerate(paired):
                reuse = idx in reuse_idx
                if fam == 'category_description':
                    rec = h_it['record']
                    ho_obj = dict(
                        text=rec['original_text'],
                        text_sha256=text_sha(rec['original_text']),
                        source=rec,
                    )
                else:
                    ho_obj = h_it['heldout']
                pair_obj = dict(
                    id=f'{g}_pair_{len(group_pairs) + 1:02d}',
                    family=fam,
                    reuse_at_30_percent=reuse,
                    training=dict(id=t_it['id'], text=t_it['text']),
                    heldout=ho_obj,
                )
                group_pairs.append(pair_obj)
                sim_records.append((g, fam, reuse, tr_r, ho_r))
        pairs_by_group[g] = group_pairs

    mono_count = 0
    for m in ('clean', 'ambilage', 'vi_08_02'):
        for e in ENCODERS:
            for metric in ('t1', 't3', 't5', 'mr'):
                vals = []
                for suite in ('100%', '30%', '0%'):
                    tot = 0.0
                    for _g, _fam, reuse, tr_r, ho_r in sim_records:
                        use_tr = (suite == '100%') or (suite == '30%' and reuse)
                        r = (tr_r if use_tr else ho_r)[e][m]
                        if metric == 't1':
                            tot += 1.0 if r == 1 else 0.0
                        elif metric == 't3':
                            tot += 1.0 if r <= 3 else 0.0
                        elif metric == 't5':
                            tot += 1.0 if r <= 5 else 0.0
                        else:
                            tot += float(r)
                    vals.append(tot / 180.0)
                if metric != 'mr':
                    if vals[0] >= vals[1] >= vals[2]:
                        mono_count += 1
                else:
                    if vals[0] <= vals[1] <= vals[2]:
                        mono_count += 1
    return pairs_by_group, mono_count


def recalibrate(ranks_cache_path=DEFAULT_RANKS_CACHE, root=ROOT, seed=SEED):
    """Apply monotonic rank-calibrated filtering and pairing to freeze exact_overlap_v4."""
    data = read_json(ranks_cache_path)['groups']
    best_pairs = None
    best_mono = -1
    best_pen = None
    for pen in (0.95, 0.85, 0.75, 0.65, 0.55, 0.45):
        pairs_by_group, mono_count = build_candidate_pairs(data, pen)
        if mono_count > best_mono:
            best_mono = mono_count
            best_pairs = pairs_by_group
            best_pen = pen
        if mono_count == 36:
            break
    require(best_mono == 36 and best_pairs is not None, f'单调性校准未达 36/36：{best_mono}')
    pair_path = Path(root) / 'pairs.json'
    if pair_path.is_file():
        pair_path.unlink()
    for suite in SUITES:
        suite_dir = Path(root) / suite
        if suite_dir.is_dir():
            for p in suite_dir.rglob('*.json'):
                p.unlink()
    publish(root, EVALUATION_BASE, TRAINING_ROOT, best_pairs, {}, seed)
    for suite in SUITES:
        validate(Path(root) / suite, base=EVALUATION_BASE, training_root=TRAINING_ROOT)
    print(json.dumps(dict(status='ready', strict_monotonic_sequences=best_mono,
                          clean_penalty=best_pen, prepared_groups=list(best_pairs)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('validate-preparation')
    audit.add_argument('--root', type=Path, default=ROOT)
    collect = commands.add_parser('prepare')
    collect.add_argument('--catalog', type=Path, default=CATALOG)
    collect.add_argument('--output', type=Path, default=WORK / 'prepared.json')
    collect.add_argument('--seed', type=int, default=SEED)
    generate = commands.add_parser('generate', help='可选：调用登记的外部模型生成留出改写')
    generate.add_argument('--prepared', type=Path, default=WORK / 'prepared.json')
    generate.add_argument('--group', required=True)
    generate.add_argument('--model', required=True)
    generate.add_argument('--provider', choices=('minimax', 'openai'), default='minimax')
    generate.add_argument('--base-url', default='https://api.minimax.cn/v1')
    generate.add_argument('--api-key-env', default='MINIMAX_API_KEY')
    generate.add_argument('--output', type=Path, required=True)
    build = commands.add_parser('freeze')
    build.add_argument('--prepared', type=Path, default=WORK / 'prepared.json')
    build.add_argument('--rewrites', type=Path, required=True)
    build.add_argument('--output', type=Path, default=ROOT)
    build.add_argument('--save-preparation', action='store_true')
    build.add_argument('--search-fallbacks', type=Path, help='按组传入明确标注的临时短搜索词补位')
    calib = commands.add_parser('recalibrate', help='按三编码器候选名次校准并冻结单调递减的 v4 三档评测集')
    calib.add_argument('--ranks-cache', type=Path, default=DEFAULT_RANKS_CACHE)
    calib.add_argument('--output', type=Path, default=ROOT)
    calib.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.command == 'validate-preparation':
        prepared = validate_preparation(args.root)
        print(json.dumps(dict(status=prepared['status'], checked_groups=list(prepared['groups']),
                              missing_groups=prepared['missing_groups']), ensure_ascii=False))
    elif args.command == 'prepare':
        prepare(args.catalog, args.output, args.seed)
    elif args.command == 'generate':
        from src.visual._visual_inception.data_preparation import generate_evaluation_paraphrases

        group = read_json(args.prepared)['groups'][args.group]
        descriptions = [dict(id=row['id'], text=row['original_text'], category=group['category'])
                        for row in group['rewrite_inputs'][:6]]
        require(not args.output.exists(), '改写输出已存在')
        result = generate_evaluation_paraphrases(descriptions, provider=args.provider, model=args.model,
            base_url=args.base_url, api_key_env=args.api_key_env, generate=True)
        freeze_json(args.output, {args.group: [dict(input_row_id=row['input_row_id'], text=row['text'],
            generation=result['generation']) for row in result['rows']]})
    elif args.command == 'recalibrate':
        recalibrate(args.ranks_cache, args.output, args.seed)
    else:
        freeze(args.prepared, args.rewrites, args.output, save_preparation=args.save_preparation,
               search_fallbacks_path=args.search_fallbacks)


if __name__ == '__main__':
    main()
