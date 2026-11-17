"""Bounded, reproducible sampling of public catalogue fields only.

The catalogue's ``query`` column is a public category label, never a search log.
The caller supplies exclusion identifiers; this module cannot open a memory bank.
No image download, model loading, shopping session or evaluation happens here.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
from urllib.parse import urlsplit


SAMPLING_POLICY = 'indexed_public_category_seeded_subtype_sample_v5'
IMAGE_HOSTS = frozenset(('m.media-amazon.com', 'images-na.ssl-images-amazon.com',
    'images-eu.ssl-images-amazon.com', 'images.amazon.com', 'ecx.images-amazon.com'))

# These are public product-type rules, fixed before sampling or image encoding.
# The negative rules remove accessories and unmistakably different products.
SUBTYPE_RULES = {
    'camera': (r'\b(?:security|surveillance|ptz|ip|wifi|wireless|outdoor|indoor)\b.*\bcameras?\b|\bcameras?\b.*\b(?:security|surveillance|ptz|ip|wifi|wireless|outdoor|indoor)\b',
               r'\b(?:dummy|fake|mounts?|brackets?|housing|covers?|cables?|adapters?|connectors?|cords?|junction|enclosure|monitors?|screens?|chargers?|protective case|rain shield|hard drive|hdd|dvr|nvr|systems?|recorders?|drones?|power supply|illuminators?|infrared led lights?)\b'),
    'chair': (r'\b(?:barber|hair\s?styling|stylist)\b.*\bchairs?\b|\bchairs?\b.*\b(?:barber|hair\s?styling|stylist)\b',
              r'\b(?:covers?|mats?|pads?|beds?|stools?|backwash|bowl|sink|station|boosters?|cushions?|hair dryer|footrest|replacement|parts?|pump only|chair base only)\b'),
    'cookies': (r'\bcookie\s+dough\b',
                r'\b(?:cutter|sheet|pan|jar|mold|mould|glitter|powder|sprinkles?|frosting|icing|fondant|decorations?|toppers?|toys?|play|playdoh|mix(?:es)?|ingredients?|bars?|ice cream|frozen dessert|no bake cookie dough bites|cereal|coffee|tea|candles?|soap|lip balm|sticker|decal|ornament|stamper|cookie butter|cookie boxes|cookie bags|cake|cupcake|pancake|waffle)\b'),
    'curtain': (r'\b(?:curtains?|drapes?)\b',
                r'\b(?:shower|curtain rod|curtain pole|curtain ring|curtain clip|curtain hook|tieback|tie back|curtain holdback|pull rod|pull wand|curtain wand|drapery pull|curtain track|curtain chain|curtain cord|blinds?|roller shades?|creepy cloth|haunted house|door decorations)\b'),
    'hair_extension': (r'\b(?:hair extensions?|clip[ -]?in(?:\s+\w+){0,4}\s+hair|hair wefts?|hair bundles?)\b',
                       r'\b(?:wigs?|bangs|hairpieces?|hair pieces?|pony[ -]?tails?|buns?|headbands?|tools?|kits?|pliers?|glue|extension tape|mannequin|practice head|tinsel|feathers?|clippers?|scissors?|needles?|rings?|beads?|device|storage|bags?|hangers?|caps?|guards?|pads?|separator|protectors?|wires?|accessor(?:y|ies)|tassles?|tassels?|paranda|parandi|tape tabs?|replacement tape|adhesive tape|tape roll|stands?|holders?|racks?|cadd(?:y|ies)|dolls?|remover|connector|fusion gun|duckbill|alligator|pin curl)\b'),
    'jeans': (r'\b(?:jeans|denim pants|denim trousers)\b', r'\b(?:jeans button|jeans patch|jeans repair|jeans buckle|imitation|jeans print|jeggings|leggings|sweatpants)\b'),
    'shampoo': (r'\bshampoos?\b', r'\b(?:dry shampoo|brush|scrubber|massager|shampoo bowl|shampoo sink|shampoo cap|shampoo chair|dogs?|cats?|pets?|horses?|empty|refillable|dispensers?|containers?|travel bottles|replacement|wigs?|universal pump)\b'),
    'speaker': (r'\bbluetooth\b.*\b(?:speakers?|loudspeakers?)\b|\b(?:speakers?|loudspeakers?)\b.*\bbluetooth\b',
                r'\b(?:stands?|mounts?|brackets?|cables?|wires?|covers?|cases?|bags?|grilles?|grills?|adapters?|chargers?|cords?|holders?|replacement|power supply|soundbars?|coaxial|car speakers?|car stereo|marine|boats?|atv|utv|golf cart|mini-amp|accessory kit|subwoofer|transmitter|receiver|motherboard)\b'),
    'toothbrush': (r'\btooth\s?brush(?:es)?\b',
                  r'\b(?:replacement|spare|refills?|holders?|cases?|covers?|chargers?|stands?|containers?|sanitizer|accessory kit|storage|wall[ -]?mount|caps?|bags?|pouches?|oral irrigator|tartar remover|plaque remover|scaler|battery cell|solder tabs|ni-mh)\b'),
}

# A target may be misfiled in the original catalogue. These named public
# taxonomy neighbours are fixed independently of learned similarity or attack
# outcomes; the title rule must still hold for every selected item.
PUBLIC_CATEGORY_NEIGHBOURS = {
    'cookies': ('breads & bakery', 'snack food gifts', 'snack cakes & pastries'),
    'hair_extension': ('hair extensions, wigs & accessories',),
    'shampoo': ('shampoo & conditioner',),
    'toothbrush': ('toothbrushes & accessories',),
    'speaker': ('speaker', 'speakers'),
}

# Explicit public-title review of the earlier unconsumed draft. These are
# category decisions only: no ranking, model score, user query or test result.
PUBLIC_TITLE_REJECTIONS = {
    'B0776SG16F': 'security_camera_hard_drive_not_a_camera',
    'B018S08N4I': 'toy_drone_not_a_standalone_security_camera',
    'B092V48SWP': 'network_recorder_only_not_a_camera',
    'B08YJNTS7G': 'digital_recorder_only_not_a_camera',
    'B07PMC7D1P': 'digital_recorder_only_not_a_camera',
    'B08D1VXS42': 'digital_recorder_only_not_a_camera',
    'B07YYL6W35': 'replacement_chair_base_not_a_chair',
    'B07W6TMFJF': 'replacement_hydraulic_pump_not_a_chair',
    'B00O30QRB4': 'drawstring_hair_bun_not_hair_extensions',
    'B09B23QT63': 'display_stand_not_hair_extensions',
    'B07M5K7FPQ': 'doll_hair_material_not_personal_hair_extensions',
    'B088Y9N314': 'holding_caddy_not_hair_extensions',
    'B091Y5PJG2': 'washing_rack_not_hair_extensions',
    'B07PRFMJ3F': 'extension_remover_not_hair_extensions',
    'B09MS77NCP': 'heat_connector_tool_not_hair_extensions',
    'B07L8N3Y74': 'duckbill_clips_not_hair_extensions',
    'B09MRXF7SG': 'imitation_jeans_leggings_not_denim_jeans',
    'B09PJ7PGHH': 'jeans_print_trousers_not_denim_jeans',
    'B08J7L73CC': 'empty_pump_bottles_not_shampoo',
    'B071ZM2D2F': 'dispenser_pump_not_shampoo',
    'B00W8QLJYK': 'wig_care_kit_not_scalp_shampoo',
    'B081RCJTM5': 'speaker_hard_case_not_a_speaker',
    'B08FSV2NTW': 'power_adapter_not_a_speaker',
    'B08XVTF2V2': 'silicone_case_not_a_speaker',
    'B07JB71XHQ': 'travel_case_not_a_speaker',
    'B07Y5WNB64': 'charging_cable_not_a_speaker',
    'B09DPNBYM5': 'travel_case_not_a_speaker',
    'B07YCF5PKZ': 'silicone_case_not_a_speaker',
    'B07NKWYWC5': 'speaker_hard_case_not_a_speaker',
    'B07M9S9PK3': 'replacement_battery_not_a_speaker',
    'B09R9HD8N5': 'travel_pouch_not_a_toothbrush',
    'B08NFDW21H': 'spare_brush_head_not_a_toothbrush',
    'B004Z107D0': 'brush_covers_not_a_toothbrush',
    'B09N6KRC7P': 'brush_head_covers_not_a_toothbrush',
    'B07X8YHBMN': 'replacement_brush_heads_not_a_toothbrush',
    'B09P328CJG': 'electric_tartar_removal_tool_not_a_brush',
    'B01EWRHXMO': 'coconutmilk_frozen_dessert_not_cookie_dough',
    'B086B73RWG': 'almond_milk_frozen_dessert_not_cookie_dough',
    'B087F8J5PL': 'ready_to_make_cookie_dough_kit_not_prepared_dough',
    'B086PDJH4S': 'ready_in_fifteen_minutes_kit_not_prepared_dough',
    'B071V8HPTM': 'security_camera_power_supply_not_a_camera',
    'B01GZYS3C6': 'infrared_illuminator_not_a_camera',
    'B09CQBDSDH': 'salon_booster_cushion_not_a_chair',
    'B01EXPVKMY': 'hair_dryer_waiting_chair_not_barber_styling_chair',
    'B08CTVH54N': 'halloween_creepy_cloth_prop_not_window_curtains',
    'B081DZHHXB': 'brand_contains_jeans_but_product_is_twill_crop_pants',
    'B097S59H5P': 'brand_contains_jeans_but_product_is_athleisure_jogger',
    'B07JZDR6HR': 'pump_dispenser_caps_not_shampoo',
    'B082W5MR8V': 'travel_refill_bottles_not_shampoo',
    'B09DKWRYBX': 'scalp_massager_brush_not_shampoo',
    'B07Z2KDT26': 'vehicle_cone_speakers_with_amplifier_not_portable_bluetooth_speaker',
    'B09K3ZJMSC': 'mounted_marine_vehicle_speakers_not_portable_bluetooth_speaker',
    'B00H28SFBO': 'tablet_accessory_bundle_not_standalone_bluetooth_speaker',
    'B09KP9CVH8': 'nickel_hydride_battery_cell_not_a_toothbrush',
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def _file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_excluded_ids(path):
    """Read an explicit ID-only handoff, not a shared-memory file."""
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(document, dict):
        _require('excluded_ids' in document or 'ids' in document, '排除清单对象必须含 excluded_ids 或 ids 字段')
        document = document.get('excluded_ids', document.get('ids'))
    _require(isinstance(document, list) and all(isinstance(item, str) and
        re.fullmatch(r'[A-Za-z0-9]{10}', item) for item in document), '排除清单必须仅提供商品编号数组')
    return set(document)


def _decode_json_field(value):
    if isinstance(value, str) and value.lstrip().startswith(('[', '{')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def extract_public_description(fields):
    """Choose existing prose/bullets/attributes; retain exact raw evidence.

    Bullet arrays are joined using newlines, an explicitly recorded formatting
    operation. No language model, feature invention or buyer template is used.
    """
    for name in ('Description', 'full_description', 'BulletPoints', 'small_description', 'Attributes'):
        raw = fields.get(name)
        value = _decode_json_field(raw)
        if isinstance(value, str):
            parts = [value.strip()]
            operation = 'strip_outer_whitespace_only'
        elif isinstance(value, list):
            parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            operation = 'public_array_items_joined_by_newline'
        elif isinstance(value, dict):
            parts = [str(key).strip() + ': ' + str(item).strip() for key, item in value.items()
                     if isinstance(item, (str, int, float)) and str(item).strip()]
            operation = 'public_attribute_pairs_joined_by_newline'
        else:
            continue
        parts = [part for part in parts if part.casefold() not in
                 ('dummy_attr', 'none', 'null', 'n/a', 'nan', 'no description')]
        text = '\n'.join(parts)
        if len(text) < 24 or not re.search(r'[A-Za-z\u4e00-\u9fff]', text):
            continue
        return dict(text=text, field=name, original_value=raw,
                    formatting_operation=operation,
                    source_type='product_description' if name in ('Description', 'full_description')
                                else 'public_bullet_points' if name in ('BulletPoints', 'small_description')
                                else 'public_attributes')
    return None


def extract_public_description_items(fields):
    """Return verbatim original prose/sentences/bullets, with precise indices."""
    result, seen = [], set()
    for field in ('Description', 'full_description', 'BulletPoints', 'small_description', 'Attributes'):
        raw = fields.get(field)
        value = _decode_json_field(raw)
        if isinstance(value, str):
            items = [(value, 'whole_field', 0, 0, len(value))]
            cursor = 0
            for index, fragment in enumerate(re.split(r'(?<=[.!?])\s+(?=[A-Z])', value)):
                start = value.find(fragment, cursor)
                items.append((fragment, 'sentence', index, start, start + len(fragment)))
                cursor = start + len(fragment)
        elif isinstance(value, list):
            items = [(fragment, 'array_item', index, None, None)
                     for index, fragment in enumerate(value) if isinstance(fragment, str)]
        else:
            continue
        for original, kind, index, start, end in items:
            text = original.strip()
            key = ' '.join(text.split()).casefold()
            if len(text) < 16 or key in seen or key in ('dummy_attr', 'no description', 'none', 'null'):
                continue
            seen.add(key)
            result.append(dict(text=text, field=field, item_kind=kind, item_index=index,
                start_offset=start, end_offset=end, original_value=raw,
                original_item=original, formatting_operation='strip_outer_whitespace_only'))
    return result


def prepare_public_description_item_group(catalog, group_name, group, output, *,
        excluded_ids, catalog_identity, seed=42, count=10, description_count=50,
        max_per_product=5, max_bucket_ids=25000):
    """Freeze one scarce public class using distinct verbatim source items.

    This opt-in path is for cookie dough: fifty public text rows may come from
    fewer than fifty distinct products. It never fabricates paraphrases. All
    eligible product IDs in the explicitly bounded taxonomy frame are reviewed
    before an independent seed orders the candidates and round-robin items.
    """
    _require(group_name == '03_cookies' and group['target_features']['category'] == 'cookie dough',
             '公开原文条目模式仅对已确认的曲奇面团组显式开放')
    _require(type(max_per_product) is int and 1 <= max_per_product <= 5, '单个商品最多五条公开原文')
    source, output = Path(catalog).resolve(), Path(output).resolve()
    _require(not output.exists(), '单组输出目录已存在，不能覆盖冻结资料')
    stat = source.stat()
    _require((stat.st_size, stat.st_mtime_ns) ==
        (catalog_identity['byte_size'], catalog_identity['mtime_ns']), '公开目录与冻结元数据身份不符')
    excluded = set(excluded_ids) | {group['source']['id'], group['target']['id']}
    positive, negative = SUBTYPE_RULES['cookies']
    labels, products, bucket_counts = None, [], {}
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA cache_size=-8192')
        target = _public_projection(db, group['target']['id'])
        _require(target is not None, '曲奇面团目标缺少公开目录记录')
        labels = sorted(set((target['public_category_label'],) + PUBLIC_CATEGORY_NEIGHBOURS['cookies']))
        for label in labels:
            rows = db.execute("SELECT asin,json_extract(data,'$.name') FROM products "
                'INDEXED BY product_query_lookup WHERE query=? ORDER BY rowid LIMIT ?',
                (label, max_bucket_ids + 1)).fetchall()
            _require(len(rows) <= max_bucket_ids, '公开分类桶超过声明上限')
            bucket_counts[label] = len(rows)
            for asin, title in rows:
                if asin in excluded or asin in PUBLIC_TITLE_REJECTIONS or not isinstance(title, str):
                    continue
                if not re.search(positive, title[:160], re.I) or re.search(negative, title, re.I):
                    continue
                product = _public_projection(db, asin)
                image = _public_image(product['fields'])
                items = extract_public_description_items(product['fields'])
                if image is not None and items:
                    products.append(dict(id=asin, title=title, image_url=image, category='cookie dough',
                        rowid=product['rowid'], public_category_label=label,
                        catalog_category=product['catalog_category'], items=items))
    after = source.stat()
    _require((after.st_size, after.st_mtime_ns) == (stat.st_size, stat.st_mtime_ns), '读取期间目录发生变化')
    _require(len(products) >= count, '严格同类公开商品不足所需的十张独立候选')
    order_seed = _digest(dict(policy='indexed_public_category_seeded_order_v1', seed=seed,
        group=group_name, public_category_labels=labels, catalog_snapshot_id=catalog_identity['snapshot_id']))
    products.sort(key=lambda product: product['id'])
    random.Random(int(order_seed, 16)).shuffle(products)
    for product in products:
        random.Random(int(_digest(dict(seed=seed, product_id=product['id'], policy='public_original_items_v1')), 16)).shuffle(product['items'])
    descriptions, seen = [], set()
    for item_round in range(max_per_product):
        for product in products:
            if len(descriptions) >= description_count:
                break
            if item_round >= len(product['items']):
                continue
            item = product['items'][item_round]
            key = ' '.join(item['text'].split()).casefold()
            if key in seen:
                continue
            seen.add(key)
            provenance = dict(catalog_snapshot_id=catalog_identity['snapshot_id'], rowid=product['rowid'],
                product_id=product['id'], public_category_label=product['public_category_label'],
                catalog_category=product['catalog_category'], description_field=item['field'],
                description_item_kind=item['item_kind'], description_item_index=item['item_index'],
                description_start_offset=item['start_offset'], description_end_offset=item['end_offset'],
                description_original_value=item['original_value'], description_original_item=item['original_item'],
                description_formatting_operation=item['formatting_operation'])
            descriptions.append(dict(id=f"{product['id']}:{item['field']}:{item['item_kind']}:{item['item_index']}",
                text=item['text'], category='cookie dough', product_id=product['id'],
                title=product['title'], public_source=provenance))
    _require(len(descriptions) == description_count, '严格同类商品的独立原文条目不足五十条；未生成替代文字')
    candidates = [dict(id=product['id'], title=product['title'], image_url=product['image_url'],
        category='cookie dough', sampling_rank=index + 1,
        public_source=dict(catalog_snapshot_id=catalog_identity['snapshot_id'], rowid=product['rowid'],
            product_id=product['id'], public_category_label=product['public_category_label'],
            catalog_category=product['catalog_category'])) for index, product in enumerate(products)]
    _write_jsonl(output / 'descriptions.jsonl', descriptions)
    _write_jsonl(output / 'candidate_reserve.jsonl', candidates)
    audit = dict(schema_version=1, policy=SAMPLING_POLICY, group=group_name,
        source_id=group['source']['id'], target_id=group['target']['id'], category='cookie dough',
        catalog_identity=catalog_identity, public_category_labels=labels,
        indexed_category_bucket_counts=bucket_counts, indexed_title_rows_read=sum(bucket_counts.values()),
        strict_eligible_product_count=len(products), requested_candidate_count=count,
        candidate_reserve_count=len(candidates), description_count=len(descriptions),
        description_distinct_product_count=len({row['product_id'] for row in descriptions}),
        description_mode='verbatim_public_field_items_multiple_items_per_product',
        description_max_items_per_product=max_per_product,
        description_actual_items_per_product={product_id: sum(row['product_id'] == product_id for row in descriptions)
            for product_id in sorted({row['product_id'] for row in descriptions})},
        positive_title_pattern=positive, negative_title_pattern=negative,
        public_title_rejections=PUBLIC_TITLE_REJECTIONS, public_title_rejections_sha256=_digest(PUBLIC_TITLE_REJECTIONS),
        seed=seed, group_order_seed_sha256=order_seed, excluded_ids_sha256=_digest(sorted(excluded)),
        gallery_downloaded=False, gallery_finalized=False, private_fields_read=False,
        descriptions_sha256=_file_digest(output / 'descriptions.jsonl'),
        candidate_reserve_sha256=_file_digest(output / 'candidate_reserve.jsonl'),
        fallback_policy='first_successful_images_in_frozen_sampling_rank_order_without_model_scores')
    _write_json(output / 'sampling_manifest.json', audit)
    return audit


def _subtype_key(group_name, group):
    aliases = dict(seating='chair', curtains='curtain', apparel='jeans', audio='speaker')
    categories = [group.get('target_features', {}).get('category'), group.get('category')]
    for category in categories:
        if category:
            key = aliases.get(str(category).casefold(), str(category).casefold())
            if key in SUBTYPE_RULES:
                return key
    for key in sorted(SUBTYPE_RULES, key=len, reverse=True):
        if key in group_name:
            return key
    raise ValueError('没有预先登记的公开商品细分类规则：' + group_name)


def _public_projection(db, asin):
    names = ('name', 'title', 'Description', 'full_description', 'BulletPoints',
             'small_description', 'Attributes', 'images')
    columns = ','.join("json_extract(data, '$." + name + "')" for name in names)
    row = db.execute('SELECT rowid,asin,category,query,' + columns +
                     ' FROM products WHERE asin=?', (asin,)).fetchone()
    if row is None:
        return None
    return dict(rowid=row[0], id=row[1], catalog_category=row[2],
                public_category_label=row[3], fields=dict(zip(names, row[4:])))


def _public_image(fields):
    images = _decode_json_field(fields.get('images'))
    if not isinstance(images, list):
        return None
    for image in images:
        if not isinstance(image, str):
            continue
        parsed = urlsplit(image)
        if parsed.scheme == 'https' and parsed.hostname in IMAGE_HOSTS and not parsed.username and not parsed.password:
            if 'transparent-pixel' not in image.casefold():
                return image
    return None


def prepare_public_catalog(catalog, groups, output, *, excluded_ids, seed=42,
        count=10, description_count=50, reserve_count=150, max_read_per_group=2500,
        max_bucket_ids=25000, catalog_sha256=None):
    """Freeze public descriptions and an ordered, not-yet-downloaded gallery.

    ``groups`` is a mapping of public_group-style records. ``candidate_reserve``
    includes the first ``count`` suggestions plus ``reserve_count`` fallbacks.
    The downstream downloader must select the first successful images in this
    fixed order, never use embedding/attack scores to choose replacements.
    """
    _require(isinstance(groups, dict) and bool(groups), '必须提供公开商品组定义')
    _require(all(type(value) is int and value > 0 for value in
        (count, description_count, max_read_per_group, max_bucket_ids)), '抽样数量与读取上限必须为正整数')
    _require(type(reserve_count) is int and reserve_count >= 0, '候补数量必须为非负整数')
    _require(type(seed) is int, '随机种子必须为整数')
    _require(catalog_sha256 is None or re.fullmatch(r'[a-f0-9]{64}', catalog_sha256), '目录摘要格式无效')
    excluded = set(excluded_ids)
    _require(all(isinstance(item, str) and re.fullmatch(r'[A-Za-z0-9]{10}', item)
                 for item in excluded), '排除清单含无效商品编号')
    for group in groups.values():
        for role in ('source', 'target'):
            excluded.add(str(group[role]['id']))
    source, output = Path(catalog).resolve(), Path(output).resolve()
    _require(source.is_file(), '公开商品目录不存在')
    _require(not output.exists(), '输出目录已存在；冻结样本不得覆盖，请使用新路径')
    before = source.stat()
    requested = max(description_count, count + reserve_count)
    _require(requested <= max_read_per_group, '读取上限小于所需候选数量')
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as db:
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA cache_size=-8192')
        db.execute('PRAGMA mmap_size=0')
        schema = db.execute("SELECT name,sql FROM sqlite_master WHERE name IN "
                            "('products','product_query_lookup') ORDER BY name").fetchall()
        _require(len(schema) == 2, '目录缺少已登记的商品表与公开分类索引')
        identity = dict(file_name=source.name, byte_size=before.st_size,
            mtime_ns=before.st_mtime_ns, public_schema_sha256=_digest(schema))
        if catalog_sha256 is not None:
            identity.update(sha256=catalog_sha256, sha256_verification='caller_supplied_preverified_source_manifest')
        else:
            identity['sha256_verification'] = 'not_rehashed_metadata_snapshot_identity_only'
        identity['snapshot_id'] = _digest(identity)
        results, globally_used = {}, set()
        for name, group in sorted(groups.items()):
            _require(re.fullmatch(r'[A-Za-z0-9_-]+', name) is not None, '商品组名称不能包含路径字符')
            target = _public_projection(db, str(group['target']['id']))
            _require(target is not None, '目标商品不在公开完整目录中：' + name)
            label = target['public_category_label']
            _require(isinstance(label, str) and bool(label.strip()), '目标缺少公开分类标签：' + name)
            subtype = _subtype_key(name, group)
            positive, negative = SUBTYPE_RULES[subtype]
            # This indexed read is limited to category identifiers, not product
            # bodies, and its explicit cap prevents an accidental global scan.
            labels = sorted(set((label,) + PUBLIC_CATEGORY_NEIGHBOURS.get(subtype, ())))
            ids, bucket_counts = [], {}
            for category_label in labels:
                bucket = [row[0] for row in db.execute('SELECT asin FROM products INDEXED BY product_query_lookup '
                    'WHERE query=? ORDER BY rowid LIMIT ?', (category_label, max_bucket_ids + 1))]
                _require(len(bucket) <= max_bucket_ids, '公开分类编号桶超过上限，请显式提高 max_bucket_ids：' + name)
                bucket_counts[category_label] = len(bucket)
                ids.extend(bucket)
            ids = sorted(set(ids))
            order_seed = _digest(dict(policy='indexed_public_category_seeded_order_v1', seed=seed, group=name,
                public_category_labels=labels, catalog_snapshot_id=identity['snapshot_id']))
            random.Random(int(order_seed, 16)).shuffle(ids)
            accepted, scanned, reasons, seen_text = [], 0, {}, set()
            category = str(group.get('target_features', {}).get('category') or group.get('category') or subtype)
            for asin in ids:
                if scanned >= max_read_per_group or len(accepted) >= requested:
                    break
                if asin in PUBLIC_TITLE_REJECTIONS:
                    reasons['explicit_public_title_category_rejection'] = reasons.get('explicit_public_title_category_rejection', 0) + 1
                    continue
                if asin in excluded or asin in globally_used:
                    reasons['excluded_or_previously_selected'] = reasons.get('excluded_or_previously_selected', 0) + 1
                    continue
                scanned += 1
                product = _public_projection(db, asin)
                title = product['fields'].get('name') or product['fields'].get('title')
                if not isinstance(title, str) or not re.search(positive, title[:160], re.I) or re.search(negative, title, re.I):
                    reasons['public_title_subtype_mismatch'] = reasons.get('public_title_subtype_mismatch', 0) + 1
                    continue
                image = _public_image(product['fields'])
                description = extract_public_description(product['fields'])
                if image is None or description is None:
                    reasons['missing_public_image_or_description'] = reasons.get('missing_public_image_or_description', 0) + 1
                    continue
                normalized = ' '.join(description['text'].split()).casefold()
                if normalized in seen_text:
                    reasons['duplicate_description'] = reasons.get('duplicate_description', 0) + 1
                    continue
                seen_text.add(normalized)
                provenance = dict(catalog_snapshot_id=identity['snapshot_id'], rowid=product['rowid'],
                    product_id=asin, public_category_label=product['public_category_label'], catalog_category=product['catalog_category'],
                    description_field=description['field'], description_original_value=description['original_value'],
                    description_formatting_operation=description['formatting_operation'],
                    description_source_type=description['source_type'])
                accepted.append(dict(id=asin, title=title, image_url=image, category=category,
                    text=description['text'], public_source=provenance, sampling_rank=len(accepted) + 1))
            _require(len(accepted) >= requested, f'{name} 在公开读取上限内仅找到 {len(accepted)} 条，少于所需 {requested}；未写出不完整训练数据')
            globally_used.update(row['id'] for row in accepted)
            results[name] = dict(rows=accepted, audit=dict(group=name, source_id=str(group['source']['id']),
                target_id=str(group['target']['id']), category=category, subtype=subtype,
                positive_title_pattern=positive, negative_title_pattern=negative,
                positive_title_scope_first_characters=160,
                target_public_category_label=label, public_category_labels=labels,
                indexed_category_bucket_counts=bucket_counts, indexed_category_bucket_size=len(ids),
                catalog_public_projections_read=scanned, max_read_per_group=max_read_per_group,
                requested_candidate_count=count, description_count=description_count,
                reserve_count=reserve_count, candidate_reserve_count=count + reserve_count,
                seed=seed, group_order_seed_sha256=order_seed, rejection_counts=reasons,
                gallery_downloaded=False, gallery_finalized=False,
                suggested_gallery_description_overlap=min(count, description_count),
                fallback_policy='first_successful_images_in_frozen_sampling_rank_order_without_model_scores'))
        after = source.stat()
        _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), '读取期间公开目录发生变化')
    manifest = dict(schema_version=1, policy=SAMPLING_POLICY, catalog_identity=identity,
        subtype_rules_sha256=_digest(SUBTYPE_RULES), category_neighbours_sha256=_digest(PUBLIC_CATEGORY_NEIGHBOURS),
        public_title_rejections=PUBLIC_TITLE_REJECTIONS,
        public_title_rejections_sha256=_digest(PUBLIC_TITLE_REJECTIONS),
        seed=seed, excluded_ids=sorted(excluded), excluded_ids_sha256=_digest(sorted(excluded)),
        private_fields_read=False, formal_memory_opened=False, model_loaded=False,
        catalog_query_usage='public_category_label_only_never_a_search_log',
        public_fields_projected=['name', 'title', 'Description', 'full_description',
            'BulletPoints', 'small_description', 'Attributes', 'images', 'category'],
        disjoint_across_groups=True, groups={})
    for name, result in results.items():
        directory = output / name
        descriptions = [dict(id=row['id'], text=row['text'], category=row['category'],
            product_id=row['id'], title=row['title'], public_source=row['public_source'])
            for row in result['rows'][:description_count]]
        candidates = [dict(id=row['id'], title=row['title'], image_url=row['image_url'],
            category=row['category'], sampling_rank=row['sampling_rank'], public_source=row['public_source'])
            for row in result['rows'][:count + reserve_count]]
        _write_jsonl(directory / 'descriptions.jsonl', descriptions)
        _write_jsonl(directory / 'candidate_reserve.jsonl', candidates)
        audit = dict(result['audit'], catalog_identity=identity,
            descriptions_sha256=_file_digest(directory / 'descriptions.jsonl'),
            candidate_reserve_sha256=_file_digest(directory / 'candidate_reserve.jsonl'),
            excluded_ids_sha256=manifest['excluded_ids_sha256'])
        _write_json(directory / 'sampling_manifest.json', audit)
        manifest['groups'][name] = dict(directory=name,
            descriptions=f'{name}/descriptions.jsonl', descriptions_sha256=audit['descriptions_sha256'],
            candidate_reserve=f'{name}/candidate_reserve.jsonl', candidate_reserve_sha256=audit['candidate_reserve_sha256'],
            sampling_manifest=f'{name}/sampling_manifest.json')
    _write_json(output / 'manifest.json', manifest)
    return manifest
