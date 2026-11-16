"""Freeze three public buyer-intent templates without reading evaluation data."""
import hashlib
import json
from pathlib import Path

from src.visual.io import sha256_file
from src.visual.target_features import build_natural_buyer_intent


def text_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def content_sha256(value):
    return text_sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def prepare_semantic_targets(source, target, *, target_text=None, definition=None):
    source, target = Path(source).resolve(), Path(target).resolve()
    if not source.is_file() or not target.is_file():
        raise FileNotFoundError('缺少冻结公开图对')
    if definition is not None and not isinstance(definition, dict):
        raise ValueError('公开图对定义必须为对象')
    documents = [('explicit_definition', definition)] if definition is not None else []
    files = []
    for name in ('selection.json', 'definition.json', 'features.json'):
        path = target.parent / name
        if path.is_file():
            doc = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(doc, dict):
                raise ValueError('公开冻结文件必须为对象：' + str(path))
            documents.append((str(path), doc))
            files.append(dict(path=str(path), sha256=sha256_file(path)))
    features, frozen_texts, provenance = [], [], []
    for origin, doc in documents:
        for key in ('target_feature_text', 'alignment_canonical_text', 'canonical_sentence'):
            if doc.get(key) is not None:
                frozen_texts.append(doc[key])
        records = [doc]
        for key in ('target_product_features', 'target_features', 'target'):
            record = doc.get(key)
            if isinstance(record, dict):
                records.append(record)
        for record in records:
            category = record.get('category')
            attributes = record.get('core_attributes') or record.get('attributes')
            if record is not doc:
                for key in ('canonical_sentence', 'feature_text', 'target_feature_text'):
                    if record.get(key) is not None:
                        frozen_texts.append(record[key])
            if category is not None or attributes is not None:
                if not isinstance(category, str) or not category.strip() or not isinstance(attributes, (list, tuple)) or not attributes:
                    continue
                if any(not isinstance(a, str) or not a.strip() for a in attributes):
                    raise ValueError('公开商品特点必须为非空字符串列表')
                canonical, queries = build_natural_buyer_intent(category, attributes)
                features.append((category.strip().lower(), list(attributes), canonical, queries))
                provenance.append(origin)
        # Validate explicit image identity when it appears in public pair records.
        for role, expected in (('source', source), ('target', target)):
            record = doc.get(role)
            if isinstance(record, dict):
                digest = record.get('image_sha256') or record.get('sha256')
                if digest is not None and digest != sha256_file(expected):
                    raise ValueError('公开图对图片摘要不一致：' + role)
    if not features:
        raise ValueError('视觉记忆植入训练需要冻结的目标公开品类及特点，不能读取正式询问或猜测属性')
    # Prefer core attributes: full catalog attributes may have additional public
    # entries. Every provided feature record must still yield the same templates.
    if len({content_sha256((f[0], f[2], f[3])) for f in features}) != 1:
        raise ValueError('不同冻结公开文件中的目标特点或训练模板冲突')
    category, attributes, canonical, queries = features[0]
    if target_text is not None:
        frozen_texts.append(target_text)
    if any(not isinstance(t, str) or not t.strip() for t in frozen_texts):
        raise ValueError('冻结规范句必须为非空字符串')
    if frozen_texts and any(t.casefold() != canonical.casefold() for t in frozen_texts):
        raise ValueError('冻结规范句与公开特点三句式模板不一致')
    if not frozen_texts:
        raise ValueError('公开特点缺少已冻结规范句；不能临时发明目标文字')
    # Reviewed product text may preserve proper-name capitalization, whereas the
    # public template builder lowercases category/attributes. Keep the reviewed
    # canonical bytes; case folding is only a consistency check, not a rewrite.
    template_canonical = canonical
    canonical = target_text if target_text is not None else frozen_texts[0]
    semantic = dict(category=category, core_attributes=attributes, canonical_text=canonical,
        canonical_text_sha256=text_sha256(canonical), semantic_texts=queries,
        template_canonical_text=template_canonical,
        template_canonical_text_sha256=text_sha256(template_canonical),
        semantic_text_sha256=[text_sha256(q) for q in queries],
        semantic_targets_sha256=content_sha256(queries),
        template='public_three_buyer_intent_templates_v1', distinct_query_count=len(set(queries)),
        canonical_template_consistency='casefold_comparison_preserve_frozen_canonical_bytes',
        query_sampling='three_fixed_public_templates_no_100_query_padding',
        paper_query_distribution_reproduced=False,
        adaptation='user_requested_public_templates_replace_paper_future_query_sources_for_semantic_alignment',
        retrieval_loss_omitted=True, target_image_used_for_optimization=False,
        source_image=dict(path=str(source), sha256=sha256_file(source)),
        target_image=dict(path=str(target), sha256=sha256_file(target)),
        provenance=sorted(set(provenance)), frozen_files=files,
        explicit_definition_sha256=content_sha256(definition) if definition is not None else None)
    semantic['audit_sha256'] = content_sha256(semantic)
    return semantic
