"""视觉记忆植入（Visual Inception）专用检索测试接口。

支持两种冻结测试环境：
1. `training_oracle`（默认）：对应原论文 Table 1 主实验及 Table 6 `100% (Oracle)` 口径，
   从各组训练查询中按 50%/30%/20% 比例分层抽取 20 条（10 条公开描述 + 6 条模型改写 + 4 条真实搜索）。
2. `zero_overlap`（可选）：每组 20 条与训练正文及原特征模板 0% 字面重合、同分布的五类问法查询
   （类别直接询问、购买意图、使用场景、功能需求、选择比较各 4 条），检验跨表述语义泛化。

本模块仅检验优化后的图片在检索阶段的覆盖情况（Top-1/Top-3/Top-5 命中与名次变化），
不执行也不冒充下游推荐或购物结果。
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
import subprocess
import sys
import unicodedata

from src.experiments import shared_evaluation
from src.experiments.base_workflow import public_group
from src.experiments.defenses import conditions, table_groups
from src.experiments.paired_metrics import paired_rank_metrics
from src.paths import EVALUATION_BASE, OUTPUT, PROJECT, RESOURCES, ProjectPath as Path
from src.visual.attack_completion import validate_attack_completion
from src.visual.audit import freeze_json, read_json
from src.visual.image_utils import sha256_file

PROTOCOL = 'visual_inception_category_diversity_v1'
ORACLE_PROTOCOL = 'visual_inception_training_oracle_v1'
REQUIRED_OBJECTIVE_KIND = 'visual_inception_retrieval_semantic_perceptual_v2'

TRAINING_ORACLE_QUERY_ROOT = RESOURCES / 'visual_inception_evaluation/training_queries_oracle_v1'
ZERO_OVERLAP_QUERY_ROOT = RESOURCES / 'visual_inception_evaluation/category_queries_v1'
CATEGORY_QUERY_ROOT = ZERO_OVERLAP_QUERY_ROOT
QUERY_SUITE_ROOTS = {
    'training_oracle': TRAINING_ORACLE_QUERY_ROOT,
    'zero_overlap': ZERO_OVERLAP_QUERY_ROOT,
}
DEFAULT_QUERY_SUITE = 'training_oracle'
KEY_MODES = ('image', 'joint')
DEFAULT_KEY_MODE = 'image'

QUERY_FAMILIES = (
    'category_direct',
    'purchase_intent',
    'use_case',
    'functional_need',
    'selection_comparison',
)
QUERY_FAMILY_LABELS_ZH = {
    'category_direct': '类别直接询问',
    'purchase_intent': '购买意图',
    'use_case': '使用场景',
    'functional_need': '功能需求',
    'selection_comparison': '选择比较',
}

ORACLE_FAMILIES = (
    'category_description',
    'model_paraphrase',
    'heldout_search_query',
)
ORACLE_FAMILY_COUNTS = {
    'category_description': 10,
    'model_paraphrase': 6,
    'heldout_search_query': 4,
}
ORACLE_FAMILY_LABELS_ZH = {
    'category_description': '同类商品公开描述(50%)',
    'model_paraphrase': '模型改写查询(30%)',
    'heldout_search_query': '独立真实搜索查询(20%)',
}

SCOPE_INTERPRETATION = (
    '本结果仅反映不同测试查询下的记忆检索覆盖情况（Top-1/Top-3/Top-5 召回与名次变化）；'
    '最终推荐或购买决策是否受到影响，须在下游推荐/购物阶段单独验证。'
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_query_text(text):
    """按 unicode_nfkc_whitespace_casefold 规则规范化查询文本。"""
    _require(isinstance(text, str) and bool(text.strip()), '查询文字必须为非空字符串')
    return ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()


def resolve_query_root(query_root=None, query_suite=DEFAULT_QUERY_SUITE):
    """解析测试查询集根目录；未显式指定目录时按测试环境标识选择。"""
    if query_root is not None:
        return Path(query_root).resolve()
    if query_suite not in QUERY_SUITE_ROOTS:
        raise ValueError('未知视觉记忆植入测试环境：' + str(query_suite))
    return Path(QUERY_SUITE_ROOTS[query_suite]).resolve()


def _protocol_spec(protocol):
    if protocol == PROTOCOL:
        return QUERY_FAMILIES, {f: 4 for f in QUERY_FAMILIES}, QUERY_FAMILY_LABELS_ZH, 'zero_overlap'
    if protocol == ORACLE_PROTOCOL:
        return ORACLE_FAMILIES, dict(ORACLE_FAMILY_COUNTS), ORACLE_FAMILY_LABELS_ZH, 'training_oracle'
    raise ValueError('未知测试查询协议标识：' + str(protocol))


def _instructions_sha256(instructions):
    serialized = json.dumps(instructions, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def _resolve_relative_under(root, relative, label):
    _require(isinstance(relative, str) and bool(relative.strip()), f'{label}缺少相对路径')
    rel = Path(relative)
    _require(not rel.is_absolute() and not rel.drive and '..' not in rel.parts, f'{label}必须为相对路径')
    path = (root / rel).resolve()
    _require(path.is_relative_to(root), f'{label}路径越界')
    _require(path.is_file(), f'{label}不存在：{path}')
    return path


def load_frozen_suite(images_root, *, base=EVALUATION_BASE):
    """校验全部九组视觉记忆植入成图均已完成且冻结，不允许缺组、夹具或早期无检索版本。"""
    images_root = Path(images_root).resolve()
    base = Path(base).resolve()
    manifest = read_json(base / 'manifest.json')
    groups = list(manifest.get('groups', {}))
    _require(len(groups) == 9, '视觉记忆植入测试需要完整九组基础清单')
    source_mapping = manifest.get('shared_memory', {}).get('source_mapping', {})
    suite = []
    for group in groups:
        frozen_path = images_root / group / 'visual_inception' / 'FROZEN_IMAGE.json'
        if not frozen_path.is_file():
            raise FileNotFoundError('视觉记忆植入测试须先完成全部九组成图并冻结，尚缺：' + group)
        item = copy.deepcopy(read_json(frozen_path))
        _require(item.get('status') == 'frozen', '成图尚未冻结：' + group)
        _require(item.get('method') == 'visual_inception',
                 '专用评估仅用于 visual_inception 方法，拒绝方法：' + str(item.get('method')))
        metadata = item.get('metadata')
        _require(isinstance(metadata, dict)
                 and metadata.get('test_fixture') is False
                 and metadata.get('production_backend') is True,
                 '测试夹具或非生产后端成图不能进入正式评估：' + group)
        objective_kind = metadata.get('training_objective', {}).get('kind')
        _require(objective_kind == REQUIRED_OBJECTIVE_KIND,
                 '早期无检索版本或未知目标版本不能用于完整版本评估：' + str(objective_kind))
        image_value = item.get('image')
        _require(isinstance(image_value, str) and bool(image_value.strip()), '冻结成图缺少图片路径：' + group)
        image_path = Path(image_value)
        _require(image_path.is_file(), '冻结成图文件不存在：' + group)
        _require(sha256_file(image_path) == item.get('image_sha256'), '冻结成图摘要改变：' + group)
        public = public_group(base, group)
        _require(isinstance(item.get('source'), dict) and isinstance(item.get('target'), dict),
                 '冻结成图缺少公开图对绑定：' + group)
        _require(all(item['source'].get(k) == public['source'][k] for k in ('id', 'title', 'image_sha256')),
                 '冻结成图与当前基础包源商品不一致：' + group)
        _require(all(item['target'].get(k) == public['target'][k] for k in ('id', 'title', 'image_sha256')),
                 '冻结成图与当前基础包目标商品不一致：' + group)
        validate_attack_completion(
            metadata,
            item.get('requested_steps'),
            dict(
                method='visual_inception',
                source_image_sha256=public['source']['image_sha256'],
                target_image_sha256=public['target']['image_sha256'],
            ),
        )
        source_memory_id = source_mapping.get(group, {}).get('source_memory_id')
        _require(isinstance(source_memory_id, str) and bool(source_memory_id.strip()),
                 '基础包缺少源记忆映射：' + group)
        item['group'] = group
        item['source_memory_id'] = source_memory_id
        suite.append(item)
    return suite


def _load_training_snapshot_texts(snapshots):
    texts = set()
    for snap in snapshots:
        snap_path = Path(snap['path'])
        if not snap_path.is_absolute():
            snap_path = (PROJECT / snap_path).resolve()
        _require(snap_path.is_file(), '缺少训练来源快照以核验一致性：' + str(snap_path))
        _require(sha256_file(snap_path) == snap.get('sha256'), '训练来源快照摘要不符：' + str(snap_path))
        rows = [json.loads(line) for line in snap_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        _require(len(rows) == snap.get('row_count'), '训练来源快照行数不符：' + str(snap_path))
        for row in rows:
            texts.add(normalize_query_text(row['text']))
    return texts


def validate_category_query_suite(query_root=None, *, query_suite=DEFAULT_QUERY_SUITE,
                                  base=EVALUATION_BASE, verify_training_snapshots=True):
    """校验九组测试查询集（training_oracle 或 zero_overlap）的结构、摘要、配额及与训练查询的关系。"""
    query_root = resolve_query_root(query_root, query_suite)
    base = Path(base).resolve()
    manifest_path = query_root / 'manifest.json'
    _require(manifest_path.is_file(), '缺少测试查询集总清单：' + str(manifest_path))
    manifest = read_json(manifest_path)
    base_manifest_path = base / 'manifest.json'
    _require(base_manifest_path.is_file(), '缺少基础包清单：' + str(base_manifest_path))
    base_manifest = read_json(base_manifest_path)

    protocol = manifest.get('protocol')
    families, expected_family_counts, labels_zh, inferred_suite = _protocol_spec(protocol)
    _require(manifest.get('method') == 'visual_inception', '测试查询集方法标识必须为 visual_inception')
    _require(manifest.get('query_mode') == 'text', '测试查询集仅支持纯文字模式')
    base_groups = list(base_manifest.get('groups', {}))
    manifest_groups = manifest.get('groups', {})
    _require(isinstance(manifest_groups, dict) and list(manifest_groups) == base_groups,
             '测试查询集组列表与基础包不一致')
    _require(manifest.get('group_count') == len(base_groups), '测试查询集组数与基础包不一致')
    _require(manifest.get('queries_per_group') == 20, '每组测试查询数量必须为二十条')
    _require(manifest.get('total_queries') == len(base_groups) * 20, '测试查询总条数与组数不匹配')
    _require(manifest.get('query_families') == list(families), '测试查询分类列表不符')
    expected_manifest_qpf = 4 if protocol == PROTOCOL else expected_family_counts
    _require(manifest.get('queries_per_family') == expected_manifest_qpf, '测试查询分类配额不符')
    _require(manifest.get('optimization_may_read') is False
             and manifest.get('is_real_search_log') is False
             and manifest.get('outcome_selection_used') is False,
             '测试查询集隔离或来源声明无效')
    if manifest.get('base_manifest_sha256') is not None:
        _require(sha256_file(base_manifest_path) == manifest['base_manifest_sha256'],
                 '测试查询集绑定的基础包清单摘要不符')

    all_norm_queries = set()
    group_audits = {}
    group_norm_queries = {}

    for group in base_groups:
        entry = manifest_groups[group]
        base_spec = base_manifest['groups'][group]
        q_path = _resolve_relative_under(query_root, entry.get('path'), f'{group} 英文查询清单')
        zh_path = _resolve_relative_under(query_root, entry.get('chinese_review_path'), f'{group} 中文对照清单')
        _require(sha256_file(q_path) == entry.get('sha256'), f'{group} 英文查询清单摘要不符')
        _require(sha256_file(zh_path) == entry.get('chinese_review_sha256'), f'{group} 中文对照清单摘要不符')

        q_doc = read_json(q_path)
        zh_doc = read_json(zh_path)
        _require(q_doc.get('status') == 'ready'
                 and q_doc.get('method') == 'visual_inception'
                 and q_doc.get('evaluation_protocol') == protocol
                 and q_doc.get('group') == group
                 and q_doc.get('query_mode') == 'text'
                 and q_doc.get('training_use_allowed') is False
                 and q_doc.get('is_real_search_log') is False,
                 f'{group} 查询清单协议或隔离字段无效')
        _require(q_doc.get('target_public_id') == base_spec.get('target_id') == entry.get('target_public_id'),
                 f'{group} 查询清单目标商品编号与基础包不一致')
        if entry.get('public_features_sha256') and base_spec.get('features_sha256'):
            _require(entry['public_features_sha256'] == base_spec['features_sha256'],
                     f'{group} 公开特点摘要与基础包不一致')

        instructions = q_doc.get('instructions')
        _require(isinstance(instructions, list) and len(instructions) == 20, f'{group} 必须恰好有二十条查询')
        norm_instructions = [normalize_query_text(text) for text in instructions]
        _require(len(set(norm_instructions)) == 20, f'{group} 二十条查询规范化后必须互异')
        instr_sha = _instructions_sha256(instructions)
        _require(instr_sha == q_doc.get('instructions_sha256') == entry.get('instructions_sha256'),
                 f'{group} 查询正文列表摘要不符')

        records = q_doc.get('query_records')
        _require(isinstance(records, list) and len(records) == 20, f'{group} 查询明细数量不符')
        _require([r.get('instruction') for r in records] == instructions, f'{group} 查询明细顺序与正文列表不一致')
        record_ids = [r.get('id') for r in records]
        _require(all(isinstance(i, str) and bool(i.strip()) for i in record_ids) and len(set(record_ids)) == 20,
                 f'{group} 查询明细编号无效或重复')
        _require(all(r.get('query_family') in families for r in records), f'{group} 出现未知查询分类')
        _require(all(isinstance(r.get('instruction_zh'), str) and bool(r['instruction_zh'].strip()) for r in records),
                 f'{group} 查询明细缺少中文说明')
        if protocol == ORACLE_PROTOCOL:
            _require(all(isinstance(r.get('source_row_id'), str) and bool(r['source_row_id'].strip())
                         and isinstance(r.get('source_file'), str) and bool(r['source_file'].strip())
                         for r in records),
                     f'{group} 训练抽样明细缺少来源行号或来源文件')
        counts = dict(Counter(r['query_family'] for r in records))
        _require(counts == expected_family_counts
                 and q_doc.get('query_family_counts') == expected_family_counts
                 and entry.get('query_family_counts') == expected_family_counts,
                 f'{group} 查询分类配额不符合协议要求')

        zh_records = zh_doc.get('query_records')
        _require(zh_doc.get('group') == group and isinstance(zh_records, list) and len(zh_records) == 20,
                 f'{group} 中文对照明细缺失或数量不符')
        for rec, zrec in zip(records, zh_records):
            _require(rec['id'] == zrec.get('id')
                     and rec['query_family'] == zrec.get('query_family')
                     and rec['instruction_zh'] == zrec.get('instruction'),
                     f'{group} 中英文对照记录不一致：{rec["id"]}')

        if protocol == PROTOCOL and base_spec.get('query_manifest'):
            existing_path = (base / base_spec['query_manifest']).resolve()
            if existing_path.is_file():
                existing_doc = read_json(existing_path)
                existing_norm = {normalize_query_text(t) for t in existing_doc.get('instructions', [])
                                 if isinstance(t, str) and t.strip()}
                overlap_existing = [t for t in instructions if normalize_query_text(t) in existing_norm]
                _require(not overlap_existing, f'{group} 类别查询与既有特征评估查询存在重合')

        all_norm_queries.update(norm_instructions)
        group_norm_queries[group] = norm_instructions
        group_audits[group] = dict(
            target_public_id=q_doc['target_public_id'],
            category=q_doc.get('category'),
            query_manifest=str(q_path),
            query_manifest_sha256=entry['sha256'],
            instructions_sha256=instr_sha,
            query_count=20,
            query_family_counts=expected_family_counts,
        )

    snapshots = manifest.get('training_overlap_audit', {}).get('training_source_snapshots', [])
    training_texts = set()
    training_checked = False
    overlap_count = None if protocol == ORACLE_PROTOCOL else 0
    if verify_training_snapshots and snapshots:
        training_texts = _load_training_snapshot_texts(snapshots)
        overlap = sorted(all_norm_queries & training_texts)
        if protocol == PROTOCOL:
            _require(not overlap, '零重合测试查询与公开训练查询存在精确重合')
            overlap_count = 0
        else:
            for group, norms in group_norm_queries.items():
                missing = [t for t in norms if t not in training_texts]
                _require(not missing, f'{group} 训练抽样测试查询未全部命中冻结训练快照')
                bundle_path = (PROJECT / 'data/public/visual_inception' / group / 'training_bundle.json').resolve()
                if bundle_path.is_file():
                    bundle_doc = read_json(bundle_path)
                    bundle_norms = {normalize_query_text(q['text']) for q in bundle_doc.get('queries', [])}
                    missing_in_bundle = [t for t in norms if t not in bundle_norms]
                    _require(not missing_in_bundle, f'{group} 训练抽样查询未落在冻结训练包内')
            overlap_count = sum(len(norms) for norms in group_norm_queries.values())
        training_checked = True

    return dict(
        schema_version=1,
        protocol=protocol,
        query_suite=inferred_suite,
        training_overlap_mode='oracle_100pct' if protocol == ORACLE_PROTOCOL else 'zero_overlap',
        status='passed',
        query_root=str(query_root),
        manifest_sha256=sha256_file(manifest_path),
        base_path=str(base),
        base_manifest_sha256=sha256_file(base_manifest_path),
        group_count=len(base_groups),
        queries_per_group=20,
        total_queries=len(base_groups) * 20,
        query_families=list(families),
        query_family_labels_zh=dict(labels_zh),
        queries_per_family=expected_manifest_qpf,
        training_overlap_checked=training_checked,
        training_snapshot_count=len(snapshots) if training_checked else 0,
        unique_training_text_count=len(training_texts),
        training_normalized_exact_overlap_count=overlap_count,
        existing_formal_query_overlap_count=0 if protocol == PROTOCOL else None,
        groups=group_audits,
    )


def _paired_summary(clean_ranks, modified_ranks):
    if not clean_ranks or set(clean_ranks) != set(modified_ranks):
        raise ValueError('原图与修改图的配对名次键不一致或为空')
    keys = sorted(clean_ranks)
    for key in keys:
        c_rank, m_rank = clean_ranks[key], modified_ranks[key]
        if type(c_rank) is not int or c_rank < 1 or type(m_rank) is not int or m_rank < 1:
            raise ValueError('配对名次必须为正整数')
    n = len(keys)
    result = {
        'query_count': n,
        'mean_rank_clean': sum(clean_ranks[k] for k in keys) / n,
        'mean_rank_modified': sum(modified_ranks[k] for k in keys) / n,
        'mean_rank_gain': sum(clean_ranks[k] - modified_ranks[k] for k in keys) / n,
    }
    for k in (1, 3, 5):
        c_hits = sum(clean_ranks[key] <= k for key in keys)
        m_hits = sum(modified_ranks[key] <= k for key in keys)
        result[f'top{k}'] = {
            'clean_hits': c_hits,
            'modified_hits': m_hits,
            'clean_rate': c_hits / n,
            'modified_rate': m_hits / n,
            'gain': (m_hits - c_hits) / n,
            'new_hits': sum(clean_ranks[key] > k and modified_ranks[key] <= k for key in keys),
            'lost_hits': sum(clean_ranks[key] <= k and modified_ranks[key] > k for key in keys),
        }
    return result


def summarize_group_evaluation(eval_dir, group_query_manifest):
    """汇总单组在各防御条件与查询分类下的 Top-1/Top-3/Top-5 配对检索指标。"""
    eval_dir = Path(eval_dir).resolve()
    eval_plan = read_json(eval_dir / 'evaluation_plan.json')
    rows = read_json(eval_dir / 'evaluation.json')
    q_doc = read_json(group_query_manifest)
    _require(eval_plan.get('query_mode') == 'text', '视觉记忆植入专用检索评估仅支持纯文字模式')
    key_mode = eval_plan.get('key_mode') or ('image' if eval_plan.get('key_image_weight') == 1.0 else 'joint')
    _require(key_mode in KEY_MODES, '评估计划中的记忆键模式无效')
    expected_key_weight = 1.0 if key_mode == 'image' else 0.5
    _require(eval_plan.get('key_image_weight') == expected_key_weight, '评估计划中的记忆键图像权重与模式不一致')
    for row in rows:
        proto = row.get('retrieval_protocol', {})
        if 'key_mode' in proto:
            _require(proto['key_mode'] == key_mode, '逐条件检索快照的记忆键模式与评估计划不一致')
        if 'key_image_weight' in proto:
            _require(proto['key_image_weight'] == expected_key_weight, '逐条件检索快照的记忆键图像权重与评估计划不一致')
    protocol = q_doc.get('evaluation_protocol')
    families, _, labels_zh, inferred_suite = _protocol_spec(protocol)
    records = q_doc.get('query_records', [])
    _require(len(records) == 20, '单组查询明细必须恰好为二十条')

    expected_task_ids = [f'feature_{i:02d}' for i in range(1, 21)]
    plan_queries = eval_plan.get('queries', [])
    _require([q.get('task_id') for q in plan_queries] == expected_task_ids, '评估计划任务编号与查询清单不符')
    _require([q.get('instruction') for q in plan_queries] == [r['instruction'] for r in records],
             '评估计划查询正文与查询清单不一致')

    lookup = {(r['condition']['label'], r['bank']): r for r in rows}
    condition_labels = [spec['label'] for spec in eval_plan.get('conditions', [])]
    _require(bool(condition_labels), '评估计划缺少条件列表')
    by_condition = {}

    for label in condition_labels:
        _require((label, 'clean') in lookup and (label, 'modified') in lookup,
                 f'缺少条件 {label} 的原图或修改图评估记录')
        clean_row, mod_row = lookup[(label, 'clean')], lookup[(label, 'modified')]
        clean_obs = {q['task_id']: q for q in clean_row['queries']}
        mod_obs = {q['task_id']: q for q in mod_row['queries']}
        _require(list(clean_obs) == expected_task_ids and list(mod_obs) == expected_task_ids,
                 f'条件 {label} 的逐条查询结果数量或顺序不符')

        clean_ranks = {tid: clean_obs[tid]['rank'] for tid in expected_task_ids}
        mod_ranks = {tid: mod_obs[tid]['rank'] for tid in expected_task_ids}
        paired = paired_rank_metrics([clean_row], [mod_row])
        overall = _paired_summary(clean_ranks, mod_ranks)
        _require(abs(overall['top3']['gain'] - paired['top3_gain']) < 1e-12
                 and abs(overall['mean_rank_gain'] - paired['mean_rank_gain']) < 1e-12,
                 '配对统计校验不一致')

        by_family = {}
        for family in families:
            tids = [tid for tid, rec in zip(expected_task_ids, records) if rec['query_family'] == family]
            fam_summary = _paired_summary(
                {tid: clean_ranks[tid] for tid in tids},
                {tid: mod_ranks[tid] for tid in tids},
            )
            by_family[family] = {
                'query_family': family,
                'query_family_zh': labels_zh[family],
                **fam_summary,
            }

        details = [
            dict(
                task_id=tid,
                query_id=rec['id'],
                query_family=rec['query_family'],
                query_family_zh=labels_zh[rec['query_family']],
                instruction=rec['instruction'],
                instruction_zh=rec['instruction_zh'],
                clean_rank=clean_obs[tid]['rank'],
                modified_rank=mod_obs[tid]['rank'],
                rank_gain=clean_obs[tid]['rank'] - mod_obs[tid]['rank'],
                clean_score=clean_obs[tid]['score'],
                modified_score=mod_obs[tid]['score'],
            )
            for tid, rec in zip(expected_task_ids, records)
        ]
        by_condition[label] = dict(overall=overall, by_family=by_family, queries=details)

    primary = 'none' if 'none' in by_condition else condition_labels[0]
    return dict(
        group=q_doc['group'],
        protocol=protocol,
        query_suite=inferred_suite,
        key_mode=key_mode,
        key_image_weight=expected_key_weight,
        category=q_doc.get('category'),
        target_public_id=q_doc['target_public_id'],
        source_memory_id=eval_plan.get('source_memory_id'),
        primary_condition=primary,
        overall=by_condition[primary]['overall'],
        by_family=by_condition[primary]['by_family'],
        by_condition=by_condition,
    )


def _aggregate_condition_labels(per_group, groups, labels, *, families, labels_zh):
    clean_all = {
        (group, label, q['task_id']): q['clean_rank']
        for group in groups
        for label in labels
        for q in per_group[group]['by_condition'][label]['queries']
    }
    mod_all = {
        (group, label, q['task_id']): q['modified_rank']
        for group in groups
        for label in labels
        for q in per_group[group]['by_condition'][label]['queries']
    }
    by_family = {}
    for family in families:
        fam_clean = {
            (group, label, q['task_id']): q['clean_rank']
            for group in groups
            for label in labels
            for q in per_group[group]['by_condition'][label]['queries']
            if q['query_family'] == family
        }
        fam_mod = {
            (group, label, q['task_id']): q['modified_rank']
            for group in groups
            for label in labels
            for q in per_group[group]['by_condition'][label]['queries']
            if q['query_family'] == family
        }
        by_family[family] = {
            'query_family': family,
            'query_family_zh': labels_zh[family],
            **_paired_summary(fam_clean, fam_mod),
        }
    return dict(overall=_paired_summary(clean_all, mod_all), by_family=by_family)


def build_category_summary(eval_root):
    """汇总全部九组检索结果，分别给出总体、分类别及逐组指标。"""
    eval_root = Path(eval_root).resolve()
    plan = read_json(eval_root / 'category_evaluation_plan.json')
    validation = read_json(eval_root / 'category_query_validation.json')
    protocol = plan.get('protocol') or validation['protocol']
    families, _, labels_zh, inferred_suite = _protocol_spec(protocol)
    groups = list(plan['groups'])
    key_mode = plan.get('key_mode') or DEFAULT_KEY_MODE
    _require(key_mode in KEY_MODES, '评估计划中的记忆键模式无效')
    key_image_weight = 1.0 if key_mode == 'image' else 0.5
    if 'key_image_weight' in plan:
        _require(plan['key_image_weight'] == key_image_weight, '评估计划中的记忆键图像权重与模式不一致')
    per_group = {}
    for group in groups:
        group_query_path = Path(validation['groups'][group]['query_manifest'])
        group_summary = summarize_group_evaluation(eval_root / 'groups' / group, group_query_path)
        _require(
            group_summary['key_mode'] == key_mode and group_summary['key_image_weight'] == key_image_weight,
            f'组 {group} 记忆键编码模式与总计划不一致',
        )
        per_group[group] = group_summary

    condition_labels = [spec['label'] for spec in plan['selected_conditions']]
    primary_condition = 'none' if 'none' in condition_labels else condition_labels[0]
    by_condition = {
        label: _aggregate_condition_labels(per_group, groups, [label], families=families, labels_zh=labels_zh)
        for label in condition_labels
    }
    by_table_group = {}
    for title, labels in table_groups():
        if all(lbl in condition_labels for lbl in labels):
            by_table_group[title] = _aggregate_condition_labels(
                per_group, groups, labels, families=families, labels_zh=labels_zh
            )

    return dict(
        schema_version=1,
        protocol=protocol,
        query_suite=plan.get('query_suite', inferred_suite),
        training_overlap_mode=validation.get('training_overlap_mode'),
        method='visual_inception',
        query_mode='text',
        key_mode=key_mode,
        key_image_weight=key_image_weight,
        defense_mode=plan['defense_mode'],
        primary_condition=primary_condition,
        group_count=len(groups),
        queries_per_group=20,
        total_queries=len(groups) * 20,
        pool_count=plan['pool_count'],
        simultaneous_replacement_count=plan['simultaneous_replacement_count'],
        query_families=list(families),
        query_family_labels_zh=dict(labels_zh),
        overall=by_condition[primary_condition]['overall'],
        by_family=by_condition[primary_condition]['by_family'],
        by_condition=by_condition,
        by_table_group=by_table_group,
        groups={
            group: dict(
                group=group,
                category=per_group[group]['category'],
                target_public_id=per_group[group]['target_public_id'],
                source_memory_id=plan['frozen_images'][group]['source_memory_id'],
                overall=per_group[group]['by_condition'][primary_condition]['overall'],
                by_family=per_group[group]['by_condition'][primary_condition]['by_family'],
                by_condition=per_group[group]['by_condition'],
            )
            for group in groups
        },
        scope_notice=dict(
            retrieval_coverage_only=True,
            recommendation_or_shopping_executed=False,
            interpretation=SCOPE_INTERPRETATION,
        ),
    )


def evaluate_category_suite(
    images_root,
    output,
    *,
    base=EVALUATION_BASE,
    query_root=None,
    query_suite=DEFAULT_QUERY_SUITE,
    key_mode=DEFAULT_KEY_MODE,
    device='cuda',
    defense_mode='none',
    encoder_config=None,
    encoder_factory=None,
    defense_factory=None,
    verify_training_snapshots=True,
):
    """在完整共享记忆库上执行视觉记忆植入九组检索评估（默认使用 training_oracle 与纯图片记忆键）。"""
    if defense_mode not in ('none', 'panel'):
        raise ValueError('防御范围只能为 none 或 panel')
    if key_mode not in KEY_MODES:
        raise ValueError('记忆库检索键模式只能为 image 或 joint')
    key_image_weight = 1.0 if key_mode == 'image' else 0.5
    base = Path(base).resolve()
    resolved_query_root = resolve_query_root(query_root, query_suite)
    output = Path(output).resolve()
    if not output.is_relative_to(Path(OUTPUT).resolve()):
        raise ValueError('评估结果必须保存在当前项目 output 目录内')

    frozen_suite = load_frozen_suite(images_root, base=base)
    validation = validate_category_query_suite(
        resolved_query_root, query_suite=query_suite, base=base,
        verify_training_snapshots=verify_training_snapshots,
    )
    base_manifest = read_json(base / 'manifest.json')
    shared_cfg = base_manifest.get('shared_memory', {})
    pool_rel = shared_cfg.get('pool_file')
    _require(isinstance(pool_rel, str) and bool(pool_rel.strip()), '基础包缺少共享记忆库路径配置')
    pool_path = (base / pool_rel).resolve()
    _require(pool_path.is_relative_to(base) and pool_path.is_file(), '基础包共享记忆库路径无效')
    if shared_cfg.get('pool_sha256') is not None:
        _require(sha256_file(pool_path) == shared_cfg['pool_sha256'], '基础包共享池摘要不符')

    pool_doc = read_json(pool_path)
    pool_count = len(pool_doc.get('entries', []))
    _require(pool_count > 0, '共享记忆库为空')

    if encoder_config is not None:
        effective_encoder_config = Path(encoder_config).resolve()
        if not effective_encoder_config.is_file():
            raise FileNotFoundError('评估编码器配置不存在：' + str(effective_encoder_config))
    elif (base / 'evaluation_encoder.json').is_file():
        effective_encoder_config = (base / 'evaluation_encoder.json').resolve()
    else:
        effective_encoder_config = None

    selected_conditions = conditions()[:1] if defense_mode == 'none' else conditions()
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / 'category_query_validation.json'
    freeze_json(validation_path, validation)

    plan = dict(
        schema_version=1,
        protocol=validation['protocol'],
        query_suite=validation['query_suite'],
        method='visual_inception',
        query_mode='text',
        key_mode=key_mode,
        key_image_weight=key_image_weight,
        defense_mode=defense_mode,
        selected_conditions=selected_conditions,
        base_path=str(base),
        base_manifest_sha256=sha256_file(base / 'manifest.json'),
        query_root=str(resolved_query_root),
        query_root_manifest_sha256=sha256_file(resolved_query_root / 'manifest.json'),
        query_validation_sha256=sha256_file(validation_path),
        pool_path=str(pool_path),
        pool_sha256=sha256_file(pool_path),
        pool_count=pool_count,
        encoder_config=str(effective_encoder_config) if effective_encoder_config else None,
        encoder_config_sha256=sha256_file(effective_encoder_config) if effective_encoder_config else None,
        groups=[item['group'] for item in frozen_suite],
        simultaneous_replacement_count=len(frozen_suite),
        shared_pool_scope='all_entries_simultaneous_nine_group_replacement',
        shopping_allowed=False,
        recommendation_executed=False,
        frozen_images={
            item['group']: dict(
                image=item['image'],
                image_sha256=item['image_sha256'],
                source_memory_id=item['source_memory_id'],
                source_id=item['source']['id'],
                target_id=item['target']['id'],
                requested_steps=item['requested_steps'],
                objective_kind=item['metadata']['training_objective']['kind'],
            )
            for item in frozen_suite
        },
    )
    freeze_json(output / 'category_evaluation_plan.json', plan)

    for item in frozen_suite:
        group = item['group']
        group_eval_dir = output / 'groups' / group
        group_query_path = Path(validation['groups'][group]['query_manifest'])
        shared_evaluation.evaluate(
            item,
            group_eval_dir,
            pool_path=pool_path,
            query_manifest=group_query_path,
            source_memory_id=item['source_memory_id'],
            device=device,
            encoder_factory=encoder_factory,
            defense_factory=defense_factory,
            selected_conditions=selected_conditions,
            encoder_config=effective_encoder_config,
            query_mode='text',
            key_mode=key_mode,
            replacement_frozens=frozen_suite,
            bank_root=output / 'shared_banks',
            memory_vector_cache_dir=output / 'memory_vectors',
        )
        shared_evaluation.report(group_eval_dir)

    summary = build_category_summary(output)
    freeze_json(output / 'category_retrieval_summary.json', summary)
    report_path = report_category_suite(output)
    return dict(
        output=str(output),
        plan_path=str(output / 'category_evaluation_plan.json'),
        validation_path=str(validation_path),
        summary_path=str(output / 'category_retrieval_summary.json'),
        report_path=str(report_path),
        summary=summary,
    )


def verify_category_suite(output, *, device='cuda', encoder_factory=None, isolated_process=True):
    """对全部九组评估执行全候选分数独立复算并核对汇总一致性。"""
    output = Path(output).resolve()
    plan = read_json(output / 'category_evaluation_plan.json')
    validation_path = output / 'category_query_validation.json'
    _require(sha256_file(validation_path) == plan['query_validation_sha256'], '查询校验凭证已改变')
    _require(sha256_file(plan['pool_path']) == plan['pool_sha256'], '基础共享池在评估后改变')
    _require(sha256_file(Path(plan['query_root']) / 'manifest.json') == plan['query_root_manifest_sha256'],
             '测试查询总清单在评估后改变')

    group_verifications = {}
    total_rankings = 0
    total_scores = 0
    max_error = 0.0

    for group in plan['groups']:
        group_dir = output / 'groups' / group
        if encoder_factory is not None:
            verified = shared_evaluation.verify(
                group_dir, encoder_factory=encoder_factory, isolated_process=isolated_process
            )
        elif isolated_process:
            command = [
                sys.executable, '-B', '-X', 'utf8', '-m', 'src.experiments.shared_evaluation',
                'verify', '--output', str(group_dir.resolve()), '--device', device,
            ]
            with (group_dir / 'independent_verification.log').open('w', encoding='utf-8') as log:
                proc = subprocess.run(
                    command,
                    cwd=PROJECT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                )
            if proc.returncode:
                raise RuntimeError(f'组 {group} 独立进程复算失败，见日志：{group_dir / "independent_verification.log"}')
            verified = read_json(group_dir / 'independent_verification.json')
        else:
            verified = shared_evaluation.verify(group_dir, device=device, isolated_process=False)

        _require(verified.get('status') == 'passed' and verified.get('all_candidate_scores_checked') is True,
                 f'组 {group} 独立复算未通过')
        total_rankings += verified['rankings_recomputed']
        total_scores += verified['candidate_scores_recomputed']
        max_error = max(max_error, float(verified['max_score_error']))
        group_verifications[group] = dict(
            rankings_recomputed=verified['rankings_recomputed'],
            candidate_scores_recomputed=verified['candidate_scores_recomputed'],
            max_score_error=verified['max_score_error'],
            evaluated_file_sha256=verified['evaluated_file_sha256'],
            scores_sha256=verified['scores_sha256'],
        )

    rebuilt_summary = build_category_summary(output)
    saved_summary = read_json(output / 'category_retrieval_summary.json')
    _require(rebuilt_summary == saved_summary, '汇总与逐组评估结果不一致')

    result = dict(
        schema_version=1,
        protocol=plan['protocol'],
        query_suite=saved_summary.get('query_suite'),
        status='passed',
        groups_verified=len(plan['groups']),
        rankings_recomputed=total_rankings,
        candidate_scores_recomputed=total_scores,
        max_score_error=max_error,
        all_candidate_scores_checked=True,
        summary_sha256=sha256_file(output / 'category_retrieval_summary.json'),
        groups=group_verifications,
    )
    freeze_json(output / 'category_independent_verification.json', result)
    return result


def _format_summary_cells(stats):
    t1, t3, t5 = stats['top1'], stats['top3'], stats['top5']
    return [
        f"{t1['clean_hits']}/{stats['query_count']} ({t1['clean_rate']*100:.1f}%)",
        f"{t1['modified_hits']}/{stats['query_count']} ({t1['modified_rate']*100:.1f}%)",
        f"{t1['gain']*100:+.1f}",
        f"{t3['clean_hits']}/{stats['query_count']} ({t3['clean_rate']*100:.1f}%)",
        f"{t3['modified_hits']}/{stats['query_count']} ({t3['modified_rate']*100:.1f}%)",
        f"{t3['gain']*100:+.1f}",
        f"{t5['clean_hits']}/{stats['query_count']} ({t5['clean_rate']*100:.1f}%)",
        f"{t5['modified_hits']}/{stats['query_count']} ({t5['modified_rate']*100:.1f}%)",
        f"{t5['gain']*100:+.1f}",
        f"{stats['mean_rank_clean']:.2f} -> {stats['mean_rank_modified']:.2f} ({stats['mean_rank_gain']:+.2f})",
    ]


def report_category_suite(output):
    """生成视觉记忆植入检索评估 Markdown 报告，并校验逐组复算凭证。"""
    output = Path(output).resolve()
    plan = read_json(output / 'category_evaluation_plan.json')
    summary = read_json(output / 'category_retrieval_summary.json')
    _require(build_category_summary(output) == summary, '汇总文件与逐组原始评估结果不一致')

    total_rankings = 0
    total_scores = 0
    max_error = 0.0
    for group in plan['groups']:
        group_dir = output / 'groups' / group
        verified = read_json(group_dir / 'independent_verification.json')
        _require(
            verified.get('status') == 'passed'
            and verified.get('evaluated_file_sha256') == sha256_file(group_dir / 'evaluation.json')
            and verified.get('evaluation_plan_sha256') == sha256_file(group_dir / 'evaluation_plan.json')
            and verified.get('scores_sha256') == sha256_file(group_dir / 'scores.npz')
            and verified.get('all_candidate_scores_checked') is True,
            f'组 {group} 缺少绑定当前结果的独立复算证据',
        )
        total_rankings += verified['rankings_recomputed']
        total_scores += verified['candidate_scores_recomputed']
        max_error = max(max_error, float(verified['max_score_error']))

    overall = summary['overall']
    primary = summary['primary_condition']
    families = summary['query_families']
    labels_zh = summary['query_family_labels_zh']
    is_oracle = summary['protocol'] == ORACLE_PROTOCOL
    key_mode = summary.get('key_mode', DEFAULT_KEY_MODE)
    key_weight = summary.get('key_image_weight', 1.0 if key_mode == 'image' else 0.5)
    key_mode_desc = (
        f'纯图片向量 `image`（`key_image_weight={key_weight}`，按原论文只编码记忆库图片，不融合商品文字）'
        if key_mode == 'image' else
        f'图文各半融合向量 `joint`（`key_image_weight={key_weight}`，与本项目其他方法一致）'
    )

    title = ('# 视觉记忆植入训练查询抽样（100% Oracle）检索评估报告'
             if is_oracle else '# 视觉记忆植入类别多样化（0% 重合）检索评估报告')
    query_desc = (
        f"- 测试查询（100% Oracle 训练查询抽样）：{summary['group_count']} 组 × {summary['queries_per_group']} 条 = {summary['total_queries']} 条纯文字查询，每组从训练查询按原论文 50% / 30% / 20% 分层抽取 10 条公开描述、6 条模型改写、4 条真实搜索词（对应论文 Table 1 与 Table 6 的 `100% Oracle` 口径）。"
        if is_oracle else
        f"- 测试查询（0% 字面重合泛化）：{summary['group_count']} 组 × {summary['queries_per_group']} 条 = {summary['total_queries']} 条纯文字查询（每组含 5 类问法各 4 条：类别直接询问、购买意图、使用场景、功能需求、选择比较），与训练查询（50/30/20）及原特点评估查询规范化精确重合数均为 0。"
    )
    breakdown_heading = ('## 按训练查询来源分类汇总（公开描述 90 条 / 模型改写 54 条 / 真实搜索 36 条）'
                         if is_oracle else '## 按问法类别汇总（每类 9 组 × 4 条 = 36 条）')
    breakdown_col = '查询来源类别' if is_oracle else '问法类别'

    lines = [
        title,
        '',
        '## 实验口径与测试范围',
        '',
        f"- 评估协议：`{summary['protocol']}`（测试环境：`{summary.get('query_suite', 'training_oracle')}`，方法：`{summary['method']}`，查询模式：纯文字 `text`，主条件：`{primary}`）。",
        f"- 记忆库检索键：{key_mode_desc}。",
        f"- 共享记忆库：{summary['pool_count']} 条完整共享库，全部 {summary['simultaneous_replacement_count']} 组冻结源图同时替换进入同一个修改库（不设按组隔离小库）。",
        query_desc,
        f"- 结果边界：{summary['scope_notice']['interpretation']}",
        '',
        '## 总体检索覆盖统计（180 条查询）',
        '',
        '| 口径 | 原图前1 | 修改前1 | 前1增量(百分点) | 原图前3 | 修改前3 | 前3增量(百分点) | 原图前5 | 修改前5 | 前5增量(百分点) | 平均名次变化(原->改, 改善) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| 全部九组汇总 | ' + ' | '.join(_format_summary_cells(overall)) + ' |',
        '',
        breakdown_heading,
        '',
        f'| {breakdown_col} | 原图前1 | 修改前1 | 前1增量(百分点) | 原图前3 | 修改前3 | 前3增量(百分点) | 原图前5 | 修改前5 | 前5增量(百分点) | 平均名次变化(原->改, 改善) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for family in families:
        fam_stats = summary['by_family'][family]
        label = f"{labels_zh[family]} (`{family}`)"
        lines.append('| ' + label + ' | ' + ' | '.join(_format_summary_cells(fam_stats)) + ' |')

    lines += [
        '',
        '## 逐组检索覆盖统计（每组 20 条）',
        '',
        '| 组别 | 目标品类 | 原图前1 | 修改前1 | 前1增量(百分点) | 原图前3 | 修改前3 | 前3增量(百分点) | 原图前5 | 修改前5 | 前5增量(百分点) | 平均名次变化(原->改, 改善) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for group in plan['groups']:
        g_stats = summary['groups'][group]
        cells = _format_summary_cells(g_stats['overall'])
        lines.append(f"| `{group}` | {g_stats.get('category') or ''} | " + ' | '.join(cells) + ' |')

    if len(summary.get('by_table_group', {})) > 1:
        lines += [
            '',
            '## 防御面板汇总（全部 9 组）',
            '',
            '| 防御分组 | 原图前1 | 修改前1 | 前1增量(百分点) | 原图前3 | 修改前3 | 前3增量(百分点) | 原图前5 | 修改前5 | 前5增量(百分点) | 平均名次变化(原->改, 改善) |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
        for title_row, stats in summary['by_table_group'].items():
            lines.append('| ' + title_row + ' | ' + ' | '.join(_format_summary_cells(stats['overall'])) + ' |')

    lines += [
        '',
        '## 范围区分与独立复算凭证',
        '',
        '- 检索覆盖与下游推荐区分：上述 Top-1、Top-3、Top-5 命中与名次变化仅反映视觉记忆植入修改图在记忆检索阶段的召回覆盖情况；是否实际改变最终商品推荐或购买行为，必须另行在推荐/购物阶段独立评测。',
        f'- 本次未启动购物流程或外部策略模型（`shopping_allowed=false`，`recommendation_executed=false`）。',
        f'- 独立重新编码并复算 {total_rankings} 次排名及 {total_scores} 个全库候选分数，最大分数误差 {max_error}，容差 {shared_evaluation.SCORE_TOLERANCE}。',
        '',
    ]
    report_path = output / 'REPORT.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_validate = subparsers.add_parser('validate-queries', help='核验测试查询集配额及与训练查询的关系')
    p_validate.add_argument('--base', type=Path, default=EVALUATION_BASE)
    p_validate.add_argument(
        '--query-suite',
        choices=tuple(QUERY_SUITE_ROOTS),
        default=DEFAULT_QUERY_SUITE,
        help='测试查询环境：training_oracle 为默认（每组从训练按 10/6/4 抽样 20 条，对齐论文 Table 1）；zero_overlap 为 0 字面重合五类问法。',
    )
    p_validate.add_argument('--query-root', type=Path, default=None, help='显式覆盖测试查询目录')
    p_validate.add_argument('--skip-training-snapshots', action='store_true')

    p_eval = subparsers.add_parser('evaluate', help='对已冻结的九组视觉记忆植入成图执行检索评估')
    p_eval.add_argument('--images-root', type=Path, required=True, help='包含九组 visual_inception/FROZEN_IMAGE.json 的根目录')
    p_eval.add_argument('--output', type=Path, required=True, help='评估输出目录（须位于 output/ 内）')
    p_eval.add_argument('--base', type=Path, default=EVALUATION_BASE)
    p_eval.add_argument(
        '--query-suite',
        choices=tuple(QUERY_SUITE_ROOTS),
        default=DEFAULT_QUERY_SUITE,
        help='测试查询环境：training_oracle（默认，每组从训练查询按 10/6/4 抽样）或 zero_overlap（0 字面重合五类问法）。',
    )
    p_eval.add_argument(
        '--key-mode',
        choices=KEY_MODES,
        default=DEFAULT_KEY_MODE,
        help='记忆库检索向量模式：image（默认，按原论文只编码记忆库图片）或 joint（图文各半融合）。',
    )
    p_eval.add_argument('--query-root', type=Path, default=None, help='显式覆盖测试查询目录')
    p_eval.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p_eval.add_argument('--defense-mode', choices=('none', 'panel'), default='none')
    p_eval.add_argument('--encoder-config', type=Path)

    p_verify = subparsers.add_parser('verify', help='独立进程复算全部九组全库候选分数及汇总')
    p_verify.add_argument('--output', type=Path, required=True)
    p_verify.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')

    p_report = subparsers.add_parser('report', help='重新生成视觉记忆植入检索评估 Markdown 报告')
    p_report.add_argument('--output', type=Path, required=True)

    args = parser.parse_args(argv)
    os.environ.update(
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1',
    )

    if args.command == 'validate-queries':
        result = validate_category_query_suite(
            args.query_root,
            query_suite=args.query_suite,
            base=args.base,
            verify_training_snapshots=not args.skip_training_snapshots,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'evaluate':
        result = evaluate_category_suite(
            args.images_root,
            args.output,
            base=args.base,
            query_root=args.query_root,
            query_suite=args.query_suite,
            key_mode=args.key_mode,
            device=args.device,
            defense_mode=args.defense_mode,
            encoder_config=args.encoder_config,
        )
        print(json.dumps(dict(
            output=result['output'],
            query_suite=result['summary']['query_suite'],
            key_mode=result['summary']['key_mode'],
            key_image_weight=result['summary']['key_image_weight'],
            protocol=result['summary']['protocol'],
            report_path=result['report_path'],
            summary_path=result['summary_path'],
            overall=result['summary']['overall'],
        ), ensure_ascii=False, indent=2))
        return 0
    if args.command == 'verify':
        result = verify_category_suite(args.output, device=args.device, isolated_process=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    report_path = report_category_suite(args.output)
    print(str(report_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
