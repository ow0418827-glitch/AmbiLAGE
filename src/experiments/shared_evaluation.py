"""冻结成图后的共享长期记忆检索评估、真实复算和报告。"""
import copy
import math
import os
import subprocess
import sys
import uuid
import numpy as np
from src.paths import ProjectPath as Path

from src.paths import SHARED_MEMORY
from src.agent.memory.shared_pool import load_shared_pool, evaluation_banks
from src.visual.audit import freeze_json, read_json, save_json, digest
from src.visual.image_utils import sha256_file
from src.visual.visual_retrieval import VisualMemoryRetriever, memory_index_text, resolve_image, detect_index_text_policy
from src.experiments.defenses import Defender, conditions, table_groups
from src.experiments.query_contract import load_formal_queries, query_retriever_options, verify_query_input


SCORE_TOLERANCE = 1e-7
KEY_MODES = ('image', 'joint')


def default_key_mode_for_method(method: str | None) -> str:
    """按方法选择默认记忆键编码模式：visual_inception 仅编码图片，其他方法使用图文联合。"""
    if isinstance(method, str) and method.strip().lower() == 'visual_inception':
        return 'image'
    return 'joint'


def _score_vector(ranking, ids):
    """按全库原始顺序保存分数；禁止只复核待替换记忆。"""
    lookup = {r['Id']: r for r in ranking}
    if len(lookup) != len(ranking) or set(lookup) != set(ids):
        raise ValueError('检索候选缺失、重复或与全库不一致')
    scores = np.asarray([lookup[i]['score'] for i in ids], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('候选分数必须全部有限')
    expected = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
    if [r['Id'] for r in ranking] != [ids[i] for i in expected]:
        raise ValueError('排名没有按分数降序和原库顺序确定性打破平局')
    return scores


def _save_scores(output, values, ids, queries, rows, source_id):
    path = Path(output) / 'scores.npz'
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, scores=np.asarray(values, dtype=np.float64),
                            memory_ids=np.asarray(ids, dtype=str),
                            query_ids=np.asarray([q['task_id'] for q in queries], dtype=str),
                            condition_labels=np.asarray([r['condition']['label'] for r in rows], dtype=str),
                            banks=np.asarray([r['bank'] for r in rows], dtype=str),
                            source_memory_id=np.asarray(source_id, dtype=str))
    os.replace(temporary, path)
    return path


def evaluate(frozen, output, *, pool_path=SHARED_MEMORY, query_manifest,
             source_memory_id=None, device='cuda', encoder_factory=None,
             defense_factory=None, selected_conditions=None, encoder_config=None, query_mode=None,
             key_mode=None, replacement_frozens=None, bank_root=None, memory_vector_cache_dir=None):
    """只有此模块读取共享库；对照和修改图始终面对同一个完整共享池。"""
    output = Path(output)
    effective_key_mode = default_key_mode_for_method(frozen.get('method')) if key_mode is None else key_mode
    if effective_key_mode not in KEY_MODES:
        raise ValueError('记忆库检索键模式只能为 image 或 joint')
    effective_key_image_weight = 1.0 if effective_key_mode == 'image' else 0.5
    from src.visual.attack_completion import validate_attack_completion
    if frozen.get('status') != 'frozen':
        raise ValueError('只能评估完成且已经冻结的最终图片')
    if frozen.get('metadata', {}).get('test_fixture') is True:
        raise ValueError('离线测试夹具不能进入正式共享池评估')
    validate_attack_completion(frozen['metadata'], frozen['requested_steps'], dict(
        method=frozen['method'], source_image_sha256=frozen['source']['image_sha256'],
        target_image_sha256=frozen['target']['image_sha256']))
    replacement_inputs = None
    if replacement_frozens is not None:
        replacement_inputs = []
        for item in replacement_frozens:
            if item.get('status') != 'frozen' or item['method'] != frozen['method']:
                raise ValueError('统一修改库只接受同一方法的全部冻结成图')
            validate_attack_completion(item['metadata'], item['requested_steps'], dict(
                method=item['method'], source_image_sha256=item['source']['image_sha256'],
                target_image_sha256=item['target']['image_sha256']))
            if sha256_file(item['image']) != item['image_sha256']:
                raise ValueError('统一修改库中的冻结图片摘要改变')
            replacement_inputs.append(dict(source=item['source'], image=item['image'],
                                           source_memory_id=item.get('source_memory_id')))
    # 正式图未就绪时，在读取共享库前退出。
    queries = load_formal_queries(query_manifest, frozen['target'], query_mode=query_mode)
    query_options = query_retriever_options(queries)
    if sha256_file(frozen['image']) != frozen['image_sha256']:
        raise ValueError('冻结成图摘要改变')
    entries = load_shared_pool(pool_path)
    banks, source_id = evaluation_banks(entries, pool_path, frozen['source'],
                                        frozen['image'], source_memory_id=source_memory_id,
                                        replacements=replacement_inputs)
    specs = list(selected_conditions if selected_conditions is not None else conditions())
    expected_labels = {s['label']: s for s in conditions()}
    if not specs or len({s['label'] for s in specs}) != len(specs) or any(expected_labels.get(s['label']) != s for s in specs):
        raise ValueError('条件须来自固定防御面板，且不可重复')
    plan = dict(schema_version=1, scope='shared', pool_path=str(Path(pool_path).resolve()),
                pool_sha256=sha256_file(pool_path), pool_count=len(entries),
                source_memory_id=source_id, frozen_image_sha256=frozen['image_sha256'],
                method=frozen['method'], queries=queries, conditions=specs,
                simultaneous_replacements=[dict(source_id=i['source']['id'], image_sha256=sha256_file(i['image']),
                                               source_memory_id=i.get('source_memory_id')) for i in replacement_inputs or []],
                memory_vector_cache_policy='content_model_decoder_text_fusion_bound_v1' if memory_vector_cache_dir else 'disabled',
                encoder_config=str(Path(encoder_config).resolve()) if encoder_config else None,
                encoder_config_sha256=sha256_file(encoder_config) if encoder_config else None,
                key_mode=effective_key_mode,
                key_image_weight=effective_key_image_weight, query_image_weight=query_options.get('query_image_weight', 0.5),
                query_mode=query_options.get('query_mode', 'joint'),
                query_manifest=str(Path(query_manifest).resolve()), query_manifest_sha256=sha256_file(query_manifest),
                index_text=detect_index_text_policy(entries),
                query_image_defended=False, all_memory_images_equally_defended=True,
                shopping_allowed=False,
                record_kinds={kind: sum(e['RecordKind'] == kind for e in entries)
                              for kind in ('purchase_memory', 'simulated_gallery')})
    freeze_json(output / 'evaluation_plan.json', plan)
    custom_encoder = encoder_factory is not None
    if encoder_factory is None:
        encoder_factory = lambda: make_eval_encoder(device, encoder_config)
    encoder = encoder_factory()
    bank_root = Path(bank_root) if bank_root is not None else output
    defender = defense_factory(bank_root / 'defended', device) if defense_factory else Defender(bank_root / 'defended', device)
    rows, score_rows = [], []
    ids = [e['Id'] for e in entries]
    try:
        for spec in specs:
            for name, original_bank in banks.items():
                bank = copy.deepcopy(original_bank)
                if spec['kind'] != 'none':
                    for entry in bank:
                        original = resolve_image(entry, pool_path)
                        image = defender(original, spec)
                        entry['RetrievalImage'] = dict(ASIN=entry['FinalASIN'], LocalPath=str(image.resolve()), SHA256=sha256_file(image))
                bank_file = bank_root / 'banks' / spec['label'] / (name + '.json')
                freeze_json(bank_file, bank)
                ranker = VisualMemoryRetriever(encoder, bank_file, key_mode=effective_key_mode, cache_dir=None,
                                               entry_cache_dir=memory_vector_cache_dir,
                                               image_base_path=pool_path, **query_options)
                observations, candidate_scores = [], []
                for query in queries:
                    verify_query_input(query)
                    ranking = ranker.rank(bank, query['instruction'], query.get('query_image'))
                    if len(ranking) != len(entries):
                        raise ValueError('检索没有覆盖完整共享池')
                    chosen = next(r for r in ranking if r['Id'] == source_id)
                    candidate_scores.append(_score_vector(ranking, ids))
                    observations.append(dict(task_id=query['task_id'], rank=chosen['rank'], score=chosen['score']))
                row = dict(condition=spec, bank=name, bank_path=str(bank_file.resolve()),
                           bank_sha256=sha256_file(bank_file), queries=observations,
                           top3=sum(q['rank'] <= 3 for q in observations),
                           top5=sum(q['rank'] <= 5 for q in observations),
                           query_count=len(queries), score_archive_index=len(rows),
                           retrieval_protocol=copy.deepcopy(ranker.snapshot))
                rows.append(row)
                score_rows.append(candidate_scores)
                save_json(output / 'evaluation.partial.json', rows)
                print(f"{spec['label']} {name}: 前三 {row['top3']}/{len(queries)}；前五 {row['top5']}/{len(queries)}", flush=True)
        _save_scores(output, score_rows, ids, queries, rows, source_id)
        freeze_json(output / 'evaluation.json', rows)
    finally:
        encoder.close()
    # 使用新建编码器重新计算每个条件、每个查询的名次及分数。
    if custom_encoder:
        verification = verify(output, encoder_factory=encoder_factory)
    else:
        from src.paths import PROJECT
        command = [sys.executable, '-B', '-X', 'utf8', '-m', 'src.experiments.shared_evaluation',
                   'verify', '--output', str(output.resolve()), '--device', device]
        with (output / 'independent_verification.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode:
            save_json(output / 'VERIFICATION_FAILED.json', dict(returncode=result.returncode,
                       log='independent_verification.log'))
            raise RuntimeError('独立进程复算失败，见本轮复算日志')
        verification = read_json(output / 'independent_verification.json')
    return dict(rows=len(rows), rankings=len(rows) * len(queries), verification=verification)


class _EqualWeightGME:
    """显式分别编码图文后等权融合，避免把原生融合误称各半。"""
    supports_image_weight = True

    def __init__(self, base):
        self.base = base
        self.snapshot = {**base.snapshot, 'fusion': 'unit_equal_image_plus_full_text',
                         'manual_image_weight_supported': True}

    def encode(self, path):
        return self.base.encode(path)

    def encode_text(self, text):
        return self.base.encode_text(text)

    def encode_joint(self, text, path, image_weight=0.5):
        from src.visual.encoders.joint import normalized, validated_image_weight
        weight = validated_image_weight(image_weight)
        return normalized(weight * self.encode(path) + (1 - weight) * self.encode_text(text))

    def close(self):
        self.base.close()


def make_eval_encoder(device='cuda', config_path=None):
    """默认使用独立留出模型；仅显式配置可以选择同代理评估，且如实标记。"""
    # 同一进程先训练再评估时，也必须与独立复算进程使用相同数值模式。
    if str(device).startswith('cuda'):
        import torch
        from src.visual.hardware import configure_tf32
        configure_tf32(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    from src.paths import VISUAL_CONFIGS, PRIVATE, PROJECT
    from src.visual.encoders.joint import ChunkedClipEncoder
    private = PRIVATE / 'evaluator.json'
    settings_path = Path(config_path or (private if private.is_file() else VISUAL_CONFIGS / 'evaluator.example.json')).resolve()
    settings = read_json(settings_path)
    setting = settings.get('setting')
    if setting not in ('held_out_public_encoder', 'matched_public_encoder'):
        raise ValueError('评估编码器须显式标明独立留出或同代理评估设置')
    if setting == 'matched_public_encoder' and config_path is None:
        raise ValueError('同代理评估只能由本次实验显式配置启用')
    from src.visual._lucid.surrogates import resolve_hf_cached_weight
    from src.visual.encoders.clip import MatchedClipEncoder
    if settings.get('provider') == 'gme':
        if setting != 'held_out_public_encoder':
            raise ValueError('多模态留出编码器不得登记为训练代理')
        from src.visual.encoders.joint import GMEProcessEncoder
        from src.model_registry import resolve_model
        model = settings.get('model_path') or resolve_model('eval_gme')
        if not Path(model).is_absolute():
            model = str((settings_path.parent / model).resolve())
        encoder = _EqualWeightGME(GMEProcessEncoder(
            python=settings.get('python'), model=model, device=device))
        encoder.snapshot['evaluation_protocol'] = dict(
            setting=setting, matches_training_surrogate=False, black_box_transfer_claimed=True,
            tf32_enabled=False, fusion='explicit_equal_unit_image_and_full_text')
        return encoder
    spec = settings['surrogate']
    training_specs = read_json(VISUAL_CONFIGS / 'optimization.json')['surrogates']
    # 两种方法全部训练代理都参与隔离校验，而非只检查原始像素方法。
    training_specs += [dict(name='ViT-L-14', pretrained='openai'),
                       dict(name='ViT-SO400M-14-SigLIP-384', pretrained='webli'),
                       dict(name='ViT-H-14', pretrained='laion2b_s32b_b79k')]
    matching = [i for i, m in enumerate(training_specs)
                if (m['name'], m['pretrained']) == (spec['name'], spec['pretrained'])]
    if matching and setting == 'held_out_public_encoder':
        raise ValueError('正式留出检索器与训练代理相同，不能作为黑盒迁移评估')
    if not matching and setting == 'matched_public_encoder':
        raise ValueError('同代理评估配置与实际训练代理身份不符')
    if spec.get('weight_path'):
        weight = Path(spec['weight_path'])
        weight = weight if weight.is_absolute() else settings_path.parent / weight
    else:
        # 独立准备的模型快照优先；不要求把训练代理权重也装载进内存。
        from src.model_registry import resolve_model
        try:
            directory = Path(resolve_model('eval_g14')) if spec['name'] == 'ViT-g-14' else None
        except FileNotFoundError:
            directory = None
        weight = next((directory / filename for filename in ('open_clip_model.safetensors',
                      'open_clip_pytorch_model.safetensors', 'open_clip_pytorch_model.bin')
                      if directory is not None and (directory / filename).is_file()), None)
        if weight is None:
            weight = next((path for filename in ('open_clip_model.safetensors', 'open_clip_pytorch_model.safetensors', 'open_clip_pytorch_model.bin')
                           if (path := resolve_hf_cached_weight(spec['hf_repo'], filename=filename)) is not None), None)
    if weight is None or not weight.is_file():
        raise FileNotFoundError('缺少评估检索器的本地权重，请先显式准备')
    if setting == 'matched_public_encoder':
        from src.visual._lucid.config import resolve_optimization
        trained = resolve_optimization(VISUAL_CONFIGS / 'optimization.json')
        chosen_hash = sha256_file(weight)
        matching = [i for i in matching if i < len(trained.surrogates)
                    and sha256_file(trained.surrogates[i].weight_path) == chosen_hash]
        if not matching:
            raise ValueError('同代理评估的实际权重与训练代理不符')
    spec = dict(spec, weight_path=str(weight.resolve()))
    for key in ('tokenizer_path',):
        if spec.get(key) and not Path(spec[key]).is_absolute():
            spec[key] = str((settings_path.parent / spec[key]).resolve())
    if not spec.get('tokenizer_path') and (weight.parent / 'tokenizer_config.json').is_file():
        spec['tokenizer_path'] = str(weight.parent)
    encoder = ChunkedClipEncoder(MatchedClipEncoder(dict(surrogate=spec, surrogate_number=0, device=device), PROJECT))
    encoder.snapshot['evaluation_protocol'] = dict(setting=setting,
        matches_training_surrogate=bool(matching), matching_training_surrogate_indices=matching,
        black_box_transfer_claimed=setting == 'held_out_public_encoder',
        tf32_enabled=False, precision='float32')
    return encoder


def verify(output, *, encoder_factory=None, device='cuda', query_mode=None, query_manifest=None,
           key_mode=None, isolated_process=False):
    output = Path(output)
    plan = read_json(output / 'evaluation_plan.json')
    if plan.get('query_manifest') and sha256_file(plan['query_manifest']) != plan['query_manifest_sha256']:
        raise ValueError('正式询问清单在评估后改变')
    query_options = query_retriever_options(plan['queries'])
    if query_mode is not None and query_mode != query_options.get('query_mode', 'joint'):
        raise ValueError('复算指定的正式查询模式与冻结评估不一致')
    effective_key_mode = plan.get('key_mode') or ('image' if plan.get('key_image_weight') == 1.0 else 'joint')
    if effective_key_mode not in KEY_MODES:
        raise ValueError('冻结评估计划中的记忆键模式无效')
    if key_mode is not None and key_mode != effective_key_mode:
        raise ValueError('复算指定的记忆键模式与冻结评估不一致')
    if query_manifest is not None and (not plan.get('query_manifest_sha256')
            or sha256_file(query_manifest) != plan['query_manifest_sha256']):
        raise ValueError('复算指定的正式询问清单与冻结评估不一致')
    if sha256_file(plan['pool_path']) != plan['pool_sha256']:
        raise ValueError('基础共享池在评估后改变')
    if plan.get('encoder_config') and sha256_file(plan['encoder_config']) != plan['encoder_config_sha256']:
        raise ValueError('评估编码器配置在评估后改变')
    entries = load_shared_pool(plan['pool_path'])
    ids = [e['Id'] for e in entries]
    rows = read_json(output / 'evaluation.json')
    expected = {(s['label'], bank) for s in plan['conditions'] for bank in ('clean', 'modified')}
    actual = [(r['condition']['label'], r['bank']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('评估条件缺失或重复')
    score_path = output / 'scores.npz'
    with np.load(score_path, allow_pickle=False) as saved:
        scores = saved['scores'].copy()
        if (saved['memory_ids'].tolist() != ids
                or saved['query_ids'].tolist() != [q['task_id'] for q in plan['queries']]
                or saved['condition_labels'].tolist() != [r['condition']['label'] for r in rows]
                or saved['banks'].tolist() != [r['bank'] for r in rows]
                or saved['source_memory_id'].item() != plan['source_memory_id']):
            raise ValueError('全候选分数归档的标识或顺序不符')
    if scores.shape != (len(rows), len(plan['queries']), len(ids)) or not np.isfinite(scores).all():
        raise ValueError('全候选分数归档维度无效或包含非有限值')
    encoder = (encoder_factory or (lambda: make_eval_encoder(device, plan.get('encoder_config'))))()
    max_error, count, score_count = 0.0, 0, 0
    try:
        for row_index, row in enumerate(rows):
            if row.get('score_archive_index') != row_index:
                raise ValueError('分数归档行号改变')
            if row['condition'] not in plan['conditions']:
                raise ValueError('条件的具体参数改变')
            if sha256_file(row['bank_path']) != row['bank_sha256']:
                raise ValueError('评估副本改变')
            bank = read_json(row['bank_path'])
            if len(bank) != plan['pool_count'] or [e['Id'] for e in bank] != ids:
                raise ValueError('评估缩减了共享池')
            ranker = VisualMemoryRetriever(encoder, row['bank_path'], key_mode=effective_key_mode, cache_dir=None,
                                           image_base_path=plan['pool_path'], **query_options)
            if 'key_mode' in row['retrieval_protocol'] and row['retrieval_protocol']['key_mode'] != effective_key_mode:
                raise ValueError('独立复算的记忆键模式与评估结果不一致')
            if digest(encoder.snapshot) != row['retrieval_protocol']['model_sha256']:
                raise ValueError('独立复算编码器的权重或实现身份改变')
            if len(row['queries']) != len(plan['queries']):
                raise ValueError('正式请求数量改变')
            for query_index, (actual_query, query) in enumerate(zip(row['queries'], plan['queries'])):
                verify_query_input(query)
                ranking = ranker.rank(bank, query['instruction'], query.get('query_image'))
                fresh_scores = _score_vector(ranking, ids)
                archived_scores = scores[row_index, query_index]
                full_error = float(np.max(np.abs(fresh_scores - archived_scores)))
                if full_error > SCORE_TOLERANCE:
                    raise ValueError('全库候选分数独立复算误差超过容差')
                score_count += len(ids)
                max_error = max(max_error, full_error)
                hit = next(r for r in ranking if r['Id'] == plan['source_memory_id'])
                archived_order = np.argsort(-archived_scores, kind='stable').tolist()
                archived_rank = archived_order.index(ids.index(plan['source_memory_id'])) + 1
                if (actual_query['task_id'] != query['task_id'] or type(actual_query['rank']) is not int
                        or actual_query['rank'] != hit['rank'] or actual_query['rank'] != archived_rank
                        or not math.isfinite(actual_query['score'])):
                    raise ValueError('独立复算的请求、名次或分数不符')
                error = abs(actual_query['score'] - hit['score'])
                if (not math.isfinite(error) or error > SCORE_TOLERANCE
                        or abs(actual_query['score'] - archived_scores[ids.index(plan['source_memory_id'])]) > SCORE_TOLERANCE):
                    raise ValueError('独立复算分数误差超过容差')
                max_error = max(max_error, error)
                count += 1
            for k in (3, 5):
                if row[f'top{k}'] != sum(q['rank'] <= k for q in row['queries']):
                    raise ValueError('汇总与逐请求名次不符')
    finally:
        encoder.close()
    result = dict(status='passed', rows=len(rows), rankings_recomputed=count,
                  max_score_error=max_error, evaluated_file_sha256=sha256_file(output / 'evaluation.json'),
                  evaluation_plan_sha256=sha256_file(output / 'evaluation_plan.json'),
                  scores_sha256=sha256_file(score_path), candidate_scores_recomputed=score_count,
                  score_tolerance=SCORE_TOLERANCE, all_candidate_scores_checked=True,
                  isolated_process=isolated_process,
                  fresh_encoder=True, recomputed_embeddings_and_rankings=True)
    freeze_json(output / 'independent_verification.json', result)
    return result


def report(output):
    from src.experiments.paired_metrics import paired_rank_metrics
    output = Path(output)
    plan = read_json(output / 'evaluation_plan.json')
    rows = read_json(output / 'evaluation.json')
    verified = read_json(output / 'independent_verification.json')
    if (verified.get('status') != 'passed'
            or verified['evaluated_file_sha256'] != sha256_file(output / 'evaluation.json')
            or verified['evaluation_plan_sha256'] != sha256_file(output / 'evaluation_plan.json')
            or verified.get('scores_sha256') != sha256_file(output / 'scores.npz')
            or verified.get('all_candidate_scores_checked') is not True):
        raise ValueError('缺少绑定当前结果的独立复算证据')
    lookup = {(r['condition']['label'], r['bank']): r for r in rows}
    query_description = ('正式查询使用冻结的纯文字购买意向；未提供或生成查询图片。'
                         if plan.get('query_mode') == 'text' else
                         '正式查询使用冻结文字及单独生成、经核对无明确名称和品牌的图片，图文各半融合。所有图库图片接受相同防御，查询图片保持原样。')
    lines = ['# 共享长期记忆检索评估', '',
             f"方法：{plan['method']}。共享池 {plan['pool_count']} 条；真实购买记忆 {plan['record_kinds']['purchase_memory']} 条，模拟图库 {plan['record_kinds']['simulated_gallery']} 条。",
             '', query_description,
             '', '前三及前五表示检索命中，不表示购买成功。本轮没有购物或外部模型调用。', '',
             '| 条件 | 原图前三 | 修改图前三 | 原图前五 | 修改图前五 | 前三增量（百分点） | 平均名次改善 |', '|---|---:|---:|---:|---:|---:|---:|']
    for title, labels in table_groups():
        if not all((lbl, b) in lookup for lbl in labels for b in ('clean', 'modified')):
            continue
        values = []
        for k in (3, 5):
            for name in ('clean', 'modified'):
                qs = [q for lbl in labels for q in lookup[(lbl, name)]['queries']]
                values.append(f"{sum(q['rank'] <= k for q in qs)}/{len(qs)}")
        delta=paired_rank_metrics([lookup[(lbl,'clean')] for lbl in labels],[lookup[(lbl,'modified')] for lbl in labels])
        values += [f"{delta['top3_gain']*100:+.1f}",f"{delta['mean_rank_gain']:+.3f}"]
        lines.append('| ' + title + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '前三增量为修改图减原图；平均名次改善为原图名次减修改图名次，正值表示名次上升。同一请求的不同种子不是独立需求。']
    lines += ['', '## 全部条件与种子', '', '| 条件 | 图库 | 前三 | 前五 |', '|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['condition']['label']} | {row['bank']} | {row['top3']}/{row['query_count']} | {row['top5']}/{row['query_count']} |")
    lines += ['', f"独立重新编码并复算 {verified['rankings_recomputed']} 次排名及 {verified['candidate_scores_recomputed']} 个候选分数，最大分数误差 {verified['max_score_error']}，容差 {SCORE_TOLERANCE}。", '']
    path = output / 'REPORT.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('verify', 'report'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--key-mode', choices=KEY_MODES, default=None)
    args = parser.parse_args()
    if args.stage == 'verify':
        print(verify(args.output, device=args.device, key_mode=args.key_mode, isolated_process=True))
    else:
        print(report(args.output))
