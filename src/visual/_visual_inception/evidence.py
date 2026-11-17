"""Validate frozen generation evidence, including the decoded on-disk image."""
import json
import math

from src.paths import ProjectPath as Path, PROJECT
from src.visual.io import sha256_file, decoded_image_audit
from src.visual.training_surrogates import validate_selection_audit, validate_loaded_surrogates
from .config import VisualInceptionConfig
from .targets import content_sha256, text_sha256
from .perceptual import OFFICIAL_ALEXNET_SHA256_PREFIX, OFFICIAL_LPIPS_ALEX_V01_SHA256

OBJECTIVE_KIND = 'visual_inception_semantic_perceptual_no_retrieval_v1'
RETRIEVAL_OBJECTIVE_KIND = 'visual_inception_retrieval_semantic_perceptual_v2'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT / path).resolve()


def _check_encoding_audits(rows, texts, *, model_count, fixture):
    _require(isinstance(rows, list) and len(rows) == model_count, '文字编码模型审计数量不一致')
    for model_rows in rows:
        _require(isinstance(model_rows, list) and len(model_rows) == len(texts), '文字编码条数不一致')
        for row, text in zip(model_rows, texts):
            _require(row.get('text_sha256') == text_sha256(text) and row.get('truncated') is False
                     and row.get('encoded_tokens') == row.get('input_tokens')
                     and type(row.get('input_tokens')) is int and row['input_tokens'] > 0
                     and row.get('test_fixture') is fixture, '文字编码身份、覆盖范围或夹具标记不一致')


def _validate_retrieval(root, metadata, ready, config, public_semantic, selection):
    from .training_data import load_training_bundle
    from .objective import sample_query_indices
    from .retrieval_cache import validate_candidate_cache
    _require(bool(config.training_manifest), '完整检索证据缺少冻结训练清单')
    path = _project_path(config.training_manifest)
    _require(Path(metadata.get('training_bundle_manifest', '')).resolve() == path, '实际训练清单与配置不一致')
    bundle = load_training_bundle(path, source=metadata['source_image'], target=metadata['target_image'],
        public_semantic=public_semantic, require_production=not config.smoke_test)
    objective = metadata['training_objective']
    manifest_sha = sha256_file(path)
    _require(manifest_sha == objective.get('training_manifest_sha256') == ready.get('training_manifest_sha256'),
             '训练清单原字节与完成证据不一致')
    _require(bundle['bundle_sha256'] == metadata.get('training_bundle_sha256')
             == objective.get('training_bundle_sha256') == ready.get('training_bundle_sha256'),
             '独立训练数据包内容与完成证据不一致')
    _require(bundle['audit'] == metadata.get('training_bundle_audit') == metadata.get('training_data_audit'),
             '训练数据来源或审计记录已改变')
    _require(bundle['audit'].get('source_counts') == {
        'category_description': 50, 'model_paraphrase': 30, 'heldout_search_query': 20},
        '查询来源比例不是五十、三十、二十')
    queries, concept = bundle['queries'], bundle['target_concept_text']
    _require(len(queries) == len(set(queries)) == 100 and metadata.get('future_queries') == queries,
             '冻结的一百条独立训练查询缺失或正文改变')
    _require(objective.get('semantic_targets') == [concept]
             and objective.get('semantic_targets_sha256') == content_sha256([concept])
             and objective.get('target_concept_text') == concept
             and objective.get('target_concept_text_sha256') == text_sha256(concept)
             and objective.get('future_queries_sha256') == content_sha256(queries),
             '独立目标概念与未来查询的正文或摘要不一致')
    _require(objective.get('query_roles_separated') is True
             and objective.get('semantic_reduction') == 'one_minus_cosine_single_independent_target_concept'
             and objective.get('retrieval_denominator') == 'one_dynamic_adversarial_positive_plus_fixed_public_image_negatives'
             and objective.get('retrieval_positive_count') == 1
             and objective.get('negative_candidate_count') == len(bundle['candidates'])
             and objective.get('clean_source_duplicate_excluded') is True
             and objective.get('proxy_memory_scope') == 'independent_public_proxy_images'
             and objective.get('memory_key') == 'image_only'
             and objective.get('retrieval_temperature') == config.retrieval_temperature,
             '检索正负例范围、分母约定或温度无效')
    _require(objective.get('paper_query_generator_reproduced') is bundle['audit'].get('paper_query_generator_reproduced', False),
             '改写生成模型的论文对齐声明不一致')
    sampling = metadata.get('query_sampling', {})
    _require(sampling.get('algorithm') == 'fixed_pool_per_step_without_replacement_cross_step_reuse_v1'
             and sampling.get('pool_size') == 100 and sampling.get('batch_size') == config.query_batch_size
             and sampling.get('seed') == config.seed
             and sampling.get('calibration_indices') == sample_query_indices(100, config.query_batch_size, config.seed, 0)
             and sampling.get('step_indices') == [sample_query_indices(100, config.query_batch_size, config.seed, step)
                  for step in range(1, config.steps + 1)], '校准或优化查询采样与冻结日程不一致')
    for row in metadata.get('history', []):
        _require(type(row.get('step')) is int and 1 <= row['step'] <= config.steps
                 and row.get('query_indices') == sampling['step_indices'][row['step'] - 1], '优化日志查询批次不一致')
    _check_encoding_audits(metadata.get('text_encoding_audits'), [concept], model_count=len(selection.specs), fixture=config.smoke_test)
    _check_encoding_audits(metadata.get('query_text_encoding_audits'), queries, model_count=len(selection.specs), fixture=config.smoke_test)
    records = metadata.get('negative_candidate_caches')
    _require(isinstance(records, list) and len(records) == len(selection.specs), '独立公开代理库的模型缓存数量不一致')
    cache_root = _project_path(config.candidate_cache_dir) if config.candidate_cache_dir else root / 'candidate_cache'
    _require(cache_root.resolve().is_relative_to(PROJECT.resolve()), '固定候选缓存不在当前项目内')
    candidate_identity = [dict(id=row['id'], image_sha256=row['image_sha256']) for row in bundle['candidates']]
    for snapshot, record in zip(metadata['surrogates'], records):
        identity = record.get('identity', {})
        key = record.get('cache_key', '')
        _require(isinstance(key, str) and len(key) == 64 and all(c in '0123456789abcdef' for c in key),
                 '候选缓存摘要键无效')
        _require(Path(record.get('metadata_path', '')).resolve().is_relative_to(cache_root.resolve())
                 and Path(record.get('vector_path', '')).resolve().is_relative_to(cache_root.resolve())
                 and Path(record.get('metadata_path', '')).resolve() == (cache_root / key / 'cache.json').resolve()
                 and Path(record.get('vector_path', '')).resolve() == (cache_root / key / 'vectors.npy').resolve(),
                 '固定候选缓存路径与冻结配置不一致')
        _require(identity.get('model') == snapshot and identity.get('bundle_sha256') == bundle['bundle_sha256']
                 and identity.get('candidates') == candidate_identity and identity.get('test_fixture') is config.smoke_test
                 and identity.get('runtime', {}).get('torch_version') == metadata['runtime']['torch_version'],
                 '固定候选缓存的模型、图片、文字或运行时身份不一致')
        validate_candidate_cache(record, identity)
    _require(metadata.get('semantic_target_audit_usage') == 'public_feature_identity_only', '公开特点审计被误标为训练查询')


def validate_ready(root, ready=None, *, require_production=True):
    root = Path(root).resolve()
    on_disk = json.loads((root / 'ready.json').read_text(encoding='utf-8'))
    if ready is not None:
        _require(ready == on_disk, '传入完成凭证与落盘内容不一致')
    ready = on_disk
    for name in ('method', 'status'):
        _require(ready.get(name) == ('visual_inception' if name == 'method' else 'completed'), '完成凭证方法或状态无效')
    metadata_path, metrics_path = root / 'metadata.json', root / 'attack_metrics.json'
    _require(sha256_file(metadata_path) == ready.get('metadata_sha256'), '生成元数据摘要不一致')
    _require(sha256_file(metrics_path) == ready.get('attack_metrics_sha256'), '优化证据摘要不一致')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    _require(metadata == metrics, '两份生成元数据不一致')
    _require(metadata.get('method') == 'visual_inception' and metadata.get('status') == 'success', '生成方法或状态无效')
    objective = metadata.get('training_objective', {})
    kind = objective.get('kind')
    _require(kind in (OBJECTIVE_KIND, RETRIEVAL_OBJECTIVE_KIND)
             and kind == ready.get('training_objective_kind'), '训练目标身份无效')
    retrieval_enabled = kind == RETRIEVAL_OBJECTIVE_KIND
    raw_config = metadata.get('config')
    _require(isinstance(raw_config, dict), '冻结配置必须为对象')
    interpretation = dict(raw_config)
    if 'objective_mode' not in interpretation and not retrieval_enabled:
        interpretation['objective_mode'] = 'semantic_perceptual'
    config = VisualInceptionConfig.from_dict(interpretation)
    _require(config.objective_mode == ('retrieval_semantic_perceptual' if retrieval_enabled else 'semantic_perceptual'),
             '目标版本与冻结配置模式不一致')
    _require(content_sha256(raw_config) == metadata.get('config_sha256') == ready.get('config_sha256'), '冻结配置摘要不一致')
    fixture = config.smoke_test
    _require(metadata.get('test_fixture') is fixture and ready.get('test_fixture') is fixture, '测试夹具标记不一致')
    _require(metadata.get('production_backend') is (not fixture) and ready.get('production_backend') is (not fixture), '生产后端标记不一致')
    if require_production:
        _require(not fixture, '微型测试夹具不能作为生产成图')
    _require(objective.get('surrogate_selection') == metadata.get('surrogate_selection'), '目标内部代理选择记录不一致')
    _require(objective.get('semantic_weight') == config.semantic_weight
             and objective.get('perceptual_weight') == config.perceptual_weight
             and objective.get('retrieval_loss_weight') == (config.retrieval_weight if retrieval_enabled else 0.0)
             and objective.get('memory_denominator_implemented') is retrieval_enabled,
             '实际损失组成或权重与冻结配置不一致')
    for key in ('victim_memory_used', 'formal_queries_used', 'formal_ranking_used', 'primary_evaluator_used_during_attack'):
        _require(metadata.get(key) is False, '生成过程使用了禁止的评估输入：' + key)
    _require(metadata.get('retrieval_loss_omitted') is (not retrieval_enabled)
             and objective.get('retrieval_loss_omitted') is (not retrieval_enabled)
             and ready.get('retrieval_loss_omitted') is (not retrieval_enabled), '检索损失启用声明无效')
    _require(objective.get('target_image_used_for_optimization') is False
             and objective.get('query_count') == (100 if retrieval_enabled else 3),
             '目标图或语义模板使用范围无效')
    audit = metadata.get('semantic_target_audit', {})
    without_digest = {k: v for k, v in audit.items() if k != 'audit_sha256'}
    _require(content_sha256(without_digest) == audit.get('audit_sha256'), '公开语义输入审计摘要不一致')
    texts = audit.get('semantic_texts')
    _require(isinstance(texts, list) and len(texts) == 3 and all(isinstance(t, str) and t.strip() for t in texts), '语义模板缺失')
    _require(audit.get('semantic_targets_sha256') == content_sha256(texts), '公开特点模板摘要不一致')
    if not retrieval_enabled:
        _require(audit['semantic_targets_sha256'] == objective.get('semantic_targets_sha256')
                 and objective.get('semantic_targets') == texts
                 and objective.get('target_concept_text') == audit.get('canonical_text'), '语义目标正文或摘要不一致')
    _require([text_sha256(t) for t in texts] == audit.get('semantic_text_sha256')
             and text_sha256(audit.get('canonical_text', '')) == audit.get('canonical_text_sha256'), '语义正文摘要不一致')
    from src.visual.target_features import build_natural_buyer_intent
    canonical, rebuilt = build_natural_buyer_intent(audit.get('category'), audit.get('core_attributes', []))
    _require(canonical.casefold() == audit.get('canonical_text', '').casefold() and rebuilt == texts,
             '语义模板不对应冻结公开特点')
    _require(canonical == audit.get('template_canonical_text')
             and text_sha256(canonical) == audit.get('template_canonical_text_sha256'),
             '模板规范句或其摘要与公开特点不一致')
    for record in audit.get('frozen_files', []):
        _require(sha256_file(Path(record['path'])) == record['sha256'], '冻结公开特点文件已改变')
    selection = validate_selection_audit(metadata.get('surrogate_selection'), config.to_dict())
    if 'paper_encoder_profile_selected' in metadata:
        paper_selected = selection.profile == 'visual_inception_paper'
        _require(metadata['paper_encoder_profile_selected'] is paper_selected
                 and metadata.get('paper_encoder_architectures_aligned') is (paper_selected and not fixture),
                 '论文代理声明与冻结选择或微型夹具状态不一致')
    validate_loaded_surrogates(metadata.get('surrogates', []), selection, allow_fixture=fixture)
    if retrieval_enabled:
        _validate_retrieval(root, metadata, ready, config, audit, selection)
    perceptual = metadata.get('perceptual', {})
    if fixture:
        _require(perceptual.get('test_fixture') is True and perceptual.get('kind') == 'tiny_test_squared_pixel_distance_not_lpips', '感知测试夹具身份无效')
    else:
        _require(perceptual.get('kind') == 'lpips_alexnet_v0.1' and perceptual.get('pretrained') is True
                 and perceptual.get('test_fixture') is False, '缺少真实预训练感知度量证据')
        _require(str(perceptual.get('backbone_sha256', '')).startswith(OFFICIAL_ALEXNET_SHA256_PREFIX)
                 and perceptual.get('linear_sha256') == OFFICIAL_LPIPS_ALEX_V01_SHA256, '预训练感知权重身份无效')
        for role in ('backbone', 'linear'):
            _require(sha256_file(Path(perceptual[role + '_path'])) == perceptual[role + '_sha256'], '冻结感知权重已改变')
    attack = metadata.get('attack', {})
    for field in ('steps_requested', 'steps_completed', 'selected_step'):
        _require(type(attack.get(field)) is int and attack.get(field) == config.steps, '优化步骤不完整或未选择最终步')
    _require(attack.get('norm') == 'linf' and attack.get('optimizer') == 'projected_sign_gradient_descent'
             and attack.get('step_size_levels') == config.step_size_levels
             and attack.get('step_size') == config.step_size_levels / 255.0
             and attack.get('gradient_restarts') == config.gradient_restarts,
             '投影优化设置与冻结配置不一致')
    _require(ready.get('steps_completed') == ready.get('steps_requested') == config.steps
             and attack.get('stop_reason') == 'requested_steps_completed', '完成凭证优化步骤不完整')
    _require(attack.get('epsilon_levels') == ready.get('epsilon_levels') == config.epsilon_levels
             and attack.get('epsilon') == ready.get('epsilon') == config.epsilon_levels / 255.0, '扰动预算不一致')
    weights, variances = attack.get('ensemble_weights', []), attack.get('gradient_variances', [])
    from .objective import inverse_variance_weights, gradient_variance_floor
    expected_weights = inverse_variance_weights(variances)
    _require(len(weights) == len(selection.specs) == len(variances)
             and all(math.isfinite(w) and abs(w-e) <= 1e-10 for w,e in zip(weights,expected_weights)), '多模型权重证据无效')
    _require(attack.get('variance_floor') == gradient_variance_floor(variances), '多模型方差下限与冻结策略不一致')
    for role in ('source', 'target'):
        path = Path(metadata[role + '_image']).resolve()
        _require(sha256_file(path) == metadata[role + '_image_sha256'] == ready.get(role + '_image_sha256')
                 == audit.get(role + '_image', {}).get('sha256'), '冻结公开图对已改变：' + role)
    output = Path(metadata['output_image']).resolve()
    _require(output == (root / 'adversarial.png').resolve() and Path(ready['output_image']).resolve() == output, '输出图片路径越界或不一致')
    _require(sha256_file(output) == metadata.get('output_image_sha256') == ready.get('output_image_sha256'), '落盘图片摘要不一致')
    candidate = ready.get('candidates', {}).get('final', {})
    _require(Path(candidate.get('image', '')).resolve() == output
             and candidate.get('sha256') == metadata['output_image_sha256'], '最终候选图片证据不一致')
    decoded = decoded_image_audit(Path(metadata['source_image']), output, maximum_levels=config.epsilon_levels)
    _require(decoded['within_budget'] and decoded == metadata.get('decoded_image_audit'), '解码图片扰动预算或审计不一致')
    access = metadata.get('training_access_audit', {})
    _require(access.get('private_access_allowed') is False and access.get('blocked_accesses') == [], '训练文件访问审计无效')
    return metadata
