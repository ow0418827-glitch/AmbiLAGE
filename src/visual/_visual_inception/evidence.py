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
ENHANCED_OBJECTIVE_KIND = 'visual_inception_retrieval_semantic_perceptual_hardneg_advquery_v3'
PAPER_STANDARD_OBJECTIVE_KIND = 'visual_inception_retrieval_semantic_perceptual_paper_standard_v4'


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
    from .retrieval_cache import CACHE_KIND, LEGACY_CACHE_KIND, validate_candidate_cache
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
    expected_query_count = int(bundle['audit']['query_count'])
    if 'declared_query_shortfall' not in bundle['audit']:
        _require(bundle['audit'].get('source_counts') == {
            'category_description': 50, 'model_paraphrase': 30, 'heldout_search_query': 20},
            '查询来源比例不是五十、三十、二十')
    queries, concept = bundle['queries'], bundle['target_concept_text']
    _require(len(queries) == len(set(queries)) == expected_query_count and metadata.get('future_queries') == queries,
             '冻结独立训练查询缺失或正文改变')
    _require(objective.get('semantic_targets') == [concept]
             and objective.get('semantic_targets_sha256') == content_sha256([concept])
             and objective.get('target_concept_text') == concept
             and objective.get('target_concept_text_sha256') == text_sha256(concept)
             and objective.get('future_queries_sha256') == content_sha256(queries),
             '独立目标概念与未来查询的正文或摘要不一致')
    _require(objective.get('query_roles_separated') is True
             and objective.get('semantic_reduction') == 'one_minus_cosine_single_independent_target_concept'
             and objective.get('retrieval_denominator') == (
                 'one_dynamic_adversarial_positive_plus_full_public_bank_and_top_hard_negatives'
                 if config.mechanism_profile == 'enhanced_v3' else
                 'one_dynamic_adversarial_positive_plus_static_three_source_public_negatives'
                 if config.mechanism_profile == 'paper_standard' else
                 'one_dynamic_adversarial_positive_plus_fixed_public_image_negatives')
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
             and sampling.get('pool_size') == expected_query_count and sampling.get('batch_size') == config.query_batch_size
             and sampling.get('seed') == config.seed
             and sampling.get('calibration_indices') == sample_query_indices(expected_query_count, config.query_batch_size, config.seed, 0)
             and sampling.get('step_indices') == [sample_query_indices(expected_query_count, config.query_batch_size, config.seed, step)
                  for step in range(1, config.steps + 1)], '校准或优化查询采样与冻结日程不一致')
    for row in metadata.get('history', []):
        _require(type(row.get('step')) is int and 1 <= row['step'] <= config.steps
                 and row.get('query_indices') == sampling['step_indices'][row['step'] - 1], '优化日志查询批次不一致')
    _check_encoding_audits(metadata.get('text_encoding_audits'), [concept], model_count=len(selection.specs), fixture=config.smoke_test)
    _check_encoding_audits(metadata.get('query_text_encoding_audits'), queries, model_count=len(selection.specs), fixture=config.smoke_test)
    records = metadata.get('negative_candidate_caches')
    _require(isinstance(records, list) and len(records) == len(selection.specs), '独立公开代理库的模型缓存数量不一致')
    cache_root = _project_path(config.candidate_cache_dir) if config.candidate_cache_dir else root / 'candidate_cache'
    if not config.smoke_test:
        _require(cache_root.resolve().is_relative_to(PROJECT.resolve()), '固定候选缓存不在当前项目内')
    candidate_identity = [dict(id=row['id'], image_sha256=row['image_sha256']) for row in bundle['candidates']]
    candidate_vectors = []
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
        _require(record.get('kind') in (CACHE_KIND, LEGACY_CACHE_KIND)
                 and identity.get('kind') == record.get('kind'), '固定候选缓存版本身份无效')
        _require(identity.get('model') == snapshot
                 and identity.get('candidates') == candidate_identity and identity.get('test_fixture') is config.smoke_test
                 and identity.get('runtime', {}).get('torch_version') == metadata['runtime']['torch_version'],
                 '固定候选缓存的模型、图片、文字或运行时身份不一致')
        candidate_vectors.append(validate_candidate_cache(record, identity,
            expected_bundle_sha256=bundle['bundle_sha256']))
    _require(metadata.get('semantic_target_audit_usage') == 'public_feature_identity_only', '公开特点审计被误标为训练查询')
    return bundle, candidate_vectors


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_enhanced_mechanisms(metadata, ready, config, bundle, candidate_vectors):
    from .pipeline import LEGACY_MECHANISM_FIELDS, mechanism_parameters
    from .negative_sampling import build_negative_policy, validate_negative_selection_audit
    from .enhanced_objective import (
        QUERY_AUGMENTATION_KIND, MARGIN_KIND, TOP_HARD_COMPETITOR_LIMIT,
        HARD_MARGIN_SHARPNESS,
    )
    raw_cfg = metadata.get('config', {})
    legacy_blend_record = (
        'bank_loss_blend' not in raw_cfg and 'hard_margin_loss_blend' not in raw_cfg
    )
    expected_params = (
        {name: getattr(config, name) for name in LEGACY_MECHANISM_FIELDS}
        if legacy_blend_record
        else mechanism_parameters(config)
    )
    execution = metadata.get('enhanced_mechanisms', {})
    body = {k: v for k, v in execution.items() if k != 'execution_sha256'}
    _require(execution.get('schema_version') == 2 and execution.get('kind') == ENHANCED_OBJECTIVE_KIND
             and execution.get('parameters') == expected_params
             and execution.get('execution_sha256') == content_sha256(body)
             == ready.get('enhanced_mechanisms_sha256'), '增强机制执行审计摘要或配置不一致')
    _require(ready.get('mechanism_profile') == 'enhanced_v3'
             and execution.get('author_exact_implementation_claimed') is False
             and metadata.get('paper_missing_mechanisms_implemented') is True
             and metadata.get('paper_full_attack_reproduced') is False,
             '增强机制不得冒称作者未公开细节的精确复现')
    objective = metadata['training_objective']
    _require(objective.get('mechanism_profile') == 'enhanced_v3'
             and objective.get('mechanism_parameters') == expected_params
             and objective.get('hard_negative_pool_size') == len(bundle['candidates'])
             and objective.get('hard_negative_limit') == config.hard_negative_count
             and objective.get('contrastive_margin') == config.contrastive_margin
             and objective.get('contrastive_margin_algorithm') == MARGIN_KIND
             and objective.get('query_adversarial_algorithm') == QUERY_AUGMENTATION_KIND
             and objective.get('query_adversarial_space') == 'frozen_text_query_subspace_on_unit_sphere'
             and objective.get('proxy_user_memory_is_independent_public_role') is True
             and objective.get('popularity_is_declared_proxy_not_real_sales') is True
             and objective.get('paper_missing_mechanisms_implemented') is True
             and objective.get('author_exact_mechanism_details_reproduced') is False,
             '增强训练目标未绑定实际机制参数及公开代理约定')
    policies = execution.get('negative_policies')
    expected_policies = [build_negative_policy(bundle['candidate_metadata'], vectors,
                         category=bundle['audit']['category']) for vectors in candidate_vectors]
    _require(policies == expected_policies, '困难负例角色策略与冻结候选和真实编码向量不一致')
    subspaces = execution.get('query_subspaces')
    pool_size = int(bundle['audit']['query_count'])
    _require(isinstance(subspaces, list) and len(subspaces) == len(policies), '查询文本流形子空间审计缺失')
    for sub, vectors in zip(subspaces, candidate_vectors):
        dim = int(vectors.shape[1])
        basis_sha = sub.get('basis_sha256')
        _require(isinstance(sub, dict) and type(sub.get('rank')) is int and 0 <= sub['rank'] <= dim
                 and sub.get('ambient_dim') == dim and sub.get('reference_count') == pool_size + 1
                 and sub.get('mean_cone_removed') is True
                 and isinstance(basis_sha, str) and len(basis_sha) == 64
                 and all(c in '0123456789abcdef' for c in basis_sha),
                 '查询文本流形子空间维度、参考数量或摘要无效')

    def check_audit(audit, model_index, query_count, *, require_per_query):
        _require(isinstance(audit, dict) and audit.get('schema_version') == 2
                 and audit.get('mechanism_profile') == 'enhanced_v3'
                 and audit.get('margin_algorithm') == MARGIN_KIND
                 and audit.get('contrastive_margin') == config.contrastive_margin
                 and audit.get('retrieval_temperature') == config.retrieval_temperature
                 and audit.get('top_hard_competitor_limit') == TOP_HARD_COMPETITOR_LIMIT
                 and audit.get('hard_margin_sharpness') == HARD_MARGIN_SHARPNESS
                 and audit.get('bank_loss_blend') == float(config.bank_loss_blend)
                 and audit.get('hard_margin_loss_blend') == float(config.hard_margin_loss_blend)
                 and audit.get('full_bank_denominator_preserved') is True
                 and audit.get('reduction') == 'query_mean_of_clean_adversarial_weighted_losses'
                 and audit.get('paper_exact_missing_mechanisms_reproduced') is False,
                 '逐次增强损失审计身份或间隔参数不符')
        for key in ('clean_retrieval_loss', 'adversarial_retrieval_loss', 'retrieval_loss', 'semantic_loss'):
            _require(_finite(audit.get(key)), '增强损失审计存在非有限值')
        expected = ((1 - config.query_adversarial_mix) * audit['clean_retrieval_loss']
                    + config.query_adversarial_mix * audit['adversarial_retrieval_loss'])
        _require(abs(audit['retrieval_loss'] - expected) <= 1e-5 * max(1.0, abs(expected)),
                 '增强检索损失与干净及对抗查询混合权重不符')
        for key in ('clean_negative_selection', 'adversarial_negative_selection'):
            validate_negative_selection_audit(audit.get(key), policies[model_index], config, query_count,
                                              require_per_query=require_per_query)
        q = audit.get('query_augmentation', {})
        enabled = (config.query_adversarial_radius > 0 and config.query_adversarial_steps > 0
                   and config.query_adversarial_step_size > 0 and config.query_adversarial_mix > 0)
        _require(q.get('algorithm') == QUERY_AUGMENTATION_KIND and q.get('enabled') is enabled
                 and q.get('radius') == config.query_adversarial_radius
                 and q.get('steps') == config.query_adversarial_steps
                 and q.get('step_size') == config.query_adversarial_step_size
                 and q.get('mix') == config.query_adversarial_mix,
                 '查询对抗增强实际参数与冻结配置不一致')
        expected_sub = subspaces[model_index]
        _require(q.get('space') == 'frozen_text_query_subspace_on_unit_sphere'
                 and q.get('optimization') == 'per_query_text_subspace_projected_spherical_gradient_ascent'
                 and q.get('projection') == 'unit_sphere_cap_intersection_with_original_l2_ball'
                 and q.get('initialization') == 'clean_query'
                 and q.get('image_features_detached_in_inner') is True
                 and q.get('negatives_detached') is True
                 and q.get('queries_detached_for_outer') is True
                 and q.get('subspace_rank') == expected_sub['rank']
                 and q.get('subspace_ambient_dim') == expected_sub['ambient_dim']
                 and q.get('subspace_reference_count') == expected_sub['reference_count']
                 and q.get('subspace_mean_cone_removed') is True
                 and q.get('subspace_basis_sha256') == expected_sub['basis_sha256']
                 and q.get('paper_exact_query_augmentation_reproduced') is False,
                 '查询增强算法空间、子空间绑定、梯度隔离或实现声明不一致')
        for key in ('clean_query_features_sha256', 'adversarial_query_features_sha256'):
            digest = q.get(key)
            _require(isinstance(digest, str) and len(digest) == 64
                     and all(c in '0123456789abcdef' for c in digest), '查询向量摘要无效')
        for key in ('clean_negative_selection', 'adversarial_negative_selection'):
            _require(q.get(key + '_sha256') == audit[key]['selection_sha256'],
                     '查询增强与外层目标的困难负例选择摘要不一致')
        tolerance = q.get('numerical_tolerance')
        _require(_finite(tolerance) and 0 < tolerance <= 1e-4
                 and _finite(q.get('max_l2_perturbation'))
                 and 0 <= q['max_l2_perturbation'] <= config.query_adversarial_radius + tolerance
                 and _finite(q.get('max_unit_norm_error')) and 0 <= q['max_unit_norm_error'] <= tolerance,
                 '对抗查询超出单位球面或扰动预算')
        clean, adversarial = q.get('clean_losses'), q.get('adversarial_losses')
        _require(isinstance(clean, list) and isinstance(adversarial, list)
                 and len(clean) == len(adversarial) == query_count
                 and all(_finite(v) for v in clean + adversarial)
                 and all(a >= c - tolerance for c, a in zip(clean, adversarial)),
                 '对抗查询未逐条保持最坏检索损失')
        _require(abs(sum(clean) / query_count - audit['clean_retrieval_loss']) <= tolerance * 4
                 and abs(sum(adversarial) / query_count - audit['adversarial_retrieval_loss']) <= tolerance * 4,
                 '对抗查询逐条损失与实际外层目标不一致')
        _require(_finite(q.get('minimum_loss_gain'))
                 and abs(q['minimum_loss_gain'] - min(a - c for c, a in zip(clean, adversarial))) <= tolerance,
                 '对抗查询最小损失增量与逐条审计不一致')
        return audit

    calibration = execution.get('calibration_audits')
    if len(policies) == 1 and execution.get('single_surrogate_calibration_skipped') is True:
        _require(calibration == [], '单代理直接优化不应包含集成方差校准审计')
    else:
        _require(isinstance(calibration, list) and len(calibration) == len(policies), '增强梯度方差校准模型数不符')
        for model_index, rows in enumerate(calibration):
            _require(isinstance(rows, list) and len(rows) == config.gradient_restarts, '增强梯度方差随机起点数不符')
            for row in rows:
                check_audit(row, model_index, config.query_batch_size, require_per_query=False)
    steps = execution.get('step_audits')
    _require(isinstance(steps, list) and len(steps) == config.steps, '增强机制实际优化步数不完整')
    for step, row in enumerate(steps, 1):
        _require(row.get('step') == step
                 and row.get('query_indices') == metadata['query_sampling']['step_indices'][step - 1]
                 and isinstance(row.get('model_audits'), list) and len(row['model_audits']) == len(policies),
                 '增强机制每步查询或模型审计不符')
        for model_index, audit in enumerate(row['model_audits']):
            check_audit(audit, model_index, config.query_batch_size, require_per_query=False)
    for row in metadata.get('history', []):
        _require('mechanism_audits' not in row, '优化历史日志不应重复拷贝逐步增强审计大对象')
    final = execution.get('final_model_audits')
    _require(isinstance(final, list) and len(final) == len(policies), '增强最终模型审计缺失')
    for model_index, audit in enumerate(final):
        check_audit(audit, model_index, pool_size, require_per_query=True)
        _require(audit['retrieval_loss'] == metadata['final_objective']['retrieval_losses'][model_index]
                 and audit['semantic_loss'] == metadata['final_objective']['semantic_losses'][model_index],
                 '增强最终目标值与实际审计不一致')


def _validate_paper_standard_mechanisms(metadata, ready, config, bundle, candidate_vectors):
    from .pipeline import mechanism_parameters
    from .paper_standard_objective import (
        PAPER_STANDARD_MARGIN_KIND,
        PAPER_STANDARD_QUERY_AUGMENTATION_KIND,
        build_static_negative_selection,
    )

    expected_params = mechanism_parameters(config)
    execution = metadata.get('paper_standard_mechanisms', {})
    body = {k: v for k, v in execution.items() if k != 'execution_sha256'}
    _require(
        execution.get('schema_version') == 1
        and execution.get('kind') == PAPER_STANDARD_OBJECTIVE_KIND
        and execution.get('parameters') == expected_params
        and execution.get('execution_sha256') == content_sha256(body) == ready.get('paper_standard_mechanisms_sha256'),
        '标准复现机制执行审计摘要或配置不一致',
    )
    _require(
        ready.get('mechanism_profile') == 'paper_standard'
        and execution.get('author_exact_implementation_claimed') is False
        and metadata.get('paper_missing_mechanisms_implemented') is True
        and metadata.get('paper_full_attack_reproduced') is False,
        '标准复现机制不得冒称作者未公开细节的精确复现',
    )

    expected_selections = [
        build_static_negative_selection(
            bundle['candidate_metadata'],
            vectors,
            category=bundle['audit']['category'],
            config=config,
        )
        for vectors in candidate_vectors
    ]
    selections = execution.get('static_negative_selections')
    _require(selections == expected_selections, '静态三来源随机负例选择与冻结候选及真实编码向量不一致')

    objective = metadata['training_objective']
    _require(
        objective.get('mechanism_profile') == 'paper_standard'
        and objective.get('mechanism_parameters') == expected_params
        and objective.get('static_negative_pool_sizes') == [row['selected_count'] for row in expected_selections]
        and objective.get('contrastive_margin') == config.contrastive_margin
        and objective.get('contrastive_margin_algorithm') == PAPER_STANDARD_MARGIN_KIND
        and objective.get('query_adversarial_algorithm') == PAPER_STANDARD_QUERY_AUGMENTATION_KIND
        and objective.get('query_adversarial_space') == 'ambient_embedding_unit_sphere_no_subspace'
        and objective.get('proxy_user_memory_is_independent_public_role') is True
        and objective.get('popularity_is_declared_proxy_not_real_sales') is True
        and objective.get('paper_missing_mechanisms_implemented') is True
        and objective.get('author_exact_mechanism_details_reproduced') is False,
        '标准复现训练目标未绑定实际机制参数及公开代理约定',
    )

    pool_size = int(bundle['audit']['query_count'])

    def check_audit(audit, model_index, query_count):
        _require(
            isinstance(audit, dict)
            and audit.get('schema_version') == 1
            and audit.get('mechanism_profile') == 'paper_standard'
            and audit.get('margin_algorithm') == PAPER_STANDARD_MARGIN_KIND
            and audit.get('contrastive_margin') == config.contrastive_margin
            and audit.get('retrieval_temperature') == config.retrieval_temperature
            and audit.get('static_negative_count') == expected_selections[model_index]['selected_count']
            and audit.get('static_negative_selection_sha256') == expected_selections[model_index]['selection_sha256']
            and audit.get('full_bank_denominator_preserved') is False
            and audit.get('reduction') == 'query_mean_of_clean_adversarial_weighted_losses'
            and audit.get('paper_exact_missing_mechanisms_reproduced') is False,
            '逐次标准复现损失审计身份或间隔参数不符',
        )
        for key in (
            'clean_infonce_loss',
            'clean_margin_loss',
            'clean_retrieval_loss',
            'adversarial_infonce_loss',
            'adversarial_margin_loss',
            'adversarial_retrieval_loss',
            'retrieval_loss',
            'semantic_loss',
        ):
            _require(_finite(audit.get(key)), '标准复现损失审计存在非有限值')

        clean_sum = audit['clean_infonce_loss'] + audit['clean_margin_loss']
        _require(
            abs(audit['clean_retrieval_loss'] - clean_sum) <= 1e-5 * max(1.0, abs(clean_sum)),
            '干净检索损失与 InfoNCE 加余弦间隔之和不符',
        )
        adv_sum = audit['adversarial_infonce_loss'] + audit['adversarial_margin_loss']
        _require(
            abs(audit['adversarial_retrieval_loss'] - adv_sum) <= 1e-5 * max(1.0, abs(adv_sum)),
            '对抗检索损失与 InfoNCE 加余弦间隔之和不符',
        )
        expected = (
            (1 - config.query_adversarial_mix) * audit['clean_retrieval_loss']
            + config.query_adversarial_mix * audit['adversarial_retrieval_loss']
        )
        _require(
            abs(audit['retrieval_loss'] - expected) <= 1e-5 * max(1.0, abs(expected)),
            '标准复现检索损失与干净及对抗查询混合权重不符',
        )

        q = audit.get('query_augmentation', {})
        enabled = (
            config.query_adversarial_steps > 0
            and config.query_adversarial_step_size > 0
            and config.query_adversarial_mix > 0
        )
        _require(
            q.get('algorithm') == PAPER_STANDARD_QUERY_AUGMENTATION_KIND
            and q.get('enabled') is enabled
            and q.get('steps') == config.query_adversarial_steps
            and q.get('step_size') == config.query_adversarial_step_size
            and q.get('mix') == config.query_adversarial_mix
            and q.get('space') == 'ambient_embedding_unit_sphere_no_subspace'
            and q.get('optimization') == 'single_step_l2_normalized_fgsm'
            and q.get('subspace_projection_used') is False
            and q.get('greedy_best_loss_used') is False
            and q.get('image_features_detached_in_inner') is True
            and q.get('negatives_detached') is True
            and q.get('queries_detached_for_outer') is True
            and q.get('paper_exact_query_augmentation_reproduced') is False,
            '标准复现查询增强参数、空间或梯度隔离声明不一致',
        )
        for key in ('clean_query_features_sha256', 'adversarial_query_features_sha256'):
            digest = q.get(key)
            _require(
                isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
                '标准复现查询向量摘要无效',
            )
        tolerance = q.get('numerical_tolerance')
        max_theoretical = q.get('max_theoretical_l2_perturbation')
        _require(
            _finite(tolerance)
            and 0 < tolerance <= 1e-4
            and _finite(max_theoretical)
            and _finite(q.get('max_l2_perturbation'))
            and 0 <= q['max_l2_perturbation'] <= max_theoretical + tolerance
            and _finite(q.get('max_unit_norm_error'))
            and 0 <= q['max_unit_norm_error'] <= tolerance,
            '标准复现对抗查询超出单位球面或扰动预算',
        )
        clean, adversarial = q.get('clean_losses'), q.get('adversarial_losses')
        _require(
            isinstance(clean, list)
            and isinstance(adversarial, list)
            and len(clean) == len(adversarial) == query_count
            and all(_finite(v) for v in clean + adversarial),
            '标准复现逐条查询损失列表无效',
        )
        _require(
            abs(sum(clean) / query_count - audit['clean_retrieval_loss']) <= tolerance * 4
            and abs(sum(adversarial) / query_count - audit['adversarial_retrieval_loss']) <= tolerance * 4,
            '标准复现逐条查询损失均值与外层检索损失不一致',
        )
        return audit

    calibration = execution.get('calibration_audits')
    if len(selections) == 1 and execution.get('single_surrogate_calibration_skipped') is True:
        _require(calibration == [], '单代理直接优化不应包含集成方差校准审计')
    else:
        _require(isinstance(calibration, list) and len(calibration) == len(selections), '标准复现梯度方差校准模型数不符')
        for model_index, rows in enumerate(calibration):
            _require(isinstance(rows, list) and len(rows) == config.gradient_restarts, '标准复现梯度方差随机起点数不符')
            for row in rows:
                check_audit(row, model_index, config.query_batch_size)
    steps = execution.get('step_audits')
    _require(isinstance(steps, list) and len(steps) == config.steps, '标准复现机制实际优化步数不完整')
    for step, row in enumerate(steps, 1):
        _require(
            row.get('step') == step
            and row.get('query_indices') == metadata['query_sampling']['step_indices'][step - 1]
            and isinstance(row.get('model_audits'), list)
            and len(row['model_audits']) == len(selections),
            '标准复现机制每步查询或模型审计不符',
        )
        for model_index, audit in enumerate(row['model_audits']):
            check_audit(audit, model_index, config.query_batch_size)
    for row in metadata.get('history', []):
        _require('mechanism_audits' not in row, '优化历史日志不应重复拷贝逐步审计大对象')
    final = execution.get('final_model_audits')
    _require(isinstance(final, list) and len(final) == len(selections), '标准复现最终模型审计缺失')
    for model_index, audit in enumerate(final):
        check_audit(audit, model_index, pool_size)
        _require(
            audit['retrieval_loss'] == metadata['final_objective']['retrieval_losses'][model_index]
            and audit['semantic_loss'] == metadata['final_objective']['semantic_losses'][model_index],
            '标准复现最终目标值与实际审计不一致',
        )


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
    _require(kind in (OBJECTIVE_KIND, RETRIEVAL_OBJECTIVE_KIND, ENHANCED_OBJECTIVE_KIND, PAPER_STANDARD_OBJECTIVE_KIND)
             and kind == ready.get('training_objective_kind'), '训练目标身份无效')
    retrieval_enabled = kind in (RETRIEVAL_OBJECTIVE_KIND, ENHANCED_OBJECTIVE_KIND, PAPER_STANDARD_OBJECTIVE_KIND)
    raw_config = metadata.get('config')
    _require(isinstance(raw_config, dict), '冻结配置必须为对象')
    interpretation = dict(raw_config)
    if 'objective_mode' not in interpretation and not retrieval_enabled:
        interpretation['objective_mode'] = 'semantic_perceptual'
    if (
        kind == ENHANCED_OBJECTIVE_KIND
        and 'bank_loss_blend' not in interpretation
        and 'hard_margin_loss_blend' not in interpretation
    ):
        interpretation['bank_loss_blend'] = 0.3
        interpretation['hard_margin_loss_blend'] = 0.7
    config = VisualInceptionConfig.from_dict(interpretation)
    if retrieval_enabled:
        expected_profile = (
            'enhanced_v3'
            if kind == ENHANCED_OBJECTIVE_KIND
            else 'paper_standard'
            if kind == PAPER_STANDARD_OBJECTIVE_KIND
            else 'legacy_v2'
        )
        _require(config.mechanism_profile == expected_profile,
                 '增强机制版本与冻结训练目标不一致，旧图不能重签')
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
    expected_query_count = int(metadata.get('training_bundle_audit', {}).get('query_count', 100)) if retrieval_enabled else 3
    _require(objective.get('target_image_used_for_optimization') is False
             and objective.get('query_count') == expected_query_count,
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
        bundle, candidate_vectors = _validate_retrieval(root, metadata, ready, config, audit, selection)
        if kind == ENHANCED_OBJECTIVE_KIND:
            _validate_enhanced_mechanisms(metadata, ready, config, bundle, candidate_vectors)
        elif kind == PAPER_STANDARD_OBJECTIVE_KIND:
            _validate_paper_standard_mechanisms(metadata, ready, config, bundle, candidate_vectors)
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
    weights, variances = attack.get('ensemble_weights', []), attack.get('gradient_variances')
    if len(selection.specs) == 1 and attack.get('attack_mode') == 'single_surrogate_direct':
        _require(
            variances is None
            and weights == [1.0]
            and attack.get('gradient_restarts_executed') == 0
            and attack.get('ensemble_weight_formula') == 'unit_single_surrogate_weight',
            '单代理直接优化权重或方差跳过证据无效',
        )
    else:
        from .objective import inverse_variance_weights, gradient_variance_floor
        variances = variances if variances is not None else []
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
