"""Public-only projected pixel optimization with sequential surrogate graphs."""
from dataclasses import replace
import gc
import json
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.paths import ProjectPath as Path, PROJECT
from src.visual.catalog_training_guard import catalog_training_guard
from src.visual.hardware import apply_hardware_profile
from src.visual.io import load_image, pil_to_tensor, save_tensor_png, save_json, sha256_file, decoded_image_audit
from src.visual.training_surrogates import (
    selection_from_config, apply_surrogate_overrides, validate_loaded_surrogates,
)
from .config import VisualInceptionConfig
from .targets import prepare_semantic_targets, content_sha256, text_sha256
from .objective import (semantic_loss, inverse_variance_weights, estimate_gradient_variances,
    gradient_variance_floor, retrieval_semantic_losses, sample_query_indices,
    single_surrogate_loss_and_gradient)
from .perceptual import load_perceptual, resolve_lpips_weights

OBJECTIVE_KIND = 'visual_inception_semantic_perceptual_no_retrieval_v1'
RETRIEVAL_OBJECTIVE_KIND = 'visual_inception_retrieval_semantic_perceptual_v2'
ENHANCED_OBJECTIVE_KIND = 'visual_inception_retrieval_semantic_perceptual_hardneg_advquery_v3'
PAPER_STANDARD_OBJECTIVE_KIND = 'visual_inception_retrieval_semantic_perceptual_paper_standard_v4'
QUERY_SAMPLING_KIND = 'fixed_pool_per_step_without_replacement_cross_step_reuse_v1'
LEGACY_MECHANISM_FIELDS = ('mechanism_profile', 'contrastive_margin', 'hard_negative_count',
    'hard_negative_same_category_count', 'hard_negative_proxy_memory_count', 'hard_negative_popularity_count',
    'query_adversarial_radius', 'query_adversarial_steps', 'query_adversarial_step_size', 'query_adversarial_mix')
MECHANISM_FIELDS = LEGACY_MECHANISM_FIELDS + ('bank_loss_blend', 'hard_margin_loss_blend')
PAPER_STANDARD_MECHANISM_FIELDS = (
    'mechanism_profile',
    'contrastive_margin',
    'hard_negative_same_category_count',
    'hard_negative_proxy_memory_count',
    'hard_negative_popularity_count',
    'query_adversarial_steps',
    'query_adversarial_step_size',
    'query_adversarial_mix',
)


def mechanism_parameters(config):
    fields = (
        PAPER_STANDARD_MECHANISM_FIELDS
        if config.mechanism_profile == 'paper_standard'
        else MECHANISM_FIELDS
    )
    return {name: getattr(config, name) for name in fields}


def _prepare_text_features(models, texts, *, fixture):
    vectors, audits = [], []
    for surrogate in models:
        rows, records = [], []
        from src.visual.encoders.joint import ChunkedClipEncoder
        encoder = ChunkedClipEncoder(SimpleNamespace(surrogate=surrogate,
            device=surrogate.device, snapshot=surrogate.snapshot()))
        for text in texts:
            row = encoder.encode_text(text)
            rows.append(torch.as_tensor(row, device=surrogate.device, dtype=torch.float32).reshape(1, -1))
            records.append(dict(encoder.last_encoding_audit, test_fixture=fixture))
        vectors.append(torch.cat(rows, dim=0).detach())
        audits.append(records)
    return vectors, audits


def _proxy_components(model, image, target, *, retrieval_state=None, model_index=0,
                      query_indices=None, temperature=.07, config=None, return_audit=False,
                      include_per_query_audit=True):
    if retrieval_state is None:
        values = (semantic_loss(model, image, target), image.new_zeros(()))
        return (*values, None) if return_audit else values
    if query_indices is None:
        raise ValueError('检索优化必须显式指定冻结查询池中的批次索引')
    feature = model.encode_image(image).global_embedding
    queries = retrieval_state['query_features'][model_index][query_indices]
    negatives = retrieval_state['negative_features'][model_index]
    if config is not None and config.mechanism_profile == 'enhanced_v3':
        from .enhanced_objective import enhanced_retrieval_semantic_losses
        subspace = (retrieval_state['query_subspaces'][model_index]
                    if 'query_subspaces' in retrieval_state else None)
        semantic, retrieval, audit = enhanced_retrieval_semantic_losses(feature, target, queries,
            negatives, negative_policy=retrieval_state['negative_policies'][model_index], config=config,
            query_subspace=subspace, include_per_query_audit=include_per_query_audit)
        return (semantic, retrieval, audit) if return_audit else (semantic, retrieval)
    if config is not None and config.mechanism_profile == 'paper_standard':
        from .paper_standard_objective import paper_standard_retrieval_semantic_losses
        semantic, retrieval, audit = paper_standard_retrieval_semantic_losses(
            feature,
            target,
            queries,
            negatives,
            static_selection=retrieval_state['static_negative_selections'][model_index],
            config=config,
        )
        return (semantic, retrieval, audit) if return_audit else (semantic, retrieval)
    values = retrieval_semantic_losses(feature, target, queries, negatives, temperature)
    return (*values, None) if return_audit else values


def _gradient(image, source, models, text_features, weights, perceptual, config, *,
              retrieval_state=None, query_indices=None):
    if config.objective_mode == 'retrieval_semantic_perceptual' and retrieval_state is None:
        raise ValueError('完整三项训练目标缺少独立查询与公开代理库向量')
    if len(models) == 1 and len(weights) == 1 and abs(float(weights[0]) - 1.0) <= 1e-12:
        return single_surrogate_loss_and_gradient(
            image,
            source,
            models[0],
            text_features[0],
            perceptual,
            config,
            proxy_fn=_proxy_components,
            retrieval_state=retrieval_state,
            query_indices=query_indices,
        )
    total_gradient = torch.zeros_like(image)
    similarities, retrieval_values, mechanism_audits = [], [], []
    for index, (model, targets, weight) in enumerate(zip(models, text_features, weights)):
        leaf = image.detach().requires_grad_(True)
        semantic, retrieval, mechanism_audit = _proxy_components(model, leaf, targets,
            retrieval_state=retrieval_state, model_index=index, query_indices=query_indices,
            temperature=config.retrieval_temperature, config=config, return_audit=True,
            include_per_query_audit=False)
        loss = config.semantic_weight * semantic + config.retrieval_weight * retrieval
        gradient, = torch.autograd.grad(loss, leaf)
        total_gradient.add_(gradient.detach(), alpha=weight)
        similarities.append(float(semantic.detach()))
        retrieval_values.append(float(retrieval.detach()))
        if mechanism_audit is not None:
            mechanism_audits.append(mechanism_audit)
        del leaf, loss, gradient, semantic, retrieval
    leaf = image.detach().requires_grad_(True)
    distance = perceptual(source, leaf)
    gradient, = torch.autograd.grad(distance, leaf)
    total_gradient.add_(gradient.detach(), alpha=config.perceptual_weight)
    record = dict(semantic_losses=similarities, retrieval_losses=retrieval_values,
        semantic_loss=sum(w * v for w, v in zip(weights, similarities)),
        retrieval_loss=sum(w * v for w, v in zip(weights, retrieval_values)),
        perceptual_loss=float(distance.detach()))
    record['loss'] = (config.semantic_weight * record['semantic_loss']
        + config.retrieval_weight * record['retrieval_loss'] + config.perceptual_weight * record['perceptual_loss'])
    if mechanism_audits:
        record['mechanism_audits'] = mechanism_audits
    if not torch.isfinite(total_gradient).all():
        raise FloatingPointError('图片梯度含非有限值')
    return total_gradient, record


def _project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT / path).resolve()


def _load_bundle(config, source, target, public_semantic):
    if config.objective_mode == 'semantic_perceptual':
        return None
    if not config.training_manifest:
        raise ValueError('完整检索训练须提供冻结的独立查询与公开代理库清单 training_manifest')
    from .training_data import load_training_bundle
    bundle = load_training_bundle(_project_path(config.training_manifest), source=source,
        target=target, public_semantic=public_semantic, require_production=not config.smoke_test)
    source_digest = sha256_file(source)
    if any(row['image_sha256'] == source_digest for row in bundle['candidates']):
        raise ValueError('固定负例库不能同时保留干净源图片；动态正例槽位只能出现一次')
    expected_queries = int(bundle['audit']['query_count'])
    if len(bundle['queries']) != expected_queries or expected_queries < config.query_batch_size or len(bundle['candidates']) < 2:
        raise ValueError('完整训练查询数量与声明配额不符或固定公开候选不足')
    return bundle


def _run(source, target, root, config, *, device, target_text, definition, allow_tf32):
    source, target = Path(source).resolve(), Path(target).resolve()
    semantic = prepare_semantic_targets(source, target, target_text=target_text, definition=definition)
    bundle = _load_bundle(config, source, target, semantic)
    retrieval_enabled = bundle is not None
    enhanced_enabled = retrieval_enabled and config.mechanism_profile == 'enhanced_v3'
    paper_standard_enabled = retrieval_enabled and config.mechanism_profile == 'paper_standard'
    mechanism_enabled = enhanced_enabled or paper_standard_enabled
    source_pil, target_pil = load_image(source), load_image(target)
    requested_device = torch.device(device)
    if config.smoke_test:
        if requested_device.type != 'cpu' or config.steps > 3 or max(*source_pil.size, *target_pil.size) > 64:
            raise ValueError('微型夹具仅允许中央处理器、最多三步及图片长边不超过六十四')
    elif requested_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('指定显卡不可用，禁止静默回退')
    elif min(source_pil.size) < 32:
        raise ValueError('真实感知度量需要源图片宽高至少三十二像素')
    selection = selection_from_config(config)
    if not config.smoke_test:
        # Missing pretrained perception weights must stop before loading large
        # training encoders, and never create a random production result.
        resolve_lpips_weights(config)
    hardware = apply_hardware_profile(device=requested_device, allow_tf32=allow_tf32, seed=config.seed)
    if config.smoke_test:
        from src.visual._ambilage.smoke import TinySurrogate
        models = [TinySurrogate(i) for i in range(len(selection.specs))]
    else:
        from src.visual.training_objectives import load_training_surrogates
        models = load_training_surrogates(device=str(requested_device), selection=selection)
    snapshots = validate_loaded_surrogates(models, selection, allow_fixture=config.smoke_test)
    if any(torch.device(model.device) != requested_device for model in models):
        raise ValueError('实际训练代理设备与请求设备不一致')
    perceptual = load_perceptual(config, requested_device)
    target_texts = [bundle['target_concept_text']] if retrieval_enabled else semantic['semantic_texts']
    text_features, text_encoding = _prepare_text_features(models, target_texts, fixture=config.smoke_test)
    retrieval_state, query_encoding, cache_records, sampling = None, [], [], None
    pool_size = len(bundle['queries']) if retrieval_enabled else 0
    if retrieval_enabled:
        from .retrieval_cache import encode_candidate_cache
        query_features, query_encoding = _prepare_text_features(models, bundle['queries'], fixture=config.smoke_test)
        negatives, negative_policies, query_subspaces, static_negative_selections = [], [], [], []
        cache_root = _project_path(config.candidate_cache_dir) if config.candidate_cache_dir else root / 'candidate_cache'
        for model_idx, (model, concept) in enumerate(zip(models, text_features)):
            vectors, record = encode_candidate_cache(model, bundle['candidates'], cache_root,
                bundle_sha256=bundle['bundle_sha256'], fixture=config.smoke_test)
            if vectors.shape[1] != concept.shape[1]:
                raise ValueError('固定公开候选向量与当前模型文字维度不一致')
            negatives.append(vectors.detach()); cache_records.append(record)
            if enhanced_enabled:
                from .negative_sampling import build_negative_policy
                from .enhanced_objective import build_query_subspace
                negative_policies.append(build_negative_policy(bundle['candidate_metadata'], vectors,
                    category=bundle['audit']['category']))
                query_subspaces.append(build_query_subspace(query_features[model_idx], concept_features=concept))
            elif paper_standard_enabled:
                from .paper_standard_objective import build_static_negative_selection
                static_negative_selections.append(
                    build_static_negative_selection(
                        bundle['candidate_metadata'],
                        vectors,
                        category=bundle['audit']['category'],
                        config=config,
                    )
                )
        retrieval_state = dict(concept_features=text_features, query_features=query_features, negative_features=negatives)
        if enhanced_enabled:
            retrieval_state['negative_policies'] = negative_policies
            retrieval_state['query_subspaces'] = query_subspaces
        elif paper_standard_enabled:
            retrieval_state['static_negative_selections'] = static_negative_selections
        sampling = dict(algorithm=QUERY_SAMPLING_KIND, pool_size=pool_size, batch_size=config.query_batch_size,
            calibration_indices=sample_query_indices(pool_size, config.query_batch_size, config.seed, 0),
            step_indices=[sample_query_indices(pool_size, config.query_batch_size, config.seed, step)
                          for step in range(1, config.steps + 1)], seed=config.seed)
    source_tensor = pil_to_tensor(source_pil, requested_device)
    started = time.perf_counter()
    epsilon = config.epsilon_levels / 255.0
    step_size = config.step_size_levels / 255.0
    single_surrogate = (len(models) == 1)
    if single_surrogate:
        variances = None
        weights = [1.0]
        calibration_audits = []
    else:
        calibration_audits = [[] for _ in models]
        def enhanced_calibration_loss(model_index, model, leaf, target_features):
            sem, ret, audit = _proxy_components(model, leaf, target_features,
                retrieval_state=retrieval_state, model_index=model_index,
                query_indices=sampling['calibration_indices'], temperature=config.retrieval_temperature,
                config=config, return_audit=True, include_per_query_audit=False)
            calibration_audits[model_index].append(audit)
            return config.semantic_weight * sem + config.retrieval_weight * ret
        variances = estimate_gradient_variances(source_tensor, models, text_features,
            epsilon=epsilon, restarts=config.gradient_restarts, seed=config.seed,
            retrieval_state=retrieval_state, calibration_indices=sampling['calibration_indices'] if sampling else None,
            retrieval_weight=config.retrieval_weight, semantic_weight=config.semantic_weight,
            temperature=config.retrieval_temperature,
            loss_function=enhanced_calibration_loss if mechanism_enabled else None)
        weights = inverse_variance_weights(variances)
    lower, upper = (source_tensor - epsilon).clamp(0, 1), (source_tensor + epsilon).clamp(0, 1)
    # Clean-image initialization is an explicit reproduction assumption; random
    # restarts estimate ensemble uncertainty, not outcome-based seed selection.
    image = source_tensor.clone()
    history, mechanism_steps = [], []
    for step in range(1, config.steps + 1):
        indices = sampling['step_indices'][step - 1] if sampling else None
        gradient, record = _gradient(image, source_tensor, models, text_features, weights, perceptual, config,
            retrieval_state=retrieval_state, query_indices=indices)
        image = torch.maximum(lower, torch.minimum(upper, image - step_size * gradient.sign())).detach()
        if mechanism_enabled:
            mechanism_steps.append(dict(step=step, query_indices=indices,
                                        model_audits=record.pop('mechanism_audits')))
        if step == 1 or step == config.steps or step % 10 == 0:
            history.append(dict(step=step, **record, query_indices=indices,
                                linf_after_update=float((image - source_tensor).abs().max())))
            print(f'视觉记忆植入优化 {step}/{config.steps}，目标值 {record["loss"]:.6f}', flush=True)
    with torch.no_grad():
        final_semantic, final_retrieval, final_mechanism_audits = [], [], []
        for index, (model, targets) in enumerate(zip(models, text_features)):
            sem, ret, mechanism_audit = _proxy_components(model, image, targets, retrieval_state=retrieval_state,
                model_index=index, query_indices=list(range(pool_size)) if retrieval_enabled else None,
                temperature=config.retrieval_temperature, config=config, return_audit=True,
                include_per_query_audit=True)
            final_semantic.append(float(sem)); final_retrieval.append(float(ret))
            if mechanism_audit is not None:
                final_mechanism_audits.append(mechanism_audit)
        final_distance = float(perceptual(source_tensor, image))
    output = save_tensor_png(root / 'adversarial.png', image)
    decoded = decoded_image_audit(source_pil, output, maximum_levels=config.epsilon_levels)
    if not decoded['within_budget']:
        raise AssertionError('实际解码图片超过冻结像素预算')
    objective_kind = (
        PAPER_STANDARD_OBJECTIVE_KIND
        if paper_standard_enabled
        else ENHANCED_OBJECTIVE_KIND
        if enhanced_enabled
        else RETRIEVAL_OBJECTIVE_KIND
        if retrieval_enabled
        else OBJECTIVE_KIND
    )
    semantic_definition = dict(kind=objective_kind,
        semantic_weight=config.semantic_weight,
        perceptual_weight=config.perceptual_weight,
        semantic_reduction='mean_one_minus_cosine_across_three_public_text_templates',
        retrieval_loss_omitted=True, retrieval_loss_weight=0.0,
        memory_denominator_implemented=False, target_image_used_for_optimization=False,
        target_concept_text=semantic['canonical_text'], semantic_targets=semantic['semantic_texts'],
        semantic_targets_sha256=semantic['semantic_targets_sha256'],
        paper_query_distribution_reproduced=False, query_count=len(semantic['semantic_texts']),
        query_template_adaptation=semantic['adaptation'])
    if retrieval_enabled:
        shortfall_meta = bundle['audit'].get('declared_query_shortfall')
        semantic_definition.update(semantic_reduction='one_minus_cosine_single_independent_target_concept',
            retrieval_loss_omitted=False, retrieval_loss_weight=config.retrieval_weight,
            memory_denominator_implemented=True, target_concept_text=bundle['target_concept_text'],
            semantic_targets=target_texts, semantic_targets_sha256=content_sha256(target_texts), query_count=pool_size,
            training_manifest_sha256=sha256_file(_project_path(config.training_manifest)),
            training_bundle_sha256=bundle['bundle_sha256'], retrieval_temperature=config.retrieval_temperature,
            retrieval_denominator='one_dynamic_adversarial_positive_plus_fixed_public_image_negatives',
            retrieval_positive_count=1, negative_candidate_count=len(bundle['candidates']),
            proxy_memory_scope='independent_public_proxy_images', memory_key='image_only',
            clean_source_duplicate_excluded=True, query_roles_separated=True,
            target_concept_text_sha256=text_sha256(bundle['target_concept_text']),
            future_queries_sha256=content_sha256(bundle['queries']),
            paper_query_source_proportions_reproduced=(pool_size == 100 and shortfall_meta is None),
            paper_query_generator_reproduced=bundle['audit'].get('paper_query_generator_reproduced', False),
            query_template_adaptation='target_concept_and_future_queries_are_separate')
        if shortfall_meta is not None:
            semantic_definition['declared_query_shortfall'] = shortfall_meta
    if enhanced_enabled:
        from .enhanced_objective import MARGIN_KIND, QUERY_AUGMENTATION_KIND
        semantic_definition.update(
            mechanism_profile='enhanced_v3', mechanism_parameters=mechanism_parameters(config),
            retrieval_denominator='one_dynamic_adversarial_positive_plus_full_public_bank_and_top_hard_negatives',
            hard_negative_pool_size=len(bundle['candidates']), hard_negative_limit=config.hard_negative_count,
            contrastive_margin=config.contrastive_margin,
            contrastive_margin_algorithm=MARGIN_KIND,
            query_adversarial_space='frozen_text_query_subspace_on_unit_sphere',
            query_adversarial_algorithm=QUERY_AUGMENTATION_KIND,
            proxy_user_memory_is_independent_public_role=True,
            popularity_is_declared_proxy_not_real_sales=True,
            paper_missing_mechanisms_implemented=True, author_exact_mechanism_details_reproduced=False)
    elif paper_standard_enabled:
        from .paper_standard_objective import (
            PAPER_STANDARD_MARGIN_KIND,
            PAPER_STANDARD_QUERY_AUGMENTATION_KIND,
        )
        semantic_definition.update(
            mechanism_profile='paper_standard',
            mechanism_parameters=mechanism_parameters(config),
            retrieval_denominator='one_dynamic_adversarial_positive_plus_static_three_source_public_negatives',
            static_negative_pool_sizes=[row['selected_count'] for row in static_negative_selections],
            contrastive_margin=config.contrastive_margin,
            contrastive_margin_algorithm=PAPER_STANDARD_MARGIN_KIND,
            query_adversarial_space='ambient_embedding_unit_sphere_no_subspace',
            query_adversarial_algorithm=PAPER_STANDARD_QUERY_AUGMENTATION_KIND,
            proxy_user_memory_is_independent_public_role=True,
            popularity_is_declared_proxy_not_real_sales=True,
            paper_missing_mechanisms_implemented=True,
            author_exact_mechanism_details_reproduced=False,
        )
    if single_surrogate:
        attack = dict(attack_mode='single_surrogate_direct',
            steps_requested=config.steps, steps_completed=config.steps, selected_step=config.steps,
            stop_reason='requested_steps_completed', epsilon=epsilon, epsilon_levels=config.epsilon_levels,
            step_size=step_size, step_size_levels=config.step_size_levels, norm='linf',
            optimizer='projected_sign_gradient_descent', initialization='clean_source_zero_delta',
            gradient_execution='single_surrogate_unified_retrieval_semantic_perceptual_graph',
            gradient_restarts=config.gradient_restarts, gradient_restarts_executed=0,
            gradient_variances=None, variance_definition='not_applicable_single_surrogate',
            variance_floor=None, variance_floor_policy='not_applicable_single_surrogate',
            ensemble_weight_formula='unit_single_surrogate_weight',
            ensemble_weights=weights, weights_frozen_before_optimization=True)
    else:
        attack = dict(steps_requested=config.steps, steps_completed=config.steps, selected_step=config.steps,
            stop_reason='requested_steps_completed', epsilon=epsilon, epsilon_levels=config.epsilon_levels,
            step_size=step_size, step_size_levels=config.step_size_levels, norm='linf',
            optimizer='projected_sign_gradient_descent', initialization='clean_source_zero_delta',
            gradient_execution='sequential_surrogate_graphs_plus_separate_perceptual_graph',
            gradient_restarts=config.gradient_restarts, gradient_variances=variances,
            variance_definition=('mean_elementwise_population_variance_of_retrieval_plus_semantic_input_gradient_over_random_projected_starts'
                if retrieval_enabled else 'mean_elementwise_population_variance_of_semantic_input_gradient_over_random_projected_starts'),
            variance_floor=gradient_variance_floor(variances),
            variance_floor_policy='max(max_model_variance*1e-12,float64_tiny)',
            ensemble_weight_formula='normalize(max(variance,relative_floor)^-2)',
            ensemble_weights=weights, weights_frozen_before_optimization=True)
    metadata = dict(
        schema_version=4 if paper_standard_enabled else 3 if enhanced_enabled else 2 if retrieval_enabled else 1,
        method='visual_inception', status='success',
        test_fixture=config.smoke_test, production_backend=not config.smoke_test,
        config=config.to_dict(), config_sha256=content_sha256(config.to_dict()), seed=config.seed,
        source_image=str(source), source_image_sha256=semantic['source_image']['sha256'],
        target_image=str(target), target_image_sha256=semantic['target_image']['sha256'],
        output_image=str(output), output_image_sha256=sha256_file(output),
        training_objective=dict(semantic_definition, surrogate_selection=selection.audit()), semantic_target_audit=semantic,
        semantic_target_audit_usage='public_feature_identity_only' if retrieval_enabled else 'training_templates',
        surrogate_selection=selection.audit(), surrogates=snapshots,
        text_encoding_audits=text_encoding, perceptual=perceptual.snapshot(),
        attack=attack, history=history, decoded_image_audit=decoded,
        final_objective=dict(semantic_losses=final_semantic, retrieval_losses=final_retrieval, perceptual_loss=final_distance,
            retrieval_query_scope='entire_frozen_pool' if retrieval_enabled else 'unused',
            loss=sum(w*(config.semantic_weight*s+config.retrieval_weight*r)
                     for w,s,r in zip(weights,final_semantic,final_retrieval)) + config.perceptual_weight * final_distance),
        hardware=hardware, runtime=dict(torch_version=str(torch.__version__), device=str(requested_device)),
        optimization_seconds=time.perf_counter() - started,
        retrieval_loss_omitted=not retrieval_enabled, victim_memory_used=False, formal_queries_used=False,
        formal_ranking_used=False, primary_evaluator_used_during_attack=False,
        single_surrogate_profile_selected=selection.profile == 'clip_l14',
        paper_encoder_profile_selected=selection.profile == 'visual_inception_paper',
        paper_encoder_architectures_aligned=not config.smoke_test and selection.profile == 'visual_inception_paper',
        paper_encoder_checkpoint_status='public_checkpoint_choices_paper_exact_revisions_unspecified',
        paper_encoder_ensemble_reproduced=False, paper_full_attack_reproduced=False,
        assumptions=['linf_projection', 'clean_source_initialization',
            'single_surrogate_direct_eq2_gradient_without_ensemble_variance_weighting' if single_surrogate else 'random_start_semantic_gradient_population_variance_mean',
            'unit_single_surrogate_weight' if single_surrogate else 'fixed_uncertainty_weights',
            'dynamic_positive_in_denominator' if retrieval_enabled else 'three_public_templates_are_semantic_targets_without_retrieval_loss'])
    if retrieval_enabled:
        metadata.update(training_bundle_manifest=str(_project_path(config.training_manifest)),
            training_bundle_sha256=bundle['bundle_sha256'], training_bundle_audit=bundle['audit'],
            training_data_audit=bundle['audit'], future_queries=bundle['queries'],
            query_text_encoding_audits=query_encoding, query_sampling=sampling, negative_candidate_caches=cache_records)
    if enhanced_enabled:
        execution = dict(schema_version=2, kind=ENHANCED_OBJECTIVE_KIND,
            parameters=mechanism_parameters(config), negative_policies=negative_policies,
            query_subspaces=[{k: v for k, v in s.items() if k != 'basis'} for s in query_subspaces],
            calibration_audits=calibration_audits, step_audits=mechanism_steps,
            final_model_audits=final_mechanism_audits,
            author_exact_implementation_claimed=False,
            paper_underspecified_choices=[
                'cross_category_public_proxy_user_memory_role',
                'category_balanced_feature_centrality_popularity_proxy',
                'full_bank_preserved_with_role_quotas_and_top5_hard_margin_duel',
                'temperature_scaled_logit_margin_and_smooth_top_competitor_softplus',
                'frozen_text_subspace_spherical_l2_adversarial_queries'])
        if single_surrogate:
            execution['single_surrogate_calibration_skipped'] = True
        execution['execution_sha256'] = content_sha256(execution)
        metadata['enhanced_mechanisms'] = execution
        metadata['paper_missing_mechanisms_implemented'] = True
        metadata['assumptions'] += execution['paper_underspecified_choices']
    elif paper_standard_enabled:
        execution = dict(
            schema_version=1,
            kind=PAPER_STANDARD_OBJECTIVE_KIND,
            parameters=mechanism_parameters(config),
            static_negative_selections=static_negative_selections,
            calibration_audits=calibration_audits,
            step_audits=mechanism_steps,
            final_model_audits=final_mechanism_audits,
            author_exact_implementation_claimed=False,
            paper_underspecified_choices=[
                'cross_category_public_proxy_user_memory_role',
                'global_cosine_centrality_popularity_proxy_without_category_round_robin',
                'static_uniform_random_three_source_negative_subset_in_infonce_denominator',
                'standard_cosine_space_hinge_margin_0_3',
                'single_step_ambient_embedding_l2_fgsm_query_perturbation',
            ],
        )
        if single_surrogate:
            execution['single_surrogate_calibration_skipped'] = True
        execution['execution_sha256'] = content_sha256(execution)
        metadata['paper_standard_mechanisms'] = execution
        metadata['paper_missing_mechanisms_implemented'] = True
        metadata['assumptions'] += execution['paper_underspecified_choices']
    for name in ('metadata.json', 'attack_metrics.json'):
        save_json(root / name, metadata)
    ready = dict(schema_version=metadata['schema_version'], method='visual_inception', status='completed',
        test_fixture=config.smoke_test, production_backend=not config.smoke_test,
        output_image=str(output), output_image_sha256=metadata['output_image_sha256'],
        source_image=str(source), source_image_sha256=metadata['source_image_sha256'],
        target_image=str(target), target_image_sha256=metadata['target_image_sha256'],
        metadata_sha256=sha256_file(root / 'metadata.json'), attack_metrics_sha256=sha256_file(root / 'attack_metrics.json'),
        config_sha256=metadata['config_sha256'], steps_requested=config.steps,
        steps_completed=config.steps, epsilon=epsilon, epsilon_levels=config.epsilon_levels,
        training_objective_kind=semantic_definition['kind'], retrieval_loss_omitted=not retrieval_enabled,
        candidates=dict(final=dict(image=str(output), sha256=metadata['output_image_sha256'])))
    if retrieval_enabled:
        ready.update(training_bundle_sha256=bundle['bundle_sha256'],
            training_manifest_sha256=semantic_definition['training_manifest_sha256'])
    if enhanced_enabled:
        ready.update(mechanism_profile='enhanced_v3',
                     enhanced_mechanisms_sha256=metadata['enhanced_mechanisms']['execution_sha256'])
    elif paper_standard_enabled:
        ready.update(
            mechanism_profile='paper_standard',
            paper_standard_mechanisms_sha256=metadata['paper_standard_mechanisms']['execution_sha256'],
        )
    save_json(root / 'ready.json', ready)
    del models, perceptual, text_features, image, retrieval_state
    gc.collect()
    if requested_device.type == 'cuda':
        torch.cuda.empty_cache()
    return output, metadata


def generate_visual_inception(source, target, output_dir, *, config=None, steps=None,
        epsilon_levels=None, device='cuda', target_text=None, definition=None,
        allow_tf32=True, seed=42, smoke_test=False, surrogate_set=None, surrogate_config=None,
        training_manifest=None):
    root = Path(output_dir).resolve()
    roots = [root, Path(source).resolve().parent, Path(target).resolve().parent]
    if isinstance(config, (str, Path)):
        roots.append(_project_path(config).parent)
    with catalog_training_guard(allowed_output_roots=roots):
        raw = config
        if isinstance(config, (str, Path)):
            raw = json.loads(_project_path(config).read_text(encoding='utf-8'))
        cfg = raw if isinstance(raw, VisualInceptionConfig) else VisualInceptionConfig.from_dict(raw)
        changes = {}
        if steps is not None:
            changes['steps'] = steps
        if epsilon_levels is not None:
            changes['epsilon_levels'] = epsilon_levels
        if config is None or seed != 42:
            changes['seed'] = seed
        if smoke_test:
            changes['smoke_test'] = True
        if training_manifest is not None:
            changes['training_manifest'] = str(training_manifest)
        cfg = apply_surrogate_overrides(replace(cfg, **changes), surrogate_set, surrogate_config)
        cfg.validate()
    if cfg.objective_mode == 'retrieval_semantic_perceptual' and not cfg.training_manifest:
        raise ValueError('完整检索训练须提供冻结的独立查询与公开代理库清单 training_manifest')
    if cfg.training_manifest:
        roots.append(_project_path(cfg.training_manifest).parent)
    if cfg.candidate_cache_dir:
        cache_root = _project_path(cfg.candidate_cache_dir)
        if not cfg.smoke_test and not cache_root.is_relative_to(PROJECT):
            raise ValueError('候选向量缓存只能写入当前项目目录')
        roots.append(cache_root)
    with catalog_training_guard(allowed_output_roots=roots) as guard:
        if any((root / name).exists() for name in ('ready.json', 'metadata.json', 'attack_metrics.json', 'adversarial.png')):
            raise FileExistsError('结果目录已有冻结图片或证据，禁止覆盖；请使用新的输出目录')
        root.mkdir(parents=True, exist_ok=True)
        output, metadata = _run(source, target, root, cfg, device=device, target_text=target_text,
                                 definition=definition, allow_tf32=allow_tf32)
        # Store the successful guard audit after all generation input reads.
        metadata['training_access_audit'] = dict(checked_accesses=guard['checked_accesses'],
            blocked_accesses=list(guard['blocked_accesses']), private_access_allowed=False)
        for name in ('metadata.json', 'attack_metrics.json'):
            save_json(root / name, metadata)
        ready = json.loads((root / 'ready.json').read_text(encoding='utf-8'))
        ready['metadata_sha256'] = sha256_file(root / 'metadata.json')
        ready['attack_metrics_sha256'] = sha256_file(root / 'attack_metrics.json')
        save_json(root / 'ready.json', ready)
        return output, metadata
