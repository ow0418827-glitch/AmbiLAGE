"""Public-only projected pixel optimization with sequential surrogate graphs."""
from dataclasses import replace
import gc
import json
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.paths import ProjectPath as Path
from src.visual.catalog_training_guard import catalog_training_guard
from src.visual.hardware import apply_hardware_profile
from src.visual.io import load_image, pil_to_tensor, save_tensor_png, save_json, sha256_file, decoded_image_audit
from src.visual.training_surrogates import (
    selection_from_config, apply_surrogate_overrides, validate_loaded_surrogates,
)
from .config import VisualInceptionConfig
from .targets import prepare_semantic_targets, content_sha256
from .objective import semantic_loss, inverse_variance_weights, estimate_gradient_variances, gradient_variance_floor
from .perceptual import load_perceptual, resolve_lpips_weights

OBJECTIVE_KIND = 'visual_inception_semantic_perceptual_no_retrieval_v1'


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


def _gradient(image, source, models, text_features, weights, perceptual, config):
    total_gradient = torch.zeros_like(image)
    similarities = []
    for model, targets, weight in zip(models, text_features, weights):
        leaf = image.detach().requires_grad_(True)
        loss = semantic_loss(model, leaf, targets)
        gradient, = torch.autograd.grad(loss, leaf)
        total_gradient.add_(gradient.detach(), alpha=config.semantic_weight * weight)
        similarities.append(float(loss.detach()))
        del leaf, loss, gradient
    leaf = image.detach().requires_grad_(True)
    distance = perceptual(source, leaf)
    gradient, = torch.autograd.grad(distance, leaf)
    total_gradient.add_(gradient.detach(), alpha=config.perceptual_weight)
    record = dict(semantic_losses=similarities,
        semantic_loss=sum(w * v for w, v in zip(weights, similarities)),
        perceptual_loss=float(distance.detach()))
    record['loss'] = config.semantic_weight * record['semantic_loss'] + config.perceptual_weight * record['perceptual_loss']
    if not torch.isfinite(total_gradient).all():
        raise FloatingPointError('图片梯度含非有限值')
    return total_gradient, record


def _run(source, target, root, config, *, device, target_text, definition, allow_tf32):
    source, target = Path(source).resolve(), Path(target).resolve()
    semantic = prepare_semantic_targets(source, target, target_text=target_text, definition=definition)
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
    text_features, text_encoding = _prepare_text_features(models, semantic['semantic_texts'], fixture=config.smoke_test)
    source_tensor = pil_to_tensor(source_pil, requested_device)
    started = time.perf_counter()
    epsilon = config.epsilon_levels / 255.0
    step_size = config.step_size_levels / 255.0
    variances = estimate_gradient_variances(source_tensor, models, text_features,
        epsilon=epsilon, restarts=config.gradient_restarts, seed=config.seed)
    weights = inverse_variance_weights(variances)
    lower, upper = (source_tensor - epsilon).clamp(0, 1), (source_tensor + epsilon).clamp(0, 1)
    # Clean-image initialization is an explicit reproduction assumption; random
    # restarts estimate ensemble uncertainty, not outcome-based seed selection.
    image = source_tensor.clone()
    history = []
    for step in range(1, config.steps + 1):
        gradient, record = _gradient(image, source_tensor, models, text_features, weights, perceptual, config)
        image = torch.maximum(lower, torch.minimum(upper, image - step_size * gradient.sign())).detach()
        if step == 1 or step == config.steps or step % 10 == 0:
            history.append(dict(step=step, **record,
                                linf_after_update=float((image - source_tensor).abs().max())))
            print(f'视觉记忆植入优化 {step}/{config.steps}，目标值 {record["loss"]:.6f}', flush=True)
    with torch.no_grad():
        final_semantic = [float(semantic_loss(model, image, targets)) for model, targets in zip(models, text_features)]
        final_distance = float(perceptual(source_tensor, image))
    output = save_tensor_png(root / 'adversarial.png', image)
    decoded = decoded_image_audit(source_pil, output, maximum_levels=config.epsilon_levels)
    if not decoded['within_budget']:
        raise AssertionError('实际解码图片超过冻结像素预算')
    semantic_definition = dict(kind=OBJECTIVE_KIND, semantic_weight=config.semantic_weight,
        perceptual_weight=config.perceptual_weight,
        semantic_reduction='mean_one_minus_cosine_across_three_public_text_templates',
        retrieval_loss_omitted=True, retrieval_loss_weight=0.0,
        memory_denominator_implemented=False, target_image_used_for_optimization=False,
        target_concept_text=semantic['canonical_text'], semantic_targets=semantic['semantic_texts'],
        semantic_targets_sha256=semantic['semantic_targets_sha256'],
        paper_query_distribution_reproduced=False, query_count=len(semantic['semantic_texts']),
        query_template_adaptation=semantic['adaptation'])
    attack = dict(steps_requested=config.steps, steps_completed=config.steps, selected_step=config.steps,
        stop_reason='requested_steps_completed', epsilon=epsilon, epsilon_levels=config.epsilon_levels,
        step_size=step_size, step_size_levels=config.step_size_levels, norm='linf',
        optimizer='projected_sign_gradient_descent', initialization='clean_source_zero_delta',
        gradient_execution='sequential_surrogate_graphs_plus_separate_perceptual_graph',
        gradient_restarts=config.gradient_restarts, gradient_variances=variances,
        variance_definition='mean_elementwise_population_variance_of_semantic_input_gradient_over_random_projected_starts',
        variance_floor=gradient_variance_floor(variances),
        variance_floor_policy='max(max_model_variance*1e-12,float64_tiny)',
        ensemble_weight_formula='normalize(max(variance,relative_floor)^-2)',
        ensemble_weights=weights, weights_frozen_before_optimization=True)
    metadata = dict(schema_version=1, method='visual_inception', status='success',
        test_fixture=config.smoke_test, production_backend=not config.smoke_test,
        config=config.to_dict(), config_sha256=content_sha256(config.to_dict()), seed=config.seed,
        source_image=str(source), source_image_sha256=semantic['source_image']['sha256'],
        target_image=str(target), target_image_sha256=semantic['target_image']['sha256'],
        output_image=str(output), output_image_sha256=sha256_file(output),
        training_objective=dict(semantic_definition, surrogate_selection=selection.audit()), semantic_target_audit=semantic,
        surrogate_selection=selection.audit(), surrogates=snapshots,
        text_encoding_audits=text_encoding, perceptual=perceptual.snapshot(),
        attack=attack, history=history, decoded_image_audit=decoded,
        final_objective=dict(semantic_losses=final_semantic, perceptual_loss=final_distance,
            loss=config.semantic_weight * sum(w*v for w,v in zip(weights,final_semantic)) + config.perceptual_weight * final_distance),
        hardware=hardware, runtime=dict(torch_version=str(torch.__version__), device=str(requested_device)),
        optimization_seconds=time.perf_counter() - started,
        retrieval_loss_omitted=True, victim_memory_used=False, formal_queries_used=False,
        formal_ranking_used=False, primary_evaluator_used_during_attack=False,
        paper_encoder_profile_selected=selection.profile == 'visual_inception_paper',
        paper_encoder_architectures_aligned=not config.smoke_test and selection.profile == 'visual_inception_paper',
        paper_encoder_checkpoint_status='public_checkpoint_choices_paper_exact_revisions_unspecified',
        paper_encoder_ensemble_reproduced=False, paper_full_attack_reproduced=False,
        assumptions=['linf_projection', 'clean_source_initialization',
            'random_start_semantic_gradient_population_variance_mean', 'fixed_uncertainty_weights',
            'three_public_templates_are_semantic_targets_without_retrieval_loss'])
    for name in ('metadata.json', 'attack_metrics.json'):
        save_json(root / name, metadata)
    ready = dict(schema_version=1, method='visual_inception', status='completed',
        test_fixture=config.smoke_test, production_backend=not config.smoke_test,
        output_image=str(output), output_image_sha256=metadata['output_image_sha256'],
        source_image=str(source), source_image_sha256=metadata['source_image_sha256'],
        target_image=str(target), target_image_sha256=metadata['target_image_sha256'],
        metadata_sha256=sha256_file(root / 'metadata.json'), attack_metrics_sha256=sha256_file(root / 'attack_metrics.json'),
        config_sha256=metadata['config_sha256'], steps_requested=config.steps,
        steps_completed=config.steps, epsilon=epsilon, epsilon_levels=config.epsilon_levels,
        training_objective_kind=OBJECTIVE_KIND, retrieval_loss_omitted=True,
        candidates=dict(final=dict(image=str(output), sha256=metadata['output_image_sha256'])))
    save_json(root / 'ready.json', ready)
    del models, perceptual, text_features, image
    gc.collect()
    if requested_device.type == 'cuda':
        torch.cuda.empty_cache()
    return output, metadata


def generate_visual_inception(source, target, output_dir, *, config=None, steps=None,
        epsilon_levels=None, device='cuda', target_text=None, definition=None,
        allow_tf32=True, seed=42, smoke_test=False, surrogate_set=None, surrogate_config=None):
    root = Path(output_dir).resolve()
    with catalog_training_guard(allowed_output_roots=[root, Path(source).resolve().parent,
                                                     Path(target).resolve().parent]) as guard:
        raw = config
        if isinstance(config, (str, Path)):
            raw = json.loads(Path(config).read_text(encoding='utf-8'))
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
        cfg = apply_surrogate_overrides(replace(cfg, **changes), surrogate_set, surrogate_config)
        cfg.validate()
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
