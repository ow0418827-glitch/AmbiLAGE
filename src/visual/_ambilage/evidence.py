"""正式成图完成凭证的独立入口校验。"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from src.visual.image_utils import sha256_file


def validate_ready(root,require_production=True):
    root=Path(root).resolve()
    ready=json.loads((root/'ready.json').read_text(encoding='utf-8'))
    if ready.get('method')!='ambilage' or ready.get('status')!='completed' or ready.get('schema_version')!=3:
        raise ValueError('新方法完成凭证格式无效')
    if require_production and (ready.get('test_fixture') is not False or ready.get('production_backend') is not True):
        raise ValueError('轻量夹具禁止进入正式评估')
    names={'foreground_segmentation','foreground_statistics','color_harmonization','latent_generation_refinement'}
    if set(ready.get('stages',{}))!=names or any(s.get('status')!='completed' for s in ready['stages'].values()):
        raise ValueError('完整四阶段证据缺失')
    req=ready.get('steps_requested')
    comp=ready.get('steps_completed')
    if type(req) is not int or type(comp) is not int or req<1 or comp<1 or comp>req:
        raise ValueError('潜空间步骤未完整完成')
    stage=ready['stages']['latent_generation_refinement']
    anneal=ready.get('refinement',{}).get('adaptive_annealing')
    cfg=ready.get('config',{})
    if comp!=req:
        min_steps=min(int(cfg.get('anneal_min_steps',req)),req) if type(cfg.get('anneal_min_steps')) is int and cfg['anneal_min_steps']>=1 else req
        early_ok=(
            isinstance(anneal,dict)
            and anneal.get('early_stopped') is True
            and anneal.get('stop_reason')=='adaptive_annealing_converged'
            and ready.get('stop_reason')=='adaptive_annealing_converged'
            and stage.get('stop_reason')=='adaptive_annealing_converged'
            and cfg.get('adaptive_annealing') is True
            and type(cfg.get('anneal_early_stop_patience')) is int
            and cfg['anneal_early_stop_patience']>0
            and anneal.get('steps_executed')==comp
            and comp>=min_steps
        )
        if not early_ok:
            raise ValueError('潜空间步骤未完整完成')
    if ready.get('pixel_optimization') is not False or ready.get('auxiliary_max_levels')!=0:
        raise ValueError('生产主方法不允许像素残差')
    if not ready.get('quality',{}).get('quality_ok') or ready.get('final_naturalness',{}).get('passed') is not True:
        raise ValueError('生成质量或自然度门禁未通过')
    for name,digest in ready.get('artifacts',{}).items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha256_file(path)!=digest:
            raise ValueError('生成文件摘要不匹配: '+name)
    required={'adversarial.png','generative_anchor.png','print_core.png','naturalness_gate.json',
              'source_foreground.png','target_foreground.png','latent_edit_mask.png','naturalness_candidate.png',
              'training_surrogates.json'}
    if not required.issubset(ready.get('artifacts',{})):
        raise ValueError('缺少关键冻结文件')
    final=(root/'adversarial.png').resolve()
    expected_digest=ready['artifacts']['adversarial.png']
    candidate=ready.get('candidates',{}).get('final',{})
    if (Path(ready.get('output_image','')).resolve()!=final or
        Path(candidate.get('image','')).resolve()!=final or
        candidate.get('sha256')!=expected_digest or ready.get('output_image_sha256')!=expected_digest):
        raise ValueError('最终候选与冻结成图未绑定')
    if stage.get('steps_requested')!=req or stage.get('steps_completed')!=comp:
        raise ValueError('阶段及顶层步数不一致')
    if cfg.get('steps')!=req or \
       ready.get('refinement',{}).get('steps_completed')!=comp or \
       ready.get('refinement',{}).get('steps_requested',req)!=req:
        raise ValueError('配置或细化过程步数不一致')
    from src.visual.training_objectives import SUPPORTED_TRAINING_OBJECTIVE_KINDS
    from src.visual.training_surrogates import (
        is_held_out_purifier_reference,
        validate_loaded_surrogates,
        validate_selection_audit,
    )
    training=ready.get('training_objective',{})
    selection_audit=training.get('surrogate_selection')
    selection=validate_selection_audit(selection_audit,ready.get('config',{}))
    if json.loads((root/'training_surrogates.json').read_text(encoding='utf-8'))!=selection_audit:
        raise ValueError('冻结训练代理文件与实际训练记录不一致')
    public_objectives=training.get('public_objectives',[])
    if (training.get('kind') not in SUPPORTED_TRAINING_OBJECTIVE_KINDS or len(public_objectives)!=len(selection.specs) or any(
        'multimodal_equal_weight' not in p.get('query_modes', []) or
        'text_only' in p.get('query_modes', []) or
        p.get('query_image_weight')!=0.5 or p.get('query_text_weight')!=0.5 or
        float(p.get('text_weight', -1.0)) != float(cfg.get('text_weight', 0.0)) or
        float(p.get('local_ot_weight', -1.0)) != float(cfg.get('local_ot_weight', 0.0))
        for p in public_objectives)):
        raise ValueError('新版图文+独立文本+前景局部最优传输训练目标审计不完整；旧版产物不能作为当前成图复用')
    purifier_snap=training.get('purifier',{}) or {}
    if (is_held_out_purifier_reference(purifier_snap.get('config',{}).get('model_path')) or
        purifier_snap.get('kind')=='sd21_base_cognitiveguard_system1_ddim_purification_v1'):
        raise ValueError('留出评估净化器 SD 2.1-base 禁止用于训练阶段')
    if cfg.get('training_manifest'):
        bundle_sha = ready.get('training_bundle_sha256')
        manifest_sha = ready.get('training_manifest_sha256')
        if not bundle_sha or not manifest_sha:
            raise ValueError('启用公开训练包时必须绑定训练包与清单摘要')
        if float(cfg.get('public_retrieval_weight', 0.0)) > 0.0 and training.get('training_bundle_sha256') != bundle_sha:
            raise ValueError('公开训练包摘要与训练目标记录不一致')
    validate_loaded_surrogates(training.get('surrogates',[]),selection,
        allow_fixture=not require_production and ready.get('test_fixture') is True)
    if require_production:
        if (training.get('eot_enabled') is not True or training.get('diffusion_weight',0)<=0 or
            training.get('diffusion_gradient')!='full' or
            training.get('purifier',{}).get('config',{}).get('reverse_steps')!=10):
            raise ValueError('期望变换或全梯度十步净化证据不完整')
        vae=ready.get('refinement',{}).get('vae',{})
        if vae.get('kind')!='flux_16_channel_vae' or vae.get('latent_channels')!=16:
            raise ValueError('缺少十六通道生成载体证据')
        if ready.get('config',{}).get('smoke_test') is not False:
            raise ValueError('生产配置不能启用轻量夹具')
    metadata=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    for name in ('method','source_image','target_image','source_sha256','target_sha256',
                 'output_image','output_image_sha256','artifacts','stages','test_fixture','production_backend',
                 'training_objective','refinement','config','final_naturalness','steps_requested','steps_completed','stop_reason'):
        if metadata.get(name)!=ready.get(name):
            raise ValueError('元数据与完成凭证不一致: '+name)
    result_path=root/'generation_result.json'
    if result_path.is_file():
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if result.get('status')!='success' or result.get('source_image_sha256')!=ready['source_sha256'] or \
           result.get('target_image_sha256')!=ready['target_sha256'] or result.get('output_image_sha256')!=expected_digest:
            raise ValueError('生成入口结果与冻结成图不一致')
    actual_gate=json.loads((root/'naturalness_gate.json').read_text(encoding='utf-8'))
    if actual_gate!=ready['final_naturalness'] or actual_gate.get('passed') is not True:
        raise ValueError('自然度门禁凭证与实际文件不一致')
    if actual_gate.get('candidate_sha256')!=ready['artifacts']['naturalness_candidate.png'] or \
       ready['artifacts']['naturalness_candidate.png']!=expected_digest:
        raise ValueError('自然度审核候选与最终成图不一致')
    source=Path(ready['source_image'])
    target=Path(ready['target_image'])
    if sha256_file(source)!=ready['source_sha256'] or sha256_file(target)!=ready['target_sha256']:
        raise ValueError('源或目标图摘要不匹配')
    roi=np.asarray(Image.open(root/'print_core.png').convert('L'))>0
    original=np.asarray(Image.open(source).convert('RGB'))
    for name in ('generative_anchor.png','adversarial.png'):
        image=np.asarray(Image.open(root/name).convert('RGB'))
        if image.shape!=original.shape or roi.shape!=image.shape[:2] or not np.array_equal(image[roi],original[roi]):
            raise ValueError('印刷核心未保持源图零改动')
    return ready
