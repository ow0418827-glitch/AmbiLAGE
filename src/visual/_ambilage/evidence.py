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
    if ready.get('steps_requested')!=ready.get('steps_completed') or ready.get('steps_completed',0)<1:
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
              'source_foreground.png','target_foreground.png','latent_edit_mask.png','naturalness_candidate.png'}
    if not required.issubset(ready.get('artifacts',{})):
        raise ValueError('缺少关键冻结文件')
    final=(root/'adversarial.png').resolve()
    expected_digest=ready['artifacts']['adversarial.png']
    candidate=ready.get('candidates',{}).get('final',{})
    if (Path(ready.get('output_image','')).resolve()!=final or
        Path(candidate.get('image','')).resolve()!=final or
        candidate.get('sha256')!=expected_digest or ready.get('output_image_sha256')!=expected_digest):
        raise ValueError('最终候选与冻结成图未绑定')
    stage=ready['stages']['latent_generation_refinement']
    if stage.get('steps_requested')!=ready['steps_requested'] or stage.get('steps_completed')!=ready['steps_completed']:
        raise ValueError('阶段及顶层步数不一致')
    if ready.get('config',{}).get('steps')!=ready['steps_completed'] or \
       ready.get('refinement',{}).get('steps_completed')!=ready['steps_completed']:
        raise ValueError('配置或细化过程步数不一致')
    from src.visual.training_objectives import TRAINING_FAMILIES, TRAINING_OBJECTIVE_KIND
    training=ready.get('training_objective',{})
    public_objectives=training.get('public_objectives',[])
    if (training.get('kind')!=TRAINING_OBJECTIVE_KIND or len(public_objectives)!=4 or any(
        p.get('query_modes')!=['multimodal_equal_weight'] or
        p.get('query_image_weight')!=0.5 or p.get('query_text_weight')!=0.5
        for p in public_objectives) or 'text_weight' in ready.get('config',{})):
        raise ValueError('仅图文联合训练目标证据不完整；旧双目标产物不能作为当前成图复用')
    if require_production:
        proxies=training.get('surrogates',[])
        if tuple(p.get('name') for p in proxies)!=TRAINING_FAMILIES or any(p.get('test_fixture') for p in proxies):
            raise ValueError('缺少四个指定真实训练代理的审计证据')
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
                 'training_objective','refinement','config','final_naturalness','steps_requested','steps_completed'):
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
