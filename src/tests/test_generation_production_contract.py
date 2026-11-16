"""生产禁止静默模型替代，以及明确微型夹具的回归验证。"""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.visual._ambilage.config import AmbiLAGEConfig
from src.visual._ambilage.pipeline import AmbiLAGEPipeline
from src.visual._ambilage.evidence import validate_ready
from src.visual._ambilage.latent_refinement import AmbiLAGEVae
from src.visual._ambilage.smoke import TinyAutoencoder
from src.visual.optimization import generate_attack


def images(root):
    source=root/'source.png';target=root/'target.png'
    Image.new('RGB',(24,24),(180,170,160)).save(source)
    Image.new('RGB',(24,24),(80,100,140)).save(target)
    return source,target


def test_production_missing_public_key_fails_before_loading(tmp_path):
    source,target=images(tmp_path)
    with pytest.raises(ValueError,match='完整公开检索键'):
        AmbiLAGEPipeline(device='cpu').execute(source,target,tmp_path/'out',target_text='公开特点')
    assert not (tmp_path/'out/ready.json').exists()


def test_production_cpu_cannot_substitute_random_network(tmp_path):
    source,target=images(tmp_path)
    with pytest.raises(RuntimeError,match='生产自然图生成要求可用显卡'):
        AmbiLAGEPipeline(device='cpu').execute(source,target,tmp_path/'out',target_text='公开特点',
            source_public_key='源商品名称及真实特点',source_category='公开品类')
    assert not (tmp_path/'out/ready.json').exists()


def test_smoke_evidence_rejected_for_formal_evaluation(tmp_path):
    source,target=images(tmp_path)
    cfg=AmbiLAGEConfig(steps=1,smoke_test=True)
    image,ready=AmbiLAGEPipeline(cfg,device='cpu').execute(source,target,tmp_path/'out',target_text='公开特点',
        print_regions=[[.2,.2,.4,.4]])
    assert ready['test_fixture'] is True and ready['production_backend'] is False
    assert validate_ready(image.parent,require_production=False)['status']=='completed'
    with pytest.raises(ValueError,match='轻量夹具禁止'):
        validate_ready(image.parent)
    original=np.asarray(Image.open(source));edited=np.asarray(Image.open(image))
    assert np.array_equal(original[5:10,5:10],edited[5:10,5:10])
    Image.new('RGB',(24,24),(0,0,0)).save(image)
    with pytest.raises(ValueError,match='摘要不匹配'):
        validate_ready(image.parent,require_production=False)


def test_vae_flux_shift_and_scaling_are_inverted():
    wrapper=AmbiLAGEVae('cpu',model=TinyAutoencoder(),test_fixture=True)
    image=torch.full((1,3,16,16),.7)
    latent=wrapper.encode(image)
    assert torch.allclose(latent,torch.full_like(latent,.15),atol=1e-6)
    assert torch.allclose(wrapper.decode_raw(latent,image.shape),image,atol=1e-6)
    changed=latent.detach().requires_grad_(True)
    wrapper.decode_raw(changed,image.shape).sum().backward()
    assert changed.grad.abs().sum()>0


def test_vae_rejects_four_channel_production_model():
    model=TinyAutoencoder();model.config.latent_channels=4
    with pytest.raises(ValueError,match='十六通道'):
        AmbiLAGEVae('cpu',model=model)


def test_unknown_config_and_excess_subject_budget_fail():
    with pytest.raises(ValueError,match='未知方法配置字段'):
        AmbiLAGEConfig.from_dict({'typo_steps':350})
    with pytest.raises(ValueError,match='\\[0, 6\\]'):
        AmbiLAGEConfig(latent_levels=7).validate()


def test_zero_steps_not_silently_replaced_by_default(tmp_path):
    source,target=images(tmp_path)
    with pytest.raises(ValueError,match='优化步数'):
        generate_attack(source,target,tmp_path/'out',steps=0,target_text='公开特点',device='cpu',smoke_test=True)


def test_smoke_limits_prevent_large_local_workload(tmp_path):
    source,target=images(tmp_path)
    with pytest.raises(ValueError,match='至多三步'):
        AmbiLAGEPipeline(AmbiLAGEConfig(steps=4,smoke_test=True),device='cpu').execute(
            source,target,tmp_path/'out',target_text='公开特点')

@pytest.mark.parametrize('level',[0.,1.])
def test_subject_and_background_both_respect_headroom(level):
    from src.visual._ambilage.ambient import AmbientCoopField
    from src.visual._ambilage.segmentation import compute_spatial_envelopes
    anchor=torch.full((1,3,16,16),level)
    fg=np.zeros((16,16),dtype=bool);fg[4:12,4:12]=True
    cfg=AmbiLAGEConfig()
    e=compute_spatial_envelopes(fg,np.zeros_like(fg),cfg)
    as_t=lambda x:torch.as_tensor(x,dtype=torch.float32)[None,None]
    field=AmbientCoopField(anchor,as_t(e['fg_level_array']/255),as_t(e['bg_level_array']/255),
        as_t(e['w_bg_mod']),as_t(fg),as_t(~fg),as_t(e['support_mask']),cfg)
    for drive in (-1e6,1e6):
        raw=anchor+field.compute_residual(torch.full_like(anchor,drive))
        assert float(raw.detach().min())>=0. and float(raw.detach().max())<=1.

@pytest.mark.parametrize('mutation',['candidate','steps','metadata'])
def test_completion_rejects_cross_file_and_candidate_tampering(tmp_path,mutation):
    source,target=images(tmp_path)
    root=tmp_path/'out'
    _,ready=AmbiLAGEPipeline(AmbiLAGEConfig(steps=1,smoke_test=True),device='cpu').execute(
        source,target,root,target_text='公开特点')
    document=json.loads((root/'ready.json').read_text(encoding='utf-8'))
    metadata=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    if mutation=='candidate':
        document['candidates']['final']['image']=str(source)
    elif mutation=='steps':
        document['stages']['latent_generation_refinement']['steps_completed']=0
    else:
        metadata['source_sha256']='0'*64
    (root/'ready.json').write_text(json.dumps(document),encoding='utf-8')
    (root/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    with pytest.raises(ValueError):
        validate_ready(root,require_production=False)


def test_ocr_uses_installed_runtime_tuple_api(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from src.visual._ambilage.backends import ProductionStages
    source,_=images(tmp_path)
    class OCR:
        def __call__(self,path):
            return [[[[2,3],[10,3],[10,9],[2,9]],'printed',.95]],.01
    monkeypatch.setitem(sys.modules,'rapidocr_onnxruntime',SimpleNamespace(RapidOCR=OCR))
    backend=ProductionStages.__new__(ProductionStages)
    regions,audit=backend.detect_print(source)
    assert regions[0]==pytest.approx([2/24,3/24,10/24,9/24])
    assert audit['confidence']==[.95]


def test_forging_top_level_production_flags_does_not_hide_tiny_models(tmp_path):
    source,target=images(tmp_path)
    root=tmp_path/'out'
    AmbiLAGEPipeline(AmbiLAGEConfig(steps=1,smoke_test=True),device='cpu').execute(source,target,root,target_text='公开特点')
    for name in ('ready.json','metadata.json'):
        document=json.loads((root/name).read_text(encoding='utf-8'))
        document['test_fixture']=False;document['production_backend']=True
        (root/name).write_text(json.dumps(document),encoding='utf-8')
    with pytest.raises(ValueError,match='所选方案的真实训练代理'):
        validate_ready(root)


@pytest.mark.parametrize('mutation', ['dual_kind', 'v2_joint_only', 'text_target', 'config_mismatch'])
def test_legacy_evidence_cannot_be_reused_as_v3_and_rejects_tampered_config(tmp_path, mutation):
    from src.visual.training_objectives import TRAINING_OBJECTIVE_KIND
    source, target = images(tmp_path)
    root = tmp_path / 'out'
    _, ready = AmbiLAGEPipeline(AmbiLAGEConfig(steps=1, smoke_test=True), device='cpu').execute(
        source, target, root, target_text='公开特点')
    assert ready['training_objective']['kind'] == TRAINING_OBJECTIVE_KIND
    assert ready['training_objective']['public_objectives'][0]['query_modes'] == [
        'multimodal_equal_weight', 'independent_target_text'
    ]
    for name in ('ready.json', 'metadata.json'):
        document = json.loads((root / name).read_text(encoding='utf-8'))
        if mutation == 'dual_kind':
            document['training_objective']['kind'] = 'four_proxy_sequential_public_dual_goal_v1'
        elif mutation == 'v2_joint_only':
            document['training_objective']['kind'] = 'sequential_public_joint_goal_v2'
        elif mutation == 'text_target':
            document['training_objective']['public_objectives'][0]['query_modes'].append('text_only')
        else:
            document['config']['text_weight'] = 0.99
        (root / name).write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ValueError, match='新版图文\\+独立文本\\+前景局部最优传输'):
        validate_ready(root, require_production=False)


def test_detect_print_multiscale_and_inverted_recovers_white_on_dark_logo(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from src.visual._ambilage.backends import ProductionStages

    # 构建黑底白字小商标（模拟 08_speaker 网罩上的白色 KICKER 标志）
    arr = np.zeros((160, 160, 3), dtype=np.uint8)
    arr[120:132, 40:80] = 245
    src_path = tmp_path / 'speaker_dark.png'
    Image.fromarray(arr).save(src_path)

    class _FakeMultiScaleOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, inp):
            if isinstance(inp, str):
                return [], 0.01
            # 仅反相 2x 图（平均亮度高 > 128）成功识别到位于 2x 坐标 [80, 240, 160, 264] 的标志
            if isinstance(inp, np.ndarray) and inp.shape[:2] == (320, 320) and float(inp.mean()) > 128.0:
                return [[[[80, 240], [160, 240], [160, 264], [80, 264]], 'KICKER', 0.91]], 0.01
            return [], 0.01

    monkeypatch.setitem(sys.modules, 'rapidocr_onnxruntime', SimpleNamespace(RapidOCR=_FakeMultiScaleOCR))
    backend = ProductionStages.__new__(ProductionStages)
    regions, audit = backend.detect_print(src_path)
    assert len(regions) == 1
    assert regions[0] == pytest.approx([40 / 160, 120 / 160, 80 / 160, 132 / 160])
    assert 'upscale_2x_inverted' in audit['passes']
    assert audit['confidence'] == [0.91]
