"""四阶段单通生成：公开规划/分割、统计、生成锚点、环境光潜空间协同。"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from src.visual.hardware import apply_hardware_profile
from src.visual.image_utils import read_rgb, read_tensor, save_tensor_image, sha256_file, write_json
from src.visual.training_surrogates import selection_from_config, validate_loaded_surrogates, apply_surrogate_overrides
from .backends import ProductionStages, release_memory
from .color_selection import extract_foreground_statistics
from .config import AmbiLAGEConfig, DEFINITION, METHOD
from .illumination_harmonization import harmonize_illumination
from .latent_refinement import AmbiLAGEVae, refine_ambient_coop_latent
from .segmentation import compute_spatial_envelopes, print_region_mask


class AmbiLAGEPipeline:
    def __init__(self, config=None, device='cuda', *, stage_backend=None):
        self.config=config or AmbiLAGEConfig()
        self.config.validate()
        self.surrogate_selection=selection_from_config(self.config)
        self.device=torch.device(device)
        self.hardware_profile=apply_hardware_profile(self.device,allow_tf32=self.config.allow_tf32,seed=self.config.seed)
        self.stage_backend=stage_backend

    def execute(self,source,target,output_dir,*,target_text=None,source_title=None,
                source_public_key=None,source_category=None,source_boxes=None,target_boxes=None,
                print_regions=None,surrogate_encoder=None):
        source,target=Path(source).resolve(),Path(target).resolve()
        for path in (source,target):
            if not path.is_file():
                raise FileNotFoundError(f'商品图片不存在: {path}')
        if not isinstance(target_text,str) or not target_text.strip():
            raise ValueError('必须显式提供冻结的目标特点训练正文，不能从文件名猜测')
        if self.config.smoke_test:
            from .smoke import SmokeStages
            backend=self.stage_backend or SmokeStages(self.config,self.device)
            source_public_key=source_public_key or source_title or '合成夹具公开商品'
            source_category=source_category or '合成夹具'
        else:
            if not isinstance(source_public_key,str) or not source_public_key.strip():
                raise ValueError('生产方法必须提供源商品自己的完整公开检索键')
            if not isinstance(source_category,str) or not source_category.strip():
                raise ValueError('生产规划必须提供源商品公开品类')
            backend=self.stage_backend or ProductionStages(self.config,self.device)
        if bool(getattr(backend,'test_fixture',False))!=self.config.smoke_test:
            raise ValueError('阶段后端身份与轻量验证开关不一致')
        root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
        if (root/'ready.json').exists():
            raise FileExistsError('输出目录已有完成凭证，请使用新目录避免覆盖冻结结果')
        write_json(root/'training_surrogates.json',self.surrogate_selection.audit())
        started=time.perf_counter()
        try:
            plan=backend.plan(source,source_category)
            backend.close()
            detected,ocr=backend.detect_print(source)
            regions=list(plan.get('print_regions',[]))+list(detected)+list(print_regions or [])
            source_fg,source_seg=backend.segment(source,source_boxes or plan['source_boxes'])
            target_fg,target_seg=backend.segment(target,target_boxes)
            backend.close()
            roi=print_region_mask(source_fg.shape,regions)
            envelopes=compute_spatial_envelopes(source_fg,roi,self.config)
            for name,mask in [('source_foreground',source_fg),('target_foreground',target_fg),('print_core',roi)]:
                Image.fromarray(mask.astype(np.uint8)*255).save(root/(name+'.png'))
            Image.fromarray(np.rint(envelopes['macro_array']*255).astype(np.uint8)).save(root/'latent_edit_mask.png')
            color_stats,_=extract_foreground_statistics(target,target_fg,output_dir=root/'foreground_statistics')
            initial,harmonization=harmonize_illumination(source,color_stats['rgb'],source_fg,regions,self.config,
                output_dir=root/'color_harmonization')
            anchor,anchor_audit=backend.generate_anchor(initial,plan['edit_prompt'],self.config.seed)
            backend.close()
            if anchor.shape!=read_rgb(source).shape:
                raise ValueError('自然锚点分辨率与源图不一致')
            anchor[roi]=read_rgb(source)[roi]
            Image.fromarray(anchor).save(root/'generative_anchor.png')
            from src.visual.training_objectives import SequentialEnsembleObjective,load_training_surrogates
            if self.config.smoke_test:
                from .smoke import TinySurrogate,TinyPurifier
                surrogates=[TinySurrogate(i) for i in range(len(self.surrogate_selection.specs))]
                purifier=TinyPurifier()
                vae=backend.make_vae()
                effective=replace(self.config,latent_working_size=64)
            else:
                surrogates=list(surrogate_encoder) if surrogate_encoder is not None else load_training_surrogates(
                    str(self.device),selection=self.surrogate_selection)
                validate_loaded_surrogates(surrogates,self.surrogate_selection)
                purifier=None
                vae=AmbiLAGEVae(self.device,custom_model_dir=self.config.vae_model_dir,
                    enable_checkpointing=self.config.enable_gradient_checkpointing)
                effective=self.config
            objective=SequentialEnsembleObjective(surrogates,read_tensor(target,self.device),source_public_key,target_text,
                effective,purifier=purifier,selection=self.surrogate_selection)
            audit=objective.audit
            anchor_t=torch.tensor(anchor.transpose(2,0,1).copy(),device=self.device,dtype=torch.float32)[None]/255.
            final,refinement=refine_ambient_coop_latent(anchor_t,read_tensor(source,self.device),envelopes,
                objective,None,vae,effective,output_dir=root)
            del objective,surrogates,vae,purifier
            release_memory()
            final_rgb=np.rint(final[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            if not np.array_equal(final_rgb[roi],read_rgb(source)[roi]):
                raise AssertionError('最终印刷核心与源原图存在像素差异')
            Image.fromarray(final_rgb).save(root/'naturalness_candidate.png')
            gate=backend.gate(source,final_rgb,source_category)
            gate['candidate_sha256']=sha256_file(root/'naturalness_candidate.png')
            backend.close()
            write_json(root/'naturalness_gate.json',gate)
            if not gate['passed']:
                raise RuntimeError('最终自然度门禁未通过，证据已保存，禁止标记为完整成图')
            adversarial=root/'adversarial.png';save_tensor_image(adversarial,final)
            stage_names=('foreground_segmentation','foreground_statistics','color_harmonization','latent_generation_refinement')
            stages={name:{'status':'completed'} for name in stage_names}
            stages['latent_generation_refinement'].update(steps_requested=self.config.steps,steps_completed=refinement['steps_completed'])
            artifacts={name:sha256_file(root/name) for name in ('adversarial.png','generative_anchor.png','print_core.png',
                'source_foreground.png','target_foreground.png','latent_edit_mask.png','naturalness_gate.json',
                'naturalness_candidate.png','training_surrogates.json')}
            metadata=dict(schema_version=3,status='completed',method=METHOD,definition=DEFINITION,
                test_fixture=self.config.smoke_test,production_backend=not self.config.smoke_test,
                source_image=str(source),source_sha256=sha256_file(source),target_image=str(target),target_sha256=sha256_file(target),
                output_image=str(adversarial),output_image_sha256=sha256_file(adversarial),
                candidates={'final':{'image':str(adversarial),'sha256':sha256_file(adversarial)}},
                pipeline='ambilage_native_4stage_single_pass',stages=stages,artifacts=artifacts,
                pixel_optimization=False,auxiliary_max_levels=0,optimization_mode='ambient_coop_latent_only',
                steps=self.config.steps,steps_requested=self.config.steps,steps_completed=refinement['steps_completed'],
                quality=refinement['final_quality'],refinement=refinement,config=asdict(self.config),
                planning=plan,ocr=ocr,segmentation={'source':source_seg,'target':target_seg},
                color_statistics=color_stats,harmonization=harmonization,anchor_generation=anchor_audit,
                training_objective=audit,final_naturalness=gate,hardware_profile=self.hardware_profile,
                victim_memory_used=False,formal_queries_used=False,shadow_pool_used=False,true_rank_used=False,
                elapsed_seconds=time.perf_counter()-started)
            write_json(root/'metadata.json',metadata)
            write_json(root/'ready.json',metadata)
            return adversarial,metadata
        except Exception as exc:
            write_json(root/'FAILED.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc),
                'test_fixture':self.config.smoke_test,'elapsed_seconds':time.perf_counter()-started})
            raise
        finally:
            backend.close()


def generate_ambilage(source,target,output_dir,*,definition=None,config=None,device='cuda',target_text=None,
                      steps=None,allow_tf32=True,seed=42,smoke_test=False,
                      surrogate_set=None,surrogate_config=None,**kwargs):
    definition=definition or {}
    cfg=config or AmbiLAGEConfig()
    cfg=apply_surrogate_overrides(cfg,surrogate_set,surrogate_config)
    overrides=dict(allow_tf32=allow_tf32,seed=seed,device=device,smoke_test=smoke_test or cfg.smoke_test)
    if steps is not None:
        overrides['steps']=steps
    cfg=replace(cfg,**overrides);cfg.validate()
    data=definition.get('data',{})
    return AmbiLAGEPipeline(cfg,device).execute(source,target,output_dir,
        target_text=target_text or (data.get('training_queries') or [None])[0],
        source_public_key=definition.get('source_public_key') or data.get('source_public_key'),
        source_title=data.get('source_title'),source_category=definition.get('source_category') or data.get('source_category'),
        source_boxes=definition.get('source_boxes'),target_boxes=definition.get('target_boxes'),
        print_regions=definition.get('print_regions'),**kwargs)


def run_attack(source,target,output_dir,*,target_text=None,steps=None,device='cuda',ambilage_config=None,
               allow_tf32=True,seed=42,definition=None,smoke_test=False,**kwargs):
    if isinstance(ambilage_config,(str,Path)):
        import json
        ambilage_config=json.loads(Path(ambilage_config).read_text(encoding='utf-8'))
    if isinstance(ambilage_config,dict):
        ambilage_config=AmbiLAGEConfig.from_dict(ambilage_config)
    return generate_ambilage(source,target,output_dir,definition=definition,config=ambilage_config,device=device,
        target_text=target_text,steps=steps,allow_tf32=allow_tf32,seed=seed,smoke_test=smoke_test,**kwargs)
