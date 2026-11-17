"""四阶段单通生成：公开规划/分割、统计、生成锚点、环境光潜空间协同。"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from src.paths import PROJECT
from src.visual.hardware import apply_hardware_profile
from src.visual.image_utils import read_rgb, read_tensor, save_tensor_image, sha256_file, write_json
from src.visual.training_surrogates import selection_from_config, validate_loaded_surrogates, apply_surrogate_overrides
from .backends import ProductionStages, release_memory
from .color_selection import extract_foreground_statistics
from .config import AmbiLAGEConfig, DEFINITION, METHOD
from .illumination_harmonization import harmonize_illumination
from .latent_refinement import AmbiLAGEVae, STAGE4_CHECKPOINT_FILENAME, refine_ambient_coop_latent
from .segmentation import compute_spatial_envelopes, extract_ink_core_mask, print_region_mask


STAGE3_STATE_FILENAME = 'stage3_state.json'


def _stage3_signature(
    source_sha256: str,
    target_sha256: str,
    target_text: str,
    source_public_key: str,
    source_category: str,
    config: AmbiLAGEConfig,
    source_boxes: Any,
    target_boxes: Any,
    print_regions: Any,
    surrogate_specs_sha256: str,
    training_manifest_sha256: str | None = None,
) -> str:
    cfg_dict = config.to_dict()
    for runtime_key in ('log_every', 'checkpoint_every', 'resume_from_checkpoint', 'save_intermediate_pngs', 'device'):
        cfg_dict.pop(runtime_key, None)
    payload = {
        'source_sha256': source_sha256,
        'target_sha256': target_sha256,
        'target_text': target_text,
        'source_public_key': source_public_key,
        'source_category': source_category,
        'source_boxes': source_boxes,
        'target_boxes': target_boxes,
        'print_regions': print_regions,
        'config': cfg_dict,
        'surrogate_specs_sha256': surrogate_specs_sha256,
    }
    if training_manifest_sha256 is not None:
        payload['training_manifest_sha256'] = training_manifest_sha256
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


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
                print_regions=None,surrogate_encoder=None,definition=None):
        source,target=Path(source).resolve(),Path(target).resolve()
        for path in (source,target):
            if not path.is_file():
                raise FileNotFoundError(f'商品图片不存在: {path}')
        if not isinstance(target_text,str) or not target_text.strip():
            raise ValueError('必须显式提供冻结的目标特点训练正文，不能从文件名猜测')
        training_bundle = None
        manifest_sha = None
        if self.config.training_manifest:
            from src.visual._visual_inception.targets import prepare_semantic_targets
            from src.visual._visual_inception.training_data import load_training_bundle
            manifest_p = Path(self.config.training_manifest).expanduser()
            manifest_p = (manifest_p if manifest_p.is_absolute() else PROJECT / manifest_p).resolve()
            semantic_targets = prepare_semantic_targets(source, target, target_text=target_text, definition=definition)
            training_bundle = load_training_bundle(
                manifest_p,
                source=source,
                target=target,
                public_semantic=semantic_targets,
                require_production=not self.config.smoke_test,
            )
            manifest_sha = training_bundle['manifest_sha256']
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
        source_sha = sha256_file(source)
        target_sha = sha256_file(target)
        surrogate_specs_sha = self.surrogate_selection.audit()['specs_sha256']
        stage3_sig = _stage3_signature(
            source_sha, target_sha, target_text, source_public_key, source_category,
            self.config, source_boxes, target_boxes, print_regions, surrogate_specs_sha,
            training_manifest_sha256=manifest_sha,
        )
        stage3_path = root / STAGE3_STATE_FILENAME
        ckpt4_path = root / STAGE4_CHECKPOINT_FILENAME
        required_stage3_imgs = (
            root / 'source_foreground.png',
            root / 'target_foreground.png',
            root / 'print_core.png',
            root / 'latent_edit_mask.png',
            root / 'generative_anchor.png',
        )
        prior_stage3_elapsed = 0.0
        resumed_stage3 = False
        try:
            if self.config.resume_from_checkpoint and stage3_path.is_file() and all(p.is_file() for p in required_stage3_imgs):
                saved_stage3 = json.loads(stage3_path.read_text(encoding='utf-8'))
                if saved_stage3.get('signature') != stage3_sig:
                    raise ValueError('Stage 3 断点状态与当前输入或配置不一致，请清理输出目录或禁用断点续跑后重试')
                plan = saved_stage3['plan']
                ocr = saved_stage3['ocr']
                source_seg = saved_stage3['source_seg']
                target_seg = saved_stage3['target_seg']
                color_stats = saved_stage3['color_stats']
                harmonization = saved_stage3['harmonization']
                anchor_audit = saved_stage3['anchor_audit']
                prior_stage3_elapsed = float(saved_stage3.get('stage3_elapsed_seconds', 0.0))
                source_rgb = read_rgb(source)
                source_fg = np.asarray(Image.open(root / 'source_foreground.png').convert('L')) > 0
                target_fg = np.asarray(Image.open(root / 'target_foreground.png').convert('L')) > 0
                roi = np.asarray(Image.open(root / 'print_core.png').convert('L')) > 0
                envelopes = compute_spatial_envelopes(source_fg, roi, self.config, source_rgb=source_rgb)
                anchor = np.asarray(Image.open(root / 'generative_anchor.png').convert('RGB')).copy()
                if anchor.shape != source_rgb.shape or not np.array_equal(anchor[roi], source_rgb[roi]):
                    raise ValueError('Stage 3 断点锚点图像与源图印刷核心不一致')
                (root / 'FAILED.json').unlink(missing_ok=True)
                resumed_stage3 = True
                print('AmbiLAGE 检测到有效 Stage 3 断点，跳过分割与锚点生成直接恢复 Stage 4', flush=True)
            else:
                if not self.config.resume_from_checkpoint:
                    stage3_path.unlink(missing_ok=True)
                    ckpt4_path.unlink(missing_ok=True)
                elif ckpt4_path.is_file():
                    raise ValueError('发现孤立 Stage 4 断点文件但缺少对应 Stage 3 状态，请清理输出目录后重试')
                (root / 'FAILED.json').unlink(missing_ok=True)
                plan=backend.plan(source,source_category)
                backend.close()
                detected,ocr=backend.detect_print(source)
                regions=list(plan.get('print_regions',[]))+list(detected)+list(print_regions or [])
                source_fg,source_seg=backend.segment(source,source_boxes or plan['source_boxes'])
                target_fg,target_seg=backend.segment(target,target_boxes)
                backend.close()
                rect_roi = print_region_mask(source_fg.shape, regions)
                source_rgb = read_rgb(source)
                roi, substrate = extract_ink_core_mask(source_rgb, source_fg, rect_roi, self.config)
                envelopes = compute_spatial_envelopes(source_fg, roi, self.config, source_rgb=source_rgb)
                for name, mask in [('source_foreground', source_fg), ('target_foreground', target_fg), ('print_core', roi)]:
                    Image.fromarray(mask.astype(np.uint8) * 255).save(root / (name + '.png'))
                Image.fromarray(np.rint(envelopes['macro_array'] * 255).astype(np.uint8)).save(root / 'latent_edit_mask.png')
                color_stats, _ = extract_foreground_statistics(target, target_fg, output_dir=root / 'foreground_statistics')
                initial, harmonization = harmonize_illumination(
                    source,
                    color_stats['rgb'],
                    source_fg,
                    regions,
                    self.config,
                    output_dir=root / 'color_harmonization',
                    ink_core_mask=roi,
                    substrate_mask=substrate,
                )
                anchor, anchor_audit = backend.generate_anchor(initial, plan['edit_prompt'], self.config.seed)
                backend.close()
                if anchor.shape != source_rgb.shape:
                    raise ValueError('自然锚点分辨率与源图不一致')
                anchor[roi] = source_rgb[roi]
                Image.fromarray(anchor).save(root / 'generative_anchor.png')
                write_json(stage3_path, {
                    'signature': stage3_sig,
                    'plan': plan,
                    'ocr': ocr,
                    'source_seg': source_seg,
                    'target_seg': target_seg,
                    'color_stats': color_stats,
                    'harmonization': harmonization,
                    'anchor_audit': anchor_audit,
                    'stage3_elapsed_seconds': float(time.perf_counter() - started),
                })
            from src.visual.training_objectives import SequentialEnsembleObjective, load_training_surrogates
            if self.config.smoke_test:
                from .smoke import TinySurrogate, TinyPurifier
                surrogates = [TinySurrogate(i) for i in range(len(self.surrogate_selection.specs))]
                purifier = TinyPurifier()
                vae = backend.make_vae()
                effective = replace(self.config, latent_working_size=64)
            else:
                surrogates = list(surrogate_encoder) if surrogate_encoder is not None else load_training_surrogates(
                    str(self.device), selection=self.surrogate_selection)
                validate_loaded_surrogates(surrogates, self.surrogate_selection)
                purifier = None
                vae_dir = self.config.vae_model_path or self.config.vae_model_dir or getattr(self.config, 'editor_model_dir', None)
                vae = AmbiLAGEVae(self.device, custom_model_dir=vae_dir,
                    enable_checkpointing=self.config.enable_gradient_checkpointing)
                effective = self.config
            source_tensor = read_tensor(source, self.device)
            resolved_cache_dir = None
            if self.config.candidate_cache_dir:
                c_path = Path(self.config.candidate_cache_dir).expanduser()
                resolved_cache_dir = str((c_path if c_path.is_absolute() else PROJECT / c_path).resolve())
            elif training_bundle is not None and self.config.smoke_test:
                resolved_cache_dir = str((root / 'candidate_cache').resolve())
            objective = SequentialEnsembleObjective(
                surrogates,
                read_tensor(target, self.device),
                source_public_key,
                target_text,
                effective,
                purifier=purifier,
                selection=self.surrogate_selection,
                source_fg_mask=source_fg & (~roi),
                target_fg_mask=target_fg,
                source_tensor=source_tensor,
                training_bundle=training_bundle,
                candidate_cache_dir=resolved_cache_dir,
            )
            audit = objective.audit
            anchor_t = torch.tensor(anchor.transpose(2, 0, 1).copy(), device=self.device, dtype=torch.float32)[None] / 255.
            checkpoint_binding = {
                'target_text': target_text,
                'source_public_key': source_public_key,
                'surrogate_specs_sha256': surrogate_specs_sha,
            }
            if manifest_sha is not None:
                checkpoint_binding['training_manifest_sha256'] = manifest_sha
            final, refinement = refine_ambient_coop_latent(anchor_t, source_tensor, envelopes,
                objective, None, vae, effective, output_dir=root, checkpoint_binding=checkpoint_binding)
            refinement_resume = dict(refinement.get('checkpoint_resume') or {})
            refinement_resume['stage3_resumed'] = resumed_stage3
            refinement['checkpoint_resume'] = refinement_resume
            del objective, surrogates, vae, purifier
            release_memory()
            final_rgb = np.rint(final[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            if not np.array_equal(final_rgb[roi], source_rgb[roi]):
                raise AssertionError('最终印刷核心与源原图存在像素差异')
            Image.fromarray(final_rgb).save(root / 'naturalness_candidate.png')
            gate = backend.gate(source, final_rgb, source_category, roi_mask=roi, fg_mask=source_fg)
            gate['candidate_sha256']=sha256_file(root/'naturalness_candidate.png')
            backend.close()
            write_json(root/'naturalness_gate.json',gate)
            if not gate['passed']:
                raise RuntimeError('最终自然度门禁未通过，证据已保存，禁止标记为完整成图')
            adversarial=root/'adversarial.png';save_tensor_image(adversarial,final)
            stage_names=('foreground_segmentation','foreground_statistics','color_harmonization','latent_generation_refinement')
            stages={name:{'status':'completed'} for name in stage_names}
            stages['latent_generation_refinement'].update(
                steps_requested=self.config.steps,
                steps_completed=refinement['steps_completed'],
                stop_reason=refinement.get('stop_reason', 'requested_steps_completed'),
            )
            artifacts={name:sha256_file(root/name) for name in ('adversarial.png','generative_anchor.png','print_core.png',
                'source_foreground.png','target_foreground.png','latent_edit_mask.png','naturalness_gate.json',
                'naturalness_candidate.png','training_surrogates.json')}
            metadata=dict(schema_version=3,status='completed',method=METHOD,definition=DEFINITION,
                test_fixture=self.config.smoke_test,production_backend=not self.config.smoke_test,
                source_image=str(source),source_sha256=source_sha,target_image=str(target),target_sha256=target_sha,
                output_image=str(adversarial),output_image_sha256=sha256_file(adversarial),
                candidates={'final':{'image':str(adversarial),'sha256':sha256_file(adversarial)}},
                pipeline='ambilage_native_4stage_single_pass',stages=stages,artifacts=artifacts,
                pixel_optimization=False,auxiliary_max_levels=0,optimization_mode='ambient_coop_latent_only',
                steps=self.config.steps,steps_requested=self.config.steps,steps_completed=refinement['steps_completed'],
                stop_reason=refinement.get('stop_reason', 'requested_steps_completed'),
                quality=refinement['final_quality'],refinement=refinement,config=asdict(self.config),
                planning=plan,ocr=ocr,segmentation={'source':source_seg,'target':target_seg},
                color_statistics=color_stats,harmonization=harmonization,anchor_generation=anchor_audit,
                training_objective=audit,final_naturalness=gate,hardware_profile=self.hardware_profile,
                victim_memory_used=False,formal_queries_used=False,shadow_pool_used=False,true_rank_used=False,
                elapsed_seconds=prior_stage3_elapsed + (time.perf_counter() - started))
            if training_bundle is not None:
                metadata.update(
                    training_bundle_manifest=training_bundle['manifest_path'],
                    training_manifest_sha256=training_bundle['manifest_sha256'],
                    training_bundle_sha256=training_bundle['bundle_sha256'],
                    training_bundle_audit=training_bundle['audit'],
                )
            write_json(root/'metadata.json',metadata)
            write_json(root/'ready.json',metadata)
            ckpt4_path.unlink(missing_ok=True)
            stage3_path.unlink(missing_ok=True)
            (root / 'FAILED.json').unlink(missing_ok=True)
            return adversarial,metadata
        except Exception as exc:
            write_json(root/'FAILED.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc),
                'test_fixture':self.config.smoke_test,'elapsed_seconds':time.perf_counter()-started})
            raise
        finally:
            backend.close()


def generate_ambilage(source,target,output_dir,*,definition=None,config=None,device='cuda',target_text=None,
                      steps=None,allow_tf32=True,seed=42,smoke_test=False,
                      surrogate_set=None,surrogate_config=None,training_manifest=None,**kwargs):
    definition=definition or {}
    cfg=config or AmbiLAGEConfig()
    cfg=apply_surrogate_overrides(cfg,surrogate_set,surrogate_config)
    overrides=dict(allow_tf32=allow_tf32,seed=seed,device=device,smoke_test=smoke_test or cfg.smoke_test)
    if steps is not None:
        overrides['steps']=steps
    if training_manifest is not None:
        overrides['training_manifest']=str(training_manifest)
    cfg=replace(cfg,**overrides);cfg.validate()
    data=definition.get('data',{})
    explicit_print = list(definition.get('print_regions') or [])
    src_meta = definition.get('source') or {}
    if isinstance(src_meta, dict) and src_meta.get('text_regions'):
        for reg in src_meta['text_regions']:
            if reg not in explicit_print:
                explicit_print.append(reg)
    return AmbiLAGEPipeline(cfg,device).execute(source,target,output_dir,
        target_text=target_text or (data.get('training_queries') or [None])[0],
        source_public_key=definition.get('source_public_key') or data.get('source_public_key'),
        source_title=data.get('source_title'),source_category=definition.get('source_category') or data.get('source_category'),
        source_boxes=definition.get('source_boxes'),target_boxes=definition.get('target_boxes'),
        print_regions=explicit_print or None,definition=definition,**kwargs)


def run_attack(source,target,output_dir,*,target_text=None,steps=None,device='cuda',ambilage_config=None,
               allow_tf32=True,seed=42,definition=None,smoke_test=False,**kwargs):
    if isinstance(ambilage_config,(str,Path)):
        import json
        ambilage_config=json.loads(Path(ambilage_config).read_text(encoding='utf-8'))
    if isinstance(ambilage_config,dict):
        ambilage_config=AmbiLAGEConfig.from_dict(ambilage_config)
    return generate_ambilage(source,target,output_dir,definition=definition,config=ambilage_config,device=device,
        target_text=target_text,steps=steps,allow_tf32=allow_tf32,seed=seed,smoke_test=smoke_test,**kwargs)
