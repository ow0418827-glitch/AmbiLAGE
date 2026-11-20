#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path('/content/AmbiLAGE').resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.experiments.base_workflow import EVALUATION_BASE, public_group
from src.experiments.pipeline import project_path, read_json
from src.visual._ambilage.config import AmbiLAGEConfig
from src.visual._ambilage.pipeline import STAGE3_STATE_FILENAME, AmbiLAGEPipeline, _stage3_signature
from src.visual._visual_inception.targets import prepare_semantic_targets
from src.visual._visual_inception.training_data import load_training_bundle
from src.visual.image_utils import read_rgb, sha256_file, write_json
from src.visual.training_surrogates import apply_surrogate_overrides

EXP_DIR = REPO / 'output' / 'exp_20261120_grad_protect'
CFG_PATH = EXP_DIR / 'configs' / 'ambilage_grad_protect.json'
BASELINE_GEN_ROOT = REPO / 'output' / 'ambilage_single_clip_l14_9groups_20261118'
BUNDLES_ROOT = REPO / 'data' / 'public' / 'visual_inception'
GROUPS = ('02_chair', '05_hair_extension', '09_toothbrush')

STAGE3_FILES = (
    'source_foreground.png',
    'target_foreground.png',
    'print_core.png',
    'latent_edit_mask.png',
    'generative_anchor.png',
)
STAGE3_DIRS = (
    'color_harmonization',
    'foreground_statistics',
)
STALE_GEN_FILES = (
    'FROZEN_IMAGE.json',
    'PUBLIC_RUN.json',
    'frozen.png',
    'generation/ready.json',
    'generation/metadata.json',
    'generation/generation_result.json',
    'generation/adversarial.png',
    'generation/naturalness_candidate.png',
    'generation/naturalness_gate.json',
    'generation/checkpoint_latest.pt',
    'generation/training_surrogates.json',
    'generation/FAILED.json',
    f'generation/{STAGE3_STATE_FILENAME}',
)


def main() -> None:
    raw_cfg = read_json(CFG_PATH)
    base_cfg = AmbiLAGEConfig.from_dict(raw_cfg)

    for eval_dir in sorted((EXP_DIR / 'eval_stage1').glob('gp_pilot3_*')):
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
            print(f'[CLEAN] removed {eval_dir}')
    env4_gp = EXP_DIR / 'eval_env4' / 'gp_pilot3'
    if env4_gp.exists():
        shutil.rmtree(env4_gp)
        print(f'[CLEAN] removed {env4_gp}')

    for grp in GROUPS:
        grp_gen_root = EXP_DIR / 'gen' / grp / 'ambilage'
        gen_dir = grp_gen_root / 'generation'
        gen_dir.mkdir(parents=True, exist_ok=True)

        for rel in STALE_GEN_FILES:
            p = grp_gen_root / rel
            p.unlink(missing_ok=True)
        stale_cache = gen_dir / 'candidate_cache'
        if stale_cache.exists():
            shutil.rmtree(stale_cache)

        baseline_gen_dir = BASELINE_GEN_ROOT / grp / 'ambilage' / 'generation'
        baseline_ready = json.loads((baseline_gen_dir / 'ready.json').read_text(encoding='utf-8'))

        for fname in STAGE3_FILES:
            src_f = baseline_gen_dir / fname
            if not src_f.is_file():
                raise FileNotFoundError(f'Missing baseline Stage 3 artifact: {src_f}')
            shutil.copy2(src_f, gen_dir / fname)

        for dname in STAGE3_DIRS:
            src_d = baseline_gen_dir / dname
            dst_d = gen_dir / dname
            if dst_d.exists():
                shutil.rmtree(dst_d)
            if src_d.is_dir():
                shutil.copytree(src_d, dst_d)

        manifest_path = project_path(BUNDLES_ROOT / grp / 'training_bundle.json')
        cfg = replace(base_cfg, training_manifest=str(manifest_path))
        cfg = apply_surrogate_overrides(cfg, None, None)
        cfg = replace(cfg, allow_tf32=True, seed=42, device='cuda', smoke_test=False or cfg.smoke_test)
        cfg.validate()

        inputs = public_group(EVALUATION_BASE, grp)
        source = Path(inputs['source']['image']).resolve()
        target = Path(inputs['target']['image']).resolve()
        definition = dict(inputs['definition'], target_product_features=inputs['target_features'])
        data = definition.get('data', {})

        explicit_print = list(definition.get('print_regions') or [])
        src_meta = definition.get('source') or {}
        if isinstance(src_meta, dict) and src_meta.get('text_regions'):
            for reg in src_meta['text_regions']:
                if reg not in explicit_print:
                    explicit_print.append(reg)

        target_text = inputs['target_text'] or (data.get('training_queries') or [None])[0]
        source_public_key = definition.get('source_public_key') or data.get('source_public_key')
        source_category = definition.get('source_category') or data.get('source_category')
        source_boxes = definition.get('source_boxes')
        target_boxes = definition.get('target_boxes')
        print_regions = explicit_print or None

        source_rgb = read_rgb(source)
        roi = np.asarray(Image.open(gen_dir / 'print_core.png').convert('L')) > 0
        anchor = np.asarray(Image.open(gen_dir / 'generative_anchor.png').convert('RGB'))
        if anchor.shape != source_rgb.shape or not np.array_equal(anchor[roi], source_rgb[roi]):
            raise ValueError(f'{grp}: Stage 3 anchor ROI mismatch against source image')

        semantic_targets = prepare_semantic_targets(source, target, target_text=target_text, definition=definition)
        training_bundle = load_training_bundle(
            manifest_path,
            source=source,
            target=target,
            public_semantic=semantic_targets,
            require_production=not cfg.smoke_test,
        )
        manifest_sha = training_bundle['manifest_sha256']

        pipeline_obj = AmbiLAGEPipeline(cfg, 'cuda')
        surrogate_specs_sha = pipeline_obj.surrogate_selection.audit()['specs_sha256']
        stage3_sig = _stage3_signature(
            sha256_file(source),
            sha256_file(target),
            target_text,
            source_public_key,
            source_category,
            cfg,
            source_boxes,
            target_boxes,
            print_regions,
            surrogate_specs_sha,
            training_manifest_sha256=manifest_sha,
        )

        write_json(gen_dir / STAGE3_STATE_FILENAME, {
            'signature': stage3_sig,
            'plan': baseline_ready['planning'],
            'ocr': baseline_ready['ocr'],
            'source_seg': baseline_ready['segmentation']['source'],
            'target_seg': baseline_ready['segmentation']['target'],
            'color_stats': baseline_ready['color_statistics'],
            'harmonization': baseline_ready['harmonization'],
            'anchor_audit': baseline_ready['anchor_generation'],
            'stage3_elapsed_seconds': 0.0,
        })
        print(f'[READY] {grp}: stage3_state.json sig={stage3_sig[:12]} manifest_sha={manifest_sha[:12]}')


if __name__ == '__main__':
    main()
