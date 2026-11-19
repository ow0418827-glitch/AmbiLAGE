"""Pre-warm all 36 non-none defended images into stage2_defenses/defended_cache once.

Covers all 189 clean shared pool images + 9 frozen AmbiLAGE images + 9 frozen
Visual Inception images (= 207 unique input images) across blur/jpeg/resize/SD1.5/SD2.1,
grouped by purifier key so SD1.5 and SD2.1 are each loaded onto GPU only once.
"""
from __future__ import annotations

import time
from pathlib import Path

from src.agent.memory.shared_pool import load_shared_pool
from src.experiments.defenses import Defender, conditions_for_mode
from src.experiments.visual_inception_category_evaluation import load_frozen_suite
from src.paths import OUTPUT, SHARED_MEMORY
from src.visual.visual_retrieval import resolve_image


def main():
    stage2_root = OUTPUT / 'eval_9groups_20261119' / 'stage2_defenses'
    cache_dir = stage2_root / 'defended_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries = load_shared_pool(SHARED_MEMORY)
    amb_suite = load_frozen_suite(OUTPUT / 'ambilage_single_clip_l14_9groups_20261118', method='ambilage')
    vi_suite = load_frozen_suite(OUTPUT / 'vi_single_clip_l14_9groups_20261118', method='visual_inception')

    seen = set()
    unique_images: list[Path] = []
    for p in (
        [resolve_image(e, SHARED_MEMORY) for e in entries]
        + [Path(it['image']).resolve() for it in amb_suite]
        + [Path(it['image']).resolve() for it in vi_suite]
    ):
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique_images.append(Path(rp))

    print(f'[PREWARM] unique_images={len(unique_images)} (expect 207)', flush=True)
    assert len(unique_images) == 207, f'Unexpected unique image count: {len(unique_images)}'

    all_specs = [s for s in conditions_for_mode('panel_with_sd21') if s['kind'] != 'none']
    ordered_specs = (
        [s for s in all_specs if s['kind'] in ('blur', 'jpeg', 'resize')]
        + [s for s in all_specs if s['kind'] in ('vae', 'diffusion') and s.get('purifier_key') is None]
        + [s for s in all_specs if s.get('purifier_key') == 'eval_purifier_sd21']
    )
    assert len(ordered_specs) == len(all_specs) == 36

    defender = Defender(cache_dir, device='cuda')
    t_total = time.perf_counter()
    try:
        for idx, spec in enumerate(ordered_specs, 1):
            t0 = time.perf_counter()
            for img_path in unique_images:
                defender(img_path, spec)
            dt = time.perf_counter() - t0
            print(
                f"[PREWARM] ({idx:02d}/36) {spec['label']} kind={spec['kind']} "
                f"purifier={spec.get('purifier_key') or 'default'} done in {dt:.1f}s",
                flush=True,
            )
    finally:
        defender.close()
    print(f'[PREWARM COMPLETE] total={time.perf_counter() - t_total:.1f}s', flush=True)


if __name__ == '__main__':
    main()
