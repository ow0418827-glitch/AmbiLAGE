#!/usr/bin/env bash
set -euo pipefail

cd /content/AmbiLAGE
export PYTHONUNBUFFERED=1
PY=/content/AmbiLAGE/.venv/bin/python
DRIVE=/content/drive/MyDrive/AmbiLAGE_backup
IMAGES_OUT=/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_golden_band_9groups
RESULTS_OUT=/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_set2_9groups_results
LOG_DIR="$RESULTS_OUT/logs"
CFG_PATH=/content/AmbiLAGE/src/visual/config/ambilage.clip_l14.example.json
SYNC_SCRIPT=/content/AmbiLAGE/output/exp_20261121_eval_v4_golden_band/logs/sync_results_to_drive.sh
STATUS_FILE="$LOG_DIR/DONE_MS_V2_GEOM_9G_FULL.status"

mkdir -p "$LOG_DIR" "$IMAGES_OUT" "$RESULTS_OUT/eval" "$RESULTS_OUT/memory_vectors" "$RESULTS_OUT/encoder_vectors_9groups" "$RESULTS_OUT/images"
rm -f "$STATUS_FILE"

SYNC_PID=""
cleanup() {
  local rc=$?
  if [ -n "${SYNC_PID}" ]; then
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
  fi
  if [ "${rc}" -ne 0 ] && [ ! -f "${STATUS_FILE}" ]; then
    printf 'FAILED(rc=%s) %s\n' "${rc}" "$(date -Is)" > "${STATUS_FILE}"
  fi
  if [ -x "$SYNC_SCRIPT" ]; then
    bash "$SYNC_SCRIPT" || true
  fi
}
trap cleanup EXIT

(
  while true; do
    sleep 600
    if [ -x "$SYNC_SCRIPT" ]; then
      bash "$SYNC_SCRIPT" >/dev/null 2>&1 || true
    fi
  done
) </dev/null >/dev/null 2>&1 &
SYNC_PID=$!

echo "[STAGE A START] $(date -Is) Generating remaining 5 Set 2 groups using default ms_v2_geom ($CFG_PATH)..."
"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group 03_cookies,04_curtain,05_hair_extension \
  --method ambilage \
  --ambilage-config "$CFG_PATH" \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$IMAGES_OUT" \
  --device cuda > "$LOG_DIR/w1_3groups.log" 2>&1 &
WPID1=$!

"$PY" -u -X utf8 -B tools/run_experiment.py generate \
  --base golden_band_9groups \
  --group 08_speaker,09_toothbrush \
  --method ambilage \
  --ambilage-config "$CFG_PATH" \
  --training-bundles-root data/public/visual_inception_golden_band \
  --output "$IMAGES_OUT" \
  --device cuda > "$LOG_DIR/w2_2groups.log" 2>&1 &
WPID2=$!

WRC1=0
WRC2=0
wait "$WPID1" || WRC1=$?
wait "$WPID2" || WRC2=$?
echo "[STAGE A SHARDS DONE] $(date -Is) w1=$WRC1 w2=$WRC2"
if [ "$WRC1" -ne 0 ] || [ "$WRC2" -ne 0 ]; then
  echo "[ERROR] Generation worker failed (w1=$WRC1, w2=$WRC2)"
  exit 1
fi

echo "[STAGE B VERIFY] $(date -Is) Verifying all 9 ms_v2_geom FROZEN_IMAGE.json..."
"$PY" -u -X utf8 -B -c "
from pathlib import Path
from src.paths import resolve_evaluation_base
from src.experiments.visual_inception_category_evaluation import load_frozen_suite

base = resolve_evaluation_base('golden_band_9groups')
out = Path('$IMAGES_OUT')
suite = load_frozen_suite(out, method='ambilage', base=base)
assert len(suite) == 9, f'Expected 9 groups, got {len(suite)}'
for item in suite:
    g = item['group']
    meta = item['metadata']
    cfg = meta['config']
    tobj = meta['training_objective']
    gate = meta['final_naturalness']
    assert cfg['eot_enabled'] is False, f'{g} cfg eot_enabled={cfg[\"eot_enabled\"]}'
    assert cfg['diffusion_eot_enabled'] is False, f'{g} cfg diff_eot={cfg[\"diffusion_eot_enabled\"]}'
    assert cfg['multiscale_eot_enabled'] is True, f'{g} cfg ms_eot={cfg[\"multiscale_eot_enabled\"]}'
    assert cfg['multiscale_eot_profile'] == 'geom_v2', f'{g} cfg profile={cfg[\"multiscale_eot_profile\"]}'
    assert abs(float(cfg['multiscale_identity_weight_end']) - 0.35) < 1e-9, f'{g} cfg id_end={cfg[\"multiscale_identity_weight_end\"]}'
    assert tobj['eot_enabled'] is False, f'{g} obj eot_enabled={tobj[\"eot_enabled\"]}'
    assert tobj['diffusion_eot_enabled'] is False, f'{g} obj diff_eot={tobj[\"diffusion_eot_enabled\"]}'
    assert float(tobj['diffusion_weight']) == 0.0, f'{g} obj diff_w={tobj[\"diffusion_weight\"]}'
    assert tobj['multiscale_eot_enabled'] is True, f'{g} obj ms_eot={tobj[\"multiscale_eot_enabled\"]}'
    assert tobj['multiscale_eot_profile'] == 'geom_v2', f'{g} obj profile={tobj[\"multiscale_eot_profile\"]}'
    assert tobj['eot_paths'] == ['identity', 'multiscale_grid_breaking_geom_v2_resize_pad_flip_shift'], f'{g} eot_paths={tobj[\"eot_paths\"]}'
    assert gate.get('passed') is True, f'{g} naturalness passed={gate.get(\"passed\")}'
    print(f'[VERIFIED 9G FROZEN] {g}: ssim={gate[\"metrics\"][\"source_ssim\"]:.4f} mae={gate[\"metrics\"][\"source_mae_levels\"]:.2f} tv={gate[\"metrics\"][\"fg_chroma_tv_levels\"]:.4f}')
"

if [ -x "$SYNC_SCRIPT" ]; then
  bash "$SYNC_SCRIPT" || true
fi

echo "[STAGE C.1 PRECOMPUTE] $(date -Is) Encoding all 9 Set 2 ms_v2_geom images across 3 encoders..."
"$PY" -u -X utf8 -B tools/precompute_eval_vectors.py build \
  --store-dir /content/AmbiLAGE/output/cache/precomputed_eval_vectors \
  --images-root "$IMAGES_OUT" \
  --device cuda

echo "[STAGE C.2 COPY STORE] $(date -Is) Mirroring precomputed vector store into standalone $RESULTS_OUT/precomputed_eval_vectors..."
mkdir -p "$RESULTS_OUT/precomputed_eval_vectors"
rsync -rt --inplace /content/AmbiLAGE/output/cache/precomputed_eval_vectors/ "$RESULTS_OUT/precomputed_eval_vectors/"

echo "[STAGE C.3 EVAL 9G] $(date -Is) Evaluating all 9 groups (--group all --replacement-groups all) into $RESULTS_OUT/eval..."
"$PY" -u -X utf8 -B tools/fast_evaluate_panel.py eval-suites \
  --base golden_band_9groups \
  --group all \
  --replacement-groups all \
  --spec "ambilage_ms_v2_geom_set2:ambilage:$IMAGES_OUT" \
  --suites training_oracle_v4 overlap_30_v4 zero_overlap_v4 \
  --output-dir "$RESULTS_OUT/eval" \
  --memory-vectors-dir "$RESULTS_OUT/memory_vectors" \
  --precomputed-dir "$RESULTS_OUT/precomputed_eval_vectors" \
  --verify-mode in_process \
  --device cuda

echo "[STAGE C.4 VERIFY 9/9] Checking all 9 independent verification files <= 1e-7 & Zero-GPU precomputed check..."
"$PY" -u -X utf8 -B -c "
import json
from pathlib import Path
eval_dir = Path('$RESULTS_OUT/eval')
count = 0
for s in ('training_oracle_v4', 'overlap_30_v4', 'zero_overlap_v4'):
    for e in ('01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m'):
        vp = eval_dir / f'ambilage_ms_v2_geom_set2_{s}' / e / 'category_independent_verification.json'
        assert vp.is_file(), f'Missing {vp}'
        v = json.loads(vp.read_text(encoding='utf-8'))
        err = float(v['max_score_error'])
        assert err <= 1e-7, f'Score error > 1e-7 in {vp}: {err}'
        count += 1
        print(f'[VERIFIED {count}/9] {s}/{e}: max_score_error={err}')
"

"$PY" -u -X utf8 -B tools/precompute_eval_vectors.py verify \
  --store-dir "$RESULTS_OUT/precomputed_eval_vectors" \
  --eval-root "$RESULTS_OUT/eval"

echo "[STAGE C.5 EXPORT 9G VECTORS & IMAGES INDEX] Extracting per-encoder 9-group vectors into $RESULTS_OUT/encoder_vectors_9groups..."
"$PY" -u -X utf8 -B -c "
import json
import shutil
from pathlib import Path
import numpy as np

from src.experiments.visual_inception_category_evaluation import load_frozen_suite
from src.paths import resolve_evaluation_base
from src.visual.image_utils import sha256_file
from src.visual.precomputed_eval_store import ENCODER_SPEC_BY_SLUG, PrecomputedEncoderStore

base = resolve_evaluation_base('golden_band_9groups')
images_out = Path('$IMAGES_OUT')
results_out = Path('$RESULTS_OUT')
store_dir = results_out / 'precomputed_eval_vectors'
vec_out = results_out / 'encoder_vectors_9groups'
img_copy_dir = results_out / 'images'

suite = load_frozen_suite(images_out, method='ambilage', base=base)
groups = [item['group'] for item in suite]
index_rows = []
for item in suite:
    g = item['group']
    src_p = Path(item['metadata']['source_image']).resolve()
    mod_p = Path(item['image']).resolve()
    src_copy = img_copy_dir / f'{g}_source{src_p.suffix}'
    mod_copy = img_copy_dir / f'{g}_modified{mod_p.suffix}'
    shutil.copy2(src_p, src_copy)
    shutil.copy2(mod_p, mod_copy)
    index_rows.append({
        'group': g,
        'source_image': str(src_p),
        'source_sha256': sha256_file(src_p),
        'modified_image': str(mod_p),
        'modified_sha256': item['image_sha256'],
        'local_source_copy': str(src_copy),
        'local_modified_copy': str(mod_copy),
        'source_memory_id': item['source_memory_id'],
    })

(results_out / 'images_index.json').write_text(json.dumps(index_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

manifest = {'groups': groups, 'images_root': str(images_out), 'encoders': {}}
mod_shas = [r['modified_sha256'] for r in index_rows]
src_shas = [r['source_sha256'] for r in index_rows]
for slug, spec in ENCODER_SPEC_BY_SLUG.items():
    store = PrecomputedEncoderStore(slug, store_dir)
    mod_vecs = np.stack([store.get_image_vector_by_sha(s) for s in mod_shas], axis=0).astype(np.float32)
    src_vecs = np.stack([store.get_image_vector_by_sha(s) for s in src_shas], axis=0).astype(np.float32)
    assert np.isfinite(mod_vecs).all() and np.isfinite(src_vecs).all()
    mod_norms = np.linalg.norm(mod_vecs, axis=1)
    src_norms = np.linalg.norm(src_vecs, axis=1)
    assert np.allclose(mod_norms, 1.0, atol=1e-4) and np.allclose(src_norms, 1.0, atol=1e-4)
    npz_path = vec_out / f'{slug}.npz'
    np.savez(
        npz_path,
        groups=np.array(groups),
        modified_vectors=mod_vecs,
        clean_vectors=src_vecs,
        modified_sha256s=np.array(mod_shas),
        clean_sha256s=np.array(src_shas),
    )
    manifest['encoders'][slug] = {
        'label': spec['label'],
        'shape': list(mod_vecs.shape),
        'dtype': str(mod_vecs.dtype),
        'npz_file': str(npz_path),
    }
    print(f'[EXPORTED VECTORS] {slug}: modified={mod_vecs.shape}, clean={src_vecs.shape}, norm_range=[{mod_norms.min():.6f}, {mod_norms.max():.6f}]')

(vec_out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
"

echo "[STAGE D SUMMARY] $(date -Is) Building 9-group comparison summary & MiniMax-ready README..."
PYTHONPATH=/content/AmbiLAGE "$PY" -u -X utf8 -B "$LOG_DIR/summarize_9g_ms_v2_geom.py"

cat > "$RESULTS_OUT/README_MINIMAX_READY.md" << 'EOF'
# AmbiLAGE 默认接口 (`ms_v2_geom`) Set 2 九组完整产物与 MiniMax-M3 测试指南

## 目录结构（独立目录）

- `precomputed_eval_vectors/`：完整的三编码器（`01_clip_l14`、`02_openclip_bigG14`、`03_siglip_so400m`）+ `sbert_ghr_targets.json` 预计算向量库（涵盖全部 `golden_band_9groups` 基础库、9 组默认 `ms_v2_geom` 成图、以及 3 套评估查询集）。
- `encoder_vectors_9groups/`：按编码器独立导出的 9 组 `modified_vectors [9, D]` 与 `clean_vectors [9, D]`（`.npz` + `manifest.json`）。
- `images/` 与 `images_index.json`：Set 2 全部 9 组的原图、修改图本地副本及 SHA-256 索引（原始冻结目录为 `/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_golden_band_9groups`）。
- `eval/`：完整 9 组同时替库（`--group all --replacement-groups all`）在 3 套查询集 × 3 编码器上的 `scores.npz` 与 `evaluation.json`（均已通过 `max_score_error <= 1e-7` 独立核验）。
- `SUMMARY_9G_MS_V2_GEOM.md` / `.json`：9 组全量评测对比汇总报告。

## 一键启动 MiniMax-M3 零 GPU 下游测试命令示例

方式一（使用独立目录内置完整预计算向量库，无需加载任何检索编码器、纯 CPU）：

```bash
cd /content/AmbiLAGE
/content/AmbiLAGE/.venv/bin/python -u tools/evaluate_downstream_recommendation.py \
  --base golden_band_9groups \
  --spec "ambilage_ms_v2_geom_set2:ambilage:/content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_golden_band_9groups" \
  --precomputed-dir /content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_set2_9groups_results/precomputed_eval_vectors \
  --suites zero_overlap_v4 \
  --encoder-slugs 02_openclip_bigG14 03_siglip_so400m 01_clip_l14 \
  --output-dir output/exp_20261121_downstream_minimax_m3_set2_ms_v2_geom
```

方式二（直接复用本目录内已冻结的 `eval/` 检索排名与 `scores.npz`）：

```bash
cd /content/AmbiLAGE
/content/AmbiLAGE/.venv/bin/python -u tools/evaluate_downstream_recommendation.py \
  --base golden_band_9groups \
  --eval-root /content/AmbiLAGE/output/exp_20261121_ambilage_ms_v2_geom_set2_9groups_results/eval \
  --tags ambilage_ms_v2_geom_set2 \
  --suites zero_overlap_v4 \
  --encoder-slugs 02_openclip_bigG14 03_siglip_so400m 01_clip_l14 \
  --output-dir output/exp_20261121_downstream_minimax_m3_set2_ms_v2_geom
```
EOF

if [ -x "$SYNC_SCRIPT" ]; then
  bash "$SYNC_SCRIPT" || true
fi

printf 'SUCCESS %s\n' "$(date -Is)" > "${STATUS_FILE}"
echo "[MS_V2_GEOM 9G FULL SUITE COMPLETED SUCCESSFULLY] $(date -Is)"
