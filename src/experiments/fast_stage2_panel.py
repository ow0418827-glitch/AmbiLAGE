"""Encoder-grouped Stage 2 panel runner with merged fresh isolated subprocess verification.

Instead of reloading 10 GB evaluation models and running 6 separate `_verify-worker`
passes over identical defended images across 3 query suites × 2 methods, this module:
  1. Iterates encoder-outer (`01_clip_l14` -> `02_openclip_bigG14` -> `03_siglip_so400m`).
  2. Pre-warms unique assets in deterministic SHA256 order via `BatchedFreshCachedEncoder`.
  3. Runs exact unmodified `shared_evaluation.evaluate(...)` across pending panels.
  4. Spawns ONE fresh isolated `_verify-encoder` subprocess per encoder (without reading
     any `memory_vectors` cache) to re-encode unique assets in RAM and verify every group's
     complete `scores.npz` via unmodified `shared_evaluation.verify(...)` (`<= 1e-7`).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from src.experiments import shared_evaluation
from src.experiments.defenses import conditions_for_mode
from src.experiments.visual_inception_category_evaluation import (
    DEFAULT_KEY_MODE,
    _format_summary_cells,
    _require,
    _resolve_transfer_panel_encoders,
    build_category_summary,
    load_frozen_suite,
    report_category_suite,
    resolve_query_root,
    validate_category_query_suite,
)
from src.paths import EVALUATION_BASE, OUTPUT, PROJECT, VISUAL_CONFIGS
from src.visual.audit import digest, freeze_json, read_json
from src.visual.encoders.eval_clip import BatchedFreshCachedEncoder, make_standalone_eval_encoder
from src.visual.image_utils import sha256_file
from src.visual.optimization import normalize_method
from src.visual.visual_retrieval import resolve_image


VERIFY_MODES = ('subprocess', 'in_process')


def _resolve_job_groups(job: dict) -> tuple[list[str] | None, str]:
    raw_groups = job.get('groups')
    replacement_groups = job.get('replacement_groups', 'all')
    if replacement_groups not in ('all', 'selected'):
        raise ValueError('替换组范围只能为 all 或 selected')
    if raw_groups is None:
        requested_groups = None
    elif isinstance(raw_groups, str):
        requested_groups = [g.strip() for g in raw_groups.split(',') if g.strip()]
    else:
        requested_groups = [str(g).strip() for g in raw_groups if str(g).strip()]
    if requested_groups is not None and requested_groups == ['all']:
        requested_groups = None
    return requested_groups, replacement_groups


def default_stage2_jobs(stage2_dir: Path | None = None) -> list[dict]:
    root = Path(stage2_dir or (OUTPUT / 'eval_9groups_20261119' / 'stage2_defenses')).resolve()
    amb_imgs = str((OUTPUT / 'ambilage_single_clip_l14_9groups_20261118').resolve())
    vi_imgs = str((OUTPUT / 'vi_single_clip_l14_9groups_20261118').resolve())
    return [
        dict(name='ambilage_training_oracle', images_root=amb_imgs, method='ambilage',
             query_suite='training_oracle', output=str(root / 'ambilage_training_oracle')),
        dict(name='vi_training_oracle', images_root=vi_imgs, method='visual_inception',
             query_suite='training_oracle', output=str(root / 'vi_training_oracle')),
        dict(name='ambilage_overlap_30', images_root=amb_imgs, method='ambilage',
             query_suite='overlap_30', output=str(root / 'ambilage_overlap_30')),
        dict(name='vi_overlap_30', images_root=vi_imgs, method='visual_inception',
             query_suite='overlap_30', output=str(root / 'vi_overlap_30')),
        dict(name='ambilage_zero_overlap', images_root=amb_imgs, method='ambilage',
             query_suite='zero_overlap', output=str(root / 'ambilage_zero_overlap')),
        dict(name='vi_zero_overlap', images_root=vi_imgs, method='visual_inception',
             query_suite='zero_overlap', output=str(root / 'vi_zero_overlap')),
    ]


def _collect_bank_assets(
    banks: dict[str, list[dict]],
    specs: list[dict],
    pool_path: Path,
    defender,
    bank_root: Path,
    *,
    image_sha_fn,
    method_bank_cache: dict,
    assets_out: dict[str, Path],
) -> dict[tuple[str, str], tuple[Path, list[dict], str]]:
    """Build (or copy from in-RAM method bank memo) all defended bank JSONs and collect unique images."""
    built: dict[tuple[str, str], tuple[Path, list[dict], str]] = {}
    for spec in specs:
        label = spec['label']
        for name, original_bank in banks.items():
            bank_file = (bank_root / 'banks' / label / f'{name}.json').resolve()
            source_shas = [(e.get('RetrievalImage') or {}).get('SHA256') for e in original_bank]
            cache_key = (name, label, tuple(source_shas))
            cached = method_bank_cache.get(cache_key)
            if cached is None:
                bank = copy.deepcopy(original_bank)
                asset_pairs: list[tuple[Path, str]] = []
                for entry in bank:
                    original = resolve_image(entry, pool_path)
                    if spec['kind'] != 'none':
                        image = defender(original, spec)
                        img_sha = image_sha_fn(image)
                        img_path = image.resolve()
                        entry['RetrievalImage'] = dict(
                            ASIN=entry['FinalASIN'],
                            LocalPath=str(img_path),
                            SHA256=img_sha,
                        )
                    else:
                        img_path = original
                        rec = (
                            entry.get('RetrievalImage')
                            if entry.get('RetrievalImage') is not None
                            else entry.get('PrimaryImage')
                        )
                        img_sha = rec.get('SHA256') or image_sha_fn(img_path)
                    asset_pairs.append((img_path, img_sha))
                method_bank_cache[cache_key] = (bank, asset_pairs)
            else:
                bank, asset_pairs = cached

            freeze_json(bank_file, bank)
            bank_sha = sha256_file(bank_file)
            built[(label, name)] = (bank_file, bank, bank_sha)
            for img_path, img_sha in asset_pairs:
                if img_sha not in assets_out:
                    assets_out[img_sha] = img_path
    return built


def _prepare_panel_encoder_plan(
    job: dict,
    enc_spec: dict,
    *,
    base: Path,
    defense_mode: str,
    key_mode: str,
    verify_training_snapshots: bool,
    require_surrogates: bool = True,
    frozen_suite_cache: dict | None = None,
    validation_cache: dict | None = None,
) -> dict:
    """Validate suite, write category_query_validation.json & category_evaluation_plan.json."""
    method = normalize_method(job['method'])
    effective_key_mode = shared_evaluation.default_key_mode_for_method(method) if key_mode is None else key_mode
    key_image_weight = 1.0 if effective_key_mode == 'image' else 0.5
    query_suite = job.get('query_suite', 'training_oracle')
    resolved_query_root = resolve_query_root(job.get('query_root'), query_suite)
    panel_out = Path(job['output']).resolve()
    sub_out = (panel_out / enc_spec['slug']).resolve()
    _require(sub_out.is_relative_to(Path(OUTPUT).resolve()), '评估结果必须保存在当前项目 output 目录内')

    requested_groups, replacement_groups = _resolve_job_groups(job)
    selected_groups_arg = requested_groups if replacement_groups == 'selected' else None
    suite_key = (
        str(Path(job['images_root']).resolve()),
        str(base),
        method,
        tuple(selected_groups_arg) if selected_groups_arg is not None else None,
    )
    if frozen_suite_cache is not None and suite_key in frozen_suite_cache:
        frozen_suite = frozen_suite_cache[suite_key]
    else:
        if replacement_groups == 'selected' and requested_groups is None:
            raise ValueError('replacement_groups=selected 时必须显式指定至少两个分组')
        frozen_suite = load_frozen_suite(
            job['images_root'],
            base=base,
            method=method,
            selected_groups=selected_groups_arg,
        )
        if frozen_suite_cache is not None:
            frozen_suite_cache[suite_key] = frozen_suite

    if replacement_groups == 'selected' or requested_groups is None:
        eval_items = frozen_suite
    else:
        suite_map = {it['group']: it for it in frozen_suite}
        unknown = [g for g in requested_groups if g not in suite_map]
        _require(not unknown, '未知评估分组：' + ', '.join(unknown))
        eval_items = [suite_map[g] for g in requested_groups]

    val_key = (str(resolved_query_root), query_suite, str(base), bool(verify_training_snapshots))
    if validation_cache is not None and val_key in validation_cache:
        validation = validation_cache[val_key]
    else:
        validation = validate_category_query_suite(
            resolved_query_root,
            query_suite=query_suite,
            base=base,
            verify_training_snapshots=verify_training_snapshots,
        )
        if validation_cache is not None:
            validation_cache[val_key] = validation

    base_manifest = read_json(base / 'manifest.json')
    shared_cfg = base_manifest.get('shared_memory', {})
    pool_rel = shared_cfg.get('pool_file')
    _require(isinstance(pool_rel, str) and bool(pool_rel.strip()), '基础包缺少共享记忆库路径配置')
    pool_path = (base / pool_rel).resolve()
    _require(pool_path.is_relative_to(base) and pool_path.is_file(), '基础包共享记忆库路径无效')
    if shared_cfg.get('pool_sha256') is not None:
        _require(sha256_file(pool_path) == shared_cfg['pool_sha256'], '基础包共享池摘要不符')
    pool_doc = read_json(pool_path)
    pool_count = len(pool_doc.get('entries', []))
    _require(pool_count > 0, '共享记忆库为空')

    selected_conditions = conditions_for_mode(defense_mode)
    effective_encoder_config = Path(enc_spec['encoder_config']).resolve()
    sub_out.mkdir(parents=True, exist_ok=True)
    validation_path = sub_out / 'category_query_validation.json'
    freeze_json(validation_path, validation)

    plan = dict(
        schema_version=1,
        protocol=validation['protocol'],
        query_suite=validation['query_suite'],
        method=method,
        query_mode='text',
        key_mode=effective_key_mode,
        key_image_weight=key_image_weight,
        defense_mode=defense_mode,
        selected_conditions=selected_conditions,
        base_path=str(base),
        base_manifest_sha256=sha256_file(base / 'manifest.json'),
        query_root=str(resolved_query_root),
        query_root_manifest_sha256=sha256_file(resolved_query_root / 'manifest.json'),
        query_validation_sha256=sha256_file(validation_path),
        pool_path=str(pool_path),
        pool_sha256=sha256_file(pool_path),
        pool_count=pool_count,
        encoder_config=str(effective_encoder_config),
        encoder_config_sha256=sha256_file(effective_encoder_config),
        groups=[item['group'] for item in eval_items],
        simultaneous_replacement_count=len(frozen_suite),
        shared_pool_scope=(
            'all_entries_simultaneous_nine_group_replacement'
            if len(frozen_suite) == 9
            else f'simultaneous_{len(frozen_suite)}_group_replacement'
        ),
        shopping_allowed=False,
        recommendation_executed=False,
        frozen_images={
            item['group']: dict(
                image=item['image'],
                image_sha256=item['image_sha256'],
                source_memory_id=item['source_memory_id'],
                source_id=item['source']['id'],
                target_id=item['target']['id'],
                requested_steps=item['requested_steps'],
                objective_kind=item.get('metadata', {}).get('training_objective', {}).get('kind'),
            )
            for item in eval_items
        },
    )
    freeze_json(sub_out / 'category_evaluation_plan.json', plan)
    training_context = shared_evaluation._frozen_training_context(
        frozen_suite[0], frozen_suite, required=require_surrogates
    )
    actual_surrogates = shared_evaluation._context_training_identities(training_context)
    return dict(
        job=job,
        sub_out=sub_out,
        method=method,
        key_mode=effective_key_mode,
        pool_path=pool_path,
        frozen_suite=frozen_suite,
        eval_items=eval_items,
        validation=validation,
        plan=plan,
        selected_conditions=selected_conditions,
        encoder_config=effective_encoder_config,
        actual_surrogates=actual_surrogates,
    )


def verify_encoder_worker(
    encoder_slug: str,
    panel_sub_dirs: list[Path],
    *,
    batch_size: int = 16,
    text_batch_size: int = 64,
    num_workers: int = 4,
    device: str = 'cuda',
    encoder_factory=None,
    shared_base_encoder=None,
    isolated_process: bool = True,
) -> dict[str, dict]:
    """Verify multiple evaluated panel subdirs for ONE encoder with a single fresh encoder load."""
    sub_dirs = [Path(p).resolve() for p in panel_sub_dirs]
    if not sub_dirs:
        return {}

    plans: dict[Path, dict] = {}
    pool_path_str = None
    enc_config_str = None
    expected_surrogates = None
    bank_cache: dict[tuple[str, str], list[dict]] = {}
    union_assets: dict[str, Path] = {}
    union_texts: set[str] = set()
    path_sha_cache: dict[tuple[str, int, int], str] = {}

    def _cached_file_sha(p: Path) -> str:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        val = path_sha_cache.get(key)
        if val is None:
            val = sha256_file(p)
            path_sha_cache[key] = val
        return val

    for sub_out in sub_dirs:
        plan = read_json(sub_out / 'category_evaluation_plan.json')
        plans[sub_out] = plan
        val_path = sub_out / 'category_query_validation.json'
        _require(sha256_file(val_path) == plan['query_validation_sha256'], f'{sub_out} 查询校验凭证已改变')
        _require(sha256_file(plan['pool_path']) == plan['pool_sha256'], f'{sub_out} 基础共享池在评估后改变')
        _require(
            sha256_file(Path(plan['query_root']) / 'manifest.json') == plan['query_root_manifest_sha256'],
            f'{sub_out} 测试查询总清单在评估后改变',
        )
        first_group_plan = read_json(sub_out / 'groups' / plan['groups'][0] / 'evaluation_plan.json')
        actual_surrogates = shared_evaluation._plan_training_identities(first_group_plan)
        if expected_surrogates is None:
            expected_surrogates = actual_surrogates
            enc_config_str = first_group_plan.get('encoder_config')
            pool_path_str = plan['pool_path']
        else:
            _require(actual_surrogates == expected_surrogates, f'{sub_out} 冻结训练代理与共享复算不一致')
            _require(first_group_plan.get('encoder_config') == enc_config_str, f'{sub_out} 编码器配置不一致')
            _require(plan['pool_path'] == pool_path_str, f'{sub_out} 共享池路径不一致')

        for g_idx, group in enumerate(plan['groups']):
            g_plan = first_group_plan if g_idx == 0 else read_json(sub_out / 'groups' / group / 'evaluation_plan.json')
            for q in g_plan.get('queries', []):
                union_texts.add(str(q.get('instruction') or ' '))

        # Pre-validate bank SHA256 once per unique bank file & collect all unique assets to encode in batch.
        sample_rows = read_json(sub_out / 'groups' / plan['groups'][0] / 'evaluation.json')
        pool_path = Path(plan['pool_path'])
        for row in sample_rows:
            bpath = str(Path(row['bank_path']).resolve())
            bkey = (bpath, row['bank_sha256'])
            if bkey not in bank_cache:
                _require(sha256_file(bpath) == row['bank_sha256'], f'评估副本改变：{bpath}')
                bank_data = read_json(bpath)
                bank_cache[bkey] = bank_data
            else:
                bank_data = bank_cache[bkey]
            for entry in bank_data:
                rec = entry.get('RetrievalImage') if entry.get('RetrievalImage') is not None else entry.get('PrimaryImage')
                if not rec or not rec.get('LocalPath'):
                    continue
                img_path = Path(rec['LocalPath'])
                if not img_path.is_absolute():
                    img_path = pool_path.parent / img_path
                img_path = img_path.resolve()
                actual_img_sha = _cached_file_sha(img_path)
                if rec.get('SHA256'):
                    _require(actual_img_sha == rec['SHA256'], f'复算前图像摘要不符：{img_path}')
                if actual_img_sha not in union_assets:
                    union_assets[actual_img_sha] = img_path

    sorted_assets = sorted(union_assets.items(), key=lambda kv: kv[0])
    owns_encoder = False
    if shared_base_encoder is not None:
        base_encoder = shared_base_encoder
    elif encoder_factory is not None:
        base_encoder = encoder_factory()
        owns_encoder = True
    else:
        base_encoder = make_standalone_eval_encoder(
            device=device,
            config_path=enc_config_str,
            training_surrogates=expected_surrogates,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        owns_encoder = True
    fresh_wrapper = BatchedFreshCachedEncoder(base_encoder)
    shared_pool = shared_evaluation.load_shared_pool(pool_path_str)
    bank_digest_cache: dict = {}

    results: dict[str, dict] = {}
    try:
        n_encoded = fresh_wrapper.prewarm_images(
            [(p, sha) for sha, p in sorted_assets],
            batch_size=batch_size,
        )
        n_texts = fresh_wrapper.prewarm_texts(union_texts, batch_size=text_batch_size)
        print(
            f'[VERIFY {encoder_slug}] prewarmed {n_encoded} unique images (batch={batch_size}), '
            f'{n_texts} unique texts (text_batch={text_batch_size})',
            flush=True,
        )

        for sub_out in sub_dirs:
            plan = plans[sub_out]
            group_verifications = {}
            total_rankings = 0
            total_scores = 0
            max_error = 0.0
            for group in plan['groups']:
                group_dir = sub_out / 'groups' / group
                verified = shared_evaluation.verify(
                    group_dir,
                    encoder_factory=lambda w=fresh_wrapper: w,
                    device=device,
                    isolated_process=isolated_process,
                    _bank_cache=bank_cache,
                    _path_sha_cache=path_sha_cache,
                    _bank_digest_cache=bank_digest_cache,
                    _shared_pool=shared_pool,
                )
                _require(
                    verified.get('status') == 'passed' and verified.get('all_candidate_scores_checked') is True,
                    f'{sub_out.parent.name}/{encoder_slug}/{group} 独立复算未通过',
                )
                total_rankings += verified['rankings_recomputed']
                total_scores += verified['candidate_scores_recomputed']
                max_error = max(max_error, float(verified['max_score_error']))
                group_verifications[group] = dict(
                    rankings_recomputed=verified['rankings_recomputed'],
                    candidate_scores_recomputed=verified['candidate_scores_recomputed'],
                    max_score_error=verified['max_score_error'],
                    evaluated_file_sha256=verified['evaluated_file_sha256'],
                    scores_sha256=verified['scores_sha256'],
                )

            rebuilt_summary = build_category_summary(sub_out)
            saved_summary = read_json(sub_out / 'category_retrieval_summary.json')
            _require(rebuilt_summary == saved_summary, f'{sub_out} 汇总与逐组评估结果不一致')
            ver_result = dict(
                schema_version=1,
                protocol=plan['protocol'],
                query_suite=saved_summary.get('query_suite'),
                status='passed',
                groups_verified=len(plan['groups']),
                rankings_recomputed=total_rankings,
                candidate_scores_recomputed=total_scores,
                max_score_error=max_error,
                all_candidate_scores_checked=True,
                summary_sha256=sha256_file(sub_out / 'category_retrieval_summary.json'),
                groups=group_verifications,
            )
            freeze_json(sub_out / 'category_independent_verification.json', ver_result)
            results[str(sub_out)] = ver_result
            print(
                f'[VERIFIED] {sub_out.parent.name}/{encoder_slug}: '
                f'rankings={total_rankings} scores={total_scores} max_error={max_error:.2e}',
                flush=True,
            )
    finally:
        if owns_encoder:
            base_encoder.close()
    return results


def _write_transfer_panel_outputs(
    job: dict,
    *,
    base: Path = EVALUATION_BASE,
    key_mode: str,
    defense_mode: str,
    encoder_slugs: list[str] | tuple[str, ...] | None = None,
) -> dict:
    output = Path(job['output']).resolve()
    summary_file = output / 'transfer_panel_summary.json'
    if summary_file.is_file():
        return read_json(summary_file)

    method = normalize_method(job['method'])
    effective_key_mode = DEFAULT_KEY_MODE if key_mode is None else key_mode
    resolved_base = Path(job.get('base') or base).resolve()
    requested_groups, replacement_groups = _resolve_job_groups(job)
    frozen_suite = load_frozen_suite(
        job['images_root'],
        base=resolved_base,
        method=method,
        selected_groups=(requested_groups if replacement_groups == 'selected' else None),
    )
    panel_encoders, trained_names = _resolve_transfer_panel_encoders(frozen_suite)
    if encoder_slugs is not None:
        allowed = set(encoder_slugs)
        panel_encoders = [e for e in panel_encoders if e['slug'] in allowed]

    encoder_results = {}
    ordered_rows = []
    for spec in panel_encoders:
        sub_out = output / spec['slug']
        summary = read_json(sub_out / 'category_retrieval_summary.json')
        verification = read_json(sub_out / 'category_independent_verification.json')
        _require(verification.get('status') == 'passed', f"{job['name']}/{spec['slug']} 尚未通过独立复算")
        entry = dict(
            slug=spec['slug'],
            label=spec['label'],
            role=spec['role'],
            role_zh=spec['role_zh'],
            surrogate_pair=spec['surrogate_pair'],
            encoder_config=str(spec['encoder_config']),
            encoder_config_sha256=sha256_file(spec['encoder_config']),
            output_dir=str(sub_out),
            summary_path=str(sub_out / 'category_retrieval_summary.json'),
            report_path=str(sub_out / 'REPORT.md'),
            overall=summary['overall'],
            by_family=summary['by_family'],
            by_table_group=summary.get('by_table_group', {}),
            groups=summary['groups'],
            verification=verification,
        )
        encoder_results[spec['slug']] = entry
        ordered_rows.append(entry)

    first_summary = read_json(Path(ordered_rows[0]['summary_path']))
    report_path = output / 'TRANSFER_PANEL_REPORT.md'
    panel_summary = dict(
        schema_version=1,
        method=method,
        query_suite=first_summary['query_suite'],
        protocol=first_summary['protocol'],
        key_mode=effective_key_mode,
        key_image_weight=first_summary['key_image_weight'],
        defense_mode=defense_mode,
        group_count=first_summary['group_count'],
        total_queries=first_summary['total_queries'],
        pool_count=first_summary['pool_count'],
        simultaneous_replacement_count=first_summary['simultaneous_replacement_count'],
        trained_surrogates=trained_names,
        encoder_order=[r['slug'] for r in ordered_rows],
        encoders=encoder_results,
        report_path=str(report_path),
    )

    lines = [
        f"# 三编码器检索迁移评测报告（{method} / {effective_key_mode} / {first_summary['query_suite']}）",
        '',
        f"- 测试环境：`{first_summary['query_suite']}`，记忆库检索键：`{effective_key_mode}`（`key_image_weight={first_summary['key_image_weight']}`），防御：`{defense_mode}`。",
        f"- 覆盖规模：{first_summary['group_count']} 组 × {first_summary['queries_per_group']} 条 = {first_summary['total_queries']} 条查询；{first_summary['pool_count']} 条共享记忆库（{first_summary['simultaneous_replacement_count']} 组同时替换）。",
        f"- 冻结训练代理：`{', '.join(trained_names) if trained_names else 'unknown'}`。",
        '',
        '| 顺序 | 评估编码器 | 口径角色 | 原图前1 | 修改前1 | 前1增量(百分点) | 原图前3 | 修改前3 | 前3增量(百分点) | 原图前5 | 修改前5 | 前5增量(百分点) | 平均名次变化(原->改, 改善) |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for idx, entry in enumerate(ordered_rows, 1):
        cells = _format_summary_cells(entry['overall'])
        lines.append(f"| {idx} | {entry['label']} | {entry['role_zh']} | " + ' | '.join(cells) + ' |')

    lines += ['', '## 各编码器逐组明细（Top-1 / Top-3 / Top-5 / 平均名次）', '']
    group_list = list(first_summary['groups'])
    lines.append('| 组别 | 品类 | ' + ' | '.join(f"{e['label']} 前1/前3/前5/均名" for e in ordered_rows) + ' |')
    lines.append('|---|---|' + '|'.join('---:' for _ in ordered_rows) + '|')
    for group in group_list:
        cat = first_summary['groups'][group].get('category') or ''
        g_cells = []
        for e in ordered_rows:
            ov = e['groups'][group]['overall']
            t1, t3, t5 = ov['top1'], ov['top3'], ov['top5']
            g_cells.append(
                f"{t1['modified_hits']}/{ov['query_count']} ({t1['modified_rate']*100:.0f}%) / "
                f"{t3['modified_hits']}/{ov['query_count']} ({t3['modified_rate']*100:.0f}%) / "
                f"{t5['modified_hits']}/{ov['query_count']} ({t5['modified_rate']*100:.0f}%) / "
                f"{ov['mean_rank_clean']:.1f}->{ov['mean_rank_modified']:.1f}"
            )
        lines.append(f'| `{group}` | {cat} | ' + ' | '.join(g_cells) + ' |')

    family_list = list(first_summary.get('query_families', ()))
    family_labels_zh = first_summary.get('query_family_labels_zh', {})
    if family_list:
        lines += ['', '## 按查询子家族三编码器对比汇总（Top-3 / Top-5 / 均名）', '']
        lines.append('| 查询子家族 | ' + ' | '.join(f"{e['label']} 前3/前5/均名" for e in ordered_rows) + ' |')
        lines.append('|---|' + '|'.join('---:' for _ in ordered_rows) + '|')
        for family in family_list:
            fam_cells = []
            for e in ordered_rows:
                st = e['by_family'][family]
                t3, t5 = st['top3'], st['top5']
                fam_cells.append(
                    f"{t3['modified_hits']}/{st['query_count']} ({t3['modified_rate']*100:.1f}%) / "
                    f"{t5['modified_hits']}/{st['query_count']} ({t5['modified_rate']*100:.1f}%) / "
                    f"{st['mean_rank_clean']:.2f}->{st['mean_rank_modified']:.2f}"
                )
            fam_label = f"{family_labels_zh.get(family, family)} (`{family}`)"
            lines.append(f"| {fam_label} | " + ' | '.join(fam_cells) + ' |')

    table_titles = list(ordered_rows[0].get('by_table_group', {}))
    if len(table_titles) > 1:
        lines += ['', '## 各防御条件三编码器对比汇总（Top-1 / Top-3 / Top-5 / 均名）', '']
        lines.append('| 防御条件 | ' + ' | '.join(f"{e['label']} 前1/前3/前5/均名" for e in ordered_rows) + ' |')
        lines.append('|---|' + '|'.join('---:' for _ in ordered_rows) + '|')
        for title_row in table_titles:
            row_cells = []
            for e in ordered_rows:
                st = e['by_table_group'][title_row]['overall']
                t1, t3, t5 = st['top1'], st['top3'], st['top5']
                row_cells.append(
                    f"{t1['modified_rate']*100:.1f}% / {t3['modified_rate']*100:.1f}% / "
                    f"{t5['modified_rate']*100:.1f}% / {st['mean_rank_clean']:.2f}->{st['mean_rank_modified']:.2f}"
                )
            lines.append(f'| {title_row} | ' + ' | '.join(row_cells) + ' |')
    lines.append('')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    freeze_json(summary_file, panel_summary)
    return panel_summary


def run_encoder_grouped_panels(
    jobs: list[dict],
    *,
    base: Path = EVALUATION_BASE,
    key_mode: str = DEFAULT_KEY_MODE,
    defense_mode: str = 'panel_with_sd21',
    defended_cache_dir: Path,
    memory_vector_cache_dir: Path,
    batch_sizes: dict[str, int] | None = None,
    text_batch_size: int = 64,
    num_workers: int = 4,
    device: str = 'cuda',
    verify_training_snapshots: bool = True,
    encoder_factory=None,
    defense_factory=None,
    isolated_process: bool | None = None,
    verify_mode: str | None = None,
    encoder_slugs: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict]:
    base = Path(base).resolve()
    defended_root = Path(defended_cache_dir).resolve()
    vec_dir = Path(memory_vector_cache_dir).resolve()
    _require(defended_root.is_relative_to(Path(OUTPUT).resolve()), '防御缓存目录必须位于当前项目 output 目录内')
    _require(vec_dir.is_relative_to(Path(OUTPUT).resolve()), '共享向量缓存目录必须位于当前项目 output 目录内')

    if verify_mode is not None:
        if verify_mode not in VERIFY_MODES:
            raise ValueError('verify_mode 只能为 subprocess 或 in_process')
        effective_verify_mode = verify_mode
    elif isolated_process is not None:
        effective_verify_mode = 'subprocess' if isolated_process else 'in_process'
    else:
        effective_verify_mode = 'subprocess'
    use_isolated_flag = effective_verify_mode == 'subprocess'

    pending_jobs = [j for j in jobs if not (Path(j['output']) / 'transfer_panel_summary.json').is_file()]
    if not pending_jobs:
        print('[FAST PANEL] All jobs already have transfer_panel_summary.json', flush=True)
        return {
            j['name']: _write_transfer_panel_outputs(
                j, base=base, key_mode=key_mode, defense_mode=defense_mode, encoder_slugs=encoder_slugs
            )
            for j in jobs
        }

    frozen_suite_cache: dict = {}
    validation_cache: dict = {}
    validated_completion_cache: set = set()

    def _get_job_suite(job_obj: dict) -> list[dict]:
        req_g, rep_g = _resolve_job_groups(job_obj)
        sel = req_g if rep_g == 'selected' else None
        resolved_base = Path(job_obj.get('base') or base).resolve()
        s_key = (
            str(Path(job_obj['images_root']).resolve()),
            str(resolved_base),
            normalize_method(job_obj['method']),
            tuple(sel) if sel is not None else None,
        )
        if s_key not in frozen_suite_cache:
            if rep_g == 'selected' and req_g is None:
                raise ValueError('replacement_groups=selected 时必须显式指定至少两个分组')
            frozen_suite_cache[s_key] = load_frozen_suite(
                job_obj['images_root'],
                base=resolved_base,
                method=normalize_method(job_obj['method']),
                selected_groups=sel,
            )
        return frozen_suite_cache[s_key]

    job_encoders: dict[str, dict[str, dict]] = {}
    slug_order: list[str] = []
    for job in pending_jobs:
        p_encs, _ = _resolve_transfer_panel_encoders(_get_job_suite(job))
        job_encoders[job['name']] = {e['slug']: e for e in p_encs}
        for e in p_encs:
            if e['slug'] not in slug_order:
                slug_order.append(e['slug'])
    if encoder_slugs is not None:
        allowed = set(encoder_slugs)
        slug_order = [s for s in slug_order if s in allowed]

    for slug in slug_order:
        enc_bs = int((batch_sizes or {}).get(slug, 16))

        pending_contexts = []
        for job in pending_jobs:
            enc_spec = job_encoders[job['name']][slug]
            sub_out = Path(job['output']).resolve() / slug
            if (sub_out / 'category_independent_verification.json').is_file() and (sub_out / 'REPORT.md').is_file():
                print(f"[SKIP SUB-OUT] {job['name']}/{slug} already verified", flush=True)
                continue
            resolved_base = Path(job.get('base') or base).resolve()
            pending_contexts.append(
                _prepare_panel_encoder_plan(
                    job,
                    enc_spec,
                    base=resolved_base,
                    defense_mode=defense_mode,
                    key_mode=key_mode,
                    verify_training_snapshots=verify_training_snapshots,
                    require_surrogates=(encoder_factory is None),
                    frozen_suite_cache=frozen_suite_cache,
                    validation_cache=validation_cache,
                )
            )

        if not pending_contexts:
            continue

        grouped_contexts: list[list[dict]] = []
        group_index_by_key: dict[tuple, int] = {}
        for ctx in pending_contexts:
            cfg_key = (
                str(ctx['encoder_config']),
                digest(ctx['actual_surrogates']),
                str(ctx['pool_path']),
            )
            idx = group_index_by_key.get(cfg_key)
            if idx is None:
                group_index_by_key[cfg_key] = len(grouped_contexts)
                grouped_contexts.append([ctx])
            else:
                grouped_contexts[idx].append(ctx)

        for cg_idx, group_contexts in enumerate(grouped_contexts):
            expected_surrogates = group_contexts[0]['actual_surrogates']
            expected_cfg = group_contexts[0]['encoder_config']
            expected_pool = group_contexts[0]['pool_path']

            shared_pool = shared_evaluation.load_shared_pool(expected_pool)
            shared_defender = (
                defense_factory(defended_root, device)
                if defense_factory is not None
                else shared_evaluation.Defender(defended_root, device)
            )
            image_sha_fn = getattr(shared_defender, '_cached_sha256', None) or sha256_file
            method_bank_cache: dict = {}
            union_assets: dict[str, Path] = {}
            union_texts: set[str] = set()
            sub_dirs = [ctx['sub_out'] for ctx in group_contexts]
            run_subprocess_verify = False

            try:
                for ctx in group_contexts:
                    frozen_suite = ctx['frozen_suite']
                    item0 = frozen_suite[0]
                    replacement_inputs = [
                        dict(source=it['source'], image=it['image'], source_memory_id=it.get('source_memory_id'))
                        for it in frozen_suite
                    ]
                    banks, _ = shared_evaluation.evaluation_banks(
                        shared_pool,
                        ctx['pool_path'],
                        item0['source'],
                        item0['image'],
                        source_memory_id=item0.get('source_memory_id'),
                        replacements=replacement_inputs,
                    )
                    built = _collect_bank_assets(
                        banks,
                        ctx['selected_conditions'],
                        ctx['pool_path'],
                        shared_defender,
                        ctx['sub_out'] / 'shared_banks',
                        image_sha_fn=image_sha_fn,
                        method_bank_cache=method_bank_cache,
                        assets_out=union_assets,
                    )
                    ctx['prewarmed_banks'] = built

                    for item in ctx['eval_items']:
                        group_query_path = Path(ctx['validation']['groups'][item['group']]['query_manifest'])
                        queries = shared_evaluation.load_formal_queries(
                            group_query_path, item['target'], query_mode='text'
                        )
                        for q in queries:
                            union_texts.add(str(q['instruction'] or ' '))

                sorted_assets = sorted(union_assets.items(), key=lambda kv: kv[0])
                print(
                    f'\n=== [ENCODER {slug}] panels={[c["job"]["name"] for c in group_contexts]} '
                    f'unique_images={len(sorted_assets)} unique_texts={len(union_texts)} '
                    f'batch_size={enc_bs} text_batch_size={text_batch_size} verify={effective_verify_mode} ===',
                    flush=True,
                )

                base_encoder = (
                    encoder_factory()
                    if encoder_factory is not None
                    else make_standalone_eval_encoder(
                        device=device,
                        config_path=expected_cfg,
                        training_surrogates=expected_surrogates,
                        batch_size=enc_bs,
                        num_workers=num_workers,
                    )
                )
                shared_enc = BatchedFreshCachedEncoder(base_encoder)
                try:
                    shared_enc.prewarm_images([(p, sha) for sha, p in sorted_assets], batch_size=enc_bs)
                    shared_enc.prewarm_texts(union_texts, batch_size=text_batch_size)
                    path_sha_cache: dict = {}
                    bank_digest_cache: dict = {}
                    for ctx in group_contexts:
                        sub_out = ctx['sub_out']
                        bank_root = sub_out / 'shared_banks'
                        shared_bank_memo = {}
                        for item in ctx['eval_items']:
                            group = item['group']
                            group_eval_dir = sub_out / 'groups' / group
                            group_query_path = Path(ctx['validation']['groups'][group]['query_manifest'])
                            shared_evaluation.evaluate(
                                item,
                                group_eval_dir,
                                pool_path=ctx['pool_path'],
                                query_manifest=group_query_path,
                                source_memory_id=item['source_memory_id'],
                                device=device,
                                encoder_factory=encoder_factory,
                                defense_factory=defense_factory,
                                selected_conditions=ctx['selected_conditions'],
                                encoder_config=ctx['encoder_config'],
                                query_mode='text',
                                key_mode=ctx['key_mode'],
                                replacement_frozens=ctx['frozen_suite'],
                                bank_root=bank_root,
                                memory_vector_cache_dir=vec_dir,
                                defer_verification=True,
                                shared_encoder=shared_enc,
                                _shared_defender=shared_defender,
                                _shared_bank_memo=shared_bank_memo,
                                _shared_pool=shared_pool,
                                _path_sha_cache=path_sha_cache,
                                _bank_digest_cache=bank_digest_cache,
                                _validated_completion_cache=validated_completion_cache,
                            )
                        summary = build_category_summary(sub_out)
                        freeze_json(sub_out / 'category_retrieval_summary.json', summary)
                        print(f"[EVAL DONE] {ctx['job']['name']}/{slug}", flush=True)

                    if encoder_factory is None and effective_verify_mode == 'in_process':
                        verify_encoder_worker(
                            slug,
                            sub_dirs,
                            batch_size=enc_bs,
                            text_batch_size=text_batch_size,
                            num_workers=num_workers,
                            device=device,
                            shared_base_encoder=base_encoder,
                            isolated_process=False,
                        )
                    elif encoder_factory is not None:
                        verify_encoder_worker(
                            slug,
                            sub_dirs,
                            batch_size=enc_bs,
                            text_batch_size=text_batch_size,
                            num_workers=num_workers,
                            device=device,
                            encoder_factory=encoder_factory,
                            isolated_process=use_isolated_flag,
                        )
                    else:
                        run_subprocess_verify = True
                finally:
                    base_encoder.close()
            finally:
                if callable(getattr(shared_defender, 'close', None)):
                    shared_defender.close()

            if run_subprocess_verify:
                log_name = f'verify_{slug}.log' if len(grouped_contexts) == 1 else f'verify_{slug}_{cg_idx}.log'
                cmd = [
                    sys.executable,
                    '-B',
                    '-X',
                    'utf8',
                    '-m',
                    'src.experiments.fast_stage2_panel',
                    '_verify-encoder',
                    '--encoder-slug',
                    slug,
                    '--batch-size',
                    str(enc_bs),
                    '--text-batch-size',
                    str(text_batch_size),
                    '--num-workers',
                    str(num_workers),
                    '--device',
                    device,
                    '--panel-sub-dirs',
                    *(str(p) for p in sub_dirs),
                ]
                log_path = Path(sub_dirs[0]).parent.parent / 'logs' / log_name
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open('w', encoding='utf-8') as log_f:
                    proc = subprocess.run(
                        cmd,
                        cwd=PROJECT,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                    )
                if proc.returncode:
                    raise RuntimeError(f'合并独立进程复算失败 ({slug})，见日志：{log_path}')

            for ctx in group_contexts:
                for item in ctx['eval_items']:
                    shared_evaluation.report(ctx['sub_out'] / 'groups' / item['group'])
                report_category_suite(ctx['sub_out'])

    return {
        j['name']: _write_transfer_panel_outputs(
            j, base=base, key_mode=key_mode, defense_mode=defense_mode, encoder_slugs=encoder_slugs
        )
        for j in jobs
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    p_run = sub.add_parser('run')
    p_run.add_argument('--stage2-dir', type=Path, default=OUTPUT / 'eval_9groups_20261119' / 'stage2_defenses')
    p_run.add_argument('--jobs-json', type=Path, default=None)
    p_run.add_argument('--base', type=Path, default=EVALUATION_BASE)
    p_run.add_argument('--key-mode', choices=('image', 'joint'), default=DEFAULT_KEY_MODE)
    p_run.add_argument('--defense-mode', default='panel_with_sd21')
    p_run.add_argument('--defended-cache-dir', type=Path, required=True)
    p_run.add_argument('--memory-vectors-dir', type=Path, required=True)
    p_run.add_argument('--batch-config', type=Path, default=None)
    p_run.add_argument('--batch-size-map', type=str, default=None)
    p_run.add_argument('--text-batch-size', type=int, default=64)
    p_run.add_argument('--verify-mode', choices=VERIFY_MODES, default='subprocess')
    p_run.add_argument('--encoder-slugs', nargs='+', default=None)
    p_run.add_argument('--num-workers', type=int, default=4)
    p_run.add_argument('--device', default='cuda')
    p_run.add_argument('--skip-training-snapshots', action='store_true')

    p_fin = sub.add_parser('finalize')
    p_fin.add_argument('--stage2-dir', type=Path, default=OUTPUT / 'eval_9groups_20261119' / 'stage2_defenses')
    p_fin.add_argument('--jobs-json', type=Path, default=None)
    p_fin.add_argument('--base', type=Path, default=EVALUATION_BASE)
    p_fin.add_argument('--key-mode', choices=('image', 'joint'), default=DEFAULT_KEY_MODE)
    p_fin.add_argument('--defense-mode', default='panel_with_sd21')
    p_fin.add_argument('--encoder-slugs', nargs='+', default=None)

    p_ver = sub.add_parser('_verify-encoder')
    p_ver.add_argument('--encoder-slug', required=True)
    p_ver.add_argument('--batch-size', type=int, default=16)
    p_ver.add_argument('--text-batch-size', type=int, default=64)
    p_ver.add_argument('--num-workers', type=int, default=4)
    p_ver.add_argument('--device', default='cuda')
    p_ver.add_argument('--panel-sub-dirs', nargs='+', type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == '_verify-encoder':
        verify_encoder_worker(
            args.encoder_slug,
            args.panel_sub_dirs,
            batch_size=args.batch_size,
            text_batch_size=args.text_batch_size,
            num_workers=args.num_workers,
            device=args.device,
            isolated_process=True,
        )
        return 0
    if args.command == 'finalize':
        jobs = read_json(args.jobs_json) if args.jobs_json else default_stage2_jobs(args.stage2_dir)
        for j in jobs:
            _write_transfer_panel_outputs(
                j, base=args.base, key_mode=args.key_mode, defense_mode=args.defense_mode, encoder_slugs=args.encoder_slugs
            )
            print('FINALIZED:', j['name'], flush=True)
        return 0

    batch_sizes = None
    if args.batch_config is not None:
        doc = read_json(args.batch_config)
        batch_sizes = {str(k): int(v) for k, v in doc.get('recommended_batch_sizes', doc).items()}
    elif args.batch_size_map:
        batch_sizes = {str(k): int(v) for k, v in json.loads(args.batch_size_map).items()}

    jobs = read_json(args.jobs_json) if args.jobs_json else default_stage2_jobs(args.stage2_dir)
    run_encoder_grouped_panels(
        jobs,
        base=args.base,
        key_mode=args.key_mode,
        defense_mode=args.defense_mode,
        defended_cache_dir=args.defended_cache_dir,
        memory_vector_cache_dir=args.memory_vectors_dir,
        batch_sizes=batch_sizes,
        text_batch_size=args.text_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        verify_training_snapshots=not args.skip_training_snapshots,
        verify_mode=args.verify_mode,
        encoder_slugs=args.encoder_slugs,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
