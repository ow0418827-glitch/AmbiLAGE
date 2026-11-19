"""Stage 3: 在 Env 4（特征文字 + 目标图片 target.jpg）下按 joint（0.5/0.5 图文联合）口径评估 9 组同时替换表现。"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from src.agent.memory.shared_pool import evaluation_banks, load_shared_pool
from src.experiments import shared_evaluation
from src.experiments.defenses import conditions_for_mode
from src.experiments.visual_inception_category_evaluation import (
    TRANSFER_PANEL_ENCODERS,
    load_frozen_suite,
)
from src.paths import EVALUATION_BASE, OUTPUT, PROJECT, SHARED_MEMORY, VISUAL_CONFIGS
from src.visual.audit import freeze_json, read_json, save_json
from src.visual.image_utils import sha256_file
from src.visual.visual_retrieval import VisualMemoryRetriever, detect_index_text_policy

STAGE3_DIR = OUTPUT / 'eval_9groups_20261119' / 'stage3_env4_joint'

ENCODER_SPECS = {
    slug: dict(
        slug=slug,
        label=label,
        role_zh='白盒代理匹配' if slug == '01_clip_l14' else '黑盒迁移留出',
        encoder_config=VISUAL_CONFIGS / (matched_cfg if slug == '01_clip_l14' else heldout_cfg),
    )
    for slug, label, _pair, matched_cfg, heldout_cfg in TRANSFER_PANEL_ENCODERS
}

METHODS = {
    'ambilage': OUTPUT / 'ambilage_single_clip_l14_9groups_20261118',
    'visual_inception': OUTPUT / 'vi_single_clip_l14_9groups_20261118',
}


def build_env4_queries(group: str) -> list[dict]:
    q_path = EVALUATION_BASE / 'evaluation_inputs' / group / 'queries.json'
    q_doc = read_json(q_path)
    target_img = (EVALUATION_BASE / 'groups' / group / 'target.jpg').resolve()
    target_sha = sha256_file(target_img)
    return [
        dict(
            task_id=f'feature_{i:02d}',
            instruction=text,
            query_mode='joint',
            query_image=str(target_img),
            query_image_sha256=target_sha,
        )
        for i, text in enumerate(q_doc['instructions'], 1)
    ]


def verify_worker(method: str, encoder_slug: str, device: str):
    run_dir = STAGE3_DIR / method / encoder_slug
    suite_plan = read_json(run_dir / 'env4_suite_plan.json')
    groups = suite_plan['groups']
    first_plan = read_json(run_dir / 'groups' / groups[0] / 'evaluation_plan.json')
    actual_surrogates = shared_evaluation._plan_training_identities(first_plan)
    base_encoder = shared_evaluation.make_eval_encoder(
        device, first_plan.get('encoder_config'), training_surrogates=actual_surrogates
    )
    fresh_wrapper = shared_evaluation.InProcessFreshCachedEncoder(base_encoder)
    shared_pool = load_shared_pool(first_plan['pool_path'])
    bank_cache, path_sha_cache, bank_digest_cache = {}, {}, {}
    group_verifications = {}
    max_error = 0.0
    try:
        for group in groups:
            gdir = run_dir / 'groups' / group
            ver = shared_evaluation.verify(
                gdir,
                encoder_factory=lambda w=fresh_wrapper: w,
                device=device,
                isolated_process=True,
                _bank_cache=bank_cache,
                _path_sha_cache=path_sha_cache,
                _bank_digest_cache=bank_digest_cache,
                _shared_pool=shared_pool,
            )
            assert ver.get('status') == 'passed' and ver.get('all_candidate_scores_checked') is True
            max_error = max(max_error, float(ver['max_score_error']))
            group_verifications[group] = ver
    finally:
        base_encoder.close()
    freeze_json(
        run_dir / 'env4_independent_verification.json',
        dict(schema_version=1, status='passed', max_score_error=max_error, groups=group_verifications),
    )


def eval_one(method: str, encoder_slug: str, device: str):
    images_root = METHODS[method]
    spec = ENCODER_SPECS[encoder_slug]
    run_dir = STAGE3_DIR / method / encoder_slug
    if (run_dir / 'env4_summary.json').is_file():
        print(f'[SKIP] {method}/{encoder_slug} already evaluated', flush=True)
        return read_json(run_dir / 'env4_summary.json')

    run_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = STAGE3_DIR / 'memory_vectors'
    vec_dir.mkdir(parents=True, exist_ok=True)
    bank_root = run_dir / 'shared_banks'

    frozen_suite = load_frozen_suite(images_root, method=method)
    training_context = shared_evaluation._frozen_training_context(frozen_suite[0], frozen_suite, required=True)
    actual_surrogates = shared_evaluation._context_training_identities(training_context)
    base_encoder = shared_evaluation.make_eval_encoder(
        device, str(spec['encoder_config']), training_surrogates=actual_surrogates
    )
    shared_enc = shared_evaluation.InProcessFreshCachedEncoder(base_encoder)
    entries = load_shared_pool(SHARED_MEMORY)
    ids = [e['Id'] for e in entries]
    pool_sha = sha256_file(SHARED_MEMORY)
    index_policy = detect_index_text_policy(entries)
    none_spec = conditions_for_mode('none')[0]
    replacement_inputs = [
        dict(source=it['source'], image=it['image'], source_memory_id=it.get('source_memory_id'))
        for it in frozen_suite
    ]
    sim_replacements = [
        dict(source_id=it['source']['id'], image_sha256=it['image_sha256'], source_memory_id=it.get('source_memory_id'))
        for it in frozen_suite
    ]

    groups = [it['group'] for it in frozen_suite]
    freeze_json(
        run_dir / 'env4_suite_plan.json',
        dict(
            schema_version=1,
            diagnostic_environment='env4_feature_text_plus_target_image',
            method=method,
            encoder_slug=encoder_slug,
            encoder_label=spec['label'],
            key_mode='joint',
            key_image_weight=0.5,
            query_mode='joint',
            query_image_weight=0.5,
            groups=groups,
        ),
    )

    path_sha_cache, bank_digest_cache = {}, {}
    shared_bank_memo = {}
    try:
        for item in frozen_suite:
            group = item['group']
            gdir = run_dir / 'groups' / group
            queries = build_env4_queries(group)
            banks, source_id = evaluation_banks(
                entries, SHARED_MEMORY, item['source'], item['image'],
                source_memory_id=item.get('source_memory_id'), replacements=replacement_inputs,
            )
            plan = dict(
                schema_version=1,
                scope='shared',
                diagnostic_environment='env4_feature_text_plus_target_image',
                query_image_is_target=True,
                pool_path=str(Path(SHARED_MEMORY).resolve()),
                pool_sha256=pool_sha,
                pool_count=len(entries),
                source_memory_id=source_id,
                frozen_image_sha256=item['image_sha256'],
                source_public_id=item['source']['id'],
                method=method,
                queries=queries,
                conditions=[none_spec],
                training_surrogate_context=training_context,
                training_identity_policy='actual_frozen_surrogates_v1',
                simultaneous_replacements=sim_replacements,
                memory_vector_cache_policy='content_model_decoder_text_fusion_bound_v1',
                encoder_config=str(spec['encoder_config'].resolve()),
                encoder_config_sha256=sha256_file(spec['encoder_config']),
                key_mode='joint',
                key_image_weight=0.5,
                query_image_weight=0.5,
                query_mode='joint',
                query_manifest=None,
                query_manifest_sha256=None,
                index_text=index_policy,
                query_image_defended=False,
                all_memory_images_equally_defended=True,
                shopping_allowed=False,
                record_kinds={kind: sum(e['RecordKind'] == kind for e in entries)
                              for kind in ('purchase_memory', 'simulated_gallery')},
            )
            freeze_json(gdir / 'evaluation_plan.json', plan)

            rows, score_rows = [], []
            for name, original_bank in banks.items():
                bank_file = bank_root / 'banks' / 'none' / f'{name}.json'
                if name not in shared_bank_memo:
                    freeze_json(bank_file, original_bank)
                    shared_bank_memo[name] = (original_bank, sha256_file(bank_file))
                bank, bank_sha = shared_bank_memo[name]
                ranker = VisualMemoryRetriever(
                    shared_enc, bank_file, key_mode='joint', query_mode='joint',
                    key_image_weight=0.5, query_image_weight=0.5,
                    index_text=index_policy, cache_dir=None,
                    entry_cache_dir=vec_dir, image_base_path=SHARED_MEMORY,
                    freeze_memory=True, path_sha_cache=path_sha_cache,
                    bank_digest_cache=bank_digest_cache,
                )
                observations, candidate_scores = [], []
                for q in queries:
                    ranking = ranker.rank(bank, q['instruction'], q['query_image'])
                    chosen = next(r for r in ranking if r['Id'] == source_id)
                    candidate_scores.append(shared_evaluation._score_vector(ranking, ids))
                    observations.append(dict(task_id=q['task_id'], rank=chosen['rank'], score=chosen['score']))
                rows.append(dict(
                    condition=none_spec, bank=name, bank_path=str(bank_file.resolve()),
                    bank_sha256=bank_sha, queries=observations,
                    top3=sum(q['rank'] <= 3 for q in observations),
                    top5=sum(q['rank'] <= 5 for q in observations),
                    query_count=len(queries), score_archive_index=len(rows),
                    retrieval_protocol=copy.deepcopy(ranker.snapshot),
                ))
                score_rows.append(candidate_scores)
            shared_evaluation._save_scores(gdir, score_rows, ids, queries, rows, source_id)
            freeze_json(gdir / 'evaluation.json', rows)
            save_json(gdir / 'evaluation.partial.json', rows)
    finally:
        base_encoder.close()

    cmd = [
        sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
        '--verify-only', '--method', method, '--encoder-slug', encoder_slug, '--device', device,
    ]
    with (run_dir / 'env4_verification.log').open('w', encoding='utf-8') as log:
        proc = subprocess.run(
            cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONPATH=str(PROJECT)),
        )
    if proc.returncode:
        raise RuntimeError(f'Env4 isolated verify failed for {method}/{encoder_slug}')

    per_group = {}
    c_t1 = c_t3 = c_t5 = m_t1 = m_t3 = m_t5 = n_q = 0
    c_rank_sum = m_rank_sum = 0.0
    for group in groups:
        gdir = run_dir / 'groups' / group
        shared_evaluation.report(gdir)
        rows = read_json(gdir / 'evaluation.json')
        by_bank = {r['bank']: r for r in rows}
        cr = [q['rank'] for q in by_bank['clean']['queries']]
        mr = [q['rank'] for q in by_bank['modified']['queries']]
        c_t1 += sum(r == 1 for r in cr)
        m_t1 += sum(r == 1 for r in mr)
        c_t3 += sum(r <= 3 for r in cr)
        m_t3 += sum(r <= 3 for r in mr)
        c_t5 += sum(r <= 5 for r in cr)
        m_t5 += sum(r <= 5 for r in mr)
        n_q += len(cr)
        c_rank_sum += sum(cr)
        m_rank_sum += sum(mr)
        per_group[group] = dict(
            query_count=len(cr),
            clean_top1=sum(r == 1 for r in cr), modified_top1=sum(r == 1 for r in mr),
            clean_top3=sum(r <= 3 for r in cr), modified_top3=sum(r <= 3 for r in mr),
            clean_top5=sum(r <= 5 for r in cr), modified_top5=sum(r <= 5 for r in mr),
            clean_mean_rank=sum(cr) / len(cr), modified_mean_rank=sum(mr) / len(mr),
        )
    ver_doc = read_json(run_dir / 'env4_independent_verification.json')
    summary = dict(
        schema_version=1,
        diagnostic_environment='env4_feature_text_plus_target_image',
        method=method,
        encoder_slug=encoder_slug,
        encoder_label=spec['label'],
        role_zh=spec['role_zh'],
        key_mode='joint',
        key_image_weight=0.5,
        query_mode='joint',
        query_image_weight=0.5,
        query_count=n_q,
        clean_top1=c_t1 / n_q, modified_top1=m_t1 / n_q,
        clean_top3=c_t3 / n_q, modified_top3=m_t3 / n_q,
        clean_top5=c_t5 / n_q, modified_top5=m_t5 / n_q,
        clean_top1_hits=c_t1, modified_top1_hits=m_t1,
        clean_top3_hits=c_t3, modified_top3_hits=m_t3,
        clean_top5_hits=c_t5, modified_top5_hits=m_t5,
        clean_mean_rank=c_rank_sum / n_q,
        modified_mean_rank=m_rank_sum / n_q,
        verification_status=ver_doc['status'],
        max_score_error=ver_doc['max_score_error'],
        groups=per_group,
    )
    freeze_json(run_dir / 'env4_summary.json', summary)
    return summary


def write_stage3_report(slugs: list[str], filename_prefix: str):
    lines = [
        '# Stage 3 Env4（特征文字 + 目标图片 `target.jpg`，`joint 0.5/0.5`）对比报告',
        '',
        '| 方法 | 编码器 | 角色 | 原图 Top-1 | 修改后 Top-1 | 原图 Top-3 | 修改后 Top-3 | 原图 Top-5 | 修改后 Top-5 | 均名(原->改) | 复算状态 / 最大误差 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    records = {}
    for slug in slugs:
        for method in ('ambilage', 'visual_inception'):
            p = STAGE3_DIR / method / slug / 'env4_summary.json'
            if not p.is_file():
                continue
            s = read_json(p)
            records[f'{method}__{slug}'] = s
            lines.append(
                f"| `{method}` | {s['encoder_label']} | {s['role_zh']} | "
                f"{s['clean_top1']*100:.1f}% ({s['clean_top1_hits']}/{s['query_count']}) | "
                f"{s['modified_top1']*100:.1f}% ({s['modified_top1_hits']}/{s['query_count']}) | "
                f"{s['clean_top3']*100:.1f}% ({s['clean_top3_hits']}/{s['query_count']}) | "
                f"{s['modified_top3']*100:.1f}% ({s['modified_top3_hits']}/{s['query_count']}) | "
                f"{s['clean_top5']*100:.1f}% ({s['clean_top5_hits']}/{s['query_count']}) | "
                f"{s['modified_top5']*100:.1f}% ({s['modified_top5_hits']}/{s['query_count']}) | "
                f"{s['clean_mean_rank']:.2f}->{s['modified_mean_rank']:.2f} | "
                f"{s['verification_status']} ({s['max_score_error']:.2e}) |"
            )
    (STAGE3_DIR / f'{filename_prefix}.json').write_text(
        json.dumps(dict(schema_version=1, records=records), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (STAGE3_DIR / f'{filename_prefix}.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('whitebox', 'blackbox', 'all'), default='whitebox')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--method', choices=tuple(METHODS))
    parser.add_argument('--encoder-slug', choices=tuple(ENCODER_SPECS))
    args = parser.parse_args(argv)

    if args.verify_only:
        assert args.method and args.encoder_slug
        verify_worker(args.method, args.encoder_slug, args.device)
        return 0

    if args.phase == 'whitebox':
        slugs = ['01_clip_l14']
        report_prefix = 'STAGE3_WHITEBOX_SUMMARY'
    elif args.phase == 'blackbox':
        slugs = ['02_openclip_bigG14', '03_siglip_so400m']
        report_prefix = 'STAGE3_FULL_SUMMARY'
    else:
        slugs = ['01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m']
        report_prefix = 'STAGE3_FULL_SUMMARY'

    for slug in slugs:
        for method in ('ambilage', 'visual_inception'):
            s = eval_one(method, slug, args.device)
            print(
                f"[ENV4 {method}/{slug}] Top-1 {s['clean_top1']*100:.1f}%->{s['modified_top1']*100:.1f}% | "
                f"Top-3 {s['clean_top3']*100:.1f}%->{s['modified_top3']*100:.1f}% | "
                f"Top-5 {s['clean_top5']*100:.1f}%->{s['modified_top5']*100:.1f}% | "
                f"MeanRank {s['clean_mean_rank']:.2f}->{s['modified_mean_rank']:.2f}",
                flush=True,
            )

    write_stage3_report(
        ['01_clip_l14', '02_openclip_bigG14', '03_siglip_so400m'] if report_prefix == 'STAGE3_FULL_SUMMARY' else slugs,
        report_prefix,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
