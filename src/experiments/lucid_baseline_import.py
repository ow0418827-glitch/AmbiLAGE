"""Import verified historical full-step pixel baselines without training or re-ranking."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageChops

GROUPS = ('01_camera', '02_chair', '03_cookies', '04_curtain', '05_hair_extension',
          '06_jeans', '07_shampoo', '08_speaker', '09_toothbrush')
OLD_RUN = 'output/lucid_groups9_20261109'
RESELECT_RUN = 'output/lucid_shampoo_reselect_20261112'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def audit_frozen(frozen_path):
    """Read pixels and completion evidence only; never initialize a model."""
    frozen_path = Path(frozen_path).resolve()
    frozen = read_json(frozen_path)
    metadata = frozen.get('metadata', {})
    attack = metadata.get('attack', {})
    if (frozen.get('method') != 'lucid' or frozen.get('status') != 'frozen'
            or metadata.get('method') != 'lucid' or metadata.get('status') != 'success'):
        raise ValueError('Historical baseline is not a completed frozen pixel result')
    if metadata.get('test_fixture') or metadata.get('production_backend') is False:
        raise ValueError('A test fixture cannot be imported as a historical baseline')
    for value in (frozen.get('requested_steps'), attack.get('steps_requested'), attack.get('steps_completed')):
        if type(value) is not int or value != 1000:
            raise ValueError('Historical baseline must contain all 1000 requested steps')
    if attack.get('stop_reason') != 'requested_steps_completed' or attack.get('selected_step') != 1000:
        raise ValueError('Historical baseline did not select the final completed step')
    if not math.isclose(float(attack.get('epsilon', -1)), 8 / 255, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Historical baseline must use an eight-level pixel budget')
    image = Path(frozen['image'])
    if sha256(image) != frozen.get('image_sha256') or frozen['image_sha256'] != metadata.get('output_image_sha256'):
        raise ValueError('Frozen image content does not match completion evidence')
    if sha256(metadata['output_image']) != frozen['image_sha256']:
        raise ValueError('Generated image and frozen image differ')
    for role in ('source', 'target'):
        if (sha256(frozen[role]['image']) != frozen[role]['image_sha256']
                or metadata.get(role + '_image_sha256') != frozen[role]['image_sha256']):
            raise ValueError('Historical public image identity does not match: ' + role)
    alignment = metadata.get('text_alignment', {})
    text = frozen['target'].get('feature_text')
    if (alignment.get('enabled') is not True or alignment.get('weight') != 1.0
            or not isinstance(text, str) or hashlib.sha256(text.encode('utf-8')).hexdigest() != alignment.get('text_sha256')):
        raise ValueError('Historical baseline is not bound to the frozen target feature text')
    public_path = frozen_path.parent / 'PUBLIC_RUN.json'
    if canonical_digest(read_json(public_path)) != frozen.get('public_binding_sha256'):
        raise ValueError('Historical public run binding changed')
    with Image.open(image) as result, Image.open(frozen['source']['image']) as source:
        a, b = result.convert('RGB'), source.convert('RGB')
        if a.size != b.size:
            raise ValueError('Historical baseline changed the native image dimensions')
        maximum = max(high for low, high in ImageChops.difference(a, b).getextrema())
    if maximum > 8:
        raise ValueError('Decoded historical image exceeds the eight-level budget')
    surrogates = metadata.get('surrogates', [])
    expected = [('ViT-B-16', 'openai'), ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion2b_s34b_b79k')]
    if [(s.get('name'), s.get('pretrained')) for s in surrogates] != expected:
        raise ValueError('Historical baseline does not contain the original three proxies')
    return frozen, {
        'full_generation_verified': True, 'decoded_maximum_pixel_levels': maximum,
        'steps': 1000, 'epsilon_levels': 8, 'seed': metadata.get('seed'), 'text_weight': 1.0,
        'target_text_sha256': alignment['text_sha256'],
        'surrogate_activations': [dict(name=s['name'], pretrained=s['pretrained'],
            weight_sha256=s.get('weight_sha256'), activation=s.get('loading', {}).get('activation'),
            actual_activation_modules=s.get('loading', {}).get('actual_activation_modules', {})) for s in surrogates],
    }


def _copy_file(source, destination, source_root, destination_root, records):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_relative_to(source_root) or not destination.is_relative_to(destination_root):
        raise ValueError('Import path escaped its explicitly selected project')
    expected = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != expected:
            raise ValueError('Import refuses to overwrite a changed existing file: ' + str(destination))
    else:
        shutil.copyfile(source, destination)
    if sha256(destination) != expected or sha256(source) != expected:
        raise ValueError('Source or copied bytes changed during import')
    records.append(dict(source=source.relative_to(source_root).as_posix(),
                        path=destination.relative_to(destination_root).as_posix(),
                        sha256=expected, bytes=source.stat().st_size))


def _historical_evaluation(source_root, group, root, source_run):
    if group != '07_shampoo':
        directory = root / 'evaluation'
        verification = read_json(directory / 'independent_verification.json')
        if (verification.get('status') != 'passed'
                or verification.get('evaluated_file_sha256') != sha256(directory / 'evaluation.json')
                or verification.get('evaluation_plan_sha256') != sha256(directory / 'evaluation_plan.json')):
            raise ValueError('Historical retrieval verification does not match saved files: ' + group)
        plan = read_json(directory / 'evaluation_plan.json')
        rows = read_json(directory / 'evaluation.json')
        banks = []
        for row in rows:
            path = Path(row['bank_path'])
            if sha256(path) != row['bank_sha256']:
                raise ValueError('Historical evaluation bank changed: ' + group)
            banks.append(dict(source=path.relative_to(source_root).as_posix(), sha256=sha256(path), bytes=path.stat().st_size))
        return dict(status='historical_passed', rankings=verification['rankings_recomputed'],
                    encoder_setting=plan.get('encoder_config'), original_bank_files_not_duplicated=banks,
                    reused_as_current_evaluation=False)
    verification = read_json(source_run / 'INDEPENDENT_VERIFICATION.json')
    completed = read_json(source_run / 'COMPLETED.json')
    if (verification.get('status') != 'passed' or verification.get('score_comparisons') != 15120
            or completed.get('status') != 'completed' or completed.get('source_id') != 'B09B63YYCW'
            or completed.get('verification_sha256') != sha256(source_run / 'INDEPENDENT_VERIFICATION.json')):
        raise ValueError('Reselected shampoo retrieval completion evidence is inconsistent')
    for path, digest in verification['files'].items():
        if sha256(path) != digest:
            raise ValueError('Reselected shampoo historical rankings changed')
    banks = [dict(source=p.relative_to(source_root).as_posix(), sha256=sha256(p), bytes=p.stat().st_size)
             for p in (source_run / 'banks').glob('*.json')]
    return dict(status='historical_passed', rankings=80, candidate_scores=15120,
                original_bank_files_not_duplicated=banks, reused_as_current_evaluation=False)


def import_baselines(source_project, project, destination):
    """One-time read-only source migration; no weight loads or model calls."""
    source_project, project, destination = (Path(p).resolve() for p in (source_project, project, destination))
    if source_project == project or not destination.is_relative_to(project / 'output'):
        raise ValueError('Use a distinct read-only source and a destination inside the new project output')
    manifest_path = destination / 'IMPORT_MANIFEST.json'
    if manifest_path.is_file():
        return verify_imported_manifest(project, manifest_path, check_public_inputs=False)
    manifest = dict(schema_version=1, method='lucid', scope='frozen_historical_baseline_import',
        source_project=str(source_project), new_generation_executed=False,
        historical_evaluations_are_current=False,
        current_evaluation_requires_selected_encoder_and_current_complete_shared_pool=True,
        activation_note='Preserve the three original recorded proxy activations; do not relabel as a fresh native-loader run.',
        groups={}, files=[])
    for group in GROUPS:
        relative_run = RESELECT_RUN if group == '07_shampoo' else OLD_RUN
        run = source_project / relative_run
        root = run / ('generation/lucid/07_shampoo' if group == '07_shampoo' else f'groups/{group}/methods/lucid')
        frozen, checked = audit_frozen(root / 'FROZEN_IMAGE.json')
        if group == '07_shampoo' and (frozen['source']['id'], frozen['target']['id']) != ('B09B63YYCW', 'B00BDJODWI'):
            raise ValueError('The seventh baseline is not the later reselected shampoo pair')
        historical = _historical_evaluation(source_project, group, root, run)
        evidence = destination / 'original_evidence' / Path(relative_run).name
        for path in sorted(root.rglob('*')):
            if not path.is_file() or 'banks' in path.relative_to(root).parts:
                continue
            if path.suffix.lower() not in ('.json', '.md', '.png'):
                continue
            target = evidence / path.relative_to(run)
            _copy_file(path, target, source_project, project, manifest['files'])
        if group == '07_shampoo':
            for name in ('EXPERIMENT_PLAN.json', 'PREPARED.json', 'FROZEN_IMAGES.json', 'EVALUATION_INPUTS.json',
                         'COMPLETED.json', 'RETRIEVAL_COMPLETED.json', 'INDEPENDENT_VERIFICATION.json',
                         'SUMMARY.json', 'REPORT.md', 'comparison_visual_07_shampoo_lucid.png'):
                _copy_file(run / name, evidence / name, source_project, project, manifest['files'])
            for subdirectory in ('retrieval', 'logs'):
                for path in sorted((run / subdirectory).rglob('*')):
                    if path.is_file():
                        _copy_file(path, evidence / path.relative_to(run), source_project, project, manifest['files'])
        image = destination / 'groups' / group / 'frozen.png'
        _copy_file(Path(frozen['image']), image, source_project, project, manifest['files'])
        generation_result = root / 'generation/generation_result.json'
        manifest['groups'][group] = dict(
            image=dict(path=image.relative_to(project).as_posix(), sha256=sha256(image)),
            original_frozen=dict(path=(evidence / (root / 'FROZEN_IMAGE.json').relative_to(run)).relative_to(project).as_posix(),
                                 sha256=sha256(root / 'FROZEN_IMAGE.json')),
            metadata=dict(path=(evidence / generation_result.relative_to(run)).relative_to(project).as_posix(),
                          sha256=sha256(generation_result)),
            source={k: copy.deepcopy(frozen['source'][k]) for k in ('id', 'title', 'image_sha256')},
            target={k: copy.deepcopy(frozen['target'][k]) for k in ('id', 'title', 'image_sha256', 'feature_text')},
            parameters={k: checked[k] for k in ('steps', 'epsilon_levels', 'seed', 'text_weight')},
            checks=checked, historical_evaluation=historical)
    old_run = source_project / OLD_RUN
    for name in ('REPORT_ALL9.md', 'summary_metrics.json'):
        _copy_file(old_run / name, destination / 'original_evidence' / old_run.name / name,
                   source_project, project, manifest['files'])
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return verify_imported_manifest(project, manifest_path, check_public_inputs=False)


def verify_imported_manifest(project, manifest_path, *, check_public_inputs=True):
    project = Path(project).resolve()
    manifest = read_json(manifest_path)
    if tuple(manifest.get('groups', {})) != GROUPS or manifest.get('new_generation_executed') is not False:
        raise ValueError('Imported baseline group set or provenance is invalid')
    for item in manifest['files']:
        path = (project / item['path']).resolve()
        if not path.is_relative_to(project) or sha256(path) != item['sha256']:
            raise ValueError('Imported historical evidence changed: ' + item['path'])
    base = read_json(project / 'resources/evaluation_base/manifest.json') if check_public_inputs else None
    for group, item in manifest['groups'].items():
        if item['parameters']['steps'] != 1000 or item['parameters']['epsilon_levels'] != 8:
            raise ValueError('Imported baseline parameters changed')
        if sha256(project / item['image']['path']) != item['image']['sha256']:
            raise ValueError('Imported frozen image changed')
        if base:
            spec = base['groups'][group]
            for role in ('source', 'target'):
                if item[role]['id'] != spec[role + '_id'] or item[role]['image_sha256'] != spec[role + '_sha256']:
                    raise ValueError('Imported baseline does not match the current public pair: ' + group + '/' + role)
            if item['target']['feature_text'] != spec['target_feature_text']:
                raise ValueError('Imported baseline training text differs from current frozen features')
    return manifest


def load_imported_baseline(group, inputs=None, index_path=None, *, project=None):
    """Materialize current-project paths while preserving original evidence bytes.

    Only selected generation evidence is read here. Historical rankings are not
    consulted, and no encoder or generator is initialized.
    """
    from src.paths import PROJECT
    from src.visual.attack_completion import validate_attack_completion
    project = Path(project or PROJECT).resolve()
    index_path = Path(index_path or project / 'output/imported_lucid_baseline_20261113/IMPORT_MANIFEST.json')
    index_path = (index_path if index_path.is_absolute() else project / index_path).resolve()
    if not index_path.is_relative_to(project):
        raise ValueError('Historical baseline index must be inside the current project')
    manifest = read_json(index_path)
    if manifest.get('method') != 'lucid' or group not in manifest.get('groups', {}):
        raise ValueError('Requested historical pixel baseline is not registered')
    item = manifest['groups'][group]
    if item['parameters'].get('steps') != 1000 or item['parameters'].get('epsilon_levels') != 8:
        raise ValueError('Historical pixel baseline parameters are not eight levels and 1000 steps')
    paths = {}
    for key in ('image', 'original_frozen', 'metadata'):
        path = (project / item[key]['path']).resolve()
        if not path.is_relative_to(project) or sha256(path) != item[key]['sha256']:
            raise ValueError('Historical baseline artifact changed: ' + key)
        paths[key] = path
    frozen = copy.deepcopy(read_json(paths['original_frozen']))
    original_metadata = read_json(paths['metadata'])
    if canonical_digest(original_metadata) != canonical_digest(frozen['metadata']):
        raise ValueError('Historical metadata and frozen record are inconsistent')
    if inputs is None:
        base = project / 'resources/evaluation_base'
        spec = read_json(base / 'manifest.json')['groups'][group]
        definition = read_json(base / spec['definition'])
        inputs = {role: dict(id=spec[role + '_id'], title=definition[role]['title'],
                    image=str(base / spec['directory'] / definition[role]['image']),
                    image_sha256=spec[role + '_sha256']) for role in ('source', 'target')}
        inputs['target_text'] = spec['target_feature_text']
    for role in ('source', 'target'):
        current = inputs[role]
        if (current['id'] != item[role]['id'] or current['image_sha256'] != item[role]['image_sha256']
                or sha256(current['image']) != current['image_sha256']):
            raise ValueError('Imported baseline no longer matches the current public pair: ' + group + '/' + role)
        frozen[role].update(id=current['id'], title=current['title'], image=str(Path(current['image']).resolve()),
                            image_sha256=current['image_sha256'])
    text = inputs.get('target_text', inputs.get('target', {}).get('feature_text'))
    if text != item['target']['feature_text']:
        raise ValueError('Historical baseline target feature text differs from the current public input')
    provenance = dict(kind='historical_completed_baseline_reuse', generation_executed=False,
        newly_evaluated=False, original_evidence_kept_unchanged=True,
        import_manifest_sha256=sha256(index_path), original_frozen_sha256=item['original_frozen']['sha256'],
        original_metadata_sha256=item['metadata']['sha256'],
        original_proxy_activation_record=item['checks']['surrogate_activations'])
    frozen.update(group=group, image=str(paths['image']), image_sha256=item['image']['sha256'],
                  requested_steps=1000, historical_reuse=provenance)
    frozen['target']['feature_text'] = text
    frozen['metadata'].update(output_image=str(paths['image']), output_metadata=str(paths['metadata']),
                              historical_reuse=provenance)
    validate_attack_completion(frozen['metadata'], 1000, dict(method='lucid',
        source_image_sha256=frozen['source']['image_sha256'], target_image_sha256=frozen['target']['image_sha256']))
    return frozen
