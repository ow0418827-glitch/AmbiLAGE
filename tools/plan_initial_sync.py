"""Plan small initial Git pushes from checked candidates; never stage or push."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT
from tools.check_github_sync import inspect_github_sync, MAX_REGULAR_GIT_FILE_BYTES


MIB = 1024 * 1024
DEFAULT_MAX_BATCH_BYTES = 512 * MIB
GITHUB_PUSH_LIMIT_BYTES = 2 * 1024 * MIB
ARTIFACT_ROOT = PurePosixPath('output/github_sync_check')


def _candidate(root, relative):
    """Inspect one project-relative regular file without reading its contents."""
    item = PurePosixPath(relative)
    if item.is_absolute() or '..' in item.parts or '\\' in relative or not item.parts:
        raise ValueError('Synchronization candidates must be project-relative literal paths')
    path = root.joinpath(*item.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError('A synchronization candidate is missing or is not a regular file: ' + relative)
    if not path.resolve().is_relative_to(root):
        raise ValueError('A synchronization candidate resolves outside the project: ' + relative)
    stat = path.stat()
    if stat.st_size > MAX_REGULAR_GIT_FILE_BYTES:
        raise ValueError('A synchronization candidate exceeds the regular Git file limit: ' + relative)
    return dict(path=item.as_posix(), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _project_file_order(item):
    path = PurePosixPath(item['path'])
    # Code, configuration and manifests remain easy to review before bulk images
    # and experiment records. All project files precede compressed catalog parts.
    manifest = 'manifest' in path.name.lower() or path.name.lower() in {'model_lock.json', 'github_sync_policy.json'}
    code = path.parts[0] in {'src', 'tools', 'requirements', 'docs'} or len(path.parts) == 1
    return (0 if code or manifest else 1, item['bytes'], item['path'])


def build_initial_sync_plan(root=PROJECT, *, max_batch_bytes=DEFAULT_MAX_BATCH_BYTES):
    """Return a checked plan. This performs no index, commit or remote mutation.

    The shared checker may create its isolated, ignored inspection repository;
    it never initializes the project's repository. No existing check report is
    trusted: the actual candidate/credential check runs afresh on every call.
    """
    root = Path(root).resolve()
    if not isinstance(max_batch_bytes, int) or not 0 < max_batch_bytes <= DEFAULT_MAX_BATCH_BYTES:
        raise ValueError('The batch budget must be between one byte and 512 MiB')
    check = inspect_github_sync(root)
    if check.get('passed') is not True:
        raise ValueError('Git synchronization inspection must pass before planning; run tools/check_github_sync.py')
    candidates = check['candidate_paths']
    if len(candidates) != len(set(candidates)):
        raise ValueError('The synchronization inspection returned duplicate candidate paths')
    files = [_candidate(root, relative) for relative in candidates]
    if any(item['bytes'] > max_batch_bytes for item in files):
        raise ValueError('A synchronization candidate exceeds the selected batch budget')
    catalog_prefix = 'resources/webshop/catalog_bundle/'
    chunks = [item for item in files if item['path'].startswith(catalog_prefix) and item['path'].endswith('.part')]
    chunk_paths = {item['path'] for item in chunks}
    project = [item for item in files if item['path'] not in chunk_paths]
    groups = [('project_files', sorted(project, key=_project_file_order)),
              ('catalog_bundle_parts', sorted(chunks, key=lambda item: item['path']))]
    batches = []
    for phase, ordered in groups:
        current = None
        for item in ordered:
            if current is None or current['bytes'] + item['bytes'] > max_batch_bytes:
                current = dict(batch=len(batches) + 1, phase=phase, file_count=0, bytes=0,
                               max_file_bytes=0, paths=[])
                batches.append(current)
            current['paths'].append(item['path'])
            current['bytes'] += item['bytes']
            current['file_count'] += 1
            current['max_file_bytes'] = max(current['max_file_bytes'], item['bytes'])
    inventory = sorted(files, key=lambda item: item['path'])
    fingerprint = hashlib.sha256(json.dumps(inventory, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    return dict(schema_version=1, planned_at=datetime.now(timezone.utc).isoformat(),
        ready=True, source_check_passed=True, source_checked_at=check.get('checked_at'),
        repository_mode=check.get('repository_mode'), scope='initial_sync_all_checked_candidates',
        candidate_count=len(files), candidate_bytes=sum(item['bytes'] for item in files),
        candidate_inventory_sha256=fingerprint, inventory_fingerprint_fields=['path', 'bytes', 'mtime_ns'],
        max_batch_bytes=max_batch_bytes, max_regular_git_file_bytes=MAX_REGULAR_GIT_FILE_BYTES,
        github_push_limit_bytes=GITHUB_PUSH_LIMIT_BYTES, batch_count=len(batches), batches=batches,
        snapshot_note='Re-run the checker and planner after any candidate changes; byte totals are file sizes, not a measured Git pack.',
        push_note='Each batch requires a separate commit and completed push; committing all batches before one push does not reduce that push size.',
        staged_committed_or_pushed=False, credential_values_included=False,
        remote_visibility_verified=False)


def write_plan_artifacts(plan, root=PROJECT, *, output_dir='output/github_sync_check/initial_sync'):
    """Write JSON and NUL-delimited literal path lists only in the ignored area."""
    root = Path(root).resolve()
    output = Path(output_dir)
    output = output.resolve() if output.is_absolute() else (root / output).resolve()
    allowed = root.joinpath(*ARTIFACT_ROOT.parts).resolve()
    if output == allowed or not output.is_relative_to(allowed) or not allowed.is_relative_to(root):
        raise ValueError('Plan artifacts must use a subdirectory of output/github_sync_check')
    if plan.get('ready') is not True or plan.get('source_check_passed') is not True:
        raise ValueError('Only a passed synchronization plan can be written')
    output.mkdir(parents=True, exist_ok=True)
    for batch in plan['batches']:
        relative = 'batch-{:03d}.paths0'.format(batch['batch'])
        path = output / relative
        if path.is_symlink():
            raise ValueError('Plan artifacts cannot overwrite symbolic links')
        # NUL separators preserve spaces, Unicode, newlines and pathspec syntax.
        # A future explicit add can use --literal-pathspecs and --pathspec-file-nul.
        path.write_bytes(b''.join(value.encode('utf-8') + b'\0' for value in batch['paths']))
        batch['literal_path_list'] = path.relative_to(root).as_posix()
    result = output / 'plan.json'
    if result.is_symlink():
        raise ValueError('Plan artifacts cannot overwrite symbolic links')
    result.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-mib', type=int, default=512,
                        help='Per-batch uncompressed file-size budget, between 1 and 512 MiB')
    parser.add_argument('--output-dir', type=Path,
                        help='Optional ignored output/github_sync_check subdirectory for plan.json and *.paths0')
    args = parser.parse_args(argv)
    try:
        plan = build_initial_sync_plan(max_batch_bytes=args.batch_mib * MIB)
        if args.output_dir is not None:
            write_plan_artifacts(plan, output_dir=args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps(dict(ready=False, error=str(error), staged_committed_or_pushed=False,
                              credential_values_included=False), ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
