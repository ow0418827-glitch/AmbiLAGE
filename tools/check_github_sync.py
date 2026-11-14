"""Inspect actual Git synchronization candidates without staging, committing or pushing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT

IGNORED_PROBES = (
    ".venv/pyvenv.cfg", "venv/pyvenv.cfg", "models/editor/model.safetensors", "dist/archive.zip",
    "output/migration_legacy/old.json", "output/archive/old.json", "output/github_sync_check/temp.git/config",
    "output/local_validation/final_suite/test.json", "output/package_check/copy.txt", "output/pv/test.png", "output/pytest-123/test.json",
    "output/formal/cache/rebuildable.json", "weights/encoder/model.pt",
    "resources/webshop/catalog/catalog.sqlite",
    "resources/webshop/catalog_bundle/manifest.partial.json",
    "resources/webshop/catalog_bundle/chunks/.write-test.tmp",
    ".env", ".env.local", "src/other.local.json", "src/agent/config/llm.local.json", "keyfile", "keyfile.txt", ".keyfile",
    "credentials.json", "secrets.json", "private.pem", "private.key", "identity.p12", "identity.pfx", "id_rsa", "id_ed25519",
)
SYNC_PROBES = (
    "resources/webshop/catalog_bundle/manifest.json",
    "resources/webshop/catalog_bundle/catalog-00000.part",
    "resources/webshop/catalog_bundle/chunks/f0000.00000.part",
    "output/imported_lucid_baseline_20261113/IMPORT_MANIFEST.json",
    "output/imported_lucid_baseline_20261113/groups/07_shampoo/frozen.png",
    "output/formal_experiment/memory_vectors/entry.json",
    "resources/model_lock.json",
    "resources/evaluation_base/shared_memory/shared_memory.json",
    "resources/evaluation_base/shared_memory/images/shared.jpg", "data/private/input.json",
    "src/agent/config/llm.example.json", "output/formal_experiment/frozen.png",
    "output/formal_experiment/evaluation/scores.npz", "output/formal_experiment/results.json",
    "output/formal_experiment/USER_PAUSE_CHECKPOINT.json",
    "output/formal_experiment/retrieval_index_cache/vector.json", "output/formal_experiment/query_index_cache/query.json",
    "output/formal_experiment/resume.pt", "output/formal_experiment/optimizer.pth",
    "output/local_validation/pytest.log", "output/local_validation/pytest.xml", "output/local_validation/preflight.json",
)
LOCAL_CREDENTIAL_CONFIG_PATH = "src/agent/config/llm.local.json"
MAX_REGULAR_GIT_FILE_BYTES = 100 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".ps1", ".sh", ".ini", ".cfg", ".env",
                 ".log", ".xml", ".jsonl", ".ndjson", ".csv", ".tsv", ".html", ".ipynb", ".conf"}
CREDENTIAL_PATTERNS = {
    "api_token_shape": re.compile(r"(?i)(sk-[a-z0-9_-]{16,}|hf_[a-z0-9]{24,})"),
    "bearer_token_shape": re.compile(r"(?i)Bearer [a-z0-9._-]{20,}"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "cloud_access_key_shape": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def _git(arguments, *, cwd, input_text=None, check=True):
    result = subprocess.run(["git", "--no-optional-locks", *arguments], cwd=cwd,
        input=input_text, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and result.returncode:
        raise RuntimeError("Git synchronization inspection failed: " + result.stderr.strip())
    return result


def forbidden_path(relative):
    value = relative.replace("\\", "/")
    parts = tuple(part.lower() for part in Path(value).parts)
    lowered = value.lower()
    if any(part in {".venv", "venv", "models", "weights", "dist", "__pycache__", ".pytest_cache", ".cache", "cache"} for part in parts):
        return True
    if lowered.startswith(tuple("output/" + name + "/" for name in ("migration_legacy", "archive", "github_sync_check", "package_check", "pv", "test_tmp"))):
        return True
    if lowered.startswith('resources/webshop/catalog/'):
        return True
    if lowered.startswith('resources/webshop/catalog_bundle/') and (
            lowered.endswith(('manifest.partial.json', '.tmp', '.building'))):
        return True
    if len(parts) > 3 and parts[:2] == ("output", "local_validation"):
        return True
    if len(parts) > 1 and parts[0] == "output" and parts[1].startswith("pytest"):
        return True
    name = parts[-1] if parts else ""
    return (name.endswith((".local.json", ".pem", ".key", ".p12", ".pfx"))
            or name.startswith(("keyfile", ".keyfile"))
            or name in {"id_rsa", "id_ed25519", "credentials.json", "secrets.json"}
            or (name.startswith(".env") and name != ".env.example"))


def read_sync_policy(root):
    policy = json.loads((root / "resources/github_sync_policy.json").read_text(encoding="utf-8"))
    paths = {item["path"] for item in policy.get("user_authorized_credentials", [])}
    visibility = policy.get('repository_visibility')
    if visibility not in {'private', 'public'} or policy.get('visibility_confirmation') != 'user_confirmed':
        raise ValueError('Repository visibility must be public or private and explicitly user_confirmed')
    if policy.get('private_repo_required') is not (visibility == 'private'):
        raise ValueError('Repository visibility and private_repo_required must agree')
    if paths or policy.get('credentials_must_not_be_committed') is not True:
        raise ValueError('Real credentials must not be committed; no credential exceptions are allowed')
    return policy, paths


def _known_local_credentials(root):
    """Read only for exact leak matching; never return these values in a report."""
    path = root / LOCAL_CREDENTIAL_CONFIG_PATH
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
        value = raw.get('model', {}).get('api_key')
    except (OSError, ValueError, AttributeError):
        return []
    return [value.strip()] if isinstance(value, str) and len(value.strip()) >= 16 else []


def credential_kinds(path, known_secrets=(), chunk_bytes=1024 * 1024):
    """Bounded-memory scanning, including tokens crossing a chunk boundary."""
    found = set()
    overlap = max([512, *(len(value.encode('utf-8')) + 1 for value in known_secrets)])
    suffix = b''
    with Path(path).open('rb') as stream:
        while block := stream.read(chunk_bytes):
            data = suffix + block
            text = data.decode('utf-8', errors='replace')
            for kind, pattern in CREDENTIAL_PATTERNS.items():
                if pattern.search(text):
                    found.add(kind)
            if any(value.encode('utf-8') in data for value in known_secrets):
                found.add('known_local_credential')
            suffix = data[-overlap:]
    return sorted(found)


def catalog_bundle_status(root):
    """Check manifest structure and part inventory only; no decompression or hashing."""
    from src.webshop.bundle import inspect_bundle
    bundle = Path(root) / 'resources/webshop/catalog_bundle'
    partial = bundle / 'manifest.partial.json'
    if partial.is_file():
        return dict(ready=False, status='packing_incomplete', chunk_hashes_checked=False)
    if not (bundle / 'manifest.json').is_file():
        return dict(ready=False, status='completed_manifest_missing', chunk_hashes_checked=False)
    try:
        manifest = inspect_bundle(bundle, verify_chunks=False)
        if not manifest.get('completed_at'):
            raise ValueError('Bundle completion timestamp is missing')
        expected = {part['path'] for item in manifest['files'] for part in item['chunks']}
        actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*.part') if p.is_file()}
        if actual != expected:
            return dict(ready=False, status='unregistered_or_missing_parts',
                        unregistered_parts=sorted(actual - expected), missing_parts=sorted(expected - actual),
                        chunk_hashes_checked=False)
        unfinished = [p.relative_to(bundle).as_posix() for p in bundle.rglob('*')
                      if p.is_file() and p.suffix in {'.tmp', '.building'}]
        if unfinished:
            return dict(ready=False, status='unfinished_bundle_files', paths=sorted(unfinished),
                        chunk_hashes_checked=False)
        if manifest.get('search_backend') == 'lucene' and not any(
                item['path'].startswith('index/') for item in manifest['files']):
            raise ValueError('Original Lucene index is missing')
        return dict(ready=True, status='manifest_and_part_sizes_match',
                    files=len(manifest['files']), parts=len(expected), stored_bytes=manifest['stored_bytes'],
                    chunk_hashes_checked=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(ready=False, status='invalid_bundle_structure', error_type=type(error).__name__,
                    chunk_hashes_checked=False)


def inspect_github_sync(root=PROJECT):
    root = Path(root).resolve()
    policy, authorized_paths = read_sync_policy(root)
    known_secrets = _known_local_credentials(root)
    probe = _git(["rev-parse", "--show-toplevel"], cwd=root, check=False)
    top = Path(probe.stdout.strip()).resolve() if probe.returncode == 0 else None
    if top is not None and os.path.normcase(str(top)) == os.path.normcase(str(root)):
        prefix = []
        mode = "existing_project_repository_read_only"
    else:
        # A separate empty index gives the exact untracked candidates selected by
        # Git's own ignore implementation, without initializing the user's project.
        git_dir = root / "output/github_sync_check" / (uuid.uuid4().hex + ".git")
        git_dir.parent.mkdir(parents=True, exist_ok=True)
        _git(["init", "--bare", "--quiet", str(git_dir)], cwd=root)
        prefix = ["--git-dir=" + str(git_dir), "--work-tree=" + str(root)]
        mode = "isolated_git_index_no_project_repository_created"
    listed = _git([*prefix, "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
    candidates = sorted(set(value for value in listed.stdout.split("\0") if value))
    missing_required = [relative for relative in policy.get("required_local_files", [])
                        if relative not in candidates or not (root / relative).is_file()]
    ignored = _git([*prefix, "check-ignore", "--no-index", "-z", "--stdin"], cwd=root,
                   input_text="\0".join(IGNORED_PROBES + SYNC_PROBES) + "\0", check=False)
    if ignored.returncode not in (0, 1):
        raise RuntimeError("Git ignore inspection failed: " + ignored.stderr.strip())
    ignored_paths = set(ignored.stdout.split("\0"))
    probes = [{"path": path, "ignored": path in ignored_paths, "expected_ignored": True} for path in IGNORED_PROBES]
    probes += [{"path": path, "ignored": path in ignored_paths, "expected_ignored": False} for path in SYNC_PROBES]
    forbidden, credentials, symlinks, oversized = [], [], [], []
    text_count = 0
    for relative in candidates:
        if forbidden_path(relative):
            forbidden.append(relative)
            continue
        path = root / relative
        if path.is_symlink():
            symlinks.append(relative)
            continue
        if path.is_file() and path.stat().st_size > MAX_REGULAR_GIT_FILE_BYTES:
            oversized.append(dict(path=relative, bytes=path.stat().st_size))
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text_count += 1
        for kind in credential_kinds(path, known_secrets):
            credentials.append({"path": relative, "kind": kind})
    bundle_required = 'resources/webshop/catalog_bundle/' in policy.get('required_sync_paths', [])
    bundle = catalog_bundle_status(root) if bundle_required else dict(ready=True, status='not_required')
    return {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
            "repository_mode": mode, "candidate_count": len(candidates),
            "private_repo_required": policy["private_repo_required"],
            "repository_visibility": policy["repository_visibility"],
            "visibility_confirmation": policy.get("visibility_confirmation"),
            "remote_visibility_verified": False, "scope": "local_candidates_only",
            "user_authorized_credentials": sorted(authorized_paths),
            "authorized_credential_candidates": [],
            "known_local_credentials_checked": bool(known_secrets),
            "candidate_paths": candidates, "text_files_scanned": text_count,
            "missing_required_candidates": missing_required,
            "catalog_bundle": bundle,
            "oversized_regular_git_candidates": oversized,
            "max_regular_git_file_bytes": MAX_REGULAR_GIT_FILE_BYTES,
            "ignore_probes": probes, "forbidden_candidates": forbidden,
            "credential_candidates": credentials, "symlink_candidates": symlinks,
            "passed": all(item["ignored"] == item["expected_ignored"] for item in probes) and bundle['ready']
                      and not (forbidden or credentials or symlinks or missing_required or oversized),
            "staged_committed_or_pushed": False,
            "credential_values_included": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "output/deployment/github_sync_check.json")
    args = parser.parse_args(argv)
    report = inspect_github_sync()
    output = args.output if args.output.is_absolute() else PROJECT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "candidate_paths"}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
