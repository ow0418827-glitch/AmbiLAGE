"""Build a portable source archive with authorized input resources, excluding runtime state."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT
from tools.preflight import check_resources

DIRECTORIES = ("src", "tools", "docs", "requirements", "resources", "data/public")
ROOT_FILES = ("README.md", "AGENTS.md", "pyproject.toml", ".gitignore", "run_pipeline.sh", "run_pipeline.ps1")
EXCLUDED_PARTS = {"__pycache__", ".cache", ".pytest_cache", ".git", ".venv", "models", "output", "dist", "history", "migration_legacy"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pem", ".key", ".safetensors", ".bin", ".ckpt", ".pt", ".pth", ".gguf"}


def source_files(root=PROJECT):
    root = Path(root).resolve()
    candidates = [root / name for name in ROOT_FILES]
    for directory in DIRECTORIES:
        candidates.extend((root / directory).rglob("*"))
    result = []
    for path in candidates:
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError("Source archives must not contain symlinks: " + str(relative))
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES or (path.name.startswith(".env") or path.name.endswith(".local.json")):
            continue
        if any(word in path.name.lower() for word in ("credentials", "secret", "access_token", "private_key")):
            continue
        if path.suffix.lower() in {".json", ".py", ".yaml", ".yml", ".md", ".txt"}:
            if re.search(r"(?i)(sk-[a-z0-9_-]{16,}|Bearer [a-z0-9._-]{20,})", path.read_text(encoding="utf-8")):
                raise ValueError("Possible embedded credential in source archive: " + str(relative))
        result.append(path)
    return sorted(set(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "dist/AmbiLAGE-source.zip")
    args = parser.parse_args(argv)
    resource_report = check_resources()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError("Choose a fresh archive path: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = source_files()
    hashes = {path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "files": hashes,
                "resource_validation": resource_report, "weights_and_environment_included": False,
                "private_runtime_logs_included": False}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, "AmbiLAGE/" + path.relative_to(PROJECT).as_posix())
        archive.writestr("AmbiLAGE/SOURCE_ARCHIVE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    archive_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(archive_hash + "  " + destination.name + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(destination), "sha256": archive_hash, "file_count": len(files), "bytes": destination.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
