"""Offline deployment checks. Never load model weights or run image encoders."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT, EVALUATION_BASE, MODEL_ROOT, RESOURCES
from src.model_registry import model_specs, verify_locked_snapshot_files


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def bounded_path(root, relative):
    if not isinstance(relative, str) or re.match(r"^[A-Za-z]:[\\/]", relative):
        raise ValueError("Non-portable resource path: " + str(relative))
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Resource path escapes its root: " + relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_resources(base=EVALUATION_BASE):
    base = Path(base).resolve()
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    checks = 0
    def bound(relative, expected=None):
        nonlocal checks
        path = bounded_path(base, relative)
        if expected and sha256(path) != expected:
            raise ValueError("Resource content hash mismatch: " + relative)
        checks += 1
        return path
    pool_meta = manifest["shared_memory"]
    pool_path = bound(pool_meta["pool_file"], pool_meta["pool_sha256"])
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    entries = pool["entries"]
    if len(entries) != 189 or len({entry["Id"] for entry in entries}) != 189:
        raise ValueError("Expected 189 unique shared entries")
    for entry in entries:
        image = entry["PrimaryImage"]
        path = bounded_path(pool_path.parent, image["LocalPath"])
        if image.get("SHA256") and sha256(path) != image["SHA256"]:
            raise ValueError("Primary image hash mismatch: " + entry["Id"])
        checks += 1
        if entry.get("RecordKind") == "simulated_gallery" and (entry.get("ObservedPurchase") or entry.get("Completed")):
            raise ValueError("Simulated reference was marked purchased: " + entry["Id"])
    metadata = manifest["public_metadata"]
    bound(metadata["path"], metadata["sha256"])
    bound(metadata["review_path"], metadata["review_sha256"])
    protocol = manifest["evaluation_protocol"]
    bound(protocol["encoder_config"], protocol["encoder_config_sha256"])
    query_count = 0
    groups = manifest["groups"]
    if len(groups) != 9:
        raise ValueError("Expected nine frozen groups")
    for key, spec in groups.items():
        definition = json.loads(bound(spec["definition"], spec.get("definition_sha256")).read_text(encoding="utf-8"))
        bound(spec["features"], spec["features_sha256"])
        query = json.loads(bound(spec["query_manifest"], spec["query_manifest_sha256"]).read_text(encoding="utf-8"))
        if len(query["instructions"]) != 20 or query["target_public_id"] != spec["target_id"]:
            raise ValueError("Query binding mismatch: " + key)
        query_count += len(query["instructions"])
        for side in ("source", "target"):
            image_name = definition[side]["image"]
            bound(spec["directory"] + "/" + image_name, spec[side + "_sha256"])
        for method, expected in spec.get("method_config_sha256", {}).items():
            bound(spec[method + "_config"], expected)
    shared_image_count = sum(p.is_file() for p in (pool_path.parent / 'images').glob('*'))
    if pool_meta.get('image_file_count', shared_image_count) != shared_image_count:
        raise ValueError('Shared image inventory count differs from the frozen manifest')
    return {"entries": len(entries), "groups": len(groups), "queries": query_count,
            "shared_image_files": shared_image_count,
            "path_and_hash_checks": checks, "model_encoding_performed": False,
            "selection_exceptions": [key for key, item in groups.items() if not item.get("passes_calibrated_threshold", False)]}



def check_code(root=PROJECT):
    root = Path(root).resolve()
    checked = 0
    for directory in ("src", "tools"):
        for path in sorted((root / directory).rglob("*.py")):
            if "__pycache__" not in path.parts:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                checked += 1
    if not checked:
        raise ValueError("Project source files are absent")
    manifest = json.loads((root / "resources/model_manifest.json").read_text(encoding="utf-8"))
    if not {"sam2", "editor", "purifier", "selection", "eval_g14", "eval_gme"} <= set(manifest["models"]):
        raise ValueError("Production model manifest is incomplete")
    return {"python_files_parsed": checked, "model_manifest_entries": len(manifest["models"]), "private_resources_read": False}


def check_webshop_resources(root=PROJECT):
    """Read manifests, small source files and sizes; never start the shop or hash the large catalog."""
    from src.webshop.bundle import inspect_bundle
    root = Path(root).resolve()
    base = root / 'resources/webshop'
    source, bundle, catalog = base / 'source', base / 'catalog_bundle', base / 'catalog'
    vendor = json.loads((source / 'VENDOR_MANIFEST.json').read_text(encoding='utf-8'))
    required = {'LICENSE.md', 'web_agent_site/app.py', 'web_agent_site/engine/goal.py'}
    names = {item['path'] for item in vendor['files']}
    if not required <= names:
        raise ValueError('Vendored shop source is incomplete')
    for item in vendor['files']:
        path = bounded_path(source, item['path'])
        if path.stat().st_size != item['vendored_bytes'] or sha256(path) != item['vendored_sha256']:
            raise ValueError('Vendored source content changed: ' + item['path'])
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    manifest = inspect_bundle(bundle)
    if (bundle / 'manifest.partial.json').exists():
        raise ValueError('Shop recovery bundle is still being built')
    expected_parts = {part['path'] for item in manifest['files'] for part in item['chunks']}
    actual_parts = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*.part')}
    if expected_parts != actual_parts:
        raise ValueError('Shop recovery bundle has missing or orphaned chunks')
    materialized = (catalog / 'RESTORED.json').is_file()
    if materialized:
        restored = json.loads((catalog / 'RESTORED.json').read_text(encoding='utf-8'))
        if restored['catalog_bundle_manifest_sha256'] != sha256(bundle / 'manifest.json'):
            raise ValueError('Recovered shop catalog belongs to another bundle')
        for item in manifest['files']:
            path = bounded_path(catalog, item['path'])
            if item['path'] not in ('data_manifest.json', 'index_manifest.json') and path.stat().st_size != item['bytes']:
                raise ValueError('Recovered shop catalog has an incomplete file: ' + item['path'])
        data = json.loads((catalog / 'data_manifest.json').read_text(encoding='utf-8'))
        index = json.loads((catalog / 'index_manifest.json').read_text(encoding='utf-8'))
        if data['bundle_id'] != manifest['bundle_id'] or data['products'] != manifest['products']:
            raise ValueError('Recovered shop catalog identity differs')
        if data.get('historical_build_inputs_required_at_runtime') is not False:
            raise ValueError('Historical raw source paths must be marked provenance-only')
        if index.get('backend', 'lucene') == 'lucene' and index.get('path') != 'index':
            raise ValueError('Shop search index must use its project-local relative path')
    return dict(source_files=len(vendor['files']), catalog_files=len(manifest['files']),
                products=manifest['products'], raw_bytes=manifest['raw_bytes'], stored_bytes=manifest['stored_bytes'],
                chunks=len(expected_parts), materialized=materialized,
                status='ready' if materialized else 'bundle_ready_restore_required',
                large_file_hashes_checked=False, models_loaded=False, shop_started=False,
                shopping_model_calls=0, old_sibling_directories_required=False)


def inspect_runtime(include_hardware=False):
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "diffusers",
        "accelerate",
        "open-clip-torch",
        "safetensors",
        "huggingface-hub",
        "rapidocr-onnxruntime",
        "scikit-image",
        "scipy",
        "sentence-transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": packages,
              "disk_free_gib": round(shutil.disk_usage(PROJECT).free / 1024 ** 3, 2)}
    if include_hardware:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            bf16 = torch.cuda.is_bf16_supported()
            result.update(
                cuda_device=props.name,
                vram_gib=round(props.total_memory / 1024 ** 3, 2),
                bf16_supported=bf16,
                recommended_dtype="bfloat16" if bf16 else "float16",
                fp32_fallback_compatible=True,
            )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true", help="Check code/runtime/models before private evaluation inputs have been synchronized")
    parser.add_argument("--hardware", action="store_true", help="Read CUDA properties only; do not load models")
    parser.add_argument("--require-models", action="store_true", help="Require production snapshot manifests on the remote server")
    parser.add_argument("--verify-model-hashes", action="store_true", help="Read and hash prepared model files; intended for remote use")
    parser.add_argument("--webshop", action="store_true", help="Check optional project-local shop source and recovery bundle without loading models or starting services")
    parser.add_argument("--output", type=Path)
    from src.visual.training_surrogates import add_surrogate_arguments,select_training_surrogates,training_model_keys
    add_surrogate_arguments(parser)
    args = parser.parse_args(argv)
    report = {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(), "code": check_code(), "resources": ({"status": "skipped_code_only", "private_resources_read": False} if args.code_only else check_resources()), "runtime": inspect_runtime(args.hardware)}
    registry=model_specs()
    selection=select_training_surrogates('custom' if args.surrogate_config else (args.surrogate_set or 'lucid'),args.surrogate_config)
    training_keys=training_model_keys(selection,registry)
    if selection.profile!='custom' and None in training_keys:
        raise ValueError('内置训练代理模型没有完整登记在模型准备清单中')
    required_training_keys={key for spec,key in zip(selection.specs,training_keys)
                            if not (selection.profile=='custom' and spec.weight_path)}
    report['training_surrogates']=selection.audit()
    models = {}
    optional_skipped = []
    lock_path = RESOURCES / 'model_lock.json'
    locked = json.loads(lock_path.read_text(encoding='utf-8')).get('models', {}) if lock_path.is_file() else {}
    for key, spec in registry.items():
        if spec["scope"] == "optional":
            optional_skipped.append(key)
            continue
        if spec['scope']=='encoders' and key not in required_training_keys:
            continue
        destination = MODEL_ROOT / key
        record = destination / "download_manifest.json"
        entry = {"present": record.is_file(), "repo_id": spec["repo_id"], "role": spec["role"]}
        if spec.get("allow_patterns"):
            entry["allow_patterns"] = spec["allow_patterns"]
        models[key] = entry
        if args.require_models and not record.is_file():
            raise FileNotFoundError("Prepare required model snapshot: " + key)
        if record.is_file() and (args.require_models or args.verify_model_hashes):
            frozen = json.loads(record.read_text(encoding="utf-8"))
            if frozen.get("repo_id") != spec["repo_id"] or not frozen.get("revision"):
                raise ValueError("Prepared snapshot identity mismatch: " + key)
            if key in locked and (
                locked[key].get('repo_id') != frozen['repo_id']
                or locked[key].get('revision') != frozen['revision']
                or not verify_locked_snapshot_files(key, spec, frozen.get('files', {}), locked[key].get('files', {}))
            ):
                raise ValueError('Prepared model differs from the repository recovery lock: ' + key)
            if not any(name.endswith(".safetensors") for name in frozen.get("files", {})):
                raise ValueError("Prepared snapshot contains no weights: " + key)
            for relative, expected in frozen["files"].items():
                model_file = bounded_path(destination, relative)
                if args.verify_model_hashes and sha256(model_file) != expected:
                    raise ValueError("Prepared model hash mismatch: " + key + "/" + relative)
            models[key]["snapshot_files_verified"] = len(frozen["files"])
            models[key]["content_hashes_verified"] = args.verify_model_hashes
    report["models"] = models
    report["optional_models_skipped"] = optional_skipped
    custom_models=[]
    if selection.profile=='custom':
        for spec,key in zip(selection.specs,training_keys):
            if spec.weight_path or key is None:
                weight=Path(spec.weight_path) if spec.weight_path else None
                weight=(weight if weight.is_absolute() else MODEL_ROOT/weight) if weight else None
                present=bool(weight and weight.is_file())
                if args.require_models and not present:
                    raise FileNotFoundError('Prepare explicit custom training weights: '+spec.name)
                custom_models.append(dict(name=spec.name,pretrained=spec.pretrained,present=present,
                    content_sha256=sha256(weight) if present and args.verify_model_hashes else None))
        report['custom_training_models']=custom_models
    if args.webshop:
        report['webshop'] = check_webshop_resources()
    report["remote_model_execution_verified"] = False
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
