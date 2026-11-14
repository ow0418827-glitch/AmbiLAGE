"""Prepare explicit production model snapshots; default execution only prints a plan."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_registry import model_specs, verify_locked_snapshot_files
from src.paths import MODEL_ROOT, RESOURCES

MODEL_LOCK_PATH = RESOURCES / 'model_lock.json'


def read_model_lock():
    if not MODEL_LOCK_PATH.is_file():
        return {"schema_version": 1, "models": {}}
    value = json.loads(MODEL_LOCK_PATH.read_text(encoding='utf-8'))
    if value.get('schema_version') != 1 or not isinstance(value.get('models'), dict):
        raise ValueError('Invalid model recovery lock')
    return value


def download_model(repo, local_dir, allow_patterns=None, endpoint=None, revision=None):
    # Pass endpoint explicitly; changing HF_ENDPOINT after importing the hub is insufficient.
    from huggingface_hub import snapshot_download
    return snapshot_download(
        repo_id=repo, local_dir=str(local_dir), revision=revision,
        endpoint=endpoint or os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        allow_patterns=allow_patterns or ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.yaml", "*.yml", "LICENSE*", "README.md"],
        ignore_patterns=["*fp8*", "*FP8*", "*fp16*", "*.gguf", "*.bin", "*.ckpt", "*.pth", "ae.safetensors", "flux1-kontext-dev.safetensors"],
    )


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("generation", "encoders", "evaluators", "selection", "all"), default="generation")
    parser.add_argument("--models", nargs="+", help="Select model manifest keys explicitly, including optional alternatives")
    parser.add_argument("--endpoint", "--mirror", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--download", action="store_true", help="Actually download snapshots; omitted means plan only")
    parser.add_argument("--revision", help="Pin the same explicit revision for the selected model; use with a single --models key")
    from src.visual.training_surrogates import add_surrogate_arguments,select_training_surrogates,training_model_keys
    add_surrogate_arguments(parser)
    args = parser.parse_args(argv)
    specs = model_specs()
    lock = read_model_lock()
    selected=None
    if args.models and (args.surrogate_set is not None or args.surrogate_config is not None):
        parser.error('显式模型编号与训练代理方案请选择一种方式')
    if not args.models and args.scope in ('encoders','all'):
        selected=select_training_surrogates('custom' if args.surrogate_config else (args.surrogate_set or 'lucid'),args.surrogate_config)
        training_keys=training_model_keys(selected,specs)
        if None in training_keys:
            parser.error('自定义代理包含未登记仓库，请先显式准备本地权重；其他模型可通过 --models 单独准备')
        keys=[key for key,spec in specs.items() if
              (key in training_keys and spec['scope']=='encoders') or
              (args.scope=='all' and spec['scope'] not in ('optional','encoders'))]
    else:
        if args.surrogate_set is not None or args.surrogate_config is not None:
            parser.error('训练代理方案用于 --scope encoders 或 --scope all')
        keys=args.models or [key for key,spec in specs.items() if spec['scope']==args.scope]
    if any(key not in specs for key in keys):
        parser.error("Unknown model key")
    if args.revision and len(keys) != 1:
        parser.error("--revision requires exactly one selected model")
    plan = [{
        "key": key,
        "repo_id": specs[key]["repo_id"],
        "local_dir": str(MODEL_ROOT / key),
        "scope": specs[key]["scope"],
        "role": specs[key]["role"],
        **({"allow_patterns": specs[key]["allow_patterns"]} if specs[key].get("allow_patterns") else {}),
        "pinned_revision": args.revision or lock['models'].get(key, {}).get('revision'),
    } for key in keys]
    print(json.dumps({"download": args.download, "endpoint": args.endpoint, "models": plan,
                      "training_surrogates":selected.audit() if selected else None}, ensure_ascii=False, indent=2), flush=True)
    if not args.download:
        return 0
    from huggingface_hub import HfApi
    api = HfApi(endpoint=args.endpoint)
    for item in plan:
        # Freeze the actual repository commit and retain it alongside content hashes.
        previous = lock['models'].get(item['key'])
        if previous and previous.get('repo_id') != item['repo_id']:
            raise ValueError('Model identity differs from the recovery lock: ' + item['key'])
        revision = api.model_info(item["repo_id"], revision=item['pinned_revision']).sha
        destination = Path(item["local_dir"])
        patterns = specs[item["key"]].get("allow_patterns")
        if patterns is None and (item["key"].startswith("clip_") or item["key"] in {"siglip_so400m", "eval_g14"}):
            patterns = ["open_clip*.safetensors", "*.json", "*.txt", "*.model", "LICENSE*", "README.md"]
        download_model(item["repo_id"], destination, allow_patterns=patterns, endpoint=args.endpoint, revision=revision)
        files = {p.relative_to(destination).as_posix(): _sha(p) for p in sorted(destination.rglob("*"))
                 if p.is_file() and ".cache" not in p.relative_to(destination).parts and p.name != "download_manifest.json"}
        if not any(name.endswith(".safetensors") for name in files):
            raise RuntimeError("Snapshot contains no model weights: " + item["key"])
        if previous and args.revision is None:
            if revision != previous['revision'] or not verify_locked_snapshot_files(
                item["key"], specs[item["key"]], files, previous['files']
            ):
                raise ValueError('Restored model files differ from the frozen recovery lock: ' + item['key'])
        manifest = dict(item, revision=revision, endpoint=args.endpoint, files=files)
        (destination / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lock['models'][item['key']] = dict(repo_id=item['repo_id'], revision=revision, files=files)
        MODEL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = MODEL_LOCK_PATH.with_name('.model-lock.tmp')
        temporary.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(MODEL_LOCK_PATH)
        print("Prepared " + item["key"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
