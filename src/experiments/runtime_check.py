"""只检查依赖版本、路径与硬件；不加载真实模型或联网。"""
import importlib.metadata
import json
import platform
from src.paths import PROJECT, MODEL_ROOT, SHARED_MEMORY
from src.model_registry import model_specs


def doctor(device='cuda'):
    import torch
    packages = {}
    for name in ('torch', 'torchvision', 'transformers', 'diffusers', 'accelerate',
                 'open-clip-torch', 'sentence-transformers', 'rapidocr-onnxruntime'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    models = {}
    for key, spec in model_specs().items():
        root = MODEL_ROOT / key
        models[key] = dict(repo_id=spec['repo_id'], directory_exists=root.is_dir(),
                          has_manifest=(root / 'download_manifest.json').is_file(),
                          role=spec['role'], loaded=False)
    result = dict(project=str(PROJECT), platform=platform.platform(), requested_device=device,
                  cuda_available=torch.cuda.is_available(), packages=packages,
                  local_models=models, shared_pool_configured=SHARED_MEMORY.is_file(),
                  real_model_execution_verified=False, external_model_calls=0)
    if torch.cuda.is_available():
        result['device_name'] = torch.cuda.get_device_name()
        result['vram_gib'] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        result['bfloat16_supported'] = torch.cuda.is_bf16_supported()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
