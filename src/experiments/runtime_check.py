"""只检查依赖版本、路径与硬件；不加载真实模型或联网。"""
import importlib.metadata
import json
import platform
from src.paths import PROJECT, MODEL_ROOT, SHARED_MEMORY
from src.model_registry import model_specs


def doctor(device='cuda',*,surrogate_set='lucid',surrogate_config=None):
    import torch
    from src.visual.training_surrogates import select_training_surrogates, training_model_keys
    selection=select_training_surrogates('custom' if surrogate_config else surrogate_set,surrogate_config)
    packages = {}
    for name in (
        'torch',
        'torchvision',
        'transformers',
        'diffusers',
        'accelerate',
        'open-clip-torch',
        'safetensors',
        'huggingface-hub',
        'rapidocr-onnxruntime',
        'scikit-image',
        'scipy',
        'sentence-transformers',
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    specs = model_specs()
    required_encoders = set(training_model_keys(selection, specs))
    models = {}
    for key, spec in specs.items():
        root = MODEL_ROOT / key
        required = spec['scope'] not in ('optional', 'encoders') or key in required_encoders
        models[key] = dict(
            repo_id=spec['repo_id'],
            scope=spec['scope'],
            required=required,
            optional=spec['scope'] == 'optional',
            directory_exists=root.is_dir(),
            has_manifest=(root / 'download_manifest.json').is_file(),
            role=spec['role'],
            **({'allow_patterns': spec['allow_patterns']} if spec.get('allow_patterns') else {}),
            loaded=False,
        )
    result = dict(project=str(PROJECT), platform=platform.platform(), requested_device=device,
                  training_surrogates=selection.audit(),
                  cuda_available=torch.cuda.is_available(), packages=packages,
                  local_models=models, shared_pool_configured=SHARED_MEMORY.is_file(),
                  real_model_execution_verified=False, external_model_calls=0)
    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        result['device_name'] = torch.cuda.get_device_name()
        result['vram_gib'] = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        result['bfloat16_supported'] = bf16
        result['recommended_dtype'] = 'bfloat16' if bf16 else 'float16'
        result['fp32_fallback_compatible'] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
