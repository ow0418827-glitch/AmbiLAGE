"""Exact-content cache for public, fixed negative image embeddings only."""
from pathlib import Path
import json
import os
import platform
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from src.paths import PROJECT
from src.visual.io import load_image, pil_to_tensor, save_json, sha256_file
from .targets import content_sha256

CACHE_KIND = 'visual_inception_fixed_public_image_vectors_v1'


def _canonical_json(value):
    """Normalize container types (e.g. tuple -> list) to match JSON round-trip values."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _runtime_identity():
    return dict(torch_version=str(torch.__version__), numpy_version=np.__version__,
        python_version=platform.python_version(), cuda_version=torch.version.cuda,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), vector_dtype='float32')


def candidate_cache_identity(model, candidates, *, bundle_sha256, fixture=False):
    return _canonical_json(dict(kind=CACHE_KIND, bundle_sha256=bundle_sha256,
        model=model.snapshot(), device=str(model.device), test_fixture=bool(fixture),
        candidates=[dict(id=row['id'], image_sha256=row['image_sha256']) for row in candidates],
        encoder_implementation_sha256=sha256_file(PROJECT / 'src/visual/encoders/backbone.py'),
        pixel_conversion_implementation_sha256=sha256_file(PROJECT / 'src/visual/io.py'),
        cache_implementation_sha256=sha256_file(Path(__file__)),
        runtime=_runtime_identity()))


def validate_candidate_cache(record, expected_identity=None):
    """Validate on-disk bytes and return CPU vectors; no unsafe deserialization."""
    if not isinstance(record, dict) or record.get('kind') != CACHE_KIND:
        raise ValueError('固定公开候选向量缓存身份无效')
    metadata_path, vector_path = Path(record['metadata_path']), Path(record['vector_path'])
    if sha256_file(metadata_path) != record.get('metadata_sha256'):
        raise ValueError('候选向量缓存元数据摘要改变')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    identity = metadata.get('identity')
    normalized_expected = _canonical_json(expected_identity) if expected_identity is not None else None
    if (not isinstance(identity, dict) or metadata.get('kind') != CACHE_KIND
            or record.get('cache_key') != content_sha256(identity)
            or metadata.get('cache_key') != record['cache_key']
            or metadata_path.parent.name != record['cache_key']
            or vector_path.resolve() != (metadata_path.parent / 'vectors.npy').resolve()):
        raise ValueError('候选向量缓存与模型、数据或路径绑定不一致')
    if normalized_expected is not None and identity != normalized_expected:
        raise ValueError('候选缓存不能跨模型、文字、图片、预处理或数值运行时复用')
    if identity != _canonical_json(record.get('identity')):
        raise ValueError('记录中的候选缓存身份与落盘元数据不一致')
    if sha256_file(vector_path) != metadata.get('vector_sha256') or metadata['vector_sha256'] != record.get('vector_sha256'):
        raise ValueError('候选向量缓存文件摘要改变')
    with vector_path.open('rb') as handle:
        vectors = np.load(handle, allow_pickle=False)
    expected_count = len(identity.get('candidates', []))
    if (vectors.dtype != np.float32 or vectors.ndim != 2 or vectors.shape[0] != expected_count
            or vectors.shape[1] < 1 or list(vectors.shape) != metadata.get('shape')
            or list(vectors.shape) != record.get('shape') or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)):
        raise ValueError('候选缓存向量的形状、数值或单位范数无效')
    return torch.from_numpy(vectors.copy())


def encode_candidate_cache(model, candidates, cache_dir, *, bundle_sha256, fixture=False):
    if not candidates:
        raise ValueError('独立公开代理库不能为空')
    for candidate in candidates:
        if sha256_file(Path(candidate['image_path'])) != candidate['image_sha256']:
            raise ValueError('公开候选图片字节与冻结清单不一致')
    identity = candidate_cache_identity(model, candidates, bundle_sha256=bundle_sha256, fixture=fixture)
    key = content_sha256(identity)
    directory = Path(cache_dir).resolve() / key
    metadata_path, vector_path = directory / 'cache.json', directory / 'vectors.npy'
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        record = dict(kind=CACHE_KIND, cache_key=key, identity=identity,
            metadata_path=str(metadata_path), metadata_sha256=sha256_file(metadata_path),
            vector_path=str(vector_path), vector_sha256=metadata.get('vector_sha256'),
            shape=metadata.get('shape'), cache_hit=True)
        vectors = validate_candidate_cache(record, identity)
        return vectors.to(model.device), record
    if vector_path.exists():
        raise ValueError('候选向量缓存不完整；请核验后使用新的缓存目录')
    rows = []
    for candidate in candidates:
        path = Path(candidate['image_path'])
        if sha256_file(path) != candidate['image_sha256']:
            raise ValueError('公开候选图片字节与冻结清单不一致')
        image = load_image(path)
        if fixture and max(image.size) > 64:
            raise ValueError('微型候选图片长边不能超过六十四')
        with torch.no_grad():
            feature = model.encode_image(pil_to_tensor(image, model.device)).global_embedding
            if feature.ndim != 2 or feature.shape[0] != 1 or not torch.isfinite(feature).all() or torch.linalg.vector_norm(feature) <= 0:
                raise ValueError('公开候选图片编码无效')
            rows.append(F.normalize(feature.float(), dim=-1).cpu())
    vectors = torch.cat(rows, dim=0).numpy().astype(np.float32)
    directory.mkdir(parents=True, exist_ok=True)
    handle_id, temporary = tempfile.mkstemp(prefix='.vectors.', suffix='.npy', dir=directory)
    try:
        with os.fdopen(handle_id, 'wb') as handle:
            np.save(handle, vectors, allow_pickle=False)
        os.replace(temporary, vector_path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    metadata = dict(kind=CACHE_KIND, cache_key=key, identity=identity,
        shape=list(vectors.shape), vector_sha256=sha256_file(vector_path))
    save_json(metadata_path, metadata)
    record = dict(kind=CACHE_KIND, cache_key=key, identity=identity, shape=list(vectors.shape),
        metadata_path=str(metadata_path), metadata_sha256=sha256_file(metadata_path),
        vector_path=str(vector_path), vector_sha256=metadata['vector_sha256'], cache_hit=False)
    checked = validate_candidate_cache(record, identity)
    return checked.to(model.device), record
