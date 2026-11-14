"""Tiny deterministic embeddings only; no images, models or shared pool read."""
import copy
import json

import numpy as np
import pytest

from src.visual.memory_vector_cache import MemoryVectorCache


def identity():
    return dict(schema=1, model_sha256='frozen-model', image_decoder_sha256='decoder',
                policy_sha256='public-product-key', image_weight=0.5,
                image_sha256='source-image', text_sha256='full-product-key')


def vector():
    value = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    return value / np.linalg.norm(value)


def test_cache_preserves_float32_bits_without_renormalizing(tmp_path):
    cache = MemoryVectorCache(tmp_path)
    value = np.asarray([np.nextafter(np.float32(1), np.float32(0)), 0., 0.], dtype=np.float32)
    # This valid unit-length float32 value would change if normalized twice.
    assert not np.array_equal(value, value / np.linalg.norm(value))
    assert cache.load(identity()) is None
    cache.save(identity(), value, {'full_text': True})
    restored, audit = cache.load(identity())
    assert restored.dtype == np.float32 and np.array_equal(restored, value)
    assert restored.tobytes() == value.tobytes()
    assert audit == {'full_text': True}
    assert cache.stats == dict(hits=1, misses=1, writes=1, rejected=0)
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('field', ['model_sha256', 'image_decoder_sha256', 'policy_sha256',
                                  'image_sha256', 'text_sha256', 'image_weight', 'schema'])
def test_changed_encoding_input_misses_instead_of_remapping(tmp_path, field):
    cache = MemoryVectorCache(tmp_path)
    original = identity()
    cache.save(original, vector(), None)
    changed = copy.deepcopy(original)
    changed[field] = 'changed' if isinstance(changed[field], str) else 0.25
    assert cache.load(changed) is None
    assert cache.stats['rejected'] == 0
    assert cache.load(original) is not None


@pytest.mark.parametrize('change', ['identity', 'checksum', 'schema', 'dtype', 'nonfinite', 'nonunit', 'malformed'])
def test_corruption_is_rejected_and_never_becomes_a_hit(tmp_path, change):
    cache = MemoryVectorCache(tmp_path)
    cache.save(identity(), vector(), None)
    path = next(tmp_path.glob('*.json'))
    payload = json.loads(path.read_text(encoding='utf-8'))
    if change == 'identity': payload['identity']['model_sha256'] = 'old-other-model'
    elif change == 'checksum': payload['vector_sha256'] = 'invalid'
    elif change == 'schema': payload['schema_version'] = -1
    elif change == 'dtype': payload['dtype'] = 'float16'
    elif change == 'nonfinite': payload['vector'][0] = float('nan')
    elif change == 'nonunit': payload['vector'] = [2.0, 0.0, 0.0]
    path.write_text('{' if change == 'malformed' else json.dumps(payload), encoding='utf-8')
    assert cache.load(identity()) is None
    assert cache.stats == dict(hits=0, misses=1, writes=1, rejected=1)


@pytest.mark.parametrize('bad', [[0., 0.], [2., 0.], [float('nan'), 0.],
                                [float('inf'), 0.], [[1., 0.]], []])
def test_invalid_vectors_cannot_be_saved(tmp_path, bad):
    cache = MemoryVectorCache(tmp_path)
    with pytest.raises(ValueError):
        cache.save(identity(), bad, None)
    assert cache.stats['writes'] == 0
    assert not list(tmp_path.glob('*.json'))

