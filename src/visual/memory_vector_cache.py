"""Content-addressed memory embeddings; no model loading or identity remapping.

The caller supplies the complete encoding identity. A cache entry is reusable
only for precisely that identity; whole-bank membership is intentionally absent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import uuid

import numpy as np


CACHE_RECORD_SCHEMA = 1
UNIT_NORM_TOLERANCE = 2e-5


def _identity_bytes(identity):
    if not isinstance(identity, dict) or not identity:
        raise ValueError('Memory vector identity must be a nonempty object')
    return json.dumps(identity, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _vector_digest(vector):
    # A fixed byte order gives the same digest across machines and JSON reloads.
    return hashlib.sha256(vector.astype('<f4', copy=False).tobytes()).hexdigest()


def _validate_vector(vector):
    value = np.asarray(vector, dtype=np.float32)
    if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError('Cached embedding must be a finite nonempty vector')
    norm = float(np.linalg.norm(value.astype(np.float64)))
    if not np.isfinite(norm) or abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
        raise ValueError('Cached embedding must already have unit norm')
    return value


class MemoryVectorCache:
    """Strict per-entry cache returning original float32 values unchanged.

    Corrupt files count as rejected misses and can be rebuilt by the caller.
    This class never substitutes an identity or normalizes a loaded vector.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self._stats = dict(hits=0, misses=0, writes=0, rejected=0)

    @property
    def stats(self):
        return dict(self._stats)

    def report(self):
        return dict(directory=str(self.directory), schema_version=CACHE_RECORD_SCHEMA,
                    **self._stats)

    def _location(self, identity):
        encoded = _identity_bytes(identity)
        key = hashlib.sha256(encoded).hexdigest()
        return self.directory / (key + '.json'), encoded, key

    def load(self, identity):
        path, encoded, key = self._location(identity)
        if not path.is_file():
            self._stats['misses'] += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if (not isinstance(payload, dict)
                    or payload.get('schema_version') != CACHE_RECORD_SCHEMA
                    or payload.get('identity_sha256') != key
                    or _identity_bytes(payload.get('identity')) != encoded
                    or payload.get('dtype') != 'float32'):
                raise ValueError('Memory vector cache identity or schema changed')
            raw = payload.get('vector')
            if (not isinstance(raw, list)
                    or any(type(item) not in (int, float) for item in raw)):
                raise ValueError('Memory vector cache has an invalid numeric payload')
            vector = _validate_vector(raw)
            if payload.get('vector_sha256') != _vector_digest(vector):
                raise ValueError('Memory vector cache checksum changed')
        except (OSError, ValueError, TypeError, OverflowError):
            self._stats['misses'] += 1
            self._stats['rejected'] += 1
            return None
        self._stats['hits'] += 1
        return vector, payload.get('encoding_audit')

    def save(self, identity, vector, audit):
        path, encoded, key = self._location(identity)
        vector = _validate_vector(vector)
        payload = dict(schema_version=CACHE_RECORD_SCHEMA,
                       identity=json.loads(encoded), identity_sha256=key,
                       dtype='float32', vector=vector.tolist(),
                       vector_sha256=_vector_digest(vector), encoding_audit=audit)
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + '\n'
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / ('.write-' + uuid.uuid4().hex[:16] + '.tmp')
        try:
            temporary.write_text(serialized, encoding='utf-8', newline='\n')
            temporary.replace(path)
        finally:
            if temporary.is_file():
                temporary.unlink()
        self._stats['writes'] += 1

