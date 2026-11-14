"""Project-local, chunked recovery of the existing full disk catalog.

Only explicitly selected immutable input files are packed. Restoring never
rebuilds the database or search index and never changes their bytes or ranking.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from src.paths import PROJECT, RESOURCES

DEFAULT_BUNDLE = RESOURCES / 'webshop/catalog_bundle'
DEFAULT_CATALOG = RESOURCES / 'webshop/catalog'
CHUNK_BYTES = 64 * 1024 * 1024
READ_BYTES = 8 * 1024 * 1024


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(READ_BYTES), b''):
            value.update(block)
    return value.hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.write-' + uuid.uuid4().hex[:16] + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Bundle paths must be relative POSIX paths')
    path = (Path(root).resolve() / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or path == Path(root).resolve():
        raise ValueError('Bundle path escapes its root')
    return path


def _destination(path):
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT.resolve()) or path == PROJECT.resolve():
        raise ValueError('Catalog writes must stay inside the current project')
    return path


def _acquire_restore_lock(path):
    """OS locks release after process termination; a leftover file is harmless."""
    handle = Path(path).open('a+b')
    try:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('Another process is actively restoring this catalog') from None
    return handle


def _release_restore_lock(handle):
    try:
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


class _ChunkWriter:
    def __init__(self, root, prefix, limit):
        self.root, self.prefix, self.limit = root, prefix, limit
        self.buffer, self.parts, self.total = bytearray(), [], 0

    def write(self, value):
        size = len(value)
        self.total += size
        while value:
            count = min(self.limit - len(self.buffer), len(value))
            self.buffer.extend(value[:count])
            value = value[count:]
            if len(self.buffer) == self.limit:
                self._part()
        return size

    def _part(self):
        relative = f'chunks/{self.prefix}.{len(self.parts):05d}.part'
        path = _inside(self.root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = bytes(self.buffer)
        path.write_bytes(data)
        self.parts.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        self.buffer.clear()

    def flush(self):
        return None

    def finish(self):
        if self.buffer:
            self._part()


class _PartReader:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def read(self, size=-1):
        if size < 0:
            raise ValueError('Bounded reads are required for catalog recovery')
        result = bytearray()
        while len(result) < size:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open('rb')
            value = self.current.read(size - len(result))
            if value:
                result.extend(value)
            else:
                self.current.close()
                self.current = None
        return bytes(result)

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None


def pack_catalog(source, destination=DEFAULT_BUNDLE, *, chunk_bytes=CHUNK_BYTES):
    """One-time explicit read-only migration; gzip chunks remain below Git limits."""
    source = Path(source).resolve()
    destination = _destination(destination)
    if destination == source or destination.is_relative_to(source):
        raise ValueError('The source catalog must remain read-only')
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= CHUNK_BYTES:
        raise ValueError('Chunk size must be positive and at most 64 MiB')
    names = ['catalog.sqlite', 'data_manifest.json', 'index_manifest.json', 'goals.json']
    if (source / 'targets.json').is_file():
        names.append('targets.json')
    index = json.loads((source / 'index_manifest.json').read_text(encoding='utf-8'))
    if index.get('backend', 'lucene') == 'lucene':
        names += [p.relative_to(source).as_posix() for p in sorted((source / 'index').rglob('*')) if p.is_file()]
        if len(names) <= 5:
            raise ValueError('The original Lucene index is missing')
    manifest_path = destination / 'manifest.json'
    if manifest_path.is_file():
        existing = inspect_bundle(destination, verify_chunks=True)
        if existing.get('source_provenance') != str(source) or set(names) != {i['path'] for i in existing['files']}:
            raise ValueError('An existing catalog bundle belongs to different source inputs')
        for item in existing['files']:
            stat = _inside(source, item['path']).stat()
            if stat.st_size != item['bytes'] or stat.st_mtime_ns != item['source_mtime_ns']:
                raise ValueError('The source catalog changed after its bundle was frozen')
        return existing
    partial = destination / 'manifest.partial.json'
    manifest = (json.loads(partial.read_text(encoding='utf-8')) if partial.is_file() else
                dict(schema_version=1, source_provenance=str(source), codec='gzip', chunk_bytes=chunk_bytes,
                     files=[], products=None, search_backend=index.get('backend', 'lucene'),
                     preserves_database_and_index_bytes=True, search_ranking_changed=False))
    if manifest['source_provenance'] != str(source) or manifest['chunk_bytes'] != chunk_bytes:
        raise ValueError('A partial bundle belongs to different source inputs')
    data = json.loads((source / 'data_manifest.json').read_text(encoding='utf-8'))
    manifest['products'] = data['products']
    manifest['bundle_id'] = data['bundle_id']
    done = {item['path']: item for item in manifest['files']}
    destination.mkdir(parents=True, exist_ok=True)
    for ordinal, name in enumerate(names):
        original = _inside(source, name)
        if not original.is_file() or original.is_symlink():
            raise ValueError('Required catalog file is missing or symlinked: ' + name)
        stat = original.stat()
        if name in done:
            item = done[name]
            if item['bytes'] != stat.st_size or item['source_mtime_ns'] != stat.st_mtime_ns:
                raise ValueError('Source catalog changed during migration: ' + name)
            for part in item['chunks']:
                if sha256(_inside(destination, part['path'])) != part['sha256']:
                    raise ValueError('Partial bundle chunk changed')
            continue
        writer = _ChunkWriter(destination, f'f{ordinal:04d}', chunk_bytes)
        digest, size = hashlib.sha256(), 0
        with original.open('rb') as stream, gzip.GzipFile(fileobj=writer, mode='wb', filename='', mtime=0, compresslevel=1) as encoded:
            for block in iter(lambda: stream.read(READ_BYTES), b''):
                digest.update(block)
                size += len(block)
                encoded.write(block)
                if size % (256 * 1024 * 1024) == 0:
                    print(json.dumps(dict(file=name, raw_bytes_processed=size, total_raw_bytes=stat.st_size)), flush=True)
        writer.finish()
        current = original.stat()
        if size != stat.st_size or current.st_size != stat.st_size or current.st_mtime_ns != stat.st_mtime_ns:
            raise ValueError('Read-only source changed during migration')
        item = dict(path=name, bytes=size, sha256=digest.hexdigest(), source_mtime_ns=stat.st_mtime_ns,
                    codec='gzip', compressed_bytes=writer.total, chunks=writer.parts)
        manifest['files'].append(item)
        _save(partial, manifest)
        print(json.dumps(dict(file=name, raw_bytes=size, stored_bytes=writer.total, chunks=len(writer.parts))), flush=True)
    manifest.update(completed_at=datetime.now(timezone.utc).isoformat(),
                    raw_bytes=sum(item['bytes'] for item in manifest['files']),
                    stored_bytes=sum(item['compressed_bytes'] for item in manifest['files']))
    _save(manifest_path, manifest)
    if partial.is_file():
        partial.unlink()
    return manifest


def inspect_bundle(bundle=DEFAULT_BUNDLE, *, verify_chunks=False):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('codec') != 'gzip' or not manifest.get('files'):
        raise ValueError('Catalog recovery manifest is invalid')
    files, chunks, stored = set(), set(), 0
    for item in manifest['files']:
        _inside(bundle, item['path'])
        if (item['path'] in files or item.get('codec') != 'gzip'
                or type(item.get('bytes')) is not int or item['bytes'] < 0
                or type(item.get('compressed_bytes')) is not int or item['compressed_bytes'] < 1
                or not isinstance(item.get('chunks'), list) or not item['chunks']):
            raise ValueError('Catalog recovery file identity is duplicated or invalid')
        files.add(item['path'])
        file_stored = 0
        for part in item['chunks']:
            path = _inside(bundle, part['path'])
            if part['path'] in chunks or not path.is_file() or path.stat().st_size != part['bytes'] or part['bytes'] > CHUNK_BYTES:
                raise ValueError('Catalog recovery chunk is missing, duplicated or oversized')
            chunks.add(part['path'])
            stored += part['bytes']
            file_stored += part['bytes']
            if verify_chunks and sha256(path) != part['sha256']:
                raise ValueError('Catalog recovery chunk checksum mismatch')
        if file_stored != item['compressed_bytes']:
            raise ValueError('Catalog recovery file compressed size differs')
    if not {'catalog.sqlite', 'goals.json', 'data_manifest.json', 'index_manifest.json'}.issubset(files):
        raise ValueError('Catalog recovery bundle is incomplete')
    if stored != manifest['stored_bytes']:
        raise ValueError('Catalog recovery stored byte count differs')
    if sum(item['bytes'] for item in manifest['files']) != manifest['raw_bytes']:
        raise ValueError('Catalog recovery raw byte count differs')
    if manifest.get('search_backend') == 'lucene' and not any(p.startswith('index/segments_') for p in files):
        raise ValueError('The original Lucene segment manifest is missing')
    return manifest


def restore_catalog(bundle=DEFAULT_BUNDLE, destination=DEFAULT_CATALOG, *, verify_existing=False):
    """Materialize exactly the original DB/index; only provenance JSON is relocated."""
    bundle, destination = Path(bundle).resolve(), _destination(destination)
    manifest = inspect_bundle(bundle, verify_chunks=True)
    destination.mkdir(parents=True, exist_ok=True)
    lock = destination / '.restore.lock'
    handle = _acquire_restore_lock(lock)
    restored = []
    try:
        for item in manifest['files']:
            target = _inside(destination, item['path'])
            # The two JSON manifests are intentionally normalized below.
            metadata = item['path'] in ('data_manifest.json', 'index_manifest.json')
            if target.is_file() and not metadata and target.stat().st_size == item['bytes']:
                if sha256(target) == item['sha256']:
                    restored.append(item['path'])
                    continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.restore-' + uuid.uuid4().hex[:16] + '.tmp')
            reader = _PartReader([_inside(bundle, p['path']) for p in item['chunks']])
            digest, size = hashlib.sha256(), 0
            try:
                with gzip.GzipFile(fileobj=reader, mode='rb') as decoded, temporary.open('wb') as stream:
                    for block in iter(lambda: decoded.read(READ_BYTES), b''):
                        digest.update(block)
                        size += len(block)
                        stream.write(block)
                if size != item['bytes'] or digest.hexdigest() != item['sha256']:
                    raise ValueError('Recovered catalog bytes do not match original: ' + item['path'])
                temporary.replace(target)
            finally:
                reader.close()
                if temporary.is_file():
                    temporary.unlink()
            restored.append(item['path'])
        data_path = destination / 'data_manifest.json'
        data = json.loads(data_path.read_text(encoding='utf-8'))
        for value in data.get('source_files', {}).values():
            if 'path' in value:
                value['historical_build_input_path'] = value.pop('path')
        data['runtime_inputs'] = 'project_local_catalog_sqlite_goals_and_original_lucene_index'
        data['historical_build_inputs_required_at_runtime'] = False
        _save(data_path, data)
        index_path = destination / 'index_manifest.json'
        index = json.loads(index_path.read_text(encoding='utf-8'))
        if index.get('backend', 'lucene') == 'lucene':
            index['path'] = 'index'
        _save(index_path, index)
        report = dict(schema_version=1, catalog_bundle_manifest_sha256=sha256(bundle / 'manifest.json'),
                      products=manifest['products'], files=len(restored), raw_bytes=manifest['raw_bytes'],
                      stored_bytes=manifest['stored_bytes'], database_and_index_original_bytes=True,
                      sidecar_changes=['historical source file paths marked provenance-only', 'Lucene path relative to catalog'],
                      search_backend=manifest['search_backend'], search_ranking_changed=False)
        _save(destination / 'RESTORED.json', report)
        return report
    finally:
        _release_restore_lock(handle)


def ensure_catalog(destination=DEFAULT_CATALOG, bundle=DEFAULT_BUNDLE):
    destination = Path(destination).resolve()
    bundle = Path(bundle).resolve()
    if (destination / 'catalog.sqlite').is_file() and (destination / 'RESTORED.json').is_file():
        manifest = inspect_bundle(bundle)
        restored = json.loads((destination / 'RESTORED.json').read_text(encoding='utf-8'))
        if restored.get('catalog_bundle_manifest_sha256') != sha256(bundle / 'manifest.json'):
            raise ValueError('Materialized shop catalog is from another recovery bundle')
        complete = all(_inside(destination, item['path']).is_file() and (
            item['path'] in ('data_manifest.json', 'index_manifest.json')
            or _inside(destination, item['path']).stat().st_size == item['bytes']) for item in manifest['files'])
        if complete:
            return destination
    restore_catalog(bundle, destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('pack', 'restore', 'inspect'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args(argv)
    if args.stage == 'pack':
        if args.source is None:
            parser.error('pack requires an explicitly selected read-only source')
        result = pack_catalog(args.source, args.bundle)
    elif args.stage == 'restore':
        result = restore_catalog(args.bundle, args.catalog, verify_existing=args.verify_existing)
    else:
        result = inspect_bundle(args.bundle, verify_chunks=args.verify_existing)
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
