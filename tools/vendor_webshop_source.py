"""Copy the runtime-only WebShop source from an explicitly selected read-only tree."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT, RESOURCES

LAZY_LANGUAGE_MODEL = '''# Runtime-only language model initialization; importing the package loads no weights.
import os
from pathlib import Path

nlp = None
LANGUAGE_MODEL = os.environ.get("AMBILAGE_WEBSHOP_SPACY_MODEL", "en_core_web_lg")


def initialize_language_model():
    global nlp
    if nlp is None:
        configured = LANGUAGE_MODEL
        if "/" in configured or "\\\\" in configured or configured.startswith("."):
            path = Path(configured).expanduser()
            if not path.is_absolute():
                project = Path(os.environ.get("AMBILAGE_PROJECT_ROOT", Path(__file__).resolve().parents[5]))
                path = project / path
            configured = str(path.resolve())
        loaded = spacy.load(configured)
        if (loaded.meta.get("lang") != "en" or loaded.meta.get("name") != "core_web_lg"
                or loaded.meta.get("version") != "3.8.0"):
            raise ValueError("The frozen shop reward protocol requires en_core_web_lg 3.8.0")
        nlp = loaded
    return nlp
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def vendor_source(source, destination=RESOURCES / 'webshop/source'):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or not destination.is_relative_to(PROJECT.resolve()):
        raise ValueError('Vendor destination must be inside the new project, separate from the read-only source')
    names = ['LICENSE.md', 'web_agent_site/__init__.py', 'web_agent_site/app.py', 'web_agent_site/utils.py',
             'web_agent_site/engine/__init__.py', 'web_agent_site/engine/engine.py',
             'web_agent_site/engine/goal.py', 'web_agent_site/engine/normalize.py']
    for folder in ('templates', 'static'):
        names += [p.relative_to(source).as_posix() for p in sorted((source / 'web_agent_site' / folder).rglob('*'))
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    records = []
    for name in names:
        original = source / name
        if not original.is_file() or original.is_symlink():
            raise ValueError('Required upstream runtime file missing: ' + name)
        raw = original.read_bytes()
        data = raw
        edits = []
        if name.endswith('/goal.py'):
            text = raw.decode('utf-8')
            start = text.index('try:\n    nlp = spacy.load')
            stop = text.index('\nPRICE_RANGE', start)
            text = text[:start] + LAZY_LANGUAGE_MODEL + '\n' + text[stop:]
            text = text.replace('purchased_type_parse = nlp(purchased_type)',
                                'language_model = initialize_language_model()\n    purchased_type_parse = language_model(purchased_type)')
            text = text.replace('desired_type_parse = nlp(desired_type)', 'desired_type_parse = language_model(desired_type)')
            data = text.encode('utf-8')
            edits = ['initialize NLP only when explicitly serving/rewarding',
                     'require the historical en_core_web_lg 3.8.0 without a small-model fallback',
                     'resolve explicit relative NLP directories from the project root']
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != data:
            raise ValueError('Refusing to overwrite a changed vendored file: ' + name)
        target.write_bytes(data)
        records.append(dict(path=name, source_sha256=digest(raw), vendored_sha256=digest(data),
                            original_bytes=len(raw), vendored_bytes=len(data), modifications=edits))
    manifest = dict(schema_version=1, origin='explicit_read_only_existing_webshop_source',
                    source_tree=str(source), files=records, license='LICENSE.md',
                    runtime_only=True, old_user_logs_models_drivers_and_raw_datasets_copied=False,
                    nlp_protocol=dict(package='en_core_web_lg', version='3.8.0', initialization='explicit_service_start'))
    (destination / 'VENDOR_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = vendor_source(args.source)
    print(json.dumps(dict(files=len(manifest['files']), bytes=sum(f['vendored_bytes'] for f in manifest['files']),
                          license_preserved=True, language_model_weights_copied=False), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
