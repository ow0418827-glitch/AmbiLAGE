"""Inspect optional shopping dependencies without importing a model or starting a shop."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import PROJECT

EXPECTED_PACKAGES = {'pyserini': '1.6.0', 'spacy': '3.8.14', 'en-core-web-lg': '3.8.0',
                     'rich': '15.0.0', 'tqdm': '4.68.3', 'cleantext': '1.1.4',
                     'rank-bm25': '0.2.2', 'thefuzz': '0.22.1'}
MINIMUM_JAVA_MAJOR = 21


def java_runtime():
    home = os.environ.get('JAVA_HOME')
    explicit = Path(home).expanduser() / 'bin' / ('java.exe' if os.name == 'nt' else 'java') if home else None
    if explicit is not None and not explicit.is_file():
        return {'available': False, 'major': None, 'minimum_major': MINIMUM_JAVA_MAJOR,
                'compatible': False, 'reason': 'Configured JAVA_HOME does not contain a Java executable'}
    executable = str(explicit) if explicit is not None and explicit.is_file() else shutil.which('java')
    if executable is None:
        return {'available': False, 'major': None, 'minimum_major': MINIMUM_JAVA_MAJOR,
                'compatible': False, 'reason': 'Java executable was not found; install a Java 21 runtime'}
    try:
        result = subprocess.run([executable, '-version'], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'available': False, 'major': None, 'minimum_major': MINIMUM_JAVA_MAJOR,
                'compatible': False, 'reason': type(error).__name__}
    output = (result.stderr + '\n' + result.stdout).strip()
    match = re.search(r'(?:version\s+"|(?:openjdk|java)\s+)(\d+)(?:\.(\d+))?', output)
    major = int(match.group(1)) if match else None
    if major == 1 and match.group(2):
        major = int(match.group(2))
    return {'available': result.returncode == 0, 'executable': executable,
            'major': major, 'minimum_major': MINIMUM_JAVA_MAJOR,
            'compatible': result.returncode == 0 and major is not None and major >= MINIMUM_JAVA_MAJOR,
            'version_line': output.splitlines()[0] if output else ''}


def english_model_metadata():
    configured = os.environ.get('AMBILAGE_WEBSHOP_SPACY_MODEL', 'en_core_web_lg')
    if configured == 'en_core_web_lg':
        try:
            version = importlib.metadata.version('en-core-web-lg')
        except importlib.metadata.PackageNotFoundError:
            version = None
        return dict(configured=configured, version=version, compatible=version == '3.8.0', source='package_metadata')
    path = Path(configured).expanduser()
    path = path if path.is_absolute() else PROJECT / path
    try:
        metadata = json.loads((path / 'meta.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return dict(configured=configured, version=None, compatible=False, source='local_directory_metadata')
    same_model = metadata.get('lang') == 'en' and metadata.get('name') == 'core_web_lg'
    return dict(configured=str(path.resolve()), version=metadata.get('version'),
                compatible=same_model and metadata.get('version') == '3.8.0', source='local_directory_metadata')


def inspect_webshop_runtime(*, java_only=False):
    java = java_runtime()
    packages = {}
    model = None
    if not java_only:
        model = english_model_metadata()
        for package, expected in EXPECTED_PACKAGES.items():
            if package == 'en-core-web-lg':
                actual, compatible = model['version'], model['compatible']
            else:
                try:
                    actual = importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:
                    actual = None
                compatible = actual == expected
            packages[package] = dict(installed=actual, expected=expected, compatible=compatible)
    return {'schema_version': 1, 'java': java, 'packages': packages,
            'ready': java['compatible'] and all(p['compatible'] for p in packages.values()),
            'java_only': java_only, 'english_model': model, 'models_loaded': False, 'shop_started': False,
            'shopping_model_calls': 0, 'linux_server_execution_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--java-only', action='store_true')
    parser.add_argument('--require', action='store_true', help='Return failure when required versions are absent')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = inspect_webshop_runtime(java_only=args.java_only)
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ready'] or not args.require else 1


if __name__ == '__main__':
    raise SystemExit(main())
