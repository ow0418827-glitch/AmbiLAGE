"""Import/freeze public training data or explicitly request model paraphrases.

No implicit downloads, paid service calls, memory reads, or shopping are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visual._visual_inception.data_preparation import (
    DEFAULT_BASE_URL, generate_paraphrases, prepare_training_bundle, _write_rows,
)
from src.visual._visual_inception.training_data import (
    read_source_rows, load_training_bundle, _assert_public_path, validate_public_bundle_root, reject_private_fields,
)
from src.visual.io import save_json


def _json(path):
    path = Path(path).resolve()
    _assert_public_path(path)
    value = json.loads(path.read_text(encoding='utf-8'))
    reject_private_fields(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='stage', required=True)
    paraphrase = subparsers.add_parser('paraphrases', help='默认仅打印改写计划，显式开关才调用模型')
    paraphrase.add_argument('--descriptions', type=Path, required=True)
    paraphrase.add_argument('--provider', choices=('minimax', 'openai'), default='minimax')
    paraphrase.add_argument('--model', required=True, help='明确模型身份，不自动更换')
    paraphrase.add_argument('--base-url', default=DEFAULT_BASE_URL)
    paraphrase.add_argument('--api-key-env', default='MINIMAX_API_KEY')
    paraphrase.add_argument('--generate-paraphrases', action='store_true')
    paraphrase.add_argument('--output', type=Path, required=True)
    freeze = subparsers.add_parser('freeze', help='导入有限公开来源并冻结严格配额训练包')
    freeze.add_argument('--group', required=True)
    freeze.add_argument('--base', type=Path, default=PROJECT_ROOT / 'resources/evaluation_base')
    freeze.add_argument('--descriptions', type=Path, required=True)
    freeze.add_argument('--paraphrases', type=Path, required=True)
    freeze.add_argument('--heldout-queries', type=Path, required=True)
    freeze.add_argument('--candidates', type=Path, required=True)
    freeze.add_argument('--source-metadata', type=Path, required=True,
                        help='登记四类来源的公开语料身份、版本、划分和实际生成模型')
    freeze.add_argument('--paraphrase-policy', choices=('declared_model', 'paper_gpt4'), default='declared_model')
    freeze.add_argument('--output', type=Path, required=True)
    validate = subparsers.add_parser('validate', help='只读核验冻结公开训练包')
    validate.add_argument('--group', required=True)
    validate.add_argument('--base', type=Path, default=PROJECT_ROOT / 'resources/evaluation_base')
    validate.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.stage in ('paraphrases', 'freeze'):
        args.output = validate_public_bundle_root(args.output)
    if args.stage == 'paraphrases':
        if args.generate_paraphrases and args.output.exists():
            raise FileExistsError('改写结果目录已存在，禁止覆盖')
        result = generate_paraphrases(list(read_source_rows(args.descriptions).values()),
            provider=args.provider, model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
            generate=args.generate_paraphrases)
        if not args.generate_paraphrases:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        args.output.mkdir(parents=True, exist_ok=False)
        path = args.output / 'model_paraphrases.jsonl'
        digest = _write_rows(path, result['rows'])
        save_json(args.output / 'generation_provenance.json', dict(result['generation'],
            output_file=path.name, output_sha256=digest, output_count=len(result['rows'])))
        print(json.dumps(dict(status='generated', output=str(path), row_count=30,
            model=result['generation']['model'],
            effective_temperature=result['generation']['effective_temperature'], credentials_logged=False),
            ensure_ascii=False, indent=2))
        return 0
    from src.experiments.base_workflow import public_group
    from src.visual._visual_inception.targets import prepare_semantic_targets
    public = public_group(args.base, args.group)
    definition = dict(public['definition'], target_product_features=public['target_features'])
    semantic = prepare_semantic_targets(public['source']['image'], public['target']['image'],
        target_text=public['target_text'], definition=definition)
    if args.stage == 'validate':
        bundle = load_training_bundle(args.manifest, source=public['source']['image'],
            target=public['target']['image'], public_semantic=semantic)
        print(json.dumps(dict(status='validated', manifest_path=bundle['manifest_path'],
            manifest_sha256=bundle['manifest_sha256'], query_count=100,
            candidate_count=len(bundle['candidates']), network_request_performed=False),
            ensure_ascii=False, indent=2))
        return 0
    bundle = prepare_training_bundle(source=public['source']['image'], target=public['target']['image'],
        public_semantic=semantic, descriptions=args.descriptions, paraphrases=args.paraphrases,
        heldout_queries=args.heldout_queries, candidates=args.candidates, source_metadata=_json(args.source_metadata),
        output_dir=args.output, paraphrase_policy=args.paraphrase_policy)
    print(json.dumps(dict(status='frozen', manifest_path=bundle['manifest_path'],
        manifest_sha256=bundle['manifest_sha256'], query_count=100,
        candidate_count=len(bundle['candidates']), paraphrase_policy=args.paraphrase_policy,
        shopping_invoked=False), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
