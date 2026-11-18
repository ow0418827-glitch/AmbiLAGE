"""Public training-proxy selections shared by generation and deployment tools."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re

from src.paths import PROJECT, MODEL_ROOT, VISUAL_CONFIGS
from src.visual.encoders.spec import BackboneConfig

SURROGATE_SETS = ('clip_l14', 'lucid', 'heterogeneous4', 'visual_inception_paper', 'custom')
HELD_OUT_MODELS = ('ViT-g-14', 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct')
HELD_OUT_PURIFIERS = ('eval_purifier_sd21', 'sd2-community/stable-diffusion-2-1-base', 'stabilityai/stable-diffusion-2-1-base')


def is_held_out_purifier_reference(value: str | Path | None) -> bool:
    if value is None:
        return False
    lowered = str(value).replace('\\', '/').casefold()
    return any(
        token in lowered
        for token in ('eval_purifier_sd21', 'stable-diffusion-2-1', 'stable_diffusion_2_1')
    )


def all_registered_training_specs() -> list[BackboneConfig]:
    seen: set[tuple[str, str, str | None]] = set()
    ordered: list[BackboneConfig] = []
    for preset in ('clip_l14', 'lucid', 'heterogeneous4', 'visual_inception_paper'):
        for spec in select_training_surrogates(preset).specs:
            ident = spec_identity(spec)
            if ident not in seen:
                seen.add(ident)
                ordered.append(spec)
    return ordered


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def spec_identity(spec):
    return spec.name, spec.pretrained, spec.hf_repo


def validate_training_specs(specs):
    specs = tuple(specs)
    if not specs or any(not s.name.strip() or not s.pretrained.strip() for s in specs):
        raise ValueError('训练代理须明确提供非空模型名称和预训练权重身份')
    if len({spec_identity(s) for s in specs}) != len(specs):
        raise ValueError('训练代理不能重复；同架构的不同预训练权重可以分别选择')
    for spec in specs:
        # The registered ViT-bigG-14 architecture is distinct from the held-out
        # ViT-g-14. Never shorten or case-normalize bigG into the evaluator name.
        fields = ' '.join(str(v or '') for v in asdict(spec).values()).casefold()
        if any(name in fields for name in ('vit-g-14', 'gme', 'eval_g14', 'eval_gme', 'eval_purifier_sd21', 'stable-diffusion-2-1')):
            raise ValueError('留出评估编码器禁止参与训练')
    return specs


@dataclass(frozen=True)
class TrainingSurrogateSelection:
    profile: str
    specs: tuple[BackboneConfig, ...]
    source: Path
    source_sha256: str

    def audit(self):
        source = self.source.relative_to(PROJECT).as_posix()
        specs = [asdict(s) for s in self.specs]
        reference = ('visual_inception_single_clip_l14_explicit_checkpoint' if self.profile == 'clip_l14'
                     else 'current_lucid_configuration_and_loader' if self.profile == 'lucid'
                     else 'visual_inception_paper_architectures_explicit_checkpoints'
                     if self.profile == 'visual_inception_paper' else 'explicit_alternative')
        audit = dict(schema_version=1, profile=self.profile, config_source=source,
                    config_sha256=self.source_sha256, specs=specs, specs_sha256=_digest(specs),
                    comparison_reference=reference,
                    activation_policy='pretrained_native', model_count=len(specs),
                    held_out_models=list(HELD_OUT_MODELS), downloads_during_training=False)
        if self.profile == 'clip_l14':
            audit['paper_single_encoder_architecture'] = 'CLIP-ViT-L/14'
            audit['paper_exact_checkpoints_specified'] = False
            audit['checkpoint_policy'] = 'explicit_public_pretrained_identity_not_claimed_author_checkpoint'
        elif self.profile == 'visual_inception_paper':
            audit['paper_architectures'] = ['CLIP-ViT-L/14', 'SigLIP-SO400M-patch14-384', 'OpenCLIP-ViT-G-14']
            audit['registered_architecture_aliases'] = {'OpenCLIP-ViT-G-14': 'ViT-bigG-14'}
            audit['paper_exact_checkpoints_specified'] = False
            audit['checkpoint_policy'] = 'explicit_public_pretrained_identity_not_claimed_author_checkpoint'
        return audit


def select_training_surrogates(profile='lucid', config_path=None):
    if profile not in SURROGATE_SETS:
        raise ValueError('未知训练代理方案：' + str(profile))
    if profile == 'custom':
        if not config_path:
            raise ValueError('自定义训练代理必须提供配置文件')
        path = Path(config_path).expanduser()
        path = (path if path.is_absolute() else PROJECT / path).resolve()
        if not path.is_relative_to(PROJECT.resolve()):
            raise ValueError('训练代理配置须位于当前项目内，以便随仓库恢复')
    else:
        if config_path is not None:
            raise ValueError('内置训练代理方案不能同时指定自定义代理配置')
        filenames = {'clip_l14': 'surrogates.clip_l14.json', 'lucid': 'optimization.json',
                     'heterogeneous4': 'surrogates.heterogeneous4.json',
                     'visual_inception_paper': 'surrogates.visual_inception.json'}
        path = VISUAL_CONFIGS / filenames[profile]
    data = path.read_bytes()
    raw = json.loads(data.decode('utf-8'))
    if profile == 'custom' and set(raw) - {'schema_version', 'surrogates', 'held_out', 'allow_downloads'}:
        raise ValueError('自定义代理文件只接受公开模型配置字段')
    if raw.get('allow_downloads', False) is not False:
        raise ValueError('训练阶段禁止自动下载代理权重')
    rows = raw.get('surrogates')
    if not isinstance(rows, list) or not rows:
        raise ValueError('训练代理配置必须提供非空 surrogates 列表')
    allowed = set(BackboneConfig.__dataclass_fields__)
    if any(not isinstance(row, dict) or set(row) - allowed or not row.get('pretrained') for row in rows):
        raise ValueError('每个训练代理须显式提供名称及预训练身份，不接受未知字段')
    specs = validate_training_specs(BackboneConfig.from_mapping(row) for row in rows)
    return TrainingSurrogateSelection(profile, specs, path, hashlib.sha256(data).hexdigest())


def selection_from_config(config):
    return select_training_surrogates(getattr(config, 'surrogate_set', 'lucid'),
                                      getattr(config, 'surrogate_config', None))


def apply_surrogate_overrides(config, surrogate_set=None, surrogate_config=None):
    if surrogate_config is not None:
        if surrogate_set not in (None, 'custom'):
            raise ValueError('训练代理内置方案与自定义文件不能同时指定')
        config = replace(config, surrogate_set='custom', surrogate_config=str(surrogate_config))
    elif surrogate_set is not None:
        config = replace(config, surrogate_set=surrogate_set,
                         surrogate_config=config.surrogate_config if surrogate_set == 'custom' else None)
    config.validate()
    return config


def validate_loaded_surrogates(surrogates, selection, *, allow_fixture=False):
    snapshots = [s.snapshot() if hasattr(s, 'snapshot') else s for s in surrogates]
    if len(snapshots) != len(selection.specs):
        raise ValueError('实际训练代理数量与所选方案不一致')
    if allow_fixture and all(s.get('test_fixture') is True for s in snapshots):
        return snapshots
    if any(s.get('test_fixture') for s in snapshots):
        raise ValueError('缺少所选方案的真实训练代理证据')
    for spec, actual in zip(selection.specs, snapshots):
        if (actual.get('name'), actual.get('pretrained'), actual.get('hf_repo')) != spec_identity(spec):
            raise ValueError('实际训练代理模型或预训练权重身份与所选方案不一致')
        if not re.fullmatch(r'[0-9a-f]{64}', actual.get('weight_sha256') or ''):
            raise ValueError('实际训练代理缺少权重内容摘要')
    return snapshots


def validate_selection_audit(audit, config):
    if not isinstance(audit, dict) or audit.get('schema_version') != 1:
        raise ValueError('缺少冻结的训练代理选择记录')
    profile = config.get('surrogate_set')
    if profile not in SURROGATE_SETS or audit.get('profile') != profile:
        raise ValueError('训练代理选择记录与方法配置不一致')
    rows = audit.get('specs', [])
    if audit.get('specs_sha256') != _digest(rows) or audit.get('model_count') != len(rows):
        raise ValueError('训练代理选择记录内容不一致')
    specs = validate_training_specs(BackboneConfig.from_mapping(row) for row in rows)
    if not re.fullmatch(r'[0-9a-f]{64}', audit.get('config_sha256') or ''):
        raise ValueError('训练代理来源配置摘要缺失')
    if profile != 'custom':
        current = select_training_surrogates(profile)
        if rows != current.audit()['specs'] or audit['config_sha256'] != current.source_sha256:
            raise ValueError('内置代理方案已改变，不能冒充同一冻结训练配置')
    source=(PROJECT / audit['config_source']).resolve()
    if not source.is_relative_to(PROJECT.resolve()):
        raise ValueError('冻结代理配置来源不是项目内路径')
    return TrainingSurrogateSelection(profile, specs, source, audit['config_sha256'])


def add_surrogate_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--surrogate-set', choices=('clip_l14', 'lucid', 'heterogeneous4', 'visual_inception_paper'),
                       help='训练代理方案：clip_l14 为单模型 CLIP-ViT-L/14；lucid 为原始像素三代理；heterogeneous4 为四代理组合；visual_inception_paper 为论文三代理架构')
    group.add_argument('--surrogate-config', type=Path,
                       help='项目内自定义公开训练代理配置文件；与内置方案互斥')


def training_model_keys(selection, registry):
    by_repo = {v['repo_id']: key for key, v in registry.items()}
    return [by_repo.get(s.hf_repo) for s in selection.specs]
