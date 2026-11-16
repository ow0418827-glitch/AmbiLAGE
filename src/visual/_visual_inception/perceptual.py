"""Local pretrained AlexNet LPIPS v0.1, with no download-capable constructor.

The feature slices, input scaling and learned channel calibration follow the
official implementation: https://github.com/richzhang/PerceptualSimilarity .
This small wrapper uses torchvision solely to construct the AlexNet topology;
every parameter is replaced by verified local pretrained state before use.
"""
from pathlib import Path
import torch
import torch.nn as nn

from src.paths import MODEL_ROOT
from src.visual.io import sha256_file

OFFICIAL_ALEXNET_SHA256_PREFIX = '7be5be79'
OFFICIAL_LPIPS_ALEX_V01_SHA256 = 'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'


class TinyTestPerceptual(nn.Module):
    """Explicit synthetic fixture; this squared pixel distance is not LPIPS."""
    def forward(self, source, modified):
        return (source - modified).square().mean()

    def snapshot(self):
        return dict(kind='tiny_test_squared_pixel_distance_not_lpips', test_fixture=True,
                    pretrained=False, production_backend=False)


def _resolve_local(path, default):
    value = Path(path).expanduser() if path is not None else MODEL_ROOT / 'lpips' / default
    if not value.is_absolute():
        from src.paths import PROJECT
        value = PROJECT / value
    value = value.resolve()
    if not value.is_file():
        raise FileNotFoundError('真实感知度量缺少本地预训练权重：' + str(value))
    return value


def resolve_lpips_weights(config):
    backbone = _resolve_local(config.lpips_backbone_weights, 'alexnet-owt-7be5be79.pth')
    linear = _resolve_local(config.lpips_linear_weights, 'alex_v0.1.pth')
    backbone_sha, linear_sha = sha256_file(backbone), sha256_file(linear)
    if not backbone_sha.startswith(OFFICIAL_ALEXNET_SHA256_PREFIX):
        raise ValueError('感知骨干权重未通过官方预训练文件摘要前缀校验')
    if linear_sha != OFFICIAL_LPIPS_ALEX_V01_SHA256:
        raise ValueError('感知校准权重不是官方零点一版文件')
    return dict(backbone_path=str(backbone), backbone_sha256=backbone_sha,
                linear_path=str(linear), linear_sha256=linear_sha)


class LocalAlexNetLPIPS(nn.Module):
    def __init__(self, config, device='cuda'):
        super().__init__()
        # Validate actual local bytes before importing/constructing the network.
        self.identity = resolve_lpips_weights(config)
        try:
            from torchvision.models import alexnet
        except ImportError as exc:
            raise ImportError('真实感知度量需要安装视觉模型依赖 torchvision') from exc
        backbone = alexnet(weights=None)
        state = torch.load(self.identity['backbone_path'], map_location='cpu', weights_only=True)
        backbone.load_state_dict(state, strict=True)
        features = list(backbone.features.children())
        boundaries = ((0, 2), (2, 5), (5, 8), (8, 10), (10, 12))
        self.slices = nn.ModuleList(nn.Sequential(*features[start:end]) for start, end in boundaries)
        del backbone, state
        calibration = torch.load(self.identity['linear_path'], map_location='cpu', weights_only=True)
        expected = {f'lin{i}.model.1.weight' for i in range(5)}
        if not isinstance(calibration, dict) or set(calibration) != expected:
            raise ValueError('官方感知校准层结构不匹配')
        self.calibration = nn.ParameterList()
        for index, channels in enumerate((64, 192, 384, 256, 256)):
            weight = calibration[f'lin{index}.model.1.weight']
            if tuple(weight.shape) != (1, channels, 1, 1) or not torch.isfinite(weight).all() or (weight < 0).any():
                raise ValueError('官方感知校准层权重无效')
            self.calibration.append(nn.Parameter(weight.detach().clone(), requires_grad=False))
        self.register_buffer('shift', torch.tensor([-.030, -.088, -.188]).reshape(1, 3, 1, 1))
        self.register_buffer('scale', torch.tensor([.458, .448, .450]).reshape(1, 3, 1, 1))
        self.requires_grad_(False).eval().to(device)

    def forward(self, source, modified):
        if min(source.shape[-2:]) < 32:
            raise ValueError('真实感知度量图片宽高至少三十二像素')
        left = (source * 2 - 1 - self.shift) / self.scale
        right = (modified * 2 - 1 - self.shift) / self.scale
        total = source.new_zeros(())
        for layer, weight in zip(self.slices, self.calibration):
            left, right = layer(left), layer(right)
            left_norm = left / (left.square().sum(1, keepdim=True).sqrt() + 1e-10)
            right_norm = right / (right.square().sum(1, keepdim=True).sqrt() + 1e-10)
            total = total + ((left_norm - right_norm).square() * weight).sum(1).mean()
        return total

    def snapshot(self):
        return dict(kind='lpips_alexnet_v0.1', version='0.1', test_fixture=False,
                    pretrained=True, production_backend=True, downloads_during_training=False,
                    backbone_initialization='strict_local_official_pretrained_state',
                    official_backbone_sha256_prefix=OFFICIAL_ALEXNET_SHA256_PREFIX,
                    official_linear_sha256=OFFICIAL_LPIPS_ALEX_V01_SHA256,
                    implementation='official_feature_slices_scaling_channel_calibration', **self.identity)


def load_perceptual(config, device):
    if config.smoke_test:
        return TinyTestPerceptual().to(device)
    return LocalAlexNetLPIPS(config, device)
