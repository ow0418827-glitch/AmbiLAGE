"""Sequential semantic gradients and explicit uncertainty weighting."""
import math
import sys
import torch
import torch.nn.functional as F


def semantic_loss(surrogate, image, text_features):
    feature = F.normalize(surrogate.encode_image(image).global_embedding.float(), dim=-1)
    targets = F.normalize(text_features.detach().float().to(feature.device), dim=-1)
    return (1.0 - feature @ targets.T).mean()


def gradient_variance_floor(variances):
    return max(max(map(float, variances)) * 1e-12, sys.float_info.min)


def inverse_variance_weights(variances, floor=None):
    if not variances:
        raise ValueError('梯度方差不能为空')
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in variances):
        raise ValueError('梯度方差必须为非负有限值')
    floor = gradient_variance_floor(variances) if floor is None else floor
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError('梯度方差的数值下限无效')
    safe = [max(float(v), floor) for v in variances]
    # Log space prevents overflow when the gradient variance is very small.
    logits = [-2.0 * math.log(v) for v in safe]
    maximum = max(logits)
    raw = [math.exp(v - maximum) for v in logits]
    return [v / sum(raw) for v in raw]


def estimate_gradient_variances(source, models, text_features, *, epsilon, restarts, seed):
    """Mean elementwise population variance of semantic image gradients.

    Online moments keep no graph or restart gradient stack alive. Each encoder
    observes the same reproducible random starts inside the projected pixel box.
    """
    if restarts < 2:
        raise ValueError('梯度方差估计至少需要两个起点')
    values = []
    for model, target in zip(models, text_features):
        generator = torch.Generator(device=source.device).manual_seed(seed)
        mean = torch.zeros_like(source, dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        for index in range(1, restarts + 1):
            noise = torch.rand(source.shape, generator=generator, device=source.device, dtype=source.dtype)
            leaf = (source + (2 * noise - 1) * epsilon).clamp(0, 1).detach().requires_grad_(True)
            loss = semantic_loss(model, leaf, target)
            gradient, = torch.autograd.grad(loss, leaf)
            gradient = gradient.detach().to(dtype=torch.float64)
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('随机起点语义梯度含非有限值')
            difference = gradient - mean
            mean += difference / index
            m2 += difference * (gradient - mean)
            del leaf, loss, gradient
        values.append(max(0.0, float((m2 / restarts).mean())))
    return values
