"""Sequential semantic gradients and explicit uncertainty weighting."""
import math
import sys
import torch
import torch.nn.functional as F


def semantic_loss(surrogate, image, text_features):
    feature = F.normalize(surrogate.encode_image(image).global_embedding.float(), dim=-1)
    targets = F.normalize(text_features.detach().float().to(feature.device), dim=-1)
    return (1.0 - feature @ targets.T).mean()


def sample_query_indices(count, batch_size, seed, step):
    """A reproducible uniform subset per step; the query pool stays frozen."""
    if (type(count) is not int or type(batch_size) is not int
            or not 1 <= batch_size <= count or type(step) is not int or step < 0):
        raise ValueError('查询采样规模或步骤无效')
    generator = torch.Generator(device='cpu').manual_seed((int(seed) + 1000003 * step) % 2**63)
    return torch.randperm(count, generator=generator)[:batch_size].tolist()


def retrieval_loss(image_features, query_features, negative_features, temperature=0.07):
    """Normalized positive probability with exactly one live positive slot.

    Fixed public negative vectors are detached. Only the adversarial image has a
    graph. The dynamic positive is present in both numerator and denominator.
    """
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError('检索温度必须为有限正数')
    if (image_features.ndim != 2 or image_features.shape[0] != 1
            or query_features.ndim != 2 or negative_features.ndim != 2
            or not query_features.shape[0] or not negative_features.shape[0]
            or image_features.shape[1] != query_features.shape[1]
            or image_features.shape[1] != negative_features.shape[1]):
        raise ValueError('动态正例、查询与固定负例向量形状不一致')
    dtype = torch.float64 if image_features.dtype == torch.float64 else torch.float32
    image = F.normalize(image_features.to(dtype=dtype), dim=-1)
    queries = F.normalize(query_features.detach().to(device=image.device, dtype=dtype), dim=-1)
    negatives = F.normalize(negative_features.detach().to(device=image.device, dtype=dtype), dim=-1)
    positive_logits = queries @ image.T / temperature
    negative_logits = queries @ negatives.T / temperature
    return (torch.logsumexp(torch.cat((positive_logits, negative_logits), dim=1), dim=1)
            - positive_logits[:, 0]).mean()


def retrieval_semantic_losses(image_features, concept_features, query_features,
                              negative_features, temperature=0.07):
    if concept_features.ndim != 2 or concept_features.shape[0] != 1:
        raise ValueError('语义目标必须为一个独立的冻结概念向量')
    dtype = torch.float64 if image_features.dtype == torch.float64 else torch.float32
    image = F.normalize(image_features.to(dtype=dtype), dim=-1)
    concept = F.normalize(concept_features.detach().to(device=image.device, dtype=dtype), dim=-1)
    semantic = (1.0 - image @ concept.T).mean()
    return semantic, retrieval_loss(image, query_features, negative_features, temperature)


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


def estimate_gradient_variances(source, models, text_features, *, epsilon, restarts, seed,
        retrieval_state=None, calibration_indices=None, retrieval_weight=1.0,
        semantic_weight=1.0, temperature=0.07):
    """Mean elementwise population variance of semantic image gradients.

    Online moments keep no graph or restart gradient stack alive. Each encoder
    observes the same reproducible random starts inside the projected pixel box.
    """
    if restarts < 2:
        raise ValueError('梯度方差估计至少需要两个起点')
    values = []
    for model_index, (model, target) in enumerate(zip(models, text_features)):
        generator = torch.Generator(device=source.device).manual_seed(seed)
        mean = torch.zeros_like(source, dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        for index in range(1, restarts + 1):
            noise = torch.rand(source.shape, generator=generator, device=source.device, dtype=source.dtype)
            leaf = (source + (2 * noise - 1) * epsilon).clamp(0, 1).detach().requires_grad_(True)
            if retrieval_state is None:
                loss = semantic_loss(model, leaf, target)
            else:
                if calibration_indices is None:
                    raise ValueError('检索梯度方差估计须冻结校准查询索引')
                feature = model.encode_image(leaf).global_embedding
                queries = retrieval_state['query_features'][model_index][calibration_indices]
                semantic, retrieval = retrieval_semantic_losses(feature, target, queries,
                    retrieval_state['negative_features'][model_index], temperature)
                loss = semantic_weight * semantic + retrieval_weight * retrieval
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
