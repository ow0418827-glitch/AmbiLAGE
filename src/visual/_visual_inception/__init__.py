"""Explicit public-only adaptation of Visual Inception without retrieval loss."""
from .config import VisualInceptionConfig


def generate_visual_inception(*args, **kwargs):
    from .pipeline import generate_visual_inception as generate
    return generate(*args, **kwargs)


__all__ = ['VisualInceptionConfig', 'generate_visual_inception']
