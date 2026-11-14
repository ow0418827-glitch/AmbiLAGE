"""Lightweight model identity; safe to import before runtime dependencies exist."""
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    pretrained: str
    hf_repo: str | None = None
    weight_path: str | None = None
    tokenizer_path: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> 'BackboneConfig':
        return cls(
            name=str(raw['name']),
            pretrained=str(raw.get('pretrained', 'openai')),
            hf_repo=str(raw['hf_repo']) if raw.get('hf_repo') else None,
            weight_path=str(raw['weight_path']) if raw.get('weight_path') else None,
            tokenizer_path=str(raw['tokenizer_path']) if raw.get('tokenizer_path') else None,
        )
