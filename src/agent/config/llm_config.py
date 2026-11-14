"""Connection settings plus environment or ignored local-file credentials."""
import json
import os
import re
from dataclasses import dataclass, field
from src.paths import ProjectPath as Path
from urllib.parse import urlsplit


from src.paths import CONFIGS

DEFAULT_CONFIG = CONFIGS / 'llm.local.json'
CONNECTION_TEMPLATE = CONFIGS / 'llm.example.json'


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)

    def public(self):
        return {'provider': self.provider, 'model': self.model, 'base_url': self.base_url}

    def client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


def load_model_config(path=DEFAULT_CONFIG):
    selected = Path(path or DEFAULT_CONFIG)
    using_template = selected.resolve() == CONNECTION_TEMPLATE.resolve()
    if selected.resolve() == DEFAULT_CONFIG.resolve() and not selected.is_file():
        selected = CONNECTION_TEMPLATE
        using_template = True
    try:
        raw = json.loads(selected.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        raise ValueError('Cannot read model configuration file') from None
    model = raw.get('model') if isinstance(raw, dict) else None
    if not isinstance(model, dict):
        raise ValueError('Configuration requires a model object')
    values = {}
    for name in ('provider', 'model', 'base_url'):
        value = model.get(name)
        if not isinstance(value, str) or not value.strip() or value.strip() == '[REDACTED]':
            raise ValueError('Missing or invalid model.' + name)
        values[name] = value.strip()
    env_name = model.get('api_key_env', 'MINIMAX_API_KEY')
    if not isinstance(env_name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', env_name):
        raise ValueError('Invalid model.api_key_env')
    local_key = model.get('api_key') or ''
    if not isinstance(local_key, str):
        raise ValueError('Missing or invalid model.api_key')
    if using_template and local_key.strip():
        raise ValueError('The committed connection template must not contain a credential')
    values['api_key'] = os.environ.get(env_name, '').strip() or local_key.strip()
    if not values['api_key'] or values['api_key'] == '[REDACTED]':
        raise ValueError('Missing model credential; configure its environment variable or ignored local file')
    if values['provider'] != 'minimax':
        raise ValueError('This local experiment requires the minimax provider')
    try:
        url = urlsplit(values['base_url'])
        valid = url.scheme in ('https', 'http') and url.hostname and not (
            url.username or url.password or url.query or url.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('Invalid model.base_url; use a service URL without credentials or query')
    values['base_url'] = values['base_url'].rstrip('/')
    return ModelConfig(**values)
