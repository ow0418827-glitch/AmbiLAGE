"""Optional local shop inputs, explicitly installed inside this project or configured by the user."""
import os
from src.paths import ProjectPath as Path, PROJECT


def _configured_path(name, default):
    value = Path(os.environ.get(name, default)).expanduser()
    value = (value if value.is_absolute() else PROJECT / value).resolve()
    if not value.is_relative_to(PROJECT):
        raise ValueError(name + ' must point inside the current project; old sibling projects are not runtime inputs')
    return value


CATALOG = _configured_path("AMBILAGE_WEBSHOP_CATALOG", "resources/webshop/catalog")
WEBSHOP_SOURCE = _configured_path("AMBILAGE_WEBSHOP_SOURCE", "resources/webshop/source")
