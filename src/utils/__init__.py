import hashlib
from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config(path: str = "../config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def passage_id(text: str) -> str:
    """Stable ID derived from passage content, not row position."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]