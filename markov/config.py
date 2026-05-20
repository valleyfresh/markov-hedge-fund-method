from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml, expanding ${ENV_VAR} references."""
    text = Path(path).read_text()
    text = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), text)
    return yaml.safe_load(text)
