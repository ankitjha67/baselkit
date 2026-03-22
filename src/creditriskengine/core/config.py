"""Configuration loader for creditriskengine.

This module delegates to ``creditriskengine.regulatory.loader`` which is
the canonical implementation.  It is kept for backward-compatibility so
that ``from creditriskengine.core.config import load_jurisdiction_config``
continues to work.
"""

from pathlib import Path
from typing import Any

from creditriskengine.core.types import Jurisdiction
from creditriskengine.regulatory.loader import load_config


def load_jurisdiction_config(
    jurisdiction: Jurisdiction,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Load regulatory configuration YAML for a jurisdiction.

    Delegates to :func:`creditriskengine.regulatory.loader.load_config`.

    Args:
        jurisdiction: Target jurisdiction.
        config_dir: Override path to regulatory config directory.

    Returns:
        Parsed YAML configuration dict.
    """
    return load_config(jurisdiction, config_dir=config_dir)
