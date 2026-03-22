"""YAML config loader with validation for regulatory parameters."""

import logging
from pathlib import Path
from typing import Any

import yaml

from creditriskengine.core.exceptions import ConfigurationError
from creditriskengine.core.types import Jurisdiction

logger = logging.getLogger(__name__)

_REGULATORY_DIR = Path(__file__).parent


def get_config_path(jurisdiction: Jurisdiction) -> Path:
    """Resolve the YAML config file path for a jurisdiction."""
    mapping: dict[str, str] = {
        "bcbs": "bcbs/bcbs_d424.yml",
        "eu": "eu/crr3.yml",
        "uk": "uk/pra_basel31.yml",
        "us": "us/us_endgame.yml",
        "india": "india/rbi.yml",
        "singapore": "singapore/mas_637.yml",
        "hong_kong": "hongkong/hkma.yml",
        "japan": "japan/jfsa.yml",
        "australia": "australia/apra.yml",
        "canada": "canada/osfi.yml",
        "china": "china/nfra.yml",
        "south_korea": "southkorea/fss.yml",
        "uae": "uae/cbuae.yml",
        "saudi_arabia": "saudi/sama.yml",
        "south_africa": "southafrica/sarb.yml",
        "brazil": "brazil/bcb.yml",
        "malaysia": "malaysia/bnm.yml",
    }
    rel = mapping.get(jurisdiction.value)
    if rel is None:
        raise ConfigurationError(f"Unknown jurisdiction: {jurisdiction}")
    return _REGULATORY_DIR / rel


def load_config(
    jurisdiction: Jurisdiction,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Load and return regulatory config for a jurisdiction.

    Args:
        jurisdiction: Target jurisdiction.
        config_dir: Optional override for config directory.

    Returns:
        Parsed config dict.
    """
    if config_dir:
        path = config_dir / get_config_path(jurisdiction).relative_to(_REGULATORY_DIR)
    else:
        path = get_config_path(jurisdiction)

    if not path.exists():
        raise ConfigurationError(f"Config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ConfigurationError(f"Empty config: {path}")

    logger.debug("Loaded regulatory config: %s", path)
    return dict(data)
