"""Configuration loader for creditriskengine."""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from creditriskengine.core.exceptions import ConfigurationError, JurisdictionError
from creditriskengine.core.types import Jurisdiction

logger = logging.getLogger(__name__)

# Default regulatory config directory
_REGULATORY_DIR = Path(__file__).parent.parent / "regulatory"


def load_jurisdiction_config(
    jurisdiction: Jurisdiction,
    config_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Load regulatory configuration YAML for a jurisdiction.

    Args:
        jurisdiction: Target jurisdiction.
        config_dir: Override path to regulatory config directory.

    Returns:
        Parsed YAML configuration dict.

    Raises:
        JurisdictionError: If config file not found.
        ConfigurationError: If YAML is invalid.
    """
    base = config_dir or _REGULATORY_DIR
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

    rel_path = mapping.get(jurisdiction.value)
    if rel_path is None:
        raise JurisdictionError(f"No config mapping for jurisdiction: {jurisdiction}")

    config_path = base / rel_path
    if not config_path.exists():
        raise JurisdictionError(
            f"Config file not found for {jurisdiction.value}: {config_path}"
        )

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}") from e

    if data is None:
        raise ConfigurationError(f"Empty config file: {config_path}")

    logger.debug("Loaded config for %s from %s", jurisdiction.value, config_path)
    return data
