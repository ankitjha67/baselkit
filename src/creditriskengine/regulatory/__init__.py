"""Regulatory configuration loader.

Loads jurisdiction-specific YAML configuration files for Basel III parameters.
"""

from creditriskengine.regulatory.loader import get_config_path, load_config

__all__ = [
    "load_config",
    "get_config_path",
]
