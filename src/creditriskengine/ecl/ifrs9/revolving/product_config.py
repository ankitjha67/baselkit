"""Default product configurations for revolving credit types.

Loads from ``regulatory/revolving_products.yml`` at module import.
Users can call :func:`load_revolving_product_configs` with a custom
YAML path to override defaults at runtime.

References:
    - PwC "In Depth" November 2017 (revolving ECL FAQs)
    - Araten & Jacobs (2001) on LEQ methodology
    - Tong et al. (2016) on credit card CCF distribution
    - BCBS d424 CRE32.22-32.24 (supervisory LGD)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from creditriskengine.ecl.ifrs9.revolving.types import RevolvingProductType

logger = logging.getLogger(__name__)

_DEFAULT_YAML = (
    Path(__file__).resolve().parents[3]
    / "regulatory"
    / "revolving_products.yml"
)


@dataclass(frozen=True)
class RevolvingProductConfig:
    """Default parameters for a revolving product type.

    Attributes:
        product_type: The revolving product classification.
        default_behavioral_life_months: Central estimate of behavioral
            life per B5.5.40, in months.
        behavioral_life_range: (min, max) plausible range in months.
        typical_ccf_range: (low, high) empirical CCF range.
        typical_lgd: Default LGD assumption for the product type.
        is_collectively_managed: True if managed on a portfolio basis
            (IFRS 9 para 5.5.20 applies); False if individually managed
            (para 5.5.19 applies, contractual period used).
        has_draw_period: True if the product has a distinct draw vs.
            repayment phase (e.g., HELOCs).
        draw_period_months: Length of the draw period, if applicable.
    """

    product_type: RevolvingProductType
    default_behavioral_life_months: int
    behavioral_life_range: tuple[int, int]
    typical_ccf_range: tuple[float, float]
    typical_lgd: float
    is_collectively_managed: bool
    has_draw_period: bool = False
    draw_period_months: int | None = None


def _parse_config(
    key: str, data: dict[str, Any],
) -> RevolvingProductConfig:
    """Parse a single product config entry from YAML data."""
    bl_range = data.get("behavioral_life_range", [24, 60])
    ccf_range = data.get("typical_ccf_range", [0.50, 0.95])
    return RevolvingProductConfig(
        product_type=RevolvingProductType(key),
        default_behavioral_life_months=int(
            data.get("default_behavioral_life_months", 36)
        ),
        behavioral_life_range=(int(bl_range[0]), int(bl_range[1])),
        typical_ccf_range=(float(ccf_range[0]), float(ccf_range[1])),
        typical_lgd=float(data.get("typical_lgd", 0.85)),
        is_collectively_managed=bool(
            data.get("is_collectively_managed", True)
        ),
        has_draw_period=bool(data.get("has_draw_period", False)),
        draw_period_months=(
            int(data["draw_period_months"])
            if data.get("draw_period_months") is not None
            else None
        ),
    )


def load_revolving_product_configs(
    yaml_path: Path | None = None,
) -> dict[RevolvingProductType, RevolvingProductConfig]:
    """Load revolving product configs from a YAML file.

    Args:
        yaml_path: Path to YAML file.  Defaults to the bundled
            ``regulatory/revolving_products.yml``.

    Returns:
        Dict mapping product type to its configuration.
    """
    path = yaml_path or _DEFAULT_YAML
    if not path.exists():
        logger.warning(
            "Revolving product config YAML not found at %s; "
            "using empty config",
            path,
        )
        return {}

    with open(path) as f:
        raw = yaml.safe_load(f)

    products_data: dict[str, Any] = raw.get("products", {})
    configs: dict[RevolvingProductType, RevolvingProductConfig] = {}
    for key, data in products_data.items():
        try:
            pt = RevolvingProductType(key)
            configs[pt] = _parse_config(key, data)
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping unknown product type %r: %s", key, exc)
    return configs


# Module-level singleton loaded from bundled YAML
PRODUCT_CONFIGS: dict[RevolvingProductType, RevolvingProductConfig] = (
    load_revolving_product_configs()
)


def get_product_config(
    product_type: RevolvingProductType,
) -> RevolvingProductConfig:
    """Return the default configuration for a revolving product type.

    Args:
        product_type: The revolving product classification.

    Returns:
        Frozen dataclass with default parameters.

    Raises:
        KeyError: If the product type has no configuration.
    """
    if product_type not in PRODUCT_CONFIGS:
        raise KeyError(
            f"No configuration for product type: {product_type!r}"
        )
    return PRODUCT_CONFIGS[product_type]
