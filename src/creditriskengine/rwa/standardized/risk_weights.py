"""Jurisdiction-configurable risk weight tables — BCBS d424, CRE20.

This module provides a registry for SA risk weights that can be loaded
from jurisdiction-specific YAML configurations, enabling multi-jurisdiction
support without code changes.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from creditriskengine.core.types import (
    CreditQualityStep,
    Jurisdiction,
    SAExposureClass,
)

logger = logging.getLogger(__name__)

# Default config directory — sibling ``configs/`` package
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "jurisdictions"


class RiskWeightRegistry:
    """Registry for SA risk weights loaded from YAML config.

    The registry translates a nested YAML structure into fast lookup
    tables keyed by ``(SAExposureClass, CreditQualityStep)`` or by
    LTV buckets for real-estate classes.

    Expected YAML schema (top-level ``sa_risk_weights`` key)::

        sa_risk_weights:
          sovereign:
            cqs_1: 0.0
            cqs_2: 20.0
            ...
          rre_whole_loan:
            - {ltv_upper: 0.50, rw: 20.0}
            - {ltv_upper: 0.60, rw: 25.0}
            ...
          cre_not_cashflow:
            - {ltv_upper: 0.60, rw: 60.0}
            ...
          cre_ipre:
            - {ltv_upper: 0.60, rw: 70.0}
            ...
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._rw_tables: dict[str, dict[int, float]] = {}
        self._ltv_tables: dict[str, list[tuple[float, float]]] = {}
        self._load_tables()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_tables(self) -> None:
        """Parse YAML config into internal lookup tables."""
        sa_cfg = self._config.get("sa_risk_weights", {})
        if not sa_cfg:
            logger.warning("No 'sa_risk_weights' key found in config; registry will be empty")
            return

        # CQS-keyed tables (sovereign, bank_ecra, corporate, ...)
        _cqs_key_map: dict[str, int] = {
            "cqs_1": CreditQualityStep.CQS_1,
            "cqs_2": CreditQualityStep.CQS_2,
            "cqs_3": CreditQualityStep.CQS_3,
            "cqs_4": CreditQualityStep.CQS_4,
            "cqs_5": CreditQualityStep.CQS_5,
            "cqs_6": CreditQualityStep.CQS_6,
            "unrated": CreditQualityStep.UNRATED,
        }

        for table_name, table_data in sa_cfg.items():
            if isinstance(table_data, dict):
                # CQS-keyed table
                parsed: dict[int, float] = {}
                for key, rw in table_data.items():
                    cqs_val = _cqs_key_map.get(str(key).lower())
                    if cqs_val is not None:
                        parsed[cqs_val] = float(rw)
                    else:
                        logger.debug("Skipping unrecognised CQS key '%s' in table '%s'", key, table_name)
                self._rw_tables[table_name] = parsed
                logger.debug("Loaded CQS table '%s' with %d entries", table_name, len(parsed))

            elif isinstance(table_data, list):
                # LTV-bucket table — list of {ltv_upper, rw}
                ltv_entries: list[tuple[float, float]] = []
                for entry in table_data:
                    ltv_upper = float(entry.get("ltv_upper", float("inf")))
                    rw = float(entry["rw"])
                    ltv_entries.append((ltv_upper, rw))
                ltv_entries.sort(key=lambda t: t[0])
                self._ltv_tables[table_name] = ltv_entries
                logger.debug("Loaded LTV table '%s' with %d buckets", table_name, len(ltv_entries))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    _EXPOSURE_CLASS_TABLE_MAP: dict[str, str] = {
        SAExposureClass.SOVEREIGN: "sovereign",
        SAExposureClass.BANK: "bank_ecra",
        SAExposureClass.SECURITIES_FIRM: "bank_ecra",
        SAExposureClass.CORPORATE: "corporate",
        SAExposureClass.CORPORATE_SME: "corporate",
        SAExposureClass.PSE: "pse",
    }

    def get_risk_weight(
        self,
        exposure_class: SAExposureClass,
        cqs: CreditQualityStep | None = None,
        **kwargs: Any,
    ) -> float:
        """Look up risk weight from registry.

        Args:
            exposure_class: SA exposure class.
            cqs: Credit quality step (required for CQS-keyed classes).
            **kwargs: Additional parameters (e.g. ``ltv`` for real estate).

        Returns:
            Risk weight as a percentage (e.g. 20.0 for 20 %).

        Raises:
            KeyError: If the exposure class has no table in this registry.
            ValueError: If required parameters are missing.
        """
        table_name = self._EXPOSURE_CLASS_TABLE_MAP.get(exposure_class)
        if table_name and table_name in self._rw_tables:
            cqs_val = cqs.value if cqs is not None else CreditQualityStep.UNRATED
            rw = self._rw_tables[table_name].get(cqs_val)
            if rw is not None:
                return rw
            logger.warning(
                "CQS %s not found in table '%s'; returning 100.0",
                cqs_val,
                table_name,
            )
            return 100.0

        if exposure_class == SAExposureClass.RESIDENTIAL_MORTGAGE:
            ltv = kwargs.get("ltv")
            if ltv is None:
                raise ValueError("LTV required for residential mortgage risk weight")
            return self.get_rre_risk_weight(
                ltv, cashflow_dependent=kwargs.get("cashflow_dependent", False)
            )

        if exposure_class == SAExposureClass.COMMERCIAL_REAL_ESTATE:
            ltv = kwargs.get("ltv")
            if ltv is None:
                raise ValueError("LTV required for commercial real estate risk weight")
            return self.get_cre_risk_weight(
                ltv, cashflow_dependent=kwargs.get("cashflow_dependent", False)
            )

        raise KeyError(
            f"No risk weight table loaded for exposure class '{exposure_class}'"
        )

    def get_rre_risk_weight(
        self, ltv: float, cashflow_dependent: bool = False
    ) -> float:
        """Residential RE risk weight by LTV bucket.

        Reference: BCBS CRE20.71-20.86, Tables 12-13.

        Args:
            ltv: Loan-to-value ratio (e.g. 0.75 for 75 %).
            cashflow_dependent: Use cashflow-dependent table if True.

        Returns:
            Risk weight as percentage.
        """
        table_key = "rre_cashflow" if cashflow_dependent else "rre_whole_loan"
        return self._lookup_ltv(table_key, ltv)

    def get_cre_risk_weight(
        self, ltv: float, cashflow_dependent: bool = False
    ) -> float:
        """Commercial RE risk weight by LTV bucket.

        Reference: BCBS CRE20.87-20.98, Tables 14-15.

        Args:
            ltv: Loan-to-value ratio.
            cashflow_dependent: Use IPRE table if True.

        Returns:
            Risk weight as percentage.
        """
        table_key = "cre_ipre" if cashflow_dependent else "cre_not_cashflow"
        return self._lookup_ltv(table_key, ltv)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lookup_ltv(self, table_key: str, ltv: float) -> float:
        """Lookup risk weight in an LTV-bucket table.

        Args:
            table_key: Internal table name.
            ltv: Loan-to-value ratio.

        Returns:
            Risk weight as percentage.

        Raises:
            KeyError: If the table does not exist in the registry.
        """
        buckets = self._ltv_tables.get(table_key)
        if buckets is None:
            raise KeyError(f"LTV table '{table_key}' not found in registry")
        for ltv_upper, rw in buckets:
            if ltv <= ltv_upper:
                return rw
        # Beyond last bucket — return last entry's risk weight
        return buckets[-1][1]


def load_risk_weight_registry(
    jurisdiction: Jurisdiction,
    config_dir: Path | None = None,
) -> RiskWeightRegistry:
    """Load risk weight registry for a jurisdiction from its YAML config.

    Looks for ``<config_dir>/<jurisdiction>.yaml`` and parses its
    ``sa_risk_weights`` section.

    Args:
        jurisdiction: Target jurisdiction.
        config_dir: Override for the config directory path.

    Returns:
        Populated ``RiskWeightRegistry``.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    base = config_dir or _CONFIG_DIR
    config_path = base / f"{jurisdiction.value}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Jurisdiction config not found: {config_path}"
        )

    logger.info("Loading risk-weight config for '%s' from %s", jurisdiction.value, config_path)

    with open(config_path, encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    return RiskWeightRegistry(config)
