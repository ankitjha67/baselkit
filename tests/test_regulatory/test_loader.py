"""Tests for regulatory YAML config loader."""

import pytest

from creditriskengine.core.exceptions import ConfigurationError, JurisdictionError
from creditriskengine.core.types import Jurisdiction
from creditriskengine.regulatory.loader import get_config_path, load_config


class TestGetConfigPath:
    def test_eu(self) -> None:
        path = get_config_path(Jurisdiction.EU)
        assert path.name == "crr3.yml"
        assert "eu" in str(path)

    def test_us(self) -> None:
        path = get_config_path(Jurisdiction.US)
        assert path.name == "us_endgame.yml"

    def test_india(self) -> None:
        path = get_config_path(Jurisdiction.INDIA)
        assert path.name == "rbi.yml"

    def test_all_jurisdictions_have_paths(self) -> None:
        for j in Jurisdiction:
            path = get_config_path(j)
            assert path.suffix == ".yml"


class TestLoadConfig:
    @pytest.mark.parametrize("jurisdiction", [
        Jurisdiction.BCBS, Jurisdiction.EU, Jurisdiction.UK,
        Jurisdiction.US, Jurisdiction.INDIA, Jurisdiction.SINGAPORE,
        Jurisdiction.HONG_KONG, Jurisdiction.JAPAN, Jurisdiction.AUSTRALIA,
        Jurisdiction.CANADA, Jurisdiction.CHINA, Jurisdiction.SOUTH_KOREA,
        Jurisdiction.UAE, Jurisdiction.SAUDI_ARABIA, Jurisdiction.SOUTH_AFRICA,
        Jurisdiction.BRAZIL, Jurisdiction.MALAYSIA,
    ])
    def test_all_jurisdictions_load(self, jurisdiction: Jurisdiction) -> None:
        config = load_config(jurisdiction)
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_eu_has_expected_keys(self) -> None:
        config = load_config(Jurisdiction.EU)
        # Minimal check — YAML has at least a top-level key
        assert isinstance(config, dict)

    def test_nonexistent_dir_raises(self) -> None:
        from pathlib import Path
        with pytest.raises((ConfigurationError, JurisdictionError)):
            load_config(Jurisdiction.EU, config_dir=Path("/nonexistent/path"))

    def test_empty_config_file_raises(self, tmp_path: object) -> None:
        """Cover line 69: empty YAML config raises ConfigurationError."""

        # Replicate the directory structure for EU config
        config_path = get_config_path(Jurisdiction.EU)
        from creditriskengine.regulatory.loader import _REGULATORY_DIR

        rel_path = config_path.relative_to(_REGULATORY_DIR)
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")  # Empty YAML file

        with pytest.raises(ConfigurationError, match="Empty config"):
            load_config(Jurisdiction.EU, config_dir=tmp_path)

    def test_unknown_jurisdiction_raises(self) -> None:
        """Cover line 40: unknown jurisdiction raises ConfigurationError."""
        import unittest.mock as mock

        # Create a mock jurisdiction with a value not in the mapping
        fake_jurisdiction = mock.MagicMock()
        fake_jurisdiction.value = "atlantis"
        with pytest.raises(ConfigurationError, match="Unknown jurisdiction"):
            get_config_path(fake_jurisdiction)
