"""Pytest options shared across the KHQuant test suite.

This conftest enables the ``--mode`` flag used by the CI guard and
``.github/workflows/governance-adr.yml`` to run strategy acceptance tests in a
fast ``leakage-only`` mode.
"""
from __future__ import annotations

import pytest
import os
import tempfile
from pathlib import Path


# Collection imports application modules which initialize SQLite on import.
# Configure defaults here, before any my_strategy import, not in a fixture.
# Never inherit a checkout's runtime directories, including when a module
# entrypoint has already loaded .env before pytest starts collecting tests.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="khquant-pytest-"))
os.environ["KHQUANT_DATA_ROOT"] = str(_TEST_ROOT / "data")
os.environ["KHQUANT_ARTIFACT_ROOT"] = str(_TEST_ROOT / "artifacts")
os.environ["KHQUANT_CURRENT_STRATEGY_CONFIG"] = str(_TEST_ROOT / "test-current-strategy.json")
_TEST_DATA = Path(os.environ["KHQUANT_DATA_ROOT"])
for _key, _path in {
    "KHQUANT_OUTPUT_ROOT": _TEST_DATA / "processed" / "ai_agent_screen",
    "KHQUANT_LOG_ROOT": _TEST_DATA / "processed" / "logs",
    "KHQUANT_METADATA_ROOT": _TEST_DATA / "metadata",
    "KHQUANT_RAW_DB": _TEST_DATA / "raw" / "khquant_raw.db",
    "KHQUANT_PROCESSED_DB": _TEST_DATA / "processed" / "warehouse" / "khquant.db",
}.items():
    os.environ[_key] = str(_path)


LEAKAGE_ONLY_TESTS = {
    "test_leakage_acceptance_passes_project_features",
    "test_leakage_acceptance_blocks_future_named_features",
}


@pytest.fixture(autouse=True)
def isolated_delivered_strategy_selection(monkeypatch):
    """Never let a workstation's current model pointer enter test routing."""
    from my_strategy.web_dashboard import trend_routes
    from my_strategy.services import czsc_trend_runtime
    current = _TEST_ROOT / 'test-current-strategy.json'
    monkeypatch.setattr(trend_routes, 'CURRENT_PATH', current)
    monkeypatch.setattr(czsc_trend_runtime, 'CURRENT_MODEL_PATH', current)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow (integration tests)")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--mode",
        choices=["leakage-only", "full-backtest"],
        default=None,
        help="Run strategy acceptance tests in fast leakage-only mode or full backtest mode.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    mode = config.getoption("--mode")
    if mode != "leakage-only":
        return
    for item in items:
        if item.name not in LEAKAGE_ONLY_TESTS:
            item.add_marker(pytest.mark.skip(reason="skipped in leakage-only CI mode"))
