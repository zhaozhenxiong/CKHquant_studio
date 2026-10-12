"""Exercise executable request routing without workstation models or databases."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from my_strategy import cli


@pytest.fixture
def cli_market(tmp_path):
    """Real read-only adapter input, isolated state and an unrelated launch cwd."""
    database = tmp_path / "data" / "raw" / "khquant_raw.db"
    database.parent.mkdir(parents=True)
    days = pd.bdate_range("2025-06-02", periods=190)
    prices = [12 + index * .008 + math.sin(index / 10) for index in range(len(days))]
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized (stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,has_trade_price INTEGER,source TEXT)")
        connection.execute("CREATE TABLE securities (stock TEXT,name TEXT,first_date TEXT,last_date TEXT,raw_rows INTEGER)")
        connection.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)", [
            ("000001.SZ", day.date().isoformat(), price, price + .15, price - .15,
             price + .04, 1000000., price * 1000000, 1, "test")
            for day, price in zip(days, prices, strict=True)
        ])
        connection.execute("INSERT INTO securities VALUES (?,?,?,?,?)", (
            "000001.SZ", "测试股", days[0].date().isoformat(), days[-1].date().isoformat(), len(days)))
    working_directory = tmp_path / "user-working-dir"
    working_directory.mkdir()
    env = {**os.environ,
           "PYTHONPATH": str(Path(cli.__file__).resolve().parents[1]),
           "KHQUANT_PROJECT_ROOT": str(Path(cli.__file__).resolve().parents[1]),
           "KHQUANT_DATA_ROOT": str(tmp_path / "data"),
           "KHQUANT_RAW_DB": str(database),
           "KHQUANT_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
           "KHQUANT_METADATA_ROOT": str(tmp_path / "data" / "metadata"),
           "KHQUANT_LOG_ROOT": str(tmp_path / "data" / "logs"),
           "KHQUANT_OUTPUT_ROOT": str(tmp_path / "data" / "processed" / "ai_agent_screen"),
           "KHQUANT_PROCESSED_DB": str(tmp_path / "data" / "processed" / "warehouse" / "khquant.db"),
           "KHQUANT_CURRENT_STRATEGY_CONFIG": str(tmp_path / "absent-current-strategy.json")}
    return working_directory, env, days[-1].date().isoformat()


@pytest.mark.parametrize("command", ["analyze", "scan", "backtest"])
def test_structure_only_executable_uses_native_contract_and_persists_result(cli_market, command):
    """Same command works after installation, away from a repository cwd."""
    cwd, env, end = cli_market
    arguments = [sys.executable, "-m", "my_strategy.cli", command, "--symbol", "000001.SZ",
                 "--structure-only", "--start", "2026-01-01", "--end", end,
                 "--device", "cpu", "--json"]
    if command != "analyze":
        arguments += ["--cpu-workers", "1"]
    completed = subprocess.run(arguments, cwd=cwd, env=env, text=True,
                               capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["run_context"]["source"] == "cli"
    assert result["strategy_version"] == "czsc_confirmed_trend_v1"
    assert result["run_id"]
    if command == "analyze":
        assert result["symbol"] == "000001.SZ" and result["data_end"] == end
        assert result["frequencies"]["日线"]["bars"]
    else:
        assert result["coverage"] == {"requested": 1, "success": 1, "failed": 0}
        assert result["failures"] == []
    reports = Path(env["KHQUANT_ARTIFACT_ROOT"]) / "runs" / result["run_id"] / "reports"
    assert any(reports.glob("*.json")), "executable must persist its real result"
    assert not list(cwd.iterdir()), "launching away from the checkout must not write to cwd"


@pytest.mark.parametrize("family,research", [("ma_trend", False), ("ma_trend", True),
                                             ("dual", True), ("wyckoff", True), ("high_return", True)])
def test_cli_filters_chart_consensus_only_for_native_analysis(monkeypatch, capsys, family, research):
    """Preserve high-return arbitration and the full request in audit context."""
    from my_strategy.core import run_context
    from my_strategy.services import czsc_analysis
    from my_strategy.storage import czsc_results
    from my_strategy.web_dashboard import high_return
    observed = {}
    def calculate(**spec):
        observed["calculation"] = spec
        return {"symbol": "000001.SZ", "data_end": "2026-09-03", "data_version": "test"}
    def context(**spec):
        observed["context"] = spec
        return SimpleNamespace(run_id="cli-routing-test", to_dict=lambda: {"source": "cli"})
    monkeypatch.setattr(czsc_analysis, "analyze_stock", calculate)
    monkeypatch.setattr(high_return, "analyze_high_return", calculate)
    monkeypatch.setattr(run_context, "create_run_context", context)
    monkeypatch.setattr(czsc_results, "ResultStore", lambda: SimpleNamespace(save=lambda *args: None))
    arguments = ["analyze", "--symbol", "000001.SZ", "--end", "2026-09-03",
                 "--model-family", family, "--json"]
    if not research:
        arguments.append("--structure-only")
    assert cli.main(arguments) == 0
    json.loads(capsys.readouterr().out)
    assert observed["calculation"]["research"] is research
    assert observed["calculation"]["model_family"] == family
    retained = family == "high_return" and research
    assert ("consensus_config" in observed["calculation"]) is retained
    assert ("consensus_version" in observed["calculation"]) is retained
    assert observed["context"]["config"]["consensus_version"]
    assert observed["context"]["config"]["consensus_config"]
