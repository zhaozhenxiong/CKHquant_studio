"""Actual current CLI analysis/scan/backtest with isolated synthetic market files."""
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pandas as pd
import pytest

from my_strategy import cli


@pytest.fixture
def market_state(tmp_path):
    database = tmp_path / "data/raw/khquant_raw.db"; database.parent.mkdir(parents=True)
    dates = pd.bdate_range("2025-06-02", periods=190)
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,has_trade_price INTEGER,source TEXT)")
        conn.execute("CREATE TABLE securities(stock TEXT,name TEXT,first_date TEXT,last_date TEXT,raw_rows INTEGER)")
        rows = []
        for i, day in enumerate(dates):
            price = 12 + i * .008 + math.sin(i / 10)
            rows.append(("000001.SZ", day.date().isoformat(), price, price + .15, price - .15, price + .04, 1e6, price * 1e6, 1, "synthetic_cli_fixture"))
        conn.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        conn.execute("INSERT INTO securities VALUES (?,?,?,?,?)", ("000001.SZ", "CLI fixture", rows[0][1], rows[-1][1], len(rows)))
    calendar = tmp_path / "artifacts/runs/synthetic-calendar/reports/calendar.json"
    calendar.parent.mkdir(parents=True)
    calendar.write_text(json.dumps({"verified": True, "source": "synthetic_cli_schedule_only", "dates": dates.strftime("%Y-%m-%d").tolist(), "fixture": True}))
    cwd = tmp_path / "unrelated-launch-directory"; cwd.mkdir()
    env = {k:v for k,v in os.environ.items() if not k.startswith("KHQUANT_")}
    env.update(PYTHONPATH=str(Path(cli.__file__).resolve().parents[1]), KHQUANT_RAW_DB=str(database),
        KHQUANT_PROJECT_ROOT=str(Path(cli.__file__).resolve().parents[1]), KHQUANT_DATA_ROOT=str(tmp_path / "data"),
        KHQUANT_ARTIFACT_ROOT=str(tmp_path / "artifacts"), KHQUANT_METADATA_ROOT=str(tmp_path / "data/metadata"),
        KHQUANT_LOG_ROOT=str(tmp_path / "data/logs"), KHQUANT_CURRENT_STRATEGY_CONFIG=str(tmp_path / "absent-current.json"))
    return cwd, env, rows[-1][1]


@pytest.mark.parametrize("command", ["analyze", "scan", "backtest"])
def test_current_cli_rebuilds_observation_and_persists_without_workstation_model(market_state, command):
    cwd, env, end = market_state
    completed = subprocess.run([sys.executable, "-m", "my_strategy.cli", command, "--symbol", "000001.SZ", "--start", "2026-01-01", "--end", end,
        "--calendar-run-id", "synthetic-calendar", "--device", "cpu", "--cpu-workers", "1", "--json"], cwd=cwd, env=env, capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["run_id"] and result["strategy_version"] == "trend_consensus"
    if command == "analyze":
        assert result["symbol"] == "000001.SZ" and result["data_end"] == end
        assert result["research"]["model_run_id"] is None
        assert result["chart_evidence"]["rows"]
        assert result["run_context"]["source"] == "cli"
    else:
        assert result["coverage"]["success"] == 1 and result["coverage"]["failed"] == 0
        assert not result["failures"]
    reports = Path(env["KHQUANT_ARTIFACT_ROOT"]) / "runs" / result["run_id"] / "reports"
    assert list(reports.glob("*.json"))
    assert not list(cwd.iterdir())
