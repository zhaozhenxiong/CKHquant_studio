"""Public defaults work without the publisher's private model/data directories.

The subprocess uses only generated provider-schema records and a disclosed
synthetic schedule. It exercises the real native engine, routing, SQLite stores
and background workers; it does not establish market effectiveness.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_APP_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_CURRENT = _APP_ROOT / "my_strategy/configs/current_strategy.json"

_PROBE = r'''
import hashlib
import faulthandler
import json
import math
import os
from pathlib import Path
import sqlite3
import time

faulthandler.dump_traceback_later(90)

import pandas as pd
from fastapi.testclient import TestClient

from my_strategy.services.czsc_trend_contracts import STRATEGY_ID
from my_strategy.services.czsc_trend_runtime import load_model_run
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.app import create_app
from my_strategy.web_dashboard.tasks import TaskManager

state = Path(os.environ["PORTABLE_DEFAULT_TEST_STATE"])
runs = Path(os.environ["KHQUANT_ARTIFACT_ROOT"]) / "runs"
assert runs.is_dir() and list(runs.iterdir()) == []
assert load_model_run() == (None, None)
try:
    load_model_run("explicit-missing-model")
except ValueError as exc:
    missing_model_error = str(exc)
else:
    raise AssertionError("Explicit unknown model silently became an automatic selection")

dates = pd.bdate_range("2024-01-02", periods=180).strftime("%Y-%m-%d").tolist()
raw = Path(os.environ["KHQUANT_RAW_DB"])
raw.parent.mkdir(parents=True)
# These generated records simulate the verified-provider schema in a disposable
# test database. They are never downloaded prices or a claim about Tushare data.
records = []
for index, day in enumerate(dates):
    price = 12 + .006 * index + 1.5 * math.sin(index / 7)
    volume = 1_000_000. * (1 + .25 * math.sin(index / 9))
    records.append(("000001.SZ", day, price, price + .15, price - .15,
                    price + .04, volume, price * volume, 1, "tushare"))
with sqlite3.connect(raw) as connection:
    connection.execute("CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,has_trade_price INTEGER,source TEXT)")
    connection.execute("CREATE TABLE securities(stock TEXT,name TEXT,first_date TEXT,last_date TEXT,raw_rows INTEGER)")
    connection.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)", records)
    connection.execute("INSERT INTO securities VALUES (?,?,?,?,?)", ("000001.SZ", "合成接口测试股", dates[0], dates[-1], len(dates)))
calendar_id = "portable-defaults-synthetic-calendar"
calendar_root = runs / calendar_id
reports = calendar_root / "reports"
reports.mkdir(parents=True)
source = calendar_root / "synthetic-schedule.json"
source.write_text(json.dumps({"dates": dates, "fixture": "synthetic_test_schedule_only"}), encoding="utf-8")
(reports / "calendar.json").write_text(json.dumps({"verified": True,
    "source": "synthetic_test_schedule_only", "dates": dates,
    "independent_from_market_bars": False, "fixture": True,
    "source_artifacts": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}), encoding="utf-8")

manager = TaskManager(state / "tasks.db", ResultStore(state / "results.db", runs), workers=1)

def wait_job(client, job_id):
    deadline = time.monotonic() + 90
    while True:
        response = client.get("/api/tasks/" + job_id)
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] not in {"queued", "running", "cancelling"}:
            return job
        assert time.monotonic() < deadline, job
        time.sleep(.02)

with TestClient(create_app(task_manager=manager)) as client:
    response = client.get("/api/research/high-return-status")
    assert response.status_code == 200, response.text
    status = response.json()
    request = {"symbol": "000001.SZ", "start": dates[0], "end": dates[-1],
        "research": True, "model_family": "high_return", "calendar_run_id": calendar_id, "device": "cpu"}
    response = client.post("/api/analysis", json=request)
    assert response.status_code == 200, response.text
    analysis = response.json()
    saved = client.get("/api/runs/" + analysis["run_id"])
    assert saved.status_code == 200 and saved.json() == analysis

    spec = {"symbols": ["000001.SZ"], "start": dates[0], "end": dates[-1],
        "research": True, "model_family": "high_return", "calendar_run_id": calendar_id,
        "device": "cpu", "cpu_workers": 1}
    response = client.post("/api/tasks", json={"kind": "scan", "spec": spec})
    assert response.status_code == 202, response.text
    scan = wait_job(client, response.json()["job_id"])
    # A model-free research display is valid; model-based account certification
    # explicitly requires a complete selected model and must not change families.
    response = client.post("/api/tasks", json={"kind": "high_return_backtest", "spec": spec})
    offline = {"status_code": response.status_code, "body": response.json()}
    response = client.post("/api/tasks", json={"kind": "backtest", "spec": spec})
    assert response.status_code == 202, response.text
    backtest = wait_job(client, response.json()["job_id"])
    bad = client.post("/api/analysis", json={**request, "model_run_id": "explicit-missing-model",
        "consensus_version": STRATEGY_ID})
    bad_default = client.post("/api/analysis", json={**request, "model_run_id": "explicit-missing-model"})
    bad_task = client.post("/api/tasks", json={"kind": "high_return_backtest", "spec": {
        **spec, "model_run_id": "explicit-missing-model", "consensus_version": STRATEGY_ID}})
    result = {"status": status, "analysis": analysis, "scan": scan, "backtest": backtest,
        "offline": offline, "missing_model_error": missing_model_error,
        "bad_model": {"status_code": bad.status_code, "body": bad.json()},
        "bad_default_model": {"status_code": bad_default.status_code, "body": bad_default.json()},
        "bad_task_model": {"status_code": bad_task.status_code, "body": bad_task.json()},
        "fixture": "synthetic_verified_provider_schema_and_synthetic_schedule_only"}
    print("PORTABLE_DEFAULT_PROBE=" + json.dumps(result, ensure_ascii=False, allow_nan=False))
'''


@pytest.fixture(scope="module")
def portable_defaults_probe(tmp_path_factory):
    """Fresh interpreter avoids conftest's deliberate workstation-config veto."""
    current = json.loads(_PUBLIC_CURRENT.read_text(encoding="utf-8"))
    assert current["strategy"] == "czsc_trend_consensus"
    assert current["model_run_id"] is None
    assert current["model_selection"] == "automatic_qualified_runs"
    state = tmp_path_factory.mktemp("portable-trend-defaults")
    (state / "artifacts/runs").mkdir(parents=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("KHQUANT_") and key not in {"PYTHONPATH", "PYTHONHOME", "CZSC_HOME"}}
    environment.update(PYTHONPATH=str(_APP_ROOT), KHQUANT_HOME=str(state),
        KHQUANT_DATA_ROOT=str(state / "data"), KHQUANT_ARTIFACT_ROOT=str(state / "artifacts"),
        KHQUANT_METADATA_ROOT=str(state / "metadata"), KHQUANT_LOG_ROOT=str(state / "logs"),
        KHQUANT_RAW_DB=str(state / "data/raw.db"), KHQUANT_COMPUTE_DEVICE="cpu",
        KHQUANT_CURRENT_STRATEGY_CONFIG=str(_PUBLIC_CURRENT), PORTABLE_DEFAULT_TEST_STATE=str(state),
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run([sys.executable, "-X", "utf8", "-c", _PROBE], cwd=state,
        env=environment, text=True, encoding="utf-8", capture_output=True, timeout=180, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    prefix = "PORTABLE_DEFAULT_PROBE="
    lines = [line for line in completed.stdout.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1, completed.stdout + completed.stderr
    return json.loads(lines[0][len(prefix):])


def _assert_untrained_probabilities(row):
    ma = row["agents"]["ma"]
    assert ma["model_available"] is False
    assert all(ma[key] is None for key in ("p_up", "p_down", "p_neutral"))
    assert ma["in_scope"] is False
    assert row["agents"]["high_return"]["probability"] is None
    assert row["high_return_probability"] is None


def test_public_default_analysis_preserves_real_rules_without_fake_model_scores(portable_defaults_probe):
    probe = portable_defaults_probe
    status = probe["status"]
    assert status["default_consensus_version"] == "czsc_trend_consensus"
    assert status["models"] == [] and status["current_model_run_id"] is None
    assert status["model_selection"] == "automatic_qualified_runs"
    result = probe["analysis"]
    assert result["strategy_version"] == "czsc_trend_consensus"
    assert result["research"]["model_run_id"] is None
    assert result["request"]["consensus_version"] == "czsc_trend_consensus"
    assert result["request"]["entry_policy"] == "risk"
    assert result["publication_allowed"] is False
    rows = result["chart_evidence"]["rows"]
    assert len(rows) == 180
    for row in rows:
        _assert_untrained_probabilities(row)
    for source in ("structure", "wyckoff"):
        scored = [row["agents"][source] for row in rows if row["agents"][source]["status"] == "scored"]
        assert scored
        assert any(max(agent["buy_strength"], agent["sell_strength"]) > 0 for agent in scored)
        assert all(agent["kind"] == "rule_evidence" for agent in scored)
    assert result["account_actions"]["ledger"] == []


def test_public_default_scan_worker_uses_same_untrained_trend_contract(portable_defaults_probe):
    job = portable_defaults_probe["scan"]
    assert job["status"] == "succeeded", job["error"]
    result = job["result"]
    assert result["strategy_version"] == "czsc_trend_consensus"
    assert result["model_run_id"] is None
    assert result["coverage"] == {"requested": 1, "successful": 1, "success": 1, "failed": 0}
    assert result["publication_allowed"] is False
    row, = result["rows"]
    _assert_untrained_probabilities(row)
    assert row["action"] in {"WAIT", "HOLD"} and row["category"] == "watch"
    assert row["holding"] is None and row["account_action"]["emitted"] is False


def test_missing_model_backtest_is_explicitly_rejected_without_old_family_fallback(portable_defaults_probe):
    probe = portable_defaults_probe
    assert probe["offline"]["status_code"] == 422
    assert "需要明确" in json.dumps(probe["offline"]["body"], ensure_ascii=False)
    job = probe["backtest"]
    assert job["status"] == "failed"
    assert "须使用显式离线 high_return_backtest" in job["error"]
    assert job["result"] is None and job["run_id"] is None
    assert "unavailable" in probe["missing_model_error"]
    for key in ("bad_model", "bad_task_model"):
        assert probe[key]["status_code"] in {400, 422}, probe[key]
    # An unknown saved-run identity without an explicit strategy uses the
    # loader's unavailable response, and cannot return another model's scores.
    assert probe["bad_default_model"]["status_code"] in {400, 422, 503}
