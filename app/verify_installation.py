"""Exercise an installed distribution using isolated, explicitly synthetic inputs."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def model_probe(device: str) -> dict:
    import numpy as np
    import pandas as pd
    from my_strategy.services.czsc_research_ml import PredictorSession, train_model

    dates = pd.bdate_range("2024-01-02", periods=180)
    values = np.arange(len(dates))
    dataset = pd.DataFrame({"date": dates, "x": np.sin(values / 5), "z": np.cos(values / 7),
                            "label": values % 2, "label_end": dates + pd.offsets.BDay(1), "label_available": True})
    directory = Path(os.environ["KHQUANT_ARTIFACT_ROOT"]) / "installation-model"
    manifest = train_model(dataset, ["x", "z"], directory, str(dates[89].date()),
                           str(dates[90].date()), str(dates[139].date()), device=device,
                           epochs=2, batch_size=32, hidden_sizes=(8, 4), min_train_rows=30, min_validation_rows=20)
    predictor = PredictorSession(device=device, batch_size=32)
    probabilities = predictor.predict(dataset.iloc[140:].copy(), directory)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Installed model inference produced non-finite values")
    return {"training_device": manifest["device"], "inference": predictor.diagnostics,
            "prediction_rows": len(probabilities), "fixture": "synthetic_installation_only",
            "trend_consensus": trend_model_probe(device)}


def trend_model_probe(device: str) -> dict:
    """Exercise new serialization/inference targets with explicitly synthetic labels."""
    import numpy as np
    import pandas as pd
    from my_strategy.core.run_context import stable_hash
    from my_strategy.services.czsc_trend_contracts import strategy_config
    from my_strategy.services.czsc_trend_dataset import make_contract
    from my_strategy.services.czsc_trend_features import ENTRY_FEATURE_COLUMNS, L1_MA_EXIT_COLUMNS
    from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS
    from my_strategy.services.czsc_trend_model import TrendSession, SCHEMA_HASH, train_trend_head, trend_contract
    from my_strategy.services.czsc_trend_interval_model import HeadSession, bind_inputs, train_head

    dates = pd.bdate_range("2023-01-02", periods=190)
    days = dates.strftime("%Y-%m-%d").tolist()
    columns = list(dict.fromkeys([*ENTRY_FEATURE_COLUMNS, *L1_MA_EXIT_COLUMNS]))
    frame = pd.DataFrame({key: np.sin(np.arange(len(days)) / 7 + index) for index, key in enumerate(columns)})
    frame = frame.assign(date=days, symbol="INSTALLATION.TEST", label=np.arange(len(days)) % 3,
                         label_end=(dates + pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
                         label_available=True, input_eligible=True, model_input_eligible=True,
                         observation_id=[stable_hash({"synthetic_day": day}) for day in days],
                         economic_group_id=[f"synthetic_episode_{index}" for index in range(len(days))])
    cfg = strategy_config({"epochs": 2, "batch_size": 32, "min_train_rows": 30, "min_calibration_rows": 20,
        "trend": {"min_train_per_class": 2, "min_calibration_per_class": 2, "min_policy_per_class": 2}})
    cfg.update(source_snapshot_hash="synthetic-installation-source", pipeline_contract_hash="synthetic-installation-protocol",
               candidate_policy_hash="synthetic-installation-candidates", E0_policy_hash="synthetic-E0",
               E1_policy_hash="synthetic-E1-pinned-exit")
    fold = {"name": "installation_only", "train_end": days[89], "calibration_start": days[91],
            "calibration_end": days[119], "policy_start": days[121], "policy_end": days[149],
            "validation_end": days[149]}
    directory = Path(os.environ["KHQUANT_ARTIFACT_ROOT"]) / "installation-trend-model"
    contract = trend_contract(cfg, source_snapshot_hash=cfg["source_snapshot_hash"],
                              pipeline_contract_hash=cfg["pipeline_contract_hash"])
    frame.attrs["label_contract"] = contract
    manifest = train_trend_head(frame, directory / "direction", fold, cfg, contract=contract)
    probe = frame.iloc[151:155].copy()
    probe.attrs["schema_hash"] = SCHEMA_HASH
    first, _ = TrendSession().predict(probe, directory / "direction", expected_contract=contract)
    reloaded, _ = TrendSession().predict(probe, directory / "direction", expected_contract=contract)
    if first.shape != (4, 3) or not np.isfinite(first).all() or not np.allclose(first.sum(axis=1), 1.) or not np.array_equal(first, reloaded):
        raise RuntimeError("Installed three-class trend serialization/inference failed")
    heads = {"trend_direction": {"classes": manifest["classes"], "input_columns": len(MA_TREND_COLUMNS),
              "prediction_rows": len(first), "save_reload_identical": True, "actual_device": "cpu"}}
    for head, inputs, stage in (("ma_exit", L1_MA_EXIT_COLUMNS, "E0"), ("entry", ENTRY_FEATURE_COLUMNS, "E1")):
        target = make_contract(head, cfg, days, stage)
        dataset = frame.copy()
        dataset["label"] = np.arange(len(dataset)) % 2
        dataset.attrs["label_contract"] = target
        saved = train_head(dataset, inputs, directory / head, fold, cfg, head=head,
                           architecture="linear", device=device, contract=target)
        probe = bind_inputs(dataset.iloc[151:155], inputs, head)
        first = HeadSession().predict(probe, directory / head, expected_head=head, expected_contract=target)
        reloaded = HeadSession().predict(probe, directory / head, expected_head=head, expected_contract=target)
        if not np.isfinite(first).all() or not ((first >= 0) & (first <= 1)).all() or not np.array_equal(first, reloaded):
            raise RuntimeError(f"Installed {head} serialization/inference failed")
        heads[head] = {"target": target["target"], "stage": stage, "input_columns": len(inputs),
                       "prediction_rows": len(first), "save_reload_identical": True,
                       "actual_device": saved["actual_device"]}
    return {"heads": heads, "fixture": "synthetic_installation_labels_only",
            "market_effectiveness_certified": False, "publication_allowed": False}


def request(base: str, path: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    with urlopen(Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"}), timeout=30) as response:
        payload = response.read()
        return json.loads(payload) if response.headers.get_content_type() == "application/json" else payload.decode()


def fixture_database(path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    date, rows = dt.date(2024, 1, 2), []
    while len(rows) < 260:
        if date.weekday() < 5:
            price = 12 + len(rows) * 0.003 + math.sin(len(rows) / 11)
            rows.append(("000001.SZ", date.isoformat(), price, price + 0.2, price - 0.2,
                         price + 0.03, 1000000.0, price * 1000000, 1, "synthetic_installation_fixture"))
        date += dt.timedelta(days=1)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,has_trade_price INTEGER,source TEXT)")
        connection.execute("CREATE TABLE securities(stock TEXT,name TEXT,first_date TEXT,last_date TEXT,raw_rows INTEGER)")
        connection.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        connection.execute("INSERT INTO securities VALUES (?,?,?,?,?)", ("000001.SZ", "安装测试数据", rows[0][1], rows[-1][1], len(rows)))
    return [row[1] for row in rows]


def fixture_calendar(state: Path, dates: list[str]) -> str:
    """A fixture schedule accepted by calendar validation, never a market claim."""
    import hashlib
    run_id = "installation-synthetic-calendar"
    directory = state / "artifacts/runs" / run_id
    directory.mkdir(parents=True)
    source = directory / "synthetic-schedule.json"
    source.write_text(json.dumps({"dates": dates, "fixture": "synthetic_installation_schedule_only"}), encoding="utf-8")
    reports = directory / "reports"
    reports.mkdir()
    (reports / "calendar.json").write_text(json.dumps({"verified": True, "source": "synthetic_installation_schedule_only",
        "dates": dates, "independent_from_market_bars": False, "fixture": True,
        "source_artifacts": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}), encoding="utf-8")
    return run_id


def verify_no_model_trend(base: str, calendar_id: str, end: str) -> dict:
    status = request(base, "/api/research/high-return-status")
    if (status.get("default_consensus_version") != "czsc_trend_consensus" or status.get("current_model_run_id") is not None
            or status.get("models") or status.get("model_selection") != "automatic_qualified_runs"):
        raise RuntimeError("Public defaults must select trend consensus without a workstation model")
    result = request(base, "/api/analysis", "POST", {"symbol": "000001.SZ", "start": "2024-01-01", "end": end,
        "research": True, "model_family": "high_return", "calendar_run_id": calendar_id, "device": "cpu"})
    rows = result.get("chart_evidence", {}).get("rows", [])
    if not rows or result.get("strategy_version") != "czsc_trend_consensus" or result.get("research", {}).get("model_run_id") is not None:
        raise RuntimeError("Untrained installation failed its actual default trend analysis")
    for row in rows:
        agent = row["agents"]["ma"]
        if agent.get("model_available") is not False or any(agent.get(key) is not None for key in ("p_up", "p_down", "p_neutral")):
            raise RuntimeError("Untrained installation fabricated MA probabilities")
        if row["agents"]["high_return"].get("probability") is not None:
            raise RuntimeError("Untrained installation fabricated high-return probability")
    return {"status": "passed", "strategy": result["strategy_version"], "model_run_id": None,
        "ma_model_available": False, "probabilities": None, "analysis_rows": len(rows),
        "run_id": result["run_id"], "publication_allowed": result.get("publication_allowed"),
        "fixture": "synthetic_unverified_prices_and_synthetic_schedule_only"}


def verify(device: str, output: Path | None) -> dict:
    state = Path(tempfile.mkdtemp(prefix="khquant-install-smoke-")).resolve()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("KHQUANT_") and key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(KHQUANT_HOME=str(state), KHQUANT_DATA_ROOT=str(state / "data"),
                       KHQUANT_ARTIFACT_ROOT=str(state / "artifacts"), KHQUANT_METADATA_ROOT=str(state / "data/metadata"),
                       KHQUANT_RAW_DB=str(state / "data/raw/khquant_raw.db"), KHQUANT_LOG_ROOT=str(state / "data/logs"),
                       KHQUANT_COMPUTE_DEVICE=device, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    dates = fixture_database(Path(environment["KHQUANT_RAW_DB"]))
    calendar_id = fixture_calendar(state, dates)

    def command(*args: str, json_output: bool = True):
        completed = subprocess.run([sys.executable, "-I", "-X", "utf8", *args], cwd=state, env=environment,
                                   text=True, encoding="utf-8", capture_output=True, check=False, timeout=180)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        return json.loads(completed.stdout) if json_output else completed.stdout

    doctor = command("-m", "my_strategy.cli", "doctor", "--check-write", "--strict", "--json")
    model = command(str(Path(__file__).resolve()), "--internal-probe", "--device", device)
    command("-m", "my_strategy.cli", "update-data", "--help", json_output=False)
    analysis = command("-m", "my_strategy.cli", "analyze", "--symbol", "000001.SZ", "--start", "2024-01-01", "--structure-only", "--device", device, "--json")
    scan = command("-m", "my_strategy.cli", "scan", "--stocks", "000001.SZ", "--start", "2024-01-01", "--structure-only", "--device", device, "--cpu-workers", "1", "--json")
    backtest = command("-m", "my_strategy.cli", "backtest", "--symbol", "000001.SZ", "--start", "2024-01-01", "--structure-only", "--device", device, "--cpu-workers", "1", "--json")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    process = None
    pages = ("analysis", "scan", "backtest", "watchlist", "holdings", "data")
    with (state / "server.log").open("w", encoding="utf-8") as log:
        for restart in range(2):
            process = subprocess.Popen([sys.executable, "-I", "-X", "utf8", "-m", "my_strategy.web_dashboard.scripts.serve_dashboard", "--port", str(port)],
                                       cwd=state, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        request(base, "/api/health")
                        break
                    except (URLError, TimeoutError, ConnectionResetError):
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError(f"Dashboard did not become healthy: {state / 'server.log'}")
                        time.sleep(0.1)
                for page in pages:
                    if "<html" not in request(base, "/" + page).lower():
                        raise RuntimeError(f"Missing installed dashboard page: {page}")
                if restart == 0:
                    trend_without_model = verify_no_model_trend(base, calendar_id, dates[-1])
                    request(base, "/api/watchlist", "POST", {"symbols": ["000001.SZ"]})
                    request(base, "/api/holdings/000001.SZ", "PUT", {"shares": 100, "average_cost": 12})
                else:
                    if "000001.SZ" not in json.dumps(request(base, "/api/watchlist")) or "000001.SZ" not in json.dumps(request(base, "/api/holdings")):
                        raise RuntimeError("Personal data did not survive dashboard restart")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    report = {"status": "passed", "python": sys.executable, "state": str(state), "device": device,
              "doctor": doctor, "model": model, "dashboard_pages": list(pages), "personal_restart": True,
              "analysis_run": analysis.get("run_id"), "scan_run": scan.get("run_id"), "backtest_run": backtest.get("run_id"),
              "update_entrypoint": True,
              "default_trend_without_model": trend_without_model,
              "input": "synthetic_installation_fixture; does not certify market data or returns"}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    from my_strategy.runtime_env import configure_utf8_stdio

    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--internal-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(json.dumps(model_probe(args.device) if args.internal_probe else verify(args.device, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
