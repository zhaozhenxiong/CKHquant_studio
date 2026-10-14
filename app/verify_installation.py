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
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def model_probe(device: str) -> dict:
    """Only current synthetic serialization operators, never retired training."""
    current = trend_model_probe(device)
    return {"training_device": current["heads"]["entry"]["actual_device"],
            "inference": {"device": current["heads"]["entry"]["inference_device"], "save_reload_identical": True},
            "prediction_rows": current["heads"]["entry"]["prediction_rows"],
            "fixture": "synthetic_installation_only", "trend_consensus": current}


def trend_model_probe(device: str) -> dict:
    """Exercise new serialization/inference targets with explicitly synthetic labels."""
    import numpy as np
    import pandas as pd
    from my_strategy.core.run_context import stable_hash
    from my_strategy.services.czsc_trend_contracts import strategy_config, TRAINING_POPULATION, SPLIT_POLICY, ECONOMIC_WEIGHTING
    from my_strategy.services.czsc_trend_dataset import make_contract
    from my_strategy.services.czsc_trend_features import ENTRY_FEATURE_COLUMNS, L1_MA_EXIT_COLUMNS
    from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS
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
    cfg["reference_population"] = {"version": TRAINING_POPULATION, "fold": fold,
        "population_end": fold["policy_end"], "prior_entry_catalog": [], "prior_entry_catalog_hash": stable_hash([]),
        "split_policy": SPLIT_POLICY, "economic_weighting": ECONOMIC_WEIGHTING}
    frame["attempt_id"] = [f"synthetic-attempt-{i}" for i in range(len(days))]
    frame["original_event_ids"] = [[f"synthetic-original-{i}"] for i in range(len(days))]
    frame["original_opportunity_ids"] = frame["original_event_ids"]
    frame["original_event_first_pre_gate_dates"] = [json.dumps({f"synthetic-original-{i}":day}) for i,day in enumerate(days)]
    directory = Path(os.environ["KHQUANT_ARTIFACT_ROOT"]) / "installation-trend-model"
    contract = trend_contract(cfg, source_snapshot_hash=cfg["source_snapshot_hash"],
                              pipeline_contract_hash=cfg["pipeline_contract_hash"])
    frame.attrs["label_contract"] = contract
    manifest = train_trend_head(frame, directory / "direction", fold, cfg, contract=contract)
    probe = frame.iloc[151:155].copy()
    probe.attrs["schema_hash"] = SCHEMA_HASH
    first_session, reloaded_session = TrendSession(), TrendSession()
    first, _ = first_session.predict(probe, directory / "direction", expected_contract=contract)
    reloaded, _ = reloaded_session.predict(probe, directory / "direction", expected_contract=contract)
    if first.shape != (4, 3) or not np.isfinite(first).all() or not np.allclose(first.sum(axis=1), 1.) or not np.array_equal(first, reloaded):
        raise RuntimeError("Installed three-class trend serialization/inference failed")
    heads = {"trend_direction": {"classes": manifest["classes"], "input_columns": len(MA_TREND_COLUMNS),
              "prediction_rows": len(first), "save_reload_identical": True, "actual_device": manifest["actual_device"],
              "training_device": manifest["actual_device"], "inference_device": first_session.diagnostics["actual_device"],
              "inference_diagnostics": dict(first_session.diagnostics),
              "reload_inference_diagnostics": dict(reloaded_session.diagnostics)}}
    for head, inputs, stage in (("ma_exit", L1_MA_EXIT_COLUMNS, "E0"), ("entry", ENTRY_FEATURE_COLUMNS, "E1")):
        target = make_contract(head, cfg, days, stage)
        dataset = frame.copy()
        dataset["label"] = np.arange(len(dataset)) % 2
        if head == "ma_exit": dataset["episode_id"] = [f"synthetic-held-{i}" for i in range(len(dataset))]
        dataset.attrs["label_contract"] = target
        saved = train_head(dataset, inputs, directory / head, fold, cfg, head=head,
                           architecture="linear", device=device, contract=target)
        probe = bind_inputs(dataset.iloc[151:155], inputs, head)
        first_session, reloaded_session = HeadSession(), HeadSession()
        first = first_session.predict(probe, directory / head, expected_head=head, expected_contract=target)
        reloaded = reloaded_session.predict(probe, directory / head, expected_head=head, expected_contract=target)
        if not np.isfinite(first).all() or not ((first >= 0) & (first <= 1)).all() or not np.array_equal(first, reloaded):
            raise RuntimeError(f"Installed {head} serialization/inference failed")
        heads[head] = {"target": target["target"], "stage": stage, "input_columns": len(inputs),
                       "prediction_rows": len(first), "save_reload_identical": True,
                       "actual_device": saved["actual_device"], "training_device": saved["actual_device"],
                       "inference_device": first_session.diagnostics["actual_device"],
                       "inference_diagnostics": dict(first_session.diagnostics),
                       "reload_inference_diagnostics": dict(reloaded_session.diagnostics),
                       "training_resources": saved.get("training_resources", {})}
    from my_strategy.services import czsc_reference_quality as quality
    quality_rows = []
    for index, day in enumerate(days):
        atom, label = f"synthetic-reference-{index}", index % 2
        quality_rows.append({**{key: 0. for key in quality.COLUMNS},
            quality.COLUMNS[0]: 4. if label else -4., "symbol": "INSTALLATION.TEST", "date": day,
            "label_end": (dates[index] + pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
            "label": label, "label_available": True, "input_eligible": True, "risk_available": True,
            "attempt_id": atom, "economic_group_id": atom, "opportunity_id": atom,
            "original_event_ids": json.dumps([atom]),
            "original_event_first_pre_gate_dates": json.dumps({atom: day})})
    quality_data = quality._frame(quality_rows)
    quality_contract = quality._target_contract(cfg, {
        "population_hash": "synthetic-installation-reference-population",
        "protocol_hash": "synthetic-installation-reference-protocol",
        "trend_catalog_hash": "synthetic-installation-reference-trend-catalog",
        "ma_exit_catalog_hash": "synthetic-installation-reference-exit-catalog"})
    quality_directory = directory / quality.HEAD
    quality_manifest = quality.train_reference_head(quality_data, quality_directory, fold, cfg, quality_contract)
    quality_bundles = [{"name": fold["name"], "available_at": quality_manifest["available_at"],
        "heads": {quality.HEAD: {"path": quality.HEAD, "manifest_sha256": quality_manifest["manifest_sha256"]}}}]

    def quality_predictions():
        cache = {}
        return np.array([quality._predict_prior(row, row["date"], quality_bundles, directory,
            quality_contract, cache)[0] for row in quality_data.iloc[151:155].to_dict("records")], dtype=float)

    first, reloaded = quality_predictions(), quality_predictions()
    if (first.shape != (4,) or not np.isfinite(first).all() or not ((first >= 0) & (first <= 1)).all()
            or not np.array_equal(first, reloaded)):
        raise RuntimeError("Installed independent reference quality serialization/inference failed")
    heads[quality.HEAD] = {"target": quality.TARGET, "population": quality.POPULATION,
        "input_columns": len(quality.COLUMNS), "account_input_columns": [],
        "prediction_rows": len(first), "save_reload_identical": True,
        "actual_device": quality_manifest["actual_device"], "training_device": "cpu", "inference_device": "cpu",
        "fee_qualified": False, "publication_allowed": False}
    return {"heads": heads, "fixture": "synthetic_installation_labels_only",
            "market_effectiveness_certified": False, "publication_allowed": False}


def request(base: str, path: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    with urlopen(Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"}), timeout=30) as response:
        payload = response.read()
        return json.loads(payload) if response.headers.get_content_type() == "application/json" else payload.decode()


def expect_removed_route(base: str, path: str, method: str = "GET", body: dict | None = None) -> None:
    try:
        request(base, path, method, body)
    except HTTPError as error:
        if error.code == 404:
            return
        raise RuntimeError(f"Retired route did not return 404: {method} {path} ({error.code})") from error
    raise RuntimeError(f"Retired route remains available: {method} {path}")


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
    status = request(base, "/api/training/status")
    if (status.get("mode") != "rule_observation_no_model" or status.get("current_selection") is not None
            or status.get("active_model") is not None or status.get("models")):
        raise RuntimeError("Public defaults must select trend consensus without a workstation model")
    result = request(base, "/api/analysis", "POST", {"symbol": "000001.SZ", "start": "2024-01-01", "end": end,
        "model_family": "current", "calendar_run_id": calendar_id, "device": "cpu"})
    rows = result.get("chart_evidence", {}).get("rows", [])
    if not rows or result.get("strategy_version") != "trend_consensus" or result.get("research", {}).get("model_run_id") is not None:
        raise RuntimeError("Untrained installation failed its actual default trend analysis")
    for row in rows:
        agent = row["agents"]["ma"]
        if agent.get("model_available") is not False or any(agent.get(key) is not None for key in ("p_up", "p_down", "p_neutral")):
            raise RuntimeError("Untrained installation fabricated MA probabilities")
        if row["agents"]["high_return"].get("probability") is not None:
            raise RuntimeError("Untrained installation fabricated high-return probability")
        if row["agents"].get("reference_quality", {}).get("probability") is not None:
            raise RuntimeError("Untrained installation fabricated reference quality probability")
    if (result.get("analysis_mode") != "reference" or result.get("account_scope") != "none"
            or result.get("account_actions") is not None or result.get("research", {}).get("account_simulated") is not False
            or result.get("chart_evidence", {}).get("statistics", {}).get("visible_window", {}).get("broker_fills") != 0):
        raise RuntimeError("Installed ordinary analysis unexpectedly simulated an account")
    source_points, final_points = result.get("source_reference_points"), result.get("reference_points")
    if not isinstance(source_points, list) or not isinstance(final_points, list):
        raise RuntimeError("Installed analysis omitted independent source/final B/S collections")
    for point in source_points + final_points:
        if (point.get("side") not in {"BUY", "SELL"} or point.get("reference_price_source") != "signal_day_close"
                or point.get("reference_price_date") != point.get("date")):
            raise RuntimeError("Installed B/S reference changed its direction or price clock")
    for row in rows:
        position = row.get("position_reference") or {}
        risk = row.get("reference_risk_plan") or {}
        if (position.get("account_scope") != "none" or position.get("reference_only") is not True
                or position.get("entry_probability") is not None or position.get("exit_probability") is not None
                or position.get("reference_quality_probability") is not None
                or position.get("reference_quality_threshold") != .1 or position.get("reference_buy_total_threshold") != .5
                or risk.get("risk_budget_fraction") != .01
                or position.get("target_position_fraction") != (0. if row.get("direction") == "SELL" else None)):
            raise RuntimeError("Untrained installation fabricated an allocation or weakened a reference gate")
    return {"status": "passed", "strategy": result["strategy_version"], "model_run_id": None,
        "ma_model_available": False, "probabilities": None, "analysis_rows": len(rows),
        "analysis_mode": "reference", "account_scope": "none", "account_simulated": False,
        "source_reference_points": len(source_points), "final_reference_points": len(final_points),
        "source_reference_sides": sorted({point["side"] for point in source_points}),
        "reference_quality_probability": None, "unqualified_allocation_blocked": True,
        "run_id": result["run_id"], "publication_allowed": result.get("publication_allowed"),
        "fixture": "synthetic_unverified_prices_and_synthetic_schedule_only"}



def verify_training_center(base: str) -> dict:
    """Read-only schema/default checks; never submit synthetic training as a job."""
    capabilities = request(base, "/api/training/capabilities")
    profiles = request(base, "/api/training/profiles")
    status = request(base, "/api/training/status")
    if capabilities.get("model_family") != "current" or capabilities.get("strategy_version") != "trend_consensus":
        raise RuntimeError("Installed training capabilities do not use the current strategy")
    fields = capabilities.get("schema", {}).get("fields")
    if not isinstance(fields, list) or not fields or len({field["key"] for field in fields}) != len(fields):
        raise RuntimeError("Installed default parameter schema is incomplete")
    by_key = {field["key"]: field for field in fields}
    declared_defaults = capabilities.get("defaults")
    if not isinstance(declared_defaults, dict) or set(declared_defaults) != set(by_key):
        raise RuntimeError("Installed editable parameter fields do not match the registered defaults")
    default = profiles.get("default_profile", {}).get("parameters", {})
    for key, floor in (("min_train_rows", 100), ("min_calibration_rows", 30), ("min_net_return", .08),
                       ("min_r_multiple", 2.), ("liquidity_lookback_sessions", 20), ("liquidity_min_median_amount", 30000000.)):
        if key not in by_key or by_key[key]["min"] < floor or default.get(key, 0) < floor:
            raise RuntimeError("Installed training schema/default weakened a constraint: " + key)
    if default.get("end", "missing") is not None or default.get("epochs") != 20 or default.get("training_batch_size") != 2048:
        raise RuntimeError("Installed factory defaults must retain nullable latest cutoff and registered training resources")
    if (status.get("mode") != "rule_observation_no_model" or status.get("models")
            or status.get("active_model") is not None or status.get("current_selection") is not None
            or status.get("certification", {}).get("publication_allowed") is not False):
        raise RuntimeError("Installed training status incorrectly qualifies a workstation model")
    return {"status": "passed", "model_family": "current", "strategy_version": "trend_consensus",
            "default_revision": profiles["default_revision"], "field_count": len(fields),
            "field_keys": sorted(by_key), "nullable_latest_end": True,
            "schema_constraint_floors": {key: by_key[key]["min"] for key in
                ("min_train_rows", "min_calibration_rows", "min_net_return", "min_r_multiple",
                 "liquidity_lookback_sessions", "liquidity_min_median_amount")},
            "training_job_submitted": False, "publication_allowed": False}


def verify(device: str, output: Path | None, *, expected_version: str | None = None, expected_wheel_sha256: str | None = None) -> dict:
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

    def reject_retired_command(name: str):
        completed = subprocess.run([sys.executable, "-I", "-X", "utf8", "-m", "my_strategy.cli", name, "--help"],
            cwd=state, env=environment, text=True, encoding="utf-8", capture_output=True, check=False, timeout=30)
        if completed.returncode != 2 or "invalid choice" not in completed.stderr:
            raise RuntimeError(f"Retired CLI command was not rejected: {name}")

    identity_script = """import importlib.metadata,json,sys
from pathlib import Path
import my_strategy
path=Path(my_strategy.__file__).resolve();prefix=Path(sys.prefix).resolve()
if not path.is_relative_to(prefix) or 'site-packages' not in path.parts:
    raise RuntimeError('Installation verification requires the application wheel from site-packages')
distribution=importlib.metadata.distribution('khquant')
direct=json.loads(distribution.read_text('direct_url.json') or '{}')
print(json.dumps({'version':distribution.version,'package_path':str(path),'prefix':str(prefix),
'application_import':'installed wheel site-packages','application_wheel_sha256':direct.get('archive_info',{}).get('hashes',{}).get('sha256')}))"""
    identity = command("-c", identity_script)
    if expected_version is not None and identity['version'] != expected_version:
        raise RuntimeError('Installed application version differs from the intended release')
    if expected_wheel_sha256 is not None and identity['application_wheel_sha256'] != expected_wheel_sha256:
        raise RuntimeError('Installed application wheel SHA differs from the intended release')
    doctor = command("-m", "my_strategy.cli", "doctor", "--check-write", "--strict", "--json")
    model = command(str(Path(__file__).resolve()), "--internal-probe", "--device", device)
    command("-m", "my_strategy.cli", "update-data", "--help", json_output=False)
    analysis = command("-m", "my_strategy.cli", "analyze", "--symbol", "000001.SZ", "--start", "2024-01-01", "--calendar-run-id", calendar_id, "--device", device, "--json")
    scan = command("-m", "my_strategy.cli", "scan", "--stocks", "000001.SZ", "--start", "2024-01-01", "--calendar-run-id", calendar_id, "--device", device, "--cpu-workers", "1", "--json")
    retired_commands = ("backtest", "backtest-batch")
    for name in retired_commands:
        reject_retired_command(name)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    process = None
    pages = ("analysis", "scan", "watchlist", "data", "training", "parameters")
    retired_routes = (("/backtest", "GET", None), ("/holdings", "GET", None),
        ("/api/backtest", "POST", {"symbol": "000001.SZ"}), ("/api/holdings", "GET", None),
        ("/api/holdings/000001.SZ", "PUT", {"shares": 100, "average_cost": 12}))
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
                for path, method, body in retired_routes:
                    expect_removed_route(base, path, method, body)
                if restart == 0:
                    trend_without_model = verify_no_model_trend(base, calendar_id, dates[-1])
                    training_center = verify_training_center(base)
                    request(base, "/api/watchlist", "POST", {"symbols": ["000001.SZ"]})
                else:
                    if "000001.SZ" not in json.dumps(request(base, "/api/watchlist")):
                        raise RuntimeError("Watchlist did not survive dashboard restart")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    report = {"status": "passed", "python": sys.executable, "state": str(state), "device": device,
              "application_identity": identity, "doctor": doctor, "model": model,
              "training_center": training_center, "dashboard_pages": list(pages), "watchlist_restart": True,
              "retired_routes_rejected": [f"{method} {path}" for path, method, _ in retired_routes],
              "retired_commands_rejected": list(retired_commands),
              "analysis_run": analysis.get("run_id"), "scan_run": scan.get("run_id"),
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
    parser.add_argument("--expected-version")
    parser.add_argument("--expected-wheel-sha256")
    parser.add_argument("--internal-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(json.dumps(model_probe(args.device) if args.internal_probe else verify(args.device, args.output, expected_version=args.expected_version, expected_wheel_sha256=args.expected_wheel_sha256), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
