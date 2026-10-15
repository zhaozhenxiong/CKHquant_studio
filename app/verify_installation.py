"""Exercise an installed distribution using isolated, explicitly synthetic inputs."""
from __future__ import annotations

import argparse
from contextlib import closing
import datetime as dt
from html.parser import HTMLParser
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

PRIMARY_PAGES = ("analysis", "scan", "watchlist", "simulation", "data")


def verify_primary_navigation(html: str) -> list[str]:
    """Inspect the shipped navigation, rather than accepting any HTML fallback."""
    class Navigation(HTMLParser):
        def __init__(self):
            super().__init__();self.in_primary=False;self.routes=[];self.ids=set()
        def handle_starttag(self, tag, attrs):
            attributes=dict(attrs)
            if attributes.get('id'):self.ids.add(attributes['id'])
            if tag=='nav' and attributes.get('aria-label')=='主要导航':self.in_primary=True
            if self.in_primary and tag=='a':
                route=attributes.get('data-route')
                if attributes.get('href')!='#'+str(route):raise RuntimeError('Installed primary navigation route mismatch')
                self.routes.append(route)
        def handle_endtag(self, tag):
            if tag=='nav':self.in_primary=False
    parsed=Navigation();parsed.feed(html)
    if tuple(parsed.routes)!=PRIMARY_PAGES or not {'view-simulation','light-predict-form','light-score-type','light-score-variant'}<=parsed.ids:
        raise RuntimeError('Installed primary navigation or Agent/MLP controls are incomplete')
    return parsed.routes


def verify_lightweight_api(base: str) -> dict:
    """Read-only fresh-install contracts; never download a model or submit work."""
    status=request(base,'/api/light/status')
    defaults=status.get('defaults') or {}
    if (status.get('version')!='agent_mlp_sim_v1' or status.get('score_model') is not None
            or status.get('score_candidates') or status.get('jobs')
            or status.get('cancelled_tasks_resumed') is not False
            or defaults.get('daily_mode')!='lightweight' or defaults.get('automatic_training') is not False):
        raise RuntimeError('Fresh installation has unexpected models, jobs or automatic training')
    assets=status.get('assets') or {}
    if assets.get('ready') is not False or not assets.get('missing'):
        raise RuntimeError('Fresh installation must explicitly report absent Kronos assets')
    expected={'/api/light/status':'get','/api/light/predict':'post','/api/light/simulate':'post',
        '/api/light/score/train':'post','/api/light/score/observations':'post','/api/light/daily':'post',
        '/api/boards/incremental/plan':'post','/api/boards/incremental/execute':'post'}
    routes=request(base,'/openapi.json').get('paths') or {}
    if any(method not in routes.get(path,{}) for path,method in expected.items()):
        raise RuntimeError('Installed lightweight/incremental API contract is incomplete')
    if request(base,'/api/light/accounts').get('accounts') or request(base,'/api/light/saved').get('results'):
        raise RuntimeError('Fresh installation contains personal research accounts')
    return {'status':'passed','version':status['version'],'kronos_assets':'absent_explicitly',
        'active_score_model':None,'automatic_training':False,'cancelled_tasks_resumed':False,
        'personal_accounts':0,'work_submitted':False,'routes_verified':sorted(expected)}


def verify_lightweight_probe(report: dict, requested_device: str) -> dict:
    if (report.get('status')!='passed' or report.get('family')!='agent_mlp_sim_v1'
            or report.get('fixture')!='synthetic_installation_only'
            or report.get('publication_allowed') is not False or report.get('market_effectiveness_certified') is not False):
        raise RuntimeError('Installed lightweight probe is incomplete or overstates qualification')
    model=report.get('mlp') or {}
    if ((requested_device!='auto' and model.get('training_device')!=requested_device)
            or model.get('training_device') not in {'cpu','mps','cuda'} or model.get('calibration_device')!='cpu'
            or model.get('inference_device')!='cpu' or model.get('save_reload_identical') is not True
            or not isinstance(report.get('incremental_source'),dict) or not isinstance(report.get('simulation_ledger'),dict)):
        raise RuntimeError('Installed lightweight probe device/serialization/component evidence is incomplete')
    return report


def model_probe(device: str) -> dict:
    """Exercise the current numeric model without creating a qualified registry entry."""
    current = reference_model_probe(device)
    return {"training_device": current["training_device"],
            "inference": {"device": current["inference_device"], "save_reload_identical": True},
            "prediction_rows": current["prediction_rows"],
            "fixture": "synthetic_installation_only", "reference_position_v1": current}


def reference_model_probe(device: str) -> dict:
    """Fit, calibrate and reload the shipped reference model on synthetic labels.

    This is an operator/serialization test. The fixture is deliberately not a
    source-qualified market population and never enters the active-model registry.
    """
    import numpy as np
    import pandas as pd
    from my_strategy.services import reference_model as model

    dates = pd.bdate_range("2023-01-02", periods=360)
    days = dates.strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(20261015)
    frame = pd.DataFrame(rng.normal(size=(len(days), len(model.FEATURE_COLUMNS))).astype(np.float32),
                         columns=model.FEATURE_COLUMNS)
    frame = frame.assign(symbol="INSTALLATION.TEST", date=days, label=np.arange(len(days)) % 3,
                         label_end=(dates + pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
                         label_available=True, input_eligible=True)
    frame.loc[::7, model.WYCKOFF_COLUMNS[0]] = np.nan
    frame.attrs["schema_hash"] = model.SCHEMA_HASH
    split = {"name": "synthetic_installation_only", "train_end": days[219],
             "calibration_start": days[220], "calibration_end": days[269],
             "policy_start": days[270], "policy_end": days[319],
             "test_start": days[320], "test_end": days[-1]}
    directory = Path(os.environ["KHQUANT_ARTIFACT_ROOT"]) / "installation-reference-model"
    fitted = model.fit_fold(frame, split, directory, device=device,
        config={"epochs": 2, "inner_epochs": 1, "batch_size": 128,
                "min_train_per_class": 5, "min_segment_per_class": 2})
    manifest, validation = fitted["manifest"], fitted["validation"]
    probe = frame.iloc[321:327].copy()
    probe.iloc[-1, probe.columns.get_loc("input_eligible")] = False
    probe.iloc[-2, probe.columns.get_loc(model.FEATURE_COLUMNS[0])] = np.nan
    first_session = model.load_model(directory, device=device)
    reloaded_session = model.load_model(directory, device=device)
    first, reloaded = first_session.predict(probe), reloaded_session.predict(probe)
    if (first.shape != (6, 3) or not np.isfinite(first[:4]).all()
            or not ((first[:4] >= 0) & (first[:4] <= 1)).all()
            or not np.allclose(first[:4].sum(axis=1), 1., atol=1e-6)
            or not np.isnan(first[4:]).all()
            or not np.array_equal(first, reloaded, equal_nan=True)):
        raise RuntimeError("Installed reference model serialization/probabilities/missing-input handling failed")
    actual = manifest["training_device"]
    if (str(first_session.device) != actual or str(reloaded_session.device) != actual
            or (device != "auto" and actual != device)
            or any(validation[key].get("device") != actual for key in ("calibration", "policy", "test"))):
        raise RuntimeError("Installed reference fit/calibration/inference did not use the requested device")
    if (validation.get("candidate_selection", {}).get("source") != "purged_train_internal_only"
            or validation.get("calibration_method") != "chronological_positive_temperature_no_intercepts"
            or not math.isfinite(manifest["temperature"]) or manifest["temperature"] <= 0
            or validation.get("independent_forward_evidence") is not False
            or manifest.get("historical_availability_claim") is not False):
        raise RuntimeError("Installed reference selection/calibration/time semantics changed")
    return {"family": model.FAMILY, "classes": manifest["classes"], "input_columns": len(model.FEATURE_COLUMNS),
            "architecture": manifest["architecture"], "training_device": actual,
            "calibration_device": validation["calibration"]["device"], "inference_device": str(first_session.device),
            "numeric_dtype": manifest["numeric_dtype"], "price_label_dtype": manifest["price_label_dtype"],
            "prediction_rows": 4, "unscored_rows": 2, "save_reload_identical": True,
            "missing_input_stays_unscored": True, "weights_sha256": manifest["weights_sha256"],
            "calibration_method": validation["calibration_method"], "temperature": manifest["temperature"],
            "selection_source": validation["candidate_selection"]["source"],
            "fixture": "synthetic_installation_labels_only", "account_simulated": False,
            "active_model_registered": False, "market_effectiveness_certified": False,
            "publication_allowed": False}


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
    # Close after committing so Windows can remove the isolated fixture directory.
    with closing(sqlite3.connect(path)) as connection, connection:
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


def _assert_untrained_status(status: dict) -> None:
    if (status.get("model_family") != "current" or status.get("strategy_version") != "reference_position_v1"
            or status.get("current_selection") is not None or status.get("active_model_run_id") is not None
            or status.get("active_model") is not None or status.get("models") or status.get("selection_events")
            or status.get("certification", {}).get("publication_allowed") is not False):
        raise RuntimeError("Public defaults must select reference_position_v1 without a workstation model")


def _assert_analysis_preview(result: dict) -> str:
    identity = result.get('preview_id')
    if (result.get('preview') is not True or result.get('saved') is not False
            or 'run_id' in result or 'run_context' in result
            or not isinstance(identity, str) or len(identity) != 64
            or any(char not in '0123456789abcdef' for char in identity)):
        raise RuntimeError('Unsaved analysis must have a preview identity and no permanent run')
    return identity


def verify_analysis_preview_cache(base: str, body: dict) -> dict:
    """Check the source-only UI path's actual same-input cache round trip."""
    body = {**body, 'research': False, 'save': False}
    first = request(base, '/api/analysis', 'POST', body)
    preview_id = _assert_analysis_preview(first)
    capabilities = first.get('capabilities') or {}
    if (first.get('projection_version') != 'structure_sources_v1'
            or capabilities != {'source_evidence': True, 'direction_model': False,
                                'reference_position': False, 'simulated_account': False}
            or first.get('reference_points') != [] or first.get('direction_points') != []
            or 'position_reference' in first
            or any(first.get('request', {}).get(key) != value for key, value in body.items())):
        raise RuntimeError('Installed source preview changed its request or source-only presentation contract')
    second = request(base, '/api/analysis', 'POST', body)
    if (_assert_analysis_preview(second) != preview_id or second.get('reused') is not True
            or {k:v for k,v in first.items() if k != 'reused'} !=
               {k:v for k,v in second.items() if k != 'reused'}):
        raise RuntimeError('Identical analysis inputs did not reuse the same immutable preview')
    return {'status': 'passed', 'preview_id': preview_id, 'same_request_cache_reused': True,
            'cached_payload_identical': True, 'source_only': True, 'saved': False, 'run_id': None}


def verify_no_model_reference(base: str, calendar_id: str, end: str) -> dict:
    _assert_untrained_status(request(base, "/api/training/status"))
    run_path = '/api/runs?kind=analysis&limit=200'
    runs_before = request(base, run_path)
    if not isinstance(runs_before.get('runs'), list):
        raise RuntimeError('Installation preview probe requires the permanent analysis run index')
    body = {"symbol": "000001.SZ", "start": "2024-01-01", "end": end,
        "model_family": "current", "calendar_run_id": calendar_id, "device": "cpu"}
    result = request(base, "/api/analysis", "POST", body)
    rows = result.get("chart_evidence", {}).get("rows", [])
    if (not rows or result.get("strategy_version") != "reference_position_v1"
            or result.get("research", {}).get("model_run_id") is not None
            or result.get("publication_allowed") is not False):
        raise RuntimeError("Untrained installation failed its actual default reference analysis")
    if (result.get("analysis_mode") != "reference" or result.get("account_scope") != "none"
            or result.get("account_actions") is not None or result.get("research", {}).get("account_simulated") is not False
            or result.get("chart_evidence", {}).get("statistics", {}).get("visible_window", {}).get("broker_fills") != 0):
        raise RuntimeError("Installed ordinary analysis unexpectedly simulated an account")
    source_points, final_points = result.get("source_reference_points"), result.get("reference_points")
    if not isinstance(source_points, list) or not isinstance(final_points, list):
        raise RuntimeError("Installed analysis omitted independent source/target B/S collections")
    for point in source_points:
        if (point.get("side") not in {"BUY", "SELL"} or point.get("reference_price_source") != "signal_day_close"
                or point.get("reference_price_date") != point.get("date")):
            raise RuntimeError("Installed source B/S reference changed its direction or price clock")
    # A fresh installation starts at zero. Missing scores preserve this reference
    # state; source SELL points cannot manufacture a target reduction or a fill.
    if final_points:
        raise RuntimeError("Untrained installation fabricated target-change points")
    for row in rows:
        agents = row.get("agents", {})
        agent = agents.get("ma", {})
        if (agent.get("status") != "model_unavailable" or agent.get("in_scope") is not False
                or any(agent.get(key) is not None for key in ("p_up", "p_down", "p_neutral"))):
            raise RuntimeError("Untrained installation fabricated reference direction probabilities")
        if (any(agents.get(head, {}).get("probability") is not None for head in ("high_return", "reference_quality"))
                or row.get("high_return_probability") is not None or row.get("exit_probability") is not None):
            raise RuntimeError("Untrained installation reintroduced retired target probabilities")
        position = row.get("position_reference") or {}
        if (position.get("schema") != "reference_position_v1" or position.get("account_scope") != "none"
                or position.get("reference_only") is not True or position.get("stale") is not True
                or position.get("predictive_gate_passed") is not False
                or position.get("action") != "WAIT" or position.get("target_position_fraction") != 0.
                or position.get("previous_reference_target") != 0. or position.get("change_fraction") != 0.
                or position.get("reference_delta") != 0.
                or position.get("change_basis") != "previous_reference_target_for_same_symbol"
                or position.get("risk_budget_fraction") != .01
                or row.get("action") != "WAIT" or row.get("emitted") is not False):
            raise RuntimeError("Untrained installation changed the stale zero reference path")
    preview_id = _assert_analysis_preview(result)
    source_preview = verify_analysis_preview_cache(base, body)
    if request(base, run_path) != runs_before:
        raise RuntimeError('Unsaved analysis created or changed a permanent analysis run')
    return {"status": "passed", "strategy": result["strategy_version"], "model_run_id": None,
        "probabilities": None, "analysis_rows": len(rows), "analysis_mode": "reference",
        "account_scope": "none", "account_simulated": False,
        "source_reference_points": len(source_points), "final_reference_points": len(final_points),
        "source_reference_sides": sorted({point["side"] for point in source_points}),
        "target_position_fraction": 0., "change_fraction": 0., "stale": True,
        "unqualified_allocation_blocked": True, "run_id": None, "preview": True, "saved": False,
        "preview_id": preview_id, "source_preview_cache": source_preview,
        "permanent_analysis_runs_unchanged": True, "publication_allowed": False,
        "fixture": "synthetic_unverified_prices_and_synthetic_schedule_only"}


def verify_training_center(base: str) -> dict:
    """Read-only current parameter checks; never submit synthetic training as a job."""
    capabilities = request(base, "/api/training/capabilities")
    profiles = request(base, "/api/training/profiles")
    _assert_untrained_status(request(base, "/api/training/status"))
    if capabilities.get("model_family") != "current" or capabilities.get("strategy_version") != "reference_position_v1":
        raise RuntimeError("Installed training capabilities do not use the current reference strategy")
    fields = capabilities.get("schema", {}).get("fields")
    if not isinstance(fields, list) or not fields or len({field["key"] for field in fields}) != len(fields):
        raise RuntimeError("Installed default parameter schema is incomplete")
    by_key = {field["key"]: field for field in fields}
    expected = {"device": "auto", "feature_start": "2023-01-01", "end": None,
        "source_native_run_id": None, "horizon_sessions": 10, "epochs": 12, "batch_size": 8192,
        "cpu_workers": 4, "seed": 42, "max_stock_weight": .20, "risk_budget": .01,
        "position_step": .025, "hysteresis": .01}
    if (capabilities.get("defaults") != expected or set(by_key) != set(expected)
            or any(by_key[key].get("default") != value for key, value in expected.items())
            or profiles.get("default_profile", {}).get("parameters") != expected):
        raise RuntimeError("Installed reference defaults or registered parameter schema changed")
    if (set(by_key["device"].get("options", [])) != {"auto", "cuda", "mps", "cpu"}
            or set(by_key["horizon_sessions"].get("options", [])) != {10, 20}
            or capabilities.get("actual_auto_device") not in {"cuda", "mps", "cpu"}
            or capabilities.get("qualification", {}).get("publication_allowed") is not False
            or {stage["key"] for stage in capabilities.get("stages", []) if stage.get("trainable")} != {"reference_model"}):
        raise RuntimeError("Installed reference device/stage/qualification contract changed")
    return {"status": "passed", "model_family": "current", "strategy_version": "reference_position_v1",
            "default_revision": profiles["default_revision"], "field_count": len(fields),
            "field_keys": sorted(by_key), "nullable_latest_end": True,
            "actual_auto_device": capabilities["actual_auto_device"], "defaults": expected,
            "training_job_submitted": False, "publication_allowed": False}


def verify(device: str, output: Path | None, *, expected_version: str | None = None, expected_wheel_sha256: str | None = None) -> dict:
    state = Path(tempfile.mkdtemp(prefix="khquant-install-smoke-")).resolve()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("KHQUANT_") and key not in {"PYTHONPATH", "PYTHONHOME"}}
    environment.update(KHQUANT_HOME=str(state), KHQUANT_DATA_ROOT=str(state / "data"),
                       KHQUANT_ARTIFACT_ROOT=str(state / "artifacts"), KHQUANT_METADATA_ROOT=str(state / "data/metadata"),
                       KHQUANT_RAW_DB=str(state / "data/raw/khquant_raw.db"), KHQUANT_LOG_ROOT=str(state / "data/logs"),
                       KHQUANT_KRONOS_ROOT=str(state / "models/Kronos-base"),
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
    lightweight = verify_lightweight_probe(command(str(Path(__file__).with_name('verify_agent_installation.py').resolve()),
        '--internal-probe','--device',device),device)
    command("-m", "my_strategy.cli", "update-data", "--help", json_output=False)
    command("-m", "my_strategy.cli", "train-reference", "--help", json_output=False)
    command("-m", "my_strategy.cli", "light-research", "--help", json_output=False)
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
    pages = PRIMARY_PAGES
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
                navigation = verify_primary_navigation(request(base,'/'))
                lightweight_api = verify_lightweight_api(base)
                for path, method, body in retired_routes:
                    expect_removed_route(base, path, method, body)
                if restart == 0:
                    reference_without_model = verify_no_model_reference(base, calendar_id, dates[-1])
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
              "lightweight_v1": lightweight, "lightweight_api":lightweight_api,
              "primary_navigation":navigation, "lightweight_entrypoint":True,
              "training_center": training_center, "dashboard_pages": list(pages), "watchlist_restart": True,
              "retired_routes_rejected": [f"{method} {path}" for path, method, _ in retired_routes],
              "retired_commands_rejected": list(retired_commands),
              "analysis_run": analysis.get("run_id"), "scan_run": scan.get("run_id"),
              "update_entrypoint": True, "reference_training_entrypoint": True,
              "default_reference_without_model": reference_without_model,
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
