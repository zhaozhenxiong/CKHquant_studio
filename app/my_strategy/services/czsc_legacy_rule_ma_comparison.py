"""Separate actual-ledger controls for the old composite rule and ten-day MA.

The offline MA arm is reconstructed research, never an actual historical
release. Its checkpoint router and predictor retain mature-date checks. This
module is deliberately outside the high-return frozen execution dependency set.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import argparse
import json
import multiprocessing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT, PROCESSED_DATA_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_research_models import file_hash, identity_path, model_hash

VERSION = "czsc_legacy_composite_fixed10_ma_offline_controls_v1"
KIND = "czsc-legacy-rule-ma-comparison"
DEFAULT_MA_RUN = "20261009-112943-155001-20261008-czsc-research-train-NO_GIT"
ARMS = {"reference": "legacy_actual_composite_rule",
    "full": "legacy_fixed10_ma_offline_chronological_shadow"}


def _read(path):
    return json.loads(Path(path).read_text())


def _save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2,
        default=lambda v: v.item() if isinstance(v,np.generic) else str(v)) + "\n")


class _FrozenReleases:
    """A captured, hash-verified release view with no writable store constructor."""
    def __init__(self, events, runs_root=ARTIFACT_RUNS_ROOT):
        self.values = events; self.runs_root = Path(runs_root)

    def events(self, limit=None):
        return self.values if limit is None else self.values[-limit:]

    def verify_release(self, release_id):
        event = next((r for r in self.values if r["event_id"] == release_id
            and r["event_type"] == "promote"), None)
        if event is None:
            raise ValueError("unknown captured actual release")
        for binding in event["artifacts"]:
            path = (self.runs_root / binding["path"]).resolve()
            if not path.is_relative_to(self.runs_root.resolve()) or file_hash(path) != binding["sha256"]:
                raise ValueError("captured release artifact changed")
        return event


def _release_events():
    path = PROCESSED_DATA_ROOT / "czsc/results.db"
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='czsc_model_release_events'").fetchone():
            return []
        rows = conn.execute("SELECT payload,payload_sha256 FROM czsc_model_release_events ORDER BY sequence").fetchall()
    values = []
    for raw, sha in rows:
        value = json.loads(raw)
        if model_hash(value) != sha:
            raise ValueError("actual release snapshot integrity mismatch")
        values.append(value)
    return values


def _resolver(legacy_run_id, calendar, releases):
    from my_strategy.services.czsc_research_models import ModelResolver
    from my_strategy.services.czsc_research_profiles import get_feature_profile
    profile = get_feature_profile("ma_trend_v1")
    return ModelResolver(usage_mode="historical", model_policy="auto", model_run_id=legacy_run_id,
        store=_FrozenReleases(releases), runs_root=ARTIFACT_RUNS_ROOT,
        runtime_calendar_dates=calendar, feature_version=profile["version"],
        feature_schema_hash=profile["schema_hash"], feature_columns=profile["columns"],
        strategy_version=profile["strategy_version"])


def _model_bindings(resolver):
    if any(item["manifest"]["label_contract"].get("horizon") != 10
            or item["manifest"]["label_contract"].get("initial_cash") != 100000
            or item["feature_profile"] != "ma_trend_v1" for item in resolver.catalog):
        raise ValueError("legacy control requires the original fixed-ten-session MA target")
    return [{**{k:v for k,v in item.items() if k != "manifest"},
        "manifest_file_sha256": file_hash(Path(item["model_dir"]) / "manifest.json"),
        "target_contract": item["manifest"]["label_contract"]} for item in resolver.catalog]


def _code_hashes(anchor):
    initial = _read(anchor / "config/initial_code_hashes.json")
    extra = [Path(__file__), *[MY_STRATEGY_ROOT / "services" / name for name in (
        "czsc_research_models.py", "czsc_research_decisions.py", "czsc_research_runtime.py",
        "czsc_entry_plan.py", "czsc_dual_research.py")],
        MY_STRATEGY_ROOT / "configs/czsc_entry_plan.json"]
    current = {str(Path(path)): file_hash(Path(path)) for path in initial}
    if current != initial:
        raise ValueError("anchor frozen execution code changed")
    current.update({str(path): file_hash(path) for path in extra})
    return current


def _chronological_probabilities(features, resolver, predictor):
    """Data-mature routes and actual-release coverage are separate facts."""
    from my_strategy.services.czsc_research_profiles import bind_feature_profile
    frame = bind_feature_profile(features, "ma_trend_v1")
    frame["input_eligible"] = frame["model_input_eligible"].fillna(False).astype(bool)
    routes = resolver.resolve_dates(frame.date.tolist())
    probabilities = np.full(len(frame), np.nan); selected = {}
    for i, (route, eligible) in enumerate(zip(routes, frame.input_eligible, strict=True)):
        if route.get("model_dir") and eligible:
            selected.setdefault(route["model_dir"], []).append(i)
    for directory, indices in selected.items():
        # Strict PredictorSession.predict rejects any row before head availability.
        values = predictor.predict(frame.iloc[indices], directory, as_of=frame.date.iloc[-1])
        probabilities[indices] = values
    return probabilities, routes


def _replay_legacy(features, calendar, config, start, end, *, probabilities=None, routes=None, offline=False):
    from my_strategy.services.czsc_dual_research import _raw
    from my_strategy.services.czsc_research_decisions import replay_decisions
    from my_strategy.services.czsc_backtest import execute_decisions
    from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
    from my_strategy.services.czsc_trade_episodes import _settings
    selected = features.date.le(end).to_numpy(); frame = features.loc[selected].copy()
    raw = _raw(frame)
    inside = frame.date.ge(start).to_numpy(); offset = int(np.flatnonzero(inside)[0])
    values = np.asarray(probabilities)[selected][inside] if probabilities is not None else None
    fallback=[{"date":day,"status":"rules_no_model","model_dir":None,"applied_to_entry":False,
        "probability_threshold":.55} for day in features.date]
    metadata = [route for route, keep in zip(routes if routes is not None else fallback, selected, strict=True) if keep][offset:]
    # The legacy API ignores position_start. Replay only this account's live
    # interval so its Position begins flat; retain causal input and Broker warmup.
    applied = np.array([bool(route.get("model_dir")) and bool(frame.iloc[offset+i].model_input_eligible)
        and np.isfinite(values[i]) for i, route in enumerate(metadata)]) if offline else np.zeros(len(metadata), bool)
    replay = replay_decisions(frame.iloc[offset:].reset_index(drop=True), raw.iloc[offset:].reset_index(drop=True),
        probabilities=values, thresholds=[r.get("probability_threshold", .55) for r in metadata],
        apply_ml_mask=applied, model_metadata=metadata, mode="rules", entry_policy="legacy", config=_settings(config))
    prefix = [{"date": day, "target_weight": 0., "eligible": False} for day in frame.date.iloc[:offset]]
    decisions = prefix + replay["decisions"]
    account = execute_decisions(str(frame.symbol.iloc[0]), raw, decisions, start, 100000., _settings(config),
        VERSION, market_dates=calendar, verified_sources=VERIFIED_SOURCES, entry_policy="legacy")
    candidates = frame.iloc[offset:].input_eligible.fillna(False).astype(bool) & frame.iloc[offset:].rule_buy.fillna(False).astype(bool)
    account["model_coverage"] = {"total_candidates": int(candidates.sum()),
        "selected_candidates": sum(r["action"] == "BUY" for r in account["ledger"]),
        "decidable_candidates": int(candidates.sum()), "model_scored_candidates": int((candidates.to_numpy() & applied).sum()),
        "actual_holding_states": sum(r.get("shares", 0) > 0 for r in account["daily"]), "model_scored_holding_states": 0}
    coverage = {"sessions": len(metadata), "data_mature_checkpoint_sessions": sum(bool(r.get("model_dir")) for r in metadata),
        "model_input_and_probability_sessions": int(applied.sum()) if offline else 0,
        "actual_release_eligible_sessions": sum(bool(r.get("applied_to_entry")) for r in metadata),
        "actual_release_applied_sessions": sum(bool(r.get("applied_to_entry")) and bool(used)
            for r,used in zip(metadata,applied,strict=True)),
        "offline_filter_candidate_sessions": int((candidates.to_numpy() & applied).sum()) if offline else 0,
        "rules_fallback_candidate_sessions": int((candidates.to_numpy() & ~applied).sum()),
        "fallback_rule": "unavailable_or_input_ineligible_old_checkpoint_preserves_composite_rule",
        "actual_production_release_claim": False}
    from my_strategy.services.czsc_research_runtime import route_segments
    columns=("date","eligible","rule_buy","rule_sell","probability","threshold","ml_filter_applied",
        "target_weight","position_intent")
    return account, {"coverage": coverage, "routes": route_segments(metadata),
        "decisions": [{k:row[k] for k in columns} for row in replay["decisions"]]}


def _worker(payload):
    from my_strategy.services.czsc_high_return_research import _read_features
    from my_strategy.services.czsc_high_return_evaluation import (_window_market_reason, _cash_account,
        _write_account, CASH_NOT_OBSERVED, SOURCE_AFTER_WINDOW)
    from my_strategy.services.czsc_research_ml import PredictorSession
    from my_strategy.services.czsc_trade_episodes import _settings, _cost, _episodes
    record, calendar, config, windows, legacy_run_id, releases, expected_models, output = payload
    symbol = record["symbol"]; result = []; failure = None; features = None; model_failure = None
    values, routes = None, None
    try:
        # Retain the verified first source date even when it follows this window.
        features = _read_features(record, config, calendar[-1])
        if not set(features.date) <= set(calendar):
            raise ValueError("comparison source dates outside verified calendar")
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        failure = str(exc)
    if failure is None:
      try:
        resolver = _resolver(legacy_run_id, calendar, releases)
        routes = resolver.resolve_dates(features.date.tolist())
        if _model_bindings(resolver) != expected_models:
            raise ValueError("comparison old-model bindings changed during replay")
        visible = features.loc[features.date <= max(w["end"] for w in windows)].copy()
        scored, known_routes = _chronological_probabilities(visible, resolver, PredictorSession(device="cpu"))
        values = np.full(len(features), np.nan); values[:len(visible)] = scored
        routes = known_routes + [{} for _ in range(len(features)-len(visible))]
      except (ValueError, RuntimeError, KeyError, OSError) as exc:
        model_failure = str(exc)
    settings = _settings(config); cost = _cost(settings); market_index = {d:i for i,d in enumerate(calendar)}
    for window in windows:
        window_reason = failure or _window_market_reason(features, window)
        for arm in ARMS:
            reason = window_reason or (model_failure if arm == "full" else None)
            diagnostic = {"coverage": {}, "routes": [], "decisions": []}
            if reason:
                status = CASH_NOT_OBSERVED if failure is None and reason == SOURCE_AFTER_WINDOW else "cash_retained_failure"
                account = _cash_account(symbol, window["dates"], 100000.)
            else:
                try:
                    account, diagnostic = _replay_legacy(features, calendar, config, window["start"], window["end"],
                        probabilities=values, routes=routes, offline=arm == "full")
                    status = "complete"
                except (ValueError, RuntimeError, KeyError, OSError) as exc:
                    account = _cash_account(symbol, window["dates"], 100000.)
                    status, reason = "cash_retained_failure", str(exc)
            episodes = _episodes(account, source=ARMS[arm], symbol=symbol, cost=cost, config=config,
                market_index=market_index, policy_hash=stable_hash({"version":VERSION,"arm":arm,"settings":settings}))
            destination = Path(output) / window["name"] / arm / symbol.replace(".", "_")
            binding = _write_account(destination, account, episodes, status=status, reason=reason,
                source_first_date=features.date.iloc[0] if status == CASH_NOT_OBSERVED else None)
            _save(destination / "legacy_routes.json", {k:v for k,v in diagnostic.items() if k!="decisions"})
            pd.DataFrame(diagnostic["decisions"],columns=["date","eligible","rule_buy","rule_sell","probability",
                "threshold","ml_filter_applied","target_weight","position_intent"]).to_parquet(destination/"legacy_decisions.parquet",index=False)
            result.append({"symbol":symbol,"window":window["name"],"arm":arm,**binding,
                "legacy_routes_sha256":file_hash(destination/"legacy_routes.json"),
                "legacy_decisions_sha256":file_hash(destination/"legacy_decisions.parquet")})
    return result


def run_legacy_comparison(anchor_run_id, *, legacy_model_run_id=DEFAULT_MA_RUN, symbols=None,
        windows=None, cpu_workers=4, progress=None, check_cancel=None):
    """Write a separate diagnostic RunContext; no training artifact is changed."""
    from my_strategy.services.czsc_high_return_evaluation import _window, _mainboard, _aggregate_bindings, _worker_init
    from my_strategy.services.czsc_trade_episodes import _settings
    from my_strategy.storage.czsc_results import ResultStore
    anchor = identity_path(ARTIFACT_RUNS_ROOT, anchor_run_id)
    records = _read(anchor/"data_snapshots/source_records.json"); snapshot = _read(anchor/"data_snapshots/source.json")
    calendar = _read(anchor/"data_snapshots/calendar.json")
    config = _read(anchor/"config/research.json")
    from my_strategy.services.czsc_high_return_research import _source
    original_records, original_dates, original_snapshot, _ = _source(snapshot["run_id"],
        snapshot["effective_data_end"], [r["symbol"] for r in records])
    if records != original_records or snapshot != original_snapshot or calendar.get("dates") != original_dates:
        raise ValueError("comparison anchor does not match its actual frozen source references")
    if calendar.get("verified") is not True or calendar["dates"] != sorted(set(calendar["dates"])):
        raise ValueError("comparison requires the anchor verified calendar")
    dates = calendar["dates"]; records = [r for r in records if _mainboard(r["symbol"])]
    requested = set(symbols or [r["symbol"] for r in records])
    if requested - {r["symbol"] for r in records} or len(requested) != len(symbols or requested):
        raise ValueError("comparison symbols must be unique anchor mainboard inputs")
    selected = [r for r in records if r["symbol"] in requested]
    if not selected or cpu_workers < 1 or int(cpu_workers) != cpu_workers or config["initial_cash"] != 100000.:
        raise ValueError("comparison requires nonempty mainboard inputs and fixed 100000 cash")
    end = snapshot["effective_data_end"]
    values = windows if windows is not None else [*config["development_windows"],
        {"name":"continuous_2023","start":config["feature_start"],"end":end,"continuous_no_checkpoint_account_reset":True}]
    plans = [_window(w, dates) for w in values]
    if len({w["name"] for w in plans}) != len(plans) or any(w["end"] > end for w in plans):
        raise ValueError("comparison window duplicates or extends frozen anchor source")
    releases = _release_events(); resolver = _resolver(legacy_model_run_id, dates, releases); models = _model_bindings(resolver)
    code = _code_hashes(anchor)
    legacy_root = identity_path(ARTIFACT_RUNS_ROOT, legacy_model_run_id)
    old_report = legacy_root/"reports/research.json"
    identity = {"version":VERSION,"anchor_run_id":anchor_run_id,"anchor_source_snapshot_hash":snapshot["snapshot_hash"],
        "records":selected,"calendar_hash":stable_hash(dates),"config":config,"windows":plans,"models":models,
        "legacy_report_sha256":file_hash(old_report),"actual_release_events_hash":stable_hash(releases),"code_hashes":code}
    context = create_run_context(task=KIND, as_of_date=end, config=identity, data_version=snapshot["snapshot_hash"],
        scope="full", stocks=[r["symbol"] for r in selected], start_date=min(w["start"] for w in plans), end_date=end)
    output = context.subdir("backtests","legacy-controls")
    for name,value in [("source_records",selected),("source",snapshot),("calendar",calendar),("models",models),
            ("actual_release_events",releases),("code_hashes",code),("identity",identity)]:
        _save(context.subdir("data_snapshots")/(name+".json"),value)
    payloads = [(r,dates,config,plans,legacy_model_run_id,releases,models,str(output)) for r in selected]
    bindings = {(w["name"],arm):[] for w in plans for arm in ARMS}; check=check_cancel or (lambda:None)
    def consume(rows, index):
        check()
        for row in rows: bindings[(row["window"],row["arm"])].append(row)
        if progress: progress("legacy_rule_ma_actual_controls",index+1,len(selected),sum(r["status"]=="cash_retained_failure" for r in rows))
    if cpu_workers == 1:
        for i,payload in enumerate(payloads):check();consume(_worker(payload),i)
    else:
        with ProcessPoolExecutor(max_workers=min(int(cpu_workers),len(selected)),
                mp_context=multiprocessing.get_context("spawn"),initializer=_worker_init) as pool:
            for i,rows in enumerate(pool.map(_worker,payloads,chunksize=1)):consume(rows,i)
    reports={}
    for window in plans:
        variants={arm:_aggregate_bindings(bindings[(window["name"],arm)],window["dates"],100000.) for arm in ARMS}
        for arm,variant in variants.items():
            coverage=[_read(Path(b["path"])/"legacy_routes.json")["coverage"] for b in bindings[(window["name"],arm)]]
            variant["legacy_model_coverage"]={key:sum(c.get(key,0) for c in coverage) for key in (
                "sessions","data_mature_checkpoint_sessions","model_input_and_probability_sessions",
                "actual_release_eligible_sessions","actual_release_applied_sessions",
                "offline_filter_candidate_sessions","rules_fallback_candidate_sessions")}
        reports[window["name"]]={"fold":{k:v for k,v in window.items() if k!="dates"},"variants":variants,
            "ledger_bindings":{arm:bindings[(window["name"],arm)] for arm in ARMS},"initial_cash":100000.*len(selected),
            "qualification":"observed_historical_development_diagnostic","model_gate":False,"publication_allowed":False}
    if any(file_hash(Path(path))!=sha for path,sha in code.items()):raise ValueError("comparison code drifted during execution")
    evaluation={"version":VERSION,"evaluation_identity":stable_hash(identity),"output_root":str(output.resolve()),
        "model_bundles_hash":stable_hash(models),"source_records_hash":stable_hash(selected),"source_snapshot_hash":snapshot["snapshot_hash"],
        "calendar_hash":stable_hash(dates),"strategy_configuration_hash":stable_hash(_settings(config)),
        "base_execution_config_hash":stable_hash(_settings(config)["execution"]),"windows":reports,
        "cost_multiplier":1.,"thresholds_refit":False,"publication_allowed":False,"model_gate":False}
    _save(output/"evaluation.json",evaluation)
    report={"version":VERSION,"run_id":context.run_id,"anchor_run_id":anchor_run_id,"legacy_model_run_id":legacy_model_run_id,
        "arm_identities":ARMS,"config":config,"dataset_records":selected,"calendar":calendar,"source_snapshot":snapshot,
        "legacy_models":models,"legacy_catalog_errors":resolver.catalog_errors,"evaluation":evaluation,
        "evaluation_sha256":file_hash(output/"evaluation.json"),"anchor_mainboard_count":len(records),
        "requested_mainboard_count":len(selected),"requested_initial_cash":len(selected)*100000.,
        "complete_anchor_mainboard_universe":len(selected)==len(records),"created_at":local_now().isoformat(),
        "qualification":"offline_development_controls_not_actual_historical_model_release",
        "legacy_model_target":"czsc_fixed_horizon_broker_v1_horizon10", "high_return_target_claim":False,
        "legacy_risk_qualified_high_outcome":"unavailable_without_actual_entry_stop_identity",
        "position_initialization":"fresh_flat_each_development_window_continuous_once_2023",
        "model_gate":False,"publication_allowed":False,"alias_updated":False}
    _save(context.subdir("reports")/"legacy_comparison.json",report)
    context.write_metadata({"status":"completed","actual_completed_at":local_now().isoformat()})
    ResultStore().save(KIND,report)
    return report


def _same_rows(actual, expected, columns):
    actual=pd.DataFrame(actual);expected=pd.DataFrame(expected)
    if len(actual)!=len(expected):return False
    if actual.empty:return True
    for key in columns:
        if key not in actual or key not in expected:return False
        left,right=actual[key],expected[key]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(left.to_numpy(float),right.to_numpy(float),rtol=1e-8,atol=1e-7,equal_nan=True):return False
        elif left.fillna("").astype(str).tolist()!=right.fillna("").astype(str).tolist():return False
    return True


def _audit_policy_replay(report, resolver, state):
    from my_strategy.services.czsc_high_return_research import _read_features
    from my_strategy.services.czsc_research_ml import PredictorSession
    predictor=PredictorSession(device="cpu");calendar=report["calendar"]["dates"]
    for record in report["dataset_records"]:
        features=_read_features(record,report["config"],max(w["fold"]["end"] for w in report["evaluation"]["windows"].values()))
        values,routes=_chronological_probabilities(features,resolver,predictor)
        for window in report["evaluation"]["windows"].values():
            for arm,bindings in window["ledger_bindings"].items():
                binding=next(b for b in bindings if b["symbol"]==record["symbol"])
                if binding["status"]!="complete":continue
                account,diagnostic=_replay_legacy(features,calendar,report["config"],window["fold"]["start"],window["fold"]["end"],
                    probabilities=values,routes=routes,offline=arm=="full")
                folder=Path(binding["path"])
                state.check("legacy_actual_policy_broker_reexecution",_same_rows(pd.read_csv(folder/"ledger.csv"),account["ledger"],
                    ("date","symbol","action","shares","price","fee","cash_flow","cash_after","position_before","position_after","signal_date"))
                    and _same_rows(pd.read_csv(folder/"daily.csv"),account["daily"],("date","cash","equity","shares","market_value")),str(folder))
                state.check("legacy_actual_probability_and_policy_reexecution",
                    _same_rows(pd.read_parquet(folder/"legacy_decisions.parquet"),diagnostic["decisions"],
                        ("date","eligible","rule_buy","rule_sell","probability","threshold","ml_filter_applied","target_weight","position_intent"))
                    and _read(folder/"legacy_routes.json")["coverage"]==diagnostic["coverage"],str(folder))


def _audit_anchor_bindings(report, identity, state):
    """Re-read the original frozen request; a supplement cannot declare its pool."""
    from my_strategy.services.czsc_high_return_research import _source
    from my_strategy.services.czsc_high_return_evaluation import _mainboard, _window
    anchor=identity_path(ARTIFACT_RUNS_ROOT,report["anchor_run_id"])
    records=_read(anchor/"data_snapshots/source_records.json")
    snapshot=_read(anchor/"data_snapshots/source.json")
    calendar=_read(anchor/"data_snapshots/calendar.json")
    config=_read(anchor/"config/research.json")
    original,dates,source,_=_source(snapshot["run_id"],snapshot["effective_data_end"],
        [r["symbol"] for r in records])
    state.check("legacy_original_frozen_source_references",records==original and snapshot==source
        and calendar.get("dates")==dates and calendar.get("verified") is True)
    mainboard=[r for r in records if _mainboard(r["symbol"])]
    symbols=[r["symbol"] for r in report["dataset_records"]]
    expected=[r for r in mainboard if r["symbol"] in set(symbols)]
    state.check("legacy_exact_anchor_input_subset",len(symbols)==len(set(symbols)) and bool(symbols)
        and report["dataset_records"]==expected and report["anchor_mainboard_count"]==len(mainboard)
        and report["requested_mainboard_count"]==len(expected)
        and report["complete_anchor_mainboard_universe"]==(len(expected)==len(mainboard)))
    state.check("legacy_anchor_config_and_calendar",report["config"]==config==identity["config"]
        and report["calendar"]==calendar and report["source_snapshot"]==snapshot
        and identity["anchor_run_id"]==report["anchor_run_id"])
    state.check("legacy_anchor_frozen_execution_code",_code_hashes(anchor)==identity["code_hashes"])
    plans=identity["windows"]
    state.check("legacy_original_requested_window_plan",len(plans)==len(report["evaluation"]["windows"])
        and all(_window(plan,dates)==plan for plan in plans)
        and all(plan["end"]<=snapshot["effective_data_end"] and
            {k:v for k,v in plan.items() if k!="dates"}==report["evaluation"]["windows"][plan["name"]]["fold"]
            for plan in plans))
    state.check("legacy_evaluation_request_hash",report["evaluation"]["evaluation_identity"]==stable_hash(identity))


def audit_legacy_comparison(run_id, *, as_of=None, replay_policy=True):
    """Independently reconcile stored source/model identities and Broker cash."""
    from my_strategy.services.czsc_high_return_audit import _Audit, _audit_evaluation, signal_time
    root=identity_path(ARTIFACT_RUNS_ROOT,run_id); report=_read(root/"reports/legacy_comparison.json")
    state=_Audit();identity=_read(root/"data_snapshots/identity.json")
    state.check("legacy_comparison_identity",report["run_id"]==run_id and report["arm_identities"]==ARMS
        and report["publication_allowed"] is False and report["high_return_target_claim"] is False)
    metadata=_read(root/"metadata.json")
    state.check("legacy_actual_completed_run",metadata.get("status")=="completed"
        and signal_time(metadata["actual_completed_at"])<=(signal_time(as_of) if as_of else local_now()))
    state.attempt("legacy_original_anchor_audit",lambda:_audit_anchor_bindings(report,identity,state))
    state.check("legacy_original_report_sha256",file_hash(identity_path(ARTIFACT_RUNS_ROOT,report["legacy_model_run_id"])/"reports/research.json")==identity["legacy_report_sha256"])
    for path,sha in identity["code_hashes"].items():state.check("legacy_comparison_code_sha256",file_hash(Path(path))==sha,path)
    releases=_read(root/"data_snapshots/actual_release_events.json")
    state.check("legacy_actual_release_registry_prefix",_release_events()[:len(releases)]==releases)
    resolver=_resolver(report["legacy_model_run_id"],report["calendar"]["dates"],releases)
    state.check("legacy_checkpoint_target_and_hash_binding",_model_bindings(resolver)==report["legacy_models"]==identity["models"])
    state.check("legacy_checkpoint_unavailability_is_explicit",resolver.catalog_errors==report["legacy_catalog_errors"])
    state.check("legacy_frozen_request_identity",report["dataset_records"]==identity["records"]
        and report["config"]==identity["config"] and report["calendar"].get("verified") is True
        and stable_hash(report["calendar"]["dates"])==identity["calendar_hash"]
        and report["source_snapshot"]["snapshot_hash"]==identity["anchor_source_snapshot_hash"]
        and stable_hash(_read(root/"data_snapshots/actual_release_events.json"))==identity["actual_release_events_hash"])
    state.check("legacy_same_requested_cash",report["requested_initial_cash"]==100000.*len(report["dataset_records"]))
    for window in report["evaluation"]["windows"].values():
        for arm,bindings in window["ledger_bindings"].items():
            counts={key:sum(_read(Path(b["path"])/"legacy_routes.json")["coverage"].get(key,0) for b in bindings)
                for key in ("sessions","data_mature_checkpoint_sessions","model_input_and_probability_sessions",
                    "actual_release_eligible_sessions","actual_release_applied_sessions",
                    "offline_filter_candidate_sessions","rules_fallback_candidate_sessions")}
            state.check("legacy_aggregate_coverage_recomputation",window["variants"][arm]["legacy_model_coverage"]==counts)
            for binding in bindings:
                path=Path(binding["path"])/"legacy_routes.json"
                state.check("legacy_routes_file_sha256",file_hash(path)==binding["legacy_routes_sha256"],str(path))
                decisions=path.parent/"legacy_decisions.parquet"
                state.check("legacy_decisions_file_sha256",file_hash(decisions)==binding["legacy_decisions_sha256"],str(decisions))
                segments=_read(path)["routes"]
                for decision in pd.read_parquet(decisions).to_dict("records"):
                    expected=resolver.resolve(decision["date"])
                    segment=next((s for s in segments if s["start"]<=decision["date"]<=s["end"]),None)
                    state.check("legacy_exact_chronological_route",segment is not None and segment["identity"]==
                        {k:v for k,v in expected.items() if k not in {"date","model_dir"}},str(path))
                    if decision["ml_filter_applied"]:
                        state.check("legacy_no_future_head_application",bool(expected.get("model_dir"))
                            and expected["available_at"]<=decision["date"] and np.isfinite(decision["probability"]))
    compact={**report["evaluation"],"evaluation_sha256":report["evaluation_sha256"]}
    state.attempt("legacy_actual_account_audit",lambda:_audit_evaluation(root,compact,report["calendar"]["dates"],report["config"],
        {r["symbol"] for r in report["dataset_records"]},state,as_of=signal_time(as_of) if as_of else local_now(),
        records={r["symbol"]:r for r in report["dataset_records"]},bundles_hash=stable_hash(report["legacy_models"]),
        source_snapshot_hash=report["source_snapshot"]["snapshot_hash"]))
    if replay_policy:
        state.attempt("legacy_policy_reexecution",lambda:_audit_policy_replay(report,resolver,state))
    return {"run_id":run_id,"artifact_integrity_passed":not state.errors,"errors":state.errors,
        "checks":state.checks,"publication_allowed":False,"alias_updated":False,"formal_qualification":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-run-id")
    parser.add_argument("--legacy-model-run-id",default=DEFAULT_MA_RUN)
    parser.add_argument("--symbols",nargs="+")
    parser.add_argument("--cpu-workers",type=int,default=4)
    parser.add_argument("--audit-run-id")
    args=parser.parse_args(argv)
    if args.audit_run_id:
        result=audit_legacy_comparison(args.audit_run_id)
    else:
        if not args.anchor_run_id:parser.error("--anchor-run-id is required for a comparison")
        result=run_legacy_comparison(args.anchor_run_id,legacy_model_run_id=args.legacy_model_run_id,
            symbols=args.symbols,cpu_workers=args.cpu_workers,
            progress=lambda stage,completed,requested,failed:print(json.dumps({"stage":stage,
                "completed":completed,"requested":requested,"failed":failed}),flush=True))
        result={"run_id":result["run_id"],"requested_mainboard_count":result["requested_mainboard_count"],
            "requested_initial_cash":result["requested_initial_cash"],"publication_allowed":False}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 1 if result.get("artifact_integrity_passed") is False else 0


if __name__=="__main__":
    raise SystemExit(main())
