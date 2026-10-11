"""Read-only integrity and qualification audit for the interval model family.

An intact research artifact is not independent evidence. Missing future data
stays pending; missing historical identity and price basis stays a formal block.
This module neither writes a certificate nor changes a release alias.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_research_models import file_hash, identity_path, signal_time

AUDIT_VERSION = "czsc_high_return_independent_protocol_audit_v1"
FORWARD_KIND = "czsc-high-return-forward-evidence"
REQUIRED_HEADS = {"ma_entry", "ma_exit", "entry", "exit", "exit_advantage", "return", "loss"}
DEPENDENCE_BLOCKS = (160, 240)
REQUIRED_CODE_NAMES = {"czsc_high_return_models.py", "czsc_high_return_runtime.py",
    "czsc_high_return_research.py", "czsc_high_return_labels.py", "czsc_high_return_features.py",
    "czsc_high_return_evaluation.py", "czsc_high_return_audit.py", "czsc_trade_episodes.py",
    "czsc_signal_points.py", "czsc_backtest.py", "czsc_analysis.py", "czsc_adapter.py",
    "czsc_research_profiles.py", "czsc_wyckoff_features.py", "broker.py", "cost_model.py",
    "czsc_research_ml.py", "czsc_research_calendar.py", "ledger.py", "order.py", "czsc_high_return_research.json"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _inside(path, boundary):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(boundary).resolve()):
        raise ValueError("artifact outside frozen run storage")
    return path


def _close(left, right):
    return left is not None and right is not None and math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-5)


class _Audit:
    def __init__(self):
        self.checks, self.errors, self.pending, self.blocks = [], [], [], []
        self._passed = {}

    def check(self, rule, condition, detail=None):
        if condition:
            if rule in self._passed:
                self._passed[rule]["count"] += 1
            else:
                row = {"rule": rule, "status": "pass", "count": 1}
                self._passed[rule] = row; self.checks.append(row)
            return
        row = {"rule": rule, "status": "pass" if condition else "fail", "detail": detail}
        self.checks.append(row)
        if not condition:
            self.errors.append(row)

    def wait(self, rule, detail=None):
        row = {"rule": rule, "status": "pending", "detail": detail}
        self.checks.append(row); self.pending.append(row)

    def block(self, rule, detail=None):
        row = {"rule": rule, "status": "blocked", "detail": detail}
        self.checks.append(row); self.blocks.append(row)

    def attempt(self, rule, operation):
        try:
            return operation()
        except (ValueError, KeyError, TypeError, OSError, RuntimeError, OverflowError) as exc:
            self.check(rule, False, str(exc))
            return None


def _audit_models(root, report, calendar, state, as_of):
    from my_strategy.services.czsc_high_return_models import load_head
    from my_strategy.services.czsc_high_return_features import (MA_ENTRY_SCHEMA_HASH, MA_EXIT_SCHEMA_HASH,
        ENTRY_SCHEMA_HASH, EXIT_SCHEMA_HASH)
    from my_strategy.services.czsc_high_return_labels import label_config
    from my_strategy.services.czsc_trade_episodes import _settings
    settings = _settings(report.get("config", {})); expected_source = report.get("source_snapshot", {}).get("snapshot_hash")
    heads, manifests = set(), {}
    # Include every saved diagnostic and selected head, even an unreferenced one.
    for path in sorted((root / "models").rglob("manifest.json")):
        manifest = state.attempt("head_integrity", lambda path=path: load_head(path.parent))
        if manifest is None:
            continue
        key = str(path.parent.relative_to(root)); manifests[key] = manifest
        head = manifest["head"]; heads.add(head); contract = manifest["target_contract"]
        raw_schema = {"ma_entry": MA_ENTRY_SCHEMA_HASH, "ma_exit": MA_EXIT_SCHEMA_HASH,
            "entry": ENTRY_SCHEMA_HASH, "return": ENTRY_SCHEMA_HASH, "loss": ENTRY_SCHEMA_HASH,
            "exit": EXIT_SCHEMA_HASH, "exit_advantage": EXIT_SCHEMA_HASH}[head]
        state.check("head_source_calendar_execution_binding", contract.get("source_snapshot_hash") == expected_source
            and contract.get("calendar_hash") == stable_hash(calendar)
            and contract.get("execution_config_hash") == stable_hash(settings["execution"])
            and contract.get("strategy_configuration_hash") == stable_hash(settings)
            and contract.get("source") == ("ma" if head.startswith("ma_") else "union")
            and contract.get("feature_schema_hash") == raw_schema
            and contract.get("high_return") == label_config(report["config"])
            and contract.get("entry_policy") == "risk" and contract.get("initial_cash") == 100000.
            and bool(contract.get("continuation_policy_hash")), key)
        state.check("head_shadow_identity", manifest.get("publication_allowed") is False, key)
        fold = manifest["fold"]
        state.check("head_maturity_chronology", manifest["train_label_end"] <= fold["train_end"]
            < fold["calibration_start"] <= fold["calibration_end"] < fold["policy_start"]
            <= fold["policy_end"] <= fold["validation_end"] == manifest["available_at"]
            and manifest["calibration_label_end"] <= fold["calibration_end"], key)
        completed = state.attempt("head_actual_completion_time", lambda: signal_time(manifest["training_completed_at"]))
        if completed:
            state.check("head_actual_completion_time", len(manifest["training_completed_at"]) > 10
                and signal_time(manifest["training_started_at"]) <= completed <= as_of, key)
        preprocess = manifest["preprocess"]
        arrays = [v for v in preprocess.values() if isinstance(v, list)]
        state.check("head_preprocess_finite", all(np.isfinite(np.asarray(v, float)).all() for v in arrays), key)
    for bundle in [*report.get("model_bundles", []), *report.get("diagnostic_bundles", [])]:
        for head, binding in bundle.get("heads", {}).items():
            path = state.attempt("bundle_head_path", lambda: _inside(root / binding["path"], root))
            manifest = manifests.get(str(path.relative_to(root))) if path else None
            state.check("bundle_head_manifest_binding", manifest is not None and manifest["head"] == head
                and manifest["manifest_sha256"] == binding.get("manifest_sha256")
                and (not binding.get("target_contract_hash") or manifest["target_contract_hash"] == binding["target_contract_hash"]),
                {"bundle": bundle.get("name"), "head": head})
    if REQUIRED_HEADS - heads:
        state.wait("complete_model_heads", {"missing": sorted(REQUIRED_HEADS - heads)})
    return manifests


def _audit_freeze(root, report, calendar, manifests, state, as_of):
    freeze_path = root / "validation/protocol_freeze.json"
    if not freeze_path.is_file():
        state.wait("actual_complete_protocol_freeze", "protocol_freeze.json is absent")
        return None
    freeze = state.attempt("protocol_freeze_read", lambda: _read(freeze_path))
    if not freeze:
        return None
    body = {k: v for k, v in freeze.items() if k != "freeze_hash"}
    state.check("protocol_freeze_hash", freeze.get("freeze_hash") == stable_hash(body)
        and freeze == report.get("protocol_freeze"))
    state.check("protocol_freeze_bindings", freeze.get("config_hash") == stable_hash(report["config"])
        and freeze.get("source_snapshot_hash") == report["source_snapshot"]["snapshot_hash"]
        and freeze.get("model_bundles_hash") == stable_hash(report.get("model_bundles", [])))
    stamp = state.attempt("freeze_actual_time", lambda: signal_time(freeze["actual_completed_at"]))
    if stamp:
        state.check("freeze_after_all_training", len(freeze["actual_completed_at"]) > 10 and stamp <= as_of
            and all(signal_time(m["training_completed_at"]) <= stamp for m in manifests.values()))
    code = freeze.get("code_hashes", {})
    required = REQUIRED_CODE_NAMES | ({"czsc_high_return_forward.py", "czsc_high_return_forward_protocol.py"}
        if freeze.get("forward_protocol") else set())
    if report.get('config', {}).get('ma_entry_alignment') == 'signal_pinned_learned_exit_v2':
        required |= {'czsc_high_return_ma_training.py', 'czsc_high_return_ma_alignment.py',
            'czsc_high_return_ma_alignment_audit.py'}
    missing = required - {Path(p).name for p in code}
    if missing:
        state.block("frozen_transitive_execution_dependencies", {"missing": sorted(missing)})
    for path, sha in code.items():
        state.attempt("frozen_code_hash", lambda path=path, sha=sha: _require_hash(Path(path), sha))
    from my_strategy.adapters.czsc_adapter import SOURCE_SHA256, native_runtime
    state.attempt("frozen_native_attachment_integrity", native_runtime)
    if freeze.get("native_source_sha256") != SOURCE_SHA256 or freeze.get("calendar_hash") != stable_hash(calendar):
        state.block("native_source_calendar_freeze_binding", "native attachment and calendar must be explicitly frozen")
    # A future session string is insufficient: verify the calendar that supplied it.
    dates = calendar
    evidence = freeze.get("independent_calendar") or freeze.get("independent_start_calendar")
    if evidence:
        value = state.attempt("independent_start_calendar_hash", lambda: _read(_require_hash(
            _inside(evidence["path"], ARTIFACT_RUNS_ROOT), evidence["sha256"])))
        if value:
            state.check("independent_start_verified_calendar", value.get("verified") is True
                and value.get("dates") == sorted(set(value.get("dates", [])))
                and (not evidence.get("dates_hash") or evidence["dates_hash"] == stable_hash(value.get("dates", []))))
            dates = value.get("dates", [])
    first = next((day for day in dates if stamp and day > stamp.date().isoformat()), None)
    if first is None:
        state.wait("first_verified_session_after_actual_freeze", "no bound verified calendar session after actual freeze")
    else:
        state.check("first_verified_session_after_actual_freeze", freeze.get("independent_start") == first
            and report.get("independent_window", {}).get("start") == first)
    state.check("frozen_independent_minimums", int(freeze.get("minimum_technical_sessions", 0)) >= 480
        and int(freeze.get("minimum_windows", 0)) >= 3)
    return freeze


def _require_hash(path, sha):
    if not sha or file_hash(path) != sha:
        raise ValueError("artifact SHA256 mismatch: " + str(path))
    return path


def _forward_registry_records(model_run_id):
    """Use the existing result index without its schema-creating constructor."""
    from my_strategy.storage import czsc_results
    # Unit fixtures must not read the user's production index after rebasing
    # artifact storage. Tests that exercise the index supply their own reader.
    if Path(czsc_results.ARTIFACT_RUNS_ROOT).resolve() != Path(ARTIFACT_RUNS_ROOT).resolve():
        return []
    db_path = czsc_results.PROCESSED_DATA_ROOT / "czsc/results.db"
    if not db_path.is_file():
        return []
    # ResultStore.list caps all model families together at 200. A continuous
    # three-window append chain can outlive that cap; inspect its full index.
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT run_id,summary,report_path FROM czsc_runs WHERE kind=? ORDER BY created_at", (FORWARD_KIND,)).fetchall()
    return [_read(_inside(ARTIFACT_RUNS_ROOT / report_path, identity_path(ARTIFACT_RUNS_ROOT, run_id)))
        for run_id, summary, report_path in rows if json.loads(summary).get("model_run_id") == model_run_id]


def _forward_value(value, *names):
    supplied = [value[name] for name in names if name in value]
    if not supplied:
        return None
    if any(item != supplied[0] for item in supplied[1:]):
        raise ValueError("ambiguous forward identity aliases: " + "/".join(names))
    return supplied[0]


def _forward_end(value):
    return _forward_value(value, "observed_end", "data_end", "end")


def _forward_source_records(value):
    return _forward_value(value, "new_dataset_records", "dataset_records") or []


def _forward_calendar(value):
    binding = value["calendar"]
    path = _require_hash(_inside(binding["path"], ARTIFACT_RUNS_ROOT),
        _forward_value(binding, "sha256", "file_sha256"))
    calendar = _read(path)
    dates = calendar["dates"]
    if calendar.get("verified") is not True or not dates or dates != sorted(set(dates)):
        raise ValueError("forward calendar must be unique ascending and verified")
    if binding.get("dates_hash") != stable_hash(dates):
        raise ValueError("forward calendar date identity mismatch")
    return dates


def _prefix_frame(path, end):
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError("forward source has duplicate/unordered dates")
    # Old ten-session outcomes can mature after an append. These fields are
    # excluded from the causal input contract, not compared as prediction X.
    outcomes = {"label", "label_end", "label_reason", "label_available", "label_available_at",
        "net_return", "entry_date", "exit_date", "entry_shares", "fees"}
    frame = frame.loc[frame.date <= end].drop(columns=[c for c in frame if c in outcomes or c.startswith("label_")])
    return frame.reindex(sorted(frame.columns), axis=1).reset_index(drop=True)


def _continuous_bindings(evidence):
    compact = evidence["continuous_evaluation"]
    path = _inside(Path(compact["output_root"]) / "evaluation.json", ARTIFACT_RUNS_ROOT)
    if compact.get("evaluation_sha256"):
        _require_hash(path, compact["evaluation_sha256"])
    report = _read(path)
    if len(report.get("windows", {})) != 1:
        raise ValueError("forward evidence must use one continuous account interval")
    window = next(iter(report["windows"].values()))
    return report, window


def _audit_forward_slice(actual, saved, state, detail):
    state.check("forward_window_continuous_slice", actual.get("daily") == saved.get("daily")
        and actual.get("window_basis") == saved.get("window_basis"), detail)
    metrics, bound = actual.get("metrics", {}), saved.get("metrics", {})
    state.check("forward_window_actual_trade_metrics", all(
        bound.get(k) == v if isinstance(v, (str, bool)) or v is None else _close(v, bound.get(k))
        for k, v in metrics.items()), detail)


def _audit_forward_account_prefix(child, parent, end, state):
    _, old_window = _continuous_bindings({"continuous_evaluation": parent})
    _, new_window = _continuous_bindings({"continuous_evaluation": child})
    for arm in ("reference", "full"):
        old_accounts = {r["symbol"]: r for r in old_window["ledger_bindings"][arm]}
        new_accounts = {r["symbol"]: r for r in new_window["ledger_bindings"][arm]}
        state.check("forward_account_prefix_universe", set(old_accounts) == set(new_accounts), arm)
        for symbol, binding in old_accounts.items():
            if symbol not in new_accounts:
                continue
            for name in ("daily", "ledger", "rejections"):
                before_path = _require_hash(_inside(Path(binding["path"]) / (name + ".csv"), ARTIFACT_RUNS_ROOT), binding[name + "_sha256"])
                after_binding = new_accounts[symbol]
                after_path = _require_hash(_inside(Path(after_binding["path"]) / (name + ".csv"), ARTIFACT_RUNS_ROOT), after_binding[name + "_sha256"])
                left, right = pd.read_csv(before_path), pd.read_csv(after_path)
                left = left.loc[left.date <= end].reset_index(drop=True)
                right = right.loc[right.date <= end].reset_index(drop=True)
                state.check("forward_" + name + "_prefix_unchanged", list(left.columns) == list(right.columns)
                    and left.to_json(orient="records", double_precision=15) == right.to_json(orient="records", double_precision=15),
                    {"arm": arm, "symbol": symbol})


def _audit_forward_prefix(child, parent, state):
    from my_strategy.services.czsc_research_calendar import calendar_prefix_compatible
    before, after = _forward_calendar(parent), _forward_calendar(child)
    state.check("forward_calendar_append_prefix", calendar_prefix_compatible(before, after))
    end = _forward_end(parent)
    old = {r["symbol"]: r for r in _forward_source_records(parent)}
    new = {r["symbol"]: r for r in _forward_source_records(child)}
    state.check("forward_input_prefix_universe", set(old) == set(new))
    for symbol, record in old.items():
        if symbol not in new:
            continue
        _require_hash(_inside(record["path"], ARTIFACT_RUNS_ROOT), record["sha256"])
        _require_hash(_inside(new[symbol]["path"], ARTIFACT_RUNS_ROOT), new[symbol]["sha256"])
        left, right = _prefix_frame(record["path"], end), _prefix_frame(new[symbol]["path"], end)
        state.check("forward_causal_input_prefix_unchanged", list(left.columns) == list(right.columns)
            and left.to_json(orient="records", date_format="iso", double_precision=15)
            == right.to_json(orient="records", date_format="iso", double_precision=15), symbol)
    _audit_forward_account_prefix(child["continuous_evaluation"], parent["continuous_evaluation"], end, state)
    prior_stress = {float(r["cost_multiplier"]): r for r in parent.get("cost_stress", [])}
    new_stress = {float(r["cost_multiplier"]): r for r in child.get("cost_stress", [])}
    state.check("forward_cost_pressure_append_prefix", set(prior_stress) == set(new_stress))
    for multiplier in prior_stress.keys() & new_stress.keys():
        _audit_forward_account_prefix(new_stress[multiplier], prior_stress[multiplier], end, state)


def _forward_window_trade_metrics(bindings, dates):
    """Independently count economic closes and actual fill costs in the slice."""
    if not dates:
        return {"completed_round_trips": 0, "fees": 0., "trade_count": 0}
    selected = set(dates); closed, fees, trades = [], 0., 0
    for binding in bindings:
        location = _inside(binding["path"], ARTIFACT_RUNS_ROOT)
        ledger = pd.read_csv(_require_hash(location / "ledger.csv", binding["ledger_sha256"]))
        fills = ledger.loc[ledger.date.astype(str).isin(selected)]
        fees += float(fills.fee.sum()); trades += len(fills)
        episodes = pd.read_csv(_require_hash(location / "episodes.csv", binding["episodes_sha256"]))
        if "exit_date" not in episodes:
            continue
        mature = episodes.loc[episodes.exit_date.notna() & episodes.exit_date.astype(str).isin(selected)]
        returns = pd.to_numeric(mature.net_return, errors="raise")
        if not np.isfinite(returns).all():
            raise ValueError("closed forward round trip has no finite net return")
        closed.extend(returns.tolist())
    return {"completed_round_trips": len(closed), "fees": fees, "trade_count": trades,
        "closed_round_trip_expectancy": float(np.mean(closed)) if closed else None,
        "trade_metric_scope": "whole_economic_roundtrips_closed_in_window_and_fill_fees_in_window"}


def _forward_economic_values(reference, full, config):
    ref, alt = reference["metrics"], full["metrics"]
    protocol = _forward_statistics_api()[0](config)
    if min(ref.get("completed_round_trips", 0), alt.get("completed_round_trips", 0)) < protocol["min_window_round_trips"]:
        return False
    values = [metrics.get(key) for metrics in (ref, alt) for key in
        ("period_net_return", "max_drawdown", "closed_round_trip_expectancy")]
    if any(value is None or not np.isfinite(value) for value in values):
        return False
    return bool(alt["period_net_return"] > ref["period_net_return"]
        and alt["max_drawdown"] <= ref["max_drawdown"] + min(.02, config.get("drawdown_tolerance", .02))
        and alt["closed_round_trip_expectancy"] > ref["closed_round_trip_expectancy"])


def _forward_economic_gate(reference, full, protocol, config):
    ref, alt = reference["metrics"], full["metrics"]
    if min(ref["completed_round_trips"], alt["completed_round_trips"]) < protocol["min_window_round_trips"]:
        return "pending"
    if ref.get("closed_round_trip_expectancy") is None or alt.get("closed_round_trip_expectancy") is None:
        return "pending"
    return "passed" if _forward_economic_values(reference, full, config) else "failed"


def _forward_statistics_api():
    from my_strategy.services.czsc_high_return_forward_protocol import (forward_protocol, forward_windows,
        slice_continuous_variant, paired_forward_statistics)
    return forward_protocol, forward_windows, slice_continuous_variant, paired_forward_statistics


def _audit_forward_registry(root, training, freeze, state, now):
    """Audit only the terminal cumulative evidence, plus its immutable chain."""
    values = state.attempt("forward_result_registry", lambda: _forward_registry_records(root.name)) or []
    if not values:
        state.wait("three_complete_independent_forward_windows", "no immutable forward evidence is registered")
        return None
    if freeze is None:
        state.wait("forward_requires_actual_protocol_freeze")
        return None
    expected_freeze = freeze.get("freeze_hash")
    registered = {}
    for value in values:
        if value.get("model_run_id") != root.name:
            continue
        if value.get("status") != "completed_shadow_evidence":
            state.wait("forward_evidence_not_completed", value.get("run_id"))
            continue
        state.check("forward_registry_freeze_identity", _forward_value(value, "freeze_hash", "original_freeze_hash") == expected_freeze,
            value.get("run_id"))
        if _forward_value(value, "freeze_hash", "original_freeze_hash") != expected_freeze:
            continue
        location = identity_path(ARTIFACT_RUNS_ROOT, value["run_id"]) / "reports/forward.json"
        saved = _read(location)
        indexed = _read(location.parent / "czsc.json")
        state.check("forward_registry_immutable_report", value == saved == indexed and saved.get("run_id") == location.parent.parent.name
            and saved.get("model_family") == "high_return" and saved.get("publication_allowed") is False
            and saved.get("model_gate") is False,
            value["run_id"])
        registered[value["run_id"]] = saved
    if not registered:
        state.wait("three_complete_independent_forward_windows", "no evidence matches the original actual freeze")
        return None
    tip = max(registered.values(), key=lambda value: (str(_forward_end(value)), str(value.get("completed_at", "")), value["run_id"]))
    chain, seen, current = [], set(), tip
    while current:
        run_id = current["run_id"]
        if run_id in seen:
            raise ValueError("forward append chain cycle")
        seen.add(run_id); chain.append(current)
        state.check("forward_same_actual_freeze_and_models", current.get("model_run_id") == root.name
            and _forward_value(current, "freeze_hash", "original_freeze_hash") == expected_freeze
            and _forward_value(current, "model_bundles_hash", "model_bundle_hash") == freeze.get("model_bundles_hash")
            and current.get("config_hash") == freeze.get("config_hash")
            and (freeze.get("forward_protocol") is None or current.get("forward_protocol") == freeze.get("forward_protocol")), run_id)
        created = _forward_value(current, "created_at") or current.get("run_context", {}).get("created_at")
        completed = _forward_value(current, "actual_completed_at", "completed_at")
        end = _forward_end(current)
        state.check("forward_actual_time_and_observed_end", bool(created) and bool(completed) and bool(end)
            and len(str(created)) > 10 and len(str(completed)) > 10
            and signal_time(freeze["actual_completed_at"]) <= signal_time(created) <= signal_time(completed) <= now
            and signal_time(end) <= signal_time(completed), run_id)
        _, current_window = _continuous_bindings(current)
        current_start = current.get("independent_start", current.get("start", current_window["fold"]["start"]))
        state.check("forward_start_after_same_actual_freeze", current_start == freeze.get("independent_start")
            and end >= str(freeze.get("independent_start")), run_id)
        previous = current.get("previous_run_id")
        if not previous:
            state.check("forward_first_append_no_parent", not current.get("append_parent_report_path")
                and not _forward_value(current, "append_parent_report_sha256", "append_parent_report_sha"), run_id)
            break
        parent_path = identity_path(ARTIFACT_RUNS_ROOT, previous) / "reports/forward.json"
        state.check("forward_parent_report_path", Path(current["append_parent_report_path"]).resolve() == parent_path.resolve(), run_id)
        _require_hash(parent_path, _forward_value(current, "append_parent_report_sha256", "append_parent_report_sha"))
        parent = _read(parent_path)
        state.check("forward_parent_registered_identity", previous in registered and registered.get(previous) == parent, previous)
        state.check("forward_append_increasing_dates", parent.get("run_id") == previous and _forward_end(parent) < end, run_id)
        state.check("forward_prefix_declared_verified", all(current.get("prefix_audit", {}).get(k) is True for k in
            ("unchanged_verified", "same_calendar_prefix", "ledger_and_daily_prefix_verified")), run_id)
        state.attempt("forward_independent_prefix_audit", lambda current=current, parent=parent: _audit_forward_prefix(current, parent, state))
        current = parent
    state.check("forward_registry_one_append_tip", set(registered) <= seen,
        {"selected_tip": tip["run_id"], "unlinked_runs": sorted(set(registered) - seen)})
    calendar = _forward_calendar(tip)
    symbols = {record["symbol"] for record in training["dataset_records"]}
    records = _forward_source_records(tip)
    state.check("forward_complete_original_frozen_universe", len(records) == len(symbols)
        and {r["symbol"] for r in records} == symbols)
    for record in records:
        _require_hash(_inside(record["path"], ARTIFACT_RUNS_ROOT), record["sha256"])
    original = _forward_value(tip, "original_source_snapshot", "model_source_snapshot")
    state.check("forward_original_model_source_snapshot", (original == training.get("source_snapshot") if original is not None
        else tip.get("original_training_source_hash") == training.get("source_snapshot", {}).get("snapshot_hash")))
    source_snapshot = tip["source_snapshot"]
    state.check("forward_new_source_snapshot_identity", source_snapshot.get("snapshot_hash") == tip.get("source_snapshot_hash")
        and source_snapshot.get("snapshot_hash") == stable_hash({k:v for k,v in source_snapshot.items()
            if k not in {"snapshot_hash", "calendar_run_id"}}))
    for name in ("report", "calendar"):
        _require_hash(_inside(source_snapshot[name + "_path"], ARTIFACT_RUNS_ROOT), source_snapshot[name + "_sha256"])
    source_calendar = _read(source_snapshot["calendar_path"])
    state.check("forward_new_source_calendar_binding", source_calendar.get("verified") is True
        and source_calendar.get("dates") == calendar and source_snapshot.get("calendar_hash") == stable_hash(calendar))
    source_report = _read(source_snapshot["report_path"])
    source_records = {r["symbol"]: r for r in source_report.get("dataset_records", [])}
    record_keys = ("symbol", "path", "sha256", "data_end", "data_version")
    state.check("forward_new_source_record_bindings", all(
        {k: source_records.get(r["symbol"], {}).get(k) for k in record_keys} == {k: r.get(k) for k in record_keys}
        for r in records) and source_snapshot.get("records_hash") == stable_hash(records))
    original_records = {r["symbol"]: r for r in training["dataset_records"]}
    from my_strategy.services.czsc_research_calendar import calendar_prefix_compatible
    original_calendar = _read(training["source_snapshot"]["calendar_path"])["dates"]
    state.check("forward_original_calendar_append_prefix", calendar_prefix_compatible(original_calendar, calendar))
    for record in records:
        old = original_records.get(record["symbol"])
        if old:
            _require_hash(_inside(old["path"], ARTIFACT_RUNS_ROOT), old["sha256"])
            left, right = _prefix_frame(old["path"], training["data_end"]), _prefix_frame(record["path"], training["data_end"])
            state.check("forward_original_causal_input_prefix_unchanged", list(left.columns) == list(right.columns)
                and left.to_json(orient="records", date_format="iso", double_precision=15)
                == right.to_json(orient="records", date_format="iso", double_precision=15), record["symbol"])
    requested = {s for s in symbols if s.startswith("60") and s.endswith(".SH") or s.startswith("00") and s.endswith(".SZ")}
    actual = _audit_evaluation(ARTIFACT_RUNS_ROOT, tip["continuous_evaluation"], calendar, training["config"], requested, state,
        as_of=now, records={r["symbol"]: r for r in records}, bundles_hash=freeze["model_bundles_hash"],
        source_snapshot_hash=tip.get("source_snapshot_hash"))
    _require_hash(_inside(Path(tip["continuous_evaluation"]["output_root"]) / "evaluation.json", ARTIFACT_RUNS_ROOT),
        tip["continuous_evaluation"].get("evaluation_sha256"))
    continuous_report, continuous = _continuous_bindings(tip)
    state.check("forward_single_continuous_reference_full_accounts", set(continuous["variants"]) == {"reference", "full"}
        and continuous["fold"]["start"] == freeze.get("independent_start")
        and continuous["fold"]["end"] == _forward_end(tip))
    stress = tip.get("cost_stress", [])
    state.check("forward_cost_pressure_unique_frozen_multipliers", len(stress) == len({float(s.get("cost_multiplier", 0)) for s in stress})
        and {float(s.get("cost_multiplier", 0)) for s in stress} <= {1.5, 2.})
    if {1.5, 2.} - {float(s.get("cost_multiplier", 0)) for s in stress}:
        state.wait("forward_complete_cost_pressure_replays")
    for item in stress:
        _require_hash(_inside(Path(item["output_root"]) / "evaluation.json", ARTIFACT_RUNS_ROOT), item.get("evaluation_sha256"))
        state.attempt("forward_cost_pressure_actual_ledger", lambda item=item: _audit_evaluation(
            ARTIFACT_RUNS_ROOT, item, calendar, training["config"], requested, state, as_of=now,
            records={r["symbol"]: r for r in records}, bundles_hash=freeze["model_bundles_hash"],
            source_snapshot_hash=tip.get("source_snapshot_hash")))
    pressure = {}
    for item in stress:
        _, window = _continuous_bindings({"continuous_evaluation": item})
        state.check("forward_cost_pressure_same_continuous_interval", window["fold"]["start"] == freeze["independent_start"]
            and window["fold"]["end"] == _forward_end(tip) and set(window["variants"]) == {"reference", "full"})
        pressure[float(item["cost_multiplier"])] = window
    protocol = freeze.get("forward_protocol")
    summary = {"latest_run_id": tip["run_id"], "chain_run_ids": [r["run_id"] for r in reversed(chain)],
        "observed_end": _forward_end(tip), "continuous_evaluation": actual, "completed_windows": 0, "qualified_windows": 0}
    if protocol is None:
        state.wait("forward_statistics_protocol_preregistered", "original actual freeze has no forward_protocol")
        state.wait("three_complete_independent_forward_windows")
        return summary
    protocol_builder, windows_builder, slicer, statistics = _forward_statistics_api()
    state.check("forward_preregistered_protocol_identity", protocol == protocol_builder(training["config"])
        and protocol.get("window_count", 0) >= 3 and protocol.get("window_sessions", 0) >= 480
        and protocol.get("block_lengths") == [160, 240] and protocol.get("minimum_power", 0) >= .8
        and protocol.get("familywise_alpha", 1) <= .05 and protocol.get("preregistered_comparisons", 0) >= 6
        and protocol.get("minimum_detectable_annual_net_improvement", 0) > 0)
    windows = windows_builder(calendar, freeze["independent_start"], _forward_end(tip), training["config"])
    stored = _forward_value(tip, "forward_windows", "windows") or []
    state.check("forward_window_plan_no_duplicates", len(stored) == len(windows)
        and len({r.get("window_id") for r in stored}) == len(stored))
    saved = {r.get("window_id"): r for r in stored}
    summaries = []
    for window in windows:
        value = saved.get(window["window_id"], {})
        for key in window:
            state.check("forward_frozen_window_identity", value.get(key) == window.get(key), {"window_id": window["window_id"], "field": key})
        dates = window.get("dates", window.get("observed_verified_dates", []))
        all_dates = [r["date"] for r in continuous["variants"]["reference"]["daily"]]
        reference = slicer(continuous["variants"]["reference"], dates, all_dates)
        full = slicer(continuous["variants"]["full"], dates, all_dates)
        for arm, variant in (("reference", reference), ("full", full)):
            trade_metrics = _forward_window_trade_metrics(continuous["ledger_bindings"][arm], dates)
            variant["metrics"].update(trade_metrics)
            saved_variant = value.get("variants", {}).get(arm, {})
            _audit_forward_slice(variant, saved_variant, state, {"window_id": window["window_id"], "arm": arm})
        state.check("forward_saved_economic_gate", value.get("economic_gate") is _forward_economic_values(reference, full, training["config"]), window["window_id"])
        recomputed = statistics(reference, full, training["config"])
        saved_statistics = value.get("statistics", value.get("paired_statistics"))
        if saved_statistics is not None:
            state.check("forward_statistics_independent_recomputation", stable_hash(saved_statistics) == stable_hash(recomputed), window["window_id"])
        else:
            state.check("forward_statistics_evidence_present", False, window["window_id"])
        complete = window.get("complete") is True
        qualified = complete and recomputed.get("qualified") is True
        economic_status = _forward_economic_gate(reference, full, protocol, training["config"]) if complete else "pending"
        qualified = qualified and economic_status == "passed"
        if complete:
            if economic_status == "failed":
                state.block("forward_fee_net_performance_gate", window["window_id"])
            elif economic_status == "pending":
                state.wait("forward_fee_net_performance_gate", window["window_id"])
        pressure_status = {}
        stored_pressure = {float(item["cost_multiplier"]): item for item in value.get("cost_stress", [])}
        state.check("forward_window_pressure_complete_identity", len(value.get("cost_stress", [])) == len(stored_pressure)
            and set(stored_pressure) == set(pressure), window["window_id"])
        for multiplier in (1.5, 2.):
            status = "pending"
            if multiplier in pressure:
                saved_pressure = pressure[multiplier]; variants = {}
                for arm in ("reference", "full"):
                    pressure_dates = [r["date"] for r in saved_pressure["variants"][arm]["daily"]]
                    variants[arm] = slicer(saved_pressure["variants"][arm], dates, pressure_dates)
                    variants[arm]["metrics"].update(_forward_window_trade_metrics(saved_pressure["ledger_bindings"][arm], dates))
                    _audit_forward_slice(variants[arm], stored_pressure.get(multiplier, {}).get("variants", {}).get(arm, {}), state,
                        {"window_id": window["window_id"], "multiplier": multiplier, "arm": arm})
                state.check("forward_saved_pressure_economic_gate", stored_pressure.get(multiplier, {}).get("economic_gate") is
                    _forward_economic_values(variants["reference"], variants["full"], training["config"]),
                    {"window_id": window["window_id"], "multiplier": multiplier})
                if any(v.get("cash_retained_failures") for v in variants.values()):
                    status = "failed"
                elif complete:
                    status = _forward_economic_gate(variants["reference"], variants["full"], protocol, training["config"])
                if status == "failed":
                    state.block("forward_cost_pressure_fee_net_gate", {"window_id": window["window_id"], "multiplier": multiplier})
                elif status == "pending":
                    state.wait("forward_cost_pressure_mature_round_trips", {"window_id": window["window_id"], "multiplier": multiplier})
            pressure_status[str(multiplier)] = status
        qualified = qualified and all(s == "passed" for s in pressure_status.values())
        summary["completed_windows"] += int(complete); summary["qualified_windows"] += int(qualified)
        if recomputed.get("status") == "failed":
            state.block("forward_window_performance_not_passed", {"window_id": window["window_id"], "statistics": recomputed})
        elif not qualified:
            state.wait("forward_window_power_and_uncertainty", {"window_id": window["window_id"], "status": recomputed.get("status")})
        summaries.append({"window_id": window["window_id"], "complete": complete, "statistics": recomputed,
            "economic_status": economic_status, "cost_pressure_status": pressure_status, "qualified": qualified})
    summary["windows"] = summaries
    total_trips = sum(_forward_window_trade_metrics(continuous["ledger_bindings"]["full"], window.get("dates", []))["completed_round_trips"]
        for window in windows if window.get("complete"))
    summary["full_completed_round_trips_in_complete_windows"] = total_trips
    if total_trips < int(protocol.get("min_total_round_trips", 200)):
        state.wait("forward_minimum_total_mature_round_trips", {"actual": total_trips, "required": protocol.get("min_total_round_trips", 200)})
    state.check("forward_saved_overall_qualification", tip.get("technical_statistics_qualified") is
        (summary["qualified_windows"] >= protocol["window_count"] and total_trips >= protocol["min_total_round_trips"]))
    if summary["qualified_windows"] < max(3, int(protocol["window_count"])):
        state.wait("three_complete_independent_forward_windows", {"completed": summary["completed_windows"], "qualified": summary["qualified_windows"]})
    else:
        state.check("three_complete_independent_forward_windows", True)
    return summary


def _verified_reused_episode_paths(root, report, calendar):
    """Allow only the exact files of an independently verified complete stage."""
    declaration = root / "dataset/final/artifact_reuse.json"
    if not declaration.is_file():
        return set()
    saved = _read(declaration)
    source_run_id = saved["source_run_id"]
    if source_run_id == root.name:
        raise ValueError("complete final artifact reuse must reference a different original run")
    bundles = _read(root / "models/ma_bundles.json")
    reported_ma = [bundle for bundle in report.get("model_bundles", [])
        if bundle.get("heads") and set(bundle["heads"]) <= {"ma_entry", "ma_exit"}]
    if bundles != reported_ma:
        raise ValueError("complete final artifact reuse MA bundles differ from audited model bindings")
    from my_strategy.services.czsc_high_return_research import _verified_final_artifacts
    final, verification = _verified_final_artifacts(source_run_id,
        records=report["dataset_records"], calendar=calendar, snapshot=report["source_snapshot"],
        config=report["config"], bundles=bundles)
    if saved != verification or _read(root / "dataset/final/manifest.json") != final:
        raise ValueError("complete final artifact reuse provenance or manifest differs")
    if report.get("episode_records") != final["records"]:
        raise ValueError("complete final artifact reuse report records differ")
    return {Path(record[key + "_path"]).resolve() for record in final["records"]
        for key in (("points", "entries", "exits", "episodes", "contracts")
            if record["episode_status"] == "completed" else ("points",))}


def _audit_episode_artifacts(root, report, calendar, state):
    from my_strategy.services.czsc_high_return_labels import label_config
    from my_strategy.services.czsc_trade_episodes import _settings
    settings = _settings(report["config"]); count = 0
    reused = state.attempt("complete_final_artifact_reuse_provenance",
        lambda: _verified_reused_episode_paths(root, report, calendar)) or set()
    if reused:
        state.check("complete_final_artifact_reuse_provenance", True)
    def artifact_path(path):
        resolved = Path(path).resolve()
        return resolved if resolved in reused else _inside(resolved, root)
    for record in report.get("episode_records", []):
        if record.get("episode_status") != "completed":
            continue
        verified = {}
        for key in ("points", "entries", "exits", "episodes", "contracts"):
            verified[key] = state.attempt("episode_artifact_sha256", lambda key=key, record=record: _require_hash(
                artifact_path(record[key + "_path"]), record[key + "_sha256"]))
        contracts = (state.attempt("episode_target_contracts_read", lambda: _read(verified["contracts"])) or {}) \
            if verified["contracts"] else {}
        for source, heads in contracts.items():
            for head, contract in heads.items():
                body = {k:v for k,v in contract.items() if k != "contract_hash"}
                state.check("episode_target_contract_identity", contract.get("contract_hash") == stable_hash(body)
                    and contract.get("source") == source and contract.get("source_snapshot_hash") == report["source_snapshot"]["snapshot_hash"]
                    and contract.get("calendar_hash") == stable_hash(calendar)
                    and contract.get("high_return") == label_config(report["config"])
                    and contract.get("execution_config_hash") == stable_hash(settings["execution"])
                    and contract.get("strategy_configuration_hash") == stable_hash(settings)
                    and contract.get("initial_cash") == 100000., {"symbol": record["symbol"], "source": source, "head": head})
        count += 1
    if not count:
        state.wait("final_episode_artifact_contracts", "final frozen-policy episode artifacts are absent")


@lru_cache(maxsize=2)
def _source_bars(path, sha):
    _require_hash(Path(path), sha)
    frame = pd.read_parquet(path, columns=["date", "symbol", "open", "high", "low", "close",
        "volume", "amount", "source", "has_trade_price"])
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    if frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError("frozen raw dates are not unique ascending")
    rows = list(frame.itertuples(index=False))
    return rows, {row.date: i for i, row in enumerate(rows)}


def _audit_account(binding, dates, cost, state, *, record=None, settings=None, config=None, window_dates=None):
    from my_strategy.services.czsc_backtest import _tick, _guard
    from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
    root = _inside(binding["path"], ARTIFACT_RUNS_ROOT)
    for name in ("daily", "ledger", "episodes", "rejections", "holding_snapshots", "entry_diagnostics",
            "exit_policy_diagnostics", "entry_gate_diagnostics", "account"):
        suffix = ".csv" if name in {"daily", "ledger", "episodes", "rejections"} else ".json" if name == "account" else ".jsonl"
        sha = binding.get(name + "_sha256")
        if sha:
            _require_hash(root / (name + suffix), sha)
        else:
            state.block("complete_account_artifact_hash_bindings", {"path": str(root), "missing": name})
    daily = pd.read_csv(root / "daily.csv"); ledger = pd.read_csv(root / "ledger.csv")
    metadata = _read(root / "account.json")
    from my_strategy.services.czsc_high_return_evaluation import CASH_NOT_OBSERVED, SOURCE_AFTER_WINDOW
    state.check("account_status_and_reason_identity", metadata.get("status") == binding.get("status")
        and metadata.get("reason") == binding.get("reason")
        and binding.get("status") in {"complete", "cash_retained_failure", CASH_NOT_OBSERVED}, str(root))
    if binding.get("status") == "complete":
        state.check("account_actual_cost_contract", metadata.get("cost_model") == cost.as_config_dict(), str(root))
    state.check("account_daily_verified_order", not daily.empty and daily.date.tolist() == sorted(set(daily.date))
        and set(daily.date) <= set(dates), str(root))
    cash, shares, bought = 100000., 0, None
    raw, raw_index = _source_bars(record["path"], record["sha256"]) if record else ([], {})
    if binding.get("status") == CASH_NOT_OBSERVED:
        expected_dates = window_dates or daily.date.tolist()
        rejections = pd.read_csv(root / "rejections.csv")
        state.check("not_observed_cash_verified_source_coverage", bool(record) and bool(raw) and bool(expected_dates)
            and all(r.symbol == binding["symbol"] for r in raw)
            and raw[0].date > expected_dates[-1]
            and binding.get("reason") == SOURCE_AFTER_WINDOW
            and binding.get("source_first_date") == metadata.get("source_first_date") == (raw[0].date if raw else None), str(root))
        state.check("not_observed_cash_no_replaced_inventory", ledger.empty and rejections.empty
            and pd.read_csv(root / "episodes.csv").empty
            and not (root / "holding_snapshots.jsonl").read_text().strip()
            and daily.date.tolist() == expected_dates
            and (daily.cash == 100000.).all() and (daily.equity == 100000.).all()
            and (daily.market_value == 0.).all() and (daily.shares == 0).all(), str(root))
    seasoning = np.cumsum([r.volume > 0 and r.amount > 0 and r.has_trade_price == 1
        and r.source in VERIFIED_SOURCES for r in raw])
    transitions = {}
    next_day = dict(zip(dates[:-1], dates[1:]))
    for row in ledger.to_dict("records"):
        day, signal = str(row["date"]), str(row["signal_date"])
        side = "BUY" if row["action"] == "BUY" else "SELL"
        qty, price, fee = int(row["shares"]), float(row["price"]), float(row["fee"])
        state.check("ledger_actual_symbol", row["symbol"] == binding["symbol"], str(root))
        state.check("ledger_next_verified_open", next_day.get(signal) == day, str(root))
        state.check("ledger_round_lot_price", qty > 0 and qty == row["shares"] and qty % 100 == 0
            and _close(price, _tick(price)), str(root))
        if raw:
            offset = raw_index.get(day)
            state.check("ledger_raw_execution_session", offset is not None and offset > 0, str(root))
            if offset is not None and offset > 0:
                current, prior = raw[offset], raw[offset - 1]
                state.check("ledger_source_open_slippage_guard", _close(price, _tick(cost.execution_price(side, current.open)))
                    and _guard(binding["symbol"], current, prior, side, int(seasoning[offset - 1]), settings["execution"]) is None
                    and current.source in VERIFIED_SOURCES and prior.source in VERIFIED_SOURCES, str(root))
                capacity = int(prior.volume * settings["execution"]["max_volume_participation"]) // 100 * 100
                state.check("ledger_prior_liquidity_capacity", qty <= capacity, str(root))
        expected_fee = cost.fees(side, qty * price)
        flow = -qty * price - expected_fee if side == "BUY" else qty * price - expected_fee
        state.check("ledger_fee_and_cash_flow", _close(fee, expected_fee) and _close(row["cash_flow"], flow), str(root))
        before = shares; shares += qty if side == "BUY" else -qty; cash += flow
        state.check("ledger_inventory_cash_conservation", before == int(row["position_before"])
            and shares == int(row["position_after"]) and shares >= 0 and cash >= -1e-7
            and _close(cash, row["cash_after"]), str(root))
        if side == "BUY":
            state.check("ledger_single_entry_flat_state", before == 0, str(root)); bought = day
        else:
            state.check("ledger_t_plus_one", bought is not None and day > bought, str(root))
        transitions.setdefault(day, []).append((cash, shares))
    cash, shares = 100000., 0
    for row in daily.to_dict("records"):
        if row["date"] in transitions:
            cash, shares = transitions[row["date"]][-1]
        close = row.get("close")
        expected_equity = cash + shares * float(close) if close is not None else cash
        state.check("daily_actual_ledger_cash_shares_equity", _close(row["cash"], cash)
            and int(row.get("shares", 0)) == shares and _close(row["equity"], expected_equity), str(root))
        if raw and row["date"] in raw_index:
            state.check("daily_source_price_mark", _close(close, raw[raw_index[row["date"]]].close), str(root))
    if binding.get("status") == "cash_retained_failure":
        state.check("failed_account_cash_retained", ledger.empty and (daily.equity == 100000.).all()
            and (daily.cash == 100000.).all() and bool(binding.get("reason")), str(root))
    from my_strategy.services.czsc_high_return_labels import episode_outcome
    saved_episodes = pd.read_csv(root / "episodes.csv")
    snapshots = [json.loads(line) for line in (root / "holding_snapshots.jsonl").read_text().splitlines() if line]
    stops = {r["entry_date"]: r.get("stop_price") for r in snapshots}
    episodes, current = [], []
    for row in ledger.to_dict("records"):
        current.append(row)
        if row["position_after"] == 0:
            episodes.append(current); current = []
    if current:
        episodes.append(current)
    state.check("actual_episode_count", len(saved_episodes) == len(episodes), str(root))
    for rows, saved in zip(episodes, saved_episodes.to_dict("records")):
        outcome = episode_outcome(rows, stop_price=stops.get(rows[0]["date"]), cost_model=cost, config=config)
        state.check("episode_actual_net_outcome_and_maturity", saved["entry_date"] == outcome["entry_date"]
            and (pd.isna(saved["exit_date"]) if outcome["exit_date"] is None else saved["exit_date"] == outcome["exit_date"])
            and (pd.isna(saved["net_return"]) if not outcome["exit_date"] else _close(saved["net_return"], outcome["net_return"]))
            and (pd.isna(saved["outcome_class"]) if outcome["outcome_class"] is None else saved["outcome_class"] == outcome["outcome_class"]), str(root))
    return len(ledger)


def _audit_evaluation(root, compact, calendar, config, requested, state, *, independent=False,
        freeze=None, as_of=None, records=None, bundles_hash=None, source_snapshot_hash=None):
    from my_strategy.services.czsc_high_return_evaluation import _aggregate_bindings, _paired_uncertainty, _stress_config
    from my_strategy.services.czsc_trade_episodes import _cost, _settings
    path = _inside(Path(compact["output_root"]) / "evaluation.json", root)
    if compact.get("evaluation_sha256"):
        _require_hash(path, compact["evaluation_sha256"])
    full = _read(path)
    state.check("evaluation_report_identity", full.get("evaluation_identity") == compact.get("evaluation_identity")
        and full.get("cost_multiplier") == compact.get("cost_multiplier") and full.get("thresholds_refit") is False)
    expected = {"calendar_hash": stable_hash(calendar), "strategy_configuration_hash": stable_hash(_settings(config)),
        "base_execution_config_hash": stable_hash(_settings(config)["execution"])}
    if records is not None:
        expected["source_records_hash"] = stable_hash([r for r in records.values() if r["symbol"] in requested])
    if bundles_hash is not None:
        expected["model_bundles_hash"] = bundles_hash
    if source_snapshot_hash is not None:
        expected["source_snapshot_hash"] = source_snapshot_hash
    for key, value in expected.items():
        if full.get(key) is None:
            state.block("evaluation_frozen_identity_bindings", {"path": str(path), "missing": key})
        else:
            state.check("evaluation_frozen_identity_bindings", full[key] == value, {"path": str(path), "key": key})
    stressed = _stress_config(config, float(full.get("cost_multiplier", 1)))
    settings = _settings(stressed); cost = _cost(settings); results = []
    for name, window in full.get("windows", {}).items():
        fold = window["fold"]; start, end = fold["start"], fold["end"]
        dates = [d for d in calendar if start <= d <= end]
        state.check("evaluation_actual_as_of", signal_time(end) <= as_of, name)
        if not independent:
            state.check("historical_window_diagnostic_only", window.get("qualification") == "observed_historical_development_diagnostic"
                and window.get("model_gate") is False and window.get("publication_allowed") is False, name)
        recomputed = {}
        for arm, bindings in window.get("ledger_bindings", {}).items():
            state.check("complete_frozen_account_universe", {r["symbol"] for r in bindings} == requested
                and len(bindings) == len(requested) and window.get("initial_cash") == 100000. * len(requested), name + ":" + arm)
            for binding in bindings:
                state.attempt("actual_account_audit", lambda binding=binding: _audit_account(binding, calendar, cost, state,
                    record=(records or {}).get(binding["symbol"]), settings=settings, config=config, window_dates=dates))
            value = _aggregate_bindings(bindings, dates, 100000.); recomputed[arm] = value
            saved = window["variants"][arm]["metrics"]
            state.check("portfolio_ledger_metric_recomputation", all(_close(value["metrics"][k], saved.get(k))
                for k in ("initial_cash", "final_equity", "total_return", "fees", "completed_round_trips", "max_drawdown")), name + ":" + arm)
            state.check("portfolio_daily_actual_account_recomputation", value["daily"] == window["variants"][arm].get("daily"), name + ":" + arm)
            state.check("portfolio_cash_coverage_classification_recomputation", value["cash_retained_failures"] == window["variants"][arm].get("cash_retained_failures")
                and value["market_unavailable_accounts"] == window["variants"][arm].get("market_unavailable_accounts", []), name + ":" + arm)
            if independent:
                state.check("independent_no_failed_cash_accounts", not value["cash_retained_failures"], name + ":" + arm)
        state.check("main_comparison_actual_accounts", {"full", "reference"} <= set(recomputed), name)
        if independent and {"full", "reference"} <= set(recomputed):
            ci = _paired_uncertainty(recomputed["reference"], recomputed["full"], config)
            state.check("independent_480_sessions", len(dates) >= 480, name)
            for block in DEPENDENCE_BLOCKS:
                item = ci[str(block)]
                state.check("independent_complete_positive_date_blocks", "unavailable" not in item
                    and item.get("block_length") == block and item.get("dates", 0) >= 2 * block
                    and item.get("lower", -1) > 0 and item.get("confidence", 0) >= .95, {"window": name, "block": block})
            ref, alt = recomputed["reference"]["metrics"], recomputed["full"]["metrics"]
            state.check("independent_fee_net_performance_gate", alt["total_return"] > ref["total_return"]
                and alt["max_drawdown"] <= ref["max_drawdown"] + min(.02, config.get("drawdown_tolerance", .02))
                and alt["completed_round_trips"] >= max(50, config.get("development_min_round_trips_per_window", 50))
                and ref["trade_expectancy"] is not None and alt["trade_expectancy"] is not None
                and alt["trade_expectancy"] > ref["trade_expectancy"], name)
        results.append({"name": name, "start": start, "end": end, "sessions": len(dates), "arms": sorted(recomputed)})
    return {"path": str(path), "sha256": file_hash(path), "cost_multiplier": full.get("cost_multiplier"), "windows": results}


def audit_high_return_run(model_run_id, *, as_of=None):
    """Return a read-only audit; pending future qualification never means pass."""
    state = _Audit(); now = signal_time(as_of) if as_of is not None else local_now()
    root = identity_path(ARTIFACT_RUNS_ROOT, model_run_id)
    report = state.attempt("high_return_report", lambda: _read(root / "reports/high_return.json"))
    models, evaluations, freeze, forward_evidence, l1_alignment = {}, [], None, None, None
    if report:
        state.check("independent_family_identity", report.get("run_id") == model_run_id
            and report.get("model_family") == "high_return" and report.get("publication_allowed") is False)
        metadata = state.attempt("run_metadata", lambda: _read(root / "metadata.json")) or {}
        snapshot = report.get("source_snapshot", {}); records = report.get("dataset_records", [])
        symbols = [r["symbol"] for r in records]
        state.check("frozen_universe_identity", len(set(symbols)) == len(symbols) and bool(symbols)
            and set(metadata.get("stocks", [])) == set(symbols))
        state.check("source_snapshot_hash", snapshot.get("snapshot_hash") == stable_hash({k:v for k,v in snapshot.items()
            if k not in {"snapshot_hash", "calendar_run_id"}}))
        for key in ("report", "calendar"):
            state.attempt("source_" + key + "_hash", lambda key=key: _require_hash(
                _inside(snapshot[key + "_path"], ARTIFACT_RUNS_ROOT), snapshot[key + "_sha256"]))
        calendar_data = state.attempt("source_verified_calendar", lambda: _read(snapshot["calendar_path"])) or {}
        calendar = calendar_data.get("dates", [])
        state.check("verified_calendar_identity", bool(calendar) and calendar == sorted(set(calendar))
            and calendar_data.get("verified") is True and snapshot.get("calendar_hash") == stable_hash(calendar))
        for record in records:
            state.attempt("source_feature_hash", lambda record=record: _require_hash(_inside(record["path"], ARTIFACT_RUNS_ROOT), record["sha256"]))
        models = state.attempt("all_model_heads", lambda: _audit_models(root, report, calendar, state, now)) or {}
        freeze = state.attempt("complete_protocol_freeze", lambda: _audit_freeze(root, report, calendar, models, state, now))
        state.attempt("episode_artifact_contracts", lambda: _audit_episode_artifacts(root, report, calendar, state))
        if report.get('config', {}).get('ma_entry_alignment') == 'signal_pinned_learned_exit_v2':
            from my_strategy.services.czsc_high_return_ma_alignment_audit import audit_ma_alignment
            l1_alignment = state.attempt('l1_signal_pinned_target_dependency_audit',
                lambda: audit_ma_alignment(root, report, calendar))
            if l1_alignment is not None:
                for error in l1_alignment.get('errors', []):
                    state.check('l1_alignment_' + error['check'], False, error)
                state.check('l1_signal_pinned_dependency_original_stage_masks',
                    l1_alignment.get('status') == 'pass',
                    {'stage_counts': l1_alignment.get('stage_counts'),
                     'fold_mature_inputs': l1_alignment.get('fold_mature_inputs'),
                     'numeric_preprocess_or_platt_refit': False})
        requested = {s for s in symbols if s.startswith("60") and s.endswith(".SH") or s.startswith("00") and s.endswith(".SZ")}
        record_map = {r["symbol"]: r for r in records}
        evaluation = report.get("evaluation")
        if evaluation:
            result = state.attempt("development_ledger_evaluation", lambda: _audit_evaluation(root, evaluation, calendar,
                report["config"], requested, state, as_of=now, records=record_map,
                bundles_hash=stable_hash(report.get("model_bundles", [])), source_snapshot_hash=snapshot.get("snapshot_hash")))
            if result: evaluations.append(result)
        else:
            state.wait("development_actual_ledger_evaluation")
        stress = report.get("cost_stress", [])
        if {1.5, 2.} - {float(s.get("cost_multiplier", 0)) for s in stress}:
            state.wait("complete_cost_pressure_replays", "both frozen-policy 1.5x and 2x Broker replays are required")
        for item in stress:
            result = state.attempt("cost_pressure_ledger_evaluation", lambda item=item: _audit_evaluation(root, item,
                calendar, report["config"], requested, state, as_of=now, records=record_map,
                bundles_hash=stable_hash(report.get("model_bundles", [])), source_snapshot_hash=snapshot.get("snapshot_hash")))
            if result: evaluations.append(result)
        forward_evidence = state.attempt("independent_forward_registry", lambda: _audit_forward_registry(root, report, freeze, state, now))
        for name in ("historical_universe", "price_basis_total_return", "historical_st"):
            # These data are absent in the current raw-source contract. A bool
            # in the model report does not certify a new formal data adapter.
            state.block("formal_" + name + "_evidence", "no audited point-in-time identity/price-basis protocol is implemented for this family")
        if snapshot.get("full_source_universe") is not True:
            state.block("formal_complete_source_universe", "training selected only part of the frozen source universe")
    passed = not state.errors and not state.pending and not state.blocks
    return {"version": AUDIT_VERSION, "model_run_id": model_run_id, "audited_at": local_now().isoformat(),
        "as_of": now.isoformat(), "passed": passed, "artifact_integrity_passed": not state.errors,
        "error_count": len(state.errors), "errors": state.errors, "checks": state.checks,
        "pending_checks": state.pending, "formal_blocks": state.blocks, "models": sorted(models),
        "evaluation": evaluations, "forward_evidence": forward_evidence, "l1_ma_target_dependency": l1_alignment,
        "publication_allowed": False, "alias_updated": False,
        "qualification": "shadow_pending_independent_forward_and_formal_data_audit"}
