"""Actual-account append evidence cannot certify partial or rewritten history."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_high_return_audit as audit
from my_strategy.services import czsc_high_return_evaluation as evaluation
from my_strategy.services.czsc_high_return_forward_protocol import (forward_protocol, forward_windows,
    slice_continuous_variant, paired_forward_statistics)
from my_strategy.services.czsc_research_models import file_hash
from my_strategy.tests.test_czsc_high_return_evaluation import inputs, actual_broker_fake


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def evidence(tmp_path, monkeypatch, request):
    runs = tmp_path / "runs"; source = runs / "source"; source.mkdir(parents=True)
    _, dates, cfg = inputs(source, periods=getattr(request, "param", 100))
    raw = pd.read_parquet(source / "stock.parquet")
    monkeypatch.setattr(audit, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(evaluation, "_replay", actual_broker_fake([]))
    root = runs / "model"
    calendar_path = source / "calendar.json"; save(calendar_path, {"verified": True, "dates": dates})

    def source_at(name, offset):
        location = runs / name; location.mkdir(exist_ok=True)
        feature = location / "stock.parquet"; raw.iloc[:offset + 1].to_parquet(feature, index=False)
        record = {"symbol": "000001.SZ", "path": str(feature), "sha256": file_hash(feature),
            "data_end": dates[offset], "data_version": "fixture"}
        source_report = location / "source.json"; save(source_report, {"dataset_records": [record]})
        snapshot = {"report_path": str(source_report), "report_sha256": file_hash(source_report),
            "calendar_path": str(calendar_path), "calendar_sha256": file_hash(calendar_path),
            "calendar_hash": stable_hash(dates), "records_hash": stable_hash([record]), "full_source_universe": True}
        snapshot["snapshot_hash"] = stable_hash(snapshot)
        return record, snapshot

    original, snapshot = source_at("original", 69)
    cfg["source_snapshot_hash"] = snapshot["snapshot_hash"]
    protocol = forward_protocol(cfg)
    freeze = {"actual_completed_at": dates[69] + "T20:00:00+08:00", "independent_start": dates[70],
        "model_bundles_hash": stable_hash([]), "config_hash": stable_hash(cfg),
        "source_snapshot_hash": snapshot["snapshot_hash"], "forward_protocol": protocol,
        "minimum_technical_sessions": 480, "minimum_windows": 3, "code_hashes": {}}
    freeze["freeze_hash"] = stable_hash(freeze)
    training = {"run_id": root.name, "model_family": "high_return", "publication_allowed": False,
        "dataset_records": [original], "source_snapshot": snapshot, "config": cfg, "model_bundles": [],
        "data_end": dates[69], "protocol_freeze": freeze, "independent_window": {"start": dates[70]}}
    save(root / "reports/high_return.json", training)
    save(root / "validation/protocol_freeze.json", freeze)
    save(root / "metadata.json", {"stocks": [original["symbol"]]})
    registered = []
    monkeypatch.setattr(audit, "_forward_registry_records", lambda _: copy.deepcopy(registered))

    def append(name="append-1", offset=90, parent=None):
        record, new_snapshot = source_at(name + "-source", offset)
        run_root = runs / name
        results = []
        for multiplier in (1., 1.5, 2.):
            result = evaluation.evaluate_bundles([record], dates, model_root=root, bundles=[],
                config={**cfg, "source_snapshot_hash": new_snapshot["snapshot_hash"]},
                windows=[{"name": "continuous", "start": dates[70], "end": dates[offset]}],
                arms=["reference", "full"], cpu_workers=1, cost_multiplier=multiplier,
                output_root=run_root / ("cost-" + str(multiplier)))
            result["evaluation_sha256"] = file_hash(Path(result["output_root"]) / "evaluation.json")
            results.append(result)
        continuous = results[0]; window = continuous["windows"]["continuous"]
        all_dates = [r["date"] for r in continuous["variants"]["reference"]["daily"]]
        windows = forward_windows(dates, dates[70], dates[offset], cfg)
        for planned in windows:
            planned["variants"] = {}
            for arm in ("reference", "full"):
                variant = slice_continuous_variant(continuous["variants"][arm], planned["dates"], all_dates)
                variant["metrics"].update(audit._forward_window_trade_metrics(window["ledger_bindings"][arm], planned["dates"]))
                planned["variants"][arm] = variant
            planned["statistics"] = paired_forward_statistics(planned["variants"]["reference"], planned["variants"]["full"], cfg)
            planned["economic_gate"] = audit._forward_economic_values(planned["variants"]["reference"], planned["variants"]["full"], cfg)
            planned["cost_stress"] = []
            for pressure in results[1:]:
                stress_variants = {}
                pressure_window = pressure["windows"]["continuous"]
                for arm in ("reference", "full"):
                    variant = slice_continuous_variant(pressure["variants"][arm], planned["dates"], all_dates)
                    variant["metrics"].update(audit._forward_window_trade_metrics(pressure_window["ledger_bindings"][arm], planned["dates"]))
                    stress_variants[arm] = variant
                planned["cost_stress"].append({"cost_multiplier": pressure["cost_multiplier"], "variants": stress_variants,
                    "economic_gate": audit._forward_economic_values(stress_variants["reference"], stress_variants["full"], cfg)})
        bound_calendar = run_root / "calendar.json"; save(bound_calendar, {"verified": True, "dates": dates})
        result = {"run_id": name, "model_run_id": root.name, "model_family": "high_return",
            "status": "completed_shadow_evidence", "freeze_hash": freeze["freeze_hash"],
            "model_bundles_hash": freeze["model_bundles_hash"], "config_hash": freeze["config_hash"],
            "calendar": {"path": str(bound_calendar), "sha256": file_hash(bound_calendar), "dates_hash": stable_hash(dates)},
            "new_dataset_records": [record], "source_snapshot": new_snapshot,
            "source_snapshot_hash": new_snapshot["snapshot_hash"], "original_training_source_hash": snapshot["snapshot_hash"],
            "data_end": dates[offset], "observed_end": dates[offset], "independent_start": dates[70],
            "created_at": dates[offset] + "T20:00:00+08:00", "actual_completed_at": dates[offset] + "T20:01:00+08:00",
            "previous_run_id": parent["run_id"] if parent else None,
            "append_parent_report_path": str(runs / parent["run_id"] / "reports/forward.json") if parent else None,
            "append_parent_report_sha256": file_hash(runs / parent["run_id"] / "reports/forward.json") if parent else None,
            "prefix_audit": {"unchanged_verified": True, "same_calendar_prefix": True,
                "ledger_and_daily_prefix_verified": True}, "continuous_evaluation": continuous,
            "cost_stress": results[1:], "forward_protocol": protocol, "forward_windows": windows,
            "publication_allowed": False, "model_gate": False, "technical_statistics_qualified": False}
        registered.append(result); persist(result)
        return result

    def persist(value):
        for filename in ("forward.json", "czsc.json"):
            save(runs / value["run_id"] / "reports" / filename, value)

    return root, training, freeze, dates, append, persist, registered


def inspect(evidence):
    root, training, freeze, *_ = evidence
    state = audit._Audit()
    result = audit._audit_forward_registry(root, training, freeze, state, audit.signal_time("2026-10-10"))
    return result, state


def test_partial_actual_accounts_remain_pending_and_readonly(evidence):
    root, _, _, _, append, _, _ = evidence
    value = append()
    before = {p: file_hash(p) for p in root.parent.rglob("*") if p.is_file()}
    result, state = inspect(evidence)
    assert not state.errors, state.errors
    assert result["completed_windows"] == result["qualified_windows"] == 0
    assert all(w["statistics"]["status"] == "pending" for w in result["windows"])
    assert "three_complete_independent_forward_windows" in {r["rule"] for r in state.pending}
    assert before == {p: file_hash(p) for p in root.parent.rglob("*") if p.is_file()}
    full = audit.audit_high_return_run(root.name, as_of="2026-10-10")
    assert full["artifact_integrity_passed"], full["errors"]
    assert full["forward_evidence"]["latest_run_id"] == value["run_id"]
    assert not full["passed"] and not full["publication_allowed"]


def test_terminal_append_reuses_true_prefix_and_counts_windows_once(evidence):
    _, _, _, _, append, _, _ = evidence
    first = append(); last = append("append-2", 95, first)
    result, state = inspect(evidence)
    assert not state.errors, state.errors
    assert result["chain_run_ids"] == [first["run_id"], last["run_id"]]
    assert len(result["windows"]) == 3
    assert {"forward_daily_prefix_unchanged", "forward_ledger_prefix_unchanged", "forward_rejections_prefix_unchanged"} <= {r["rule"] for r in state.checks}


@pytest.mark.parametrize("field", ["dates", "protocol_hash", "continuous_account_reset", "end_session_offset"])
def test_window_plan_fields_cannot_be_forged(evidence, field):
    _, _, _, _, append, persist, _ = evidence
    value = append(); window = value["forward_windows"][0]
    window[field] = [] if field == "dates" else "forged"
    persist(value)
    _, state = inspect(evidence)
    assert "forward_frozen_window_identity" in {r["rule"] for r in state.errors}


def test_resigned_input_and_cash_prefix_are_independently_rejected(evidence):
    _, _, _, _, append, persist, _ = evidence
    parent = append(); value = append("append-2", 95, parent)
    record = value["new_dataset_records"][0]; path = Path(record["path"])
    frame = pd.read_parquet(path); frame.loc[0, "close"] += .01; frame.to_parquet(path, index=False)
    record["sha256"] = file_hash(path)
    binding = value["continuous_evaluation"]["windows"]["continuous"]["ledger_bindings"]["full"][0]
    path = Path(binding["path"]) / "daily.csv"
    daily = pd.read_csv(path); daily.loc[0, "cash"] += 1; daily.to_csv(path, index=False)
    binding["daily_sha256"] = file_hash(path)
    save(Path(value["continuous_evaluation"]["output_root"]) / "evaluation.json", value["continuous_evaluation"])
    value["continuous_evaluation"]["evaluation_sha256"] = file_hash(Path(value["continuous_evaluation"]["output_root"]) / "evaluation.json")
    persist(value)
    _, state = inspect(evidence)
    assert {"forward_causal_input_prefix_unchanged", "forward_daily_prefix_unchanged",
        "daily_actual_ledger_cash_shares_equity"} <= {r["rule"] for r in state.errors}


def test_unlinked_duplicate_append_and_forged_statistics_fail(evidence):
    _, _, _, _, append, persist, _ = evidence
    append(); value = append("unlinked", 95)
    value["forward_windows"][0]["statistics"]["qualified"] = True; persist(value)
    _, state = inspect(evidence)
    assert {"forward_registry_one_append_tip", "forward_statistics_independent_recomputation"} <= {r["rule"] for r in state.errors}


def test_absent_registry_and_unregistered_protocol_never_pass(evidence):
    _, _, freeze, _, append, _, _ = evidence
    result, state = inspect(evidence)
    assert result is None and not state.errors and state.pending
    append(); freeze.pop("forward_protocol")
    result, state = inspect(evidence)
    assert not state.errors, state.errors
    assert result["qualified_windows"] == 0
    assert "forward_statistics_protocol_preregistered" in {r["rule"] for r in state.pending}


@pytest.mark.parametrize("evidence", [550], indirect=True)
def test_complete_480_calendar_window_without_mature_trades_is_pending(evidence):
    *_, append, _, _ = evidence
    append(offset=549)
    result, state = inspect(evidence)
    assert not state.errors, state.errors
    assert result["completed_windows"] == 1 and result["qualified_windows"] == 0
    assert result["windows"][0]["economic_status"] == "pending"
    assert result["windows"][0]["cost_pressure_status"] == {"1.5": "pending", "2.0": "pending"}
    assert "forward_cost_pressure_mature_round_trips" in {r["rule"] for r in state.pending}
    assert not any(r["rule"] == "forward_fee_net_performance_gate" for r in state.blocks)


def test_same_freeze_parent_hash_and_calendar_date_binding_are_required(evidence):
    _, _, _, _, append, persist, _ = evidence
    parent = append(); child = append("append-2", 95, parent)
    child["calendar"]["dates_hash"] = "forged"; persist(child)
    with pytest.raises(ValueError, match="date identity mismatch"):
        inspect(evidence)
    child["calendar"]["dates_hash"] = parent["calendar"]["dates_hash"]
    child["append_parent_report_sha256"] = "forged"; persist(child)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        inspect(evidence)


def test_mature_economics_need_return_drawdown_and_trade_expectancy():
    protocol = forward_protocol()
    ref = {"metrics": {"completed_round_trips": 50, "period_net_return": .1,
        "max_drawdown": .05, "closed_round_trip_expectancy": .02}}
    alt = copy.deepcopy(ref)
    alt["metrics"].update(period_net_return=.11, closed_round_trip_expectancy=.03)
    assert audit._forward_economic_gate(ref, alt, protocol, {}) == "passed"
    alt["metrics"]["max_drawdown"] = .071
    assert audit._forward_economic_gate(ref, alt, protocol, {}) == "failed"
    alt["metrics"].update(max_drawdown=.06, closed_round_trip_expectancy=.019)
    assert audit._forward_economic_gate(ref, alt, protocol, {}) == "failed"
    alt["metrics"].update(completed_round_trips=49, closed_round_trip_expectancy=None)
    assert audit._forward_economic_gate(ref, alt, protocol, {}) == "pending"
