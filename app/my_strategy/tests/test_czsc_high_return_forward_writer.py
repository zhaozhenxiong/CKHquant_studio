"""Append-only forward writer with real Broker and isolated SQLite registry."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime
import json
from pathlib import Path
from threading import Barrier
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from my_strategy.core import paths
from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_high_return_forward as writer
from my_strategy.services import czsc_high_return_evaluation as evaluation
from my_strategy.services import czsc_high_return_research as research
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_high_return_forward_protocol import forward_protocol
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_research_models import file_hash
from my_strategy.storage.czsc_results import ResultStore


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(writer, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(research, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(writer, "PROCESSED_DATA_ROOT", tmp_path / "processed")
    store = ResultStore(db_path=tmp_path / "processed/czsc/results.db", runs_root=runs)
    monkeypatch.setattr(writer, "ResultStore", lambda: store)
    calendar = pd.bdate_range("2024-01-02", periods=100).strftime("%Y-%m-%d").tolist()
    official = runs / "verified-calendar/reports/calendar.json"
    save(official, {"verified": True, "source": "isolated_verified_test_schedule", "dates": calendar})
    close = pd.Series(10., index=range(100))
    close.loc[80:] = 10.4
    market = pd.DataFrame({"date": calendar, "symbol": "000001.SZ", "open": close,
        "high": close * 1.02, "low": close * .98, "close": close, "volume": 1000000.,
        "amount": close * 1000000, "source": "tushare", "has_trade_price": 1})

    def source(name, offset):
        location = runs / name
        feature = location / "dataset/stock.parquet"
        feature.parent.mkdir(parents=True, exist_ok=True)
        market.iloc[:offset + 1].to_parquet(feature, index=False)
        bound_calendar = location / "data_snapshots/calendar.json"
        save(bound_calendar, {"verified": True, "dates": calendar})
        record = {"symbol": "000001.SZ", "path": str(feature), "sha256": file_hash(feature),
            "data_end": calendar[offset], "data_version": name}
        report = {"run_id": name, "dataset_records": [record], "data_end": calendar[offset],
            "calendar": {"path": str(bound_calendar), "file_sha256": file_hash(bound_calendar),
                "run_id": "verified-calendar"}}
        save(location / "reports/czsc.json", report)
        return record, report

    original, _ = source("original-source", 69)
    _, _, snapshot, _ = research._source("original-source")
    cfg = research.research_config({"version": "czsc_high_return_v1", "feature_start": calendar[0], "strategy": strategy_config(),
        "bootstrap_repetitions": 5, "initial_cash": 100000})
    root = runs / "model"
    freeze = {"actual_completed_at": calendar[69] + "T20:00:00+08:00", "independent_start": calendar[70],
        "model_bundles_hash": stable_hash([]), "config_hash": stable_hash(cfg),
        "source_snapshot_hash": snapshot["snapshot_hash"], "forward_protocol": forward_protocol(cfg),
        "minimum_technical_sessions": 480, "minimum_windows": 3,
        "calendar_hash": stable_hash(calendar),
        "code_hashes": {str(path): file_hash(path) for path in research._code_files()}}
    freeze["freeze_hash"] = stable_hash(freeze)
    model = {"run_id": root.name, "model_family": "high_return", "status": "completed",
        "dataset_records": [original], "source_snapshot": snapshot, "config": cfg,
        "calendar": {"dates": calendar}, "model_bundles": [], "data_end": calendar[69],
        "protocol_freeze": freeze, "publication_allowed": False}
    save(root / "reports/high_return.json", model)
    save(root / "validation/protocol_freeze.json", freeze)
    source("new-source-90", 90)
    source("new-source-95", 95)
    source("new-source-98", 98)
    replays = []

    def replay(frame, verified, **kwargs):
        cfg = kwargs["config"]
        decisions = [{"date": day, "action": "HOLD", "target_weight": 0., "entry_plan": None,
            "hold_current_position": True} for day in frame.date]
        def plan(signal, name, floor=9.51):
            return {"plan_id": name, "symbol": "000001.SZ", "signal_date": signal,
                "available_at": signal + "T15:00:00+08:00", "policy": "risk", "status": "active",
                "entry_allowed": True, "valid_for_sessions": 1, "stop_price": 9.5,
                "price_floor": floor, "price_ceiling": 11., "risk_fraction": .01,
                "target_weight": .25, "reference_price": 10.}
        for row in decisions:
            if row["date"] == calendar[70]:
                row.update(action="BUY", target_weight=.25, entry_plan=plan(calendar[70], "rejected", 10.5), hold_current_position=False)
            elif row["date"] == calendar[74]:
                row.update(action="BUY", target_weight=.25, entry_plan=plan(calendar[74], "actual"), hold_current_position=False)
            elif row["date"] == calendar[84]:
                row.update(action="SELL", hold_current_position=False)
        account = execute_decisions("000001.SZ", frame, decisions, kwargs["start"], 100000., cfg["strategy"],
            "writer-test", market_dates=verified, verified_sources=VERIFIED_SOURCES,
            entry_policy="risk", exit_policy=lambda state: None)
        account.update(candidate_points=[{"date": calendar[74], "candidate_eligible": True}], policy_source="union")
        replays.append({"arm": kwargs["arm"], "start": kwargs["start"], "end": kwargs["end"],
            "cost_multiplier": cfg.get("_cost_multiplier", 1.), "account": account})
        return account
    monkeypatch.setattr(evaluation, "_replay", replay)

    def persist_model():
        model["protocol_freeze"]["freeze_hash"] = stable_hash({k: v for k, v in model["protocol_freeze"].items() if k != "freeze_hash"})
        save(root / "reports/high_return.json", model)
        save(root / "validation/protocol_freeze.json", model["protocol_freeze"])

    return {"runs": runs, "root": root, "model": model, "calendar": calendar, "source": source,
        "store": store, "replays": replays, "persist_model": persist_model, "market": market}


def build(ws, source="new-source-90", **kwargs):
    return writer.build_forward_evidence("model", source, cpu_workers=1, **kwargs)


def bindings(report, multiplier=1.):
    compact = report["continuous_evaluation"] if multiplier == 1 else next(
        value for value in report["cost_stress"] if value["cost_multiplier"] == multiplier)
    value = json.loads((Path(compact["output_root"]) / "evaluation.json").read_text())
    return next(iter(value["windows"].values()))["ledger_bindings"]


def test_first_append_uses_real_accounts_and_leaves_training_artifacts_unchanged(workspace):
    ws = workspace
    original = {p: file_hash(p) for p in ws["root"].rglob("*") if p.is_file()}
    result = build(ws)
    assert original == {p: file_hash(p) for p in ws["root"].rglob("*") if p.is_file()}
    assert result["previous_run_id"] is None and result["append_parent_report_path"] is None
    assert result["model_run_id"] == "model" and not result["publication_allowed"]
    assert result["source_snapshot_hash"] != ws["model"]["source_snapshot"]["snapshot_hash"]
    assert result["original_training_source_hash"] == ws["model"]["source_snapshot"]["snapshot_hash"]
    assert len(ws["replays"]) == 6 and all(r["start"] == ws["calendar"][70] for r in ws["replays"])
    for replay in ws["replays"]:
        assert len(replay["account"]["ledger"]) == 2 and len(replay["account"]["rejections"]) >= 1
        assert replay["account"]["ledger"][0]["date"] == ws["calendar"][75]
        assert replay["account"]["ledger"][1]["date"] == ws["calendar"][85]
    assert len(result["forward_windows"]) == 3
    assert all(w["statistics"]["status"] == "pending" for w in result["forward_windows"])
    assert not result["technical_statistics_qualified"]
    status = writer.forward_status("model")
    assert status["append_registry_count"] == 1 and status["mature_technical_windows"] == 0


def test_append_replays_one_continuous_interval_and_verifies_all_real_prefixes(workspace):
    ws = workspace
    first = build(ws)
    first_path = ws["runs"] / first["run_id"] / "reports/forward.json"
    original_hash = file_hash(first_path)
    latest = build(ws, "new-source-95", previous_run_id=first["run_id"])
    assert latest["previous_run_id"] == first["run_id"]
    assert latest["append_parent_report_sha256"] == original_hash == file_hash(first_path)
    assert latest["prefix_audit"]["prefix_through"] == ws["calendar"][90]
    assert latest["prefix_audit"]["files_checked"] == 18
    assert latest["prefix_audit"]["unchanged_verified"]
    assert len(ws["store"].list(writer.KIND)) == 2
    assert len(ws["replays"]) == 12
    for cost in (1., 1.5, 2.):
        before, after = bindings(first, cost), bindings(latest, cost)
        for arm in ("reference", "full"):
            for filename in ("daily", "ledger", "rejections"):
                left = pd.read_csv(Path(before[arm][0]["path"]) / (filename + ".csv"))
                right = pd.read_csv(Path(after[arm][0]["path"]) / (filename + ".csv"))
                right = right.loc[right.date <= ws["calendar"][90]].reset_index(drop=True)
                pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    assert writer.forward_status("model")["latest_forward_run_id"] == latest["run_id"]


def test_missing_parent_older_tip_and_nonincreasing_source_cannot_create_branch(workspace):
    ws = workspace
    with pytest.raises(ValueError, match="parent.*not registered"):
        build(ws, previous_run_id="absent")
    first = build(ws)
    with pytest.raises(ValueError, match="single current.*tip"):
        build(ws, "new-source-95")
    with pytest.raises(ValueError, match="increase observed end"):
        build(ws, previous_run_id=first["run_id"])
    latest = build(ws, "new-source-95", previous_run_id=first["run_id"])
    with pytest.raises(ValueError, match="single current.*tip"):
        build(ws, "new-source-98", previous_run_id=first["run_id"])
    assert len(ws["store"].list(writer.KIND)) == 2
    assert writer.forward_status("model")["latest_forward_run_id"] == latest["run_id"]


def test_old_snapshot_or_request_past_immutable_source_end_is_rejected(workspace):
    with pytest.raises(ValueError, match="new verified data|pending"):
        build(workspace, "original-source")
    with pytest.raises(ValueError, match="frozen source ends"):
        build(workspace, end=workspace["calendar"][95])
    assert workspace["store"].list(writer.KIND) == []


@pytest.mark.parametrize("clock_hour", [10, 20])
def test_future_dates_or_same_day_unobserved_close_are_rejected(workspace, monkeypatch, clock_hour):
    day = workspace["calendar"][90] if clock_hour == 10 else workspace["calendar"][89]
    now = datetime.fromisoformat(day + f"T{clock_hour}:00:00").replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr(writer, "local_now", lambda: now)
    with pytest.raises(ValueError, match="future|observed"):
        build(workspace)
    assert workspace["store"].list(writer.KIND) == []


def test_resigned_old_causal_input_prefix_is_rejected_before_replay(workspace):
    ws = workspace
    location = ws["runs"] / "new-source-90"
    source_report = location / "reports/czsc.json"
    report = json.loads(source_report.read_text())
    path = Path(report["dataset_records"][0]["path"])
    frame = pd.read_parquet(path)
    frame.loc[0, "close"] += .1
    frame.to_parquet(path, index=False)
    report["dataset_records"][0]["sha256"] = file_hash(path)
    save(source_report, report)
    with pytest.raises(ValueError, match="causal input prefix"):
        build(ws)
    assert ws["replays"] == [] and ws["store"].list(writer.KIND) == []


def test_feature_content_hash_and_frozen_execution_code_hash_are_enforced(workspace):
    ws = workspace
    path = ws["runs"] / "new-source-90/dataset/stock.parquet"
    frame = pd.read_parquet(path); frame.loc[0, "volume"] += 1
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="feature hash"):
        build(ws)
    key = next(iter(ws["model"]["protocol_freeze"]["code_hashes"]))
    ws["model"]["protocol_freeze"]["code_hashes"][key] = "0" * 64
    ws["persist_model"]()
    with pytest.raises(ValueError, match="execution.*code changed"):
        build(ws)


@pytest.mark.parametrize("filename,column", [("daily", "cash"), ("ledger", "cash_flow"), ("rejections", "reason")])
def test_changed_parent_real_daily_ledger_or_rejection_prefix_cannot_be_appended(workspace, filename, column):
    ws = workspace
    first = build(ws)
    bound = bindings(first)["full"][0]
    path = Path(bound["path"]) / (filename + ".csv")
    frame = pd.read_csv(path)
    frame.loc[0, column] = "rewritten_rejection" if column == "reason" else frame.loc[0, column] + 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="prefix|hash"):
        build(ws, "new-source-95", previous_run_id=first["run_id"])
    assert len(ws["store"].list(writer.KIND)) == 1


def test_frozen_config_binding_and_fixed_100000_capital_are_required(workspace):
    ws = workspace
    ws["model"]["config"]["initial_cash"] = 20000
    ws["persist_model"]()
    with pytest.raises(ValueError, match="config|cash|100000"):
        build(ws)
    ws["model"]["protocol_freeze"]["config_hash"] = stable_hash(ws["model"]["config"])
    ws["persist_model"]()
    with pytest.raises(ValueError, match="cash|capital|100000"):
        build(ws)
    assert ws["store"].list(writer.KIND) == []


def test_registry_and_immutable_forward_report_must_agree(workspace):
    ws = workspace
    first = build(ws)
    path = ws["runs"] / first["run_id"] / "reports/forward.json"
    report = json.loads(path.read_text()); report["observed_end"] = ws["calendar"][95]
    save(path, report)
    with pytest.raises(ValueError, match="registry/report changed"):
        writer.forward_status("model")


def test_pending_status_without_forward_data_has_all_three_registered_plans(workspace):
    status = writer.forward_status("model")
    assert status["status"] == "pending_future_data" and status["append_registry_count"] == 0
    assert status["mature_technical_windows"] == 0 and not status["publication_allowed"]
    assert len(status["forward_windows"]) == 3 and not any(w["complete"] for w in status["forward_windows"])


def test_unlimited_readonly_registry_finds_old_model_after_more_than_200_other_runs(workspace):
    ws = workspace
    first = build(ws)
    for index in range(205):
        report = {"run_id": f"other-model-forward-{index}", "model_run_id": "other-model", "status": "completed_shadow_evidence"}
        save(ws["runs"] / report["run_id"] / "reports/forward.json", report)
        ws["store"].save(writer.KIND, report)
    before = file_hash(ws["store"].db_path)
    status = writer.forward_status("model")
    assert status["append_registry_count"] == 1 and status["latest_forward_run_id"] == first["run_id"]
    assert file_hash(ws["store"].db_path) == before
    latest = build(ws, "new-source-95", previous_run_id=first["run_id"])
    assert latest["previous_run_id"] == first["run_id"]


@pytest.mark.parametrize("corruption", ["missing_parent", "disconnected_self_cycle"])
def test_single_tip_with_missing_parent_or_disconnected_cycle_is_not_a_valid_chain(workspace, corruption):
    ws = workspace
    first = build(ws)
    broken = copy.deepcopy(first)
    if corruption == "missing_parent":
        broken["previous_run_id"] = "absent-parent"
        broken["append_parent_report_path"] = str(ws["runs"] / "absent-parent/reports/forward.json")
    else:
        broken["run_id"] = "isolated-loop"
        broken["previous_run_id"] = broken["run_id"]
        broken["append_parent_report_path"] = str(ws["runs"] / "isolated-loop/reports/forward.json")
    save(ws["runs"] / broken["run_id"] / "reports/forward.json", broken)
    ws["store"].save(writer.KIND, broken)
    with pytest.raises(ValueError, match="chain|parent|cycle|registered|registry|unlinked|tip"):
        writer.forward_status("model")
    with pytest.raises(ValueError, match="chain|parent|cycle|registered|registry|unlinked|tip"):
        build(ws, "new-source-95", previous_run_id=first["run_id"])


def test_whole_economic_trip_closes_in_window_while_fill_fees_use_only_window_dates(workspace):
    ws = workspace
    first = build(ws)
    bound = bindings(first)["full"]
    selected = ws["calendar"][80:91]
    result = writer.window_trade_metrics(bound, selected)
    assert result["completed_round_trips"] == result["trade_count"] == 1
    ledger = pd.read_csv(Path(bound[0]["path"]) / "ledger.csv")
    episode = pd.read_csv(Path(bound[0]["path"]) / "episodes.csv").iloc[0]
    assert result["fees"] == pytest.approx(ledger.loc[ledger.action == "SELL", "fee"].sum())
    assert result["closed_round_trip_expectancy"] == pytest.approx(episode.net_return)
    assert episode.entry_date < selected[0] < episode.exit_date


def test_closed_unavailable_episode_cannot_be_dropped_or_fabricated_negative(workspace):
    ws = workspace
    first = build(ws)
    bound = bindings(first)["full"]
    path = Path(bound[0]["path"]) / "episodes.csv"
    episodes = pd.read_csv(path); episodes.loc[0, "net_return"] = float("nan")
    episodes.to_csv(path, index=False)
    bound[0]["episodes_sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="finite|economic returns"):
        writer.window_trade_metrics(bound, ws["calendar"][70:91])


def test_real_feature_snapshot_append_builds_only_causal_x_without_training(workspace, monkeypatch):
    from my_strategy.adapters import czsc_adapter
    from my_strategy.services import czsc_high_return_models
    ws = workspace
    def bars(symbol, *, end=None, **kwargs):
        result = ws["market"].loc[ws["market"].date <= end].copy()
        # Non-price raw metadata must not expand model input or leak labels.
        result["future_redundant_metadata"] = range(len(result))
        result["label_available_at"] = "2099-12-31"
        result.attrs["data_version"] = "isolated-real-feature-test"
        return result
    monkeypatch.setattr(czsc_adapter, "load_bars", bars)
    monkeypatch.setattr(czsc_adapter, "latest_market_date", lambda end=None: end or ws["calendar"][95])
    def forbidden_training(*args, **kwargs):
        raise AssertionError("forward source preparation cannot train models")
    monkeypatch.setattr(czsc_high_return_models, "train_head", forbidden_training)
    original_dir = ws["runs"] / "original-source/dataset"
    record = writer._source_worker(("000001.SZ", ws["calendar"][69], ws["calendar"], str(original_dir)))
    report_path = ws["runs"] / "original-source/reports/czsc.json"
    original = json.loads(report_path.read_text()); original["dataset_records"] = [record]
    save(report_path, original)
    _, _, snapshot, _ = research._source("original-source")
    ws["model"]["dataset_records"] = [record]
    ws["model"]["source_snapshot"] = snapshot
    ws["model"]["protocol_freeze"]["source_snapshot_hash"] = snapshot["snapshot_hash"]
    ws["persist_model"]()
    model_hash = file_hash(ws["root"] / "reports/high_return.json")
    result = writer.build_forward_source_snapshot("model", end=ws["calendar"][95], cpu_workers=1)
    assert result["data_end"] == ws["calendar"][95] and result["coverage"]["failed"] == 0
    assert result["outcome_labels_generated"] is False and result["training_performed"] is False
    assert result["same_frozen_universe"] and result["historical_causal_prefix_unchanged"]
    frame = pd.read_parquet(result["dataset_records"][0]["path"])
    assert len(frame) == 96 and set(frame.date) == set(ws["calendar"][:96])
    assert not set(writer.OUTCOMES) & set(frame.columns)
    assert "future_redundant_metadata" not in frame and "label_available_at" not in frame
    assert file_hash(ws["root"] / "reports/high_return.json") == model_hash
    evidence = build(ws, result["run_id"])
    assert evidence["observed_end"] == ws["calendar"][95] and len(ws["replays"]) == 6


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_worker_counts_are_strict_positive_integers(workspace, workers):
    with pytest.raises(ValueError, match="workers"):
        writer.build_forward_evidence("model", "new-source-90", cpu_workers=workers)


def test_runtime_code_drift_before_append_publication_is_rejected(workspace, monkeypatch):
    ws = workspace
    original_model = writer._model
    checks = 0
    def check_model(run_id):
        nonlocal checks
        checks += 1
        if checks == 2:
            raise ValueError("frozen model execution/feature code changed")
        return original_model(run_id)
    monkeypatch.setattr(writer, "_model", check_model)
    with pytest.raises(ValueError, match="code changed"):
        build(ws)
    assert checks == 2 and len(ws["replays"]) == 6
    assert ws["store"].list(writer.KIND) == []


def test_competing_real_replays_from_same_parent_only_one_can_commit(workspace, monkeypatch):
    ws = workspace
    parent = build(ws)
    parent_path = ws["runs"] / parent["run_id"] / "reports/forward.json"
    parent_hash = file_hash(parent_path)
    register = writer._register_forward
    monkeypatch.setattr(writer, "_register_forward", lambda report, freeze: report)
    # Two jobs finish their actual Broker replays against the same current tip.
    left = build(ws, "new-source-95", previous_run_id=parent["run_id"])
    right = build(ws, "new-source-98", previous_run_id=parent["run_id"])
    assert len(ws["replays"]) == 18 and len(writer._registry("model")) == 1
    monkeypatch.setattr(writer, "_register_forward", register)
    barrier = Barrier(2)
    def commit(report):
        barrier.wait(timeout=5)
        try:
            return "committed", register(report, ws["model"]["protocol_freeze"])
        except ValueError as exc:
            return "rejected", str(exc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(commit, [left, right]))
    assert sorted(status for status, _ in results) == ["committed", "rejected"]
    winner = next(value for status, value in results if status == "committed")
    reason = next(value for status, value in results if status == "rejected")
    assert "committed first" in reason
    assert len(writer._registry("model")) == 2
    assert writer.forward_status("model")["latest_forward_run_id"] == winner["run_id"]
    assert file_hash(parent_path) == parent_hash
    loser = right if winner["run_id"] == left["run_id"] else left
    assert not (ws["runs"] / loser["run_id"] / "reports/czsc.json").exists()


@pytest.mark.parametrize("mutation", ["saved_freeze", "source_hash", "missing_code", "future_freeze", "skip_first_session"])
def test_actual_freeze_file_identity_clock_and_complete_code_coverage_are_enforced(workspace, mutation):
    ws = workspace
    freeze = ws["model"]["protocol_freeze"]
    if mutation == "saved_freeze":
        save(ws["root"] / "validation/protocol_freeze.json", {"freeze_hash": "changed"})
    else:
        if mutation == "source_hash": freeze["source_snapshot_hash"] = "changed"
        elif mutation == "missing_code": freeze["code_hashes"].pop(next(iter(freeze["code_hashes"])))
        elif mutation == "future_freeze": freeze["actual_completed_at"] = "2099-12-31T20:00:00+08:00"
        elif mutation == "skip_first_session": freeze["independent_start"] = ws["calendar"][71]
        ws["persist_model"]()
    with pytest.raises(ValueError):
        build(ws)
    assert ws["replays"] == [] and ws["store"].list(writer.KIND) == []
