"""Shared immutable L1 facts, independently executed actual policy accounts."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_high_return_evaluation as evaluation
from my_strategy.services import czsc_high_return_runtime as runtime
from my_strategy.services.czsc_high_return_features import ENTRY_FEATURE_COLUMNS
from my_strategy.services.czsc_high_return_features import EXIT_FEATURE_COLUMNS, L1_MA_ENTRY_COLUMNS
from my_strategy.services.czsc_high_return_models import HeadSession, bind_inputs, schema_hash, split_validation, train_head
from my_strategy.services.czsc_high_return_research import research_config, _selection_score
from my_strategy.services.czsc_trade_episodes import _settings


@pytest.fixture
def market(tmp_path):
    days = pd.bdate_range("2024-01-02", periods=100).strftime("%Y-%m-%d").tolist()
    close = np.full(len(days), 10.)
    close[72:76] = [10.15, 10.3, 10.45, 10.6]
    close[76:80] = 10.6
    close[80:85] = [10.3, 10.1, 9.9, 9.7, 9.6]
    close[85:] = 9.6
    frame = pd.DataFrame({column: .1 for column in ENTRY_FEATURE_COLUMNS}, index=range(len(days)))
    frame = frame.assign(date=days, symbol="000001.SZ", open=close, high=close + .02,
        low=close - .02, close=close, volume=1_000_000., amount=10_000_000.,
        source="tushare", has_trade_price=1, input_eligible=False,
        model_input_eligible=True, structure_input_eligible=False, wyckoff_input_eligible=False,
        ma_alignment=1., ma20_slope5=.01, ma5_ma10_gap=.01, atr14_ratio=.01,
        ret_1=.01, ret_5=.01, close_above_ma20=-1., confirmed_native_bs2="其他",
        confirmed_native_bs3="其他", wyckoff_rule_buy=False, wyckoff_rule_sell=False)
    frame.loc[70:83, "close_above_ma20"] = 1.
    frame.loc[84:, ["close_above_ma20", "ma20_slope5", "ma5_ma10_gap", "ret_5"]] = [-1., -.01, -.01, -.01]
    path = tmp_path / "actual-market.parquet"
    frame.to_parquet(path, index=False)
    records = [{"symbol": "000001.SZ", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    config = research_config({"feature_start": days[0], "bootstrap_repetitions": 5,
        "source_snapshot_hash": "frozen-test-market", "min_validation_round_trips": 1})
    fold = {"name": "held-out-policy", "calibration_end": days[69], "policy_start": days[70], "policy_end": days[95]}
    options = []
    for name, entry, exit in (("take_profit", .6, .55), ("stay_flat", .95, .55), ("wait_for_stop", .6, .95)):
        options.append({"name": name, "fold": fold, "available_at": days[95], "heads": {
            "entry": "models/entry", "exit": "models/exit", "exit_advantage": "models/advantage"},
            "policy": {"entry_threshold": entry, "exit_threshold": exit}})
    return records, days, config, fold, options


def install_real_replay_spies(monkeypatch):
    """Only classifier outputs are controlled; preparation/Broker remain real."""
    original_prepare, original_replay = runtime.prepare_points, runtime.replay_high_return
    prepared, replays, held = [], [], []
    def prepare(*args, **kwargs):
        points = original_prepare(*args, **kwargs)
        prepared.append((points, stable_hash(points)))
        return points
    def replay(*args, **kwargs):
        account = original_replay(*args, **kwargs)
        replays.append({"mode": kwargs["mode"], "option": (kwargs.get("validation_bundle") or {}).get("name"),
            "points": kwargs["prepared_points"], "session": kwargs["session"], "account": account})
        return account
    def predict(session, values, root, bundle, head, **kwargs):
        assert kwargs["validation"] and kwargs["source"] == "union"
        assert bundle["fold"]["calibration_end"] < values["date"] <= bundle["fold"]["policy_end"]
        if bundle["name"] == "injected_failure":
            raise ValueError("injected_option_runtime_failure")
        session.diagnostics["rows"] += 1
        session.diagnostics["batches"] += 1
        if head in {"return", "loss"}: return None, None
        if head == "entry": probability = .8
        else:
            held.append({"option": bundle["name"], "head": head, "values": copy.deepcopy(values)})
            probability = .03 if head == "exit_advantage" else .99 if bundle["name"] == "exit_too_early" else .8 if values["holding_sessions"] >= 5 else .1
        return probability, {"target_contract_hash": stable_hash({"head": head, "source": "frozen-test-market"})}
    monkeypatch.setattr(runtime, "prepare_points", prepare)
    monkeypatch.setattr(runtime, "replay_high_return", replay)
    monkeypatch.setattr(runtime, "_predict", predict)
    return prepared, replays, held


def run_options(market, tmp_path, monkeypatch, *, records=None, options=None):
    source, days, config, fold, choices = market
    prepared, replays, held = install_real_replay_spies(monkeypatch)
    before = copy.deepcopy(choices if options is None else options)
    reports = evaluation.evaluate_policy_options(source if records is None else records, days,
        model_root=tmp_path, ma_bundles=[], options=choices if options is None else options,
        config=config, fold=fold, cpu_workers=1, output_root=tmp_path / "policies")
    assert before == (choices if options is None else options)
    return reports, prepared, replays, held


def test_reference_and_l1_prepare_once_with_separate_real_account_states(market, tmp_path, monkeypatch):
    reports, prepared, replays, held = run_options(market, tmp_path, monkeypatch)
    assert len(prepared) == 1 and sum(r["mode"] == "reference" for r in replays) == 1
    assert len(replays) == 4 and len({id(row["account"]) for row in replays}) == 4
    assert len({id(row["points"]) for row in replays}) == 1
    assert len({id(row["session"]) for row in replays}) == 1
    assert stable_hash(prepared[0][0]) == prepared[0][1]
    assert all(row["account"]["compute_info"]["l1_points_reused"] for row in replays)
    profit, flat, stop = [r["account"] for r in replays if r["mode"] == "full"]
    assert len(profit["ledger"]) == len(stop["ledger"]) == 2 and flat["ledger"] == []
    first = {name: next(r["values"] for r in held if r["option"] == name and r["head"] == "exit")
        for name in ("take_profit", "wait_for_stop")}
    for state in first.values(): assert state["holding_sessions"] == 1.
    assert first["take_profit"]["running_peak_return"] == first["wait_for_stop"]["running_peak_return"]
    assert first["take_profit"]["remaining_fraction"] == first["wait_for_stop"]["remaining_fraction"] == 1.
    references = [next(iter(r["windows"].values()))["ledger_bindings"]["reference"] for r in reports]
    assert references[0] == references[1] == references[2]


def test_options_rank_by_actual_fee_net_accounts_and_record_frozen_contracts(market, tmp_path, monkeypatch):
    records, days, config, fold, options = market
    reports, _, _, _ = run_options(market, tmp_path, monkeypatch)
    returns = [r["variants"]["full"]["metrics"]["net_return"] for r in reports]
    assert returns[0] > returns[1] == 0. > returns[2]
    scores = [_selection_score(report, config) for report in reports]
    assert scores[0] == returns[0] and scores[1] == float("-inf")
    for index, report in enumerate(reports):
        assert report["option"] == index and report["common_reference_reused"]
        assert report["validation_bundle_hash"] == stable_hash(options[index])
        assert report["source_records_hash"] == stable_hash(records)
        assert report["source_snapshot_hash"] == config["source_snapshot_hash"]
        assert report["calendar_hash"] == stable_hash(days)
        assert report["strategy_configuration_hash"] == stable_hash(_settings(config))
        assert report["base_execution_config_hash"] == stable_hash(_settings(config)["execution"])
        assert not report["thresholds_refit"] and not report["publication_allowed"] and not report["model_gate"]
        binding = next(iter(report["windows"].values()))["ledger_bindings"]["full"][0]
        ledger = pd.read_csv(Path(binding["path"]) / "ledger.csv")
        metrics = report["variants"]["full"]["metrics"]
        assert metrics["fees"] == pytest.approx(ledger.fee.sum() if len(ledger) else 0.)
        assert metrics["final_equity"] == pytest.approx(100000 + ledger.cash_flow.sum() if len(ledger) else 100000.)
        assert report["variants"]["full"]["cash_retained_failures"] == []
        serialized = json.loads((Path(report["output_root"]) / f"option_{index:02d}/evaluation.json").read_text())
        assert serialized["validation_bundle_hash"] == report["validation_bundle_hash"]


def test_equal_actual_reference_account_has_no_incremental_policy_qualification(market, tmp_path, monkeypatch):
    reports, _, _, _ = run_options(market, tmp_path, monkeypatch)
    unchanged = reports[2]
    assert unchanged["variants"]["full"]["metrics"]["completed_round_trips"] == 1
    assert unchanged["variants"]["full"]["metrics"]["net_return"] == unchanged["variants"]["reference"]["metrics"]["net_return"]
    assert _selection_score(unchanged, market[2]) == float("-inf")


def test_actual_positive_profit_below_same_reference_is_rejected(market, tmp_path, monkeypatch):
    records, days, config, _, original = market
    frame = pd.read_parquet(records[0]["path"])
    # The frozen reference sells at the rise's end. A model's earlier sell is
    # profitable in cash but earns less from exactly the same initial budget.
    frame.loc[75:, ["close_above_ma20", "ma20_slope5", "ma5_ma10_gap", "ret_5"]] = [-1., -.01, -.01, -.01]
    frame.to_parquet(records[0]["path"], index=False)
    records[0]["sha256"] = hashlib.sha256(Path(records[0]["path"]).read_bytes()).hexdigest()
    early = copy.deepcopy(original[0]); early["name"] = "exit_too_early"
    reports, _, replays, _ = run_options(market, tmp_path, monkeypatch, options=[early])
    report = reports[0]
    actual = report["variants"]["full"]["metrics"]
    reference = report["variants"]["reference"]["metrics"]
    assert actual["completed_round_trips"] == reference["completed_round_trips"] == 1
    assert 0 < actual["net_return"] < reference["net_return"]
    assert actual["initial_cash"] == reference["initial_cash"] == 100000.
    assert _selection_score(report, config) == float("-inf")
    assert [row["account"]["ledger"][-1]["date"] for row in replays] == [days[76], days[72]]


def test_missing_stock_keeps_cash_in_every_option_and_one_common_reference(market, tmp_path, monkeypatch):
    records = copy.deepcopy(market[0])
    records.append({"symbol": "000002.SZ", "path": str(tmp_path / "missing.parquet"), "sha256": "missing"})
    reports, prepared, replays, _ = run_options(market, tmp_path, monkeypatch, records=records)
    assert len(prepared) == 1 and sum(row["mode"] == "reference" for row in replays) == 1
    for report in reports:
        for arm in ("full", "reference"):
            actual = report["variants"][arm]
            assert actual["account_count"] == 2 and actual["metrics"]["initial_cash"] == 200000
            assert [failure["symbol"] for failure in actual["cash_retained_failures"]] == ["000002.SZ"]
            rows = next(iter(report["windows"].values()))["ledger_bindings"][arm]
            failed = next(row for row in rows if row["symbol"] == "000002.SZ")
            daily = pd.read_csv(Path(failed["path"]) / "daily.csv")
            assert daily.cash.eq(100000).all() and daily.equity.eq(100000).all()
            assert actual["metrics"]["final_equity"] == pytest.approx(sum(
                pd.read_csv(Path(row["path"]) / "daily.csv").equity.iloc[-1] for row in rows))
        assert _selection_score(report, market[2]) == float("-inf")


def test_failed_option_retains_cash_without_poisoning_next_option_state(market, tmp_path, monkeypatch):
    options = copy.deepcopy(market[4])
    broken = copy.deepcopy(options[0]); broken["name"] = "injected_failure"
    options.insert(0, broken)
    reports, prepared, replays, held = run_options(market, tmp_path, monkeypatch, options=options)
    broken_account = reports[0]["variants"]["full"]
    assert broken_account["metrics"]["final_equity"] == 100000.
    assert broken_account["metrics"]["trade_count"] == 0
    assert broken_account["cash_retained_failures"][0]["reason"] == "injected_option_runtime_failure"
    assert reports[1]["variants"]["full"]["metrics"]["net_return"] > 0
    assert reports[1]["variants"]["full"]["cash_retained_failures"] == []
    assert len(prepared) == 1 and sum(row["mode"] == "reference" for row in replays) == 1
    assert next(row["values"]["holding_sessions"] for row in held if row["option"] == "take_profit") == 1.


def test_failure_after_episode_extraction_cannot_leave_phantom_closed_trades(market, tmp_path, monkeypatch):
    from my_strategy.services import czsc_dual_research
    original = czsc_dual_research.round_trip_statistics
    def wrong_statistics(ledgers):
        result = original(ledgers)
        if ledgers[0] and ledgers[0][-1]["action"] == "SELL" and ledgers[0][-1]["price"] > 10.5:
            result["completed_round_trips"] = 0  # Inject a failure after successful Broker+episode extraction.
        return result
    monkeypatch.setattr(czsc_dual_research, "round_trip_statistics", wrong_statistics)
    reports, _, _, _ = run_options(market, tmp_path, monkeypatch)
    full = reports[0]["variants"]["full"]
    assert full["cash_retained_failures"][0]["reason"] == "actual ledger roundtrip counts differ"
    assert full["metrics"]["final_equity"] == 100000.
    assert full["metrics"]["trade_count"] == full["metrics"]["completed_round_trips"] == 0
    binding = next(iter(reports[0]["windows"].values()))["ledger_bindings"]["full"][0]
    assert pd.read_csv(Path(binding["path"]) / "episodes.csv").empty
    assert reports[2]["variants"]["full"]["metrics"]["completed_round_trips"] == 1


@pytest.mark.parametrize("workers", [0, -1, 1.5])
def test_policy_option_workers_cannot_silently_fall_back(market, tmp_path, workers):
    records, days, config, fold, options = market
    with pytest.raises(ValueError, match="workers"):
        evaluation.evaluate_policy_options(records, days, model_root=tmp_path, ma_bundles=[],
            options=options, config=config, fold=fold, cpu_workers=workers)


def test_policy_options_preserve_frozen_actual_capital_target(market, tmp_path):
    records, days, config, fold, options = market
    with pytest.raises(ValueError, match="100000"):
        evaluation.evaluate_policy_options(records, days, model_root=tmp_path, ma_bundles=[],
            options=options, config={**config, "initial_cash": 20000.}, fold=fold, cpu_workers=1)


@pytest.fixture(scope="module", params=[("ma_entry", "linear"), ("ma_entry", "tree"),
    ("exit_advantage", "linear"), ("exit_advantage", "tree")])
def real_head(request, tmp_path_factory):
    """Small genuine fits, including Platt and continuous target calibration."""
    head, architecture = request.param
    columns = L1_MA_ENTRY_COLUMNS if head == "ma_entry" else EXIT_FEATURE_COLUMNS
    days = pd.bdate_range("2024-01-02", periods=100).strftime("%Y-%m-%d").tolist()
    ordinal = np.arange(100); label = ((ordinal // 2) % 2).astype(float)
    frame = pd.DataFrame({column: (2 * label - 1) * (1 + index / 100) for index, column in enumerate(columns)})
    frame["date"] = days; frame["symbol"] = "000001.SZ"; frame["input_eligible"] = True
    frame["episode_id"] = [f"actual-episode-{index//2}" for index in ordinal]
    frame["economic_group_id"] = frame.episode_id
    frame["label"] = label if head == "ma_entry" else (2 * label - 1) * .02
    frame["label_available"] = ordinal < 98
    frame["label_end"] = [days[index+2] if index < 98 else None for index in ordinal]
    target = "actual_complete_fee_net_roundtrip_positive" if head == "ma_entry" else "remaining_net_sell_proceeds_exit_next_open_minus_frozen_continue_over_current_market_value"
    contract = {"target": target, "source_snapshot_hash": "frozen-real-fast-path-test",
        "model_head": head, "model_input_schema_hash": schema_hash(columns, head), "label_version": "actual_episode_test_v1"}
    contract["contract_hash"] = stable_hash(contract)
    frame.attrs["label_contract"] = contract
    fold = split_validation({"train_end": days[49], "validation_start": days[50], "validation_end": days[79]}, days)
    path = tmp_path_factory.mktemp("real-single-row-head") / (head + "-" + architecture)
    train_head(frame, columns, path, fold, {"epochs": 3, "batch_size": 16, "min_train_rows": 10,
        "min_calibration_rows": 5}, head=head, architecture=architecture,
        regression=head == "exit_advantage", device="cpu", contract=contract)
    return frame, columns, fold, contract, path, head


def test_actual_single_row_fast_inference_matches_batch_for_nan_and_eligibility(real_head):
    frame, columns, fold, contract, path, head = real_head
    future = frame.loc[frame.date > fold["validation_end"]].head(8).copy()
    future["input_eligible"] = future.input_eligible.astype("boolean")
    future.loc[future.index[0], columns[0]] = np.nan
    future.loc[future.index[1], columns[0]] = np.inf
    future.loc[future.index[2], "input_eligible"] = False
    future.loc[future.index[3], "input_eligible"] = np.nan
    bound = bind_inputs(future, columns, head)
    batch_session = HeadSession()
    batch = batch_session.predict(bound, path, expected_head=head, expected_contract=contract)
    scalar_session = HeadSession()
    individual = np.array([scalar_session.predict_one(row, path, expected_head=head, expected_contract=contract)
        for row in bound.to_dict("records")])
    assert np.allclose(batch, individual, equal_nan=True, atol=2e-6, rtol=2e-6)
    assert np.isfinite(individual[[0, 1, 4, 5, 6, 7]]).all() and np.isnan(individual[[2, 3]]).all()
    assert scalar_session.diagnostics["model_loads"] == 1
    assert scalar_session.diagnostics["rows"] == scalar_session.diagnostics["batches"] == len(bound)
    assert batch_session.diagnostics["batches"] == 1


def test_single_row_fast_path_enforces_same_time_head_and_target_contract(real_head):
    frame, columns, fold, contract, path, head = real_head
    row = frame.loc[frame.date > fold["validation_end"]].iloc[0].to_dict()
    session = HeadSession()
    wrong = {**contract, "source_snapshot_hash": "other-source"}
    with pytest.raises(ValueError, match="contract"):
        session.predict_one(row, path, expected_contract=wrong)
    with pytest.raises(ValueError, match="contract"):
        session.predict_one(row, path)
    with pytest.raises(ValueError, match="head"):
        session.predict_one(row, path, expected_head="exit" if head == "ma_entry" else "ma_entry", expected_contract=contract)
    unavailable = {**row, "date": fold["validation_end"]}
    with pytest.raises(ValueError, match="unavailable"):
        session.predict_one(unavailable, path, expected_contract=contract)
    with pytest.raises(ValueError, match="actually completed"):
        session.predict_one(row, path, expected_contract=contract, usage_mode="production")
    missing = copy.deepcopy(row); missing.pop(columns[0])
    with pytest.raises(ValueError, match="columns"):
        session.predict_one(missing, path, expected_contract=contract)
    policy = {**row, "date": fold["policy_start"]}
    assert np.isfinite(session.predict_one(policy, path, expected_contract=contract, validation=True))
    with pytest.raises(ValueError, match="unavailable"):
        session.predict_one({**row, "date": fold["calibration_end"]}, path, expected_contract=contract, validation=True)


@pytest.mark.parametrize("invalid_date", [None, pd.NaT, "NaT"])
def test_single_row_fast_path_rejects_missing_or_nat_date(real_head, invalid_date):
    frame, columns, _, contract, path, head = real_head
    row = frame.iloc[-1].to_dict(); row["date"] = invalid_date
    with pytest.raises(ValueError, match="date|unavailable"):
        HeadSession().predict_one(row, path, expected_head=head, expected_contract=contract)
