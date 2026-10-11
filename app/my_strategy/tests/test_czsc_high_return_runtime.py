"""Forward checkpoint dates and true Broker prefixes govern interval decisions."""
import copy

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_high_return_runtime as runtime
from my_strategy.services.czsc_high_return_features import ENTRY_FEATURE_COLUMNS, L1_MA_ENTRY_COLUMNS, L1_MA_EXIT_COLUMNS
from my_strategy.services.czsc_high_return_labels import label_config
from my_strategy.services.czsc_high_return_models import schema_hash


def market_frame(n=110):
    dates = pd.bdate_range("2024-01-02", periods=n)
    frame = pd.DataFrame({key: .1 for key in ENTRY_FEATURE_COLUMNS}, index=range(n))
    frame = frame.assign(date=dates, symbol="000001.SZ", open=10., high=10.2, low=9.9, close=10.,
        volume=1_000_000., amount=10_000_000., source="tushare", has_trade_price=1,
        model_input_eligible=True, structure_input_eligible=False, wyckoff_input_eligible=False,
        input_eligible=False, ma_alignment=1., ma20_slope5=.01, ma5_ma10_gap=.01,
        atr14_ratio=.01, ret_1=.01, ret_5=.01, close_above_ma20=-1.,
        confirmed_native_bs2="其他", confirmed_native_bs3="其他", wyckoff_rule_buy=False, wyckoff_rule_sell=False)
    frame.loc[70:, "close_above_ma20"] = 1.
    return frame, dates.strftime("%Y-%m-%d").tolist()


def manifest(head, config, *, available="2024-04-05", probability=.8):
    columns = L1_MA_ENTRY_COLUMNS if head == "ma_entry" else L1_MA_EXIT_COLUMNS
    target = "actual_complete_fee_net_roundtrip_positive" if head == "ma_entry" else "remaining_net_sell_proceeds_exit_next_open_minus_frozen_continue_over_current_market_value"
    contract = {"high_return": label_config(config), "strategy_hash": stable_hash(config), "initial_cash": 100000,
                "source": "ma", "target": target}
    return {"columns": columns, "head": head, "available_at": available, "train_label_end": "2024-03-01",
        "training_completed_at": "2026-10-10T20:00:00+08:00", "manifest_sha256": "test-hash",
        "target_contract": contract, "target_contract_hash": stable_hash(contract), "test_probability": probability,
        "fold": {"calibration_end": "2024-03-29", "policy_start": "2024-04-01", "policy_end": "2024-04-05"}}


class FakeSession:
    def __init__(self, lookup):
        self.cache = {}
        self.lookup = lookup
        self.seen = []
        self.diagnostics = {"rows": 0, "batches": 0, "model_loads": 0, "actual_device": "cpu"}
    def predict(self, frame, path, **kwargs):
        self.seen.append((frame.copy(), kwargs))
        self.diagnostics["rows"] += len(frame)
        self.diagnostics["batches"] += 1
        return np.full(len(frame), self.lookup(path)["test_probability"])
    def predict_one(self, values, path, **kwargs):
        self.seen.append((copy.deepcopy(values), kwargs))
        self.diagnostics["rows"] += 1
        return self.lookup(path)["test_probability"]


def test_forward_ma_reconstruction_routes_strictly_after_available_and_records_actual_clock(monkeypatch, tmp_path):
    frame, days = market_frame()
    config = runtime._config()
    saved = manifest("ma_entry", config)
    monkeypatch.setattr(runtime, "load_head", lambda path: saved)
    session = FakeSession(lambda path: saved)
    bundles = [{"name": "first", "available_at": saved["available_at"], "heads": {"ma_entry": "models/first/ma_entry"}}]
    result = runtime.forward_ma_outputs(frame, model_root=tmp_path, bundles=bundles, session=session, config=config)
    for day, p, fallback, lineage in zip(days, result["ma_entry_probabilities"], result["ma_fallback_mask"], result["ma_lineages"], strict=True):
        if day <= saved["available_at"]:
            assert np.isnan(p) and fallback and not lineage
        else:
            assert p == .8 and not fallback
            assert lineage["entry"]["actual_completion_record"].startswith("2026-10-10")
            assert lineage["entry"]["historical_reconstruction"]
            assert "actual_training_completed_at" not in lineage["entry"]
    assert runtime._bundle_for(saved["available_at"], bundles) is None
    assert runtime._bundle_for(days[-1], bundles, usage_mode="production", head="ma_entry", model_root=tmp_path) is None


def test_validation_route_is_only_disjoint_policy_half(monkeypatch, tmp_path):
    config = runtime._config(); saved = manifest("ma_entry", config)
    monkeypatch.setattr(runtime, "load_head", lambda path: saved)
    bundle = {"name": "first", "available_at": saved["available_at"], "heads": {"ma_entry": "entry"}}
    for day in ("2024-03-28", "2024-03-29", "2024-04-08"):
        assert runtime._bundle_for(day, [], validation_bundle=bundle, head="ma_entry", model_root=tmp_path) is None
    assert runtime._bundle_for("2024-04-01", [], validation_bundle=bundle, head="ma_entry", model_root=tmp_path) is bundle


def test_empty_model_route_uses_explicit_rules_and_true_max_holding_exit(tmp_path):
    frame, days = market_frame()
    config = runtime._config({"high_return": {"max_holding_sessions": 3}})
    points = runtime.prepare_points(frame, days, model_root=tmp_path, bundles=[], config=config)
    buy = [event for event in points["events"] if event["side"] == "BUY"]
    assert len(buy) == 1 and buy[0]["date"] == days[70]
    assert buy[0]["model_lineage"]["source"] == "causal_rule_v0"
    account = runtime.replay_high_return(frame, days, model_root=tmp_path, bundles=[], config=config,
                                         mode="reference", start=days[60])
    assert [fill["date"] for fill in account["ledger"]] == [days[71], days[74]]
    assert account["ledger"][1]["signal_date"] == days[73]
    assert account["holding_snapshots"][0]["actual_cost_per_share"] > 10.
    assert account["exit_policy_diagnostics"][-1]["forced_by_policy"]
    assert not account["publication_allowed"]


def test_ma_callback_prefix_is_fresh_for_each_actual_replay(monkeypatch, tmp_path):
    frame, days = market_frame()
    config = runtime._config({"version": "czsc_high_return_v1"})
    saved = manifest("ma_exit", config, available="2023-12-29", probability=.2)
    monkeypatch.setattr(runtime, "load_head", lambda path: saved)
    session = FakeSession(lambda path: saved)
    bundle = {"name": "ma", "available_at": "2023-12-29", "heads": {"ma_exit": "ma_exit"}}
    callback = runtime.make_ma_exit_policy(frame, days, tmp_path, [bundle], config=config, session=session)
    base = {"symbol": "000001.SZ", "date": days[71], "available_at": days[71]+"T15:00:00+08:00",
        "entry_date": days[71], "shares": 100, "entry_shares": 100, "plan_id": "same-economic-entry", "close": 10.,
        "actual_cost_per_share": 10., "stop_price": 9.8, "cash": 99000., "equity": 100000., "initial_cash": 100000.,
        "bar": {"high": 10.2, "low": 9.9}}
    callback(base)
    future = {**base, "date": days[72], "available_at": days[72]+"T15:00:00+08:00", "bar": {"high": 30., "low": 9.9}}
    callback(future)
    with pytest.raises(ValueError, match="prefix reused"):
        callback(base)
    other = callback.new_replay(); other(base)
    values = session.seen[-1][0]
    assert values["running_peak_return"] == pytest.approx(.02)
    assert len(other.diagnostics) == 1 and len(callback.diagnostics) == 2
    assert values["decision_at"] == base["available_at"]


def test_target_risk_capital_and_source_contracts_are_rejected():
    config = runtime._config()
    contract = manifest("ma_entry", config)["target_contract"]
    with pytest.raises(ValueError, match="initial-capital"):
        runtime._validate_compatibility(contract, initial_cash=200000)
    with pytest.raises(ValueError, match="source mismatch"):
        runtime._validate_compatibility(contract, source="union")
    with pytest.raises(ValueError, match="outcome/exit"):
        runtime._validate_compatibility(contract, contract_config=runtime._config({"high_return": {"min_net_return": .2}}))


def test_overlay_uses_raw_risk_plan_without_overriding_first_stage_veto(monkeypatch, tmp_path):
    frame, days = market_frame()
    config = runtime._config()
    points = runtime.prepare_points(frame, days, model_root=tmp_path, bundles=[], config=config)
    candidate = next(point for point in points["candidates"] if point["date"] == days[70])
    candidate["selected_by_reference"] = False
    candidate["source_selected_by_reference"]["ma"] = False
    saved = manifest("entry", config, available="2023-12-29")
    saved["columns"] = ENTRY_FEATURE_COLUMNS
    saved["target_contract"]["source"] = "union"
    saved["target_contract_hash"] = stable_hash(saved["target_contract"])
    session = FakeSession(lambda path: saved)
    monkeypatch.setattr(runtime, "HeadSession", lambda: session)
    monkeypatch.setattr(runtime, "load_head", lambda path: saved)
    monkeypatch.setattr(runtime, "prepare_points", lambda *args, **kwargs: copy.deepcopy(points))
    report = {"config": config, "source_snapshot": {"full_source_universe": False, "selected_count": 8},
        "model_bundles": [{"name": "entry", "available_at": "2023-12-29", "heads": {"entry": "entry"}}]}
    monkeypatch.setattr(runtime, "_read_run", lambda identity: (tmp_path, report, {"status": "complete"}))
    result = runtime.runtime_overlay(frame, days, model_run_id="partial-diagnostic", model_mode="historical", as_of=days[70],
        holding_snapshot={"shares": 100, "avg_cost": 10.})
    row = result["latest_decision"]
    assert row["risk_eligible"] and row["candidate_eligible"]
    assert not row["source_selected_by_reference"]
    assert row["raw_risk_plan"]["entry_allowed"] and not row["entry_plan"]["entry_allowed"]
    assert row["entry_probability"] == .8 and row["action"] == "WAIT"
    assert row["model_resolution"]["partial_scope"]
    assert row["exit_probability"] is None and not row["exit_model"]["holding_contract_compatible"]
    assert not result["publication_allowed"]
    # A raw point with invalid frozen risk inputs never receives a score.
    candidate["stop_price"] = 10.1
    blocked = runtime.runtime_overlay(frame, days, model_run_id="partial-diagnostic", model_mode="historical", as_of=days[70])
    assert not blocked["latest_decision"]["risk_eligible"]
    assert blocked["latest_decision"]["entry_probability"] is None


def test_automatic_overlay_ignores_partial_diagnostic_models(monkeypatch, tmp_path):
    frame, days = market_frame()
    selected = []
    monkeypatch.setattr(runtime, "high_return_research_status", lambda: {"models": [
        {"run_id": "tiny-newest", "training_completed_at": "2026-10-10T22:00:00+08:00", "automatic_selection_eligible": False},
        {"run_id": "full-pool", "training_completed_at": "2026-10-10T21:00:00+08:00", "automatic_selection_eligible": True}]})
    def read(identity):
        selected.append(identity)
        return tmp_path, {"model_bundles": [], "source_snapshot": {"full_source_universe": True, "selected_count": 1000}}, {}
    monkeypatch.setattr(runtime, "_read_run", read)
    overlay = runtime.runtime_overlay(frame, days, model_mode="historical")
    assert selected == ["full-pool"]
    assert not overlay["model_scope"]["partial_scope"]


def test_prepared_full_prefix_survives_bounded_execution_and_cache_identity_is_strict(monkeypatch, tmp_path):
    frame, days = market_frame(205)
    # An observed MA onset precedes the execution warmup. The later independent
    # structure candidate must retain its exact MA event age during scoring.
    frame.loc[130:149, ["volume", "amount"]] = 0.
    frame["structure_confirmed_at"] = None
    frame["confirmed_native_bs2_at"] = None
    frame["structure_low"] = 9.8
    frame.loc[190, ["structure_input_eligible", "confirmed_native_bs2", "structure_confirmed_at", "confirmed_native_bs2_at"]] = [True, "二买", days[190]+"T15:00:00+08:00", days[190]+"T15:00:00+08:00"]
    config = runtime._config({"high_return": {"max_holding_sessions": 3}})
    saved = manifest("entry", config, available="2023-12-29")
    saved.update(columns=ENTRY_FEATURE_COLUMNS)
    saved["target_contract"]["source"] = "union"
    saved["target_contract_hash"] = stable_hash(saved["target_contract"])
    monkeypatch.setattr(runtime, "load_head", lambda path: saved)
    bundles = [{"name": "entry", "available_at": "2023-12-29", "heads": {"entry": "entry"}}]
    session = FakeSession(lambda path: saved)
    points = runtime.prepare_points(frame, days, model_root=tmp_path, bundles=bundles, config=config, session=session)
    bounded = runtime.replay_high_return(frame, days, model_root=tmp_path, bundles=bundles, config=config,
        start=days[190], prepared_points=points, session=session)
    assert bounded["ledger"]
    entry_values = next(values for values, _ in session.seen if isinstance(values, dict))
    assert entry_values["ma_buy_event_age"] == 120.
    assert bounded["compute_info"]["execution_rows"] < len(frame)
    trimmed = runtime._execution_frame(frame.assign(date=days), days[190], config, {"tushare"})
    assert set(days[130:150]) <= set(trimmed.date)
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "_execution_frame", lambda frame, *args: frame.reset_index(drop=True))
        full_session = FakeSession(lambda path: saved)
        full = runtime.replay_high_return(frame, days, model_root=tmp_path, bundles=bundles, config=config,
            start=days[190], prepared_points=points, session=full_session)
    assert full["ledger"] == bounded["ledger"]
    assert full["daily"] == bounded["daily"]
    assert next(values for values, _ in full_session.seen if isinstance(values, dict))["ma_buy_event_age"] == 120.
    # Shared model caches never share a running holding prefix across accounts.
    again = runtime.replay_high_return(frame, days, model_root=tmp_path, bundles=bundles, config=config,
        start=days[190], prepared_points=points, session=session)
    assert again["ledger"] == bounded["ledger"]
    assert again["compute_info"]["model_inference_rows"] == bounded["compute_info"]["model_inference_rows"]
    for altered, kwargs in ((frame.iloc[:-1], {}), (frame.assign(close=10.01), {}),
        (frame, {"end": days[-2]}), (frame, {"config": runtime._config({"signal_points": {"ma_entry_threshold": .7}})})):
        with pytest.raises(ValueError, match="prepared high-return points"):
            runtime.replay_high_return(altered, days, model_root=tmp_path, bundles=bundles,
                config=kwargs.pop("config", config), start=days[190], prepared_points=points, session=session, **kwargs)


def test_l2_validation_replay_cannot_borrow_future_first_stage_ma_heads(monkeypatch, tmp_path):
    frame, days = market_frame()
    config = runtime._config({"version": "czsc_high_return_v1", "high_return": {"max_holding_sessions": 3}})
    seen = []
    original_prepare = runtime.prepare_points
    original_ma_exit = runtime.make_ma_exit_policy
    def prepare(*args, **kwargs):
        seen.append(("points", kwargs.get("validation_bundle")))
        return original_prepare(*args, **kwargs)
    def ma_exit(*args, **kwargs):
        seen.append(("ma_exit", kwargs.get("validation_bundle")))
        return original_ma_exit(*args, **kwargs)
    monkeypatch.setattr(runtime, "prepare_points", prepare)
    monkeypatch.setattr(runtime, "make_ma_exit_policy", ma_exit)
    validation = {"name": "future-ma", "available_at": days[-1],
        "heads": {"ma_entry": "future-entry", "ma_exit": "future-exit"},
        "fold": {"calibration_end": days[60], "policy_start": days[61], "policy_end": days[90]}}
    account = runtime.replay_high_return(frame, days, model_root=tmp_path, bundles=[], config=config,
        source="ma", mode="reference", start=days[60], validation_bundle=validation)
    assert seen == [("points", None), ("ma_exit", None)]
    assert account["ledger"] and account["compute_info"]["model_inference_rows"] == 0
    assert all(route["checkpoint"] is None for route in account["model_routes"])
