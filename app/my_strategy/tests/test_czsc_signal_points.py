"""Causal independent point coverage, model identity and Broker-plan contracts."""
import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS
from my_strategy.services.czsc_signal_points import (
    MA_ENTRY_TARGET, MA_EXIT_TARGET, build_signal_points, make_point_decisions, signal_point_parameters,
)


def market_frame(n=15):
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=n),
        "symbol": "000001.SZ", "open": 10., "high": 10.2, "low": 9.9,
        "close": 10., "volume": 1_000_000., "amount": 10_000_000.,
        "source": "sina_unadjusted_v1", "has_trade_price": 1,
        "input_eligible": False, "model_input_eligible": True,
        "structure_input_eligible": False, "wyckoff_input_eligible": False,
        "rule_buy": False, "rule_sell": False, "rule_czsc_buy": False,
        "rule_price_volume_buy": False, "weekly_direction": -1.,
        "structure_low": 9.8, "structure_confirmed_at": None,
        "point_anchor": None, "point_confirmed_at": None,
        "confirmed_native_bs2": "其他", "confirmed_native_bs2_at": None,
        "confirmed_native_bs3": "其他", "confirmed_native_bs3_at": None,
        "wyckoff_rule_buy": False, "wyckoff_rule_sell": False,
        "wyckoff_event": "none", "wyckoff_stop": 9.8,
        "wyckoff_anchor_at": None, "wyckoff_observed_at": None,
        "wyckoff_available_at": None, "wyckoff_range_formed_at": None})
    for column in MA_TREND_COLUMNS:
        frame[column] = 0.
    frame["atr14_ratio"] = .01
    frame["ma20_slope5"] = .01
    frame["ma60_slope5"] = -.01  # breadth does not require the old joint gate
    frame["close_above_ma20"] = 1.
    frame["ma_alignment"] = 1.
    frame["ret_1"] = .01
    frame.index = np.arange(n) * 3 + 7
    return frame


def add_structure(frame, i=0, side="二买", confirmed=None):
    label = frame.index[i]
    close_at = frame.iloc[i].date.strftime("%Y-%m-%d") + "T15:00:00+08:00"
    frame.loc[label, "structure_input_eligible"] = True
    frame.loc[label, "structure_confirmed_at"] = confirmed or close_at
    frame.loc[label, "point_anchor"] = frame.iloc[i].date.strftime("%Y-%m-%d")
    frame.loc[label, "point_confirmed_at"] = confirmed or close_at
    frame.loc[label, "confirmed_native_bs2"] = side
    frame.loc[label, "confirmed_native_bs2_at"] = confirmed or close_at


def add_wyckoff(frame, i=0, side="BUY", observed=None):
    label = frame.index[i]
    close_at = frame.iloc[i].date.strftime("%Y-%m-%d") + "T15:00:00+08:00"
    frame.loc[label, "wyckoff_input_eligible"] = True
    frame.loc[label, "wyckoff_rule_" + side.lower()] = True
    frame.loc[label, "wyckoff_event"] = "Test" if side == "BUY" else "SOW"
    frame.loc[label, "wyckoff_observed_at"] = observed or close_at
    frame.loc[label, "wyckoff_available_at"] = observed or close_at
    frame.loc[label, "wyckoff_range_formed_at"] = close_at
    frame.loc[label, "wyckoff_anchor_at"] = frame.iloc[i].date.strftime("%Y-%m-%d")


def lineages(frame, *, target=MA_ENTRY_TARGET):
    return [{"available_at": "2023-12-29T15:00:00+08:00", "label_cutoff": "2023-12-28",
        "target": target, "target_contract_hash": "frozen-test-target", "checkpoint": "forward_fold"} for _ in range(len(frame))]


def test_ma_wide_candidate_ignores_structural_and_combined_rule_gate():
    frame = market_frame()
    points = build_signal_points(frame)
    assert points["candidates"][0]["sources"] == ["ma"]
    assert points["candidates"][0]["source_eligibility"] == {"ma": True, "structure": False, "wyckoff": False}
    assert not frame.input_eligible.any() and not frame.rule_buy.any()
    assert [event["side"] for event in points["events"]] == ["BUY"]
    assert points["events"][0]["probability"] is None
    assert points["events"][0]["ma_exit_status"] == "independent_exit_model_unavailable"


def test_wyckoff_own_qualification_and_same_day_union_keep_all_sources():
    frame = market_frame()
    add_structure(frame)
    add_wyckoff(frame)
    points = build_signal_points(frame)
    candidate = points["candidates"][0]
    assert candidate["sources"] == ["ma", "structure", "wyckoff"]
    assert len(candidate["source_event_ids"]) == 3
    assert len({event["opportunity_id"] for event in points["events"] if event["side"] == "BUY"}) == 1
    assert "native_ma_assisted_definition" in points["events"][1]["dependencies"]
    frame["model_input_eligible"] = False
    frame["structure_input_eligible"] = False
    points = build_signal_points(frame)
    assert points["candidates"][0]["source_eligibility"]["wyckoff"]
    assert not frame.input_eligible.any()


def test_future_confirmation_rejected_but_later_legitimate_identity_not_poisoned():
    frame = market_frame(3)
    future = frame.iloc[1].date.strftime("%Y-%m-%d") + "T15:00:00+08:00"
    add_structure(frame, 0, confirmed=future)
    add_structure(frame, 1, confirmed=future)
    frame.loc[frame.index[1], "point_anchor"] = frame.iloc[0].date.strftime("%Y-%m-%d")
    add_wyckoff(frame, 0, observed=future)
    points = build_signal_points(frame)
    rejected = [event for event in points["events"] if event["date"] == "2024-01-02" and event["source"] != "ma"]
    assert len(rejected) == 2 and not any(event["candidate_eligible"] for event in rejected)
    assert not any(event["evidence_visible"] for event in rejected)
    valid = [event for event in points["events"] if event["source"] == "structure" and event["candidate_eligible"]]
    assert len(valid) == 1 and valid[0]["date"] == "2024-01-03"
    assert valid[0]["evidence_visible"]


def test_future_append_never_changes_points_or_causal_stops():
    frame = market_frame(20)
    add_structure(frame, 2)
    add_wyckoff(frame, 4)
    original = frame.copy(deep=True)
    prefix = build_signal_points(frame.iloc[:10].copy())
    frame.loc[frame.index[10]:, ["close", "low", "high"]] = 500.
    full = build_signal_points(frame)
    assert prefix["events"] == [event for event in full["events"] if event["feature_index"] < 10]
    assert prefix["candidates"] == [item for item in full["candidates"] if item["feature_index"] < 10]
    assert original.iloc[0].low == 9.9


def test_ma_model_veto_retains_raw_candidate_and_counterfactual_plan():
    frame = market_frame()
    probs = np.full(len(frame), .2)
    points = build_signal_points(frame, ma_entry_probabilities=probs, ma_lineages=lineages(frame))
    candidate = points["candidates"][0]
    assert candidate["candidate_eligible"] and not candidate["selected_by_reference"]
    assert "ma_entry_probability_veto" in points["events"][0]["reason_codes"]
    assert points["events"][0]["evidence_visible"]
    reference = make_point_decisions(frame, points, source="ma")
    clone = make_point_decisions(frame, points, source="ma", include_unselected=True)
    assert reference[0]["hold_current_position"] and not reference[0]["entry_plan"]["entry_allowed"]
    assert clone[0]["entry_plan"]["entry_allowed"]
    assert not clone[0]["entry_plan"]["selected_by_reference"]


def test_every_source_candidate_carries_only_its_day_valid_forward_ma_score():
    frame = market_frame(5)
    add_wyckoff(frame, 2)
    add_structure(frame, 3)
    ancestry = lineages(frame)
    for item in ancestry:
        item["target_contract_hash"] = "frozen-ma-entry-target"
    ancestry[3]["available_at"] = "2025-01-01"
    points = build_signal_points(frame, ma_entry_probabilities=np.full(len(frame), .2), ma_lineages=ancestry)
    by_index = {candidate["feature_index"]: candidate for candidate in points["candidates"]}
    assert by_index[2]["sources"] == ["wyckoff"]
    assert by_index[2]["l1_ma_entry_available"] is True
    assert by_index[2]["l1_ma_entry_probability"] == .2  # A threshold veto is still a valid score.
    assert by_index[2]["l1_ma_entry_target_contract_hash"] == "frozen-ma-entry-target"
    assert by_index[3]["sources"] == ["structure"]
    assert by_index[3]["l1_ma_entry_available"] is False
    assert by_index[3]["l1_ma_entry_probability"] is None
    assert by_index[3]["l1_ma_entry_target_contract_hash"] is None
    no_model = build_signal_points(frame)["candidates"]
    assert all(item["l1_ma_entry_probability"] is None and not item["l1_ma_entry_available"]
        and item["l1_ma_entry_target_contract_hash"] is None for item in no_model)


@pytest.mark.parametrize("kind,reason", [
    ("nan", "ma_entry_probability_unavailable"),
    ("future", "ma_model_not_available"),
    ("retrospective", "ma_retrospective_model_rejected"),
    ("old_target", "ma_fixed_horizon_target_rejected"),
    ("disguised_target", "ma_entry_target_mismatch"),
    ("missing_target_hash", "ma_target_contract_hash_unavailable"),
    ("missing_lineage", "ma_model_lineage_unavailable"),
])
def test_ma_probabilities_do_not_bypass_lineage_or_use_nan_as_zero(kind, reason):
    frame = market_frame()
    probs = np.full(len(frame), .9)
    ancestry = lineages(frame)
    if kind == "nan": probs[0] = np.nan
    if kind == "future": ancestry[0]["available_at"] = "2025-01-01"
    if kind == "retrospective": ancestry[0]["retrospective"] = True
    if kind == "old_target": ancestry[0]["target"] = "fixed_10_sessions_net_positive"
    if kind == "disguised_target": ancestry[0]["target"] = "net_return_positive"
    if kind == "missing_target_hash": ancestry[0].pop("target_contract_hash")
    if kind == "missing_lineage": ancestry = None
    points = build_signal_points(frame, ma_entry_probabilities=probs, ma_lineages=ancestry)
    assert reason in points["events"][0]["reason_codes"]
    assert not points["events"][0]["selected_by_reference"]
    if kind == "nan": assert points["events"][0]["probability"] is None


def test_sell_output_independent_of_entry_probability():
    frame = market_frame()
    ancestry = lineages(frame)
    entry = np.zeros(len(frame))
    points = build_signal_points(frame, ma_entry_probabilities=entry, ma_lineages=ancestry)
    assert not any(event["side"] == "SELL" for event in points["events"])
    exit_probability = np.full(len(frame), np.nan)
    exit_probability[3] = .8
    ancestry[3] = {"entry": ancestry[3], "exit": {**ancestry[3], "target": MA_EXIT_TARGET}}
    points = build_signal_points(frame, ma_entry_probabilities=entry,
        ma_exit_probabilities=exit_probability, ma_lineages=ancestry)
    decisions = make_point_decisions(frame, points, source="ma")
    assert decisions[3]["rule_sell"] and not decisions[3]["hold_current_position"]
    assert decisions[3]["target_weight"] == 0.
    assert decisions[2]["hold_current_position"]


def test_future_exit_model_cannot_generate_sell_intent():
    frame = market_frame()
    ancestry = lineages(frame, target=MA_EXIT_TARGET)
    ancestry[2]["available_at"] = "2025-01-01"
    probabilities = np.full(len(frame), np.nan)
    probabilities[2] = .8
    points = build_signal_points(frame, ma_exit_probabilities=probabilities, ma_lineages=ancestry)
    assert not next(event for event in points["events"] if event["event_type"] == "ma_independent_exit_evidence")["evidence_visible"]
    assert not make_point_decisions(frame, points, source="ma")[2]["rule_sell"]


def test_cooldown_recovery_and_unique_plans_do_not_consume_theoretical_position():
    frame = market_frame(15)
    for i in (1, 6, 11): frame.loc[frame.index[i], "ret_1"] = -.01
    points = build_signal_points(frame)
    candidates = [candidate for candidate in points["candidates"] if "ma" in candidate["sources"]]
    assert [candidate["feature_index"] for candidate in candidates] == [0, 7, 12]
    decisions = make_point_decisions(frame, points, source="ma")
    plans = [decision["entry_plan"] for decision in decisions if decision["entry_gate_passed"]]
    assert len(plans) == len({plan["plan_id"] for plan in plans}) == 3
    assert all(plan["policy"] == "risk" and plan["valid_for_sessions"] == 1 for plan in plans)
    assert all(plan["price_floor"] > plan["stop_price"] < plan["price_ceiling"] for plan in plans)
    assert all(decision["hold_current_position"] for decision in decisions if decision["entry_plan"] is None)


def test_opposing_sell_retains_buy_fact_but_prevents_reference_entry():
    frame = market_frame()
    add_wyckoff(frame, side="SELL")
    points = build_signal_points(frame)
    assert "ma" in points["candidates"][0]["sources"]
    union = make_point_decisions(frame, points)
    assert union[0]["rule_sell"] and not union[0]["entry_plan"]["entry_allowed"]
    assert make_point_decisions(frame, points, source="ma")[0]["entry_plan"]["entry_allowed"]


def test_risk_floor_is_causal_and_invalid_stop_retains_unavailable_candidate():
    frame = market_frame()
    points = build_signal_points(frame, {"signal_points": {"min_risk_fraction": .03}})
    assert points["candidates"][0]["stop_price"] <= 9.7
    frame["atr14_ratio"] = .2
    unavailable = build_signal_points(frame)
    assert unavailable["candidates"] and not unavailable["candidates"][0]["candidate_eligible"]
    assert "stop_distance_exceeds_frozen_maximum" in unavailable["events"][0]["reason_codes"]
    assert not make_point_decisions(frame, unavailable)[0]["entry_gate_passed"]


def test_bad_contracts_rejected_and_input_not_mutated():
    frame = market_frame()
    original = frame.copy(deep=True)
    build_signal_points(frame)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError): build_signal_points(frame.iloc[::-1])
    with pytest.raises(ValueError): build_signal_points(frame, ma_entry_probabilities=[.5])
    with pytest.raises(ValueError): signal_point_parameters({"signal_points": {"min_risk_fraction": .3}})
    with pytest.raises(ValueError): signal_point_parameters({"signal_points": {"lookback_stop_bars": 2.5}})


def test_factual_points_and_neutral_rows_use_actual_broker_inventory():
    from my_strategy.services.czsc_analysis import strategy_config
    from my_strategy.services.czsc_backtest import execute_decisions
    from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
    frame = market_frame(75)
    frame.loc[frame.index[:65], "model_input_eligible"] = False
    add_wyckoff(frame, 68, side="SELL")
    points = build_signal_points(frame)
    decisions = make_point_decisions(frame, points,
        config={"signal_points": {"risk_fraction": .01}}, start=frame.iloc[65].date)
    account = execute_decisions("000001.SZ", frame, decisions,
        frame.iloc[65].date.strftime("%Y-%m-%d"), 100000., strategy_config(),
        "independent-points-broker-test", market_dates=frame.date.dt.strftime("%Y-%m-%d").tolist(),
        verified_sources=VERIFIED_SOURCES, entry_policy="risk")
    assert [row["action"] for row in account["ledger"]] == ["BUY", "SELL"]
    assert account["ledger"][0]["date"] == frame.iloc[66].date.strftime("%Y-%m-%d")
    assert account["ledger"][1]["date"] == frame.iloc[69].date.strftime("%Y-%m-%d")
    assert not account["rejections"]
    assert decisions[67]["hold_current_position"]


def test_explicit_pre_model_nan_fallback_remains_causal_rule_with_no_probability():
    frame = market_frame(15)
    frame.loc[frame.index[6], "ret_1"] = -.01
    probabilities = np.full(len(frame), np.nan)
    probabilities[7:] = .9
    ancestry = [{} for _ in range(7)] + lineages(frame)[7:]
    mask = np.arange(len(frame)) < 7
    points = build_signal_points(frame, ma_entry_probabilities=probabilities,
        ma_lineages=ancestry, ma_fallback_mask=mask)
    assert points["candidates"][0]["selected_by_reference"]
    event = points["events"][0]
    assert event["probability"] is None and event["probability_target"] is None
    assert event["model_lineage"]["source"] == "causal_rule_v0"
    assert event["model_lineage"]["fallback_reason"] == "no_compatible_ma_model_available_at_signal"
    assert points["ma_routes"][0]["status"] == "explicit_pre_model_rule_fallback"
    assert points["ma_routes"][7]["status"] == "model_selected"
    prefix = build_signal_points(frame.iloc[:7].copy(), ma_entry_probabilities=probabilities[:7],
        ma_lineages=ancestry[:7], ma_fallback_mask=mask[:7])
    assert prefix["ma_routes"] == points["ma_routes"][:7]
    assert prefix["events"] == [event for event in points["events"] if event["feature_index"] < 7]


def test_model_available_but_score_missing_cannot_use_v0_fallback():
    frame = market_frame()
    probabilities = np.full(len(frame), np.nan)
    points = build_signal_points(frame, ma_entry_probabilities=probabilities,
        ma_lineages=lineages(frame), ma_fallback_mask=np.ones(len(frame), dtype=bool))
    assert not points["candidates"][0]["selected_by_reference"]
    assert points["ma_routes"][0]["probability"] is None
    assert "ma_fallback_after_available_model_rejected" in points["ma_routes"][0]["reason_codes"]
    assert not make_point_decisions(frame, points)[0]["entry_gate_passed"]


def test_fallback_requires_explicit_mask_and_rejects_finite_or_invalid_mask():
    frame = market_frame()
    probabilities = np.full(len(frame), np.nan)
    points = build_signal_points(frame, ma_entry_probabilities=probabilities)
    assert not points["candidates"][0]["selected_by_reference"]
    with pytest.raises(ValueError):
        build_signal_points(frame, ma_entry_probabilities=np.full(len(frame), .9),
            ma_fallback_mask=np.ones(len(frame), dtype=bool))
    with pytest.raises(ValueError):
        build_signal_points(frame, ma_fallback_mask=[np.nan] * len(frame))
    with pytest.raises(ValueError):
        build_signal_points(frame, ma_fallback_mask=[True])
