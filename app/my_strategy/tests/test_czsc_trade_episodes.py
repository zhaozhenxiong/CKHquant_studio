"""Meaningful actual-fill, maturity, counterfactual and causal-prefix checks."""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.execution.cost_model import CostModel
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_high_return_features import ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS
from my_strategy.services.czsc_high_return_labels import episode_outcome
from my_strategy.services.czsc_signal_points import SIGNAL_POINTS_VERSION, signal_point_parameters
from my_strategy.services.czsc_trade_episodes import build_episode_dataset, training_frame


def test_event_ages_never_include_future_confirmation_but_keep_model_veto():
    from my_strategy.services.czsc_high_return_features import event_prefix_rows
    days = ["2024-01-02", "2024-01-03"]
    features = pd.DataFrame({"date": days, "symbol": "000001.SZ"})
    base = {"symbol": "000001.SZ", "date": days[0], "side": "BUY",
        "available_at": days[0] + "T15:00:00+08:00", "input_eligible": True}
    events = [{**base, "source": "structure", "reason_codes": ["structure_confirmation_not_visible"]},
        {**base, "source": "wyckoff", "observed_at": days[1] + "T15:00:00+08:00"},
        {**base, "source": "ma", "reason_codes": ["ma_model_veto"], "selected_by_reference": False}]
    values = event_prefix_rows(features, events, days)
    assert np.isnan(values[0]["structure_buy_event_age"])
    assert np.isnan(values[1]["wyckoff_buy_event_age"])
    assert values[1]["ma_buy_event_age"] == 1


@pytest.fixture
def market():
    dates = pd.bdate_range("2024-01-02", periods=100)
    days = dates.strftime("%Y-%m-%d").tolist()
    raw = pd.DataFrame({k: .1 for k in ENTRY_FEATURE_COLUMNS}, index=range(100))
    raw = raw.assign(date=days, symbol="000001.SZ", open=10., high=10.2,
        low=9.8, close=10., volume=1000000., amount=10000000., source="tushare", has_trade_price=1,
        input_eligible=True, model_input_eligible=True, structure_input_eligible=True, wyckoff_input_eligible=True)
    cfg = strategy_config(); cfg["high_return"] = {"max_holding_sessions": 60}
    cfg["signal_points"] = {"min_risk_fraction": .001}
    parameters = signal_point_parameters(cfg)
    buy = {"event_id": "buy70", "symbol": "000001.SZ", "date": days[70], "source": "ma",
        "side": "BUY", "event_type": "ma_rule", "available_at": days[70] + "T15:00:00+08:00",
        "input_eligible": True, "candidate_eligible": True, "selected_by_reference": True}
    sell = {**buy, "event_id": "sell80", "date": days[80], "side": "SELL",
        "available_at": days[80] + "T15:00:00+08:00"}
    candidate = {"symbol": "000001.SZ", "date": days[70], "opportunity_id": "opportunity70",
        "sources": ["ma"], "source_event_ids": ["buy70"], "source_eligibility": {"ma": True, "structure": False, "wyckoff": False},
        "source_selected_by_reference": {"ma": True, "structure": False, "wyckoff": False},
        "source_stops": {"ma": 9.6, "structure": None, "wyckoff": None}, "stop_price": 9.6,
        "atr14": .5, "candidate_eligible": True, "selected_by_reference": True}
    points = {"version": SIGNAL_POINTS_VERSION, "config": parameters, "config_hash": stable_hash(parameters),
        "events": [buy, sell], "candidates": [candidate]}
    return raw, days, cfg, points


def build(market, **kwargs):
    raw, days, cfg, points = market
    return build_episode_dataset(raw, days, cfg, points=points, sources=("ma", "union"), **kwargs)


def test_episode_actual_cash_and_main_target_match_ledger(market):
    raw, days, _, _ = market
    raw.loc[72:81, ["open", "close"]] = 11.
    raw.loc[72:81, "high"] = 11.2; raw.loc[72:81, "low"] = 10.8
    # Enter at 10, exit at 11; the guard uses yesterday's 11 close on exit.
    result = build(market)
    account = result["reference_accounts"]["union:000001.SZ"]
    buy, sell = account["ledger"]
    row = result["entries"].query("source == 'union'").iloc[0]
    assert row.entry_date == days[71] and row.exit_date == days[81]
    assert row.net_return == pytest.approx((sell["cash_flow"] + buy["cash_flow"]) / -buy["cash_flow"])
    assert row.net_return > .08 and row.r_multiple >= 2 and row.outcome_class == "high"
    episode = result["episodes"].query("source == 'union'").iloc[0]
    assert episode.net_return == pytest.approx(row.net_return)
    assert row.label_end == sell["date"] and row.fees == sum(r["fee"] for r in account["ledger"])
    assert list(training_frame(result).attrs["feature_columns"]) == list(ENTRY_FEATURE_COLUMNS)


def test_partial_exit_is_real_and_matures_only_when_flat(market):
    raw, days, _, _ = market
    raw.loc[80:85, "volume"] = 10000.
    result = build(market)
    account = result["reference_accounts"]["union:000001.SZ"]
    assert any(r["action"] == "REDUCE" for r in account["ledger"])
    final = account["ledger"][-1]
    row = result["entries"].query("source == 'union'").iloc[0]
    assert final["position_after"] == 0 and row.label_end == final["date"] > days[81]
    exits = result["exits"].query("source == 'union'")
    assert all(exits.label_end == exits[["forced_exit_date", "continuation_exit_date"]].max(axis=1))
    assert all(exits.continuation_exit_date == final["date"])


def test_rejected_candidate_has_no_false_loss_or_holding(market):
    raw, _, _, _ = market
    raw.loc[71, "open"] = 10.6
    result = build(market)
    assert result["episodes"].empty and result["exits"].empty
    assert not result["entries"].label_available.any()
    assert result["entries"].label.isna().all()
    assert all(result["entries"].label_reason.str.startswith("entry_rejected:"))


def test_unselected_flat_candidate_is_cloned_without_changing_reference(market):
    _, _, _, points = market
    points["candidates"][0]["source_selected_by_reference"]["ma"] = False
    points["candidates"][0]["selected_by_reference"] = False
    points["events"][0]["selected_by_reference"] = False
    result = build(market)
    assert result["episodes"].empty
    assert not result["entries"].selected_by_reference.any()
    assert result["entries"].input_eligible.all() and result["entries"].label_available.all()


def test_tail_is_censored_and_prefix_x_stays_unchanged(market):
    raw, days, cfg, points = market
    short = (raw.iloc[:77].copy(), days, cfg, points)
    before = build(short)
    after = build(market)
    assert not before["entries"].label_available.any()
    assert before["entries"].label.isna().all()
    prefix = after["exits"].loc[after["exits"].date <= days[76]]
    pd.testing.assert_frame_equal(before["exits"][["source", "date", *EXIT_FEATURE_COLUMNS]].reset_index(drop=True),
        prefix[["source", "date", *EXIT_FEATURE_COLUMNS]].reset_index(drop=True))
    assert all(before["exits"].label_reason == "continuation_unclosed")


def test_max_holding_is_an_intent_and_exit_waits_for_legal_open(market):
    raw, days, cfg, points = market
    cfg["high_return"]["max_holding_sessions"] = 5
    points["events"] = points["events"][:1]
    raw.loc[76, "open"] = 9.4  # rejected next-open sale by conservative lower guard
    raw.loc[76, "close"] = 10.
    result = build(market)
    row = result["entries"].query("source == 'union'").iloc[0]
    assert row.exit_date == days[77] and row.label_end == days[77]
    assert row.exit_rejection_count >= 1


def test_held_candidate_is_retained_without_virtual_new_entry(market):
    _, days, _, points = market
    candidate = copy.deepcopy(points["candidates"][0])
    candidate.update(date=days[75], opportunity_id="opportunity75")
    points["candidates"].append(candidate)
    result = build(market)
    held = result["entries"].query("date == @days[75]")
    assert len(held) == 2 and not held.label_available.any()
    assert all(held.label_reason == "candidate_not_decidable_existing_position")
    assert all(held.behavior_shares > 0)


def test_exit_source_does_not_require_unrelated_ma_eligibility(market):
    raw, _, _, points = market
    raw["model_input_eligible"] = False
    candidate = points["candidates"][0]
    candidate["sources"] = ["wyckoff"]
    candidate["source_eligibility"].update(ma=False, wyckoff=True)
    candidate["source_selected_by_reference"].update(ma=False, wyckoff=True)
    candidate["source_stops"].update(ma=None, wyckoff=9.6)
    for event in points["events"]:
        event["source"] = "wyckoff"
    result = build_episode_dataset(raw, market[1], market[2], points=points, sources=("wyckoff", "union"))
    assert result["exits"].input_eligible.all()
    assert not result["exits"].ma_input_available.any()


def test_stream_callbacks_preserve_count_without_retaining_clone_accounts(market):
    records, candidates, references = [], [], []
    result = build(market, collect=False, record_callback=lambda kind, row: records.append((kind, row)),
        candidate_callback=lambda row, account: candidates.append(row),
        reference_callback=lambda source, symbol, account: references.append((source, symbol)))
    assert result["entries"].empty and not result["reference_accounts"]
    assert len(candidates) == 2 and len(references) == 2
    assert len(records) == sum(result["coverage"].values())


def test_riskless_closed_episode_cannot_acquire_high_return_label():
    ledger = [{"action": "BUY", "shares": 100, "price": 10., "cash_flow": -1005., "fee": 5., "date": "2024-01-02"},
        {"action": "SELL", "shares": 100, "price": 12., "cash_flow": 1194., "fee": 6., "date": "2024-01-03"}]
    cost = CostModel(version="flat_v1", impact_tiers=())
    result = episode_outcome(ledger, stop_price=None, cost_model=cost)
    assert result["net_return"] > .08 and not result["label_available"] and np.isnan(result["label"])


def test_future_model_lineage_cannot_generate_historical_behavior(market):
    market[3]["events"][0]["model_lineage"] = {"available_at": market[1][70] + "T18:00:00+08:00"}
    with pytest.raises(ValueError, match="unavailable"):
        build(market)


def test_stateful_exit_policy_has_independent_reference_and_clone_prefixes(market):
    raw, days, cfg, points = market
    raw.loc[78, "high"] = 30.
    seen = []
    class Factory:
        def __call__(self, snapshot):
            raise AssertionError("shared policy closure must never execute")
        def new_replay(self):
            prefix = {"peak": 0., "rows": []}; seen.append(prefix)
            def policy(snapshot):
                prefix["peak"] = max(prefix["peak"], snapshot["bar"]["high"])
                prefix["rows"].append((snapshot["date"], prefix["peak"]))
                return None
            return policy
    result = build_episode_dataset(raw, days, cfg, points=points, sources=("ma",),
        exit_policies={"ma": Factory()}, exit_policy_hashes={"ma": "frozen-test-policy"})
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert seen[0]["peak"] == 30. and seen[1]["rows"][0] == (days[71], 10.2)
    assert result["entries"].label_available.all()


def test_l1_score_is_optional_and_holding_carries_only_entry_visible_score(market):
    result = build(market)
    assert result["entries"].l1_ma_entry_probability.isna().all()
    assert result["entries"].l1_ma_entry_available.eq(0).all()
    market[3]["candidates"][0].update(l1_ma_entry_probability=.7, l1_ma_entry_available=True)
    market[3]["events"][0]["model_lineage"] = {"available_at": market[1][69], "label_cutoff": market[1][68]}
    known = build(market)
    assert known["entries"].l1_ma_entry_probability.eq(.7).all()
    assert known["exits"].l1_ma_entry_probability.eq(.7).all()
    from my_strategy.services.czsc_high_return_features import (
        FEATURE_VERSION, L1_MA_ENTRY_COLUMNS, L1_MA_EXIT_COLUMNS, MA_ENTRY_SCHEMA_HASH, MA_EXIT_SCHEMA_HASH)
    assert MA_ENTRY_SCHEMA_HASH == stable_hash({"version": FEATURE_VERSION, "columns": L1_MA_ENTRY_COLUMNS, "scope": "independent_ma_entry"})
    assert MA_EXIT_SCHEMA_HASH == stable_hash({"version": FEATURE_VERSION, "columns": L1_MA_EXIT_COLUMNS, "scope": "independent_ma_actual_holding"})


def test_candidate_cash_clone_keeps_original_model_capital_identity(market):
    _, days, _, points = market
    later = copy.deepcopy(points["candidates"][0])
    later.update(date=days[85], opportunity_id="opportunity85")
    points["candidates"].append(later)
    class FrozenPolicy:
        def new_replay(self):
            def callback(state):
                assert state["initial_cash"] == 100000.
                return None
            return callback
    result = build_episode_dataset(market[0], days, market[2], points=points, sources=("ma",),
        exit_policies={"ma": FrozenPolicy()}, exit_policy_hashes={"ma": "fixed-capital-model"})
    second = result["entries"].loc[result["entries"].date == days[85]].iloc[0]
    assert second.clone_initial_cash != 100000. and second.clone_initial_cash == second.behavior_cash
