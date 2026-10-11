"""Continuous actual NAV slicing and preregistered forward statistics."""
from __future__ import annotations

import copy
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_high_return_forward_protocol as forward
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES


def dates(count):
    return pd.bdate_range("2027-01-04", periods=count).strftime("%Y-%m-%d").tolist()


def variant(returns, *, trips=50):
    values = np.asarray(returns, dtype=float)
    equity = 100000 * np.cumprod(1 + values)
    return {"metrics": {"initial_cash": 100000, "completed_round_trips": trips},
        "account_count": 1, "cash_retained_failures": [],
        "daily": [{"date": day, "equity": float(nav), "return": float(ret), "market_value": 0.}
            for day, nav, ret in zip(dates(len(values)), equity, values)]}


def test_protocol_is_canonical_immutable_and_six_comparisons_are_registered():
    cfg = {"forward_protocol": {"seed": 17, "bootstrap_repetitions": 1000}, "unrelated": 5}
    before = copy.deepcopy(cfg)
    result = forward.forward_protocol(cfg)
    assert cfg == before
    assert result["window_count"] == 3 and result["window_sessions"] == 480
    assert result["block_lengths"] == [160, 240] and result["minimum_complete_blocks"] == 2
    assert result["min_window_round_trips"] == 50 and result["min_total_round_trips"] == 200
    assert result["preregistered_comparisons"] == 6
    assert result["per_comparison_alpha"] == pytest.approx(.05 / 6)
    assert result["one_sided_confidence"] == pytest.approx(1 - .05 / 6)
    assert result["minimum_detectable_daily_net_improvement"] == pytest.approx(.02 / 252)
    assert result["protocol_hash"] == stable_hash({k: v for k, v in result.items() if k != "protocol_hash"})
    assert forward.forward_protocol(result) == result
    result["block_lengths"].append(360)
    assert forward.forward_protocol()["block_lengths"] == [160, 240]


@pytest.mark.parametrize("setting,value", [
    ("window_count", 2), ("window_sessions", 479), ("block_lengths", [120, 160]),
    ("minimum_complete_blocks", 1), ("minimum_power", .79), ("familywise_alpha", .06),
    ("minimum_detectable_annual_net_improvement", .03), ("minimum_detectable_annual_net_improvement", 0),
    ("min_window_round_trips", 49), ("min_total_round_trips", 199),
    ("trading_sessions_per_year", 365), ("bootstrap_repetitions", 999),
    ("window_sessions", 480.5), ("seed", -1), ("seed", True), ("minimum_power", float("nan")),
])
def test_protocol_cannot_relax_registration(setting, value):
    with pytest.raises(ValueError):
        forward.forward_protocol({"forward_protocol": {setting: value}})


def test_frozen_derived_threshold_or_protocol_hash_tampering_is_rejected():
    for field, value in (("per_comparison_alpha", .05), ("protocol_hash", "changed")):
        result = forward.forward_protocol()
        result[field] = value
        with pytest.raises(ValueError, match="threshold changed|hash mismatch"):
            forward.forward_protocol(result)


def test_three_windows_are_consecutive_disjoint_and_only_observed_dates_complete():
    calendar = dates(1440)
    plans = forward.forward_windows(calendar, calendar[0], calendar[999], {})
    assert len(plans) == 3 and [w["ordinal"] for w in plans] == [1, 2, 3]
    assert [w["complete"] for w in plans] == [True, True, False]
    assert [w["observed_sessions"] for w in plans] == [480, 480, 40]
    assert [w["start"] for w in plans] == [calendar[0], calendar[480], calendar[960]]
    assert [w["end"] for w in plans] == [calendar[479], calendar[959], calendar[1439]]
    assert sum((w["dates"] for w in plans), []) == calendar[:1000]
    assert all(w["continuous_account_reset"] is False for w in plans)
    assert all(w["protocol_hash"] == forward.forward_protocol()["protocol_hash"] for w in plans)


def test_future_window_dates_are_never_extrapolated_from_business_days():
    calendar = dates(500)
    plans = forward.forward_windows(calendar, "2027-01-02", calendar[-1], {})
    assert plans[0]["end"] == calendar[479] and plans[0]["complete"]
    assert plans[1]["start"] == calendar[480] and plans[1]["end"] is None
    assert plans[1]["dates"] == calendar[480:]
    assert plans[2]["start"] is None and plans[2]["end"] is None
    assert plans[2]["dates"] == [] and plans[2]["complete"] is False
    empty = forward.forward_windows(calendar, None, calendar[-1], {})
    assert len(empty) == 3 and all(w["start"] is None and not w["dates"] for w in empty)


@pytest.mark.parametrize("calendar", [["2027-01-05", "2027-01-04"], ["2027-01-04"] * 2, ["NaT"]])
def test_unverified_order_duplicate_or_missing_calendar_dates_rejected(calendar):
    with pytest.raises(ValueError):
        forward.forward_windows(calendar, "2027-01-04", "2027-01-05", {})


def test_slice_uses_prior_actual_nav_and_keeps_original_capital_separate():
    account = variant([.10, .10, -.05, .02])
    account["metrics"].update(fees=12., net_return=.17249)
    before = copy.deepcopy(account)
    sliced = forward.slice_continuous_variant(account, dates(4)[2:], dates(4))
    previous = account["daily"][1]["equity"]
    final = account["daily"][-1]["equity"]
    assert sliced["window_basis"]["prior_date"] == dates(4)[1]
    assert sliced["window_basis"]["prior_equity"] == pytest.approx(previous)
    assert sliced["daily"][0]["return"] == pytest.approx(-.05)
    assert sliced["metrics"]["period_net_return"] == pytest.approx(final / previous - 1)
    assert sliced["metrics"]["cumulative_net_return_from_original_capital"] == pytest.approx(final / 100000 - 1)
    assert np.prod([1 + row["return"] for row in sliced["daily"]]) - 1 == pytest.approx(final / previous - 1)
    assert sliced["window_basis"]["account_reset"] is False
    assert "fees" not in sliced["metrics"] and sliced["account_lifetime_metrics"]["fees"] == 12
    assert "completed_round_trips" not in sliced["metrics"]
    assert account == before


def test_first_slice_uses_total_actual_portfolio_capital():
    account = variant([.01, -.02])
    account["metrics"]["initial_cash"] = 200000.
    account["account_count"] = 2
    for row in account["daily"]:
        row["equity"] *= 2
    sliced = forward.slice_continuous_variant(account, dates(2), dates(2), initial_cash=100000)
    assert sliced["window_basis"]["prior_equity"] == 200000.
    assert sliced["daily"][0]["return"] == pytest.approx(.01)
    empty = forward.slice_continuous_variant(account, [], dates(2))
    assert empty["daily"] == [] and empty["metrics"]["period_net_return"] is None


def test_missing_daily_rows_and_noncontiguous_cherry_picked_slices_rejected():
    account = variant([.01] * 4)
    with pytest.raises(ValueError, match="contiguous"):
        forward.slice_continuous_variant(account, [dates(4)[0], dates(4)[2]], dates(4))
    account["daily"].pop(1)
    with pytest.raises(ValueError, match="every verified date"):
        forward.slice_continuous_variant(account, dates(4), dates(4))


def test_real_broker_position_and_nav_continue_across_window_boundary():
    calendar = dates(960)
    close = np.full(960, 10.)
    close[479:487] = np.linspace(10., 11., 8)
    close[487:] = 11.
    market = pd.DataFrame({"date": calendar, "symbol": "000001.SZ", "open": close,
        "high": close * 1.02, "low": close * .98, "close": close, "volume": 1000000.,
        "amount": close * 1000000, "source": "tushare", "has_trade_price": 1})
    decisions = [{"date": day, "action": "HOLD", "target_weight": 0., "entry_plan": None,
        "hold_current_position": True} for day in calendar]
    plan = {"plan_id": "continuous_fixture", "symbol": "000001.SZ", "signal_date": calendar[478],
        "available_at": calendar[478] + "T15:00:00+08:00", "policy": "risk", "status": "active",
        "entry_allowed": True, "valid_for_sessions": 1, "stop_price": 9.5, "price_floor": 9.51,
        "price_ceiling": 12., "risk_fraction": .01, "target_weight": .25, "reference_price": 10.}
    decisions[478].update(action="BUY", target_weight=.25, entry_plan=plan, hold_current_position=False)
    decisions[485].update(action="SELL", hold_current_position=False)
    actual = execute_decisions("000001.SZ", market, decisions, calendar[0], 100000., strategy_config(),
        "forward-test", market_dates=calendar, verified_sources=VERIFIED_SOURCES,
        entry_policy="risk", exit_policy=lambda state: None)
    assert len(actual["ledger"]) == 2
    assert actual["ledger"][0]["date"] == calendar[479]
    assert actual["ledger"][1]["date"] == calendar[486]
    continuous = {"daily": actual["daily"], "metrics": {"initial_cash": 100000.}, "account_count": 1}
    second = forward.slice_continuous_variant(continuous, calendar[480:], calendar)
    assert second["daily"][0]["market_value"] > 0
    assert second["window_basis"]["prior_equity"] == pytest.approx(actual["daily"][479]["equity"])
    assert second["daily"][0]["return"] == pytest.approx(actual["daily"][480]["equity"] / actual["daily"][479]["equity"] - 1)
    assert second["metrics"]["period_net_return"] == pytest.approx(actual["daily"][-1]["equity"] / actual["daily"][479]["equity"] - 1)
    assert second["metrics"]["cumulative_net_return_from_original_capital"] == pytest.approx(actual["daily"][-1]["equity"] / 100000 - 1)


def test_paired_positive_economic_effect_uses_correct_one_sided_six_way_alpha():
    reference = variant(np.zeros(480))
    full = variant(np.full(480, .0002))
    result = forward.paired_forward_statistics(reference, full, {})
    assert result["status"] == "passed" and result["qualified"]
    assert result["publication_allowed"] is False
    assert result["paired_sessions"] == 480 and len(result["code_sha256"]) == 64
    assert result["thresholds"]["method"].startswith("registered_moving_date_blocks_diagnostic")
    assert set(result["comparisons"]) == {"160", "240"}
    for block, detail in result["comparisons"].items():
        assert detail["complete_blocks"] == 480 // int(block)
        assert detail["alpha"] == pytest.approx(.008333333333333333)
        assert detail["normal_critical_value"] == pytest.approx(NormalDist().inv_cdf(1 - .05 / 6))
        assert detail["one_sided_lower_ci"] == pytest.approx(.0002)
        assert detail["power_at_preregistered_mde"] >= .8
        assert detail["economic_increment_passed"]


@pytest.mark.parametrize("effect", [0., -.0002, .00001])
def test_no_increment_negative_or_below_registered_economic_effect_fails(effect):
    result = forward.paired_forward_statistics(variant(np.zeros(480)), variant(np.full(480, effect)))
    assert result["status"] == "failed" and not result["qualified"]
    assert "mean_below_registered_economic_increment" in result["reasons"]


def test_minimum_detectable_effect_and_power_cannot_be_replaced_by_observed_large_mean():
    # One long gain/loss regime: high observed mean with insufficient precision
    # at the preregistered 2%-annual effect must remain pending.
    returns = np.r_[np.full(240, -.009), np.full(240, .011)]
    result = forward.paired_forward_statistics(variant(np.zeros(480)), variant(returns))
    assert result["status"] == "pending" and not result["qualified"]
    assert result["comparisons"]["240"]["mean_daily_net_improvement"] > .02 / 252
    assert result["comparisons"]["240"]["power_at_preregistered_mde"] < .8
    assert result["comparisons"]["240"]["achieved_minimum_detectable_annual_improvement"] > .02


@pytest.mark.parametrize("periods,block", [(480, 160), (480, 240), (500, 240)])
def test_one_sided_lower_quantile_matches_existing_date_block_return_contract(periods, block):
    from my_strategy.services.czsc_research_ml import date_block_bootstrap
    returns = .0002 + .00004 * np.sin(np.arange(periods) / 19.)
    cfg = {"forward_protocol": {"window_sessions": periods, "bootstrap_repetitions": 1000, "seed": 27}}
    result = forward.paired_forward_statistics(variant(np.zeros(periods)), variant(returns), cfg)
    protocol = result["thresholds"]
    # Existing utility reports a two-sided interval. confidence=1-2*alpha
    # selects the SAME alpha lower tail as the new one-sided comparison.
    expected = date_block_bootstrap(pd.DataFrame({"date": dates(periods), "difference": returns}),
        "difference", block_length=block, repetitions=1000, seed=27,
        confidence=1 - 2 * protocol["per_comparison_alpha"], dependence_horizon=120)
    actual = result["comparisons"][str(block)]
    assert actual["one_sided_lower_ci"] == pytest.approx(expected["lower"], abs=1e-14)
    assert actual["mean_daily_net_improvement"] == pytest.approx(expected["mean"], abs=1e-14)
    assert result["qualified"]


@pytest.mark.parametrize("periods", [0, 40, 320, 479, 481, 960])
def test_only_exact_complete_registered_window_can_be_statistically_qualified(periods):
    result = forward.paired_forward_statistics(variant(np.zeros(periods)), variant(np.full(periods, .0002)))
    assert result["status"] == "pending" and not result["qualified"]


@pytest.mark.parametrize("mutation", ["missing_date", "missing_return", "nan_return", "forged_return", "changed_cash", "changed_universe", "failure", "failed_status", "immature", "few_trips", "missing_trip_count"])
def test_unavailable_failed_immature_or_unpaired_actual_evidence_stays_pending(mutation):
    reference, full = variant(np.zeros(480)), variant(np.full(480, .0002))
    if mutation == "missing_date": full["daily"].pop(100)
    elif mutation == "missing_return":
        for row in full["daily"]: row.pop("return")
    elif mutation == "nan_return": full["daily"][100]["return"] = float("nan")
    elif mutation == "forged_return": full["daily"][100]["return"] = .3
    elif mutation == "changed_cash": full["metrics"]["initial_cash"] = 200000
    elif mutation == "changed_universe": full["account_count"] = 2
    elif mutation == "failure": full["cash_retained_failures"] = [{"symbol": "000002.SZ", "reason": "missing"}]
    elif mutation == "failed_status": full["status"] = "failed"
    elif mutation == "immature": full["label_maturity_pending"] = True
    elif mutation == "few_trips": full["metrics"]["completed_round_trips"] = 49
    elif mutation == "missing_trip_count": full["metrics"].pop("completed_round_trips")
    result = forward.paired_forward_statistics(reference, full)
    assert result["status"] == "pending" and not result["qualified"] and result["reasons"]


def test_marked_open_positions_do_not_invalidate_mature_nav_evidence():
    reference, full = variant(np.zeros(480)), variant(np.full(480, .0002))
    for value in (reference, full):
        value.update(open_shares=1000, censored_trips=10)
        value["metrics"]["open_positions"] = 10
    result = forward.paired_forward_statistics(reference, full)
    assert result["qualified"] and result["status"] == "passed"


def test_pairing_is_daily_portfolio_net_return_not_cumulative_account_profit():
    reference = variant(np.full(480, .001))
    full = variant(np.full(480, .0012))
    result = forward.paired_forward_statistics(reference, full)
    assert result["qualified"]
    assert result["comparisons"]["160"]["mean_daily_net_improvement"] == pytest.approx(.0002)
    assert result["comparisons"]["240"]["annualized_arithmetic_mean_net_improvement"] == pytest.approx(.0002 * 252)
    # Separate window returns and lifetime NAV are still available for audit.
    slice_ref = forward.slice_continuous_variant(reference, dates(480)[240:], dates(480))
    assert slice_ref["metrics"]["period_net_return"] != pytest.approx(slice_ref["metrics"]["cumulative_net_return_from_original_capital"])
