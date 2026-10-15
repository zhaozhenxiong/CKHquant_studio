import copy
import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.position_simulation import (
    _fifo, _sellable, run_position_simulation, continue_position_simulation,
    simulation_metrics,
)

SYMBOL = "002436.SZ"


def fixture(n=90):
    dates = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d").tolist()
    prices = 10 + np.arange(n)*.01
    frame = pd.DataFrame(dict(date=dates, open=prices, high=prices+.2, low=prices-.2,
                              close=prices, volume=10_000_000, amount=100_000_000,
                              has_trade_price=1))
    return dates, frame


def event(dates, frame, i, side="B", **extra):
    return dict(date=dates[i], event_id=f"{side}-{i}", confirmed=True, side=side,
                risk_boundary=float(frame.close.iloc[i])*.95, **extra)


def simulate(frame, events, **kwargs):
    config = dict(initial_cash=1_000_000, commission=0, min_commission=0, slippage=0,
                  transfer_fee=0, ma_period=3, atr_period=3,
                  max_holding_sessions=200, time_exit_sessions=200)
    config.update(kwargs.pop("config", {}))
    return run_position_simulation({SYMBOL: frame}, {SYMBOL: events}, config=config,
                                   calendar=frame.date.tolist(), **kwargs)


def test_confirmed_source_is_required_and_future_event_does_not_backfill():
    dates, frame = fixture()
    unconfirmed = event(dates, frame, 5); unconfirmed["confirmed"] = False
    future = event(dates, frame, 6, available_at=dates[10]+"T15:00:00+08:00")
    result = simulate(frame, [unconfirmed, future])
    assert result["trades"][0]["signal_date"] == dates[10]
    assert result["trades"][0]["date"] == dates[11]
    assert result["trades"][0]["price"] == frame.open.iloc[11]
    assert result["profit_certification"] is False
    json.dumps(result, allow_nan=False)


def test_actual_cash_exposure_is_not_full_stock_return():
    dates, frame = fixture(20)
    result = simulate(frame, [event(dates, frame, 5)], config=dict(
        initial_weight=.30, max_stock_weight=.30, position_step=.05, risk_budget=.03))
    buy = result["trades"][0]
    shares, initial = buy["shares"], result["metrics"]["initial_cash"]
    expected = initial + shares*(frame.close.iloc[-1]-buy["price"])
    assert result["daily"][-1]["equity"] == pytest.approx(expected)
    assert result["daily"][-1]["equity"] == pytest.approx(result["daily"][-1]["cash"]+shares*frame.close.iloc[-1])
    assert result["metrics"]["total_return"] < frame.close.iloc[-1]/buy["price"]-1
    pending=result["daily"][5]["positions"][SYMBOL]
    assert pending["shares"] == 0 and pending["weight"] == 0
    assert pending["target_weight"] == .30 and pending["state"] == "BUY_PENDING"


def test_missing_model_does_not_block_held_hard_stop():
    dates, frame = fixture(25)
    frame.loc[9:, ["open", "close"]] = 9.4
    frame.loc[9:, "high"] = 9.6
    frame.loc[9:, "low"] = 9.2
    forecast = dict(quality={"prediction_usable": True}, summary={"expected_return": .02, "up_fraction": .8})
    result = simulate(frame, [event(dates, frame, 5, forecast=forecast)], config={"policy": "structure_kronos"})
    assert [t["action"] for t in result["trades"]] == ["BUY", "SELL"]
    assert result["trades"][-1]["reason"] == "frozen_stop_gap"
    assert result["trades"][-1]["date"] == dates[9]
    assert result["daily"][9]["positions"][SYMBOL]["shares"] == 0


def test_hard_exit_remains_pending_during_suspension_and_limit_down():
    dates, frame = fixture(25)
    frame.loc[9, "close"] = 9.2
    frame.loc[9, "low"] = 9.0
    frame.loc[10, ["open", "close"]] = 9.1
    frame.loc[10, "suspended"] = True
    frame.loc[11, ["open", "close", "limit_down"]] = 8.19
    frame.loc[12:, ["open", "close"]] = 8.3
    result = simulate(frame, [event(dates, frame, 5)])
    sells = [t for t in result["trades"] if t["action"] == "SELL"]
    assert len(sells) == 1 and sells[0]["date"] == dates[12]
    order = next(o for o in result["orders"] if o["side"] == "SELL")
    reasons = [a.get("reason") for a in order["attempts"]]
    assert "suspended" in reasons
    assert "limit_down_or_slippage_outside_limit" in reasons
    assert result["daily"][11]["positions"][SYMBOL]["shares"] > 0


def test_t_plus_one_and_fifo_consume_only_unlocked_batches():
    lots = [dict(lot_id="old", date="2025-01-01", shares=100, unit_cost=10),
            dict(lot_id="new", date="2025-01-02", shares=100, unit_cost=20)]
    assert _sellable(lots, "2025-01-02") == 100
    with pytest.raises(ValueError, match="T\\+1"):
        _fifo(lots, 200, "2025-01-02")
    matched = _fifo(lots, 100, "2025-01-02")
    assert matched[0]["basis"] == 1000
    assert lots == [dict(lot_id="new", date="2025-01-02", shares=100, unit_cost=20)]


def test_partial_sale_recomputes_remaining_fifo_cost_and_does_not_repeat_source_s():
    dates, frame = fixture(35)
    source_s = event(dates, frame, 15, side="S")
    result = simulate(frame, [event(dates, frame, 5), event(dates, frame, 12), source_s,
                              {**source_s, "date": dates[18]}])
    buys = [t for t in result["trades"] if t["action"] == "BUY"]
    reductions = [t for t in result["trades"] if t["action"] == "REDUCE"]
    assert len(buys) == 2 and len(reductions) == 1
    sale = reductions[0]
    lots = result["state"]["lots"][SYMBOL]
    expected_cost = sum(l["shares"]*l["unit_cost"] for l in lots)/sum(l["shares"] for l in lots)
    assert sale["holding_cost_after"] == pytest.approx(expected_cost)
    assert sale["holding_cost_after"] > buys[0]["price"]
    assert sale["realized_pnl"] == pytest.approx(sale["cash_flow"]-sum(x["basis"] for x in sale["fifo"]))


def test_replay_resume_is_exact_and_same_day_is_idempotent():
    dates, frame = fixture(70)
    events = [event(dates, frame, 5), event(dates, frame, 12), event(dates, frame, 30, "S")]
    full = simulate(frame, events)
    prefix = simulate(frame, events, end=dates[20])
    resumed = simulate(frame, events, state=prefix["state"])
    for key in ("daily", "trades", "orders", "decisions", "metrics", "lineage"):
        assert resumed[key] == full[key], key
    again = simulate(frame, events, state=resumed["state"])
    assert again == resumed


def test_resume_rejects_revised_history_and_policy():
    dates, frame = fixture(40)
    events = [event(dates, frame, 5)]
    prefix = simulate(frame, events, end=dates[20])
    changed = frame.copy(); changed.loc[4, "close"] += .01
    with pytest.raises(ValueError, match="prefix changed"):
        simulate(changed, events, state=prefix["state"])
    with pytest.raises(ValueError, match="configuration"):
        simulate(frame, events, state=prefix["state"], config={"commission": .0004})


def test_future_prices_never_change_prefix_decisions_or_trades():
    dates, frame = fixture(40)
    events = [event(dates, frame, 5), event(dates, frame, 12)]
    original = simulate(frame, events, end=dates[20])
    changed = frame.copy(); changed.loc[21:, ["open", "high", "low", "close"]] *= 3
    replay = simulate(changed, events, end=dates[20])
    assert replay == original


def test_unfilled_buy_expires_and_end_of_period_does_not_create_fill():
    dates, frame = fixture(20)
    frame.loc[6, "suspended"] = True
    result = simulate(frame, [event(dates, frame, 5), event(dates, frame, 19)])
    assert result["trades"] == []
    assert result["orders"][0]["status"] == "expired"
    assert result["orders"][-1]["status"] == "queued"
    assert result["metrics"]["sharpe"] is None


def test_shared_portfolio_marks_all_positions_and_respects_cash():
    dates, frame = fixture(25)
    other = frame.copy(); other[["open", "high", "low", "close"]] *= 2
    config = dict(ma_period=3, atr_period=3, max_holding_sessions=200,
        time_exit_sessions=200, initial_weight=.50, max_stock_weight=.50,
        max_total_weight=.60, risk_budget=.10, min_commission=5)
    frames = {SYMBOL: frame, "600519.SH": other}
    events = {s: [event(dates, f, 5)] for s, f in frames.items()}
    result = run_position_simulation(frames, events, config=config, calendar=dates)
    day = result["daily"][6]
    assert day["cash"] >= 0
    assert day["exposure"] <= .601
    assert day["equity"] == pytest.approx(day["cash"]+sum(p["shares"]*p["mark"] for p in day["positions"].values()))
    reverse = run_position_simulation(dict(reversed(list(frames.items()))), events, config=config, calendar=dates)
    assert result == reverse


def test_fees_tax_slippage_and_ledger_cash_reconcile():
    dates, frame = fixture(30)
    result = simulate(frame, [event(dates, frame, 5), event(dates, frame, 10, "S")],
        config=dict(commission=.0003, min_commission=5, slippage=.001, transfer_fee=.00001))
    for trade in result["trades"]:
        assert trade["fee"] == pytest.approx(trade["commission"]+trade["stamp_tax"]+trade["transfer_fee"])
        assert trade["slippage_cost"] == pytest.approx(abs(trade["price"]-trade["reference_open"])*trade["shares"])
    assert result["daily"][-1]["cash"] == pytest.approx(1_000_000+sum(t["cash_flow"] for t in result["trades"]))
    buy = result["trades"][0]
    assert buy["price"] == pytest.approx(buy["reference_open"]*1.001)


def test_sharpe_uses_cash_days_initial_capital_and_sample_std():
    equity = 1000.0
    daily = []
    expected = [0.0]*20 + [.001, -.0005]*25
    for r in expected:
        equity *= 1+r
        daily.append(dict(equity=equity, exposure=.2, positions={}, stale_symbols=[]))
    trade = dict(price=10, shares=100, fee=0)
    metrics = simulation_metrics(daily, [trade], 1000)
    assert metrics["sharpe"] == pytest.approx(np.sqrt(252)*np.mean(expected)/np.std(expected, ddof=1))
    assert metrics["trading_sessions"] == 70
    assert metrics["daily_returns"] == pytest.approx(expected)
    assert simulation_metrics(daily[:59], [trade], 1000)["sharpe"] is None
    assert simulation_metrics(daily, [], 1000)["sharpe"] is None


def test_missing_close_keeps_inventory_and_labels_stale_valuation():
    dates, frame = fixture(70)
    shortened = frame[frame.date != dates[12]].copy()
    result = run_position_simulation({SYMBOL: shortened}, {SYMBOL:[event(dates,frame,5)]},
        calendar=dates, config=dict(ma_period=3, atr_period=3, time_exit_sessions=200, max_holding_sessions=200))
    assert result["daily"][12]["stale_symbols"] == [SYMBOL]
    assert result["daily"][12]["positions"][SYMBOL]["shares"] > 0
    assert "stale_held_valuation" in result["metrics"]["sharpe_reasons"]


def test_score_pending_and_missing_forecast_are_explicit_skips():
    dates, frame = fixture(20)
    base = event(dates, frame, 5)
    missing = simulate(frame, [base], config={"policy":"structure_kronos"})
    assert missing["decisions"][0]["reason"] == "forecast_missing_or_unusable"
    base["forecast"] = dict(quality={"prediction_usable":True}, summary={"expected_return":.02,"up_fraction":.6})
    base["score"] = {"status":"pending", "score":90}
    pending = simulate(frame, [base], config={"policy":"structure_kronos_score"})
    assert pending["decisions"][0]["reason"] == "simple_score_pending"
    assert not pending["trades"]


def test_kronos_actual_summary_names_and_policy_alias():
    dates, frame = fixture(20)
    forecast = dict(quality={"prediction_usable":True}, summary={"terminal_return_median":.02,"up_path_fraction":.6})
    result = simulate(frame, [event(dates, frame, 5, forecast=forecast)],
                      config={"strategy":"structure_kronos_sim_v1"})
    assert result["policy"] == "structure_kronos"
    assert len(result["trades"]) == 1


def test_cash_dividend_is_receivable_until_paid_and_does_not_fake_stop():
    dates, frame = fixture(25)
    # Cash dividend mechanically moves the price by exactly one yuan.
    frame.loc[11:, ["open", "high", "low", "close"]] -= 1
    action = dict(action_id="dividend", symbol=SYMBOL, source_id="verified-fixture",
        record_date=dates[10], ex_date=dates[11], pay_date=dates[15],
        cash_per_share=1.0, tax_per_share=.10)
    result = simulate(frame, [event(dates, frame, 5)], corporate_actions=[action])
    count = result["daily"][10]["positions"][SYMBOL]["shares"]
    assert count > 0
    assert result["daily"][11]["cash"] == result["daily"][10]["cash"]
    assert result["daily"][11]["receivable_cash"] == count
    assert result["daily"][11]["tax_payable"] == pytest.approx(count*.1)
    assert result["daily"][15]["cash"] == pytest.approx(result["daily"][10]["cash"]+count*.9)
    assert result["daily"][15]["receivable_cash"] == 0
    assert not [t for t in result["trades"] if t["action"] in {"SELL", "REDUCE"}]
    day = result["daily"][11]
    assert day["equity"] == pytest.approx(day["cash"]+day["market_value"]+day["receivable_cash"]-day["tax_payable"])
    replay = simulate(frame, [event(dates, frame, 5)], corporate_actions=[action], end=dates[12])
    resumed = simulate(frame, [event(dates, frame, 5)], corporate_actions=[action], state=replay["state"])
    assert resumed == result


def test_split_delayed_delivery_preserves_equity_and_fifo_cost_basis():
    dates, frame = fixture(25)
    frame.loc[11:, ["open", "high", "low", "close"]] /= 2
    action = dict(action_id="split", symbol=SYMBOL, source_id="verified-fixture",
        record_date=dates[10], ex_date=dates[11], share_multiplier=2.0,
        shares_pay_date=dates[13], sellable_date=dates[14], price_factor=.5)
    result = simulate(frame, [event(dates, frame, 5)], corporate_actions=[action])
    before = result["daily"][10]["positions"][SYMBOL]
    ex = result["daily"][11]["positions"][SYMBOL]
    delivered = result["daily"][13]["positions"][SYMBOL]
    assert ex["shares"] == before["shares"]
    assert ex["receivable_shares"] == before["shares"]
    assert delivered["shares"] == 2*before["shares"]
    assert delivered["locked_shares"] == before["shares"]
    assert result["daily"][14]["positions"][SYMBOL]["locked_shares"] == 0
    assert delivered["average_cost"] == pytest.approx(before["average_cost"]/2)
    assert result["daily"][11]["equity"] > result["daily"][10]["equity"]
    assert len(result["trades"]) == 1  # A split is not a BUY execution.
    assert result["daily"][11]["positions"][SYMBOL]["stop_price"] == pytest.approx(before["stop_price"]*.5)


def test_unknown_dividend_tax_and_fractional_shares_are_not_guessed():
    dates, frame = fixture(20)
    action = dict(action_id="bad", symbol=SYMBOL, source_id="fixture",
        record_date=dates[10], ex_date=dates[11], cash_per_share=1)
    with pytest.raises(ValueError, match="tax_per_share"):
        simulate(frame, [], corporate_actions=[action])


def test_time_failure_and_maximum_duration_exit_without_new_sell_evidence():
    dates, frame = fixture(40)
    frame.loc[7:, "close"] = frame.loc[6, "open"]
    frame.loc[7:, "open"] = frame.loc[6, "open"]
    result = simulate(frame, [event(dates, frame, 5)], config={"time_exit_sessions":5})
    assert result["trades"][-1]["reason"] == "time_failure_without_one_r"
    assert result["trades"][-1]["date"] == dates[11]
    dates, frame = fixture(40)
    expiry = simulate(frame, [event(dates, frame, 5)], config={"max_holding_sessions":5})
    assert expiry["trades"][-1]["reason"] == "holding_expiry"
    assert expiry["trades"][-1]["date"] == dates[11]


def test_portfolio_drawdown_exit_cooldown_can_rearm_without_recovering_old_peak():
    dates, frame = fixture(30)
    # Looser individual stop lets the shared drawdown policy act first.
    frame.loc[10:, ["open", "high", "low", "close"]] -= .45
    events = [event(dates, frame, 5), event(dates, frame, 15), event(dates, frame, 20)]
    for e in events: e["risk_boundary"] = 8
    result = simulate(frame, events, config=dict(initial_weight=.50, max_stock_weight=.50,
        max_total_weight=.90, risk_budget=.20, stop_loss=.50,
        ma_exit_closes=30, dd_rearm=.005, dd_reduce=.01, dd_exit=.015, dd_cooldown=3))
    assert any(t["reason"] == "portfolio_drawdown_exit" for t in result["trades"])
    assert any(d["portfolio_risk_stage"]==2 for d in result["daily"])
    assert result["daily"][14]["portfolio_risk_stage"] == 0
    assert result["daily"][14]["drawdown"] > .015
    assert result["daily"][14]["risk_cycle_drawdown"] == 0
    assert len([t for t in result["trades"] if t["action"] == "BUY"]) >= 2


def test_reduce_drawdown_latches_once_and_rearms_after_sustained_recovery():
    dates, frame = fixture(35)
    frame.loc[10:16, ["open", "high", "low", "close"]] -= .30
    events = [event(dates, frame, 5)]
    events[0]["risk_boundary"] = 8
    result = simulate(frame, events, config=dict(initial_weight=.50, max_stock_weight=.50,
        risk_budget=.20, stop_loss=.50, ma_exit_closes=40,
        dd_rearm=.005, dd_reduce=.01, dd_exit=.10, dd_rearm_sessions=3))
    reductions = [o for o in result["orders"] if o["reason"]=="portfolio_drawdown_reduce"]
    assert len(reductions) == 1
    assert result["daily"][12]["portfolio_risk_stage"] == 1
    assert result["daily"][23]["portfolio_risk_stage"] == 0


def test_benchmarks_use_shared_cash_fees_and_no_fabricated_source_signals():
    dates, frame = fixture(70)
    result = simulate(frame, [event(dates, frame, 5)])
    baseline = result["benchmarks"]
    assert set(baseline) == {"cash", "buy_hold_initial_budget", "buy_hold_full_budget"}
    assert baseline["cash"]["metrics"]["total_return"] == 0
    assert baseline["cash"]["metrics"]["sharpe"] is None
    assert baseline["buy_hold_initial_budget"]["initial_weight_per_symbol"] == .1
    assert baseline["buy_hold_full_budget"]["metrics"]["average_exposure"] > baseline["buy_hold_initial_budget"]["metrics"]["average_exposure"]
    assert baseline["buy_hold_full_budget"]["profit_certification"] is False


def test_score_actual_failed_qualification_blocks_only_scored_policy():
    dates, frame = fixture(20)
    e = event(dates, frame, 5, forecast=dict(quality={"prediction_usable":True},
        summary={"terminal_return_median":.02,"up_path_fraction":.6}),
        score=dict(status="scored", score=99, qualification={"policy_baseline_passed":False,
            "positive_allocation_eligible":False}))
    blocked = simulate(frame, [e], config={"policy":"structure_kronos_score"})
    assert not blocked["trades"]
    assert blocked["decisions"][0]["reason"] == "simple_score_policy_not_passed"
    allowed = simulate(frame, [e], config={"policy":"structure_kronos"})
    assert allowed["trades"]


def test_strict_execution_rules_and_unsupported_markets_reject_instead_of_guessing():
    dates, frame = fixture(20)
    strict = simulate(frame, [event(dates, frame, 5)], config={"require_verified_rules":True})
    assert not strict["trades"]
    assert strict["orders"][0]["terminal_reason"] == "historical_execution_rules_unverified"
    unsupported = run_position_simulation({"300001.SZ":frame}, {"300001.SZ":[event(dates,frame,5)]},
        config={"ma_period":3}, calendar=dates)
    assert not unsupported["trades"]
    assert unsupported["orders"][0]["terminal_reason"] == "unsupported_market_mainboard_only"


def test_stamp_tax_changes_by_effective_date_and_stress_keeps_tax_fixed():
    from my_strategy.services.position_simulation import _costs, simulation_config
    cfg = simulation_config({"cost_stress":2})
    assert _costs(cfg, "2023-08-25").stamp_tax == .001
    assert _costs(cfg, "2023-08-28").stamp_tax == .0005
    assert _costs(cfg, "2023-08-28").commission == .0006


def test_late_available_evidence_waits_until_next_close_then_next_open():
    dates, frame = fixture(20)
    e = event(dates, frame, 5, available_at=dates[5]+"T15:00:01+08:00")
    result = simulate(frame, [e])
    assert result["trades"][0]["signal_date"] == dates[6]
    assert result["trades"][0]["date"] == dates[7]


def test_boolean_cancel_callback_interrupts_before_account_mutation():
    dates, frame = fixture(20)
    with pytest.raises(InterruptedError, match="cancelled"):
        simulate(frame, [event(dates, frame, 5)], check_cancel=lambda: True)


def test_company_action_prefix_changes_require_branch_and_future_rights_do_not_leak():
    dates, frame = fixture(25)
    action = dict(action_id="dividend", symbol=SYMBOL, source_id="fixture", record_date=dates[15],
        ex_date=dates[16], pay_date=dates[20], cash_per_share=.10, tax_per_share=.01)
    before = simulate(frame, [event(dates,frame,5)], end=dates[12])
    with_future = simulate(frame, [event(dates,frame,5)], end=dates[12], corporate_actions=[action])
    assert before == with_future
    frozen = simulate(frame, [event(dates,frame,5)], end=dates[17], corporate_actions=[action])
    changed = {**action, "cash_per_share":.20}
    with pytest.raises(ValueError, match="prefix changed"):
        simulate(frame, [event(dates,frame,5)], state=frozen["state"], corporate_actions=[changed])


def test_continuous_weak_forecasts_reduce_once_without_source_s():
    dates, frame = fixture(25)
    good = dict(quality={"prediction_usable":True}, summary={"terminal_return_median":.02,"up_path_fraction":.7})
    weak = dict(quality={"prediction_usable":True}, summary={"terminal_return_median":-.02,"up_path_fraction":.3})
    evidence = [event(dates,frame,5,forecast=good)]
    evidence += [dict(date=dates[i],forecast=weak) for i in range(9,16)]
    result=simulate(frame,evidence,config={"policy":"structure_kronos"})
    reductions=[t for t in result["trades"] if t["action"]=="REDUCE"]
    assert len(reductions)==1 and reductions[0]["reason"]=="consecutive_forecast_weakness"
    assert reductions[0]["date"]==dates[11]
