"""Real Broker timing, exit ownership, persistent intents and actual-state clones."""
import copy

import pandas as pd
import pytest

from my_strategy.services.czsc_trend_account import (
    replay_trend_account, clone_candidate_account, clone_holding_branches,
    candidate_from_row, prepare_account_context,
)
from my_strategy.tests.test_czsc_trend_consensus import identity

SYMBOL = "002436.SZ"


def fixture(length=78):
    dates = pd.bdate_range("2026-01-01", periods=length).strftime("%Y-%m-%d").tolist()
    frame = pd.DataFrame([{"date": d, "symbol": SYMBOL, "open": 10., "high": 10.2, "low": 9.8, "close": 10.,
        "volume": 1000000., "amount": 10000000., "source": "baostock", "has_trade_price": 1} for d in dates])
    at = dates[60] + "T15:00:00+08:00"
    plan = {"plan_id": "known-plan", "symbol": SYMBOL, "signal_date": dates[60], "available_at": at,
        "policy": "risk", "status": "active", "entry_allowed": True, "valid_for_sessions": 1,
        "stop_price": 9., "price_floor": 9.05, "price_ceiling": 10.5, "reference_price": 10., "target_weight": .8,
        "sources": ["ma", "structure"], "source_event_ids": ["candidate-origin"]}
    rows = [{"date": d, "available_at": d + "T15:00:00+08:00", "symbol": SYMBOL, "action": "WAIT", "emitted": False,
        "buy_base_score": .65, "fresh_candidate_ids": [], "entry_plan": None,
        "agents": {"ma": {"status": "scored", "p_up": .65, "p_down": .1, "p_neutral": .25, "model_identity": identity()}}} for d in dates]
    rows[60].update(action="BUY", emitted=True, event_id="B", fresh_candidate_ids=["candidate-origin"], entry_plan=plan)
    prepared = {"rows": rows, "direction_points": [{"date": dates[60], "side": "BUY", "event_id": "B"}]}
    frame.attrs["source_context_hash"] = "frozen-quote-calendar-vintage"
    return frame, dates, prepared


def replay(frame, dates, prepared, **kwargs):
    return replay_trend_account(frame, dates, prepared, start=dates[60], mode="E0", run_id="test", **kwargs)


def test_ordinary_sell_only_final_consensus_raw_union_cannot_close():
    frame, dates, prepared = fixture()
    prepared["rows"][62].update(rule_sell=True, source_exit_event_ids=["old-raw-source"])
    account = replay(frame, dates, prepared)
    assert [row["action"] for row in account["ledger"]] == ["BUY"]
    assert account["ledger"][0]["date"] == dates[61]
    assert account["ledger"][0]["signal_date"] == dates[60]
    prepared["rows"][65].update(action="SELL", emitted=True, event_id="S")
    account = replay(frame, dates, prepared)
    assert [row["action"] for row in account["ledger"]] == ["BUY", "SELL"]
    sell = account["ledger"][-1]
    assert sell["reason"] == "ordinary_consensus_sell" and sell["date"] == dates[66]
    assert sell["signal_date"] == dates[65]
    assert sell["intent_id"] == account["final_action_intents"][-1]["intent_id"]


def test_risk_stop_overrides_ordinary_sell_and_does_not_need_two_sources_or_HR():
    frame, dates, prepared = fixture()
    for i, price in [(62, 9.7), (63, 9.4), (64, 9.2), (65, 9.2)]:
        frame.loc[i, ["open", "close", "high", "low"]] = [price, price, price+.05, price-.05]
    frame.loc[64, ["open", "high"]] = [9.3, 9.35]  # stop first observed at this close
    prepared["rows"][64].update(action="SELL", emitted=True, event_id="S")
    account = replay(frame, dates, prepared, end=dates[65])
    sell = account["ledger"][-1]
    assert sell["action"] == "SELL" and sell["reason"] == "actual_account_stop"
    assert sell["date"] == dates[65]
    assert "ordinary_consensus_sell" in sell["all_triggered_reasons"]
    assert account["final_action_intents"][-1]["reason"] == "actual_account_stop"


def test_gap_open_protection_is_timestamped_today_and_new_buy_cannot_sell_same_day():
    frame, dates, prepared = fixture()
    frame.loc[62, ["open", "close", "high", "low"]] = [9.6, 9.6, 9.7, 9.5]
    frame.loc[63, ["open", "close", "high", "low"]] = [9.2, 9.2, 9.3, 9.1]
    account = replay(frame, dates, prepared, end=dates[63])
    sell = account["ledger"][-1]
    assert sell["date"] == dates[63]
    assert sell["original_signal_at"] == dates[63] + "T09:30:00+08:00"
    assert sell["signal_date"] == dates[63]
    assert account["final_action_intents"][-1]["reference_price"] == 9.2
    assert account["final_action_intents"][-1]["reference_price_source"] == "trigger_open"
    frame, dates, prepared = fixture()
    frame.loc[61, ["open", "close", "high", "low"]] = [9.6, 9., 9.7, 8.9]
    frame.loc[62, ["open", "close", "high", "low"]] = [9., 9., 9.1, 8.9]
    account = replay(frame, dates, prepared, end=dates[62])
    assert [row["date"] for row in account["ledger"]] == [dates[61], dates[62]]


def test_max_holding_is_direct_risk_action_without_fabricated_model_probability():
    frame, dates, prepared = fixture()
    account = replay(frame, dates, prepared, config={"account": {"max_holding_sessions": 3}})
    sell = account["ledger"][-1]
    assert sell["reason"] == "max_holding_sessions" and sell["date"] == dates[64]
    assert not account["exit_policy_diagnostics"]


def test_only_entry_owned_confirmed_boundary_can_exit_above_hard_stop():
    frame, dates, prepared = fixture()
    plan = prepared["rows"][60]["entry_plan"]
    boundary = {"symbol": SYMBOL, "source": "structure", "event_id": "candidate-origin", "original_event_id": "candidate-origin",
        "pattern_id": "owned-pattern", "price": 9.7, "first_available_at": dates[60]+"T15:00:00+08:00"}
    plan["holding_boundaries"] = [boundary]
    frame.loc[62:, ["open", "close", "high", "low"]] = [9.6, 9.6, 9.7, 9.5]
    owned = replay(frame, dates, prepared)
    assert owned["ledger"][-1]["reason"] == "holding_confirmed_boundary"
    assert owned["ledger"][-1]["date"] == dates[63]
    plan["holding_boundaries"][0]["event_id"] = "unowned-event"
    other = replay(frame, dates, prepared)
    assert [row["action"] for row in other["ledger"]] == ["BUY"]


def test_unverified_quote_cannot_create_risk_sell_or_poison_actual_valuation():
    frame, dates, prepared = fixture()
    frame.loc[62, ["close", "low"]] = [1., .5]
    frame.loc[62, "source"] = "unverified"
    account = replay(frame, dates, prepared)
    assert [row["action"] for row in account["ledger"]] == ["BUY"]
    assert next(row for row in account["daily"] if row["date"]==dates[62])["close"] == 10.
    assert next(row for row in account["rows"] if row["date"]==dates[62])["reason_codes"] == ["unverified_current_trade_price"]


def test_calendar_duration_exit_queues_on_unquoted_day_without_fake_price():
    frame, dates, prepared = fixture()
    frame.loc[63, "source"] = "unverified"
    account = replay(frame, dates, prepared, config={"account": {"max_holding_sessions": 3}})
    intent = account["final_action_intents"][-1]
    assert intent["date"] == dates[63] and intent["reference_price"] is None
    assert account["ledger"][-1]["signal_date"] == dates[63]
    assert account["ledger"][-1]["date"] == dates[65]


def test_missing_next_open_and_rejected_sell_keep_first_request_and_do_not_backdate():
    frame, dates, prepared = fixture()
    prepared["rows"][65].update(action="SELL", emitted=True, event_id="S")
    frame = frame.drop(index=66).reset_index(drop=True)
    prepared["rows"] = [row for row in prepared["rows"] if row["date"] != dates[66]]
    frame.attrs["source_context_hash"] = "frozen-quote-calendar-vintage"
    account = replay(frame, dates, prepared)
    sell = account["ledger"][-1]
    assert sell["date"] == dates[68] and sell["signal_date"] == dates[65]
    refusals = [r for r in account["rejections"] if r["action"] == "SELL"]
    assert [r["reason"] for r in refusals] == ["missing_target_market_session", "missing_previous_verified_session"]
    assert {r["intent_id"] for r in refusals} == {sell["intent_id"]}
    assert dates[66] in {row["date"] for row in account["daily"]}
    assert len([row for row in account["final_action_intents"] if row["side"] == "SELL"]) == 1


def test_partial_sell_retains_request_until_remainder_flat():
    frame, dates, prepared = fixture()
    prepared["rows"][65].update(action="SELL", emitted=True, event_id="S")
    frame.loc[65:, "volume"] = 20000.
    account = replay(frame, dates, prepared)
    sells = [row for row in account["ledger"] if row["action"] in {"SELL", "REDUCE"}]
    assert sells and sells[0]["action"] == "REDUCE"
    assert len({row["intent_id"] for row in sells}) == 1
    assert {row["signal_date"] for row in sells} == {dates[65]}
    assert account["metrics"]["open_shares"] > 0 and account["sell_pending"]


def test_actual_flat_quality_uses_same_builder_and_null_risk_block_never_calls_model():
    frame, dates, prepared = fixture()
    calls = []
    def quality(snapshot, candidate):
        calls.append((snapshot, candidate))
        assert snapshot["shares"] == 0 and snapshot["cash"] == 100000. and snapshot["close"] == 10.
        assert candidate["sources"] == ["ma", "structure"]
        assert candidate["trend_probability"]["p_down"] == .1
        return {"date": snapshot["date"], "available_at": snapshot["available_at"], "status": "scored", "probability": .12,
            "in_scope": True, "model_identity": identity("entry")}
    account = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", entry_evaluator=quality)
    assert len(calls) == 1 and account["ledger"][0]["action"] == "BUY"
    prepared["rows"][60]["entry_plan"]["status"] = "rejected"
    calls.clear()
    account = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", entry_evaluator=quality)
    assert not calls and not account["ledger"]
    assert account["entry_diagnostics"][0]["evaluation"]["probability"] is None


def test_clone_candidate_keeps_actual_cash_and_one_episode_only():
    frame, dates, prepared = fixture()
    reference = replay(frame, dates, prepared, end=dates[60])
    state = copy.deepcopy(reference["terminal_state"])
    state["cash"] = 20000.  # previously occurred withdrawals; capital identity remains 100000
    prepared["rows"][64].update(action="SELL", emitted=True, event_id="S")
    prepared["rows"][67].update(action="BUY", emitted=True, event_id="second-B", entry_plan={**prepared["rows"][60]["entry_plan"], "signal_date": dates[67], "available_at": dates[67]+"T15:00:00+08:00", "plan_id": "second"})
    cache = prepare_account_context(frame, dates)
    clone = clone_candidate_account(frame, dates, prepared, state, candidate_from_row(prepared["rows"][60]), account_context=cache)
    assert [row["action"] for row in clone["ledger"]] == ["BUY", "SELL"]
    assert clone["ledger"][0]["cash_after"] >= 0
    assert clone["daily"][-1]["date"] == dates[65]
    assert clone["series_context"]["initial_cash"] == 100000.


def test_forced_and_continued_holding_branches_use_real_prefix_and_new_fills_only():
    frame, dates, prepared = fixture()
    frame.loc[62, "high"] = 12.
    prepared["rows"][66].update(action="SELL", emitted=True, event_id="S")
    reference = replay(frame, dates, prepared, end=dates[63])
    snapshot = reference["holding_snapshots"][-1]
    assert snapshot["holding_prefix"]["peak"] == 12.
    forced = clone_holding_branches(frame, dates, prepared, snapshot, forced=True)
    continued = clone_holding_branches(frame, dates, prepared, snapshot, forced=False)
    assert forced["ledger"][0]["date"] == dates[64]
    assert continued["ledger"][0]["date"] == dates[67]
    assert all(row["action"] == "SELL" for row in forced["ledger"])
    assert forced["history_ledger"] == continued["history_ledger"]
    assert forced["terminal_state"]["cash"] == pytest.approx(snapshot["cash"] + sum(row["cash_flow"] for row in forced["ledger"]))


def test_incompatible_exit_head_cannot_invert_entry_probability_or_upgrade_checkpoint():
    frame, dates, prepared = fixture()
    def bad(snapshot):
        return {"head": "entry", "probability": .99, "apply": True, "request_exit": True,
            "date": snapshot["date"], "available_at": snapshot["available_at"],
            "holding_contract_compatible": True, "in_scope": True,
            "contract_hash": "old-entry", "model_identity": identity("entry")}
    account = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E1", exit_policy=bad)
    assert [row["action"] for row in account["ledger"]] == ["BUY"]
    assert "non_ma_exit_head_is_shadow" in account["exit_policy_diagnostics"][0]["validation_reasons"]
    evaluation = next(row["exit_evaluation"] for row in account["rows"] if row["exit_evaluation"])
    assert evaluation["probability"] is None and evaluation["raw_probability"] == .99
    assert evaluation["holding_contract_compatible"] is False and evaluation["in_scope"] is False
    assert evaluation["validation_reasons"] == account["exit_policy_diagnostics"][0]["validation_reasons"]


@pytest.mark.parametrize("probability,base,scope,model_family,reason", [
    (.05, .65, True, "czsc_trend_consensus", "high_return_below_threshold"),
    (.12, .45, True, "czsc_trend_consensus", "buy_total_support_below_threshold"),
    (.12, .65, False, "czsc_trend_consensus", "high_return_population_out_of_scope"),
    (.12, .65, True, "legacy", "identity_error"),
    (None, .65, True, "czsc_trend_consensus", "model_unavailable"),
])
def test_final_quality_rejection_explains_actual_failed_gate(probability, base, scope, model_family, reason):
    frame, dates, prepared = fixture()
    prepared["rows"][60]["buy_base_score"] = base
    def quality(snapshot, candidate):
        model = identity("entry")
        model["family"] = model_family
        return {"date": snapshot["date"], "available_at": snapshot["available_at"], "status": "scored",
            "probability": probability, "in_scope": scope, "model_identity": model}
    account = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", entry_evaluator=quality)
    assert not account["ledger"] and not account["final_action_intents"]
    assert account["entry_diagnostics"][0]["reason"] == reason
    assert account["entry_diagnostics"][0]["allowed"] is False


def test_valid_pinned_MA_exit_and_hard_risk_priority():
    frame, dates, prepared = fixture()
    def model(snapshot):
        return {"head": "holding_exit", "probability": .8, "apply": True, "request_exit": snapshot["date"] == dates[63],
            "date": snapshot["date"], "available_at": snapshot["available_at"],
            "contract_hash": "E0-target", "model_identity": identity("ma_exit"),
            "candidate_exit_binding": {"candidate_signal_date": dates[60], "plan_id": "known-plan", "checkpoint": "prior"}}
    account = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E1", exit_policy=model)
    assert account["ledger"][-1]["reason"] == "signal_pinned_ma_exit"
    assert account["ledger"][-1]["date"] == dates[64]
    risk = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E1", exit_policy=model,
        config={"account": {"max_holding_sessions": 3}})
    assert risk["ledger"][-1]["reason"] == "max_holding_sessions"
    assert risk["exit_policy_diagnostics"][-1]["hard_exit_priority"]


@pytest.mark.parametrize("drift", ["future_score", "future_policy_label", "foreign_checkpoint", "late_actual_training"])
def test_holding_head_must_be_causal_and_bound_to_original_entry(drift):
    frame, dates, prepared = fixture()
    def model(snapshot):
        result = {"head": "holding_exit", "probability": .8, "apply": True, "request_exit": True,
            "date": snapshot["date"], "available_at": snapshot["available_at"],
            "contract_hash": "E0-target", "model_identity": identity("ma_exit"),
            "candidate_exit_binding": {"candidate_signal_date": dates[60], "plan_id": "known-plan", "checkpoint": "prior"}}
        if drift == "future_score":
            result["available_at"] = (pd.Timestamp(snapshot["date"])+pd.Timedelta(days=1)).strftime("%Y-%m-%d")+"T15:00:00+08:00"
        elif drift == "future_policy_label":
            result["model_identity"]["policy_label_end"] = dates[60]
        elif drift == "foreign_checkpoint":
            result["candidate_exit_binding"]["checkpoint"] = "other"
        else:
            result["usage_mode"] = "production"
            result["model_identity"]["actual_training_completed_at"] = dates[61]+"T12:00:00+08:00"
        return result
    account = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E1", exit_policy=model)
    assert [row["action"] for row in account["ledger"]] == ["BUY"]
    assert all(row["probability"] is None and row["validation_reasons"] for row in account["exit_policy_diagnostics"])


def test_fixed_vintage_prefix_ids_actions_fills_and_rows_are_future_invariant():
    frame, dates, prepared = fixture()
    prepared["rows"][65].update(action="SELL", emitted=True, event_id="S")
    full = replay(frame, dates, prepared)
    cutoff = dates[66]
    clipped = frame.iloc[:67].copy()
    prior = {**prepared, "rows": prepared["rows"][:67]}
    prefix = replay(clipped, dates, prior)
    assert full["series_context_hash"] == prefix["series_context_hash"]
    assert [row for row in full["rows"] if row["date"] <= cutoff] == prefix["rows"]
    assert [row for row in full["ledger"] if row["date"] <= cutoff] == prefix["ledger"]
    assert [row for row in full["final_action_intents"] if row["date"] <= cutoff] == prefix["final_action_intents"]
    other = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E0", run_id="test", account_id="other")
    assert other["series_context_hash"] != full["series_context_hash"]
