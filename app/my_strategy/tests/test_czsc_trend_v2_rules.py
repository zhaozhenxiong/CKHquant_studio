"""Daily reassessment is observed before the gate; real intents reserve only once."""
import copy

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_v2_account import replay_trend_account, clone_candidate_account, _flat_state_hash
from my_strategy.services.czsc_trend_v2_consensus import prepare_trend_decisions
from my_strategy.services.czsc_trend_v2_contracts import STRATEGY_ID, strategy_config


SYMBOL = "002436.SZ"


def identity(head="trend_direction", family=STRATEGY_ID):
    return {"family": family, "head": head, "verified_frozen_contract": True,
        "checkpoint": "strict-prior", "manifest_sha256": "frozen-manifest", "feature_schema_hash": "frozen-schema",
        "label_contract_hash": "frozen-target", "train_label_end": "2024-12-31", "calibration_label_end": "2025-01-31",
        "policy_label_end": "2025-02-28", "simulated_available_at": "2025-03-01T15:00:00+08:00"}


def fixture(length=85, *, stop=9., boundary=9.3, prices=None, sell_from=65):
    dates = pd.bdate_range("2026-01-01", periods=length).strftime("%Y-%m-%d").tolist()
    frame = pd.DataFrame([{"date": date, "symbol": SYMBOL, "open": (prices or {}).get(i, 10.),
        "high": (prices or {}).get(i, 10.) + .2, "low": (prices or {}).get(i, 10.) - .2,
        "close": (prices or {}).get(i, 10.), "volume": 1000000., "amount": 10000000.,
        "source": "baostock", "has_trade_price": 1,
        "close_above_ma20": 1., "ma20_slope5": .1, "ma_alignment": 1., "ret_5": .1} for i, date in enumerate(dates)])
    frame.attrs["source_context_hash"] = "frozen-quotes-for-Broker-contract-test"
    at = dates[60] + "T15:00:00+08:00"
    fact = {"symbol": SYMBOL, "date": dates[60], "available_at": at, "first_available_at": at,
        "first_confirmed_at": at, "event_id": "original-buy", "original_event_id": "original-buy",
        "pattern_id": "original-pattern", "source": "structure", "side": "BUY", "confirmed": True,
        "input_eligible": True, "invalidation_price": boundary, "risk_stop_price": stop}
    evaluations, trends = [], []
    for i, date in enumerate(dates):
        at, close = date + "T15:00:00+08:00", float(frame.iloc[i].close)
        bearish = sell_from is not None and i >= sell_from
        structure = {"date": date, "available_at": at, "source_id": "structure:" + date,
            "version": "pure_confirmed_czsc_v4_20261012", "input_eligible": True,
            "support_facts": [copy.deepcopy(fact)] if i >= 60 else [],
            "current_state": {"close": close, "buy_score": .1 if bearish else .9 if i >= 60 else 0.,
                "sell_score": .85 if bearish else 0., "support_valid": True, "support_floor": boundary,
                "resistance_valid": True, "resistance_ceiling": 11.}}
        wyckoff = {"date": date, "available_at": at, "source_id": "wyckoff:" + date,
            "version": "wyckoff_current_pv_v4_20261012", "input_eligible": True, "support_facts": [],
            "current_state": {"close": close, "buy_score": .05 if bearish else .65, "sell_score": .8 if bearish else .05}}
        evaluations.append({"date": date, "available_at": at, "structure": structure, "wyckoff": wyckoff})
        p = (.1, .5, .4) if bearish else (.65, .1, .25)
        trends.append({"date": date, "available_at": at, "input_eligible": True, "model_available": True,
            "p_up": p[0], "p_down": p[1], "p_neutral": p[2], "model_identity": identity()})
    return frame, dates, {"events": [fact], "evaluations": evaluations}, trends


def prepare(frame, dates, pack, trends, cfg=None):
    return prepare_trend_decisions(frame, dates, native_pack=pack, trend_probabilities=trends, config=cfg, mode="E0")


def score(snapshot, probability=.2, *, family=STRATEGY_ID, in_scope=True):
    return {"date": snapshot["date"], "available_at": snapshot["available_at"], "status": "scored",
        "probability": probability, "in_scope": in_scope, "model_identity": identity("entry", family)}


def test_prepare_reassesses_every_valid_original_day_without_consumption_or_entry_cooldown():
    frame, dates, pack, trends = fixture(sell_from=None)
    result = prepare(frame, dates, pack, trends)
    assert all(result["rows"][i]["entry_eligible"] for i in range(60, 80))
    assert all(result["rows"][i]["action"] == "BUY" for i in range(60, 80))
    assert not result["rows"][80]["entry_eligible"]
    assert result["rows"][60]["original_opportunity_ids"] == result["rows"][79]["original_opportunity_ids"]
    assert result["rows"][60]["entry_plan"]["plan_id"] != result["rows"][61]["entry_plan"]["plan_id"]
    assert prepare(frame.iloc[:70], dates, pack, trends)["rows"] == result["rows"][:70]


def test_l2_reject_missing_then_pass_same_opportunity_real_fill_and_no_event_continuous_sell():
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    observed = []

    def evaluator(snapshot, candidate):
        observed.append((copy.deepcopy(snapshot), copy.deepcopy(candidate)))
        if snapshot["date"] == dates[61]:
            return {"status": "model_unavailable", "probability": None, "reason_codes": ["no_prior_high_return_head"]}
        return score(snapshot, .02 if snapshot["date"] == dates[60] else .2)

    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", entry_evaluator=evaluator,
        run_id="same-opportunity", account_id="same-account")
    diagnostics = result["entry_diagnostics"]
    assert [row["date"] for row in diagnostics] == dates[60:63]
    assert [row["allowed"] for row in diagnostics] == [False, False, True]
    assert len({row["attempt_id"] for row in diagnostics}) == 3
    assert len({row["economic_group_id"] for row in diagnostics}) == 1
    for row in diagnostics:
        state = row["pre_gate_state"]
        assert state["shares"] == 0 and state["pending_intent"] is None
        assert state["cash"] == state["initial_cash"] == 100000
        assert state["reserved_opportunity_ids"] == []
        assert row["reference_state_hash"] == stable_hash(state)
        assert row["entry_plan_hash"] == stable_hash(row["candidate"]["entry_plan"])
        assert row["snapshot"]["cash"] == state["cash"]
    buy, sell = result["ledger"]
    assert (buy["action"], buy["signal_date"], buy["date"]) == ("BUY", dates[62], dates[63])
    assert (sell["action"], sell["signal_date"], sell["date"]) == ("SELL", dates[65], dates[66])
    assert buy["shares"] > 0 and buy["fee"] > 0 and sell["position_after"] == 0 and sell["fee"] > 0
    assert sell["reason"] == "ordinary_consensus_sell"
    assert prepared["rows"][65]["sell_pass"] is True
    assert prepared["rows"][65]["candidate_evidence"] == []
    assert not any(event["side"] == "SELL" for event in prepared["rows"][65]["active_candidates"])
    assert result["terminal_state"]["reserved_opportunity_ids"] == diagnostics[2]["original_opportunity_ids"]


def test_pre_gate_clone_exists_after_reference_E1_already_admitted_and_owns_real_cash():
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    reference = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E1", run_id="E1-reference")
    assert reference["holding_snapshots"][0]["date"] == dates[61]
    final = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="gated-reference",
        entry_evaluator=lambda snapshot, candidate: score(snapshot, .02 if snapshot["date"] == dates[60] else .2))
    diagnostic = final["entry_diagnostics"][1]
    assert diagnostic["date"] == dates[61] and diagnostic["pre_gate_state"]["shares"] == 0
    assert not any(row["date"] == dates[61] for row in reference["entry_diagnostics"])
    clone = clone_candidate_account(frame, dates, prepared, diagnostic["pre_gate_state"], diagnostic["candidate"], run_id="label-clone")
    assert [row["action"] for row in clone["ledger"]] == ["BUY", "SELL"]
    assert clone["ledger"][0]["date"] == dates[62]
    assert clone["ledger"][0]["signal_date"] == dates[61]
    assert clone["metrics"]["open_shares"] == 0 and clone["metrics"]["fees"] > 0


def test_quality_probability_alone_cannot_bypass_total_gate_or_consume_opportunity():
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    prepared["rows"][60]["buy_base_score"] = .4
    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="total-gate",
        entry_evaluator=lambda snapshot, candidate: score(snapshot))
    rejected, accepted = result["entry_diagnostics"][:2]
    assert rejected["reason"] == "buy_total_support_below_threshold" and not rejected["allowed"]
    assert accepted["date"] == dates[61] and accepted["allowed"]
    assert rejected["original_opportunity_ids"] == accepted["original_opportunity_ids"]
    assert result["ledger"][0]["date"] == dates[62]


@pytest.mark.parametrize("head,expected_sell", [("holding_exit", 64), ("exit", 66)])
def test_exit_binding_uses_accepted_signal_after_reassessment_and_L2_exit_stays_shadow(head, expected_sell):
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    snapshots = []

    def exit_policy(snapshot):
        snapshots.append(copy.deepcopy(snapshot))
        return {"date": snapshot["date"], "available_at": snapshot["available_at"], "usage_mode": "historical",
            "head": head, "probability": .9, "apply": True, "request_exit": True, "research_arm": True,
            "holding_contract_compatible": True, "contract_hash": "frozen-exit-target",
            "model_identity": identity("ma_exit" if head == "holding_exit" else "exit"),
            "entry_exit_binding": {"plan_id": snapshot["plan_id"], "candidate_signal_date": snapshot["entry_signal_date"],
                "checkpoint": "strict-prior"}}

    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="pinned-reassessment",
        entry_evaluator=lambda snapshot, candidate: score(snapshot, .02 if snapshot["date"] < dates[62] else .2),
        exit_policy=exit_policy)
    assert result["ledger"][0]["signal_date"] == dates[62]
    assert snapshots[0]["entry_signal_date"] == dates[62] != pack["events"][0]["date"]
    assert result["ledger"][-1]["date"] == dates[expected_sell]
    assert result["ledger"][-1]["reason"] == ("signal_pinned_ma_exit" if head == "holding_exit" else "ordinary_consensus_sell")
    if head == "exit":
        assert all(not row["applied"] for row in result["exit_policy_diagnostics"])
        assert all(row["probability"] is None for row in result["exit_policy_diagnostics"])


def test_continuous_sell_partial_fills_keep_one_first_intent_until_inventory_is_flat():
    frame, dates, pack, trends = fixture()
    frame.loc[[65,66], "volume"] = 15000.
    frame.loc[[65,66], "amount"] = 150000.
    result = replay_trend_account(frame, dates, prepare(frame, dates, pack, trends), start=dates[60], mode="E0",
        run_id="partial-continuous-sell")
    buy, *sells = result["ledger"]
    assert len(sells) == 3 and [row["date"] for row in sells] == dates[66:69]
    assert all(row["signal_date"] == dates[65] for row in sells)
    assert len({row["intent_id"] for row in sells}) == 1
    assert sum(row["shares"] for row in sells) == buy["shares"]
    assert len([row for row in result["final_action_intents"] if row["side"] == "SELL"]) == 1
    assert result["metrics"]["open_shares"] == 0


def test_risk_block_does_not_call_l2_or_reserve_and_later_valid_day_can_enter():
    prices = {i: 9.4 for i in range(62, 85)}
    prices[61] = 9.6
    frame, dates, pack, trends = fixture(stop=8., boundary=7.7, prices=prices, sell_from=65)
    prepared = prepare(frame, dates, pack, trends)
    assert not prepared["rows"][60]["entry_eligible"] and not prepared["rows"][61]["entry_eligible"]
    assert prepared["rows"][62]["entry_eligible"]
    scored = []
    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final",
        entry_evaluator=lambda snapshot, candidate: (scored.append(snapshot["date"]) or score(snapshot)), run_id="risk-recovers")
    assert scored == [dates[62]]
    assert result["ledger"][0]["signal_date"] == dates[62]
    assert result["ledger"][0]["date"] == dates[63]
    assert all(not state["reserved_opportunity_ids"] for state in result["account_snapshots"][:2])


def test_broker_rejection_keeps_one_original_reservation_and_never_retries_buy():
    frame, dates, pack, trends = fixture(sell_from=None)
    frame.loc[61, ["open", "high"]] = [12., 12.1]
    prepared = prepare(frame, dates, pack, trends)
    evaluations = []
    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="one-shot",
        entry_evaluator=lambda snapshot, candidate: (evaluations.append(snapshot["date"]) or score(snapshot)))
    assert evaluations == [dates[60]]
    assert len(result["final_action_intents"]) == len(result["rejections"]) == 1
    assert result["ledger"] == [] and result["metrics"]["open_shares"] == 0
    assert result["terminal_state"]["reserved_opportunity_ids"] == result["entry_diagnostics"][0]["original_opportunity_ids"]
    resumed = replay_trend_account(frame, dates, prepared, initial_state=result["account_snapshots"][4], mode="final",
        entry_evaluator=lambda snapshot, candidate: pytest.fail("reserved original must not be evaluated again"), run_id="resumed")
    assert resumed["ledger"] == [] and resumed["entry_diagnostics"] == []


def test_future_calendar_or_run_lineage_does_not_change_old_attempt_but_known_cash_does():
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    original = replay_trend_account(frame, dates, prepared, start=dates[60], end=dates[62], mode="final", run_id="original")
    calendar = [*dates, "2027-01-04"]
    changed = replay_trend_account(frame, calendar, prepared, start=dates[60], end=dates[62], mode="final", run_id="new-run")
    for old, new in zip(original["entry_diagnostics"], changed["entry_diagnostics"], strict=True):
        assert old["reference_state_hash"] != new["reference_state_hash"]
        assert old["flat_state_hash"] == new["flat_state_hash"] and old["attempt_id"] == new["attempt_id"]
    state = copy.deepcopy(original["entry_diagnostics"][0]["pre_gate_state"])
    state["cash"] = 90000.
    assert _flat_state_hash(state) != original["entry_diagnostics"][0]["flat_state_hash"]


def test_new_unreserved_source_does_not_borrow_old_reserved_opportunity_stop():
    frame, dates, pack, trends = fixture(stop=9.5)
    new = copy.deepcopy(pack["events"][0])
    new.update(event_id="new-buy", original_event_id="new-buy", pattern_id="new-pattern", date=dates[70],
        available_at=dates[70]+"T15:00:00+08:00", first_available_at=dates[70]+"T15:00:00+08:00",
        first_confirmed_at=dates[70]+"T15:00:00+08:00", risk_stop_price=9.)
    pack["events"].append(new)
    for i in range(70, len(dates)):
        pack["evaluations"][i]["structure"]["support_facts"].append(copy.deepcopy(new))
        if i < 75:
            pack["evaluations"][i]["structure"]["current_state"].update(buy_score=.9, sell_score=0.)
            pack["evaluations"][i]["wyckoff"]["current_state"].update(buy_score=.65, sell_score=.05)
            trends[i].update(p_up=.55, p_down=.1, p_neutral=.35)  # Continuous support without a new MA cross.
    prepared = prepare(frame, dates, pack, trends)
    assert prepared["rows"][70]["entry_plan"]["stop_price"] == 9.5
    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E0", run_id="new-source-only")
    second = next(row for row in result["entry_diagnostics"] if row["date"] == dates[70])
    assert second["candidate"]["entry_plan"]["stop_price"] == 9.
    assert second["candidate"]["entry_plan"]["source_event_ids"] == ["new-buy"]
    assert second["original_opportunity_ids"] == ["structure:new-pattern:BUY"]
    assert [row["action"] for row in result["ledger"]] == ["BUY", "SELL", "BUY", "SELL"]


def test_original_expiry_clock_and_permanent_boundary_break_do_not_resurrect():
    frame, dates, pack, trends = fixture(sell_from=None)
    delayed = copy.deepcopy(pack["events"][0]); delayed.update(date=dates[70], available_at=dates[70]+"T15:00:00+08:00")
    pack["events"] = [delayed]
    result = prepare(frame, dates, pack, trends)
    assert result["rows"][70]["entry_eligible"] and not result["rows"][80]["entry_eligible"]
    frame, dates, pack, trends = fixture(prices={61:9.2}, sell_from=None)
    result = prepare(frame, dates, pack, trends)
    assert result["rows"][60]["entry_eligible"]
    assert not result["rows"][61]["active_candidates"]
    assert not result["rows"][62]["active_candidates"]  # Rebound must not revive the broken original boundary.


@pytest.mark.parametrize("family,in_scope,reason", [("czsc_trend_consensus", True, "identity_error"),
    (STRATEGY_ID, False, "high_return_population_out_of_scope")])
def test_old_family_or_out_of_scope_probability_never_bypasses_entry_gate(family, in_scope, reason):
    frame, dates, pack, trends = fixture()
    result = replay_trend_account(frame, dates, prepare(frame, dates, pack, trends), start=dates[60], mode="final",
        entry_evaluator=lambda snapshot, candidate: score(snapshot, .9, family=family, in_scope=in_scope), run_id="reject-contract")
    assert result["ledger"] == []
    assert all(row["reason"] == reason and not row["allowed"] for row in result["entry_diagnostics"])
    assert result["terminal_state"]["reserved_opportunity_ids"] == []


def test_actual_stop_overrides_continuous_sell_and_flat_sell_creates_no_broker_trade():
    frame, dates, pack, trends = fixture(prices={65:9.1, **{i:9.1 for i in range(66,85)}})
    frame.loc[65, ["open", "high"]] = [9.35, 9.4]
    prepared = prepare(frame, dates, pack, trends)
    actual = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E0", run_id="stop-priority")
    sell = actual["ledger"][-1]
    assert sell["reason"] == "actual_account_stop" and "ordinary_consensus_sell" in sell["all_triggered_reasons"]
    flat = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="flat-no-model")
    assert flat["ledger"] == [] and flat["final_action_intents"] == []
    assert len(flat["entry_diagnostics"]) > 1
