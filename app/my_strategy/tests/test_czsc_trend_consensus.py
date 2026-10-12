"""Continuous budgets, independent evidence and irreversible opportunity clocks."""
import copy

import pandas as pd
import pytest

from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions, consensus_config


SYMBOL = "002436.SZ"


def identity(head="trend_direction"):
    return {"family": "czsc_trend_consensus", "head": head, "verified_frozen_contract": True,
        "checkpoint": "prior", "manifest_sha256": "manifest", "feature_schema_hash": "schema",
        "label_contract_hash": "target", "train_label_end": "2024-12-31", "calibration_label_end": "2025-01-31",
        "policy_label_end": "2025-02-28", "simulated_available_at": "2025-03-01T15:00:00+08:00"}


def fixture(side="BUY", probabilities=(.4658, .2, .3342), strength=.8):
    dates = pd.bdate_range("2026-01-01", periods=25).strftime("%Y-%m-%d").tolist()
    frame = pd.DataFrame([{"symbol": SYMBOL, "date": d, "close": 10., "high": 10.2, "low": 9.8,
        "close_above_ma20": 1., "ma20_slope5": .1, "ma_alignment": 1., "ret_5": .1} for d in dates])
    day = dates[-1]; at = day + "T15:00:00+08:00"
    fact = {"symbol": SYMBOL, "date": day, "available_at": at, "first_available_at": at, "first_confirmed_at": at,
        "event_id": "raw", "original_event_id": "raw", "pattern_id": "original-pattern", "source": "structure", "side": side,
        "confirmed": True, "input_eligible": True, "invalidation_price": 9. if side == "BUY" else 11., "risk_stop_price": 9.}
    structure = {"date": day, "available_at": at, "source_id": "structure", "version": "pure_confirmed_czsc_v4_20261012",
        "input_eligible": True, "support_facts": [fact], "current_state": {"close": 10., "buy_score": strength if side == "BUY" else 0.,
        "sell_score": strength if side == "SELL" else 0., "support_valid": True, "support_floor": 9., "resistance_valid": True, "resistance_ceiling": 11.}}
    wyckoff = {"date": day, "available_at": at, "source_id": "w", "version": "wyckoff_current_pv_v4_20261012",
        "input_eligible": True, "support_facts": [], "current_state": {"close": 10., "buy_score": .8 if side == "BUY" else .05, "sell_score": .05 if side == "BUY" else .8}}
    trend = [{"date": d, "available_at": d + "T15:00:00+08:00", "p_up": probabilities[0], "p_down": probabilities[1], "p_neutral": probabilities[2],
        "input_eligible": True, "model_available": True, "model_identity": identity()} for d in dates]
    pack = {"events": [fact], "evaluations": [{"date": day, "available_at": at, "structure": structure, "wyckoff": wyckoff}]}
    return frame, dates, pack, trend


def test_sub_55_probability_contributes_continuously_and_quality_11pct_is_1point1():
    frame, dates, pack, trend = fixture()
    quality = [{"date": dates[-1], "available_at": dates[-1] + "T15:00:00+08:00", "status": "scored", "probability": .11,
        "in_scope": True, "model_identity": identity("entry")}]
    result = prepare_trend_decisions(frame, dates, pack, trend, quality=quality, mode="final")
    row = result["rows"][-1]
    assert row["agents"]["ma"]["buy_contribution"] == pytest.approx(.4 * .4658)
    assert row["buy_quality_bonus"] == pytest.approx(.011)
    assert row["action"] == "BUY"
    assert row["buy_total_score"] < 1.


def test_sell_margin_never_uses_high_return_probability_or_requires_quality():
    frame, dates, pack, trend = fixture("SELL", (.1, .7, .2))
    a = prepare_trend_decisions(frame, dates, pack, trend, mode="final")
    b = prepare_trend_decisions(frame, dates, pack, trend, quality=[{"date": dates[-1], "probability": 1., "status": "scored"}], mode="final")
    assert a["rows"][-1]["action"] == b["rows"][-1]["action"] == "SELL"
    assert a["rows"][-1]["sell_base_score"] == b["rows"][-1]["sell_base_score"]
    assert b["rows"][-1]["agents"]["high_return"]["status"] == "no_candidate"


def test_opposing_native_strengths_share_one_budget_and_assisted_structure_not_independent():
    frame, dates, pack, trend = fixture()
    s = pack["evaluations"][-1]["structure"]
    s["current_state"].update(buy_score=1., sell_score=1.)
    s["auxiliary"] = True
    row = prepare_trend_decisions(frame, dates, pack, trend)["rows"][-1]
    assert row["agents"]["ma"]["buy_contribution"] + row["agents"]["ma"]["sell_contribution"] + row["agents"]["structure"]["buy_contribution"] + row["agents"]["structure"]["sell_contribution"] <= .4 + 1e-12
    assert "structure" not in row["buy_independent_families"]


def test_missing_is_null_and_not_down_probability_complement():
    frame, dates, pack, trend = fixture(probabilities=(.65, .1, .25))
    trend[-1].update(p_up=None, p_down=None, p_neutral=None, model_available=False)
    row = prepare_trend_decisions(frame, dates, pack, trend)["rows"][-1]
    assert row["agents"]["ma"]["sell_strength"] is None
    assert row["agents"]["ma"]["buy_contribution"] is None
    assert row["agents"]["high_return"]["probability"] is None
    assert row["agents"]["high_return"]["status"] == "bootstrap_not_evaluated"


def test_future_original_confirmation_cannot_borrow_today_support():
    frame, dates, pack, trend = fixture()
    pack["events"][0]["first_confirmed_at"] = "2030-01-01T15:00:00+08:00"
    assert prepare_trend_decisions(frame, dates, pack, trend)["rows"][-1]["action"] == "WAIT"


def test_broken_original_w_boundary_cannot_borrow_strong_continuous_opinion():
    frame, dates, pack, trend = fixture()
    fact = pack["events"][0]
    fact.update(source="wyckoff", invalidation_price=11., risk_stop_price=9., event_strength=1.)
    pack["evaluations"][-1]["structure"]["support_facts"] = []
    pack["evaluations"][-1]["wyckoff"]["support_facts"] = [fact]
    assert prepare_trend_decisions(frame, dates, pack, trend)["rows"][-1]["action"] == "WAIT"


def test_model_checkpoint_switch_is_not_cross_and_staying_high_is_not_daily_event():
    frame, dates, pack, trend = fixture(probabilities=(.7, .1, .2))
    pack["events"] = []; pack["evaluations"] = []
    trend[-2].update(p_up=.55, p_down=.2, p_neutral=.25)
    trend[-1]["model_identity"] = {**identity(), "checkpoint": "new", "manifest_sha256": "new"}
    result = prepare_trend_decisions(frame, dates, pack, trend)
    assert not result["events"]
    assert not any(row["candidate_evidence"] for row in result["rows"])


def test_true_same_head_cross_produces_once_with_causal_current_stamp():
    frame, dates, pack, trend = fixture(probabilities=(.7, .1, .2))
    pack["events"] = []
    trend[-2].update(p_up=.55, p_down=.2, p_neutral=.25)
    result = prepare_trend_decisions(frame, dates, pack, trend)
    facts = result["rows"][-1]["candidate_evidence"]
    assert len(facts) == 1 and facts[0]["event_type"] == "trend_probability_cross"
    assert facts[0]["available_at"].startswith(dates[-1])


def test_flat_config_round_trip_and_rejects_legacy_shape():
    cfg = consensus_config({"buy_base_threshold": .35})
    frame, dates, pack, trend = fixture()
    assert prepare_trend_decisions(frame, dates, pack, trend, config=cfg)["config"]["buy_base_threshold"] == .35
    with pytest.raises(ValueError): consensus_config({"version": "czsc_agent_consensus_v4"})
    with pytest.raises(ValueError): consensus_config({"ma_entry_threshold": .55})


def test_provisional_policy_head_is_only_usable_in_explicit_training_window():
    frame, dates, pack, trend = fixture()
    trend[-1]["usage_mode"] = "policy_validation"
    trend[-1]["model_identity"].update(provisional_checkpoint=True, policy_start=dates[-2], policy_end=dates[-1], policy_label_end=dates[-1])
    blocked = prepare_trend_decisions(frame, dates, pack, trend)
    assert blocked["rows"][-1]["agents"]["ma"]["status"] == "unavailable"
    allowed = prepare_trend_decisions(frame, dates, pack, trend, allow_policy_validation=True)
    assert allowed["rows"][-1]["agents"]["ma"]["status"] == "scored"
    trend[-1]["model_identity"]["calibration_label_end"] = dates[-1]
    assert prepare_trend_decisions(frame, dates, pack, trend, allow_policy_validation=True)["rows"][-1]["agents"]["ma"]["status"] == "unavailable"


def test_full_vs_prefix_scores_events_opportunity_identity_equal():
    frame, dates, pack, trend = fixture()
    early = dates[-2]
    pack["events"][0].update(date=early, available_at=early+"T15:00:00+08:00", first_confirmed_at=early+"T15:00:00+08:00", first_available_at=early+"T15:00:00+08:00")
    previous = copy.deepcopy(pack["evaluations"][0]); previous.update(date=early, available_at=early+"T15:00:00+08:00")
    for role in ("structure", "wyckoff"):
        previous[role].update(date=early, available_at=early+"T15:00:00+08:00")
    pack["evaluations"].insert(0, previous)
    full = prepare_trend_decisions(frame, dates, pack, trend)
    prefix = prepare_trend_decisions(frame.iloc[:-1], dates, pack, trend)
    assert full["rows"][:-1] == prefix["rows"]
    assert [e for e in full["events"] if e["date"] <= early] == prefix["events"]
