"""New native policy must preserve scope, causal state and original event clocks."""
import copy

import pytest

from my_strategy.services.czsc_agent_consensus import build_consensus, consensus_config
from my_strategy.services import czsc_agent_consensus_v3 as frozen_v3
from my_strategy.services import czsc_agent_consensus_v4 as v4
from my_strategy.tests.test_czsc_agent_consensus_v3 import _v3_row, SYMBOL


def _row(index=0, specs=(("wyckoff", "BUY"),), **kwargs):
    row = _v3_row(index, specs, **kwargs)
    bundle = row["candidate_evidence"]
    bundle["version"] = "czsc_candidate_evidence_v4"
    bundle["provenance"].update(candidate_policy_version=v4.CANDIDATE_POLICY,
        candidate_policy_hash="candidatehash", risk_policy_hash="riskhash")
    for event in bundle["facts"]:
        event.update(pattern_id=event["event_id"], original_event_id=event["event_id"],
                     first_confirmed_at=event["available_at"], activation_kind="confirmation",
                     confirmed=True, invalidation_price=95. if event["side"] == "BUY" else 105.,
                     candidate_direction_valid=True)
    for source in ("ma", "structure", "wyckoff"):
        evaluation = bundle["evaluations"][source]
        state = evaluation["direction_state"]
        if source != "ma":
            state["version"] = "czsc_native_evidence_v4_20261012"
            state["support_facts"] = [event for event in bundle["facts"] if event["source"] == source]
            if source == "structure":
                state["current_state"]["resistance_valid"] = state["sell_valid"]
            if source == "wyckoff":
                state["current_state"]["continuous_direction_valid"] = state["current_state"]["current_direction_valid"]
        else:
            evaluation["model_identity"]["candidate_policy_version"] = v4.CANDIDATE_POLICY
            evaluation["candidate_envelope"] = {"version": v4.CANDIDATE_ENVELOPE,
                "in_scope": True, "candidate_event_ids": [event["event_id"] for event in bundle["facts"]]}
    row["agents"]["high_return"].update(in_scope=True)
    row["agents"]["high_return"]["model_identity"]["candidate_policy_version"] = v4.CANDIDATE_POLICY
    return row


def _build(rows, config=None):
    return build_consensus(rows, symbol=SYMBOL, version=v4.CONSENSUS_VERSION, config=config)


def test_new_policy_weights_and_frozen_v3_replay_independent_of_current_default(monkeypatch):
    values = consensus_config(version=v4.CONSENSUS_VERSION)
    assert values["source_weights"] == {"ma": .4, "structure": .2, "wyckoff": .3}
    assert values["high_return_quality_weight"] == .1
    import my_strategy.services.czsc_agent_consensus as router
    monkeypatch.setattr(router, "load_config", lambda _: {"version": v4.CONSENSUS_VERSION})
    old = [_v3_row(specs=[("wyckoff", "BUY")])]
    assert build_consensus(old, symbol=SYMBOL, version=frozen_v3.CONSENSUS_VERSION) == frozen_v3.build_consensus(old, symbol=SYMBOL)


def test_new_union_scope_gives_verified_ma_40_and_independent_wyckoff_30():
    row = _row()
    decision = _build([row])["rows"][0]
    assert decision["action"] == "BUY"
    assert decision["buy_base_score"] == .7 and decision["buy_total_score"] == .8
    assert decision["buy_independent_families"] == ["ma", "wyckoff"]


@pytest.mark.parametrize("status,expected", [("no_candidate", "no_candidate"),
    ("candidate_risk_ineligible", "risk_blocked_unscored"), ("feature_ineligible", "input_ineligible"),
    ("model_unavailable", "model_unavailable"), ("out_of_scope", "out_of_scope")])
def test_unscored_quality_preserves_null_and_raw_cause_before_identity_checks(status, expected):
    row = _row(l2=None, risk=False)
    row["agents"]["high_return"].update(status=status, reason_codes=["actual_raw_cause"], model_identity={})
    agent = _build([row])["rows"][0]["agents"]["high_return"]
    assert agent["status"] == expected and agent["raw_status"] == status
    assert agent["probability"] is None and agent["reason_codes"] == ["actual_raw_cause"]


def test_scored_zero_is_legitimate_rejection_and_actual_contract_error_remains_error():
    row = _row(l2=0)
    quality = _build([row])["rows"][0]["agents"]["high_return"]
    assert quality["status"] == "rejected" and quality["probability"] == 0
    row["agents"]["high_return"]["model_identity"]["risk_plan_policy_hash"] = "changed-risk"
    assert _build([row])["rows"][0]["agents"]["high_return"]["status"] == "identity_error"


@pytest.mark.parametrize("scope", [None, {"version": v4.CANDIDATE_ENVELOPE, "in_scope": False},
    {"version": "old_ma_population", "in_scope": True, "candidate_event_ids": ["x"]},
    {"version": v4.CANDIDATE_ENVELOPE, "in_scope": True, "candidate_event_ids": ["other"]}])
def test_old_daily_ma_score_cannot_borrow_cross_source_population_for_full_vote(scope):
    row = _row()
    row["candidate_evidence"]["evaluations"]["ma"]["candidate_envelope"] = scope
    decision = _build([row])["rows"][0]
    assert decision["agents"]["ma"]["buy_contribution"] == 0
    assert decision["buy_base_score"] == .3 and decision["action"] == "WAIT"


def test_bullish_long_trend_and_short_weakening_remain_distinct_without_inverted_sell_score():
    row = _row()
    state = row["candidate_evidence"]["evaluations"]["ma"]["direction_state"]
    state["sell_valid"] = True
    state["current_state"].update(ma_sell=True, ma5_ma10_gap=-.004546, ret_5=-.07438)
    decision = _build([row])["rows"][0]
    ma = decision["agents"]["ma"]
    assert ma["judgment"] == "BUY" and ma["trend_state"] == "bullish" and ma["tactical_weakening"] is True
    assert ma["sell_contribution"] == 0 and ma["exit_probability"] is None
    row["candidate_evidence"]["evaluations"]["ma"]["probability"] = .01
    decision = _build([row])["rows"][0]
    assert decision["agents"]["ma"]["judgment"] == "ABSTAIN"
    assert decision["sell_base_score"] == 0


def test_current_continuous_native_state_is_independent_without_same_day_event_metadata():
    row = _row(specs=[("structure", "BUY")], p=None)
    row["candidate_evidence"]["evaluations"]["ma"]["status"] = "model_unavailable"
    w = row["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    w.update(buy_valid=True, sell_valid=False, origin_event_ids=[])
    w["current_state"].update(buy_score=.95, sell_score=.05, continuous_direction_valid=True,
                             wyckoff_observed_at=None, wyckoff_range_formed_at=None)
    decision = _build([row])["rows"][0]
    assert decision["action"] == "BUY"
    assert decision["buy_base_score"] == .5 and decision["buy_total_score"] == .6
    assert decision["buy_independent_families"] == ["structure", "wyckoff"]
    assert decision["agents"]["wyckoff"]["evidence_mode"] == "current_snapshot"


def test_neutral_zero_rule_strength_is_valid_input_not_missing_data():
    row = _row(specs=[])
    w = row["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    w["current_state"].update(buy_score=0, sell_score=0, continuous_direction_valid=True)
    agent = _build([row])["rows"][0]["agents"]["wyckoff"]
    assert agent["status"] == "not_supported" and agent["input_eligible"] is True
    # Neutral native snapshots retain observed zeroes for the UI.
    assert agent["current_state"]["buy_score"] == 0


def test_same_pattern_cannot_repeat_after_risk_rejection_but_one_new_eligibility_activation_can():
    first = _row(risk=False, l2=None)
    first["agents"]["high_return"]["status"] = "candidate_risk_ineligible"
    retry = _row(1)
    retry["candidate_evidence"]["facts"][0]["pattern_id"] = first["candidate_evidence"]["facts"][0]["pattern_id"]
    result = _build([first, retry], config={"cooldown_bars": 0})
    assert result["events"] == []
    assert any("pattern_already_assessed" in item["reason_codes"] for item in result["rows"][1]["ignored_events"])
    retry["candidate_evidence"]["facts"][0].update(activation_kind="first_risk_eligibility",
        original_event_id=first["candidate_evidence"]["facts"][0]["event_id"],
        first_confirmed_at=first["date"] + "T15:00:00+08:00")
    assert _build([first, retry], config={"cooldown_bars": 0})["events"][0]["date"] == retry["date"]


def test_aged_structure_support_remains_valid_but_cannot_trigger_new_candidate_alone():
    first = _row(specs=[("structure", "BUY")], risk=False)
    later = _row(5, specs=[("wyckoff", "BUY")])
    structure = later["candidate_evidence"]["evaluations"]["structure"]["direction_state"]
    structure.update(buy_valid=True, sell_valid=False, support_facts=first["candidate_evidence"]["facts"],
        origin_event_ids=[first["candidate_evidence"]["facts"][0]["event_id"]])
    structure["current_state"].update(support_valid=True,
        structure_confirmed_at=first["date"] + "T15:00:00+08:00")
    rows = [first] + [_row(index, specs=[]) for index in range(1, 5)] + [later]
    result = _build(rows, config={"evidence_valid_bars": 1, "cooldown_bars": 0})
    assert [event["date"] for event in result["events"]] == [later["date"]]
    assert result["rows"][-1]["agents"]["structure"]["buy_contribution"] == .2
    assert result["rows"][-1]["agents"]["structure"]["evidence_mode"] == "continuous_state"


def test_original_boundary_invalidation_cannot_be_resurrected_by_fresh_id_or_continuous_score():
    first = _row(risk=False)
    event = first["candidate_evidence"]["facts"][0]
    invalidation = _row(1, specs=[])
    w = invalidation["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    w["invalidated_event_ids"] = [event["event_id"]]
    retry = _row(2)
    retry["candidate_evidence"]["facts"][0]["original_event_id"] = event["event_id"]
    assert not _build([first, invalidation, retry], config={"cooldown_bars": 0})["events"]


def test_future_support_origin_cannot_vote_with_claimed_present_snapshot():
    row = _row(specs=[("structure", "BUY")], p=None)
    row["candidate_evidence"]["evaluations"]["ma"]["status"] = "model_unavailable"
    row["candidate_evidence"]["evaluations"]["structure"]["direction_state"]["support_facts"][0]["available_at"] = "2026-01-06"
    decision = _build([row])["rows"][0]
    assert decision["agents"]["structure"]["buy_contribution"] == 0
    assert decision["agents"]["structure"]["status"] == "input_ineligible"


def test_raw_direction_counts_preserve_risk_rejected_two_structure_facts():
    row = _row(specs=[("structure", "BUY"), ("structure", "BUY")], risk=False, l2=None)
    for event in row["candidate_evidence"]["facts"]:
        event["candidate_eligible"] = False
    row["agents"]["high_return"]["status"] = "candidate_risk_ineligible"
    decision = _build([row])["rows"][0]
    assert decision["agents"]["structure"]["raw_direction_counts"] == {"buy": 2, "sell": 0}
    assert decision["agents"]["structure"]["risk_eligible_counts"] == {"buy": 0, "sell": 0}


def test_input_immutability_and_full_prefix_invariance():
    rows = [_row(), _row(1, specs=[])]
    snapshot = copy.deepcopy(rows)
    short = _build(rows[:1])
    long = _build(rows)
    assert rows == snapshot and short["rows"] == long["rows"][:1]
    assert short["events"] == long["events"]


def test_pure_structure_sell_uses_resistance_and_is_not_blocked_by_daily_buy_invalidation():
    row = _row(specs=[("ma", "SELL"), ("structure", "SELL")], ma_side="SELL", p=None)
    state = row["candidate_evidence"]["evaluations"]["structure"]["direction_state"]
    state["current_state"].update(support_valid=False, resistance_valid=True, invalidated=True)
    decision = _build([row])["rows"][0]
    assert decision["agents"]["structure"]["judgment"] == "SELL"
    assert decision["sell_base_score"] == .6 and decision["action"] == "SELL"
    assert decision["agents"]["high_return"]["status"] == "no_candidate"


def test_foreign_source_invalidation_cannot_poison_another_source_original():
    first = _row(risk=False)
    first_event = first["candidate_evidence"]["facts"][0]
    first_event.update(confirmed=True, event_strength=.9, invalidation_price=95.)
    middle = _row(1, specs=[])
    middle["candidate_evidence"]["evaluations"]["structure"]["direction_state"]["invalidated_event_ids"] = [first_event["event_id"]]
    next_row = _row(2, specs=[("structure", "BUY")])
    w = next_row["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    w.update(buy_valid=True, sell_valid=False, support_facts=[first_event])
    w["current_state"].update(buy_score=.9, sell_score=.1, continuous_direction_valid=True)
    result = _build([first, middle, next_row], config={"cooldown_bars": 0})
    assert result["events"][0]["date"] == next_row["date"]
    assert result["rows"][-1]["agents"]["wyckoff"]["origin_event_ids"] == [first_event["event_id"]]


def test_confirmed_valid_lps_can_support_when_current_continuous_is_neutral():
    row = _row()
    w = row["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    event = w["support_facts"][0]
    event.update(confirmed=True, event_type="LPS", event_strength=1., invalidation_price=95.)
    w.update(buy_valid=False, sell_valid=False)
    w["current_state"].update(buy_score=.153, sell_score=.085, continuous_direction_valid=True)
    decision = _build([row])["rows"][0]
    assert decision["action"] == "BUY" and decision["agents"]["wyckoff"]["buy_contribution"] == .3
    assert decision["agents"]["wyckoff"]["strength"] == {"buy": .153, "sell": .085}
    assert decision["agents"]["wyckoff"]["event_strength"]["buy"] == 1
    assert "wyckoff_valid_confirmed_event_support" in decision["agents"]["wyckoff"]["reason_codes"]


@pytest.mark.parametrize("confirmed,strength,boundary", [(False, 1., 95.), (True, .49, 95.), (True, 1., 101.)])
def test_pending_weak_or_boundary_broken_event_cannot_borrow_event_support(confirmed, strength, boundary):
    row = _row()
    w = row["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    w["support_facts"][0].update(confirmed=confirmed, event_strength=strength, invalidation_price=boundary)
    w.update(buy_valid=False, sell_valid=False)
    w["current_state"].update(buy_score=.153, sell_score=.085)
    assert _build([row])["rows"][0]["agents"]["wyckoff"]["judgment"] == "ABSTAIN"


def test_confirmed_buy_event_and_qualified_supply_disagree_within_one_w_budget():
    row = _row()
    w = row["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    w["support_facts"][0].update(confirmed=True, event_strength=1., invalidation_price=95.)
    w.update(buy_valid=False, sell_valid=True)
    w["current_state"].update(buy_score=.1, sell_score=.8, continuous_direction_valid=True)
    decision = _build([row])["rows"][0]
    assert decision["agents"]["wyckoff"]["judgment"] == "CONFLICT"
    assert decision["agents"]["wyckoff"]["buy_contribution"] == 0 and decision["agents"]["wyckoff"]["sell_contribution"] == 0
    assert decision["family_conflicts"] == ["wyckoff"]


def test_current_long_failure_supports_other_fresh_sell_after_original_candidate_window():
    first = _row(specs=[("ma", "SELL")], ma_side="SELL", p=None)
    rows = [first] + [_row(index, specs=[], ma_side=None) for index in range(1, 5)]
    later = _row(5, specs=[("structure", "SELL")], ma_side="SELL", p=.01, l2=None)
    rows.append(later)
    result = _build(rows, config={"evidence_valid_bars": 1})
    assert [event["date"] for event in result["events"]] == [later["date"]]
    ma = result["rows"][-1]["agents"]["ma"]
    assert ma["judgment"] == "SELL" and ma["sell_contribution"] == .4
    assert ma["type"] == "causal_rule_evidence" and ma["exit_probability"] is None
    assert ma["evidence_mode"] == "current_snapshot" and ma["origin_event_ids"] == []


def test_causal_long_failure_snapshot_alone_does_not_manufacture_fresh_sell():
    row = _row(specs=[], ma_side="SELL", p=.99, l2=None)
    decision = _build([row])["rows"][0]
    assert decision["agents"]["ma"]["judgment"] == "SELL"
    assert decision["agents"]["ma"]["sell_contribution"] == .4
    assert decision["candidate_event_ids"] == [] and decision["action"] == "WAIT"


def _add_independent_current_buy_states(row):
    """Qualified opinions exist already; only the original MA fact is new."""
    structure = row["candidate_evidence"]["evaluations"]["structure"]["direction_state"]
    structure.update(buy_valid=True, sell_valid=False)
    structure["current_state"].update(support_valid=True, resistance_valid=False,
        support_floor=95., structure_confirmed_at="2025-12-30T15:00:00+08:00")
    w = row["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    w.update(buy_valid=True, sell_valid=False)
    w["current_state"].update(buy_score=.8, sell_score=.1, continuous_direction_valid=True)
    return row


@pytest.mark.parametrize("probability", [.466, .0, None])
def test_valid_raw_ma_candidate_survives_low_or_missing_ma_vote_when_other_experts_pass(probability):
    row = _add_independent_current_buy_states(_row(specs=[("ma", "BUY")], p=probability, l2=.1127))
    event = row["candidate_evidence"]["facts"][0]
    event["event_type"] = "ma_trend_onset"
    decision = _build([row])["rows"][0]
    assert decision["agents"]["ma"]["judgment"] == "ABSTAIN"
    assert decision["agents"]["ma"]["buy_contribution"] == 0
    assert decision["buy_base_score"] == .5 and decision["buy_total_score"] == .6
    assert decision["buy_independent_families"] == ["structure", "wyckoff"]
    assert decision["funnel"]["fresh_candidates"] == 1 and decision["action"] == "BUY"
    assert decision["trigger_event_ids"] == [event["event_id"]]
    assert event["event_id"] not in decision["evidence_event_ids"]
    assert _build([row])["events"][0]["trigger_event_ids"] == [event["event_id"]]


def test_invalid_raw_ma_long_trend_cannot_borrow_other_sources_to_create_buy():
    row = _add_independent_current_buy_states(_row(specs=[("ma", "BUY")], p=.466, l2=.1127))
    state = row["candidate_evidence"]["evaluations"]["ma"]["direction_state"]
    state["current_state"].update(ma_buy=False, ma20_slope5=-.1)
    decision = _build([row])["rows"][0]
    assert decision["proposed_action"] == "BUY" and decision["buy_total_score"] == .6
    assert decision["funnel"]["fresh_candidates"] == 0 and decision["action"] == "WAIT"
    assert any("fresh_ma_long_trend_not_currently_valid" in item["reason_codes"] for item in decision["ignored_events"])


def test_invalid_raw_native_boundary_cannot_borrow_positive_continuous_or_other_sources():
    row = _row()
    _add_independent_current_buy_states(row)
    w = row["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    w["support_facts"][0]["invalidation_price"] = 101.
    decision = _build([row])["rows"][0]
    assert decision["proposed_action"] == "BUY" and decision["buy_base_score"] == .9
    assert decision["funnel"]["fresh_candidates"] == 0 and decision["action"] == "WAIT"
    assert any("fresh_original_native_pattern_not_currently_valid" in item["reason_codes"] for item in decision["ignored_events"])


def test_valid_raw_native_candidate_survives_its_own_weak_opinion_when_ma_and_structure_pass():
    row = _row(l2=.1067)
    _add_independent_current_buy_states(row)
    w = row["candidate_evidence"]["evaluations"]["wyckoff"]["direction_state"]
    w.update(buy_valid=False, sell_valid=False)
    w["support_facts"][0].update(event_strength=.49, confirmed=True)
    w["current_state"].update(buy_score=.1, sell_score=.1)
    decision = _build([row])["rows"][0]
    assert decision["agents"]["wyckoff"]["judgment"] == "ABSTAIN" and decision["agents"]["wyckoff"]["buy_contribution"] == 0
    assert decision["buy_base_score"] == .6 and decision["buy_independent_families"] == ["ma", "structure"]
    assert decision["funnel"]["fresh_candidates"] == 1 and decision["action"] == "BUY"


def test_no_raw_fact_still_cannot_emit_from_two_qualified_current_native_snapshots():
    row = _add_independent_current_buy_states(_row(specs=[]))
    decision = _build([row], config={"source_weights": {"ma": 0., "structure": .6, "wyckoff": .3}})["rows"][0]
    assert decision["buy_base_score"] == .9 and decision["proposed_action"] == "BUY"
    assert decision["action"] == "WAIT" and decision["candidate_event_ids"] == [] and decision["trigger_event_ids"] == []
    assert "no_fresh_candidate_trigger" in decision["reason_codes"]


def test_risk_blocked_raw_candidate_remains_visible_but_unscored_quality_cannot_buy():
    row = _add_independent_current_buy_states(_row(specs=[("ma", "BUY")], p=.8, l2=None, risk=False))
    row["candidate_evidence"]["facts"][0]["candidate_eligible"] = False
    row["agents"]["high_return"].update(status="candidate_risk_ineligible", model_identity={},
        reason_codes=["stop_distance_exceeds_frozen_maximum"])
    decision = _build([row])["rows"][0]
    assert decision["funnel"]["fresh_candidates"] == 1 and decision["proposed_action"] == "BUY"
    assert decision["buy_base_score"] == .9 and decision["action"] == "WAIT"
    assert decision["agents"]["high_return"]["probability"] is None
    assert decision["agents"]["high_return"]["status"] == "risk_blocked_unscored"
    assert decision["agents"]["high_return"]["reason_codes"] == ["stop_distance_exceeds_frozen_maximum"]
    assert "entry_risk_gate_rejected" in decision["reason_codes"]


@pytest.mark.parametrize("field,value", [("candidate_state_valid", False), ("candidate_state_date", "2026-01-04"),
    ("candidate_state_available_at", "2026-01-06T15:00:00+08:00"), ("candidate_state_available_at", "2026-01-04T15:00:00+08:00")])
def test_raw_state_validation_flag_requires_same_day_observable_evidence(field, value):
    row = _add_independent_current_buy_states(_row(specs=[("ma", "BUY")], p=.466))
    event = row["candidate_evidence"]["facts"][0]
    event.update(candidate_state_valid=True, candidate_state_date=row["date"], candidate_state_available_at=row["date"]+"T15:00:00+08:00")
    event[field] = value
    decision = _build([row])["rows"][0]
    assert decision["action"] == "WAIT" and decision["funnel"]["fresh_candidates"] == 0
