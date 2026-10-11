"""Candidate-centred arbitration: causality, independence and opportunity facts."""
import copy
import datetime as dt

import pytest

from my_strategy.services.czsc_agent_consensus import build_consensus, consensus_config
from my_strategy.services import czsc_agent_consensus_v2
from my_strategy.tests.test_czsc_agent_consensus import SYMBOL, _row

V3 = "czsc_agent_consensus_v3"


def _identity(head="ma_entry", day="2026-01-05"):
    return {"model_run_id": "frozen-run", "checkpoint": "fold-one", "manifest_sha256": "artifacthash",
        "available_at": "2025-12-31", "train_label_end": "2025-12-30", "model_head": head,
        "target_contract_hash": "targethash", "feature_schema_hash": "schemahash",
        "signal_points_config_hash": "candidatehash", "risk_plan_policy_hash": "riskhash",
        "verified_frozen_contract": True, "score_date": day}


def _v3_row(index=0, specs=(), *, ma_side="BUY", origins=None, p=.8, l2=.2, risk=True):
    row = _row(index, specs, p=p, l2=l2, risk=risk)
    day = row["date"]
    facts = copy.deepcopy(row["events"])
    for event in facts:
        event["direction_eligible"] = True
    evaluations = {source: copy.deepcopy(row["agents"][source]) for source in ("ma", "structure", "wyckoff")}
    for source in ("ma", "structure", "wyckoff"):
        own = [event for event in facts if event["source"] == source]
        if origins and source in origins:
            own = origins[source]
        sides = {event["side"] for event in own}
        if source == "ma":
            sides = {ma_side} if ma_side else set()
        current = {"close": 100.}
        if source == "ma":
            current.update(ma_buy=ma_side == "BUY", ma_sell=ma_side == "SELL",
                close_above_ma20=1 if ma_side == "BUY" else -1, ma20_slope5=.1 if ma_side == "BUY" else -.1,
                ma_alignment=1, ret_5=.1 if ma_side == "BUY" else -.1,
                ret_1=.1, ma5_ma10_gap=.1 if ma_side == "BUY" else -.1)
            evaluations[source].update(model_identity=_identity(day=day), score_date=day)
        elif source == "structure":
            current.update(support_valid=bool(sides), invalidated=False, support_floor=95.,
                           resistance_ceiling=105., structure_confirmed_at=day + "T15:00:00+08:00")
        else:
            current.update(current_direction_valid=bool(sides), buy_score=.8 if "BUY" in sides else .1,
                           sell_score=.8 if "SELL" in sides else .1,
                           wyckoff_available_at=day + "T15:00:00+08:00",
                           wyckoff_observed_at=day + "T15:00:00+08:00",
                           wyckoff_range_formed_at="2025-12-30T15:00:00+08:00")
        evaluations[source]["direction_state"] = {"date": day, "available_at": day + "T15:00:00+08:00",
            "input_eligible": True, "source_id": f"{source}-{day}", "buy_valid": "BUY" in sides,
            "sell_valid": "SELL" in sides, "origin_event_ids": [event["event_id"] for event in own],
            "current_state": current}
    row["agents"]["high_return"].update(model_identity=_identity("entry", day), score_date=day)
    row["candidate_evidence"] = {"version": "czsc_candidate_evidence_v1", "facts": facts,
                                 "evaluations": evaluations, "provenance": {"signal_points_config_hash": "candidatehash", "risk_plan_policy_hash": "riskhash"}}
    return row


def _build(rows, **kwargs):
    return build_consensus(rows, symbol=SYMBOL, **kwargs)


def test_v3_default_fixed_weights_and_window():
    config = consensus_config()
    assert config["version"] == V3
    assert config["evidence_valid_bars"] == 3
    assert config["source_weights"] == {"ma": .4, "structure": .2, "wyckoff": .3}
    assert config["high_return_quality_weight"] == .1


def test_one_wyckoff_fact_triggers_full_verified_daily_ma_assessment_and_80_buy():
    row = _v3_row(specs=[("wyckoff", "BUY")], p=.634, l2=.114)
    result = _build([row])
    decision = result["rows"][0]
    assert decision["action"] == "BUY"
    assert decision["buy_base_score"] == .7
    assert decision["buy_total_score"] == .8
    assert decision["agents"]["ma"]["evidence_mode"] == "current_snapshot"
    assert decision["agents"]["ma"]["origin_event_ids"] == []
    assert decision["evidence_event_ids"] == [row["events"][0]["event_id"]]
    event = result["events"][0]
    assert event["source_frequency"] == "日线"
    assert event["first_available_at"] == row["date"] + "T15:00:00+08:00"
    assert event["anchor_dates"] == [row["date"]]
    assert event["consensus"]["raw_risk_plan"] == row["raw_risk_plan"]


def test_risk_rejection_preserves_directional_fact_but_still_vetoes_final_buy():
    row = _v3_row(specs=[("wyckoff", "BUY")], risk=False)
    row["candidate_evidence"]["facts"][0]["candidate_eligible"] = False
    result = _build([row])
    decision = result["rows"][0]
    assert decision["proposed_action"] == "BUY"
    assert decision["action"] == "WAIT"
    assert decision["buy_total_score"] == .8
    assert decision["funnel"]["observable_events"] == 1
    assert "entry_risk_gate_rejected" in decision["reason_codes"]
    assert result["funnel"]["final_emitted"] == 0


@pytest.mark.parametrize("source,field,value", [
    ("ma", "available_at", "2026-01-06"), ("ma", "train_label_end", "2026-01-05"),
    ("ma", "model_head", "holding_exit"), ("ma", "verified_frozen_contract", False),
    ("ma", "manifest_sha256", None), ("ma", "target_contract_hash", None),
    ("ma", "risk_plan_policy_hash", None), ("ma", "score_date", "2026-01-04"),
    ("high_return", "verified_frozen_contract", False), ("high_return", "available_at", "2026-01-06"),
    ("high_return", "train_label_end", "2026-01-05"), ("high_return", "manifest_sha256", None),
])
def test_buy_rejects_incomplete_future_or_incompatible_model_lineage(source, field, value):
    row = _v3_row(specs=[("wyckoff", "BUY")])
    evidence = row["candidate_evidence"]["evaluations"][source] if source == "ma" else row["agents"][source]
    evidence["model_identity"][field] = value
    # MA exact score_date is independently bound too.
    if source == "ma" and field == "score_date":
        evidence["score_date"] = value
    assert not _build([row])["events"]


@pytest.mark.parametrize("field,value", [("date", "2026-01-04"), ("available_at", "2026-01-06"),
                                        ("input_eligible", False), ("source_id", None)])
def test_cross_source_snapshot_requires_exact_day_observable_input(field, value):
    row = _v3_row(specs=[("wyckoff", "BUY")])
    row["candidate_evidence"]["evaluations"]["ma"]["direction_state"][field] = value
    assert not _build([row])["events"]


def test_causal_ma_state_required_and_entry_probability_never_inferred_as_sell():
    row = _v3_row(specs=[("wyckoff", "BUY")])
    row["candidate_evidence"]["evaluations"]["ma"]["direction_state"]["current_state"]["ma20_slope5"] = -.1
    assert not _build([row])["events"]
    low = _v3_row(specs=[("wyckoff", "SELL")], p=.01)
    decision = _build([low])["rows"][0]
    assert decision["agents"]["ma"]["judgment"] == "ABSTAIN"
    assert decision["sell_base_score"] == .3
    assert decision["action"] == "WAIT"


def test_ma_sell_snapshot_always_requires_real_causal_trend_failure_origin():
    row = _v3_row(specs=[("wyckoff", "SELL")], ma_side="SELL", p=.99)
    assert _build([row])["rows"][0]["sell_base_score"] == .3
    row = _v3_row(specs=[("ma", "SELL"), ("wyckoff", "SELL")], ma_side="SELL", p=None)
    assert _build([row])["rows"][0]["action"] == "SELL"
    row["candidate_evidence"]["facts"][0]["event_type"] = "ma_independent_exit_evidence"
    assert not _build([row])["events"]


def test_previous_ma_sell_and_fresh_wyckoff_sell_emit_only_on_second_date():
    first = _v3_row(specs=[("ma", "SELL")], ma_side="SELL", p=None)
    second = _v3_row(1, [("wyckoff", "SELL")], ma_side="SELL",
        origins={"ma": first["events"]}, p=None, l2=None, risk=False)
    result = _build([first, second])
    assert [row["action"] for row in result["rows"]] == ["WAIT", "SELL"]
    assert result["events"][0]["date"] == second["date"]
    assert result["events"][0]["anchor_dates"] == [first["date"], second["date"]]
    assert any(event["age_bars"] == 1 for event in result["rows"][1]["active_evidence"])
    assert _build([first, second], config={"evidence_valid_bars": 1})["events"] == []


def test_no_event_never_emits_from_snapshot_active_support_or_expiry():
    first = _v3_row(specs=[("wyckoff", "BUY")])
    next_row = _v3_row(1, origins={"wyckoff": first["events"]})
    result = _build([first, next_row], config={"cooldown_bars": 0})
    assert [row["action"] for row in result["rows"]] == ["BUY", "WAIT"]
    assert "no_fresh_candidate_trigger" in result["rows"][1]["reason_codes"]
    expired = _v3_row(3, origins={"wyckoff": first["events"]})
    assert len(_build([first, next_row, _v3_row(2), expired], config={"signal_valid_bars": 1})["events"]) == 1


@pytest.mark.parametrize("age,window,expected", [(1, 1, False), (2, 3, True), (3, 3, False), (4, 5, True), (5, 5, False)])
def test_exact_bar_window_boundary(age, window, expected):
    first = _v3_row(specs=[("ma", "SELL")], ma_side="SELL", p=None)
    rows = [first] + [_v3_row(index, ma_side=None) for index in range(1, age)]
    rows.append(_v3_row(age, [("wyckoff", "SELL")], ma_side="SELL", p=None, origins={"ma": first["events"]}))
    result = _build(rows, config={"evidence_valid_bars": window})
    assert bool(result["events"]) is expected


def test_ma_and_assisted_structure_still_one_family_even_across_days():
    first = _v3_row(specs=[("structure", "BUY", ["native_ma_assisted_definition"])])
    second = _v3_row(1, [("ma", "BUY")], origins={"structure": first["events"]})
    result = _build([first, second])
    assert not result["events"]
    assert result["rows"][1]["buy_base_score"] == .4
    assert result["rows"][1]["buy_independent_count"] == 1


def test_new_independent_same_trend_opportunity_can_emit_after_cooldown():
    rows = [_v3_row(index, [("wyckoff", "BUY")]) for index in range(7)]
    result = _build(rows)
    assert [row["date"] for row in result["events"]] == [rows[0]["date"], rows[5]["date"]]
    assert result["rows"][5]["state_before"] == "bullish"
    assert "independent_opportunity_cooldown" in result["rows"][1]["reason_codes"]


def test_consumed_opportunity_with_new_event_id_cannot_repeat_after_cooldown():
    first = _v3_row(specs=[("wyckoff", "BUY")])
    retry = _v3_row(5, [("wyckoff", "BUY")])
    retry["candidate_evidence"]["facts"][0]["opportunity_id"] = first["events"][0]["opportunity_id"]
    rows = [first] + [_v3_row(index) for index in range(1, 5)] + [retry]
    assert len(_build(rows)["events"]) == 1
    assert any("opportunity_already_consumed" in item["reason_codes"] for item in _build(rows)["rows"][-1]["ignored_events"])


def test_duplicate_foreign_future_and_disabled_facts_do_not_trigger():
    row = _v3_row(specs=[("wyckoff", "BUY")])
    row["candidate_evidence"]["facts"].append(copy.deepcopy(row["candidate_evidence"]["facts"][0]))
    assert len(_build([row])["events"]) == 1
    for field, value in (("symbol", "other"), ("available_at", "2026-01-06"), ("input_eligible", False)):
        bad = _v3_row(specs=[("wyckoff", "BUY")])
        bad["candidate_evidence"]["facts"][0][field] = value
        assert not _build([bad])["events"]
    disabled = _v3_row(specs=[("ma", "BUY")])
    assert not _build([disabled], config={"source_weights": {"ma": 0, "structure": .6, "wyckoff": .3}})["events"]


def test_family_conflicts_and_mixed_dependencies_keep_budget_zero():
    row = _v3_row(specs=[("ma", "SELL"), ("structure", "BUY", ["native_ma_assisted_definition"]), ("wyckoff", "BUY")], ma_side="SELL")
    decision = _build([row])["rows"][0]
    assert decision["family_conflicts"] == ["ma"]
    assert decision["buy_base_score"] == .3
    assert not decision["emitted"]


@pytest.mark.parametrize("window", [None, True, 0, 2, 4, 6, 3.1, "bad"])
def test_window_configuration_strictly_validates(window):
    with pytest.raises(ValueError):
        consensus_config({"evidence_valid_bars": window})


def test_v1_v2_replay_unversioned_v2_defaults_and_input_immutability():
    rows = [_row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])]
    config = czsc_agent_consensus_v2.consensus_config()
    assert _build(rows, version="czsc_agent_consensus_v2") == czsc_agent_consensus_v2.build_consensus(rows, symbol=SYMBOL)
    unversioned = {key: value for key, value in config.items() if key != "version"}
    assert _build(rows, config=unversioned) == _build(rows, version="czsc_agent_consensus_v2")
    v3rows = [_v3_row(specs=[("wyckoff", "BUY")])]
    snapshot = copy.deepcopy(v3rows)
    result = _build(v3rows)
    assert v3rows == snapshot
    larger = _build(v3rows + [_v3_row(1)])
    assert larger["rows"][:1] == result["rows"]
    assert larger["events"] == result["events"]


def test_definitive_invalidation_permanent_and_foreign_or_future_invalidation_rejected():
    first = _v3_row(specs=[("ma", "SELL")], ma_side="SELL", p=None)
    intermediate = _v3_row(1, ma_side=None)
    intermediate["candidate_evidence"]["evaluations"]["ma"]["direction_state"]["invalidated_event_ids"] = [first["events"][0]["event_id"]]
    third = _v3_row(2, [("wyckoff", "SELL")], ma_side="SELL", p=None, origins={"ma": first["events"]})
    result = _build([first, intermediate, third])
    assert not result["events"]
    assert result["rows"][1]["invalidated_evidence"][0]["event_id"] == first["events"][0]["event_id"]
    assert result["rows"][2]["sell_base_score"] == .3
    intermediate["candidate_evidence"]["evaluations"]["ma"]["direction_state"]["available_at"] = "2026-01-08"
    assert _build([first, intermediate, third])["events"][0]["date"] == third["date"]
    intermediate["candidate_evidence"]["evaluations"]["ma"]["direction_state"].pop("invalidated_event_ids")
    intermediate["candidate_evidence"]["evaluations"]["structure"]["direction_state"]["invalidated_event_ids"] = [first["events"][0]["event_id"]]
    assert _build([first, intermediate, third])["events"][0]["date"] == third["date"]


def test_temporary_weak_or_missing_evidence_state_cannot_delete_original_fact():
    first = _v3_row(specs=[("ma", "SELL")], ma_side="SELL", p=None)
    middle = _v3_row(1, ma_side=None)
    middle["candidate_evidence"]["evaluations"]["ma"]["direction_state"]["input_eligible"] = False
    third = _v3_row(2, [("wyckoff", "SELL")], ma_side="SELL", p=None, origins={"ma": first["events"]})
    assert _build([first, middle, third])["events"][0]["date"] == third["date"]


@pytest.mark.parametrize("source", ["ma", "high_return"])
def test_current_candidate_policy_or_risk_hash_mismatch_is_not_repaired(source):
    row = _v3_row(specs=[("wyckoff", "BUY")])
    evidence = row["candidate_evidence"]["evaluations"][source] if source == "ma" else row["agents"][source]
    evidence["model_identity"]["risk_plan_policy_hash"] = "other-risk-policy"
    assert not _build([row])["events"]


def test_same_day_structural_conflicts_follow_original_dependency_families():
    row = _v3_row(specs=[("structure", "BUY", ["native_ma_assisted_definition"]),
                         ("structure", "SELL", ["native_ma_assisted_definition"]), ("wyckoff", "BUY")])
    decision = _build([row])["rows"][0]
    assert decision["agents"]["structure"]["judgment"] == "CONFLICT"
    assert decision["family_conflicts"] == ["ma"]
    assert decision["buy_base_score"] == .3
    assert decision["action"] == "WAIT"
    independent = _v3_row(specs=[("structure", "BUY"), ("structure", "SELL"), ("wyckoff", "BUY")])
    decision = _build([independent])["rows"][0]
    assert decision["family_conflicts"] == ["structure"]
    assert decision["action"] == "BUY"


def test_zero_weight_source_does_not_create_conflict_or_an_extra_family():
    row = _v3_row(specs=[("ma", "SELL"), ("structure", "BUY"), ("wyckoff", "BUY")], ma_side="SELL")
    decision = _build([row], config={"source_weights": {"ma": 0, "structure": .6, "wyckoff": .3}})["rows"][0]
    assert decision["action"] == "BUY"
    assert decision["buy_base_score"] == .9
    assert decision["sell_base_score"] == 0
    assert decision["family_conflicts"] == []


def test_invalidated_new_fact_cannot_trigger_point_with_older_same_source_support():
    first = _v3_row(specs=[("ma", "SELL"), ("structure", "SELL")], ma_side="SELL", p=None)
    # The first candidate has adequate evidence but cannot emit at a wrong-day price.
    first["reference_price_date"] = "2026-01-04"
    second = _v3_row(1, [("structure", "SELL")], ma_side="SELL", p=None,
        origins={"ma": [first["events"][0]], "structure": [first["events"][1]]})
    second["candidate_evidence"]["evaluations"]["structure"]["direction_state"]["invalidated_event_ids"] = [second["events"][0]["event_id"]]
    result = _build([first, second])
    assert result["events"] == []
    decision = result["rows"][1]
    assert decision["sell_base_score"] == .6
    assert decision["proposed_action"] == "SELL"
    assert decision["action"] == "WAIT"
    assert decision["funnel"]["fresh_candidates"] == 0
    assert "no_fresh_candidate_trigger" in decision["reason_codes"]


def test_unqualified_new_fact_cannot_borrow_older_same_source_opinion():
    first = _v3_row(specs=[("ma", "SELL"), ("structure", "SELL")], ma_side="SELL", p=None)
    first["reference_price_date"] = "2026-01-04"
    second = _v3_row(1, [("structure", "SELL")], ma_side="SELL", p=None,
        origins={"ma": [first["events"][0]], "structure": [first["events"][1]]})
    result = _build([first, second])
    assert not result["events"]
    assert result["rows"][1]["sell_base_score"] == .6
    assert result["rows"][1]["funnel"]["fresh_candidates"] == 0


def test_family_zero_fresh_assisted_origin_still_triggers_valid_other_agents():
    first = _v3_row(specs=[("wyckoff", "BUY")], risk=False)
    second = _v3_row(1, [("structure", "BUY", ["native_ma_assisted_definition"])],
        origins={"wyckoff": first["events"]})
    result = _build([first, second])
    decision = result["rows"][1]
    assert decision["agents"]["structure"]["buy_contribution"] == 0
    assert decision["buy_total_score"] == .8
    assert decision["action"] == "BUY"
    assert result["events"][0]["date"] == second["date"]
    assert decision["opportunity_ids"] == [second["events"][0]["opportunity_id"]]
