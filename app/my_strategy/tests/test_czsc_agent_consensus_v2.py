"""Weighted v2 scores, independent families, replay versions and causal guards."""
import copy
import hashlib
import json

import pytest

from my_strategy.services.czsc_agent_consensus import (
    build_consensus as _versioned_build, consensus_config as _versioned_config,
)

def build_consensus(*args, **kwargs):
    kwargs.setdefault("version", "czsc_agent_consensus_v2")
    return _versioned_build(*args, **kwargs)

def consensus_config(config=None, **kwargs):
    kwargs.setdefault("version", "czsc_agent_consensus_v2")
    return _versioned_config(config, **kwargs)
from my_strategy.tests.test_czsc_agent_consensus import SYMBOL, _row

V1 = "czsc_agent_consensus_v1"
V2 = "czsc_agent_consensus_v2"


def _build(rows, **kwargs):
    return build_consensus(rows, symbol=SYMBOL, **kwargs)


def _decision(specs, **kwargs):
    return _build([_row(specs=specs, **kwargs)])["rows"][0]


def test_v2_defaults_and_scores_are_fixed_weight_support_not_probability_products():
    config = consensus_config()
    assert config["version"] == V2
    assert config["source_weights"] == {"ma": .4, "structure": .2, "wyckoff": .3}
    assert config["high_return_quality_weight"] == .1
    assert config["min_buy_base_support"] == .5
    assert config["min_buy_support"] == .6
    assert config["min_sell_support"] == .5
    assert config["min_direction_margin"] == .2
    assert config["min_independent_families"] == 2
    result = _build([_row(specs=[("ma", "BUY"), ("wyckoff", "BUY")], p=.55, l2=.1)])
    decision = result["rows"][0]
    assert result["version"] == decision["version"] == result["events"][0]["version"] == V2
    assert decision["buy_support"] == decision["buy_base_score"] == .7
    assert decision["buy_total_score"] == .8
    assert decision["buy_quality_bonus"] == .1
    assert decision["agents"]["ma"]["buy_contribution"] == .4
    assert decision["agents"]["wyckoff"]["buy_contribution"] == .3
    assert decision["agents"]["high_return"]["quality_contribution"] == .1
    assert decision["agents"]["high_return"]["counts_as_vote"] is False
    assert decision["agents"]["arbiter"]["configured_weight"] == 0
    high = _decision([("ma", "BUY"), ("wyckoff", "BUY")], p=.99, l2=.99)
    assert high["buy_total_score"] == decision["buy_total_score"]


@pytest.mark.parametrize("specs,action,buy,sell,bonus", [
    ([("ma", "BUY"), ("structure", "BUY"), ("wyckoff", "SELL")], "BUY", .6, .3, .1),
    ([("structure", "BUY"), ("wyckoff", "BUY")], "BUY", .5, 0, .1),
    ([("structure", "BUY"), ("wyckoff", "BUY"), ("ma", "SELL")], "WAIT", .5, .4, .1),
    ([("ma", "BUY")], "WAIT", .4, 0, .1),
    ([("structure", "SELL"), ("wyckoff", "SELL")], "SELL", 0, .5, 0),
    ([("ma", "SELL"), ("structure", "SELL")], "SELL", 0, .6, 0),
    ([("ma", "SELL"), ("wyckoff", "SELL")], "SELL", 0, .7, 0),
    ([("ma", "BUY"), ("structure", "SELL"), ("wyckoff", "SELL")], "WAIT", .4, .5, .1),
    ([], "WAIT", 0, 0, 0),
])
def test_authorized_plan_examples(specs, action, buy, sell, bonus):
    decision = _decision(specs)
    assert decision["action"] == action
    assert decision["buy_base_score"] == buy
    assert decision["sell_base_score"] == sell
    assert decision["buy_quality_bonus"] == bonus
    assert decision["buy_total_score"] == float(str(round(buy + bonus, 10)))


def test_quality_bonus_never_changes_l1_direction_margin_or_becomes_a_second_family():
    decision = _decision([("structure", "BUY"), ("wyckoff", "BUY"), ("ma", "SELL")])
    assert decision["direction_margin"] == .1
    assert decision["buy_total_score"] - decision["sell_base_score"] > .19
    assert decision["proposed_action"] == decision["action"] == "WAIT"
    only_ma = _decision([("ma", "BUY")])
    assert only_ma["buy_independent_count"] == 1
    assert "insufficient_independent_families" in only_ma["reason_codes"]


def test_decimal_total_and_margin_boundaries_are_inclusive():
    exact = _decision([("structure", "BUY"), ("wyckoff", "BUY")], l2=.1)
    assert exact["buy_base_score"] == .5 and exact["buy_total_score"] == .6
    assert exact["action"] == "BUY"
    config = {"source_weights": {"ma": .3, "structure": .3, "wyckoff": .4},
              "high_return_quality_weight": 0}
    row = _row(specs=[("ma", "BUY"), ("structure", "BUY"), ("wyckoff", "SELL")])
    exact_margin = _build([row], config=config)["rows"][0]
    assert exact_margin["direction_margin"] == .2
    assert exact_margin["action"] == "BUY"
    assert _build([row], config={**config, "min_direction_margin": .200000001})["rows"][0]["action"] == "WAIT"
    assert _build([_row(specs=[("structure", "BUY"), ("wyckoff", "BUY")])],
                  config={"min_buy_support": .600000001})["rows"][0]["action"] == "WAIT"


@pytest.mark.parametrize("model_family", ["high_return", "czsc_research", "ma_trend", "unknown"])
@pytest.mark.parametrize("status,probability", [("score_unavailable", .9), ("scored", None), ("scored", .099999)])
def test_v2_missing_or_rejected_quality_vetoes_buy_in_every_family(model_family, status, probability):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")], l2=probability)
    row["agents"]["high_return"]["status"] = status
    decision = _build([row], model_family=model_family)["rows"][0]
    assert decision["proposed_action"] == "BUY"
    assert decision["action"] == "WAIT"
    assert decision["buy_quality_bonus"] == 0
    assert decision["agents"]["high_return"]["required"] is True


@pytest.mark.parametrize("probability", [None, .01, .99])
def test_sell_does_not_use_l2_or_entry_risk_and_never_inverts_failed_quality(probability):
    row = _row(specs=[("structure", "SELL"), ("wyckoff", "SELL")], l2=probability, risk=False)
    row["agents"]["high_return"]["status"] = "score_unavailable"
    assert _build([row])["rows"][0]["action"] == "SELL"
    decision = _decision([("ma", "BUY"), ("wyckoff", "BUY")], l2=.01)
    assert decision["action"] == "WAIT"
    assert decision["sell_base_score"] == 0


def test_saved_quality_without_a_buy_candidate_has_no_bonus():
    decision = _decision([("ma", "SELL"), ("structure", "SELL")])
    assert decision["agents"]["high_return"]["status"] == "no_candidate"
    assert decision["buy_quality_bonus"] == 0


def test_source_abstention_keeps_configured_weights_and_does_not_renormalize():
    row = _row(specs=[("ma", "BUY"), ("structure", "BUY"), ("wyckoff", "BUY")], p=.54)
    row["agents"]["wyckoff"]["buy_score"] = .49
    decision = _build([row])["rows"][0]
    assert decision["buy_base_score"] == .2
    assert decision["action"] == "WAIT"
    assert decision["agents"]["ma"]["configured_weight"] == .4
    assert decision["agents"]["ma"]["buy_contribution"] == 0


def test_same_source_duplicates_and_opposing_opinions_never_spend_twice():
    row = _row(specs=[("ma", "BUY"), ("ma", "BUY"), ("structure", "BUY")])
    row["events"].append(copy.deepcopy(row["events"][0]))
    decision = _build([row])["rows"][0]
    assert decision["buy_base_score"] == .6
    assert decision["agents"]["ma"]["buy_contribution"] == .4
    conflict = _decision([("ma", "BUY"), ("ma", "SELL"), ("structure", "BUY"), ("wyckoff", "BUY")])
    assert conflict["agents"]["ma"]["judgment"] == "CONFLICT"
    assert conflict["buy_base_score"] == .5 and conflict["sell_base_score"] == 0


def test_assisted_structure_merges_same_direction_without_repeating_budget():
    decision = _decision([("ma", "BUY"), ("structure", "BUY", ["native_ma_assisted_definition"]), ("wyckoff", "BUY")])
    assert decision["buy_base_score"] == .7
    assert decision["buy_independent_families"] == ["ma", "wyckoff"]
    assert decision["agents"]["structure"]["buy_contribution"] == 0
    assert decision["family_details"]["ma"]["sources"] == ["ma", "structure"]
    assert decision["family_details"]["ma"]["weight"] == .4
    known = _row(specs=[("ma", "BUY"), ("structure", "BUY")])
    known["events"][1]["event_type"] = "native_bs3_三买"
    known["events"][1].pop("dependencies")
    assert _build([known])["rows"][0]["buy_base_score"] == .4


def test_dependency_family_cross_source_disagreement_abstains_both_directions():
    decision = _decision([("ma", "SELL"), ("structure", "BUY", ["native_ma_assisted_definition"]), ("wyckoff", "BUY")])
    assert decision["buy_base_score"] == .3 and decision["sell_base_score"] == 0
    assert decision["family_conflicts"] == ["ma"]
    assert decision["family_details"]["ma"]["side"] == "CONFLICT"
    assert decision["family_details"]["ma"]["weight"] == 0
    assert decision["agents"]["ma"]["sell_contribution"] == 0
    assert decision["agents"]["structure"]["buy_contribution"] == 0


def test_mixed_structural_events_allocate_once_and_retain_dependency_conflicts():
    specs = [("ma", "BUY"), ("structure", "BUY", ["native_ma_assisted_definition"]),
             ("structure", "BUY", ["native_definition"]), ("wyckoff", "BUY")]
    decision = _decision(specs)
    assert decision["buy_base_score"] == .9
    assert decision["agents"]["structure"]["buy_contribution"] == .2
    assert decision["buy_independent_count"] == 3
    specs[0] = ("ma", "SELL")
    row = _row(specs=specs)
    conflict = _build([row])["rows"][0]
    assert conflict["buy_base_score"] == .5 and conflict["sell_base_score"] == 0
    assert conflict["action"] == "BUY"
    assert conflict["buy_independent_families"] == ["structure", "wyckoff"]
    assert row["events"][1]["event_id"] not in conflict["evidence_event_ids"]
    assert row["events"][2]["event_id"] in conflict["evidence_event_ids"]
    specs.pop(0)
    without_ma = _decision(specs)
    assert without_ma["buy_base_score"] == .5
    assert without_ma["buy_independent_count"] == 2
    assert without_ma["family_details"]["ma"]["weight"] == 0


def test_dependency_family_maximum_can_come_from_structure_with_no_source_budget_overrun():
    config = {"source_weights": {"ma": .1, "structure": .5, "wyckoff": .3}}
    row = _row(specs=[("ma", "BUY"), ("structure", "BUY", ["native_ma_assisted_definition"]), ("wyckoff", "BUY")])
    decision = _build([row], config=config)["rows"][0]
    assert decision["buy_base_score"] == .8
    assert decision["agents"]["structure"]["buy_contribution"] == .5
    assert decision["agents"]["ma"]["buy_contribution"] == 0
    row["events"].append(_row(specs=[("structure", "BUY")])["events"][0])
    row["events"][-1]["event_id"] += "-independent"
    mixed = _build([row], config=config)["rows"][0]
    assert mixed["buy_base_score"] == .9
    assert mixed["agents"]["structure"]["buy_contribution"] == .5


def test_zero_weight_sources_neither_count_nor_create_dependency_conflicts():
    config = {"source_weights": {"ma": 0, "structure": .6, "wyckoff": .3}}
    row = _row(specs=[("ma", "SELL"), ("structure", "BUY", ["native_ma_assisted_definition"]), ("wyckoff", "BUY")])
    decision = _build([row], config=config)["rows"][0]
    assert decision["action"] == "BUY"
    assert decision["buy_base_score"] == .9
    assert decision["family_conflicts"] == []
    only = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    assert _build([only], config=config)["rows"][0]["buy_independent_count"] == 1


def test_independent_family_requirement_can_be_raised_to_three():
    config = {"min_independent_families": 3}
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    assert _build([row], config=config)["rows"][0]["action"] == "WAIT"
    row = _row(specs=[("ma", "BUY"), ("structure", "BUY"), ("wyckoff", "BUY")])
    assert _build([row], config=config)["rows"][0]["action"] == "BUY"


@pytest.mark.parametrize("config", [
    {"version": None}, {"version": "v9"}, {"version": {}}, {"extra": 1},
    {"source_weights": {"ma": True}}, {"source_weights": {"ma": float("nan")}},
    {"source_weights": {"ma": -.1}}, {"source_weights": {"ma": .5}},
    {"high_return_quality_weight": True}, {"high_return_quality_weight": -.1},
    {"high_return_quality_weight": float("inf")}, {"high_return_quality_weight": .2},
    {"min_buy_base_support": .900001}, {"min_buy_support": 1.000001},
    {"min_sell_support": .900001}, {"min_direction_margin": .900001},
    {"min_buy_base_support": False}, {"min_sell_support": 0},
    {"min_independent_families": 1}, {"min_independent_families": 4},
    {"min_independent_families": 2.1}, {"min_independent_families": True},
    {"source_weights": {"ma": .9, "structure": 0, "wyckoff": 0}},
    {"require_high_return_when_family": False}, {"require_high_return_when_family": 1},
])
def test_v2_invalid_or_unreachable_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        consensus_config(config)


def test_unknown_and_conflicting_versions_are_rejected():
    for version in ("v9", True, {}):
        with pytest.raises(ValueError):
            consensus_config(version=version)
    with pytest.raises(ValueError, match="conflicting"):
        consensus_config({"version": V1}, version=V2)
    with pytest.raises(ValueError, match="conflicting"):
        _build([], config={"version": V2}, version=V1)


def test_complete_legacy_configuration_automatically_replays_v1_without_new_hash_fields():
    legacy = _versioned_config(version=V1)
    assert "version" not in legacy and len(legacy) == 10
    row = _row(specs=[("ma", "BUY"), ("structure", "BUY")], l2=.2)
    explicit = _build([row], config=legacy, version=V1)
    assert _versioned_build([row], symbol=SYMBOL, config=legacy) == explicit
    assert _build([row], version=V1) == explicit
    assert explicit["version"] == V1
    assert "version" not in explicit["rows"][0]
    assert "buy_base_score" not in explicit["rows"][0]
    assert explicit["rows"][0]["buy_support"] == 2
    assert explicit["config_hash"] == hashlib.sha256(json.dumps(legacy, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert _versioned_config({"ma_entry_threshold": .9}, version="czsc_agent_consensus_v3")["version"] == "czsc_agent_consensus_v3"


def test_legacy_replay_does_not_depend_on_current_configuration_file(monkeypatch):
    import my_strategy.services.czsc_agent_consensus as module
    def fail(_):
        raise AssertionError("v1 replay must not load v2 defaults")
    monkeypatch.setattr(module, "load_config", fail)
    assert module.consensus_config(version=V1)["source_weights"] == {"ma": 1, "structure": 1, "wyckoff": 1}
    assert module.build_consensus([_row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])],
                                 symbol=SYMBOL, version=V1)["events"]


@pytest.mark.parametrize("source,field,value", [
    ("ma", "available_at", "2026-01-06"), ("high_return", "available_at", "2026-01-06"),
    ("ma", "head", "holding_exit"), ("high_return", "head", "exit_advantage"),
    ("ma", "train_label_end", "2026-01-05"), ("high_return", "calibration_label_end", "2026-01-05"),
    ("ma", "retrospective", True), ("high_return", "actual_training_completed_at", "2026-01-06"),
])
def test_v2_saved_score_causality_and_identity_checks_remain(source, field, value):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"][source]["model_identity"][field] = value
    assert not _build([row])["events"]


@pytest.mark.parametrize("missing", ["exact_price", "wrong_price_day", "risk_plan", "risk_rejected", "foreign", "future"])
def test_v2_price_risk_and_event_checks_remain(missing):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    if missing == "exact_price":
        row["reference_close"] = None
    elif missing == "wrong_price_day":
        row["reference_price_date"] = "2026-01-06"
    elif missing == "risk_plan":
        row["raw_risk_plan"] = None
    elif missing == "risk_rejected":
        row["raw_risk_plan"]["entry_allowed"] = False
    elif missing == "foreign":
        row["events"][0]["symbol"] = "600519.SH"
    else:
        row["events"][0]["available_at"] = "2026-01-06"
    assert not _build([row])["events"]


def test_v2_prefix_dedup_lifetime_and_opportunity_consumption_remain():
    first = _row(0, [("structure", "BUY"), ("wyckoff", "BUY")])
    sustained = _row(1, [("structure", "BUY"), ("wyckoff", "BUY")])
    opposite = _row(2, [("structure", "SELL"), ("wyckoff", "SELL")], l2=None)
    prefix = [first, sustained]
    snapshot = copy.deepcopy(prefix)
    result = _build(prefix)
    larger = _build(prefix + [opposite])
    assert larger["rows"][:2] == result["rows"]
    assert larger["events"][:1] == result["events"]
    assert [row["action"] for row in larger["rows"]] == ["BUY", "WAIT", "SELL"]
    assert prefix == snapshot
    expired = _build([first, _row(1), _row(2)], config={"signal_valid_bars": 2, "cooldown_bars": 0})
    assert expired["final_state"] == "neutral"
    retry = _row(3, [("structure", "BUY"), ("wyckoff", "BUY")])
    for event in retry["events"]:
        event["opportunity_id"] = first["events"][0]["opportunity_id"]
    assert len(_build([first, _row(1), _row(2), retry],
                     config={"signal_valid_bars": 2, "cooldown_bars": 0})["events"]) == 1


def test_all_opinion_and_structural_dependency_combinations_conserve_source_budgets():
    """Conflicts and mixed dependencies must never create extra voting budget."""
    ma_options = [[], [("ma", "BUY")], [("ma", "SELL")], [("ma", "BUY"), ("ma", "SELL")]]
    wyckoff_options = [[], [("wyckoff", "BUY")], [("wyckoff", "SELL")], [("wyckoff", "BUY"), ("wyckoff", "SELL")]]
    structural_options = [[]]
    for side in ("BUY", "SELL"):
        structural_options.extend([
            [("structure", side)],
            [("structure", side, ["native_ma_assisted_definition"])],
            [("structure", side), ("structure", side, ["native_ma_assisted_definition"])],
        ])
    structural_options.extend([
        [("structure", "BUY"), ("structure", "SELL")],
        [("structure", "BUY", ["native_ma_assisted_definition"]), ("structure", "SELL")],
    ])
    for ma in ma_options:
        for structure in structural_options:
            for wyckoff in wyckoff_options:
                decision = _decision(ma + structure + wyckoff)
                assert decision["buy_base_score"] + decision["sell_base_score"] <= .900000000001
                assert decision["buy_total_score"] <= 1
                buy, sell = [], []
                for source, weight in (("ma", .4), ("structure", .2), ("wyckoff", .3)):
                    agent = decision["agents"][source]
                    assert agent["buy_contribution"] + agent["sell_contribution"] <= weight
                    assert not (agent["buy_contribution"] and agent["sell_contribution"])
                    buy.append(agent["buy_contribution"])
                    sell.append(agent["sell_contribution"])
                assert round(sum(buy), 10) == decision["buy_base_score"]
                assert round(sum(sell), 10) == decision["sell_base_score"]
                if decision["action"] in {"BUY", "SELL"}:
                    side = decision["action"].lower()
                    assert decision[side + "_independent_count"] >= 2
                if decision["action"] == "BUY":
                    assert decision["buy_base_score"] >= .5
                    assert decision["buy_total_score"] >= .6
                    assert decision["buy_quality_bonus"] == .1


def test_v2_legal_config_hash_binds_every_arbiter_and_does_not_mutate_override():
    config = {"high_return_threshold": .2, "cooldown_bars": 0}
    snapshot = copy.deepcopy(config)
    result = _build([_row(specs=[("structure", "BUY"), ("wyckoff", "BUY")])], config=config)
    assert result["events"]
    assert config == snapshot
    assert result["events"][0]["config_hash"] == result["rows"][0]["config_hash"] == result["config_hash"]
    assert result["config"]["version"] == V2
    assert result["config_hash"] != _build([])["config_hash"]
    result["config"]["source_weights"]["ma"] = .9
    assert consensus_config()["source_weights"]["ma"] == .4


@pytest.mark.parametrize("probability", [None, True, float("nan"), float("inf"), -.1, 1.1])
def test_v2_invalid_quality_probability_never_gets_a_bonus(probability):
    decision = _decision([("ma", "BUY"), ("wyckoff", "BUY")], l2=probability)
    assert decision["action"] == "WAIT"
    assert decision["buy_quality_bonus"] == 0


def test_assisted_structural_source_conflict_propagates_to_ma_family():
    decision = _decision([
        ("ma", "BUY"),
        ("structure", "BUY", ["native_ma_assisted_definition"]),
        ("structure", "SELL", ["native_ma_assisted_definition"]),
        ("wyckoff", "BUY"),
    ])
    assert decision["agents"]["structure"]["judgment"] == "CONFLICT"
    assert decision["family_conflicts"] == ["ma"]
    assert decision["buy_base_score"] == .3 and decision["sell_base_score"] == 0
    assert decision["agents"]["ma"]["buy_contribution"] == 0
    assert decision["agents"]["structure"]["buy_contribution"] == 0
    assert decision["action"] == "WAIT"


def test_ma_source_conflict_cannot_be_hidden_by_assisted_structural_support():
    decision = _decision([
        ("ma", "BUY"), ("ma", "SELL"),
        ("structure", "BUY", ["native_ma_assisted_definition"]),
        ("wyckoff", "BUY"),
    ])
    assert decision["agents"]["ma"]["judgment"] == "CONFLICT"
    assert decision["family_conflicts"] == ["ma"]
    assert decision["family_details"]["ma"]["side"] == "CONFLICT"
    assert decision["buy_base_score"] == .3 and decision["sell_base_score"] == 0
    assert decision["buy_independent_families"] == ["wyckoff"]
    assert decision["action"] == "WAIT"


def test_independent_structural_conflict_does_not_pollute_ma_family():
    decision = _decision([
        ("ma", "BUY"), ("structure", "BUY"), ("structure", "SELL"), ("wyckoff", "BUY"),
    ])
    assert decision["family_conflicts"] == ["structure"]
    assert decision["family_details"]["ma"]["side"] == "BUY"
    assert decision["buy_base_score"] == .7 and decision["sell_base_score"] == 0
    assert decision["action"] == "BUY"


def test_mixed_conflicting_structural_events_follow_their_actual_dependencies_only():
    independent_sell = _decision([
        ("ma", "BUY"), ("structure", "BUY", ["native_ma_assisted_definition"]),
        ("structure", "SELL"), ("wyckoff", "BUY"),
    ])
    assert independent_sell["agents"]["structure"]["judgment"] == "CONFLICT"
    assert independent_sell["family_conflicts"] == []
    assert independent_sell["buy_base_score"] == .7 and independent_sell["action"] == "BUY"
    assert independent_sell["agents"]["structure"]["buy_contribution"] == 0
    assert independent_sell["agents"]["structure"]["sell_contribution"] == 0
    assisted_sell = _decision([
        ("ma", "BUY"), ("structure", "BUY"),
        ("structure", "SELL", ["native_ma_assisted_definition"]), ("wyckoff", "BUY"),
    ])
    assert assisted_sell["family_conflicts"] == ["ma"]
    assert assisted_sell["buy_base_score"] == .3 and assisted_sell["action"] == "WAIT"
