"""Contract checks for causal direction arbitration, separate from inventory."""
import copy
import datetime as dt

import pytest

from my_strategy.services.czsc_agent_consensus import build_consensus, consensus_config


SYMBOL = "000001.SZ"


def _row(index=0, specs=(), *, p=.8, l2=.2, risk=True):
    day = (dt.date(2026, 1, 5) + dt.timedelta(days=index)).isoformat()
    events = []
    for source, side, *extra in specs:
        ordinal = len(events)
        events.append({"event_id": f"{day}-{source}-{side}-{ordinal}", "symbol": SYMBOL,
            "source": source, "side": side, "date": day, "evidence_date": day,
            "event_type": "ma_trend_failure" if source == "ma" and side == "SELL" else "confirmed_rule",
            "available_at": day + "T15:00:00+08:00", "signal_at": day + "T15:00:00+08:00",
            "candidate_eligible": True, "evidence_visible": True, "input_eligible": True,
            "opportunity_id": f"op-{day}-{side}", "dependencies": extra[0] if extra else []})
    return {"date": day, "symbol": SYMBOL, "reference_close": 100., "reference_price_date": day,
        "reference_price_basis": "unadjusted", "events": events,
        "raw_risk_plan": {"symbol": SYMBOL, "signal_date": day, "entry_allowed": risk},
        "agents": {
            "ma": {"kind": "model_probability", "status": "scored", "probability": p,
                   "target": "actual_complete_fee_net_roundtrip_positive",
                   "model_identity": {"available_at": "2025-12-31"}, "threshold": .55},
            "structure": {"kind": "rule_evidence", "status": "rule_triggered"},
            "wyckoff": {"kind": "rule_evidence", "status": "scored", "buy_score": .8, "sell_score": .8},
            "high_return": {"kind": "model_probability", "status": "scored", "probability": l2,
                            "target": "费用后收益至少8%且至少2倍初始风险",
                            "model_identity": {"available_at": "2025-12-31"}},
        }}


def _build(rows, **kwargs):
    return build_consensus(rows, symbol=SYMBOL, **kwargs)


def test_default_requires_two_independent_sources_and_l2_is_not_a_vote():
    one = _build([_row(specs=[("ma", "BUY")])])
    assert not one["events"]
    assert one["rows"][0]["buy_support"] == 1
    assert one["rows"][0]["agents"]["high_return"]["status"] == "passed"
    assert one["rows"][0]["agents"]["high_return"]["counts_as_vote"] is False
    two = _build([_row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])])
    assert two["events"][0]["side"] == "BUY"
    decision = two["rows"][0]
    assert decision["buy_independent_count"] == 2
    assert decision["state_before"] == "neutral" and decision["state_after"] == "bullish"
    assert two["events"][0]["original_event_ids"] == decision["evidence_event_ids"]
    assert two["events"][0]["reference_price"] == 100
    assert decision["agents"]["arbiter"]["probability"] is None


@pytest.mark.parametrize("probability", [None, 0., True, float("nan"), float("inf"), -1., 1.1])
def test_absent_zero_invalid_ma_probability_never_becomes_rule_support(probability):
    result = _build([_row(specs=[("ma", "BUY"), ("wyckoff", "BUY")], p=probability)])
    agent = result["rows"][0]["agents"]["ma"]
    assert not result["events"]
    assert agent["judgment"] == "ABSTAIN"
    assert agent["probability"] == (0 if probability == 0 and not isinstance(probability, bool) else None)


def test_ma_rule_fallback_must_be_explicit():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")], p=None)
    row["agents"]["ma"].update(status="rule_triggered", kind="rule_evidence")
    assert not _build([row])["events"]
    row["agents"]["ma"]["status"] = "rule_fallback"
    assert _build([row])["events"][0]["side"] == "BUY"


def test_correlated_structure_and_ma_merge_dependency_family():
    row = _row(specs=[("ma", "BUY"), ("structure", "BUY", ["native_ma_assisted_definition"])])
    decision = _build([row])["rows"][0]
    assert decision["buy_support"] == 1
    assert decision["buy_independent_families"] == ["ma"]
    assert "ma_assisted_structure_dependency_merged" in decision["reason_codes"]
    row["events"][1]["dependencies"] = ["native_definition"]
    assert _build([row])["rows"][0]["buy_support"] == 2


def test_known_ma_assisted_native_definition_cannot_gain_independence_from_missing_metadata():
    row = _row(specs=[("ma", "BUY"), ("structure", "BUY")])
    row["events"][1]["event_type"] = "native_bs3_三买"
    row["events"][1].pop("dependencies")
    assert _build([row])["rows"][0]["buy_support"] == 1


def test_dependency_family_uses_max_weight_not_sum():
    row = _row(specs=[("ma", "BUY"), ("structure", "BUY", ["native_ma_assisted_definition"])])
    config = {"source_weights": {"ma": 2, "structure": 3, "wyckoff": 1},
              "min_buy_support": 4}
    result = _build([row], config=config)
    assert result["rows"][0]["buy_support"] == 3
    assert not result["events"]


def test_duplicate_events_and_same_source_do_not_add_votes():
    row = _row(specs=[("ma", "BUY"), ("ma", "BUY")])
    row["events"].append(copy.deepcopy(row["events"][0]))
    result = _build([row])
    assert not result["events"]
    assert result["rows"][0]["buy_support"] == 1
    assert len(result["rows"][0]["agents"]["ma"]["event_ids"]) == 2
    assert any("duplicate_event_identity" in item["reason_codes"] for item in result["rows"][0]["ignored_events"])


def test_opposing_sources_abstain_until_direction_margin_is_sufficient():
    row = _row(specs=[("ma", "BUY"), ("structure", "SELL")])
    assert _build([row])["rows"][0]["action"] == "WAIT"
    row = _row(specs=[("ma", "BUY"), ("structure", "BUY"), ("wyckoff", "SELL")])
    result = _build([row])
    assert result["rows"][0]["action"] == "BUY"
    assert result["rows"][0]["sell_support"] == 1
    assert not _build([row], config={"min_direction_margin": 2})["events"]


def test_conflict_within_one_source_has_no_two_direction_vote():
    row = _row(specs=[("ma", "BUY"), ("ma", "SELL"), ("structure", "BUY")])
    decision = _build([row])["rows"][0]
    assert decision["agents"]["ma"]["judgment"] == "CONFLICT"
    assert decision["buy_support"] == 1 and decision["sell_support"] == 0


@pytest.mark.parametrize("status,probability", [("scored", None), ("score_unavailable", .9), ("scored", 0.)])
def test_high_return_missing_or_below_quality_gate_is_wait(status, probability):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")], l2=probability)
    row["agents"]["high_return"]["status"] = status
    result = _build([row])
    assert not result["events"]
    assert result["rows"][0]["proposed_action"] == "BUY"
    assert result["rows"][0]["agents"]["high_return"]["probability"] == probability
    assert _build([row], model_family="czsc_research")["events"][0]["side"] == "BUY"


def test_high_return_gate_can_be_disabled_explicitly_but_risk_veto_remains():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")], l2=None)
    assert _build([row], config={"require_high_return_when_family": False})["events"]
    row["raw_risk_plan"]["entry_allowed"] = False
    assert not _build([row], config={"require_high_return_when_family": False})["events"]


@pytest.mark.parametrize("risk", [None, {}, {"entry_allowed": False}, {"entry_allowed": 1},
                                    {"entry_allowed": True, "symbol": "600519.SH"},
                                    {"entry_allowed": True, "available_at": "2026-01-06"}])
def test_high_return_buy_requires_compatible_visible_risk_plan(risk):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["raw_risk_plan"] = risk
    assert not _build([row])["events"]


def test_sell_is_direction_without_inventory_and_independent_exit_cannot_vote():
    row = _row(specs=[("ma", "SELL"), ("structure", "SELL")], p=None, l2=None, risk=False)
    result = _build([row])
    assert result["events"][0]["side"] == "SELL"
    assert result["semantics"] == "market_direction_research"
    assert all(agent["exit_probability"] is None for agent in result["rows"][0]["agents"].values())
    row["events"][0]["event_type"] = "ma_independent_exit_evidence"
    assert not _build([row])["events"]
    assert "actual_holding_required_for_independent_exit" in _build([row])["rows"][0]["agents"]["ma"]["reason_codes"]


def test_same_direction_is_suppressed_but_distinct_opposite_direction_can_switch():
    rows = [_row(0, [("ma", "BUY"), ("wyckoff", "BUY")]),
            _row(1, [("ma", "BUY"), ("wyckoff", "BUY")]),
            _row(2, [("ma", "SELL"), ("structure", "SELL")]),
            _row(3, [("ma", "BUY"), ("wyckoff", "BUY")])]
    result = _build(rows)
    assert [row["action"] for row in result["rows"]] == ["BUY", "WAIT", "SELL", "BUY"]
    assert "same_direction_already_active" in result["rows"][1]["reason_codes"]


def test_expiration_does_not_emit_aged_evidence_and_consumed_opportunity_stays_consumed():
    rows = [_row(0, [("ma", "BUY"), ("wyckoff", "BUY")]), _row(1), _row(2)]
    result = _build(rows, config={"signal_valid_bars": 2, "cooldown_bars": 0})
    assert [row["action"] for row in result["rows"]] == ["BUY", "WAIT", "WAIT"]
    assert result["final_state"] == "neutral"
    assert "direction_evidence_expired" in result["rows"][2]["reason_codes"]
    retry = _row(3, [("ma", "BUY"), ("wyckoff", "BUY")])
    for event in retry["events"]:
        event["opportunity_id"] = rows[0]["events"][0]["opportunity_id"]
    assert len(_build(rows + [retry], config={"signal_valid_bars": 2, "cooldown_bars": 0})["events"]) == 1
    new = _row(4, [("ma", "BUY"), ("wyckoff", "BUY")])
    assert len(_build(rows + [retry, new], config={"signal_valid_bars": 2, "cooldown_bars": 0})["events"]) == 2


def test_expiration_and_same_direction_cooldown_are_separate():
    rows = [_row(0, [("ma", "BUY"), ("wyckoff", "BUY")]), _row(1),
            _row(2, [("ma", "BUY"), ("wyckoff", "BUY")]),
            _row(3, [("ma", "BUY"), ("wyckoff", "BUY")])]
    result = _build(rows, config={"signal_valid_bars": 2, "cooldown_bars": 3})
    assert [row["action"] for row in result["rows"]] == ["BUY", "WAIT", "WAIT", "BUY"]
    assert "same_direction_cooldown" in result["rows"][2]["reason_codes"]


def test_prefix_is_immutable_when_future_evidence_changes_and_input_is_not_mutated():
    prefix = [_row(0, [("ma", "BUY"), ("wyckoff", "BUY")]), _row(1)]
    snapshot = copy.deepcopy(prefix)
    old = _build(prefix)
    future = _row(2, [("ma", "SELL"), ("structure", "SELL")])
    new = _build(prefix + [future])
    assert new["rows"][:2] == old["rows"]
    assert new["events"][:1] == old["events"]
    assert prefix == snapshot
    future["reference_close"] = 999999
    future["agents"]["ma"]["probability"] = 0
    assert _build(prefix + [future])["rows"][:2] == old["rows"]


@pytest.mark.parametrize("field,value", [("available_at", "2026-01-06"),
                                        ("actual_training_completed_at", "2026-01-06"),
                                        ("label_cutoff", "2026-01-05"),
                                        ("retrospective", True)])
def test_future_or_incompatible_saved_scores_cannot_authorize_buy(field, value):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"]["ma"]["model_identity"][field] = value
    assert not _build([row])["events"]
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"]["high_return"]["model_identity"][field] = value
    assert not _build([row])["events"]


def test_foreign_future_and_rejected_events_abstain_without_poisoning_good_sources():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY"), ("structure", "BUY")])
    row["events"][0]["symbol"] = "600519.SH"
    row["events"][1]["available_at"] = "2026-01-05T15:01:00+08:00"
    row["events"][2]["candidate_eligible"] = False
    result = _build([row])
    assert result["rows"][0]["buy_support"] == 0
    assert len(result["rows"][0]["ignored_events"]) == 3


def test_unavailable_wyckoff_input_is_not_a_vote_even_with_numerical_strength():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"]["wyckoff"]["status"] = "input_unavailable"
    assert not _build([row])["events"]
    row["agents"]["wyckoff"].update(status="rule_triggered", buy_score=None)
    assert _build([row])["events"]


@pytest.mark.parametrize("price", [None, True, float("nan"), float("inf"), 0, -1])
def test_missing_exact_price_is_wait_with_no_point(price):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["reference_close"] = price
    result = _build([row])
    assert not result["events"]
    assert result["rows"][0]["state_after"] == "neutral"
    assert "exact_signal_day_reference_price_unavailable" in result["rows"][0]["reason_codes"]


@pytest.mark.parametrize("change", [
    {"unknown": 1}, {"source_weights": {"oops": 1}}, {"source_weights": {"ma": True}},
    {"source_weights": {"ma": -1}}, {"source_weights": {"ma": 0, "structure": 0, "wyckoff": 0}},
    {"min_buy_support": 4}, {"min_sell_support": 0}, {"min_direction_margin": float("nan")},
    {"ma_entry_threshold": True}, {"wyckoff_min_strength": float("inf")}, {"high_return_threshold": 1.1},
    {"signal_valid_bars": 0}, {"signal_valid_bars": 1.5}, {"cooldown_bars": -1},
    {"cooldown_bars": True}, {"require_high_return_when_family": 1},
])
def test_bad_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        consensus_config(change)


def test_legal_config_is_copied_and_its_hash_binds_events():
    config = {"min_buy_support": 1, "cooldown_bars": 0, "high_return_threshold": 0}
    result = _build([_row(specs=[("ma", "BUY")], l2=0)], config=config)
    assert result["events"]
    assert result["events"][0]["config_hash"] == result["config_hash"]
    result["config"]["min_buy_support"] = 3
    assert config["min_buy_support"] == 1
    assert _build([], config={"min_buy_support": 1})["config_hash"] != _build([])["config_hash"]


@pytest.mark.parametrize("rows", [[_row(1), _row(0)], [_row(0), _row(0)],
                                 [{"symbol": SYMBOL, "date": "bad"}],
                                 [{"symbol": "600519.SH", "date": "2026-01-05"}]])
def test_row_identity_and_order_are_required(rows):
    with pytest.raises(ValueError):
        _build(rows)


def test_empty_prefix_has_explicit_neutral_state():
    result = _build([])
    assert result["rows"] == result["events"] == []
    assert result["initial_state"] == result["final_state"] == "neutral"


def test_ma_consensus_threshold_is_adjustable_without_changing_saved_model_threshold():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")], p=.8)
    assert _build([row])["events"]
    result = _build([row], config={"ma_entry_threshold": .9})
    assert not result["events"]
    ma = result["rows"][0]["agents"]["ma"]
    assert ma["threshold"] == .9 and ma["model_threshold"] == .55


def test_fresh_qualified_same_direction_support_keeps_one_point_for_long_trend():
    rows = [_row(index, [("ma", "BUY"), ("wyckoff", "BUY")]) for index in range(35)]
    result = _build(rows)
    assert len(result["events"]) == 1
    assert result["final_state"] == "bullish"
    assert all("direction_evidence_expired" not in row["reason_codes"] for row in result["rows"])


def test_blocked_same_direction_does_not_extend_lifetime():
    rows = [_row(0, [("ma", "BUY"), ("wyckoff", "BUY")]),
            _row(1, [("ma", "BUY"), ("wyckoff", "BUY")], risk=False), _row(2)]
    result = _build(rows, config={"signal_valid_bars": 2})
    assert result["final_state"] == "neutral"
    assert "direction_evidence_expired" in result["rows"][2]["reason_codes"]


def test_supported_suppressed_opportunities_are_consumed_for_future_replays():
    first = _row(0, [("ma", "BUY"), ("wyckoff", "BUY")])
    suppressed = _row(1, [("ma", "BUY"), ("wyckoff", "BUY")])
    retry = _row(4, [("ma", "BUY"), ("wyckoff", "BUY")])
    for event in retry["events"]:
        event["opportunity_id"] = suppressed["events"][0]["opportunity_id"]
    result = _build([first, suppressed, _row(2), _row(3), retry],
                    config={"signal_valid_bars": 2, "cooldown_bars": 0})
    assert len(result["events"]) == 1
    assert any("opportunity_already_consumed" in item["reason_codes"] for item in result["rows"][-1]["ignored_events"])


def test_unavailable_price_preserves_direction_and_does_not_consume_opportunity():
    first = _row(0, [("ma", "BUY"), ("wyckoff", "BUY")])
    bad = _row(2, [("ma", "SELL"), ("structure", "SELL")])
    bad["reference_close"] = None
    retry = _row(3, [("ma", "SELL"), ("structure", "SELL")])
    for event in retry["events"]:
        event["opportunity_id"] = bad["events"][0]["opportunity_id"]
    result = _build([first, _row(1), bad, retry], config={"signal_valid_bars": 2})
    assert result["rows"][2]["state_after"] == "bullish"
    assert result["rows"][2]["action"] == "WAIT"
    assert result["rows"][3]["action"] == "SELL"


def test_explicit_historical_reconstruction_retains_actual_completion_without_claiming_live_availability():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    for source in ("ma", "high_return"):
        row["agents"][source]["model_identity"].update(
            available_at="2025-12-31", label_cutoff="2025-12-30",
            actual_training_completed_at="2026-10-11", actual_completion_record="2026-10-11",
            training_completed_at="2026-10-11", historical_reconstruction=True)
    result = _build([row])
    assert result["events"]
    for source in ("ma", "high_return"):
        agent = result["rows"][0]["agents"][source]
        assert agent["historical_reconstruction"] is True
        assert agent["model_identity"]["actual_training_completed_at"] == "2026-10-11"
    row["agents"]["ma"]["model_identity"]["historical_reconstruction"] = False
    assert not _build([row])["events"]


@pytest.mark.parametrize("field", ["actual_completion_record", "training_completed_at"])
def test_non_reconstruction_completion_metadata_must_be_observable(field):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"]["ma"]["model_identity"][field] = "2026-01-06"
    assert not _build([row])["events"]


def test_historical_reconstruction_does_not_override_future_checkpoint_or_overlapping_labels():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"]["ma"]["model_identity"].update(historical_reconstruction=True,
                                                   available_at="2026-01-06")
    assert not _build([row])["events"]
    row["agents"]["ma"]["model_identity"].update(available_at="2025-12-31", label_cutoff="2026-01-05")
    assert not _build([row])["events"]


@pytest.mark.parametrize("source", ["ma", "high_return"])
def test_reconstruction_requires_saved_label_cutoff_not_just_boolean(source):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"][source]["model_identity"].update(historical_reconstruction=True,
                                                    actual_training_completed_at="2026-10-11")
    result = _build([row])
    assert not result["events"]
    assert "score_identity_incomplete" in result["rows"][0]["agents"][source]["reason_codes"]
    row["agents"][source]["model_identity"]["train_label_end"] = "2025-12-30"
    assert _build([row])["events"]


def test_reconstruction_rejects_calibration_labels_that_overlap_signal():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"]["high_return"]["model_identity"].update(historical_reconstruction=True,
        train_label_end="2025-12-30", calibration_label_end="2026-01-05")
    assert not _build([row])["events"]


def test_model_score_requires_saved_lineage_availability_and_rechecks_nested_event_identity():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"]["ma"]["model_identity"] = {"target_contract_hash": "abc"}
    assert not _build([row])["events"]
    row["agents"]["ma"]["model_identity"] = {"available_at": "2025-12-31"}
    row["events"][0]["model_lineage"] = {"entry": {"available_at": "2026-01-06"}}
    assert not _build([row])["events"]


def test_future_event_is_not_consumed_until_first_observable_day():
    first = _row(0, [("ma", "BUY"), ("wyckoff", "BUY")])
    for event in first["events"]:
        event.update(evidence_date="2026-01-06", available_at="2026-01-06T15:00:00+08:00")
    later = _row(1)
    later["events"] = copy.deepcopy(first["events"])
    result = _build([first, later])
    assert [row["action"] for row in result["rows"]] == ["WAIT", "BUY"]


def test_zero_weight_does_not_count_independent_family():
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    result = _build([row], config={"source_weights": {"ma": 0},
                                  "min_buy_support": 1, "min_sell_support": 1})
    assert result["rows"][0]["buy_independent_families"] == ["wyckoff"]
    assert result["rows"][0]["buy_support"] == 1


def test_shared_config_loader_is_used_and_malformed_stored_weights_raise_value_error(monkeypatch):
    import my_strategy.services.czsc_agent_consensus as module

    config = consensus_config()
    config["ma_entry_threshold"] = .9
    seen = []
    def loader(name):
        seen.append(name)
        return copy.deepcopy(config)
    monkeypatch.setattr(module, "load_config", loader)
    assert consensus_config()["ma_entry_threshold"] == .9
    assert seen == ["czsc_agent_consensus"]
    config["source_weights"] = {"ma": 1}
    with pytest.raises(ValueError, match="file source weights"):
        consensus_config()


@pytest.mark.parametrize("source,key,head", [("ma", "head", "holding_exit"),
                                          ("ma", "model_head", "ma_exit"),
                                          ("high_return", "head", "policy_entry"),
                                          ("high_return", "model_head", "exit_advantage")])
def test_provided_head_must_match_the_source_entry_identity(source, key, head):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"][source]["model_identity"][key] = head
    result = _build([row])
    assert not result["events"]
    assert "score_head_mismatch" in result["rows"][0]["agents"][source]["reason_codes"]


@pytest.mark.parametrize("source", ["ma", "high_return"])
@pytest.mark.parametrize("target", [None, "fee_net_fixed_10d_return_positive"])
def test_missing_or_fixed_horizon_target_cannot_be_used_as_full_interval_entry_score(source, target):
    row = _row(specs=[("ma", "BUY"), ("wyckoff", "BUY")])
    row["agents"][source]["target"] = target
    result = _build([row])
    assert not result["events"]
    assert any("target" in code for code in result["rows"][0]["agents"][source]["reason_codes"])
