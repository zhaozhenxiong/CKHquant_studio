"""Candidate assessment is causal, risk-separated and opt-in for chart v3."""
import copy

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_candidate_evidence import (
    build_candidate_evidence, recover_daily_evidence, _verify_manifest, _policy_binding,
)
from my_strategy.services.czsc_high_return_runtime import _config
from my_strategy.services.czsc_high_return_models import _columns_for, schema_hash
from my_strategy.services.czsc_high_return_labels import target_contract

SYMBOL = "600000.SH"
DAYS = ["2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10"]


def _identity(day):
    return {"model_run_id": "saved-model", "checkpoint": "2026Q2", "available_at": "2026-03-31",
        "train_label_end": "2025-12-31", "target_contract_hash": "verified-target", "manifest_sha256": "verified-manifest",
        "model_head": "ma_entry", "score_date": day, "verified_frozen_contract": True}


def _event(day, source="wyckoff", side="BUY", **extra):
    return {"event_id": source + side + day, "symbol": SYMBOL, "date": day, "source": source, "side": side,
        "available_at": day + "T15:00:00+08:00", "input_eligible": True, "evidence_visible": True,
        "candidate_eligible": True, "reason_codes": [], "dependencies": ["causal_prior_range"],
        "event_type": "sos" if source == "wyckoff" else "ma_trend_failure" if source == "ma" and side == "SELL" else "native_bs1_一买",
        "stop_price": 9., **extra}


def _inputs(days=DAYS):
    rows, features = [], []
    for day in days:
        rows.append({"date": day, "symbol": SYMBOL, "reference_close": 10., "events": [],
            "raw_risk_plan": {"entry_allowed": True, "reason_codes": []},
            "agents": {"ma": {"status": "scored", "probability": .64, "target": "actual_complete_fee_net_roundtrip_positive",
                       "model_identity": _identity(day)}, "structure": {"events": []}, "wyckoff": {"events": []}}})
        features.append({"date": day, "symbol": SYMBOL, "model_input_eligible": True,
            "close_above_ma20": 1., "ma20_slope5": .01, "ma_alignment": 1., "ret_5": .01, "ret_1": .005, "ma5_ma10_gap": .01,
            "structure_input_eligible": True, "structure_confirmed_at": "2026-06-01T15:00:00+08:00",
            "structure_low": 9., "structure_high": 11., "wyckoff_input_eligible": True,
            "w_demand_score": .8, "w_supply_score": .0, "wyckoff_range_low": 8., "wyckoff_range_high": 12.,
            "wyckoff_available_at": day + "T15:00:00+08:00", "wyckoff_observed_at": "2026-06-04T15:00:00+08:00",
            "wyckoff_range_formed_at": "2026-06-03T15:00:00+08:00"})
    return rows, pd.DataFrame(features)


def _state(output, day, source):
    return output[day]["evaluations"][source]["direction_state"]


def test_daily_ma_evaluates_other_source_candidate_without_same_day_ma_event():
    rows, frame = _inputs()
    rows[0]["events"] = [_event(DAYS[0])]
    original = copy.deepcopy(rows)
    output = build_candidate_evidence(rows, frame)
    assert _state(output, DAYS[0], "ma")["buy_valid"] is True
    assert _state(output, DAYS[0], "ma")["origin_event_ids"] == []
    assert _state(output, DAYS[0], "ma")["current_state"]["score_date"] == DAYS[0]
    assert output[DAYS[0]]["evaluations"]["ma"]["model_identity"] == _identity(DAYS[0])
    assert rows == original


@pytest.mark.parametrize("changes", [
    {"model_run_id": None}, {"manifest_sha256": None}, {"verified_frozen_contract": False},
    {"available_at": "2026-06-05"}, {"train_label_end": "2026-06-04"},
    {"score_date": "2026-06-03"}, {"model_head": "entry"},
])
def test_missing_or_future_ma_lineage_cannot_become_support(changes):
    rows, frame = _inputs()
    rows[0]["agents"]["ma"]["model_identity"].update(changes)
    assert _state(build_candidate_evidence(rows, frame), DAYS[0], "ma")["buy_valid"] is False


def test_bare_target_hash_never_authorizes_ma_probability():
    rows, frame = _inputs()
    rows[0]["agents"]["ma"]["model_identity"] = {"target_contract_hash": "known"}
    state = _state(build_candidate_evidence(rows, frame), DAYS[0], "ma")
    assert state["buy_valid"] is False
    assert "ma_current_complete_lineage_unavailable" in state["reason_codes"]


def test_direction_survives_risk_rejection_but_final_buy_risk_stays_rejected():
    rows, frame = _inputs()
    rows[0]["raw_risk_plan"] = {"entry_allowed": False, "reason_codes": ["source_candidate_ineligible"]}
    rows[0]["events"] = [_event(DAYS[0], candidate_eligible=False, stop_price=8.4,
                                reason_codes=["stop_distance_exceeds_frozen_maximum"])]
    output = build_candidate_evidence(rows, frame)
    evidence = output[DAYS[0]]
    assert evidence["facts"][0]["candidate_eligible"] is False
    assert evidence["facts"][0]["direction_eligible"] is True
    assert _state(output, DAYS[0], "wyckoff")["buy_valid"] is True
    assert evidence["candidate_risk_eligible"] is False
    assert evidence["risk_details"][0]["stop_distance"] == pytest.approx(.16)
    assert "stop_distance_exceeds_frozen_maximum" in evidence["candidate_risk_reasons"]


@pytest.mark.parametrize("stop,distance", [(8.5, .15), (8.499, .1501)])
def test_exact_risk_boundary_is_inspectable_without_rounding_away(stop, distance):
    rows, frame = _inputs()
    rows[0]["events"] = [_event(DAYS[0], stop_price=stop)]
    assert build_candidate_evidence(rows, frame)[DAYS[0]]["risk_details"][0]["stop_distance"] == pytest.approx(distance)


def test_source_memory_uses_original_ids_and_expires_after_bound():
    rows, frame = _inputs()
    event = _event(DAYS[0])
    rows[0]["events"] = [event]
    output = build_candidate_evidence(rows, frame, config={"evidence_valid_bars": 3})
    assert [_state(output, day, "wyckoff")["buy_valid"] for day in DAYS] == [True, True, True, False, False]
    assert _state(output, DAYS[2], "wyckoff")["origin_event_ids"] == [event["event_id"]]
    assert _state(output, DAYS[2], "wyckoff")["date"] == DAYS[2]
    assert _state(output, DAYS[2], "wyckoff")["source_id"] != _state(output, DAYS[0], "wyckoff")["source_id"]


def test_current_wyckoff_strength_cannot_carry_stale_direction():
    rows, frame = _inputs()
    rows[0]["events"] = [_event(DAYS[0])]
    frame.loc[1, "w_demand_score"] = .49
    frame.loc[2, "w_supply_score"] = .9
    output = build_candidate_evidence(rows, frame)
    assert _state(output, DAYS[0], "wyckoff")["buy_valid"] is True
    assert _state(output, DAYS[1], "wyckoff")["buy_valid"] is False
    assert _state(output, DAYS[2], "wyckoff")["buy_valid"] is False


@pytest.mark.parametrize("key", ["wyckoff_available_at", "wyckoff_observed_at", "wyckoff_range_formed_at"])
def test_future_wyckoff_metadata_blocks_current_score(key):
    rows, frame = _inputs()
    rows[0]["events"] = [_event(DAYS[0])]
    frame.loc[0, key] = "2026-06-05T15:00:00+08:00"
    state = _state(build_candidate_evidence(rows, frame), DAYS[0], "wyckoff")
    assert state["input_eligible"] is False and state["buy_valid"] is False


def test_structure_revalidates_original_support_and_opposite_confirmation():
    rows, frame = _inputs()
    rows[0]["events"] = [_event(DAYS[0], source="structure")]
    rows[1]["reference_close"] = 8.9
    rows[2]["events"] = [_event(DAYS[2], source="structure", side="SELL")]
    output = build_candidate_evidence(rows, frame)
    assert _state(output, DAYS[0], "structure")["buy_valid"] is True
    assert _state(output, DAYS[1], "structure")["buy_valid"] is False
    assert _state(output, DAYS[2], "structure")["buy_valid"] is False
    assert _state(output, DAYS[2], "structure")["sell_valid"] is True


def test_broken_structure_support_never_resurrects_after_price_recovers():
    rows, frame = _inputs()
    rows[0]["events"] = [_event(DAYS[0], source="structure")]
    rows[1]["reference_close"] = 8.9
    output = build_candidate_evidence(rows, frame)
    assert _state(output, DAYS[1], "structure")["invalidated_event_ids"] == [rows[0]["events"][0]["event_id"]]
    assert _state(output, DAYS[2], "structure")["buy_valid"] is False
    assert _state(output, DAYS[2], "structure")["origin_event_ids"] == []


def test_same_day_structure_opposition_keeps_both_dependency_families_visible():
    rows, frame = _inputs()
    rows[0]["events"] = [_event(DAYS[0], source="structure", dependencies=["native_definition"]),
        _event(DAYS[0], source="structure", side="SELL", dependencies=["native_ma_assisted_definition"])]
    state = _state(build_candidate_evidence(rows, frame), DAYS[0], "structure")
    assert state["side"] == "CONFLICT" and state["buy_valid"] and state["sell_valid"]
    assert set(state["origin_event_ids"]) == {event["event_id"] for event in rows[0]["events"]}
    assert set(state["dependencies"]) == {"native_definition", "native_ma_assisted_definition"}
    assert state["invalidated_event_ids"] == []


def test_opposite_wyckoff_state_permanently_invalidates_old_buy_origin():
    rows, frame = _inputs()
    rows[0]["events"] = [_event(DAYS[0])]
    frame.loc[1, "w_supply_score"] = .9
    output = build_candidate_evidence(rows, frame)
    assert _state(output, DAYS[1], "wyckoff")["invalidated_event_ids"] == [rows[0]["events"][0]["event_id"]]
    assert _state(output, DAYS[2], "wyckoff")["buy_valid"] is False


def test_temporary_weak_strength_can_revalidate_without_manufacturing_new_fact():
    rows, frame = _inputs()
    rows[0]["events"] = [_event(DAYS[0])]
    frame.loc[1, "w_demand_score"] = .49
    output = build_candidate_evidence(rows, frame)
    assert _state(output, DAYS[1], "wyckoff")["invalidated_event_ids"] == []
    assert _state(output, DAYS[2], "wyckoff")["buy_valid"] is True
    assert output[DAYS[2]]["facts"] == []


def test_ma_sell_requires_actual_trend_failure_fact_and_current_failure():
    rows, frame = _inputs()
    frame.loc[:, "close_above_ma20"] = -1.
    frame.loc[:, "ma20_slope5"] = -.01
    # An entry probability is never inverted into a sell decision.
    assert _state(build_candidate_evidence(rows, frame), DAYS[0], "ma")["sell_valid"] is False
    rows[0]["events"] = [_event(DAYS[0], source="ma", side="SELL")]
    output = build_candidate_evidence(rows, frame)
    assert _state(output, DAYS[0], "ma")["sell_valid"] is True
    assert _state(output, DAYS[0], "ma")["buy_valid"] is False


def test_overlapping_ma_trend_conditions_keep_sell_origin_for_conflict():
    rows, frame = _inputs()
    # Positive medium-term alignment and a negative short-term crossing can
    # qualify opposing causal rules. Neither side may erase the other origin.
    frame.loc[0, "ma5_ma10_gap"] = -.01
    frame.loc[0, "ret_5"] = -.01
    rows[0]["events"] = [_event(DAYS[0], source="ma", side="SELL")]
    state = _state(build_candidate_evidence(rows, frame), DAYS[0], "ma")
    assert state["buy_valid"] and state["sell_valid"] and state["side"] == "CONFLICT"
    assert state["invalidated_event_ids"] == []


def _manifest(config, head):
    cols = _columns_for(head)
    contract = target_contract(head=head, source="ma" if head == "ma_entry" else "union", config=config,
        market_dates=DAYS, source_snapshot_hash="frozen-source", feature_schema_hash="operation-schema",
        continuation_policy_hash="frozen-continuation", initial_cash=config["initial_cash"])
    contract.update(model_head=head, model_input_schema_hash=schema_hash(cols, head))
    if head == "ma_entry" and config.get("ma_entry_alignment") == "signal_pinned_learned_exit_v2":
        contract["candidate_policy"] = "broad_causal_ma_rule_no_entry_head"
    return {"head": head, "columns": list(cols), "feature_schema_hash": schema_hash(cols, head),
        "target_contract": contract, "target_contract_hash": stable_hash(contract), "manifest_sha256": "verified-manifest",
        "available_at": "2026-03-31", "train_label_end": "2025-12-31"}


@pytest.mark.parametrize("head", ["ma_entry", "entry"])
def test_saved_contract_binds_original_signal_and_risk_configuration(head):
    config = _config()
    manifest = _manifest(config, head)
    verified = _verify_manifest(manifest, config, head=head)
    assert verified["signal_points_config_hash"] == _policy_binding(config)["signal_points_config_hash"]
    assert verified["risk_plan_policy_hash"] == _policy_binding(config)["risk_plan_policy_hash"]
    changed = copy.deepcopy(config)
    changed["signal_points"]["max_stop_distance"] = .2
    with pytest.raises(ValueError, match="binding mismatch"):
        _verify_manifest(manifest, changed, head=head)


@pytest.mark.parametrize("key,value", [("candidate_policy", "different_policy"), ("target", "fixed_10_day"),
                                         ("model_input_schema_hash", "other_schema")])
def test_self_consistent_wrong_stored_contract_cannot_pass(key, value):
    config = _config()
    manifest = _manifest(config, "ma_entry")
    manifest["target_contract"][key] = value
    manifest["target_contract_hash"] = stable_hash(manifest["target_contract"])
    with pytest.raises(ValueError, match="binding mismatch"):
        _verify_manifest(manifest, config, head="ma_entry")


def test_optin_recovery_keeps_original_facts_and_reports_extra_inference(monkeypatch, tmp_path):
    from my_strategy.services import czsc_high_return_runtime as runtime
    from my_strategy.services import czsc_high_return_models as models
    config = _config()
    manifest = _manifest(config, "ma_entry")
    identity = {"checkpoint": "2026Q2", "model_run_id": "saved-model", "available_at": manifest["available_at"],
        "train_label_end": manifest["train_label_end"], "manifest_sha256": manifest["manifest_sha256"],
        "target_contract_hash": manifest["target_contract_hash"]}
    points = {"events": [_event(DAYS[0])], "candidates": [], "config": config["signal_points"],
        "ma_routes": [{"date": day, "mode": "forward_model", "lineage": copy.deepcopy(identity), "probability": .64}
                      for day in DAYS], "compute_info": {"rows": 5, "batches": 1, "model_loads": 1}}
    overlay = {"signal_points": points["events"], "candidates": [],
        "rows": [{"date": day, "symbol": SYMBOL, "high_return_probability": None,
                 "model_resolution": {"model_run_id": "saved-model", "checkpoint": "2026Q2"}} for day in DAYS],
        "compute_info": {"model_inference_rows": 7, "model_inference_batches": 2, "model_loads": 1}}
    original = copy.deepcopy(overlay)
    monkeypatch.setattr(runtime, "_read_run", lambda _: (tmp_path, {"config": config, "model_bundles": [{"name": "2026Q2", "heads": {"ma_entry": "head"}}]}, {}))
    monkeypatch.setattr(runtime, "prepare_points", lambda *args, **kwargs: copy.deepcopy(points))
    monkeypatch.setattr(models, "load_head", lambda _: copy.deepcopy(manifest))
    frame = pd.DataFrame({"date": DAYS, "symbol": [SYMBOL] * len(DAYS)})
    recovered = recover_daily_evidence(overlay, frame, DAYS)
    assert overlay == original
    assert recovered["signal_points"] == original["signal_points"]
    assert recovered["candidates"] == original["candidates"]
    assert recovered["ma_routes"][1]["lineage"]["score_date"] == DAYS[1]
    assert recovered["ma_routes"][1]["lineage"]["verified_frozen_contract"] is True
    assert recovered["compute_info"]["model_inference_rows"] == 12
    assert recovered["compute_info"]["chart_evidence_recovery"]["rows"] == 5
    points["events"] = []
    with pytest.raises(ValueError, match="differs from the original"):
        recover_daily_evidence(overlay, frame, DAYS)


def test_window_comparison_reassesses_each_variant_and_keeps_true_v2_control(monkeypatch):
    from my_strategy.services import czsc_analysis
    from my_strategy.web_dashboard import high_return, chart_evidence
    frame = pd.DataFrame({"date": DAYS, "symbol": [SYMBOL] * len(DAYS), "close": [10.] * len(DAYS)})
    item = {"symbol": SYMBOL, "data_end": DAYS[-1], "data_version": "frozen", "features": frame, "raw": frame}
    base = {"symbol": SYMBOL, "data_end": DAYS[-1], "events": [], "next_session_decision": {}}
    original = {"rows": [{"date": DAYS[-1], "symbol": SYMBOL}], "signal_points": [], "compute_info": {"model_inference_rows": 7}}
    calls, decorated = [], []
    monkeypatch.setattr(czsc_analysis, "analyze_stock", lambda *a, **k: copy.deepcopy(base))
    monkeypatch.setattr(high_return, "_calendar", lambda *a: (DAYS, "verified-calendar"))
    def prepare(*a, **k):
        calls.append("prepare")
        return item
    def overlay(*a, **k):
        calls.append("overlay")
        return original
    def recover(value, *a, **k):
        calls.append("recover")
        return {**copy.deepcopy(value), "ma_routes": [{"date": DAYS[-1], "lineage": {"verified_frozen_contract": True}}]}
    def decorate(value, *, overlay, consensus_config, consensus_version, **kwargs):
        decorated.append((consensus_version, copy.deepcopy(consensus_config), copy.deepcopy(overlay)))
        return {"consensus_version": consensus_version, "evidence_valid_bars": (consensus_config or {}).get("evidence_valid_bars")}
    monkeypatch.setattr(high_return, "_prepare", prepare)
    monkeypatch.setattr(high_return, "_overlay", overlay)
    monkeypatch.setattr(high_return, "_display_model_lineage", copy.deepcopy)
    monkeypatch.setattr("my_strategy.services.czsc_candidate_evidence.recover_daily_evidence", recover)
    monkeypatch.setattr(chart_evidence, "decorate_analysis", decorate)
    result = high_return.compare_high_return_chart_windows(symbol=SYMBOL)
    assert calls == ["prepare", "overlay", "recover"]
    assert list(result["variants"]) == ["v2", "v3_1", "v3_3", "v3_5"]
    assert "ma_routes" not in decorated[0][2]
    assert "chart_model_fold" not in original
    assert [entry[1]["evidence_valid_bars"] for entry in decorated[1:]] == [1, 3, 5]
    assert all(entry[2]["ma_routes"][0]["lineage"]["verified_frozen_contract"] for entry in decorated[1:])
    assert result["diagnostic_only"] is True and result["publication_allowed"] is False


@pytest.mark.parametrize("windows", [(2,), (0,), (True,), (), (1, 1)])
def test_window_comparison_rejects_invalid_window_before_data_or_inference(monkeypatch, windows):
    from my_strategy.web_dashboard import high_return
    from my_strategy.services import czsc_analysis
    monkeypatch.setattr(czsc_analysis, "analyze_stock", lambda *a, **k: pytest.fail("invalid diagnostic must not prepare data"))
    with pytest.raises(ValueError, match="allowed bar counts"):
        high_return.compare_high_return_chart_windows(symbol=SYMBOL, windows=windows)


def test_window_comparison_rejects_invalid_config_before_data_or_inference(monkeypatch):
    from my_strategy.web_dashboard import high_return
    from my_strategy.services import czsc_analysis
    monkeypatch.setattr(czsc_analysis, "analyze_stock", lambda *a, **k: pytest.fail("invalid diagnostic must not prepare data"))
    with pytest.raises(ValueError):
        high_return.compare_high_return_chart_windows(symbol=SYMBOL, consensus_config={"source_weights": {"ma": .8, "structure": .2, "wyckoff": .3}})


def test_comparison_variants_equal_standalone_analysis_exactly(monkeypatch):
    from my_strategy.services import czsc_analysis
    from my_strategy.web_dashboard import high_return
    rows, frame = _inputs()
    frame["close"] = 10.
    frame["open"] = 9.9
    frame["high"] = 10.1
    frame["low"] = 9.8
    bars = [{"time": day, "open": 9.9, "high": 10.1, "low": 9.8, "close": 10.} for day in DAYS]
    base = {"symbol": SYMBOL, "as_of": DAYS[-1], "data_end": DAYS[-1], "data_version": "frozen",
        "price_basis": "unadjusted", "frequencies": {"日线": {"bars": bars}}, "events": [], "next_session_decision": {}}
    original_overlay = {"rows": [{"date": day, "symbol": SYMBOL, "raw_risk_plan": {"entry_allowed": False},
                                  "model_resolution": {}, "high_return_probability": None} for day in DAYS],
        "signal_points": [_event(DAYS[0])], "candidates": [], "compute_info": {"model_inference_rows": 7},
        "latest_decision": {"candidate_action": "WAIT"}}
    item = {"symbol": SYMBOL, "data_end": DAYS[-1], "data_version": "frozen", "features": frame, "raw": frame}
    def recover(overlay, *a, **k):
        output = copy.deepcopy(overlay)
        output["ma_routes"] = [{"date": day, "mode": "forward_model", "probability": .64,
                                "target": "actual_complete_fee_net_roundtrip_positive", "lineage": _identity(day)} for day in DAYS]
        output["compute_info"]["model_inference_rows"] = 12
        return output
    monkeypatch.setattr(czsc_analysis, "analyze_stock", lambda *a, **k: copy.deepcopy(base))
    monkeypatch.setattr(high_return, "_calendar", lambda *a: (DAYS, "calendar"))
    monkeypatch.setattr(high_return, "_prepare", lambda *a, **k: item)
    monkeypatch.setattr(high_return, "_overlay", lambda *a, **k: copy.deepcopy(original_overlay))
    monkeypatch.setattr(high_return, "_display_model_lineage", copy.deepcopy)
    monkeypatch.setattr("my_strategy.services.czsc_candidate_evidence.recover_daily_evidence", recover)
    args = {"symbol": SYMBOL, "start": DAYS[0], "end": DAYS[-1], "as_of": DAYS[-1], "device": "cpu"}
    comparison = high_return.compare_high_return_chart_windows(**args)
    from my_strategy.services.czsc_agent_consensus import consensus_config
    selected = consensus_config(version="czsc_agent_consensus_v3")
    for name in ("v2", "v3_1", "v3_3", "v3_5"):
        version = "czsc_agent_consensus_v2" if name == "v2" else "czsc_agent_consensus_v3"
        config = None if name == "v2" else {**selected, "evidence_valid_bars": int(name[-1])}
        standalone = high_return.analyze_high_return(**args, consensus_config=config, consensus_version=version)
        assert stable_hash(comparison["variants"][name]) == stable_hash(standalone), name
    assert "candidate_evidence" not in comparison["variants"]["v2"]["chart_evidence"]["rows"][0]
    assert comparison["variants"]["v2"]["compute_info"]["model_inference_rows"] == 7
