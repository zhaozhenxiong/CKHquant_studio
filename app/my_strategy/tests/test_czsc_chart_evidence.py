"""Display joins preserve signal dates, nullable scores and frozen evidence."""
import copy

import pandas as pd
import pytest

from my_strategy.web_dashboard.chart_evidence import decorate_analysis as _decorate_analysis


def decorate_analysis(*args, **kwargs):
    """These frozen display-contract fixtures explicitly replay chart v2."""
    kwargs.setdefault("consensus_version", "czsc_agent_consensus_v2")
    return _decorate_analysis(*args, **kwargs)


SYMBOL = "600000.SH"
DAYS = ["2026-07-02", "2026-07-03", "2026-07-06"]


def _analysis():
    bars = [{"time": day, "open": price - .5, "high": price + 1, "low": price - 1, "close": price}
            for day, price in zip(DAYS, [10., 12., 15.])]
    return {"symbol": SYMBOL, "as_of": DAYS[-1], "data_end": DAYS[-1], "data_version": "display-fixture", "price_basis": "unadjusted",
            "frequencies": {"日线": {"bars": bars}}, "events": [],
            "next_session_decision": {"reference_price_date": DAYS[-1], "next_market_session": "2026-07-07"}}


def _event(day, source="ma", side="BUY", **fields):
    return {"event_id": f"{source}-{side}-{day}", "symbol": SYMBOL, "date": day, "time": day,
            "available_at": day + "T15:00:00+08:00", "source": source, "side": side,
            "action": side, "event_type": "rule", "evidence_visible": True, **fields}


def _row(result, day):
    return next(row for row in result["chart_evidence"]["rows"] if row["date"] == day)


def test_event_price_uses_own_signal_day_and_never_anchor_or_latest_quote():
    source = _analysis()
    source["events"] = [_event(DAYS[1], anchor_at=DAYS[0]), _event(DAYS[0], side="SELL")]
    original = copy.deepcopy(source)
    result = decorate_analysis(source)
    assert [event["reference_price"] for event in result["events"]] == [12., 10.]
    assert result["events"][0]["reference_price_date"] == DAYS[1]
    assert result["events"][0]["reference_price_source"] == "signal_day_close"
    assert result["events"][0]["reference_price_basis"] == "unadjusted"
    assert result["events"][0]["anchor_at"] == DAYS[0]
    assert source == original


def test_strict_symbol_and_day_join_keeps_missing_price_null():
    source = _analysis()
    source["events"] = [_event(DAYS[0]), _event(DAYS[1], symbol="000001.SZ")]
    raw = pd.DataFrame([{"symbol": SYMBOL, "date": DAYS[1], "close": 12.},
                        {"symbol": SYMBOL, "date": DAYS[2], "close": 15.},
                        {"symbol": "000001.SZ", "date": DAYS[0], "close": 999.}])
    result = decorate_analysis(source, raw=raw)
    assert [event["reference_price"] for event in result["events"]] == [None, None]
    assert all(event["reference_price_reason"] == "signal_day_bar_unavailable" for event in result["events"])
    assert _row(result, DAYS[0])["reference_close"] is None
    assert _row(result, DAYS[1])["reference_close"] == 12.


def test_asof_truncation_removes_future_display_scores_and_events():
    source = _analysis()
    source["as_of"] = DAYS[1]
    source["events"] = [_event(DAYS[1]), _event(DAYS[2])]
    overlay = {"rows": [{"date": day, "symbol": SYMBOL, "candidate_sources": ["ma"], "high_return_probability": .8}
                        for day in DAYS]}
    result = decorate_analysis(source, overlay=overlay)
    assert [row["date"] for row in result["chart_evidence"]["rows"]] == DAYS[:2]
    assert len(result["events"]) == 1
    assert result["events"][0]["date"] == DAYS[1]


def test_each_historical_candidate_keeps_its_own_score_without_latest_fill():
    source = _analysis()
    overlay = {"rows": [
        {"date": DAYS[0], "symbol": SYMBOL, "candidate_sources": ["ma"], "high_return_probability": .25,
         "probability_target": "frozen_high_return", "model_resolution": {"checkpoint": "old", "available_at": "2026-06-30"}},
        {"date": DAYS[1], "symbol": SYMBOL, "candidate_sources": [], "high_return_probability": .99},
        {"date": DAYS[2], "symbol": SYMBOL, "candidate_sources": ["structure"], "high_return_probability": .75,
         "model_resolution": {"checkpoint": "new", "available_at": DAYS[1]}}],
        "latest_decision": {"high_return_probability": .99}}
    result = decorate_analysis(source, overlay=overlay)
    agents = [_row(result, day)["agents"]["high_return"] for day in DAYS]
    assert [agent["probability"] for agent in agents] == [.25, None, .75]
    assert agents[0]["model_identity"]["checkpoint"] == "old"
    assert agents[1]["status"] == "no_candidate"
    assert agents[2]["model_identity"]["checkpoint"] == "new"


def test_available_model_zero_is_real_and_missing_is_not_zero():
    source = _analysis()
    source["events"] = [_event(DAYS[0], probability=0., probability_target="actual_roundtrip", model_lineage={"available_at": "2026-06-30"})]
    overlay = {"candidates": [{"symbol": SYMBOL, "date": DAYS[2], "l1_ma_entry_available": True,
                               "l1_ma_entry_probability": .65, "l1_ma_entry_target_contract_hash": "contract"}]}
    result = decorate_analysis(source, overlay=overlay)
    assert _row(result, DAYS[0])["agents"]["ma"]["probability"] == 0.
    assert _row(result, DAYS[1])["agents"]["ma"]["probability"] is None
    assert _row(result, DAYS[2])["agents"]["ma"]["probability"] == .65
    assert _row(result, DAYS[1])["agents"]["wyckoff"]["buy_score"] is None


def test_future_checkpoint_cannot_create_historical_probability():
    source = _analysis()
    source["events"] = [_event(DAYS[0], ma_entry_route={"mode": "forward_model", "probability": .8,
                                                      "lineage": {"available_at": DAYS[1]}})]
    overlay = {"rows": [{"date": DAYS[0], "symbol": SYMBOL, "candidate_sources": ["ma"],
                         "high_return_probability": .9, "expected_net_return": .2, "loss_probability": .1,
                         "model_resolution": {"available_at": DAYS[1]}}]}
    result = decorate_analysis(source, overlay=overlay)
    ma, high = (_row(result, DAYS[0])["agents"][name] for name in ("ma", "high_return"))
    assert ma["probability"] is None and ma["reason"] == "ma_model_not_available_at_signal"
    assert high["probability"] is None and high["expected_net_return"] is None and high["loss_probability"] is None


def test_exit_probabilities_stay_null_without_actual_holding_contract():
    source = _analysis()
    source["events"] = [_event(DAYS[0], side="SELL", probability=.99, ma_exit_probability=.95,
                               ma_entry_route={"mode": "forward_model", "probability": .6, "lineage": {"available_at": "2026-06-30"}})]
    overlay = {"rows": [{"date": DAYS[0], "symbol": SYMBOL, "candidate_sources": ["ma"], "high_return_probability": .7,
                         "exit_probability": .9, "exit_model": {"probability": .9, "holding_contract_compatible": False}}]}
    result = decorate_analysis(source, overlay=overlay)
    agents = _row(result, DAYS[0])["agents"]
    assert agents["ma"]["probability"] == .6
    assert agents["ma"]["exit_probability"] is None and agents["high_return"]["exit_probability"] is None
    assert agents["ma"]["exit_status"] == "actual_holding_required"


def test_nested_ma_entry_head_cannot_bypass_signal_time_availability():
    source = _analysis()
    source["events"] = [_event(DAYS[0], probability=.8,
        model_lineage={"entry": {"available_at": DAYS[1], "model_run_id": "future-head"},
                       "exit": {"available_at": "2026-06-30"}})]
    agent = _row(decorate_analysis(source), DAYS[0])["agents"]["ma"]
    assert agent["probability"] is None
    assert agent["reason"] == "ma_model_not_available_at_signal"
    assert agent["model_identity"]["model_run_id"] == "future-head"


def test_rule_strengths_and_structure_counts_are_not_model_probabilities():
    source = _analysis()
    source["events"] = [_event(DAYS[0], source="structure", event_type="native_bs1_一买"),
                        _event(DAYS[0], source="structure", side="SELL", event_type="native_bs3_三卖")]
    features = pd.DataFrame([{"symbol": SYMBOL, "date": DAYS[0], "w_demand_score": .75, "w_supply_score": 0.,
                              "wyckoff_input_eligible": True, "wyckoff_event": "SOS", "wyckoff_state": "sos_pending"}])
    result = decorate_analysis(source, features=features)
    agents = _row(result, DAYS[0])["agents"]
    assert agents["wyckoff"]["buy_score"] == .75 and agents["wyckoff"]["sell_score"] == 0.
    assert agents["wyckoff"]["probability"] is None and agents["wyckoff"]["score_kind"] == "rule_evidence"
    assert agents["structure"]["buy_count"] == 1 and agents["structure"]["sell_count"] == 1
    assert agents["structure"]["probability"] is None


def test_rejected_evidence_is_retained_without_counting_as_agent_intention():
    source = _analysis()
    source["events"] = [_event(DAYS[0], source="structure", candidate_eligible=False),
                        _event(DAYS[0], source="structure", side="SELL", evidence_visible=False)]
    result = decorate_analysis(source)
    agent = _row(result, DAYS[0])["agents"]["structure"]
    assert agent["buy_count"] == agent["sell_count"] == 0
    assert agent["status"] == "no_signal"
    assert len(agent["excluded_events"]) == 2
    assert len(result["events"]) == 2


@pytest.mark.parametrize("code", ["ma_retrospective_model_rejected", "ma_training_after_signal",
                                  "ma_label_overlap", "ma_entry_target_mismatch", "ma_model_head_mismatch"])
def test_rejected_ma_contract_is_unavailable_even_with_finite_probability(code):
    source = _analysis()
    source["events"] = [_event(DAYS[0], ma_entry_route={"mode": "forward_model", "probability": .8,
        "lineage": {"available_at": "2026-06-30"}, "reason_codes": [code]})]
    agent = _row(decorate_analysis(source), DAYS[0])["agents"]["ma"]
    assert agent["probability"] is None and agent["status"] == "score_unavailable"
    assert code in agent["reason"]


def test_ma_threshold_veto_keeps_valid_model_score_visible():
    source = _analysis()
    source["events"] = [_event(DAYS[0], selected_by_reference=False, ma_entry_route={
        "mode": "forward_model", "probability": .2, "lineage": {"available_at": "2026-06-30"},
        "reason_codes": ["ma_entry_probability_veto"]})]
    agent = _row(decorate_analysis(source), DAYS[0])["agents"]["ma"]
    assert agent["probability"] == .2 and agent["status"] == "scored"


def test_delayed_confirmation_uses_observable_signal_close_and_preserves_anchor_date():
    source = _analysis()
    source["events"] = [_event(DAYS[0], source="structure",
                               available_at=DAYS[1] + "T15:00:00+08:00")]
    result = decorate_analysis(source)
    event = result["events"][0]
    assert event["date"] == event["original_event_date"] == DAYS[0]
    assert event["evidence_date"] == event["reference_price_date"] == DAYS[1]
    assert event["reference_price"] == 12.
    assert _row(result, DAYS[0])["agents"]["structure"]["buy_count"] == 0
    assert _row(result, DAYS[1])["agents"]["structure"]["buy_count"] == 1
    source["as_of"] = DAYS[0]
    assert decorate_analysis(source)["events"] == []


def test_future_wyckoff_metadata_cannot_show_rule_strength():
    source = _analysis()
    features = pd.DataFrame([{"symbol": SYMBOL, "date": DAYS[0], "w_demand_score": .75,
        "w_supply_score": 0., "wyckoff_input_eligible": True,
        "wyckoff_observed_at": DAYS[1] + "T15:00:00+08:00"}])
    agent = _row(decorate_analysis(source, features=features), DAYS[0])["agents"]["wyckoff"]
    assert agent["buy_score"] is agent["sell_score"] is None
    assert agent["status"] == "input_unavailable"


def test_raw_signal_point_payload_and_hash_are_unchanged():
    source = _analysis()
    source["signal_points"] = [_event(DAYS[0], probability=.65, feature_snapshot_hash="immutable")]
    source["events"] = copy.deepcopy(source["signal_points"])
    original = copy.deepcopy(source)
    result = decorate_analysis(source)
    assert result["signal_points"] == original["signal_points"]
    assert "reference_price" not in result["signal_points"][0]
    assert source == original
    assert result["events"][0]["feature_snapshot_hash"] == "immutable"


def test_existing_risk_band_and_next_session_are_dated_not_new_orders():
    source = _analysis()
    plan = {"signal_date": DAYS[0], "price_floor": 9.5, "price_ceiling": 11., "stop_price": 9.,
            "entry_allowed": False, "parameters": {"ma_entry_threshold": .55}}
    overlay = {"rows": [{"symbol": SYMBOL, "date": DAYS[0], "raw_risk_plan": plan, "next_market_session": DAYS[1]}]}
    result = decorate_analysis(source, overlay=overlay)
    row = _row(result, DAYS[0])
    assert row["raw_risk_plan"] == plan and row["next_market_session"] == DAYS[1]
    assert row["agents"]["ma"]["threshold"] == .55
    assert _row(result, DAYS[2])["next_market_session"] == "2026-07-07"
    assert "entry_allowed" not in row and "orders" not in result


def test_duplicate_symbol_day_observations_fail_instead_of_silently_selecting():
    raw = pd.DataFrame([{"symbol": SYMBOL, "date": DAYS[0], "close": 1.},
                        {"symbol": SYMBOL, "date": DAYS[0], "close": 2.}])
    with pytest.raises(ValueError, match="日期重复"):
        decorate_analysis(_analysis(), raw=raw)


def test_non_high_return_api_appends_basic_evidence_without_running_models(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from my_strategy.services import czsc_analysis
    from my_strategy.web_dashboard.api import router

    source = _analysis()
    source["events"] = [_event(DAYS[0])]
    monkeypatch.setattr(czsc_analysis, "analyze_stock", lambda **kwargs: copy.deepcopy(source))
    monkeypatch.setattr(czsc_analysis, "strategy_config", lambda: {})
    application = FastAPI()
    application.state.tasks = SimpleNamespace(results=SimpleNamespace(db_path="unused", runs_root="unused", save=lambda kind, result: result))
    application.include_router(router)
    with TestClient(application) as client:
        response = client.post("/api/analysis", json={"symbol": SYMBOL, "research": False,
            "consensus_version": "czsc_agent_consensus_v2"})
    assert response.status_code == 200
    result = response.json()
    assert result["chart_evidence"]["version"] == "chart_evidence_v1"
    assert result["events"][0]["reference_price"] == 10.
    assert all(row["agents"]["high_return"]["probability"] is None for row in result["chart_evidence"]["rows"])


def test_high_return_adapter_retains_historical_scores_and_frozen_signal_points(monkeypatch):
    from my_strategy.services import czsc_analysis
    from my_strategy.web_dashboard import high_return

    source = _analysis()
    raw = pd.DataFrame([{**bar, "symbol": SYMBOL, "date": bar["time"]}
                        for bar in source["frequencies"]["日线"]["bars"]])
    points = [_event(DAYS[0], probability=.4, probability_target="ma_roundtrip",
                     model_lineage={"available_at": "2026-06-30"}, feature_snapshot_hash="unchanged")]
    overlay = {"signal_points": points,
               "rows": [{"symbol": SYMBOL, "date": DAYS[0], "candidate_sources": ["ma"], "high_return_probability": .3},
                        {"symbol": SYMBOL, "date": DAYS[2], "candidate_sources": ["structure"], "high_return_probability": .8}],
               "latest_decision": {"date": DAYS[2], "high_return_probability": .8}}
    original = copy.deepcopy(overlay)
    monkeypatch.setattr(czsc_analysis, "analyze_stock", lambda *a, **k: copy.deepcopy(source))
    monkeypatch.setattr(high_return, "_calendar", lambda identity: (DAYS, "verified-calendar"))
    monkeypatch.setattr(high_return, "_prepare", lambda *a, **k: {"symbol": SYMBOL, "data_end": DAYS[-1], "data_version": "hash", "raw": raw, "features": raw})
    monkeypatch.setattr(high_return, "_overlay", lambda *a, **k: overlay)
    result = high_return.analyze_high_return(symbol=SYMBOL, start=DAYS[0], consensus_version="czsc_agent_consensus_v2")
    assert _row(result, DAYS[0])["agents"]["high_return"]["probability"] == .3
    assert _row(result, DAYS[1])["agents"]["high_return"]["probability"] is None
    assert _row(result, DAYS[2])["agents"]["high_return"]["probability"] == .8
    assert result["events"][0]["reference_price"] == 10.
    assert result["signal_points"] == points and "reference_price" not in result["signal_points"][0]
    assert overlay == original
