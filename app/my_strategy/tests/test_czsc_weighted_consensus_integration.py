"""Weighted arbitration stays causal and saved v1 requests stay reproducible."""
import copy
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from my_strategy.services.czsc_agent_consensus import consensus_config
from my_strategy.tests.test_czsc_consensus_integration import fixture, SYMBOL, DAYS
from my_strategy.web_dashboard.chart_evidence import decorate_analysis


V1 = "czsc_agent_consensus_v1"
V2 = "czsc_agent_consensus_v2"


def weighted_fixture():
    result, raw, features = fixture()
    result["model_family"] = "high_return"
    rows = [{"symbol": SYMBOL, "date": day, "source_candidate_eligible": True,
             "high_return_probability": .2, "candidate_sources": ["structure", "wyckoff"],
             "probability_target": "actual_complete_fee_net_return_8pct_and_2R",
             "model_resolution": {"head": "entry", "available_at": "2026-06-30",
                                  "train_label_end": "2026-03-31"},
             "raw_risk_plan": {"symbol": SYMBOL, "signal_date": day, "entry_allowed": True}}
            for day in DAYS]
    return result, raw, features, {"rows": rows, "signal_points": result["signal_points"]}


def test_weighted_exact_boundaries_full_prefix_and_display_crop():
    result, raw, features, overlay = weighted_fixture()
    original = copy.deepcopy(result)
    full = decorate_analysis(result, raw=raw, features=features, overlay=overlay, consensus_version=V2)["consensus_decisions"]
    assert full["version"] == V2
    assert [(event["date"], event["side"]) for event in full["events"]] == [(DAYS[0], "BUY"), (DAYS[2], "SELL")]
    assert full["rows"][0]["buy_base_score"] == .5
    assert full["rows"][0]["buy_quality_bonus"] == .1
    assert full["rows"][0]["buy_total_score"] == .6
    assert full["rows"][2]["sell_base_score"] == .5
    assert result == original
    clipped = copy.deepcopy(result)
    clipped["frequencies"]["日线"]["bars"] = clipped["frequencies"]["日线"]["bars"][1:]
    clipped["events"] = [point for point in clipped["events"] if point["date"] >= DAYS[1]]
    actual = decorate_analysis(clipped, raw=raw, features=features, overlay=overlay, consensus_version=V2)["consensus_decisions"]
    assert actual["rows"] == full["rows"][1:]
    assert actual["events"] == [event for event in full["events"] if event["date"] >= DAYS[1]]
    assert actual["state_before_display"] == "bullish"


def test_weighted_history_asof_uses_only_visible_scores_and_events():
    result, raw, features, overlay = weighted_fixture()
    full = decorate_analysis(result, raw=raw, features=features, overlay=overlay, consensus_version=V2)["consensus_decisions"]
    result["as_of"] = DAYS[1]
    historical = decorate_analysis(result, raw=raw, features=features, overlay=overlay, consensus_version=V2)["consensus_decisions"]
    assert historical["rows"] == full["rows"][:2]
    assert historical["events"] == full["events"][:1]
    overlay["rows"][0]["model_resolution"]["available_at"] = DAYS[1]
    unavailable = decorate_analysis(result, raw=raw, features=features, overlay=overlay, consensus_version=V2)["consensus_decisions"]
    assert unavailable["rows"][0]["action"] == "WAIT"
    assert unavailable["rows"][0]["buy_quality_bonus"] == 0


def api_client(monkeypatch):
    from my_strategy.services import czsc_analysis
    from my_strategy.web_dashboard.api import router
    result, _, _ = fixture()
    result["data_version"] = "fixture"
    forwarded = []
    monkeypatch.setattr(czsc_analysis, "analyze_stock", lambda **kwargs: forwarded.append(kwargs) or copy.deepcopy(result))
    monkeypatch.setattr(czsc_analysis, "strategy_config", lambda: {})
    app = FastAPI()
    app.state.tasks = SimpleNamespace(results=SimpleNamespace(save=lambda kind, value: value))
    app.include_router(router)
    return TestClient(app), forwarded


def test_api_explicit_v2_saves_effective_version_config(monkeypatch):
    client, forwarded = api_client(monkeypatch)
    with client:
        response = client.post("/api/analysis", json={"symbol": SYMBOL, "research": False, "consensus_version": V2})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["consensus_decisions"]["version"] == V2
        assert data["request"]["consensus_version"] == V2
        assert data["request"]["consensus_config"] == data["consensus_decisions"]["config"]
        assert data["request"]["consensus_config"]["source_weights"] == {"ma": .4, "structure": .2, "wyckoff": .3}
        assert "consensus_version" not in forwarded[0]
        assert "consensus_config" not in forwarded[0]
        replay = client.post("/api/analysis", json={**data["request"], "as_of": DAYS[1]})
        assert replay.status_code == 200, replay.text
        assert replay.json()["consensus_decisions"]["config_hash"] == data["consensus_decisions"]["config_hash"]


def test_api_legacy_saved_full_config_routes_to_v1_without_explicit_version(monkeypatch):
    client, _ = api_client(monkeypatch)
    legacy = consensus_config(version=V1)
    with client:
        response = client.post("/api/analysis", json={"symbol": SYMBOL, "research": False,
                                                     "consensus_config": legacy})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["consensus_decisions"]["version"] == V1
        assert data["request"]["consensus_version"] == V1
        assert data["consensus_decisions"]["config"] == legacy
        assert [(event["date"], event["side"]) for event in data["consensus_decisions"]["events"]] == [(DAYS[0], "BUY"), (DAYS[2], "SELL")]


@pytest.mark.parametrize("version,config", [("unknown", None), (V1, {"version": V2}),
                                           (V2, {"version": V1}),
                                           (V2, {"high_return_quality_weight": True})])
def test_api_rejects_unknown_or_mismatched_arbitration_versions(monkeypatch, version, config):
    client, forwarded = api_client(monkeypatch)
    with client:
        response = client.post("/api/analysis", json={"symbol": SYMBOL, "research": False,
                                                     "consensus_version": version, "consensus_config": config})
        assert response.status_code == 422
        assert not forwarded
