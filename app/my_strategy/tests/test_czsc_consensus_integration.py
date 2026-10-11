"""The display window and saved requests cannot alter historical arbitration."""
import copy
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from my_strategy.web_dashboard.chart_evidence import decorate_analysis


SYMBOL = "600000.SH"
DAYS = ["2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07"]


def fixture():
    bars = [{"time": day, "date": day, "symbol": SYMBOL, "open": 10., "high": 12.,
             "low": 9., "close": 11.} for day in DAYS]
    points = []
    for day, side in zip(DAYS, ["BUY", "BUY", "SELL", "SELL"]):
        for source in ("structure", "wyckoff"):
            points.append({"event_id": f"{source}-{day}", "opportunity_id": day,
                           "symbol": SYMBOL, "date": day, "source": source, "side": side,
                           "event_type": "native_bs1_一买" if source == "structure" else "SOS",
                           "available_at": day + "T15:00:00+08:00", "signal_at": day + "T15:00:00+08:00",
                           "observed_at": day + "T15:00:00+08:00", "anchor_at": day,
                           "input_eligible": True, "candidate_eligible": True, "evidence_visible": True,
                           "dependencies": [], "reason_codes": []})
    features = pd.DataFrame([{**bar, "w_demand_score": .8 if index < 2 else .1,
                             "w_supply_score": .1 if index < 2 else .8,
                             "wyckoff_input_eligible": True} for index, bar in enumerate(bars)])
    result = {"symbol": SYMBOL, "model_family": "ma_trend", "as_of": DAYS[-1],
              "data_end": DAYS[-1], "price_basis": "unadjusted",
              "frequencies": {"日线": {"bars": bars}}, "signal_points": points,
              "events": points, "next_session_decision": {}}
    return result, pd.DataFrame(bars), features


def test_display_start_keeps_warmup_direction_and_identical_overlapping_events():
    result, raw, features = fixture()
    full = decorate_analysis(result, raw=raw, features=features,
                             consensus_version="czsc_agent_consensus_v1")["consensus_decisions"]
    clipped = copy.deepcopy(result)
    clipped["frequencies"]["日线"]["bars"] = clipped["frequencies"]["日线"]["bars"][1:]
    clipped["events"] = [point for point in clipped["events"] if point["date"] >= DAYS[1]]
    actual = decorate_analysis(clipped, raw=raw, features=features,
                               consensus_version="czsc_agent_consensus_v1")["consensus_decisions"]
    assert [(event["date"], event["side"]) for event in full["events"]] == [(DAYS[0], "BUY"), (DAYS[2], "SELL")]
    assert actual["events"] == [event for event in full["events"] if event["date"] >= DAYS[1]]
    assert actual["rows"] == full["rows"][1:]
    assert actual["state_before_display"] == full["rows"][0]["state_after"]
    assert actual["prefix_bars"] == 4


def test_asof_suffix_cannot_change_already_observable_final_decisions():
    result, raw, features = fixture()
    full = decorate_analysis(result, raw=raw, features=features,
                             consensus_version="czsc_agent_consensus_v1")["consensus_decisions"]
    result["as_of"] = DAYS[1]
    historical = decorate_analysis(result, raw=raw, features=features,
                                   consensus_version="czsc_agent_consensus_v1")["consensus_decisions"]
    assert historical["rows"] == full["rows"][:2]
    assert historical["events"] == [event for event in full["events"] if event["date"] <= DAYS[1]]
    assert historical["prefix_bars"] == 2


def test_api_binds_effective_config_and_never_passes_arbitration_to_frozen_strategy(monkeypatch):
    from my_strategy.services import czsc_analysis
    from my_strategy.web_dashboard.api import router

    result, _, _ = fixture()
    result.update(data_version="fixture")
    forwarded = []
    monkeypatch.setattr(czsc_analysis, "analyze_stock", lambda **kwargs: forwarded.append(kwargs) or copy.deepcopy(result))
    monkeypatch.setattr(czsc_analysis, "strategy_config", lambda: {})
    app = FastAPI()
    app.state.tasks = SimpleNamespace(results=SimpleNamespace(save=lambda kind, value: value))
    app.include_router(router)
    with TestClient(app) as client:
        response = client.post("/api/analysis", json={"symbol": SYMBOL, "research": False,
                                                     "consensus_version": "czsc_agent_consensus_v1",
                                                     "consensus_config": {"min_buy_support": 1}})
        assert response.status_code == 200
        data = response.json()
        assert "consensus_config" not in forwarded[0]
        assert "consensus_version" not in forwarded[0]
        assert data["request"]["consensus_version"] == "czsc_agent_consensus_v1"
        assert data["request"]["consensus_config"] == data["consensus_decisions"]["config"]
        assert data["consensus_decisions"]["config"]["min_buy_support"] == 1
        assert data["run_context"]["config_hash"]
        invalid = client.post("/api/analysis", json={"symbol": SYMBOL, "consensus_config": {"cooldown_bars": True}})
        assert invalid.status_code == 422


def test_historical_score_identity_uses_verified_entry_manifest_without_mutating_overlay(monkeypatch):
    from my_strategy.web_dashboard.high_return import _display_model_lineage
    from my_strategy.services import czsc_high_return_runtime as runtime, czsc_high_return_models as models

    identity = {"model_run_id": "frozen-run", "checkpoint": "2026Q3", "available_at": "2026-06-30",
                "historical_reconstruction": True, "actual_training_completed_at": "2026-10-11"}
    overlay = {"rows": [{"high_return_probability": .8, "model_resolution": identity}]}
    original = copy.deepcopy(overlay)
    manifest = {"head": "entry", "available_at": "2026-06-30", "train_label_end": "2026-03-31",
                "training_completed_at": "2026-10-11", "target_contract_hash": "contract",
                "manifest_sha256": "verified", "fold": {"train_end": "2026-03-31"}}
    monkeypatch.setattr(runtime, "_read_run", lambda run: ("root", {"model_bundles": [{"name": "2026Q3"}]}, {}))
    monkeypatch.setattr(runtime, "_head_path", lambda *args: "verified-path")
    monkeypatch.setattr(models, "load_head", lambda path: copy.deepcopy(manifest))
    result = _display_model_lineage(overlay)
    assert overlay == original
    assert result["rows"][0]["model_resolution"]["train_label_end"] == "2026-03-31"
    assert result["rows"][0]["model_resolution"]["manifest_sha256"] == "verified"
    manifest["available_at"] = "2026-09-30"
    with pytest.raises(ValueError, match="冻结清单不一致"):
        _display_model_lineage(overlay)
