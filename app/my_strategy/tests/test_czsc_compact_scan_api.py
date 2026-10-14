"""Current scan transport preserves causal B/S, risk, and saved identities."""
import copy
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.api import AnalysisRequest, compact_scan_view, router


def scan_fixture():
    model = {"model_run_id": "final-model", "manifest_sha256": "actual-manifest",
             "model_input": {"raw_features": [1.] * 1000}}
    source = {"event_id": "source-sell", "original_event_id": "source-sell", "pattern_id": "pattern",
              "symbol": "002436.SZ", "source": "structure", "side": "SELL", "action": "SELL",
              "source_label": "结构", "reference_layer": "source", "date": "2026-10-09",
              "available_at": "2026-10-09T15:00:00+08:00", "confirmed": True,
              "consensus_action": "WAIT", "consensus_qualified": False,
              "consensus_reason_codes": ["insufficient_independent_sources"], "raw_model_inputs": [0.] * 1000}
    risk = {"status": "available", "risk_budget_fraction": .01, "risk_position_cap_fraction": .08,
            "stop_price": 31., "reference_price": 35.,
            "source_boundaries": [{"event_id": "support", "boundary_price": 31.5}],
            "reference_price_band": {"low": 32., "high": 36., "reference_only": True}}
    position = {"status": "pending", "action": "WAIT", "target_position_fraction": None,
                "risk_position_cap_fraction": .08, "current_position_fraction": .5,
                "reasons": ["entry_quality_not_evaluated_reference_population"]}
    row = {"symbol": "002436.SZ", "name": "兴森科技", "date": "2026-10-09",
           "analysis_mode": "reference", "account_scope": "none", "direction": "WAIT",
           "reason_codes": ["insufficient_independent_sources"],
           "agents": {"ma": {"status": "scored", "p_up": .6, "p_down": .2, "p_neutral": .2,
                              "model_identity": model}},
           "reference_risk_plan": risk, "position_reference": position,
           "reference_recommendation": {"action": "WAIT", "reference_risk_plan": risk, "position_reference": position},
           "source_reference_points": [source], "reference_points": [],
           "candidate_evidence": [source], "active_candidates": [{"raw": [0.] * 1000}]}
    return {"run_id": "saved-scan", "strategy_version": "trend_consensus", "analysis_mode": "reference",
            "model_run_id": "final-model", "calendar_run_id": "verified-calendar",
            "run_context": {"config_hash": "saved-config", "stocks": [row["symbol"]]},
            "data_range": {"start": "2026-01-01", "end": "2026-10-09"},
            "coverage": {"requested": 1, "successful": 1}, "rows": [row], "failures": []}


@pytest.fixture
def client(tmp_path):
    store = ResultStore(tmp_path / "results.db", tmp_path / "runs")
    app = FastAPI(); app.include_router(router); app.state.tasks = SimpleNamespace(results=store)
    with TestClient(app) as value:
        yield value, store


def test_compact_reference_preserves_bs_risk_position_and_saved_bytes(client):
    client, store = client; source = scan_fixture(); store.save("scan", source)
    path = store.runs_root / "saved-scan/reports/czsc.json"; original = path.read_bytes()
    before = copy.deepcopy(source)
    assert client.get("/api/runs/saved-scan").json() == source
    response = client.get("/api/runs/saved-scan?view=scan"); assert response.status_code == 200
    view = response.json(); row = view["rows"][0]; full = source["rows"][0]
    for key in ("run_id", "model_run_id", "calendar_run_id", "run_context", "coverage", "data_range"):
        assert view[key] == source[key]
    for key in ("reference_risk_plan", "position_reference", "reference_recommendation", "reference_points"):
        assert row[key] == full[key]
    sell = row["source_reference_points"][0]
    assert sell["side"] == "SELL" and sell["reference_layer"] == "source"
    assert sell["consensus_qualified"] is False and sell["consensus_action"] == "WAIT"
    assert sell["available_at"] == full["source_reference_points"][0]["available_at"]
    assert "raw_model_inputs" not in sell and "active_candidates" not in row
    assert "model_input" not in row["agents"]["ma"]["model_identity"]
    assert row["agents"]["ma"]["model_identity"]["manifest_sha256"] == "actual-manifest"
    assert view["scan_view"]["analysis_mode"] == "reference"
    assert source == before and path.read_bytes() == original


def test_position_and_source_projection_does_not_invent_recommendations():
    source = scan_fixture(); before = copy.deepcopy(source)
    view = compact_scan_view(source)
    assert view["rows"][0]["position_reference"]["target_position_fraction"] is None
    assert view["rows"][0]["reference_points"] == []
    assert source == before


@pytest.mark.parametrize("kind", ["analysis", "model_training", "model_validation"])
def test_compact_view_only_projects_scans(client, kind):
    client, store = client; source = {"run_id": "other-run", "rows": [{"raw": [1., 2.]}]}
    store.save(kind, source)
    assert client.get("/api/runs/other-run").json() == source
    assert client.get("/api/runs/other-run?view=scan").status_code == 422


@pytest.mark.parametrize("kind", ["backtest", "czsc-trend-consensus-v2-train"])
def test_retired_results_have_no_public_read_route(client, kind):
    client, store = client; source = {"run_id": "retired-run", "rows": [{"cash": 1000}]}
    store.save(kind, source)
    assert client.get("/api/runs/retired-run").status_code == 404
    assert client.get("/api/runs/retired-run?view=scan").status_code == 404
    assert store.get("retired-run") == source


def test_illegal_view_and_missing_run_are_rejected(client):
    client, store = client; store.save("scan", scan_fixture())
    assert client.get("/api/runs/saved-scan?view=unknown").status_code == 422
    assert client.get("/api/runs/absent?view=scan").status_code == 404


@pytest.mark.parametrize('status', ['succeeded', 'running', 'failed'])
def test_task_view_only_projects_successful_scans(client, status):
    client, store = client; source = scan_fixture(); store.save('scan', source)
    saved = {'job_id': 'job-1', 'kind': 'scan', 'status': status, 'run_id': source['run_id'],
             'spec': {'model_run_id': 'final-model', 'model_policy': 'auto'},
             'result': source if status == 'succeeded' else None}
    before = copy.deepcopy(saved); client.app.state.tasks.get = lambda job_id: saved
    assert client.get('/api/tasks/job-1').json() == before
    response = client.get('/api/tasks/job-1?view=scan'); assert response.status_code == 200
    if status == 'succeeded':
        body = response.json(); assert body['spec'] == before['spec']
        assert body['result']['rows'][0]['source_reference_points'][0]['side'] == 'SELL'
    else:
        assert response.json() == before
    assert saved == before


@pytest.mark.parametrize('fraction', [True, False, float('nan'), float('inf'), -.01, 1.01])
def test_current_position_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        AnalysisRequest(symbol='002436.SZ', current_position_fraction=fraction)


@pytest.mark.parametrize('field', ['account_initial_cash', 'initial_cash', 'entry_policy'])
def test_analysis_rejects_removed_account_parameters(field):
    with pytest.raises(ValueError):
        AnalysisRequest(symbol='002436.SZ', **{field: 100000})


def test_automatic_requests_inherit_actual_selected_consensus_without_default_override():
    from my_strategy.web_dashboard.api import TaskRequest
    automatic=AnalysisRequest(symbol='002436.SZ')
    assert automatic.consensus_config is None and automatic.consensus_version=='reference_position_v1'
    scan=TaskRequest(kind='scan').spec
    assert scan.consensus_config is None and scan.consensus_version=='reference_position_v1'
    explicit=AnalysisRequest(symbol='002436.SZ',consensus_config={'max_stock_weight':.15})
    assert explicit.consensus_config['max_stock_weight']==.15
