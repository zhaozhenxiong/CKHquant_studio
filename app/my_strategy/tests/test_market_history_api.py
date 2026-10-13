"""Drill-down belongs to one completed scan and keeps bounded query filters."""
from fastapi.testclient import TestClient

from my_strategy.storage.czsc_market_history import MarketHistoryStore
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.app import create_app
from my_strategy.web_dashboard.tasks import TaskManager


def test_market_history_api_is_run_bound_and_rejects_invalid_filters(tmp_path, monkeypatch):
    results = ResultStore(tmp_path / "results.db", tmp_path / "runs")
    store = MarketHistoryStore(results.db_path)
    run_id, day = "scan-test", "2026-10-09"
    summary = {"run_id": run_id, "dates": [day], "universe_size": 1}
    store.begin(run_id, {})
    store.save_stock(run_id, [{"symbol": "002436.SZ", "date": day, "sector_id": "unknown",
        "counts": {"new_candidates": True, "direction_buy": True}, "new_events": [], "active_events": []}])
    store.finish(run_id, summary)
    results.save("scan", {"run_id": run_id, "market_history": summary})
    results.save("scan", {"run_id": "old-scan", "rows": []})
    results.save("analysis", {"run_id": "analysis-run", "market_history": summary})
    store.begin("building-scan", {})
    results.save("scan", {"run_id": "building-scan", "rows": []})
    manager = TaskManager(tmp_path / "tasks.db", results=results)
    # Paged drill-down must not decode the full detailed scan report.
    def fail_report_load(*args, **kwargs):
        raise AssertionError("market history unexpectedly loaded the full run report")
    monkeypatch.setattr(results, "get", fail_report_load)
    with TestClient(create_app(manager)) as client:
        url = f"/api/runs/{run_id}/market-history"
        response = client.get(url, params={"date": day, "state": "new_candidates", "page_size": 1})
        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert response.json()["rows"][0]["symbol"] == "002436.SZ"
        assert client.get(url, params={"sector": "' OR 1=1 --"}).json()["total"] == 0
        assert client.get(url, params={"state": "anything"}).status_code == 422
        assert client.get(url, params={"page_size": 101}).status_code == 422
        assert client.get(url, params={"date": "2026-01-01"}).status_code == 422
        assert client.get("/api/runs/old-scan/market-history").status_code == 404
        assert client.get("/api/runs/missing-scan/market-history").status_code == 404
        assert client.get("/api/runs/analysis-run/market-history").status_code == 404
        assert client.get("/api/runs/building-scan/market-history").status_code == 422
