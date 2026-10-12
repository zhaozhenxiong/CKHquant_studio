"""Market statistics retain causal event clocks and every requested stock/day."""
from datetime import date, timedelta
import json
import sqlite3

import pytest

from my_strategy.services.czsc_market_history import (MarketHistoryBuilder, load_classification,
    stock_facts, window_dates)
from my_strategy.storage.czsc_market_history import (MarketHistoryStore, market_history_facts,
    market_history_summary)


DATES = ["2026-09-28", "2026-09-29", "2026-09-30"]


def event(identity, day, *, side="BUY", kind="Test", confirmed=True, **extra):
    return {"event_id": identity, "original_event_id": identity, "date": day,
        "available_at": day + "T15:00:00+08:00", "observed_at": day + "T15:00:00+08:00",
        "event_type": kind, "side": side, "confirmed": confirmed, "input_eligible": True,
        "event_strength": .4, **extra}


def row(day, *, direction="BUY", status="scored_passed", candidate="candidate-1"):
    return {"date": day, "direction": direction, "fresh_candidate_ids": [candidate],
        "active_candidates": [{"event_id": candidate, "side": "BUY"}], "entry_plan": {"entry_allowed": True},
        "agents": {"ma": {"status": "scored"}, "structure": {"status": "scored"},
            "wyckoff": {"status": "scored", "current_demand": .2, "current_supply": .4,
                "event_demand": 0., "event_supply": 0., "buy_strength": .2, "sell_strength": .4},
            "high_return": {"status": "scored", "evaluation_status": status,
                "probability": .3 if status == "scored_passed" else None, "threshold": .1}}}


def overlay():
    first = event("test", DATES[0]); later = event("sow", DATES[1], side="SELL", kind="SOW")
    return {"rows": [row(DATES[0]), row(DATES[1], direction="SELL", status="actual_existing_position")],
        "native_pack": {"wyckoff_events": [first, first, later],
            "evaluations": [{"date": DATES[0], "wyckoff": {"support_facts": [first, first]}},
                {"date": DATES[1], "wyckoff": {"support_facts": [first, later], "invalidations_at_day": [{}]}}]},
        "account_actions": {"rows": [{"date": DATES[0], "action": "BUY"},
                {"date": DATES[1], "action": "HOLD"}, {"date": DATES[2], "action": "SELL"}],
            "daily": [{"date": DATES[0], "shares": 0}, {"date": DATES[1], "shares": 100},
                {"date": DATES[2], "shares": 100, "price_available": False}],
            "ledger": [{"date": DATES[1], "action": "BUY"}]}}


def test_window_uses_verified_sessions_and_never_appends_weekends():
    day, calendar = date(2026, 7, 1), []
    while len(calendar) < 35:
        if day.weekday() < 5:
            calendar.append(day.isoformat())
        day += timedelta(days=1)
    result = window_dates(calendar, "2026-12-31")
    assert result == calendar[-30:]
    assert len(result) == 30
    with pytest.raises(ValueError, match="有序唯一"):
        window_dates(calendar + [calendar[-1]], "2026-12-31")


def test_fixed_pool_keeps_missing_and_failed_rows_and_actual_account_states():
    builder = MarketHistoryBuilder(["a", "b", "c"], DATES, DATES[-1], run_id="scan")
    a = builder.add_stock("a", overlay=overlay())
    builder.add_stock("b", overlay={"rows": [row(DATES[0], direction="WAIT", status="no_candidate")]})
    c = builder.add_stock("c", failure={"error": "broken input"})
    summary = builder.finish()
    first, second, third = summary["daily"]
    assert first["universe_size"] == 3
    assert first["coverage"] == {"price_available": 2, "input_available": 2, "model_scored": 2, "account_available": 1, "failed": 1, "missing_price": 0}
    assert second["coverage"]["missing_price"] == 1
    assert third["coverage"]["missing_price"] == 2
    assert first["counts"]["account_buy"] == 1
    assert second["counts"]["filled_buy"] == 1
    # A duration exit and inventory are known from the Broker calendar even
    # when this session has no quote. No direction is copied from yesterday.
    assert third["counts"]["account_sell"] == third["counts"]["holding"] == 1
    assert a[-1]["direction"] == "UNAVAILABLE"
    assert a[-1]["holding_shares"] == 100
    assert c[0]["holding_shares"] is None
    assert all(fact["counts"]["failed"] and fact["high_return_probability"] is None for fact in c)
    assert summary["sectors"][0]["member_count"] == 3
    assert first["mean_support"]["event_demand"] == 0.
    assert third["mean_support"]["event_demand"] is None
    assert first["buy_rate"] == pytest.approx(1 / 3)
    assert first["counts"]["new_candidates"] == 2
    assert second["counts"]["new_candidates"] == 0


def test_first_visible_events_are_deduplicated_without_backdating_or_pending_support():
    value = overlay()
    pending = event("pending", DATES[0], confirmed=False, kind="Spring")
    future = event("future", DATES[0], available_at=DATES[1] + "T15:00:00+08:00")
    before = event("old", "2026-09-25")
    value["native_pack"]["wyckoff_events"] += [pending, future, before]
    value["native_pack"]["evaluations"][0]["wyckoff"]["support_facts"] += [pending, future, before]
    builder = MarketHistoryBuilder(["a"], DATES, DATES[-1], run_id="scan")
    facts = builder.add_stock("a", overlay=value)
    summary = builder.finish()
    assert {item["id"] for item in facts[0]["new_events"]} == {"test", "pending"}
    assert {item["id"] for item in facts[0]["active_events"]} == {"test", "old"}
    assert {item["id"] for item in facts[1]["new_events"]} == {"sow", "future"}
    assert summary["daily"][0]["events"]["new_total"] == 2
    assert summary["daily"][1]["events"]["active_total"] == 2
    assert next(item for item in summary["event_types"] if item["id"] == "Test")["active_stock_days"] == 2
    # Later facts cannot alter the first day's counts or means.
    prefix = {**value, "rows": value["rows"][:1], "native_pack": {
        **value["native_pack"], "evaluations": value["native_pack"]["evaluations"][:1]}}
    assert stock_facts("a", DATES[:1], overlay=prefix)[0] == facts[0]


def test_industry_read_is_read_only_and_snapshot_conflicts_remain_unknown(tmp_path):
    raw = tmp_path / "raw.db"
    with sqlite3.connect(raw) as conn:
        conn.execute("CREATE TABLE untouched (v INTEGER)")
    before = raw.read_bytes()
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"kind": "industry", "source": "public_verified", "observed_at": "2026-10-12T20:00:00+08:00",
        "boards": [{"id": "i1", "name": "行业1", "members": ["a", "c"]},
            {"id": "i2", "name": "行业2", "members": ["b", "c"]}]}))
    mapping, metadata = load_classification(["a", "b", "c", "d"], db_path=raw, snapshot_path=snapshot)
    assert raw.read_bytes() == before
    assert metadata["mode"] == "current_snapshot"
    assert metadata["classified_symbols"] == metadata["unknown_symbols"] == 2
    assert mapping["c"]["id"] == mapping["d"]["id"] == "unknown"
    assert metadata["source_hash"]
    assert "不代表历史" in metadata["note"]
    assert metadata["diagnostics"]


def test_storage_filters_complete_runs_only_and_preserves_run_identity(tmp_path):
    db = tmp_path / "results.db"
    store = MarketHistoryStore(db)
    builder = MarketHistoryBuilder(["a", "b"], DATES, DATES[-1], run_id="scan")
    store.begin("scan", {"dates": DATES})
    store.save_stock("scan", builder.add_stock("a", overlay=overlay()))
    with pytest.raises(ValueError, match="尚未完成"):
        market_history_facts("scan", db_path=db)
    with pytest.raises(ValueError, match="未覆盖"):
        builder.finish()
    store.save_stock("scan", builder.add_stock("b", failure={"error": "x"}))
    summary = builder.finish(); store.finish("scan", summary)
    assert market_history_summary("scan", db_path=db) == summary
    assert market_history_facts("scan", db_path=db)["total"] == 6
    found = market_history_facts("scan", date=DATES[1], state="filled_buy", event_type="SOW", db_path=db)
    assert found["total"] == 1 and found["rows"][0]["symbol"] == "a"
    assert market_history_facts("scan", state="failed", page_size=1, page=2, db_path=db)["rows"][0]["date"] == DATES[1]
    assert market_history_facts("scan", event_type="' OR 1=1 --", db_path=db)["total"] == 0
    with pytest.raises(ValueError, match="状态"):
        market_history_facts("scan", state="' OR 1=1 --", db_path=db)
    with pytest.raises(ValueError, match="窗口"):
        market_history_facts("scan", date="2020-01-01", db_path=db)
    with pytest.raises(sqlite3.IntegrityError):
        store.begin("scan", {})
    with pytest.raises(ValueError, match="尚在构建"):
        store.save_stock("scan", [stock_facts("new", DATES)[0]])
    with pytest.raises(KeyError, match="旧策略"):
        market_history_facts("older_scan", db_path=db)


def test_old_result_database_queries_never_create_new_tables(tmp_path):
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE czsc_runs (run_id TEXT)")
    before = db.read_bytes()
    with pytest.raises(KeyError, match="重新扫描"):
        market_history_facts("old", db_path=db)
    assert db.read_bytes() == before


def test_scan_uses_existing_overlays_once_per_stock_and_persists_bounded_facts(tmp_path, monkeypatch):
    import pandas as pd
    from types import SimpleNamespace
    from my_strategy.web_dashboard import trend_strategy as route
    from my_strategy.web_dashboard import high_return
    from my_strategy.adapters import czsc_adapter
    from my_strategy.services import czsc_trend_runtime
    from my_strategy.storage import czsc_results
    from my_strategy.services import czsc_market_history

    db = tmp_path / "results.db"; reports = tmp_path / "runs"
    context = SimpleNamespace(run_id="scan", subdir=lambda name: reports / "scan" / name, to_dict=lambda: {"run_id": "scan"})
    real_store = czsc_results.ResultStore(db_path=db, runs_root=reports)
    monkeypatch.setattr(czsc_adapter, "latest_market_date", lambda end=None: DATES[-1])
    monkeypatch.setattr(czsc_adapter, "list_symbols", lambda: [{"symbol": "a", "name": "A"}, {"symbol": "b", "name": "B"}])
    monkeypatch.setattr(high_return, "_calendar", lambda value: (DATES, "calendar"))
    monkeypatch.setattr(czsc_trend_runtime, "load_model_run", lambda value: (None, None))
    monkeypatch.setattr(route, "create_run_context", lambda **values: context)
    monkeypatch.setattr(czsc_results, "ResultStore", lambda: real_store)
    monkeypatch.setattr(czsc_market_history, "load_classification", lambda symbols: ({s: {"id": "unknown", "name": "未知行业"} for s in symbols},
        {"mode": "unavailable", "note": "行业未知", "classified_symbols": 0, "unknown_symbols": len(symbols)}))
    calls = []
    def batch(symbols, end, dates, params, workers, **kwargs):
        calls.extend(symbols)
        value = overlay(); value["latest_decision"] = value["rows"][-1]
        value["features"] = pd.DataFrame({"date": DATES[:2]})
        yield ("a", {"data_version": "v1", "data_end": end}, value), None
        yield None, {"symbol": "b", "error": "broken"}
    monkeypatch.setattr(route, "_batch", batch)
    result = route.scan_trend_consensus()
    assert calls == ["a", "b"]
    assert result["market_history"]["universe_size"] == 2
    assert len(result["market_history"]["daily"]) == 3
    assert market_history_facts("scan", db_path=db)["total"] == 6
    assert real_store.get("scan")["market_history"] == result["market_history"]
