"""Watchlist persistence, actual local quotes and retired account endpoints."""
from __future__ import annotations

from contextlib import closing
import hashlib
import sqlite3

from fastapi.testclient import TestClient
import pytest

from my_strategy.storage.czsc_results import ResultStore
from my_strategy.storage.personal_portfolio import PersonalStore
from my_strategy.web_dashboard.app import create_app
from my_strategy.web_dashboard.tasks import TaskManager


@pytest.fixture
def store(tmp_path):
    raw = tmp_path / "raw.db"
    with closing(sqlite3.connect(raw)) as conn:
        conn.execute("CREATE TABLE securities (stock TEXT PRIMARY KEY,name TEXT)")
        conn.executemany("INSERT INTO securities VALUES (?,?)", [("000001.SZ", "平安银行"), ("600036.SH", "招商银行"), ("999999.SZ", "无行情测试")])
        conn.execute("CREATE TABLE stock_daily_normalized (stock TEXT,date TEXT,close REAL,source TEXT,has_trade_price INTEGER,PRIMARY KEY(stock,date))")
        conn.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?)", [
            ("000001.SZ", "2026-09-25", 8, "test", 1),
            ("000001.SZ", "2026-09-28", 12, "tushare", 1),
            ("600036.SH", "2026-09-25", 20, "fuyao", 1),
        ])
        conn.commit()
    return PersonalStore(tmp_path / "personal.db", raw)


def test_persisted_watchlist_is_additive_and_idempotent(store):
    store.add_watchlist(["000001.SZ", "000001.SZ"])
    created = store.watchlist()["items"][0]["added_at"]
    reopened = PersonalStore(store.db_path, store.raw_db)
    result = reopened.add_watchlist(["000001.SZ", "600036.SH"])
    assert [row["symbol"] for row in result["items"]] == ["000001.SZ", "600036.SH"]
    assert result["items"][0]["added_at"] == created
    assert reopened.remove_watchlist("000001.SZ")["coverage"]["total"] == 1
    assert store.watchlist()["items"][0]["symbol"] == "600036.SH"


def test_latest_local_quote_has_actual_date_source_and_readonly_raw(store):
    original = hashlib.sha256(store.raw_db.read_bytes()).hexdigest()
    row = store.add_watchlist(["000001.SZ", "600036.SH"])["items"][0]
    assert (row["name"], row["close"], row["price_date"], row["source"]) == ("平安银行", 12, "2026-09-28", "tushare")
    legacy = store.watchlist()["items"][1]
    assert legacy["price_date"] == "2026-09-25"
    assert legacy["price_basis"] == "mixed_legacy_adjustment_unverified"
    assert "Fuyao" in legacy["warnings"][0]
    assert hashlib.sha256(store.raw_db.read_bytes()).hexdigest() == original


def test_missing_watchlist_quotes_remain_visible_without_invented_price(store):
    result = store.add_watchlist(["000001.SZ", "999999.SZ"])
    assert result["coverage"] == {"total": 2, "priced": 1, "missing": 1}
    missing = result["items"][1]
    assert missing["name"] == "无行情测试"
    assert missing["close"] is None and missing["quote_available"] is False
    assert missing["warnings"]


def test_missing_raw_db_keeps_watchlist_records_available(store):
    store.add_watchlist(["000001.SZ"])
    result = PersonalStore(store.db_path, store.raw_db.parent / "absent.db").watchlist()
    assert result["items"][0]["symbol"] == "000001.SZ"
    assert result["items"][0]["close"] is None
    assert result["coverage"]["missing"] == 1
    assert "行情库不可用" in result["items"][0]["warnings"][0]


@pytest.mark.parametrize("price", [None, 0, -1, float("inf"), float("nan")])
def test_invalid_latest_watchlist_quote_is_not_substituted(store, price):
    with closing(sqlite3.connect(store.raw_db)) as conn:
        conn.execute("UPDATE stock_daily_normalized SET close=? WHERE stock=? AND date=?", (price, "000001.SZ", "2026-09-28"))
        conn.commit()
    row = store.add_watchlist(["000001.SZ"])["items"][0]
    assert not row["quote_available"] and row["close"] is None
    assert row["price_date"] == "2026-09-28"


def test_storage_rejects_invalid_symbol_without_partially_persisting(store):
    with pytest.raises(ValueError):
        store.add_watchlist(["000001.SZ", "../escape"])
    assert store.watchlist()["items"] == []


def test_watchlist_http_crud_and_retired_account_endpoints(store, tmp_path):
    manager = TaskManager(tmp_path / "jobs.db", ResultStore(tmp_path / "results.db", tmp_path / "runs"), workers=1)
    with TestClient(create_app(manager, store)) as client:
        assert client.get("/watchlist").status_code == 200
        for path in ("/holdings", "/backtest", "/api/holdings"):
            assert client.get(path).status_code == 404
        assert client.put("/api/holdings/000001.SZ",json={"shares":137,"average_cost":10}).status_code==404
        assert client.delete("/api/holdings/000001.SZ").status_code==404
        response = client.post("/api/watchlist", json={"symbols": ["000001.SZ", "600036.SH", "000001.SZ"]})
        assert response.status_code == 200 and response.json()["coverage"]["total"] == 2
        assert client.post("/api/watchlist", json={"symbols": ["bad"]}).status_code == 422
        assert client.post("/api/watchlist", json={"symbols": []}).status_code == 422
        assert client.delete("/api/watchlist/000001.SZ").json()["coverage"]["total"] == 1
        assert client.delete("/api/watchlist/000001.SZ").status_code == 200
        assert client.post("/api/watchlist",json={"symbols":["000001.SZ"]},headers={"Origin":"https://other.example"}).status_code==403


def test_watchlist_records_survive_new_app_instance(store, tmp_path):
    results = ResultStore(tmp_path / "results.db", tmp_path / "runs")
    first = TaskManager(tmp_path / "jobs.db", results, workers=1)
    with TestClient(create_app(first, store)) as client:
        assert client.post("/api/watchlist", json={"symbols": ["000001.SZ"]}).status_code == 200
    reopened = PersonalStore(store.db_path, store.raw_db)
    second = TaskManager(tmp_path / "jobs.db", results, workers=1)
    with TestClient(create_app(second, reopened)) as client:
        assert client.get("/api/watchlist").json()["items"][0]["symbol"] == "000001.SZ"


def test_watchlist_endpoints_apply_global_auth(store, tmp_path, monkeypatch):
    monkeypatch.setenv("KHQUANT_API_TOKEN", "test-personal-secret")
    manager = TaskManager(tmp_path / "jobs.db", ResultStore(tmp_path / "results.db", tmp_path / "runs"), workers=1)
    with TestClient(create_app(manager, store)) as client:
        assert client.get("/api/watchlist").status_code == 401
        assert client.post("/api/watchlist", json={"symbols": ["000001.SZ"]}).status_code == 401
        headers = {"Authorization": "Bearer test-personal-secret"}
        response = client.get("/api/watchlist", headers=headers)
        assert response.status_code == 200 and "test-personal-secret" not in response.text
        assert client.post("/api/watchlist",json={"symbols":["000001.SZ"]},headers=headers).status_code==200
