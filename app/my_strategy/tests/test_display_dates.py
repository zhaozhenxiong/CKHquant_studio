"""Latest controls disclose partial coverage and preserve a stock's real date."""
import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from my_strategy.web_dashboard import display_dates


def test_latest_stock_date_is_not_market_date_and_coverage_is_disclosed(tmp_path, monkeypatch):
    path = tmp_path / "raw.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("CREATE TABLE securities(stock TEXT PRIMARY KEY);"
                                 "CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT);")
        connection.executemany("INSERT INTO securities VALUES (?)", [("002436.SZ",), ("000001.SZ",)])
        connection.executemany("INSERT INTO stock_daily_normalized VALUES (?,?)", [
            ("002436.SZ", "2026-10-09"), ("000001.SZ", "2026-10-09"), ("000001.SZ", "2026-10-12")])
    monkeypatch.setattr(display_dates, "local_now", lambda: datetime(2026, 10, 12, 20, tzinfo=ZoneInfo("Asia/Shanghai")))
    dates = display_dates.latest_display_dates("002436.SZ", db_path=path)
    assert dates["latest_date"] == "2026-10-09"
    assert dates["market_latest_date"] == "2026-10-12"
    assert dates["latest_symbols"] == 1 and dates["total_symbols"] == 2
    assert dates["partial_coverage"] and dates["coverage_fraction"] == .5
    monkeypatch.setattr(display_dates, "local_now", lambda: datetime(2026, 10, 12, 14, tzinfo=ZoneInfo("Asia/Shanghai")))
    dates = display_dates.latest_display_dates(db_path=path)
    assert dates["latest_date"] == "2026-10-09"
    assert dates["latest_symbols"] == 2 and not dates["partial_coverage"]
