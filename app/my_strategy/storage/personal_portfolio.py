"""Persist watchlists with selected local quotes only."""
from __future__ import annotations

from contextlib import closing
import math
from pathlib import Path
import sqlite3
from typing import Any

from my_strategy.adapters.czsc_adapter import _connection, legacy_fuyao_source, normalize_symbol
from my_strategy.core.paths import METADATA_ROOT
from my_strategy.core.tz import local_now


class PersonalStore:
    def __init__(self, db_path: Path | None = None, raw_db: Path | None = None) -> None:
        self.db_path = db_path or METADATA_ROOT / "personal_portfolio.db"
        self.raw_db = raw_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS watchlist (symbol TEXT PRIMARY KEY, added_at TEXT NOT NULL)")
            conn.commit()

    def add_watchlist(self, symbols: list[str]) -> dict[str, Any]:
        symbols = list(dict.fromkeys(normalize_symbol(value) for value in symbols))
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.executemany("INSERT OR IGNORE INTO watchlist VALUES (?,?)", [(value, local_now().isoformat()) for value in symbols])
            conn.commit()
        return self.watchlist()

    def remove_watchlist(self, symbol: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute("DELETE FROM watchlist WHERE symbol=?", (normalize_symbol(symbol),))
            conn.commit()
        return self.watchlist()


    def _quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        quotes = {symbol: dict(symbol=symbol, name="", close=None, price_date=None, source=None,
                               price_basis=None, warnings=["该股票暂无本地日线行情"], quote_available=False) for symbol in symbols}
        if not symbols:
            return quotes
        try:
            with _connection(self.raw_db) as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(stock_daily_normalized)")}
                names = {row[1] for row in conn.execute("PRAGMA table_info(securities)")}
                for symbol in symbols:
                    quote = quotes[symbol]
                    if {"stock", "name"} <= names:
                        name = conn.execute("SELECT name FROM securities WHERE stock=?", (symbol,)).fetchone()
                        quote["name"] = (name[0] or "") if name else ""
                    if not {"stock", "date", "close"} <= columns:
                        quote["warnings"] = ["本地行情字段不完整"]
                        continue
                    extras = [value for value in ("source", "has_trade_price") if value in columns]
                    selected = "date,close" + ("," + ",".join(extras) if extras else "")
                    row = conn.execute(f"SELECT {selected} FROM stock_daily_normalized WHERE stock=? ORDER BY date DESC LIMIT 1", (symbol,)).fetchone()
                    if row is None:
                        continue
                    quote.update(price_date=row["date"], source=row["source"] if "source" in extras else None)
                    try:
                        price = float(row["close"])
                    except (TypeError, ValueError):
                        price = float("nan")
                    if not math.isfinite(price) or price <= 0:
                        quote["warnings"] = ["最新本地收盘价无效"]
                        continue
                    warnings = []
                    legacy = legacy_fuyao_source(quote["source"])
                    if legacy:
                        warnings.append("该收盘价来自旧 Fuyao 行情，复权口径未核验")
                    if "has_trade_price" not in extras or row["has_trade_price"] != 1:
                        warnings.append("该行情的实际交易价格口径未核验")
                    quote.update(close=price, quote_available=True, warnings=warnings,
                                 price_basis="mixed_legacy_adjustment_unverified" if legacy else "unadjusted")
        except FileNotFoundError:
            for quote in quotes.values():
                quote["warnings"] = ["本地行情库不可用"]
        return quotes

    def _items(self, table: str) -> list[dict[str, Any]]:
        # The watchlist supplies a fixed local table name, never a request parameter.
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
        quotes = self._quotes([row["symbol"] for row in rows])
        return [{**row, **quotes[row["symbol"]]} for row in rows]

    @staticmethod
    def _coverage(items: list[dict[str, Any]]) -> dict[str, int]:
        priced = sum(row["quote_available"] for row in items)
        return {"total": len(items), "priced": priced, "missing": len(items) - priced}

    def watchlist(self) -> dict[str, Any]:
        items = self._items("watchlist")
        return {"items": items, "coverage": self._coverage(items), "valuation_basis": "本地日线收盘价"}
