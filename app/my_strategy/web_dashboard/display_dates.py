"""Small read-only availability query for date controls, independent of scoring."""
from __future__ import annotations

from datetime import timedelta

from my_strategy.adapters.czsc_adapter import _connection, normalize_symbol
from my_strategy.core.tz import local_now


def latest_display_dates(symbol: str | None = None, *, db_path=None):
    now = local_now()
    cutoff = now.date() if now.hour >= 15 else now.date() - timedelta(days=1)
    with _connection(db_path) as connection:
        market = connection.execute(
            "SELECT MAX(date) FROM stock_daily_normalized WHERE date <= ?", (cutoff.isoformat(),)
        ).fetchone()[0]
        total = connection.execute("SELECT COUNT(*) FROM securities").fetchone()[0]
        covered = connection.execute(
            "SELECT COUNT(DISTINCT stock) FROM stock_daily_normalized WHERE date=?", (market,)
        ).fetchone()[0] if market else 0
        stock = normalize_symbol(symbol) if symbol else None
        stock_latest = connection.execute(
            "SELECT MAX(date) FROM stock_daily_normalized WHERE stock=? AND date<=?",
            (stock, cutoff.isoformat()),
        ).fetchone()[0] if stock else None
    return {"latest_date": stock_latest if stock else market, "symbol": stock,
            "symbol_latest_date": stock_latest, "market_latest_date": market,
            "latest_symbols": covered, "total_symbols": total,
            "coverage_fraction": covered / total if total else None,
            "scope": "latest_stored_closed_daily_bars", "partial_coverage": covered < total}
