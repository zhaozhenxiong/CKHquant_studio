"""Versioned scan facts in the CZSC result database, with bounded drill-down."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from my_strategy.core.paths import PROCESSED_DATA_ROOT

VERSION = "czsc_market_history_v1"
STATES = frozenset({"direction_buy", "direction_sell", "candidates", "new_candidates",
    "risk_blocked", "hr_scored_passed", "hr_scored_rejected", "hr_unscored", "account_buy",
    "account_sell", "filled_buy", "filled_sell", "holding", "failed", "missing_price"})


def _path(db_path=None):
    return Path(db_path or PROCESSED_DATA_ROOT / "czsc" / "results.db")


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class MarketHistoryStore:
    """A new run cannot overwrite an earlier run's daily evidence."""

    def __init__(self, db_path=None):
        self.db_path = _path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS czsc_market_history_runs (
                    run_id TEXT PRIMARY KEY, version TEXT NOT NULL,
                    status TEXT NOT NULL, metadata_json TEXT NOT NULL,
                    summary_json TEXT
                );
                CREATE TABLE IF NOT EXISTS czsc_market_history_facts (
                    run_id TEXT NOT NULL, symbol TEXT NOT NULL, date TEXT NOT NULL,
                    sector_id TEXT NOT NULL, fact_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, symbol, date),
                    FOREIGN KEY (run_id) REFERENCES czsc_market_history_runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_market_history_date_sector
                    ON czsc_market_history_facts(run_id, date, sector_id, symbol);
                CREATE INDEX IF NOT EXISTS idx_market_history_sector_date
                    ON czsc_market_history_facts(run_id, sector_id, date, symbol);
            """)
            conn.commit()

    def begin(self, run_id, metadata):
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute("INSERT INTO czsc_market_history_runs VALUES (?, ?, ?, ?, NULL)",
                (run_id, VERSION, "building", _json(metadata)))
            conn.commit()

    def save_stock(self, run_id, facts):
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            run = conn.execute("SELECT status FROM czsc_market_history_runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None or run[0] != "building":
                raise ValueError("只能追加尚在构建的市场历史运行")
            conn.executemany("INSERT INTO czsc_market_history_facts VALUES (?, ?, ?, ?, ?)",
                [(run_id, row["symbol"], row["date"], row["sector_id"], _json(row)) for row in facts])
            conn.commit()

    def finish(self, run_id, summary):
        expected = int(summary["universe_size"]) * len(summary["dates"])
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM czsc_market_history_facts WHERE run_id=?",
                (run_id,)).fetchone()[0]
            if count != expected:
                raise ValueError(f"市场历史事实不完整：{count}/{expected}")
            changed = conn.execute("UPDATE czsc_market_history_runs SET status='complete', summary_json=? "
                "WHERE run_id=? AND status='building'", (_json(summary), run_id)).rowcount
            if changed != 1:
                raise ValueError("市场历史运行不存在或已经完成")
            conn.commit()

    def query_summary(self, run_id):
        return market_history_summary(run_id, db_path=self.db_path)


def _read_connection(db_path):
    path = _path(db_path).resolve()
    if not path.is_file():
        raise KeyError("该运行没有市场历史统计")
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)
    conn.execute("PRAGMA query_only=ON")
    return conn


def _require_complete(conn, run_id):
    present = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='czsc_market_history_runs'").fetchone()
    if not present:
        raise KeyError("该运行没有市场历史统计；请用当前趋势策略重新扫描")
    row = conn.execute("SELECT status, summary_json FROM czsc_market_history_runs WHERE run_id=?",
        (run_id,)).fetchone()
    if row is None:
        raise KeyError("该运行没有市场历史统计；旧策略扫描不能混入当前统计")
    if row[0] != "complete":
        raise ValueError("市场历史统计尚未完成")
    return json.loads(row[1])


def market_history_summary(run_id, *, db_path=None):
    with closing(_read_connection(db_path)) as conn:
        return _require_complete(conn, run_id)


def market_history_facts(run_id, *, date=None, sector=None, event_type=None, state=None,
                         page=1, page_size=25, db_path=None):
    """Every filter is bound; state expressions come only from a fixed whitelist."""
    if state and state not in STATES:
        raise ValueError("不支持的市场状态筛选")
    try:
        page, page_size = int(page), int(page_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("分页参数需要整数") from exc
    if page < 1 or not 1 <= page_size <= 100:
        raise ValueError("页码须从1开始，每页须为1至100条")
    clauses, params = ["run_id=?"], [run_id]
    if date:
        clauses.append("date=?"); params.append(date)
    if sector:
        clauses.append("sector_id=?"); params.append(sector)
    if state:
        clauses.append("json_extract(fact_json, ?) = 1")
        params.append("$.counts." + state)
    if event_type:
        clauses.append("(EXISTS (SELECT 1 FROM json_each(fact_json, '$.new_events') "
            "WHERE json_extract(value, '$.type')=?) OR EXISTS "
            "(SELECT 1 FROM json_each(fact_json, '$.active_events') "
            "WHERE json_extract(value, '$.type')=?))")
        params.extend([event_type, event_type])
    where = " AND ".join(clauses)
    with closing(_read_connection(db_path)) as conn:
        summary = _require_complete(conn, run_id)
        if date and date not in summary["dates"]:
            raise ValueError("所选日期不在该扫描的历史窗口内")
        count = conn.execute("SELECT COUNT(*) FROM czsc_market_history_facts WHERE " + where,
            params).fetchone()[0]
        rows = conn.execute("SELECT fact_json FROM czsc_market_history_facts WHERE " + where +
            " ORDER BY date DESC, symbol LIMIT ? OFFSET ?", [*params, page_size, (page - 1) * page_size]).fetchall()
    return {"run_id": run_id, "rows": [json.loads(row[0]) for row in rows], "total": count,
        "page": page, "page_size": page_size,
        "filters": {"date": date, "sector": sector, "event_type": event_type, "state": state}}
