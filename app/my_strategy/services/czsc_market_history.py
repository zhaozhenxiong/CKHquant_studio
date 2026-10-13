"""Summarize existing causal overlays without changing their decisions."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

from my_strategy.core.paths import METADATA_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID
from my_strategy.storage.czsc_market_history import VERSION

UNKNOWN = "unknown"
COUNT_KEYS = ("direction_buy", "direction_sell", "candidates", "new_candidates", "risk_blocked",
    "hr_scored_passed", "hr_scored_rejected", "hr_unscored", "account_buy", "account_sell",
    "filled_buy", "filled_sell", "holding", "failed", "missing_price")
SUPPORT_KEYS = ("continuous_demand", "continuous_supply", "event_demand", "event_supply",
    "effective_demand", "effective_supply")
EVENT_LABELS = {"Spring": "Spring · 弹簧", "Test": "Test · 测试", "SOS": "SOS · 强势",
    "LPS": "LPS · 最后支撑", "Upthrust": "Upthrust · 上冲回落", "SOW": "SOW · 弱势",
    "LPSY": "LPSY · 最后供给"}
CLASSIFICATION_NOTE = "按当前行业快照归组历史信号；当前成员关系不代表历史时点板块归属。"


def _instant(value):
    if value is None:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.replace(tzinfo=ZoneInfo("Asia/Shanghai")) if stamp.tzinfo is None else stamp
    except (TypeError, ValueError):
        return None


def _day(value):
    stamp = _instant(value)
    return stamp.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat() if stamp else str(value)[:10]


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def load_classification(symbols, *, db_path=None, snapshot_path=None):
    """Read current classification only; never initialize or update the raw DB."""
    from my_strategy.adapters.czsc_adapter import raw_path
    wanted = set(symbols)
    mapping, diagnostics = {}, []
    source, observed_at, source_hash = "", None, None
    path = raw_path(db_path)
    if path.is_file():
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
                conn.execute("PRAGMA query_only=ON")
                exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='stock_industry_metadata'").fetchone()
                if exists:
                    conn.row_factory = sqlite3.Row
                    records = [dict(row) for row in conn.execute("SELECT * FROM stock_industry_metadata "
                        "WHERE classification_eligible=1")]
                    for row in records:
                        code, name = str(row.get("industry_l1_code") or ""), str(row.get("industry_l1_name") or "")
                        if row["stock"] in wanted and code and name:
                            mapping[row["stock"]] = {"id": str(row.get("industry_taxonomy") or "CSRC") + ":" + code,
                                "name": name, "kind": "industry"}
                    if mapping:
                        source = "raw_sqlite:stock_industry_metadata"
                        observed_at = max((str(row.get("industry_source_fetched_at") or "") for row in records), default="")
                        source_hash = stable_hash(records)
        except (sqlite3.Error, KeyError) as exc:
            diagnostics.append("本地行业元数据读取失败：" + str(exc))
    snapshot = Path(snapshot_path or METADATA_ROOT / "market_sector_snapshot.json")
    if not mapping and snapshot.is_file():
        try:
            value = json.loads(snapshot.read_text(encoding="utf-8"))
            if value.get("kind") != "industry" or not value.get("source") or not _instant(value.get("observed_at")):
                raise ValueError("行业快照缺少可核验来源、获取时间或行业类型")
            conflicts, board_names = set(), {}
            for board in value["boards"]:
                if not isinstance(board.get("id"), str) or not board["id"] or board["id"] == UNKNOWN or not board.get("name"):
                    raise ValueError("行业快照板块身份不完整")
                if board["id"] in board_names and board_names[board["id"]] != board["name"]:
                    raise ValueError("行业快照同一板块代码存在冲突名称")
                board_names[board["id"]] = board["name"]
                for symbol in board["members"]:
                    if not isinstance(symbol, str):
                        raise ValueError("行业快照成员代码必须为字符串")
                    if symbol not in wanted:
                        continue
                    item = {"id": board["id"], "name": str(board["name"]), "kind": "industry"}
                    if symbol in mapping and mapping[symbol] != item:
                        conflicts.add(symbol)
                    mapping[symbol] = item
            for symbol in conflicts:
                mapping.pop(symbol, None)
            if conflicts:
                diagnostics.append(f"{len(conflicts)}只股票存在多行业归属冲突，归入未知行业")
            source, observed_at, source_hash = value["source"], value["observed_at"], stable_hash(value)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            mapping = {}
            diagnostics.append("本地行业快照不可用：" + str(exc))
    for symbol in symbols:
        mapping.setdefault(symbol, {"id": UNKNOWN, "name": "未知行业", "kind": "industry"})
    known = sum(row["id"] != UNKNOWN for row in mapping.values())
    return mapping, {"mode": "current_snapshot" if known else "unavailable", "source": source or None,
        "observed_at": observed_at, "source_hash": source_hash, "note": CLASSIFICATION_NOTE if known else
        "缺少可核验行业归属，股票保留在未知行业及全市场分母中。", "classified_symbols": known,
        "unknown_symbols": len(symbols) - known, "diagnostics": diagnostics}


def window_dates(calendar, end, sessions=30):
    if list(calendar) != sorted(set(calendar)):
        raise ValueError("市场历史需要按日有序唯一的已核验交易日历")
    dates = [str(value)[:10] for value in calendar if str(value)[:10] <= str(end)[:10]][-sessions:]
    if not dates:
        raise ValueError("截止日之前没有已核验市场历史日期")
    return dates


def _visible(event, cutoff):
    clocks = [event.get(key) for key in ("available_at", "first_available_at", "observed_at",
        "first_confirmed_at", "origin_observed_at") if event.get(key) is not None]
    return bool(clocks and event.get("input_eligible", True) and event.get("evidence_visible", True)
        and all(_instant(value) is not None and _instant(value) <= cutoff for value in clocks))


def _event(event):
    return {"id": str(event.get("original_event_id") or event["event_id"]),
        "type": str(event.get("event_type") or event.get("event") or "unknown"),
        "side": str(event.get("side") or "UNKNOWN"), "confirmed": event.get("confirmed") is True,
        "available_at": str(event.get("first_available_at") or event.get("available_at"))}


def stock_facts(symbol, dates, *, overlay=None, failure=None, sector=None, name=""):
    """Missing market sessions remain facts, never forward-filled opinions."""
    sector = sector or {"id": UNKNOWN, "name": "未知行业"}
    overlay = overlay or {}
    rows = {_day(row["date"]): row for row in overlay.get("rows", [])}
    first_candidates, candidate_seen = {}, set()
    for day, row in sorted(rows.items()):
        identifiers = {str(event["event_id"]) for event in row.get("active_candidates", [])
            if event.get("side") == "BUY" and event.get("event_id")}
        first_candidates[day] = bool(identifiers - candidate_seen)
        candidate_seen.update(identifiers)
    account = overlay.get("account_actions") or {}
    account_rows = {_day(row["date"]): row for row in account.get("rows", [])}
    account_daily = {_day(row["date"]): row for row in account.get("daily", [])}
    holding_rows = {_day(row["date"]): row for row in account.get("holding_snapshots", [])}
    fills = defaultdict(set)
    for fill in account.get("ledger", []):
        fills[_day(fill["date"])].add(fill.get("action", fill.get("side")))
    pack = overlay.get("native_pack") or {}
    evaluations = {_day(row["date"]): row for row in pack.get("evaluations", [])}
    raw_events = pack.get("wyckoff_events", overlay.get("wyckoff_events", []))
    new_by_day = defaultdict(dict)
    # The first visible clock, rather than a pattern's retrospective anchor,
    # determines which session receives this event. Deduplicate its original ID.
    seen = set()
    for event in sorted(raw_events, key=lambda item: str(item.get("available_at") or "")):
        if not event.get("event_id"):
            continue
        item = _event(event)
        if item["id"] in seen:
            continue
        clocks = [_instant(event.get(key)) for key in ("available_at", "first_available_at", "observed_at",
            "first_confirmed_at", "origin_observed_at") if event.get(key) is not None]
        if not clocks or any(clock is None for clock in clocks):
            continue
        visible_at = max(clocks)
        first = next((day for day in dates if _instant(day + "T15:00:00+08:00") >= visible_at), None)
        # Events that predate this window are already stock, not newly observed.
        if visible_at < _instant(dates[0] + "T00:00:00+08:00"):
            seen.add(item["id"])
            continue
        if first and _visible(event, _instant(first + "T15:00:00+08:00")):
            new_by_day[first][item["id"]] = item
            seen.add(item["id"])
    facts = []
    for day in dates:
        row, account_row = rows.get(day), account_rows.get(day)
        available = row is not None and failure is None
        agents = (row or {}).get("agents") or {}
        quality = agents.get("high_return") or {}
        quality_status = str(quality.get("evaluation_status") or quality.get("status") or "model_unavailable")
        if quality_status == "scored":
            probability, threshold = _number(quality.get("probability")), _number(quality.get("threshold"))
            quality_status = ("scored_passed" if probability >= threshold else "scored_rejected") if probability is not None and threshold is not None else "model_unavailable"
        if not available:
            quality_status = "failed" if failure is not None else "missing_price"
        direction = str((row or {}).get("direction") or "WAIT") if available else "UNAVAILABLE"
        action = str((account_row or {}).get("action") or "WAIT")
        holding = account_daily.get(day) or holding_rows.get(day) or (row or {}).get("holding")
        shares = int(holding.get("shares") or 0) if holding is not None else None
        plan = (row or {}).get("entry_plan") or {}
        counts = {key: False for key in COUNT_KEYS}
        counts.update(direction_buy=available and direction == "BUY", direction_sell=available and direction == "SELL",
            candidates=available and any(event.get("side") == "BUY" for event in (row or {}).get("active_candidates", [])),
            new_candidates=available and first_candidates.get(day, False),
            risk_blocked=available and (quality_status == "risk_blocked" or plan.get("entry_allowed") is False),
            hr_scored_passed=available and quality_status == "scored_passed",
            hr_scored_rejected=available and quality_status == "scored_rejected",
            hr_unscored=quality_status not in {"scored_passed", "scored_rejected"},
            account_buy=account_row is not None and action == "BUY", account_sell=account_row is not None and action == "SELL",
            filled_buy="BUY" in fills[day], filled_sell="SELL" in fills[day], holding=shares is not None and shares > 0,
            failed=failure is not None, missing_price=not available and failure is None)
        wyckoff = agents.get("wyckoff") or {}
        support = dict.fromkeys(SUPPORT_KEYS)
        if available and wyckoff.get("status") == "scored":
            for target, source in ( ("continuous_demand", "current_demand"), ("continuous_supply", "current_supply"),
                ("event_demand", "event_demand"), ("event_supply", "event_supply"),
                ("effective_demand", "buy_strength"), ("effective_supply", "sell_strength")):
                support[target] = _number(wyckoff.get(source))
        w_eval = (evaluations.get(day) or {}).get("wyckoff") or {}
        cutoff = _instant(day + "T15:00:00+08:00")
        active = {}
        if available and wyckoff.get("status") == "scored":
            for event in w_eval.get("support_facts", []):
                if event.get("event_id") and event.get("confirmed") is True and (_number(event.get("event_strength")) or 0) > 0 and _visible(event, cutoff):
                    item = _event(event); active[item["id"]] = item
        new = list(new_by_day.get(day, {}).values()) if available else []
        resolution = (row or {}).get("model_resolution") or {}
        # The direction row and the validated account quality evaluation keep
        # different reasons. Preserve both without turning a quality veto into
        # an unexplained "no action" row or repeating the same saved code.
        reasons = list(dict.fromkeys([*((row or {}).get("reason_codes") or []),
            *(resolution.get("reason_codes") or [])]))
        if not available:
            reasons = ["scan_failed" if failure is not None else "missing_market_session_price"]
        facts.append({"symbol": symbol, "name": name, "date": day, "sector_id": sector["id"],
            "sector": sector["name"], "price_available": available,
            "input_available": available and any((agents.get(source) or {}).get("status") == "scored" for source in ("ma", "structure", "wyckoff")),
            "model_scored": available and (agents.get("ma") or {}).get("status") == "scored",
            "quality_status": quality_status, "high_return_probability": _number(quality.get("probability")) if available and quality_status in {"scored_passed", "scored_rejected"} else None,
            "direction": direction, "account_action": action if account_row else "UNAVAILABLE",
            "holding_shares": shares, "account_available": account_row is not None, "counts": counts, "support": support,
            "reason_codes": reasons, "failure": str((failure or {}).get("error") or "") if failure is not None else None,
            "new_events": new, "active_events": list(active.values()),
            "invalidated_events": len(w_eval.get("invalidations_at_day", [])) if available else 0,
            "model_resolution": resolution,
            "account_id": (account.get("context") or account.get("series_context") or {}).get("account_id")})
    return facts


def _aggregate():
    return {"coverage": Counter(), "counts": Counter(), "hr_status": Counter(), "events": Counter(),
        "new_by_type": Counter(), "active_by_type": Counter(), "sums": Counter(), "samples": Counter()}


def _add(target, fact):
    for key in ("price_available", "input_available", "model_scored", "account_available"):
        target["coverage"][key] += int(fact[key])
    for key in ("failed", "missing_price"):
        target["coverage"][key] += int(fact["counts"][key])
    target["counts"].update({key: int(fact["counts"][key]) for key in COUNT_KEYS})
    target["hr_status"][fact["quality_status"]] += 1
    for kind in ("new", "active"):
        items = fact[kind + "_events"]
        target["events"][kind + "_total"] += len(items)
        for event in items:
            target["events"][kind + "_" + event["side"].lower()] += 1
            target[kind + "_by_type"][event["type"]] += 1
    target["events"]["invalidated"] += fact["invalidated_events"]
    for key, value in fact["support"].items():
        if value is not None:
            target["sums"][key] += value; target["samples"][key] += 1


def _finish_day(date, value, universe_size):
    counts = {key: int(value["counts"][key]) for key in COUNT_KEYS}
    return {"date": date, "universe_size": universe_size,
        "coverage": {key: int(value["coverage"][key]) for key in ("price_available", "input_available", "model_scored", "account_available", "failed", "missing_price")},
        "counts": counts, "hr_status": dict(value["hr_status"]),
        "events": {**{key: int(value["events"][key]) for key in ("new_buy", "new_sell", "new_total", "active_buy", "active_sell", "active_total", "invalidated")},
            "new_by_type": dict(value["new_by_type"]), "active_by_type": dict(value["active_by_type"])},
        "mean_support": {key: value["sums"][key] / value["samples"][key] if value["samples"][key] else None for key in SUPPORT_KEYS},
        "support_sample_count": dict(value["samples"]),
        "buy_rate": counts["direction_buy"] / universe_size if universe_size else None,
        "sell_rate": counts["direction_sell"] / universe_size if universe_size else None}


class MarketHistoryBuilder:
    def __init__(self, symbols, calendar, end, *, run_id, classifications=None, classification=None,
                 names=None, window_sessions=30, contract=None):
        self.symbols = list(dict.fromkeys(symbols)); self.dates = window_dates(calendar, end, window_sessions)
        self.run_id, self.end, self.window_sessions = run_id, end, window_sessions
        self.classifications = classifications or {s: {"id": UNKNOWN, "name": "未知行业", "kind": "industry"} for s in self.symbols}
        self.classification = classification or {"mode": "unavailable", "source": None,
            "classified_symbols": 0, "unknown_symbols": len(self.symbols), "note": "行业归属不可用，全部计入未知行业。"}
        self.names, self.contract = names or {}, contract or {}
        self.daily = {date: _aggregate() for date in self.dates}
        self.sectors, self.members = {}, Counter()
        for symbol in self.symbols:
            sector = self.classifications[symbol]
            self.members[sector["id"]] += 1
            self.sectors.setdefault(sector["id"], {"id": sector["id"], "name": sector["name"],
                "kind": "industry", "daily": {date: _aggregate() for date in self.dates}})
        self.processed, self.types = set(), {}

    def add_stock(self, symbol, *, overlay=None, failure=None):
        if symbol not in self.symbols or symbol in self.processed:
            raise ValueError("市场历史股票必须属于固定股票池且不能重复")
        facts = stock_facts(symbol, self.dates, overlay=overlay, failure=failure,
            sector=self.classifications[symbol], name=self.names.get(symbol, ""))
        for fact in facts:
            _add(self.daily[fact["date"]], fact)
            _add(self.sectors[fact["sector_id"]]["daily"][fact["date"]], fact)
            for event in fact["new_events"] + fact["active_events"]:
                self.types.setdefault(event["type"], {"id": event["type"], "label": EVENT_LABELS.get(event["type"], event["type"]),
                    "side": event["side"], "new_total": 0, "active_stock_days": 0})
            for event in fact["new_events"]:
                self.types[event["type"]]["new_total"] += 1
            for kind in {event["type"] for event in fact["active_events"]}:
                self.types[kind]["active_stock_days"] += 1
        self.processed.add(symbol)
        return facts

    def finish(self):
        if set(self.symbols) != self.processed:
            raise ValueError("市场历史未覆盖固定股票池的全部成功及失败标的")
        return {"version": VERSION, "run_id": self.run_id,
            "strategy_version": self.contract.get("strategy_version") or STRATEGY_ID,
            "dates": self.dates, "window_sessions": self.window_sessions, "available_sessions": len(self.dates),
            "window_start": self.dates[0], "window_end": self.dates[-1], "requested_end": self.end,
            "universe_size": len(self.symbols), "classification": self.classification, "contract": self.contract,
            "daily": [_finish_day(date, self.daily[date], len(self.symbols)) for date in self.dates],
            "sectors": [{"id": sector["id"], "name": sector["name"], "kind": sector["kind"],
                "member_count": self.members[sector["id"]], "daily": [_finish_day(date, sector["daily"][date], self.members[sector["id"]]) for date in self.dates]}
                for sector in sorted(self.sectors.values(), key=lambda item: (item["id"] == UNKNOWN, item["name"]))],
            "unknown_sector_id": UNKNOWN, "event_types": sorted(self.types.values(), key=lambda item: item["id"]),
            "support_unit": "fraction_0_1", "publication_allowed": False,
            "notes": ["最近30个已核验交易日；每股一次连续回放，显示窗口不重置账户。",
                "分母为此次固定扫描股票池；缺行情、不可评分与失败均保留，不自动认定为停牌。",
                "固定股票池回溯统计不等于历史上市、退市及ST身份完整复原的历史市场股票池。",
                "方向诊断、最终账户意图和实际成交分别统计；未评估概率不是0或负例。",
                "新增事件按首次可见时间去重；有效存量仅计已确认且仍有有效强度的量价事件。",
                "历史判断由既有按日模型路由和冻结持仓政策生成，不改变评分及交易规则。",
                self.classification["note"], "均值仅使用可用量价输入，强度单位为0至1。"]}
