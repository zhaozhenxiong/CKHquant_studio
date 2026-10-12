"""Project one saved account and causal evidence onto the requested chart view."""
from __future__ import annotations

import copy

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, DISPLAY_NAME


def decorate_analysis(result, *, raw, overlay, start=None):
    out = copy.deepcopy(result)
    visible = {str(b.get("date") or b.get("time"))[:10] for b in result["frequencies"]["日线"]["bars"]}
    observed = {str(r["date"])[:10]: r for r in raw.to_dict("records")}
    account = copy.deepcopy(overlay["account_actions"])
    account_rows = {str(r["date"])[:10]: r for r in account.get("rows", [])}
    direction = copy.deepcopy(overlay["trend_consensus"])
    direction_rows = {str(r["date"])[:10]: r for r in direction.get("rows", [])}
    rows = []
    basis = result.get("price_basis", "unadjusted")
    for original in overlay["rows"]:
        day = str(original["date"])[:10]
        if day not in visible:
            continue
        row = copy.deepcopy(original)
        bar = observed[day]
        row.update(symbol=result["symbol"], reference_close=float(bar["close"]), reference_price_date=day,
            reference_price_source="signal_day_close", reference_price_basis=basis,
            ohlc={key: float(bar[key]) for key in ("open", "high", "low", "close")})
        row["consensus"] = direction_rows.get(day)
        row["account_action"] = account_rows.get(day)
        if row["account_action"]:
            row["holding"] = copy.deepcopy(row["account_action"].get("holding_snapshot") or row["account_action"].get("holding_before"))
        for event in row.get("events", []):
            event.update(evidence_date=day, reference_price=float(bar["close"]), reference_price_date=day,
                reference_price_source="signal_day_close", reference_price_basis=basis)
            eligible = event.get("direction_eligible")
            if eligible is None:
                eligible = (event.get("candidate_direction_valid") is True
                    and event.get("confirmed") is True and event.get("input_eligible") is True)
            event["direction_eligible"] = eligible is True
            plan = row.get("entry_plan") or {}
            event["risk_candidate_eligible"] = (event["direction_eligible"] and event.get("side") == "BUY"
                and plan.get("entry_allowed") is True and plan.get("status") == "active"
                and event.get("event_id") in plan.get("source_event_ids", []))
        for source in ("ma", "structure", "wyckoff"):
            agent = row.get("agents", {}).get(source, {})
            facts = [e for e in row.get("events", []) if e.get("source") == source]
            agent.update(events=facts, buy_count=sum(e.get("side") == "BUY" and e.get("direction_eligible") is True for e in facts),
                sell_count=sum(e.get("side") == "SELL" and e.get("direction_eligible") is True for e in facts),
                risk_buy_count=sum(e.get("risk_candidate_eligible") is True for e in facts))
        rows.append(row)
    all_points = account.get("intents", [])
    visible_points = [e for e in all_points if str(e.get("date") or e.get("signal_date"))[:10] in visible]
    account["intents"] = visible_points
    account["rows"] = [r for r in account.get("rows", []) if str(r["date"])[:10] in visible]
    for key in ("ledger", "trades", "daily", "holding_snapshots", "rejections"):
        account[key] = [r for r in account.get(key, []) if str(r.get("date"))[:10] in visible]
    account["request_context_hash"] = stable_hash({"series_context_hash": account["series_context_hash"],
        "as_of": result.get("decision_date", result["data_end"]),
        "display_start": start, "display_end": result["data_end"],
        "projection": "daily_weekly_monthly_original_clock"})
    direction["rows"] = [r for r in direction.get("rows", []) if str(r["date"])[:10] in visible]
    direction["events"] = [e for e in direction.get("events", direction.get("direction_points", [])) if str(e["date"])[:10] in visible]
    direction["direction_points"] = direction["events"]
    wfull = copy.deepcopy(overlay.get("wyckoff_events", []))
    wvisible = [e for e in wfull if str(e.get("first_available_at") or e.get("available_at"))[:10] in visible]
    for event in wvisible:
        day = str(event.get("first_available_at") or event.get("available_at"))[:10]
        event.update(reference_price=float(observed[day]["close"]), reference_price_date=day,
            reference_price_source="signal_day_close", reference_price_basis=basis)
    out.update(display_name=DISPLAY_NAME, strategy_version=STRATEGY_ID, research_version=STRATEGY_ID,
        model_family="high_return", trend_consensus=direction, consensus_decisions=direction,
        account_actions=account, wyckoff_events=wvisible, publication_allowed=False,
        model_gate=overlay.get("model_gate", {"passed": False}), compute_info=overlay.get("compute_info", {}),
        next_session_decision=copy.deepcopy(overlay["latest_decision"]))
    out["events"] = [{**e, "time": str(e.get("first_available_at") or e.get("available_at"))[:10],
        "action": str(e.get("side", "")).lower(), "signal": e.get("event_type", "来源候选"),
        "reason": "原始方向证据；账户B/S由趋势共识策略和持仓退出政策生成"}
        for row in rows for e in row.get("events", [])]
    out["chart_evidence"] = {"version": "czsc_trend_chart_evidence", "display_name": DISPLAY_NAME,
        "rows": rows, "statistics": {"visible_window": {"bars": len(rows), "source_events": len(out["events"]),
            "wyckoff_events": len(wvisible), "final_points": len(visible_points),
            "direction_points": len(direction["events"]), "broker_fills": len(account.get("ledger", []))},
            "full_prefix": {"bars": len(overlay["rows"]), "source_events": sum(len(r.get("events", [])) for r in overlay["rows"]),
                "wyckoff_events": len(wfull), "final_points": len(all_points)}}}
    out["research"] = {**out.get("research", {}), "model_family": "high_return", "display_name": DISPLAY_NAME,
        "strategy_version": STRATEGY_ID, "account_context": copy.deepcopy(account["context"]),
        "usage_mode": overlay.get("usage_mode", "historical"), "publication_allowed": False,
        "pricing_assumptions": copy.deepcopy(overlay.get("pricing_assumptions", {})),
        "high_return_target": copy.deepcopy(overlay.get("high_return_target", {}))}
    return out
