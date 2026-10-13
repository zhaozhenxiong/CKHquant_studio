"""Project causal references or an explicit saved account onto a chart."""
from __future__ import annotations

import copy

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, DISPLAY_NAME


def decorate_analysis(result, *, raw, overlay, start=None):
    from my_strategy.web_dashboard.trend_routes import TREND_FAMILIES, family_module
    family = overlay.get("strategy_version", STRATEGY_ID)
    if family not in TREND_FAMILIES:
        raise ValueError("图表账户版本未登记")
    display_name = family_module(family, "contracts").DISPLAY_NAME
    out = copy.deepcopy(result)
    reference = overlay.get("analysis_mode") == "reference"
    visible = {str(b.get("date") or b.get("time"))[:10] for b in result["frequencies"]["日线"]["bars"]}
    observed = {str(r["date"])[:10]: r for r in raw.to_dict("records")}
    account = copy.deepcopy(overlay.get("account_actions") or {})
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
        if not reference:
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
    all_points = overlay.get("reference_points", []) if reference else account.get("intents", [])
    visible_points = [e for e in all_points if str(e.get("date") or e.get("signal_date"))[:10] in visible]
    account["intents"] = visible_points
    account["rows"] = [r for r in account.get("rows", []) if str(r["date"])[:10] in visible]
    for key in ("ledger", "trades", "daily", "holding_snapshots", "rejections"):
        account[key] = [r for r in account.get(key, []) if str(r.get("date"))[:10] in visible]
    request_context_hash = stable_hash({"series_context_hash": overlay.get("source_context_hash") if reference else account["series_context_hash"],
        "as_of": result.get("decision_date", result["data_end"]),
        "display_start": start, "display_end": result["data_end"],
        "projection": "daily_weekly_monthly_original_clock"})
    if not reference:
        account["request_context_hash"] = request_context_hash
    direction["rows"] = [r for r in direction.get("rows", []) if str(r["date"])[:10] in visible]
    direction["events"] = [e for e in direction.get("events", direction.get("direction_points", [])) if str(e["date"])[:10] in visible]
    direction["direction_points"] = direction["events"]
    wfull = copy.deepcopy(overlay.get("wyckoff_events", []))
    wvisible = [e for e in wfull if str(e.get("first_available_at") or e.get("available_at"))[:10] in visible]
    for event in wvisible:
        day = str(event.get("first_available_at") or event.get("available_at"))[:10]
        event.update(reference_price=float(observed[day]["close"]), reference_price_date=day,
            reference_price_source="signal_day_close", reference_price_basis=basis)
    out.update(display_name=display_name, strategy_version=family, research_version=family,
        model_family="current", trend_consensus=direction, consensus_decisions=direction,
        account_actions=None if reference else account, wyckoff_events=wvisible, publication_allowed=False,
        model_gate=overlay.get("model_gate", {"passed": False}), compute_info=overlay.get("compute_info", {}),
        next_session_decision=copy.deepcopy(overlay["latest_decision"]))
    out["events"] = [{**e, "time": str(e.get("first_available_at") or e.get("available_at"))[:10],
        "action": str(e.get("side", "")).lower(), "signal": e.get("event_type", "来源候选"),
        "reason": "原始方向证据；B/S是独立方向参考，非成交或账户操作" if reference else "原始方向证据；账户B/S由趋势共识策略和持仓退出政策生成"}
        for row in rows for e in row.get("events", [])]
    out["chart_evidence"] = {"version": "czsc_trend_chart_evidence", "display_name": display_name,
        "analysis_mode": "reference" if reference else "backtest", "reference_only": reference,
        "rows": rows, "statistics": {"visible_window": {"bars": len(rows), "source_events": len(out["events"]),
            "wyckoff_events": len(wvisible), "final_points": len(visible_points),
            "direction_points": len(direction["events"]), "broker_fills": len(account.get("ledger", []))},
            "full_prefix": {"bars": len(overlay["rows"]), "source_events": sum(len(r.get("events", [])) for r in overlay["rows"]),
                "wyckoff_events": len(wfull), "final_points": len(all_points)}}}
    out["research"] = {**out.get("research", {}), "model_family": "current", "display_name": display_name,
        "strategy_version": family,
        "usage_mode": overlay.get("usage_mode", "historical"), "publication_allowed": False,
        "pricing_assumptions": copy.deepcopy(overlay.get("pricing_assumptions", {})),
        "high_return_target": copy.deepcopy(overlay.get("high_return_target", {}))}
    if reference:
        out.update(analysis_mode="reference", reference_only=True, account_scope="none",
            reference_points=copy.deepcopy(visible_points), reference_recommendation=copy.deepcopy(overlay["reference_recommendation"]),
            models_enabled=overlay.get("models_enabled", True), structure_only=overlay.get("structure_only", False))
        out["research"].update(analysis_mode="reference", account_scope="none", account_simulated=False,
            request_context_hash=request_context_hash, source_context_hash=overlay.get("source_context_hash"),
            models_enabled=overlay.get("models_enabled", True))
        out["research"].pop("high_return_target", None)
        out["chart_evidence"]["statistics"]["visible_window"]["reference_points"] = len(visible_points)
        out["chart_evidence"]["statistics"]["full_prefix"]["reference_points"] = len(all_points)
    else:
        out["research"]["account_context"] = copy.deepcopy(account["context"])
    return out
