"""Project causal B/S sources, final consensus and risk references onto a chart."""
from __future__ import annotations

import copy

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID


def decorate_analysis(result, *, raw, overlay, start=None):
    from my_strategy.web_dashboard.trend_routes import TREND_FAMILIES, family_module
    family = overlay.get("strategy_version", STRATEGY_ID)
    if family not in TREND_FAMILIES:
        raise ValueError("图表模型体系未登记")
    if overlay.get("analysis_mode") != "reference":
        raise ValueError("图表账户回放入口已退役；只提供B/S与仓位参考")
    display_name = overlay.get("display_name") or family_module(family, "contracts").DISPLAY_NAME
    position_schema = family == "reference_position_v1"
    def describe_event(event, action, *, source=False):
        if not position_schema:
            return
        event.update(reference_position_schema=family, target_change_action=action)
        if source:
            change = {"BUY": "增加", "SELL": "减少"}.get(action, "未变化")
            event["source_relationship_text"] = f"同日参考目标{change}；来源点独立保留"
        else:
            event["reference_display_label"] = {"BUY": "参考增仓", "SELL": "参考减仓"}.get(action, "参考目标保持")
    out = copy.deepcopy(result)
    visible = {str(b.get("date") or b.get("time"))[:10] for b in result["frequencies"]["日线"]["bars"]}
    observed = {str(r["date"])[:10]: r for r in raw.to_dict("records")}
    direction = copy.deepcopy(overlay["trend_consensus"])
    direction_rows = {str(r["date"])[:10]: r for r in direction.get("rows", [])}
    rows = []
    basis = result.get("price_basis", "unadjusted")
    for original in overlay["rows"]:
        day = str(original["date"])[:10]
        if day not in visible:
            continue
        row, bar = copy.deepcopy(original), observed[day]
        row.update(symbol=result["symbol"], reference_close=float(bar["close"]), reference_price_date=day,
            reference_price_source="signal_day_close", reference_price_basis=basis,
            ohlc={key: float(bar[key]) for key in ("open", "high", "low", "close")},
            consensus=direction_rows.get(day))
        risk = row.get("reference_risk_plan") or {}
        for event in row.get("source_reference_points", []):
            describe_event(event, row.get("action"), source=True)
        for event in row.get("events", []):
            event.update(evidence_date=day, reference_price=float(bar["close"]), reference_price_date=day,
                reference_price_source="signal_day_close", reference_price_basis=basis)
            eligible = event.get("direction_eligible")
            if eligible is None:
                eligible = (event.get("candidate_direction_valid") is True
                    and event.get("confirmed") is True and event.get("input_eligible") is True)
            event["direction_eligible"] = eligible is True
            describe_event(event, row.get("action"), source=True)
            event["reference_risk_eligible"] = (event["direction_eligible"] and event.get("side") == "BUY"
                and risk.get("status") == "available" and event.get("event_id") in risk.get("source_event_ids", []))
        for source in ("ma", "structure", "wyckoff"):
            agent = row.get("agents", {}).get(source, {})
            facts = [e for e in row.get("events", []) if e.get("source") == source]
            agent.update(events=facts, buy_count=sum(e.get("side") == "BUY" and e.get("direction_eligible") is True for e in facts),
                sell_count=sum(e.get("side") == "SELL" and e.get("direction_eligible") is True for e in facts),
                risk_buy_count=sum(e.get("reference_risk_eligible") is True for e in facts))
        rows.append(row)
    all_points = overlay.get("reference_points", [])
    all_source_points = overlay.get("source_reference_points", [])
    visible_points = [copy.deepcopy(e) for e in all_points if str(e.get("date") or e.get("signal_date"))[:10] in visible]
    visible_source_points = [copy.deepcopy(e) for e in all_source_points if str(e.get("date") or e.get("signal_date"))[:10] in visible]
    for point in visible_source_points + visible_points:
        day = str(point.get("date") or point.get("signal_date"))[:10]
        point.update(reference_price=float(observed[day]["close"]), reference_price_date=day,
            reference_price_source="signal_day_close", reference_price_basis=basis)
        describe_event(point, direction_rows.get(day, {}).get("action", "WAIT"),
            source=point.get("reference_layer") == "source")
    request_context_hash = stable_hash({"series_context_hash": overlay.get("source_context_hash"),
        "as_of": result.get("decision_date", result["data_end"]), "display_start": start,
        "display_end": result["data_end"], "projection": "daily_weekly_monthly_original_clock"})
    direction["rows"] = [r for r in direction.get("rows", []) if str(r["date"])[:10] in visible]
    direction["events"] = [e for e in direction.get("events", direction.get("direction_points", [])) if str(e["date"])[:10] in visible]
    direction["direction_points"] = direction["events"]
    if position_schema:
        for row in direction["rows"]:
            for key in ("events", "source_reference_points"):
                for event in row.get(key, []):
                    describe_event(event, row.get("action"), source=True)
        for event in direction["events"]:
            describe_event(event, event.get("action") or event.get("side"))
    wfull = copy.deepcopy(overlay.get("wyckoff_events", []))
    wvisible = [e for e in wfull if str(e.get("first_available_at") or e.get("available_at"))[:10] in visible]
    for event in wvisible:
        day = str(event.get("first_available_at") or event.get("available_at"))[:10]
        event.update(reference_price=float(observed[day]["close"]), reference_price_date=day,
            reference_price_source="signal_day_close", reference_price_basis=basis)
    out.update(display_name=display_name, strategy_version=family, research_version=family,
        model_family="current", trend_consensus=direction, consensus_decisions=direction,
        account_actions=None, wyckoff_events=wvisible, publication_allowed=False,
        model_gate=overlay.get("model_gate", {"passed": False}), compute_info=overlay.get("compute_info", {}),
        next_session_decision=copy.deepcopy(overlay["latest_decision"]),
        analysis_mode="reference", reference_only=True, account_scope="none",
        reference_points=visible_points, source_reference_points=visible_source_points,
        reference_risk_plan=copy.deepcopy(overlay.get("reference_risk_plan")),
        position_reference=copy.deepcopy(overlay.get("position_reference")),
        reference_recommendation=copy.deepcopy(overlay["reference_recommendation"]),
        models_enabled=overlay.get("models_enabled", True), structure_only=overlay.get("structure_only", False))
    out["events"] = [{**e, "time": str(e.get("first_available_at") or e.get("available_at"))[:10],
        "action": str(e.get("side", "")).lower(), "signal": e.get("event_type", "来源参考"),
        "reason": "原始规则来源；与逐股参考目标变化独立展示，参考价不是成交价。" if position_schema else
            "原始方向证据；B/S是独立方向参考，非成交或账户操作"}
        for row in rows for e in row.get("events", [])]
    out["chart_evidence"] = {"version": "czsc_trend_chart_evidence", "display_name": display_name,
        "analysis_mode": "reference", "reference_only": True, "rows": rows,
        "source_reference_points": copy.deepcopy(visible_source_points),
        "statistics": {"visible_window": {"bars": len(rows), "source_events": len(out["events"]),
            "wyckoff_events": len(wvisible), "final_points": len(visible_points),
            "direction_points": len(direction["events"]), "broker_fills": 0},
            "full_prefix": {"bars": len(overlay["rows"]), "source_events": sum(len(r.get("events", [])) for r in overlay["rows"]),
                "wyckoff_events": len(wfull), "final_points": len(all_points)}}}
    for scope, final, sources in (("visible_window", visible_points, visible_source_points),
                                  ("full_prefix", all_points, all_source_points)):
        out["chart_evidence"]["statistics"][scope].update(reference_points=len(final), source_reference_points=len(sources),
            source_reference_buy=sum(item["side"] == "BUY" for item in sources),
            source_reference_sell=sum(item["side"] == "SELL" for item in sources))
    out["research"] = {**out.get("research", {}), "model_family": "current", "display_name": display_name,
        "strategy_version": family, "usage_mode": overlay.get("usage_mode", "historical"),
        "publication_allowed": False, "pricing_assumptions": copy.deepcopy(overlay.get("pricing_assumptions", {})),
        "analysis_mode": "reference", "account_scope": "none", "account_simulated": False,
        "request_context_hash": request_context_hash, "source_context_hash": overlay.get("source_context_hash"),
        "models_enabled": overlay.get("models_enabled", True),
        "reference_position_schema": overlay.get("position_reference", {}).get("schema"),
        "historical_reconstruction": any(r.get("historical_reconstruction") for r in rows)}
    out["research"].pop("high_return_target", None)
    # Replay caches retain verified physical assets and the actual runtime.
    for key in ("model_asset_identity", "model_asset_identities", "reference_context_hash",
                "runtime_code_hash", "runtime_identity"):
        if key in overlay:
            out[key] = copy.deepcopy(overlay[key])
            out["research"][key] = copy.deepcopy(overlay[key])
    return out
