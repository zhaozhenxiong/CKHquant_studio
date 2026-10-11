"""Display-only dated prices and agent evidence from already computed inputs.

The adapter joins exact symbol/day observations and derives independent market
direction arbitration from saved evidence. It never runs models, changes the
Broker, or uses a later quote to fill an earlier date.
"""
from __future__ import annotations

import copy
import math

import pandas as pd


CHART_EVIDENCE_VERSION = "chart_evidence_v1"


def _day(value):
    if value is None or str(value) in {"", "None", "nan", "NaT"}:
        return None
    return str(value)[:10]


def _number(value, *, unit_interval=False):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and (not unit_interval or 0 <= number <= 1) else None


def _records(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    return list(value or [])


def _index(value, symbol):
    result = {}
    for row in _records(value):
        key = (str(row.get("symbol", symbol)), _day(row.get("date") or row.get("time")))
        if key[1] is None:
            continue
        if key in result:
            raise ValueError("图表展示证据的股票和日期重复")
        result[key] = row
    return result


def _event_day(event):
    # Confirmation can occur after a structural anchor.  The display decision
    # and its close reference belong to the first observable signal day.
    observed = [_day(event.get(key)) for key in ("signal_at", "available_at")]
    return max(filter(None, observed), default=None) or _day(event.get("date") or event.get("time"))


def _visible(instant, day):
    if not instant:
        return True
    try:
        value = pd.Timestamp(instant)
        if value.tzinfo is None:
            value = value.tz_localize("Asia/Shanghai")
        return value <= pd.Timestamp(day + "T15:00:00+08:00")
    except (ValueError, TypeError):
        return False


def _identity_visible(identity, day):
    # Historical reconstruction retains its own completion record.  Its
    # checkpoint availability must still precede the observed decision.
    return _visible(identity.get("available_at"), day)


def _agent(label, kind, status="score_unavailable", reason="当日没有保存的独立评分"):
    return {"label": label, "kind": kind, "score_kind": kind, "status": status,
            "probability": None, "buy_score": None, "sell_score": None,
            "exit_probability": None, "threshold": None, "target": None,
            "reason": reason, "model_identity": {}, "events": []}


def _side(event):
    value = str(event.get("side") or event.get("action") or "").upper()
    return "BUY" if value in {"BUY", "OPEN"} else "SELL" if value in {"SELL", "CLOSE"} else None


def _ma_agent(events, candidate, routes, day, parameters):
    result = _agent("均线 ML", "model_probability")
    result["events"] = events
    route = routes.get(day)
    if route is None:
        route = next((event.get("ma_entry_route") for event in events if event.get("ma_entry_route")), None)
    route = route or {}
    model_event = next((event for event in events if _side(event) == "BUY" and event.get("probability") is not None), {})
    identity = route.get("lineage") or model_event.get("model_lineage") or {}
    # Some existing adapters retain separate entry/exit head identities.  Only
    # the entry head can authorize this entry probability; checking an absent
    # top-level timestamp would accidentally accept a future nested head.
    if isinstance(identity.get("entry"), dict):
        identity = {**{key: value for key, value in identity.items() if key not in {"entry", "exit"}},
                    **identity["entry"]}
    probability = _number(route.get("probability", model_event.get("probability")), unit_interval=True)
    if probability is None and candidate.get("l1_ma_entry_available") is True:
        probability = _number(candidate.get("l1_ma_entry_probability"), unit_interval=True)
        identity = {"target_contract_hash": candidate.get("l1_ma_entry_target_contract_hash")}
    target = route.get("target") or model_event.get("probability_target")
    if target is None and candidate.get("l1_ma_entry_available") is True:
        target = "actual_complete_fee_net_roundtrip_positive"
    result.update(model_identity=copy.deepcopy(identity), target=target,
                  threshold=_number(parameters.get("ma_entry_threshold")),
                  buy_count=sum(_side(event) == "BUY" for event in events),
                  sell_count=sum(_side(event) == "SELL" for event in events),
                  exit_status="actual_holding_required", exit_reason="缺少兼容的真实 Broker 持仓历史，未评估退出概率")
    blocking_reasons = [code for code in [*(route.get("reason_codes") or []),
                                         *(model_event.get("reason_codes") or [])]
                        if code.startswith("ma_") and code != "ma_entry_probability_veto"]
    if not _identity_visible(identity, day):
        result.update(status="score_unavailable", reason="ma_model_not_available_at_signal")
    elif blocking_reasons:
        result.update(status="score_unavailable", reason="；".join(dict.fromkeys(blocking_reasons)))
    elif probability is not None:
        result.update(status="scored", probability=probability, reason="；".join(route.get("reason_codes") or []) or "当日已保存的前向均线入场评分")
    elif route.get("mode") == "causal_rule_v0":
        result.update(kind="rule_evidence", score_kind="rule_evidence", status="rule_fallback", reason=route.get("source_reason") or "当日使用均线规则回退，模型概率不可用")
    elif events:
        result.update(status="rule_triggered", reason="当日均线规则证据；独立模型概率不可用")
    return result


def _structure_agent(events):
    result = _agent("CZSC 结构", "rule_evidence", "no_signal", "当日未触发独立结构买卖点")
    result.update(events=events, buy_count=sum(_side(event) == "BUY" for event in events),
                  sell_count=sum(_side(event) == "SELL" for event in events),
                  point_types=list(dict.fromkeys(event.get("event_type") or event.get("signal") for event in events)))
    if events:
        result.update(status="rule_triggered", reason="已确认结构买卖点；该规则未输出连续概率")
    return result


def _wyckoff_agent(events, feature, day):
    result = _agent("威科夫量价", "rule_evidence")
    result.update(events=events, buy_score=_number(feature.get("w_demand_score"), unit_interval=True),
                  sell_score=_number(feature.get("w_supply_score"), unit_interval=True),
                  evidence_type=feature.get("wyckoff_event"), evidence_state=feature.get("wyckoff_state"),
                  target="因果需求／供给证据强度，非获利概率")
    metadata_visible = all(_visible(feature.get(key), day) for key in
                           ("wyckoff_available_at", "wyckoff_observed_at", "wyckoff_range_formed_at"))
    if not metadata_visible:
        result.update(buy_score=None, sell_score=None, status="input_unavailable",
                      reason="量价事件在该时点尚不可见，规则分数不可用")
    elif result["buy_score"] is not None or result["sell_score"] is not None:
        result.update(status="scored" if feature.get("wyckoff_input_eligible") else "input_unavailable",
                      reason="事件权重与事件龄衰减形成的规则证据强度" if feature.get("wyckoff_input_eligible") else "量价输入未通过核验，分数仅为已计算证据")
    elif events:
        result.update(status="rule_triggered", reason="当日量价规则已触发，未保存连续证据强度")
    return result


def _high_return_agent(row, day, candidate):
    result = _agent("高收益区间 ML", "model_probability")
    identity = row.get("model_resolution") or {}
    probability = _number(row.get("high_return_probability", row.get("entry_probability", identity.get("probability"))), unit_interval=True)
    # A union entry model scores only factual eligible candidates, never all
    # bars.  In particular, do not carry the latest probability through time.
    has_candidate = bool(candidate or row.get("candidate_sources") or row.get("source_candidate_eligible"))
    result.update(model_identity=copy.deepcopy(identity), target=row.get("probability_target") or identity.get("probability_target"),
                  threshold=_number(identity.get("threshold", identity.get("probability_threshold"))),
                  expected_net_return=_number(row.get("expected_net_return")),
                  loss_probability=_number(row.get("loss_probability"), unit_interval=True),
                  exit_status="actual_holding_required", exit_reason="缺少兼容的真实 Broker 持仓历史，未评估退出概率")
    reason = "；".join(identity.get("reason_codes") or [])
    if not has_candidate:
        result.update(status="no_candidate", reason="no_candidate_signal", expected_net_return=None, loss_probability=None)
    elif not _identity_visible(identity, day):
        result.update(status="score_unavailable", reason="model_not_available_at_signal", expected_net_return=None, loss_probability=None)
    elif probability is not None:
        result.update(status="scored", probability=probability, reason=reason or "当日候选的高收益区间影子评分")
    else:
        result.update(status="score_unavailable", reason=reason or "当日候选的兼容高收益评分不可用")
    return result


def decorate_analysis(result, *, raw=None, features=None, overlay=None, consensus_config=None, consensus_version=None):
    """Append display evidence and arbitrate the full causal history before clipping."""
    output = copy.deepcopy(result)
    symbol = str(result.get("symbol", ""))
    cutoff = min(filter(None, (_day(result.get("as_of")), _day(result.get("data_end")))), default=None)
    display = _records(result.get("frequencies", {}).get("日线", {}).get("bars", []))
    observations = _index(raw if raw is not None else display, symbol)
    feature_index = _index(features, symbol)
    overlay = overlay or {}
    decisions = _index(overlay.get("rows", []), symbol)
    candidates = _index(overlay.get("candidates", []), symbol)
    routes = {_day(row.get("date")): row for row in _records(overlay.get("ma_routes", []))}
    basis = result.get("price_basis") or getattr(raw, "attrs", {}).get("price_basis") or "unavailable"
    def dated_events(values):
        events = []
        for original in _records(values):
            event = copy.deepcopy(original)
            day = _event_day(event)
            event_symbol = str(event.get("symbol", symbol))
            if not day or cutoff and day > cutoff or not _visible(event.get("available_at"), cutoff or day):
                continue
            observation = observations.get((event_symbol, day), {})
            close = _number(observation.get("close"))
            event.update(evidence_date=day, original_event_date=_day(original.get("date") or original.get("time")),
                         reference_price=close, reference_price_date=day, reference_price_source="signal_day_close",
                         reference_price_basis=basis, reference_price_reason=None if close is not None else "signal_day_bar_unavailable")
            events.append(event)
        return events

    # The original displayed event list remains an inspectable candidate list.
    # Arbitration uses the saved full signal-point prefix, including warm-up.
    events = dated_events(result.get("events", []))
    output["events"] = events
    full_events = dated_events(overlay.get("signal_points", result.get("signal_points", result.get("events", []))))
    full_events_by_day, events_by_day = {}, {}
    for event in full_events:
        full_events_by_day.setdefault((str(event.get("symbol", symbol)), event["evidence_date"]), []).append(event)
    for event in events:
        events_by_day.setdefault((str(event.get("symbol", symbol)), event["evidence_date"]), []).append(event)

    def evidence_row(day, dated_events):
        key = (symbol, day)
        observation, feature, decision, candidate = (index.get(key, {}) for index in (observations, feature_index, decisions, candidates))
        visible = [event for event in dated_events if event.get("evidence_visible") is not False
                   and event.get("candidate_eligible") is not False]
        by_source = {source: [event for event in visible if event.get("source") == source] for source in ("ma", "structure", "wyckoff")}
        risk_plan = decision.get("raw_risk_plan") or decision.get("entry_plan")
        parameters = (risk_plan or {}).get("parameters", {})
        following = decision.get("next_market_session")
        if following is None:
            latest = result.get("next_session_decision") or {}
            if _day(latest.get("reference_price_date") or latest.get("signal_date")) == day:
                following = latest.get("next_market_session")
        agents = {"ma": _ma_agent(by_source["ma"], candidate, routes, day, parameters),
                  "structure": _structure_agent(by_source["structure"]),
                  "wyckoff": _wyckoff_agent(by_source["wyckoff"], feature, day),
                  "high_return": _high_return_agent(decision, day, candidate)}
        for source in ("ma", "structure", "wyckoff"):
            agents[source]["excluded_events"] = [event for event in dated_events
                if event.get("source") == source and (event.get("evidence_visible") is False
                                                      or event.get("candidate_eligible") is False)]
        return {"date": day, "symbol": symbol, "reference_close": _number(observation.get("close")),
            "reference_price_date": day, "reference_price_source": "signal_day_close", "reference_price_basis": basis,
            "reference_price_reason": None if _number(observation.get("close")) is not None else "signal_day_bar_unavailable",
            "ohlc": {column: _number(observation.get(column)) for column in ("open", "high", "low", "close")},
            "next_market_session": following, "raw_risk_plan": copy.deepcopy(risk_plan), "events": dated_events,
            "agents": agents}

    display_days = [_day(bar.get("time") or bar.get("date")) for bar in display]
    display_days = [day for day in display_days if day and (not cutoff or day <= cutoff)]
    rows = [evidence_row(day, events_by_day.get((symbol, day), [])) for day in display_days]
    output["chart_evidence"] = {"version": CHART_EVIDENCE_VERSION, "price_basis": basis, "rows": rows,
        "execution_basis": "信号日收盘参考；实际成交价、执行日与费用以 Broker 账本为准"}

    from my_strategy.services.czsc_agent_consensus import build_consensus
    prefix_days = sorted({day for (stock, day) in observations if stock == symbol and (not cutoff or day <= cutoff)}
                         | set(display_days))
    prefix_rows = [evidence_row(day, full_events_by_day.get((symbol, day), [])) for day in prefix_days]
    from my_strategy.services.czsc_agent_consensus import consensus_config as resolve_consensus_config
    from my_strategy.services.czsc_candidate_evidence import V3_VERSION, build_candidate_evidence
    effective_config = resolve_consensus_config(consensus_config, version=consensus_version)
    if effective_config.get("version") == V3_VERSION:
        # v1/v2 adapters stay byte-compatible. New daily assessments are an
        # explicit additional contract, independent of original entry risk.
        candidate_evidence = build_candidate_evidence(prefix_rows, features,
            policy=overlay.get("candidate_evidence_policy"), config=effective_config)
        for row in prefix_rows:
            row["candidate_evidence"] = candidate_evidence[row["date"]]
        for row in rows:
            row["candidate_evidence"] = copy.deepcopy(candidate_evidence[row["date"]])
        output["chart_evidence"]["candidate_assessment_version"] = "czsc_candidate_evidence_v1"
    consensus = build_consensus(prefix_rows, symbol=symbol,
                                model_family=result.get("model_family", "ma_trend"), config=consensus_config,
                                version=consensus_version)
    visible_days = set(display_days)
    hidden_rows = [row for row in consensus["rows"] if display_days and row["date"] < min(display_days)]
    consensus.update(state_before_display=copy.deepcopy(hidden_rows[-1]["state_after"] if hidden_rows else consensus["initial_state"]),
                     prefix_start=prefix_days[0] if prefix_days else None, prefix_bars=len(prefix_days),
                     rows=[row for row in consensus["rows"] if row["date"] in visible_days],
                     events=[event for event in consensus["events"] if _day(event.get("date") or event.get("time")) in visible_days])
    output["consensus_decisions"] = consensus
    return output
