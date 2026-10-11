"""Causal multi-source arbitration for market-direction research chart points.

This module reads saved evidence only. It owns no account, model inference or
execution policy. A displayed S is a market-direction research point; it does
not establish a holding or an order. The saved L2 score is an entry quality gate
under its original frozen target, never an additional independent L1 vote.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
from collections.abc import Mapping

from my_strategy.core.config_loader import load_config


CONSENSUS_VERSION = "czsc_agent_consensus_v1"
SEMANTICS = "market_direction_research"
SOURCES = ("ma", "structure", "wyckoff")
_ZONE = dt.timezone(dt.timedelta(hours=8))
_DEFAULTS = {
    "source_weights": {"ma": 1., "structure": 1., "wyckoff": 1.},
    "min_buy_support": 2., "min_sell_support": 2., "min_direction_margin": 1.,
    "ma_entry_threshold": .55, "wyckoff_min_strength": .5,
    "high_return_threshold": .1, "require_high_return_when_family": True,
    "signal_valid_bars": 20, "cooldown_bars": 5,
}


def _finite(value, *, unit=False):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and (not unit or 0 <= number <= 1) else None


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def consensus_config(config=None):
    """Validate legacy overrides against frozen v1 defaults for reproducibility."""
    stored = copy.deepcopy(_DEFAULTS)
    if not isinstance(stored, Mapping) or set(stored) != set(_DEFAULTS):
        raise ValueError("invalid consensus configuration file contract")
    if not isinstance(stored["source_weights"], Mapping) or set(stored["source_weights"]) != set(SOURCES):
        raise ValueError("invalid consensus configuration file source weights")
    if config is not None and (not isinstance(config, Mapping) or set(config) - set(_DEFAULTS)):
        raise ValueError("unknown consensus configuration keys")
    values = {**copy.deepcopy(stored), **copy.deepcopy(dict(config or {}))}
    weights = values["source_weights"]
    if not isinstance(weights, Mapping) or set(weights) - set(SOURCES):
        raise ValueError("invalid consensus source weights")
    weights = {**stored["source_weights"], **weights}
    for source in SOURCES:
        weight = _finite(weights[source])
        if weight is None or weight < 0:
            raise ValueError("invalid consensus source weight: " + source)
        weights[source] = weight
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("consensus requires a positive source weight")
    values["source_weights"] = weights
    for key in ("min_buy_support", "min_sell_support", "min_direction_margin"):
        number = _finite(values[key])
        if number is None or not 0 < number <= total:
            raise ValueError("invalid or impossible consensus support: " + key)
        values[key] = number
    for key in ("ma_entry_threshold", "wyckoff_min_strength", "high_return_threshold"):
        number = _finite(values[key], unit=True)
        if number is None:
            raise ValueError("invalid consensus threshold: " + key)
        values[key] = number
    for key, minimum in (("signal_valid_bars", 1), ("cooldown_bars", 0)):
        number = _finite(values[key])
        if number is None or number != int(number) or number < minimum:
            raise ValueError("invalid consensus bar count: " + key)
        values[key] = int(number)
    if not isinstance(values["require_high_return_when_family"], bool):
        raise ValueError("consensus high-return requirement must be boolean")
    return values


def _date(value):
    try:
        text = str(value)
        day = dt.date.fromisoformat(text[:10])
        return day.isoformat() if text[:10] == day.isoformat() else None
    except (TypeError, ValueError):
        return None


def _instant(value):
    try:
        instant = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return instant.replace(tzinfo=_ZONE) if instant.tzinfo is None else instant
    except (TypeError, ValueError):
        return None


def _observable(value, close_at):
    instant = _instant(value)
    return instant is not None and instant <= close_at


def _codes(value):
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _side(event):
    side = str(event.get("side") or event.get("action") or "").upper()
    return "BUY" if side in {"BUY", "OPEN"} else "SELL" if side in {"SELL", "CLOSE"} else None


def _identity(agent):
    identity = agent.get("model_identity") or {}
    if not isinstance(identity, Mapping):
        return {}
    if isinstance(identity.get("entry"), Mapping):
        identity = {**{key: value for key, value in identity.items() if key not in {"entry", "exit"}},
                    **identity["entry"]}
    return identity


def _score_reasons(agent, day, close_at, *, require_lineage=False, expected_heads=None,
                   require_target=False):
    """Recheck supplied saved-score metadata without creating a new lineage."""
    reasons = []
    identity = _identity(agent)
    reconstruction = identity.get("historical_reconstruction") is True
    if require_lineage and not identity.get("available_at"):
        reasons.append("score_lineage_availability_unavailable")
    if reconstruction and not (identity.get("train_label_end") or identity.get("label_cutoff")):
        # An asserted reconstruction flag alone cannot excuse a checkpoint
        # trained later. Its saved target dependency must predate the signal.
        reasons.append("score_identity_incomplete")
    if expected_heads is not None:
        for key in ("head", "model_head"):
            if identity.get(key) is not None and identity[key] not in expected_heads:
                reasons.append("score_head_mismatch")
    target = agent.get("target") or identity.get("target") or identity.get("probability_target")
    if require_target and not target:
        reasons.append("score_target_unavailable")
    if target and any(token in str(target).lower() for token in ("fixed_10", "fixed_horizon", "fixed10", "固定10", "固定十")):
        reasons.append("fixed_horizon_score_target_rejected")
    for mapping in (agent, identity):
        for key in ("available_at", "score_available_at", "trained_at"):
            if mapping.get(key) is not None and not _observable(mapping[key], close_at):
                reasons.append("score_not_observable")
        if not reconstruction:
            for key in ("actual_training_completed_at", "actual_completion_record", "training_completed_at"):
                if mapping.get(key) is not None and not _observable(mapping[key], close_at):
                    reasons.append("score_not_observable")
        if mapping.get("retrospective") or mapping.get("retrospective_inference"):
            reasons.append("retrospective_score_rejected")
    for key in ("score_date", "date"):
        if agent.get(key) is not None and _date(agent[key]) != day:
            reasons.append("score_date_mismatch")
    for key in ("label_cutoff", "train_label_end", "training_cutoff", "train_end",
                "validation_label_end", "calibration_label_end"):
        if identity.get(key) is not None and (_date(identity[key]) is None or _date(identity[key]) >= day):
            reasons.append("score_label_overlap")
    for code in [*_codes(agent.get("reason_codes")), *_codes(identity.get("reason_codes"))]:
        if any(token in code for token in ("mismatch", "not_available", "unavailable", "retrospective", "label_overlap")):
            reasons.append(code)
    return list(dict.fromkeys(reasons))


def _base_agent(source, evidence):
    return {"judgment": "ABSTAIN", "reason_codes": [], "type": evidence.get("kind") or "rule_evidence",
            "status": "no_signal", "source_status": evidence.get("status", "unavailable"),
            "probability": None, "strength": None, "exit_probability": None,
            "model_identity": copy.deepcopy(evidence.get("model_identity") or {}),
            "target": evidence.get("target"), "source": source,
            "historical_reconstruction": _identity(evidence).get("historical_reconstruction") is True,
            "event_ids": [], "opportunity_ids": [], "supported_sides": []}


def _event_reasons(event, *, symbol, day, close_at):
    reasons = []
    if str(event.get("symbol", "")) != symbol:
        reasons.append("foreign_or_missing_event_symbol")
    if event.get("candidate_eligible") is not True or event.get("evidence_visible") is not True:
        reasons.append("event_not_eligible_or_visible")
    if event.get("input_eligible") is False:
        reasons.append("event_input_unavailable")
    if not event.get("event_id"):
        reasons.append("event_identity_unavailable")
    evidence_day = _date(event.get("evidence_date"))
    if evidence_day is None:
        timestamps = [_date(event.get(key)) for key in ("signal_at", "available_at")]
        evidence_day = max(filter(None, timestamps), default=None) or _date(event.get("date") or event.get("time"))
    if evidence_day != day:
        reasons.append("event_not_observed_today")
    if not event.get("available_at") or not _observable(event["available_at"], close_at):
        reasons.append("event_not_observable")
    for key in ("signal_at", "observed_at", "confirmed_at", "structure_confirmed_at", "point_confirmed_at"):
        if event.get(key) is not None and not _observable(event[key], close_at):
            reasons.append("event_not_observable")
    return list(dict.fromkeys(reasons))


def _opportunity(event):
    return str(event.get("opportunity_id") or "event:" + str(event["event_id"]))


def _source_agent(source, evidence, events, *, day, close_at, config):
    agent = _base_agent(source, evidence)
    agent["probability"] = _finite(evidence.get("probability"), unit=True)
    agent["strength"] = {"buy": _finite(evidence.get("buy_score"), unit=True),
                         "sell": _finite(evidence.get("sell_score"), unit=True)} if source == "wyckoff" else None
    if not events:
        agent["reason_codes"] = ["no_fresh_eligible_candidate"]
        return agent
    buy, sell = [], []
    score_reasons = _score_reasons(evidence, day, close_at,
                                 require_lineage=source == "ma" and evidence.get("status") == "scored",
                                 expected_heads={"entry", "ma_entry"} if source == "ma" else None,
                                 require_target=source == "ma" and evidence.get("status") == "scored")
    for event in events:
        side = _side(event)
        if side is None:
            agent["reason_codes"].append("unknown_event_direction")
            continue
        kind = str(event.get("event_type") or event.get("signal") or "")
        if source == "ma":
            if side == "SELL":
                if kind != "ma_trend_failure":
                    agent["reason_codes"].append("actual_holding_required_for_independent_exit")
                    continue
            elif evidence.get("status") == "rule_fallback" and not score_reasons:
                agent["type"] = "rule_evidence"
                agent["reason_codes"].append("explicit_causal_rule_fallback")
            else:
                threshold = config["ma_entry_threshold"]
                agent["threshold"] = threshold
                agent["model_threshold"] = _finite(evidence.get("threshold"), unit=True)
                event_identity = event.get("model_lineage")
                event_score_reasons = _score_reasons({"model_identity": event_identity}, day, close_at,
                                                     expected_heads={"entry", "ma_entry"}) if event_identity else []
                if evidence.get("status") != "scored" or agent["probability"] is None or score_reasons or event_score_reasons:
                    agent["reason_codes"].extend(score_reasons or event_score_reasons or ["ma_entry_score_unavailable"])
                    continue
                if agent["probability"] < threshold:
                    agent["reason_codes"].append("ma_entry_below_threshold")
                    continue
        elif source == "wyckoff":
            if evidence.get("status") in {"input_unavailable", "score_unavailable", "unavailable"} or score_reasons:
                agent["reason_codes"].extend(score_reasons or ["wyckoff_input_unavailable"])
                continue
            strength = agent["strength"][side.lower()]
            if strength is None:
                if evidence.get("status") != "rule_triggered":
                    agent["reason_codes"].append("wyckoff_strength_unavailable")
                    continue
                agent["reason_codes"].append("explicit_wyckoff_rule_support")
            elif strength < config["wyckoff_min_strength"]:
                agent["reason_codes"].append("wyckoff_strength_below_threshold")
                continue
            agent["threshold"] = config["wyckoff_min_strength"]
        elif evidence.get("status") in {"input_unavailable", "score_unavailable", "unavailable"}:
            agent["reason_codes"].append("structure_input_unavailable")
            continue
        (buy if side == "BUY" else sell).append(event)
    supported = buy + sell
    agent["event_ids"] = sorted({str(event["event_id"]) for event in supported})
    agent["opportunity_ids"] = sorted({_opportunity(event) for event in supported})
    agent["supported_sides"] = (["BUY"] if buy else []) + (["SELL"] if sell else [])
    if buy and sell:
        agent.update(judgment="CONFLICT", status="conflict")
        agent["reason_codes"].append("opposing_events_within_source")
    elif supported:
        agent.update(judgment="BUY" if buy else "SELL", status="supported")
        agent["reason_codes"].append("eligible_causal_source_support")
    else:
        agent["status"] = "unavailable" if any("unavailable" in code or "observable" in code or "holding_required" in code
                                             for code in agent["reason_codes"]) else "not_supported"
    agent["reason_codes"] = list(dict.fromkeys(agent["reason_codes"]))
    return agent


def _gate_agent(evidence, *, day, close_at, config, required, has_buy_candidate):
    agent = _base_agent("high_return", evidence)
    agent["type"] = "model_quality_gate"
    agent["probability"] = _finite(evidence.get("probability"), unit=True)
    agent["threshold"] = config["high_return_threshold"]
    agent["model_threshold"] = _finite(evidence.get("threshold"), unit=True)
    agent["counts_as_vote"] = False
    agent["required"] = required
    agent["threshold_scope"] = "consensus_research_quality_filter"
    reasons = _score_reasons(evidence, day, close_at, require_lineage=True,
                             expected_heads={"entry"}, require_target=True)
    if not has_buy_candidate:
        agent.update(status="no_candidate", reason_codes=["no_buy_candidate"])
    elif evidence.get("status") != "scored" or agent["probability"] is None or reasons:
        agent.update(status="unavailable", reason_codes=reasons or ["high_return_score_unavailable"])
    elif agent["probability"] < agent["threshold"]:
        agent.update(judgment="WAIT", status="rejected", reason_codes=["high_return_below_consensus_threshold"])
    else:
        agent.update(judgment="BUY", status="passed", reason_codes=["high_return_quality_gate_passed"])
    return agent


def _supports(agents, by_source, config):
    families = {"BUY": {}, "SELL": {}}
    reasons = []
    for source in SOURCES:
        side = agents[source]["judgment"]
        if side not in families:
            continue
        family = source
        supporting = [event for event in by_source[source] if str(event["event_id"]) in agents[source]["event_ids"]]
        if source == "structure" and supporting and all(
                "native_ma_assisted_definition" in (event.get("dependencies") or [])
                or str(event.get("event_type", "")).startswith(("native_bs2_", "native_bs3_"))
                for event in supporting):
            family = "ma"
            reasons.append("ma_assisted_structure_dependency_merged")
        families[side][family] = max(families[side].get(family, 0.), config["source_weights"][source])
    # Zero-weight sources are recorded as opinions, but not independent votes.
    families = {side: {name: weight for name, weight in values.items() if weight > 0}
                for side, values in families.items()}
    return families, reasons


def _risk_reasons(plan, *, symbol, day, close_at, required):
    if plan is None:
        return ["entry_risk_plan_unavailable"] if required else []
    if not isinstance(plan, Mapping):
        return ["entry_risk_plan_invalid"]
    reasons = []
    if plan.get("symbol") is not None and str(plan["symbol"]) != symbol:
        reasons.append("entry_risk_plan_symbol_mismatch")
    for key in ("available_at", "created_at"):
        if plan.get(key) is not None and not _observable(plan[key], close_at):
            reasons.append("entry_risk_plan_not_observable")
    if plan.get("signal_date") is not None and _date(plan["signal_date"]) != day:
        reasons.append("entry_risk_plan_date_mismatch")
    if plan.get("entry_allowed") is not True:
        reasons.extend(["entry_risk_gate_rejected", *_codes(plan.get("reason_codes"))])
    return list(dict.fromkeys(reasons))


def build_consensus(rows, *, symbol, model_family="high_return", config=None):
    """Arbitrate a full chronological evidence prefix before display cropping.

    Each L1 source has at most one opinion per day. A consumed opportunity can
    never produce another point. Expiration merely resets direction; it never
    creates a point from aged evidence. Missing/invalid prices remain WAIT.
    """
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("consensus requires a symbol")
    values = consensus_config(config)
    config_hash = _hash(values)
    if isinstance(rows, Mapping) or isinstance(rows, (str, bytes)):
        raise ValueError("consensus rows must be a chronological sequence")
    rows = list(rows)
    previous = None
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("symbol", "")) != symbol:
            raise ValueError("consensus rows require the exact symbol")
        day = _date(row.get("date"))
        if day is None or str(row.get("date")) != day or previous is not None and day <= previous:
            raise ValueError("consensus rows require unique ascending ISO dates")
        previous = day
    state = "neutral"
    last_direction_index = None
    last_emit = {"BUY": None, "SELL": None}
    consumed = set()
    seen_events = set()
    decisions, final_events = [], []
    for index, row in enumerate(rows):
        day = row["date"]
        close_at = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(15), tzinfo=_ZONE)
        available_at = close_at.isoformat()
        state_before = state
        reasons = []
        price = _finite(row.get("reference_close", row.get("reference_price")))
        price_date = _date(row.get("reference_price_date"))
        exact_price = price is not None and price > 0 and price_date == day
        if not exact_price:
            reasons.append("exact_signal_day_reference_price_unavailable")
        if exact_price and last_direction_index is not None and index - last_direction_index >= values["signal_valid_bars"]:
            state = "neutral"
            last_direction_index = None
            reasons.append("direction_evidence_expired")
        event_rows = row.get("events") or []
        if not isinstance(event_rows, (list, tuple)):
            raise ValueError("consensus events must be a sequence")
        by_source = {source: [] for source in SOURCES}
        ignored = []
        daily_ids = set()
        for event in event_rows:
            if not isinstance(event, Mapping):
                ignored.append({"event_id": None, "reason_codes": ["invalid_event"]})
                continue
            event_reasons = _event_reasons(event, symbol=symbol, day=day, close_at=close_at)
            identifier = str(event.get("event_id") or "")
            source = event.get("source")
            if source not in SOURCES:
                event_reasons.append("unknown_evidence_source")
            if identifier and (identifier in seen_events or identifier in daily_ids):
                event_reasons.append("duplicate_event_identity")
            if identifier:
                daily_ids.add(identifier)
            if not event_reasons and _opportunity(event) in consumed:
                event_reasons.append("opportunity_already_consumed")
            if event_reasons:
                ignored.append({"event_id": identifier or None, "reason_codes": list(dict.fromkeys(event_reasons))})
            else:
                by_source[source].append(event)
                seen_events.add(identifier)
        evidence = row.get("agents") or {}
        if not isinstance(evidence, Mapping):
            evidence = {}
        agents = {source: _source_agent(source, evidence.get(source) if isinstance(evidence.get(source), Mapping) else {},
            by_source[source], day=day, close_at=close_at, config=values) for source in SOURCES}
        families, dependency_reasons = _supports(agents, by_source, values)
        reasons.extend(dependency_reasons)
        buy_support, sell_support = (sum(families[side].values()) for side in ("BUY", "SELL"))
        proposed = "WAIT"
        if buy_support >= values["min_buy_support"] and buy_support - sell_support >= values["min_direction_margin"]:
            proposed = "BUY"
        elif sell_support >= values["min_sell_support"] and sell_support - buy_support >= values["min_direction_margin"]:
            proposed = "SELL"
        else:
            reasons.append("direction_conflict_or_insufficient_margin" if buy_support and sell_support else
                           "insufficient_independent_support" if buy_support or sell_support else "no_eligible_source_support")
        required = model_family == "high_return" and values["require_high_return_when_family"]
        l2_evidence = evidence.get("high_return") if isinstance(evidence.get("high_return"), Mapping) else {}
        agents["high_return"] = _gate_agent(l2_evidence, day=day, close_at=close_at, config=values,
                                             required=required, has_buy_candidate=any(_side(event) == "BUY"
                                                for events in by_source.values() for event in events))
        action = proposed
        if not exact_price:
            action = "WAIT"
        if proposed == "BUY":
            risk = _risk_reasons(row.get("raw_risk_plan"), symbol=symbol, day=day,
                                 close_at=close_at, required=required)
            reasons.extend(risk)
            if risk:
                action = "WAIT"
            if required and agents["high_return"]["status"] != "passed":
                reasons.extend(agents["high_return"]["reason_codes"])
                action = "WAIT"
        if action in {"BUY", "SELL"}:
            desired = "bullish" if action == "BUY" else "bearish"
            if state == desired:
                reasons.append("same_direction_already_active")
                # Fresh, qualified support sustains the existing direction
                # without placing another marker during the same trend.
                last_direction_index = index
                action = "WAIT"
            elif state == "neutral" and last_emit[action] is not None and index - last_emit[action] < values["cooldown_bars"]:
                reasons.append("same_direction_cooldown")
                action = "WAIT"
        emitted = action in {"BUY", "SELL"}
        selected_ids = sorted({identifier for source in SOURCES if agents[source]["judgment"] == proposed
                               for identifier in agents[source]["event_ids"]})
        selected_opportunities = sorted({identifier for source in SOURCES if agents[source]["judgment"] == proposed
                                         for identifier in agents[source]["opportunity_ids"]})
        if "same_direction_already_active" in reasons:
            consumed.update(selected_opportunities)
        if emitted:
            state = "bullish" if action == "BUY" else "bearish"
            last_direction_index = index
            last_emit[action] = index
            consumed.update(selected_opportunities)
            reasons.append("multi_agent_direction_confirmed")
        agents["arbiter"] = {"judgment": action, "type": "direction_arbitration", "status": "emitted" if emitted else "wait",
            "reason_codes": list(dict.fromkeys(reasons)), "probability": None, "exit_probability": None,
            "counts_as_vote": False, "semantics": SEMANTICS}
        decision = {"date": day, "symbol": symbol, "action": action, "proposed_action": proposed,
            "reason_codes": list(dict.fromkeys(reasons)), "agents": agents,
            "buy_support": buy_support, "sell_support": sell_support,
            "buy_independent_families": sorted(families["BUY"]), "sell_independent_families": sorted(families["SELL"]),
            "buy_independent_count": len(families["BUY"]), "sell_independent_count": len(families["SELL"]),
            "evidence_event_ids": selected_ids, "opportunity_ids": selected_opportunities,
            "state_before": state_before, "state_after": state, "emitted": emitted,
            "reference_price": price, "reference_price_date": price_date, "available_at": available_at,
            "config_hash": config_hash, "semantics": SEMANTICS, "ignored_events": ignored}
        decisions.append(decision)
        if emitted:
            identifier = _hash({"version": CONSENSUS_VERSION, "symbol": symbol, "date": day,
                "action": action, "config_hash": config_hash, "event_ids": selected_ids})
            final_events.append({"event_id": identifier, "symbol": symbol, "date": day, "time": day,
                "evidence_date": day, "signal_at": available_at, "available_at": available_at,
                "source": "consensus", "side": action, "action": action, "event_type": "multi_agent_direction",
                "version": CONSENSUS_VERSION, "candidate_eligible": True, "evidence_visible": True,
                "original_event_ids": selected_ids, "source_event_ids": selected_ids,
                "opportunity_ids": selected_opportunities, "reference_price": price,
                "reference_price_date": day, "reference_price_source": "signal_day_close",
                "reference_price_basis": row.get("reference_price_basis", "unavailable"),
                "reason_codes": decision["reason_codes"], "config_hash": config_hash,
                "semantics": SEMANTICS, "consensus": copy.deepcopy(decision)})
    return {"version": CONSENSUS_VERSION, "semantics": SEMANTICS, "config": values,
            "config_hash": config_hash, "rows": decisions, "events": final_events,
            "initial_state": "neutral", "final_state": state}
