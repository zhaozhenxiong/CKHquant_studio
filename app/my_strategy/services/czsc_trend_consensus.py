"""Continuous, causal direction evidence for the trend consensus strategy.

This module owns opportunities and direction points, never cash or inventory.
The account adapter alone resolves a direction into an executable intent.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, DISPLAY_NAME, strategy_config

CONSENSUS_VERSION = STRATEGY_ID
SOURCES = ("ma", "structure", "wyckoff")
NATIVE_VERSIONS = {"czsc_native_evidence_v4_20261012", "pure_confirmed_czsc_v4_20261012",
                   "wyckoff_current_pv_v4_20261012", "czsc_trend_native_evidence"}


def consensus_config(config=None):
    """Validate a new flat consensus envelope; old policy shapes are rejected."""
    base = strategy_config()["consensus"]
    supplied = (config or {}).get("consensus", config or {})
    if not isinstance(supplied, Mapping) or set(supplied) - set(base):
        raise ValueError("unknown trend consensus configuration keys")
    if supplied.get("version", STRATEGY_ID) != STRATEGY_ID:
        raise ValueError("incompatible consensus identity")
    values = strategy_config({"consensus": supplied})["consensus"]
    for key in ("quality_weight", "buy_base_threshold", "buy_total_threshold", "sell_threshold", "margin", "min_source_strength", "high_return_threshold"):
        value = finite(values[key])
        if value is None or not 0 <= value <= 1:
            raise ValueError("invalid continuous consensus parameter: " + key)
    for key in ("min_independent", "cooldown", "signal_valid_bars", "evidence_valid_bars"):
        value = finite(values[key])
        if value is None or value != int(value) or value < (0 if key == "cooldown" else 1):
            raise ValueError("invalid consensus integer: " + key)
    if values["min_independent"] > 3 or values["evidence_valid_bars"] not in {1, 3, 5}:
        raise ValueError("invalid consensus independence/window")
    if not isinstance(values["require_high_return"], bool):
        raise ValueError("quality requirement must be a boolean")
    return values


validate_consensus_config = consensus_config


def _strategy_config(config):
    if isinstance(config, Mapping) and "consensus" not in config and (set(config).issubset(set(strategy_config()["consensus"])) or "source_weights" in config):
        config = {"consensus": consensus_config(config)}
    cfg = strategy_config(config)
    consensus_config(cfg)
    return cfg


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def day(value):
    return pd.Timestamp(value).date().isoformat()


def observable(value, cutoff):
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            return False
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("Asia/Shanghai")
        return stamp <= pd.Timestamp(cutoff)
    except (ValueError, TypeError):
        return False


def _records(value):
    if value is None:
        return {}
    if isinstance(value, pd.DataFrame):
        value = value.to_dict("records")
    if isinstance(value, Mapping):
        if isinstance(value.get("rows"), list):
            value = value["rows"]
        elif isinstance(value.get("evaluations"), list):
            value = value["evaluations"]
        else:
            return {day(k): v for k, v in value.items() if isinstance(v, Mapping)}
    return {day(item["date"]): item for item in value if isinstance(item, Mapping) and item.get("date")}


def _pair(buy, sell):
    buy, sell = finite(buy), finite(sell)
    if buy is None or sell is None or not 0 <= buy <= 1 or not 0 <= sell <= 1:
        return None, None
    denominator = max(1., buy + sell)
    return buy / denominator, sell / denominator


def _identity_reasons(item, close_at, *, head, allow_policy_validation=False):
    identity = item.get("model_identity") or {}
    reasons = []
    if identity.get("family", identity.get("model_family")) != STRATEGY_ID:
        reasons.append("legacy_or_foreign_model_family")
    if identity.get("verified_frozen_contract") is not True:
        reasons.append("model_contract_unverified")
    if identity.get("head", identity.get("model_head")) != head:
        reasons.append("model_head_mismatch")
    for key in ("checkpoint", "manifest_sha256", "feature_schema_hash"):
        if not identity.get(key):
            reasons.append("model_" + key + "_missing")
    if not (identity.get("label_contract_hash") or identity.get("target_contract_hash")):
        reasons.append("model_target_contract_missing")
    policy = item.get("usage_mode") == "policy_validation"
    policy_allowed = (policy and allow_policy_validation and identity.get("provisional_checkpoint") is True
        and identity.get("policy_start") and identity.get("policy_end")
        and day(identity["policy_start"]) <= day(close_at) <= day(identity["policy_end"]))
    if policy and not policy_allowed:
        reasons.append("provisional_policy_checkpoint_out_of_scope")
    for key in ("train_label_end", "calibration_label_end", "policy_label_end"):
        if key == "policy_label_end" and policy_allowed:
            continue
        if not identity.get(key) or day(identity[key]) >= day(close_at):
            reasons.append("model_" + key + "_not_prior")
    available = identity.get("simulated_available_at", identity.get("available_at"))
    if not available or not observable(available, close_at):
        reasons.append("model_not_available_at_signal")
    if item.get("usage_mode") == "production":
        if not identity.get("actual_training_completed_at") or not observable(identity["actual_training_completed_at"], close_at):
            reasons.append("model_actual_training_after_signal")
    if item.get("date") and day(item["date"]) != day(close_at):
        reasons.append("score_date_mismatch")
    if not item.get("available_at") or not observable(item["available_at"], close_at):
        reasons.append("score_not_observable")
    return reasons


def _ma(item, close_at, allow_policy_validation=False):
    result = {"label": "均线 ML", "status": "unavailable", "buy_strength": None, "sell_strength": None,
              "p_up": finite(item.get("p_up")), "p_down": finite(item.get("p_down")),
              "p_neutral": finite(item.get("p_neutral")), "model_identity": copy.deepcopy(item.get("model_identity")),
              "reason_codes": [], "family": "ma"}
    values = [result[key] for key in ("p_up", "p_down", "p_neutral")]
    if item.get("input_eligible") is not True:
        result["reason_codes"] = ["trend_input_ineligible"]
    elif item.get("model_available") is not True or any(v is None for v in values):
        result["reason_codes"] = list(item.get("reason_codes") or ["trend_model_unavailable"])
    elif any(not 0 <= v <= 1 for v in values) or not math.isclose(sum(values), 1., abs_tol=1e-6):
        result["reason_codes"] = ["trend_probability_simplex_invalid"]
    else:
        reasons = _identity_reasons(item, close_at, head="trend_direction", allow_policy_validation=allow_policy_validation)
        result["reason_codes"] = reasons
        if not reasons:
            result.update(status="scored", buy_strength=values[0], sell_strength=values[1])
    return result


def _native(item, source, close_at, price, permanently_invalidated=(), symbol=None):
    state = item.get("current_state") or {}
    result = {"label": "结构" if source == "structure" else "威科夫量价", "status": "unavailable",
              "buy_strength": None, "sell_strength": None, "current_state": copy.deepcopy(state),
              "reason_codes": [], "family": "ma" if item.get("auxiliary") else source,
              "origin_event_ids": [], "snapshot_id": item.get("snapshot_id", item.get("source_id"))}
    reasons = []
    if item.get("input_eligible") is not True:
        reasons.append("native_input_ineligible")
    if item.get("version") not in NATIVE_VERSIONS or not result["snapshot_id"]:
        reasons.append("native_identity_missing")
    if item.get("date") and day(item["date"]) != day(close_at):
        reasons.append("native_snapshot_date_mismatch")
    if not item.get("available_at") or not observable(item["available_at"], close_at):
        reasons.append("native_snapshot_not_observable")
    if finite(state.get("close")) != price:
        reasons.append("native_snapshot_price_mismatch")
    if reasons:
        result["reason_codes"] = reasons
        return result
    invalid = set(item.get("invalidated_event_ids") or []) | set(permanently_invalidated)
    facts = [fact for fact in item.get("support_facts", []) if isinstance(fact, Mapping)
        and fact.get("confirmed") is True and fact.get("event_id") not in invalid
        and fact.get("original_event_id") not in invalid and observable(fact.get("available_at"), close_at)
        and fact.get("source") == source and (symbol is None or fact.get("symbol") == symbol)
        and all(not fact.get(key) or observable(fact[key], close_at) for key in ("first_confirmed_at", "first_available_at", "origin_observed_at", "observed_at"))]
    valid_facts = []
    for fact in facts:
        boundary = finite(fact.get("invalidation_price"))
        if boundary is None or price is None:
            continue
        if source == "structure":
            valid = price > boundary if fact.get("side") == "BUY" else price < boundary
        else:
            valid = price >= boundary if fact.get("side") == "BUY" else price <= boundary
        if valid:
            valid_facts.append(fact)
    facts = valid_facts
    if source == "structure":
        buy, sell = finite(state.get("buy_score")), finite(state.get("sell_score"))
        if state.get("support_valid") is not True or price is None or finite(state.get("support_floor")) is None or price <= float(state["support_floor"]):
            buy = 0.
        if state.get("resistance_valid") is not True or price is None or finite(state.get("resistance_ceiling")) is None or price >= float(state["resistance_ceiling"]):
            sell = 0.
        result["raw_buy_strength"], result["raw_sell_strength"] = buy, sell
    else:
        buy, sell = finite(state.get("buy_score")), finite(state.get("sell_score"))
        result.update(current_demand=buy, current_supply=sell)
        event_strengths = {"BUY": 0., "SELL": 0.}
        for fact in facts:
            side = fact.get("side")
            boundary, strength = finite(fact.get("invalidation_price")), finite(fact.get("event_strength"))
            if side not in event_strengths or boundary is None or strength is None or price is None:
                continue
            if (side == "BUY" and price >= boundary) or (side == "SELL" and price <= boundary):
                event_strengths[side] = max(event_strengths[side], strength)
        result.update(event_demand=event_strengths["BUY"], event_supply=event_strengths["SELL"])
        buy = max(buy, event_strengths["BUY"]) if buy is not None else None
        sell = max(sell, event_strengths["SELL"]) if sell is not None else None
        result["raw_buy_strength"], result["raw_sell_strength"] = buy, sell
    buy, sell = _pair(buy, sell)
    if buy is None:
        result["reason_codes"] = ["native_continuous_strength_missing"]
    elif item.get("direction_conflict") is True:
        result.update(status="conflict", buy_strength=0., sell_strength=0., reason_codes=["same_source_explicit_conflict"])
    else:
        result.update(status="scored", buy_strength=buy, sell_strength=sell,
            origin_event_ids=sorted({str(fact.get("original_event_id") or fact["event_id"]) for fact in facts}))
    return result


def _quality(item, close_at, has_candidate, required):
    probability = finite(item.get("probability"))
    result = {"label": "高收益", "probability": probability, "status": str(item.get("evaluation_status") or item.get("status") or "model_unavailable"),
              "required": required, "buy_strength": None, "sell_strength": 0., "reason_codes": list(item.get("reason_codes") or [])}
    if not required:
        result.update(status="bootstrap_not_evaluated", reason_codes=["E0_E1_do_not_read_high_return"], probability=None)
    elif not has_candidate:
        result.update(status="no_candidate", probability=None, reason_codes=["no_buy_candidate"])
    elif probability is not None and 0 <= probability <= 1 and item.get("status") in {"scored", "scored_passed", "scored_rejected"}:
        reasons = _identity_reasons(item, close_at, head="entry")
        if item.get("in_scope") is not True:
            reasons.append("high_return_population_out_of_scope")
        result.update(status="identity_error" if reasons else "scored", buy_strength=probability if not reasons else None,
                      reason_codes=reasons)
    return result


def _risk_plan(row, events, index, bars, config):
    supplied = row.get("raw_risk_plan") or row.get("entry_plan")
    if isinstance(supplied, Mapping):
        return copy.deepcopy(dict(supplied))
    close = finite(row.get("close", row.get("reference_price")))
    params = config["signal_points"]
    history = bars[max(0, index - 20):index]
    trs = []
    for i in range(max(1, index - 14), index):
        bar, previous = bars[i], bars[i - 1]
        hi, lo, prev = (finite(bar.get("high")), finite(bar.get("low")), finite(previous.get("close")))
        if None in (hi, lo, prev):
            break
        trs.append(max(hi - lo, abs(hi - prev), abs(lo - prev)))
    atr = sum(trs) / len(trs) if len(trs) == 14 else None
    lows = [finite(bar.get("low")) for bar in history]
    stops = [finite(event.get("risk_stop_price", event.get("stop_price"))) for event in events]
    stops = [stop for stop in stops if stop is not None]
    if any(event.get("source") == "ma" for event in events) and close and atr and lows and all(v is not None for v in lows):
        stops.append(max(min(lows), close - params["atr_stop_multiple"] * atr) - params["stop_atr_buffer"] * atr)
    stop = max(stops) if stops else None
    errors = []
    if close is None or atr is None or stop is None or not 0 < stop < close:
        errors.append("risk_inputs_unavailable")
    elif not params["min_risk_fraction"] <= (close - stop) / close <= params["max_stop_distance"]:
        errors.append("risk_stop_distance_outside_contract")
    floor = max(stop + .01, stop / (1 - params["min_risk_fraction"])) if not errors else None
    ceiling = min(close + params["max_next_open_gap_atr"] * atr, stop / (1 - params["max_stop_distance"])) if not errors else None
    if floor is not None and floor > ceiling:
        errors.append("risk_execution_band_invalid")
    identifiers = sorted(str(event["event_id"]) for event in events)
    boundaries = [{"symbol": row["symbol"], "source": event["source"], "event_id": event["event_id"],
        "original_event_id": event["original_event_id"], "pattern_id": event["pattern_id"],
        "price": finite(event.get("invalidation_price")), "first_available_at": event.get("first_available_at", event["available_at"])}
        for event in events if event.get("source") in {"structure", "wyckoff"}
        and event.get("confirmed") is True and finite(event.get("invalidation_price")) is not None]
    return {"plan_id": stable_hash({"strategy": STRATEGY_ID, "symbol": row["symbol"], "date": row["date"], "events": identifiers,
        "risk": params}), "symbol": row["symbol"], "signal_date": row["date"], "available_at": row["date"] + "T15:00:00+08:00",
        "policy": "risk", "status": "rejected" if errors else "active", "entry_allowed": not errors,
        "valid_for_sessions": 1, "stop_price": stop, "price_floor": floor, "price_ceiling": ceiling,
        "reference_price": close, "atr14": atr,
        "target_weight": min(config["account"]["max_weight"], params["risk_fraction"] * close / (close - stop)) if not errors else 0.,
        "source_event_ids": identifiers, "sources": sorted({event["source"] for event in events}),
        "holding_boundaries": boundaries, "reason_codes": errors}


def _long_confirmation(bar, side):
    above, slope, alignment, ret = (finite(bar.get(key)) for key in ("close_above_ma20", "ma20_slope5", "ma_alignment", "ret_5"))
    if None in (above, slope, alignment, ret):
        return False
    return above > 0 and slope > 0 and (alignment > 0 or ret > 0) if side == "BUY" else above <= 0 and slope < 0


def prepare_trend_decisions(frame, market_dates, evidence_pack=None, trend_probabilities=None, config=None,
                            *, quality=None, mode="E0", native_pack=None, allow_policy_validation=False):
    """Build immutable daily opportunities and continuous direction decisions.

    E0/E1 are training continuation policies and do not inspect quality scores.
    A final request uses mode='final' and its matched L2 candidate predictions.
    """
    cfg = _strategy_config(config)
    if mode not in {"E0", "E1", "final"}:
        raise ValueError("unknown continuation mode")
    rows = frame.to_dict("records") if isinstance(frame, pd.DataFrame) else copy.deepcopy(list(frame))
    dates = [day(row["date"]) for row in rows]
    if dates != sorted(set(dates)) or not rows or len({row["symbol"] for row in rows}) != 1:
        raise ValueError("direction requires one symbol, unique ascending daily bars")
    if list(market_dates) != sorted(set(market_dates)) or not set(dates).issubset(set(market_dates)):
        raise ValueError("verified calendar must include every bar")
    pack = evidence_pack or native_pack or (frame.attrs.get("native_pack") if isinstance(frame, pd.DataFrame) else None) or {}
    native = _records(pack.get("evaluations", []))
    trend, qualities = _records(trend_probabilities), _records(quality)
    events_by_day = {}
    for event in pack.get("events", []):
        if event.get("source") == "ma" and not event.get("auxiliary") and event.get("event_type") not in {"trend_probability_cross", "trend_confirmed_activation"}:
            continue
        events_by_day.setdefault(day(event["date"]), []).append(copy.deepcopy(event))
    output, emitted, known, consumed, permanently_invalidated = [], [], {}, set(), set()
    last_emit = {"BUY": -100000, "SELL": -100000}
    previous_trend, previous_bar = None, None
    c = cfg["consensus"]
    for index, original in enumerate(rows):
        bar = copy.deepcopy(original); bar["date"] = dates[index]
        date, symbol = bar["date"], bar["symbol"]
        close_at = date + "T15:00:00+08:00"
        price = finite(bar.get("close", bar.get("reference_price")))
        ma_item = copy.deepcopy(trend.get(date) or (bar.get("agents") or {}).get("ma") or {})
        native_day = native.get(date) or bar.get("native_evaluation") or {}
        for source in ("structure", "wyckoff"):
            permanently_invalidated.update((native_day.get(source) or {}).get("invalidated_event_ids") or [])
        agents = {"ma": _ma(ma_item, close_at, allow_policy_validation)}
        for source in ("structure", "wyckoff"):
            item = copy.deepcopy(native_day.get(source) or (bar.get("agents") or {}).get(source) or {})
            item.setdefault("date", date); item.setdefault("available_at", native_day.get("available_at"))
            agents[source] = _native(item, source, close_at, price, permanently_invalidated, symbol)
        facts = copy.deepcopy(events_by_day.get(date, []) + list(bar.get("candidate_evidence") or []))
        if agents["ma"]["status"] == "scored":
            identity = ma_item.get("model_identity") or {}
            key = (identity.get("checkpoint"), identity.get("manifest_sha256"))
            previous_key = ((previous_trend or {}).get("model_identity") or {})
            previous_key = (previous_key.get("checkpoint"), previous_key.get("manifest_sha256"))
            for side, field, opposite in (("BUY", "p_up", "p_down"), ("SELL", "p_down", "p_up")):
                threshold = cfg["trend"]["candidate_cross_threshold"]
                current_p = agents["ma"][field]
                previous_p = finite((previous_trend or {}).get(field))
                crossed = previous_p is not None and key == previous_key and previous_p < threshold <= current_p
                onset = previous_bar is not None and not _long_confirmation(previous_bar, side) and _long_confirmation(bar, side)
                if (crossed or onset) and current_p >= threshold and current_p - agents["ma"][opposite] >= cfg["trend"]["candidate_margin"] and _long_confirmation(bar, side):
                    pattern = stable_hash({"strategy": STRATEGY_ID, "symbol": symbol, "side": side, "date": date,
                        "checkpoint": key, "cause": "cross" if crossed else "trend_confirmation"})
                    facts.append({"event_id": pattern, "original_event_id": pattern, "pattern_id": pattern,
                        "symbol": symbol, "date": date, "available_at": close_at, "first_available_at": close_at,
                        "source": "ma", "side": side, "event_type": "trend_probability_cross" if crossed else "trend_confirmed_activation",
                        "confirmed": True, "input_eligible": True, "direction_eligible": True, "candidate_eligible": True,
                        "dependencies": ["trailing_sma_5_10_20_60"], "model_identity": copy.deepcopy(identity)})
        previous_trend = ma_item if agents["ma"]["status"] == "scored" else None
        previous_bar = bar
        invalid = permanently_invalidated
        reasons = []
        for fact in facts:
            if fact.get("source") not in SOURCES or fact.get("side") not in {"BUY", "SELL"} or fact.get("symbol") != symbol:
                reasons.append("foreign_or_unknown_candidate")
                continue
            if fact.get("confirmed") is not True or fact.get("input_eligible") is not True or not observable(fact.get("available_at"), close_at):
                reasons.append("candidate_not_confirmed_or_observable")
                continue
            if not fact.get("date") or day(fact["date"]) > date or any(fact.get(key) and not observable(fact[key], close_at)
                    for key in ("first_confirmed_at", "first_available_at", "origin_observed_at", "observed_at")):
                reasons.append("candidate_origin_not_observable")
                continue
            if not fact.get("event_id") or not fact.get("pattern_id") or not fact.get("original_event_id"):
                reasons.append("candidate_origin_identity_missing")
                continue
            opportunity = fact["source"] + ":" + str(fact["pattern_id"]) + ":" + fact["side"]
            if opportunity not in known:
                known[opportunity] = {"event": fact, "index": index}
        active = []
        for opportunity, entry in known.items():
            fact = entry["event"]; source = fact["source"]
            if opportunity in consumed or index - entry["index"] >= c["signal_valid_bars"]:
                continue
            if fact["event_id"] in invalid or fact["original_event_id"] in invalid:
                consumed.add(opportunity)
                continue
            if agents[source]["status"] != "scored":
                continue
            if source == "ma":
                valid = _long_confirmation(bar, fact["side"])
            else:
                origins = agents[source]["origin_event_ids"]
                valid = fact["original_event_id"] in origins or fact["event_id"] in origins
            if valid:
                active.append({**copy.deepcopy(fact), "opportunity_id": opportunity})
        buy_candidates = [fact for fact in active if fact["side"] == "BUY"]
        plan = _risk_plan(bar, buy_candidates, index, rows, cfg) if buy_candidates else None
        quality_item = copy.deepcopy(qualities.get(date) or (bar.get("agents") or {}).get("high_return") or {})
        agents["high_return"] = _quality(quality_item, close_at, bool(buy_candidates), mode == "final" and c["require_high_return"])
        family_allocations = {}
        for source in SOURCES:
            agent = agents[source]; weight = c["source_weights"][source]
            agent.update(configured_weight=weight, buy_contribution=None, sell_contribution=None)
            if agent["buy_strength"] is not None and agent["sell_strength"] is not None:
                family_allocations.setdefault(agent["family"], []).append((source, weight * agent["buy_strength"], weight * agent["sell_strength"]))
        # Related definitions share the existing MA budget, never spend 40+20.
        contributions = {}
        for family, entries in family_allocations.items():
            ceiling = c["source_weights"].get(family, 0.)
            total = sum(buy + sell for _, buy, sell in entries)
            scale = min(1., ceiling / total) if total else 1.
            for source, buy, sell in entries:
                contributions[source] = (buy * scale, sell * scale)
        for source, (buy, sell) in contributions.items():
            agents[source].update(buy_contribution=buy, sell_contribution=sell)
        buy_base = math.fsum(agent["buy_contribution"] or 0. for agent in (agents[k] for k in SOURCES))
        sell_base = math.fsum(agent["sell_contribution"] or 0. for agent in (agents[k] for k in SOURCES))
        families = {}
        for side in ("BUY", "SELL"):
            families[side] = sorted({agents[source]["family"] for source in SOURCES
                if agents[source][side.lower() + "_strength"] is not None
                and agents[source][side.lower() + "_strength"] >= c["min_source_strength"]
                and agents[source][side.lower() + "_strength"] > agents[source][("sell" if side == "BUY" else "buy") + "_strength"]})
        quality_strength = agents["high_return"]["buy_strength"]
        quality_contribution = c["quality_weight"] * quality_strength if quality_strength is not None else None
        agents["high_return"].update(configured_weight=c["quality_weight"], buy_contribution=quality_contribution,
            sell_contribution=0., quality_contribution=quality_contribution)
        buy_total = buy_base + (quality_contribution or 0.)
        buy_pass = buy_base + 1e-12 >= c["buy_base_threshold"] and buy_base - sell_base + 1e-12 >= c["margin"] and len(families["BUY"]) >= c["min_independent"]
        sell_pass = sell_base + 1e-12 >= c["sell_threshold"] and sell_base - buy_base + 1e-12 >= c["margin"] and len(families["SELL"]) >= c["min_independent"]
        proposed = "BUY" if buy_pass else "SELL" if sell_pass else "WAIT"
        action = proposed
        selected = [fact for fact in active if fact["side"] == proposed]
        if not selected:
            action = "WAIT"; reasons.append("no_live_unconsumed_opportunity")
        if index - last_emit.get(proposed, -100000) < c["cooldown"]:
            action = "WAIT"; reasons.append("opportunity_cooldown")
        if proposed == "BUY":
            if not plan or plan.get("status") != "active" or plan.get("entry_allowed") is not True:
                action = "WAIT"; reasons.extend((plan or {}).get("reason_codes") or ["risk_plan_unavailable"])
            if mode == "final" and c["require_high_return"]:
                if quality_strength is None or quality_strength < c["high_return_threshold"] or buy_total + 1e-12 < c["buy_total_threshold"]:
                    action = "WAIT"; reasons.append("high_return_or_total_support_not_qualified")
        row = {"version": STRATEGY_ID, "display_name": DISPLAY_NAME, "date": date, "available_at": close_at,
            "symbol": symbol, "reference_price": price, "agents": agents, "action": action, "direction": proposed,
            "buy_base_score": buy_base, "sell_base_score": sell_base, "buy_total_score": buy_total,
            "buy_quality_bonus": quality_contribution, "buy_independent_families": families["BUY"],
            "sell_independent_families": families["SELL"], "candidate_evidence": facts,
            "active_candidates": active, "fresh_candidate_ids": [fact["event_id"] for fact in selected],
            "entry_plan": plan, "raw_risk_plan": plan, "reason_codes": sorted(set(reasons)), "mode": mode,
            "emitted": action in {"BUY", "SELL"}, "config_hash": stable_hash(cfg)}
        if row["emitted"]:
            row["event_id"] = stable_hash({"strategy": STRATEGY_ID, "symbol": symbol, "date": date,
                "side": action, "opportunities": sorted(fact["opportunity_id"] for fact in selected), "config": row["config_hash"]})
            consumed.update(fact["opportunity_id"] for fact in selected)
            last_emit[action] = index
            emitted.append({"event_id": row["event_id"], "date": date, "available_at": close_at,
                "symbol": symbol, "side": action, "source": "consensus", "source_event_ids": row["fresh_candidate_ids"],
                "reason_codes": row["reason_codes"], "reference_price": price, "consensus": copy.deepcopy(row)})
        output.append(row)
    return {"version": STRATEGY_ID, "display_name": DISPLAY_NAME, "rows": output, "events": emitted,
            "direction_points": emitted, "config": cfg["consensus"], "config_hash": stable_hash(cfg),
            "candidate_policy": cfg["candidate_policy"], "mode": mode}


def build_consensus(rows, *, symbol=None, config=None, version=None, **kwargs):
    """Adapter for callers that already supply one enriched row per session."""
    if version not in {None, STRATEGY_ID}:
        raise ValueError("incompatible trend consensus identity")
    frame = pd.DataFrame(rows)
    if symbol is not None and set(frame.symbol) != {symbol}:
        raise ValueError("consensus symbol mismatch")
    return prepare_trend_decisions(frame, sorted(day(v) for v in frame.date), config=config, **kwargs)
