"""Independent causal signal facts and one-session risk plans for new research.

This layer owns neither a Position nor inventory. All raw evidence is retained;
the actual Broker decides whether a candidate can open and whether a sell can
close. Old combined rules and fixed-ten-day model identities remain untouched.
"""
from __future__ import annotations

import copy
import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS


SIGNAL_POINTS_VERSION = "czsc_independent_native_signal_points_v4"
SOURCES = ("ma", "structure", "wyckoff")
MA_ENTRY_TARGET = "actual_complete_fee_net_roundtrip_positive"
MA_EXIT_TARGET = "remaining_net_sell_proceeds_exit_next_open_minus_frozen_continue_over_current_market_value"
DEFAULT_PARAMETERS = {
    "risk_fraction": .01, "min_risk_fraction": .01, "max_stop_distance": .15,
    "atr_stop_multiple": 2., "lookback_stop_bars": 20, "stop_atr_buffer": .25,
    "max_next_open_gap_atr": 1., "ma_cooldown_bars": 5,
    "ma_entry_threshold": .55, "ma_exit_threshold": .55,
}


def signal_point_parameters(config=None):
    values = {**DEFAULT_PARAMETERS, **(config or {}).get("signal_points", {})}
    for key in DEFAULT_PARAMETERS:
        value = _finite(values[key])
        if value is None or value < 0:
            raise ValueError("invalid signal point parameter: " + key)
        values[key] = value
    for key in ("lookback_stop_bars", "ma_cooldown_bars"):
        if values[key] != int(values[key]) or values[key] < 1:
            raise ValueError("invalid signal point integer: " + key)
        values[key] = int(values[key])
    for key in ("risk_fraction", "min_risk_fraction", "max_stop_distance"):
        if not 0 < values[key] < 1:
            raise ValueError("invalid signal point risk bound: " + key)
    if values["min_risk_fraction"] > values["max_stop_distance"]:
        raise ValueError("risk floor exceeds maximum stop distance")
    for key in ("ma_entry_threshold", "ma_exit_threshold"):
        if not 0 < values[key] < 1:
            raise ValueError("invalid MA threshold: " + key)
    if values["atr_stop_multiple"] <= 0:
        raise ValueError("ATR stop multiple must be positive")
    return values


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _flag(value):
    return value is True or isinstance(value, (int, float, np.bool_, np.number)) and _finite(value) == 1.


def _text(value):
    return None if value is None or str(value) in {"", "nan", "NaT", "None"} else str(value)


def _visible(value, close_at):
    if _text(value) is None:
        return False
    try:
        instant = pd.Timestamp(value)
        if instant.tzinfo is None:
            instant = instant.tz_localize("Asia/Shanghai")
        return instant <= pd.Timestamp(close_at)
    except (ValueError, TypeError):
        return False


def _records(frame):
    if not {"symbol", "date", "close", "low"} <= set(frame):
        raise ValueError("signal points require symbol, date, close and low")
    dates = pd.to_datetime(frame.date, errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("signal points require unique ascending dates")
    if not frame.empty and frame.symbol.nunique(dropna=False) != 1:
        raise ValueError("signal points require exactly one symbol")
    return frame.to_dict("records"), dates.dt.strftime("%Y-%m-%d").tolist()


def _probabilities(values, count, name):
    if values is None:
        return None
    result = np.asarray(values, dtype=float)
    if result.shape != (count,) or ((np.isfinite(result)) & ((result < 0) | (result > 1))).any():
        raise ValueError(name + " must be one probability per row within [0,1]")
    return result


def _fallback_mask(values, count):
    if values is None:
        return np.zeros(count, dtype=bool)
    result = np.asarray(values)
    if result.shape != (count,) or any(not isinstance(value, (bool, np.bool_)) for value in result):
        raise ValueError("MA fallback mask must be one explicit boolean per row")
    return result.astype(bool)


def _entry_route(entry, fallback, lineages, index, close_at, parameters):
    """Distinguish no model yet from a missing score of an available model."""
    rule_lineage = {"source": "causal_rule_v0", "rule_version": SIGNAL_POINTS_VERSION,
        "available_at": close_at, "config_hash": stable_hash(parameters)}
    if entry is None:
        return {"mode": "causal_rule_v0", "status": "rule_v0_model_not_requested", "probability": None,
            "target": None, "reason_codes": [], "lineage": rule_lineage,
            "source_reason": "ma_entry_model_not_requested"}
    probability = _finite(entry[index])
    lineage, reasons, target = _model_lineage(lineages, index, "entry", close_at)
    if fallback[index] and probability is not None:
        raise ValueError("MA fallback cannot replace a finite model probability")
    if probability is None and fallback[index]:
        model_available = lineage.get("available_at") and _visible(lineage["available_at"], close_at)
        model_available = model_available or lineage.get("model_available") is True
        if not model_available:
            rule_lineage.update(fallback_reason="no_compatible_ma_model_available_at_signal",
                unavailable_model_lineage=lineage, unavailable_model_reason_codes=reasons)
            return {"mode": "causal_rule_v0", "status": "explicit_pre_model_rule_fallback", "probability": None,
                "target": None, "reason_codes": [], "lineage": rule_lineage,
                "source_reason": "no_compatible_ma_model_available_at_signal"}
        reasons.append("ma_fallback_after_available_model_rejected")
    if probability is None:
        reasons.append("ma_entry_probability_unavailable")
    elif probability < parameters["ma_entry_threshold"]:
        reasons.append("ma_entry_probability_veto")
    return {"mode": "forward_model", "status": "model_selected" if not reasons else "model_score_unavailable_or_vetoed",
        "probability": probability, "target": target, "reason_codes": list(dict.fromkeys(reasons)),
        "lineage": lineage, "source_reason": "forward_ma_entry_model"}


def _model_lineage(lineages, index, head, close_at):
    value = copy.deepcopy(lineages[index]) if lineages is not None else {}
    if head in value:
        value = value[head]
    if not isinstance(value, dict):
        raise ValueError("MA lineage must be a mapping")
    reasons = []
    if not value or not value.get("available_at"):
        reasons.append("ma_model_lineage_unavailable")
    elif not _visible(value["available_at"], close_at):
        reasons.append("ma_model_not_available")
    if value.get("retrospective") or value.get("retrospective_inference"):
        reasons.append("ma_retrospective_model_rejected")
    for key in ("trained_at", "actual_training_completed_at"):
        if value.get(key) and not _visible(value[key], close_at):
            reasons.append("ma_training_after_signal")
    cutoff = value.get("label_cutoff") or value.get("train_label_end")
    if cutoff and (not _visible(cutoff, close_at) or str(cutoff)[:10] >= close_at[:10]):
        reasons.append("ma_label_overlap")
    target = value.get("target") or value.get("probability_target")
    if not target:
        reasons.append("ma_probability_target_unavailable")
    elif "fixed_10" in str(target) or "fixed_horizon" in str(target):
        reasons.append("ma_fixed_horizon_target_rejected")
    elif head == "exit" and target != MA_EXIT_TARGET:
        reasons.append("ma_exit_target_mismatch")
    elif head == "entry" and target != MA_ENTRY_TARGET:
        reasons.append("ma_entry_target_mismatch")
    if not _text(value.get("target_contract_hash")):
        reasons.append("ma_target_contract_hash_unavailable")
    if value.get("head") and value["head"] not in {head, "policy_entry" if head == "entry" else "holding_exit"}:
        reasons.append("ma_model_head_mismatch")
    return value, list(dict.fromkeys(reasons)), target


def _risk(row, source, prior_low, parameters, close_at):
    close, ratio = _finite(row.get("close")), _finite(row.get("atr14_ratio"))
    if close is None or close <= 0 or ratio is None or ratio <= 0:
        return None, None, ["risk_inputs_unavailable"]
    atr = close * ratio
    support = None
    if source == "structure" and _visible(row.get("structure_confirmed_at"), close_at):
        support = _finite(row.get("structure_low"))
    elif source == "wyckoff":
        support = _finite(row.get("wyckoff_stop"))
        if support is None:
            support = _finite(row.get("wyckoff_range_low"))
    if support is None or not 0 < support < close:
        support = _finite(prior_low)
    candidate = support - parameters["stop_atr_buffer"] * atr if support is not None and 0 < support < close else close - parameters["atr_stop_multiple"] * atr
    # The frozen floor limits near-zero R denominators; never fit it to outcomes.
    stop = min(candidate, close * (1 - parameters["min_risk_fraction"]))
    if not 0 < stop < close:
        return None, atr, ["invalid_causal_stop"]
    if (close - stop) / close > parameters["max_stop_distance"]:
        return stop, atr, ["stop_distance_exceeds_frozen_maximum"]
    return stop, atr, []




CANDIDATE_POLICY = "independent_points_union_v4_native"
RISK_POLICY = "source_confirmed_boundary_prior_atr_v4"

def policy_identity(config, native_config_hash):
    parameters = signal_point_parameters(config)
    risk = {"version": (config or {}).get("risk_policy", RISK_POLICY), "parameters": parameters,
            "ma_boundary": "prior20_low" if (config or {}).get("risk_policy")=="legacy_trailing20_v4_control" else "max_prior20_low_close_minus_2_prior_atr14", "native_boundary": "frozen_event_risk_stop"}
    return {"candidate_policy_version": CANDIDATE_POLICY,
            "candidate_policy_hash": stable_hash({"version": CANDIDATE_POLICY, "native": native_config_hash,
                "risk": risk, "activation": "first_risk_eligible_while_original_still_valid_once"}),
            "risk_policy_hash": stable_hash(risk)}


def build_signal_points(frame, config=None, *, native_pack=None, ma_routes=None):
    """New population, stable facts, and one explicit first-eligibility transition.

    Continuous support never manufactures a candidate. MA entry scores are
    applied to the common envelope, not used to manufacture the raw population.
    """
    from my_strategy.services.czsc_signal_points import build_signal_points as legacy_points
    rows, dates = _records(frame)
    native_pack = native_pack or frame.attrs.get("native_pack")
    if native_pack is None:
        from my_strategy.services.czsc_native_evidence_v4 import build_native_evidence_v4
        native_pack = build_native_evidence_v4(frame, config)
        frame = native_pack["frame"]; rows, dates = _records(frame)
    parameters = signal_point_parameters(config)
    identity = policy_identity(config, stable_hash({"config":native_pack["config_hash"],"schema":native_pack["schema_hash"]}))
    atr = pd.concat([(frame.high-frame.low).abs(), (frame.high-frame.close.shift()).abs(),
                     (frame.low-frame.close.shift()).abs()], axis=1).max(axis=1).shift().rolling(14,min_periods=14).mean()
    lows = frame.low.shift().rolling(20,min_periods=1).min()
    broad = legacy_points(frame, config)
    fresh = []
    previous_buy, previous_failure, previous_return = False, False, None
    last_buy, last_sell = -parameters["ma_cooldown_bars"], -parameters["ma_cooldown_bars"]
    for i,(row,day) in enumerate(zip(rows,dates,strict=True)):
        okay = _flag(row.get("model_input_eligible")) and all(_finite(row.get(k)) is not None for k in MA_TREND_COLUMNS)
        buy = bool(okay and row["close_above_ma20"]>0 and row["ma20_slope5"]>0 and (row["ma_alignment"]>0 or row["ret_5"]>0))
        failure = bool(okay and row["close_above_ma20"]<=0 and row["ma20_slope5"]<0)
        recovery = buy and previous_return is not None and previous_return<=0 and row["ret_1"]>0
        for side,kind,trigger in (("BUY","ma_trend_recovery" if recovery else "ma_trend_onset",buy and (not previous_buy or recovery) and i-last_buy>=parameters["ma_cooldown_bars"]),
                                  ("SELL","ma_trend_failure",failure and not previous_failure and i-last_sell>=parameters["ma_cooldown_bars"])):
            if not trigger:continue
            stamp=day+"T15:00:00+08:00";eid=stable_hash({"version":CANDIDATE_POLICY,"symbol":row["symbol"],"type":kind,"first_confirmed":stamp})
            fresh.append({"event_id":eid,"pattern_id":eid,"symbol":row["symbol"],"date":day,"source":"ma","side":side,
                "event_type":kind,"observed_at":stamp,"first_confirmed_at":stamp,"first_available_at":stamp,"available_at":stamp,
                "confirmed":True,"input_eligible":okay,"evidence_visible":okay,"candidate_eligible":True,
                "dependencies":["trailing_sma_5_10_20_60"],"model_lineage":{}})
            if side=="BUY":last_buy=i
            else:last_sell=i
        previous_buy,previous_failure=buy,failure
        previous_return=_finite(row.get("ret_1")) if okay else None
    # Auxiliary definitions remain in the MA dependency family. Risk rejection
    # cannot alter their original pattern clock or make repeated source facts.
    seen_aux = set()
    for event in broad["events"]:
        if event["source"] != "structure": continue
        key = stable_hash({"symbol":event["symbol"],"type":event["event_type"],
                          "anchor":event.get("anchor_at"),"confirmed":event.get("observed_at")})
        if key in seen_aux: continue
        seen_aux.add(key)
        event = copy.deepcopy(event); event["input_eligible"] = bool(event.get("input_eligible") and rows[event["feature_index"]].get("v4_auxiliary_input_eligible",False)); event.update(source="ma", event_id=key, pattern_id=key,
            first_confirmed_at=event.get("observed_at"), auxiliary=True,
            dependencies=["native_ma_assisted_definition"])
        fresh.append(event)
    fresh.extend(copy.deepcopy(native_pack.get("events", [])))
    by_day = {}
    for e in fresh: by_day.setdefault(str(e["date"])[:10], []).append(e)
    evaluations = {e["date"]: e for e in native_pack.get("evaluations", [])} if isinstance(native_pack.get("evaluations"),list) else native_pack.get("evaluations",{})
    events, candidates, ledger, activated = [], {}, {}, set()
    for i, (row,day) in enumerate(zip(rows, dates, strict=True)):
        stamp = day+"T15:00:00+08:00"
        def risk(event):
            if event["side"] != "BUY": return None, []
            close = _finite(row.get("close")); current_atr = _finite(atr.iloc[i])
            stop = _finite(event.get("risk_stop_price"))
            if event["source"] == "ma":
                boundary = _finite(lows.iloc[i])
                if current_atr and (config or {}).get("risk_policy", RISK_POLICY) != "legacy_trailing20_v4_control":
                    boundary = max(boundary or -math.inf, close-parameters["atr_stop_multiple"]*current_atr)
                stop = boundary-parameters["stop_atr_buffer"]*current_atr if boundary and current_atr else None
            elif stop is None:
                boundary = _finite(event.get("stop_price") or event.get("support_low"))
                stop = boundary-parameters["stop_atr_buffer"]*current_atr if boundary and current_atr else None
            reasons = []
            if current_atr is None or current_atr <= 0: reasons.append("risk_atr_unavailable")
            if close is None or stop is None or not 0 < stop < close: reasons.append("risk_stop_unavailable_or_broken")
            elif (close-stop)/close > parameters["max_stop_distance"]: reasons.append("stop_distance_exceeds_frozen_maximum")
            elif (close-stop)/close < parameters["min_risk_fraction"]: reasons.append("risk_distance_below_frozen_minimum")
            return stop, reasons
        def record(original, activation=False):
            event = copy.deepcopy(original)
            origin = event.get("original_event_id") or event["event_id"]
            event.update(origin_reference_price=event.get("origin_reference_price",event.get("reference_price")), origin_reference_price_date=event.get("origin_reference_price_date",event.get("reference_price_date")), reference_price=float(row["close"]),reference_price_date=day,reference_price_source="signal_day_close", original_event_id=origin, pattern_id=event.get("pattern_id",origin),
                first_confirmed_at=event.get("first_confirmed_at") or event.get("observed_at"),
                origin_observed_at=event.get("origin_observed_at") or event.get("observed_at"),
                observed_at=event.get("observed_at") or stamp, available_at=stamp if activation else event.get("available_at",stamp),
                date=day, signal_at=stamp, feature_index=i, version=SIGNAL_POINTS_VERSION, **identity)
            if activation:
                event.update(event_id=stable_hash({"original":origin,"first_eligibility_at":stamp,"policy":identity}),
                    activation_kind="first_risk_eligibility", activation_at=stamp)
            else:
                event.update(activation_kind="new_confirmation")
            stop,reasons = risk(event)
            qualified = bool(event.get("input_eligible", True)) and _visible(event.get("available_at"), stamp) and all(not event.get(key) or _visible(event[key],stamp) for key in ("first_confirmed_at","origin_observed_at","observed_at","first_available_at"))
            directional = bool(event.get("confirmed",True)) and event.get("candidate_eligible") is not False if event["source"] != "ma" else qualified
            # Old risk flags are replaced by the explicit new risk policy;
            # source timing/input validity still survives.
            reasons = list(dict.fromkeys([*([] if qualified else ["source_input_ineligible"]),*reasons]))
            eligible = qualified and directional and not reasons
            event.update(direction_eligible=bool(qualified and directional), evidence_visible=qualified,
                input_eligible=qualified, candidate_eligible=eligible, selected_by_reference=eligible,
                reason_codes=reasons, selection_reason_codes=[], stop_price=stop, atr14=_finite(atr.iloc[i]),
                risk_fraction=parameters["risk_fraction"], model_lineage=event.get("model_lineage",{}))
            event["feature_snapshot_hash"] = row.get("input_hash") or stable_hash({k:row.get(k) for k in ("symbol","date","close","volume")})
            events.append(event)
            ledger[origin]=event
            if eligible and event["side"] == "BUY": activated.add(origin)
            if event["side"] != "BUY" or not qualified or not directional: return
            oid = stable_hash({"version":CANDIDATE_POLICY,"symbol":row["symbol"],"date":day,"policy":identity})
            event["opportunity_id"] = oid
            c = candidates.setdefault(day,{"opportunity_id":oid,"symbol":row["symbol"],"date":day,"available_at":stamp,
                "side":"BUY","sources":[],"source_event_ids":[],"source_eligibility":{},"source_selected_by_reference":{},
                "source_reasons":{},"source_stops":{},"feature_index":i,"risk_fraction":parameters["risk_fraction"],
                "reference_price":float(row["close"]),"atr14":_finite(atr.iloc[i]), **identity})
            source=event["source"]; c["sources"].append(source);c["source_event_ids"].append(event["event_id"])
            c["source_eligibility"][source] = c["source_eligibility"].get(source,False) or eligible
            c["source_selected_by_reference"][source] = c["source_eligibility"][source]
            c["source_reasons"][source]=reasons
            if eligible: c["source_stops"][source] = min(c["source_stops"].get(source,stop),stop)
            else: c["source_stops"].setdefault(source,stop)
        for e in by_day.get(day,[]): record(e)
        current = evaluations.get(day,{})
        for source in ("structure","wyckoff"):
            ev = current.get(source,{})
            for supported in ev.get("support_facts",[]):
                origin = supported.get("original_event_id") or supported.get("event_id")
                prior = ledger.get(origin)
                if not prior or prior["side"] != "BUY" or not prior.get("direction_eligible") or not supported.get("confirmed",True) or supported.get("candidate_eligible") is False or origin in activated or prior["date"] == day: continue
                stop,reasons=risk(supported)
                if not reasons and supported.get("input_eligible",True): record(supported,activation=True)
    for day,c in candidates.items():
        c["sources"]=sorted(set(c["sources"]),key=SOURCES.index);c["source_event_ids"]=sorted(set(c["source_event_ids"]))
        for source in SOURCES:
            c["source_eligibility"].setdefault(source,False);c["source_selected_by_reference"].setdefault(source,False)
            c["source_stops"].setdefault(source,None)
        c["candidate_eligible"]=any(c["source_eligibility"].values());c["selected_by_reference"]=c["candidate_eligible"]
        stops=[c["source_stops"][s] for s in SOURCES if c["source_eligibility"][s]]
        c["stop_price"]=min(stops) if stops else None
        route=(ma_routes or {}).get(day,{})
        c.update(l1_ma_entry_probability=route.get("probability"),l1_ma_entry_available=route.get("in_scope",False) and route.get("probability") is not None,
                 l1_ma_entry_target_contract_hash=route.get("model_identity",{}).get("target_contract_hash"))
    return {"version":SIGNAL_POINTS_VERSION,"config":parameters,"config_hash":stable_hash({"parameters":parameters,**identity}),
        "events":events,"candidates":list(candidates.values()),"ma_routes":list((ma_routes or {}).values()),"policy":identity,
        "coverage":{"rows":len(frame),"events":len(events),"opportunities":len(candidates),
            "risk_qualified":sum(c["candidate_eligible"] for c in candidates.values()),
            "reason_codes":dict(Counter(r for e in events for r in e["reason_codes"]))}}


def make_point_decisions(frame, points, *, source="union", config=None, start=None,
                         entry_probabilities=None, entry_threshold=None, include_unselected=False):
    """All factual plans plus source exits; no theoretical inventory is created.

    Neutral rows opt in to Broker's actual-inventory hold contract. They create
    no new entry and cannot cancel a held position. Only actual-state snapshots
    can establish whether a factual BUY/SELL intent results in a fill.
    """
    rows, dates = _records(frame)
    if source not in (*SOURCES, "union") or points.get("version") != SIGNAL_POINTS_VERSION:
        raise ValueError("unsupported source or signal-point contract")
    parameters = signal_point_parameters(config) if config is not None else dict(points["config"])
    settings = copy.deepcopy(strategy_config())
    overrides = (config or {}).get("strategy", config or {})
    def merge(base, changes):
        for key, value in changes.items():
            if key not in base:
                continue
            if isinstance(base[key], dict) and isinstance(value, dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
    merge(settings, overrides)
    settings = strategy_config(settings)
    max_weight = float(settings["execution"]["max_weight"])
    probabilities = _probabilities(entry_probabilities, len(rows), "entry")
    threshold = parameters["ma_entry_threshold"] if entry_threshold is None else _finite(entry_threshold)
    if threshold is None or not 0 < threshold < 1:
        raise ValueError("invalid point entry threshold")
    candidates = {point["date"]: point for point in points["candidates"] if source == "union" or source in point["sources"]}
    sells = {}
    for event in points["events"]:
        if event["side"] == "SELL" and event["selected_by_reference"] and (source == "union" or event["source"] == source):
            sells.setdefault(event["date"], []).append(event["event_id"])
    result = []
    for index, (row, day) in enumerate(zip(rows, dates, strict=True)):
        point = candidates.get(day)
        exit_ids = sells.get(day, [])
        before_start = start is not None and day < pd.Timestamp(start).date().isoformat()
        p = None if probabilities is None else _finite(probabilities[index])
        veto = probabilities is not None and (p is None or p < threshold)
        plan = None
        reason = "pre_start_warmup" if before_start else "opposing_source_exit" if exit_ids else "entry_probability_veto" if veto else None
        if point is not None:
            source_ok = point["candidate_eligible"] if source == "union" else point["source_eligibility"][source]
            selected = point["selected_by_reference"] if source == "union" else point["source_selected_by_reference"][source]
            stop = point["stop_price"] if source == "union" else point["source_stops"][source]
            atr, close = point["atr14"], float(row["close"])
            floor, ceiling, weight = None, None, 0.
            if not source_ok and reason is None:
                reason = "source_candidate_ineligible"
            elif not include_unselected and not selected and reason is None:
                reason = "source_reference_model_veto"
            if stop is not None and atr is not None and 0 < stop < close:
                floor = max(stop + .01, stop / (1 - parameters["min_risk_fraction"]))
                ceiling = min(close + parameters["max_next_open_gap_atr"] * atr,
                    stop / (1 - parameters["max_stop_distance"]))
                if floor > ceiling and reason is None:
                    reason = "risk_execution_band_invalid"
                weight = min(max_weight, parameters["risk_fraction"] * close / (close - stop))
            elif reason is None:
                reason = "risk_inputs_unavailable"
            plan = {"version": SIGNAL_POINTS_VERSION, "plan_id": stable_hash({"opportunity_id": point["opportunity_id"],
                "source": source, "config_hash": stable_hash(parameters)}), "opportunity_id": point["opportunity_id"],
                "symbol": str(row["symbol"]), "signal_date": day, "available_at": day + "T15:00:00+08:00",
                "created_at": day + "T15:00:00+08:00", "policy": "risk", "status": "active" if reason is None else "rejected",
                "entry_allowed": reason is None, "valid_for_sessions": 1, "stop_price": stop,
                "price_floor": floor, "price_ceiling": ceiling, "risk_fraction": parameters["risk_fraction"],
                "reference_price": close, "reference_price_source": "signal_day_close", "target_weight": weight,
                "atr14": atr, "sources": point["sources"] if source == "union" else [source],
                "source_event_ids": point["source_event_ids"], "parameters": dict(parameters),
                "selected_by_reference": selected,
                "reason_codes": [] if reason is None else [reason], "experimental": True}
        allowed = plan is not None and plan["entry_allowed"]
        sell = bool(exit_ids) and not before_start
        result.append({"date": day, "available_at": day + "T15:00:00+08:00", "symbol": str(row["symbol"]),
            "target_weight": plan["target_weight"] if allowed else 0.,
            "action": "SELL" if sell else "BUY" if allowed else "HOLD", "position_intent": "SELL" if sell else "BUY" if allowed else "HOLD",
            "candidate_action": "SELL" if sell else "BUY" if point is not None else "WAIT",
            "eligible": bool(allowed), "entry_gate_passed": bool(allowed), "entry_policy": "risk", "entry_plan": plan,
            "rule_sell": sell, "source_exit_event_ids": exit_ids, "source": source,
            "hold_current_position": not allowed and not sell,
            "opportunity_id": point["opportunity_id"] if point is not None else None,
            "probability": p, "threshold": threshold, "ml_filter_applied": probabilities is not None,
            "reference_price": _finite(row["close"]), "position_scope": "actual_broker_state_required"})
    return result
