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


SIGNAL_POINTS_VERSION = "czsc_independent_signal_points_v0"
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


def build_signal_points(frame, config=None, *, ma_entry_probabilities=None,
                        ma_exit_probabilities=None, ma_lineages=None, ma_fallback_mask=None):
    """Keep per-source raw evidence; deduplicate eligible buys by symbol/day.

    Model outputs must have explicit forward lineage and a non-fixed-horizon
    target. A missing exit model stays unavailable; entry probabilities are
    never inverted into exit probabilities. ``feature_index`` is positional.
    """
    rows, dates = _records(frame)
    parameters = signal_point_parameters(config)
    count = len(rows)
    entry = _probabilities(ma_entry_probabilities, count, "MA entry")
    exits = _probabilities(ma_exit_probabilities, count, "MA exit")
    fallback = _fallback_mask(ma_fallback_mask, count)
    if ma_lineages is not None and len(ma_lineages) != count:
        raise ValueError("MA lineages must align with frame rows")
    lows = pd.to_numeric(frame.low, errors="coerce").shift(1).rolling(
        parameters["lookback_stop_bars"], min_periods=1).min().to_numpy()
    events, opportunities, seen, ma_routes = [], {}, set(), []
    previous_ma_buy, previous_ma_sell, previous_ret = False, False, None
    last_ma = {"BUY": -parameters["ma_cooldown_bars"], "SELL": -parameters["ma_cooldown_bars"]}
    coverage = {source: {"input_eligible_rows": 0, "events": 0, "buy_candidates": 0} for source in SOURCES}

    def add(index, source, side, kind, qualified, reasons, *, anchor=None, observed=None,
            identity_at=None, probability=None, target=None, lineage=None, dependencies=None):
        row, day = rows[index], dates[index]
        close_at = day + "T15:00:00+08:00"
        identifier = stable_hash({"version": SIGNAL_POINTS_VERSION, "symbol": str(row["symbol"]),
            "source": source, "side": side, "type": kind, "anchor": anchor,
            "identity_at": identity_at or close_at})
        stop, atr, risk_reasons = _risk(row, source, lows[index], parameters, close_at) if side == "BUY" else (None, None, [])
        reasons = list(dict.fromkeys([*reasons, *risk_reasons]))
        selection_reasons = [reason for reason in reasons if reason.startswith("ma_") and reason not in {"ma_input_ineligible"}]
        candidate_reasons = [reason for reason in reasons if reason not in selection_reasons]
        candidate_ok = qualified and not candidate_reasons
        selected = candidate_ok and not selection_reasons
        evidence_visible = not any(reason in {"structure_confirmation_not_visible", "wyckoff_metadata_not_visible"}
            for reason in reasons)
        if kind == "ma_independent_exit_evidence" and selection_reasons:
            # A model-only SELL has no independent causal rule underneath a
            # unavailable model output. It cannot become a visible event age.
            evidence_visible = False
        if not candidate_ok:
            identifier = stable_hash({"event_id": identifier, "rejected_at": close_at})
        if identifier in seen:
            return
        seen.add(identifier)
        event = {"event_id": identifier, "symbol": str(row["symbol"]), "date": day,
            "source": source, "side": side, "event_type": kind, "version": SIGNAL_POINTS_VERSION,
            "anchor_at": anchor, "observed_at": observed or close_at, "available_at": close_at,
            "signal_at": close_at, "feature_index": index, "input_eligible": bool(qualified),
            "evidence_visible": bool(evidence_visible),
            "candidate_eligible": bool(candidate_ok), "reason_codes": reasons,
            "selected_by_reference": bool(selected), "selection_reason_codes": selection_reasons,
            "stop_price": stop, "atr14": atr, "risk_fraction": parameters["risk_fraction"],
            "probability": probability, "probability_target": target, "model_lineage": lineage or {},
            "ma_entry_route": copy.deepcopy(ma_routes[index]) if source == "ma" else None,
            "dependencies": dependencies or [], "feature_snapshot_hash": row.get("input_hash") or stable_hash(
                {key: row.get(key) for key in ("symbol", "date", "open", "high", "low", "close", "volume", "amount", *MA_TREND_COLUMNS)}),
            "ma_exit_probability": None if exits is None else _finite(exits[index]),
            "ma_exit_status": "independent_model_output_requires_actual_holding" if exits is not None else "independent_exit_model_unavailable"}
        events.append(event)
        coverage[source]["events"] += 1
        if side != "BUY":
            return
        coverage[source]["buy_candidates"] += int(candidate_ok)
        opportunity_id = stable_hash({"version": SIGNAL_POINTS_VERSION, "symbol": event["symbol"], "date": day, "side": "BUY"})
        event["opportunity_id"] = opportunity_id
        candidate = opportunities.setdefault(opportunity_id, {"opportunity_id": opportunity_id,
            "symbol": event["symbol"], "date": day, "available_at": close_at, "side": "BUY",
            "sources": [], "source_event_ids": [], "source_eligibility": {}, "source_reasons": {},
            "source_selected_by_reference": {}, "source_stops": {},
            "feature_index": index, "risk_fraction": parameters["risk_fraction"], "atr14": atr,
            "reference_price": _finite(row["close"]), "stop_price": stop})
        candidate["sources"].append(source)
        candidate["source_event_ids"].append(identifier)
        candidate["source_eligibility"][source] = candidate["source_eligibility"].get(source, False) or candidate_ok
        candidate["source_selected_by_reference"][source] = candidate["source_selected_by_reference"].get(source, False) or selected
        candidate["source_reasons"][source] = reasons
        candidate["source_stops"][source] = stop
        # A union keeps the most conservative valid support risk, fixed ex ante.
        valid_stops = [item for item_source, item in candidate["source_stops"].items()
            if candidate["source_eligibility"][item_source] and item is not None]
        candidate["stop_price"] = min(valid_stops) if valid_stops else None

    for index, (row, day) in enumerate(zip(rows, dates, strict=True)):
        close_at = day + "T15:00:00+08:00"
        route = _entry_route(entry, fallback, ma_lineages, index, close_at, parameters)
        ma_routes.append({"date": day, "available_at": close_at, **route})
        ma_ok = _flag(row.get("model_input_eligible")) and all(_finite(row.get(key)) is not None for key in MA_TREND_COLUMNS)
        structure_ok = _flag(row.get("structure_input_eligible", row.get("input_eligible")))
        w_ok = _flag(row.get("wyckoff_input_eligible"))
        for source, eligible in zip(SOURCES, (ma_ok, structure_ok, w_ok), strict=True):
            coverage[source]["input_eligible_rows"] += int(eligible)
        ma_buy = ma_ok and row["close_above_ma20"] > 0 and row["ma20_slope5"] > 0 and (row["ma_alignment"] > 0 or row["ret_5"] > 0)
        ma_sell = ma_ok and (row["close_above_ma20"] <= 0 and row["ma20_slope5"] < 0 or row["ma5_ma10_gap"] < 0 and row["ret_5"] < 0)
        recovery = ma_buy and previous_ret is not None and previous_ret <= 0 and row["ret_1"] > 0
        if ma_buy and (not previous_ma_buy or recovery) and index - last_ma["BUY"] >= parameters["ma_cooldown_bars"]:
            reasons, p, lineage, target = route["reason_codes"], route["probability"], route["lineage"], route["target"]
            add(index, "ma", "BUY", "ma_trend_recovery" if recovery else "ma_trend_onset", ma_ok, reasons,
                probability=p, target=target, lineage=lineage, dependencies=["trailing_sma_5_10_20_60"])
            last_ma["BUY"] = index
        if ma_sell and not previous_ma_sell and index - last_ma["SELL"] >= parameters["ma_cooldown_bars"]:
            add(index, "ma", "SELL", "ma_trend_failure", ma_ok, [])
            last_ma["SELL"] = index
        if exits is not None and ma_ok and _finite(exits[index]) is not None and exits[index] >= parameters["ma_exit_threshold"]:
            lineage, reasons, target = _model_lineage(ma_lineages, index, "exit", close_at)
            add(index, "ma", "SELL", "ma_independent_exit_evidence", ma_ok, reasons,
                probability=float(exits[index]), target=target, lineage=lineage)
        previous_ma_buy, previous_ma_sell = ma_buy, ma_sell
        previous_ret = _finite(row.get("ret_1")) if ma_ok else None

        for name in ("bs1", "bs2", "bs3"):
            kind = _text(row.get("confirmed_native_" + name))
            if kind not in {"一买", "二买", "三买", "一卖", "二卖", "三卖"}:
                continue
            confirmed = row.get("confirmed_native_" + name + "_at") or row.get("point_confirmed_at")
            reasons = [] if structure_ok else ["structure_input_ineligible"]
            if not _visible(confirmed, close_at) or not _visible(row.get("structure_confirmed_at"), close_at):
                reasons.append("structure_confirmation_not_visible")
            add(index, "structure", "BUY" if kind.endswith("买") else "SELL", "native_" + name + "_" + kind,
                structure_ok, reasons, anchor=_text(row.get("point_anchor")), observed=_text(confirmed),
                identity_at=_text(confirmed), dependencies=["native_ma_assisted_definition"] if name in {"bs2", "bs3"} else ["native_definition"])
        # Native metadata can be absent in older snapshots; do not invent a point
        # from combined rule_buy or backfill an unconfirmed display structure.
        for side in ("BUY", "SELL"):
            if not _flag(row.get("wyckoff_rule_" + side.lower())):
                continue
            reasons = [] if w_ok else ["wyckoff_input_ineligible"]
            for key in ("wyckoff_available_at", "wyckoff_observed_at", "wyckoff_range_formed_at"):
                if not _visible(row.get(key), close_at):
                    reasons.append("wyckoff_metadata_not_visible")
                    break
            add(index, "wyckoff", side, _text(row.get("wyckoff_event")) or "wyckoff_unknown",
                w_ok, reasons, anchor=_text(row.get("wyckoff_anchor_at")),
                observed=_text(row.get("wyckoff_observed_at")), identity_at=_text(row.get("wyckoff_observed_at")),
                dependencies=["causal_prior_range", "price_volume_event_state"])

    candidates = list(opportunities.values())
    for candidate in candidates:
        candidate["sources"] = sorted(set(candidate["sources"]), key=SOURCES.index)
        candidate["source_event_ids"] = sorted(set(candidate["source_event_ids"]))
        for source in SOURCES:
            candidate["source_eligibility"].setdefault(source, False)
            candidate["source_selected_by_reference"].setdefault(source, False)
        candidate["candidate_eligible"] = any(candidate["source_eligibility"].values())
        candidate["selected_by_reference"] = any(candidate["source_selected_by_reference"].values())
        # Attach today's forward MA score to every source's opportunity. Its
        # presence is evidence, independent of whether MA emitted a BUY today
        # or whether the reference threshold selected that MA score.
        route = ma_routes[candidate["feature_index"]]
        score_available = route["mode"] == "forward_model" and route["probability"] is not None and not any(
            reason != "ma_entry_probability_veto" for reason in route["reason_codes"])
        candidate["l1_ma_entry_probability"] = route["probability"] if score_available else None
        candidate["l1_ma_entry_available"] = bool(score_available)
        candidate["l1_ma_entry_target_contract_hash"] = route["lineage"].get("target_contract_hash") if score_available else None
    return {"version": SIGNAL_POINTS_VERSION, "config": parameters, "config_hash": stable_hash(parameters),
        "events": events, "candidates": candidates, "ma_routes": ma_routes, "coverage": {"rows": count, "sources": coverage,
            "opportunities": len(candidates), "unavailable_reasons": dict(Counter(reason for event in events for reason in event["reason_codes"]))}}


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
