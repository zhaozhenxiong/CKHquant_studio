"""Ordered, causal entry and actual-held prefix features for the new family."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS, STRUCTURE_COLUMNS
from my_strategy.services.czsc_wyckoff_features import W_INPUT_COLUMNS

FEATURE_VERSION = "czsc_trend_operation"
FULL_SCHEMA_VERSION = "czsc_trend_operation_calendar_bound"
NATIVE_COLUMNS = ("v4_structure_buy_support", "v4_structure_sell_support", "v4_w_demand_score", "v4_w_supply_score")
L1_OUTPUT_COLUMNS = ("trend_p_up", "trend_p_down", "trend_p_neutral", "trend_probability_available")
SOURCE_COLUMNS = tuple(f"source_{s}" for s in ("ma", "structure", "wyckoff"))
SOURCE_AVAILABILITY_COLUMNS = tuple(f"{s}_input_available" for s in ("ma", "structure", "wyckoff"))
EVENT_COLUMNS = tuple(f"{s}_{side}_event_age" for s in ("ma", "structure", "wyckoff") for side in ("buy", "sell"))
ENTRY_STATE_COLUMNS = ("candidate_cash_ratio", "candidate_account_return", "candidate_stop_distance",
    "candidate_risk_fraction", "candidate_target_weight", "source_count")
HOLDING_COLUMNS = ("holding_sessions", "holding_return", "stop_distance", "position_weight",
    "remaining_fraction", "cash_weight", "running_peak_return", "running_trough_return",
    "peak_giveback", "running_close_drawdown")
ENTRY_FEATURE_COLUMNS = tuple((*MA_TREND_COLUMNS, *STRUCTURE_COLUMNS, *W_INPUT_COLUMNS,
    *NATIVE_COLUMNS, *SOURCE_COLUMNS, *SOURCE_AVAILABILITY_COLUMNS, *EVENT_COLUMNS, *L1_OUTPUT_COLUMNS, *ENTRY_STATE_COLUMNS))
EXIT_FEATURE_COLUMNS = tuple((*ENTRY_FEATURE_COLUMNS, *HOLDING_COLUMNS))
L1_MA_ENTRY_COLUMNS = tuple(MA_TREND_COLUMNS)
L1_MA_EXIT_COLUMNS = tuple((*MA_TREND_COLUMNS, *HOLDING_COLUMNS))
ENTRY_SCHEMA_HASH = stable_hash({"version": FULL_SCHEMA_VERSION, "columns": ENTRY_FEATURE_COLUMNS, "scope": "signal_close_only",
    "candidate_risk_fraction": "known_signal_target_weight_times_close_minus_plan_stop_over_close;no_future_fill_or_outcome",
    "source_qualification":"unqualified_MA65_native_structure_or_W_groups_are_missing_values_not_zero"})
EXIT_SCHEMA_HASH = stable_hash({"version": FULL_SCHEMA_VERSION, "columns": EXIT_FEATURE_COLUMNS, "scope": "actual_holding_observed_prefix_only",
    "original_entry_schema_hash":ENTRY_SCHEMA_HASH})
MA_ENTRY_SCHEMA_HASH = stable_hash({"version": FEATURE_VERSION, "columns": L1_MA_ENTRY_COLUMNS, "scope": "independent_ma_entry"})
MA_EXIT_SCHEMA_HASH = stable_hash({"version": FEATURE_VERSION, "columns": L1_MA_EXIT_COLUMNS, "scope": "independent_ma_actual_holding"})


def causal_market_row(feature, date):
    """Future structure metadata can never expose that group's values."""
    def numeric(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan
    values = {k: numeric(feature.get(k, np.nan)) for k in (*MA_TREND_COLUMNS, *STRUCTURE_COLUMNS, *W_INPUT_COLUMNS, *NATIVE_COLUMNS)}
    if not feature.get('model_input_eligible',False):
        values.update({k:np.nan for k in MA_TREND_COLUMNS})
    cutoff = pd.Timestamp(str(date) + "T15:00:00+08:00")
    for fields, columns in ((("available_at",), MA_TREND_COLUMNS),
        (("structure_confirmed_at", "zone_confirmed_at", "weekly_available_at", "monthly_available_at"), STRUCTURE_COLUMNS),
        (("wyckoff_available_at", "wyckoff_observed_at", "wyckoff_range_formed_at"), W_INPUT_COLUMNS)):
        for field in fields:
            value = feature.get(field)
            if value is not None and not pd.isna(value):
                stamp = pd.to_datetime(value, errors="coerce", utc=True)
                if pd.isna(stamp) or stamp > cutoff:
                    values.update({k: np.nan for k in columns})
                    break
    if not feature.get("v4_structure_input_eligible",False):
        values.update({k:np.nan for k in (*STRUCTURE_COLUMNS,"v4_structure_buy_support","v4_structure_sell_support")})
    if not feature.get("v4_w_input_eligible",False):
        values.update({k:np.nan for k in (*W_INPUT_COLUMNS,"v4_w_demand_score","v4_w_supply_score")})
    values.update(ma_input_available=float(bool(feature.get("model_input_eligible", False)) and
        np.isfinite([values[k] for k in MA_TREND_COLUMNS]).all()),
        structure_input_available=float(bool(feature.get("v4_structure_input_eligible",False))
            and any(np.isfinite(values[k]) for k in STRUCTURE_COLUMNS)),
        wyckoff_input_available=float(bool(feature.get("v4_w_input_eligible",False)) and
            np.isfinite([values[k] for k in W_INPUT_COLUMNS if k != "w_event_age"]).all()))
    return values


def event_prefix_rows(features, events, market_dates):
    """Ages include only observed events, never future interval information."""
    lookup = {d: i for i, d in enumerate(market_dates)}
    by_day = {}
    for event in events:
        by_day.setdefault((event["symbol"], event["date"]), []).append(event)
    last, result = {}, []
    for row in features.to_dict("records"):
        day = pd.Timestamp(row["date"]).date().isoformat(); symbol = row["symbol"]
        cutoff = pd.Timestamp(day + "T15:00:00+08:00")
        for event in by_day.get((symbol, day), []):
            stamp = pd.to_datetime(event.get("available_at"), errors="coerce", utc=True)
            observed = event.get("observed_at")
            observed = pd.to_datetime(observed, errors="coerce", utc=True) if observed is not None else stamp
            invisible = {"structure_confirmation_not_visible", "wyckoff_metadata_not_visible"}
            visible = event.get("evidence_visible", True) and not invisible.intersection(event.get("reason_codes") or [])
            if (visible and pd.notna(stamp) and stamp <= cutoff and pd.notna(observed)
                    and observed <= cutoff and event.get("input_eligible")):
                last[(symbol, event["source"], event["side"].lower())] = lookup[day]
        values = {f"{source}_{side}_event_age": float(lookup[day] - last[(symbol, source, side)])
            if (symbol, source, side) in last else np.nan
            for source in ("ma", "structure", "wyckoff") for side in ("buy", "sell")}
        result.append(values)
    return result


def original_evidence_events(prepared):
    """Shared training/runtime original-clock facts; final B/S is not evidence."""
    facts={}
    for row in prepared['rows']:
        for event in row.get('candidate_evidence') or []:
            if event.get('source') not in {'ma','structure','wyckoff'}:continue
            key=(event.get('event_id'),event.get('date'),event.get('available_at'),event.get('side'))
            if key not in facts:facts[key]=event
    return sorted(facts.values(),key=lambda e:(e['date'],str(e.get('event_id')),e.get('source','')))


def entry_feature_row(feature, state, candidate, plan, *, initial_cash, event_prefix=None):
    day = state["date"]
    sources = candidate.get("sources", [])
    values = causal_market_row(feature, day)
    values.update({f"source_{s}": float(s in sources) for s in ("ma", "structure", "wyckoff")})
    values.update(event_prefix or {k: np.nan for k in EVENT_COLUMNS})
    trend = candidate.get("trend_probability") or feature.get("trend_probability") or {}
    probabilities = [trend.get("p_" + name, np.nan) for name in ("up", "down", "neutral")]
    available = bool(trend.get("model_available") and all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                     and abs(sum(probabilities) - 1) < 1e-6)
    values.update({"trend_p_" + name: probabilities[i] if available else np.nan for i, name in enumerate(("up", "down", "neutral"))})
    values["trend_probability_available"] = float(available)
    close = float(state["close"]); stop = float(plan.get("stop_price") or np.nan)
    values.update(candidate_cash_ratio=float(state["cash"]) / initial_cash,
        candidate_account_return=float(state["equity"]) / initial_cash - 1,
        candidate_stop_distance=close / stop - 1 if stop > 0 else np.nan,
        candidate_risk_fraction=float(plan.get("target_weight",candidate.get("target_weight",0.)))*(close-stop)/close if 0<stop<close else np.nan,
        candidate_target_weight=float(plan.get("target_weight", candidate.get("target_weight", 0.))),
        source_count=float(len(sources)))
    return {**values, "symbol": feature["symbol"], "date": day,
        "decision_at": day + "T15:00:00+08:00"}


def holding_feature_row(feature, snapshot, *, entry_values, prefix, market_index):
    """Prefix is updated at the current close, before any future execution."""
    values = causal_market_row(feature, snapshot["date"])
    values.update({k: entry_values.get(k, np.nan) for k in (*SOURCE_COLUMNS, *L1_OUTPUT_COLUMNS, *ENTRY_STATE_COLUMNS)})
    values.update({k: prefix.get(k, np.nan) for k in EVENT_COLUMNS})
    cost = float(snapshot["actual_cost_per_share"]); close = float(snapshot["close"])
    stop = snapshot.get("stop_price"); equity = float(snapshot["equity"])
    prefix["peak"] = max(prefix.get("peak", cost), float(snapshot["bar"]["high"]))
    prefix["trough"] = min(prefix.get("trough", cost), float(snapshot["bar"]["low"]))
    prefix["close_peak"] = max(prefix.get("close_peak", cost), close)
    values.update(holding_sessions=float(market_index[snapshot["date"]] - market_index[snapshot["entry_date"]] + 1),
        holding_return=close / cost - 1, stop_distance=close / float(stop) - 1 if stop else np.nan,
        position_weight=int(snapshot["shares"]) * close / equity,
        remaining_fraction=int(snapshot["shares"]) / int(snapshot["entry_shares"]),
        cash_weight=float(snapshot["cash"]) / equity, running_peak_return=prefix["peak"] / cost - 1,
        running_trough_return=prefix["trough"] / cost - 1, peak_giveback=(prefix["peak"] - close) / cost,
        running_close_drawdown=close / prefix["close_peak"] - 1)
    return {**values, "symbol": snapshot["symbol"], "date": snapshot["date"],
        "decision_at": snapshot["available_at"]}


def bind_frame(frame, *, head, contract, ma_only=False):
    is_exit = head == "exit"
    columns = (L1_MA_EXIT_COLUMNS if is_exit else L1_MA_ENTRY_COLUMNS) if ma_only else (EXIT_FEATURE_COLUMNS if is_exit else ENTRY_FEATURE_COLUMNS)
    schema = (MA_EXIT_SCHEMA_HASH if is_exit else MA_ENTRY_SCHEMA_HASH) if ma_only else (EXIT_SCHEMA_HASH if is_exit else ENTRY_SCHEMA_HASH)
    frame.attrs.update(feature_version=FEATURE_VERSION, schema_hash=schema, feature_columns=list(columns),
        feature_profile="ma_operation_v1" if ma_only else "high_return_operation_v1",
        label_version=contract["label_version"], label_contract=contract,
        data_version=stable_hash({"contract": contract["contract_hash"], "columns": columns}))
    return frame
