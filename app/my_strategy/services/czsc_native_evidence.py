"""Independent, causal structural and current price/volume evidence for the current pipeline.

This module does not alter other feature or trading contracts. Native
patterns use only finished pens. A pattern's first observation is immutable;
daily risk eligibility is deliberately left to the frozen current policy.
Continuous W scores are rule strength, never a probability or a fresh trigger.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import engine_contract
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_analysis import CzscSession, _pen_key, strategy_config
from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES, _json
from my_strategy.services.czsc_market_features import _validate_input
from my_strategy.services.czsc_price_volume_features import prepare_wyckoff_inputs, get_wyckoff_profile, WYCKOFF_SCHEMA_HASH
from my_strategy.services.czsc_price_volume_events import event_config

NATIVE_EVIDENCE_VERSION = "native_price_volume_evidence"
STRUCTURE_RULE_VERSION = "confirmed_structure_rules"
W_RULE_VERSION = "current_price_volume_rules"
DEFAULT_NATIVE_CONFIG = {
    "risk_buffer_atr": .25, "event_score_decay_bars": 20,
    "continuous_direction_threshold": .30, "continuous_direction_margin": .10,
    "structure_zone_strength": .50, "structure_pattern_strength": 1.0,
}
NATIVE_SEMANTICS = {
    "version": NATIVE_EVIDENCE_VERSION, "engine_contract": engine_contract(),
    "structure": {"engine": {"min_bi_len": 6, "max_bi_num": 200, "higher_period_closure": "next_period_observed"},
        "rules": ["cxt_first_buy_V221126", "cxt_first_sell_V221126",
                              "cxt_second_bs_V240524", "cxt_third_buy_V230228"],
        "finished_di": "len(bi_list)-len(finished_bis)+1",
        "third_sell": {"rule": "confirmed_third_sell_price_mirror",
            "basis": "exact_price_reflection_of_cxt_third_buy_V230228_geometry",
            "input": "finished_pens_only_no_MA;minimum12_finished;counts14_12_10_8_6",
            "key_pens": "even_offset_down_pens_with_strictly_lower_lows",
            "break": "last_high<max_key_low<min_key_high",
            "price": "last_high>max_all_high-1.618*mean_key_power_price",
            "third_party_source_modified": False},
        "identity": "rule,label,all_referenced_finished_pens,symbol;first_observed_immutable",
        "support": "buy_last_finished_pen_low;sell_last_finished_pen_high",
        "zone": "latest_valid_first_three_finished_pens;forming_zone_does_not_hide_previous",
        "calendar": "explicit_verified_market_dates_required;every_dependency_span_expected_sessions_equal_observed_raw_rows;exclude_gap_before_first_dependency_bar;future_calendar_extension_not_policy_hash",
        "auxiliary_ma": "last5_finished_pens_plus34_prior_raw_bar_MA_lookback_same_full_calendar_and_source_checks;does_not_receive_independent_structure_weight",
        "invalidation": "first_close_beyond_frozen_boundary_or_new_confirmed_opposite_pattern;permanent"},
    "continuous_w": {"input": "OHLCV_only;no_MA_CZSC_or_final_arbitration",
        "baseline": "prior14_mean_true_range;prior20_mean_volume;exclude_current",
        "buy": ".35*positive_close_location+.30*positive_price_change_ATR+.20*positive_body_ATR+.15*lower_shadow_fraction",
        "sell": ".35*negative_close_location+.30*negative_price_change_ATR+.20*negative_body_ATR+.15*upper_shadow_fraction",
        "participation": "multiply_by_.5+.5*clip(relative_volume,0,2)/2",
        "recovery": "add_.15_when_close_exceeds_previous_high_after_down_bar_and_relative_volume>=.8;mirror_supply",
        "contraction": "add_.10_on_down_bar_relvol<=.8_close_location>=.5_low_holds_prior20_low;mirror_supply",
        "range": "clip_both_independent_channels_to_0..1",
        "interpretation": "contemporaneous_rule_strength_not_profit_probability_not_fresh_candidate"},
    "event_w": {"event_input_schema_hash": WYCKOFF_SCHEMA_HASH,
        "origin": "causal_event_machine_full_event_export",
        "boundary": "Spring_Test_original_range_support;SOS_LPS_original_breakout_support;SOW_LPSY_original_breakdown_resistance;Upthrust_original_resistance",
        "channels": "independent_demand_and_supply_event_ledgers",
        "invalidation": "first_close_beyond_original_boundary_or_input_quality_gap;never_resurrect",
        "stop_price": "raw_event_stop_without_ATR_buffer",
        "risk_stop_price": "raw_stop_minus_BUY_plus_SELL_.25_origin_prior_ATR_frozen_once",
        "snapshot": "current_available_at_separate_from_origin_observed_at"},
    "config": DEFAULT_NATIVE_CONFIG,
}
NATIVE_SCHEMA_HASH = stable_hash(NATIVE_SEMANTICS)
_RULES = (("first_buy", "cxt_first_buy_V221126"),
          ("first_sell", "cxt_first_sell_V221126"),
          ("second_bs", "cxt_second_bs_V240524"),
          ("third_buy", "cxt_third_buy_V230228"),
          ("third_sell", "confirmed_third_sell_price_mirror"))
_SIDES = {"一买": "BUY", "二买": "BUY", "三买": "BUY", "一卖": "SELL", "二卖": "SELL", "三卖": "SELL"}
_W_EVENTS = {"Spring", "Test", "SOS", "LPS", "Upthrust", "SOW", "LPSY"}
_W_BUYS = {"Spring", "Test", "SOS", "LPS"}
_W_WEIGHTS = {"Spring": .5, "Test": 1., "SOS": .75, "LPS": 1., "Upthrust": .5, "SOW": .75, "LPSY": 1.}


def _pure_third_sell_signal(analysis: Any, *, finished=None) -> tuple[str, int | None]:
    """Exact price-reflected native third-buy geometry on finished pens.

    The frozen vendor only exposes a pure third BUY. Its third-BS alternatives
    also read moving averages, so they cannot spend independent structure
    weight. This owned rule mirrors each inequality and direction without
    changing vendor code or using a forming pen or an eventual chart anchor.
    """
    finished = analysis.finished_bis if finished is None else finished
    # Native's len(bi_list) >= di+11, with the delivered finished-di mapping.
    if len(finished) < 12:
        return "其他", None
    for length in (14, 12, 10, 8, 6):
        if len(finished) < length:
            continue
        pens = finished[-length:]
        if str(pens[-1].direction) != "向上" or pens[0].direction == pens[-1].direction:
            continue
        keys = [pen for offset, pen in enumerate(pens[:-2]) if offset % 2 == 0
                and (offset == 0 or float(pen.low) < float(pens[offset - 2].low))]
        if len(keys) < 2:
            continue
        last_high = float(pens[-1].high)
        key_low = max(float(pen.low) for pen in keys)
        key_high = min(float(pen.high) for pen in keys)
        power = sum(float(pen.power_price) for pen in keys) / len(keys)
        if last_high < key_low < key_high and last_high > max(float(pen.high) for pen in pens) - 1.618 * power:
            return "三卖", length
    return "其他", None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _day(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _settings(config: dict | None) -> dict:
    result = dict(DEFAULT_NATIVE_CONFIG)
    incoming = (config or {}).get("native_evidence", {})
    for key in result:
        if key in incoming:
            result[key] = incoming[key]
    for key, value in result.items():
        if _number(value) is None or float(value) < 0:
            raise ValueError("invalid native evidence setting: " + key)
    if int(result["event_score_decay_bars"]) != result["event_score_decay_bars"] or result["event_score_decay_bars"] < 1:
        raise ValueError("event_score_decay_bars must be a positive integer")
    if any(result[key] > 1 for key in ("continuous_direction_threshold", "continuous_direction_margin",
                                      "structure_zone_strength", "structure_pattern_strength")):
        raise ValueError("native rule strength settings must be in 0..1")
    return result


def _prior_metrics(bars: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = [bars[key] for key in ("close", "high", "low", "volume")]
    previous = close.shift(1)
    tr = pd.concat([high - low, (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    return pd.DataFrame({"atr": tr.shift(1).rolling(14, min_periods=14).mean(),
                         "relvol": volume / volume.shift(1).rolling(20, min_periods=20).mean().replace(0, np.nan),
                         "low20": low.shift(1).rolling(20, min_periods=20).min(),
                         "high20": high.shift(1).rolling(20, min_periods=20).max()}, index=bars.index)


class _VerifiedCalendar:
    """Verify a span in O(1), consulting only its <= decision-day endpoints."""
    def __init__(self, bars: pd.DataFrame, market_dates: list[str] | None):
        self.dates = [_day(value) for value in bars.date]
        self.provided = market_dates is not None
        calendar = [] if market_dates is None else [_day(value) for value in market_dates]
        if calendar != sorted(set(calendar)):
            raise ValueError("native calendar must be sorted and unique")
        if self.provided and not calendar:
            raise ValueError("native verified calendar cannot be empty")
        self.market_index = {day: i for i, day in enumerate(calendar)}
        self.known = np.array([day in self.market_index for day in self.dates], dtype=bool)
        self.unknown_prefix = np.r_[0, np.cumsum(~self.known)]
        self.provenance = {"provided": self.provided,
            "calendar_prefix_hash": stable_hash([day for day in calendar if not self.dates or day <= self.dates[-1]]),
            "calendar_prefix_end": max([day for day in calendar if not self.dates or day <= self.dates[-1]], default=None),
            "future_calendar_entries_used_for_historical_qualification": False}

    def span(self, start: int, end: int) -> dict:
        observed = end - start + 1
        known = bool(self.provided and self.unknown_prefix[end + 1] == self.unknown_prefix[start])
        expected = self.market_index[self.dates[end]] - self.market_index[self.dates[start]] + 1 if known else None
        return {"dependency_start": self.dates[start], "dependency_end": self.dates[end],
                "dependency_expected_sessions": expected, "dependency_observed_sessions": observed,
                "dependency_calendar_complete": bool(known and expected == observed)}


def _continuous_scores(bars: pd.DataFrame, metrics: pd.DataFrame, eligible: np.ndarray) -> pd.DataFrame:
    """Deterministic causal rule scores. Missing input remains NaN, not neutral 0."""
    high, low, op, close = [bars[key].to_numpy(dtype=float) for key in ("high", "low", "open", "close")]
    atr, rel = [metrics[key].to_numpy(dtype=float) for key in ("atr", "relvol")]
    spread = high - low
    previous = np.r_[np.nan, close[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        loc = np.divide(2 * close - high - low, spread, out=np.zeros(len(bars)), where=spread > 0)
        change, body = np.clip((close - previous) / atr, -1, 1), np.clip((close - op) / atr, -1, 1)
        lower = np.divide(np.minimum(op, close) - low, spread, out=np.zeros(len(bars)), where=spread > 0)
        upper = np.divide(high - np.maximum(op, close), spread, out=np.zeros(len(bars)), where=spread > 0)
    participation = .5 + .5 * np.clip(rel, 0, 2) / 2
    demand = (.35 * np.maximum(loc, 0) + .30 * np.maximum(change, 0) + .20 * np.maximum(body, 0) + .15 * lower) * participation
    supply = (.35 * np.maximum(-loc, 0) + .30 * np.maximum(-change, 0) + .20 * np.maximum(-body, 0) + .15 * upper) * participation
    prev_high, prev_low, prev_op = np.r_[np.nan, high[:-1]], np.r_[np.nan, low[:-1]], np.r_[np.nan, op[:-1]]
    demand += .15 * ((close > prev_high) & (previous < prev_op) & (rel >= .8))
    supply += .15 * ((close < prev_low) & (previous > prev_op) & (rel >= .8))
    demand += .10 * ((close < previous) & (rel <= .8) & (loc >= 0) & (low >= metrics["low20"].to_numpy()))
    supply += .10 * ((close > previous) & (rel <= .8) & (loc <= 0) & (high <= metrics["high20"].to_numpy()))
    valid = eligible & np.isfinite(atr) & (atr > 0) & np.isfinite(rel) & (rel > 0)
    return pd.DataFrame({"demand": np.where(valid, np.clip(demand, 0, 1), np.nan),
                         "supply": np.where(valid, np.clip(supply, 0, 1), np.nan),
                         "eligible": valid, "atr": atr, "relative_volume": rel,
                         "close_location": loc, "price_change_atr": change, "body_atr": body}, index=bars.index)


def _fact(symbol: str, date: str, source: str, side: str, event_type: str,
          pattern: dict, anchor: str, stop: float | None, boundary: float | None,
          atr: float | None, settings: dict, *, candidate: bool, confirmed: bool,
          eligible: bool, reference: float, extra: dict | None = None) -> dict:
    available = date + "T15:00:00+08:00"
    pattern_id = source + ":native:" + stable_hash({"symbol": symbol, **pattern})
    risk = None if stop is None or atr is None else stop + (-1 if side == "BUY" else 1) * settings["risk_buffer_atr"] * atr
    return {"symbol": symbol, "date": date, "source": source, "side": side, "event_type": event_type,
            "event": event_type, "event_id": pattern_id + ":confirmed:" + available,
            "original_event_id": pattern_id + ":confirmed:" + available,
            "pattern_id": pattern_id, "first_confirmed_at": available if confirmed else None,
            "first_available_at": available, "origin_observed_at": available,
            "observed_at": available, "available_at": available, "confirmed_at": available if confirmed else None,
            "anchor_at": anchor, "time": date, "support_low": boundary if side == "BUY" else None,
            "resistance_high": boundary if side == "SELL" else None, "invalidation_price": boundary,
            "stop_price": stop, "risk_stop_price": risk, "risk_atr_at_origin": atr,
            "confirmed": confirmed, "is_confirmed": confirmed, "candidate_eligible": candidate and eligible,
            "input_eligible": eligible, "evidence_visible": eligible,
            "candidate_direction_valid": candidate and eligible,
            "reference_price": reference, "reference_price_date": date,
            "reference_price_source": "signal_day_close", "dependencies": [],
            "state": "confirmed" if confirmed else "pending", **(extra or {})}


def _invalidate(active: dict[str, dict], invalidated: dict[str, dict], ids: list[str],
                available: str, reason: str) -> list[dict]:
    result = []
    for identity in ids:
        item = active.pop(identity)
        record = {"event_id": identity, "pattern_id": item["pattern_id"], "side": item["side"],
                  "invalidated_at": available, "reason": reason, "invalidation_price": item.get("invalidation_price")}
        invalidated[identity] = record
        result.append(record)
    return result


def _zone_snapshot(zone: Any, available: str, seen: dict[str, dict], finished_keys: set[tuple]) -> dict | None:
    # A native bis getter clones the complete sequence; read it once per zone.
    bis = zone.bis
    if len(bis) < 3 or not zone.is_valid():
        return None
    keys = [_pen_key(pen) for pen in bis[:3]]
    if not all(key in finished_keys for key in keys):
        return None
    identity = "zone:native:" + stable_hash({"pens": keys, "low": float(zone.zd), "high": float(zone.zg)})
    if identity in seen:
        return seen[identity]
    value = {"zone_id": identity, "first_confirmed_at": available,
        "available_at": available, "anchor_at": _day(bis[2].edt), "support_low": float(zone.zd),
        "resistance_high": float(zone.zg), "dependencies": [list(key) for key in keys], "confirmed": True,
        "dependency_start_index": min(int(raw.id) for bi in bis[:3] for raw in bi.raw_bars)}
    seen[identity] = value
    return value


def _structure_evidence(bars: pd.DataFrame, raw_quality: np.ndarray, metrics: pd.DataFrame,
                        settings: dict, czsc_config: dict | None,
                        calendar: _VerifiedCalendar, *, resume_state=None, state_sink=None) -> tuple[list, list, list, list]:
    from my_strategy.services.czsc_incremental_state import dump_session, restore_session
    session = restore_session(resume_state['session']) if resume_state else CzscSession(
        str(bars.iloc[0].symbol), strategy_config(czsc_config), len(bars),
        collect_confirmations=False, create_position=False)
    facts: list[dict] = []; evaluations: list[dict] = []; daily_events: list[list] = []; auxiliary: list[dict] = []
    seen: dict[str, dict] = {}; zone_seen: dict[str, dict] = {}; active: dict[str, dict] = {}; invalidated: dict[str, dict] = {}
    start_index=0
    if resume_state:
        start_index=resume_state['rows']
        if not 0 < start_index < len(bars): raise ValueError('native continuation must append observed bars')
        seen,zone_seen,active,invalidated=[copy.deepcopy(resume_state[key]) for key in ('seen','zone_seen','active','invalidated')]
    source_bad = ~bars["source"].fillna("").astype(str).isin(VERIFIED_SOURCES).to_numpy() if "source" in bars else np.ones(len(bars), bool)
    trade_bad = ~pd.to_numeric(bars["has_trade_price"], errors="coerce").eq(1).to_numpy() if "has_trade_price" in bars else np.ones(len(bars), bool)
    quantity = bars[["volume", "amount"]].to_numpy(dtype=float)
    bad = source_bad | trade_bad | ~np.isfinite(quantity).all(axis=1) | (quantity < 0).any(axis=1) | ((quantity[:, 0] == 0) != (quantity[:, 1] == 0))
    bad_prefix = np.r_[0, np.cumsum(bad)]
    engine = bars.copy()
    for key in ("volume", "amount"):
        engine[key] = engine[key].where(np.isfinite(engine[key]) & (engine[key] >= 0), 0.)
    for i, row in enumerate(engine.itertuples(index=False)):
        if i < start_index: continue
        date, close = _day(row.date), float(row.close); available = date + "T15:00:00+08:00"
        session.advance_structure(row)
        analysis = session.kas["日线"]; finished = analysis.finished_bis
        pen_keys = [_pen_key(pen) for pen in finished]
        finished_keys = set(pen_keys)
        # All entries below belong to this exact causal observation. Reading
        # dependency ids once avoids repeated native RawBar sequence exports.
        dependency_ids = {}
        def ids(pen):
            key = id(pen)
            if key not in dependency_ids:
                dependency_ids[key] = tuple(int(raw.id) for raw in pen.raw_bars)
            return dependency_ids[key]
        di = len(analysis.bi_list) - len(finished) + 1
        new: list[dict] = []; current_labels = {}; dependency_start = i
        zones = [_zone_snapshot(z, available, zone_seen, finished_keys) for z in analysis.zs_list]
        complete_zones = []
        for item in zones:
            if item is None:continue
            first = item["dependency_start_index"]
            span = calendar.span(first, i)
            if span["dependency_calendar_complete"] and bad_prefix[i + 1] == bad_prefix[first]:
                complete_zones.append({**copy.deepcopy(item), **span})
        zone = next(iter(reversed(complete_zones)), None)
        # Geometry is causal and retained even when native's newest zone is forming.
        if zone:
            dependency_start = min(dependency_start, zone["dependency_start_index"])
        aux_start = max(0, min((raw_id for pen in finished[-5:] for raw_id in ids(pen)), default=i) - 34)
        aux_span = calendar.span(aux_start, i)
        aux_eligible = bool(raw_quality[i] and len(finished) >= 5 and
                            aux_span["dependency_calendar_complete"] and bad_prefix[i + 1] == bad_prefix[aux_start])
        auxiliary.append({"date": date, "available_at": available, "input_eligible": aux_eligible, **aux_span,
            "reason_codes": [] if aux_eligible else ["auxiliary_full_dependency_calendar_or_source_ineligible"]})
        invalidations = _invalidate(active, invalidated,
            [key for key, value in active.items() if value.get("invalidation_price") is not None and
             (close < value["invalidation_price"] if value["side"] == "BUY" else close > value["invalidation_price"])],
            available, "close_broke_frozen_structure_boundary")
        if not raw_quality[i]:
            invalidations.extend(_invalidate(active, invalidated, list(active), available, "input_quality_gap"))
        for short, name in _RULES:
            if short == "third_sell":
                label, count = _pure_third_sell_signal(analysis, finished=finished)
                signal = None
            else:
                signal = session.czsc._native.call_signal(name, analysis, {"di": di, **({"w": 9, "t": 2} if short == "second_bs" else {})})[0] if finished else None
                label, count = str(signal.v1) if signal is not None else "其他", None
            current_labels[short] = label
            if label not in _SIDES:
                continue
            match = re.search(r"(\d+)笔", str(signal.v2)) if signal is not None else None
            n = count or (int(match.group(1)) if match else 9)
            dependencies = finished[-n:]
            start = min(raw_id for pen in dependencies for raw_id in ids(pen))
            dependency_start = min(dependency_start, start)
            span = calendar.span(start, i)
            eligible = bool(raw_quality[i] and span["dependency_calendar_complete"] and bad_prefix[i + 1] == bad_prefix[start])
            pattern = {"version": STRUCTURE_RULE_VERSION, "rule": name, "label": label,
                       "parameters": {"w": 9, "t": 2} if short == "second_bs" else {},
                       "pens": pen_keys[-n:]}
            pattern_id = "structure:native:" + stable_hash({"symbol": str(row.symbol), **pattern})
            if pattern_id in seen:
                continue
            side = _SIDES[label]; last = finished[-1]
            boundary = float(last.low) if side == "BUY" else float(last.high)
            fact = _fact(str(row.symbol), date, "structure", side, "pure_" + short + "_" + side.lower(), pattern,
                         _day(last.fx_b.dt), boundary, boundary, _number(metrics.iloc[i].atr), settings,
                         candidate=True, confirmed=True, eligible=eligible, reference=close,
                         extra={"rule_name": name, "label": label, "family": "structure_native",
                                "version": STRUCTURE_RULE_VERSION, "dependency_start": _day(bars.iloc[start].date),
                                "dependencies": [list(key) for key in pen_keys[-n:]],
                                "dependency_bars": i - start + 1, **span,
                                "reason_codes": [] if eligible else ["native_structure_input_ineligible", *([] if span["dependency_calendar_complete"] else ["native_structure_dependency_calendar_incomplete"])]})
            seen[pattern_id] = fact; facts.append(fact); new.append(fact)
            if eligible and (close >= boundary if side == "BUY" else close <= boundary):
                active[fact["event_id"]] = fact
        # An actually confirmed opposite pattern ends previous support. Same-day
        # opposing confirmations retain an explicit conflict rather than order bias.
        new_sides = {fact["side"] for fact in new if fact["input_eligible"]}
        if len(new_sides) == 1:
            side = next(iter(new_sides))
            invalidations.extend(_invalidate(active, invalidated,
                [key for key, value in active.items() if value["side"] != side], available, "new_confirmed_opposite_pattern"))
        # A broken BUY support creates one directional research SELL fact, never
        # a backdated execution. Opposite native SELL facts remain independently visible.
        for broken in invalidations:
            if broken["side"] != "BUY" or broken["reason"] != "close_broke_frozen_structure_boundary" or not raw_quality[i]:
                continue
            pattern = {"version": STRUCTURE_RULE_VERSION, "invalidated_event_id": broken["event_id"]}
            stop = float(finished[-1].high) if finished else None
            fact = _fact(str(row.symbol), date, "structure", "SELL", "structure_invalidation", pattern,
                         date, stop, stop, _number(metrics.iloc[i].atr), settings, candidate=True,
                         confirmed=True, eligible=True, reference=close,
                         extra={"rule_name": "confirmed_support_close_break", "family": "structure_native",
                                "version": STRUCTURE_RULE_VERSION, "invalidates_event_id": broken["event_id"],
                                "reason_codes": [broken["reason"]]})
            facts.append(fact); new.append(fact)
            if stop is not None and close <= stop:
                active[fact["event_id"]] = fact
        # Active origins cannot have their original full dependency silently
        # replaced by a shorter current neutral-pattern snapshot.
        for fact in active.values():
            if fact.get("dependency_start"):
                dependency_start = min(dependency_start, int(bars.date.searchsorted(pd.Timestamp(fact["dependency_start"]))))
        current_span = calendar.span(dependency_start, i)
        eligible = bool(raw_quality[i] and current_span["dependency_calendar_complete"] and bad_prefix[i + 1] == bad_prefix[dependency_start])
        supports = copy.deepcopy(list(active.values())) if eligible else []
        buys = [fact for fact in supports if fact["side"] == "BUY"]
        sells = [fact for fact in supports if fact["side"] == "SELL"]
        zone_buy = bool(zone and close >= zone["resistance_high"])
        zone_sell = bool(zone and close <= zone["support_low"])
        buy_strength = settings["structure_pattern_strength"] if buys else settings["structure_zone_strength"] if zone_buy else 0.
        sell_strength = settings["structure_pattern_strength"] if sells else settings["structure_zone_strength"] if zone_sell else 0.
        support = max([fact["support_low"] for fact in buys if fact["support_low"] is not None] +
                      ([zone["support_low"]] if zone and close >= zone["support_low"] else []), default=None)
        resistance = min([fact["resistance_high"] for fact in sells if fact["resistance_high"] is not None] +
                         ([zone["resistance_high"]] if zone and close <= zone["resistance_high"] else []), default=None)
        state = {"close": close, "support_valid": bool(buys or zone_buy), "resistance_valid": bool(sells or zone_sell),
                 "invalidated": bool(invalidations), "support_floor": support, "resistance_ceiling": resistance,
                 "structure_confirmed_at": max([fact["first_confirmed_at"] for fact in supports] +
                     ([zone["first_confirmed_at"]] if zone else []), default=None),
                 "buy_score": float(buy_strength) if eligible else None, "sell_score": float(sell_strength) if eligible else None,
                 "current_valid_zone": copy.deepcopy(zone), "native_labels": current_labels}
        state.update(current_span)
        evaluations.append({"source_id": "structure", "family": "structure_native", "version": STRUCTURE_RULE_VERSION,
            "date": date, "available_at": available, "snapshot_at": available, "input_eligible": eligible,
            "buy_valid": bool(eligible and buy_strength > 0), "sell_valid": bool(eligible and sell_strength > 0),
            "direction_state": "CONFLICT" if eligible and buy_strength > 0 and sell_strength > 0 else "BUY" if eligible and buy_strength > 0 else "SELL" if eligible and sell_strength > 0 else "ABSTAIN",
            "current_state": state, "support_facts": supports,
            "active_buy_event_ids": [fact["event_id"] for fact in buys], "active_sell_event_ids": [fact["event_id"] for fact in sells],
            "invalidated_event_ids": list(invalidated), "invalidations_at_day": invalidations,
            "reason_codes": [] if eligible else ["native_structure_input_ineligible", *([] if current_span["dependency_calendar_complete"] else ["native_structure_dependency_calendar_incomplete"])]})
        daily_events.append(copy.deepcopy(new))
    if state_sink is not None:
        state_sink.update(rows=len(bars), session=dump_session(session), seen=seen,
            zone_seen=zone_seen, active=active, invalidated=invalidated)
    return facts, evaluations, daily_events, auxiliary


def _wyckoff_evidence(bars: pd.DataFrame, legacy: pd.DataFrame, metrics: pd.DataFrame,
                      scores: pd.DataFrame, settings: dict, w_config: dict | None = None) -> tuple[list, list, list]:
    facts: list[dict] = []; evaluations: list[dict] = []; daily_events: list[list] = []
    pending: dict[str, dict] = {}; active: dict[str, dict] = {}; invalidated: dict[str, dict] = {}
    origin_index: dict[str, int] = {}
    event_settings = event_config(w_config)
    for i, row in enumerate(legacy.to_dict(orient="records")):
        date = _day(bars.iloc[i].date); available = date + "T15:00:00+08:00"; close = float(bars.iloc[i].close)
        eligible = bool(row["wyckoff_input_eligible"]); atr = _number(metrics.iloc[i].atr)
        kind = str(row["wyckoff_event"]); anchor = row.get("wyckoff_anchor_at") or date; new = []
        invalidations = _invalidate(active, invalidated,
            [key for key, value in active.items() if value.get("invalidation_price") is not None and
             (close < value["invalidation_price"] if value["side"] == "BUY" else close > value["invalidation_price"])],
            available, "close_broke_frozen_wyckoff_boundary")
        if not eligible:
            invalidations.extend(_invalidate(active, invalidated, list(active), available, "input_quality_gap"))
            pending.clear()
        for key in list(active):
            item = active[key]
            max_age = event_settings["test_max_bars"] if item["event_type"] == "Spring" else event_settings["retest_max_bars"]
            if not item["confirmed"] and i - origin_index[key] > max_age:
                invalidations.extend(_invalidate(active, invalidated, [key], available, "pending_confirmation_expired"))
        if kind in _W_EVENTS and eligible:
            side = "BUY" if kind in _W_BUYS else "SELL"
            original = pending.get(str(anchor))
            if kind == "SOS":
                boundary = _number(row["wyckoff_range_high"])
            elif kind in ("Spring", "SOW"):
                boundary = _number(row["wyckoff_range_low"])
            elif kind == "Upthrust":
                boundary = _number(row["wyckoff_range_high"])
            else:
                boundary = original["invalidation_price"] if original else None
            stop = _number(row.get("wyckoff_stop")) if side == "BUY" else None
            if stop is None:
                stop = boundary
            confirmed = kind not in {"Spring", "SOS"}
            candidate = bool(row.get("wyckoff_rule_buy") if side == "BUY" else row.get("wyckoff_rule_sell"))
            if kind in {"Test", "LPS", "LPSY"} and original is None:
                # No ancestry must never be filled with a moving current range.
                candidate = False
            pattern = {"version": W_RULE_VERSION, "event_type": kind, "origin_anchor": str(anchor),
                       "observed_at": available, "origin_event_id": original["event_id"] if original else None}
            fact = _fact(str(bars.iloc[i].symbol), date, "wyckoff", side, kind, pattern, str(anchor),
                         stop, boundary, atr, settings, candidate=candidate, confirmed=confirmed,
                         eligible=eligible, reference=close,
                         extra={"version": W_RULE_VERSION, "family": "wyckoff", "legacy_state": row.get("wyckoff_state"),
                                "origin_event_id": original["event_id"] if original else None,
                                "range_formed_at": original.get("range_formed_at") if original else row.get("wyckoff_range_formed_at"),
                                "frozen_range_low": original.get("frozen_range_low") if original else _number(row.get("wyckoff_range_low")),
                                "frozen_range_high": original.get("frozen_range_high") if original else _number(row.get("wyckoff_range_high")),
                                "reason_codes": list(row.get("wyckoff_entry_evidence") or []),
                                "state": str(row.get("wyckoff_state") or "unknown")})
            facts.append(fact); new.append(fact); origin_index[fact["event_id"]] = i
            if boundary is not None and (close >= boundary if side == "BUY" else close <= boundary):
                active[fact["event_id"]] = fact
            if kind in {"Spring", "SOS", "Upthrust", "SOW"}:
                pending[date] = fact
            elif original:
                pending.pop(str(anchor), None)
                if original["event_id"] in active:
                    invalidations.extend(_invalidate(active, invalidated, [original["event_id"]], available, "later_event_confirmed"))
        if kind.endswith("invalidated"):
            matching = [key for key, value in active.items() if value["anchor_at"] == str(anchor)]
            invalidations.extend(_invalidate(active, invalidated, matching, available, "legacy_pending_invalidated"))
        supports = []
        for item in active.values():
            snapshot = copy.deepcopy(item)
            age = i - origin_index[item["event_id"]]
            strength = _W_WEIGHTS[item["event_type"]] * max(0., 1. - age / settings["event_score_decay_bars"])
            snapshot.update(snapshot_at=available, current_available_at=available, age_bars=age,
                            event_strength=strength, event_expired=strength == 0,
                            lifecycle_state="strength_expired" if strength == 0 else "valid_confirmed" if item["confirmed"] else "pending")
            supports.append(snapshot)
        buys = [item for item in supports if item["side"] == "BUY"]
        sells = [item for item in supports if item["side"] == "SELL"]
        event_buy, event_sell = max([item["event_strength"] for item in buys], default=0.), max([item["event_strength"] for item in sells], default=0.)
        continuous = bool(scores.iloc[i].eligible)
        demand, supply = _number(scores.iloc[i].demand), _number(scores.iloc[i].supply)
        buy_valid = bool(continuous and demand >= settings["continuous_direction_threshold"] and demand - supply >= settings["continuous_direction_margin"])
        sell_valid = bool(continuous and supply >= settings["continuous_direction_threshold"] and supply - demand >= settings["continuous_direction_margin"])
        state = {"close": close, "buy_score": demand, "sell_score": supply,
                 "event_demand_support": event_buy, "event_supply_support": event_sell,
                 "continuous_direction_valid": continuous, "snapshot_at": available,
                 "support_valid": any(item["event_strength"] > 0 for item in buys),
                 "resistance_valid": any(item["event_strength"] > 0 for item in sells), "invalidated": bool(invalidations),
                 "support_floor": max([item["support_low"] for item in buys], default=None),
                 "resistance_ceiling": min([item["resistance_high"] for item in sells], default=None),
                 "origin_observed_at": max([item["origin_observed_at"] for item in supports], default=None),
                 "prior_atr": atr, "relative_volume": _number(scores.iloc[i].relative_volume),
                 "close_location": _number(scores.iloc[i].close_location)}
        evaluations.append({"source_id": "wyckoff", "family": "wyckoff", "version": W_RULE_VERSION,
            "date": date, "available_at": available, "snapshot_at": available, "input_eligible": bool(eligible and continuous),
            "buy_valid": buy_valid, "sell_valid": sell_valid,
            "direction_state": "BUY" if buy_valid else "SELL" if sell_valid else "ABSTAIN",
            "current_state": state, "support_facts": supports, "active_buy_event_ids": [item["event_id"] for item in buys],
            "active_sell_event_ids": [item["event_id"] for item in sells], "invalidated_event_ids": list(invalidated),
            "invalidations_at_day": invalidations,
            "reason_codes": list(row["wyckoff_reason_codes"]) if not eligible else [] if continuous else ["current_volume_metrics_unavailable"]})
        daily_events.append(copy.deepcopy(new))
    return facts, evaluations, daily_events


@dataclass(frozen=True)
class PreparedNativeWyckoff:
    """A local preparation bound to its inputs, policy and unmodified output."""
    frame: pd.DataFrame
    input_hash: str
    output_hash: str
    profile_hash: str


def _frame_digest(frame: pd.DataFrame, columns) -> str:
    digest = hashlib.sha256()
    digest.update(stable_hash({"columns": list(columns), "dtypes": [str(frame[key].dtype) for key in columns],
        "index_dtype": str(frame.index.dtype), "index_name": frame.index.name}).encode())
    digest.update(pd.util.hash_pandas_object(frame.index, index=True).to_numpy().tobytes())
    for key in columns:
        values = frame[key]
        # Nested reason/evidence objects must be protected along with numerical
        # columns; pandas cannot hash lists directly.
        if values.dtype == object:
            values = values.map(lambda value: stable_hash(_json(value)))
        digest.update(pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def _wyckoff_config(config, market_dates):
    settings = copy.deepcopy((config or {}).get("wyckoff") or {})
    if market_dates is not None:
        settings["market_dates"] = list(market_dates)
    return settings


def _wyckoff_input_hash(frame, bars, settings):
    columns = [key for key in ("symbol", "date", "open", "high", "low", "close", "volume", "amount",
        "source", "has_trade_price") if key in bars]
    identity = bars[columns].copy()
    # build_features deliberately stores text metadata as object, whereas raw
    # Parquet can use pandas StringDtype. Compare the same effective text used
    # by source-quality checks, retaining exact numeric/date/index identity.
    for key in ("symbol", "source"):
        if key in identity:
            identity[key] = identity[key].fillna("").astype(str).astype(object)
    body = {"raw": _frame_digest(identity, columns), "profile": get_wyckoff_profile(settings)["schema_hash"],
        "calendar": settings.get("market_dates"),
        "quality_attrs": {key: bars.attrs.get(key) for key in ("price_basis", "volume_unit", "amount_unit")}}
    if settings.get("market_dates") is None:
        key = "model_reason_codes" if "model_reason_codes" in frame else "reason_codes" if "reason_codes" in frame else None
        body["calendar_reason_codes"] = None if key is None else _frame_digest(frame, [key])
    return stable_hash(body)


def _wyckoff_output_hash(frame):
    columns = [key for key in frame if key.startswith("w_") or key.startswith("wyckoff_")]
    return stable_hash({"data": _frame_digest(frame, columns),
        "attrs": {key: value for key, value in frame.attrs.items() if key.startswith("wyckoff_")}})


def prepare_native_wyckoff_inputs(frame: pd.DataFrame, raw: pd.DataFrame | None = None,
        config: dict | None = None, *, market_dates: list[str] | None = None) -> PreparedNativeWyckoff:
    """Prepare once for reuse only under the same complete observed contract.

    Callers retain their existing feature preparation configuration. A native
    policy with different W settings must perform its own preparation.
    """
    bars = _validate_input(frame if raw is None else raw)
    bars.attrs = dict(frame.attrs)
    dates = market_dates if market_dates is not None else frame.attrs.get("verified_market_dates")
    settings = _wyckoff_config(config, dates)
    result = prepare_wyckoff_inputs(frame, bars, config=settings)
    return PreparedNativeWyckoff(result, _wyckoff_input_hash(frame, bars, settings),
        _wyckoff_output_hash(result), get_wyckoff_profile(settings)["schema_hash"])


def can_reuse_native_wyckoff(prepared: PreparedNativeWyckoff, config: dict | None = None) -> bool:
    """Compare effective W settings; build_native_evidence validates the data."""
    return isinstance(prepared, PreparedNativeWyckoff) and prepared.profile_hash == get_wyckoff_profile(
        (config or {}).get("wyckoff"))["schema_hash"]


def build_native_evidence(frame: pd.DataFrame, config: dict | None = None, *,
                             market_dates: list[str] | None = None,
                             prepared_wyckoff: PreparedNativeWyckoff | None = None,
                             incremental_state: dict | None = None, capture_state: bool = False) -> dict:
    """Enrich a complete single-stock causal prefix; never mutate its old columns.

    Optional ``config['native_evidence']`` changes obtain a distinct schema.
    A caller may pass a strategy configuration in ``config['czsc_strategy']``;
    Wyckoff event/quality configuration lives in ``config['wyckoff']``. An empty
    input returns an empty pack with the same contract and explicit float/object
    columns. ``market_dates`` must come from a verified calendar. Its historical
    prefix is provenance, while future extension cannot change a policy hash.
    With no explicit calendar (or ``attrs['verified_market_dates']``), input
    eligibility remains false rather than inventing a schedule from stock rows.
    A prepared W result is reused only after raw/index, policy/calendar,
    quality metadata and generated output identity checks all succeed.
    """
    settings = _settings(config)
    semantics = {**copy.deepcopy(NATIVE_SEMANTICS), "config": settings}
    czsc_settings = strategy_config((config or {}).get("czsc_strategy") or (config or {}).get("strategy"))
    semantics["structure"]["engine"] = {key: czsc_settings[key] for key in ("min_bi_len", "max_bi_num", "higher_period_closure")}
    semantics["event_w"]["event_input_schema_hash"] = get_wyckoff_profile((config or {}).get("wyckoff"))["schema_hash"]
    schema_hash = stable_hash(semantics)
    config_hash = stable_hash({"native_schema_hash": schema_hash,
                               "native": settings, "czsc": semantics["structure"]["engine"],
                               "wyckoff_schema_hash": semantics["event_w"]["event_input_schema_hash"]})
    if frame.empty:
        empty = frame.copy()
        for key in ("native_structure_buy_support", "native_structure_sell_support", "native_structure_support_low",
                    "native_structure_resistance_high", "native_w_demand_score", "native_w_supply_score",
                    "native_w_event_demand_support", "native_w_event_supply_support",
                    "native_auxiliary_expected_sessions", "native_auxiliary_observed_sessions"):
            empty[key] = pd.Series(index=frame.index, dtype=float)
        for key in ("native_structure_input_eligible", "native_w_input_eligible", "native_auxiliary_input_eligible", "native_auxiliary_calendar_complete"):
            empty[key] = pd.Series(index=frame.index, dtype=bool)
        for key in ("native_structure_current_valid_zone", "native_structure_snapshot", "native_w_snapshot", "native_events_at_day",
                    "native_version", "native_schema_hash"):
            empty[key] = pd.Series(index=frame.index, dtype=object)
        empty["native_auxiliary_dependency_start"] = pd.Series(index=frame.index, dtype=object)
        empty.attrs.update(native_version=NATIVE_EVIDENCE_VERSION, native_schema_hash=schema_hash,
                           native_semantics=semantics, native_config_hash=config_hash)
        return {"frame": empty, "events": [], "evaluations": [], "wyckoff_events": [],
                "auxiliary_evaluations": [], "calendar_provenance": {"provided": market_dates is not None},
                "version": NATIVE_EVIDENCE_VERSION, "config_hash": config_hash,
                "schema_hash": schema_hash, "semantics": semantics}
    bars = _validate_input(frame)
    bars.attrs = dict(frame.attrs)
    market_dates = market_dates if market_dates is not None else frame.attrs.get("verified_market_dates")
    calendar = _VerifiedCalendar(bars, market_dates)
    w_config = _wyckoff_config(config, market_dates)
    if prepared_wyckoff is None:
        legacy = prepare_wyckoff_inputs(frame, bars, config=w_config)
    else:
        if (not can_reuse_native_wyckoff(prepared_wyckoff, config) or
                prepared_wyckoff.input_hash != _wyckoff_input_hash(frame, bars, w_config) or
                prepared_wyckoff.output_hash != _wyckoff_output_hash(prepared_wyckoff.frame)):
            raise ValueError("prepared Wyckoff input, policy or output identity drift")
        # The no-calendar branch below adjusts eligibility, so keep the bound
        # prepared result immutable and independently reusable.
        legacy = prepared_wyckoff.frame.copy()
    if not calendar.provided:
        legacy["wyckoff_input_eligible"] = False
        legacy["wyckoff_reason_codes"] = [list(codes) + ["native_verified_calendar_unavailable"] for codes in legacy.wyckoff_reason_codes]
    metrics = _prior_metrics(bars)
    raw_quality = legacy["wyckoff_input_eligible"].to_numpy(dtype=bool)
    scores = _continuous_scores(bars, metrics, raw_quality)
    structure_state={} if capture_state else None
    prefix_rows=0
    if incremental_state:
        if (incremental_state.get('format')!='native_evidence_state_v1' or
            incremental_state.get('state_sha256')!=stable_hash({k:v for k,v in incremental_state.items() if k!='state_sha256'})):
            raise ValueError('native continuation state SHA or schema mismatch')
        prefix_rows=incremental_state['structure']['rows']
        if type(prefix_rows) is not int or not 0<prefix_rows<len(bars):
            raise ValueError('native continuation requires a nonempty strict observed prefix')
        prefix=bars.iloc[:prefix_rows]
        actual=stable_hash(_json(prefix.to_dict('records')))
        prefix_dates=[d for d in (market_dates or []) if d<=_day(prefix.iloc[-1].date)]
        if (incremental_state.get('schema_hash')!=schema_hash or incremental_state.get('config_hash')!=config_hash or
            incremental_state.get('raw_prefix_hash')!=actual or incremental_state.get('calendar_prefix_hash')!=stable_hash(prefix_dates)):
            raise ValueError('native continuation input prefix or contract changed')
    structural, structure_eval, structure_daily, auxiliary = _structure_evidence(bars, raw_quality, metrics,
        settings, czsc_settings, calendar, resume_state=incremental_state['structure'] if incremental_state else None,
        state_sink=structure_state)
    wyckoff, w_eval, w_daily = _wyckoff_evidence(bars, legacy, metrics, scores, settings, (config or {}).get("wyckoff"))
    if prefix_rows:
        frame=frame.iloc[prefix_rows:]; scores=scores.iloc[prefix_rows:]
        w_eval=w_eval[prefix_rows:];w_daily=w_daily[prefix_rows:]
        wyckoff=[item for item in wyckoff if item['date']>=_day(frame.iloc[0].date)]
    for item in structure_eval + w_eval:
        item["evidence_schema_hash"] = schema_hash
        item["evidence_config_hash"] = config_hash
        item["snapshot_id"] = item["source_id"] + ":snapshot:native:" + stable_hash({
            "schema_hash": schema_hash, "symbol": str(bars.iloc[0].symbol), "date": item["date"],
            "input_eligible": item["input_eligible"], "current_state": item["current_state"],
            "origin_event_ids": [fact["event_id"] for fact in item["support_facts"]],
            "invalidated_event_ids": item["invalidated_event_ids"]})
    result = frame.copy(); result.attrs = dict(frame.attrs)
    result["native_structure_buy_support"] = [item["current_state"]["buy_score"] for item in structure_eval]
    result["native_structure_sell_support"] = [item["current_state"]["sell_score"] for item in structure_eval]
    result["native_structure_input_eligible"] = [item["input_eligible"] for item in structure_eval]
    result["native_structure_support_low"] = [item["current_state"]["support_floor"] for item in structure_eval]
    result["native_structure_resistance_high"] = [item["current_state"]["resistance_ceiling"] for item in structure_eval]
    result["native_structure_current_valid_zone"] = pd.Series([item["current_state"]["current_valid_zone"] for item in structure_eval], index=frame.index, dtype=object)
    result["native_structure_snapshot"] = pd.Series(structure_eval, index=frame.index, dtype=object)
    result["native_w_demand_score"] = scores["demand"]
    result["native_w_supply_score"] = scores["supply"]
    result["native_w_event_demand_support"] = [item["current_state"]["event_demand_support"] for item in w_eval]
    result["native_w_event_supply_support"] = [item["current_state"]["event_supply_support"] for item in w_eval]
    result["native_w_input_eligible"] = scores["eligible"]
    result["native_w_snapshot"] = pd.Series(w_eval, index=frame.index, dtype=object)
    result["native_events_at_day"] = pd.Series([s + w for s, w in zip(structure_daily, w_daily)], index=frame.index, dtype=object)
    result["native_version"] = NATIVE_EVIDENCE_VERSION
    result["native_schema_hash"] = schema_hash
    result["native_auxiliary_input_eligible"] = [item["input_eligible"] for item in auxiliary]
    result["native_auxiliary_calendar_complete"] = [item["dependency_calendar_complete"] for item in auxiliary]
    result["native_auxiliary_dependency_start"] = pd.Series([item["dependency_start"] for item in auxiliary], index=frame.index, dtype=object)
    for column in ("expected_sessions", "observed_sessions"):
        result["native_auxiliary_" + column] = pd.Series([item["dependency_" + column] for item in auxiliary], index=frame.index, dtype=float)
    # Stable explicit dtypes prevent a later first event changing an earlier prefix.
    numeric = [key for key in result if key.startswith("native_") and key.endswith(("support", "score", "low", "high"))]
    for key in numeric:
        result[key] = pd.to_numeric(result[key], errors="coerce").astype(float)
    result.attrs.update(native_version=NATIVE_EVIDENCE_VERSION, native_schema_hash=schema_hash,
                        native_semantics=semantics, native_config_hash=config_hash)
    output={"frame": result, "events": sorted(structural + wyckoff, key=lambda item: (item["date"], item["source"], item["event_id"])),
            "evaluations": [{"date": s["date"], "available_at": s["available_at"], "structure": s, "wyckoff": w}
                            for s, w in zip(structure_eval, w_eval)],
            "wyckoff_events": copy.deepcopy(wyckoff), "version": NATIVE_EVIDENCE_VERSION,
            "auxiliary_evaluations": auxiliary, "calendar_provenance": calendar.provenance,
            "config_hash": config_hash, "schema_hash": schema_hash, "semantics": semantics}
    if capture_state:
        output['incremental_state']={'format':'native_evidence_state_v1','structure':structure_state,
            'schema_hash':schema_hash,'config_hash':config_hash,'raw_prefix_hash':stable_hash(_json(bars.to_dict('records'))),
            'calendar_prefix_hash':stable_hash([d for d in (market_dates or []) if d<=_day(bars.iloc[-1].date)])}
        output['incremental_state']['state_sha256']=stable_hash(output['incremental_state'])
        output['incremental']={'native_rows_appended':len(bars)-prefix_rows,'native_prefix_reused':prefix_rows,
            'wyckoff_mode':'single_stock_vector_replay','wyckoff_rows_replayed':len(bars)}
    return output
