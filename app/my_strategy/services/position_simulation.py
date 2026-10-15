"""Small causal long-only research account, backed by BrokerSimulator's ledger.

This module has no filesystem or model-training side effects. Callers supply
observed prices, confirmed evidence and optionally a verified session calendar.
The returned checkpoint can be passed back with the same historical prefix.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.execution.broker import BrokerSimulator
from my_strategy.execution.cost_model import CostModel
from my_strategy.execution.ledger import LedgerEntry
from my_strategy.execution.order import Order

_IMPLEMENTATION_HASHES = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (Path(__file__), Path(inspect.getfile(BrokerSimulator)), Path(inspect.getfile(CostModel)),
                 Path(inspect.getfile(LedgerEntry)), Path(inspect.getfile(Order)))}

VERSION = "structure_position_simulation_v2"
POLICIES = {"structure_rules", "structure_kronos", "structure_kronos_score", "structure_kronos_l2", "cash", "buy_hold"}
DEFAULTS = dict(
    policy="structure_rules", research_only=True, initial_cash=1_000_000.0,
    max_stock_weight=.20, max_total_weight=.90, max_positions=10,
    initial_weight=.10, position_step=.05, risk_budget=.01,
    stop_loss=.08, ma_period=20, ma_exit_closes=2, atr_period=14,
    trailing_atr=2.5, add_interval=5, cooldown=3, time_exit_sessions=20,
    max_holding_sessions=60, dd_reduce=.10, dd_exit=.15,
    dd_rearm=.05, dd_rearm_sessions=5, dd_cooldown=5,
    commission=.0003, min_commission=5.0, slippage=.0005,
    transfer_fee=.00001, cost_stress=1.0, cost_version="flat_v1",
    max_volume_participation=.01, lot_size=100,
    min_forecast_return=0.0, min_up_fraction=.50, min_score=50.0,
    forecast_exit_closes=2, candidate_sessions=1, fusion_mode="hard_filters",
    research_mode="legacy", enable_agent_evidence=False,
    annualization=252, risk_free_annual=0.0, min_sharpe_sessions=60,
    require_verified_rules=False, company_actions_verified=False,
)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp,)): return value.isoformat()
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_clean(value), sort_keys=True,
        ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (ValueError, TypeError): return default


def _day(value):
    value = pd.Timestamp(value)
    if value.tzinfo is not None: value = value.tz_convert("Asia/Shanghai")
    return value.date().isoformat()


def simulation_config(config=None):
    supplied = dict(config or {})
    if "strategy" in supplied:
        strategy = supplied.pop("strategy")
        if "policy" in supplied and supplied["policy"] != strategy:
            raise ValueError("strategy and policy disagree")
        supplied["policy"] = strategy
    aliases = {"structure_rules_sim_v1": "structure_rules", "structure_kronos_sim_v1": "structure_kronos",
               "structure_kronos_score_sim_v1": "structure_kronos_score"}
    if "policy" in supplied: supplied["policy"] = aliases.get(supplied["policy"], supplied["policy"])
    unknown = set(supplied) - set(DEFAULTS)
    if unknown: raise ValueError("Unknown simulation configuration: " + ", ".join(sorted(unknown)))
    cfg = {**DEFAULTS, **supplied}
    if cfg["policy"] not in POLICIES: raise ValueError("Unsupported simulation policy")
    if cfg["research_only"] is not True:
        raise ValueError("This simulator supports research_only=True; it does not certify returns")
    for key in ("max_positions", "ma_period", "ma_exit_closes", "atr_period", "add_interval",
                "cooldown", "time_exit_sessions", "max_holding_sessions", "dd_rearm_sessions",
                "dd_cooldown", "annualization", "min_sharpe_sessions", "lot_size"):
        if isinstance(cfg[key], bool) or _number(cfg[key]) is None or int(cfg[key]) != cfg[key] or cfg[key] < 1:
            raise ValueError(key + " must be a positive integer")
        cfg[key] = int(cfg[key])
    if isinstance(cfg["forecast_exit_closes"],bool) or int(cfg["forecast_exit_closes"]) != cfg["forecast_exit_closes"] or cfg["forecast_exit_closes"] < 1:
        raise ValueError("forecast_exit_closes must be a positive integer")
    for key in ("initial_cash", "trailing_atr", "cost_stress"):
        if _number(cfg[key]) is None or cfg[key] <= 0: raise ValueError(key + " must be positive")
    for key in ("max_stock_weight", "max_total_weight", "initial_weight", "position_step", "risk_budget",
                "stop_loss", "dd_reduce", "dd_exit", "dd_rearm", "max_volume_participation"):
        if _number(cfg[key]) is None or not 0 < cfg[key] <= 1: raise ValueError(key + " must be in (0,1]")
    if not cfg["dd_rearm"] < cfg["dd_reduce"] < cfg["dd_exit"] < 1:
        raise ValueError("Drawdown thresholds must satisfy rearm < reduce < exit < 1")
    if not cfg["position_step"] <= cfg["initial_weight"] <= cfg["max_stock_weight"] <= cfg["max_total_weight"]:
        raise ValueError("Position limits must satisfy step <= initial <= stock <= total")
    if cfg["lot_size"] != 100:
        raise ValueError("First implementation supports 100-share main-board purchases only")
    for key in ("commission", "min_commission", "slippage", "transfer_fee"):
        if _number(cfg[key]) is None or cfg[key] < 0: raise ValueError(key + " must be finite and nonnegative")
    if cfg["cost_version"] not in {"flat_v1", "tiered_v1"}: raise ValueError("Unknown cost_version")
    if _number(cfg["risk_free_annual"]) is None or cfg["risk_free_annual"] <= -1:
        raise ValueError("risk_free_annual must be finite and greater than -1")
    if not 0 <= cfg["min_up_fraction"] <= 1 or not 0 <= cfg["min_score"] <= 100:
        raise ValueError("Invalid forecast/score threshold")
    if _number(cfg["min_forecast_return"]) is None: raise ValueError("Invalid min_forecast_return")
    if cfg["candidate_sessions"] not in (1, 3): raise ValueError("candidate_sessions must be 1 or 3")
    if cfg["fusion_mode"] not in {"hard_filters", "evidence"}: raise ValueError("Unknown fusion_mode")
    if cfg["research_mode"] not in {"legacy", "historical_candidate", "paper"}: raise ValueError("Unknown research_mode")
    return cfg


@dataclass(frozen=True)
class _SimulationCosts(CostModel):
    transfer_rate: float = 0.0

    def fees(self, side, notional):
        return super().fees(side, notional) + notional * self.transfer_rate


def _costs(cfg, day):
    # Published rate change is a dated rule, not a stress-test parameter.
    stamp = .0005 if day >= "2023-08-28" else .001
    options = dict(commission=cfg["commission"] * cfg["cost_stress"], stamp_tax=stamp,
        min_commission=cfg["min_commission"] * cfg["cost_stress"],
        slippage=cfg["slippage"] * cfg["cost_stress"], version=cfg["cost_version"],
        transfer_rate=cfg["transfer_fee"])
    if cfg["cost_version"] == "flat_v1": options["impact_tiers"] = ()
    else:
        from my_strategy.execution.cost_model import DEFAULT_IMPACT_TIERS, ImpactTier
        options["impact_tiers"] = tuple(ImpactTier(t.max_amount_ma20, t.impact_bp * cfg["cost_stress"])
                                         for t in DEFAULT_IMPACT_TIERS)
    return _SimulationCosts(**options)


def _prepare_price_inputs(frames, calendar):
    if not frames: raise ValueError("At least one symbol is required")
    bars, normalized, observed = {}, {}, set()
    for symbol, frame in sorted(frames.items()):
        # Numerical price preparation never reads metadata. A fresh constructor
        # keeps columns/index/dtypes without deep-copying source evidence attrs.
        frame = pd.DataFrame(frame, copy=True)
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame): raise ValueError(f"{symbol}: missing price columns {sorted(required-set(frame))}")
        frame["date"] = frame["date"].map(_day)
        if frame.date.duplicated().any(): raise ValueError(symbol + ": duplicate price date")
        frame = frame.sort_values("date")
        records = _clean(frame.to_dict("records"))
        normalized[symbol] = records
        bars[symbol] = {r["date"]: r for r in records}
        observed.update(bars[symbol])
    dates = [_day(x) for x in calendar] if calendar is not None else sorted(observed)
    if dates != sorted(set(dates)): raise ValueError("Calendar must be sorted and unique")
    if not observed.issubset(set(dates)) and calendar is not None:
        # Warmup may predate the explicitly supplied account calendar.
        outside = [d for d in observed if dates and dates[0] <= d <= dates[-1] and d not in dates]
        if outside: raise ValueError("Observed price dates disagree with calendar")
    return bars, normalized, dates


def _prepare(frames, evidence, start, end, calendar, prepared_runtime=None):
    if prepared_runtime is None:
        bars,normalized,dates=_prepare_price_inputs(frames,calendar)
    else:
        if frames is not prepared_runtime.frames or calendar is not prepared_runtime.calendar:
            raise ValueError('Prepared runtime belongs to another frozen request')
        bars,normalized,dates=prepared_runtime.bars,prepared_runtime.raw,prepared_runtime.dates
    start = _day(start) if start else dates[0]
    end = _day(end) if end else dates[-1]
    if start > end: raise ValueError("start must not exceed end")
    sessions = [d for d in dates if start <= d <= end]
    if not sessions: raise ValueError("No sessions in requested interval")
    if (prepared_runtime is not None and prepared_runtime.evidence_epoch is not None
            and prepared_runtime.cached_evidence is evidence
            and prepared_runtime.cached_epoch==prepared_runtime.evidence_epoch):
        return bars,normalized,prepared_runtime.cached_daily_evidence,sessions,start
    daily_evidence = {s: {} for s in bars}
    for symbol, entries in (evidence or {}).items():
        if symbol not in bars: raise ValueError("Evidence symbol outside frozen population")
        for original in entries:
            entry = _clean(copy.deepcopy(original))
            if not entry.get("date"): raise ValueError("Evidence date is required")
            timestamp = pd.Timestamp(entry.get("available_at") or (str(entry["date"])[:10] + "T15:00:00+08:00"))
            timestamp = timestamp.tz_localize("Asia/Shanghai") if timestamp.tzinfo is None else timestamp.tz_convert("Asia/Shanghai")
            available_day = timestamp.date().isoformat()
            late = timestamp.hour > 15 or (timestamp.hour == 15 and (timestamp.minute or timestamp.second or timestamp.microsecond))
            effective = next((d for d in dates if d > available_day or (d == available_day and not late)), None)
            if entry.get("observation_kind") == "daily_agent_observation":
                # An after-close forecast belongs to this close's decision slot,
                # but retains its actual availability. The order separately
                # enforces that it existed before its next executable open.
                effective = _day(entry["date"])
            if effective is None: continue
            if _day(entry["date"]) > available_day: raise ValueError("Evidence cannot be available before its event date")
            entry["available_at"] = timestamp.isoformat()
            entry["event_id"] = str(entry.get("event_id") or _hash({"symbol": symbol, "date": entry["date"],
                "available_at": entry["available_at"], "side": entry.get("side"), "source": entry.get("source")}))
            daily_evidence[symbol].setdefault(effective, []).append(entry)
    for by_day in daily_evidence.values():
        for entries in by_day.values(): entries.sort(key=lambda r: (r["available_at"], r["event_id"]))
    if prepared_runtime is not None and prepared_runtime.evidence_epoch is not None:
        prepared_runtime.cached_evidence=evidence
        prepared_runtime.cached_epoch=prepared_runtime.evidence_epoch
        prepared_runtime.cached_daily_evidence=daily_evidence
    return bars, normalized, daily_evidence, sessions, start


class PreparedSimulationRuntime:
    """Request-local price snapshot, never persisted or reused across requests.

    Price normalization, indicator histories and price serialization are shared
    by the independent accounts. Evidence remains causal and is rechecked on
    each step. A historical evidence change invalidates the cached hash suffix;
    a new request always normalizes actual inputs and verifies saved prefixes.
    """
    def __init__(self,frames,calendar,*,start,end):
        self.frames,self.calendar=frames,calendar
        self.bars,self.raw,self.dates=_prepare_price_inputs(frames,calendar)
        self.start,self.end=_day(start),_day(end)
        self.symbols=sorted(self.bars)
        self.indicator_cache={};self.prefix_cache={};self.serialized_bars={}
        self.evidence_epoch=None;self.cached_epoch=None;self.cached_evidence=None
        self.cached_daily_evidence=None
        self.stats={'price_preparations':1,'indicator_preparations':0,'prefix_days_hashed':0}

    @staticmethod
    def encoded(value):
        return json.dumps(_clean(value),sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)

    def indicators(self,cfg,actions):
        key=(cfg['ma_period'],cfg['atr_period'],_hash(actions))
        if key not in self.indicator_cache:
            self.indicator_cache[key]={s:_indicators(self.raw[s],cfg,[a for a in actions if a['symbol']==s]) for s in self.symbols}
            self.stats['indicator_preparations']+=1
        return self.indicator_cache[key]

    def begin_step(self):
        # The scheduler owns an evidence snapshot and adds observations only
        # between steps. Every step reparses/verifies it once for all accounts.
        self.evidence_epoch=0 if self.evidence_epoch is None else self.evidence_epoch+1

    def prefixes(self,evidence,sessions,start,actions):
        if start!=self.start or sessions[-1]>self.end:raise ValueError('Prepared runtime interval changed')
        # Benchmarks have no source evidence; their chain must not invalidate the
        # independent source-account chain at the final comparison step.
        populated=any(any(rows.values()) for rows in evidence.values())
        key=(_hash(actions),populated)
        if key not in self.prefix_cache:
            historical=sorted({r['date'] for records in self.raw.values() for r in records}
                | {d for d in self.dates if self.start<=d<=self.end} | {a['record_date'] for a in actions})
            historical=[d for d in historical if d<=self.end]
            by_action={day:[a for a in actions if a['record_date']==day] for day in {a['record_date'] for a in actions}}
            self.prefix_cache[key]={'base_dates':historical,'hashes':{},'states':{},'evidence':{},
                'actions':{day:self.encoded(rows) for day,rows in by_action.items()}}
        cached=self.prefix_cache[key]
        if cached.get('last_evidence') is evidence and sessions[-1] in cached['hashes']:
            return cached['hashes']
        observed_days={day for rows in evidence.values() for day in rows}
        # A confirmed warmup event can fall on a calendar day with no price
        # for any symbol. Its evidence must still bind the historical prefix.
        # Rebuild this union when evidence changes so added/removed dates are
        # hashed exactly as in the ordinary, uncached path.
        cached['dates']=sorted(set(cached['base_dates']) | {day for day in observed_days if day<=self.end})
        encoded_evidence={day:self.encoded({s:evidence[s].get(day,[]) for s in self.symbols}) for day in observed_days}
        cutoff=sessions[-1]
        changed=[day for day in set(cached['evidence'])|set(encoded_evidence)
            if day<=cutoff and cached['evidence'].get(day)!=encoded_evidence.get(day)]
        if changed:
            first=min(changed)
            for day in [d for d in cached['hashes'] if d>=first]:
                del cached['hashes'][day];del cached['states'][day]
        cached['evidence']=encoded_evidence
        cached['last_evidence']=evidence
        last=max(cached['hashes'],default=None)
        prefix=cached['states'][last].copy() if last else hashlib.sha256()
        empty=self.encoded({s:[] for s in self.symbols})
        account_days=set(self.dates)
        for day in cached['dates']:
            if (last and day<=last) or day>cutoff:continue
            if day not in self.serialized_bars:
                self.serialized_bars[day]=self.encoded({s:self.bars[s].get(day) for s in self.symbols})
            payload=('{"account_session":'+('true' if day>=start and day in account_days else 'false')
                +',"bars":'+self.serialized_bars[day]+',"corporate_actions":'+cached['actions'].get(day,'[]')
                +',"date":'+json.dumps(day)+',"evidence":'+encoded_evidence.get(day,empty)+'}')
            prefix.update(hashlib.sha256(payload.encode()).hexdigest().encode())
            cached['hashes'][day]=prefix.hexdigest();cached['states'][day]=prefix.copy()
            self.stats['prefix_days_hashed']+=1
        return cached['hashes']


def _indicators(records, cfg, actions=()):
    frame = pd.DataFrame(records)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    close = frame.close
    frame["_ma"] = close.rolling(cfg["ma_period"], min_periods=cfg["ma_period"]).mean()
    frame["_previous_ma"] = frame._ma.shift(1)
    ranges = pd.concat([frame.high-frame.low, (frame.high-close.shift(1)).abs(),
                        (frame.low-close.shift(1)).abs()], axis=1).max(axis=1)
    frame["_atr"] = ranges.rolling(cfg["atr_period"], min_periods=cfg["atr_period"]).mean()
    amount = pd.to_numeric(frame.get("amount", pd.Series(np.nan, index=frame.index)), errors="coerce")
    frame["_amount_ma20"] = amount.rolling(20, min_periods=5).mean().shift(1)
    frame["_previous_volume"] = frame.volume.shift(1)
    frame["_previous_close"] = close.shift(1)
    result = {r["date"]: _clean(r) for r in frame.to_dict("records")}
    if not actions: return result
    closes, ranges, previous_ma = [], [], None
    previous_close, previous_volume, amounts, applied = None, None, [], set()
    for record in records:
        day = record["date"]
        ratio, volume_factor = 1.0, 1.0
        for action in actions:
            if action["ex_date"] > day or action["action_id"] in applied: continue
            applied.add(action["action_id"])
            if previous_close is None: continue  # No pre-action observations to transform.
            factor = action.get("price_factor")
            if factor is None:
                if previous_close is None or previous_close <= action["cash_per_share"]:
                    raise ValueError("Corporate action needs a prior close or explicit price_factor")
                factor = (previous_close-action["cash_per_share"])/previous_close/action["share_multiplier"]
            ratio *= factor
            volume_factor *= action["share_multiplier"]
        if ratio != 1:
            closes = [c*ratio for c in closes]
            ranges = [r*ratio for r in ranges]
            if previous_close is not None: previous_close *= ratio
            if previous_ma is not None: previous_ma *= ratio
        close, high, low = (float(record[k]) for k in ("close", "high", "low"))
        tr = max(high-low, abs(high-previous_close), abs(low-previous_close)) if previous_close is not None else high-low
        closes.append(close); ranges.append(tr)
        ma = sum(closes[-cfg["ma_period"]:])/cfg["ma_period"] if len(closes)>=cfg["ma_period"] else None
        result[day].update(_ma=ma, _previous_ma=previous_ma,
            _atr=sum(ranges[-cfg["atr_period"]:])/cfg["atr_period"] if len(ranges)>=cfg["atr_period"] else None,
            _previous_close=previous_close, _previous_volume=previous_volume*volume_factor if previous_volume is not None else None,
            _amount_ma20=sum(amounts[-20:])/min(20,len(amounts)) if len(amounts)>=5 else None)
        previous_ma, previous_close = ma, close
        previous_volume = _number(record.get("volume"))
        amount = _number(record.get("amount"))
        if amount is not None: amounts.append(amount)
    return result


def _valid_price(row, field="close"):
    return row is not None and (_number(row.get(field), 0) > 0)


def _sellable(lots, day):
    return sum(lot["shares"] for lot in lots if
        (day >= lot["sellable_from"] if lot.get("sellable_from") else lot["date"] < day))


def _fifo(lots, quantity, day):
    """Consume only unlocked lots and return their fee-inclusive acquisition basis."""
    if quantity > _sellable(lots, day): raise ValueError("T+1: requested sale exceeds unlocked lots")
    remaining, matched = quantity, []
    for lot in lots:
        if (day < lot["sellable_from"] if lot.get("sellable_from") else lot["date"] >= day): continue
        count = min(remaining, lot["shares"])
        if count:
            matched.append(dict(lot_id=lot["lot_id"], buy_date=lot["date"], shares=count,
                                unit_cost=lot["unit_cost"], basis=count*lot["unit_cost"]))
            lot["shares"] -= count
            remaining -= count
        if remaining == 0: break
    lots[:] = [lot for lot in lots if lot["shares"]]
    return matched


def _rule_failure(symbol, row, side, cfg, price):
    if not ((symbol.endswith(".SH") and symbol.startswith("60")) or
            (symbol.endswith(".SZ") and symbol.startswith("00"))):
        return "unsupported_market_mainboard_only"
    if row is None or not _valid_price(row, "open"): return "missing_open"
    if row.get("suspended") is True or row.get("is_suspended") is True:
        return "suspended"
    if row.get("has_trade_price") == 0: return "unverified_trade_price"
    if cfg["require_verified_rules"] and row.get("rules_verified") is not True:
        return "historical_execution_rules_unverified"
    if row.get("ipo_unrestricted") is True: return "unsupported_ipo_execution_rules"
    limit_up, limit_down = _number(row.get("limit_up")), _number(row.get("limit_down"))
    previous = _number(row.get("_previous_close"))
    if limit_up is None or limit_down is None:
        if cfg["require_verified_rules"]: return "missing_actual_price_limits"
        if previous is None: return "missing_previous_close_for_research_limits"
        limit = .05 if row.get("is_st") is True else .10
        limit_up = round(previous*(1+limit)+1e-8, 2)
        limit_down = round(previous*(1-limit)+1e-8, 2)
    # No end-of-day high/low/volume is consulted to infer opening queue fills.
    if side == "BUY" and (float(row["open"]) >= limit_up-1e-8 or price > limit_up+1e-8):
        return "limit_up_or_slippage_outside_limit"
    if side == "SELL" and (float(row["open"]) <= limit_down+1e-8 or price < limit_down-1e-8):
        return "limit_down_or_slippage_outside_limit"
    return None


def _forecast_reasons(event, cfg):
    if cfg["policy"] in {"structure_rules", "cash", "buy_hold"}: return []
    forecast = event.get("forecast") or {}
    quality = forecast.get("quality") or {}
    score = event.get("score") or {}
    # A qualified B fusion model consumes Kronos and its explicit missing masks;
    # never add its component votes back to the fused score.
    fused_b = (cfg["policy"] in {"structure_kronos_score","structure_kronos_l2"} and cfg["fusion_mode"] == "evidence"
               and score.get("variant") == "B" and score.get("consumes_kronos") is True)
    reasons = []
    if forecast.get("late_for_decision") is True: reasons.append("forecast_arrived_after_executable_open")
    if not fused_b:
        if quality.get("prediction_usable", forecast.get("prediction_usable")) is not True:
            reasons.append("forecast_missing_or_unusable")
        else:
            summary = forecast.get("summary") or forecast
            expected = _number(summary.get("expected_return", summary.get("terminal_return_median", summary.get("mean_return"))))
            fraction = _number(summary.get("up_fraction", summary.get("up_path_fraction", summary.get("up_probability", summary.get("probability_up")))))
            if expected is None or fraction is None: reasons.append("forecast_summary_missing")
            elif expected <= cfg["min_forecast_return"] or (cfg["fusion_mode"] == "hard_filters" and fraction < cfg["min_up_fraction"]):
                reasons.append("forecast_direction_blocked")
    if cfg["policy"] in {"structure_kronos_score","structure_kronos_l2"}:
        if not isinstance(score, dict) or score.get("status") != "scored": reasons.append("simple_score_pending")
        else:
            qualification = score.get("qualification") or {}
            if cfg["research_mode"] == "legacy":
                if any(qualification.get(key) is False for key in
                    ("allow_increase", "positive_allocation_eligible", "policy_passed", "policy_baseline_passed")):
                    reasons.append("simple_score_policy_not_passed")
            else:
                from my_strategy.services.agent_evidence import qualification_reasons
                reasons.extend(qualification_reasons(score, cfg["research_mode"]))
            value = _number(score.get("score"))
            if value is None or value < cfg["min_score"]: reasons.append("simple_score_below_threshold")
    return list(dict.fromkeys(reasons))


def _forecast_reason(event, cfg):
    reasons = _forecast_reasons(event, cfg)
    return reasons[0] if reasons else None


def _quantity_plan(before, delta, price, equity):
    price, equity = _number(price), _number(equity)
    usable = price is not None and equity is not None and equity > 0
    return dict(shares_before=int(before), share_delta=int(delta), target_shares=int(before+delta),
        signal_price=price, signal_equity=equity,
        weight_before=before*price/equity if usable else None,
        target_weight=(before+delta)*price/equity if usable else None,
        weight_delta=delta*price/equity if usable else None,
        sold_fraction=-delta/before if delta<0 and before else None,
        denominator="signal_equity", valuation="observed_signal_price")


def _corporate_actions(actions, symbols, calendar):
    """Normalize explicit rights. Unknown ratio, fractions or tax timing fail closed.

    Cash: cash_per_share (gross), tax_per_share (explicit research assumption),
    record_date, ex_date, pay_date. Shares: share_multiplier>=1 and optional
    shares_pay_date/sellable_date. One source_id is mandatory for provenance.
    Share rights require ex-date to immediately follow record-date, avoiding
    unsupported transfers of detached rights between those sessions.
    """
    normalized, seen = [], set()
    for original in actions or []:
        action = copy.deepcopy(original)
        aid = str(action.get("action_id") or "")
        if not aid or aid in seen: raise ValueError("Company action IDs must be nonempty and unique")
        seen.add(aid)
        if action.get("symbol") not in symbols or not action.get("source_id"):
            raise ValueError("Company action requires a covered symbol and explicit source_id")
        for key in ("record_date", "ex_date"):
            if not action.get(key): raise ValueError("Company action requires " + key)
            action[key] = _day(action[key])
        if action["record_date"] >= action["ex_date"]:
            raise ValueError("Company action record_date must precede ex_date")
        if action.get("available_at") and _day(action["available_at"]) > action["record_date"]:
            raise ValueError("Company-action terms must be known by record date")
        action["cash_per_share"] = _number(action.get("cash_per_share"), 0)
        action["share_multiplier"] = _number(action.get("share_multiplier"), 1)
        if action["cash_per_share"] < 0 or action["share_multiplier"] < 1:
            raise ValueError("Negative dividends and reverse splits require a separate corporate-action contract")
        if action["cash_per_share"]:
            if "tax_per_share" not in action or not action.get("pay_date"):
                raise ValueError("Cash dividend requires explicit tax_per_share and pay_date")
            action["tax_per_share"] = _number(action["tax_per_share"])
            if action["tax_per_share"] is None or not 0 <= action["tax_per_share"] <= action["cash_per_share"]:
                raise ValueError("Invalid explicit dividend tax")
            action["pay_date"] = _day(action["pay_date"])
            if action["pay_date"] < action["ex_date"]: raise ValueError("Dividend cannot pay before ex-date")
        else:
            action["tax_per_share"] = 0.0
            action["pay_date"] = action["ex_date"]
        action["shares_pay_date"] = _day(action.get("shares_pay_date") or action["ex_date"])
        action["sellable_date"] = _day(action.get("sellable_date") or action["shares_pay_date"])
        if action["shares_pay_date"] < action["ex_date"] or action["sellable_date"] < action["shares_pay_date"]:
            raise ValueError("Share entitlement delivery/availability precedes its effective date")
        if action["share_multiplier"] > 1:
            between = [d for d in calendar if action["record_date"] < d < action["ex_date"]]
            if between: raise ValueError("Share rights require consecutive record/ex sessions in this implementation")
        if action.get("price_factor") is not None:
            action["price_factor"] = _number(action["price_factor"])
            if action["price_factor"] is None or action["price_factor"] <= 0:
                raise ValueError("Corporate price_factor must be positive")
        normalized.append(action)
    return sorted(normalized, key=lambda a: (a["record_date"], a["symbol"], a["action_id"]))


def _entitlements_value(state, prices):
    cash = sum(e["cash_gross"] for e in state["entitlements"].values() if e.get("recognized") and not e.get("cash_paid"))
    taxes = sum(e["tax"] for e in state["entitlements"].values() if e.get("recognized") and not e.get("cash_paid"))
    shares = sum(e["new_shares"]*prices.get(e["symbol"], 0) for e in state["entitlements"].values()
                 if e.get("recognized") and not e.get("shares_delivered"))
    return cash, taxes, shares


def _equity(broker, state, prices):
    cash, tax, shares = _entitlements_value(state, prices)
    return broker.equity(prices)+cash-tax+shares


def _pending_rights(state, symbol):
    return sum(e["new_shares"] for e in state["entitlements"].values()
               if e["symbol"] == symbol and e.get("recognized") and not e.get("shares_delivered"))


def _record_rights(actions, day, state):
    for action in actions:
        if action["record_date"] != day or action["action_id"] in state["entitlements"]: continue
        symbol = action["symbol"]
        eligible = copy.deepcopy(state["lots"][symbol])
        count = sum(l["shares"] for l in eligible)
        raw_new = count*(action["share_multiplier"]-1)
        if abs(raw_new-round(raw_new)) > 1e-8:
            raise ValueError("Fractional share entitlement requires explicit cash-in-lieu rules")
        state["entitlements"][action["action_id"]] = dict(action=action, symbol=symbol,
            cycle_opened_index=state["cycle_start"].get(symbol),
            eligible_shares=count, eligible_lots=eligible, cash_gross=count*action["cash_per_share"],
            tax=count*action["tax_per_share"], new_shares=int(round(raw_new)), recognized=False,
            cash_paid=False, shares_delivered=False, new_lots=[])
        state["corporate_ledger"].append(dict(date=day, action_id=action["action_id"], symbol=symbol,
            phase="record", eligible_shares=count, source_id=action["source_id"], cash_flow=0.0))


def _apply_rights(day, state, broker, rows, marks):
    """Recognize rights before opening orders; settle only on actual payment date."""
    for aid, entitlement in state["entitlements"].items():
        action, symbol = entitlement["action"], entitlement["symbol"]
        if day >= action["ex_date"] and not entitlement["recognized"]:
            multiplier = action["share_multiplier"]
            row = rows.get(symbol)
            previous = _number(marks.get(symbol), (row or {}).get("_previous_close"))
            factor = action.get("price_factor")
            if factor is None:
                if previous is None or previous <= action["cash_per_share"]:
                    raise ValueError("Cannot determine causal company-action price adjustment")
                factor = (previous-action["cash_per_share"])/previous/multiplier
            entitlement["price_factor"] = factor
            if symbol in state["risk"]:
                position = state["risk"][symbol]
                for key in ("stop", "initial_r", "initial_cost", "peak"):
                    position[key] *= factor
                position["below_ma"] = 0
            if multiplier > 1 and entitlement["eligible_shares"]:
                current = {l["lot_id"]: l for l in state["lots"][symbol]}
                for original in entitlement["eligible_lots"]:
                    existing = current.get(original["lot_id"])
                    if existing is None or existing["shares"] != original["shares"]:
                        raise ValueError("Share entitlement inventory changed between record and ex-date")
                    extra = original["shares"]*(multiplier-1)
                    if abs(extra-round(extra)) > 1e-8:
                        raise ValueError("Fractional lot shares require explicit cash-in-lieu rules")
                    existing["unit_cost"] = original["unit_cost"]/multiplier
                    entitlement["new_lots"].append(dict(lot_id=aid+":"+original["lot_id"],
                        date=action["shares_pay_date"], shares=int(round(extra)),
                        unit_cost=original["unit_cost"]/multiplier, sellable_from=action["sellable_date"]))
                broker.avg_cost[symbol] = sum(l["shares"]*l["unit_cost"] for l in state["lots"][symbol])/broker.shares(symbol)
            entitlement["recognized"] = True
            if symbol in state["cycle_cash"] and state["cycle_start"].get(symbol) == entitlement["cycle_opened_index"]:
                state["cycle_cash"][symbol] += entitlement["cash_gross"]-entitlement["tax"]
            elif entitlement["eligible_shares"]:
                for cycle in state["cycles"]:
                    if cycle["symbol"] == symbol and cycle.get("opened_index") == entitlement["cycle_opened_index"]:
                        cycle["pnl"] += entitlement["cash_gross"]-entitlement["tax"]
            state["corporate_ledger"].append(dict(date=day, action_id=aid, symbol=symbol,
                phase="ex", receivable=entitlement["cash_gross"], tax_payable=entitlement["tax"],
                share_receivable=entitlement["new_shares"], price_factor=factor,
                source_id=action["source_id"], cash_flow=0.0))
        if not entitlement["recognized"]: continue
        if day >= action["shares_pay_date"] and not entitlement["shares_delivered"]:
            state["lots"][symbol].extend(copy.deepcopy(entitlement["new_lots"]))
            count = entitlement["new_shares"]
            broker.positions[symbol] = broker.shares(symbol)+count
            if broker.shares(symbol):
                broker.avg_cost[symbol] = sum(l["shares"]*l["unit_cost"] for l in state["lots"][symbol])/broker.shares(symbol)
            entitlement["shares_delivered"] = True
            state["corporate_ledger"].append(dict(date=day, action_id=aid, symbol=symbol,
                phase="shares_pay", shares=count, source_id=action["source_id"], cash_flow=0.0))
        if day >= action["pay_date"] and not entitlement["cash_paid"]:
            cash = entitlement["cash_gross"]-entitlement["tax"]
            broker.cash += cash
            entitlement["cash_paid"] = True
            state["corporate_ledger"].append(dict(date=day, action_id=aid, symbol=symbol,
                phase="cash_pay", cash_flow=cash, gross=entitlement["cash_gross"], tax=entitlement["tax"],
                source_id=action["source_id"]))


def simulation_metrics(daily, trades, initial_cash, config=None, completed_cycles=None):
    """Compute account metrics from complete daily equity, including cash-only days."""
    cfg = simulation_config(config)
    equities = np.array([r["equity"] for r in daily], dtype=float)
    previous = np.r_[float(initial_cash), equities[:-1]]
    valid = bool(len(equities) and np.all(np.isfinite(equities)) and np.all(equities > 0) and initial_cash > 0)
    returns = equities / previous - 1 if valid else np.array([])
    annual = cfg["annualization"]
    rf = (1+cfg["risk_free_annual"])**(1/annual)-1
    excess = returns-rf
    reasons = []
    if not valid: reasons.append("nonpositive_or_invalid_equity")
    if not trades: reasons.append("no_fills")
    if len(daily) < max(60, cfg["min_sharpe_sessions"]): reasons.append("fewer_than_60_verified_sessions")
    if any(r.get("stale_symbols") for r in daily): reasons.append("stale_held_valuation")
    std = float(np.std(excess, ddof=1)) if len(excess)>1 else 0.0
    if std <= 1e-14: reasons.append("zero_or_insufficient_variance")
    peaks = np.maximum.accumulate(np.r_[initial_cash, equities])[1:] if valid else np.array([])
    drawdowns = 1-equities/peaks if valid else np.array([])
    total = equities[-1]/initial_cash-1 if valid else None
    years = len(daily)/annual
    growth = (equities[-1]/initial_cash)**(1/years)-1 if valid and years else None
    cycles = completed_cycles or []
    wins = [x["pnl"] for x in cycles if x["pnl"]>0]
    losses = [x["pnl"] for x in cycles if x["pnl"]<0]
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0)**2))) if len(excess) else 0.0
    return _clean(dict(
        initial_cash=float(initial_cash), ending_equity=float(equities[-1]) if valid else None,
        total_return=total, annualized_return=growth, annualized_estimate=len(daily)<annual,
        annualization=annual, risk_free_annual=cfg["risk_free_annual"], risk_free_daily=rf,
        sharpe=None if reasons else float(np.sqrt(annual)*excess.mean()/std),
        sharpe_status="unavailable" if reasons else "research_estimate", sharpe_reasons=reasons,
        annualized_volatility=float(std*np.sqrt(annual)) if len(excess)>1 else None,
        sortino=float(np.sqrt(annual)*excess.mean()/downside) if not reasons and downside>0 else None,
        max_drawdown=float(drawdowns.max()) if len(drawdowns) else None,
        calmar=growth/float(drawdowns.max()) if growth is not None and len(drawdowns) and drawdowns.max()>0 else None,
        trading_sessions=len(daily), daily_returns=returns.tolist(), fills=len(trades),
        total_fees=sum(t["fee"] for t in trades),
        slippage_cost=sum(t.get("slippage_cost", 0) for t in trades),
        turnover=sum(abs(t["price"]*t["shares"]) for t in trades)/float(np.mean(equities)) if valid else None,
        average_exposure=float(np.mean([r["exposure"] for r in daily])) if daily else 0.0,
        maximum_exposure=max([r["exposure"] for r in daily], default=0.0),
        completed_cycles=len(cycles), win_rate=len(wins)/len(cycles) if cycles else None,
        profit_factor=sum(wins)/abs(sum(losses)) if losses else None,
        average_holding_sessions=float(np.mean([x["holding_sessions"] for x in cycles])) if cycles else None,
        open_positions=sum(1 for p in daily[-1]["positions"].values() if p["shares"]>0) if daily else 0,
        valid_valuation_fraction=sum(not r.get("stale_symbols") for r in daily)/len(daily) if daily else None,
        daily_return_autocorrelation=float(pd.Series(returns).autocorr()) if len(returns)>2 and std>1e-14 else None,
        profit_certification=False,
    ))


def run_position_simulation(frames: dict[str, pd.DataFrame], evidence: dict[str, list[dict]], *,
                            config=None, start=None, end=None, calendar=None, state=None,
                            corporate_actions=None, progress=None, check_cancel=None,
                            include_benchmarks=True, risk_only=False, prepared_runtime=None):
    """Replay new sessions only; return JSON-safe results and an immutable checkpoint.

    Evidence uses confirmed=True, side=B/BUY/S/SELL, date, available_at,
    event_id and risk_boundary. Forecast-only observations may omit side.
    Prices can include suspended, is_st, limit_up/down and rules_verified.
    Company actions require explicit record/ex/payment dates and sourced terms;
    missing events are never guessed from price gaps.
    """
    cfg = simulation_config(config)
    passive = cfg["policy"] in {"cash", "buy_hold"}
    if state and start is None: start = state["start"]
    bars, raw, by_evidence, sessions, start = _prepare(frames, evidence, start, end, calendar,prepared_runtime)
    symbols = sorted(bars)
    action_calendar = prepared_runtime.dates if prepared_runtime is not None else sorted(set(_day(d) for d in calendar)) if calendar is not None else sorted({d for b in bars.values() for d in b})
    actions = _corporate_actions(corporate_actions, symbols, action_calendar)
    contract = dict(version=VERSION, config=cfg, symbols=symbols, start=start,
                    calendar_verified=calendar is not None, implementation=_IMPLEMENTATION_HASHES)
    contract_hash = _hash(contract)
    if prepared_runtime is not None:
        prefix_hashes=prepared_runtime.prefixes(by_evidence,sessions,start,actions)
        rows=prepared_runtime.indicators(cfg,actions)
    else:
        prefix = hashlib.sha256()
        prefix_hashes = {}
        # Include warmup and causal evidence, never future observations in earlier identities.
        historical_dates = sorted({r["date"] for rows in raw.values() for r in rows} | set(sessions) |
                                  {a["record_date"] for a in actions} |
                                  {d for mapping in by_evidence.values() for d in mapping})
        for day in historical_dates:
            prefix.update(_hash({"date": day, "bars": {s: bars[s].get(day) for s in symbols},
                                 "evidence": {s: by_evidence[s].get(day, []) for s in symbols},
                                 "corporate_actions": [a for a in actions if a["record_date"] == day],
                                 "account_session": day in sessions}).encode())
            prefix_hashes[day] = prefix.hexdigest()
        rows = {s: _indicators(raw[s], cfg, [a for a in actions if a["symbol"] == s]) for s in symbols}
    if state:
        state = copy.deepcopy(state)
        if state.get("version") != VERSION or state.get("contract_hash") != contract_hash:
            raise ValueError("Simulation configuration/population identity changed; create a new branch")
        last = state["last_date"]
        if state["input_prefix_hash"] != prefix_hashes.get(last):
            raise ValueError("Historical price/evidence/calendar prefix changed; create a new branch")
        if last > sessions[-1]: raise ValueError("Cannot resume before the checkpoint date")
    else:
        state = dict(version=VERSION, contract_hash=contract_hash, start=start, last_date=None,
            cash=cfg["initial_cash"], positions={}, avg_cost={}, ledger=[], trade_details=[],
            lots={s: [] for s in symbols}, risk={}, pending=[], orders=[], decisions=[], daily=[],
            marks={}, consumed=[], cooldown={}, cycles=[], cycle_cash={}, cycle_start={},
            active_candidates={}, opportunities=[], agent_evidence=[], risk_evidence=[],
            entitlements={}, corporate_ledger=[],
            peak=cfg["initial_cash"], risk_peak=cfg["initial_cash"], dd_stage=0,
            dd_recovery=0, dd_exit_completed_index=None)
    state.setdefault("active_candidates", {})
    state.setdefault("opportunities", [])
    state.setdefault("agent_evidence", [])
    state.setdefault("risk_evidence", [])
    broker = BrokerSimulator(initial_cash=cfg["initial_cash"], run_id=contract_hash[:24])
    broker.cash = state["cash"]
    broker.positions = state["positions"]
    broker.avg_cost = state["avg_cost"]
    broker.ledger.entries = [LedgerEntry(**item) for item in state["ledger"]]
    consumed = set(state["consumed"])
    orders = {o["order_id"]: o for o in state["orders"]}
    lots, risk, marks = state["lots"], state["risk"], state["marks"]

    def decide(day, symbol, action, reason, **extra):
        entry = dict(date=day, symbol=symbol, action=action, reason=reason, **extra)
        state["decisions"].append(entry)
        return entry

    def cancel_buys(symbol, day, reason):
        for oid in list(state["pending"]):
            order = orders[oid]
            if order["symbol"] == symbol and order["side"] == "BUY":
                order.update(status="cancelled", terminal_date=day, terminal_reason=reason)
                state["pending"].remove(oid)

    def sell(day, symbol, quantity, reason, *, immediate=False, hard=False):
        cancel_buys(symbol, day, reason)
        if hard and _pending_rights(state, symbol) and symbol in risk:
            risk[symbol]["exit_pending_rights"] = True
        active = [orders[i] for i in state["pending"] if orders[i]["symbol"]==symbol and orders[i]["side"]=="SELL"]
        if active:
            if hard:
                existing = active[0]
                existing.update(remaining=broker.shares(symbol), requested_shares=existing["filled_shares"]+broker.shares(symbol),
                                reason=reason, hard=True, target_weight=0.0, target_shares=0,
                                requested_share_delta=-broker.shares(symbol),
                                quantity_plan=_quantity_plan(broker.shares(symbol), -broker.shares(symbol),
                                    marks.get(symbol), _equity(broker,state,marks)))
            return
        if quantity <= 0:
            if broker.shares(symbol): decide(day,symbol,"SKIP","reduction_below_sell_lot",trigger=reason)
            return
        before = broker.shares(symbol)
        signal_marks = dict(opening_marks) if immediate else dict(marks)
        if immediate and _valid_price(rows[symbol].get(day), "open"):
            signal_marks[symbol] = float(rows[symbol][day]["open"])
        signal_equity = _equity(broker, state, signal_marks)
        plan = _quantity_plan(before, -int(quantity), signal_marks.get(symbol), signal_equity)
        ordinal = len(state["orders"])
        order = dict(order_id=_hash([contract_hash, day, symbol, "SELL", ordinal])[:24], symbol=symbol,
            side="SELL", signal_date=day, execute_after=day, immediate=immediate,
            remaining=int(quantity), requested_shares=int(quantity), filled_shares=0,
            reason=reason, hard=hard, status="queued", attempts=[],
            target_weight=0.0 if hard else plan["target_weight"], quantity_plan=plan,
            requested_share_delta=-int(quantity), target_shares=before-int(quantity))
        orders[order["order_id"]] = order
        state["orders"].append(order)
        state["pending"].append(order["order_id"])
        decide(day, symbol, "EXIT" if hard else "REDUCE", reason, order_id=order["order_id"], requested_shares=quantity)

    for day in sessions:
        if state["last_date"] and day <= state["last_date"]: continue
        if check_cancel and check_cancel(): raise InterruptedError("Position simulation cancelled")
        index = len(state["daily"])
        day_rows = {s: rows[s].get(day) for s in symbols}
        _apply_rights(day, state, broker, day_rows, marks)
        opening_marks = dict(marks)
        for symbol, row in day_rows.items():
            if _valid_price(row, "open"): opening_marks[symbol] = float(row["open"])
        for symbol in symbols:
            row = day_rows[symbol]
            if broker.shares(symbol) and risk.get(symbol, {}).get("exit_pending_rights"):
                sell(day, symbol, broker.shares(symbol), "previous_exit_delivered_entitlement", immediate=True, hard=True)
            if not passive and broker.shares(symbol) and _valid_price(row, "open") and row["open"] <= risk[symbol]["stop"]:
                sell(day, symbol, broker.shares(symbol), "frozen_stop_gap", immediate=True, hard=True)
        # Previous-close orders execute sells first with fixed symbol ordering.
        eligible = [orders[oid] for oid in list(state["pending"])
                    if orders[oid]["execute_after"] < day or orders[oid].get("immediate")]
        eligible.sort(key=lambda o: (o["side"]=="BUY", o["symbol"], o["order_id"]))
        for order in eligible:
            symbol, side = order["symbol"], order["side"]
            row = day_rows[symbol]
            costs = _costs(cfg, day)
            amount = row.get("_amount_ma20") if row else None
            reference = float(row["open"]) if _valid_price(row, "open") else 0.0
            price = costs.execution_price(side, reference, amount_ma20=amount)
            failure = _rule_failure(symbol, row, side, cfg, price)
            if side == "BUY" and order.get("decision_available_at"):
                available = pd.Timestamp(order["decision_available_at"])
                available = available.tz_localize("Asia/Shanghai") if available.tzinfo is None else available.tz_convert("Asia/Shanghai")
                if available >= pd.Timestamp(day+"T09:30:00+08:00"):
                    failure = "decision_unavailable_at_executable_open"
            quantity = 0
            equity = _equity(broker, state, opening_marks)
            participation = int((_number((row or {}).get("_previous_volume"), 0) * cfg["max_volume_participation"]) // 100)*100
            if not failure and participation <= 0: failure = "prior_session_liquidity_unavailable"
            if not failure and side == "SELL":
                quantity = min(order["remaining"], broker.shares(symbol), _sellable(lots[symbol], day), participation)
                if quantity <= 0: failure = "t_plus_one_locked"
                elif quantity < broker.shares(symbol): quantity = quantity//100*100
                if quantity <= 0: failure = "below_sell_lot"
            if not failure and side == "BUY":
                boundary = order["risk_boundary"]
                if boundary >= price: failure = "opening_price_below_entry_risk_boundary"
                else:
                    held_value = sum(broker.shares(s)*opening_marks[s] for s in symbols if broker.shares(s)) + _entitlements_value(state, opening_marks)[2]
                    cash_budget = min(order["cash_budget"], broker.cash,
                        max(0, equity*cfg["max_total_weight"]-held_value),
                        max(0, equity*cfg["max_stock_weight"]-broker.shares(symbol)*price))
                    effective_stop = max(boundary, price*(1-cfg["stop_loss"]))
                    existing_risk = broker.shares(symbol)*max(0, price-risk.get(symbol, {}).get("stop", effective_stop))
                    risk_budget = max(0, equity*cfg["risk_budget"]-existing_risk)
                    quantity = min(int(cash_budget/price)//100*100, participation,
                                   int(risk_budget/(price-effective_stop))//100*100 if not passive else participation)
                    while quantity > 0:
                        fee = costs.fees(side, quantity*price)
                        unit = price+fee/quantity
                        new_risk = quantity*(unit-max(boundary, unit*(1-cfg["stop_loss"])))
                        if quantity*price+fee <= cash_budget+1e-8 and (passive or new_risk <= risk_budget+1e-8): break
                        quantity -= 100
                    if quantity <= 0: failure = "cash_risk_or_lot_budget_insufficient"
            if failure:
                order["attempts"].append(dict(date=day, status="rejected", reason=failure))
                if side == "BUY":
                    order.update(status="expired", terminal_date=day, terminal_reason=failure)
                    state["pending"].remove(order["order_id"])
                else: order["status"] = "partial" if order["filled_shares"] else "queued"
                continue
            broker.cost_model = costs
            before = broker.shares(symbol)
            actual = Order(date=day, symbol=symbol, side=side, shares=quantity, price=price,
                reason=order["reason"], target_weight=order.get("target_weight"),
                signal_date=order["signal_date"], applied_slippage=costs.resolve_slippage(amount),
                cost_tier=costs.resolve_tier_index(amount))
            entry = broker.submit(actual, mark_price=reference, equity_before=equity,
                                  portfolio_prices=opening_marks)
            if entry is None: raise AssertionError("Validated order was rejected by BrokerSimulator")
            if side == "BUY":
                unit_cost = -entry.cash_flow/quantity
                lots[symbol].append(dict(lot_id=order["order_id"], date=day, shares=quantity, unit_cost=unit_cost))
                stop = max(order["risk_boundary"], unit_cost*(1-cfg["stop_loss"]))
                if before == 0:
                    risk[symbol] = dict(stop=stop, initial_r=unit_cost-stop, initial_cost=unit_cost,
                        peak=unit_cost, opened_index=index, last_buy_index=index, below_ma=0,
                        forecast_weak_closes=0, forecast_reduced=False, target_weight=order["target_weight"])
                    state["cycle_start"][symbol] = index
                    state["cycle_cash"][symbol] = 0.0
                else:
                    risk[symbol].update(stop=max(risk[symbol]["stop"], stop), last_buy_index=index,
                                        target_weight=order["target_weight"])
                matched, realized = [], None
            else:
                matched = _fifo(lots[symbol], quantity, day)
                realized = entry.cash_flow-sum(m["basis"] for m in matched)
                remaining = broker.shares(symbol)
                if remaining:
                    broker.avg_cost[symbol] = sum(l["shares"]*l["unit_cost"] for l in lots[symbol])/remaining
                    risk[symbol]["target_weight"] = remaining*price/max(equity, 1e-12)
                else:
                    broker.avg_cost.pop(symbol, None)
                    if not _pending_rights(state, symbol):
                        state["cooldown"][symbol] = index+cfg["cooldown"]
                        risk.pop(symbol, None)
                entry.holding_cost_after = broker.avg_cost.get(symbol, 0.0)
            entry.equity_after = _equity(broker, state, opening_marks)
            state["cycle_cash"][symbol] = state["cycle_cash"].get(symbol, 0)+entry.cash_flow
            if side == "SELL" and broker.shares(symbol)==0 and not _pending_rights(state, symbol):
                state["cycles"].append(dict(symbol=symbol, exit_date=day,
                    opened_index=state["cycle_start"][symbol], pnl=state["cycle_cash"].pop(symbol),
                    holding_sessions=index-state["cycle_start"].pop(symbol)+1))
            notional = quantity*price
            state["trade_details"].append(dict(order_id=order["order_id"], fifo=matched, realized_pnl=realized,
                reference_open=reference, slippage_cost=abs(price-reference)*quantity,
                commission=max(notional*costs.commission, costs.min_commission),
                stamp_tax=notional*costs.stamp_tax if side=="SELL" else 0,
                transfer_fee=notional*costs.transfer_rate,
                share_delta=quantity if side=="BUY" else -quantity,
                execution_change=dict(share_delta=quantity if side=="BUY" else -quantity,
                    shares_before=before, shares_after=broker.shares(symbol), valuation_price=reference,
                    equity_before=equity, equity_after=entry.equity_after,
                    weight_before=before*reference/equity if equity>0 else None,
                    weight_after=broker.shares(symbol)*reference/entry.equity_after if entry.equity_after>0 else None,
                    trade_weight_delta=(quantity if side=="BUY" else -quantity)*reference/equity if equity>0 else None,
                    denominator="execution_equity", valuation="same_open_reference_price")))
            order["filled_shares"] += quantity
            order["remaining"] = max(0, order.get("remaining", quantity)-quantity)
            order["attempts"].append(dict(date=day, status="filled", shares=quantity, price=price))
            if side == "BUY" or not order["remaining"] or broker.shares(symbol)==0:
                order.update(status="filled" if side=="SELL" or quantity==order.get("requested_shares", quantity) else "partial",
                             terminal_date=day)
                state["pending"].remove(order["order_id"])
            else: order["status"] = "partial"
        stale = []
        for symbol, row in day_rows.items():
            if _valid_price(row): marks[symbol] = float(row["close"])
            elif broker.shares(symbol) or _pending_rights(state, symbol): stale.append(symbol)
        equity = _equity(broker, state, marks)
        state["peak"] = max(state["peak"], equity)
        state["risk_peak"] = max(state["risk_peak"], equity)
        risk_dd = 1-equity/state["risk_peak"]
        desired_stage = 0 if passive else 2 if risk_dd >= cfg["dd_exit"] else 1 if risk_dd >= cfg["dd_reduce"] else 0
        if desired_stage > state["dd_stage"]:
            state["dd_stage"] = desired_stage
            state["dd_recovery"] = 0
            for symbol in symbols:
                count = broker.shares(symbol)
                sell(day, symbol, count if desired_stage==2 else count//200*100,
                     "portfolio_drawdown_exit" if desired_stage==2 else "portfolio_drawdown_reduce", hard=desired_stage==2)
        if state["dd_stage"] == 2 and not any(broker.positions.values()) and not any(_pending_rights(state,s) for s in symbols):
            if state["dd_exit_completed_index"] is None: state["dd_exit_completed_index"] = index
            if index-state["dd_exit_completed_index"] >= cfg["dd_cooldown"]:
                state.update(dd_stage=0, dd_recovery=0, risk_peak=equity, dd_exit_completed_index=None)
        elif state["dd_stage"] == 1:
            state["dd_recovery"] = state["dd_recovery"]+1 if risk_dd <= cfg["dd_rearm"] else 0
            if state["dd_recovery"] >= cfg["dd_rearm_sessions"]:
                state.update(dd_stage=0, dd_recovery=0)
        for symbol in symbols:
            row = day_rows[symbol]
            events = by_evidence[symbol].get(day, [])
            new = [e for e in events if e.get("confirmed") is True and e["event_id"] not in consumed
                   and e.get("side", "").upper() in {"B", "BUY", "S", "SELL"}]
            consumed.update(e["event_id"] for e in new if e.get("side", "").upper() in {"S", "SELL"})
            for e in new:
                if e.get("side", "").upper() in {"B", "BUY"}:
                    state["active_candidates"].setdefault(e["event_id"],dict(event=copy.deepcopy(e),
                        symbol=symbol, created_index=index, expires_index=index+cfg["candidate_sessions"]-1))
            held = broker.shares(symbol)
            active_sells = [orders[i] for i in state["pending"] if orders[i]["symbol"]==symbol and orders[i]["side"]=="SELL"]
            if not passive and held and not _valid_price(row):
                risk[symbol]["below_ma"] = 0
                if index-risk[symbol]["opened_index"]+1 >= cfg["max_holding_sessions"]:
                    sell(day, symbol, held, "holding_expiry_missing_quote", hard=True)
                    active_sells = [True]
            if not passive and held and _valid_price(row):
                position = risk[symbol]
                close = float(row["close"])
                age = index-position["opened_index"]+1
                ma, atr = _number(row.get("_ma")), _number(row.get("_atr"))
                position["below_ma"] = position["below_ma"]+1 if ma and close < ma else 0
                position["peak"] = max(position["peak"], close)
                reason = None
                if close <= position["stop"]: reason = "frozen_price_stop"
                elif position["below_ma"] >= cfg["ma_exit_closes"]: reason = "ma_trend_failure"
                elif age >= cfg["max_holding_sessions"]: reason = "holding_expiry"
                elif (age >= cfg["time_exit_sessions"] and position["peak"]-position["initial_cost"] < position["initial_r"]
                      and held*close-sum(l["shares"]*l["unit_cost"] for l in lots[symbol])-_costs(cfg,day).fees("SELL",held*close) <= 0):
                    reason = "time_failure_without_one_r"
                if reason:
                    sell(day, symbol, held, reason, hard=True)
                    active_sells = [True]
                elif not active_sells and any(e["side"].upper() in {"S", "SELL"} for e in new):
                    sell(day, symbol, held//200*100, "confirmed_structure_sell")
                    active_sells = [True]
                if cfg["policy"] in {"structure_kronos","structure_kronos_score","structure_kronos_l2"}:
                    forecasts = [e["forecast"] for e in events if e.get("forecast") and
                        (e["forecast"].get("quality") or {}).get("prediction_usable") is True]
                    latest = (forecasts[-1].get("summary") or {}) if forecasts else {}
                    direction = _number(latest.get("terminal_return_median",latest.get("expected_return")))
                    fraction_up = _number(latest.get("up_path_fraction",latest.get("up_fraction")))
                    weak = direction is not None and fraction_up is not None and (direction < 0 or fraction_up < cfg["min_up_fraction"])
                    observed_score=next((e.get('score') for e in reversed(events) if e.get('score')),None) or {}
                    if cfg['policy']=='structure_kronos_l2':observed_score=observed_score.get('l2_comparison') or {}
                    fused_b=(cfg['policy'] in {'structure_kronos_score','structure_kronos_l2'}
                        and cfg['fusion_mode']=='evidence' and observed_score.get('variant')=='B'
                        and observed_score.get('consumes_kronos') is True)
                    if fused_b:
                        value=_number(observed_score.get('score'))
                        weak=observed_score.get('status')=='scored' and value is not None and value<cfg['min_score']
                    position["forecast_weak_closes"] = position.get("forecast_weak_closes",0)+1 if weak else 0
                    if forecasts and not weak: position["forecast_reduced"] = False
                    if (not reason and not active_sells and not position.get("forecast_reduced")
                            and position["forecast_weak_closes"] >= cfg["forecast_exit_closes"]):
                        sell(day,symbol,held//200*100,"consecutive_fusion_weakness" if fused_b else "consecutive_forecast_weakness")
                        position["forecast_reduced"] = True
                        active_sells = [True]
                # The close-updated protection is first considered at tomorrow's open.
                if atr and position["peak"]-position["initial_cost"] >= position["initial_r"]:
                    position["stop"] = max(position["stop"], position["peak"]-cfg["trailing_atr"]*atr)
            buys = []
            if not passive and not risk_only:
                latest_observation = next((e for e in reversed(events) if e.get("observation_kind") == "daily_agent_observation"), None)
                for eid, candidate in list(state["active_candidates"].items()):
                    if candidate["symbol"] != symbol: continue
                    if eid in consumed or index > candidate["expires_index"]:
                        state["active_candidates"].pop(eid, None); continue
                    e = copy.deepcopy(candidate["event"])
                    e["candidate_age_sessions"] = index-candidate["created_index"]
                    if latest_observation:
                        for field in ("forecast", "score", "forecast_status"):
                            if field in latest_observation: e[field] = latest_observation[field]
                    buys.append(e)
            if cfg["policy"] == "buy_hold" and index == 0 and _valid_price(row):
                # An explicit benchmark allocation, never a claimed native source B.
                buys = [dict(event_id="baseline_initial_allocation:"+symbol,
                    risk_boundary=row["close"]*.001, source="benchmark_allocation")]
            if not buys: continue
            event = copy.deepcopy(buys[-1])
            if cfg["policy"] == "structure_kronos_l2":
                event["score"] = (event.get("score") or {}).get("l2_comparison") or {"status":"unavailable"}
            reasons = []
            if active_sells: reasons.append("exit_pending")
            if state["dd_stage"]: reasons.append("portfolio_drawdown_buy_block")
            if index < state["cooldown"].get(symbol, -1): reasons.append("reentry_cooldown")
            if not _valid_price(row): reasons.append("missing_close")
            if not held and sum(bool(v) for v in broker.positions.values()) + sum(orders[i]["side"]=="BUY" for i in state["pending"]) >= cfg["max_positions"]:
                reasons.append("position_count_limit")
            if any(orders[i]["symbol"]==symbol and orders[i]["side"]=="BUY" for i in state["pending"]):
                reasons.append("entry_already_pending")
            trend_ready = bool(row and row.get("_ma") is not None and row.get("_previous_ma") is not None
                               and row["close"] > row["_ma"] >= row["_previous_ma"])
            if not passive and not trend_ready and (cfg["fusion_mode"] == "hard_filters" or cfg["policy"] == "structure_rules"):
                reasons.append("structure_entry_trend_not_ready")
            boundary = _number(event.get("risk_boundary", event.get("stop_price")))
            if boundary is None or boundary <= 0 or not _valid_price(row) or boundary >= row["close"]:
                reasons.append("missing_or_invalid_structure_risk_boundary")
                if boundary and row and _valid_price(row) and row["close"] <= boundary:
                    consumed.update(e["event_id"] for e in buys)
            if held and (not _valid_price(row) or row["close"] <= broker.avg_cost[symbol] or index-risk[symbol]["last_buy_index"] < cfg["add_interval"]):
                reasons.append("add_requires_profit_and_interval")
            reasons.extend(_forecast_reasons(event, cfg))
            current_weight = held*marks.get(symbol, 0)/equity if equity > 0 else 0
            fraction = min(cfg["initial_weight"] if not held else cfg["position_step"],
                           max(0, cfg["max_stock_weight"]-current_weight))
            if fraction < cfg["position_step"]-1e-12: reasons.append("below_position_step")
            reasons = list(dict.fromkeys(reasons))
            from my_strategy.services.agent_evidence import agents_for
            agents = agents_for(row, event, day, held=held, cash=broker.cash, equity=equity,
                                hard_risk="exit_pending" if active_sells else None)
            oid = _hash([symbol, sorted(e["event_id"] for e in buys)])[:24]
            opportunity = dict(opportunity_id=oid, decision_id=_hash([oid,day,cfg["policy"]])[:24],
                date=day,symbol=symbol,source_event_ids=[e["event_id"] for e in buys],
                candidate_age_sessions=event.get("candidate_age_sessions",0),
                primary_reason=reasons[0] if reasons else None,all_reasons=reasons,
                qualification_mode=cfg["research_mode"],agents=agents)
            state["opportunities"].append(opportunity)
            if cfg["enable_agent_evidence"]: state["agent_evidence"].append(opportunity)
            if reasons:
                decide(day, symbol, "SKIP", reasons[0], event_id=event["event_id"],
                       all_reasons=reasons, decision_id=opportunity["decision_id"])
                if cfg["candidate_sessions"] == 1: consumed.update(e["event_id"] for e in buys)
                continue
            consumed.update(e["event_id"] for e in buys)
            cash_budget = equity*fraction
            order = dict(order_id=_hash([contract_hash, day, symbol, "BUY", event["event_id"]])[:24],
                symbol=symbol, side="BUY", signal_date=day, execute_after=day, immediate=False,
                cash_budget=cash_budget, risk_boundary=boundary, target_weight=current_weight+fraction,
                reason="benchmark_initial_allocation" if passive else "confirmed_structure_entry" if not held else "confirmed_profitable_add",
                event_id=event["event_id"], status="queued", filled_shares=0, attempts=[],
                decision_available_at=(event.get("forecast") or {}).get("actual_available_at")
                    if cfg['policy'] in {'structure_kronos','structure_kronos_score','structure_kronos_l2'} else None,
                decision_id=opportunity["decision_id"],
                quantity_plan=_quantity_plan(held, int(cash_budget/row["close"])//100*100,
                                             row["close"], equity))
            orders[order["order_id"]] = order
            state["orders"].append(order)
            state["pending"].append(order["order_id"])
            decide(day, symbol, "BUY" if not held else "ADD", order["reason"], order_id=order["order_id"],
                   event_id=event["event_id"], target_weight=order["target_weight"],
                   forecast_identity=(event.get("forecast") or {}).get("identity"))
        if cfg["enable_agent_evidence"]:
            from my_strategy.services.agent_evidence import observation
            for symbol in symbols:
                outstanding = [orders[i] for i in state["pending"] if orders[i]["symbol"]==symbol and orders[i]["side"]=="SELL"]
                hard = next((o["reason"] for o in outstanding if o.get("hard")), None)
                state["risk_evidence"].append(observation("risk",day,source_id=_hash([day,symbol]),
                    direction="SELL" if outstanding else "HOLD" if broker.shares(symbol) else "FLAT",
                    quality={"shares":broker.shares(symbol),"sellable_shares":_sellable(lots[symbol],day),
                        "stop_price":risk.get(symbol,{}).get("stop"),"hard_risk":hard,
                        "pending_exit":bool(outstanding),"checked_independently_of_models":True,
                        "cash":broker.cash,"equity":equity},symbol=symbol))
        _record_rights(actions, day, state)
        positions = {}
        for symbol in symbols:
            held = broker.shares(symbol)
            pending_symbol = [orders[i] for i in state["pending"] if orders[i]["symbol"] == symbol]
            exits = [o for o in pending_symbol if o["side"] == "SELL"]
            entries = [o for o in pending_symbol if o["side"] == "BUY"]
            target = risk.get(symbol, {}).get("target_weight", 0)
            if exits:
                target = 0.0 if any(o.get("hard") for o in exits) else max(0,held-exits[0]["remaining"])*marks.get(symbol,0)/equity
            elif entries: target = entries[-1]["target_weight"]
            status = ("EXIT_PENDING" if any(o.get("hard") for o in exits) else "REDUCE_PENDING") if exits else \
                "BUY_PENDING" if entries else "HOLDING" if held or _pending_rights(state,symbol) else \
                "COOLDOWN" if index < state["cooldown"].get(symbol,-1) else "FLAT"
            positions[symbol] = dict(shares=held, sellable_shares=_sellable(lots[symbol], day),
                state=status,
                locked_shares=held-_sellable(lots[symbol], day), average_cost=broker.avg_cost.get(symbol, 0),
                mark=marks.get(symbol), value=held*marks.get(symbol, 0),
                weight=held*marks.get(symbol, 0)/equity if equity>0 else None,
                target_weight=target,
                receivable_shares=_pending_rights(state, symbol),
                stop_price=risk.get(symbol, {}).get("stop"),
                pending_exit=any(orders[i]["symbol"]==symbol and orders[i]["side"]=="SELL" for i in state["pending"]))
        if broker.cash < -1e-7 or any(v<0 for v in broker.positions.values()): raise AssertionError("Negative account state")
        if any(sum(l["shares"] for l in lots[s]) != broker.shares(s) for s in symbols): raise AssertionError("Lot inventory mismatch")
        receivable, tax_payable, receivable_stock_value = _entitlements_value(state, marks)
        market_value = sum(broker.shares(s)*marks.get(s, 0) for s in symbols)
        reserved = min(broker.cash, sum(orders[i].get("cash_budget", 0) for i in state["pending"] if orders[i]["side"]=="BUY"))
        state["daily"].append(dict(date=day, cash=broker.cash, equity=equity,
            reserved_cash=reserved, available_cash=broker.cash-reserved,
            receivable_cash=receivable, tax_payable=tax_payable, receivable_stock_value=receivable_stock_value,
            net_value=equity/cfg["initial_cash"], market_value=market_value,
            stock_market_value=market_value,
            exposure=(market_value+receivable_stock_value)/equity if equity>0 else None,
            drawdown=1-equity/state["peak"], risk_cycle_drawdown=1-equity/state["risk_peak"],
            portfolio_risk_stage=state["dd_stage"], stale_symbols=stale, positions=positions,
            input_prefix_hash=prefix_hashes[day]))
        state["last_date"] = day
        if progress: progress("simulation", index+1, len(sessions), 0, date=day)
    state.update(cash=broker.cash, positions=broker.positions, avg_cost=broker.avg_cost,
        ledger=[entry.to_dict() for entry in broker.ledger.entries], consumed=sorted(consumed),
        input_prefix_hash=prefix_hashes[state["last_date"]])
    # The trade table derives exclusively from the shared Broker Ledger.
    ledger = broker.ledger.to_frame().to_dict("records")
    trades = [{**record, **detail} for record, detail in zip(ledger, state["trade_details"])]
    metrics = simulation_metrics(state["daily"], trades, cfg["initial_cash"], cfg, state["cycles"])
    metrics.update(rejected_attempts=sum(a["status"]=="rejected" for o in state["orders"] for a in o["attempts"]),
        pending_exits=sum(orders[i]["side"]=="SELL" for i in state["pending"]))
    lineage = dict(**contract, input_prefix_hash=state["input_prefix_hash"],
        identity=_hash([contract_hash, state["input_prefix_hash"], state["last_date"]]),
        account_scope="shared_cash_account", execution="next_verified_open_with_declared_slippage",
        source_events="caller_supplied_confirmed_available_at", profit_certification=False,
        warnings=["Historical research only; model pretraining and forward performance are not certified.",
                  "Company-action coverage is unverified; unadjusted price returns are not certified total returns."
                    if not cfg["company_actions_verified"] else "Caller supplied company-action coverage; explicit rights/payment assumptions are recorded.",
                  "Missing historical execution limits use declared main-board research assumptions."
                    if not cfg["require_verified_rules"] else "Execution requires supplied verified dated limits.",
                  "Calendar inferred from observations; missing market sessions cannot be identified."
                    if calendar is None else "Caller supplied the session calendar.",
                  "Daily-bar opening fills are a research approximation, not verified historical order fills."])
    result = dict(version=VERSION, status="completed", policy=cfg["policy"], daily=state["daily"],
        trades=trades, orders=state["orders"], decisions=state["decisions"], metrics=metrics,
        corporate_ledger=state["corporate_ledger"], opportunities=state["opportunities"],
        agent_evidence=state["agent_evidence"], risk_evidence=state["risk_evidence"], qualification_mode=cfg["research_mode"],
        lineage=lineage, state=state, profit_certification=False, research_only=True)
    from my_strategy.services.agent_evidence import opportunity_funnel
    result["opportunity_funnel"] = opportunity_funnel(state["opportunities"],state["orders"],trades,state["cycles"])
    if include_benchmarks and not passive:
        benchmarks = {}
        for label, weight in (("cash", 0), ("buy_hold_initial_budget", min(cfg["initial_weight"], cfg["max_total_weight"]/len(symbols))),
                              ("buy_hold_full_budget", 1/len(symbols))):
            if check_cancel and check_cancel(): raise InterruptedError("Position simulation cancelled")
            benchmark_cfg = {**cfg, "policy": "cash" if label=="cash" else "buy_hold"}
            if weight:
                benchmark_cfg.update(initial_weight=weight, max_stock_weight=weight,
                    max_total_weight=min(1, weight*len(symbols)), position_step=min(cfg["position_step"], weight),
                    max_positions=max(cfg["max_positions"],len(symbols)))
            baseline = run_position_simulation(frames, {}, config=benchmark_cfg, start=start, end=state["last_date"],
                calendar=calendar, corporate_actions=corporate_actions, include_benchmarks=False, check_cancel=check_cancel,
                prepared_runtime=prepared_runtime)
            benchmarks[label] = dict(policy=benchmark_cfg["policy"], initial_weight_per_symbol=weight,
                initial_cash=cfg["initial_cash"], metrics=baseline["metrics"],
                daily=[{k:d[k] for k in ("date","equity","net_value","drawdown","exposure","cash")}
                       for d in baseline["daily"]],
                timing="first_account_close_allocation_next_executable_open_one_attempt",
                lineage_identity=baseline["lineage"]["identity"], profit_certification=False)
        result["benchmarks"] = benchmarks
    return _clean(result)


def continue_position_simulation(frames, evidence, *, state, **kwargs):
    return run_position_simulation(frames, evidence, state=state, **kwargs)
