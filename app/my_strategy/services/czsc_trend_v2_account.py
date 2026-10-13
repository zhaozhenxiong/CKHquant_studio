"""One causal Broker account for consensus entry and independent exits.

Ordinary sells use qualified continuous sell evidence while actually held. Hard risk and the
signal-pinned MA holding model have separate, audited exit channels. There is
no old source-union exit and no theoretical inventory in this adapter.
"""
from __future__ import annotations

import copy
import inspect
import hashlib
import math
from collections.abc import Mapping

import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.execution.broker import BrokerSimulator
from my_strategy.execution.cost_model import CostModel
from my_strategy.execution.ledger import LedgerEntry
from my_strategy.execution.order import Order
from my_strategy.services.czsc_backtest import _guard, _planned_entry_rejection
from my_strategy.services.czsc_trend_v2_contracts import STRATEGY_ID, DISPLAY_NAME, strategy_config
from my_strategy.services.czsc_trend_v2_consensus import prepare_trend_decisions, finite, day, observable, _identity_reasons, _strategy_config, _risk_plan


def prepare_account_context(frame, market_dates):
    """Read-only quote/calendar snapshot reusable by many causal label clones."""
    keys = ("date", "symbol", "open", "high", "low", "close", "volume", "amount", "source", "has_trade_price")
    raw = frame[list(key for key in keys if key in frame)].to_dict("records") if isinstance(frame, pd.DataFrame) else copy.deepcopy(list(frame))
    records = [{key: row.get(key) for key in keys} for row in raw]
    for row in records:
        row["date"] = day(row["date"])
    calendar = list(market_dates)
    dates = [row["date"] for row in records]
    if dates != sorted(set(dates)) or calendar != sorted(set(calendar)) or not set(dates).issubset(set(calendar)):
        raise ValueError("prepared account quote/calendar alignment mismatch")
    by_day = {row["date"]: row for row in records}
    digest = hashlib.sha256(); prefix_hashes = {}
    for session in calendar:
        digest.update(stable_hash(by_day.get(session, {"missing_verified_session": session})).encode("ascii"))
        prefix_hashes[session] = digest.hexdigest()
    return {"frame_object": frame, "calendar": calendar, "records": records, "dates": dates,
        "by_day": by_day, "prefix_hashes": prefix_hashes}


def candidate_from_row(opinion):
    """The one training/runtime candidate adapter, before any quality scoring."""
    plan = copy.deepcopy(opinion.get("entry_plan") or opinion.get("raw_risk_plan"))
    ma = (opinion.get("agents") or {}).get("ma") or {}
    identifiers = copy.deepcopy(opinion.get("fresh_candidate_ids") or (plan or {}).get("source_event_ids") or [])
    originals = sorted(set(opinion.get("original_opportunity_ids") or (plan or {}).get("original_opportunity_ids") or
        [event.get("opportunity_id") or event.get("original_event_id") or event.get("event_id")
         for event in opinion.get("active_candidates", []) if event.get("side") == "BUY"] or identifiers))
    if plan is not None:
        plan.setdefault("original_opportunity_ids", originals)
    return {"date": opinion["date"], "symbol": opinion["symbol"], "entry_plan": plan, "raw_risk_plan": plan,
        "candidate_event_ids": identifiers, "source_event_ids": identifiers,
        "original_opportunity_ids": originals,
        "economic_group_id": stable_hash({"family": STRATEGY_ID, "symbol": opinion["symbol"], "original_opportunity_ids": originals}),
        "sources": copy.deepcopy((plan or {}).get("sources") or sorted({e["source"] for e in opinion.get("active_candidates", []) if e.get("side") == "BUY"})),
        "trend_probability": {"p_up": ma.get("p_up"), "p_down": ma.get("p_down"), "p_neutral": ma.get("p_neutral"),
            "model_available": ma.get("status") == "scored", "input_eligible": ma.get("status") == "scored",
            "model_identity": copy.deepcopy(ma.get("model_identity"))},
        "p_up": ma.get("p_up"), "p_down": ma.get("p_down"), "p_neutral": ma.get("p_neutral"),
        "reference_price": opinion.get("reference_price"), "stop_price": (plan or {}).get("stop_price"),
        "candidate_id": stable_hash({"plan_id": (plan or {}).get("plan_id"), "events": sorted(identifiers), "date": opinion["date"]}),
        "candidate_policy": "valid_original_opportunity_daily_reassessment_v2", "consensus": copy.deepcopy(opinion)}


def _unreserved_opinion(opinion, reserved, bar, index, bars, cfg):
    """A used source cannot lend its frozen stop to a different new opportunity."""
    candidate = candidate_from_row(opinion)
    originals = set(candidate["original_opportunity_ids"])
    if not originals or originals.issubset(reserved):
        return None
    if not originals.intersection(reserved):
        return opinion
    active = [event for event in opinion.get("active_candidates", []) if event.get("side") == "BUY"
        and (event.get("opportunity_id") or event.get("original_event_id") or event.get("event_id")) not in reserved]
    if not active:
        return None
    current = copy.deepcopy(opinion)
    quote = {key: value for key, value in bar.items() if key not in {"entry_plan", "raw_risk_plan"}}
    current["entry_plan"] = current["raw_risk_plan"] = _risk_plan(quote, active, index, bars, cfg)
    current["active_candidates"] = active
    current["fresh_candidate_ids"] = [event["event_id"] for event in active]
    current["original_opportunity_ids"] = current["entry_plan"]["original_opportunity_ids"]
    return current


def _call(callback, snapshot, candidate=None):
    if callback is None:
        return {}
    state = copy.deepcopy(snapshot)
    if candidate is not None:
        state["candidate"] = copy.deepcopy(candidate)
    signature = inspect.signature(callback)
    params = list(signature.parameters.values())
    positional = [p for p in params if p.kind in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}]
    if candidate is not None and (len(positional) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in params)):
        result = callback(state, copy.deepcopy(candidate))
    elif candidate is not None and "candidate" in signature.parameters:
        result = callback(state, candidate=copy.deepcopy(candidate))
    else:
        result = callback(state)
    if result is None:
        return {}
    if not isinstance(result, Mapping):
        raise ValueError("account evaluator must return an explicit mapping")
    return copy.deepcopy(dict(result))


def _settings(cfg, initial_cash):
    a = cfg["account"]
    execution = {"lot_size": a["lot_size"], "max_weight": a["max_weight"], "seasoned_bars": a["seasoned_bars"],
        "max_volume_participation": a["max_volume_participation"], "mainboard_conservative_limit": a["mainboard_conservative_limit"],
        "commission": a["commission"], "min_commission": a["min_fee"], "stamp_tax": a["stamp"], "slippage": a["slippage"]}
    if execution["lot_size"] != 100 or execution["seasoned_bars"] < 60 or not 0 < execution["max_weight"] <= 1:
        raise ValueError("account requires frozen main-board sizing and seasoning")
    if not 0 < execution["mainboard_conservative_limit"] <= .05 or not 0 < execution["max_volume_participation"] <= 1:
        raise ValueError("invalid conservative price/liquidity protection")
    if initial_cash <= 0 or not math.isfinite(initial_cash) or a["max_holding_sessions"] < 1 or not 0 < a["stop_loss"] < 1:
        raise ValueError("invalid capital or risk duration")
    return execution


def _state(broker, position, pending, last_buy_date, holding_prefix, last_close, last_mark, seasoned, used,
           *, symbol, context_hash, mode, entry_diagnostics, current_snapshot=None, reserved_opportunity_ids=()):
    return {"version": STRATEGY_ID, "symbol": symbol, "initial_cash": broker.initial_cash, "cash": broker.cash,
        "shares": broker.shares(symbol), "positions": copy.deepcopy(broker.positions), "avg_cost": copy.deepcopy(broker.avg_cost),
        "ledger": [e.to_dict() for e in broker.ledger.entries], "actual_position": copy.deepcopy(position),
        "pending_intent": copy.deepcopy(pending), "last_buy_date": last_buy_date, "holding_prefix": copy.deepcopy(holding_prefix),
        "last_close_date": last_close, "last_mark": last_mark, "seasoned_bars": seasoned,
        "consumed_plan_ids": sorted(used), "series_context_hash": context_hash, "mode": mode,
        "reserved_opportunity_ids": sorted(reserved_opportunity_ids),
        "entry_diagnostics": copy.deepcopy(entry_diagnostics), "current_snapshot": copy.deepcopy(current_snapshot)}


def _flat_state_hash(state):
    """Hash economic observations; future run/catalog/calendar lineage stays separate."""
    if state.get("shares") != 0 or state.get("pending_intent"):
        raise ValueError("entry attempt identity requires a flat account without pending orders")
    state_keys = ("version", "symbol", "initial_cash", "cash", "shares", "positions", "avg_cost",
        "actual_position", "last_buy_date", "holding_prefix", "last_close_date", "last_mark",
        "seasoned_bars", "consumed_plan_ids", "reserved_opportunity_ids", "observed_prefix_hash")
    ledger_keys = ("date", "symbol", "action", "shares", "price", "notional", "fee", "cash_flow",
        "cash_after", "position_after", "reason", "signal_date", "target_weight", "applied_slippage",
        "equity_before", "equity_after")
    snapshot_keys = ("date", "available_at", "cash", "shares", "equity", "initial_cash",
        "actual_cost_per_share", "close", "bar_index", "ledger_entries_seen")
    value = {key: state.get(key) for key in state_keys}
    value["ledger"] = [{key: row.get(key) for key in ledger_keys} for row in state.get("ledger", [])]
    value["current_snapshot"] = {key: (state.get("current_snapshot") or {}).get(key) for key in snapshot_keys}
    return stable_hash(value)


def _restore(state, broker, symbol):
    if state.get("version") != STRATEGY_ID or state.get("symbol") != symbol:
        raise ValueError("clone account identity mismatch")
    if float(state["initial_cash"]) != broker.initial_cash:
        raise ValueError("clone initial capital contract drift")
    broker.cash = float(state["cash"])
    broker.positions = copy.deepcopy(state.get("positions", {symbol: int(state["shares"])}))
    broker.avg_cost = copy.deepcopy(state.get("avg_cost", {}))
    broker.ledger.entries = [LedgerEntry(**{key: value for key, value in row.items() if key in LedgerEntry.__dataclass_fields__})
        for row in state.get("ledger", [])]
    if broker.shares(symbol) != int(state["shares"]) or set(k for k, v in broker.positions.items() if v > 0) - {symbol}:
        raise ValueError("single-stock clone cannot omit another held asset")


def _intent(symbol, date, side, reason, context_hash, *, event=None, plan=None, triggered_at=None, priority=0,
            reference_price=None, reference_price_source=None):
    stamp = triggered_at or date + "T15:00:00+08:00"
    return {"intent_id": stable_hash({"account": context_hash, "symbol": symbol, "created_at": stamp,
        "side": side, "reason": reason, "event_id": (event or {}).get("event_id"), "plan_id": (plan or {}).get("plan_id")}),
        "symbol": symbol, "date": date, "side": side, "action": side, "reason": reason,
        "original_signal_at": stamp, "first_available_at": stamp, "triggered_at": stamp,
        "last_attempt_at": None, "intended_session": None, "attempts": [], "status": "pending",
        "priority": priority, "all_triggered_reasons": [reason], "source_event_ids": copy.deepcopy((event or {}).get("fresh_candidate_ids", [])),
        "consensus_event_id": (event or {}).get("event_id"), "entry_plan": copy.deepcopy(plan),
        "series_context_hash": context_hash, "source": "account_action", "source_frequency": "日线",
        "reference_price": finite(reference_price if reference_price is not None else (plan or {}).get("reference_price", (event or {}).get("reference_price"))),
        "reference_price_date": date,
        "reference_price_source": reference_price_source or ("signal_day_close" if plan or event else None)}


def _exit_result(response, snapshot, close_at):
    """A holding head is not a trend score or an L2 admission score."""
    reasons = []
    probability = finite(response.get("probability"))
    if probability is None:
        return False, list(response.get("reason_codes") or [])
    if response.get("head") not in {"holding_exit", "ma_exit"}:
        reasons.append("non_ma_exit_head_is_shadow")
    reasons.extend(_identity_reasons(response, close_at, head="ma_exit"))
    identity = response.get("model_identity") or {}
    binding = response.get("candidate_exit_binding") or response.get("entry_exit_binding") or {}
    if identity.get("verified_frozen_contract") is not True or identity.get("head", identity.get("model_head")) != "ma_exit" or identity.get("family", identity.get("model_family")) != STRATEGY_ID:
        reasons.append("holding_exit_model_contract_unverified")
    if not response.get("contract_hash") or not identity.get("manifest_sha256"):
        reasons.append("holding_exit_contract_missing")
    signal = snapshot.get("entry_signal_date")
    if not signal or binding.get("candidate_signal_date", binding.get("signal_date")) != signal or binding.get("plan_id") != snapshot.get("plan_id"):
        reasons.append("holding_exit_original_binding_mismatch")
    if not binding.get("checkpoint") or binding.get("checkpoint") != identity.get("checkpoint"):
        reasons.append("holding_exit_checkpoint_binding_mismatch")
    for key in ("train_label_end", "calibration_label_end", "policy_label_end"):
        if not identity.get(key) or not signal or day(identity[key]) >= signal:
            reasons.append("holding_exit_head_not_prior_to_entry")
    available = identity.get("simulated_available_at", identity.get("available_at"))
    if not available or not signal or not observable(available, signal + "T15:00:00+08:00"):
        reasons.append("holding_exit_head_not_available_at_entry")
    if response.get("usage_mode") == "production" and (not signal or not identity.get("actual_training_completed_at")
        or not observable(identity["actual_training_completed_at"], signal + "T15:00:00+08:00")):
        reasons.append("holding_exit_actual_training_after_entry")
    if probability is None or not 0 <= probability <= 1:
        reasons.append("holding_exit_probability_unavailable")
    if snapshot.get("exit_model_binding") and binding != snapshot["exit_model_binding"]:
        reasons.append("holding_exit_checkpoint_upgrade_forbidden")
    return not reasons and response.get("apply") is True and response.get("request_exit") is True, sorted(set(reasons))


def _close_plan_reason(plan, symbol, date, close, cfg):
    if not isinstance(plan, Mapping) or plan.get("status") != "active" or plan.get("entry_allowed") is not True:
        return "risk_blocked_unscored"
    if plan.get("symbol") != symbol or plan.get("signal_date") != date or plan.get("valid_for_sessions") != 1 or plan.get("policy") != "risk":
        return "risk_plan_identity_or_time_mismatch"
    if not plan.get("plan_id") or not observable(plan.get("available_at"), date + "T15:00:00+08:00"):
        return "risk_plan_unavailable_at_signal"
    values = [finite(plan.get(key)) for key in ("stop_price", "price_floor", "price_ceiling", "reference_price", "target_weight")]
    if any(value is None for value in values):
        return "risk_plan_frozen_fields_missing"
    stop, floor, ceiling, reference, weight = values
    distance = (close - stop) / close
    if (not 0 < stop < close or reference != close or floor > ceiling or not 0 < weight <= cfg["account"]["max_weight"]
        or distance + 1e-12 < cfg["signal_points"]["min_risk_fraction"] or distance - 1e-12 > cfg["signal_points"]["max_stop_distance"]):
        return "risk_plan_outside_frozen_contract"
    return None


def replay_trend_account(frame, market_dates, decisions=None, *, evidence_pack=None, native_pack=None,
                         trend_probabilities=None, config=None, exit_policy=None, quality_filter=None,
                         entry_evaluator=None, start=None, end=None, initial_cash=None, run_id="",
                         initial_state=None, forced_exit_at=None, forced_entry=None, stop_after_flat=False,
                         mode="final", account_id=None, model_identity=None, source_context_hash=None, account_context=None):
    """Execute close intentions at the next verified open, retaining sell retries.

    initial_state is a complete close-state Broker clone. A clone's returned
    ledger contains only new fills; history_ledger preserves the shared prefix.
    Missing verified sessions remain daily records and explicit refusals.
    """
    cfg = _strategy_config(config)
    if mode not in {"E0", "E1", "final"}:
        raise ValueError("unsupported account continuation policy")
    cash = float(initial_cash if initial_cash is not None else (initial_state or {}).get("initial_cash", cfg["account"]["initial_cash"]))
    settings = _settings(cfg, cash)
    cache = account_context or prepare_account_context(frame, market_dates)
    if cache["frame_object"] is not frame or cache["calendar"] != list(market_dates):
        raise ValueError("prepared quote context belongs to another frame/calendar")
    records, dates, calendar = cache["records"], cache["dates"], cache["calendar"]
    if not records or dates != sorted(set(dates)) or calendar != sorted(set(calendar)) or not set(dates).issubset(set(calendar)):
        raise ValueError("account requires unique ascending bars and independently verified calendar")
    symbols = {bar["symbol"] for bar in records}
    if len(symbols) != 1:
        raise ValueError("one explicit account must own one symbol")
    symbol = next(iter(symbols)); by_day = cache["by_day"]
    observed_index = {date: index for index, date in enumerate(dates)}
    prepared = decisions or prepare_trend_decisions(frame, calendar, evidence_pack, trend_probabilities, cfg, native_pack=native_pack, mode="E0")
    direction_rows = prepared["rows"] if isinstance(prepared, Mapping) else list(prepared)
    direction = {day(row["date"]): row for row in direction_rows}
    if set(direction) != set(dates):
        raise ValueError("direction/account bar alignment mismatch")
    begin = day(start or ((initial_state or {}).get("last_close_date") or dates[0]))
    finish = day(end or dates[-1]); next_day = dict(zip(calendar[:-1], calendar[1:])); market_index = {d: i for i, d in enumerate(calendar)}
    source_rows, prefix_hashes = records, cache["prefix_hashes"]
    source_hash = (source_context_hash or (initial_state or {}).get("source_context_hash")
        or (frame.attrs.get("source_context_hash") or frame.attrs.get("source_snapshot_hash") if isinstance(frame, pd.DataFrame) else None)
        or cfg.get("source_snapshot_hash"))
    source_verified = bool(source_hash)
    if not source_hash:
        # Ad-hoc synthetic inputs may be exercised, but have no certified data
        # vintage. Production/runtime and training pass the frozen revision.
        source_hash = stable_hash({"unfrozen_source": symbol, "first_known_bar": source_rows[0]})
    context = {"strategy": STRATEGY_ID, "symbol": symbol, "account_id": account_id or "research:" + run_id,
        "account_kind": cfg["account"]["account_kind"], "account_start": begin, "initial_cash": cash,
        "initial_state_hash": stable_hash(initial_state) if initial_state else None, "source_hash": source_hash,
        "calendar_hash": stable_hash(calendar), "config_hash": stable_hash(cfg), "mode": mode,
        "model_identity": copy.deepcopy(model_identity), "exit_recipe_hash": getattr(exit_policy, "recipe_hash", None),
        "entry_recipe_hash": getattr(entry_evaluator or quality_filter, "recipe_hash", None)}
    context.update(consensus=copy.deepcopy(cfg["consensus"]), account=copy.deepcopy(cfg["account"]),
        risk_policy=cfg["risk_policy"], risk_semantics=copy.deepcopy(cfg.get("risk_semantics")),
        signal_points=copy.deepcopy(cfg["signal_points"]), candidate_policy=cfg["candidate_policy"],
        costs={"version": "flat_v1", "commission": settings["commission"], "min_commission": settings["min_commission"],
            "stamp_tax": settings["stamp_tax"], "slippage": settings["slippage"], "impact_tiers": []})
    context["source_revision_verified"] = source_verified
    context_hash = stable_hash(context)
    cost = CostModel(commission=settings["commission"], stamp_tax=settings["stamp_tax"], slippage=settings["slippage"],
        min_commission=settings["min_commission"], version="flat_v1", impact_tiers=())
    broker = BrokerSimulator(cash, cost, run_id)
    position, pending, last_buy, prefix, seasoned, used, reserved = None, None, None, {}, 0, set(), set()
    last_mark, resume_after, history_count = None, None, 0
    if initial_state:
        _restore(initial_state, broker, symbol)
        position = copy.deepcopy(initial_state.get("actual_position")); pending = copy.deepcopy(initial_state.get("pending_intent"))
        last_buy = initial_state.get("last_buy_date"); prefix = copy.deepcopy(initial_state.get("holding_prefix") or {})
        seasoned = int(initial_state.get("seasoned_bars", 0)); used = set(initial_state.get("consumed_plan_ids") or [])
        reserved = set(initial_state.get("reserved_opportunity_ids") or [])
        last_mark = finite(initial_state.get("last_mark")); resume_after = day(initial_state["last_close_date"])
        history_count = len(broker.ledger.entries)
    history_ledger = [e.to_dict() for e in broker.ledger.entries]
    actions, rejections, daily, holding_snapshots, account_snapshots, result_rows = [], [], [], [], [], []
    entry_diagnostics, exit_diagnostics, fills = [], [], []
    had_position = broker.shares(symbol) > 0
    if exit_policy is not None and hasattr(exit_policy, "new_replay"):
        exit_policy = exit_policy.new_replay()
    evaluator = entry_evaluator or quality_filter
    if forced_entry:
        if not initial_state or broker.shares(symbol) != 0 or pending and pending.get("side") == "SELL":
            raise ValueError("candidate clone requires a genuinely flat reference account")
        signal = resume_after
        if forced_entry.get("signal_date") != signal:
            raise ValueError("candidate clone plan is not the known close plan")
        pending = _intent(symbol, signal, "BUY", "candidate_clone_admission", context_hash, plan=forced_entry)
        pending["intended_session"] = next_day.get(signal)
        pending["original_opportunity_ids"] = sorted(forced_entry.get("original_opportunity_ids") or forced_entry.get("source_event_ids") or [])
        reserved.update(pending["original_opportunity_ids"])
        actions.append(copy.deepcopy(pending))
    if forced_exit_at and resume_after and day(forced_exit_at) == resume_after:
        if broker.shares(symbol) <= 0:
            raise ValueError("forced holding branch must own actual remaining shares")
        if pending and pending["side"] == "SELL":
            pending["forced_branch_request_at"] = resume_after + "T15:00:00+08:00"
        else:
            pending = _intent(symbol, resume_after, "SELL", "forced_next_open_exit_branch", context_hash, priority=100,
                reference_price=last_mark, reference_price_source="signal_day_close")
            pending["intended_session"] = next_day.get(resume_after)
            actions.append(copy.deepcopy(pending))

    def refuse(intent, date, reason):
        intent["last_attempt_at"] = date + "T09:30:00+08:00"
        attempt = {"execution_at": intent["last_attempt_at"], "status": "rejected", "reason": reason}
        intent["attempts"].append(attempt)
        rejections.append({"date": date, "signal_date": day(intent["original_signal_at"]), "symbol": symbol,
            "action": intent["side"], "reason": reason, "intent_id": intent["intent_id"],
            "original_signal_at": intent["original_signal_at"], "intended_session": intent["intended_session"],
            "execution_at": attempt["execution_at"], "series_context_hash": context_hash})

    def schedule_exit(date, reason, priority, *, event=None, stamp=None, response=None):
        nonlocal pending
        if broker.shares(symbol) <= 0:
            return
        if pending and pending["side"] == "SELL":
            if reason not in pending["all_triggered_reasons"]:
                pending["all_triggered_reasons"].append(reason)
            if priority > pending["priority"]:
                pending["priority"] = priority
                pending["reason"] = reason
                pending.setdefault("escalations", []).append({"at": stamp or date + "T15:00:00+08:00", "reason": reason,
                    "reference_price": finite((bar or {}).get("open" if stamp and "T09:30:" in stamp else "close")),
                    "reference_price_source": "trigger_open" if stamp and "T09:30:" in stamp else "signal_day_close"})
            return
        pending = _intent(symbol, date, "SELL", reason, context_hash, event=event, triggered_at=stamp, priority=priority,
            reference_price=(bar or {}).get("open" if stamp and "T09:30:" in stamp else "close") if quote_valid else None,
            reference_price_source=("trigger_open" if stamp and "T09:30:" in stamp else "signal_day_close") if quote_valid else None)
        pending["intended_session"] = date if stamp and "T09:30:" in stamp else next_day.get(date)
        if response:
            pending["exit_model"] = copy.deepcopy(response)
        actions.append(copy.deepcopy(pending))

    for date in calendar:
        if date > finish:
            break
        bar = by_day.get(date)
        quote_valid = bool(bar and bar.get("has_trade_price") == 1 and bar.get("source") in cfg["account"]["valid_sources"]
            and (finite(bar.get("volume")) or 0.) > 0 and (finite(bar.get("amount")) or 0.) > 0)
        actions_before = len(actions)
        current_holding, current_exit, current_entry = None, None, None
        if resume_after is not None and date <= resume_after:
            continue
        if date < begin:
            if bar and bar.get("has_trade_price") == 1 and bar.get("source") in cfg["account"]["valid_sources"] and finite(bar.get("volume")) and finite(bar.get("amount")):
                seasoned += 1
                last_mark = finite(bar.get("close")) or last_mark
            continue
        prev_date = calendar[market_index[date] - 1] if market_index[date] > 0 else None
        previous = by_day.get(prev_date)
        # Frozen protective order existed before observing the gap-open quote.
        if quote_valid and position and finite(bar.get("open")) is not None and float(bar["open"]) <= position["stop_price"]:
            schedule_exit(date, "actual_account_stop", 100, stamp=date + "T09:30:00+08:00")
        if pending and pending.get("intended_session") and date >= pending["intended_session"]:
            intent = pending
            reason = None
            if not bar:
                reason = "missing_target_market_session"
            elif not previous:
                reason = "missing_previous_verified_session"
            elif bar.get("source") not in cfg["account"]["valid_sources"] or previous.get("source") not in cfg["account"]["valid_sources"]:
                reason = "unverified_price_source"
            else:
                reason = _guard(symbol, pd.Series(bar), pd.Series(previous), intent["side"], seasoned, settings)
            plan = intent.get("entry_plan")
            if intent["side"] == "BUY":
                if broker.shares(symbol) > 0:
                    reason = reason or "actual_position_already_open"
                check_intent = {"date": day(intent["original_signal_at"])}
                reason = reason or _planned_entry_rejection(plan, check_intent, symbol, date, next_day, "risk", used)
                if not reason and any(finite(plan.get(key)) is None for key in ("price_floor", "price_ceiling", "stop_price", "target_weight")):
                    reason = "risk_plan_frozen_fields_missing"
                if not reason and not float(plan["price_floor"]) <= float(bar["open"]) <= float(plan["price_ceiling"]):
                    reason = "risk_open_outside_frozen_price_band"
            elif last_buy == date:
                reason = reason or "t1_same_day_sale_blocked"
            if reason:
                refuse(intent, date, reason)
                if intent["side"] == "BUY":
                    intent["status"] = "expired_or_rejected"
                    if isinstance(plan, Mapping) and plan.get("plan_id"):
                        used.add(plan["plan_id"])
                    pending = None
            else:
                open_price = float(bar["open"])
                price = cost.execution_price(intent["side"], open_price)
                lot = settings["lot_size"]
                liquidity = int(float(previous["volume"]) * settings["max_volume_participation"] / lot) * lot
                if intent["side"] == "BUY":
                    equity = broker.equity({symbol: open_price})
                    weight = float(plan["target_weight"])
                    if not 0 < weight <= settings["max_weight"]:
                        raise ValueError("candidate target exceeds frozen account budget")
                    shares = min(int(equity * weight / price / lot) * lot, liquidity)
                    while shares > 0 and shares * price + cost.fees("BUY", shares * price) > broker.cash:
                        shares -= lot
                else:
                    weight = 0.; shares = min(broker.shares(symbol), liquidity)
                    # Closing odd residual shares is permitted once liquidity can cover them.
                    if 0 < broker.shares(symbol) < lot and liquidity >= lot:
                        shares = broker.shares(symbol)
                if shares <= 0:
                    refuse(intent, date, "insufficient_cash_or_prior_liquidity")
                    if intent["side"] == "BUY":
                        pending = None; used.add(plan["plan_id"])
                else:
                    entry = broker.submit(Order(date, symbol, intent["side"], shares, price, reason=intent["reason"],
                        target_weight=weight, signal_date=day(intent["original_signal_at"]), applied_slippage=cost.slippage),
                        mark_price=open_price, equity_before=broker.equity({symbol: open_price}))
                    if entry is None:
                        refuse(intent, date, "broker_rejected")
                        if intent["side"] == "BUY":
                            pending = None
                    else:
                        intent["last_attempt_at"] = date + "T09:30:00+08:00"
                        intent["attempts"].append({"execution_at": intent["last_attempt_at"], "status": "filled" if entry.position_after == 0 or intent["side"] == "BUY" else "partial", "shares": entry.shares})
                        record = {**entry.to_dict(), "intent_id": intent["intent_id"], "original_signal_at": intent["original_signal_at"],
                            "first_available_at": intent["first_available_at"], "triggered_at": intent["triggered_at"],
                            "execution_at": intent["last_attempt_at"], "series_context_hash": context_hash,
                            "all_triggered_reasons": copy.deepcopy(intent["all_triggered_reasons"])}
                        fills.append(record)
                        if intent["side"] == "BUY":
                            used.add(plan["plan_id"]); last_buy = date; had_position = True
                            stop = max(float(plan["stop_price"]), broker.avg_cost[symbol] * (1 - cfg["account"]["stop_loss"]))
                            position = {"plan_id": plan["plan_id"], "entry_date": date, "entry_signal_date": day(intent["original_signal_at"]),
                                "entry_shares": entry.shares, "stop_price": stop, "frozen_structure_stop": plan["stop_price"],
                                "entry_plan": copy.deepcopy(plan), "exit_model_binding": None, "holding_boundaries": []}
                            for boundary in plan.get("holding_boundaries", []):
                                if (boundary.get("symbol") == symbol and boundary.get("source") in plan.get("sources", [])
                                    and boundary.get("event_id") in plan.get("source_event_ids", [])
                                    and finite(boundary.get("price")) is not None
                                    and observable(boundary.get("first_available_at"), plan["available_at"])):
                                    position["holding_boundaries"].append(copy.deepcopy(boundary))
                            prefix = {"peak": price, "trough": price, "close_peak": price}
                            intent["status"] = "filled"; pending = None
                        elif broker.shares(symbol) == 0:
                            position = None; prefix = {}; intent["status"] = "filled"; pending = None
                        else:
                            intent["status"] = "partial"
        if quote_valid:
            mark = finite(bar.get("close"))
            if mark is None or mark <= 0:
                raise ValueError("invalid actual close mark")
            last_mark = mark
        else:
            mark = last_mark
        if mark is None and broker.shares(symbol):
            raise ValueError("held clone has no known valuation mark")
        equity = broker.equity({symbol: mark or 1.})
        daily.append({"date": date, "symbol": symbol, "cash": broker.cash, "shares": broker.shares(symbol),
            "close": mark, "equity": equity, "price_available": quote_valid, "series_context_hash": context_hash,
            "sell_pending": pending["intent_id"] if pending and pending["side"] == "SELL" else None})
        if had_position and stop_after_flat and broker.shares(symbol) == 0:
            break
        if not quote_valid:
            # Duration is known from the independent calendar, even when no
            # eligible price is available. Queue liquidation; never invent a
            # valuation or an execution price for that session.
            if position and market_index[date] - market_index[last_buy] + 1 >= cfg["account"]["max_holding_sessions"]:
                schedule_exit(date, "max_holding_sessions", 90)
            new_actions = copy.deepcopy(actions[actions_before:])
            result_rows.append({"date": date, "available_at": date + "T15:00:00+08:00", "symbol": symbol,
                "action": new_actions[-1]["side"] if new_actions else "WAIT", "emitted": bool(new_actions),
                "final_action_intents": new_actions, "holding_snapshot": None, "exit_evaluation": None,
                "entry_evaluation": None, "account_id": context["account_id"], "series_context_hash": context_hash,
                "reason_codes": ["missing_target_market_session" if not bar else "unverified_current_trade_price"]})
            continue
        close_at = date + "T15:00:00+08:00"
        base_snapshot = {"symbol": symbol, "date": date, "available_at": close_at, "cash": broker.cash,
            "shares": broker.shares(symbol), "equity": equity, "initial_cash": broker.initial_cash,
            "actual_cost_per_share": broker.avg_cost.get(symbol, 0.), "close": mark, "bar_index": market_index[date], "series_context_hash": context_hash,
            "pending_intent": copy.deepcopy(pending), "bar": {k: bar.get(k) for k in ("open", "high", "low", "close", "volume", "amount", "source", "has_trade_price")},
            "next_market_session": next_day.get(date), "ledger_entries_seen": len(broker.ledger.entries)}
        base_snapshot.update(account_id=context["account_id"], source_context_hash=source_hash)
        opinion = direction[date]
        if broker.shares(symbol) > 0:
            prefix["peak"] = max(prefix.get("peak", mark), float(bar["high"]))
            prefix["trough"] = min(prefix.get("trough", mark), float(bar["low"]))
            prefix["close_peak"] = max(prefix.get("close_peak", mark), mark)
            snapshot = {**base_snapshot, **copy.deepcopy(position), "actual_cost_per_share": broker.avg_cost[symbol],
                "holding_bars": market_index[date] - market_index[last_buy] + 1,
                "holding_sessions": market_index[date] - market_index[last_buy] + 1,
                "holding_prefix": copy.deepcopy(prefix), "sell_pending": pending["reason"] if pending and pending["side"] == "SELL" else None}
            snapshot["episode_id"] = stable_hash({"symbol": symbol, "plan_id": snapshot["plan_id"], "entry_date": snapshot["entry_date"]})
            holding_snapshots.append(copy.deepcopy(snapshot))
            current_holding = snapshot
            if forced_exit_at and date == day(forced_exit_at):
                schedule_exit(date, "forced_next_open_exit_branch", 100)
            if mark <= position["stop_price"]:
                schedule_exit(date, "actual_account_stop", 100)
            if snapshot["holding_sessions"] >= cfg["account"]["max_holding_sessions"]:
                schedule_exit(date, "max_holding_sessions", 90)
            for boundary in position.get("holding_boundaries", []):
                broken = mark <= boundary["price"] if boundary["source"] == "structure" else mark < boundary["price"]
                if broken:
                    schedule_exit(date, "holding_confirmed_boundary", 85)
            if opinion.get("sell_pass") is True or opinion.get("sell_base_qualified") is True:
                schedule_exit(date, "ordinary_consensus_sell", 70, event=opinion)
            if mode in {"E1", "final"} and exit_policy is not None:
                response = _call(exit_policy, snapshot)
                valid, errors = _exit_result(response, snapshot, close_at)
                # The account's validated evaluation is also the presentation
                # contract. An invalid raw head must not retain a usable score
                # merely because its adapter claimed compatibility.
                current_exit = {**response, "validation_reasons": errors}
                if errors:
                    current_exit.update(raw_probability=response.get("probability"), probability=None,
                        holding_contract_compatible=False, in_scope=False, evaluation_status="identity_error")
                elif finite(response.get("probability")) is None:
                    current_exit.update(probability=None, holding_contract_compatible=False, in_scope=False)
                binding = response.get("candidate_exit_binding") or response.get("entry_exit_binding")
                fallback = response.get("status", "").startswith("explicit_rule")
                if binding and position.get("exit_model_binding") is None and (not errors and finite(response.get("probability")) is not None or fallback):
                    position["exit_model_binding"] = copy.deepcopy(binding)
                exit_diagnostics.append({**current_exit, "date": date, "shares": snapshot["shares"],
                    "hard_exit_priority": bool(pending and pending["priority"] >= 90), "series_context_hash": context_hash,
                    "applied": bool(valid and not (pending and pending["priority"] >= 90))})
                if valid and not (pending and pending["priority"] >= 90):
                    schedule_exit(date, "signal_pinned_ma_exit", 80, response=response)
        elif (opinion.get("entry_eligible") is True or ("entry_eligible" not in opinion and opinion.get("action") == "BUY")) and not pending and not stop_after_flat:
            eligible_opinion = _unreserved_opinion(opinion, reserved, bar, observed_index[date], records, cfg)
            candidate = candidate_from_row(eligible_opinion) if eligible_opinion is not None else None
            if candidate is not None:
                plan = candidate["entry_plan"]
                current_state = _state(broker, position, pending, last_buy, prefix, date, mark, seasoned + 1, used,
                    symbol=symbol, context_hash=context_hash, mode=mode, entry_diagnostics=[],
                    current_snapshot=base_snapshot, reserved_opportunity_ids=reserved)
                current_state.update(source_context_hash=source_hash, account_id=context["account_id"],
                    observed_prefix_hash=prefix_hashes[date])
                state_hash, flat_hash, plan_hash = stable_hash(current_state), _flat_state_hash(current_state), stable_hash(plan)
                attempt = stable_hash({"family": STRATEGY_ID, "symbol": symbol, "date": date,
                    "original_opportunity_ids": candidate["original_opportunity_ids"], "flat_state_hash": flat_hash,
                    "entry_plan_hash": plan_hash})
                candidate.update(candidate_id=attempt, attempt_id=attempt, reference_state_hash=state_hash,
                    flat_state_hash=flat_hash, entry_plan_hash=plan_hash)
                allowed, reason, evaluation = True, None, {}
                reason = _close_plan_reason(plan, symbol, date, mark, cfg)
                risk_qualified = reason is None
                if reason:
                    allowed = False
                    evaluation = {"status": reason, "probability": None}
                elif mode == "final" and cfg["consensus"]["require_high_return"]:
                    evaluation = _call(evaluator, base_snapshot, candidate)
                    if not evaluation:
                        evaluation = {"status": "model_unavailable", "probability": None,
                            "reason_codes": ["entry_evaluator_unavailable"]}
                    probability = finite(evaluation.get("probability"))
                    identity_errors = _identity_reasons(evaluation, close_at, head="entry")
                    total = float(opinion["buy_base_score"]) + cfg["consensus"]["quality_weight"] * (probability or 0.)
                    if evaluation.get("status") != "scored" or probability is None:
                        reason = evaluation.get("status") if evaluation.get("status") not in {None, "scored"} else "model_unavailable"
                    elif not 0 <= probability <= 1:
                        reason = "high_return_probability_invalid"
                    elif identity_errors:
                        reason = "identity_error"
                        evaluation["reason_codes"] = identity_errors
                    elif evaluation.get("in_scope") is not True:
                        reason = "high_return_population_out_of_scope"
                    elif probability < cfg["consensus"]["high_return_threshold"]:
                        reason = "high_return_below_threshold"
                    elif total + 1e-12 < cfg["consensus"]["buy_total_threshold"]:
                        reason = "buy_total_support_below_threshold"
                    else:
                        reason = None
                    allowed = reason is None
                    if reason == "model_unavailable" and evaluation.get("status") == "scored":
                        evaluation.update(status="model_unavailable", evaluation_status="model_unavailable")
                    if not allowed and probability is not None:
                        evaluation["evaluation_status"] = "identity_error" if reason == "identity_error" else "scored_rejected"
                if reason and reason not in evaluation.setdefault("reason_codes", []):
                    evaluation["reason_codes"].append(reason)
                entry_diagnostics.append({"date": date, "snapshot": copy.deepcopy(base_snapshot), "candidate": copy.deepcopy(candidate),
                    "pre_gate_state": copy.deepcopy(current_state), "reference_state": copy.deepcopy(current_state),
                    "attempt_id": attempt, "economic_group_id": candidate["economic_group_id"],
                    "original_opportunity_ids": copy.deepcopy(candidate["original_opportunity_ids"]),
                    "reference_state_hash": state_hash, "flat_state_hash": flat_hash,
                    "entry_plan_hash": plan_hash, "risk_qualified": risk_qualified,
                    "evaluation": evaluation, "allowed": allowed, "reason": reason, "series_context_hash": context_hash,
                    "account_id": context["account_id"]})
                current_entry = evaluation
                if allowed:
                    pending = _intent(symbol, date, "BUY", "continuous_consensus_buy", context_hash, event=eligible_opinion, plan=plan)
                    pending["intended_session"] = next_day.get(date)
                    pending["entry_evaluation"] = evaluation
                    pending.update(attempt_id=attempt, economic_group_id=candidate["economic_group_id"],
                        original_opportunity_ids=copy.deepcopy(candidate["original_opportunity_ids"]))
                    reserved.update(candidate["original_opportunity_ids"])
                    actions.append(copy.deepcopy(pending))
        if bar.get("has_trade_price") == 1 and bar.get("source") in cfg["account"]["valid_sources"] and finite(bar.get("volume")) and finite(bar.get("amount")):
            seasoned += 1
        state = _state(broker, position, pending, last_buy, prefix, date, mark, seasoned, used,
            symbol=symbol, context_hash=context_hash, mode=mode, entry_diagnostics=[], current_snapshot=base_snapshot,
            reserved_opportunity_ids=reserved)
        state["source_context_hash"] = source_hash
        state["account_id"] = context["account_id"]
        account_snapshots.append(state)
        if holding_snapshots and holding_snapshots[-1]["date"] == date:
            holding_snapshots[-1]["account_state"] = copy.deepcopy(state)
        prefix_hash = prefix_hashes[date]
        new_actions = copy.deepcopy(actions[actions_before:])
        result_rows.append({"date": date, "available_at": close_at, "symbol": symbol,
            "action": new_actions[-1]["side"] if new_actions else "HOLD", "emitted": bool(new_actions),
            "holding_snapshot": copy.deepcopy(current_holding), "exit_evaluation": copy.deepcopy(current_exit),
            "entry_evaluation": copy.deepcopy(current_entry), "direction_decision": copy.deepcopy(opinion),
            "final_action_intents": new_actions, "account_id": context["account_id"], "series_context_hash": context_hash,
            "observed_prefix_hash": prefix_hash, "cash": broker.cash, "shares": broker.shares(symbol), "equity": equity})
    if not daily:
        raise ValueError("account range contains no verified sessions after its state")
    ledger = [e.to_dict() for e in broker.ledger.entries[history_count:]]
    # Enrich fills without modifying the shared frozen Ledger schema.
    enriched = { (f["date"], f["action"], f["shares"], f["cash_after"]): f for f in fills }
    ledger = [enriched.get((f["date"], f["action"], f["shares"], f["cash_after"]), f) for f in ledger]
    values = pd.Series([cash, *[row["equity"] for row in daily]], dtype=float)
    returns = values.pct_change().iloc[1:]; volatility = float(returns.std(ddof=0) * math.sqrt(252))
    metrics = {"total_return": float(values.iloc[-1] / cash - 1), "max_drawdown": float((values / values.cummax() - 1).min()),
        "final_equity": float(values.iloc[-1]), "trade_count": len(ledger), "rejection_count": len(rejections),
        "fees": sum(row["fee"] for row in ledger), "open_shares": broker.shares(symbol), "days": len(daily),
        "annualized_volatility": volatility, "sharpe": float(returns.mean() * 252 / volatility) if volatility else 0.}
    terminal = _state(broker, position, pending, last_buy, prefix, daily[-1]["date"], last_mark, seasoned, used,
        symbol=symbol, context_hash=context_hash, mode=mode, entry_diagnostics=entry_diagnostics,
        reserved_opportunity_ids=reserved)
    terminal.update(source_context_hash=source_hash, account_id=context["account_id"])
    for collection in (actions, rejections, daily, holding_snapshots, fills, ledger):
        for row in collection:
            row["account_id"] = context["account_id"]
    return {"version": STRATEGY_ID, "display_name": DISPLAY_NAME, "symbol": symbol, "mode": mode,
        "series_context_hash": context_hash, "series_context": context, "ledger": ledger, "trades": ledger, "broker_fills": ledger,
        "rows": result_rows, "account_id": context["account_id"], "observed_prefix_hash": prefix_hashes[daily[-1]["date"]],
        "history_ledger": history_ledger, "daily": daily, "metrics": metrics, "rejections": rejections,
        "holding_snapshots": holding_snapshots, "account_snapshots": account_snapshots,
        "exit_policy_diagnostics": exit_diagnostics, "entry_diagnostics": entry_diagnostics,
        "final_action_intents": actions, "direction_points": copy.deepcopy(prepared.get("direction_points", []) if isinstance(prepared, Mapping) else []),
        "terminal_state": terminal, "actual_position": copy.deepcopy(position), "sell_pending": copy.deepcopy(pending) if pending and pending["side"] == "SELL" else None,
        "cost_model": cost.as_config_dict(), "completed": bool(had_position and broker.shares(symbol) == 0)}


def replay_rule_account(frame, calendar, evidence_pack=None, trend_probabilities=None, config=None,
                        exit_policy=None, quality_filter=None, start=None, initial_cash=None, run_id="", **kwargs):
    mode = "E1" if exit_policy is not None else "E0"
    return replay_trend_account(frame, calendar, evidence_pack=evidence_pack, trend_probabilities=trend_probabilities,
        config=config, exit_policy=exit_policy, quality_filter=None, start=start, initial_cash=initial_cash,
        run_id=run_id, mode=mode, **kwargs)


def clone_candidate_account(frame, calendar, prepared, reference_state, candidate, config=None, exit_policy=None, run_id="", *, account_context=None):
    """Open a same-close risk candidate using the actual flat state's cash."""
    state = copy.deepcopy(reference_state)
    if state.get("shares") != 0 or (state.get("pending_intent") or {}).get("side") == "SELL":
        raise ValueError("candidate reference is not actually flat")
    plan = candidate.get("entry_plan") or candidate.get("raw_risk_plan") or candidate
    # The clone creates exactly one episode. The reference pending admission is
    # replaced by this explicitly scoped candidate, not executed twice.
    state["pending_intent"] = None
    return replay_trend_account(frame, calendar, prepared, config=config, exit_policy=exit_policy,
        initial_state=state, forced_entry=plan, stop_after_flat=True, mode="E1" if exit_policy else "E0", run_id=run_id,
        account_context=account_context)


def clone_holding_branches(frame, calendar, prepared, holding_snapshot, config=None, exit_policy=None, run_id="", *, forced=True, account_context=None):
    """Clone a complete close state; forced branch requests next verified open."""
    state = holding_snapshot.get("account_state", holding_snapshot)
    if state.get("shares", 0) <= 0 or not state.get("actual_position"):
        raise ValueError("holding branch requires a complete real Broker state")
    return replay_trend_account(frame, calendar, prepared, config=config, exit_policy=exit_policy,
        initial_state=copy.deepcopy(state), forced_exit_at=state["last_close_date"] if forced else None,
        stop_after_flat=True, mode="E1" if exit_policy else "E0", run_id=run_id, account_context=account_context)
