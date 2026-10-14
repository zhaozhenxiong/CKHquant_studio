"""Causal account-free references and explicit research Broker simulation."""
from __future__ import annotations

import copy
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_contracts import (STRATEGY_ID, DISPLAY_NAME,
    TRAINING_POPULATION, ATTEMPT_IDENTITY, ECONOMIC_GROUP_IDENTITY, strategy_config)

CURRENT_MODEL_PATH = Path(os.environ.get("KHQUANT_CURRENT_STRATEGY_CONFIG",
    str(MY_STRATEGY_ROOT / "configs/current_strategy.json")))

PRICING_ASSUMPTIONS = {
    "broker_fill": "continuous_cost_model_simulated_average_after_slippage",
    "initial_R": "frozen_stop_reference_half_up_cent_then_sell_fees",
    "initial_R_is_actual_stop_loss": False,
    "scope": "research_simulation",
}

REFERENCE_PRICING_ASSUMPTIONS = {
    "reference_price": "last_observed_signal_day_close",
    "reference_is_fill": False,
    "scope": "analysis_reference",
    "account_simulated": False,
}


def _source_reference_point(event, row):
    """Project an original confirmed source at its causal observation clock."""
    from my_strategy.services.czsc_trend_consensus import observable
    available = row["available_at"]
    direction_eligible = event.get("direction_eligible", event.get("candidate_direction_valid", event.get("candidate_eligible")))
    if (event.get("symbol") != row["symbol"] or event.get("source") not in {"ma", "structure", "wyckoff"}
            or event.get("side") not in {"BUY", "SELL"} or event.get("confirmed") is not True
            or event.get("input_eligible") is not True or not event.get("event_id")
            or not event.get("original_event_id") or not event.get("pattern_id")
            or direction_eligible is not True or event.get("evidence_visible", True) is not True
            or not event.get("date") or str(event["date"])[:10] > row["date"]
            or not observable(event.get("available_at"), available)
            or any(event.get(key) and not observable(event[key], available)
                   for key in ("first_confirmed_at", "first_available_at", "origin_observed_at", "observed_at"))):
        return None
    if event["source"] == "ma" and row["agents"]["ma"].get("status") != "scored":
        return None
    point = copy.deepcopy(event)
    source = event["source"]
    point.update(date=row["date"], signal_date=row["date"], reference_layer="source",
        reference_only=True, analysis_mode="reference", account_scope="none",
        action=event["side"], direction_eligible=True,
        reference_price=row["reference_price"], reference_price_date=row["date"],
        reference_price_source="signal_day_close", consensus_action=row["direction"],
        consensus_qualified=row["direction"] == event["side"],
        consensus_reason_codes=copy.deepcopy(row.get("reason_codes", [])),
        source_label={"ma": "均线 ML", "structure": "结构", "wyckoff": "量价"}[source])
    return point


def _reference_risk_plan(row, bars, index, native, cfg):
    """A causal long-risk ceiling, independent from cash or held inventory."""
    from my_strategy.services.czsc_trend_consensus import finite, observable
    params, at = cfg["signal_points"], row["available_at"]
    price = finite(row.get("reference_price"))
    tr = []
    for i in range(max(1, index - 14), index):
        hi, lo, previous = finite(bars[i].get("high")), finite(bars[i].get("low")), finite(bars[i - 1].get("close"))
        if None in (hi, lo, previous):
            break
        tr.append(max(hi - lo, abs(hi - previous), abs(lo - previous)))
    atr = sum(tr) / 14 if len(tr) == 14 else None
    boundaries = []
    for source in ("structure", "wyckoff"):
        evaluation = native.get(source) or {}
        if row["agents"][source].get("status") != "scored":
            continue
        for event in evaluation.get("support_facts", []):
            boundary, stop = finite(event.get("invalidation_price")), finite(event.get("risk_stop_price"))
            if (event.get("side") != "BUY" or event.get("source") != source or event.get("symbol") != row["symbol"]
                    or event.get("event_id") in set(evaluation.get("invalidated_event_ids") or [])
                    or event.get("confirmed") is not True
                    or event.get("input_eligible") is not True or boundary is None or stop is None
                    or price is None or price <= boundary or not 0 < stop <= boundary < price
                    or not observable(event.get("available_at"), at)
                    or any(event.get(key) and not observable(event[key], at)
                           for key in ("first_confirmed_at", "first_available_at", "origin_observed_at", "observed_at"))
                    or (source == "wyckoff" and (finite(event.get("event_strength")) or 0) <= 0)):
                continue
            boundaries.append({"source": source, "event_id": event.get("event_id"),
                "boundary_price": boundary, "stop_price": stop,
                "available_at": event.get("first_available_at") or event["available_at"]})
        zone = (evaluation.get("current_state") or {}).get("current_valid_zone") if source == "structure" else None
        boundary = finite((zone or {}).get("support_low"))
        if (zone and zone.get("confirmed") is True and zone.get("dependency_calendar_complete") is True
                and boundary is not None and price is not None and atr is not None and price > boundary
                and observable(zone.get("first_confirmed_at"), at)):
            stop = boundary - params["stop_atr_buffer"] * atr
            if 0 < stop < price:
                boundaries.append({"source": "confirmed_structure_zone", "event_id": zone["zone_id"],
                    "boundary_price": boundary, "stop_price": stop, "available_at": zone["first_confirmed_at"]})
    # A qualified MA channel may supply the same frozen prior20 / priorATR
    # risk geometry as training, but absence of a head cannot create a stop.
    if row["agents"]["ma"].get("status") == "scored" and price and atr:
        history = bars[max(0, index - params["lookback_stop_bars"]):index]
        lows = [finite(bar.get("low")) for bar in history]
        if len(lows) == params["lookback_stop_bars"] and all(value is not None for value in lows):
            boundary = max(min(lows), price - params["atr_stop_multiple"] * atr)
            stop = boundary - params["stop_atr_buffer"] * atr
            if 0 < stop < price:
                boundaries.append({"source": "qualified_ma_prior_risk", "event_id": None,
                    "boundary_price": boundary, "stop_price": stop, "available_at": at})
    # A live BUY opportunity must use the exact signal-close geometry that
    # generated its fee-net quality target. Other confirmed boundaries are
    # useful ceilings, but cannot silently replace that frozen BUY plan.
    original_plan, _ = _reference_buy_plan(row, cfg)
    if original_plan is not None:
        original_stops = [item for item in boundaries
            if item.get("event_id") in original_plan["source_event_ids"]
            and item["stop_price"] == original_plan["stop_price"]]
        chosen = max(original_stops, key=lambda item: item["boundary_price"], default={
            "source": "original_buy_risk_plan", "event_id": None,
            "boundary_price": original_plan["stop_price"], "stop_price": original_plan["stop_price"],
            "available_at": original_plan["available_at"]})
    else:
        chosen = max(boundaries, key=lambda item: item["stop_price"], default=None)
    stop = chosen["stop_price"] if chosen else None
    distance = (price - stop) / price if price and stop else None
    reasons = []
    if price is None or price <= 0:
        reasons.append("reference_quote_unavailable")
    if atr is None or atr <= 0:
        reasons.append("prior_atr_unavailable")
    if stop is None:
        reasons.append("confirmed_reference_stop_unavailable")
    if distance is not None and (distance + 1e-12 < params["min_risk_fraction"] or distance - 1e-12 > params["max_stop_distance"]):
        reasons.append("reference_stop_distance_outside_contract")
    floor = max(stop + .01, stop / (1 - params["min_risk_fraction"])) if not reasons else None
    ceiling = min(price + params["max_next_open_gap_atr"] * atr, stop / (1 - params["max_stop_distance"])) if not reasons else None
    if floor is not None and floor > ceiling + 1e-12:
        reasons.append("reference_price_band_invalid")
        floor = ceiling = None
    cap = (original_plan["target_weight"] if original_plan else
        min(cfg["account"]["max_weight"], params["risk_fraction"] / distance)) if not reasons else None
    return {"analysis_mode": "reference", "reference_only": True, "account_scope": "none",
        "status": "unavailable" if reasons else "available", "symbol": row["symbol"],
        "date": row["date"], "available_at": at, "basis_date": row["date"],
        "reference_price": price, "reference_price_date": row["date"], "reference_price_source": "signal_day_close",
        "stop_price": stop, "invalidation_price": chosen["boundary_price"] if chosen else None,
        "stop_distance_fraction": distance, "atr14": atr,
        "reference_price_floor": floor, "reference_price_ceiling": ceiling,
        "reference_price_band": {"low": floor, "high": ceiling, "basis_date": row["date"],
            "reference_only": True, "not_fill_or_order_range": True},
        "risk_budget_fraction": params["risk_fraction"], "risk_position_cap_fraction": cap,
        "max_position_fraction": cfg["account"]["max_weight"],
        "min_stop_distance_fraction": params["min_risk_fraction"], "max_stop_distance_fraction": params["max_stop_distance"],
        "source_boundaries": boundaries, "source_event_ids": copy.deepcopy(original_plan["source_event_ids"])
            if original_plan else sorted(item["event_id"] for item in boundaries if item["event_id"]),
        "original_buy_plan_id": original_plan["plan_id"] if original_plan else None,
        "original_buy_plan_hash": stable_hash(original_plan) if original_plan else None,
        "reason_codes": reasons, "basis": "已确认边界与严格先验ATR形成的风险仓位上限；上限不等于建议建仓比例。"}


def _reference_buy_plan(row, cfg):
    """Read a live original BUY plan, never manufacture account state."""
    from my_strategy.services.czsc_trend_consensus import finite, observable
    c = cfg["consensus"]
    reasons = []
    buy, sell = finite(row.get("buy_base_score")), finite(row.get("sell_base_score"))
    families = row.get("buy_independent_families") or []
    if (row.get("direction") != "BUY" or buy is None or sell is None
            or buy + 1e-12 < c["buy_base_threshold"] or buy - sell + 1e-12 < c["margin"]
            or len(set(families)) < c["min_independent"]
            or row.get("buy_base_qualified") is not True or row.get("entry_eligible") is not True):
        reasons.append("no_qualified_live_original_buy_opportunity")
    plan = row.get("entry_plan")
    active = [item for item in row.get("active_candidates") or [] if item.get("side") == "BUY"]
    ids = sorted(set(row.get("original_opportunity_ids") or []))
    close, at = finite(row.get("reference_price")), row.get("available_at")
    if (not active or not ids or any(item.get("symbol") != row.get("symbol")
            or item.get("source") not in {"ma", "structure", "wyckoff"}
            or item.get("confirmed") is not True or item.get("input_eligible") is not True
            or not item.get("event_id") or not item.get("original_event_id")
            or item.get("opportunity_id") not in ids
            or not observable(item.get("available_at"), at)
            or any(item.get(key) and not observable(item[key], at) for key in
                ("first_confirmed_at", "first_available_at", "origin_observed_at", "observed_at"))
            for item in active)):
        reasons.append("original_buy_opportunity_identity_unavailable")
    if (not isinstance(plan, dict) or plan.get("status") != "active" or plan.get("entry_allowed") is not True
            or plan.get("policy") != "risk" or plan.get("valid_for_sessions") != 1
            or not plan.get("plan_id") or plan.get("symbol") != row.get("symbol")
            or plan.get("signal_date") != row.get("date") or plan.get("available_at") != at
            or finite(plan.get("reference_price")) != close
            or sorted(set(plan.get("original_opportunity_ids") or [])) != ids
            or sorted(plan.get("source_event_ids") or []) != sorted(item.get("event_id") for item in active)):
        reasons.append("original_buy_risk_plan_identity_mismatch")
        return None, list(dict.fromkeys(reasons))
    stop, weight, atr = (finite(plan.get(key)) for key in ("stop_price", "target_weight", "atr14"))
    if close is None or stop is None or atr is None or not 0 < stop < close or atr <= 0:
        reasons.append("original_buy_risk_plan_unavailable")
    else:
        distance = (close - stop) / close
        p = cfg["signal_points"]
        expected = min(cfg["account"]["max_weight"], p["risk_fraction"] * close / (close - stop))
        floor = max(stop + .01, stop / (1 - p["min_risk_fraction"]))
        ceiling = min(close + p["max_next_open_gap_atr"] * atr, stop / (1 - p["max_stop_distance"]))
        if (not p["min_risk_fraction"] <= distance <= p["max_stop_distance"]
                or weight != expected or plan.get("price_floor") != floor or plan.get("price_ceiling") != ceiling
                or floor > ceiling):
            reasons.append("original_buy_risk_geometry_mismatch")
    return (None if reasons else plan), list(dict.fromkeys(reasons))


def _reference_position(row, risk, *, current_position_fraction=None,
                        reference_quality=None, config=None, model_run_id=None):
    """Allocate only after an independently verified reference quality score."""
    from my_strategy.services.czsc_trend_consensus import finite
    cfg = strategy_config(config)
    ma = row["agents"]["ma"]
    reasons = list(risk["reason_codes"])
    if row["direction"] != "BUY":
        reasons.append("no_final_buy_consensus")
    if ma.get("status") != "scored":
        reasons.append("qualified_trend_model_unavailable")
    quality_status, probability, identity, population_hash, qualification_hash = (
        "reference_population_not_scored", None, {}, None, None)
    if reference_quality is None:
        reasons.append("entry_quality_not_evaluated_reference_population")
    else:
        from my_strategy.services.czsc_reference_quality import validate_reference_quality_score
        score_reasons = validate_reference_quality_score(reference_quality, row, risk, cfg, run_id=model_run_id)
        reasons.extend(score_reasons)
        if not score_reasons:
            quality_status = reference_quality.status
            identity = dict(reference_quality.model_identity)
            population_hash, qualification_hash = (reference_quality.reference_population_hash,
                reference_quality.qualification_hash)
            reasons.extend(reference_quality.reason_codes)
            if reference_quality.status == "scored" and reference_quality.fee_qualified is True:
                probability = finite(reference_quality.probability)
            if probability is None or not 0 <= probability <= 1:
                probability = None
                reasons.append("reference_quality_not_qualified")
        else:
            quality_status = "identity_error"
    original_plan, original_reasons = _reference_buy_plan(row, cfg)
    if row["direction"] == "BUY":
        reasons.extend(original_reasons)
    c = cfg["consensus"]
    buy_base = finite(row.get("buy_base_score"))
    total = buy_base + c["quality_weight"] * probability if buy_base is not None and probability is not None else None
    if probability is not None and probability < c["high_return_threshold"]:
        reasons.append("reference_quality_below_threshold")
    if total is not None and total + 1e-12 < c["buy_total_threshold"]:
        reasons.append("reference_total_support_below_threshold")
    if original_plan is not None and (risk.get("original_buy_plan_hash") != stable_hash(original_plan)
            or risk.get("symbol") != row["symbol"] or risk.get("date") != row["date"]
            or risk.get("available_at") != row["available_at"]
            or risk.get("reference_price") != row["reference_price"]
            or risk.get("stop_price") != original_plan["stop_price"]
            or risk.get("risk_position_cap_fraction") != original_plan["target_weight"]):
        reasons.append("reference_quality_risk_geometry_mismatch")
    reduce = row["direction"] == "SELL"
    current = finite(current_position_fraction)
    if current_position_fraction is not None and (current is None or not 0 <= current <= 1):
        raise ValueError("当前持仓比例必须为0到1之间的有限数")
    increase = row["direction"] == "BUY" and not reasons and probability is not None
    target = 0. if reduce else risk["risk_position_cap_fraction"] if increase else None
    return {"analysis_mode": "reference", "reference_only": True, "account_scope": "none",
        "status": "reduce_exposure" if reduce else "increase_exposure" if increase else
            "unavailable" if risk["status"] == "unavailable" else "pending",
        "action": row["direction"], "date": row["date"], "available_at": row["available_at"],
        "reference_price": risk["reference_price"], "reference_price_date": risk["reference_price_date"],
        "target_position_fraction": target, "risk_position_cap_fraction": risk["risk_position_cap_fraction"],
        "exposure_cap_fraction": 0. if reduce else risk["risk_position_cap_fraction"],
        "stop_price": risk["stop_price"], "invalidation_price": risk["invalidation_price"],
        "reference_price_band": copy.deepcopy(risk["reference_price_band"]),
        "risk_budget_fraction": risk["risk_budget_fraction"], "current_position_fraction": current,
        "current_position_basis": "manual_comparison_input" if current is not None else None,
        "current_position_is_historical_inventory": False,
        "comparison_scope": "hypothetical_reference_on_basis_date",
        "change_fraction": target - current if target is not None and current is not None else None,
        "quality_status": quality_status, "reference_quality_probability": probability,
        "reference_quality_identity": copy.deepcopy(identity), "reference_population_hash": population_hash,
        "reference_quality_qualification_hash": qualification_hash,
        "reference_quality_threshold": c["high_return_threshold"],
        "reference_buy_total_score": total, "reference_buy_total_threshold": c["buy_total_threshold"],
        "entry_probability": None, "exit_probability": None,
        "reason_codes": list(dict.fromkeys(reasons)),
        "basis": "最终S表示降低参考敞口，不增加该方向风险；没有实际持仓退出概率。" if reduce else
            "独立参考质量与费用证据合格，原有效买方机会和风险约束均通过；目标采用原风险预算仓位。" if increase else
            "推荐尚未生成；风险上限仅用于约束参考敞口，缺少完整质量资格时不建议增加仓位。"}


def reference_overlay(features, calendar, *, model_run_id=None, model_fold=None,
                      model_mode="historical", as_of=None, device="cpu",
                      source_context_hash=None, consensus_config=None, use_models=True,
                      model_selection_frozen=False, current_position_fraction=None):
    """Score each observed close independently, without inventory or a Broker.

    Direction retains the frozen base-score, margin and independent-source
    thresholds. A separate, fee-qualified reference population can support a
    risk allocation. It never consumes an opportunity or infers an actual
    account entry or held-position exit probability.
    """
    from my_strategy.services.czsc_trend_model import TrendSession, bind_trend_inputs
    from my_strategy.services.czsc_trend_pipeline import forward_trend
    from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions, consensus_config as validate
    from my_strategy.services.czsc_native_evidence import build_native_evidence
    from my_strategy.services.czsc_trend_math import _device
    from my_strategy.adapters.czsc_adapter import date_cutoff

    if model_mode not in {"historical", "production"}:
        raise ValueError("参考分析只支持按时点判断")
    actual_device = str(_device(device or "cpu"))
    frame = features.copy()
    if frame.empty:
        raise ValueError("参考分析需要单股按日有序唯一行情")
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    requested_cutoff = date_cutoff(as_of or str(frame.iloc[-1].date))
    visible_sessions = [day for day in calendar if pd.Timestamp(day) + pd.Timedelta(hours=15) <= requested_cutoff]
    if not visible_sessions:
        raise ValueError("请求时点之前没有已收盘的核验交易日")
    cutoff = visible_sessions[-1]
    frame = frame[frame.date <= cutoff].reset_index(drop=True)
    if frame.empty or frame.symbol.nunique() != 1 or list(frame.date) != sorted(set(frame.date)):
        raise ValueError("参考分析需要单股按日有序唯一行情")
    load_selected = use_models and not (model_selection_frozen and model_run_id is None)
    root, report = load_reference_model_run(model_run_id, as_of=cutoff) if load_selected else (None, None)
    cfg = strategy_config(report["config"]) if report else strategy_config()
    if consensus_config is not None:
        cfg["consensus"].update(validate({**cfg["consensus"], **consensus_config}))
    pack = build_native_evidence(frame, cfg, market_dates=calendar)
    frame = bind_trend_inputs(pack["frame"], calendar)
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    bundles = copy.deepcopy((report or {}).get("trend_bundles", []))
    if use_models and model_fold:
        if report is None:
            raise ValueError("固定检查点需要完整模型")
        available = [bundle for bundle in bundles if bundle.get("available_at", "9999") < cutoff
            and bundle.get("heads", {}).get("trend_direction")]
        matching = available if model_fold == "production" else [bundle for bundle in available if bundle["name"] == model_fold]
        if not matching:
            raise ValueError("固定趋势检查点在此判断时点尚不可用")
        bundles = [max(matching, key=lambda bundle: (bundle["available_at"], bundle["name"]))]
    session = TrendSession() if use_models else None
    trends = forward_trend(frame, root or ARTIFACT_RUNS_ROOT, bundles, cfg, session,
        model_mode, market_dates=calendar) if use_models else {day: {
            "date": day, "available_at": day + "T15:00:00+08:00", "input_eligible": True,
            "model_available": False, "p_up": None, "p_down": None, "p_neutral": None,
            "model_identity": {}, "reason_codes": ["models_disabled_for_structure_reference"]} for day in frame.date}
    prepared = prepare_trend_decisions(frame, calendar, native_pack=pack,
        trend_probabilities=trends, config=cfg, mode="E0")
    quality_session, reference_prefixes = None, {}
    if use_models and root is not None and report is not None:
        from my_strategy.services.czsc_reference_quality import ReferenceQualitySession
        from my_strategy.services.czsc_trend_features import event_prefix_rows, original_evidence_events
        quality_session = ReferenceQualitySession(root, report, config=cfg, usage_mode=model_mode)
        reference_prefixes = dict(zip(frame.date, event_prefix_rows(frame,
            original_evidence_events(prepared), calendar)))
    following = dict(zip(calendar[:-1], calendar[1:]))
    rows, points, source_points, counts = [], [], [], Counter()
    observed_bars = frame.to_dict("records")
    native_days = {str(value["date"])[:10]: value for value in pack.get("evaluations", [])}
    for index, original in enumerate(prepared["rows"]):
        row = copy.deepcopy(original)
        day = row["date"]
        ma = {**copy.deepcopy(trends[day]), **copy.deepcopy(row["agents"]["ma"])}
        scored = ma.get("status") == "scored"
        # Invalid or missing heads remain unavailable, never neutral/zero.
        if not scored:
            ma.update(p_up=None, p_down=None, p_neutral=None)
        ma.update(kind="model_probability", target="next_verified_open_horizon_direction",
            in_scope=scored, model_identity=_display_identity(ma.get("model_identity"), model_mode))
        row["agents"]["ma"] = ma
        for source in ("structure", "wyckoff"):
            agent = row["agents"][source]
            agent.update(kind="rule_evidence", input_eligible=agent["status"] == "scored")
            if source == "wyckoff":
                agent.update(effective_demand=agent.get("buy_strength"), effective_supply=agent.get("sell_strength"))
        row["agents"]["high_return"] = {
            "label": "高收益", "kind": "model_probability", "status": "reference_out_of_scope",
            "evaluation_status": "reference_out_of_scope", "probability": None,
            "in_scope": False, "applied_to_entry": False, "buy_contribution": None,
            "sell_contribution": None, "quality_contribution": None,
            "reason_codes": ["reference_has_no_actual_flat_account_population"],
            "target": "actual_complete_fee_net_roundtrip_high_return_and_initial_risk"}
        action = row["direction"]
        basis = {"BUY": "多源方向支持达到买方参考门槛。", "SELL": "多源方向支持达到卖方参考门槛。",
            "WAIT": "当前多源方向支持未达到参考门槛。"}[action]
        reasons = [reason for reason in row.get("reason_codes", []) if not (
            reason.startswith("risk_") or reason in {"no_live_original_buy_opportunity",
                "high_return_or_total_support_not_qualified"})]
        recommendation = {"action": action, "reference_only": True, "analysis_mode": "reference",
            "date": day, "signal_date": day, "available_at": row["available_at"], "basis": basis,
            "reference_price": row["reference_price"], "reference_close": row["reference_price"],
            "reference_price_date": day, "reference_price_source": "signal_day_close",
            "execution_basis": "已知收盘价上的方向参考，非成交或账户操作。"}
        risk = _reference_risk_plan(row, observed_bars, index, native_days.get(day, {}), cfg)
        quality = quality_session.score(observed_bars[index], row, reference_prefixes.get(day, {}),
            risk_plan=risk) if quality_session is not None else None
        position = _reference_position(row, risk, current_position_fraction=current_position_fraction,
            reference_quality=quality, config=cfg, model_run_id=root.name if root is not None else None)
        row["agents"]["reference_quality"] = {
            "label": "参考质量", "kind": "model_probability", "status": position["quality_status"],
            "probability": position["reference_quality_probability"],
            "probability_scope": "independent_reference_buy_opportunity_fee_net_quality",
            "target": "complete_fee_net_roundtrip_high_return_and_initial_risk",
            "in_scope": position["reference_quality_probability"] is not None,
            "threshold": cfg["consensus"]["high_return_threshold"],
            "model_identity": copy.deepcopy(position["reference_quality_identity"]),
            "reference_population_hash": position["reference_population_hash"],
            "qualification_hash": position["reference_quality_qualification_hash"],
            "applied_to_reference_position": position["status"] == "increase_exposure",
            "applied_to_actual_entry": False, "entry_probability": None, "exit_probability": None,
            "reason_codes": copy.deepcopy(position["reason_codes"])}
        day_source_points = [point for event in row.get("candidate_evidence", [])
            if (point := _source_reference_point(event, row)) is not None]
        source_points.extend(day_source_points)
        recommendation.update(reference_risk_plan=copy.deepcopy(risk), position_reference=copy.deepcopy(position))
        row.update(reference_risk_plan=risk, position_reference=position,
            source_reference_points=day_source_points)
        for key in ("entry_plan", "raw_risk_plan", "entry_eligible", "original_opportunity_ids", "mode"):
            row.pop(key, None)
        row.update(analysis_mode="reference", reference_only=True, account_scope="none",
            reference_action=action, action=action, reference_recommendation=recommendation,
            reason_codes=reasons, emitted=action in {"BUY", "SELL"},
            events=copy.deepcopy(row.get("candidate_evidence", [])),
            reference_close=row["reference_price"], reference_price_date=day,
            reference_price_source="signal_day_close", next_market_session=following.get(day),
            high_return_probability=None, exit_probability=None, buy_quality_bonus=None,
            buy_total_score=row["buy_base_score"],
            reference_quality_probability=position["reference_quality_probability"],
            reference_buy_total_score=position["reference_buy_total_score"],
            reference_quality_identity=copy.deepcopy(position["reference_quality_identity"]),
            model_resolution={**ma.get("model_identity", {}), "model_family": STRATEGY_ID,
                "schema_family": STRATEGY_ID, "display_name": DISPLAY_NAME, "probability": None,
                "direction_probabilities": {key: ma.get(key) for key in ("p_up", "p_down", "p_neutral")},
                "reason_codes": ma.get("reason_codes", []), "publication_allowed": False})
        if row["emitted"]:
            point = {"event_id": stable_hash({"purpose": "reference", "symbol": row["symbol"],
                    "date": day, "action": action, "config_hash": prepared["config_hash"]}),
                "date": day, "signal_date": day, "available_at": row["available_at"],
                "first_available_at": row["available_at"], "symbol": row["symbol"],
                "side": action, "action": action, "source": "reference_consensus", "reference_layer": "consensus",
                "event_type": "direction_reference", "reference_only": True, "analysis_mode": "reference",
                "reference_price": row["reference_price"], "reference_price_date": day,
                "reference_price_source": "signal_day_close", "reason_codes": reasons,
                "source_event_ids": copy.deepcopy(row.get("fresh_candidate_ids", [])),
                "buy_base_score": row["buy_base_score"], "sell_base_score": row["sell_base_score"]}
            row["event_id"] = point["event_id"]
            points.append(point)
        counts[ma.get("status", "unavailable")] += 1
        rows.append(row)
    prepared.update(rows=copy.deepcopy(rows), events=points, direction_points=points,
        mode="reference", analysis_mode="reference", reference_only=True,
        candidate_policy="independent_daily_direction_reference")
    latest = copy.deepcopy(rows[-1]) if rows[-1]["date"] == cutoff else {
        "date": cutoff, "symbol": str(frame.iloc[0].symbol), "available_at": cutoff + "T15:00:00+08:00",
        "action": "WAIT", "direction": "UNAVAILABLE", "reference_action": "WAIT",
        "reference_price": None, "reference_close": None, "reference_price_date": cutoff,
        "events": [], "agents": {source: {"status": "input_ineligible", "in_scope": False,
            "p_up": None, "p_down": None, "p_neutral": None, "probability": None,
            "buy_contribution": None, "sell_contribution": None,
            "reason_codes": ["missing_target_market_session"]} for source in ("ma", "structure", "wyckoff")},
        "buy_base_score": None, "sell_base_score": None,
        "high_return_probability": None, "exit_probability": None,
        "reason_codes": ["missing_target_market_session"],
        "reference_recommendation": {"action": "WAIT", "reference_only": True, "analysis_mode": "reference",
            "basis": "请求交易日缺少行情，参考判断不可用。", "reference_price": None},
        "model_resolution": {"probability": None, "reason_codes": ["missing_target_market_session"]}}
    if "reference_risk_plan" not in latest:
        latest["reference_risk_plan"] = _reference_risk_plan(latest, [], 0, {}, cfg)
        latest["position_reference"] = _reference_position(latest, latest["reference_risk_plan"],
            current_position_fraction=current_position_fraction, config=cfg,
            model_run_id=root.name if root is not None else None)
        latest["source_reference_points"] = []
        latest["reference_recommendation"].update(reference_risk_plan=copy.deepcopy(latest["reference_risk_plan"]),
            position_reference=copy.deepcopy(latest["position_reference"]))
    latest.update(analysis_mode="reference", reference_only=True, account_scope="none",
        model_family=STRATEGY_ID, schema_family=STRATEGY_ID, display_name=DISPLAY_NAME,
        probability_scope="observed_market_direction", next_market_session=following.get(cutoff),
        model_policy="pinned" if model_fold and use_models else "auto" if use_models else "disabled",
        model_fold=model_fold if use_models else None,
        resolved_model_fold=bundles[0]["name"] if model_fold and use_models and bundles else None)
    stats = copy.deepcopy(getattr(session, "diagnostics", {}))
    reference_stats = copy.deepcopy(getattr(quality_session, "diagnostics", {}))
    return {"strategy_version": STRATEGY_ID, "display_name": DISPLAY_NAME,
        "analysis_mode": "reference", "reference_only": True, "account_scope": "none",
        "models_enabled": bool(use_models), "structure_only": not use_models,
        "rows": rows, "trend_consensus": prepared, "reference_points": points,
        "source_reference_points": source_points,
        "reference_risk_plan": copy.deepcopy(latest["reference_risk_plan"]),
        "position_reference": copy.deepcopy(latest["position_reference"]),
        "reference_quality_probability": latest["position_reference"]["reference_quality_probability"],
        "reference_recommendation": copy.deepcopy(latest["reference_recommendation"]),
        "account_actions": None, "latest_decision": latest, "features": frame, "native_pack": pack,
        "wyckoff_events": pack.get("wyckoff_events", []), "usage_mode": model_mode,
        "model_run_id": root.name if root else None, "model_gate": (report or {}).get("model_gate", {"passed": False}),
        "publication_allowed": False, "source_context_hash": source_context_hash,
        "pricing_assumptions": copy.deepcopy(REFERENCE_PRICING_ASSUMPTIONS),
        "compute_info": {"model_family": STRATEGY_ID, "display_name": DISPLAY_NAME,
            "selected_device": actual_device, "actual_device": "cpu", "model_inference_device": "cpu",
            "research": True, "account_simulated": False,
            "actual_cuda_inference": False, "actual_mps_inference": False,
            "model_inference_rows": int(stats.get("rows", 0)) + int(reference_stats.get("rows", 0)),
            "model_inference_batches": int(stats.get("batches", 0)) + int(reference_stats.get("batches", 0)),
            "model_loads": int(stats.get("model_loads", 0)) + int(reference_stats.get("model_loads", 0)),
            "trend_inference": stats, "reference_quality_inference": reference_stats,
            "source_reason_counts": dict(counts)}}


def load_model_run(model_run_id=None, *, as_of=None, scope='research'):
    """Default inference consumes only a qualified, effective registry selection."""
    from my_strategy.services.czsc_trend_pipeline import read_run
    if model_run_id is None:
        from my_strategy.services.model_training_center import TrainingCenter
        model_run_id = TrainingCenter().active_model_run_id(as_of=as_of, scope=scope)
        if not model_run_id:
            return None, None
    return read_run(model_run_id)


def load_reference_model_run(model_run_id=None, *, as_of=None):
    """A fixed reference checkpoint cannot bypass the qualified selection."""
    if model_run_id is not None:
        from my_strategy.services.model_training_center import TrainingCenter
        selected = TrainingCenter().active_model_run_id(as_of=as_of, scope="research")
        if str(model_run_id) != str(selected) or selected is None:
            raise ValueError("参考分析固定模型必须是此时点已合格且生效的当前选择；候选诊断请使用模型训练诊断")
    return load_model_run(model_run_id, as_of=as_of, scope="research")


def _display_identity(identity, mode):
    value = copy.deepcopy(identity or {})
    value.update(usage_mode=mode, historical_reconstruction=mode == "historical")
    return value


def _entry_status(status):
    if str(status).startswith("risk_"):
        return "risk_blocked"
    return {"no_prior_high_return_head": "model_unavailable",
        "candidate_risk_ineligible": "risk_blocked"}.get(status, status)


def _pinned_catalogues(report, checkpoint, cutoff):
    entries = [b for b in report.get("model_bundles", []) if b["name"] == checkpoint]
    if not entries or entries[0]["available_at"] >= cutoff:
        raise ValueError("固定检查点在此判断时点尚不可用")
    # Pin only the entry filter. Ordinary trend/exit continuation keeps the
    # same strictly chronological catalogue that generated its E1 targets.
    # forward_trend enforces every individual head's availability clock.
    return copy.deepcopy(report.get("trend_bundles", [])), entries


def _latest_observed_decision(rows, account_rows, cutoff, symbol):
    if rows[-1]["date"] == cutoff:
        return copy.deepcopy(rows[-1])
    reason = "missing_target_market_session"
    return {"date": cutoff, "symbol": symbol, "available_at": cutoff + "T15:00:00+08:00",
        "reference_price": None, "reference_close": None, "reference_price_date": cutoff,
        "reason_codes": [reason], "events": [], "candidate_evidence": [], "entry_plan": None,
        "buy_base_score": None, "sell_base_score": None, "buy_total_score": None,
        "high_return_probability": None, "exit_probability": None,
        "account_action": copy.deepcopy(account_rows.get(cutoff)), "holding": None,
        "model_resolution": {"probability": None, "reason_codes": [reason]},
        "agents": {source: {"status": "input_ineligible", "evaluation_status": "input_ineligible",
            "in_scope": False, "input_eligible": False, "kind": "rule_evidence" if source in {"structure", "wyckoff"} else "model_probability",
            "p_up": None, "p_down": None, "p_neutral": None, "probability": None,
            "buy_contribution": None, "sell_contribution": None, "reason_codes": [reason]}
            for source in ("ma", "structure", "wyckoff", "high_return")}}


def runtime_overlay(features, calendar, *, model_run_id=None, model_fold=None, model_mode="historical",
                    as_of=None, device="cpu", account_start=None, initial_cash=None,
                    source_context_hash=None, consensus_config=None):
    from my_strategy.services.czsc_trend_model import TrendSession, bind_trend_inputs
    from my_strategy.services.czsc_trend_interval_model import HeadSession
    from my_strategy.services.czsc_trend_pipeline import forward_trend, predict_entry, pinned_exit_policy
    from my_strategy.services.czsc_trend_features import event_prefix_rows, original_evidence_events
    from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
    from my_strategy.services.czsc_trend_account import replay_trend_account
    from my_strategy.services.czsc_native_evidence import build_native_evidence
    from my_strategy.services.czsc_trend_math import _device

    if model_mode not in {"historical", "production"}:
        raise ValueError("趋势共识策略只支持按时点判断")
    actual_device = str(_device(device or "cpu"))
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    cutoff = str(as_of or frame.iloc[-1].date)[:10]
    frame = frame[frame.date <= cutoff].reset_index(drop=True)
    if frame.empty or frame.symbol.nunique() != 1 or list(frame.date) != sorted(set(frame.date)):
        raise ValueError("趋势共识输入需要单股按日有序唯一行情")
    root, report = load_model_run(model_run_id, as_of=cutoff)
    cfg = strategy_config(report["config"]) if report else strategy_config()
    label_target = {"min_net_return": float(cfg["high_return"]["min_net_return"]),
                    "min_r_multiple": float(cfg["high_return"]["min_r_multiple"])}
    account_start = str(account_start or cfg["account"].get("start", cfg["feature_start"]))[:10]
    if consensus_config is not None:
        from my_strategy.services.czsc_trend_consensus import consensus_config as validate
        # A partial live-quality override inherits this model's frozen policy,
        # rather than resetting selected thresholds to the template defaults.
        changed = validate({**cfg["consensus"], **consensus_config})
        if report and any(changed[k] != cfg["consensus"][k] for k in
            ("source_weights", "quality_weight", "buy_base_threshold", "sell_threshold", "margin",
             "min_independent", "min_source_strength", "cooldown", "evidence_valid_bars", "signal_valid_bars")):
            raise ValueError("候选生成或普通退出政策改变，需要重建对应入场标签和模型")
        cfg["consensus"].update(changed)
    if initial_cash is not None and float(initial_cash) != float(cfg["initial_cash"]):
        raise ValueError("账户初始资金与模型训练契约不匹配，请训练对应资金条件模型")
    pack = build_native_evidence(frame, cfg, market_dates=calendar)
    frame = bind_trend_inputs(pack["frame"], calendar)
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    trend_session, binary_session = TrendSession(), HeadSession()
    model_root = root or ARTIFACT_RUNS_ROOT
    trend_bundles = report.get("trend_bundles", []) if report else []
    entry_bundles = report.get("model_bundles", []) if report else []
    if model_fold:
        if report is None:
            raise ValueError("固定检查点需要完整模型")
        trend_bundles, entry_bundles = _pinned_catalogues(report, model_fold, cutoff)
    trends = forward_trend(frame, model_root, trend_bundles, cfg, trend_session, model_mode, market_dates=calendar)
    prepared = prepare_trend_decisions(frame, calendar, native_pack=pack,
        trend_probabilities=trends, config=cfg, mode="E0")
    facts = original_evidence_events(prepared)
    prefixes = event_prefix_rows(frame, facts, calendar)
    values_by_day = {row["date"]: row for row in frame.to_dict("records")}
    prefix_by_day = dict(zip(frame.date, prefixes, strict=True))
    evaluation_by_day = {}

    def evaluate_entry(snapshot, candidate):
        day = snapshot["date"]
        plan = candidate.get("entry_plan") or {}
        p, identity, status = predict_entry(values_by_day[day], snapshot, candidate, plan,
            prefix_by_day[day], model_root, entry_bundles, cfg, binary_session, model_mode)
        normalized = _entry_status(status)
        passed = status == "scored" and p is not None and p >= cfg["consensus"]["high_return_threshold"]
        value = {"date": day, "available_at": day + "T15:00:00+08:00", "kind": "model_probability",
            "status": normalized, "evaluation_status": ("scored_passed" if passed else "scored_rejected") if status == "scored" else normalized,
            "probability": p, "model_identity": _display_identity(identity, model_mode), "in_scope": status == "scored",
            "threshold": cfg["consensus"]["high_return_threshold"],
            "target": "actual_complete_fee_net_roundtrip_high_return_and_initial_risk",
            "reason_codes": ([] if passed else ["high_return_below_threshold"]) if status == "scored" else [status],
            "probability_scope": "actual_flat_account_at_signal_close", "usage_mode": model_mode,
            "training_population": TRAINING_POPULATION,
            **{key: copy.deepcopy(candidate.get(key)) for key in
                ("attempt_id", "economic_group_id", "original_opportunity_ids", "reference_state_hash", "flat_state_hash", "entry_plan_hash")}}
        evaluation_by_day[day] = copy.deepcopy(value)
        return value

    evaluate_entry.recipe_hash = stable_hash({"strategy": STRATEGY_ID, "entry_models": entry_bundles,
        "capital": cfg["initial_cash"], "scope": "actual_flat_account_at_signal_close", "continuation": "E1"})
    exit_policy = pinned_exit_policy(frame, calendar, prepared, model_root,
        report.get("ma_exit_bundles", []) if report else [], cfg, binary_session, model_mode)
    identity = {"model_run_id": root.name if root else None, "strategy": STRATEGY_ID,
        "model_policy": "pinned" if model_fold else "auto", "model_fold": model_fold,
        "protocol_freeze_hash": (report or {}).get("protocol_freeze", {}).get("freeze_hash"),
        "usage_mode": model_mode}
    account_id = "research:" + stable_hash({"symbol": str(frame.iloc[0].symbol), "account_start": account_start,
        "config": cfg, "models": identity, "data_context": source_context_hash})[:24]
    account = replay_trend_account(frame, calendar, prepared, config=cfg, exit_policy=exit_policy,
        entry_evaluator=evaluate_entry, start=account_start, end=cutoff, mode="final",
        run_id=account_id, account_id=account_id, model_identity=identity,
        source_context_hash=source_context_hash)
    for account_row in account.get("rows", []):
        for key in ("exit_evaluation", "entry_evaluation"):
            evaluation = account_row.get(key)
            if isinstance(evaluation, dict) and evaluation.get("model_identity"):
                evaluation["model_identity"] = _display_identity(evaluation["model_identity"], model_mode)
        reasons = list(account_row.get("reason_codes") or [])
        for intent in account_row.get("final_action_intents", []):
            reasons.extend(intent.get("all_triggered_reasons") or [intent["reason"]])
        if not account_row.get("emitted"):
            reasons.extend((account_row.get("entry_evaluation") or {}).get("reason_codes") or [])
        account_row["reason_codes"] = list(dict.fromkeys(reasons))
    account_rows = {r["date"]: r for r in account.get("rows", [])}
    attempts_by_day = {r["date"]: r for r in account.get("entry_diagnostics", [])}
    holding_by_day = {r["date"]: r for r in account.get("holding_snapshots", [])}
    following = dict(zip(calendar[:-1], calendar[1:]))
    rows, counts = [], Counter()
    for original in prepared["rows"]:
        row = copy.deepcopy(original)
        day = row["date"]
        ma = {**copy.deepcopy(trends[day]), **copy.deepcopy(row["agents"]["ma"])}
        ma.update(kind="model_probability", target="next_verified_open_horizon_direction",
            in_scope=bool(trends[day].get("model_available") and trends[day].get("input_eligible")),
            model_identity=_display_identity(trends[day].get("model_identity"), model_mode))
        row["agents"]["ma"] = ma
        for source in ("structure", "wyckoff"):
            agent = row["agents"][source]
            agent.update(kind="rule_evidence", input_eligible=agent["status"] == "scored")
            if source == "wyckoff":
                agent.update(effective_demand=agent.get("buy_strength"), effective_supply=agent.get("sell_strength"))
        # The Broker validates identity, population and both quality gates.
        # Export that actual decision rather than the adapter's unvalidated copy.
        quality = copy.deepcopy((account_rows.get(day) or {}).get("entry_evaluation") or evaluation_by_day.get(day))
        if quality is not None:
            quality.setdefault("kind", "model_probability")
            quality.setdefault("threshold", cfg["consensus"]["high_return_threshold"])
            original_status = quality.get("status", "model_unavailable")
            quality.setdefault("reason_codes", [] if original_status == "scored" else [original_status])
            quality["status"] = _entry_status(original_status)
            reasons = quality["reason_codes"]
            if quality.get("evaluation_status") == "identity_error" or "high_return_probability_invalid" in reasons:
                quality.update(status="identity_error", evaluation_status="identity_error",
                    raw_probability=quality.get("probability"), probability=None, in_scope=False)
            elif "high_return_population_out_of_scope" in reasons:
                quality.update(status="input_ineligible", evaluation_status="input_ineligible",
                    raw_probability=quality.get("probability"), probability=None, in_scope=False)
        if quality is None:
            plan = row.get("entry_plan") or {}
            status = "risk_blocked" if plan and plan.get("entry_allowed") is False else "no_candidate"
            reason = "risk_blocked_unscored" if status == "risk_blocked" else "no_qualified_direction_candidate"
            if day in holding_by_day:
                status, reason = "actual_existing_position", "actual_existing_position"
            quality = {"status": status, "evaluation_status": status, "probability": None,
                "kind": "model_probability", "in_scope": False, "reason_codes": [reason],
                "threshold": cfg["consensus"]["high_return_threshold"],
                "probability_scope": "actual_flat_account_at_signal_close"}
        if cfg["consensus"]["require_high_return"] is False:
            quality.update(status="entry_quality_gate_disabled", evaluation_status="entry_quality_gate_disabled",
                probability=None, in_scope=False, reason_codes=["high_return_filter_disabled"],
                applied_to_entry=False)
        else:
            quality["applied_to_entry"] = quality.get("status") == "scored"
        quality.setdefault("target", "actual_complete_fee_net_roundtrip_high_return_and_initial_risk")
        attempt = attempts_by_day.get(day) or {}
        quality.update(training_population=TRAINING_POPULATION,
            attempt_identity=ATTEMPT_IDENTITY, economic_group_identity=ECONOMIC_GROUP_IDENTITY)
        for key in ("attempt_id", "economic_group_id", "original_opportunity_ids", "reference_state_hash", "flat_state_hash", "entry_plan_hash"):
            if key in attempt:
                quality[key] = copy.deepcopy(attempt[key])
        quality["label_target"] = copy.deepcopy(label_target)
        quality["configured_weight"] = cfg["consensus"]["quality_weight"]
        quality["quality_contribution"] = quality["buy_contribution"] = (
            cfg["consensus"]["quality_weight"] * quality["probability"] if quality["probability"] is not None else None)
        quality["sell_contribution"] = 0.
        row["agents"]["high_return"] = quality
        row["buy_quality_bonus"] = quality["quality_contribution"]
        row["buy_total_score"] = row["buy_base_score"] + (quality["quality_contribution"] or 0.)
        row.update(events=copy.deepcopy(row.get("candidate_evidence", [])), reference_close=row["reference_price"],
            reference_price_date=day, next_market_session=following.get(day),
            high_return_probability=quality["probability"], exit_probability=None,
            entry_attempt={key: copy.deepcopy(quality.get(key)) for key in
                ("attempt_id", "economic_group_id", "original_opportunity_ids", "reference_state_hash", "flat_state_hash", "entry_plan_hash")}
                if attempt else None,
            account_action=copy.deepcopy(account_rows.get(day)), holding=copy.deepcopy(holding_by_day.get(day)),
            model_resolution={**quality.get("model_identity", {}), "model_family": STRATEGY_ID,
                "schema_family": STRATEGY_ID, "display_name": DISPLAY_NAME,
                "probability": quality["probability"], "reason_codes": quality["reason_codes"],
                "training_population": TRAINING_POPULATION,
                "attempt_id": quality.get("attempt_id"), "economic_group_id": quality.get("economic_group_id"),
                "original_opportunity_ids": copy.deepcopy(quality.get("original_opportunity_ids")), "publication_allowed": False})
        counts[quality["status"]] += 1
        rows.append(row)
    prepared["rows"] = copy.deepcopy(rows)
    for event in prepared["events"]:
        event.pop("consensus", None)
    prepared["direction_points"] = prepared["events"]
    account.setdefault("context", account.get("series_context", {}))
    account.setdefault("intents", account.get("final_action_intents", []))
    latest = _latest_observed_decision(rows, account_rows, cutoff, str(frame.iloc[0].symbol))
    if latest.get("account_action"):
        latest.update(action=latest["account_action"].get("action", "WAIT"),
            position_intent=latest["account_action"].get("action", "WAIT"))
    else:
        latest.update(action="WAIT", position_intent="WAIT")
    latest.update(model_family=STRATEGY_ID, schema_family=STRATEGY_ID, display_name=DISPLAY_NAME,
        account_context=account["context"], probability_scope="actual_flat_account_at_signal_close")
    trend_stats = copy.deepcopy(getattr(trend_session, "diagnostics", {}))
    interval_stats = copy.deepcopy(getattr(binary_session, "diagnostics", {}))
    return {"strategy_version": STRATEGY_ID, "display_name": DISPLAY_NAME, "rows": rows,
        "trend_consensus": prepared, "account_actions": account, "latest_decision": latest,
        "features": frame, "native_pack": pack, "wyckoff_events": pack.get("wyckoff_events", []),
        "usage_mode": model_mode, "model_run_id": root.name if root else None,
        "model_gate": (report or {}).get("model_gate", {"passed": False}), "publication_allowed": False,
        "pricing_assumptions": copy.deepcopy(PRICING_ASSUMPTIONS),
        "high_return_target": copy.deepcopy(label_target),
        "compute_info": {"model_family": STRATEGY_ID, "display_name": DISPLAY_NAME,
            "selected_device": actual_device, "actual_device": "cpu", "model_inference_device": "cpu",
            "research": True, "actual_cuda_inference": False, "actual_mps_inference": False,
            "model_inference_rows": int(trend_stats.get("rows", 0)) + int(interval_stats.get("rows", 0)),
            "model_inference_batches": int(trend_stats.get("batches", 0)) + int(interval_stats.get("batches", 0)),
            "model_loads": int(trend_stats.get("model_loads", 0)) + int(interval_stats.get("model_loads", 0)),
            "trend_inference": trend_stats, "interval_inference": interval_stats,
            "source_reason_counts": dict(counts)}}
