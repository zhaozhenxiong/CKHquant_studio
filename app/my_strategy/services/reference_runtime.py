"""Causal, per-symbol reference exposure paths; no account, fills or broker state."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash

REFERENCE_FAMILY = "reference_position_v1"
DISPLAY_NAME = "参考买卖点与逐股仓位"
DEFAULT_POSITION_POLICY = {
    "version": REFERENCE_FAMILY, "max_stock_weight": .20, "risk_budget": .005,
    "position_step": .025, "hysteresis": .005, "edge_entry": .05, "edge_full": .35,
    "wait_band": .03, "atr_stop_multiple": 2.5, "min_stop_distance": .01,
    "max_stop_distance": .25,
}


def finite(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def validate_position_policy(config=None):
    supplied = (config or {}).get("position", config or {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_POSITION_POLICY):
        raise ValueError("未知逐股参考仓位参数")
    result = {**DEFAULT_POSITION_POLICY, **supplied}
    if result["version"] != REFERENCE_FAMILY:
        raise ValueError("参考仓位版本不匹配")
    for key in set(result) - {"version"}:
        value = finite(result[key])
        if value is None or value < 0:
            raise ValueError("参考仓位参数必须是非负有限数：" + key)
        result[key] = value
    if not (0 < result["max_stock_weight"] <= 1 and 0 < result["risk_budget"] <= 1
            and 0 < result["position_step"] <= result["max_stock_weight"]
            and result["hysteresis"] < result["position_step"] / 2
            and 0 <= result["wait_band"] <= result["edge_entry"] < result["edge_full"] <= 1
            and 0 < result["min_stop_distance"] < result["max_stop_distance"] < 1
            and result["atr_stop_multiple"] > 0):
        raise ValueError("参考仓位步长、滞回、概率优势或风险参数范围错误")
    return result


class ReferencePositionPath:
    """A reference trajectory, never actual inventory or portfolio allocation.

    Missing scores carry the last reference as stale. The optional manual
    position is a separate comparison and cannot alter this trajectory.
    """
    def __init__(self, policy=None):
        self.policy = validate_position_policy(policy)
        self.target = 0.0
        self.last_scored_date = None
        self.stop_price = None
        self.steps = 0

    def advance(self, *, day, p_up=None, p_down=None, risk_cap=None, stop_price=None,
                scored=True, predictive_passed=True, hard_invalidation=False,
                current_position_fraction=None, reason_codes=()):
        policy, previous, prior_stop = self.policy, self.target, self.stop_price
        up, down, cap = finite(p_up), finite(p_down), finite(risk_cap)
        current = finite(current_position_fraction)
        if current_position_fraction is not None and (current is None or not 0 <= current <= 1):
            raise ValueError("当前仓位比较输入必须为0到1之间的有限数")
        reasons = list(reason_codes)
        usable = bool(scored and up is not None and down is not None and 0 <= up <= 1
            and 0 <= down <= 1 and up + down <= 1 + 1e-6 and cap is not None and 0 <= cap <= 1)
        edge, score, continuous = None, None, None
        target = previous
        if not usable:
            reasons.append("reference_input_or_model_unavailable")
            status = "stale"
            # An observed breach of an already frozen boundary is a separate
            # risk fact. Missing probabilities alone can never manufacture it.
            if hard_invalidation:
                target = 0.0
                reasons.append("reference_frozen_stop_invalidated")
        else:
            edge = up - down
            score = max(0.0, min(1.0, (edge - policy["edge_entry"]) /
                                         (policy["edge_full"] - policy["edge_entry"])))
            cap = min(cap, policy["max_stock_weight"])
            # The ceiling may drift every day. Quantize it down before any
            # allocation so all target changes remain whole reference levels.
            cap_level = math.floor((cap + 1e-12) / policy["position_step"]) * policy["position_step"]
            continuous = cap * score
            if hard_invalidation:
                target = 0.0
                reasons.append("reference_frozen_stop_invalidated")
            elif not predictive_passed:
                # A failed predictive gate is unavailable evidence, never a
                # bearish score. Only the observable risk ceiling can reduce.
                target = min(previous, cap_level)
                reasons.append("reference_predictive_gate_pending")
            elif abs(edge) <= policy["wait_band"]:
                target = min(previous, cap_level)
                reasons.append("reference_wait_band")
            else:
                step = policy["position_step"]
                threshold = step / 2 + policy["hysteresis"]
                if abs(continuous - previous) + 1e-12 >= threshold:
                    target = min(cap_level, max(0.0, math.floor(continuous / step + .5) * step))
                target = min(target, cap_level)
            target = round(target, 10)
            if predictive_passed:
                self.last_scored_date = day
                if finite(stop_price) is not None and target > 0:
                    self.stop_price = max(float(stop_price), self.stop_price or float(stop_price))
            status = "scored" if predictive_passed else "pending"
        if target == 0:
            self.stop_price = None
        if usable and target < previous and abs(target - cap_level) < 1e-10:
            reasons.append("reference_risk_cap_tightened")
        self.target = target
        delta = round(target - previous, 10)
        action = "BUY" if delta > 1e-10 else "SELL" if delta < -1e-10 else "WAIT"
        if action == "BUY" and not predictive_passed:
            raise AssertionError("unqualified reference cannot increase exposure")
        self.steps += 1
        return {
            "schema": REFERENCE_FAMILY, "analysis_mode": "reference", "reference_only": True,
            "account_scope": "none", "date": day, "available_at": day + "T15:00:00+08:00",
            "status": "increase_exposure" if action == "BUY" else "reduce_exposure" if action == "SELL" else status,
            "action": action, "target_position_fraction": target,
            "previous_reference_target": previous, "change_fraction": delta,
            "reference_delta": delta, "continuous_target_fraction": continuous,
            "risk_position_cap_fraction": cap, "exposure_cap_fraction": cap,
            "risk_budget_fraction": policy["risk_budget"], "stop_price": finite(stop_price),
            "invalidation_price": prior_stop if hard_invalidation else self.stop_price,
            "previous_frozen_stop_price": prior_stop, "probability_edge": edge,
            "allocation_score": score, "stale": not usable or not predictive_passed,
            "score_status": status, "last_scored_date": self.last_scored_date,
            "initialization": "zero_reference_target" if self.steps == 1 else None,
            "change_basis": "previous_reference_target_for_same_symbol",
            "current_position_fraction": current,
            "current_position_basis": "manual_comparison_input" if current is not None else None,
            "current_position_is_historical_inventory": False,
            "manual_comparison_change_fraction": target - current if current is not None else None,
            "predictive_gate_passed": bool(predictive_passed),
            "reason_codes": list(dict.fromkeys(reasons)),
            "basis": "逐股参考路径；变化相对上一参考目标。风险上限约束目标，不代表账户资金分配。",
        }


def risk_geometry(frame, policy=None):
    """Strictly prior ATR/low risk geometry in CPU float64."""
    cfg = validate_position_policy(policy)
    close = pd.to_numeric(frame.close, errors="coerce").astype(float)
    high = pd.to_numeric(frame.high, errors="coerce").astype(float)
    low = pd.to_numeric(frame.low, errors="coerce").astype(float)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.shift().rolling(14, min_periods=14).mean()
    boundary = pd.concat([low.shift().rolling(20, min_periods=20).min(),
                          close - cfg["atr_stop_multiple"] * atr], axis=1).max(axis=1, skipna=False)
    distance = (close - boundary) / close
    eligible = ((close > 0) & (boundary > 0) & (atr > 0)
        & distance.between(cfg["min_stop_distance"], cfg["max_stop_distance"]))
    cap = (cfg["risk_budget"] / distance).clip(upper=cfg["max_stock_weight"]).where(eligible)
    return pd.DataFrame({"atr": atr, "stop": boundary.where(eligible), "distance": distance.where(eligible),
                         "cap": cap, "eligible": eligible}, index=frame.index)


def _source_reference_events(pack, symbol, cutoff):
    """Keep original source identities only at their verified visible clocks."""
    from my_strategy.services.czsc_trend_consensus import observable
    result = []
    for event in pack.get("events", []):
        day = str(event.get("first_available_at") or event.get("available_at") or "")[:10]
        at = day + "T15:00:00+08:00"
        clocks = [event.get(key) for key in ("available_at", "first_available_at", "first_confirmed_at",
            "origin_observed_at", "observed_at") if event.get(key) is not None]
        if (event.get("symbol") != symbol or event.get("source") not in {"structure", "wyckoff"}
            or not event.get("event_id") or not event.get("original_event_id") or not event.get("pattern_id")
            or event.get("confirmed") is not True or event.get("input_eligible") is not True
            or event.get("candidate_direction_valid", event.get("candidate_eligible")) is not True
            or event.get("evidence_visible", True) is not True or event.get("side") not in {"BUY", "SELL"}
            or not event.get("date") or str(event["date"])[:10] > day or day > cutoff
            or not event.get("available_at") or not clocks or not all(observable(clock, at) for clock in clocks)):
            continue
        result.append(copy.deepcopy(event))
    return result


@lru_cache(maxsize=12)
def _model_session(path, mtime_ns, device):
    from my_strategy.services.reference_model import load_model
    return load_model(path, device=device)


def _gate_passed(value):
    return value is True or isinstance(value, dict) and value.get("passed") is True


def reference_overlay(features, calendar, *, model_run_id=None, model_fold=None,
                      model_mode="historical", as_of=None, device="auto", source_context_hash=None,
                      consensus_config=None, use_models=True, model_selection_frozen=False,
                      current_position_fraction=None, prepared_native=None):
    """Batch numeric inference by effective model, followed by a causal path."""
    from my_strategy.adapters.czsc_adapter import date_cutoff
    from my_strategy.core.device import get_default_device
    from my_strategy.services.czsc_native_evidence import build_native_evidence
    from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES
    from my_strategy.services.czsc_trend_consensus import observable
    from my_strategy.services.reference_model import prepare_inputs
    from my_strategy.services.reference_training import active_reference_run_id, load_reference_run, model_for_date
    if model_mode not in {"historical", "production"}:
        raise ValueError("参考分析仅支持历史研究重建或生效后的当前参考")
    frame = features.copy()
    if frame.empty:
        raise ValueError("参考分析需要单股按日有序唯一行情")
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    cutoff_at = date_cutoff(as_of or frame.iloc[-1].date)
    sessions = [str(day)[:10] for day in calendar if pd.Timestamp(day) + pd.Timedelta(hours=15) <= cutoff_at]
    if not sessions:
        raise ValueError("请求时点之前没有已收盘的核验交易日")
    cutoff = sessions[-1]
    frame = frame[frame.date <= cutoff].reset_index(drop=True)
    if frame.empty or frame.symbol.nunique() != 1 or list(frame.date) != sorted(set(frame.date)):
        raise ValueError("参考分析需要单股按日有序唯一行情")
    selected_run = model_run_id
    if use_models and selected_run is None and not model_selection_frozen:
        selected_run = active_reference_run_id()
    root, report = load_reference_run(selected_run) if use_models and selected_run else (None, None)
    cfg = validate_position_policy({**(report or {}).get("config", {}).get("position", {}),
                                    **(consensus_config or {})})
    if prepared_native is None:
        pack = build_native_evidence(frame, market_dates=calendar)
    else:
        pack = prepared_native
        packed = pack["frame"]
        if list(pd.to_datetime(packed.date).dt.strftime("%Y-%m-%d")) != list(frame.date):
            raise ValueError("预备原生证据与请求时点不一致")
    inputs = prepare_inputs(pack["frame"], calendar)
    frame = pack["frame"].copy().reset_index(drop=True)
    for column in inputs:
        frame[column] = inputs[column].to_numpy(copy=False)
    frame.attrs.update(inputs.attrs)
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    probs = np.full((len(frame), 3), np.nan, dtype=np.float32)
    resolutions = [None] * len(frame)
    groups = defaultdict(list)
    for index, day in enumerate(frame.date):
        selected = model_for_date(report, day, mode=model_mode) if report and use_models else None
        if model_fold and selected and selected.get("name") != model_fold:
            selected = None
        if selected:
            resolutions[index] = selected
            if bool(frame.iloc[index].get("input_eligible", False)):
                groups[str(selected["model_path"])].append(index)
    actual_device = get_default_device(device or "auto")
    inferred = 0
    for path, indices in groups.items():
        manifest = Path(path)
        if manifest.is_dir():
            manifest = manifest / "manifest.json"
        session = _model_session(str(manifest), manifest.stat().st_mtime_ns, actual_device)
        prediction = np.asarray(session.predict(frame.iloc[indices]), dtype=np.float32)
        if prediction.shape != (len(indices), 3):
            raise ValueError("参考模型三分类输出维度错误")
        probs[indices] = prediction
        inferred += len(indices)
    risk_values = risk_geometry(frame, cfg)
    native_days = {str(item["date"])[:10]: item for item in pack.get("evaluations", [])}
    source_by_day = defaultdict(list)
    for event in _source_reference_events(pack, str(frame.iloc[0].symbol), cutoff):
        day = str(event.get("first_available_at") or event["available_at"])[:10]
        source_by_day[day].append(event)
    path = ReferencePositionPath(cfg)
    rows, points, source_points = [], [], []
    counts = Counter()
    symbol = str(frame.iloc[0].symbol)
    for index, bar in enumerate(frame.to_dict("records")):
        day, selected = str(bar["date"])[:10], resolutions[index]
        quote, risk = finite(bar.get("close")), risk_values.iloc[index]
        eligible = bool(bar.get("input_eligible", False))
        scored = bool(selected and eligible and np.isfinite(probs[index]).all() and risk.eligible)
        gate = _gate_passed((selected or {}).get("predictive_gate"))
        up, down, neutral = [finite(value) for value in probs[index]]
        reasons = []
        if selected is None:
            reasons.append("no_effective_reference_model")
        if not eligible:
            reasons.append("reference_feature_input_ineligible")
        if not risk.eligible:
            reasons.append("reference_risk_geometry_unavailable")
        quote_reliable = (quote is not None and bar.get("source") in VERIFIED_SOURCES
            and bar.get("has_trade_price") == 1 and not bool(bar.get("wyckoff_corporate_action_suspected", False))
            and observable(bar.get("available_at"), day + "T15:00:00+08:00"))
        invalid = quote_reliable and path.stop_price is not None and quote <= path.stop_price
        position = path.advance(day=day, p_up=up, p_down=down, risk_cap=risk.cap,
            stop_price=risk.stop, scored=scored, predictive_passed=gate,
            hard_invalidation=invalid, current_position_fraction=current_position_fraction,
            reason_codes=reasons)
        risk_plan = {"status": "available" if risk.eligible else "unavailable", "date": day,
            "reference_price": quote, "reference_price_date": day, "reference_price_source": "signal_day_close",
            "stop_price": finite(risk.stop), "invalidation_price": finite(risk.stop),
            "stop_distance_fraction": finite(risk.distance), "atr14": finite(risk.atr),
            "risk_position_cap_fraction": finite(risk.cap), "max_position_fraction": cfg["max_stock_weight"],
            "risk_budget_fraction": cfg["risk_budget"], "source_event_ids": [],
            "reference_price_band": {"low": None, "high": None, "reference_only": True},
            "reason_codes": [] if risk.eligible else ["reference_risk_geometry_unavailable"],
            "basis": "严格先验ATR与20日低价形成的风险上限；不是积极目标。"}
        position.update(reference_price=quote, reference_price_date=day,
                        reference_price_band=copy.deepcopy(risk_plan["reference_price_band"]))
        action = position["action"]
        source_events = source_by_day.get(day, [])
        for event in source_events:
            event.update(reference_layer="source", reference_only=True, analysis_mode="reference",
                account_scope="none", date=day, signal_date=day, direction_eligible=True,
                reference_price=quote, reference_price_date=day, action=event["side"],
                source_label={"structure": "结构", "wyckoff": "量价"}.get(event["source"], event["source"]))
        source_points.extend(source_events)
        native = native_days.get(day, {})
        agents = {"ma": {"label": "联合方向模型", "status": "scored" if scored else "model_unavailable",
            "p_up": up, "p_down": down, "p_neutral": neutral, "in_scope": scored,
            "buy_contribution": None, "sell_contribution": None,
            "model_identity": copy.deepcopy(selected or {}), "reason_codes": reasons}}
        for source in ("structure", "wyckoff"):
            evidence = native.get(source, {})
            state = evidence.get("current_state", {})
            agents[source] = {"status": "scored" if evidence.get("input_eligible") else "input_ineligible",
                "in_scope": bool(evidence.get("input_eligible")), "buy_contribution": None, "sell_contribution": None,
                "buy_strength": state.get("buy_score", state.get("effective_demand")),
                "sell_strength": state.get("sell_score", state.get("effective_supply")),
                "current_demand": state.get("buy_score"), "current_supply": state.get("sell_score"),
                "event_demand": state.get("event_demand_support"), "event_supply": state.get("event_supply_support"),
                "reason_codes": evidence.get("reason_codes", [])}
        recommendation = {"action": action, "reference_only": True, "analysis_mode": "reference",
            "date": day, "signal_date": day, "available_at": day + "T15:00:00+08:00",
            "reference_price": quote, "reference_close": quote, "reference_price_date": day,
            "reference_price_source": "signal_day_close", "position_reference": copy.deepcopy(position),
            "reference_risk_plan": copy.deepcopy(risk_plan), "basis": position["basis"],
            "execution_basis": "参考目标变化，不模拟成交或记录实际持仓。"}
        row = {"symbol": symbol, "date": day, "available_at": day + "T15:00:00+08:00",
            "analysis_mode": "reference", "reference_only": True, "account_scope": "none",
            "action": action, "direction": action if action != "WAIT" or not position["stale"] else "UNAVAILABLE",
            "reference_action": action, "emitted": action != "WAIT", "reference_price": quote,
            "reference_close": quote, "reference_price_date": day, "reference_price_source": "signal_day_close",
            "position_reference": position, "reference_risk_plan": risk_plan,
            "reference_recommendation": recommendation, "source_reference_points": source_events,
            "events": copy.deepcopy(source_events), "agents": agents, "buy_base_score": up,
            "sell_base_score": down, "high_return_probability": None, "exit_probability": None,
            "reason_codes": position["reason_codes"], "model_resolution": {**(selected or {}),
                "model_family": REFERENCE_FAMILY, "schema_family": REFERENCE_FAMILY, "display_name": DISPLAY_NAME,
                "direction_probabilities": {"p_up": up, "p_down": down, "p_neutral": neutral},
                "publication_allowed": False, "probability": None},
            "historical_reconstruction": bool((selected or {}).get("research_reconstruction"))}
        if action != "WAIT":
            point = {"event_id": stable_hash({"version": REFERENCE_FAMILY, "symbol": symbol,
                "date": day, "target": position["target_position_fraction"], "run": str(root)}),
                "date": day, "signal_date": day, "available_at": row["available_at"],
                "first_available_at": row["available_at"], "symbol": symbol, "side": action,
                "action": action, "source": "reference_consensus", "reference_layer": "consensus",
                "event_type": "reference_target_change", "reference_only": True, "analysis_mode": "reference",
                "reference_price": quote, "reference_price_date": day,
                "target_position_fraction": position["target_position_fraction"],
                "previous_reference_target": position["previous_reference_target"],
                "change_fraction": position["change_fraction"], "historical_reconstruction": row["historical_reconstruction"],
                "reason_codes": position["reason_codes"]}
            points.append(point)
            row["event_id"] = point["event_id"]
        counts[position["score_status"]] += 1
        rows.append(row)
    latest = copy.deepcopy(rows[-1])
    if latest["date"] != cutoff:
        position = path.advance(day=cutoff, scored=False, predictive_passed=False,
            current_position_fraction=current_position_fraction, reason_codes=["missing_target_market_session"])
        latest.update(date=cutoff, available_at=cutoff + "T15:00:00+08:00", action="WAIT", direction="UNAVAILABLE",
            reference_action="WAIT", emitted=False, reference_price=None, reference_close=None,
            reference_price_date=cutoff, position_reference=position, events=[], source_reference_points=[],
            reason_codes=position["reason_codes"], model_resolution={})
        latest["reference_recommendation"] = {"action": "WAIT", "reference_only": True,
            "analysis_mode": "reference", "reference_price": None, "position_reference": copy.deepcopy(position),
            "basis": "请求日缺少行情；上一参考目标仅作为陈旧记录保留。"}
        latest["reference_risk_plan"] = {"status": "unavailable", "reference_price": None,
            "reference_price_date": cutoff, "risk_position_cap_fraction": None,
            "reason_codes": ["missing_target_market_session"]}
    latest.update(model_family=REFERENCE_FAMILY, schema_family=REFERENCE_FAMILY, display_name=DISPLAY_NAME,
        model_policy="pinned" if model_fold else "auto" if use_models else "disabled", model_fold=model_fold)
    prepared = {"version": REFERENCE_FAMILY, "config": cfg, "config_hash": stable_hash(cfg),
        "rows": copy.deepcopy(rows), "events": points, "direction_points": points,
        "mode": "reference", "reference_only": True, "analysis_mode": "reference",
        "candidate_policy": "causal_per_symbol_target_changes"}
    return {"strategy_version": REFERENCE_FAMILY, "display_name": DISPLAY_NAME,
        "analysis_mode": "reference", "reference_only": True, "account_scope": "none",
        "models_enabled": bool(use_models), "structure_only": not use_models,
        "rows": rows, "trend_consensus": prepared, "reference_points": points,
        "source_reference_points": source_points, "reference_risk_plan": copy.deepcopy(latest["reference_risk_plan"]),
        "position_reference": copy.deepcopy(latest["position_reference"]),
        "reference_recommendation": copy.deepcopy(latest["reference_recommendation"]),
        "account_actions": None, "latest_decision": latest, "features": frame, "native_pack": pack,
        "wyckoff_events": pack.get("wyckoff_events", []), "usage_mode": model_mode,
        "model_run_id": root.name if root else None, "model_gate": (report or {}).get("predictive_gate", {"passed": False}),
        "publication_allowed": False, "source_context_hash": source_context_hash,
        "pricing_assumptions": {"reference_price": "last_observed_signal_day_close", "reference_is_fill": False,
            "scope": "analysis_reference", "account_simulated": False},
        "compute_info": {"model_family": REFERENCE_FAMILY, "display_name": DISPLAY_NAME,
            "selected_device": actual_device, "actual_device": actual_device if inferred else "cpu",
            "model_inference_device": actual_device if inferred else None, "structure_device": "cpu_float64",
            "research": model_mode == "historical", "account_simulated": False,
            "actual_cuda_inference": bool(inferred and actual_device.startswith("cuda")),
            "actual_mps_inference": bool(inferred and actual_device == "mps"),
            "model_inference_rows": inferred, "model_inference_batches": len(groups),
            "gpu_owner": "request_parent_process", "source_reason_counts": dict(counts)}}
