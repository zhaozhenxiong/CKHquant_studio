"""Causal trend inference and one actual research Broker timeline for the UI."""
from __future__ import annotations

import copy
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_v2_contracts import (STRATEGY_ID, DISPLAY_NAME,
    TRAINING_POPULATION, ATTEMPT_IDENTITY, ECONOMIC_GROUP_IDENTITY, strategy_config)

CURRENT_MODEL_PATH = Path(os.environ.get("KHQUANT_CURRENT_STRATEGY_CONFIG",
    str(MY_STRATEGY_ROOT / "configs/current_strategy.json")))

PRICING_ASSUMPTIONS = {
    "broker_fill": "continuous_cost_model_simulated_average_after_slippage",
    "initial_R": "frozen_stop_reference_half_up_cent_then_sell_fees",
    "initial_R_is_actual_stop_loss": False,
    "scope": "research_simulation",
}


def load_model_run(model_run_id=None):
    from my_strategy.services.czsc_trend_v2_pipeline import read_run
    if model_run_id is None and not ARTIFACT_RUNS_ROOT.is_dir():
        return None, None
    if model_run_id is None and CURRENT_MODEL_PATH.is_file():
        current = json.loads(CURRENT_MODEL_PATH.read_text(encoding="utf-8"))
        if current.get("strategy") == STRATEGY_ID and current.get("model_run_id"):
            # A delivered application uses its deliberately selected run.
            # Within that run, each forecast still needs a legitimately prior
            # checkpoint. This is separate from automatic run qualification.
            model_run_id = current["model_run_id"]
    return read_run(model_run_id)


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
    from my_strategy.services.czsc_trend_v2_model import TrendSession, bind_trend_inputs
    from my_strategy.services.czsc_trend_v2_interval_model import HeadSession
    from my_strategy.services.czsc_trend_v2_pipeline import forward_trend, predict_entry, pinned_exit_policy
    from my_strategy.services.czsc_trend_v2_features import event_prefix_rows, original_evidence_events
    from my_strategy.services.czsc_trend_v2_consensus import prepare_trend_decisions
    from my_strategy.services.czsc_trend_v2_account import replay_trend_account
    from my_strategy.services.czsc_native_evidence_v4 import build_native_evidence_v4
    from my_strategy.services.czsc_research_ml import _device

    if model_mode not in {"historical", "production"}:
        raise ValueError("趋势共识策略只支持按时点判断")
    actual_device = str(_device(device or "cpu"))
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    cutoff = str(as_of or frame.iloc[-1].date)[:10]
    frame = frame[frame.date <= cutoff].reset_index(drop=True)
    if frame.empty or frame.symbol.nunique() != 1 or list(frame.date) != sorted(set(frame.date)):
        raise ValueError("趋势共识输入需要单股按日有序唯一行情")
    root, report = load_model_run(model_run_id)
    cfg = strategy_config(report["config"]) if report else strategy_config()
    label_target = {"min_net_return": float(cfg["high_return"]["min_net_return"]),
                    "min_r_multiple": float(cfg["high_return"]["min_r_multiple"])}
    account_start = str(account_start or cfg["account"].get("start", cfg["feature_start"]))[:10]
    if consensus_config is not None:
        from my_strategy.services.czsc_trend_v2_consensus import consensus_config as validate
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
    pack = build_native_evidence_v4(frame, cfg, market_dates=calendar)
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
            model_resolution={**quality.get("model_identity", {}), "model_family": "high_return",
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
    latest.update(model_family="high_return", schema_family=STRATEGY_ID, display_name=DISPLAY_NAME,
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
        "compute_info": {"model_family": "high_return", "display_name": DISPLAY_NAME,
            "selected_device": actual_device, "actual_device": "cpu", "model_inference_device": "cpu",
            "research": True, "actual_cuda_inference": False, "actual_mps_inference": False,
            "model_inference_rows": int(trend_stats.get("rows", 0)) + int(interval_stats.get("rows", 0)),
            "model_inference_batches": int(trend_stats.get("batches", 0)) + int(interval_stats.get("batches", 0)),
            "model_loads": int(trend_stats.get("model_loads", 0)) + int(interval_stats.get("model_loads", 0)),
            "trend_inference": trend_stats, "interval_inference": interval_stats,
            "source_reason_counts": dict(counts)}}
