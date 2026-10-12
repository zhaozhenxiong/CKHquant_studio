"""Causal, display-only candidate assessments for chart arbitration v3.

The frozen signal/episode/model pipeline remains untouched. The optional read-only
recovery repeats its original L1 preparation to expose complete daily routes; it
never changes candidate policies, stops, training data or Broker decisions.
"""
from __future__ import annotations

import copy
import math

import pandas as pd

from my_strategy.core.run_context import stable_hash

EVIDENCE_VERSION = "czsc_candidate_evidence_v1"
V3_VERSION = "czsc_agent_consensus_v3"
RISK_REASONS = {"risk_inputs_unavailable", "invalid_causal_stop", "stop_distance_exceeds_frozen_maximum"}
MA_STATE_COLUMNS = ("close_above_ma20", "ma20_slope5", "ma_alignment", "ret_5", "ret_1", "ma5_ma10_gap")


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _flag(value):
    return value is True or isinstance(value, (int, float)) and value == 1


def _visible(value, day):
    if value is None or str(value) in {"", "None", "nan", "NaT"}:
        return False
    try:
        instant = pd.Timestamp(value)
        if instant.tzinfo is None:
            instant = instant.tz_localize("Asia/Shanghai")
        return instant <= pd.Timestamp(day + "T15:00:00+08:00")
    except (ValueError, TypeError):
        return False


def _records(value):
    return value.to_dict("records") if isinstance(value, pd.DataFrame) else list(value or [])


def _policy_binding(config):
    from my_strategy.services.czsc_signal_points import signal_point_parameters
    from my_strategy.services.czsc_trade_episodes import _settings
    parameters = signal_point_parameters(config)
    return {"candidate_policy_version": "independent_points_union_v1",
            "candidate_policy": "independent_points_union_v1",
            "signal_points_config_hash": stable_hash(parameters),
            "risk_plan_policy_hash": stable_hash({"parameters": parameters, "strategy": _settings(config)}),
            "frozen_configuration_hash": stable_hash(config)}


def _verify_manifest(manifest, config, *, head):
    """Bind the verified saved contract to the exact selected frozen policy.

    Older immutable manifests bind signal and risk parameters through the full
    configuration hash rather than separate keys. Derive the explicit display
    binding only after verifying that complete hash, never from a bare target ID.
    """
    from my_strategy.services.czsc_high_return_runtime import _validate_compatibility
    from my_strategy.services.czsc_high_return_models import _columns_for, schema_hash
    from my_strategy.services.czsc_signal_points import MA_ENTRY_TARGET
    contract = manifest.get("target_contract") or {}
    binding = _policy_binding(config)
    expected_policy = ("broad_causal_ma_rule_no_entry_head"
                       if head == "ma_entry" and config.get("ma_entry_alignment") == "signal_pinned_learned_exit_v2"
                       else "independent_points_union_v1")
    expected_target = MA_ENTRY_TARGET if head == "ma_entry" else "actual_complete_fee_net_roundtrip_high_return_and_initial_risk"
    if (manifest.get("head") != head or not manifest.get("manifest_sha256")
            or stable_hash(contract) != manifest.get("target_contract_hash")
            or contract.get("strategy_hash") != stable_hash(config)
            or contract.get("candidate_policy") != expected_policy
            or contract.get("target") != expected_target
            or tuple(manifest.get("columns") or []) != tuple(_columns_for(head))
            or manifest.get("feature_schema_hash") != schema_hash(_columns_for(head), head)
            or contract.get("model_input_schema_hash", contract.get("feature_schema_hash")) != manifest.get("feature_schema_hash")):
        raise ValueError("v3 chart model candidate/risk/target/schema binding mismatch")
    _validate_compatibility(contract, contract_config=config, source="ma" if head == "ma_entry" else "union",
                            initial_cash=config.get("initial_cash", 100000))
    # Optional explicit binding on newer models must agree exactly too.
    for key in ("signal_points_config_hash", "risk_plan_policy_hash"):
        if key in contract and contract[key] != binding[key]:
            raise ValueError("v3 chart model " + key + " mismatch")
    return {**binding, "model_candidate_policy": contract["candidate_policy"],
            "target_contract_hash": manifest["target_contract_hash"],
            "feature_schema_hash": manifest["feature_schema_hash"], "verified_frozen_contract": True}


def recover_daily_evidence(overlay, features, calendar, *, usage_mode="historical"):
    """Expose original daily MA routes and explicit model-policy verification.

    The runtime intentionally drops these routes. This opt-in adapter reruns the
    unchanged prepare_points function with the same selected run/checkpoints and
    reports that extra inference honestly. Existing overlay facts must match.
    """
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    from my_strategy.services.czsc_high_return_runtime import _config, _read_run, _head_path, prepare_points
    from my_strategy.services.czsc_high_return_models import load_head
    output = copy.deepcopy(overlay)
    rows = _records(output.get("rows"))
    if not rows:
        return output
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    frame = frame[frame.date <= rows[-1]["date"]].reset_index(drop=True)
    run_ids = {row.get("model_resolution", {}).get("model_run_id") for row in rows}
    if len(run_ids) != 1:
        raise ValueError("v3 chart evidence requires one exact selected model run")
    run_id = next(iter(run_ids))
    root, bundles, config = ARTIFACT_RUNS_ROOT, [], _config()
    if run_id:
        root, report, _ = _read_run(run_id)
        config = _config(report.get("config"))
        bundles = report.get("model_bundles", [])
        selected_checkpoints = {row.get("model_resolution", {}).get("checkpoint") for row in rows}
        # A fixed fold has only one routed checkpoint; infer its exact restriction
        # from every row, including rows before the checkpoint becomes available.
        if output.get("chart_model_fold"):
            bundles = [bundle for bundle in bundles if bundle["name"] == output["chart_model_fold"]]
        if not selected_checkpoints - {None} <= {bundle["name"] for bundle in bundles}:
            raise ValueError("v3 chart evidence model route not in selected frozen report")
    points = prepare_points(frame, calendar, model_root=root, bundles=bundles, config=config, usage_mode=usage_mode)
    if (stable_hash(points["events"]) != stable_hash(output.get("signal_points", []))
            or stable_hash(points["candidates"]) != stable_hash(output.get("candidates", []))):
        raise ValueError("v3 recovered L1 evidence differs from the original frozen runtime facts")
    manifests = {}
    for route in points["ma_routes"]:
        identity = route.get("lineage") or {}
        checkpoint = identity.get("checkpoint")
        if route.get("mode") != "forward_model" or not checkpoint:
            continue
        key = (checkpoint, "ma_entry")
        if key not in manifests:
            bundle = next(bundle for bundle in bundles if bundle["name"] == checkpoint)
            manifest = load_head(_head_path(root, bundle, "ma_entry"))
            manifests[key] = (manifest, _verify_manifest(manifest, config, head="ma_entry"))
        manifest, binding = manifests[key]
        if (identity.get("target_contract_hash") != manifest["target_contract_hash"]
                or identity.get("manifest_sha256") != manifest["manifest_sha256"]
                or identity.get("available_at") != manifest["available_at"]
                or identity.get("train_label_end") != manifest["train_label_end"]):
            raise ValueError("v3 daily MA route lineage differs from verified saved head")
        identity.update(binding, score_date=route["date"], model_head="ma_entry")
    binding = _policy_binding(config)
    for row in rows:
        identity = row.get("model_resolution") or {}
        if row.get("high_return_probability") is None:
            continue
        checkpoint = identity.get("checkpoint")
        key = (checkpoint, "entry")
        if key not in manifests:
            bundle = next(bundle for bundle in bundles if bundle["name"] == checkpoint)
            manifest = load_head(_head_path(root, bundle, "entry"))
            manifests[key] = (manifest, _verify_manifest(manifest, config, head="entry"))
        manifest, model_binding = manifests[key]
        if identity.get("available_at") != manifest["available_at"]:
            raise ValueError("v3 high-return route availability differs from saved head")
        identity.update(model_binding, train_label_end=manifest["train_label_end"], model_head="entry",
                        manifest_sha256=manifest["manifest_sha256"], score_date=row["date"])
    output["rows"] = rows
    output["ma_routes"] = points["ma_routes"]
    output["candidate_evidence_policy"] = {**binding, "parameters": points["config"],
        "evidence_version": EVIDENCE_VERSION, "model_run_id": run_id,
        "policy_scope": "display_only_original_candidate_and_risk_contract"}
    compute = dict(output.get("compute_info") or {})
    extra = dict(points.get("compute_info") or {})
    compute["chart_evidence_recovery"] = {"method": "repeat_frozen_prepare_points_read_only", **extra}
    for displayed, counted in (("model_inference_rows", "rows"), ("model_inference_batches", "batches"), ("model_loads", "model_loads")):
        compute[displayed] = int(compute.get(displayed) or 0) + int(extra.get(counted) or 0)
    output["compute_info"] = compute
    return output


def _direction_fact(event, day):
    source, side = event.get("source"), event.get("side")
    reasons = event.get("reason_codes") or []
    # Risk disqualifies an entry, not its already observed directional evidence.
    # MA selection errors remain independently checked against today's route.
    directional_blockers = [code for code in reasons if code not in RISK_REASONS and not str(code).startswith("ma_")]
    return (source in {"ma", "structure", "wyckoff"} and side in {"BUY", "SELL"}
            and event.get("evidence_visible") is True and event.get("input_eligible") is True
            and _visible(event.get("available_at"), day)
            and all(_visible(event[key], day) for key in ("observed_at", "signal_at") if event.get(key))
            and not directional_blockers)


def build_candidate_evidence(rows, features, *, policy=None, config=None):
    """Assess all agents on each observed bar; a new fact still triggers decisions.

    Source memories contain only original causal event IDs. Their current validity
    is recomputed from today's observed input; the arbiter independently enforces
    the same age bound and final-entry risk eligibility.
    """
    policy, config = policy or {}, config or {}
    max_age = int(config.get("evidence_valid_bars", 3))
    min_w = float(config.get("wyckoff_min_strength", .5))
    threshold = float(config.get("ma_entry_threshold", .55))
    feature_rows = {(str(row.get("symbol")), str(row.get("date"))[:10]): row for row in _records(features)}
    original_features, memory, output = {}, [], {}
    for position, row in enumerate(rows):
        day, symbol = row["date"], row["symbol"]
        feature = feature_rows.get((symbol, day), {})
        close = _finite(row.get("reference_close"))
        facts = []
        for event in row.get("events", []):
            fact = copy.deepcopy(event)
            fact["direction_eligible"] = bool(_direction_fact(fact, day))
            facts.append(fact)
            if fact["direction_eligible"]:
                original_features[fact["event_id"]] = copy.deepcopy(feature)
                memory.append((position, fact))
        memory = [(index, event) for index, event in memory if position - index < max_age]
        active = {source: [event for _, event in memory if event.get("source") == source]
                  for source in ("ma", "structure", "wyckoff")}
        evaluations, invalidated_ids = {}, set()
        for source in active:
            evaluation = copy.deepcopy(row.get("agents", {}).get(source) or {})
            state = {"date": day, "available_at": day + "T15:00:00+08:00", "source_id": None,
                     "input_eligible": False, "valid": False, "buy_valid": False, "sell_valid": False,
                     "side": "ABSTAIN", "origin_event_ids": [], "dependencies": [], "reason_codes": [],
                     "invalidated_event_ids": [], "current_state": {"close": close}}
            if source == "ma":
                values = {key: _finite(feature.get(key)) for key in MA_STATE_COLUMNS}
                input_ok = (_flag(feature.get("model_input_eligible")) and close is not None
                            and all(value is not None for value in values.values())
                            and (not feature.get("available_at") or _visible(feature["available_at"], day)))
                ma_buy = bool(input_ok and values["close_above_ma20"] > 0 and values["ma20_slope5"] > 0
                              and (values["ma_alignment"] > 0 or values["ret_5"] > 0))
                ma_sell = bool(input_ok and (values["close_above_ma20"] <= 0 and values["ma20_slope5"] < 0
                                            or values["ma5_ma10_gap"] < 0 and values["ret_5"] < 0))
                identity = evaluation.get("model_identity") or {}
                p = _finite(evaluation.get("probability"))
                full_identity = all(identity.get(key) for key in ("model_run_id", "checkpoint", "available_at", "train_label_end", "target_contract_hash", "manifest_sha256"))
                visible = (full_identity and _visible(identity.get("available_at"), day)
                           and str(identity["train_label_end"])[:10] < day and identity.get("score_date") == day
                           and identity.get("model_head") == "ma_entry" and identity.get("verified_frozen_contract") is True)
                buy = bool(ma_buy and visible and evaluation.get("status") == "scored" and p is not None and p >= threshold)
                sells = [event for event in active[source] if event.get("side") == "SELL" and event.get("event_type") == "ma_trend_failure"]
                sell = bool(ma_sell and sells)
                invalidated = [event["event_id"] for event in active[source] if input_ok and
                               (event["side"] == "SELL" and ma_buy and not ma_sell
                                or event["side"] == "BUY" and ma_sell and not ma_buy)]
                state.update(input_eligible=input_ok, buy_valid=buy, sell_valid=sell,
                    invalidated_event_ids=invalidated,
                    origin_event_ids=[event["event_id"] for event in sells] if sell else [event["event_id"] for event in active[source] if event.get("side") == "BUY"] if buy else [],
                    dependencies=["trailing_sma_5_10_20_60", "verified_daily_ma_entry_head" if buy else "causal_ma_trend_failure"])
                state["current_state"].update(values, ma_buy=ma_buy, ma_sell=ma_sell, score_date=day)
                evaluation["score_date"] = day
                if not input_ok:
                    state["reason_codes"].append("ma_current_input_unavailable")
                elif ma_buy and not visible:
                    state["reason_codes"].append("ma_current_complete_lineage_unavailable")
                elif ma_buy and (p is None or p < threshold):
                    state["reason_codes"].append("ma_current_entry_probability_veto")
            elif source == "structure":
                current_confirmation = feature.get("structure_confirmed_at")
                input_ok = (_flag(feature.get("structure_input_eligible", feature.get("input_eligible")))
                            and close is not None and _visible(current_confirmation, day))
                today_sides = {event["side"] for event in facts if event.get("source") == source and event["direction_eligible"]}
                valid, floors, ceilings = [], [], []
                for event in active[source]:
                    origin = original_features[event["event_id"]]
                    boundary = _finite(origin.get("structure_low" if event["side"] == "BUY" else "structure_high"))
                    # Fresh opposite facts preserve same-day conflict. A newly
                    # confirmed opposite direction permanently invalidates only
                    # older origins; it must not hide an assisted conflict.
                    opposite_today = (("SELL" if event["side"] == "BUY" else "BUY") in today_sides
                                      and str(event.get("evidence_date", event.get("date")))[:10] != day)
                    bound_ok = boundary is not None and boundary > 0 and (close > boundary if event["side"] == "BUY" else close < boundary) if close is not None else False
                    if input_ok and boundary is not None and (not bound_ok or opposite_today):
                        state["invalidated_event_ids"].append(event["event_id"])
                    if input_ok and bound_ok and not opposite_today:
                        valid.append(event)
                        (floors if event["side"] == "BUY" else ceilings).append(boundary)
                state.update(input_eligible=input_ok, buy_valid=bool(floors), sell_valid=bool(ceilings),
                    origin_event_ids=[event["event_id"] for event in valid],
                    dependencies=list(dict.fromkeys(dependency for event in valid for dependency in event.get("dependencies", []))))
                state["current_state"].update(structure_confirmed_at=current_confirmation,
                    support_valid=bool(valid), invalidated=not bool(valid),
                    support_floor=max(floors) if floors else None, resistance_ceiling=min(ceilings) if ceilings else None)
                if not input_ok:
                    state["reason_codes"].append("structure_current_input_or_confirmation_unavailable")
                elif active[source] and not valid:
                    state["reason_codes"].append("structure_current_support_invalidated")
            else:
                demand, supply = _finite(feature.get("w_demand_score")), _finite(feature.get("w_supply_score"))
                visible = all(_visible(feature.get(key), day) for key in ("wyckoff_available_at", "wyckoff_observed_at", "wyckoff_range_formed_at"))
                input_ok = _flag(feature.get("wyckoff_input_eligible")) and close is not None and visible
                valid = []
                for event in active[source]:
                    origin = original_features[event["event_id"]]
                    boundary = _finite(origin.get("wyckoff_range_low" if event["side"] == "BUY" else "wyckoff_range_high"))
                    bound_ok = boundary is not None and (close > boundary if event["side"] == "BUY" else close < boundary) if close is not None else False
                    strength = (demand is not None and demand >= min_w and demand > (supply or 0)) if event["side"] == "BUY" else (supply is not None and supply >= min_w and supply > (demand or 0))
                    opposite = (supply is not None and demand is not None and supply >= min_w and supply > demand) if event["side"] == "BUY" else (demand is not None and supply is not None and demand >= min_w and demand > supply)
                    if input_ok and (boundary is not None and not bound_ok or opposite):
                        state["invalidated_event_ids"].append(event["event_id"])
                    if input_ok and bound_ok and strength:
                        valid.append(event)
                state.update(input_eligible=input_ok,
                    buy_valid=any(event["side"] == "BUY" for event in valid), sell_valid=any(event["side"] == "SELL" for event in valid),
                    origin_event_ids=[event["event_id"] for event in valid],
                    dependencies=list(dict.fromkeys(dependency for event in valid for dependency in event.get("dependencies", []))))
                state["current_state"].update(buy_score=demand, sell_score=supply, current_direction_valid=bool(valid),
                    wyckoff_event=feature.get("wyckoff_event"), wyckoff_state=feature.get("wyckoff_state"),
                    wyckoff_available_at=feature.get("wyckoff_available_at"), wyckoff_observed_at=feature.get("wyckoff_observed_at"),
                    wyckoff_range_formed_at=feature.get("wyckoff_range_formed_at"),
                    wyckoff_range_low=_finite(feature.get("wyckoff_range_low")), wyckoff_range_high=_finite(feature.get("wyckoff_range_high")))
                if not input_ok:
                    state["reason_codes"].append("wyckoff_current_metadata_or_input_unavailable")
                elif active[source] and not valid:
                    state["reason_codes"].append("wyckoff_current_strength_or_range_invalidated")
            state["valid"] = state["buy_valid"] or state["sell_valid"]
            invalidated_ids.update(state["invalidated_event_ids"])
            state["side"] = "CONFLICT" if state["buy_valid"] and state["sell_valid"] else "BUY" if state["buy_valid"] else "SELL" if state["sell_valid"] else "ABSTAIN"
            state["source_id"] = stable_hash({"version": EVIDENCE_VERSION, "symbol": symbol, "source": source, **state})
            evaluation["direction_state"] = state
            evaluation.update(source=source, available_at=state["available_at"], source_id=state["source_id"], valid=state["valid"],
                              input_eligible=state["input_eligible"], side=state["side"], origin_event_ids=state["origin_event_ids"])
            evaluations[source] = evaluation
        memory = [(index, event) for index, event in memory if event["event_id"] not in invalidated_ids]
        plan = row.get("raw_risk_plan") or {}
        details = []
        for event in facts:
            stop = _finite(event.get("stop_price"))
            details.append({"event_id": event.get("event_id"), "source": event.get("source"), "side": event.get("side"),
                "direction_eligible": event["direction_eligible"], "candidate_eligible": event.get("candidate_eligible") is True,
                "stop_price": stop, "reference_price": close,
                "stop_distance": (close - stop) / close if close and stop is not None else None,
                "risk_reason_codes": [code for code in event.get("reason_codes", []) if code in RISK_REASONS]})
        risk_reasons = list(dict.fromkeys([*(plan.get("reason_codes") or []),
            *(code for detail in details for code in detail["risk_reason_codes"])]))
        output[day] = {"version": EVIDENCE_VERSION, "facts": facts, "evaluations": evaluations,
                       "candidate_required": True, "candidate_risk_eligible": plan.get("entry_allowed") is True,
                       "candidate_risk_reasons": risk_reasons, "risk_details": details,
                       "provenance": copy.deepcopy(policy)}
    return output
