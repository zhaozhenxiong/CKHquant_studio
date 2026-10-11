"""Strictly forward signal routing and real-Broker high-return research policy.

Historical reconstruction is explicitly separate from actual production time.
All application scoring remains shadow; the offline replay is the only caller
which authorizes these heads to change orders.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_high_return_features import (
    entry_feature_row, holding_feature_row, event_prefix_rows,
)
from my_strategy.services.czsc_high_return_labels import label_config
from my_strategy.services.czsc_high_return_models import HeadSession, load_head, bind_inputs
from my_strategy.services.czsc_signal_points import build_signal_points, make_point_decisions

FAMILY = "high_return"
ENTRY_TARGET = "按冻结退出政策形成实际完整回合：费用后收益至少8%且至少2倍初始风险"
EXIT_TARGET = "下一核验开盘退出相对冻结续持政策的剩余净价值优势"


def _config(config=None):
    from my_strategy.core.config_loader import load_config
    base = load_config("czsc_high_return_research")
    for key, value in (config or {}).items():
        base[key] = {**base.get(key, {}), **value} if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    if config and config.get('version') == 'czsc_high_return_v1' and 'ma_entry_alignment' not in config:
        # Historical v1 reports retain their original exit routing semantics.
        base.pop('ma_entry_alignment', None)
    from my_strategy.services.czsc_high_return_ma_training import validate_alignment_version
    validate_alignment_version(base)
    label_config(base)
    return base


def _entry_target(config):
    settings = label_config(config)
    return f"按冻结退出政策形成实际完整回合：费用后收益至少{settings['min_net_return']:.0%}且至少{settings['min_r_multiple']:g}倍初始风险"


def _day(value):
    return pd.Timestamp(value).date().isoformat()


def _head_path(root, bundle, head):
    value = (bundle or {}).get("heads", {}).get(head)
    if not value:
        return None
    path = value.get("path") if isinstance(value, dict) else value
    if not isinstance(path, str):
        raise ValueError("high-return head path missing")
    root = Path(root).resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError("high-return head outside immutable model run")
    return target.parent if target.name == "manifest.json" else target


def _bundle_for(day, bundles, *, validation_bundle=None, usage_mode="historical", head=None, model_root=None):
    if validation_bundle is not None:
        if head is not None and not validation_bundle.get("heads", {}).get(head):
            return None
        fold = validation_bundle.get("fold", validation_bundle)
        if fold.get("calibration_end") and fold.get("policy_start") and fold.get("policy_end"):
            return validation_bundle if fold["calibration_end"] < day and fold["policy_start"] <= day <= fold["policy_end"] else None
        if model_root is None or head is None:
            raise ValueError("validation route requires the frozen policy interval")
        path = _head_path(model_root, validation_bundle, head)
        if path is None:
            return None
        fold = load_head(path)["fold"]
        return validation_bundle if fold["calibration_end"] < day and fold["policy_start"] <= day <= fold["policy_end"] else None
    choices = [bundle for bundle in bundles if str(bundle.get("available_at", "9999"))[:10] < day
               and (head is None or bundle.get("heads", {}).get(head))]
    if usage_mode == "production" and model_root is not None and head:
        cutoff = pd.Timestamp(day + "T15:00:00+08:00")
        choices = [bundle for bundle in choices if pd.to_datetime(load_head(_head_path(model_root, bundle, head))["training_completed_at"], utc=True) <= cutoff]
    return max(choices, key=lambda item: (item["available_at"], item.get("name", ""))) if choices else None


def _manifest(session, path):
    key = str(Path(path).resolve())
    return session.cache[key][0] if key in session.cache else load_head(path)


def _validate_compatibility(contract, *, contract_config=None, source=None, initial_cash=None):
    if contract_config is not None:
        from my_strategy.services.czsc_trade_episodes import _settings
        if contract.get("high_return") != label_config(contract_config):
            raise ValueError("high-return model frozen outcome/exit protocol mismatch")
        execution_hash = contract.get("execution_config_hash")
        strategy_hash = contract.get("strategy_configuration_hash")
        if execution_hash and execution_hash != stable_hash(_settings(contract_config)["execution"]):
            raise ValueError("high-return model Broker execution contract mismatch")
        if strategy_hash and strategy_hash != stable_hash(_settings(contract_config)):
            raise ValueError("high-return model strategy contract mismatch")
        if not execution_hash and contract.get("strategy_hash") != stable_hash(contract_config):
            raise ValueError("high-return model frozen configuration mismatch")
    if source is not None and contract.get("source") != source:
        raise ValueError("high-return model candidate/continuation source mismatch")
    if initial_cash is not None and not math.isclose(float(contract["initial_cash"]), float(initial_cash), rel_tol=0, abs_tol=.000001):
        raise ValueError("high-return model initial-capital contract mismatch")


def _predict(session, values, root, bundle, head, *, validation=False, initial_cash=None, usage_mode="historical", contract_config=None, source=None):
    path = _head_path(root, bundle, head)
    if path is None:
        return None, None
    manifest = _manifest(session, path)
    if manifest["head"] != head:
        raise ValueError("bundle high-return head identity mismatch")
    contract = manifest["target_contract"]
    validated = getattr(session, "_verified_usage_contracts", set())
    compatibility_key = stable_hash({"target": contract, "config": contract_config, "source": source, "cash": initial_cash})
    if compatibility_key not in validated:
        _validate_compatibility(contract, contract_config=contract_config, source=source, initial_cash=initial_cash)
        validated.add(compatibility_key)
        session._verified_usage_contracts = validated
    from threadpoolctl import threadpool_limits
    with nullcontext() if getattr(session, "_cpu_inference_scope_active", False) else threadpool_limits(limits=1):
        result = session.predict_one(values, path, expected_head=head, expected_contract=contract,
            validation=validation, usage_mode="production" if usage_mode == "production" else "historical_reconstruction")
    return float(result) if np.isfinite(result) else None, manifest


def _point_cache_identity(features, calendar, model_root, bundles, config, usage_mode):
    frame = features.copy(deep=False).reset_index(drop=True)
    frame = frame.assign(date=pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d"))
    content = frame.to_json(orient="records", date_format="iso", double_precision=15)
    ma_routes = [{"name": bundle.get("name"), "available_at": bundle.get("available_at"),
        "ma_entry": bundle["heads"]["ma_entry"]} for bundle in bundles if bundle.get("heads", {}).get("ma_entry")]
    from my_strategy.services.czsc_signal_points import signal_point_parameters
    return {"symbol": str(frame.iloc[0].symbol) if len(frame) else None,
        "rows": len(frame), "dates_hash": stable_hash(frame.date.tolist()),
        "input_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(), "columns": list(frame.columns),
        "calendar_hash": stable_hash(list(calendar)), "config_hash": stable_hash(signal_point_parameters(config)),
        "l1_bundles_hash": stable_hash(ma_routes), "model_root": str(Path(model_root).resolve()), "usage_mode": usage_mode}


def _point_payload_hash(points):
    return stable_hash({key: points.get(key) for key in ("version", "config_hash", "events", "candidates", "l1_oof_lineage")})


def _execution_frame(frame, start, config, verified_sources):
    """Retain observed seasoning bars without recomputing any causal X/event."""
    if not start:
        return frame.reset_index(drop=True)
    before = frame.loc[frame.date < start]
    active = np.flatnonzero(((before.volume > 0) & (before.amount > 0) &
        (before.has_trade_price == 1) & before.source.isin(verified_sources)).to_numpy())
    from my_strategy.services.czsc_trade_episodes import _settings
    count = max(70, int(_settings(config)["execution"]["seasoned_bars"]) + 10)
    begin = int(active[-count]) if len(active) >= count else 0
    return pd.concat([before.iloc[begin:], frame.loc[frame.date >= start]], ignore_index=True)


def forward_ma_outputs(features, *, model_root, bundles, validation_bundle=None, session=None, usage_mode="historical", config=None):
    """One forward entry score per date; no synthetic holding creates MA exits."""
    session = session or HeadSession()
    count = len(features)
    probability = np.full(count, np.nan)
    lineages = [{} for _ in range(count)]
    fallback = np.ones(count, dtype=bool)
    dates = pd.to_datetime(features.date).dt.strftime("%Y-%m-%d").tolist()
    groups = {}
    for index, day in enumerate(dates):
        bundle = _bundle_for(day, bundles, validation_bundle=validation_bundle, usage_mode=usage_mode,
                             head="ma_entry", model_root=model_root)
        if bundle is not None:
            groups.setdefault(bundle["name"], (bundle, []))[1].append(index)
    for bundle, indices in groups.values():
        path = _head_path(model_root, bundle, "ma_entry")
        manifest = _manifest(session, path)
        if config is not None:
            _validate_compatibility(manifest["target_contract"], contract_config=_config(config), source="ma", initial_cash=_config(config).get("initial_cash", 100000))
        frame = features.iloc[indices].copy()
        frame["input_eligible"] = frame.get("model_input_eligible", pd.Series(False, index=frame.index)).fillna(False)
        frame = bind_inputs(frame, manifest["columns"], manifest["head"])
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            outputs = session.predict(frame, path, expected_head="ma_entry", expected_contract=manifest["target_contract"],
                validation=validation_bundle is not None,
                usage_mode="production" if usage_mode == "production" else "historical_reconstruction")
        for index, value in zip(indices, outputs, strict=True):
            probability[index] = value
            fallback[index] = False
            lineages[index] = {"entry": {"head": "entry", "model_head": "ma_entry", "model_run_id": Path(model_root).name,
                "checkpoint": bundle["name"], "available_at": manifest["fold"]["calibration_end"] if validation_bundle is not None else manifest["available_at"],
                "train_label_end": manifest["train_label_end"], "probability_target": manifest["target_contract"]["target"],
                "target": manifest["target_contract"]["target"], "target_contract_hash": manifest["target_contract_hash"],
                "manifest_sha256": manifest["manifest_sha256"], "historical_reconstruction": usage_mode != "production",
                "actual_completion_record": manifest["training_completed_at"], "model_available": True,
                "validation_policy_selection": validation_bundle is not None}}
    return {"ma_entry_probabilities": probability, "ma_lineages": lineages, "ma_fallback_mask": fallback,
            "unavailable_rows": int((~np.isfinite(probability)).sum()), "session": session}


def prepare_points(features, calendar, *, model_root, bundles, config=None, validation_bundle=None, session=None, usage_mode="historical"):
    output = forward_ma_outputs(features, model_root=model_root, bundles=bundles,
                                validation_bundle=validation_bundle, session=session, usage_mode=usage_mode, config=config)
    points = build_signal_points(features, _config(config), ma_entry_probabilities=output["ma_entry_probabilities"],
        ma_lineages=output["ma_lineages"], ma_fallback_mask=output["ma_fallback_mask"])
    points["l1_oof_lineage"] = [{"date": _day(day), "entry": lineage.get("entry", {}),
        "mode": "causal_rule_v0" if fallback else "forward_model", "probability": float(p) if np.isfinite(p) else None}
        for day, lineage, fallback, p in zip(features.date, output["ma_lineages"], output["ma_fallback_mask"], output["ma_entry_probabilities"], strict=True)]
    points["compute_info"] = dict(output["session"].diagnostics)
    if validation_bundle is None:
        points["prepared_cache_identity"] = _point_cache_identity(features, calendar, model_root, bundles, _config(config), usage_mode)
        points["prepared_payload_hash"] = _point_payload_hash(points)
    return points


def make_ma_exit_policy(features, calendar, model_root, bundles, *, validation_bundle=None, config=None,
                        session=None, points=None, usage_mode='historical', allow_training=False):
    """Score actual holdings under the explicitly frozen L1 exit version."""
    session = session or HeadSession()
    frozen_config = _config(config)
    if frozen_config.get('ma_entry_alignment') == 'signal_pinned_learned_exit_v2':
        from my_strategy.services.czsc_high_return_ma_training import read_exit_bootstrap_lineage
        from my_strategy.services.czsc_high_return_ma_alignment import make_signal_pinned_ma_exit_policy
        lineage = read_exit_bootstrap_lineage(model_root,calendar,frozen_config,bundles=bundles,
            allow_training=allow_training)
        if points is None:
            points = prepare_points(features,calendar,model_root=model_root,bundles=bundles,
                config=frozen_config,session=session,usage_mode=usage_mode)
        return make_signal_pinned_ma_exit_policy(features,calendar,model_root,lineage['exit_only_bundles'],
            config=frozen_config,points=points,session=session,usage_mode=usage_mode,
            expected_exit_contract_hash=lineage['expected_exit_target_contract_hash'],
            source_snapshot_hash=lineage['source_snapshot_hash'],
            exit_training_calendar_hash=lineage['calendar_hash'],
            exit_training_calendar_dates=lineage['_frozen_calendar_dates'])
    rows = {_day(row["date"]): row for row in features.to_dict("records")}
    market_index = {str(day): i for i, day in enumerate(calendar)}
    prefixes, diagnostics = {}, []
    def callback(snapshot):
        day = snapshot["date"]
        key = (snapshot["symbol"], snapshot["entry_date"], snapshot.get("plan_id"))
        prefix = prefixes.setdefault(key, {})
        if prefix.get("through_date") and day <= prefix["through_date"]:
            raise ValueError("MA exit prefix reused across Broker replays; use callback.new_replay()")
        prefix["through_date"] = day
        values = holding_feature_row(rows[day], snapshot, entry_values={}, prefix=prefix, market_index=market_index)
        values["input_eligible"] = bool(rows[day].get("model_input_eligible", False))
        bundle = _bundle_for(day, bundles, validation_bundle=validation_bundle, head="ma_exit", model_root=model_root)
        response = {"head": "holding_exit", "contract_hash": stable_hash({"mode": "ma_exit", "bundles": bundles}),
            "probability": None, "request_exit": False, "apply": False, "research_arm": True,
            "status": "ma_exit_model_unavailable_at_signal", "holding_contract_compatible": True,
            "prefix_through_date": day, "holding_prefix_source": "actual_broker_daily_close_snapshots"}
        if bundle:
            probability, manifest = _predict(session, values, model_root, bundle, "ma_exit",
                validation=validation_bundle is not None, initial_cash=snapshot["initial_cash"], contract_config=frozen_config, source="ma")
            threshold = float(bundle.get("policy", {}).get("ma_exit_threshold", frozen_config["signal_points"]["ma_exit_threshold"]))
            response.update(probability=probability, request_exit=probability is not None and probability >= threshold,
                apply=probability is not None, threshold=threshold, contract_hash=manifest["target_contract_hash"],
                checkpoint=bundle["name"], status="forward_ma_actual_holding_exit" if probability is not None else "ma_exit_input_unavailable")
        diagnostics.append({**response, "date": day, "entry_date": snapshot["entry_date"]})
        return response
    callback.diagnostics = diagnostics
    callback.session = session
    callback.new_replay = lambda: make_ma_exit_policy(features, calendar, model_root, bundles,
        validation_bundle=validation_bundle, config=frozen_config, session=session)
    return callback


def replay_high_return(features, calendar, *, model_root, bundles, config, source="union", mode="full",
                       start=None, end=None, initial_cash=100000, validation_bundle=None, arm=None, contract_config=None,
                       prepared_points=None, session=None, allow_training=False):
    """Bound CPU inference threads once per replay, sharing only model caches."""
    from threadpoolctl import threadpool_limits
    session = session or HeadSession()
    previous = getattr(session, "_cpu_inference_scope_active", False)
    with threadpool_limits(limits=1):
        session._cpu_inference_scope_active = True
        try:
            return _replay_high_return(features, calendar, model_root=model_root, bundles=bundles, config=config,
                source=source, mode=mode, start=start, end=end, initial_cash=initial_cash,
                validation_bundle=validation_bundle, arm=arm, contract_config=contract_config,
                prepared_points=prepared_points, session=session, allow_training=allow_training)
        finally:
            session._cpu_inference_scope_active = previous


def _replay_high_return(features, calendar, *, model_root, bundles, config, source="union", mode="full",
                       start=None, end=None, initial_cash=100000, validation_bundle=None, arm=None, contract_config=None,
                       prepared_points=None, session=None, allow_training=False):
    """Continuous real cash/inventory across checkpoints; explicitly offline."""
    from my_strategy.services.czsc_backtest import execute_decisions
    from my_strategy.services.czsc_trade_episodes import _settings
    from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
    mode = arm or mode
    if mode not in {"full", "entry_only", "exit_only", "reference"}:
        raise ValueError("unsupported high-return policy mode")
    config = _config(config)
    contract_config = _config(contract_config) if contract_config is not None else config
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    if end:
        frame = frame[frame.date <= str(end)[:10]].reset_index(drop=True)
    if frame.empty or frame.symbol.nunique(dropna=False) != 1:
        raise ValueError("high-return replay requires a nonempty single-symbol frame")
    if list(frame.date) != sorted(set(frame.date)) or not set(frame.date).issubset(calendar):
        raise ValueError("high-return replay requires ascending verified dates")
    session = session or HeadSession()
    compute_start = dict(session.diagnostics)
    expected_identity = _point_cache_identity(frame, calendar, model_root, bundles, contract_config, "historical")
    if prepared_points is not None:
        if prepared_points.get("prepared_cache_identity") != expected_identity or prepared_points.get("prepared_payload_hash") != _point_payload_hash(prepared_points):
            raise ValueError("prepared high-return points do not match the frozen full input/date/config/model identity")
        points = prepared_points
    else:
        # The first stage must use the same already available MA checkpoints
        # as P2b; L2 validation selection cannot borrow a newer first stage.
        points = prepare_points(frame, calendar, model_root=model_root, bundles=bundles,
                                config=contract_config, session=session)
    cache = getattr(session, "_high_return_prefix_cache", {})
    prefix_key = stable_hash({"identity": expected_identity, "points": points.get("prepared_payload_hash")})
    if prefix_key not in cache:
        cache[prefix_key] = dict(zip(frame.date, event_prefix_rows(frame, points["events"], calendar), strict=True))
        session._high_return_prefix_cache = cache
    full_prefix = cache[prefix_key]
    frame = _execution_frame(frame, start, config, VERIFIED_SOURCES)
    decisions = make_point_decisions(frame, points, source=source, config=config, start=start)
    rows = frame.to_dict("records")
    by_date = {row["date"]: i for i, row in enumerate(rows)}
    candidates = {c["date"]: c for c in points["candidates"]}
    event_prefixes = [full_prefix[day] for day in frame.date]
    index = {str(day): i for i, day in enumerate(calendar)}
    saved_entries, prefixes = {}, {}
    policy_hash = stable_hash({"mode": mode, "source": source, "bundles": bundles,
                               "config": config, "initial_cash": initial_cash})
    ma_exit = make_ma_exit_policy(frame, calendar, model_root, bundles, config=contract_config,
                                  session=session, points=points,
                                  allow_training=allow_training) if source == "ma" else None
    def entry_gate(state):
        day = state["date"]; position = by_date[day]
        candidate = candidates[day]; plan = state["entry_plan"]
        values = entry_feature_row(rows[position], state, candidate, plan,
                                   initial_cash=initial_cash, event_prefix=event_prefixes[position])
        values["input_eligible"] = bool(candidate.get("candidate_eligible"))
        saved_entries[state["plan_id"]] = values
        response = {"head": "policy_entry", "contract_hash": policy_hash, "probability": None,
            "request_entry": True, "apply": False, "research_arm": True, "status": "frozen_reference_entry",
            "model_family": FAMILY, "probability_target": _entry_target(config)}
        if mode not in {"full", "entry_only"}:
            return response
        bundle = _bundle_for(day, bundles, validation_bundle=validation_bundle, head="entry", model_root=model_root)
        if bundle is None:
            response["status"] = "entry_model_unavailable_at_signal_reference_fallback"
            return response
        probability, manifest = _predict(session, values, model_root, bundle, "entry",
            validation=validation_bundle is not None, initial_cash=initial_cash, contract_config=contract_config, source="union")
        expected_return, _ = _predict(session, values, model_root, bundle, "return",
            validation=validation_bundle is not None, initial_cash=initial_cash, contract_config=contract_config, source="union")
        loss_probability, _ = _predict(session, values, model_root, bundle, "loss",
            validation=validation_bundle is not None, initial_cash=initial_cash, contract_config=contract_config, source="union")
        threshold = float(bundle.get("policy", {}).get("entry_threshold", .55))
        response.update(expected_net_return=expected_return, loss_probability=loss_probability,
            probability=probability, request_entry=probability is None or probability >= threshold,
            apply=probability is not None, threshold=threshold, checkpoint=bundle["name"],
            contract_hash=manifest["target_contract_hash"], status="offline_high_return_entry" if probability is not None else "entry_input_unavailable_reference_fallback")
        return response
    def holding_callback(snapshot):
        day = snapshot["date"]; position = by_date[day]
        key = (snapshot["entry_date"], snapshot.get("plan_id"))
        prefix = prefixes.setdefault(key, {})
        prefix.update(event_prefixes[position])
        values = holding_feature_row(rows[position], snapshot, entry_values=saved_entries.get(snapshot.get("plan_id"), {}),
                                     prefix=prefix, market_index=index)
        values["input_eligible"] = any(values.get(source + "_input_available", 0) for source in ("ma", "structure", "wyckoff"))
        bundle = _bundle_for(day, bundles, validation_bundle=validation_bundle, head="exit", model_root=model_root)
        age = index[day] - index[snapshot["entry_date"]] + 1
        maximum = int(config["high_return"]["max_holding_sessions"])
        response = {"head": "holding_exit", "contract_hash": policy_hash, "probability": None,
            "request_exit": False, "apply": False, "research_arm": True, "status": "frozen_reference_exit",
            "model_family": FAMILY, "holding_contract_compatible": True, "holding_prefix_source": "actual_broker_daily_close_snapshots",
            "prefix_through_date": day, "holding_entry_date": snapshot["entry_date"], "probability_target": EXIT_TARGET}
        if age >= maximum:
            return {**response, "probability": 1., "request_exit": True, "apply": True,
                    "status": "frozen_max_holding_sessions", "forced_by_policy": True}
        if ma_exit:
            return ma_exit(snapshot)
        if mode not in {"full", "exit_only"} or bundle is None:
            if bundle is None:
                response["status"] = "exit_model_unavailable_at_signal_reference_fallback"
            return response
        probability, manifest = _predict(session, values, model_root, bundle, "exit",
            validation=validation_bundle is not None, initial_cash=initial_cash, contract_config=contract_config, source="union")
        advantage, _ = _predict(session, values, model_root, bundle, "exit_advantage",
            validation=validation_bundle is not None, initial_cash=initial_cash, contract_config=contract_config, source="union")
        threshold = float(bundle.get("policy", {}).get("exit_threshold", .55))
        margin = float(config.get("high_return", {}).get("exit_advantage_margin", .002))
        response.update(probability=probability, request_exit=probability is not None and probability >= threshold and advantage is not None and advantage > margin,
            exit_advantage=advantage, no_action_margin=margin, apply=probability is not None, threshold=threshold,
            contract_hash=manifest["target_contract_hash"], checkpoint=bundle["name"],
            status="exit_advantage_model_unavailable_reference_fallback" if advantage is None else "offline_high_return_exit" if probability is not None else "exit_input_unavailable_reference_fallback")
        return response
    account = execute_decisions(str(frame.iloc[0].symbol), frame, decisions, start or frame.iloc[0].date,
        float(initial_cash), _settings(config), "offline-high-return-" + mode,
        market_dates=list(calendar), verified_sources=VERIFIED_SOURCES, entry_policy="risk",
        entry_gate=entry_gate, exit_policy=holding_callback)
    account.update(symbol=str(frame.iloc[0].symbol), model_family=FAMILY, policy_mode=mode, policy_source=source,
        decisions=decisions, signal_points=points["events"], candidate_points=points["candidates"], policy_hash=policy_hash,
        publication_allowed=False, model_routes=[{"date": day, "checkpoint": (_bundle_for(day, bundles, validation_bundle=validation_bundle, head="entry", model_root=model_root) or {}).get("name")}
            for day in frame.date if not start or day >= start],
        compute_info={"model_family": FAMILY, "research": True, "actual_device": "cpu", "selected_device": "cpu",
            "model_inference_rows": session.diagnostics["rows"] - compute_start["rows"],
            "model_inference_batches": session.diagnostics["batches"] - compute_start["batches"],
            "model_loads": session.diagnostics["model_loads"] - compute_start["model_loads"],
            "l1_points_reused": prepared_points is not None,
            "full_prefix_rows": expected_identity["rows"], "execution_rows": len(frame),
            "inference_implementation": "verified_cpu_export"})
    gates = account.get("entry_gate_diagnostics", [])
    exits = account.get("exit_policy_diagnostics", [])
    account["model_coverage"] = {
        "total_candidates": sum(bool(point.get("candidate_eligible") if source == "union" else point.get("source_eligibility", {}).get(source))
            for point in points["candidates"] if (not start or point["date"] >= start) and (not end or point["date"] <= end)),
        "selected_candidates": sum(fill["action"] == "BUY" for fill in account["ledger"]),
        "decidable_candidates": len(gates), "model_scored_candidates": sum(item.get("probability") is not None for item in gates),
        "actual_holding_states": len(account.get("holding_snapshots", [])),
        "model_scored_holding_states": sum(item.get("probability") is not None and not item.get("forced_by_policy") for item in exits)}
    return account


def _read_run(model_run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", str(model_run_id)):
        raise ValueError("invalid high-return model run identity")
    root = (ARTIFACT_RUNS_ROOT / str(model_run_id)).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):
        raise ValueError("high-return model run outside storage")
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((root / "reports/high_return.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "complete" or report.get("model_family") != FAMILY:
        raise ValueError("high-return training has not completed")
    return root, report, metadata


def high_return_research_status():
    models, failures = [], []
    for report_path in sorted(ARTIFACT_RUNS_ROOT.glob("*/reports/high_return.json")):
        try:
            root, report, metadata = _read_run(report_path.parent.parent.name)
            checkpoints = []
            for bundle in report.get("model_bundles", []):
                entry_path = _head_path(root, bundle, "entry")
                if entry_path is None:
                    continue
                manifests = [load_head(_head_path(root, bundle, head)) for head in bundle.get("heads", {}) if bundle["heads"][head]]
                entry = load_head(entry_path)
                checkpoints.append({"name": bundle["name"], "available_at": bundle["available_at"],
                    "train_label_end": entry["train_label_end"], "training_completed_at": max(m["training_completed_at"] for m in manifests),
                    "model_family": FAMILY, "policy": bundle.get("policy", {}), "feature_schema_hash": entry["feature_schema_hash"]})
            if checkpoints:
                snapshot = report.get("source_snapshot", {})
                coverage = report.get("coverage", {})
                evaluation = report.get("evaluation", {})
                evaluation_windows = evaluation.get("windows", {}) if isinstance(evaluation, dict) else {}
                account_failures = any(variant.get("cash_retained_failures") for window in evaluation_windows.values()
                    for variant in window.get("variants", {}).values())
                completed_without_failures = metadata.get("status") == "complete" and report.get("status") == "completed" and not (
                    report.get("failures") or coverage.get("failures") or coverage.get("failed") or account_failures)
                models.append({"run_id": root.name, "model_family": FAMILY,
                    "strategy_version": report.get("config", {}).get("version", "czsc_high_return_v1"), "checkpoints": checkpoints,
                    "initial_cash": float(_config(report.get("config")).get("initial_cash", 100000)),
                    "coverage": coverage, "model_gate": report.get("model_gate", {"passed": False}),
                    "source_training_run_id": report.get("source_training_run_id", snapshot.get("run_id", snapshot.get("source_run_id"))),
                    "full_source_universe": bool(snapshot.get("full_source_universe")), "selected_count": snapshot.get("selected_count"),
                    "source_snapshot_hash": snapshot.get("snapshot_hash"), "data_end": report.get("data_end"),
                    "completed_without_failures": completed_without_failures,
                    "automatic_selection_eligible": bool(snapshot.get("full_source_universe")) and completed_without_failures,
                    "training_completed_at": max(c["training_completed_at"] for c in checkpoints), "publication_allowed": False})
        except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            failures.append({"run_id": report_path.parent.parent.name, "error": str(exc)})
    return {"model_family": FAMILY, "models": models, "failures": failures, "active_release": None,
            "publication_allowed": False, "source_training_run_id": _config().get("source_training_run_id"),
            "reason": "新独立验证尚未完成，高收益区间模型保持影子"}


status_high_return = high_return_research_status


def runtime_overlay(features, market_dates, *, model_run_id=None, model_fold=None,
                    model_mode="production", as_of=None, holding_snapshot=None, device="cpu"):
    """Causal candidate scores for display; no simulated inventory is invented."""
    from my_strategy.services.czsc_research_ml import _device
    if model_mode not in {"production", "historical"}:
        raise ValueError("高收益区间模型仅支持当前生产影子或历史滚动研究；事后模型模式不可用")
    selected_device = str(_device(device or "cpu"))
    frame = features.copy()
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    cutoff = str(as_of)[:10] if as_of else frame.iloc[-1].date
    frame = frame[frame.date <= cutoff].reset_index(drop=True)
    if frame.empty or frame.symbol.nunique(dropna=False) != 1:
        raise ValueError("高收益判断需要截至决策日的非空单股因果输入")
    if list(frame.date) != sorted(set(frame.date)) or not set(frame.date).issubset(market_dates):
        raise ValueError("高收益输入日期必须与核验日历一致")
    root, bundles, config = None, [], _config()
    if model_run_id is None:
        models = [model for model in high_return_research_status()["models"] if model.get("automatic_selection_eligible")]
        if models:
            model_run_id = max(models, key=lambda item: item["training_completed_at"])["run_id"]
    if model_run_id:
        root, report, _ = _read_run(model_run_id)
        bundles = report.get("model_bundles", [])
        config = _config(report.get("config"))
        if model_fold:
            bundles = [bundle for bundle in bundles if bundle["name"] == model_fold]
            if not bundles:
                raise ValueError("固定高收益检查点不存在")
            if str(bundles[0]["available_at"])[:10] >= frame.iloc[-1].date:
                raise ValueError("固定高收益检查点在决策日尚不可用")
    snapshot_scope = report.get("source_snapshot", {}) if model_run_id else {}
    scope = {"full_source_universe": bool(snapshot_scope.get("full_source_universe")),
        "partial_scope": bool(model_run_id and not snapshot_scope.get("full_source_universe")),
        "selected_count": snapshot_scope.get("selected_count"), "source_snapshot_hash": snapshot_scope.get("snapshot_hash")}
    session = HeadSession()
    points = prepare_points(frame, market_dates, model_root=root or ARTIFACT_RUNS_ROOT, bundles=bundles,
                            config=config, session=session, usage_mode=model_mode)
    decisions = make_point_decisions(frame, points, config=config)
    scoring_decisions = make_point_decisions(frame, points, config=config, include_unselected=True)
    raw_rows = frame.to_dict("records")
    candidates = {point["date"]: point for point in points["candidates"]}
    events_by_day = {}
    for point in points["events"]:
        events_by_day.setdefault(point["date"], []).append(point)
    prefixes = event_prefix_rows(frame, points["events"], market_dates)
    rows, reason_counts = [], Counter()
    following = dict(zip(market_dates[:-1], market_dates[1:]))
    for position, decision in enumerate(decisions):
        day = decision["date"]; feature = raw_rows[position]
        candidate, probability, manifest = candidates.get(day), None, None
        expected_return, loss_probability = None, None
        raw_plan = scoring_decisions[position].get("entry_plan") or {}
        risk_eligible = bool(candidate and candidate.get("candidate_eligible") and raw_plan.get("entry_allowed"))
        bundle = _bundle_for(day, bundles, usage_mode=model_mode, head="entry", model_root=root)
        reason = "no_candidate_signal" if candidate is None else "candidate_risk_ineligible" if not risk_eligible else "model_unavailable_at_signal" if bundle is None else None
        if candidate and bundle and risk_eligible:
            plan = raw_plan
            initial_cash = float(config.get("initial_cash", 100000))
            state = {"date": day, "close": float(feature["close"]), "cash": initial_cash, "equity": initial_cash}
            values = entry_feature_row(feature, state, candidate, plan, initial_cash=initial_cash, event_prefix=prefixes[position])
            values["input_eligible"] = risk_eligible
            probability, manifest = _predict(session, values, root, bundle, "entry", initial_cash=initial_cash, usage_mode=model_mode, contract_config=config, source="union")
            if probability is not None:
                expected_return, _ = _predict(session, values, root, bundle, "return", initial_cash=initial_cash, usage_mode=model_mode, contract_config=config, source="union")
                loss_probability, _ = _predict(session, values, root, bundle, "loss", initial_cash=initial_cash, usage_mode=model_mode, contract_config=config, source="union")
            if probability is None:
                reason = "candidate_input_ineligible"
        reason_counts[reason or "shadow_scored"] += 1
        input_available = bool(feature.get("model_input_eligible", False) or feature.get("structure_input_eligible", False) or feature.get("wyckoff_input_eligible", False))
        identity = {"model_family": FAMILY, "model_run_id": model_run_id, **scope,
            "checkpoint": bundle.get("name") if bundle else None,
            "available_at": bundle.get("available_at") if bundle else None,
            "actual_training_completed_at": manifest.get("training_completed_at") if manifest else None,
            "probability": probability, "probability_target": _entry_target(config),
            "model_input_eligible": input_available,
            "applied_to_entry": False, "status": "historical_shadow" if model_mode == "historical" else "production_shadow",
            "reason_codes": [reason] if reason else ["high_return_independent_window_missing"],
            "publication_allowed": False, "historical_reconstruction": model_mode == "historical"}
        row = {**decision, "model_family": FAMILY, "model_resolution": identity,
            "entry_probability": probability, "high_return_probability": probability,
            "expected_net_return": expected_return, "loss_probability": loss_probability,
            "candidate_sources": candidate.get("sources", []) if candidate else [],
            "signal_points": events_by_day.get(day, []), "probability_target": _entry_target(config),
            "exit_probability": None, "exit_model": {"probability": None, "reason": "缺少兼容真实Broker持仓历史，未生成退出概率", "holding_contract_compatible": False},
            "reference_close": float(feature["close"]), "reference_price_date": day, "next_market_session": following.get(day),
            "category": "buy" if decision["candidate_action"] == "BUY" else "watch",
            "action": "WAIT", "position_intent": "WAIT", "eligible": input_available, "input_eligible": input_available,
            "source_candidate_eligible": bool(candidate and candidate.get("candidate_eligible")),
            "risk_eligible": risk_eligible, "candidate_eligible": risk_eligible,
            "source_selected_by_reference": bool(candidate and candidate.get("selected_by_reference")),
            "raw_risk_plan": raw_plan or None,
            "entry_state_scope": "hypothetical_flat_account_frozen_initial_cash", "basis": "假设按冻结初始资金新入场的三路候选与高收益影子判断；实际交易以独立Broker回放为准", "publication_allowed": False}
        rows.append(row)
    # Only a complete observed holding history can support the running extrema.
    if holding_snapshot is not None:
        snapshot = copy.deepcopy(holding_snapshot)
        history = snapshot.get("history", [])
        required = {"shares", "entry_shares", "entry_date", "actual_cost_per_share", "cash", "equity", "bar", "date", "symbol", "available_at"}
        day = rows[-1]["date"]
        if required <= set(snapshot) and snapshot["shares"] > 0 and snapshot["symbol"] == str(frame.iloc[0].symbol) and snapshot["date"] == day and history:
            dates = [_day(value["date"]) for value in history]
            expected = [value for value in market_dates if snapshot["entry_date"] <= value <= day]
            compatible = dates == expected and all(value.get("entry_date") == snapshot["entry_date"] and value.get("shares", 0) > 0 for value in history)
            bundle = _bundle_for(day, bundles, usage_mode=model_mode, head="exit", model_root=root)
            if compatible and bundle:
                prefix = {}
                values = None
                market_index = {d: i for i, d in enumerate(market_dates)}
                for current in history:
                    if current["date"] not in set(frame.date):
                        compatible = False; break
                    i = list(frame.date).index(current["date"])
                    prefix.update(prefixes[i])
                    values = holding_feature_row(raw_rows[i], current, entry_values=snapshot.get("entry_values", {}), prefix=prefix, market_index=market_index)
                if compatible and values is not None:
                    values["input_eligible"] = True
                    probability, manifest = _predict(session, values, root, bundle, "exit", initial_cash=snapshot.get("initial_cash", config.get("initial_cash", 100000)), usage_mode=model_mode, contract_config=config, source="union")
                    rows[-1].update(holding_snapshot=snapshot, exit_probability=probability,
                        exit_model={"probability": probability, "holding_contract_compatible": True,
                            "probability_target": EXIT_TARGET, "reason": "真实Broker持仓历史的独立退出优势影子判断",
                            "prefix_through_date": day, "holding_prefix_source": "supplied_actual_broker_history"})
    compute = {"research": True, "model_family": FAMILY, "selected_device": selected_device,
        "actual_device": "cpu" if session.diagnostics["rows"] else None,
        "model_inference_rows": session.diagnostics["rows"], "model_inference_batches": session.diagnostics["batches"],
        "model_loads": session.diagnostics["model_loads"], "inference_implementation": "verified_cpu_export",
        "inference_reason_counts": dict(reason_counts), "mps_work": False, "cuda_work": False}
    return {"model_family": FAMILY, "model_scope": scope, "signal_points": points["events"], "candidates": points["candidates"], "rows": rows,
        "latest_decision": rows[-1], "model_resolution": rows[-1]["model_resolution"], "compute_info": compute,
        "model_gate": {"passed": False, "reason": "高收益区间模型仅作影子，新独立验证尚未完成"},
        "high_return_evidence": {"sources": rows[-1]["candidate_sources"], "signal_points": rows[-1]["signal_points"]},
        "probability_target": _entry_target(config), "publication_allowed": False}
