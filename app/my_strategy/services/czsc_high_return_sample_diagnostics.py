"""Read frozen reassessment samples without changing training or admission rules.

This module deliberately lives outside the generating-code closure. It explains
the existing entry target and calls the original masks/weights as its oracle.
All output goes to a new run, never into a frozen model directory.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_trend_support import _file_hash
from my_strategy.services.czsc_trend_interval_model import (
    _atomic_first_observations, _group_ids, _weights, economic_component_ids, time_masks,
)
from my_strategy.services.czsc_trend_labels import episode_outcome, label_config

VERSION = "high_return_sample_attrition_diagnostic"
REASON_LABELS = {
    "decision_segment": "该段信号日候选",
    "selected": "实际入选", "outside_decision_segment": "信号日不在本段",
    "label_unavailable": "标签不可用", "input_ineligible": "输入资格失败",
    "invalid_binary_label": "标签不是有效二分类", "label_end_missing": "缺清仓成熟日",
    "family_first_outside_segment": "同一机会首次尝试不在本段",
    "label_end_after_cutoff": "清仓晚于本段截止", "atomic_dependency_outside_segment": "原机会原子跨段依赖",
    "entry_unfilled": "Broker买入未成交", "episode_unclosed": "实际持仓未清仓",
    "price_basis_or_corporate_action_ambiguous": "价基或公司行动有歧义",
    "risk_identity_unavailable": "初始风险身份不合格",
    "net_nonpositive": "费用后未盈利", "net_below_target": "盈利但净收益低于目标",
    "r_below_target": "净收益达标但R倍数不足", "high": "净收益与R均达标",
    "no_direction_observation": "缺有效当日方向输入", "no_active_buy_opportunity": "无有效原买方机会",
    "buy_base_not_qualified": "买方基础共识未通过", "entry_envelope_not_eligible": "风险或机会入场资格未通过",
    "account_state_or_reservation_block": "账户持仓、待执行或机会预留阻断",
    "risk_plan_rejected": "真实空仓尝试的风险计划被拒", "pre_gate_attempt": "进入质量评分前克隆人口",
    "buy_base_below_threshold": "买方基础支持不足", "independent_sources_insufficient": "独立来源不足",
    "buy_sell_margin_insufficient": "买卖支持差不足",
}


def _counts(values):
    return {str(k): int(v) for k, v in pd.Series(values).fillna("unspecified").value_counts().items()}


def _quantiles(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    values = values[np.isfinite(values)]
    return {str(q): float(values.quantile(q)) for q in (0, .25, .5, .75, .9, .99, 1)} if len(values) else {}


def _summary(frame, mask):
    selected = frame.loc[mask]
    labels = pd.to_numeric(selected.label, errors="coerce")
    return {"rows": len(selected), "positive": int(labels.eq(1).sum()),
            "negative": int(labels.eq(0).sum()), "unknown": int((~labels.isin([0, 1])).sum())}


def _target_counts(frame, mask, config):
    selected = frame.loc[mask]
    settings = label_config(config)
    valid = selected.label_available.fillna(False).astype(bool)
    net = pd.to_numeric(selected.net_return, errors="coerce")
    ratio = pd.to_numeric(selected.r_multiple, errors="coerce")
    returns = net.ge(settings["min_net_return"])
    risks = ratio.ge(settings["min_r_multiple"])
    reason = pd.Series("label_unavailable", index=selected.index)
    reason.loc[~valid] = selected.loc[~valid, "label_reason"].fillna("unspecified")
    reason.loc[valid & net.le(0)] = "net_nonpositive"
    reason.loc[valid & net.gt(0) & ~returns] = "net_below_target"
    reason.loc[valid & returns & ~risks] = "r_below_target"
    reason.loc[valid & returns & risks] = "high"
    eligible = selected.loc[valid]
    fee = pd.to_numeric(eligible.fees, errors="coerce")
    paid = pd.to_numeric(eligible.entry_cash, errors="coerce")
    # Observational comparison at the SAME frozen cash/risk denominators. This
    # does not create a zero-cost label or recompute a counterfactual risk plan.
    fee_excluded_net = pd.to_numeric(eligible.net_profit, errors="coerce") + fee
    fee_excluded_high = (fee_excluded_net.div(paid).ge(settings["min_net_return"]) &
                         fee_excluded_net.div(eligible.risk_cash).ge(settings["min_r_multiple"]))
    return {"candidates": len(selected), "label_reasons": _counts(selected.label_reason),
        "exclusive_target_reasons": _counts(reason), "available_labels": int(valid.sum()),
        "fee_net_profitable": int((valid & net.gt(0)).sum()),
        "net_return_target_pass": int((valid & returns).sum()),
        "r_multiple_target_pass": int((valid & risks).sum()),
        "both_target_pass": int((valid & returns & risks).sum()),
        "target_quadrants": {"both": int((valid & returns & risks).sum()),
            "return_only": int((valid & returns & ~risks).sum()),
            "r_only": int((valid & ~returns & risks).sum()),
            "neither": int((valid & ~returns & ~risks).sum())},
        "net_return_quantiles": _quantiles(net[valid]), "r_multiple_quantiles": _quantiles(ratio[valid]),
        "risk_fraction_quantiles": _quantiles(eligible.risk_fraction),
        "required_net_return_quantiles": _quantiles(np.maximum(settings["min_net_return"],
            settings["min_r_multiple"] * eligible.risk_fraction)),
        "fees_total": float(fee.sum()), "fees_over_entry_cash_quantiles": _quantiles(fee.div(paid)),
        "fee_excluded_same_denominator_high": int(fee_excluded_high.sum()),
        "fee_only_high_to_negative": int((fee_excluded_high & eligible.label.ne(1)).sum()),
        "fee_comparison_scope": "observed_fill_slippage_retained_same_frozen_entry_cash_and_risk_cash_no_alternative_label"}


def diagnose_entry_frame(frame, fold, config):
    """Return mutually exclusive losses and an audit row for every candidate.

    Reason order is explicit; a row failing several gates is charged only to
    its first failure. Final selection and economic weights must equal the
    unchanged training functions, including original atom dependency purges.
    """
    frame = frame.reset_index(drop=True).copy()
    dates = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    ends = pd.to_datetime(frame.label_end, errors="coerce").dt.strftime("%Y-%m-%d")
    labels = pd.to_numeric(frame.label, errors="coerce")
    available = frame.label_available.fillna(False).to_numpy(bool)
    input_ok = frame.input_eligible.fillna(False).to_numpy(bool)
    group_start = dates.groupby(_group_ids(frame)).transform("min")
    atoms, first_dates = _atomic_first_observations(frame)
    original_masks = time_masks(frame, fold, head="entry")
    audit_columns = [key for key in ("symbol", "date", "attempt_id", "economic_group_id", "original_event_ids",
        "label", "label_available", "label_reason", "label_end", "input_eligible", "net_return", "net_profit",
        "r_multiple", "risk_cash", "risk_fraction", "entry_cash", "exit_cash", "fees", "entry_date", "exit_date",
        "sources", "reference_allowed", "reference_gate_status", "reference_gate_reason", "clone_ledger_hash") if key in frame]
    audit = frame[audit_columns].copy()
    audit["family_first_date"] = group_start
    audit["required_net_return"] = np.maximum(label_config(config)["min_net_return"],
        label_config(config)["min_r_multiple"] * frame.risk_fraction)
    segments = {}
    for name, start, end, oracle in (
        ("train", None, fold["train_end"], original_masks[0]),
        ("calibration", fold["calibration_start"], fold["calibration_end"], original_masks[1]),
        ("policy", fold["policy_start"], fold["policy_end"], original_masks[2]),
    ):
        in_dates = dates.le(end).to_numpy() if start is None else dates.between(start, end).to_numpy()
        group_ok = group_start.le(end).to_numpy() if start is None else group_start.between(start, end).to_numpy()
        atomic_ok = np.ones(len(frame), dtype=bool) if start is None else np.asarray([
            bool(ids) and all(start <= first_dates[(symbol, atom)] <= end for atom in ids)
            for symbol, ids in atoms], dtype=bool)
        gates = [("label_unavailable", available), ("input_ineligible", input_ok),
            ("invalid_binary_label", labels.isin([0, 1]).to_numpy()), ("label_end_missing", ends.notna().to_numpy()),
            ("family_first_outside_segment", group_ok),
            ("label_end_after_cutoff", ends.le(end).fillna(False).to_numpy()),
            ("atomic_dependency_outside_segment", atomic_ok)]
        reasons = np.full(len(frame), "outside_decision_segment", dtype=object)
        remaining = in_dates.copy()
        stages = [{"stage": "decision_segment", **_summary(frame, remaining), "removed": 0}]
        losses = {}
        for reason, passes in gates:
            removed = remaining & ~passes
            reasons[removed] = reason
            losses[reason] = _summary(frame, removed)
            remaining &= passes
            stages.append({"stage": reason, **_summary(frame, remaining), "removed": int(removed.sum()),
                           "positive_removed": int((removed & labels.eq(1).to_numpy()).sum())})
        if not np.array_equal(remaining, oracle):
            raise ValueError("diagnostic selection differs from frozen training time_masks")
        reasons[remaining] = "selected"
        audit[name + "_reason"] = reasons
        audit[name + "_selected"] = remaining
        audit[name + "_weight"] = 0.
        components = economic_component_ids(frame, remaining, head="entry")
        inverse = 1. / components.map(components.value_counts()).to_numpy(float)
        weights = _weights(frame, remaining, head="entry") if remaining.any() else np.array([])
        audit.loc[remaining, name + "_weight"] = weights
        chosen_labels = labels.loc[remaining].to_numpy()
        sources = {}
        for source in ("ma", "structure", "wyckoff"):
            values = pd.to_numeric(frame.loc[in_dates, source + "_input_available"], errors="coerce")
            sources[source] = {"available": int(values.eq(1).sum()), "unavailable": int(values.ne(1).sum())}
        from my_strategy.services.czsc_trend_features import ENTRY_FEATURE_COLUMNS
        features = frame.loc[in_dates, list(ENTRY_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        finite = np.isfinite(features.to_numpy(float))
        segments[name] = {"start": start, "end": end, "funnel": stages, "exclusive_losses": losses,
            "mask_matches_training": True, "selected": _summary(frame, remaining),
            "economic_components": int(components.nunique()),
            "positive_normalized_weight": float(weights[chosen_labels == 1].sum()),
            "positive_inverse_component_mass": float(inverse[chosen_labels == 1].sum()),
            "target": _target_counts(frame, in_dates, config), "source_input_availability": sources,
            "source_membership": _counts(frame.loc[in_dates, "sources"]),
            "feature_observations": {"rows_with_missing_features": int((~finite).any(axis=1).sum()),
                "nonfinite_by_column": {column: int((~finite[:, i]).sum()) for i, column in enumerate(features) if (~finite[:, i]).any()},
                "scope": "observational_missing_features_imputed_by_existing_train_only_preprocessing_no_new_gate"},
            "reference_quality": {"allowed": int(frame.loc[in_dates, "reference_allowed"].fillna(False).sum()),
                "status": _counts(frame.loc[in_dates, "reference_gate_status"]),
                "reason": _counts(frame.loc[in_dates, "reference_gate_reason"]),
                "high_labels_retained_despite_reference_rejection": int((in_dates & labels.eq(1).to_numpy() &
                    ~frame.reference_allowed.fillna(False).to_numpy(bool)).sum()),
                "quality_rejection_removes_training_candidate": False}}
    train, calibration = segments["train"]["selected"], segments["calibration"]["selected"]
    row_gate = train["rows"] >= config.get("min_train_rows", 100) and calibration["rows"] >= config.get("min_calibration_rows", 30)
    both = all(item["positive"] > 0 and item["negative"] > 0 for item in (train, calibration))
    reason = None if row_gate and both else (f"insufficient mature entry rows train={train['rows']} calibration={calibration['rows']}"
        if not row_gate else "entry train/calibration require both classes")
    return {"fold": fold, "population_rows": len(frame), "segments": segments,
        "qualification": {"min_train_rows": config.get("min_train_rows", 100),
            "min_calibration_rows": config.get("min_calibration_rows", 30), "both_classes_required": True,
            "row_gate_passed": row_gate, "both_classes_passed": both, "passed": row_gate and both, "reason": reason}}, audit


def _read_verified(path, expected):
    data = Path(path).read_bytes()
    import hashlib
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("diagnostic input SHA256 mismatch: " + str(path))
    return json.loads(data)


def _ledger_values(ledger):
    buys = [row for row in ledger if row["action"] == "BUY"]
    if not buys:
        return {"ledger_filled": False}
    if len(buys) != 1:
        raise ValueError("diagnostic clone has multiple BUY fills")
    paid = -float(buys[0]["cash_flow"])
    gross, fees, slippage = 0., 0., 0.
    inventory = 0
    for row in ledger:
        is_buy = row["action"] == "BUY"
        shares, price, fee = int(row["shares"]), float(row["price"]), float(row["fee"])
        signed = -1 if is_buy else 1
        flow = signed * price * shares - fee
        if not math.isclose(flow, float(row["cash_flow"]), abs_tol=1e-7):
            raise ValueError("Broker cash flow does not reconcile with observed fills and fees")
        gross += signed * price * shares
        fees += fee
        inventory += shares if is_buy else -shares
        if inventory < 0:
            raise ValueError("clone inventory below zero")
        slip = float(row["applied_slippage"])
        base = price / (1 + slip if is_buy else 1 - slip)
        slippage += abs(price - base) * shares
    return {"ledger_filled": True, "ledger_complete": inventory == 0,
        "ledger_entry_cash": paid, "ledger_gross_profit_after_slippage": gross,
        "ledger_fees": fees, "ledger_net_profit": gross - fees,
        "ledger_net_return": (gross - fees) / paid if inventory == 0 else None,
        "observed_slippage_cash": slippage, "ledger_open_shares": inventory,
        "last_exit_reason": next((row.get("reason") for row in reversed(ledger) if row["action"] != "BUY"), None)}


def _inspect_symbol(payload):
    directory, record, fold, config, calendar = payload
    directory = Path(directory) / record["symbol"].replace(".", "_")
    assets = record["files"]
    for name in ("reference_account.json", "candidate_clones.json"):
        suffix = "/" + "/".join(directory.parts[directory.parts.index("runs") + 1:]) + "/" + name
        if not str(assets[name]["path"]).endswith(suffix):
            raise ValueError("frozen account path does not match its run-relative diagnostic asset")
    account = _read_verified(directory / "reference_account.json", assets["reference_account.json"]["sha256"])
    costs = account["cost_model"]
    diagnostics = {str(row["date"]): row for row in account["entry_diagnostics"]}
    daily = []
    for row in account["rows"]:
        day = str(row["date"])[:10]
        if day < config["feature_start"] or day > fold["policy_end"]:
            continue
        direction = row.get("direction_decision") or {}
        attempt = diagnostics.get(day)
        if not direction:
            reason = "no_direction_observation"
        elif not any(candidate.get("side") == "BUY" for candidate in direction.get("active_candidates", [])):
            reason = "no_active_buy_opportunity"
        elif not direction.get("buy_base_qualified"):
            reason = "buy_base_not_qualified"
        elif not direction.get("entry_eligible"):
            reason = "entry_envelope_not_eligible"
        elif not attempt:
            reason = "account_state_or_reservation_block"
        elif attempt.get("risk_qualified") is not True:
            reason = "risk_plan_rejected"
        else:
            reason = "pre_gate_attempt"
        # The actual recorded attempt is authoritative over display evidence.
        if attempt and attempt.get("risk_qualified") is True:
            reason = "pre_gate_attempt"
        value = {"symbol": record["symbol"], "date": day, "reason": reason,
            "reason_codes": json.dumps(direction.get("reason_codes") or row.get("reason_codes") or [], ensure_ascii=False),
            "risk_reason": (attempt or {}).get("reason"), "attempt_id": (attempt or {}).get("attempt_id"),
            "has_recorded_attempt": bool(attempt), "risk_qualified": (attempt or {}).get("risk_qualified") is True,
            "reference_allowed": (attempt or {}).get("allowed") is True}
        value["buy_base_score"] = direction.get("buy_base_score")
        value["sell_base_score"] = direction.get("sell_base_score")
        value["buy_independent_families"] = direction.get("buy_independent_families")
        plan = direction.get("raw_risk_plan") or direction.get("entry_plan") or {}
        value["risk_plan_reason_codes"] = json.dumps(plan.get("reason_codes") or [], ensure_ascii=False)
        for source in ("ma", "structure", "wyckoff"):
            agent = direction.get("agents", {}).get(source) or {}
            value[source + "_status"] = agent.get("status", "unspecified")
            value[source + "_reasons"] = json.dumps(agent.get("reason_codes") or [], ensure_ascii=False)
        daily.append(value)
    del account
    clones = _read_verified(directory / "candidate_clones.json", assets["candidate_clones.json"]["sha256"])
    from my_strategy.execution.cost_model import CostModel
    cost = CostModel(commission=costs["commission"], stamp_tax=costs["stamp_tax"],
        slippage=costs["slippage"], min_commission=costs["min_commission"], version="flat_v1", impact_tiers=())
    rows, rejections = [], []
    for clone in clones:
        ledger = clone["ledger"]
        outcome = episode_outcome(ledger, stop_price=clone["actual_initial_stop"], cost_model=cost, config=config)
        saved = clone["outcome"]
        for key in ("net_profit", "net_return", "risk_cash", "risk_fraction", "r_multiple", "entry_cash", "exit_cash", "fees"):
            a, b = saved.get(key), outcome.get(key)
            if a is None or not math.isfinite(float(a)):
                continue
            if b is None or not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-7):
                raise ValueError("saved clone outcome cannot be recomputed: " + key)
        if saved.get("label_available") and (saved["label"] != outcome["label"] or saved["label_end"] != outcome["label_end"]):
            raise ValueError("saved clone label differs from unchanged episode_outcome")
        observed_sessions = (calendar.index(outcome["exit_date"]) - calendar.index(outcome["entry_date"])) if outcome.get("exit_date") else None
        rows.append({"symbol": record["symbol"], "attempt_id": clone["id"],
            "recomputed_label": outcome["label"] if outcome["label_available"] else None,
            "recomputed_label_available": outcome["label_available"],
            "recomputed_label_end": outcome["label_end"], "recomputed_ledger_hash": stable_hash(ledger),
            "recomputed_risk_cash": outcome["risk_cash"], "recomputed_risk_fraction": outcome["risk_fraction"],
            "recomputed_r_multiple": outcome["r_multiple"],
            "saved_outcome_recomputed": True, "verified_holding_session_distance": observed_sessions,
            **_ledger_values(ledger)})
        rejections.extend({"symbol": record["symbol"], "attempt_id": clone["id"],
                           "signal_date": clone["signal_date"], "action": rejection.get("action"),
                           "date": rejection.get("date"), "reason": rejection.get("reason", "unspecified")}
                          for rejection in clone.get("rejections", []))
    return daily, rows, rejections


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def verify_pooled_outcomes(audit):
    """Cross-check the fields used for target and maturity against the clone."""
    checked = audit.get("saved_outcome_recomputed", pd.Series(False, index=audit.index)).fillna(False)
    for key in ("risk_cash", "risk_fraction", "r_multiple"):
        saved = pd.to_numeric(audit.loc[checked, key], errors="coerce").to_numpy(float)
        actual = pd.to_numeric(audit.loc[checked, "recomputed_" + key], errors="coerce").to_numpy(float)
        if not np.allclose(saved, actual, rtol=1e-9, atol=1e-7, equal_nan=True):
            raise ValueError("pooled outcome differs from recomputed clone: " + key)
    saved_ends = pd.to_datetime(audit.loc[checked, "label_end"], errors="coerce")
    actual_ends = pd.to_datetime(audit.loc[checked, "recomputed_label_end"], errors="coerce")
    if not (saved_ends.eq(actual_ends) | (saved_ends.isna() & actual_ends.isna())).all():
        raise ValueError("pooled outcome differs from recomputed clone: label_end")


def run_diagnostics(model_run_id, *, cpu_workers=2, progress=None):
    if not 1 <= cpu_workers <= 4:
        raise ValueError("diagnostic CPU workers must be between 1 and 4")
    from my_strategy.services.czsc_trend_pipeline import read_run
    root, report = read_run(model_run_id)
    config = report["config"]
    calendar_path = root / "data_snapshots/calendar.json"
    calendar_record = json.loads(calendar_path.read_text())
    calendar = calendar_record["dates"]
    if (calendar_record.get("verified") is not True or calendar != sorted(set(calendar)) or
            stable_hash(calendar) != report["source_snapshot"]["source_snapshot"]["calendar_hash"]):
        raise ValueError("diagnostic verified calendar differs from frozen source")
    context = create_run_context(task="high-return-sample-diagnostics", as_of_date=report["data_end"],
        config={"version": VERSION, "model_run_id": model_run_id}, scope="implementation",
        data_version=config["source_snapshot_hash"], stocks=[r["symbol"] for r in report["dataset_records"]])
    context.write_metadata({"status": "running", "source_model_run_id": model_run_id})
    output = context.subdir("reports")
    result = {"version": VERSION, "run_id": context.run_id, "model_run_id": model_run_id,
        "family": report["strategy_version"], "data_end": report["data_end"], "cohort_size": len(report["dataset_records"]),
        "source_report_sha256": _file_hash(root / "reports/trend_consensus.json"),
        "protocol_freeze_hash": report["protocol_freeze"]["freeze_hash"], "source_snapshot_hash": config["source_snapshot_hash"],
        "source_observations": {key: report["source_snapshot"].get(key) for key in
            ("price_basis", "price_basis_counts", "legacy_fuyao_bars", "unverified_trade_price_bars", "scope")},
        "rules": {"high_return": config["high_return"], "consensus": config["consensus"],
            "split_policy": config["split_policy"], "economic_weighting": config["economic_weighting"]},
        "scope": "existing_frozen_entry_calibration_populations_no_retraining_or_rule_changes",
        "fold_population_counts_are_additive": False, "publication_allowed": False,
        "folds": [], "input_assets": [{"path": str(calendar_path), "sha256": _file_hash(calendar_path)}], "output_assets": [], "limitations": [
            "未复权研究股票池；本诊断不取得收益、历史ST或生产认证。",
            "不同折人口有重叠，逐日重评也有共享经济机会，不可合并成独立样本数。",
            "来源价基计数沿用冻结manifest；未重新扫描全量原始库。",
            "滑点已包含在成交价；反解金额仅为费用诊断，不构造新标签或新交易。"]}
    try:
        for population in report["population_folds"]:
            fold = population["reference_population"]["fold"]
            directory = root / "dataset/final_population" / fold["name"]
            manifest = _read_verified(directory / "manifest.json", population["manifest_sha256"])
            expected_symbols = {record["symbol"] for record in report["dataset_records"]}
            records = manifest["records"]
            if manifest.get("failures") or {row["symbol"] for row in records} != expected_symbols or len(records) != len(expected_symbols):
                raise ValueError("diagnostic population cohort incomplete")
            if manifest["reference_population"] != population["reference_population"]:
                raise ValueError("diagnostic reference population mismatch")
            entries = directory / "entries.parquet"
            expected_sha = manifest["pooled_files"]["entries"]["sha256"]
            if _file_hash(entries) != expected_sha:
                raise ValueError("pooled entry samples changed")
            frame = pd.read_parquet(entries)
            if frame.duplicated(["symbol", "attempt_id"]).any():
                raise ValueError("duplicate entry attempt identity")
            diagnostic, audit = diagnose_entry_frame(frame, fold, config)
            diagnostic["stored_unavailable_heads"] = [item for item in report["unavailable_heads"]
                if item["head"] == "entry" and item["fold"] == fold["name"]]
            for item in diagnostic["stored_unavailable_heads"]:
                if item["reason"] != diagnostic["qualification"]["reason"]:
                    raise ValueError("diagnostic qualification differs from stored training failure")
            if diagnostic["qualification"]["passed"] and not any(bundle["name"] == fold["name"] and
                    bundle.get("heads", {}).get("entry") for bundle in report["architecture_bundles"]):
                raise ValueError("stored model catalogue lacks an entry head that passes sample gates")
            result["input_assets"].extend([{"path": str(directory / "manifest.json"), "sha256": population["manifest_sha256"]},
                                           {"path": str(entries), "sha256": expected_sha}])
            payloads = [(str(directory), record, fold, config, calendar) for record in records]
            daily_rows, clone_rows, rejected = [], [], []
            with ProcessPoolExecutor(max_workers=cpu_workers) as pool:
                futures = [pool.submit(_inspect_symbol, payload) for payload in payloads]
                for i, future in enumerate(as_completed(futures), 1):
                    daily, clones, rejections = future.result()
                    daily_rows.extend(daily); clone_rows.extend(clones); rejected.extend(rejections)
                    if progress and (i % 20 == 0 or i == len(records)):
                        progress(fold["name"], i, len(records))
            daily = pd.DataFrame(daily_rows).sort_values(["symbol", "date"])
            clones = pd.DataFrame(clone_rows)
            audit = audit.merge(clones, on=["symbol", "attempt_id"], how="left", validate="one_to_one") if len(clones) else audit
            if len(clones) and (~audit.loc[audit.recomputed_ledger_hash.notna(), "clone_ledger_hash"].eq(
                    audit.loc[audit.recomputed_ledger_hash.notna(), "recomputed_ledger_hash"])).any():
                raise ValueError("pooled entry clone ledger identity mismatch")
            checked = audit.get("saved_outcome_recomputed", pd.Series(False, index=audit.index)).fillna(False)
            # A last source day has no future session and therefore no clone.
            missing = ~checked & audit.label_reason.ne("next_session_not_observed")
            if missing.any():
                raise ValueError("candidate clone coverage incomplete")
            verify_pooled_outcomes(audit)
            for key, ledger_key in (("net_return", "ledger_net_return"), ("net_profit", "ledger_net_profit"),
                                    ("entry_cash", "ledger_entry_cash"), ("fees", "ledger_fees")):
                actual = pd.to_numeric(audit.loc[checked, ledger_key], errors="coerce")
                saved = pd.to_numeric(audit.loc[checked, key], errors="coerce")
                compare = np.isfinite(saved) & np.isfinite(actual)
                if not np.allclose(saved[compare], actual[compare], rtol=1e-9, atol=1e-7):
                    raise ValueError("pooled outcome differs from recomputed ledger: " + key)
            valid_labels = checked & audit.label_available.fillna(False)
            if (audit.loc[valid_labels, "label"].ne(audit.loc[valid_labels, "recomputed_label"])).any():
                raise ValueError("pooled high-return label differs from recomputed outcome")
            for record in records:
                for name in ("reference_account.json", "candidate_clones.json"):
                    result["input_assets"].append({"path": str(directory / record["symbol"].replace(".", "_") / name),
                        "sha256": record["files"][name]["sha256"]})
            diagnostic["daily_candidate_path"] = {}
            for name, segment in diagnostic["segments"].items():
                scope = daily.date.le(segment["end"]) if segment["start"] is None else daily.date.between(segment["start"], segment["end"])
                part = daily.loc[scope]
                diagnostic["daily_candidate_path"][name] = {"observed_stock_sessions": len(part),
                    "exclusive_first_block": _counts(part.reason),
                    "direction_reason_codes": _counts(part.reason_codes),
                    "risk_plan_reason_codes": _counts(part.risk_plan_reason_codes),
                    "source_status": {source: _counts(part[source + "_status"]) for source in ("ma", "structure", "wyckoff")}}
                if int(part.reason.eq("pre_gate_attempt").sum()) != segment["target"]["candidates"]:
                    raise ValueError("recorded risk-qualified daily attempts do not reconcile with entry samples")
                scoped = audit.date.le(segment["end"]) if segment["start"] is None else audit.date.between(segment["start"], segment["end"])
                selected = audit.loc[scoped]
                mature = selected.loc[selected.label_available.fillna(False)]
                added_slippage = pd.to_numeric(mature.net_profit) + pd.to_numeric(mature.fees) + pd.to_numeric(mature.observed_slippage_cash)
                settings = label_config(config)
                fee_and_slippage_excluded = (added_slippage.div(mature.entry_cash).ge(settings["min_net_return"]) &
                    added_slippage.div(mature.risk_cash).ge(settings["min_r_multiple"]))
                segment["ledger_audit"] = {"verified_clones": int(selected.get("saved_outcome_recomputed", pd.Series(dtype=bool)).fillna(False).sum()),
                    "last_exit_reasons": _counts(mature.get("last_exit_reason", pd.Series(dtype=str))),
                    "observed_slippage_cash": float(selected.get("observed_slippage_cash", pd.Series(dtype=float)).sum()),
                    "holding_session_distance_quantiles": _quantiles(mature.verified_holding_session_distance),
                    "fee_and_slippage_excluded_same_denominator_high": int(fee_and_slippage_excluded.sum()),
                    "cost_comparison_scope": "observed_slippage_reverse_calculated_same_frozen_denominators_not_alternative_trade_labels",
                    "rejection_actions_and_reasons": dict(Counter(str(row['action'])+":"+str(row['reason']) for row in rejected
                        if (not segment["start"] or row["signal_date"] >= segment["start"]) and row["signal_date"] <= segment["end"]))}
            diagnostic["clone_rejections"] = dict(Counter(row["reason"] for row in rejected))
            diagnostic["integrity"] = {"input_SHA256_verified": True, "ledger_and_outcome_recomputed": True,
                "time_masks_match": True, "daily_attempts_reconciled": True,
                "pooled_risk_R_and_maturity_match_recomputed_clones": True,
                "absolute_frozen_paths_relocated_by_exact_run_relative_suffix": True}
            for name, value in (("samples", audit), ("daily_candidate_path", daily), ("clone_rejections", pd.DataFrame(rejected))):
                path = context.subdir("diagnostics", fold["name"]) / (name + ".parquet")
                value.to_parquet(path, index=False)
                result["output_assets"].append({"path": str(path), "sha256": _file_hash(path), "rows": len(value)})
            result["folds"].append(diagnostic)
        result["artifact_integrity_passed"] = True
        result["source_report_unchanged"] = _file_hash(root / "reports/trend_consensus.json") == result["source_report_sha256"]
        if not result["source_report_unchanged"]:
            raise ValueError("source report changed during diagnosis")
        result["report_paths"] = {"json": str(output / "high_return_sample_diagnostics.json"),
                                  "markdown": str(output / "high_return_sample_diagnostics.md")}
        add_candidate_details(result, calendar)
        result["diagnostic_code_sha256"] = _file_hash(Path(__file__))
        Path(result["report_paths"]["markdown"]).write_text(render_markdown(result), encoding="utf-8")
        _save_json(result["report_paths"]["json"], result)
        context.write_metadata({"status": "completed", "source_model_run_id": model_run_id,
            "report_paths": result["report_paths"], "publication_allowed": False})
        return result
    except Exception as exc:
        context.write_metadata({"status": "failed", "source_model_run_id": model_run_id, "reason": str(exc)})
        raise


def add_candidate_details(report, calendar):
    """Derive inspectable gate details from the saved daily/sample audit rows."""
    settings = report["rules"]["consensus"]
    market_index = {day: i for i, day in enumerate(calendar)}
    for item in report["folds"]:
        prefix = Path(report["report_paths"]["json"]).parents[1] / "diagnostics" / item["fold"]["name"]
        daily = pd.read_parquet(prefix / "daily_candidate_path.parquet")
        samples = pd.read_parquet(prefix / "samples.parquet")
        for name, segment in item["segments"].items():
            in_days = daily.date.le(segment["end"]) if segment["start"] is None else daily.date.between(segment["start"], segment["end"])
            base = daily.loc[in_days & daily.reason.eq("buy_base_not_qualified")]
            support = pd.to_numeric(base.buy_base_score, errors="coerce") + 1e-12 < settings["buy_base_threshold"]
            independent = base.buy_independent_families.map(lambda values: len(values) if isinstance(values, (list, np.ndarray)) else 0) < settings["min_independent"]
            margin = (pd.to_numeric(base.buy_base_score, errors="coerce") - pd.to_numeric(base.sell_base_score, errors="coerce") + 1e-12 < settings["margin"])
            first = pd.Series("buy_base_not_qualified", index=base.index)
            first.loc[margin] = "buy_sell_margin_insufficient"
            first.loc[independent] = "independent_sources_insufficient"
            first.loc[support] = "buy_base_below_threshold"
            path = item["daily_candidate_path"][name]
            path["base_gate_exclusive_details"] = _counts(first)
            path["base_gate_overlapping_failures"] = {"buy_base_below_threshold": int(support.sum()),
                "independent_sources_insufficient": int(independent.sum()), "buy_sell_margin_insufficient": int(margin.sum())}
            path["base_details_order"] = ["buy_base_below_threshold", "independent_sources_insufficient", "buy_sell_margin_insufficient"]
            path["entry_envelope_risk_reasons"] = _counts(daily.loc[in_days & daily.reason.eq("entry_envelope_not_eligible"), "risk_plan_reason_codes"])
            scope = samples.date.le(segment["end"]) if segment["start"] is None else samples.date.between(segment["start"], segment["end"])
            mature = samples.loc[scope & samples.label_available.fillna(False)]
            distance = [market_index[end] - market_index[start] for start, end in zip(mature.entry_date, mature.exit_date)]
            segment["ledger_audit"]["holding_session_distance_quantiles"] = _quantiles(distance)
            segment["ledger_audit"]["holding_session_distance_mean"] = float(np.mean(distance)) if distance else None
            segment["ledger_audit"]["holding_duration_scope"] = "verified_calendar_exit_index_minus_entry_index"
            segment["ledger_audit"]["last_exit_reasons"] = _counts(mature.last_exit_reason)
            segment["target"]["availability_scope"] = "decisions_in_segment_complete_by_frozen_source_not_segment_cutoff"
            segment["near_return_target_samples"] = mature.loc[mature.net_return.ge(report["rules"]["high_return"]["min_net_return"]),
                ["symbol", "date", "label_end", "net_return", "r_multiple", "risk_fraction", "required_net_return",
                 name + "_reason", "last_exit_reason"]].head(20).to_dict("records")
        cal = item["segments"]["calibration"]
        item["diagnosis"] = {"calibration_positive_before_input_and_split_gates": cal["target"]["both_target_pass"],
            "calibration_positive_after_gates": cal["selected"]["positive"],
            "primary_observed_cause": "frozen_outcomes_do_not_meet_joint_return_and_R_target" if
                cal["target"]["available_labels"] and not cal["target"]["both_target_pass"] else
                "input_maturity_or_dependency_gates_remove_positives" if not cal["selected"]["positive"] else
                "positive_class_present_check_row_gate_and_economic_dependence",
            "quality_gate_directly_removes_label_clones": False,
            "cost_only_positive_loss": cal["target"]["fee_only_high_to_negative"],
            "interpretation_limit": "observed_population_not_proof_of_exit_model_error_or_all_raw_data_quality"}
    return report


def render_markdown(report):
    lines = ["# 高收益校准正类流失诊断", "", f"模型运行：`{report['model_run_id']}`",
        f"诊断运行：`{report['run_id']}`；冻结行情截止 **{report['data_end']}**；股票池 **{report['cohort_size']}** 股。", "",
        "本结果读取冻结人口和真实Broker克隆账本；不训练、不改变交易或模型资格。每折人口独立呈现，不能相加。", "",
        "有效标签及目标计数均指该段信号截至冻结行情源已完成的结果；在校准截止前成熟另见时间清洗表。", "",
        "| 折 | 校准期候选 | 有效标签 | 费用后盈利 | ≥净收益目标 | ≥R目标 | 同时达标 | 实际校准正/负 | 训练正/负 | 资格 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for item in report["folds"]:
        cal = item["segments"]["calibration"]; target = cal["target"]; train = item["segments"]["train"]["selected"]
        chosen = cal["selected"]
        lines.append(f"| {item['fold']['name']} | {target['candidates']} | {target['available_labels']} | {target['fee_net_profitable']} | "
            f"{target['net_return_target_pass']} | {target['r_multiple_target_pass']} | {target['both_target_pass']} | "
            f"{chosen['positive']}/{chosen['negative']} | {train['positive']}/{train['negative']} | {'通过样本门槛' if item['qualification']['passed'] else item['qualification']['reason']} |")
    for item in report["folds"]:
        cal = item["segments"]["calibration"]; target = cal["target"]
        lines.extend(["", "## " + item["fold"]["name"], "",
            f"校准：{cal['start']}—{cal['end']}。有效标签中净收益和R同时达标 {target['both_target_pass']}；"
            f"净收益达标但R不足 {target['target_quadrants']['return_only']}；仅R达标 {target['target_quadrants']['r_only']}。", "",
            "| 时间和输入清洗（按表顺序归因） | 保留行 | 保留正类 | 本步排除 |", "|---|---:|---:|---:|"])
        for stage in cal["funnel"]:
            lines.append(f"| {REASON_LABELS.get(stage['stage'],stage['stage'])} | {stage['rows']} | {stage['positive']} | {stage['removed']} |")
        lines.extend(["", "| 股票日候选路径（互斥首个阻断） | 数量 |", "|---|---:|"])
        for reason, count in item["daily_candidate_path"]["calibration"]["exclusive_first_block"].items():
            lines.append(f"| {REASON_LABELS.get(reason,reason)} | {count} |")
        details = item["daily_candidate_path"]["calibration"]
        lines.extend(["", "基础共识首次失败细分：" + "、".join(
            f"{REASON_LABELS.get(reason,reason)} {count}" for reason, count in details.get("base_gate_exclusive_details", {}).items()) + "。",
            "基础门槛重叠失败（不能相加）：`" + json.dumps(details.get("base_gate_overlapping_failures", {}), ensure_ascii=False) + "`。",
            "入场风险原因：`" + json.dumps(details.get("entry_envelope_risk_reasons", {}), ensure_ascii=False) + "`。"])
        lines.extend(["", "| 标签与经济目标（互斥首个原因） | 数量 |", "|---|---:|"])
        for reason, count in target["exclusive_target_reasons"].items():
            lines.append(f"| {REASON_LABELS.get(reason,reason)} | {count} |")
        lines.extend(["", f"费用：有效标签合计 {target['fees_total']:.2f} 元；保持已滑点成交价与原冻结分母，仅加回费用后，"
            f"同时达标 {target['fee_excluded_same_denominator_high']} 个，费用边缘损失 {target['fee_only_high_to_negative']} 个。这个对照不替代正式标签。",
            f"加回费用与观测滑点后、保留原冻结分母，同时达标 {cal['ledger_audit']['fee_and_slippage_excluded_same_denominator_high']} 个。",
            f"入选校准人口对应 {cal['economic_components']} 个经济连通分量；正类逆分量质量 {cal['positive_inverse_component_mass']:.6f}。",
            f"参考账户质量状态：`{json.dumps(cal['reference_quality']['reason'],ensure_ascii=False)}`。质量拒绝仍保留强制入场克隆，"
            f"没有直接删除这些训练候选；拒绝后仍保留的高收益标签 {cal['reference_quality']['high_labels_retained_despite_reference_rejection']}。",
            f"缺列观测：{cal['feature_observations']['rows_with_missing_features']} 行有缺失特征，沿用训练段拟合的填补，不能误称为资格排除。"])
        if target['available_labels']:
            lines.extend(["", f"有效标签的初始风险中位数 **{target['risk_fraction_quantiles']['0.5']:.2%}**，联合目标所需净收益中位数 "
                f"**{target['required_net_return_quantiles']['0.5']:.2%}**；实际净收益最大 **{target['net_return_quantiles']['1']:.2%}**，"
                f"R最大 **{target['r_multiple_quantiles']['1']:.3f}**。",
                f"实际清仓原因：`{json.dumps(cal['ledger_audit']['last_exit_reasons'],ensure_ascii=False)}`；核验持仓交易日距离"
                f"中位 {cal['ledger_audit']['holding_session_distance_quantiles'].get('0.5')}，平均 {cal['ledger_audit'].get('holding_session_distance_mean',0):.2f}。",
                "这些结果解释联合目标在当前冻结续持政策下很难达成，不能据此断言退出头或全部行情数据有错误。"])
        if cal.get('near_return_target_samples'):
            lines.extend(["", "| 净收益达标样本 | 信号日 | 成熟日 | 净收益 | R | 初始风险 | 所需净收益 | 校准资格 |", "|---|---|---|---:|---:|---:|---:|---|"])
            for row in cal['near_return_target_samples']:
                lines.append(f"| {row['symbol']} | {row['date']} | {row['label_end']} | {row['net_return']:.2%} | {row['r_multiple']:.3f} | "
                    f"{row['risk_fraction']:.2%} | {row['required_net_return']:.2%} | {REASON_LABELS.get(row['calibration_reason'],row['calibration_reason'])} |")
        winners = [stage["positive_removed"] for stage in cal["funnel"][1:]]
        if target["both_target_pass"] == 0:
            lines.extend(["", "**定位：校准日期内的有效标签在时间清洗前已无高收益正类，主要原因是冻结净收益／初始风险目标。"
                "输入和跨段门槛会减少行数，但本折没有可供它们删除的高收益正类。**"])
        elif any(winners):
            lines.extend(["", "**定位：校准日期内有高收益正类；表中的“保留正类”给出跨段与成熟清洗的正类损失。**"])
    lines.extend(["", "## 核验与口径", "",
        "JSON包含全部折的train/calibration/policy明细、来源可用性、特征缺失、收益/R/风险分位数、冻结门槛和输入SHA256。",
        "每折samples.parquet逐行给出attempt_id、label_end、费用净收益、R、分段互斥原因和真实训练权重；"
        "daily_candidate_path.parquet给出股票日首个阻断；clone_rejections.parquet保留Broker拒单原因。",
        "所有读取的账户及克隆文件校验SHA256；账本现金流、费用、初始风险与有效标签复算；最终mask逐行等于原time_masks，"
        "资格错误信息与冻结训练报告比对。原绝对目录已迁移，仅从当前模型run内固定相对位置读取并校验原SHA，不改写冻结manifest。", "",
        "输入资格只要求原候选所属来源中至少一项可用，不要求每一列都有限；缺失观测不新增过滤。", "",
        *["- " + text for text in report["limitations"]], "",
        f"机器结果：[JSON]({report['report_paths']['json']})"])
    return "\n".join(lines) + "\n"
