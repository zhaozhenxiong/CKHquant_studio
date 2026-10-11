"""Independent post-training diagnostics; never imported by the frozen strategy.

Templates describe mature training outcomes, not tradable rules or a certificate.
Calibration and policy probabilities are diagnostic reconstructions, not forecasts
available before the actual training completion clock. No model is fitted here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_high_return_features import (
    ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS, HOLDING_COLUMNS, SOURCE_COLUMNS,
)
from my_strategy.services.czsc_high_return_models import _group_ids, load_head, time_masks

VERSION = "independent_causal_operation_diagnostics_v1"
TEMPLATE_ENTRY_COLUMNS = (
    "ma20_bias", "ma60_bias", "ma20_slope5", "ma_alignment", "ret_5", "ret_20",
    "atr14_ratio", "drawdown20", "bi1_return", "zone_low_distance", "zone_width_ratio",
    "weekly_return", "w_relative_volume20", "w_range_position", "w_support_distance_atr",
    "w_demand_score", "w_supply_score", "source_count", "ma_buy_event_age",
    "structure_buy_event_age", "wyckoff_buy_event_age", "l1_ma_entry_probability",
    "candidate_stop_distance", "candidate_target_weight", *SOURCE_COLUMNS,
)
TEMPLATE_HOLDING_COLUMNS = (*TEMPLATE_ENTRY_COLUMNS, *HOLDING_COLUMNS)
CLASSES = ("high", "ordinary", "loss")
RETROSPECTIVE_COLUMNS = (
    "max_favorable_excursion", "max_adverse_excursion", "final_peak_giveback",
    "net_return", "r_multiple", "holding_sessions", "entry_fee_ratio",
)
# Registered domain combinations supplement the univariate comparisons. They
# are all reported; no winner, threshold or production action is selected.
OPERATION_TEMPLATES = (
    ("trend_with_demand", (("ma_alignment", "gt", 0.), ("ma20_slope5", "gt", "median"),
                           ("w_demand_score", "gt", "median"))),
    ("support_with_demand", (("ma20_bias", "le", "median"), ("w_support_distance_atr", "le", "median"),
                             ("w_demand_score", "gt", "median"))),
    ("volume_without_high_supply", (("w_supply_score", "le", "median"),
                                    ("w_relative_volume20", "gt", "median"), ("ret_5", "gt", 0.))),
    ("structure_support_and_risk", (("zone_low_distance", "le", "median"),
                                   ("structure_buy_event_age", "le", 5.),
                                   ("candidate_stop_distance", "le", "median"))),
    ("multiple_sources_with_trend", (("source_count", "gt", 1.), ("ma20_slope5", "gt", 0.),
                                    ("w_demand_score", "gt", "median"))),
    ("held_profit_giveback_and_supply", (("holding_return", "gt", 0.),
                                        ("peak_giveback", "gt", "median"),
                                        ("w_supply_score", "gt", "median"))),
    ("held_weakness_near_stop", (("running_close_drawdown", "gt", "median"),
                               ("stop_distance", "le", "median"), ("w_supply_score", "gt", "median"))),
)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("diagnostic outputs are immutable; create a new RunContext")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _dates(values):
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def _mature(frame, cutoff, *, separate_sources=False):
    dates, ends = _dates(frame.date), _dates(frame.label_end)
    valid = frame.label_available.fillna(False).to_numpy(dtype=bool, copy=True)
    valid &= frame.get("input_eligible", pd.Series(True, index=frame.index)).fillna(False).to_numpy(bool)
    valid &= (dates.notna() & ends.notna() & (dates <= cutoff) & (ends <= cutoff) & (ends > dates)).to_numpy()
    valid &= frame.outcome_class.isin(CLASSES).to_numpy()
    # Every state of a repeated economic episode must be mature and in scope.
    groups = _group_ids(frame)
    if separate_sources:
        # Four reference policies are separate accounting paths. A future or
        # unfinished structure interval must not censor a mature MA interval.
        groups = frame.source.astype(str) + ":" + groups
    valid &= pd.Series(valid, index=frame.index).groupby(groups).transform("all").to_numpy(bool)
    return valid


def _group_weights(frame):
    groups = _group_ids(frame)
    weights = 1. / groups.map(groups.value_counts()).to_numpy(float)
    return groups, weights


def _class_summary(frame, mask, groups, weights):
    selected = frame.loc[mask]
    w = weights[mask]
    total = float(w.sum())
    labels = selected.outcome_class.to_numpy()
    counts = {name: int((labels == name).sum()) for name in CLASSES}
    masses = {name: float(w[labels == name].sum()) for name in CLASSES}
    shares = {name: mass / total if total else None for name, mass in masses.items()}
    sources = {name.removeprefix("source_"): float(np.dot(w, selected[name].fillna(0).to_numpy(float) > 0) / total)
               if total else None for name in SOURCE_COLUMNS if name in selected}
    return {"rows": int(mask.sum()), "economic_groups": int(groups.loc[mask].nunique()),
            "economic_group_weight_mass": total, "class_rows": counts, "weighted_class_mass": masses,
            "weighted_class_share": shares, "source_proportions_nonexclusive": sources}


def discover_causal_templates(frame, *, columns, cutoff="2026-06-30",
                             min_template_groups=30, min_block_groups=10, population="entry"):
    """Training-only frozen quantile templates; stability is descriptive only.

    Entire economic episodes belong to their first observed decision's quarter.
    Thresholds use all eligible production training data. Thus earlier blocks are
    retrospective training stability checks, never unseen validation evidence.
    """
    columns = tuple(columns)
    if population not in {"entry", "holding"}:
        raise ValueError("unknown diagnostic decision population")
    approved = ENTRY_FEATURE_COLUMNS if population == "entry" else EXIT_FEATURE_COLUMNS
    if not columns or len(set(columns)) != len(columns) or not set(columns) <= set(approved):
        raise ValueError("templates require approved causal input columns only")
    if not set(columns).issubset(frame):
        raise ValueError("template input columns missing")
    eligible = _mature(frame, cutoff)
    selected = frame.loc[eligible].copy().reset_index(drop=True)
    groups, weights = _group_weights(selected)
    mask = np.ones(len(selected), bool)
    baseline = _class_summary(selected, mask, groups, weights)
    dates = _dates(selected.date)
    first = dates.groupby(groups).transform("min")
    blocks = pd.to_datetime(first).dt.to_period("Q").astype(str)
    block_masks = {name: blocks.eq(name).to_numpy() for name in sorted(blocks.unique())}
    block_baselines = {name: _class_summary(selected, value, groups, weights) for name, value in block_masks.items()}
    templates = []
    medians = {}
    def append_template(conditions, active, family):
        if not active.any():
            return
        summary = _class_summary(selected, active, groups, weights)
        historical = []
        for name, block in block_masks.items():
            current = _class_summary(selected, active & block, groups, weights)
            base_high = block_baselines[name]["weighted_class_share"]["high"]
            actual_high = current["weighted_class_share"]["high"]
            enough = current["economic_groups"] >= min_block_groups
            historical.append({"block": name, **current, "enough_groups": enough,
                "high_share_minus_same_block_baseline": actual_high - base_high if actual_high is not None else None})
        usable = [x for x in historical if x["enough_groups"]]
        delta = summary["weighted_class_share"]["high"] - baseline["weighted_class_share"]["high"]
        signs = [x["high_share_minus_same_block_baseline"] > 0 for x in usable]
        sufficient = summary["economic_groups"] >= min_template_groups and len(usable) >= 4
        status = "insufficient_temporal_support" if not sufficient else (
            "consistent_positive_descriptive_only" if delta > 0 and all(signs) else "not_consistently_positive")
        templates.append({"template_id": stable_hash({"version": VERSION, "conditions": conditions,
            "family": family, "cutoff": cutoff}), "family": family, "conditions": conditions,
            **summary, "high_share_minus_training_baseline": delta,
            "time_blocks": historical, "stability_status": status,
            "sufficient_blocks": len(usable), "positive_blocks": sum(signs),
            "threshold_fit_scope": "eligible_mature_production_training_only",
            "tradable_rule": False, "statistical_pass": False})
    for column in columns:
        values = pd.to_numeric(selected[column], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        # Quantiles fitted with equal economic-episode mass, not row counts.
        order = np.argsort(values[finite], kind="stable")
        x, w = values[finite][order], weights[finite][order]
        cdf = np.cumsum(w) / w.sum()
        medians[column] = float(x[min(np.searchsorted(cdf, .5), len(x)-1)])
        thresholds = sorted({float(x[min(np.searchsorted(cdf, q), len(x)-1)]) for q in (.2, .5, .8)})
        for threshold in thresholds:
            for operator, active in (("le", finite & (values <= threshold)), ("gt", finite & (values > threshold))):
                append_template([{"column": column, "operator": operator, "value": threshold}], active, "univariate")
    for family, definition in OPERATION_TEMPLATES:
        if not all(column in columns and column in medians for column, _, _ in definition):
            continue
        active = np.ones(len(selected), bool)
        conditions = []
        for column, operator, threshold in definition:
            threshold = medians[column] if threshold == "median" else threshold
            values = pd.to_numeric(selected[column], errors="coerce").to_numpy(float)
            active &= np.isfinite(values) & ((values <= threshold) if operator == "le" else (values > threshold))
            conditions.append({"column": column, "operator": operator, "value": threshold})
        append_template(conditions, active, family)
    return {"version": VERSION, "cutoff": cutoff, "population_rows": len(frame),
            "decision_population": population,
            "eligible_rows": len(selected), "excluded_rows": int((~eligible).sum()),
            "baseline": baseline, "time_block_baselines": block_baselines, "templates": templates,
            "minimum_template_groups": min_template_groups, "minimum_block_groups": min_block_groups,
            "minimum_temporal_blocks": 4, "quantiles": [.2, .5, .8],
            "feature_columns": list(columns), "feature_columns_hash": stable_hash(columns),
            "operation_template_definitions": OPERATION_TEMPLATES,
            "weighting": "one_total_mass_per_economic_episode_before_template_filter",
            "interpretation": "retrospective_training_association_not_out_of_sample_or_causal_effect",
            "selection_or_prediction_X_modified": False, "publication_allowed": False}


def weighted_probability_diagnostics(frame, probabilities, *, head, segment, fold,
                                     thresholds=(.05, .1, .2, .35, .5, .55, .65)):
    """Exact tied-score weighted AP and PR, without fitting or threshold choice."""
    if segment not in {"calibration", "policy"}:
        raise ValueError("only independent calibration/policy segments are allowed")
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or len(values) != len(frame):
        raise ValueError("score rows do not match diagnostic inputs")
    regression = head in {"return", "exit_advantage"}
    masks = time_masks(frame, fold, regression=regression)
    allowed = masks[1 if segment == "calibration" else 2]
    target = frame.loc[allowed].reset_index(drop=True)
    score = values[allowed]
    groups, weights = _group_weights(target)
    finite = np.isfinite(score)
    if not regression and ((score[finite] < 0).any() or (score[finite] > 1).any()):
        raise ValueError("binary diagnostic scores must be probabilities")
    result = {"head": head, "segment": segment, "fold": fold, "eligible_rows": len(target),
        "eligible_economic_groups": int(groups.nunique()), "scored_rows": int(finite.sum()),
        "unavailable_score_rows": int((~finite).sum()),
        "scored_group_weight_mass": float(weights[finite].sum()),
        "score_coverage": float(weights[finite].sum() / weights.sum()) if weights.sum() else None,
        "weighting": "one_total_mass_per_economic_episode_in_entire_eligible_segment",
        "calibration_in_sample_for_calibrator": segment == "calibration",
        "forecast_identity": "historical_diagnostic_reconstruction_not_contemporaneous_forecast",
        "threshold_selection_performed": False, "statistical_pass": False, "publication_allowed": False}
    if not finite.any():
        return {**result, "status": "unavailable_no_mature_scored_rows", "average_precision": None, "pr_curve": []}
    y = pd.to_numeric(target.label).to_numpy(float)[finite]
    p, w = score[finite], weights[finite]
    total = float(w.sum())
    if regression:
        return {**result, "status": "continuous_head_diagnostic", "average_precision": None,
            "weighted_MSE": float(np.dot(w, (p-y)**2) / total),
            "weighted_MAE": float(np.dot(w, np.abs(p-y)) / total)}
    positive, negative = float(w[y == 1].sum()), float(w[y == 0].sum())
    order = np.argsort(-p, kind="stable")
    sp, sy, sw = p[order], y[order], w[order]
    stops = np.r_[np.flatnonzero(np.diff(sp) != 0), len(sp)-1]
    tp = np.cumsum(sw * sy)[stops]
    fp = np.cumsum(sw * (1-sy))[stops]
    precision = tp / (tp + fp)
    recall = tp / positive if positive else np.zeros_like(tp)
    ap = float(np.dot(np.diff(np.r_[0., recall]), precision)) if positive else None
    indices = np.unique(np.linspace(0, len(stops)-1, min(101, len(stops))).astype(int))
    curve = [{"threshold": float(sp[stops[i]]), "precision": float(precision[i]),
              "recall": float(recall[i]) if positive else None} for i in indices]
    fixed = []
    for threshold in thresholds:
        hit = p >= float(threshold)
        selected, positives = float(w[hit].sum()), float(np.dot(w[hit], y[hit]))
        fixed.append({"threshold": float(threshold), "selected_weight_mass": selected,
            "precision": positives / selected if selected else None,
            "recall": positives / positive if positive else None})
    bins = []
    bin_index = np.minimum(np.floor(p * 10).astype(int), 9)
    for index in range(10):
        low = index / 10.
        active = bin_index == index
        mass = float(w[active].sum())
        if mass:
            bins.append({"lower": round(float(low), 1), "upper": round(float(low+.1), 1),
                "rows": int(active.sum()), "weight_mass": mass,
                "mean_probability": float(np.dot(w[active], p[active])/mass),
                "observed_positive_share": float(np.dot(w[active], y[active])/mass)})
    clipped = np.clip(p, 1e-12, 1-1e-12)
    return {**result, "status": "descriptive_only" if positive and negative else "insufficient_two_class_support",
        "target": "high_return_candidate_recall" if head == "entry" else "positive_head_label_recall",
        "positive_group_weight_mass": positive, "negative_group_weight_mass": negative,
        "weighted_positive_share": positive / total, "average_precision": ap,
        "pr_curve": curve, "pr_curve_max_points": 101, "AP_uses_all_score_ties_exactly": True,
        "weighted_brier": float(np.dot(w, (p-y)**2)/total),
        "weighted_log_loss": float(-np.dot(w, y*np.log(clipped)+(1-y)*np.log1p(-clipped))/total),
        "fixed_threshold_precision_recall": fixed, "calibration_bins": bins}


def episode_excursion_diagnostics(episode, snapshots, *, ledger):
    """Independently recompute frozen P2b retrospective excursion convention."""
    if not episode.get("label_available") or episode.get("open_shares", 0) or not episode.get("exit_date"):
        return {"available": False, "reason": "episode_not_completely_mature",
                "scope": "whole_episode_retrospective_never_prediction_input"}
    if not ledger or ledger[0].get("action") != "BUY" or sum(x["action"] == "BUY" for x in ledger) != 1:
        raise ValueError("excursion diagnostics require the exact single-entry actual ledger")
    if episode.get("ledger_sha256") != stable_hash(ledger):
        raise ValueError("actual episode ledger identity mismatch")
    paid, shares = -float(ledger[0]["cash_flow"]), int(ledger[0]["shares"])
    if shares <= 0 or paid <= 0 or ledger[0]["date"] != episode["entry_date"] or ledger[-1]["date"] != episode["exit_date"]:
        raise ValueError("actual interval identity or fee cost basis invalid")
    if shares != sum(int(x["shares"]) for x in ledger[1:]):
        raise ValueError("actual interval inventory is not completely liquidated")
    basis = paid / shares
    observed = [x for x in snapshots if x.get("entry_date") == episode["entry_date"]
                and episode["entry_date"] <= x["date"] <= episode["exit_date"] and x.get("shares", 0) > 0]
    prices = [float(x["price"]) for x in ledger]
    peak = max([basis, *prices, *[float(x["bar"]["high"]) for x in observed]])
    trough = min([basis, *prices, *[float(x["bar"]["low"]) for x in observed]])
    return {"available": True, "scope": "whole_episode_retrospective_never_prediction_input",
            "max_favorable_excursion": peak / basis - 1,
            "max_adverse_excursion": trough / basis - 1,
            "final_peak_giveback": (peak - prices[-1]) / basis,
            "entry_fee_ratio": float(ledger[0]["fee"]) / paid,
            "basis": "actual_entry_debit_including_fees_per_share",
            "prices": "held_OHLC_and_actual_fill_prices_under_next_verified_open_execution"}


def summarize_episode_excursions(frame, *, cutoff="2026-06-30"):
    """Whole actual or cloned intervals stay posthoc and out of every X schema."""
    eligible = _mature(frame, cutoff, separate_sources=True)
    selected = frame.loc[eligible].copy()
    groups = selected.source.astype(str) + ":" + _group_ids(selected)
    result = {"cutoff": cutoff, "eligible_rows": len(selected), "excluded_rows": int((~eligible).sum()),
        "scope": "whole_episode_retrospective_never_prediction_input", "groups": [],
        "prediction_X_contains_retrospective_fields": bool(set(RETROSPECTIVE_COLUMNS[:-2]) & set(ENTRY_FEATURE_COLUMNS)),
        "input_metric_definition": "frozen_P2b_fee_cost_basis_held_OHLC_and_actual_fill_prices",
        "statistical_pass": False}
    for source in sorted(selected.source.unique()):
        for name in CLASSES:
            mask = (selected.source.eq(source) & selected.outcome_class.eq(name)).to_numpy()
            current = selected.loc[mask]
            metrics = {}
            for column in RETROSPECTIVE_COLUMNS:
                if column not in current:
                    continue
                values = pd.to_numeric(current[column], errors="coerce").to_numpy(float)
                values = values[np.isfinite(values)]
                metrics[column] = {"available_rows": len(values), "median": float(np.median(values)) if len(values) else None,
                    "p10": float(np.quantile(values, .1)) if len(values) else None,
                    "p90": float(np.quantile(values, .9)) if len(values) else None}
            result["groups"].append({"source": source, "outcome_class": name, "rows": len(current),
                "economic_groups": int(groups.loc[mask].nunique()), "metrics": metrics})
    return result


def _project(path, columns):
    import pyarrow.parquet as pq
    available = pq.read_schema(path).names
    return pd.read_parquet(path, columns=[c for c in columns if c in available])


def run_operation_diagnostics(model_run_id, *, cutoff="2026-06-30", include_probabilities=True,
                              progress=None, expected_core=None):
    """Write a fresh, separately indexed report without touching training runs."""
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    from my_strategy.services.czsc_high_return_models import HeadSession
    from my_strategy.services.czsc_high_return_research import _code_files, _head_data
    from my_strategy.services.czsc_trade_episodes import training_frame
    from my_strategy.storage.czsc_results import ResultStore
    root = (ARTIFACT_RUNS_ROOT / model_run_id).resolve()
    if Path(model_run_id).name != model_run_id or not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):
        raise ValueError("invalid diagnostic model root")
    cfg = _read(root / "config/research.json")
    if cutoff != cfg["production"]["train_end"]:
        raise ValueError("template cutoff must equal the frozen production training end")
    initial = _read(root / "config/initial_code_hashes.json")
    core = {str(p): _sha(p) for p in _code_files()}
    if core != initial or expected_core is not None and core != expected_core:
        raise ValueError("frozen core changed before diagnostics")
    final_path = root / "dataset/final/manifest.json"
    final = _read(final_path)
    if final.get("failures") or final["requested"] != final["successful"] or final["sources"] != ["ma", "structure", "wyckoff", "union"]:
        raise ValueError("diagnostics require a complete four-source P2b stage")
    snapshot = _read(root / "data_snapshots/source.json")
    source_hash = snapshot["snapshot_hash"]
    paths = sorted((root / "models/l2").glob("*/*/*/manifest.json")) if include_probabilities else []
    manifests = [(p, load_head(p.parent)) for p in paths]
    observed_end = snapshot.get("effective_data_end", cutoff)
    context = create_run_context(task="czsc-operation-diagnostics", as_of_date=observed_end,
        config={"version": VERSION, "model_run_id": model_run_id, "cutoff": cutoff,
                "template_entry_columns": TEMPLATE_ENTRY_COLUMNS, "quantiles": [.2, .5, .8],
                "min_template_groups": 30, "min_block_groups": 10, "include_probabilities": include_probabilities},
        data_version=source_hash, scope="train", stocks=[r["symbol"] for r in final["records"]],
        start_date=cfg["feature_start"], end_date=observed_end)
    inputs = [{"path": str(final_path), "sha256": _sha(final_path)},
              {"path": str(root / "config/research.json"), "sha256": _sha(root / "config/research.json")},
              {"path": str(root / "data_snapshots/source.json"), "sha256": _sha(root / "data_snapshots/source.json")}]
    entry_chunks, held_chunks, episode_chunks = [], [], []
    probability_chunks = {str(p): {"calibration": [], "policy": []} for p, _ in manifests}
    session = HeadSession()
    # joblib imports sklearn/OpenMP lazily. Load verified checkpoints before
    # entering threadpool_limits, so late-loaded pools cannot escape the cap.
    for model_path, _ in manifests:
        session._model(model_path.parent)
    independently_recomputed_union_intervals = 0
    from threadpoolctl import threadpool_limits
    metadata = ("symbol", "date", "label", "label_end", "label_available", "input_eligible", "source",
        "opportunity_id", "episode_id", "economic_group_id", "outcome_class", "net_return", "exit_advantage",
        "source_snapshot_hash", "continuation_policy_hash")
    needed_entry = tuple(dict.fromkeys((*ENTRY_FEATURE_COLUMNS, *metadata, *RETROSPECTIVE_COLUMNS)))
    needed_exit = tuple(dict.fromkeys((*EXIT_FEATURE_COLUMNS, *metadata)))
    requested = [r for r in final["records"] if r.get("episode_status") == "completed"]
    with threadpool_limits(limits=1):
        for number, record in enumerate(requested, 1):
            for key in ("entries", "exits", "episodes", "contracts"):
                p = Path(record[key + "_path"])
                actual = _sha(p)
                if actual != record[key + "_sha256"]:
                    raise ValueError("P2b diagnostic artifact hash mismatch")
                inputs.append({"path": str(p), "sha256": actual, "kind": key, "symbol": record["symbol"]})
            contracts = _read(record["contracts_path"])
            entries = _project(record["entries_path"], needed_entry)
            exits = _project(record["exits_path"], needed_exit)
            episodes = pd.read_parquet(record["episodes_path"])
            if episodes.empty:
                episodes = pd.DataFrame(columns=("source", "date", "signal_date", "episode_id",
                    "label_available", "label_end", "outcome_class"))
                mature_episodes = episodes.copy()
            else:
                episodes["date"] = episodes["signal_date"]
                mature_episodes = episodes.loc[_mature(episodes, cutoff, separate_sources=True)].copy()
            # Reference files were saved by P2b but not in its five-file manifest.
            # Bind them here, and bind each ledger to the saved episode identity.
            union_actual = mature_episodes.loc[mature_episodes.source.eq("union")]
            if len(union_actual):
                account_root = Path(record["path"]) / "reference/union"
                snapshot_path, ledger_path = account_root / "holding_snapshots.json", account_root / "ledger.json"
                snapshots, ledger = _read(snapshot_path), _read(ledger_path)
                inputs.extend({"path": str(p), "sha256": _sha(p), "kind": "union_reference_actual_account"}
                              for p in (snapshot_path, ledger_path))
                batches, active, inventory = {}, [], 0
                for fill in ledger:
                    if fill["action"] == "BUY":
                        if inventory:
                            raise ValueError("unexpected added actual entry")
                        active = [fill]; inventory = int(fill["shares"])
                    else:
                        active.append(fill); inventory -= int(fill["shares"])
                    if inventory == 0 and active:
                        batches[active[0]["date"]] = active; active = []
                by_entry = {}
                for value in snapshots:
                    by_entry.setdefault(value["entry_date"], []).append(value)
                for index, episode in union_actual.iterrows():
                    recalculated = episode_excursion_diagnostics(episode.to_dict(), by_entry.get(episode.entry_date, []),
                                                                 ledger=batches.get(episode.entry_date, []))
                    for field in ("max_favorable_excursion", "max_adverse_excursion", "final_peak_giveback", "entry_fee_ratio"):
                        if not math.isclose(recalculated[field], float(episode[field]), rel_tol=1e-9, abs_tol=1e-12):
                            raise ValueError("saved P2b excursion differs from independent actual ledger/holding recomputation")
                    mature_episodes.loc[index, "union_actual_excursion_independently_recomputed"] = True
                    independently_recomputed_union_intervals += 1
            episode_chunks.append(mature_episodes)
            union_entries = entries.loc[entries.source.eq("union")].copy() if len(entries) else entries
            union_exits = exits.loc[exits.source.eq("union")].copy() if len(exits) else exits
            if len(union_entries):
                entry_chunks.append(union_entries.loc[_mature(union_entries, cutoff),
                    [*TEMPLATE_ENTRY_COLUMNS, *[c for c in metadata if c in union_entries]]].copy())
            if len(union_exits):
                identity = episodes.loc[episodes.source.eq("union") & episodes.label_available & episodes.label_end.le(cutoff),
                    ["episode_id", "outcome_class"]].drop_duplicates("episode_id")
                held = union_exits.drop(columns="outcome_class", errors="ignore").merge(identity, on="episode_id", how="left")
                held_chunks.append(held.loc[_mature(held, cutoff),
                    [*TEMPLATE_HOLDING_COLUMNS, *[c for c in metadata if c in held]]].copy())
            for model_path, manifest in manifests:
                head = manifest["head"]
                kind = "exit" if head in {"exit", "exit_advantage"} else "entry"
                raw = union_exits if kind == "exit" else union_entries
                if raw.empty:
                    continue
                base_contract = contracts["union"][kind]
                data = training_frame({"exits" if kind == "exit" else "entries": raw, "contracts": contracts},
                                      source="union", head=kind)
                data, bound = _head_data(data, base_contract, head, tuple(manifest["columns"]))
                if stable_hash(bound) != manifest["target_contract_hash"] or bound["source_snapshot_hash"] != source_hash:
                    raise ValueError("diagnostic model target differs from exact P2b target")
                _, calibration, policy = time_masks(data, manifest["fold"], manifest["regression"])
                selected = calibration | policy
                if not selected.any():
                    continue
                scorer_manifest, scorer = session._model(model_path.parent)
                values = data.loc[selected, manifest["columns"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
                score = session._inference(values, scorer_manifest, scorer)
                scored = data.loc[selected, [c for c in metadata if c in data]].copy()
                scored["probability"] = score
                for name, mask in (("calibration", calibration), ("policy", policy)):
                    probability_chunks[str(model_path)][name].append(scored.loc[mask[selected]].copy())
            if progress and (number % 100 == 0 or number == len(requested)):
                progress("diagnostic_P2b_inputs", number, len(requested))
    entries = pd.concat(entry_chunks, ignore_index=True) if entry_chunks else pd.DataFrame()
    held = pd.concat(held_chunks, ignore_index=True) if held_chunks else pd.DataFrame()
    episodes = pd.concat(episode_chunks, ignore_index=True) if episode_chunks else pd.DataFrame()
    if entries.empty or held.empty:
        raise ValueError("mature training candidate/holding diagnostic populations are empty")
    with threadpool_limits(limits=1):
        entry_report = discover_causal_templates(entries, columns=TEMPLATE_ENTRY_COLUMNS, cutoff=cutoff)
        if progress:
            progress("entry_templates", len(entry_report["templates"]), len(entry_report["templates"]))
        holding_report = discover_causal_templates(held, columns=TEMPLATE_HOLDING_COLUMNS, cutoff=cutoff, population="holding")
        excursion_report = summarize_episode_excursions(episodes, cutoff=cutoff)
    excursion_report["independently_recomputed_union_intervals"] = independently_recomputed_union_intervals
    probabilities = []
    for model_path, manifest in manifests:
        for name in ("calibration", "policy"):
            chunks = probability_chunks[str(model_path)][name]
            if not chunks:
                probabilities.append({"head": manifest["head"], "fold": manifest["fold"]["name"],
                    "architecture": manifest["architecture"], "segment": name, "status": "unavailable_no_mature_rows"})
                continue
            data = pd.concat(chunks, ignore_index=True)
            values = data.pop("probability").to_numpy(float)
            with threadpool_limits(limits=1):
                result = weighted_probability_diagnostics(data, values, head=manifest["head"], segment=name, fold=manifest["fold"])
            result.update(architecture=manifest["architecture"], model_path=str(model_path.parent),
                manifest_sha256=manifest["manifest_sha256"], target_contract_hash=manifest["target_contract_hash"],
                population="mature_candidate_clone" if manifest["head"] in {"entry", "return", "loss"} else "mature_actual_reference_holding_states")
            probabilities.append(result)
        for file in ("manifest.json", manifest["checkpoint"], "linear-layers.json"):
            path = model_path.parent / file
            if path.exists():
                inputs.append({"path": str(path), "sha256": _sha(path), "kind": "model"})
        if load_head(model_path.parent) != manifest:
            raise ValueError("existing model changed during diagnostics")
    if {str(p): _sha(p) for p in _code_files()} != core:
        raise ValueError("frozen core changed during independent diagnostics")
    # Only already written inputs are frozen; a running model may append new heads.
    if any(_sha(x["path"]) != x["sha256"] for x in inputs):
        raise ValueError("diagnostic source input changed during reading")
    pending = []
    for fold in [*cfg["development_windows"], cfg["production"]]:
        for architecture in cfg["architectures"]:
            for head in ("entry", "exit", "exit_advantage"):
                path = root / "models/l2" / fold["name"] / architecture / head / "manifest.json"
                if path not in paths:
                    pending.append({"fold": fold["name"], "architecture": architecture, "head": head,
                        "reason": "head_not_completed_at_diagnostic_snapshot"})
    report = {"version": VERSION, "run_id": context.run_id, "model_run_id": model_run_id,
        "run_context": context.to_dict(), "status": "completed_available_diagnostics_pending_model_heads" if pending else "completed_diagnostics",
        "completed_at": local_now().isoformat(), "source_snapshot_hash": source_hash, "cutoff": cutoff,
        "coverage": {"requested_source_symbols": final["requested"], "supported_actual_symbols": len(requested),
            "entry_rows": len(entries), "holding_rows": len(held), "actual_episode_rows": len(episodes),
            "complete_model_heads_at_snapshot": len(manifests), "probability_segment_reports": len(probabilities)},
        "entry_templates": entry_report, "holding_templates": holding_report,
        "retrospective_actual_interval_diagnostics": excursion_report,
        "retrospective_interval_file": "diagnostics/actual-interval-outcomes.parquet",
        "probability_diagnostics": probabilities, "pending_model_heads": pending,
        "diagnostic_input_manifest_path": "data_snapshots/diagnostic-inputs.json",
        "frozen_core_hashes_unchanged": core, "independent_diagnostic_code_sha256": _sha(__file__),
        "model_selection_budget_changed": False, "training_artifacts_modified": False,
        "model_gate": {"pass": False, "reason": "exploratory_diagnostics_not_strategy_acceptance"},
        "publication_allowed": False, "interpretation": "Templates describe training outcomes. PR evaluates mature labels, not realized strategy NAV. Calibration metrics reuse calibrator-fit labels and are descriptive. Policy data are not used here to retune anything."}
    interval_columns = [c for c in ("episode_id", "symbol", "source", "opportunity_id", "economic_group_id",
        "entry_date", "exit_date", "label_end", "outcome_class", *RETROSPECTIVE_COLUMNS,
        "outcome_scope", "ledger_sha256", "source_snapshot_hash", "continuation_policy_hash",
        "union_actual_excursion_independently_recomputed") if c in episodes]
    interval_file = context.subdir("diagnostics") / "actual-interval-outcomes.parquet"
    episodes[interval_columns].to_parquet(interval_file, index=False)
    report["retrospective_interval_file_sha256"] = _sha(interval_file)
    _write(context.subdir("data_snapshots") / "diagnostic-inputs.json", inputs)
    report["diagnostic_input_manifest_sha256"] = _sha(context.subdir("data_snapshots") / "diagnostic-inputs.json")
    _write(context.subdir("reports") / "operation-diagnostics.json", report)
    ResultStore().save("czsc-operation-diagnostics", report)
    context.write_metadata({"status": "complete", "model_family": "operation_diagnostics", "model_run_id": model_run_id,
        "coverage": report["coverage"], "publication_allowed": False})
    return report
