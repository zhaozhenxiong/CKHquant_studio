"""Small, chronological three-class L2 score; no automatic production promotion."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

SCHEMA = "simple_score_v1"
SPLITS = ("train", "tune", "calibration", "policy", "test")
CLASSES = ("down", "flat", "up")


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _dates(values, *, daily_close: bool = False) -> pd.DatetimeIndex:
    result = []
    for value in values:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError("Dates may not be missing")
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("Asia/Shanghai")
        else:
            stamp = stamp.tz_convert("Asia/Shanghai")
        if daily_close and stamp == stamp.normalize():
            stamp += pd.Timedelta(hours=15)
        result.append(stamp.tz_convert("UTC"))
    return pd.DatetimeIndex(result)


def _probabilities(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _metrics(labels, probabilities) -> dict:
    labels = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    targets = np.eye(3)[labels]
    return {"rows": len(labels), "log_loss": float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
            "brier": float(np.square(p - targets).sum(axis=1).mean()),
            "accuracy": float((p.argmax(axis=1) == labels).mean()),
            "class_counts": {name: int((labels == i).sum()) for i, name in enumerate(CLASSES)}}


def fit_simple_score(frame: pd.DataFrame, feature_columns: list[str], split_ranges: dict,
                     *, label_column: str = "label", decision_column: str = "as_of",
                     maturity_column: str = "label_mature_at", source_identity=None,
                     label_contract: dict | None = None, regularization_candidates=(.1, 1.0, 10.0),
                     seed: int = 42, min_segment_rows: int = 30,
                     feature_available_column: str | None = None, cancel_check=None) -> dict:
    """Fit on train; choose C on tune; temperature on calibration; lock policy/test.

    Features must be point-in-time observations supplied by the caller. Future
    labels are admitted only when mature inside their own explicitly frozen split.
    This function returns compact JSON weights and writes no files or registry.
    """
    from scipy.optimize import minimize_scalar
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    if not source_identity or not label_contract:
        raise ValueError("Frozen source_identity and label_contract are required")
    features = list(feature_columns)
    if not 1 <= len(features) <= 20 or len(set(features)) != len(features):
        raise ValueError("Require 1–20 distinct numeric feature columns")
    forbidden = {label_column, decision_column, maturity_column, feature_available_column}
    if any(feature in forbidden for feature in features):
        raise ValueError("Labels and temporal metadata cannot be score features")
    required = set(features) | {label_column, decision_column, maturity_column}
    if feature_available_column:
        required.add(feature_available_column)
    if not required.issubset(frame.columns):
        raise ValueError("Missing score inputs: " + ", ".join(sorted(required - set(frame.columns))))
    if not 1 <= len(regularization_candidates) <= 3 or any(not np.isfinite(c) or c <= 0 for c in regularization_candidates):
        raise ValueError("At most three positive L2 C candidates are allowed")
    if len(frame) > 100000 or min_segment_rows < 10:
        raise ValueError("Lightweight score budget: at most 100000 rows, at least 10 rows per evaluation split")
    if set(split_ranges) != set(SPLITS):
        raise ValueError("Exactly train/tune/calibration/policy/test ranges are required")
    bounds = {}
    previous_end = None
    for name in SPLITS:
        interval = split_ranges[name]
        if len(interval) != 2:
            raise ValueError("Each split requires [start, end]")
        start, end = _dates(interval)
        # Date-only boundaries include the full calendar day, never a later day.
        if len(str(interval[1])) == 10:
            end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError("Chronological split ranges must be ordered and disjoint")
        bounds[name] = (start, end)
        previous_end = end
    decisions = _dates(frame[decision_column], daily_close=True)
    mature = _dates(frame[maturity_column], daily_close=True)
    if (mature <= decisions).any():
        raise ValueError("Future direction labels must mature after their decision")
    if feature_available_column:
        available = _dates(frame[feature_available_column], daily_close=True)
        if (available > decisions).any():
            raise ValueError("Feature became available after its decision")
    columns = frame[features].to_numpy(dtype=np.float64)
    if not np.isfinite(columns).all():
        raise ValueError("SimpleScore requires finite features; missing Kronos values must not be filled with zero")
    raw_labels = frame[label_column].to_numpy()
    if not np.isin(raw_labels, [0, 1, 2]).all():
        raise ValueError("label must be 0=down, 1=flat, 2=up")
    labels = raw_labels.astype(int)
    subsets, populations = {}, {}
    for name in SPLITS:
        start, end = bounds[name]
        assigned = (decisions >= start) & (decisions <= end)
        used = assigned & (mature <= end)
        count = int(used.sum())
        minimum = max(60, min_segment_rows) if name == "train" else min_segment_rows
        if count < minimum:
            raise ValueError(f"{name}: only {count} mature rows; require {minimum}")
        subsets[name] = used
        populations[name] = {"assigned_rows": int(assigned.sum()), "mature_rows": count,
                             "excluded_immature_rows": int((assigned & ~used).sum()),
                             "class_counts": {key: int((labels[used] == i).sum()) for i, key in enumerate(CLASSES)}}
    if min(populations["train"]["class_counts"].values()) < 5:
        raise ValueError("Training needs at least five mature examples of each of the three classes")
    train = subsets["train"]
    mean = columns[train].mean(axis=0)
    scale = columns[train].std(axis=0)
    scale[scale < 1e-12] = 1
    x = (columns - mean) / scale
    counts = np.bincount(labels[train], minlength=3)
    prior = (counts + .5) / (counts.sum() + 1.5)
    trials = []
    best = None
    def check_cancel():
        if cancel_check and cancel_check():
            raise InterruptedError("SimpleScore training cancelled")
    for c in regularization_candidates:
        check_cancel()
        model = LogisticRegression(C=float(c), penalty="l2", solver="lbfgs", max_iter=300, random_state=seed)
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x[train], labels[train])
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in seen)
        metrics = _metrics(labels[subsets["tune"]], model.predict_proba(x[subsets["tune"]]))
        trials.append({"C": float(c), "converged": converged, "tune": metrics})
        if converged and (best is None or metrics["log_loss"] < best[0]):
            best = (metrics["log_loss"], model)
    if best is None:
        raise ValueError("No converged simple-score candidate")
    model = best[1]
    logits = x @ model.coef_.T + model.intercept_
    calibration = subsets["calibration"]
    objective = lambda log_t: _metrics(labels[calibration], _probabilities(logits[calibration], np.exp(log_t)))["log_loss"]
    calibrated = minimize_scalar(objective, bounds=(-2.302585093, 2.302585093), method="bounded")
    if not calibrated.success or not np.isfinite(calibrated.fun):
        raise ValueError("Independent positive-temperature calibration failed")
    temperature = float(np.exp(calibrated.x))
    metrics = {}
    for name, mask in subsets.items():
        metrics[name] = _metrics(labels[mask], _probabilities(logits[mask], temperature))
        metrics[name]["frozen_train_prior_baseline"] = _metrics(labels[mask], np.tile(prior, (int(mask.sum()), 1)))
    def passes(name):
        evaluated = metrics[name]
        baseline = evaluated["frozen_train_prior_baseline"]
        return (min(evaluated["class_counts"].values()) >= 3 and
                evaluated["log_loss"] < baseline["log_loss"] and evaluated["brier"] < baseline["brier"])
    check_cancel()
    identity = {"features": features, "source": source_identity, "label_contract": label_contract,
                "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "split_ranges": split_ranges, "seed": seed, "regularization_candidates": list(regularization_candidates),
                "data_sha256": _hash({"features": columns.tolist(), "labels": labels.tolist(),
                                      "decisions": decisions.astype(str).tolist(), "maturity": mature.astype(str).tolist()})}
    artifact = {"schema": SCHEMA, "status": "completed", "model_type": "multinomial_logistic_L2",
                "identity": identity, "features": features, "classes": list(CLASSES),
                "coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
                "standardization": {"mean": mean.tolist(), "scale": scale.tolist(), "fitted_on": "train_only"},
                "temperature": temperature, "calibration_kind": "positive_temperature_only",
                "regularization_C": float(model.C), "train_prior": prior.tolist(),
                "trials": trials, "metrics": metrics, "population": populations,
                "available_after": bounds["policy"][1].isoformat(),
                "evaluation_end": bounds["test"][1].isoformat(),
                "qualification": {"policy_baseline_passed": passes("policy"),
                                  "heldout_baseline_passed": passes("test"),
                                  "positive_allocation_eligible": False,
                                  "independent_forward_status": "pending",
                                  "research_only": True,
                                  "reason": "Historical model fit is not execution or independent forward certification"}}
    artifact["created_at"] = datetime.now(timezone.utc).isoformat()
    artifact["model_version"] = _hash(artifact)
    return artifact


def score_samples(artifact: dict, frame: pd.DataFrame, *, as_of: str,
                  historical_reconstruction: bool = False) -> dict:
    """Read compact weights without pickle; reject retroactive application by default."""
    if artifact.get("schema") != SCHEMA:
        raise ValueError("Unsupported score schema")
    body = {key: value for key, value in artifact.items() if key != "model_version"}
    if _hash(body) != artifact.get("model_version"):
        raise ValueError("SimpleScore artifact identity mismatch")
    requested = _dates([as_of], daily_close=True)[0]
    earliest = _dates([artifact["available_after"]])[0]
    timing_ok = requested > earliest
    if not timing_ok:
        raise ValueError("Score training/calibration/policy evidence was not available at this decision")
    actually_frozen = requested >= _dates([artifact["created_at"]])[0]
    if not actually_frozen and not historical_reconstruction:
        raise ValueError("Historical application requires explicit historical_reconstruction=True")
    features = artifact["features"]
    if not set(features).issubset(frame.columns):
        raise ValueError("Score feature schema mismatch")
    x = frame[features].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("SimpleScore features are unavailable/nonfinite")
    standardized = (x - np.asarray(artifact["standardization"]["mean"])) / np.asarray(artifact["standardization"]["scale"])
    logits = standardized @ np.asarray(artifact["coefficients"]).T + np.asarray(artifact["intercept"])
    p = _probabilities(logits, artifact["temperature"])
    scores = 100 * (p[:, 2] + .5 * p[:, 1])
    return {"schema": SCHEMA, "model_version": artifact["model_version"], "as_of": str(as_of),
            "features": features, "probabilities": [dict(zip(CLASSES, row.tolist())) for row in p],
            "scores": scores.tolist(), "score_semantics": "100 * (P(up) + 0.5 * P(flat))",
            "historical_reconstruction": bool(historical_reconstruction), "temporal_available": timing_ok,
            "model_actually_frozen": actually_frozen,
            "qualification": dict(artifact["qualification"], positive_allocation_eligible=False)}
