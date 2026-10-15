"""Small chronological forecast comparisons; test cases never select parameters."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd

from my_strategy.services.kronos_prediction import predict

DEFAULT_CONFIGURATIONS = (
    {"lookback": 128, "temperature": .8, "top_p": .9, "samples": 8},
    {"lookback": 256, "temperature": .8, "top_p": .9, "samples": 8},
)


def _date(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Tuning dates must be real dates")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai")
    return stamp.date().isoformat()


def _actual(case):
    data = case["actual_future"]
    if not isinstance(data, pd.DataFrame) or "close" not in data:
        raise ValueError("actual_future must contain real future close prices")
    column = next((key for key in ("date", "timestamps", "dt") if key in data), None)
    if not column and not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Actual future observations require dates")
    dates = [_date(x) for x in (data[column] if column else data.index)]
    expected = [_date(x) for x in case["future_dates"]]
    if dates != expected or len(set(dates)) != len(dates):
        raise ValueError("Actual evaluation dates must exactly match the requested forecast horizon")
    close = data.close.to_numpy(dtype=np.float64)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Future observed close prices must be finite and positive")
    return close


def _case_metrics(result, actual):
    quality = result.get("quality") or {}
    forecast, summary = result.get("forecast") or [], result.get("summary")
    common = {"prediction_usable": bool(quality.get("prediction_usable")),
              "valid_samples": int(quality.get("valid_samples", 0)),
              "requested_samples": int(quality.get("requested_samples", 0))}
    if not forecast or summary is None:
        return dict(common, evaluable=False)
    predicted = np.array([x["close"] for x in forecast])
    lower = np.array([x["close_p10"] for x in forecast])
    upper = np.array([x["close_p90"] for x in forecast])
    if len(predicted) != len(actual) or not np.isfinite(np.concatenate((predicted, lower, upper))).all():
        return dict(common, evaluable=False)
    last = float(result["input"]["last_close"])
    target = float(actual[-1] > last)
    probability = float(summary["up_path_fraction"])
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return dict(common, evaluable=True,
                mae=float(np.abs(predicted - actual).mean()),
                normalized_mae=float((np.abs(predicted - actual) / last).mean()),
                brier=float((probability - target) ** 2),
                log_loss=float(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped))),
                interval_coverage=float(((actual >= lower) & (actual <= upper)).mean()),
                relative_interval_width=float(((upper - lower) / last).mean()))


def _aggregate(records, requested, samples_per_case):
    valid = [row["metrics"] for row in records if row.get("metrics", {}).get("evaluable")]
    metrics = {key: float(np.mean([row[key] for row in valid])) if valid else None
               for key in ("mae", "normalized_mae", "brier", "log_loss", "interval_coverage", "relative_interval_width")}
    attempted_paths = sum(row.get("metrics", {}).get("requested_samples", 0) for row in records)
    generated = requested * samples_per_case
    usable = sum(bool(row.get("metrics", {}).get("prediction_usable")) for row in records)
    metrics.update(requested_cases=requested, attempted_cases=len(records), evaluable_cases=len(valid),
                   failed_cases=sum("error" in row for row in records), usable_cases=usable,
                   usable_case_fraction=usable / requested if requested else 0,
                   requested_paths=generated, attempted_paths=attempted_paths,
                   valid_path_fraction=(sum(row.get("metrics", {}).get("valid_samples", 0) for row in records) / generated) if generated else 0,
                   elapsed_seconds=sum(row.get("elapsed_seconds", 0) for row in records))
    if valid:
        metrics["selection_loss"] = (metrics["normalized_mae"] + metrics["brier"]
                                     + .25 * abs(metrics["interval_coverage"] - .8)
                                     + .1 * metrics["relative_interval_width"]
                                     + 1 - metrics["usable_case_fraction"])
    else:
        metrics["selection_loss"] = None
    return metrics


def tune_kronos(cases: list[dict], validation_dates, test_dates, *,
                configurations=None, predict_fn=predict, budget_seconds: float = 300,
                stop=None, seed: int = 42, device: str = "auto") -> dict:
    """Tune <=8 frozen candidates on validation, then evaluate one locked test pass.

    `stop` follows task cancel_check semantics (raise InterruptedError), and may
    alternatively return True. Budget expiry is checked before and after calls;
    the real predictor checks it between autoregressive steps.
    """
    started = time.monotonic()
    if not 0 < budget_seconds <= 3600 or not 1 <= len(cases) <= 100:
        raise ValueError("Bounded tuning needs 1–100 cases and a budget in (0, 3600] seconds")
    validation = sorted({_date(x) for x in validation_dates})
    test = sorted({_date(x) for x in test_dates})
    if not validation or not test or validation[-1] >= test[0]:
        raise ValueError("Validation and test decision dates must be disjoint and chronological")
    configs = [dict(x) for x in (configurations or DEFAULT_CONFIGURATIONS)]
    if not 1 <= len(configs) <= 8:
        raise ValueError("At most eight configurations are allowed")
    allowed = {"lookback", "temperature", "top_p", "samples"}
    for config in configs:
        if set(config) != allowed:
            raise ValueError("Each frozen configuration needs lookback, temperature, top_p and samples")
        if not 16 <= config["lookback"] <= 512 or not 1 <= config["samples"] <= 64:
            raise ValueError("Configuration exceeds lightweight context/sample limits")
    prepared, unique = [], set()
    for case in cases:
        key = (case["symbol"], _date(case["as_of"]))
        if key in unique:
            raise ValueError("Duplicate symbol/as_of case would distort the evaluation population")
        unique.add(key)
        if key[1] not in set(validation + test):
            raise ValueError("Every case must belong to validation or the locked test set")
        actual = _actual(case)
        if not 1 <= len(actual) <= 10:
            raise ValueError("Tuning horizon must be 1–10 trading days")
        # Test targets cannot enter parameter selection. Validation targets must
        # all mature before the first held-out test decision.
        if key[1] in validation and _date(case["future_dates"][-1]) >= test[0]:
            raise ValueError("Validation labels have not matured before the test decision boundary")
        prepared.append((case, actual, "validation" if key[1] in validation else "test"))
    if any(not any(split == name for _, _, split in prepared) for name in ("validation", "test")):
        raise ValueError("Both validation and test require actual cases")
    def check():
        if stop and stop():
            raise InterruptedError("cancelled")
        if time.monotonic() - started >= budget_seconds:
            raise TimeoutError("forecast_tuning_budget_exhausted")
    active_evaluation = {}
    def evaluate(config, split):
        population = [(case, actual) for case, actual, name in prepared if name == split]
        records = []
        active_evaluation.clear()
        active_evaluation.update(configuration=config, split=split, records=records, requested_cases=len(population))
        for case, actual in population:
            check()
            before = time.monotonic()
            row = {"symbol": case["symbol"], "as_of": _date(case["as_of"])}
            try:
                result = predict_fn(case["frame"], case["future_dates"], symbol=case["symbol"],
                                    as_of=case["as_of"], horizon=len(actual), seed=seed, device=device,
                                    calendar_identity=case.get("calendar_identity"),
                                    calendar_verified=case.get("calendar_verified", False),
                                    cancel_check=check, **config)
                row.update(cache_key=result.get("cache_key"), metrics=_case_metrics(result, actual))
            except (InterruptedError, TimeoutError):
                raise
            except (ValueError, RuntimeError, OSError) as exc:
                row.update(error=f"{type(exc).__name__}: {exc}",
                           metrics={"evaluable": False, "prediction_usable": False,
                                    "valid_samples": 0, "requested_samples": int(config["samples"])})
            row["elapsed_seconds"] = time.monotonic() - before
            records.append(row)
            check()
        completed = {"records": records, "metrics": _aggregate(records, len(population), config["samples"])}
        active_evaluation.clear()
        return completed
    comparisons, selected, heldout = [], None, None
    status, reason = "completed", None
    try:
        for config in configs:
            check()
            comparison = {"configuration": config, **evaluate(config, "validation")}
            comparisons.append(comparison)
        candidates = [row for row in comparisons if row["metrics"]["selection_loss"] is not None]
        if candidates:
            selected = min(candidates, key=lambda row: row["metrics"]["selection_loss"])["configuration"]
            check()
            heldout = evaluate(selected, "test")
        else:
            reason = "no_evaluable_validation_candidate"
    except TimeoutError as exc:
        status, reason = "partial", str(exc)
    except InterruptedError as exc:
        status, reason = "cancelled", str(exc)
    result = {"schema": "kronos_tuning_v1", "status": status, "reason": reason,
              "validation_dates": validation, "test_dates": test, "seed": seed,
              "planned_configurations": configs, "completed_comparisons": comparisons,
              "selected_configuration": selected, "test_evaluation": heldout,
              "interrupted_evaluation": ({**active_evaluation, "metrics": _aggregate(
                  active_evaluation["records"], active_evaluation["requested_cases"],
                  active_evaluation["configuration"]["samples"])} if active_evaluation else None),
              "selection_rule": "validation: normalized_MAE + Brier + 0.25*abs(coverage-0.8) + 0.1*relative_width + (1-usable_case_fraction)",
              "test_used_for_selection": False, "probabilities_calibrated": False,
              "qualification": {"research_only": True, "positive_allocation_eligible": False,
                                "independent_forward_status": "pending"},
              "budget_seconds": budget_seconds, "elapsed_seconds": time.monotonic() - started}
    result["experiment_id"] = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return result
