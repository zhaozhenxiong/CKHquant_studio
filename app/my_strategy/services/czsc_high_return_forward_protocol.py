"""Frozen, consecutive forward windows on one actual continuous account.

Statistics concern paired fee-net daily portfolio returns. They do not confer
publication permission or substitute for source, model and ledger audits.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash

FORWARD_PROTOCOL_VERSION = "czsc_high_return_continuous_forward_protocol_v1"
FORWARD_PROTOCOL = {
    "version": FORWARD_PROTOCOL_VERSION,
    "window_count": 3,
    "window_sessions": 480,
    "block_lengths": [160, 240],
    "minimum_complete_blocks": 2,
    "min_window_round_trips": 50,
    "min_total_round_trips": 200,
    "familywise_alpha": .05,
    "minimum_power": .80,
    "minimum_detectable_annual_net_improvement": .02,
    "trading_sessions_per_year": 252,
    "bootstrap_repetitions": 10000,
    "seed": 42,
}


def _date(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("forward dates must be real dates")
    return stamp.date().isoformat()


def _dates(values):
    dates = [_date(value) for value in values]
    if dates != sorted(set(dates)):
        raise ValueError("forward dates must be unique and strictly ascending")
    return dates


def forward_protocol(config=None):
    """Canonical frozen thresholds; config may wrap them in forward_protocol."""
    config = config or {}
    supplied = config.get("forward_protocol", config if "window_count" in config else {})
    if not isinstance(supplied, dict):
        raise ValueError("forward_protocol must be a mapping")
    generated = {"preregistered_comparisons", "per_comparison_alpha", "one_sided_confidence",
        "minimum_detectable_daily_net_improvement", "protocol_hash", "method", "null_hypothesis"}
    unknown = set(supplied) - set(FORWARD_PROTOCOL) - generated
    if unknown:
        raise ValueError("unknown forward protocol settings: " + ",".join(sorted(unknown)))
    result = copy.deepcopy(FORWARD_PROTOCOL)
    result.update({key: value for key, value in supplied.items() if key in FORWARD_PROTOCOL})
    for key in ("window_count", "window_sessions", "minimum_complete_blocks",
            "min_window_round_trips", "min_total_round_trips",
            "trading_sessions_per_year", "bootstrap_repetitions", "seed"):
        value = result[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or int(value) != value:
            raise ValueError(key + " must be an integer")
        result[key] = int(value)
    for key in ("familywise_alpha", "minimum_power", "minimum_detectable_annual_net_improvement"):
        value = float(result[key])
        if not math.isfinite(value):
            raise ValueError(key + " must be finite")
        result[key] = value
    if result["version"] != FORWARD_PROTOCOL_VERSION or result["window_count"] != 3:
        raise ValueError("this protocol preregisters exactly three windows")
    if list(result["block_lengths"]) != [160, 240]:
        raise ValueError("forward dependence blocks are frozen to 160/240")
    result["block_lengths"] = [160, 240]
    if result["window_sessions"] < 480 or result["minimum_complete_blocks"] < 2:
        raise ValueError("forward windows need at least 480 sessions and two complete blocks")
    if result["window_sessions"] < result["minimum_complete_blocks"] * max(result["block_lengths"]):
        raise ValueError("window is shorter than its registered complete-block minimum")
    if result["min_window_round_trips"] < 50 or result["min_total_round_trips"] < 200:
        raise ValueError("mature closed-trip evidence cannot relax the 50/200 minimums")
    if not 0 < result["familywise_alpha"] <= .05 or not .8 <= result["minimum_power"] < 1:
        raise ValueError("forward alpha/power cannot relax the preregistered minimums")
    if not 0 < result["minimum_detectable_annual_net_improvement"] <= .02:
        raise ValueError("annual design effect must be positive and no greater than .02")
    if result["trading_sessions_per_year"] != 252 or result["bootstrap_repetitions"] < 1000 or result["seed"] < 0:
        raise ValueError("forward annualization, bootstrap repetitions or seed is invalid")
    result.update(preregistered_comparisons=6,
        per_comparison_alpha=result["familywise_alpha"] / 6,
        one_sided_confidence=1 - result["familywise_alpha"] / 6,
        minimum_detectable_daily_net_improvement=result["minimum_detectable_annual_net_improvement"] / 252,
        method="registered_moving_date_blocks_diagnostic_paired_fee_net_daily_return_bootstrap",
        null_hypothesis="mean_full_minus_reference_daily_net_return_le_zero")
    result["protocol_hash"] = stable_hash(result)
    for key in generated - {"protocol_hash"}:
        if key in supplied and supplied[key] != result[key]:
            raise ValueError("forward derived threshold changed: " + key)
    if supplied.get("protocol_hash") and supplied["protocol_hash"] != result["protocol_hash"]:
        raise ValueError("forward protocol hash mismatch")
    return result


def forward_windows(calendar, start, observed_end, config=None):
    """Ordinal plans use verified dates only, without projecting future dates."""
    protocol = forward_protocol(config)
    calendar = _dates(calendar)
    start = _date(start) if start is not None else None
    observed_end = _date(observed_end) if observed_end is not None else None
    registered = [day for day in calendar if start is not None and day >= start]
    plans = []
    sessions = protocol["window_sessions"]
    for offset in range(protocol["window_count"]):
        planned = registered[offset * sessions:(offset + 1) * sessions]
        observed = [day for day in planned if observed_end is not None and day <= observed_end]
        name = "forward_" + str(offset + 1)
        plans.append({"name": name, "window_id": name, "ordinal": offset + 1,
            "start": planned[0] if planned else None,
            "end": planned[-1] if len(planned) == sessions else None,
            "dates": observed, "observed_verified_dates": observed.copy(),
            "observed_sessions": len(observed), "required_sessions": sessions,
            "start_session_offset": offset * sessions,
            "end_session_offset": (offset + 1) * sessions - 1,
            "complete": len(observed) == sessions,
            "continuous_account_reset": False, "protocol_hash": protocol["protocol_hash"]})
    return plans


def _original_cash(variant, initial_cash):
    value = variant.get("metrics", {}).get("initial_cash")
    if value is None:
        value = float(initial_cash) * int(variant.get("account_count", 1))
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("original continuous account capital must be finite and positive")
    return value


def _daily_frame(variant):
    frame = pd.DataFrame(variant.get("daily", []))
    if frame.empty:
        return pd.DataFrame(columns=["date", "equity", "return"])
    if not {"date", "equity"}.issubset(frame.columns):
        raise ValueError("actual continuous daily equity is missing")
    frame = frame.copy()
    frame["date"] = _dates(frame["date"].tolist())
    frame["equity"] = pd.to_numeric(frame.equity, errors="coerce")
    if not np.isfinite(frame.equity.to_numpy(dtype=float)).all() or (frame.equity <= 0).any():
        raise ValueError("actual daily equity must be finite and positive")
    return frame.set_index("date", drop=False)


def slice_continuous_variant(variant, dates, all_dates, initial_cash=100000):
    """Slice equity without restarting cash, positions or the actual Broker."""
    dates, all_dates = _dates(dates), _dates(all_dates)
    frame = _daily_frame(variant)
    if frame.index.tolist() != all_dates:
        raise ValueError("continuous variant must cover every verified date exactly")
    if dates and (dates != [day for day in all_dates if dates[0] <= day <= dates[-1]]):
        raise ValueError("forward slice must be a contiguous verified-date interval")
    original = _original_cash(variant, initial_cash)
    result = copy.deepcopy(variant)
    lifetime = copy.deepcopy(variant.get("metrics", {}))
    result["account_lifetime_metrics"] = lifetime
    if not dates:
        result.update(daily=[], metrics={"initial_cash": original, "period_net_return": None,
            "net_return": None, "cumulative_net_return_from_original_capital": None},
            window_basis={"prior_date": None, "prior_equity": None,
                "original_initial_cash": original, "account_reset": False, "observed_sessions": 0})
        return result
    index = all_dates.index(dates[0])
    prior_date = all_dates[index - 1] if index else None
    prior_equity = float(frame.loc[prior_date, "equity"]) if prior_date else original
    selected = frame.loc[dates].copy()
    prior = selected.equity.shift(1)
    prior.iloc[0] = prior_equity
    selected["return"] = selected.equity / prior - 1
    selected["drawdown"] = selected.equity / selected.equity.cummax().clip(lower=prior_equity) - 1
    ending = float(selected.equity.iloc[-1])
    daily_metrics = {"initial_cash": original, "period_start_equity": prior_equity,
        "final_equity": ending, "net_return": ending / prior_equity - 1,
        "total_return": ending / prior_equity - 1, "period_net_return": ending / prior_equity - 1,
        "cumulative_net_return_from_original_capital": ending / original - 1,
        "max_drawdown": float(-selected.drawdown.min()),
        "tail_daily_return_5pct": float(selected["return"].quantile(.05))}
    if "market_value" in selected:
        daily_metrics["mean_capital_occupancy"] = float((selected.market_value / original).mean())
    result.update(daily=selected.to_dict("records"), metrics=daily_metrics,
        window_basis={"start": dates[0], "end": dates[-1], "prior_date": prior_date,
            "prior_equity": prior_equity, "original_initial_cash": original,
            "observed_sessions": len(dates), "account_reset": False},
        metrics_scope="window_daily_equity_metrics_trade_metrics_in_account_lifetime_metrics")
    return result


def _bootstrap_means(values, block, repetitions, seed):
    """Same moving-date blocks as date_block_bootstrap, without dropping dates."""
    size = len(values)
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, size - block + 1, size=(repetitions, math.ceil(size / block)))
    sums = np.concatenate(([0.], np.cumsum(values)))
    totals = (sums[starts + block] - sums[starts]).sum(axis=1)
    last = size % block
    if last:
        end = starts[:, -1]
        totals -= sums[end + block] - sums[end + last]
    return totals / size


def paired_forward_statistics(reference_variant, full_variant, config=None):
    """Six preregistered comparisons: three windows times two block lengths."""
    protocol = forward_protocol(config)
    result = {"version": FORWARD_PROTOCOL_VERSION, "code_version": FORWARD_PROTOCOL_VERSION,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_hash": protocol["protocol_hash"], "thresholds": protocol,
        "status": "pending", "qualified": False, "publication_allowed": False,
        "reasons": [], "comparisons": {}, "paired_dates": [],
        "estimand": "equal-date arithmetic mean of full minus reference fee-net portfolio daily returns"}
    try:
        left, right = _daily_frame(reference_variant), _daily_frame(full_variant)
        if left.index.tolist() != right.index.tolist():
            raise ValueError("paired strategies require exactly identical verified dates")
        for variant, frame in ((reference_variant, left), (full_variant, right)):
            if variant.get("cash_retained_failures"):
                result["reasons"].append("failed_accounts_retained_as_cash")
            if variant.get("status") in {"failed", "unavailable", "incomplete", "pending"}:
                result["reasons"].append("explicit_account_evidence_not_complete")
            if variant.get("label_maturity_pending"):
                result["reasons"].append("explicit_label_maturity_pending")
            trips = variant.get("metrics", {}).get("completed_round_trips")
            if trips is None or not math.isfinite(float(trips)) or float(trips) < protocol["min_window_round_trips"]:
                result["reasons"].append("insufficient_mature_window_round_trips")
            if any(variant.get(key) is False for key in ("complete", "mature", "window_complete")):
                result["reasons"].append("window_not_complete_or_mature")
            if frame.empty:
                continue
            if "return" not in frame:
                raise ValueError("actual fee-net daily returns are missing")
            values = pd.to_numeric(frame["return"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("actual fee-net daily returns contain unavailable values")
            prior = frame.equity.shift(1)
            prior.iloc[0] = variant.get("window_basis", {}).get("prior_equity") or _original_cash(variant, 100000)
            if not np.allclose(values, (frame.equity / prior - 1).to_numpy(), rtol=1e-8, atol=1e-10):
                raise ValueError("daily return does not match continuous actual equity")
        result["paired_dates"] = left.index.tolist()
        result["paired_sessions"] = len(left)
        result["paired_daily_hash"] = stable_hash({"reference": reference_variant.get("daily", []),
            "full": full_variant.get("daily", [])})
        if _original_cash(reference_variant, 100000) != _original_cash(full_variant, 100000):
            raise ValueError("paired accounts must retain the same original capital")
        if reference_variant.get("account_count") != full_variant.get("account_count"):
            raise ValueError("paired accounts must retain the same frozen universe count")
        if len(left) < protocol["window_sessions"]:
            result["reasons"].append("insufficient_verified_window_sessions")
        elif len(left) != protocol["window_sessions"]:
            result["reasons"].append("paired_data_must_cover_exact_preregistered_window")
        if result["reasons"]:
            result["reasons"] = sorted(set(result["reasons"]))
            return result
        values = pd.to_numeric(right["return"]).to_numpy() - pd.to_numeric(left["return"]).to_numpy()
        normal = NormalDist()
        critical = normal.inv_cdf(1 - protocol["per_comparison_alpha"])
        required_z = normal.inv_cdf(protocol["minimum_power"])
        for block in protocol["block_lengths"]:
            complete = len(values) // block
            detail = {"block_length": block, "complete_blocks": complete,
                "minimum_complete_blocks": protocol["minimum_complete_blocks"],
                "dates": len(values), "alpha": protocol["per_comparison_alpha"],
                "one_sided_confidence": protocol["one_sided_confidence"],
                "normal_critical_value": critical, "status": "pending"}
            result["comparisons"][str(block)] = detail
            if complete < protocol["minimum_complete_blocks"]:
                detail["reason"] = "insufficient_complete_dependence_blocks"
                continue
            means = _bootstrap_means(values, block, protocol["bootstrap_repetitions"], protocol["seed"])
            standard_error = float(means.std(ddof=1))
            lower = float(np.quantile(means, protocol["per_comparison_alpha"]))
            power = normal.cdf(protocol["minimum_detectable_daily_net_improvement"] / standard_error - critical) if standard_error > 0 else 1.
            detail.update(mean_daily_net_improvement=float(values.mean()),
                annualized_arithmetic_mean_net_improvement=float(values.mean() * 252),
                one_sided_lower_ci=lower, bootstrap_standard_error=standard_error,
                power_at_preregistered_mde=float(power),
                achieved_minimum_detectable_annual_improvement=standard_error * (critical + required_z) * 252,
                repetitions=protocol["bootstrap_repetitions"], seed=protocol["seed"],
                power_method="normal_approximation_using_moving_block_bootstrap_standard_error")
            if power < protocol["minimum_power"]:
                detail.update(status="pending", reason="insufficient_power_at_preregistered_effect")
            else:
                meets_effect = float(values.mean()) >= protocol["minimum_detectable_daily_net_improvement"] - 1e-14
                detail["economic_increment_passed"] = meets_effect
                detail["status"] = "passed" if lower > 0 and meets_effect else "failed"
                if not meets_effect:
                    detail["reason"] = "mean_below_registered_economic_increment"
        statuses = {item["status"] for item in result["comparisons"].values()}
        result["status"] = "pending" if "pending" in statuses else "failed" if "failed" in statuses else "passed"
        result["qualified"] = result["status"] == "passed"
        result["reasons"] = sorted({item.get("reason", "nonpositive_one_sided_lower_ci")
            for item in result["comparisons"].values() if item["status"] != "passed"})
    except (ValueError, TypeError, KeyError) as exc:
        result["reasons"].append("unavailable_paired_actual_data: " + str(exc))
    return result
