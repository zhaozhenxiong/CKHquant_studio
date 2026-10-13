"""Causal liquidity qualification on an independently verified market calendar.

Qualification is decided at each session's close.  Missing local observations
remain unknown; they are never interpreted as suspension or repaired with later
bars.  These columns are a population filter for new entries, not model inputs
or an instruction to liquidate existing positions.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES


DEFAULT_POLICY = {
    "lookback_sessions": 20,
    "min_median_amount": 30_000_000.0,
    "complete_verified_window": True,
    "selection": "historical_daily_eligible_union",
}
LIQUIDITY_COLUMNS = (
    "liquidity_pool_eligible", "liquidity_observed_at", "liquidity_median_amount",
    "liquidity_window_start", "liquidity_expected_sessions",
    "liquidity_observed_sessions", "liquidity_qualified_sessions",
    "liquidity_status", "liquidity_reason_codes", "liquidity_policy_hash",
)
RAW_IDENTITY_COLUMNS = (
    "symbol", "date", "open", "high", "low", "close", "volume", "amount",
    "source", "has_trade_price",
)


def liquidity_policy(config=None):
    """Normalize the prior policy; incomplete-window qualification is forbidden."""
    value = config or {}
    if not isinstance(value, dict):
        raise ValueError("liquidity policy must be an object")
    if "liquidity_universe" in value:
        value = value["liquidity_universe"]
        if not isinstance(value, dict):
            raise ValueError("liquidity_universe must be an object")
    elif not set(value).intersection(DEFAULT_POLICY):
        value = {}
    if set(value) - set(DEFAULT_POLICY):
        raise ValueError("unknown liquidity policy fields")
    result = {**DEFAULT_POLICY, **value}
    count = result["lookback_sessions"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("lookback_sessions must be a positive integer")
    amount = result["min_median_amount"]
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount <= 0:
        raise ValueError("min_median_amount must be finite positive CNY")
    result["min_median_amount"] = float(amount)
    if result["complete_verified_window"] is not True:
        raise ValueError("a complete verified liquidity window is required")
    if result["selection"] != DEFAULT_POLICY["selection"]:
        raise ValueError("only historical daily eligible union is supported")
    return result


def _dates(values):
    stamps = pd.to_datetime(list(values), errors="raise")
    dates = [pd.Timestamp(value).date().isoformat() for value in stamps]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("verified calendar must be nonempty, ascending and unique")
    return dates


def daily_eligibility(frame, calendar, policy=None, *, end=None, start=None):
    """Return one decision per verified session, using only its closed prefix.

    ``start`` trims output after calculation, so it never shortens the lookback.
    The caller supplies the independently verified calendar, not pandas weekdays.
    """
    cfg = liquidity_policy(policy)
    dates = _dates(calendar)
    if end is not None:
        cutoff = pd.Timestamp(end).date().isoformat()
        dates = [date for date in dates if date <= cutoff]
    if not dates:
        raise ValueError("no verified session at liquidity cutoff")
    required = set(RAW_IDENTITY_COLUMNS) - {"symbol"}
    if not required <= set(frame):
        raise ValueError("raw liquidity fields missing: " + ",".join(sorted(required - set(frame))))
    attrs = dict(frame.attrs)
    raw = frame.copy()
    raw["_day"] = pd.to_datetime(raw["date"], errors="raise").dt.strftime("%Y-%m-%d")
    raw = raw.loc[raw["_day"] <= dates[-1]].copy()
    # A duplicate decision day is invalid even if one of its records looks valid.
    duplicates = set(raw.loc[raw["_day"].duplicated(keep=False), "_day"])
    raw = raw.drop_duplicates("_day", keep="first").set_index("_day")
    observed = pd.Series(np.array([date in raw.index for date in dates]), index=dates)
    aligned = raw.reindex(dates)
    numeric = aligned[["open", "high", "low", "close", "volume", "amount"]].apply(pd.to_numeric, errors="coerce")
    prices = numeric[["open", "high", "low", "close"]]
    valid_price = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1) &
                   numeric["high"].ge(prices.max(axis=1)) & numeric["low"].le(prices.min(axis=1)))
    units_ok = attrs.get("volume_unit") == "shares" and attrs.get("amount_unit") == "CNY"
    reason_flags = {
        "missing_market_observation_unknown": ~observed,
        "duplicate_market_observation": pd.Series([date in duplicates for date in dates], index=dates),
        "unverified_source": observed & ~aligned["source"].isin(VERIFIED_SOURCES),
        "unverified_trade_price": observed & ~aligned["has_trade_price"].eq(1),
        "invalid_price": observed & ~valid_price,
        "nonpositive_volume": observed & ~(np.isfinite(numeric["volume"]) & numeric["volume"].gt(0)),
        "nonpositive_amount": observed & ~(np.isfinite(numeric["amount"]) & numeric["amount"].gt(0)),
        "unverified_units": pd.Series(bool(not units_ok), index=dates) & observed,
    }
    if "available_at" in aligned:
        available = pd.to_datetime(aligned["available_at"], errors="coerce", utc=True)
        close = pd.to_datetime([date + "T15:00:00+08:00" for date in dates], utc=True)
        reason_flags["observation_not_available_at_close"] = observed & (available.isna() | (available.to_numpy() > close.to_numpy()))
    valid = observed.copy()
    for flags in reason_flags.values():
        valid &= ~flags
    window = cfg["lookback_sessions"]
    observed_count = observed.astype(int).rolling(window, min_periods=1).sum().astype(int)
    qualified_count = valid.astype(int).rolling(window, min_periods=1).sum().astype(int)
    median = numeric["amount"].rolling(window, min_periods=window).median()
    flag_windows = {reason: flags.astype(int).rolling(window, min_periods=1).max().astype(bool).to_numpy()
                    for reason, flags in reason_flags.items()}
    cumulative = observed.astype(int).cumsum().to_numpy()
    observed_values = observed_count.to_numpy()
    qualified_values = qualified_count.to_numpy()
    median_values = median.to_numpy()
    policy_hash = stable_hash(cfg)
    first_output = 0
    if start is not None:
        start_date = pd.Timestamp(start).date().isoformat()
        first_output = int(np.searchsorted(dates, start_date))
    rows = []
    for index in range(first_output, len(dates)):
        date = dates[index]
        reasons = []
        if index + 1 < window:
            reasons.append("insufficient_verified_window")
        if cumulative[index] == 0:
            reasons.append("no_prior_local_observation")
        reasons += [reason for reason, flags in flag_windows.items() if flags[index]]
        complete = index + 1 >= window and qualified_values[index] == window
        if complete and median_values[index] < cfg["min_median_amount"]:
            reasons.append("below_median_amount")
        eligible = bool(complete and median_values[index] >= cfg["min_median_amount"])
        rows.append({"date": date, "liquidity_pool_eligible": eligible,
            "liquidity_observed_at": date + "T15:00:00+08:00",
            "liquidity_median_amount": float(median_values[index]),
            "liquidity_window_start": dates[index - window + 1] if index + 1 >= window else None,
            "liquidity_expected_sessions": window,
            "liquidity_observed_sessions": int(observed_values[index]),
            "liquidity_qualified_sessions": int(qualified_values[index]),
            "liquidity_status": "eligible" if eligible else reasons[0],
            "liquidity_reason_codes": json.dumps(reasons, ensure_ascii=False, separators=(",", ":")),
            "liquidity_policy_hash": policy_hash})
    result = pd.DataFrame(rows, columns=['date', *LIQUIDITY_COLUMNS])
    result.attrs.update(policy=cfg, policy_hash=policy_hash, calendar_hash=stable_hash(dates),
        observation_clock="verified_session_close", unknown_missing_observations="not_inferred_as_suspension")
    return result


def annotate_liquidity(frame, calendar, policy=None, *, end=None, verify_existing=False):
    """Preserve raw rows and features; add or verify the independent pool flags."""
    result = frame.copy()
    result.attrs.update(frame.attrs)
    daily = daily_eligibility(frame, calendar, policy, end=end).set_index("date")
    days = pd.to_datetime(frame.date, errors="raise").dt.strftime("%Y-%m-%d")
    matched = daily.reindex(days.to_numpy())
    if matched["liquidity_policy_hash"].isna().any():
        raise ValueError("raw observation outside verified liquidity calendar/cutoff")
    present = set(LIQUIDITY_COLUMNS).intersection(frame)
    if verify_existing and present:
        if present != set(LIQUIDITY_COLUMNS):
            raise ValueError("partial frozen liquidity qualification")
        for column in LIQUIDITY_COLUMNS:
            left = frame[column].reset_index(drop=True)
            right = matched[column].reset_index(drop=True)
            equal = left.eq(right) | (left.isna() & right.isna())
            if not equal.all():
                raise ValueError("frozen liquidity qualification drift: " + column)
    for column in LIQUIDITY_COLUMNS:
        result[column] = matched[column].to_numpy(copy=True)
    result.attrs.update(liquidity_policy=liquidity_policy(policy), liquidity_policy_hash=stable_hash(liquidity_policy(policy)))
    return result
