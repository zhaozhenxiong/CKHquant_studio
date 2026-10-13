"""Paired date-block uncertainty from saved outcomes, without model routing."""
from __future__ import annotations
import math
from typing import Any, Sequence
import numpy as np
import pandas as pd

def _date(value):
    return pd.Timestamp(value).date().isoformat()

def date_block_bootstrap(frame: pd.DataFrame, value_column: str, *, date_column: str = "date",
                         block_length: int = 20, repetitions: int = 1000, seed: int = 42,
                         confidence: float = 0.95, dependence_horizon: int = 10) -> dict[str, Any]:
    """CI of the equal-date mean, keeping all same-date securities together.

    For a paired comparison pass per-stock/date differences, not separately
    resampled strategies. Contiguous moving date blocks preserve dependence.
    This utility does not calculate or replace actual ledger return/drawdown.
    """
    if block_length < dependence_horizon + 1 or repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("block must cover holding dependence; invalid repetitions/confidence")
    data = frame[[date_column, value_column]].copy()
    data[date_column] = data[date_column].map(_date)
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    data.loc[~np.isfinite(data[value_column]), value_column] = np.nan
    daily = data.groupby(date_column)[value_column].mean().dropna().sort_index()
    values = daily.to_numpy(dtype=float)
    if len(values) < 2 * block_length:
        raise ValueError("insufficient dates for at least two full dependence blocks")
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for index in range(repetitions):
        starts = generator.integers(0, len(values) - block_length + 1, size=math.ceil(len(values) / block_length))
        sample = np.concatenate([values[start:start + block_length] for start in starts])[:len(values)]
        means[index] = sample.mean()
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(means, [tail, 1 - tail])
    return {"estimand": "equal-date mean of same-date cross-sectional means", "mean": float(values.mean()),
            "lower": float(lower), "upper": float(upper), "confidence": confidence, "dates": len(values), "rows": len(frame),
            "date_start": daily.index[0], "date_end": daily.index[-1], "block_length": block_length, "repetitions": repetitions, "seed": seed}

def bootstrap_sensitivity(frame: pd.DataFrame, value_column: str, *, block_lengths: Sequence[int] = (11, 20, 40),
                          **kwargs: Any) -> list[dict[str, Any]]:
    return [date_block_bootstrap(frame, value_column, block_length=length, **kwargs) for length in block_lengths]
