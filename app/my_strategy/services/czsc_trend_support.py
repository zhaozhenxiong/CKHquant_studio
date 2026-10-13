"""Current pipeline numerical helpers without legacy model routing."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from my_strategy.core.run_context import stable_hash


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _calendar_quality(features: pd.DataFrame, raw: pd.DataFrame, market_dates: list[str]) -> pd.DataFrame:
    """Keep sparse rows visible while vetoing windows with unverified missing sessions."""
    dates = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d").tolist()
    following = dict(zip(market_dates[:-1], market_dates[1:]))
    known = set(market_dates)
    gaps = np.array([day not in known or (i > 0 and following.get(dates[i - 1]) != day) for i, day in enumerate(dates)])
    window = int(features.attrs.get("quality_window_bars", 120))
    affected = pd.Series(gaps).rolling(window, min_periods=1).max().astype(bool).to_numpy()
    result = features.copy()
    for i in np.flatnonzero(affected):
        reasons = list(result.iloc[i]["reason_codes"])
        result.iat[i, result.columns.get_loc("reason_codes")] = list(dict.fromkeys([*reasons, "calendar_missing_market_bar_window"]))
    result.loc[affected, "input_eligible"] = False
    if "model_input_eligible" in result:
        model_window = int(features.attrs.get("model_quality_window_bars", 120))
        model_affected = pd.Series(gaps).rolling(model_window, min_periods=1).max().astype(bool).to_numpy()
        result.loc[model_affected, "model_input_eligible"] = False
        for i in np.flatnonzero(model_affected):
            codes = list(result.iloc[i]["model_reason_codes"])
            result.iat[i, result.columns.get_loc("model_reason_codes")] = list(dict.fromkeys([*codes, "calendar_missing_market_bar_window"]))
    for column in ("rule_czsc_buy", "rule_czsc_sell", "rule_price_volume_buy", "rule_price_volume_sell", "rule_buy", "rule_sell"):
        result.loc[affected, column] = False
    result.attrs.update(calendar_hash=stable_hash(market_dates), calendar_gap_policy="veto_trailing_quality_window_preserve_rows")
    return result


def _calendar(calendar_run_id=None):
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    from my_strategy.services.czsc_research_calendar import load_verified_calendar
    candidates = [calendar_run_id] if calendar_run_id else [p.parent.parent.name for p in sorted(ARTIFACT_RUNS_ROOT.glob("*/reports/calendar.json"), key=lambda p: p.stat().st_mtime, reverse=True)]
    for identity in candidates:
        try:
            dates, metadata = load_verified_calendar(identity, runs_root=ARTIFACT_RUNS_ROOT)
            return dates, identity
        except (OSError, ValueError, KeyError):
            if calendar_run_id:
                raise
    raise ValueError("当前训练需要完整已核验交易日历")

# Actual raw retrieval contracts; legacy adjusted rows remain rejected.
VERIFIED_SOURCES = frozenset({"tushare", "tushare:daily", "fuyao_unadjusted_v1", "baostock", "eastmoney_unadjusted_v1", "sina_unadjusted_v1", "tencent_star_unadjusted_v1"})


def resolve_frozen_asset(value, *, runs_root=None):
    """Resolve saved asset provenance against current storage after relocation.

    Serialized paths remain immutable. A path containing runs/<identity>/...
    always uses that contained current location, even if the old path exists.
    Explicit standalone paths are supported by isolated low-level producers;
    callers still verify the frozen content hash before reading.
    """
    from pathlib import PureWindowsPath
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    raw = str(value)
    root = Path(runs_root or ARTIFACT_RUNS_ROOT).resolve()
    parts = PureWindowsPath(raw).parts if "\\" in raw else Path(raw).parts
    if ".." in parts:
        raise ValueError("frozen asset path escapes storage")
    if "runs" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("runs")
        tail = parts[index + 1:]
        if len(tail) < 2:
            raise ValueError("frozen asset path requires a run and file")
        candidate = root.joinpath(*tail)
    elif not Path(raw).is_absolute() and not PureWindowsPath(raw).drive:
        candidate = root / raw
    else:
        candidate = Path(raw)
    resolved = candidate.resolve()
    if "runs" in parts or not Path(raw).is_absolute() and not PureWindowsPath(raw).drive:
        if not resolved.is_relative_to(root) or any(p.is_symlink() for p in (candidate, *candidate.parents) if p != root.parent):
            raise ValueError("frozen asset path escapes current storage")
    if not resolved.is_file():
        raise FileNotFoundError("frozen asset missing from current storage: " + str(candidate))
    return resolved
