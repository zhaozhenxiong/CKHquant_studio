"""Generic immutable research-file and verified-calendar utilities.

Retired model families, training and release routes are not provided.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import numpy as np
import pandas as pd
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
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

def run_path(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_id):
        raise ValueError("无效研究运行标识")
    return ARTIFACT_RUNS_ROOT / run_id

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

def _calendar(calendar_run_id: str) -> tuple[list[str], dict[str, Any]]:
    from my_strategy.services.czsc_research_calendar import load_verified_calendar
    return load_verified_calendar(calendar_run_id, runs_root=ARTIFACT_RUNS_ROOT)
