"""Current ordered market input semantics, independent from retired model identities."""
from __future__ import annotations
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES
MA_TREND_COLUMNS = (
    "ma5_bias", "ma10_bias", "ma20_bias", "ma60_bias",
    "ma5_slope5", "ma10_slope5", "ma20_slope5", "ma60_slope5",
    "ma_alignment", "close_above_ma20", "ret_1", "ret_5", "ret_20",
    "atr14_ratio", "volatility20", "drawdown20", "drawdown60",
    "ma5_ma10_gap", "ma10_ma20_gap", "ma20_ma60_gap",
)

STRUCTURE_COLUMNS = (
    "bi1_direction", "bi1_return", "bi1_length", "bi1_volume_ratio",
    "bi2_direction", "bi2_return", "bi2_length",
    "bi3_direction", "bi3_return", "bi3_length", "bi1_power_ratio",
    "confirmed_age", "zone_low_distance", "zone_high_distance", "zone_width_ratio",
    "weekly_return", "weekly_direction", "monthly_return", "monthly_direction",
    "weekly_bi_direction", "monthly_bi_direction", "finished_bi_count",
)
def get_feature_profile(name="ma_trend"):
    if name == "ma_trend":
        columns = MA_TREND_COLUMNS
        semantics = {"input": "trailing_SMA5_10_20_60", "quality_bars": 120, "active_bars": 60,
            "source_qualification": sorted(VERIFIED_SOURCES), "target": "input_features_only"}
    elif name == "confirmed_structure":
        columns = STRUCTURE_COLUMNS
        semantics = {"input": "finished_pens_confirmed_zones_closed_higher_periods", "minimum_finished_pens": 3,
            "optional_missing": "training_only_imputation", "target": "input_features_only"}
    else:
        raise ValueError("unsupported current market input profile")
    body = {"profile": name, "columns": list(columns), "semantics": semantics}
    return {"name": name, "version": name, "schema_hash": stable_hash(body), "columns": list(columns),
            "strategy_version": "trend_consensus"}


def bind_feature_profile(frame, profile_name: str | None = None):
    """Copy model metadata without changing rows, eligibility or trading rules."""
    profile = get_feature_profile(profile_name)
    if any(column not in frame for column in profile["columns"]):
        raise ValueError("feature frame is missing profile columns")
    result = frame.copy()
    result.attrs = dict(frame.attrs)
    result.attrs.update(base_feature_version=frame.attrs.get("base_feature_version", frame.attrs.get("feature_version")),
                        base_schema_hash=frame.attrs.get("base_schema_hash", frame.attrs.get("schema_hash")),
                        feature_version=profile["version"], schema_hash=profile["schema_hash"],
                        feature_columns=list(profile["columns"]), feature_profile=profile["name"])
    return result

def prepare_structure_inputs(frame):
    """Derive an independent confirmed-structure model copy; rules stay unchanged.

    Optional zones/higher-period structures may be missing and are imputed only
    by training preprocessing. The three daily finished pens must exist, be
    finite and already observable. The current structure inputs carry their own causal schema.
    """
    import numpy as np
    import pandas as pd
    result = bind_feature_profile(frame, "confirmed_structure")
    required = {"date", "input_eligible", "structure_confirmed_at", "finished_bi_count"}
    if not required <= set(result):
        raise ValueError("structure eligibility metadata missing")
    now = pd.to_datetime(result.get("available_at", result["date"]), utc=True, errors="coerce", format="mixed")
    confirmed = pd.to_datetime(result["structure_confirmed_at"], utc=True, errors="coerce", format="mixed")
    core = [f"bi{i}_{field}" for i in (1, 2, 3) for field in ("direction", "return", "length")]
    numeric = result[core].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    counts = pd.to_numeric(result["finished_bi_count"], errors="coerce").to_numpy(dtype=float)
    original_eligible = result["input_eligible"].fillna(False).astype(bool).to_numpy()
    masks = {"structural_input_ineligible": ~original_eligible,
             "insufficient_confirmed_structure": ~np.isfinite(numeric).all(axis=1) | ~np.isfinite(counts) | (counts < 3),
             "structure_not_observable": (now.isna() | confirmed.isna() | (confirmed > now)).to_numpy()}
    future = np.zeros(len(result), dtype=bool)
    for field in ("zone_confirmed_at", "weekly_available_at", "monthly_available_at"):
        if field in result:
            observed = pd.to_datetime(result[field], utc=True, errors="coerce", format="mixed")
            future |= (result[field].notna() & (observed.isna() | (observed > now))).to_numpy()
    masks["future_structure_metadata"] = future
    reasons = [[] for _ in range(len(result))]
    original_reasons = result.get("reason_codes")
    bad = np.logical_or.reduce(list(masks.values()))
    for i in np.flatnonzero(bad):
        current = list(original_reasons.iloc[i]) if original_reasons is not None and not original_eligible[i] else []
        current.extend(key for key, mask in masks.items() if mask[i])
        reasons[i] = list(dict.fromkeys(current))
    result["structure_reason_codes"] = reasons
    result["structure_input_eligible"] = ~bad
    return result
