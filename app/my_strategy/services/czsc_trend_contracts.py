"""Names and explicit configuration for the delivered trend consensus strategy."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from my_strategy.core.paths import MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash

STRATEGY_ID = "czsc_trend_consensus"
DISPLAY_NAME = "趋势共识策略"
TREND_TARGET = "next_verified_open_horizon_direction"
TRAIN_KIND = "czsc-trend-consensus-train"
CONFIG_PATH = MY_STRATEGY_ROOT / "configs/czsc_trend_strategy.json"


def _merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def strategy_config(overrides=None):
    values = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _merge(values, overrides or {})
    if values.get("version") != STRATEGY_ID:
        raise ValueError("趋势共识策略身份不匹配")
    weights = values["consensus"]["source_weights"]
    quality = values["consensus"]["quality_weight"]
    amounts = [weights.get(k) for k in ("ma", "structure", "wyckoff")] + [quality]
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x < 0 for x in amounts):
        raise ValueError("Agent权重必须为有限非负数")
    if not math.isclose(sum(amounts), 1.0, abs_tol=1e-10):
        raise ValueError("四路Agent权重合计必须为1")
    horizon = values["trend"]["horizon_sessions"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("趋势预测周期必须为正整数核验交易日")
    cash = values["account"]["initial_cash"]
    if not math.isfinite(float(cash)) or cash <= 0:
        raise ValueError("研究账户初始资金必须为正数")
    for key in ("buy_base_threshold", "buy_total_threshold", "sell_threshold", "margin", "min_source_strength", "high_return_threshold"):
        value = values["consensus"][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"仲裁参数{key}必须在0到1之间")
    for key in ("min_independent", "cooldown", "evidence_valid_bars", "signal_valid_bars"):
        value = values["consensus"][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key != "cooldown" else 0):
            raise ValueError(f"仲裁参数{key}必须为合法整数")
    values["initial_cash"] = float(cash)
    values["display_name"] = DISPLAY_NAME
    return values


def config_hash(config):
    return stable_hash(config)


def code_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
