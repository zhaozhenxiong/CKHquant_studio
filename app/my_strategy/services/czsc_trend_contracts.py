"""Names and explicit configuration for the delivered trend consensus strategy."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from my_strategy.core.paths import MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash

STRATEGY_ID = "trend_consensus"
DISPLAY_NAME = "趋势共识策略 · 有效机会重评"
TREND_TARGET = "next_verified_open_horizon_direction"
TRAIN_KIND = "trend-consensus-train"
TRAINING_POPULATION = "strict_prior_l2_actual_flat_prefix_pre_gate_attempts"
CANDIDATE_POLICY = "valid_original_opportunity_daily_reassessment"
ATTEMPT_IDENTITY = "original_opportunity_signal_day_actual_flat_state_plan_hash"
ECONOMIC_GROUP_IDENTITY = "symbol_original_opportunity_ids"
SPLIT_POLICY = "atomic_original_first_pre_gate_dependency_purge_each_row_real_label_end"
ECONOMIC_WEIGHTING = "selected_segment_symbol_shared_original_atom_connected_components_inverse_rows"
HELD_SPLIT_POLICY = "actual_held_episode_first_observed_group_each_row_real_label_end"
HELD_ECONOMIC_WEIGHTING = "selected_segment_inverse_rows_per_actual_held_episode"
ENTRY_HEADS = frozenset({"entry", "return", "loss"})
HOLDING_HEADS = frozenset({"ma_exit", "exit", "exit_advantage"})
CONFIG_PATH = MY_STRATEGY_ROOT / "configs/czsc_trend_strategy.json"


def head_dependency_policy(head):
    if head in ENTRY_HEADS:
        return {"split_policy": SPLIT_POLICY, "economic_weighting": ECONOMIC_WEIGHTING,
            "economic_group_identity": ECONOMIC_GROUP_IDENTITY, "attempt_identity": ATTEMPT_IDENTITY}
    if head in HOLDING_HEADS:
        return {"split_policy": HELD_SPLIT_POLICY, "economic_weighting": HELD_ECONOMIC_WEIGHTING,
            "economic_group_identity": "symbol_actual_held_episode_id",
            "attempt_identity": "actual_held_episode_signal_close_prefix_state"}
    raise ValueError("unknown trend interval head")


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
    policy = values.get("reassessment") or {}
    expected = {"version": CANDIDATE_POLICY,
        "consumption": "actual_account_single_execution_reservation",
        "quality_rejection_consumes": False, "risk_rejection_consumes": False,
        "missing_model_consumes": False, "broker_rejection_retry": False,
        "expiration": "original_first_available_clock",
        "permanent_invalidation": "original_boundary_break_no_resurrection",
        "holding_sell": "continuous_sell_pass_not_new_event_or_entry_cooldown",
        "training_population": TRAINING_POPULATION,
        "no_prior_l2_entry": "blocked_actual_account_with_forced_entry_label_clone",
        "economic_group": ECONOMIC_GROUP_IDENTITY, "attempt_identity": ATTEMPT_IDENTITY}
    if (values.get("candidate_policy") != CANDIDATE_POLICY
            or values.get("candidate_envelope") != "actual_flat_pre_gate_daily_attempts"
            or values.get("training_population") != TRAINING_POPULATION
            or values.get("split_policy") != SPLIT_POLICY
            or values.get("economic_weighting") != ECONOMIC_WEIGHTING
            or values.get("holding_split_policy") != HELD_SPLIT_POLICY
            or values.get("holding_economic_weighting") != HELD_ECONOMIC_WEIGHTING
            or values.get("consensus", {}).get("version") != STRATEGY_ID
            or any(policy.get(k) != v or isinstance(v, bool) and type(policy.get(k)) is not bool
                   for k, v in expected.items())):
        raise ValueError("有效机会重评与实际账户训练人口契约不匹配")
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
    from my_strategy.services.czsc_liquidity import liquidity_policy
    values["liquidity_universe"] = liquidity_policy(values.get("liquidity_universe"))
    if type(values.get("full_qualified_pool")) is not bool:
        raise ValueError("完整历史流动性股票池参数必须为布尔值")
    batch = values.get("stock_batch_size")
    if type(batch) is not int or not 1 <= batch <= 10000:
        raise ValueError("股票准备批大小必须为1到10000的整数")
    return values


def config_hash(config):
    return stable_hash(config)


def code_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_training_config(overrides=None):
    """Public fresh training cannot weaken the current model qualifications.

    Low-level fit functions remain testable with isolated small fixtures. Every
    real full training and loaded application package passes these floors.
    """
    config = strategy_config(overrides)
    if config["full_qualified_pool"] is not True:
        raise ValueError("当前训练必须完整执行冻结历史流动性合格股票并集")
    if config['liquidity_universe']['lookback_sessions'] < 20:
        raise ValueError('当前流动性资格窗口不得少于20个核验交易日')
    if config['liquidity_universe']['min_median_amount'] < 30000000.:
        raise ValueError('当前流动性资格成交额中位数不得低于3000万元')
    for key, floor in (("min_train_rows", 100), ("min_calibration_rows", 30)):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < floor:
            raise ValueError(f"当前训练资格 {key} 不得低于 {floor}")
    from my_strategy.services.czsc_trend_model import trend_config
    from my_strategy.services.czsc_trend_labels import label_config
    trend = trend_config(config)
    for key, floor in (("min_train_per_class", 100), ("min_calibration_per_class", 30), ("min_policy_per_class", 30)):
        if trend[key] < floor:
            raise ValueError(f"当前三类资格 {key} 不得低于 {floor}")
    target = label_config(config)
    if target["min_net_return"] < .08 or target["min_r_multiple"] < 2.:
        raise ValueError("当前高收益资格不得低于费用后8%且2R")
    if target["min_risk_fraction"] < .001 or target["max_risk_fraction"] > .15:
        raise ValueError("当前初始风险资格不得超出0.1%至15%范围")
    if target["max_holding_sessions"] > 60 or config["account"]["max_holding_sessions"] > 60:
        raise ValueError("当前退出意图上限不得超过60个核验交易日")
    return config
