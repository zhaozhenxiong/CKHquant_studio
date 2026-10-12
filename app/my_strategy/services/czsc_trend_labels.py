"""Fee-inclusive episode outcomes; none of these fields are model inputs."""
from __future__ import annotations

import math
import numpy as np

from my_strategy.core.run_context import stable_hash
from my_strategy.execution.cost_model import CostModel
from my_strategy.services.czsc_backtest import _tick

ENTRY_LABEL_VERSION = "czsc_trend_actual_episode_high_return"
EXIT_LABEL_VERSION = "czsc_trend_remaining_value_exit_advantage"
DEFAULT_HIGH_RETURN_CONFIG = {"min_net_return": .08, "min_r_multiple": 2.,
    "max_holding_sessions": 60, "min_risk_fraction": .001,
    "max_risk_fraction": .15, "low_return_ceiling": .02}


def label_config(config=None):
    values = dict(DEFAULT_HIGH_RETURN_CONFIG)
    incoming = (config or {}).get("high_return", config or {})
    values.update({k: incoming[k] for k in values if k in incoming})
    if (not all(math.isfinite(float(v)) for v in values.values())
            or not 0 < values["min_net_return"] < 1 or values["min_r_multiple"] <= 0
            or not 0 < values["min_risk_fraction"] < values["max_risk_fraction"] < 1
            or int(values["max_holding_sessions"]) != values["max_holding_sessions"]
            or values["max_holding_sessions"] < 1 or values["low_return_ceiling"] < 0):
        raise ValueError("invalid frozen high-return protocol")
    values["max_holding_sessions"] = int(values["max_holding_sessions"])
    return values


def episode_outcome(ledger, *, stop_price, cost_model: CostModel, config=None):
    """Aggregate actual debit/credits, freezing initial risk at the actual fill.

    Single-entry risk policy only. REDUCE proceeds accumulate until inventory is
    zero. A still-open episode never becomes a loss or a negative high label.
    """
    settings = label_config(config)
    buys = [row for row in ledger if row["action"] == "BUY"]
    result = {"label": np.nan, "positive_return_label": np.nan,
        "label_available": False, "label_end": None, "label_available_at": None,
        "label_reason": "entry_unfilled", "outcome_class": None,
        "net_return": np.nan, "net_profit": np.nan, "r_multiple": np.nan,
        "risk_cash": np.nan, "risk_fraction": np.nan, "risk_available": False,
        "entry_cash": 0., "exit_cash": 0., "fees": 0., "low_return": False,
        "entry_date": None, "exit_date": None, "entry_shares": 0, "open_shares": 0}
    if not buys:
        return result
    if len(buys) != 1:
        raise ValueError("episode labels prohibit reinvestment or added entries")
    buy = buys[0]
    shares = int(buy["shares"])
    paid = -float(buy["cash_flow"])
    received, inventory = 0., shares
    for row in ledger:
        if row["action"] != "BUY":
            inventory -= int(row["shares"])
            received += float(row["cash_flow"])
            if inventory < 0:
                raise ValueError("episode exits exceed actual shares")
    result.update(entry_cash=paid, exit_cash=received,
        fees=sum(float(r["fee"]) for r in ledger), entry_date=buy["date"],
        entry_shares=shares, open_shares=inventory, label_reason="episode_unclosed")
    if stop_price is not None and math.isfinite(float(stop_price)) and 0 < float(stop_price) < float(buy["price"]):
        stop_fill = _tick(cost_model.execution_price("SELL", float(stop_price)))
        stopped = shares * stop_fill - cost_model.fees("SELL", shares * stop_fill)
        risk_cash = paid - stopped
        fraction = risk_cash / paid
        result.update(risk_cash=risk_cash, risk_fraction=fraction,
            risk_available=bool(settings["min_risk_fraction"] <= fraction <= settings["max_risk_fraction"]))
    if inventory:
        return result
    net, end = received - paid, ledger[-1]["date"]
    result.update(net_profit=net, net_return=net / paid, exit_date=end,
        label_end=end, label_available_at=end + "T09:30:00+08:00",
        positive_return_label=float(net > 0), label_reason="risk_identity_unavailable")
    if result["risk_available"]:
        ratio = net / result["risk_cash"]
        high = result["net_return"] >= settings["min_net_return"] and ratio >= settings["min_r_multiple"]
        result.update(r_multiple=ratio, label=float(high), label_available=True,
            label_reason="available", outcome_class="high" if high else "loss" if net <= 0 else "ordinary",
            low_return=bool(0 < result["net_return"] < settings["low_return_ceiling"]))
    return result


def exit_outcome(*, forced, continued_proceeds, continued_end, shares, close, config=None):
    result = {"label": np.nan, "label_available": False, "label_end": None,
        "label_available_at": None, "label_reason": "continuation_unclosed",
        "exit_advantage": np.nan, "forced_net_proceeds": np.nan,
        "continuation_net_proceeds": continued_proceeds if continued_end else np.nan,
        "forced_exit_date": forced.get("date"), "continuation_exit_date": continued_end,
        "forced_rejection_count": len(forced.get("rejections", []))}
    if not continued_end:
        return result
    if not forced.get("date"):
        result["label_reason"] = "forced_exit_unclosed"
        return result
    denominator = int(shares) * float(close)
    if denominator <= 0:
        raise ValueError("exit advantage requires actual positive marked inventory")
    delta = (float(forced["proceeds"]) - float(continued_proceeds)) / denominator
    end = max(forced["date"], continued_end)
    margin = float((config or {}).get("high_return", {}).get("exit_advantage_margin", .002))
    result.update(label=float(delta > margin), label_available=True, label_end=end,
        label_available_at=end + "T09:30:00+08:00", label_reason="available",
        exit_advantage=delta, exit_advantage_margin=margin, forced_net_proceeds=float(forced["proceeds"]))
    return result


def target_contract(*, head, config, market_dates, source_snapshot_hash,
                    feature_schema_hash, continuation_policy_hash, initial_cash, stage, candidate_policy_hash):
    if head not in {"entry", "ma_exit", "exit", "exit_advantage"}:
        raise ValueError("unknown trend interval head")
    value = {"family": "czsc_trend_consensus", "model_head": head,
        "label_version": ENTRY_LABEL_VERSION if head == "entry" else EXIT_LABEL_VERSION,
        "target": "actual_complete_fee_net_roundtrip_high_return_and_initial_risk" if head == "entry" else
                  "remaining_net_sell_proceeds_exit_next_open_minus_frozen_continue_over_current_market_value",
        "source": "union", "stage": stage, "high_return": label_config(config),
        "exit_advantage_margin": float(config["high_return"].get("exit_advantage_margin", .002)),
        "source_snapshot_hash": source_snapshot_hash, "feature_schema_hash": feature_schema_hash,
        "pipeline_contract_hash": config.get("pipeline_contract_hash"),
        "calendar_hash": stable_hash(market_dates), "initial_cash": float(initial_cash),
        "execution_config_hash": stable_hash(config["account"]), "candidate_policy_hash": candidate_policy_hash,
        "continuation_policy_hash": continuation_policy_hash, "strategy_hash": stable_hash(config),
        "population": "actual_flat_candidate_clone" if head == "entry" else "actual_holding_remaining_inventory_pair",
        "admission": "same_runtime_base_direction_BUY_and_actual_flat_no_pending_SELL" if head == "entry" else "actual_held_Broker_state_only",
        "opportunity_identity": "stable_original_event_ids_and_policy_no_daily_regeneration",
        "maturity": "actual_complete_liquidation" if head == "entry" else "both_complete_liquidations_later_date",
        "execution": "next_verified_open_guarded_Broker_partial_pending", "publication_allowed": False}
    return {**value, "contract_hash": stable_hash(value)}
