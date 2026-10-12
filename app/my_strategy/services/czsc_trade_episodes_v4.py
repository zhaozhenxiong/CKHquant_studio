"""Independent real Broker paths and flat-state candidate counterfactuals.

The immutable core Broker executes every branch. Candidate labels and continuous
account episodes are separate populations. Outcome reads never create an X row.
"""
from __future__ import annotations

from collections import Counter
import copy
import math
import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.execution.broker import BrokerSimulator
from my_strategy.execution.cost_model import CostModel
from my_strategy.execution.order import Order
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions, _guard, _tick
from my_strategy.services.czsc_high_return_features_v4 import (
    ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS, ENTRY_SCHEMA_HASH, EXIT_SCHEMA_HASH,
    MA_ENTRY_SCHEMA_HASH, MA_EXIT_SCHEMA_HASH, bind_frame, event_prefix_rows,
    entry_feature_row, holding_feature_row,
)
from my_strategy.services.czsc_high_return_labels_v4 import (
    label_config, episode_outcome, exit_outcome, target_contract,
)
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES

EPISODE_VERSION = "czsc_native_actual_broker_episode_v4"
SOURCES = ("ma", "structure", "wyckoff", "union")


def _settings(config):
    if config is None:
        return strategy_config()
    value = config.get("strategy", config)
    if "czsc_version" in value:
        return strategy_config(value)
    base = strategy_config()
    base.update({key: value[key] for key in base if key in value})
    return strategy_config(base)


def _cost(settings):
    return CostModel(**{k: float(settings["execution"][k]) for k in
        ("commission", "stamp_tax", "min_commission", "slippage")}, version="flat_v1", impact_tiers=())


def _check_lineage(points):
    for event in points.get("events", []):
        if event.get("retrospective"):
            raise ValueError("retrospective points cannot create behavior episodes")
        lineage = event.get("model_lineage") or {}
        if lineage.get("retrospective"):
            raise ValueError("retrospective model lineage")
        day = event["date"]
        for key in ("available_at", "train_label_end", "label_cutoff"):
            value = lineage.get(key)
            if value and (pd.to_datetime(value, utc=True) > pd.Timestamp(day + "T15:00:00+08:00") if key == "available_at"
                    else pd.Timestamp(value).date().isoformat() >= day):
                raise ValueError("point model or labels were unavailable at decision")


def _max_exit_callback(market_index, maximum, contract_hash, *, collect=None, extra_policy=None,
        contract_initial_cash=None):
    def callback(snapshot):
        if collect is not None:
            collect(copy.deepcopy(snapshot))
        age = market_index[snapshot["date"]] - market_index[snapshot["entry_date"]] + 1
        if age >= maximum:
            return {"head": "holding_exit", "contract_hash": contract_hash, "probability": 1.,
                "request_exit": True, "apply": True, "research_arm": True,
                "status": "frozen_max_holding_sessions"}
        if extra_policy:
            observed = copy.deepcopy(snapshot)
            if contract_initial_cash is not None:
                # Candidate cash clones a historical account's available cash;
                # the trained capital identity remains the original 100000.
                observed["initial_cash"] = float(contract_initial_cash)
            return extra_policy(observed)
        return None
    return callback


def _fresh_policy(policy):
    """Every execution branch owns its prefix state; factories never share it.

    Plain callbacks must be stateless. A callback storing historical prefixes
    exposes new_replay() and receives a new instance for each actual execution,
    including a candidate clone's extended liquidation replay.
    """
    if policy is None:
        return None
    selected = policy.new_replay() if hasattr(policy, "new_replay") else policy
    if not callable(selected):
        raise ValueError("frozen exit policy replay factory must return a callback")
    return selected


def _liquidate(snapshot, raw_rows, seasoned, market_next, settings):
    """Persistent next-verified-open branch, including partial sells and fees."""
    broker = BrokerSimulator(initial_cash=float(snapshot["cash"]), cost_model=_cost(settings))
    symbol = snapshot["symbol"]
    broker.positions[symbol] = int(snapshot["shares"])
    broker.avg_cost[symbol] = float(snapshot["actual_cost_per_share"])
    rejects = []
    execution = settings["execution"]
    for index in range(int(snapshot["bar_index"]) + 1, len(raw_rows)):
        row, previous = raw_rows[index], raw_rows[index - 1]
        day = pd.Timestamp(row.date).date().isoformat()
        previous_day = pd.Timestamp(previous.date).date().isoformat()
        reason = _guard(symbol, row, previous, "SELL", int(seasoned[index]), execution)
        if market_next.get(previous_day) != day:
            reason = "missing_next_market_bar"
        if any(bar.source not in VERIFIED_SOURCES for bar in (row, previous)):
            reason = "unverified_research_source"
        if day == snapshot["entry_date"]:
            reason = "t_plus_one"
        capacity = int(float(previous.volume) * float(execution["max_volume_participation"])) // 100 * 100
        shares = min(broker.shares(symbol), capacity)
        if not reason and shares <= 0:
            reason = "insufficient_prior_liquidity"
        if reason:
            rejects.append({"date": day, "reason": reason}); continue
        price = _tick(broker.cost_model.execution_price("SELL", float(row.open)))
        fill = broker.submit(Order(date=day, symbol=symbol, side="SELL", shares=shares,
            price=price, target_weight=0., signal_date=snapshot["date"],
            reason="episode_forced_exit_label", applied_slippage=broker.cost_model.slippage),
            portfolio_prices={symbol: float(row.open)})
        if fill is None:
            rejects.append({"date": day, "reason": "broker_rejected"})
        if broker.shares(symbol) == 0:
            ledger = broker.ledger.to_frame().to_dict("records")
            return {"date": day, "proceeds": sum(float(r["cash_flow"]) for r in ledger),
                "ledger": ledger, "rejections": rejects}
    return {"date": None, "proceeds": None, "ledger": broker.ledger.to_frame().to_dict("records"), "rejections": rejects}


def _episodes(account, *, source, symbol, cost, config, market_index, policy_hash):
    """Derive identities and cash outcomes solely from actual inventory changes."""
    held, current, result = 0, [], []
    plans = {row["entry_date"]: row for row in account.get("holding_snapshots", [])}
    def completed(ledger):
        row = _episode_row(ledger, plans, source, symbol, cost, config, market_index, policy_hash)
        snapshots = [s for s in account.get("holding_snapshots", []) if s["entry_date"] == ledger[0]["date"]]
        basis = -float(ledger[0]["cash_flow"]) / int(ledger[0]["shares"])
        highs = [float(s["bar"]["high"]) for s in snapshots]
        lows = [float(s["bar"]["low"]) for s in snapshots]
        prices = [float(fill["price"]) for fill in ledger]
        peak, trough = max([basis, *highs, *prices]), min([basis, *lows, *prices])
        row.update(max_favorable_excursion=peak / basis - 1,
            max_adverse_excursion=trough / basis - 1,
            final_peak_giveback=(peak - prices[-1]) / basis if row["exit_date"] else np.nan,
            entry_fee_ratio=float(ledger[0]["fee"]) / -float(ledger[0]["cash_flow"]),
            outcome_scope="whole_episode_retrospective_never_prediction_input")
        return row
    for row in account["ledger"]:
        if row["action"] == "BUY":
            if held:
                raise ValueError("single-entry episode unexpectedly added shares")
            current = [row]; held = int(row["shares"])
        else:
            if not held:
                raise ValueError("sell without episode inventory")
            current.append(row); held -= int(row["shares"])
        if current and held == 0:
            result.append(completed(current))
            current = []
    if current:
        result.append(completed(current))
    return result


def _episode_row(ledger, snapshots, source, symbol, cost, config, market_index, policy_hash):
    first = ledger[0]; state = snapshots.get(first["date"], {})
    plan = state.get("plan_id")
    identity = stable_hash({"version": EPISODE_VERSION, "source": source, "symbol": symbol,
        "entry_date": first["date"], "plan_id": plan, "policy": policy_hash})
    outcome = episode_outcome(ledger, stop_price=state.get("stop_price"), cost_model=cost, config=config)
    end = outcome["exit_date"]
    return {"episode_id": identity, "source": source, "symbol": symbol, "plan_id": plan,
        "signal_date": first.get("signal_date"), "continuation_policy_hash": policy_hash,
        "ledger_sha256": stable_hash(ledger), "ledger_rows": len(ledger),
        "holding_sessions": market_index[end] - market_index[first["date"]] + 1 if end else None,
        "population": "continuous_reference_account", **outcome}


def build_episode_dataset(features, market_dates, config=None, *, points=None, start=None,
        initial_cash=100000., source_snapshot_hash="", sources=SOURCES,
        candidate_callback=None, reference_callback=None, collect=True,
        record_callback=None, exit_policies=None, exit_policy_hashes=None):
    """Return candidate entry, daily exit, actual episode and event frames.

    callbacks: candidate_callback(record, cloned_account),
    reference_callback(source, symbol, continuous_account), record_callback(kind,row).
    collect=False streams rows without retaining accounts/frames. A supplied exit
    policy must be frozen and its hash supplied separately; defaults are v0 rules.
    """
    from my_strategy.services.czsc_signal_points_v4 import build_signal_points, make_point_decisions
    settings = _settings(config); protocol = label_config(config)
    calendar = [pd.Timestamp(d).date().isoformat() for d in market_dates]
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError("episodes require an ascending unique verified calendar")
    if not math.isfinite(float(initial_cash)) or initial_cash <= 0:
        raise ValueError("invalid episode initial cash")
    if any(s not in SOURCES for s in sources):
        raise ValueError("unknown episode source")
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "amount", "source", "has_trade_price"}
    if features.empty or not required <= set(features):
        raise ValueError("episodes require actual raw OHLCV and source metadata")
    frame = features.copy(); frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    if frame.duplicated(["symbol", "date"]).any() or not set(frame.date).issubset(calendar):
        raise ValueError("duplicate or non-calendar market rows")
    point_config = {**settings, **(config or {})}
    if points is None:
        point_sets = [build_signal_points(group.reset_index(drop=True), point_config)
            for _, group in frame.groupby("symbol", sort=False)]
        all_points = {**point_sets[0], "events": [e for item in point_sets for e in item["events"]],
            "candidates": [c for item in point_sets for c in item["candidates"]]}
    else:
        all_points = points
    _check_lineage(all_points)
    market_index = {d: i for i, d in enumerate(calendar)}
    market_next = dict(zip(calendar[:-1], calendar[1:]))
    entries, exits, episodes, accounts, contracts = [], [], [], {}, {}
    coverage = Counter()
    def emit(kind, row):
        coverage[kind + "_rows"] += 1
        if record_callback:
            record_callback(kind, row)
        if collect:
            {"entries": entries, "exits": exits, "episodes": episodes}[kind].append(row)
    for symbol, group in frame.groupby("symbol", sort=False):
        raw = group.reset_index(drop=True)
        rows = raw.to_dict("records"); tuple_rows = list(raw.itertuples(index=False))
        local = {**all_points, "events": [e for e in all_points.get("events", []) if e["symbol"] == symbol],
            "candidates": [c for c in all_points.get("candidates", []) if c["symbol"] == symbol]}
        prefixes = event_prefix_rows(raw, local["events"], calendar)
        by_date = {r["date"]: i for i, r in enumerate(rows)}
        active = np.array([r.volume > 0 and r.amount > 0 and r.has_trade_price == 1 and r.source in VERIFIED_SOURCES for r in tuple_rows])
        seasoned = np.r_[0, np.cumsum(active[:-1])]
        valid_indices = np.flatnonzero(active)
        first_day = start or rows[0]["date"]
        for source in sources:
            policy_hash = stable_hash({"version": EPISODE_VERSION, "source": source,
                "points_config": all_points.get("config_hash"), "strategy": settings, "protocol": protocol,
                "exit_policy": (exit_policy_hashes or {}).get(source, "source_rule_exit_v0")})
            if (exit_policies or {}).get(source) and not (exit_policy_hashes or {}).get(source):
                raise ValueError("a frozen external exit policy requires its identity hash")
            contract_args = dict(source=source, config=config or {}, market_dates=calendar,
                source_snapshot_hash=source_snapshot_hash, continuation_policy_hash=policy_hash, initial_cash=initial_cash,
                strategy_settings=settings)
            source_contracts = {"entry": target_contract(head="entry", feature_schema_hash=ENTRY_SCHEMA_HASH, **contract_args),
                "exit": target_contract(head="exit", feature_schema_hash=EXIT_SCHEMA_HASH, **contract_args),
                "ma_entry": target_contract(head="ma_entry", feature_schema_hash=MA_ENTRY_SCHEMA_HASH, **contract_args),
                "ma_exit": target_contract(head="exit", feature_schema_hash=MA_EXIT_SCHEMA_HASH, **contract_args)}
            contracts[source] = source_contracts
            decisions = make_point_decisions(raw, local, source=source, config=point_config, start=first_day)
            clone_decisions = make_point_decisions(raw, local, source=source, config=point_config,
                start=first_day, include_unselected=True)
            reference_path_hash, candidate_path_hash = stable_hash(decisions), stable_hash(clone_decisions)
            if len(decisions) != len(raw) or [r["date"] for r in decisions] != list(raw.date):
                raise ValueError("point decisions must be aligned to actual bars")
            extra = (exit_policies or {}).get(source)
            actual = execute_decisions(symbol, raw, decisions, first_day, initial_cash, settings,
                "point-episode-reference-" + source, market_dates=calendar, verified_sources=VERIFIED_SOURCES,
                entry_policy="risk", exit_policy=_max_exit_callback(market_index,
                    protocol["max_holding_sessions"], policy_hash, extra_policy=_fresh_policy(extra)))
            if reference_callback:
                reference_callback(source, symbol, actual)
            if collect:
                accounts[source + ":" + symbol] = actual
            actual_episodes = _episodes(actual, source=source, symbol=symbol, cost=_cost(settings),
                config=config, market_index=market_index, policy_hash=policy_hash)
            entry_episodes = {r["entry_date"]: r for r in actual_episodes}
            states = {r["date"]: r for r in actual["daily"]}
            candidate_by_day = {c["date"]: c for c in local["candidates"] if source == "union" or source in c.get("sources", [])}
            for episode in actual_episodes:
                candidate = candidate_by_day.get(episode["signal_date"], {})
                episode.update(source_snapshot_hash=source_snapshot_hash,
                    opportunity_id=candidate.get("opportunity_id"),
                    economic_group_id=candidate.get("opportunity_id", episode["episode_id"]),
                    source_event_ids=candidate.get("source_event_ids", []),
                    reference_path_hash=reference_path_hash, candidate_path_hash=candidate_path_hash)
                emit("episodes", episode)
            entry_values_by_day = {}
            for day, candidate in candidate_by_day.items():
                if day < first_day:
                    continue
                index = by_date[day]; state = states[day]; intent = decisions[index]
                plan = copy.deepcopy(clone_decisions[index].get("entry_plan") or {})
                entry_values = entry_feature_row(rows[index], state, candidate, plan,
                    initial_cash=initial_cash, event_prefix=prefixes[index])
                entry_values_by_day[day] = entry_values
                record = {**entry_values, "source": source, "opportunity_id": candidate["opportunity_id"],
                    "source_event_ids": candidate.get("source_event_ids", []),
                    "l1_ma_entry_target_contract_hash": candidate.get("l1_ma_entry_target_contract_hash"),
                    "economic_group_id": candidate["opportunity_id"], "population": "flat_actual_state_candidate_clone",
                    "continuation_policy_hash": policy_hash, "source_snapshot_hash": source_snapshot_hash,
                    "reference_path_hash": reference_path_hash, "candidate_path_hash": candidate_path_hash,
                    "candidate_plan_hash": stable_hash(plan), "behavior_cash": state["cash"],
                    "behavior_shares": state["shares"], "state_sha256": stable_hash(state),
                    "selected_by_reference": bool(intent.get("entry_plan", {}).get("entry_allowed")) if isinstance(intent.get("entry_plan"), dict) else False,
                    "input_eligible": bool(plan.get("entry_allowed")) and bool(
                        any(entry_values.get(s + "_input_available", 0) for s in candidate.get("sources", []))
                        if source == "union" else entry_values.get(source + "_input_available", 0)), "label": np.nan,
                    "positive_return_label": np.nan, "label_available": False, "label_end": None,
                    "label_available_at": None, "label_reason": "candidate_not_decidable_existing_position",
                    "outcome_class": None, "net_return": np.nan, "net_profit": np.nan,
                    "r_multiple": np.nan, "risk_cash": np.nan, "clone_ledger_sha256": None}
                clone = None
                if state["shares"] == 0:
                    if not plan.get("entry_allowed") or plan.get("status") != "active":
                        record["label_reason"] = "candidate_risk_or_input_ineligible"
                    else:
                        known = valid_indices[valid_indices <= index]
                        count = int(settings["execution"]["seasoned_bars"])
                        begin = int(known[-count - 1]) if len(known) > count else 0
                        entry_market_offset = market_index[day] + 1
                        # Bound the first replay by the frozen time limit, then
                        # extend only if guarded/partial liquidation needs it.
                        max_offset = min(len(calendar) - 1, entry_market_offset + protocol["max_holding_sessions"] + 1)
                        finish = max(index + 2, int(np.searchsorted(raw.date.to_numpy(), calendar[max_offset], side="right")))
                        finish = min(len(raw), finish)
                        while True:
                            replay = copy.deepcopy(decisions[begin:finish])
                            for offset, decision in enumerate(replay, begin):
                                if offset < index:
                                    decision.update(target_weight=0., entry_plan=None, hold_current_position=False)
                                elif offset > index:
                                    decision["entry_plan"] = None
                                    if decision.get("action") != "SELL" and not decision.get("rule_sell"):
                                        decision.update(hold_current_position=True, action="HOLD")
                            replay[index - begin] = {**copy.deepcopy(clone_decisions[index]), "entry_plan": plan,
                                "target_weight": plan["target_weight"], "hold_current_position": False, "action": "BUY"}
                            clone = execute_decisions(symbol, raw.iloc[begin:finish].reset_index(drop=True), replay,
                                day, float(state["cash"]), settings, "point-candidate-clone-" + source,
                                market_dates=calendar, verified_sources=VERIFIED_SOURCES, entry_policy="risk",
                                exit_policy=_max_exit_callback(market_index, protocol["max_holding_sessions"], policy_hash,
                                    extra_policy=_fresh_policy(extra), contract_initial_cash=initial_cash))
                            if clone["metrics"]["open_shares"] == 0 or finish == len(raw):
                                break
                            finish = min(len(raw), finish + 32)
                        snapshots = clone.get("holding_snapshots", [])
                        stop = snapshots[0].get("stop_price") if snapshots else plan.get("stop_price")
                        record.update(episode_outcome(clone["ledger"], stop_price=stop, cost_model=_cost(settings), config=config),
                            clone_ledger_sha256=stable_hash(clone["ledger"]),
                            clone_initial_cash=float(state["cash"]), entry_rejection_count=sum(r["action"] == "BUY" for r in clone["rejections"]),
                            exit_rejection_count=sum(r["action"] == "SELL" for r in clone["rejections"]))
                        if clone["ledger"]:
                            clone_episode = _episodes(clone, source=source, symbol=symbol, cost=_cost(settings),
                                config=config, market_index=market_index, policy_hash=policy_hash)[0]
                            record.update(clone_episode_id=clone_episode["episode_id"],
                                **{k: clone_episode[k] for k in ("max_favorable_excursion", "max_adverse_excursion",
                                    "final_peak_giveback", "entry_fee_ratio", "outcome_scope", "holding_sessions")})
                        if not clone["ledger"]:
                            matching = [r for r in clone["rejections"] if r["action"] == "BUY" and r.get("signal_date") == day]
                            record["label_reason"] = "entry_rejected:" + matching[0]["reason"] if matching else "entry_unobserved_or_calendar_tail"
                if candidate_callback:
                    candidate_callback(record, clone)
                emit("entries", record)
            running = {}
            for snapshot in actual.get("holding_snapshots", []):
                episode = entry_episodes[snapshot["entry_date"]]
                entry_day = episode["signal_date"]
                ev = entry_values_by_day.get(entry_day, {})
                prefix = running.setdefault(episode["episode_id"], {})
                prefix.update(prefixes[by_date[snapshot["date"]]])
                values = holding_feature_row(rows[by_date[snapshot["date"]]], snapshot,
                    entry_values=ev, prefix=prefix, market_index=market_index)
                remaining, proceeds, continued_end = int(snapshot["shares"]), 0., None
                for fill in actual["ledger"][snapshot["ledger_entries_seen"]:]:
                    if fill["action"] == "BUY":
                        break
                    remaining -= int(fill["shares"]); proceeds += float(fill["cash_flow"])
                    if remaining == 0:
                        continued_end = fill["date"]; break
                forced = _liquidate(snapshot, tuple_rows, seasoned, market_next, settings)
                record = {**values, "source": source, "episode_id": episode["episode_id"],
                    "economic_group_id": episode["episode_id"], "population": "continuous_actual_holding_state",
                    "actual_shares": snapshot["shares"], "actual_entry_date": snapshot["entry_date"],
                    "actual_cost_per_share": snapshot["actual_cost_per_share"],
                    "snapshot_sha256": stable_hash(snapshot), "source_snapshot_hash": source_snapshot_hash,
                    "reference_path_hash": reference_path_hash, "candidate_path_hash": candidate_path_hash,
                    "continuation_policy_hash": policy_hash, "input_eligible": bool(
                        any(values.get(s + "_input_available", 0) for s in ("ma", "structure", "wyckoff"))
                        if source == "union" else values.get(source + "_input_available", 0)),
                    "forced_ledger_sha256": stable_hash(forced["ledger"]),
                    **exit_outcome(forced=forced, continued_proceeds=proceeds, continued_end=continued_end,
                        shares=snapshot["shares"], close=snapshot["close"])}
                emit("exits", record)
    result = {"entries": pd.DataFrame(entries), "exits": pd.DataFrame(exits),
        "episodes": pd.DataFrame(episodes), "events": pd.DataFrame(all_points.get("events", [])) if collect else pd.DataFrame(),
        "reference_accounts": accounts, "contracts": contracts, "coverage": dict(coverage), "version": EPISODE_VERSION}
    # A multi-policy frame carries every source contract explicitly. Consumers
    # bind the selected source before training; targets are never silently pooled.
    for kind in ("entries", "exits"):
        result[kind].attrs.update(feature_columns=list(ENTRY_FEATURE_COLUMNS if kind == "entries" else EXIT_FEATURE_COLUMNS),
            feature_version="czsc_high_return_native_operation_v4", schema_hash=ENTRY_SCHEMA_HASH if kind == "entries" else EXIT_SCHEMA_HASH,
            source_contracts={s: c["entry" if kind == "entries" else "exit"] for s, c in contracts.items()})
    return result


def training_frame(dataset, *, source="union", head="entry", ma_only=False):
    """Bind one exact policy and target; MA bootstrap entry is net-positive."""
    kind = "exits" if head == "exit" else "entries"
    frame = dataset[kind]
    result = frame.loc[frame.source == source].copy() if len(frame) else pd.DataFrame(columns=frame.columns)
    key = "ma_exit" if ma_only and head == "exit" else "ma_entry" if ma_only else head
    contract = dataset["contracts"][source][key]
    if ma_only and head == "entry" and len(result):
        result["label"] = result.positive_return_label
        # Risk identity is still required by the shared risk strategy target.
    if head == "exit" and len(result):
        counts = result.groupby("episode_id").episode_id.transform("size")
        result["sample_weight"] = 1. / counts
        available = result.groupby("episode_id").label_available.transform("all")
        group_end = result.groupby("episode_id").label_end.transform("max")
        result["group_label_available"] = available
        result["group_label_end"] = group_end.where(available, None)
        result["label_available"] = result.label_available & available
    if ma_only and len(result):
        result["input_eligible"] &= result.ma_input_available.eq(1.)
    return bind_frame(result, head=head, contract=contract, ma_only=ma_only)
