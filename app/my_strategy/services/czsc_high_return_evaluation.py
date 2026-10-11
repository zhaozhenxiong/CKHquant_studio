"""Fixed-universe actual-ledger evaluation; observed history is diagnostic only.

Workers write one stock's accounts and return file references. The coordinator
marks every requested account on the same verified dates, retaining failed
accounts as cash, and resamples paired portfolio returns in long date blocks.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_high_return_labels import label_config
from my_strategy.services.czsc_research_ml import date_block_bootstrap
from my_strategy.services.czsc_trade_episodes import _cost, _episodes, _settings

EVALUATION_VERSION = "czsc_high_return_fixed_accounts_evaluation_v2"
CASH_NOT_OBSERVED = "cash_only_not_observed_yet"
SOURCE_AFTER_WINDOW = "source_begins_after_window"
DEFAULT_ARMS = ("reference", "entry_only", "exit_only", "full", "ma", "structure", "wyckoff")


def _worker_init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _mainboard(symbol):
    return symbol.startswith("60") and symbol.endswith(".SH") or symbol.startswith("00") and symbol.endswith(".SZ")


def _window(value, calendar):
    start = value.get("test_start") or value.get("start") or value.get("validation_start")
    end = value.get("test_end") or value.get("end") or value.get("validation_end")
    if not start or not end or str(start) > str(end):
        raise ValueError("evaluation window needs chronological start/end")
    dates = [d for d in calendar if str(start) <= d <= str(end)]
    if not dates:
        raise ValueError("no verified sessions in evaluation window")
    return {**value, "name": value.get("name", str(start) + "_" + str(end)),
        "start": str(start), "end": str(end), "dates": dates,
        "initial_state": "fresh_flat_independent_stock_account"}


def _stress_config(config, multiplier):
    if not math.isfinite(float(multiplier)) or multiplier < 1:
        raise ValueError("cost stress multiplier must be finite and at least one")
    result = copy.deepcopy(config)
    strategy = copy.deepcopy(_settings(config))
    original = copy.deepcopy(strategy["execution"])
    if multiplier != 1:
        result["_cost_stress_base_execution"] = original
        result["_cost_multiplier"] = float(multiplier)
        for key in ("commission", "stamp_tax", "min_commission", "slippage"):
            strategy["execution"][key] = float(original[key]) * float(multiplier)
    result["strategy"] = strategy
    return result


def _replay(features, calendar, *, model_root, bundles, config, start, end, arm, validation_bundle,
        session=None, prepared_cache=None, allow_training=False):
    from my_strategy.services.czsc_high_return_runtime import replay_high_return, prepare_points
    source = arm if arm in {"ma", "structure", "wyckoff"} else "union"
    runtime_arm = "reference" if source != "union" else arm
    contract_config = copy.deepcopy(config)
    if "_cost_stress_base_execution" in contract_config:
        contract_config["strategy"]["execution"] = contract_config.pop("_cost_stress_base_execution")
        contract_config.pop("_cost_multiplier", None)
    if prepared_cache is not None and 'points' not in prepared_cache:
        prepared_cache['points']=prepare_points(features,calendar,model_root=model_root,bundles=bundles,
            config=contract_config,session=session)
    return replay_high_return(features, calendar, model_root=model_root, bundles=bundles,
        config=config, contract_config=contract_config, start=start, end=end, mode=runtime_arm, source=source,
        validation_bundle=validation_bundle,session=session,allow_training=allow_training,
        prepared_points=prepared_cache.get('points') if prepared_cache is not None else None)


def _write_account(destination, account, episodes, *, status, reason=None, source_first_date=None):
    destination.mkdir(parents=True, exist_ok=True)
    for key in ("daily", "ledger", "rejections"):
        frame = pd.DataFrame(account.get(key, []))
        if frame.empty and not len(frame.columns):
            frame = pd.DataFrame(columns={"daily": ["date", "cash", "equity", "market_value"],
                "ledger": ["date", "symbol", "action", "shares", "price", "cash_flow", "fee"],
                "rejections": ["date", "symbol", "action", "reason"]}[key])
        frame.to_csv(destination / (key + ".csv"), index=False)
    pd.DataFrame(episodes, columns=None if episodes else ["episode_id", "net_return", "net_profit", "outcome_class", "exit_date"]).to_csv(
        destination / "episodes.csv", index=False)
    for key in ("holding_snapshots", "entry_diagnostics", "exit_policy_diagnostics", "entry_gate_diagnostics"):
        with (destination / (key + ".jsonl")).open("w") as stream:
            for row in account.get(key, []):
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    model_coverage = account.get("model_coverage")
    if model_coverage is None:
        source = account.get("policy_source", "union")
        days = [r["date"] for r in account.get("daily", [])]
        candidates = [c for c in account.get("candidate_points", []) if days and days[0] <= c["date"] <= days[-1]
            and (c.get("candidate_eligible") if source == "union" else (c.get("source_eligibility") or {}).get(source))]
        gates = account.get("entry_gate_diagnostics", [])
        model_coverage = {"total_candidates": len(candidates),
            "selected_candidates": sum(r["action"] == "BUY" for r in account.get("ledger", [])),
            "decidable_candidates": len(gates),
            "model_scored_candidates": sum(r.get("probability") is not None for r in gates),
            "actual_holding_states": len(account.get("holding_snapshots", [])),
            "model_scored_holding_states": sum(r.get("probability") is not None and not r.get("forced_by_policy")
                for r in account.get("exit_policy_diagnostics", []))}
    metadata = {"status": status, "reason": reason, "metrics": account.get("metrics", {}),
        "cost_model": account.get("cost_model"), "model_coverage": model_coverage,
        "qualification": "observed_historical_development_diagnostic", "publication_allowed": False}
    if source_first_date is not None:
        metadata["source_first_date"] = source_first_date
    _save(destination / "account.json", metadata)
    return {"path": str(destination.resolve()), "status": status, "reason": reason,
        **({"source_first_date": source_first_date} if source_first_date is not None else {}),
        "daily_sha256": _sha(destination / "daily.csv"), "ledger_sha256": _sha(destination / "ledger.csv"),
        "episodes_sha256": _sha(destination / "episodes.csv"),
        "rejections_sha256": _sha(destination / "rejections.csv"),
        "holding_snapshots_sha256": _sha(destination / "holding_snapshots.jsonl"),
        "entry_diagnostics_sha256": _sha(destination / "entry_diagnostics.jsonl"),
        "exit_policy_diagnostics_sha256": _sha(destination / "exit_policy_diagnostics.jsonl"),
        "entry_gate_diagnostics_sha256": _sha(destination / "entry_gate_diagnostics.jsonl"),
        "account_sha256": _sha(destination / "account.json")}


def _cash_account(symbol, dates, initial_cash):
    return {"daily": [{"date": d, "cash": initial_cash, "equity": initial_cash, "market_value": 0.,
        "shares": 0, "symbol": symbol} for d in dates], "ledger": [], "rejections": [],
        "holding_snapshots": [], "metrics": {"total_return": 0., "max_drawdown": 0.,
            "final_equity": initial_cash, "fees": 0., "open_shares": 0}}


def _window_market_reason(features, window):
    """Describe validated source coverage, without inferring listing identity."""
    if not ((features.date >= window["start"]) & (features.date <= window["end"])).any():
        return SOURCE_AFTER_WINDOW if not features.empty and features.date.iloc[0] > window["end"] else "no_window_market_bars"
    return None


def _evaluation_worker(payload):
    from my_strategy.services.czsc_high_return_models import HeadSession
    record, calendar, windows, arms, model_root, bundles, config, validation_bundle, output = payload[:9]
    allow_training = payload[9] if len(payload) > 9 else False
    symbol = record["symbol"]; initial_cash = float(config.get("initial_cash", 100000.))
    frozen_failure, features = None, None
    try:
        if _sha(record["path"]) != record["sha256"]:
            raise ValueError("frozen_source_hash_mismatch")
        from my_strategy.services.czsc_high_return_research import _read_features
        features = _read_features(record,config,calendar[-1])
        if (features.symbol.astype(str).nunique() != 1 or str(features.symbol.iloc[0]) != symbol
                or features.date.duplicated().any() or not features.date.is_monotonic_increasing
                or not set(features.date) <= set(calendar)):
            raise ValueError("frozen_source_symbol_or_dates_invalid")
    except Exception as exc:
        frozen_failure = str(exc)
    results = []
    market_index = {d: i for i, d in enumerate(calendar)}
    for window in windows:
        session=HeadSession(); prepared_cache={}
        selected, reason = None, frozen_failure
        if reason is None:
            reason = _window_market_reason(features, window)
            selected = features.loc[features.date <= window["end"]].copy()
            # Runtime rebuilds causal point events and their ages. Preserve the
            # whole prefix, including observed events older than Broker seasoning.
            selected = selected.reset_index(drop=True)
        for arm in arms:
            destination = Path(output) / window["name"] / arm / symbol.replace(".", "_")
            failure = reason
            episodes = []
            try:
                if failure:
                    raise ValueError(failure)
                account = _replay(selected, calendar, model_root=model_root, bundles=bundles,
                    config=config, start=window["start"], end=window["end"], arm=arm,
                    validation_bundle=validation_bundle,session=session,prepared_cache=prepared_cache,
                    allow_training=allow_training)
                if not account.get("daily"):
                    raise ValueError("runtime_returned_no_daily_account")
                episodes = _episodes(account, source=arm, symbol=symbol, cost=_cost(_settings(config)),
                    config=config, market_index=market_index,
                    policy_hash=account.get("continuation_policy_hash", stable_hash({"arm": arm, "config": config})))
                from my_strategy.services.czsc_dual_research import round_trip_statistics
                statistics = round_trip_statistics([account["ledger"]])
                if statistics["completed_round_trips"] != sum(row["exit_date"] is not None for row in episodes):
                    raise ValueError("episode and actual ledger roundtrip counts differ")
                account.setdefault("metrics", {}).update(statistics)
                status = "complete"
            except Exception as exc:
                failure = str(exc)
                status = CASH_NOT_OBSERVED if frozen_failure is None and reason == SOURCE_AFTER_WINDOW else "cash_retained_failure"
                account = _cash_account(symbol, window["dates"], initial_cash); episodes=[]
            binding = _write_account(destination, account, episodes, status=status, reason=failure,
                source_first_date=features.date.iloc[0] if status == CASH_NOT_OBSERVED else None)
            results.append({"symbol": symbol, "window": window["name"], "arm": arm, **binding})
    return {"symbol": symbol, "accounts": results}


def _aggregate_bindings(bindings, dates, initial_cash):
    """Sum independently marked stock budgets, including explicit failed cash."""
    equity = pd.Series(0., index=dates); occupied = pd.Series(0., index=dates)
    fees, notional, rejects, open_positions = 0., 0., 0, 0
    closed_returns, pnls, holding, classes, reasons, unavailable = [], [], [], Counter(), [], []
    trade_count, selected_candidates, total_candidates = 0, 0, 0
    decidable_candidates, scored_candidates, holding_states, scored_holding_states = 0, 0, 0, 0
    for binding in bindings:
        root = Path(binding["path"])
        if _sha(root / "daily.csv") != binding["daily_sha256"] or _sha(root / "ledger.csv") != binding["ledger_sha256"]:
            raise ValueError("evaluation ledger changed before aggregation")
        frame = pd.read_csv(root / "daily.csv")
        if frame.date.duplicated().any():
            raise ValueError("duplicate account daily date")
        frame = frame.set_index("date").sort_index().reindex(dates).ffill()
        equity += pd.to_numeric(frame.equity).fillna(initial_cash)
        occupied += pd.to_numeric(frame.get("market_value", pd.Series(0., index=dates))).fillna(0.)
        ledger = pd.read_csv(root / "ledger.csv")
        rejections = pd.read_csv(root / "rejections.csv")
        rejects += len(rejections); trade_count += len(ledger)
        if len(ledger):
            fees += float(pd.to_numeric(ledger.fee).sum())
            notional += float((pd.to_numeric(ledger.price) * pd.to_numeric(ledger.shares)).sum())
        eps = pd.read_csv(root / "episodes.csv")
        if len(eps):
            closed = eps.loc[eps.exit_date.notna()]
            closed_returns.extend(pd.to_numeric(closed.net_return).tolist())
            pnls.extend(pd.to_numeric(closed.net_profit).tolist())
            holding.extend(pd.to_numeric(closed.holding_sessions).tolist())
            classes.update(eps.outcome_class.fillna("risk_unavailable_or_censored"))
            open_positions += int(eps.exit_date.isna().sum())
        metadata = json.loads((root / "account.json").read_text())
        if binding["status"] == CASH_NOT_OBSERVED:
            unavailable.append({"symbol": binding["symbol"], "reason": binding["reason"],
                "status": CASH_NOT_OBSERVED, "source_first_date": binding.get("source_first_date")})
        elif binding["status"] != "complete":
            reasons.append({"symbol": binding["symbol"], "reason": binding["reason"]})
        coverage = metadata.get("model_coverage") or {}
        if isinstance(coverage, dict):
            selected_candidates += int(coverage.get("selected_candidates", 0))
            total_candidates += int(coverage.get("total_candidates", 0))
            decidable_candidates += int(coverage.get("decidable_candidates", 0))
            scored_candidates += int(coverage.get("model_scored_candidates", 0))
            holding_states += int(coverage.get("actual_holding_states", 0))
            scored_holding_states += int(coverage.get("model_scored_holding_states", 0))
    total_cash = initial_cash * len(bindings)
    prior = equity.shift(1); prior.iloc[0] = total_cash
    returns = equity / prior - 1
    drawdown = equity / equity.cummax().clip(lower=total_cash) - 1
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p < 0]
    volatility = float(returns.std(ddof=0) * np.sqrt(252))
    metrics = {"initial_cash": total_cash, "net_return": float(equity.iloc[-1] / total_cash - 1),
        "total_return": float(equity.iloc[-1] / total_cash - 1), "final_equity": float(equity.iloc[-1]),
        "max_drawdown": float(-drawdown.min()), "fees": fees, "trade_count": trade_count,
        "rejection_count": rejects, "open_positions": open_positions,
        "completed_round_trips": len(pnls), "trade_expectancy": float(np.mean(closed_returns)) if pnls else None,
        "mean_net_pnl_CNY": float(np.mean(pnls)) if pnls else None,
        "win_rate": len(wins) / len(pnls) if pnls else None,
        "payoff_ratio": float(np.mean(wins) / -np.mean(losses)) if wins and losses else None,
        "profit_factor": float(sum(wins) / -sum(losses)) if losses else None,
        "high_return_count": int(classes["high"]), "loss_count": int(classes["loss"]),
        "ordinary_count": int(classes["ordinary"]), "high_return_share": classes["high"] / len(pnls) if pnls else None,
        "mean_holding_sessions": float(np.mean(holding)) if holding else None,
        "tail_daily_return_5pct": float(np.quantile(returns, .05)),
        "mean_capital_occupancy": float((occupied / total_cash).mean()),
        "mean_daily_turnover": notional / total_cash / len(dates),
        "sharpe": float(returns.mean() * 252 / volatility) if volatility else 0.,
        "candidate_coverage": selected_candidates / total_candidates if total_candidates else None,
        "raw_eligible_candidates": total_candidates, "actual_entry_count": selected_candidates,
        "decidable_candidates": decidable_candidates, "model_scored_candidates": scored_candidates,
        "entry_model_score_coverage": scored_candidates / decidable_candidates if decidable_candidates else None,
        "actual_holding_states": holding_states, "model_scored_holding_states": scored_holding_states,
        "exit_model_score_coverage": scored_holding_states / holding_states if holding_states else None}
    daily = [{"date": d, "equity": float(equity[d]), "drawdown": float(drawdown[d]),
        "market_value": float(occupied[d]), "return": float(returns[d])} for d in dates]
    return {"metrics": metrics, "daily": daily, "account_count": len(bindings),
        "cash_retained_failures": reasons, "market_unavailable_accounts": unavailable, "outcome_classes": dict(classes),
        "accounting": "fixed_equal_cash_per_stock_no_shared_cash_no_failed_stock_deletion"}


def _paired_uncertainty(reference, alternative, config):
    left = pd.DataFrame(reference["daily"]).set_index("date")["return"]
    right = pd.DataFrame(alternative["daily"]).set_index("date")["return"]
    paired = pd.DataFrame({"difference": right - left}).reset_index()
    result = {}
    for block in (160, 240):
        try:
            result[str(block)] = date_block_bootstrap(paired, "difference", block_length=block,
                dependence_horizon=max(120, label_config(config)["max_holding_sessions"]),
                repetitions=int(config.get("bootstrap_repetitions", 1000)), seed=int(config.get("seed", 42)))
        except ValueError as exc:
            result[str(block)] = {"unavailable": str(exc), "block_length": block,
                "dates": len(paired), "minimum_complete_blocks": 2}
    return result


def evaluate_bundles(records, calendar, *, model_root, bundles, config, windows=None,
        cpu_workers=4, progress=None, check_cancel=None, output_root=None,
        validation_bundle=None, arms=None, cost_multiplier=1, allow_training=False):
    """Evaluate frozen arms on observed windows and optionally a long account.

    Explicit windows (e.g. a policy-selection half) are used verbatim. Defaults
    append a 2023-to-local-end continuous account without checkpoint resets.
    No probability/threshold is refit in a cost stress replay.
    """
    dates = [pd.Timestamp(d).date().isoformat() for d in calendar]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("evaluation requires verified unique ascending calendar")
    if int(cpu_workers) != cpu_workers or cpu_workers < 1:
        raise ValueError("evaluation workers must be positive")
    if not records or len({r["symbol"] for r in records}) != len(records):
        raise ValueError("frozen evaluation universe must be nonempty and unique")
    requested = [r for r in records if _mainboard(r["symbol"])]
    excluded = [{"symbol": r["symbol"], "reason": "unsupported_board"} for r in records if not _mainboard(r["symbol"])]
    if not requested:
        raise ValueError("no supported mainboard accounts")
    selected_arms = tuple(arms or DEFAULT_ARMS)
    if len(set(selected_arms)) != len(selected_arms) or set(selected_arms) - set(DEFAULT_ARMS):
        raise ValueError("invalid evaluation arms")
    cfg = _stress_config(config, cost_multiplier)
    initial_cash = float(cfg.get("initial_cash", 100000.))
    if initial_cash != 100000.:
        raise ValueError("v1 evaluation target is frozen to 100000 cash per stock")
    if windows is None:
        values = list(config.get("development_windows", []))
        values.append({"name": "continuous_2023", "start": config.get("feature_start", "2023-01-01"),
            "end": config.get("effective_end", dates[-1]), "continuous_no_checkpoint_account_reset": True})
    else:
        values = list(windows)
    selected_windows = [_window(w, dates) for w in values]
    if len({w["name"] for w in selected_windows}) != len(selected_windows):
        raise ValueError("duplicate evaluation window name")
    identity = stable_hash({"version": EVALUATION_VERSION, "records": records, "calendar": dates,
        "windows": selected_windows, "arms": selected_arms, "bundles": bundles,
        "validation_bundle": validation_bundle, "cost_multiplier": cost_multiplier, "config": config})
    root = Path(output_root or Path(model_root).parent / "high-return-evaluation") / identity[:12]
    root.mkdir(parents=True, exist_ok=True)
    payloads = [(r, dates, selected_windows, selected_arms, str(model_root), bundles, cfg,
        validation_bundle, str(root), bool(allow_training)) for r in requested]
    bindings = {(w["name"], arm): [] for w in selected_windows for arm in selected_arms}
    check = check_cancel or (lambda: None)
    def accept(result, offset):
        check()
        for account in result["accounts"]:
            bindings[(account["window"], account["arm"])].append(account)
        if progress:
            progress("高收益实际Broker账本", offset + 1, len(payloads), 0)
    if cpu_workers == 1:
        for i, payload in enumerate(payloads):
            check(); accept(_evaluation_worker(payload), i)
    else:
        with ProcessPoolExecutor(max_workers=min(int(cpu_workers), len(payloads)),
                mp_context=multiprocessing.get_context("spawn"),initializer=_worker_init) as pool:
            for i, result in enumerate(pool.map(_evaluation_worker, payloads, chunksize=1)):
                accept(result, i)
    reports = {}
    for window in selected_windows:
        variants = {arm: _aggregate_bindings(bindings[(window["name"], arm)], window["dates"], initial_cash)
            for arm in selected_arms}
        comparisons = {arm: _paired_uncertainty(variants["reference"], value, config)
            for arm, value in variants.items() if arm != "reference"} if "reference" in variants else {}
        for arm, value in variants.items():
            pd.DataFrame(value["daily"]).to_csv(root / window["name"] / arm / "aggregate_daily.csv", index=False)
        reports[window["name"]] = {"fold": {k: v for k, v in window.items() if k != "dates"},
            "variants": variants, "paired_daily_return_uncertainty": comparisons,
            "qualification": "observed_historical_development_diagnostic", "model_gate": False,
            "publication_allowed": False, "frozen_mainboard_accounts": len(requested),
            "initial_cash": initial_cash * len(requested), "ledger_bindings": {
                arm: bindings[(window["name"], arm)] for arm in selected_arms}}
    report = {"version": EVALUATION_VERSION, "evaluation_identity": identity,
        "model_bundles_hash":stable_hash(bundles),"source_records_hash":stable_hash(requested),
        "source_snapshot_hash":config.get('source_snapshot_hash'),
        "calendar_hash":stable_hash(dates),"strategy_configuration_hash":stable_hash(_settings(config)),
        "base_execution_config_hash":stable_hash(_settings(config)['execution']),
        "windows": reports, "output_root": str(root.resolve()), "model_gate": False,
        "publication_allowed": False, "qualification": "observed_historical_development_diagnostic",
        "frozen_mainboard_accounts": len(requested), "excluded_accounts": excluded,
        "initial_cash": initial_cash * len(requested), "cost_multiplier": float(cost_multiplier),
        "thresholds_refit": False, "validation_bundle_mode": validation_bundle is not None,
        "bootstrap_contract": {"blocks": [160, 240], "minimum_complete_blocks": 2,
            "minimum_technical_sessions": 480, "same_verified_dates_paired": True}}
    if len(reports) == 1:
        report["variants"] = next(iter(reports.values()))["variants"]
        report["paired_daily_return_uncertainty"] = next(iter(reports.values()))["paired_daily_return_uncertainty"]
    _save(root / "evaluation.json", report)
    return report


def _policy_options_worker(payload):
    """One stock, one immutable L1 prefix and reference, all actual policies."""
    record,calendar,window,model_root,ma_bundles,options,config,output=payload
    from my_strategy.services.czsc_high_return_runtime import prepare_points,replay_high_return
    from my_strategy.services.czsc_high_return_models import HeadSession
    from my_strategy.services.czsc_dual_research import round_trip_statistics
    symbol=record['symbol']; cash=float(config['initial_cash']); root=Path(output)
    session=HeadSession(); failure=None; features=None; points=None; market_not_observed=False
    try:
        if _sha(record['path'])!=record['sha256']: raise ValueError('frozen_source_hash_mismatch')
        from my_strategy.services.czsc_high_return_research import _read_features
        features=_read_features(record,config,calendar[-1])
        if features.empty or features.symbol.nunique()!=1 or str(features.symbol.iloc[0])!=symbol or features.date.duplicated().any() or not features.date.is_monotonic_increasing or not set(features.date)<=set(calendar):
            raise ValueError('frozen_source_symbol_or_dates_invalid')
        failure=_window_market_reason(features,window)
        market_not_observed=failure==SOURCE_AFTER_WINDOW
        if failure: raise ValueError(failure)
        features=features.loc[features.date<=window['end']].reset_index(drop=True)
        points=prepare_points(features,calendar,model_root=model_root,bundles=ma_bundles,config=config,session=session)
    except Exception as exc: failure=str(exc)
    market_index={d:i for i,d in enumerate(calendar)}
    def run(bundle,mode,destination):
        error=failure; episodes=[]
        try:
            if error: raise ValueError(error)
            account=replay_high_return(features,calendar,model_root=model_root,bundles=ma_bundles,config=config,
                start=window['start'],end=window['end'],mode=mode,source='union',initial_cash=cash,
                validation_bundle=bundle,prepared_points=points,session=session)
            if not account.get('daily'): raise ValueError('runtime_returned_no_daily_account')
            episodes=_episodes(account,source=mode,symbol=symbol,cost=_cost(_settings(config)),config=config,
                market_index=market_index,policy_hash=account['policy_hash'])
            stats=round_trip_statistics([account['ledger']])
            if stats['completed_round_trips']!=sum(r['exit_date'] is not None for r in episodes):
                raise ValueError('actual ledger roundtrip counts differ')
            account['metrics'].update(stats); status='complete'
        except Exception as exc:
            error=str(exc)
            status=CASH_NOT_OBSERVED if market_not_observed else 'cash_retained_failure'
            account=_cash_account(symbol,window['dates'],cash);episodes=[]
        return {'symbol':symbol,**_write_account(destination,account,episodes,status=status,reason=error,
            source_first_date=features.date.iloc[0] if status==CASH_NOT_OBSERVED else None)}
    reference=run(None,'reference',root/'reference'/symbol.replace('.','_'))
    actual=[]
    for offset,bundle in enumerate(options):
        actual.append(run(bundle,'full',root/f'option_{offset:02d}'/'full'/symbol.replace('.','_')))
    return {'symbol':symbol,'reference':reference,'options':actual}


def evaluate_policy_options(records,calendar,*,model_root,ma_bundles,options,config,fold,
        cpu_workers=4,progress=None,check_cancel=None,output_root=None):
    """Batch the declared validation budget without shrinking stocks or fills.

    Each policy receives its own fresh real account. Only immutable L1 facts,
    inference weights and the identical reference account are shared.
    """
    if not options: return []
    if int(cpu_workers)!=cpu_workers or cpu_workers<1: raise ValueError('evaluation workers must be positive')
    if float(config.get('initial_cash',100000.))!=100000.: raise ValueError('v1 evaluation target is frozen to 100000 cash per stock')
    config={**config,'initial_cash':100000.}
    dates=[pd.Timestamp(d).date().isoformat() for d in calendar]
    if not dates or dates!=sorted(set(dates)): raise ValueError('verified chronological calendar required')
    requested=[r for r in records if _mainboard(r['symbol'])]
    if not requested or len({r['symbol'] for r in requested})!=len(requested): raise ValueError('unique stock accounts required')
    window=_window({'name':fold['name']+'_policy','start':fold['policy_start'],'end':fold['policy_end']},dates)
    identity=stable_hash({'version':EVALUATION_VERSION,'mode':'batched_actual_policy_validation',
        'records':requested,'calendar':dates,'fold':fold,'ma_bundles':ma_bundles,'options':options,'config':config})
    root=Path(output_root or Path(model_root)/'validation'/'policy-options')/identity[:12];root.mkdir(parents=True,exist_ok=True)
    payloads=[(r,dates,window,str(model_root),ma_bundles,options,config,str(root)) for r in requested]
    reference=[]; bindings=[[] for _ in options]; check=check_cancel or (lambda:None)
    def accept(value,n):
        check();reference.append(value['reference'])
        for offset,binding in enumerate(value['options']): bindings[offset].append(binding)
        if progress: progress(f"policy_{fold['name']}_options_{len(options)}",n,len(requested),
            sum(v['status']=='cash_retained_failure' for v in reference))
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=min(cpu_workers,len(requested)),mp_context=multiprocessing.get_context('spawn'),initializer=_worker_init) as pool:
            for n,result in enumerate(pool.map(_policy_options_worker,payloads,chunksize=1),1): accept(result,n)
    else:
        for n,payload in enumerate(payloads,1): check();accept(_policy_options_worker(payload),n)
    ref=_aggregate_bindings(reference,window['dates'],float(config['initial_cash']))
    reports=[]
    for offset,bundle in enumerate(options):
        full=_aggregate_bindings(bindings[offset],window['dates'],float(config['initial_cash']))
        comparison=_paired_uncertainty(ref,full,config)
        report={'version':EVALUATION_VERSION,'evaluation_identity':identity,'option':offset,'output_root':str(root.resolve()),
            'model_bundles_hash':stable_hash(ma_bundles),'validation_bundle_hash':stable_hash(bundle),
            'source_records_hash':stable_hash(requested),'source_snapshot_hash':config.get('source_snapshot_hash'),
            'calendar_hash':stable_hash(dates),'strategy_configuration_hash':stable_hash(_settings(config)),
            'base_execution_config_hash':stable_hash(_settings(config)['execution']),
            'windows':{window['name']:{'fold':{k:v for k,v in window.items() if k!='dates'},
                'variants':{'reference':ref,'full':full},'ledger_bindings':{'reference':reference,'full':bindings[offset]},
                'paired_daily_return_uncertainty':{'full':comparison},'publication_allowed':False,'model_gate':False}},
            'validation_bundle_mode':True,'qualification':'observed_historical_development_diagnostic',
            'publication_allowed':False,'model_gate':False,'cost_multiplier':1.,'thresholds_refit':False,
            'fixed_mainboard_accounts':len(requested),'batched_policy_count':len(options),
            'common_reference_reused':True,'prefix_scoring':'L1_strict_forward_L2_policy_segment_only'}
        report['variants']={'reference':ref,'full':full}
        _save(root/f'option_{offset:02d}'/'evaluation.json',report);reports.append(report)
    _save(root/'batch.json',{'identity':identity,'options':options,'reports':[str(root/f'option_{i:02d}'/'evaluation.json') for i in range(len(options))],
        'requested':len(requested),'common_reference_reused':True,'publication_allowed':False})
    return reports
