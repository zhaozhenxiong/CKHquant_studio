"""Replay final chart B/S through actual Broker accounts under frozen policies.

Examples::
  python -m my_strategy.scripts.validate_native_v4 --model-run-id RUN --cohort controls
  python -m my_strategy.scripts.validate_native_v4 --model-run-id RUN --cohort training --cpu-workers 4

This is development evidence. Final B gates entry, while actual continuation
retains the model target's raw union source exits. Each window starts with independent cash; chart
arbitration still reads its full causal prefix. Existing v3 uses its own frozen
candidate/model/risk contracts, while v4 pins its new union MA exit head at the
signal. Normal and double slippage replay distinct actual Broker ledgers.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import normalize_symbol
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_high_return_labels_v4 import label_config
from my_strategy.services.czsc_high_return_pipeline_v4 import pinned_exit_policy
from my_strategy.services.czsc_high_return_runtime_v4 import read_run, runtime_overlay
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_trade_episodes_v4 import _max_exit_callback
from my_strategy.web_dashboard.high_return import _calendar, _prepare

VERSION = "native_v4_final_points_actual_broker_validation_20261012"
CONTROL_VERSION = "czsc_agent_consensus_v3"
V4_VERSION = "czsc_agent_consensus_v4"
CONTROL_STOCKS = ("002436.SZ", "600519.SH", "000333.SZ")
OLD_MODEL_RUN = "20261011-053553-815813-20261008-czsc-high-return-research-train-NO_GIT"
DEVELOPMENT_WINDOWS = {
    "2025Q2": ("2025-04-01", "2025-06-30"),
    "2025Q4": ("2025-10-01", "2025-12-31"),
    "2026Q2": ("2026-04-01", "2026-06-30"),
    "2026Q3": ("2026-07-01", "2026-09-30"),
    "2026YTD": ("2026-01-01", "2026-10-09"),
}


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def map_final_points_to_decisions(rows: list[dict], consensus: dict, *, symbol: str,
                                  start: str, end: str, max_weight: float) -> tuple[list[dict], list[dict]]:
    """Final B gates entry; frozen target continuation owns actual exits.

    Neutral rows preserve actual inventory. A final BUY without an active plan
    is explicitly blocked and reported; no candidate or price band is rebuilt.
    A chart S is audited against the same-day raw source exit; an unsupported S
    cannot silently replace the learned label continuation. Events before the
    fresh account boundary cannot open the new account.
    """
    selected: dict[str, dict] = {}
    for event in consensus.get("events", []):
        day = str(event.get("date") or event.get("time"))[:10]
        side = str(event.get("side") or event.get("action") or event.get("direction") or "").upper()
        side = {"B": "BUY", "S": "SELL"}.get(side, side)
        if not start <= day <= end:
            continue
        if side not in {"BUY", "SELL"}:
            raise ValueError("final consensus event has no explicit B/S direction")
        if day in selected:
            raise ValueError("multiple final consensus events on one daily bar")
        selected[day] = {**copy.deepcopy(event), "_side": side}
    decisions, blocked = [], []
    for original in rows:
        day = str(original["date"])[:10]
        if day > end:
            break
        row = {"date": day, "available_at": day + "T15:00:00+08:00", "symbol": symbol,
               "target_weight": 0., "hold_current_position": True, "eligible": True,
               "action": "HOLD", "position_intent": "HOLD", "entry_plan": None,
               "raw_source_exit": False, "chart_sell": False}
        event = selected.get(day)
        raw_exit = bool(start <= day <= end and original.get("rule_sell") is True)
        chart_sell = bool(event and event["_side"] == "SELL")
        row.update(raw_source_exit=raw_exit, chart_sell=chart_sell,
                   source_exit_event_ids=copy.deepcopy(original.get("source_exit_event_ids", [])))
        if chart_sell and not raw_exit:
            blocked.append({"date": day, "symbol": symbol, "reason": "chart_sell_outside_frozen_source_exit_policy",
                            "event": event, "source_exit_event_ids": row["source_exit_event_ids"]})
        if raw_exit:
            row.update(action="SELL", position_intent="SELL", hold_current_position=False,
                       policy_exit_kind="frozen_raw_union_source_exit",
                       final_event_id=event.get("event_id") if chart_sell else None,
                       final_consensus_event=event if chart_sell else None)
            if event and event["_side"] == "BUY":
                blocked.append({"date": day, "symbol": symbol, "reason": "final_buy_conflicts_frozen_source_exit",
                                "event": event, "original_plan": original.get("raw_risk_plan")})
        elif event and event["_side"] == "BUY":
            plan = copy.deepcopy(original.get("raw_risk_plan") or original.get("entry_plan"))
            reason = None
            if not isinstance(plan, dict) or plan.get("status") != "active" or plan.get("entry_allowed") is not True:
                reason = "final_buy_without_active_source_policy_plan"
            elif (plan.get("symbol") != symbol or plan.get("signal_date") != day
                  or plan.get("valid_for_sessions") != 1 or plan.get("policy") != "risk"):
                reason = "final_buy_risk_plan_identity_mismatch"
            elif not plan.get("plan_id") or not isinstance(plan.get("stop_price"), (int, float)):
                reason = "final_buy_risk_plan_missing_frozen_fields"
            if reason:
                blocked.append({"date": day, "symbol": symbol, "reason": reason,
                                "event": event, "original_plan": plan})
            else:
                weight = float(plan["target_weight"])
                if not 0 < weight <= max_weight:
                    raise ValueError("final BUY plan weight exceeds frozen execution policy")
                row.update(action="BUY", position_intent="BUY", target_weight=weight,
                           hold_current_position=False, entry_plan=plan,
                           final_event_id=event.get("event_id"), final_consensus_event=event)
        decisions.append(row)
    return decisions, blocked


def verify_actual_account(account: dict, *, calendar: list[str], decisions: list[dict],
                          initial_cash: float, symbol: str) -> dict:
    """Assert actual cash, inventory, next-open fills, T+1 and frozen plans."""
    following = dict(zip(calendar[:-1], calendar[1:])); intents = {row["date"]: row for row in decisions}
    cash, held, entry_day = float(initial_cash), 0, None
    fills_by_day: dict[str, list] = {}
    for fill in account.get("ledger", []):
        if fill["symbol"] != symbol or fill["date"] != following.get(fill["signal_date"]):
            raise ValueError("actual fill violates symbol/next-verified-session contract")
        if fill["action"] == "BUY":
            if held:
                raise ValueError("final-point replay unexpectedly adds inventory")
            intent = intents[fill["signal_date"]]
            if intent["action"] != "BUY" or not intent.get("entry_plan"):
                raise ValueError("actual entry has no final B plus frozen plan")
            held += int(fill["shares"]); entry_day = fill["date"]
        elif fill["action"] == "SELL":
            if held <= 0 or int(fill["shares"]) > held or fill["date"] <= entry_day:
                raise ValueError("actual exit violates inventory or T+1")
            held -= int(fill["shares"])
        else:
            raise ValueError("unexpected ledger action")
        if int(fill["shares"]) <= 0 or int(fill["shares"]) % 100 or float(fill["fee"]) < 0:
            raise ValueError("actual fill violates positive lots/nonnegative fees")
        cash += float(fill["cash_flow"])
        if cash < -1e-6:
            raise ValueError("actual account cash became negative")
        fills_by_day.setdefault(fill["date"], []).append(fill)
    daily_cash, daily_held = float(initial_cash), 0
    for day in account.get("daily", []):
        for fill in fills_by_day.get(day["date"], []):
            daily_cash += float(fill["cash_flow"])
            daily_held += int(fill["shares"]) * (1 if fill["action"] == "BUY" else -1)
        if abs(float(day["cash"]) - daily_cash) > 1e-6 or int(day["shares"]) != daily_held:
            raise ValueError("daily account cannot be reconstructed from actual ledger")
        expected = daily_cash + daily_held * float(day["close"])
        if abs(float(day["equity"]) - expected) > 1e-6:
            raise ValueError("daily equity violates cash plus actual inventory valuation")
    if held != account["metrics"]["open_shares"]:
        raise ValueError("period end inventory differs from actual ledger")
    return {"next_verified_open": True, "t_plus_one": True, "no_entry_addition": True,
            "cash_inventory_equity_reconstructed": True, "actual_ledger_rows": len(account.get("ledger", []))}


def _base_result(raw: pd.DataFrame, symbol: str, end: str, start: str | None = None) -> dict:
    bars = [{"time": pd.Timestamp(row.date).date().isoformat(), **{key: float(getattr(row, key))
             for key in ("open", "high", "low", "close", "volume")}} for row in raw.itertuples(index=False)
            if (start is None or pd.Timestamp(row.date).date().isoformat() >= start)
            and pd.Timestamp(row.date).date().isoformat() <= end]
    return {"symbol": symbol, "as_of": end, "data_end": end, "model_family": "high_return",
            "price_basis": raw.attrs.get("price_basis", "unadjusted"),
            "frequencies": {"日线": {"bars": bars}}, "events": []}


def _clip_overlay(overlay: dict, end: str) -> dict:
    # Shallow wrapper; decorators own their independent deep copies and states.
    out = dict(overlay)
    for key in ("rows", "signal_points", "candidates", "ma_routes", "wyckoff_events"):
        out[key] = [row for row in overlay.get(key, []) if str(row.get("date") or row.get("available_at"))[:10] <= end]
    return out


def _turnover(account: dict, initial_cash: float) -> float:
    return sum(float(row["shares"]) * float(row["price"]) for row in account.get("ledger", [])) / initial_cash


def _actual_episodes(account: dict) -> list[dict]:
    """Complete fee-net outcomes derived only from actual inventory transitions."""
    stops = {row['entry_date']:row.get('stop_price') for row in account.get('holding_snapshots',[])}
    held, episode, result = 0,None,[]
    for fill in account.get('ledger',[]):
        if fill['action']=='BUY':
            if held:raise ValueError('episode unexpectedly adds shares')
            held=int(fill['shares']);debit=-float(fill['cash_flow']);stop=stops.get(fill['date'])
            episode={'signal_date':fill['signal_date'],'entry_date':fill['date'],'entry_cash_debit':debit,
                'entry_shares':held,'exit_proceeds':0.,'stop_price':stop,
                'initial_risk_cash':debit-held*float(stop) if stop is not None else None,'fills':[]}
        elif episode is None:raise ValueError('episode SELL has no actual BUY')
        else:
            held-=int(fill['shares']);episode['exit_proceeds']+=float(fill['cash_flow'])
        episode['fills'].append(copy.deepcopy(fill))
        if held==0:
            net=episode['exit_proceeds']/episode['entry_cash_debit']-1
            risk=episode['initial_risk_cash'];pnl=episode['exit_proceeds']-episode['entry_cash_debit']
            result.append({**episode,'exit_date':fill['date'],'complete':True,'net_return':net,'net_pnl':pnl,
                'r_multiple':pnl/risk if risk is not None and risk>0 else None})
            episode=None
    if episode:
        result.append({**episode,'exit_date':None,'complete':False,'net_return':None,'net_pnl':None,
                       'r_multiple':None,'remaining_shares':held})
    return result


def _cash_failure(symbol: str, day_start: str, day_end: str, calendar: list[str], initial_cash: float,
                  reason: str) -> dict:
    days = [day for day in calendar if day_start <= day <= day_end]
    return {"ledger": [], "rejections": [], "holding_snapshots": [], "exit_policy_diagnostics": [],
        "daily": [{"date": day, "cash": initial_cash, "shares": 0, "close": None, "market_value": 0.,
                   "equity": initial_cash, "target_weight": 0.} for day in days],
        "actual_position": None, "sell_pending": None,
        "metrics": {"total_return": 0., "max_drawdown": 0., "final_equity": initial_cash,
                    "trade_count": 0, "rejection_count": 0, "fees": 0., "open_shares": 0, "days": len(days)},
        "failure": {"symbol": symbol, "reason": reason, "cash_retained": True}}


def _persist_account(directory: Path, account: dict, summary: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for key in ("ledger", "daily", "rejections", "holding_snapshots", "exit_policy_diagnostics", "entry_diagnostics", "episodes"):
        path = directory / (key + ".json")
        _save(path, account.get(key, [])); artifacts[key] = str(path)
    _save(directory / "open_position.json", {"actual_position": account.get("actual_position"),
        "open_shares": account["metrics"]["open_shares"], "sell_pending": account.get("sell_pending")})
    summary["artifacts"] = artifacts
    _save(directory / "summary.json", summary)
    return summary


def _worker_init() -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _evaluate_stock(payload: dict) -> dict:
    """Each process owns its own frozen inference cache and actual accounts."""
    _worker_init()
    symbol, calendar = payload["symbol"], payload["calendar"]
    run_root, report = read_run(payload["model_run_id"])
    cfg = report["config"]; initial_cash = float(cfg["initial_cash"])
    output = Path(payload["output"]) / "accounts" / symbol.replace(".", "_")
    windows = payload["windows"]; ages = payload["ages"]; costs = payload["costs"]
    accounts, failures, facts = [], [], {}
    try:
        item = _prepare(symbol, payload["end"], calendar)
        overlay = runtime_overlay(item["features"], calendar, model_run_id=payload["model_run_id"],
                                  as_of=payload["end"], device="cpu")
        from my_strategy.services.czsc_signal_points_v4 import build_signal_points
        points = build_signal_points(overlay["features"], cfg, native_pack=overlay["native_pack"],
                                      ma_routes={row["date"]: row for row in overlay["ma_routes"]})
        settings = strategy_config(cfg["strategy"])
        exit_template = pinned_exit_policy(overlay["features"], calendar, points, run_root,
                                          report["ma_bundles"], cfg)
        native_available = True
        facts.update(v4_full_prefix_rows=len(overlay["rows"]), v4_events=len(overlay["signal_points"]),
            v4_wyckoff_events=len(overlay["wyckoff_events"]), v4_policy=overlay["candidate_evidence_policy"],
            v4_compute=overlay["compute_info"], data_end=item["data_end"], data_version=item["data_version"])
    except Exception as exc:
        native_available = False
        failures.append({"symbol": symbol, "variant": "v4", "error": repr(exc)})
    old_overlay = None
    if payload["control_model_run_id"]:
        try:
            from my_strategy.web_dashboard.high_return import _overlay, _display_model_lineage
            from my_strategy.services.czsc_candidate_evidence import recover_daily_evidence
            from my_strategy.services.czsc_high_return_runtime import _read_run
            if not native_available:
                item = _prepare(symbol, payload["end"], calendar)
            original = _overlay(item, calendar, model_run_id=payload["control_model_run_id"], as_of=payload["end"])
            recovered = recover_daily_evidence(original, item["features"], calendar, usage_mode="historical")
            legacy_identity = _display_model_lineage(recovered)
            _, old_report, _ = _read_run(payload["control_model_run_id"])
            old_cfg = copy.deepcopy(old_report["config"])
            old_settings = strategy_config(old_cfg["strategy"])
            old_overlay = legacy_identity
            facts["v3_model_run_id"] = payload["control_model_run_id"]
        except Exception as exc:
            failures.append({"symbol": symbol, "variant": "v3", "error": repr(exc)})
    for window_name, (start, requested_end) in windows.items():
        end = min(requested_end, payload["end"])
        if start > end:
            continue
        for version, available, source_overlay in (("v4", native_available, overlay if native_available else None),
                                                   ("v3", old_overlay is not None, old_overlay)):
            if version == "v3" and not payload["control_model_run_id"]:
                continue
            for age in ages:
                consensus, decisions, blocked = None, None, []
                if available:
                    try:
                        local_overlay = _clip_overlay(source_overlay, end)
                        raw = item["raw"].loc[pd.to_datetime(item["raw"].date) <= end].reset_index(drop=True)
                        features = (source_overlay["features"] if version == "v4" else item["features"])
                        features = features.loc[pd.to_datetime(features.date) <= end].reset_index(drop=True)
                        base = _base_result(raw, symbol, end, start)
                        if version == "v4":
                            from my_strategy.web_dashboard.chart_evidence_v4 import decorate_analysis
                            chart = decorate_analysis(base, raw=raw, features=features, overlay=local_overlay,
                                                      consensus_config={"evidence_valid_bars": age})
                            strategy = settings; config = cfg
                        else:
                            from my_strategy.web_dashboard.chart_evidence import decorate_analysis
                            base["events"] = local_overlay["signal_points"]
                            chart = decorate_analysis(base, raw=raw, features=features, overlay=local_overlay,
                                consensus_config={"evidence_valid_bars": age}, consensus_version=CONTROL_VERSION)
                            strategy = old_settings; config = old_cfg
                        consensus = chart["consensus_decisions"]
                        decisions, blocked = map_final_points_to_decisions(local_overlay["rows"], consensus,
                            symbol=symbol, start=start, end=end, max_weight=float(strategy["execution"]["max_weight"]))
                        if len(decisions) != len(raw):
                            raise ValueError("chart final decisions/raw rows are not aligned")
                        _save(output / window_name / f"{version}_age{age}" / "consensus.json", consensus)
                        _save(output / window_name / f"{version}_age{age}" / "blocked_final_points.json", blocked)
                        _save(output / window_name / f"{version}_age{age}" / "broker_decisions.json", decisions)
                    except Exception as exc:
                        failures.append({"symbol": symbol, "window": window_name, "variant": version,
                                         "evidence_bars": age, "error": repr(exc)})
                        decisions = None
                for multiplier in costs:
                    account_id = f"{payload['run_id']}:{symbol}:{window_name}:{version}:age{age}:slip{multiplier}"
                    summary = {"symbol": symbol, "window": window_name, "start": start, "end": end,
                        "variant": version, "evidence_bars": age, "slippage_multiplier": multiplier,
                        "initial_cash": initial_cash, "publication_allowed": False, "status": "cash_failure",
                        "final_point_count": len([e for e in (consensus or {}).get("events", []) if start <= e['date'] <= end]),
                        "blocked_final_buys": sum(row['reason']!='chart_sell_outside_frozen_source_exit_policy' for row in blocked),
                        "raw_source_exit_days": sum(row['raw_source_exit'] for row in decisions or []),
                        "chart_sell_days": sum(row['chart_sell'] for row in decisions or []),
                        "chart_sell_source_exit_overlap": sum(row['raw_source_exit'] and row['chart_sell'] for row in decisions or []),
                        "chart_sell_outside_frozen_source_exit_policy": sum(row['reason']=='chart_sell_outside_frozen_source_exit_policy' for row in blocked),
                        "holding_policy": "raw_union_source_exits_plus_signal_pinned_native_union_ma_exit_hard_stop_max60" if version == "v4" else "frozen_v3_raw_union_source_rule_exit_plus_hard_stop_max60"}
                    if decisions is not None:
                        try:
                            execution = copy.deepcopy(strategy)
                            execution["execution"]["slippage"] *= multiplier
                            index = {day: i for i, day in enumerate(calendar)}
                            if version == "v4":
                                extra_policy = exit_template.new_replay()
                                policy_hash = extra_policy.recipe_hash
                            else:
                                extra_policy = None
                                policy_hash = stable_hash({"v3_union_rule": old_cfg, "model_run_id": payload["control_model_run_id"]})
                            maximum = int(config["high_return"]["max_holding_sessions"])
                            callback = _max_exit_callback(index, maximum, policy_hash, extra_policy=extra_policy)
                            account = execute_decisions(symbol, raw, decisions, start, initial_cash, execution, account_id,
                                market_dates=calendar, verified_sources=VERIFIED_SOURCES, entry_policy="risk", exit_policy=callback)
                            checked = verify_actual_account(account, calendar=calendar, decisions=decisions,
                                                            initial_cash=initial_cash, symbol=symbol)
                            account['episodes']=_actual_episodes(account)
                            complete=[e for e in account['episodes'] if e['complete']]
                            summary.update(status="completed", checks=checked, metrics=account["metrics"],
                                turnover=_turnover(account, initial_cash),
                                buy_fills=sum(row['action']=='BUY' for row in account['ledger']),
                                sell_fills=sum(row['action']=='SELL' for row in account['ledger']),
                                completed_round_trips=len(complete),win_rate=sum(e['net_return']>0 for e in complete)/len(complete) if complete else None,
                                rejection_reasons=dict(Counter(row["reason"] for row in account["rejections"])),
                                execution_config_hash=stable_hash(execution), consensus_config_hash=consensus['config_hash'])
                        except Exception as exc:
                            reason = repr(exc)
                            failures.append({"symbol": symbol, "window": window_name, "variant": version,
                                "evidence_bars": age, "slippage_multiplier": multiplier, "error": reason})
                            account = _cash_failure(symbol, start, end, calendar, initial_cash, reason)
                    else:
                        reason = next((f["error"] for f in reversed(failures) if f["variant"] == version), "source_unavailable")
                        account = _cash_failure(symbol, start, end, calendar, initial_cash, reason)
                    if summary["status"] != "completed":
                        summary.update(metrics=account['metrics'],turnover=0.,buy_fills=0,sell_fills=0,
                                       failure=account.get('failure'), rejection_reasons={})
                    account_dir = output / window_name / f"{version}_age{age}" / f"slippage_{multiplier:g}"
                    accounts.append(_persist_account(account_dir, account, summary))
    stock = {"symbol": symbol, "facts": facts, "accounts": accounts, "failures": failures}
    _save(output / "stock_summary.json", stock)
    return stock


def aggregate_accounts(stocks: list[dict], requested: list[str], initial_cash: float) -> list[dict]:
    """Fixed equal cash accounts; failed selections remain cash, never disappear."""
    by_variant: dict[tuple, list] = {}
    for stock in stocks:
        for account in stock['accounts']:
            key = (account['window'],account['variant'],account['evidence_bars'],account['slippage_multiplier'])
            by_variant.setdefault(key,[]).append(account)
    output = []
    for key, accounts in sorted(by_variant.items()):
        # Failure accounts are already in the denominator and equity files.
        rows_by_account = []
        for account in accounts:
            rows = json.loads(Path(account['artifacts']['daily']).read_text())
            rows_by_account.append({row['date']:float(row['equity']) for row in rows})
        dates = sorted({day for rows in rows_by_account for day in rows})
        daily = {day:0. for day in dates}
        for rows in rows_by_account:
            last_observed = initial_cash
            for day in dates:
                last_observed = rows.get(day,last_observed)
                daily[day] += last_observed
        equity = pd.Series([initial_cash*len(requested), *[daily[d] for d in sorted(daily)]],dtype=float)
        output.append({'window':key[0],'variant':key[1],'evidence_bars':key[2],'slippage_multiplier':key[3],
            'requested_accounts':len(requested),'retained_accounts':len(accounts),
            'completed_accounts':sum(a['status']=='completed' for a in accounts),'failed_cash_accounts':sum(a['status']!='completed' for a in accounts),
            'total_return':float(equity.iloc[-1]/equity.iloc[0]-1),
            'max_drawdown':float((equity/equity.cummax()-1).min()),
            'trade_count':sum(a['metrics']['trade_count'] for a in accounts),
            'rejection_count':sum(a['metrics']['rejection_count'] for a in accounts),
            'open_accounts':sum(a['metrics']['open_shares']>0 for a in accounts),
            'fees':sum(a['metrics']['fees'] for a in accounts),'mean_turnover':float(np.mean([a['turnover'] for a in accounts])),
            'completed_round_trips':sum(a.get('completed_round_trips',0) for a in accounts),
            'blocked_final_buys':sum(a.get('blocked_final_buys',0) for a in accounts),
            'chart_sell_outside_frozen_source_exit_policy':sum(a.get('chart_sell_outside_frozen_source_exit_policy',0) for a in accounts),
            'basis':'independent_equal_initial_cash_accounts_no_shared_cash',
            'missing_quote_valuation':'carry_last_observed_account_equity_no_future_quotes',
            'daily':[{'date':d,'equity':daily[d]} for d in sorted(daily)]})
    return output


def _cohort(value: str, report: dict) -> list[str]:
    if value == 'controls':
        values = list(CONTROL_STOCKS)
    elif value == 'training':
        values = report['source_snapshot']['selected_symbols']
    elif Path(value).is_file():
        loaded = json.loads(Path(value).read_text())
        values = loaded if isinstance(loaded,list) else loaded.get('symbols',loaded.get('selected_symbols'))
        if not isinstance(values,list):raise ValueError('cohort file needs an explicit frozen symbol list')
    else:
        values = value.split(',')
    if not values:
        raise ValueError('cohort cannot be empty')
    return list(dict.fromkeys(normalize_symbol(str(symbol)) for symbol in values))


def main(argv=None) -> dict:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run-id',required=True)
    parser.add_argument('--cohort',default='controls',help='controls, training, frozen JSON path, or comma-separated symbols')
    parser.add_argument('--control-model-run-id',default=OLD_MODEL_RUN)
    parser.add_argument('--skip-v3-control',action='store_true')
    parser.add_argument('--cpu-workers',type=int,default=4)
    parser.add_argument('--calendar-run-id')
    parser.add_argument('--end',default='2026-10-09')
    parser.add_argument('--windows',default=','.join(DEVELOPMENT_WINDOWS),help='comma-separated predefined development windows')
    parser.add_argument('--evidence-bars',default='1,3,5')
    parser.add_argument('--slippage-multipliers',default='1,2')
    args=parser.parse_args(argv)
    run_root,report=read_run(args.model_run_id)
    if report is None:raise ValueError('a completed frozen native model run is required')
    calendar,calendar_id=_calendar(args.calendar_run_id)
    symbols=_cohort(args.cohort,report);names=args.windows.split(',')
    if len(names)!=len(set(names)) or set(names)-set(DEVELOPMENT_WINDOWS):raise ValueError('unknown or repeated development window')
    windows={name:DEVELOPMENT_WINDOWS[name] for name in names}
    ages=[int(v) for v in args.evidence_bars.split(',')]
    costs=[float(v) for v in args.slippage_multipliers.split(',')]
    if not ages or len(set(ages))!=len(ages) or set(ages)-{1,3,5}:raise ValueError('evidence bars must be unique 1,3,5')
    if not costs or len(set(costs))!=len(costs) or any(v not in {1.,2.} for v in costs):raise ValueError('slippage multipliers must be unique normal or double')
    if args.cpu_workers<1:raise ValueError('cpu-workers must be positive')
    end=pd.Timestamp(args.end).date().isoformat()
    spec={'version':VERSION,'model_run_id':args.model_run_id,'control_model_run_id':None if args.skip_v3_control else args.control_model_run_id,
        'symbols':symbols,'cohort_hash':stable_hash(symbols),'windows':windows,'evidence_bars':ages,'slippage_multipliers':costs,
        'calendar_run_id':calendar_id,'calendar_hash':stable_hash(calendar),'raw_end':end,
        'frozen_model_feature_end':report['source_snapshot']['source_data_end'],
        'initial_cash_per_account':float(report['config']['initial_cash']),'read_windows_development_only':True,'publication_allowed':False,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'price_basis':'unadjusted_no_historical_ST_or_corporate_action_identity',
        'replay_basis':'final_B_entry_gate_same_risk_plan_frozen_raw_union_exit_continuation_full_causal_prefix_independent_fresh_cash_each_window',
        'cost_stress':'double_slippage_only_commission_minimum_and_stamp_tax_unchanged',
        'prediction_basis':'final_display_BS_with_frozen_initial_flat_chart_state_actual_Broker_cash_and_inventory_replayed'}
    context=create_run_context(task='czsc-native-v4-broker-validation',as_of_date=end,config=spec,
        data_version=report['source_snapshot']['snapshot_hash'],stocks=symbols,start_date=min(v[0] for v in windows.values()),end_date=end)
    _save(context.subdir('config')/'validation.json',spec)
    print('RUN_ID '+context.run_id,flush=True)
    payload={'model_run_id':args.model_run_id,'control_model_run_id':spec['control_model_run_id'],
        'calendar':calendar,'windows':windows,'ages':ages,'costs':costs,'end':end,'output':str(context.run_dir()),'run_id':context.run_id}
    results=[]
    # The three representative controls finish before widening the frozen cohort.
    controls=[symbol for symbol in CONTROL_STOCKS if symbol in symbols]
    remaining=[symbol for symbol in symbols if symbol not in controls]
    def accept(stock):
        results.append(stock)
        print(json.dumps({'completed':len(results),'requested':len(symbols),'symbol':stock['symbol'],
                          'accounts':len(stock['accounts']),'failures':len(stock['failures'])}),flush=True)
    for symbol in controls:
        accept(_evaluate_stock({**payload,'symbol':symbol}))
    if args.cpu_workers==1:
        for symbol in remaining:accept(_evaluate_stock({**payload,'symbol':symbol}))
    elif remaining:
        with ProcessPoolExecutor(max_workers=args.cpu_workers,mp_context=multiprocessing.get_context('spawn'),initializer=_worker_init) as pool:
            pending={pool.submit(_evaluate_stock,{**payload,'symbol':symbol}):symbol for symbol in remaining}
            for future in as_completed(pending):
                symbol=pending[future]
                try:accept(future.result())
                except Exception as exc:
                    # A top-level worker error also retains its selected cash.
                    stock={'symbol':symbol,'accounts':[],'facts':{},'failures':[{'variant':'worker','error':repr(exc)}]}
                    for name,(start,finish) in windows.items():
                        finish=min(finish,end)
                        if start>finish:continue
                        for variant in ('v4','v3') if spec['control_model_run_id'] else ('v4',):
                            for age in ages:
                                for cost in costs:
                                    account=_cash_failure(symbol,start,finish,calendar,spec['initial_cash_per_account'],repr(exc))
                                    summary={'symbol':symbol,'window':name,'variant':variant,'evidence_bars':age,'slippage_multiplier':cost,
                                        'status':'cash_failure','metrics':account['metrics'],'turnover':0.,'failure':account['failure']}
                                    directory=context.run_dir()/'accounts'/symbol.replace('.','_')/name/f'{variant}_age{age}'/f'slippage_{cost:g}'
                                    stock['accounts'].append(_persist_account(directory,account,summary))
                    accept(stock)
    order={s:i for i,s in enumerate(symbols)};results.sort(key=lambda stock:order[stock['symbol']])
    aggregate=aggregate_accounts(results,symbols,spec['initial_cash_per_account'])
    report_out={'run_id':context.run_id,'config':spec,'status':'completed' if all(not s['failures'] for s in results) else 'completed_with_failures_retained',
        'stock_records':[{k:v for k,v in stock.items() if k!='accounts'} for stock in results],
        'account_summaries':[a for stock in results for a in stock['accounts']],
        'aggregate':aggregate,'actual_account_count':sum(len(s['accounts']) for s in results),
        'failed_cash_accounts':sum(a['status']!='completed' for stock in results for a in stock['accounts']),
        'independent_window_status':'pending','publication_allowed':False}
    _save(context.subdir('reports')/'broker_validation.json',report_out)
    context.write_metadata({'status':report_out['status'],'config':spec,'publication_allowed':False})
    print(json.dumps({'run_id':context.run_id,'status':report_out['status'],'stocks':len(results),
        'actual_accounts':report_out['actual_account_count'],'failed_cash_accounts':report_out['failed_cash_accounts'],
        'report':str(context.subdir('reports')/'broker_validation.json')}),flush=True)
    return report_out


if __name__=='__main__':
    main()
