"""Offline diagnostic comparison of saved chart directions through the Broker.

Every symbol/window is a separate flat account. This is a new diagnostic policy,
not the frozen continuation policy behind the existing L2 target and not a model
promotion. It never writes training data, old ledgers, or publication aliases.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES


POLICY = "chart_consensus_next_open_diagnostic_v1"
WINDOWS = (("2025Q2", "2025-04-01", "2025-06-30"),
           ("2025Q4", "2025-10-01", "2025-12-31"),
           ("2026Q2", "2026-04-01", "2026-06-30"),
           ("2026Q3", "2026-07-01", "2026-09-30"))


def broker_decisions(frame, evidence_rows, consensus_rows, *, start, end):
    """Map final points to guarded one-session plans; WAIT preserves inventory."""
    symbol = str(frame.iloc[0]["symbol"])
    evidence = {r["date"]: r for r in evidence_rows}
    opinions = {r["date"]: r for r in consensus_rows}
    if len(evidence) != len(evidence_rows) or len(opinions) != len(consensus_rows):
        raise ValueError("duplicate diagnostic evidence dates")
    result = []
    for day in pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d"):
        opinion = opinions.get(day, {}) if start <= day <= end else {}
        side = opinion.get("action", "WAIT")
        if side not in {"BUY", "SELL", "WAIT"}:
            raise ValueError("unknown consensus direction")
        plan = copy.deepcopy(evidence.get(day, {}).get("raw_risk_plan")) if side == "BUY" else None
        if side == "BUY" and (not isinstance(plan, dict) or plan.get("entry_allowed") is not True
                              or plan.get("signal_date") != day or plan.get("symbol") != symbol
                              or plan.get("valid_for_sessions") != 1):
            raise ValueError("final BUY lacks its exact-day executable risk plan")
        result.append({"date": day, "symbol": symbol, "available_at": day + "T15:00:00+08:00",
                       "target_weight": float(plan["target_weight"]) if plan else 0.,
                       "action": side, "position_intent": side, "candidate_action": side,
                       "hold_current_position": side == "WAIT", "eligible": side == "BUY",
                       "entry_plan": plan, "entry_policy": "risk",
                       "reason_codes": opinion.get("reason_codes", [])})
    return result


def replay(frame, evidence_rows, consensus_rows, market_dates, *, start, end,
           run_id, config=None, initial_cash=100000., slippage_multiplier=1.):
    """Execute final B/S at next verified open, retaining guards and hard stops."""
    settings = copy.deepcopy(strategy_config(config))
    settings["execution"]["slippage"] *= float(slippage_multiplier)
    frame = frame[pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d") <= end].copy()
    decisions = broker_decisions(frame, evidence_rows, consensus_rows, start=start, end=end)
    market_index = {day: i for i, day in enumerate(market_dates)}
    contract = stable_hash({"policy": POLICY, "maximum_holding_sessions": 60,
                            "execution": settings["execution"], "end": end})
    def maximum_holding(snapshot):
        age = market_index[snapshot["date"]] - market_index[snapshot["entry_date"]] + 1
        return {"head": "holding_exit", "contract_hash": contract, "probability": 1.,
                "request_exit": age >= 60, "apply": True, "research_arm": True,
                "status": "diagnostic_maximum_60_verified_sessions"}
    result = execute_decisions(str(frame.iloc[0]["symbol"]), frame, decisions, start,
                               float(initial_cash), settings, run_id, market_dates=market_dates,
                               verified_sources=VERIFIED_SOURCES, entry_policy="risk",
                               exit_policy=maximum_holding)
    for fill in result["ledger"]:
        if not fill.get("signal_date") or fill["date"] <= fill["signal_date"]:
            raise AssertionError("diagnostic fill is not after its signal")
    result["policy"] = {"version": POLICY, "hash": contract,
                        "scope": "development_diagnostic_separate_flat_account",
                        "l2_continuation_contract_matches": False, "publication_allowed": False}
    result["completed_round_trips"] = sum(r["action"] == "SELL" and r["position_after"] == 0
                                          for r in result["ledger"])
    return result


def run(symbols, *, end="2026-10-09", device="cpu"):
    from my_strategy.adapters.czsc_adapter import load_bars
    from my_strategy.web_dashboard.high_return import compare_high_return_chart_windows, _calendar
    dates, calendar_id = _calendar()
    symbols = list(dict.fromkeys(symbols))
    context = create_run_context(task="candidate-consensus-diagnostic", as_of_date=end,
        config={"policy": POLICY, "symbols": symbols, "windows": WINDOWS,
                "evidence_valid_bars": [1, 3, 5], "slippage_multipliers": [1., 2.],
                "calendar_run_id": calendar_id, "initial_cash": 100000., "device": device},
        data_version=end, stocks=symbols, start_date=WINDOWS[0][1], end_date=end)
    folder = context.subdir("diagnostics")
    accounts, failures, coverage, analysis_runs, completed_symbols = [], [], {}, {}, []
    for symbol in symbols:
        try:
            comparison = compare_high_return_chart_windows(symbol=symbol, start="2000-01-01", end=end,
                usage_mode="historical", device=device, windows=(1, 3, 5))
            analyses = {name if name == "v2" else name + "bars": value
                        for name, value in comparison["variants"].items()}
            analysis = analyses["v3_3bars"]
            frame = load_bars(symbol, end=end)
            evidence = analysis["chart_evidence"]["rows"]
            if analysis["data_end"] != end:
                raise ValueError("diagnostic symbol lacks requested end-date bar")
            variants = {name: value["consensus_decisions"] for name, value in analyses.items()}
            coverage[symbol] = {key: {"version": value["version"], "config": value["config"],
                "config_hash": value["config_hash"], "action_counts": dict(Counter(
                    r["action"] for r in value["rows"] if r["date"] >= "2025-01-01")),
                "diagnostics": value.get("funnel", {}),
                "events_sha256": stable_hash(value["events"])} for key, value in variants.items()}
            analysis_runs[symbol] = {"data_version": analysis["data_version"],
                "data_end": analysis["data_end"], "model_run_id": (analysis["next_session_decision"].get(
                    "model_resolution") or {}).get("model_run_id"), "evidence_sha256": stable_hash(evidence),
                "comparison_basis": comparison["comparison_basis"], "preparation": comparison["preparation"]}
            for variant, value in analyses.items():
                (folder / f"{symbol.replace('.', '_')}-{variant}-analysis.json").write_text(
                    json.dumps(value, ensure_ascii=False), encoding="utf-8")
            windows = [w for w in WINDOWS if w[2] <= end]
            windows.append(("2026YTD", "2026-01-01", end))
            for name, first, last in windows:
                for variant, decisions in variants.items():
                    for multiplier in (1., 2.):
                        account_id = f"{symbol.replace('.', '_')}-{name}-{variant}-slip{int(multiplier)}"
                        result = replay(frame, analyses[variant]["chart_evidence"]["rows"], decisions["rows"], dates,
                            start=first, end=last, run_id=context.run_id,
                            slippage_multiplier=multiplier)
                        account_folder = folder / account_id
                        account_folder.mkdir()
                        for key in ("ledger", "daily", "rejections"):
                            empty_columns = {"ledger": ["date", "symbol", "action", "signal_date", "fees", "position_after"],
                                "daily": ["date", "symbol", "equity", "shares"], "rejections": ["date", "symbol", "reason"]}
                            table = pd.DataFrame(result[key]) if result[key] else pd.DataFrame(columns=empty_columns[key])
                            table.to_csv(account_folder / f"{key}.csv", index=False)
                        summary = {"symbol": symbol, "window": name, "start": first, "end": last,
                            "variant": variant, "slippage_multiplier": multiplier,
                            "metrics": result["metrics"], "completed_round_trips": result["completed_round_trips"],
                            "rejection_reasons": dict(Counter(r["reason"] for r in result["rejections"])),
                            "policy": result["policy"], "ledger_sha256": stable_hash(result["ledger"]),
                            "daily_sha256": stable_hash(result["daily"]), "artifacts": str(account_folder)}
                        (account_folder / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                        accounts.append(summary)
            completed_symbols.append(symbol)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
    report = {"run_id": context.run_id, "context": context.to_dict(), "policy": POLICY,
              "status": "completed" if not failures else "completed_with_failures",
              "data_end": end, "diagnostic_only": True, "publication_allowed": False,
              "independent_validation": "pending", "shared_cash_portfolio": False,
              "l2_continuation_contract_matches": False,
              "coverage": {"requested": len(symbols), "success": len(completed_symbols), "failed": len(failures),
                           "analysis_success": len(coverage), "completed_symbols": completed_symbols,
                           "partial_symbols": sorted(set(coverage) - set(completed_symbols)),
                           "completed_accounts": len(accounts)},
              "failures": failures, "source_analyses": analysis_runs,
              "signal_diagnostics": coverage, "accounts": accounts}
    destination = folder / "report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    context.write_metadata({"status": report["status"], "report": str(destination),
                            "coverage": report["coverage"], "publication_allowed": False})
    print(json.dumps({"run_id": context.run_id, "report": str(destination),
                      "coverage": report["coverage"], "accounts": len(accounts), "failures": failures}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--end", default="2026-10-09")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from my_strategy.web_dashboard.api import clean_symbol
    report = run([clean_symbol(s) for s in args.symbols], end=pd.Timestamp(args.end).date().isoformat(), device=args.device)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
