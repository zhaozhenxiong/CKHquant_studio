"""Actual-ledger fixed-universe aggregation and frozen cost stress semantics."""
from __future__ import annotations

import copy
import hashlib
import pandas as pd
import pytest

from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services import czsc_high_return_evaluation as evaluation


def inputs(tmp_path, periods=100):
    dates = pd.bdate_range("2024-01-02", periods=periods).strftime("%Y-%m-%d").tolist()
    raw = pd.DataFrame({"date": dates, "symbol": "000001.SZ", "open": 10., "high": 10.2,
        "low": 9.8, "close": 10., "volume": 1000000., "amount": 10000000.,
        "source": "tushare", "has_trade_price": 1})
    path = tmp_path / "stock.parquet"; raw.to_parquet(path, index=False)
    records = [{"symbol": "000001.SZ", "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    cfg = {"strategy": strategy_config(), "initial_cash": 100000., "seed": 42,
        "feature_start": dates[0], "high_return": {"max_holding_sessions": 60},
        "development_windows": [{"name": "quarter", "start": dates[70], "end": dates[90]}],
        "bootstrap_repetitions": 10}
    return records, dates, cfg


def actual_broker_fake(calls):
    def replay(frame, calendar, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        cfg = kwargs["config"]; start = kwargs["start"]; arm = kwargs["arm"]
        rows = [{"date": d, "target_weight": 0., "action": "HOLD", "hold_current_position": True,
            "entry_plan": None} for d in frame.date]
        signal = calendar[70]; sell = calendar[80]
        plan = {"plan_id": "fixture", "symbol": "000001.SZ", "signal_date": signal,
            "available_at": signal + "T15:00:00+08:00", "policy": "risk", "status": "active",
            "entry_allowed": True, "valid_for_sessions": 1, "stop_price": 9.6,
            "price_floor": 9.61, "price_ceiling": 10.5, "risk_fraction": .01,
            "target_weight": .25, "reference_price": 10.}
        if arm == "reference":
            for row in rows:
                if row["date"] == signal:
                    row.update(target_weight=.25, action="BUY", entry_plan=plan, hold_current_position=False)
                elif row["date"] == sell:
                    row.update(action="SELL", hold_current_position=False)
        result = execute_decisions("000001.SZ", frame, rows, start, 100000., cfg["strategy"],
            "evaluation-test", market_dates=calendar, verified_sources=VERIFIED_SOURCES,
            entry_policy="risk", exit_policy=lambda state: None)
        result.update(candidate_points=[{"date": signal, "candidate_eligible": True}], policy_source="union")
        return result
    return replay


def test_failed_stock_keeps_its_cash_budget_and_quarter_blocks_stay_unavailable(tmp_path, monkeypatch):
    records, dates, cfg = inputs(tmp_path)
    records.append({"symbol": "000002.SZ", "path": str(tmp_path / "absent.parquet"), "sha256": "missing"})
    calls = []; monkeypatch.setattr(evaluation, "_replay", actual_broker_fake(calls))
    report = evaluation.evaluate_bundles(records, dates, model_root=tmp_path, bundles=[], config=cfg,
        windows=cfg["development_windows"], cpu_workers=1, output_root=tmp_path / "evaluation", arms=["reference", "full"])
    assert report["initial_cash"] == 200000 and report["frozen_mainboard_accounts"] == 2
    reference = report["variants"]["reference"]
    assert reference["account_count"] == 2 and len(reference["cash_retained_failures"]) == 1
    assert reference["metrics"]["completed_round_trips"] == 1
    assert reference["metrics"]["final_equity"] == pytest.approx(200000 - reference["metrics"]["fees"])
    assert report["variants"]["full"]["metrics"]["final_equity"] == 200000.
    assert all("unavailable" in ci for ci in report["paired_daily_return_uncertainty"]["full"].values())
    assert not report["model_gate"] and not report["publication_allowed"]
    binding = report["windows"]["quarter"]["ledger_bindings"]["reference"][0]
    root = __import__("pathlib").Path(binding["path"])
    assert (root / "holding_snapshots.jsonl").is_file()
    assert len(pd.read_csv(root / "ledger.csv")) == 2
    assert len(calls) == 2


def test_stress_replays_real_fees_and_preserves_bundle_thresholds(tmp_path, monkeypatch):
    records, dates, cfg = inputs(tmp_path)
    calls = []; monkeypatch.setattr(evaluation, "_replay", actual_broker_fake(calls))
    bundle = {"name": "frozen", "policy": {"entry_threshold": .55, "exit_threshold": .65}}
    before = copy.deepcopy(bundle)
    reports = [evaluation.evaluate_bundles(records, dates, model_root=tmp_path, bundles=[bundle], config=cfg,
        windows=cfg["development_windows"], cpu_workers=1, output_root=tmp_path / "evaluation",
        arms=["reference"], validation_bundle=bundle, cost_multiplier=multiplier) for multiplier in (1, 2)]
    assert bundle == before
    assert reports[1]["variants"]["reference"]["metrics"]["fees"] > reports[0]["variants"]["reference"]["metrics"]["fees"]
    assert reports[1]["variants"]["reference"]["metrics"]["net_return"] < reports[0]["variants"]["reference"]["metrics"]["net_return"]
    stressed = calls[-1]["config"]
    assert stressed["strategy"]["execution"]["slippage"] == 2 * cfg["strategy"]["execution"]["slippage"]
    assert stressed["_cost_stress_base_execution"] == cfg["strategy"]["execution"]
    assert reports[1]["thresholds_refit"] is False and reports[1]["validation_bundle_mode"] is True


def test_continuous_600_sessions_supports_frozen_160_and_240_blocks(tmp_path, monkeypatch):
    records, dates, cfg = inputs(tmp_path, periods=600)
    monkeypatch.setattr(evaluation, "_replay", actual_broker_fake([]))
    report = evaluation.evaluate_bundles(records, dates, model_root=tmp_path, bundles=[], config=cfg,
        cpu_workers=1, output_root=tmp_path / "evaluation", arms=["reference", "full"])
    assert set(report["windows"]) == {"quarter", "continuous_2023"}
    continuous = report["windows"]["continuous_2023"]
    for block, ci in continuous["paired_daily_return_uncertainty"]["full"].items():
        assert ci["block_length"] == int(block) and ci["dates"] == 600
        assert "unavailable" not in ci
    assert not continuous["model_gate"] and not report["publication_allowed"]


def test_late_window_preserves_complete_causal_event_prefix(tmp_path, monkeypatch):
    records, dates, cfg = inputs(tmp_path, periods=300)
    window = [{"name": "late", "start": dates[250], "end": dates[280]}]
    observed = []
    def replay(frame, calendar, **kwargs):
        observed.append(frame.date.tolist())
        return actual_broker_fake([])(frame, calendar, **kwargs)
    monkeypatch.setattr(evaluation, "_replay", replay)
    evaluation.evaluate_bundles(records, dates, model_root=tmp_path, bundles=[], config=cfg,
        windows=window, cpu_workers=1, output_root=tmp_path / "evaluation", arms=["full"])
    assert observed == [dates[:281]]
