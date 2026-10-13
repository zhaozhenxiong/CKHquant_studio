"""Reference scores have causal prices and no simulated account population."""
from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import czsc_trend_runtime as runtime
from my_strategy.services.czsc_trend_contracts import strategy_config

DATES = ["2026-09-28", "2026-09-29", "2026-09-30"]


@pytest.fixture
def reference_inputs(monkeypatch):
    from my_strategy.services import czsc_native_evidence, czsc_trend_model, czsc_trend_pipeline, czsc_trend_account
    frame = pd.DataFrame([{"symbol": "000001.SZ", "date": day, "close": 10. + i,
        "open": 10. + i, "high": 10.2 + i, "low": 9.8 + i} for i, day in enumerate(DATES)])
    phases = [("BUY", .9, .05), ("SELL", .05, .9), ("BUY", .9, .05)]

    def native(data, cfg, **kwargs):
        evaluations = []
        for i, bar in enumerate(data.to_dict("records")):
            day = bar["date"]
            _, buy, sell = phases[DATES.index(day)]
            at = day + "T15:00:00+08:00"
            evaluations.append({"date": day, "available_at": at, **{source: {
                "date": day, "available_at": at, "snapshot_id": source + ":" + day,
                "input_eligible": True, "support_facts": [], "current_state": {
                    "close": bar["close"], "buy_score": buy, "sell_score": sell,
                    "support_valid": True, "support_floor": bar["close"] - 1,
                    "resistance_valid": True, "resistance_ceiling": bar["close"] + 1}}
                for source in ("structure", "wyckoff")}})
        return {"frame": data.copy(), "evaluations": evaluations, "events": [], "wyckoff_events": []}

    def trends(data, *args, **kwargs):
        return {day: {"date": day, "available_at": day + "T15:00:00+08:00",
            "input_eligible": True, "model_available": False,
            "p_up": None, "p_down": None, "p_neutral": None,
            "model_identity": {}, "reason_codes": ["no_prior_trend_head"]} for day in data.date}

    monkeypatch.setattr(runtime, "load_model_run", lambda *a, **k: (None, None))
    monkeypatch.setattr(czsc_native_evidence, "build_native_evidence", native)
    monkeypatch.setattr(czsc_trend_model, "bind_trend_inputs", lambda data, dates: data.copy())
    monkeypatch.setattr(czsc_trend_pipeline, "forward_trend", trends)
    monkeypatch.setattr(czsc_trend_account, "replay_trend_account", lambda *a, **k: pytest.fail("reference must never replay an account"))
    monkeypatch.setattr(czsc_trend_pipeline, "predict_entry", lambda *a, **k: pytest.fail("reference is not the actual-flat entry population"))
    monkeypatch.setattr(czsc_trend_pipeline, "pinned_exit_policy", lambda *a, **k: pytest.fail("reference has no actual held position"))
    return frame


def test_independent_reference_points_use_real_multisource_support_without_account(reference_inputs):
    out = runtime.reference_overlay(reference_inputs, DATES)
    assert out["analysis_mode"] == "reference" and out["account_scope"] == "none"
    assert out["account_actions"] is None and out["compute_info"]["account_simulated"] is False
    assert [point["side"] for point in out["reference_points"]] == ["BUY", "SELL", "BUY"]
    # No original candidate, risk plan or previous inventory is needed to
    # express continuously supported independent daily references.
    for i, row in enumerate(out["rows"]):
        assert row["reference_only"] is True and "holding" not in row and "account_action" not in row
        assert "entry_plan" not in row and "entry_eligible" not in row
        assert row["high_return_probability"] is row["exit_probability"] is None
        assert row["agents"]["high_return"]["status"] == "reference_out_of_scope"
        assert row["agents"]["ma"]["p_up"] is None
        assert row["agents"]["ma"]["buy_contribution"] is None
        buy, sell = (.9, .05) if i != 1 else (.05, .9)
        assert row["buy_base_score"] == pytest.approx(.2 * buy + .3 * buy)
        assert row["sell_base_score"] == pytest.approx(.2 * sell + .3 * sell)
        assert row["reference_recommendation"]["reference_price"] == 10. + i
    cfg = strategy_config()
    assert out["trend_consensus"]["config"]["buy_base_threshold"] == cfg["consensus"]["buy_base_threshold"]
    assert cfg["high_return"]["min_net_return"] == .08 and cfg["high_return"]["min_r_multiple"] == 2


def test_reference_cutoff_excludes_unclosed_day_and_future_mutations(reference_inputs):
    first = runtime.reference_overlay(reference_inputs, DATES, as_of=DATES[1] + "T14:59:00+08:00")
    altered = reference_inputs.copy()
    altered.loc[altered.date > DATES[0], ["open", "high", "low", "close"]] *= 100
    second = runtime.reference_overlay(altered, DATES, as_of=DATES[0])
    assert first["rows"] == second["rows"]
    assert first["reference_points"] == second["reference_points"]
    assert first["latest_decision"]["date"] == DATES[0]
    assert first["latest_decision"]["reference_close"] == 10.
    assert first["reference_points"][0]["first_available_at"] == DATES[0] + "T15:00:00+08:00"


def test_missing_market_session_never_reuses_old_reference(reference_inputs):
    out = runtime.reference_overlay(reference_inputs.iloc[:2], DATES, as_of=DATES[-1])
    assert out["latest_decision"]["direction"] == "UNAVAILABLE"
    assert out["latest_decision"]["action"] == "WAIT"
    assert out["latest_decision"]["reference_close"] is None
    assert out["latest_decision"]["reason_codes"] == ["missing_target_market_session"]
    assert out["reference_recommendation"]["reference_price"] is None


@pytest.mark.parametrize("selected", [None, "qualified-incumbent"])
def test_fixed_reference_cannot_use_unqualified_candidate(reference_inputs, monkeypatch, selected):
    from my_strategy.services.model_training_center import TrainingCenter
    monkeypatch.setattr(TrainingCenter, "active_model_run_id", lambda *a, **k: selected)
    monkeypatch.setattr(runtime, "load_model_run", lambda *a, **k: pytest.fail("unqualified fixed candidate must fail before loading"))
    with pytest.raises(ValueError, match="已合格且生效"):
        runtime.reference_overlay(reference_inputs, DATES, model_run_id="unqualified-candidate")


def test_one_source_cannot_spend_two_independent_budgets(reference_inputs, monkeypatch):
    from my_strategy.services import czsc_native_evidence
    original = czsc_native_evidence.build_native_evidence
    def auxiliary(*args, **kwargs):
        pack = original(*args, **kwargs)
        for row in pack["evaluations"]:
            row["wyckoff"]["auxiliary"] = True
            row["structure"]["input_eligible"] = False
        return pack
    monkeypatch.setattr(czsc_native_evidence, "build_native_evidence", auxiliary)
    out = runtime.reference_overlay(reference_inputs, DATES)
    assert out["reference_points"] == []
    assert all(row["direction"] == "WAIT" for row in out["rows"])


def test_structure_only_reference_never_loads_or_predicts_models(reference_inputs, monkeypatch):
    from my_strategy.services import czsc_trend_pipeline
    monkeypatch.setattr(runtime, "load_reference_model_run", lambda *a, **k: pytest.fail("structure-only must not resolve models"))
    monkeypatch.setattr(czsc_trend_pipeline, "forward_trend", lambda *a, **k: pytest.fail("structure-only must not infer models"))
    out = runtime.reference_overlay(reference_inputs, DATES, use_models=False,
        model_run_id="ignored-candidate", model_fold="ignored-fold")
    assert out["structure_only"] is True and out["models_enabled"] is False
    assert out["model_run_id"] is None and out["account_actions"] is None
    assert [point["side"] for point in out["reference_points"]] == ["BUY", "SELL", "BUY"]
    assert all(row["agents"]["ma"]["p_up"] is None for row in out["rows"])
    assert all(row["agents"]["ma"]["reason_codes"] == ["models_disabled_for_structure_reference"] for row in out["rows"])


def test_scan_freezes_missing_selection_across_worker_inference(reference_inputs, monkeypatch):
    monkeypatch.setattr(runtime, "load_reference_model_run", lambda *a, **k: pytest.fail("scan must retain its initially missing selection"))
    out = runtime.reference_overlay(reference_inputs, DATES, model_selection_frozen=True)
    assert out["model_run_id"] is None and out["models_enabled"] is True
    assert all(row["agents"]["ma"]["p_up"] is None for row in out["rows"])


def test_fixed_reference_pins_the_actual_direction_head_and_rejects_future_head(reference_inputs, monkeypatch):
    from my_strategy.services import czsc_trend_pipeline
    bundles = [{"name": "trend_prior", "available_at": "2026-06-30", "heads": {"trend_direction": {"path": "prior"}}},
        {"name": "trend_latest", "available_at": "2026-09-28", "heads": {"trend_direction": {"path": "latest"}}},
        {"name": "trend_future", "available_at": "2026-09-30", "heads": {"trend_direction": {"path": "future"}}}]
    monkeypatch.setattr(runtime, "load_reference_model_run", lambda *a, **k: (Path("qualified"),
        {"config": strategy_config(), "trend_bundles": bundles, "model_bundles": []}))
    original = czsc_trend_pipeline.forward_trend
    seen = []
    def infer(frame, root, selected, *args, **kwargs):
        seen.append(selected)
        return original(frame, root, selected, *args, **kwargs)
    monkeypatch.setattr(czsc_trend_pipeline, "forward_trend", infer)
    latest = runtime.reference_overlay(reference_inputs, DATES, model_fold="production")
    assert [bundle["name"] for bundle in seen[-1]] == ["trend_latest"]
    assert latest["latest_decision"]["resolved_model_fold"] == "trend_latest"
    runtime.reference_overlay(reference_inputs, DATES, model_fold="trend_prior")
    assert [bundle["name"] for bundle in seen[-1]] == ["trend_prior"]
    with pytest.raises(ValueError, match="尚不可用"):
        runtime.reference_overlay(reference_inputs, DATES, model_fold="trend_future")


def test_reference_chart_and_market_history_never_project_account_fields(reference_inputs):
    from my_strategy.web_dashboard.chart_evidence_trend import decorate_analysis
    from my_strategy.services.czsc_market_history import MarketHistoryBuilder
    overlay = runtime.reference_overlay(reference_inputs, DATES)
    result = {"symbol": "000001.SZ", "data_end": DATES[-1], "price_basis": "unadjusted",
        "frequencies": {"日线": {"bars": [{"time": day} for day in DATES[1:]]}}}
    chart = decorate_analysis(result, raw=reference_inputs, overlay=overlay, start=DATES[1])
    assert chart["account_actions"] is None
    assert len(chart["reference_points"]) == 2
    assert chart["research"]["account_simulated"] is False
    assert "account_context" not in chart["research"]
    assert chart["chart_evidence"]["statistics"]["visible_window"]["broker_fills"] == 0
    assert all("account_action" not in row for row in chart["chart_evidence"]["rows"])
    history = MarketHistoryBuilder(["000001.SZ", "failed"], DATES, DATES[-1], run_id="reference",
        contract={"analysis_mode": "reference", "account_scope": "none"})
    facts = history.add_stock("000001.SZ", overlay=overlay)
    failed = history.add_stock("failed", failure={"error": "no prices"})
    summary = history.finish()
    assert summary["analysis_mode"] == "reference" and summary["account_scope"] == "none"
    assert summary["daily"][0]["counts"]["direction_buy"] == 1
    assert summary["daily"][1]["counts"]["direction_sell"] == 1
    assert summary["daily"][0]["hr_status"] == {}
    assert all(fact["account_available"] is False and fact["holding_shares"] is None for fact in facts + failed)


def test_worker_dispatch_keeps_broker_only_for_explicit_backtest(monkeypatch):
    from my_strategy.web_dashboard import trend_strategy
    seen = []
    monkeypatch.setattr(trend_strategy, "_source_vintage", lambda *a, **k: ({"features": pd.DataFrame({"date": DATES})}, "source", DATES[-1]))
    monkeypatch.setattr(runtime, "reference_overlay", lambda *a, **k: seen.append(("reference", k)) or {})
    monkeypatch.setattr(runtime, "runtime_overlay", lambda *a, **k: seen.append(("backtest", k)) or {})
    trend_strategy._stock_worker(("a", DATES[-1], DATES, {}, DATES[-1]))
    trend_strategy._stock_worker(("a", DATES[-1], DATES, {"analysis_mode": "backtest", "initial_cash": 100000.}, DATES[-1]))
    assert [kind for kind, _ in seen] == ["reference", "backtest"]
    assert "initial_cash" not in seen[0][1] and seen[1][1]["initial_cash"] == 100000.


def test_native_reference_chart_preserves_geometry_without_creating_position(tmp_path, monkeypatch):
    from my_strategy.adapters.czsc_adapter import load_bars
    from my_strategy.services.czsc_analysis import IntentSession, run_session, strategy_config as native_config
    from my_strategy.web_dashboard.trend_strategy import _reference_structure
    path = tmp_path / "raw.db"
    days = pd.bdate_range("2020-01-02", periods=130)
    prices = 12 + np.arange(len(days)) * .006 + np.sin(np.arange(len(days)) / 8)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE stock_daily_normalized (stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,has_trade_price INTEGER,source TEXT)")
        conn.executemany("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)", [
            ("000001.SZ", day.date().isoformat(), float(price), float(price + .2), float(price - .2),
                float(price + .04), 1000000., float(price * 1000000), 1, "baostock") for day, price in zip(days, prices)])
    raw = load_bars("000001.SZ", db_path=path)
    cutoff = days[105].date().isoformat()
    observed = raw[raw.date <= pd.Timestamp(cutoff)].copy()
    baseline = run_session(observed, native_config()).payload()
    monkeypatch.setattr(IntentSession, "new_position", lambda *a: pytest.fail("reference charts must not create Position"))
    actual = _reference_structure(raw, start=None, decision_date=cutoff)
    for frequency in ("日线", "周线", "月线"):
        for field in ("bars", "fractals", "pens", "zones", "indicators", "divergences"):
            assert actual["frequencies"][frequency][field] == baseline[frequency][field]
        assert actual["frequencies"][frequency]["signals"] == []
    assert actual["data_end"] == cutoff and actual["events"] == []
