"""Future source coverage is cash, without disguising market/execution failures."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from my_strategy.services import czsc_high_return_audit as audit
from my_strategy.services import czsc_high_return_evaluation as evaluation
from my_strategy.services.czsc_high_return_research import _selection_score
from my_strategy.services.czsc_research_models import file_hash
from my_strategy.services.czsc_trade_episodes import _cost, _settings
from my_strategy.tests.test_czsc_high_return_evaluation import inputs, actual_broker_fake
from my_strategy.tests.test_czsc_high_return_policy_options import market, run_options


def source_variant(record, path, *, first, stop=None, symbol="000002.SZ", bad_hash=False):
    raw = pd.read_parquet(record["path"]).iloc[first:stop].copy()
    raw["symbol"] = symbol; raw.to_parquet(path, index=False)
    return {"symbol": "000002.SZ", "path": str(path), "sha256": "bad" if bad_hash else file_hash(path)}


def test_verified_future_source_keeps_cash_separate_from_execution_failure(tmp_path, monkeypatch):
    records, dates, cfg = inputs(tmp_path)
    late = source_variant(records[0], tmp_path / "late.parquet", first=91)
    records.append(late); calls = []
    monkeypatch.setattr(evaluation, "_replay", actual_broker_fake(calls))
    report = evaluation.evaluate_bundles(records, dates, model_root=tmp_path, bundles=[], config=cfg,
        windows=cfg["development_windows"], arms=["reference", "full"], cpu_workers=1, output_root=tmp_path / "eval")
    assert len(calls) == 2  # Neither arm invents a Broker replay before this source exists.
    for arm, variant in report["variants"].items():
        assert variant["account_count"] == 2 and variant["metrics"]["initial_cash"] == 200000.
        assert variant["cash_retained_failures"] == []
        assert variant["market_unavailable_accounts"] == [{"symbol": "000002.SZ", "reason": evaluation.SOURCE_AFTER_WINDOW,
            "status": evaluation.CASH_NOT_OBSERVED, "source_first_date": dates[91]}]
        binding = next(r for r in report["windows"]["quarter"]["ledger_bindings"][arm] if r["symbol"] == late["symbol"])
        assert binding["status"] == evaluation.CASH_NOT_OBSERVED
        metadata = json.loads((Path(binding["path"]) / "account.json").read_text())
        assert metadata["source_first_date"] == dates[91] and not metadata["publication_allowed"]
        daily = pd.read_csv(Path(binding["path"]) / "daily.csv")
        assert daily.date.tolist() == dates[70:91] and daily.cash.eq(100000).all() and daily.equity.eq(100000).all()


@pytest.mark.parametrize("kind", ["old_missing_window", "bad_hash", "wrong_symbol"])
def test_missing_old_market_and_invalid_future_source_still_fail(tmp_path, monkeypatch, kind):
    records, dates, cfg = inputs(tmp_path)
    additional = source_variant(records[0], tmp_path / (kind + ".parquet"),
        first=0 if kind == "old_missing_window" else 91,
        stop=70 if kind == "old_missing_window" else None,
        symbol="000003.SZ" if kind == "wrong_symbol" else "000002.SZ", bad_hash=kind == "bad_hash")
    records.append(additional); monkeypatch.setattr(evaluation, "_replay", actual_broker_fake([]))
    report = evaluation.evaluate_bundles(records, dates, model_root=tmp_path, bundles=[], config=cfg,
        windows=cfg["development_windows"], arms=["reference", "full"], cpu_workers=1, output_root=tmp_path / "eval")
    for variant in report["variants"].values():
        assert len(variant["cash_retained_failures"]) == 1 and variant["market_unavailable_accounts"] == []
        assert variant["cash_retained_failures"][0]["symbol"] == "000002.SZ"
        if kind == "old_missing_window":
            assert variant["cash_retained_failures"][0]["reason"] == "no_window_market_bars"
        assert variant["metrics"]["initial_cash"] == 200000.


def test_policy_selection_retains_future_stock_budget_without_false_failure(market, tmp_path, monkeypatch):
    records, days, cfg, fold, _ = market
    late = source_variant(records[0], tmp_path / "not-yet-observed.parquet", first=96)
    pooled = [*records, late]
    reports, prepared, replays, _ = run_options(market, tmp_path, monkeypatch, records=pooled)
    assert len(prepared) == 1 and len(replays) == 4
    assert _selection_score(reports[0], cfg) == reports[0]["variants"]["full"]["metrics"]["net_return"] > 0
    for report in reports:
        for variant in report["variants"].values():
            assert variant["metrics"]["initial_cash"] == 200000. and variant["account_count"] == 2
            assert variant["cash_retained_failures"] == []
            assert len(variant["market_unavailable_accounts"]) == 1
    # The same frozen zero-execution-failure gate remains strict for a broken source.
    bad = copy.deepcopy(reports[0]); bad["variants"]["reference"]["cash_retained_failures"] = [{"symbol": late["symbol"], "reason": "invalid_source"}]
    assert _selection_score(bad, cfg) == float("-inf")


@pytest.mark.parametrize("kind", ["old_missing_window", "bad_hash", "wrong_symbol"])
def test_batched_policy_preserves_real_failure_gate(market, tmp_path, monkeypatch, kind):
    records = market[0]
    additional = source_variant(records[0], tmp_path / (kind + ".parquet"),
        first=0 if kind == "old_missing_window" else 96,
        stop=70 if kind == "old_missing_window" else None,
        symbol="000003.SZ" if kind == "wrong_symbol" else "000002.SZ", bad_hash=kind == "bad_hash")
    reports, _, _, _ = run_options(market, tmp_path, monkeypatch, records=[*records, additional])
    for report in reports:
        assert _selection_score(report, market[2]) == float("-inf")
        for variant in report["variants"].values():
            assert variant["market_unavailable_accounts"] == [] and len(variant["cash_retained_failures"]) == 1
            assert variant["metrics"]["initial_cash"] == 200000.


def late_evaluation(tmp_path, monkeypatch):
    records, dates, cfg = inputs(tmp_path)
    records.append(source_variant(records[0], tmp_path / "late.parquet", first=91))
    monkeypatch.setattr(evaluation, "_replay", actual_broker_fake([]))
    monkeypatch.setattr(audit, "ARTIFACT_RUNS_ROOT", tmp_path)
    result = evaluation.evaluate_bundles(records, dates, model_root=tmp_path, bundles=[], config=cfg,
        windows=cfg["development_windows"], arms=["reference", "full"], cpu_workers=1, output_root=tmp_path / "eval")
    return records, dates, cfg, result


def test_audit_independently_verifies_cash_coverage_and_aggregate_classification(tmp_path, monkeypatch):
    records, dates, cfg, result = late_evaluation(tmp_path, monkeypatch)
    state = audit._Audit()
    audit._audit_evaluation(tmp_path, result, dates, cfg, {r["symbol"] for r in records}, state,
        as_of=audit.signal_time("2026-10-10"), records={r["symbol"]: r for r in records})
    assert not state.errors, state.errors
    checks = {r["rule"] for r in state.checks}
    assert {"not_observed_cash_verified_source_coverage", "not_observed_cash_no_replaced_inventory",
        "portfolio_cash_coverage_classification_recomputation"} <= checks
    # A report cannot hide unavailable stock coverage in its summary.
    result["windows"]["quarter"]["variants"]["full"]["market_unavailable_accounts"] = []
    (Path(result["output_root"]) / "evaluation.json").write_text(json.dumps(result))
    state = audit._Audit()
    audit._audit_evaluation(tmp_path, result, dates, cfg, {r["symbol"] for r in records}, state,
        as_of=audit.signal_time("2026-10-10"), records={r["symbol"]: r for r in records})
    assert "portfolio_cash_coverage_classification_recomputation" in {r["rule"] for r in state.errors}


def test_an_actual_position_cannot_be_resigned_as_not_observed_cash(tmp_path, monkeypatch):
    records, dates, cfg, result = late_evaluation(tmp_path, monkeypatch)
    binding = copy.deepcopy(result["windows"]["quarter"]["ledger_bindings"]["reference"][0])
    path = Path(binding["path"]); metadata = json.loads((path / "account.json").read_text())
    binding.update(status=evaluation.CASH_NOT_OBSERVED, reason=evaluation.SOURCE_AFTER_WINDOW, source_first_date=dates[91])
    metadata.update(status=binding["status"], reason=binding["reason"], source_first_date=dates[91])
    (path / "account.json").write_text(json.dumps(metadata)); binding["account_sha256"] = file_hash(path / "account.json")
    state = audit._Audit(); settings = _settings(cfg)
    audit._audit_account(binding, dates, _cost(settings), state, record=records[0], settings=settings,
        config=cfg, window_dates=dates[70:91])
    assert {"not_observed_cash_verified_source_coverage", "not_observed_cash_no_replaced_inventory"} <= {r["rule"] for r in state.errors}
