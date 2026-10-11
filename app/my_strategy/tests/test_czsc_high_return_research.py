"""Frozen source and actual-episode identity checks for the new research run."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_high_return_research as research
from my_strategy.services.czsc_high_return_features import ENTRY_FEATURE_COLUMNS, ENTRY_SCHEMA_HASH
from my_strategy.services.czsc_high_return_labels import target_contract
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class Context:
    def __init__(self, root, run_id="new-shadow-research"):
        self.root, self.run_id = Path(root), run_id
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self):
        return self.root

    def subdir(self, *parts):
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def to_dict(self):
        return {"run_id": self.run_id, "artifact_root": str(self.root)}

    def write_metadata(self, value):
        write_json(self.root / "metadata.json", value)


@pytest.fixture
def frozen_source(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    root = runs / "frozen-source"
    dates = pd.bdate_range("2022-01-03", periods=180).strftime("%Y-%m-%d").tolist()
    start = dates[130]
    records = []
    for symbol in ("000001.SZ", "600001.SH"):
        frame = pd.DataFrame({"date": dates, "symbol": symbol, "open": 10., "high": 10.2,
            "low": 9.9, "close": 10., "volume": 1_000_000., "amount": 10_000_000.,
            "source": "sina_unadjusted_v1", "has_trade_price": 1,
            "model_input_eligible": True, "input_eligible": False, "structure_input_eligible": False,
            "wyckoff_input_eligible": False, "w_test": 0., "wyckoff_rule_buy": False,
            "wyckoff_rule_sell": False, "label": 1., "label_end": dates[-1],
            "label_available": True, "label_reason": "fixed_ten_sessions",
            "net_return": 500., "entry_date": dates[0], "exit_date": dates[-1],
            "entry_shares": 5000, "fees": 100.})
        for column in MA_TREND_COLUMNS:
            frame[column] = 0.
        frame["close_above_ma20"] = 1.; frame["ma20_slope5"] = .01
        frame["ma_alignment"] = 1.; frame["ret_1"] = .01; frame["atr14_ratio"] = .01
        # Raw row counts do not provide Broker seasoning across a suspension.
        frame.loc[110:129, ["volume", "amount", "has_trade_price"]] = 0
        path = root / "dataset" / (symbol.replace(".", "_") + ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        records.append({"symbol": symbol, "path": str(path), "sha256": research._file_hash(path),
            "data_end": dates[-1], "data_version": "immutable_source_v1"})
    calendar_path = root / "calendar.json"
    calendar = {"verified": True, "dates": dates, "source": "test_verified_source"}
    write_json(calendar_path, calendar)
    report = {"data_end": dates[-1], "dataset_records": records, "calendar": {
        "run_id": "frozen-calendar", "path": str(calendar_path),
        "file_sha256": research._file_hash(calendar_path), "hash": stable_hash(calendar)}}
    report_path = root / "reports/czsc.json"
    write_json(report_path, report)
    monkeypatch.setattr(research, "ARTIFACT_RUNS_ROOT", runs)
    config = research.research_config({"version": "czsc_high_return_v1", "feature_start": start})
    return root, dates, records, report, config


def test_full_and_subset_source_snapshots_are_frozen_and_distinct(frozen_source):
    root, dates, records, _, _ = frozen_source
    before = {str(path): research._file_hash(path) for path in root.rglob("*") if path.is_file()}
    full, calendar, snapshot, end = research._source("frozen-source")
    selected, _, subset, _ = research._source("frozen-source", dates[-2], [records[1]["symbol"]])
    assert full == records and calendar == dates and end == dates[-1]
    assert snapshot["full_source_universe"] and snapshot["source_universe_count"] == 2
    assert snapshot["selected_count"] == 2 and snapshot["records_hash"] == stable_hash(records)
    assert selected == [records[1]] and not subset["full_source_universe"]
    assert subset["snapshot_hash"] != snapshot["snapshot_hash"]
    assert snapshot["calendar_run_id"] == "frozen-calendar"
    assert snapshot["snapshot_hash"] == stable_hash({k: v for k, v in snapshot.items()
        if k not in {"snapshot_hash", "calendar_run_id"}})
    assert before == {str(path): research._file_hash(path) for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("end,symbols,reason", [
    ("2030-01-01", None, "frozen source ends"),
    ("2022-01-01", None, "verified"),
    (None, ["000999.SZ"], "universe"),
])
def test_frozen_source_rejects_later_noncalendar_or_foreign_universe(frozen_source, end, symbols, reason):
    with pytest.raises(ValueError, match=reason):
        research._source("frozen-source", end, symbols)


def test_frozen_source_rejects_boundary_duplicate_and_calendar_tampering(frozen_source):
    root, _, records, report, _ = frozen_source
    with pytest.raises(ValueError, match="invalid source"):
        research._source("../../outside")
    corrupt = copy.deepcopy(report)
    corrupt["dataset_records"].append(records[0])
    write_json(root / "reports/czsc.json", corrupt)
    with pytest.raises(ValueError, match="unique"):
        research._source("frozen-source")
    write_json(root / "reports/czsc.json", report)
    calendar_path = Path(report["calendar"]["path"])
    calendar_path.write_text(calendar_path.read_text() + " ")
    with pytest.raises(ValueError, match="calendar hash"):
        research._source("frozen-source")


def test_features_keep_valid_prestart_seasoning_and_remove_fixed_ten_outcomes(frozen_source):
    _, dates, records, _, config = frozen_source
    end = dates[-5]
    frame = research._read_features(records[0], config, end)
    prefix = frame.loc[frame.date < config["feature_start"]]
    valid = (prefix.volume > 0) & (prefix.amount > 0) & prefix.has_trade_price.eq(1)
    assert valid.sum() >= config["strategy"]["execution"]["seasoned_bars"]
    assert frame.date.max() == end and frame.date.is_monotonic_increasing
    assert not set(("label", "label_end", "label_available", "label_reason", "net_return",
        "entry_date", "exit_date", "entry_shares", "fees")) & set(frame.columns)
    original = pd.read_parquet(records[0]["path"])
    assert original.net_return.eq(500.).all() and len(original) == len(dates)
    with Path(records[0]["path"]).open("ab") as stream:
        stream.write(b"changed-source")
    with pytest.raises(ValueError, match="source feature hash"):
        research._read_features(records[0], config, end)


@pytest.mark.parametrize("fault", ["foreign_symbol", "duplicate_date", "reversed_dates"])
def test_valid_file_hash_cannot_hide_wrong_symbol_or_date_identity(frozen_source, fault):
    _, dates, records, _, config = frozen_source
    record = copy.deepcopy(records[0])
    frame = pd.read_parquet(record["path"])
    if fault == "foreign_symbol": frame["symbol"] = "600002.SH"
    if fault == "duplicate_date": frame.loc[1, "date"] = frame.loc[0, "date"]
    if fault == "reversed_dates": frame = frame.iloc[::-1]
    frame.to_parquet(record["path"], index=False)
    record["sha256"] = research._file_hash(Path(record["path"]))
    with pytest.raises(ValueError, match="symbol/date identity"):
        research._read_features(record, config, dates[-1])


def test_points_build_writes_new_run_and_preserves_source_identity(frozen_source, tmp_path, monkeypatch):
    root, _, records, _, config = frozen_source
    context = Context(tmp_path / "new-run")
    saved = []
    class Store:
        def save(self, kind, report): saved.append((kind, report))
    monkeypatch.setattr(research, "ResultStore", Store)
    monkeypatch.setattr(research, "create_run_context", lambda **kwargs: context)
    hashes = {str(path): research._file_hash(path) for path in root.rglob("*") if path.is_file()}
    report = research.build_high_return_artifacts(source_training_run_id="frozen-source", config=config,
        cpu_workers=1, stage="points")
    assert report["status"] == "completed" and report["model_family"] == "high_return"
    assert report["publication_allowed"] is False and report["promotion_status"] == "research_only"
    assert report["coverage"]["requested"] == report["coverage"]["hash_verified_count"] == len(records)
    assert report["failures"] == [] and len(saved) == 1
    for record in report["dataset_records"]:
        assert record["source_hash_verified"] and record["source_sha256"] in {r["sha256"] for r in records}
        assert Path(record["points_path"]).is_relative_to(context.root)
    assert hashes == {str(path): research._file_hash(path) for path in root.rglob("*") if path.is_file()}


def test_backtest_rejects_source_report_changed_after_model_freeze(frozen_source):
    root, _, records, report, config = frozen_source
    _, _, snapshot, end = research._source("frozen-source")
    model_root = root.parent / "frozen-model"
    write_json(model_root / "reports/high_return.json", {"model_family": "high_return", "config": config,
        "source_snapshot": snapshot, "data_end": end, "dataset_records": records, "model_bundles": []})
    changed = copy.deepcopy(report)
    changed["extra_source_revision"] = "unapproved-after-model-freeze"
    write_json(root / "reports/czsc.json", changed)
    with pytest.raises(ValueError, match="frozen source report/calendar changed"):
        research.backtest_high_return(model_run_id="frozen-model", cpu_workers=1)


def test_backtest_defaults_to_model_frozen_subset(frozen_source, tmp_path, monkeypatch):
    from my_strategy.services import czsc_high_return_evaluation as evaluation
    root, _, records, _, config = frozen_source
    frozen, _, snapshot, end = research._source("frozen-source", symbols=[records[0]["symbol"]])
    write_json(root.parent / "subset-model/reports/high_return.json", {"model_family": "high_return",
        "config": config, "source_snapshot": snapshot, "data_end": end,
        "dataset_records": frozen, "model_bundles": []})
    seen = []
    def evaluate(selected, calendar, **kwargs):
        seen.extend(selected)
        full = {"metrics": {"initial_cash": 100000., "net_return": 0.}, "daily": [], "cash_retained_failures": []}
        return {"variants": {"full": full}, "windows": {"explicit_offline": {"ledger_bindings": {"full": []}}},
            "output_root": str(tmp_path / "evaluation")}
    class Store:
        def save(self, kind, report): return report
    monkeypatch.setattr(evaluation, "evaluate_bundles", evaluate)
    monkeypatch.setattr(research, "ResultStore", Store)
    monkeypatch.setattr(research, "create_run_context", lambda **kwargs: Context(tmp_path / "subset-backtest"))
    result = research.backtest_high_return(model_run_id="subset-model", cpu_workers=1)
    assert seen == frozen and result["source_snapshot"]["selected_count"] == 1
    assert result["publication_allowed"] is False


def episode_artifacts(tmp_path, *, source_hash="frozen-source-hash", row_hash=None):
    cfg = research.research_config()
    contract = target_contract(head="entry", source="union", config=cfg,
        market_dates=["2024-01-02", "2024-01-03"], source_snapshot_hash=source_hash,
        feature_schema_hash=ENTRY_SCHEMA_HASH, continuation_policy_hash="frozen-union-exit",
        initial_cash=100000., strategy_settings=cfg["strategy"])
    frame = pd.DataFrame({column: [0., 0.] for column in ENTRY_FEATURE_COLUMNS})
    frame["source"] = "union"; frame["source_snapshot_hash"] = row_hash or source_hash
    frame["continuation_policy_hash"] = contract["continuation_policy_hash"]
    frame["symbol"] = "000001.SZ"; frame["date"] = ["2024-01-02", "2024-01-03"]
    frame["label"] = [1., 0.]; frame["label_available"] = True
    frame["label_end"] = "2024-01-04"; frame["input_eligible"] = True
    path = tmp_path / "entries.parquet"; frame.to_parquet(path, index=False)
    contracts_path = tmp_path / "contracts.json"
    write_json(contracts_path, {"union": {"entry": contract}})
    artifacts = {"records": [{"symbol": "000001.SZ", "episode_status": "completed",
        "entries_path": str(path), "entries_sha256": research._file_hash(path),
        "contracts_path": str(contracts_path), "contracts_sha256": research._file_hash(contracts_path)}]}
    return artifacts, contract


def test_training_dataset_contract_is_bound_to_frozen_source_and_file_hashes(tmp_path):
    artifacts, contract = episode_artifacts(tmp_path)
    data, loaded = research._load_training(artifacts, "union", "entry", source_snapshot_hash="frozen-source-hash")
    assert loaded == contract and data.attrs["label_contract"] == contract
    assert set(data.source_snapshot_hash) == {contract["source_snapshot_hash"]}
    with pytest.raises(ValueError, match="source|snapshot"):
        research._load_training(artifacts, "union", "entry", source_snapshot_hash="other-source")
    path = Path(artifacts["records"][0]["contracts_path"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="contract.*hash|hash.*contract"):
        research._load_training(artifacts, "union", "entry", source_snapshot_hash="frozen-source-hash")


def test_training_dataset_rejects_valid_file_with_foreign_row_source(tmp_path):
    artifacts, _ = episode_artifacts(tmp_path, row_hash="foreign-row-source")
    with pytest.raises(ValueError, match="source|snapshot"):
        research._load_training(artifacts, "union", "entry", source_snapshot_hash="frozen-source-hash")


def test_head_targets_never_reuse_fixed_ten_day_labels(tmp_path):
    artifacts, contract = episode_artifacts(tmp_path)
    data = pd.read_parquet(artifacts["records"][0]["entries_path"])
    data["net_return"] = [.1, -.1]; data["exit_advantage"] = [.01, -.02]
    expected = {"return": [ .1, -.1], "loss": [0., 1.], "entry": [1., 0.]}
    for head, labels in expected.items():
        converted, bound = research._head_data(data, contract, head, ENTRY_FEATURE_COLUMNS)
        assert converted.label.tolist() == labels
        assert bound["source_snapshot_hash"] == contract["source_snapshot_hash"]
        assert bound["contract_hash"] == stable_hash({k: v for k, v in bound.items() if k != "contract_hash"})
        assert "fixed_10" not in bound["target"]
        assert converted.attrs["label_contract"] == bound


def validation_report(*, coverage=.2, failure=None):
    metrics = {"completed_round_trips": 10, "net_return": .03, "max_drawdown": .02,
        "candidate_coverage": coverage}
    return {"windows": {"validation": {"variants": {
        "full": {"metrics": metrics, "cash_retained_failures": [failure] if failure else []},
        "reference": {"metrics": {**metrics, "net_return": .01}}}}}}


def test_validation_selection_checks_actual_trade_coverage_and_replay_failures():
    cfg = research.research_config({"min_validation_round_trips": 1, "min_candidate_coverage": .1})
    assert research._selection_score(validation_report(), cfg) == .03
    assert research._selection_score(validation_report(coverage=.01), cfg) == float("-inf")
    assert research._selection_score(validation_report(failure={"symbol": "000001.SZ", "reason": "runtime_failed"}), cfg) == float("-inf")
    assert research._selection_score({"windows": {}}, cfg) == float("-inf")


def test_status_shape_has_no_automatic_production_release(monkeypatch):
    report = {"data_end": "2026-10-09", "model_bundles": [{"name": "development", "available_at": "2026-06-30",
        "heads": {"entry": {}, "exit": {}}}], "model_gate": {"pass": False},
        "independent_window": {"status": "pending_future_data"}, "promotion_status": "shadow_only_no_alias"}
    class Store:
        def list(self, kind, limit):
            assert kind == "czsc-high-return-research-train"
            return [{"run_id": "new-shadow", "kind": kind}, {"run_id": "missing", "kind": kind}]
        def get(self, run_id):
            if run_id == "missing": raise KeyError(run_id)
            return report
    monkeypatch.setattr(research, "ResultStore", Store)
    status = research.status_high_return()
    assert status["active_release"] is None and status["publication_allowed"] is False
    assert status["source_policy"] == "independent_ma_structure_wyckoff_points_union"
    assert status["production_status"] == "shadow_pending_future_certification"
    assert len(status["runs"]) == 1 and status["runs"][0]["model_bundles"][0]["heads"] == ["entry", "exit"]
    assert status["runs"][0]["model_gate"]["pass"] is False


def test_config_freezes_architecture_comparison_and_publication_is_disabled():
    cfg = research.research_config({"epochs": 2, "signal_points": {"ma_entry_threshold": .45}, "publication_allowed": True})
    assert cfg["epochs"] == 2 and cfg["signal_points"]["ma_entry_threshold"] == .45
    assert cfg["signal_points"]["ma_exit_threshold"] == .55 and cfg["strategy"]["execution"]["seasoned_bars"] >= 60
    assert not cfg["publication_allowed"] and cfg["observed_windows_diagnostic_only"]
    with pytest.raises(ValueError, match="architecture"):
        research.research_config({"architectures": ["mlp"]})
