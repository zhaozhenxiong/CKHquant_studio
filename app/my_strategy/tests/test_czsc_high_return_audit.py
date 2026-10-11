"""Read-only new-family audit and the legacy release boundary."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_high_return_audit as audit
from my_strategy.services import czsc_high_return_evaluation as evaluation
from my_strategy.services.czsc_high_return_labels import target_contract
from my_strategy.services.czsc_high_return_features import L1_MA_ENTRY_COLUMNS, MA_ENTRY_SCHEMA_HASH
from my_strategy.services.czsc_high_return_models import FEATURE_VERSION, VERSION, schema_hash
from my_strategy.services.czsc_research_models import file_hash
from my_strategy.storage.czsc_model_releases import ModelReleaseStore
from my_strategy.tests.test_czsc_high_return_evaluation import inputs, actual_broker_fake


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def run(tmp_path, monkeypatch):
    runs = tmp_path / "runs"; root = runs / "high-run"; source = runs / "source"
    source.mkdir(parents=True)
    records, dates, cfg = inputs(source)
    monkeypatch.setattr(audit, "ARTIFACT_RUNS_ROOT", runs)
    monkeypatch.setattr(evaluation, "_replay", actual_broker_fake([]))
    calendar_path = source / "calendar.json"; source_path = source / "report.json"
    save(calendar_path, {"verified": True, "dates": dates})
    save(source_path, {"dataset_records": records})
    snapshot = {"report_path": str(source_path), "report_sha256": file_hash(source_path),
        "calendar_path": str(calendar_path), "calendar_sha256": file_hash(calendar_path),
        "calendar_hash": stable_hash(dates), "full_source_universe": True}
    snapshot["snapshot_hash"] = stable_hash(snapshot)
    cfg["source_snapshot_hash"] = snapshot["snapshot_hash"]
    saved_evaluation = evaluation.evaluate_bundles(records, dates, model_root=root, bundles=[], config=cfg,
        windows=cfg["development_windows"], cpu_workers=1, output_root=root / "evaluation", arms=["reference", "full"])
    report = {"run_id": root.name, "model_family": "high_return", "config": cfg,
        "publication_allowed": False, "source_snapshot": snapshot, "dataset_records": records,
        "model_bundles": [], "evaluation": saved_evaluation, "cost_stress": [], "independent_window": {"start": None}}
    save(root / "metadata.json", {"stocks": [r["symbol"] for r in records]})
    save(root / "reports/high_return.json", report)
    return root, report, dates


def test_intact_research_is_pending_and_formal_blocked_without_writing(run):
    root, _, _ = run
    before = {p: file_hash(p) for p in root.rglob("*") if p.is_file()}
    report = audit.audit_high_return_run(root.name, as_of="2026-10-10")
    assert report["artifact_integrity_passed"], report["errors"]
    assert not report["passed"] and not report["publication_allowed"] and not report["alias_updated"]
    pending = {item["rule"] for item in report["pending_checks"]}
    assert {"complete_model_heads", "actual_complete_protocol_freeze", "three_complete_independent_forward_windows",
        "complete_cost_pressure_replays"} <= pending
    assert {row["rule"] for row in report["formal_blocks"]} == {
        "formal_historical_universe_evidence", "formal_price_basis_total_return_evidence", "formal_historical_st_evidence"}
    assert before == {p: file_hash(p) for p in root.rglob("*") if p.is_file()}


def test_mutated_actual_fees_and_invented_cash_are_recomputed(run):
    root, training, _ = run
    binding = training["evaluation"]["windows"]["quarter"]["ledger_bindings"]["reference"][0]
    location = Path(binding["path"])
    ledger = pd.read_csv(location / "ledger.csv")
    ledger.loc[0, "fee"] += 10; ledger.to_csv(location / "ledger.csv", index=False)
    binding["ledger_sha256"] = file_hash(location / "ledger.csv")  # even a re-signed file must reconcile
    daily = pd.read_csv(location / "daily.csv")
    daily.loc[0, "cash"] += 100; daily.to_csv(location / "daily.csv", index=False)
    binding["daily_sha256"] = file_hash(location / "daily.csv")
    save(Path(training["evaluation"]["output_root"]) / "evaluation.json", training["evaluation"])
    result = audit.audit_high_return_run(root.name, as_of="2026-10-10")
    rules = {r["rule"] for r in result["errors"]}
    assert {"ledger_fee_and_cash_flow", "daily_actual_ledger_cash_shares_equity", "portfolio_ledger_metric_recomputation"} <= rules
    assert not result["passed"]


def add_head(root, report, dates):
    directory = root / "models/l1/ma_entry"; directory.mkdir(parents=True)
    checkpoint = directory / "linear-layers.json"; save(checkpoint, {"layers": []})
    contract = target_contract(head="ma_entry", source="ma", config=report["config"], market_dates=dates,
        source_snapshot_hash=report["source_snapshot"]["snapshot_hash"], feature_schema_hash=MA_ENTRY_SCHEMA_HASH,
        continuation_policy_hash="rule-v0", initial_cash=100000.)
    contract.pop("contract_hash")
    contract.update(model_head="ma_entry", model_input_schema_hash=schema_hash(L1_MA_ENTRY_COLUMNS, "ma_entry"))
    contract["contract_hash"] = stable_hash(contract)
    fold = {"train_end": "2023-12-31", "calibration_start": dates[0], "calibration_end": dates[20],
        "policy_start": dates[21], "policy_end": dates[40], "validation_end": dates[40]}
    manifest = {"family": VERSION, "head": "ma_entry", "architecture": "linear", "regression": False,
        "feature_version": FEATURE_VERSION, "feature_schema_hash": schema_hash(L1_MA_ENTRY_COLUMNS, "ma_entry"),
        "columns": list(L1_MA_ENTRY_COLUMNS), "target_contract": contract, "target_contract_hash": stable_hash(contract),
        "fold": fold, "preprocess": {"median": [0.] * 20}, "available_at": dates[40],
        "training_started_at": "2026-10-09T20:00:00+08:00", "training_completed_at": "2026-10-09T20:10:00+08:00",
        "train_label_end": "2023-12-29", "calibration_label_end": dates[20],
        "checkpoint": checkpoint.name, "checkpoint_sha256": file_hash(checkpoint),
        "export_sha256": file_hash(checkpoint), "publication_allowed": False}
    manifest["manifest_sha256"] = stable_hash(manifest)
    save(directory / "manifest.json", manifest)
    report["model_bundles"] = [{"name": "ma", "heads": {"ma_entry": {"path": "models/l1/ma_entry",
        "manifest_sha256": manifest["manifest_sha256"], "target_contract_hash": manifest["target_contract_hash"]}}}]
    return directory, manifest


def test_head_hash_contract_and_same_actual_freeze_are_independently_bound(run):
    root, report, dates = run
    directory, manifest = add_head(root, report, dates)
    freeze = {"actual_completed_at": "2026-10-09T20:05:00+08:00", "code_hashes": {},
        "config_hash": stable_hash(report["config"]), "source_snapshot_hash": report["source_snapshot"]["snapshot_hash"],
        "model_bundles_hash": stable_hash(report["model_bundles"]), "independent_start": "2026-10-09",
        "minimum_technical_sessions": 480, "minimum_windows": 3}
    freeze["freeze_hash"] = stable_hash(freeze); report["protocol_freeze"] = freeze
    save(root / "validation/protocol_freeze.json", freeze); save(root / "reports/high_return.json", report)
    result = audit.audit_high_return_run(root.name, as_of="2026-10-10")
    assert "freeze_after_all_training" in {r["rule"] for r in result["errors"]}
    # A contract changed and re-signed at the head level cannot bypass the
    # frozen source/strategy and bundle binding.
    manifest["target_contract"]["execution_config_hash"] = "different"
    contract = manifest["target_contract"]; contract.pop("contract_hash"); contract["contract_hash"] = stable_hash(contract)
    manifest["target_contract_hash"] = stable_hash(contract)
    manifest.pop("manifest_sha256"); manifest["manifest_sha256"] = stable_hash(manifest)
    save(directory / "manifest.json", manifest)
    changed = audit.audit_high_return_run(root.name, as_of="2026-10-10")
    assert {"head_source_calendar_execution_binding", "bundle_head_manifest_binding"} <= {r["rule"] for r in changed["errors"]}


def test_high_return_never_enters_old_fixed_ten_release_transaction(run, tmp_path):
    root, report, _ = run
    store = ModelReleaseStore(tmp_path / "results.db", root.parent)
    with pytest.raises(ValueError, match="own independent forward certification"):
        store.promote(root.name, reason="attempt")
    assert store.events() == [] and store.active() is None


def test_historical_report_cannot_claim_a_gate_pass(run):
    root, training, _ = run
    full = copy.deepcopy(training["evaluation"])
    full["windows"]["quarter"]["publication_allowed"] = True
    full["windows"]["quarter"]["model_gate"] = True
    save(Path(full["output_root"]) / "evaluation.json", full)
    result = audit.audit_high_return_run(root.name, as_of="2026-10-10")
    assert "historical_window_diagnostic_only" in {r["rule"] for r in result["errors"]}


def test_quarter_cannot_replace_480_sessions_or_shorten_dependence_blocks(run):
    root, report, dates = run
    state = audit._Audit()
    result = audit._audit_evaluation(root, report["evaluation"], dates, report["config"], {"000001.SZ"}, state,
        independent=True, as_of=audit.signal_time("2026-10-10"), records={r["symbol"]:r for r in report["dataset_records"]})
    assert result["windows"][0]["sessions"] == 21
    assert "independent_480_sessions" in {r["rule"] for r in state.errors}
    ci_failures = [r for r in state.errors if r["rule"] == "independent_complete_positive_date_blocks"]
    assert {r["detail"]["block"] for r in ci_failures} == {160, 240}
