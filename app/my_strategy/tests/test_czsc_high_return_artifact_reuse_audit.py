"""Cross-run episode paths require complete, independently rechecked provenance."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from my_strategy.services import czsc_high_return_audit as audit
from my_strategy.services import czsc_high_return_research as research
from my_strategy.services.czsc_high_return_labels import target_contract
from my_strategy.services.czsc_research_models import file_hash
from my_strategy.tests.test_czsc_high_return_complete_artifact_reuse import complete_stage, invoke
from my_strategy.tests.test_czsc_high_return_research import frozen_source, write_json


@pytest.fixture
def reused_stage(complete_stage, monkeypatch):
    original, records, calendar, snapshot, config, bundles, context, final = complete_stage
    monkeypatch.setattr(audit, "ARTIFACT_RUNS_ROOT", research.ARTIFACT_RUNS_ROOT)
    contract = target_contract(head="entry", source="union", config=config,
        market_dates=calendar, source_snapshot_hash=snapshot["snapshot_hash"],
        feature_schema_hash="fixture-schema", continuation_policy_hash="fixture-policy",
        initial_cash=100000.)
    for record in final["records"]:
        path = Path(record["contracts_path"])
        write_json(path, {"union": {"entry": contract}})
        record["contracts_sha256"] = file_hash(path)
    write_json(original / "dataset/final/manifest.json", final)
    invoke(complete_stage)
    write_json(context.root / "models/ma_bundles.json", bundles)
    report = {"config": config, "dataset_records": records, "source_snapshot": snapshot,
        "model_bundles": bundles, "episode_records": copy.deepcopy(final["records"])}
    return original, context.root, report, calendar


def run_audit(stage):
    _, root, report, calendar = stage
    state = audit._Audit()
    audit._audit_episode_artifacts(root, report, calendar, state)
    return state


def test_real_readonly_verifier_allows_only_original_complete_files(reused_stage):
    original, root, report, calendar = reused_stage
    before = {str(path): file_hash(path) for base in (original, root)
        for path in base.rglob("*") if path.is_file()}
    state = run_audit(reused_stage)
    assert not state.errors, state.errors
    assert state._passed["complete_final_artifact_reuse_provenance"]["count"] == 1
    assert state._passed["episode_target_contract_identity"]["count"] == 2
    allowed = audit._verified_reused_episode_paths(root, report, calendar)
    assert len(allowed) == 10
    assert all(path.is_relative_to(original / "dataset/final") for path in allowed)
    assert before == {str(path): file_hash(path) for base in (original, root)
        for path in base.rglob("*") if path.is_file()}


def test_old_paths_without_reuse_declaration_are_not_allowed(reused_stage):
    (reused_stage[1] / "dataset/final/artifact_reuse.json").unlink()
    state = run_audit(reused_stage)
    assert len(state.errors) == 10
    assert {row["rule"] for row in state.errors} == {"episode_artifact_sha256"}


@pytest.mark.parametrize("artifact", ["artifact_reuse.json", "manifest.json"])
def test_new_saved_manifest_or_provenance_cannot_be_resigned(reused_stage, artifact):
    path = reused_stage[1] / "dataset/final" / artifact
    value = json.loads(path.read_text())
    value["unverified_extra_claim"] = True
    write_json(path, value)
    state = run_audit(reused_stage)
    assert "complete_final_artifact_reuse_provenance" in {row["rule"] for row in state.errors}
    assert sum(row["rule"] == "episode_artifact_sha256" for row in state.errors) == 10


def test_report_record_substitution_cannot_expand_old_run_boundary(reused_stage):
    original, _, report, _ = reused_stage
    outside = original / "unrelated/points.json"
    outside.parent.mkdir(); outside.write_text("{}")
    report["episode_records"][0].update(points_path=str(outside), points_sha256=file_hash(outside))
    state = run_audit(reused_stage)
    assert any(row["rule"] == "complete_final_artifact_reuse_provenance" for row in state.errors)
    assert any("outside frozen run" in str(row["detail"]) for row in state.errors)


def test_saved_MA_bindings_must_equal_audited_report_models(reused_stage):
    root = reused_stage[1]
    bundles = json.loads((root / "models/ma_bundles.json").read_text())
    bundles[0]["available_at"] = "2030-01-01"
    write_json(root / "models/ma_bundles.json", bundles)
    state = run_audit(reused_stage)
    assert "MA bundles differ" in state.errors[0]["detail"]


@pytest.mark.parametrize("change", ["source_bytes", "episode_bytes", "behavior", "ma_provenance"])
def test_reuse_claim_does_not_replace_actual_source_target_and_behavior_checks(reused_stage, change):
    _, root, report, _ = reused_stage
    if change in {"source_bytes", "episode_bytes"}:
        path = Path(report["dataset_records"][0]["path"] if change == "source_bytes"
            else report["episode_records"][0]["entries_path"])
        path.write_bytes(path.read_bytes() + b"tampered")
    elif change == "behavior":
        report["config"]["initial_cash"] = 200000.
    else:
        report["model_bundles"][0]["available_at"] = "2030-01-01"
        write_json(root / "models/ma_bundles.json", report["model_bundles"])
    state = run_audit(reused_stage)
    assert state.errors[0]["rule"] == "complete_final_artifact_reuse_provenance"
    assert sum(row["rule"] == "episode_artifact_sha256" for row in state.errors) == 10


def test_generator_dependency_drift_still_blocks_reused_paths(reused_stage, monkeypatch):
    dependency = next(path for path in research._code_files() if path.name == "czsc_trade_episodes.py")
    original = research._file_hash
    monkeypatch.setattr(research, "_file_hash", lambda path: "changed"
        if Path(path) == dependency else original(path))
    state = run_audit(reused_stage)
    assert "generating dependency changed" in state.errors[0]["detail"]


def test_declaration_cannot_reference_its_own_new_run(reused_stage):
    root = reused_stage[1]; path = root / "dataset/final/artifact_reuse.json"
    value = json.loads(path.read_text()); value["source_run_id"] = root.name
    write_json(path, value)
    assert "different original run" in run_audit(reused_stage).errors[0]["detail"]
