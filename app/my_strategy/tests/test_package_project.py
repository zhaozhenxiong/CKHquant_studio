"""Portable CZSC packages retain committed data and exclude retired payloads."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from my_strategy.scripts.package_project import build_package, main, verify_package
from my_strategy.scripts.snapshot_sqlite_databases import CANONICAL_DATABASE_PATHS, runtime_layout
from my_strategy.storage.personal_portfolio import PersonalStore
from my_strategy.core.run_context import stable_hash
from my_strategy.scripts.migration_manifest import file_sha256, write_integrity_manifest
from my_strategy.services.czsc_artifact_integrity import model_hash


def source_repo(root: Path) -> Path:
    repo = root / "source"
    app = repo / "app"
    (app / "my_strategy/core").mkdir(parents=True)
    (app / "my_strategy/core/paths.py").write_text("# marker", encoding="utf-8")
    (app / "AGENTS.md").write_text("# source project instructions", encoding="utf-8")
    (repo / "install.py").write_text("# installer", encoding="utf-8")
    (app / "vendor/czsc").mkdir(parents=True)
    (app / "vendor/czsc/LICENSE").write_text("source license", encoding="utf-8")
    return repo


def test_package_only_current_databases_and_czsc_runs(tmp_path):
    repo = source_repo(tmp_path)
    app = repo / "app"
    for relative in CANONICAL_DATABASE_PATHS:
        database = app / relative
        database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute("CREATE TABLE evidence (value TEXT)")
            connection.execute("INSERT INTO evidence VALUES ('current')")
            connection.commit()
    for relative in ("my_strategy/data/processed/warehouse/old.db", "my_strategy/artifacts/vector_datasets/old.csv",
                     ".scratch/secret.txt", "my_strategy/__pycache__/old.pyc", ".venv/lib/old.txt"):
        path = app / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retired", encoding="utf-8")
    run = app / "my_strategy/artifacts/runs/current/reports"
    run.mkdir(parents=True)
    (run / "czsc.json").write_text(json.dumps({"run_id": "current"}), encoding="utf-8")
    old_run = app / "my_strategy/artifacts/runs/old/reports"
    old_run.mkdir(parents=True)
    (old_run / "report.json").write_text("{}", encoding="utf-8")
    with closing(sqlite3.connect(app / CANONICAL_DATABASE_PATHS[1])) as connection:
        connection.execute("CREATE TABLE czsc_runs(run_id TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO czsc_runs VALUES (?)", [("current",), ("unselected",)])
        connection.commit()
    target = tmp_path / "package"
    manifest = build_package(source=app, target=target, include_data=True, include_artifacts="all")
    assert manifest["run_ids"] == ["current"]
    assert verify_package(target) == []
    assert (target / "app/vendor/czsc/LICENSE").is_file()
    for relative in CANONICAL_DATABASE_PATHS[:-1]:
        assert (target / "app" / relative).is_file()
    assert not (target / "app" / CANONICAL_DATABASE_PATHS[-1]).exists()
    assert (target / "app/my_strategy/artifacts/runs/current/reports/czsc.json").is_file()
    assert not (target / "app/my_strategy/artifacts/runs/old").exists()
    assert not (target / "app/my_strategy/data/processed/warehouse").exists()
    assert not (target / "app/.scratch").exists()
    assert not (target / "app/.venv").exists()
    assert not (target / "app/my_strategy/__pycache__").exists()
    assert runtime_layout(target)["app_root"] == target / "app"
    assert runtime_layout(target / "app") == runtime_layout(target)
    with closing(sqlite3.connect(target / "app" / CANONICAL_DATABASE_PATHS[1])) as connection:
        assert connection.execute("SELECT run_id FROM czsc_runs").fetchall() == [("current",)]


def test_package_online_backup_captures_active_wal(tmp_path):
    repo = source_repo(tmp_path)
    raw = repo / "app" / CANONICAL_DATABASE_PATHS[0]
    raw.parent.mkdir(parents=True)
    with closing(sqlite3.connect(raw)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE evidence (value TEXT)")
        writer.execute("INSERT INTO evidence VALUES ('committed-in-wal')")
        writer.commit()
        assert raw.with_name(raw.name + "-wal").exists()
        target = tmp_path / "package"
        build_package(source=repo, target=target, include_data=True, include_artifacts="none")
    copied = target / "app" / CANONICAL_DATABASE_PATHS[0]
    with closing(sqlite3.connect(copied)) as connection:
        assert connection.execute("SELECT value FROM evidence").fetchone()[0] == "committed-in-wal"
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not copied.with_name(copied.name + "-wal").exists()


def test_external_paths_become_portable_and_integrity_is_checked(tmp_path):
    repo = source_repo(tmp_path)
    external = tmp_path / "external"
    raw = external / "raw/khquant_raw.db"
    raw.parent.mkdir(parents=True)
    with closing(sqlite3.connect(raw)) as connection:
        connection.execute("CREATE TABLE stock_daily_normalized(stock TEXT, date TEXT)")
        connection.commit()
    (repo / ".env").write_text(f'KHQUANT_PROJECT_ROOT=./app\nKHQUANT_DATA_ROOT="{external.as_posix()}"\n', encoding="utf-8")
    target = tmp_path / "portable"
    build_package(source=repo, target=target, include_data=True, include_artifacts="none")
    assert str(external) not in (target / ".env").read_text(encoding="utf-8")
    assert verify_package(target) == []
    (target / "app/my_strategy/core/paths.py").write_text("tampered", encoding="utf-8")
    assert any("mismatch" in issue for issue in verify_package(target))


def test_package_preserves_saved_personal_records_from_external_metadata(tmp_path):
    repo = source_repo(tmp_path)
    metadata = tmp_path / "personal metadata"
    (repo / ".env").write_text(
        f'KHQUANT_PROJECT_ROOT=./app\nKHQUANT_METADATA_ROOT="{metadata.as_posix()}"\n', encoding="utf-8")
    unavailable_raw = tmp_path / "no-market.db"
    personal = PersonalStore(metadata / "personal_portfolio.db", unavailable_raw)
    personal.add_watchlist(["000001.SZ", "600036.SH"])
    target = tmp_path / "personal-package"
    build_package(source=repo, target=target, include_personal=True)
    assert verify_package(target) == []
    saved = target / "app/my_strategy/data/metadata/personal_portfolio.db"
    reopened = PersonalStore(saved, unavailable_raw)
    assert [row["symbol"] for row in reopened.watchlist()["items"]] == ["000001.SZ", "600036.SH"]
    assert reopened.watchlist()["coverage"]=={"total":2,"priced":0,"missing":2}
    assert not (target / "app" / CANONICAL_DATABASE_PATHS[0]).exists()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def current_state(repo: Path) -> tuple[Path, Path]:
    """Content-closure fixture only; it grants no model serving qualification."""
    runs = repo / "app/my_strategy/artifacts/runs"
    dates = ["2026-01-05", "2026-01-06"]
    write_json(runs / "calendar/reports/calendar.json", {"verified": True, "source": "verified fixture", "dates": dates})
    raw_file = runs / "raw/data_snapshots/raw/000001_SZ/raw.parquet"
    raw_file.parent.mkdir(parents=True); raw_file.write_bytes(b"frozen raw input bytes")
    raw_record = {"symbol": "000001.SZ", "raw_path": str(raw_file), "raw_sha256": file_sha256(raw_file)}
    raw = {"run_id": "raw", "strategy_version": "trend_consensus", "calendar_run_id": "calendar",
           "dates": dates, "calendar_hash": stable_hash(dates), "raw_records": [raw_record]}
    raw["snapshot_hash"] = stable_hash(raw)
    write_json(runs / "raw/reports/raw_source.json", raw)
    native_dir = runs / "native/data_snapshots/native/000001_SZ"; native_dir.mkdir(parents=True)
    (native_dir / "features.parquet").write_bytes(b"frozen feature bytes")
    write_json(native_dir / "native.json", {"observed_only": True})
    record = {**raw_record, "path": str(native_dir / "features.parquet"), "sha256": file_sha256(native_dir / "features.parquet"),
              "native_path": str(native_dir / "native.json"), "native_sha256": file_sha256(native_dir / "native.json")}
    lineage = {"raw_source_run_id": "raw", "raw_snapshot_hash": raw["snapshot_hash"],
               "calendar_run_id": "calendar", "calendar_hash": stable_hash(dates)}
    write_json(runs / "native/reports/native_source.json", {"run_id": "native", "strategy_version": "trend_consensus",
        "calendar_dates": dates, "source_snapshot": lineage, "dataset_records": [record]})
    model = runs / "training/models/production/trend_direction"; model.mkdir(parents=True)
    (model / "model.joblib").write_bytes(b"immutable CPU checkpoint fixture")
    target = {"test_fixture": "content integrity, not training qualification"}
    manifest = {"family": "trend_consensus", "checkpoint": "model.joblib", "checkpoint_sha256": file_sha256(model / "model.joblib"),
                "target_contract": target, "target_contract_hash": stable_hash(target)}
    manifest["manifest_sha256"] = stable_hash(manifest); write_json(model / "manifest.json", manifest)
    source = {"source_native_run_id": "native", "source_report_sha256": file_sha256(runs / "native/reports/native_source.json"),
              "source_snapshot": lineage, "records_hash": stable_hash([record])}
    source["snapshot_hash"] = stable_hash(source)
    cfg = {"version": "trend_consensus", "source_snapshot_hash": source["snapshot_hash"]}
    freeze = {"config_hash": stable_hash(cfg)}; freeze["freeze_hash"] = stable_hash(freeze)
    write_json(runs / "training/reports/trend_consensus.json", {"run_id": "training", "strategy_version": "trend_consensus",
        "automatic_selection_eligible": False, "config": cfg, "protocol_freeze": freeze, "source_snapshot": source, "dataset_records": [record]})
    return runs, model


def add_current_selection_database(repo: Path, runs: Path) -> Path:
    plan = {"plan_id": "frozen-plan", "calendar_run_id": "calendar", "incumbent_run_id": None, "cohort": ["000001.SZ"]}
    plan["plan_hash"] = stable_hash(plan)
    training_db = repo / "app/my_strategy/data/metadata/model_training.db"; training_db.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(training_db)) as conn:
        conn.executescript("CREATE TABLE training_profiles(profile_id TEXT,revision INTEGER,parameters TEXT,hash TEXT);"
                           "CREATE TABLE training_default(profile_id TEXT,revision INTEGER);"
                           "CREATE TABLE training_plans(plan_id TEXT,payload TEXT);")
        parameters = {"cohort_size": 360}
        conn.execute("INSERT INTO training_profiles VALUES ('personal',1,?,?)", (json.dumps(parameters), stable_hash(parameters)))
        conn.execute("INSERT INTO training_default VALUES ('personal',1)")
        conn.execute("INSERT INTO training_plans VALUES (?,?)", (plan["plan_id"], json.dumps(plan))); conn.commit()
    evidence = runs / "training/validation/model_comparison/window/candidate.json"
    write_json(evidence, {"ledger": [], "qualification": "package content fixture only"})
    comparison = {"run_id": "training", "plan_id": plan["plan_id"], "plan_hash": plan["plan_hash"],
        "windows": [{"arms": {"candidate": {"assets": [{"path": evidence.relative_to(runs / 'training').as_posix(), "sha256": file_sha256(evidence)}]}}}]}
    path = runs / "training/validation/model_comparison" / plan["plan_id"] / "comparison.json"; write_json(path, comparison)
    event = {"run_id": "training", "previous_run_id": None, "comparison_sha256": stable_hash(comparison),
             "plan_hash": plan["plan_hash"], "effective_from": "2026-01-07"}
    database = repo / "app" / CANONICAL_DATABASE_PATHS[1]; database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as conn:
        conn.executescript("CREATE TABLE current_model_events(event_id INTEGER,kind TEXT,payload TEXT,created_at TEXT);"
                           "CREATE TABLE current_model_selection(id INTEGER,run_id TEXT,event_id INTEGER,effective_from TEXT);"
                           "CREATE TABLE current_model_comparisons(run_id TEXT,report_path TEXT,sha256 TEXT);"
                           "CREATE TABLE czsc_runs(run_id TEXT PRIMARY KEY,report_path TEXT);"
                           "CREATE TRIGGER immutable_events BEFORE DELETE ON current_model_events BEGIN SELECT RAISE(ABORT,'immutable'); END;")
        conn.execute("INSERT INTO current_model_events VALUES (1,'select',?,'2026-01-06')", (json.dumps(event),))
        conn.execute("INSERT INTO current_model_selection VALUES (1,'training',1,'2026-01-07')")
        conn.execute("INSERT INTO current_model_comparisons VALUES ('training',?,?)", (path.relative_to(runs).as_posix(), file_sha256(path)))
        conn.execute("INSERT INTO czsc_runs VALUES ('scan','scan/reports/czsc.json')"); conn.commit()
    return database


def test_code_only_default_carries_installation_and_current_source_without_private_state(tmp_path):
    repo = source_repo(tmp_path)
    runs, _model = current_state(repo)
    write_json(repo / "runtime-manifest.json", {"schema": "fixture"})
    for name in ("setup.py", "pyproject.toml", "MANIFEST.in"):
        (repo / name).write_text("fixture", encoding="utf-8")
    for relative in (".github/workflows/portable.yml", "app/my_strategy/services/new_business.py",
                     "app/my_strategy/configs/new_business.json", "app/my_strategy/knowledge_base_system/prompts/private.md",
                     "app/my_strategy/knowledge_base/private.md", "app/AI_CHANGELOG.md", "app/.env.production",
                     "app/.pytest-old/private.txt", "app/build/lib/stale_business.py",
                     "app/dist/old.whl", "app/my_strategy/khquant.egg-info/PKG-INFO"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    target = tmp_path / "code"
    manifest = build_package(source=repo, target=target)
    assert manifest["run_ids"] == [] and manifest["include_personal"] is False
    assert list((target / "app/my_strategy/artifacts/runs").iterdir()) == []
    assert (target / "runtime-manifest.json").exists()
    assert (target / "setup.py").exists()
    assert (target / ".github/workflows/portable.yml").exists()
    assert (target / "app/my_strategy/services/new_business.py").exists()
    assert (target / "app/my_strategy/configs/new_business.json").exists()
    for relative in ("app/my_strategy/knowledge_base_system", "app/my_strategy/knowledge_base",
                     "app/AI_CHANGELOG.md", "app/.env.production", "app/.pytest-old",
                     "app/build", "app/dist", "app/my_strategy/khquant.egg-info"):
        assert not (target / relative).exists()
    assert (runs / "training/models/production/trend_direction/model.joblib").exists()


@pytest.mark.parametrize("missing", ["install.py", "app/AGENTS.md"])
def test_export_rejects_non_source_layout_without_creating_target(tmp_path, missing):
    repo = source_repo(tmp_path)
    (repo / missing).unlink()
    target = tmp_path / "exports/package"
    with pytest.raises(RuntimeError, match="Use --from-project"):
        build_package(source=repo, target=target)
    assert not target.parent.exists()


def test_wheel_layout_cannot_export_site_packages_but_explicit_source_can(tmp_path):
    site_packages = tmp_path / "environment/lib/site-packages"
    marker = site_packages / "my_strategy/core/paths.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("# installed wheel module", encoding="utf-8")
    third_party = site_packages / "third_party/private.txt"
    third_party.parent.mkdir()
    third_party.write_text("unrelated installed package", encoding="utf-8")
    target = tmp_path / "exports/package"
    with pytest.raises(RuntimeError, match="wheel installations cannot be used as source"):
        main(["--from-project", str(site_packages), "--target", str(target)])
    assert not target.exists()
    repo = source_repo(tmp_path)
    assert main(["--from-project", str(repo), "--target", str(target)]) == 0
    assert (target / "install.py").is_file()
    assert not (target / "app/third_party").exists()


def test_current_model_and_sources_survive_relocation_without_rewriting_frozen_hashes(tmp_path):
    repo = source_repo(tmp_path)
    runs, model = current_state(repo)
    write_json(runs / "study/metadata.json", {"task": "model-training-study"})
    write_json(runs / "study/reports/summary.json", {"training_run_id": "training"})
    original = (model / "manifest.json").read_bytes()
    target = tmp_path / "other path/portable"
    manifest = build_package(source=repo, target=target, include_artifacts="all")
    assert manifest["run_ids"] == ["calendar", "native", "raw", "study", "training"]
    copied_runs = target / "app/my_strategy/artifacts/runs"
    assert (copied_runs / "training/models/production/trend_direction/manifest.json").read_bytes() == original
    assert (copied_runs / "native/data_snapshots/native/000001_SZ/features.parquet").read_bytes() == b"frozen feature bytes"
    assert verify_package(target) == []


def test_latest_closes_current_sources_selection_plan_and_keeps_immutable_events(tmp_path):
    repo = source_repo(tmp_path)
    runs, _model = current_state(repo)
    write_json(runs / "scan/reports/czsc.json", {"run_id": "scan", "model_run_id": "training"})
    database = add_current_selection_database(repo, runs)
    with closing(sqlite3.connect(database)) as connection:
        original_event = connection.execute("SELECT kind,payload FROM current_model_events").fetchone()
    target = tmp_path / "latest"
    manifest = build_package(source=repo, target=target, include_data=True, include_artifacts="latest")
    assert manifest["run_ids"] == ["calendar", "native", "raw", "scan", "training"]
    assert manifest["selection_event_count"] == 1
    with closing(sqlite3.connect(target / "app" / CANONICAL_DATABASE_PATHS[1])) as connection:
        assert connection.execute("SELECT run_id FROM czsc_runs").fetchall() == [("scan",)]
        assert connection.execute("SELECT kind,payload FROM current_model_events").fetchone() == original_event
    assert verify_package(target) == []
    with pytest.raises(ValueError, match="current model selection evidence"):
        build_package(source=repo, target=tmp_path / "broken-data-only", include_data=True)


@pytest.mark.parametrize("damage", ["calendar", "checkpoint", "manifest", "unsafe-reference"])
def test_package_rejects_broken_current_state_without_replacing_previous(tmp_path, damage):
    repo = source_repo(tmp_path)
    runs, model = current_state(repo)
    if damage == "calendar":
        (runs / "calendar/reports/calendar.json").unlink()
    elif damage == "checkpoint":
        (model / "model.joblib").write_bytes(b"damaged")
    elif damage == "manifest":
        value = json.loads((model / "manifest.json").read_text(encoding="utf-8"))
        value["available_at"] = "2030-01-01"
        write_json(model / "manifest.json", value)
    else:
        write_json(runs / "scan/reports/czsc.json", {"run_id": "scan", "model_run_id": "../outside"})
    target = tmp_path / "previous"
    target.mkdir()
    (target / "keep.txt").write_text("previous", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        build_package(source=repo, target=target, include_artifacts="all", force=True)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "previous"


def test_verify_reports_current_ledger_tamper_even_with_rebuilt_outer_manifest(tmp_path):
    repo = source_repo(tmp_path)
    runs, _model = current_state(repo)
    write_json(runs / "scan/reports/czsc.json", {"run_id": "scan", "model_run_id": "training"})
    add_current_selection_database(repo, runs)
    target = tmp_path / "released"
    build_package(source=repo, target=target, include_data=True, include_artifacts="all")
    (target / "app/my_strategy/artifacts/runs/training/validation/model_comparison/window/candidate.json").write_text("{}", encoding="utf-8")
    write_integrity_manifest(target, source_project_root=repo / "app", filename="image-manifest.json", reject_external_symlinks=True)
    assert any("frozen artifact hash mismatch" in issue for issue in verify_package(target))


def test_force_does_not_remove_previous_package_on_build_failure(tmp_path, monkeypatch):
    from my_strategy.scripts import package_project
    repo = source_repo(tmp_path)
    target = tmp_path / "package"
    target.mkdir()
    (target / "keep.txt").write_text("previous", encoding="utf-8")
    def fail(*args):
        raise RuntimeError("failed stage")
    monkeypatch.setattr(package_project, "_contents", fail)
    with pytest.raises(RuntimeError, match="failed stage"):
        build_package(source=repo, target=target, include_data=False, include_artifacts="none", force=True)
    assert (target / "keep.txt").read_text(encoding="utf-8") == "previous"


def test_package_refuses_overlapping_target(tmp_path):
    repo = source_repo(tmp_path)
    with pytest.raises(RuntimeError, match="overlap"):
        build_package(source=repo, target=repo / "app/nested", include_data=False, include_artifacts="none")


def test_relative_verify_uses_source_repository_after_cli_changes_directory(tmp_path, monkeypatch):
    repo = source_repo(tmp_path)
    target = repo / "exports/portable"
    build_package(source=repo, target=target, include_data=False, include_artifacts="none")
    monkeypatch.chdir(tmp_path)
    assert main(["--from-project", str(repo), "--verify", "exports/portable"]) == 0


@pytest.mark.parametrize("target", ["raw", "native", "plan", "default"])
def test_rebuilt_outer_manifest_cannot_hide_current_input_or_parameter_drift(tmp_path, target):
    repo = source_repo(tmp_path)
    runs, _ = current_state(repo)
    add_current_selection_database(repo, runs)
    packaged = tmp_path / "package"
    build_package(source=repo, target=packaged, include_data=True, include_artifacts="all")
    copied = packaged / "app/my_strategy/artifacts/runs"
    if target == "raw":
        (copied / "raw/data_snapshots/raw/000001_SZ/raw.parquet").write_bytes(b"new prices")
    elif target == "native":
        (copied / "native/data_snapshots/native/000001_SZ/features.parquet").write_bytes(b"changed causal inputs")
    else:
        database = packaged / "app/my_strategy/data/metadata/model_training.db"
        with closing(sqlite3.connect(database)) as conn:
            if target == "default": conn.execute("UPDATE training_default SET revision=99")
            else:
                plan = json.loads(conn.execute("SELECT payload FROM training_plans").fetchone()[0])
                plan["cohort"] = ["600519.SH"]
                conn.execute("UPDATE training_plans SET payload=?", (json.dumps(plan),))
            conn.commit()
    write_integrity_manifest(packaged, source_project_root=repo / "app", filename="image-manifest.json", reject_external_symlinks=True)
    assert verify_package(packaged), "outer package hash cannot authorize changed frozen source or parameters"


def test_pending_activation_copies_same_frozen_comparison_and_rejects_proof_drift(tmp_path):
    repo = source_repo(tmp_path)
    runs, _ = current_state(repo)
    database = add_current_selection_database(repo, runs)
    with closing(sqlite3.connect(database)) as conn:
        payload = json.loads(conn.execute("SELECT payload FROM current_model_events WHERE event_id=1").fetchone()[0])
        payload.update(effective_from="2026-01-08", pending_event_id=1)
        conn.execute("INSERT INTO current_model_events VALUES (2,'activation',?,'2026-01-07')", (json.dumps(payload),))
        conn.execute("UPDATE current_model_selection SET event_id=2,effective_from='2026-01-08'"); conn.commit()
    target = tmp_path / "activated"
    manifest = build_package(source=repo, target=target, include_data=True, include_artifacts="all")
    assert manifest["selection_event_count"] == 2 and verify_package(target) == []
    comparison = target / "app/my_strategy/artifacts/runs/training/validation/model_comparison/frozen-plan/comparison.json"
    value = json.loads(comparison.read_text()); value["plan_hash"] = "changed"
    write_json(comparison, value)
    write_integrity_manifest(target, source_project_root=repo / "app", filename="image-manifest.json", reject_external_symlinks=True)
    assert any("comparison" in error or "plan" in error for error in verify_package(target))
