"""Build and verify a portable package of the current CZSC application."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
import sqlite3
import tempfile
from typing import Any

from my_strategy.core.paths import PROJECT_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.scripts.migration_manifest import file_sha256, verify_integrity_manifest, write_integrity_manifest
from my_strategy.scripts.snapshot_sqlite_databases import _canonical_database_pairs, copy_path_consistently, runtime_layout
from my_strategy.services.czsc_artifact_integrity import model_hash


ROOT_FILES = (".env.example", ".env.native.example", ".gitignore", "README.md", "INSTALL_NATIVE.md", "RELEASE_NOTES.md",
              "RELEASE_VALIDATION.md", ".gitattributes", "pyproject.toml", "setup.py", "MANIFEST.in", "runtime-manifest.json",
              "LICENSE", "install.py", "install.bat", "install.command", "khquant-native.bat", "khquant-native.sh", "khquant.sh")
EXCLUDED_NAMES = {".git", ".scratch", ".venv", ".venv-win", "venv", "__pycache__", ".pytest_cache",
                  ".ruff_cache", ".mypy_cache", ".claude", ".codex", ".serena", "node_modules", ".env",
                  "target", "build", "dist", ".runtime-build", ".runtime-wheels", "AI_CHANGELOG.md",
                  ".DS_Store", "Thumbs.db"}


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in EXCLUDED_NAMES or name.startswith((".venv", "._"))
            or name.startswith((".pytest", "_tmp_pytest", ".env.")) and not name.endswith(".example")
            or name.endswith((".egg-info", ".pyc", ".pyo", ".log", "-wal", "-shm", "-journal"))}


def _copy_application(app: Path, target: Path) -> None:
    target.mkdir(parents=True)
    for source in sorted(app.iterdir()):
        if source.name in {"data", "artifacts", "logs"} or source.name in _ignore(str(app), [source.name]):
            continue
        destination = target / source.name
        if source.name == "my_strategy":
            def strategy_ignore(directory: str, names: list[str]) -> set[str]:
                excluded = _ignore(directory, names)
                if Path(directory) == source:
                    excluded |= {"data", "artifacts", "logs", "knowledge_base", "knowledge_base_system", "skills"}
                return excluded
            copy_path_consistently(source, destination, ignore=strategy_ignore)
        else:
            copy_path_consistently(source, destination, ignore=_ignore)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _run_path(runs: Path, identity: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", identity):
        raise ValueError(f"unsafe artifact run identity: {identity}")
    path = runs / identity
    if path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()):
        raise ValueError(f"artifact run cannot be a link: {identity}")
    if not path.resolve().is_relative_to(runs.resolve()):
        raise ValueError(f"artifact run escapes storage: {identity}")
    if not path.is_dir():
        raise ValueError(f"missing artifact run dependency: {identity}")
    return path


def _artifact_path(runs: Path, relative: str) -> Path:
    path = Path(relative)
    if (not relative or path.is_absolute() or PureWindowsPath(relative).drive
            or "\\" in relative or ".." in path.parts or len(path.parts) < 2):
        raise ValueError(f"unsafe artifact path: {relative}")
    target = _run_path(runs, path.parts[0]) / Path(*path.parts[1:])
    if not target.resolve().is_relative_to(runs.resolve()) or target.is_symlink():
        raise ValueError(f"unsafe artifact path: {relative}")
    if not target.is_file():
        raise ValueError(f"missing artifact dependency: {relative}")
    return target


RUN_REFERENCE_KEYS = frozenset({"model_run_id", "training_run_id", "calendar_run_id", "freeze_run_id",
    "source_native_run_id", "raw_source_run_id", "source_run_id", "previous_run_id", "incumbent_run_id", "candidate_run_id"})
CURRENT_REPORTS = ("czsc.json", "trend_consensus.json", "native_source.json", "raw_source.json", "calendar.json")
TRAINING_DB_RELATIVE = Path("my_strategy/data/metadata/model_training.db")


def _saved_path(runs: Path, saved: str, owner: Path | None = None, *, directory=False) -> Path:
    """Rebase immutable absolute provenance to the package's contained run tree."""
    raw = str(saved)
    parts = PureWindowsPath(raw).parts if "\\" in raw else Path(raw).parts
    if ".." in parts:
        raise ValueError(f"unsafe frozen artifact path: {raw}")
    if "runs" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("runs")
        parts = parts[index + 1:]
        if len(parts) < 2:
            raise ValueError(f"unsafe frozen artifact path: {raw}")
        target = _run_path(runs, parts[0]) / Path(*parts[1:])
    elif Path(raw).is_absolute() or PureWindowsPath(raw).drive:
        raise ValueError(f"frozen artifact is outside run storage: {raw}")
    elif owner is not None and (owner / raw).exists():
        target = owner / raw
    else:
        if len(parts) < 2:
            raise ValueError(f"unsafe artifact path: {raw}")
        target = _run_path(runs, parts[0]) / Path(*parts[1:])
    if not target.resolve().is_relative_to(runs.resolve()) or any(node.is_symlink() for node in (target, *target.parents)):
        raise ValueError(f"artifact path escapes storage: {raw}")
    if not (target.is_dir() if directory else target.is_file()):
        raise ValueError(f"missing frozen artifact: {raw}")
    return target


def _references(value: Any, runs: Path, owner: Path | None = None) -> set[str]:
    references: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RUN_REFERENCE_KEYS and item:
                references.add(_run_path(runs, str(item)).name)
            elif key in {"calendar", "parent_calendar"} and isinstance(item, dict) and item.get("run_id"):
                references.add(_run_path(runs, str(item["run_id"])).name)
            elif key == "report_path" and item:
                references.add(_saved_path(runs, str(item), owner).relative_to(runs).parts[0])
        for key, digest in (("path", "sha256"), ("raw_path", "raw_sha256"), ("native_path", "native_sha256"),
                            ("ledger_path", "ledger_sha256"), ("manifest_path", "manifest_sha256")):
            if value.get(key) and value.get(digest):
                target = _saved_path(runs, str(value[key]), owner)
                if file_sha256(target) != value[digest]:
                    raise ValueError(f"frozen artifact hash mismatch: {value[key]}")
                references.add(target.relative_to(runs).parts[0])
        references.update(set().union(*(_references(item, runs, owner) for item in value.values())) if value else set())
    elif isinstance(value, list):
        references.update(set().union(*(_references(item, runs, owner) for item in value)) if value else set())
    return references


def _czsc_runs(artifacts: Path, selection: str) -> list[Path]:
    if selection == "none": return []
    runs = artifacts / "runs"
    if not runs.is_dir(): return []
    candidates = []
    for run in runs.iterdir():
        if not run.is_dir(): continue
        supported = any((run / "reports" / name).is_file() for name in CURRENT_REPORTS)
        metadata = run / "metadata.json"
        if metadata.is_file():
            task = str(_read_json(metadata).get("task", ""))
            supported |= task.startswith(("trend-", "high-return-sample-diagnostics", "model-training", "czsc-verified-calendar"))
        if supported: candidates.append(_run_path(runs, run.name))
    if selection == "latest":
        candidates = [run for run in candidates if (run / "reports/czsc.json").is_file()] or candidates
    candidates.sort(key=lambda run: (run.stat().st_mtime_ns, run.name))
    return candidates[-1:] if selection == "latest" else candidates


def _selection_state(database: Path) -> dict:
    result = {"events": [], "comparisons": [], "selection": None}
    if not database.is_file(): return result
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "current_model_events" in tables:
            result["events"] = [{**dict(row), "payload": json.loads(row["payload"])}
                for row in connection.execute("SELECT * FROM current_model_events ORDER BY event_id")]
        if "current_model_comparisons" in tables:
            result["comparisons"] = [dict(row) for row in connection.execute("SELECT * FROM current_model_comparisons")]
        if "current_model_selection" in tables:
            row = connection.execute("SELECT * FROM current_model_selection WHERE id=1").fetchone()
            result["selection"] = dict(row) if row else None
        selected = result["selection"]
        if selected:
            event = next((row for row in result["events"] if row["event_id"] == selected["event_id"]), None)
            if not event or event["payload"].get("run_id") != selected["run_id"]:
                raise ValueError("current selection does not match its immutable event")
            if event["kind"] in {"select", "activation"} and event["payload"].get("effective_from") != selected["effective_from"]:
                raise ValueError("current selection effective time differs from event")
    return result


def _training_plans(database: Path) -> dict:
    plans = {}
    if not database.is_file(): return plans
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "training_profiles" in tables:
            for parameters, digest in conn.execute("SELECT parameters,hash FROM training_profiles"):
                if stable_hash(json.loads(parameters)) != digest: raise ValueError("training profile hash mismatch")
        if "training_default" in tables:
            for profile, revision in conn.execute("SELECT profile_id,revision FROM training_default"):
                if (profile, revision) != ("builtin", 0) and not conn.execute(
                    "SELECT 1 FROM training_profiles WHERE profile_id=? AND revision=?", (profile, revision)).fetchone():
                    raise ValueError("training default points to a missing immutable profile")
        if "training_plans" in tables:
            for identity, text in conn.execute("SELECT plan_id,payload FROM training_plans"):
                plan = json.loads(text)
                if plan.get("plan_id") != identity or plan.get("plan_hash") != stable_hash({k:v for k,v in plan.items() if k != "plan_hash"}):
                    raise ValueError("training plan hash mismatch")
                plans[identity] = plan
    return plans


def _event_comparison(runs: Path, payload: dict, state: dict, plans: dict) -> tuple[Path, dict]:
    plan = next((p for p in plans.values() if p["plan_hash"] == payload.get("plan_hash")), None)
    if not plan: raise ValueError("current selection requires its frozen training plan")
    root = _run_path(runs, payload["run_id"])
    path = root / "validation/model_comparison" / plan["plan_id"] / "comparison.json"
    if not path.is_file():
        row = next((r for r in state["comparisons"] if r["run_id"] == payload["run_id"]), None)
        if not row: raise ValueError("current selection comparison is missing")
        path = _artifact_path(runs, row["report_path"])
    value = _read_json(path)
    if stable_hash(value) != payload.get("comparison_sha256") or value.get("plan_hash") != plan["plan_hash"]:
        raise ValueError("current selection immutable comparison binding mismatch")
    return path, value


def _artifact_closure(runs: Path, selected: list[Path], state: dict, plans: dict | None = None) -> list[Path]:
    identities = {run.name for run in selected}
    for row in state["events"]:
        identities.update(_references(row["payload"], runs))
        if row["kind"] in {"select", "activation"}:
            path, value = _event_comparison(runs, row["payload"], state, plans or {})
            identities.update(_references(value, runs, _run_path(runs, row["payload"]["run_id"])))
    for row in state["comparisons"]:
        path = _artifact_path(runs, row["report_path"])
        if file_sha256(path) != row["sha256"]: raise ValueError("current comparison report hash mismatch")
        owner = _run_path(runs, path.relative_to(runs).parts[0])
        value = _read_json(path)
        plan = (plans or {}).get(value.get("plan_id"))
        if not plan or value.get("plan_hash") != plan["plan_hash"]:
            raise ValueError("current comparison requires its frozen training plan")
        identities.update(_references(value, runs, owner))
        identities.update(_references(plan, runs, owner))
        identities.add(owner.name)
    pending, visited = list(sorted(identities)), set()
    while pending:
        identity = pending.pop()
        if identity in visited: continue
        run = _run_path(runs, identity); visited.add(identity)
        files = list((run / "reports").glob("*.json")) + list((run / "validation").glob("*.json"))
        if (run / "metadata.json").is_file(): files.append(run / "metadata.json")
        for path in files:
            value = _read_json(path)
            references = _references(value, runs, run)
            if value.get("plan_id") and "plan_hash" in value:
                plan = (plans or {}).get(value["plan_id"])
                if not plan or plan["plan_hash"] != value["plan_hash"]:
                    raise ValueError("current comparison requires its frozen training plan")
                references.update(_references(plan, runs, run))
            identities.update(references); pending.extend(references - visited)
    return [_run_path(runs, identity) for identity in sorted(identities)]


def _verify_current_run(run: Path, runs: Path):
    for name in CURRENT_REPORTS:
        path = run / "reports" / name
        if not path.is_file(): continue
        value = _read_json(path)
        if name != "calendar.json" and value.get("run_id") != run.name:
            raise ValueError(f"current report identity mismatch: {run.name}/{name}")
        if name in {"trend_consensus.json", "native_source.json", "raw_source.json"} and value.get("strategy_version") != "trend_consensus":
            raise ValueError(f"retired model identity is not current: {run.name}")
        if name == "raw_source.json" and value.get("snapshot_hash") != stable_hash({k:v for k,v in value.items() if k != "snapshot_hash"}):
            raise ValueError("raw source snapshot hash mismatch")
        if name == "trend_consensus.json":
            freeze, source = value["protocol_freeze"], value["source_snapshot"]
            if freeze["freeze_hash"] != stable_hash({k:v for k,v in freeze.items() if k != "freeze_hash"}):
                raise ValueError("current protocol freeze hash mismatch")
            if source["snapshot_hash"] != stable_hash({k:v for k,v in source.items() if k != "snapshot_hash"}):
                raise ValueError("current source snapshot hash mismatch")
            if stable_hash(value["config"]) != freeze["config_hash"] or value["config"].get("source_snapshot_hash") != source["snapshot_hash"]:
                raise ValueError("current config/source freeze binding mismatch")
            if stable_hash(value["dataset_records"]) != source["records_hash"]:
                raise ValueError("current dataset catalogue hash mismatch")
            if "models_catalog_hash" in freeze:
                catalogs = {"trend":value["trend_bundles"], "ma_exit":value["ma_exit_bundles"], "architecture":value["architecture_bundles"]}
                if stable_hash(catalogs) != freeze["models_catalog_hash"]: raise ValueError("current model catalogue freeze hash mismatch")
                for bundle in [*value["trend_bundles"], *value["ma_exit_bundles"], *value["architecture_bundles"]]:
                    for asset in bundle["heads"].values():
                        model_dir = _saved_path(runs, asset["path"], run, directory=True)
                        manifest = _read_json(model_dir / "manifest.json")
                        if manifest["manifest_sha256"] != asset["manifest_sha256"]:
                            raise ValueError("current served head catalogue hash mismatch")
            native = _read_json(_run_path(runs, source["source_native_run_id"]) / "reports/native_source.json")
            if file_sha256(_run_path(runs, source["source_native_run_id"]) / "reports/native_source.json") != source["source_report_sha256"]:
                raise ValueError("current native source report hash mismatch")
            if native["source_snapshot"] != source["source_snapshot"]:
                raise ValueError("current native source lineage mismatch")
        if name == "native_source.json":
            raw = _read_json(_run_path(runs, value["source_snapshot"]["raw_source_run_id"]) / "reports/raw_source.json")
            if raw["snapshot_hash"] != value["source_snapshot"]["raw_snapshot_hash"]:
                raise ValueError("current raw-to-native source binding mismatch")
        if name in {"native_source.json", "raw_source.json"}:
            calendar_id = value.get("calendar_run_id") or value["source_snapshot"]["calendar_run_id"]
            calendar = _read_json(_run_path(runs, calendar_id) / "reports/calendar.json")
            dates = value.get("dates") or value["calendar_dates"]
            if calendar.get("verified") is not True or not calendar.get("source") or dates != sorted(set(dates)) or calendar["dates"][:len(dates)] != dates:
                raise ValueError("current source verified calendar binding mismatch")
            digest = value.get("calendar_hash") or value["source_snapshot"]["calendar_hash"]
            if digest != stable_hash(dates): raise ValueError("current source calendar hash mismatch")
    for path in (run / "models").rglob("manifest.json"):
        value = _read_json(path); body = {k:v for k,v in value.items() if k != "manifest_sha256"}
        if value.get("manifest_sha256") != stable_hash(body): raise ValueError("current head manifest hash mismatch")
        checkpoint = (path.parent / str(value.get("checkpoint", ""))).resolve()
        if not checkpoint.is_relative_to(path.parent.resolve()) or not checkpoint.is_file() or file_sha256(checkpoint) != value.get("checkpoint_sha256"):
            raise ValueError("current head checkpoint hash mismatch")
        if value.get("target_contract_hash") != stable_hash(value.get("target_contract", {})):
            raise ValueError("current head target contract hash mismatch")
        if value.get("export_sha256") and file_sha256(path.parent / "linear-layers.json") != value["export_sha256"]:
            raise ValueError("current head export hash mismatch")
    # Saved calendar source artifacts carry actual source hashes and are rebased,
    # never rewritten to make a historical hash appear valid.
    calendar = run / "reports/calendar.json"
    if calendar.is_file():
        value = _read_json(calendar)
        for artifact in value.get("source_artifacts", []):
            target = _saved_path(runs, artifact["path"], run)
            if file_sha256(target) != artifact["sha256"]: raise ValueError("verified calendar source artifact hash mismatch")
        if value.get("artifact") and value.get("sha256"):
            if file_sha256(_saved_path(runs, value["artifact"], run)) != value["sha256"]:
                raise ValueError("verified calendar date artifact hash mismatch")


def _verify_artifact_state(runs: Path, database: Path, selected: list[Path] | None = None, *, training_database=None) -> None:
    state = _selection_state(database); plans = _training_plans(Path(training_database)) if training_database else {}
    selected = selected if selected is not None else sorted(path for path in runs.iterdir() if path.is_dir()) if runs.exists() else []
    selected = _artifact_closure(runs, selected, state, plans)
    for run in selected: _verify_current_run(run, runs)
    for row in state["events"]:
        if row["kind"] in {"select", "activation"}: _event_comparison(runs, row["payload"], state, plans)
    if database.is_file():
        with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='czsc_runs'").fetchone():
                columns = {row[1] for row in conn.execute("PRAGMA table_info(czsc_runs)")}
                if "report_path" in columns:
                    for identity, relative in conn.execute("SELECT run_id,report_path FROM czsc_runs"):
                        report = _artifact_path(runs, relative)
                        if report.relative_to(runs).parts[0] != identity or _read_json(report).get("run_id") != identity:
                            raise ValueError(f"result index/report identity mismatch: {identity}")


def verify_package(package_root: Path, *, allow_runtime_additions: bool = False) -> list[str]:
    """Verify immutable payloads, allowing databases to change after installation."""
    root = package_root.resolve()
    mutable = [f"app/{relative}" for _source, relative in _canonical_database_pairs(root)] + [f"app/{TRAINING_DB_RELATIVE.as_posix()}"] if allow_runtime_additions else []
    issues = verify_integrity_manifest(root, filename="image-manifest.json", check_sqlite=True,
                                       check_unexpected=not allow_runtime_additions, mutable_paths=mutable)
    if not issues:
        try:
            _verify_artifact_state(root / "app/my_strategy/artifacts/runs",
                                   root / "app/my_strategy/data/processed/czsc/results.db",
                                   training_database=root / "app" / TRAINING_DB_RELATIVE)
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            issues.append(f"artifact state invalid: {exc}")
    return issues


def _contents(source: Path, target: Path, include_data: bool, include_artifacts: str,
              include_personal: bool = False) -> dict[str, Any]:
    layout = runtime_layout(source)
    repo, app = layout["repo_root"], layout["app_root"]
    target.mkdir()
    for filename in ROOT_FILES:
        if (repo / filename).is_file():
            copy_path_consistently(repo / filename, target / filename)
    for directory in ("services", ".github"):
        if (repo / directory).is_dir():
            copy_path_consistently(repo / directory, target / directory, ignore=_ignore)
    _copy_application(app, target / "app")
    environment = ("KHQUANT_ENV_SCHEMA=2\nKHQUANT_PROJECT_ROOT=./app\nKHQUANT_VENV_DIR=./.venv\n"
                   "KHQUANT_DATA_ROOT=./my_strategy/data\nKHQUANT_ARTIFACT_ROOT=./my_strategy/artifacts\n"
                   "KHQUANT_METADATA_ROOT=./my_strategy/data/metadata\n"
                   "KHQUANT_RAW_DB=./my_strategy/data/raw/khquant_raw.db\n"
                   "KHQUANT_LOG_ROOT=./my_strategy/data/logs\nPYTHONPATH=./app\n")
    environment += "KHQUANT_COMPUTE_DEVICE=auto\nKHQUANT_CPU_WORKERS=4\nKHQUANT_GPU_BATCH_SIZE=32\n"
    (target / ".env").write_text(environment, encoding="utf-8")
    for relative in ("my_strategy/data/raw", "my_strategy/data/processed/czsc", "my_strategy/data/metadata", "my_strategy/artifacts/runs"):
        (target / "app" / relative).mkdir(parents=True, exist_ok=True)
    selected_runs = _czsc_runs(layout["artifact_root"], include_artifacts)
    copied_results = target / "app/my_strategy/data/processed/czsc/results.db"
    if include_data or include_personal:
        for database, relative in _canonical_database_pairs(source):
            is_personal = database == layout["personal_database"]
            if is_personal and not include_personal or not is_personal and not include_data:
                continue
            copy_path_consistently(database, target / "app" / relative)
    training_source = layout["metadata_root"] / "model_training.db"
    training_copy = target / "app" / TRAINING_DB_RELATIVE
    if (include_data or include_artifacts != "none") and training_source.is_file():
        copy_path_consistently(training_source, training_copy)
    state = _selection_state(copied_results)
    if (state["events"] or state["comparisons"] or state["selection"]) and include_artifacts == "none":
        raise ValueError("data contains current model selection evidence; use --include-artifacts latest or all")
    plans = _training_plans(training_copy)
    selected_runs = _artifact_closure(layout["artifact_root"] / "runs", selected_runs, state, plans)
    if include_data:
        if copied_results.is_file():
            with closing(sqlite3.connect(copied_results)) as connection:
                if connection.execute("SELECT name FROM sqlite_master WHERE name='czsc_runs'").fetchone():
                    keep = [run.name for run in selected_runs]
                    if keep:
                        connection.execute(f"DELETE FROM czsc_runs WHERE run_id NOT IN ({','.join('?' for _ in keep)})", keep)
                    else:
                        connection.execute("DELETE FROM czsc_runs")
                    connection.commit()
                for retired in ("czsc_model_aliases", "czsc_model_release_events"):
                    connection.execute("DROP TABLE IF EXISTS " + retired)
                connection.commit()
    for run in selected_runs:
        copy_path_consistently(run, target / "app/my_strategy/artifacts/runs" / run.name, ignore=_ignore)
    (target / "RUNBOOK.md").write_text(
        "# KHQuant CZSC portable package\n\nRun `python install.py` from this directory. "
        "Use `python -m my_strategy.cli package --verify .` from this directory to verify the payload. "
        "Runtime paths resolve from `app`; the raw market database is an input. "
        "Code-only is the default. Data and model/report/calendar state are optional; "
        "saved watchlist/holdings require the separate --include-personal export option. "
        "Run `python install.py --device auto` to select available CPU, CUDA or Apple MPS support.\n", encoding="utf-8")
    return write_integrity_manifest(target, source_project_root=app, filename="image-manifest.json",
                                    reject_external_symlinks=True, metadata={"architecture": "czsc",
                                    "include_data": include_data, "include_artifacts": include_artifacts,
                                     "include_personal": include_personal,
                                     "artifact_selection": "current_model_dependency_closure",
                                     "selection_event_count": len(state["events"]),
                                    "run_ids": [run.name for run in selected_runs]})


def build_package(*, source: Path, target: Path, include_data: bool = False, include_artifacts: str = "none",
                  include_personal: bool = False, force: bool = False) -> dict[str, Any]:
    """Stage and verify before atomically replacing an explicitly selected target."""
    if include_artifacts not in {"none", "latest", "all"}:
        raise ValueError("include_artifacts must be none, latest or all")
    layout = runtime_layout(source)
    if not (layout["repo_root"] / "install.py").is_file() or not (layout["app_root"] / "AGENTS.md").is_file():
        raise RuntimeError("package export requires a KHQuant source checkout with install.py and app/AGENTS.md; "
                           "wheel installations cannot be used as source. Use --from-project /path/to/KHQuant "
                           "to select a cloned repository or extracted source Release.")
    lexical_target = target.absolute()
    for path in (lexical_target, *lexical_target.parents):
        if path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()):
            raise RuntimeError(f"package target cannot use symlink or junction: {path}")
    target = lexical_target.resolve()
    repo = layout["repo_root"]
    if repo.is_relative_to(target) or target.is_relative_to(layout["app_root"]):
        raise RuntimeError("package target cannot overlap the application or replace its repository")
    if target.exists() and not force:
        raise FileExistsError(f"target exists; pass --force to replace: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent)).resolve()
    staging, previous = stage_root / "package", stage_root / "previous"
    preserve_previous = False
    try:
        manifest = _contents(source, staging, include_data, include_artifacts, include_personal)
        issues = verify_package(staging)
        if issues:
            raise RuntimeError("staged package verification failed: " + "; ".join(issues))
        if target.exists():
            target.rename(previous)
        try:
            staging.rename(target)
        except BaseException:
            if previous.exists():
                preserve_previous = True
                previous.rename(target)
                preserve_previous = False
            raise
        return manifest
    finally:
        # Both paths belong to this newly-created, verified staging directory.
        if not preserve_previous and stage_root.parent == target.parent and stage_root.name.startswith(f".{target.name}.stage-"):
            shutil.rmtree(stage_root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--target")
    destination.add_argument("--verify")
    parser.add_argument("--from-project", default=str(PROJECT_ROOT),
                        help="Source checkout or extracted source Release; required for export from a wheel installation.")
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--include-artifacts", choices=["none", "latest", "all"], default="none",
                        help="Optional current model/source/calendar state with selection and comparison evidence dependencies.")
    parser.add_argument("--include-personal", action="store_true",
                        help="Explicitly include saved watchlist and manual holdings (private by default).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify:
        verify = Path(args.verify)
        if not verify.is_absolute():
            verify = runtime_layout(Path(args.from_project))["repo_root"] / verify
        issues = verify_package(verify)
        print(json.dumps({"ok": not issues, "issues": issues}, ensure_ascii=False))
        return int(bool(issues))
    target = Path(args.target)
    if not target.is_absolute():
        target = runtime_layout(Path(args.from_project))["repo_root"] / target
    manifest = build_package(source=Path(args.from_project), target=target, include_data=args.include_data,
                             include_artifacts=args.include_artifacts, include_personal=args.include_personal,
                             force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2) if args.json else f"Package created: {target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
