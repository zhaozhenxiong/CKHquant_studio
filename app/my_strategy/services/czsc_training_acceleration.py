"""Versioned orchestration checkpoints around the unchanged research algorithms.

Only runs created through this adapter support arbitrary-stage resume. The old
E1 recovery entrypoint remains available for its original frozen runs.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid

import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_trend_support import _file_hash, _json, _save, resolve_frozen_asset
from my_strategy.services.czsc_trend_resume import _run_lease, _ResumeContext

VERSION = "training_acceleration_v1"
ADAPTER_FILES = ("services/czsc_training_acceleration.py", "services/czsc_training_fast.py",
                 "services/czsc_training_cache.py", "scripts/train_trend.py",
                 "scripts/resume_training_fast.py", "services/czsc_trend_resume.py")
_LOCK = threading.Lock()


def code_identity():
    from my_strategy.services.czsc_trend_pipeline import GENERATING_FILES
    return {name: _file_hash(MY_STRATEGY_ROOT / name) for name in (*GENERATING_FILES, *ADAPTER_FILES)}


def frame_identity(frame):
    """Hash exact values, dtype/order, index and causal contract, not just shape."""
    from my_strategy.services.czsc_training_fast import _digest
    return _digest(frame)


def _files(root, directory):
    directory = Path(directory).resolve(); root = Path(root).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("checkpoint output outside run")
    return [p.relative_to(root).as_posix() for p in sorted(directory.rglob("*")) if p.is_file()]


def _reference_directory(root, directory):
    """The frozen caller precreates an empty directory; its trainer requires none.

    This runs only after a checkpoint miss. Preserve any unverified partial
    output before the original trainer creates and commits a fresh directory.
    """
    root = Path(root).resolve(); directory = Path(directory)
    if directory.is_symlink() or not directory.resolve().is_relative_to(root):
        raise ValueError("reference head directory outside training run")
    if directory.exists():
        if any(directory.iterdir()):
            quarantine = root / "quarantine/training_fast/reference_head"
            quarantine.mkdir(parents=True, exist_ok=True)
            directory.rename(quarantine / (directory.name + "-" + uuid.uuid4().hex))
        else:
            directory.rmdir()


def _resume_root(run_id):
    """Validate the scope before even creating its lease file."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_id):
        raise ValueError("invalid training run identifier")
    base = ARTIFACT_RUNS_ROOT.resolve()
    candidate = base / run_id
    root = candidate.resolve()
    if candidate.is_symlink() or not root.is_relative_to(base):
        raise ValueError("training run outside artifact root")
    return root


def inspect_fast_resume(run_id):
    """Verify provenance before acquiring any reusable computational outputs."""
    root = _resume_root(run_id)
    if (root / "reports/trend_consensus.json").exists():
        raise ValueError("completed training cannot be resumed")
    def read(relative):
        return json.loads((root / relative).read_text())
    state = read("reports/training_acceleration.json")
    initial = read("config/initial_config.json"); metadata = read("metadata.json")
    source = read("data_snapshots/source.json"); records = read("data_snapshots/records.json")
    calendar = read("data_snapshots/calendar.json"); code = read("config/generating_code_hashes.json")
    from my_strategy.services import czsc_trend_pipeline as pipeline
    from my_strategy.services.czsc_trend_contracts import validate_training_config
    validate_training_config(initial)
    current_code = code_identity()
    if (state.get("version") != VERSION or state.get("run_id") != run_id
            or state.get("code_identity") != current_code
            or state.get("config_hash") != stable_hash(initial)
            or state.get("data_version") != source.get("snapshot_hash")
            or metadata.get("run_id") != run_id or metadata.get("task") != "trend-consensus-train"
            or metadata.get("config_hash") != stable_hash(initial)
            or metadata.get("data_version") != source.get("snapshot_hash")
            or initial.get("source_snapshot_hash") != source.get("snapshot_hash")
            or source.get("snapshot_hash") != stable_hash({k: v for k, v in source.items() if k != "snapshot_hash"})
            or stable_hash(records) != source.get("records_hash")
            or metadata.get("stocks") != [r["symbol"] for r in records]
            or source.get("selected_symbols") != metadata.get("stocks")
            or initial.get("training_end") != metadata.get("as_of_date")
            or calendar.get("verified") is not True or not calendar.get("dates")
            or calendar["dates"] != sorted(set(calendar["dates"]))
            or set(code) != set(pipeline.GENERATING_FILES)):
        raise ValueError("accelerated resume frozen source/config/code identity mismatch")
    for relative, digest in code.items():
        if current_code.get(relative) != digest or _file_hash(root / "config/source_code" / relative) != digest:
            raise ValueError("accelerated resume generating source archive changed")
    native_path = ARTIFACT_RUNS_ROOT / source["source_native_run_id"] / "reports/native_source.json"
    native = json.loads(native_path.read_text())
    if (_file_hash(native_path) != source["source_report_sha256"] or native.get("status") != "completed"
            or native.get("failures") or native.get("dataset_records") != records
            or native.get("source_snapshot") != source.get("source_snapshot")
            or [d for d in native["calendar_dates"] if d <= initial["training_end"]] != calendar["dates"]
            or calendar.get("source_run_id") != source["source_snapshot"].get("calendar_run_id")):
        raise ValueError("accelerated resume native source/calendar mismatch")
    pipeline.verify_full_population(native, initial, records)
    pipeline.verify_source_generators(source["source_snapshot"])
    for record in records:
        for path_key, hash_key in (("raw_path", "raw_sha256"), ("path", "sha256"), ("native_path", "native_sha256")):
            if _file_hash(resolve_frozen_asset(record[path_key])) != record[hash_key]:
                raise ValueError("accelerated resume input content changed")
    return dict(root=root, run_id=run_id, metadata=metadata, initial_config=initial,
                source=source, records=records, calendar=calendar, state=state)


@contextmanager
def accelerated_pipeline(cpu_workers, device="cpu", cache_path=None, *, resume_plan=None):
    """Use cache only within one leased, separately frozen orchestration attempt."""
    from my_strategy.services import czsc_trend_pipeline as p, czsc_reference_quality as q
    from my_strategy.services.czsc_training_fast import accelerated_training, _encode, _decode
    from my_strategy.services.czsc_training_cache import TrainingCache
    cache = TrainingCache(cache_path); identity = code_identity(); state = {}; context = None
    if not _LOCK.acquire(blocking=False):
        raise ValueError("a training acceleration context is already active")
    original = {name: getattr(p, name) for name in ("create_run_context", "_save", "_pool_parquets",
        "train_trend_head", "_fit_interval", "select_source_policy", "select_holding_exit_policy",
        "freeze_holding_population")}
    original_reference = q.train_reference_head
    counters = {"stage_hits": 0, "stage_computed": 0, "pool_hits": 0}
    def key(kind, value):
        return stable_hash({"version": VERSION, "code": identity, "run_id": context.run_id,
                            "kind": kind, "input": value})
    def checkpoint(kind, value, call, output_files):
        token = key(kind, value)
        hit = cache.get_checkpoint(token, context.run_dir())
        if hit is not None:
            counters["stage_hits"] += 1
            print("TRAINING_CACHE " + json.dumps({"stage": kind, "status": "reused"}), flush=True)
            return _decode(hit["typed_result"])
        result = call()
        # Persist only after all expected outputs have been committed successfully.
        cache.put_checkpoint(token, context.run_dir(), {"typed_result": _encode(result)}, output_files(result))
        counters["stage_computed"] += 1
        return result
    def create(**kwargs):
        nonlocal context
        if context is not None:
            raise ValueError("acceleration context may own only one training run")
        if resume_plan:
            if (stable_hash(kwargs["config"]) != resume_plan["metadata"]["config_hash"]
                    or kwargs["stocks"] != resume_plan["source"]["selected_symbols"]
                    or kwargs["data_version"] != resume_plan["source"]["snapshot_hash"]):
                raise ValueError("resumed pipeline derived a different frozen identity")
            context = _ResumeContext(resume_plan)
        else:
            context = original["create_run_context"](**kwargs)
            stack.enter_context(_run_lease(context.run_dir()))
        state.update(version=VERSION, run_id=context.run_id, status="running", owner_pid=os.getpid(),
            attempt_id=uuid.uuid4().hex, code_identity=identity, device=device, cpu_workers=cpu_workers,
            config_hash=stable_hash(kwargs["config"]), data_version=kwargs["data_version"],
            cache_path=str(cache.db_path),
            started_at=local_now().isoformat(), native_rebuilt=False, model_qualification_changed=False)
        _save(context.subdir("reports") / "training_acceleration.json", state)
        return context
    def save(path, value):
        path = Path(path)
        if path.is_file():
            old = json.loads(path.read_text()); new = _json(value)
            if old == new:
                return
            # Replay prior folds without truncating a later committed catalogue.
            if resume_plan and path.name in {"l2_bundles.json", "population_folds.json"}:
                if isinstance(old, list) and old[:len(new)] == new:
                    return
        return original["_save"](path, value)
    def pool(paths, destination, *, load=True):
        paths = list(paths); destination = Path(destination)
        token = key("pool", {"paths": [(str(x), _file_hash(Path(x))) for x in paths], "output": str(destination)})
        hit = cache.get_checkpoint(token, context.run_dir())
        if hit is not None:
            counters["pool_hits"] += 1
            return pd.read_parquet(destination) if load else None
        result = original["_pool_parquets"](paths, destination, load=load)
        cache.put_checkpoint(token, context.run_dir(), {"output": str(destination)},
                             [destination.relative_to(context.run_dir()).as_posix()])
        return result
    def trend(data, directory, fold, cfg, *, contract):
        return checkpoint("trend_head", {"data": frame_identity(data), "fold": fold, "cfg": cfg, "contract": contract},
            lambda: original["train_trend_head"](data, directory, fold, cfg, contract=contract),
            lambda _: _files(context.run_dir(), directory))
    def interval(data, head, fold, calendar, run_context, cfg, architecture="linear", device="cpu"):
        result = checkpoint("interval_head", {"data": frame_identity(data), "head": head, "fold": fold,
            "calendar": calendar, "cfg": cfg, "architecture": architecture, "device": device},
            lambda: original["_fit_interval"](data, head, fold, calendar, run_context, cfg, architecture, device),
            lambda result: _files(context.run_dir(), context.run_dir() / result[0]["heads"][head]["path"]))
        return tuple(result)
    def source_policy(records, calendar, cfg, root, trends, run_context, workers):
        identity_value = {"records": records, "calendar": calendar, "cfg": copy.deepcopy(cfg), "trends": trends}
        def call():
            report = original["select_source_policy"](records, calendar, cfg, root, trends, run_context, workers)
            return {"report": report, "trend": copy.deepcopy(cfg["trend"]), "consensus": copy.deepcopy(cfg["consensus"])}
        result = checkpoint("source_policy", identity_value, call,
            lambda r: [r["report"]["accounts_asset"]["path"], "validation/source_policy_selection.json"])
        cfg["trend"] = copy.deepcopy(result["trend"]); cfg["consensus"] = copy.deepcopy(result["consensus"])
        return result["report"]
    def holding(data, bundle, root, cfg):
        def call():
            report = original["select_holding_exit_policy"](data, bundle, root, cfg)
            return {"report": report, "bundle": copy.deepcopy(bundle)}
        result = checkpoint("holding_policy", {"data": frame_identity(data), "bundle": copy.deepcopy(bundle), "cfg": cfg},
            call, lambda _: _files(context.run_dir(), Path(root) / bundle["heads"]["ma_exit"]["path"]))
        bundle.clear(); bundle.update(copy.deepcopy(result["bundle"]))
        return result["report"]
    def reference(data, directory, fold, cfg, contract):
        def fit():
            _reference_directory(context.run_dir(), directory)
            return original_reference(data, directory, fold, cfg, contract)
        return checkpoint("reference_head", {"data": frame_identity(data), "fold": fold, "cfg": cfg, "contract": contract},
            fit,
            lambda _: _files(context.run_dir(), directory))
    def holding_population(dataset, root, cfg, records):
        root = Path(root).resolve()
        if root != context.run_dir().resolve():
            raise ValueError("holding population belongs to another training run")
        manifest = root / "dataset/E1/manifest.json"
        exits = root / "dataset/E1/exits.parquet"
        frozen = root / "data_snapshots/holding_population.json"
        def freeze():
            result = original["freeze_holding_population"](dataset, root, cfg, records)
            _save(frozen, result)
            return result
        result = checkpoint("holding_population", {"cfg": cfg, "records": records,
            "manifest_sha256": _file_hash(manifest), "exits_sha256": _file_hash(exits),
            "reference_population": dataset.get("reference_population"), "reference_mode": dataset.get("reference_mode")},
            freeze, lambda _: [x.relative_to(root).as_posix() for x in (manifest, exits, frozen)])
        p.audit_holding_population(result, root, cfg, records)
        dataset["holding_population"] = copy.deepcopy(result)
        return result
    replacements = dict(create_run_context=create, _save=save, _pool_parquets=pool, train_trend_head=trend,
        _fit_interval=interval, select_source_policy=source_policy, select_holding_exit_policy=holding,
        freeze_holding_population=holding_population)
    try:
        # Lease stack outlives the runtime's final content verification.
        with ExitStack() as stack:
            try:
                with accelerated_training(cpu_workers, cache_path=cache_path) as runtime:
                    for name, function in replacements.items():
                        setattr(p, name, function)
                    q.train_reference_head = reference
                    try:
                        yield counters
                        if code_identity() != identity:
                            raise ValueError("training acceleration code changed during frozen run")
                    finally:
                        for name, function in original.items():
                            setattr(p, name, function)
                        q.train_reference_head = original_reference
                if context:
                    state.update(status="completed", finished_at=local_now().isoformat(), counters=counters)
                    _save(context.subdir("reports") / "training_acceleration.json", state)
            except BaseException as error:
                if context:
                    state.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
                        finished_at=local_now().isoformat(), counters=counters, error=type(error).__name__ + ": " + str(error))
                    _save(context.subdir("reports") / "training_acceleration.json", state)
                raise
    finally:
        _LOCK.release()


def resume_fast_training(run_id, *, verify_only=False, cache_path=None, **request):
    from my_strategy.services import czsc_trend_pipeline as pipeline
    from my_strategy.services.czsc_trend_contracts import TRAIN_KIND
    from my_strategy.storage.czsc_results import ResultStore
    # Same lock file as the original E1 adapter, so the two cannot race.
    root = _resume_root(run_id)
    with _run_lease(root):
        plan = inspect_fast_resume(run_id); cfg = plan["initial_config"]
        frozen_cache = Path(plan["state"]["cache_path"])
        if not frozen_cache.is_absolute() or not frozen_cache.is_file():
            raise ValueError("the original training cache database is missing; preserve completed checkpoints")
        if cache_path is not None and Path(cache_path).resolve() != frozen_cache.resolve():
            raise ValueError("accelerated resume requested another checkpoint database")
        cache_path = frozen_cache
        for name, expected in (("end", cfg["training_end"]), ("cpu_workers", cfg["cpu_workers"]),
                ("device", plan["state"]["device"]), ("source_native_run_id", plan["source"]["source_native_run_id"]),
                ("stock_batch_size", cfg["stock_batch_size"]), ("calendar_run_id", plan["calendar"]["source_run_id"])):
            if request.get(name) is not None and request[name] != expected:
                raise ValueError("accelerated resume requested changed " + name)
        if request.get("cohort") is not None:
            cohort = request["cohort"]
            if (cohort.get("symbols") if isinstance(cohort, dict) else cohort) != plan["source"]["selected_symbols"]:
                raise ValueError("accelerated resume requested another stock cohort")
        if request.get("config") is not None:
            from my_strategy.services.czsc_trend_contracts import validate_training_config
            derived = {"pipeline_contract_hash", "source_snapshot_hash", "training_end", "cpu_workers",
                       "training_capacity_tiers", "capacity_expansion_criterion"}
            registered = validate_training_config(request["config"])
            if {k: v for k, v in registered.items() if k not in derived} != {k: v for k, v in cfg.items() if k not in derived}:
                raise ValueError("accelerated resume model/label parameters changed")
        if verify_only:
            return {"run_id": run_id, "status": "source_verified", "checkpoint_validation": "on_replay", "data_end": cfg["training_end"],
                    "source_count": len(plan["records"]), "native_rebuilt": False}
        initial = copy.deepcopy(cfg); initial.pop("pipeline_contract_hash", None)
        with accelerated_pipeline(cfg["cpu_workers"], plan["state"]["device"], cache_path, resume_plan=plan):
            report = pipeline.train_trend_consensus(source_native_run_id=plan["source"]["source_native_run_id"],
                end=cfg["training_end"], config=initial, cpu_workers=cfg["cpu_workers"], device=plan["state"]["device"])
        pipeline._verify_run(report, plan["root"])
        ResultStore().save(TRAIN_KIND, _json(report))
        return report
