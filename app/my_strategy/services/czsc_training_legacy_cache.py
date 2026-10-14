"""Continue the frozen legacy run after its complete reference population.

The original E1 resume adapter verifies and reuses earlier stages. This thin
outer adapter additionally proves the completed E1 pools and reference probes,
then lets the unchanged reference fitting, fee qualification and L2 algorithms
continue. It repairs only the empty-directory orchestration mismatch: the
original caller creates a head directory that its trainer expects not to exist.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import re
import threading
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_trend_support import _file_hash, _json, _save


VERSION = "training_legacy_reference_cache_v1"
ADAPTER_FILES = ("services/czsc_training_legacy_cache.py", "services/czsc_training_fast.py",
                 "services/czsc_training_cache.py", "services/czsc_trend_resume.py",
                 "scripts/resume_trend_training.py", "scripts/resume_training_fast.py",
                 "scripts/resume_training_legacy_cached.py")
_LOCK = threading.Lock()


def _root(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_id):
        raise ValueError("invalid legacy training run identifier")
    root = ARTIFACT_RUNS_ROOT / run_id
    resolved = root.resolve(strict=True)
    if root.is_symlink() or not resolved.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()) or not resolved.is_dir():
        raise ValueError("legacy training run escapes artifact root")
    if (resolved / ".training_resume.lock").is_symlink():
        raise ValueError("legacy training lease must not be a symlink")
    return resolved


def _inside(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)):
        raise ValueError("legacy checkpoint file escapes its run")
    return resolved


def _tracked(root, value, proof, expected=None):
    path = _inside(root, value)
    if not path.is_file():
        raise ValueError("legacy completed checkpoint file missing: " + str(path))
    actual = _file_hash(path)
    if expected is not None and actual != expected:
        raise ValueError("legacy completed checkpoint SHA mismatch: " + str(path))
    if str(path) in proof and proof[str(path)] != actual:
        raise ValueError("legacy completed checkpoint changed during verification")
    proof[str(path)] = actual
    return path


def _read(root, value, proof):
    return json.loads(_tracked(root, value, proof).read_text())


def _code_identity():
    return {name: _file_hash(MY_STRATEGY_ROOT / name) for name in ADAPTER_FILES}


def _verify_pooled_reference(root, manifest):
    """Compare every pooled row with the ordered immutable stock tables.

    Arrow batches bound memory; no Broker replay or label recalculation occurs.
    Semantic equality includes null masks, exact values, field types and order.
    """
    pooled = pq.ParquetFile(_inside(root, manifest["pooled_path"]))
    if pooled.metadata.num_rows != sum(r["rows"] for r in manifest["records"]):
        raise ValueError("legacy reference pooled population row count differs")
    batches = iter(pooled.iter_batches(batch_size=8192))
    pending = None
    for record in manifest["records"]:
        stock = pq.read_table(_inside(root, record["files"]["observations.parquet"]["path"]))
        if stock.num_rows != record["rows"] or not stock.schema.equals(pooled.schema_arrow, check_metadata=False):
            raise ValueError("legacy reference stock row/schema differs")
        offset = 0
        while offset < stock.num_rows:
            if pending is None or pending.num_rows == 0:
                try:
                    pending = pa.Table.from_batches([next(batches)])
                except StopIteration as exc:
                    raise ValueError("legacy reference pooled rows are missing") from exc
            count = min(stock.num_rows - offset, pending.num_rows)
            if not stock.slice(offset, count).equals(pending.slice(0, count), check_metadata=False):
                raise ValueError("legacy reference pooled rows differ from stock order/content")
            offset += count
            pending = pending.slice(count)
    if (pending is not None and pending.num_rows) or next(batches, None) is not None:
        raise ValueError("legacy reference pooled rows contain extras")


def verify_completed_reference(plan, *, check_cancel=None, progress=None):
    """Extend the original adapter's proof with complete, bound later assets."""
    from my_strategy.services import czsc_reference_quality as q
    from my_strategy.services.czsc_account_artifacts import validate_account_artifact
    root = plan["root"].resolve(); proof = plan["proof"]
    records = plan["records"]; cfg = plan["config"]; calendar = plan["calendar"]["dates"]
    expected_symbols = [r["symbol"] for r in records]
    if len(plan["cached_E1"]) != len(records):
        raise ValueError("legacy reference continuation requires complete E1")
    e1 = _read(root, "dataset/E1/manifest.json", proof)
    if (e1.get("stage") != "E1" or e1.get("failures") or e1.get("reference_population") != {}
            or e1.get("population_end") is not None or e1.get("reference_population_hash") != stable_hash({})
            or e1.get("source_snapshot_hash") != cfg["source_snapshot_hash"]
            or e1.get("records") != [plan["cached_E1"][s] for s in expected_symbols]
            or set(e1.get("pooled_files", {})) != {"entries", "exits", "episodes"}):
        raise ValueError("legacy complete E1 manifest differs from verified stocks")
    for name, asset in e1["pooled_files"].items():
        path = _tracked(root, asset["path"], proof, asset["sha256"])
        if path != root / "dataset/E1" / (name + ".parquet"):
            raise ValueError("legacy E1 pooled asset has another stage path")
    protocol = _read(root, "config/reference_quality_protocol.json", proof)
    manifest = _read(root, "dataset/reference_quality/manifest.json", proof)
    if (protocol != q.reference_protocol(cfg) or manifest.get("protocol") != protocol
            or manifest.get("protocol_hash") != protocol["protocol_hash"]
            or manifest.get("population") != q.POPULATION or manifest.get("probe_initial_cash") != q.PROBE_CASH
            or manifest.get("failures") or manifest.get("source_snapshot_hash") != cfg["source_snapshot_hash"]
            or manifest.get("pipeline_contract_hash") != cfg["pipeline_contract_hash"]
            or manifest.get("continuation_policy_hash") != cfg["E1_policy_hash"]
            or manifest.get("calendar_hash") != stable_hash(calendar)
            or manifest.get("trend_catalog_hash") != stable_hash(plan["trends"])
            or manifest.get("ma_exit_catalog_hash") != stable_hash(plan["ma"])
            or manifest.get("population_hash") != stable_hash({k: v for k, v in manifest.items() if k != "population_hash"})
            or [r.get("symbol") for r in manifest.get("records", [])] != expected_symbols
            or manifest.get("pooled_path") != "dataset/reference_quality/observations.parquet"):
        raise ValueError("legacy reference population/source/config/calendar/catalog identity mismatch")
    _tracked(root, manifest["pooled_path"], proof, manifest["pooled_sha256"])
    for index, (item, record) in enumerate(zip(manifest["records"], records, strict=True)):
        if check_cancel:
            check_cancel()
        if (item.get("source_record") != record or item.get("source_record_sha256") != record["sha256"]
                or type(item.get("rows")) is not int or item["rows"] < 0):
            raise ValueError("legacy reference source record binding mismatch")
        files = item.get("files", {})
        graph_names = [name for name in files if re.fullmatch(r"account_evidence\.[a-f0-9]{64}\.json\.gz", name)]
        if len(graph_names) != 1 or set(files) != {"observations.parquet", "probes.json", graph_names[0]}:
            raise ValueError("legacy reference completed stock file inventory incomplete")
        stock_root = root / "dataset/reference_quality/stocks" / item["symbol"].replace(".", "_")
        for name, asset in files.items():
            path = _tracked(root, asset["path"], proof, asset["sha256"])
            if path != stock_root / name:
                raise ValueError("legacy reference file belongs to another stock")
        account = validate_account_artifact(stock_root / "probes.json", expected_sha256=files["probes.json"]["sha256"])
        if account.get("asset") != graph_names[0] or account.get("asset_sha256") != files[graph_names[0]]["sha256"]:
            raise ValueError("legacy reference compact account graph binding mismatch")
        if progress and (index % 40 == 0 or index + 1 == len(records)):
            progress(dict(stage="verify_reference_population", completed=index + 1, total=len(records),
                          verified_reference_stocks=index + 1, native_rebuilt=False))
    _verify_pooled_reference(root, manifest)
    plan["reference_population"] = manifest
    plan["reference_protocol"] = protocol
    plan["reused"].update(reference_population=len(records), reference_pooled=True, E1_pooled=True)
    return plan


def _archive_holding_diagnostics(plan, attempt):
    """Preserve the one pre-L2 diagnostic whose unpersisted clock must change."""
    root = plan["root"]
    path = root / "validation/holding_population.json"
    if not path.exists():
        return None
    if (root / "dataset/final_population").exists() or (root / "models/l2_bundles.json").exists():
        raise ValueError("cannot refreeze a holding population after L2 started")
    old = _read(root, path, plan["proof"])
    digest = plan["proof"][str(path)]
    archive = _inside(root, attempt.with_suffix(".previous_holding_population.json"))
    payload = path.read_bytes()
    if _file_hash(path) != digest:
        raise ValueError("holding diagnostic changed before archival")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("xb") as stream:
        stream.write(payload)
    _tracked(root, archive, plan["proof"], digest)
    plan["proof"].pop(str(path))
    plan["previous_holding_diagnostics"] = old
    plan["previous_holding_diagnostics_sha256"] = digest
    return dict(reason="original pre-L2 freeze timestamp was not persisted; refreeze the same verified E1 population",
                original_path=str(path.relative_to(root)), original_sha256=digest,
                archived_path=str(archive.relative_to(root)), archived_sha256=digest,
                original_population_hash=old.get("holding_population_hash"))


def _save_refrozen_holding(plan, value, population, save):
    """Permit a new freeze clock only; samples, masks and lineage stay exact."""
    from my_strategy.services import czsc_trend_pipeline as p
    root = plan["root"]; path = root / "validation/holding_population.json"
    old = plan["previous_holding_diagnostics"]
    if (root / "dataset/final_population").exists() or (root / "models/l2_bundles.json").exists():
        raise ValueError("cannot refreeze a holding population after L2 started")
    if _file_hash(path) != plan["previous_holding_diagnostics_sha256"]:
        raise ValueError("holding diagnostic changed before refreeze")
    if ({k: v for k, v in value.items() if k != "holding_population_hash"}
            != {k: v for k, v in old.items() if k != "holding_population_hash"}
            or value.get("holding_population_hash") != stable_hash(population)):
        raise ValueError("holding refreeze would change samples, masks or population binding")
    p.audit_holding_population(population, root, plan["config"], plan["records"])
    save(path, value)
    digest = _file_hash(path)
    plan["proof"][str(path)] = digest
    return dict(new_sha256=digest, new_population_hash=stable_hash(population),
                frozen_before_L2_training_at=population["frozen_before_L2_training_at"],
                diagnostic_fields_unchanged_except_population_hash=True,
                original_native_labels_and_models_unchanged=True)


@contextmanager
def _reference_replay(plan_holder, cache_path, identity):
    from my_strategy.services import czsc_reference_quality as q, czsc_trend_pipeline as p
    from my_strategy.services.czsc_training_cache import TrainingCache
    from my_strategy.services.czsc_training_fast import _digest, _encode, _decode
    original_run, original_fit, original_save = q.run_reference_quality, q.train_reference_head, q._save
    cache = TrainingCache(cache_path)
    def save(path, value):
        plan = plan_holder.get("plan"); path = Path(path).resolve()
        expected = plan and plan["proof"].get(str(path))
        if expected:
            if _file_hash(path) != expected or json.loads(path.read_text()) != _json(value):
                raise ValueError("reference replay would overwrite completed frozen evidence")
            return None
        return original_save(path, value)
    def fit(data, directory, fold, cfg, contract):
        plan = plan_holder["plan"]; root = plan["root"]
        directory = _inside(root, directory)
        token = _digest({"version": VERSION, "code": identity, "root": str(root),
                         "data": data, "fold": fold, "config": cfg, "contract": contract})
        cached = cache.get_checkpoint(token, root)
        if cached is not None:
            manifest = q.load_reference_quality_head(directory, expected_contract=contract)
            result = _decode(cached["manifest"])
            if manifest != result:
                raise ValueError("legacy reference head checkpoint summary differs")
            return result
        if directory.exists():
            if any(directory.iterdir()):
                raise ValueError("legacy reference head is nonempty without a verified checkpoint")
            directory.rmdir()
        result = original_fit(data, directory, fold, cfg, contract)
        files = {str((directory / name).relative_to(root)): _file_hash(directory / name)
                 for name in ("model.joblib", "manifest.json")}
        cache.put_checkpoint(token, root, {"manifest": _encode(result)}, files)
        return result
    def run(records, calendar, cfg, context, trends, ma, workers, folds):
        plan = plan_holder["plan"]; root = plan["root"]
        if (context.run_dir().resolve() != root or records != plan["records"] or calendar != plan["calendar"]["dates"]
                or cfg != plan["config"] or trends != plan["trends"] or ma != plan["ma"]
                or workers != plan["initial_config"]["cpu_workers"]):
            raise ValueError("legacy reference replay requested another computation")
        if "reference_population" not in plan:
            return original_run(records, calendar, cfg, context, trends, ma, workers, folds)
        summaries = plan["reference_population"]["records"]
        delegate = p._bounded_results
        def results(worker, payloads, cpu_workers, batch_size):
            if worker is q._probe_worker:
                count = 0
                for payload, summary, record in zip(payloads, summaries, records, strict=True):
                    if payload != (record, calendar, cfg, str(root), trends, ma):
                        raise ValueError("legacy reference worker payload changed")
                    count += 1
                    yield copy.deepcopy({k: v for k, v in summary.items() if k != "source_record"})
                print("TRAINING_CACHE " + json.dumps({"stage": "reference_population", "status": "reused", "stocks": count}), flush=True)
                return
            yield from delegate(worker, payloads, cpu_workers, batch_size)
        p._bounded_results = results
        try:
            return original_run(records, calendar, cfg, context, trends, ma, workers, folds)
        finally:
            p._bounded_results = delegate
    q.run_reference_quality, q.train_reference_head, q._save = run, fit, save
    try:
        yield cache
    finally:
        q.run_reference_quality, q.train_reference_head, q._save = original_run, original_fit, original_save


def resume_legacy_cached(run_id, *, verify_only=False, cache_path=None, **request):
    """Continue the same failed run with exactly its original training request."""
    from my_strategy.services import czsc_trend_resume as legacy
    from my_strategy.services import czsc_training_fast as fast
    root = _root(run_id)
    identity = _code_identity()
    if not _LOCK.acquire(blocking=False):
        raise ValueError("a legacy cached continuation is already active")
    if legacy._THREAD_LOCK.locked():
        _LOCK.release()
        raise ValueError("an original legacy continuation is already active")
    original_inspect, original_lease = legacy.inspect_training_resume, legacy._run_lease
    original_boundaries = legacy._reuse_boundaries
    holder = {}; audit = {}; attempt = None
    progress = request.pop("progress", None); check_cancel = request.pop("check_cancel", None)
    def unchanged(plan):
        if _code_identity() != identity:
            raise ValueError("legacy cached continuation code changed during execution")
        for name, digest in plan["proof"].items():
            if Path(name) != root / "metadata.json" and _file_hash(Path(name)) != digest:
                raise ValueError("legacy cached continuation changed completed evidence")
    def inspect(run_id, *, progress=None, check_cancel=None):
        nonlocal attempt
        plan = original_inspect(run_id, progress=progress, check_cancel=check_cancel)
        legacy._verify_request(plan, **request)
        reference_manifest = root / "dataset/reference_quality/manifest.json"
        if reference_manifest.exists() or reference_manifest.is_symlink():
            verify_completed_reference(plan, check_cancel=check_cancel, progress=progress)
        holding_path = root / "validation/holding_population.json"
        if holding_path.exists() or holding_path.is_symlink():
            _read(root, holding_path, plan["proof"])
        for name, digest in identity.items():
            path = str((MY_STRATEGY_ROOT / name).resolve())
            if path in plan["proof"] and plan["proof"][path] != digest:
                raise ValueError("legacy cached adapter code differs from original frozen source")
            plan["proof"][path] = digest
        holder["plan"] = plan
        if not verify_only:
            attempt = root / "reports" / ("training_legacy_cache." + local_now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:12] + ".json")
            holding_refreeze = _archive_holding_diagnostics(plan, attempt)
            audit.update(version=VERSION, run_id=run_id, status="running", code_identity=identity,
                         original_config_hash=stable_hash(plan["initial_config"]),
                         protected_files=len(plan["proof"]), protected_identity_hash=stable_hash(plan["proof"]),
                         reused=copy.deepcopy(plan["reused"]), cache_path=str(holder["cache"].db_path),
                         started_at=local_now().isoformat(), native_rebuilt=False, training_parameters_changed=False,
                         reference_population_hash=plan.get("reference_population", {}).get("population_hash"),
                         holding_refreeze=holding_refreeze)
            _save(attempt, audit)
        return plan
    @contextmanager
    def boundaries(plan, check_cancel=None):
        from my_strategy.services import czsc_trend_pipeline as p
        with original_boundaries(plan, check_cancel) as stats:
            original_save, original_freeze = p._save, p.freeze_holding_population
            holding = {}
            def freeze(dataset, current_root, cfg, records):
                if (Path(current_root).resolve() != root or cfg != plan["config"] or records != plan["records"]):
                    raise ValueError("holding refreeze requested another frozen computation")
                result = original_freeze(dataset, current_root, cfg, records)
                holding["population"] = copy.deepcopy(result)
                return result
            def before_report(path, value):
                if (Path(path).resolve() == root / "validation/holding_population.json"
                        and "previous_holding_diagnostics" in plan):
                    info = _save_refrozen_holding(plan, value, holding["population"], original_save)
                    audit["holding_refreeze"].update(info)
                    _save(attempt, audit)
                    return None
                if Path(path).resolve() == root / "reports/trend_consensus.json":
                    if fast._STATE is not None:
                        fast._STATE.verify()
                    unchanged(plan)
                return original_save(path, value)
            p._save, p.freeze_holding_population = before_report, freeze
            try:
                yield stats
                if _code_identity() != identity:
                    raise ValueError("legacy cached continuation code changed during execution")
            finally:
                p._save, p.freeze_holding_population = original_save, original_freeze
    @contextmanager
    def lease(path):
        with original_lease(path):
            try:
                with fast.accelerated_training(initial["cpu_workers"], cache_path=cache_path):
                    with _reference_replay(holder, cache_path, identity) as cache:
                        holder["cache"] = cache
                        yield
                if attempt:
                    audit.update(status="completed", finished_at=local_now().isoformat(), publication_allowed=False)
                    _save(attempt, audit)
            except BaseException as exc:
                if attempt:
                    audit.update(status="failed", finished_at=local_now().isoformat(), error=type(exc).__name__ + ": " + str(exc))
                    _save(attempt, audit)
                raise
    try:
        if verify_only:
            with original_lease(root):
                plan = inspect(run_id, progress=progress, check_cancel=check_cancel)
                return {"run_id": run_id, "status": "source_verified", "native_rebuilt": False,
                        "reused": copy.deepcopy(plan["reused"]), "protected_files": len(plan["proof"])}
        initial = json.loads((root / "config/initial_config.json").read_text())
        # The original adapter acquires the same-run lease before invoking our
        # strengthened inspector. Never acquire a second competing lease here.
        legacy.inspect_training_resume, legacy._run_lease, legacy._reuse_boundaries = inspect, lease, boundaries
        report = legacy.resume_training_run(run_id, progress=progress, check_cancel=check_cancel, request=request)
        return report
    finally:
        legacy.inspect_training_resume, legacy._run_lease = original_inspect, original_lease
        legacy._reuse_boundaries = original_boundaries
        _LOCK.release()
