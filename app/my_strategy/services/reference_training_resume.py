"""Strict same-run continuation for the account-free reference training family.

This separately frozen adapter may repair orchestration after an interrupted
attempt. It cannot change the original model, labels, windows or configuration.
Completed weights are physically verified and remain immutable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import time
import uuid

import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services import reference_training as training

FAMILY = "reference_position_v1"


def predictive_gate(validation):
    policy = validation.get("policy", validation.get("metrics", {}).get("policy", {}))
    baseline = policy.get("baseline", {})
    if not isinstance(baseline, dict):
        baseline = {}
    values = (policy.get("log_loss"), policy.get("brier"),
              policy.get("baseline_log_loss", baseline.get("log_loss")),
              policy.get("baseline_brier", baseline.get("brier")))
    valid = all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values)
    passed = bool(valid and values[0] <= values[2] and values[1] <= values[3])
    return dict(passed=passed, status="passed" if passed else "pending",
        basis="policy_only_prior_baseline_logloss_and_brier", log_loss=values[0], brier=values[1],
        baseline_log_loss=values[2], baseline_brier=values[3])


def _inside(root, path):
    path = Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("参考恢复资产不属于原运行目录")
    return path


@contextmanager
def _lease(root):
    with (root/".reference_resume.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            if not handle.seek(0, os.SEEK_END):
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise ValueError("同一参考训练已有恢复任务持有租约") from exc
        try:
            yield
        finally:
            release()


def _read_original(root, run_id, plan_id=None):
    report = training._read(root/"reports/reference_training.json")
    if report.get("family") != FAMILY or report.get("run_id") != run_id:
        raise ValueError("仅允许恢复原 reference_position_v1 运行")
    if report.get("status") not in {"failed", "cancelled"}:
        raise ValueError("仅允许恢复已失败或已取消且不活跃的参考训练")
    plan = training._read(root/"config/training_plan.json")
    if (plan.get("family") != FAMILY or plan.get("plan_hash") != stable_hash({k:v for k,v in plan.items() if k!="plan_hash"})
            or report.get("plan_id") != plan.get("plan_id") or (plan_id is not None and plan_id != plan["plan_id"])):
        raise ValueError("原冻结计划身份或哈希不符")
    cfg = training.parameters(plan["parameters"])
    if any(report.get("config", {}).get(k) != v for k,v in cfg.items()):
        raise ValueError("原参考运行配置与冻结计划不同")
    metadata = training._read(root/"metadata.json")
    if metadata.get("run_id") != run_id or metadata.get("config_hash") != stable_hash(cfg):
        raise ValueError("原参考运行元信息与冻结配置不同")
    frozen = root/"config/frozen_code"
    for name, digest in plan["code_hashes"].items():
        path = _inside(frozen, frozen/name)
        if training._sha(path) != digest:
            raise ValueError("原冻结源码哈希改变："+name)
    from my_strategy.services import reference_model
    if training._sha(Path(reference_model.__file__)) != plan["code_hashes"].get("reference_model.py"):
        raise ValueError("数值模型源码已改变，禁止恢复原参考运行")
    if not report.get("dataset", {}).get("manifest_sha256"):
        raise ValueError("原运行尚无完整且绑定的全量数据集，必须新建训练计划")
    return report, plan, cfg


def _verify_dataset(root, plan, report, runs, emit, check_cancel):
    native, calendar, proof = training._source_header(plan["parameters"]["source_native_run_id"], runs)
    if proof != plan["source_reuse"] or proof != report["source_reuse"]:
        raise ValueError("恢复来源与原冻结来源不同")
    manifest_path = root/"dataset/manifest.json"
    if training._sha(manifest_path) != report["dataset"]["manifest_sha256"]:
        raise ValueError("原训练数据集 manifest 哈希改变")
    manifest = training._read(manifest_path)
    if (manifest.get("family") != FAMILY or manifest.get("source_reuse") != proof
            or manifest.get("code_hash") != plan["code_hashes"]["reference_model.py"]):
        raise ValueError("原训练数据集来源或数值代码不符")
    records = manifest["records"]
    if [r["symbol"] for r in records] != plan["cohort"] or native["selected_symbols"] != plan["cohort"]:
        raise ValueError("恢复数据集不包含完整冻结人口")
    if len(records) != report["dataset"]["stocks"] or len(records) != report["historical_union_count"]:
        raise ValueError("恢复数据集人口分母不符")
    source_records = {r["symbol"]: r for r in native["dataset_records"]}
    from my_strategy.services.reference_model import FEATURE_COLUMNS, SCHEMA_HASH
    def one(record):
        if check_cancel: check_cancel()
        source = source_records[record["symbol"]]
        training._verified_record(source, runs)
        path = _inside(root/"dataset", Path(record["path"]))
        if training._sha(path) != record["sha256"] or record["source_sha256"] != source["sha256"]:
            raise ValueError("原逐股数据或来源 SHA 改变："+record["symbol"])
        frame = pd.read_parquet(path)
        if (not set(FEATURE_COLUMNS).issubset(frame) or frame.attrs.get("schema_hash") != SCHEMA_HASH
                or len(frame) != record["rows"] or frame.symbol.nunique() != 1 or str(frame.symbol.iloc[0]) != record["symbol"]
                or int(frame.input_eligible.sum()) != record["eligible"] or int(frame.label_available.sum()) != record["labels"]
                or frame.date.astype(str).min() < plan["parameters"]["feature_start"]
                or frame.date.astype(str).max() > plan["parameters"]["end"]):
            raise ValueError("原逐股数据身份、资格或时间范围不符："+record["symbol"])
        return frame
    frames = []
    with ThreadPoolExecutor(max_workers=plan["parameters"]["cpu_workers"]) as pool:
        for i, frame in enumerate(pool.map(one, records), 1):
            frames.append(frame)
            if i % 40 == 0 or i == len(records): emit("reference_resume_verification", i, len(records), device="cpu")
    data = pd.concat(frames, ignore_index=True); del frames
    if (len(data) != report["dataset"]["rows"] or int(data.input_eligible.sum()) != report["dataset"]["eligible"]
            or int(data.label_available.sum()) != report["dataset"]["labeled"]):
        raise ValueError("原数据集汇总人口不符")
    for column in ("symbol", "date"): data[column] = data[column].astype("category")
    return data, calendar, manifest


def _completed_events(root):
    path = root/"logs/training.jsonl"
    events = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            event = json.loads(line)
            if event.get("stage") == "fold_complete": events.append(event)
    return events


def _reusable_fold(root, split, cfg, report, plan, events):
    """Require an original same-run completion event, not merely a weight file."""
    from my_strategy.services.reference_model import _config, load_model
    directory = _inside(root/"models", root/"models"/split["name"])
    path = directory/"manifest.json"
    if not path.is_file(): return None
    session = load_model(directory, device="cpu")
    manifest = session.manifest
    if manifest.get("split") != split or manifest.get("config") != _config(cfg):
        raise ValueError("已完成参考模型配置或时间切分与原计划不同")
    if manifest.get("validation", {}).get("split") != split:
        raise ValueError("已完成参考模型验证窗口与原计划不同")
    trained_at = pd.Timestamp(manifest["actual_trained_at"])
    original_start = pd.Timestamp(report["started_at"])
    end = pd.Timestamp(report.get("ended_at") or local_now().isoformat())
    completion = [e for e in events if e.get("fold") == split["name"] and e.get("total", 0) > 0 and e.get("current") == e.get("total")
                  and trained_at <= pd.Timestamp(e["at"]) <= end]
    if not original_start <= trained_at <= end or not completion:
        raise ValueError("已完成参考模型缺少原运行的完成与时间归属证据")
    prior_fold = next((f for f in [*report.get("folds", []), report.get("production")] if f and f.get("name") == split["name"]), None)
    if prior_fold and (prior_fold.get("split") != split or prior_fold.get("validation") != manifest["validation"]):
        raise ValueError("原报告与已完成模型验证事实不同")
    proof = dict(run_id=report["run_id"], plan_hash=plan["plan_hash"],
        source_reuse_hash=stable_hash(plan["source_reuse"]), dataset_manifest_sha256=report["dataset"]["manifest_sha256"],
        model_code_sha256=plan["code_hashes"]["reference_model.py"],
        manifest_sha256=training._sha(path), weights_sha256=manifest["weights_sha256"],
        original_completion_at=completion[0]["at"], actual_trained_at=manifest["actual_trained_at"],
        same_run_origin="frozen_dataset_and_code_verified_original_fold_completion_event")
    return dict(manifest=manifest, validation=manifest["validation"], model_path=str(directory), reuse_proof=proof)


def _verify_immutable(proofs):
    for path, digest in proofs.items():
        if training._sha(path) != digest:
            raise ValueError("恢复期间已完成资产被修改："+str(path))


def resume_training(run_id, *, plan_id=None, runs_root=None, metadata_root=None,
                    progress=None, check_cancel=None, run_callback=None, **kwargs):
    runs, meta = training._roots(runs_root, metadata_root)
    run_id = training._id(run_id); root = _inside(runs, runs/run_id)
    _read_original(root, run_id)
    with _lease(root):
        report, plan, cfg = _read_original(root, run_id)
        if plan_id is not None and plan_id != plan["plan_id"]:
            dispatch = training.validate_plan(plan_id, runs_root=runs, metadata_root=meta)
            if (dispatch.get("parameters") != cfg or dispatch.get("source_reuse") != plan["source_reuse"]
                    or dispatch.get("cohort") != plan["cohort"] or dispatch.get("windows") != plan["windows"]
                    or dispatch.get("code_hashes", {}).get("reference_model.py") != plan["code_hashes"]["reference_model.py"]):
                raise ValueError("恢复请求计划与原冻结数值、来源、人口或窗口不同")
        if kwargs.get("parameters") is not None:
            values = training.parameters(kwargs["parameters"])
            for key in ("end", "source_native_run_id"):
                if values[key] is None: values[key] = cfg[key]
            if values != cfg: raise ValueError("恢复参数不能改变原冻结计划")
        began = time.monotonic(); started = local_now().isoformat()
        attempt_id = local_now().strftime("%Y%m%d-%H%M%S-%f")+"-"+uuid.uuid4().hex[:8]
        attempt_dir = root/"config/resume_attempts"/attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=False)
        adapter = attempt_dir/"reference_training_resume.py"; shutil.copy2(__file__, adapter)
        attempt = dict(attempt_id=attempt_id, run_id=run_id, status="verifying", started_at=started,
            original_status=report["status"], original_error=report.get("error"), dispatch_plan_id=plan_id, adapter_sha256=training._sha(adapter),
            plan_hash=plan["plan_hash"], model_code_sha256=plan["code_hashes"]["reference_model.py"],
            reused_folds=[], trained_folds=[], removed_partial_folds=[])
        def save_attempt(): training._save(attempt_dir/"attempt.json", attempt)
        def emit(stage, current=0, total=1, **details):
            if check_cancel: check_cancel()
            event = dict(stage=stage, current=int(current), total=int(total), at=local_now().isoformat(),
                resume_attempt_id=attempt_id, **details)
            (root/"logs").mkdir(parents=True, exist_ok=True)
            with (root/"logs/training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False, default=str)+"\n")
            if progress: progress(stage, int(current), int(total), **details)
        save_attempt()
        if run_callback: run_callback(run_id)
        immutable = {}
        try:
            data, calendar, _ = _verify_dataset(root, plan, report, runs, emit, check_cancel)
            events = _completed_events(root)
            cached = {}
            for split in plan["windows"]:
                result = _reusable_fold(root, split, cfg, report, plan, events)
                if result is not None:
                    cached[split["name"]] = result
                    directory = Path(result["model_path"])
                    immutable[directory/"manifest.json"] = training._sha(directory/"manifest.json")
                    immutable[directory/"weights.pt"] = training._sha(directory/"weights.pt")
            attempt.update(status="running", dataset_manifest_sha256=report["dataset"]["manifest_sha256"],
                source_reuse_hash=stable_hash(plan["source_reuse"])); save_attempt()
            report.update(status="running", folds=[], production=None, last_resume_attempt_id=attempt_id)
            report.pop("error", None); report.pop("ended_at", None); report.pop("completed_at", None)
            training._save(root/"reports/reference_training.json", report)
            from my_strategy.services.reference_model import fit_fold
            for index, split in enumerate(plan["windows"]):
                emit("reference_model", index, len(plan["windows"]), fold=split["name"], device=plan["actual_device"])
                result = cached.get(split["name"])
                if result is None:
                    directory = _inside(root/"models", root/"models"/split["name"])
                    if directory.exists():
                        shutil.rmtree(directory); attempt["removed_partial_folds"].append(split["name"]); save_attempt()
                    def model_progress(event):
                        event = dict(event); stage = event.pop("stage", "model_fit")
                        current = event.pop("current", 0); total = event.pop("total", 1)
                        event.setdefault("fold", split["name"]); emit(stage, current, total, **event)
                    result = fit_fold(data, split, directory, device=cfg["device"], config=cfg, progress=model_progress)
                    attempt["trained_folds"].append(split["name"])
                else:
                    attempt["reused_folds"].append(dict(name=split["name"], **result["reuse_proof"]))
                    emit("reference_fold_reused", index+1, len(plan["windows"]), fold=split["name"], weights_unchanged=True)
                fold = dict(name=split["name"], split=split, model_path=str(result["model_path"]),
                    validation=result["validation"], predictive_gate=predictive_gate(result["validation"]),
                    research_reconstruction=split["name"]!="production", effective_from=split.get("test_start"))
                if split["name"] == "production": report["production"] = fold
                else: report["folds"].append(fold)
                _verify_immutable(immutable); save_attempt(); training._save(root/"reports/reference_training.json", report)
            completed = local_now().isoformat(); production = report["production"]
            production.update(effective_from=next((d for d in calendar if d > completed[:10]), None), frozen_at=completed,
                effective_rule="first_verified_session_strictly_after_actual_freeze")
            attempt.update(status="validating", validation_started_at=completed, elapsed_seconds=round(time.monotonic()-began,3))
            report.update(status="validating", validation_started_at=completed, actual_device=plan["actual_device"],
                elapsed_seconds=round((pd.Timestamp(completed)-pd.Timestamp(report["started_at"])).total_seconds(),3),
                predictive_gate=production["predictive_gate"], qualified=production["predictive_gate"]["passed"],
                technical_validation={"status":"pending","full_population":True,"artifact_hashes":True,"chronological_purge":True},
                model_selection="inner_train_only", calibration="separate_chronological_positive_temperature",
                execution="none_reference_only", resumed=True)
            history = report.setdefault("resume_attempts", [])
            history.append(dict(attempt_id=attempt_id, adapter_sha256=attempt["adapter_sha256"], started_at=started,
                validation_started_at=completed, status="validating", reused_folds=[x["name"] for x in attempt["reused_folds"]],
                trained_folds=attempt["trained_folds"]))
            _verify_immutable(immutable); save_attempt(); training._save(root/"reports/reference_training.json", report)
            training.evaluate(run_id, runs_root=runs, metadata_root=meta, check_cancel=check_cancel)
            _verify_immutable(immutable)
            completed = local_now().isoformat()
            attempt.update(status="completed", completed_at=completed, elapsed_seconds=round(time.monotonic()-began,3))
            history[-1].update(status="completed", completed_at=completed)
            report.update(status="completed", completed_at=completed,
                elapsed_seconds=round((pd.Timestamp(completed)-pd.Timestamp(report["started_at"])).total_seconds(),3))
            report["technical_validation"]["status"] = "passed"
            save_attempt(); training._save(root/"reports/reference_training.json", report)
            event = dict(run_id=run_id, family=FAMILY, effective_from=production["effective_from"], selected_at=completed,
                predictive_gate=production["predictive_gate"], publication_allowed=False)
            meta.mkdir(parents=True, exist_ok=True)
            with (meta/"reference_selection_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")
            training._save(meta/"reference_current.json", {**event,"report_sha256":training._sha(root/"reports/reference_training.json")})
            emit("completed", len(plan["windows"]), len(plan["windows"]), run_id=run_id, predictive_gate=production["predictive_gate"])
            return dict(run_id=run_id, train_run_id=run_id, model_run_id=run_id, status="completed", report=report,
                training_resumed=True, resume_attempt_id=attempt_id, original_plan_id=plan["plan_id"],
                dispatch_plan_id=plan_id, reference_only=True, publication_allowed=False)
        except BaseException as exc:
            status = "cancelled" if "cancel" in type(exc).__name__.lower() else "failed"
            attempt.update(status=status, ended_at=local_now().isoformat(), error=f"{type(exc).__name__}: {exc}"); save_attempt()
            for entry in report.get("resume_attempts", []):
                if entry.get("attempt_id") == attempt_id: entry.update(status=status, error=attempt["error"])
            report.setdefault("technical_validation", {})["status"] = "failed" if status == "failed" else "pending"
            report.update(status=status, ended_at=attempt["ended_at"], error=attempt["error"], last_resume_attempt_id=attempt_id)
            training._save(root/"reports/reference_training.json", report)
            raise
