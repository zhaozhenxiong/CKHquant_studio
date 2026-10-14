"""CZSC-only persistent jobs with cancellation and exact run results."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

import pandas as pd
import psutil

from my_strategy.core.paths import METADATA_ROOT, PROJECT_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.storage.czsc_results import ResultStore

ACTIVE = ("queued", "running", "cancelling")


class Cancelled(RuntimeError):
    pass


class TaskManager:
    def __init__(self, db_path: Path | None = None, results: ResultStore | None = None, workers: int = 2) -> None:
        self.db_path = db_path or METADATA_ROOT / "czsc_tasks.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.results = results or ResultStore()
        from my_strategy.services.model_training_center import TrainingCenter
        self.training = TrainingCenter(store=self.results, metadata_root=self.db_path.parent)
        self._lock = threading.RLock()
        self._cancel: dict[str, threading.Event] = {}
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="czsc-job")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS czsc_tasks (job_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL)")
            conn.commit()
        with closing(sqlite3.connect(self.db_path)) as conn:
            active = conn.execute("SELECT payload FROM czsc_tasks WHERE status IN ('queued','running','cancelling')").fetchall()
        self.training.selection_guard = self._active_selection_reasons
        for payload, in active:
            job = json.loads(payload)
            try:
                owner = psutil.Process(job.get("owner_pid", -1))
                live = abs(owner.create_time() - job.get("owner_started", 0)) < .001
            except (psutil.Error, ValueError):
                live = False
            if not live:
                self._patch(job["job_id"], status="failed", error="服务重启中断了任务，请重新提交。")

    def _active_selection_reasons(self):
        """A running reference scan must finish under its frozen model selection."""
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            rows = conn.execute("SELECT payload FROM czsc_tasks WHERE status IN ('queued','running','cancelling')").fetchall()
        jobs = [json.loads(row[0]) for row in rows]
        active = [job for job in jobs if job.get('kind') == 'scan']
        return [f"参考扫描任务 {job['job_id']} 正在运行，完成或取消后才能切换模型" for job in active]

    def _write(self, job: dict[str, Any]) -> None:
        with closing(sqlite3.connect(self.db_path, timeout=30)) as conn:
            payload = {key: value for key, value in job.items() if key != "result"}
            conn.execute("INSERT INTO czsc_tasks VALUES (?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,payload=excluded.payload", (job["job_id"], stable_hash({"kind": job["kind"], "spec": job["spec"]}), job["status"], json.dumps(payload, ensure_ascii=False, allow_nan=False)))
            conn.commit()

    def get(self, job_id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute("SELECT payload FROM czsc_tasks WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        job = json.loads(row[0])
        job["result"] = self.results.get(job["run_id"]) if job["status"] == "succeeded" else None
        return job

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute("SELECT payload FROM czsc_tasks ORDER BY rowid DESC LIMIT ?", (min(max(limit, 1), 200),)).fetchall()
        return [{**json.loads(row[0]), "result": None} for row in rows]

    def _patch(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            job.update(updates)
            job["updated_at"] = local_now().isoformat()
            self._write(job)
            return job

    def submit(self, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
        if kind not in {"scan", "update", "model_training", "model_validation"}:
            raise ValueError("任务类型不受支持")
        with self._lock:
            if self._closed:
                raise RuntimeError("任务服务正在关闭")
            fingerprint = stable_hash({"kind": kind, "spec": spec})
            with closing(sqlite3.connect(self.db_path)) as conn:
                row = conn.execute("SELECT payload FROM czsc_tasks WHERE fingerprint=? AND status IN ('queued','running','cancelling')", (fingerprint,)).fetchone()
            if row:
                return json.loads(row[0])
            job_id = uuid.uuid4().hex
            job = dict(job_id=job_id, kind=kind, spec=spec, status="queued", progress=0, progress_detail={"stage": "等待", "current": 0, "total": 0, "failed": 0}, error=None, result=None, run_id=None, created_at=local_now().isoformat())
            job.update(owner_pid=os.getpid(), owner_started=psutil.Process().create_time())
            self._write(job)
            self._cancel[job_id] = threading.Event()
            self._executor.submit(self._execute, job_id)
            return job

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            if job["status"] not in ACTIVE:
                return job
            event = self._cancel.get(job_id)
            if event:
                event.set()
            return self._patch(job_id, status="cancelling")

    def _check(self, job_id: str) -> None:
        if self._cancel[job_id].is_set() or self.get(job_id)["status"] == "cancelling":
            raise Cancelled("用户取消任务")

    def _execute(self, job_id: str) -> None:
        try:
            self._check(job_id)
            self._patch(job_id, status="running")
            job = self.get(job_id)
            result = self._run(job_id, job["kind"], job["spec"])
            self._check(job_id)
            self.results.save(job["kind"], result)
            updates = {"status": "succeeded", "progress": 100, "result": result, "run_id": result["run_id"]}
            if "compute_info" in result:
                updates["compute_info"] = result["compute_info"]
            self._patch(job_id, **updates)
        except Cancelled as exc:
            job = self._patch(job_id, status="cancelled", error=str(exc))
            self._finish_research_run(job)
        except Exception as exc:
            job = self._patch(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            self._finish_research_run(job)
        finally:
            with self._lock:
                self._cancel.pop(job_id, None)

    def _finish_research_run(self, job: dict[str, Any]) -> None:
        if not job.get("run_id") or not (job["kind"] in {"model_training", "model_validation"} or job["spec"].get("research")):
            return
        path = (self.results.runs_root / job["run_id"] / "metadata.json").resolve()
        if not path.is_relative_to(self.results.runs_root.resolve()):
            raise ValueError("Research metadata outside run storage")
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata.update(status=job["status"], error=job["error"], finished_at=job["updated_at"])
            temporary = path.with_suffix(f".{job['job_id']}.tmp")
            temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)

    def _run(self, job_id: str, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
        from my_strategy.adapters.czsc_adapter import latest_market_date, list_symbols
        from my_strategy.services.czsc_analysis import strategy_config
        if kind == "update":
            return self._update(job_id, spec)
        def report_progress(stage, current=0, total=0, failed=0, **details):
            self._check(job_id)
            previous = self.get(job_id).get("progress", 0)
            if current < 0 or total < 0 or (total and current > total):
                raise ValueError("阶段进度超出实际总数")
            overall = (min(95, current / total * 95) if total else 0) if kind in {"model_training", "model_validation"} else max(previous, min(95, current / total * 95) if total else previous)
            scope = "current_stage" if kind in {"model_training", "model_validation"} else "task"
            self._patch(job_id, progress=overall,
                progress_detail={"stage": stage, "current": current, "total": total,
                                 "failed": failed, "progress_scope": scope, **details})
        callbacks = {"progress": report_progress, "check_cancel": lambda: self._check(job_id),
                     "run_callback": lambda run_id: self._patch(job_id, run_id=run_id)}
        if kind == "model_training":
            return self.training.run_training_job(spec, **callbacks)
        if kind == "model_validation":
            return self.training.run_validation_job(spec, **callbacks)
        if kind != "scan":
            raise ValueError("任务类型不受支持")
        from my_strategy.web_dashboard.trend_strategy import scan_trend_consensus
        params = {key: spec.get(key) for key in ("symbols", "start", "end", "model_run_id", "model_fold",
            "usage_mode", "calendar_run_id", "device", "cpu_workers", "consensus_config") if spec.get(key) is not None}
        params['symbols'] = spec.get('symbols') or None
        params.setdefault('start', '2026-01-01')
        return scan_trend_consensus(**params, structure_only=not spec.get('research', True), **callbacks)

    def _update(self, job_id: str, spec: dict[str, Any]) -> dict[str, Any]:
        from my_strategy.adapters.czsc_adapter import data_status
        context = create_run_context(task="czsc-data-update", config=spec, scope="daily_update", source="web", as_of_date=spec.get("end"))
        log_path = context.subdir("logs") / "data-update.log"
        command = [sys.executable, "-u", str(PROJECT_ROOT / "my_strategy" / "scripts" / "update_daily_data.py"), "--mode", "local", "--update-id", context.run_id, "--json", "--no-progress-bar"]
        if spec.get("symbols"):
            command.extend(["--stocks", ",".join(spec["symbols"])])
        if spec.get("end"):
            command.extend(["--end-date", spec["end"]])
        self._patch(job_id, run_id=context.run_id, progress_detail={"stage": "更新行情", "current": 0, "total": 0, "failed": 0})
        offset, pending, failed = 0, "", 0
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=output, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                while process.poll() is None:
                    if self.get(job_id)["status"] == "cancelling":
                        self._cancel[job_id].set()
                    if self._cancel[job_id].wait(.25):
                        process.terminate()
                        process.wait(timeout=15)
                        raise Cancelled("用户取消行情更新")
                    with log_path.open(encoding="utf-8", errors="replace") as progress_log:
                        progress_log.seek(offset)
                        pending += progress_log.read()
                        offset = progress_log.tell()
                    lines = pending.split("\n")
                    pending = lines.pop()
                    for line in lines:
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event.get("event") == "progress":
                            current, total = event.get("current", 0), event.get("total", 0)
                            failed += int(event.get("row", {}).get("status") in {"failed", "error"})
                            self._patch(job_id, progress=current / total * 95 if total else 0, progress_detail={"stage": "更新行情", "current": current, "total": total, "failed": failed})
                if process.returncode:
                    raise RuntimeError(log_path.read_text(encoding="utf-8", errors="replace")[-3000:])
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        status = data_status()
        log = log_path.read_text(encoding="utf-8", errors="replace")
        marker = log.rfind('{\n  "summary":')
        update_result = json.loads(log[marker:]) if marker >= 0 else {}
        return {"run_id": context.run_id, "run_context": context.to_dict(), "data_status": status, "update_summary": update_result.get("summary"), "log": log[-6000:]}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for event in self._cancel.values():
                event.set()
        self._executor.shutdown(wait=True, cancel_futures=False)


def aggregate_compute(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep capability metadata separate from the measured work of each batch."""
    if not batches:
        return {"cuda_work": False, "mps_work": False, "gpu_work": False, "batches": 0}
    info = dict(batches[0])
    devices = list(dict.fromkeys(batch.get("selected_device") for batch in batches))
    actual_devices = list(dict.fromkeys(batch.get("actual_device", batch.get("selected_device")) for batch in batches))
    backends = list(dict.fromkeys(batch.get("backend") for batch in batches))
    reasons = list(dict.fromkeys(batch["fallback_reason"] for batch in batches if batch.get("fallback_reason")))
    info.update(batches=len(batches), cuda_work=any(batch.get("cuda_work", False) for batch in batches),
                mps_work=any(batch.get("mps_work", False) for batch in batches),
                gpu_work=any(batch.get("gpu_work", batch.get("cuda_work", False)) for batch in batches),
                actual_devices=actual_devices, actual_device=actual_devices[0] if len(actual_devices) == 1 else "mixed",
                batch_size=max(batch.get("batch_size", 0) for batch in batches), selected_devices=devices,
                selected_device=devices[0] if len(devices) == 1 else "mixed",
                backend=backends[0] if len(backends) == 1 else "mixed", fallback_reason="；".join(reasons) or None)
    for field in ("cuda_batches", "bars", "signal_elements", "tensor_seconds", "position_seconds", "preparation_wall_seconds"):
        info[field] = sum(batch.get(field) or 0 for batch in batches)
    info["tensor_operations"] = list(dict.fromkeys(operation for batch in batches for operation in batch.get("tensor_operations", [])))
    elapsed = [batch["cuda_elapsed_ms"] for batch in batches if batch.get("cuda_elapsed_ms") is not None and batch.get("cuda_work")]
    info["cuda_elapsed_ms"] = sum(elapsed) if elapsed else None
    info["stages"] = [{**stage, "device": info["actual_device"] if stage.get("stage") == "batched_signals" else stage.get("device")} for stage in info.get("stages", [])]
    info["batch_records"] = [{key: batch.get(key) for key in ("selected_device", "actual_device", "backend", "cuda_work", "mps_work", "gpu_work", "batch_size", "bars", "cuda_elapsed_ms", "fallback_reason")} for batch in batches]
    return info
