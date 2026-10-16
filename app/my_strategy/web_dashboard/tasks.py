"""CZSC-only persistent jobs with cancellation and exact run results."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import signal
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


def _daily_update_result(log: str) -> dict[str, Any]:
    marker = log.rfind('{\n  "summary":')
    if marker < 0:
        return {}
    try:
        result, _ = json.JSONDecoder().raw_decode(log[marker:])
        return result if isinstance(result, dict) else {}
    except ValueError:
        return {}


def _daily_update_failure(summary: dict[str, Any], returncode: int) -> str:
    quality = summary.get("quality_check") or {}
    reasons = []
    if summary.get("failed"):
        reasons.append(f"行情请求失败 {summary['failed']} 只")
    if quality.get("has_errors"):
        details = quality.get("error_details") or []
        names = ", ".join(dict.fromkeys(str(item.get("entity_id", "")) for item in details))
        reasons.append(f"数据质量检查未通过：{quality.get('errors', '?')} 项错误" + (f"（{names}）" if names else ""))
    if quality.get("error"):
        reasons.append(f"数据质量检查异常：{quality['error']}")
    return "；".join(reasons) or f"行情更新进程退出码 {returncode}，请查看完整日志"


def _project_completed_scan_progress(job: dict[str, Any], coverage: Any) -> dict[str, Any]:
    """Read final population counts for scans saved by older progress emitters."""
    if job.get("kind") != "scan" or job.get("status") not in {"succeeded", "partial"} or not isinstance(coverage, dict):
        return job
    requested, failed = coverage.get("requested"), coverage.get("failed")
    successful = coverage.get("successful", coverage.get("success"))
    if (type(requested) is not int or type(failed) is not int or not 0 <= failed <= requested
            or successful is not None and (type(successful) is not int or successful + failed != requested)):
        return job
    return {**job, "progress_detail": {**(job.get("progress_detail") or {}),
        "current": requested, "total": requested, "failed": failed}}


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
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            # An UPSERT's DO UPDATE conflict policy overrides INSERT OR REPLACE
            # in the history trigger. Use an ordinary update under one writer
            # transaction so existing display rows can be replaced atomically.
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute("UPDATE czsc_tasks SET status=?,payload=? WHERE job_id=?",
                                   (job["status"], encoded, job["job_id"])).rowcount
            if not changed:
                conn.execute("INSERT INTO czsc_tasks(job_id,fingerprint,status,payload) VALUES (?,?,?,?)",
                             (job["job_id"], stable_hash({"kind": job["kind"], "spec": job["spec"]}), job["status"], encoded))
            conn.commit()

    def get(self, job_id: str) -> dict[str, Any]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute("SELECT payload FROM czsc_tasks WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        job = json.loads(row[0])
        job["result"] = self.results.get(job["run_id"]) if job.get("result_saved") or job["status"] in {"succeeded", "partial"} else None
        return _project_completed_scan_progress(job, (job["result"] or {}).get("coverage"))

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            # UI retention is a read-only projection; exact get(job_id), audit
            # records and all business assets remain available. Active jobs and
            # legacy rows with an unknown timestamp must not silently vanish.
            rows = conn.execute("""SELECT payload FROM czsc_tasks
                WHERE status IN ('queued','running','cancelling')
                   OR julianday(COALESCE(json_extract(payload,'$.updated_at'),json_extract(payload,'$.created_at'))) IS NULL
                   OR julianday(COALESCE(json_extract(payload,'$.updated_at'),json_extract(payload,'$.created_at'))) >=
                      julianday(?) - CASE WHEN status='failed' THEN 90 ELSE 30 END
                ORDER BY status IN ('queued','running','cancelling') DESC,rowid DESC LIMIT ?""",
                (local_now().isoformat(), min(max(int(limit), 1), 1000))).fetchall()
        jobs = [{**json.loads(row[0]), "result": None} for row in rows]
        scan_runs = list(dict.fromkeys(job["run_id"] for job in jobs
            if job.get("kind") == "scan" and job.get("status") in {"succeeded", "partial"} and job.get("run_id")))
        coverage = {}
        if scan_runs:
            # One bounded index query also covers older task results outside the
            # latest scans. Never read a full market report per listed job.
            with closing(sqlite3.connect(self.results.db_path)) as conn:
                summaries = conn.execute(
                    "SELECT run_id,summary FROM czsc_runs WHERE kind='scan' AND run_id IN (" +
                    ",".join("?" for _ in scan_runs) + ")", scan_runs).fetchall()
            coverage = {run_id: json.loads(summary).get("coverage") for run_id, summary in summaries}
        return [_project_completed_scan_progress(job, coverage.get(job.get("run_id"))) for job in jobs]

    def _patch(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            job.update(updates)
            job["updated_at"] = local_now().isoformat()
            self._write(job)
            return job

    def submit(self, kind: str, spec: dict[str, Any]) -> dict[str, Any]:
        if kind not in {"scan", "update", "model_training", "model_validation", "board_features", "board_sync", "board_knowledge"}:
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
            if result.get('status') != 'cancelled': self._check(job_id)
            self.results.save(job["kind"], result)
            outcome = 'succeeded'
            if job['kind'] in {'scan', 'update', 'board_features', 'board_sync', 'board_knowledge'}:
                reported = result.get('status')
                if reported in {'partial', 'failed', 'cancelled'}: outcome = reported
                elif reported == 'interrupted': outcome = 'partial' if result.get('market_published') or result.get('completed_boards', 0) else 'failed'
            updates = {"status": outcome, "progress": 100 if outcome == 'succeeded' else self.get(job_id)['progress'],
                       "result": result, "run_id": result["run_id"], "result_saved": True, 'error': result.get('error') if outcome != 'succeeded' else None}
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
        if kind.startswith('board_'):
            return self._board(job_id, kind, spec)
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
        from my_strategy.services.analysis_methods import COMBINED
        if spec.get('analysis_method') in COMBINED:
            from my_strategy.services.method_scan_acceleration import scan_methods
            from my_strategy.services.light_research import LightResearch
            light = getattr(self, 'light', None)
            owned = light is None
            if owned: light = LightResearch(root=self.db_path.parent / 'light_research')
            try:
                return scan_methods(light, self.results, spec, **callbacks)
            finally:
                if owned: light.close()
        from my_strategy.web_dashboard.trend_strategy import scan_trend_consensus
        from my_strategy.core.market_scope import new_scan_symbols
        params = {key: spec.get(key) for key in ("symbols", "start", "end", "model_run_id", "model_fold",
            "usage_mode", "calendar_run_id", "device", "cpu_workers", "consensus_config") if spec.get(key) is not None}
        params['symbols'] = new_scan_symbols(spec.get('symbols'))
        params.setdefault('start', '2026-01-01')
        return scan_trend_consensus(**params, structure_only=not spec.get('research', True), **callbacks)

    def _board(self, job_id, kind, spec):
        context = create_run_context(task=kind, config=spec, scope='daily_update' if kind == 'board_sync' else 'batch', source='web')
        self._patch(job_id, run_id=context.run_id)
        def progress(event):
            self._check(job_id)
            if not isinstance(event, dict): return
            current = event.get('current', event.get('completed', 0)) or 0
            total = event.get('total', 0) or 0
            self._patch(job_id, progress=min(95, current / total * 95) if total else 0,
                progress_detail={**{key: event[key] for key in ('stage', 'status', 'failed', 'cache_hits', 'sync_id', 'knowledge_run_id',
                    'workers', 'requested_workers', 'min_interval', 'fresh_boards', 'reused_boards', 'request_count', 'elapsed_seconds') if key in event},
                    'stage': event.get('stage', kind), 'current': current, 'total': total, 'progress_scope': 'current_stage'})
        if kind == 'board_knowledge':
            from my_strategy.scripts.refresh_board_knowledge import refresh_knowledge
            from my_strategy.web_dashboard.board_api import _refresh_summary
            from my_strategy.services.board_fluent import source_witness
            from my_strategy.storage.board_fluent import BoardFluentStore
            store = BoardFluentStore()
            identity = {'kind': kind, 'spec': spec, 'source_witness': source_witness()}
            claim = store.acquire_sync(stable_hash(identity), identity, parent_update_id=job_id)
            if not claim['owned']:
                return {**(claim.get('result') or {}), 'status': 'partial', 'sync_id': claim['sync_id'],
                    'error': '已有板块同步持有数据库，请等待该任务完成', 'run_id': context.run_id,
                    'training_started': False}
            outcome = 'failed'
            result = None
            def refresh_progress(event):
                store.heartbeat_sync(claim['sync_id'], stage='board_knowledge',
                    checkpoint={'knowledge_run_id': event['run_id']} if event.get('run_id') else None)
                if event.get('run_id'): self._patch(job_id, knowledge_run_id=event['run_id'])
                progress({**event, 'stage': 'board_knowledge', 'sync_id': claim['sync_id'],
                    'knowledge_run_id': event.get('run_id')})
            try:
                result = _refresh_summary(refresh_knowledge(kinds=spec.get('kinds'), boards=spec.get('boards') or None,
                    resume=spec.get('resume'), limit_budget=spec.get('limit_budget', 2000),
                    min_interval=spec.get('min_interval', 1.3), workers=spec.get('workers', 4),
                    reuse_today=spec.get('reuse_today', True), on_progress=refresh_progress,
                    check_cancel=lambda: self._check(job_id)))
                result['knowledge_run_id'] = result.pop('run_id', None)
                result['sync_id'] = claim['sync_id']
                outcome = 'completed' if result.get('status') in {'completed', 'succeeded'} else 'failed' if result.get('status') == 'failed' else 'partial'
            except Cancelled:
                outcome = 'cancelled'
                # refresh_knowledge has drained its bounded workers before it
                # propagates cancellation. Seal the actual committed successes
                # now; a later explicit retry creates a new child manifest.
                from my_strategy.storage.board_knowledge import BoardKnowledgeStore
                knowledge_id = self.get(job_id).get('knowledge_run_id')
                if knowledge_id:
                    knowledge = BoardKnowledgeStore()
                    state = knowledge.get_refresh(knowledge_id)
                    if state and state['status'] == 'running': knowledge.finish_refresh(knowledge_id)
                raise
            finally:
                store.finish_sync(claim['sync_id'], status=outcome, result=result)
        elif kind == 'board_features':
            from my_strategy.services.board_fluent import build_board_fluent
            result = build_board_fluent(price_date=spec.get('price_date'), with_native=spec.get('with_native', True),
                source_root=PROJECT_ROOT, progress=progress, check_cancel=lambda: self._check(job_id))
        elif kind == 'board_sync':
            from my_strategy.services.daily_feature_sync import sync_daily_features, SyncCancelled
            try:
                result = sync_daily_features(sync_id=spec.get('resume_sync_id'), source_root=PROJECT_ROOT,
                    cpu_workers=spec.get('cpu_workers', 1), stock_batch_size=spec.get('stock_batch_size', 360),
                    progress=progress, check_cancel=lambda: self._check(job_id))
            except SyncCancelled as exc:
                raise Cancelled(str(exc)) from exc
        else: raise ValueError('任务类型不受支持')
        if result.get('status') == 'busy' or result.get('reused_sync') and result.get('status') in {'running', 'queued'}:
            result = {**result, 'status': 'partial', 'error': result.get('error') or '已有同步任务持有数据库，请等待当前任务完成'}
        return {**result, 'kind': kind, 'run_id': context.run_id, 'run_context': context.to_dict(), 'training_started': False}

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
        offset, pending, failed, skipped = 0, "", 0, 0
        with log_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=output, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                while process.poll() is None:
                    if self.get(job_id)["status"] == "cancelling":
                        self._cancel[job_id].set()
                    if self._cancel[job_id].wait(.25):
                        # SIGINT lets derived workflow persist checkpoints and reap native children.
                        process.send_signal(signal.SIGINT if os.name != 'nt' else signal.SIGTERM)
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            try: children = psutil.Process(process.pid).children(recursive=True)
                            except psutil.Error: children = []
                            for child in children:
                                try: child.kill()
                                except psutil.Error: pass
                            process.kill(); process.wait()
                            psutil.wait_procs(children, timeout=5)
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
                        if not isinstance(event, dict):
                            continue
                        if event.get('event') == 'stage':
                            current = self.get(job_id).get('progress_detail') or {}
                            self._patch(job_id, progress_detail={**current, 'stage': event.get('message') or event.get('stage')})
                        elif event.get('event') == 'feature_sync':
                            current, total = event.get('current', event.get('completed', 0)) or 0, event.get('total', 0) or 0
                            self._patch(job_id, progress=min(95, current / total * 95) if total else 0,
                                progress_detail={**{key: event[key] for key in ('stage', 'status', 'sync_id', 'failed') if key in event},
                                    'stage': event.get('stage', '同步特征'), 'current': current, 'total': total, 'progress_scope': 'current_stage'})
                        elif event.get("event") == "progress":
                            current, total = event.get("current", 0), event.get("total", 0)
                            failed += int(event.get("row", {}).get("status") in {"failed", "error"})
                            skipped += int(event.get("row", {}).get("status") == "skipped")
                            self._patch(job_id, progress=current / total * 95 if total else 0, progress_detail={"stage": "更新行情", "current": current, "total": total, "failed": failed, "not_updated": skipped})
                if process.returncode:
                    summary = (_daily_update_result(log_path.read_text(encoding="utf-8", errors="replace")).get("summary") or {})
                    if not summary:
                        raise RuntimeError(f"行情更新进程退出码 {process.returncode}；" + log_path.read_text(encoding="utf-8", errors="replace")[-1500:])
                    return {"run_id": context.run_id, "run_context": context.to_dict(), "status": "failed",
                            "update_summary": summary, "feature_sync": summary.get("feature_sync") or {},
                            "error": _daily_update_failure(summary, process.returncode), "log_path": str(log_path)}
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        status = data_status()
        log = log_path.read_text(encoding="utf-8", errors="replace")
        update_result = _daily_update_result(log)
        summary = update_result.get('summary') or {}
        derived = summary.get('feature_sync') or update_result.get('feature_sync') or {}
        universe = summary.get('update_universe') or {}
        derived_pending = (derived.get('status') in {'partial', 'failed', 'interrupted', 'cancelled', 'blocked', 'busy', 'queued', 'running'}
            or derived.get('reason_code') == 'target_date_not_covered')
        raw_status = summary.get('raw_status', summary.get('status'))
        stage = ("行情已提交，结构与特征同步未完成" if derived_pending else
            "行情与特征更新完成" if raw_status == "completed" and derived.get('status') == 'completed' else
            "行情处理完成" if raw_status == "completed" else "行情处理结束，部分股票未更新")
        self._patch(job_id, progress_detail={"stage": stage,
            "current": summary.get("total", 0), "total": summary.get("total", 0), "failed": summary.get("failed", 0),
            "not_updated": summary.get("not_updated", summary.get("skipped", 0)), "excluded": universe.get("excluded"),
            "target_date_covered": summary.get("target_date_covered"), "feature_sync_status": derived.get('status'),
            "sync_id": derived.get('sync_id')})
        return {"run_id": context.run_id, "run_context": context.to_dict(), "data_status": status,
            "update_summary": summary, "feature_sync": derived,
            "status": 'partial' if summary.get('status') in {'partial', 'no_updates', 'cancelled'} or derived_pending else 'completed',
            "error": derived.get('error') or (derived.get('reason') if derived_pending else None), "log": log[-6000:]}

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
