"""Scan progress reports real failures without changing callback/error semantics."""
from concurrent.futures import Future
from contextlib import closing
from itertools import accumulate
from types import SimpleNamespace
import json
import sqlite3
import threading
import time

import pandas as pd
import pytest

from my_strategy.web_dashboard import trend_strategy


END = "2026-10-14"


class ImmediatePool:
    """Run the real parallel orchestration against deterministic completed futures."""
    instances = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.closed = None
        self.instances.append(self)

    def submit(self, fn, payload):
        future = Future()
        try:
            future.set_result(fn(payload))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        self.closed = kwargs


@pytest.fixture
def batch_inputs(monkeypatch):
    from my_strategy.adapters import czsc_adapter
    from my_strategy.services import czsc_trend_runtime

    monkeypatch.setattr(czsc_adapter, "latest_market_date", lambda: END)
    monkeypatch.setattr(trend_strategy, "ProcessPoolExecutor", ImmediatePool)
    ImmediatePool.instances.clear()

    def prepare(payload):
        symbol = payload[0]
        if symbol == "missing":
            raise ValueError(f"行情缺少扫描日 {END}，标的最后日期为 2026-09-03")
        return symbol, {"features": symbol}, "source", {}

    def infer(features, *args, **kwargs):
        if features == "bad-inference":
            raise RuntimeError("invalid prepared input")
        return {"latest_decision": {"action": "WAIT"}}

    monkeypatch.setattr(trend_strategy, "_stock_worker", prepare)
    monkeypatch.setattr(czsc_trend_runtime, "reference_overlay", infer)
    return ["good", "missing", "bad-inference", "good-again"]


@pytest.mark.parametrize("workers", [1, 2])
def test_batch_progress_matches_yielded_preparation_and_inference_failures(batch_inputs, workers):
    events = []

    def progress(stage, current, total, failed=0, **details):
        events.append((stage, current, total, failed))

    results = list(trend_strategy._batch(batch_inputs, END, [END], {}, workers, progress=progress))
    assert len(results) == len(batch_inputs)
    failures = [failure for value, failure in results if failure]
    assert {failure["symbol"] for failure in failures} == {"missing", "bad-inference"}
    assert [event[1] for event in events] == [1, 2, 3, 4]
    assert all(event[2] == 4 for event in events)
    assert [event[3] for event in events] == list(accumulate(int(bool(failure)) for _, failure in results))
    assert events[-1][3] == 2
    if workers > 1:
        assert ImmediatePool.instances[0].closed == {"wait": True, "cancel_futures": True}


@pytest.mark.parametrize("variadic", [False, True])
def test_legacy_three_argument_callbacks_keep_their_contract(batch_inputs, variadic):
    events = []
    if variadic:
        callback = lambda *args: events.append(args)
    else:
        callback = lambda stage, current, total: events.append((stage, current, total))
    list(trend_strategy._batch(batch_inputs, END, [END], {}, 1, progress=callback))
    assert [event[1] for event in events] == [1, 2, 3, 4]
    assert all(len(event) == 3 for event in events)


def test_positional_only_failure_callback_receives_real_count(batch_inputs):
    events = []

    def callback(stage, current, total, failed, /):
        events.append(failed)

    list(trend_strategy._batch(batch_inputs, END, [END], {}, 1, progress=callback))
    assert events == [0, 1, 2, 2]


def test_cli_progress_displays_the_actual_cumulative_failure_count(batch_inputs, capsys):
    from my_strategy.cli import _progress
    list(trend_strategy._batch(batch_inputs, END, [END], {}, 1, progress=_progress))
    assert capsys.readouterr().err.splitlines()[-1] == "多源参考分析 4/4 failures=2"


@pytest.mark.parametrize("workers", [1, 2])
def test_callback_type_error_propagates_once_without_becoming_a_stock_failure(batch_inputs, workers):
    calls = []

    def callback(stage, current, total, failed=0):
        calls.append((current, failed))
        raise TypeError("progress storage failed")

    with pytest.raises(TypeError, match="progress storage failed"):
        list(trend_strategy._batch(["good", "good-again"], END, [END], {}, workers, progress=callback))
    assert len(calls) == 1
    if workers > 1:
        assert ImmediatePool.instances[0].closed == {"wait": True, "cancel_futures": True}


def test_terminal_task_and_persisted_scan_keep_the_actual_failed_denominator(tmp_path, monkeypatch):
    from my_strategy.adapters import czsc_adapter
    from my_strategy.services import czsc_trend_runtime, czsc_trend_support, czsc_market_history
    from my_strategy.storage import czsc_results
    from my_strategy.storage.czsc_market_history import market_history_facts
    from my_strategy.web_dashboard.tasks import TaskManager

    symbols = ["000001.SZ", "000016.SZ"]
    store = czsc_results.ResultStore(tmp_path / "results.db", tmp_path / "runs")
    context = SimpleNamespace(run_id="scan-failures", to_dict=lambda: {"run_id": "scan-failures"},
        subdir=lambda name: tmp_path / "runs/scan-failures" / name)
    monkeypatch.setattr(czsc_adapter, "latest_market_date", lambda end=None: END)
    monkeypatch.setattr(czsc_adapter, "list_symbols", lambda: [{"symbol": s} for s in symbols])
    monkeypatch.setattr(czsc_trend_support, "_calendar", lambda _: ([END], "calendar"))
    monkeypatch.setattr(czsc_results, "ResultStore", lambda: store)
    monkeypatch.setattr(trend_strategy, "create_run_context", lambda **kwargs: context)
    monkeypatch.setattr(czsc_market_history, "load_classification", lambda selected: ({}, {"mode": "unavailable", "note": "fixture"}))

    def prepare(payload):
        symbol = payload[0]
        if symbol == symbols[1]:
            raise ValueError(f"行情缺少扫描日 {END}，标的最后日期为 2026-09-03")
        return symbol, {"features": pd.DataFrame({"date": [END]}), "data_version": "raw-v1", "data_end": END}, "source", {}

    def infer(features, *args, **kwargs):
        row = {"date": END, "action": "WAIT", "direction": "UNAVAILABLE", "agents": {}}
        return {"latest_decision": row, "rows": [row], "features": features}

    monkeypatch.setattr(trend_strategy, "_stock_worker", prepare)
    monkeypatch.setattr(czsc_trend_runtime, "reference_overlay", infer)
    manager = TaskManager(tmp_path / "tasks.db", store, workers=1)
    try:
        job = manager.submit("scan", {"symbols": symbols, "end": END, "research": False, "cpu_workers": 1})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            final = manager.get(job["job_id"])
            if final["status"] not in {"queued", "running", "cancelling"}:
                break
            threading.Event().wait(.01)
        assert final["status"] == "succeeded", final["error"]
        assert final["progress_detail"] == {"stage": "多源参考分析", "current": 2, "total": 2,
            "failed": 1, "progress_scope": "task"}
        assert final["result"]["coverage"] == {"requested": 2, "successful": 1, "success": 1, "failed": 1}
        assert final["result"]["failures"][0]["symbol"] == symbols[1]
        assert "2026-09-03" in final["result"]["failures"][0]["error"]
        assert store.list("scan")[0]["summary"]["coverage"]["failed"] == 1
        facts = market_history_facts(context.run_id, db_path=store.db_path, date=END, state="failed")
        assert facts["total"] == 1 and facts["rows"][0]["symbol"] == symbols[1]
        assert final["result"]["market_history"]["daily"][0]["coverage"]["failed"] == 1
    finally:
        manager.close()


def test_completed_legacy_task_reads_final_counts_without_rewriting_saved_task(tmp_path, monkeypatch):
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.web_dashboard.tasks import TaskManager

    store = ResultStore(tmp_path / "results.db", tmp_path / "runs")
    result = {"run_id": "legacy-scan", "coverage": {"requested": 5572, "successful": 5561, "success": 5561, "failed": 11},
        "rows": [], "failures": [{"symbol": "000016.SZ", "error": "missing target session"}]}
    store.save("scan", result)
    # More recent CLI reports must not hide the old task's small indexed summary.
    with closing(sqlite3.connect(store.db_path)) as conn:
        conn.executemany("INSERT INTO czsc_runs VALUES (?,?,?,?,?,?)", [
            (f"newer-{n}", "scan", "", "2099-01-01", json.dumps({"coverage": {"requested": 1, "success": 1, "failed": 0}}), "unread.json")
            for n in range(205)])
        conn.commit()
    manager = TaskManager(tmp_path / "tasks.db", store, workers=1)
    try:
        saved = {"job_id": "legacy-task", "run_id": "legacy-scan", "kind": "scan", "spec": {}, "status": "succeeded",
            "progress": 100, "result_saved": True, "error": None,
            "progress_detail": {"stage": "多源参考分析", "current": 5572, "total": 5572, "failed": 0, "progress_scope": "task"}}
        manager._write(saved)
        def payload():
            with closing(sqlite3.connect(manager.db_path)) as conn:
                return conn.execute("SELECT payload FROM czsc_tasks WHERE job_id=?", [saved["job_id"]]).fetchone()[0]
        before = payload()
        actual = manager.get(saved["job_id"])
        assert actual["progress_detail"]["failed"] == actual["result"]["coverage"]["failed"] == 11
        monkeypatch.setattr(store, "get", lambda _: pytest.fail("task list must not read full scan reports"))
        listed = manager.list()
        assert listed[0]["progress_detail"]["failed"] == 11
        assert listed[0]["result"] is None
        assert listed[0]["progress_detail"]["stage"] == saved["progress_detail"]["stage"]
        assert payload() == before
    finally:
        manager.close()


@pytest.mark.parametrize("coverage", [None, {"requested": 2, "failed": 3}, {"requested": 2, "success": 2, "failed": 1},
    {"requested": 2, "failed": True}, {"requested": 2, "failed": "1"}])
def test_invalid_or_missing_completed_coverage_never_fabricates_progress(coverage):
    from my_strategy.web_dashboard.tasks import _project_completed_scan_progress
    job = {"kind": "scan", "status": "succeeded", "progress_detail": {"failed": 7}}
    assert _project_completed_scan_progress(job, coverage) is job


@pytest.mark.parametrize("kind,status", [("scan", "running"), ("scan", "failed"), ("update", "succeeded")])
def test_final_count_projection_never_changes_running_or_other_task_progress(kind, status):
    from my_strategy.web_dashboard.tasks import _project_completed_scan_progress
    job = {"kind": kind, "status": status, "progress_detail": {"failed": 0}}
    assert _project_completed_scan_progress(job, {"requested": 2, "success": 1, "failed": 1}) is job
