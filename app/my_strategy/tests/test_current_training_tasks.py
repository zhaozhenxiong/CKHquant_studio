"""Current task progress, cancellation, terminal evidence and HTTP contracts."""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.app import create_app
from my_strategy.web_dashboard.tasks import TaskManager, Cancelled


@pytest.fixture
def manager(tmp_path):
    instance = TaskManager(tmp_path/"metadata/tasks.db",ResultStore(tmp_path/"results.db",tmp_path/"runs"),workers=1)
    yield instance
    instance.close()


def terminal(manager, job_id):
    deadline = time.monotonic()+5
    while time.monotonic()<deadline:
        result=manager.get(job_id)
        if result["status"] not in {"queued","running","cancelling"}: return result
        time.sleep(.01)
    raise AssertionError("current task did not finish")


@pytest.mark.parametrize("status",["cancelled","failed"])
def test_training_terminal_metadata_retains_frozen_source_and_no_success_registration(manager,monkeypatch,status):
    def run(job_id,kind,spec):
        path=manager.results.runs_root/"terminal-current"/"metadata.json"
        path.parent.mkdir(parents=True); path.write_text(json.dumps(dict(status="running",data_version="frozen-input")))
        manager._patch(job_id,run_id="terminal-current")
        if status=="cancelled": raise Cancelled("cancelled during native preparation")
        raise ValueError("frozen source hash differs")
    monkeypatch.setattr(manager,"_run",run)
    job=manager.submit("model_training",dict(plan_id="frozen-plan"))
    final=terminal(manager,job["job_id"])
    manager._executor.shutdown(wait=True)
    metadata=json.loads((manager.results.runs_root/"terminal-current/metadata.json").read_text())
    assert final["status"]==metadata["status"]==status
    assert metadata["error"]==final["error"] and metadata["finished_at"]
    assert metadata["data_version"]=="frozen-input"
    assert manager.results.list("model_training")==[]


def test_training_progress_keeps_stage_counts_without_fake_overall_percentage(manager,monkeypatch):
    seen=[]
    def train(spec,progress,check_cancel,run_callback):
        for stage,current,total in (("raw",2,2),("trend",1,30),("entry",0,1)):
            progress(stage,current,total,message="actual stage")
            seen.append(manager.list()[0])
        return dict(run_id="current-train-progress",status="completed_unqualified",selection=dict(status="blocked"))
    monkeypatch.setattr(manager.training,"run_training_job",train)
    job=manager.submit("model_training",dict(plan_id="frozen-plan"))
    final=terminal(manager,job["job_id"])
    assert final["status"]=="succeeded",final["error"]
    assert [row["progress"] for row in seen]==pytest.approx([95,95/30,0])
    assert [(row["progress_detail"]["current"],row["progress_detail"]["total"]) for row in seen]==[(2,2),(1,30),(0,1)]
    assert all(row["progress_detail"]["progress_scope"]=="current_stage" for row in seen)
    assert final["result"]["selection"]["status"]=="blocked"


def test_training_rejects_fold_accumulated_progress_beyond_actual_stage_total(manager,monkeypatch):
    def train(spec,progress,**kwargs):
        progress("actual_stock_stage",3195,3194)
        pytest.fail("overflow stage progress passed validation")
    monkeypatch.setattr(manager.training,"run_training_job",train)
    job=manager.submit("model_training",dict(plan_id="frozen-plan"))
    final=terminal(manager,job["job_id"])
    assert final["status"]=="failed" and "阶段进度超出实际总数" in final["error"]
    assert final["result"] is None and manager.results.list("model_training")==[]


def test_current_scan_cancel_at_stage_boundary_stops_before_inference(manager,monkeypatch):
    from my_strategy.web_dashboard import trend_strategy
    entered,released=threading.Event(),threading.Event()
    def scan(**kwargs):
        entered.set(); assert released.wait(3)
        kwargs["progress"]("prepare",1,2)
        pytest.fail("cancelled current scan reached inference")
    monkeypatch.setattr(trend_strategy,"scan_trend_consensus",scan)
    job=manager.submit("scan",dict(research=True,device="cpu",symbols=["600000.SH"]))
    assert entered.wait(3)
    manager.cancel(job["job_id"]); released.set()
    final=terminal(manager,job["job_id"])
    assert final["status"]=="cancelled" and final["result"] is None
    assert manager.results.list("scan")==[]


def test_current_scan_progress_is_monotonic_and_compute_records_actual_work(manager,monkeypatch):
    from my_strategy.web_dashboard import trend_strategy
    updates=[]; old_patch=manager._patch
    def capture(job_id,**kwargs):
        if "progress" in kwargs: updates.append(kwargs["progress"])
        return old_patch(job_id,**kwargs)
    monkeypatch.setattr(manager,"_patch",capture)
    actual=dict(research=True,selected_device="cpu",model_inference_rows=4,model_inference_batches=1,cuda_work=False)
    def scan(**kwargs):
        assert kwargs["model_run_id"]=="current-model" and kwargs["cpu_workers"]==5
        for stage,count in (("prepare",2),("infer",2),("prepare",1),("scan",4)):
            kwargs["progress"](stage,count,4)
        return dict(run_id="current-scan-progress",compute_info=actual,rows=[],failures=[])
    monkeypatch.setattr(trend_strategy,"scan_trend_consensus",scan)
    job=manager.submit("scan",dict(research=True,symbols=["600000.SH"],cpu_workers=5,device="cpu",model_run_id="current-model"))
    final=terminal(manager,job["job_id"])
    assert final["status"]=="succeeded",final["error"]
    assert updates==sorted(updates)
    assert final["compute_info"]==final["result"]["compute_info"]==actual


def test_current_checkpoint_dispatch_and_http_identity_contract(manager,monkeypatch):
    from my_strategy.web_dashboard import trend_strategy
    monkeypatch.setattr(trend_strategy,"scan_trend_consensus",lambda **kwargs:kwargs)
    symbols=["600000.SH"]  # Explicit synthetic scope; no local securities catalog is required.
    result=manager._run("fixture","scan",dict(research=True,symbols=symbols,model_run_id="current-model",model_fold="trend_2026Q3",device="cpu"))
    assert result["model_run_id"]=="current-model" and result["model_fold"]=="trend_2026Q3"
    assert result["symbols"]==symbols
    assert result["structure_only"] is False
    with pytest.raises(ValueError,match="任务类型不受支持"):
        manager.submit("backtest",{})
    with pytest.raises(ValueError,match="任务类型不受支持"):
        manager._run("fixture","backtest",{})
    monkeypatch.setattr(manager,"submit",lambda kind,spec:dict(kind=kind,spec=spec))
    with TestClient(create_app(manager)) as client:
        assert client.get("/api/research/status").status_code==410
        assert client.get("/api/training/status").json()["active_model"] is None
        response=client.post("/api/tasks",json=dict(kind="scan",spec=dict(research=True,symbols=symbols,model_family="current")))
        assert response.status_code==202
        assert client.post("/api/tasks",json=dict(kind="backtest")).status_code==422
        for field in ("model_run_id","calendar_run_id"):
            for value in ("../escape","D:\\secret",".","", "a"*181):
                assert client.post("/api/tasks",json=dict(kind="scan",spec={field:value})).status_code==422
        assert client.post("/api/tasks",json=dict(kind="research_train")).status_code==422


def test_current_report_preserves_coverage_categories_and_failure(manager):
    result=dict(run_id="current-screen",model_family="current",model_gate=dict(passed=False),category_counts=dict(watch=1,excluded=1),
        coverage=dict(requested=3,success=2,failed=1),data_range=dict(end="2026-10-12"),
        rows=[dict(symbol="600000.SH",category="watch",model_probability=None),dict(symbol="600519.SH",category="excluded")],
        failures=[dict(symbol="000001.SZ",error="missing data")])
    manager.results.save("scan",result)
    assert manager.results.list()[0]["summary"]["category_counts"]==dict(watch=1,excluded=1)
    with TestClient(create_app(manager)) as client:
        assert client.get("/api/runs/current-screen").json()==result
