"""High-return entry points keep target identity and real-holding scope explicit."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import types

import pandas as pd
import pytest
from pydantic import ValidationError

from my_strategy.cli import build_parser
from my_strategy.web_dashboard.api import AnalysisRequest, TaskRequest


def test_high_return_commands_require_frozen_source_or_model_identity():
    for command in ("research-points-build", "research-episodes-build", "research-high-return-train"):
        args = build_parser().parse_args([command, "--source-training-run-id", "frozen-data", "--stocks", "600000.SH", "--cpu-workers", "1"])
        assert args.model_family == "high_return"
        assert args.research is True
        assert args.end is None
    args = build_parser().parse_args(["research-high-return-backtest", "--model-run-id", "high-return-run", "--start", "2025-01-01"])
    assert args.model_run_id == "high-return-run"
    assert build_parser().parse_args(["research-high-return-status"]).command == "research-high-return-status"
    audit = build_parser().parse_args(["research-high-return-audit", "--model-run-id", "high-return-run", "--as-of", "2026-10-10"])
    assert audit.model_run_id == "high-return-run" and audit.as_of == "2026-10-10"
    assert AnalysisRequest(symbol="600000.SH", model_family="high_return").model_family == "high_return"
    for kind in ("high_return_points_build", "high_return_episodes_build", "high_return_train"):
        for source in (None, "../foreign"):
            with pytest.raises(ValidationError):
                TaskRequest(kind=kind, spec={"source_training_run_id": source})
        task = TaskRequest(kind=kind, spec={"source_training_run_id": "frozen-data"})
        assert task.spec.model_family == "high_return" and task.spec.research
    with pytest.raises(ValidationError):
        TaskRequest(kind="high_return_backtest")
    with pytest.raises(ValidationError):
        TaskRequest(kind="high_return_backtest", spec={"model_run_id": "run", "usage_mode": "retrospective"})
    with pytest.raises(ValidationError, match="当前生产资格"):
        TaskRequest(kind="high_return_backtest", spec={"model_run_id": "run", "usage_mode": "production"})
    with pytest.raises(ValidationError, match="不能固定检查点"):
        TaskRequest(kind="high_return_backtest", spec={"research": True, "model_run_id": "run", "model_fold": "production"})
    with pytest.raises(ValidationError, match="不能使用事后模型"):
        AnalysisRequest(symbol="600000.SH", model_family="high_return", usage_mode="retrospective", model_run_id="run", model_fold="production")
    assert AnalysisRequest(symbol="600000.SH", model_family="high_return", usage_mode="production", entry_policy="risk").entry_policy == "risk"
    assert TaskRequest(kind="scan", spec={"research": True, "model_family": "high_return", "usage_mode": "production", "entry_policy": "risk"}).spec.research
    with pytest.raises(ValidationError, match="价格与风险入场尚为实验"):
        AnalysisRequest(symbol="600000.SH", model_family="wyckoff", usage_mode="production", entry_policy="risk")


def test_new_tasks_reach_new_research_apis_and_keep_cancellation(monkeypatch, tmp_path):
    from my_strategy.web_dashboard.tasks import TaskManager
    from my_strategy.storage.czsc_results import ResultStore
    calls = []
    module = types.ModuleType("my_strategy.services.czsc_high_return_research")
    def operation(**kwargs):
        calls.append(kwargs)
        kwargs["check_cancel"]()
        kwargs["progress"]("causal-labels", 3, 8, 1)
        kwargs["run_callback"]("new-run")
        return {"run_id": "new-run"}
    module.build_high_return_artifacts = operation
    module.train_high_return_research = operation
    module.backtest_high_return = operation
    monkeypatch.setitem(sys.modules, module.__name__, module)
    manager = TaskManager(db_path=tmp_path / "tasks.db", results=ResultStore(tmp_path / "results.db", tmp_path / "runs"), workers=1)
    patches = []
    monkeypatch.setattr(manager, "_check", lambda job: None)
    monkeypatch.setattr(manager, "_patch", lambda job, **updates: patches.append(updates))
    try:
        for kind in ("high_return_points_build", "high_return_episodes_build", "high_return_train"):
            result = manager._run("job", kind, {"source_training_run_id": "frozen-data", "symbols": ["600000.SH"], "device": "mps", "cpu_workers": 2})
            assert result["run_id"] == "new-run"
            assert calls[-1]["source_training_run_id"] == "frozen-data"
            assert calls[-1]["symbols"] == ["600000.SH"]
        assert calls[0]["stage"] == "points" and calls[1]["stage"] == "episodes"
        assert calls[2]["device"] == "mps"
        manager._run("job", "high_return_backtest", {"model_run_id": "frozen-model", "start": "2025-01-01", "device": "cpu"})
        assert calls[-1]["model_run_id"] == "frozen-model"
        assert calls[-1]["start"] == "2025-01-01"
        assert patches[-1] == {"run_id": "new-run"}
        assert patches[-2]["progress_detail"]["failed"] == 1
        with pytest.raises(ValueError, match="显式离线"):
            manager._run("job", "backtest", {"research": True, "model_family": "high_return"})
    finally:
        manager.close()


def test_audit_cli_outputs_complete_readonly_report_and_error_exit(monkeypatch, capsys):
    from my_strategy.cli import main
    audit_module = types.ModuleType("my_strategy.services.czsc_high_return_audit")
    calls = []
    result = {"passed": True, "error_count": 0, "errors": [], "pending_checks": ["future_window"],
        "formal_blocks": ["shadow_only"], "publication_allowed": False}
    def audit(identity, **kwargs):
        calls.append((identity, kwargs))
        return result
    audit_module.audit_high_return_run = audit
    monkeypatch.setitem(sys.modules, audit_module.__name__, audit_module)
    assert main(["research-high-return-audit", "--model-run-id", "run", "--as-of", "2026-10-10"]) == 0
    assert json.loads(capsys.readouterr().out) == result
    assert calls == [("run", {"as_of": "2026-10-10"})]
    result.update(passed=False, error_count=1, errors=["head_integrity_mismatch"])
    assert main(["research-high-return-audit", "--model-run-id", "run", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error_count"] == 1


def test_forward_cli_uses_explicit_new_source_and_reports_pending_status(monkeypatch, capsys):
    from my_strategy.cli import main
    from my_strategy import services
    # Full-suite collection can import the real module before this test. The
    # CLI's package import must resolve to the same fake in either order.
    from my_strategy.services import czsc_high_return_forward
    forward_module = types.ModuleType("my_strategy.services.czsc_high_return_forward")
    calls = []
    def build(model, source, **kwargs):
        calls.append((model, source, kwargs))
        kwargs["progress"]("forward", 1, 2, 0)
        return {"publication_allowed": False, "status": "pending_future_data"}
    forward_module.build_forward_evidence = build
    forward_module.forward_status = lambda identity: {"model_run_id": identity, "technical_windows": [], "statistical_status": "pending"}
    source_calls = []
    def source_snapshot(identity, **kwargs):
        source_calls.append((identity, kwargs))
        return {"run_id": "new-source", "data_end": kwargs["end"], "publication_allowed": False}
    forward_module.build_forward_source_snapshot = source_snapshot
    monkeypatch.setitem(sys.modules, forward_module.__name__, forward_module)
    monkeypatch.setattr(services, "czsc_high_return_forward", forward_module)
    args = ["research-high-return-forward", "--model-run-id", "model", "--source-training-run-id", "new-source",
            "--end", "2026-11-02", "--previous-run-id", "previous", "--cpu-workers", "2", "--json"]
    assert main(args) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "pending_future_data"
    assert "forward 1/2" in output.err
    assert calls[0][:2] == ("model", "new-source")
    assert calls[0][2]["previous_run_id"] == "previous" and calls[0][2]["cpu_workers"] == 2
    assert main(["research-high-return-forward-status", "--model-run-id", "model", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["statistical_status"] == "pending"
    assert main(["research-high-return-forward-source", "--model-run-id", "model", "--end", "2026-11-02", "--calendar-run-id", "calendar", "--cpu-workers", "2", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "new-source"
    assert source_calls[0][0] == "model"
    assert source_calls[0][1]["calendar_run_id"] == "calendar" and source_calls[0][1]["cpu_workers"] == 2
    assert main(["research-high-return-forward-source", "--model-run-id", "model", "--cpu-workers", "0"]) == 1
    assert "1到16" in capsys.readouterr().err
    assert len(source_calls) == 1


def test_forward_status_api_is_readonly_and_rejects_foreign_paths(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from my_strategy.web_dashboard.api import router
    module = types.ModuleType("my_strategy.services.czsc_high_return_forward")
    calls = []
    def status(identity):
        calls.append(identity)
        return {"model_run_id": identity, "statistical_status": "pending", "publication_allowed": False}
    module.forward_status = status
    monkeypatch.setitem(sys.modules, module.__name__, module)
    application = FastAPI(); application.include_router(router)
    with TestClient(application) as client:
        response = client.get("/api/research/high-return-forward-status", params={"model_run_id": "frozen-model"})
        assert response.status_code == 200 and response.json()["publication_allowed"] is False
        assert calls == ["frozen-model"]
        assert client.get("/api/research/high-return-forward-status", params={"model_run_id": "../foreign"}).status_code == 422
        assert calls == ["frozen-model"]


def test_analysis_adapter_preserves_null_entry_and_refuses_stale_exit_probability():
    from my_strategy.web_dashboard.high_return import _latest, _events
    item = {"symbol": "600000.SH", "data_end": "2026-10-09", "data_version": "hash", "raw": pd.DataFrame({"close": [12.0]})}
    overlay = {"latest_decision": {"model_resolution": {"model_family": "high_return", "probability": None},
                                  "exit_model": {"probability": .7, "reason": "真实状态缺失"}},
               "signal_points": [{"source": "ma", "side": "buy", "event_type": "trend_entry", "anchor_at": "2026-10-07", "available_at": "2026-10-09T15:00:00+08:00"}]}
    row = _latest(overlay, item)
    assert row["model_resolution"]["probability"] is None
    assert row["exit_model"]["probability"] is None and row["exit_probability"] is None
    event = _events(overlay, "2026-10-08")[0]
    assert event["time"] == "2026-10-09" and event["anchor_at"] == "2026-10-07"
    assert _events(overlay, "2026-10-10") == []


def test_preparation_restores_observed_bars_without_reordering_model_inputs(monkeypatch):
    from my_strategy.services import czsc_research_runtime
    from my_strategy.web_dashboard.high_return import _prepare
    raw = pd.DataFrame({"symbol": ["600000.SH"] * 2, "date": pd.to_datetime(["2026-10-08", "2026-10-09"]),
                        "close": [12.0, 12.2], "low": [11.8, 12.0], "volume": [1000, 1200]})
    features = raw[["symbol", "date"]].copy()
    features["ma20_bias"] = [.1, .12]
    features.attrs["schema_hash"] = "frozen-profile"
    monkeypatch.setattr(czsc_research_runtime, "_prepare_stock", lambda *a, **k: {"features": features, "raw": raw})
    prepared = _prepare("600000.SH", "2026-10-09", ["2026-10-08", "2026-10-09"])["features"]
    assert prepared["close"].tolist() == [12.0, 12.2]
    assert prepared["low"].tolist() == [11.8, 12.0]
    assert prepared["ma20_bias"].tolist() == [.1, .12]
    assert prepared.attrs["schema_hash"] == "frozen-profile"
    assert "close" not in features
    mismatched = raw.copy(); mismatched.loc[0, "date"] = pd.Timestamp("2026-10-07")
    monkeypatch.setattr(czsc_research_runtime, "_prepare_stock", lambda *a, **k: {"features": features, "raw": mismatched})
    with pytest.raises(ValueError, match="股票和日期不一致"):
        _prepare("600000.SH", "2026-10-09", ["2026-10-08", "2026-10-09"])


def test_ui_separates_family_target_and_holding_probability():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for actual shipped JavaScript contract")
    source = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map();
function el(id) { if (!elements.has(id)) elements.set(id,{value:'',disabled:false,hidden:false,innerHTML:'',textContent:'',options:[],dataset:{},querySelector:()=>el(id+'-child')});return elements.get(id); }
global.document={getElementById:el};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);','globalThis.ui={state,researchSpec,populateResearchModels,updateResearchControls,modelResolutionText,researchEvidence,scanActionCell,evaluationWindows,highReturnBacktestView,researchComparisonHtml,forwardSummaryText};'));
const {state,researchSpec,populateResearchModels,updateResearchControls,modelResolutionText,researchEvidence,scanActionCell,evaluationWindows,highReturnBacktestView,researchComparisonHtml,forwardSummaryText}=ui;
state.research={models:[{run_id:'old-ma'}],high_return:{models:[{run_id:'new-high-return',model_family:'high_return',checkpoints:[{name:'production',available_at:'2026-09-30'}]}]}};
el('analysis-model-family').value='high_return';el('analysis-mode').value='research';el('analysis-model-policy').value='auto';el('analysis-entry-policy').value='fresh';
populateResearchModels('analysis');assert(el('analysis-model').innerHTML.includes('new-high-return'));assert(!el('analysis-model').innerHTML.includes('old-ma'));
assert.equal(researchSpec('analysis').entry_policy,'risk');updateResearchControls('analysis');assert.equal(el('analysis-entry-policy').value,'risk');assert.equal(el('analysis-entry-policy').disabled,true);el('analysis-usage-mode').value='production';assert.equal(researchSpec('analysis').usage_mode,'production');
const resolution={model_family:'high_return',status:'historical_shadow',model_input_eligible:true,probability:.68,applied_to_entry:false};
const row={model_family:'high_return',category:'buy',candidate_action:'BUY',model_resolution:resolution,candidate_sources:['ma','wyckoff'],exit_model:{probability:.32},signal_points:[{source:'ma',event_type:'entry',available_at:'2026-10-09',anchor_at:'2026-10-07'}]};
const text=modelResolutionText(resolution,row);assert(text.includes('高收益区间研究概率 68%'));assert(text.includes('缺少兼容真实持仓'));assert(!text.includes('32%'));assert(!text.includes('10交易日'));
const html=researchEvidence(row)+scanActionCell(row);assert(html.includes('均线ML、威科夫量价'));assert(html.includes('确认可用'));assert(html.includes('高收益区间研究概率 68%'));assert(!html.includes('10日研究概率'));
assert(!modelResolutionText({...resolution,probability:null},row).includes('68%'));
const real={...row,holding_snapshot:{shares:100},exit_model:{probability:.32,holding_contract_compatible:true}};
assert(modelResolutionText(resolution,real).includes('退出优于冻结续持政策的概率 32%'));
el('analysis-usage-mode').value='retrospective';assert.throws(()=>researchSpec('analysis'),/不能使用事后模型/);updateResearchControls('analysis');assert.equal(el('analysis-usage-mode').value,'historical');
el('initial-cash').value='200000';el('backtest-model-family').value='high_return';el('backtest-mode').value='research';el('backtest-model-policy').value='auto';el('backtest-entry-policy').value='fresh';updateResearchControls('backtest');assert.equal(el('backtest-model').disabled,false);assert.equal(el('initial-cash').disabled,true);assert.equal(el('initial-cash').value,'100000');assert(el('initial-cash-label').textContent.includes('每只股票'));
assert.throws(()=>researchSpec('backtest'),/明确选择/);el('backtest-model').value='new-high-return';assert.equal(researchSpec('backtest').model_run_id,'new-high-return');assert.equal(researchSpec('backtest').model_fold,undefined);
el('backtest-model-family').value='wyckoff';updateResearchControls('backtest');assert.equal(el('initial-cash').disabled,false);assert.equal(el('initial-cash').value,'200000');assert.equal(el('backtest-entry-policy').value,'fresh');
const pending=forwardSummaryText({status:'pending_future_data',append_registry_count:0,mature_technical_windows:0,statistical_status:'pending_or_failed',observed_end:'2026-10-08'});assert(pending.includes('冻结后核验行情不足'));assert(pending.includes('统计检查待完成或未通过'));assert(!pending.includes('技术统计检查通过'));
const invalid=forwardSummaryText({status:'pending_artifact_validation',append_registry_count:0,mature_technical_windows:0,statistical_status:'pending_or_failed'});assert(invalid.includes('冻结证据尚未通过校验'));assert(!invalid.includes('连续账户影子观察'));
const report={model_family:'high_return',evaluation:{windows:{evaluation_1:{fold:{name:'evaluation_1'},initial_cash:300000,variants:{full:{metrics:{total_return:.12,max_drawdown:.04,completed_round_trips:4,high_return_share:.5,entry_model_score_coverage:.8,exit_model_score_coverage:.7},daily:[{date:'2026-01-02',equity:300000}]}}}}}};
assert.equal(evaluationWindows(report).length,1);const view=highReturnBacktestView(report);assert.equal(view.metrics.total_return,.12);assert.equal(view.daily.length,1);assert.equal(view.initial_cash,300000);const comparison=researchComparisonHtml(report);assert(comparison.includes('高收益入场＋退出'));assert(comparison.includes('12%'));assert(comparison.includes('80%'));assert(comparison.includes('70%'));
"""
    result = subprocess.run([node, "-e", script, str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
