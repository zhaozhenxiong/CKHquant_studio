"""Contracts at the default training, CLI and display boundaries."""
from contextlib import contextmanager
from pathlib import Path
import json
import subprocess
import sys
import types

import pytest

from my_strategy.services.model_training_center import TrainingCenter


def center(tmp_path):
    result = TrainingCenter.__new__(TrainingCenter)
    result.runs_root = tmp_path / "runs"
    result.db_path = tmp_path / "metadata" / "model_training.db"
    result.store = types.SimpleNamespace(save=lambda kind, payload: {**payload, "saved_kind":kind})
    return result


def install_module(monkeypatch, **functions):
    module = types.ModuleType("my_strategy.services.reference_training")
    for name, function in functions.items():
        setattr(module, name, function)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    import my_strategy.services
    monkeypatch.setattr(my_strategy.services, "reference_training", module, raising=False)
    return module


def test_new_default_task_uses_resource_lease_and_passes_cancellation(tmp_path, monkeypatch):
    facade = center(tmp_path)
    events = []
    @contextmanager
    def lease(progress, check_cancel):
        events.append("lease")
        check_cancel()
        yield
        events.append("released")
    facade._training_resource = lease
    def execute(plan_id, *, parameters, progress, check_cancel, run_callback, runs_root, metadata_root):
        assert events == ["lease", "cancel_checked"]
        assert plan_id == "new-plan" and parameters == {"device":"auto"}
        assert runs_root == facade.runs_root and metadata_root == facade.db_path.parent
        run_callback("reference-run")
        progress("reference_training", 1, 2, actual_device="mps")
        check_cancel()
        return {"run_id":"reference-run", "strategy_version":"reference_position_v1"}
    install_module(monkeypatch, execute_training=execute)
    progress = []
    result = facade.run_training_job({"plan_id":"new-plan", "parameters":{"device":"auto"}},
        progress=lambda *args, **kwargs: progress.append((args,kwargs)),
        check_cancel=lambda: events.append("cancel_checked"), run_callback=lambda run_id:events.append(run_id))
    assert result["saved_kind"] == "model-training"
    assert events == ["lease", "cancel_checked", "reference-run", "cancel_checked", "released"]
    assert progress[0][1]["actual_device"] == "mps"


def test_default_task_never_resumes_legacy_weights(tmp_path):
    facade = center(tmp_path)
    with pytest.raises(ValueError, match="旧模型标签和权重"):
        facade.run_training_job({"plan_id":"old", "resume_training_run_id":"old-run"})
    with pytest.raises(ValueError, match="旧训练恢复入口"):
        facade.preflight({}, resume_native_run_id="old-native-checkpoint")


def test_preflight_keeps_custom_roots_and_native_source(tmp_path, monkeypatch):
    facade = center(tmp_path)
    seen = {}
    def preflight(parameters, **kwargs):
        seen.update(kwargs)
        return {"valid":True, "plan":{"plan_id":"new"}}
    install_module(monkeypatch, preflight=preflight)
    assert facade.preflight({"device":"auto"}, source_native_run_id="causal-native")["valid"]
    assert seen["runs_root"] == facade.runs_root
    assert seen["source_native_run_id"] == "causal-native"


def test_cli_new_training_defaults_to_auto_and_old_entry_is_explicitly_retired(capsys):
    from my_strategy.cli import build_parser, main
    assert build_parser().parse_args(["train-reference"]).device == "auto"
    assert main(["train-trend", "--end", "2026-10-12"]) == 1
    assert "train-reference" in capsys.readouterr().err


def test_reference_position_display_separates_delta_staleness_and_manual_comparison():
    source = Path(__file__).parents[1] / "web_dashboard/static/js/chart_evidence.js"
    script = """
const evidence = require(process.argv[1]);
const row = {date:'2026-10-12', position_reference:{schema:'reference_position_v1',status:'increase_exposure',target_position_fraction:.15,previous_reference_target:.10,change_fraction:.05,manual_comparison_change_fraction:-.2}};
const normal = evidence.positionHtml(row);
row.position_reference.stale = true;
row.position_reference.status = 'unavailable';
const stale = evidence.positionHtml(row);
process.stdout.write(JSON.stringify({normal,stale}));
"""
    result = json.loads(subprocess.run(["node", "-e", script, str(source)], check=True, capture_output=True, text=True).stdout)
    assert "B · 增加参考仓位" in result["normal"]
    assert "+5 个百分点" in result["normal"]
    assert "-20 个百分点（单独比较" in result["normal"]
    assert "8%" not in result["normal"]
    assert "本日无新调整" in result["stale"]
    assert "变化 +5" not in result["stale"]
    assert "S · 减少" not in result["stale"]


def test_reference_position_distinguishes_failed_predictive_gate_from_missing_score():
    source = Path(__file__).parents[1] / "web_dashboard/static/js/chart_evidence.js"
    script = """
const evidence = require(process.argv[1]);
const row = {date:'2026-10-12', position_reference:{schema:'reference_position_v1',status:'pending',stale:true,target_position_fraction:.10,previous_reference_target:.10,change_fraction:0,reason_codes:['reference_predictive_gate_pending']}};
const pending = evidence.positionHtml(row);
row.position_reference.status = 'unavailable';
row.position_reference.reason_codes = ['model_unavailable'];
const unavailable = evidence.positionHtml(row);
process.stdout.write(JSON.stringify({pending,unavailable}));
"""
    result = json.loads(subprocess.run(["node", "-e", script, str(source)], check=True, capture_output=True, text=True).stdout)
    assert "预测基线未通过，保留参考目标" in result["pending"]
    assert "不生成积极增仓目标" in result["pending"]
    assert "评分不可用" not in result["pending"]
    assert "缺失评分" not in result["pending"]
    assert "reference_predictive_gate_pending" not in result["pending"]
    assert "评分不可用" in result["unavailable"]
    assert "预测基线未通过" not in result["unavailable"]


def test_status_exposes_distinct_historical_and_forward_checkpoint_times(tmp_path, monkeypatch):
    facade = center(tmp_path)
    run = facade.runs_root / "reference-run"
    (run / "reports").mkdir(parents=True)
    (run / "config").mkdir()
    (run / "config/training_plan.json").write_text(json.dumps({"latest_eligible_count":2842}))
    report = {"run_id":"reference-run", "status":"completed", "historical_union_count":3194,
        "data_end":"2026-10-12", "source_reuse":{"calendar_run_id":"calendar"},
        "folds":[{"name":"2026Q3", "split":{"test_start":"2026-07-01", "policy_end":"2026-06-30"}}],
        "production":{"name":"production", "split":{"policy_end":"2026-09-30"}, "effective_from":"2026-10-15"}}
    (run / "reports/reference_training.json").write_text(json.dumps(report))
    install_module(monkeypatch, status=lambda **unused: {"models":[{"run_id":"reference-run", "dataset":{"stocks":3194}}], "active_model":{"run_id":"reference-run"}})
    result = facade.status()
    assert result["data_end"] == "2026-10-12" and result["calendar_run_id"] == "calendar"
    model = result["models"][0]
    assert model["historical_union_count"] == model["trained_stock_count"] == 3194
    assert model["latest_eligible_count"] == 2842
    historical, production = model["checkpoints"]
    assert historical["forward_available_at"] is None
    assert production["historical_available_at"] == "2026-10-01"
    assert production["forward_available_at"] == "2026-10-15"


def test_new_reference_resume_routes_in_resource_lease_and_retains_run_identity(tmp_path, monkeypatch):
    facade = center(tmp_path)
    events = []
    @contextmanager
    def lease(*unused):
        events.append("lease")
        yield
        events.append("released")
    facade._training_resource = lease
    adapter = types.ModuleType("my_strategy.services.reference_training_resume")
    def resume(run_id, *, plan_id, runs_root, metadata_root, progress, check_cancel, run_callback):
        assert events == ["lease"]
        assert run_id == "new-schema-run" and plan_id == "repair-plan"
        assert runs_root == facade.runs_root and metadata_root == facade.db_path.parent
        check_cancel()
        run_callback(run_id)
        return {"run_id":run_id, "model_run_id":run_id, "report":{"technical_validation":{"status":"passed"}}}
    adapter.resume_training = resume
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    result = facade.run_training_job({"plan_id":"repair-plan", "resume_reference_run_id":"new-schema-run"},
        check_cancel=lambda:events.append("checked"), run_callback=lambda run_id:events.append(run_id))
    assert result["run_id"] == result["training_run_id"] == "new-schema-run"
    assert events == ["lease", "checked", "new-schema-run", "released"]
    from my_strategy.web_dashboard.api import TrainingJobRequest
    assert TrainingJobRequest(plan_id="repair-plan",resume_reference_run_id="new-schema-run").resume_reference_run_id == "new-schema-run"
    with pytest.raises(ValueError):
        TrainingJobRequest(plan_id="repair-plan",resume_reference_run_id="../escape")


def test_new_chart_uses_joint_probability_and_source_evidence_without_old_weighting():
    source = Path(__file__).parents[1] / "web_dashboard/static/js/chart_evidence.js"
    script = """
const evidence = require(process.argv[1]);
const row={analysis_mode:'reference',position_reference:{schema:'reference_position_v1'},agents:{
 ma:{status:'scored',p_up:.6,p_down:.1,p_neutral:.3},
 structure:{status:'scored',buy_strength:.7,sell_strength:.2},
 wyckoff:{status:'scored',buy_strength:.4,sell_strength:.3}}};
process.stdout.write(evidence.agentOverviewHtml(row));
"""
    html = subprocess.run(["node", "-e", script, str(source)], check=True, capture_output=True, text=True).stdout
    assert "联合方向模型" in html and "上涨 60%" in html and "买方 70 / 100" in html
    assert "权重" not in html and "实际贡献" not in html and "高收益" not in html
