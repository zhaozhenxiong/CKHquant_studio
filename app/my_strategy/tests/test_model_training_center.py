"""Training defaults, immutable plans and real Broker ledger selection gates."""
import copy
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.execution.broker import BrokerSimulator
from my_strategy.execution.cost_model import CostModel
from my_strategy.execution.order import Order
from my_strategy.services import model_training_center as center
from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.services.czsc_account_artifacts import read_account_artifact, write_account_artifact


@pytest.fixture
def service(tmp_path):
    store = ResultStore(tmp_path / "results.db", tmp_path / "runs")
    return center.TrainingCenter(store=store, metadata_root=tmp_path / "metadata")


def account(symbol, dates, profit=.05, multiplier=1., model_id="candidate", source_hash="raw-sha"):
    costs = CostModel(version="flat_v1", impact_tiers=(), commission=.0003*multiplier,
        min_commission=5.*multiplier, stamp_tax=.0005*multiplier, slippage=.0005*multiplier)
    broker = BrokerSimulator(initial_cash=100000., cost_model=costs)
    daily = [dict(date=dates[0], cash=broker.cash, shares=0, close=10., equity=broker.cash)]
    broker.submit(Order(date=dates[1], symbol=symbol, side="BUY", shares=9000, price=10., signal_date=dates[0]))
    daily.append(dict(date=dates[1], cash=broker.cash, shares=9000, close=10., equity=broker.equity({symbol:10.})))
    broker.submit(Order(date=dates[2], symbol=symbol, side="SELL", shares=9000, price=10.*(1+profit), signal_date=dates[1]))
    daily.append(dict(date=dates[2], cash=broker.cash, shares=0, close=10.*(1+profit), equity=broker.cash))
    return dict(symbol=symbol, series_context=dict(strategy="trend_consensus", initial_cash=100000., source_hash=source_hash,
        source_revision_verified=True, calendar_hash=stable_hash(dates), account_start=dates[0],
        model_identity=dict(model_run_id=model_id), mode="final" if model_id else "E0"),
        ledger=broker.ledger.to_frame().to_dict("records"), daily=daily, cost_model=costs.as_config_dict(),
        terminal_state=dict(shares=0, cash=broker.cash))


@pytest.mark.parametrize("parameters", [
    {"min_train_rows":99}, {"min_calibration_rows":29}, {"min_net_return":.079}, {"min_r_multiple":1.99},
    {"max_holding_sessions":61}, {"commission":0}, {"slippage":0}, {"min_commission":0},
    {"stop_loss":.2}, {"epochs":True}, {"learning_rate":.1}, {"ma_weight":.5},
])
def test_current_constraints_cannot_be_weakened(parameters):
    with pytest.raises(ValueError): center.validate_parameters(parameters)


def test_defaults_revision_cas_persist_and_dont_mutate_frozen_config(service):
    before = service.profiles()
    assert before["default_revision"] == 0
    result = service.save_default({"epochs":3}, expected_revision=0)
    assert result["default_profile"]["parameters"]["epochs"] == 3
    with pytest.raises(center.RevisionConflict): service.save_default({"epochs":4}, expected_revision=0)
    fresh = center.TrainingCenter(store=service.store, db_path=service.db_path)
    assert fresh.profiles()["default_revision"] == 1
    cfg = center._configuration(result["default_profile"]["parameters"], "2027-04-01")
    assert cfg["epochs"] == 3 and cfg["batch_size"] == 2048
    assert cfg["development_windows"][-1]["name"] == "2027Q1"
    assert cfg["production"]["validation_end"] == "2027-03-31"
    assert cfg["high_return"]["min_net_return"] == .08


def test_plan_cannot_change_parameters_or_be_claimed_twice(service):
    parameters = center.validate_parameters({"end":"2027-04-01"})
    config = center._configuration(parameters, parameters["end"])
    plan = dict(plan_id="frozen-plan", valid=True, errors=[], parameters=parameters, data_end="2027-04-01",
                config=config, config_hash=stable_hash(config), training_code_freeze=center.training_code_freeze())
    plan["plan_hash"] = stable_hash(plan)
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("INSERT INTO training_plans(plan_id,payload,created_at) VALUES (?,?,?)", (plan["plan_id"],json.dumps(plan),"now"))
    assert service.validate_plan("frozen-plan") == plan
    with pytest.raises(ValueError, match="不同"):
        service.validate_plan("frozen-plan", {**parameters,"epochs":1})
    service.load_plan("frozen-plan", claim=True)
    with pytest.raises(center.RevisionConflict): service.load_plan("frozen-plan", claim=True)


def test_plan_without_code_freeze_cannot_validate_or_claim(service):
    parameters = center.validate_parameters({"end":"2027-04-01"})
    config = center._configuration(parameters, parameters["end"])
    plan = dict(plan_id="old-unfrozen-plan", valid=True, errors=[], parameters=parameters,
                data_end=parameters["end"], config=config, config_hash=stable_hash(config))
    plan["plan_hash"] = stable_hash(plan)
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("INSERT INTO training_plans(plan_id,payload,created_at) VALUES (?,?,?)",
                     (plan["plan_id"], json.dumps(plan), "archived"))
    for action in (lambda: service.validate_plan(plan["plan_id"]),
                   lambda: service.load_plan(plan["plan_id"], claim=True)):
        with pytest.raises(ValueError, match="生成代码已改变"):
            action()
        with sqlite3.connect(service.db_path) as conn:
            assert conn.execute("SELECT claimed FROM training_plans WHERE plan_id=?", (plan["plan_id"],)).fetchone()[0] == 0
    assert service.load_plan(plan["plan_id"]) == plan


def test_actual_broker_ledger_fee_and_maturity_recompute():
    dates = ["2027-01-04", "2027-01-05", "2027-01-06"]
    result = account("600000.SH", dates)
    metrics = center.audit_account(result, dates)
    assert metrics["completed_round_trips"] == 1
    assert metrics["total_return"] == pytest.approx((4500-27-28.35-47.25)/100000.)
    broken = copy.deepcopy(result); broken["ledger"][0]["fee"] = 0
    with pytest.raises(ValueError, match="费用"): center.audit_account(broken, dates)
    broken = copy.deepcopy(result); broken["ledger"][1]["signal_date"] = dates[2]
    with pytest.raises(ValueError, match="执行时点"): center.audit_account(broken, dates)
    broken = copy.deepcopy(result); broken["terminal_state"]["shares"] = 100
    with pytest.raises(ValueError, match="期末"): center.audit_account(broken, dates)


def windows_from_brokers(count=60, candidate_profit=.05, stress_profit=.04):
    windows = []
    for index in range(4):
        dates = [f"2027-{index+1:02d}-{day:02d}" for day in (4,5,6)]
        arms = {}
        for name, profit, multiplier in (("candidate",candidate_profit,1.),("rules",.02,1.),("candidate_stress",stress_profit,2.),("rules_stress",.01,2.)):
            accounts = [center.audit_account(account(f"60{n:04d}.SH", dates, profit, multiplier, "candidate" if name.startswith("candidate") else None),dates) for n in range(count)]
            arms[name] = dict(metrics=center.aggregate_account_metrics(accounts))
        windows.append(dict(window=dict(name=f"window{index}",test_start=dates[0],test_end=dates[-1]),arms=arms))
    return windows


POLICY = dict(min_stocks=60,min_round_trips_per_window=50,min_round_trips_total=200,drawdown_tolerance=.02)


def test_positive_selection_gate_uses_real_broker_ledger():
    result = center.selection_gate(windows_from_brokers(),POLICY,60,False)
    assert result["passed"] and result["completed_round_trips"] == 240
    assert result["paired_statistics"][0]["positive_pairs"] == 60


@pytest.mark.parametrize("count,candidate_profit,stress_profit,reason", [
    (12,.05,.04,"股票"),(60,.01,.04,"收益"),(60,.05,0.,"加倍"),
])
def test_negative_broker_comparison_refuses_replacement(count,candidate_profit,stress_profit,reason):
    result = center.selection_gate(windows_from_brokers(count,candidate_profit,stress_profit),POLICY,count,False)
    assert not result["passed"] and any(reason in value for value in result["reasons"])


def test_drawdown_gate_uses_stricter_baseline():
    windows = windows_from_brokers()
    windows[0]["arms"]["candidate"]["metrics"]["drawdown"] = .05
    result = center.selection_gate(windows,POLICY,60,False)
    assert not result["passed"] and any("回撤" in v for v in result["reasons"])


def test_incumbent_cas_next_session_and_append_only_events(verified_comparison):
    service,_,_=verified_comparison
    result = service.select_candidate("candidate",None)
    assert result["status"] == "selected"
    assert service.active_model_run_id("2026-10-13") is None
    assert service.active_model_run_id("2029-01-03") == "candidate"
    assert service.active_model_run_id("2029-01-03",scope="production") is None
    with pytest.raises(center.RevisionConflict): service.select_candidate("candidate",None)
    with sqlite3.connect(service.registry_path) as conn:
        with pytest.raises(sqlite3.IntegrityError): conn.execute("DELETE FROM current_model_events")
    assert len(service.selection_events()["items"]) == 1


def add_current_derivations(report,plan):
    cfg=report["config"]; source=dict(snapshot_hash="source-fixture",records_hash="records-fixture",source_snapshot=dict(calendar_hash="calendar-fixture"))
    report.update(source_snapshot=source,trend_bundles=[],ma_exit_bundles=[])
    plan.setdefault("cohort", ["600000.SH"])
    report.setdefault("dataset_records", [dict(symbol=s) for s in plan["cohort"]])
    cfg.update(cpu_workers=plan["parameters"]["cpu_workers"],training_end=plan["data_end"],source_snapshot_hash=source["snapshot_hash"],
        training_capacity_tiers=[len(plan["cohort"])],
        capacity_expansion_criterion="all_frozen_historical_liquidity_qualified_symbols_no_readiness_early_stop")
    cfg["pipeline_contract_hash"]=stable_hash(dict(family="trend_consensus",source=source,calendar_hash="calendar-fixture",registered_config=copy.deepcopy(cfg),
        order=["independent_trend","policy_only_sources","E0_no_HR","MA_exit_E0","pinned_E1",
            "preregistered_actual_E1_holding_pairs","account_free_reference_quality_probes",
            "independent_reference_quality_heads_and_fee_qualification","strict_prior_L2_actual_flat_population_by_fold",
            "L2_entry_forced_E1_clone","L2_exit_preregistered_actual_E1_pairs"],
        reference_population_protocol="strict_prior_l2_actual_flat_prefix_pre_gate_attempts",self_training_population_feedback=False,reassessment=cfg["reassessment"],no_E2_entry_reuse=True))
    cfg["candidate_policy_hash"]=stable_hash(dict(candidate=cfg["candidate_policy"],risk_policy=cfg["risk_policy"],trend_candidate={k:cfg["trend"][k] for k in ("candidate_cross_threshold","candidate_margin")},
        consensus=cfg["consensus"],signal_points=cfg["signal_points"],native_records="records-fixture",reassessment=cfg["reassessment"],training_population="strict_prior_l2_actual_flat_prefix_pre_gate_attempts"))
    cfg["E0_policy_hash"]=stable_hash(dict(stage="E0",candidate_policy_hash=cfg["candidate_policy_hash"],account=cfg["account"],trend_catalog_hash=stable_hash([]),ordinary_sell="continuous_sell_pass_for_actual_holding",reassessment=cfg["reassessment"],entry_HR=False))
    cfg["E1_policy_hash"]=stable_hash(dict(stage="E1",E0=cfg["E0_policy_hash"],signal_time_pinned_MA_exit=stable_hash([]),holding_threshold=cfg["account"]["holding_exit_threshold"],no_head="entire_position_E0",checkpoint_upgrade=False,L2_exit_apply=False))
    from my_strategy.services.czsc_trend_contracts import (
        HELD_TRAINING_POPULATION,HELD_CONDITION_DOMAIN,HELD_SPLIT_POLICY,HELD_ECONOMIC_WEIGHTING)
    held=dict(version=HELD_TRAINING_POPULATION,stage="E1",reference_mode="E1",
        conditional_holding_domain=HELD_CONDITION_DOMAIN,pending_L2_entry_or_exit_used=False,
        candidate_clones_are_holding_population=False,holding_checkpoint_upgrade=False,
        split_policy=HELD_SPLIT_POLICY,economic_weighting=HELD_ECONOMIC_WEIGHTING,
        source_snapshot_hash=cfg["source_snapshot_hash"],candidate_policy_hash=cfg["candidate_policy_hash"],
        continuation_policy_hash=cfg["E1_policy_hash"],manifest_path="dataset/E1/manifest.json",
        manifest_sha256="test-held-manifest",pooled_exits_path="dataset/E1/exits.parquet",
        pooled_exits_sha256="test-held-exits",records_hash="test-held-records")
    report["holding_population"]=held
    report.setdefault("protocol_freeze",{})["holding_population_hash"]=stable_hash(held)


def test_active_reference_work_blocks_switch_and_rollback(service, monkeypatch):
    service.selection_guard=lambda:["active reference scan"]
    monkeypatch.setattr(service,"_verify_comparison",lambda run:(dict(incumbent_run_id=None),{},{}))
    assert service.select_candidate("candidate")["status"] == "blocked"
    assert service.rollback(None)["status"] == "blocked"
    assert service.current_selection() is None


def test_rollback_to_observation_cas(service):
    with sqlite3.connect(service.registry_path) as conn:
        conn.execute("INSERT INTO current_model_selection VALUES (1,'current-good',1,'2026-01-01')")
    with pytest.raises(center.RevisionConflict): service.rollback("stale")
    result = service.rollback("current-good")
    assert result["status"] == "rolled_back"
    assert service.active_model_run_id("2027-01-01") is None


def test_comparison_hash_drift_rejected(service):
    service._save_comparison("candidate",dict(passed=False,status="pending"))
    path = next((service.runs_root / "candidate/validation/model_comparison").glob("*/comparison.json"))
    path.write_text('{"passed":true}')
    with pytest.raises(ValueError,match="哈希"): service.comparison("candidate")


@pytest.fixture
def verified_comparison(service, monkeypatch):
    from my_strategy.services import czsc_trend_support
    cohort = [f"60{n:04d}.SH" for n in range(60)]
    windows = [dict(name=f"window{i}",test_start=f"2027-{i+1:02d}-04",test_end=f"2027-{i+1:02d}-06") for i in range(4)]
    calendar = [f"2027-{i+1:02d}-{day:02d}" for i in range(4) for day in (4,5,6)]
    cfg = strategy_config()
    root = service.runs_root / "candidate"; (root / "reports").mkdir(parents=True)
    report = dict(run_id="candidate",config=cfg,completed_at="2027-04-06T16:00:00+08:00",
                  dataset_records=[dict(symbol=s,sha256="sha-"+s) for s in cohort])
    report_path = root / "reports/trend_consensus.json"; center._save(report_path, report)
    monkeypatch.setattr(service,"_model",lambda run:(root,report))
    monkeypatch.setattr(service,"_evaluation_records",lambda current, plan:report["dataset_records"])
    monkeypatch.setattr(service,"_source_provenance_files",lambda current, plan:[])
    monkeypatch.setattr(czsc_trend_support,"_calendar",lambda identity:(calendar+["2029-01-02"],"calendar"))
    plan = dict(plan_id="real-broker-plan",cohort=cohort,windows=windows,data_end="2027-04-06",calendar_run_id="calendar",
                selection_policy=POLICY, incumbent_run_id=None,valid=True,parameters=center.validate_parameters(),errors=[])
    plan["plan_hash"] = stable_hash(plan)
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("INSERT INTO training_plans(plan_id,payload,created_at) VALUES (?,?,?)",(plan["plan_id"],json.dumps(plan),"now"))
    result = dict(run_id="candidate",plan_id=plan["plan_id"],plan_hash=plan["plan_hash"],incumbent_run_id=None,
        candidate_report_sha256=center._file_hash(report_path),incumbent_report_sha256=None,windows=[],passed=True,status="passed")
    for window in windows:
        dates = [d for d in calendar if window["test_start"] <= d <= window["test_end"]]
        arms = {}
        for name, profit, multiplier in (("candidate",.05,1.),("rules",.02,1.),("candidate_stress",.04,2.),("rules_stress",.01,2.)):
            assets, measures = [], []
            for symbol in cohort:
                value = account(symbol,dates,profit,multiplier,"candidate" if name.startswith("candidate") else None,"sha-"+symbol)
                value["series_context"]["calendar_hash"] = stable_hash(calendar)
                path = root / "validation" / window["name"] / name / (symbol+".json"); write_account_artifact(path,value)
                ledger_path = path.with_suffix(".csv"); pd.DataFrame(value["ledger"]).to_csv(ledger_path,index=False)
                assets.append(dict(path=path.relative_to(root).as_posix(),sha256=center._file_hash(path),ledger_path=ledger_path.relative_to(root).as_posix(),ledger_sha256=center._file_hash(ledger_path)))
                measures.append(center.audit_account(value,calendar))
            arms[name] = dict(assets=assets,metrics=center.aggregate_account_metrics(measures))
        result["windows"].append(dict(window=window,arms=arms))
    service._save_comparison("candidate",result)
    return service, result, plan


def test_full_persisted_broker_comparison_then_selection(verified_comparison):
    service, result, plan = verified_comparison
    verified, frozen, _ = service._verify_comparison("candidate")
    assert verified == result and frozen == plan
    selected = service.select_candidate("candidate")
    assert selected["status"] == "selected"
    assert len(service.selection_events()["items"]) == 1


@pytest.mark.parametrize("change", ["window", "arm", "repeat_stock", "cost", "identity", "source", "metrics"])
def test_selection_cannot_accept_changed_broker_bindings(verified_comparison, change):
    service, result, _ = verified_comparison
    changed = copy.deepcopy(result)
    window = changed["windows"][0]
    if change == "window": window["window"]["test_start"] = "2027-01-05"
    elif change == "arm": del window["arms"]["rules_stress"]
    elif change == "repeat_stock": window["arms"]["candidate"]["assets"][-1] = window["arms"]["candidate"]["assets"][0]
    elif change == "metrics": window["arms"]["candidate"]["metrics"]["total_return"] = 100.
    else:
        asset = window["arms"]["candidate"]["assets"][0]
        path = service.runs_root / "candidate" / asset["path"]; account_data = read_account_artifact(path)
        if change == "cost": account_data["cost_model"]["commission"] = 0.
        elif change == "identity": account_data["series_context"]["model_identity"]["model_run_id"] = "retired"
        elif change == "source": account_data["series_context"]["source_hash"] = "not-frozen"
        write_account_artifact(path,account_data); asset["sha256"] = center._file_hash(path)
    monkeypatch_result = lambda _:changed
    service.comparison = monkeypatch_result
    with pytest.raises(ValueError): service.select_candidate("candidate")
    assert service.current_selection() is None


def test_training_resource_wait_can_cancel_without_locking_defaults(service):
    cancelled=threading.Event(); waited=threading.Event(); entered=threading.Event(); errors=[]
    def check():
        if cancelled.is_set(): raise RuntimeError("cancel resource wait")
    def contender():
        try:
            with service._training_resource(lambda *args,**kwargs:waited.set(),check): entered.set()
        except RuntimeError as exc: errors.append(str(exc))
    with service._training_resource(None,None):
        thread=threading.Thread(target=contender); thread.start()
        assert waited.wait(2) and not entered.is_set()
        assert service.save_default({"epochs":2},0)["default_revision"]==1
        cancelled.set(); thread.join(2)
        assert not thread.is_alive()
    assert errors==["cancel resource wait"] and not entered.is_set()
    with service._training_resource(None,None): entered.set()
    assert entered.is_set()


def test_unavailable_explicit_device_is_preflight_error(service,monkeypatch):
    from my_strategy.services import czsc_trend_math
    def unavailable(device): raise RuntimeError("explicit CUDA unavailable")
    monkeypatch.setattr(czsc_trend_math,"_device",unavailable)
    result=service.preflight({"device":"cuda"})
    assert not result["valid"] and result["errors"]==["explicit CUDA unavailable"]


def test_subprocess_cancel_terminates_own_child_tree(tmp_path):
    import psutil
    cancelled=threading.Event(); child_pids=[]
    code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print('STAGE owned_child CHILD '+str(p.pid),flush=True); time.sleep(30)"
    def progress(payload):
        if payload.get("stage")=="owned_child":
            child_pids.append(int(payload["message"].split()[-1])); cancelled.set()
    def check():
        if cancelled.is_set(): raise RuntimeError("cancel requested")
    with pytest.raises(RuntimeError,match="cancel requested"):
        center.TrainingCenter._run_process([sys.executable,"-u","-c",code],tmp_path/"child.log",progress,check)
    assert child_pids
    deadline=time.monotonic()+2
    while psutil.pid_exists(child_pids[0]) and time.monotonic()<deadline:
        if psutil.Process(child_pids[0]).status()==psutil.STATUS_ZOMBIE: break
        time.sleep(.02)
    assert not psutil.pid_exists(child_pids[0]) or psutil.Process(child_pids[0]).status()==psutil.STATUS_ZOMBIE


def test_capacity_tier_uses_frozen_full_native_evaluation_pool(service,monkeypatch):
    from my_strategy.services import czsc_trend_pipeline
    native_root=service.runs_root/"native-source"
    service.runs_root.mkdir(parents=True,exist_ok=True)
    records=[]
    for symbol in ("600000.SH","600001.SH"):
        raw=native_root/"data"/(symbol+".raw"); raw.parent.mkdir(parents=True,exist_ok=True); raw.write_text("frozen raw "+symbol)
        records.append(dict(symbol=symbol,raw_path=str(raw),raw_sha256=center._file_hash(raw)))
    native=dict(strategy_version="trend_consensus",raw_only_rebuild=True,reused_model_or_label=False,data_end="2026-10-12",
        dataset_records=records,source_snapshot=dict(raw_input_hashes={r["symbol"]:r["raw_sha256"] for r in records}))
    path=native_root/"reports/native_source.json"; center._save(path,native)
    report=dict(dataset_records=records[:1],validation_records=records,
        source_snapshot=dict(source_native_run_id="native-source",source_report_sha256=center._file_hash(path)))
    plan=dict(cohort=[r["symbol"] for r in records],data_end="2026-10-12")
    monkeypatch.setattr(czsc_trend_pipeline,"read_snapshot",lambda record:(None,None))
    assert service._evaluation_records(report,plan)==records
    Path(records[1]["raw_path"]).write_text("changed raw")
    with pytest.raises(ValueError,match="资产改变"): service._evaluation_records(report,plan)


@pytest.mark.parametrize("failure",[False,True])
def test_revalidation_does_not_retrain_or_claim_original_plan(service,monkeypatch,failure):
    from types import SimpleNamespace
    from my_strategy.services import czsc_trend_pipeline
    parameters=center.validate_parameters({"end":"2026-10-12"})
    config=center._configuration(parameters,"2026-10-12")
    plan=dict(plan_id="original-claimed",valid=True,errors=[],parameters=parameters,config=config,
        data_end="2026-10-12",cohort=["600000.SH"],incumbent_run_id=None,calendar_run_id="calendar",
        windows=config["development_windows"],selection_policy=POLICY,
        config_hash=stable_hash(config),training_code_freeze=center.training_code_freeze())
    plan["plan_hash"]=stable_hash(plan)
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("INSERT INTO training_plans(plan_id,payload,created_at,claimed) VALUES (?,?,?,1)",(plan["plan_id"],json.dumps(plan),"original"))
    trained=dict(status="completed",strategy_version="trend_consensus",data_end=plan["data_end"],config=copy.deepcopy(config),required_head_status=dict(production_entry_ready=False))
    trained["protocol_freeze"] = copy.deepcopy(plan["training_code_freeze"])
    add_current_derivations(trained,plan)
    model_root=service.runs_root/"trained"; model_root.mkdir(parents=True)
    weights=model_root/"weights.bin"; weights.write_bytes(b"immutable fitted fixture weights")
    before=center._file_hash(weights)
    monkeypatch.setattr(czsc_trend_pipeline,"read_run",lambda run:(model_root,trained))
    monkeypatch.setattr(service,"_evaluation_records",lambda report,plan:[])
    parent=service.runs_root/"validation-parent"
    def subdir(name):
        target=parent/name; target.mkdir(parents=True,exist_ok=True); return target
    context=SimpleNamespace(run_id="validation-parent",subdir=subdir,write_metadata=lambda value:center._save(parent/"metadata.json",value))
    monkeypatch.setattr(center,"create_run_context",lambda **kwargs:context)
    def diagnostic(*args):
        if failure: raise RuntimeError("diagnostic source failed")
        return dict(status="completed",run_id="diagnostic",report_paths={},report_sha256="hash")
    monkeypatch.setattr(service,"_run_diagnostics",diagnostic)
    monkeypatch.setattr(service,"evaluate_candidate",lambda run,plan,**kwargs:dict(run_id=run,plan_id=plan["plan_id"],status="pending",passed=False,reasons=["entry class unavailable"]))
    callbacks=[]
    result=service.run_validation_job(dict(plan_id=plan["plan_id"],model_run_id="trained"),run_callback=callbacks.append)
    assert callbacks==["validation-parent"]
    assert result["status"]=="completed_unqualified" and result["training_started"] is False
    assert result["weights_unchanged"] and center._file_hash(weights)==before
    assert result["original_plan_id"]==plan["plan_id"] and result["plan_id"]!=plan["plan_id"]
    assert service.validate_plan(plan["plan_id"])==plan
    assert result["diagnostics"]["status"]==("failed" if failure else "completed")
    assert service.current_selection() is None


def test_revalidation_frozen_parameters_cannot_be_redefined():
    values=center.validate_parameters({"end":"2026-10-12"})
    config=center._configuration(values,values["end"])
    plan=dict(config=config,data_end=values["end"],parameters=values)
    report=dict(strategy_version="trend_consensus",status="completed",data_end=values["end"],config=copy.deepcopy(config))
    add_current_derivations(report,plan)
    center.TrainingCenter._validate_trained_plan(report,plan)
    report["config"]["account"]["commission"]=0.
    with pytest.raises(ValueError,match="account"): center.TrainingCenter._validate_trained_plan(report,plan)


@pytest.mark.parametrize("stocks",[12,60])
def test_mutable_readiness_flags_cannot_qualify_missing_frozen_heads(service,monkeypatch,stocks):
    from my_strategy.services import czsc_trend_pipeline
    root=service.runs_root/"missing-heads"
    center._save(root/"data_snapshots/calendar.json",dict(dates=["2026-04-01","2026-04-02","2026-04-03","2026-04-06"]))
    report=dict(config=strategy_config(),data_end="2026-10-12",dataset_records=[dict(symbol=f"60{i:04d}.SH") for i in range(stocks)],
        trend_bundles=[],ma_exit_bundles=[],model_bundles=[],required_recent_heads_ready=True,automatic_selection_eligible=True)
    monkeypatch.setattr(czsc_trend_pipeline,"read_run",lambda run:(root,report))
    with pytest.raises(ValueError,match="覆盖不足|实际必需头"):
        service._model("missing-heads")


def test_explicit_scalar_thresholds_freeze_search_without_implicit_weakening():
    values=center.validate_parameters(dict(buy_base_threshold=.8,sell_threshold=.75,ma_exit_threshold=.8))
    cfg=center._configuration(values,"2026-10-12")
    assert cfg["consensus"]["policy_grid"]["buy_base_threshold"]==[.8]
    assert cfg["consensus"]["policy_grid"]["sell_threshold"]==[.75]
    assert cfg["exit_threshold_grid"]==[.8]
    defaults=center._configuration(center.validate_parameters(),"2026-10-12")
    assert defaults["consensus"]["policy_grid"]["buy_base_threshold"]==[.35,.4,.45]
    assert defaults["exit_threshold_grid"]==[.55,.65]


def test_alias_without_qualified_event_is_integrity_blocked(service):
    with sqlite3.connect(service.registry_path) as conn:
        conn.execute("INSERT INTO current_model_selection VALUES (1,'forged',999,'2020-01-01')")
    assert service.active_model_run_id() is None
    assert service.status()["mode"]=="integrity_blocked"


def test_active_proof_uses_immutable_selected_comparison_not_latest_row(verified_comparison):
    service,result,plan=verified_comparison
    service.select_candidate("candidate")
    assert service.active_model_run_id("2029-01-03")=="candidate"
    service._save_comparison("candidate",dict(run_id="candidate",plan_id="later-pending-plan",passed=False,status="pending"))
    assert service.active_model_run_id("2029-01-03")=="candidate"
    asset=result["windows"][0]["arms"]["candidate"]["assets"][0]
    path=service.runs_root/"candidate"/asset["path"]
    before=path.stat(); original=path.read_text()
    envelope=json.loads(original)
    # Mutate same-size envelope metadata without changing its mtime.
    assert "schema" in envelope
    changed=original.replace(envelope["schema"], envelope["schema"][:-1]+"X",1)
    assert original!=changed and len(original)==len(changed)
    path.write_text(changed); os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert service.active_model_run_id("2029-01-03") is None
    assert service.status()["mode"]=="integrity_blocked"


@pytest.mark.parametrize("parameters", [
    {"liquidity_min_median_amount":29999999}, {"liquidity_lookback_sessions":19},
    {"stock_batch_size":True}, {"cohort_size":360}, {"capacity_tiers":[60,180,360]},
])
def test_liquidity_full_pool_cannot_be_truncated_or_automatically_weakened(parameters):
    with pytest.raises(ValueError): center.validate_parameters(parameters)


def test_archived_fixed_size_defaults_migrate_once_without_rewriting_history(service):
    archived={k:v for k,v in center.DEFAULTS.items() if not k.startswith("liquidity_") and k!="stock_batch_size"}
    archived.update(cohort_size=360,capacity_tiers=[60,180,360],end="2026-10-01",device="mps",epochs=20)
    serialized=json.dumps(archived)
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("INSERT INTO training_profiles VALUES ('personal',4,'old',?,?,?)",(serialized,stable_hash(archived),"before"))
        conn.execute("UPDATE training_default SET profile_id='personal',revision=4 WHERE id=1")
    migrated=center.TrainingCenter(store=service.store,db_path=service.db_path)
    saved=migrated.profiles()["default_profile"]
    assert saved["revision"]==5 and saved["parameters"]["end"] is None
    assert saved["parameters"]["liquidity_min_median_amount"]==30000000
    assert saved["parameters"]["stock_batch_size"]==360 and saved["parameters"]["device"]=="mps"
    assert "cohort_size" not in saved["parameters"]
    with sqlite3.connect(service.db_path) as conn:
        assert conn.execute("SELECT parameters FROM training_profiles WHERE revision=4").fetchone()[0]==serialized
    assert center.TrainingCenter(store=service.store,db_path=service.db_path).profiles()["default_revision"]==5


def _preflight_dependencies(monkeypatch, universe):
    from contextlib import contextmanager
    from my_strategy.adapters import czsc_adapter
    from my_strategy.services import czsc_trend_support,czsc_trend_source,czsc_trend_math
    class Raw:
        def execute(self,*args): return self
        def fetchone(self): return ("2026-10-12",)
    @contextmanager
    def connection(*args): yield Raw()
    monkeypatch.setattr(czsc_adapter,"_connection",connection)
    monkeypatch.setattr(czsc_trend_math,"_device",lambda value:None)
    monkeypatch.setattr(czsc_trend_support,"_calendar",lambda identity:(["2026-09-30","2026-10-08","2026-10-09","2026-10-12"],"verified"))
    monkeypatch.setattr(czsc_trend_source,"build_liquidity_universe",universe)


def test_holiday_preflight_has_adjacent_verified_days_without_silent_shift(service,monkeypatch):
    _preflight_dependencies(monkeypatch,lambda **kwargs:pytest.fail("holiday proceeded to stock selection"))
    result=service.preflight(dict(end="2026-10-01"))
    assert not result["valid"] and "不是核验交易日" in result["errors"][0]
    status=result["plan"]["date_status"]
    assert status["requested_end"]=="2026-10-01" and status["effective_end"] is None
    assert status["nearest_verified_before"]=="2026-09-30" and status["nearest_verified_after"]=="2026-10-08"


def test_blank_end_freezes_latest_and_batch_does_not_cap_historical_union(service,monkeypatch):
    symbols=[f"60{i:04d}.SH" for i in range(500)]
    calls=[]
    def universe(**kwargs):
        calls.append(kwargs)
        return dict(selected_symbols=symbols,latest_count=450,population_hash="frozen-population")
    _preflight_dependencies(monkeypatch,universe)
    result=service.preflight(dict(stock_batch_size=7))
    plan=result["plan"]
    assert result["valid"] and len(plan["cohort"])==500
    assert plan["historical_union_count"]==500 and plan["latest_eligible_count"]==450
    assert plan["data_end"]=="2026-10-12" and plan["requested_end"] is None
    assert plan["stock_batch_size"]==7 and plan["config"]["training_capacity_tiers"]==[500]
    assert calls[0]["policy"]["min_median_amount"]==30000000
    assert plan["config"]["full_qualified_pool"] is True


def test_late_listing_retains_real_cash_broker_and_never_invents_quotes():
    dates=["2026-01-05","2026-01-06","2026-01-07"]
    window=dict(test_start=dates[0],test_end=dates[-1])
    result=center._cash_evaluation_account(dict(symbol="600000.SH",sha256="frozen"),dates,
        (Path("candidate"),dict(run_id="candidate")),strategy_config(),window,2.)
    measured=center.audit_account(result,dates)
    assert result["ledger"]==[] and all(r["close"] is None for r in result["daily"])
    assert measured["total_return"]==0 and measured["completed_round_trips"]==0
    assert result["cost_model"]["commission"]==.0006 and result["cost_model"]["min_commission"]==10


def test_missing_heads_still_replays_real_fee_net_brokers_without_qualification(service,monkeypatch):
    from my_strategy.services import czsc_trend_pipeline
    root=service.runs_root/"missing-head-candidate"
    dates=["2027-01-04","2027-01-05","2027-01-06"]
    window=dict(name="locked",test_start=dates[0],test_end=dates[-1])
    cfg=strategy_config()
    report=dict(run_id=root.name,data_end=dates[-1],config=cfg,required_head_status=dict(production_entry_ready=False))
    center._save(root/"reports/trend_consensus.json",report)
    center._save(root/"data_snapshots/calendar.json",dict(dates=dates))
    plan=dict(plan_id="head-diagnostic",plan_hash="frozen",incumbent_run_id=None,data_end=dates[-1],
        cohort=["600000.SH"],windows=[window],parameters=dict(cpu_workers=1),selection_policy=POLICY)
    monkeypatch.setattr(czsc_trend_pipeline,"read_run",lambda run:(root,report))
    monkeypatch.setattr(service,"_validate_trained_plan",lambda *args:None)
    monkeypatch.setattr(service,"_evaluation_records",lambda *args:[dict(symbol="600000.SH")])
    def unqualified(run): raise center.ModelQualificationError("missing entry head")
    monkeypatch.setattr(service,"_model",unqualified)
    calls=[]
    def measured(record,specification):
        calls.append(record["symbol"])
        arms={}
        for key,multiplier in (("candidate",1.),("rules",1.),("candidate_stress",2.),("rules_stress",2.)):
            value=account(record["symbol"],dates,.05,multiplier,root.name if key.startswith("candidate") else None)
            arms[key]=dict(metrics=center.audit_account(value,dates),asset={})
        return dict(symbol=record["symbol"],windows=[arms])
    monkeypatch.setattr(center,"_compare_stock",measured)
    result=service.evaluate_candidate(root.name,plan)
    assert calls==["600000.SH"] and len(result["windows"])==1
    assert result["windows"][0]["arms"]["candidate"]["metrics"]["completed_round_trips"]==1
    assert result["windows"][0]["arms"]["candidate_stress"]["metrics"]["fees"]>result["windows"][0]["arms"]["candidate"]["metrics"]["fees"]
    assert result["model_qualification"]["passed"] is False and result["diagnostic_replay_only"] is True
    assert not result["passed"] and "missing entry head" in result["reasons"]


def test_evaluation_execution_code_drift_blocks_even_saved_passed_comparison(verified_comparison,monkeypatch):
    service,result,plan=verified_comparison
    before=copy.deepcopy(result["evaluation_freeze"])
    monkeypatch.setattr(center,"evaluation_freeze",lambda:{**before,"freeze_hash":"changed-execution"})
    with pytest.raises(ValueError,match="评估或Broker执行代码"):
        service.select_candidate("candidate")
    assert service.current_selection() is None


def test_different_incumbent_liquidity_policy_is_not_silently_rewritten(service,monkeypatch):
    from my_strategy.services import czsc_trend_pipeline
    root=service.runs_root/"different-policy"
    cfg=strategy_config()
    report=dict(config=cfg,data_end="2026-10-12",run_id=root.name)
    incumbent=copy.deepcopy(report)
    incumbent["config"]["liquidity_universe"]["min_median_amount"]=60000000
    monkeypatch.setattr(czsc_trend_pipeline,"read_run",lambda run:(root,report))
    monkeypatch.setattr(service,"_validate_trained_plan",lambda *args:None)
    monkeypatch.setattr(service,"_evaluation_records",lambda *args:[])
    monkeypatch.setattr(service,"_model",lambda run:(root,incumbent if run=="incumbent" else report))
    plan=dict(plan_id="liquidity-drift",plan_hash="frozen",incumbent_run_id="incumbent",data_end=report["data_end"])
    result=service.evaluate_candidate(root.name,plan)
    assert not result["passed"] and any("流动性政策不同" in reason for reason in result["reasons"])
    assert incumbent["config"]["liquidity_universe"]["min_median_amount"]==60000000


@pytest.mark.parametrize("field,value", [("expectancy",float("nan")),("fees",float("inf")),("completed_round_trips",True)])
def test_nonfinite_or_boolean_metrics_cannot_pass_selection(field,value):
    windows=windows_from_brokers()
    for item in windows: item["arms"]["candidate"]["metrics"][field]=value
    result=center.selection_gate(windows,POLICY,60,False)
    assert not result["passed"] and any("有限指标" in reason for reason in result["reasons"])


def test_cancel_kills_own_group_when_child_ignores_sigterm(tmp_path):
    import psutil
    cancelled=threading.Event(); pids=[]
    child="import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('READY',flush=True); time.sleep(60)"
    code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"+repr(child)+"],stdout=subprocess.PIPE,text=True); p.stdout.readline(); print('STAGE owned_child CHILD '+str(p.pid),flush=True); time.sleep(60)"
    def progress(payload):
        if payload.get("stage")=="owned_child":
            pids.append(int(payload["message"].split()[-1])); cancelled.set()
    def check():
        if cancelled.is_set(): raise RuntimeError("cancel own isolated group")
    try:
        with pytest.raises(RuntimeError,match="cancel own"):
            center.TrainingCenter._run_process([sys.executable,"-u","-c",code],tmp_path/"group.log",progress,check)
        assert pids, "The cancellation must observe an actual owned child"
        for pid in pids:
            try:
                process = psutil.Process(pid)
                assert not process.is_running() or process.status()==psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                # The owned job can finish between the liveness and status reads.
                pass
    finally:
        for pid in pids:
            if psutil.pid_exists(pid):
                try: psutil.Process(pid).kill()
                except psutil.Error: pass


def test_pending_same_or_future_target_cannot_bypass_next_session_by_rollback(service,monkeypatch):
    monkeypatch.setattr(service,"_model",lambda run:(None,{}))
    for target in ("current", "future"):
        monkeypatch.setattr(service,"selection_events",lambda *args:dict(items=[dict(kind="select",event_id=1,
            payload=dict(run_id=target,effective_from="2099-01-01"))]))
        with pytest.raises(ValueError,match="提前激活|待生效"):
            service.rollback("current",target)
    monkeypatch.setattr(service,"selection_events",lambda *args:dict(items=[dict(kind="select",event_id=1,
        payload=dict(run_id="unknown-day",effective_from=None))]))
    with pytest.raises(ValueError,match="待生效"):
        service.rollback("current","unknown-day")
    assert service.current_selection() is None


def test_archived_fixed_size_plan_requires_fresh_liquidity_preflight(service):
    plan=dict(plan_id="old-fixed-plan",valid=True,errors=[],parameters=dict(cohort_size=360,capacity_tiers=[60,180,360]))
    plan["plan_hash"]=stable_hash(plan)
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("INSERT INTO training_plans(plan_id,payload,created_at) VALUES (?,?,?)",(plan["plan_id"],json.dumps(plan),"archived"))
    with pytest.raises(ValueError,match="旧固定数量预检已退役"):
        service.validate_plan(plan["plan_id"])


def test_alias_cannot_change_run_or_effective_date(verified_comparison):
    service,_,_=verified_comparison
    service.select_candidate("candidate")
    with sqlite3.connect(service.registry_path) as conn:
        conn.execute("UPDATE current_model_selection SET effective_from='2020-01-01'")
    assert service.active_model_run_id() is None and service.integrity_errors


def test_transaction_rechecks_active_reference_work(verified_comparison):
    service,_,_=verified_comparison
    checks=[]
    def guard():
        checks.append(True)
        return [] if len(checks)==1 else ["active reference task appeared"]
    service.selection_guard=guard
    assert service.select_candidate("candidate")["status"]=="blocked"
    assert service.current_selection() is None


def test_training_center_has_no_retired_account_constructor_fields(service):
    import inspect
    parameters=inspect.signature(center.TrainingCenter).parameters
    assert "holdings_path" not in parameters and "personal_db" not in parameters
    assert not hasattr(service,"holding_guard") and not hasattr(service,"holdings_path")
    assert service._selection_reasons()==[]


def test_process_population_progress_resets_each_fold_and_ignores_stock_details(tmp_path):
    def event(population,fold,completed):
        return "PROGRESS "+json.dumps(dict(stage="dataset",population=population,fold=fold,
            completed=completed,total=3,scope="current_population_only"))
    lines=["TRAIN_RUN_ID fresh-model",event("E1",None,0),event("E1",None,3),
        *["DATASET E1 000001.SZ entries=1 exits=0"]*12,
        event("final_population/first","first",0),event("final_population/first","first",3),
        event("final_population/production","production",0),event("final_population/production","production",2)]
    command=[sys.executable,"-u","-c","\n".join("print("+repr(line)+",flush=True)" for line in lines)]
    events=[]
    center.TrainingCenter._run_process(command,tmp_path/"progress.log",events.append,None,source_total=3)
    counts=[(item["population"],item["completed"],item["total"]) for item in events if "population" in item]
    assert counts==[("E1",0,3),("E1",3,3),("final_population/first",0,3),
        ("final_population/first",3,3),("final_population/production",0,3),("final_population/production",2,3)]
    assert all(item["completed"]<=item["total"] for item in events if "completed" in item)


@pytest.mark.parametrize("change",[{"completed":4},{"completed":-1},{"completed":True},
    {"total":4},{"population":"final_population/foreign"},{"scope":"all_folds_summed"}])
def test_process_rejects_wrong_population_progress_even_without_callback(tmp_path,change):
    event=dict(stage="dataset",population="final_population/production",fold="production",
        completed=1,total=3,scope="current_population_only")
    event.update(change)
    line="PROGRESS "+json.dumps(event)
    command=[sys.executable,"-u","-c","print("+repr(line)+",flush=True)"]
    with pytest.raises(ValueError,match="人口进度"):
        center.TrainingCenter._run_process(command,tmp_path/"invalid-progress.log",None,None,source_total=3)


def test_process_detail_logs_and_identity_updates_preserve_dataset_progress(tmp_path):
    lines=["STAGE E0_actual_rule_bootstrap_no_HR",
        "PROGRESS "+json.dumps(dict(stage="dataset",population="E0",fold=None,
            completed=2,total=3,scope="current_population_only")),
        "DATASET E0 000001.SZ entries=1 exits=0","TREND_HEAD detail","warning: detail","",
        "TRAIN_RUN_ID fresh-model"]
    command=[sys.executable,"-u","-c","\n".join("print("+repr(line)+",flush=True)" for line in lines)]
    events=[]
    ids=center.TrainingCenter._run_process(command,tmp_path/"detail.log",events.append,None,source_total=3)
    assert ids["model_run_id"]=="fresh-model"
    assert len(events)==3
    assert [(item["stage"],item["completed"],item["total"]) for item in events]==[
        ("E0_actual_rule_bootstrap_no_HR",0,0),("E0",2,3),("E0",2,3)]
    assert events[-1]["population"]=="E0" and events[-1]["children"]["model_run_id"]=="fresh-model"
    assert "warning: detail" in (tmp_path/"detail.log").read_text()


def test_process_diagnostic_progress_counts_only_frozen_fold_stocks(tmp_path):
    def event(fold,completed):
        return "DIAGNOSTIC_PROGRESS "+json.dumps(dict(stage="diagnostics",fold=fold,message=fold,
            scope="current_diagnostic_fold_stocks",completed=completed,total=3))
    lines=["STAGE diagnostics",event("first",3),"diagnostic detail",event("production",1),
        "DIAGNOSTIC_RUN_ID diagnostic-result"]
    command=[sys.executable,"-u","-c","\n".join("print("+repr(line)+",flush=True)" for line in lines)]
    events=[]
    center.TrainingCenter._run_process(command,tmp_path/"diagnostic-progress.log",events.append,None,source_total=3)
    assert [(item.get("fold"),item["completed"],item["total"]) for item in events]==[
        (None,0,0),("first",3,3),("production",1,3),("production",1,3)]
    assert events[-1]["children"]["diagnostic_run_id"]=="diagnostic-result"


@pytest.mark.parametrize("change",[{"total":4},{"completed":4},{"completed":True},
    {"scope":"diagnostic_rows"},{"scope":None},{"fold":None}])
def test_process_rejects_diagnostic_rows_or_wrong_frozen_population(tmp_path,change):
    event=dict(stage="diagnostics",fold="production",scope="current_diagnostic_fold_stocks",
        completed=1,total=3);event.update(change)
    line="DIAGNOSTIC_PROGRESS "+json.dumps(event)
    command=[sys.executable,"-u","-c","print("+repr(line)+",flush=True)"]
    with pytest.raises(ValueError,match="诊断进度"):
        center.TrainingCenter._run_process(command,tmp_path/"bad-diagnostic-progress.log",None,None,source_total=3)


def test_status_reports_effective_incumbent_date_while_replacement_waits(service,monkeypatch):
    old=dict(run_id="old-current",previous_run_id=None,effective_from="2020-01-01")
    pending=dict(run_id="new-current",previous_run_id="old-current",effective_from="2029-01-02")
    with sqlite3.connect(service.registry_path) as conn:
        conn.execute("INSERT INTO current_model_events(kind,payload,created_at) VALUES ('select',?,'then')",(json.dumps(old),))
        conn.execute("INSERT INTO current_model_events(kind,payload,created_at) VALUES ('select',?,'now')",(json.dumps(pending),))
        conn.execute("INSERT INTO current_model_selection VALUES (1,'new-current',2,'2029-01-02')")
    monkeypatch.setattr(service,"_event_proof",lambda identity:old if identity==1 else pending)
    assert service.status()["active_model"]["run_id"]=="old-current"
    assert service.status()["active_model"]["effective_from"]=="2020-01-01"


def test_selected_compact_graph_tamper_invalidates_cached_proof(verified_comparison):
    service,result,plan=verified_comparison
    service.select_candidate("candidate")
    assert service.active_model_run_id("2029-01-03")=="candidate"
    asset=result["windows"][0]["arms"]["candidate"]["assets"][0]
    reference=service.runs_root/"candidate"/asset["path"]
    envelope=json.loads(reference.read_text())
    blob=reference.parent/envelope["asset"]
    before=blob.stat(); content=bytearray(blob.read_bytes()); content[-8]^=1
    blob.write_bytes(content); os.utime(blob,ns=(before.st_atime_ns,before.st_mtime_ns))
    assert service.active_model_run_id("2029-01-03") is None
    assert service.status()["mode"]=="integrity_blocked"


def test_comparison_cannot_substitute_plain_archive_account(verified_comparison):
    service,result,plan=verified_comparison
    asset=result["windows"][0]["arms"]["candidate"]["assets"][0]
    reference=service.runs_root/"candidate"/asset["path"]
    logical=read_account_artifact(reference)
    center._save(reference,logical); asset["sha256"]=center._file_hash(reference)
    service.comparison=lambda _:result
    with pytest.raises(ValueError,match="schema|archive"):
        service.select_candidate("candidate")
    assert service.current_selection() is None


def test_reference_quality_progress_keeps_separate_complete_stock_populations(tmp_path):
    def event(stage,completed):
        return 'REFERENCE_QUALITY_PROGRESS '+json.dumps(dict(stage=stage,fold=None,
            scope='current_reference_population_stocks',completed=completed,total=3))
    lines=['STAGE reference_quality_population',event('reference_quality_population',3),
        'DATASET reference_quality 000001.SZ entries=12 exits=0',
        'STAGE reference_quality_fee_qualification',event('reference_quality_fee_qualification',1),
        'TRAIN_RUN_ID model-result']
    command=[sys.executable,'-u','-c','\n'.join('print('+repr(line)+',flush=True)' for line in lines)]
    events=[]
    center.TrainingCenter._run_process(command,tmp_path/'reference-progress.log',events.append,None,source_total=3)
    records=[e for e in events if e.get('scope')=='current_reference_population_stocks']
    assert [(e['stage'],e['completed'],e['total']) for e in records]==[
        ('reference_quality_population',3,3),('reference_quality_fee_qualification',1,3),
        ('reference_quality_fee_qualification',1,3)]
    assert all(e['device']=='cpu' for e in records)
    assert records[-1]['children']['model_run_id']=='model-result'


@pytest.mark.parametrize('change',[{'stage':'entry'},{'total':4},{'completed':4},
    {'completed':True},{'completed':-1},{'scope':'reference_rows'},{'fold':'production'}])
def test_reference_progress_rejects_wrong_stock_population_without_callback(tmp_path,change):
    value=dict(stage='reference_quality_population',fold=None,scope='current_reference_population_stocks',
        completed=1,total=3);value.update(change)
    line='REFERENCE_QUALITY_PROGRESS '+json.dumps(value)
    command=[sys.executable,'-u','-c','print('+repr(line)+',flush=True)']
    with pytest.raises(ValueError,match='参考质量进度'):
        center.TrainingCenter._run_process(command,tmp_path/'bad-reference-progress.log',None,None,source_total=3)


def test_reference_summary_reports_independent_fee_qualification_as_optional():
    report={'reference_quality':{'status':'unqualified','bundles':[{'name':'fold'}],
        'unavailable_heads':[{'fold':'production','reason':'both classes missing'}],
        'population':{'population':'account_free_daily_original_BUY_risk_probes','population_hash':'population'},
        'qualification':{'status':'rejected','passed':False,'path':'validation/reference_quality/qualification.json','sha256':'actual'}}}
    summary=center._reference_quality_summary(report)
    assert summary['heads']==1 and summary['status']=='unqualified' and not summary['fee_qualified']
    assert summary['fee_status']=='rejected' and summary['required_for_core_selection'] is False
    assert summary['population_hash']=='population' and summary['qualification_sha256']=='actual'
