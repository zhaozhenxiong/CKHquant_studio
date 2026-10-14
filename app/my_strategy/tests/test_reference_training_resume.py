"""Same-run recovery binds original inputs and never rewrites completed weights."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import reference_model as model
from my_strategy.services import reference_training as training
from my_strategy.services import reference_training_resume as resume


def _write_model(directory, split, config, trained_at):
    import torch
    directory.mkdir(parents=True, exist_ok=True)
    net = model._network(torch, "linear", len(model.FEATURE_COLUMNS))
    torch.save({"state_dict":net.state_dict(), "mean":torch.zeros(len(model.FEATURE_COLUMNS)),
                "scale":torch.ones(len(model.FEATURE_COLUMNS))}, directory/"weights.pt")
    validation = dict(split=split, policy=dict(log_loss=1., brier=.60, baseline_log_loss=1.1,
        baseline_brier=.67, baseline="frozen_train_class_prior"))
    manifest = dict(family=model.FAMILY, schema_hash=model.SCHEMA_HASH, feature_columns=list(model.FEATURE_COLUMNS),
        classes=list(model.CLASSES), architecture="linear", config=model._config(config), split=split,
        temperature=1., validation=validation, actual_trained_at=trained_at, weights_file="weights.pt",
        weights_sha256=training._sha(directory/"weights.pt"))
    manifest["manifest_hash"] = stable_hash(manifest)
    training._save(directory/"manifest.json", manifest)
    return dict(manifest=manifest, validation=validation, model_path=str(directory))


@pytest.fixture
def original(tmp_path, monkeypatch):
    runs = tmp_path/"runs"; root = runs/"original"; meta = tmp_path/"metadata"
    cfg = training.parameters({"end":"2026-10-12", "source_native_run_id":"native", "cpu_workers":1})
    source = {"source_native_run_id":"native", "proof":"frozen"}
    windows = [training.development_windows(cfg["end"])[0], training.development_windows(cfg["end"])[-1]]
    code = {}
    for name in ("reference_model.py", "reference_training.py", "reference_runtime.py"):
        src = Path(model.__file__).parent/name; dest=root/"config/frozen_code"/name
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest); code[name]=training._sha(src)
    plan = dict(plan_id="original-plan", family=training.FAMILY, parameters=cfg, source_reuse=source,
        cohort=["000001.SZ","000002.SZ"], code_hashes=code, actual_device="cpu", windows=windows)
    plan["plan_hash"] = stable_hash(plan)
    training._save(root/"config/training_plan.json", plan)
    training._save(root/"metadata.json", dict(run_id="original", config_hash=stable_hash(cfg)))
    started = "2026-10-14T10:00:00+08:00"; ended = "2026-10-14T11:00:00+08:00"
    records = []; source_records=[]
    for symbol in plan["cohort"]:
        data = pd.DataFrame(np.zeros((2,len(model.FEATURE_COLUMNS)),dtype=np.float32),columns=model.FEATURE_COLUMNS)
        data["symbol"] = symbol; data["date"]=["2023-01-03","2023-01-04"]
        data["input_eligible"]=True; data["label_available"]=True; data["label"]=[0.,1.]
        data["label_end"]=["2023-01-17","2023-01-18"]
        data.attrs.update(schema_hash=model.SCHEMA_HASH)
        path = root/"dataset"/(symbol+".parquet"); path.parent.mkdir(parents=True,exist_ok=True)
        data.to_parquet(path,index=False)
        records.append(dict(symbol=symbol,path=str(path),sha256=training._sha(path),source_sha256="source-sha",rows=2,eligible=2,labels=2))
        source_records.append(dict(symbol=symbol,sha256="source-sha"))
    manifest = dict(family=training.FAMILY, source_reuse=source, code_hash=code["reference_model.py"],records=records)
    training._save(root/"dataset/manifest.json",manifest)
    report = dict(run_id="original", family=training.FAMILY, plan_id=plan["plan_id"], status="failed",
        config=cfg, source_reuse=source, started_at=started, ended_at=ended, historical_union_count=2,
        dataset=dict(rows=4,eligible=4,labeled=4,stocks=2,manifest_sha256=training._sha(root/"dataset/manifest.json")),
        folds=[],production=None,error="AttributeError: str has no attribute get")
    training._save(root/"reports/reference_training.json",report)
    native=dict(selected_symbols=plan["cohort"],dataset_records=source_records)
    calendar=["2026-10-14","2026-10-15","2026-12-31","2027-01-04"]
    monkeypatch.setattr(training,"_source_header",lambda *args:(native,calendar,source))
    monkeypatch.setattr(training,"_verified_record",lambda *args: {})
    return SimpleNamespace(root=root,runs=runs,meta=meta,plan=plan,cfg=cfg,report=report,manifest=manifest)


def test_predictive_gate_accepts_named_baseline_and_fails_closed():
    p=dict(log_loss=1.,brier=.6,baseline_log_loss=1.1,baseline_brier=.67,baseline="frozen_train_class_prior")
    assert resume.predictive_gate({"policy":p})["passed"]
    assert not resume.predictive_gate({"policy":{**p,"log_loss":1.2},"test":{"log_loss":.1}})["passed"]
    assert not resume.predictive_gate({})["passed"]


def test_only_failed_same_family_run_can_resume(original):
    report=original.report.copy();report["status"]="running"
    training._save(original.root/"reports/reference_training.json",report)
    with pytest.raises(ValueError,match="不活跃"):
        resume.resume_training("original",runs_root=original.runs,metadata_root=original.meta)
    assert not (original.root/"config/resume_attempts").exists()


def test_original_frozen_code_cannot_change(original):
    (original.root/"config/frozen_code/reference_model.py").write_text("changed")
    with pytest.raises(ValueError,match="冻结源码哈希"):
        resume._read_original(original.root,"original")


def test_dataset_hash_and_population_are_bound(original):
    data,_,_=resume._verify_dataset(original.root,original.plan,original.report,original.runs,lambda *a,**k:None,None)
    assert len(data)==4
    path=Path(original.manifest["records"][0]["path"])
    with path.open("ab") as stream:stream.write(b"tamper")
    with pytest.raises(ValueError,match="SHA 改变"):
        resume._verify_dataset(original.root,original.plan,original.report,original.runs,lambda *a,**k:None,None)


def test_completed_model_requires_original_completion_event(original):
    split=original.plan["windows"][0];directory=original.root/"models"/split["name"]
    _write_model(directory,split,original.cfg,"2026-10-14T10:30:00+08:00")
    with pytest.raises(ValueError,match="完成与时间归属"):
        resume._reusable_fold(original.root,split,original.cfg,original.report,original.plan,[])
    event=dict(stage="fold_complete",fold=split["name"],current=1,total=1,at="2026-10-14T10:30:01+08:00")
    result=resume._reusable_fold(original.root,split,original.cfg,original.report,original.plan,[event])
    assert result["reuse_proof"]["dataset_manifest_sha256"]==original.report["dataset"]["manifest_sha256"]
    with pytest.raises(ValueError,match="配置或时间"):
        resume._reusable_fold(original.root,{**split,"train_end":"2025-05-30"},original.cfg,original.report,original.plan,[event])


def test_resume_reuses_completed_weights_cleans_partial_and_selects_only_after_validation(original,monkeypatch):
    first,production=original.plan["windows"]
    directory=original.root/"models"/first["name"]
    _write_model(directory,first,original.cfg,"2026-10-14T10:30:00+08:00")
    immutable=training._sha(directory/"weights.pt")
    logs=original.root/"logs";logs.mkdir()
    (logs/"training.jsonl").write_text(json.dumps(dict(stage="fold_complete",fold=first["name"],current=1,total=1,at="2026-10-14T10:30:01+08:00"))+"\n")
    partial=original.root/"models"/"production";partial.mkdir();(partial/"weights.pt.tmp").write_text("half")
    trained=[]
    def fit(data,split,path,**kwargs):
        trained.append(split["name"])
        assert not (path/"weights.pt.tmp").exists()
        return _write_model(path,split,original.cfg,"2026-10-14T12:00:00+08:00")
    monkeypatch.setattr(model,"fit_fold",fit)
    evaluated=[]
    def evaluate(run_id,**kwargs):
        assert not (original.meta/"reference_current.json").exists()
        report=training._read(original.root/"reports/reference_training.json")
        assert report["status"]=="validating"
        assert report["technical_validation"]["status"]=="pending"
        assert "completed_at" not in report
        evaluated.append(run_id)
    monkeypatch.setattr(training,"evaluate",evaluate)
    result=resume.resume_training("original",runs_root=original.runs,metadata_root=original.meta)
    assert trained==["production"] and evaluated==["original"]
    assert result["status"]=="completed" and result["original_plan_id"]==original.plan["plan_id"]
    assert training._sha(directory/"weights.pt")==immutable
    assert training._read(original.meta/"reference_current.json")["run_id"]=="original"
    attempts=list((original.root/"config/resume_attempts").glob("*/attempt.json"))
    attempt=training._read(attempts[0])
    assert attempt["removed_partial_folds"]==["production"]
    assert attempt["reused_folds"][0]["name"]==first["name"]
    assert attempt["status"]=="completed"


def test_validation_failure_never_publishes_success(original,monkeypatch):
    def fit(data,split,path,**kwargs):
        return _write_model(path,split,original.cfg,"2026-10-14T12:00:00+08:00")
    monkeypatch.setattr(model,"fit_fold",fit)
    def fail(*args,**kwargs):
        report=training._read(original.root/"reports/reference_training.json")
        assert report["status"]=="validating"
        assert report["technical_validation"]["status"]=="pending"
        assert "completed_at" not in report
        raise ValueError("dataset audit failed")
    monkeypatch.setattr(training,"evaluate",fail)
    with pytest.raises(ValueError,match="dataset audit failed"):
        resume.resume_training("original",runs_root=original.runs,metadata_root=original.meta)
    assert not (original.meta/"reference_current.json").exists()
    report=training._read(original.root/"reports/reference_training.json")
    assert report["status"]=="failed"
    assert report["technical_validation"]["status"]=="failed"
    assert "completed_at" not in report
    assert report["resume_attempts"][-1]["status"]=="failed"


def test_same_run_lease_rejects_second_owner(original):
    with resume._lease(original.root):
        with pytest.raises(ValueError,match="持有租约"):
            with resume._lease(original.root):
                pytest.fail("second owner entered")


def test_new_dispatch_plan_must_preserve_original_population(original,monkeypatch):
    dispatch={**original.plan,"plan_id":"repair-plan","cohort":["different.SZ"]}
    monkeypatch.setattr(training,"validate_plan",lambda *a,**kw:dispatch)
    with pytest.raises(ValueError,match="人口或窗口不同"):
        resume.resume_training("original",plan_id="repair-plan",runs_root=original.runs,metadata_root=original.meta)
    assert not (original.root/"config/resume_attempts").exists()
