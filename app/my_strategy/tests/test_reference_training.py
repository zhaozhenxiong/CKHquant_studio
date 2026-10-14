import json
from pathlib import Path

import pytest

from my_strategy.services import reference_training as training


def test_reference_parameters_reject_retired_targets_and_bad_position_policy():
    with pytest.raises(ValueError,match="旧训练参数"):
        training.parameters({"min_net_return":.08})
    with pytest.raises(ValueError,match="滞回"):
        training.parameters({"hysteresis":.02})
    assert training.parameters()["device"] == "auto"


def test_windows_roll_and_never_include_incomplete_quarter():
    windows=training.development_windows("2026-10-12")
    assert [w["name"] for w in windows]==["2025Q4","2026Q1","2026Q2","2026Q3","production"]
    for window in windows:
        assert window["train_end"] < window["calibration_start"] <= window["calibration_end"] < window["policy_start"] <= window["policy_end"]
        if window["test_start"]:
            assert window["policy_end"] < window["test_start"] <= window["test_end"] < "2026-10-12"
    assert windows[-1]["policy_end"] == "2026-09-30"


def test_no_production_model_before_actual_freeze_and_reconstruction_explicit():
    folds=training.development_windows("2026-10-12")
    report={"folds":[{"name":f["name"],"split":f,"research_reconstruction":True} for f in folds[:-1]],
        "production":{"name":"production","split":folds[-1],"effective_from":"2026-10-15","research_reconstruction":False}}
    assert training.model_for_date(report,"2026-10-12","production") is None
    assert training.model_for_date(report,"2026-10-15","production")["research_reconstruction"] is False
    assert training.model_for_date(report,"2026-10-12","historical")["research_reconstruction"] is True
    assert training.model_for_date(report,"2025-01-01","historical") is None
    assert training.model_for_date(report,"2026-07-01","historical")["name"]=="2026Q3"


def test_predictive_gate_fails_closed_and_test_cannot_rescue_policy():
    assert not training._predictive_gate({})["passed"]
    policy={"log_loss":1.1,"brier":.6,"baseline_log_loss":1.,"baseline_brier":.65}
    assert not training._predictive_gate({"policy":policy,"test":{"log_loss":.1}})["passed"]
    policy["log_loss"]=.99
    assert training._predictive_gate({"policy":policy})["passed"]
    policy["baseline"]="frozen_train_class_prior"
    assert training._predictive_gate({"policy":policy})["passed"]


def test_asset_verification_catches_changed_raw_and_prevents_escape(tmp_path):
    runs=tmp_path/"runs";root=runs/"source";root.mkdir(parents=True)
    r={"symbol":"000001.SZ"}
    for name,path_key,hash_key in (("features","path","sha256"),("raw","raw_path","raw_sha256"),("liq","liquidity_path","liquidity_sha256")):
        p=root/name;p.write_text(name);r[path_key]=str(p);r[hash_key]=training._sha(p)
    assert set(training._verified_record(r,runs))=={"features","raw","liquidity"}
    (root/"raw").write_text("revised")
    with pytest.raises(ValueError,match="哈希不符"):
        training._verified_record(r,runs)


def test_profile_revision_and_roots_are_isolated(tmp_path):
    assert training.profiles(metadata_root=tmp_path)["default_revision"]==0
    training.save_default({"epochs":2},expected_revision=0,metadata_root=tmp_path)
    assert training.profiles(metadata_root=tmp_path)["default_profile"]["parameters"]["epochs"]==2
    with pytest.raises(ValueError,match="改变"):
        training.save_default({"epochs":3},expected_revision=0,metadata_root=tmp_path)


def test_validating_model_is_unavailable_to_normal_inference(tmp_path):
    runs=tmp_path/"runs";meta=tmp_path/"metadata"
    report={"family":training.FAMILY,"status":"validating","folds":[],
        "production":{"name":"production"}}
    training._save(runs/"candidate/reports/reference_training.json",report)
    with pytest.raises(ValueError,match="完整验证"):
        training.load_reference_run("candidate",runs_root=runs,metadata_root=meta)
    _,value=training.load_reference_run("candidate",runs_root=runs,metadata_root=meta,allow_validating=True)
    assert value["status"]=="validating"
