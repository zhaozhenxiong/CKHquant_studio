"""Causal selection and three-class calibration regression checks."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS
from my_strategy.services import czsc_trend_model as trend


def _dataset():
    dates=pd.bdate_range("2023-01-02",periods=360)
    rng=np.random.default_rng(493)
    y=np.arange(len(dates))%3
    x=rng.normal(size=(len(dates),20));x[:,0]+=np.choose(y,[2.,-2.,0.])
    frame=pd.DataFrame(x,columns=MA_TREND_COLUMNS)
    frame["date"]=dates.strftime("%Y-%m-%d");frame["symbol"]="TEST.SZ"
    frame["label_end"]=(dates+pd.offsets.BDay(10)).strftime("%Y-%m-%d")
    frame["label"]=y;frame["input_eligible"]=True;frame["label_available"]=True
    frame["observation_id"]=[stable_hash({"date":d}) for d in frame.date]
    config={"trend":{"min_train_per_class":2,"min_calibration_per_class":2,"min_policy_per_class":2}}
    contract=trend.trend_contract(config,source_snapshot_hash="test-source",pipeline_contract_hash="test-pipeline")
    frame.attrs["label_contract"]=contract
    fold={"name":"causal","train_end":frame.date.iloc[219],"calibration_start":frame.date.iloc[220],
        "calibration_end":frame.date.iloc[279],"policy_start":frame.date.iloc[280],"policy_end":frame.date.iloc[339]}
    return frame,fold,config,contract


@pytest.mark.parametrize("temperature",[.0001,.2,1.,7.4,10000.])
def test_positive_temperature_keeps_every_argmax_including_ties(temperature):
    z=np.random.default_rng(19).normal(size=(1000,3))*50
    z=np.r_[z,[[1.,1.,-2.],[0.,0.,0.],[-1.,0.,0.]]]
    original=z.copy()
    assert np.array_equal(trend._softmax(z/temperature).argmax(axis=1),z.argmax(axis=1))
    trend._softmax(z)
    assert np.array_equal(z,original)


def test_calibration_fits_only_positive_scale_and_reports_all_classes():
    z=np.random.default_rng(18).normal(size=(500,3))*8
    y=np.arange(500)%3
    cal=trend._fit_temperature(z,y,np.linspace(.5,2.,500))
    assert cal["method"]==trend.CALIBRATION_METHOD
    assert cal["temperature"]>0 and cal["bias"]==[0.,0.,0.]
    assert cal["argmax_changed"]==0
    assert cal["objective_calibrated"]<=cal["objective_uncalibrated"]+1e-8
    metrics=trend.trend_metrics(y,trend._softmax(z/cal["temperature"]))
    assert set(metrics["per_class"])==set(trend.CLASSES)
    assert sum(metrics["predicted_class_counts"].values())==500
    assert sum(sum(row) for row in metrics["confusion"])==500


def test_policy_and_future_rows_cannot_change_fit_selection_or_calibration(tmp_path):
    frame,fold,config,contract=_dataset()
    before=trend.train_trend_head(frame,tmp_path/"before",fold,config,contract=contract)
    mutated=frame.copy();future=mutated.date>=fold["policy_start"]
    mutated.loc[future,list(MA_TREND_COLUMNS)]=1e8
    mutated.loc[future,"label"]=(mutated.loc[future,"label"]+1)%3
    after=trend.train_trend_head(mutated,tmp_path/"after",fold,config,contract=contract)
    assert before["preprocess"]==after["preprocess"]
    assert before["training_selection"]==after["training_selection"]
    assert before["class_weighting"]==after["class_weighting"]
    assert before["calibration"]==after["calibration"]
    first=joblib.load(tmp_path/"before/model.joblib");second=joblib.load(tmp_path/"after/model.joblib")
    assert np.array_equal(first.coef_,second.coef_)
    assert np.array_equal(first.intercept_,second.intercept_)
    assert set(before["raw_calibrated_metrics"])=={"train","calibration","policy"}
    assert before["calibration_argmax_changed"]=={"train":0,"calibration":0,"policy":0}
    assert before["training_selection"]["inner_train_label_end"]<=before["training_selection"]["inner_train_end"]
    assert before["training_selection"]["inner_validation_end"]<=fold["train_end"]


def test_calibration_features_and_labels_cannot_choose_training_candidate(tmp_path):
    frame,fold,config,contract=_dataset()
    first=trend.train_trend_head(frame,tmp_path/"a",fold,config,contract=contract)
    cal=(frame.date>=fold["calibration_start"])&(frame.date<fold["policy_start"])
    frame.loc[cal,list(MA_TREND_COLUMNS)]*=10
    frame.loc[cal,"label"]=(frame.loc[cal,"label"]+1)%3
    second=trend.train_trend_head(frame,tmp_path/"b",fold,config,contract=contract)
    assert first["training_selection"]==second["training_selection"]
    assert first["preprocess"]==second["preprocess"]
    assert np.array_equal(joblib.load(tmp_path/"a/model.joblib").coef_,joblib.load(tmp_path/"b/model.joblib").coef_)


def test_immature_future_tail_and_nonunique_dataframe_index_do_not_change_fit(tmp_path):
    frame,fold,config,contract=_dataset()
    first=trend.train_trend_head(frame,tmp_path/"a",fold,config,contract=contract)
    frame.loc[340:,"label_end"]=None
    frame.loc[340:,"label"]=np.nan
    frame.loc[340:,"label_available"]=False
    frame.index=np.zeros(len(frame),dtype=int)
    second=trend.train_trend_head(frame,tmp_path/"b",fold,config,contract=contract)
    assert first["training_selection"]==second["training_selection"]
    assert first["sample_counts"]==second["sample_counts"]
    assert first["calibration"]==second["calibration"]
    assert np.array_equal(joblib.load(tmp_path/"a/model.joblib").coef_,joblib.load(tmp_path/"b/model.joblib").coef_)


@pytest.mark.parametrize("change",[{"method":"separate_chronological_temperature_and_class_intercepts"},
    {"bias":[0.,-.1,0.]},{"temperature":0.},{"temperature":float("nan")}])
def test_retired_or_invalid_calibration_is_refused_even_with_valid_manifest_hash(tmp_path,change):
    frame,fold,config,contract=_dataset()
    manifest=trend.train_trend_head(frame,tmp_path/"model",fold,config,contract=contract)
    manifest["calibration"].update(change)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"]=stable_hash(manifest)
    (tmp_path/"model/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="positive-temperature zero-intercept"):
        trend.load_trend_head(tmp_path/"model")


def test_class_balancing_preserves_inverse_concurrency_and_uses_training_only():
    y=np.array([0,0,0,1,2]);information=np.array([.5,1.,1.5,2.,3.])
    factors=trend._class_factors(y,information,"information_balanced")
    resulting=np.bincount(y,weights=information*factors[y],minlength=3)
    assert np.allclose(resulting,resulting[0])
    assert np.allclose(trend._class_factors(y,information,"none"),1.)
