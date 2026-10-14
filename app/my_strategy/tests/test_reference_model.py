"""Reference labels, chronological isolation, and physical model loading."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import reference_model as model


def feature_frame(n=640):
    calendar = pd.bdate_range("2022-01-03", periods=n).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(7)
    close = 100*np.exp(.08*np.sin(np.arange(n)/9))
    frame = pd.DataFrame({"symbol": "000001.SZ", "date": calendar, "source": "tushare",
        "has_trade_price": 1, "open": close, "high": close*1.003, "low": close*.997, "close": close,
        "model_input_eligible": True, "liquidity_pool_eligible": True,
        "available_at": [d+"T15:00:00+08:00" for d in calendar],
        "liquidity_observed_at": [d+"T15:00:00+08:00" for d in calendar],
        "structure_input_eligible": True, "structure_confirmed_at": [d+"T15:00:00+08:00" for d in calendar],
        "wyckoff_input_eligible": True, "wyckoff_available_at": [d+"T15:00:00+08:00" for d in calendar],
        "wyckoff_corporate_action_suspected": False})
    for column in model.FEATURE_COLUMNS:
        frame[column] = rng.normal(size=n).astype(np.float32)
    frame["atr14_ratio"] = .02
    return frame, calendar


def split_for(calendar):
    return {"name": "test_fold", "train_end": calendar[330], "calibration_start": calendar[331],
        "calibration_end": calendar[410], "policy_start": calendar[411], "policy_end": calendar[500],
        "test_start": calendar[501], "test_end": calendar[-1]}


def test_exact_next_open_label_and_maturity():
    frame, calendar = feature_frame()
    frame.loc[101, "open"] = frame.loc[101, "close"]*1.03
    result = model.make_observations(frame, calendar)
    row = result.iloc[100]
    assert row.label_start == calendar[101]
    assert row.label_end == calendar[110]
    assert row.future_log_return == pytest.approx(np.log(frame.loc[110, "close"]/frame.loc[101, "open"]))
    assert result.iloc[-10:].label_available.eq(False).all()
    assert result.iloc[-10:].label.isna().all()
    assert not result.iloc[:64].input_eligible.any()


@pytest.mark.parametrize("change", ["missing_date", "missing_quote", "corporate_action", "late_quote"])
def test_future_incomplete_is_unavailable_not_neutral(change):
    frame, calendar = feature_frame()
    if change == "missing_date":
        frame = frame.drop(index=105).reset_index(drop=True)
    elif change == "missing_quote":
        frame.loc[105, "has_trade_price"] = 0
    elif change == "corporate_action":
        frame.loc[105, "wyckoff_corporate_action_suspected"] = True
    else:
        frame.loc[105, "available_at"] = calendar[106]+"T15:00:00+08:00"
    result = model.make_observations(frame, calendar)
    row = result[result.date == calendar[100]].iloc[0]
    assert row.input_eligible
    assert not row.label_available
    assert pd.isna(row.label)


def test_prefix_inputs_and_optional_evidence_availability():
    frame, calendar = feature_frame()
    frame.loc[150, "structure_confirmed_at"] = calendar[151]+"T15:00:00+08:00"
    frame.loc[150, "wyckoff_input_eligible"] = False
    complete = model.prepare_inputs(frame, calendar)
    prefix = model.prepare_inputs(frame.iloc[:151].copy(), calendar[:151])
    pd.testing.assert_frame_equal(complete.iloc[:151], prefix, check_flags=False)
    assert complete.loc[150, list(model.STRUCTURE_COLUMNS)].isna().all()
    assert complete.loc[150, list(model.WYCKOFF_COLUMNS)].isna().all()
    assert complete.loc[150, "input_eligible"]
    frame.loc[150, "liquidity_observed_at"] = calendar[151]+"T15:00:00+08:00"
    assert not model.prepare_inputs(frame, calendar).loc[150, "input_eligible"]


def test_time_masks_purge_and_refuse_overlaps():
    frame, calendar = feature_frame(); observations = model.make_observations(frame, calendar)
    split = split_for(calendar); masks = model.time_masks(observations, split)
    total = sum(mask.astype(int) for mask in masks.values())
    assert total.max() == 1
    for segment, mask in masks.items():
        assert observations.loc[mask, "label_end"].le(split[segment+"_end"]).all()
    crossing = observations.date.le(split["train_end"]) & observations.label_end.gt(split["train_end"])
    assert crossing.any() and not masks["train"][crossing].any()
    no_test = model.time_masks(observations, {**split, "test_start": None, "test_end": None})
    assert not no_test["test"].any()
    assert model.evaluate_probabilities([], np.empty((0, 3)), device="cpu")["count"] == 0
    with pytest.raises(ValueError, match="overlap"):
        model.time_masks(observations, {**split, "policy_start": split["calibration_end"]})


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    frame, calendar = feature_frame(); observations = model.make_observations(frame, calendar)
    split = split_for(calendar)
    # A future-only distribution change must not affect fitted normalization.
    observations.loc[observations.date.ge(split["policy_start"]), "ma5_bias"] = 10000.
    result = model.fit_fold(observations, split, tmp_path_factory.mktemp("reference-model"), device="cpu",
        config={"epochs": 2, "inner_epochs": 1, "batch_size": 128, "min_train_per_class": 5, "min_segment_per_class": 2})
    return observations, split, result


def test_train_only_preprocessor_and_scoring(trained):
    data, split, result = trained
    session = model.load_model(result["model_path"], "cpu")
    mask = model.time_masks(data, split)["train"]
    expected = data.loc[mask, list(model.FEATURE_COLUMNS)].mean().to_numpy()
    np.testing.assert_allclose(session.mean.cpu().numpy(), expected, rtol=1e-5, atol=1e-6)
    validation = result["validation"]
    assert not validation["independent_forward_evidence"]
    assert validation["candidate_selection"]["source"] == "purged_train_internal_only"
    for name in ("calibration", "policy", "test"):
        assert {"log_loss", "brier", "baseline_log_loss", "baseline_brier", "macro_f1", "ece_10"} <= validation[name].keys()
    probabilities = session.predict(data)
    np.testing.assert_allclose(probabilities[data.input_eligible].sum(1), 1., atol=1e-6)
    assert np.isnan(probabilities[~data.input_eligible]).all()
    np.testing.assert_array_equal(model.predict(result["model_path"], data, "cpu"), probabilities)


def test_probability_metric_baseline():
    y = np.array([0, 0, 1, 2]); prior = np.array([.5, .25, .25])
    metrics = model.evaluate_probabilities(y, np.tile(prior, (4, 1)), prior, device="cpu")
    assert metrics["log_loss"] == pytest.approx(metrics["baseline_log_loss"])
    assert metrics["brier"] == pytest.approx(metrics["baseline_brier"])
    assert metrics["count"] == 4


def test_loading_checks_manifest_and_weights(trained, tmp_path):
    _, _, result = trained
    copied = tmp_path/"model"; shutil.copytree(result["model_path"], copied)
    manifest_path = copied/"manifest.json"
    original = manifest_path.read_text(); value = json.loads(original)
    value["classes"] = ["DOWN", "NEUTRAL", "UP"]
    manifest_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        model.load_model(copied, "cpu")
    manifest_path.write_text(original)
    with open(copied/"weights.pt", "ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="weights hash"):
        model.load_model(copied, "cpu")


def test_explicit_device_unavailable_is_not_cpu_fallback(monkeypatch, tmp_path):
    from my_strategy.core import device
    monkeypatch.setattr(device, "is_mps_available", lambda torch=None: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        model.load_model(tmp_path, "mps")


def test_mps_inference_parity_if_available(trained):
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    data, _, result = trained
    cpu = model.predict(result["model_path"], data, "cpu")
    mps = model.predict(result["model_path"], data, "mps")
    np.testing.assert_allclose(cpu, mps, rtol=2e-5, atol=2e-6, equal_nan=True)
