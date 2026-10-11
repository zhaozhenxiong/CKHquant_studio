"""Real small CPU fits exercise grouped causal boundaries and model contracts."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_high_return_features import ENTRY_FEATURE_COLUMNS, L1_MA_ENTRY_COLUMNS
from my_strategy.services.czsc_high_return_models import (
    HeadSession, _group_ids, _weights, bind_inputs, load_head, schema_hash,
    split_validation, time_masks, train_head,
)
from my_strategy.services.czsc_research_ml import _network, _transform


CONFIG = {"seed": 42, "epochs": 30, "batch_size": 64,
    "min_train_rows": 100, "min_calibration_rows": 30}


def contract_for(head="ma_entry", columns=L1_MA_ENTRY_COLUMNS, target=None, source="frozen_synthetic_source"):
    targets = {"ma_entry": "actual_complete_fee_net_roundtrip_positive",
        "return": "actual_complete_fee_net_roundtrip_return", "loss": "actual_complete_fee_net_roundtrip_loss",
        "entry": "high_return_and_initial_risk", "ma_exit": "exit_next_open_minus_frozen_continue",
        "exit": "exit_next_open_minus_frozen_continue", "exit_advantage": "exit_next_open_minus_frozen_continue"}
    value = {"target": target or targets[head], "source_snapshot_hash": source,
        "model_head": head, "model_input_schema_hash": schema_hash(columns, head),
        "label_version": "synthetic_actual_episode_v1", "behavior_policy": "causal_rule_v0",
        "execution": "next_verified_open_actual_broker", "calendar_hash": "synthetic_verified_calendar"}
    value["contract_hash"] = stable_hash(value)
    return value


def training_data(head="ma_entry", columns=L1_MA_ENTRY_COLUMNS, continuous=False):
    dates = pd.bdate_range("2024-01-02", periods=240)
    ordinal = np.arange(len(dates))
    label = ((ordinal // 2) % 2).astype(float)
    signal = label * 2 - 1
    frame = pd.DataFrame({column: signal * (1 + number / 20) + np.sin(ordinal / 11) * .01
        for number, column in enumerate(columns)})
    frame["date"] = dates.strftime("%Y-%m-%d")
    frame["symbol"] = "000001.SZ"
    frame["episode_id"] = [f"episode-{index//2}" for index in ordinal]
    frame["economic_group_id"] = frame.episode_id
    frame["opportunity_id"] = [f"opportunity-{index//2}" for index in ordinal]
    frame["label"] = signal * .12 if continuous else label
    frame["label_end"] = [dates[index+2].strftime("%Y-%m-%d") if index+2 < len(dates) else None for index in ordinal]
    frame["label_available"] = ordinal + 2 < len(dates)
    frame["label_available_at"] = [day + "T09:30:00+08:00" if isinstance(day, str) else None for day in frame.label_end]
    frame["input_eligible"] = True
    frame.loc[~frame.label_available, "label"] = np.nan
    contract = contract_for(head, columns)
    frame.attrs["label_contract"] = contract
    frame.attrs["label_version"] = contract["label_version"]
    fold = split_validation({"train_end": frame.date.iloc[119],
        "validation_start": frame.date.iloc[120], "validation_end": frame.date.iloc[199]},
        frame.date.tolist())
    return frame, fold, contract


@pytest.fixture(scope="module", params=("linear", "tree", "mlp"))
def fitted_head(request, tmp_path_factory):
    from threadpoolctl import threadpool_limits
    frame, fold, contract = training_data()
    path = tmp_path_factory.mktemp("real-causal-heads") / request.param
    # This tiny real fit needs one CPU thread; large OpenMP pools dominate its
    # work and needlessly contend with the user's other research tasks.
    with threadpool_limits(limits=1):
        manifest = train_head(frame, L1_MA_ENTRY_COLUMNS, path, fold, CONFIG,
            head="ma_entry", architecture=request.param, device="cpu", contract=contract)
    return frame, fold, contract, path, manifest


def future_inputs(frame):
    result = frame.loc[frame.date > frame.date.iloc[199]].copy()
    return bind_inputs(result, L1_MA_ENTRY_COLUMNS, "ma_entry")


def test_real_training_calibration_and_obvious_signal(fitted_head):
    frame, fold, contract, path, manifest = fitted_head
    train, calibration, policy = time_masks(frame, fold)
    assert manifest["actual_device"] == "cpu" and not manifest["actual_mps_training"]
    assert manifest["training_batches"] > 0 and manifest["publication_allowed"] is False
    assert manifest["sample_counts"] == {"train": int(train.sum()), "calibration": int(calibration.sum()),
        "policy": int(policy.sum()), "all": len(frame)}
    assert manifest["calibration"]["method"] == "separate_chronological_platt"
    assert manifest["train_label_end"] <= fold["train_end"]
    assert manifest["calibration_label_end"] <= fold["calibration_end"]
    values = future_inputs(frame)
    probability = HeadSession().predict(values, path, expected_head="ma_entry", expected_contract=contract)
    label = ((np.arange(200, 240) // 2) % 2).astype(bool)
    assert np.isfinite(probability).all()
    assert probability[label].mean() > probability[~label].mean() + .4


def test_future_values_and_outcomes_do_not_fit_training_preprocess(tmp_path):
    frame, fold, contract = training_data()
    changed = frame.copy(deep=True)
    future = changed.date > fold["train_end"]
    changed.loc[future, list(L1_MA_ENTRY_COLUMNS)] = 1_000_000.
    changed.loc[future & changed.label_available, "label"] = 1 - changed.loc[future & changed.label_available, "label"]
    first = train_head(frame, L1_MA_ENTRY_COLUMNS, tmp_path / "first", fold, CONFIG,
        head="ma_entry", device="cpu", contract=contract)
    second = train_head(changed, L1_MA_ENTRY_COLUMNS, tmp_path / "changed", fold, CONFIG,
        head="ma_entry", device="cpu", contract=contract)
    assert first["preprocess"] == second["preprocess"]
    assert json.loads((tmp_path / "first/linear-layers.json").read_text()) == json.loads(
        (tmp_path / "changed/linear-layers.json").read_text())


@pytest.mark.parametrize("field,value", [
    ("train_end", "2025-01-01"), ("calibration_end", "2025-01-01"),
    ("policy_start", "2024-01-01"), ("policy_end", "2025-01-01"),
])
def test_overlapping_or_reversed_fold_rejected(field, value):
    frame, fold, _ = training_data()
    fold[field] = value
    with pytest.raises(ValueError, match="chronology"):
        time_masks(frame, fold)


def test_blank_episode_falls_back_to_opportunity_or_individual_row():
    frame = pd.DataFrame({"symbol": ["000001.SZ"] * 4, "episode_id": ["", None, "", None],
        "economic_group_id": [None] * 4, "opportunity_id": ["same", "same", None, ""]})
    groups = _group_ids(frame)
    assert groups.iloc[0] == groups.iloc[1]
    assert len(set(groups.iloc[2:])) == 2 and groups.iloc[2] != groups.iloc[0]
    weights = _weights(frame, np.ones(len(frame), dtype=bool))
    assert weights[0] == weights[1] and weights[2] == pytest.approx(weights[0] * 2)


def test_all_branches_of_cross_boundary_economic_episode_are_purged():
    frame, fold, _ = training_data()
    # Different policy branches retain one economic identity across the cutoff.
    frame.loc[[116, 120], "episode_id"] = ["policy-a", "policy-b"]
    frame.loc[[116, 120], "economic_group_id"] = "shared-candidate-state"
    train, calibration, policy = time_masks(frame, fold)
    assert not train[116] and not calibration[120]
    assert not train[120] and not calibration[116]
    assert not policy[[116, 120]].any()
    assert train[112] and calibration[124]


def test_nan_and_immature_branch_make_entire_episode_unavailable():
    frame, fold, _ = training_data()
    frame.loc[20, "label"] = np.nan
    frame.loc[30, ["label_available", "label_end"]] = [False, None]
    train, calibration, policy = time_masks(frame, fold)
    assert not train[[20, 21, 30, 31]].any()
    assert not any(mask[-2:].any() for mask in (train, calibration, policy))
    assert train[24]


def test_policy_scoring_is_explicit_and_forward_does_not_replay_validation(fitted_head):
    frame, fold, contract, path, _ = fitted_head
    sample = bind_inputs(frame.loc[(frame.date >= fold["policy_start"]) &
        (frame.date <= fold["policy_end"])].copy(), L1_MA_ENTRY_COLUMNS, "ma_entry")
    session = HeadSession()
    with pytest.raises(ValueError, match="unavailable"):
        session.predict(sample, path, expected_contract=contract)
    probability = session.predict(sample, path, expected_contract=contract, validation=True)
    assert np.isfinite(probability).all()
    calibration = bind_inputs(frame.loc[frame.date == fold["calibration_end"]].copy(), L1_MA_ENTRY_COLUMNS, "ma_entry")
    with pytest.raises(ValueError, match="unavailable"):
        session.predict(calibration, path, expected_contract=contract, validation=True)


def test_production_requires_actual_training_completion(fitted_head):
    frame, _, contract, path, _ = fitted_head
    values = future_inputs(frame)
    session = HeadSession()
    assert np.isfinite(session.predict(values, path, expected_contract=contract,
        usage_mode="historical_reconstruction")).all()
    with pytest.raises(ValueError, match="actually completed"):
        session.predict(values, path, expected_contract=contract, usage_mode="production")


def test_source_contract_head_schema_and_missing_contract_are_rejected(fitted_head):
    frame, _, contract, path, _ = fitted_head
    values = future_inputs(frame)
    changed = contract_for(source="different_source")
    session = HeadSession()
    with pytest.raises(ValueError, match="contract"):
        session.predict(values, path, expected_contract=changed)
    with pytest.raises(ValueError, match="head"):
        session.predict(values, path, expected_contract=contract, expected_head="ma_exit")
    values.attrs["schema_hash"] = "different_schema"
    with pytest.raises(ValueError, match="schema"):
        session.predict(values, path, expected_contract=contract)
    values = future_inputs(frame)
    values.attrs.pop("label_contract")
    with pytest.raises(ValueError, match="contract"):
        session.predict(values, path)


def test_prediction_nonfinite_inputs_use_training_imputation(fitted_head):
    frame, _, contract, path, _ = fitted_head
    missing = future_inputs(frame).iloc[:4].copy()
    invalid = missing.copy()
    missing.loc[:, L1_MA_ENTRY_COLUMNS[0]] = np.nan
    invalid.loc[:, L1_MA_ENTRY_COLUMNS[0]] = np.inf
    session = HeadSession()
    np.testing.assert_allclose(session.predict(missing, path, expected_contract=contract),
        session.predict(invalid, path, expected_contract=contract), rtol=0, atol=0)


def test_training_rejects_foreign_label_contract_and_future_feature_columns(tmp_path):
    frame, fold, contract = training_data()
    foreign = contract_for(source="another_training_source")
    with pytest.raises(ValueError, match="contract"):
        train_head(frame, L1_MA_ENTRY_COLUMNS, tmp_path / "foreign", fold, CONFIG,
            head="ma_entry", device="cpu", contract=foreign)
    extended = (*L1_MA_ENTRY_COLUMNS, "label")
    with pytest.raises(ValueError, match="feature columns"):
        bind_inputs(frame, extended, "ma_entry")
    with pytest.raises(ValueError, match="feature columns"):
        train_head(frame, tuple(reversed(L1_MA_ENTRY_COLUMNS)), tmp_path / "reordered", fold, CONFIG,
            head="ma_entry", device="cpu", contract=contract)
    old_target = contract_for(target="fixed_10_sessions_net_positive")
    frame.attrs["label_contract"] = old_target
    with pytest.raises(ValueError, match="fixed-horizon"):
        train_head(frame, L1_MA_ENTRY_COLUMNS, tmp_path / "old-target", fold, CONFIG,
            head="ma_entry", device="cpu", contract=old_target)


def test_cpu_export_matches_real_torch_inference(fitted_head):
    import torch
    frame, _, contract, path, manifest = fitted_head
    if manifest["architecture"] == "tree":
        pytest.skip("tree uses its verified estimator rather than Torch export")
    payload = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    network = _network(len(L1_MA_ENTRY_COLUMNS), payload["hidden_sizes"])
    network.load_state_dict(payload["state_dict"])
    network.eval()
    values = future_inputs(frame)
    array = _transform(values[list(L1_MA_ENTRY_COLUMNS)].to_numpy(float), manifest["preprocess"])
    with torch.no_grad():
        logits = network(torch.from_numpy(array)).numpy().ravel()
    calibrated = 1 / (1 + np.exp(-np.clip(logits * manifest["calibration"]["slope"] +
        manifest["calibration"]["intercept"], -40, 40)))
    actual = HeadSession().predict(values, path, expected_contract=contract)
    np.testing.assert_allclose(actual, calibrated, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("asset", ["manifest", "checkpoint", "export"])
def test_manifest_checkpoint_export_tampering_is_detected(fitted_head, tmp_path, asset):
    _, _, _, path, manifest = fitted_head
    if asset == "export" and manifest["architecture"] == "tree":
        pytest.skip("tree has no linear export")
    copy_path = tmp_path / "tampered"
    shutil.copytree(path, copy_path)
    if asset == "manifest":
        data = json.loads((copy_path / "manifest.json").read_text())
        data["calibration"]["intercept"] += .1
        (copy_path / "manifest.json").write_text(json.dumps(data))
    else:
        destination = copy_path / (manifest["checkpoint"] if asset == "checkpoint" else "linear-layers.json")
        with destination.open("ab") as output:
            output.write(b" tampered")
    with pytest.raises(ValueError, match="integrity|identity"):
        load_head(copy_path)


def test_loading_checks_output_role_even_when_manifest_hash_is_consistent(fitted_head, tmp_path):
    _, _, _, path, _ = fitted_head
    copy_path = tmp_path / "wrong-output-role"
    shutil.copytree(path, copy_path)
    manifest = json.loads((copy_path / "manifest.json").read_text())
    manifest["regression"] = True
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = stable_hash(manifest)
    (copy_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="regression|role|binary|continuous"):
        load_head(copy_path)


def test_binary_and_continuous_head_roles_are_frozen(tmp_path):
    frame, fold, contract = training_data()
    with pytest.raises(ValueError, match="regression|role|binary|continuous"):
        train_head(frame, L1_MA_ENTRY_COLUMNS, tmp_path / "wrong-binary-role", fold, CONFIG,
            head="ma_entry", regression=True, device="cpu", contract=contract)
    continuous, fold, contract = training_data("return", ENTRY_FEATURE_COLUMNS, continuous=True)
    with pytest.raises(ValueError, match="regression|role|binary|continuous"):
        train_head(continuous, ENTRY_FEATURE_COLUMNS, tmp_path / "wrong-continuous-role", fold, CONFIG,
            head="return", regression=False, device="cpu", contract=contract)
    path = tmp_path / "continuous"
    manifest = train_head(continuous, ENTRY_FEATURE_COLUMNS, path, fold, CONFIG,
        head="return", regression=True, device="cpu", contract=contract)
    assert manifest["regression"] and manifest["calibration"]["method"] == "none_continuous_target"
    values = bind_inputs(continuous.loc[continuous.date > fold["validation_end"]].copy(), ENTRY_FEATURE_COLUMNS, "return")
    predicted = HeadSession().predict(values, path, expected_contract=contract, expected_head="return")
    assert np.isfinite(predicted).all()
    assert predicted.min() < 0 < predicted.max()
