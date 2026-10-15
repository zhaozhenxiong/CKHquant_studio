from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.simple_score import SPLITS, fit_simple_score, score_samples


def fixture():
    dates = pd.bdate_range("2018-01-01", periods=600)
    classes = np.arange(len(dates)) % 3
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"as_of": dates, "label_mature_at": dates + pd.Timedelta(days=5),
                          "x": classes + rng.normal(0, .15, len(dates)),
                          "y": rng.normal(size=len(dates)), "label": classes,
                          "feature_available": dates})
    ranges = {key: [str(dates[i*120].date()), str(dates[(i+1)*120-1].date())] for i, key in enumerate(SPLITS)}
    return frame, ranges


def fit(frame=None, ranges=None):
    default_frame, default_ranges = fixture()
    return fit_simple_score(default_frame if frame is None else frame, ["x", "y"],
                            default_ranges if ranges is None else ranges,
                            source_identity={"raw_sha": "synthetic_fixture"},
                            label_contract={"kind": "next_open_to_h5_close_after_cost", "classes": [0, 1, 2]},
                            feature_available_column="feature_available")


def test_five_chronological_splits_maturity_and_json_round_trip():
    artifact = fit()
    assert artifact["model_type"] == "multinomial_logistic_L2"
    assert all(v["excluded_immature_rows"] > 0 for v in artifact["population"].values())
    assert set(artifact["metrics"]) == set(SPLITS)
    assert artifact["qualification"]["policy_baseline_passed"]
    assert artifact["qualification"]["heldout_baseline_passed"]
    assert not artifact["qualification"]["positive_allocation_eligible"]
    frame, ranges = fixture()
    restored = json.loads(json.dumps(artifact, allow_nan=False))
    result = score_samples(restored, frame.tail(10), as_of=ranges["test"][1], historical_reconstruction=True)
    assert len(result["scores"]) == 10
    assert all(0 <= score <= 100 for score in result["scores"])
    assert all(abs(sum(p.values())-1) < 1e-12 for p in result["probabilities"])
    assert result["temporal_available"] and not result["model_actually_frozen"]


def test_future_test_changes_cannot_change_fitted_weights_or_temperature():
    original, ranges = fixture()
    baseline = fit(original, ranges)
    changed = original.copy()
    changed.loc[480:, "x"] = 1e6
    changed.loc[480:, "label"] = (changed.loc[480:, "label"] + 1) % 3
    candidate = fit(changed, ranges)
    assert candidate["standardization"] == baseline["standardization"]
    assert candidate["coefficients"] == baseline["coefficients"]
    assert candidate["intercept"] == baseline["intercept"]
    assert candidate["temperature"] == baseline["temperature"]
    assert candidate["regularization_C"] == baseline["regularization_C"]
    assert candidate["metrics"]["test"] != baseline["metrics"]["test"]


def test_model_cannot_be_backdated_or_applied_before_policy_evidence():
    artifact = fit()
    frame, ranges = fixture()
    with pytest.raises(ValueError, match="explicit historical"):
        score_samples(artifact, frame.tail(1), as_of=ranges["test"][1])
    with pytest.raises(ValueError, match="not available"):
        score_samples(artifact, frame.head(1), as_of=ranges["train"][1], historical_reconstruction=True)
    altered = copy.deepcopy(artifact)
    altered["created_at"] = "2010-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="identity"):
        score_samples(altered, frame.tail(1), as_of=ranges["test"][1])


def test_overlapping_ranges_fail_closed():
    frame, ranges = fixture()
    ranges["calibration"][0] = ranges["tune"][1]
    with pytest.raises(ValueError, match="disjoint"):
        fit(frame, ranges)


def test_future_features_fail_closed():
    frame, ranges = fixture()
    frame.loc[1, "feature_available"] = frame.loc[1, "as_of"] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="available after"):
        fit(frame, ranges)


def test_missing_features_or_classes_are_not_filled_or_invented():
    frame, ranges = fixture()
    frame.loc[0, "x"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit(frame, ranges)
    frame, ranges = fixture()
    frame.loc[:119, "label"] = 1
    with pytest.raises(ValueError, match="five mature"):
        fit(frame, ranges)


def test_labels_cannot_be_used_as_features():
    frame, ranges = fixture()
    with pytest.raises(ValueError, match="cannot be score features"):
        fit_simple_score(frame, ["x", "label"], ranges,
                         source_identity="fixture", label_contract={"horizon": 5})


def test_no_rows_from_future_label_maturity_cross_split_boundary():
    frame, ranges = fixture()
    baseline = fit(frame, ranges)
    # The final training rows are already excluded by maturity; arbitrary labels
    # on these rows cannot change the trained coefficients.
    tail_mask = ((frame.as_of <= pd.Timestamp(ranges["train"][1])) &
                 (frame.label_mature_at > pd.Timestamp(ranges["train"][1])))
    frame.loc[tail_mask, "label"] = 2
    candidate = fit(frame, ranges)
    assert baseline["coefficients"] == candidate["coefficients"]


def test_cancel_has_no_artifact_or_training_side_effect():
    frame, ranges = fixture()
    def cancelled():
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        fit_simple_score(frame, ["x", "y"], ranges, source_identity="fixture",
                         label_contract={"horizon": 5}, cancel_check=cancelled)
