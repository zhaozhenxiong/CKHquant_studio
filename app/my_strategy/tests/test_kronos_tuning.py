import copy

import pandas as pd
import pytest

from my_strategy.services.kronos_tuning import DEFAULT_CONFIGURATIONS, tune_kronos


def cases():
    result = []
    for symbol in ("000001.SZ", "600519.SH"):
        for date in ("2026-01-12", "2026-02-12"):
            future = pd.bdate_range(pd.Timestamp(date) + pd.Timedelta(days=1), periods=5)
            result.append({"frame": pd.DataFrame(), "future_dates": future,
                           "actual_future": pd.DataFrame({"date": future, "close": 20.}),
                           "symbol": symbol, "as_of": date, "calendar_verified": True,
                           "calendar_identity": "fixture"})
    return result


def predictor(calls):
    def call(frame, future_dates, **kwargs):
        calls.append((kwargs["as_of"], kwargs["lookback"]))
        kwargs["cancel_check"]()
        value = 20. if kwargs["lookback"] == 128 else 21.
        return {"input": {"last_close": 19}, "cache_key": str(calls[-1]),
                "quality": {"prediction_usable": True, "valid_samples": 8, "requested_samples": 8},
                "summary": {"up_path_fraction": .8},
                "forecast": [{"close": value, "close_p10": value-1, "close_p90": value+1} for _ in future_dates]}
    return call


def test_selects_only_on_validation_and_tests_selected_config_once():
    calls = []
    original = tune_kronos(cases(), ["2026-01-12"], ["2026-02-12"], predict_fn=predictor(calls))
    assert original["selected_configuration"]["lookback"] == 128
    assert len([x for x in calls if x[0] == "2026-02-12"]) == 2
    assert all(x[1] == 128 for x in calls if x[0] == "2026-02-12")
    changed = cases()
    for case in changed:
        if case["as_of"] == "2026-02-12":
            case["actual_future"]["close"] = 21
    rerun = tune_kronos(changed, ["2026-01-12"], ["2026-02-12"], predict_fn=predictor([]))
    assert original["selected_configuration"] == rerun["selected_configuration"]
    assert not original["test_used_for_selection"]
    assert not original["qualification"]["positive_allocation_eligible"]


def test_all_failures_stay_in_requested_denominator():
    def failed(*args, **kwargs):
        raise ValueError("invalid price")
    result = tune_kronos(cases(), ["2026-01-12"], ["2026-02-12"], predict_fn=failed)
    assert result["selected_configuration"] is None
    for row in result["completed_comparisons"]:
        assert row["metrics"]["failed_cases"] == 2
        assert row["metrics"]["requested_cases"] == 2
        assert row["metrics"]["requested_paths"] == 16
        assert row["metrics"]["valid_path_fraction"] == 0


def test_boolean_cancellation_stops_without_test_calls():
    result = tune_kronos(cases(), ["2026-01-12"], ["2026-02-12"],
                         predict_fn=lambda *_args, **_kwargs: pytest.fail("cancelled"), stop=lambda: True)
    assert result["status"] == "cancelled"
    assert result["test_evaluation"] is None


def test_budget_exhaustion_reports_partial(monkeypatch):
    from my_strategy.services import kronos_tuning
    clock = iter([0, 2, 3])
    monkeypatch.setattr(kronos_tuning.time, "monotonic", lambda: next(clock))
    result = tune_kronos(cases(), ["2026-01-12"], ["2026-02-12"], budget_seconds=1,
                         predict_fn=lambda *_args, **_kwargs: pytest.fail("over budget"))
    assert result["status"] == "partial"
    assert result["reason"] == "forecast_tuning_budget_exhausted"


def test_validation_label_must_mature_before_test_boundary():
    data = cases()
    for case in data:
        if case["as_of"] == "2026-02-12":
            case["as_of"] = "2026-01-14"
    with pytest.raises(ValueError, match="matured"):
        tune_kronos(data, ["2026-01-12"], ["2026-01-14"], predict_fn=predictor([]))


def test_actual_future_dates_may_not_be_silently_filled():
    data = cases()
    data[0]["actual_future"] = data[0]["actual_future"].iloc[1:]
    with pytest.raises(ValueError, match="exactly match"):
        tune_kronos(data, ["2026-01-12"], ["2026-02-12"], predict_fn=predictor([]))


def test_configuration_limit_is_enforced_before_inference():
    with pytest.raises(ValueError, match="eight"):
        tune_kronos(cases(), ["2026-01-12"], ["2026-02-12"],
                     configurations=[dict(DEFAULT_CONFIGURATIONS[0])] * 9, predict_fn=predictor([]))
