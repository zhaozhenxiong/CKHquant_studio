"""Exact native equivalence and rejection checks for reusable W preparation."""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import czsc_native_evidence as current
from my_strategy.services.czsc_market_features import build_features
from my_strategy.services.czsc_market_profiles import bind_feature_profile, prepare_structure_inputs


def _raw(n=230):
    rng = np.random.default_rng(412)
    close = 17 * np.exp(np.cumsum(rng.normal(.0002, .025, n)))
    volume = rng.uniform(600_000, 2_000_000, n)
    frame = pd.DataFrame({"symbol": "000001.SZ", "date": pd.bdate_range("2020-01-02", periods=n),
        "open": close * .998, "high": close * 1.009, "low": close * .99, "close": close,
        "volume": volume, "amount": close * volume, "source": "sina_unadjusted_v1", "has_trade_price": 1},
        index=np.arange(n) * 3 + 5)
    frame.attrs.update(price_basis="unadjusted", volume_unit="shares", amount_unit="CNY",
        verified_market_dates=frame.date.dt.strftime("%Y-%m-%d").tolist())
    return frame


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def baseline():
    default = Path(__file__).resolve().parents[2].parent / "baseline"
    root = Path(os.environ.get("KHQUANT_NATIVE_BASELINE", default))
    if not (root / "my_strategy/services/czsc_native_evidence.py").is_file():
        pytest.skip("Frozen pre-change baseline is supplied for audit equivalence runs")
    analysis = _load(root / "my_strategy/services/czsc_analysis.py", "native_performance_baseline_analysis")
    native = _load(root / "my_strategy/services/czsc_native_evidence.py", "native_performance_baseline_native")
    native.CzscSession = analysis.CzscSession
    native._pen_key = analysis._pen_key
    return native


def _assert_pack_equal(actual, expected):
    pd.testing.assert_frame_equal(actual["frame"], expected["frame"], check_exact=True)
    assert actual["frame"].attrs == expected["frame"].attrs
    assert {key: value for key, value in actual.items() if key != "frame"} == {
        key: value for key, value in expected.items() if key != "frame"}


@pytest.mark.parametrize("config", [{}, {"native_evidence": {"structure_zone_strength": .61}},
    {"wyckoff": {"quality": {"minimum_active_bars": 70}}}])
def test_native_matches_frozen_baseline_for_default_and_custom_config(baseline, config):
    raw = _raw()
    expected = baseline.build_native_evidence(raw, config)
    _assert_pack_equal(current.build_native_evidence(raw, config), expected)


def test_native_matches_frozen_baseline_with_gaps_and_suspension(baseline):
    raw = _raw()
    raw = raw.drop(raw.index[135]).copy()
    raw.loc[raw.index[160], ["volume", "amount"]] = 0.
    _assert_pack_equal(current.build_native_evidence(raw), baseline.build_native_evidence(raw))


def test_small_aggregation_buffer_preserves_closed_history_and_position(baseline):
    bars = current._validate_input(_raw())
    settings = current.strategy_config()
    expected = baseline.CzscSession(str(bars.symbol.iloc[0]), settings, len(bars))
    actual = current.CzscSession(str(bars.symbol.iloc[0]), settings, len(bars))
    fields = ("id", "dt", "open", "high", "low", "close", "vol", "amount")
    for row in bars.itertuples(index=False):
        expected.update(row)
        actual.update(row)
        for frequency in ("日线", "周线", "月线"):
            current_bars, original_bars = actual.bg.bars[frequency], expected.bg.bars[frequency]
            assert len(current_bars) <= 2
            assert tuple(getattr(current_bars[-1], key) for key in fields) == tuple(
                getattr(original_bars[-1], key) for key in fields)
        assert actual.closed == expected.closed
        assert actual.closed_available == expected.closed_available
        assert actual.decisions == expected.decisions
    assert len(actual.closed["日线"]) == len(bars)
    assert actual.payload() == expected.payload()
    assert actual.events == expected.events
    assert actual.position.operates == expected.position.operates


def test_prepared_wyckoff_calls_event_preparation_once_and_matches_full_pack(monkeypatch):
    raw = _raw()
    calls = []
    original = current.prepare_wyckoff_inputs
    def prepare(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(current, "prepare_wyckoff_inputs", prepare)
    prepared = current.prepare_native_wyckoff_inputs(raw)
    expected = current.build_native_evidence(prepared.frame)
    assert len(calls) == 2
    snapshot = copy.deepcopy(prepared.frame)
    actual = current.build_native_evidence(prepared.frame, prepared_wyckoff=prepared)
    assert len(calls) == 2
    _assert_pack_equal(actual, expected)
    pd.testing.assert_frame_equal(prepared.frame, snapshot, check_exact=True)
    assert prepared.frame.attrs == snapshot.attrs


def test_prepared_wyckoff_matches_real_feature_then_raw_column_append_path(baseline):
    raw = _raw(180)
    features = prepare_structure_inputs(bind_feature_profile(build_features(raw), "ma_trend"))
    calendar = raw.attrs["verified_market_dates"]
    expected = current.prepare_wyckoff_inputs(features, raw=raw, config={"market_dates": calendar})
    prepared = current.prepare_native_wyckoff_inputs(features, raw=raw, market_dates=calendar)
    for column in raw:
        if column not in expected:
            expected[column] = raw[column].to_numpy(copy=True)
        if column not in prepared.frame:
            prepared.frame[column] = raw[column].to_numpy(copy=True)
    actual = current.build_native_evidence(prepared.frame, market_dates=calendar, prepared_wyckoff=prepared)
    _assert_pack_equal(actual, baseline.build_native_evidence(expected, market_dates=calendar))


@pytest.mark.parametrize("change", ["raw", "index", "units", "calendar", "policy", "output", "output_attrs"])
def test_prepared_wyckoff_rejects_input_policy_or_output_drift(change):
    raw = _raw(160)
    prepared = current.prepare_native_wyckoff_inputs(raw)
    frame = prepared.frame.copy(deep=True)
    calendar = raw.attrs["verified_market_dates"].copy()
    config = {}
    if change == "raw":
        frame.loc[frame.index[-1], "close"] *= .999
    elif change == "index":
        frame.index = pd.RangeIndex(len(frame))
    elif change == "units":
        frame.attrs["volume_unit"] = "lots"
    elif change == "calendar":
        calendar.pop(30)
    elif change == "policy":
        config = {"wyckoff": {"quality": {"minimum_active_bars": 70}}}
    elif change == "output":
        prepared.frame.loc[prepared.frame.index[-1], "w_close_location"] = .123
    elif change == "output_attrs":
        prepared.frame.attrs["wyckoff_schema_hash"] = "altered"
    with pytest.raises(ValueError, match="prepared Wyckoff"):
        current.build_native_evidence(frame, config, market_dates=calendar, prepared_wyckoff=prepared)


def test_reuse_is_unavailable_for_different_w_policy_and_default_features_stay_unchanged():
    raw = _raw(160)
    prepared = current.prepare_native_wyckoff_inputs(raw)
    assert current.can_reuse_native_wyckoff(prepared)
    config = {"wyckoff": {"quality": {"minimum_active_bars": 70}}}
    assert not current.can_reuse_native_wyckoff(prepared, config)
    actual = current.build_native_evidence(prepared.frame, config)
    pd.testing.assert_frame_equal(actual["frame"][prepared.frame.columns], prepared.frame, check_exact=True)


@pytest.mark.parametrize("end", [119, 120, 180])
def test_reuse_keeps_exact_causal_prefix(end):
    raw = _raw()
    prepared = current.prepare_native_wyckoff_inputs(raw)
    complete = current.build_native_evidence(prepared.frame, prepared_wyckoff=prepared)
    prefix = current.prepare_native_wyckoff_inputs(raw.iloc[:end].copy())
    actual = current.build_native_evidence(prefix.frame, prepared_wyckoff=prefix)
    pd.testing.assert_frame_equal(actual["frame"], complete["frame"].iloc[:end], check_exact=True)
    assert actual["evaluations"] == complete["evaluations"][:end]
    day = raw.date.iloc[end - 1].strftime("%Y-%m-%d")
    assert actual["events"] == [item for item in complete["events"] if item["date"] <= day]


def test_no_calendar_reuse_does_not_mutate_prepared_eligibility():
    raw = _raw(160)
    raw.attrs.pop("verified_market_dates")
    prepared = current.prepare_native_wyckoff_inputs(raw)
    before = copy.deepcopy(prepared.frame)
    actual = current.build_native_evidence(prepared.frame, prepared_wyckoff=prepared)
    assert not actual["frame"].native_w_input_eligible.any()
    pd.testing.assert_frame_equal(prepared.frame, before, check_exact=True)
    _assert_pack_equal(actual, current.build_native_evidence(prepared.frame))
