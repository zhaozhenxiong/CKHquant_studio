"""Historical hover snapshots use exactly the corresponding observed prefix."""
import json

import pandas as pd
import pytest

from my_strategy.services.czsc_chart_indicators import chart_indicators, daily_volume_profile


def bars(count=170):
    return [{"time": value.date().isoformat(), "open": 10 + index / 10,
             "high": 11 + index / 10, "low": 9 + index / 10,
             "close": 10 + index / 10, "volume": index + 1, "is_closed": True}
            for index, value in enumerate(pd.bdate_range("2025-01-01", periods=count))]


def calculate(daily, visible=None, start=None):
    selected = daily if visible is None else visible
    return chart_indicators(selected, [row["time"] + "T15:00:00+08:00" for row in selected],
                            daily_volume_profile(daily), start, daily_bars=daily)


def test_each_hover_histogram_matches_independent_prefix_and_conserves_volume():
    daily = bars()
    result = calculate(daily, start=daily[115]["time"])
    assert set(result["chip_profiles"]) == {row["time"] for row in daily[115:]}
    for index in (115, 119, 130, 169):
        profile = result["chip_profiles"][daily[index]["time"]]
        assert profile == calculate(daily[:index + 1])["chip_profile"]
        if index >= 119:
            expected = sum(row["volume"] for row in daily[index - 119:index + 1])
            assert profile["total_volume"] == expected
            assert sum(bin_["volume"] for bin_ in profile["bins"]) == expected
            assert sum(bin_["mass_fraction"] for bin_ in profile["bins"]) == pytest.approx(1)
    assert result["chip_profile"] == result["chip_profiles"][daily[-1]["time"]]
    json.dumps(result, allow_nan=False)


def test_week_month_and_forming_hover_exclude_future_observations():
    daily = bars()
    anchors = [{**daily[index], "is_closed": closed}
               for index, closed in ((119, True), (129, True), (139, False))]
    baseline = calculate(daily, anchors)
    for row in daily[140:]:
        row.update(high=1e8, low=1e8, close=1e8, volume=1e8)
    current = calculate(daily, anchors)
    assert current["chip_profiles"] == baseline["chip_profiles"]
    for index in (119, 129, 139):
        profile = current["chip_profiles"][daily[index]["time"]]
        assert profile == calculate(daily[:index + 1])["chip_profile"]
        assert profile["as_of"] == daily[index]["time"]


def test_short_history_and_no_volume_remain_explicit_for_each_date():
    daily = bars(122)
    for row in daily:
        row["volume"] = 0
    result = calculate(daily)
    assert result["chip_profiles"][daily[118]["time"]]["status"] == "insufficient_history"
    assert result["chip_profiles"][daily[119]["time"]]["status"] == "no_volume"
    assert result["chip_profiles"][daily[-1]["time"]]["bins"] == []
    assert calculate([], [])["chip_profiles"] == {}
