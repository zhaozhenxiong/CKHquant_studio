import copy

import pytest

from my_strategy.scripts.compare_native_v4_broker import compare_reports


def _reports():
    config = {"cohort_hash": "same", "windows": {"2026Q2": ["2026-04-01", "2026-06-30"]},
              "evidence_bars": [3], "slippage_multipliers": [1], "initial_cash_per_account": 100000.,
              "calendar_hash": "verified", "raw_end": "2026-10-09"}
    aggregate = {"window": "2026Q2", "evidence_bars": 3, "slippage_multiplier": 1., "variant": "v4",
                 "requested_accounts": 2, "retained_accounts": 2, "failed_cash_accounts": 1,
                 "total_return": .02, "max_drawdown": -.01, "completed_round_trips": 4}
    main = {"config": config, "aggregate": [aggregate, {**aggregate, "variant": "v3", "total_return": .01}]}
    control = {"config": copy.deepcopy(config), "aggregate": [{**aggregate, "total_return": -.01}]}
    return main, control


def test_real_failed_cash_accounts_remain_in_pair_and_sample_gate():
    main, control = _reports()
    result = compare_reports(main, control)
    assert result[0]["main_return_minus_frozen_v3"] == pytest.approx(.01)
    assert result[0]["main_return_minus_risk_control"] == pytest.approx(.03)
    assert result[0]["failed_cash_accounts_retained"] == {"main": 1, "risk_control": 1, "v3": 1}
    assert result[0]["main_sample_minimum_met"] is False


def test_missing_selected_cash_account_is_rejected():
    main, control = _reports()
    control["aggregate"][0]["retained_accounts"] = 1
    with pytest.raises(ValueError, match="denominator"):
        compare_reports(main, control)


def test_different_windows_cannot_be_called_paired():
    main, control = _reports()
    control["config"]["calendar_hash"] = "different"
    with pytest.raises(ValueError, match="calendar_hash"):
        compare_reports(main, control)
