from __future__ import annotations

import json

import pandas as pd
import pytest

from my_strategy.services.czsc_liquidity import (
    LIQUIDITY_COLUMNS, annotate_liquidity, daily_eligibility, liquidity_policy,
)


def bars(calendar, amount=40_000_000):
    frame = pd.DataFrame({'symbol': '000001.SZ', 'date': pd.to_datetime(calendar),
        'open': 10., 'high': 11., 'low': 9., 'close': 10.,
        'volume': amount / 10., 'amount': float(amount),
        'source': 'tushare', 'has_trade_price': 1})
    frame.attrs.update(volume_unit='shares', amount_unit='CNY')
    return frame


def calendar(count=45):
    return pd.bdate_range('2025-01-02', periods=count).strftime('%Y-%m-%d').tolist()


def test_default_policy_complete_window_and_threshold_boundary():
    dates = calendar(21)
    frame = bars(dates, 30_000_000)
    daily = daily_eligibility(frame, dates)
    assert liquidity_policy() == {'lookback_sessions': 20, 'min_median_amount': 30_000_000.,
        'complete_verified_window': True, 'selection': 'historical_daily_eligible_union'}
    assert not daily.iloc[:19].liquidity_pool_eligible.any()
    assert daily.iloc[19:].liquidity_pool_eligible.all()
    assert daily.iloc[19].liquidity_window_start == dates[0]
    assert daily.iloc[19].liquidity_observed_at == dates[19] + 'T15:00:00+08:00'
    assert daily.iloc[19].liquidity_qualified_sessions == 20


def test_future_appending_cannot_rewrite_decisions_or_unknown_missing_history():
    dates = calendar()
    frame = bars(dates[6:])
    frame.loc[frame.index[-5:], 'amount'] = 1e12
    cutoff = dates[29]
    prefix = daily_eligibility(frame.loc[frame.date <= pd.Timestamp(cutoff)], dates[:30])
    full = daily_eligibility(frame, dates)
    pd.testing.assert_frame_equal(prefix, full.iloc[:30].reset_index(drop=True))
    assert prefix.iloc[5].liquidity_status == 'insufficient_verified_window'
    assert 'no_prior_local_observation' in json.loads(prefix.iloc[5].liquidity_reason_codes)
    assert not prefix.iloc[19].liquidity_pool_eligible


def test_missing_verified_day_cannot_be_replaced_by_next_observed_day():
    dates = calendar(42)
    # Simulated verified holiday is omitted from the calendar, not counted as a
    # missing observation. A different missing verified session stays unknown.
    dates.remove(dates[7])
    frame = bars(dates).drop(index=14)
    daily = daily_eligibility(frame, dates)
    assert daily.iloc[32].liquidity_status == 'missing_market_observation_unknown'
    assert daily.iloc[32].liquidity_observed_sessions == 19
    assert not daily.iloc[32].liquidity_pool_eligible
    assert daily.iloc[34].liquidity_pool_eligible
    assert 'suspension' not in daily.iloc[32].liquidity_reason_codes
    assert daily.iloc[34].liquidity_window_start == dates[15]


@pytest.mark.parametrize('column,value,reason', [
    ('source','fuyao','unverified_source'),
    ('has_trade_price',0,'unverified_trade_price'),
    ('close',0,'invalid_price'),
    ('high',8,'invalid_price'),
    ('volume',0,'nonpositive_volume'),
    ('amount',0,'nonpositive_amount'),
    ('amount',float('nan'),'nonpositive_amount'),
])
def test_any_unqualified_observation_blocks_entire_verified_window(column,value,reason):
    dates = calendar(40)
    frame = bars(dates)
    frame.loc[19,column] = value
    daily = daily_eligibility(frame,dates)
    assert reason in json.loads(daily.iloc[38].liquidity_reason_codes)
    assert daily.iloc[38].liquidity_qualified_sessions == 19
    assert not daily.iloc[38].liquidity_pool_eligible
    assert daily.iloc[39].liquidity_pool_eligible


def test_units_and_availability_are_real_qualification_inputs():
    dates = calendar(25)
    frame = bars(dates)
    frame.attrs['volume_unit'] = 'lots'
    assert daily_eligibility(frame, dates).iloc[-1].liquidity_status == 'unverified_units'
    frame.attrs['volume_unit'] = 'shares'
    frame['available_at'] = [date + 'T15:00:00+08:00' for date in dates]
    frame.loc[19, 'available_at'] = dates[20] + 'T15:00:00+08:00'
    daily = daily_eligibility(frame, dates)
    assert daily.iloc[-1].liquidity_status == 'observation_not_available_at_close'


def test_overlapping_reasons_are_not_the_exclusive_primary_count():
    dates = calendar(20)
    frame = bars(dates)
    frame.loc[19,['source','volume','amount']] = ['unknown',0,0]
    daily = daily_eligibility(frame, dates)
    assert daily.iloc[-1].liquidity_status == 'unverified_source'
    assert set(json.loads(daily.iloc[-1].liquidity_reason_codes)) == {
        'unverified_source','nonpositive_volume','nonpositive_amount'}


def test_annotation_preserves_model_features_index_attrs_and_verifies_all_frozen_fields():
    dates = calendar(25)
    frame = bars(dates)
    frame['model_input_eligible'] = True
    frame.index = range(100,125)
    frame.attrs['price_basis'] = 'unadjusted'
    annotated = annotate_liquidity(frame, dates)
    assert len(annotated) == len(frame)
    assert annotated.index.equals(frame.index)
    assert annotated.model_input_eligible.all()
    assert annotated.attrs['price_basis'] == 'unadjusted'
    assert set(LIQUIDITY_COLUMNS) <= set(annotated)
    annotate_liquidity(annotated, dates, verify_existing=True)
    annotated.loc[119,'liquidity_pool_eligible'] = False
    with pytest.raises(ValueError,match='qualification drift'):
        annotate_liquidity(annotated, dates, verify_existing=True)
    with pytest.raises(ValueError,match='partial frozen'):
        annotate_liquidity(annotated.drop(columns='liquidity_status'), dates, verify_existing=True)


@pytest.mark.parametrize('override',[
    {'complete_verified_window':False}, {'lookback_sessions':True},
    {'min_median_amount':float('inf')}, {'selection':'latest_survivors'},
])
def test_policy_cannot_relax_completeness_or_select_latest_survivors(override):
    with pytest.raises(ValueError):
        liquidity_policy(override)
