"""Calendar filtering hoists an invariant without changing input identities."""
from copy import deepcopy

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import incremental_source as source


def raw_frame():
    value = pd.DataFrame({'symbol': ['000001.SZ']*3,
        'date': pd.to_datetime(['2026-10-09', '2026-10-12', '2026-10-13']),
        'open': [10., 11., 12.], 'high': [11., 12., 13.], 'low': [9., 10., 11.],
        'close': [10.5, 11.5, 12.5], 'volume': [100., 200., 300.],
        'amount': [1000., 2000., 3000.], 'source': ['tushare']*3,
        'has_trade_price': [1]*3})
    value.attrs.update(price_basis='unadjusted', volume_unit='shares', amount_unit='CNY')
    return value


@pytest.mark.parametrize('requested_end', ['2026-10-13', '2026-10-16'])
def test_exact_original_calendar_row_hashes_and_identity_are_retained(monkeypatch, requested_end):
    raw = raw_frame()
    dates = ['2026-10-09', '2026-10-12', '2026-10-13', '2026-10-14', '2026-10-15', '2026-10-16']
    producer = {'fully_verified_producer': 'fixed-for-numerical-equivalence'}
    monkeypatch.setattr(source, 'code_identity', lambda: deepcopy(producer))
    reader = source.IncrementalSource(store=None)
    frame, calendar, hashes, identity = reader._parts('000001.SZ', requested_end, dates, raw)
    expected_dates = [d for d in dates if d <= str(pd.Timestamp(raw.date.iloc[-1]).date())]
    expected_hashes = source._row_hashes(raw)
    expected = {'kind': source.VERSION, 'symbol': '000001.SZ', 'as_of': requested_end,
        'observed_end': str(pd.Timestamp(raw.date.iloc[-1]).date()),
        'input_sha': stable_hash(expected_hashes), 'price_basis': 'unadjusted',
        'units': {'volume_unit': 'shares', 'amount_unit': 'CNY'},
        'calendar': {'sha256': stable_hash(expected_dates), 'prefix_end': expected_dates[-1]}, 'code': producer}
    assert frame is raw
    assert calendar == expected_dates and hashes == expected_hashes and identity == expected
    assert identity['observed_end'] == '2026-10-13'


def test_timestamp_work_is_constant_in_calendar_length(monkeypatch):
    raw = raw_frame()
    dates = pd.bdate_range('2000-01-01', '2026-10-16').strftime('%Y-%m-%d').tolist()
    calls = []
    timestamp = source.pd.Timestamp
    def observed(value):
        calls.append(value)
        return timestamp(value)
    monkeypatch.setattr(source.pd, 'Timestamp', observed)
    monkeypatch.setattr(source, '_row_hashes', lambda raw: ['full-original-row-sha']*len(raw))
    monkeypatch.setattr(source, 'code_identity', lambda: {'verified': 'producer'})
    _, calendar, _, _ = source.IncrementalSource(store=None)._parts('000001.SZ', '2026-10-16', dates, raw)
    assert len(calendar) > 6000
    assert len(calls) == 3  # input cutoff, requested cutoff and one observed day


@pytest.mark.parametrize('failure', ['missing_end', 'wrong_symbol', 'future_row', 'empty'])
def test_original_admission_failures_remain_explicit(monkeypatch, failure):
    raw = raw_frame()
    dates = ['2026-10-09', '2026-10-12', '2026-10-13']
    requested = '2026-10-16'
    if failure == 'missing_end':
        dates.remove('2026-10-13')
        message = '末日不在已核验日历'
    elif failure == 'wrong_symbol':
        raw.loc[0, 'symbol'] = '000002.SZ'
        message = '共享行情快照'
    elif failure == 'future_row':
        requested = '2026-10-12'
        message = '共享行情快照'
    else:
        raw = raw.iloc[:0]
        message = '没有已核验行情'
    with pytest.raises(ValueError, match=message):
        source.IncrementalSource(store=None)._parts('000001.SZ', requested, dates, raw)
