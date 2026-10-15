import copy
import pandas as pd
import pytest

from my_strategy.services import incremental_source as module
from my_strategy.storage.research_store import ResearchStore
from my_strategy.tests.test_current_native_evidence import _raw


@pytest.fixture
def source(tmp_path,monkeypatch):
    raw=_raw(340,varied=True).reset_index(drop=True)
    def load(symbol,end,db_path=None):
        result=raw.loc[raw.date<=pd.Timestamp(end)].copy();result.symbol=symbol
        result.attrs.update(raw.attrs,data_version='fixture',volume_unit='shares',amount_unit='CNY')
        return result
    monkeypatch.setattr(module,'load_bars',load)
    store=ResearchStore(tmp_path/'cache')
    source=module.IncrementalSource(store=store)
    yield source,store,raw,raw.date.dt.strftime('%Y-%m-%d').tolist()
    store.close()


def test_same_input_zero_compute_and_no_repeated_history(source,monkeypatch):
    service,store,raw,dates=source;end=dates[309]
    first=service.source('002436.SZ',end,calendar_dates=dates)
    before=len(store.list(limit=1000))
    from my_strategy.services import czsc_native_evidence
    monkeypatch.setattr(czsc_native_evidence,'build_native_evidence',lambda *a,**k:pytest.fail('native reran'))
    repeat=service.source('002436.SZ',end,calendar_dates=dates+[dates[-1]+'x'])
    pd.testing.assert_frame_equal(first[1],repeat[1],check_dtype=False)
    assert first[3]==repeat[3]
    assert repeat[1].attrs['incremental']['reused']
    assert len(store.list(limit=1000))==before


def test_append_restart_reuses_chunks_and_matches_full(source,tmp_path):
    service,store,raw,dates=source
    prefix=service.source('002436.SZ',dates[309],calendar_dates=dates)
    old=service._peek(prefix[3]['source_cache_key'])
    resumed=module.IncrementalSource(store=store)
    result=resumed.source('002436.SZ',dates[-1],calendar_dates=dates)
    assert result[1].attrs['incremental']['mode']=='native_state_append'
    assert result[1].attrs['incremental']['native_rows_appended']==30
    saved=resumed._peek(result[3]['source_cache_key'])
    assert old['chunks'][0]==saved['chunks'][0]
    with ResearchStore(tmp_path/'independent') as other:
        expected=module.IncrementalSource(store=other).source('002436.SZ',dates[-1],calendar_dates=dates)
    pd.testing.assert_frame_equal(result[1],expected[1],check_dtype=False,check_exact=True)
    assert result[2]==expected[2]


def test_historical_revision_rebuilds_only_changed_symbol(source,monkeypatch):
    service,store,raw,dates=source
    service.source('002436.SZ',dates[-1],calendar_dates=dates)
    load=module.load_bars
    def revised(symbol,end,db_path=None):
        result=load(symbol,end,db_path)
        if symbol=='002436.SZ':result.loc[10,'volume']+=2
        return result
    monkeypatch.setattr(module,'load_bars',revised)
    plan=service.inspect('002436.SZ',dates[-1],calendar_dates=dates)
    assert plan['action']=='rebuild' and plan['first_changed_date']==dates[10]
    result=service.source('002436.SZ',dates[-1],calendar_dates=dates)
    assert result[1].attrs['incremental']['mode']=='single_stock_full_replay'


def test_cancel_before_source_writes_nothing(source):
    service,store,raw,dates=source
    with pytest.raises(InterruptedError):service.source('002436.SZ',dates[-1],calendar_dates=dates,stop=lambda:True)
    assert store.list()==[]


def test_missing_cache_chunk_is_local_repair_not_false_reuse(source):
    service,store,raw,dates=source
    result=service.source('002436.SZ',dates[-1],calendar_dates=dates)
    manifest=service._peek(result[3]['source_cache_key'])
    with store._tx():store.connection.execute('DELETE FROM results WHERE cache_key=?',(manifest['chunks'][0]['cache_key'],))
    assert service.inspect('002436.SZ',dates[-1],calendar_dates=dates)['action']=='rebuild'
    repaired=service.source('002436.SZ',dates[-1],calendar_dates=dates)
    assert not repaired[1].attrs['incremental']['reused']
    assert repaired[3]==result[3]
