"""Real SQLite-prefix equivalence and bounded account preparation scheduling."""
from __future__ import annotations

from copy import deepcopy
import json
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import pytest

from my_strategy.adapters.czsc_adapter import load_bars
from my_strategy.core.run_context import stable_hash
from my_strategy.services.light_prediction_inputs import AccountPredictionInputs,account_prediction_callback
from my_strategy.services import light_research as research


@pytest.fixture
def raw_market(tmp_path):
    path=tmp_path/'raw.db'
    dates=pd.bdate_range('2025-01-02',periods=180).strftime('%Y-%m-%d').tolist()
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER,pct_chg REAL,available_at TEXT)')
        for symbol in ('002156.SZ','002436.SZ'):
            rows=[(symbol,day,10+i*.01,11+i*.01,9+i*.01,10+i*.01,1e7,1e8,
                   'fuyao_unadjusted_v1',1,None,day+'T15:00:00+08:00') for i,day in enumerate(dates)]
            con.executemany('INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',rows)
    return path,dates


def test_actual_adapter_prefix_values_attrs_and_prediction_identity_are_identical(raw_market):
    path,days=raw_market
    raw=load_bars('002156.SZ',end=days[-1],db_path=path)
    original=raw.copy(deep=True);calendar=days+['2026-01-02']
    cal={'run_id':'frozen-calendar','sha256':stable_hash(calendar)}
    with AccountPredictionInputs({'002156.SZ':raw},calendar,cal) as inputs:
        for day in (days[130],days[131],days[131],days[150],days[-1]):
            actual=inputs.get('002156.SZ',day)
            expected=load_bars('002156.SZ',end=day,db_path=path)
            pd.testing.assert_frame_equal(actual.raw,expected,check_exact=True)
            assert actual.raw.attrs==expected.attrs
            assert actual.exact_price_input_sha256==stable_hash(research.plain(expected.to_dict('records')))
            assert actual.calendar_dates==tuple(calendar) and actual.calendar_identity==cal
            assert str(actual.raw.iloc[-1].date)[:10]==day
        assert inputs.stats['peak_pending']<=8
    pd.testing.assert_frame_equal(raw,original,check_exact=True)
    assert inputs.closed and not inputs.pending


def test_prefix_metadata_never_inherits_future_legacy_or_unverified_rows(raw_market):
    path,days=raw_market
    with sqlite3.connect(path) as con:
        con.execute('UPDATE stock_daily_normalized SET source=?,has_trade_price=0 WHERE date=?',('fuyao',days[-1]))
    raw=load_bars('002156.SZ',end=days[-1],db_path=path)
    with AccountPredictionInputs({'002156.SZ':raw},days,{}) as inputs:
        prefix=inputs.get('002156.SZ',days[-2]).raw
        expected=load_bars('002156.SZ',end=days[-2],db_path=path)
        assert prefix.attrs==expected.attrs
        assert prefix.attrs['unverified_trade_price_bars']==0


def test_missing_target_date_or_foreign_symbol_never_uses_later_or_stale_price(raw_market):
    path,days=raw_market
    raw=load_bars('002156.SZ',db_path=path)
    with AccountPredictionInputs({'002156.SZ':raw},days,{}) as inputs:
        with pytest.raises(ValueError,match='不能使用旧日报价'):inputs.get('002156.SZ','2024-01-01')
        with pytest.raises(ValueError,match='不能使用旧日报价'):inputs.get('002156.SZ','2030-01-01')
        with pytest.raises(ValueError,match='outside frozen'):inputs.get('600519.SH',days[-1])


def test_cancel_drains_all_workers_and_never_calls_model_or_store(raw_market,monkeypatch):
    path,days=raw_market;stop=threading.Event()
    raw=load_bars('002156.SZ',db_path=path)
    inputs=AccountPredictionInputs({'002156.SZ':raw},days,{},stop=stop)
    threads=set();running=0;maximum=0;lock=threading.Lock();original=inputs._prepare
    def prepare(*args):
        nonlocal running,maximum
        with lock:running+=1;maximum=max(maximum,running);threads.add(threading.current_thread().name)
        try:time.sleep(.01);return original(*args)
        finally:
            with lock:running-=1
    monkeypatch.setattr(inputs,'_prepare',prepare)
    for day in days[-10:]:inputs._submit('002156.SZ',day)
    assert len(inputs.pending)<=8 and inputs.stats['peak_pending']<=8
    stop.set();inputs.close()
    assert running==0 and maximum<=2 and all(t.startswith('account-input') for t in threads)
    assert not inputs.pending and inputs.stats['model_calls_in_workers']==0
    with pytest.raises(InterruptedError):inputs.get('002156.SZ',days[-1])


def test_cached_prediction_reuses_the_identical_original_request_key(raw_market,tmp_path,monkeypatch):
    path,days=raw_market
    from my_strategy.services import kronos_prediction
    monkeypatch.setattr(research,'prediction_identity',lambda *_:{'model':'frozen'})
    monkeypatch.setattr(kronos_prediction,'release_prediction_resources',lambda:None)
    calls=[]
    def predict(raw,future,**kw):
        calls.append((raw.copy(deep=True),list(future),kw))
        return {'as_of':kw['as_of'],'samples':[], 'quality':{'prediction_usable':False},
                'summary':None,'identity':{'frozen':'prediction'},'resource':{'elapsed_seconds':1.}}
    monkeypatch.setattr(kronos_prediction,'predict',predict)
    light=research.LightResearch(tmp_path/'isolated',raw_db=path)
    calendar=days+pd.bdate_range('2025-10-01',periods=10).strftime('%Y-%m-%d').tolist()
    monkeypatch.setattr(light,'calendar',lambda:(calendar,{'sha256':stable_hash(calendar)}))
    day=days[150];raw=load_bars('002156.SZ',db_path=path)
    try:
        first=light.prediction('002156.SZ',day)
        with light.account_prediction_inputs({'002156.SZ':raw},*light.calendar()) as inputs:
            prepared=inputs.get('002156.SZ',day)
            monkeypatch.setattr(research,'load_bars',lambda *a,**kw:pytest.fail('repeated raw DB read'))
            monkeypatch.setattr(light,'calendar',lambda:pytest.fail('repeated calendar read'))
            second=light.prediction('002156.SZ',day,_prepared_input=prepared)
            assert second['reused'] and second['result_ref']['cache_key']==first['result_ref']['cache_key']
            assert len(calls)==1
            assert calls[0][2]['samples']==8 and calls[0][2]['seed']==42
            with pytest.raises(ValueError,match='causal request'):
                light.prediction('002156.SZ',days[151],_prepared_input=prepared)
    finally:light.close()


def test_model_calls_and_progress_remain_in_original_causal_thread(raw_market):
    path,days=raw_market
    raw=load_bars('002156.SZ',db_path=path);owner=threading.current_thread().name;seen=[];updates=[]
    class Service:
        def prediction(self,symbol,day,**kw):
            seen.append((threading.current_thread().name,symbol,day,kw['_prepared_input'].as_of))
            return {'quality':{'prediction_usable':True}}
    with AccountPredictionInputs({'002156.SZ':raw},days,{}) as inputs:
        predict,progress=account_prediction_callback(Service(),inputs,{'samples':8,'seed':42},None,
            lambda **kw:updates.append(kw))
        for i,day in enumerate(days[-4:-1]):
            progress(stage='agent_observation',date=day,symbol='002156.SZ',completed_sessions=i,total_sessions=3,
                     observations_completed=i,completed=0,total=1)
            predict('002156.SZ',day)
    assert all(t==owner and d==prefix for t,s,d,prefix in seen)
    assert [x['phase'] for x in updates if 'phase' in x]==['input_preparation','model_inference']*3
    assert updates[-1]['observations_completed']==2 and updates[-1]['total_sessions']==3
