"""Strict labels, immutable input reuse, deterministic bounded preparation."""
from copy import deepcopy
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import analysis_methods as methods
from my_strategy.services.light_score_training import make_direction_training_frame, LABEL_SCHEMA


def price_history(n=150):
    dates=pd.bdate_range('2025-01-02',periods=n)
    close=10+np.sin(np.arange(n)/5)
    return pd.DataFrame(dict(symbol='000001.SZ',date=dates,open=close+.01,
        high=close+.1,low=close-.1,close=close,volume=1000.,amount=10000.,
        source='tushare',has_trade_price=1))


def test_qualified_labels_are_exact_old_formula_and_exclusions_are_inspectable():
    frame=price_history();dates=frame.date.tolist()
    actual=make_direction_training_frame(frame,dates)
    returns=frame.close.shift(-5)/frame.open.shift(-1)-1
    expected=np.where(returns>.003,2,np.where(returns<-.003,0,1))[:-5]
    np.testing.assert_array_equal(actual.label.to_numpy(),expected)
    assert actual.label_schema.eq(LABEL_SCHEMA).all()
    audit=actual.attrs['label_audit']
    assert audit['eligible_rows']==145 and audit['excluded_rows']==5
    assert audit['reason_counts']=={'immature_label':5}
    assert frame.equals(price_history())


@pytest.mark.parametrize('offset,changes,reason',[
    (1,{'has_trade_price':0},'label_entry_trade_price_unverified'),
    (1,{'volume':0.,'amount':0.},'label_entry_suspended'),
    (5,{'has_trade_price':0},'label_terminal_trade_price_unverified'),
    (5,{'volume':0.,'amount':0.},'label_terminal_suspended'),
    (3,{'source':'unknown'},'label_intermediate_unverified_source'),
    (3,{'amount':0.},'label_intermediate_volume_amount_mismatch'),
    (5,{'close':float('nan')},'label_terminal_invalid_price'),
    (1,{'available_at':'2099-01-01'},'label_entry_unavailable_price'),
])
def test_future_price_counterexamples_are_excluded(offset,changes,reason):
    frame=price_history();signal=125
    for key,value in changes.items():frame.loc[signal+offset,key]=value
    result=make_direction_training_frame(frame,frame.date.tolist())
    day=frame.date.iloc[signal].strftime('%Y-%m-%d')
    assert day not in set(result.as_of)
    row=next(item for item in result.attrs['label_audit']['exclusions'] if item['as_of']==day)
    assert reason in row['reason_codes']


def test_labels_use_verified_calendar_across_holidays_and_reject_missing_middle():
    frame=price_history(20).drop(index=[3,4]).reset_index(drop=True)
    result=make_direction_training_frame(frame,frame.date.tolist())
    assert len(result)==len(frame)-5
    missing=frame.drop(index=7)
    result=make_direction_training_frame(missing,frame.date.tolist())
    assert result.attrs['label_audit']['reason_counts']['calendar_gap']==5


def test_missing_quality_metadata_never_qualifies_a_future_price():
    frame=price_history(10).drop(columns=['source','has_trade_price','amount'])
    result=make_direction_training_frame(frame,frame.date.tolist())
    assert result.empty and result.attrs['label_audit']['excluded_rows']==10


def test_prepare_stock_shares_one_raw_read_with_source(monkeypatch):
    from my_strategy.adapters import czsc_adapter
    raw=price_history(10);calls=[];end=str(raw.date.iloc[-1].date())
    monkeypatch.setattr(czsc_adapter,'load_bars',lambda *a,**k:calls.append('read') or raw)
    def source(symbol,day,**kwargs):
        assert kwargs['raw_snapshot'] is raw
        return raw,raw,[],{}
    light=SimpleNamespace(raw_db=None,source=source)
    assert methods.prepare_stock(light,'000001.SZ',end,methods.Stop(time.monotonic()+5))[0] is raw
    assert calls==['read']


def test_native_compact_preserves_every_scalar_event_and_latest_explanation():
    from my_strategy.tests.test_current_native_evidence import _raw
    from my_strategy.services.czsc_native_evidence import build_native_evidence
    raw=_raw(180,varied=True);dates=raw.date.dt.strftime('%Y-%m-%d').tolist()
    full=build_native_evidence(raw,market_dates=dates,capture_state=True)
    compact=build_native_evidence(raw,market_dates=dates,capture_state=True,compact=True)
    columns=[column for column in compact['frame']]
    pd.testing.assert_frame_equal(compact['frame'],full['frame'][columns],check_exact=True)
    assert compact['events']==full['events']
    assert compact['evaluations']==full['evaluations'][-1:]
    assert compact['incremental_state']==full['incremental_state']


def test_native_cancel_checks_inside_long_stock_and_never_publishes_manifest(tmp_path,monkeypatch):
    from my_strategy.tests.test_current_native_evidence import _raw
    from my_strategy.services import incremental_source,czsc_analysis
    from my_strategy.storage.research_store import ResearchStore
    raw=_raw(180,varied=True).assign(symbol='002436.SZ');dates=raw.date.dt.strftime('%Y-%m-%d').tolist();stop=threading.Event();calls=[]
    original=czsc_analysis.CzscSession.advance_structure
    def advance(self,row):
        calls.append(True)
        if len(calls)==40:stop.set()
        return original(self,row)
    monkeypatch.setattr(czsc_analysis.CzscSession,'advance_structure',advance)
    monkeypatch.setattr(incremental_source,'load_bars',lambda *a,**k:raw)
    with ResearchStore(tmp_path/'cache') as store:
        source=incremental_source.IncrementalSource(store=store)
        with pytest.raises(InterruptedError):source.source('002436.SZ',dates[-1],calendar_dates=dates,stop=stop)
        assert 40<=len(calls)<=64
        assert store.list()==[] and not store.get_setting('incremental_source_latest:002436.SZ')
        assert store.connection.execute('SELECT COUNT(*) FROM leases').fetchone()[0]==0


def test_generation_identity_excludes_chart_and_account_methods_but_binds_numeric_methods(monkeypatch):
    from my_strategy.services import czsc_analysis,light_research
    original=czsc_analysis.native_generation_identity()
    method_original=methods._code_identity()
    def display_only(self,*args,**kwargs):return {'display':'changed'}
    monkeypatch.setattr(czsc_analysis.CzscSession,'payload',display_only)
    monkeypatch.setattr(light_research.LightResearch,'backtest',display_only)
    assert czsc_analysis.native_generation_identity()==original
    assert methods._code_identity()==method_original
    def numeric_change(self,*args,**kwargs):return ('changed',None,None)
    monkeypatch.setattr(czsc_analysis.CzscSession,'advance_structure',numeric_change)
    assert czsc_analysis.native_generation_identity()!=original
    assert methods._code_identity()!=method_original


@pytest.mark.parametrize('seed',[1,17,91])
def test_exact_rolling_state_matches_pandas_after_json_restart(seed):
    import json
    from my_strategy.services.incremental_rolling import trailing_features_with_state,COLUMNS
    from my_strategy.services.czsc_market_features import _trailing_features
    rng=np.random.default_rng(seed);raw=price_history(410)
    raw['close']=10*np.exp(np.cumsum(rng.normal(0,.012,len(raw))))
    raw['open']=raw.close*.999;raw['high']=raw.close*1.02;raw['low']=raw.close*.98
    raw['volume']=rng.uniform(1e4,1e8,len(raw));raw['amount']=raw.volume*raw.close
    # Legal suspensions and missing quantity preserve the old NaN contract.
    raw.loc[123,['volume','amount']]=0
    raw.loc[201,['volume','amount']]=np.nan
    state=None;frames=[];start=0
    for end in (121,151,187,241,310,410):
        frame,state=trailing_features_with_state(raw.iloc[:end],state)
        assert state is not None,'installed pandas kernel failed exact mean admission'
        frames.append(frame[list(COLUMNS)]);state=json.loads(json.dumps(state));start=end
    expected=_trailing_features(raw)[list(COLUMNS)]
    pd.testing.assert_frame_equal(pd.concat(frames),expected,check_exact=True)
    changed=raw.copy();changed.loc[2,'close']+=.001
    with pytest.raises(ValueError,match='prefix'):
        trailing_features_with_state(pd.concat([changed,changed.iloc[-1:]],ignore_index=True),state)


def test_source_identity_binds_price_loader_but_excludes_status_queries(monkeypatch):
    from my_strategy.adapters import czsc_adapter
    from my_strategy.services.incremental_source import code_identity
    original=code_identity()
    def display_only(**kwargs):return {'status':'new_summary'}
    monkeypatch.setattr(czsc_adapter,'data_status',display_only)
    assert code_identity()==original
    def different_prices(*args,**kwargs):raise ValueError('changed loader')
    monkeypatch.setattr(czsc_adapter,'load_bars',different_prices)
    assert code_identity()!=original


def test_native_budget_expiry_is_timeout_separate_from_explicit_cancel():
    from my_strategy.services.light_research import DeadlineStop
    from my_strategy.services.czsc_native_evidence import _check_cancel
    from my_strategy.services.incremental_source import _stop
    for check in (_check_cancel,_stop):
        with pytest.raises(TimeoutError):check(DeadlineStop(threading.Event(),time.monotonic()-1))
        event=threading.Event();event.set()
        with pytest.raises(InterruptedError):check(DeadlineStop(event,time.monotonic()+10))


def test_generation_contract_binds_shared_qualification_serialization_and_calendar(monkeypatch):
    from my_strategy.core import run_context
    from my_strategy.services import czsc_trend_support as support, czsc_research_calendar as calendar, light_research
    from my_strategy.services.incremental_source import code_identity, numerical_input_contract
    initial=code_identity();training=light_research.score_code_identity()
    with monkeypatch.context() as change:
        change.setattr(support,'VERIFIED_SOURCES',support.VERIFIED_SOURCES|{'different_price_admission'})
        assert code_identity()!=initial
        assert light_research.score_code_identity()!=training
    for module,name in ((support,'_json'),(support,'_calendar'),(support,'resolve_frozen_asset'),
                        (run_context,'stable_hash'),(calendar,'load_verified_calendar'),
                        (calendar,'_valid_sessions'),(calendar,'calendar_prefix_compatible'),(calendar,'_file_hash')):
        before=numerical_input_contract()
        with monkeypatch.context() as change:
            def different_contract(*args,**kwargs):raise AssertionError('identity inspection does not run input helpers')
            change.setattr(module,name,different_contract)
            assert numerical_input_contract()!=before
            assert code_identity()!=initial
            assert light_research.score_code_identity()!=training


def test_generation_contract_binds_configuration_helpers_and_prediction_verification(monkeypatch):
    from my_strategy.core import config_loader
    from my_strategy.services import czsc_analysis, kronos_assets
    from my_strategy.services import analysis_methods as methods
    before=czsc_analysis.native_generation_identity();generation=methods._code_identity()
    def different_helper(*args,**kwargs):return None
    with monkeypatch.context() as change:
        change.setattr(config_loader,'_parse_env_value',different_helper)
        assert czsc_analysis.native_generation_identity()!=before
    with monkeypatch.context() as change:
        change.setattr(kronos_assets,'_verify',different_helper)
        assert methods._code_identity()!=generation
