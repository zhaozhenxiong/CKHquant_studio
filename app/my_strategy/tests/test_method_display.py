"""Frozen method display is causal, identity-bound and free of computation."""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import json
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from my_strategy.services import method_display as display
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.storage.research_store import ResearchStore

DAY = '2024-10-15'
MODEL = 'a'*64
VERSION = 'b'*64
REF = 'c'*64


def sample(method='structure_kronos_ma_mlp', direction='BUY'):
    median = -.02 if direction == 'SELL' else .02
    probabilities = {'down': .7, 'flat': .2, 'up': .1} if direction == 'SELL' else {'down': .1, 'flat': .2, 'up': .7}
    if direction == 'WAIT': probabilities = {'down': .1, 'flat': .8, 'up': .1}
    kind = 'l2' if method.endswith('_l2') else 'mlp'
    scored = method != 'structure_kronos'
    score = {'status':'scored','score':100*(probabilities['up']+.5*probabilities['flat']),
        'probabilities':probabilities,'model_version':VERSION,'model_type':'l2' if kind=='l2' else 'torch_mlp',
        'feature_profile':'ma_trend_v1','variant':'A','consumes_kronos':False,
        'scorer_id':display._digest({'model':VERSION,'head':kind}),
        'qualification':{'mode':'historical_candidate','paper_allocation_eligible':False}}
    context = {'analysis_method':method,'score_model_id':MODEL if scored else None,'start':'2024-01-01',
        'end':DAY,'usage_mode':'historical','lookback':128,'horizon':5,'samples':32,'seed':42,'device':'cpu'}
    quality = {'prediction_usable':True,'calendar_verified':True,'input_contract_verified':True,
        'source_verified':True,'requested_samples':32,'valid_samples':32,'invalid_samples':0,'valid_fraction':1.}
    summary = {'terminal_return_p10':-.05,'terminal_return_median':median,'terminal_return_p90':.05,
        'terminal_return_std':.02,'up_path_fraction':.6,'worst_path_low_return':-.1}
    return {'symbol':'000001.SZ','date':DAY,'data_end':DAY,'analysis_method':method,'method_context':context,
        'method_identity':{'model_version':VERSION if scored else None},
        'close':12.,'reference_price':12.,'reference_price_date':DAY,'reference_price_source':'verified_local_close',
        'reference_only':True,'account_scope':'none','events':[{'date':'2024-08-01','action':'SELL','source':'structure'}],
        'method_evaluation':{'schema':'account_free_methods_v1','method':method,'as_of':DAY,'status':'complete',
            'direction':direction,'missing_reasons':[],'model_id':MODEL if scored else None,
            'feature_profile':'ma_trend_v1' if scored else None,'score':score if scored else None,
            'kronos':{'status':'completed','as_of':DAY,'quality':quality,'summary':summary,'result_ref':{'cache_key':REF}}}}


def analysis_sample():
    row=sample()
    row.update(as_of=DAY,data_version='synthetic',method_identity={'model_version':VERSION},price_basis='unadjusted',
        frequencies={'日线':{'bars':[{'time':'2024-10-14','open':11.,'high':12.,'low':10.,'close':11.5,'is_closed':True},
            {'time':DAY,'open':11.5,'high':13.,'low':11.,'close':12.,'is_closed':True}]}},
        source_reference_points=deepcopy(row['events']))
    for key in ('reference_price','reference_price_date','reference_price_source','close'):row.pop(key)
    return row


@pytest.mark.parametrize('method',sorted(display.COMBINED))
@pytest.mark.parametrize('direction',['BUY','SELL'])
def test_saved_cutoff_direction_is_bound_to_method_and_never_a_fill(method,direction):
    row=sample(method,direction);before=deepcopy(row)
    view=display.classify_method_row(row,context=row['method_context'],as_of=DAY,model_version=VERSION if method!='structure_kronos' else None)
    assert view['valid'] and view['status']=='complete' and view['direction']==direction
    assert view['forecast_valid'] and view['prediction_ref']==REF
    assert len(view['method_reference_points'])==1
    point=view['method_reference_points'][0]
    assert point['date']==DAY and point['action']==direction and point['reference_price']==12.
    assert point['reference_layer']=='method' and point['actual_available_at'] is None
    assert point['execution'] is False and point['first_trigger'] is False
    assert row==before


def test_valid_wait_does_not_invent_a_marker():
    view=display.build_method_view(sample(direction='WAIT'))
    assert view['valid'] and view['direction']=='WAIT' and view['method_reference_points']==[]


@pytest.mark.parametrize('status',['partial','failed','deferred'])
def test_real_placeholder_or_missing_evidence_never_becomes_valid_wait(status):
    row=sample(direction='WAIT')
    if status!='partial':
        from my_strategy.services.analysis_methods import _placeholder
        row=_placeholder(row['symbol'],DAY,row['analysis_method'],status,'synthetic_missing')
    else:row['method_evaluation'].update(status=status,missing_reasons=['synthetic_missing'])
    view=display.classify_method_row(row,context=sample()['method_context'],as_of=DAY,model_version=VERSION)
    assert view['status']==('unavailable' if status=='partial' else status)
    assert not view['valid'] and view['direction'] is None and not view['method_reference_points']


@pytest.mark.parametrize('change,reason',[
    (lambda r:r['method_evaluation'].update(method='structure_kronos'), 'method_identity_mismatch'),
    (lambda r:r['method_evaluation'].update(as_of='2024-10-14'), 'method_cutoff_mismatch'),
    (lambda r:r['method_evaluation'].update(model_id='d'*64), 'score_model_identity_mismatch'),
    (lambda r:r['method_evaluation']['score'].update(model_version='d'*64), 'score_model_version_mismatch'),
    (lambda r:r['method_evaluation']['score'].update(model_type='l2'), 'score_method_identity_mismatch'),
    (lambda r:r['method_evaluation']['score'].update(scorer_id='d'*64), 'score_head_identity_mismatch'),
    (lambda r:r['method_evaluation']['score'].update(variant='B'), 'score_method_identity_mismatch'),
    (lambda r:r['method_evaluation']['score'].update(consumes_kronos=True), 'score_method_identity_mismatch'),
    (lambda r:r['method_evaluation']['score'].update(feature_profile='mixed_v1'), 'score_method_identity_mismatch'),
    (lambda r:r['method_evaluation']['score'].update(score=99), 'score_probability_value_mismatch'),
    (lambda r:r['method_evaluation']['score'].update(probabilities={'down':.1,'flat':.2,'up':.8}), 'score_unavailable'),
    (lambda r:r['method_evaluation']['score'].update(probabilities={'down':False,'flat':.2,'up':.8}), 'score_unavailable'),
    (lambda r:r['method_evaluation']['score']['qualification'].update(mode='paper'), 'score_usage_identity_mismatch'),
    (lambda r:r['method_evaluation'].update(direction='SELL'), 'method_direction_evidence_mismatch'),
    (lambda r:r['method_evaluation'].update(missing_reasons=['missing']), 'complete_method_has_missing_evidence'),
    (lambda r:r['method_evaluation']['kronos']['quality'].update(prediction_usable=False), 'forecast_quality_unavailable'),
    (lambda r:r['method_evaluation']['kronos']['quality'].update(valid_samples=20,valid_fraction=.625), 'forecast_sample_quality_invalid'),
    (lambda r:r['method_evaluation']['kronos']['quality'].update(requested_samples=64,valid_samples=64), 'forecast_sample_identity_mismatch'),
    (lambda r:r['method_evaluation']['kronos']['summary'].update(terminal_return_median=float('nan')), 'forecast_summary_invalid'),
    (lambda r:r.update(reference_price_date='2024-10-14'), 'cutoff_reference_price_unavailable'),
    (lambda r:r.update(reference_price=0), 'cutoff_reference_price_unavailable'),
    (lambda r:r.update(reference_price=True), 'cutoff_reference_price_unavailable'),
])
def test_complete_flag_cannot_hide_broken_evidence(change,reason):
    row=sample();change(row)
    view=display.classify_method_row(row,context=sample()['method_context'],as_of=DAY,model_version=VERSION)
    assert view['status']=='unavailable' and not view['valid'] and view['direction'] is None
    assert reason in view['reason_codes'] and view['method_reference_points']==[]


def test_context_drift_clears_chart_ref_too():
    row=sample();row['method_context']['seed']=43
    view=display.classify_method_row(row,context=sample()['method_context'],as_of=DAY,model_version=VERSION)
    assert 'method_context_mismatch' in view['reason_codes'] and view['prediction_ref'] is None


def test_complete_ma_requires_frozen_header_version_and_full_context():
    row=sample();row.pop('method_identity')
    view=display.build_method_view(row)
    assert not view['valid'] and 'score_model_version_mismatch' in view['reason_codes']
    row=sample();row.pop('method_context')
    view=display.classify_method_row(row,context=sample()['method_context'],as_of=DAY,model_version=VERSION)
    assert not view['valid'] and 'method_context_missing' in view['reason_codes']
    row=sample();row['method_context'].pop('seed')
    view=display.classify_method_row(row,context=sample()['method_context'],as_of=DAY,model_version=VERSION)
    assert not view['valid'] and 'method_context_mismatch' in view['reason_codes']


@pytest.mark.parametrize('field,value',[
    ('horizon',True),('horizon',[]),('lookback',{}),('samples','32'),
    ('seed',False),('seed',2147483648),('usage_mode',[]),
])
def test_malformed_parameters_are_unavailable_without_an_exception(field,value):
    row=sample();row['method_context'][field]=value
    view=display.build_method_view(row)
    assert not view['valid'] and 'method_parameters_invalid' in view['reason_codes']


@pytest.mark.parametrize('change',[
    lambda r:r['method_evaluation']['score'].update(model_version={'invalid':True}),
    lambda r:r['method_evaluation'].update(direction=['BUY']),
    lambda r:r['method_evaluation'].update(status=['complete'],missing_reasons={'invalid':True}),
    lambda r:r['method_evaluation']['kronos']['summary'].update(terminal_return_median=10**1000),
])
def test_malformed_saved_evidence_never_raises_or_produces_a_point(change):
    row=sample();change(row)
    view=display.build_method_view(row)
    assert not view['valid'] and view['method_reference_points']==[]


def test_explicit_forecast_horizon_is_checked_without_requiring_absent_legacy_field():
    row=sample();assert display.build_method_view(row)['valid']
    row['method_evaluation']['kronos']['future_dates']=['2024-10-16','2024-10-17','2024-10-18','2024-10-21','2024-10-22']
    assert display.build_method_view(row)['valid']
    for dates in ([DAY],['2024-10-16']*5,[{}]*5,None):
        row['method_evaluation']['kronos']['future_dates']=dates
        view=display.build_method_view(row)
        assert not view['valid'] and 'forecast_horizon_identity_mismatch' in view['reason_codes']


@pytest.mark.parametrize('method,reason',[
    ('structure_kronos_ma_mlp','ma_score_model_missing'),
    ('structure_kronos_ma_l2','same_input_l2_missing'),
])
def test_missing_score_evidence_keeps_valid_forecast_and_is_not_wait(method,reason):
    row=sample(method)
    row['method_evaluation'].update(status='partial',missing_reasons=[reason],score={'status':'unavailable','reason':reason})
    if reason=='ma_score_model_missing':
        row['method_evaluation']['model_id']=None;row['method_context']['score_model_id']=None
    view=display.build_method_view(row)
    assert view['status']=='unavailable' and view['direction'] is None
    assert view['forecast_valid'] and not view['score_valid'] and view['prediction_ref']==REF


def test_forecast_and_score_availability_are_separate():
    row=sample();row['method_evaluation'].update(status='partial',missing_reasons=['ma_input_ineligible'])
    row['method_evaluation']['score'].update(status='unavailable',score=None,probabilities=None)
    view=display.build_method_view(row)
    assert view['forecast_valid'] and view['forecast_summary'] and not view['score_valid']
    assert view['prediction_ref']==REF and view['direction'] is None


def test_unusable_forecast_keeps_saved_chart_reference_but_no_direction():
    row=sample();row['method_evaluation']['kronos']['quality']['prediction_usable']=False
    view=display.build_method_view(row)
    assert view['prediction_ref']==REF and not view['forecast_valid'] and not view['valid']


def test_missing_or_expired_cas_does_not_erase_saved_statistics():
    row=sample();row['method_evaluation']['kronos'].pop('result_ref')
    view=display.build_method_view(row)
    assert view['valid'] and view['forecast_valid'] and view['prediction_ref'] is None
    assert view['prediction_reference_status']=='missing'


@pytest.mark.parametrize('method',['structure','legacy_reference'])
def test_other_methods_keep_original_semantics_and_all_source_points(method):
    row=analysis_sample();row.update(analysis_method=method,method_context={'analysis_method':method})
    before=deepcopy(row);out=display.with_method_view(row)
    assert row==before and out['source_reference_points']==before['source_reference_points']
    assert out['method_view']['status']=='not_applicable' and out['method_view']['prediction_ref'] is None
    assert out['method_reference_points']==[]


@pytest.mark.parametrize('research',[None,{'legacy':True}])
def test_original_analysis_without_any_method_contract_is_preserved(research):
    original={'symbol':'000001.SZ','data_end':DAY,'events':[{'date':DAY,'action':'BUY'}]}
    if research:original['research']=research
    result=display.with_method_view(original)
    assert result['events']==original['events'] and 'method_view' not in original
    assert result['method_view']['status']=='not_applicable' and result['method_reference_points']==[]


def test_daily_chart_requires_exact_closed_positive_cutoff_bar_and_never_changes_cached_payload():
    row=analysis_sample();before=deepcopy(row);out=display.with_method_view(row)
    assert out['method_view']['valid'] and out['method_reference_points'][0]['reference_price']==12.
    assert row==before and 'method_view' not in row
    for kind in ('missing','forming','duplicate','invalid_geometry','future_only'):
        altered=deepcopy(row);bars=altered['frequencies']['日线']['bars']
        if kind=='missing':bars.pop()
        elif kind=='forming':bars[-1]['is_closed']=False
        elif kind=='duplicate':bars.append(deepcopy(bars[-1]))
        elif kind=='invalid_geometry':bars[-1]['high']=10.
        else:bars[-1]['time']='2024-10-16'
        view=display.build_method_view(altered)
        assert not view['valid'] and view['direction'] is None and not view['method_reference_points']


@pytest.fixture
def api_fixture(tmp_path,monkeypatch):
    from my_strategy.web_dashboard import api
    from my_strategy.services import analysis_methods,czsc_analysis
    store=ResultStore(tmp_path/'results.db',tmp_path/'runs')
    cache=ResearchStore(tmp_path/'research')
    app=FastAPI();app.include_router(api.router)
    app.state.tasks=SimpleNamespace(results=store,list=lambda:[])
    app.state.light=SimpleNamespace(store=cache,jobs={})
    calls=[]
    def analyze(light,spec):calls.append(spec);return analysis_sample()
    monkeypatch.setattr(analysis_methods,'analyze_method',analyze)
    monkeypatch.setattr(czsc_analysis,'strategy_config',lambda:{})
    from my_strategy.core import run_context
    monkeypatch.setattr(run_context,'create_run_context',lambda **kw:SimpleNamespace(run_id='synthetic-method-view',to_dict=lambda:{'synthetic':True}))
    with TestClient(app) as client:yield client,store,cache,calls
    cache.close()


@pytest.mark.parametrize('save',[True,False])
def test_analysis_post_enriches_response_after_saving_original_evidence(api_fixture,save):
    client,store,cache,calls=api_fixture
    response=client.post('/api/analysis',json={'symbol':'000001.SZ','analysis_method':'structure_kronos_ma_mlp',
        'score_model_id':MODEL,'start':'2024-01-01','end':DAY,'save':save,'samples':32})
    assert response.status_code==200,response.text
    result=response.json();assert result['method_view']['valid'] and len(calls)==1
    if save:
        stored=store.get('synthetic-method-view')
        assert 'method_view' not in stored and 'method_reference_points' not in stored
        path=store.runs_root/'synthetic-method-view/reports/czsc.json';before=path.read_bytes()
        get=client.get('/api/runs/synthetic-method-view')
        assert get.status_code==200 and get.json()['method_view']==result['method_view']
        assert path.read_bytes()==before and len(calls)==1
    else:
        rows=cache.list('analysis_previews')
        assert len(rows)==1
        stored=cache.get_by_key(rows[0]['cache_key'],touch=False)
        assert 'method_view' not in stored and 'method_reference_points' not in stored
