"""Adapter-only regressions for requested market clocks and saved account replay."""
import asyncio
import copy
from types import SimpleNamespace

import pandas as pd
import pytest

from my_strategy.services.czsc_trend_contracts import STRATEGY_ID,strategy_config


@pytest.mark.parametrize('requested,expected', [('2026-01-05','2026-01-05'),('2026-01-04','2026-01-02')])
def test_analysis_keeps_missing_stock_day_or_disclosed_nontrading_roll_clock(monkeypatch,requested,expected):
    from my_strategy.services import czsc_analysis,czsc_trend_runtime
    from my_strategy.web_dashboard import high_return,trend_strategy,chart_evidence_trend
    calendar=['2026-01-01','2026-01-02','2026-01-05']
    observed='2026-01-02'
    result={'symbol':'002436.SZ','data_end':observed,'as_of':requested,'research':{},'warnings':[]}
    frame=pd.DataFrame([{'date':observed,'symbol':result['symbol'],'close':10.}])
    monkeypatch.setattr(czsc_analysis,'analyze_stock',lambda *args,**kwargs:copy.deepcopy(result))
    monkeypatch.setattr(high_return,'_calendar',lambda run_id=None:(calendar,'calendar-fixed'))
    monkeypatch.setattr(trend_strategy,'_source_vintage',lambda *args,**kwargs:({'features':frame,'raw':frame,
        'data_version':'fixed-vintage','data_end':'2026-01-05'},'source-fixed','2026-01-05'))
    seen={}
    def overlay(frame,calendar,**kwargs):
        seen.update(kwargs)
        return {'model_run_id':None,'latest_decision':{'date':kwargs['as_of'],'reference_price':None}}
    monkeypatch.setattr(czsc_trend_runtime,'runtime_overlay',overlay)
    monkeypatch.setattr(chart_evidence_trend,'decorate_analysis',lambda result,**kwargs:{**result,'research':{},
        'next_session_decision':kwargs['overlay']['latest_decision']})
    out=trend_strategy.analyze_trend_consensus(symbol=result['symbol'],as_of=requested)
    assert seen['as_of']==expected
    assert out['data_end']==observed  # Observed price is never invented on the missing day.
    assert out['next_session_decision']['date']==expected
    assert out['research']['observed_end']==observed
    assert out['research']['decision_date']==expected
    assert out['research']['requested_end']==requested


def test_analysis_saved_request_binds_actual_run_and_account_with_auto_heads(monkeypatch):
    from my_strategy.web_dashboard import api,high_return
    from my_strategy.core import run_context
    chosen='immutable-trend-run'
    account={'account_id':'selected','account_start':'2023-01-01','initial_cash':100000.}
    cfg=strategy_config()['consensus']
    returned={'symbol':'002436.SZ','data_end':'2026-01-02','data_version':'raw-vintage','strategy_version':STRATEGY_ID,
        'chart_evidence':{},
        'consensus_decisions':{'version':STRATEGY_ID,'config':cfg,'config_hash':'strategy-frozen'},
        'research':{'model_run_id':chosen,'calendar_run_id':'calendar-fixed','account_context':account},
        'account_actions':{'context':account}}
    monkeypatch.setattr(high_return,'analyze_high_return',lambda **kwargs:copy.deepcopy(returned))
    monkeypatch.setattr(run_context,'create_run_context',lambda **kwargs:SimpleNamespace(run_id='stored-analysis',to_dict=lambda:{'request':kwargs['config']['request']}))
    saved=[]
    request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(tasks=SimpleNamespace(results=SimpleNamespace(save=lambda kind,result:saved.append(result) or result)))))
    spec=api.AnalysisRequest(symbol='002436.SZ',model_family='high_return',research=True,
        consensus_version=STRATEGY_ID,consensus_config=cfg,model_policy='auto')
    result=asyncio.run(api.analysis(spec,request))
    assert result['request']['model_run_id']==chosen
    assert result['request']['model_policy']=='auto'
    assert result['request']['account_start']=='2023-01-01'
    assert result['request']['account_initial_cash']==100000.
    assert result['request']['calendar_run_id']=='calendar-fixed'
    assert saved[0]['request']==result['run_context']['request']
