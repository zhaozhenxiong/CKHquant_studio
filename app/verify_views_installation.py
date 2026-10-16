"""Read-only installed view probes; all prices and saved evidence are synthetic.

The subprocess imports runtime services from the installed wheel. It never loads
model weights or creates a business task. A private loopback ASGI server exercises
the actual shipped routes, with computation and store writes replaced by guards.
"""
from __future__ import annotations

from contextlib import ExitStack, closing
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import gc
from html.parser import HTMLParser
import json
import os
import sys
from pathlib import Path
import socket
import sqlite3
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen

METHODS = ['structure_kronos', 'structure_kronos_ma_mlp', 'structure_kronos_ma_l2']
COUNTS = dict(expected=6, matched=6, complete=3, buy=1, sell=1, wait=1,
              unavailable=1, failed=1, deferred=1, not_in_scan=0)
DAY = '2024-10-15'
MODEL = 'a'*64
VERSION = 'b'*64


def require(condition, message):
    if not condition:
        raise RuntimeError('Installed v1.0.2 view probe: '+message)


def navigation_probe(html):
    """Check real element ancestry, so duplicated or merely present IDs fail."""
    class Elements(HTMLParser):
        def __init__(self):
            super().__init__(); self.stack=[]; self.parents={}
        def handle_starttag(self, tag, attrs):
            name=dict(attrs).get('id')
            if name:
                require(name not in self.parents, 'duplicate HTML identity '+name)
                self.parents[name]=[value for _,value in self.stack if value]
            if tag not in {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}:
                self.stack.append((tag,name))
        def handle_endtag(self, tag):
            for i in range(len(self.stack)-1,-1,-1):
                if self.stack[i][0] == tag:
                    del self.stack[i:]; break
    parsed=Elements(); parsed.feed(html)
    def below(child, parent): return parent in parsed.parents.get(child, [])
    require(all(below(key,'view-watchlist') and below(key,'watchlist-accounts-panel')
        for key in ('light-sim-form','light-saved','light-accounts-state','light-results','light-daily')), 'account controls outside watchlist')
    require(below('light-predict-form','view-simulation') and below('light-training-panel','view-simulation'), 'prediction/model controls missing')
    require(all(below(key,'light-prediction') for key in ('light-history-window','light-history-start','light-prediction-chart')), 'history controls missing')
    require(all(below(key,'view-analysis') for key in ('layer-method','layer-sources','analysis-prediction')), 'method/source layers missing')
    require(below('board-method-panel','view-scan'), 'board method panel missing')
    return dict(status='passed', accounts_in_watchlist=True, simulation_in_watchlist=True,
        prediction_history_controls=True, separate_method_source_layers=True, board_method_panel=True)


def method_row(symbol='000001.SZ', method=METHODS[1], direction='BUY', state='complete', prediction_ref='c'*64):
    from my_strategy.services.method_display import _digest
    if state in {'failed','deferred'}:
        return dict(symbol=symbol,date=DAY,data_end=None,analysis_method=method,action='WAIT',
            method_evaluation=dict(method=method,as_of=DAY,status=state,direction='WAIT',
                missing_reasons=['synthetic_'+state],score=None,kronos=None),reference_only=True,account_scope='none')
    scored=method!=METHODS[0]; head='l2' if method==METHODS[2] else 'mlp'
    probabilities=({'down':.7,'flat':.2,'up':.1} if direction=='SELL' else
                   {'down':.1,'flat':.8,'up':.1} if direction=='WAIT' else {'down':.1,'flat':.2,'up':.7})
    score=dict(status='scored', score=100*(probabilities['up']+.5*probabilities['flat']),
        probabilities=probabilities,model_version=VERSION,model_type='l2' if head=='l2' else 'torch_mlp',
        feature_profile='ma_trend_v1',variant='A',consumes_kronos=False,
        scorer_id=_digest({'model':VERSION,'head':head}),qualification={'mode':'historical_candidate','paper_allocation_eligible':False})
    context=dict(analysis_method=method,score_model_id=MODEL if scored else None,start='2024-01-01',
        end=DAY,usage_mode='historical',lookback=128,horizon=5,samples=32,seed=42,device='cpu')
    quality=dict(prediction_usable=True,calendar_verified=True,input_contract_verified=True,source_verified=True,
        requested_samples=32,valid_samples=32,invalid_samples=0,valid_fraction=1.)
    summary=dict(terminal_return_p10=-.05,terminal_return_median=-.02 if direction=='SELL' else .02,
        terminal_return_p90=.05,terminal_return_std=.02,up_path_fraction=.6,worst_path_low_return=-.1)
    value=dict(symbol=symbol,date=DAY,as_of=DAY,data_end=DAY,analysis_method=method,method_context=context,
        method_identity={'model_version':VERSION if scored else None},close=12.,reference_price=12.,
        reference_price_date=DAY,reference_price_source='verified_local_close',reference_only=True,account_scope='none',
        source_reference_points=[{'date':'2024-08-01','action':'SELL','source':'structure'}],
        method_evaluation=dict(schema='account_free_methods_v1',method=method,as_of=DAY,status=state,
            direction=direction,missing_reasons=[],model_id=MODEL if scored else None,feature_profile='ma_trend_v1' if scored else None,
            score=score if scored else None,kronos=dict(status='completed',as_of=DAY,quality=quality,
                summary=summary,result_ref={'cache_key':prediction_ref})))
    if state=='partial':
        value['method_evaluation'].update(missing_reasons=['synthetic_missing_score'],score={'status':'unavailable'})
    return value


def scan_fixture(results):
    rows=[method_row(direction=direction,symbol=f'00000{i}.SZ',state=state)
          for i,(direction,state) in enumerate([('BUY','complete'),('SELL','complete'),('WAIT','complete'),
              ('WAIT','partial'),('WAIT','failed'),('WAIT','deferred')],1)]
    context=rows[0]['method_context']; population=[row['symbol'] for row in rows]
    result=dict(run_id='synthetic-v102-scan',status='partial',strategy_version='account_free_methods_v1',
        analysis_method=METHODS[1],analysis_mode='reference',method_context=context,as_of=DAY,data_end=DAY,
        population=population,rows=rows,frozen_identity=dict(population=population,context=deepcopy(context),
            model_version=VERSION,raw={'sha256':'synthetic'},calendar={'id':'synthetic'},code={'id':'synthetic'}),
        coverage=dict(requested=6,failed=1,deferred=1,unavailable=1,evaluated=5,complete=False))
    results.save('scan',result)
    for method in METHODS:
        result=method_row(method=method); result['run_id']='synthetic-'+method
        results.save('analysis',result)
    for method in ('structure','legacy_reference'):
        result=dict(run_id='synthetic-'+method,symbol='000001.SZ',analysis_method=method,data_end=DAY,
                    source_reference_points=[{'date':'2024-08-01','action':'SELL','source':'structure'}])
        results.save('analysis',result)


def board_fixture(path):
    from my_strategy.storage import board_knowledge as kb
    members=[f'00000{i}.SZ' for i in range(1,8)]+['830001.BJ']
    value=dict(format=kb.FORMAT,source=kb.SOURCE,taxonomy=kb.TAXONOMY,board_id='THS_I_881155',source_code='881155',
        name='Synthetic installation board',kind='industry',observed_at='2024-10-14T13:59:00+08:00',
        available_at='2024-10-14T13:59:00+08:00',effective_at='2024-10-14T13:59:00+08:00',
        complete=True,expected_count=len(members),members=members,source_url='https://q.10jqka.com.cn/thshy/detail/code/881155/',
        provenance=dict(pagination_complete=True,reported_count=len(members),unique_members=len(members)))
    # Synthetic historical observation clock belongs only to fixture creation.
    with patch.object(kb,'_now',lambda:datetime(2024,10,14,6,tzinfo=timezone.utc)):
        kb.BoardKnowledgeStore(path).import_snapshot(value)


def prediction_fixture(raw, store):
    import numpy as np
    import pandas as pd
    from my_strategy.adapters.czsc_adapter import load_bars
    from my_strategy.services import kronos_prediction as kronos
    days=pd.bdate_range('2024-01-02',DAY).strftime('%Y-%m-%d').tolist()
    # A later raw bar already exists, proving the GET truncates to the original cutoff.
    days.append('2024-10-16')
    with closing(sqlite3.connect(raw)) as db, db:
        db.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER)')
        db.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
            [('000001.SZ',day,10+i/100,11+i/100,9+i/100,10.5+i/100,10000+i,100000+i*10,'sina_unadjusted_v1',1) for i,day in enumerate(days)])
    frame=load_bars('000001.SZ',end=DAY,db_path=raw)
    future=pd.bdate_range('2024-10-16',periods=5).strftime('%Y-%m-%d').tolist()
    _,_,info=kronos.prepare_input(frame,future,symbol='000001.SZ',as_of=DAY,lookback=128,horizon=5)
    parameters=dict(lookback=128,horizon=5,samples=32,seed=42,temperature=.8,top_p=.9,device='cpu',dtype='float32')
    calendar={'run_id':'synthetic-calendar','sha256':'a'*64}
    identity=dict(schema=kronos.SCHEMA,input=info,parameters=parameters,future_dates=future,
        calendar={'verified':True,'identity':calendar},assets=[{'fixture':True}],source={'fixture':True})
    samples=np.empty((32,5,6))
    for i in range(32):
        close=info['last_close']+i/100+np.arange(5)/100
        samples[i]=np.column_stack((close,close+1,close-1,close,close*0+10000,close*0+100000))
    original=kronos.summarize_samples(samples,future,info['last_close'])
    original.update(schema=kronos.SCHEMA,status='completed',symbol='000001.SZ',as_of=DAY,identity=identity,
        input=deepcopy(info),cache_key=kronos._hash(identity),samples=samples[:8].tolist(),retained_diagnostic_samples=8)
    original['quality'].update(calendar_verified=True,input_contract_verified=True,source_verified=True)
    outer=dict(kind='prediction-v1',symbol='000001.SZ',as_of=DAY,parameters=parameters,calendar=calendar,
        data=frame.attrs['data_version'],code='kronos_request_cache_v2')
    refs={'original':store.put(outer,original,namespace='predictions')['cache_key']}
    low=deepcopy(original);low['quality']['prediction_usable']=False
    refs['low']=store.put({**outer,'fixture':'low'},low,namespace='predictions')['cache_key']
    revised=deepcopy(original); revised['identity']['input']['data_sha256']='0'*64
    revised['input']=deepcopy(revised['identity']['input']); revised['cache_key']=kronos._hash(revised['identity'])
    refs['revised']=store.put({**outer,'fixture':'revision'},revised,namespace='predictions')['cache_key']
    refs['expired']=store.put({**outer,'fixture':'expired'},original,namespace='predictions',ttl_seconds=1)['cache_key']
    return refs,original,kronos.summarize_samples(samples[:8],future,info['last_close'])


def _file_snapshot(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*')
            if p.is_file() and not p.name.endswith('-shm') and not (p.name.endswith('-wal') and p.stat().st_size==0)}


def run_probe():
    from fastapi import FastAPI
    import uvicorn
    from my_strategy.services import kronos_prediction as kronos
    from my_strategy.services.light_research import LightResearch
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.storage.research_store import ResearchStore
    from my_strategy.web_dashboard.api import router as api
    from my_strategy.web_dashboard.board_api import router as boards
    from my_strategy.web_dashboard.light_api import router as light_api
    with tempfile.TemporaryDirectory(prefix='khquant-v102-views-') as temporary:
        root=Path(temporary); tick=[100.]; raw=root/'synthetic.db'; calls=[]; forbidden_calls=[]
        with ResearchStore(root/'research',clock=lambda:tick[0],ttl_seconds=1000) as store:
            refs,original,diagnostic=prediction_fixture(raw,store)
            results=ResultStore(root/'results.db',root/'runs'); scan_fixture(results); board_fixture(root/'boards.db')
            store.set_setting('enabled_accounts',{'synthetic-account':{'enabled':False}})
            store.put({'fixture':'saved-account'},{'ledger':[],'synthetic':True},namespace='simulations',pin='synthetic-protection')
            tick[0]+=61 # default touching reads would mutate the access timestamp; expiry passes too.
            def forbidden(*_args,**_kwargs):
                forbidden_calls.append('business_or_storage_write'); raise RuntimeError('View read attempted computation or storage mutation')
            light=SimpleNamespace(store=store,raw_db=raw)
            for name in ('prediction','source','submit','train_score','backtest','enable_account','calendar'):
                setattr(light,name,forbidden)
            app=FastAPI();app.state.light=light;app.state.tasks=SimpleNamespace(results=results,submit=forbidden)
            app.state.board_db_path=root/'boards.db'
            for router in (api,boards,light_api):app.include_router(router)
            with ExitStack() as guards:
                for name in ('predict','_load_model','_device','asset_status','_source_identity'):
                    guards.enter_context(patch.object(kronos,name,forbidden))
                for name in ('source','prediction','submit','train_score','backtest'):
                    if hasattr(LightResearch,name):guards.enter_context(patch.object(LightResearch,name,forbidden))
                for name in ('put','pin','unpin','set_setting','gc','lease'):
                    if hasattr(store,name):guards.enter_context(patch.object(store,name,forbidden))
                guards.enter_context(patch.object(ResultStore,'save',forbidden))
                gc.collect() # Finish fixture writer finalizers before observing read-only behavior.
                before=_file_snapshot(root); logical='\n'.join(store.connection.iterdump()); changes=store.connection.total_changes
                listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(128)
                base='http://127.0.0.1:'+str(listener.getsockname()[1])
                server=uvicorn.Server(uvicorn.Config(app,log_level='critical',access_log=False))
                worker=threading.Thread(target=server.run,kwargs={'sockets':[listener]},name='synthetic-view-server')
                worker.start()
                try:
                    deadline=time.monotonic()+15
                    while not server.started:
                        require(worker.is_alive() and time.monotonic()<deadline,'private HTTP server did not start')
                        time.sleep(.01)
                    def read(path,expected=200):
                        calls.append(('GET',path))
                        try:
                            with urlopen(base+path,timeout=15) as response:
                                code=response.status; data=json.load(response)
                        except HTTPError as exc:
                            with exc:
                                code=exc.code
                                try:data=json.load(exc)
                                except ValueError:data={'error':'non-JSON HTTP error'}
                        require(code==expected,f'GET {path} returned {code}, expected {expected}: {data}')
                        return data
                    chart=lambda key,code=200:read('/api/light/predictions/'+key+'/chart',code)
                    actual=chart(refs['original'])
                    require(actual['prediction_ref']==refs['original']!=original['cache_key'],'outer prediction reference mismatch')
                    require(len(actual['history'])==128 and len(actual['forecast'])==5,'history/forecast window mismatch')
                    require(actual['history_verification']['status']=='verified','history input identity was not verified')
                    require(actual['history'][-1]['date']==DAY and all(row['date']<=DAY for row in actual['history'])
                        and all(row['date']>DAY for row in actual['forecast']),'cutoff not preserved')
                    require(all(actual[key]==original[key] for key in ('forecast','summary','quality','input'))
                        and actual['forecast']!=diagnostic['forecast'],'full sample evidence changed or diagnostic samples recomputed')
                    require(actual['parameters']==original['identity']['parameters'],'saved parameters changed')
                    chart(original['cache_key'],404); chart(refs['expired'],404)
                    revised=chart(refs['revised'])
                    require(revised['history']==[] and revised['history_verification']['reason_code']=='history_input_changed'
                        and revised['forecast']==original['forecast'],'revised history was joined or forecast changed')
                    low=chart(refs['low']); require(low['quality']['prediction_usable'] is False and low['forecast']==original['forecast'],'low quality was promoted')
                    for method in METHODS:
                        value=read('/api/runs/synthetic-'+method); view=value['method_view']; points=view['method_reference_points']
                        require(view['valid'] is True and view['direction']=='BUY' and view['method']==method,'saved method classification differs')
                        require(len(points)==1 and points[0]['date']==DAY and points[0]['actual_available_at'] is None
                            and points[0]['execution'] is False and points[0]['first_trigger'] is False,'method marker invents history or execution')
                        require(value['source_reference_points'][0]['date']=='2024-08-01','historical source layer was altered')
                    for method in ('structure','legacy_reference'):
                        value=read('/api/runs/synthetic-'+method)
                        require(value['method_view']['status']=='not_applicable' and value['method_reference_points']==[]
                            and value['source_reference_points'][0]['action']=='SELL','legacy/source semantics changed')
                    binding='?method='+METHODS[1]+'&as_of='+DAY
                    prefix='/api/boards/method-scans/synthetic-v102-scan'
                    listed=read('/api/boards/method-scans'+binding)
                    require(listed.get('pagination',{}).get('total')==1,'saved method run was not listed')
                    summary=read(prefix+binding)
                    require(summary['population_summary']['counts']==COUNTS,'population counts differ')
                    require(summary['membership_semantics']['historical_membership_verified'] is False,'current membership was promoted to historical')
                    board=summary['boards'][0]
                    require(board['counts']['expected']==7 and board['counts']['not_in_scan']==1 and board['membership']['excluded_count']==1,'board scope denominator differs')
                    one=read(prefix+'/stocks'+binding+'&page_size=1&page=1')
                    two=read(prefix+'/stocks'+binding+'&page_size=1&page=2')
                    require(one['rows'][0]['symbol']!=two['rows'][0]['symbol'] and one['population_summary']==two['population_summary']==summary['population_summary'],'pagination changes full-population summary')
                    for category in ('buy','sell','wait','unavailable','failed','deferred'):
                        page=read(prefix+'/stocks'+binding+'&category='+category)
                        require(page['scan_view']['total']==1 and page['scan_view']['total_rows']==6,'filter total is wrong: '+category)
                        view=page['rows'][0]['method_view']
                        require((view['direction'].lower() if view['valid'] else view['status'])==category,'filter classification is wrong')
                        if category in ('unavailable','failed','deferred'):
                            require(view['direction'] is None and view['method_reference_points']==[],'missing row became WAIT or B/S')
                    missing=read(prefix+'/members'+binding+'&board_id=THS_I_881155&category=not_in_scan')
                    require(len(missing['rows'])==1 and missing['rows'][0]['symbol']=='000007.SZ'
                        and missing['rows'][0]['method_view']['direction'] is None,'missing board member misclassified')
                    require(missing['identity']['summary_sha256']==summary['identity']['summary_sha256'],'board member binding differs')
                    read(prefix+'?method='+METHODS[0]+'&as_of='+DAY,409)
                    read(prefix+'?method='+METHODS[1]+'&as_of=2024-10-14',409)
                    require(not forbidden_calls,'read invoked business code')
                    after=_file_snapshot(root)
                    delta=sorted(key for key in set(before)|set(after) if before.get(key)!=after.get(key))
                    require(after==before and '\n'.join(store.connection.iterdump())==logical
                        and store.connection.total_changes==changes,'read altered saved objects or SQLite state: '+repr(delta))
                finally:
                    server.should_exit=True;worker.join(timeout=20);listener.close()
                    require(not worker.is_alive(),'private HTTP worker did not join')
        require(all(method=='GET' for method,_ in calls),'non-read request sent')
    return dict(schema='readonly_method_views_installation_v1',status='passed',fixture='synthetic_installation_only',
        production_state_touched=False,work_submitted=False,inference_calls=0,readonly_state_unchanged=True,
        prediction_chart=dict(status='passed',history_rows=128,forecast_rows=5,requested_samples=32,retained_diagnostic_samples=8,
            outer_ref_required=True,full_sample_summary_preserved=True,cutoff_preserved=True,revised_history_unavailable=True,
            low_quality_preserved=True,expired_ref_rejected=True),
        method_display=dict(status='passed',methods=METHODS,cutoff_only=True,invalid_direction_null=True,legacy_semantics_preserved=True),
        board_summary=dict(status='passed',population=6,counts=COUNTS,board_expected=7,board_not_in_scan=1,excluded_members=1,
            full_population_pagination=True,exact_filters=True,mismatched_binding_rejected=True,historical_membership_verified=False))


def main():
    # Standalone invocation is just as isolated as the parent installer. Set
    # paths before importing runtime modules, which can initialize metadata.
    with tempfile.TemporaryDirectory(prefix='khquant-view-import-') as temporary:
        state=Path(temporary).resolve(); previous=Path.cwd()
        for key in list(os.environ):
            if key.startswith('KHQUANT_') or key in {'PYTHONPATH','PYTHONHOME'}:os.environ.pop(key)
        os.environ.update(KHQUANT_HOME=str(state),KHQUANT_DATA_ROOT=str(state/'data'),
            KHQUANT_ARTIFACT_ROOT=str(state/'artifacts'),KHQUANT_METADATA_ROOT=str(state/'data/metadata'),
            KHQUANT_RAW_DB=str(state/'data/raw/khquant_raw.db'),KHQUANT_LOG_ROOT=str(state/'data/logs'),
            KHQUANT_KRONOS_ROOT=str(state/'models/Kronos-base'),KHQUANT_COMPUTE_DEVICE='cpu')
        try:
            os.chdir(state)
            import my_strategy
            package=Path(my_strategy.__file__).resolve()
            require(package.is_relative_to(Path(sys.prefix).resolve()) and 'site-packages' in package.parts,
                    'standalone probe requires the installed wheel')
            from my_strategy.runtime_env import configure_utf8_stdio
            configure_utf8_stdio()
            print(json.dumps(run_probe(),ensure_ascii=False,indent=2))
        finally:
            os.chdir(previous)
            gc.collect()


if __name__=='__main__':
    main()
