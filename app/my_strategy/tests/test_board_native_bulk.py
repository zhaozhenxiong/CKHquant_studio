"""Bounded full-native orchestration uses isolated stores and frozen inputs."""
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
import threading
import time

import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import board_native_bulk as bulk
from my_strategy.services import incremental_sync as incremental
from my_strategy.storage.board_fluent import BoardFluentStore
from my_strategy.tests.test_incremental_sync import planned
from my_strategy.tests.test_board_fluent import actual_market
from my_strategy.tests.test_board_features import market


@pytest.fixture
def setup(planned,tmp_path,monkeypatch):
    options,calls,cache,market=planned
    base=incremental.IncrementalSource
    class Source(base):
        def __init__(self,**kwargs):
            super().__init__(**kwargs);self.store=kwargs['store']
        def summary(self,symbol,end,**kwargs):
            self.source(symbol,end,**kwargs)
            result=super().summary(symbol,end,**kwargs)
            result['identity']=self.inspect(symbol,end,**kwargs)['identity']
            result['row_hash']=stable_hash({key:value for key,value in result.items() if key!='row_hash'})
            return result
    monkeypatch.setattr(bulk,'IncrementalSource',Source)
    monkeypatch.setattr(bulk,'code_identity',incremental.code_identity)
    checkpoint=tmp_path/'native-bulk.json'
    state=bulk.preview_bulk(**options,checkpoint=checkpoint,cpu_workers=1,batch_size=2)
    return options,checkpoint,state,calls,cache,market,Source


def test_preview_does_not_compute_or_publish_and_limits_are_checked(setup,tmp_path):
    options,path,state,calls,cache,market,_=setup
    assert calls==[] and cache==set() and not market['fluent'].exists()
    assert state['status']=='planned' and len(state['rows'])==7
    assert all('identity' not in row and row['source_identity_sha256'] for row in state['rows'])
    with pytest.raises(ValueError,match='exists'):bulk.preview_bulk(**options,checkpoint=path)
    for field,value in [('cpu_workers',13),('batch_size',101)]:
        with pytest.raises(ValueError):bulk.preview_bulk(**options,checkpoint=tmp_path/'bad.json',**{field:value})


def test_partial_batches_never_aggregate_then_resume_publishes_exactly_once(setup,monkeypatch):
    options,path,state,calls,cache,market,_=setup
    import my_strategy.services.board_fluent as fluent
    original=fluent.build_board_fluent;publications=[]
    def publish(**kwargs):
        publications.append(kwargs['native_result']);return original(**kwargs)
    monkeypatch.setattr(fluent,'build_board_fluent',publish)
    first=bulk.execute_bulk(store=options['store'],checkpoint=path,max_stocks=2)
    assert first['status']=='partial' and first['completed']==2 and first['deferred']==5,first.get('error')
    assert not publications and not market['fluent'].exists()
    second=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert second['status']=='completed' and second['full_published'] and len(calls)==7
    assert len(publications)==1 and list(publications[0]['summaries'])==sorted(market['symbols'])
    store=BoardFluentStore(market['fluent'],readonly=True)
    assert store.current_generation('market')['generation_id']==store.current_generation('full')['generation_id']==second['generation_id']
    assert store.verify_generation(second['generation_id'],level='full')
    assert bulk.execute_bulk(store=options['store'],checkpoint=path)==second
    assert len(publications)==1


def test_checkpoint_summary_matches_existing_source_algorithm(setup):
    options,path,state,_,_,market,Source=setup
    result=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert result['status']=='completed',result.get('error')
    source=Source(store=options['store'],raw_db=market['path'])
    published=BoardFluentStore(market['fluent'],readonly=True).get_native_component(result['generation_id'])
    assert published['summaries']=={symbol:source.summary(symbol,state['price_date'],calendar_dates=state['dates']) for symbol in market['symbols']}


def test_soft_budget_stops_new_dispatch_and_preserves_an_inflight_success(setup,monkeypatch):
    options,path,_,_,_,market,Source=setup;original=Source.summary;started=[]
    def slow(self,symbol,end,**kwargs):
        started.append(symbol);time.sleep(.03);return original(self,symbol,end,**kwargs)
    monkeypatch.setattr(Source,'summary',slow)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path,budget_seconds=.02)
    assert result['status']=='partial' and result['completed']==1 and len(started)==1
    assert not market['fluent'].exists()
    assert 'finish already running' in result['budget_policy']


def test_cancelled_task_is_terminal_but_new_plan_reuses_immutable_sources(setup,monkeypatch,tmp_path):
    options,path,_,calls,_,market,_=setup;stop=threading.Event()
    def progress(value):
        if value['stage']=='native_stock':stop.set()
    result=bulk.execute_bulk(store=options['store'],checkpoint=path,stop=stop,progress=progress)
    assert result['status']=='cancelled' and len(calls)==1 and not market['fluent'].exists()
    with pytest.raises(ValueError,match='terminal'):bulk.execute_bulk(store=options['store'],checkpoint=path)
    fresh=bulk.preview_bulk(**options,checkpoint=tmp_path/'new.json',cpu_workers=1)
    assert fresh['bulk_id']!=result['bulk_id'] and fresh['counts']['reuse']==1
    assert bulk._load(path)['status']=='cancelled'


@pytest.mark.parametrize('changed',['raw','producer','executor','calendar'])
def test_changed_inputs_or_code_reject_before_any_execution(setup,monkeypatch,changed):
    options,path,state,calls,_,market,_=setup
    if changed=='raw':
        with sqlite3.connect(market['path']) as conn:conn.execute('UPDATE stock_daily_normalized SET volume=volume+1 WHERE rowid=1')
    elif changed=='producer':monkeypatch.setattr(bulk,'code_identity',lambda:{'different':True})
    elif changed=='executor':monkeypatch.setattr(bulk,'_executor_identity',lambda:{'different':True})
    else:
        from my_strategy.services import czsc_trend_support
        monkeypatch.setattr(czsc_trend_support,'_calendar',lambda _: (state['dates'][:-1],'different'))
    with pytest.raises(ValueError,match='changed'):bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert calls==[] and not market['fluent'].exists()


def test_stock_checkpoint_tampering_never_publishes(setup):
    options,path,_,_,_,market,_=setup
    bulk.execute_bulk(store=options['store'],checkpoint=path,max_stocks=1)
    state=bulk._load(path);record=next(iter(state['records'].values()))
    from pathlib import Path
    Path(record['summary_file']).write_text('{}')
    result=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert result['status']=='failed' and 'SHA mismatch' in result['error'] and not market['fluent'].exists()


def test_failed_stock_cannot_be_marked_full_and_explicit_retry_repairs_it(setup,monkeypatch):
    options,path,_,_,_,market,Source=setup;original=Source.summary
    def faulty(self,symbol,end,**kwargs):
        if symbol==market['symbols'][0]:raise OSError('fixture missing object')
        return original(self,symbol,end,**kwargs)
    monkeypatch.setattr(Source,'summary',faulty)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert result['status']=='partial' and result['failed']==1 and not result['full_published']
    assert not market['fluent'].exists()
    monkeypatch.setattr(Source,'summary',original)
    done=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert done['status']=='completed' and done['full_published']


def test_final_identity_change_prevents_all_publication(setup,monkeypatch):
    options,path,state,_,_,market,Source=setup;original=Source.inspect;count=[]
    def altered(self,symbol,end,**kwargs):
        value=original(self,symbol,end,**kwargs)
        if len(count)>=len(market['symbols']):value['identity']={**value['identity'],'source':'changed'}
        count.append(symbol);return value
    monkeypatch.setattr(Source,'inspect',altered)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert result['status']=='failed' and not result['full_published'] and not market['fluent'].exists()


def test_out_of_order_workers_are_bounded_and_publish_in_frozen_order(setup,monkeypatch):
    options,path,state,calls,cache,market,Source=setup;active=0;maximum=0;lock=threading.Lock();finished=[]
    source=Source(store=options['store'],raw_db=market['path'])
    def worker(symbol,day,dates):
        nonlocal active,maximum
        with lock:active+=1;maximum=max(maximum,active)
        time.sleep(.005*(8-int(symbol[5])))
        result={'symbol':symbol,'summary':source.summary(symbol,day,calendar_dates=dates)}
        with lock:active-=1;finished.append(symbol)
        return result
    monkeypatch.setattr(bulk,'_worker_summary',worker)
    monkeypatch.setattr(bulk,'_worker_verify',lambda symbol,day,dates,identity:bulk._source_verification(
        source,symbol,day,dates,identity,None))
    monkeypatch.setattr(bulk,'_pool',lambda workers,*_:ThreadPoolExecutor(max_workers=workers))
    # Resource selection changes no numerical/plan identity.
    result=bulk.execute_bulk(store=options['store'],checkpoint=path,cpu_workers=3)
    assert result['status']=='completed' and 1<maximum<=3 and finished!=sorted(finished),result.get('error')
    published=BoardFluentStore(market['fluent'],readonly=True).get_native_component(result['generation_id'])
    assert published['identity']['required_symbols']==sorted(market['symbols'])


def test_unknown_or_unavailable_history_remains_failed_without_fabricated_full(setup):
    options,path,state,_,_,market,_=setup
    state['rows'][0].update(action='failed',reason='股票没有已核验行情')
    bulk._save(path,state)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert result['status']=='partial' and result['failed']==1 and not result['full_published']
    assert not market['fluent'].exists()


def test_checkpoint_index_sha_and_population_are_enforced(setup):
    options,path,state,calls,_,market,_=setup
    corrupted=json.loads(path.read_text());corrupted['rows'].pop();path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError,match='SHA mismatch'):bulk.execute_bulk(store=options['store'],checkpoint=path)
    bulk._save(path,corrupted)
    with pytest.raises(ValueError,match='population'):bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert calls==[]


@pytest.mark.parametrize('reason',['target_day_price_missing','verified_history_unavailable'])
def test_explicit_missing_price_stays_unavailable_in_complete_population(setup,reason):
    options,path,state,_,_,market,_=setup
    symbol=state['rows'][0]['symbol']
    with sqlite3.connect(market['path']) as conn:
        if reason=='verified_history_unavailable':conn.execute('DELETE FROM stock_daily_normalized WHERE stock=?',(symbol,))
        else:conn.execute('DELETE FROM stock_daily_normalized WHERE stock=? AND date=?',(symbol,state['price_date']))
    path=path.with_name('actual-missing.json')
    state=bulk.preview_bulk(**options,checkpoint=path,cpu_workers=1)
    state['rows'][0].update(action='missing_price',
        observed_end=None if reason=='verified_history_unavailable' else market['dates'][-2],reason=reason)
    bulk._save(path,state)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert result['status']=='completed' and result['full_published']
    rows=BoardFluentStore(market['fluent'],readonly=True).get_native_component(result['generation_id'])['summaries']
    assert len(rows)==len(market['symbols']) and rows[symbol]['status']=='unavailable'
    assert rows[symbol]['reason_codes']==[reason] and rows[symbol]['structure'] is None and rows[symbol]['wyckoff'] is None


@pytest.mark.parametrize('reason',['target_day_price_missing','verified_history_unavailable'])
def test_rehashed_missing_price_action_cannot_hide_actual_current_price(setup,reason):
    options,path,state,_,_,market,_=setup
    state['rows'][0].update(action='missing_price',observed_end=None,reason=reason)
    bulk._save(path,state)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert result['status']=='failed' and 'actual raw history' in result['error'],result
    assert not result['full_published'] and not market['fluent'].exists()


@pytest.mark.parametrize('failure',['bad_ohlc','schema','io','wrong_reason','wrong_observed_end','missing_reason','missing_observed_end'])
def test_missing_price_requires_actual_valid_history_and_canonical_reason(setup,monkeypatch,failure):
    options,path,state,_,_,market,_=setup;symbol=state['rows'][0]['symbol']
    with sqlite3.connect(market['path']) as conn:
        conn.execute('DELETE FROM stock_daily_normalized WHERE stock=? AND date=?',(symbol,state['price_date']))
        if failure=='bad_ohlc':conn.execute('UPDATE stock_daily_normalized SET high=0 WHERE stock=?',(symbol,))
        elif failure=='schema':conn.execute('ALTER TABLE stock_daily_normalized RENAME COLUMN low TO missing_low')
    path=path.with_name('invalid-missing.json');state=bulk.preview_bulk(**options,checkpoint=path,cpu_workers=1)
    state['rows'][0].update(action='missing_price',observed_end=market['dates'][-2],reason='target_day_price_missing')
    if failure=='wrong_reason':state['rows'][0]['reason']='verified_history_unavailable'
    elif failure=='wrong_observed_end':state['rows'][0]['observed_end']=market['dates'][-3]
    elif failure=='missing_reason':state['rows'][0].pop('reason')
    elif failure=='missing_observed_end':state['rows'][0].pop('observed_end')
    elif failure=='io':
        from my_strategy.adapters import czsc_adapter
        def unreadable(*args,**kwargs):raise OSError('isolated input I/O failure')
        monkeypatch.setattr(czsc_adapter,'load_bars',unreadable)
    bulk._save(path,state)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert result['status']=='failed' and not result['full_published'] and not market['fluent'].exists(),result


def test_atomic_stock_checkpoint_survives_missing_batch_index_commit(setup):
    options,path,_,calls,_,market,_=setup
    bulk.execute_bulk(store=options['store'],checkpoint=path,max_stocks=1)
    state=bulk._load(path);state['records'].clear();bulk._save(path,state)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path,max_stocks=1)
    assert result['status']=='partial' and result['completed']==2 and len(calls)==2
    assert sum(row.get('recovered_atomic_stock',False) for row in bulk._load(path)['records'].values())==1


def test_rehashed_checkpoint_numeric_forgery_does_not_replace_verified_cas_summary(setup):
    import hashlib
    from pathlib import Path
    options,path,state,_,_,market,_=setup
    bulk.execute_bulk(store=options['store'],checkpoint=path,max_stocks=1)
    state=bulk._load(path);record=next(iter(state['records'].values()));stock_path=Path(record['summary_file'])
    payload=json.loads(stock_path.read_text());summary=payload['summary']
    summary['structure']['buy_score']=.999
    summary['row_hash']=stable_hash({key:value for key,value in summary.items() if key!='row_hash'})
    bulk._atomic_json(stock_path,payload);record['summary_sha256']=hashlib.sha256(stock_path.read_bytes()).hexdigest()
    bulk._save(path,state)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path)
    assert result['status']=='failed' and 'verified numerical CAS' in result['error']
    assert not market['fluent'].exists()


def test_real_spawn_workers_use_independent_cache_connections_and_publish_same_algorithms(actual_market,tmp_path):
    import uuid
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    from my_strategy.storage.research_store import ResearchStore
    from my_strategy.services.incremental_source import IncrementalSource
    calendar=ARTIFACT_RUNS_ROOT/('bulk-worker-fixture-'+uuid.uuid4().hex)/'reports/calendar.json'
    calendar.parent.mkdir(parents=True,exist_ok=True)
    calendar.write_text(json.dumps({'verified':True,'source':'isolated synthetic schedule',
        'dates':actual_market['dates']}))
    checkpoint=tmp_path/'real-spawn.json'
    with ResearchStore(tmp_path/'real-cache') as store:
        state=bulk.preview_bulk(store=store,checkpoint=checkpoint,raw_db_path=actual_market['path'],
            kb_db_path=actual_market['kb'],fluent_db_path=actual_market['fluent'],cpu_workers=2,batch_size=3)
        sentinel=store.put({'id':'account-preserved'},{'account':'isolated'},namespace='lease_sentinel')
        store.acquire_lease(sentinel['cache_key'],'other-account-owner',ttl_seconds=3600)
        pilot=bulk.execute_bulk(store=store,checkpoint=checkpoint,budget_seconds=60,max_stocks=2)
        assert pilot['status']=='partial' and pilot['completed']==2
        assert pilot['cache_metrics']['quota_bytes']==8*1024**3
        assert pilot['cache_metrics']['new_cas_objects']>0 and pilot['cache_metrics']['new_cas_object_bytes']>0
        owner='board_native_bulk:'+state['bulk_id']
        leased=store.connection.execute('SELECT r.sha256 FROM leases l JOIN results r USING(cache_key) WHERE l.owner=?',(owner,)).fetchall()
        assert len(leased)>=4
        dry=store.gc(dry_run=True,required_bytes=store.quota_bytes)
        assert not {row[0] for row in leased}.intersection(row['sha256'] for row in dry['candidates'])
        result=bulk.execute_bulk(store=store,checkpoint=checkpoint,budget_seconds=60,cache_quota_bytes=32*1024**3)
        assert result['status']=='completed' and result['full_published'],result
        assert store.quota_bytes==8*1024**3 and result['cache_metrics']['quota_bytes']==32*1024**3
        assert bulk._load(checkpoint)['contract']==state['contract']
        assert {record['cache_quota_bytes'] for record in bulk._load(checkpoint)['records'].values()}=={8*1024**3,32*1024**3}
        source=IncrementalSource(store=store,raw_db=actual_market['path'])
        native=BoardFluentStore(actual_market['fluent'],readonly=True).get_native_component(result['generation_id'])
        assert len(native['summaries'])==7
        assert native['summaries']=={symbol:source.summary(symbol,state['price_date'],calendar_dates=state['dates'])
            for symbol in actual_market['symbols']}
        assert store.connection.execute('SELECT COUNT(*) FROM leases WHERE owner=?',(owner,)).fetchone()[0]==0
        assert store.connection.execute("SELECT COUNT(*) FROM leases WHERE owner='other-account-owner'").fetchone()[0]==1


@pytest.mark.parametrize('quota',[0,-1,True,64*1024**3+1])
def test_invalid_capacity_override_does_not_start_or_change_global_defaults(setup,quota):
    options,path,state,calls,_,_,_=setup;before=options['store'].quota_bytes
    with pytest.raises(ValueError,match='cache_quota_bytes'):
        bulk.execute_bulk(store=options['store'],checkpoint=path,cache_quota_bytes=quota)
    assert options['store'].quota_bytes==before and calls==[]


@pytest.mark.parametrize('workers',[0,-1,13,True,1.5,'12'])
def test_invalid_worker_override_does_not_compute_or_mutate_checkpoint(setup,tmp_path,workers):
    options,path,state,calls,_,_,_=setup;before=path.read_bytes()
    with pytest.raises(ValueError,match='cpu_workers'):
        bulk.execute_bulk(store=options['store'],checkpoint=path,cpu_workers=workers)
    with pytest.raises(ValueError,match='cpu_workers'):
        bulk.preview_bulk(**options,checkpoint=tmp_path/'invalid-workers.json',cpu_workers=workers)
    assert calls==[] and path.read_bytes()==before


@pytest.mark.parametrize('workers',['0','-1','13','1.5','True'])
@pytest.mark.parametrize('action',['preview','execute'])
def test_cli_rejects_invalid_worker_count_before_opening_cache(monkeypatch,tmp_path,workers,action):
    from my_strategy.scripts.fill_board_native import main
    import my_strategy.storage.research_store as research
    monkeypatch.setattr(research,'ResearchStore',lambda *a,**k:pytest.fail('invalid worker count opened cache'))
    with pytest.raises(SystemExit) as error:
        main([action,'--checkpoint',str(tmp_path/'cli.json'),'--cpu-workers',workers])
    assert error.value.code==2


def test_explicit_twelve_workers_leave_default_and_frozen_executor_gate_unchanged(setup,tmp_path,monkeypatch):
    options,path,state,calls,_,_,_=setup
    default=bulk.preview_bulk(**options,checkpoint=tmp_path/'default.json')
    explicit=bulk.preview_bulk(**options,checkpoint=tmp_path/'twelve.json',cpu_workers=12)
    assert default['cpu_workers']==4 and explicit['cpu_workers']==12
    assert default['contract']==explicit['contract']==state['contract']
    monkeypatch.setattr(bulk,'_executor_identity',lambda:{'new_worker_limit_executor':'changed'})
    with pytest.raises(ValueError,match='executor changed'):
        bulk.execute_bulk(store=options['store'],checkpoint=path,cpu_workers=12)
    assert calls==[] and bulk._load(path)['status']=='planned'


_TWELVE_WORKER_BARRIER=None
_TWELVE_CANCEL_FINAL=False


def _initialize_twelve_workers(root,raw_db,stop,quota_bytes,barrier,cancel_final=False):
    global _TWELVE_WORKER_BARRIER,_TWELVE_CANCEL_FINAL
    bulk._worker_initialize(root,raw_db,stop,quota_bytes)
    _TWELVE_WORKER_BARRIER=barrier
    _TWELVE_CANCEL_FINAL=cancel_final


def _twelve_worker_summary(symbol,day,dates):
    # Every worker must enter this bounded barrier before any can finish a
    # stock: proves twelve actual child processes, not only max_workers=12.
    _TWELVE_WORKER_BARRIER.wait(timeout=60)
    return bulk._worker_summary(symbol,day,dates)


def _twelve_worker_verify(symbol,day,dates,identity):
    _TWELVE_WORKER_BARRIER.wait(timeout=60)
    if _TWELVE_CANCEL_FINAL and symbol=='000001.SZ':return {'symbol':symbol,'cancelled':True}
    return bulk._worker_verify(symbol,day,dates,identity)


@pytest.mark.parametrize('cancel_final',[False,True])
def test_twelve_real_spawn_workers_keep_exact_final_verification_and_cancel(actual_market,tmp_path,monkeypatch,cancel_final):
    import multiprocessing,uuid
    from concurrent.futures import ProcessPoolExecutor
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    from my_strategy.storage.board_knowledge import FORMAT
    from my_strategy.storage.research_store import ResearchStore
    from my_strategy.services.incremental_source import IncrementalSource
    import pandas as pd
    # Expand only this isolated fixture to twelve distinct SH/SZ stocks.
    extras=[f'{index:06d}.SZ' for index in range(8,13)]
    with sqlite3.connect(actual_market['path']) as conn:
        for symbol in extras:
            conn.execute('INSERT INTO stock_daily_normalized SELECT ?,date,open,high,low,close,volume,amount,source,has_trade_price FROM stock_daily_normalized WHERE stock=?',
                (symbol,actual_market['symbols'][0]))
    symbols=sorted([*actual_market['symbols'],*extras]);now=pd.Timestamp.now(tz='Asia/Shanghai').isoformat()
    for old in actual_market['store'].catalog(current=True):
        row=actual_market['snapshot'](old['kind'],old['source_code'],members=symbols,at=now)
        row.pop('snapshot_hash')
        row.update(format=FORMAT,source_url='https://q.10jqka.com.cn/thshy/',
            provenance={'pagination_complete':True,'reported_count':12,'unique_members':12})
        actual_market['store'].import_snapshot(row)
    calendar=ARTIFACT_RUNS_ROOT/('bulk-twelve-fixture-'+uuid.uuid4().hex)/'reports/calendar.json'
    calendar.parent.mkdir(parents=True,exist_ok=True)
    calendar.write_text(json.dumps({'verified':True,'source':'isolated synthetic schedule','dates':actual_market['dates']}))
    context=multiprocessing.get_context('spawn')
    with context.Manager() as manager:
        barrier=manager.Barrier(12)
        monkeypatch.setattr(bulk,'_worker_summary',_twelve_worker_summary)
        monkeypatch.setattr(bulk,'_worker_verify',_twelve_worker_verify)
        monkeypatch.setattr(bulk,'_pool',lambda workers,root,raw,stop,quota:ProcessPoolExecutor(
            max_workers=workers,mp_context=context,initializer=_initialize_twelve_workers,
            initargs=(root,raw,stop,quota,barrier,cancel_final)))
        checkpoint=tmp_path/'twelve-spawn.json'
        with ResearchStore(tmp_path/'twelve-cache') as store:
            state=bulk.preview_bulk(store=store,checkpoint=checkpoint,raw_db_path=actual_market['path'],
                kb_db_path=actual_market['kb'],fluent_db_path=actual_market['fluent'],cpu_workers=12,batch_size=12)
            assert len(state['rows'])==12
            result=bulk.execute_bulk(store=store,checkpoint=checkpoint,budget_seconds=120)
            records=bulk._load(checkpoint)['records']
            assert len({row['worker_pid'] for row in records.values()})==12
            if cancel_final:
                assert result['status']=='cancelled' and not result['full_published'] and not actual_market['fluent'].exists(),result
                with pytest.raises(ValueError,match='terminal'):bulk.execute_bulk(store=store,checkpoint=checkpoint)
                return
            assert result['status']=='completed' and result['full_published'],result
            assert result['final_verification']['verified']==12
            assert len(result['final_verification']['worker_pids'])==12
            assert set(result['final_verification']['worker_pids'])=={row['worker_pid'] for row in records.values()}
            source=IncrementalSource(store=store,raw_db=actual_market['path'])
            native=BoardFluentStore(actual_market['fluent'],readonly=True).get_native_component(result['generation_id'])
            assert native['summaries']=={symbol:source.summary(symbol,state['price_date'],calendar_dates=state['dates']) for symbol in symbols}


@pytest.fixture
def parallel_setup(setup,monkeypatch):
    options,path,state,calls,cache,market,Source=setup
    source=Source(store=options['store'],raw_db=market['path'])
    monkeypatch.setattr(bulk,'_pool',lambda workers,*_:ThreadPoolExecutor(max_workers=workers))
    monkeypatch.setattr(bulk,'_worker_summary',lambda symbol,day,dates:{
        'symbol':symbol,'summary':source.summary(symbol,day,calendar_dates=dates)})
    verify=lambda symbol,day,dates,identity:bulk._source_verification(source,symbol,day,dates,identity,None)
    monkeypatch.setattr(bulk,'_worker_verify',verify)
    return (*setup,source,verify)


def test_parallel_final_verification_is_bounded_and_keeps_population_order(parallel_setup,monkeypatch):
    options,path,state,calls,_,market,_,source,verify=parallel_setup
    active=0;maximum=0;finished=[];lock=threading.Lock()
    def unordered(symbol,day,dates,identity):
        nonlocal active,maximum
        with lock:active+=1;maximum=max(active,maximum)
        try:
            time.sleep(.005*(8-int(symbol[5])))
            return verify(symbol,day,dates,identity)
        finally:
            with lock:active-=1;finished.append(symbol)
    monkeypatch.setattr(bulk,'_worker_verify',unordered)
    import my_strategy.services.board_fluent as fluent
    original=fluent.build_board_fluent;published=[]
    def publish(**kwargs):
        published.append(list(kwargs['native_result']['summaries']));return original(**kwargs)
    monkeypatch.setattr(fluent,'build_board_fluent',publish)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path,cpu_workers=3)
    assert result['status']=='completed' and result['final_verification']['verified']==7,result
    assert 1<maximum<=3 and finished!=sorted(finished)
    assert published==[[row['symbol'] for row in state['rows']]]
    assert len(calls)==7


@pytest.mark.parametrize('tamper',['numeric_summary','row_hash','source_cache_key','observed_identity','wrong_symbol','cache_quota','worker_error'])
def test_parallel_final_bad_actual_proof_never_publishes(parallel_setup,monkeypatch,tamper):
    options,path,_,_,_,market,_,source,verify=parallel_setup
    def damaged(symbol,day,dates,identity):
        result=verify(symbol,day,dates,identity)
        if symbol!=market['symbols'][0]:return result
        if tamper=='numeric_summary':
            result['summary']['structure']['buy_score']=.999
            result['summary']['row_hash']=stable_hash({key:value for key,value in result['summary'].items() if key!='row_hash'})
        elif tamper=='row_hash':result['summary']['row_hash']='0'*64
        elif tamper=='source_cache_key':result['observed']['cache_key']='f'*64
        elif tamper=='observed_identity':result['observed']['identity']['source']='forged'
        elif tamper=='wrong_symbol':result['symbol']='600000.SH'
        elif tamper=='cache_quota':result['cache_quota_bytes']=1
        else:result={'symbol':symbol,'error':'fixture verification failed'}
        return result
    monkeypatch.setattr(bulk,'_worker_verify',damaged)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path,cpu_workers=3)
    assert result['status']=='failed' and not result['full_published'] and not market['fluent'].exists(),result


def test_parallel_final_rehashed_checkpoint_numeric_forgery_is_rejected(parallel_setup):
    import hashlib
    from pathlib import Path
    options,path,_,_,_,market,_,_,_=parallel_setup
    bulk.execute_bulk(store=options['store'],checkpoint=path,max_stocks=1)
    state=bulk._load(path);record=next(iter(state['records'].values()));stock_path=Path(record['summary_file'])
    payload=json.loads(stock_path.read_text());summary=payload['summary']
    summary['structure']['buy_score']=.999
    summary['row_hash']=stable_hash({key:value for key,value in summary.items() if key!='row_hash'})
    bulk._atomic_json(stock_path,payload);record['summary_sha256']=hashlib.sha256(stock_path.read_bytes()).hexdigest()
    bulk._save(path,state)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path,cpu_workers=3)
    assert result['status']=='failed' and 'verified numerical CAS' in result['error']
    assert not result['full_published'] and not market['fluent'].exists()


@pytest.mark.parametrize('cancel_kind',['parent_stop','worker_cancelled'])
def test_parallel_final_cancel_is_terminal_and_never_publishes(parallel_setup,monkeypatch,cancel_kind):
    options,path,_,calls,_,market,_,_,verify=parallel_setup;stop=threading.Event();started=[]
    def cancel(symbol,day,dates,identity):
        started.append(symbol)
        if symbol==market['symbols'][0]:
            if cancel_kind=='parent_stop':stop.set()
            else:return {'symbol':symbol,'cancelled':True}
        time.sleep(.01)
        return verify(symbol,day,dates,identity)
    monkeypatch.setattr(bulk,'_worker_verify',cancel)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path,cpu_workers=3,stop=stop)
    assert result['status']=='cancelled' and len(calls)==7 and 1<=len(started)<=3,result
    assert not result['full_published'] and not market['fluent'].exists()
    with pytest.raises(ValueError,match='terminal'):bulk.execute_bulk(store=options['store'],checkpoint=path)


def test_parallel_final_input_change_after_proof_cannot_publish(parallel_setup,monkeypatch):
    options,path,_,_,_,market,_,_,verify=parallel_setup;changed=threading.Event()
    def change(symbol,day,dates,identity):
        result=verify(symbol,day,dates,identity)
        if not changed.is_set():
            changed.set()
            with sqlite3.connect(market['path']) as conn:
                conn.execute('UPDATE stock_daily_normalized SET volume=volume+1 WHERE rowid=1')
        return result
    monkeypatch.setattr(bulk,'_worker_verify',change)
    result=bulk.execute_bulk(store=options['store'],checkpoint=path,cpu_workers=3)
    assert result['status']=='failed' and 'database changed' in result['error'] and not market['fluent'].exists(),result


@pytest.mark.parametrize('inspection',['rebuild','changed_identity'])
def test_final_verifier_does_not_rebuild_when_frozen_inspection_fails(setup,monkeypatch,inspection):
    options,path,state,calls,_,market,Source=setup
    source=Source(store=options['store'],raw_db=market['path']);row=state['rows'][0]
    original=source.inspect
    def bad(*args,**kwargs):
        observed=original(*args,**kwargs)
        if inspection=='rebuild':observed['action']='rebuild'
        else:observed['action']='reuse';observed['identity']['source']='different'
        return observed
    monkeypatch.setattr(source,'inspect',bad)
    monkeypatch.setattr(source,'summary',lambda *a,**k:pytest.fail('failed proof recomputed summary'))
    with pytest.raises(ValueError,match='proof changed'):
        bulk._source_verification(source,row['symbol'],state['price_date'],state['dates'],row['source_identity_sha256'],None)
    assert calls==[]


def test_parallel_final_corrupt_actual_cas_chunk_fails_and_preserves_other_leases(actual_market,tmp_path):
    import uuid
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    from my_strategy.storage.research_store import ResearchStore
    calendar=ARTIFACT_RUNS_ROOT/('bulk-corrupt-fixture-'+uuid.uuid4().hex)/'reports/calendar.json'
    calendar.parent.mkdir(parents=True,exist_ok=True)
    calendar.write_text(json.dumps({'verified':True,'source':'isolated synthetic schedule','dates':actual_market['dates']}))
    with ResearchStore(tmp_path/'bad-final-cache') as store:
        path=tmp_path/'bad-final.json'
        state=bulk.preview_bulk(store=store,checkpoint=path,raw_db_path=actual_market['path'],
            kb_db_path=actual_market['kb'],fluent_db_path=actual_market['fluent'],cpu_workers=2)
        sentinel=store.put({'id':'account-kept'},{'account':'isolated'},namespace='lease_sentinel')
        store.acquire_lease(sentinel['cache_key'],'other-account-owner',ttl_seconds=3600)
        owner='board_native_bulk:'+state['bulk_id'];corrupted=[]
        def corrupt_at_final(value):
            if value['stage']!='final_source_verification':return
            leased=store.connection.execute('SELECT r.sha256 FROM leases l JOIN results r USING(cache_key) WHERE l.owner=?',(owner,)).fetchall()
            dry=store.gc(dry_run=True,required_bytes=store.quota_bytes)
            assert leased and not {row[0] for row in leased}.intersection(row['sha256'] for row in dry['candidates'])
            current=bulk._load(path);row=current['rows'][0]
            saved=bulk._read_stock(path,current,row,current['records'][row['symbol']])
            manifest=store.get_by_key(saved['identity']['source_cache_key'],touch=False)
            key=manifest['chunks'][0]['cache_key']
            sha=store.connection.execute('SELECT sha256 FROM results WHERE cache_key=?',(key,)).fetchone()[0]
            chunk=store._path(sha);chunk.chmod(0o600)
            chunk.write_bytes(b'corrupt isolated chunk');corrupted.append(sha)
        result=bulk.execute_bulk(store=store,checkpoint=path,progress=corrupt_at_final)
        assert corrupted and result['status']=='failed' and not result['full_published'],result
        assert not actual_market['fluent'].exists()
        assert store.connection.execute('SELECT COUNT(*) FROM leases WHERE owner=?',(owner,)).fetchone()[0]==0
        assert store.connection.execute("SELECT COUNT(*) FROM leases WHERE owner='other-account-owner'").fetchone()[0]==1
