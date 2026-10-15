import copy
import sqlite3
import threading

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import incremental_sync as sync
from my_strategy.storage.research_store import ResearchStore
from my_strategy.storage.board_fluent import BoardFluentStore
from my_strategy.tests.test_board_features import market
from my_strategy.tests.test_board_fluent import actual_market


@pytest.fixture
def planned(actual_market,tmp_path,monkeypatch):
    from my_strategy.services import czsc_trend_support
    dates=actual_market['dates'];cache=set();calls=[]
    monkeypatch.setattr(czsc_trend_support,'_calendar',lambda _: (dates,'fixture-calendar'))
    monkeypatch.setattr(sync,'code_identity',lambda:{'producer':'fixture-v1'})
    class Source:
        def __init__(self,**kwargs):pass
        def inspect(self,symbol,end,**kwargs):
            return {'symbol':symbol,'price_date':end,'observed_end':end,'action':'reuse' if symbol in cache else 'rebuild',
                'reason':None,'raw_rows':100,'identity':{'symbol':symbol,'date':end,'source':'fixture'}}
        def source(self,symbol,end,**kwargs):
            reused=symbol in cache
            if not reused:calls.append(symbol);cache.add(symbol)
            frame=pd.DataFrame({'date':[end]});frame.attrs['incremental']={'mode':'reused' if reused else 'single_stock_full_replay','reused':reused}
            return None,frame,[],{}
        def summary(self,symbol,end,**kwargs):
            evidence={'input_eligible':True,'buy_score':.7,'sell_score':.1,'direction_state':'BUY',
                'active_buy_event_ids':[symbol+':buy'],'active_sell_event_ids':[],'reason_codes':[]}
            value={'symbol':symbol,'price_date':end,'mode':'current_reference','training_eligible':False,'status':'available',
                'structure':evidence,'wyckoff':evidence,'identity':{'symbol':symbol,'date':end}}
            value['row_hash']=stable_hash(value);return value
    monkeypatch.setattr(sync,'IncrementalSource',Source)
    store=ResearchStore(tmp_path/'cache');options={'store':store,'raw_db_path':actual_market['path'],
        'kb_db_path':actual_market['kb'],'fluent_db_path':actual_market['fluent']}
    yield options,calls,cache,actual_market
    store.close()


def test_plan_no_compute_local_budget_and_same_generation_partial(planned):
    options,calls,cache,market=planned
    board=market['store'].catalog(current=True)[0]
    plan=sync.create_plan(**options,scope='current_board',board_id=board['board_id'],max_symbols=1)
    assert calls==[] and plan['estimated_compute']==plan['requested']
    assert plan['estimated_deferred']==plan['requested']-1
    result=sync.execute_plan(**options,plan_id=plan['plan_id'])
    assert result['status']=='partial' and len(calls)==1 and not result['full_published']
    assert result['counts']['deferred']==plan['requested']-1
    assert sum(result['counts'].values())==result['requested']
    store=BoardFluentStore(market['fluent'],readonly=True)
    generation=store.current_generation('market')
    assert generation['generation_id']==result['generation_id']
    assert not generation['full_ready'] and store.current_generation('full') is None
    assert store.board(board['board_id'])['native_structure']
    assert store.board(board['board_id'])['native_component_complete'] is False


def test_new_plan_reuses_successes_and_accumulates_partial_coverage(planned):
    options,calls,cache,market=planned;symbols=sorted(market['symbols'])
    first=sync.create_plan(**options,scope='selected',symbols=symbols[:1])
    one=sync.execute_plan(**options,plan_id=first['plan_id'])
    second=sync.create_plan(**options,scope='selected',symbols=symbols[1:2])
    two=sync.execute_plan(**options,plan_id=second['plan_id'])
    store=BoardFluentStore(market['fluent'],readonly=True)
    latest=store.get_native_component(two['generation_id'],allow_partial=True)
    assert latest['identity']['completed_symbols']==symbols[:2]
    repeat=sync.create_plan(**options,scope='selected',symbols=symbols[:2])
    before=len(calls);done=sync.execute_plan(**options,plan_id=repeat['plan_id'])
    assert len(calls)==before and done['counts']['reused']==2
    assert not done['full_published']


def test_only_explicit_full_scope_can_publish_all_layers(planned):
    options,calls,cache,market=planned
    selected=sync.create_plan(**options,scope='selected',symbols=market['symbols'])
    local=sync.execute_plan(**options,plan_id=selected['plan_id'])
    assert local['status']=='completed' and not local['full_published']
    plan=sync.create_plan(**options,scope='full')
    before=len(calls);result=sync.execute_plan(**options,plan_id=plan['plan_id'])
    assert result['status']=='completed' and result['full_published'] and len(calls)==before
    store=BoardFluentStore(market['fluent'],readonly=True)
    assert store.current_generation('full')['generation_id']==store.current_generation('market')['generation_id']


def test_changed_input_rejects_plan_before_any_compute(planned):
    options,calls,cache,market=planned
    plan=sync.create_plan(**options,scope='selected',symbols=market['symbols'][:1])
    with sqlite3.connect(market['path']) as con:con.execute('UPDATE stock_daily_normalized SET volume=volume+1 WHERE rowid=1')
    with pytest.raises(ValueError,match='输入已变化'):sync.execute_plan(**options,plan_id=plan['plan_id'])
    assert calls==[]


def test_cancel_keeps_success_objects_but_never_restarts_old_plan(planned):
    options,calls,cache,market=planned
    plan=sync.create_plan(**options,scope='selected',symbols=market['symbols'][:2])
    stop=threading.Event()
    result=sync.execute_plan(**options,plan_id=plan['plan_id'],stop=stop,progress=lambda _:stop.set())
    assert result['status']=='cancelled' and len(calls)==1
    with pytest.raises(ValueError,match='取消'):sync.execute_plan(**options,plan_id=plan['plan_id'])
    next_plan=sync.create_plan(**options,scope='selected',symbols=market['symbols'][:2])
    assert next_plan['counts']['reuse']==1
    assert next_plan['plan_id']!=plan['plan_id']


def test_cancelled_legacy_sync_is_terminal_even_for_same_fingerprint(tmp_path):
    store=BoardFluentStore(tmp_path/'fluent.db')
    first=store.acquire_sync('same',{'input':1})
    store.finish_sync(first['sync_id'],status='cancelled')
    with pytest.raises(ValueError,match='terminal'):store.acquire_sync('same',{'input':1},sync_id=first['sync_id'])
    second=store.acquire_sync('same',{'input':1})
    assert second['sync_id']!=first['sync_id']
    assert store.get_sync(first['sync_id'])['status']=='cancelled'


def test_expired_plan_cannot_execute(planned):
    options,calls,cache,market=planned
    plan=sync.create_plan(**options,scope='selected',symbols=market['symbols'][:1])
    options['store'].clock=lambda:plan['expires_at']+1
    with pytest.raises(KeyError):sync.execute_plan(**options,plan_id=plan['plan_id'])
    assert calls==[]


def test_plan_runtime_summaries_expire_without_accumulating_run_history(planned):
    options,calls,cache,market=planned
    plan=sync.create_plan(**options,scope='selected',symbols=market['symbols'][:1])
    options['store'].set_setting('incremental_plan:'+plan['plan_id'],{'status':'cancelled'})
    options['store'].clock=lambda:plan['expires_at']+1
    sync.create_plan(**options,scope='selected',symbols=market['symbols'][:1])
    assert options['store'].get_setting('incremental_plan:'+plan['plan_id']) is None


def test_partial_native_producer_is_checked_without_legacy_cache_dependency(planned,monkeypatch):
    from my_strategy.services import board_fluent,incremental_source
    options,calls,cache,market=planned
    plan=sync.create_plan(**options,scope='selected',symbols=market['symbols'][:1])
    result=sync.execute_plan(**options,plan_id=plan['plan_id'])
    generation=BoardFluentStore(market['fluent'],readonly=True).current_generation('market')
    assert not result['full_published'] and 'native_cache_witness' not in generation['publication_inputs']
    monkeypatch.setattr(board_fluent,'source_witness',lambda **_:generation['publication_inputs']['source_witness'])
    monkeypatch.setattr(incremental_source,'code_identity',lambda:{'producer':'fixture-v1'})
    params={key:value for key,value in options.items() if key!='store'}
    assert not board_fluent.freshness_status(**params)['native_stale']
    monkeypatch.setattr(incremental_source,'code_identity',lambda:{'producer':'changed'})
    assert board_fluent.freshness_status(**params)['native_stale']
