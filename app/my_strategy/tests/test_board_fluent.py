from __future__ import annotations
import copy
import hashlib
import multiprocessing
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from my_strategy.services import board_features as math_features
from my_strategy.services import board_fluent as service
from my_strategy.storage.board_fluent import BoardFluentStore, TABLES, canonical, digest, rank_boards
from my_strategy.core.run_context import stable_hash
from my_strategy.tests.test_board_features import market


def bundle(market):
    return math_features.build_board_features(market['store'],raw_db_path=market['path'],
        calendar_dates=market['dates'],price_date=market['dates'][-1],mode='current_reference')


@pytest.fixture
def actual_market(market,tmp_path):
    from my_strategy.storage.board_knowledge import BoardKnowledgeStore,FORMAT
    kb=tmp_path/'kb.db';store=BoardKnowledgeStore(kb)
    now=(pd.Timestamp.now(tz='Asia/Shanghai')-pd.Timedelta(seconds=2)).isoformat()
    for snapshot in market['store'].snapshots:
        row={key:value for key,value in snapshot.items() if key!='snapshot_hash'}
        row.update(format=FORMAT,observed_at=now,available_at=now,effective_at=now,
            source_url='https://q.10jqka.com.cn/thshy/',
            provenance={'pagination_complete':True,'reported_count':len(row['members']),'unique_members':len(row['members'])})
        store.import_snapshot(row)
    return {**market,'store':store,'kb':kb,'fluent':tmp_path/'fluent.db'}


def build(actual_market,**kwargs):
    return service.build_board_fluent(kb_db_path=actual_market['kb'],raw_db_path=actual_market['path'],
        fluent_db_path=actual_market['fluent'],calendar_dates=actual_market['dates'],**kwargs)


def _lease_child(path,ready,stop):
    store=BoardFluentStore(path)
    claim=store.acquire_sync('same',{'raw':'verified'})
    store.heartbeat_sync(claim['sync_id'],stage='market',checkpoint={'market_done':True})
    ready.put(claim['sync_id']);stop.wait(20)


def native_result(market,boards):
    day=market['dates'][-1]
    identity={'price_date':day,'mode':'current_reference','training_eligible':False,'source':'fixture',
              'calendar_hash':stable_hash(market['dates']),'required_symbols':sorted(market['symbols'])}
    rows={}
    for symbol in market['symbols']:
        row={'symbol':symbol,'price_date':day,'mode':'current_reference','training_eligible':False,
             'native_cache_key':'a'*64,'status':'available'}
        row['row_hash']=stable_hash(row);rows[symbol]=row
    aggregate=[]
    for board in boards:
        row={'board_id':board['board_id'],'price_date':day,'structure_basis':'member_native',
             'training_eligible':False,'structure_coverage':1.,'wyckoff_coverage':1.}
        row['row_hash']=stable_hash(row);aggregate.append(row)
    return {'identity':identity,'complete':True,'summaries':rows,'input_refs':[{'entity_id':symbol,'cache_key':'a'*64} for symbol in rows]},aggregate


def test_readonly_missing_does_not_create_and_exact_eight_tables(tmp_path):
    path=tmp_path/'missing.db';assert BoardFluentStore(path,readonly=True).current_generation() is None
    assert not path.exists()
    store=BoardFluentStore(path)
    with sqlite3.connect(path) as connection:
        assert {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}==TABLES
    assert store.status()['market'] is None


@pytest.mark.parametrize('changed_source', ['storage/board_fluent.py', 'services/board_fluent.py'])
def test_generation_contract_binds_both_same_named_source_files(monkeypatch, changed_source):
    before=service.generation_contract()
    assert set(before['code_hashes'])=={'storage/board_fluent.py','services/board_fluent.py'}
    original=service._file_hash
    # inspect.getfile uses native separators; the contract uses POSIX keys on every OS.
    monkeypatch.setattr(service,'_file_hash',lambda path:'0'*64 if Path(path).as_posix().endswith('/'+changed_source) else original(path))
    after=service.generation_contract()
    for source, checksum in before['code_hashes'].items():
        if source == changed_source:
            assert after['code_hashes'][source] != checksum
        else:
            assert after['code_hashes'][source] == checksum
    assert digest(after)!=digest(before)


def test_rank_formula_ties_singleton_negative_return_and_separate_types():
    rows=[{'board_id':name,'kind':'concept','board_relative_amount20':volume,'board_return_1':change,'board_amount':amount}
          for name,volume,change,amount in [('A',1,.01,10),('B',2,-.03,20),('C',3,.02,20),('D',None,None,None)]]
    rows.append({'board_id':'I','kind':'industry','board_relative_amount20':1,'board_return_1':.01,'board_amount':1})
    result=rank_boards(rows)
    activity=[row for row in result if row['kind']=='concept' and row['ranking_type']=='activity']
    assert [(row['board_id'],row['score']) for row in activity]==[('B',75.),('C',75.),('A',0.)]
    capital=[row for row in result if row['kind']=='concept' and row['ranking_type']=='capital']
    assert [row['board_id'] for row in capital]==['B','C','A']
    assert next(row for row in result if row['kind']=='industry' and row['ranking_type']=='activity')['score']==50.
    assert all(row['capital_basis']=='turnover_amount_proxy' for row in result)


def test_original_bundle_equivalence_and_indexed_memberships(market,tmp_path,monkeypatch):
    payload=bundle(market);store=BoardFluentStore(tmp_path/'fluent.db')
    generation=store.save_market(payload)
    assert store.load_market(payload['cache_key'])==payload
    reader=BoardFluentStore(store.path,readonly=True)
    monkeypatch.setattr(reader,'load_market',lambda *a,**k:pytest.fail('row GET loaded whole bundle'))
    row=reader.stock(market['symbols'][0]);assert len(row['board_ids'])==3 and len(row['all_industry_ids'])==1
    assert reader.board(payload['boards'][0]['board_id'])['generation_id']==generation
    assert len(reader.rankings('concept','activity'))==2
    assert reader.get_generation(generation)['stock_count']==7


def test_immutable_rows_and_payload_and_numeric_sha_detection(market,tmp_path):
    payload=bundle(market);store=BoardFluentStore(tmp_path/'fluent.db');generation=store.save_market(payload)
    with sqlite3.connect(store.path) as conn:
        with pytest.raises(sqlite3.IntegrityError):conn.execute('UPDATE board_daily_features SET board_amount=0')
        conn.execute('DROP TRIGGER immutable_board_daily_features_update')
        conn.execute('UPDATE board_daily_features SET activity_score=999')
    with pytest.raises(ValueError,match='derived metadata SHA'):store.board(payload['boards'][0]['board_id'])
    with pytest.raises(ValueError):store.verify_generation(generation)


def test_full_publication_atomic_and_mismatched_stage_keeps_previous(market,tmp_path):
    payload=bundle(market);store=BoardFluentStore(tmp_path/'fluent.db');generation=store.save_market(payload)
    result,aggregates=native_result(market,payload['boards'])
    bad=copy.deepcopy(result);bad['identity']['price_date']='2020-01-01'
    with pytest.raises(ValueError,match='different price date'):store.attach_native(generation,bad,aggregates)
    assert store.current_generation('full') is None
    with pytest.raises(ValueError,match='omitted'):store.attach_native(generation,result,aggregates[:-1])
    assert store.get_generation(generation)['full_ready'] is False
    store.attach_native(generation,result,aggregates,publish=False);store.publish_full(generation)
    assert store.current_generation('market')['generation_id']==store.current_generation('full')['generation_id']==generation
    assert len(store.stock(market['symbols'][0])['native_summary'])>0
    assert len(store.board(payload['boards'][0]['board_id'])['native_structure'])==1
    assert store.verify_generation(generation,level='full')['full_published']
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM feature_input_refs WHERE ref_type='native_stock'").fetchone()[0]==7


def test_full_hash_drift_rejected_and_market_only_ready_cannot_publish_full(market,tmp_path):
    payload=bundle(market);store=BoardFluentStore(tmp_path/'fluent.db');generation=store.save_market(payload)
    with pytest.raises(ValueError,match='not ready'):store.publish_full(generation)
    result,aggregates=native_result(market,payload['boards']);result['summaries'][market['symbols'][0]]['status']='changed'
    with pytest.raises(ValueError,match='row hash'):store.attach_native(generation,result,aggregates)
    assert store.current_generation('market')['generation_id']==generation


@pytest.mark.parametrize('table,selector',[
    ('native_stock_summaries','symbol'),('board_structure_features','board_id'),
    ('feature_input_refs','entity_id')])
def test_full_population_deletions_rejected_before_hot_reuse_and_publication(market,tmp_path,table,selector):
    payload=bundle(market);store=BoardFluentStore(tmp_path/'fluent.db');generation=store.save_market(payload)
    result,aggregates=native_result(market,payload['boards'])
    result['identity'].update(format='native_summary_collection_v1',required_symbols=sorted(result['summaries']),
                              input_refs_hash=stable_hash(sorted(result['input_refs'],key=lambda value:value['entity_id'])))
    store.attach_native(generation,result,aggregates,publish=False)
    assert store.verify_generation(generation,level='full')['full_ready']
    with sqlite3.connect(store.path) as conn:
        conn.execute('DROP TRIGGER immutable_'+table+'_delete')
        entity=payload['boards'][0]['board_id'] if selector=='board_id' else market['symbols'][0]
        conn.execute('DELETE FROM '+table+' WHERE generation_id=? AND '+selector+'=?',(generation,entity))
    with pytest.raises(ValueError):store.verify_generation(generation,level='full')
    with pytest.raises(ValueError):store.publish_full(generation)
    assert store.current_generation('full') is None


@pytest.mark.parametrize('missing',['summary','reference'])
def test_full_attachment_requires_complete_population_and_references(market,tmp_path,missing):
    payload=bundle(market);store=BoardFluentStore(tmp_path/'fluent.db');generation=store.save_market(payload)
    result,aggregates=native_result(market,payload['boards'])
    result['identity'].update(format='native_summary_collection_v1',required_symbols=sorted(result['summaries']),
                              input_refs_hash=stable_hash(sorted(result['input_refs'],key=lambda value:value['entity_id'])))
    if missing=='summary':result['summaries'].pop(market['symbols'][0])
    else:result['input_refs'].pop(0)
    with pytest.raises(ValueError):store.attach_native(generation,result,aggregates)
    assert store.get_generation(generation)['full_ready'] is False
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM native_stock_summaries').fetchone()[0]==0


def test_actual_legacy_migration_and_warm_same_bytes_skip_numerical(actual_market,monkeypatch):
    original=bundle(actual_market)
    raw_before=hashlib.sha256(actual_market['path'].read_bytes()).hexdigest()
    kb_before=hashlib.sha256(actual_market['kb'].read_bytes()).hexdigest()
    def forbidden(*a,**k):pytest.fail('cache recalculated compound features')
    monkeypatch.setattr(math_features,'_compound',forbidden)
    first=build(actual_market);assert first['migration'] and first['cache_source']=='legacy'
    store=BoardFluentStore(actual_market['fluent'],readonly=True)
    assert store.load_market(original['cache_key'],service.generation_contract())==original
    second=build(actual_market);assert second['cache_source']=='fluent' and second['cache_hit']
    assert first['generation_id']==second['generation_id']
    assert hashlib.sha256(actual_market['path'].read_bytes()).hexdigest()==raw_before
    assert hashlib.sha256(actual_market['kb'].read_bytes()).hexdigest()==kb_before


def test_raw_revision_stales_and_rebuilds_without_old_date_fallback(actual_market):
    first=build(actual_market,migrate=False)
    assert not service.freshness_status(fluent_db_path=actual_market['fluent'],raw_db_path=actual_market['path'],kb_db_path=actual_market['kb'])['stale']
    with sqlite3.connect(actual_market['path']) as conn:
        conn.execute('UPDATE stock_daily_normalized SET open=open*1.1,close=close*1.1,high=high*1.1,low=low*1.1 WHERE stock=? AND date=?',
                     (actual_market['symbols'][0],actual_market['dates'][-3]))
    assert service.freshness_status(fluent_db_path=actual_market['fluent'],raw_db_path=actual_market['path'],kb_db_path=actual_market['kb'])['stale']
    second=build(actual_market,migrate=False)
    assert first['generation_id']!=second['generation_id'] and first['raw_fingerprint']!=second['raw_fingerprint']
    with pytest.raises(ValueError,match='latest qualified'):build(actual_market,price_date=actual_market['dates'][-2])


def test_freshness_and_row_reads_never_call_math_or_catalog(actual_market,monkeypatch):
    result=build(actual_market)
    monkeypatch.setattr(math_features,'_read_prices',lambda *a,**k:pytest.fail('GET read raw matrix'))
    monkeypatch.setattr(service.BoardKnowledgeStore,'catalog',lambda *a,**k:pytest.fail('GET parsed catalog'))
    status=service.freshness_status(fluent_db_path=actual_market['fluent'],raw_db_path=actual_market['path'],kb_db_path=actual_market['kb'])
    assert status['available'] and not status['stale']
    store=BoardFluentStore(actual_market['fluent'],readonly=True)
    assert store.board('THS_I_111111')['price_date']==result['price_date']


def test_no_matching_native_publishes_market_partial_not_full(actual_market,monkeypatch):
    import my_strategy.services.board_native_summary as native
    monkeypatch.setattr(native,'collect_native_summaries',lambda **kwargs:{'identity':{'price_date':kwargs['price_date']},'complete':False,
        'summaries':{},'diagnostics':{'unavailable':7,'reason':'no_current_native'}})
    result=build(actual_market,with_native=True)
    assert result['status']=='partial' and result['market_published'] and not result['full_published']
    assert BoardFluentStore(actual_market['fluent']).get_sync(result['sync_id'])['status']=='partial'


def test_indexed_full_warm_revalidates_inputs_without_cold_native_decode(actual_market,monkeypatch):
    import my_strategy.services.board_native_summary as native
    market_bundle=bundle(actual_market)
    result,_=native_result(actual_market,market_bundle['boards'])
    monkeypatch.setattr(native,'collect_native_summaries',lambda **kwargs:result)
    first=build(actual_market,with_native=True)
    assert first['full_published']
    def forbidden(**kwargs):pytest.fail('warm full decoded native blobs')
    monkeypatch.setattr(native,'collect_native_summaries',forbidden)
    checked=[]
    def revalidate(identity,**kwargs):
        checked.append(identity);return {'valid':True,'diagnostics':{'input_proof_verified':True},'native_blob_decode':False}
    monkeypatch.setattr(native,'validate_native_reuse',revalidate)
    second=build(actual_market,with_native=True)
    assert checked and second['generation_id']==first['generation_id'] and second['full_published']
    assert second['native_diagnostics']['indexed_full_reused']


@pytest.mark.parametrize('change',['snapshot','contract','historical'])
def test_independent_native_index_reuse_across_new_market_generation(actual_market,market,monkeypatch,change):
    import my_strategy.services.board_native_summary as native
    from my_strategy.storage.board_knowledge import FORMAT
    payload=bundle(actual_market);native_data,_=native_result(actual_market,payload['boards'])
    first=build(actual_market,with_native=True,native_result=native_data)
    store=BoardFluentStore(actual_market['fluent'])
    component=store.get_native_component(first['generation_id'])
    assert component['identity']==native_data['identity'] and component['summaries']==native_data['summaries']
    assert component['input_refs']==native_data['input_refs']
    if change=='snapshot':
        row={key:value for key,value in market['store'].snapshots[0].items() if key!='snapshot_hash'}
        now=pd.Timestamp.now(tz='Asia/Shanghai').isoformat()
        row.update(format=FORMAT,observed_at=now,available_at=now,effective_at=now,
            source_url='https://q.10jqka.com.cn/thshy/',
            provenance={'pagination_complete':True,'reported_count':len(row['members']),'unique_members':len(row['members'])})
        actual_market['store'].import_snapshot(row)
    else:
        contract=service.generation_contract()
        monkeypatch.setattr(service,'generation_contract',lambda:{**contract,'fluent_contract_revision':'new'})
        if change=='historical':
            build(actual_market,with_native=False)
            assert store.current_generation('full') is None
    checks=[]
    def revalidate(identity,**kwargs):
        checks.append(identity);return {'valid':True,'native_blob_decode':False,'checked_native_inputs':7}
    monkeypatch.setattr(native,'validate_native_reuse',revalidate)
    monkeypatch.setattr(native,'collect_native_summaries',lambda **kwargs:pytest.fail('compatible indexed native evidence decoded full blobs'))
    second=build(actual_market,with_native=True)
    assert checks and second['full_published'] and second['generation_id']!=first['generation_id']
    assert second['native_diagnostics']['indexed_native_reused']
    assert second['native_diagnostics']['indexed_native_generation_id']==first['generation_id']
    assert store.get_generation(first['generation_id'])['full_ready']
    assert store.get_native_component(second['generation_id'])['summaries']==component['summaries']


def test_independent_native_reuse_refuses_changed_requested_population(actual_market,monkeypatch):
    import my_strategy.services.board_native_summary as native
    payload=bundle(actual_market);native_data,_=native_result(actual_market,payload['boards'])
    native_data['identity']['required_symbols']=native_data['identity']['required_symbols'][:-1]
    build(actual_market,with_native=True,native_result=native_data)
    contract=service.generation_contract()
    monkeypatch.setattr(service,'generation_contract',lambda:{**contract,'fluent_contract_revision':'new'})
    monkeypatch.setattr(native,'validate_native_reuse',lambda *a,**k:pytest.fail('changed requested population reused'))
    collected=[]
    def collect(**kwargs):
        collected.append(kwargs);return {'identity':{},'complete':False,'summaries':{},'diagnostics':{'reason':'fixture_new_population'}}
    monkeypatch.setattr(native,'collect_native_summaries',collect)
    assert build(actual_market,with_native=True)['status']=='partial'
    assert collected


def test_indexed_complete_generation_can_restore_publication_without_rewriting_diagnostics(actual_market,monkeypatch):
    import my_strategy.services.board_native_summary as native
    payload=bundle(actual_market);native_data,_=native_result(actual_market,payload['boards'])
    first=build(actual_market,with_native=True,native_result=native_data)
    store=BoardFluentStore(actual_market['fluent']);full=store.get_generation(first['generation_id'])
    market_inputs={key:value for key,value in full['publication_inputs'].items() if key not in {'native_identity_hash','native_cache_witness'}}
    store.save_market(payload,contract=service.generation_contract(),publication_inputs=market_inputs,publish=True)
    assert store.current_generation('full') is None
    monkeypatch.setattr(native,'validate_native_reuse',lambda *a,**k:{'valid':True,'native_blob_decode':False})
    monkeypatch.setattr(native,'collect_native_summaries',lambda **kwargs:pytest.fail('restoring indexed publication decoded blobs'))
    monkeypatch.setattr(BoardFluentStore,'get_native_component',lambda *a,**k:pytest.fail('restoring same generation rewrote immutable diagnostics'))
    second=build(actual_market,with_native=True)
    assert second['generation_id']==first['generation_id'] and second['full_published']


def test_current_calendar_prefix_gate_rejects_mismatch_and_accepts_future_extension(actual_market,monkeypatch):
    import my_strategy.services.board_native_summary as native
    payload=bundle(actual_market);result,_=native_result(actual_market,payload['boards'])
    bad=copy.deepcopy(result);bad['identity']['calendar_hash']='0'*64
    with pytest.raises(ValueError,match='calendar prefix'):build(actual_market,with_native=True,native_result=bad)
    assert BoardFluentStore(actual_market['fluent']).current_generation('full') is None
    # Dates after the price day are not inputs to current native evidence.
    actual_market={**actual_market,'dates':actual_market['dates']+['2099-01-01']}
    first=build(actual_market,with_native=True,native_result=result)
    assert first['full_published']
    store=BoardFluentStore(actual_market['fluent']);identity=store.get_generation(first['generation_id'])['native_identity']
    identity['calendar_hash']='f'*64
    with sqlite3.connect(store.path) as conn:
        conn.execute('UPDATE feature_generations SET native_identity_json=?,native_identity_sha=? WHERE generation_id=?',
                     (canonical(identity),digest(identity),first['generation_id']))
    monkeypatch.setattr(native,'validate_native_reuse',lambda *a,**k:pytest.fail('mismatched calendar accepted hot reuse'))
    monkeypatch.setattr(native,'collect_native_summaries',lambda **kwargs:bad)
    with pytest.raises(ValueError,match='calendar prefix'):build(actual_market,with_native=True)


def test_sync_checkpoint_merge_and_strict_resume_input(tmp_path):
    store=BoardFluentStore(tmp_path/'fluent.db');claim=store.acquire_sync('fp',{'raw':'one'})
    assert claim['owned']
    store.heartbeat_sync(claim['sync_id'],stage='market',checkpoint={'market':True})
    store.heartbeat_sync(claim['sync_id'],stage='native',checkpoint={'native_run':'source'})
    assert store.get_sync(claim['sync_id'])['checkpoint']=={'market':True,'native_run':'source'}
    assert not store.acquire_sync('other',{'raw':'two'})['owned']
    store.finish_sync(claim['sync_id'],status='partial',result={'market_published':True})
    with pytest.raises(ValueError,match='input mismatch'):store.acquire_sync('fp',{'raw':'changed'},sync_id=claim['sync_id'])
    resumed=store.acquire_sync('fp',{'raw':'one'},sync_id=claim['sync_id']);assert resumed['owned'] and resumed['checkpoint']['market']
    store.finish_sync(claim['sync_id'],result={'generation_id':'g'})
    assert store.acquire_sync('fp',{'raw':'one'})['reused']


def test_real_process_lease_blocks_and_dead_owner_resumes_checkpoint(tmp_path):
    path=tmp_path/'fluent.db';context=multiprocessing.get_context('spawn');ready=context.Queue();stop=context.Event()
    process=context.Process(target=_lease_child,args=(path,ready,stop));process.start()
    try:
        sync_id=ready.get(timeout=20);store=BoardFluentStore(path)
        assert not store.acquire_sync('same',{'raw':'verified'})['owned']
        assert not store.acquire_sync('other',{'raw':'different'})['owned']
    finally:process.terminate();process.join(timeout=10)
    resumed=store.acquire_sync('same',{'raw':'verified'},sync_id=sync_id)
    assert resumed['owned'] and resumed['checkpoint']['market_done']
    store.finish_sync(sync_id)


def test_source_changes_during_build_do_not_publish(actual_market):
    first=build(actual_market)
    def mutate(progress):
        if progress['stage']=='board_market' and progress['current']==0:
            with sqlite3.connect(actual_market['path']) as conn:conn.execute('UPDATE stock_daily_normalized SET amount=amount+1')
    with pytest.raises(ValueError,match='source changed'):build(actual_market,progress=mutate)
    assert BoardFluentStore(actual_market['fluent']).current_generation()['generation_id']==first['generation_id']


def test_cancel_gate_keeps_published_generation_and_closes_lease(actual_market):
    first=build(actual_market);cancelled=False
    class Cancelled(RuntimeError):pass
    def progress(value):
        nonlocal cancelled
        if value['stage']=='board_market':cancelled=True
    def check():
        if cancelled:raise Cancelled('cancel requested')
    with pytest.raises(Cancelled):build(actual_market,progress=progress,check_cancel=check)
    store=BoardFluentStore(actual_market['fluent'])
    assert store.current_generation()['generation_id']==first['generation_id']
    assert store.latest_sync()['status']=='cancelled'
