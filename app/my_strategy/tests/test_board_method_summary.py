"""Synthetic saved scans: no inference, production files or account operations."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from my_strategy.services import board_method_summary as service
from my_strategy.storage import board_knowledge as kb
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.tests.test_board_knowledge import snapshot
from my_strategy.tests.test_method_display import sample, VERSION
from my_strategy.web_dashboard.board_api import router

DAY = '2026-10-15'
METHOD = 'structure_kronos_ma_mlp'


def row(symbol, *, direction='BUY', state='complete'):
    value = sample(direction=direction)
    value.update(symbol=symbol, date=DAY, data_end=DAY, reference_price_date=DAY)
    value['method_context']['end'] = DAY
    evaluation = value['method_evaluation']; evaluation['as_of'] = DAY
    evaluation['kronos']['as_of'] = DAY
    evaluation['kronos']['future_dates'] = ['2026-10-16','2026-10-19','2026-10-20','2026-10-21','2026-10-22']
    if state in {'failed','deferred'}:
        from my_strategy.services.analysis_methods import _placeholder
        return _placeholder(symbol, DAY, METHOD, state, 'synthetic_'+state)
    if state == 'partial':
        evaluation.update(status='partial', missing_reasons=['synthetic_missing_score'])
        evaluation['score'] = {'status':'unavailable','reason':'synthetic_missing_score','feature_profile':'ma_trend_v1'}
    return value


def report(rows=None, run_id='synthetic-method-scan'):
    rows = rows or [row('000001.SZ'), row('000002.SZ',direction='SELL'), row('000003.SZ',direction='WAIT'),
        row('000004.SZ',state='partial'),row('000005.SZ',state='failed'),row('000006.SZ',state='deferred')]
    context = row('000001.SZ')['method_context']
    counts = {status: sum(r['method_evaluation']['status']==status for r in rows) for status in ['complete','partial','failed','deferred']}
    population = [r['symbol'] for r in rows]
    return {'run_id':run_id,'status':'partial','strategy_version':'account_free_methods_v1',
        'analysis_method':METHOD,'analysis_mode':'reference','method_context':context,
        'as_of':DAY,'data_end':DAY,'population':population,'rows':rows,
        'frozen_identity':{'population':population,'context':context,'model_version':VERSION,
            'raw':{'sha256':'synthetic-original-retained-even-if-current-raw-changes'},'calendar':{'id':'synthetic'},'code':{'id':'synthetic'}},
        'coverage':{'requested':len(rows),'failed':counts['failed'],'deferred':counts['deferred'],
            'unavailable':counts['partial'],'evaluated':len(rows)-counts['deferred'],'complete':counts['deferred']==0},
        'updated_at':'2026-10-16T09:00:00+08:00'}


@pytest.fixture
def saved(tmp_path,monkeypatch):
    monkeypatch.setattr(kb,'_now',lambda:datetime(2026,10,14,6,tzinfo=timezone.utc))
    result = ResultStore(tmp_path/'results.db',tmp_path/'runs')
    original=report(); result.save('scan',original)
    store=kb.BoardKnowledgeStore(tmp_path/'kb.db')
    symbols=['000001.SZ','000002.SZ','000003.SZ','000004.SZ','000005.SZ','000006.SZ','000007.SZ','830001.BJ']
    store.import_snapshot(snapshot(members=symbols))
    store.import_snapshot(snapshot(board='THS_C_885001',members=['000001.SZ','000007.SZ']))
    reader=service.BoardMethodSummary(result.db_path,result.runs_root,store.path)
    return result,store,reader,original


def read(reader,original,**kwargs):
    return reader.summary(original['run_id'],method=METHOD,as_of=DAY,**kwargs)


def test_complete_population_and_separate_denominators_no_placeholder_wait(saved):
    _,_,reader,original=saved
    output=read(reader,original); stats=output['population_summary']
    assert stats['counts']==dict(expected=6,matched=6,complete=3,buy=1,sell=1,wait=1,unavailable=1,failed=1,deferred=1,not_in_scan=0)
    assert stats['forecast']['count']==4 and stats['score']['count']==3
    assert stats['direction_ratios']==dict(denominator=3,buy=1/3,sell=1/3,wait=1/3)
    assert stats['forecast']['win_rate'] is None and stats['forecast']['joint_return_interval'] is None
    assert 'terminal_return_p10' not in stats['forecast']
    board=next(b for b in output['boards'] if b['kind']=='industry')
    assert board['counts']['expected']==7 and board['counts']['not_in_scan']==1
    assert board['membership']['excluded_count']==1 and board['coverage']['method_complete_ratio']==3/7
    assert board['ranking_eligible'] is False
    assert sum(b['counts']['matched'] for b in output['boards'])>stats['counts']['matched']
    assert output['membership_semantics']['historical_membership_verified'] is False


@pytest.mark.parametrize('category,count',[('all',6),('buy',1),('sell',1),('wait',1),('unavailable',1),('failed',1),('deferred',1)])
def test_paging_filters_complete_report_without_changing_summary(saved,category,count):
    _,_,reader,original=saved
    value=reader.stocks(original['run_id'],method=METHOD,as_of=DAY,page_size=1,category=category)
    assert value['scan_view']['total']==count and value['scan_view']['total_rows']==6
    assert value['population_summary']['counts']['expected']==6
    assert len(value['rows'])==1
    if category!='all':
        view=value['rows'][0]['method_view']
        assert (view['direction'].lower() if view['valid'] else view['status'])==category
    favorites=reader.stocks(original['run_id'],method=METHOD,as_of=DAY,symbols={'000001.SZ','000002.SZ'},query='000002')
    assert [r['symbol'] for r in favorites['rows']]==['000002.SZ']
    assert favorites['population_summary']==value['population_summary']


def test_member_missing_keeps_board_denominator_and_context(saved):
    _,_,reader,original=saved
    value=read(reader,original,board_id='THS_I_881155',category='not_in_scan')
    assert len(value['rows'])==1 and value['rows'][0]['symbol']=='000007.SZ'
    assert value['rows'][0]['method_view']['direction'] is None
    assert value['board']['counts']['expected']==7
    assert value['identity']['memberships']['THS_I_881155']['projection_hash']


def test_top_partial_with_complete_coverage_is_readable_and_five_eighty_percent_gate(saved):
    result,_,reader,_=saved
    rows=[row(f'00000{i}.SZ') for i in range(1,7)]
    value=report(rows,'finished-synthetic');result.save('scan',value)
    output=read(reader,value); board=next(b for b in output['boards'] if b['kind']=='industry')
    assert output['coverage']['complete'] and board['ranking_eligible'] and board['counts']['not_in_scan']==1
    assert board['coverage']['method_complete_ratio']==6/7


@pytest.mark.parametrize('mutate',[
    lambda r:r.update(analysis_method='structure_kronos'),
    lambda r:r['method_context'].update(seed=99),
    lambda r:r['frozen_identity'].update(population=['000099.SZ']),
    lambda r:r['rows'][0]['method_evaluation'].update(method='structure_kronos'),
    lambda r:r['rows'][0]['method_evaluation'].update(as_of='2026-10-14'),
    lambda r:r['rows'][0]['method_evaluation'].update(model_id='d'*64),
    lambda r:r['rows'][0]['method_evaluation']['score'].update(model_version='d'*64),
    lambda r:r['rows'][0]['method_evaluation']['score'].update(scorer_id='d'*64),
    lambda r:r['coverage'].update(evaluated=6),
])
def test_mixed_frozen_identity_is_rejected(saved,mutate):
    result,_,reader,original=saved
    value=deepcopy(original); value['frozen_identity']['context']=deepcopy(value['method_context'])
    mutate(value);result.save('scan',value)
    with pytest.raises(ValueError):read(reader,value)


def test_byte_tampering_not_hidden_by_cache(saved):
    result,_,reader,original=saved;read(reader,original)
    path=result.runs_root/original['run_id']/'reports/czsc.json'
    path.write_text(path.read_text().replace('synthetic_missing_score','tampered_missing_score'))
    with pytest.raises(ValueError,match='SHA'):read(reader,original)


def test_expired_prediction_object_not_required_and_cache_is_bounded(saved):
    result,_,_,original=saved
    reader=service.BoardMethodSummary(result.db_path,result.runs_root,cache_entries=1)
    first=reader.stocks(original['run_id'],method=METHOD,as_of=DAY)
    value=deepcopy(original);value['run_id']='second-synthetic';result.save('scan',value)
    reader.stocks(value['run_id'],method=METHOD,as_of=DAY)
    assert len(reader._cache)==1 and reader._bytes<=reader.cache_bytes
    first['rows'][0]['method_view']['direction']='FAKE'
    assert reader.stocks(original['run_id'],method=METHOD,as_of=DAY)['rows'][0]['method_view']['direction']=='BUY'


def test_missing_kb_and_results_do_not_initialize_any_storage(tmp_path):
    reader=service.BoardMethodSummary(tmp_path/'missing/results.db',tmp_path/'missing/runs',tmp_path/'missing/kb.db')
    assert reader.list_runs(method=METHOD,as_of=DAY)['selection']['status']=='pending'
    with pytest.raises(KeyError):reader.stocks('missing',method=METHOD,as_of=DAY)
    assert not (tmp_path/'missing').exists()


def test_read_routes_are_pure_and_validate_binding(saved,monkeypatch):
    result,store,_,original=saved
    app=FastAPI();app.state.tasks=SimpleNamespace(results=result);app.state.board_db_path=store.path;app.include_router(router)
    def forbidden(*args,**kwargs):raise AssertionError('read must never migrate or write')
    monkeypatch.setattr(ResultStore,'get_scan_page',forbidden)
    monkeypatch.setattr(ResultStore,'save',forbidden)
    # SQLite read-only WAL readers may create empty coordination sidecars;
    # nonempty WAL, database and immutable payload bytes must remain exact.
    def snapshot_files():
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in result.db_path.parent.rglob('*') if p.is_file() and not str(p).endswith('-shm') and not (str(p).endswith('-wal') and p.stat().st_size == 0)}
    # Finish fixture writer finalizers before checking that GET does not write.
    import gc
    gc.collect()
    before=snapshot_files()
    with TestClient(app) as client:
        base='/api/boards/method-scans';query=f'?method={METHOD}&as_of={DAY}'
        assert client.get(base+query).json()['runs'][0]['run_id']==original['run_id']
        endpoint=base+'/'+original['run_id']
        assert client.get(endpoint+query).status_code==200
        assert client.get(endpoint+'/stocks'+query+'&category=wait').json()['scan_view']['total']==1
        assert client.get(endpoint+'/members'+query+'&board_id=THS_I_881155').json()['board']['counts']['expected']==7
        assert client.get(endpoint+'/stocks'+query+'&symbols=830001.BJ').status_code==422
        assert client.get(endpoint+query.replace(METHOD,'structure_kronos')).status_code==409
        assert client.get(endpoint+query.replace(DAY,'2026-10-14')).status_code==409
        assert client.get(endpoint+'/stocks'+query+'&category=watch').status_code==422
        assert client.get(base+'/missing'+query).status_code==404
    assert snapshot_files()==before


def test_future_membership_is_not_backfilled(saved,monkeypatch):
    _,store,reader,original=saved
    monkeypatch.setattr(kb,'_now',lambda:datetime(2026,10,16,6,tzinfo=timezone.utc))
    later=snapshot(board='THS_C_885002',members=['000001.SZ'])
    later.update(observed_at='2026-10-16T13:59:00+08:00',available_at='2026-10-16T13:59:00+08:00',effective_at='2026-10-16T13:59:00+08:00')
    store.import_snapshot(later)
    assert 'THS_C_885002' not in [b['board_id'] for b in read(reader,original)['boards']]


def test_empty_and_incomplete_membership_never_qualifies(saved,monkeypatch):
    _,_,reader,original=saved
    values=[dict(board_id='empty',snapshot_hash='e'*64,members=[],expected_count=0,complete=True,
                 name='synthetic empty',kind='industry',observed_at=DAY+'T09:00:00+08:00',
                 available_at=DAY+'T09:00:00+08:00',effective_at=DAY+'T09:00:00+08:00'),
            dict(board_id='incomplete',snapshot_hash='f'*64,members=['000001.SZ'],expected_count=2,complete=False,
                 name='synthetic incomplete',kind='concept',observed_at=DAY+'T09:00:00+08:00',
                 available_at=DAY+'T09:00:00+08:00',effective_at=DAY+'T09:00:00+08:00')]
    monkeypatch.setattr(kb.BoardKnowledgeStore,'catalog',lambda self,**options:deepcopy(values))
    output=read(reader,original)
    empty=output['boards'][0]
    assert empty['counts']['expected']==0 and empty['coverage']['method_complete_ratio'] is None
    assert empty['forecast']['count']==0 and empty['forecast']['mean_member_return_median'] is None
    assert all(not b['ranking_eligible'] for b in output['boards'])
    assert all(not b['membership']['usable_membership'] for b in output['boards'])


def test_710_overlapping_boards_do_not_multiply_market_denominator(saved,monkeypatch):
    _,_,reader,original=saved
    values=[dict(board_id=f'synthetic-{i}',snapshot_hash=hashlib.sha256(str(i).encode()).hexdigest(),
        members=['000001.SZ','000007.SZ'],expected_count=2,complete=True,name='synthetic',kind='concept',
        observed_at=DAY+'T09:00:00+08:00',available_at=DAY+'T09:00:00+08:00',effective_at=DAY+'T09:00:00+08:00') for i in range(710)]
    monkeypatch.setattr(kb.BoardKnowledgeStore,'catalog',lambda self,**options:deepcopy(values))
    output=read(reader,original)
    assert len(output['boards'])==710 and output['population_summary']['counts']['expected']==6
    assert sum(b['counts']['matched'] for b in output['boards'])==710
    assert all(b['counts']['expected']==2 and b['counts']['not_in_scan']==1 for b in output['boards'])


@pytest.mark.parametrize('change',[
    lambda r:r.update(frozen_identity=[]),
    lambda r:r.update(coverage=['invalid']),
    lambda r:r['coverage'].update(failed=True),
    lambda r:r['rows'][0].update(method_evaluation=[]),
    lambda r:r['rows'][0]['method_evaluation'].update(score=['bad']),
    lambda r:r['rows'][0]['method_evaluation'].update(kronos=['bad']),
    lambda r:r['rows'][0]['method_evaluation']['score'].update(qualification=['bad']),
])
def test_malformed_persisted_records_fail_closed(saved,change):
    result,_,reader,original=saved
    value=deepcopy(original);change(value);result.save('scan',value)
    with pytest.raises(ValueError):read(reader,value)


@pytest.mark.parametrize('relative',['../outside.json','/outside.json'])
def test_index_path_escape_refused(saved,relative):
    result,_,reader,original=saved
    with sqlite3.connect(result.db_path) as con:
        con.execute('UPDATE czsc_runs SET report_path=? WHERE run_id=?',(relative,original['run_id']))
    with pytest.raises(ValueError,match='path'):read(reader,original)


def test_indexed_report_symlink_refused(saved,tmp_path):
    result,_,reader,original=saved
    path=result.runs_root/original['run_id']/'reports/czsc.json';outside=tmp_path/'outside.json'
    outside.write_bytes(path.read_bytes());path.unlink()
    try:path.symlink_to(outside)
    except OSError:pytest.skip('symlink creation unavailable on this host')
    with pytest.raises(ValueError,match='symlink'):read(reader,original)


def test_corrupt_original_board_proof_is_not_read_as_an_empty_generation(saved):
    _,store,reader,original=saved
    with sqlite3.connect(store.path) as con:
        con.execute('DROP TRIGGER immutable_source_payloads_update')
        con.execute("UPDATE source_payloads SET blob_sha256=?",('0'*64,))
    with pytest.raises(ValueError,match='physical SHA'):read(reader,original)


def test_zero_cache_limit_is_supported_without_disk_cache(saved):
    result,store,_,original=saved
    reader=service.BoardMethodSummary(result.db_path,result.runs_root,store.path,cache_bytes=0)
    assert read(reader,original)['population_summary']['counts']['complete']==3
    assert reader._cache=={} and reader._bytes==0


def test_explicit_forecast_horizon_mix_is_rejected_but_absent_dates_are_legacy_compatible(saved):
    result,_,reader,original=saved
    value=deepcopy(original)
    value['rows'][0]['method_evaluation']['kronos']['future_dates']=['2026-10-16']
    result.save('scan',value)
    with pytest.raises(ValueError,match='horizon'):read(reader,value)
    for member in value['rows']:
        if member['method_evaluation'].get('kronos'):
            member['method_evaluation']['kronos'].pop('future_dates',None)
    result.save('scan',value)
    assert read(reader,value)['population_summary']['counts']['complete']==3
