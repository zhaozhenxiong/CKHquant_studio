from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import copy
import json
from pathlib import Path
import os
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_trend_source as source
from my_strategy.services import czsc_trend_support as support


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    dates=pd.bdate_range('2025-02-03',periods=25).strftime('%Y-%m-%d').tolist()
    db=tmp_path/'raw.db'
    symbols=['000001.SZ','600001.SH']
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,'
            'close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER,PRIMARY KEY(stock,date))')
        for symbol in symbols:
            conn.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                [(symbol,day,10.,11.,9.,10.,4_000_000.,40_000_000.,'tushare',1) for day in dates])
    monkeypatch.setattr(support,'_calendar',lambda run_id=None:(dates,'checkpoint-fixture-calendar'))
    population=source.build_liquidity_universe(feature_start=dates[0],end=dates[-1],db_path=db)
    _,raw=source.freeze_raw_source(symbols=symbols,end=dates[-1],feature_start=dates[0],db_path=db,
        cpu_workers=1,preflight_population=population,prepare_batch_size=3)
    cfg=source.strategy_config({'feature_start':dates[0]})
    calls=[];fail=set();interrupt=set()

    def worker(payload):
        record,cfg,calendar,directory,timings=payload
        symbol=record['symbol'];calls.append(symbol)
        if symbol in interrupt: raise KeyboardInterrupt('fixture interruption')
        if symbol in fail: raise RuntimeError('fixture failed stock')
        frame=pd.read_parquet(record['raw_path'])
        target=Path(directory)/symbol.replace('.','_');target.mkdir(parents=True,exist_ok=False)
        from my_strategy.services.czsc_native_evidence import build_native_evidence
        from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS
        empty=build_native_evidence(pd.DataFrame(),cfg,market_dates=[])
        for column in MA_TREND_COLUMNS:frame[column]=0.
        frame['native_schema_hash']=empty['schema_hash'];frame['native_version']=empty['version']
        path=target/'features.parquet';frame.to_parquet(path,index=False)
        native=target/'native.json.gz'
        pack={key:value for key,value in empty.items() if key!='frame'}
        pack['evaluations']=[{'date':str(pd.Timestamp(day).date())} for day in frame.date]
        pack['auxiliary_evaluations']=list(pack['evaluations'])
        pack['calendar_provenance']=dict(provided=True,calendar_prefix_hash=stable_hash(calendar),
            calendar_prefix_end=calendar[-1],future_calendar_entries_used_for_historical_qualification=False)
        source._save_native_pack(native,pack)
        timings.update(total_ns=20,parquet_ns=5,json_gzip_ns=3)
        return {**record,'path':str(path),'sha256':support._file_hash(path),
            'native_path':str(native),'native_sha256':support._file_hash(native),'native_encoding':'gzip_json',
            'native_config_hash':empty['config_hash'],'native_schema_hash':empty['schema_hash'],
            'rows':len(frame),'raw_only_rebuild':True,'reused_model_or_label':False}

    original_worker=source._native_worker
    monkeypatch.setattr(source,'_native_worker',worker)
    return dict(raw=raw,cfg=cfg,calls=calls,fail=fail,interrupt=interrupt,symbols=symbols,
        cache_db_path=tmp_path/'native-cache.db',original_worker=original_worker)


def prepare(frozen, **kwargs):
    return source.prepare_native_source(frozen['raw']['run_id'],config=frozen['cfg'],cpu_workers=1,
        prepare_batch_size=3,cache_db_path=frozen['cache_db_path'],**kwargs)


def interrupted(frozen):
    frozen['interrupt'].add(frozen['symbols'][1])
    with pytest.raises(KeyboardInterrupt): prepare(frozen)
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    headers=list(ARTIFACT_RUNS_ROOT.glob('*/reports/native_checkpoint.json'))
    run=max(headers,key=lambda path:path.stat().st_mtime_ns).parent.parent
    frozen['interrupt'].clear()
    return run


def test_pool_replenishes_before_first_slow_stock_and_bounds_inflight(monkeypatch,capsys):
    release=threading.Event();executed=[]
    monkeypatch.setattr(source,'ProcessPoolExecutor',lambda max_workers,mp_context:ThreadPoolExecutor(max_workers=max_workers))

    def worker(value):
        executed.append(value)
        if value==0: assert release.wait(3), 'pool did not replenish until its first task completed'
        if value>=3: release.set()
        return value*2

    result=source._map_batches(worker,list(range(8)),cpu_workers=2,prepare_batch_size=4,stage='native')
    assert result==[value*2 for value in range(8)]
    events=[json.loads(line.split(' ',1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert len(events)==8 and events[-1]['completed']==8
    assert all(event['inflight']<=4 and event['max_inflight']==4 for event in events)
    assert events[-1]['eta_seconds']==0 and events[-1]['failed']==0


def test_interruption_resumes_only_verified_success_and_preserves_stock_order(frozen,capsys):
    run=interrupted(frozen)
    proof=source.verify_native_checkpoint(run.name)
    assert proof['completed']==1 and proof['completed_report'] is False
    assert not (run/'reports/native_source.json').exists()
    before=list(frozen['calls']);report=prepare(frozen,resume_native_run_id=run.name)
    assert frozen['calls']==before+[frozen['symbols'][1]]
    assert report['run_id']==run.name and report['status']=='completed'
    assert [record['symbol'] for record in report['dataset_records']]==frozen['symbols']
    assert report['preparation_performance']['reused']==1
    assert report['preparation_performance']['timings_ns']['total_ns']==40
    assert not any('timings_ns' in record for record in report['dataset_records'])
    assert source.verify_native_checkpoint(run.name)['completed_report'] is True
    events=[json.loads(line.split(' ',1)[1]) for line in capsys.readouterr().out.splitlines() if line.startswith('SOURCE_BATCH ')]
    assert any(event['reused']==1 and event['completed']==1 for event in events)
    assert events[-1]['completed']==2
    with pytest.raises(ValueError,match='completed native sources'):prepare(frozen,resume_native_run_id=run.name)


def test_failures_are_explicit_and_retried_without_importing_failed_rows(frozen):
    frozen['fail'].add(frozen['symbols'][0])
    with pytest.raises(ValueError,match='native source incomplete'):prepare(frozen)
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    header=max(ARTIFACT_RUNS_ROOT.glob('*/reports/native_checkpoint.json'),key=lambda path:path.stat().st_mtime_ns)
    run=header.parent.parent;report=json.loads((run/'reports/native_source.json').read_text())
    assert report['status']=='failed' and len(report['failures'])==1 and len(report['dataset_records'])==1
    proof=source.verify_native_checkpoint(run.name)
    assert proof['completed']==1 and proof['failures'][0]['symbol']==frozen['symbols'][0]
    before=list(frozen['calls']);frozen['fail'].clear();report=prepare(frozen,resume_native_run_id=run.name)
    assert frozen['calls']==before+[frozen['symbols'][0]] and report['failures']==[]
    attempts=list((run/'data_snapshots/checkpoints/attempts').glob('000001_SZ-*.json'))
    assert {json.loads(path.read_text())['status'] for path in attempts}=={'failed','completed'}


@pytest.mark.parametrize('change',['raw','parquet','native','checkpoint','native_schema','rows'])
def test_resume_refuses_corrupt_assets_or_checkpoints(frozen,change):
    run=interrupted(frozen)
    path=run/'data_snapshots/checkpoints/000001_SZ.json';checkpoint=json.loads(path.read_text())
    record=checkpoint['record']
    if change in {'raw','parquet','native'}:
        target=Path(record[{'raw':'raw_path','parquet':'path','native':'native_path'}[change]])
        target.write_bytes(target.read_bytes()+b'changed')
    elif change=='checkpoint':
        checkpoint['identity']['raw_sha256']='0'*64
        checkpoint['checkpoint_hash']=stable_hash({k:v for k,v in checkpoint.items() if k!='checkpoint_hash'})
        support._save(path,checkpoint)
    else:
        if change=='native_schema':
            native=Path(record['native_path'])
            with gzip.open(native,'rt',encoding='utf-8') as stream:pack=json.load(stream)
            pack['schema_hash']='bad-schema';source._save_native_pack(native,pack)
            record['native_sha256']=support._file_hash(native)
        else:
            parquet=Path(record['path']);pd.read_parquet(parquet).iloc[:-1].to_parquet(parquet,index=False)
            record['sha256']=support._file_hash(parquet)
        checkpoint['checkpoint_hash']=stable_hash({k:v for k,v in checkpoint.items() if k!='checkpoint_hash'})
        support._save(path,checkpoint)
    with pytest.raises(ValueError,match='drift'):source.verify_native_checkpoint(run.name)
    before=list(frozen['calls'])
    with pytest.raises(ValueError):prepare(frozen,resume_native_run_id=run.name)
    assert frozen['calls']==before


@pytest.mark.parametrize('change',['config','workers','batch','code','raw_report'])
def test_resume_requires_identical_source_configuration_and_generator(frozen,monkeypatch,change):
    run=interrupted(frozen);kwargs={}
    if change=='config':kwargs['config']={**frozen['cfg'],'feature_start':'2025-01-01'}
    elif change=='workers':kwargs['cpu_workers']=2
    elif change=='batch':kwargs['prepare_batch_size']=4
    elif change=='code':
        hashes=source._native_code_hashes();hashes[next(iter(hashes))]='changed'
        monkeypatch.setattr(source,'_native_code_hashes',lambda:hashes)
    else:
        report=source._source_run_root(frozen['raw']['run_id'])/'reports/raw_source.json'
        report.write_bytes(report.read_bytes()+b' ')
    with pytest.raises(ValueError,match='identity drift'):source.verify_native_checkpoint(run.name,**kwargs)


def test_legacy_partial_source_has_no_resumable_checkpoint(tmp_path,monkeypatch):
    from my_strategy.core import paths
    monkeypatch.setattr(paths,'ARTIFACT_RUNS_ROOT',tmp_path/'runs')
    (tmp_path/'runs/legacy/data_snapshots/native').mkdir(parents=True)
    with pytest.raises(ValueError,match='legacy partial'):source.verify_native_checkpoint('legacy')


def test_uncommitted_output_is_retained_and_rebuilt_without_counting_as_complete(frozen):
    run=interrupted(frozen)
    orphan=run/'data_snapshots/native/600001_SH';orphan.mkdir();(orphan/'features.parquet').write_text('uncommitted')
    assert source.verify_native_checkpoint(run.name)['completed']==1
    report=prepare(frozen,resume_native_run_id=run.name)
    assert report['status']=='completed'
    retained=list((run/'data_snapshots/uncommitted').glob('600001_SH-*'))
    assert len(retained)==1 and (retained[0]/'features.parquet').read_text()=='uncommitted'


def test_generator_change_before_final_report_cannot_complete_source(frozen,monkeypatch):
    code=source._native_code_hashes();changed=dict(code);changed[next(iter(code))]='changed'
    original=source._map_batches

    def map_then_change(*args,**kwargs):
        result=original(*args,**kwargs)
        monkeypatch.setattr(source,'_native_code_hashes',lambda:changed)
        return result

    monkeypatch.setattr(source,'_map_batches',map_then_change)
    with pytest.raises(ValueError,match='completed identity refused'):prepare(frozen)
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    header=max(ARTIFACT_RUNS_ROOT.glob('*/reports/native_checkpoint.json'),key=lambda path:path.stat().st_mtime_ns)
    assert not (header.parent/'native_source.json').exists()


def test_new_run_uses_only_exact_verified_database_entries(frozen,capsys):
    first=prepare(frozen);before=list(frozen['calls'])
    second=prepare(frozen)
    assert first['run_id']!=second['run_id'] and frozen['calls']==before
    assert second['preparation_performance']['cache_hits']==2
    assert second['preparation_performance']['cache_misses']==0
    assert second['preparation_performance']['reused']==0
    for old,new in zip(first['dataset_records'],second['dataset_records'],strict=True):
        assert old['path']!=new['path'] and old['sha256']==new['sha256']
        assert old['native_path']!=new['native_path'] and old['native_sha256']==new['native_sha256']
    events=[json.loads(line.split(' ',1)[1]) for line in capsys.readouterr().out.splitlines() if line.startswith('SOURCE_BATCH ')]
    assert events[-1]['completed']==2 and events[-1]['cache_hits']==2
    assert source.verify_native_checkpoint(second['run_id'])['completed']==2


def test_corrupt_database_entry_is_an_explicit_failure_never_a_cache_miss(frozen):
    prepare(frozen);before=list(frozen['calls'])
    with sqlite3.connect(frozen['cache_db_path']) as conn:
        conn.execute('UPDATE native_structure_entries SET native_blob=?',[b'corrupted'])
    with pytest.raises(ValueError,match='blob SHA drift'):prepare(frozen)
    assert frozen['calls']==before
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    header=max(ARTIFACT_RUNS_ROOT.glob('*/reports/native_checkpoint.json'),key=lambda path:path.stat().st_mtime_ns)
    status=json.loads((header.parent/'native_prepare_status.json').read_text())
    assert status['status']=='interrupted' and 'blob SHA drift' in status['error']
    assert not (header.parent/'native_source.json').exists()


@pytest.mark.skipif(sys.platform=='win32',reason='POSIX lock child; Windows uses the native msvcrt backend')
def test_native_preparation_lock_refuses_parallel_process_and_releases_on_exception(tmp_path):
    root=tmp_path/'native-run';(root/'reports').mkdir(parents=True)
    path=root/'reports/native_preparation.lock'
    script='''import fcntl,sys
with open(sys.argv[1], 'r+b') as handle:
    try: fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: sys.exit(3)
'''
    with pytest.raises(RuntimeError,match='intentional'):
        with source._native_prepare_lock(root):
            process=subprocess.run([sys.executable,'-c',script,str(path)],timeout=10)
            assert process.returncode==3
            with pytest.raises(ValueError,match='already running'):
                with source._native_prepare_lock(root):pytest.fail('second preparation entered its lock')
            raise RuntimeError('intentional')
    process=subprocess.run([sys.executable,'-c',script,str(path)],timeout=10)
    assert process.returncode==0 and path.is_file() and path.stat().st_size==1


def test_native_cache_hit_progress_is_immediate_and_checkpoint_flag_never_leaks_to_raw(frozen,monkeypatch,capsys):
    prepare(frozen);capsys.readouterr()
    from my_strategy.services.czsc_native_cache import NativeStructureCache
    original=NativeStructureCache.lookup;observations=[]

    def lookup(cache,identity):
        observations.append(capsys.readouterr().out)
        return original(cache,identity)

    monkeypatch.setattr(NativeStructureCache,'lookup',lookup)
    prepare(frozen)
    assert 'SOURCE_BATCH ' in observations[1] and '"completed": 1' in observations[1]
    assert '"checkpoint_available": true' in observations[1]
    source._map_batches(lambda value:value,[1],cpu_workers=1,prepare_batch_size=1,stage='raw')
    events=[json.loads(line.split(' ',1)[1]) for line in capsys.readouterr().out.splitlines() if line.startswith('SOURCE_BATCH ')]
    assert events[-1]['stage']=='raw' and 'checkpoint_available' not in events[-1]


@pytest.mark.parametrize('expired',['attempt','dead_parent'])
def test_expired_worker_only_leaves_attempt_outputs_and_never_publishes(frozen,monkeypatch,expired):
    run=interrupted(frozen);header=json.loads((run/'reports/native_checkpoint.json').read_text())
    raw=frozen['raw']['raw_records'][1];attempt='stale-attempt'
    current='new-attempt' if expired=='attempt' else attempt
    status=dict(status='running',run_id=run.name,attempt_id=current,owner_pid=os.getpid())
    support._save(run/'reports/native_prepare_status.json',status)
    if expired=='dead_parent':
        monkeypatch.setattr(source.multiprocessing,'parent_process',lambda:SimpleNamespace(pid=os.getpid(),is_alive=lambda:False))
    original=(run/'data_snapshots/checkpoints/000001_SZ.json').read_bytes()
    with pytest.raises(ValueError,match='expired native preparation attempt'):
        source._checkpoint_native_worker((raw,frozen['cfg'],frozen['raw']['dates'],
            str(run/'data_snapshots/native'),header,attempt))
    assert not (run/'data_snapshots/native/600001_SH').exists()
    assert not (run/'data_snapshots/checkpoints/600001_SH.json').exists()
    assert (run/'data_snapshots/native_attempts'/attempt/'native/600001_SH/native.json.gz').is_file()
    assert (run/'data_snapshots/checkpoints/000001_SZ.json').read_bytes()==original


def test_spawned_workers_publish_only_under_the_live_parent_attempt(frozen,monkeypatch):
    monkeypatch.setattr(source,'_native_worker',frozen['original_worker'])
    report=source.prepare_native_source(frozen['raw']['run_id'],config=frozen['cfg'],cpu_workers=2,
        prepare_batch_size=3,cache_db_path=frozen['cache_db_path'])
    assert report['status']=='completed' and len(report['dataset_records'])==2
    root=source._source_run_root(report['run_id'])
    status=json.loads((root/'reports/native_prepare_status.json').read_text())
    for symbol in frozen['symbols']:
        checkpoint=json.loads((root/'data_snapshots/checkpoints'/(symbol.replace('.','_')+'.json')).read_text())
        assert checkpoint['attempt_id']==status['attempt_id'] and checkpoint['status']=='completed'
        assert '/native_attempts/' not in checkpoint['record']['path']
    assert source.verify_native_checkpoint(report['run_id'])['completed_report'] is True


def test_database_cache_reuses_training_resource_changes_but_rebuilds_native_policy(frozen):
    prepare(frozen);before=list(frozen['calls'])
    cfg=copy.deepcopy(frozen['cfg']);cfg.update(epochs=cfg['epochs']+1,stock_batch_size=4)
    report=source.prepare_native_source(frozen['raw']['run_id'],config=cfg,cpu_workers=2,
        prepare_batch_size=4,cache_db_path=frozen['cache_db_path'])
    assert report['preparation_performance']['cache_hits']==2 and frozen['calls']==before
    cfg['native_evidence']={'structure_zone_strength':.6}
    report=source.prepare_native_source(frozen['raw']['run_id'],config=cfg,cpu_workers=1,
        prepare_batch_size=4,cache_db_path=frozen['cache_db_path'])
    assert report['preparation_performance']['cache_hits']==0
    assert report['preparation_performance']['cache_misses']==2
    assert frozen['calls']==before+frozen['symbols']
