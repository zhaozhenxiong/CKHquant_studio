"""Isolated crash/freshness/pagination acceptance; no production data or work."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from my_strategy.storage.task_journal import TaskJournal, JournalStop
from my_strategy.storage.status_summary import read_status
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.storage.research_store import ResearchStore
from my_strategy.services.task_maintenance import TaskMaintenance
from my_strategy.storage.scan_view import compact_scan_view


def receipt(job_id='one',fingerprint='same'):
    return {'job_id':job_id,'fingerprint':fingerprint,'kind':'prediction','spec':{},
            'status':'queued','created_at':'2026-10-16T01:00:00+08:00','progress':{}}


@pytest.mark.parametrize('phase',['queued','running','checkpoint','result_committed','cancel_requested','terminal'])
def test_crash_receipt_is_durable_without_auto_resume(tmp_path,phase):
    path=tmp_path/'journal.db'
    script='''
import os,sys
from my_strategy.storage.task_journal import TaskJournal
journal=TaskJournal(sys.argv[1]); phase=sys.argv[2]
job,_=journal.create({'job_id':'crash','fingerprint':'f','kind':'prediction','spec':{'symbol':'600000.SH'},'status':'queued','progress':{}})
if phase!='queued':
    job.update(status='running',progress={'completed':17,'checkpoint':'frozen-checkpoint'})
    if phase=='result_committed':job['result_ref']={'type':'cas','cache_key':'immutable-result'}
    if phase=='terminal':job.update(status='succeeded',finished_at='2026-10-16T01:10:00+08:00')
    journal.update(job)
if phase=='cancel_requested':journal.cancel('crash')
os._exit(0)
'''
    subprocess.run([sys.executable,'-c',script,str(path),phase],check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    journal=TaskJournal(path); saved=journal.get('crash')
    assert saved['status']==('succeeded' if phase=='terminal' else 'interrupted')
    assert saved['spec']=={'symbol':'600000.SH'}
    if phase!='queued': assert saved['progress']['completed']==17
    if phase=='result_committed': assert saved['result_ref']['cache_key']=='immutable-result'
    if phase=='cancel_requested': assert saved['cancel_requested'] and saved['interrupted_from']=='cancelling'
    assert journal.recent()==[saved]


def test_active_dedup_and_remote_cancel_survive_two_instances(tmp_path):
    left=TaskJournal(tmp_path/'journal.db'); right=TaskJournal(left.path)
    with ThreadPoolExecutor(2) as pool:
        answers=list(pool.map(lambda pair:pair[0].create(receipt(pair[1])),[(left,'left'),(right,'right')]))
    assert sum(created for _,created in answers)==1
    job=answers[0][0]; assert answers[1][0]['job_id']==job['job_id']
    right.cancel(job['job_id'])
    stop=JournalStop(left,job['job_id'],threading.Event())
    assert stop.is_set()
    job.update(status='running',progress={'completed':3});left.update(job)
    assert left.get(job['job_id'])['status']=='cancelling'
    job['status']='succeeded';left.update(job)
    assert left.get(job['job_id'])['status']=='cancelled'


def test_pid_reuse_is_not_a_live_owner(tmp_path):
    journal=TaskJournal(tmp_path/'journal.db');job,_=journal.create(receipt())
    job['owner_started']-=1;journal.update(job)
    assert TaskJournal(journal.path).get('one')['status']=='interrupted'


def raw_fixture(tmp_path):
    path=tmp_path/'raw.db'
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE bars(value INTEGER)');conn.execute('INSERT INTO bars VALUES(1)')
    return path


def test_status_hot_reads_do_not_execute_aggregate_and_wal_commit_invalidates(tmp_path):
    path=raw_fixture(tmp_path);cache=tmp_path/'external.db'; calls=[]
    def compute():
        calls.append(1)
        with closing(sqlite3.connect(path)) as conn:
            return {'available':True,'rows':conn.execute('SELECT COUNT(*) FROM bars').fetchone()[0]}
    first=read_status(path,compute,cache_path=cache)
    for _ in range(10):assert read_status(path,compute,cache_path=cache)['rows']==1
    assert len(calls)==1 and not first['summary_stale']
    with sqlite3.connect(path) as conn:
        conn.execute('PRAGMA journal_mode=WAL');conn.execute('INSERT INTO bars VALUES(2)');conn.commit()
        assert read_status(path,compute,cache_path=cache)['rows']==2
    read_status(path,compute,cache_path=cache,refresh=True)
    assert len(calls)==3
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()==[('bars',)]


def test_status_concurrent_refresh_single_aggregate_and_failed_new_generation_stays_stale(tmp_path):
    path=raw_fixture(tmp_path);cache=tmp_path/'external.db';calls=[]
    def compute():calls.append(1);time.sleep(.03);return {'rows':1}
    with ThreadPoolExecutor(8) as pool:
        results=list(pool.map(lambda _:read_status(path,compute,cache_path=cache),range(8)))
    assert len(calls)==1 and all(not item['summary_stale'] for item in results)
    with sqlite3.connect(path) as conn:conn.execute('INSERT INTO bars VALUES(2)')
    def fail():raise ValueError('partial update verification failed')
    saved=read_status(path,fail,cache_path=cache)
    assert saved['rows']==1 and saved['summary_stale'] and 'partial' in saved['summary_error']
    assert saved['summary_revision']==results[0]['summary_revision']


def test_status_changed_during_aggregate_is_never_published_as_new_revision(tmp_path):
    path=raw_fixture(tmp_path);cache=tmp_path/'external.db'
    def changing():
        with sqlite3.connect(path) as conn:conn.execute('INSERT INTO bars VALUES(2)')
        return {'rows':1}
    answer=read_status(path,changing,cache_path=cache)
    assert answer['summary_stale']
    with sqlite3.connect(cache) as conn: assert conn.execute('SELECT COUNT(*) FROM raw_status').fetchone()[0]==0


def scan_fixture(n=203):
    return {'run_id':'scan','coverage':{'requested':n,'success':n},'rows':[
        {'symbol':f'{i:06d}.SZ','name':f'Stock {i}','category':'buy' if i%2 else 'watch',
         'reference_action':'SELL' if i%7==0 else 'WAIT','events':[{'side':'SELL','time':'2026-10-15','private_vector':[1]*200}],
         'raw_history':[i]*500} for i in range(n)]}


def test_scan_pages_equal_original_projection_and_do_not_read_full_report(tmp_path,monkeypatch):
    store=ResultStore(tmp_path/'results.db',tmp_path/'runs');source=scan_fixture();store.save('scan',source)
    path=tmp_path/'runs/scan/reports/czsc.json';before=path.read_bytes();expected=compact_scan_view(source)['rows']
    original=Path.read_text
    def forbidden(self,*args,**kw):
        if self==path:raise AssertionError('hot page read the whole report')
        return original(self,*args,**kw)
    monkeypatch.setattr(Path,'read_text',forbidden)
    pages=[store.get_scan_page('scan',page=i,page_size=20) for i in range(1,12)]
    assert sum((p['rows'] for p in pages),[])==expected
    assert all(p['scan_view']['report_sha256']==hashlib.sha256(before).hexdigest() for p in pages)
    assert len(store.get_scan_page('scan',page=1,page_size=20,query='stock 3')['rows'])==11
    assert store.get_scan_page('scan',page=1,category='favorites',symbols=[])['rows']==[]
    assert store.get_scan_page('scan',page=1,category='buy')['scan_view']['total']==101
    assert path.read_bytes()==before


def test_old_report_lazily_indexed_without_mutating_file_and_changed_identity_rejected(tmp_path):
    store=ResultStore(tmp_path/'results.db',tmp_path/'runs');source=scan_fixture(3);store.save('scan',source)
    path=tmp_path/'runs/scan/reports/czsc.json';before=path.read_bytes()
    with sqlite3.connect(store.db_path) as conn:
        conn.execute('DELETE FROM scan_views');conn.execute('DELETE FROM scan_view_rows')
    assert store.get_scan_page('scan',page=1)['rows']==compact_scan_view(source)['rows']
    assert path.read_bytes()==before
    path.write_text(json.dumps({**source,'run_id':'forged'}))
    with pytest.raises(ValueError,match='identity'):store.get_scan_page('scan',page=1)


def test_ten_thousand_tasks_only_decode_the_page_and_do_not_open_cas(tmp_path,monkeypatch):
    task_path=tmp_path/'tasks.db'
    with sqlite3.connect(task_path) as conn:conn.execute('CREATE TABLE czsc_tasks(job_id TEXT PRIMARY KEY,fingerprint TEXT,status TEXT,payload TEXT)')
    with ResearchStore(tmp_path/'cache') as store:
        journal=TaskJournal(tmp_path/'cache/task_journal.db')
        light=SimpleNamespace(store=store,journal=journal,lock=threading.RLock(),jobs={},stops={})
        tasks=SimpleNamespace(db_path=task_path,results=ResultStore(tmp_path/'results.db',tmp_path/'runs'),_lock=threading.RLock())
        maintenance=TaskMaintenance(tasks,light,metadata_root=tmp_path)
        with sqlite3.connect(task_path) as conn:
            conn.executemany('INSERT INTO czsc_tasks VALUES(?,?,?,?)',[(str(i),str(i),'succeeded',json.dumps({**receipt(str(i),str(i)),'status':'succeeded','kind':'scan','result':{'large':[0]*100}})) for i in range(10000)])
        monkeypatch.setattr(store,'get_by_key',lambda *_:pytest.fail('history opened a CAS object'))
        import my_strategy.services.task_maintenance as module
        original=module.json.loads;decoded=[]
        def load(value,*args,**kw):decoded.append(1);return original(value,*args,**kw)
        monkeypatch.setattr(module.json,'loads',load)
        answer=maintenance.overview(status='terminal',page=499,page_size=20)
        assert answer['total']==10000 and len(answer['items'])==20 and answer['active_total']==0
        assert len(decoded)==20
        assert all('result' not in item for item in answer['items'])


def test_light_receipt_precedes_queue_and_final_result_rehydrates(tmp_path,monkeypatch):
    from my_strategy.services.light_research import LightResearch
    from my_strategy.services import kronos_prediction
    monkeypatch.setattr(kronos_prediction,'release_prediction_resources',lambda:None)
    service=LightResearch(tmp_path/'light')
    seen=[]
    def predict(**kwargs):
        seen.extend(service.journal.recent())
        return {'status':'completed','forecast':[{'date':'2026-10-16','close':10}], 'reused':False}
    monkeypatch.setattr(service,'prediction',predict)
    job=service.submit('prediction',{'symbol':'600000.SH'})
    service.queue.shutdown(wait=True)
    saved=service.journal.get(job['job_id'])
    assert seen and seen[0]['status']=='running' and saved['status']=='succeeded'
    assert saved['result_ref']['type']=='cas'
    service.close()
    restored=LightResearch(tmp_path/'light')
    try:
        assert restored.get_job(job['job_id'])['result']['forecast'][0]['close']==10
        assert not restored.jobs
    finally:restored.close()


def test_failed_queue_submission_keeps_failed_receipt(tmp_path,monkeypatch):
    from my_strategy.services.light_research import LightResearch
    from my_strategy.services import kronos_prediction
    monkeypatch.setattr(kronos_prediction,'release_prediction_resources',lambda:None)
    service=LightResearch(tmp_path/'light')
    monkeypatch.setattr(service.queue,'submit',lambda *_:(_ for _ in ()).throw(RuntimeError('queue refused')))
    try:
        with pytest.raises(RuntimeError,match='queue refused'):service.submit('prediction',{})
        assert service.journal.recent()[0]['status']=='failed'
    finally:service.close()


def test_explicit_submit_reaps_dead_owner_without_restarting_existing_instance(tmp_path):
    journal=TaskJournal(tmp_path/'journal.db')
    old,_=journal.create(receipt('old'))
    old['owner_started']-=1;journal.update(old)
    new,created=journal.create(receipt('new'))
    assert created and new['job_id']=='new'
    assert journal.get('old')['status']=='interrupted'


@pytest.mark.parametrize('phase',['before_cas','after_cas'])
def test_business_output_intent_recovers_real_crash_before_terminal_receipt(tmp_path,phase):
    script='''
import os,sys
from my_strategy.storage.task_journal import TaskJournal,bind_publication_intents
from my_strategy.storage.research_store import ResearchStore
root=sys.argv[1];phase=sys.argv[2]
journal=TaskJournal(root+'/journal.db');store=ResearchStore(root+'/cache')
job,_=journal.create({'job_id':'crash','fingerprint':'f','kind':'score_training','spec':{},'status':'running','progress':{}})
context=bind_publication_intents(store,journal);context.job=job
if phase=='before_cas':
    intent=journal.output_intent
    def exit_after_intent(*args):intent(*args);os._exit(0)
    journal.output_intent=exit_after_intent
else:
    journal.output_committed=lambda *args:os._exit(0)
store.put({'model':'frozen'}, {'weights':[1.,2.,3.]},namespace='mlp_score_models',pin='model:frozen')
raise AssertionError('fault injection missed')
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path),phase],check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    journal=TaskJournal(tmp_path/'journal.db')
    assert journal.get('crash')['status']=='interrupted'
    with ResearchStore(tmp_path/'cache') as store:
        output=journal.recovered_outputs('crash',store)
        assert len(output)==1 and output[0]['namespace']=='mlp_score_models'
        assert output[0]['publication_phase']=='intended'
        assert output[0]['availability']==('verified' if phase=='after_cas' else 'not_committed_or_retired')
        if phase=='after_cas':
            assert store.get_by_key(output[0]['cache_key'])=={'weights':[1.,2.,3.]}
            # A distinct intended value never becomes a falsely recovered result.
            with journal.connection() as conn:
                conn.execute("UPDATE job_outputs SET value_sha256='different'")
            assert journal.recovered_outputs('crash',store)[0]['availability']=='value_mismatch'
        assert journal.get('crash')['status']=='interrupted'
