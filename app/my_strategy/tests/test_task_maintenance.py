from types import SimpleNamespace
from pathlib import Path
import json
import sqlite3
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from my_strategy.services.task_maintenance import TaskMaintenance, MaintenanceConflict
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.storage.research_store import ResearchStore
from my_strategy.web_dashboard.task_history_api import router


@pytest.fixture
def maintenance(tmp_path):
    root = tmp_path / 'data' / 'metadata'; root.mkdir(parents=True)
    tasks_db = root / 'czsc_tasks.db'
    with sqlite3.connect(tasks_db) as conn:
        conn.execute('CREATE TABLE czsc_tasks(job_id TEXT PRIMARY KEY,fingerprint TEXT,status TEXT,payload TEXT)')
    store = ResearchStore(root / 'light_research')
    light = SimpleNamespace(store=store, jobs={}, stops={}, lock=threading.RLock())
    results = ResultStore(tmp_path / 'data/processed/czsc/results.db', tmp_path / 'artifacts/runs')
    tasks = SimpleNamespace(db_path=tasks_db, results=results, _lock=threading.RLock())
    service = TaskMaintenance(tasks, light)
    yield service
    store.close()


def add_task(m, job='job', status='succeeded', backend='legacy', run_id=None, **extra):
    value = dict(job_id=job, kind='scan', status=status, spec={}, created_at='2026-10-15T00:00:00+08:00', run_id=run_id, **extra)
    if backend == 'legacy':
        with sqlite3.connect(m.tasks.db_path) as conn:
            conn.execute('INSERT OR REPLACE INTO czsc_tasks VALUES(?,?,?,?)', (job, job, status, json.dumps(value)))
    else:
        m.light.jobs[job] = value
        if status not in {'running', 'queued', 'cancelling'}:
            m.light.store.put({'job_id': job}, value, namespace='jobs')
    return {'backend': backend, 'job_id': job}


def add_analysis(m, run='analysis-run'):
    result = {'run_id': run, 'symbol': '000001.SZ', 'result': 'independent chart'}
    m.tasks.results.save('analysis', result)
    return m.runs / run


def test_unified_history_retains_all_kinds_and_pages_without_big_results(maintenance):
    m = maintenance
    add_task(m, 'same', backend='legacy', status='cancelled')
    add_task(m, 'same', backend='light', status='succeeded')
    add_task(m, 'live', backend='light', status='running')
    m.light.jobs['live']['result'] = {'do_not_copy': True}
    page = m.list(page_size=1)
    assert page['total'] == 3 and page['counts'] == {'total': 3, 'active': 1, 'terminal': 2}
    assert page['items'][0]['job_id'] == 'live' and 'result' not in page['items'][0]
    assert not page['items'][0]['can_delete']
    assert len(m.list(status='terminal', backend='light')['items']) == 1
    assert m.list(page=4, page_size=1)['items'] == []


def test_delete_records_preserves_analysis_and_refuses_active(maintenance):
    m = maintenance; path = add_analysis(m)
    done = add_task(m, run_id=path.name)
    live = add_task(m, 'live', status='running')
    result = m.delete_records([done, live])
    assert result['deleted'] == [done] and len(result['skipped']) == 1
    assert path.exists() and m.tasks.results.get(path.name)
    assert m.list()['items'][0]['job_id'] == 'live'


def test_light_record_deletion_removes_memory_and_index_but_preserves_object(maintenance):
    m = maintenance; ref = add_task(m, backend='light')
    row = m.light.store.connection.execute("SELECT sha256 FROM results WHERE namespace='jobs'").fetchone()
    assert m.delete_records([ref])['deleted'] == [ref]
    assert not m.list()['items']
    assert m.light.store._path(row[0]).exists()


def test_preview_is_only_manifest_and_execution_cleans_exact_analysis(maintenance):
    m = maintenance; path = add_analysis(m); add_task(m, run_id=path.name)
    plan = m.preview()
    assert path.exists() and len(m.list()['items']) == 1
    assert any(c['type'] == 'analysis_run' for c in plan['candidates'])
    result = m.execute(plan['plan_id'], plan['plan_sha256'])
    assert not result['failed'] and not result['skipped']
    assert result['deleted_bytes'] == plan['candidate_bytes'] and not path.exists()
    with pytest.raises(KeyError): m.tasks.results.get(path.name)
    with sqlite3.connect(m.db_path) as conn:
        assert conn.execute("SELECT count(*) FROM cleanup_events WHERE status='deleted'").fetchone()[0] == 2


@pytest.mark.parametrize('reference_kind', ['model', 'native', 'calendar', 'board', 'account'])
def test_external_references_protect_old_analysis(maintenance, reference_kind):
    m = maintenance; path = add_analysis(m)
    if reference_kind == 'account':
        m.light.store.put({'account': 'saved'}, {'source_run_id': path.name}, namespace='simulations', pin='saved_account')
    elif reference_kind == 'calendar':
        folder = m.runs / 'calendar' / 'reports'; folder.mkdir(parents=True)
        (folder / 'calendar.json').write_text(json.dumps({'parents': [path.name]}))
    else:
        with sqlite3.connect(m.root / (reference_kind + '.db')) as conn:
            conn.execute('CREATE TABLE dependencies(payload TEXT)')
            conn.execute('INSERT INTO dependencies VALUES(?)', (json.dumps({'path': str(path / 'reports/czsc.json')}),))
    plan = m.preview()
    assert not any(c['type'] == 'analysis_run' for c in plan['candidates'])
    assert any(p.get('run_id') == path.name and p.get('references') for p in plan['protected'])


def test_model_file_inside_analysis_directory_excludes_the_whole_directory(maintenance):
    m = maintenance; path = add_analysis(m)
    (path / 'needed_model.bin').write_bytes(b'protected')
    plan = m.preview()
    assert not plan['candidates'] and plan['protected']


def test_new_reference_between_preview_and_execute_is_rechecked(maintenance):
    m = maintenance; path = add_analysis(m); plan = m.preview()
    m.light.store.set_setting('enabled_accounts', {'source': path.name})
    result = m.execute(plan['plan_id'], plan['plan_sha256'])
    assert path.exists() and not result['deleted_bytes'] and result['skipped']


def test_changed_manifest_is_not_deleted(maintenance):
    m = maintenance; path = add_analysis(m); plan = m.preview()
    (path / 'reports/czsc.json').write_text(json.dumps({'run_id': path.name, 'changed': True}))
    result = m.execute(plan['plan_id'], plan['plan_sha256'])
    assert path.exists() and result['skipped'] and result['deleted_bytes'] == 0


def test_active_job_arriving_after_preview_preserves_all_artifacts(maintenance):
    m = maintenance; path = add_analysis(m); plan = m.preview()
    add_task(m, 'live', backend='light', status='running')
    result = m.execute(plan['plan_id'], plan['plan_sha256'])
    assert path.exists() and result['skipped'] and not result['deleted_bytes']


@pytest.mark.parametrize('protection', ['pin', 'lease', 'settings', 'shared'])
def test_job_summary_shared_and_fixed_references_are_not_deleted(maintenance, protection):
    m = maintenance; add_task(m, backend='light'); plan = m.preview()
    store = m.light.store; key = store.cache_key({'job_id': 'job'}, 'jobs')
    if protection == 'pin': store.pin(key, 'account')
    elif protection == 'lease': store.acquire_lease(key, 'reader')
    elif protection == 'settings': store.set_setting('account', {'job': key})
    else:
        value = store.get_by_key(key)
        store.put({'other': 'reference'}, value, namespace='shared', pin='model')
    result = m.execute(plan['plan_id'], plan['plan_sha256'])
    assert result['skipped'] and result['deleted_bytes'] == 0
    assert store.get_by_key(key)


def test_unique_job_summary_is_deleted_only_after_its_record(maintenance):
    m = maintenance; add_task(m, backend='light'); plan = m.preview()
    objects = [x for x in plan['candidates'] if x['type'] == 'cache_job_object']
    assert len(objects) == 1
    result = m.execute(plan['plan_id'], plan['plan_sha256'])
    assert not result['failed'] and not result['skipped']
    assert result['deleted_bytes'] == objects[0]['size_bytes']
    assert not Path(objects[0]['path']).exists() and not m.list()['items']


def test_symlink_and_shared_hardlink_are_never_removed(maintenance, tmp_path):
    m = maintenance; path = add_analysis(m)
    external = tmp_path / 'private'; external.write_text('preserve')
    (path / 'metadata.json').symlink_to(external)
    plan = m.preview()
    assert not plan['candidates'] and external.read_text() == 'preserve'


def test_api_rejects_path_injection_and_changed_plan_identity(maintenance):
    m = maintenance; add_analysis(m)
    app = FastAPI(); app.state.tasks = m.tasks; app.state.light = m.light
    app.state.task_maintenance = m; app.state.compute_submit_lock = threading.Lock()
    app.include_router(router)
    with TestClient(app) as client:
        assert client.get('/api/task-history?status=bad').status_code == 422
        assert client.post('/api/task-history/delete', json={'tasks': [{'backend': 'legacy', 'job_id': '../../raw'}]}).status_code == 422
        plan = client.post('/api/task-history/cleanup/preview', json={}).json()
        result = client.post('/api/task-history/cleanup/execute', json={'plan_id': plan['plan_id'], 'plan_sha256': '0' * 64})
        assert result.status_code == 409


def test_interrupted_retirement_restores_exact_directory_and_index(maintenance):
    m = maintenance; path = add_analysis(m); plan = m.preview()
    item = next(c for c in plan['candidates'] if c['type'] == 'analysis_run')
    # Simulate process death after the result index commit, before physical purge.
    with m._frozen_indexes() as indexes:
        quarantine = m._stage_analysis(plan['plan_id'], item, indexes[m.results_db])
    assert not path.exists() and quarantine.exists()
    restored = m.recover(plan['plan_id'], plan['plan_sha256'])
    assert restored['restored'] == [path.name] and not restored['failed']
    assert path.exists() and not quarantine.exists() and m.tasks.results.get(path.name)


def test_execute_is_idempotent_after_success(maintenance):
    m = maintenance; add_analysis(m); plan = m.preview()
    first = m.execute(plan['plan_id'], plan['plan_sha256'])
    second = m.execute(plan['plan_id'], plan['plan_sha256'])
    assert second['replayed'] and second['deleted'] == first['deleted']


def test_all_dependency_writers_are_locked_while_graph_and_index_change(maintenance):
    m = maintenance; add_analysis(m)
    deps = m.root / 'model.db'
    with sqlite3.connect(deps) as conn: conn.execute('CREATE TABLE refs(value TEXT)')
    with m._frozen_indexes():
        with sqlite3.connect(deps, timeout=.01) as conn:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                conn.execute("INSERT INTO refs VALUES('new reference')")


def test_quarantine_parent_symlink_cannot_move_a_report_outside_root(maintenance, tmp_path):
    m = maintenance; path = add_analysis(m); plan = m.preview()
    outside = tmp_path / 'outside'; outside.mkdir()
    (m.runs / '.task-cleanup').symlink_to(outside, target_is_directory=True)
    result = m.execute(plan['plan_id'], plan['plan_sha256'])
    assert path.exists() and not list(outside.iterdir()) and result['skipped']


def test_cross_process_compute_lock_blocks_execution_without_deleting_records(maintenance):
    from filelock import FileLock
    m = maintenance; add_task(m); plan = m.preview()
    lock_path = m.root / 'compute.lock'
    m.light.compute_lock = FileLock(str(lock_path))
    with FileLock(str(lock_path)):
        with pytest.raises(MaintenanceConflict, match='计算资源'):
            m.execute(plan['plan_id'], plan['plan_sha256'])
    assert m.list()['total'] == 1


def test_sha_only_reference_also_protects_an_analysis_report(maintenance):
    from my_strategy.services.task_maintenance import file_digest
    m = maintenance; path = add_analysis(m)
    digest = file_digest(path / 'reports/czsc.json')
    m.light.store.set_setting('frozen_input', {'sha256': digest})
    plan = m.preview()
    assert not plan['candidates'] and plan['protected'][0]['references']
