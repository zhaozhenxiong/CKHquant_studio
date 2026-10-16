"""TaskManager writes with the real TaskMaintenance SQLite projection triggers."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.task_maintenance import TaskMaintenance
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.storage.research_store import ResearchStore
from my_strategy.web_dashboard.tasks import TaskManager


@pytest.fixture
def indexed(tmp_path):
    results = ResultStore(tmp_path/'results.db', tmp_path/'runs')
    manager = TaskManager(tmp_path/'metadata/tasks.db', results, workers=1)
    with ResearchStore(tmp_path/'metadata/light_research') as store:
        light = SimpleNamespace(store=store, jobs={}, stops={}, lock=threading.RLock())
        maintenance = TaskMaintenance(manager, light)
        try:
            yield SimpleNamespace(manager=manager, maintenance=maintenance, results=results)
        finally:
            manager.close()


def synthetic(job_id='synthetic-job', status='queued', **values):
    return dict(job_id=job_id, kind='board_features', spec={'label': '合成板块'},
                status=status, progress=0, error=None, run_id=None,
                created_at='2026-01-01T00:00:00+08:00', **values)


def read_tables(manager):
    with closing(sqlite3.connect(manager.db_path)) as db:
        return {table: db.execute('SELECT * FROM '+table+' ORDER BY job_id').fetchall()
                for table in ('czsc_tasks', 'task_display')}


def assert_projection(manager, job_id, status):
    with closing(sqlite3.connect(manager.db_path)) as db:
        task = db.execute('SELECT status,payload FROM czsc_tasks WHERE job_id=?', (job_id,)).fetchone()
        display = db.execute('SELECT status,payload FROM task_display WHERE job_id=?', (job_id,)).fetchone()
    assert task[0] == display[0] == status
    public = {k: v for k, v in json.loads(task[1]).items()
              if k not in {'result', 'fingerprint', 'owner_pid', 'owner_started'}}
    assert json.loads(display[1]) == public


def terminal(manager, job_id):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job['status'] not in {'queued', 'running', 'cancelling'}:
            return job
        time.sleep(.01)
    raise AssertionError('Task did not reach a persisted terminal state')


def test_updates_preserve_fingerprint_payload_and_existing_trigger_schema(indexed):
    manager = indexed.manager
    first = synthetic(result={'must_not_persist': True})
    manager._write(first)
    with closing(sqlite3.connect(manager.db_path)) as db:
        schema = db.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
    changed = {**first, 'status': 'running', 'spec': {'label': '中文更新'}, 'kind': 'scan'}
    manager._write(changed)
    assert_projection(manager, first['job_id'], 'running')
    with closing(sqlite3.connect(manager.db_path)) as db:
        row = db.execute('SELECT fingerprint,payload FROM czsc_tasks').fetchone()
        assert row[0] == stable_hash({'kind': first['kind'], 'spec': first['spec']})
        assert row[1] == json.dumps({k: v for k, v in changed.items() if k != 'result'},
                                   ensure_ascii=False, allow_nan=False)
        assert db.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall() == schema


@pytest.mark.parametrize('outcome', ['succeeded', 'partial', 'failed'])
def test_real_worker_persists_running_and_terminal_with_index(indexed, monkeypatch, outcome):
    manager = indexed.manager
    entered, release = threading.Event(), threading.Event()
    def run(job_id, kind, spec):
        entered.set()
        assert release.wait(5)
        if outcome == 'failed':
            raise ValueError('synthetic worker failure')
        return {'run_id': 'synthetic-'+job_id, 'status': 'completed' if outcome == 'succeeded' else outcome}
    monkeypatch.setattr(manager, '_run', run)
    job = manager.submit('board_features', {'fixture': outcome})
    try:
        assert entered.wait(5)
        assert_projection(manager, job['job_id'], 'running')
    finally:
        release.set()
    finished = terminal(manager, job['job_id'])
    assert finished['status'] == outcome
    assert_projection(manager, job['job_id'], outcome)
    assert indexed.maintenance.list()['items'][0]['status'] == outcome
    if outcome == 'failed':
        assert 'synthetic worker failure' in finished['error']
        assert finished['result'] is None
    else:
        assert finished['result']['run_id'] == finished['run_id']


@pytest.mark.parametrize('queued', [False, True])
def test_cancel_running_or_queued_job_is_persisted_without_execution_after_cancel(indexed, monkeypatch, queued):
    manager = indexed.manager
    entered, release = threading.Event(), threading.Event()
    executed = []
    def run(job_id, kind, spec):
        executed.append(job_id)
        entered.set()
        assert release.wait(5)
        manager._check(job_id)
        return {'run_id': 'synthetic-'+job_id, 'status': 'completed'}
    monkeypatch.setattr(manager, '_run', run)
    first = manager.submit('board_features', {'fixture': 'first'})
    target = first
    try:
        assert entered.wait(5)
        if queued:
            target = manager.submit('board_features', {'fixture': 'queued'})
            assert_projection(manager, target['job_id'], 'queued')
        receipt = manager.cancel(target['job_id'])
        assert receipt['status'] == 'cancelling'
        assert_projection(manager, target['job_id'], 'cancelling')
    finally:
        release.set()
    assert terminal(manager, target['job_id'])['status'] == 'cancelled'
    assert_projection(manager, target['job_id'], 'cancelled')
    assert executed == [first['job_id']]
    if queued:
        assert terminal(manager, first['job_id'])['status'] == 'succeeded'


def test_dead_owner_reconciliation_uses_existing_history_index(indexed):
    manager = indexed.manager
    job = synthetic(owner_pid=-1, owner_started=0)
    manager._write(job)
    reopened = TaskManager(manager.db_path, indexed.results, workers=1)
    try:
        finished = reopened.get(job['job_id'])
        assert finished['status'] == 'failed' and '服务重启中断' in finished['error']
        assert_projection(reopened, job['job_id'], 'failed')
        assert finished['run_id'] is None and not reopened._cancel
    finally:
        reopened.close()


def test_display_trigger_failure_rolls_back_both_tables_and_releases_writer(indexed):
    manager = indexed.manager
    manager._write(synthetic())
    before = read_tables(manager)
    with closing(sqlite3.connect(manager.db_path)) as db:
        db.execute("""CREATE TRIGGER synthetic_display_failure AFTER INSERT ON task_display
                      WHEN NEW.status='failed' BEGIN SELECT RAISE(ABORT,'synthetic projection failure'); END""")
        db.commit()
    with pytest.raises(sqlite3.IntegrityError, match='synthetic projection failure'):
        manager._patch('synthetic-job', status='failed')
    assert read_tables(manager) == before
    # A new independent connection can write immediately after rollback/close.
    with closing(sqlite3.connect(manager.db_path, timeout=.1)) as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('DROP TRIGGER synthetic_display_failure')
        db.commit()
    manager._patch('synthetic-job', status='cancelled')
    assert_projection(manager, 'synthetic-job', 'cancelled')


def test_nonfinite_payload_rejection_leaves_both_tables_unchanged(indexed):
    manager = indexed.manager
    job = synthetic(); manager._write(job)
    before = read_tables(manager)
    with pytest.raises(ValueError, match='JSON'):
        manager._write({**job, 'progress': float('nan')})
    assert read_tables(manager) == before


def test_independent_writers_serialize_missing_row_insertion_and_display_update(indexed):
    manager = indexed.manager
    second = TaskManager(manager.db_path, indexed.results, workers=1)
    start = threading.Barrier(2)
    def write(instance, status):
        start.wait(timeout=5)
        for index in range(12):
            instance._write(synthetic(job_id=f'synthetic-concurrent-{index}', status=status))
    try:
        with ThreadPoolExecutor(max_workers=2) as writers:
            futures = [writers.submit(write, instance, status)
                       for instance, status in ((manager, 'running'), (second, 'queued'))]
            for future in futures:
                future.result(timeout=10)
        tables = read_tables(manager)
        assert len(tables['czsc_tasks']) == len(tables['task_display']) == 12
        for job_id, _, status, _ in tables['czsc_tasks']:
            assert_projection(manager, job_id, status)
    finally:
        second.close()
