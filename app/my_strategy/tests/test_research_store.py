from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
from pathlib import Path
import sqlite3
import threading

import pytest

from my_strategy.storage.research_store import ResearchStore, CacheCapacityError, _enable_wal


def test_existing_store_adds_sha_index_without_rewriting_results_or_gc(tmp_path):
    now = [100.]
    tables = ('objects', 'results', 'pins', 'leases', 'settings', 'gc_journal', 'result_retirements')
    with ResearchStore(tmp_path, ttl_seconds=10, clock=lambda: now[0]) as store:
        receipt = store.put({'input': 1}, {'value': 1}, pin='saved')
        store.acquire_lease(receipt['cache_key'], 'reader', ttl_seconds=100)
        store.set_setting('current', receipt)
        # Model the prior on-disk schema, preserving actual persisted contents.
        store.connection.execute('DROP INDEX research_results_sha256')
        before = {table: [tuple(row) for row in store.connection.execute(f'SELECT * FROM {table}')]
                  for table in tables}
        content = store._path(receipt['sha256']).read_bytes()
    now[0] = 111.
    for _ in range(2):
        with ResearchStore(tmp_path, ttl_seconds=10, clock=lambda: now[0]) as store:
            indexes = [row[1] for row in store.connection.execute('PRAGMA index_list(results)')]
            assert indexes.count('research_results_sha256') == 1
            assert [row[2] for row in store.connection.execute('PRAGMA index_info(research_results_sha256)')] == ['sha256']
            after = {table: [tuple(row) for row in store.connection.execute(f'SELECT * FROM {table}')]
                     for table in tables}
            assert after == before
            assert store._path(receipt['sha256']).read_bytes() == content
            assert store.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_gc_sha_query_uses_index_and_preserves_pin_lease_orphan_rules(tmp_path):
    now = [100.]
    with ResearchStore(tmp_path, ttl_seconds=10, clock=lambda: now[0]) as store:
        pinned = store.put({'key': 1}, {'value': 1}, pin='saved')
        leased = store.put({'key': 2}, {'value': 2})
        store.acquire_lease(leased['cache_key'], 'reader', ttl_seconds=100)
        expired = store.put({'key': 3}, {'value': 3})
        orphan = gzip.compress(b'{"uncommitted":true}', mtime=0)
        sha = hashlib.sha256(orphan).hexdigest()
        path = store._path(sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orphan)
        now[0] = 111.
        statements = []
        store.connection.set_trace_callback(statements.append)
        preview = store.gc()
        store.connection.set_trace_callback(None)
        # Explain the actual production query captured above, not a test-only
        # approximation of its correlated pin/lease/shared-object predicates.
        query = next(sql for sql in statements if 'MAX(r.expires_at) AS expires_at' in sql)
        plan = [row[3] for row in store.connection.execute('EXPLAIN QUERY PLAN ' + query)]
        assert any('SEARCH x USING INDEX research_results_sha256 (sha256=?)' in step for step in plan)
        assert not any(step == 'SCAN x' for step in plan)
        assert {row['sha256'] for row in preview['candidates']} == {expired['sha256'], sha}
        assert {row['reason'] for row in preview['candidates']} == {'expired', 'uncommitted_orphan'}
        assert path.exists() and store._path(expired['sha256']).exists()
        actual = store.gc(False)
        assert actual['deleted_bytes'] == len(orphan) + expired['size_bytes']
        assert not path.exists() and not store._path(expired['sha256']).exists()
        assert store.get({'key': 1}) == {'value': 1}
        assert store.get({'key': 2}) == {'value': 2}
        assert store.is_pinned(pinned['cache_key'])


def test_candidate_cap_preserves_each_owner_lease_setting_and_shared_content(tmp_path):
    now = [100.]
    with ResearchStore(tmp_path, clock=lambda: now[0]) as store:
        items = []
        for index in range(8):
            now[0] += 1
            items.append(store.put({'model': index}, {'output': index}, namespace='score_models'))
        keys = [item['cache_key'] for item in items]
        store.pin(keys[0], 'saved_result:A')
        store.pin(keys[0], 'saved_account:B')
        assert store.is_pinned(keys[0])
        store.unpin(keys[0], 'saved_result:A')
        assert store.is_pinned(keys[0])
        store.acquire_lease(keys[1], 'running_job')
        store.set_setting('current_score', {'cache_key': keys[2]})
        store.put({'shared_saved': 3}, {'output': 3}, pin='saved_backtest')
        preview = store.limit_unpinned('score_models')
        assert {row['cache_key'] for row in preview['retired']} == set(keys[3:6])
        assert len(store.list('score_models')) == 8
        receipt = store.limit_unpinned('score_models', dry_run=False)
        assert receipt['retired_count'] == 3
        assert {row['cache_key'] for row in store.list('score_models')} == set(keys[:3] + keys[6:])
        assert store.get({'shared_saved': 3}) == {'output': 3}
        assert store.connection.execute('SELECT count(*) FROM result_retirements').fetchone()[0] == 3
        assert store._path(items[3]['sha256']).exists()
        assert not store._path(items[4]['sha256']).exists()


def test_preview_identity_and_serialization_do_not_create_artifacts(tmp_path, monkeypatch):
    from my_strategy.core import paths, run_context
    monkeypatch.setattr(paths, 'ARTIFACT_RUNS_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(run_context, 'current_git_commit', lambda **_: 'test')
    for _ in range(100):
        identity = run_context.create_run_identity(task='preview', run_id='preview')
        assert identity.to_dict()['run_id'] == 'preview'
        run_context.RunContext(**identity.__dict__).to_dict()
    assert not (tmp_path / 'runs').exists()
    run_context.create_run_context(task='saved', run_id='saved')
    assert (tmp_path / 'runs/saved/metadata.json').exists()


def test_complete_identity_dedup_and_corruption(tmp_path):
    with ResearchStore(tmp_path) as store:
        identity = {'data_sha': 'a', 'model_sha': 'm', 'as_of': '2026-10-14', 'seed': 3, 'device': 'cpu'}
        for _ in range(10): receipt = store.put(identity, {'predictions': [1, 2, 3]})
        assert len(list(store.objects.rglob('*.json.gz'))) == 1
        assert store.get(identity) == {'predictions': [1, 2, 3]}
        for key in identity:
            assert store.get({**identity, key: 'changed'}) is None
        with pytest.raises(ValueError, match='different immutable'):
            store.put(identity, {'predictions': [3, 2, 1]})
        store._path(receipt['sha256']).chmod(0o644)
        store._path(receipt['sha256']).write_bytes(b'corrupt')
        with pytest.raises(ValueError, match='SHA/size'): store.get(identity)


def test_pin_owners_and_leases_survive_expiry(tmp_path):
    now = [100.]
    with ResearchStore(tmp_path, ttl_seconds=10, clock=lambda: now[0]) as store:
        item = store.put({'input': 1}, {'value': 1}, pin='account:A')
        key = item['cache_key']
        store.pin(key, 'saved:B')
        store.acquire_lease(key, 'reader', ttl_seconds=100)
        now[0] = 111
        assert store.gc(False)['deleted_bytes'] == 0
        store.unpin(key, 'account:A')
        assert store.get({'input': 1}) is not None
        store.unpin(key, 'saved:B')
        assert store.gc(False)['deleted_bytes'] == 0
        store.release_lease(key, 'reader')
        preview = store.gc()
        assert preview['candidate_bytes'] == item['size_bytes']
        assert store._path(item['sha256']).exists()
        assert store.gc(False)['deleted_bytes'] == item['size_bytes']
        assert store.get({'input': 1}) is None
        assert store.connection.execute("SELECT state FROM gc_journal").fetchone()[0] == 'deleted'


def test_quota_lru_and_leased_capacity_fail_without_overcommit(tmp_path):
    now = [100.]
    with ResearchStore(tmp_path, quota_bytes=120, clock=lambda: now[0]) as store:
        a = store.put({'n': 1}, {'a': 'first result'})
        now[0] += 1
        b = store.put({'n': 2}, {'a': 'second result'})
        store.acquire_lease(b['cache_key'], 'reader')
        now[0] += 1
        c = store.put({'n': 3}, {'a': 'third result'})
        store.acquire_lease(c['cache_key'], 'reader')
        assert store.get({'n': 1}) is None
        assert store.get({'n': 2}) and store.get({'n': 3})
        with pytest.raises(CacheCapacityError): store.put({'n': 4}, {'a': 'fourth result'})
        assert sum(x['size_bytes'] for x in store.list()) <= 120


def test_pinned_bytes_are_outside_disposable_cache_quota(tmp_path):
    with ResearchStore(tmp_path, quota_bytes=40) as store:
        saved = store.put({'key': 1}, {'some_long_saved_field': 'a saved backtest'}, pin='saved:1')
        assert saved['size_bytes'] > 40
        assert store.gc()['cache_bytes'] == 0
        assert store.gc()['pinned_bytes'] == saved['size_bytes']
        assert store.get_by_key(saved['cache_key'])['some_long_saved_field'] == 'a saved backtest'
        assert store.list()[0]['identity'] == {'key': 1}


def test_mutable_small_settings_do_not_create_or_change_cas_objects(tmp_path):
    with ResearchStore(tmp_path) as store:
        store.set_setting('current_score', {'id': 'one'})
        store.set_setting('current_score', {'id': 'two'})
        assert store.get_setting('current_score') == {'id': 'two'}
        assert store.get_setting('missing') is None
        assert store.list() == []
        with pytest.raises(ValueError): store.set_setting('large', 'x' * 70000)


def test_shared_object_not_deleted_while_any_identity_is_pinned(tmp_path):
    now = [1.]
    with ResearchStore(tmp_path, ttl_seconds=1, clock=lambda: now[0]) as store:
        first = store.put({'key': 1}, {'shared': True})
        second = store.put({'key': 2}, {'shared': True}, pin='saved')
        assert first['sha256'] == second['sha256']
        now[0] = 10.
        assert store.gc(False)['deleted_bytes'] == 0
        store.unpin(second['cache_key'], 'saved')
        assert store.gc(False)['deleted_bytes'] == first['size_bytes']


def test_gc_retries_interrupted_unlink_without_adding_references(tmp_path, monkeypatch):
    now = [1.]
    with ResearchStore(tmp_path, ttl_seconds=1, clock=lambda: now[0]) as store:
        item = store.put({'key': 1}, {'value': 1})
        now[0] = 10.
        original = Path.unlink
        def denied(path, *args, **kwargs):
            if path.name.endswith('.json.gz'): raise PermissionError('simulated interruption')
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'unlink', denied)
        assert store.gc(False)['deleted_bytes'] == 0
        assert store.get({'key': 1}) is None
        with pytest.raises(KeyError): store.pin(item['cache_key'], 'new-reader')
        assert store.connection.execute('SELECT state FROM gc_journal').fetchone()[0] == 'pending'
        monkeypatch.setattr(Path, 'unlink', original)
        assert store.gc(False)['deleted_bytes'] == item['size_bytes']


@pytest.mark.parametrize('attempt', range(3))
def test_concurrent_same_result_has_one_object_and_one_identity(tmp_path, attempt):
    barrier = threading.Barrier(4)
    def write(_):
        barrier.wait(timeout=10)
        with ResearchStore(tmp_path) as store: return store.put({'input': 1}, {'value': 2})['sha256']
    with ThreadPoolExecutor(max_workers=4) as pool: hashes = list(pool.map(write, range(12)))
    assert len(set(hashes)) == 1
    with ResearchStore(tmp_path) as store:
        assert len(store.list()) == 1
        assert len(list(store.objects.rglob('*.json.gz'))) == 1
        assert store.connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        assert store.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_first_wal_transition_retries_busy_and_existing_wal_needs_no_transition(tmp_path):
    class BusyOnce(sqlite3.Connection):
        transitions = 0
        def execute(self, sql, *args, **kwargs):
            if sql == 'PRAGMA journal_mode=WAL':
                self.transitions += 1
                if self.transitions == 1:
                    error = sqlite3.OperationalError('database is locked')
                    error.sqlite_errorcode = sqlite3.SQLITE_BUSY
                    raise error
            return super().execute(sql, *args, **kwargs)
    connection = sqlite3.connect(tmp_path / 'first.db', isolation_level=None, factory=BusyOnce)
    try:
        _enable_wal(connection, timeout=1)
        assert connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        assert connection.transitions == 2
        _enable_wal(connection, timeout=1)
        assert connection.transitions == 2
    finally:
        connection.close()


def test_wal_lock_retry_has_a_deadline():
    class Locked:
        attempts = 0
        def execute(self, _):
            self.attempts += 1
            raise sqlite3.OperationalError('database is locked')
    connection = Locked()
    with pytest.raises(sqlite3.OperationalError, match='locked'):
        _enable_wal(connection, timeout=0)
    assert connection.attempts == 1


def test_non_lock_database_error_is_not_retried_and_failed_constructor_closes(tmp_path, monkeypatch):
    from my_strategy.storage import research_store
    original_connect = sqlite3.connect
    opened = []
    class ReadOnly(sqlite3.Connection):
        attempts = 0
        def execute(self, sql, *args, **kwargs):
            if sql == 'PRAGMA journal_mode':
                self.attempts += 1
                raise sqlite3.OperationalError('attempt to write a readonly database')
            return super().execute(sql, *args, **kwargs)
    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs, factory=ReadOnly)
        opened.append(connection)
        return connection
    monkeypatch.setattr(research_store.sqlite3, 'connect', connect)
    with pytest.raises(sqlite3.OperationalError, match='readonly'):
        ResearchStore(tmp_path)
    assert opened[0].attempts == 1
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        opened[0].execute('SELECT 1')


def test_expired_and_missing_pin_cannot_resurrect_gc_target(tmp_path):
    now = [1.]
    with ResearchStore(tmp_path, ttl_seconds=1, clock=lambda: now[0]) as store:
        item = store.put({'n': 1}, {'x': 2})
        now[0] = 20
        with pytest.raises(KeyError): store.acquire_lease(item['cache_key'], 'reader')
        with pytest.raises(KeyError): store.pin(item['cache_key'], 'saved')
        with pytest.raises(ValueError): store.unpin(item['cache_key'], '')


def test_crash_between_file_write_and_index_commit_recovers_orphan(tmp_path):
    with ResearchStore(tmp_path) as store:
        content=gzip.compress(b'{"aborted":true}',mtime=0)
        sha=hashlib.sha256(content).hexdigest();path=store._path(sha)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        preview=store.gc()
        assert preview['candidates'][0]['reason']=='uncommitted_orphan'
        assert path.exists()
        assert store.gc(False)['deleted_bytes']==len(content)
        assert not path.exists()
