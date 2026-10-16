"""Accounting, publication and reader/GC safety across connection boundaries."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
import hashlib
import multiprocessing
from pathlib import Path
import sqlite3
import threading

import pytest

from my_strategy.storage.research_store import ResearchStore, CacheCapacityError


def assert_accounting(store):
    conn = store.connection
    expected = conn.execute('''SELECT COALESCE(SUM(size_bytes),0),COALESCE(SUM(CASE WHEN NOT EXISTS(
        SELECT 1 FROM results r JOIN pins p USING(cache_key) WHERE r.sha256=o.sha256)
        THEN size_bytes ELSE 0 END),0) FROM objects o''').fetchone()
    assert tuple(conn.execute('SELECT total_bytes,cache_bytes FROM research_totals').fetchone()) == tuple(expected)
    for obj in conn.execute('SELECT * FROM objects'):
        sha = obj['sha256']
        expected = conn.execute('''SELECT COUNT(*),COALESCE(MAX(expires_at),0),COALESCE(MAX(accessed_at),0)
            FROM results WHERE sha256=?''', (sha,)).fetchone()
        pin_count = conn.execute('SELECT COUNT(*) FROM pins JOIN results USING(cache_key) WHERE sha256=?', (sha,)).fetchone()[0]
        lease_until = conn.execute('SELECT COALESCE(MAX(l.expires_at),0) FROM leases l JOIN results USING(cache_key) WHERE sha256=?', (sha,)).fetchone()[0]
        usage = conn.execute('SELECT * FROM object_usage WHERE sha256=?', (sha,)).fetchone()
        assert (usage['ref_count'], usage['expires_at'], usage['accessed_at']) == tuple(expected)
        assert usage['pin_count'] == pin_count and usage['lease_until'] == lease_until
        assert usage['state'] == obj['state'] and usage['size_bytes'] == obj['size_bytes']
    assert conn.execute('SELECT COUNT(*) FROM object_usage').fetchone()[0] == conn.execute('SELECT COUNT(*) FROM objects').fetchone()[0]


def test_legacy_migration_is_once_and_preserves_all_original_rows(tmp_path):
    tables = ('objects', 'results', 'pins', 'leases', 'settings', 'gc_journal', 'result_retirements')
    with ResearchStore(tmp_path, clock=lambda: 100.) as store:
        refs = store.put_many([{'identity': {'n': n}, 'result': {'shared': n // 2}, 'pin': 'account' if n == 0 else None} for n in range(4)])
        store.acquire_lease(refs[2]['cache_key'], 'worker', 500)
        store.set_setting('pointer', refs[0])
        before = {t: [tuple(r) for r in store.connection.execute('SELECT * FROM '+t)] for t in tables}
        files = {r['sha256']: store._path(r['sha256']).read_bytes() for r in refs}
        # Remove only new derived schema to reproduce the actual legacy layout.
        for name, in store.connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
            store.connection.execute('DROP TRIGGER '+name)
        for t in ('object_usage', 'research_totals', 'research_accounting_version'):
            store.connection.execute('DROP TABLE '+t)
    for _ in range(2):
        with ResearchStore(tmp_path, clock=lambda: 200.) as store:
            assert {t: [tuple(r) for r in store.connection.execute('SELECT * FROM '+t)] for t in tables} == before
            assert {sha: store._path(sha).read_bytes() for sha in files} == files
            assert_accounting(store)
            assert store.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_shared_sha_pins_leases_refresh_retirement_and_transaction_rollback(tmp_path):
    now = [100.]
    with ResearchStore(tmp_path, clock=lambda: now[0], ttl_seconds=20) as store:
        a, b = store.put_many([{'identity': {'n': n}, 'result': {'shared': True}} for n in range(2)])
        store.pin(a['cache_key'], 'account:A'); store.pin(a['cache_key'], 'model:B'); store.pin(b['cache_key'], 'model:C')
        store.acquire_leases([a['cache_key'], b['cache_key']], 'reader', 200)
        assert_accounting(store)
        store.unpin(a['cache_key'], 'account:A'); store.unpin(b['cache_key'], 'model:C')
        now[0] = 110.
        store.put({'n': 1}, {'shared': True}, ttl_seconds=1)
        store.acquire_lease(a['cache_key'], 'reader', 5)  # shrinking one lease cannot lose the other
        assert_accounting(store)
        with pytest.raises(RuntimeError):
            with store._tx():
                store.connection.execute('DELETE FROM results WHERE cache_key=?', (a['cache_key'],))
                raise RuntimeError('rollback')
        assert store.is_pinned(a['cache_key']); assert_accounting(store)
        # Maintenance may open its own connection without enabling foreign keys.
        with sqlite3.connect(tmp_path / 'research.db') as maintenance:
            maintenance.execute('DELETE FROM results WHERE cache_key=?', (a['cache_key'],))
        assert_accounting(store)
        now[0] = 150.
        assert store.gc(False)['deleted_bytes'] == 0
        store.release_lease(b['cache_key'], 'reader')
        assert store.gc(False)['deleted_bytes'] == a['size_bytes']
        assert_accounting(store)


def test_hot_reads_settings_and_listing_do_not_need_sqlite_writer(tmp_path):
    with ResearchStore(tmp_path, clock=lambda: 100.) as store:
        receipt = store.put({'n': 1}, {'value': 2}, pin='saved')
        store.set_setting('value', 3)
        with sqlite3.connect(tmp_path / 'research.db', isolation_level=None) as writer:
            writer.execute('BEGIN IMMEDIATE')
            try:
                store.connection.execute('PRAGMA busy_timeout=1')
                assert store.get({'n': 1}) == {'value': 2}
                assert store.get_by_key(receipt['cache_key'], touch=False) == {'value': 2}
                assert store.get_setting('value') == 3
                assert store.is_pinned(receipt['cache_key'])
                assert len(store.list()) == 1
            finally:
                writer.rollback()


def test_decode_holds_neither_sqlite_writer_nor_python_connection_lock(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = gzip.decompress
    def slow_decode(content):
        entered.set()
        assert release.wait(5)
        return original(content)
    with ResearchStore(tmp_path, clock=lambda: 100.) as store:
        store.put({'n': 1}, {'value': 2})
        monkeypatch.setattr(gzip, 'decompress', slow_decode)
        with ThreadPoolExecutor(max_workers=1) as pool:
            read = pool.submit(store.get, {'n': 1})
            assert entered.wait(5)
            try:
                store.put({'n': 2}, {'value': 3})
            finally:
                release.set()
            assert read.result() == {'value': 2}


def test_gc_wins_before_open_returns_miss_without_resurrecting_reference(tmp_path, monkeypatch):
    with ResearchStore(tmp_path, clock=lambda: 100.) as reader, ResearchStore(tmp_path, clock=lambda: 100.) as collector:
        receipt = reader.put({'n': 1}, {'value': 2})
        original = Path.open
        target = reader._path(receipt['sha256'])
        def racing_open(path, *args, **kwargs):
            if path == target:
                collector.gc(False, required_bytes=collector.quota_bytes)
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'open', racing_open)
        assert reader.get({'n': 1}) is None
        assert reader.get({'n': 1}) is None
        assert_accounting(reader)


def test_open_descriptor_preserves_complete_verified_bytes_after_gc_unlink(tmp_path, monkeypatch):
    with ResearchStore(tmp_path, clock=lambda: 100.) as reader, ResearchStore(tmp_path, clock=lambda: 100.) as collector:
        receipt = reader.put({'n': 1}, {'value': 'retained by the descriptor'})
        original = Path.open
        target = reader._path(receipt['sha256'])
        def racing_open(path, *args, **kwargs):
            stream = original(path, *args, **kwargs)
            if path == target:
                collector.gc(False, required_bytes=collector.quota_bytes)
            return stream
        monkeypatch.setattr(Path, 'open', racing_open)
        assert reader.get({'n': 1}) == {'value': 'retained by the descriptor'}
        assert reader.get({'n': 1}) is None
        assert_accounting(reader)


def test_missing_active_file_is_corruption_not_hidden_as_gc_miss(tmp_path):
    with ResearchStore(tmp_path) as store:
        receipt = store.put({'n': 1}, {'value': 2})
        path = store._path(receipt['sha256'])
        # Deliberately corrupt the fixture; Windows cannot unlink a read-only CAS object.
        path.chmod(path.stat().st_mode | 0o200)
        path.unlink()
        with pytest.raises(FileNotFoundError):
            store.get_by_key(receipt['cache_key'], touch=False)


def test_batch_publication_is_atomic_and_leased_shared_bytes_count_once(tmp_path):
    now = [100.]
    with ResearchStore(tmp_path, quota_bytes=40, clock=lambda: now[0], ttl_seconds=1) as store:
        refs = store.put_many([{'identity': {'n': n}, 'result': {'shared': True}} for n in range(4)], lease_owner='build', lease_ttl_seconds=100)
        assert len({r['sha256'] for r in refs}) == 1
        assert store._cache_bytes() == refs[0]['size_bytes']
        assert_accounting(store)
        now[0] = 110.
        assert store.gc(False)['deleted_bytes'] == 0
        assert store.release_leases([r['cache_key'] for r in refs], 'build') == 4
        assert store.gc(False)['deleted_bytes'] == refs[0]['size_bytes']
        assert_accounting(store)


def test_batch_capacity_or_immutable_conflict_never_publishes_partial_results(tmp_path):
    with ResearchStore(tmp_path, quota_bytes=65) as store:
        with pytest.raises(CacheCapacityError):
            store.put_many([{'identity': {'n': n}, 'result': {'value': 'result '+str(n)}} for n in range(3)], lease_owner='build')
        assert store.list() == []
        assert_accounting(store)
        with pytest.raises(ValueError, match='different immutable'):
            store.put_many([{'identity': {'n': 1}, 'result': {'value': n}} for n in range(2)])
        assert store.list() == []
        assert_accounting(store)


def test_batch_failed_filesystem_publish_leaves_only_recoverable_orphan(tmp_path, monkeypatch):
    with ResearchStore(tmp_path) as store:
        original = store._path
        calls = [0]
        def interrupted(sha):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError('simulated crash after one file')
            return original(sha)
        monkeypatch.setattr(store, '_path', interrupted)
        with pytest.raises(OSError, match='simulated crash'):
            store.put_many([{'identity': {'n': n}, 'result': {'value': n}} for n in range(2)], lease_owner='build')
        assert store.list() == []
        assert store.connection.execute('SELECT COUNT(*) FROM leases').fetchone()[0] == 0
        assert_accounting(store)
        monkeypatch.setattr(store, '_path', original)
        preview = store.gc()
        assert len(preview['candidates']) == 1 and preview['candidates'][0]['reason'] == 'uncommitted_orphan'
        assert store.gc(False)['deleted_bytes'] == preview['candidate_bytes']
        assert_accounting(store)


def test_batch_lease_failure_rolls_back_preceding_lease_and_read_only_peek(tmp_path):
    now = [100.]
    with ResearchStore(tmp_path, clock=lambda: now[0]) as store:
        ref = store.put({'n': 1}, {'value': 1})
        with pytest.raises(KeyError):
            store.acquire_leases([ref['cache_key'], 'missing'], 'build')
        assert store.connection.execute('SELECT COUNT(*) FROM leases').fetchone()[0] == 0
        now[0] = 140.
        assert store.get_by_key(ref['cache_key'], touch=False) == {'value': 1}
        assert store.list()[0]['accessed_at'] == 100.
        assert store.get_by_key(ref['cache_key']) == {'value': 1}
        assert store.list()[0]['accessed_at'] == 140.
        assert_accounting(store)


def test_regular_put_uses_indexed_probes_and_no_full_cache_aggregation(tmp_path):
    with ResearchStore(tmp_path, clock=lambda: 100.) as store:
        store.put_many([{'identity': {'n': n}, 'result': {'value': n}} for n in range(100)])
        trace = []
        store.connection.set_trace_callback(trace.append)
        store.put({'n': 101}, {'value': 101})
        store.connection.set_trace_callback(None)
        assert not any('GROUP BY' in sql.upper() or 'SUM(' in sql.upper() for sql in trace)
        assert not any(sql == 'SELECT sha256 FROM objects' for sql in trace)
        expiry = next(sql for sql in trace if 'MAX(expires_at,lease_until)<=' in sql)
        plan = [r[3] for r in store.connection.execute('EXPLAIN QUERY PLAN '+expiry)]
        assert any('research_usage_expiry' in step for step in plan)
        trace.clear(); store.connection.set_trace_callback(trace.append)
        store.list('default', 20)
        store.connection.set_trace_callback(None)
        query = next(sql for sql in trace if 'FROM results r JOIN objects' in sql)
        plan = [r[3] for r in store.connection.execute('EXPLAIN QUERY PLAN '+query)]
        assert any('research_results_namespace_access' in step for step in plan)


def _process_put(root, n):
    with ResearchStore(root, quota_bytes=90, clock=lambda: 100.) as store:
        try:
            return store.put_many([{'identity': {'process': n}, 'result': {'value': 'payload '+str(n)}}], lease_owner='worker')[0]
        except CacheCapacityError:
            return None


def test_cross_process_quota_and_leases_never_overcommit(tmp_path):
    with ResearchStore(tmp_path, quota_bytes=90, clock=lambda: 100.):
        pass
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(_process_put, [str(tmp_path)] * 6, range(6)))
    assert 0 < sum(r is not None for r in results) < 6
    with ResearchStore(tmp_path, quota_bytes=90, clock=lambda: 100.) as store:
        assert store._cache_bytes() <= 90
        assert len(store.list()) == sum(r is not None for r in results)
        assert_accounting(store)


def test_manifest_publication_refreshes_dependencies_in_same_transaction(tmp_path):
    now = [100.]
    with ResearchStore(tmp_path, ttl_seconds=10, clock=lambda: now[0]) as store:
        chunk = store.put({'chunk': 1}, {'rows': [1, 2]})
        now[0] = 109.
        manifest = store.put({'manifest': 1}, {'chunk': chunk['cache_key']}, refresh_keys=[chunk['cache_key']])
        expiry = store.connection.execute('SELECT expires_at FROM results WHERE cache_key=?', (chunk['cache_key'],)).fetchone()[0]
        assert expiry >= manifest['expires_at']
        now[0] = 111.
        assert store.gc(False)['deleted_bytes'] == 0
        assert store.get_by_key(chunk['cache_key'], touch=False) == {'rows': [1, 2]}
        with pytest.raises(KeyError):
            store.put({'manifest': 2}, {'chunk': 'missing'}, refresh_keys=[chunk['cache_key'], 'missing'])
        assert store.get({'manifest': 2}) is None
        assert_accounting(store)
        now[0] = 120.
        with pytest.raises(KeyError): store.refresh_expiry([chunk['cache_key']])
        with pytest.raises(KeyError): store.put({'manifest': 3}, {'chunk': chunk['cache_key']}, refresh_keys=[chunk['cache_key']])
        assert store.get({'manifest': 3}) is None


def test_batch_ttl_and_leases_begin_after_slow_publication_io(tmp_path, monkeypatch):
    now=[100.]
    with ResearchStore(tmp_path,clock=lambda:now[0],ttl_seconds=1) as store:
        original=store._path
        def slow_path(sha):
            now[0]+=5
            return original(sha)
        monkeypatch.setattr(store,'_path',slow_path)
        refs=store.put_many([{'identity':{'slow':n},'result':{'value':n}} for n in range(2)],
            lease_owner='build',lease_ttl_seconds=2)
        monkeypatch.setattr(store,'_path',original)
        assert all(ref['expires_at']==now[0]+1 for ref in refs)
        assert {row[0] for row in store.connection.execute('SELECT expires_at FROM leases')}=={now[0]+2}
        assert all(store.get_by_key(ref['cache_key'],touch=False) is not None for ref in refs)
        assert_accounting(store)
