"""Real threads preserve source identities, global limits and coordinator writes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import timedelta
import json
import sqlite3
import threading
import time

import pytest

from my_strategy.data_manager import board_provider as provider_module
from my_strategy.data_manager.board_provider import (
    FuyaoBoardProvider, BoardAccessError, BoardBudgetExhausted, BoardCollectionStopped,
    CATALOG_PATH, MEMBERS_PATH, SOURCE,
)
from my_strategy.scripts.refresh_board_knowledge import refresh_knowledge, build_parser
from my_strategy.storage.board_knowledge import BoardKnowledgeStore
from my_strategy.tests.test_board_provider import Response, envelope, members, raw_db


class Source:
    def __init__(self, count=8, delay=.02, interval=0, budget=2000):
        self.rows = [{'thscode': f'{881100+i}.TI', 'name': f'行业{i}'} for i in range(count)]
        self.delay, self.interval, self.budget = delay, interval, budget
        self.lock = threading.Lock(); self.sessions = []; self.calls = []
        self.active = 0; self.peak = 0

    def session(self):
        source = self
        class Session:
            def __init__(self): self.threads = set(); self.closed = False
            def get(self, url, **kwargs):
                with source.lock:
                    self.threads.add(threading.get_ident())
                    source.calls.append({'at': time.monotonic(), 'url': url, 'params': kwargs['params']})
                    source.active += 1; source.peak = max(source.peak, source.active)
                try:
                    if url.endswith(CATALOG_PATH): return Response(envelope(source.rows))
                    time.sleep(source.delay)
                    # Independent order of worker completion; exact member proof remains unchanged.
                    return Response(members(('000001.SZ', '600000.SH')))
                finally:
                    with source.lock: source.active -= 1
            def close(self): self.closed = True
        session = Session(); self.sessions.append(session); return session

    def provider(self):
        return FuyaoBoardProvider(session_factory=self.session, request_budget=self.budget,
            min_interval=self.interval)


@pytest.fixture(autouse=True)
def credential(monkeypatch): monkeypatch.setenv('FUYAO_API_KEY', 'test-only-credential')


def run(source, raw_db, path, **kwargs):
    provider = source.provider()
    try:
        return refresh_knowledge(db_path=path, raw_db_path=raw_db, provider=provider,
            kinds=['industry'], min_interval=source.interval, **kwargs)
    finally: provider.close()


def test_real_threads_overlap_with_independent_sessions_and_single_writer(raw_db, tmp_path, monkeypatch):
    source = Source()
    owner = threading.get_ident(); writers = []; progress = []
    original = BoardKnowledgeStore.import_snapshot
    def imported(self, *args, **kwargs):
        writers.append(threading.get_ident()); return original(self, *args, **kwargs)
    monkeypatch.setattr(BoardKnowledgeStore, 'import_snapshot', imported)
    result = run(source, raw_db, tmp_path/'parallel.db', workers=4, on_progress=progress.append)
    assert result['status'] == 'succeeded' and result['fresh_boards'] == 8
    assert result['workers'] == 4 and result['reused_boards'] == 0
    assert source.peak == 4 and len(source.calls) == 18
    assert set(writers) == {owner} and len(writers) == 8
    assert all(len(session.threads) == 1 for session in source.sessions)
    assert len({next(iter(session.threads)) for session in source.sessions}) == 5
    assert all(session.closed for session in source.sessions)
    assert [p['completed'] for p in progress] == list(range(9)) + [8]
    rows = BoardKnowledgeStore(tmp_path/'parallel.db', readonly=True).catalog(current=True)
    assert [row['board_id'] for row in rows] == ['THS_I_'+item['thscode'] for item in source.rows]
    assert all(row['provenance']['membership_set_stable'] for row in rows)


def test_all_threads_share_exact_budget_and_preserve_resumable_successes(raw_db, tmp_path):
    source = Source(count=12, delay=.01, budget=7)
    result = run(source, raw_db, tmp_path/'budget.db', workers=8)
    assert result['status'] == 'interrupted' and result['request_count'] == 7
    assert len(source.calls) == 7 and source.active == 0
    assert result['completed_boards'] < 12
    saved = BoardKnowledgeStore(tmp_path/'budget.db').get_refresh(result['run_id'])
    assert saved['status'] == 'running' and len(saved['board_ids']) == 12
    assert saved['completed_board_ids'] == result['completed_board_ids']


def test_global_start_interval_applies_across_actual_threads():
    source = Source(count=8, delay=.025, interval=.01, budget=10)
    provider = source.provider()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(provider._get, MEMBERS_PATH, {'thscode': '881100.TI'}) for _ in range(10)]
            for future in futures: future.result()
        starts = sorted(item['at'] for item in source.calls)
        assert len(starts) == 10 and provider.request_count == 10
        assert min(b-a for a,b in zip(starts,starts[1:])) >= .009
        assert source.peak > 1
    finally: provider.close()


def test_access_refusal_wakes_all_rate_waiters_without_another_request():
    source = Source(interval=.5)
    provider = source.provider()
    source.session  # The provider's initial session owns the first request.
    for session in source.sessions:
        session.get = lambda *args, **kwargs: Response({}, 429)
    try:
        # Freeze factories to sessions that all reject, while retaining independent objects.
        original = provider._session_factory
        def factory():
            session = original(); session.get = lambda *args, **kwargs: Response({}, 429); return session
        provider._session_factory = factory
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(provider._get, MEMBERS_PATH, {'thscode': '881100.TI'}) for _ in range(8)]
            for future in futures:
                with pytest.raises(BoardAccessError, match='429'): future.result()
        assert provider.request_count == 1 and time.monotonic()-started < .4
    finally: provider.close()


def test_cancel_during_network_wait_stops_new_dispatch_and_never_publishes_full(raw_db, tmp_path):
    source = Source(count=20, delay=.06)
    class Cancelled(RuntimeError): pass
    def cancelled():
        if any(item['url'].endswith(MEMBERS_PATH) for item in source.calls):
            raise Cancelled('explicit test cancellation')
    with pytest.raises(Cancelled):
        run(source, raw_db, tmp_path/'cancel.db', workers=4, check_cancel=cancelled)
    assert 1 <= len([item for item in source.calls if item['url'].endswith(MEMBERS_PATH)]) <= 4
    store = BoardKnowledgeStore(tmp_path/'cancel.db')
    state = store.status()['latest_run']
    assert state['status'] == 'running' and len(state['completed_board_ids']) == 0
    assert store.catalog(current=True) == [] and source.active == 0


def test_same_day_reuse_keeps_actual_snapshot_sha_and_original_observation(raw_db, tmp_path):
    path = tmp_path/'reuse.db'; first = Source(count=4)
    original = run(first, raw_db, path, workers=4)
    before = BoardKnowledgeStore(path).catalog(current=True)
    second = Source(count=4)
    result = run(second, raw_db, path, workers=4)
    after = BoardKnowledgeStore(path).catalog(current=True)
    assert result['status'] == 'succeeded' and result['run_id'] != original['run_id']
    assert result['reused_boards'] == 4 and result['fresh_boards'] == 0
    assert result['request_count'] == len(second.calls) == 2
    assert before == after
    saved = BoardKnowledgeStore(path).get_refresh(result['run_id'])
    assert saved['snapshot_refs'] == {row['board_id']:row['snapshot_hash'] for row in before}
    assert BoardKnowledgeStore(path).get_refresh(original['run_id'])['status'] == 'succeeded'


def test_force_refresh_does_not_reuse_same_day_members(raw_db, tmp_path):
    path = tmp_path/'force.db'
    run(Source(count=3), raw_db, path)
    source = Source(count=3)
    result = run(source, raw_db, path, reuse_today=False)
    assert result['fresh_boards'] == 3 and result['reused_boards'] == 0 and len(source.calls) == 8


def test_one_changed_directory_member_prevents_reuse_of_whole_frozen_kind(raw_db, tmp_path):
    path = tmp_path/'changed.db'
    run(Source(count=3), raw_db, path)
    source = Source(count=3); source.rows[-1]['name'] = '新名称'
    result = run(source, raw_db, path)
    assert result['fresh_boards'] == 3 and result['reused_boards'] == 0
    assert result['request_count'] == 8


def test_yesterdays_verified_observations_are_fetched_again(raw_db, tmp_path, monkeypatch):
    path = tmp_path/'yesterday.db'; actual = provider_module.local_now
    monkeypatch.setattr(provider_module, 'local_now', lambda: actual()-timedelta(days=1))
    run(Source(count=2), raw_db, path)
    monkeypatch.setattr(provider_module, 'local_now', actual)
    result = run(Source(count=2), raw_db, path)
    assert result['fresh_boards'] == 2 and result['reused_boards'] == 0


def test_signed_snapshot_tampering_cannot_be_masked_by_today_cache(raw_db, tmp_path):
    path = tmp_path/'tamper.db'; run(Source(count=2), raw_db, path)
    with sqlite3.connect(path) as con:
        con.execute('DROP TRIGGER immutable_board_snapshots_update')
        con.execute("UPDATE board_snapshots SET payload_json=payload_json || ' '")
    with pytest.raises(ValueError, match='SHA drift'):
        run(Source(count=2), raw_db, path)


def test_unpublished_partial_worker_output_cannot_be_reused(raw_db, tmp_path):
    source = Source(count=1); provider = source.provider()
    try:
        boards = provider.catalog(['industry']); snapshot = provider.snapshot(boards[0])
    finally: provider.close()
    store = BoardKnowledgeStore(tmp_path/'unpublished.db')
    first = store.begin_refresh(SOURCE, boards)
    value = store.import_snapshot(snapshot, first)
    second = store.begin_refresh(SOURCE, boards)
    with pytest.raises(ValueError, match='completed verified publication'):
        store.reuse_current_snapshot(second, boards[0]['board_id'], value['snapshot_hash'])
    assert store.get_refresh(first)['status'] == store.get_refresh(second)['status'] == 'running'


@pytest.mark.parametrize('workers', [0,9,True,1.5])
def test_invalid_worker_values_rejected_before_state_mutation(tmp_path, workers):
    path = tmp_path/'invalid.db'
    with pytest.raises(ValueError, match='workers'):
        refresh_knowledge(db_path=path, workers=workers)
    assert not path.exists()


def test_cli_exposes_bounded_threads_and_explicit_force_refresh():
    parser = build_parser(); args = parser.parse_args(['--workers','8','--no-reuse-today'])
    assert args.workers == 8 and not args.reuse_today
    assert parser.parse_args([]).workers == 4 and parser.parse_args([]).reuse_today
    with pytest.raises(SystemExit): parser.parse_args(['--workers','9'])


def test_request_budget_cannot_be_overshot_by_simultaneous_threads():
    source = Source(delay=.005, budget=7); provider = source.provider()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(provider._get, MEMBERS_PATH, {'thscode':'881100.TI'}) for _ in range(30)]
            completed, refused = 0, 0
            for future in futures:
                try: future.result(); completed += 1
                except BoardBudgetExhausted: refused += 1
        assert completed == len(source.calls) == provider.request_count == 7
        assert refused == 23
    finally: provider.close()


def test_completed_parent_is_immutable_and_retry_only_fetches_frozen_remaining(raw_db, tmp_path):
    path = tmp_path/'retry.db'; source = Source(count=4, budget=4)
    first = run(source, raw_db, path, workers=1)
    assert first['status'] == 'interrupted' and first['completed_boards'] == 1
    store = BoardKnowledgeStore(path)
    terminal = store.finish_refresh(first['run_id'])
    assert terminal['status'] == 'partial'
    parent = store.get_refresh(first['run_id'])
    second = Source(count=100)  # Changed live directory must never replace frozen retry population.
    result = run(second, raw_db, path, workers=4, resume=first['run_id'])
    assert result['status'] == 'succeeded' and result['run_id'] != first['run_id']
    assert result['resumed_from'] == first['run_id'] and result['requested_boards'] == 3
    assert result['request_count'] == 6
    assert all(item['url'].endswith(MEMBERS_PATH) for item in second.calls)
    assert {item['params']['thscode'] for item in second.calls} == {'881101.TI','881102.TI','881103.TI'}
    assert store.get_refresh(first['run_id']) == parent
    assert len(store.catalog(current=True)) == 4


def test_three_failures_stop_dispatch_and_cannot_publish_fake_success(raw_db, tmp_path):
    from my_strategy.data_manager.board_provider import BoardSourceError
    source = Source(count=20, delay=.005); provider = source.provider()
    calls = []
    def broken(board):
        calls.append(board['board_id']); time.sleep(.005)
        raise BoardSourceError('fixture full response invalid')
    provider.snapshot = broken
    try:
        result = refresh_knowledge(db_path=tmp_path/'failures.db', raw_db_path=raw_db,
            provider=provider, kinds=['industry'], workers=4)
    finally: provider.close()
    assert result['status'] == 'failed' and result['completed_boards'] == 0
    # Three deterministic failures plus at most four already outstanding stocks.
    assert len(calls) <= 6
    assert len([item for item in result['failures'] if item['board_id']]) == 3
    assert any(item['board_id'] is None and item['scope'] == 'source' for item in result['failures'])
    assert BoardKnowledgeStore(tmp_path/'failures.db').catalog(current=True) == []


def test_stop_interrupts_global_interval_wait_without_spending_request_budget():
    source = Source(interval=10); provider = source.provider()
    try:
        provider._get(CATALOG_PATH, {'tag':'industry'})
        started = threading.Event()
        def waiting():
            started.set(); return provider._get(CATALOG_PATH, {'tag':'industry'})
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(waiting); assert started.wait(1)
            provider.stop()
            with pytest.raises(BoardCollectionStopped): future.result(timeout=.4)
        assert len(source.calls) == provider.request_count == 1
    finally: provider.close()


def test_worker_completion_order_keeps_frozen_numerical_snapshot_content(raw_db, tmp_path, monkeypatch):
    serial, parallel = Source(count=5), Source(count=5)
    # Match the observation clock; thread scheduling may affect no source facts.
    actual = provider_module.local_now()
    from my_strategy.scripts import refresh_board_knowledge
    from my_strategy.storage import board_knowledge
    monkeypatch.setattr(provider_module, 'local_now', lambda: actual)
    monkeypatch.setattr(refresh_board_knowledge, 'local_now', lambda: actual)
    monkeypatch.setattr(board_knowledge, '_now', lambda: actual)
    one = run(serial, raw_db, tmp_path/'serial.db', workers=1)
    many = run(parallel, raw_db, tmp_path/'parallel-identical.db', workers=4)
    first = BoardKnowledgeStore(tmp_path/'serial.db').catalog(current=True)
    second = BoardKnowledgeStore(tmp_path/'parallel-identical.db').catalog(current=True)
    assert first == second and one['request_count'] == many['request_count'] == 12


def test_api_exposes_workers_reuse_and_counts_without_leaking_source_envelopes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from my_strategy.web_dashboard.scripts.serve_board_knowledge import create_board_app
    from my_strategy.scripts import refresh_board_knowledge
    from my_strategy.tests.test_board_api import _wait_job
    calls = []
    def refresh(**kwargs):
        calls.append(kwargs)
        kwargs['on_progress']({'completed':3,'total':3,'workers':6,'reused_boards':2,'fresh_boards':1,'request_count':6})
        return {'status':'succeeded','run_id':'a'*32,'workers':6,'requested_workers':6,'min_interval':1.3,
            'reused_boards':2,'fresh_boards':1,'request_count':6,'requested_boards':3,'completed_boards':3,
            'catalog_provenance':{'source_response':'NEVER_TO_BROWSER'}}
    monkeypatch.setattr(refresh_board_knowledge, 'refresh_knowledge', refresh)
    app = create_board_app(db_path=tmp_path/'api.db')
    with TestClient(app) as client:
        launched = client.post('/api/boards/refresh',json={'workers':6,'reuse_today':False,'min_interval':1.3})
        assert launched.status_code == 202
        result = _wait_job(client, launched.json()['job_id'])
        assert result['status'] == 'succeeded' and result['result']['reused_boards'] == 2
        assert result['result']['fresh_boards'] == 1 and result['result']['workers'] == 6
        assert 'NEVER_TO_BROWSER' not in str(result)
    assert calls[0]['workers'] == 6 and calls[0]['reuse_today'] is False and calls[0]['min_interval'] == 1.3


@pytest.mark.parametrize('bad', [{'workers':0},{'workers':9},{'workers':True},{'workers':1.5},
    {'min_interval':-1},{'reuse_today':'yes'}])
def test_api_rejects_invalid_concurrency_before_a_job_exists(tmp_path, bad):
    from fastapi.testclient import TestClient
    from my_strategy.web_dashboard.scripts.serve_board_knowledge import create_board_app
    app = create_board_app(db_path=tmp_path/'api-invalid.db')
    with TestClient(app) as client:
        assert client.post('/api/boards/refresh', json=bad).status_code == 422
        assert client.get('/api/boards/jobs').json()['jobs'] == []


from my_strategy.tests.test_current_dashboard import dashboard


def test_main_dashboard_passes_workers_limits_reuse_and_cancel_callback(dashboard, tmp_path, monkeypatch):
    from my_strategy.scripts import refresh_board_knowledge
    from my_strategy.storage import board_fluent
    client, manager = dashboard
    actual_store = board_fluent.BoardFluentStore
    monkeypatch.setattr(board_fluent, 'BoardFluentStore', lambda *a, **k: actual_store(tmp_path/'task-fluent.db'))
    calls = []
    def refresh(**kwargs):
        calls.append(kwargs); kwargs['check_cancel']()
        kwargs['on_progress']({'run_id':'a'*32,'completed':2,'total':2,'workers':6,'min_interval':1.3,
            'reused_boards':1,'fresh_boards':1,'request_count':4})
        return {'status':'succeeded','run_id':'a'*32,'workers':6,'min_interval':1.3,'reused_boards':1,
            'fresh_boards':1,'request_count':4,'requested_boards':2,'completed_boards':2}
    monkeypatch.setattr(refresh_board_knowledge, 'refresh_knowledge', refresh)
    launched = client.post('/api/boards/refresh',json={'workers':6,'reuse_today':False,'limit_budget':17})
    assert launched.status_code == 202
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        job = client.get('/api/boards/jobs/'+launched.json()['job_id']).json()
        if job['status'] not in {'queued','running'}: break
        time.sleep(.01)
    assert job['status']=='succeeded',job
    assert calls[0]['workers']==6 and calls[0]['min_interval']==1.3
    assert calls[0]['reuse_today'] is False and calls[0]['limit_budget']==17 and callable(calls[0]['check_cancel'])
    assert job['progress_detail']['workers']==6 and job['progress_detail']['reused_boards']==1
    assert job['result']['workers']==6 and job['result']['fresh_boards']==1


def test_main_cancel_drains_workers_and_seals_actual_partial_manifest(dashboard, raw_db, tmp_path, monkeypatch):
    from my_strategy.scripts import refresh_board_knowledge
    from my_strategy.storage import board_fluent, board_knowledge
    client, manager = dashboard
    actual_fluent = board_fluent.BoardFluentStore; actual_knowledge=board_knowledge.BoardKnowledgeStore
    monkeypatch.setattr(board_fluent, 'BoardFluentStore', lambda *a, **k: actual_fluent(tmp_path/'cancel-fluent.db'))
    monkeypatch.setattr(board_knowledge, 'BoardKnowledgeStore', lambda *a, **k: actual_knowledge(tmp_path/'cancel-kb.db'))
    original=refresh_board_knowledge.refresh_knowledge; source=Source(count=40,delay=.05)
    seen=threading.Event()
    def refresh(**kwargs):
        progress=kwargs.pop('on_progress')
        def emit(value):
            progress(value)
            if value['completed']>=1: seen.set()
        provider=source.provider()
        try:return original(**kwargs,db_path=tmp_path/'cancel-kb.db',raw_db_path=raw_db,
            provider=provider,on_progress=emit)
        finally:provider.close()
    monkeypatch.setattr(refresh_board_knowledge,'refresh_knowledge',refresh)
    launched=client.post('/api/boards/refresh',json={'workers':4,'min_interval':0})
    assert launched.status_code==202 and seen.wait(3)
    identifier=launched.json()['job_id']
    assert client.post('/api/boards/jobs/'+identifier+'/cancel',json={}).status_code==200
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        job=client.get('/api/boards/jobs/'+identifier).json()
        if job['status']=='cancelled':break
        time.sleep(.01)
    assert job['status']=='cancelled',job
    state=actual_knowledge(tmp_path/'cancel-kb.db').get_refresh(job['knowledge_run_id'])
    assert state['status']=='partial' and 1<=len(state['completed_board_ids'])<40
    assert source.active==0
    sync=actual_fluent(tmp_path/'cancel-fluent.db').get_sync(job['progress_detail']['sync_id'])
    assert sync['status']=='cancelled'
