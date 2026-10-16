"""Board reads are isolated from training and never initialize missing state."""
from pathlib import Path

from fastapi.testclient import TestClient

from my_strategy.web_dashboard.scripts.serve_board_knowledge import create_board_app


def test_missing_knowledge_reads_do_not_create_a_database(tmp_path):
    db = tmp_path / 'basic_kb.db'
    with TestClient(create_board_app(db_path=db)) as client:
        status = client.get('/api/boards/status')
        assert status.status_code == 200
        assert status.json()['available'] is False
        assert client.get('/api/boards/catalog').json()['boards'] == []
        stock = client.get('/api/boards/stocks/000001.SZ').json()
        assert stock['boards'] == [] and stock['features'] is None
        assert client.get('/api/boards/features/THS_I_881155.TI').json()['available'] is False
    assert not db.exists()


def test_board_routes_reject_bad_stock_and_kind(tmp_path):
    with TestClient(create_board_app(db_path=tmp_path/'kb.db')) as client:
        assert client.get('/api/boards/stocks/invalid').status_code == 422
        assert client.get('/api/boards/catalog?kind=invalid').status_code == 422
        assert client.post('/api/boards/refresh', json={'kinds':['invalid']}).status_code == 422


def test_cross_origin_writes_are_rejected_without_starting_a_job(tmp_path):
    app = create_board_app(db_path=tmp_path/'kb.db')
    with TestClient(app) as client:
        result = client.post('/api/boards/refresh', json={}, headers={'origin':'https://foreign.example'})
        assert result.status_code == 403
        assert app.state.board_jobs == {}


def test_board_workspace_is_local_editable_content():
    root = Path(__file__).resolve().parents[1] / 'web_dashboard/static'
    page = (root/'board_knowledge.html').read_text()
    script = (root/'js/board_knowledge.js').read_text()
    assert '同花顺板块知识库' in page
    assert '当前成员' in page
    assert 'id="mode"' not in page and 'id="as-of"' not in page
    assert 'FUYAO_API_KEY' not in script and 'X-api-key' not in script
    assert 'textContent' in script and 'innerHTML' not in script
    assert '/refresh' in script and '/stocks/' in script


def _wait_job(client, identifier):
    import time
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.get('/api/boards/jobs/' + identifier).json()
        if result['status'] != 'running': return result
        time.sleep(.01)
    raise AssertionError('board background job did not finish')


def _stored_cache(tmp_path, monkeypatch, feature_overrides=None):
    from datetime import datetime, timezone
    from my_strategy.storage import board_knowledge as kb
    from my_strategy.tests.test_board_knowledge import snapshot
    clock = [datetime(2026, 10, 14, 6, tzinfo=timezone.utc)]
    monkeypatch.setattr(kb, '_now', lambda: clock[0])
    path = tmp_path / 'actual-kb.db'; store = kb.BoardKnowledgeStore(path)
    store.import_snapshot(snapshot())
    original = store.catalog(current=True)[0]
    from my_strategy.services.board_features import (BOARD_FEATURE_PROFILE, BOARD_SCHEMA_HASH,
        DEFAULT_BOARD_POLICY, _code_hashes, stable_hash)
    feature = {'mode': 'current_reference', 'price_date': '2026-10-12', 'training_eligible': False,
        'profile': BOARD_FEATURE_PROFILE['name'], 'schema_hash': BOARD_SCHEMA_HASH,
        'policy_hash': stable_hash(DEFAULT_BOARD_POLICY), 'code_hashes': _code_hashes(store),
        'cache_key': 'a' * 64, 'price_content_sha256': 'b' * 64, 'calendar_hash': 'c' * 64,
        'available_at': '2026-10-14T15:00:00+08:00', 'membership_dependency_hashes': [original['snapshot_hash']]}
    feature.update(feature_overrides or {})
    store.put_feature('daily', 'original-cache', {'mode': 'current_reference', 'price_date': '2026-10-12',
        'boards': [{**feature, 'board_id': original['board_id'], 'snapshot_hash': original['snapshot_hash'], 'board_return_1': .02}],
        'stocks': [{**feature, 'symbol': '000001.SZ', 'snapshot_hashes': [original['snapshot_hash']], 'industry_return_1': .03}]})
    return path, store, original, clock


def test_actual_cached_rows_match_membership_and_catalog_sends_only_metadata(tmp_path, monkeypatch):
    path, _, original, _ = _stored_cache(tmp_path, monkeypatch)
    with TestClient(create_board_app(db_path=path)) as client:
        stock = client.get('/api/boards/stocks/000001.SZ').json()
        assert stock['features']['industry_return_1'] == .03 and stock['needs_rebuild'] is False
        assert stock['feature_reference'] == 'stored_artifact' and stock['price_date_is_frozen'] is True
        board = client.get('/api/boards/features/' + original['board_id']).json()
        assert board['features']['board_return_1'] == .02 and board['needs_rebuild'] is False
        assert board['features']['price_date'] == '2026-10-12' and board['price_date_is_frozen'] is True
        catalog = client.get('/api/boards/catalog').json()['boards'][0]
        assert catalog['member_count'] == 2
        assert 'members' not in catalog and 'provenance' not in catalog
        assert client.get('/api/boards/stocks/000001.SZ?current=false&as_of=2025-01-01').json()['features'] is None
        assert client.get('/api/boards/stocks/000001.SZ?current=true&as_of=2025-01-01').status_code == 422


def test_new_current_snapshot_invalidates_old_cached_stock_and_board_without_computing(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from my_strategy.tests.test_board_knowledge import snapshot
    from my_strategy.services import board_features
    path, store, original, clock = _stored_cache(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs): raise AssertionError('GET must never compute features')
    monkeypatch.setattr(board_features, 'build_board_features', forbidden)
    clock[0] = datetime(2026, 10, 14, 6, 10, tzinfo=timezone.utc)
    changed = snapshot(members=['000001.SZ', '600000.SH'])
    changed.update(observed_at='2026-10-14T14:09:00+08:00', available_at='2026-10-14T14:09:00+08:00', effective_at='2026-10-14T14:09:00+08:00')
    store.import_snapshot(changed)
    with TestClient(create_board_app(db_path=path)) as client:
        stock = client.get('/api/boards/stocks/000001.SZ').json()
        assert stock['features'] is None and stock['needs_rebuild'] is True
        board = client.get('/api/boards/features/' + original['board_id']).json()
        assert board['features'] is None and board['needs_rebuild'] is True
        assert client.get('/api/boards/status').status_code == 200
        assert client.get('/api/boards/catalog').status_code == 200


import pytest


@pytest.mark.parametrize('overrides', [
    {'schema_hash': 'd' * 64}, {'profile': 'old-profile'}, {'policy_hash': 'd' * 64},
    {'code_hashes': {'board_features.py': 'old-code'}}, {'price_content_sha256': None},
    {'calendar_hash': None}, {'cache_key': None}, {'price_date': None},
    {'membership_dependency_hashes': []}, {'available_at': None}, {'price_date': '2026-99-99'}])
def test_incompatible_stored_artifact_is_unavailable_without_reading_raw_or_building(tmp_path, monkeypatch, overrides):
    from my_strategy.services import board_features, czsc_trend_support
    path, _, original, _ = _stored_cache(tmp_path, monkeypatch, overrides)
    def forbidden(*args, **kwargs): raise AssertionError('GET may only read a stored artifact')
    monkeypatch.setattr(board_features, '_read_prices', forbidden)
    monkeypatch.setattr(board_features, 'build_board_features', forbidden)
    monkeypatch.setattr(czsc_trend_support, '_calendar', forbidden)
    with TestClient(create_board_app(db_path=path)) as client:
        stock = client.get('/api/boards/stocks/000001.SZ').json()
        assert stock['features'] is None and stock['needs_rebuild'] is True
        board = client.get('/api/boards/features/' + original['board_id']).json()
        assert board['features'] is None and board['needs_rebuild'] is True


def test_historical_intraday_reads_hide_later_features_and_use_shanghai_date(tmp_path, monkeypatch):
    path, store, original, _ = _stored_cache(tmp_path, monkeypatch)
    bundle = store.get_feature('daily', 'original-cache')
    bundle.update(mode='causal', price_date='2026-10-14')
    for row in bundle['boards'] + bundle['stocks']:
        row.update(mode='causal', price_date='2026-10-14', available_at='2026-10-14T15:00:00+08:00')
    store.put_feature('daily', 'historical-cache', bundle)
    with TestClient(create_board_app(db_path=path)) as client:
        stock = client.get('/api/boards/stocks/000001.SZ', params={'current': False, 'as_of': '2026-10-14T14:30:00+08:00'}).json()
        assert len(stock['boards']) == 1 and stock['features'] is None and stock['needs_rebuild'] is True
        board = client.get('/api/boards/features/' + original['board_id'], params={'current': False, 'date': '2026-10-14T14:30:00+08:00'}).json()
        assert board['features'] is None and board['needs_rebuild'] is True
        # This ISO input is the previous date in its own timezone, but the 14th in Shanghai.
        instant = '2026-10-13T23:00:00-08:00'
        stock = client.get('/api/boards/stocks/000001.SZ', params={'current': False, 'as_of': instant}).json()
        assert stock['features']['price_date'] == '2026-10-14'
        board = client.get('/api/boards/features/' + original['board_id'], params={'current': False, 'date': instant}).json()
        assert board['features']['price_date'] == '2026-10-14'
        stock = client.get('/api/boards/stocks/000001.SZ', params={'current': False, 'as_of': '2026-10-14'}).json()
        assert stock['features']['price_date'] == '2026-10-14'


@pytest.mark.parametrize('outcome', ['succeeded', 'partial', 'failed', 'interrupted'])
def test_refresh_preserves_source_outcome_and_strips_big_source_proofs(tmp_path, monkeypatch, outcome):
    from my_strategy.scripts import refresh_board_knowledge
    calls = []
    def refresh(**kwargs):
        calls.append(kwargs)
        kwargs['on_progress']({'completed': 1, 'total': 2, 'request_count': 6})
        return {'status': outcome, 'run_id': 'a' * 32, 'requested_boards': 2, 'completed_boards': 1,
            'failed_board_ids': ['THS_I_881155.TI'], 'missing_board_ids': ['THS_I_881155.TI'],
            'current_only': True, 'historical_membership_verified': False, 'request_count': 6,
            'catalog_provenance': {'source_response': 'BULK_PROOF_MUST_STAY_SERVER_SIDE'},
            'snapshot_refs': {'THS_I_881155.TI': 'b' * 64}, 'failures': [{'board_id': 'THS_I_881155.TI', 'error': 'source incomplete'}]}
    monkeypatch.setattr(refresh_board_knowledge, 'refresh_knowledge', refresh)
    app = create_board_app(db_path=tmp_path / 'no-write.db')
    with TestClient(app) as client:
        launched = client.post('/api/boards/refresh', json={'kinds': ['industry'], 'resume': 'a' * 32, 'limit_budget': 50})
        assert launched.status_code == 202
        job = _wait_job(client, launched.json()['job_id'])
        assert job['status'] == outcome and job['result']['status'] == outcome
        assert job['result']['failed_board_count'] == 1 and job['progress']['completed'] == 1
        assert 'BULK_PROOF_MUST_STAY_SERVER_SIDE' not in str(job)
        assert 'catalog_provenance' not in job['result'] and 'snapshot_refs' not in job['result']
        assert calls[0]['resume'] == 'a' * 32 and calls[0]['limit_budget'] == 50
    assert app.state.board_job_lock.acquire(blocking=False)
    app.state.board_job_lock.release()
    assert not (tmp_path / 'no-write.db').exists()


def test_refresh_and_feature_jobs_share_one_lock(tmp_path, monkeypatch):
    import threading
    from my_strategy.scripts import refresh_board_knowledge
    started, release = threading.Event(), threading.Event()
    def refresh(**kwargs):
        started.set(); assert release.wait(5)
        return {'status': 'succeeded', 'run_id': 'a' * 32}
    monkeypatch.setattr(refresh_board_knowledge, 'refresh_knowledge', refresh)
    with TestClient(create_board_app(db_path=tmp_path / 'kb.db')) as client:
        job = client.post('/api/boards/refresh', json={}).json()
        try:
            assert started.wait(2)
            assert client.post('/api/boards/features/build', json={}).status_code == 409
            assert client.post('/api/boards/refresh', json={}).status_code == 409
        finally: release.set()
        assert _wait_job(client, job['job_id'])['status'] == 'succeeded'


def test_successful_resume_does_not_present_resolved_failure_as_current_error(tmp_path, monkeypatch):
    from my_strategy.scripts import refresh_board_knowledge
    monkeypatch.setattr(refresh_board_knowledge, 'refresh_knowledge', lambda **_: {
        'status': 'succeeded', 'requested_boards': 1, 'completed_boards': 1,
        'failed_board_ids': [], 'missing_board_ids': [],
        'failures': [{'board_id': 'THS_I_881155.TI', 'error': 'previous budget interruption'}]})
    with TestClient(create_board_app(db_path=tmp_path / 'kb.db')) as client:
        launched = client.post('/api/boards/refresh', json={}).json()
        result = _wait_job(client, launched['job_id'])
        assert result['status'] == 'succeeded' and 'error' not in result and 'error' not in result['result']
        assert result['result']['failed_board_count'] == result['result']['missing_board_count'] == 0


def test_feature_build_requires_real_verified_calendar_and_creates_no_fake_state(tmp_path, monkeypatch):
    from my_strategy.services import czsc_trend_support, board_features
    def calendar(_): raise ValueError('verified calendar unavailable')
    def forbidden(*args, **kwargs): raise AssertionError('builder cannot run without calendar')
    monkeypatch.setattr(czsc_trend_support, '_calendar', calendar)
    monkeypatch.setattr(board_features, 'build_board_features', forbidden)
    path = tmp_path / 'missing.db'
    with TestClient(create_board_app(db_path=path)) as client:
        assert client.post('/api/boards/features/build', json={'mode': 'causal'}).status_code == 422
        job = client.post('/api/boards/features/build', json={}).json()
        result = _wait_job(client, job['job_id'])
        assert result['status'] == 'failed' and result['error'] == 'verified calendar unavailable'
    assert not path.exists()


def test_feature_build_uses_latest_qualified_local_price_and_returns_only_summary(tmp_path, monkeypatch):
    import sqlite3
    from my_strategy.services import czsc_trend_support, board_features
    raw = tmp_path / 'raw.db'
    with sqlite3.connect(raw) as connection:
        connection.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,source TEXT,has_trade_price INTEGER)')
        connection.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?)', [
            ('000001.SZ', '2026-10-12', 'tushare', 1), ('000001.SZ', '2026-10-13', 'unverified_fixture', 1),
            ('000001.SZ', '2026-10-14', 'tushare', 0), ('920001.BJ', '2026-10-15', 'tushare', 1)])
    monkeypatch.setattr(czsc_trend_support, '_calendar', lambda _: (['2026-10-12'], 'verified-fixture'))
    calls = []
    def builder(store, **kwargs):
        calls.append((store, kwargs))
        return {'mode': 'current_reference', 'price_date': kwargs['price_date'], 'training_eligible': False,
            'qualification': 'pending', 'cache_key': 'a' * 64, 'schema_hash': 'b' * 64,
            'boards': [{'member_blob': 'DO_NOT_SEND_FULL_DATA'}] * 4, 'stocks': [{'member_blob': 'DO_NOT_SEND_FULL_DATA'}] * 3,
            'cache_identity': {'source_blob': 'DO_NOT_SEND_FULL_DATA'}}
    monkeypatch.setattr(board_features, 'build_board_features', builder)
    with TestClient(create_board_app(db_path=tmp_path / 'kb.db', raw_db_path=raw)) as client:
        launched = client.post('/api/boards/features/build', json={})
        assert launched.status_code == 202
        job = _wait_job(client, launched.json()['job_id'])
        assert job['status'] == 'completed'
        assert job['result']['price_date'] == '2026-10-12' and job['result']['board_count'] == 4
        assert job['result']['stock_count'] == 3 and job['result']['training_eligible'] is False
        assert 'DO_NOT_SEND_FULL_DATA' not in str(job)
        assert calls[0][1]['mode'] == 'current_reference' and calls[0][1]['calendar_dates'] == ['2026-10-12']
        assert calls[0][0].readonly is False


def test_standalone_nonloopback_needs_connection_token_without_launching_server(tmp_path, monkeypatch):
    from my_strategy.web_dashboard.scripts import serve_board_knowledge
    monkeypatch.delenv('KHQUANT_API_TOKEN', raising=False)
    with pytest.raises(ValueError, match='KHQUANT_API_TOKEN'):
        serve_board_knowledge.main(['--host', '0.0.0.0', '--db', str(tmp_path / 'kb.db')])
    assert not (tmp_path / 'kb.db').exists()


def test_connection_token_protects_reads_without_exposing_fuyao_key(tmp_path, monkeypatch):
    monkeypatch.setenv('KHQUANT_API_TOKEN', 'test-local-connection')
    with TestClient(create_board_app(db_path=tmp_path / 'kb.db')) as client:
        assert client.get('/api/boards/status').status_code == 401
        value = client.get('/api/boards/status', headers={'authorization': 'Bearer test-local-connection'})
        assert value.status_code == 200
        assert 'FUYAO_API_KEY' not in value.text and 'test-local-connection' not in value.text
