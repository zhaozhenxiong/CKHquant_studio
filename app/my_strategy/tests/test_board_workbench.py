from pathlib import Path
import time

from my_strategy.tests.test_current_dashboard import dashboard
from my_strategy.tests.test_board_features import market, build


def isolate(client, tmp_path):
    client.app.state.board_db_path = tmp_path / 'kb.db'
    client.app.state.board_raw_db_path = tmp_path / 'raw.db'
    client.app.state.board_fluent_db_path = tmp_path / 'fluent.db'


def test_integrated_shell_and_readonly_missing_storage(dashboard, tmp_path):
    client, _ = dashboard
    isolate(client, tmp_path)
    page = client.get('/').text
    assert 'id="scan-board-panel"' in page and 'board_panel.js' in page
    redirected = client.get('/boards', follow_redirects=False)
    assert redirected.status_code == 307 and redirected.headers['location'] == '/?board=1#scan'
    result = client.get('/api/boards/rankings').json()
    assert result['rankings'] == [] and result['training_eligible'] is False
    assert client.get('/api/boards/status').json()['fluent']['available'] is False
    assert client.get('/api/boards/stocks/000001.SZ').json()['features'] is None
    assert client.get('/api/boards/rankings?kind=bad').status_code == 422
    assert client.get('/api/boards/stocks/invalid').status_code == 422
    assert not (tmp_path / 'fluent.db').exists() and not (tmp_path / 'kb.db').exists()


def test_top30_is_generation_pinned_and_never_calls_math(dashboard, tmp_path, market, monkeypatch):
    from my_strategy.storage.board_fluent import BoardFluentStore
    from my_strategy.services import board_features
    client, _ = dashboard
    isolate(client, tmp_path)
    store = BoardFluentStore(tmp_path / 'fluent.db')
    generation = store.save_market(build(market, mode='current_reference'))
    generation_id = generation['generation_id'] if isinstance(generation, dict) else generation
    def forbidden(*args, **kwargs): raise AssertionError('GET cannot calculate or load a bulk feature cache')
    monkeypatch.setattr(board_features, 'build_board_features', forbidden)
    from my_strategy.storage.board_knowledge import BoardKnowledgeStore
    monkeypatch.setattr(BoardKnowledgeStore, 'latest_feature', forbidden)
    response = client.get('/api/boards/rankings', params={'kind':'concept','ranking_type':'capital','limit':100}).json()
    assert response['generation_id'] == generation_id
    assert response['rankings'] == store.rankings('concept', 'capital')
    row = response['rankings'][0]
    detail = client.get('/api/boards/detail/' + row['board_id'], params={'generation_id':generation_id}).json()
    assert detail['features']['board_amount'] == 7000
    assert detail['membership_matches_generation'] is True and detail['members'] == market['symbols']
    assert detail['source_membership_available'] is False and detail['source_members'] == []
    assert detail['features']['native_structure'] is None


def test_board_partial_result_survives_restart_and_retry_has_own_identity(dashboard, tmp_path, monkeypatch):
    from my_strategy.services import daily_feature_sync
    client, manager = dashboard
    isolate(client, tmp_path)
    # Global test data roots already isolate RunContext; do not use production inputs.
    def sync(**kwargs):
        kwargs['progress']({'stage':'native','current':1,'total':2,'sync_id':'persisted-sync'})
        return {'status':'partial','sync_id':'persisted-sync','market_published':True,'full_published':False,
                'price_date':'2026-10-12','error':'native incomplete','training_started':False}
    monkeypatch.setattr(daily_feature_sync, 'sync_daily_features', sync)
    assert client.post('/api/boards/sync', json={}).status_code == 409
    launched = client.post('/api/boards/sync', json={'scope':'full'})
    assert launched.status_code == 202
    identifier = launched.json()['job_id']
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = client.get('/api/boards/jobs/' + identifier).json()
        if job['status'] not in {'running','queued'}: break
        time.sleep(.02)
    assert job['status'] == 'partial', job.get('error')
    assert job['result']['sync_id'] == 'persisted-sync' and job['result']['market_published'] is True
    assert job['progress_detail']['sync_id'] == 'persisted-sync'
    assert client.get('/api/boards/jobs').json()['jobs'][0]['job_id'] == identifier
    assert client.get('/api/tasks').json()['tasks'][0]['kind'] == 'board_sync'
    assert client.get('/api/tasks/' + identifier).json()['result']['sync_id'] == 'persisted-sync'
    assert client.get('/api/runs/' + job['run_id']).json()['kind'] == 'board_sync'
    assert client.get('/api/runs?kind=board_sync').json()['runs'][0]['run_id'] == job['run_id']
    # Stored result is read back from SQLite and ResultStore, not an in-memory jobs map.
    manager._cancel.clear()
    assert manager.get(identifier)['result']['full_published'] is False
    assert client.post('/api/boards/features/build', json={'mode':'causal'}).status_code == 422
    assert client.post('/api/boards/features/build', json={'symbols':['000001.SZ']}).status_code == 422


def test_board_writes_use_main_security_and_safe_dom(dashboard, tmp_path, monkeypatch):
    client, manager = dashboard
    isolate(client, tmp_path)
    assert client.post('/api/boards/sync', json={}, headers={'origin':'https://foreign.example'}).status_code == 403
    monkeypatch.setenv('KHQUANT_API_TOKEN','test-only')
    assert client.post('/api/boards/sync', json={}).status_code == 401
    assert manager.list() == []
    script = (Path(__file__).resolve().parents[1] / 'web_dashboard/static/js/board_panel.js').read_text()
    assert 'innerHTML' not in script and 'textContent' in script
    assert 'FUYAO_API_KEY' not in script and 'X-api-key' not in script
    assert "credentials:'same-origin'" in script
