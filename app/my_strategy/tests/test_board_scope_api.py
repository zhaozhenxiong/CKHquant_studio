"""The current board view excludes old scope without rewriting saved history."""
import copy
import gc
import hashlib
import json
import sqlite3

import pytest

from my_strategy.tests.test_current_dashboard import dashboard
from my_strategy.tests.test_board_features import market, build, MemoryStore
from my_strategy.tests.test_board_workbench import isolate
from my_strategy.tests.test_board_knowledge import snapshot
from my_strategy.storage.board_knowledge import BoardKnowledgeStore
from my_strategy.storage.board_fluent import BoardFluentStore, canonical, digest


@pytest.fixture
def scoped(dashboard, market, tmp_path):
    client, _ = dashboard
    isolate(client, tmp_path)
    client.app.state.board_raw_db_path = market['path']
    knowledge = BoardKnowledgeStore(tmp_path / 'kb.db')
    members = [*market['symbols'], '920001.BJ', '900001.SH']
    knowledge.import_snapshot(snapshot(members=members))
    original = knowledge.catalog(current=True)[0]
    market['store'] = MemoryStore([original])
    bundle = build(market, mode='current_reference')
    old = copy.deepcopy(bundle)
    old['cache_key'] = 'e' * 64
    old.pop('calculation_scope', None)
    old.pop('calculation_membership', None)
    for row in old['boards']:
        row.pop('calculation_scope', None)
        row.pop('calculation_membership', None)
        row['expected_members'] = len(members)
        row['coverage'] = 7 / 9
        row['board_amount'] = 987654321.
    store = BoardFluentStore(tmp_path / 'fluent.db')
    old_id = store.save_market(old)
    return dict(client=client, store=store, old_id=old_id, original=original,
        new=bundle, kb=tmp_path / 'kb.db', members=sorted(members))


def _no_math(monkeypatch):
    from my_strategy.services import board_features
    def forbidden(*args, **kwargs):
        raise AssertionError('GET must not read prices, calculate, or rebuild')
    monkeypatch.setattr(board_features, '_read_prices', forbidden)
    monkeypatch.setattr(board_features, 'build_board_features', forbidden)


def test_old_scope_is_history_only_and_read_paths_do_not_write(scoped, monkeypatch):
    _no_math(monkeypatch)
    client, store = scoped['client'], scoped['store']
    # Finalize fixture writers before comparing physical SQLite file bytes.
    # Otherwise GC can checkpoint their committed WAL during a read request.
    gc.collect()
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [scoped['kb'], store.path]}
    board = scoped['original']['board_id']
    for url in ['/status', '/rankings', '/capital-visualization', '/features/' + board,
                '/stocks/000001.SZ', '/detail/' + board]:
        response = client.get('/api/boards' + url)
        assert response.status_code == 200, response.text
        payload = response.json().get('fluent', response.json())
        assert payload['availability'] == 'pending'
        assert payload['needs_rebuild'] is True
        assert payload.get('features') is None
        assert payload.get('generation_id') is None
        assert not payload.get('rankings') and not payload.get('rows')
    catalog = client.get('/api/boards/generations').json()
    assert catalog['current_generation_id'] is None
    assert catalog['generations'][0]['scope_label'] == '历史原范围'
    response = client.get('/api/boards/detail/' + board, params={'generation_id': scoped['old_id']}).json()
    assert response['members'] == scoped['members']
    assert response['features']['board_amount'] == 987654321.
    assert response['features']['coverage'] == 7 / 9
    assert response['scope_label'] == '历史原范围' and response['historical_view'] is True
    assert response['calculation_membership'] is None
    status = client.get('/api/boards/status', params={'generation_id': scoped['old_id']}).json()['fluent']
    assert status['market']['generation_id'] == scoped['old_id'] and status['historical_view']
    ranking = client.get('/api/boards/rankings', params={'generation_id': scoped['old_id']}).json()
    assert ranking['rankings'][0]['board_amount'] == 987654321.
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before


def test_new_default_detail_uses_saved_projection_and_separates_source(scoped, monkeypatch):
    store, client = scoped['store'], scoped['client']
    new_id = store.save_market(scoped['new'])
    _no_math(monkeypatch)
    board = scoped['original']['board_id']
    result = client.get('/api/boards/detail/' + board).json()
    projection = scoped['new']['boards'][0]['calculation_membership']
    assert result['features']['generation_id'] == new_id
    assert result['calculation_scope'] == 'sh_sz_a_v1'
    assert result['members'] == projection['eligible_members']
    assert result['source_members'] == scoped['members']
    assert result['calculation_membership']['excluded_members'] == ['900001.SH', '920001.BJ']
    assert result['membership_matches_generation'] is True
    assert result['source_membership_available'] is True
    assert result['features']['coverage'] == 1
    ranking = client.get('/api/boards/rankings').json()
    assert ranking['generation_id'] == new_id and ranking['scope_label'] == '沪深 A 股'
    visual = client.get('/api/boards/capital-visualization').json()
    assert visual['generation_id'] == new_id and visual['rows'][0]['calculation_scope'] == 'sh_sz_a_v1'
    old = client.get('/api/boards/detail/' + board, params={'generation_id': scoped['old_id']}).json()
    assert old['members'] == scoped['members'] and old['features']['board_amount'] == 987654321.


def test_missing_or_invalid_selected_history_does_not_fall_back(scoped, monkeypatch):
    scoped['store'].save_market(scoped['new'])
    _no_math(monkeypatch)
    for path in ['/status', '/rankings', '/capital-visualization', '/detail/' + scoped['original']['board_id']]:
        response = scoped['client'].get('/api/boards' + path, params={'generation_id': 'missing'})
        assert response.status_code == 404


def test_tampered_projection_is_pending_by_default_and_rejected_explicitly(scoped, monkeypatch):
    store, client = scoped['store'], scoped['client']
    new_id = store.save_market(scoped['new'])
    with sqlite3.connect(store.path) as conn:
        # Simulate disk corruption beyond the production immutable-write trigger.
        conn.execute('DROP TRIGGER immutable_board_daily_features_update')
        value = json.loads(conn.execute('SELECT payload_json FROM board_daily_features WHERE generation_id=?', (new_id,)).fetchone()[0])
        value['calculation_membership']['eligible_members'].append('920001.BJ')
        conn.execute('UPDATE board_daily_features SET payload_json=?,payload_sha=? WHERE generation_id=?',
            (canonical(value), digest(value), new_id))
    _no_math(monkeypatch)
    for path in ['/status', '/rankings', '/capital-visualization', '/detail/' + scoped['original']['board_id']]:
        response = client.get('/api/boards' + path)
        assert response.status_code == 200
        result = response.json().get('fluent', response.json())
        assert result['availability'] == 'pending'
        assert result.get('generation_id') is None
    for path in ['/status', '/rankings', '/capital-visualization', '/detail/' + scoped['original']['board_id']]:
        response = client.get('/api/boards' + path, params={'generation_id': new_id})
        assert response.status_code == 409
