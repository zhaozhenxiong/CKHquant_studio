"""Small status responses follow the same authenticated publication path."""
from copy import deepcopy
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from my_strategy.tests.test_current_dashboard import dashboard
from my_strategy.tests.test_board_features import market
from my_strategy.tests.test_board_scope_api import scoped, _no_math
from my_strategy.tests.test_board_workbench import isolate
from my_strategy.storage.board_fluent import canonical, digest
from my_strategy.web_dashboard.scripts.serve_board_knowledge import create_board_app


def without_generation_bodies(result):
    value = deepcopy(result)
    value['fluent'].pop('market')
    value['fluent'].pop('full')
    return value


@pytest.mark.parametrize('history', [False, True])
def test_default_full_and_compact_status_coexist_with_identical_verified_identity(scoped, monkeypatch, history):
    store, client = scoped['store'], scoped['client']
    new_id = store.save_market(scoped['new'])
    _no_math(monkeypatch)
    params = {'generation_id': scoped['old_id']} if history else {}
    complete = client.get('/api/boards/status', params=params).json()
    explicit_full = client.get('/api/boards/status', params={**params, 'compact': False}).json()
    compact = client.get('/api/boards/status', params={**params, 'compact': True}).json()
    assert complete == explicit_full
    assert without_generation_bodies(compact) == without_generation_bodies(complete)
    whole, small = complete['fluent']['market'], compact['fluent']['market']
    assert small['generation_id'] == (scoped['old_id'] if history else new_id)
    assert small['price_date'] == whole['price_date']
    assert small['membership_date'] == whole['membership_date']
    assert small['created_at'] == whole['created_at']
    assert small['board_count'] == whole['board_count']
    assert small['market_ready'] is True and small['full_ready'] is False
    assert small['market_published'] == whole['market_published']
    assert small['full_published'] == whole['full_published']
    assert small['training_eligible'] is False
    assert all(whole[key] == value for key, value in small.items())
    assert 'publication_inputs' in whole and 'publication_inputs' not in small
    assert 'fluent_contract' in whole and 'fluent_contract' not in small
    assert 'calculation_membership' not in small and 'native_identity' not in small
    if history:
        assert compact['fluent']['historical_view'] is True
        assert compact['fluent']['scope_label'] == '历史原范围'
    else:
        assert 'calculation_membership' in whole
        assert small['calculation_scope'] == 'sh_sz_a_v1'
    assert compact['fluent']['full'] is None
    assert len(json.dumps(compact)) < len(json.dumps(complete))
    # A compact GET cannot mutate the stored publication or change default API output.
    assert client.get('/api/boards/status', params=params).json() == complete


def test_no_current_scope_keeps_legacy_history_selectable_in_both_response_forms(scoped, monkeypatch):
    _no_math(monkeypatch)
    client = scoped['client']
    complete = client.get('/api/boards/status').json()
    compact = client.get('/api/boards/status?compact=true').json()
    assert compact == complete
    assert compact['fluent']['market'] is compact['fluent']['full'] is None
    assert compact['fluent']['availability'] == 'pending'
    assert compact['fluent']['needs_rebuild'] is True
    assert compact['generations'][0]['generation_id'] == scoped['old_id']


def test_missing_databases_are_not_created_for_compact_status(dashboard, tmp_path):
    client, _ = dashboard
    isolate(client, tmp_path)
    compact = client.get('/api/boards/status?compact=true').json()
    assert compact == client.get('/api/boards/status').json()
    assert compact['available'] is False and compact['fluent']['market'] is None
    assert not (tmp_path/'kb.db').exists() and not (tmp_path/'fluent.db').exists()


def test_standalone_status_remains_compatible_with_optional_compact(tmp_path):
    path = tmp_path/'missing.db'
    with TestClient(create_board_app(db_path=path)) as client:
        assert client.get('/api/boards/status?compact=true').json() == client.get('/api/boards/status').json()
    assert not path.exists()


@pytest.mark.parametrize('compact', [False, True])
def test_missing_selected_history_never_falls_back_with_compact(scoped, compact):
    scoped['store'].save_market(scoped['new'])
    response = scoped['client'].get('/api/boards/status', params={'generation_id': 'missing', 'compact': compact})
    assert response.status_code == 404


@pytest.mark.parametrize('corruption', ['public_sha', 'projection'])
def test_compact_cannot_bypass_publication_sha_or_scope_checks(scoped, monkeypatch, corruption):
    store, client = scoped['store'], scoped['client']
    generation = store.save_market(scoped['new'])
    with sqlite3.connect(store.path) as connection:
        if corruption == 'public_sha':
            connection.execute('UPDATE feature_generations SET public_sha=? WHERE generation_id=?', ('0'*64, generation))
        else:
            connection.execute('DROP TRIGGER immutable_board_daily_features_update')
            value = json.loads(connection.execute('SELECT payload_json FROM board_daily_features WHERE generation_id=?', (generation,)).fetchone()[0])
            value['calculation_membership']['eligible_members'].append('920001.BJ')
            connection.execute('UPDATE board_daily_features SET payload_json=?,payload_sha=? WHERE generation_id=?',
                (canonical(value), digest(value), generation))
    _no_math(monkeypatch)
    for compact in (False, True):
        response = client.get('/api/boards/status', params={'compact': compact})
        assert response.status_code == 200
        assert response.json()['fluent']['market'] is None
        assert response.json()['fluent']['availability'] == 'pending'
        explicit = client.get('/api/boards/status', params={'generation_id': generation, 'compact': compact})
        assert explicit.status_code == 409


def test_compact_full_publication_retains_status_flags_and_identity(monkeypatch, scoped):
    # Scope, freshness and catalog validation still run before projection. Use
    # an authenticated real market as the stand-in for a full-ready publication
    # to exercise its response shape without manufacturing native structures.
    from my_strategy.services import board_fluent
    store, client = scoped['store'], scoped['client']
    generation_id = store.save_market(scoped['new'])
    generation = store.verify_calculation_scope(generation_id)
    generation.update(full_ready=True, full_published=True,
        native_identity={'large_proof': ['original-source']*100})
    calls = []
    def freshness(**kwargs):
        calls.append(kwargs)
        return {'available': True, 'market': generation, 'full': generation,
            'stale': True, 'stale_reasons': ['membership_source_changed'], 'native_stale': True}
    monkeypatch.setattr(board_fluent, 'freshness_status', freshness)
    complete = client.get('/api/boards/status').json()
    compact = client.get('/api/boards/status?compact=true').json()
    assert len(calls) == 2
    assert without_generation_bodies(compact) == without_generation_bodies(complete)
    for key in ('market', 'full'):
        assert compact['fluent'][key]['generation_id'] == generation_id
        assert compact['fluent'][key]['full_ready'] is True
        assert compact['fluent'][key]['full_published'] is True
        assert 'native_identity' not in compact['fluent'][key]
        assert 'native_identity' in complete['fluent'][key]
