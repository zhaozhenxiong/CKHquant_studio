"""Compact ranking tables are projections of the complete verified API."""
import sqlite3

import pytest

from my_strategy.tests.test_current_dashboard import dashboard
from my_strategy.tests.test_board_features import market
from my_strategy.tests.test_board_scope_api import scoped, _no_math
from my_strategy.tests.test_board_fluent import native_result


@pytest.mark.parametrize('full', [False, True])
def test_compact_current_partial_and_full_preserve_all_metadata_and_visible_values(scoped, market, monkeypatch, full):
    store, client = scoped['store'], scoped['client']
    generation = store.save_market(scoped['new'])
    if full:
        native, boards = native_result(market, scoped['new']['boards'])
        store.attach_native(generation, native, boards)
    _no_math(monkeypatch)
    params = {'kind': 'industry', 'ranking_type': 'capital'}
    complete = client.get('/api/boards/rankings', params=params).json()
    assert client.get('/api/boards/rankings', params={**params, 'compact': False}).json() == complete
    compact = client.get('/api/boards/rankings', params={**params, 'compact': True}).json()
    assert {k: v for k, v in compact.items() if k != 'rankings'} == {k: v for k, v in complete.items() if k != 'rankings'}
    assert compact['generation_id'] == generation
    assert compact['training_eligible'] is False
    assert len(compact['rankings']) == len(complete['rankings']) > 0
    for small, whole in zip(compact['rankings'], complete['rankings']):
        assert all(whole[key] == value for key, value in small.items())
        assert small['generation_id'] == generation and small['price_date'] == complete['price_date']
        assert small['kind'] == 'industry' and small['calculation_scope'] == 'sh_sz_a_v1'
        assert small['coverage'] == whole['coverage']
        assert 'calculation_membership' in whole and 'calculation_membership' not in small
        assert 'capital_visualization' not in small and 'native_structure' not in small
        assert set(small) == {'rank', 'board_id', 'name', 'kind', 'calculation_scope', 'activity_score',
            'board_return_1', 'board_amount', 'board_market_amount_share', 'board_relative_amount20',
            'effective_members', 'expected_members', 'coverage', 'generation_id', 'price_date'}
    assert client.get('/api/boards/rankings', params=params).json() == complete
    detail = client.get('/api/boards/detail/'+compact['rankings'][0]['board_id'], params={'generation_id': generation}).json()
    assert detail['features']['native_component_complete'] is full
    assert detail['calculation_membership'] == scoped['new']['boards'][0]['calculation_membership']


def test_compact_history_preserves_original_scope_and_pending_current(scoped, monkeypatch):
    _no_math(monkeypatch)
    client = scoped['client']
    compact = client.get('/api/boards/rankings?compact=true').json()
    assert compact == client.get('/api/boards/rankings').json()
    assert compact['availability'] == 'pending' and compact['rankings'] == []
    params = {'generation_id': scoped['old_id'], 'compact': True}
    historical = client.get('/api/boards/rankings', params=params).json()
    whole = client.get('/api/boards/rankings', params={'generation_id': scoped['old_id']}).json()
    assert historical['generation_id'] == scoped['old_id'] and historical['historical_view'] is True
    assert historical['calculation_scope'] is None
    assert historical['rankings'][0]['board_amount'] == whole['rankings'][0]['board_amount'] == 987654321.
    assert historical['rankings'][0]['expected_members'] == len(scoped['members'])
    assert historical['rankings'][0].get('calculation_scope') is None


@pytest.mark.parametrize('compact', [False, True])
def test_missing_explicit_history_still_rejects_without_fallback(scoped, compact):
    scoped['store'].save_market(scoped['new'])
    response = scoped['client'].get('/api/boards/rankings', params={'generation_id': 'missing', 'compact': compact})
    assert response.status_code == 404


@pytest.mark.parametrize('damage', ['ranking_sha', 'ranking_index', 'board_numeric', 'native_sha'])
def test_compact_runs_the_same_ranking_board_and_native_authentication(scoped, market, damage):
    store, client = scoped['store'], scoped['client']
    generation = store.save_market(scoped['new'])
    native, boards = native_result(market, scoped['new']['boards'])
    store.attach_native(generation, native, boards)
    with sqlite3.connect(store.path) as connection:
        if damage.startswith('ranking'):
            connection.execute('DROP TRIGGER immutable_board_rankings_update')
            if damage == 'ranking_sha':
                connection.execute('UPDATE board_rankings SET payload_sha=?', ('0'*64,))
            else:
                connection.execute('UPDATE board_rankings SET score=score+1')
        elif damage == 'board_numeric':
            connection.execute('DROP TRIGGER immutable_board_daily_features_update')
            connection.execute('UPDATE board_daily_features SET board_amount=board_amount+1')
        else:
            connection.execute('UPDATE feature_generations SET native_identity_sha=? WHERE generation_id=?', ('0'*64, generation))
    for compact in (False, True):
        params = {'generation_id': generation, 'compact': compact}
        if damage.startswith('ranking'):
            response = client.get('/api/boards/rankings', params=params)
            assert response.status_code == 400
            assert 'SHA drift' in response.json()['detail'] or 'ranking index drift' in response.json()['detail']
        else:
            assert client.get('/api/boards/rankings', params=params).status_code == 409
