"""Published capital charts keep missingness and compare identical populations."""
import hashlib
import sqlite3

import pytest

from my_strategy.tests.test_board_features import market, build
from my_strategy.tests.test_current_dashboard import dashboard
from my_strategy.tests.test_board_workbench import isolate


def test_display_history_is_current_membership_and_does_not_change_training_inputs(market):
    before=hashlib.sha256(market['path'].read_bytes()).hexdigest()
    causal=build(market)
    assert all('capital_visualization' not in row for row in causal['boards'])
    current=build(market,mode='current_reference')
    row=current['boards'][0]; visual=row['capital_visualization']
    assert visual['training_eligible'] is False and len(visual['history'])==21
    assert visual['snapshot_hash']==row['snapshot_hash']
    assert visual['history'][-1]['amount']==row['board_amount']==7000
    assert visual['history'][-1]['market_amount']==10000
    assert visual['history'][-1]['market_share']==pytest.approx(.7)
    assert visual['amount_change_1']==0 and visual['amount_change_ratio_1']==0
    assert visual['prior20_mean_amount']==7000
    assert hashlib.sha256(market['path'].read_bytes()).hexdigest()==before


def test_amount_change_uses_only_same_verified_members_not_coverage_drift(market):
    with sqlite3.connect(market['path']) as conn:
        conn.execute('UPDATE stock_daily_normalized SET source=? WHERE stock=? AND date=?',
            ('unknown',market['symbols'][0],market['dates'][-2]))
    row=build(market,mode='current_reference')['boards'][0];visual=row['capital_visualization']
    assert visual['history'][-2]['amount']==6000 and visual['history'][-1]['amount']==7000
    assert visual['comparable_members']==6 and visual['comparable_coverage']==pytest.approx(6/7)
    assert visual['amount_change_1']==0  # The apparent +1000 is restored coverage, not more trading.
    with sqlite3.connect(market['path']) as conn:
        conn.execute('UPDATE stock_daily_normalized SET amount=? WHERE stock=? AND date=?',
            (1600,market['symbols'][1],market['dates'][-1]))
    visual=build(market,mode='current_reference')['boards'][0]['capital_visualization']
    assert visual['amount_change_1']==600 and visual['amount_change_ratio_1']==pytest.approx(.1)


def test_insufficient_comparable_coverage_preserves_null_history_and_delta(market):
    with sqlite3.connect(market['path']) as conn:
        for symbol in market['symbols'][:2]:
            conn.execute('UPDATE stock_daily_normalized SET has_trade_price=0 WHERE stock=? AND date=?',
                (symbol,market['dates'][-2]))
    visual=build(market,mode='current_reference')['boards'][0]['capital_visualization']
    assert visual['history'][-2]['amount'] is None
    assert visual['history'][-1]['amount']==7000
    assert visual['amount_change_1'] is None and visual['amount_change_ratio_1'] is None
    assert visual['comparable_coverage']==pytest.approx(5/7)
    assert visual['reason_codes']==['comparable_amount_population_unavailable']


def test_visual_api_reads_sha_verified_published_rows_without_rebuild(dashboard,market,tmp_path,monkeypatch):
    from my_strategy.storage.board_fluent import BoardFluentStore
    from my_strategy.services import board_features
    client,_=dashboard;isolate(client,tmp_path)
    store=BoardFluentStore(tmp_path/'fluent.db')
    saved=store.save_market(build(market,mode='current_reference'))
    generation=saved['generation_id'] if isinstance(saved,dict) else saved
    def forbidden(*args,**kwargs):raise AssertionError('GET must not read or calculate prices')
    monkeypatch.setattr(board_features,'_read_prices',forbidden)
    monkeypatch.setattr(board_features,'build_board_features',forbidden)
    response=client.get('/api/boards/capital-visualization',params={'generation_id':generation,'kind':'industry'})
    assert response.status_code==200;data=response.json()
    assert data['generation_id']==generation and data['board_count']==1
    assert data['market_amount']==10000 and data['needs_rebuild'] is False
    assert data['rows'][0]['capital_visualization']['history'][-1]['amount']==7000
    assert data['increases']==[] and data['decreases']==[]
    assert data['training_eligible'] is False and data['overlapping_boards'] is True
    assert client.get('/api/boards/capital-visualization?kind=bad').status_code==422
    assert client.get('/api/boards/capital-visualization?limit=500').status_code==422


def test_missing_visual_storage_is_readonly_and_empty(dashboard,tmp_path):
    client,_=dashboard;isolate(client,tmp_path)
    result=client.get('/api/boards/capital-visualization').json()
    assert result['rows']==[] and result['generation_id'] is None
    assert result['market_amount'] is None and result['prepared_board_count']==0
    assert not (tmp_path/'fluent.db').exists()
