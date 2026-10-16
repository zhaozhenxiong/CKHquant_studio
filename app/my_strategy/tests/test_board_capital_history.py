"""Extended chart windows preserve fixed members, source checks and missingness."""
import copy
import hashlib
import sqlite3

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import board_capital_history as history
from my_strategy.services.board_fluent import _witness
from my_strategy.storage.board_fluent import BoardFluentStore
from my_strategy.tests.test_board_features import market, build
from my_strategy.tests.test_current_dashboard import dashboard
from my_strategy.tests.test_board_workbench import isolate


@pytest.fixture
def extended(market, tmp_path):
    original = market['dates']; before = pd.Timestamp(original[0])
    dates = pd.bdate_range(end=before-pd.Timedelta(days=1), periods=60).strftime('%Y-%m-%d').tolist()+original
    with sqlite3.connect(market['path']) as conn:
        rows = conn.execute('SELECT * FROM stock_daily_normalized WHERE date=?', (original[0],)).fetchall()
        for day in dates[:60]:
            conn.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                [(row[0], day, *row[2:]) for row in rows])
    market['dates'] = dates
    fluent = BoardFluentStore(tmp_path/'extended.db')
    saved = fluent.save_market(build(market, mode='current_reference'))
    generation_id = saved['generation_id'] if isinstance(saved, dict) else saved
    generation = fluent.get_generation(generation_id)
    generation['native_identity'] = {'calendar_hash': stable_hash(dates)}
    generation['publication_inputs']['source_witness'] = {'raw': _witness(market['path'])}
    row = fluent.boards('industry', generation_id)[0]
    history.clear_cache()
    return market, generation, row


@pytest.mark.parametrize('window', [90, 50, 20])
def test_actual_window_lengths_same_members_and_no_business_writes(extended, window):
    market, generation, row = extended
    before = hashlib.sha256(market['path'].read_bytes()).hexdigest()
    published = copy.deepcopy(row)
    result = history.capital_history(generation=generation, board=row, raw_db_path=market['path'],
        window=window, calendar_dates=market['dates'])
    assert result['requested_window'] == result['actual_points'] == len(result['history']) == window
    assert result['history'][0]['date'] == market['dates'][-window]
    assert result['history'][-1]['date'] == row['price_date']
    assert all(point['amount'] == 7000 and point['market_amount'] == 10000 for point in result['history'])
    assert all(point['market_share'] == .7 and point['effective_members'] == 7 for point in result['history'])
    assert result['projection_hash'] == row['calculation_membership']['projection_hash']
    assert result['training_eligible'] is False and result['mode'] == 'current_reference'
    assert row == published
    assert hashlib.sha256(market['path'].read_bytes()).hexdigest() == before


def test_one_90_session_read_serves_50_and_20_and_second_board_without_raw_reload(extended, monkeypatch):
    market, generation, row = extended
    first = history.capital_history(generation=generation, board=row, raw_db_path=market['path'], calendar_dates=market['dates'])
    def forbidden(*args, **kwargs):
        raise AssertionError('cached window must not load the price matrix again')
    monkeypatch.setattr(history, '_read_prices', forbidden)
    second = history.capital_history(generation=generation, board=row, raw_db_path=market['path'], window=50, calendar_dates=market['dates'])
    assert first['cache_hit'] is False and second['cache_hit'] is True
    assert second['history'] == first['history'][-50:]


def test_missing_old_coverage_is_a_null_not_zero_and_excluded_exchange_never_enters_market(extended):
    market, generation, row = extended
    missing = market['dates'][-40]
    with sqlite3.connect(market['path']) as conn:
        conn.execute('UPDATE stock_daily_normalized SET source=? WHERE date=? AND stock IN (?,?)',
            ('unknown', missing, *market['symbols'][:2]))
        sample = conn.execute('SELECT * FROM stock_daily_normalized LIMIT 1').fetchone()
        for day in market['dates']:
            conn.execute('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                ('920001.BJ', day, *sample[2:7], 1e12, *sample[8:]))
    generation['publication_inputs']['source_witness']['raw'] = _witness(market['path'])
    value = history.capital_history(generation=generation, board=row, raw_db_path=market['path'], calendar_dates=market['dates'])
    point = next(item for item in value['history'] if item['date'] == missing)
    assert point['amount'] is None and point['market_share'] is None
    assert point['effective_members'] == 5 and point['coverage'] == 5/7
    assert value['history'][-1]['market_amount'] == 10000


def test_source_or_calendar_or_projection_drift_never_relabels_new_data_as_original(extended):
    market, generation, row = extended
    for changed in ('raw', 'calendar', 'projection'):
        value, board = copy.deepcopy(generation), copy.deepcopy(row)
        if changed == 'raw':
            value['publication_inputs']['source_witness']['raw']['db']['size'] += 1
        elif changed == 'calendar':
            value['native_identity']['calendar_hash'] = 'changed'
        else:
            board['calculation_membership']['eligible_members'].pop()
        with pytest.raises(ValueError):
            history.capital_history(generation=value, board=board, raw_db_path=market['path'], calendar_dates=market['dates'])


def test_readonly_extended_api_validates_windows_and_does_not_republish(dashboard, extended, tmp_path, monkeypatch):
    client, _ = dashboard
    market, _, row = extended
    isolate(client, tmp_path)
    client.app.state.board_raw_db_path = market['path']
    client.app.state.board_fluent_db_path = tmp_path/'extended.db'
    monkeypatch.setattr(history, '_calendar', lambda: (market['dates'], 'fixture-calendar'))
    store = BoardFluentStore(tmp_path/'extended.db', readonly=True)
    generation = store.current_generation()['generation_id']
    source_before = hashlib.sha256((tmp_path/'extended.db').read_bytes()).hexdigest()
    for window in (90, 50, 20):
        response = client.get('/api/boards/capital-history/'+row['board_id'], params={'window': window, 'generation_id': generation})
        assert response.status_code == 200, response.text
        assert len(response.json()['history']) == window
        assert response.json()['generation_id'] == generation
    assert client.get('/api/boards/capital-history/'+row['board_id']+'?window=10').status_code == 422
    assert client.get('/api/boards/capital-history/missing').status_code == 404
    assert hashlib.sha256((tmp_path/'extended.db').read_bytes()).hexdigest() == source_before
