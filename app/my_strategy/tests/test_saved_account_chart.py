"""Existing execution views cannot infer, mutate accounts or repaint forecasts."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from my_strategy.services.saved_account_chart import account_chart, chart_options
from my_strategy.web_dashboard.light_api import router

ID = 'a' * 64
SYMBOL = '002156.SZ'


@pytest.fixture
def result():
    bars = [{'date': day, 'open': 10., 'high': 12., 'low': 9., 'close': 11.}
            for day in ('2026-10-13', '2026-10-14', '2026-10-15', '2026-10-16')]
    trades = [dict(date=day, symbol=SYMBOL, action=action, price=price, shares=100,
        fee=5.01, ledger_version=3, order_id=str(index), signal_date='2026-10-13')
        for index, (day, action, price) in enumerate([
            ('2026-10-14', 'BUY', 10.005), ('2026-10-15', 'REDUCE', 10.995),
            ('2026-10-16', 'SELL', 10.995)])]
    account = dict(lineage=dict(account_scope='shared_cash_account',
        execution='next_verified_open_with_declared_slippage'), trades=trades)
    return dict(result_id=ID, symbols=[SYMBOL], start='2026-10-13', end='2026-10-16',
        status='completed', saved=False, scheduler=dict(complete_through='2026-10-16'),
        chart_bars={SYMBOL: bars}, comparisons=dict(structure_kronos=account),
        identity=dict(sources=[dict(symbol=SYMBOL, price_basis='unadjusted')],
            prediction=dict(samples=8, seed=42)))


@pytest.fixture
def light(result):
    return SimpleNamespace(result=Mock(return_value=result),
        store=SimpleNamespace(list=Mock(return_value=[])), prediction=Mock(), backtest=Mock())


def test_saved_chart_preserves_real_reduce_price_fee_and_does_not_recompute(light, result):
    before = deepcopy(result)
    value = account_chart(light, ID, SYMBOL, 'structure_kronos', '2026-10-16')
    assert result == before
    assert [(r['action'], r['price'], r['fee'], r['order_id']) for r in value['trades']] == [
        ('BUY', 10.005, 5.01, '0'), ('REDUCE', 10.995, 5.01, '1'), ('SELL', 10.995, 5.01, '2')]
    assert value['counts'] == dict(BUY=1, SELL=2, ADD=0, REDUCE=1)
    assert value['semantics'] == 'broker_ledger_fills'
    assert not value['account_mutated'] and not value['prediction_started']
    assert not value['current_method_qualification_changed']
    light.prediction.assert_not_called()
    light.backtest.assert_not_called()


def test_historical_cutoff_never_shows_future_fill_or_future_candle(light):
    value = account_chart(light, ID, SYMBOL, 'structure_kronos', '2026-10-14')
    assert value['original_end'] == '2026-10-16'
    assert value['display_end'] == '2026-10-14'
    assert len(value['bars']) == 2
    assert len(value['trades']) == 1


def test_partial_risk_only_tail_is_not_relabelled_evaluated(light, result):
    result['status'] = 'partial'
    result['scheduler'] = dict(complete_through='2026-10-14', risk_through='2026-10-16')
    value = account_chart(light, ID, SYMBOL, 'structure_kronos', '2026-10-16')
    assert value['source_status'] == 'partial'
    assert value['display_end'] == '2026-10-14'
    assert len(value['trades']) == 1


@pytest.mark.parametrize('change', [
    lambda r: r.update(result_id='b'*64),
    lambda r: r.update(symbols=['002436.SZ']),
    lambda r: r['comparisons']['structure_kronos']['lineage'].update(execution='signal_close'),
    lambda r: r['identity']['sources'][0].update(price_basis='adjusted'),
    lambda r: r['chart_bars'][SYMBOL][0].update(close=100.),
    lambda r: r['chart_bars'][SYMBOL][0].update(date='2026-10-14'),
    lambda r: r['comparisons']['structure_kronos']['trades'][0].update(date='2026-10-12'),
    lambda r: r['comparisons']['structure_kronos']['trades'][0].update(price=float('nan')),
    lambda r: r['comparisons']['structure_kronos']['trades'][0].update(ledger_version=None),
    lambda r: r['comparisons']['structure_kronos']['trades'][0].update(signal_date='2026-10-16'),
])
def test_invalid_saved_evidence_is_not_drawn_as_fills(light, result, change):
    change(result)
    if result['comparisons']['structure_kronos']['trades'][0]['date'] == '2026-10-12':
        # A genuine earlier fill outside this explicitly saved range is simply
        # outside the projection; it is not moved to the first displayed bar.
        value = account_chart(light, ID, SYMBOL, 'structure_kronos', '2026-10-16')
        assert all(t['date'] != '2026-10-12' for t in value['trades'])
    else:
        with pytest.raises(ValueError):
            account_chart(light, ID, SYMBOL, 'structure_kronos', '2026-10-16')


def test_other_stock_fills_and_nonledger_orders_cannot_create_bs(light, result):
    account = result['comparisons']['structure_kronos']
    account['trades'] = [dict(account['trades'][0], symbol='002436.SZ')]
    account['orders'] = [dict(date='2026-10-14', symbol=SYMBOL, action='BUY', status='rejected')]
    value = account_chart(light, ID, SYMBOL, 'structure_kronos', '2026-10-16')
    assert value['trades'] == [] and value['counts']['BUY'] == 0


def test_options_only_read_bounded_metadata_and_preserve_saved_vs_preview(light):
    rows = [dict(cache_key=ID, created_at=1, pin_count=0, identity=dict(start='2026-10-13',
        end='2026-10-16', sources=[dict(symbol=SYMBOL)], configs=dict(structure_kronos={}))),
        dict(cache_key='b'*64, identity=dict(start='2027-01-01', end='2027-01-05',
             sources=[dict(symbol=SYMBOL)], configs=dict(structure_kronos={}))),
        dict(cache_key='c'*64, identity=dict(start='2026-10-13', end='2026-10-16',
             sources=[dict(symbol='002436.SZ')], configs=dict(structure_kronos={})))]
    light.store.list.return_value = rows
    value = chart_options(light, SYMBOL, '2026-10-16')
    assert [r['result_id'] for r in value['results']] == [ID]
    assert value['results'][0]['saved'] is False
    light.store.list.assert_called_once_with(namespace='simulations', limit=50)
    light.result.assert_not_called()


def test_http_read_only_routes_preserve_prices_and_return_missing_without_fallback(light):
    app = FastAPI(); app.state.light = light; app.include_router(router)
    with TestClient(app) as client:
        path = f'/api/light/account-charts/{ID}?symbol={SYMBOL}&strategy=structure_kronos&as_of=2026-10-16'
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()['trades'][1]['action'] == 'REDUCE'
        assert client.get(path.replace(ID, '../unknown')).status_code == 404
        light.result.side_effect = KeyError('expired')
        assert client.get(path).status_code == 404
    light.prediction.assert_not_called(); light.backtest.assert_not_called()
