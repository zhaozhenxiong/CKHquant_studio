"""Verify v1.0.3 installed views against isolated synthetic, immutable facts.

No model weights, market source, existing account, GPU inference or business task
is used. Public API calls below are GET requests to a private ASGI test client.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing
import hashlib
from html.parser import HTMLParser
from importlib.resources import files
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def require(condition, reason):
    if not condition:
        raise RuntimeError('Installed v1.0.3 follow-up probe: ' + reason)


def navigation_probe():
    root = files('my_strategy') / 'web_dashboard/static'
    html = (root / 'index.html').read_text(encoding='utf-8')

    class Elements(HTMLParser):
        def __init__(self):
            super().__init__(); self.stack = []; self.controls = {}; self.identities = set()

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs); name = attrs.get('id')
            if name:
                require(name not in self.identities, 'duplicate identity ' + name)
                self.identities.add(name)
                self.controls[name] = list(self.stack)
            if tag not in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}:
                self.stack.append((tag, name, attrs.get('class', '')))

        def handle_endtag(self, tag):
            for index in range(len(self.stack)-1, -1, -1):
                if self.stack[index][0] == tag:
                    del self.stack[index:]; break

    parsed = Elements(); parsed.feed(html)
    require(all(name in parsed.identities for name in (
        'analysis-chart-content', 'analysis-account-view', 'analysis-account-chart',
        'analysis-account-result', 'analysis-account-strategy', 'analysis-forecast-view',
        'market-backtest-form', 'scan-opportunities', 'scan-backtest-panel', 'board-refresh-workers')),
        'new chart or market controls missing')
    removed = {'marker-details', 'chart-range-summary', 'replay-evidence-card',
        'event-context', 'analysis-prediction-history-state'}
    require(not removed.intersection(parsed.identities), 'removed explanatory controls reappeared')
    require('class="time-window"' not in html and 'method-transfer-note' not in html, 'removed toolbar or transfer note reappeared')
    for name in ('analysis-mode', 'analysis-device'):
        require(any('analysis-method-device' in item[2].split() for item in parsed.controls[name]),
            'device and analysis method are no longer parallel controls')
    panel = (root / 'js/board_panel.js').read_text(encoding='utf-8')
    require("[['90','近90个交易日'],['50','近50个交易日'],['20','近20个交易日']]" in panel
        and "$('capital-window').value='20'" in panel, 'capital windows/default changed')
    require('/capital-history/' in panel, 'long capital windows are not backed by the history endpoint')
    for name in ('star.svg', 'star-fill.svg', 'chevron-up.svg', 'chevron-down.svg', 'LICENSE'):
        require((root / 'vendor/bootstrap-icons' / name).is_file(), 'market icon/license missing')
    return dict(status='passed', existing_account_chart=True, parallel_method_device=True,
        requested_explanations_removed=True, market_workspace=True,
        capital_windows=[90, 50, 20], capital_default=20, icon_license_present=True)


def run_probe():
    import pandas as pd
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from my_strategy.core.run_context import stable_hash
    from my_strategy.services import board_capital_history as history
    from my_strategy.services.board_calculation_scope import calculation_membership, CALCULATION_SCOPE
    from my_strategy.services.board_fluent import _witness
    from my_strategy.storage.research_store import ResearchStore
    from my_strategy.web_dashboard.light_api import router

    navigation = navigation_probe()
    with tempfile.TemporaryDirectory(prefix='synthetic-v103-followup-') as directory:
        root = Path(directory); raw = root / 'raw.db'
        dates = pd.bdate_range('2024-01-02', periods=91).strftime('%Y-%m-%d').tolist()
        symbols = [f'00000{i}.SZ' for i in range(1, 8)]
        with closing(sqlite3.connect(raw)) as connection, connection:
            connection.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER,PRIMARY KEY(stock,date))')
            connection.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                [(symbol, day, 10., 11., 9., 10., 100., 3000. if symbol == '600010.SH' else 1000., 'tushare', 1)
                    for symbol in [*symbols, '600010.SH'] for day in dates])
        snapshot = dict(board_id='THS_I_111111', snapshot_hash='a'*64,
            members=[*symbols, '830001.BJ'], complete=True, expected_count=8)
        projection = calculation_membership(snapshot)
        board = dict(board_id=snapshot['board_id'], generation_id='synthetic-generation',
            price_date=dates[-1], membership_date=dates[-1], snapshot_hash=snapshot['snapshot_hash'],
            calculation_scope=CALCULATION_SCOPE, calculation_membership=projection)
        generation = dict(generation_id=board['generation_id'], price_date=dates[-1],
            calculation_scope=CALCULATION_SCOPE, native_identity={'calendar_hash': stable_hash(dates)},
            publication_inputs=dict(source_witness={'raw': _witness(raw)},
                projection_hashes={board['board_id']: projection['projection_hash']}))
        board_before, generation_before = deepcopy(board), deepcopy(generation)
        raw_before = hashlib.sha256(raw.read_bytes()).hexdigest()
        history.clear_cache()
        ninety = history.capital_history(generation=generation, board=board,
            raw_db_path=raw, calendar_dates=dates, window=90)
        require(len(ninety['history']) == 90 and ninety['actual_points'] == 90
            and ninety['generation_id'] == board['generation_id']
            and ninety['projection_hash'] == projection['projection_hash'], '90-day history or fixed identity incomplete')
        require(all(row['amount'] == 7000. and row['market_amount'] == 10000.
            and row['market_share'] == .7 and row['expected_members'] == 7
            for row in ninety['history']), 'capital formula or scope changed')
        def forbidden(*args, **kwargs):
            raise RuntimeError('No model, task, raw reload or storage write is allowed during this read probe')
        with patch.object(history, '_read_prices', forbidden):
            for window in (50, 20):
                actual = history.capital_history(generation=generation, board=board,
                    raw_db_path=raw, calendar_dates=dates, window=window)
                require(actual['cache_hit'] is True and actual['history'] == ninety['history'][-window:],
                    'cached history reread or changed facts')
        try:
            history.capital_history(generation=generation, board=board, raw_db_path=raw,
                calendar_dates=dates, window=10)
        except ValueError:
            pass
        else:
            raise RuntimeError('Invalid old 10-day window was accepted')
        require(hashlib.sha256(raw.read_bytes()).hexdigest() == raw_before, 'capital reads changed raw data')
        require(board == board_before and generation == generation_before, 'capital reads changed frozen publication facts')
        history.clear_cache()

        symbol = '000001.SZ'; days = dates[-4:]
        identity = dict(kind='synthetic-v103-account', sources=[dict(symbol=symbol, price_basis='unadjusted')],
            start=days[0], end=days[-1], configs={'structure_kronos': {}}, prediction={'samples': 8, 'seed': 42})
        with ResearchStore(root/'research', clock=lambda: 100., ttl_seconds=1000) as store:
            result_id = store.cache_key(identity, 'simulations')
            trades = [dict(date=day, symbol=symbol, action=action, price=price,
                shares=100, fee=5.01, ledger_version=3, order_id=str(index), signal_date=days[0])
                for index, (day, action, price) in enumerate(zip(days[1:], ['BUY', 'REDUCE', 'SELL'], [10.005, 10.995, 10.995]))]
            original = dict(result_id=result_id, symbols=[symbol], start=days[0], end=days[-1], status='completed', saved=False,
                scheduler={'complete_through': days[-1]}, identity=identity,
                chart_bars={symbol: [dict(date=day, open=10., high=12., low=9., close=11.) for day in days]},
                comparisons={'structure_kronos': dict(lineage=dict(account_scope='shared_cash_account',
                    execution='next_verified_open_with_declared_slippage'), trades=trades)})
            store.put(identity, original, namespace='simulations')
            def result(key):
                value = store.get_by_key(key, touch=False)
                if value is None:
                    raise KeyError(key)
                return value
            light = SimpleNamespace(store=store, result=result, prediction=forbidden, backtest=forbidden,
                submit=forbidden, source=forbidden, train_score=forbidden, enable_account=forbidden)
            app = FastAPI(); app.state.light = light; app.include_router(router)
            before = '\n'.join(store.connection.iterdump()); changes = store.connection.total_changes
            with patch.object(store, 'put', forbidden), patch.object(store, 'pin', forbidden), patch.object(store, 'set_setting', forbidden):
                with TestClient(app) as client:
                    listed = client.get('/api/light/account-charts', params={'symbol': symbol, 'as_of': days[-1]})
                    require(listed.status_code == 200 and len(listed.json()['results']) == 1, 'existing account metadata not listed')
                    params = dict(symbol=symbol, strategy='structure_kronos', as_of=days[-1])
                    response = client.get('/api/light/account-charts/'+result_id, params=params)
                    require(response.status_code == 200, 'existing account chart route failed')
                    actual = response.json()
                    for expected, actual_trade in zip(trades, actual['trades']):
                        require(all(actual_trade[key] == value for key, value in expected.items()), 'original Ledger facts changed')
                    require(actual['counts'] == dict(BUY=1, SELL=2, ADD=0, REDUCE=1)
                        and len(actual['trades']) == 3 and actual['semantics'] == 'broker_ledger_fills', 'REDUCE was lost or relabelled')
                    require(actual['prediction_started'] is False and actual['account_mutated'] is False
                        and actual['current_method_qualification_changed'] is False, 'read promoted qualification or submitted work')
                    earlier = client.get('/api/light/account-charts/'+result_id, params={**params, 'as_of': days[1]}).json()
                    require(len(earlier['bars']) == 2 and len(earlier['trades']) == 1, 'historical cutoff leaked future candles or trades')
                    require(client.get('/api/light/account-charts/'+'f'*64, params=params).status_code == 404,
                        'missing account did not remain unavailable')
            require(before == '\n'.join(store.connection.iterdump()) and store.connection.total_changes == changes
                and result(result_id) == original, 'GET changed synthetic account or store facts')
    return dict(schema='followup_views_installation_v1', status='passed', fixture='synthetic_installation_only',
        production_state_touched=False, inference_calls=0, work_submitted=False, navigation=navigation,
        capital_history=dict(status='passed', windows=[90, 50, 20], exact_amount_and_scope=True,
            fixed_generation=True, cached_slices_no_reload=True, raw_unchanged=True, invalid_window_rejected=True),
        account_chart=dict(status='passed', http_gets=4, actual_fill_actions=['BUY', 'REDUCE', 'SELL'],
            exact_price_fee_order_preserved=True, cutoff_preserved=True, missing_unavailable=True,
            qualification_preserved=True, readonly_store_unchanged=True))


def main():
    with tempfile.TemporaryDirectory(prefix='khquant-followup-import-') as directory:
        previous = Path.cwd(); state = Path(directory).resolve()
        os.environ.update(KHQUANT_STATE_ROOT=str(state), KHQUANT_DATA_ROOT=str(state/'data'),
            KHQUANT_ARTIFACT_ROOT=str(state/'artifacts'), KHQUANT_RAW_DB=str(state/'raw.db'),
            PYTHON_DOTENV_DISABLED='1')
        os.chdir(state)
        try:
            print(json.dumps(run_probe(), ensure_ascii=False, allow_nan=False))
        finally:
            os.chdir(previous)


if __name__ == '__main__':
    main()
