"""Display-only prediction history: synthetic prices, no models or account work."""
from __future__ import annotations

from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import stat
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from my_strategy.adapters.czsc_adapter import load_bars
from my_strategy.services import kronos_prediction as kronos
from my_strategy.services import prediction_chart as chart
from my_strategy.storage.research_store import ResearchStore
from my_strategy.web_dashboard.light_api import router


def make_raw(path, count=280, symbol='002436.SZ', source='sina_unadjusted_v1', verified=1):
    days = pd.bdate_range('2024-01-02', periods=count).strftime('%Y-%m-%d').tolist()
    with closing(sqlite3.connect(path)) as db:
        db.execute('''CREATE TABLE stock_daily_normalized(
            stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,
            volume REAL,amount REAL,source TEXT,has_trade_price INTEGER)''')
        db.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                       [(symbol, day, 10+i/100, 11+i/100, 9+i/100, 10.5+i/100,
                         10000+i, 100000+i*10, source, verified) for i, day in enumerate(days)])
        db.commit()
    return days


def make_prediction(raw_db, *, lookback=128, horizon=5, samples=32, symbol='002436.SZ', as_of=None):
    raw = load_bars(symbol, end=as_of, db_path=raw_db)
    as_of = as_of or raw.date.iloc[-1].date().isoformat()
    future = pd.bdate_range(pd.Timestamp(as_of)+pd.Timedelta(days=1), periods=horizon).strftime('%Y-%m-%d').tolist()
    _, _, info = kronos.prepare_input(raw, future, symbol=symbol, as_of=as_of,
                                      lookback=lookback, horizon=horizon)
    parameters = dict(lookback=lookback, horizon=horizon, samples=samples, seed=42,
                      temperature=.8, top_p=.9, device='cpu', dtype='float32')
    calendar = {'run_id': 'synthetic-calendar', 'sha256': 'a'*64}
    identity = dict(schema=kronos.SCHEMA, input=info, parameters=parameters,
                    future_dates=future, calendar={'verified': True, 'identity': calendar},
                    assets=[{'fixture': True}], source={'fixture': True})
    paths = np.empty((samples, horizon, 6))
    for i in range(samples):
        close = info['last_close'] + i/100 + np.arange(horizon)/100
        paths[i] = np.column_stack((close, close+1, close-1, close, close*0+10000, close*0+100000))
    value = kronos.summarize_samples(paths, future, info['last_close'])
    value.update(schema=kronos.SCHEMA, status='completed', symbol=symbol, as_of=as_of,
                 identity=identity, input=deepcopy(info), cache_key=kronos._hash(identity),
                 samples=paths[:8].tolist(), retained_diagnostic_samples=min(samples, 8))
    value['quality'].update(calendar_verified=True, input_contract_verified=True, source_verified=True)
    outer = dict(kind='prediction-v1', symbol=symbol, as_of=as_of, parameters=parameters,
                 calendar=calendar, data=raw.attrs['data_version'], code='kronos_request_cache_v2')
    return outer, value


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    raw = tmp_path/'raw.db'; days = make_raw(raw)
    tick = [100.]
    with ResearchStore(tmp_path/'cache', clock=lambda: tick[0], ttl_seconds=1000) as store:
        light = SimpleNamespace(store=store, raw_db=raw, jobs={'active': {'status': 'running'}})
        def forbidden(*_args, **_kwargs):
            pytest.fail('Chart read invoked computation or a business write')
        # Missing weights, a busy GPU, or another task must not affect a GET.
        for name in ('predict', '_load_model', '_device', 'asset_status', '_source_identity'):
            monkeypatch.setattr(kronos, name, forbidden)
        for name in ('prediction', 'source', 'train_score', 'submit', 'backtest', 'enable_account', 'calendar'):
            setattr(light, name, forbidden)
        app = FastAPI(); app.state.light = light; app.include_router(router)
        with TestClient(app) as client:
            yield SimpleNamespace(raw=raw, days=days, tick=tick, store=store, light=light,
                                  client=client, forbidden=forbidden, monkeypatch=monkeypatch)


def save(fixture, **kwargs):
    outer, value = make_prediction(fixture.raw, **kwargs)
    ref = fixture.store.put(outer, value, namespace='predictions')
    return ref, value


def get(fixture, reference):
    return fixture.client.get('/api/light/predictions/'+reference+'/chart')


def test_read_is_causal_exact_and_does_not_touch_cache_accounts_or_settings(fixture):
    f = fixture; ref, original = save(f)
    f.store.set_setting('enabled_accounts', {'fixture-account': {'enabled': True, 'last_date': '2024-01-01'}})
    account = f.store.put({'account': 'fixture'}, {'frozen_ledger': [1, 2]}, pin='saved_account')
    f.store.pin(ref['cache_key'], 'fixture-protected-forecast')
    f.tick[0] += 61  # A default touching read would now issue an UPDATE.
    before = '\n'.join(f.store.connection.iterdump())
    changes = f.store.connection.total_changes
    objects = {p: p.read_bytes() for p in f.store.objects.rglob('*.json.gz')}
    raw_bytes = f.raw.read_bytes()
    for name in ('put', 'pin', 'unpin', 'set_setting', 'gc', 'lease'):
        if hasattr(f.store, name):f.monkeypatch.setattr(f.store, name, f.forbidden)
    response = get(f, ref['cache_key'])
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['history_verification'] == dict(status='verified', reason=None, reason_code=None,
                                                 price_basis='unadjusted', input_sha256_match=True)
    assert len(result['history']) == 128
    assert result['history'][0]['date'] == original['input']['history_start']
    assert result['history'][-1]['date'] == original['as_of']
    assert result['history'][-1]['close'] == original['input']['last_close']
    assert all(row['date'] <= result['as_of'] for row in result['history'])
    assert all(row['date'] > result['as_of'] for row in result['forecast'])
    for key in ('forecast', 'summary', 'quality', 'input'):
        assert result[key] == original[key]
    assert result['parameters'] == original['identity']['parameters']
    assert result['calendar'] == original['identity']['calendar']
    assert result['prediction_ref'] == ref['cache_key'] != original['cache_key']
    assert result['prediction_identity_key'] == original['cache_key']
    assert '\n'.join(f.store.connection.iterdump()) == before
    assert f.store.connection.total_changes == changes
    assert {p: p.read_bytes() for p in objects} == objects
    assert f.raw.read_bytes() == raw_bytes
    assert f.store.get_by_key(account['cache_key'], touch=False) == {'frozen_ledger': [1, 2]}


@pytest.mark.parametrize('lookback,horizon,samples',[(128, 1, 8), (128, 5, 32), (256, 10, 64)])
def test_original_window_dates_and_full_sample_quantiles_are_preserved(fixture, lookback, horizon, samples):
    ref, original = save(fixture, lookback=lookback, horizon=horizon, samples=samples)
    response = get(fixture, ref['cache_key'])
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result['history']) == lookback and len(result['forecast']) == horizon
    assert result['forecast'] == original['forecast'] and result['quality'] == original['quality']
    assert [r['date'] for r in result['forecast']] == original['identity']['future_dates']
    if samples > 8:
        diagnostic = kronos.summarize_samples(np.asarray(original['samples']),
                                              original['identity']['future_dates'], original['input']['last_close'])
        assert result['forecast'] != diagnostic['forecast']


def test_future_and_other_stock_updates_do_not_change_original_history(fixture):
    f = fixture; ref, _ = save(f)
    first = get(f, ref['cache_key']).json()
    future = first['forecast'][0]['date']
    with closing(sqlite3.connect(f.raw)) as db:
        db.execute('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                   ('002436.SZ', future, 100, 101, 99, 100, 1, 100, 'sina_unadjusted_v1', 1))
        db.execute('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                   ('600519.SH', f.days[-1], 1000, 1001, 999, 1000, 1, 1000, 'sina_unadjusted_v1', 1))
        db.commit()
    assert get(f, ref['cache_key']).json() == first


@pytest.mark.parametrize('change,code',[
    ('revision', 'history_input_changed'), ('missing_cutoff', 'history_input_invalid'),
    ('duplicate', 'history_input_invalid'), ('invalid_ohlc', 'history_input_invalid'),
    ('source_changed', 'history_input_changed'), ('missing_database', 'history_read_failed'),
])
def test_bad_history_does_not_replace_or_recompute_the_forecast(fixture, change, code):
    f = fixture; ref, original = save(f)
    if change == 'missing_database':
        f.raw.rename(f.raw.with_suffix('.away'))
    else:
        with closing(sqlite3.connect(f.raw)) as db:
            if change == 'revision':db.execute('UPDATE stock_daily_normalized SET close=close+.01 WHERE date=?', (f.days[-1],))
            elif change == 'missing_cutoff':db.execute('DELETE FROM stock_daily_normalized WHERE date=?', (f.days[-1],))
            elif change == 'duplicate':db.execute('INSERT INTO stock_daily_normalized SELECT * FROM stock_daily_normalized WHERE date=?', (f.days[-1],))
            elif change == 'invalid_ohlc':db.execute('UPDATE stock_daily_normalized SET high=0 WHERE date=?', (f.days[-1],))
            elif change == 'source_changed':db.execute("UPDATE stock_daily_normalized SET source='legacy_fuyao' WHERE date=?", (f.days[-1],))
            db.commit()
    response = get(f, ref['cache_key'])
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['history'] == []
    assert result['history_verification']['status'] == 'unavailable'
    assert result['history_verification']['reason_code'] == code
    for key in ('forecast', 'summary', 'quality', 'input'):assert result[key] == original[key]
    assert str(f.raw) not in result['history_verification']['reason']


def test_original_unverified_basis_is_never_relabelled_unadjusted(fixture):
    f = fixture
    with closing(sqlite3.connect(f.raw)) as db:
        db.execute("UPDATE stock_daily_normalized SET source='legacy_fuyao'"); db.commit()
    ref, original = save(f)
    result = get(f, ref['cache_key']).json()
    assert result['input']['price_basis'] == 'mixed_legacy_adjustment_unverified'
    assert result['history'] == []
    assert result['history_verification']['reason_code'] == 'history_price_contract_unverified'
    assert result['history_verification']['input_sha256_match'] is True
    assert result['quality'] == original['quality']  # Display never grants or alters qualification.


@pytest.mark.parametrize('empty', [False, True])
def test_unusable_and_empty_forecasts_still_have_verified_historical_context(fixture, empty):
    outer, value = make_prediction(fixture.raw)
    value['quality']['prediction_usable'] = False
    if empty:value.update(forecast=[], summary=None)
    ref = fixture.store.put(outer, value, namespace='predictions')
    result = get(fixture, ref['cache_key']).json()
    assert result['history_verification']['status'] == 'verified'
    assert len(result['history']) == 128
    assert result['forecast'] == value['forecast'] and result['summary'] == value['summary']
    assert result['quality'] == value['quality']


@pytest.mark.parametrize('kind', ['invalid', 'missing', 'inner_key', 'wrong_namespace', 'expired'])
def test_wrong_missing_or_expired_references_are_404_without_raw_reads(fixture, kind):
    f = fixture; ref, value = save(f)
    key = ref['cache_key']
    if kind == 'invalid':key = 'not-a-cas-key'
    elif kind == 'missing':key = '0'*64
    elif kind == 'inner_key':key = value['cache_key']
    elif kind == 'wrong_namespace':key = f.store.put({'fixture': 'account'}, value, namespace='simulations')['cache_key']
    elif kind == 'expired':f.tick[0] += 1001
    f.monkeypatch.setattr(chart, 'load_bars', f.forbidden)
    response = get(f, key)
    assert response.status_code == 404
    assert response.json()['detail']['code'] == 'prediction_not_found'


def test_ttl_boundary_is_unavailable_before_physical_gc_without_touching_store(fixture):
    f = fixture; ref, _ = save(f)
    key = ref['cache_key']; path = f.store._path(ref['sha256'])
    content = path.read_bytes()
    expiry = f.store.connection.execute('SELECT expires_at FROM results WHERE cache_key=?', (key,)).fetchone()[0]
    f.tick[0] = expiry - 1
    assert get(f, key).status_code == 200
    before = '\n'.join(f.store.connection.iterdump()); changes = f.store.connection.total_changes
    f.monkeypatch.setattr(chart, 'load_bars', f.forbidden)
    f.monkeypatch.setattr(f.store, 'gc', f.forbidden)
    f.tick[0] = expiry
    assert get(f, key).status_code == 404
    assert path.read_bytes() == content
    assert '\n'.join(f.store.connection.iterdump()) == before
    assert f.store.connection.total_changes == changes
    assert f.store.connection.execute('SELECT state FROM objects WHERE sha256=?', (ref['sha256'],)).fetchone()[0] == 'active'


@pytest.mark.parametrize('protection', ['pin', 'lease'])
def test_expired_but_store_protected_prediction_remains_readable_without_refresh(fixture, protection):
    f = fixture; ref, original = save(f)
    key = ref['cache_key']; owner = 'protected-prediction-fixture'
    if protection == 'pin':f.store.pin(key, owner)
    else:f.store.acquire_lease(key, owner, ttl_seconds=2000)
    f.tick[0] += 1001
    before = '\n'.join(f.store.connection.iterdump()); changes = f.store.connection.total_changes
    content = f.store._path(ref['sha256']).read_bytes()
    response = get(f, key)
    assert response.status_code == 200, response.text
    assert response.json()['forecast'] == original['forecast']
    assert response.json()['history_verification']['status'] == 'verified'
    assert '\n'.join(f.store.connection.iterdump()) == before
    assert f.store.connection.total_changes == changes
    if protection == 'pin':f.store.unpin(key, owner)
    else:f.tick[0] = 2100  # Lease equality is expired, independently of GC.
    before = '\n'.join(f.store.connection.iterdump()); changes = f.store.connection.total_changes
    f.monkeypatch.setattr(chart, 'load_bars', f.forbidden)
    assert get(f, key).status_code == 404
    assert f.store._path(ref['sha256']).read_bytes() == content
    assert '\n'.join(f.store.connection.iterdump()) == before
    assert f.store.connection.total_changes == changes


@pytest.mark.parametrize('mutation', ['bytes', 'identity_hash', 'symbol', 'cutoff', 'parameters', 'future_dates'])
def test_corrupt_or_inconsistent_saved_predictions_are_409_before_reading_prices(fixture, mutation):
    f = fixture; outer, value = make_prediction(f.raw)
    if mutation == 'identity_hash':value['cache_key'] = '0'*64
    elif mutation == 'symbol':value['symbol'] = '600519.SH'
    elif mutation == 'cutoff':value['as_of'] = '2024-01-01'
    elif mutation == 'parameters':outer['parameters'] = {**outer['parameters'], 'samples': 64}
    elif mutation == 'future_dates':
        value['identity']['future_dates'][0] = value['as_of']
        value['cache_key'] = kronos._hash(value['identity'])
    ref = f.store.put(outer, value, namespace='predictions')
    if mutation == 'bytes':
        obj = f.store._path(ref['sha256']); obj.chmod(obj.stat().st_mode | stat.S_IWRITE)
        obj.write_bytes(b'invalid object bytes')
    f.monkeypatch.setattr(chart, 'load_bars', f.forbidden)
    response = get(f, ref['cache_key'])
    assert response.status_code == 409
    assert response.json()['detail']['code'] == 'prediction_integrity_failed'


def test_beijing_reference_cannot_start_a_new_historical_chart_read(fixture):
    f = fixture
    with closing(sqlite3.connect(f.raw)) as db:
        db.execute("UPDATE stock_daily_normalized SET stock='920001.BJ'"); db.commit()
    outer, value = make_prediction(f.raw, symbol='920001.BJ')
    ref = f.store.put(outer, value, namespace='predictions')
    f.monkeypatch.setattr(chart, 'load_bars', f.forbidden)
    assert get(f, ref['cache_key']).status_code == 409
