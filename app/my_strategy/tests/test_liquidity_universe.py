from __future__ import annotations

import json
import sqlite3
import gzip
from pathlib import Path

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_trend_source as source
from my_strategy.services import czsc_trend_support as support


@pytest.fixture
def warehouse(tmp_path,monkeypatch):
    dates=pd.bdate_range('2025-02-03',periods=25).strftime('%Y-%m-%d').tolist()
    path=tmp_path/'raw.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE stock_daily_normalized('
            'stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,'
            'source TEXT,has_trade_price INTEGER,PRIMARY KEY(stock,date))')
        stocks=[('000001.SZ',dates[:20],40_000_000), # Historical qualifier, no latest quote.
                ('600001.SH',dates[5:],50_000_000), # Later entrant, no pre2023 history.
                ('000002.SZ',dates,10_000_000),    # Always below prior threshold.
                ('688001.SH',dates,70_000_000)]    # Unsupported execution board.
        for symbol,observed,amount in stocks:
            connection.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
                [(symbol,date,10.,11.,9.,10.,amount/10.,amount,'tushare',1) for date in observed])
    monkeypatch.setattr(support,'_calendar',lambda run_id=None:(dates,'verified-fixture-calendar'))
    return path,dates


def population(warehouse,**kwargs):
    path,dates=warehouse
    return source.build_liquidity_universe(feature_start=dates[0],end=dates[-1],db_path=path,**kwargs)


def test_history_union_preserves_exited_stocks_and_admits_later_entrants(warehouse):
    report=population(warehouse)
    assert report['selected_symbols']==['000001.SZ','600001.SH']
    assert report['latest_symbols']==['600001.SH']
    assert report['selected_count']==2 and report['latest_count']==1
    assert report['candidate_count']==3 and report['unsupported_stock_count']==1
    assert report['daily_counts'][19]['eligible_count']==1
    assert report['daily_counts'][-1]['eligible_count']==1
    assert report['latest_exclusion_counts']=={
        'below_median_amount':1,'missing_market_observation_unknown':1}
    excluded=sum(report['exclusion_counts'].values())
    qualified=sum(row['eligible_count'] for row in report['daily_counts'])
    assert excluded+qualified==report['candidate_count']*len(report['daily_counts'])
    assert report['exclusion_reason_counts_overlap'] is True
    assert sum(report['exclusion_reason_counts'].values())>excluded
    assert report['publication_allowed'] is False
    assert 'not_independently_verified' in report['historical_identity_status']


def test_raw_catalog_change_requires_new_preflight_even_if_pool_unchanged(warehouse):
    path,dates=warehouse
    before=population(warehouse)
    with sqlite3.connect(path) as connection:
        connection.execute('UPDATE stock_daily_normalized SET amount=20000000 '
                           'WHERE stock=? AND date=?',['000002.SZ',dates[0]])
    after=population(warehouse)
    assert after['selected_symbols']==before['selected_symbols']
    assert after['daily_membership_hash']==before['daily_membership_hash']
    assert after['input_catalog_hash']!=before['input_catalog_hash']
    assert after['population_hash']!=before['population_hash']
    with pytest.raises(ValueError,match='new preflight'):
        source.freeze_raw_source(symbols=before['selected_symbols'],end=dates[-1],feature_start=dates[0],
            db_path=path,cpu_workers=1,preflight_population=before,prepare_batch_size=1)


def test_source_freezes_all_union_with_batched_resource_limits_and_all_reasons(warehouse):
    path,dates=warehouse
    preflight=population(warehouse)
    base,report=source.freeze_raw_source(symbols=preflight['selected_symbols'],end=dates[-1],feature_start=dates[0],
        db_path=path,cpu_workers=1,preflight_population=preflight,prepare_batch_size=1)
    assert report['population_mode']=='full_liquidity_union'
    assert report['selected_symbols']==preflight['selected_symbols']
    assert len(report['raw_records'])==2
    assert report['latest_observed_symbols']==1
    assert report['snapshot_hash']==stable_hash({k:v for k,v in report.items() if k!='snapshot_hash'})
    liquidity=report['liquidity']
    assert liquidity['population_hash']==preflight['population_hash']
    assert {row['symbol'] for row in liquidity['membership_assets']}==set(preflight['candidate_symbols'])
    for asset in liquidity['membership_assets']:
        assert support._file_hash(Path(asset['path']))==asset['sha256']
        daily=pd.read_parquet(asset['path'])
        assert len(daily)==len(dates)
    for record in report['raw_records']:
        raw=pd.read_parquet(record['raw_path'])
        assert 'liquidity_pool_eligible' in raw
        assert raw['liquidity_policy_hash'].eq(preflight['policy_hash']).all()
    saved=json.loads((base/'reports/liquidity_universe.json').read_text())
    assert saved['daily_membership_hash']==preflight['daily_membership_hash']
    source._verify_population(saved)


def test_fixed_subset_cannot_replace_preflight_historical_union(warehouse):
    path,dates=warehouse
    report=population(warehouse)
    with pytest.raises(ValueError,match='complete frozen historical liquidity union'):
        source.freeze_raw_source(symbols=report['latest_symbols'],end=dates[-1],feature_start=dates[0],
            db_path=path,cpu_workers=1,preflight_population=report)


def test_native_prepare_rejects_changed_daily_asset_before_processing(warehouse):
    path,dates=warehouse
    preflight=population(warehouse)
    _,report=source.freeze_raw_source(symbols=preflight['selected_symbols'],end=dates[-1],feature_start=dates[0],
        db_path=path,cpu_workers=1,preflight_population=preflight)
    member=report['liquidity']['membership_assets'][0]
    with open(member['path'],'ab') as handle:
        handle.write(b'corrupted')
    with pytest.raises(ValueError,match='population asset drift'):
        source.prepare_native_source(report['run_id'],cpu_workers=1)


def test_native_worker_recomputes_frozen_flags_instead_of_trusting_valid_sha(warehouse,tmp_path):
    path,dates=warehouse
    report=population(warehouse)
    _,frozen=source.freeze_raw_source(symbols=report['selected_symbols'],end=dates[-1],feature_start=dates[0],
        db_path=path,cpu_workers=1,preflight_population=report)
    record=frozen['raw_records'][0]
    raw=pd.read_parquet(record['raw_path'])
    raw.loc[0,'liquidity_pool_eligible']=True # Impossible incomplete 20-session window.
    raw.to_parquet(record['raw_path'],index=False)
    record['raw_sha256']=support._file_hash(Path(record['raw_path']))
    with pytest.raises(ValueError,match='qualification drift'):
        source._native_worker((record,source.strategy_config(),dates,str(tmp_path/'native')))


def test_population_has_no_dynamic_identity_and_explicit_scope_is_disclosed(warehouse):
    first=population(warehouse,preferred_symbols=['600001.SH','000001.SZ'])
    second=population(warehouse,preferred_symbols=['000001.SZ','600001.SH'])
    assert first==second
    assert first['basis_mode']=='explicit_observable_subset'
    assert first['requested_symbols']==['000001.SZ','600001.SH']
    with pytest.raises(ValueError,match='unsupported'):
        population(warehouse,preferred_symbols=['688001.SH'])


def test_explicit_verified_future_session_still_requires_closed_raw_observations(warehouse,monkeypatch):
    path,dates=warehouse
    next_session=(pd.Timestamp(dates[-1])+pd.offsets.BDay()).date().isoformat()
    monkeypatch.setattr(support,'_calendar',lambda run_id=None:(dates+[next_session],'verified-fixture-calendar'))
    with pytest.raises(ValueError,match='available closed raw'):
        source.build_liquidity_universe(feature_start=dates[0],end=next_session,db_path=path)


def test_real_native_rebuild_keeps_liquidity_separate_from_existing_feature_quality(warehouse,monkeypatch):
    path,old_dates=warehouse
    dates=pd.bdate_range(old_dates[0],periods=140).strftime('%Y-%m-%d').tolist()
    with sqlite3.connect(path) as connection:
        connection.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',
            [('000001.SZ',date,10.,11.,9.,10.,4_000_000.,40_000_000.,'tushare',1) for date in dates[20:]])
    monkeypatch.setattr(support,'_calendar',lambda run_id=None:(dates,'verified-fixture-calendar'))
    preflight=source.build_liquidity_universe(feature_start=dates[0],end=dates[-1],db_path=path,
        preferred_symbols=['000001.SZ'])
    _,raw=source.freeze_raw_source(symbols=preflight['selected_symbols'],end=dates[-1],feature_start=dates[0],
        db_path=path,cpu_workers=1,preflight_population=preflight)
    native=source.prepare_native_source(raw['run_id'],cpu_workers=1,prepare_batch_size=1)
    assert native['selected_symbols']==['000001.SZ']
    assert native['source_snapshot']['liquidity']['population_hash']==preflight['population_hash']
    assert 'services/czsc_liquidity.py' in native['source_snapshot']['code_hashes']
    frame=pd.read_parquet(native['dataset_records'][0]['path'])
    assert len(frame)==140 and frame.iloc[19].liquidity_pool_eligible
    assert not frame.iloc[19].model_input_eligible
    assert not frame.iloc[19].input_eligible
    assert frame.iloc[-1].liquidity_pool_eligible
    record=native['dataset_records'][0]
    assert record['native_encoding']=='gzip_json' and record['native_path'].endswith('.json.gz')
    with gzip.open(record['native_path'],'rt',encoding='utf-8') as handle:
        pack=json.load(handle)
    assert pack['config_hash']==record['native_config_hash']
    assert pack['schema_hash']==record['native_schema_hash']


def test_native_gzip_encoding_is_deterministic_and_json_semantics_preserved(tmp_path):
    import numpy as np
    value={'source':'核验原始行情','values':[np.float64(1.25),np.float64('nan'),np.int64(3)],
           'dates':[pd.Timestamp('2026-10-12')],'flag':np.bool_(True)}
    first=tmp_path/'first.json.gz';second=tmp_path/'second.json.gz'
    source._save_native_pack(first,value);source._save_native_pack(second,value)
    assert first.read_bytes()==second.read_bytes()
    with gzip.open(first,'rt',encoding='utf-8') as handle:
        assert json.load(handle)==support._json(value)
    assert not (tmp_path/'first.json.gz.tmp').exists()
