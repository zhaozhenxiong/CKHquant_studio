"""Cache content, causal identity, corruption rejection and SQLite concurrency."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_native_cache as cache
from my_strategy.services.czsc_native_evidence import build_native_evidence
from my_strategy.services.czsc_market_features import build_features
from my_strategy.services.czsc_trend_source import _save_native_pack, _raw_content_hash
from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES


def sha(value): return hashlib.sha256(value).hexdigest()


@pytest.fixture
def candidate(tmp_path):
    dates = pd.bdate_range('2024-01-02', periods=85)
    close = 20 + np.sin(np.arange(len(dates)) / 3) * 2 + np.arange(len(dates)) / 40
    raw = pd.DataFrame({'symbol': '000001.SZ', 'date': dates, 'open': close * .999,
        'high': close * 1.02, 'low': close * .98, 'close': close,
        'volume': 1000000., 'amount': close * 1000000.,
        'source': sorted(VERIFIED_SOURCES)[0], 'has_trade_price': 1})
    root = tmp_path / 'producer'; root.mkdir()
    raw_path = root / 'raw.parquet'; raw.to_parquet(raw_path, index=False)
    features = build_features(raw)
    for column in raw:
        if column not in features: features[column] = raw[column].to_numpy(copy=True)
    pack = build_native_evidence(features, market_dates=dates.strftime('%Y-%m-%d').tolist())
    numeric = pack['frame'].drop(columns=[c for c in pack['frame'] if pack['frame'][c].map(lambda v: isinstance(v, (dict, list, tuple))).any()])
    numeric.attrs = {}
    numeric.to_parquet(root / 'features.parquet', index=False)
    _save_native_pack(root / 'native.json.gz', {k: v for k, v in pack.items() if k != 'frame'})
    series_hash = stable_hash(raw.to_csv(index=False))
    record = {'symbol': '000001.SZ', 'raw_path': str(raw_path), 'raw_sha256': sha(raw_path.read_bytes()),
        'raw_content_sha256': _raw_content_hash(raw), 'raw_series_hash': series_hash,
        'raw_columns': list(raw), 'raw_rows': len(raw), 'raw_data_start': str(dates[0].date()),
        'data_end': str(dates[-1].date()), 'raw_attrs': {'price_basis': 'unadjusted', 'legacy_fuyao_bars': 0,
            'unverified_trade_price_bars': 0, 'volume_unit': 'shares', 'amount_unit': 'CNY', 'data_version': series_hash},
        'liquidity_policy_hash': stable_hash({'policy': 'fixture'}), 'liquidity_sha256': stable_hash({'liquidity': 'fixture'}),
        'liquidity_rows': len(raw), 'liquidity_eligible_sessions': len(raw), 'liquidity_path': str(root / 'liquidity.parquet'),
        'path': str(root / 'features.parquet'), 'sha256': sha((root / 'features.parquet').read_bytes()),
        'native_path': str(root / 'native.json.gz'), 'native_sha256': sha((root / 'native.json.gz').read_bytes()),
        'native_encoding': 'gzip_json', 'native_config_hash': pack['config_hash'],
        'native_schema_hash': pack['schema_hash'], 'rows': len(raw), 'raw_only_rebuild': True, 'reused_model_or_label': False}
    calendar = dates.strftime('%Y-%m-%d').tolist()
    header = {'calendar_hash': stable_hash(calendar), 'config_hash': stable_hash({'cfg': 'fixture'}),
        'native_config_hash': pack['config_hash'], 'native_schema_hash': pack['schema_hash'],
        'code_hashes': {'services/generator.py': stable_hash({'code': 'fixture'})},
        'cpu_workers': 6, 'run_id': 'producer', 'prepare_batch_size': 360}
    identity = cache.make_native_cache_identity(record, header, calendar)
    return record, header, calendar, identity


def test_cache_reuses_identical_input_across_runs_with_verified_frozen_bytes(candidate, tmp_path):
    record, header, calendar, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        assert store.lookup(identity) is None
        produced = store.store(identity, record)
        second_raw = {**record, 'raw_path': str(tmp_path / 'new-raw.parquet'),
            'liquidity_path': str(tmp_path / 'new-liquidity.parquet')}
        Path(second_raw['raw_path']).write_bytes(Path(record['raw_path']).read_bytes())
        second_header = {**header, 'run_id': 'new-run', 'cpu_workers': 10, 'prepare_batch_size': 20}
        assert cache.make_native_cache_identity(second_raw, second_header, calendar) == identity
        result = store.lookup(identity).materialize(tmp_path / 'new-native', second_raw)
        assert result['raw_path'] == second_raw['raw_path']
        assert result['path'] != record['path']
        assert Path(result['path']).read_bytes() == Path(record['path']).read_bytes()
        assert Path(result['native_path']).read_bytes() == Path(record['native_path']).read_bytes()
        assert produced.producer['raw_path'] == record['raw_path']
        pd.testing.assert_frame_equal(pd.read_parquet(result['path']), pd.read_parquet(record['path']))
        assert store.connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        assert store.connection.execute('SELECT COUNT(*) FROM native_structure_entries').fetchone()[0] == 1
        with pytest.raises(ValueError, match='already exists'):
            produced.materialize(tmp_path / 'new-native', second_raw)


@pytest.mark.parametrize('part,key', [
    ('raw', 'raw_sha256'), ('raw', 'raw_content_sha256'), ('raw', 'raw_series_hash'),
    ('liquidity', 'liquidity_sha256'), ('liquidity', 'liquidity_policy_hash'),
    ('calendar', 'dates_hash'), ('calendar', 'source_hash'), ('calendar', 'prefix_hash'),
    ('root', 'config_hash'), ('root', 'native_schema_hash'), ('root', 'native_config_hash'),
    ('code_hashes', 'services/generator.py'), ('engine_contract', 'source_tree_sha256'),
    ('engine_contract', 'manifest_sha256'), ('feature_schema_hashes', 'market'),
    ('feature_schema_hashes', 'profile')])
def test_every_generating_input_change_is_a_miss(candidate, tmp_path, part, key):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        store.store(identity, record)
        changed = copy.deepcopy(identity)
        target = changed if part == 'root' else changed[part]
        target[key] = stable_hash({'changed': target[key]})
        if part == 'raw' and key == 'raw_series_hash':
            changed['raw']['raw_attrs']['data_version'] = target[key]
        assert store.lookup(changed) is None


@pytest.mark.parametrize('column', ['features_blob', 'native_blob', 'metadata_json', 'identity_json', 'producer_json'])
def test_existing_cache_corruption_is_explicit_failure(candidate, tmp_path, column):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        store.store(identity, record)
        with store.connection:
            value = b'corrupted' if column.endswith('_blob') else '{}'
            store.connection.execute(f'UPDATE native_structure_entries SET {column}=?', [value])
        with pytest.raises(ValueError): store.lookup(identity)
        with pytest.raises(ValueError): store.store(identity, record)


def test_rehashed_blob_with_wrong_parsed_config_is_rejected(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        pack = json.loads(gzip.decompress(entry.native_blob)); pack['config_hash'] = stable_hash({'drift': True})
        blob = gzip.compress(json.dumps(pack, ensure_ascii=False).encode(), mtime=0)
        metadata = {**entry.metadata, 'native_sha256': sha(blob)}
        metadata_json = cache._canonical(metadata)
        with store.connection:
            store.connection.execute('UPDATE native_structure_entries SET native_blob=?,metadata_json=?,metadata_sha256=?',
                [blob, metadata_json, sha(metadata_json.encode())])
        with pytest.raises(ValueError, match='schema/config'): store.lookup(identity)


def test_rehashed_parquet_with_wrong_rows_is_rejected(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        frame = pd.read_parquet(io.BytesIO(entry.features_blob)).iloc[:-1]
        output = io.BytesIO(); frame.to_parquet(output, index=False); blob = output.getvalue()
        metadata = {**entry.metadata, 'sha256': sha(blob)}
        metadata_json = cache._canonical(metadata)
        with store.connection:
            store.connection.execute('UPDATE native_structure_entries SET features_blob=?,metadata_json=?,metadata_sha256=?',
                [blob, metadata_json, sha(metadata_json.encode())])
        with pytest.raises(ValueError, match='row/schema'): store.lookup(identity)


def test_different_output_for_same_input_preserves_first_entry(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        original = store.store(identity, record)
        frame = pd.read_parquet(record['path']); frame.loc[5, 'ret_5'] += .001
        frame.to_parquet(record['path'], index=False)
        changed = {**record, 'sha256': sha(Path(record['path']).read_bytes())}
        with pytest.raises(ValueError, match='different output'): store.store(identity, changed)
        assert store.lookup(identity).features_blob == original.features_blob


def test_parallel_sqlite_connections_commit_atomic_idempotent_rows(candidate, tmp_path):
    record, _, _, identity = candidate
    db = tmp_path / 'cache.db'
    first_open = threading.Barrier(4)
    def put(index):
        if index < 4: first_open.wait(timeout=5)
        with cache.NativeStructureCache(db) as store:
            hit = store.store(identity, record)
            return hit.cache_key, len(hit.features_blob)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(put, range(8)))
    assert len(set(results)) == 1
    with cache.NativeStructureCache(db) as store:
        assert store.connection.execute('SELECT COUNT(*) FROM native_structure_entries').fetchone()[0] == 1
        assert store.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert store.lookup(identity) is not None


def test_existing_unknown_database_is_rejected_without_repurposing(tmp_path):
    db = tmp_path / 'other.db'
    with sqlite3.connect(db) as connection: connection.execute('CREATE TABLE valuable_data (id INTEGER)')
    with pytest.raises(ValueError, match='identity missing'): cache.NativeStructureCache(db)
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [('valuable_data',)]


def test_actual_frozen_engine_and_schema_from_header_override_cached_generation(candidate):
    record, header, calendar, _ = candidate
    generation = cache._generation_contract()
    engine = {**generation['engine'], 'source_tree_sha256': stable_hash({'actual_engine': 'replacement'})}
    schemas = {'market': stable_hash({'actual_market': True}), 'profile': stable_hash({'actual_profile': True})}
    identity = cache.make_native_cache_identity(record, {**header, 'engine_contract': engine,
        'feature_schema_hashes': schemas}, calendar)
    assert identity['engine_contract'] == engine
    assert identity['feature_schema_hashes'] == schemas


def test_rehashed_native_blob_with_wrong_calendar_provenance_is_rejected(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        pack = json.loads(gzip.decompress(entry.native_blob))
        pack['calendar_provenance']['calendar_prefix_hash'] = stable_hash({'drift': True})
        blob = gzip.compress(json.dumps(pack, ensure_ascii=False).encode(), mtime=0)
        metadata = {**entry.metadata, 'native_sha256': sha(blob)}
        metadata_json = cache._canonical(metadata)
        with store.connection:
            store.connection.execute('UPDATE native_structure_entries SET native_blob=?,metadata_json=?,metadata_sha256=?',
                [blob, metadata_json, sha(metadata_json.encode())])
        with pytest.raises(ValueError, match='calendar provenance'): store.lookup(identity)


def test_rehashed_parquet_with_changed_price_is_rejected_by_frozen_raw_content(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        frame = pd.read_parquet(io.BytesIO(entry.features_blob)); frame.loc[0, 'close'] += .001
        output = io.BytesIO(); frame.to_parquet(output, index=False); blob = output.getvalue()
        metadata = {**entry.metadata, 'sha256': sha(blob)}
        metadata_json = cache._canonical(metadata)
        with store.connection:
            store.connection.execute('UPDATE native_structure_entries SET features_blob=?,metadata_json=?,metadata_sha256=?',
                [blob, metadata_json, sha(metadata_json.encode())])
        with pytest.raises(ValueError, match='raw content identity'): store.lookup(identity)


def test_first_wrong_price_candidate_is_rejected_before_any_cache_insert(candidate, tmp_path):
    record, _, _, identity = candidate
    frame = pd.read_parquet(record['path']); frame.loc[0, 'close'] += .001
    frame.to_parquet(record['path'], index=False)
    wrong = {**record, 'sha256': sha(Path(record['path']).read_bytes())}
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        with pytest.raises(ValueError, match='raw content identity'): store.store(identity, wrong)
        assert store.lookup(identity) is None


def test_rehashed_metadata_with_wrong_type_is_explicit_value_error(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        store.store(identity, record)
        metadata_json = '[]'
        with store.connection:
            store.connection.execute('UPDATE native_structure_entries SET metadata_json=?,metadata_sha256=?',
                [metadata_json, sha(metadata_json.encode())])
        with pytest.raises(ValueError, match='metadata schema'): store.lookup(identity)


def _actual_header(candidate, tmp_path, monkeypatch, config, *, workers=6, batch=360):
    from my_strategy.services import czsc_trend_source as source
    record, _, calendar, _ = candidate
    raw_root = tmp_path / 'actual-header-raw'; (raw_root / 'reports').mkdir(parents=True, exist_ok=True)
    (raw_root / 'reports/raw_source.json').write_text('{}')
    monkeypatch.setattr(source, '_source_run_root', lambda _: raw_root)
    raw_source = {'snapshot_hash': stable_hash({'snapshot': 'actual-fixture'}),
        'raw_records': [record], 'calendar_hash': stable_hash(calendar),
        'liquidity': {'population_hash': stable_hash({'population': 'fixture'})},
        'selected_symbols': [record['symbol']]}
    return source._checkpoint_header('native-test', 'raw-test', raw_source, config,
        workers, batch, {'services/generator.py': stable_hash({'generator': 'fixture'})})['identity']


def test_actual_checkpoint_headers_reuse_structure_when_only_training_and_workers_change(candidate, tmp_path, monkeypatch):
    from my_strategy.services.czsc_trend_contracts import strategy_config
    from my_strategy.services.model_training_center import _native_input_contract
    record, _, calendar, _ = candidate
    config = strategy_config()
    first = _actual_header(candidate, tmp_path, monkeypatch, config)
    changed = copy.deepcopy(config)
    changed.update(epochs=config['epochs'] + 1, batch_size=config['batch_size'] * 2,
        stock_batch_size=64, initial_cash=config['initial_cash'] * 2)
    second = _actual_header(candidate, tmp_path, monkeypatch, changed, workers=10, batch=64)
    assert first['config_hash'] != second['config_hash']
    assert first['native_input_contract_hash'] == second['native_input_contract_hash']
    assert first['native_input_contract_hash'] == _native_input_contract(config)['contract_hash']
    one = cache.make_native_cache_identity(record, first, calendar)
    two = cache.make_native_cache_identity(record, second, calendar)
    assert one == two
    assert one['config_hash'] == first['native_input_contract_hash']
    with cache.NativeStructureCache(tmp_path / 'training-independent-cache.db') as store:
        store.store(one, record)
        assert store.lookup(two) is not None


@pytest.mark.parametrize('changes', [
    {'native_evidence': {'risk_buffer_atr': .30}},
    {'wyckoff': {'events': {'range_bars': 65}}},
    {'strategy': {'min_bi_len': 7}},
    {'feature_start': '2023-02-01'},
    {'liquidity_universe': {'min_median_amount': 31000000.}},
])
def test_actual_checkpoint_headers_miss_when_native_inputs_change(candidate, tmp_path, monkeypatch, changes):
    from my_strategy.services.czsc_trend_contracts import strategy_config
    from my_strategy.services.model_training_center import _native_input_contract
    record, _, calendar, _ = candidate
    first_cfg, second_cfg = strategy_config(), strategy_config(changes)
    first = _actual_header(candidate, tmp_path, monkeypatch, first_cfg)
    second = _actual_header(candidate, tmp_path, monkeypatch, second_cfg)
    assert first['native_input_contract_hash'] != second['native_input_contract_hash']
    assert second['native_input_contract_hash'] == _native_input_contract(second_cfg)['contract_hash']
    one = cache.make_native_cache_identity(record, first, calendar)
    two = cache.make_native_cache_identity(record, second, calendar)
    assert one != two
    with cache.NativeStructureCache(tmp_path / 'changed-input-cache.db') as store:
        store.store(one, record)
        assert store.lookup(two) is None


class _WalConnection:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.journal_calls = 0
        self.timeout = None
        self.mode = None

    def execute(self, statement):
        if statement.startswith('PRAGMA busy_timeout='):
            self.timeout = int(statement.rsplit('=', 1)[1])
            return self
        assert statement == 'PRAGMA journal_mode=WAL'
        self.journal_calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception): raise outcome
        self.mode = outcome
        return self

    def fetchone(self): return self.mode


def _sqlite_error(code):
    error = sqlite3.OperationalError('test SQLite initialization error')
    if code is not None: error.sqlite_errorcode = code
    return error


@pytest.mark.parametrize('code', [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED,
    sqlite3.SQLITE_BUSY | (1 << 8), sqlite3.SQLITE_LOCKED | (1 << 8)])
def test_wal_retry_is_limited_to_sqlite_lock_error_codes(monkeypatch, code):
    connection = _WalConnection([_sqlite_error(code), ('wal',)])
    sleeps = []
    monkeypatch.setattr(cache.time, 'sleep', sleeps.append)
    cache._enable_wal(connection)
    assert connection.journal_calls == 2
    assert sleeps == [.05]
    assert connection.timeout == 60000


@pytest.mark.parametrize('code', [None, sqlite3.SQLITE_IOERR, sqlite3.SQLITE_CORRUPT])
def test_wal_non_lock_errors_are_immediate_and_preserve_the_exception(monkeypatch, code):
    error = _sqlite_error(code)
    connection = _WalConnection([error, ('wal',)])
    sleeps = []
    monkeypatch.setattr(cache.time, 'sleep', sleeps.append)
    with pytest.raises(sqlite3.OperationalError) as captured: cache._enable_wal(connection)
    assert captured.value is error
    assert connection.journal_calls == 1
    assert sleeps == []


@pytest.mark.parametrize('mode', [('delete',), ('memory',), None])
def test_wal_mode_change_must_return_wal(monkeypatch, mode):
    connection = _WalConnection([mode])
    sleeps = []
    monkeypatch.setattr(cache.time, 'sleep', sleeps.append)
    with pytest.raises(ValueError, match='did not become WAL'): cache._enable_wal(connection)
    assert connection.journal_calls == 1
    assert sleeps == []


def test_wal_retry_does_not_attempt_again_after_the_shared_deadline(monkeypatch):
    error = _sqlite_error(sqlite3.SQLITE_BUSY)
    connection = _WalConnection([error, ('wal',)])
    clock = {'now': 0.}
    monkeypatch.setattr(cache, '_SQLITE_BUSY_TIMEOUT_SECONDS', .05)
    monkeypatch.setattr(cache.time, 'monotonic', lambda: clock['now'])
    def advance(seconds): clock['now'] += seconds
    monkeypatch.setattr(cache.time, 'sleep', advance)
    with pytest.raises(sqlite3.OperationalError) as captured: cache._enable_wal(connection)
    assert captured.value is error
    assert clock['now'] == .05
    assert connection.journal_calls == 1
    assert connection.timeout == 0


def _hold_cache_read_lock(path):
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA user_version=0')
    connection.execute('BEGIN')
    connection.execute('SELECT COUNT(*) FROM sqlite_master').fetchone()
    return connection


def test_first_wal_initialization_retries_real_sqlite_lock_until_released(tmp_path, monkeypatch):
    path = tmp_path / 'held-cache.db'
    holder = _hold_cache_read_lock(path)
    attempted = threading.Event()
    errors = []
    original_connect = sqlite3.connect
    class ObservedConnection(sqlite3.Connection):
        def execute(self, statement, *args, **kwargs):
            try: return super().execute(statement, *args, **kwargs)
            except sqlite3.OperationalError as exc:
                if statement == 'PRAGMA journal_mode=WAL':
                    errors.append(exc.sqlite_errorcode); attempted.set()
                raise
    def connect(*args, **kwargs):
        return original_connect(*args, **kwargs, factory=ObservedConnection)
    monkeypatch.setattr(cache.sqlite3, 'connect', connect)
    def open_cache():
        with cache.NativeStructureCache(path) as store:
            assert store.connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
            assert store.connection.execute('PRAGMA busy_timeout').fetchone()[0] == 60000
            assert store.connection.execute('PRAGMA user_version').fetchone()[0] == cache.CACHE_VERSION
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(open_cache)
        try: assert attempted.wait(5), 'expected a genuine SQLite mode-change lock'
        finally: holder.rollback(); holder.close()
        future.result(timeout=5)
    assert errors and all((code & 0xff) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) for code in errors)


def test_wal_real_lock_timeout_is_bounded_and_preserves_sqlite_error(tmp_path, monkeypatch):
    path = tmp_path / 'timeout-cache.db'
    holder = _hold_cache_read_lock(path)
    monkeypatch.setattr(cache, '_SQLITE_BUSY_TIMEOUT_SECONDS', .08)
    started = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError) as captured: cache.NativeStructureCache(path)
        assert (captured.value.sqlite_errorcode & 0xff) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
        assert .06 <= time.monotonic() - started < .5
        assert holder.execute('PRAGMA user_version').fetchone()[0] == 0
    finally: holder.rollback(); holder.close()
    with cache.NativeStructureCache(path) as store:
        assert store.connection.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'


def test_parallel_first_open_pressure_preserves_wal_and_schema(tmp_path):
    for iteration in range(25):
        path = tmp_path / f'first-open-{iteration}.db'
        barrier = threading.Barrier(8)
        def initialize(_):
            barrier.wait(timeout=5)
            with cache.NativeStructureCache(path) as store:
                return (store.connection.execute('PRAGMA journal_mode').fetchone()[0],
                    store.connection.execute('PRAGMA user_version').fetchone()[0],
                    store.connection.execute('PRAGMA integrity_check').fetchone()[0],
                    store.connection.execute('PRAGMA busy_timeout').fetchone()[0])
        with ThreadPoolExecutor(max_workers=8) as pool:
            result = list(pool.map(initialize, range(8)))
        assert result == [('wal', cache.CACHE_VERSION, 'ok', 60000)] * 8
