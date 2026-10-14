"""Verified native structure outputs, keyed by complete causal inputs.

The coordinator alone reads/writes this SQLite cache. It contains numerical
feature Parquet and native evidence gzip, never a model, target, or label.
Each run receives its own immutable files materialized from verified blobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from typing import Any

from my_strategy.core.paths import DATA_ROOT

CACHE_FORMAT = 'native_structure_cache'
CACHE_VERSION = 1
_OUTPUT_KEYS = {'events', 'evaluations', 'wyckoff_events', 'auxiliary_evaluations',
                'calendar_provenance', 'version', 'config_hash', 'schema_hash', 'semantics'}
_RAW_KEYS = ('symbol', 'raw_sha256', 'raw_content_sha256', 'raw_series_hash', 'raw_columns',
             'raw_attrs', 'raw_rows', 'raw_data_start', 'data_end')
_LIQUIDITY_KEYS = ('liquidity_policy_hash', 'liquidity_sha256', 'liquidity_rows',
                   'liquidity_eligible_sessions')


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _object_hash(value):
    return _hash(_canonical(value).encode('utf-8'))


def _strict_json(value):
    def invalid_constant(constant):
        raise ValueError('native cache JSON has nonfinite constant: ' + constant)
    return json.loads(value, parse_constant=invalid_constant)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value) is not None


@lru_cache(maxsize=1)
def _generation_contract():
    from my_strategy.adapters.czsc_adapter import engine_contract
    from my_strategy.services.czsc_market_features import FEATURE_SCHEMA_HASH
    from my_strategy.services.czsc_market_profiles import get_feature_profile
    return {'engine': engine_contract(), 'market_schema_hash': FEATURE_SCHEMA_HASH,
            'profile_schema_hash': get_feature_profile('ma_trend')['schema_hash']}


def make_native_cache_identity(raw, header_identity, calendar_dates):
    """Exclude physical paths/run/worker counts; include all generating inputs."""
    from my_strategy.core.run_context import stable_hash
    dates = list(calendar_dates)
    if (not dates or dates != sorted(set(dates)) or
            any(not isinstance(day, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day) for day in dates)):
        raise ValueError('native cache requires sorted unique verified calendar dates')
    prefix = [day for day in dates if day <= raw['data_end']]
    generation = None
    if not header_identity.get('engine_contract') or not header_identity.get('feature_schema_hashes'):
        generation = _generation_contract()
    engine = header_identity.get('engine_contract') or generation['engine']
    feature_schemas = header_identity.get('feature_schema_hashes') or {
        'market': generation['market_schema_hash'], 'profile': generation['profile_schema_hash']}
    result = {'format': CACHE_FORMAT, 'version': CACHE_VERSION,
              'raw': {key: raw[key] for key in _RAW_KEYS},
              'liquidity': {key: raw[key] for key in _LIQUIDITY_KEYS},
              'calendar': {'source_hash': header_identity['calendar_hash'],
                           'dates_hash': stable_hash(dates), 'prefix_hash': stable_hash(prefix),
                           'prefix_end': prefix[-1] if prefix else None},
              'config_hash': header_identity.get('native_input_contract_hash') or header_identity['config_hash'],
              'native_config_hash': header_identity['native_config_hash'],
              'native_schema_hash': header_identity['native_schema_hash'],
              'code_hashes': header_identity['code_hashes'],
              'engine_contract': engine, 'feature_schema_hashes': feature_schemas}
    _validate_identity(result)
    return _strict_json(_canonical(result))


def _validate_identity(identity):
    keys = {'format', 'version', 'raw', 'liquidity', 'calendar', 'config_hash',
            'native_config_hash', 'native_schema_hash', 'code_hashes', 'engine_contract',
            'feature_schema_hashes'}
    if (not isinstance(identity, dict) or set(identity) != keys or
            identity.get('format') != CACHE_FORMAT or identity.get('version') != CACHE_VERSION):
        raise ValueError('native cache generating identity format invalid')
    raw = identity['raw']
    if (not isinstance(raw, dict) or set(raw) != set(_RAW_KEYS) or
            not re.fullmatch(r'\d{6}\.(?:SH|SZ|BJ)', str(raw.get('symbol', ''))) or
            type(raw.get('raw_rows')) is not int or raw['raw_rows'] < 1 or
            not isinstance(raw.get('raw_columns'), list) or not raw['raw_columns'] or
            not isinstance(raw.get('raw_attrs'), dict)):
        raise ValueError('native cache raw identity incomplete')
    for key in ('raw_sha256', 'raw_content_sha256', 'raw_series_hash'):
        if not _sha(raw.get(key)): raise ValueError('native cache raw SHA invalid: ' + key)
    attrs = raw['raw_attrs']
    required_attrs = {'price_basis', 'legacy_fuyao_bars', 'unverified_trade_price_bars',
                      'volume_unit', 'amount_unit', 'data_version'}
    if (not required_attrs <= set(attrs) or attrs.get('data_version') != raw['raw_series_hash'] or
            attrs.get('volume_unit') != 'shares' or attrs.get('amount_unit') != 'CNY'):
        raise ValueError('native cache source/units identity incomplete')
    for key in ('raw_data_start', 'data_end'):
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(raw.get(key, ''))):
            raise ValueError('native cache date boundary invalid')
    if raw['raw_data_start'] > raw['data_end']: raise ValueError('native cache date boundaries reversed')
    liquidity = identity['liquidity']
    if not isinstance(liquidity, dict) or set(liquidity) != set(_LIQUIDITY_KEYS):
        raise ValueError('native cache liquidity identity incomplete')
    for key in ('liquidity_policy_hash', 'liquidity_sha256'):
        if not _sha(liquidity.get(key)): raise ValueError('native cache liquidity SHA invalid')
    if (type(liquidity.get('liquidity_rows')) is not int or liquidity['liquidity_rows'] < 1 or
            type(liquidity.get('liquidity_eligible_sessions')) is not int or
            not 0 <= liquidity['liquidity_eligible_sessions'] <= liquidity['liquidity_rows']):
        raise ValueError('native cache liquidity row identity invalid')
    calendar = identity['calendar']
    if (not isinstance(calendar, dict) or set(calendar) != {'source_hash', 'dates_hash', 'prefix_hash', 'prefix_end'} or
            not all(_sha(calendar[k]) for k in ('source_hash', 'dates_hash', 'prefix_hash')) or
            (calendar['prefix_end'] is not None and not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(calendar['prefix_end'])))):
        raise ValueError('native cache verified calendar identity incomplete')
    for key in ('config_hash', 'native_config_hash', 'native_schema_hash'):
        if not _sha(identity.get(key)): raise ValueError('native cache schema/config SHA invalid')
    code = identity['code_hashes']
    if not isinstance(code, dict) or not code or not all(isinstance(k, str) and _sha(v) for k, v in code.items()):
        raise ValueError('native cache generating code identity incomplete')
    engine = identity['engine_contract']
    if (not isinstance(engine, dict) or not _sha(engine.get('source_tree_sha256')) or
            not _sha(engine.get('manifest_sha256')) or not engine.get('build_version')):
        raise ValueError('native cache actual engine identity incomplete')
    schemas = identity['feature_schema_hashes']
    if not isinstance(schemas, dict) or set(schemas) != {'market', 'profile'} or not all(_sha(v) for v in schemas.values()):
        raise ValueError('native cache feature schema identity incomplete')
    _canonical(identity)


def _inspect_blobs(identity, features_blob, native_blob):
    """Parse both physical blobs, including schema/config and input row boundaries."""
    import pandas as pd
    import pyarrow.parquet as pq
    from my_strategy.core.run_context import stable_hash
    from my_strategy.services.czsc_liquidity import RAW_IDENTITY_COLUMNS
    from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS
    if not features_blob or not native_blob: raise ValueError('native cache blob missing')
    try:
        parquet = pq.ParquetFile(io.BytesIO(features_blob))
        frame = pd.read_parquet(io.BytesIO(features_blob))
        pack = _strict_json(gzip.decompress(native_blob).decode('utf-8'))
    except (OSError, EOFError, ValueError, TypeError) as exc:
        raise ValueError('native cache blob parse failed') from exc
    raw = identity['raw']
    if (parquet.metadata.num_rows != raw['raw_rows'] or len(frame) != raw['raw_rows'] or
            not (set(RAW_IDENTITY_COLUMNS) | set(MA_TREND_COLUMNS) | {'native_schema_hash', 'native_version'}) <= set(frame.columns)):
        raise ValueError('native cache Parquet row/schema mismatch')
    observed_raw = frame[list(RAW_IDENTITY_COLUMNS)].copy()
    observed_raw['date'] = pd.to_datetime(observed_raw.date).dt.strftime('%Y-%m-%d')
    content_hash = _hash(observed_raw.to_csv(index=False, float_format='%.17g').encode('utf-8'))
    if content_hash != raw['raw_content_sha256']:
        raise ValueError('native cache Parquet immutable raw content identity mismatch')
    if not frame.symbol.eq(raw['symbol']).all(): raise ValueError('native cache Parquet stock identity mismatch')
    dates = pd.to_datetime(frame.date, errors='raise')
    if (dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing or
            str(dates.iloc[0].date()) != raw['raw_data_start'] or str(dates.iloc[-1].date()) != raw['data_end']):
        raise ValueError('native cache Parquet causal input boundaries mismatch')
    if (not isinstance(pack, dict) or set(pack) != _OUTPUT_KEYS or
            pack.get('schema_hash') != identity['native_schema_hash'] or
            pack.get('config_hash') != identity['native_config_hash'] or
            stable_hash(pack.get('semantics')) != identity['native_schema_hash'] or
            not isinstance(pack.get('version'), str) or not pack['version'] or
            not frame.native_schema_hash.eq(identity['native_schema_hash']).all() or
            not frame.native_version.eq(pack['version']).all()):
        raise ValueError('native cache native schema/config mismatch')
    provenance = pack['calendar_provenance']
    if (not isinstance(provenance, dict) or provenance.get('provided') is not True or
            provenance.get('calendar_prefix_hash') != identity['calendar']['prefix_hash'] or
            provenance.get('calendar_prefix_end') != identity['calendar']['prefix_end'] or
            provenance.get('future_calendar_entries_used_for_historical_qualification') is not False):
        raise ValueError('native cache verified causal calendar provenance mismatch')
    for key in ('evaluations', 'auxiliary_evaluations'):
        if not isinstance(pack[key], list) or len(pack[key]) != raw['raw_rows']:
            raise ValueError('native cache native evidence row count mismatch')
    expected_dates = dates.dt.strftime('%Y-%m-%d').tolist()
    for key in ('evaluations', 'auxiliary_evaluations'):
        if [x.get('date') for x in pack[key] if isinstance(x, dict)] != expected_dates:
            raise ValueError('native cache native evaluation causal dates mismatch')
    if not isinstance(pack['events'], list) or not isinstance(pack['wyckoff_events'], list):
        raise ValueError('native cache native events schema mismatch')
    return {'rows': len(frame), 'feature_columns': list(frame.columns),
            'parquet_schema': str(parquet.schema_arrow), 'native_version': pack['version'],
            'native_config_hash': pack['config_hash'], 'native_schema_hash': pack['schema_hash'],
            'sha256': _hash(features_blob), 'native_sha256': _hash(native_blob),
            'native_encoding': 'gzip_json', 'raw_only_rebuild': True, 'reused_model_or_label': False}


@dataclass(frozen=True)
class NativeCacheEntry:
    identity: dict[str, Any]
    metadata: dict[str, Any]
    producer: dict[str, Any]
    features_blob: bytes
    native_blob: bytes
    cache_key: str

    def materialize(self, directory, raw_record):
        """Atomically publish exactly these verified bytes in a fresh run directory."""
        _validate_identity(self.identity)
        if (_object_hash(self.identity) != self.cache_key or
                any(raw_record.get(key) != value for key, value in self.identity['raw'].items()) or
                any(raw_record.get(key) != value for key, value in self.identity['liquidity'].items())):
            raise ValueError('native cache materialization raw identity mismatch')
        observed = _inspect_blobs(self.identity, self.features_blob, self.native_blob)
        if observed != self.metadata: raise ValueError('native cache materialization metadata/blob mismatch')
        directory = Path(directory).resolve(); directory.mkdir(parents=True, exist_ok=True)
        target = directory / self.identity['raw']['symbol'].replace('.', '_')
        if target.exists(): raise ValueError('native cache output target already exists')
        temporary = Path(tempfile.mkdtemp(prefix='.cache-', dir=directory))
        try:
            for filename, blob in [('features.parquet', self.features_blob), ('native.json.gz', self.native_blob)]:
                with (temporary / filename).open('wb') as output:
                    output.write(blob); output.flush(); os.fsync(output.fileno())
            temporary.replace(target)
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
        return {**raw_record, 'path': str(target / 'features.parquet'), 'sha256': observed['sha256'],
                'native_path': str(target / 'native.json.gz'), 'native_sha256': observed['native_sha256'],
                'native_encoding': 'gzip_json', 'native_config_hash': observed['native_config_hash'],
                'native_schema_hash': observed['native_schema_hash'], 'rows': observed['rows'],
                'raw_only_rebuild': True, 'reused_model_or_label': False}


_SQLITE_BUSY_TIMEOUT_SECONDS = 60.0


def _enable_wal(connection):
    """Mode changes can bypass SQLite's busy handler during simultaneous opens."""
    deadline = time.monotonic() + _SQLITE_BUSY_TIMEOUT_SECONDS
    # One retry budget controls mode changes; an inner busy handler must not
    # add another full timeout to a late attempt. Construction closes failures.
    connection.execute('PRAGMA busy_timeout=0')
    last_busy = None
    while True:
        if time.monotonic() >= deadline:
            if last_busy is not None:
                raise last_busy
            raise sqlite3.OperationalError('native cache WAL initialization timed out')
        try:
            mode = connection.execute('PRAGMA journal_mode=WAL').fetchone()
        except sqlite3.OperationalError as exc:
            code = getattr(exc, 'sqlite_errorcode', None)
            if not isinstance(code, int) or (code & 0xff) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
            last_busy = exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(.05, remaining))
        else:
            if not mode or str(mode[0]).lower() != 'wal':
                raise ValueError('native cache SQLite journal mode did not become WAL')
            connection.execute('PRAGMA busy_timeout=' + str(int(_SQLITE_BUSY_TIMEOUT_SECONDS * 1000)))
            return


class NativeStructureCache:
    def __init__(self, cache_db_path=None):
        self.path = Path(cache_db_path or DATA_ROOT / 'processed' / 'czsc' / 'native_structure.db').expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
        try:
            self.connection.execute('PRAGMA busy_timeout=' + str(int(_SQLITE_BUSY_TIMEOUT_SECONDS * 1000)))
            _enable_wal(self.connection)
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.execute('BEGIN IMMEDIATE')
            with self.connection:
                version = self.connection.execute('PRAGMA user_version').fetchone()[0]
                if version not in (0, CACHE_VERSION): raise ValueError('unsupported native cache database version')
                tables = {x[0] for x in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if version == 0 and tables: raise ValueError('native cache database identity missing')
                self.connection.execute('''CREATE TABLE IF NOT EXISTS native_structure_entries (
                    cache_key TEXT PRIMARY KEY, identity_json TEXT NOT NULL, metadata_json TEXT NOT NULL,
                    metadata_sha256 TEXT NOT NULL, producer_json TEXT NOT NULL, producer_sha256 TEXT NOT NULL,
                    features_blob BLOB NOT NULL, native_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)''')
                self.connection.execute('PRAGMA user_version=1')
            expected = {'cache_key', 'identity_json', 'metadata_json', 'metadata_sha256', 'producer_json', 'producer_sha256', 'features_blob', 'native_blob', 'created_at'}
            if {x[1] for x in self.connection.execute('PRAGMA table_info(native_structure_entries)')} != expected:
                raise ValueError('native cache database schema drift')
        except BaseException:
            self.connection.close()
            raise

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self): self.connection.close()

    def lookup(self, identity):
        _validate_identity(identity)
        key = _object_hash(identity)
        row = self.connection.execute('''SELECT identity_json,metadata_json,metadata_sha256,producer_json,producer_sha256,
            features_blob,native_blob FROM native_structure_entries WHERE cache_key=?''', (key,)).fetchone()
        if row is None: return None
        try:
            stored_identity, metadata = _strict_json(row[0]), _strict_json(row[1])
            if not isinstance(metadata, dict): raise ValueError('native cache metadata schema invalid')
            if stored_identity != identity or _object_hash(stored_identity) != key:
                raise ValueError('native cache stored input identity drift')
            if _hash(row[1].encode('utf-8')) != row[2]: raise ValueError('native cache metadata SHA drift')
            producer = _strict_json(row[3])
            if (not isinstance(producer, dict) or _hash(row[3].encode('utf-8')) != row[4] or
                    set(producer) != {'raw_path', 'features_path', 'native_path'}):
                raise ValueError('native cache producer audit identity drift')
            features_blob, native_blob = bytes(row[5]), bytes(row[6])
            if (_hash(features_blob) != metadata.get('sha256') or
                    _hash(native_blob) != metadata.get('native_sha256')):
                raise ValueError('native cache actual blob SHA drift')
            if _inspect_blobs(identity, features_blob, native_blob) != metadata:
                raise ValueError('native cache parsed schema/config/metadata drift')
        except (OSError, EOFError, TypeError, KeyError) as exc:
            raise ValueError('native cache stored content invalid') from exc
        return NativeCacheEntry(stored_identity, metadata, producer, features_blob, native_blob, key)

    def store(self, identity, record):
        _validate_identity(identity)
        if (any(record.get(k) != v for k, v in identity['raw'].items()) or
                any(record.get(k) != v for k, v in identity['liquidity'].items()) or
                record.get('raw_only_rebuild') is not True or record.get('reused_model_or_label') is not False):
            raise ValueError('native cache candidate raw/output identity mismatch')
        features_blob, native_blob = Path(record['path']).read_bytes(), Path(record['native_path']).read_bytes()
        metadata = _inspect_blobs(identity, features_blob, native_blob)
        for key in ('sha256', 'native_sha256', 'rows', 'native_config_hash', 'native_schema_hash', 'native_encoding'):
            if metadata[key] != record.get(key): raise ValueError('native cache candidate output identity mismatch: ' + key)
        key = _object_hash(identity)
        metadata_json = _canonical(metadata)
        producer_json = _canonical({'raw_path': str(record['raw_path']),
            'features_path': str(record['path']), 'native_path': str(record['native_path'])})
        with self.connection:
            self.connection.execute('''INSERT OR IGNORE INTO native_structure_entries
                (cache_key,identity_json,metadata_json,metadata_sha256,producer_json,producer_sha256,features_blob,native_blob)
                VALUES (?,?,?,?,?,?,?,?)''', (key, _canonical(identity), metadata_json,
                    _hash(metadata_json.encode('utf-8')), producer_json, _hash(producer_json.encode('utf-8')), features_blob, native_blob))
        result = self.lookup(identity)
        if (result.features_blob != features_blob or result.native_blob != native_blob or result.metadata != metadata):
            raise ValueError('native cache equivalent input produced different output; existing entry preserved')
        return result
