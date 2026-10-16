"""Read verified native cache evidence once per stock; aggregate current members.

This reader never creates a cache, replays a model, or grants training eligibility.
The producer is the explicitly selected, byte-verified current application root.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, time
import gzip
import copy
import hashlib
import io
import importlib
import json
import math
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

from my_strategy.core.paths import PROJECT_ROOT, DATA_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.services.board_calculation_scope import (CALCULATION_SCOPE, calculation_membership,
    calculation_contract, is_calculation_symbol)
from my_strategy.adapters.czsc_adapter import normalize_symbol, raw_path
from my_strategy.services.czsc_liquidity import RAW_IDENTITY_COLUMNS, daily_eligibility, liquidity_policy
from my_strategy.services.czsc_native_cache import (_RAW_KEYS, _LIQUIDITY_KEYS, _OUTPUT_KEYS,
    _validate_identity, make_native_cache_identity, read_native_payload)

NATIVE_CODE_FILES = tuple('services/' + name for name in (
    'czsc_trend_source.py', 'czsc_native_cache.py', 'czsc_market_features.py', 'czsc_market_profiles.py',
    'czsc_price_volume_features.py', 'czsc_price_volume_events.py', 'czsc_native_evidence.py',
    'czsc_analysis.py', 'czsc_trend_support.py', 'czsc_liquidity.py')) + (
    'adapters/czsc_adapter.py', 'configs/czsc_strategy.json')
SUMMARY_SCHEMA = dict(version='member_native_summary_v2', calculation_scope=CALCULATION_SCOPE, sources=['structure', 'wyckoff'],
    basis='verified_member_native_target_close', missing='null_not_zero', training_eligible=False,
    aggregation='equal_weight_eligible_current_members;each_source_separate;SH_SZ_A_projection_denominator')
SUMMARY_SCHEMA_HASH = stable_hash(SUMMARY_SCHEMA)
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}')
_SOURCES = ('structure', 'wyckoff')
_ZONE = ZoneInfo('Asia/Shanghai')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _parse(value):
    def reject(_): raise ValueError('nonfinite native JSON')
    return json.loads(value, parse_constant=reject)


def _sha(value): return hashlib.sha256(value).hexdigest()


def _file_sha(path): return _sha(Path(path).read_bytes())


def _app_root(value=None):
    root = Path(value or PROJECT_ROOT).expanduser().resolve()
    if (root / 'app' / 'my_strategy').is_dir(): root /= 'app'
    if not (root / 'my_strategy').is_dir(): raise ValueError('native producer application root missing')
    return root


def _close(day):
    if not isinstance(day, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day):
        raise ValueError('native summary price_date must be an ISO date')
    return datetime.combine(datetime.fromisoformat(day).date(), time(15), _ZONE)


def _instant(value):
    if not isinstance(value, str): raise ValueError('native evidence aware time missing')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset() is None: raise ValueError('native evidence timezone missing')
    return stamp


def _engine_contract(root):
    manifest_path = root / 'czsc-source-manifest.json'
    manifest = _parse(manifest_path.read_text())
    if manifest.get('package') != 'czsc' or not manifest.get('files'):
        raise ValueError('native producer engine manifest missing')
    vendor = (root / 'vendor/czsc').resolve(); observed = []
    for item in manifest['files']:
        relative = Path(item['path']); path = (vendor / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(vendor) or _file_sha(path) != item['sha256']:
            raise ValueError('native producer actual engine source drift')
        observed.append({'path': relative.as_posix(), 'sha256': item['sha256']})
    return dict(engine='confirmed_structure', source_tree_sha256=_sha(_json(observed).encode()),
        manifest_sha256=_file_sha(manifest_path), package='czsc', build_version=manifest.get('version'),
        capabilities=['causal_finished_pens', 'confirmed_zones', 'native_signal_calls'])


def _derive_schema(root, config):
    """Current reader schemas can only describe identical actual generating modules."""
    from my_strategy.services.czsc_native_evidence import build_native_evidence
    from my_strategy.services.czsc_market_features import FEATURE_SCHEMA_HASH
    from my_strategy.services.czsc_market_profiles import get_feature_profile
    # The cache reader's WAL plumbing may differ from the historical producer;
    # semantic generation modules must remain identical, never admitted by name.
    for name in NATIVE_CODE_FILES:
        if name in ('services/czsc_native_cache.py', 'services/czsc_trend_source.py', 'configs/czsc_strategy.json'):
            continue
        loaded = importlib.import_module('my_strategy.'+name[:-3].replace('/','.'))
        # Business path overrides do not identify the code already imported in
        # this interpreter. Bind its actual source file, independent of .env.
        if _file_sha(root / 'my_strategy' / name) != _file_sha(Path(loaded.__file__).resolve()):
            raise ValueError('native summary reader generating semantics differ from producer')
    empty = build_native_evidence(pd.DataFrame(), config, market_dates=[])
    semantics=copy.deepcopy(empty['semantics'])
    actual_engine=_engine_contract(root);reader_engine=semantics['engine_contract']
    # Source-identical deployments can have differently serialized manifest
    # files. Bind the explicitly verified producer's physical manifest, never
    # substitute the staging manifest into its native schema/config hash.
    if {k:v for k,v in actual_engine.items() if k!='manifest_sha256'}!={k:v for k,v in reader_engine.items() if k!='manifest_sha256'}:
        raise ValueError('native reader actual engine source contract differs from producer')
    semantics['engine_contract']=actual_engine
    schema_hash=stable_hash(semantics)
    config_hash=stable_hash(dict(native_schema_hash=schema_hash,native=semantics['config'],
        czsc=semantics['structure']['engine'],wyckoff_schema_hash=semantics['event_w']['event_input_schema_hash']))
    return dict(native_schema_hash=schema_hash, native_config_hash=config_hash,
        native_version=empty['version'], feature_schema_hashes=dict(market=FEATURE_SCHEMA_HASH,
            profile=get_feature_profile('ma_trend')['schema_hash']))


def native_generation_contract(*, source_root=None, config=None, frozen_code_hashes=None):
    """Read actual current producer bytes and engine, without reading cache/price rows."""
    from my_strategy.services.czsc_trend_contracts import strategy_config
    root = _app_root(source_root); cfg = strategy_config(config)
    code = {name: _file_sha(root / 'my_strategy' / name) for name in NATIVE_CODE_FILES}
    archived = {}
    if frozen_code_hashes is not None:
        if set(frozen_code_hashes) != set(code): raise ValueError('native frozen producer file set drift')
        for name, expected in frozen_code_hashes.items():
            if code[name] == expected: continue
            # Storage plumbing can evolve independently. Historical sources
            # retain their exact producer; numerical modules never use this.
            if name not in ('services/czsc_native_cache.py', 'services/czsc_trend_source.py') or not re.fullmatch(r'[a-f0-9]{64}', expected):
                raise ValueError('native producer current generating contract drift: code_hashes')
            registry = root / 'my_strategy/data/metadata/producer_snapshots'
            snapshot = registry / expected / name
            if snapshot.is_symlink() or not snapshot.resolve().is_relative_to(registry.resolve()) or not snapshot.is_file() or _file_sha(snapshot) != expected:
                raise ValueError('native historical storage producer snapshot missing or changed')
            code[name] = expected
            archived[name] = str(snapshot)
    result = dict(source_root=str(root), code_hashes=code, engine_contract=_engine_contract(root),
        **_derive_schema(root, cfg), config=cfg)
    if archived: result['historical_storage_producer_paths'] = archived
    result['contract_hash'] = stable_hash(result)
    return result


def _run_root(artifacts, identifier):
    if not isinstance(identifier, str) or not _ID.fullmatch(identifier): raise ValueError('native source run id invalid')
    result = (artifacts / 'runs' / identifier).resolve()
    if not result.is_relative_to((artifacts / 'runs').resolve()): raise ValueError('native source run path escapes artifact root')
    return result


def _load_contract(source_root, artifacts_root, native_id, raw_id, day, config):
    root = _app_root(source_root)
    artifacts = Path(artifacts_root or root / 'my_strategy/artifacts').expanduser().resolve()
    if native_id is None:
        candidates=[]
        for folder in (artifacts/'runs').iterdir():
            status_path=folder/'reports/native_prepare_status.json'
            if not folder.is_dir() or not status_path.is_file() or not (folder/'reports/native_source.json').is_file(): continue
            status=_parse(status_path.read_text())
            metadata_path=folder/'metadata.json'
            if status.get('status')=='completed' and metadata_path.is_file():
                metadata=_parse(metadata_path.read_text())
                if metadata.get('task')=='trend-native-source' and metadata.get('as_of_date')==day:
                    candidates.append(folder.name)
        if not candidates: raise ValueError('no completed native source at target date')
        native_id=max(candidates)
    native = _run_root(artifacts, native_id)
    header_path = native / 'reports/native_checkpoint.json'; header = _parse(header_path.read_text())
    checkpoint_hash = header.pop('checkpoint_hash', None)
    if header.get('format') != 'native_source_checkpoint' or header.get('run_id') != native_id or stable_hash(header) != checkpoint_hash:
        raise ValueError('native source checkpoint hash or run identity drift')
    identity = header['identity']; raw_id = raw_id or identity['raw_source_run_id']
    if raw_id != identity['raw_source_run_id']: raise ValueError('native summary raw source id differs from producer')
    cfg = header['config']
    if config is not None:
        from my_strategy.services.czsc_trend_contracts import strategy_config
        candidate = strategy_config(config)
        if _derive_schema(root, candidate) != _derive_schema(root, cfg) or candidate['feature_start'] != cfg['feature_start'] or candidate['liquidity_universe'] != cfg['liquidity_universe']:
            raise ValueError('native summary requested generating config differs from producer')
    generation = native_generation_contract(source_root=root, config=cfg, frozen_code_hashes=identity['code_hashes'])
    for key in ('code_hashes', 'engine_contract', 'feature_schema_hashes', 'native_config_hash', 'native_schema_hash'):
        if generation[key] != identity[key]: raise ValueError('native producer current generating contract drift: ' + key)
    if stable_hash(cfg) != identity['config_hash']: raise ValueError('native source config hash drift')
    expected_input=stable_hash(dict(feature_start=cfg['feature_start'],liquidity_universe=cfg['liquidity_universe'],
        native_schema_hash=generation['native_schema_hash'],native_config_hash=generation['native_config_hash']))
    if identity.get('native_input_contract_hash')!=expected_input:
        raise ValueError('native complete input configuration hash drift')
    raw = _run_root(artifacts, raw_id); raw_report_path = raw / 'reports/raw_source.json'
    raw_report = _parse(raw_report_path.read_text()); native_report_path = native / 'reports/native_source.json'
    native_report = _parse(native_report_path.read_text()); raw_body = dict(raw_report); raw_hash = raw_body.pop('snapshot_hash', None)
    if (raw_report.get('status') != 'completed' or raw_report.get('failures') or raw_report.get('run_id') != raw_id or
            raw_hash != stable_hash(raw_body) or raw_hash != identity['raw_snapshot_hash'] or
            _file_sha(raw_report_path) != identity['raw_report_sha256'] or
            stable_hash(raw_report['raw_records']) != identity['raw_records_hash'] or
            raw_report.get('data_end') != day or native_report.get('data_end') != day or
            native_report.get('status') != 'completed' or native_report.get('failures') or
            native_report.get('run_id') != native_id or native_report.get('native_checkpoint_hash') != checkpoint_hash):
        raise ValueError('native completed source/raw manifest identity drift')
    dates = raw_report['dates']
    if dates != sorted(set(dates)) or not dates or dates[-1] != day or stable_hash(dates) != identity['calendar_hash']:
        raise ValueError('native frozen calendar contract drift')
    raw_records = {item['symbol']: item for item in raw_report['raw_records']}
    native_records = {item['symbol']: item for item in native_report['dataset_records']}
    if (len(raw_records) != len(raw_report['raw_records']) or len(native_records) != len(native_report['dataset_records']) or
            sorted(raw_records) != sorted(identity['selected_symbols']) or set(native_records) != set(raw_records)):
        raise ValueError('native frozen population/record identity drift')
    for symbol, record in raw_records.items():
        if any(native_records[symbol].get(key) != value for key, value in record.items()):
            raise ValueError('native source record differs from frozen raw record')
    return dict(root=root, artifacts=artifacts, raw_id=raw_id, native_id=native_id, cfg=cfg,
        generation=generation, header=identity, records=raw_records, native_records=native_records, dates=dates,
        population=raw_report['liquidity'],calendar_run_id=raw_report.get('calendar_run_id'),
        raw_report_sha256=_file_sha(raw_report_path), native_report_sha256=_file_sha(native_report_path),
        checkpoint_hash=checkpoint_hash)


def _asset(path, contract):
    result = Path(path).expanduser().resolve()
    if not result.is_file() or not result.is_relative_to(contract['artifacts']):
        raise ValueError('native frozen asset missing or outside artifact root')
    return result


def _content_hash(frame):
    raw = frame[list(RAW_IDENTITY_COLUMNS)].copy(); raw['date'] = pd.to_datetime(raw.date).dt.strftime('%Y-%m-%d')
    return _sha(raw.to_csv(index=False, float_format='%.17g').encode())


def _verify_population_current(contract, connection, *, progress=None, check_cancel=None):
    population=contract['population'];catalog=population['input_catalog']
    if stable_hash(catalog)!=population['input_catalog_hash']:
        raise ValueError('native full raw candidate catalog hash drift')
    expected={row['symbol']:row for row in catalog}
    if len(expected)!=len(catalog) or sorted(expected)!=sorted(population['candidate_symbols']):
        raise ValueError('native full raw candidate population drift')
    if population.get('requested_symbols') is None:
        observed=[row[0] for row in connection.execute('SELECT DISTINCT stock FROM stock_daily_normalized WHERE date<=? ORDER BY stock',[contract['dates'][-1]])]
        mainboards=[symbol for symbol in observed if re.fullmatch(r'(?:60\d{4}\.SH|00\d{4}\.SZ)',symbol)]
        if mainboards!=sorted(expected):raise ValueError('native current warehouse candidate population changed')
    verified={};symbols=sorted(expected)
    for offset in range(0,len(symbols),256):
        if check_cancel:check_cancel()
        chunk=symbols[offset:offset+256];marks=','.join('?' for _ in chunk)
        frame=pd.read_sql_query('SELECT stock AS symbol,date,open,high,low,close,volume,amount,source,has_trade_price '
            f'FROM stock_daily_normalized WHERE stock IN ({marks}) AND date<=? ORDER BY stock,date',connection,
            params=[*chunk,contract['dates'][-1]])
        for symbol,current in frame.groupby('symbol',sort=True):
            digest=_content_hash(current)
            if digest!=expected[symbol]['raw_content_sha256'] or len(current)!=expected[symbol]['rows']:
                raise ValueError('native current full candidate raw prefix drift: '+symbol)
            verified[symbol]=digest
        if progress:progress(stage='native_raw_identity',current=len(verified),total=len(symbols),failed=0,actual_device='cpu')
    if set(verified)!=set(expected):raise ValueError('native current raw candidate data missing')
    return verified


def _verify_raw(record, contract, connection, current_hashes=None):
    path = _asset(record['raw_path'], contract)
    if _file_sha(path) != record['raw_sha256']: raise ValueError('native immutable raw physical SHA drift')
    frozen = pd.read_parquet(path)
    if len(frozen) != record['raw_rows'] or list(frozen) != record['raw_columns'] or _content_hash(frozen) != record['raw_content_sha256']:
        raise ValueError('native immutable raw content/columns drift')
    if current_hashes is None:
        current = pd.read_sql_query('SELECT stock AS symbol,date,open,high,low,close,volume,amount,source,has_trade_price '
            'FROM stock_daily_normalized WHERE stock=? AND date<=? ORDER BY date', connection,
            params=[record['symbol'], contract['dates'][-1]])
        digest=_content_hash(current)
    else:current=frozen;digest=current_hashes.get(record['symbol'])
    if digest != record['raw_content_sha256']: raise ValueError('native current full raw prefix differs from producer')
    # The producer deliberately clears parquet attrs. Restore only the frozen,
    # independently checked observation contract before repeating its liquidity
    # calculation; otherwise a verified shares/CNY input appears unitless.
    attrs = record.get('raw_attrs')
    required_attrs = {'price_basis','legacy_fuyao_bars','unverified_trade_price_bars',
        'volume_unit','amount_unit','data_version'}
    if not isinstance(attrs,dict) or not required_attrs <= set(attrs):
        raise ValueError('native raw observation attributes missing')
    from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
    legacy = int(frozen.source.map(legacy_fuyao_source).sum())
    unverified = int(frozen.has_trade_price.ne(1).sum())
    if (attrs['legacy_fuyao_bars'] != legacy or attrs['unverified_trade_price_bars'] != unverified or
            attrs['price_basis'] != ('mixed_legacy_adjustment_unverified' if legacy else 'unadjusted') or
            attrs['data_version'] != record['raw_series_hash'] or
            attrs['volume_unit'] != 'shares' or attrs['amount_unit'] != 'CNY'):
        raise ValueError('native raw observation attributes differ from frozen rows')
    current.attrs.update(attrs)
    frozen.attrs.update(attrs)
    daily_path = _asset(record['liquidity_path'], contract)
    if _file_sha(daily_path) != record['liquidity_sha256']: raise ValueError('native immutable liquidity SHA drift')
    daily = pd.read_parquet(daily_path)
    expected = daily_eligibility(current, contract['dates'], contract['cfg'],
        start=contract['cfg']['feature_start'], end=contract['dates'][-1])
    expected.insert(0, 'symbol', record['symbol']); expected.attrs = {}; daily.attrs = {}
    if len(daily) != record['liquidity_rows'] or int(daily.liquidity_pool_eligible.sum()) != record['liquidity_eligible_sessions'] or record['liquidity_policy_hash'] != stable_hash(liquidity_policy(contract['cfg'])):
        raise ValueError('native liquidity identity/count drift')
    pd.testing.assert_frame_equal(daily, expected, check_dtype=False)
    return frozen


def _read_cache_entry(connection, identity):
    # Cache keys use its compact JSON contract; RunContext stable_hash keeps
    # whitespace and is reserved for source/schema manifests.
    _validate_identity(identity); key = _sha(_json(identity).encode())
    row = connection.execute('SELECT * FROM native_structure_entries WHERE cache_key=?', [key]).fetchone()
    if row is None: raise ValueError('native_cache_entry_missing')
    stored = _parse(row['identity_json']); metadata = _parse(row['metadata_json']); producer = _parse(row['producer_json'])
    if stored != identity or _sha(_json(stored).encode()) != key: raise ValueError('native cached generating identity drift')
    if _sha(row['metadata_json'].encode()) != row['metadata_sha256'] or _sha(row['producer_json'].encode()) != row['producer_sha256'] or set(producer) != {'raw_path', 'features_path', 'native_path'}:
        raise ValueError('native cached metadata/producer SHA drift')
    features_blob, native_blob, _, _ = read_native_payload(connection, key, row['features_blob'], row['native_blob'])
    if _sha(features_blob) != metadata['sha256'] or _sha(native_blob) != metadata['native_sha256']:
        raise ValueError('native cached dual physical SHA drift')
    frame = pd.read_parquet(io.BytesIO(features_blob)); parquet = pq.ParquetFile(io.BytesIO(features_blob))
    pack = _parse(gzip.decompress(native_blob).decode())
    from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS
    required = set(RAW_IDENTITY_COLUMNS) | set(MA_TREND_COLUMNS) | {'native_schema_hash', 'native_version'}
    raw = identity['raw']; dates = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d').tolist()
    if (len(frame) != raw['raw_rows'] or not required <= set(frame) or _content_hash(frame) != raw['raw_content_sha256'] or
            not frame.symbol.eq(raw['symbol']).all() or dates != sorted(set(dates)) or
            dates[0] != raw['raw_data_start'] or dates[-1] != raw['data_end']):
        raise ValueError('native cached parquet full raw/schema/date drift')
    if (not isinstance(pack, dict) or set(pack) != _OUTPUT_KEYS or pack['schema_hash'] != identity['native_schema_hash'] or
            stable_hash(pack['semantics']) != identity['native_schema_hash'] or pack['config_hash'] != identity['native_config_hash'] or
            not isinstance(pack['version'], str) or not pack['version'] or
            not frame.native_schema_hash.eq(pack['schema_hash']).all() or not frame.native_version.eq(pack['version']).all()):
        raise ValueError('native cached evidence semantics/schema/config drift')
    provenance = pack['calendar_provenance']
    if (provenance.get('provided') is not True or provenance.get('calendar_prefix_hash') != identity['calendar']['prefix_hash'] or
            provenance.get('calendar_prefix_end') != identity['calendar']['prefix_end'] or
            provenance.get('future_calendar_entries_used_for_historical_qualification') is not False):
        raise ValueError('native cached calendar provenance drift')
    for name in ('evaluations', 'auxiliary_evaluations'):
        if not isinstance(pack[name], list) or [item.get('date') for item in pack[name]] != dates:
            raise ValueError('native cached complete evaluation causal rows drift')
    if not isinstance(pack['events'], list) or not isinstance(pack['wyckoff_events'], list):
        raise ValueError('native cached events schema drift')
    observed = dict(rows=len(frame), feature_columns=list(frame), parquet_schema=str(parquet.schema_arrow),
        native_version=pack['version'], native_config_hash=pack['config_hash'], native_schema_hash=pack['schema_hash'],
        sha256=_sha(features_blob), native_sha256=_sha(native_blob), native_encoding='gzip_json',
        raw_only_rebuild=True, reused_model_or_label=False)
    if observed != metadata: raise ValueError('native cached physical parsed metadata drift')
    return key, metadata, pack, producer


def _fact(fact, symbol, source, cutoff):
    if (not isinstance(fact, dict) or fact.get('symbol') != symbol or fact.get('source') != source or
            not isinstance(fact.get('event_id'), str) or not fact['event_id'] or
            fact.get('original_event_id', fact['event_id']) != fact['event_id'] or _instant(fact.get('available_at')) > cutoff):
        raise ValueError('native original event identity/time drift')
    fields = ('event_id', 'original_event_id', 'pattern_id', 'event_type', 'side', 'date', 'available_at',
        'first_available_at', 'first_confirmed_at', 'confirmed', 'input_eligible', 'reason_codes',
        'event_strength', 'lifecycle_state', 'support_low', 'resistance_high', 'invalidation_price')
    return {key: fact[key] for key in fields if key in fact}


def _evaluation(value, symbol, source, day):
    cutoff = _close(day)
    if (not isinstance(value, dict) or value.get('date') != day or value.get('source_id') != source or
            type(value.get('input_eligible')) is not bool or _instant(value.get('available_at')) > cutoff or
            value.get('direction_state') not in {'BUY', 'SELL', 'CONFLICT', 'ABSTAIN'}):
        raise ValueError('native target evaluation identity/time/eligibility drift')
    state = value.get('current_state'); eligible = value['input_eligible']
    if not isinstance(state, dict): raise ValueError('native current state missing')
    scores = {name: state.get(name) for name in ('buy_score', 'sell_score')}
    if eligible and any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 for score in scores.values()):
        raise ValueError('native eligible source score missing or invalid')
    supports = [_fact(item, symbol, source, cutoff) for item in value.get('support_facts', [])]
    ids = {item['event_id'] for item in supports}; fields = ('active_buy_event_ids', 'active_sell_event_ids', 'invalidated_event_ids', 'reason_codes')
    for name in fields:
        if not isinstance(value.get(name), list) or any(not isinstance(item, str) for item in value[name]):
            raise ValueError('native source original event ids or reasons invalid')
    if not set(value['active_buy_event_ids'] + value['active_sell_event_ids']) <= ids:
        raise ValueError('native active original event is absent from support facts')
    return dict(input_eligible=eligible, direction_state=value['direction_state'],
        buy_score=scores['buy_score'] if eligible else None, sell_score=scores['sell_score'] if eligible else None,
        buy_valid=value.get('buy_valid') if eligible else False, sell_valid=value.get('sell_valid') if eligible else False,
        support_facts=supports, available_at=value['available_at'], **{name: value[name] for name in fields})


def _unavailable(symbol, day, reason):
    value = dict(symbol=symbol, price_date=day, mode='current_reference', training_eligible=False,
        status='unavailable', reason_codes=[reason], schema_hash=SUMMARY_SCHEMA_HASH,
        native_cache_key=None, features_sha256=None, native_sha256=None, structure=None, wyckoff=None)
    value['row_hash'] = stable_hash(value)
    return value


def collect_native_summaries(*, price_date, symbols=None, cache_db_path=None, source_native_run_id=None,
        raw_source_run_id=None, source_root=None, artifacts_root=None, raw_db_path=None, config=None,
        progress=None, check_cancel=None, calculation_memberships=None):
    """Cold, complete dependency validation; no reads initialize or alter any DB.

    ``complete`` describes a completed requested population scan, not market
    coverage. Ineligible/missing rows remain explicit unavailable references.
    """
    _close(price_date); requested = sorted({normalize_symbol(value) for value in symbols}) if symbols is not None else None
    scope = calculation_contract(calculation_memberships or {})
    if requested is not None and any(not is_calculation_symbol(symbol) for symbol in requested):
        raise ValueError('native board collection requires SH/SZ A-share symbols')
    if calculation_memberships is not None and requested != scope['required_symbols']:
        raise ValueError('native requested population differs from calculation projections')
    summaries = {}; refs = []; failures = Counter(); contract = None; integrity_failures=Counter()
    base_identity = dict(format='native_summary_collection_v1', schema_hash=SUMMARY_SCHEMA_HASH,
        **scope,
        price_date=price_date, mode='current_reference', training_eligible=False,
        reader_code_sha256=_file_sha(Path(__file__)))
    try:
        if check_cancel: check_cancel()
        contract = _load_contract(source_root, artifacts_root, source_native_run_id, raw_source_run_id, price_date, config)
        requested = requested if requested is not None else sorted(symbol for symbol in contract['records'] if is_calculation_symbol(symbol))
        base_identity.update(generation_contract_hash=contract['generation']['contract_hash'],
            generation_contract=contract['generation'], source_native_run_id=contract['native_id'],
            raw_source_run_id=contract['raw_id'], native_report_sha256=contract['native_report_sha256'],
            raw_report_sha256=contract['raw_report_sha256'], checkpoint_hash=contract['checkpoint_hash'],
            calendar_hash=contract['header']['calendar_hash'],calendar_run_id=contract['calendar_run_id'])
        cache_path = Path(cache_db_path or DATA_ROOT / 'processed/czsc/native_structure.db').resolve()
        warehouse = raw_path(raw_db_path)
        if not cache_path.is_file() or not warehouse.is_file(): raise ValueError('native cache or current raw warehouse missing')
        with closing(sqlite3.connect(cache_path.as_uri()+'?mode=ro', uri=True, timeout=30)) as cache, closing(sqlite3.connect(warehouse.as_uri()+'?mode=ro', uri=True, timeout=30)) as raw:
            cache.row_factory = sqlite3.Row; cache.execute('PRAGMA query_only=ON'); raw.execute('PRAGMA query_only=ON')
            cache.execute('BEGIN'); raw.execute('BEGIN')
            if cache.execute('PRAGMA user_version').fetchone()[0] not in (1, 2):
                raise ValueError('native cache schema version drift')
            current_hashes=_verify_population_current(contract,raw,progress=progress,check_cancel=check_cancel)
            base_identity['raw_candidate_catalog_hash']=stable_hash(current_hashes)
            base_identity['required_symbols']=requested
            for current, symbol in enumerate(requested,1):
                if check_cancel: check_cancel()
                try:
                    record = contract['records'].get(symbol)
                    if record is None: raise ValueError('native_stock_outside_frozen_population')
                    _verify_raw(record, contract, raw,current_hashes)
                    expected = make_native_cache_identity(record, contract['header'], contract['dates'])
                    key, metadata, pack, producer = _read_cache_entry(cache, expected)
                    native_record = contract['native_records'][symbol]
                    if metadata['sha256'] != native_record['sha256'] or metadata['native_sha256'] != native_record['native_sha256']:
                        raise ValueError('native cache output differs from completed source manifest')
                    evaluation = next((item for item in pack['evaluations'] if item['date'] == price_date), None)
                    if evaluation is None or _instant(evaluation.get('available_at')) > _close(price_date):
                        raise ValueError('native_target_day_unavailable')
                    value = dict(symbol=symbol, price_date=price_date, mode='current_reference', training_eligible=False,
                        status='available', reason_codes=[], schema_hash=SUMMARY_SCHEMA_HASH, native_cache_key=key,
                        features_sha256=metadata['sha256'], native_sha256=metadata['native_sha256'],
                        raw_content_sha256=record['raw_content_sha256'], raw_sha256=record['raw_sha256'],
                        liquidity_sha256=record['liquidity_sha256'], calendar_hash=contract['header']['calendar_hash'],
                        native_schema_hash=metadata['native_schema_hash'], native_config_hash=metadata['native_config_hash'],
                        structure=_evaluation(evaluation['structure'], symbol, 'structure', price_date),
                        wyckoff=_evaluation(evaluation['wyckoff'], symbol, 'wyckoff', price_date),
                        events_at_day=[_fact(item, symbol, item['source'], _close(price_date)) for item in pack['events'] if item['date'] == price_date])
                    value['row_hash'] = stable_hash(value); summaries[symbol] = value
                    refs.append(dict(entity_type='stock', entity_id=symbol, ref_type='native', native_cache_key=key,
                        features_sha256=metadata['sha256'], native_sha256=metadata['native_sha256'],
                        raw_sha256=record['raw_sha256'], raw_content_sha256=record['raw_content_sha256'],
                        liquidity_sha256=record['liquidity_sha256'], row_hash=value['row_hash'], producer=producer))
                except (ValueError, KeyError, TypeError, OSError, sqlite3.Error, AssertionError) as exc:
                    reason = str(exc) if isinstance(exc, ValueError) else 'native_input_validation_failed:'+type(exc).__name__
                    failures[reason] += 1; summaries[symbol] = _unavailable(symbol, price_date, reason)
                    if symbol in contract['records'] and reason!='native_target_day_unavailable': integrity_failures[reason]+=1
                if progress: progress(stage='native_summary',current=current,total=len(requested),
                    failed=sum(integrity_failures.values()),verified=len(refs),actual_device='cpu')
            cache.rollback(); raw.rollback()
        base_identity['input_refs_hash'] = stable_hash(refs)
        return dict(summaries=summaries, input_refs=refs, identity=base_identity, complete=not integrity_failures,
            diagnostics=dict(requested=len(requested), verified=len(refs), unavailable=len(requested)-len(refs),
                reasons=dict(failures),integrity_failures=dict(integrity_failures)))
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        requested = requested or []; reason = 'native_generating_contract_unavailable:'+str(exc)
        return dict(summaries={symbol:_unavailable(symbol, price_date, reason) for symbol in requested},
            input_refs=[], identity=base_identity, complete=False,
            diagnostics=dict(requested=len(requested), verified=0, unavailable=len(requested), reasons={reason:len(requested)}))


def aggregate_board_native(snapshot, native_result, *, min_members=5, min_coverage=.8):
    """Aggregate the immutable SH/SZ calculation projection of a source snapshot."""
    if type(min_members) is not int or min_members<5 or isinstance(min_coverage, bool) or not isinstance(min_coverage,(int,float)) or not math.isfinite(min_coverage) or not .8<=min_coverage<=1:
        raise ValueError('native board minimum is five members and eighty-percent coverage')
    projection=calculation_membership(snapshot); members=projection['eligible_members']
    if not projection['source_complete'] or projection['source_expected_count']!=projection['source_member_count']:
        raise ValueError('native aggregate requires a complete unique current member snapshot')
    identity=native_result['identity']; day=identity['price_date']; _close(day); rows=native_result['summaries']
    if (identity.get('calculation_scope')!=CALCULATION_SCOPE or
            identity.get('projection_hashes',{}).get(snapshot['board_id'])!=projection['projection_hash'] or
            not set(members)<=set(identity.get('required_symbols',[]))):
        raise ValueError('native aggregate calculation projection identity drift')
    for symbol in members:
        row=rows.get(symbol)
        if row is not None:
            body={key:value for key,value in row.items() if key!='row_hash'}
            if row.get('row_hash')!=stable_hash(body) or row.get('symbol')!=symbol or row.get('price_date')!=day or row.get('training_eligible') is not False:
                raise ValueError('native per-stock summary row identity drift')
    result=dict(board_id=snapshot['board_id'],price_date=day,mode='current_reference',training_eligible=False,
        calculation_scope=CALCULATION_SCOPE,calculation_membership=projection,
        source_snapshot_hash=snapshot['snapshot_hash'],projection_hash=projection['projection_hash'],
        structure_basis='member_native',snapshot_hash=snapshot['snapshot_hash'],membership_available_at=snapshot['available_at'],
        expected_members=len(members),native_members=sum(rows.get(symbol,{}).get('status')=='available' for symbol in members),
        native_identity_hash=stable_hash(identity),schema_hash=SUMMARY_SCHEMA_HASH,
        policy=dict(min_members=min_members,min_coverage=min_coverage),reason_codes=[] if members else ['no_supported_members'])
    for source in _SOURCES:
        eligible=[rows[symbol][source] for symbol in members if rows.get(symbol,{}).get('status')=='available' and
            isinstance(rows[symbol].get(source),dict) and rows[symbol][source].get('input_eligible') is True]
        count=len(eligible); coverage=count/len(members) if members else 0.; qualified=count>=min_members and coverage>=min_coverage and native_result.get('complete') is True
        result[source+'_eligible_members']=count;result[source+'_coverage']=coverage;result[source+'_available']=qualified
        result[source+'_buy_score_mean']=sum(row['buy_score'] for row in eligible)/count if qualified else None
        result[source+'_sell_score_mean']=sum(row['sell_score'] for row in eligible)/count if qualified else None
        for direction in ('BUY','SELL','ABSTAIN','CONFLICT'):
            result[source+'_'+direction.lower()+'_ratio']=sum(row['direction_state']==direction for row in eligible)/count if qualified else None
        result[source+'_active_buy_event_ids']=sorted({event for row in eligible for event in row['active_buy_event_ids']})
        result[source+'_active_sell_event_ids']=sorted({event for row in eligible for event in row['active_sell_event_ids']})
        reasons=Counter(code for symbol in members for code in (rows.get(symbol,{}).get(source) or {}).get('reason_codes',[]))
        result[source+'_reason_counts']=dict(reasons)
        if not qualified: result['reason_codes'].append(source+'_eligible_coverage_below_minimum')
    result['status']='available' if result['structure_available'] and result['wyckoff_available'] else 'partial' if result['structure_available'] or result['wyckoff_available'] else 'unavailable'
    result['row_hash']=stable_hash(result)
    return result


def validate_native_reuse(identity, *, raw_db_path=None, source_root=None, artifacts_root=None,
        progress=None, check_cancel=None):
    """Revalidate indexed evidence dependencies without decoding native/cache blobs.

    An immutable Fluent index independently checks all summary/input-ref row SHAs.
    This gate checks the actual producer, manifests, full candidate raw prefixes,
    and frozen physical input assets. It does not claim a fresh cache blob audit.
    """
    try:
        if identity.get('schema_hash')!=SUMMARY_SCHEMA_HASH or identity.get('reader_code_sha256')!=_file_sha(Path(__file__)):
            raise ValueError('native indexed reader contract changed')
        if check_cancel:check_cancel()
        contract=_load_contract(source_root or identity['generation_contract']['source_root'],artifacts_root,
            identity['source_native_run_id'],identity['raw_source_run_id'],identity['price_date'],
            identity['generation_contract']['config'])
        fields={'generation_contract_hash':contract['generation']['contract_hash'],
            'raw_report_sha256':contract['raw_report_sha256'],'native_report_sha256':contract['native_report_sha256'],
            'checkpoint_hash':contract['checkpoint_hash'],'calendar_hash':contract['header']['calendar_hash'],
            'calendar_run_id':contract['calendar_run_id']}
        if any(identity.get(key)!=value for key,value in fields.items()):
            raise ValueError('native indexed producer/manifest contract changed')
        warehouse=raw_path(raw_db_path)
        with closing(sqlite3.connect(warehouse.as_uri()+'?mode=ro',uri=True,timeout=30)) as connection:
            connection.execute('PRAGMA query_only=ON');connection.execute('BEGIN')
            current=_verify_population_current(contract,connection,progress=progress,check_cancel=check_cancel)
            if stable_hash(current)!=identity.get('raw_candidate_catalog_hash'):
                raise ValueError('native indexed complete raw catalog identity changed')
            requested=identity.get('required_symbols')
            if not isinstance(requested,list):raise ValueError('native indexed requested population missing')
            count=0
            for symbol in requested:
                if check_cancel:check_cancel()
                record=contract['records'].get(symbol)
                if record is None:continue
                for path_key,sha_key in (('raw_path','raw_sha256'),('liquidity_path','liquidity_sha256')):
                    if _file_sha(_asset(record[path_key],contract))!=record[sha_key]:
                        raise ValueError('native indexed immutable input physical SHA drift')
                count+=1
            connection.rollback()
        return dict(valid=True,checked_native_inputs=count,checked_raw_candidates=len(current),
            native_blob_decode=False,reason=None)
    except (ValueError,KeyError,TypeError,OSError,sqlite3.Error) as exc:
        return dict(valid=False,native_blob_decode=False,reason=str(exc))
