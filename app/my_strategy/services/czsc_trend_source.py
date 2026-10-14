"""Fresh, immutable raw inputs and recomputed native evidence for reassessment."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import json
import hashlib
import gzip
import io
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import pandas as pd

from my_strategy.adapters.czsc_adapter import load_bars, raw_path
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_trend_support import _file_hash, _save, _json, resolve_frozen_asset
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, strategy_config
from my_strategy.services.czsc_liquidity import (RAW_IDENTITY_COLUMNS, LIQUIDITY_COLUMNS,
    liquidity_policy, daily_eligibility, annotate_liquidity)


NATIVE_GENERATING_FILES = tuple('services/' + name for name in (
    'czsc_trend_source.py', 'czsc_native_cache.py', 'czsc_market_features.py', 'czsc_market_profiles.py',
    'czsc_price_volume_features.py', 'czsc_price_volume_events.py', 'czsc_native_evidence.py',
    'czsc_analysis.py', 'czsc_trend_support.py', 'czsc_liquidity.py',
)) + ('adapters/czsc_adapter.py', 'configs/czsc_strategy.json')


def _mainboard(symbol):
    return bool(re.fullmatch(r'(?:60\d{4}\.SH|00\d{4}\.SZ)', str(symbol)))


def _raw_content_hash(frame):
    raw = frame[list(RAW_IDENTITY_COLUMNS)].copy()
    raw['date'] = pd.to_datetime(raw.date).dt.strftime('%Y-%m-%d')
    return hashlib.sha256(raw.to_csv(index=False, float_format='%.17g').encode('utf-8')).hexdigest()


def _save_native_pack(path, value):
    """Stream canonical JSON through deterministic standard gzip, atomically."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='\n') as output:
                json.dump(_json(value),output,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    temporary.replace(path)


def build_liquidity_universe(*, feature_start, end, calendar_run_id=None, policy=None,
                             preferred_symbols=None, db_path=None, _membership_directory=None):
    """Freeze the observable historical daily pool, never today's survivors.

    The warehouse, not its current securities/name table, supplies the observable
    stock basis.  Historical listing/ST/suspension/delisting identity is unknown
    unless separately verified; this object grants no formal publication status.
    ``preferred_symbols`` restricts an explicitly requested fixture/research
    basis, not a capacity-ranked sample.  No size limit or outcome is consulted.
    """
    from my_strategy.adapters.czsc_adapter import _connection, normalize_symbol
    from my_strategy.services.czsc_trend_support import _calendar
    cfg = liquidity_policy(policy)
    start = pd.Timestamp(feature_start).date().isoformat()
    end = pd.Timestamp(end).date().isoformat()
    dates, calendar_id = _calendar(calendar_run_id)
    dates = [date for date in dates if date <= end]
    if not dates or dates[-1] != end or start > end:
        raise ValueError('liquidity end must be a verified session after feature_start')
    task_dates = [date for date in dates if date >= start]
    if not task_dates:
        raise ValueError('empty verified liquidity task window')
    with _connection(db_path) as connection:
        schema = {row[1] for row in connection.execute('PRAGMA table_info(stock_daily_normalized)')}
        required = {'stock'} | (set(RAW_IDENTITY_COLUMNS) - {'symbol'})
        if not required <= schema:
            raise ValueError('raw liquidity warehouse schema missing: ' + ','.join(sorted(required - schema)))
        observed_end = connection.execute(
            'SELECT MAX(date) FROM stock_daily_normalized WHERE date<=?', [end]).fetchone()[0]
        if observed_end != end:
            raise ValueError('liquidity end exceeds available closed raw market observations')
        observable = sorted(row[0] for row in connection.execute(
            'SELECT DISTINCT stock FROM stock_daily_normalized WHERE date<=? ORDER BY stock', [end]))
        supported = [symbol for symbol in observable if _mainboard(symbol)]
        requested = None if preferred_symbols is None else sorted(set(normalize_symbol(value) for value in preferred_symbols))
        if requested is not None:
            if not requested or set(requested) - set(supported):
                raise ValueError('explicit liquidity basis contains unavailable or unsupported mainboard stocks')
            candidates = requested
        else:
            candidates = supported
        daily_members = {date: [] for date in task_dates}
        primary_counts = {}
        overlapping_counts = {}
        latest_exclusions = {}
        catalog = []
        selected = []
        latest = []
        stock_summaries = []
        membership_assets = []
        # The fixed read batch bounds memory only; all observed candidates are
        # visited and the same deterministic identity is independent of batching.
        for offset in range(0, len(candidates), 256):
            chunk = candidates[offset:offset + 256]
            marks = ','.join('?' for _ in chunk)
            frame = pd.read_sql_query(
                f'SELECT stock AS symbol,date,open,high,low,close,volume,amount,source,has_trade_price '
                f'FROM stock_daily_normalized WHERE stock IN ({marks}) AND date<=? ORDER BY stock,date',
                connection, params=[*chunk, end])
            for symbol, raw in frame.groupby('symbol', sort=True):
                raw = raw.reset_index(drop=True)
                raw.attrs.update(volume_unit='shares', amount_unit='CNY')
                daily = daily_eligibility(raw, dates, cfg, start=start, end=end)
                if _membership_directory is not None:
                    target = Path(_membership_directory) / (symbol.replace('.', '_') + '.parquet')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    frozen = daily.copy(); frozen.insert(0, 'symbol', symbol); frozen.attrs={}
                    frozen.to_parquet(target, index=False)
                    membership_assets.append({'symbol':symbol,'path':str(target),
                        'sha256':_file_hash(target),'rows':len(frozen)})
                qualified = daily.loc[daily.liquidity_pool_eligible, 'date'].tolist()
                if qualified:
                    selected.append(symbol)
                if bool(daily.iloc[-1].liquidity_pool_eligible):
                    latest.append(symbol)
                else:
                    status = daily.iloc[-1].liquidity_status
                    latest_exclusions[status] = latest_exclusions.get(status, 0) + 1
                for date in qualified:
                    daily_members[date].append(symbol)
                for status, count in daily.liquidity_status.value_counts().items():
                    if status != 'eligible':
                        primary_counts[status] = primary_counts.get(status, 0) + int(count)
                for reasons in daily.liquidity_reason_codes:
                    for reason in json.loads(reasons):
                        overlapping_counts[reason] = overlapping_counts.get(reason, 0) + 1
                content_hash = _raw_content_hash(raw)
                catalog.append({'symbol': symbol, 'rows': len(raw), 'raw_content_sha256': content_hash,
                    'data_start': str(pd.Timestamp(raw.date.iloc[0]).date()),
                    'data_end': str(pd.Timestamp(raw.date.iloc[-1]).date()),
                    'volume_unit': 'shares', 'amount_unit': 'CNY'})
                stock_summaries.append({'symbol': symbol, 'eligible_sessions': len(qualified),
                    'first_eligible_date': qualified[0] if qualified else None,
                    'last_eligible_date': qualified[-1] if qualified else None,
                    'latest_eligible': bool(daily.iloc[-1].liquidity_pool_eligible),
                    'latest_status': daily.iloc[-1].liquidity_status,
                    'latest_reason_codes': json.loads(daily.iloc[-1].liquidity_reason_codes)})
    selected = sorted(selected)
    latest = sorted(latest)
    membership_hash = stable_hash([{'date': date, 'symbols': sorted(daily_members[date])} for date in task_dates])
    report = {'population_mode': 'full_liquidity_union', 'feature_start': start, 'end': end,
        'policy': cfg, 'policy_hash': stable_hash(cfg), 'calendar_run_id': calendar_id,
        'calendar_hash': stable_hash(dates), 'observation_clock': 'verified_session_close',
        'basis_mode': 'explicit_observable_subset' if requested is not None else 'warehouse_observed_mainboard',
        'requested_symbols': requested, 'candidate_symbols': candidates,
        'observed_stock_count': len(observable), 'supported_stock_count': len(supported),
        'unsupported_stock_count': len(observable) - len(supported),
        'candidate_count': len(candidates), 'selected_symbols': selected, 'selected_count': len(selected),
        'latest_symbols': latest, 'latest_count': len(latest),
        'daily_counts': [{'date': date, 'eligible_count': len(daily_members[date]),
                          'excluded_count': len(candidates) - len(daily_members[date])} for date in task_dates],
        'exclusion_counts': dict(sorted(primary_counts.items())),
        'exclusion_counts_unit': 'candidate_stock_session_exclusive_primary_reason',
        'exclusion_reason_counts': dict(sorted(overlapping_counts.items())),
        'exclusion_reason_counts_overlap': True, 'latest_exclusion_counts': dict(sorted(latest_exclusions.items())),
        'input_catalog': catalog, 'input_catalog_hash': stable_hash(catalog),
        'daily_membership_hash': membership_hash, 'stock_summaries': stock_summaries,
        'historical_identity_status': 'listing_suspension_ST_delisting_population_not_independently_verified',
        'missing_observation_status': 'unknown_not_inferred_as_suspension',
        'publication_allowed': False}
    report['population_hash'] = stable_hash(report)
    if _membership_directory is not None:
        report['membership_assets'] = membership_assets
    return report


def _raw_worker(payload):
    symbol, end, directory, db_path = payload[:4]
    calendar, cfg, feature_start = payload[4:] if len(payload) > 4 else (None, None, None)
    try:
        frame = load_bars(symbol, end=end, as_of=end, db_path=db_path)
        columns = ['symbol','date','open','high','low','close','volume','amount','source','has_trade_price']
        if not set(columns).issubset(frame):
            raise ValueError('raw observation identity/source fields missing')
        raw = frame[columns].copy(); raw.attrs.update(frame.attrs)
        daily = None
        if calendar is not None:
            daily = daily_eligibility(raw, calendar, cfg, start=feature_start, end=end)
            raw = annotate_liquidity(raw, calendar, cfg, end=end)
        content_hash = _raw_content_hash(raw)
        raw.attrs={}
        target = Path(directory) / symbol.replace('.','_'); target.mkdir(parents=True, exist_ok=False)
        path = target / 'raw.parquet'; raw.to_parquet(path, index=False)
        record = {'symbol':symbol,'raw_path':str(path),'raw_sha256':_file_hash(path),
                'raw_columns':list(raw),'raw_content_sha256':content_hash,
                'raw_data_start':str(pd.Timestamp(raw.date.iloc[0]).date()),
                'data_end':str(pd.Timestamp(raw.date.iloc[-1]).date()),'raw_rows':len(raw),
                'raw_series_hash':frame.attrs['data_version'],
                'raw_attrs':{key:frame.attrs[key] for key in ['price_basis','legacy_fuyao_bars',
                    'unverified_trade_price_bars','volume_unit','amount_unit','data_version']}}
        if daily is not None:
            daily.insert(0, 'symbol', symbol)
            daily.attrs={}
            member_path=target/'liquidity.parquet';daily.to_parquet(member_path,index=False)
            record.update(liquidity_policy_hash=stable_hash(liquidity_policy(cfg)),
                liquidity_path=str(member_path),liquidity_sha256=_file_hash(member_path),
                liquidity_rows=len(daily),liquidity_eligible_sessions=int(daily.liquidity_pool_eligible.sum()))
        return record, None
    except Exception as exc:
        return None, {'symbol':symbol,'error':f'{type(exc).__name__}: {exc}'}


def _batch_size(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('stock preparation batch size must be positive')
    return value


def _population_body(population):
    return {key: value for key, value in population.items()
            if key not in {'population_hash', 'membership_assets', 'summary_assets'}}


def _verify_population(population):
    if (not isinstance(population,dict) or population.get('population_mode')!='full_liquidity_union' or
        population.get('population_hash')!=stable_hash(_population_body(population)) or
        population.get('policy_hash')!=stable_hash(liquidity_policy(population.get('policy'))) or
        population.get('input_catalog_hash')!=stable_hash(population.get('input_catalog'))):
        raise ValueError('frozen liquidity population identity drift')


def _map_batches(worker, payloads, *, cpu_workers, prepare_batch_size, stage,
                 on_result=None, initial_completed=0, total=None, reused=0,
                 initial_results=None, progress_details=None, progress_started=None,
                 progress_baseline=None, initial_processed=0):
    """Keep a bounded pool busy; publish only individually accepted results."""
    _batch_size(prepare_batch_size)
    if isinstance(cpu_workers,bool) or not isinstance(cpu_workers,int) or cpu_workers < 1:
        raise ValueError('cpu_workers must be positive')
    total = len(payloads) + initial_completed if total is None else total
    output = [None] * len(payloads)
    completed, failed, processed = initial_completed, 0, initial_processed
    baseline=initial_completed if progress_baseline is None else progress_baseline
    started = time.monotonic() if progress_started is None else progress_started
    maximum = max(1, min(2 * cpu_workers, prepare_batch_size))

    def publish(inflight):
        elapsed = time.monotonic() - started
        rate = (completed - baseline) / elapsed if elapsed > 0 else 0
        remaining = max(0, total - completed - failed)
        details = progress_details() if progress_details else {}
        print('SOURCE_BATCH ' + json.dumps(dict(stage=stage, completed=completed, total=total,
            failed=failed, reused=reused, inflight=inflight, processed=processed,
            elapsed_seconds=round(elapsed, 3), throughput_per_minute=round(rate * 60, 3),
            eta_seconds=0 if remaining == 0 else round(remaining / rate, 1) if rate > 0 else None,
            max_inflight=maximum, actual_device='cpu', **details)), flush=True)

    def accept(index, result, inflight):
        nonlocal completed, failed, processed
        output[index] = result
        accepted = on_result(result) if on_result else True
        completed += int(accepted is not False)
        failed += int(accepted is False)
        processed += 1
        publish(inflight)

    if initial_completed or not payloads or progress_details:
        publish(0)
    for result in initial_results or []:
        accepted = on_result(result) if on_result else True
        completed += int(accepted is not False)
        failed += int(accepted is False)
        processed += 1
        publish(0)
    if cpu_workers > 1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending, next_index = {}, 0
            while next_index < len(payloads) or pending:
                while next_index < len(payloads) and len(pending) < maximum:
                    future = pool.submit(worker, payloads[next_index])
                    pending[future] = next_index
                    next_index += 1
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    accept(index, future.result(), len(pending))
    else:
        for index, payload in enumerate(payloads):
            accept(index, worker(payload), 0)
    return output


def freeze_raw_source(*, symbols, end, calendar_run_id=None, cpu_workers=4, db_path=None,
                      selection_metadata=None, feature_start='2023-01-01', policy=None,
                      preflight_population=None, prepare_batch_size=360):
    from my_strategy.services.czsc_trend_support import _calendar
    dates, calendar_id = _calendar(calendar_run_id); dates=[d for d in dates if d <= end]
    if not dates or dates[-1] != end:
        raise ValueError('end must be an independently verified market session')
    _batch_size(prepare_batch_size)
    symbols=sorted(set(symbols)); path=raw_path(db_path);cfg=liquidity_policy(policy)
    if not symbols:
        raise ValueError('empty historical liquidity-qualified training union')
    if preflight_population is not None:
        _verify_population(preflight_population)
        requested = preflight_population.get('requested_symbols')
    else:
        requested = symbols
    temporary_memberships=tempfile.TemporaryDirectory(prefix='khquant-liquidity-freeze-')
    population=build_liquidity_universe(feature_start=feature_start,end=end,calendar_run_id=calendar_id,
        policy=cfg,preferred_symbols=requested,db_path=db_path,_membership_directory=temporary_memberships.name)
    if preflight_population is not None and population['population_hash']!=preflight_population['population_hash']:
        raise ValueError('raw population changed after preflight; a new preflight is required')
    if symbols!=population['selected_symbols']:
        raise ValueError('source symbols differ from the complete frozen historical liquidity union')
    context=create_run_context(task='trend-raw-source',as_of_date=end,
        config={'strategy_version':STRATEGY_ID,'selection':'historical_daily_eligible_union',
                'price_basis':'stored_raw_basis_with_row_source_qualification','calendar_run_id':calendar_id,'raw_db':str(path),
                'liquidity_policy':cfg,'population_hash':population['population_hash'],
                'prepare_batch_size':prepare_batch_size,'selection_metadata':selection_metadata or {}},
        stocks=symbols,scope='train',end_date=end)
    print('RAW_SOURCE_RUN_ID '+context.run_id,flush=True)
    member_directory=context.subdir('data_snapshots','liquidity')
    for asset in population['membership_assets']:
        target=member_directory/Path(asset['path']).name
        shutil.move(asset['path'],target);asset['path']=str(target)
    temporary_memberships.cleanup()
    payloads=[(symbol,end,str(context.subdir('data_snapshots','raw')),str(path),dates,cfg,feature_start) for symbol in symbols]
    output=_map_batches(_raw_worker,payloads,cpu_workers=cpu_workers,prepare_batch_size=prepare_batch_size,stage='raw',
        on_result=lambda result:result[0] is not None)
    records=[record for record,error in output if record is not None]
    failures=[error for record,error in output if error is not None]
    catalog={record['symbol']:record for record in population['input_catalog']}
    for record in records:
        if record['raw_content_sha256']!=catalog[record['symbol']]['raw_content_sha256']:
            failures.append({'symbol':record['symbol'],'error':'raw input changed during source freeze; repeat preflight'})
    frozen_members={asset['symbol']:asset for asset in population['membership_assets']}
    for record in records:
        if record['liquidity_sha256']!=frozen_members[record['symbol']]['sha256']:
            failures.append({'symbol':record['symbol'],'error':'daily qualification changed during source freeze'})
    summary_path=context.subdir('data_snapshots')/'liquidity_stocks.csv'
    pd.DataFrame(population['stock_summaries']).to_csv(summary_path,index=False)
    daily_path=context.subdir('data_snapshots')/'liquidity_daily_counts.csv'
    pd.DataFrame(population['daily_counts']).to_csv(daily_path,index=False)
    population['summary_assets']=[{'path':str(value),'sha256':_file_hash(value)} for value in [summary_path,daily_path]]
    source={'strategy_version':STRATEGY_ID,'status':'completed' if not failures else 'failed',
        'run_id':context.run_id,'end':end,'data_end':end,'selected_symbols':symbols,'raw_records':records,
        'failures':failures,'calendar_run_id':calendar_id,'calendar_hash':stable_hash(dates),
        'population_mode':'full_liquidity_union','liquidity':population,'liquidity_universe':population,
        'prepare_batch_size':prepare_batch_size,
        'dates':dates,'raw_only_rebuild':True,'reused_model_or_label':False,
        'selection_metadata':selection_metadata or {},
        'latest_observed_symbols':sum(record['data_end']==end for record in records),
        'price_basis_counts':{basis:sum(r['raw_attrs']['price_basis']==basis for r in records)
                             for basis in sorted({r['raw_attrs']['price_basis'] for r in records})},
        'publication_allowed':False,'recorded_at':local_now().isoformat()}
    source['snapshot_hash']=stable_hash(source)
    _save(context.subdir('reports')/'raw_source.json',source)
    _save(context.subdir('reports')/'liquidity_universe.json',population)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'source_run_id':calendar_id,'dates':dates})
    from my_strategy.storage.czsc_results import ResultStore
    ResultStore().save('trend-raw-source',_json(source))
    if failures:
        raise ValueError(f'raw source incomplete: {len(failures)}/{len(symbols)}; retained {context.run_id}')
    return context.run_dir(),source


def _native_worker(payload):
    from threadpoolctl import threadpool_limits
    from my_strategy.services.czsc_market_features import build_features
    from my_strategy.services.czsc_trend_support import _calendar_quality
    from my_strategy.services.czsc_market_profiles import bind_feature_profile, prepare_structure_inputs
    from my_strategy.services.czsc_native_evidence import (build_native_evidence,
        prepare_native_wyckoff_inputs, can_reuse_native_wyckoff)
    threadpool_limits(limits=1)
    record,cfg,calendar,directory=payload[:4]
    timings = payload[4] if len(payload) > 4 else {}
    started = phase = time.perf_counter_ns()
    path=resolve_frozen_asset(record['raw_path'])
    if _file_hash(path) != record['raw_sha256']:
        raise ValueError('immutable raw source drift')
    raw=pd.read_parquet(path);raw['date']=pd.to_datetime(raw.date)
    attrs=record.get('raw_attrs')
    if not isinstance(attrs,dict) or not set(['price_basis','legacy_fuyao_bars','unverified_trade_price_bars',
            'volume_unit','amount_unit','data_version']).issubset(attrs):
        raise ValueError('raw price-basis and observation identity metadata missing')
    from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
    legacy=int(raw.source.map(legacy_fuyao_source).sum());unverified=int(raw.has_trade_price.ne(1).sum())
    if (attrs['legacy_fuyao_bars']!=legacy or attrs['unverified_trade_price_bars']!=unverified or
            attrs['price_basis']!=('mixed_legacy_adjustment_unverified' if legacy else 'unadjusted') or
            attrs['data_version']!=record['raw_series_hash']):
        raise ValueError('raw basis does not match immutable observed source rows')
    raw.attrs.update(attrs)
    if 'liquidity_path' in record:
        if record.get('liquidity_policy_hash')!=stable_hash(liquidity_policy(cfg)):
            raise ValueError('native policy differs from frozen liquidity policy')
        member_path=resolve_frozen_asset(record['liquidity_path'])
        if _file_hash(member_path)!=record.get('liquidity_sha256'):
            raise ValueError('immutable daily liquidity membership drift')
        frozen=pd.read_parquet(member_path)
        recomputed=daily_eligibility(raw,calendar,cfg,start=frozen.date.iloc[0],end=calendar[-1])
        recomputed.insert(0,'symbol',record['symbol']);recomputed.attrs={};frozen.attrs={}
        try:
            pd.testing.assert_frame_equal(frozen,recomputed,check_dtype=False)
        except AssertionError as exc:
            raise ValueError('daily liquidity membership differs from frozen raw observations') from exc
        raw=annotate_liquidity(raw,calendar,cfg,end=calendar[-1],verify_existing=True)
        if _raw_content_hash(raw)!=record.get('raw_content_sha256'):
            raise ValueError('raw content identity differs from frozen population')
    timings['raw_verify_ns'] = time.perf_counter_ns() - phase
    phase = time.perf_counter_ns()
    features=bind_feature_profile(_calendar_quality(build_features(raw),raw,calendar),'ma_trend')
    features=prepare_structure_inputs(features)
    timings['features_ns'] = time.perf_counter_ns() - phase
    phase = time.perf_counter_ns()
    prepared=prepare_native_wyckoff_inputs(features,raw=raw,market_dates=calendar)
    features=prepared.frame
    timings['wyckoff_ns'] = time.perf_counter_ns() - phase
    for column in raw:
        if column not in features:features[column]=raw[column].to_numpy(copy=True)
    phase = time.perf_counter_ns()
    pack=build_native_evidence(features,cfg,market_dates=calendar,
        prepared_wyckoff=prepared if can_reuse_native_wyckoff(prepared,cfg) else None)
    timings['native_ns'] = time.perf_counter_ns() - phase
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=directory/record['symbol'].replace('.','_')
    if target.exists(): raise ValueError('uncommitted or existing native output directory')
    temporary=Path(tempfile.mkdtemp(prefix='.work-'+record['symbol'].replace('.','_')+'-',dir=directory))
    try:
        phase = time.perf_counter_ns()
        numeric=pack['frame'].drop(columns=[c for c in pack['frame'] if pack['frame'][c].map(lambda v:isinstance(v,(dict,list,tuple))).any()])
        numeric.attrs={};numeric.to_parquet(temporary/'features.parquet',index=False)
        timings['parquet_ns'] = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        _save_native_pack(temporary/'native.json.gz',{k:v for k,v in pack.items() if k!='frame'})
        timings['json_gzip_ns'] = time.perf_counter_ns() - phase
        temporary.replace(target)
    finally:
        if temporary.exists(): shutil.rmtree(temporary)
    native_path=target/'native.json.gz'
    phase = time.perf_counter_ns()
    parquet_sha, native_sha = _file_hash(target/'features.parquet'), _file_hash(native_path)
    timings['output_hash_ns'] = time.perf_counter_ns() - phase
    timings['total_ns'] = time.perf_counter_ns() - started
    return {**record,'path':str(target/'features.parquet'),'sha256':parquet_sha,
        'native_path':str(native_path),'native_sha256':native_sha,'native_encoding':'gzip_json',
        'native_config_hash':pack['config_hash'],'native_schema_hash':pack['schema_hash'],'rows':len(numeric),
        'raw_only_rebuild':True,'reused_model_or_label':False}


def _source_run_root(run_id):
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}',run_id):
        raise ValueError('invalid native/raw source run identifier')
    root=(ARTIFACT_RUNS_ROOT/run_id).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):
        raise ValueError('native/raw source escapes run storage')
    return root


def _native_code_hashes():
    return {name:_file_hash(Path(__file__).resolve().parents[1]/name) for name in NATIVE_GENERATING_FILES}


def _load_native_raw_source(raw_source_run_id, cfg):
    base=_source_run_root(raw_source_run_id)
    source=json.loads((base/'reports/raw_source.json').read_text())
    if (source.get('run_id')!=raw_source_run_id or source.get('strategy_version')!=STRATEGY_ID or
            source.get('status')!='completed' or source.get('failures')):
        raise ValueError('a complete new raw-only current snapshot is required')
    if source['snapshot_hash']!=stable_hash({k:v for k,v in source.items() if k!='snapshot_hash'}):
        raise ValueError('raw snapshot identity drift')
    population=source.get('liquidity')
    _verify_population(population)
    if source.get('population_mode')!='full_liquidity_union' or population['policy']!=liquidity_policy(cfg):
        raise ValueError('native preparation requires the same complete historical liquidity policy')
    if source['selected_symbols']!=population['selected_symbols'] or sorted(r['symbol'] for r in source['raw_records'])!=population['selected_symbols']:
        raise ValueError('raw records do not cover the complete historical liquidity union')
    assets=population.get('membership_assets')
    if not isinstance(assets,list) or sorted(asset.get('symbol','') for asset in assets)!=population['candidate_symbols']:
        raise ValueError('daily liquidity assets do not cover the observed population')
    memberships={item['date']:[] for item in population['daily_counts']}
    for asset in assets:
        path=resolve_frozen_asset(asset['path'])
        if _file_hash(path)!=asset['sha256']:
            raise ValueError('immutable liquidity population asset drift')
        daily=pd.read_parquet(path,columns=['symbol','date','liquidity_pool_eligible','liquidity_policy_hash'])
        if (len(daily)!=asset['rows'] or not daily.symbol.eq(asset['symbol']).all() or
            not daily.liquidity_policy_hash.eq(population['policy_hash']).all() or
            daily.date.tolist()!=list(memberships)):
            raise ValueError('frozen liquidity membership schema/identity drift')
        for date in daily.loc[daily.liquidity_pool_eligible,'date']:
            memberships[date].append(asset['symbol'])
    if stable_hash([{'date':date,'symbols':sorted(symbols)} for date,symbols in memberships.items()])!=population['daily_membership_hash']:
        raise ValueError('frozen daily liquidity membership hash drift')
    return base, source


def _checkpoint_header(run_id, raw_id, source, cfg, cpu_workers, prepare_batch_size, code):
    from my_strategy.services.czsc_native_evidence import build_native_evidence
    from my_strategy.adapters.czsc_adapter import engine_contract
    from my_strategy.services.czsc_market_features import FEATURE_SCHEMA_HASH
    from my_strategy.services.czsc_market_profiles import get_feature_profile
    empty=build_native_evidence(pd.DataFrame(),cfg,market_dates=[])
    identity=dict(strategy_version=STRATEGY_ID,raw_source_run_id=raw_id,
        raw_report_sha256=_file_hash(_source_run_root(raw_id)/'reports/raw_source.json'),
        raw_snapshot_hash=source['snapshot_hash'],raw_records_hash=stable_hash(source['raw_records']),
        config_hash=stable_hash(cfg),native_config_hash=empty['config_hash'],native_schema_hash=empty['schema_hash'],
        native_input_contract_hash=stable_hash(dict(feature_start=cfg['feature_start'],
            liquidity_universe=cfg['liquidity_universe'],native_schema_hash=empty['schema_hash'],
            native_config_hash=empty['config_hash'])),
        code_hashes=code,calendar_hash=source['calendar_hash'],population_hash=source['liquidity']['population_hash'],
        selected_symbols=source['selected_symbols'],cpu_workers=cpu_workers,prepare_batch_size=prepare_batch_size,
        engine_contract=engine_contract(),feature_schema_hashes=dict(market=FEATURE_SCHEMA_HASH,
            profile=get_feature_profile('ma_trend')['schema_hash']))
    header=dict(format='native_source_checkpoint',run_id=run_id,config=cfg,identity=identity,
        created_at=local_now().isoformat())
    header['checkpoint_hash']=stable_hash(header)
    return header


def _stock_checkpoint_identity(header, raw):
    identity=header['identity']
    return dict(run_id=header['run_id'],identity_hash=stable_hash(identity),symbol=raw['symbol'],
        raw_sha256=raw['raw_sha256'],raw_record_hash=stable_hash(raw),config_hash=identity['config_hash'],
        native_config_hash=identity['native_config_hash'],native_schema_hash=identity['native_schema_hash'],
        code_hashes=identity['code_hashes'])


def _verify_native_record(record, raw, header, *, verify_json=True, output_directory=None):
    import pyarrow.parquet as pq
    if not isinstance(record,dict) or any(record.get(key)!=value for key,value in raw.items()):
        raise ValueError('checkpoint raw observation record identity drift')
    identity=header['identity'];root=_source_run_root(header['run_id'])
    if (record.get('native_schema_hash')!=identity['native_schema_hash'] or
            record.get('native_config_hash')!=identity['native_config_hash'] or
            record.get('raw_only_rebuild') is not True or record.get('reused_model_or_label') is not False):
        raise ValueError('checkpoint native schema/config identity drift')
    raw_path=resolve_frozen_asset(raw['raw_path'])
    if _file_hash(raw_path)!=raw['raw_sha256'] or pq.ParquetFile(raw_path).metadata.num_rows!=raw['raw_rows']:
        raise ValueError('checkpoint immutable raw SHA or row count drift')
    target=(root/'data_snapshots/native' if output_directory is None else Path(output_directory))/raw['symbol'].replace('.','_')
    for field,sha,filename in [('path','sha256','features.parquet'),('native_path','native_sha256','native.json.gz')]:
        path=resolve_frozen_asset(record[field])
        if path.resolve()!=(target/filename).resolve() or not path.resolve().is_relative_to(root):
            raise ValueError('checkpoint native output boundary drift')
        if _file_hash(path)!=record[sha]: raise ValueError('checkpoint native output SHA drift')
    rows=pq.ParquetFile(target/'features.parquet').metadata.num_rows
    if rows!=record.get('rows') or rows!=raw['raw_rows']:
        raise ValueError('checkpoint native Parquet row count drift')
    if verify_json:
        with gzip.open(target/'native.json.gz','rt',encoding='utf-8') as stream: pack=json.load(stream)
        if pack.get('schema_hash')!=identity['native_schema_hash'] or pack.get('config_hash')!=identity['native_config_hash']:
            raise ValueError('checkpoint native JSON schema/config drift')


def _verify_stock_checkpoint(checkpoint, raw, header, *, verify_json=True):
    if (not isinstance(checkpoint,dict) or checkpoint.get('checkpoint_hash')!=stable_hash(
            {k:v for k,v in checkpoint.items() if k!='checkpoint_hash'}) or
            checkpoint.get('identity')!=_stock_checkpoint_identity(header,raw)):
        raise ValueError('native stock checkpoint identity/hash drift')
    timings=checkpoint.get('timings_ns')
    if not isinstance(timings,dict) or any(type(value) is not int or value<0 for value in timings.values()):
        raise ValueError('native checkpoint performance timing invalid')
    if checkpoint.get('status')=='completed':
        _verify_native_record(checkpoint.get('record'),raw,header,verify_json=verify_json)
    elif checkpoint.get('status')!='failed' or not isinstance(checkpoint.get('error'),str):
        raise ValueError('native checkpoint outcome invalid')
    return checkpoint


def verify_native_checkpoint(native_run_id, *, raw_source_run_id=None, config=None,
                             cpu_workers=None, prepare_batch_size=None):
    """Cold verification only. Legacy partial directories have no resumable identity."""
    root=_source_run_root(native_run_id);path=root/'reports/native_checkpoint.json'
    if not path.is_file(): raise ValueError('native source has no strict checkpoint; legacy partial sources cannot resume')
    header=json.loads(path.read_text())
    if (not isinstance(header,dict) or header.get('format')!='native_source_checkpoint' or
            header.get('run_id')!=native_run_id or header.get('checkpoint_hash')!=stable_hash(
                {k:v for k,v in header.items() if k!='checkpoint_hash'})):
        raise ValueError('native checkpoint header identity/hash drift')
    saved=header.get('identity')
    if not isinstance(saved,dict) or not isinstance(header.get('config'),dict):
        raise ValueError('native checkpoint identity/config invalid')
    raw_id=raw_source_run_id or saved['raw_source_run_id']
    cfg=strategy_config(header['config'] if config is None else config)
    workers=saved['cpu_workers'] if cpu_workers is None else cpu_workers
    batch=saved['prepare_batch_size'] if prepare_batch_size is None else prepare_batch_size
    base,source=_load_native_raw_source(raw_id,cfg)
    current=_checkpoint_header(native_run_id,raw_id,source,cfg,workers,batch,_native_code_hashes())
    if current['identity']!=saved or cfg!=header['config']:
        raise ValueError('native resume raw/config/schema/code/resources identity drift; a new source is required')
    raw_by_symbol={record['symbol']:record for record in source['raw_records']}
    completed,failures,checkpoints={},{},{}
    directory=root/'data_snapshots/checkpoints'
    for checkpoint_path in directory.glob('*.json'):
        checkpoint=json.loads(checkpoint_path.read_text());symbol=(checkpoint.get('identity') or {}).get('symbol')
        if symbol not in raw_by_symbol or checkpoint_path.name!=symbol.replace('.','_')+'.json':
            raise ValueError('native checkpoint contains an unknown or mismatched stock')
        checkpoint=_verify_stock_checkpoint(checkpoint,raw_by_symbol[symbol],header)
        checkpoints[symbol]=checkpoint
        if checkpoint['status']=='completed': completed[symbol]=checkpoint['record']
        else: failures[symbol]=dict(symbol=symbol,error=checkpoint['error'])
    report_path=root/'reports/native_source.json'
    completed_report=False
    if report_path.is_file():
        report=json.loads(report_path.read_text());completed_report=report.get('status')=='completed'
        if completed_report and (len(completed)!=len(raw_by_symbol) or failures or
                report.get('dataset_records')!=[completed[symbol] for symbol in source['selected_symbols']]):
            raise ValueError('completed native report differs from its actual checkpoints')
    return dict(header=header,raw_source_run_id=raw_id,completed_records=completed,
        failures=list(failures.values()),completed=len(completed),reused=len(completed),total=len(raw_by_symbol),
        completed_report=completed_report,checkpoints=checkpoints)


def _save_stock_checkpoint(root, checkpoint):
    checkpoint['checkpoint_hash']=stable_hash({k:v for k,v in checkpoint.items() if k!='checkpoint_hash'})
    checkpoint_directory=root/'data_snapshots/checkpoints'
    symbol=checkpoint['identity']['symbol'].replace('.','_')
    attempt=checkpoint_directory/'attempts'/(symbol+'-'+str(time.time_ns())+'.json')
    _save(attempt,checkpoint)
    _save(checkpoint_directory/(symbol+'.json'),checkpoint)
    return checkpoint


def _checkpoint_native_worker(payload):
    raw,cfg,calendar,directory,header,attempt_id=payload
    root=_source_run_root(header['run_id']);timings={}
    work_directory=root/'data_snapshots/native_attempts'/attempt_id/'native'
    checkpoint=dict(identity=_stock_checkpoint_identity(header,raw),timings_ns=timings,
        recorded_at=local_now().isoformat())
    try:
        if _native_code_hashes()!=header['identity']['code_hashes']:
            raise ValueError('native generating code changed during preparation')
        record=_native_worker((raw,cfg,calendar,str(work_directory),timings))
        phase=time.perf_counter_ns()
        _verify_native_record(record,raw,header,verify_json=False,output_directory=work_directory)
        if _native_code_hashes()!=header['identity']['code_hashes']:
            raise ValueError('native generating code changed during preparation')
        timings['checkpoint_verify_ns']=time.perf_counter_ns()-phase
        checkpoint.update(status='completed',record=record)
    except Exception as exc:
        checkpoint.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    with _native_commit_lock(root):
        _verify_preparation_attempt(root,attempt_id)
        if checkpoint['status']=='completed':
            computed=work_directory/raw['symbol'].replace('.','_')
            target=root/'data_snapshots/native'/computed.name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise ValueError('native attempt cannot overwrite a committed output')
            computed.replace(target)
            record['path']=str(target/'features.parquet');record['native_path']=str(target/'native.json.gz')
        checkpoint['attempt_id']=attempt_id
        return _save_stock_checkpoint(root,checkpoint)


@contextmanager
def _native_file_lock(root, filename, *, blocking=False):
    """Persistent lock inode; the OS releases ownership after process exit."""
    path=Path(root)/'reports'/filename
    if not path.parent.is_dir(): raise ValueError('native source run does not exist')
    handle=os.fdopen(os.open(path,os.O_RDWR|os.O_CREAT,0o600),'r+b');locked=False
    try:
        if path.stat().st_size==0:
            handle.write(b'\0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|(0 if blocking else fcntl.LOCK_NB))
            locked=True
        except OSError as exc:
            raise ValueError('native source preparation is already running in another process; cannot resume concurrently') from exc
        yield
    finally:
        try:
            if locked:
                handle.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        finally:
            handle.close()


def _native_prepare_lock(root):
    return _native_file_lock(root,'native_preparation.lock')


def _native_commit_lock(root):
    return _native_file_lock(root,'native_commit.lock',blocking=True)


def _verify_preparation_attempt(root, attempt_id):
    status=json.loads((root/'reports/native_prepare_status.json').read_text())
    parent=multiprocessing.parent_process()
    owner_alive=(parent.is_alive() and parent.pid==status.get('owner_pid')) if parent is not None else status.get('owner_pid')==os.getpid()
    if (status.get('attempt_id')!=attempt_id or status.get('status')!='running' or not owner_alive):
        raise ValueError('expired native preparation attempt cannot publish outputs or checkpoints')


def prepare_native_source(raw_source_run_id, *, config=None, cpu_workers=4, prepare_batch_size=360,
                          resume_native_run_id=None, cache_db_path=None):
    if type(cpu_workers) is not int or cpu_workers<1: raise ValueError('cpu_workers must be positive')
    _batch_size(prepare_batch_size)
    context=None;raw_input=None
    if resume_native_run_id:
        root=_source_run_root(resume_native_run_id)
    else:
        cfg=strategy_config(config);base,source=_load_native_raw_source(raw_source_run_id,cfg)
        raw_input=(base,source,cfg)
        context=create_run_context(task='trend-native-source',as_of_date=source['end'],
            config={'strategy_version':STRATEGY_ID,'raw_source_run_id':raw_source_run_id,'source_config':cfg,
                'prepare_batch_size':prepare_batch_size,'cpu_workers':cpu_workers},
            stocks=source['selected_symbols'],scope='train',data_version=source['snapshot_hash'],end_date=source['end'])
        root=context.run_dir()
    with _native_prepare_lock(root):
        return _prepare_native_source_unlocked(raw_source_run_id,config=config,cpu_workers=cpu_workers,
            prepare_batch_size=prepare_batch_size,resume_native_run_id=resume_native_run_id,
            cache_db_path=cache_db_path,_context=context,_raw_input=raw_input)


def _prepare_native_source_unlocked(raw_source_run_id, *, config=None, cpu_workers=4, prepare_batch_size=360,
                          resume_native_run_id=None, cache_db_path=None, _context=None, _raw_input=None):
    from my_strategy.services.czsc_native_cache import NativeStructureCache, make_native_cache_identity
    if type(cpu_workers) is not int or cpu_workers<1: raise ValueError('cpu_workers must be positive')
    _batch_size(prepare_batch_size)
    if _raw_input is None:
        cfg=strategy_config(config);base,source=_load_native_raw_source(raw_source_run_id,cfg)
    else:
        base,source,cfg=_raw_input
    population=source['liquidity'];code=_native_code_hashes()
    completed,previous_checkpoints={},{}
    attempt_id=str(time.time_ns())
    if resume_native_run_id:
        run_id=resume_native_run_id;root=_source_run_root(run_id)
        with _native_commit_lock(root):
            proof=verify_native_checkpoint(resume_native_run_id,raw_source_run_id=raw_source_run_id,config=cfg,
                cpu_workers=cpu_workers,prepare_batch_size=prepare_batch_size)
            if proof['completed_report']: raise ValueError('completed native sources must use source-native-run-id, not resume')
            header=proof['header'];completed=proof['completed_records'];previous_checkpoints=proof['checkpoints']
            _save(root/'reports/native_prepare_status.json',dict(status='running',run_id=run_id,
                completed=len(completed),total=len(source['raw_records']),reused=len(completed),
                attempt_id=attempt_id,owner_pid=os.getpid(),started_at=local_now().isoformat()))
    else:
        context=_context
        run_id=context.run_id;root=context.run_dir()
        header=_checkpoint_header(run_id,raw_source_run_id,source,cfg,cpu_workers,prepare_batch_size,code)
        with _native_commit_lock(root):
            _save(root/'reports/native_checkpoint.json',header)
            _save(root/'reports/native_prepare_status.json',dict(status='running',run_id=run_id,
                completed=0,total=len(source['raw_records']),reused=0,attempt_id=attempt_id,
                owner_pid=os.getpid(),started_at=local_now().isoformat()))
    print('NATIVE_SOURCE_RUN_ID '+run_id,flush=True)
    reused=len(completed)
    _save(root/'data_snapshots/calendar.json',{'verified':True,'source_run_id':source['calendar_run_id'],'dates':source['dates']})
    pending=[record for record in source['raw_records'] if record['symbol'] not in completed]
    pending.sort(key=lambda record:(-record['raw_rows'],record['symbol']))
    # A directory without a success checkpoint is uncommitted, never reusable.
    # Retain it explicitly, then rebuild in a fresh temporary directory.
    with _native_commit_lock(root):
        _verify_preparation_attempt(root,attempt_id)
        for record in pending:
            orphan=root/'data_snapshots/native'/record['symbol'].replace('.','_')
            if orphan.exists():
                quarantine=root/'data_snapshots/uncommitted';quarantine.mkdir(parents=True,exist_ok=True)
                orphan.replace(quarantine/(orphan.name+'-'+str(time.time_ns())))
    raw_by_symbol={record['symbol']:record for record in source['raw_records']}
    failures={};checkpoints=dict(previous_checkpoints)
    started=time.monotonic()
    cache_stats=dict(cache_hits=0,cache_misses=0,cache_lookup_ns=0,cache_store_ns=0,cache_materialize_ns=0)
    cache_identities={symbol:make_native_cache_identity(raw,header['identity'],source['dates'])
        for symbol,raw in raw_by_symbol.items()}

    def accept(checkpoint):
        symbol=checkpoint['identity']['symbol']
        _verify_stock_checkpoint(checkpoint,raw_by_symbol[symbol],header,verify_json=False)
        checkpoints[symbol]=checkpoint
        if checkpoint['status']=='completed':
            if checkpoint.get('origin')!='structure_cache':
                phase=time.perf_counter_ns()
                cache.store(cache_identities[symbol],checkpoint['record'])
                cache_stats['cache_store_ns']+=time.perf_counter_ns()-phase
            completed[symbol]=checkpoint['record'];return True
        failures[symbol]=dict(symbol=symbol,error=checkpoint['error']);return False

    state_path=root/'reports/native_prepare_status.json'
    try:
        with NativeStructureCache(cache_db_path) as cache:
            # A worker can commit its checkpoint just before parent cancellation.
            # Ensure restored success records also reach the verified parent-only cache.
            for symbol,record in completed.items():
                phase=time.perf_counter_ns();cache.store(cache_identities[symbol],record)
                cache_stats['cache_store_ns']+=time.perf_counter_ns()-phase
            compute_records=[]
            for raw in pending:
                symbol=raw['symbol'];phase=time.perf_counter_ns()
                entry=cache.lookup(cache_identities[symbol])
                lookup_ns=time.perf_counter_ns()-phase
                cache_stats['cache_lookup_ns']+=lookup_ns
                if entry is None:
                    cache_stats['cache_misses']+=1;compute_records.append(raw);continue
                cache_stats['cache_hits']+=1;phase=time.perf_counter_ns()
                record=entry.materialize(root/'data_snapshots/native',raw)
                _verify_native_record(record,raw,header,verify_json=False)
                materialize_ns=time.perf_counter_ns()-phase
                cache_stats['cache_materialize_ns']+=materialize_ns
                checkpoint=dict(identity=_stock_checkpoint_identity(header,raw),status='completed',record=record,
                    origin='structure_cache',recorded_at=local_now().isoformat(),
                    timings_ns=dict(cache_lookup_ns=lookup_ns,cache_materialize_ns=materialize_ns,
                        total_ns=lookup_ns+materialize_ns))
                checkpoint=_save_stock_checkpoint(root,checkpoint)
                accept(checkpoint)
                elapsed=time.monotonic()-started;rate=(len(completed)-reused)/elapsed if elapsed>0 else 0
                remaining=max(0,len(raw_by_symbol)-len(completed))
                print('SOURCE_BATCH '+json.dumps(dict(stage='native',completed=len(completed),total=len(raw_by_symbol),
                    failed=len(failures),reused=reused,inflight=0,processed=cache_stats['cache_hits'],
                    elapsed_seconds=round(elapsed,3),throughput_per_minute=round(rate*60,3),
                    eta_seconds=0 if remaining==0 else round(remaining/rate,1) if rate>0 else None,
                    max_inflight=max(1,min(2*cpu_workers,prepare_batch_size)),actual_device='cpu',
                    cache_hits=cache_stats['cache_hits'],cache_misses=cache_stats['cache_misses'],
                    checkpoint_available=True)),flush=True)
                del entry
            payloads=[(raw,cfg,source['dates'],str(root/'data_snapshots/native'),header,attempt_id) for raw in compute_records]
            _map_batches(_checkpoint_native_worker,payloads,cpu_workers=cpu_workers,prepare_batch_size=prepare_batch_size,
                stage='native',on_result=accept,initial_completed=len(completed),total=len(raw_by_symbol),reused=reused,
                progress_started=started,progress_baseline=reused,initial_processed=cache_stats['cache_hits'],
                progress_details=lambda:dict(cache_hits=cache_stats['cache_hits'],cache_misses=cache_stats['cache_misses'],
                    checkpoint_available=True))
        if _native_code_hashes()!=header['identity']['code_hashes']:
            raise ValueError('native generating code changed during preparation; completed identity refused')
        if _file_hash(base/'reports/raw_source.json')!=header['identity']['raw_report_sha256']:
            raise ValueError('raw source report changed during native preparation')
        final_identity=_checkpoint_header(run_id,raw_source_run_id,source,cfg,cpu_workers,prepare_batch_size,code)['identity']
        if final_identity!=header['identity']:
            raise ValueError('native source engine, schema or input identity changed before completion')
    except BaseException as exc:
        status=dict(status='interrupted',run_id=run_id,completed=len(completed),total=len(raw_by_symbol),
            reused=reused,failures=list(failures.values()),attempt_id=attempt_id,
            error=f'{type(exc).__name__}: {exc}',finished_at=local_now().isoformat())
        _save(state_path,status);_save(root/'reports/preparation_attempts'/(attempt_id+'.json'),status)
        raise
    records=[completed[symbol] for symbol in source['selected_symbols'] if symbol in completed]
    timings={}
    for checkpoint in checkpoints.values():
        for name,value in checkpoint.get('timings_ns',{}).items():timings[name]=timings.get(name,0)+value
    performance=dict(timings_ns=timings,elapsed_seconds=round(time.monotonic()-started,3),reused=reused,
        cpu_workers=cpu_workers,max_inflight=max(1,min(2*cpu_workers,prepare_batch_size)),
        pooling='bounded_continuous_per_stock',stock_checkpoints=len(checkpoints),**cache_stats)
    print('DATASET_COMPLETE '+str(len(records))+'/'+str(len(raw_by_symbol)),flush=True)
    failed=list(failures.values())
    report={'run_id':run_id,'strategy_version':STRATEGY_ID,'status':'failed' if failed else 'completed','data_end':source['end'],
        'config':cfg,'dataset_records':records,'raw_only_rebuild':True,'reused_model_or_label':False,'failures':failed,
        'population_mode':'full_liquidity_union','selected_symbols':population['selected_symbols'],
        'liquidity':population,'liquidity_universe':population,'prepare_batch_size':prepare_batch_size,
        'preparation_performance':performance,'native_checkpoint_hash':header['checkpoint_hash'],
        'source_snapshot':{'raw_source_run_id':raw_source_run_id,'raw_snapshot_hash':source['snapshot_hash'],
            'population_mode':'full_liquidity_union','liquidity':population,
            'raw_input_hashes':{r['symbol']:r['raw_sha256'] for r in source['raw_records']},'code_hashes':code,
            'calendar_run_id':source['calendar_run_id'],'calendar_hash':source['calendar_hash'],
            'engine_contract':header['identity']['engine_contract'],
            'feature_schema_hashes':header['identity']['feature_schema_hashes']},
        'calendar_dates':source['dates'],'publication_allowed':False,'completed_at':local_now().isoformat()}
    _save(root/'reports/native_source.json',report)
    status=dict(status=report['status'],run_id=run_id,completed=len(records),total=len(raw_by_symbol),reused=reused,
        failures=failed,attempt_id=attempt_id,performance=performance,finished_at=local_now().isoformat())
    _save(state_path,status);_save(root/'reports/preparation_attempts'/(attempt_id+'.json'),status)
    if failed: raise ValueError(f'native source incomplete: {len(failed)}/{len(raw_by_symbol)}; retained {run_id}')
    return report
