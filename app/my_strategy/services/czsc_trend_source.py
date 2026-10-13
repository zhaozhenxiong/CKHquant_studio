"""Fresh, immutable raw inputs and recomputed native evidence for reassessment."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
import gzip
import io
import multiprocessing
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd

from my_strategy.adapters.czsc_adapter import load_bars, raw_path
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_trend_support import _file_hash, _save, _json, resolve_frozen_asset
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, strategy_config
from my_strategy.services.czsc_liquidity import (RAW_IDENTITY_COLUMNS, LIQUIDITY_COLUMNS,
    liquidity_policy, daily_eligibility, annotate_liquidity)


NATIVE_GENERATING_FILES = tuple('services/' + name for name in (
    'czsc_trend_source.py', 'czsc_market_features.py', 'czsc_market_profiles.py',
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


def _map_batches(worker, payloads, *, cpu_workers, prepare_batch_size, stage):
    _batch_size(prepare_batch_size)
    if isinstance(cpu_workers,bool) or not isinstance(cpu_workers,int) or cpu_workers < 1:
        raise ValueError('cpu_workers must be positive')
    output=[]
    if cpu_workers > 1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            for offset in range(0,len(payloads),prepare_batch_size):
                output.extend(pool.map(worker,payloads[offset:offset+prepare_batch_size]))
                print('SOURCE_BATCH '+json.dumps({'stage':stage,'completed':len(output),'total':len(payloads)}),flush=True)
    else:
        for offset in range(0,len(payloads),prepare_batch_size):
            output.extend(worker(payload) for payload in payloads[offset:offset+prepare_batch_size])
            print('SOURCE_BATCH '+json.dumps({'stage':stage,'completed':len(output),'total':len(payloads)}),flush=True)
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
    output=_map_batches(_raw_worker,payloads,cpu_workers=cpu_workers,prepare_batch_size=prepare_batch_size,stage='raw')
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
    from my_strategy.services.czsc_price_volume_features import prepare_wyckoff_inputs
    from my_strategy.services.czsc_native_evidence import build_native_evidence
    threadpool_limits(limits=1)
    record,cfg,calendar,directory=payload
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
    features=bind_feature_profile(_calendar_quality(build_features(raw),raw,calendar),'ma_trend')
    features=prepare_wyckoff_inputs(prepare_structure_inputs(features),raw=raw,config={'market_dates':calendar})
    for column in raw:
        if column not in features:features[column]=raw[column].to_numpy(copy=True)
    pack=build_native_evidence(features,cfg,market_dates=calendar)
    target=Path(directory)/record['symbol'].replace('.','_');target.mkdir(parents=True,exist_ok=False)
    numeric=pack['frame'].drop(columns=[c for c in pack['frame'] if pack['frame'][c].map(lambda v:isinstance(v,(dict,list,tuple))).any()])
    numeric.attrs={};numeric.to_parquet(target/'features.parquet',index=False)
    native_path=target/'native.json.gz'
    _save_native_pack(native_path,{k:v for k,v in pack.items() if k!='frame'})
    return {**record,'path':str(target/'features.parquet'),'sha256':_file_hash(target/'features.parquet'),
        'native_path':str(native_path),'native_sha256':_file_hash(native_path),'native_encoding':'gzip_json',
        'native_config_hash':pack['config_hash'],'native_schema_hash':pack['schema_hash'],'rows':len(numeric),
        'raw_only_rebuild':True,'reused_model_or_label':False}


def prepare_native_source(raw_source_run_id, *, config=None, cpu_workers=4, prepare_batch_size=360):
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    base=ARTIFACT_RUNS_ROOT/raw_source_run_id
    source=json.loads((base/'reports/raw_source.json').read_text())
    if source.get('strategy_version')!=STRATEGY_ID or source.get('status')!='completed' or source.get('failures'):
        raise ValueError('a complete new raw-only current snapshot is required')
    if source['snapshot_hash']!=stable_hash({k:v for k,v in source.items() if k!='snapshot_hash'}):
        raise ValueError('raw snapshot identity drift')
    cfg=strategy_config(config)
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
    context=create_run_context(task='trend-native-source',as_of_date=source['end'],
        config={'strategy_version':STRATEGY_ID,'raw_source_run_id':raw_source_run_id,'source_config':cfg,
                'prepare_batch_size':_batch_size(prepare_batch_size)},
        stocks=source['selected_symbols'],scope='train',data_version=source['snapshot_hash'],end_date=source['end'])
    print('NATIVE_SOURCE_RUN_ID '+context.run_id,flush=True)
    payloads=[(record,cfg,source['dates'],str(context.subdir('data_snapshots','native'))) for record in source['raw_records']]
    records=_map_batches(_native_worker,payloads,cpu_workers=cpu_workers,
                         prepare_batch_size=prepare_batch_size,stage='native')
    print('DATASET_COMPLETE '+str(len(records))+'/'+str(len(payloads)),flush=True)
    code={name:_file_hash(Path(__file__).resolve().parents[1]/name) for name in NATIVE_GENERATING_FILES}
    report={'run_id':context.run_id,'strategy_version':STRATEGY_ID,'status':'completed','data_end':source['end'],
        'config':cfg,'dataset_records':records,'raw_only_rebuild':True,'reused_model_or_label':False,
        'population_mode':'full_liquidity_union','selected_symbols':population['selected_symbols'],
        'liquidity':population,'liquidity_universe':population,'prepare_batch_size':prepare_batch_size,
        'source_snapshot':{'raw_source_run_id':raw_source_run_id,'raw_snapshot_hash':source['snapshot_hash'],
            'population_mode':'full_liquidity_union','liquidity':population,
            'raw_input_hashes':{r['symbol']:r['raw_sha256'] for r in source['raw_records']},'code_hashes':code,
            'calendar_run_id':source['calendar_run_id'],'calendar_hash':source['calendar_hash']},
        'calendar_dates':source['dates'],'publication_allowed':False,'completed_at':local_now().isoformat()}
    _save(context.subdir('reports')/'native_source.json',report)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'source_run_id':source['calendar_run_id'],'dates':source['dates']})
    return report
