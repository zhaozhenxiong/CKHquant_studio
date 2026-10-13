"""Fresh, immutable raw inputs and recomputed native evidence for reassessment."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import pandas as pd

from my_strategy.adapters.czsc_adapter import load_bars, list_symbols, raw_path
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_research import _file_hash, _save, _json
from my_strategy.services.czsc_trend_v2_contracts import STRATEGY_ID, strategy_config


def select_cohort(size=180, *, db_path=None, preferred=None):
    """Rank only a preregistered pool and pre-feature-start input coverage."""
    from my_strategy.adapters.czsc_adapter import _connection
    from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("cohort size must be positive")
    with _connection(db_path) as connection:
        sources = list(VERIFIED_SOURCES)
        placeholders = ','.join('?' for _ in sources)
        rows = connection.execute(
            f"SELECT stock FROM stock_daily_normalized WHERE date < ? AND source IN ({placeholders}) "
            "AND has_trade_price=1 GROUP BY stock HAVING COUNT(*)>=65", ['2023-01-01', *sources]).fetchall()
    qualified = {r[0] for r in rows if (r[0].startswith('00') and r[0].endswith('.SZ')) or
                 (r[0].startswith('60') and r[0].endswith('.SH'))}
    first = list(dict.fromkeys(preferred or []))
    if set(first) - qualified:
        raise ValueError("preferred cohort violates preregistered pre2023 coverage")
    rest = sorted(qualified - set(first), key=lambda symbol: stable_hash({'seed':42,'symbol':symbol}))
    chosen = (first + rest)[:size]
    if len(chosen) != size:
        raise ValueError("raw warehouse cannot fill the registered cohort")
    return chosen


def _raw_worker(payload):
    symbol, end, directory, db_path = payload
    try:
        frame = load_bars(symbol, end=end, as_of=end, db_path=db_path)
        columns = ['symbol','date','open','high','low','close','volume','amount','source','has_trade_price']
        if not set(columns).issubset(frame):
            raise ValueError('raw observation identity/source fields missing')
        raw = frame[columns].copy(); raw.attrs={}
        target = Path(directory) / symbol.replace('.','_'); target.mkdir(parents=True, exist_ok=False)
        path = target / 'raw.parquet'; raw.to_parquet(path, index=False)
        return {'symbol':symbol,'raw_path':str(path),'raw_sha256':_file_hash(path),
                'raw_columns':columns,'raw_data_start':str(pd.Timestamp(raw.date.iloc[0]).date()),
                'data_end':str(pd.Timestamp(raw.date.iloc[-1]).date()),'raw_rows':len(raw),
                'raw_series_hash':frame.attrs['data_version'],
                'raw_attrs':{key:frame.attrs[key] for key in ['price_basis','legacy_fuyao_bars',
                    'unverified_trade_price_bars','volume_unit','amount_unit','data_version']}}, None
    except Exception as exc:
        return None, {'symbol':symbol,'error':f'{type(exc).__name__}: {exc}'}


def freeze_raw_source(*, symbols, end, calendar_run_id=None, cpu_workers=4, db_path=None, selection_metadata=None):
    from my_strategy.web_dashboard.high_return import _calendar
    dates, calendar_id = _calendar(calendar_run_id); dates=[d for d in dates if d <= end]
    if not dates or dates[-1] != end:
        raise ValueError('end must be an independently verified market session')
    symbols=list(dict.fromkeys(symbols)); path=raw_path(db_path)
    if not symbols:
        raise ValueError('empty fixed training pool')
    context=create_run_context(task='czsc-trend-v2-raw-snapshot',as_of_date=end,
        config={'strategy_version':STRATEGY_ID,'selection':'fixed_preregistered_input_coverage_pool',
                'price_basis':'stored_raw_basis_with_row_source_qualification','calendar_run_id':calendar_id,'raw_db':str(path),
                'selection_metadata':selection_metadata or {}},
        stocks=symbols,scope='train',end_date=end)
    print('RAW_SOURCE_RUN_ID '+context.run_id,flush=True)
    payloads=[(symbol,end,str(context.subdir('data_snapshots','raw')),str(path)) for symbol in symbols]
    if cpu_workers > 1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            output=list(pool.map(_raw_worker,payloads))
    else:
        output=[_raw_worker(payload) for payload in payloads]
    records=[record for record,error in output if record is not None]
    failures=[error for record,error in output if error is not None]
    source={'strategy_version':STRATEGY_ID,'status':'completed' if not failures else 'failed',
        'run_id':context.run_id,'end':end,'data_end':end,'selected_symbols':symbols,'raw_records':records,
        'failures':failures,'calendar_run_id':calendar_id,'calendar_hash':stable_hash(dates),
        'dates':dates,'raw_only_rebuild':True,'reused_model_or_label':False,
        'selection_metadata':selection_metadata or {},
        'latest_observed_symbols':sum(record['data_end']==end for record in records),
        'price_basis_counts':{basis:sum(r['raw_attrs']['price_basis']==basis for r in records)
                             for basis in sorted({r['raw_attrs']['price_basis'] for r in records})},
        'publication_allowed':False,'recorded_at':local_now().isoformat()}
    source['snapshot_hash']=stable_hash(source)
    _save(context.subdir('reports')/'raw_source.json',source)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'source_run_id':calendar_id,'dates':dates})
    from my_strategy.storage.czsc_results import ResultStore
    ResultStore().save('czsc-trend-v2-raw-snapshot',_json(source))
    if failures:
        raise ValueError(f'raw source incomplete: {len(failures)}/{len(symbols)}; retained {context.run_id}')
    return context.run_dir(),source


def _native_worker(payload):
    from threadpoolctl import threadpool_limits
    from my_strategy.services.czsc_research_features import build_features
    from my_strategy.services.czsc_research import _calendar_quality
    from my_strategy.services.czsc_research_profiles import bind_feature_profile, prepare_structure_inputs
    from my_strategy.services.czsc_wyckoff_features import prepare_wyckoff_inputs
    from my_strategy.services.czsc_native_evidence_v4 import build_native_evidence_v4
    threadpool_limits(limits=1)
    record,cfg,calendar,directory=payload
    path=Path(record['raw_path'])
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
    features=bind_feature_profile(_calendar_quality(build_features(raw),raw,calendar),'ma_trend_v1')
    features=prepare_wyckoff_inputs(prepare_structure_inputs(features),raw=raw,config={'market_dates':calendar})
    for column in raw:
        if column not in features:features[column]=raw[column].to_numpy(copy=True)
    pack=build_native_evidence_v4(features,cfg,market_dates=calendar)
    target=Path(directory)/record['symbol'].replace('.','_');target.mkdir(parents=True,exist_ok=False)
    numeric=pack['frame'].drop(columns=[c for c in pack['frame'] if pack['frame'][c].map(lambda v:isinstance(v,(dict,list,tuple))).any()])
    numeric.attrs={};numeric.to_parquet(target/'features.parquet',index=False)
    _save(target/'native.json',{k:v for k,v in pack.items() if k!='frame'})
    return {**record,'path':str(target/'features.parquet'),'sha256':_file_hash(target/'features.parquet'),
        'native_path':str(target/'native.json'),'native_sha256':_file_hash(target/'native.json'),
        'native_config_hash':pack['config_hash'],'native_schema_hash':pack['schema_hash'],'rows':len(numeric),
        'raw_only_rebuild':True,'reused_model_or_label':False}


def prepare_native_source(raw_source_run_id, *, config=None, cpu_workers=4):
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    base=ARTIFACT_RUNS_ROOT/raw_source_run_id
    source=json.loads((base/'reports/raw_source.json').read_text())
    if source.get('strategy_version')!=STRATEGY_ID or source.get('status')!='completed' or source.get('failures'):
        raise ValueError('a complete new raw-only v2 snapshot is required')
    if source['snapshot_hash']!=stable_hash({k:v for k,v in source.items() if k!='snapshot_hash'}):
        raise ValueError('raw snapshot identity drift')
    cfg=strategy_config(config)
    context=create_run_context(task='czsc-trend-v2-native-snapshot',as_of_date=source['end'],
        config={'strategy_version':STRATEGY_ID,'raw_source_run_id':raw_source_run_id,'source_config':cfg},
        stocks=source['selected_symbols'],scope='train',data_version=source['snapshot_hash'],end_date=source['end'])
    print('NATIVE_SOURCE_RUN_ID '+context.run_id,flush=True)
    payloads=[(record,cfg,source['dates'],str(context.subdir('data_snapshots','native'))) for record in source['raw_records']]
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            records=list(pool.map(_native_worker,payloads))
    else:records=[_native_worker(payload) for payload in payloads]
    generators=['services/'+name for name in ['czsc_trend_v2_source.py','czsc_research_features.py',
        'czsc_research_profiles.py','czsc_wyckoff_features.py','czsc_wyckoff_events.py','czsc_native_evidence_v4.py',
        'czsc_analysis.py','czsc_research.py','czsc_research_data.py']]+['adapters/czsc_adapter.py','configs/czsc_strategy.json']
    code={name:_file_hash(Path(__file__).resolve().parents[1]/name) for name in generators}
    report={'run_id':context.run_id,'strategy_version':STRATEGY_ID,'status':'completed','data_end':source['end'],
        'config':cfg,'dataset_records':records,'raw_only_rebuild':True,'reused_model_or_label':False,
        'source_snapshot':{'raw_source_run_id':raw_source_run_id,'raw_snapshot_hash':source['snapshot_hash'],
            'raw_input_hashes':{r['symbol']:r['raw_sha256'] for r in source['raw_records']},'code_hashes':code,
            'calendar_run_id':source['calendar_run_id'],'calendar_hash':source['calendar_hash']},
        'calendar_dates':source['dates'],'publication_allowed':False,'completed_at':local_now().isoformat()}
    _save(context.subdir('reports')/'native_v4.json',report)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'source_run_id':source['calendar_run_id'],'dates':source['dates']})
    return report
