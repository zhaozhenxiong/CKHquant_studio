"""Small-pool causal source cache with explicit native continuation state.

Raw prices are read-only. Feature history is content-addressed in fixed chunks;
adding a day replaces only its open chunk and a compact manifest/state. Native
restoration is validated; unsupported geometry replays this stock only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time
import uuid

import pandas as pd

from my_strategy.adapters.czsc_adapter import load_bars, normalize_symbol
from my_strategy.core.run_context import stable_hash

VERSION='incremental_source_v1'
CHUNK_ROWS=256
FEATURE_COLUMNS=['ret_5','ret_20','ma20_bias','ma20_slope5','ma60_bias','atr14_ratio',
    'volatility20','vol_ratio20','amount_ratio20','drawdown20','native_structure_buy_support',
    'native_structure_sell_support','native_w_demand_score','native_w_supply_score']


def plain(value):
    from my_strategy.services.czsc_trend_support import _json
    return _json(value)


def numerical_input_contract():
    """Bind transitive input/qualification helpers, without status/account code.

    Resolved raw rows and calendar dates are hashed separately. Their admission
    rules and lossless serialization are also generation dependencies: identical
    rows must not reuse a result after those rules change.
    """
    import inspect
    import numpy as np
    from my_strategy.adapters import czsc_adapter
    from my_strategy.core import run_context
    from my_strategy.services import czsc_trend_support, czsc_research_calendar
    dependencies=(
        (czsc_adapter,('load_bars','normalize_symbol','date_cutoff','raw_path','_connection','legacy_fuyao_source')),
        (czsc_trend_support,('_json','_calendar','resolve_frozen_asset')),
        (czsc_research_calendar,('load_verified_calendar','_valid_sessions','calendar_prefix_compatible','_file_hash')),
        (run_context,('stable_hash',)),
    )
    return {'schema':'numerical_input_contract_v1',
        'functions':{module.__name__+'.'+name:hashlib.sha256(inspect.getsource(getattr(module,name)).encode()).hexdigest()
                     for module,names in dependencies for name in names},
        'verified_sources':sorted(czsc_trend_support.VERIFIED_SOURCES),
        'numpy':np.__version__,'pandas':pd.__version__,
        'identity_contract':hashlib.sha256(inspect.getsource(numerical_input_contract).encode()).hexdigest()}


def code_identity():
    from my_strategy.adapters.czsc_adapter import engine_contract
    from my_strategy.services.czsc_native_evidence import NATIVE_SCHEMA_HASH
    from my_strategy.services.czsc_analysis import strategy_config, native_generation_identity
    names=['incremental_source.py','incremental_rolling.py','czsc_incremental_state.py','czsc_native_evidence.py',
           'czsc_market_features.py','czsc_market_profiles.py','czsc_price_volume_features.py','czsc_price_volume_events.py']
    root=Path(__file__).parent
    return {'code':{**{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
                    'czsc_analysis.native_generation':native_generation_identity()},
            'input_contract':numerical_input_contract(),
            'engine':engine_contract(),'schema':NATIVE_SCHEMA_HASH,'config':strategy_config(),
            'pandas':pd.__version__}


def _stop(stop):
    if stop is not None and hasattr(stop,'check'):
        stop.check();return
    if stop and (stop.is_set() if hasattr(stop,'is_set') else stop()):
        raise InterruptedError('结构增量补齐已取消')


def _row_hashes(raw):
    fields=[c for c in ('symbol','date','open','high','low','close','volume','amount','source','has_trade_price','available_at') if c in raw]
    return [stable_hash(row) for row in plain(raw[fields].to_dict('records'))]


class IncrementalSource:
    def __init__(self, *, store, raw_db=None):
        self.store,self.raw_db=store,raw_db

    def _peek(self,key):
        # Verify the immutable object outside the SQLite write lock. A missing
        # object remains an explicit cache miss, never a stale value.
        return self.store.get_by_key(key, touch=False)

    def _setting(self,name):
        with self.store._lock:
            row=self.store.connection.execute('SELECT value_json FROM settings WHERE name=?',(name,)).fetchone()
            return json.loads(row[0]) if row else None

    def _parts(self,symbol,end,calendar_dates=None,raw_snapshot=None):
        raw=load_bars(symbol,end=end,db_path=self.raw_db) if raw_snapshot is None else raw_snapshot
        if not raw.empty and (set(raw.symbol.astype(str))!={symbol} or pd.Timestamp(raw.date.iloc[-1])>pd.Timestamp(end)):
            raise ValueError('共享行情快照与请求股票或截止日不一致')
        if raw.empty:raise ValueError('股票没有已核验行情')
        if calendar_dates is None:
            from my_strategy.services.czsc_trend_support import _calendar
            calendar_dates,_=_calendar(None)
        dates=[d for d in calendar_dates if d<=str(pd.Timestamp(raw.date.iloc[-1]).date())]
        if not dates or str(pd.Timestamp(raw.date.iloc[-1]).date()) not in dates:
            raise ValueError('股票末日不在已核验日历')
        hashes=_row_hashes(raw)
        identity={'kind':VERSION,'symbol':symbol,'as_of':str(end),'observed_end':str(pd.Timestamp(raw.date.iloc[-1]).date()),
            'input_sha':stable_hash(hashes),'price_basis':raw.attrs.get('price_basis'),
            'units':{k:raw.attrs.get(k) for k in ('volume_unit','amount_unit')},
            'calendar':{'sha256':stable_hash(dates),'prefix_end':dates[-1]},'code':code_identity()}
        return raw,dates,hashes,identity

    def _manifest_frame(self,manifest,*,lease_owner=None,leased=None):
        owner=lease_owner or 'incremental_read:'+uuid.uuid4().hex
        keys=[part['cache_key'] for part in manifest['chunks']]
        try:self.store.acquire_leases(keys,owner,ttl_seconds=300)
        except KeyError:raise KeyError('incremental feature chunk expired')
        if leased is not None:leased.extend(keys)
        try:return self._read_manifest_frame(manifest)
        finally:
            if lease_owner is None:self.store.release_leases(keys,owner)

    def _read_manifest_frame(self,manifest):
        rows=[]
        for part in manifest['chunks']:
            chunk=self._peek(part['cache_key'])
            if chunk is None:raise KeyError('incremental feature chunk expired')
            if stable_hash(chunk)!=part['content_hash']:raise ValueError('incremental feature chunk identity drift')
            rows.extend(chunk['rows'])
        frame=pd.DataFrame(rows,columns=manifest['columns']);frame['date']=pd.to_datetime(frame.date)
        return frame

    def inspect(self,symbol,end,*,stop=None,calendar_dates=None):
        symbol=normalize_symbol(symbol);_stop(stop)
        raw,dates,hashes,identity=self._parts(symbol,end,calendar_dates)
        key=self.store.cache_key(identity,'incremental_sources');current=self._peek(key)
        reason=None
        if current:
            try:self._manifest_frame(current)
            except KeyError:current=None;reason='feature_chunk_missing'
        latest_ref=self._setting('incremental_source_latest:'+symbol)
        previous=self._peek(latest_ref['cache_key']) if latest_ref else None
        if current:action='reuse';changed=None
        elif previous and previous['identity']['code']==identity['code']:
            old=previous['raw_row_hashes'];common=0
            for left,right in zip(old,hashes):
                if left!=right:break
                common+=1
            append=common==len(old) and len(hashes)>len(old)
            old_end=previous['identity']['observed_end']
            calendar_same=stable_hash([d for d in dates if d<=old_end])==previous['identity']['calendar']['sha256']
            action='append' if append and calendar_same else 'rebuild'
            changed=str(pd.Timestamp(raw.date.iloc[min(common,len(raw)-1)]).date())
            if not calendar_same:reason='historical_calendar_changed'
        else:action='rebuild';changed=str(pd.Timestamp(raw.date.iloc[0]).date());reason=reason or ('producer_changed' if previous else 'source_missing')
        return {'symbol':symbol,'price_date':str(end),'observed_end':identity['observed_end'],
                'action':action,'reason':reason,'first_changed_date':changed,'raw_rows':len(raw),
                'identity':identity,'cache_key':key,'input_sha':identity['input_sha']}

    def source(self,symbol,end,*,stop=None,calendar_dates=None,calendar_identity=None,raw_snapshot=None):
        owner='incremental_build:'+uuid.uuid4().hex;leased=[]
        try:
            return self._source(symbol,end,stop=stop,calendar_dates=calendar_dates,
                calendar_identity=calendar_identity,raw_snapshot=raw_snapshot,owner=owner,leased=leased)
        finally:
            self.store.release_leases(leased,owner)

    def _source(self,symbol,end,*,stop=None,calendar_dates=None,calendar_identity=None,raw_snapshot=None,owner,leased):
        from my_strategy.services.czsc_native_evidence import build_native_evidence
        from my_strategy.services.incremental_rolling import trailing_features_with_state
        from my_strategy.services.czsc_incremental_state import StateNotRestorable
        symbol=normalize_symbol(symbol);_stop(stop)
        raw,dates,hashes,identity=self._parts(symbol,end,calendar_dates,raw_snapshot)
        key=self.store.cache_key(identity,'incremental_sources');saved=self._peek(key)
        if saved:
            try:frame=self._manifest_frame(saved)
            except KeyError:saved=None
            else:
                frame.attrs.update(raw.attrs);frame.attrs['incremental']={'mode':'reused','reused':True,'native_rows_replayed':0,'native_rows_appended':0}
                return raw,frame,saved['events'],{**identity,'source_cache_key':key}
        latest_ref=self._setting('incremental_source_latest:'+symbol)
        previous=self._peek(latest_ref['cache_key']) if latest_ref else None
        old_frame=None;state=None;fallback=None;prefix_rows=0
        if previous and previous['identity']['code']==identity['code']:
            old=previous['raw_row_hashes'];n=len(old)
            if len(hashes)>n and hashes[:n]==old and stable_hash([d for d in dates if d<=previous['identity']['observed_end']])==previous['identity']['calendar']['sha256']:
                try:old_frame=self._manifest_frame(previous,lease_owner=owner,leased=leased)
                except KeyError:fallback='previous_feature_chunk_missing'
                else:state=previous['native_state'];prefix_rows=n
        _stop(stop)
        try:pack=build_native_evidence(raw,market_dates=dates,incremental_state=state,capture_state=True,cancel_check=stop,compact=True)
        except StateNotRestorable as exc:
            # No cross-stock fallback, no old-day summary, no arbitrary N-bar tail.
            fallback=str(exc);state=None;prefix_rows=0;old_frame=None
            pack=build_native_evidence(raw,market_dates=dates,capture_state=True,cancel_check=stop,compact=True)
        _stop(stop)
        frame=pack['frame']
        rolling_previous=previous.get('trailing_state') if state else None
        trailing,trailing_state=trailing_features_with_state(raw,rolling_previous,verified_row_hashes=hashes)
        for column in trailing:frame[column]=trailing[column].reindex(frame.index)
        columns=list(dict.fromkeys(['symbol','date','open','high','low','close','volume','amount','source','has_trade_price','available_at',
            'ma20','ma60',*FEATURE_COLUMNS,'native_structure_input_eligible','native_structure_support_low']))
        frame=frame[[c for c in columns if c in frame]].copy()
        events=[{**e,'risk_boundary':e.get('risk_stop_price') or e.get('invalidation_price')}
                for e in pack['events'] if e.get('confirmed') and e.get('input_eligible')]
        if state:
            frame=pd.concat([old_frame,frame],ignore_index=True)
            events=previous['events']+events
        frame=frame.reset_index(drop=True)
        execution={'mode':'native_state_append' if state else 'single_stock_full_replay','reused':False,
            'native_rows_appended':len(raw)-prefix_rows if state else 0,'native_rows_replayed':0 if state else len(raw),
            'prefix_rows_reused':prefix_rows,'fallback_reason':fallback,
            'wyckoff_mode':pack.get('incremental',{}).get('wyckoff_mode','single_stock_vector_replay'),
            'wyckoff_rows_appended':pack.get('incremental',{}).get('wyckoff_rows_appended',0),
            'rolling_metrics_mode':('exact_mean_and_variance_state' if trailing_state.get('variance') else 'exact_mean_state_and_full_prefix_variance')
                if rolling_previous and trailing_state else 'exact_full_prefix_pandas',
            'rolling_state_admitted':trailing_state is not None}
        # A verified append keeps all closed immutable chunks. Only the old
        # open tail and appended rows are encoded, hashed and published again.
        closed_count=prefix_rows//CHUNK_ROWS if state else 0
        chunks=list(previous['chunks'][:closed_count]) if state else []
        records=plain(frame.iloc[closed_count*CHUNK_ROWS:].to_dict('records'))
        items=[]
        for offset in range(0,len(records),CHUNK_ROWS):
            _stop(stop);chunk={'columns':list(frame),'rows':records[offset:offset+CHUNK_ROWS]}
            content_hash=stable_hash(chunk)
            items.append({'identity':{'format':VERSION,'content_hash':content_hash},
                'result':chunk,'namespace':'incremental_chunks'})
            chunks.append({'cache_key':self.store.cache_key(items[-1]['identity'],'incremental_chunks'),
                'content_hash':content_hash})
        _stop(stop)
        refs=self.store.put_many(items,lease_owner=owner,lease_ttl_seconds=300)
        leased.extend(ref['cache_key'] for ref in refs)
        execution.update(feature_closed_chunks_reused=closed_count,feature_chunks_written=len(items))
        manifest={'identity':identity,'columns':list(frame),'chunks':chunks,'events':events,'raw_row_hashes':hashes,
            'native_state':pack['incremental_state'],'trailing_state':trailing_state,'native_schema_hash':pack['schema_hash'],
            'latest_evaluation':pack['evaluations'][-1]}
        _stop(stop);self.store.put(identity,manifest,namespace='incremental_sources',
            refresh_keys=[part['cache_key'] for part in chunks])
        # Pointer publication follows all immutable objects; an interrupted
        # writer never exposes a partially materialized state.
        if previous is None or previous['identity']['observed_end']<=identity['observed_end']:
            self.store.set_setting('incremental_source_latest:'+symbol,{'cache_key':key})
        frame.attrs.update(raw.attrs);frame.attrs['incremental']=execution
        return raw,frame,events,{**identity,'source_cache_key':key}

    def summary(self,symbol,end,*,stop=None,calendar_dates=None):
        raw,frame,events,identity=self.source(symbol,end,stop=stop,calendar_dates=calendar_dates)
        if str(pd.Timestamp(raw.date.iloc[-1]).date())!=str(end):
            return {'symbol':symbol,'price_date':str(end),'status':'unavailable','reason_codes':['target_day_price_missing']}
        manifest=self._peek(identity['source_cache_key'])
        evaluation=manifest['latest_evaluation']
        result=plain({'symbol':symbol,'price_date':str(end),'status':'available',
            'training_eligible':False,'mode':'current_reference','identity':identity,
            'structure':evaluation['structure'],'wyckoff':evaluation['wyckoff'],
            'events_at_day':[e for e in events if e['date']==str(end)]})
        result['row_hash']=stable_hash(result)
        return result
