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


def code_identity():
    from my_strategy.adapters.czsc_adapter import engine_contract
    from my_strategy.services.czsc_native_evidence import NATIVE_SCHEMA_HASH
    from my_strategy.services.czsc_analysis import strategy_config
    names=['incremental_source.py','czsc_incremental_state.py','czsc_native_evidence.py','czsc_analysis.py',
           'czsc_market_features.py','czsc_market_profiles.py','czsc_price_volume_features.py','czsc_price_volume_events.py']
    root=Path(__file__).parent
    return {'code':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
            'engine':engine_contract(),'schema':NATIVE_SCHEMA_HASH,'config':strategy_config(),
            'pandas':pd.__version__}


def _stop(stop):
    if stop and (stop.is_set() if hasattr(stop,'is_set') else stop()):
        raise InterruptedError('结构增量补齐已取消')


def _row_hashes(raw):
    fields=[c for c in ('symbol','date','open','high','low','close','volume','amount','source','has_trade_price','available_at') if c in raw]
    return [stable_hash(row) for row in plain(raw[fields].to_dict('records'))]


class IncrementalSource:
    def __init__(self, *, store, raw_db=None):
        self.store,self.raw_db=store,raw_db

    def _peek(self,key):
        # Planning never refreshes TTL, writes a result, or runs numerical code.
        with self.store._lock:
            row=self.store._live_row(key,self.store.clock())
            if row is None:return None
            content=self.store._path(row['sha256']).read_bytes()
            if len(content)!=row['size_bytes'] or hashlib.sha256(content).hexdigest()!=row['sha256']:
                raise ValueError('incremental source cached object SHA mismatch')
            if self.store.cache_key(json.loads(row['identity_json']),row['namespace'])!=key:
                raise ValueError('incremental source cached identity mismatch')
            return json.loads(gzip.decompress(content))

    def _setting(self,name):
        with self.store._lock:
            row=self.store.connection.execute('SELECT value_json FROM settings WHERE name=?',(name,)).fetchone()
            return json.loads(row[0]) if row else None

    def _parts(self,symbol,end,calendar_dates=None):
        raw=load_bars(symbol,end=end,db_path=self.raw_db)
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

    def _manifest_frame(self,manifest):
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

    def source(self,symbol,end,*,stop=None,calendar_dates=None,calendar_identity=None):
        from my_strategy.services.czsc_native_evidence import build_native_evidence
        from my_strategy.services.czsc_market_features import _trailing_features
        from my_strategy.services.czsc_incremental_state import StateNotRestorable
        symbol=normalize_symbol(symbol);_stop(stop)
        raw,dates,hashes,identity=self._parts(symbol,end,calendar_dates)
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
                try:old_frame=self._manifest_frame(previous)
                except KeyError:fallback='previous_feature_chunk_missing'
                else:state=previous['native_state'];prefix_rows=n
        _stop(stop)
        try:pack=build_native_evidence(raw,market_dates=dates,incremental_state=state,capture_state=True)
        except StateNotRestorable as exc:
            # No cross-stock fallback, no old-day summary, no arbitrary N-bar tail.
            fallback=str(exc);state=None;prefix_rows=0;old_frame=None
            pack=build_native_evidence(raw,market_dates=dates,capture_state=True)
        _stop(stop)
        frame=pack['frame'];trailing=_trailing_features(raw)
        for column in trailing:frame[column]=trailing[column].reindex(frame.index)
        columns=list(dict.fromkeys(['symbol','date','open','high','low','close','volume','amount','source','has_trade_price',
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
            'prefix_rows_reused':prefix_rows,'fallback_reason':fallback,'wyckoff_mode':'single_stock_vector_replay'}
        chunks=[];owner='incremental_build:'+uuid.uuid4().hex;leased=[]
        try:
            records=plain(frame.to_dict('records'))
            for offset in range(0,len(records),CHUNK_ROWS):
                _stop(stop);chunk={'columns':list(frame),'rows':records[offset:offset+CHUNK_ROWS]}
                content_hash=stable_hash(chunk)
                ref=self.store.put({'format':VERSION,'content_hash':content_hash},chunk,namespace='incremental_chunks')
                self.store.acquire_lease(ref['cache_key'],owner,ttl_seconds=300);leased.append(ref['cache_key'])
                chunks.append({'cache_key':ref['cache_key'],'content_hash':content_hash})
            manifest={'identity':identity,'columns':list(frame),'chunks':chunks,'events':events,'raw_row_hashes':hashes,
                'native_state':pack['incremental_state'],'native_schema_hash':pack['schema_hash'],
                'latest_evaluation':pack['evaluations'][-1]}
            _stop(stop);self.store.put(identity,manifest,namespace='incremental_sources')
            # Pointer publication follows all immutable objects; an interrupted
            # writer never exposes a partially materialized state.
            if previous is None or previous['identity']['observed_end']<=identity['observed_end']:
                self.store.set_setting('incremental_source_latest:'+symbol,{'cache_key':key})
        finally:
            for leased_key in leased:self.store.release_lease(leased_key,owner)
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
