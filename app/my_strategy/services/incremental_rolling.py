"""Explicit float64 rolling means, checked against the installed pandas kernel.

The Kahan add/remove order follows pandas' BSD-licensed rolling mean algorithm:
https://github.com/pandas-dev/pandas/blob/v3.0.6/pandas/_libs/window/aggregations.pyx
State is JSON/SHA/version bound. Variance keeps the full-prefix pandas kernel: a
bounded-window or scalar Welford restart failed exact float64 admission.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash

SCHEMA='exact_trailing_mean_state_v1'
COLUMNS=('ret_5','ret_20','ma20','ma60','ma20_bias','ma20_slope5','ma60_bias',
         'atr14_ratio','volatility20','vol_ratio20','amount_ratio20','drawdown20')


class MeanState:
    def __init__(self,window,state=None):
        self.window=window;self.values=deque();self.total=0.;self.add_comp=0.;self.remove_comp=0.
        self.nobs=0;self.negative=0;self.same=0;self.previous=float('nan')
        if state:
            if state['window']!=window:raise ValueError('Rolling window changed')
            self.values=deque(float('nan') if x is None else float(x) for x in state['values'])
            for key in ('total','add_comp','remove_comp','nobs','negative','same'):
                setattr(self,key,state[key])
            self.previous=float('nan') if state['previous'] is None else float(state['previous'])

    def push(self,value):
        value=float(value)
        if len(self.values)==self.window:
            old=self.values.popleft()
            if not math.isnan(old):
                self.nobs-=1;y=-old-self.remove_comp;t=self.total+y
                self.remove_comp=(t-self.total)-y;self.total=t
                if math.copysign(1.,old)<0:self.negative-=1
        self.values.append(value)
        if not math.isnan(value):
            self.nobs+=1;y=value-self.add_comp;t=self.total+y
            self.add_comp=(t-self.total)-y;self.total=t
            if math.copysign(1.,value)<0:self.negative+=1
            self.same=self.same+1 if value==self.previous else 1;self.previous=value
        if self.nobs<self.window:return float('nan')
        result=self.total/self.nobs
        if self.same>=self.nobs:return self.previous
        if (self.negative==0 and result<0) or (self.negative==self.nobs and result>0):return 0.
        return result

    def dump(self):
        from my_strategy.services.czsc_trend_support import _json
        return _json({'window':self.window,'values':list(self.values),'total':self.total,
            'add_comp':self.add_comp,'remove_comp':self.remove_comp,'nobs':self.nobs,
            'negative':self.negative,'same':self.same,'previous':self.previous})


def trailing_features_with_state(raw, previous=None, *, verified_row_hashes=None):
    """Return full features on rebuild and only appended rows on continuation.

    The caller already validates every raw row, calendar and producer identity.
    This state independently binds its numerical input prefix and pandas version.
    A cold capture must reproduce all retained columns bit-for-bit before it is
    admitted for future continuation; no epsilon or dtype downcast is accepted.
    """
    from my_strategy.services.czsc_market_features import _trailing_features
    from my_strategy.services.czsc_trend_support import _json
    fields=['open','high','low','close','volume','amount']
    if verified_row_hashes is not None and len(verified_row_hashes)!=len(raw):raise ValueError('Shared row identity length mismatch')
    hash_mode='shared_exact_row_hashes' if verified_row_hashes is not None else 'numerical_price_rows'
    def input_hash(count):
        return stable_hash(verified_row_hashes[:count]) if verified_row_hashes is not None else stable_hash(_json(raw[fields].iloc[:count].to_dict('records')))
    start=0
    if previous:
        if (previous.get('format')!=SCHEMA or previous.get('sha256')!=stable_hash({k:v for k,v in previous.items() if k!='sha256'})
            or previous.get('pandas_version')!=pd.__version__ or previous.get('input_hash_mode')!=hash_mode):raise ValueError('Rolling state schema, SHA or pandas version mismatch')
        start=previous['rows']
        if type(start) is not int or not 0<start<len(raw):raise ValueError('Rolling state requires strict append')
        if previous['raw_prefix_hash']!=input_hash(start):
            raise ValueError('Rolling state input prefix changed')
    close=raw.close.astype(float);volume=raw.volume.where(np.isfinite(raw.volume)&(raw.volume>=0));amount=raw.amount.where(np.isfinite(raw.amount)&(raw.amount>=0))
    previous_close=close.shift(1)
    tr=pd.concat([raw.high-raw.low,(raw.high-previous_close).abs(),(raw.low-previous_close).abs()],axis=1).max(axis=1)
    streams={'ma20':(20,close),'ma60':(60,close),'atr14':(14,tr),
             'volume20':(20,volume.shift(1)),'amount20':(20,amount.shift(1))}
    states={name:MeanState(window,(previous or {}).get('means',{}).get(name)) for name,(window,_) in streams.items()}
    outputs={name:[] for name in streams}
    returns=close.pct_change(1,fill_method=None)
    ma_history=deque((previous or {}).get('ma20_history',[]),maxlen=5);lagged=[]
    for index in range(start,len(raw)):
        for name,(_,series) in streams.items():outputs[name].append(states[name].push(series.iloc[index]))
        lagged.append(ma_history[0] if len(ma_history)==5 else float('nan'))
        ma_history.append(outputs['ma20'][-1])
    result=pd.DataFrame(index=raw.index[start:]);tail=close.iloc[start:]
    ma20=pd.Series(outputs['ma20'],index=result.index);ma60=pd.Series(outputs['ma60'],index=result.index)
    result['ret_5']=close.pct_change(5,fill_method=None).iloc[start:]
    result['ret_20']=close.pct_change(20,fill_method=None).iloc[start:]
    result['ma20']=ma20;result['ma60']=ma60
    result['ma20_bias']=tail/ma20-1
    result['ma20_slope5']=ma20/pd.Series(lagged,index=result.index,dtype=float)-1
    result['ma60_bias']=tail/ma60-1
    result['atr14_ratio']=pd.Series(outputs['atr14'],index=result.index)/tail
    result['volatility20']=returns.rolling(20,min_periods=20).std(ddof=0).iloc[start:]
    result['vol_ratio20']=volume.iloc[start:]/pd.Series(outputs['volume20'],index=result.index).replace(0,np.nan)
    result['amount_ratio20']=amount.iloc[start:]/pd.Series(outputs['amount20'],index=result.index).replace(0,np.nan)
    result['drawdown20']=(close/close.rolling(20,min_periods=20).max()-1).iloc[start:]
    result=result.replace([np.inf,-np.inf],np.nan)
    state={'format':SCHEMA,'pandas_version':pd.__version__,'rows':len(raw),
        'raw_prefix_hash':input_hash(len(raw)),'input_hash_mode':hash_mode,
        'means':{name:value.dump() for name,value in states.items()},'ma20_history':_json(list(ma_history)),
        'variance_mode':'exact_full_prefix_pandas'}
    if not previous:
        expected=_trailing_features(raw)
        # Runtime admission is cheap relative to a native cold rebuild and
        # catches changes to pandas, CPU numerical kernels or feature formulas.
        means=[column for column in COLUMNS if column!='volatility20']
        if not np.array_equal(result[means].to_numpy(float),expected[means].to_numpy(float),equal_nan=True):
            return expected,None
        result=expected
    state['sha256']=stable_hash(state)
    return result,state
