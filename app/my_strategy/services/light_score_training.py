"""Pure, causal feature/label preparation for the bounded light-score models.

No file writes, downloads, model prediction, or registry changes occur here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = [
    'ret_5', 'ret_20', 'ma20_bias', 'ma20_slope5', 'ma60_bias',
    'atr14_ratio', 'volatility20', 'vol_ratio20', 'amount_ratio20', 'drawdown20',
    'native_structure_buy_support', 'native_structure_sell_support',
    'native_w_demand_score', 'native_w_supply_score',
]
EXTRA_FEATURES = [
    'structure_age_sessions', 'support_distance', 'resistance_distance',
    'confirmed_balance', 'demand_supply_gap', 'range_position20',
    'volume_trend5_20', 'intraday_range_ratio', 'close_location', 'trend_alignment',
]
KRONOS_FEATURES = [
    'kronos_return_median', 'kronos_return_p10', 'kronos_return_p90',
    'kronos_interval_width', 'kronos_up_fraction', 'kronos_valid_fraction',
]
KRONOS_STATES = ('valid', 'invalid', 'expired', 'missing')


def _day(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError('Missing decision date')
    if stamp.tzinfo:
        stamp = stamp.tz_convert('Asia/Shanghai').tz_localize(None)
    return stamp.normalize()


def _available_stamp(value):
    stamp=pd.Timestamp(value)
    if pd.isna(stamp):raise ValueError('Missing evidence availability time')
    return stamp.tz_localize('Asia/Shanghai') if stamp.tzinfo is None else stamp.tz_convert('Asia/Shanghai')


def prepare_score_features(frame, events=(), forecasts=None, *, variant='A', decision_time=None,
                           feature_profile='mixed_v1', calendar_dates=None):
    """Append 10 trailing features and optional frozen Kronos summaries.

    `events` must carry actual availability, never a future event anchor. Forecasts
    are a mapping of (symbol, YYYY-MM-DD), or YYYY-MM-DD for a one-stock frame,
    to *already generated* observations. This function never generates forecasts.
    B's optional `decision_time` is an exact timestamp or a stock/date mapping.
    It permits an after-close observation only once actually available and before
    its declared executable-open deadline; default historical timing is 15:00.
    A's numerical features and existing model identities are unchanged.
    """
    if feature_profile == 'ma_trend_v1':
        if variant != 'A':
            raise ValueError('MA profile supports A only; Kronos remains external evidence')
        from my_strategy.services.light_ma_features import prepare_ma_features
        return prepare_ma_features(frame, calendar_dates=calendar_dates)
    if feature_profile != 'mixed_v1':
        raise ValueError('Unknown score feature profile')
    if variant not in {'A', 'B'}:
        raise ValueError('Score variant must be A or B')
    # Source evidence is supplied explicitly through events. DataFrame attrs
    # can carry a large audit payload; copying it into each row adds no feature.
    data = pd.DataFrame(frame, copy=True)
    required = {'date', 'open', 'high', 'low', 'close', 'volume', 'symbol'}
    if not required <= set(data):
        raise ValueError('Feature input missing: '+', '.join(sorted(required-set(data))))
    data['date'] = data['date'].map(_day)
    if data.duplicated(['symbol', 'date']).any():
        raise ValueError('Duplicate stock sessions')
    data = data.sort_values(['symbol', 'date']).reset_index(drop=True)
    if data.symbol.nunique()>1 and any(not event.get('symbol') for event in events):
        raise ValueError('Multi-stock feature input requires event symbols')
    forecasts = forecasts or {}
    event_rows = []
    for event in events:
        if event.get('confirmed') is not True or event.get('input_eligible') is False:
            continue
        available = pd.Timestamp(event.get('available_at') or str(event['date'])[:10]+'T15:00:00+08:00')
        available = available.tz_localize('Asia/Shanghai') if available.tzinfo is None else available.tz_convert('Asia/Shanghai')
        event_rows.append((available, event))
    event_rows.sort(key=lambda item: item[0])
    for symbol, indexes in data.groupby('symbol', sort=False).groups.items():
        group = data.loc[indexes].copy()
        close, high, low, volume = (pd.to_numeric(group[k], errors='coerce') for k in ('close', 'high', 'low', 'volume'))
        spread = high-low
        top, bottom = high.rolling(20, min_periods=20).max(), low.rolling(20, min_periods=20).min()
        group['range_position20'] = (close-bottom)/(top-bottom).replace(0, np.nan)
        group['volume_trend5_20'] = volume.rolling(5, min_periods=5).mean()/volume.rolling(20, min_periods=20).mean().replace(0, np.nan)-1
        group['intraday_range_ratio'] = spread/close.replace(0, np.nan)
        group['close_location'] = (close-low)/spread.replace(0, np.nan)
        ma20, ma60 = close.rolling(20, min_periods=20).mean(), close.rolling(60, min_periods=60).mean()
        group['trend_alignment'] = ((close-ma20)/close.replace(0,np.nan)+(ma20-ma60)/close.replace(0,np.nan))/2
        group['demand_supply_gap'] = pd.to_numeric(group.get('native_w_demand_score',np.nan))-pd.to_numeric(group.get('native_w_supply_score',np.nan))
        known = [(at,e) for at,e in event_rows if e.get('symbol',symbol)==symbol]
        cursor, last_index, support, resistance, balance = 0, None, np.nan, np.nan, 0
        ages, supports, resistances, balances = [], [], [], []
        for ordinal, (_, row) in enumerate(group.iterrows()):
            cutoff = row.date.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=15)
            while cursor < len(known) and known[cursor][0] <= cutoff:
                event = known[cursor][1]
                side = str(event.get('side','')).upper()
                level = event.get('risk_boundary',event.get('risk_stop_price',event.get('invalidation_price')))
                if side in {'B','BUY'}:
                    balance += 1
                    support = float(level) if level is not None else np.nan
                elif side in {'S','SELL'}:
                    balance -= 1
                    resistance = float(level) if level is not None else np.nan
                last_index = ordinal
                cursor += 1
            ages.append(np.nan if last_index is None else ordinal-last_index)
            supports.append((row.close-support)/row.close if row.close>0 and np.isfinite(support) else np.nan)
            resistances.append((resistance-row.close)/row.close if row.close>0 and np.isfinite(resistance) else np.nan)
            balances.append(balance if last_index is not None else np.nan)
        group['structure_age_sessions'],group['support_distance'],group['resistance_distance'],group['confirmed_balance'] = ages,supports,resistances,balances
        for column in EXTRA_FEATURES:
            data.loc[indexes,column] = group[column]
    data['as_of'] = data.date.dt.strftime('%Y-%m-%d')
    data['feature_available_at'] = data.as_of+'T15:00:00+08:00'
    if variant=='B':
        records=[]
        for row in data.itertuples():
            day=row.date.strftime('%Y-%m-%d')
            observation=forecasts.get((row.symbol,day),forecasts.get(day))
            values={key:np.nan for key in KRONOS_FEATURES}
            state='missing'
            bar_close=pd.Timestamp(day+'T15:00:00+08:00')
            feature_available=bar_close
            if observation:
                availability=[observation[key] for key in ('actual_available_at','available_at') if observation.get(key) is not None]
                stamp=max(_available_stamp(value) for value in availability) if availability else _available_stamp(observation.get('as_of',day))
                declared=decision_time.get((row.symbol,day),decision_time.get(day)) if isinstance(decision_time,dict) else decision_time
                cutoff=_available_stamp(declared) if declared is not None else bar_close
                if cutoff<bar_close:raise ValueError('Daily close features unavailable before the close')
                if stamp>cutoff:
                    raise ValueError('Kronos evidence became available after its decision')
                if observation.get('execution_deadline') is not None and cutoff>=_available_stamp(observation['execution_deadline']):
                    raise ValueError('Kronos decision reached or passed its executable open')
                feature_available=max(bar_close,stamp)
                identity=observation.get('identity') or {}
                identity_input=identity.get('input') or {}
                known_as_of=str(observation.get('as_of',identity_input.get('as_of',day)))[:10]
                state='expired' if known_as_of!=day else 'invalid'
                if state!='expired' and (observation.get('quality') or {}).get('prediction_usable') is True:
                    summary=observation.get('summary') or {}
                    values=dict(zip(KRONOS_FEATURES,[summary.get('terminal_return_median'),summary.get('terminal_return_p10'),
                        summary.get('terminal_return_p90'),None,summary.get('up_path_fraction'),
                        (observation.get('quality') or {}).get('valid_fraction')]))
                    if all(values[k] is not None and np.isfinite(values[k]) for k in KRONOS_FEATURES if k!='kronos_interval_width'):
                        values['kronos_interval_width']=values['kronos_return_p90']-values['kronos_return_p10']
                        state='valid' if (values['kronos_return_p10']<=values['kronos_return_median']<=values['kronos_return_p90']
                            and 0<=values['kronos_up_fraction']<=1 and .8<=values['kronos_valid_fraction']<=1) else 'invalid'
                    if state!='valid': values={key:np.nan for key in KRONOS_FEATURES}
            records.append({**values,'kronos_state':state,'feature_available_at':feature_available.isoformat()})
        for column in [*KRONOS_FEATURES,'kronos_state','feature_available_at']:
            data[column]=[record[column] for record in records]
    return data.replace([np.inf,-np.inf],np.nan)


LABEL_SCHEMA = 'verified_direction_label_v2'


def direction_label_contract(horizon=5, threshold=.003):
    return {'schema': LABEL_SCHEMA, 'horizon': horizon, 'threshold': threshold,
            'entry': 'next_verified_open', 'exit': 'fifth_session_close' if horizon == 5 else 'horizon_session_close',
            'future_price_qualification': 'every_horizon_session_verified_active_finite',
            'availability': 'explicit_bar_availability_by_session_close_or_verified_daily_close',
            'fee_threshold_is_estimate': True}


def make_direction_training_frame(features, calendar_dates, *, horizon=5, threshold=.003):
    """Label only complete verified future price paths and retain exclusion facts.

    Labels describe prices, not a Broker execution or an investment qualification.
    A suspended or unverified bar is never replaced by a previous close, including
    the entry/terminal dates. Explicit availability after that session's close is
    ineligible; absent availability uses the verified daily price contract.
    """
    from collections import Counter
    from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES
    if isinstance(horizon,bool) or not 1<=int(horizon)<=20 or int(horizon)!=horizon:
        raise ValueError('horizon must be an integer in 1..20')
    if not np.isfinite(threshold) or threshold<=0:
        raise ValueError('threshold must be positive')
    calendar=[_day(day) for day in calendar_dates]
    if calendar!=sorted(set(calendar)):
        raise ValueError('Calendar must be sorted and unique')
    lookup={date:i for i,date in enumerate(calendar)}
    output=[];exclusions=[];counts=Counter()
    for symbol,group in features.groupby('symbol',sort=False):
        group=group.sort_values('date').copy().reset_index(drop=True)
        group['date']=group.date.map(_day)
        if group.date.duplicated().any():raise ValueError('Duplicate stock sessions')
        positions=group.date.map(lookup)
        entry=pd.to_numeric(group.open,errors='coerce').shift(-1)
        future=pd.to_numeric(group.close,errors='coerce').shift(-horizon)
        price=group.reindex(columns=['open','high','low','close']).apply(pd.to_numeric,errors='coerce')
        quantity=group.reindex(columns=['volume','amount']).apply(pd.to_numeric,errors='coerce')
        finite=np.isfinite(price.to_numpy(float)).all(axis=1)&(price.to_numpy(float)>0).all(axis=1)
        geometry=(price.high>=price[['open','close','low']].max(axis=1))&(price.low<=price[['open','close','high']].min(axis=1))
        source=group.get('source',pd.Series('',index=group.index)).fillna('').astype(str)
        verified=pd.to_numeric(group.get('has_trade_price',pd.Series(0,index=group.index)),errors='coerce').eq(1)
        available=[]
        for i,row in group.iterrows():
            value=row.get('available_at')
            if value is None or (not isinstance(value,(dict,list)) and pd.isna(value)):
                available.append(True)
            else:
                try:available.append(_available_stamp(value)<=row.date.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=15))
                except (ValueError,TypeError):available.append(False)
        masks={'unverified_source':~source.isin(VERIFIED_SOURCES).to_numpy(),
               'trade_price_unverified':~verified.to_numpy(),
               'invalid_price':~finite|~geometry.to_numpy(),
               'invalid_quantity':~np.isfinite(quantity.to_numpy(float)).all(axis=1)|(quantity.to_numpy(float)<0).any(axis=1),
               'suspended':(quantity.to_numpy(float)==0).all(axis=1),
               'volume_amount_mismatch':((quantity.volume==0)!=(quantity.amount==0)).to_numpy(),
               'unavailable_price':~np.array(available,dtype=bool)}
        reasons=[[] for _ in range(len(group))]
        for i in range(len(group)):
            if i+horizon>=len(group):reasons[i].append('immature_label');continue
            path=positions.iloc[i:i+horizon+1].to_numpy(float)
            if not np.isfinite(path).all() or not np.array_equal(np.diff(path),np.ones(horizon)):
                reasons[i].append('calendar_gap')
            for name,mask in masks.items():
                bad=np.flatnonzero(mask[i+1:i+horizon+1])+1
                if len(bad):
                    for role in dict.fromkeys('entry' if at==1 else 'terminal' if at==horizon else 'intermediate' for at in bad):
                        reasons[i].append('label_'+role+'_'+name)
        returns=future/entry-1
        group['label']=np.where(returns>threshold,2,np.where(returns< -threshold,0,1))
        group['label_mature_at']=group.date.shift(-horizon).dt.strftime('%Y-%m-%d')
        group['as_of']=group.date.dt.strftime('%Y-%m-%d')
        group['label_schema']=LABEL_SCHEMA
        valid=np.array([not codes for codes in reasons])
        for i,codes in enumerate(reasons):
            if codes:
                exclusions.append({'symbol':str(symbol),'as_of':group.as_of.iloc[i],
                    'label_mature_at':None if pd.isna(group.label_mature_at.iloc[i]) else group.label_mature_at.iloc[i],
                    'reason_codes':codes})
                counts.update(codes)
        output.append(group.loc[valid].copy())
    if output:result=pd.concat(output,ignore_index=True).sort_values(['as_of','symbol']).reset_index(drop=True)
    else:result=features.copy().assign(label=pd.Series(dtype=int),label_mature_at=pd.Series(dtype=str),as_of=pd.Series(dtype=str),label_schema=LABEL_SCHEMA)
    result.attrs['label_audit']={'schema':LABEL_SCHEMA,'contract':direction_label_contract(horizon,threshold),
        'candidate_rows':len(features),'eligible_rows':len(result),'excluded_rows':len(exclusions),
        'reason_counts':dict(sorted(counts.items())),'exclusions':exclusions}
    return result


def chronological_split_ranges(frame, *, fractions=(.55,.13,.11,.10,.11)):
    days=sorted(set(frame['as_of'].astype(str).str[:10]))
    if len(days)<50 or len(fractions)!=5 or not np.isclose(sum(fractions),1) or min(fractions)<=0:
        raise ValueError('Need at least 50 distinct dates and five positive split fractions')
    boundaries=[0,*[int(len(days)*x) for x in np.cumsum(fractions)[:-1]],len(days)]
    if len(set(boundaries))!=6:
        raise ValueError('Empty chronological split')
    return {name:[days[boundaries[i]],days[boundaries[i+1]-1]] for i,name in enumerate(('train','tune','calibration','policy','test'))}
