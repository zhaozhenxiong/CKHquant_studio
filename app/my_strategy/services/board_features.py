"""Causal Tonghuashun board context and explicit current-membership diagnostics.

These are equal-weight research summaries of stored unadjusted observations,
not official board indices or executable portfolio returns. No current member
is assigned to an earlier decision. Existing model contracts are untouched.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import _connection, normalize_symbol
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES
from my_strategy.services.board_calculation_scope import (CALCULATION_SCOPE, calculation_membership,
    is_calculation_symbol, verify_calculation_membership)


BOARD_FEATURE_COLUMNS = (
    'board_return_1', 'board_return_5', 'board_return_10', 'board_return_20',
    'board_advance_ratio', 'board_above_ma20_ratio', 'board_breakout20_ratio',
    'board_amount', 'board_market_amount_share', 'board_relative_amount20',
    'board_volatility20', 'board_drawdown20',
)
STOCK_BOARD_FEATURE_COLUMNS = tuple('industry_' + name[6:] for name in BOARD_FEATURE_COLUMNS) + (
    'relative_industry_1', 'relative_industry_5', 'relative_industry_20', 'industry_peer_rank20',
    'concept_return_1_mean', 'concept_return_5_mean', 'concept_return_20_mean',
    'concept_advance_ratio_mean', 'concept_positive_ratio20', 'concept_known_count',
)
DEFAULT_BOARD_POLICY = dict(min_members=5, min_coverage=.8, volatility_sessions_per_year=252)
BOARD_FEATURE_PROFILE = dict(name='tonghuashun_board_context', columns=list(STOCK_BOARD_FEATURE_COLUMNS),
    calculation_scope=CALCULATION_SCOPE,
    board_columns=list(BOARD_FEATURE_COLUMNS), missing='null_not_zero', price_basis='verified_stored_unadjusted',
    formulas=dict(returns='compound_daily_mean_simple_returns;complete_verified_sessions',
        breadth='valid_members_only;metric_specific_min_members_and_coverage',
        ma20='close>mean(last20_closes_including_current)',
        breakout20='close>max(previous20_highs)',
        amount='CNY_sum_of_qualified_current_observations',
        market_share='board_amount/local_observed_SH_SZ_A_share_amount',
        relative_amount='current_amount/mean(previous20_board_amounts)',
        volatility='sample_std(last20_daily_returns)*sqrt(252)',
        drawdown='max_peak_to_trough_of_last20_returns_with_initial_NAV_1',
        stock_relative='stock_simple_interval_return-compounded_peer_daily_mean_return',
        rank='fraction_of_valid_peers_below_stock_plus_half_ties;self_excluded',
        concept='one_row_per_stock;mean_across_qualified_self_excluded_concepts'))
BOARD_SCHEMA_HASH = stable_hash(BOARD_FEATURE_PROFILE)
PRICE_COLUMNS = ('symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'source', 'has_trade_price')


def board_policy(value=None):
    value=value or {}
    if not isinstance(value,dict) or set(value)-set(DEFAULT_BOARD_POLICY):
        raise ValueError('unknown board feature policy')
    result={**DEFAULT_BOARD_POLICY,**value}
    if type(result['min_members']) is not int or result['min_members']<2:
        raise ValueError('board min_members must be an integer of at least 2')
    if (isinstance(result['min_coverage'],bool) or not isinstance(result['min_coverage'],(int,float)) or
            not math.isfinite(result['min_coverage']) or not 0<result['min_coverage']<=1):
        raise ValueError('board coverage must be in (0,1]')
    if result['volatility_sessions_per_year']!=252:raise ValueError('board volatility uses frozen 252-session scaling')
    return result


def _timestamp(value):
    stamp=pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:raise ValueError('board evidence time must include a timezone')
    return stamp.tz_convert('Asia/Shanghai')


def _close_time(day):
    return pd.Timestamp(day+'T15:00:00+08:00')


def _calendar(values, day):
    dates=[pd.Timestamp(value).date().isoformat() for value in values]
    if not dates or dates!=sorted(set(dates)) or day not in dates:
        raise ValueError('board price date must be in an ascending unique verified calendar')
    return [value for value in dates if value<=day]


def _snapshot(value, cutoff=None):
    row=dict(value);projection=calculation_membership(value)
    # This internal calculation view is never written back to the source KB.
    row['members']=list(projection['eligible_members'])
    row['calculation_membership']=projection
    row['calculation_scope']=CALCULATION_SCOPE
    if row.get('kind') not in {'industry','concept'} or not row.get('board_id') or not row.get('snapshot_hash'):
        raise ValueError('board snapshot identity incomplete')
    if row.get('source') not in {'tonghuashun_public','tonghuashun_quant','tonghuashun','fuyao:ths'}:
        raise ValueError('board features require explicit Tonghuashun provenance')
    for key in ('observed_at','available_at','effective_at'):row[key]=_timestamp(row[key]).isoformat()
    if cutoff is not None and any(_timestamp(row[key])>cutoff for key in ('observed_at','available_at','effective_at')):
        raise ValueError('future board membership is not visible at this decision')
    row['expected_count']=projection['eligible_count']
    row['usable_membership']=projection['usable_membership']
    return row


def _read_prices(db_path, dates):
    with _connection(db_path) as conn:
        columns={row[1] for row in conn.execute('PRAGMA table_info(stock_daily_normalized)')}
        if not (set(PRICE_COLUMNS)-{'symbol'} | {'stock'})<=columns:
            raise ValueError('raw board price/source fields missing')
        available=',available_at' if 'available_at' in columns else ''
        frame=pd.read_sql_query('SELECT stock AS symbol,date,open,high,low,close,volume,amount,source,has_trade_price'+
            available+' FROM stock_daily_normalized WHERE date>=? AND date<=? ORDER BY stock,date',
            conn,params=[dates[0],dates[-1]])
    if frame.empty:return frame,pd.DataFrame(index=dates),{},stable_hash([])
    # Excluded exchanges must not enter the price content hash, pivots, or the
    # whole-market amount denominator, even when their quotes are valid.
    frame=frame.loc[frame.symbol.map(is_calculation_symbol)].copy()
    if frame.empty:return frame,pd.DataFrame(index=dates),{},stable_hash([])
    frame['symbol']=frame.symbol.map(normalize_symbol)
    frame['date']=pd.to_datetime(frame.date,errors='raise').dt.strftime('%Y-%m-%d')
    frame=frame.loc[frame.date.isin(dates)].sort_values(['date','symbol']).reset_index(drop=True)
    raw_bytes=frame.to_csv(index=False,float_format='%.17g').encode('utf-8')
    identity=hashlib.sha256(raw_bytes).hexdigest()
    duplicate=frame.duplicated(['symbol','date'],keep=False)
    numeric=frame[['open','high','low','close','volume','amount']].apply(pd.to_numeric,errors='coerce')
    prices=numeric[['open','high','low','close']]
    valid=(~duplicate & frame.source.isin(VERIFIED_SOURCES) & frame.has_trade_price.eq(1) &
        np.isfinite(numeric).all(axis=1) & prices.gt(0).all(axis=1) & numeric.volume.ge(0) & numeric.amount.ge(0) &
        numeric.high.ge(prices.max(axis=1)) & numeric.low.le(prices.min(axis=1)))
    if 'available_at' in frame:
        known=pd.to_datetime(frame.available_at,errors='coerce',utc=True)
        cutoff=pd.to_datetime(frame.date+'T15:00:00+08:00',utc=True)
        valid &=known.notna() & known.le(cutoff)
    frame[numeric.columns]=numeric
    frame['_valid']=valid
    unique=frame.drop_duplicates(['symbol','date'],keep='first')
    matrices={}
    for column in ('close','high','amount'):
        matrices[column]=unique.assign(_value=unique[column].where(unique._valid)).pivot(
            index='date',columns='symbol',values='_value').reindex(dates)
    close=matrices['close']
    matrices['return']=close/close.shift(1)-1
    matrices['ma20']=close.rolling(20,min_periods=20).mean()
    matrices['high20']=matrices['high'].shift(1).rolling(20,min_periods=20).max()
    matrices['advance']=matrices['return'].where(matrices['return'].isna(),matrices['return'].gt(0).astype(float))
    matrices['above_ma20']=close.gt(matrices['ma20']).astype(float).where(close.notna() & matrices['ma20'].notna())
    matrices['breakout20']=close.gt(matrices['high20']).astype(float).where(close.notna() & matrices['high20'].notna())
    return frame,close,matrices,identity


def _number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _metric(values, expected, cfg, *, total=False):
    values=[float(value) for value in values if value is not None and math.isfinite(float(value))]
    count=len(values);coverage=count/expected if expected else 0.
    if count<cfg['min_members'] or coverage<cfg['min_coverage']:return None,count,coverage
    return (sum(values) if total else sum(values)/count),count,coverage


def _compound(values):
    if not values or any(value is None for value in values):return None
    return float(np.prod(1+np.array(values,dtype=float))-1)


def _code_hashes(store):
    from my_strategy.services import board_calculation_scope
    paths=[Path(__file__),Path(board_calculation_scope.__file__)]
    try:paths.append(Path(inspect.getfile(type(store))))
    except (TypeError,OSError):pass
    return {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _member_board_index(snapshots):
    """Preserve snapshot order while indexing each verified membership once."""
    result={}
    for snapshot in snapshots:
        for symbol in snapshot['members']:
            result.setdefault(symbol,[]).append(snapshot)
    return result


def build_board_features(store, *, raw_db_path=None, calendar_dates, price_date, mode='causal',
                         policy=None, symbols=None, use_cache=True, component_store=None, component_stats=None):
    """Read current/causal membership and raw prices; never mutate the price DB."""
    if mode not in {'causal','current_reference'}:raise ValueError('unknown board feature mode')
    day=pd.Timestamp(price_date).date().isoformat();calendar=_calendar(calendar_dates,day)
    cfg=board_policy(policy);code=_code_hashes(store);cutoff=_close_time(day)
    dates=calendar[-41:];output_dates=dates[-21:]
    by_date={}
    if mode=='current_reference':
        current=[_snapshot(value) for value in store.catalog(current=True)]
        by_date={date:{row['board_id']:row for row in current} for date in output_dates}
    else:
        for date in output_dates:
            rows=[_snapshot(value,_close_time(date)) for value in store.catalog(as_of=_close_time(date).isoformat())]
            by_date[date]={row['board_id']:row for row in rows}
        current=list(by_date[day].values())
    frame,close,matrices,price_hash=_read_prices(raw_db_path,dates)
    requested=None if symbols is None else sorted(set(normalize_symbol(symbol) for symbol in symbols if is_calculation_symbol(symbol)))
    membership_identity={date:{board:row['snapshot_hash'] for board,row in rows.items()} for date,rows in by_date.items()}
    projection_identity={date:{board:row['calculation_membership']['projection_hash'] for board,row in rows.items()}
        for date,rows in by_date.items()}
    identity=dict(profile=BOARD_FEATURE_PROFILE,schema_hash=BOARD_SCHEMA_HASH,mode=mode,price_date=day,
        calculation_scope=CALCULATION_SCOPE,calculation_projections=projection_identity,
        membership_snapshots=membership_identity,price_content_sha256=price_hash,price_rows=len(frame),
        calendar_hash=stable_hash(calendar),policy=cfg,code_hashes=code,symbols=requested,
        verified_price_sources=sorted(VERIFIED_SOURCES),price_basis='stored_unadjusted_no_total_return_factor')
    key=stable_hash(identity)
    if use_cache:
        cached=store.get_feature('daily',key)
        if cached is not None:
            if (cached.get('cache_identity')!=identity or cached.get('cache_key')!=key or
                    cached.get('schema_hash')!=BOARD_SCHEMA_HASH or cached.get('profile')!=BOARD_FEATURE_PROFILE['name'] or
                    cached.get('policy_hash')!=stable_hash(cfg) or cached.get('mode')!=mode or cached.get('price_date')!=day):
                raise ValueError('board feature cache identity drift')
            if cached.get('calculation_scope')!=CALCULATION_SCOPE:raise ValueError('board calculation scope drift')
            for board in cached.get('boards',[]):
                verify_calculation_membership(board.get('calculation_membership'))
                if board['calculation_membership']!=by_date[day][board['board_id']]['calculation_membership']:
                    raise ValueError('board cached calculation membership drift')
            return cached
    daily={};lookups={name:matrix.to_dict(orient='index') for name,matrix in matrices.items()}
    def observation(name,date,symbol):return _number(lookups.get(name,{}).get(date,{}).get(symbol))
    market={date:_number(matrices['amount'].loc[date].sum(min_count=1)) if matrices else None for date in output_dates}

    def daily_row(snapshot,date,exclude=None):
        if exclude is not None and (snapshot['board_id'],date) in daily:
            original=daily[(snapshot['board_id'],date)]
            contained=exclude in original['member_set'];expected=original['expected_count']-int(contained)
            result={**original,'expected_count':expected,'metrics':{},'metric_counts':{},'metric_coverage':{}}
            for name in ('return','advance','above_ma20','breakout20','amount'):
                own=observation(name,date,exclude) if contained else None
                count=original['metric_counts'][name]-int(own is not None)
                total=original['metric_sums'][name]-(own or 0.)
                coverage=count/expected if expected else 0.
                usable=snapshot['usable_membership'] and count>=cfg['min_members'] and coverage>=cfg['min_coverage']
                result['metrics'][name]=total if usable and name=='amount' else total/count if usable else None
                result['metric_counts'][name]=count;result['metric_coverage'][name]=coverage
            result['effective_members']=original['effective_members']-int(contained and observation('close',date,exclude) is not None)
            result['coverage']=result['effective_members']/expected if expected else 0.
            denominator=market.get(date)
            if denominator is not None:denominator-=observation('amount',date,exclude) or 0.
            amount=result['metrics']['amount']
            result['metrics']['market_share']=amount/denominator if amount is not None and denominator is not None and denominator>0 else None
            return result
        members=[symbol for symbol in snapshot['members'] if symbol!=exclude];expected=len(members)
        result={'snapshot_hash':snapshot['snapshot_hash'],'expected_count':expected,'members':members,
            'member_set':set(members),'snapshot':snapshot,'date':date,'metrics':{},'metric_counts':{},'metric_coverage':{},'metric_sums':{}}
        for name in ('return','advance','above_ma20','breakout20','amount'):
            observations=[observation(name,date,symbol) for symbol in members]
            value,count,coverage=_metric(observations,expected,cfg,total=name=='amount')
            if not snapshot['usable_membership']:value=None
            result['metrics'][name]=value;result['metric_counts'][name]=count;result['metric_coverage'][name]=coverage
            result['metric_sums'][name]=sum(value for value in observations if value is not None)
        result['effective_members']=sum(observation('close',date,symbol) is not None for symbol in members)
        result['coverage']=result['effective_members']/expected if expected else 0.
        denominator=market.get(date)
        if exclude and denominator is not None:
            denominator-=observation('amount',date,exclude) or 0.
        amount=result['metrics']['amount']
        result['metrics']['market_share']=amount/denominator if amount is not None and denominator is not None and denominator>0 else None
        return result

    # A current-board component depends only on its members' observations.
    # Global market totals and rankings remain cheap derived values. Changes in
    # another board must not invalidate this board's member sums/counts.
    if component_store is not None and mode=='current_reference':
        for snapshot in current:
            members=snapshot['members'];board=snapshot['board_id']
            local_identity={'format':'board_member_sums_v2','snapshot_hash':snapshot['snapshot_hash'],
                'calculation_scope':CALCULATION_SCOPE,'projection_hash':snapshot['calculation_membership']['projection_hash'],
                'policy':cfg,'code':code,'dates':output_dates,
                'observations':[[date,[[symbol,*[observation(field,date,symbol) for field in
                    ('close','return','advance','above_ma20','breakout20','amount')]] for symbol in members]] for date in output_dates]}
            component_key={'content_hash':stable_hash(local_identity)}
            cached=component_store.get(component_key,namespace='board_member_components') if use_cache else None
            if cached is not None and cached.get('identity')!=local_identity:raise ValueError('board component identity drift')
            if component_stats is not None:
                name='reused' if cached is not None else 'rebuilt';component_stats[name]=component_stats.get(name,0)+1
            payload=[]
            for date in output_dates:
                if cached is None:
                    row=daily_row(snapshot,date)
                    slim={key:row[key] for key in ('date','snapshot_hash','expected_count','members','metric_counts','metric_coverage',
                        'metric_sums','effective_members','coverage')}
                    slim['metrics']={key:value for key,value in row['metrics'].items() if key!='market_share'}
                    payload.append(slim)
                else:
                    slim=next(item for item in cached['rows'] if item['date']==date)
                    row={**slim,'metrics':dict(slim['metrics']),'snapshot':snapshot,'member_set':set(members)}
                    amount=row['metrics']['amount'];denominator=market.get(date)
                    row['metrics']['market_share']=amount/denominator if amount is not None and denominator is not None and denominator>0 else None
                daily[(board,date)]=row
            if cached is None and use_cache:
                component_store.put(component_key,{'identity':local_identity,'rows':payload},namespace='board_member_components')
    else:
        for date,rows in by_date.items():
            for board,snapshot in rows.items():daily[(board,date)]=daily_row(snapshot,date)

    def window(board,field,count,exclude=None,previous=False):
        end=len(output_dates)-(1 if previous else 0);selected=output_dates[max(0,end-count):end]
        if len(selected)!=count:return None
        values=[]
        for date in selected:
            row=daily.get((board,date))
            if row is None:return None
            if exclude is None:value=row['metrics'][field]
            else:
                contained=exclude in row['member_set'];expected=row['expected_count']-int(contained)
                own=observation(field,date,exclude) if contained else None
                valid_count=row['metric_counts'][field]-int(own is not None)
                total=row['metric_sums'][field]-(own or 0.)
                usable=row['snapshot']['usable_membership'] and valid_count>=cfg['min_members'] and expected>0 and valid_count/expected>=cfg['min_coverage']
                value=total if usable and field=='amount' else total/valid_count if usable else None
            values.append(value)
        return values if all(value is not None for value in values) else None

    def board_values(board,exclude=None):
        current_row=daily[(board,day)]
        if exclude is not None:current_row=daily_row(current_row['snapshot'],day,exclude)
        metrics=current_row['metrics'];values={}
        for count in (1,5,10,20):
            returns=window(board,'return',count,exclude)
            values['board_return_'+str(count)]=_compound(returns) if returns is not None else None
        for column,field in [('board_advance_ratio','advance'),('board_above_ma20_ratio','above_ma20'),
                ('board_breakout20_ratio','breakout20'),('board_amount','amount'),('board_market_amount_share','market_share')]:
            values[column]=metrics[field]
        amounts=window(board,'amount',20,exclude,previous=True);amount=metrics['amount']
        mean=sum(amounts)/20 if amounts is not None else None
        values['board_relative_amount20']=amount/mean if amount is not None and mean is not None and mean>0 else None
        returns=window(board,'return',20,exclude)
        values['board_volatility20']=float(np.std(returns,ddof=1)*math.sqrt(252)) if returns is not None else None
        if returns is None:values['board_drawdown20']=None
        else:
            nav=np.r_[1.,np.cumprod(1+np.array(returns))]
            values['board_drawdown20']=float(np.max(1-nav/np.maximum.accumulate(nav)))
        return values,current_row

    all_symbols=sorted(set(frame.symbol)) if not frame.empty else []
    own_returns={}
    for symbol in all_symbols:
        series=close[symbol]
        for count in (1,5,20):
            prefix=series.iloc[-count-1:]
            own_returns[(symbol,count)]=float(prefix.iloc[-1]/prefix.iloc[0]-1) if len(prefix)==count+1 and prefix.notna().all() else None
    board_rows=[]
    for snapshot in sorted(current,key=lambda row:row['board_id']):
        values,row=board_values(snapshot['board_id'])
        # Display-only history is built in this explicit preparation job, not
        # reconstructed by GET. Current members never gain historical eligibility.
        capital_visualization=None
        if mode=='current_reference':
            history=[]
            for date in output_dates:
                point=daily[(snapshot['board_id'],date)]
                history.append({'date':date,'amount':point['metrics']['amount'],
                    'market_amount':market.get(date),'market_share':point['metrics']['market_share'],
                    'return_1':point['metrics']['return'],'coverage':point['metric_coverage']['amount'],
                    'effective_members':point['metric_counts']['amount'],'expected_members':point['expected_count']})
            comparable=[]
            if len(output_dates)>1:
                previous=output_dates[-2]
                comparable=[(observation('amount',day,symbol),observation('amount',previous,symbol))
                    for symbol in snapshot['members']]
                comparable=[pair for pair in comparable if all(value is not None for value in pair)]
            coverage=len(comparable)/snapshot['expected_count'] if snapshot['expected_count'] else 0.
            usable=(snapshot['usable_membership'] and len(comparable)>=cfg['min_members']
                and coverage>=cfg['min_coverage'] and history[-1]['amount'] is not None
                and len(history)>1 and history[-2]['amount'] is not None)
            current_amount=sum(pair[0] for pair in comparable) if usable else None
            previous_amount=sum(pair[1] for pair in comparable) if usable else None
            change=current_amount-previous_amount if usable else None
            relative=values['board_relative_amount20']
            capital_visualization={'version':'current_member_capital_visualization_v1',
                'mode':'current_reference','training_eligible':False,'capital_basis':'turnover_amount_proxy',
                'membership_basis':'fixed_current_snapshot_historical_price_comparison',
                'snapshot_hash':snapshot['snapshot_hash'],'history':history,
                'amount_change_1':change,
                'amount_change_ratio_1':change/previous_amount if usable and previous_amount>0 else None,
                'comparable_current_amount':current_amount,'comparable_previous_amount':previous_amount,
                'comparable_members':len(comparable),'comparable_coverage':coverage,
                'previous_date':output_dates[-2] if len(output_dates)>1 else None,
                'prior20_mean_amount':values['board_amount']/relative if relative is not None and relative>0 else None,
                'reason_codes':[] if usable else ['comparable_amount_population_unavailable']}
        eligible=mode=='causal' and all(value is not None for value in values.values())
        board_rows.append({**values,'board_id':snapshot['board_id'],'kind':snapshot['kind'],'name':snapshot['name'],
            'calculation_scope':CALCULATION_SCOPE,'calculation_membership':snapshot['calculation_membership'],
            'source_snapshot_hash':snapshot['snapshot_hash'],'projection_hash':snapshot['calculation_membership']['projection_hash'],
            **({'capital_visualization':capital_visualization} if capital_visualization is not None else {}),
            'effective_members':row['effective_members'],'expected_members':row['expected_count'],'coverage':row['coverage'],
            'membership_scope':snapshot.get('membership_scope','source_full'),
            'source_member_count':snapshot.get('source_member_count',snapshot['calculation_membership']['source_member_count']),
            'excluded_source_members':list(snapshot.get('excluded_source_members',[])),
            'metric_counts':row['metric_counts'],'metric_coverage':row['metric_coverage'],
            'membership_date':_timestamp(snapshot['observed_at']).date().isoformat(),'membership_available_at':snapshot['available_at'],
            'snapshot_hash':snapshot['snapshot_hash'],'price_date':day,'mode':mode,'training_eligible':eligible,
            'profile':BOARD_FEATURE_PROFILE['name'],'schema_hash':BOARD_SCHEMA_HASH,'available_at':max(cutoff,_timestamp(snapshot['available_at'])).isoformat(),
            'taxonomy':snapshot.get('taxonomy'),'industry_level':snapshot.get('level'),'hierarchy_verified':snapshot.get('hierarchy_verified',False),
            'policy_hash':stable_hash(cfg),'cache_key':key,'price_content_sha256':price_hash,'calendar_hash':identity['calendar_hash'],'code_hashes':code,
            'membership_dependency_hashes':sorted({rows[snapshot['board_id']]['snapshot_hash'] for rows in by_date.values() if snapshot['board_id'] in rows}),
            'status':'available' if all(value is not None for value in values.values()) else 'unavailable',
            'reason_codes':([] if snapshot['calculation_membership']['source_complete'] and
                snapshot['calculation_membership']['source_expected_count']==snapshot['calculation_membership']['source_member_count']
                else ['membership_incomplete'])+
                (['verified_empty_membership'] if snapshot['calculation_membership']['source_expected_count']==0 else [])+
                (['no_supported_members'] if snapshot['expected_count']==0 else [])+
                ([] if row['effective_members']>=cfg['min_members'] and row['coverage']>=cfg['min_coverage'] else ['members_or_coverage_below_frozen_minimum'])+
                ([] if eligible or mode=='current_reference' else ['historical_dependency_or_price_window_unavailable'])})
    stock_rows=[]
    selected=requested if requested is not None else sorted({symbol for snapshot in current for symbol in snapshot['members']})
    member_boards=_member_board_index(current)
    peer_cache={}
    for symbol in selected:
        relevant=member_boards.get(symbol,[])
        industries=[snapshot for snapshot in relevant if snapshot['kind']=='industry']
        concepts=[snapshot for snapshot in relevant if snapshot['kind']=='concept']
        values={name:None for name in STOCK_BOARD_FEATURE_COLUMNS};industry_id=None;peer_count=0;peer_coverage=None
        def peers(snapshot,concept=False):
            pair=(snapshot['board_id'],symbol)
            if pair not in peer_cache:
                if concept:
                    current_row=daily_row(snapshot,day,symbol)
                    peer_values={'board_return_'+str(count):_compound(window(snapshot['board_id'],'return',count,symbol)) for count in (1,5,20)}
                    peer_values['board_advance_ratio']=current_row['metrics']['advance']
                    peer_cache[pair]=(peer_values,current_row)
                else:peer_cache[pair]=board_values(snapshot['board_id'],symbol)
            return peer_cache[pair]
        hierarchy_unverified=any(snapshot.get('hierarchy_verified') is not True for snapshot in industries)
        if len(industries)==1 and (mode=='current_reference' or not hierarchy_unverified):
            industry=industries[0];industry_id=industry['board_id'];peer_values,peer_row=peers(industry)
            peer_count=peer_row['effective_members']
            peer_coverage=peer_row['coverage']
            values.update({'industry_'+name[6:]:value for name,value in peer_values.items()})
            for count in (1,5,20):
                own=own_returns.get((symbol,count));peer=peer_values['board_return_'+str(count)]
                values['relative_industry_'+str(count)]=own-peer if own is not None and peer is not None else None
            peer_returns=[own_returns.get((member,20)) for member in industry['members'] if member!=symbol]
            valid=[value for value in peer_returns if value is not None]
            own=own_returns.get((symbol,20))
            if own is not None and len(valid)>=cfg['min_members'] and len(valid)/len(peer_returns)>=cfg['min_coverage']:
                values['industry_peer_rank20']=(sum(value<own for value in valid)+.5*sum(value==own for value in valid))/len(valid)
        concept_values=[peers(snapshot,concept=True)[0] for snapshot in concepts]
        for count in (1,5,20):
            valid=[row['board_return_'+str(count)] for row in concept_values if row['board_return_'+str(count)] is not None]
            values['concept_return_'+str(count)+'_mean']=sum(valid)/len(valid) if valid else None
        advances=[row['board_advance_ratio'] for row in concept_values if row['board_advance_ratio'] is not None]
        values['concept_advance_ratio_mean']=sum(advances)/len(advances) if advances else None
        twenty=[row['board_return_20'] for row in concept_values if row['board_return_20'] is not None]
        values['concept_positive_ratio20']=sum(value>0 for value in twenty)/len(twenty) if twenty else None
        values['concept_known_count']=len(concepts) if relevant else None
        incomplete=any(value is None for value in values.values())
        peers_below_minimum=industry_id is not None and (peer_count<cfg['min_members'] or peer_coverage<cfg['min_coverage'])
        available_times=[cutoff,*[_timestamp(snapshot['available_at']) for snapshot in relevant]]
        stock_rows.append({**values,'symbol':symbol,'industry_id':industry_id,'industry_match_count':len(industries),
            'calculation_scope':CALCULATION_SCOPE,
            'calculation_projection_hashes':{row['board_id']:row['calculation_membership']['projection_hash'] for row in relevant},
            'industry_matches':[{'board_id':row['board_id'],'name':row['name'],'taxonomy':row.get('taxonomy'),
                'level':row.get('level'),'hierarchy_verified':row.get('hierarchy_verified',False)} for row in industries],
            'industry_hierarchy_verified':not hierarchy_unverified if industries else False,
            'concept_ids':sorted(snapshot['board_id'] for snapshot in concepts),'industry_peer_count':peer_count,
            'mode':mode,'profile':BOARD_FEATURE_PROFILE['name'],'schema_hash':BOARD_SCHEMA_HASH,'price_date':day,
            'membership_date':max((_timestamp(snapshot['observed_at']).date().isoformat() for snapshot in relevant),default=None),
            'snapshot_hashes':sorted(snapshot['snapshot_hash'] for snapshot in relevant),
            'membership_dependency_hashes':sorted({rows[snapshot['board_id']]['snapshot_hash'] for rows in by_date.values()
                for snapshot in relevant if snapshot['board_id'] in rows}),
            'policy_hash':stable_hash(cfg),'cache_key':key,'price_content_sha256':price_hash,'calendar_hash':identity['calendar_hash'],'code_hashes':code,
            'available_at':max(available_times).isoformat(),'peer_excludes_self':True,
            'training_eligible':mode=='causal' and observation('close',day,symbol) is not None and all(value is not None for value in values.values()),
            'status':'available' if all(value is not None for value in values.values()) else 'unavailable',
            'reason_codes':(['industry_membership_unknown'] if not industries else ['industry_membership_ambiguous'] if len(industries)>1 else [])+
                (['industry_hierarchy_unverified'] if hierarchy_unverified else [])+
                (['peer_members_or_coverage_below_frozen_minimum'] if peers_below_minimum else [])+
                (['rolling_price_or_membership_window_unavailable'] if incomplete else [])+
                ([] if observation('close',day,symbol) is not None else ['current_price_unavailable'])+
                ([] if concepts else ['concept_membership_unknown']),
            'reason_messages':(['行业指数层级尚未核验；多个行业指数不能擅定为唯一行业。'] if hierarchy_unverified or len(industries)>1 else [])+
                (['剔除自身后的同业有效成员或行情覆盖未达到冻结阈值。'] if peers_below_minimum else [])+
                ([('行业归属或行情窗口尚不完整，缺失特征保持空值。' if mode=='current_reference' else
                    '历史成员归属或行情窗口尚不完整，缺失特征保持空值。')] if incomplete else [])})
    if _code_hashes(store)!=code:raise ValueError('board generating code changed during feature preparation')
    result=dict(profile=BOARD_FEATURE_PROFILE['name'],schema_hash=BOARD_SCHEMA_HASH,mode=mode,price_date=day,
        calculation_scope=CALCULATION_SCOPE,
        calculation_membership={row['board_id']:row['calculation_membership'] for row in current},
        membership_date=max((_timestamp(row['observed_at']).date().isoformat() for row in current),default=None),
        available_at=max([cutoff,*[_timestamp(row['available_at']) for row in current]]).isoformat(),
        training_eligible=mode=='causal' and bool(stock_rows) and all(row['training_eligible'] for row in stock_rows),
        qualification='pending',historical_membership_status='current_snapshots_only_no_historical_backfill',
        price_basis='stored_unadjusted_no_total_return_factor',market_scope='local_verified_observed_SH_SZ_A_shares',
        policy=cfg,policy_hash=stable_hash(cfg),cache_key=key,cache_identity=identity,boards=board_rows,stocks=stock_rows,
        diagnostics=dict(price_rows=len(frame),board_count=len(board_rows),stock_count=len(stock_rows),
            current_membership_historical_replay=mode=='current_reference'))
    if use_cache:store.put_feature('daily',key,result)
    return result


def require_board_training_features(row, *, decision_at, expected_schema_hash=BOARD_SCHEMA_HASH,
                                     expected_policy_hash=None, expected_cache_key=None):
    """Read-time gate for future new datasets; never grants model qualification."""
    decision=_timestamp(decision_at)
    policy_hash=expected_policy_hash or stable_hash(DEFAULT_BOARD_POLICY)
    if (row.get('mode')!='causal' or row.get('training_eligible') is not True or
            row.get('calculation_scope')!=CALCULATION_SCOPE or not row.get('calculation_projection_hashes') or
            row.get('schema_hash')!=expected_schema_hash or row.get('profile')!=BOARD_FEATURE_PROFILE['name'] or
            row.get('price_date')!=decision.date().isoformat() or _timestamp(row['available_at'])>decision or
            not row.get('snapshot_hashes') or not row.get('membership_dependency_hashes') or
            row.get('policy_hash')!=policy_hash or not row.get('price_content_sha256') or not row.get('calendar_hash') or
            not row.get('code_hashes') or row.get('peer_excludes_self') is not True or
            expected_cache_key is not None and row.get('cache_key')!=expected_cache_key):
        raise ValueError('board training input is missing, diagnostic-only, future-visible or incompatible')
    values=[row.get(name) for name in STOCK_BOARD_FEATURE_COLUMNS]
    if any(_number(value) is None for value in values):raise ValueError('board training feature window incomplete')
    return [float(value) for value in values]
