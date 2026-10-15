"""Deterministic, causal observations shared by replay and daily accounts.

Agents describe evidence; they are not independent votes. In particular an MLP
which consumes price/volume/Kronos features is the fusion layer, not a fourth
vote added back to its own inputs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

VERSION = 'agent_evidence_v1'


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError): return None


def _stable(value):
    if isinstance(value, dict): return {k:_stable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [_stable(v) for v in value]
    if isinstance(value,float): return round(value,12) if math.isfinite(value) else None
    return value


def identity(value):
    return hashlib.sha256(json.dumps(_stable(value), sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), default=str).encode()).hexdigest()


def observation(kind, day, *, source_id=None, direction=None, score=None,
                quality=None, missing_reason=None, evidence_refs=(), valid_until=None, **extra):
    return dict(agent=kind, as_of=day, available_at=day+'T15:00:00+08:00',
        source_id=source_id, feature_version=VERSION, valid_until=valid_until or day,
        direction=direction, score=number(score), quality=_stable(quality or {}),
        missing_reason=missing_reason, evidence_refs=list(evidence_refs), **extra)


def agents_for(row, event, day, *, held=0, cash=None, equity=None, hard_risk=None):
    row, event = row or {}, event or {}
    boundary = number(event.get('risk_boundary', event.get('stop_price')))
    close = number(row.get('close'))
    valid_structure = bool(event.get('confirmed') and boundary and close and 0 < boundary < close)
    trend = all(number(row.get(k)) is not None for k in ('_ma', '_previous_ma', 'close'))
    trend = bool(trend and close > row['_ma'] >= row['_previous_ma'])
    forecast = event.get('forecast') or {}
    summary = forecast.get('summary') or {}
    usable = (forecast.get('quality') or {}).get('prediction_usable') is True
    median = number(summary.get('terminal_return_median', summary.get('expected_return')))
    fraction = number(summary.get('up_path_fraction', summary.get('up_fraction')))
    score = event.get('score') or {}
    refs = [event['event_id']] if event.get('event_id') else []
    return {
        'structure': observation('structure', day, source_id=event.get('event_id'),
            direction='BUY' if valid_structure else None,
            quality={'confirmed':event.get('confirmed') is True, 'boundary':boundary,
                     'candidate_age_sessions':event.get('candidate_age_sessions', 0)},
            missing_reason=None if valid_structure else 'structure_unconfirmed_or_invalidated', evidence_refs=refs),
        'price_volume': observation('price_volume', day, source_id=identity([day, row]),
            direction='BUY' if trend else 'WAIT',
            quality={key:row.get(key) for key in ('_ma','_previous_ma','vol_ratio20','amount_ratio20',
                'atr14_ratio','native_w_demand_score','native_w_supply_score')},
            missing_reason=None if close else 'price_unavailable', evidence_refs=refs),
        'kronos': observation('kronos', day, source_id=identity(forecast.get('identity',{})) if forecast else None,
            direction=('BUY' if median > 0 else 'SELL') if usable and median is not None else None,
            score=None, quality={**(forecast.get('quality') or {}), 'median_return':median,
                'up_path_fraction':fraction, 'fraction_is_calibrated_probability':False},
            missing_reason=None if usable else forecast.get('reason','forecast_missing_or_unusable'),
            evidence_refs=[forecast.get('cache_key')] if forecast.get('cache_key') else [],
            actual_available_at=forecast.get('actual_available_at')),
        'fusion': observation('fusion', day, source_id=score.get('model_version'),
            direction='BUY' if score.get('status') == 'scored' and number(score.get('score')) is not None
                and number(score['score']) >= 50 else None,
            score=score.get('score'), quality={'qualification':score.get('qualification') or {},
                'probabilities':score.get('probabilities'), 'model_type':score.get('model_type'),
                'input_variant':score.get('variant'), 'independent_vote':False},
            missing_reason=None if score.get('status') == 'scored' else score.get('reason','score_unavailable')),
        'risk': observation('risk', day, source_id=identity([day,held,cash,equity,hard_risk]),
            direction='SELL' if hard_risk else 'WAIT', quality={'shares':held,'cash':cash,
                'equity':equity,'hard_risk':hard_risk,'cannot_be_vetoed_by_models':True})}


def qualification_reasons(score, mode='historical_candidate'):
    if not isinstance(score, dict) or score.get('status') != 'scored': return ['simple_score_pending']
    q = score.get('qualification') or {}
    # Explicit allow, never "no field was false". Historical candidate trials
    # evaluate a frozen model; paper promotion remains a separate evidence event.
    eligible = q.get('allow_increase') is True
    if mode == 'historical_candidate':
        eligible = eligible and q.get('research_trial_eligible') is True
    else:
        eligible = eligible and q.get('policy_passed') is True
    return [] if eligible else ['simple_score_policy_not_passed']


def opportunity_funnel(opportunities, orders=(), trades=(), cycles=()):
    reasons = Counter(r for opportunity in opportunities for r in opportunity.get('all_reasons', []))
    primary = Counter(o.get('primary_reason') or 'eligible' for o in opportunities)
    accepted = sum(not o.get('all_reasons') for o in opportunities)
    return {'unit':'symbol_candidate_decision_day', 'evaluated':len(opportunities),
        'eligible':accepted, 'blocked':len(opportunities)-accepted,
        'unique_opportunities':len({o.get('opportunity_id') for o in opportunities}),
        'primary_reason_counts':dict(primary), 'all_reason_counts':dict(reasons),
        'orders':len(orders), 'fills':len(trades), 'completed_cycles':len(cycles),
        'conservation_passed':sum(primary.values()) == len(opportunities)}


def volume_candidates(frame, symbol):
    """Frozen breakout + demand rule; only past rolling prices enter the trigger."""
    import pandas as pd
    # Candidates use columns only. Do not copy potentially large source-event
    # metadata into every Series yielded by pandas.iterrows().
    frame = pd.DataFrame(frame, copy=False)
    result = []
    prior_high = frame.high.shift(1).rolling(20, min_periods=20).max()
    prior_low = frame.low.shift(1).rolling(20, min_periods=20).min()
    for i, (_, row) in enumerate(frame.iterrows()):
        close = number(row.get('close')); demand = number(row.get('native_w_demand_score'))
        supply = number(row.get('native_w_supply_score')); volume = number(row.get('vol_ratio20'))
        hi, lo = number(prior_high.iloc[i]), number(prior_low.iloc[i])
        if (close is None or hi is None or lo is None or demand is None or supply is None
                or volume is None or not close > hi or volume < 1.2 or demand <= supply
                or not 0 < lo < close or row.get('native_structure_input_eligible') is not True):
            continue
        day = str(row['date'])[:10]
        result.append(dict(symbol=symbol,date=day,available_at=day+'T15:00:00+08:00',
            event_id='price-volume:'+identity([symbol,day,hi,lo,VERSION])[:24],
            side='BUY',confirmed=True,input_eligible=True,risk_boundary=lo,
            source='price_volume_breakout_v1',source_family='price_volume',
            observation_kind='confirmed_price_volume_candidate'))
    return result
