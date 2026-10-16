"""Causal MA-only score inputs, without running native structure generation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS, get_feature_profile
from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES

PROFILE = 'ma_trend_v1'


def profile_contract():
    return {'name': PROFILE, 'input_profile': get_feature_profile('ma_trend'),
            'quality_window_bars': 120, 'minimum_active_bars': 60,
            'calendar_gap_policy': 'veto_trailing_quality_window_preserve_rows',
            'kronos_input': False, 'structure_input': False}


def prepare_ma_features(frame, *, calendar_dates=None):
    """Use the shared SMA definitions and the existing MA quality contract.

    Missing/unverified sources, quantity mismatches, suspension, and calendar
    gaps remain explicit; none is silently converted into an eligible score.
    """
    from my_strategy.services.czsc_market_features import _validate_input, _trailing_features
    data = pd.DataFrame(frame, copy=True)
    if data.empty or 'symbol' not in data:
        raise ValueError('MA features require nonempty stock history')
    if data.duplicated(['symbol', 'date']).any():
        raise ValueError('Duplicate stock sessions')
    output = []
    calendar = [str(day)[:10] for day in calendar_dates] if calendar_dates is not None else None
    if calendar is not None and calendar != sorted(set(calendar)):
        raise ValueError('Calendar must be sorted and unique')
    following = dict(zip(calendar[:-1], calendar[1:])) if calendar else {}
    known = set(calendar or [])
    for _, group in data.groupby('symbol', sort=False):
        bars = _validate_input(group.sort_values('date').reset_index(drop=True))
        trailing = _trailing_features(bars)
        for column in MA_TREND_COLUMNS:
            bars[column] = trailing[column]
        source = bars.get('source', pd.Series('', index=bars.index)).fillna('').astype(str)
        verified = pd.to_numeric(bars.get('has_trade_price', pd.Series(0, index=bars.index)), errors='coerce').eq(1)
        quantity = bars[['volume', 'amount']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        missing = ~np.isfinite(quantity).all(axis=1)
        suspended = (quantity == 0).all(axis=1) & ~missing
        masks = {'source_missing_window': source.str.strip().eq('').to_numpy(),
                 'unverified_source_window': ~source.isin(VERIFIED_SOURCES).to_numpy(),
                 'legacy_fuyao_window': source.map(legacy_fuyao_source).to_numpy(bool),
                 'trade_price_unverified_window': ~verified.to_numpy(),
                 'missing_values_window': missing,
                 'negative_volume_amount_window': (quantity < 0).any(axis=1),
                 'volume_amount_mismatch_window': (quantity[:, 0] == 0) != (quantity[:, 1] == 0)}
        invalid = np.logical_or.reduce(list(masks.values()))
        active = pd.Series(~invalid & ~suspended).rolling(120, min_periods=1).sum().to_numpy(int)
        rolling = {key: pd.Series(mask).rolling(120, min_periods=1).max().to_numpy(bool)
                   for key, mask in masks.items()}
        days = bars.date.dt.strftime('%Y-%m-%d').tolist()
        if calendar is not None:
            gaps = [day not in known or (i > 0 and following.get(days[i-1]) != day) for i, day in enumerate(days)]
            rolling['calendar_missing_market_bar_window'] = pd.Series(gaps).rolling(120, min_periods=1).max().to_numpy(bool)
        else:
            rolling['verified_calendar_missing'] = np.ones(len(bars), dtype=bool)
        finite = np.isfinite(bars[list(MA_TREND_COLUMNS)].to_numpy(float)).all(axis=1)
        reasons = []
        for i in range(len(bars)):
            codes = [key for key, mask in rolling.items() if mask[i]]
            if i < 119: codes.append('quality_warmup')
            if active[i] < 60: codes.append('insufficient_active_history')
            if not finite[i]: codes.append('ma_history_unavailable')
            if suspended[i]: codes.append('suspended_current_bar')
            reasons.append(codes)
        bars['score_reason_codes'] = reasons
        bars['score_input_eligible'] = [not codes for codes in reasons]
        bars['model_input_eligible'] = bars.score_input_eligible
        bars['model_reason_codes'] = reasons
        bars['model_active_bars_window'] = active
        bars['as_of'] = days
        bars['feature_available_at'] = bars.as_of + 'T15:00:00+08:00'
        output.append(bars)
    result = pd.concat(output, ignore_index=True)
    result.attrs['score_feature_profile'] = profile_contract()
    return result
