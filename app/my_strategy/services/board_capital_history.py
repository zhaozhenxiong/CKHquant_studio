"""Bounded, display-only history for one immutable published board projection.

This module never publishes generations, writes raw data, or prepares native
structure. It extends a chart on demand using the publication's fixed current
members, verified calendar and unchanged raw source witness.
"""
from __future__ import annotations

import copy
from collections import OrderedDict
import threading

from my_strategy.adapters.czsc_adapter import raw_path
from my_strategy.core.run_context import stable_hash
from my_strategy.services.board_calculation_scope import CALCULATION_SCOPE, verify_calculation_membership
from my_strategy.services.board_features import _read_prices, board_policy
from my_strategy.services.board_fluent import _witness
from my_strategy.services.czsc_trend_support import _calendar

_lock = threading.RLock()
_basis_cache = OrderedDict()  # One 91-session pair of compact price matrices.
_history_cache = OrderedDict()  # At most 64 small, per-board 90-point responses.


def clear_cache():
    with _lock:
        _basis_cache.clear()
        _history_cache.clear()


def _calendar_prefix(generation, calendar_dates=None):
    dates = list(calendar_dates) if calendar_dates is not None else _calendar()[0]
    dates = [str(value)[:10] for value in dates if str(value)[:10] <= generation['price_date']]
    if not dates or dates != sorted(set(dates)) or dates[-1] != generation['price_date']:
        raise ValueError('资金走势的核验交易日历不包含发布截止日')
    expected = (generation.get('native_identity') or {}).get('calendar_hash')
    if expected is not None and stable_hash(dates) != expected:
        raise ValueError('资金走势的交易日历与发布版本不一致')
    return dates


def capital_history(*, generation, board, raw_db_path=None, window=90, calendar_dates=None):
    if type(window) is not int or window not in (90, 50, 20):
        raise ValueError('资金走势区间仅支持90、50或20个交易日')
    if generation.get('calculation_scope') != CALCULATION_SCOPE or board.get('calculation_scope') != CALCULATION_SCOPE:
        raise ValueError('该历史版本不支持当前沪深A股资金走势')
    if board.get('generation_id') != generation.get('generation_id') or board.get('price_date') != generation.get('price_date'):
        raise ValueError('资金走势板块与发布版本不一致')
    projection = verify_calculation_membership(board.get('calculation_membership'))
    if projection['source_snapshot_hash'] != board.get('snapshot_hash') or projection['board_id'] != board['board_id']:
        raise ValueError('资金走势的冻结计算成员核验失败')
    published = generation.get('publication_inputs', {}).get('projection_hashes', {}).get(board['board_id'])
    if published is not None and published != projection['projection_hash']:
        raise ValueError('资金走势的成员投影与发布版本不一致')
    dates = _calendar_prefix(generation, calendar_dates)
    path = raw_path(raw_db_path)
    witness = _witness(path)
    frozen = generation.get('publication_inputs', {}).get('source_witness', {}).get('raw')
    if frozen is not None and frozen != witness:
        raise ValueError('本地行情已变化，无法核验该发布版本的扩展资金走势')
    basis_key = stable_hash({'raw': witness, 'dates': dates[-91:]})
    key = stable_hash({'basis': basis_key, 'generation': generation['generation_id'],
        'projection': projection['projection_hash'], 'policy': generation.get('policy')})
    with _lock:
        cached = _history_cache.get(key)
        if cached is not None:
            _history_cache.move_to_end(key)
            result = copy.deepcopy(cached)
            result['cache_hit'] = True
        else:
            basis = _basis_cache.get(basis_key)
            if basis is None:
                _, _, matrices, price_sha = _read_prices(path, dates[-91:])
                if _witness(path) != witness:
                    raise ValueError('资金走势读取期间行情已变化，请刷新后重试')
                basis = {'amount': matrices.get('amount'), 'returns': matrices.get('return'), 'price_sha256': price_sha}
                _basis_cache.clear()
                _basis_cache[basis_key] = basis
            cfg = board_policy(generation.get('policy'))
            members = projection['eligible_members']; expected = len(members)
            amounts, returns = basis['amount'], basis['returns']
            points = []
            for date in dates[-90:]:
                observations = amounts.loc[date].reindex(members).dropna() if amounts is not None else []
                count = len(observations); coverage = count / expected if expected else 0.
                usable = projection['usable_membership'] and count >= cfg['min_members'] and coverage >= cfg['min_coverage']
                amount = float(observations.sum()) if usable else None
                market = amounts.loc[date].sum(min_count=1) if amounts is not None else None
                market = float(market) if market is not None and market == market else None
                changes = returns.loc[date].reindex(members).dropna() if returns is not None else []
                return_1 = float(changes.mean()) if projection['usable_membership'] and len(changes) >= cfg['min_members'] and expected and len(changes) / expected >= cfg['min_coverage'] else None
                points.append({'date': date, 'amount': amount, 'market_amount': market,
                    'market_share': amount / market if amount is not None and market is not None and market > 0 else None,
                    'return_1': return_1, 'coverage': coverage, 'effective_members': count, 'expected_members': expected})
            # The overlap is an additional guard against reading a different
            # population or treating unavailable source observations as zero.
            original = {point['date']: point for point in (board.get('capital_visualization') or {}).get('history', [])}
            for point in points:
                previous = original.get(point['date'])
                if previous is None:
                    continue
                for field in ('amount', 'market_amount', 'effective_members', 'expected_members'):
                    left, right = point[field], previous.get(field)
                    if left is None or right is None:
                        same = left is right
                    else:
                        same = abs(left - right) <= max(1e-7, abs(right) * 1e-12)
                    if not same:
                        raise ValueError('扩展资金走势与原发布历史不一致，保持原图')
            result = {'generation_id': generation['generation_id'], 'board_id': board['board_id'],
                'snapshot_hash': board['snapshot_hash'], 'projection_hash': projection['projection_hash'],
                'price_date': board['price_date'], 'membership_date': board['membership_date'],
                'calculation_scope': CALCULATION_SCOPE, 'mode': 'current_reference', 'training_eligible': False,
                'membership_basis': 'fixed_current_snapshot_historical_price_comparison',
                'capital_basis': 'turnover_amount_proxy', 'history': points,
                'raw_price_sha256': basis['price_sha256'], 'calendar_prefix_hash': stable_hash(dates),
                'source_witness': witness, 'cache_hit': False}
            _history_cache[key] = copy.deepcopy(result)
            while len(_history_cache) > 64:
                _history_cache.popitem(last=False)
        if _witness(path) != witness:
            raise ValueError('资金走势读取期间行情已变化，请刷新后重试')
    result['history'] = result['history'][-window:]
    result['requested_window'] = window
    result['actual_points'] = len(result['history'])
    return result
