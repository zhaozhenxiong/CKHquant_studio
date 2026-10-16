"""Read-only chart projections of existing Broker/Ledger simulation results.

This view never runs an account, predicts a missing day, or turns historical
fills into cutoff-day method signals. Its candles and fills come from the same
immutable stored result, independently of the current analysis forecast.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import math
import re

STRATEGIES = {
    'structure_rules': '结构与量价',
    'structure_kronos': '结构与量价 + Kronos',
    'structure_kronos_score': '结构与量价 + Kronos + 评分模型',
    'structure_kronos_l2': '结构与量价 + Kronos + 同输入 L2',
}
HEX = re.compile(r'[a-f0-9]{64}')


def _labels(identity):
    labels = dict(STRATEGIES)
    if identity.get('feature_profile') == 'ma_trend_v1':
        labels['structure_kronos_score'] = '结构与量价 + Kronos + 均线 MLP'
    elif identity.get('score_model_type') in {'l2', 'mlp', 'torch_mlp'}:
        kind = 'L2' if identity['score_model_type'] == 'l2' else 'MLP'
        labels['structure_kronos_score'] = '结构与量价 + Kronos + 历史融合 ' + kind
    return labels


def _day(value):
    if not isinstance(value, str):
        raise ValueError('回测日期未记录')
    return date.fromisoformat(value[:10]).isoformat()


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not positive or value > 0))


def chart_options(light, symbol, as_of):
    """Bounded metadata lookup, without opening reports or changing pin state."""
    cutoff = _day(as_of)
    rows = light.store.list(namespace='simulations', limit=50)
    options = []
    for row in rows:
        identity = row.get('identity') or {}
        sources = identity.get('sources') or []
        if symbol not in [source.get('symbol') for source in sources if isinstance(source, dict)]:
            continue
        try:
            start, end = _day(identity.get('start')), _day(identity.get('end'))
        except (ValueError, TypeError):
            continue
        if start > cutoff or not HEX.fullmatch(str(row.get('cache_key', ''))):
            continue
        strategies = [key for key in STRATEGIES if key in (identity.get('configs') or {})]
        if not strategies:
            continue
        options.append(dict(result_id=row['cache_key'], snapshot_sha256=row.get('sha256'),
            start=start, end=end, saved=bool(row.get('pin_count')),
            created_at=row.get('created_at'), strategies=strategies,
            prediction_parameters=deepcopy(identity.get('prediction') or {}),
            feature_profile=identity.get('feature_profile'), strategy_labels=_labels(identity)))
    options.sort(key=lambda row: (row.get('created_at') or 0, row['result_id']), reverse=True)
    return dict(symbol=symbol, as_of=cutoff, results=options, lookup_limit=50,
                prediction_started=False, account_mutated=False)


def account_chart(light, result_id, symbol, strategy, as_of):
    if not HEX.fullmatch(result_id):
        raise ValueError('回测结果标识不正确')
    if strategy not in STRATEGIES:
        raise ValueError('不支持的回测策略')
    cutoff = _day(as_of)
    result = light.result(result_id)  # ResearchStore verifies the actual CAS SHA.
    if result.get('result_id') != result_id or symbol not in result.get('symbols', []):
        raise ValueError('回测结果与当前股票不一致')
    account = (result.get('comparisons') or {}).get(strategy)
    if not isinstance(account, dict):
        raise ValueError('这份回测未保存所选策略')
    lineage = account.get('lineage') or {}
    if (lineage.get('account_scope') != 'shared_cash_account'
            or lineage.get('execution') != 'next_verified_open_with_declared_slippage'):
        raise ValueError('这份结果未记录可核验的 Broker 成交来源')
    start, end = _day(result.get('start')), _day(result.get('end'))
    scheduler = result.get('scheduler') or {}
    completed = scheduler.get('complete_through')
    if completed is None and result.get('status') != 'partial':
        completed = end
    if not completed:
        raise ValueError('这份回测尚无完整评估日')
    through = min(cutoff, end, _day(completed))
    if through < start:
        raise ValueError('当前截止日早于回测覆盖区间')
    sources = (result.get('identity') or {}).get('sources') or []
    source = next((value for value in sources if isinstance(value, dict) and value.get('symbol') == symbol), None)
    if not source or source.get('price_basis') != 'unadjusted':
        raise ValueError('这份回测未记录该股票的实际未复权价格口径')
    bars, previous = [], None
    for row in (result.get('chart_bars') or {}).get(symbol, []):
        day = _day(row.get('date'))
        if previous and day <= previous:
            raise ValueError('回测行情日期重复或顺序异常')
        previous = day
        if not start <= day <= through:
            continue
        if (any(not _number(row.get(key), positive=True) for key in ('open', 'high', 'low', 'close'))
                or row['high'] < max(row['open'], row['low'], row['close'])
                or row['low'] > min(row['open'], row['high'], row['close'])):
            raise ValueError('回测行情价格不合法')
        bars.append(dict(time=day, **{key: row[key] for key in ('open', 'high', 'low', 'close')}))
    if not bars:
        raise ValueError('这份回测在所选截止日没有可展示行情')
    bar_dates = {row['time'] for row in bars}
    trades = []
    for index, row in enumerate(account.get('trades') or []):
        if row.get('symbol') != symbol:
            continue
        day = _day(row.get('date'))
        if not start <= day <= through:
            continue
        if (day not in bar_dates or row.get('action') not in {'BUY', 'ADD', 'SELL', 'REDUCE'}
                or not _number(row.get('price'), positive=True)
                or not _number(row.get('shares'), positive=True)
                or not _number(row.get('fee')) or row['fee'] < 0
                or type(row.get('ledger_version')) is not int
                or row['ledger_version'] < 1 or not row.get('order_id')):
            raise ValueError('已存成交缺少完整 Ledger 价格、数量或日期证据')
        signal = row.get('signal_date')
        if signal is not None and _day(signal) > day:
            raise ValueError('成交对应信号日期晚于成交日')
        # Preserve actual execution price/fee and order, including same-day
        # hard-risk fills. Never substitute the candle close for a fill price.
        fields = ('date', 'symbol', 'action', 'price', 'shares', 'fee', 'reason',
                  'order_id', 'signal_date', 'share_delta', 'position_before',
                  'position_after', 'execution_change', 'ledger_version')
        trades.append({**deepcopy({key: row[key] for key in fields if key in row}),
                       'date': day, 'ledger_index': index})
    trades.sort(key=lambda row: (row['date'], row['ledger_index']))
    identity = result.get('identity') or {}
    return dict(schema='saved_account_chart_v1', result_id=result_id, symbol=symbol,
        strategy=strategy, label=_labels(identity)[strategy], as_of=cutoff,
        display_start=bars[0]['time'], display_end=bars[-1]['time'],
        original_start=start, original_end=end, complete_through=_day(completed),
        source_status=result.get('status'), saved=result.get('saved') is True,
        qualification_mode=result.get('qualification_mode'), price_basis='unadjusted',
        prediction_parameters=deepcopy(identity.get('prediction') or {}),
        feature_profile=result.get('feature_profile'),
        bars=bars, trades=trades,
        counts={'BUY': sum(t['action'] in {'BUY', 'ADD'} for t in trades),
                'SELL': sum(t['action'] in {'SELL', 'REDUCE'} for t in trades),
                'ADD': sum(t['action'] == 'ADD' for t in trades),
                'REDUCE': sum(t['action'] == 'REDUCE' for t in trades)},
        semantics='broker_ledger_fills', prediction_started=False,
        account_mutated=False, current_method_qualification_changed=False)
