"""Read-only historical context for an existing, immutable Kronos prediction.

This presentation adapter never invokes prediction, native preparation, training,
or account code. It verifies the original numerical input before joining bars.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
import math
import re
import sqlite3

from my_strategy.adapters.czsc_adapter import load_bars
from my_strategy.core.market_scope import require_supported_symbol
from my_strategy.services.kronos_prediction import SCHEMA, _hash, prepare_input


class PredictionChartNotFound(KeyError):
    """The reference does not identify an available prediction object."""


class PredictionChartIntegrityError(ValueError):
    """Saved prediction bytes or their internal identity cannot be verified."""


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('expected an ISO date')
    return value


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _saved_prediction(store, reference):
    if not isinstance(reference, str) or re.fullmatch(r'[a-f0-9]{64}', reference) is None:
        raise PredictionChartNotFound(reference)
    # Namespace is stored in the immutable request identity, not inferred from
    # user-supplied result fields. Do not touch TTL, access times, pins or leases.
    try:
        with store._lock:
            row = store.connection.execute(
                'SELECT namespace,identity_json FROM results WHERE cache_key=?', (reference,)).fetchone()
            if row is None or row['namespace'] != 'predictions':
                raise PredictionChartNotFound(reference)
            request_identity = json.loads(row['identity_json'])
            value = store.get_by_key(reference, touch=False)
    except PredictionChartNotFound:
        raise
    except (OSError, ValueError, TypeError, EOFError, sqlite3.Error) as exc:
        raise PredictionChartIntegrityError('预测对象完整性核验失败，未拼接历史行情。') from exc
    if value is None:
        raise PredictionChartNotFound(reference)
    if not isinstance(value, dict) or value.get('schema') != SCHEMA:
        raise PredictionChartIntegrityError('保存的对象不是受支持的预测结果。')
    try:
        identity, info = value['identity'], value['input']
        parameters, future = identity['parameters'], identity['future_dates']
        symbol, cutoff = value['symbol'], _day(value['as_of'])
        if require_supported_symbol(symbol) != symbol:
            raise ValueError('noncanonical symbol')
        if (value.get('status') != 'completed' or identity['schema'] != SCHEMA
                or value['cache_key'] != _hash(identity) or identity['input'] != info
                or info['symbol'] != symbol or info['as_of'] != cutoff
                or request_identity['kind'] != 'prediction-v1'
                or request_identity['symbol'] != symbol or request_identity['as_of'] != cutoff):
            raise ValueError('prediction identity mismatch')
        lookback, horizon = parameters['lookback'], parameters['horizon']
        if (not _integer(lookback, 16, 512) or not _integer(horizon, 1, 10)
                or info['lookback'] != lookback or info['rows'] != lookback
                or info['horizon'] != horizon or _day(info['history_end']) != cutoff
                or _day(info['history_start']) > cutoff or info['timezone'] != 'Asia/Shanghai'
                or not _finite(info['last_close']) or info['last_close'] <= 0
                or re.fullmatch(r'[a-f0-9]{64}', info['data_sha256']) is None):
            raise ValueError('invalid input window')
        if (not isinstance(future, list) or len(future) != horizon
                or [_day(day) for day in future] != sorted(set(future)) or future[0] <= cutoff):
            raise ValueError('invalid frozen forecast dates')
        if any(request_identity['parameters'][key] != parameters[key]
               for key in ('lookback', 'horizon', 'samples', 'seed', 'temperature', 'top_p')):
            raise ValueError('request parameters differ from saved prediction')
        if request_identity['calendar'] != identity['calendar']['identity']:
            raise ValueError('calendar identity mismatch')
        forecast = value['forecast']
        if (not isinstance(forecast, list) or forecast and [row['date'] for row in forecast] != future
                or not isinstance(value['quality'], dict)
                or value['summary'] is not None and not isinstance(value['summary'], dict)):
            raise ValueError('invalid forecast result')
        for row in forecast:
            if (any(not _finite(row[key]) or row[key] <= 0 for key in ('close', 'close_p10', 'close_p90'))
                    or not row['close_p10'] <= row['close'] <= row['close_p90']):
                raise ValueError('invalid forecast quantiles')
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
        raise PredictionChartIntegrityError('预测对象身份或输入合同不一致，未拼接历史行情。') from exc
    return value


def _unavailable(envelope, code, reason, *, matches=None):
    envelope['history_verification'].update(status='unavailable', reason_code=code,
                                          reason=reason, input_sha256_match=matches)
    return envelope


def prediction_chart(light, prediction_ref):
    """Read an outer prediction CAS key and return verified display data.

    Missing references and TTL-expired references without a pin or live lease
    raise NotFound, even before physical GC. Store-protected references remain
    readable. Corrupted saved evidence raises IntegrityError. Missing or revised
    historical inputs retain the original forecast with unavailable history.
    """
    value = _saved_prediction(light.store, prediction_ref)
    info, identity = value['input'], value['identity']
    envelope = {
        'schema': 'prediction_chart_v1', 'prediction_ref': prediction_ref,
        'prediction_identity_key': value['cache_key'], 'symbol': value['symbol'],
        'as_of': value['as_of'], 'timezone': info['timezone'], 'input': deepcopy(info),
        'parameters': deepcopy(identity['parameters']), 'calendar': deepcopy(identity['calendar']),
        'history': [], 'forecast': deepcopy(value['forecast']),
        'quality': deepcopy(value['quality']), 'summary': deepcopy(value['summary']),
        'history_verification': {'status': 'unavailable', 'reason_code': None, 'reason': None,
                                 'price_basis': info.get('price_basis'), 'input_sha256_match': None},
    }
    try:
        # Match the original read exactly, including prefix-derived price-basis
        # attrs. Merely loading the last N rows can relabel legacy input history.
        raw = load_bars(value['symbol'], end=value['as_of'], db_path=light.raw_db)
        current, _, observed = prepare_input(
            raw, identity['future_dates'], symbol=value['symbol'], as_of=value['as_of'],
            lookback=info['lookback'], horizon=info['horizon'])
    except (OSError, sqlite3.Error) as exc:
        return _unavailable(envelope, 'history_read_failed', '历史行情无法读取，原预测保留；未重新计算。')
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        return _unavailable(envelope, 'history_input_invalid', '历史行情缺失或不满足原输入合同，原预测保留；未重新计算。')
    if observed != info:
        return _unavailable(envelope, 'history_input_changed', '历史输入已修订或口径不一致，无法与原预测拼接。',
                            matches=observed.get('data_sha256') == info['data_sha256'])
    source = info.get('source')
    contract_verified = (info.get('price_basis') == 'unadjusted'
                         and info.get('volume_unit') == 'shares' and info.get('amount_unit') == 'CNY'
                         and info.get('unverified_trade_price_bars') == 0
                         and isinstance(source, list) and bool(source)
                         and set(source).issubset({'sina_unadjusted_v1', 'fuyao_unadjusted_v1'}))
    if not contract_verified:
        return _unavailable(envelope, 'history_price_contract_unverified',
                            '原预测使用的价格口径或量额单位未核验，未将其标为真实未复权历史。', matches=True)
    envelope['history'] = [
        {'date': day.date().isoformat(), **{key: float(row[key]) for key in
                                          ('open', 'high', 'low', 'close', 'volume', 'amount')}}
        for day, row in current.iterrows()
    ]
    envelope['history_verification'].update(status='verified', input_sha256_match=True)
    return envelope
