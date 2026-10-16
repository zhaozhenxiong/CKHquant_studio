"""Pure presentation of frozen, account-free method evidence.

No database, model, prediction or account operations occur here. A cutoff-day
reference is not a historical signal series, a first trigger, or an execution.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
import math
import re

SCHEMA = 'method_display_v1'
METHODS = {
    'structure': '结构与量价',
    'structure_kronos': '结构与量价 + Kronos',
    'structure_kronos_ma_mlp': '结构与量价 + Kronos + 均线模型 MLP',
    'structure_kronos_ma_l2': '结构与量价 + Kronos + 同输入 L2',
    'legacy_reference': '历史参考模型（对照）',
}
COMBINED = frozenset({'structure_kronos', 'structure_kronos_ma_mlp', 'structure_kronos_ma_l2'})
SCORED = COMBINED - {'structure_kronos'}
HEX = re.compile(r'[a-f0-9]{64}')


def _dict(value):
    return value if isinstance(value, dict) else {}


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _day(value):
    try:
        return value if isinstance(value, str) and date.fromisoformat(value).isoformat() == value else None
    except ValueError:
        return None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _reason_codes(value, fallback):
    if value is None or value == []:
        return [fallback]
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return ['method_reasons_invalid']
    return list(value)


def _forecast_reasons(forecast, context, day):
    reasons = []
    if forecast.get('status') != 'completed' or forecast.get('as_of') != day:
        reasons.append('forecast_identity_mismatch')
    quality = _dict(forecast.get('quality'))
    if any(quality.get(key) is not True for key in
           ('prediction_usable', 'calendar_verified', 'input_contract_verified', 'source_verified')):
        reasons.append('forecast_quality_unavailable')
    requested, valid = quality.get('requested_samples'), quality.get('valid_samples')
    counts = type(requested) is int and type(valid) is int and requested >= 8 and 8 <= valid <= requested
    if not counts or valid / requested < .8:
        reasons.append('forecast_sample_quality_invalid')
    if counts:
        fraction = quality.get('valid_fraction')
        if not _finite(fraction) or not math.isclose(fraction, valid / requested, abs_tol=1e-12):
            reasons.append('forecast_sample_quality_invalid')
        if 'invalid_samples' in quality and quality['invalid_samples'] != requested-valid:
            reasons.append('forecast_sample_quality_invalid')
    if context.get('samples') != requested:
        reasons.append('forecast_sample_identity_mismatch')
    # Older saved summaries omit future dates. Their absence is not a new
    # quality gate; a conflicting explicitly saved horizon still fails closed.
    for key in ('future_dates', 'forecast_dates'):
        if key in forecast:
            dates = forecast[key]
            if (not isinstance(dates, list) or len(dates) != context.get('horizon')
                    or not all(_day(value) and value > day for value in dates)
                    or dates != sorted(set(dates))):
                reasons.append('forecast_horizon_identity_mismatch')
    summary = _dict(forecast.get('summary'))
    keys = ('terminal_return_p10', 'terminal_return_median', 'terminal_return_p90',
            'terminal_return_std', 'up_path_fraction', 'worst_path_low_return')
    if not all(_finite(summary.get(key)) for key in keys):
        reasons.append('forecast_summary_invalid')
    elif (not summary['terminal_return_p10'] <= summary['terminal_return_median'] <= summary['terminal_return_p90']
          or summary['terminal_return_std'] < 0 or not 0 <= summary['up_path_fraction'] <= 1):
        reasons.append('forecast_summary_invalid')
    return reasons


def _probability_direction(probabilities):
    if not isinstance(probabilities, dict) or set(probabilities) != {'down', 'flat', 'up'}:
        return None
    values = [probabilities[key] for key in ('down', 'flat', 'up')]
    if any(not _finite(value) or not 0 <= value <= 1 for value in values) or not math.isclose(sum(values), 1., abs_tol=1e-6):
        return None
    largest = max(values)
    return ('SELL', 'WAIT', 'BUY')[values.index(largest)] if values.count(largest) == 1 else 'WAIT'


def _score_reasons(evaluation, context, expected_version, method):
    score = _dict(evaluation.get('score'))
    model_id, version = evaluation.get('model_id'), score.get('model_version')
    reasons = []
    if not isinstance(model_id, str) or HEX.fullmatch(model_id) is None or model_id != context.get('score_model_id'):
        reasons.append('score_model_identity_mismatch')
    if (not isinstance(version, str) or HEX.fullmatch(version) is None
            or not isinstance(expected_version, str) or HEX.fullmatch(expected_version) is None
            or version != expected_version):
        reasons.append('score_model_version_mismatch')
    kind = 'l2' if method == 'structure_kronos_ma_l2' else 'mlp'
    expected_type = 'l2' if kind == 'l2' else 'torch_mlp'
    if (evaluation.get('feature_profile') != 'ma_trend_v1' or score.get('feature_profile') != 'ma_trend_v1'
            or score.get('variant') != 'A' or score.get('model_type') != expected_type
            or score.get('consumes_kronos') is not False):
        reasons.append('score_method_identity_mismatch')
    if (not isinstance(version, str) or HEX.fullmatch(version) is None
            or score.get('scorer_id') != _digest({'model': version, 'head': kind})):
        reasons.append('score_head_identity_mismatch')
    direction = _probability_direction(score.get('probabilities'))
    if score.get('status') != 'scored' or not _finite(score.get('score')) or not 0 <= score['score'] <= 100 or direction is None:
        reasons.append('score_unavailable')
    elif not math.isclose(score['score'], 100*(score['probabilities']['up']+.5*score['probabilities']['flat']), abs_tol=1e-6):
        reasons.append('score_probability_value_mismatch')
    qualification = _dict(score.get('qualification'))
    expected_mode = 'historical_candidate' if context.get('usage_mode', 'historical') == 'historical' else 'paper'
    if qualification.get('mode') != expected_mode:
        reasons.append('score_usage_identity_mismatch')
    return reasons, direction


def _reference_price(row, day):
    """Use one genuine saved cutoff close, never forecast prices or earlier bars."""
    if 'frequencies' in row:
        bars = _dict(_dict(row.get('frequencies')).get('日线')).get('bars')
        if not isinstance(bars, list):
            return None
        selected = [bar for bar in bars if isinstance(bar, dict) and _day(bar.get('time')) == day]
        if len(selected) != 1:
            return None
        bar = selected[0]
        values = [bar.get(key) for key in ('open', 'high', 'low', 'close')]
        if (bar.get('is_closed') is not True or any(not _finite(value) or value <= 0 for value in values)
                or bar['high'] < max(bar['open'], bar['close'], bar['low'])
                or bar['low'] > min(bar['open'], bar['close'], bar['high'])):
            return None
        return float(bar['close'])
    value, close = row.get('reference_price'), row.get('close')
    if (row.get('reference_price_date') != day or row.get('reference_price_source') != 'verified_local_close'
            or row.get('data_end') != day or not _finite(value) or value <= 0
            or not _finite(close) or close != value):
        return None
    return float(value)


def classify_method_row(row, *, context=None, as_of=None, model_version=None):
    """Return the shared classification for one saved row, without changing it.

    Scan callers supply the frozen header context, cutoff and model version.
    Forecast summary validity is separate from MA availability. A missing CAS
    reference limits chart access but does not erase saved numerical evidence.
    """
    row = _dict(row)
    evaluation = _dict(row.get('method_evaluation'))
    own_context = _dict(row.get('method_context'))
    expected_context = _dict(context) if context is not None else own_context
    method = expected_context.get('analysis_method') or row.get('analysis_method') or evaluation.get('method')
    if method is None:
        method = 'legacy_reference' if row.get('research') else 'structure'
    elif not isinstance(method, str):
        method = 'unknown'
    day = as_of or evaluation.get('as_of') or row.get('as_of') or row.get('data_end')
    day = _day(day)
    source_status = evaluation.get('status')
    score = _dict(evaluation.get('score'))
    forecast = _dict(evaluation.get('kronos'))
    reference = _dict(forecast.get('result_ref')).get('cache_key')
    reference_status = 'recorded' if isinstance(reference, str) and HEX.fullmatch(reference) else 'missing' if reference is None else 'invalid'
    result = dict(schema=SCHEMA, method=method, label=METHODS.get(method, '未知方法'), as_of=day,
        status='unavailable', source_status=source_status, valid=False, direction=None, reason_codes=[],
        prediction_ref=reference if reference_status == 'recorded' else None,
        prediction_reference_status=reference_status,
        model=dict(model_id=evaluation.get('model_id'), model_version=score.get('model_version'),
                   scorer_id=score.get('scorer_id'), kind=score.get('model_type'),
                   feature_profile=evaluation.get('feature_profile'), qualification=deepcopy(score.get('qualification'))),
        forecast_valid=False, forecast_summary=None, forecast_quality=deepcopy(forecast.get('quality')),
        score_valid=False, score_value=None, method_reference_points=[],
        timing=dict(market_cutoff_at=day+'T15:00:00+08:00' if day else None,
                    actual_available_at=None, usage_mode=expected_context.get('usage_mode', 'historical'),
                    semantics='as_of_market_close_reference_not_actual_prediction_creation',
                    historical_series_generated=False, first_trigger=False, execution=False))
    if method not in COMBINED:
        result.update(status='not_applicable', reason_codes=['separate_source_or_legacy_semantics' if method in METHODS else 'unknown_method'],
                      prediction_ref=None, prediction_reference_status='not_applicable')
        return result
    identity_reasons = []
    if not day:
        identity_reasons.append('method_cutoff_invalid')
    if not re.fullmatch(r'\d{6}\.(SH|SZ)', str(row.get('symbol', ''))):
        identity_reasons.append('method_symbol_invalid')
    if (evaluation.get('schema') not in (None, 'account_free_methods_v1')
            or source_status not in ('failed', 'deferred') and evaluation.get('schema') is None):
        identity_reasons.append('method_schema_unsupported')
    if evaluation.get('method') != method or row.get('analysis_method') != method:
        identity_reasons.append('method_identity_mismatch')
    if not expected_context or expected_context.get('analysis_method') != method:
        identity_reasons.append('method_context_missing')
    if source_status not in ('failed', 'deferred') and not own_context:
        identity_reasons.append('method_context_missing')
    if (expected_context.get('usage_mode', 'historical') not in ('historical', 'forward')
            or any(type(expected_context.get(key)) is not int or expected_context[key] not in allowed
                   for key, allowed in (('lookback', (128, 256)), ('horizon', (1, 5, 10)), ('samples', (8, 32, 64))))
            or type(expected_context.get('seed')) is not int or not 0 <= expected_context['seed'] <= 2147483647):
        identity_reasons.append('method_parameters_invalid')
    if own_context and own_context != expected_context:
        identity_reasons.append('method_context_mismatch')
    if (evaluation.get('as_of') != day or expected_context.get('end') != day
            or any(row.get(key) is not None and row[key] != day for key in ('date', 'data_end', 'as_of'))):
        identity_reasons.append('method_cutoff_mismatch')
    if identity_reasons:
        result.update(reason_codes=sorted(set(identity_reasons)), prediction_ref=None,
                      prediction_reference_status='identity_unverified')
        return result
    if source_status in ('failed', 'deferred'):
        result.update(status=source_status, reason_codes=_reason_codes(evaluation.get('missing_reasons'), source_status),
                      prediction_ref=None, prediction_reference_status='not_available')
        return result
    forecast_reasons = _forecast_reasons(forecast, expected_context, day)
    result['forecast_valid'] = not forecast_reasons
    if result['forecast_valid']:
        result['forecast_summary'] = deepcopy(forecast['summary'])
    score_reasons, score_direction = [], None
    if method in SCORED:
        score_reasons, score_direction = _score_reasons(evaluation, expected_context, model_version, method)
        result['score_valid'] = not score_reasons
        if result['score_valid']:
            result['score_value'] = score['score']
    elif evaluation.get('score') is not None or evaluation.get('model_id') is not None or expected_context.get('score_model_id') is not None:
        score_reasons.append('unexpected_score_model')
    reasons = [*forecast_reasons, *score_reasons]
    if source_status != 'complete':
        reasons.extend(_reason_codes(evaluation.get('missing_reasons'), 'method_evidence_incomplete'))
    elif evaluation.get('missing_reasons'):
        reasons.append('complete_method_has_missing_evidence')
    price = _reference_price(row, day)
    if price is None:
        reasons.append('cutoff_reference_price_unavailable')
    direction = evaluation.get('direction')
    if direction not in ('BUY', 'SELL', 'WAIT'):
        reasons.append('method_direction_invalid')
    if result['forecast_valid'] and not score_reasons:
        median = forecast['summary']['terminal_return_median']
        expected_direction = 'BUY' if median > 0 else 'SELL' if median < 0 else 'WAIT'
        if method in SCORED and score_direction != expected_direction:
            expected_direction = 'WAIT'
        if direction != expected_direction:
            reasons.append('method_direction_evidence_mismatch')
    if reasons:
        result['reason_codes'] = sorted(set(str(value) for value in reasons))
        return result
    result.update(status='complete', valid=True, direction=direction)
    if direction in {'BUY', 'SELL'}:
        result['method_reference_points'] = [dict(
            symbol=row['symbol'], date=day, time=day, as_of=day, action=direction,
            reference_price=price, reference_price_date=day, reference_price_source='saved_cutoff_close',
            reference_layer='method', reference_only=True, account_scope='none',
            method=method, label=result['label'], model_id=result['model']['model_id'],
            model_version=result['model']['model_version'], market_cutoff_at=result['timing']['market_cutoff_at'],
            actual_available_at=None, first_trigger=False, execution=False,
            signal_semantics='所选方法的截止日方向参考，不是首次触发或模拟成交')]
    return result


def build_method_view(result):
    result = _dict(result)
    identity = _dict(result.get('method_identity')) or _dict(result.get('frozen_identity'))
    return classify_method_row(result, context=_dict(result.get('method_context')),
        as_of=result.get('as_of') or result.get('data_end'), model_version=identity.get('model_version'))


def with_method_view(result):
    """Add response-only fields; saved reports/cached structures stay untouched."""
    if not isinstance(result, dict):
        return result
    view = build_method_view(result)
    return {**result, 'method_view': view, 'method_reference_points': deepcopy(view['method_reference_points'])}
