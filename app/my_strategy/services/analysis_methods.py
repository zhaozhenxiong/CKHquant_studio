"""Account-free method evaluation shared by the chart and bounded market scans.

Method/model selection is frozen once per request. No training, promotion or
Broker orders occur here; cancelled scans are immutable, explicitly resumable
inputs to a *new* scan. Missing observations remain in the population.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import logging
import os
import uuid
import math
from pathlib import Path
import threading
import time

import pandas as pd

from my_strategy.core.run_context import stable_hash, create_run_context
from my_strategy.core.tz import local_now

VERSION = 'account_free_methods_v1'
PROFILE = 'ma_trend_v1'
METHODS = {
    'structure': '结构与量价',
    'structure_kronos': '结构与量价 + Kronos',
    'structure_kronos_ma_mlp': '结构与量价 + Kronos + 均线模型 MLP',
    'structure_kronos_ma_l2': '结构与量价 + Kronos + 同输入 L2',
    'legacy_reference': '历史参考模型（对照）',
}
COMBINED = frozenset(set(METHODS) - {'structure', 'legacy_reference'})
SCORED = frozenset({'structure_kronos_ma_mlp', 'structure_kronos_ma_l2'})
REQUEST_FIELDS = ('analysis_method', 'score_model_id', 'start', 'end', 'usage_mode',
                  'lookback', 'horizon', 'samples', 'seed', 'device')
_HASH_CACHE = {}
_HASH_LOCK = threading.Lock()


def request_context(spec):
    return {key: spec[key] for key in REQUEST_FIELDS if spec.get(key) is not None}


def _code_identity():
    from my_strategy.services import light_research
    from my_strategy.services import kronos_assets
    from my_strategy.services.incremental_source import code_identity as source_identity
    # Account/Broker evolution and UI presentation do not regenerate numerical
    # method inputs. Keep every actual scorer, prediction and source dependency.
    names=('light_ma_features.py','light_score_training.py','light_score_v2.py','simple_score.py',
           'light_qualification.py','light_experiment_registry.py','czsc_market_features.py','czsc_market_profiles.py',
           'kronos_prediction.py','agent_evidence.py')
    root=Path(__file__).parent
    research={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    methods=('source','prediction','scorer','resolve_score_model','score_model','current_score','_check')
    research.update({name:hashlib.sha256(inspect.getsource(getattr(light_research.LightResearch,name)).encode()).hexdigest()
                     for name in methods})
    research.update({name:hashlib.sha256(inspect.getsource(getattr(light_research,name)).encode()).hexdigest()
                     for name in ('prediction_identity','serialized_compute','plain')})
    research['score_columns']=list(light_research.SCORE_COLUMNS)
    research['prediction_asset_verification']={name:hashlib.sha256(inspect.getsource(getattr(kronos_assets,name)).encode()).hexdigest()
        for name in ('asset_root','_stat_identity','_memo_version','_verify','asset_status')}
    research['prediction_asset_contract']={'pins':list(kronos_assets.PINNED_ASSETS),
        'memo_min_bytes':kronos_assets._MEMO_MIN_BYTES,'memo_limit':kronos_assets._MEMO_LIMIT,'windows':kronos_assets._WINDOWS}
    return {'evaluator': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'research': research, 'source': source_identity(), 'schema':'method_generation_identity_v2'}


def database_identity(light, check=None):
    """Hash physical raw input and WAL; never create, checkpoint or vacuum it."""
    from my_strategy.adapters.czsc_adapter import raw_path
    path = raw_path(light.raw_db)
    if check: check()
    result = {}
    for candidate in (path, Path(str(path) + '-wal')):
        if not candidate.exists():
            continue
        before = candidate.stat()
        key = (str(candidate), before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        with _HASH_LOCK:
            digest = _HASH_CACHE.get(key)
        if digest is None:
            digest = hashlib.sha256()
            checked_bytes = 0
            next_check = time.monotonic() + .05 if check else None
            with candidate.open('rb') as stream:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block: break
                    digest.update(block)
                    checked_bytes += len(block)
                    # External cancellation can query the task database. Bound
                    # polling independently of the 1 MiB full-SHA read blocks.
                    if check and (checked_bytes >= 16 * 1024 * 1024 or time.monotonic() >= next_check):
                        check()
                        checked_bytes = 0
                        next_check = time.monotonic() + .05
            digest = digest.hexdigest()
            if check: check()
            after = candidate.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError('行情正在改变，请在更新完成后扫描')
            with _HASH_LOCK:
                if len(_HASH_CACHE) > 12: _HASH_CACHE.clear()
                _HASH_CACHE[key] = digest
        if check: check()
        result['wal' if candidate != path else 'database'] = digest
    return result


class Stop:
    def __init__(self, deadline, check_cancel=None):
        self.deadline, self.check_cancel = deadline, check_cancel
        self.cancelled = threading.Event()

    def is_set(self):
        # Kronos and source adapters poll is_set while the coordinator is busy;
        # cancellation must not depend on returning from the current prediction.
        if not self.cancelled.is_set() and self.check_cancel:
            try: self.check_cancel()
            except Exception: self.cancelled.set()
        return self.cancelled.is_set() or time.monotonic() >= self.deadline

    def check(self):
        if self.check_cancel:
            try: self.check_cancel()
            except Exception as exc:
                self.cancelled.set()
                raise InterruptedError('组合方法扫描已取消') from exc
        if self.cancelled.is_set(): raise InterruptedError('组合方法扫描已取消')
        if self.is_set():
            if self.cancelled.is_set(): raise InterruptedError('组合方法扫描已取消')
            raise TimeoutError('本批时间预算已用尽')


class _ScanTiming:
    """Coordinator wall times; overlapping source worker seconds stay separate.

    A report contains only measurements available before its save. A separate
    atomic receipt records final save, safe drain and total elapsed time without
    rewriting the report or returning a different in-memory result.
    """
    def __init__(self, store, spec, check_cancel, started=None):
        self.started = time.monotonic() if started is None else started
        self.budget = float(spec.get('budget_seconds', 300))
        if not math.isfinite(self.budget) or self.budget <= 0:
            raise ValueError('扫描时间预算必须为正有限秒数')
        self.stop = Stop(self.started + self.budget, check_cancel)
        self.store = store
        self.run_id = None
        self.status = None
        self.initialized = False
        self.initial_cancelled = False
        self.attempt_id = uuid.uuid4().hex
        self.relative_path = '.method-execution/' + self.attempt_id + '.json'
        self.seconds = {name: 0. for name in (
            'initialization', 'waiting_preparation', 'evaluation',
            'checkpoint_verification', 'checkpoint_save', 'drain', 'final_verification')}

    def __enter__(self):
        return self

    def check_initialization(self):
        # TaskManager's original Cancelled type must survive before a resumable
        # identity exists. Later cancellation is represented by a saved batch.
        if self.stop.check_cancel:
            try:
                self.stop.check_cancel()
            except Exception:
                self.initial_cancelled = True
                self.stop.cancelled.set()
                raise
        if self.stop.cancelled.is_set():
            raise InterruptedError('组合方法扫描已取消')
        if time.monotonic() >= self.stop.deadline:
            raise TimeoutError('本批时间预算已用尽')

    def initialization_complete(self):
        self.seconds['initialization'] = time.monotonic() - self.started
        self.initialized = True

    @contextmanager
    def measure(self, phase):
        started = time.monotonic()
        try:
            yield
        finally:
            self.seconds[phase] = self.seconds.get(phase, 0.) + time.monotonic() - started

    def snapshot(self):
        elapsed = time.monotonic() - self.started
        return {'schema': 'method_execution_timing_v1',
            'budget_scope': 'scan_entry_through_computation_with_mandatory_safe_finalization',
            'budget_seconds': self.budget, 'elapsed_before_save_seconds': elapsed,
            'budget_overrun_before_save_seconds': max(0., elapsed - self.budget),
            'phase_wall_seconds': dict(self.seconds),
            'phase_semantics': 'coordinator wall; source preparation overlaps evaluation; evaluation includes cache lookup and scoring',
            'final_save_and_total': self.relative_path}

    def __exit__(self, exc_type, exc, tb):
        if not self.initialized:
            self.seconds['initialization'] = time.monotonic() - self.started
        elapsed = time.monotonic() - self.started
        receipt = {'schema': 'method_execution_timing_v1', 'attempt_id': self.attempt_id,
            'run_id': self.run_id, 'status': self.status if exc is None else
                ('budget_exhausted' if isinstance(exc, TimeoutError) else
                 'cancelled' if isinstance(exc, InterruptedError) or self.initial_cancelled else 'failed'),
            'error': f'{type(exc).__name__}: {exc}' if exc is not None else None,
            'pid': os.getpid(), 'recorded_at': local_now().isoformat(),
            'budget_scope': 'scan_entry_through_computation_with_mandatory_safe_finalization',
            'budget_seconds': self.budget, 'phase_wall_seconds': dict(self.seconds),
            'total_before_receipt_seconds': elapsed, 'budget_overrun_seconds': max(0., elapsed - self.budget),
            'receipt_write_included': False}
        root = getattr(self.store, 'runs_root', None)
        try:
            if root is None:
                logging.getLogger(__name__).info('method_execution_timing %s', json.dumps(receipt, ensure_ascii=False))
            else:
                target = Path(root) / self.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps(receipt, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                temporary.replace(target)
        except OSError:
            # Observation failure never changes a committed business result.
            logging.getLogger(__name__).exception('Unable to persist method execution timing: %s', receipt)
        return False


@contextmanager
def compute_lock(light):
    from filelock import Timeout
    try:
        with light.compute_lock.acquire(timeout=0): yield
    except Timeout as exc:
        raise RuntimeError('已有轻量任务正在使用计算资源') from exc


def target_date(light, spec):
    from my_strategy.adapters.czsc_adapter import latest_market_date, date_cutoff
    dates, calendar = light.calendar()
    latest = latest_market_date(spec.get('end'), db_path=light.raw_db)
    cutoffs = [date_cutoff(str(value)) for value in (latest, spec.get('end'), spec.get('as_of')) if value]
    if not cutoffs: raise ValueError('没有可用行情截止日期')
    cutoff = min(cutoffs)
    eligible = [day for day in dates if pd.Timestamp(day) + pd.Timedelta(hours=15) <= cutoff]
    if not eligible: raise ValueError('截止时点之前没有核验交易日')
    day = eligible[-1]
    if spec.get('start') and str(spec['start']) > day: raise ValueError('开始日期晚于可用行情截止日')
    return day, calendar


def freeze_method(light, spec, day):
    method = spec.get('analysis_method')
    if method not in COMBINED: raise ValueError('不支持的组合分析方法')
    model = None
    if method in SCORED:
        model = light.resolve_score_model(spec.get('score_model_id'), feature_profile=PROFILE,
            as_of=day, research_mode='historical_candidate' if spec.get('usage_mode', 'historical') == 'historical' else 'paper')
    from my_strategy.services.kronos_prediction import asset_status
    from my_strategy.services.light_research import prediction_identity
    assets = asset_status()
    params = {key: spec.get(key, default) for key, default in
              [('lookback', 128), ('horizon', 5), ('samples', 8), ('seed', 42), ('device', 'auto')]}
    identity = prediction_identity(params['device']) if assets.get('ready') else {'assets': assets}
    model_id = (model or {}).get('model_id') or ((model or {}).get('model_ref') or {}).get('cache_key')
    context = {**request_context(spec), 'analysis_method': method, 'end': day,
               'score_model_id': model_id, **params}
    return {'context': context, 'model': model, 'assets_ready': assets.get('ready') is True,
            'prediction_identity': identity, 'code': _code_identity(),
            'model_version': ((model or {}).get('artifact') or {}).get('model_version')}


def _forecast_summary(forecast):
    return {key: forecast[key] for key in ('status', 'reason', 'as_of', 'actual_available_at',
        'summary', 'quality', 'result_ref', 'resource', 'reused', 'future_dates', 'forecast_dates') if key in forecast}


def _score_direction(score):
    """Actual scorers return named class probabilities, not positional lists."""
    probabilities = score.get('probabilities')
    if isinstance(probabilities, dict):
        probabilities = [probabilities.get(key) for key in ('down', 'flat', 'up')]
    if not isinstance(probabilities, (list, tuple)) or len(probabilities) != 3:
        return 'WAIT'
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
           for v in probabilities) or not math.isclose(sum(probabilities), 1., abs_tol=1e-6):
        return 'WAIT'
    largest = max(probabilities)
    if sum(value == largest for value in probabilities) != 1: return 'WAIT'
    return ('SELL', 'WAIT', 'BUY')[probabilities.index(largest)]


def prepare_stock(light, symbol, day, stop):
    stop.check()
    from my_strategy.adapters.czsc_adapter import load_bars
    # Avoid an expensive native replay when the requested stock has no close.
    observed = load_bars(symbol, end=day, db_path=light.raw_db)
    if observed.empty or str(observed.date.iloc[-1])[:10] != day:
        raise ValueError(f'{symbol} 缺少 {day} 行情，不沿用前日结果')
    return light.source(symbol, day, stop=stop, raw_snapshot=observed)


def evaluate_prepared(light, symbol, day, frozen, prepared, stop):
    raw, frame, events, source = prepared
    context = frozen['context']; method = context['analysis_method']
    reasons = []
    stop.check()
    forecast = {'status': 'unavailable', 'reason': 'kronos_assets_missing', 'as_of': day,
                'quality': {'prediction_usable': False}}
    if frozen['assets_ready']:
        try:
            forecast = light.prediction(symbol, day, stop=stop, raw_snapshot=raw,
                **{key: context[key] for key in ('lookback', 'horizon', 'samples', 'seed', 'device')})
        except (InterruptedError, TimeoutError):
            raise
        except (ValueError, OSError, RuntimeError) as exc:
            forecast = {**forecast, 'reason': str(exc)}
    usable = (forecast.get('quality') or {}).get('prediction_usable') is True
    if not usable: reasons.append(forecast.get('reason') or 'kronos_prediction_unusable')
    score = None
    if method in SCORED:
        if frozen['model'] is None:
            score = {'status': 'unavailable', 'reason': 'ma_score_model_missing', 'feature_profile': PROFILE}
        else:
            try:
                score = light.scorer(frozen['model'], {symbol: frame}, {symbol: events},
                    research_mode='historical_candidate' if context.get('usage_mode', 'historical') == 'historical' else 'paper')(symbol, day, forecast)
                if method == 'structure_kronos_ma_l2': score = score.get('l2_comparison') or {'status': 'unavailable', 'reason': 'same_input_l2_missing'}
                score['feature_profile'] = PROFILE
            except (ValueError, KeyError) as exc:
                score = {'status': 'unavailable', 'reason': str(exc), 'feature_profile': PROFILE}
        if score.get('status') != 'scored' or score.get('score') is None:
            reasons.append(score.get('reason') or 'ma_score_unavailable')
    summary = forecast.get('summary') or {}
    median = summary.get('terminal_return_median')
    direction = 'WAIT'
    if usable and median is not None and not reasons:
        direction = 'BUY' if median > 0 else 'SELL' if median < 0 else 'WAIT'
        if score:
            if _score_direction(score) != direction: direction = 'WAIT'
    visible = [event for event in events if event.get('confirmed') is True and
               str(event.get('available_at') or event.get('date'))[:10] <= day]
    latest = [event for event in visible if str(event.get('available_at') or event.get('date'))[:10] == day]
    row = frame.iloc[-1]
    evaluation = {'schema': VERSION, 'method': method, 'label': METHODS[method], 'as_of': day,
        'status': 'partial' if reasons else 'complete', 'missing_reasons': reasons,
        'kronos': _forecast_summary(forecast), 'score': score, 'direction': direction,
        'model_id': context.get('score_model_id'), 'feature_profile': PROFILE if method in SCORED else None,
        'source_identity': source, 'source_event_count': len(latest),
        'reference_only': True, 'account_scope': 'none', 'prediction_fraction_is_calibrated_probability': False,
        'signal_semantics': '模型方向参考；来源事件与实际模拟成交分别核验'}
    return {'symbol': symbol, 'date': day, 'data_end': day, 'available_at': day + 'T15:00:00+08:00',
        'close': float(row.close), 'reference_price': float(row.close), 'reference_price_date': day,
        'reference_price_source': 'verified_local_close', 'analysis_mode': 'reference',
        'analysis_method': method, 'method_evaluation': evaluation, 'method_context': context,
        'reference_only': True, 'account_scope': 'none', 'structure_only': False,
        'action': direction, 'category': 'buy' if direction == 'BUY' else 'sell' if direction == 'SELL' else 'watch',
        'reason_codes': reasons, 'events': latest, 'source_reference_points': latest,
        'data_version': raw.attrs.get('data_version'), 'strategy_version': VERSION}


def _structure_chart(light, spec, day, calendar, prepared, raw_identity):
    """One expiring structural chart shared across all model combinations."""
    from my_strategy.web_dashboard.trend_strategy import analyze_trend_consensus
    root=Path(__file__).parent.parent
    paths=('web_dashboard/trend_strategy.py','web_dashboard/chart_evidence_trend.py',
           'web_dashboard/structure_projection.py','services/czsc_trend_runtime.py',
           'services/reference_runtime.py')
    identity={'kind':'method_structure_chart_v1','symbol':spec['symbol'],
        'start':spec.get('start'),'end':day,'calendar':calendar,'source':prepared[3],
        'adapter':hashlib.sha256(inspect.getsource(_structure_chart).encode()).hexdigest(),
        'generating_code':{path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths}}
    cached=light.store.get(identity,namespace='method_structure_charts')
    if cached:return cached
    chart=analyze_trend_consensus(symbol=spec['symbol'],start=spec.get('start'),end=day,
        as_of=day,structure_only=True,db_path=light.raw_db)
    if database_identity(light) != raw_identity:
        raise ValueError('分析期间原始行情发生改变，结构图表未缓存')
    # This is an ordinary bounded preview, never pinned and never a run. Model
    # identity deliberately does not enter the native chart generation key.
    light.store.gc(dry_run=False)
    light.store.put(identity,chart,namespace='method_structure_charts')
    light.store.limit_unpinned('method_structure_charts',keep=20,dry_run=False)
    return chart


def analyze_method(light, spec):
    day, calendar = target_date(light, spec)
    stop = Stop(time.monotonic() + float(spec.get('budget_seconds', 300)))
    with compute_lock(light):
        raw_identity = database_identity(light)
        frozen = freeze_method(light, spec, day)
        prepared = prepare_stock(light, spec['symbol'], day, stop)
        row = evaluate_prepared(light, spec['symbol'], day, frozen, prepared, stop)
        stop.check()
        result = _structure_chart(light,spec,day,calendar,prepared,raw_identity)
        result.update(analysis_method=spec['analysis_method'], method_context=frozen['context'],
            method_evaluation=row['method_evaluation'], display_name=METHODS[spec['analysis_method']],
            strategy_version=VERSION, request={**spec, **frozen['context']},
            method_identity={'code': frozen['code'], 'model_version': frozen['model_version'],
                'prediction': frozen['prediction_identity'], 'calendar': calendar})
        result['capabilities'].update(kronos_prediction=True, ma_score=spec['analysis_method'] in SCORED)
        if database_identity(light) != raw_identity:
            raise ValueError('分析期间原始行情发生改变，请重新分析')
        result['method_identity']['raw'] = raw_identity
        return result


def _placeholder(symbol, day, method, status, reason):
    return {'symbol': symbol, 'data_end': None, 'date': day, 'analysis_method': method,
            'category': 'excluded', 'action': 'WAIT', 'reason_codes': [reason],
            'method_evaluation': {'method': method, 'label': METHODS[method], 'as_of': day,
                'status': status, 'missing_reasons': [reason], 'direction': 'WAIT', 'score': None,
                'kronos': None}, 'reference_only': True, 'account_scope': 'none'}


def scan_methods(light, result_store, spec, *, progress=None, check_cancel=None, run_callback=None, invocation_source='web', _execution_started=None):
    with _ScanTiming(result_store, spec, check_cancel, _execution_started) as timing:
        stop = timing.stop
        timing.check_initialization()
        from my_strategy.adapters.czsc_adapter import list_symbols, normalize_symbol
        method = spec['analysis_method']
        catalog = list_symbols(limit=10000, db_path=light.raw_db)
        timing.check_initialization()
        names = {row['symbol']: row.get('name', '') for row in catalog}
        previous = result_store.get(spec['resume_scan_id']) if spec.get('resume_scan_id') else None
        selected = list(dict.fromkeys(normalize_symbol(s) for s in spec.get('symbols', [])))
        if previous:
            if previous.get('strategy_version') != VERSION: raise ValueError('仅可续算同一组合方法的扫描')
            if selected and selected != previous['population']: raise ValueError('续算股票池已改变')
            selected = previous['population']
            original = previous['method_context']
            for key, value in request_context(spec).items():
                if key in original and value != original[key] and not (key == 'score_model_id' and value is None):
                    raise ValueError('续算方法、日期、模型或预测参数已改变')
            spec = {**spec, **original}
        # Resume the original fixed end date even when the caller omitted end and
        # newer market bars are now present. Identity checks still reject changes.
        timing.check_initialization()
        day, calendar = target_date(light, spec)
        timing.check_initialization()
        if not selected:
            selected = [s for s in names if s.endswith('.SH') and s.startswith('6') or s.endswith('.SZ') and s.startswith(('0', '3'))]
        if not selected: raise ValueError('没有可扫描的沪深股票')
        with compute_lock(light):
            timing.check_initialization()
            frozen = freeze_method(light, spec, day)
            timing.check_initialization()
            raw_identity = database_identity(light, timing.check_initialization)
            timing.check_initialization()
            identity = {'population': selected, 'context': frozen['context'], 'calendar': calendar,
                'raw': raw_identity, 'code': frozen['code'], 'prediction': frozen['prediction_identity'],
                'model_version': frozen['model_version']}
            if previous and previous['frozen_identity'] != identity:
                raise ValueError('行情、日历、模型或生成代码已改变，请新建扫描')
            context = create_run_context(task='method-scan', as_of_date=day, stocks=selected,
                config=identity, data_version=stable_hash(raw_identity), scope='full' if not spec.get('symbols') else 'batch',
                source=invocation_source, start_date=spec.get('start'), end_date=day)
            timing.run_id = context.run_id
            if run_callback: run_callback(context.run_id)
            timing.check_initialization()
            timing.initialization_complete()
            completed = {row['symbol']: row for row in (previous or {}).get('rows', [])
                         if row.get('method_evaluation', {}).get('status') not in {'deferred'}}
            # Failures are terminal for this frozen batch; they remain inspectable.
            pending = [s for s in selected if s not in completed]
            max_calls = int(spec.get('max_predictions', 200))
            workers = min(4, int(spec.get('cpu_workers') or 2))
            calls = 0; status = 'partial'
            prepare_stats={'maximum_buffered_stocks':0,'buffer_limit':2*workers,'prediction_order':'frozen_population'}

            def snapshot(final_status):
                rows = [{**(completed.get(s) or _placeholder(s, day, method, 'deferred', 'batch_budget_deferred')),
                         'name': names.get(s, '')} for s in selected]
                failed = sum(row['method_evaluation']['status'] == 'failed' for row in rows)
                deferred = sum(row['method_evaluation']['status'] == 'deferred' for row in rows)
                success = len(rows) - failed - deferred
                unavailable = sum(row['method_evaluation']['status'] == 'partial' for row in rows)
                return {'run_id': context.run_id, 'run_context': context.to_dict(), 'status': final_status,
                    'strategy_version': VERSION, 'analysis_mode': 'reference', 'analysis_method': method,
                    'method_context': frozen['context'], 'request': spec, 'frozen_identity': identity,
                    'population': selected, 'parent_scan_id': spec.get('resume_scan_id'),
                    'data_end': day, 'as_of': day, 'data_range': {'start': spec.get('start'), 'end': day, 'requested_end': spec.get('end')},
                    'coverage': {'requested': len(selected), 'success': success, 'successful': success,
                                 'failed': failed, 'deferred': deferred, 'unavailable': unavailable,
                                 'evaluated': len(completed), 'complete': deferred == 0},
                    'rows': rows, 'failures': [{'symbol': row['symbol'], 'error': '; '.join(row['reason_codes'])}
                        for row in rows if row['method_evaluation']['status'] == 'failed'],
                    'compute_info': {'method': method, 'cpu_workers': workers, 'prediction_calls_this_batch': calls,
                        'prediction_calls_total': calls + (previous or {}).get('compute_info', {}).get('prediction_calls_total', 0),
                        'single_prediction_owner': True, 'preparation':dict(prepare_stats), 'score_model_id': frozen['context'].get('score_model_id'),
                        'timing': timing.snapshot()},
                    'market_history': {'status': 'unavailable', 'reason': 'method_history_not_generated',
                        'message': '本方法仅计算截止日；未生成的历史预测不会回填。'},
                    'reference_only': True, 'account_scope': 'none', 'updated_at': local_now().isoformat()}

            # CPU completion refills worker slots while the parent predicts. A
            # bounded ready buffer preserves frozen population/seed/quota order;
            # faster stocks never consume a slow earlier stock's prediction slot.
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='method-source')
            futures = {}; iterator = iter(pending)
            schedule_lock=threading.RLock();wake=threading.Event();feeder_exit=threading.Event()
            scheduling_errors=[]
            def fill():
                with schedule_lock:
                    running=sum(not future.done() for future in futures.values())
                    while (running<workers and len(futures)<2*workers and
                           calls+len(futures)<max_calls and not stop.is_set()):
                        symbol=next(iterator,None)
                        if symbol is None:break
                        futures[symbol]=pool.submit(prepare_stock,light,symbol,day,stop)
                        running+=1
                        prepare_stats['maximum_buffered_stocks']=max(prepare_stats['maximum_buffered_stocks'],len(futures))
            def feed():
                try:
                    while not feeder_exit.is_set():
                        fill()
                        wake.wait(.025);wake.clear()
                except BaseException as exc:
                    with schedule_lock:scheduling_errors.append(exc)
            feeder=threading.Thread(target=feed,name='method-source-refill',daemon=True)
            try:
                fill();feeder.start()
                while True:
                    stop.check()
                    with schedule_lock:
                        if scheduling_errors:raise scheduling_errors[0]
                        if not futures:break
                        symbol=next(iter(futures));future=futures[symbol]
                    try:
                        with timing.measure('waiting_preparation'):
                            while not future.done():
                                stop.check()
                                try: prepared = future.result(timeout=.2)
                                except TimeoutError: continue
                            prepared = future.result()
                        stop.check()
                        with schedule_lock:
                            futures.pop(symbol);calls+=1
                        wake.set()
                        fill()
                        with timing.measure('evaluation'):
                            completed[symbol] = evaluate_prepared(light, symbol, day, frozen, prepared, stop)
                    except (InterruptedError, TimeoutError):
                        raise
                    except Exception as exc:
                        completed[symbol] = _placeholder(symbol, day, method, 'failed', f'{type(exc).__name__}: {exc}')
                    with schedule_lock:futures.pop(symbol,None)
                    wake.set()
                    if progress:
                        progress('组合方法扫描', len(completed), len(selected),
                            failed=sum(r['method_evaluation']['status'] == 'failed' for r in completed.values()))
                    if len(completed) % 10 == 0:
                        with timing.measure('checkpoint_verification'):
                            if database_identity(light, stop.check) != raw_identity:
                                raise ValueError('扫描期间原始行情发生改变，检查点未发布')
                        stop.check()
                        with timing.measure('checkpoint_save'):
                            result_store.save('scan', snapshot('partial'))
                    fill()
                stop.check()
                status = 'succeeded' if len(completed) == len(selected) else 'partial'
            except Exception as exc:
                if stop.cancelled.is_set(): status = 'cancelled'
                elif isinstance(exc, (TimeoutError, InterruptedError)) and stop.is_set(): status = 'partial'
                else: raise
            finally:
                with timing.measure('drain'):
                    stop.cancelled.set();feeder_exit.set();wake.set()
                    if feeder.ident is not None:feeder.join()
                    for future in futures.values():future.cancel()
                    pool.shutdown(wait=True,cancel_futures=True)
            # Integrity verification, draining and atomic publication must finish
            # safely even after the computation deadline; receipt records overrun.
            with timing.measure('final_verification'):
                if database_identity(light) != raw_identity: raise ValueError('扫描期间原始行情发生改变，结果未发布')
            result = snapshot(status)
            if result['coverage']['failed'] and status == 'succeeded': result['status'] = 'partial'
            with timing.measure('final_save'):
                saved = result_store.save('scan', result)
            timing.status = result['status']
            return saved
