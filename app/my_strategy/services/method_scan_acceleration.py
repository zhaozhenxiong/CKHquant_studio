"""Spawn-only CPU preparation around the unchanged bounded scan algorithm.

Numerical functions, source identities, population order and the single parent
Kronos owner are unchanged. The adapter has its own recorded execution identity;
it never hides behind the original evaluator's source hash. Workers exchange
only trusted local task data, not persisted executable/native-state pickles.
"""
from __future__ import annotations

from concurrent.futures import CancelledError, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
import hashlib
import math
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import pickle
import sqlite3
import threading
import time
from types import FunctionType


VERSION = 'method_source_spawn_v1'
_WORKER_LIGHT = None
_WORKER_CANCEL = None


class PreparationInfrastructureError(InterruptedError):
    """Escape the evaluator's ordinary per-stock failure handler."""


class _WorkerStop:
    def __init__(self, deadline):
        self.deadline = deadline

    def is_set(self):
        return _WORKER_CANCEL.is_set() or time.monotonic() >= self.deadline

    def check(self):
        if _WORKER_CANCEL.is_set():
            raise InterruptedError('组合方法扫描已取消')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('本批时间预算已用尽')


class _SourceReader:
    """Only the fields consumed by the original LightResearch.source method."""
    def __init__(self, config):
        from my_strategy.storage.research_store import ResearchStore
        self.raw_db = config['raw_db']
        self.store = ResearchStore(config['cache_root'], ttl_seconds=config['ttl_seconds'],
            quota_bytes=config['quota_bytes'],
            orphan_scan_interval_seconds=config['orphan_scan_interval_seconds'])
        self._calendar = config['calendar']
        Finalize(self, self.store.close, exitpriority=10)

    def calendar(self):
        return deepcopy(self._calendar)

    def source(self, *args, **kwargs):
        from my_strategy.services.light_research import LightResearch
        return LightResearch.source(self, *args, **kwargs)


def _worker_initialize(config, cancel):
    global _WORKER_LIGHT, _WORKER_CANCEL
    from my_strategy.services import analysis_methods
    from my_strategy.services import light_research
    from my_strategy.services.incremental_source import code_identity
    if (hashlib.sha256(Path(analysis_methods.__file__).read_bytes()).hexdigest() != config['base_evaluator_sha256']
            or hashlib.sha256(Path(light_research.__file__).read_bytes()).hexdigest() != config['source_adapter_sha256']
            or code_identity() != config['source_identity']
            or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != config['adapter_sha256']):
        raise RuntimeError('CPU preparation worker code differs from its parent')
    if not Path(config['raw_db']).is_file():
        raise FileNotFoundError('CPU preparation raw database is unavailable')
    _WORKER_CANCEL = cancel
    _WORKER_LIGHT = _SourceReader(config)


def _worker_prepare(symbol, day, deadline):
    from my_strategy.services.analysis_methods import prepare_stock
    from my_strategy.storage.research_store import CacheCapacityError
    started = time.monotonic()
    try:
        prepared = prepare_stock(_WORKER_LIGHT, symbol, day, _WorkerStop(deadline))
    except (InterruptedError, TimeoutError):
        raise
    except (sqlite3.Error, CacheCapacityError, OSError) as exc:
        raise PreparationInfrastructureError(f'CPU source storage failed: {type(exc).__name__}: {exc}') from exc
    return {'prepared': prepared, 'pid': os.getpid(),
            'elapsed_seconds': time.monotonic() - started}


class _Execution:
    def __init__(self, original):
        from my_strategy.services import light_research
        self.identity = {'schema': VERSION, 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'strategy': 'cpu_process_spawn', 'start_method': 'spawn',
            'base_evaluator_sha256': hashlib.sha256(Path(original.__globals__['__file__']).read_bytes()).hexdigest(),
            'source_adapter_sha256': hashlib.sha256(Path(light_research.__file__).read_bytes()).hexdigest(),
            'algorithm_bytecode_preserved': True, 'prediction_owner': 'parent_process'}
        self.lock = threading.Lock()
        self.worker_pids = set()
        self.submitted = 0
        self.received = 0
        self.worker_seconds = 0.
        self.max_workers = None
        self.fatal = None

    def failure(self, exc):
        with self.lock:
            self.fatal = f'{type(exc).__name__}: {exc}'
        return PreparationInfrastructureError('CPU preparation infrastructure failed: ' + self.fatal)

    def report(self):
        with self.lock:
            return {**self.identity, 'cpu_workers': self.max_workers,
                'worker_pids': sorted(self.worker_pids), 'source_tasks_submitted': self.submitted,
                'source_tasks_received': self.received, 'source_worker_seconds': round(self.worker_seconds, 6)}


class _SourceFuture:
    def __init__(self, future, execution):
        self._future, self._execution = future, execution
        self._recorded = False

    def __getattr__(self, name):
        return getattr(self._future, name)

    def result(self, timeout=None):
        try:
            value = self._future.result(timeout=timeout)
        except (PreparationInfrastructureError, BrokenProcessPool, CancelledError, EOFError, ConnectionError, pickle.PicklingError) as exc:
            raise self._execution.failure(exc) from exc
        if (not isinstance(value, dict) or not isinstance(value.get('prepared'), tuple)
                or len(value['prepared']) != 4 or type(value.get('pid')) is not int or value['pid'] <= 0
                or not isinstance(value.get('elapsed_seconds'), (int, float))
                or not math.isfinite(value['elapsed_seconds']) or value['elapsed_seconds'] < 0):
            raise self._execution.failure(ValueError('Malformed CPU preparation worker response'))
        if not self._recorded:
            with self._execution.lock:
                self._execution.worker_pids.add(value['pid'])
                self._execution.received += 1
                self._execution.worker_seconds += value['elapsed_seconds']
            self._recorded = True
        return value['prepared']


class _PreparationExecutor:
    """The small Future/Executor surface actually used by scan_methods."""
    def __init__(self, light, execution, *, max_workers, thread_name_prefix=None):
        from my_strategy.adapters.czsc_adapter import raw_path
        from my_strategy.services.incremental_source import code_identity
        self.execution = execution
        self._closed = False
        self._parent_stop = None
        self._monitor = None
        self._monitor_exit = threading.Event()
        context = multiprocessing.get_context('spawn')
        self._cancel = context.Event()
        dates, calendar = light.calendar()
        config = {'raw_db': str(raw_path(light.raw_db).resolve()), 'cache_root': str(light.store.root.resolve()),
            'ttl_seconds': light.store.ttl_seconds, 'quota_bytes': light.store.quota_bytes,
            'orphan_scan_interval_seconds': light.store.orphan_scan_interval_seconds,
            'calendar': (list(dates), deepcopy(calendar)), 'source_identity': code_identity(),
            'base_evaluator_sha256': execution.identity['base_evaluator_sha256'],
            'source_adapter_sha256': execution.identity['source_adapter_sha256'],
            'adapter_sha256': execution.identity['sha256']}
        self._pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=context,
            initializer=_worker_initialize, initargs=(config, self._cancel))
        execution.max_workers = max_workers

    def _watch_cancel(self):
        while not self._monitor_exit.wait(.05):
            if self._parent_stop.is_set():
                self._cancel.set()
                return

    def submit(self, function, light, symbol, day, stop):
        from my_strategy.services.analysis_methods import prepare_stock
        if function is not prepare_stock:
            raise ValueError('CPU preparation accepts only the original prepare_stock')
        if self._parent_stop is None:
            self._parent_stop = stop
            self._monitor = threading.Thread(target=self._watch_cancel, name='method-source-cancel', daemon=True)
            self._monitor.start()
        elif self._parent_stop is not stop:
            raise ValueError('A preparation executor cannot mix scan cancellation scopes')
        stop.check()
        try:
            future = self._pool.submit(_worker_prepare, symbol, day, stop.deadline)
        except (BrokenProcessPool, RuntimeError) as exc:
            raise self.execution.failure(exc) from exc
        with self.execution.lock:
            self.execution.submitted += 1
        return _SourceFuture(future, self.execution)

    def shutdown(self, wait=True, *, cancel_futures=False):
        if self._closed:
            return
        self._closed = True
        self._cancel.set()
        self._monitor_exit.set()
        try:
            # A native stock replay can finish its current call before observing
            # cancellation. Always join it; never leave a writer behind a batch.
            self._pool.shutdown(wait=True, cancel_futures=cancel_futures)
        finally:
            if self._monitor is not None:
                self._monitor.join()


class _RecordedResults:
    def __init__(self, store, execution):
        self.store, self.execution = store, execution

    @property
    def runs_root(self):
        return getattr(self.store, 'runs_root', None)

    def get(self, *args, **kwargs):
        return self.store.get(*args, **kwargs)

    def save(self, kind, result):
        if self.execution.fatal:
            raise RuntimeError('CPU preparation infrastructure failed: ' + self.execution.fatal)
        value = {**result, 'compute_info': {**result.get('compute_info', {}),
            'execution_adapter': self.execution.report()}}
        return self.store.save(kind, value)


def _clone_scan(original, executor_factory):
    namespace = {**original.__globals__, 'ThreadPoolExecutor': executor_factory}
    cloned = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    cloned.__kwdefaults__ = original.__kwdefaults__
    return cloned


def scan_methods(light, result_store, spec, *, progress=None, check_cancel=None,
                 run_callback=None, invocation_source='web'):
    """Run the original evaluator with explicit, separately recorded scheduling."""
    started = time.monotonic()
    from my_strategy.services.analysis_methods import scan_methods as original
    execution = _Execution(original)
    executors = []
    def factory(**kwargs):
        executor = _PreparationExecutor(light, execution, **kwargs)
        executors.append(executor)
        return executor
    cloned = _clone_scan(original, factory)
    try:
        return cloned(light, _RecordedResults(result_store, execution), spec,
            progress=progress, check_cancel=check_cancel, run_callback=run_callback,
            invocation_source=invocation_source, _execution_started=started)
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
