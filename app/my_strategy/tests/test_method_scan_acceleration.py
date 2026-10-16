"""Equivalent spawn preparation, explicit provenance and bounded cleanup."""
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import deepcopy
import hashlib
import multiprocessing
import os
from pathlib import Path
import sqlite3
import threading
import time
from types import MethodType, SimpleNamespace

from filelock import FileLock
import pandas as pd
import pytest

from my_strategy.services import analysis_methods as methods
from my_strategy.services import method_scan_acceleration as accelerated
from my_strategy.services.light_research import LightResearch
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.storage.research_store import ResearchStore
from my_strategy.tests.test_current_native_evidence import _raw


def _clone_probe():
    return ThreadPoolExecutor


def test_clone_retains_original_bytecode_and_only_overrides_local_executor():
    replacement = object()
    original_global = _clone_probe.__globals__['ThreadPoolExecutor']
    cloned = accelerated._clone_scan(_clone_probe, replacement)
    assert cloned.__code__ is _clone_probe.__code__
    assert cloned() is replacement
    assert _clone_probe() is original_global is ThreadPoolExecutor
    assert cloned.__globals__ is not _clone_probe.__globals__


class RecordingResults(ResultStore):
    def __init__(self, root):
        super().__init__(root / 'results.db', root / 'runs')
        self.saved = []

    def save(self, kind, value):
        self.saved.append(deepcopy(value))
        return super().save(kind, value)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    symbols = ['000001.SZ', '600000.SH']
    raw = _raw(160, varied=True).reset_index(drop=True)
    dates = raw.date.dt.strftime('%Y-%m-%d').tolist()
    db = tmp_path / 'raw.db'
    with sqlite3.connect(db) as connection:
        for index, symbol in enumerate(symbols):
            frame = raw.copy(); frame['stock'] = symbol
            frame['date'] = frame.date.dt.strftime('%Y-%m-%d')
            frame.drop(columns=['symbol']).to_sql('stock_daily_normalized', connection,
                if_exists='replace' if index == 0 else 'append', index=False)
        connection.execute('CREATE TABLE securities(stock TEXT,name TEXT,first_date TEXT,last_date TEXT,raw_rows INTEGER)')
        connection.executemany('INSERT INTO securities VALUES(?,?,?,?,?)',
            [(symbol, symbol, dates[0], dates[-1], len(raw)) for symbol in symbols])
    from my_strategy.services import kronos_prediction, light_research
    monkeypatch.setattr(kronos_prediction, 'asset_status', lambda: {'ready': True})
    monkeypatch.setattr(light_research, 'prediction_identity', lambda *args: {'model': 'frozen-test-kronos'})
    lights = []
    def make(name):
        root = tmp_path / name; root.mkdir()
        store = ResearchStore(root / 'research', ttl_seconds=12345, quota_bytes=1024**3)
        score = {'status': 'scored', 'score': 70., 'probabilities': {'down': .1, 'flat': .2, 'up': .7},
            'model_version': 'test-ma', 'qualification': {'paper_allocation_eligible': False}}
        light = SimpleNamespace(raw_db=db, store=store, compute_lock=FileLock(str(root / 'compute.lock')),
            calendar=lambda: (list(dates), {'sha256': 'test-calendar'}),
            resolve_score_model=lambda *a, **k: {'model_id': 'a' * 64, 'artifact': {'model_version': 'test-ma'}},
            scorer=lambda *a, **k: lambda *args: {**deepcopy(score), 'l2_comparison': {**deepcopy(score), 'score': 60.}},
            prediction=lambda symbol, end, **kwargs: {'status': 'completed', 'as_of': end,
                'summary': {'terminal_return_median': .02}, 'quality': {'prediction_usable': True},
                'result_ref': {'cache_key': 'forecast-' + symbol}})
        light.source = MethodType(LightResearch.source, light)
        lights.append(light)
        return light, RecordingResults(root)
    spec = {'analysis_method': 'structure_kronos_ma_mlp', 'start': dates[0], 'end': dates[-1],
        'cpu_workers': 2, 'max_predictions': 2, 'budget_seconds': 120, 'lookback': 128,
        'horizon': 5, 'samples': 32, 'seed': 42, 'device': 'mps', 'usage_mode': 'historical'}
    yield make, spec, symbols
    for light in lights:
        light.store.close()


def _children():
    return {child.pid for child in multiprocessing.active_children()}


def test_spawn_scan_exact_rows_identity_order_and_parameters_match_original(environment):
    make, spec, symbols = environment
    base, base_results = make('base')
    process, process_results = make('spawn')
    before = _children(); original_spec = deepcopy(spec)
    baseline = methods.scan_methods(base, base_results, spec)
    actual = accelerated.scan_methods(process, process_results, spec)
    assert spec == original_spec
    assert actual['rows'] == baseline['rows']
    assert actual['frozen_identity'] == baseline['frozen_identity']
    assert actual['population'] == symbols == baseline['population']
    assert actual['coverage'] == baseline['coverage']
    assert actual['status'] == baseline['status'] == 'succeeded'
    assert actual['method_context'] == baseline['method_context']
    adapter = actual['compute_info']['execution_adapter']
    assert adapter['start_method'] == 'spawn' and adapter['cpu_workers'] == 2
    assert adapter['worker_pids'] and os.getpid() not in adapter['worker_pids']
    assert adapter['sha256'] == hashlib.sha256(Path(accelerated.__file__).read_bytes()).hexdigest()
    assert adapter['base_evaluator_sha256'] == hashlib.sha256(Path(methods.__file__).read_bytes()).hexdigest()
    assert adapter['source_tasks_submitted'] == adapter['source_tasks_received'] == 2
    assert adapter['algorithm_bytecode_preserved'] and adapter['prediction_owner'] == 'parent_process'
    assert all('execution_adapter' in value['compute_info'] for value in process_results.saved)
    assert _children() <= before


def test_spawn_resumes_original_checkpoint_without_rewriting_previous_rows(environment):
    make, spec, _ = environment
    light, results = make('resume')
    first = methods.scan_methods(light, results, {**spec, 'max_predictions': 1})
    frozen_first = deepcopy(results.get(first['run_id']))
    second = accelerated.scan_methods(light, results,
        {**spec, 'max_predictions': 1, 'resume_scan_id': first['run_id']})
    assert second['status'] == 'succeeded'
    assert second['rows'][0] == first['rows'][0]
    assert second['parent_scan_id'] == first['run_id']
    assert second['frozen_identity'] == first['frozen_identity']
    assert second['compute_info']['execution_adapter']['source_tasks_submitted'] == 1
    assert results.get(first['run_id']) == frozen_first


def test_spawn_cancel_after_first_prediction_preserves_remaining_population_and_joins(environment):
    make, spec, _ = environment
    light, results = make('cancel')
    requested = threading.Event(); original = light.prediction
    def prediction(*args, **kwargs):
        value = original(*args, **kwargs)
        requested.set()
        return value
    def cancelled():
        if requested.is_set():
            raise RuntimeError('user cancelled')
    light.prediction = prediction
    before = _children()
    actual = accelerated.scan_methods(light, results, {**spec, 'cpu_workers': 1}, check_cancel=cancelled)
    assert actual['status'] == 'cancelled'
    assert actual['coverage']['evaluated'] == 1 and actual['coverage']['deferred'] == 1
    assert actual['coverage']['failed'] == 0
    assert _children() <= before


def test_spawn_deadline_is_original_partial_and_leaves_no_worker(environment):
    make, spec, _ = environment
    light, results = make('deadline')
    before = _children()
    # The total budget now includes freeze/hash initialization. It cannot mint
    # a resumable identity or launch workers after preflight exhausted it.
    with pytest.raises(TimeoutError, match='预算'):
        accelerated.scan_methods(light, results, {**spec, 'budget_seconds': .001})
    assert results.list() == [] and results.saved == []
    assert _children() <= before


def test_process_pool_failure_cannot_be_swallowed_as_a_stock_failure_or_published(tmp_path):
    execution = accelerated._Execution(methods.scan_methods)
    inner = Future(); inner.set_exception(BrokenProcessPool('worker exited unexpectedly'))
    future = accelerated._SourceFuture(inner, execution)
    with pytest.raises(accelerated.PreparationInfrastructureError, match='worker exited'):
        future.result()
    assert execution.fatal
    store = RecordingResults(tmp_path)
    with pytest.raises(RuntimeError, match='infrastructure'):
        accelerated._RecordedResults(store, execution).save('scan', {'run_id': 'must-not-publish'})
    assert store.saved == [] and store.list() == []


def test_malformed_worker_response_is_infrastructure_error():
    execution = accelerated._Execution(methods.scan_methods)
    inner = Future(); inner.set_result({'prepared': None})
    with pytest.raises(accelerated.PreparationInfrastructureError, match='Malformed'):
        accelerated._SourceFuture(inner, execution).result()


def test_worker_storage_failures_escape_stock_failure_handler(monkeypatch):
    monkeypatch.setattr(accelerated, '_WORKER_CANCEL', threading.Event())
    def fail(*args):
        raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(methods, 'prepare_stock', fail)
    with pytest.raises(accelerated.PreparationInfrastructureError, match='database is locked') as error:
        accelerated._worker_prepare('000001.SZ', '2026-10-15', time.monotonic() + 60)
    execution = accelerated._Execution(methods.scan_methods)
    inner = Future(); inner.set_exception(error.value)
    with pytest.raises(accelerated.PreparationInfrastructureError):
        accelerated._SourceFuture(inner, execution).result()
    assert execution.fatal


def test_ordinary_source_value_error_retains_original_stock_failure_behavior():
    execution = accelerated._Execution(methods.scan_methods)
    inner = Future(); inner.set_exception(ValueError('missing target-day price'))
    with pytest.raises(ValueError, match='missing target'):
        accelerated._SourceFuture(inner, execution).result()
    assert execution.fatal is None


def test_worker_stop_checks_shared_cancel_and_original_monotonic_deadline(monkeypatch):
    event = threading.Event(); monkeypatch.setattr(accelerated, '_WORKER_CANCEL', event)
    live = accelerated._WorkerStop(time.monotonic() + 60)
    assert not live.is_set()
    event.set()
    with pytest.raises(InterruptedError): live.check()
    event.clear()
    expired = accelerated._WorkerStop(time.monotonic() - 1)
    with pytest.raises(TimeoutError): expired.check()


def test_recorded_results_do_not_mutate_input_or_previous_execution_metadata(tmp_path):
    store = RecordingResults(tmp_path)
    execution = accelerated._Execution(methods.scan_methods)
    value = {'run_id': 'metadata', 'rows': [{'symbol': '000001.SZ'}],
        'frozen_identity': {'unchanged': True}, 'compute_info': {'prediction_calls_total': 12}}
    previous = deepcopy(value)
    result = accelerated._RecordedResults(store, execution).save('scan', value)
    assert value == previous
    assert result['rows'] == value['rows'] and result['frozen_identity'] == value['frozen_identity']
    assert result['compute_info']['prediction_calls_total'] == 12
    assert 'execution_adapter' in result['compute_info']
