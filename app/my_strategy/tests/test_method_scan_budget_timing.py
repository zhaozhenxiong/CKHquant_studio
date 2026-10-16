"""End-to-end scan budget, integrity and truthful execution receipts."""
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from my_strategy.services import analysis_methods as methods
from my_strategy.services import method_scan_acceleration as accelerated
from my_strategy.tests.test_analysis_methods import environment, spec


def receipts(store):
    return [json.loads(p.read_text()) for p in (store.runs_root / '.method-execution').glob('*.json')]


@pytest.mark.parametrize('phase', ['catalog', 'freeze', 'physical_hash'])
def test_initialization_consumes_budget_without_source_or_frozen_report(environment, monkeypatch, phase):
    light, store, predictions = environment
    clock = [100.]
    monkeypatch.setattr(methods, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    from my_strategy.adapters import czsc_adapter
    if phase == 'catalog':
        original = czsc_adapter.list_symbols
        def slow(*args, **kwargs):
            result = original(*args, **kwargs); clock[0] += 2.; return result
        monkeypatch.setattr(czsc_adapter, 'list_symbols', slow)
    elif phase == 'freeze':
        original = methods.freeze_method
        def slow(*args, **kwargs):
            result = original(*args, **kwargs); clock[0] += 2.; return result
        monkeypatch.setattr(methods, 'freeze_method', slow)
    else:
        def slow(light, check=None):
            clock[0] += 2.
            if check: check()
            return {'database': 'raw-sha'}
        monkeypatch.setattr(methods, 'database_identity', slow)
    monkeypatch.setattr(methods, 'ThreadPoolExecutor', lambda **kw: pytest.fail('source executor started after preflight timeout'))
    with pytest.raises(TimeoutError, match='预算'):
        methods.scan_methods(light, store, spec(budget_seconds=1))
    assert not predictions and store.list() == []
    receipt, = receipts(store)
    assert receipt['status'] == 'budget_exhausted' and receipt['run_id'] is None
    assert receipt['phase_wall_seconds']['initialization'] == 2.
    assert receipt['budget_overrun_seconds'] == 1.


def test_initialization_preserves_task_cancel_exception(environment, monkeypatch):
    light, store, predictions = environment
    from my_strategy.web_dashboard.tasks import Cancelled
    def cancel(): raise Cancelled('requested')
    monkeypatch.setattr(methods, 'ThreadPoolExecutor', lambda **kw: pytest.fail('executor started for cancelled scan'))
    with pytest.raises(Cancelled, match='requested'):
        accelerated.scan_methods(light, store, spec(), check_cancel=cancel)
    assert store.list() == [] and not predictions
    receipt, = receipts(store)
    assert receipt['status'] == 'cancelled' and receipt['run_id'] is None


def test_budget_during_prediction_is_partial_but_user_cancel_is_cancelled(environment, monkeypatch):
    light, store, predictions = environment
    clock = [100.]
    monkeypatch.setattr(methods, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    original = light.prediction
    def expire(*args, **kwargs):
        clock[0] += 2.
        kwargs['stop'].check()
        return original(*args, **kwargs)
    light.prediction = expire
    result = methods.scan_methods(light, store, spec(budget_seconds=1, cpu_workers=1))
    assert result['status'] == 'partial' and result['coverage']['evaluated'] == 0
    assert result['population'] == ['000001.SZ', '600000.SH']
    receipt, = receipts(store)
    assert receipt['status'] == 'partial' and receipt['phase_wall_seconds']['evaluation'] == 2.
    assert receipt['budget_overrun_seconds'] == 1.
    assert store.get(result['run_id']) == result
    assert not any(t.name.startswith('method-source') and t.is_alive() for t in threading.enumerate())


def test_final_save_and_drain_receipt_do_not_rewrite_report(environment, monkeypatch):
    light, store, predictions = environment
    clock = [100.]
    monkeypatch.setattr(methods, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    original_prediction = light.prediction
    def prediction(*args, **kwargs):
        value = original_prediction(*args, **kwargs); clock[0] += 3.; return value
    light.prediction = prediction
    pool_type = methods.ThreadPoolExecutor
    class SlowDrain(pool_type):
        def shutdown(self, *args, **kwargs):
            result = super().shutdown(*args, **kwargs); clock[0] += 4.; return result
    monkeypatch.setattr(methods, 'ThreadPoolExecutor', SlowDrain)
    original_save = store.save
    save_count = []
    def save(*args, **kwargs):
        value = original_save(*args, **kwargs); save_count.append(1); clock[0] += 2.; return value
    store.save = save
    result = methods.scan_methods(light, store, spec(symbols=['000001.SZ'], budget_seconds=6))
    assert result['status'] == 'succeeded' and len(save_count) == 1
    assert store.get(result['run_id']) == result
    assert result['compute_info']['timing']['phase_wall_seconds']['evaluation'] == 3.
    assert result['compute_info']['timing']['phase_wall_seconds']['drain'] == 4.
    assert 'final_save' not in result['compute_info']['timing']['phase_wall_seconds']
    receipt = json.loads((store.runs_root / result['compute_info']['timing']['final_save_and_total']).read_text())
    assert receipt['phase_wall_seconds']['final_save'] == 2.
    assert receipt['total_before_receipt_seconds'] == 9.
    assert receipt['budget_overrun_seconds'] == 3.
    assert not receipt['receipt_write_included']


def test_checkpoint_save_measured_separately_and_resume_keeps_old_report(environment, monkeypatch):
    light, store, _ = environment
    clock = [100.]
    monkeypatch.setattr(methods, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    original = store.save
    def save(*args, **kwargs):
        result = original(*args, **kwargs); clock[0] += 2.; return result
    store.save = save
    result = methods.scan_methods(light, store, spec(symbols=[f'{n:06d}.SZ' for n in range(1,12)], max_predictions=10))
    assert result['status'] == 'partial' and result['coverage']['deferred'] == 1
    assert result['compute_info']['timing']['phase_wall_seconds']['checkpoint_save'] == 2.
    frozen = (store.runs_root / result['run_id'] / 'reports' / 'czsc.json').read_bytes()
    resumed = methods.scan_methods(light, store, spec(resume_scan_id=result['run_id']))
    assert resumed['status'] == 'succeeded'
    assert (store.runs_root / result['run_id'] / 'reports' / 'czsc.json').read_bytes() == frozen


def test_final_integrity_failure_cannot_be_reported_as_cancellation(environment, monkeypatch):
    light, store, _ = environment
    values = iter([{'database': 'before'}, {'database': 'after'}])
    monkeypatch.setattr(methods, 'database_identity', lambda *args: next(values))
    with pytest.raises(ValueError, match='结果未发布'):
        methods.scan_methods(light, store, spec(symbols=['000001.SZ']))
    receipt, = receipts(store)
    assert receipt['status'] == 'failed' and store.list() == []


def test_physical_hash_is_exact_and_cancellable_without_partial_memo(tmp_path):
    path = tmp_path / 'raw.db'
    path.write_bytes(b'abcd' * (10 * 1024 * 1024))
    wal = tmp_path / 'raw.db-wal'; wal.write_bytes(b'wal-input')
    light = SimpleNamespace(raw_db=path)
    methods._HASH_CACHE.clear()
    calls = []
    def cancel():
        calls.append(1)
        if len(calls) == 2: raise InterruptedError('cancel during full SHA')
    with pytest.raises(InterruptedError, match='full SHA'):
        methods.database_identity(light, cancel)
    assert not methods._HASH_CACHE
    assert methods.database_identity(light) == {
        'database': hashlib.sha256(path.read_bytes()).hexdigest(),
        'wal': hashlib.sha256(wal.read_bytes()).hexdigest()}
    with pytest.raises(InterruptedError):
        methods.database_identity(light, lambda: (_ for _ in ()).throw(InterruptedError('cached SHA cancellation')))


def test_execution_adapter_identity_setup_is_inside_same_total_budget(environment, monkeypatch):
    light, store, predictions = environment
    clock = [100.]
    timer = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(methods, 'time', timer)
    monkeypatch.setattr(accelerated, 'time', timer)
    execution_type = accelerated._Execution
    class SlowIdentity(execution_type):
        def __init__(self, *args):
            super().__init__(*args)
            clock[0] += 2.
    monkeypatch.setattr(accelerated, '_Execution', SlowIdentity)
    monkeypatch.setattr(accelerated, '_PreparationExecutor', lambda *a, **k: pytest.fail('executor started after adapter initialization deadline'))
    with pytest.raises(TimeoutError, match='预算'):
        accelerated.scan_methods(light, store, spec(budget_seconds=1))
    assert not predictions and store.list() == []
    receipt, = receipts(store)
    assert receipt['status'] == 'budget_exhausted'
    assert receipt['phase_wall_seconds']['initialization'] == 2.



def test_cold_hash_external_poll_count_is_bounded_independently_of_read_blocks(tmp_path, monkeypatch):
    path=tmp_path/'raw.db';path.write_bytes(b'x'*(40*1024*1024))
    methods._HASH_CACHE.clear()
    monkeypatch.setattr(methods,'time',SimpleNamespace(monotonic=lambda:100.))
    calls=[]
    actual=methods.database_identity(SimpleNamespace(raw_db=path),lambda:calls.append(1))
    assert actual['database']==hashlib.sha256(path.read_bytes()).hexdigest()
    # Entry, 16/32 MiB, verified digest, exit: not 40 one-MiB task queries.
    assert len(calls)==5


def test_cold_hash_time_gate_polls_before_sixteen_mib_and_never_memoizes_cancel(tmp_path, monkeypatch):
    path=tmp_path/'raw.db';path.write_bytes(b'x'*(4*1024*1024))
    methods._HASH_CACHE.clear()
    clock=[100.]
    def monotonic():
        clock[0]+=.06
        return clock[0]
    monkeypatch.setattr(methods,'time',SimpleNamespace(monotonic=monotonic))
    calls=[]
    def check():
        calls.append(1)
        if len(calls)==2:raise InterruptedError('time gate cancelled')
    with pytest.raises(InterruptedError,match='time gate'):
        methods.database_identity(SimpleNamespace(raw_db=path),check)
    assert len(calls)==2 and not methods._HASH_CACHE
