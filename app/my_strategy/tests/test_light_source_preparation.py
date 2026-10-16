from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
import threading,time
from pathlib import Path
import pandas as pd
import pytest
from my_strategy.services import light_source_preparation as preparation
from my_strategy.services.light_research import DeadlineStop
from my_strategy.tests.test_method_scan_acceleration import environment,_children


@pytest.mark.parametrize('logical_cpus,expected_workers',[(3,1),(4,2)])
def test_actual_account_spawn_sources_equal_original_with_full_history_and_identity(environment,monkeypatch,
        logical_cpus,expected_workers):
    from my_strategy.services import method_compute
    monkeypatch.setattr(method_compute.os,'cpu_count',lambda:logical_cpus)
    make,spec,symbols=environment
    serial,_=make('account-serial');parallel,_=make('account-parallel')
    before=_children();stop=DeadlineStop(None,time.monotonic()+120)
    expected=[(s,serial.source(s,spec['end'],stop=stop)) for s in symbols]
    progress=[];stats={}
    actual=list(preparation.prepare_account_sources(parallel,symbols,spec['end'],stop=stop,
        progress=lambda **v:progress.append(v),stats=stats))
    for (symbol,left),(other,right) in zip(expected,actual):
        assert symbol==other
        for index in (0,1):pd.testing.assert_frame_equal(left[index],right[index])
        assert left[2:]==right[2:]
    assert progress
    assert all(value['cpu_workers']==expected_workers for value in progress)
    assert stats['cpu_workers']==expected_workers
    assert stats['scheduling']==('spawn_fixed_population' if expected_workers>1 else 'original_serial')
    assert stats['source_tasks_submitted']==stats['source_tasks_received']==len(symbols)
    assert _children()<=before


def test_account_source_does_not_gain_scan_only_target_day_requirement(environment):
    make,spec,symbols=environment;light,_=make('after-last-price')
    day=(pd.Timestamp(spec['end'])+pd.Timedelta(days=1)).date().isoformat()
    result=list(preparation.prepare_account_sources(light,symbols,day,
        stop=DeadlineStop(None,time.monotonic()+120)))
    assert len(result)==len(symbols)
    assert all(str(value[0].date.iloc[-1].date())==spec['end'] for _,value in result)
    assert all(value[3]['as_of']==day for _,value in result)


def test_generator_close_always_drains_and_releases_workers(environment):
    make,spec,symbols=environment;light,_=make('close');before=_children()
    stream=preparation.prepare_account_sources(light,symbols,spec['end'],
        stop=DeadlineStop(None,time.monotonic()+120))
    next(stream);stream.close()
    assert _children()<=before


def test_custom_source_keeps_serial_admission_and_order():
    calls=[];light=SimpleNamespace(_check=lambda stop:None,
        source=lambda symbol,end,stop:(calls.append(symbol) or ('raw','frame','events',{'symbol':symbol})))
    result=list(preparation.prepare_account_sources(light,['600000.SH','000001.SZ'],'2026-10-16',
        stop=DeadlineStop(None,time.monotonic()+120)))
    assert calls==[s for s,_ in result]==['600000.SH','000001.SZ']


@pytest.mark.parametrize('helper',['light_source_preparation.py','method_scan_acceleration.py',
    'method_compute.py','analysis_methods.py'])
def test_new_account_freezes_every_execution_helper_and_one_byte_change_rejects(helper,monkeypatch):
    from my_strategy.services.light_research import code_identity
    from my_strategy.services.light_account_compatibility import account_code_compatible
    frozen=code_identity();assert helper in frozen
    original=Path.read_bytes
    def read(path):
        value=original(path)
        return value+b'\n' if path.name==helper else value
    monkeypatch.setattr(Path,'read_bytes',read)
    current=code_identity()
    assert set(frozen)==set(current)
    assert [key for key in frozen if frozen[key]!=current[key]]==[helper]
    assert not account_code_compatible(frozen,current)


def test_serial_source_budget_is_not_mislabeled_as_cancellation():
    light=SimpleNamespace(source=lambda *a,**k:None)
    with pytest.raises(TimeoutError):
        list(preparation.prepare_account_sources(light,['000001.SZ'],'2026-10-16',
            stop=DeadlineStop(None,time.monotonic()-1)))
    cancelled=threading.Event();cancelled.set()
    with pytest.raises(InterruptedError):
        list(preparation.prepare_account_sources(light,['000001.SZ'],'2026-10-16',
            stop=DeadlineStop(cancelled,time.monotonic()+120)))
