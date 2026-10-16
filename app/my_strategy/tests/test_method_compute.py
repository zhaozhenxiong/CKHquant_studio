from copy import deepcopy
import pytest
from my_strategy.services import method_compute as compute


def test_source_workers_are_bounded_reserve_coordinator_and_cap_population(monkeypatch):
    monkeypatch.setattr(compute.os,'cpu_count',lambda:14)
    assert compute.source_workers()==8
    assert compute.source_workers(12)==12
    assert compute.source_workers(population=2)==2
    monkeypatch.setattr(compute.os,'cpu_count',lambda:2)
    assert compute.source_workers()==1
    for value in (0,13,-1,True,2.5):
        with pytest.raises(ValueError):compute.source_workers(value)


def forecast(*,reused=False,device='mps',**execution):
    return {'reused':reused,'resource':{'execution':dict(schema='kronos_execution_v1',
        fresh_inference=True,actual_device=device,tensor_devices_verified=True,synchronization_completed=True,
        inference_calls=1,model_parameter_devices=[device],tokenizer_parameter_devices=[device],
        input_tensor_devices=[device],**execution)}}


def test_execution_counts_only_this_batch_verified_tensors_and_never_reuses_gpu_claim():
    calls=compute.PredictionExecution()
    calls.add(forecast());calls.add(forecast(reused=True));calls.add({'resource':{'device':'mps'}})
    assert calls.report()==dict(fresh_calls=2,reused_calls=1,execution_verified_calls=1,
        actual_devices={'mps':1},cached_provenance_devices={'mps':1})
    report=calls.report();report['actual_devices']['mps']=999
    assert calls.report()['actual_devices']['mps']==1


@pytest.mark.parametrize('key,value',[
    ('schema','old'),('tensor_devices_verified',False),('synchronization_completed',False),
    ('fresh_inference',False),('inference_calls',0),('model_parameter_devices',[]),
    ('tokenizer_parameter_devices',['cpu']),('input_tensor_devices',['mps','cpu'])])
def test_missing_or_conflicting_tensor_evidence_never_claims_gpu(key,value):
    prediction=forecast();prediction['resource']['execution'][key]=value
    calls=compute.PredictionExecution();calls.add(prediction)
    assert calls.report()['fresh_calls']==1
    assert calls.report()['execution_verified_calls']==0


def test_combined_api_cap_is_twelve_without_changing_legacy_sixteen():
    from my_strategy.web_dashboard.api import TaskSpec
    for method in ('structure_kronos','structure_kronos_ma_mlp','structure_kronos_ma_l2'):
        assert TaskSpec(analysis_method=method,cpu_workers=12).cpu_workers==12
        for value in (13,14,15,16):
            with pytest.raises(ValueError,match='12'):TaskSpec(analysis_method=method,cpu_workers=value)
    assert TaskSpec(analysis_method='legacy_reference',cpu_workers=16).cpu_workers==16
