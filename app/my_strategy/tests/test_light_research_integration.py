"""Lifecycle regressions at the lightweight orchestration/cache boundary.

No real model, market database or native producer is started by these tests.
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import light_research as research
from my_strategy.services import kronos_prediction, kronos_tuning, simple_score


def history(n=520):
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = np.linspace(10, 15, n)
    frame = pd.DataFrame({"date": dates, "symbol": "002436.SZ", "open": close,
                          "high": close + 1, "low": close - 1, "close": close,
                          "volume": 10000, "amount": 100000})
    for name in research.SCORE_COLUMNS:
        frame[name] = .5
    frame.attrs.update(data_version="fixture-raw-v1")
    return frame


@pytest.fixture
def service(tmp_path, monkeypatch):
    frame = history()
    monkeypatch.setattr(research, "load_bars", lambda *_a, **_k: frame.copy())
    monkeypatch.setattr(research, "code_identity", lambda: {"code": "v1"})
    monkeypatch.setattr(research, "prediction_identity", lambda _device="auto": {"model": "v1"})
    monkeypatch.setattr(kronos_prediction, "release_prediction_resources", lambda: None)
    light = research.LightResearch(tmp_path / "cache")
    dates = pd.bdate_range("2024-01-02", periods=550).strftime("%Y-%m-%d").tolist()
    monkeypatch.setattr(light, "calendar", lambda: (dates, {"calendar": "v1"}))
    monkeypatch.setattr(light, "source", lambda *_a, **_k: (frame, frame.copy(), [], {"source": "v1"}))
    yield light
    light.close()


def forecast_call(counter):
    def predict(*_a, **_kw):
        counter.append(True)
        return {"status": "completed", "cache_key": "immutable-inner-forecast",
                "identity": {"prediction": "same-input"}, "samples": [],
                "summary": {"terminal_return_median": .01},
                "quality": {"prediction_usable": True},
                "resource": {"elapsed_seconds": len(counter) / 10,
                             "model_reused": len(counter) > 1, "device": "cpu"}}
    return predict


def test_cached_prediction_retains_inspectable_result_reference(service, monkeypatch):
    calls = []
    monkeypatch.setattr(kronos_prediction, "predict", forecast_call(calls))
    end = history().date.iloc[-1].strftime("%Y-%m-%d")
    first = service.prediction("002436.SZ", end)
    second = service.prediction("002436.SZ", end)
    assert len(calls) == 1
    assert second["reused"]
    assert second["result_ref"]["cache_key"] == first["result_ref"]["cache_key"]


def test_checkpoint_updates_are_immutable_and_preserve_other_account_cursor(service):
    first=service._write_checkpoint({'account':'A'},{'cursor':1},'account_checkpoints')
    other=service._write_checkpoint({'account':'B'},{'cursor':2},'account_checkpoints')
    second=service._write_checkpoint({'account':'A'},{'cursor':3},'account_checkpoints')
    assert first['cache_key']!=second['cache_key']
    assert service._read_checkpoint({'account':'A'},'account_checkpoints')=={'cursor':3}
    assert service._read_checkpoint({'account':'B'},'account_checkpoints')=={'cursor':2}
    assert service.store.is_pinned(second['cache_key']) and service.store.is_pinned(other['cache_key'])
    assert not service.store.is_pinned(first['cache_key'])


def test_frozen_observation_batches_resume_all_outcomes_and_count_failed_calls(service,monkeypatch):
    from my_strategy.services import position_simulation
    frame=history(35);days=frame.date.dt.strftime('%Y-%m-%d').tolist();symbol='002436.SZ'
    monkeypatch.setattr(service,'source',lambda *a,**kw:(frame,frame,[],{'source':'test'}))
    baseline={'opportunities':[{'date':day,'symbol':symbol} for day in days[22:25]],
        'daily':[{'date':day,'positions':{}} for day in days[22:25]]}
    monkeypatch.setattr(position_simulation,'run_position_simulation',lambda *a,**kw:baseline)
    calls=[]
    def forecast(s,d,**kw):
        calls.append(d)
        assert isinstance(kw['stop'],research.DeadlineStop)
        if d==days[23]:raise ValueError('known unusable input')
        return {'as_of':d,'quality':{'prediction_usable':True}}
    monkeypatch.setattr(service,'prediction',forecast)
    first=service.collect_score_observations([symbol],days[20],days[-1],max_calls=2)
    assert first['coverage']=={'succeeded':1,'failed':1,'deferred':1}
    final=service.collect_score_observations(manifest_id=first['manifest_id'],max_calls=2)
    assert final['manifest_id']==first['manifest_id'] and final['status']=='completed'
    assert final['total_calls']==3 and final['batch_count']==2 and len(calls)==len(set(calls))
    assert final['observations'][1]['status']=='failed'


def test_b_training_manifest_keeps_prediction_failures_and_explains_only_input_exclusions():
    import copy
    frame=history(15);frame['as_of']=frame.date.dt.strftime('%Y-%m-%d');days=frame.as_of.tolist()
    manifest={'observations':[{'symbol':'002436.SZ','as_of':d,'status':'failed','forecast':None} for d in days]}
    original=copy.deepcopy(manifest)
    mature=frame.iloc[:10];eligible=mature.iloc[1:]
    derived=research.training_observation_manifest(manifest,frame,mature,eligible,days,days[-1])
    assert manifest==original and len(derived['observations'])==15
    assert derived['observations'][0]['training_exclusion_reason']=='core_feature_unavailable'
    assert all(o['training_exclusion_reason']=='immature_label' for o in derived['observations'][10:])
    assert all(o['status']=='failed' and 'training_exclusion_reason' not in o for o in derived['observations'][1:10])


def test_incomplete_b_observation_batch_is_pending_without_starting_training(service,monkeypatch):
    frame=history(35);days=frame.date.dt.strftime('%Y-%m-%d').tolist()
    monkeypatch.setattr(service,'source',lambda *a,**kw:(frame,frame,[],{'source':'test'}))
    manifest={'observations':[{'symbol':'002436.SZ','as_of':day,'status':'deferred','forecast':None} for day in days[25:]]}
    result=service.train_score(['002436.SZ'],days[-1],variant='B',observation_manifest=manifest)
    assert result['status']=='blocked' and result['reason']=='fusion_data_pending'
    assert not service.store.get_setting('light_experiment_registry')
    assert len(result['observation_manifest']['observations'])==10
    assert sum(bool(o.get('training_exclusion_reason')) for o in result['observation_manifest']['observations'])==5


def test_scorer_structure_prefix_does_not_reset_at_backtest_start(service,monkeypatch):
    from my_strategy.services import light_score_v2
    frame=history(70);day=frame.date.iloc[60].strftime('%Y-%m-%d')
    full=[{'date':frame.date.iloc[20].strftime('%Y-%m-%d'),'side':'BUY','confirmed':True,'risk_boundary':9}]
    seen=[]
    def score(artifact,row,**kwargs):
        seen.append(row[['structure_age_sessions','support_distance','confirmed_balance']].to_dict('records'))
        return {'scores':[60.],'probabilities':[[.2,.2,.6]],'qualification':{},'model_version':'frozen'}
    monkeypatch.setattr(light_score_v2,'score_light_samples',score)
    model={'artifact':{'schema':'light_score_v2','variant':'A'}}
    service.scorer(model,{'002436.SZ':frame},{'002436.SZ':full},
        score_source_events={'002436.SZ':full})('002436.SZ',day,{})
    service.scorer(model,{'002436.SZ':frame},{'002436.SZ':[]},
        score_source_events={'002436.SZ':full})('002436.SZ',day,{})
    assert seen[0]==seen[2] and seen[0][0]['structure_age_sessions']==40
    assert 'score_source_events' not in frame.attrs


def test_candidate_and_price_preparation_never_copy_event_metadata():
    from my_strategy.services.agent_evidence import volume_candidates
    from my_strategy.services.position_simulation import PreparedSimulationRuntime,_prepare_price_inputs
    class NoCopy:
        def __deepcopy__(self,memo):raise AssertionError('source events entered a pandas copy')
    frame=history(70)
    expected_candidates=volume_candidates(frame,'002436.SZ')
    dates=frame.date.dt.strftime('%Y-%m-%d').tolist()
    expected_prices=_prepare_price_inputs({'002436.SZ':frame},dates)
    frame.attrs['score_source_events']=NoCopy()
    assert volume_candidates(frame,'002436.SZ')==expected_candidates
    runtime=PreparedSimulationRuntime({'002436.SZ':frame},dates,start=dates[20],end=dates[-1])
    assert (runtime.bars,runtime.raw,runtime.dates)==expected_prices
    assert 'score_source_events' in frame.attrs  # The caller's frame stays intact.


def test_policy_registration_compacts_legacy_mlp_provenance_before_setting_limit(service):
    from my_strategy.services.light_experiment_registry import load_experiment_input
    from my_strategy.services.light_qualification import content_hash
    import json
    input_identity={'frozen_sources':'x'*24000,'as_of':'2025-12-31'}
    entries=[]
    for seed in (42,43):
        item={'kind':'mlp','variant':'A','configuration':{'seed':seed},'input_identity':input_identity}
        item['experiment_id']=content_hash(item);entries.append(item)
    legacy={'schema':'light_experiment_registry_v1','entries':entries}
    service.store.set_setting('light_experiment_registry',legacy)
    # The previous inline layout would exceed the setting cap at this append.
    config={'cost_stress':1.5,'frozen_policy':'y'*20000}
    assert len(json.dumps(legacy).encode())+len(json.dumps(config).encode())>65536
    result=service._register_experiments('policy',[config])
    assert [e['experiment_id'] for e in result['entries'][:2]]==[e['experiment_id'] for e in entries]
    assert len(json.dumps(service.store.get_setting('light_experiment_registry')).encode())<30000
    refs=[e['input_identity_ref'] for e in result['entries'][:2]]
    assert refs[0]==refs[1] and load_experiment_input(service.store,refs[0])==input_identity
    assert service._register_experiments('policy',[config])==result
    service.store.gc(dry_run=False)
    assert load_experiment_input(service.store,refs[0])==input_identity


def test_b_scorer_passes_actual_decision_time_to_both_fusion_heads(service,monkeypatch):
    from my_strategy.services import light_score_v2
    frame=history(70);day=frame.date.iloc[60].strftime('%Y-%m-%d');seen=[]
    actual=day+'T15:05:00+08:00';deadline=frame.date.iloc[61].strftime('%Y-%m-%d')+'T09:30:00+08:00'
    forecast={'as_of':day,'actual_available_at':actual,'execution_deadline':deadline,
        'quality':{'prediction_usable':True,'valid_fraction':1},
        'summary':{'terminal_return_median':.02,'terminal_return_p10':-.01,'terminal_return_p90':.05,'up_path_fraction':.7}}
    def score(artifact,row,**kwargs):
        seen.append((row.feature_available_at.iloc[0],kwargs['decision_time'],kwargs.get('model_kind','mlp')))
        return {'scores':[60.],'probabilities':[[.2,.2,.6]],'qualification':{},'model_version':'frozen'}
    monkeypatch.setattr(light_score_v2,'score_light_samples',score)
    model={'artifact':{'schema':'light_score_v2','variant':'B'}}
    result=service.scorer(model,{'002436.SZ':frame},{})('002436.SZ',day,forecast)
    assert result['consumes_kronos'] and result['l2_comparison']['consumes_kronos']
    assert [v[2] for v in seen]==['mlp','l2']
    assert all(pd.Timestamp(v[0])==pd.Timestamp(actual) and v[1]==actual for v in seen)


def test_agent_saved_daily_continuation_matches_initial_replay_and_shadow_role(service,monkeypatch):
    from my_strategy.services.light_account_continuation import continue_saved_account
    frame=history(45);days=frame.date.dt.strftime('%Y-%m-%d').tolist();symbol='002436.SZ'
    events=[dict(event_id='frozen-buy',date=days[22],side='BUY',confirmed=True,risk_boundary=10)]
    monkeypatch.setattr(service,'source',lambda *a,**kw:(frame,frame.copy(),events,{'source':'test'}))
    calls=[]
    def forecast(s,d,**kw):
        calls.append(d)
        return {'as_of':d,'quality':{'prediction_usable':True},'summary':{'terminal_return_median':.02,'up_path_fraction':.7}}
    monkeypatch.setattr(service,'prediction',forecast)
    previous=service.backtest([symbol],days[0],days[27],save=True)
    with pytest.raises(ValueError,match='历史候选'):service.enable_account(previous['result_id'])
    account=service.enable_account(previous['result_id'],shadow_research=True)
    assert account['role']=='shadow_research'
    assert service.enable_account(previous['result_id'],enabled=False)['role']=='shadow_research'
    calls.clear()
    continued=continue_saved_account(service,account,previous,days[-1])
    assert continued['account_role']=='shadow_research' and all(d>days[27] for d in calls)
    full=service.backtest([symbol],days[0],days[-1])
    for policy in full['comparisons']:
        for key in ('daily','orders','trades','decisions','state','metrics'):
            assert continued['comparisons'][policy][key]==full['comparisons'][policy][key],(policy,key)


def test_agent_continuation_missing_model_preserves_price_risk(service,monkeypatch):
    from my_strategy.services.light_account_continuation import continue_saved_account
    frame=history(45);days=frame.date.dt.strftime('%Y-%m-%d').tolist();symbol='002436.SZ'
    events=[dict(event_id='frozen-buy',date=days[22],side='BUY',confirmed=True,risk_boundary=10)]
    monkeypatch.setattr(service,'source',lambda *a,**kw:(frame,frame.copy(),events,{'source':'test'}))
    monkeypatch.setattr(service,'prediction',lambda s,d,**kw:{'as_of':d,'quality':{'prediction_usable':True},
        'summary':{'terminal_return_median':.02,'up_path_fraction':.7}})
    previous=service.backtest([symbol],days[0],days[27])
    previous['score_model']={'cache_key':'missing-frozen-score'};previous['identity']['score']='missing'
    price=frame.close.iloc[27]*.91
    frame.loc[28:,'open']=price;frame.loc[28:,'close']=price
    frame.loc[28:,'high']=price+.1;frame.loc[28:,'low']=price-.1
    account={'account_id':previous['result_id'],'last_result_id':previous['result_id'],'role':'shadow_research'}
    result=continue_saved_account(service,account,previous,days[-1])
    assert result['status']=='partial' and result['dependency_errors'][0]['dependency']=='frozen_score'
    baseline=result['comparisons']['structure_rules']
    assert baseline['daily'][-1]['date']==days[-1]
    assert any(o['side']=='SELL' and o['hard'] for o in baseline['orders'])


def test_expired_prediction_can_recompute_without_volatile_identity_collision(service, monkeypatch):
    calls, now = [], [1000.]
    monkeypatch.setattr(kronos_prediction, "predict", forecast_call(calls))
    service.store.clock = lambda: now[0]
    service.store.ttl_seconds = 1
    end = history().date.iloc[-1].strftime("%Y-%m-%d")
    first = service.prediction("002436.SZ", end)
    now[0] += 2
    second = service.prediction("002436.SZ", end)
    assert second["cache_key"] == first["cache_key"]
    assert second["summary"] == first["summary"]
    assert len(service.store.list(namespace="predictions")) == 1


@pytest.mark.parametrize("status", ["cancelled", "partial"])
def test_incomplete_tuning_is_retryable_and_not_a_completed_cache_hit(service, monkeypatch, status):
    calls = []
    def tune(*_a, **_k):
        calls.append(True)
        return {"status": status, "selected_configuration": None}
    monkeypatch.setattr(kronos_tuning, "tune_kronos", tune)
    args = (["002436.SZ"], ["2024-03-01"], ["2024-04-01"])
    assert service.tune(*args)["status"] == status
    assert service.tune(*args)["status"] == status
    assert len(calls) == 2
    assert not service.store.list(namespace="tuning")


def test_verified_calendar_revision_invalidates_tuning_cache(service, monkeypatch):
    calls = []
    def tune(*_a, **_k):
        calls.append(True)
        return {"status": "completed", "selected_configuration": {"lookback": 128}}
    monkeypatch.setattr(kronos_tuning, "tune_kronos", tune)
    args = (["002436.SZ"], ["2024-03-01"], ["2024-04-01"])
    service.tune(*args)
    days, _ = service.calendar()
    monkeypatch.setattr(service, "calendar", lambda: (days, {"calendar": "corrected-v2"}))
    service.tune(*args)
    assert len(calls) == 2


def test_cancellation_during_score_fit_cannot_publish_current_model(service, monkeypatch):
    stop = threading.Event()
    def fit(*_a, **_kw):
        stop.set()
        return {"model_version": "unfinished-model"}
    monkeypatch.setattr(simple_score, "fit_simple_score", fit)
    with pytest.raises(InterruptedError):
        service.train_score(["002436.SZ"], stop=stop,model_type="l2")
    assert service.current_score() is None
    assert not service.store.list(namespace="score_models")


def test_failed_predictions_count_against_the_inference_budget(service, monkeypatch):
    frame=history(30)
    symbols=['002436.SZ','002156.SZ','600519.SH']
    def source(symbol,*args,**kwargs):
        f=frame.assign(symbol=symbol)
        events=[dict(event_id='B-'+str(i),date=day.strftime('%Y-%m-%d'),side='BUY',confirmed=True,risk_boundary=9)
                for i,day in enumerate(frame.date[5:8])]
        return f,f,events,{'symbol':symbol,'source':'v1'}
    monkeypatch.setattr(service,'source',source)
    calls=[]
    def fail(*args,**kwargs):calls.append(args);raise ValueError('unusable input')
    monkeypatch.setattr(service,'prediction',fail)
    result=service.backtest(symbols,'2024-01-02','2024-02-12',max_predictions=1,candidate_sessions=1)
    assert len(calls)==3  # one logical call each day, including failures
    coverage=result['prediction_coverage']
    assert coverage['requested']==sum(coverage[k] for k in ('succeeded','unusable','failed','budget_skipped'))
    assert coverage['failed']==3 and coverage['budget_skipped']==6


def test_cancelled_worker_result_is_not_marked_succeeded(service, monkeypatch):
    monkeypatch.setattr(service, "tune", lambda **_kw: {"status": "cancelled"})
    service.jobs["fixture"] = {"job_id": "fixture", "kind": "tuning", "spec": {},
                               "status": "queued", "fingerprint": "fixture"}
    service.stops["fixture"] = threading.Event()
    service._run("fixture")
    assert service.get_job("fixture")["status"] == "cancelled"


def test_cancelling_job_remains_deduplicated_until_worker_exits(service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def predict(*_a, stop=None, **_kw):
        entered.set()
        assert release.wait(3), "test failed to release its worker"
        if stop.is_set():
            raise InterruptedError("cancelled")
        return {"status": "completed"}
    monkeypatch.setattr(service, "prediction", predict)
    first = service.submit("prediction", {"symbol": "002436.SZ"})
    try:
        assert entered.wait(3)
        service.cancel(first["job_id"])
        duplicate = service.submit("prediction", {"symbol": "002436.SZ"})
        assert duplicate["job_id"] == first["job_id"]
    finally:
        release.set()


def test_daily_cancellation_after_last_forecast_never_repeats_account_continuation(service, monkeypatch):
    from my_strategy.services import light_daily
    stop, calls = threading.Event(), []
    monkeypatch.setattr(light_daily, "latest_market_date", lambda **_kw: "2025-12-29")
    def prediction(*_a, **_kw):
        calls.append("prediction")
        stop.set()
        return {"status": "completed", "cache_key": "fixture"}
    monkeypatch.setattr(service, "prediction", prediction)
    monkeypatch.setattr(service, "continue_accounts", lambda *_a, **_kw: calls.append("accounts") or [])
    try:
        result = light_daily.sync_light_daily(required_price_date="2025-12-29", symbols=["002436.SZ"],
                                              service=service, check_cancel=stop.is_set)
        assert result["status"] == "cancelled"
    except InterruptedError:
        pass
    assert calls == ["accounts", "prediction"]


def test_cancel_at_final_forecast_boundary_prevents_publication(service, monkeypatch):
    stop = threading.Event()
    calls = []
    def predict(*args, **kwargs):
        result = forecast_call(calls)(*args, **kwargs)
        stop.set()
        return result
    monkeypatch.setattr(kronos_prediction, "predict", predict)
    end = history().date.iloc[-1].strftime("%Y-%m-%d")
    with pytest.raises(InterruptedError):
        service.prediction("002436.SZ", end, stop=stop)
    assert not service.store.list(namespace="predictions")


def test_blocked_worker_result_cannot_be_reported_as_success(service, monkeypatch):
    monkeypatch.setattr(service, "tune", lambda **_kw: {"status": "blocked", "reason": "missing source"})
    service.jobs["fixture"] = {"job_id": "fixture", "kind": "tuning", "spec": {},
                               "status": "queued", "fingerprint": "fixture"}
    service.stops["fixture"] = threading.Event()
    service._run("fixture")
    assert service.get_job("fixture")["status"] in {"blocked", "failed"}


def test_saving_reused_partial_preview_pins_result_and_lookup(service, monkeypatch):
    frame=history(30)
    event=dict(event_id='B',date=frame.date.iloc[5].strftime('%Y-%m-%d'),side='BUY',confirmed=True,risk_boundary=9)
    monkeypatch.setattr(service,'source',lambda *a,**kw:(frame,frame,[event],{'source':'v1'}))
    def fail(*args,**kwargs):raise ValueError('fixture failure')
    monkeypatch.setattr(service,'prediction',fail)
    first=service.backtest(['002436.SZ'],'2024-01-02','2024-02-12',max_predictions=1)
    assert first['status']=='partial' and not first['saved']
    saved=service.backtest(['002436.SZ'],'2024-01-02','2024-02-12',max_predictions=1,save=True)
    assert saved['result_id']==first['result_id'] and saved['reused'] and saved['saved']
    for namespace in ('simulations','result_lookup'):
        assert all(row['pin_count'] for row in service.store.list(namespace=namespace))


def test_separate_service_instances_cannot_run_inference_at_the_same_time(service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def predict(*args, **kwargs):
        entered.set()
        assert release.wait(3), "test failed to release inference"
        return forecast_call(calls)(*args, **kwargs)
    monkeypatch.setattr(kronos_prediction, "predict", predict)
    end = history().date.iloc[-1].strftime("%Y-%m-%d")
    other = research.LightResearch(service.store.root)
    future = service.queue.submit(service.prediction, "002436.SZ", end)
    try:
        assert entered.wait(3)
        with pytest.raises(RuntimeError, match="计算资源"):
            other.prediction("002436.SZ", end)
    finally:
        release.set()
        future.result(timeout=3)
        other.close()
    assert len(calls) == 1


def test_equivalent_integer_and_float_cash_reuse_the_same_backtest(service, monkeypatch):
    from my_strategy.services import light_daily_scheduler
    original=light_daily_scheduler.run_daily_schedule;calls=[]
    def run(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(light_daily_scheduler,'run_daily_schedule',run)
    first=service.backtest(['002436.SZ'],'2025-12-24','2025-12-29',initial_cash=1000000)
    second=service.backtest(['002436.SZ'],'2025-12-24','2025-12-29',initial_cash=1000000.0)
    assert second['reused'] and second['result_id']==first['result_id']
    assert len(calls)==1 and len(service.store.list(namespace='simulations'))==1
