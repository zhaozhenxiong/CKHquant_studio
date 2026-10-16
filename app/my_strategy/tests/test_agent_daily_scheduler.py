import copy
import threading
import time

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.light_daily_scheduler import run_daily_schedule,ScheduleCancelled
from my_strategy.services.position_simulation import run_position_simulation

S='002436.SZ'


def fixture():
    days=pd.bdate_range('2025-01-01',periods=32).strftime('%Y-%m-%d').tolist()
    price=10+np.arange(32)*.03
    frame=pd.DataFrame(dict(symbol=S,date=days,open=price,high=price+.2,low=price-.2,
        close=price,volume=10000000,amount=100000000,has_trade_price=1))
    event=dict(event_id='source-5',date=days[5],available_at=days[5]+'T15:00:00+08:00',
        side='BUY',confirmed=True,risk_boundary=9.5)
    configs={policy:dict(policy=policy,ma_period=3,atr_period=3,initial_cash=100000,
        commission=0,min_commission=0,slippage=0,transfer_fee=0,max_holding_sessions=200,
        time_exit_sessions=200,candidate_sessions=3,research_mode='historical_candidate',
        enable_agent_evidence=True) for policy in ('structure_rules','structure_kronos','structure_kronos_score')}
    return days,{S:frame},{S:[event]},configs


def predict(symbol,day):
    return dict(as_of=day,quality={'prediction_usable':True},summary={'terminal_return_median':.02,'up_path_fraction':.7})


def score(symbol,day,forecast):
    return dict(status='scored',score=70,model_type='torch_mlp',variant='A',
        qualification={'allow_increase':True,'research_trial_eligible':True,'policy_passed':False})


def run(days,frames,events,configs,**kw):
    return run_daily_schedule(frames,events,configs=configs,calendar=days,start=days[0],
        end=kw.pop('end',days[-1]),prediction=kw.pop('prediction',predict),score=score,
        include_benchmarks=False,**kw)


def test_initial_and_daily_continuation_share_observations_and_all_account_facts():
    days,frames,events,configs=fixture()
    full=run(days,frames,events,configs)
    partial=run(days,frames,events,configs,end=days[10])
    for day in days[11:]:
        partial=run(days,frames,partial['evidence'],configs,end=day,checkpoint=partial['checkpoint'])
    for policy in configs:
        for key in ('daily','orders','trades','decisions','metrics','state','opportunities'):
            assert full['comparisons'][policy][key]==partial['comparisons'][policy][key],(policy,key)
    assert full['evidence']==partial['evidence']
    assert full['coverage']==partial['coverage']
    assert full['scheduler']==partial['scheduler']
    assert full['comparisons']['structure_kronos_score']['trades']


def test_cache_hits_do_not_buy_more_observations_or_change_account():
    days,frames,events,configs=fixture()
    cold=run(days,frames,events,configs,max_predictions_per_day=1)
    hot=run(days,frames,events,configs,max_predictions_per_day=1,
        prediction=lambda s,d:{**predict(s,d),'reused':True,'resource':{'elapsed_seconds':0}})
    assert cold['coverage']==hot['coverage']
    assert cold['evidence']==hot['evidence']
    for policy in configs:assert cold['comparisons'][policy]==hot['comparisons'][policy]


def test_progress_reports_actual_whole_window_sessions_and_completed_observations():
    days,frames,events,configs=fixture();updates=[]
    result=run(days,frames,events,configs,progress=lambda **kw:updates.append(kw))
    assert updates
    for index,item in enumerate(updates):
        assert item['completed_sessions']==days.index(item['date'])
        assert item['total_sessions']==len(days)
        assert item['observations_completed']==index
        assert item['completed']==0 and item['total']==1
    assert len(updates)==result['coverage']['requested']


def test_checkpoint_pacing_preserves_every_account_fact_and_forces_final_snapshot(monkeypatch):
    import my_strategy.services.light_daily_scheduler as scheduler
    days,frames,events,configs=fixture();original=run(days,frames,events,configs)
    monkeypatch.setattr(scheduler.time,'monotonic',lambda:100.)
    snapshots=[]
    paced=run(days,frames,events,configs,checkpoint_callback=snapshots.append,
        checkpoint_stages={'day_committed','completed','partial','cancelled','interrupted'},checkpoint_interval_seconds=5.)
    assert paced==original
    assert [s['stage'] for s in snapshots]==['day_committed','completed']
    assert snapshots[-1]['complete_through']==days[-1]
    assert snapshots[-1]['comparisons']==original['comparisons']


def test_cancel_checkpoint_ignores_pacing_and_drains_exact_observation(monkeypatch):
    import my_strategy.services.light_daily_scheduler as scheduler
    days,frames,events,configs=fixture();stop=threading.Event();snapshots=[]
    monkeypatch.setattr(scheduler.time,'monotonic',lambda:100.)
    def cancel(s,d):stop.set();return predict(s,d)
    with pytest.raises(ScheduleCancelled) as exc:
        run(days,frames,events,configs,prediction=cancel,stop=stop,checkpoint_callback=snapshots.append,
            checkpoint_stages={'day_committed','completed','partial','cancelled','interrupted'},checkpoint_interval_seconds=5.)
    assert snapshots[-1]['stage']=='cancelled'
    assert snapshots[-1]==exc.value.checkpoint
    assert len(snapshots[-1]['pending']['outcomes'])==1
    stop.clear()
    resumed=run(days,frames,events,configs,checkpoint=exc.value.checkpoint)
    full=run(days,frames,events,configs)
    assert resumed['comparisons']==full['comparisons'] and resumed['evidence']==full['evidence']


def test_paced_deadline_checkpoint_keeps_quota_and_resume_calls_exactly_once(monkeypatch):
    import my_strategy.services.light_daily_scheduler as scheduler
    days,frames,events,configs=fixture();clock=[100.];calls=[];snapshots=[]
    monkeypatch.setattr(scheduler.time,'monotonic',lambda:clock[0])
    def infer(s,d):
        calls.append((s,d));clock[0]=102.;return predict(s,d)
    partial=run(days,frames,events,configs,prediction=infer,deadline=101.,checkpoint_callback=snapshots.append,
        checkpoint_stages={'day_committed','completed','partial','cancelled','interrupted'},checkpoint_interval_seconds=5.)
    assert partial['status']=='partial' and snapshots[-1]['stage']=='interrupted'
    assert partial['coverage']['requested']==1
    assert partial['checkpoint']['pending'] is not None
    resumed=run(days,frames,partial['evidence'],configs,checkpoint=partial['checkpoint'],
        prediction=lambda s,d:(calls.append((s,d)) or predict(s,d)))
    full=run(days,frames,events,configs)
    assert len(calls)==len(set(calls))==resumed['coverage']['requested']
    assert resumed['comparisons']==full['comparisons'] and resumed['coverage']==full['coverage']


def test_paced_prediction_failure_remains_partial_and_forces_final_evidence():
    days,frames,events,configs=fixture();snapshots=[]
    def broken(s,d):raise ValueError('real unavailable input')
    result=run(days,frames,events,configs,prediction=broken,checkpoint_callback=snapshots.append,
        checkpoint_stages={'day_committed','completed','partial','cancelled','interrupted'},checkpoint_interval_seconds=5.)
    assert result['status']=='partial' and snapshots[-1]['stage']=='partial'
    assert result['coverage']['failed'] and result['errors']
    assert snapshots[-1]['errors']==result['errors']
    assert result['scheduler']['complete_through']<days[-1]


def test_unexpected_prediction_exception_seals_pending_without_publishing_or_double_quota():
    days,frames,events,configs=fixture();snapshots=[];completed=[]
    def broken(s,d):
        if completed:raise OSError('unexpected local input failure')
        completed.append((s,d));return predict(s,d)
    with pytest.raises(OSError,match='unexpected local input'):
        run(days,frames,events,configs,prediction=broken,checkpoint_callback=snapshots.append,
            checkpoint_stages={'day_committed','completed','partial','cancelled','interrupted'},checkpoint_interval_seconds=5.)
    checkpoint=snapshots[-1]
    assert checkpoint['stage']=='interrupted' and checkpoint['pending'] is not None
    assert checkpoint['coverage']['requested']==sum(checkpoint['coverage'][k] for k in ('succeeded','unusable','failed','budget_skipped'))==1
    resumed_calls=[]
    resumed=run(days,frames,checkpoint['evidence'],configs,checkpoint=checkpoint,
        prediction=lambda s,d:(resumed_calls.append((s,d)) or predict(s,d)))
    assert not set(completed)&set(resumed_calls)
    full=run(days,frames,events,configs)
    assert resumed['comparisons']==full['comparisons'] and resumed['coverage']==full['coverage']


@pytest.mark.parametrize('interval',[-1,6,float('nan'),float('inf')])
def test_checkpoint_pacing_rejects_unbounded_or_invalid_interval(interval):
    days,frames,events,configs=fixture()
    with pytest.raises(ValueError,match='Checkpoint interval'):
        run(days,frames,events,configs,checkpoint_interval_seconds=interval)


def test_cancel_substage_restarts_new_request_without_duplicate_calls():
    days,frames,events,configs=fixture();stop=threading.Event();calls=[];saved=[]
    def cancel_after_first(s,d):
        calls.append((s,d));stop.set();return predict(s,d)
    with pytest.raises(ScheduleCancelled) as exc:
        run(days,frames,events,configs,prediction=cancel_after_first,stop=stop,checkpoint_callback=saved.append)
    assert exc.value.checkpoint['stage']=='cancelled'
    stop.clear()
    resumed=run(days,frames,events,configs,checkpoint=exc.value.checkpoint,
        prediction=lambda s,d:(calls.append((s,d)) or predict(s,d)))
    full=run(days,frames,events,configs)
    assert len(calls)==len(set(calls))
    for policy in configs:assert full['comparisons'][policy]==resumed['comparisons'][policy]


def test_timeout_leaves_incomplete_cursor_but_price_risk_continues():
    days,frames,events,configs=fixture()
    first=run(days,frames,events,configs,end=days[8])
    frames[S].loc[9:,'open']=8.5;frames[S].loc[9:,'close']=8.5
    frames[S].loc[9:,'high']=8.7;frames[S].loc[9:,'low']=8.3
    result=run(days,frames,first['evidence'],configs,checkpoint=first['checkpoint'],deadline=time.monotonic()-1)
    assert result['status']=='partial'
    assert result['scheduler']['risk_through']==days[-1]
    assert result['scheduler']['complete_through']<days[-1]
    for v in result['risk_comparisons'].values():
        assert any(o['side']=='SELL' and o['hard'] for o in v['orders'])
    assert result['checkpoint']['pending'] is not None
    resumed=run(days,frames,result['evidence'],configs,checkpoint=result['checkpoint'])
    assert resumed['status']=='completed'
    assert resumed['scheduler']['complete_through']==days[-1]
    assert result['coverage']['requested']==sum(result['coverage'][k] for k in ('succeeded','unusable','failed','budget_skipped'))


def test_candidate_can_reassess_within_three_sessions_only():
    days,frames,events,configs=fixture()
    def initially_weak(s,d):
        p=predict(s,d)
        if d==days[5]:p['summary']['terminal_return_median']=-.02
        return p
    result=run(days,frames,events,configs,prediction=initially_weak)
    v=result['comparisons']['structure_kronos']
    assert v['trades'][0]['signal_date']==days[6]
    assert v['trades'][0]['date']==days[7]
    assert v['opportunities'][0]['source_event_ids']==v['opportunities'][1]['source_event_ids']
    never=run(days,frames,events,configs,prediction=lambda s,d:{**predict(s,d),'quality':{'prediction_usable':False}})
    opportunities=never['comparisons']['structure_kronos']['opportunities']
    assert [o['date'] for o in opportunities]==days[5:8]


def test_all_blockers_are_visible_and_missing_qualification_is_not_pass():
    days,frames,events,configs=fixture()
    bad={**events[S][0],'score':{'status':'scored','score':40,'qualification':{}},
         'forecast':{'quality':{'prediction_usable':False}}}
    v=run_position_simulation(frames,{S:[bad]},config=configs['structure_kronos_score'],
        calendar=days,start=days[0],end=days[-1],include_benchmarks=False)
    reasons=v['opportunities'][0]['all_reasons']
    assert set(reasons)>={'forecast_missing_or_unusable','simple_score_policy_not_passed','simple_score_below_threshold'}
    assert not v['trades']
    assert v['opportunity_funnel']['conservation_passed']


def test_partial_sell_has_negative_plan_and_actual_delta_not_zero_target():
    days,frames,events,configs=fixture()
    events[S].append(dict(event_id='sell',date=days[10],side='SELL',confirmed=True))
    v=run_position_simulation(frames,events,config=configs['structure_rules'],calendar=days,include_benchmarks=False)
    o=next(o for o in v['orders'] if o['side']=='SELL');t=next(t for t in v['trades'] if t['action']=='REDUCE')
    plan=o['quantity_plan'];actual=t['execution_change']
    assert plan['share_delta']<0 and plan['weight_delta']<0 and plan['target_shares']>0
    assert o['target_weight']>0 and t['target_weight']>0
    assert plan['target_shares']==plan['shares_before']+plan['share_delta']
    assert actual['shares_after']==actual['shares_before']+actual['share_delta']
    assert actual['trade_weight_delta']<0


def test_future_changes_do_not_alter_earlier_agent_evidence():
    days,frames,events,configs=fixture()
    before=run(days,frames,events,configs,end=days[12])
    frames[S].loc[20:,'close']*=1.1
    after=run(days,frames,events,configs,end=days[12])
    assert before['comparisons']==after['comparisons']


@pytest.mark.parametrize('late',[False,True])
def test_forward_forecast_after_close_uses_actual_time_and_never_backfills_open(late):
    days,frames,events,configs=fixture();observed=[None]
    def forecast(symbol,day):
        next_day=days[days.index(day)+1]
        observed[0]=next_day+'T09:31:00+08:00' if late else day+'T15:05:00+08:00'
        return predict(symbol,day)
    result=run(days,frames,events,configs,end=days[10],prediction=forecast,
        execution_mode='forward',observed_at=lambda:observed[0])
    observations=[e for e in result['evidence'][S] if e.get('observation_kind')=='daily_agent_observation']
    assert observations
    assert all(e['available_at']==e['forecast']['actual_available_at'] for e in observations)
    trades=result['comparisons']['structure_kronos']['trades']
    if late:
        assert not trades
        assert all(e['forecast'].get('late_for_decision') for e in observations)
    else:
        assert trades[0]['signal_date']==days[5] and trades[0]['date']==days[6]


def test_timeout_before_first_evaluation_can_resume_without_using_risk_cursor():
    days,frames,events,configs=fixture()
    events[S][0]['date']=days[0];events[S][0]['available_at']=days[0]+'T15:00:00+08:00'
    result=run(days,frames,events,configs,end=days[8],deadline=time.monotonic()-1)
    assert result['checkpoint']['comparisons']=={}
    assert result['comparison_scope']=='risk_only_no_completed_evaluation'
    resumed=run(days,frames,result['evidence'],configs,end=days[8],checkpoint=result['checkpoint'])
    complete=run(days,frames,events,configs,end=days[8])
    assert resumed['comparisons']==complete['comparisons']


def test_prepared_runtime_four_accounts_equal_uncached_and_normalize_prices_once(monkeypatch):
    from my_strategy.services import position_simulation as engine,light_daily_scheduler as scheduler
    days,frames,events,configs=fixture()
    configs['structure_kronos_l2']={**configs['structure_kronos_score'],'policy':'structure_kronos_l2'}
    events[S].append(dict(date=days[12],event_id='source-S',side='SELL',confirmed=True))
    actions=[dict(action_id='cash-dividend',source_id='fixture',symbol=S,record_date=days[9],
        ex_date=days[10],pay_date=days[13],cash_per_share=.05,tax_per_share=0)]
    original=engine._prepare_price_inputs;calls=[]
    def prepare(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(engine,'_prepare_price_inputs',prepare)
    optimized=run(days,frames,events,configs,corporate_actions=actions)
    assert len(calls)==1
    calls.clear()
    monkeypatch.setattr(scheduler,'PreparedSimulationRuntime',lambda *a,**kw:None)
    original_result=run(days,frames,events,configs,corporate_actions=actions)
    assert len(calls)==len(days)*len(configs)*2
    for key in ('comparisons','evidence','coverage','scheduler','checkpoint','status'):
        assert optimized[key]==original_result[key],key


def test_prepared_runtime_still_rejects_revised_price_or_evidence_prefix():
    from my_strategy.services.position_simulation import PreparedSimulationRuntime
    days,frames,events,configs=fixture()
    first=run(days,frames,events,configs,end=days[8])
    revised=copy.deepcopy(frames);revised[S].loc[2,'close']+=.01
    with pytest.raises(ValueError,match='prefix changed'):
        run(days,revised,first['evidence'],configs,end=days[9],checkpoint=first['checkpoint'])
    runtime=PreparedSimulationRuntime(frames,days,start=days[0],end=days[-1])
    before=run_position_simulation(frames,events,config=configs['structure_rules'],
        calendar=days,start=days[0],end=days[8],prepared_runtime=runtime,include_benchmarks=False)
    events[S][0]['risk_boundary']+=.01
    with pytest.raises(ValueError,match='prefix changed'):
        run_position_simulation(frames,events,config=configs['structure_rules'],calendar=days,
            start=days[0],end=days[9],state=before['state'],prepared_runtime=runtime,include_benchmarks=False)


@pytest.mark.parametrize('change', ['revise', 'add', 'remove'])
def test_prepared_runtime_binds_warmup_evidence_on_days_without_prices(change):
    from my_strategy.services.position_simulation import PreparedSimulationRuntime
    days,frames,events,configs=fixture()
    # Neither warmup confirmation date has a bar in the frozen population.
    frames[S]=frames[S].loc[~frames[S].date.isin([days[1],days[2]])].copy()
    warmup=dict(event_id='warmup-confirmation',date=days[1],available_at=days[1]+'T15:00:00+08:00',
        side='BUY',confirmed=True,risk_boundary=9.)
    events[S].append(warmup)
    runtime=PreparedSimulationRuntime(frames,days,start=days[3],end=days[-1])
    options=dict(config=configs['structure_rules'],calendar=days,start=days[3],end=days[10],
        include_benchmarks=False)
    uncached=run_position_simulation(frames,events,**options)
    prepared=run_position_simulation(frames,events,prepared_runtime=runtime,**options)
    assert prepared==uncached
    unchanged=run_position_simulation(frames,events,state=prepared['state'],prepared_runtime=runtime,
        **{**options,'end':days[11]})
    full=run_position_simulation(frames,events,**{**options,'end':days[11]})
    assert unchanged==full
    if change=='revise':warmup['risk_boundary']+=.01
    elif change=='add':
        events[S].append({**warmup,'event_id':'added-warmup','date':days[2],
            'available_at':days[2]+'T15:00:00+08:00'})
    else:events[S].remove(warmup)
    revised=run_position_simulation(frames,events,prepared_runtime=runtime,**{**options,'end':days[12]})
    assert revised==run_position_simulation(frames,events,**{**options,'end':days[12]})
    with pytest.raises(ValueError,match='prefix changed'):
        run_position_simulation(frames,events,state=unchanged['state'],prepared_runtime=runtime,
            **{**options,'end':days[12]})


def test_private_append_runtime_parses_each_observation_once_and_preserves_callbacks(monkeypatch):
    from my_strategy.services import light_daily_scheduler as scheduler
    from my_strategy.services.position_simulation import PreparedSimulationRuntime
    days, frames, events, configs = fixture()
    runtimes = []
    def make(*args, **kwargs):
        runtime = PreparedSimulationRuntime(*args, **kwargs)
        runtimes.append(runtime)
        return runtime
    monkeypatch.setattr(scheduler, 'PreparedSimulationRuntime', make)
    snapshots = []
    value = run(days, frames, events, configs, checkpoint_callback=snapshots.append,
                checkpoint_stages={'day_committed', 'completed', 'cancelled'})
    assert runtimes[0].stats['evidence_rows_parsed'] == len(value['evidence'][S])
    assert len(snapshots) == len(days) + 1
    assert all(s['stage'] != 'observing' for s in snapshots)
    saved = copy.deepcopy(snapshots[10])
    value['comparisons']['structure_rules']['state']['daily'][0]['cash'] = -1
    value['evidence'][S][0]['side'] = 'SELL'
    assert snapshots[10] == saved
    assert events[S][0]['side'] == 'BUY'


def test_cancel_checkpoint_is_complete_even_without_callback_stages():
    days, frames, events, configs = fixture()
    stop = threading.Event()
    def cancel(symbol, day):
        stop.set()
        return predict(symbol, day)
    with pytest.raises(ScheduleCancelled) as error:
        run(days, frames, events, configs, prediction=cancel, stop=stop, checkpoint_stages=set())
    stop.clear()
    resumed = run(days, frames, events, configs, checkpoint=error.value.checkpoint)
    complete = run(days, frames, events, configs)
    assert resumed['comparisons'] == complete['comparisons']


def test_metrics_materialize_only_at_public_checkpoint(monkeypatch):
    from my_strategy.services import position_simulation as engine
    days, frames, events, configs = fixture()
    original = engine._simulation_summary
    calls = []
    def materialize(*args):
        calls.append(args[0]['last_date'])
        return original(*args)
    monkeypatch.setattr(engine, '_simulation_summary', materialize)
    result = run(days, frames, events, configs)
    assert calls == [days[-1]] * len(configs)
    # Neither internal deferred markers nor mutable private state is published.
    assert '_deferred_summary' not in repr(result)
    resumed = run(days, frames, events, configs, checkpoint=result['checkpoint'])
    assert resumed['comparisons'] == result['comparisons']

