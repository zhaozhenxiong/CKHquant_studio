"""One causal observation schedule for initial replay and account continuation.

Logical quotas reset on verified sessions, including cache hits. A wall-clock
deadline records explicit deferred observations; risk processing still advances.
Cancellation checkpoints belong to a new request, never revive a cancelled job.
"""
from __future__ import annotations

from copy import deepcopy
from bisect import bisect_left, bisect_right
import time

import pandas as pd

from my_strategy.services.agent_evidence import identity
from my_strategy.services.position_simulation import run_position_simulation,PreparedSimulationRuntime,materialize_simulation

VERSION = 'causal_daily_scheduler_v1'
OBSERVATION = 'daily_agent_observation'
OUTCOMES = ('succeeded', 'unusable', 'failed', 'budget_skipped')


def stopped(stop):
    return bool(stop() if callable(stop) else stop.is_set() if stop is not None else False)


def decision_day(event, calendar):
    value = pd.Timestamp(event.get('available_at') or (str(event['date'])[:10]+'T15:00:00+08:00'))
    value = value.tz_localize('Asia/Shanghai') if value.tzinfo is None else value.tz_convert('Asia/Shanghai')
    day = value.date().isoformat()
    late = value > pd.Timestamp(day+'T15:00:00+08:00')
    index = bisect_right(calendar, day) if late else bisect_left(calendar, day)
    return calendar[index] if index < len(calendar) else None


class ScheduleCancelled(InterruptedError):
    def __init__(self, checkpoint):
        super().__init__('已取消；逐日观察检查点已保留，取消任务不会恢复')
        self.checkpoint = checkpoint


def run_daily_schedule(frames, evidence, *, configs, calendar, start, end,
                       prediction, score=None, comparisons=None, checkpoint=None,
                       max_predictions_per_day=60, deadline=None, stop=None,
                       progress=None, checkpoint_callback=None, corporate_actions=None,
                       execution_mode='historical', observed_at=None,
                       include_benchmarks=True, checkpoint_stages=None):
    if execution_mode not in {'historical','forward'}: raise ValueError('Unknown execution_mode')
    if max_predictions_per_day < 0: raise ValueError('Daily quota cannot be negative')
    prepared_runtime=PreparedSimulationRuntime(frames,calendar,start=start,end=end)
    state = deepcopy(checkpoint or {})
    if state and state.get('version') != VERSION: raise ValueError('Scheduler version changed; create a branch')
    all_evidence = deepcopy(state.get('evidence', evidence))
    if prepared_runtime is not None:
        prepared_runtime.own_append_only_evidence(all_evidence)
    accounts = deepcopy(state.get('comparisons', comparisons)) or {}
    original_last = min((v['state']['last_date'] for v in accounts.values()), default=None)
    coverage = deepcopy(state.get('coverage')) or dict(requested=0, succeeded=0, unusable=0,failed=0,budget_skipped=0)
    errors = deepcopy(state.get('errors', []))
    day_records = deepcopy(state.get('days', []))
    complete_through = state.get('complete_through', original_last)
    risk_through = state.get('risk_through', original_last)
    evaluation_gap = bool(state.get('evaluation_gap', False))
    pending = deepcopy(state.get('pending'))
    new_days = [d for d in calendar if start <= d <= end and (original_last is None or d > original_last)]
    effective = {s:{} for s in frames}
    for symbol, entries in all_evidence.items():
        for event in entries:
            if event.get('observation_kind') == OBSERVATION: continue
            day = decision_day(event, calendar)
            if day: effective[symbol].setdefault(day, []).append(event)

    def snapshot(stage, pending_value=None, *, required=False):
        deliver = checkpoint_callback is not None and (checkpoint_stages is None or stage in checkpoint_stages)
        if not required and not deliver:
            return None
        # Private daily states skip repeated metrics/JSON conversion. Every
        # externally visible checkpoint still has the original complete shape.
        published_accounts = {policy: materialize_simulation(value) for policy, value in accounts.items()}
        if pending_value is not None:
            pending_value = dict(pending_value)
            for key in ('day_start', 'risk_preview'):
                pending_value[key] = {policy: materialize_simulation(value)
                                      for policy, value in pending_value.get(key, {}).items()}
        item = dict(version=VERSION, stage=stage, comparisons=published_accounts,
            evidence=all_evidence, coverage=coverage, errors=errors, days=day_records,
            complete_through=complete_through, risk_through=risk_through,
            evaluation_gap=evaluation_gap, pending=pending_value,
            quota_contract={'per_session':max_predictions_per_day,'cache_hits_consume_quota':True,
                            'order':'held_then_active_candidates_then_new_candidates_symbol_ascending'},
            execution_mode=execution_mode)
        value = deepcopy(item)
        if deliver: checkpoint_callback(value)
        return value

    def check(pending_value=None):
        if stopped(stop): raise ScheduleCancelled(snapshot('cancelled', pending_value, required=True))

    def advance(day, previous, *, benchmarks=False, risk_only=False):
        if prepared_runtime is not None:prepared_runtime.begin_step()
        return {policy:run_position_simulation(frames,all_evidence,config=config,
            start=start,end=day,calendar=calendar,state=(previous.get(policy) or {}).get('state'),
            corporate_actions=corporate_actions,include_benchmarks=benchmarks,risk_only=risk_only,
            prepared_runtime=prepared_runtime,_defer_summary=not benchmarks)
            for policy,config in configs.items()}

    def interrupted_result(day, current):
        # Preserve the evaluation day-start and completed observation outcomes.
        # Independently process existing inventory risk through requested end;
        # these states never replace the resumable evaluation cursor.
        risk_accounts=deepcopy(accounts)
        for risk_day in [d for d in new_days if d>=day]:
            check(current)
            risk_accounts=advance(risk_day,risk_accounts,risk_only=True)
        value=snapshot('interrupted',current,required=True)
        value['risk_through']=end
        value['pending_observation_count']=len(current['queue'])-len(current['outcomes'])
        if checkpoint_callback:checkpoint_callback(value)
        risk_accounts={policy:materialize_simulation(value) for policy,value in risk_accounts.items()}
        return dict(comparisons=value['comparisons'] or risk_accounts,risk_comparisons=risk_accounts,
            evidence=all_evidence,coverage=coverage,errors=errors,checkpoint=value,status='partial',
            scheduler={k:v for k,v in value.items() if k not in {'comparisons','evidence','errors'}},
            comparison_scope='completed_evaluation' if accounts else 'risk_only_no_completed_evaluation')

    for day in new_days:
        check(pending)
        # This preview is deterministic and unpublished. It executes existing
        # orders and price/portfolio risk before optional expensive inference.
        preview = advance(day, accounts)
        held = {s for v in preview.values() for s,p in v['daily'][-1]['positions'].items()
                if p['shares'] or p.get('receivable_shares',0)}
        active = set()
        for result in accounts.values():
            index = len(result['state']['daily'])
            active.update(c['symbol'] for eid,c in result['state'].get('active_candidates',{}).items()
                if eid not in result['state'].get('consumed',[]) and index <= c['expires_index'])
        fresh = {s for s in frames if any(e.get('confirmed') is True and e.get('side','').upper() in {'B','BUY'}
                                          for e in effective[s].get(day,[]))}
        queue = sorted(held)+sorted(active-held)+sorted(fresh-held-active)
        current = pending if pending and pending.get('date') == day else dict(date=day,queue=queue,outcomes=[],
            day_start=accounts,risk_preview=preview,phase='risk_complete')
        if current['queue'] != queue: raise ValueError('Checkpoint causal queue changed')
        risk_through = day
        snapshot('observing', current)
        for ordinal,symbol in enumerate(queue):
            if ordinal < len(current['outcomes']): continue
            check(current)
            if deadline is not None and time.monotonic()>=deadline:
                return interrupted_result(day,current)
            coverage['requested'] += 1
            if progress: progress(stage='agent_observation',date=day,symbol=symbol,
                completed=ordinal,total=len(queue),risk_through=risk_through,complete_through=complete_through)
            status,forecast = None,None
            if ordinal >= max_predictions_per_day:
                status = 'budget_skipped'
                forecast = dict(as_of=day,status='deferred',reason='daily_logical_quota',
                                quality={'prediction_usable':False})
            else:
                try:
                    forecast = prediction(symbol,day)
                    status = 'succeeded' if (forecast.get('quality') or {}).get('prediction_usable') is True else 'unusable'
                except InterruptedError:
                    coverage['requested'] -= 1
                    if not stopped(stop) and deadline is not None and time.monotonic()>=deadline:
                        return interrupted_result(day,current)
                    raise ScheduleCancelled(snapshot('cancelled',current,required=True))
                except (ValueError,RuntimeError,TimeoutError) as exc:
                    status = 'failed'
                    forecast = dict(as_of=day,status='unavailable',reason=str(exc),quality={'prediction_usable':False})
                    errors.append(dict(symbol=symbol,date=day,error=str(exc)))
            reused = forecast.get('reused') is True
            forecast = deepcopy(forecast)
            forecast.pop('reused',None)
            # Runtime cache/resource diagnostics cannot change causal evidence.
            forecast.pop('resource',None)
            available = day+'T15:00:00+08:00'
            if execution_mode == 'forward':
                actual = pd.Timestamp(observed_at() if callable(observed_at) else observed_at or pd.Timestamp.now(tz='Asia/Shanghai'))
                actual = actual.tz_localize('Asia/Shanghai') if actual.tzinfo is None else actual.tz_convert('Asia/Shanghai')
                forecast['actual_available_at'] = actual.isoformat()
                available = actual.isoformat()
                next_session=next((d for d in calendar if d>day),None)
                executable_open=pd.Timestamp(next_session+'T09:30:00+08:00') if next_session else None
                forecast['execution_deadline']=executable_open.isoformat() if executable_open is not None else None
                if executable_open is None or actual >= executable_open:
                    # Never back-date a late model response into yesterday's
                    # close. It is inspectable evidence, not an executable B.
                    forecast['late_for_decision'] = True
                    forecast['quality'] = {**forecast.get('quality',{}),'prediction_usable':False}
                    forecast['reason'] = 'forecast_arrived_after_executable_open'
                    status = 'unusable'
            scored = None
            if score:
                try: scored = score(symbol,day,forecast)
                except ValueError as exc: scored = dict(status='unavailable',reason=str(exc))
            observation = dict(date=day,available_at=available,event_id='agent-observation:'+identity([day,symbol,VERSION])[:24],
                confirmed=False,observation_kind=OBSERVATION,forecast=forecast,forecast_status=status,score=scored,
                logical_ordinal=ordinal,quota_consumed=ordinal<max_predictions_per_day,
                execution_mode=execution_mode)
            # Exactly one observation per symbol/day. Existing immutable facts
            # must agree; continuation cannot upgrade an old missing response.
            known = next((e for e in all_evidence.setdefault(symbol,[]) if e.get('event_id')==observation['event_id']),None)
            if known is not None and known != observation: raise ValueError('Observation already frozen with different evidence')
            if known is None: all_evidence[symbol].append(observation)
            coverage[status] += 1
            current['outcomes'].append(dict(symbol=symbol,status=status,logical_ordinal=ordinal,
                cache_reused=reused))
            snapshot('observing',current)
        check(current)
        accounts = advance(day,accounts,benchmarks=include_benchmarks and day==new_days[-1])
        incomplete = any(x['status'] in {'budget_skipped','failed'} for x in current['outcomes'])
        evaluation_gap = evaluation_gap or incomplete
        if not evaluation_gap: complete_through = day
        day_records.append(dict(date=day,queue=queue,outcomes=current['outcomes'],
            risk_complete=True,evaluation_complete=not incomplete))
        pending=None
        snapshot('day_committed')
    if coverage['requested'] != sum(coverage[k] for k in OUTCOMES): raise AssertionError('Observation counts do not reconcile')
    final=snapshot('completed' if not evaluation_gap else 'partial',required=True)
    return dict(comparisons=final['comparisons'],evidence=all_evidence,coverage=coverage,errors=errors,
        scheduler={k:v for k,v in final.items() if k not in {'comparisons','evidence','errors'}},
        checkpoint=final,status='partial' if evaluation_gap else 'completed')
