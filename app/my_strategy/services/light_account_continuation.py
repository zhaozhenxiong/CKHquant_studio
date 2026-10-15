"""Append verified account sessions without regenerating past decisions.

The caller owns publication and the current account pointer. This adapter reads
the frozen previous result, verifies its economic input prefix, and returns an
immutable successor. It never changes a previous result or current-score pointer.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import time

import pandas as pd

from my_strategy.services.position_simulation import run_position_simulation

VERSION = "light_account_continuation_v2"
HOLDING_OBSERVATION = "held_position_forecast"


def _stopped(stop):
    return bool(stop() if callable(stop) else stop.is_set() if stop is not None else False)


def _check(stop):
    if _stopped(stop): raise InterruptedError("Daily account continuation cancelled")


class _StopAdapter:
    def __init__(self, stop): self.stop = stop
    def is_set(self): return _stopped(self.stop)


def _timestamp(event):
    value = pd.Timestamp(event.get("available_at") or (str(event["date"])[:10]+"T15:00:00+08:00"))
    return value.tz_localize("Asia/Shanghai") if value.tzinfo is None else value.tz_convert("Asia/Shanghai")


def _decision_day(event, dates):
    available = _timestamp(event)
    return next((day for day in dates if pd.Timestamp(day+"T15:00:00+08:00") >= available), None)


def _canonical(value):
    from my_strategy.services.light_research import plain
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _source_only(event):
    return {key:value for key,value in event.items() if key not in {"forecast", "score", "forecast_status"}}


def _is_holding_observation(event):
    return event.get("observation_kind") == HOLDING_OBSERVATION


def continue_saved_account(service, account, previous_result, end, *, stop=None, deadline=None, progress=None):
    """Validate the frozen prefix and return a successor; no publication occurs.

    `deadline` is an absolute time.monotonic() limit for optional prediction.
    Exhausting it never skips held-position risk processing. On cancellation or
    a changed historical input, raise and leave the caller's pointer unchanged.
    """
    if previous_result.get("version") == "agent_mlp_sim_v1":
        return _continue_agent_account(service,account,previous_result,end,stop=stop,deadline=deadline,progress=progress)
    _check(stop)
    started = time.monotonic()
    end = pd.Timestamp(end).date().isoformat()
    previous = deepcopy(previous_result)
    previous_end = previous["end"]
    if end < previous_end: raise ValueError("Cannot rewind an existing simulation account")
    if end == previous_end: return previous  # Exact immutable replay, including stored status fields.
    if account.get("last_result_id") and account["last_result_id"] != previous["result_id"]:
        raise ValueError("Account pointer does not match its supplied previous result")
    symbols = previous["symbols"]
    if sorted(account["symbols"]) != sorted(symbols) or account["start"] != previous["start"]:
        raise ValueError("Account population/start changed; create a new branch")
    if not previous.get("comparisons") or any(not r.get("state") for r in previous["comparisons"].values()):
        raise ValueError("Saved simulation has no resumable account checkpoint")
    dates, calendar_identity = service.calendar()
    if end not in dates: raise ValueError("Continuation end is not a verified session")
    adapter = _StopAdapter(stop)
    frames, evidence, identities, fresh_events = {}, {}, [], {}
    cutoff = pd.Timestamp(previous_end+"T15:00:00+08:00")
    beginning = pd.Timestamp(previous["start"]+"T00:00:00+08:00")
    for symbol in symbols:
        _check(stop)
        if progress: progress(stage="account_verify_prefix", symbol=symbol)
        _, frame, events, identity = service.source(symbol, end, stop=adapter)
        frames[symbol] = frame
        identities.append(identity)
        old = [deepcopy(e) for e in previous.get("source_events", {}).get(symbol, [])
               if (observed := _decision_day(e,dates)) is not None and observed <= previous_end]
        # Confirmed event history must agree independently of frozen forecasts.
        known_old = [_source_only(e) for e in old
                     if not _is_holding_observation(e) and beginning <= _timestamp(e) <= cutoff]
        known_now = [_source_only(e) for e in events
                     if not _is_holding_observation(e) and beginning <= _timestamp(e) <= cutoff]
        sorter = lambda e: (str(e.get("event_id", "")), str(e.get("available_at", "")), str(e.get("side", "")))
        if _canonical(sorted(known_old,key=sorter)) != _canonical(sorted(known_now,key=sorter)):
            raise ValueError(f"{symbol}: historical confirmed evidence changed; create a new branch")
        added = [deepcopy(e) for e in events
                 if (decision := _decision_day(e, dates)) is not None and previous_end < decision <= end]
        # Prior result's observations, probabilities and explicit missing outcomes remain frozen.
        evidence[symbol] = old+added
        fresh_events[symbol] = added
    corporate_actions = deepcopy(previous.get("corporate_actions") or [])
    known_actions = {a["action_id"]: a for a in corporate_actions}
    for action in account.get("corporate_actions") or []:
        existing = known_actions.get(action["action_id"])
        if existing is not None and _canonical(existing) != _canonical(action):
            raise ValueError("Previously frozen company-action terms changed; create a new branch")
        if existing is None:
            if str(action["record_date"])[:10] <= previous_end:
                raise ValueError("Historical company-action record was added after checkpoint; create a new branch")
            corporate_actions.append(deepcopy(action))
    new_days = [day for day in dates if previous_end < day <= end]
    comparisons = previous["comparisons"]

    def advance(day, checkpoints, *, hard_risk=False, benchmarks=False):
        advanced = {}
        for policy, earlier in checkpoints.items():
            _check(stop)
            if progress:
                progress(stage="account_hard_risk_first" if hard_risk else "account_append",
                         policy=policy, date=day)
            advanced[policy] = run_position_simulation(frames, evidence,
                config=earlier["lineage"]["config"], start=previous["start"], end=day,
                calendar=dates, state=earlier["state"], corporate_actions=corporate_actions,
                check_cancel=lambda:_stopped(stop), include_benchmarks=benchmarks)
        return advanced

    # Verify every policy's complete frozen input prefix before calling a model.
    # A day's preview fills opening orders and applies price/time/portfolio risk;
    # its closing inventory determines whether that day's forecast is needed.
    preview = advance(new_days[0], comparisons, hard_risk=True)
    # Read the exact saved scorer, not whichever model became current meanwhile.
    model_ref = previous.get("score_model")
    model, dependency_errors = None, []
    if model_ref:
        key = model_ref.get("cache_key")
        try:
            if not key: raise ValueError("Saved score model has no immutable object reference")
            service.store.pin(key,"account:"+str(account.get("account_id",previous["result_id"])))
            model = service.store.get_by_key(key)
            if model is None: raise ValueError("Saved score model object is unavailable; no substitute selected")
            expected_identity = (previous.get("identity") or {}).get("score")
            actual_identity = (model.get("artifact") or {}).get("model_version") \
                if isinstance(expected_identity, str) else model.get("identity")
            if expected_identity is not None and actual_identity != expected_identity:
                raise ValueError("Saved scorer identity differs from the account's frozen identity")
        except (ValueError, KeyError) as exc:
            model = None
            dependency_errors.append(dict(dependency="frozen_simple_score", cache_key=key, error=str(exc)))
    parameters = deepcopy(account.get("prediction") or previous["identity"]["prediction"])
    expected_prediction = previous["identity"].get("model")
    prediction_dependency_error = None
    try:
        from my_strategy.services.light_research import prediction_identity
        identity_reader = getattr(service,"prediction_identity",prediction_identity)
        actual_prediction = identity_reader(parameters.get("device","auto"))
        if expected_prediction is None or _canonical(actual_prediction) != _canonical(expected_prediction):
            raise ValueError("Kronos/tokenizer/runtime identity changed or was not frozen; new model requires a new account branch")
    except (ValueError, RuntimeError, OSError) as exc:
        prediction_dependency_error = str(exc)
        dependency_errors.append(dict(dependency="frozen_kronos",error=str(exc)))
    maximum = int(account.get("max_predictions", previous["identity"].get("max_predictions",60)))
    if deadline is None:
        deadline = started+float(account.get("budget_seconds", previous["identity"].get("budget_seconds",300)))
    by_day = {}
    errors = []
    coverage = dict(requested=0, succeeded=0, unusable=0, failed=0, budget_skipped=0)
    from my_strategy.services.simple_score import score_samples
    for symbol, added in fresh_events.items():
        frame = frames[symbol]
        observed_days = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        for event in added:
            day = _decision_day(event,dates)
            by_day.setdefault(day, {}).setdefault(symbol, []).append(event)
            event["forecast"] = None
            score_error = next((e for e in dependency_errors if e["dependency"]=="frozen_simple_score"), None)
            if score_error:
                event["score"] = dict(status="unavailable",reason=score_error["error"])
            if model:
                row = frame.loc[observed_days==day]
                try:
                    if len(row)!=1: raise ValueError("No uniquely observed scoring row")
                    scored = score_samples(model["artifact"],row,as_of=day,historical_reconstruction=True)
                    event["score"] = dict(status="scored",score=scored["scores"][0],
                        probabilities=scored["probabilities"][0],qualification=scored["qualification"],
                        model_version=scored["model_version"])
                except ValueError as exc: event["score"] = dict(status="unavailable",reason=str(exc))
    attempts = 0
    for day_index, day in enumerate(new_days):
        _check(stop)
        if day_index:
            preview = advance(day, comparisons, hard_risk=True)
        held = {symbol for result in preview.values()
                for symbol, position in result["daily"][-1]["positions"].items()
                if position["shares"] > 0 or position.get("receivable_shares", 0) > 0}
        day_events = by_day.get(day, {})
        candidates = {symbol for symbol, entries in day_events.items()
                      if any(e.get("side", "").upper() in {"B", "BUY"}
                             and e.get("confirmed") is True for e in entries)}
        # Deduplicate across policies and source events. Held stocks take the
        # day's remaining budget before new entry candidates. No later day's
        # position is consulted when deciding to observe today's forecast.
        needed = sorted(held) + sorted(candidates-held)
        coverage["requested"] += len(needed)
        for symbol in needed:
            _check(stop)
            if prediction_dependency_error:
                coverage["failed"] += 1
                forecast = dict(status="unavailable",reason=prediction_dependency_error,
                    as_of=day,quality={"prediction_usable":False})
                errors.append(dict(symbol=symbol,date=day,error=prediction_dependency_error))
            elif attempts >= maximum or time.monotonic() >= deadline:
                coverage["budget_skipped"] += 1
                forecast = dict(status="unavailable", reason="continuation_prediction_budget",
                    as_of=day, quality={"prediction_usable":False})
            else:
                attempts += 1
                if progress:
                    progress(stage="account_new_prediction",symbol=symbol,date=day,
                             completed=attempts,total=coverage["requested"])
                try:
                    forecast = service.prediction(symbol,day,stop=adapter,**parameters)
                    coverage["succeeded" if (forecast.get("quality") or {}).get("prediction_usable") is True else "unusable"] += 1
                except InterruptedError: raise
                except (ValueError, RuntimeError, TimeoutError) as exc:
                    coverage["failed"] += 1
                    forecast = dict(status="unavailable",reason=str(exc),as_of=day,
                                    quality={"prediction_usable":False})
                    errors.append(dict(symbol=symbol,date=day,error=str(exc)))
            entries = day_events.get(symbol, [])
            if entries:
                for event in entries: event["forecast"] = forecast
            else:
                # This is an observation, never a fabricated structural B/S.
                observation = dict(date=day, available_at=day+"T15:00:00+08:00",
                    event_id=f"holding-forecast:{symbol}:{day}", confirmed=False,
                    observation_kind=HOLDING_OBSERVATION, forecast=forecast)
                evidence[symbol].append(observation)
                fresh_events[symbol].append(observation)
        # Commit from the untouched day-start checkpoint with the observations
        # just made. Preview orders are not published or executed a second time.
        comparisons = advance(day, comparisons, benchmarks=day == end)
    identity = {**deepcopy(previous["identity"]), "kind":"simulation-continuation-v1",
        "continuation_version":VERSION, "sources":identities, "calendar":calendar_identity, "end":end,
        "parent_result_id":previous["result_id"],
        "frozen_prefixes":{policy:r["state"]["input_prefix_hash"] for policy,r in previous["comparisons"].items()},
        "new_evidence_sha256":_hash(fresh_events),
        "input_prefixes":{policy:r["state"]["input_prefix_hash"] for policy,r in comparisons.items()}}
    result_id = service.store.cache_key(identity,"simulations")
    chart_bars = {}
    for symbol, frame in frames.items():
        date_column = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
        rows = frame.loc[(date_column>=previous["start"]) & (date_column<=end),
                         ["date","open","high","low","close"]].copy()
        rows["date"] = pd.to_datetime(rows.date).dt.strftime("%Y-%m-%d")
        chart_bars[symbol] = rows.to_dict("records")
    from my_strategy.services.light_research import plain
    partial = bool(dependency_errors or any(coverage[key] for key in ("failed", "unusable", "budget_skipped")))
    result = {**previous, "status":"partial" if partial else "completed",
        "result_id":result_id, "end":end, "identity":identity,
        "comparisons":comparisons, "chart_bars":chart_bars, "source_events":evidence,
        "prediction_coverage":coverage, "prediction_errors":errors, "saved":True, "reused":False,
        "parent_result_id":previous["result_id"], "corporate_actions":corporate_actions,
        "dependency_errors":dependency_errors,
        "continuation":dict(from_date=previous_end,through_date=end,
            previous_prediction_results_frozen=True,hard_risk_processed_before_new_prediction=True,
            state_resumed=True, prediction_coverage_scope="new_sessions_only",
            prediction_schedule="daily_close_actual_holdings_then_confirmed_buy_candidates")}
    if sum(coverage[k] for k in ("succeeded","unusable","failed","budget_skipped")) != coverage["requested"]:
        raise AssertionError("Continuation prediction coverage does not reconcile")
    return plain(result)


def _continue_agent_account(service,account,previous_result,end,*,stop=None,deadline=None,progress=None):
    """The v1 Agent account shares exactly the initial replay's daily scheduler."""
    from my_strategy.services.light_daily_scheduler import run_daily_schedule,OBSERVATION
    from my_strategy.services.agent_evidence import volume_candidates
    from my_strategy.services.light_research import plain,prediction_identity
    previous=deepcopy(previous_result)
    prior_schedule=previous.get('scheduler') or {}
    evaluation_accounts=(prior_schedule.get('pending') or {}).get('day_start',previous['comparisons'])
    before=min((v['state']['last_date'] for v in evaluation_accounts.values()),
        default=(pd.Timestamp(previous['start'])-pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    end=str(end)[:10]
    if end<before:raise ValueError('Cannot rewind a saved account')
    if end==before and not (previous.get('scheduler') or {}).get('pending'):return previous
    if account.get('last_result_id') and account['last_result_id']!=previous['result_id']:
        raise ValueError('Account pointer changed')
    dates,cal=service.calendar()
    if end not in dates:raise ValueError('Continuation date is not verified')
    adapter=_StopAdapter(stop);frames={};sources=[];evidence={};score_source_events={}
    for symbol in previous['symbols']:
        _check(stop)
        _,frame,events,source=service.source(symbol,end,stop=adapter)
        if 'symbol' not in frame:frame=frame.assign(symbol=symbol)
        score_source_events[symbol]=events
        frames[symbol]=frame;sources.append(source)
        fresh=[{**e} for e in events+volume_candidates(frame,symbol)
               if previous['start']<=str(e['date'])[:10]<=end]
        known=[e for e in previous['source_events'][symbol] if e.get('observation_kind')!=OBSERVATION
               and (_decision_day(e,dates) or '9999')<=before]
        prior=[e for e in fresh if (_decision_day(e,dates) or '9999')<=before]
        sort=lambda e:(str(e.get('event_id')),str(e.get('available_at')))
        if _canonical(sorted(known,key=sort))!=_canonical(sorted(prior,key=sort)):
            raise ValueError('Historical confirmed evidence changed; create a new branch')
        evidence[symbol]=deepcopy(previous['source_events'][symbol])
        known_ids={e.get('event_id') for e in evidence[symbol]}
        evidence[symbol].extend(e for e in fresh if e.get('event_id') not in known_ids and (_decision_day(e,dates) or '0000')>before)
    model_ref=previous.get('score_model');model=None;dependency_errors=[]
    if model_ref:
        try:
            model=service.score_model(model_ref['cache_key'])
            if model['artifact']['model_version']!=previous['identity'].get('score'):
                raise ValueError('Saved scorer identity changed; no substitute selected')
        except (ValueError,KeyError) as exc:
            model=None;dependency_errors.append({'dependency':'frozen_score','error':str(exc)})
    parameters=previous['identity']['prediction']
    identity_reader=getattr(service,'prediction_identity',prediction_identity)
    prediction_error=None
    try:
        if _canonical(identity_reader(parameters.get('device','auto')))!=_canonical(previous['identity'].get('model')):
            raise ValueError('Frozen Kronos identity changed; no substitute selected')
    except (ValueError,RuntimeError,OSError) as exc:
        prediction_error=str(exc);dependency_errors.append({'dependency':'frozen_kronos','error':str(exc)})
    configs={policy:result['lineage']['config'] for policy,result in previous['comparisons'].items()}
    # A model can be historically evaluated without being automatically promoted.
    # A forward branch must have been explicitly created in paper mode.
    mode=previous['identity'].get('execution_mode','historical')
    research_mode=next(iter(configs.values())).get('research_mode','historical_candidate')
    scorer=service.scorer(model,frames,evidence,score_source_events=score_source_events,research_mode=research_mode,
        qualification_event=previous['identity'].get('qualification_event'))
    checkpoint={**deepcopy(prior_schedule),'comparisons':evaluation_accounts,'evidence':evidence,
        'errors':previous.get('prediction_errors',[])}
    checkpoint_identity={'account_id':account.get('account_id',previous['result_id']),
                         'parent_result':previous['result_id'],'end':end}
    saved=service._read_checkpoint(checkpoint_identity,'account_checkpoints') if hasattr(service,'_read_checkpoint') else None
    if saved and saved.get('stage') in {'cancelled','interrupted','day_committed'}:checkpoint=saved
    def save_checkpoint(value):
        if value.get('stage')!='observing' and hasattr(service,'_write_checkpoint'):
            service._write_checkpoint(checkpoint_identity,plain(value),'account_checkpoints')
    deadline=deadline or time.monotonic()+previous['identity']['budget_seconds']
    from my_strategy.services.light_research import DeadlineStop
    prediction_stop=DeadlineStop(adapter,deadline)
    def predict(symbol,day):
        if prediction_error:raise ValueError(prediction_error)
        return service.prediction(symbol,day,stop=prediction_stop,**parameters)
    scheduled=run_daily_schedule(frames,evidence,configs=configs,calendar=dates,start=previous['start'],end=end,
        comparisons=evaluation_accounts,checkpoint=checkpoint,
        prediction=predict,score=scorer,
        max_predictions_per_day=previous['identity']['max_predictions'],
        deadline=deadline,stop=stop,progress=progress,
        checkpoint_callback=save_checkpoint,corporate_actions=previous.get('corporate_actions'),execution_mode=mode)
    identity={**deepcopy(previous['identity']),'kind':'agent-simulation-continuation-v1','sources':sources,
        'calendar':cal,'end':end,'parent_result_id':previous['result_id'],
        'observation_scope':_hash(scheduled['evidence']),
        'input_prefixes':{k:v['state']['input_prefix_hash'] for k,v in scheduled['comparisons'].items()}}
    result_id=service.store.cache_key(identity,'simulations')
    chart={}
    for symbol,frame in frames.items():
        days=pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
        rows=frame.loc[(days>=previous['start'])&(days<=end),['date','open','high','low','close']].copy()
        rows['date']=pd.to_datetime(rows.date).dt.strftime('%Y-%m-%d');chart[symbol]=rows.to_dict('records')
    result=plain({**previous,'status':'partial' if dependency_errors else scheduled['status'],'result_id':result_id,'identity':identity,'end':end,
        'account_role':account.get('role',account.get('continuation_mode','historical_candidate')),
        'dependency_errors':dependency_errors,'risk_comparisons':scheduled.get('risk_comparisons'),
        'comparison_scope':scheduled.get('comparison_scope','completed_evaluation'),
        'parent_result_id':previous['result_id'],'comparisons':scheduled['comparisons'],
        'source_events':scheduled['evidence'],'prediction_coverage':scheduled['coverage'],
        'prediction_errors':scheduled['errors'],'scheduler':scheduled['scheduler'],'chart_bars':chart,
        'diagnostics':{'opportunity_funnel':{k:v['opportunity_funnel'] for k,v in scheduled['comparisons'].items()},
                       'score_model_missing':model is None},'saved':True,'reused':False,
        'continuation':{'from_date':before,'through_date':end,'state_resumed':True,
            'prediction_schedule':'causal_daily_scheduler_v1','previous_prediction_results_frozen':True,
            'hard_risk_processed_before_new_prediction':True,'prediction_coverage_scope':'whole_account',
            'role':account.get('role',account.get('continuation_mode','historical_candidate'))}})
    from my_strategy.services.light_experiment_report import build_experiment_report
    result['experiment_report']=build_experiment_report(result)
    return result
