import copy
import hashlib
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.light_account_continuation import continue_saved_account
from my_strategy.services.position_simulation import run_position_simulation

SYMBOL = "002436.SZ"


class Store:
    def __init__(self): self.objects = {}; self.reads=[]
    def cache_key(self, identity, namespace):
        return hashlib.sha256(json.dumps([namespace,identity],sort_keys=True,default=str).encode()).hexdigest()
    def get_by_key(self,key):
        self.reads.append(key)
        return self.objects.get(key)
    def pin(self,key,reason): return None


class Service:
    def __init__(self, frame, events):
        self.frame,self.events=frame,events
        self.store=Store();self.calls=[];self.source_calls=0
    def calendar(self): return self.frame.date.tolist(), {"sha256":"frozen-test-calendar"}
    def prediction_identity(self,device): return {"model":"frozen-fixture"}
    def source(self,symbol,end,stop=None):
        self.source_calls += 1
        return self.frame.copy(),self.frame.copy(),copy.deepcopy(self.events),{"symbol":symbol,"through":end}
    def prediction(self,symbol,day,stop=None,**parameters):
        self.calls.append((symbol,day))
        if stop and stop.is_set(): raise InterruptedError("cancelled")
        return dict(as_of=day,quality={"prediction_usable":True},
                    summary={"terminal_return_median":.02,"up_path_fraction":.7},
                    identity={"frozen_model":"test"})


def setup():
    dates=pd.bdate_range("2025-01-01",periods=40).strftime("%Y-%m-%d").tolist()
    prices=10+np.arange(40)*.01
    frame=pd.DataFrame(dict(date=dates,open=prices,high=prices+.2,low=prices-.2,close=prices,
                            volume=10_000_000,amount=100_000_000,has_trade_price=1))
    def event(i):return dict(date=dates[i],available_at=dates[i]+"T15:00:00+08:00",
        event_id=f"B-{i}",side="B",confirmed=True,risk_boundary=float(prices[i]*.95))
    source=[event(5),event(23)]
    frozen=[{**source[0],"forecast":None,"forecast_status":"budget_skipped"}]
    service=Service(frame,source)
    cfg=dict(ma_period=3,atr_period=3,commission=0,min_commission=0,slippage=0,
             transfer_fee=0,max_holding_sessions=200,time_exit_sessions=200)
    comparisons={policy:run_position_simulation({SYMBOL:frame},{SYMBOL:frozen},
        config={**cfg,"policy":policy},calendar=dates,end=dates[20])
        for policy in ("structure_rules","structure_kronos","structure_kronos_score")}
    identity={"kind":"simulation-v1","sources":[],"start":dates[0],"end":dates[20],
        "prediction":{"samples":8},"max_predictions":60,"budget_seconds":300,"score":None,
        "model":{"model":"frozen-fixture"}}
    previous=dict(result_id="parent",start=dates[0],end=dates[20],symbols=[SYMBOL],
        source_events={SYMBOL:frozen},comparisons=comparisons,identity=identity,score_model=None,
        chart_bars={},created_at="frozen-original-created-at",saved=True)
    account=dict(account_id="parent",last_result_id="parent",symbols=[SYMBOL],
        start=dates[0],prediction={"samples":8},max_predictions=60,budget_seconds=300)
    return dates,service,previous,account,cfg


def test_only_new_predictions_and_old_missing_forecasts_stay_frozen():
    dates,service,previous,account,cfg=setup()
    successor=continue_saved_account(service,account,previous,dates[-1])
    assert service.calls == [(SYMBOL,day) for day in dates[21:]]
    assert successor["source_events"][SYMBOL][0] == previous["source_events"][SYMBOL][0]
    assert successor["parent_result_id"] == "parent"
    for policy,before in previous["comparisons"].items():
        after=successor["comparisons"][policy]
        assert after["daily"][:21] == before["daily"]
        assert after["trades"][:len(before["trades"])] == before["trades"]
        full=run_position_simulation({SYMBOL:service.frame},successor["source_events"],
            config=before["lineage"]["config"],calendar=dates)
        for key in ("daily","trades","orders","decisions","metrics","state"):
            assert after[key] == full[key], (policy,key)
    assert successor["prediction_coverage"] == dict(requested=19,succeeded=19,unusable=0,failed=0,budget_skipped=0)
    assert successor["status"] == "completed"
    # Daily held-position predictions are observations, never invented source signals.
    native=[e for e in successor["source_events"][SYMBOL] if e.get("confirmed")]
    assert [e["event_id"] for e in native] == ["B-5","B-23"]
    observed=[e for e in successor["source_events"][SYMBOL] if e.get("observation_kind")]
    assert len(observed) == 18 and all("side" not in e and not e["confirmed"] for e in observed)
    assert not previous["source_events"][SYMBOL][0]["forecast"]


def test_same_day_is_exact_no_io_no_new_result_identity():
    dates,service,previous,account,cfg=setup()
    result=continue_saved_account(service,account,previous,previous["end"])
    assert result == previous
    assert service.source_calls == 0 and service.calls == []


def test_changed_historical_price_or_confirmed_event_rejected_without_prediction():
    dates,service,previous,account,cfg=setup()
    service.frame.loc[8,"close"] += .01
    with pytest.raises(ValueError,match="prefix changed"):
        continue_saved_account(service,account,previous,dates[-1])
    assert not service.calls
    dates,service,previous,account,cfg=setup()
    service.events[0]["risk_boundary"] += .01
    with pytest.raises(ValueError,match="confirmed evidence changed"):
        continue_saved_account(service,account,previous,dates[-1])
    assert not service.calls


def test_exhausted_prediction_budget_keeps_price_stop_processing():
    dates,service,previous,account,cfg=setup()
    service.frame.loc[21:, ["open","close"]] = 9.4
    service.frame.loc[21:, "high"] = 9.6
    service.frame.loc[21:, "low"] = 9.2
    result=continue_saved_account(service,account,previous,dates[-1],deadline=time.monotonic()-1)
    assert not service.calls
    assert result["prediction_coverage"]["budget_skipped"] == 1
    assert result["status"] == "partial"
    sales=[t for t in result["comparisons"]["structure_rules"]["trades"] if t["action"]=="SELL"]
    assert sales and sales[-1]["date"] == dates[21]
    assert result["continuation"]["hard_risk_processed_before_new_prediction"] is True


def test_cancellation_keeps_previous_result_unmodified_for_event_and_callable():
    dates,service,previous,account,cfg=setup()
    original=copy.deepcopy(previous)
    stop=threading.Event();stop.set()
    for cancellation in (stop,lambda:True):
        with pytest.raises(InterruptedError):
            continue_saved_account(service,account,previous,dates[-1],stop=cancellation)
    assert previous == original and not service.calls and service.source_calls==0


def test_missing_saved_model_does_not_use_current_replacement():
    dates,service,previous,account,cfg=setup()
    previous["score_model"]={"cache_key":"old-scorer"}
    result=continue_saved_account(service,account,previous,dates[-1],deadline=time.monotonic()-1)
    assert service.store.reads == ["old-scorer"]
    assert result["status"] == "partial"
    assert "unavailable" in result["dependency_errors"][0]["error"]
    assert result["comparisons"]["structure_kronos_score"]["daily"][-1]["date"] == dates[-1]
    event=next(e for e in result["source_events"][SYMBOL] if e["event_id"]=="B-23")
    assert event["score"]["status"] == "unavailable"
    assert not service.calls


def test_new_backdated_company_action_requires_branch():
    dates,service,previous,account,cfg=setup()
    account["corporate_actions"]=[dict(action_id="late",symbol=SYMBOL,source_id="test",record_date=dates[5],
        ex_date=dates[6],pay_date=dates[7],cash_per_share=1,tax_per_share=0)]
    with pytest.raises(ValueError,match="Historical company-action"):
        continue_saved_account(service,account,previous,dates[-1])


def test_changed_kronos_prevents_new_predictions_but_preserves_hard_risk():
    dates,service,previous,account,cfg=setup()
    service.prediction_identity=lambda device:{"model":"replacement"}
    service.frame.loc[21:, ["open","close"]] = 9.4
    service.frame.loc[21:, "high"] = 9.6
    service.frame.loc[21:, "low"] = 9.2
    result=continue_saved_account(service,account,previous,dates[-1])
    assert result["status"] == "partial"
    assert not service.calls
    assert result["dependency_errors"][0]["dependency"] == "frozen_kronos"
    assert result["prediction_coverage"]["failed"] == 1
    assert any(t["action"]=="SELL" for t in result["comparisons"]["structure_rules"]["trades"])
    assert result["source_events"][SYMBOL][0] == previous["source_events"][SYMBOL][0]


@pytest.mark.parametrize("outcome", ["failed","unusable","budget_skipped"])
def test_incomplete_prediction_outcomes_are_partial_and_counted(outcome):
    dates,service,previous,account,cfg=setup()
    def failed(symbol,day,**kwargs):
        service.calls.append((symbol,day))
        raise RuntimeError("declared inference failure")
    def unusable(symbol,day,**kwargs):
        service.calls.append((symbol,day))
        return dict(as_of=day,quality={"prediction_usable":False},summary={})
    if outcome=="failed": service.prediction=failed
    elif outcome=="unusable": service.prediction=unusable
    else: account["max_predictions"]=0
    original=copy.deepcopy(previous)
    result=continue_saved_account(service,account,previous,dates[25])
    assert result["status"] == "partial"
    coverage=result["prediction_coverage"]
    assert coverage["requested"] == coverage[outcome] == 5
    assert coverage["requested"] == sum(coverage[k] for k in ("succeeded","failed","unusable","budget_skipped"))
    assert len(service.calls) == (0 if outcome=="budget_skipped" else 5)
    assert previous == original
    assert all(r["daily"][-1]["date"]==dates[25] for r in result["comparisons"].values())
    observations=[e for e in result["source_events"][SYMBOL] if e.get("observation_kind")]
    assert len(observations)==4 and all(e["forecast"]["quality"]["prediction_usable"] is False for e in observations)


def test_second_continuation_keeps_observed_forecasts_and_decisions_frozen():
    dates,service,previous,account,cfg=setup()
    first=continue_saved_account(service,account,previous,dates[25])
    first_copy=copy.deepcopy(first)
    account["last_result_id"]=first["result_id"]
    service.calls=[]
    second=continue_saved_account(service,account,first,dates[29])
    assert service.calls == [(SYMBOL,day) for day in dates[26:30]]
    assert second["source_events"][SYMBOL][:len(first["source_events"][SYMBOL])] == first["source_events"][SYMBOL]
    assert first == first_copy
    for policy,before in first["comparisons"].items():
        after=second["comparisons"][policy]
        assert after["daily"][:len(before["daily"])] == before["daily"]
        full=run_position_simulation({SYMBOL:service.frame},second["source_events"],
            config=before["lineage"]["config"],calendar=dates,end=dates[29])
        for key in ("daily","trades","orders","decisions","metrics","state"):
            assert after[key] == full[key], (policy,key)


def test_changed_old_forecast_observation_rejected_without_new_prediction():
    dates,service,previous,account,cfg=setup()
    first=continue_saved_account(service,account,previous,dates[25])
    account["last_result_id"]=first["result_id"]
    observation=next(e for e in first["source_events"][SYMBOL] if e.get("observation_kind"))
    observation["forecast"]["summary"]["terminal_return_median"] = -.99
    service.calls=[]
    with pytest.raises(ValueError,match="prefix changed"):
        continue_saved_account(service,account,first,dates[29])
    assert not service.calls


def test_actual_held_daily_weak_forecasts_reduce_next_open_without_new_source_signal():
    dates,service,previous,account,cfg=setup()
    previous["source_events"][SYMBOL][0]["forecast"] = dict(quality={"prediction_usable":True},
        summary={"terminal_return_median":.02,"up_path_fraction":.7})
    for policy in previous["comparisons"]:
        previous["comparisons"][policy]=run_position_simulation({SYMBOL:service.frame},previous["source_events"],
            config={**cfg,"policy":policy},calendar=dates,end=dates[20])
    def weak(symbol,day,**kwargs):
        service.calls.append((symbol,day))
        return dict(as_of=day,quality={"prediction_usable":True},
            summary={"terminal_return_median":-.02,"up_path_fraction":.3})
    service.prediction=weak
    result=continue_saved_account(service,account,previous,dates[25])
    assert service.calls == [(SYMBOL,day) for day in dates[21:26]]
    after=result["comparisons"]["structure_kronos"]
    reductions=[o for o in after["orders"] if o["reason"]=="consecutive_forecast_weakness"]
    assert len(reductions)==1 and reductions[0]["signal_date"]==dates[22]
    fills=[t for t in after["trades"] if t["action"]=="REDUCE"]
    assert len(fills)==1 and fills[0]["date"]==dates[23]
    assert after["daily"][:21] == previous["comparisons"]["structure_kronos"]["daily"]
    assert after["daily"][23]["positions"][SYMBOL]["shares"] < after["daily"][21]["positions"][SYMBOL]["shares"]
    assert not any(e.get("side")=="S" for e in result["source_events"][SYMBOL])


def test_held_stock_has_daily_prediction_priority_over_new_buy_candidate():
    dates,service,previous,account,cfg=setup()
    candidate="000001.SZ"  # Sorts before the holding, so a simple symbol sort is insufficient.
    frames={SYMBOL:service.frame,candidate:service.frame.copy()}
    previous["symbols"]=[candidate,SYMBOL]
    previous["source_events"][candidate]=[]
    for policy in previous["comparisons"]:
        previous["comparisons"][policy]=run_position_simulation(frames,previous["source_events"],
            config={**cfg,"policy":policy},calendar=dates,end=dates[20])
    original_source=service.source
    candidate_event={**service.events[1],"date":dates[21],"available_at":dates[21]+"T15:00:00+08:00","event_id":"candidate-B"}
    def source(symbol,end,stop=None):
        if symbol==SYMBOL:return original_source(symbol,end,stop=stop)
        return frames[symbol],frames[symbol],copy.deepcopy([candidate_event]),{"symbol":symbol,"through":end}
    service.source=source
    account.update(symbols=[candidate,SYMBOL],max_predictions=1)
    result=continue_saved_account(service,account,previous,dates[21])
    assert service.calls == [(SYMBOL,dates[21])]
    assert result["prediction_coverage"] == dict(requested=2,succeeded=1,unusable=0,failed=0,budget_skipped=1)
    assert result["status"]=="partial"
    assert result["source_events"][candidate][0]["forecast"]["reason"]=="continuation_prediction_budget"


def test_saved_score_model_version_matches_artifact_and_never_current_pointer(monkeypatch):
    dates,service,previous,account,cfg=setup()
    previous["score_model"]={"cache_key":"frozen-score"}
    previous["identity"]["score"]="frozen-score-v1"
    service.store.objects["frozen-score"]={"identity":{"kind":"training-input"},
        "artifact":{"model_version":"frozen-score-v1"}}
    seen=[]
    def score(artifact,row,**kwargs):
        seen.append((artifact["model_version"],kwargs["as_of"]))
        return dict(scores=[.2],probabilities=[[.6,.3,.1]],
            qualification={"positive_allocation_eligible":False},model_version=artifact["model_version"])
    monkeypatch.setattr("my_strategy.services.simple_score.score_samples",score)
    result=continue_saved_account(service,account,previous,dates[24])
    assert service.store.reads == ["frozen-score"]
    assert seen == [("frozen-score-v1",dates[23])]
    assert not result["dependency_errors"]
    event=next(e for e in result["source_events"][SYMBOL] if e["event_id"]=="B-23")
    assert event["score"]["status"]=="scored" and event["score"]["model_version"]=="frozen-score-v1"
