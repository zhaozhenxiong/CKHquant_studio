"""Actual accepted plans pin exits even after reserved sources are filtered."""
import copy

import pandas as pd
import pytest

from my_strategy.services import czsc_trend_v2_pipeline as pipeline
from my_strategy.services.czsc_trend_v2_contracts import strategy_config


def fixture(monkeypatch,tmp_path,signal='2025-04-01'):
    cfg=strategy_config();cfg.update(E0_policy_hash='e0',E1_policy_hash='e1',candidate_policy_hash='c')
    dates=['2025-04-02','2025-04-04','2025-04-07']
    frame=pd.DataFrame({'date':dates,'symbol':['TEST.SZ']*3,'close':[10.]*3,'model_input_eligible':[True]*3})
    monkeypatch.setattr(pipeline,'bind_trend_inputs',lambda frame,calendar:frame.copy())
    monkeypatch.setattr(pipeline,'holding_feature_row',lambda feature,state,**kwargs:{'actual_shares':state['shares']})
    bundles=[{'name':name,'available_at':available,'heads':{'ma_exit':{'path':name}}}
        for name,available in [('prior','2025-03-31'),('newer','2025-04-03')]]
    manifests={bundle['name']:{'head':'ma_exit','manifest_sha256':bundle['name'],
        'target_contract_hash':'target','feature_schema_hash':'schema','train_label_end':'2025-02-28',
        'calibration_label_end':'2025-03-15','policy_label_end':bundle['available_at'],
        'fold':{'policy_end':bundle['available_at']},'target_contract':{'stage':'E0','target':'actual_held',
            'source_snapshot_hash':'source','split_policy':cfg['holding_split_policy'],
            'economic_weighting':cfg['holding_economic_weighting']}}
        for bundle in bundles}
    monkeypatch.setattr(pipeline,'load_head',lambda path:manifests[path.name])
    calls=[]
    def predict(values,root,bundle,*args,**kwargs):
        calls.append((bundle['name'],values['actual_shares']))
        return .7,manifests[bundle['name']]
    monkeypatch.setattr(pipeline,'_predict_interval',predict)
    # The actual B-only plan is absent from the unfiltered A+B preparation.
    prepared={'config_hash':'prepared','rows':[{'date':signal,'entry_plan':{'plan_id':'unfiltered-A+B'}}]}
    plan={'plan_id':'accepted-B-only','symbol':'TEST.SZ','signal_date':signal,
        'available_at':signal+'T15:00:00+08:00','stop_price':9.,'original_opportunity_ids':['B'],
        'holding_boundaries':[{'first_available_at':'2025-03-01T15:00:00+08:00'}]}
    snapshot={'plan_id':plan['plan_id'],'symbol':'TEST.SZ','date':dates[0],
        'entry_date':dates[0],'entry_signal_date':signal,'entry_plan':plan,'shares':1000,
        'holding_prefix':{'peak':10.,'trough':10.},'exit_model_binding':None}
    callback=pipeline.pinned_exit_policy(frame,dates,prepared,tmp_path,bundles,cfg)
    return callback,snapshot,calls,bundles


def test_filtered_actual_plan_is_pinned_at_accepted_signal_and_survives_partial_clone(monkeypatch,tmp_path):
    callback,snapshot,calls,_=fixture(monkeypatch,tmp_path)
    first=callback(snapshot)
    binding=first['entry_exit_binding']
    assert binding['plan_id']=='accepted-B-only' and binding['candidate_signal_date']=='2025-04-01'
    assert binding['checkpoint']=='prior' and binding['available_at']=='2025-03-31'
    assert binding['binding_source']=='actual_Broker_accepted_entry_plan'
    later={**copy.deepcopy(snapshot),'date':'2025-04-04','exit_model_binding':binding,'shares':800}
    assert callback(later)['entry_exit_binding']==binding
    cloned={**later,'date':'2025-04-07','shares':300}
    assert callback.new_replay()(cloned)['entry_exit_binding']==binding
    assert calls==[('prior',1000),('prior',800),('prior',300)]


def test_no_prior_exit_head_remains_rule_fallback_for_entire_holding_and_clone(monkeypatch,tmp_path):
    callback,snapshot,calls,_=fixture(monkeypatch,tmp_path,signal='2025-03-28')
    first=callback(snapshot);binding=first['entry_exit_binding']
    assert binding['checkpoint'] is None and first['probability'] is None and not first['apply']
    assert binding['mode']=='explicit_rule_E0_no_prior_exit_head'
    later={**copy.deepcopy(snapshot),'date':'2025-04-07','exit_model_binding':binding,'shares':500}
    assert callback.new_replay()(later)['entry_exit_binding']==binding
    assert calls==[]


@pytest.mark.parametrize('field,value',[('plan_id','other-plan'),('candidate_signal_date','2025-04-04'),
    ('checkpoint','newer'),('available_at','2025-04-03'),('mode','upgrade'),('source','other')])
def test_saved_actual_exit_binding_cannot_change_any_frozen_anchor(monkeypatch,tmp_path,field,value):
    callback,snapshot,calls,_=fixture(monkeypatch,tmp_path)
    binding=callback(snapshot)['entry_exit_binding'];binding[field]=value
    later={**copy.deepcopy(snapshot),'date':'2025-04-04','exit_model_binding':binding}
    with pytest.raises(ValueError,match='original exit binding'):
        callback.new_replay()(later)


def test_snapshot_must_retain_exact_actual_entry_plan_and_signal(monkeypatch,tmp_path):
    callback,snapshot,calls,_=fixture(monkeypatch,tmp_path)
    callback(snapshot)
    later={**copy.deepcopy(snapshot),'date':'2025-04-04'}
    later['entry_plan']['stop_price']=8.
    with pytest.raises(ValueError,match='original accepted plan'):
        callback.new_replay()(later)
    later=copy.deepcopy(snapshot);later['entry_plan']['signal_date']='2025-03-28'
    with pytest.raises(ValueError,match='original signal/plan identity'):
        callback.new_replay()(later)
