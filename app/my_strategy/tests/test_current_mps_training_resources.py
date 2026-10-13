"""Real MPS residency/bounded-transfer parity with strict default fit constraints.

Synthetic rare-class data exercises training mechanics, not Broker-derived
8%/2R qualification. The forced1byte budget is private test instrumentation;
512MiB describes X/y/weight residency, not a total GPU-memory guarantee.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_trend_interval_model as interval
from my_strategy.services.czsc_trend_contracts import (
    strategy_config, TRAINING_POPULATION, SPLIT_POLICY, ECONOMIC_WEIGHTING,
)
from my_strategy.services.czsc_trend_features import ENTRY_FEATURE_COLUMNS
from my_strategy.services.czsc_trend_labels import target_contract


def sparse_entry_data():
    dates=pd.bdate_range('2023-01-02',periods=350).strftime('%Y-%m-%d').tolist()
    ids=np.arange(len(dates)*32)
    frame=pd.DataFrame({column:np.sin(ids/(13+j%11)+j)+np.cos(ids/(37+j%7))*.2
                        for j,column in enumerate(ENTRY_FEATURE_COLUMNS)})
    days=np.repeat(dates,32)
    frame=frame.assign(date=days,symbol=[f'SYNTH:{i%32}' for i in ids],
        label=(ids%53==0).astype(int),
        label_end=(pd.to_datetime(days)+pd.offsets.BDay(1)).strftime('%Y-%m-%d'),
        label_available=True,input_eligible=True,
        attempt_id=[f'attempt:{i}' for i in ids],economic_group_id=[f'original:{i}' for i in ids])
    frame['original_opportunity_ids']=[[f'structure:original:{i}:BUY'] for i in ids]
    frame['original_event_ids']=frame.original_opportunity_ids
    frame['original_event_first_pre_gate_dates']=[json.dumps({atoms[0]:day}) for atoms,day in zip(frame.original_event_ids,days)]
    fold=dict(name='resource_test',train_end=dates[199],calibration_start=dates[200],calibration_end=dates[249],
              policy_start=dates[250],policy_end=dates[299],validation_end=dates[299])
    config=strategy_config()
    config['reference_population']=dict(version=TRAINING_POPULATION,fold=fold,population_end=fold['policy_end'],
        prior_entry_catalog=[],prior_entry_catalog_hash=stable_hash([]),split_policy=SPLIT_POLICY,economic_weighting=ECONOMIC_WEIGHTING)
    contract=target_contract(head='entry',config=config,market_dates=dates,source_snapshot_hash='synthetic-test-not-Broker-qualification',
        feature_schema_hash=interval.schema_hash(ENTRY_FEATURE_COLUMNS,'entry'),continuation_policy_hash='synthetic-E1',
        initial_cash=config['initial_cash'],stage='E1',candidate_policy_hash='synthetic-candidate')
    contract={k:v for k,v in {**contract,'model_input_schema_hash':interval.schema_hash(ENTRY_FEATURE_COLUMNS,'entry')}.items() if k!='contract_hash'}
    contract['contract_hash']=stable_hash(contract);frame.attrs['label_contract']=contract
    return frame,fold,config,contract


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual MPS unavailable; no CPU substitute or fallback')
@pytest.mark.parametrize('architecture',['linear','mlp'])
def test_real_mps_resident_and_bounded_transfer_are_exactly_equivalent(tmp_path,monkeypatch,architecture):
    frame,fold,config,contract=sparse_entry_data()
    assert (config['epochs'],config['batch_size'],config['min_train_rows'],config['min_calibration_rows'])==(20,2048,100,30)
    assert config['high_return']['min_net_return']==.08 and config['high_return']['min_r_multiple']==2
    masks=interval.time_masks(frame,fold,head='entry')
    assert masks[0].sum()>=100 and masks[1].sum()>=30
    for mask in masks:
        assert set(frame.loc[mask,'label'])=={0,1}
    assert int(frame.loc[masks[0],'label'].sum())<int(masks[0].sum())/30
    default_budget=interval._DEVICE_RESIDENT_BUDGET_BYTES
    assert default_budget==512*1024**2
    randperm=torch.randperm
    orders=[]
    def traced_order(*args,**kwargs):
        values=randperm(*args,**kwargs)
        assert values.device.type=='cpu'
        orders.append(hashlib.sha256(values.numpy().tobytes()).hexdigest())
        return values
    monkeypatch.setattr(torch,'randperm',traced_order)
    results=[]
    for name,budget in [('resident',default_budget),('bounded',1)]:
        monkeypatch.setattr(interval,'_DEVICE_RESIDENT_BUDGET_BYTES',budget)
        orders.clear();output=tmp_path/name
        manifest=interval.train_head(frame,ENTRY_FEATURE_COLUMNS,output,fold,config,
            head='entry',architecture=architecture,device='mps',contract=contract)
        torch.mps.synchronize()
        assert manifest['actual_device']=='mps' and manifest['actual_mps_training'] is True
        assert manifest['requested_device']=='mps' and manifest['positive_weight']>30
        assert manifest['publication_allowed'] is False
        loaded=interval.load_head(output)
        state=torch.load(output/loaded['checkpoint'],map_location='cpu',weights_only=True)['state_dict']
        policy=interval.bind_inputs(frame.loc[(frame.date>=fold['policy_start']) & (frame.date<=fold['policy_end'])].iloc[:256],ENTRY_FEATURE_COLUMNS,'entry')
        prediction=interval.HeadSession().predict(policy,output,validation=True,expected_head='entry',expected_contract=contract)
        results.append((manifest,state,list(orders),prediction))
    resident,bounded=results
    assert len(resident[2])==20 and resident[2]==bounded[2]
    assert all(torch.equal(resident[1][key],bounded[1][key]) for key in resident[1])
    assert resident[0]['training_loss']==bounded[0]['training_loss']
    assert resident[0]['calibration']==bounded[0]['calibration']
    np.testing.assert_array_equal(resident[3],bounded[3])
    excluded={'manifest_sha256','training_started_at','training_completed_at','training_resources'}
    assert {k:v for k,v in resident[0].items() if k not in excluded}=={k:v for k,v in bounded[0].items() if k not in excluded}
    for manifest,_,_,_ in results:
        resources=manifest['training_resources']
        assert resources['train_tensor_bytes']==int(masks[0].sum())*(len(ENTRY_FEATURE_COLUMNS)+2)*4
        assert resources['permutation_generator']=='cpu_seeded_same_order'
    assert resident[0]['training_resources']['input_staging']=='device_resident'
    assert bounded[0]['training_resources']['input_staging']=='bounded_batch_transfer'
    assert bounded[0]['training_resources']['device_resident_budget_bytes']==1
