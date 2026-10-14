"""Separate E1 held targets from blocked strict-prior entry populations."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_contracts import strategy_config, validate_holding_population
from my_strategy.services.czsc_trend_dataset import training_frame
from my_strategy.services.czsc_trend_features import EXIT_FEATURE_COLUMNS
from my_strategy.services.czsc_trend_interval_model import train_head, load_head, time_masks
from my_strategy.services.czsc_trend_pipeline import (
    freeze_holding_population, audit_holding_population, holding_population_diagnostics, holding_head_status, _file_hash)


def actual_population(tmp_path):
    dates=pd.bdate_range('2024-01-02',periods=320).strftime('%Y-%m-%d').tolist()
    values=pd.DataFrame({column:np.sin(np.arange(320)/17+i) for i,column in enumerate(EXIT_FEATURE_COLUMNS)})
    values=values.assign(symbol='TEST.SZ',date=dates,label=np.arange(320)%2,
        label_available=True,input_eligible=True,label_end=(pd.to_datetime(dates)+pd.offsets.BDay(1)).strftime('%Y-%m-%d'),
        exit_advantage=(np.arange(320)%2-.5)*.01,episode_id=[f'actual-held-{i}' for i in range(320)],
        stage='E1',holding_reference_mode='E1')
    cfg=strategy_config({'epochs':1})
    cfg.update(source_snapshot_hash='input',candidate_policy_hash='candidate',E1_policy_hash='fixed-MA-exit',
        pipeline_contract_hash='new-split-population')
    directory=tmp_path/'dataset/E1';directory.mkdir(parents=True)
    dataset={'entries':pd.DataFrame(),'exits':values,'episodes':pd.DataFrame(),
        'reference_mode':'E1','reference_population':{}}
    assets={}
    for key in ('entries','exits','episodes'):
        path=directory/(key+'.parquet');dataset[key].to_parquet(path,index=False)
        assets[key]={'path':str(path),'sha256':_file_hash(path)}
    records=[{'symbol':'TEST.SZ'}]
    manifest={'stage':'E1','source_snapshot_hash':'input','reference_population':{},'failures':[],
        'records':[{'symbol':'TEST.SZ','reference_mode':'E1','reference_population':{}}],
        'pooled_files':assets}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    population=freeze_holding_population(dataset,tmp_path,cfg,records)
    fold={'name':'production','train_end':dates[179],'calibration_start':dates[180],
        'calibration_end':dates[249],'policy_start':dates[250],'policy_end':dates[318],
        'validation_end':dates[318]}
    return dataset,population,cfg,records,dates,fold


def test_zero_final_holding_does_not_erase_actual_E1_rows_and_masks(tmp_path):
    dataset,population,cfg,records,dates,fold=actual_population(tmp_path)
    blocked_entry={'entries':pd.DataFrame(),'exits':pd.DataFrame(),'reference_population':{'fold':fold},
        'reference_mode':'final'}
    assert blocked_entry['exits'].empty
    for head in ('exit','exit_advantage'):
        frame=training_frame(dataset,head,cfg,dates,'E1')
        contract=frame.attrs['label_contract']
        assert contract['holding_population']==population
        assert contract['holding_population_hash']==stable_hash(population)
        assert 'reference_population' not in contract
        original=dataset['exits'].copy()
        if head=='exit_advantage':original['label']=original.exit_advantage
        actual=time_masks(frame,fold,head=head,regression=head=='exit_advantage')
        expected=time_masks(original,fold,head=head,regression=head=='exit_advantage')
        assert all(np.array_equal(a,b) for a,b in zip(actual,expected))
        assert tuple(int(mask.sum()) for mask in actual)==(179,69,68)
    diagnostic=holding_population_diagnostics(dataset,[fold,fold])
    assert diagnostic['rows']==320 and not diagnostic['fold_counts_are_additive']
    assert diagnostic['folds'][0]==diagnostic['folds'][1]


@pytest.mark.parametrize('mutation',['final','candidate','feedback','hash','unbound'])
def test_population_and_rows_cannot_be_spliced_from_gated_or_clone_accounts(tmp_path,mutation):
    dataset,population,cfg,records,dates,fold=actual_population(tmp_path)
    if mutation=='final':dataset['reference_population']={'fold':fold}
    elif mutation=='candidate':dataset['holding_population']['candidate_clones_are_holding_population']=True
    elif mutation=='feedback':dataset['holding_population']['pending_L2_entry_or_exit_used']=True
    elif mutation=='hash':dataset['holding_population']['source_snapshot_hash']='foreign'
    elif mutation=='unbound':dataset.pop('holding_population')
    with pytest.raises(ValueError,match='holding|L2|source|E1'):
        training_frame(dataset,'exit',cfg,dates,'E1')


@pytest.mark.parametrize('mutation',['manifest','rows','final-root','record-mode'])
def test_current_freeze_audits_manifest_pooled_target_and_account_mode(tmp_path,mutation):
    dataset,population,cfg,records,dates,fold=actual_population(tmp_path)
    if mutation=='manifest':(tmp_path/population['manifest_path']).write_text('{}')
    elif mutation=='rows':(tmp_path/population['pooled_exits_path']).write_bytes(b'changed')
    elif mutation=='final-root':population['manifest_path']='dataset/final_population/production/manifest.json'
    else:
        path=tmp_path/population['manifest_path'];value=json.loads(path.read_text())
        value['records'][0]['reference_mode']='final';path.write_text(json.dumps(value))
        population['manifest_sha256']=_file_hash(path);population['records_hash']=stable_hash(value['records'])
    with pytest.raises(ValueError,match='holding|E1|final|account'):
        audit_holding_population(population,tmp_path,cfg,records)


def test_exit_training_requires_actual_E1_prefix_and_complete_bound_target(tmp_path):
    dataset,population,cfg,records,dates,fold=actual_population(tmp_path)
    frame=training_frame(dataset,'exit',cfg,dates,'E1')
    contract=frame.attrs['label_contract']
    manifest=train_head(frame,EXIT_FEATURE_COLUMNS,tmp_path/'new-head',fold,cfg,
        head='exit',device='cpu',contract=contract)
    assert manifest['sample_counts']=={'train':179,'calibration':69,'policy':68,'all':320}
    assert load_head(tmp_path/'new-head')['target_contract']['holding_population_hash']==stable_hash(population)
    frame.loc[0,'holding_reference_mode']='final'
    with pytest.raises(ValueError,match='actual E1'):
        train_head(frame,EXIT_FEATURE_COLUMNS,tmp_path/'invalid',fold,cfg,head='exit',device='cpu',contract=contract)
    obsolete=copy.deepcopy(contract);obsolete.pop('holding_population')
    obsolete['contract_hash']=stable_hash({k:v for k,v in obsolete.items() if k!='contract_hash'})
    with pytest.raises(ValueError,match='holding population'):
        train_head(frame,EXIT_FEATURE_COLUMNS,tmp_path/'obsolete',fold,cfg,head='exit',device='cpu',contract=obsolete)


def test_actual_head_completion_does_not_claim_final_account_qualification():
    status=holding_head_status(
        [{'name':'production','architecture':'linear','heads':{'exit':{'path':'actual'}}}],
        [{'head':'exit_advantage','fold':'production','reason':'insufficient mature exit_advantage rows'}],
        'production')
    assert status['exit']['production_linear_completed']
    assert not status['exit']['final_account_performance_qualified']
    assert not status['exit_advantage']['production_linear_completed']
    assert status['exit_advantage']['unavailable_heads'][0]['reason']=='insufficient mature exit_advantage rows'
