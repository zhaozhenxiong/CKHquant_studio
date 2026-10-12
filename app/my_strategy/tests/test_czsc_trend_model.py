import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_trend_model import build_trend_labels, trend_time_masks, CLASSES, trend_contract
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS


def _frame(n=90):
    dates=pd.bdate_range('2025-01-01',periods=n).strftime('%Y-%m-%d').tolist()
    f=pd.DataFrame({'date':dates,'symbol':'TEST.SZ','open':100.,'high':101.,'low':99.,'close':100.,
                    'source':'sina_unadjusted_v1','has_trade_price':1,'model_input_eligible':True,
                    'wyckoff_corporate_action_suspected':False})
    for c in MA_TREND_COLUMNS:f[c]=.1
    return f,dates


def _labels(f,calendar):return build_trend_labels(f,calendar,source_snapshot_hash='snapshot',pipeline_contract_hash='pipeline')


def test_horizon_includes_reference_entry_and_missing_quotes_are_not_neutral():
    f,cal=_frame();f.loc[80,'close']=110.;data=_labels(f,cal)
    row=data.iloc[70];assert row.label_start==cal[71] and row.label_end==cal[80]
    assert row.label_class=='UP' and row.label==CLASSES.index('UP')
    missing=_labels(f.drop(index=71).reset_index(drop=True),cal)
    assert not missing.loc[missing.date==cal[70],'label_available'].iloc[0]
    assert pd.isna(missing.loc[missing.date==cal[70],'label_class'].iloc[0])


def test_future_labels_do_not_change_causal_input_or_start_band():
    f,cal=_frame();a=_labels(f,cal);f.loc[71:,'high']=999.;f.loc[80,'close']=80.;b=_labels(f,cal)
    assert a.iloc[70].trend_band_log==b.iloc[70].trend_band_log
    assert np.array_equal(a.loc[70,list(MA_TREND_COLUMNS)],b.loc[70,list(MA_TREND_COLUMNS)])
    assert b.iloc[70].label_class=='DOWN'


def test_horizon_tail_and_corporate_action_stay_unavailable():
    f,cal=_frame();f.loc[77,'wyckoff_corporate_action_suspected']=True;data=_labels(f,cal)
    assert data.iloc[70].label_reason=='future_price_basis_or_corporate_action_ambiguous'
    clean,calendar=_frame();assert _labels(clean,calendar).iloc[-1].label_reason=='horizon_not_mature'


def test_information_intervals_are_purged_without_transitive_stock_group():
    f,cal=_frame(200);data=_labels(f,cal)
    fold={'train_end':cal[100],'calibration_start':cal[101],'calibration_end':cal[139],
          'policy_start':cal[140],'policy_end':cal[199]}
    train,calibration,policy=trend_time_masks(data,fold)
    assert train[80] and not train[95]
    assert calibration[120] and not calibration[135]
    assert policy[170]
    assert not (train&calibration).any() and not (calibration&policy).any()


def test_old_profit_target_is_not_new_three_class_identity():
    contract=trend_contract(source_snapshot_hash='s',pipeline_contract_hash='p')
    assert contract['target']=='next_verified_open_horizon_direction'
    assert contract['classes']==['UP','DOWN','NEUTRAL']
    assert 'profit' not in contract['target']


def test_three_class_fit_prior_identity_and_production_clock(tmp_path):
    from my_strategy.services.czsc_trend_model import train_trend_head,trend_contract,TrendSession,SCHEMA_HASH
    from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS
    from my_strategy.core.run_context import stable_hash
    import numpy as np
    dates=pd.bdate_range('2023-01-02',periods=190)
    frame=pd.DataFrame({k:np.sin(np.arange(190)/7+i) for i,k in enumerate(MA_TREND_COLUMNS)})
    frame['date']=dates.strftime('%Y-%m-%d');frame['symbol']='TEST';frame['label']=np.arange(190)%3
    frame['label_end']=(dates+pd.offsets.BDay(1)).strftime('%Y-%m-%d');frame['label_available']=True
    frame['input_eligible']=True;frame['model_input_eligible']=True
    frame['observation_id']=[stable_hash({'day':d}) for d in frame.date]
    config={'trend':{'min_train_per_class':2,'min_calibration_per_class':2,'min_policy_per_class':2}}
    contract=trend_contract(config,source_snapshot_hash='synthetic-unit',pipeline_contract_hash='unit-protocol')
    frame.attrs['label_contract']=contract
    fold={'name':'unit','train_end':frame.date.iloc[89],'calibration_start':frame.date.iloc[91],
        'calibration_end':frame.date.iloc[119],'policy_start':frame.date.iloc[121],'policy_end':frame.date.iloc[149]}
    manifest=train_trend_head(frame,tmp_path/'model',fold,config,contract=contract)
    assert manifest['classes']==['UP','DOWN','NEUTRAL']
    assert manifest['train_label_end']<=fold['train_end']
    assert manifest['calibration_label_end']<=fold['calibration_end']
    assert manifest['preprocess']['median']==pytest.approx(frame.loc[:88,list(MA_TREND_COLUMNS)].median().tolist())
    later=frame.iloc[151:154].copy();later.attrs['schema_hash']=SCHEMA_HASH
    p,_=TrendSession().predict(later,tmp_path/'model',expected_contract=contract)
    assert p.shape==(3,3) and np.allclose(p.sum(axis=1),1)
    assert np.all(p[:,2]>0) and not np.allclose(p[:,1],1-p[:,0])
    with pytest.raises(ValueError,match='not actually trained'):
        TrendSession().predict(later,tmp_path/'model',expected_contract=contract,mode='production')
    early=frame.iloc[120:121].copy();early.attrs['schema_hash']=SCHEMA_HASH
    with pytest.raises(ValueError,match='not available'):
        TrendSession().predict(early,tmp_path/'model',expected_contract=contract)


def test_independent_MA_dependency_requires_complete_verified_65_raw_sessions():
    from my_strategy.services.czsc_trend_model import bind_trend_inputs
    frame,calendar=_frame(200)
    full=bind_trend_inputs(frame,calendar)
    assert not full.iloc[63].model_input_eligible and full.iloc[64].model_input_eligible
    missing=bind_trend_inputs(frame.drop(index=100).reset_index(drop=True),calendar)
    assert not missing.loc[missing.date==calendar[150],'model_input_eligible'].iloc[0]
    assert missing.loc[missing.date==calendar[166],'model_input_eligible'].iloc[0]
    prefix=bind_trend_inputs(frame.iloc[:130],calendar)
    assert prefix.model_input_eligible.tolist()==full.iloc[:130].model_input_eligible.tolist()
