from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.light_score_training import (
    BASE_FEATURES, EXTRA_FEATURES, KRONOS_FEATURES, prepare_score_features,
    make_direction_training_frame, chronological_split_ranges,
)
from my_strategy.services.light_score_v2 import fit_light_score, score_light_samples
from my_strategy.services.light_qualification import content_hash, evaluate_qualification, resolve_qualification


@pytest.fixture(scope='module')
def sample():
    rng=np.random.default_rng(81)
    dates=pd.bdate_range('2010-01-01',periods=1200)
    frame=pd.DataFrame(rng.normal(size=(1200,len(BASE_FEATURES+EXTRA_FEATURES))),columns=BASE_FEATURES+EXTRA_FEATURES)
    frame['symbol']='TEST.SZ';frame['as_of']=dates.strftime('%Y-%m-%d')
    frame['label_mature_at']=(dates+pd.offsets.BDay(5)).strftime('%Y-%m-%d')
    frame['label']=np.arange(len(frame))%3
    frame['feature_available_at']=frame.as_of+'T15:00:00+08:00'
    frame.loc[:50,'support_distance']=np.nan
    ranges=chronological_split_ranges(frame)
    return frame,ranges


def fit(frame,ranges,**kwargs):
    return fit_light_score(frame,ranges,source_identity={'fixture':'isolated_test'},
        label_contract={'horizon':5,'entry':'next_verified_open','threshold':.003},
        trials=[{'widths':[8,4],'max_epochs':3,'patience':2}],**kwargs)


@pytest.fixture(scope='module')
def trained(sample):
    return fit(*sample)


def test_real_torch_fit_and_compact_roundtrip(trained,sample):
    import json
    assert trained['model_type']=='torch_mlp'
    assert trained['runtime']['device']=='cpu'
    assert trained['architecture']['input_dimensions']==34
    assert trained['runtime']['parameter_count']<5000
    assert trained['preprocessing']['fitted_on']=='train_only'
    assert trained['l2_baseline']['metrics']['policy']['rows']>0
    assert trained['population']['train']['excluded_immature_rows']==5
    restored=json.loads(json.dumps(trained,allow_nan=False))
    row=sample[0].iloc[[-1]]
    result=score_light_samples(restored,row,as_of=row.iloc[0].as_of,historical_reconstruction=True)
    again=score_light_samples(restored,row,as_of=row.iloc[0].as_of,historical_reconstruction=True)
    assert result==again
    assert result['status']=='scored'
    assert sum(result['probabilities'][0].values())==pytest.approx(1)
    assert result['qualification']['allow_increase'] is True
    assert result['qualification']['paper_allocation_eligible'] is False
    assert 0<=result['scores'][0]<=100


def test_preprocessing_has_no_test_information(trained,sample):
    frame,ranges=sample
    training=frame[(frame.as_of>=ranges['train'][0])&(frame.as_of<=ranges['train'][1])&(frame.label_mature_at<=ranges['train'][1])]
    column=BASE_FEATURES[0]
    index=trained['features'].index(column)
    assert trained['preprocessing']['medians'][index]==pytest.approx(training[column].median())
    assert trained['preprocessing']['clip_high'][index]==pytest.approx(training[column].quantile(.99))
    assert trained['available_after']<trained['evaluation_end']


def test_temporal_misuse_and_tamper_fail_closed(trained,sample):
    row=sample[0].iloc[[-1]]
    with pytest.raises(ValueError,match='Historical application'):
        score_light_samples(trained,row,as_of=row.iloc[0].as_of)
    with pytest.raises(ValueError,match='development/calibration'):
        score_light_samples(trained,row,as_of='2010-01-01',historical_reconstruction=True)
    bad=deepcopy(trained);bad['temperature']+=1
    with pytest.raises(ValueError,match='identity mismatch'):
        score_light_samples(bad,row,as_of=row.iloc[0].as_of,historical_reconstruction=True)
    paper=score_light_samples(trained,row,as_of=row.iloc[0].as_of,historical_reconstruction=True,mode='paper')
    assert paper['qualification']['allow_increase'] is False


def test_split_feature_availability_and_budget_validation(sample):
    frame,ranges=sample
    bad=deepcopy(ranges);bad['policy']=bad['calibration']
    with pytest.raises(ValueError,match='Overlapping'):
        fit(frame,bad)
    delayed=frame.copy();delayed.loc[0,'feature_available_at']='2099-01-01'
    with pytest.raises(ValueError,match='available after'):
        fit(delayed,ranges)
    with pytest.raises(InterruptedError): fit(frame,ranges,cancel_check=lambda:True)
    with pytest.raises(TimeoutError): fit(frame,ranges,budget_seconds=1e-12)
    with pytest.raises(ValueError,match='thread limit'): fit(frame,ranges,cpu_threads=3)
    with pytest.raises(ValueError,match='core feature'): fit(frame.drop(columns=[BASE_FEATURES[0]]),ranges)


def test_registry_bounds_and_invalid_candidate(sample):
    frame,ranges=sample
    registry={'entries':[{'kind':'mlp','experiment_id':str(i)} for i in range(6)]}
    with pytest.raises(ValueError,match='experiment budget'):
        fit(frame,ranges,experiment_registry=registry)
    with pytest.raises(ValueError,match='dimension limit'):
        fit(
            frame.assign(**{f'extra{i}':0 for i in range(20)}),ranges,
            feature_columns=BASE_FEATURES+EXTRA_FEATURES+[f'extra{i}' for i in range(20)],optional_columns=EXTRA_FEATURES)


def test_registration_survives_cancel_before_optimizer(sample):
    registered=[];stop={'requested':False}
    def register(registry):
        registered.append(registry)
        stop['requested']=True
    with pytest.raises(InterruptedError):
        fit(*sample,registration_callback=register,cancel_check=lambda:stop['requested'])
    assert len(registered)==1
    assert registered[0]['entries'][0]['kind']=='mlp'


def test_compact_registration_is_committed_before_training_cancellation(sample,tmp_path):
    from my_strategy.storage.research_store import ResearchStore
    from my_strategy.services.light_experiment_registry import persist_experiment_registry,load_experiment_input
    stop={'requested':False}
    with ResearchStore(tmp_path) as store:
        def register(registry):
            committed=persist_experiment_registry(store,registry)
            stop['requested']=True
            return committed
        with pytest.raises(InterruptedError):
            fit(*sample,registration_callback=register,cancel_check=lambda:stop['requested'])
        saved=store.get_setting('light_experiment_registry')
        assert len(saved['entries'])==1 and 'input_identities' not in saved
        assert load_experiment_input(store,saved['entries'][0]['input_identity_ref'])['source']=={'fixture':'isolated_test'}
        assert not store.list('mlp_score_models')


def test_feature_preparation_ignores_large_attrs_without_mutating_input_or_events():
    class AuditPayload:
        def __deepcopy__(self,memo):
            raise AssertionError('Non-feature attrs must not be copied into every row')
    days=pd.bdate_range('2020-01-01',periods=80)
    close=np.arange(80)*.01+10
    frame=pd.DataFrame({'date':days,'symbol':'002436.SZ','open':close,'high':close+.2,
        'low':close-.2,'close':close,'volume':1000000,
        'native_w_demand_score':.4,'native_w_supply_score':.2})
    events=[{'date':str(days[5].date()),'available_at':str(days[7].date())+'T15:00:00+08:00',
        'symbol':'002436.SZ','confirmed':True,'side':'BUY','risk_boundary':9.5}]
    expected=prepare_score_features(frame,events)
    marker=AuditPayload();frame.attrs['score_source_events']=marker
    frame.attrs['audit_payload']='x'*510000
    actual=prepare_score_features(frame,events)
    pd.testing.assert_frame_equal(actual,expected)
    assert not actual.attrs
    assert frame.attrs['score_source_events'] is marker and len(frame.attrs['audit_payload'])==510000
    assert events[0]['available_at']==str(days[7].date())+'T15:00:00+08:00'


def test_b_incomplete_coverage_honestly_pending(sample):
    frame,ranges=sample
    frame=frame.copy();frame['kronos_state']='missing'
    for column in KRONOS_FEATURES: frame[column]=np.nan
    result=fit(frame,ranges,variant='B')
    assert result['status']=='pending' and result['model_created'] is False
    assert 'state_dict' not in result
    assert result['coverage']['reasons']==['frozen_observation_manifest_missing']


def test_b_fits_frozen_population_and_abstains_on_unseen_missing_mode(sample):
    frame,ranges=sample
    frame=frame.copy();frame['symbol']=[f'{i%5:06d}.SZ' for i in range(len(frame))]
    frame['kronos_state']='valid'
    values=[.01,-.03,.05,.08,.6,1.]
    for column,value in zip(KRONOS_FEATURES,values): frame[column]=value
    observations=[]
    for row in frame.itertuples():
        observations.append(dict(symbol=row.symbol,date=row.as_of,status='succeeded',forecast=dict(
            identity={'input':{'symbol':row.symbol,'as_of':row.as_of,'data_sha256':'frozen-test-input'}},
            quality={'prediction_usable':True,'valid_fraction':1.},
            summary={'terminal_return_median':.01,'terminal_return_p10':-.03,'terminal_return_p90':.05,'up_path_fraction':.6})))
    manifest=dict(model_independent=True,source_policy_id='frozen-structure-baseline',frozen_at='2026-01-01',
        total_calls=len(observations),observations=observations)
    artifact=fit(frame,ranges,variant='B',observation_manifest=manifest)
    assert artifact['status']=='completed'
    assert artifact['architecture']['input_dimensions']==44
    assert artifact['supported_missing_patterns']==['valid']
    row=frame.iloc[[-1]].copy()
    result=score_light_samples(artifact,row,as_of=row.iloc[0].as_of,historical_reconstruction=True)
    assert result['status']=='scored'
    delayed=row.copy();day=row.iloc[0].as_of
    delayed['feature_available_at']=day+'T15:05:00+08:00'
    with pytest.raises(ValueError,match='Features unavailable'):
        score_light_samples(artifact,delayed,as_of=day,historical_reconstruction=True)
    actual=score_light_samples(artifact,delayed,as_of=day,decision_time=day+'T15:05:00+08:00',historical_reconstruction=True)
    assert actual['scores']==result['scores']
    assert pd.Timestamp(actual['decision_time'])==pd.Timestamp(day+'T15:05:00+08:00')
    assert actual['model_version']==artifact['model_version']
    row['kronos_state']='missing';row[KRONOS_FEATURES]=np.nan
    unavailable=score_light_samples(artifact,row,as_of=row.iloc[0].as_of,historical_reconstruction=True)
    assert unavailable['scores']==[None]
    assert unavailable['qualification']['allow_increase'] is False
    corrupted=frame.copy();corrupted.loc[0,'kronos_return_median']=99
    pending=fit(corrupted,ranges,variant='B',observation_manifest=manifest)
    assert pending['status']=='pending'
    assert 'forecast_feature_values_do_not_match_manifest' in pending['coverage']['reasons']
    omitted=fit(frame.iloc[1:],ranges,variant='B',observation_manifest=manifest)
    assert omitted['status']=='pending'
    assert 'frozen_training_population_not_complete' in omitted['coverage']['reasons']


def test_l2_same_input_comparator_is_separately_identified(trained,sample):
    row=sample[0].iloc[[-1]]
    mlp=score_light_samples(trained,row,as_of=row.iloc[0].as_of,historical_reconstruction=True)
    linear=score_light_samples(trained,row,as_of=row.iloc[0].as_of,historical_reconstruction=True,model_kind='l2')
    assert linear['model_type']=='multinomial_logistic_L2'
    assert linear['scorer_id']!=mlp['scorer_id']
    with pytest.raises(ValueError,match='no independent paper-policy'):
        score_light_samples(trained,row,as_of=row.iloc[0].as_of,historical_reconstruction=True,model_kind='l2',mode='paper')


def test_qualification_is_not_a_hardcoded_false(trained):
    artifact=deepcopy(trained)
    # Synthetic qualification evidence only; does not mutate/select a real model.
    for split in ('policy','test'):
        artifact['metrics'][split]['log_loss']=.2;artifact['metrics'][split]['brier']=.1
        artifact['metrics'][split]['frozen_train_prior_baseline'].update(log_loss=.9,brier=.6)
        artifact['l2_baseline']['metrics'][split].update(log_loss=.8,brier=.5)
    artifact['model_version']=content_hash({k:v for k,v in artifact.items() if k!='model_version'})
    source=content_hash(artifact['identity']['source'])
    windows=[]
    for year in (2027,2028,2029):
        windows.append(dict(start=f'{year}-01-01',end=f'{year}-12-31',symbols=[f'{x:06d}.SZ' for x in range(10)],
            completed_cycles=10,total_return=.1,baseline_total_return=.05,sharpe=1.,baseline_sharpe=.5,
            max_drawdown=.05,turnover=1.,maximum_exposure=.5,coverage_fraction=1.,
            cost_stress={k:dict(passed=True,total_return=.1,baseline_total_return=.05) for k in ('1.0','1.5','2.0')}))
    registry=deepcopy(artifact['experiment_registry'])
    registry['entries'].append(dict(kind='policy',experiment_id='frozen-policy-trial',strategy_identity='frozen-policy'))
    evidence=dict(model_version=artifact['model_version'],source_identity_sha256=source,
        strategy_identity='frozen-policy',cost_identity='actual-costs',experiment_registry=registry,
        experiment_registry_sha256=content_hash(registry),policy_trial_id='frozen-policy-trial',
        policy_frozen_at='2026-12-01',evaluated_at='2030-01-01',windows=windows,
        frozen_limits=dict(max_drawdown=.1,turnover=3.,maximum_exposure=.9,minimum_coverage=.9),
        execution_contract={k:True for k in ('broker_ledger_verified','next_verified_open','t_plus_one','costs_included','historical_rules_verified','corporate_actions_verified')})
    event=evaluate_qualification(artifact,policy_evidence=evidence,now='2030-01-02')
    assert event['paper_allocation_eligible'] is True,event['failed_checks']
    assert event['independent_forward_status']=='pending'
    assert resolve_qualification(artifact,as_of='2030-01-03',mode='paper',qualification_event=event)['allow_increase'] is True
    assert resolve_qualification(artifact,as_of='2029-01-03',mode='paper',qualification_event=event)['allow_increase'] is False
    forged=deepcopy(event);forged['effective_after']='2020-01-01';forged['event_id']=content_hash({k:v for k,v in forged.items() if k!='event_id'})
    with pytest.raises(ValueError,match='evidence mismatch'):
        resolve_qualification(artifact,as_of='2030-01-03',mode='paper',qualification_event=forged)
    evidence['execution_contract'].pop('t_plus_one')
    failed=evaluate_qualification(artifact,policy_evidence=evidence,now='2030-01-02')
    assert failed['paper_allocation_eligible'] is False
    assert 't_plus_one' in failed['failed_checks']


def test_qualification_matches_legacy_inline_trials_to_compact_cas_without_rewriting_model(trained,tmp_path):
    from my_strategy.services.light_experiment_registry import persist_experiment_registry
    from my_strategy.storage.research_store import ResearchStore
    artifact=deepcopy(trained)
    registry=artifact['experiment_registry']
    inputs=registry.pop('input_identities')
    for trial in registry['entries']:
        trial['input_identity']=inputs[trial.pop('input_identity_ref')['sha256']]
    artifact['model_version']=content_hash({k:v for k,v in artifact.items() if k!='model_version'})
    frozen=deepcopy(artifact)
    with ResearchStore(tmp_path) as store:
        compact=persist_experiment_registry(store,registry)
    def check(value):
        evidence={'experiment_registry':value,'experiment_registry_sha256':content_hash(value)}
        return evaluate_qualification(artifact,policy_evidence=evidence,now='2030-01-01')
    verified=check(compact)
    assert verified['execution_checks']['training_trials_preserved'] is True
    assert verified['paper_allocation_eligible'] is False  # Storage migration is no promotion.
    for change in ('configuration','input_sha','missing_trial'):
        bad=deepcopy(compact)
        if change=='configuration':bad['entries'][0]['configuration']['seed']=999
        elif change=='input_sha':bad['entries'][0]['input_identity_ref']['sha256']='f'*64
        else:bad['entries']=[]
        assert check(bad)['execution_checks']['training_trials_preserved'] is False
    assert artifact==frozen


def test_features_use_confirmation_time_and_labels_use_verified_sessions():
    dates=pd.bdate_range('2020-01-01',periods=80)
    frame=pd.DataFrame(dict(date=dates,symbol='TEST.SZ',open=np.arange(80)+100.,close=np.arange(80)+101.,
        high=np.arange(80)+102.,low=np.arange(80)+99.,volume=np.arange(80)+1000.,
        amount=100000.,source="tushare",has_trade_price=1,
        native_w_demand_score=.5,native_w_supply_score=.2))
    event=dict(symbol='TEST.SZ',date=str(dates[10].date()),available_at=str(dates[20].date())+'T15:01:00+08:00',
        side='BUY',confirmed=True,input_eligible=True,risk_boundary=100.)
    features=prepare_score_features(frame,[event])
    assert features.loc[20,'structure_age_sessions']!=features.loc[20,'structure_age_sessions']
    assert features.loc[21,'structure_age_sessions']==0
    assert features.loc[22,'structure_age_sessions']==1
    altered=frame.copy();altered.loc[60:,'close']*=2
    future=prepare_score_features(altered,[event])
    pd.testing.assert_frame_equal(features.loc[:59],future.loc[:59])
    training=make_direction_training_frame(features,list(dates))
    assert len(training)==75
    broken=features.drop(index=30)
    gap=make_direction_training_frame(broken,list(dates))
    assert not gap.as_of.isin([str(d.date()) for d in dates[25:30]]).any()


def test_kronos_future_observation_is_rejected():
    dates=pd.bdate_range('2020-01-01',periods=2)
    frame=pd.DataFrame(dict(date=dates,symbol='T',open=1.,close=1.,high=2.,low=.5,volume=100.))
    with pytest.raises(ValueError,match='available after'):
        prepare_score_features(frame,variant='B',forecasts={str(dates[0].date()):{'available_at':'2099-01-01'}})


def test_forward_b_requires_actual_availability_before_decision_and_open():
    dates=pd.bdate_range('2020-01-01',periods=2)
    frame=pd.DataFrame(dict(date=dates,symbol='T',open=1.,close=1.,high=2.,low=.5,volume=100.))
    forecast={'as_of':'2020-01-01','available_at':'2020-01-01T15:00:00+08:00',
        'actual_available_at':'2020-01-01T15:05:00+08:00',
        'execution_deadline':'2020-01-02T09:30:00+08:00',
        'quality':{'prediction_usable':True,'valid_fraction':1.},
        'summary':{'terminal_return_median':.01,'terminal_return_p10':-.03,'terminal_return_p90':.05,'up_path_fraction':.6}}
    snapshots={('T','2020-01-01'):forecast}
    with pytest.raises(ValueError,match='available after'):
        prepare_score_features(frame,variant='B',forecasts=snapshots)
    with pytest.raises(ValueError,match='available after'):
        prepare_score_features(frame,variant='B',forecasts=snapshots,decision_time='2020-01-01T15:04:59+08:00')
    for decision in ('2020-01-01T15:05:00+08:00','2020-01-02T09:29:59+08:00'):
        features=prepare_score_features(frame,variant='B',forecasts=snapshots,decision_time={('T','2020-01-01'):decision})
        assert features.iloc[0].kronos_state=='valid'
        assert features.iloc[0].feature_available_at=='2020-01-01T15:05:00+08:00'
        assert features.iloc[1].feature_available_at=='2020-01-02T15:00:00+08:00'
    for decision in ('2020-01-02T09:30:00+08:00','2020-01-02T09:30:01+08:00'):
        with pytest.raises(ValueError,match='executable open'):
            prepare_score_features(frame,variant='B',forecasts=snapshots,decision_time=decision)
    forecast['available_at']='2020-01-01T15:06:00+08:00'
    with pytest.raises(ValueError,match='available after'):
        prepare_score_features(frame,variant='B',forecasts=snapshots,decision_time='2020-01-01T15:05:00+08:00')


def test_explicit_decision_time_preserves_a_artifact_identity_and_numbers(trained,sample):
    original=deepcopy(trained);row=sample[0].iloc[[-1]].copy();day=row.iloc[0].as_of
    ordinary=score_light_samples(trained,row,as_of=day,historical_reconstruction=True)
    explicit=score_light_samples(trained,row,as_of=day,decision_time=day+'T15:00:00+08:00',historical_reconstruction=True)
    explicit.pop('decision_time')
    assert ordinary==explicit
    assert trained==original
    with pytest.raises(ValueError,match='precedes'):
        score_light_samples(trained,row,as_of=day,decision_time=day+'T00:00:00+08:00',historical_reconstruction=True)
