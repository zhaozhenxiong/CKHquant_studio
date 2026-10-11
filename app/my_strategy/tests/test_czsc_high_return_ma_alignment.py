"""Signal-time exit binding and mature target integrity regressions."""
from pathlib import Path
import copy
import math
import pytest
import numpy as np
import pandas as pd
from my_strategy.services import czsc_high_return_ma_alignment as stage
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_high_return_features import ENTRY_FEATURE_COLUMNS, L1_MA_ENTRY_COLUMNS, L1_MA_EXIT_COLUMNS, MA_ENTRY_SCHEMA_HASH, MA_EXIT_SCHEMA_HASH
from my_strategy.services.czsc_high_return_labels import target_contract
from my_strategy.services.czsc_high_return_models import schema_hash, time_masks
from my_strategy.services.czsc_signal_points import SIGNAL_POINTS_VERSION, signal_point_parameters, make_point_decisions
from my_strategy.services.czsc_trade_episodes import training_frame
from my_strategy.services.czsc_analysis import strategy_config

@pytest.fixture
def market(monkeypatch, tmp_path):
    days = pd.bdate_range('2024-01-02', periods=110).strftime('%Y-%m-%d').tolist()
    cfg = strategy_config(); cfg.update(initial_cash=100000.,high_return={'max_holding_sessions':60},signal_points={'min_risk_fraction':.001})
    frame=pd.DataFrame({k:.1 for k in ENTRY_FEATURE_COLUMNS},index=range(len(days))).assign(date=days,symbol='000001.SZ',open=10.,high=10.2,low=9.8,close=10.,volume=1000000.,amount=10000000.,source='tushare',has_trade_price=1,model_input_eligible=True,structure_input_eligible=True,wyckoff_input_eligible=True,input_eligible=True)
    params=signal_point_parameters(cfg)
    buy={'event_id':'b70','symbol':'000001.SZ','date':days[70],'source':'ma','side':'BUY','event_type':'ma_rule','available_at':days[70]+'T15:00:00+08:00','input_eligible':True,'candidate_eligible':True,'selected_by_reference':True}
    sell={**buy,'event_id':'s90','date':days[90],'side':'SELL','available_at':days[90]+'T15:00:00+08:00'}
    candidate={'symbol':'000001.SZ','date':days[70],'opportunity_id':'o70','sources':['ma'],'source_event_ids':['b70'],'source_eligibility':{'ma':True,'structure':False,'wyckoff':False},'source_selected_by_reference':{'ma':True,'structure':False,'wyckoff':False},'source_stops':{'ma':9.6,'structure':None,'wyckoff':None},'stop_price':9.6,'atr14':.5,'candidate_eligible':True,'selected_by_reference':True}
    points={'version':SIGNAL_POINTS_VERSION,'config':params,'config_hash':stable_hash(params),'events':[buy,sell],'candidates':[candidate]}
    frame.loc[:69,['close_above_ma20','ma20_slope5']]=-.1
    frame.loc[90:,['close_above_ma20','ma20_slope5']]=-.1
    frame['atr14_ratio']=.08
    points=stage.build_signal_points(frame,cfg)
    target=target_contract(head='exit',source='ma',config=cfg,market_dates=days,source_snapshot_hash='source',feature_schema_hash=MA_EXIT_SCHEMA_HASH,continuation_policy_hash='v0',initial_cash=100000.)
    from my_strategy.services.czsc_high_return_research import _head_data
    target=_head_data(pd.DataFrame(),target,'ma_exit',L1_MA_EXIT_COLUMNS)[1]
    root=(tmp_path/'models').resolve()
    bundles=[]; manifests={}
    for name,avail,score in [('old',days[65],.2),('new',days[72],.95)]:
        bundle={'name':name,'available_at':avail,'heads':{'ma_exit':{'path':name,'manifest_sha256':name+'manifest'}},'policy':{'ma_exit_threshold':.55}}
        manifest={'head':'ma_exit','columns':list(L1_MA_EXIT_COLUMNS),'feature_schema_hash':schema_hash(L1_MA_EXIT_COLUMNS,'ma_exit'),'target_contract':target,'target_contract_hash':stable_hash(target),'available_at':avail,'manifest_sha256':name+'manifest','checkpoint_sha256':name+'weights','export_sha256':name+'export','training_completed_at':'2026-10-11T02:00:00+08:00','train_label_end':days[40],'calibration_label_end':days[50],'fold':{'name':name,'train_end':days[45],'calibration_start':days[46],'calibration_end':days[55],'policy_start':days[56],'policy_end':days[60],'validation_end':avail},'score':score}
        bundles.append(bundle);manifests[str(root/name)]=manifest
    monkeypatch.setattr(stage,'load_head',lambda path:copy.deepcopy(manifests[str(path)]))
    calls=[]
    def predict(session,values,model_root,bundle,head,**kwargs):
        m=manifests[str(root/bundle['name'])];calls.append((bundle['name'],values['date'],values['running_peak_return']))
        return (m['score'] if values['input_eligible'] else None),m
    monkeypatch.setattr(stage,'_predict',predict)
    return frame,days,cfg,points,root,bundles,manifests,calls

def policy(market,**kw):
    frame,days,cfg,points,root,bundles,_,_=market
    kw.setdefault('expected_exit_contract_hash', market[6][str(root/'old')]['target_contract_hash'])
    return stage.make_signal_pinned_ma_exit_policy(frame,days,root,bundles,config=cfg,points=points,source_snapshot_hash='source',session=object(),**kw)

def snap(market,index=71):
    frame,days,cfg,points,*_=market
    plan=make_point_decisions(frame,points,source='ma',config=cfg,include_unselected=True)[70]['entry_plan']
    return {'symbol':'000001.SZ','date':days[index],'available_at':days[index]+'T15:00:00+08:00','entry_date':days[71],'plan_id':plan['plan_id'],'shares':1000,'entry_shares':1000,'cash':89990.,'equity':99990.,'close':10.,'actual_cost_per_share':10.01,'stop_price':9.6,'initial_cash':100000.,'bar':{'high':10.2,'low':9.8}}

def test_pin_signal_not_future_holding_checkpoint(market):
    p=policy(market)
    before=p(snap(market,71));after=p(snap(market,80))
    assert before['checkpoint']==after['checkpoint']=='old'
    assert after['probability']==.2 and not after['request_exit']
    assert before['candidate_exit_binding']==after['candidate_exit_binding']
    assert all(x[0]=='old' for x in market[-1])

def test_available_at_same_day_is_not_prior(market):
    frame,days,cfg,points,root,bundles,manifests,_=market
    bundles[0]['available_at']=days[70]; manifests[str(root/'old')]['available_at']=days[70];manifests[str(root/'old')]['fold']['validation_end']=days[70]
    result=policy(market)(snap(market))
    assert result['status']=='explicit_rule_v0_no_earlier_exit_head' and result['probability'] is None and not result['apply']

def test_actual_completion_guard_different_from_reconstruction(market):
    assert policy(market)(snap(market))['checkpoint']=='old'
    p=policy(market,usage_mode='production')
    r=p(snap(market));assert r['status']=='explicit_rule_v0_no_earlier_exit_head' and not r['apply']

def test_wrong_plan_and_wrong_fill_clock_fail(market):
    p=policy(market); s=snap(market);s['plan_id']='unknown'
    with pytest.raises(ValueError,match='no frozen'):p(s)
    s=snap(market);s['entry_date']=market[1][70]
    with pytest.raises(ValueError,match='mismatch'):p(s)

def test_prefix_factory_is_independent(market):
    p=policy(market);s=snap(market,80);s['close']=14.;s['bar']['high']=14.3
    p(s)
    with pytest.raises(ValueError,match='reused'):p(snap(market,71))
    fresh=p.new_replay();r=fresh(snap(market,71))
    assert r['probability']==.2
    assert market[-1][-1][2] < .1

def test_binding_external_mutation_cannot_change_live_policy(market):
    p=policy(market);key=next(iter(p.bindings));p.bindings[key]['checkpoint']='new';p.recipe['clock']='holding_day'
    assert p(snap(market,80))['checkpoint']=='old'
    assert p.new_replay().recipe['clock']=='candidate_signal_close'

def test_available_head_missing_input_abstains_not_pre_model_fallback(market):
    market[0].loc[71,'model_input_eligible']=False
    r=policy(market)(snap(market))
    assert r['status']=='pinned_ma_exit_input_unavailable_abstain' and r['checkpoint']=='old'
    assert r['probability'] is None and not r['apply']
    assert r['candidate_exit_binding']['mode']=='signal_time_pinned_ma_exit'

def test_corrupt_or_immature_exit_catalog_rejected(market):
    root=market[4];manifest=market[6][str(root/'old')]
    manifest['train_label_end']=market[1][60]
    with pytest.raises(ValueError,match='immature'):policy(market)
    manifest['train_label_end']=market[1][40];manifest['checkpoint_sha256']='changed'
    with pytest.raises(ValueError,match='target missing or changed'):policy(market,expected_exit_contract_hash='another')

def test_broad_targets_broker_fee_maturity_and_contract_are_new(market):
    frame,days,cfg,points,root,bundles,_,calls=market
    result=stage.build_exit_bound_ma_entry_dataset(frame,days,model_root=root,bundles=bundles,config=cfg,points=points,source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'])
    row=result['entries'].iloc[0]; account=result['reference_accounts']['ma:000001.SZ']
    assert row.exit_date==days[91] and row.label_end==days[91] and row.net_return<0
    assert row.fees==pytest.approx(sum(fill['fee'] for fill in account['ledger']))
    assert set(result['entries'].ma_exit_binding_mode)=={'signal_time_pinned_ma_exit'}
    assert all(call[0]=='old' for call in calls)
    training=training_frame(result,source='ma',head='entry',ma_only=True)
    assert tuple(training.attrs['feature_columns'])==tuple(L1_MA_ENTRY_COLUMNS)
    assert training.attrs['label_contract']['feature_schema_hash']==MA_ENTRY_SCHEMA_HASH
    assert training.attrs['label_contract']['label_version']==stage.ALIGNMENT_VERSION
    assert training.label.iloc[0]==0. and training.label_available.iloc[0]
    assert all(result['exits'].label_end==result['exits'][['forced_exit_date','continuation_exit_date']].max(axis=1))

def test_ma_entry_model_cannot_condition_own_target_candidates(market):
    market[3]['events'][0]['probability']=.7
    with pytest.raises(ValueError,match='cannot depend'):stage.build_exit_bound_ma_entry_dataset(market[0],market[1],model_root=market[4],bundles=market[5],config=market[2],points=market[3],source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(market[4]/'old')]['target_contract_hash'])

def test_partial_liquidation_final_date_is_maturity_and_future_head_not_upgraded(market):
    frame,days,cfg,points,root,bundles,_,calls=market
    frame.loc[90:98,'volume']=10000.
    points=stage.build_signal_points(frame,cfg)
    result=stage.build_exit_bound_ma_entry_dataset(frame,days,model_root=root,bundles=bundles,config=cfg,points=points,source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'])
    row=result['entries'].iloc[0];account=result['reference_accounts']['ma:000001.SZ']
    assert row.label_end>days[91] and account['ledger'][-1]['position_after']==0
    assert any(fill['action']=='REDUCE' for fill in account['ledger'])
    assert all(call[0]=='old' for call in calls)

def test_immature_target_not_loss_or_training_sample(market):
    frame,days,cfg,points,root,bundles,_,_=market
    result=stage.build_exit_bound_ma_entry_dataset(frame.iloc[:80],days,model_root=root,bundles=bundles,config=cfg,points=None,source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'])
    row=result['entries'].iloc[0]
    assert not row.label_available and math.isnan(row.positive_return_label)
    assert not result['exits'].label_available.any()

def test_group_purge_uses_latest_mature_branch_not_row_date():
    dates=['2024-04-01','2024-04-02']
    data=pd.DataFrame({'symbol':['000001.SZ']*2,'date':dates,'label':[1.,0.],'label_available':[True,True],'label_end':['2024-04-10','2024-07-01'],'economic_group_id':['same']*2,'input_eligible':[True,True]})
    fold={'train_end':'2024-06-30','calibration_start':'2024-07-01','calibration_end':'2024-08-01','policy_start':'2024-08-02','policy_end':'2024-09-30','validation_end':'2024-09-30'}
    masks=time_masks(data,fold)
    assert not any(mask.any() for mask in masks)


def test_bootstrap_exit_target_identity_cannot_be_implicit(market):
    with pytest.raises(ValueError,match='target missing or changed'):policy(market,expected_exit_contract_hash=None)

def test_label_stage_source_must_match_exit_bootstrap_source(market):
    frame,days,cfg,points,root,bundles,_,_=market
    with pytest.raises(ValueError,match='bootstrap source/calendar mismatch'):
        stage.build_exit_bound_ma_entry_dataset(frame,days,model_root=root,bundles=bundles,config=cfg,points=points,source_snapshot_hash='different-source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'])


def test_all_future_catalog_sources_checked_even_before_first_available_head(market):
    frame,days,cfg,points,root,bundles,manifests,_=market
    for item in bundles:
        m=manifests[str(root/item['name'])]
        item['available_at']=m['available_at']=m['fold']['validation_end']=days[100]
    manifests[str(root/'new')]['target_contract']['source_snapshot_hash']='wrong-source'
    with pytest.raises(ValueError,match='bootstrap source/calendar mismatch'):policy(market)

def test_canonical_target_population_rejects_deleted_points_or_candidate_only_scores(market):
    frame,days,cfg,points,root,bundles,_,_=market
    for mutation in ('remove','score'):
        changed=copy.deepcopy(points)
        if mutation=='remove':changed['candidates']=[]
        else:
            changed['candidates'][0]['l1_ma_entry_probability']=.9
            changed['candidates'][0]['l1_ma_entry_available']=True
        with pytest.raises(ValueError,match='altered candidate population'):
            stage.build_exit_bound_ma_entry_dataset(frame,days,model_root=root,bundles=bundles,config=cfg,points=changed,source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'])

def test_snapshot_clock_and_exact_next_session_enforced(market):
    for field,value in [('entry_date',market[1][72]),('date',market[1][70]),('available_at',market[1][71]+'T14:00:00+08:00')]:
        p=policy(market);s=snap(market);s[field]=value
        with pytest.raises(ValueError,match='binding mismatch'):p(s)

def test_inference_identity_swap_rejected_after_binding(market):
    p=policy(market);market[6][str(market[4]/'old')]['checkpoint_sha256']='replacement'
    with pytest.raises(ValueError,match='head drifted'):p(snap(market))

def test_reference_callback_and_serialized_attrs_target_contract_match(market,tmp_path):
    frame,days,cfg,points,root,bundles,_,_=market
    saved=[]
    result=stage.build_exit_bound_ma_entry_dataset(frame,days,model_root=root,bundles=bundles,config=cfg,points=points,source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'],reference_callback=lambda *args:saved.append(args))
    assert len(saved)==1 and saved[0][0]=='ma' and saved[0][2]['ledger']
    for kind,head in [('entries','entry'),('exits','exit')]:
        assert result[kind].attrs['source_contracts']['ma']==result['contracts']['ma'][head]
        destination=tmp_path/(kind+'.parquet');result[kind].to_parquet(destination)
        reread=pd.read_parquet(destination)
        assert reread.attrs['source_contracts']['ma']==result['contracts']['ma'][head]


def test_extended_runtime_calendar_preserves_original_training_prefix(market):
    frame,days,cfg,points,root,bundles,_,_=market
    expanded=[*days,*pd.bdate_range(pd.Timestamp(days[-1])+pd.Timedelta(days=1),periods=3).strftime('%Y-%m-%d')]
    p=stage.make_signal_pinned_ma_exit_policy(frame,expanded,root,bundles,config=cfg,points=points,source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'],exit_training_calendar_hash=stable_hash(days),exit_training_calendar_dates=days)
    assert p.recipe['calendar_compatibility']=='verified_original_prefix_then_appended'
    assert p(snap(market))['checkpoint']=='old'
    with pytest.raises(ValueError,match='requires frozen bootstrap'):
        stage.make_signal_pinned_ma_exit_policy(frame,expanded,root,bundles,config=cfg,points=points,source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'],exit_training_calendar_hash=stable_hash(days))

def test_deleted_original_calendar_session_is_not_forward_append(market):
    frame,days,cfg,points,root,bundles,_,_=market
    broken=[day for i,day in enumerate(days) if i!=10]
    with pytest.raises(ValueError,match='changed frozen bootstrap prefix'):
        stage.make_signal_pinned_ma_exit_policy(frame[frame.date!=days[10]],broken,root,bundles,config=cfg,points=points,source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'],exit_training_calendar_hash=stable_hash(days),exit_training_calendar_dates=days)


def test_outer_bundle_cannot_alias_bootstrap_exit_fold(market):
    market[5][0]['name']='outer_L2_alias'
    with pytest.raises(ValueError,match='identity/schema/availability'):policy(market)

def test_lineage_annotation_cannot_relabel_existing_v0_dataset(market):
    from my_strategy.services.czsc_trade_episodes import build_episode_dataset
    frame,days,cfg,points,*_=market
    old=build_episode_dataset(frame,days,cfg,points=points,sources=('ma',),source_snapshot_hash='source')
    with pytest.raises(ValueError,match='not generated under'):stage.bind_ma_exit_lineage(old,policy(market))

def test_full_four_path_lineage_only_changes_ma_contract(market):
    from my_strategy.services.czsc_trade_episodes import build_episode_dataset
    frame,days,cfg,points,*_=market;p=policy(market)
    full=build_episode_dataset(frame,days,cfg,points=points,sources=('ma','union'),source_snapshot_hash='source',exit_policies={'ma':p},exit_policy_hashes={'ma':p.recipe_hash})
    unchanged=copy.deepcopy(full['contracts']['union'])
    stage.bind_ma_exit_lineage(full,p)
    assert full['contracts']['union']==unchanged
    assert full['entries'].query("source=='union'").ma_exit_binding_hash.isna().all()
    assert full['entries'].query("source=='ma'").ma_exit_binding_hash.notna().all()


def test_root_expected_source_and_calendar_names_and_hash_semantics(market):
    frame,days,cfg,points,root,bundles,_,_=market
    m=market[6][str(root/'old')]
    assert m['target_contract_hash']==stable_hash(m['target_contract'])
    assert m['target_contract_hash']!=m['target_contract']['contract_hash']
    p=stage.make_signal_pinned_ma_exit_policy(frame,days,root,bundles,config=cfg,points=points,session=object(),expected_source_snapshot_hash='source',expected_calendar_hash=stable_hash(days),expected_exit_contract_hash=m['target_contract_hash'])
    assert p(snap(market))['checkpoint']=='old'
    with pytest.raises(ValueError,match='conflicting declared bootstrap source'):
        stage.make_signal_pinned_ma_exit_policy(frame,days,root,bundles,config=cfg,points=points,session=object(),source_snapshot_hash='wrong',expected_source_snapshot_hash='source',expected_exit_contract_hash=m['target_contract_hash'])

def test_label_wrapper_declared_original_calendar_is_exact(market):
    frame,days,cfg,points,root,bundles,_,_=market
    with pytest.raises(ValueError,match='label stage calendar differs'):
        stage.build_exit_bound_ma_entry_dataset(frame,days,model_root=root,bundles=bundles,config=cfg,points=points,source_snapshot_hash='source',session=object(),expected_exit_contract_hash=market[6][str(root/'old')]['target_contract_hash'],expected_calendar_hash='wrong-calendar')

def test_future_factory_reads_verified_original_calendar_file(market,tmp_path):
    import json
    frame,days,cfg,points,root,bundles,manifests,_=market
    expanded=[*days,*pd.bdate_range(pd.Timestamp(days[-1])+pd.Timedelta(days=1),periods=3).strftime('%Y-%m-%d')]
    relocated=tmp_path/'models';original=relocated/'data_snapshots';original.mkdir(parents=True)
    original.joinpath('calendar.json').write_text(json.dumps({'verified':True,'dates':days,'source':{'snapshot_hash':'source'}}))
    for item in bundles:manifests[str(relocated.resolve()/item['name'])]=manifests[str(root/item['name'])]
    p=stage.make_signal_pinned_ma_exit_policy(frame,expanded,relocated,bundles,config=cfg,points=points,session=object(),expected_source_snapshot_hash='source',expected_calendar_hash=stable_hash(days),expected_exit_contract_hash=manifests[str(root/'old')]['target_contract_hash'])
    # Constructor/factory verification suffices; fake predictor binds original-root paths.
    assert p.recipe['calendar_compatibility']=='verified_original_prefix_then_appended'


from pathlib import Path
import copy
import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_high_return_features import L1_MA_ENTRY_COLUMNS
from my_strategy.services.czsc_high_return_labels import target_contract
from my_strategy.services.czsc_trade_episodes import build_episode_dataset

from my_strategy.services import czsc_high_return_ma_alignment_audit as audit
AUDIT_PATH = Path(audit.__file__)


@pytest.fixture
def bound(market, monkeypatch):
    frame, days, cfg, points, root, bundles, manifests, calls = market
    p = policy(market)
    data = stage.build_exit_bound_ma_entry_dataset(frame, days, model_root=root,
        bundles=bundles, config=cfg, points=points, source_snapshot_hash='source',
        session=object(), expected_exit_contract_hash=manifests[str(root/'old')]['target_contract_hash'])
    return data, p, market


def test_original_binding_audit_accepts_real_broker_fixture(bound):
    data, p, market = bound
    audit._unavailable_and_maturity(data)
    result = audit._annotated_rows(data, p, market[1], '000001.SZ')
    assert result['entries:signal_time_pinned_ma_exit'] == len(data['entries'])
    assert result['exits:signal_time_pinned_ma_exit'] == len(data['exits'])
    assert data['reference_accounts']['ma:000001.SZ']['ledger']


def test_altered_saved_binding_is_rejected_even_with_recomputed_self_hash(bound):
    data, p, market = bound
    record = json.loads(data['entries'].iloc[0].ma_exit_binding_json)
    record['threshold'] = .01
    record['binding_hash'] = stable_hash({k:v for k,v in record.items() if k != 'binding_hash'})
    data['entries'].loc[0, 'ma_exit_binding_json'] = json.dumps(record)
    data['entries'].loc[0, 'ma_exit_binding_hash'] = record['binding_hash']
    with pytest.raises(ValueError, match='stored candidate exit binding'):
        audit._annotated_rows(data, p, market[1], '000001.SZ')


def test_audit_independently_rejects_future_head_bound_to_own_candidate(bound):
    data, p, market = bound
    item = next(b for b in p.bindings.values() if b['checkpoint'])
    item['available_at'] = item['candidate_signal_date']
    item['binding_hash'] = stable_hash({k:v for k,v in item.items() if k != 'binding_hash'})
    with pytest.raises(ValueError, match='future or immature'):
        audit._annotated_rows(data, p, market[1], '000001.SZ')


def test_wrong_episode_signal_or_delayed_fill_cannot_borrow_same_plan(bound):
    data, p, market = bound
    original = data['episodes'].copy(deep=True)
    data['episodes'].loc[0, 'signal_date'] = market[1][69]
    with pytest.raises(ValueError, match='episode plan was rebound'):
        audit._annotated_rows(data, p, market[1], '000001.SZ')
    data['episodes'] = original
    data['episodes'].loc[0, 'entry_date'] = market[1][72]
    with pytest.raises(ValueError, match='next verified|original episode'):
        audit._annotated_rows(data, p, market[1], '000001.SZ')


def test_raw_wide_population_does_not_accept_deleted_or_scored_candidates(market):
    raw = market[3]
    audit._raw_population_identity(copy.deepcopy(raw), raw)
    for field in ('delete', 'score'):
        changed = copy.deepcopy(raw)
        if field == 'delete':
            changed['candidates'] = []
        else:
            changed['candidates'][0]['l1_ma_entry_probability'] = .99
        with pytest.raises(ValueError, match='exact independent raw-wide'):
            audit._raw_population_identity(changed, raw)


def test_counterfeit_v0_bootstrap_contract_rejected(market):
    frame, days, cfg, points, *_ = market
    data = build_episode_dataset(frame, days, cfg, points=points, sources=('ma',), source_snapshot_hash='source')
    expected = copy.deepcopy(data['contracts']['ma']['ma_exit'])
    audit._bootstrap_original_contract(data, expected, 'source')
    changed = data['contracts']['ma']['ma_exit']
    changed['continuation_policy_hash'] = 'another-learned-policy'
    changed['contract_hash'] = stable_hash({k:v for k,v in changed.items() if k != 'contract_hash'})
    with pytest.raises(ValueError, match='counterfeit bootstrap'):
        audit._bootstrap_original_contract(data, expected, 'source')


def test_old_target_cannot_be_mislabeled_as_v2_by_only_changing_version(bound):
    data, p, market = bound
    for c in data['contracts']['ma'].values():
        c['exit_alignment_recipe_hash'] = 'old-v0-target'
        c['contract_hash'] = stable_hash({k:v for k,v in c.items() if k != 'contract_hash'})
    with pytest.raises(ValueError, match='old target was relabeled'):
        audit._annotated_rows(data, p, market[1], '000001.SZ')


def test_censored_entry_or_exit_cannot_become_negative(bound):
    data, _, _ = bound
    data['entries'].loc[0, 'label_available'] = False
    data['entries'].loc[0, 'label'] = 0.
    with pytest.raises(ValueError, match='unavailable/censored'):
        audit._unavailable_and_maturity(data)
    data['entries'].loc[0, 'label'] = np.nan
    data['exits'].loc[0, 'label_available'] = False
    data['exits'].loc[0, 'label'] = 0.
    with pytest.raises(ValueError, match='unavailable/censored'):
        audit._unavailable_and_maturity(data)


def test_maturity_is_final_complete_liquidation_and_latest_exit_branch(bound):
    data, _, market = bound
    data['entries'].loc[0, 'label_end'] = market[1][90]
    with pytest.raises(ValueError, match='final actual liquidation'):
        audit._unavailable_and_maturity(data)
    data['entries'].loc[0, 'label_end'] = data['entries'].iloc[0].exit_date
    data['exits'].loc[0, 'label_end'] = market[1][73]
    with pytest.raises(ValueError, match='both complete liquidation|original decision'):
        audit._unavailable_and_maturity(data)


def test_whole_group_late_branch_purges_all_earlier_states(market):
    frame = pd.DataFrame({c:[.1,.1,.1] for c in L1_MA_ENTRY_COLUMNS})
    frame = frame.assign(symbol='000001.SZ', source='ma',
        date=['2024-04-01', '2024-04-02', '2024-04-03'],
        label=[1., 0., 1.], label_available=True, input_eligible=True,
        label_end=['2024-04-10', '2024-07-01', '2024-04-11'],
        economic_group_id=['same', 'same', 'separate'],
        ma_exit_binding_mode=['signal_time_pinned_ma_exit']*3)
    base = target_contract(head='ma_entry', source='ma', config=market[2], market_dates=market[1],
        source_snapshot_hash='source', feature_schema_hash='fixture', continuation_policy_hash='pinned', initial_cash=100000.)
    fold = {'name':'one', 'train_end':'2024-06-30', 'calibration_start':'2024-07-01',
        'calibration_end':'2024-08-01', 'policy_start':'2024-08-02', 'policy_end':'2024-09-30', 'validation_end':'2024-09-30'}
    counts, derived = audit._fold_counts(frame, base, 'ma_entry', [fold])
    assert counts['one']['all'] == 3 and counts['one']['train'] == 1
    assert counts['one']['train_binding_modes'] == {'signal_time_pinned_ma_exit':1}
    assert counts['one']['calibration'] == counts['one']['policy'] == 0
    assert derived['model_head'] == 'ma_entry'


def test_empty_supported_no_candidate_stock_has_zero_masks(market):
    base = target_contract(head='ma_entry', source='ma', config=market[2], market_dates=market[1],
        source_snapshot_hash='source', feature_schema_hash='fixture', continuation_policy_hash='pinned', initial_cash=100000.)
    counts, _ = audit._fold_counts(pd.DataFrame(), base, 'ma_entry', [{'name':'one'}])
    assert counts == {'one':{'all':0,'train':0,'calibration':0,'policy':0}}


def test_manifest_missing_source_failed_or_wrong_unsupported_status_rejected(tmp_path):
    records = [{'symbol':'000001.SZ','path':'/fixture/original','sha256':'source-file'}]
    item = {'symbol':'000001.SZ','source_path':'/fixture/original','source_sha256':'source-file',
        'source_hash_verified':True,'tradable_board':True,'episode_status':'completed'}
    value = {'records':[item], 'sources':['ma'], 'stage':'ma_entry_alignment', 'failures':[],
        'requested':1, 'successful':1, 'hash_verified_count':1, 'supported_mainboard':1}
    path = tmp_path/'dataset/ma_entry_alignment/manifest.json'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value))
    assert audit._stage_manifest(tmp_path, 'ma_entry_alignment', ['ma'], records) == value
    for mutation in ('missing', 'failure', 'unsupported'):
        changed = copy.deepcopy(value)
        if mutation == 'missing':changed['records'] = []
        elif mutation == 'failure':changed['failures'] = [{'symbol':'000001.SZ','error':'missing'}]
        else:changed['records'][0]['episode_status'] = 'cash_only_unsupported_board'
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='coverage|unsupported'):
            audit._stage_manifest(tmp_path, 'ma_entry_alignment', ['ma'], records)


def test_serialized_stale_contract_attrs_rejected(bound, tmp_path):
    data, _, _ = bound
    record = {}
    for kind in ('entries', 'exits', 'episodes'):
        path = tmp_path/(kind+'.parquet')
        data[kind].to_parquet(path, index=False)
        record[kind+'_path'] = str(path); record[kind+'_sha256'] = audit._file_hash(path)
    path = tmp_path/'contracts.json'; path.write_text(json.dumps(data['contracts']))
    record.update(contracts_path=str(path), contracts_sha256=audit._file_hash(path))
    audit._frame_files(tmp_path, record)
    data['entries'].attrs['source_contracts']['ma']['continuation_policy_hash'] = 'stale'
    path = tmp_path/'entries.parquet'; data['entries'].to_parquet(path, index=False)
    record['entries_sha256'] = audit._file_hash(path)
    with pytest.raises(ValueError, match='serialized alignment frame target attrs'):
        audit._frame_files(tmp_path, record)


def test_actual_four_path_ma_annotation_does_not_cover_union(bound):
    _, p, market = bound
    frame, days, cfg, points, *_ = market
    full = build_episode_dataset(frame, days, cfg, points=points, sources=('ma','union'),
        source_snapshot_hash='source', exit_policies={'ma':p}, exit_policy_hashes={'ma':p.recipe_hash})
    stage.bind_ma_exit_lineage(full, p)
    audit._annotated_rows(full, p, days, '000001.SZ')
    first = full['entries'].index[full['entries'].source.eq('union')][0]
    full['entries'].loc[first, 'ma_exit_binding_hash'] = 'learned-exit'
    with pytest.raises(ValueError, match='non-MA target received'):
        audit._annotated_rows(full, p, days, '000001.SZ')


def test_full_audit_traverses_original_stages_and_rejects_head_count_drift(market, tmp_path, monkeypatch):
    """One synthetic fold, real tiny Broker/Parquet files, fake model integrity.

    This exercises the full aggregation API; it is not a real model fit, a
    twelve-fold acceptance or a test of the separately tested lineage loader.
    """
    from my_strategy.services.czsc_high_return_research import _head_data
    from my_strategy.services.czsc_high_return_models import schema_hash, split_validation
    from my_strategy.services.czsc_trade_episodes import training_frame
    from my_strategy.services.czsc_research import _save
    import my_strategy.services.czsc_high_return_research as research
    from my_strategy.services import czsc_high_return_ma_training as trainer
    frame, days, old_cfg, _, _, _, _, _ = market
    cfg = copy.deepcopy(old_cfg)
    cfg.update(version='czsc_high_return_v2_ma_exit_bound', ma_entry_alignment=trainer.ALIGNMENT_MODE)
    frame.loc[95:97, ['close_above_ma20','ma20_slope5']] = .1
    points = stage.build_signal_points(frame, cfg)
    raw_fold = {'name':'old','train_end':days[92], 'validation_start':days[93], 'validation_end':days[109]}
    fold = split_validation(raw_fold, days)
    monkeypatch.setattr(research, '_ma_folds', lambda end:[raw_fold])
    source_path = tmp_path/'original_source.parquet'; frame.to_parquet(source_path, index=False)
    source = {'symbol':'000001.SZ','path':str(source_path),'sha256':audit._file_hash(source_path)}
    bootstrap = build_episode_dataset(frame, days, cfg, points=points, sources=('ma',), source_snapshot_hash='source')
    expected_v0 = trainer._expected_v0_exit_contract(cfg, days, 'source')
    assert bootstrap['contracts']['ma']['ma_exit'] == expected_v0
    _, exit_contract = _head_data(pd.DataFrame(), expected_v0, 'ma_exit', stage.L1_MA_EXIT_COLUMNS)
    exit_manifest = copy.deepcopy(market[6][str(market[4]/'old')])
    exit_manifest.update(fold=fold, available_at=days[109], target_contract=exit_contract,
        target_contract_hash=stable_hash(exit_contract), train_label_end=days[91], calibration_label_end=days[99])
    exit_binding = {'path':'old','manifest_sha256':exit_manifest['manifest_sha256'],
        'target_contract_hash':exit_manifest['target_contract_hash']}
    exit_bundle = {'name':'old','available_at':days[109], 'fold':fold,
        'heads':{'ma_exit':exit_binding},'policy':{'ma_exit_threshold':.55},'failures':[]}
    monkeypatch.setattr(stage, 'load_head', lambda path:copy.deepcopy(exit_manifest))
    aligned = stage.build_exit_bound_ma_entry_dataset(frame, days, model_root=tmp_path,
        bundles=[exit_bundle], config=cfg, points=points, source_snapshot_hash='source', session=object(),
        expected_exit_contract_hash=exit_manifest['target_contract_hash'])
    p = stage.make_signal_pinned_ma_exit_policy(frame, days, tmp_path, [exit_bundle],
        config=cfg, points=points, source_snapshot_hash='source', session=object(),
        expected_exit_contract_hash=exit_manifest['target_contract_hash'])
    final = build_episode_dataset(frame, days, cfg, points=points, sources=('ma','structure','wyckoff','union'),
        source_snapshot_hash='source', exit_policies={'ma':p}, exit_policy_hashes={'ma':p.recipe_hash})
    stage.bind_ma_exit_lineage(final, p)
    manifests = {}
    for stage_name, data, sources in [('bootstrap',bootstrap,['ma']),('ma_entry_alignment',aligned,['ma']),
                                    ('final',final,['ma','structure','wyckoff','union'])]:
        root = tmp_path/'dataset'/stage_name/'000001_SZ'; root.mkdir(parents=True)
        _save(root/'points.json', points)
        record = {'symbol':'000001.SZ','source_path':source['path'],'source_sha256':source['sha256'],
            'source_hash_verified':True,'tradable_board':True,'episode_status':'completed',
            'source_rows':len(frame),'events':len(points['events']),'candidates':len(points['candidates']),
            'points_path':str(root/'points.json'),'points_sha256':audit._file_hash(root/'points.json')}
        for kind in ('entries','exits','episodes'):
            path = root/(kind+'.parquet'); data[kind].to_parquet(path, index=False)
            record[kind+'_path'] = str(path); record[kind+'_sha256'] = audit._file_hash(path)
        _save(root/'contracts.json',data['contracts'])
        record.update(contracts_path=str(root/'contracts.json'), contracts_sha256=audit._file_hash(root/'contracts.json'))
        if stage_name != 'bootstrap':
            _save(root/'ma_exit_alignment_recipe.json',data['alignment_recipe'])
            _save(root/'ma_candidate_exit_bindings.json',data['candidate_exit_bindings'])
            for key, name in [('ma_exit_alignment_recipe','ma_exit_alignment_recipe.json'),
                              ('ma_candidate_exit_bindings','ma_candidate_exit_bindings.json')]:
                record[key+'_path'] = str(root/name); record[key+'_sha256'] = audit._file_hash(root/name)
        manifest = {'records':[record],'sources':sources,'stage':'ma_entry_alignment' if stage_name=='ma_entry_alignment' else 'episodes',
            'failures':[],'requested':1,'successful':1,'hash_verified_count':1,'supported_mainboard':1}
        _save(root.parent/'manifest.json',manifest); manifests[stage_name] = manifest
    model_manifests = {}
    bindings = {}
    for head, data, kind in [('ma_exit',bootstrap,'exit'),('ma_entry',aligned,'entry')]:
        training = training_frame(data, source='ma', head=kind, ma_only=True)
        stats, contract = audit._fold_counts(training, data['contracts']['ma'][head], head, [fold])
        stats = stats['old']
        m = copy.deepcopy(exit_manifest)
        m.update(head=head, target_contract=contract, target_contract_hash=stable_hash(contract),
            columns=list(stage.L1_MA_EXIT_COLUMNS if head=='ma_exit' else L1_MA_ENTRY_COLUMNS),
            sample_counts={k:stats[k] for k in ('train','calibration','policy','all')},
            train_label_end=stats.get('train_label_end'), calibration_label_end=stats.get('calibration_label_end'))
        m['feature_schema_hash'] = schema_hash(m['columns'],head)
        m['manifest_sha256'] = head+'-manifest'
        name = 'old' if head=='ma_exit' else 'entry-old'
        bindings[head] = {'path':name,'manifest_sha256':m['manifest_sha256'],'target_contract_hash':m['target_contract_hash']}
        model_manifests[str(tmp_path/name)] = m
        if head == 'ma_exit':
            # Update the fixture's frozen head identity to the same fake model
            # manifest used by the mask audit, then regenerate stage lineage.
            assert m['train_label_end'] == days[91] and m['calibration_label_end'] == days[99]
    # The factory identity above used its original fake manifest identity; keep
    # that identity for the exit head, while masks are independently counted.
    model_manifests[str(tmp_path/'old')]['manifest_sha256'] = exit_manifest['manifest_sha256']
    bindings['ma_exit']['manifest_sha256'] = exit_manifest['manifest_sha256']
    pair = {**exit_bundle,'heads':bindings,'l1_target_dependency':{
        'entry_target_contract_hash':bindings['ma_entry']['target_contract_hash'],
        'exit_bootstrap_target_contract_hash':stable_hash(exit_contract),
        'label_recipe_hash':aligned['contracts']['ma']['ma_entry']['exit_alignment_recipe_hash'],
        'label_exit_routing':'strictly_prior_inner_oof_at_candidate_close','holding_checkpoint_upgrade':False}}
    _save(tmp_path/'models/ma_bundles.json',[pair])
    lineage = {'exit_only_bundles':[exit_bundle], 'expected_exit_target_contract_hash':stable_hash(exit_contract),
        'calendar_hash':stable_hash(days),'_frozen_calendar_dates':days}
    _save(tmp_path/'models/ma_exit_bootstrap_lineage.json',lineage)
    path = tmp_path/'dataset/ma_entry_alignment/manifest.json'
    pipeline = {'stage_order':audit.STAGE_ORDER,'status':'completed','entry_heads_used_in_label_generation':0,
        'holding_checkpoint_upgrade':False,'exit_heads':1,'entry_heads':1,
        'lineage_path':'models/ma_exit_bootstrap_lineage.json',
        'lineage_sha256':audit._file_hash(tmp_path/'models/ma_exit_bootstrap_lineage.json'),
        'artifact_manifest_path':'dataset/ma_entry_alignment/manifest.json','artifact_manifest_sha256':audit._file_hash(path),
        'artifact_coverage':{k:v for k,v in manifests['ma_entry_alignment'].items() if k!='records'},
        'entry_target_contract':aligned['contracts']['ma']['ma_entry']}
    report = {'config':cfg,'source_snapshot':{'snapshot_hash':'source'},'data_end':days[-1],
        'model_bundles':[pair],'dataset_records':[source],'episode_records':manifests['final']['records'],
        'l1_ma_target_pipeline':pipeline,'protocol_freeze':{'code_hashes':
            {str(AUDIT_PATH.with_name(name)):'fixture-sha' for name in audit.REQUIRED_ALIGNMENT_CODE}}}
    _save(tmp_path/'reports/high_return.json', report)
    _save(tmp_path/'data_snapshots/calendar.json',{'dates':days})
    monkeypatch.setattr(trainer, 'read_exit_bootstrap_lineage', lambda *a,**k:copy.deepcopy(lineage))
    monkeypatch.setattr(audit, 'load_head', lambda path:copy.deepcopy(model_manifests[str(path)]))
    result = audit.audit_ma_alignment(tmp_path,report,days)
    assert result['errors'] == []
    assert result['status']=='pass' and not result['publication_allowed']
    assert result['checks']['l1_head_original_masks_and_dependency'] == 2
    assert result['fold_mature_inputs']['ma_entry']['old']['train_binding_modes'] == {'explicit_rule_v0_no_earlier_exit_head':1}
    model_manifests[str(tmp_path/'entry-old')]['sample_counts']['all'] += 1
    failed = audit.audit_ma_alignment(tmp_path,report,days)
    assert failed['status']=='fail'
    assert any(e['check']=='l1_head_original_masks_and_dependency' and 'sample counts differ' in e['error'] for e in failed['errors'])
    model_manifests[str(tmp_path/'entry-old')]['sample_counts']['all'] -= 1
    report['protocol_freeze']['code_hashes'] = {}
    _save(tmp_path/'reports/high_return.json',report)
    failed = audit.audit_ma_alignment(tmp_path,report,days)
    assert any(e['check']=='l1_pipeline_dependency_order_coverage' and 'dependencies were not frozen' in e['error'] for e in failed['errors'])
