import copy
import pandas as pd
import pytest
from my_strategy.tests.test_czsc_trend_account import fixture
from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS
from my_strategy.services.czsc_trend_dataset import build_account_dataset, complete_proceeds, make_contract


def inputs(monkeypatch, *, close=True):
    frame,calendar,prepared=fixture()
    for key in MA_TREND_COLUMNS:frame[key]=.1
    frame['model_input_eligible']=True;frame['v4_structure_input_eligible']=False;frame['v4_w_input_eligible']=False
    prepared['events']=[]
    if close:prepared['rows'][65].update(action='SELL',emitted=True,event_id='S')
    monkeypatch.setattr('my_strategy.services.czsc_trend_dataset.prepare_trend_decisions',lambda *a,**k:copy.deepcopy(prepared))
    config=strategy_config({'feature_start':calendar[60]})
    config.update(source_snapshot_hash='unit-known-source',pipeline_contract_hash='unit-registered-protocol',
        candidate_policy_hash='unit-candidate',E0_policy_hash='unit-E0',E1_policy_hash='unit-E1')
    return frame,calendar,prepared,config


def test_shared_real_candidate_and_paired_cash_labels(monkeypatch,tmp_path):
    frame,calendar,prepared,config=inputs(monkeypatch)
    dataset,summary=build_account_dataset(frame,calendar,{}, {},config,directory=tmp_path)
    entry=dataset['entries'].iloc[0]
    assert entry['trend_p_up']==.65 and entry['trend_p_down']==.1 and entry['trend_p_neutral']==.25
    assert entry.candidate_cash_ratio==1. and entry.actual_flat_reference
    assert entry.label_available and entry.label==0. and entry.entry_date==calendar[61]
    assert entry.net_return<0 and entry.fees>0
    assert entry.candidate_risk_fraction==pytest.approx(.08)
    # Broker's tighter actual risk boundary, including fill-time account stop,
    # determines R; the later theoretical quoted risk is not substituted.
    assert entry.original_stop_price>prepared['rows'][60]['entry_plan']['stop_price']
    assert summary['mature_exits']>0
    exits=dataset['exits'];assert exits.label_available.all() and (exits.label_end>exits.date).all()
    assert (tmp_path/'candidate_clones.json').exists() and (tmp_path/'holding_pairs.json').exists()


def test_open_episode_and_unclosed_continuation_are_not_loss_labels(monkeypatch):
    frame,calendar,prepared,config=inputs(monkeypatch,close=False)
    dataset,summary=build_account_dataset(frame,calendar,{}, {},config)
    entry=dataset['entries'].iloc[0]
    assert not entry.label_available and pd.isna(entry.label) and entry.open_shares>0
    assert not dataset['exits'].label_available.any() and dataset['exits'].label.isna().all()
    assert summary['mature_entries']==summary['mature_exits']==0


def test_remaining_net_proceeds_accumulate_partial_fills_until_flat():
    ledger=[{'action':'REDUCE','shares':100,'cash_flow':900.,'date':'2026-01-02'},
        {'action':'SELL','shares':100,'cash_flow':850.,'date':'2026-01-05'}]
    result=complete_proceeds(ledger,200)
    assert result=={'proceeds':1750.,'date':'2026-01-05','remaining':0}
    assert complete_proceeds(ledger[:1],200)['date'] is None
    with pytest.raises(ValueError,match='more than actual'):
        complete_proceeds(ledger,150)


def test_E0_exit_and_E1_entry_contracts_are_distinct():
    cfg=strategy_config();cfg.update(source_snapshot_hash='s',pipeline_contract_hash='p',candidate_policy_hash='c',
        E0_policy_hash='E0-independent-rule',E1_policy_hash='E1-signal-pinned-exit')
    a=make_contract('ma_exit',cfg,['2026-01-01','2026-01-02'],'E0')
    b=make_contract('entry',cfg,['2026-01-01','2026-01-02'],'E1')
    assert a['stage']=='E0' and b['stage']=='E1' and a['continuation_policy_hash']!=b['continuation_policy_hash']
    assert a['model_input_schema_hash']!=b['model_input_schema_hash']


def test_policy_worker_preserves_prior_risk_and_cross_prefix(monkeypatch):
    from my_strategy.services import czsc_trend_pipeline as pipeline
    frame,calendar,prepared=fixture(90)
    cfg=strategy_config();cfg['source_snapshot_hash']='s'
    fold={'name':'policy-unit','policy_start':calendar[60],'policy_end':calendar[77]}
    observed={}
    monkeypatch.setattr(pipeline,'read_snapshot',lambda r:(frame.copy(),{}))
    monkeypatch.setattr(pipeline,'policy_trend',lambda f,*a,**k:{})
    def prepare(full,*a,**k):
        observed['prepare_dates']=full.date.tolist();return {'rows':[]}
    def replay(full,*a,**k):
        observed['account_dates']=full.date.tolist();observed['start']=k['start'];observed['seed']=k['initial_state']
        return {'metrics':{},'ledger':[],'rejections':[],'actual_position':None}
    monkeypatch.setattr('my_strategy.services.czsc_trend_consensus.prepare_trend_decisions',prepare)
    monkeypatch.setattr('my_strategy.services.czsc_trend_account.replay_trend_account',replay)
    result=pipeline._policy_worker(({'symbol':frame.symbol.iloc[0]},calendar,cfg,'unused',{'fold':fold},
        [{'cross':.6,'buy_base':.4,'sell':.4,'margin':.1}]))
    assert observed['prepare_dates']==calendar[:78]==observed['account_dates']
    assert observed['start']==calendar[60] and observed['seed']['seasoned_bars']==60
    assert observed['seed']['shares']==0 and observed['seed']['cash']==cfg['initial_cash']
    assert result[0]['ledger']==[]


def test_original_event_clock_features_match_runtime_candidate_builder(monkeypatch):
    from my_strategy.services.czsc_trend_features import original_evidence_events,event_prefix_rows,entry_feature_row,ENTRY_FEATURE_COLUMNS
    from my_strategy.services.czsc_trend_account import candidate_from_row
    import numpy as np
    frame,calendar,prepared,config=inputs(monkeypatch)
    fact={'event_id':'candidate-origin','symbol':frame.symbol.iloc[0],'date':calendar[60],
        'source':'ma','side':'BUY','available_at':calendar[60]+'T15:00:00+08:00',
        'observed_at':calendar[60]+'T15:00:00+08:00','input_eligible':True,'evidence_visible':True}
    prepared['rows'][60]['candidate_evidence']=[fact]
    # Final B is intentionally unqualified as source evidence, matching actual
    # consensus outputs. Its existence cannot set a source event clock.
    prepared['events']=[{'symbol':fact['symbol'],'date':calendar[60],'source':'consensus','side':'BUY'}]
    monkeypatch.setattr('my_strategy.services.czsc_trend_dataset.prepare_trend_decisions',lambda *a,**k:copy.deepcopy(prepared))
    dataset,_=build_account_dataset(frame,calendar,{}, {},config)
    age=event_prefix_rows(frame,original_evidence_events(prepared),calendar)[60]
    candidate=candidate_from_row(prepared['rows'][60])
    state={'date':calendar[60],'close':10.,'cash':100000.,'equity':100000.}
    from my_strategy.services.czsc_trend_model import bind_trend_inputs
    expected=entry_feature_row(bind_trend_inputs(frame,calendar).iloc[60].to_dict(),state,candidate,candidate['entry_plan'],
        initial_cash=100000.,event_prefix=age)
    actual=dataset['entries'].iloc[0]
    assert actual.ma_buy_event_age==0. and np.isnan(actual.structure_buy_event_age)
    assert all(np.isclose(actual[k],expected[k],equal_nan=True) for k in ENTRY_FEATURE_COLUMNS)
