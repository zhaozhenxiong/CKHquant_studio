"""Final chart intents must execute through actual cash/inventory Broker paths."""
from __future__ import annotations
import copy
import json

import pandas as pd
import pytest

from my_strategy.scripts.validate_native_v4 import (
    _actual_episodes, _cash_failure, aggregate_accounts, map_final_points_to_decisions, verify_actual_account,
)
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_trade_episodes_v4 import _max_exit_callback
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES


def _scenario(*, sell=True):
    calendar=pd.bdate_range('2025-01-02',periods=100).strftime('%Y-%m-%d').tolist()
    raw=pd.DataFrame({'symbol':'000001.SZ','date':calendar,'open':20.,'high':20.3,'low':19.7,
        'close':20.,'volume':1000000.,'amount':20000000.,'source':'sina_unadjusted_v1','has_trade_price':1})
    plan={'plan_id':'original-frozen-risk-plan','symbol':'000001.SZ','policy':'risk',
        'signal_date':calendar[70],'available_at':calendar[70]+'T15:00:00+08:00','status':'active',
        'entry_allowed':True,'valid_for_sessions':1,'stop_price':18.9,'risk_fraction':.01,
        'price_floor':19.1,'price_ceiling':21.4,'target_weight':.2,'reference_price':20.}
    rows=[{'date':day,'raw_risk_plan':plan if i==70 else None,'rule_sell':bool(sell and i==75),
           'source_exit_event_ids':['original-source-S'] if sell and i==75 else []} for i,day in enumerate(calendar)]
    consensus={'events':[{'date':calendar[70],'side':'BUY','event_id':'finalB'}]}
    if sell:consensus['events'].append({'date':calendar[75],'side':'SELL','event_id':'finalS'})
    decisions,blocked=map_final_points_to_decisions(rows,consensus,symbol='000001.SZ',
        start=calendar[69],end=calendar[-1],max_weight=.95)
    assert not blocked
    return raw,calendar,plan,decisions


def _execute(raw,calendar,decisions,*,settings=None,exit_policy=None):
    return execute_decisions('000001.SZ',raw,decisions,calendar[69],100000.,settings or strategy_config(),
        'isolated-validation-test',market_dates=calendar,verified_sources=VERIFIED_SOURCES,entry_policy='risk',exit_policy=exit_policy)


def test_final_b_s_and_neutral_use_exact_risk_plan_actual_cash_and_next_open():
    raw,calendar,plan,decisions=_scenario()
    original=copy.deepcopy(plan)
    assert decisions[70]['entry_plan']==plan
    assert decisions[71]['hold_current_position'] is True
    account=_execute(raw,calendar,decisions)
    assert [(fill['action'],fill['date'],fill['signal_date']) for fill in account['ledger']]==[
        ('BUY',calendar[71],calendar[70]),('SELL',calendar[76],calendar[75])]
    assert account['metrics']['open_shares']==0
    checks=verify_actual_account(account,calendar=calendar,decisions=decisions,initial_cash=100000.,symbol='000001.SZ')
    assert checks['cash_inventory_equity_reconstructed']
    assert checks['t_plus_one'] and checks['next_verified_open']
    assert plan==original


def test_final_buy_without_plan_is_explicitly_blocked_never_reconstructed():
    raw,calendar,plan,decisions=_scenario()
    rows=[{'date':day,'raw_risk_plan':{**plan,'status':'rejected','entry_allowed':False} if i==70 else None}
          for i,day in enumerate(calendar)]
    mapped,blocked=map_final_points_to_decisions(rows,{'events':[{'date':calendar[70],'side':'BUY'}]},
        symbol='000001.SZ',start=calendar[69],end=calendar[-1],max_weight=.95)
    assert len(blocked)==1 and blocked[0]['reason']=='final_buy_without_active_source_policy_plan'
    assert mapped[70]['action']=='HOLD'
    account=_execute(raw,calendar,mapped)
    assert account['ledger']==[] and account['metrics']['final_equity']==100000.


def test_raw_union_exit_without_chart_s_preserves_label_continuation():
    raw,calendar,plan,decisions=_scenario(sell=False)
    rows=[{'date':day,'raw_risk_plan':plan if i==70 else None,'rule_sell':i==75,
           'source_exit_event_ids':['raw-S-with-no-chart-point'] if i==75 else []} for i,day in enumerate(calendar)]
    mapped,blocked=map_final_points_to_decisions(rows,{'events':[{'date':calendar[70],'side':'BUY'}]},
        symbol='000001.SZ',start=calendar[69],end=calendar[-1],max_weight=.95)
    assert not blocked
    assert mapped[75]['raw_source_exit'] and not mapped[75]['chart_sell']
    assert mapped[75]['policy_exit_kind']=='frozen_raw_union_source_exit'
    account=_execute(raw,calendar,mapped)
    assert account['ledger'][1]['date']==calendar[76]


def test_chart_s_outside_raw_exit_policy_is_audited_without_changing_target():
    raw,calendar,plan,decisions=_scenario(sell=False)
    rows=[{'date':day,'raw_risk_plan':plan if i==70 else None,'rule_sell':False} for i,day in enumerate(calendar)]
    mapped,blocked=map_final_points_to_decisions(rows,{'events':[{'date':calendar[70],'side':'BUY'},
        {'date':calendar[75],'side':'SELL'}]},symbol='000001.SZ',start=calendar[69],end=calendar[-1],max_weight=.95)
    assert blocked[0]['reason']=='chart_sell_outside_frozen_source_exit_policy'
    assert mapped[75]['chart_sell'] and not mapped[75]['raw_source_exit']
    assert mapped[75]['hold_current_position'] and mapped[75]['action']=='HOLD'
    account=_execute(raw,calendar,mapped)
    assert len(account['ledger'])==1 and account['metrics']['open_shares']>0


def test_events_before_fresh_cash_boundary_cannot_open_new_account():
    raw,calendar,plan,decisions=_scenario()
    rows=[{'date':day,'raw_risk_plan':plan if i==70 else None} for i,day in enumerate(calendar)]
    mapped,blocked=map_final_points_to_decisions(rows,{'events':[{'date':calendar[70],'side':'BUY'}]},
        symbol='000001.SZ',start=calendar[72],end=calendar[-1],max_weight=.95)
    assert not blocked and all(row['action']=='HOLD' for row in mapped)


def test_hard_stop_has_priority_without_final_s_and_neutral_does_not_sell():
    raw,calendar,plan,decisions=_scenario(sell=False)
    for key in ('open','close'):raw.loc[74:,key]=18.5
    raw.loc[74:,'low']=18.4;raw.loc[74:,'high']=18.7;raw.loc[74:,'amount']=raw.loc[74:,'volume']*18.5
    account=_execute(raw,calendar,decisions)
    assert account['ledger'][0]['date']==calendar[71]
    assert len(account['ledger'])==2
    assert account['ledger'][1]['date']==calendar[75]
    assert account['ledger'][1]['reason']=='actual_account_stop'
    assert any(row['reason']=='conservative_lower_price_guard' for row in account['rejections'])
    verify_actual_account(account,calendar=calendar,decisions=decisions,initial_cash=100000.,symbol='000001.SZ')


def test_maximum_holding_uses_actual_verified_sessions_and_next_open():
    raw,calendar,plan,decisions=_scenario(sell=False)
    callback=_max_exit_callback({d:i for i,d in enumerate(calendar)},3,'test-policy-contract')
    account=_execute(raw,calendar,decisions,exit_policy=callback)
    assert account['ledger'][1]['date']==calendar[74]
    assert account['exit_policy_diagnostics'][2]['status']=='frozen_max_holding_sessions'
    verify_actual_account(account,calendar=calendar,decisions=decisions,initial_cash=100000.,symbol='000001.SZ')


def test_double_slippage_is_independent_actual_ledger_cost_stress():
    raw,calendar,plan,decisions=_scenario()
    settings=strategy_config();double=copy.deepcopy(settings);double['execution']['slippage']*=2
    normal=_execute(raw,calendar,decisions,settings=settings)
    stressed=_execute(raw,calendar,decisions,settings=double)
    assert normal['ledger'][0]['price']<stressed['ledger'][0]['price']
    assert normal['ledger'][1]['price']>stressed['ledger'][1]['price']
    assert stressed['metrics']['total_return']<normal['metrics']['total_return']
    assert settings['execution']['slippage']==.0005


def test_missing_next_session_never_uses_a_later_open_for_entry():
    raw,calendar,plan,decisions=_scenario(sell=False)
    raw=raw.drop(index=71).reset_index(drop=True);decisions=[row for i,row in enumerate(decisions) if i!=71]
    account=_execute(raw,calendar,decisions)
    assert account['ledger']==[]
    assert any(row['reason']=='entry_plan_expired' for row in account['rejections'])


def test_incomplete_end_inventory_is_retained_not_forced_liquidation():
    raw,calendar,plan,decisions=_scenario(sell=False)
    account=_execute(raw.iloc[:74].copy(),calendar,decisions[:74])
    assert len(account['ledger'])==1
    assert account['metrics']['open_shares']>0
    assert account['actual_position']['plan_id']==plan['plan_id']
    verify_actual_account(account,calendar=calendar,decisions=decisions[:74],initial_cash=100000.,symbol='000001.SZ')


def test_completed_fee_net_outcomes_reconcile_with_final_cash_and_open_stays_unlabeled():
    raw,calendar,plan,decisions=_scenario()
    callback=_max_exit_callback({d:i for i,d in enumerate(calendar)},60,'frozen-test-policy')
    account=_execute(raw,calendar,decisions,exit_policy=callback)
    episodes=_actual_episodes(account)
    assert len(episodes)==1 and episodes[0]['complete']
    assert episodes[0]['net_pnl']==pytest.approx(account['metrics']['final_equity']-100000.)
    assert episodes[0]['net_return']<0  # unchanged price still pays real fees/slippage
    assert episodes[0]['initial_risk_cash']>0
    assert episodes[0]['r_multiple']<0
    raw,calendar,plan,decisions=_scenario(sell=False)
    open_account=_execute(raw.iloc[:74].copy(),calendar,decisions[:74],exit_policy=callback)
    unfinished=_actual_episodes(open_account)[0]
    assert unfinished['complete'] is False and unfinished['net_return'] is None
    assert unfinished['r_multiple'] is None and unfinished['remaining_shares']>0


def test_account_verifier_rejects_unreconstructable_inventory_and_same_day_fill():
    raw,calendar,plan,decisions=_scenario();account=_execute(raw,calendar,decisions)
    broken=copy.deepcopy(account);broken['daily'][2]['shares']+=100
    with pytest.raises(ValueError,match='reconstructed'):
        verify_actual_account(broken,calendar=calendar,decisions=decisions,initial_cash=100000.,symbol='000001.SZ')
    broken=copy.deepcopy(account);broken['ledger'][0]['date']=broken['ledger'][0]['signal_date']
    with pytest.raises(ValueError,match='next-verified'):
        verify_actual_account(broken,calendar=calendar,decisions=decisions,initial_cash=100000.,symbol='000001.SZ')


def test_failure_cash_is_retained_and_missing_quotes_do_not_drop_account_equity(tmp_path):
    records=[]
    for symbol,rows,status in [('000001.SZ',[{'date':'2026-01-05','equity':101000.},{'date':'2026-01-07','equity':102000.}],'completed'),
                               ('000002.SZ',[{'date':d,'equity':100000.} for d in ('2026-01-05','2026-01-06','2026-01-07')],'cash_failure')]:
        path=tmp_path/(symbol+'.json');path.write_text(json.dumps(rows))
        account={'window':'2026YTD','variant':'v4','evidence_bars':3,'slippage_multiplier':1.,'status':status,
            'artifacts':{'daily':str(path)},'metrics':{'trade_count':0,'rejection_count':0,'open_shares':0,'fees':0},'turnover':0.}
        records.append({'symbol':symbol,'accounts':[account]})
    summary=aggregate_accounts(records,['000001.SZ','000002.SZ'],100000.)[0]
    assert summary['failed_cash_accounts']==1 and summary['requested_accounts']==summary['retained_accounts']==2
    assert summary['daily'][1]['equity']==201000.
    assert summary['total_return']==pytest.approx(.01)
    assert summary['max_drawdown']==0.
    assert _cash_failure('000002.SZ','2026-01-05','2026-01-07',
        ['2026-01-05','2026-01-06','2026-01-07'],100000.,'coverage failure')['failure']['cash_retained']
