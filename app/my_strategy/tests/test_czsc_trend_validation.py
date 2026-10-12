import pytest
from my_strategy.scripts.validate_trend_consensus import verify_account,aggregate
from my_strategy.tests.test_czsc_trend_account import fixture,replay


def test_actual_ledger_verification_allows_known_open_risk_but_rejects_backdated_close():
    frame,calendar,prepared=fixture();prepared['rows'][65].update(action='SELL',emitted=True,event_id='S')
    account=replay(frame,calendar,prepared)
    assert verify_account(account,calendar[60],calendar[-1],100000.)['cash_reconciliation']
    account['ledger'][-1]['date']=calendar[65]
    with pytest.raises(AssertionError):verify_account(account,calendar[60],calendar[-1],100000.)


def test_equal_cash_aggregate_retains_failed_account_and_does_not_share_funds():
    accounts=[{'symbol':'A','window':'2026Q3','variant':'final_linear','slippage_multiplier':1,'status':'completed',
        'daily':[{'date':'2026-09-30','equity':110000.}],
        'metrics':{'trade_count':2,'fees':10.},'completed_round_trips':1},
        {'symbol':'B','window':'2026Q3','variant':'final_linear','slippage_multiplier':1,'status':'failed_cash_retained',
        'daily':[{'date':'2026-09-30','equity':100000.}],
        'metrics':{'trade_count':0,'fees':0.},'completed_round_trips':0}]
    row=aggregate(accounts,['A','B'],100000.)[0]
    assert row['total_return']==pytest.approx(.05) and row['failed_cash_accounts']==1
    assert row['completed_round_trips']==1 and row['basis']=='independent_equal_cash_no_shared_capital'


def test_stock_preparation_failure_preserves_all_requested_equal_cash_accounts(tmp_path, monkeypatch):
    from my_strategy.scripts import validate_trend_consensus as validation
    def failed(payload):raise ValueError('frozen source checksum mismatch')
    monkeypatch.setattr(validation,'_worker',failed)
    monkeypatch.setattr(validation,'_old_prefix',lambda *a:None)
    payload=({'symbol':'000001.SZ'},{'config':{'initial_cash':100000.},'data_end':'2026-09-30'},
        ['2026-09-29','2026-09-30'],['2026Q3'],'unused-model-root',str(tmp_path))
    rows,failures=validation._worker_safe(payload)
    assert len(rows)==len(failures)==12
    assert {r['variant'] for r in rows}=={'old_frozen_source_union_control','E0_rule','E1_pinned_exit',
        'final_linear','final_tree','final_mlp'}
    assert all(r['metrics']['final_equity']==100000. and not r['ledger'] for r in rows)
    assert all(len(r['daily'])==2 and r['status']=='failed_cash_retained' for r in rows)
    assert all(f['stage']=='stock_preparation' for f in failures)
    assert (tmp_path/'accounts'/'000001_SZ'/'stock_summary.json').exists()


def test_missing_account_day_preserves_cash_without_backfilling_future_gain():
    def row(symbol,daily):
        return {'symbol':symbol,'window':'2026Q3','variant':'old_frozen_source_union_control','slippage_multiplier':1,
            'status':'completed','daily':daily,'metrics':{'trade_count':0,'fees':0.},'completed_round_trips':0}
    accounts=[row('A',[{'date':'2026-09-30','equity':110000.}]),
        row('B',[{'date':'2026-09-29','equity':100000.},{'date':'2026-09-30','equity':100000.}])]
    result=aggregate(accounts,['A','B'],100000.,['2026-09-29','2026-09-30'])[0]
    assert result['daily']==[{'date':'2026-09-29','equity':200000.},{'date':'2026-09-30','equity':210000.}]
    assert result['max_drawdown']==0. and result['total_return']==pytest.approx(.05)
    assert result['daily_alignment'].startswith('initial_cash_then_carry_prior')
