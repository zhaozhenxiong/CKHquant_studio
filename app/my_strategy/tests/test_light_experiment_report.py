from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.light_experiment_report import POLICIES, build_experiment_report
from my_strategy.services.position_simulation import simulation_config, simulation_metrics


def completed_result(start='2020-01-01', *, scale=1., cycles=10, cash_only=False):
    """Synthetic ledgers only; no raw data, models, prediction or disk state."""
    dates=pd.bdate_range(start,periods=80).strftime('%Y-%m-%d').tolist()
    result={'status':'completed','currency':'CNY','start':dates[0],'end':dates[-1],
        'symbols':['A','B','C','D','E'],'result_id':f'fixture-{start}-{scale}',
        'identity':{'cash':1_000_000.,'sources':{'fixture':'frozen'},'score':'fixed_mlp',
                    'prediction':{'lookback':128,'horizon':5}},
        'prediction_coverage':{'requested':20,'succeeded':19,'unusable':1,'failed':0,'budget_skipped':0},
        'scheduler':{'complete_through':dates[-1],'evaluation_gap':False},'comparisons':{}}
    for index,policy in enumerate(POLICIES):
        cfg=simulation_config({'initial_cash':1_000_000.,'policy':policy,'cost_stress':scale})
        returns=np.zeros(80) if cash_only else .0002+np.sin(np.arange(80)*.7+index*.03)*.006-index*.00005-scale*.000001
        equities=1_000_000.*np.cumprod(1+returns)
        daily=[{'date':date,'equity':float(equity),'exposure':0. if cash_only else .2,
                'positions':{},'stale_symbols':[]} for date,equity in zip(dates,equities)]
        trades=[] if cash_only else [dict(fee=scale*5.,price=20.,shares=100,slippage_cost=scale*.1)]*20
        completed=[] if cash_only else [{'pnl':1.,'holding_sessions':3}]*cycles
        metrics=simulation_metrics(daily,trades,1_000_000.,cfg,completed)
        result['comparisons'][policy]={'status':'completed','metrics':metrics,'daily':daily,
            'trades':trades,'orders':[{'side':'SELL','status':'pending','attempts':[{'status':'rejected'}]}],
            'lineage':{'config':cfg}}
    return result


def test_report_is_paired_deterministic_and_does_not_select_or_mutate():
    result=completed_result();before=deepcopy(result)
    report=build_experiment_report(result,bootstrap_samples=40)
    assert report==build_experiment_report(result,bootstrap_samples=40)
    assert result==before
    assert report['comparison_issues']==[]
    assert report['status']=='pending'
    assert report['readiness']['cost_stress_complete'] is False
    assert report['research_sufficiency']['checks']['three_nonoverlapping_windows'] is False
    assert report['qualification_granted'] is False
    assert report['best_model_selected'] is None
    for policy,comparison in report['paired_differences'].items():
        assert comparison['status']=='completed'
        assert comparison['method']=='paired_moving_date_blocks'
        assert comparison['block_sessions']==5
        assert comparison['cumulative_net_return_difference']['valid_draws']==40
        expected=result['comparisons'][policy]['metrics']['total_return']-result['comparisons']['structure_rules']['metrics']['total_return']
        assert comparison['cumulative_net_return_difference']['estimate']==pytest.approx(expected)
    metrics=report['metrics']['structure_rules']
    assert metrics['fills']==20 and metrics['completed_cycles']==10
    assert metrics['rejected_attempts']==1 and metrics['pending_exits']==1
    assert metrics['average_exposure']==pytest.approx(.2)
    delta=report['fusion_paired_differences']['structure_kronos_l2']['cumulative_net_return_difference']
    assert delta['estimate']==pytest.approx(result['comparisons']['structure_kronos_score']['metrics']['total_return']-
        result['comparisons']['structure_kronos_l2']['metrics']['total_return'])


def test_complete_report_requires_actual_cost_replay_and_sample_sufficiency():
    base=completed_result()
    stresses={scale:completed_result(scale=float(scale)) for scale in ('1.0','1.5','2.0')}
    windows=[base,completed_result('2021-01-01'),completed_result('2022-01-01')]
    report=build_experiment_report(base,cost_stress_results=stresses,window_results=windows,bootstrap_samples=20)
    assert report['status']=='completed'
    assert all(report['readiness'].values())
    assert report['research_sufficiency']['completed_cycles_by_policy']==dict.fromkeys(POLICIES,30)
    assert report['cost_stress']['2.0']['metrics']['structure_rules']['total_fees']==200.
    assert report['qualification_granted'] is False


@pytest.mark.parametrize('change,issue',[
    (lambda r:r.pop('currency'),'currency_explicit'),
    (lambda r:r.update(status='partial'),'completed'),
    (lambda r:r['scheduler'].update(evaluation_gap=True),'no_evaluation_gap'),
    (lambda r:r['prediction_coverage'].update(succeeded=18),'coverage_conservation'),
    (lambda r:r['comparisons']['structure_rules']['metrics'].update(initial_cash=500_000.),'structure_rules:cash_matches'),
    (lambda r:r['comparisons']['structure_rules']['metrics'].update(total_fees=0.),'structure_rules:ledger_fees'),
    (lambda r:r['comparisons']['structure_rules']['daily'][0].update(equity='invalid'),'structure_rules:equity_valid'),
    (lambda r:r['comparisons']['structure_kronos']['daily'][0].update(date='2020-01-02'),'structure_kronos:same_daily_dates'),
    (lambda r:r['comparisons']['structure_rules']['daily'][1].update(stale_symbols=['A']),'structure_rules:valuations_current'),
])
def test_invalid_comparison_contract_stays_pending(change,issue):
    result=completed_result();change(result)
    report=build_experiment_report(result,bootstrap_samples=1)
    assert issue in report['comparison_issues']
    assert report['paired_differences']=={}
    assert report['status']=='pending'


def test_cost_stress_cannot_change_policy_or_sources_or_relabel_baseline():
    result=completed_result()
    wrong=completed_result(scale=1.5)
    wrong['comparisons']['structure_kronos']['lineage']['config']['initial_weight']=.19
    report=build_experiment_report(result,cost_stress_results={'1.5':wrong,'2.0':result},bootstrap_samples=1)
    assert 'same_policy_contracts' in report['cost_stress']['1.5']['issues']
    assert 'stress_factor' in report['cost_stress']['2.0']['issues']
    changed=completed_result(scale=1.5);changed['identity']['sources']={'fixture':'changed'}
    report=build_experiment_report(result,cost_stress_results={'1.5':changed},bootstrap_samples=1)
    assert 'same_input_sources' in report['cost_stress']['1.5']['issues']


def test_windows_must_not_overlap_or_change_frozen_model():
    base=completed_result()
    windows=[base,completed_result(),completed_result('2022-01-01')]
    windows[-1]['identity']['score']='selected_after_inspection'
    report=build_experiment_report(base,window_results=windows,bootstrap_samples=1)
    suff=report['research_sufficiency']
    assert suff['status']=='pending'
    assert suff['checks']['three_nonoverlapping_windows'] is False
    assert suff['checks']['all_windows_valid'] is False
    assert any(x['reason']=='frozen_study_contract_mismatch' for x in suff['issues'])


def test_cash_only_accounts_do_not_invent_sharpe_or_cycles():
    report=build_experiment_report(completed_result(cash_only=True),bootstrap_samples=12)
    assert report['metrics']['structure_rules']['sharpe'] is None
    assert report['metrics']['structure_rules']['completed_cycles']==0
    assert report['readiness']['paired_sharpe_estimable'] is False
    for comparison in report['paired_differences'].values():
        assert comparison['sharpe_difference']['valid_draws']==0
        assert comparison['sharpe_difference']['estimate'] is None
        assert comparison['cumulative_net_return_difference']['estimate']==0.


@pytest.mark.parametrize('kwargs',[{'bootstrap_samples':501},{'bootstrap_samples':True},{'block_size':4},{'block_size':5.},{'seed':-1}])
def test_computation_budget_is_bounded(kwargs):
    with pytest.raises(ValueError):build_experiment_report(completed_result(),**kwargs)
