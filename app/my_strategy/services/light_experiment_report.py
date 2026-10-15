"""Bounded, read-only paired research reports from completed account ledgers.

No model calls, selection, promotion, files, or external data. Confidence intervals
describe this historical sample; they do not certify independent forward returns.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

SCHEMA='light_experiment_report_v1'
POLICIES=('structure_rules','structure_kronos','structure_kronos_l2','structure_kronos_score')
FEE_KEYS=('commission','min_commission','slippage','transfer_fee','cost_version','cost_stress')
SUMMARY_KEYS=('total_return','sharpe','sharpe_status','sharpe_reasons','max_drawdown','turnover',
              'total_fees','slippage_cost','average_exposure','maximum_exposure','completed_cycles',
              'fills','rejected_attempts','pending_exits','trading_sessions','ending_equity')


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _finite(value):
    return isinstance(value,(int,float,np.number)) and not isinstance(value,(bool,np.bool_)) and math.isfinite(value)


def _close(a,b):
    return _finite(a) and _finite(b) and math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-8)


def _inspect(result):
    checks={};issues=[];series={};summaries={};policy_contracts={}
    if not isinstance(result,dict): return dict(checks={'result_object':False},issues=['result_object'],series={},summaries={})
    accounts=result.get('comparisons') or {}
    checks['completed']=result.get('status')=='completed'
    checks['four_accounts']=all(policy in accounts for policy in POLICIES)
    identity=result.get('identity') or {}
    currency=result.get('currency',identity.get('currency'))
    checks['currency_explicit']=isinstance(currency,str) and len(currency)==3 and currency.isupper()
    initial=identity.get('cash',result.get('initial_cash'))
    checks['positive_initial_cash']=_finite(initial) and initial>0
    start,end=str(result.get('start',''))[:10],str(result.get('end',''))[:10]
    try:
        checks['date_range']=len(start)==10 and len(end)==10 and pd.Timestamp(start)<=pd.Timestamp(end)
    except (TypeError,ValueError): checks['date_range']=False
    schedule=result.get('scheduler') or {}
    checks['no_evaluation_gap']=not schedule.get('evaluation_gap',False)
    checks['evaluation_through_end']=not schedule or schedule.get('complete_through')==end
    coverage=result.get('prediction_coverage') or {}
    if coverage:
        counts=[coverage.get(k) for k in ('succeeded','unusable','failed','budget_skipped')]
        checks['coverage_conservation']=all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in counts) and sum(counts)==coverage.get('requested')
        checks['no_prediction_deferred_or_failure']=coverage.get('failed')==0 and coverage.get('budget_skipped')==0
    reference_dates=None;reference_fees=None;reference_frequency=None
    for policy in POLICIES:
        account=accounts.get(policy)
        if not isinstance(account,dict): continue
        metrics=account.get('metrics') or {};config=(account.get('lineage') or {}).get('config') or {}
        summary={key:metrics.get(key) for key in SUMMARY_KEYS}
        policy_contracts[policy]={key:value for key,value in config.items() if key!='cost_stress'}
        daily=account.get('daily') or [];trades=account.get('trades') or []
        dates=[str(row.get('date',''))[:10] for row in daily]
        equities=np.asarray([row.get('equity') if _finite(row.get('equity')) else np.nan for row in daily],dtype=float)
        prefix=policy+':'
        checks[prefix+'account_completed']=account.get('status')=='completed'
        checks[prefix+'cash_matches']=_close(metrics.get('initial_cash'),initial) and _close(config.get('initial_cash'),initial)
        checks[prefix+'currency_matches']=config.get('currency',currency)==currency and checks['currency_explicit']
        checks[prefix+'date_order']=bool(dates) and dates==sorted(set(dates)) and all(start<=day<=end for day in dates)
        checks[prefix+'last_date_matches']=bool(dates) and dates[-1]==end
        checks[prefix+'equity_valid']=bool(len(equities)) and bool(np.isfinite(equities).all()) and bool((equities>0).all())
        checks[prefix+'valuations_current']=all(not row.get('stale_symbols') for row in daily)
        checks[prefix+'session_count']=metrics.get('trading_sessions')==len(daily)
        checks[prefix+'fills_count']=metrics.get('fills')==len(trades)
        fees=[row.get('fee') for row in trades]
        checks[prefix+'ledger_fees']=all(_finite(fee) and fee>=0 for fee in fees) and _close(metrics.get('total_fees'),sum(fees) if all(_finite(fee) for fee in fees) else None)
        fee_contract={key:config.get(key) for key in FEE_KEYS}
        checks[prefix+'fee_contract_explicit']=bool(config.get('cost_version')) and all(_finite(config.get(key)) and config[key]>=0 for key in FEE_KEYS if key!='cost_version')
        if reference_fees is None: reference_fees=fee_contract
        checks[prefix+'same_fee_contract']=fee_contract==reference_fees
        frequency=(config.get('annualization'),config.get('risk_free_annual'))
        checks[prefix+'frequency_explicit']=_finite(frequency[0]) and frequency[0]>0 and _finite(frequency[1]) and frequency[1]>-1
        if reference_frequency is None: reference_frequency=frequency
        checks[prefix+'same_frequency']=frequency==reference_frequency
        if reference_dates is None: reference_dates=dates
        checks[prefix+'same_daily_dates']=dates==reference_dates
        if checks[prefix+'equity_valid'] and checks['positive_initial_cash']:
            returns=equities/np.r_[float(initial),equities[:-1]]-1
            checks[prefix+'return_reconciles']=_close(metrics.get('total_return'),equities[-1]/float(initial)-1)
            series[policy]=returns
        else: checks[prefix+'return_reconciles']=False
        if summary['rejected_attempts'] is None:
            summary['rejected_attempts']=sum(attempt.get('status')=='rejected' for order in account.get('orders',[]) for attempt in order.get('attempts',[]))
        if summary['pending_exits'] is None:
            summary['pending_exits']=sum(order.get('side')=='SELL' and order.get('status') in {'queued','pending','partial','partially_filled'} for order in account.get('orders',[]))
        summary.update(currency=currency,initial_cash=initial,first_session=dates[0] if dates else None,last_session=dates[-1] if dates else None)
        summaries[policy]=summary
    issues=[name for name,passed in checks.items() if not passed]
    return dict(checks=checks,issues=issues,series=series,summaries=summaries,dates=reference_dates or [],
        fee_contract=reference_fees,currency=currency,initial_cash=initial,frequency=reference_frequency,
        source_identity=identity.get('sources'),prediction_contract=identity.get('prediction'),score_identity=identity.get('score'),
        policy_contracts=policy_contracts,symbols=sorted(result.get('symbols') or []))


def _sharpe(returns,annual,rf):
    if len(returns)<60:return None
    excess=returns-((1+rf)**(1/annual)-1)
    deviation=float(np.std(excess,ddof=1))
    return float(np.sqrt(annual)*np.mean(excess)/deviation) if np.isfinite(deviation) and deviation>1e-14 else None


def _interval(draws,point):
    return dict(estimate=point,lower=float(np.quantile(draws,.025)) if draws else None,
        upper=float(np.quantile(draws,.975)) if draws else None,confidence=.95,valid_draws=len(draws))


def _paired(candidate,baseline,*,annual,rf,draws,block_size,seed):
    if len(candidate)!=len(baseline):raise ValueError('Paired dates must have equal observations')
    size=len(candidate)
    if size<max(60,block_size):return dict(status='pending',reason='fewer_than_60_paired_sessions',sessions=size)
    rng=np.random.default_rng(seed)
    differences=[];sharpes=[]
    count=int(np.ceil(size/block_size))
    for _ in range(draws):
        starts=rng.integers(0,size-block_size+1,size=count)
        indexes=(starts[:,None]+np.arange(block_size)).reshape(-1)[:size]
        a,b=candidate[indexes],baseline[indexes]
        differences.append(float(np.prod(1+a)-np.prod(1+b)))
        ash,bsh=_sharpe(a,annual,rf),_sharpe(b,annual,rf)
        if ash is not None and bsh is not None:sharpes.append(ash-bsh)
    actual_a,actual_b=_sharpe(candidate,annual,rf),_sharpe(baseline,annual,rf)
    return dict(status='completed' if len(sharpes)==draws else 'partial',sessions=size,
        method='paired_moving_date_blocks',block_sessions=block_size,draws=draws,seed=seed,
        cumulative_net_return_difference=_interval(differences,float(np.prod(1+candidate)-np.prod(1+baseline))),
        sharpe_difference=_interval(sharpes,actual_a-actual_b if actual_a is not None and actual_b is not None else None),
        limitation='Historical paired resampling; zero-variance Sharpe draws remain unavailable, not zero.')


def _window_sufficiency(results):
    intervals=[];stocks=set();cycles={p:0 for p in POLICIES};failures=[];reference=None
    for index,result in enumerate(results):
        inspection=_inspect(result)
        contract={key:inspection.get(key) for key in ('currency','initial_cash','fee_contract','prediction_contract','score_identity','policy_contracts','symbols')}
        if reference is None:reference=contract
        elif contract!=reference:failures.append(dict(window=index,reason='frozen_study_contract_mismatch'))
        if inspection['issues']:failures.append(dict(window=index,reason='window_contract_failed',checks=inspection['issues']))
        try:
            start,end=pd.Timestamp(result['start']),pd.Timestamp(result['end'])
            if start>end:raise ValueError('reversed window')
            intervals.append((start,end))
        except (KeyError,ValueError,TypeError):failures.append(dict(window=index,reason='invalid_window_dates'))
        stocks.update(result.get('symbols') or [])
        for policy in POLICIES:
            count=inspection['summaries'].get(policy,{}).get('completed_cycles')
            if isinstance(count,int) and not isinstance(count,bool) and count>=0:cycles[policy]+=count
            else:failures.append(dict(window=index,reason='cycle_count_missing',policy=policy))
    intervals.sort()
    disjoint=all(intervals[i][0]>intervals[i-1][1] for i in range(1,len(intervals)))
    checks=dict(five_stocks=len(stocks)>=5,three_nonoverlapping_windows=len(intervals)>=3 and disjoint,
        all_windows_valid=not failures,thirty_cycles_each=all(count>=30 for count in cycles.values()))
    return dict(status='sufficient_for_further_review' if all(checks.values()) else 'pending',checks=checks,
        stocks=len(stocks),windows=len(intervals),completed_cycles_by_policy=cycles,issues=failures,
        requirements=dict(minimum_stocks=5,minimum_windows=3,minimum_completed_cycles_per_policy=30),
        qualification_granted=False)


def build_experiment_report(result,*,cost_stress_results=None,window_results=None,
                            bootstrap_samples=300,block_size=5,seed=42):
    """Summarize four complete accounts, paired differences, costs and sufficiency.

    Cost inputs map '1.0'/'1.5'/'2.0' to independently replayed results. This
    function never manufactures a stress scenario by subtracting estimated fees.
    `window_results` must be predeclared nonoverlapping complete result objects.
    """
    if isinstance(bootstrap_samples,bool) or not isinstance(bootstrap_samples,int) or not 1<=bootstrap_samples<=500:
        raise ValueError('Bootstrap draws must be an integer in 1..500')
    if isinstance(block_size,bool) or not isinstance(block_size,int) or block_size!=5:raise ValueError('Frozen block size is five sessions')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:raise ValueError('Bootstrap seed must be a nonnegative integer')
    inspected=_inspect(result);aligned=not inspected['issues']
    paired={};fusion_comparisons={}
    if aligned:
        annual,rf=inspected['frequency']
        for policy in POLICIES[1:]:
            paired[policy]=_paired(inspected['series'][policy],inspected['series']['structure_rules'],
                annual=annual,rf=rf,draws=bootstrap_samples,block_size=block_size,seed=seed)
        for baseline in ('structure_kronos','structure_kronos_l2'):
            fusion_comparisons[baseline]=_paired(inspected['series']['structure_kronos_score'],inspected['series'][baseline],
                annual=annual,rf=rf,draws=bootstrap_samples,block_size=block_size,seed=seed)
    stress_inputs={str(key):value for key,value in (cost_stress_results or {}).items()}
    stress_inputs.setdefault('1.0',result)
    stress={}
    for scale in ('1.0','1.5','2.0'):
        candidate=stress_inputs.get(scale)
        if candidate is None:
            stress[scale]=dict(status='pending',issues=['actual_cost_replay_missing']);continue
        analysis=_inspect(candidate)
        checks=dict(actual_replay_complete=not analysis['issues'],same_dates=analysis.get('dates')==inspected.get('dates'),
            same_currency=analysis.get('currency')==inspected.get('currency') and bool(analysis.get('currency')),
            same_initial_cash=_close(analysis.get('initial_cash'),inspected.get('initial_cash')),
            same_input_sources=analysis.get('source_identity')==inspected.get('source_identity') and analysis.get('source_identity') is not None,
            same_prediction_contract=analysis.get('prediction_contract')==inspected.get('prediction_contract') and analysis.get('prediction_contract') is not None,
            same_score_identity=analysis.get('score_identity')==inspected.get('score_identity'),
            same_policy_contracts=analysis.get('policy_contracts')==inspected.get('policy_contracts'),
            same_symbols=analysis.get('symbols')==inspected.get('symbols'),
            stress_factor=bool(analysis.get('fee_contract')) and _close(analysis['fee_contract'].get('cost_stress'),float(scale)))
        actual_fee={k:v for k,v in (analysis.get('fee_contract') or {}).items() if k!='cost_stress'}
        reference_fee={k:v for k,v in (inspected.get('fee_contract') or {}).items() if k!='cost_stress'}
        checks['same_underlying_fees']=actual_fee==reference_fee and bool(actual_fee)
        stress[scale]=dict(status='completed' if all(checks.values()) else 'pending',checks=checks,
            issues=analysis['issues']+[k for k,v in checks.items() if not v],metrics=analysis.get('summaries',{}),
            result_id=candidate.get('result_id'))
    sufficient=_window_sufficiency(list(window_results) if window_results is not None else [result])
    conditions=dict(comparison_contract=aligned,cost_stress_complete=all(item['status']=='completed' for item in stress.values()),
        research_population=sufficient['status']=='sufficient_for_further_review',
        paired_sharpe_estimable=bool(paired) and all(item['status']=='completed' for item in paired.values()))
    report=dict(schema=SCHEMA,status='completed' if all(conditions.values()) else 'pending',
        report_generated=True,result_id=result.get('result_id'),currency=inspected.get('currency'),
        start=result.get('start'),end=result.get('end'),metrics=inspected['summaries'],
        comparison_checks=inspected['checks'],comparison_issues=inspected['issues'],readiness=conditions,
        paired_vs='structure_rules',paired_differences=paired,
        fusion_candidate='structure_kronos_score',fusion_paired_differences=fusion_comparisons,
        cost_stress=stress,research_sufficiency=sufficient,
        parameters=dict(seed=seed,block_sessions=block_size,bootstrap_samples=bootstrap_samples),
        qualification_granted=False,best_model_selected=None,
        limitations=['历史研究报告不授予资格，不选择最高夏普策略。',
            '同频每日净值包含空仓日；成交笔数与完整持仓周期分别报告。',
            '日期块抽样仅描述本窗口不确定性，不能消除多次择优、幸存者或Kronos预训练范围未知的影响。'])
    report['report_id']=_hash(report)
    return report
