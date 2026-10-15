"""Evidence-based, immutable qualification events for lightweight research.

Historical candidate evaluation is separate from enabling a paper-account policy.
Nothing here selects a current model or publishes a qualification automatically.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math

import pandas as pd

SCHEMA='light_qualification_v1'


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _stamp(value):
    result=pd.Timestamp(value)
    if pd.isna(result): raise ValueError('Missing qualification time')
    return result.tz_localize('Asia/Shanghai') if result.tzinfo is None else result.tz_convert('Asia/Shanghai')


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def verify_model(artifact):
    if artifact.get('schema')!='light_score_v2' or artifact.get('status')!='completed':
        raise ValueError('A completed light_score_v2 artifact is required')
    body={k:v for k,v in artifact.items() if k!='model_version'}
    if content_hash(body)!=artifact.get('model_version'):
        raise ValueError('Light score model identity mismatch')
    return True


def prediction_checks(artifact):
    """No test-period result controls a decision *inside* that test period."""
    checks={}
    for split in ('policy','test'):
        actual=(artifact.get('metrics') or {}).get(split) or {}
        prior=actual.get('frozen_train_prior_baseline') or {}
        linear=((artifact.get('l2_baseline') or {}).get('metrics') or {}).get(split) or {}
        counts=actual.get('class_counts') or {}
        checks[split+'_classes']=all(isinstance(counts.get(c),int) and counts[c]>=3 for c in ('down','flat','up'))
        for metric in ('log_loss','brier'):
            checks[f'{split}_{metric}_vs_prior']=all(_finite(x) for x in (actual.get(metric),prior.get(metric))) and actual[metric]<prior[metric]
            checks[f'{split}_{metric}_vs_l2']=all(_finite(x) for x in (actual.get(metric),linear.get(metric))) and actual[metric]<linear[metric]
        if artifact.get('variant')=='B':
            checks[split+'_supported_observation_patterns']=bool(artifact.get('supported_missing_patterns'))
            for pattern in artifact.get('supported_missing_patterns',[]):
                pattern_metrics=((artifact.get('missing_pattern_metrics') or {}).get(split) or {}).get(pattern) or {}
                checks[f'{split}_{pattern}_class_support']=all((pattern_metrics.get('class_counts') or {}).get(c,0)>=3 for c in ('down','flat','up'))
                for baseline in ('frozen_train_prior_baseline','l2_baseline'):
                    for metric in ('log_loss','brier'):
                        value=pattern_metrics.get(metric);reference=(pattern_metrics.get(baseline) or {}).get(metric)
                        checks[f'{split}_{pattern}_{metric}_{baseline}']=_finite(value) and _finite(reference) and value<reference
    return checks


def _execution_checks(artifact,evidence,*,forward=False):
    evidence=evidence or {}
    checks={}
    checks['evidence_present']=bool(evidence)
    checks['model_bound']=evidence.get('model_version')==artifact.get('model_version')
    checks['source_bound']=evidence.get('source_identity_sha256')==content_hash(artifact['identity']['source'])
    checks['strategy_bound']=bool(evidence.get('strategy_identity'))
    checks['cost_bound']=bool(evidence.get('cost_identity'))
    registry=evidence.get('experiment_registry') or {}
    entries=registry.get('entries') or []
    original=(artifact.get('experiment_registry') or {}).get('entries') or []
    checks['trial_registry_bound']=bool(registry) and evidence.get('experiment_registry_sha256')==content_hash(registry)
    from my_strategy.services.light_experiment_registry import preserves_experiment_trials
    try:
        checks['training_trials_preserved']=preserves_experiment_trials(registry,artifact.get('experiment_registry'))
    except (ValueError,TypeError,KeyError):
        checks['training_trials_preserved']=False
    checks['trial_budget']=len(entries)<=14 and all(sum(e.get('kind')==kind for e in entries)<=limit
        for kind,limit in [('mlp',6),('kronos',2),('policy',6)]) and all(e.get('kind') in {'mlp','kronos','policy'} for e in entries)
    checks['policy_trial_registered']=any(e.get('kind')=='policy' and e.get('experiment_id')==evidence.get('policy_trial_id')
        and e.get('strategy_identity')==evidence.get('strategy_identity') for e in entries)
    contract=evidence.get('execution_contract') or {}
    for key in ('broker_ledger_verified','next_verified_open','t_plus_one','costs_included','historical_rules_verified','corporate_actions_verified'):
        checks[key]=contract.get(key) is True
    windows=evidence.get('windows') or []
    checks['three_windows']=isinstance(windows,list) and len(windows)>=3
    symbols=set();cycles=0;previous=None;last_end=None
    checks['ordered_disjoint_windows']=bool(windows)
    checks['windows_after_model_data']=bool(windows)
    checks['frozen_before_observation']=bool(windows) if forward else True
    for index,window in enumerate(windows if isinstance(windows,list) else []):
        prefix=f'window_{index}'
        try:
            start,end=_stamp(window['start']),_stamp(window['end'])
            if start>end or (previous is not None and start<=previous): checks['ordered_disjoint_windows']=False
            if start<=_stamp(artifact['available_after']): checks['windows_after_model_data']=False
            if forward and start<=_stamp(artifact['created_at']): checks['frozen_before_observation']=False
            previous=end;last_end=end
        except (KeyError,ValueError,TypeError):
            checks['ordered_disjoint_windows']=False
        names=window.get('symbols') or []
        if isinstance(names,list): symbols.update(str(s) for s in names)
        count=window.get('completed_cycles')
        if isinstance(count,int) and not isinstance(count,bool) and count>=0: cycles+=count
        else: checks[prefix+'_cycle_count']=False
        for key in ('total_return','baseline_total_return','sharpe','baseline_sharpe','max_drawdown','turnover','maximum_exposure','coverage_fraction'):
            checks[prefix+'_'+key+'_finite']=_finite(window.get(key))
        checks[prefix+'_return_advantage']=all(_finite(window.get(k)) for k in ('total_return','baseline_total_return')) and window['total_return']>window['baseline_total_return']
        checks[prefix+'_sharpe_advantage']=all(_finite(window.get(k)) for k in ('sharpe','baseline_sharpe')) and window['sharpe']>window['baseline_sharpe']
        limits=evidence.get('frozen_limits') or {}
        for metric in ('max_drawdown','turnover','maximum_exposure'):
            checks[prefix+'_'+metric+'_limit']=_finite(window.get(metric)) and _finite(limits.get(metric)) and 0<=window[metric]<=limits[metric]
        checks[prefix+'_coverage']=_finite(window.get('coverage_fraction')) and _finite(limits.get('minimum_coverage')) and .8<=limits['minimum_coverage']<=1 and window['coverage_fraction']>=limits['minimum_coverage']
        stress=window.get('cost_stress') or {}
        checks[prefix+'_cost_stress']=all((stress.get(scale) or {}).get('passed') is True and
            _finite((stress.get(scale) or {}).get('total_return')) and _finite((stress.get(scale) or {}).get('baseline_total_return')) and
            stress[scale]['total_return']>=stress[scale]['baseline_total_return'] for scale in ('1.0','1.5','2.0'))
    checks['ten_stocks']=len(symbols)>=10
    checks['thirty_completed_cycles']=cycles>=30
    try:
        frozen=_stamp(evidence['policy_frozen_at']);evaluated=_stamp(evidence['evaluated_at'])
        checks['policy_frozen_before_windows']=bool(windows) and frozen<_stamp(windows[0]['start'])
        checks['evidence_completed_before_evaluation']=last_end is not None and last_end<evaluated
        checks['test_finished_before_evaluation']=_stamp(artifact['evaluation_end'])<evaluated
    except (KeyError,TypeError,ValueError):
        checks.update(policy_frozen_before_windows=False,evidence_completed_before_evaluation=False,test_finished_before_evaluation=False)
    return checks,dict(stocks=len(symbols),completed_cycles=cycles,windows=len(windows) if isinstance(windows,list) else 0)


def evaluate_qualification(artifact,*,policy_evidence=None,forward_evidence=None,now=None):
    """Build an event; exact model/source/cost/trial evidence is mandatory to pass.

    Passing this function does not activate an account. The caller may append the
    returned immutable event and explicitly select it for *subsequent* decisions.
    """
    verify_model(artifact)
    stamp=_stamp(now or datetime.now(timezone.utc).isoformat())
    predictions=prediction_checks(artifact)
    execution,population=_execution_checks(artifact,policy_evidence)
    execution['evaluation_not_in_future']=False
    if policy_evidence and policy_evidence.get('evaluated_at'):
        execution['evaluation_not_in_future']=_stamp(policy_evidence['evaluated_at'])<=stamp
    passed=all(predictions.values()) and all(execution.values())
    forward,population_forward=_execution_checks(artifact,forward_evidence,forward=True)
    forward['historical_policy_passed']=passed
    forward['evaluation_not_in_future']=False
    if forward_evidence and forward_evidence.get('evaluated_at'):
        forward['evaluation_not_in_future']=_stamp(forward_evidence['evaluated_at'])<=stamp
    forward_passed=all(forward.values())
    body=dict(schema=SCHEMA,model_version=artifact['model_version'],created_at=stamp.isoformat(),
        effective_after=stamp.isoformat(),source_identity_sha256=content_hash(artifact['identity']['source']),
        research_trial_eligible=True,paper_allocation_eligible=passed,
        positive_allocation_eligible=passed,policy_passed=passed,
        independent_forward_status='passed' if forward_passed else 'pending',
        prediction_checks=predictions,execution_checks=execution,forward_checks=forward,
        failed_checks=[k for k,v in {**predictions,**execution}.items() if not v],
        population=population,forward_population=population_forward,
        evidence=dict(policy=policy_evidence,forward=forward_evidence),research_only=True)
    return {**body,'event_id':content_hash(body)}


def resolve_qualification(artifact,*,mode='historical_candidate',as_of,qualification_event=None):
    verify_model(artifact)
    if mode not in {'historical_candidate','paper'}: raise ValueError('Unknown qualification mode')
    requested=_stamp(as_of)
    if len(str(as_of))==10: requested+=pd.Timedelta(hours=15)
    timing=requested>_stamp(artifact['available_after'])
    research=bool(timing and artifact.get('research_admission',{}).get('temporal_contract_passed') is True)
    paper=False;event_valid=False
    if qualification_event:
        event=dict(qualification_event)
        body={k:v for k,v in event.items() if k!='event_id'}
        if content_hash(body)!=event.get('event_id'): raise ValueError('Qualification event identity mismatch')
        if event.get('schema')!=SCHEMA or event.get('model_version')!=artifact['model_version']:
            raise ValueError('Qualification event belongs to another model')
        # Recompute evidence checks, rather than trusting a hand-written true flag.
        proof=event.get('evidence') or {}
        rebuilt=evaluate_qualification(artifact,policy_evidence=proof.get('policy'),forward_evidence=proof.get('forward'),now=event['created_at'])
        if rebuilt['event_id']!=event['event_id']: raise ValueError('Qualification event evidence mismatch')
        event_valid=True
        paper=research and event.get('paper_allocation_eligible') is True and requested>_stamp(event['effective_after'])
    allow=research if mode=='historical_candidate' else paper
    return dict(mode=mode,research_only=True,research_trial_eligible=research,paper_allocation_eligible=paper,
        allow_increase=allow,positive_allocation_eligible=allow,policy_passed=paper,
        independent_forward_status=(qualification_event or {}).get('independent_forward_status','pending'),
        qualification_event_id=(qualification_event or {}).get('event_id') if event_valid else None,
        reason='historical_candidate_evaluation_only' if allow and mode=='historical_candidate' else
            'qualified_paper_policy' if allow else 'missing_or_ineligible_qualification')
