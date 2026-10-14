"""Actual flat-cash candidates and paired actual-held liquidation targets.

The account adapter is shared with runtime. Outcomes are calculated from
Broker fills, never from a theoretical signal price or hypothetical inventory.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from my_strategy.core.run_context import stable_hash
from my_strategy.execution.cost_model import CostModel
from my_strategy.services.czsc_trend_support import _save, _file_hash
from my_strategy.services.czsc_trend_account import (replay_trend_account,
    clone_candidate_account, clone_holding_branches, prepare_account_context)
from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
from my_strategy.services.czsc_trend_features import (ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS,
    L1_MA_EXIT_COLUMNS, entry_feature_row, holding_feature_row, event_prefix_rows,original_evidence_events)
from my_strategy.services.czsc_trend_labels import episode_outcome, exit_outcome, target_contract
from my_strategy.services.czsc_trend_interval_model import schema_hash
from my_strategy.services.czsc_trend_model import bind_trend_inputs
from my_strategy.services.czsc_trend_contracts import ENTRY_HEADS, L2_HOLDING_HEADS, validate_holding_population
from my_strategy.services.czsc_account_artifacts import write_account_artifacts


def make_contract(head, config, calendar, stage):
    columns = L1_MA_EXIT_COLUMNS if head == 'ma_exit' else ENTRY_FEATURE_COLUMNS if head in ENTRY_HEADS else EXIT_FEATURE_COLUMNS
    value = target_contract(head=head, config=config, market_dates=calendar,
        source_snapshot_hash=config['source_snapshot_hash'], feature_schema_hash=schema_hash(columns,head),
        continuation_policy_hash=config[stage+'_policy_hash'], initial_cash=config['initial_cash'], stage=stage,
        candidate_policy_hash=config['candidate_policy_hash'])
    value['model_input_schema_hash']=schema_hash(columns,head)
    value['output_type']='continuous' if head in {'return','exit_advantage'} else 'classification'
    value['contract_hash']=stable_hash({k:v for k,v in value.items() if k!='contract_hash'})
    return value


def training_frame(dataset, head, config, calendar, stage):
    key='entries' if head in ENTRY_HEADS else 'exits'; frame=dataset[key].copy()
    if head=='exit_advantage':frame['label']=frame['exit_advantage']
    if head in {'return','loss'}:
        returns=pd.to_numeric(frame['net_return'],errors='coerce')
        complete=frame.label_available.fillna(False) & frame.label_end.notna() & np.isfinite(returns)
        frame['label']=returns.where(complete) if head=='return' else returns.le(0).astype(float).where(complete)
    contract=make_contract(head,config,calendar,stage)
    population=dataset.get('reference_population')
    if population and head in ENTRY_HEADS:
        contract['reference_population']=copy.deepcopy(population)
        contract['reference_population_hash']=stable_hash(population)
        contract['contract_hash']=stable_hash({k:v for k,v in contract.items() if k!='contract_hash'})
    if head in L2_HOLDING_HEADS:
        held=validate_holding_population(dataset.get('holding_population'))
        if stage!='E1' or population or dataset.get('reference_mode')!='E1':
            raise ValueError('L2 exits cannot train from final-entry gated accounts or candidate clones')
        contract['holding_population']=copy.deepcopy(held)
        contract['holding_population_hash']=stable_hash(held)
        contract['contract_hash']=stable_hash({k:v for k,v in contract.items() if k!='contract_hash'})
        validate_holding_population(held,contract=contract)
    frame.attrs['label_contract']=contract
    return frame


def label_maturity_counts(dataset, *, source_end, population_end=None):
    """Separate observed frozen-source outcomes from those known by the fold."""
    cutoff=population_end or source_end
    result={}
    for key in ('entries','exits'):
        frame=dataset[key]
        ends=pd.to_datetime(frame.label_end,errors='coerce').dt.strftime('%Y-%m-%d')
        observed=frame.label_available.fillna(False) & ends.notna() & ends.le(source_end)
        at_population=observed & ends.le(cutoff)
        result[key+'_mature_at_frozen_source']=int(observed.sum())
        result[key+'_mature_by_population_end']=int(at_population.sum())
        if key=='entries':
            positive=frame.label.eq(1)
            result['entries_positive_at_frozen_source']=int((positive & observed).sum())
            result['entries_positive_by_population_end']=int((positive & at_population).sum())
    return result


def complete_proceeds(ledger, shares):
    remaining=int(shares);proceeds=0.;end=None
    for row in ledger:
        if row['action']=='BUY':raise ValueError('remaining-inventory branch added shares')
        remaining-=int(row['shares']);proceeds+=float(row['cash_flow'])
        if remaining<0:raise ValueError('branch sold more than actual remaining shares')
        if remaining==0:end=row['date'];break
    return {'proceeds':proceeds,'date':end,'remaining':remaining}


def _reject_ambiguous_price_basis(outcome,frame,start):
    end=outcome.get('label_end')
    if not end or 'wyckoff_corporate_action_suspected' not in frame:return outcome
    scope=frame[(frame.date>=start)&(frame.date<=end)]
    ambiguous=scope.wyckoff_corporate_action_suspected.fillna(False).eq(True).any()
    if ambiguous:
        outcome={**outcome,'label':np.nan,'label_available':False,'label_reason':'price_basis_or_corporate_action_ambiguous'}
        if 'exit_advantage' in outcome:outcome['exit_advantage']=np.nan
    return outcome


def _entry_values(feature,state,candidate,plan,prefix,config):
    values=entry_feature_row(feature,state,candidate,plan,initial_cash=config['initial_cash'],event_prefix=prefix)
    # A source can abstain; it is not a fabricated zero input. The candidate's
    # original source membership and current causal availability stay explicit.
    values['input_eligible']=bool(candidate.get('liquidity_qualified', feature.get('liquidity_pool_eligible',False))
        and any(values.get(source+'_input_available',0) for source in candidate['sources']))
    values.update({key: value for key, value in feature.items() if key.startswith('liquidity_')})
    return values


def pre_gate_attempt(diagnostic, symbol, config):
    """Use the actual account before quality admission, never its pending BUY."""
    state=copy.deepcopy(diagnostic.get('pre_gate_state') or diagnostic.get('reference_state'))
    if not isinstance(state,dict) or state.get('shares')!=0 or state.get('pending_intent'):
        raise ValueError('training attempt requires its genuinely flat pre-gate account state')
    day=str(diagnostic['date'])[:10]
    snapshot=diagnostic['snapshot'];candidate=copy.deepcopy(diagnostic['candidate']);plan=candidate.get('entry_plan') or {}
    if diagnostic.get('risk_qualified') is not True or plan.get('entry_allowed') is not True:
        raise ValueError('training attempt must be risk-qualified before its quality gate')
    if str(state.get('last_close_date'))[:10]!=day or str(snapshot.get('date'))[:10]!=day:
        raise ValueError('pre-gate attempt state/date mismatch')
    observed=state.get('current_snapshot') or {}
    if any(observed.get(key)!=snapshot.get(key) for key in ('cash','shares','close','date')):
        raise ValueError('pre-gate attempt differs from the observed actual account snapshot')
    originals=sorted(set(diagnostic.get('original_opportunity_ids') or candidate.get('original_opportunity_ids') or candidate.get('candidate_event_ids') or []))
    if not originals:
        raise ValueError('daily training attempt requires original opportunity identity')
    group=diagnostic.get('economic_group_id') or stable_hash({'family':config['version'],
        'symbol':symbol,'original_opportunity_ids':originals})
    identifier=diagnostic.get('attempt_id')
    flat_hash=diagnostic.get('flat_state_hash') or candidate.get('flat_state_hash')
    if not identifier or not flat_hash:
        raise ValueError('daily training attempt requires its causal flat-state and online attempt hashes')
    return state,candidate,{'attempt_id':identifier,'opportunity_id':group,'economic_group_id':group,
        'candidate_id':identifier,'original_opportunity_ids':originals,'original_event_ids':originals,'candidate_signal_date':day,
        'flat_state_hash':flat_hash,
        'reference_state_hash':stable_hash(state),'reference_prefix_ledger_hash':stable_hash(state.get('ledger') or []),
        'entry_plan_hash':stable_hash(plan),'plan_id':plan['plan_id'],'actual_flat_reference':True,
        'reference_allowed':diagnostic.get('allowed') is True,'reference_gate_status':(diagnostic.get('evaluation') or {}).get('status'),
        'reference_gate_reason':diagnostic.get('reason')}


def build_account_dataset(frame, calendar, native_pack, trend_probabilities, config, *, stage='E0',
                          exit_policy=None, directory=None, run_id='', progress=None,
                          reference_entry_evaluator=None, reference_population=None, population_end=None):
    if stage not in {'E0','E1'} or (stage=='E0' and exit_policy is not None):
        raise ValueError('E0 cannot depend on a learned exit or high-return entry head')
    frame=bind_trend_inputs(frame,calendar,config)
    account_context=prepare_account_context(frame,calendar,config)
    prepared=prepare_trend_decisions(frame,calendar,evidence_pack=native_pack,
        trend_probabilities=trend_probabilities,config=config,mode='E0')
    if stage=='E0' and reference_entry_evaluator is not None:
        raise ValueError('E0 cannot read any L2 entry head')
    reference_mode='final' if reference_entry_evaluator is not None else stage
    reference=replay_trend_account(frame,calendar,prepared,config=config,exit_policy=exit_policy,
        entry_evaluator=reference_entry_evaluator,start=config['feature_start'],
        run_id=run_id+':reference',mode=reference_mode,account_context=account_context)
    byday={r['date']:r for r in frame.to_dict('records')}
    eventrows=event_prefix_rows(frame,original_evidence_events(prepared),calendar)
    eventmap=dict(zip(frame.date.astype(str),eventrows));midx={d:i for i,d in enumerate(calendar)}
    costs=reference['cost_model']
    cost=CostModel(commission=costs['commission'],stamp_tax=costs['stamp_tax'],slippage=costs['slippage'],
        min_commission=costs['min_commission'],version='flat_v1',impact_tiers=())
    entries=[];episodes=[];entrymap={};clone_audits=[];seen=set();atomic_first={};counts={'raw_risk_opportunities':0,'nonadmitted_opportunities':0,'flat_candidates':0,'candidate_rejections':0,'liquidity_rejected_attempts':0}
    for diagnostic in reference['entry_diagnostics']:
        day=str(diagnostic['date'])[:10]
        if day<config['feature_start'] or population_end and day>population_end:continue
        if diagnostic.get('risk_qualified') is not True:
            counts['nonadmitted_opportunities']+=1;continue
        state,candidate,identity=pre_gate_attempt(diagnostic,str(frame.symbol.iloc[0]),config)
        for atom in identity['original_event_ids']:atomic_first.setdefault(atom,day)
        identity['original_event_first_pre_gate_dates']=json.dumps(
            {atom:atomic_first[atom] for atom in identity['original_event_ids']},sort_keys=True,separators=(',',':'))
        plan=candidate['entry_plan'];observed=diagnostic['snapshot']
        values=_entry_values(byday[day],observed,candidate,plan,eventmap[day],config)
        identifier=identity['attempt_id']
        if identifier in seen:raise ValueError('duplicate same-close training attempt identity')
        seen.add(identifier);counts['raw_risk_opportunities']+=1;counts['flat_candidates']+=1
        counts['nonadmitted_opportunities']+=int(not diagnostic.get('allowed'))
        counts['liquidity_rejected_attempts']+=int(diagnostic.get('liquidity_qualified') is not True)
        meta={**identity,'source':'union','stage':stage,'sources':','.join(candidate['sources']),
            'candidate_policy_hash':config['candidate_policy_hash'],'continuation_policy_hash':config[stage+'_policy_hash'],
            'candidate_reference_account':True,'reference_population_hash':stable_hash(reference_population or {}),
            'label_clone_scope':'force_once_from_actual_flat_pre_gate_then_frozen_E1',
            'reference_entry_gate_applied':reference_mode=='final'}
        if day==frame.date.iloc[-1]:
            outcome={'label':np.nan,'label_available':False,'label_end':None,'label_reason':'next_session_not_observed'}
            entries.append({**values,**meta,**outcome});episodes.append({**meta,**outcome});continue
        cloned=clone_candidate_account(frame,calendar,prepared,state,candidate,config,exit_policy,run_id+':'+identifier[:12],account_context=account_context)
        snapshots=cloned['holding_snapshots'];actual_stop=snapshots[0]['stop_price'] if snapshots else plan.get('stop_price')
        outcome=episode_outcome(cloned['ledger'],stop_price=actual_stop,cost_model=cost,config=config)
        outcome=_reject_ambiguous_price_basis(outcome,frame,outcome.get('entry_date') or day)
        audit={'id':identifier,'signal_date':day,'ledger':cloned['ledger'],'rejections':cloned['rejections'],
            'open_position':cloned['actual_position'],'outcome':outcome,'daily':cloned['daily'],
            'original_risk_plan':plan,'actual_initial_stop':actual_stop,'reference_state_hash':meta['reference_state_hash']}
        clone_audits.append(audit)
        row={**values,**meta,**outcome,'clone_ledger_hash':stable_hash(cloned['ledger']),
            'clone_rejection_count':len(cloned['rejections']),'original_stop_price':actual_stop}
        entries.append(row);episodes.append({**meta,**outcome,'ledger_hash':row['clone_ledger_hash']})
        counts['candidate_rejections']+=int(not cloned['ledger'])
        entrymap[plan['plan_id']]=values
    # Actual reference entries may be admitted on an older still-valid candidate.
    # Their entry feature identity comes from its actual same-close admission.
    for diagnostic in reference['entry_diagnostics']:
        candidate=diagnostic['candidate'];plan=candidate['entry_plan'];day=diagnostic['date']
        if plan:
            entrymap[plan['plan_id']]=_entry_values(byday[day],diagnostic['snapshot'],candidate,plan,eventmap[day],config)
    exits=[];pairs=[]
    for i,snapshot in enumerate(reference['holding_snapshots']):
        day=snapshot['date'];prefix=copy.deepcopy(snapshot['holding_prefix']);prefix.update(eventmap[day])
        if population_end and day>population_end:continue
        original=entrymap.get(snapshot['plan_id'],{})
        values=holding_feature_row(byday[day],snapshot,entry_values=original,prefix=prefix,market_index=midx)
        values['input_eligible']=bool(byday[day].get('model_input_eligible'))
        following=reference['ledger'][int(snapshot['ledger_entries_seen']):]
        continued=complete_proceeds([r for r in following if r['action']!='BUY'],snapshot['shares'])
        forced=clone_holding_branches(frame,calendar,prepared,snapshot,config,exit_policy,run_id+':forced:'+day,forced=True,account_context=account_context) if day<frame.date.iloc[-1] else {
            'ledger':[],'rejections':[{'reason':'next_session_not_observed','date':day}]}
        forced_value=complete_proceeds(forced['ledger'],snapshot['shares']);forced_value['rejections']=forced['rejections']
        outcome=exit_outcome(forced=forced_value,continued_proceeds=continued['proceeds'],continued_end=continued['date'],
            shares=snapshot['shares'],close=snapshot['close'],config=config)
        outcome=_reject_ambiguous_price_basis(outcome,frame,day)
        group=stable_hash({'symbol':snapshot['symbol'],'plan':snapshot['plan_id'],'entry':snapshot['entry_date'],'stage':stage})
        pair={'symbol':snapshot['symbol'],'date':day,'economic_group_id':group,'episode_id':group,'stage':stage,
            'plan_id':snapshot['plan_id'],'original_signal_date':snapshot['entry_signal_date'],
            'holding_snapshot_hash':stable_hash(snapshot),'holding_reference_mode':reference_mode,
            'forced_ledger_hash':stable_hash(forced['ledger']),
            'continued_ledger_hash':stable_hash(following),'continuation_policy_hash':config[stage+'_policy_hash'],
            'shares':snapshot['shares'],'forced':forced_value,'continued':continued,'outcome':outcome}
        pairs.append({**pair,'forced_ledger':forced['ledger'],'forced_rejections':forced['rejections']})
        exits.append({**values,**{k:v for k,v in pair.items() if k not in {'forced','continued','outcome'}},**outcome})
        if progress and (i+1)%100==0:progress({'holding_pairs':i+1,'symbol':snapshot['symbol'],'stage':stage})
    # Both paired targets must mature. The remainder may be unclosed; it is not
    # a loss and is never used to fit or calibrate a head.
    result={'entries':pd.DataFrame(entries),'exits':pd.DataFrame(exits),'episodes':pd.DataFrame(episodes),
        'reference_population':copy.deepcopy(reference_population or {}),'reference_mode':reference_mode}
    for key,columns in (('entries',ENTRY_FEATURE_COLUMNS),('exits',EXIT_FEATURE_COLUMNS)):
        required={column:'float64' for column in (*columns,'label','exit_advantage')}
        required.update(symbol='string',date='string',label_end='string',label_available='bool',input_eligible='bool',
            economic_group_id='string')
        if key=='entries':
            required.update(attempt_id='string',original_opportunity_ids='object',original_event_ids='object',
                original_event_first_pre_gate_dates='string',flat_state_hash='string',reference_state_hash='string',
                entry_plan_hash='string',net_return='float64')
        else:required.update(episode_id='string',holding_reference_mode='string',stage='string')
        missing={column:pd.Series(False,index=result[key].index,dtype=dtype) if dtype=='bool'
            else pd.Series(index=result[key].index,dtype=dtype)
            for column,dtype in required.items() if column not in result[key]}
        if missing:result[key]=pd.concat([result[key],pd.DataFrame(missing)],axis=1)
    summary={'symbol':frame.symbol.iloc[0],'stage':stage,'counts':counts,'entry_rows':len(entries),
        'exit_rows':len(exits),**label_maturity_counts(result,source_end=frame.date.iloc[-1],population_end=population_end),
        'frozen_source_end':frame.date.iloc[-1],'effective_population_end':population_end or frame.date.iloc[-1],
        'label_counts_scope':'actual_liquidation_cutoff_no_input_or_split_eligibility_inference',
        'reference_ledger_hash':stable_hash(reference['ledger']),'reference_metrics':reference['metrics'],
        'reference_mode':reference_mode,'reference_population':copy.deepcopy(reference_population or {}),
        'reference_population_hash':stable_hash(reference_population or {}),'population_end':population_end,
        'economic_entry_identity_groups':int(result['entries'].economic_group_id.nunique()) if 'economic_group_id' in result['entries'] else 0,
        'atomic_split_policy':config['split_policy'],
        'atomic_economic_weighting':config['economic_weighting'],
        'ordinary_sell_policy':'continuous_sell_pass_for_actual_holding','holding_checkpoint_upgrade':False}
    if directory:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for key in ('entries','exits','episodes'):result[key].to_parquet(directory/(key+'.parquet'),index=False)
        account_artifacts=write_account_artifacts(directory,{
            'reference_account.json':reference,'candidate_clones.json':clone_audits,'holding_pairs.json':pairs})
        summary['account_storage']=account_artifacts['storage_stats']
        summary['files']={p.name:{'path':str(p),'sha256':_file_hash(p)} for p in directory.iterdir() if p.is_file()}
        summary['files'].update(account_artifacts['files'])
        _save(directory/'manifest.json',summary)
    return result,summary
