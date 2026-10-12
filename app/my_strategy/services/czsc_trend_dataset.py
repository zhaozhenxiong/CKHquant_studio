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
from my_strategy.services.czsc_research import _save, _file_hash
from my_strategy.services.czsc_trend_account import (candidate_from_row, replay_trend_account,
    clone_candidate_account, clone_holding_branches, prepare_account_context)
from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
from my_strategy.services.czsc_trend_features import (ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS,
    L1_MA_EXIT_COLUMNS, entry_feature_row, holding_feature_row, event_prefix_rows,original_evidence_events)
from my_strategy.services.czsc_trend_labels import episode_outcome, exit_outcome, target_contract
from my_strategy.services.czsc_trend_interval_model import schema_hash
from my_strategy.services.czsc_trend_model import bind_trend_inputs


def make_contract(head, config, calendar, stage):
    columns = L1_MA_EXIT_COLUMNS if head == 'ma_exit' else ENTRY_FEATURE_COLUMNS if head == 'entry' else EXIT_FEATURE_COLUMNS
    value = target_contract(head=head, config=config, market_dates=calendar,
        source_snapshot_hash=config['source_snapshot_hash'], feature_schema_hash=schema_hash(columns,head),
        continuation_policy_hash=config[stage+'_policy_hash'], initial_cash=config['initial_cash'], stage=stage,
        candidate_policy_hash=config['candidate_policy_hash'])
    value['model_input_schema_hash']=schema_hash(columns,head)
    value['output_type']='continuous' if head=='exit_advantage' else 'classification'
    value['contract_hash']=stable_hash({k:v for k,v in value.items() if k!='contract_hash'})
    return value


def training_frame(dataset, head, config, calendar, stage):
    key='entries' if head=='entry' else 'exits'; frame=dataset[key].copy()
    if head=='exit_advantage':frame['label']=frame['exit_advantage']
    frame.attrs['label_contract']=make_contract(head,config,calendar,stage)
    return frame


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
    values['input_eligible']=bool(any(values.get(source+'_input_available',0) for source in candidate['sources']))
    return values


def build_account_dataset(frame, calendar, native_pack, trend_probabilities, config, *, stage='E0',
                          exit_policy=None, directory=None, run_id='', progress=None):
    if stage not in {'E0','E1'} or (stage=='E0' and exit_policy is not None):
        raise ValueError('E0 cannot depend on a learned exit or high-return entry head')
    frame=bind_trend_inputs(frame,calendar)
    account_context=prepare_account_context(frame,calendar)
    prepared=prepare_trend_decisions(frame,calendar,evidence_pack=native_pack,
        trend_probabilities=trend_probabilities,config=config,mode='E0')
    reference=replay_trend_account(frame,calendar,prepared,config=config,exit_policy=exit_policy,
        start=config['feature_start'],run_id=run_id+':reference',mode=stage,account_context=account_context)
    byday={r['date']:r for r in frame.to_dict('records')};opinions={r['date']:r for r in prepared['rows']}
    states={r['last_close_date']:r for r in reference['account_snapshots']}
    admissions={r['date']:r for r in reference['entry_diagnostics']}
    eventrows=event_prefix_rows(frame,original_evidence_events(prepared),calendar)
    eventmap=dict(zip(frame.date.astype(str),eventrows));midx={d:i for i,d in enumerate(calendar)}
    costs=reference['cost_model']
    cost=CostModel(commission=costs['commission'],stamp_tax=costs['stamp_tax'],slippage=costs['slippage'],
        min_commission=costs['min_commission'],version='flat_v1',impact_tiers=())
    entries=[];episodes=[];entrymap={};clone_audits=[];seen=set();counts={'raw_risk_opportunities':0,'nonadmitted_opportunities':0,'flat_candidates':0,'held_candidates':0,'candidate_rejections':0}
    for day,opinion in opinions.items():
        if day<config['feature_start'] or day not in states:continue
        candidate=candidate_from_row(opinion);plan=candidate['entry_plan']
        if not plan or not opinion.get('fresh_candidate_ids'):continue
        counts['raw_risk_opportunities']+=1
        if opinion.get('action')!='BUY' or day not in admissions:counts['nonadmitted_opportunities']+=1;continue
        candidate=admissions[day]['candidate'];plan=candidate['entry_plan']
        state=states[day];observed=state['current_snapshot'];values=_entry_values(byday[day],observed,candidate,plan,eventmap[day],config)
        identifier=stable_hash({'symbol':opinion['symbol'],'original_event_ids':sorted(candidate['candidate_event_ids']),
            'candidate_policy':config['candidate_policy_hash']})
        if identifier in seen:continue
        seen.add(identifier)
        meta={'opportunity_id':identifier,'economic_group_id':identifier,'candidate_id':identifier,
            'candidate_signal_date':day,'plan_id':plan['plan_id'],'entry_plan_hash':stable_hash(plan),
            'source':'union','stage':stage,'sources':','.join(candidate['sources']),
            'candidate_policy_hash':config['candidate_policy_hash'],'continuation_policy_hash':config[stage+'_policy_hash'],
            'reference_state_hash':stable_hash(state),'candidate_reference_account':True,'actual_flat_reference':state['shares']==0}
        if state['shares']!=0 or (state.get('pending_intent') or {}).get('side')=='SELL':
            counts['held_candidates']+=1
            outcome={'label':np.nan,'label_available':False,'label_end':None,'label_reason':'actual_reference_not_flat'}
            episodes.append({**meta,**outcome});continue
        counts['flat_candidates']+=1
        if not plan.get('entry_allowed'):
            outcome={'label':np.nan,'label_available':False,'label_end':None,'label_reason':'candidate_risk_ineligible'}
            entries.append({**values,**meta,**outcome});episodes.append({**meta,**outcome});continue
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
            'holding_snapshot_hash':stable_hash(snapshot),'forced_ledger_hash':stable_hash(forced['ledger']),
            'continued_ledger_hash':stable_hash(following),'continuation_policy_hash':config[stage+'_policy_hash'],
            'shares':snapshot['shares'],'forced':forced_value,'continued':continued,'outcome':outcome}
        pairs.append({**pair,'forced_ledger':forced['ledger'],'forced_rejections':forced['rejections']})
        exits.append({**values,**{k:v for k,v in pair.items() if k not in {'forced','continued','outcome'}},**outcome})
        if progress and (i+1)%100==0:progress({'holding_pairs':i+1,'symbol':snapshot['symbol'],'stage':stage})
    # Both paired targets must mature. The remainder may be unclosed; it is not
    # a loss and is never used to fit or calibrate a head.
    result={'entries':pd.DataFrame(entries),'exits':pd.DataFrame(exits),'episodes':pd.DataFrame(episodes)}
    for key,columns in (('entries',ENTRY_FEATURE_COLUMNS),('exits',EXIT_FEATURE_COLUMNS)):
        for column in (*columns,'symbol','date','label','label_available','label_end','input_eligible','exit_advantage'):
            if column not in result[key]:result[key][column]=pd.Series(dtype=float if column in columns else object)
    summary={'symbol':frame.symbol.iloc[0],'stage':stage,'counts':counts,'entry_rows':len(entries),
        'mature_entries':int(result['entries'].label_available.fillna(False).sum()),
        'positive_entries':int(result['entries'].label.eq(1).sum()),'exit_rows':len(exits),
        'mature_exits':int(result['exits'].label_available.fillna(False).sum()),
        'reference_ledger_hash':stable_hash(reference['ledger']),'reference_metrics':reference['metrics'],
        'ordinary_sell_policy':'new_final_direction_not_raw_union','holding_checkpoint_upgrade':False}
    if directory:
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for key in result:result[key].to_parquet(directory/(key+'.parquet'),index=False)
        _save(directory/'reference_account.json',reference)
        _save(directory/'candidate_clones.json',clone_audits);_save(directory/'holding_pairs.json',pairs)
        summary['files']={p.name:{'path':str(p),'sha256':_file_hash(p)} for p in directory.iterdir() if p.is_file()}
        _save(directory/'manifest.json',summary)
    return result,summary
