"""Independent account-free quality population and real Broker probe evidence.

This head never consumes actual-flat account features or user inventory. Its
100k accounts are separately identified training probes, with real next-open
fills and signal-pinned E1 continuation, not a shared trading portfolio.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import hashlib
import hmac
import os
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, SPLIT_POLICY, ECONOMIC_WEIGHTING
from my_strategy.services.czsc_trend_features import (causal_market_row, event_prefix_rows,
    original_evidence_events, NATIVE_COLUMNS, SOURCE_COLUMNS, SOURCE_AVAILABILITY_COLUMNS,
    EVENT_COLUMNS, L1_OUTPUT_COLUMNS)
from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS, STRUCTURE_COLUMNS
from my_strategy.services.czsc_price_volume_features import W_INPUT_COLUMNS
from my_strategy.services.czsc_trend_math import _fit_preprocess, _transform, _metrics
from my_strategy.services.czsc_trend_interval_model import time_masks, _weights, _calibrate, _sigmoid, split_validation
from my_strategy.services.czsc_trend_support import _save, _file_hash, _json, resolve_frozen_asset

HEAD = 'reference_quality'
POPULATION = 'account_free_daily_original_BUY_risk_probes'
TARGET = 'independent_reference_probe_complete_fee_net_8pct_and_2R'
PROBE_CASH = 100000.
PREREGISTRATION = '3fcd718962eb2ca783b4faf2f00bd1ebe86c8f7f81f86dabfed3c116cd941208'
RISK_COLUMNS = ('reference_stop_distance_fraction','reference_risk_budget_fraction',
    'reference_risk_position_cap_fraction','reference_source_count','reference_buy_base_support','reference_sell_base_support')
COLUMNS = tuple((*MA_TREND_COLUMNS,*STRUCTURE_COLUMNS,*W_INPUT_COLUMNS,*NATIVE_COLUMNS,
    *SOURCE_COLUMNS,*SOURCE_AVAILABILITY_COLUMNS,*EVENT_COLUMNS,*L1_OUTPUT_COLUMNS,*RISK_COLUMNS))
SCHEMA_HASH = stable_hash({'head':HEAD,'columns':COLUMNS,'population':POPULATION,
    'scope':'causal_market_real_event_prefix_original_risk_only_no_account_or_manual_inventory',
    'split_policy':SPLIT_POLICY,'economic_weighting':ECONOMIC_WEIGHTING})
_ISSUER = object()
_SEAL_KEY = os.urandom(32)
_AUDIT_CACHE = {}
_TEXT = ('symbol','date','label_end','attempt_id','economic_group_id','opportunity_id','original_event_ids',
    'original_event_first_pre_gate_dates','risk_identity_hash','opportunity_hash','label_reason',
    'label_available_at','entry_date','exit_date','outcome_class')


def reference_protocol(config):
    body={'head':HEAD,'population':POPULATION,'target':TARGET,'probe_initial_cash':PROBE_CASH,
        'preregistration_sha256':PREREGISTRATION,'columns':list(COLUMNS),'feature_schema_hash':SCHEMA_HASH,
        'admission':'qualified_prior_trend_final_BUY_live_original_opportunity_original_signal_risk_plan',
        'upstream_trend_route':'canonical_latest_strict_prior_auto;other_pinned_trend_is_out_of_population',
        'seed':'actual_Broker_all_WAIT_prefix_not_user_or_actual_flat_population',
        'trial':'daily_independent_one_forced_next_verified_open_then_signal_pinned_E1_no_upgrade',
        'split_policy':SPLIT_POLICY,'economic_weighting':ECONOMIC_WEIGHTING,
        'no_account_inputs':True,'no_manual_position_inputs':True,'cost_multipliers':[1.,2.],
        'high_return':copy.deepcopy(config['high_return']),'risk':copy.deepcopy(config['signal_points']),
        'costs':{k:config['account'][k] for k in ('commission','min_fee','stamp','slippage')},
        'account_guards':{k:config['account'][k] for k in ('max_weight','lot_size','max_holding_sessions',
            'stop_loss','max_volume_participation','seasoned_bars','mainboard_conservative_limit')},
        'consensus':copy.deepcopy(config['consensus']),'liquidity':copy.deepcopy(config['liquidity_universe']),
        'fit':{'architecture':'binary_logistic_C1_lbfgs','max_iter':500,'seed':config.get('seed',42),
            'min_train_rows':config['min_train_rows'],'min_calibration_rows':config['min_calibration_rows'],
            'both_classes_train_calibration':True,'calibration':'chronological_positive_slope_Platt'},
        'qualification':{'windows':copy.deepcopy(config['development_windows']),
            'min_stocks':60,'min_round_trips_per_window':50,'min_round_trips_total':200,'drawdown_tolerance':.02,
            'same_all_probe_denominator':True,'unadmitted_probes_cash':True,'all_frozen_stocks_cash_coverage':True,
            'aggregation':'equal_scaled_independent_probe_accounts_per_stock_not_shared_portfolio',
            'core_selection_gate_changed':False,'formal_certification':'pending_development_only'}}
    return {**body,'protocol_hash':stable_hash(body)}


def reference_probe_plan(row,risk,config):
    from my_strategy.services.czsc_trend_runtime import _reference_buy_plan
    plan,reasons=_reference_buy_plan(row,config)
    if reasons or not plan or row['agents']['ma'].get('status')!='scored':return None
    if (risk.get('status')!='available' or risk.get('original_buy_plan_hash')!=stable_hash(plan)
            or risk.get('original_buy_plan_id')!=plan['plan_id']
            or risk.get('stop_price')!=plan['stop_price'] or risk.get('risk_position_cap_fraction')!=plan['target_weight']
            or risk.get('reference_price')!=plan['reference_price'] or risk.get('date')!=row['date']):
        return None
    return copy.deepcopy(plan)


def reference_feature_row(feature,row,event_prefix,risk):
    values=causal_market_row(feature,row['date']);plan=row.get('entry_plan') or {}
    sources=plan.get('sources') or []
    values.update({f'source_{source}':float(source in sources) for source in ('ma','structure','wyckoff')})
    values.update({k:(event_prefix or {}).get(k,np.nan) for k in EVENT_COLUMNS})
    ma=row['agents']['ma'];available=ma.get('status')=='scored'
    values.update({f'trend_p_{name}':ma.get(f'p_{name}',np.nan) if available else np.nan for name in ('up','down','neutral')})
    values.update(trend_probability_available=float(available),
        reference_stop_distance_fraction=risk.get('stop_distance_fraction',np.nan),
        reference_risk_budget_fraction=risk.get('risk_budget_fraction',np.nan),
        reference_risk_position_cap_fraction=risk.get('risk_position_cap_fraction',np.nan),
        reference_source_count=float(len(sources)),reference_buy_base_support=row['buy_base_score'],
        reference_sell_base_support=row['sell_base_score'])
    return {k:float(values[k]) if values.get(k) is not None and math.isfinite(float(values[k])) else np.nan for k in COLUMNS}


def _frame(rows):
    raw=pd.DataFrame(rows);columns={}
    for key in COLUMNS+('label','net_return','r_multiple'):
        columns[key]=pd.to_numeric(raw.get(key,pd.Series(np.nan,index=raw.index)),errors='coerce').astype(float)
    for key in _TEXT:
        columns[key]=raw.get(key,pd.Series(None,index=raw.index,dtype=object)).astype('string')
    for key in ('label_available','input_eligible','risk_available'):
        columns[key]=raw.get(key,pd.Series(False,index=raw.index)).fillna(False).astype(bool)
    return pd.DataFrame(columns,index=raw.index)


def _rows_digest(data):
    """Exact canonical rows, including unknown values and original atoms."""
    columns=[*COLUMNS,'label','net_return','r_multiple',*_TEXT,'label_available','input_eligible','risk_available']
    data=data.loc[:,columns]
    h=hashlib.sha256(json.dumps(columns).encode())
    h.update(pd.util.hash_pandas_object(data,index=False).to_numpy(np.uint64).tobytes())
    return h.hexdigest()


def _label_counts(data):
    available=data.label_available & data.label.notna()
    return {'total':len(data),'mature':int(available.sum()),'positive':int((available & data.label.eq(1.)).sum()),
        'negative':int((available & data.label.eq(0.)).sum()),'unknown':int((~available).sum()),
        'unknown_reasons':{str(k):int(v) for k,v in data.loc[~available,'label_reason'].fillna('unknown').value_counts().items()}}


def _validate_probe_rows(root,record,artifact,frame,config,calendar):
    """Rebuild targets from real fee-bearing fills, never from saved labels."""
    from my_strategy.services.czsc_trend_labels import episode_outcome
    from my_strategy.services.czsc_trend_dataset import _reject_ambiguous_price_basis
    from my_strategy.execution.cost_model import CostModel
    if artifact.get('population')!=POPULATION or artifact.get('symbol')!=record['symbol']:
        raise ValueError('reference probe population identity differs')
    rows=[];seen=set();atom_dates={}
    for probe in artifact['probes']:
        meta=probe['metadata'];day=probe['signal_date'];plan=probe['original_plan'];atoms=sorted(set(plan['original_opportunity_ids']))
        if (probe.get('scope')!=POPULATION or probe.get('probe_cash')!=PROBE_CASH or probe['probe_id'] in seen
                or meta['attempt_id']!=probe['probe_id'] or meta['symbol']!=record['symbol'] or meta['date']!=day
                or day not in calendar or plan['signal_date']!=day or plan['symbol']!=record['symbol']
                or meta['risk_identity_hash']!=stable_hash(plan) or meta['opportunity_hash']!=stable_hash(atoms)
                or json.loads(meta['original_event_ids'])!=atoms):
            raise ValueError('reference probe original plan/atom identity differs')
        seen.add(probe['probe_id']);first=json.loads(meta['original_event_first_pre_gate_dates'])
        if set(first)!=set(atoms) or any(first[a]>day or a in atom_dates and atom_dates[a]!=first[a] for a in atoms):
            raise ValueError('reference probe atomic first-observation clock changed')
        atom_dates.update(first)
        actual=None
        for multiplier in (1.,2.):
            account=probe['accounts'][str(multiplier)];costs=account['cost_model']
            for field,key in (('commission','commission'),('min_commission','min_fee'),('stamp_tax','stamp'),('slippage','slippage')):
                if costs[field]!=config['account'][key]*multiplier:raise ValueError('reference probe fees differ from frozen execution contract')
            cost=CostModel(commission=costs['commission'],min_commission=costs['min_commission'],stamp_tax=costs['stamp_tax'],
                slippage=costs['slippage'],version='flat_v1',impact_tiers=())
            outcome=episode_outcome(account['ledger'],stop_price=account['actual_initial_stop'],cost_model=cost,config=config)
            outcome=_reject_ambiguous_price_basis(outcome,frame,outcome.get('entry_date') or day)
            if _json(outcome)!=account['outcome']:raise ValueError('reference saved outcome differs from actual fee-net Broker episode')
            if outcome['label_available'] and (not outcome['label_end'] or outcome['label_end']<=day):
                raise ValueError('reference probe mature label is not after its signal')
            if multiplier==1.:actual=outcome
        rows.append({**probe['feature_row'],**meta,**actual})
    expected=_frame(rows);observed=pd.read_parquet(_asset(root,record['files']['observations.parquet']['path']))
    try:pd.testing.assert_frame_equal(observed,expected,check_exact=True)
    except AssertionError as exc:raise ValueError('reference observation rows differ from actual probe outcomes and original metadata') from exc
    if record['rows']!=len(expected):raise ValueError('reference observation count differs from frozen population')
    return expected


def _probe_worker(payload):
    from my_strategy.services.czsc_trend_pipeline import read_snapshot,forward_trend,pinned_exit_policy
    from my_strategy.services.czsc_trend_model import bind_trend_inputs
    from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
    from my_strategy.services.czsc_trend_runtime import _reference_risk_plan
    from my_strategy.services.czsc_trend_account import replay_trend_account,prepare_account_context,clone_candidate_account
    from my_strategy.services.czsc_trend_labels import episode_outcome
    from my_strategy.services.czsc_trend_dataset import _reject_ambiguous_price_basis
    from my_strategy.execution.cost_model import CostModel
    from my_strategy.services.czsc_account_artifacts import write_account_artifact
    record,calendar,cfg,root,trends,ma_bundles=payload
    root=Path(root);frame,pack=read_snapshot(record);frame=frame.loc[frame.date<=cfg['training_end']].copy()
    frame=bind_trend_inputs(frame,calendar,cfg)
    routes=forward_trend(frame,root,trends,cfg,market_dates=calendar)
    prepared=prepare_trend_decisions(frame,calendar,evidence_pack=pack,trend_probabilities=routes,config=cfg,mode='E0')
    exit_policy=pinned_exit_policy(frame,calendar,prepared,root,ma_bundles,cfg)
    seed_cfg=copy.deepcopy(cfg);seed_cfg['initial_cash']=PROBE_CASH;seed_cfg['account']['initial_cash']=PROBE_CASH
    wait={'rows':[{'date':row['date'],'symbol':row['symbol'],'action':'WAIT','direction':'WAIT',
        'entry_eligible':False,'sell_pass':False} for row in prepared['rows']]}
    contexts={};seed_accounts={};states={};probe_configs={}
    for multiplier in (1.,2.):
        current_cfg=copy.deepcopy(seed_cfg)
        for key in ('commission','min_fee','stamp','slippage'):current_cfg['account'][key]*=multiplier
        contexts[multiplier]=prepare_account_context(frame,calendar,current_cfg);probe_configs[multiplier]=current_cfg
        seed=replay_trend_account(frame,calendar,wait,config=current_cfg,start=cfg['feature_start'],
            account_context=contexts[multiplier],initial_cash=PROBE_CASH,
            run_id=root.name+':reference-probe-seed:'+record['symbol']+':'+str(multiplier),
            account_id='reference_probe_seed:'+record['symbol']+':'+str(multiplier),mode='E0')
        if seed['ledger'] or any(s['shares']!=0 or s['cash']!=PROBE_CASH for s in seed['account_snapshots']):
            raise ValueError('reference probe seed is not the actual no-entry Broker prefix')
        seed_accounts[multiplier]=seed;states[multiplier]={s['last_close_date']:s for s in seed['account_snapshots']}
    prefix=dict(zip(frame.date.astype(str),event_prefix_rows(frame,original_evidence_events(prepared),calendar),strict=True))
    bars=frame.to_dict('records');native={r['date']:r for r in pack['evaluations']}
    rows=[];probes=[];first_atoms={}
    for index,row in enumerate(prepared['rows']):
        day=row['date']
        # Atomic opportunities retain their first observable pre-admission
        # date, including days before consensus/risk/liquidity admission.
        for event in (row.get('candidate_evidence') or [])+(row.get('active_candidates') or []):
            atom=event.get('opportunity_id')
            if atom and event.get('side')=='BUY':first_atoms.setdefault(atom,day)
        if day<cfg['feature_start']:continue
        risk=_reference_risk_plan(row,bars,index,native.get(day,{}),cfg)
        plan=reference_probe_plan(row,risk,cfg)
        if not plan:continue
        if not bars[index].get('liquidity_pool_eligible'):continue
        atoms=sorted(set(plan['original_opportunity_ids']));sources=sorted(plan['sources'])
        for atom in atoms:first_atoms.setdefault(atom,day)
        identifier=stable_hash({'population':POPULATION,'symbol':row['symbol'],'day':day,
            'originals':atoms,'plan':stable_hash(plan),'source_snapshot_hash':cfg['source_snapshot_hash']})
        values=reference_feature_row(bars[index],row,prefix[day],risk)
        meta={'symbol':row['symbol'],'date':day,'attempt_id':identifier,
            'economic_group_id':stable_hash({'symbol':row['symbol'],'atoms':atoms}),
            'opportunity_id':stable_hash({'symbol':row['symbol'],'atoms':atoms}),
            'original_event_ids':json.dumps(atoms),'original_event_first_pre_gate_dates':json.dumps({a:first_atoms[a] for a in atoms}),
            'risk_identity_hash':stable_hash(plan),'opportunity_hash':stable_hash(atoms),'input_eligible':True}
        accounts={};outcome={'label':np.nan,'label_available':False,'label_end':None,'label_reason':'next_session_not_observed'}
        for multiplier in (1.,2.):
            probe_cfg=probe_configs[multiplier];seed=seed_accounts[multiplier]
            if day==frame.date.iloc[-1]:
                account={'ledger':[],'rejections':[{'date':day,'reason':'next_session_not_observed'}],
                    'holding_snapshots':[],'cost_model':seed['cost_model'],'completed':False,'actual_position':None}
            else:
                account=clone_candidate_account(frame,calendar,prepared,states[multiplier][day],{'entry_plan':plan},probe_cfg,
                    exit_policy,root.name+':reference-probe:'+identifier[:20]+':'+str(multiplier),account_context=contexts[multiplier])
            costs=account['cost_model'];cost=CostModel(commission=costs['commission'],stamp_tax=costs['stamp_tax'],
                slippage=costs['slippage'],min_commission=costs['min_commission'],version='flat_v1',impact_tiers=())
            stop=account['holding_snapshots'][0]['stop_price'] if account['holding_snapshots'] else plan['stop_price']
            result=episode_outcome(account['ledger'],stop_price=stop,cost_model=cost,config=cfg)
            result=_reject_ambiguous_price_basis(result,frame,result.get('entry_date') or day)
            accounts[str(multiplier)]={'ledger':account['ledger'],'rejections':account['rejections'],'outcome':result,
                'cost_model':costs,'exit_model_bindings':[{k:s.get(k) for k in ('date','entry_signal_date','exit_model_binding')}
                    for s in account['holding_snapshots']],'actual_initial_stop':stop}
            if multiplier==1.:outcome=result
        rows.append({**values,**meta,**outcome})
        probes.append({'probe_id':identifier,'signal_date':day,'scope':POPULATION,'probe_cash':PROBE_CASH,
            'seed_state_hashes':{str(k):stable_hash(states[k][day]) for k in states},
            'seed_account_ids':{str(k):states[k][day]['account_id'] for k in states},
            'original_plan':plan,'reference_risk':risk,'feature_row':values,'metadata':meta,'accounts':accounts})
    directory=root/'dataset/reference_quality/stocks'/record['symbol'].replace('.','_');directory.mkdir(parents=True)
    path=directory/'observations.parquet';_frame(rows).to_parquet(path,index=False)
    write_account_artifact(directory/'probes.json',{'population':POPULATION,'symbol':record['symbol'],
        'seed_scope':'actual_reference_probe_Broker_no_entry_prefix',
        'seed_ledgers':{str(k):s['ledger'] for k,s in seed_accounts.items()},
        'seed_contexts':{str(k):s['series_context'] for k,s in seed_accounts.items()},'probes':probes})
    files={p.name:{'path':str(p.relative_to(root)),'sha256':_file_hash(p)} for p in directory.iterdir() if p.is_file()}
    print('DATASET reference_quality '+record['symbol']+' entries='+str(len(rows))+' exits=0',flush=True)
    return {'symbol':record['symbol'],'rows':len(rows),'files':files,'source_record_sha256':record['sha256']}


def _target_contract(config,population):
    body={'family':STRATEGY_ID,'head':HEAD,'population':POPULATION,'target':TARGET,
        'feature_schema_hash':SCHEMA_HASH,'columns':list(COLUMNS),'reference_population_hash':population['population_hash'],
        'source_snapshot_hash':config['source_snapshot_hash'],'pipeline_contract_hash':config['pipeline_contract_hash'],
        'continuation_policy_hash':config['E1_policy_hash'],'protocol_hash':population['protocol_hash'],
        'probe_initial_cash':PROBE_CASH,'no_account_inputs':True,'no_actual_flat_entry_head':True,
        'upstream_trend_catalog_hash':population['trend_catalog_hash'],
        'signal_pinned_ma_exit_catalog_hash':population['ma_exit_catalog_hash'],
        'split_policy':SPLIT_POLICY,'economic_weighting':ECONOMIC_WEIGHTING,
        'high_return':copy.deepcopy(config['high_return'])}
    return {**body,'contract_hash':stable_hash(body)}


def _validate_contract(contract):
    if (contract.get('contract_hash')!=stable_hash({k:v for k,v in contract.items() if k!='contract_hash'})
            or contract.get('family')!=STRATEGY_ID or contract.get('head')!=HEAD or contract.get('population')!=POPULATION
            or contract.get('target')!=TARGET or contract.get('feature_schema_hash')!=SCHEMA_HASH
            or contract.get('columns')!=list(COLUMNS) or contract.get('probe_initial_cash')!=PROBE_CASH
            or contract.get('no_account_inputs') is not True or contract.get('no_actual_flat_entry_head') is not True
            or contract.get('split_policy')!=SPLIT_POLICY or contract.get('economic_weighting')!=ECONOMIC_WEIGHTING):
        raise ValueError('reference-quality target/input/population contract mismatch')
    for key in ('reference_population_hash','source_snapshot_hash','pipeline_contract_hash','continuation_policy_hash','protocol_hash',
                'upstream_trend_catalog_hash','signal_pinned_ma_exit_catalog_hash'):
        if not contract.get(key):raise ValueError('reference quality missing immutable source identity')


def train_reference_head(data,path,fold,config,contract):
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    _validate_contract(contract);train,calibration,policy=time_masks(data,fold,head='entry')
    y=data.label.to_numpy(float)
    if train.sum()<config['min_train_rows'] or calibration.sum()<config['min_calibration_rows']:
        raise ValueError('insufficient mature reference_quality rows')
    if len(np.unique(y[train]))!=2 or len(np.unique(y[calibration]))!=2:
        raise ValueError('reference_quality train/calibration require both classes')
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise FileExistsError('reference quality requires a fresh head directory')
    start=local_now().isoformat();x=data[list(COLUMNS)].to_numpy(float);pre=_fit_preprocess(x[train]);xx=_transform(x,pre)
    model=LogisticRegression(C=1.,max_iter=500,solver='lbfgs',random_state=int(config.get('seed',42)))
    with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=2):
        warnings.simplefilter('always',ConvergenceWarning);model.fit(xx[train],y[train].astype(int),sample_weight=_weights(data,train,head='entry'))
    z=model.decision_function(xx[calibration]);cal=_calibrate(z,y[calibration],_weights(data,calibration,head='entry'))
    if not math.isfinite(cal['slope']) or cal['slope']<=0:raise ValueError('reference quality calibration requires positive slope')
    path.mkdir(parents=True);joblib.dump(model,path/'model.joblib')
    manifest={'family':STRATEGY_ID,'head':HEAD,'population':POPULATION,'target':TARGET,'columns':list(COLUMNS),
        'feature_schema_hash':SCHEMA_HASH,'target_contract':contract,'target_contract_hash':stable_hash(contract),
        'fold':fold,'preprocess':pre,'calibration':cal,'checkpoint':'model.joblib','checkpoint_sha256':_file_hash(path/'model.joblib'),
        'training_started_at':start,'actual_training_completed_at':local_now().isoformat(),'available_at':fold['policy_end'],
        'train_label_end':str(data.loc[train,'label_end'].max()),'calibration_label_end':str(data.loc[calibration,'label_end'].max()),
        'policy_label_end':str(data.loc[policy,'label_end'].max()) if policy.any() else fold['policy_end'],
        'sample_counts':{name:{'rows':int(mask.sum()),'positive':int((y[mask]==1).sum()),'negative':int((y[mask]==0).sum())}
            for name,mask in (('train',train),('calibration',calibration),('policy',policy))},
        'segment_data_digests':{name:_rows_digest(data.loc[mask]) for name,mask in
            (('train',train),('calibration',calibration),('policy',policy))},
        'segment_weight_digests':{name:hashlib.sha256(_weights(data,mask,head='entry').tobytes()).hexdigest()
            for name,mask in (('train',train),('calibration',calibration))},
        'calibration_metrics':_metrics(y[calibration],_sigmoid(z*cal['slope']+cal['intercept'])),
        'convergence':{'iterations':model.n_iter_.tolist(),'max_iter':500,'converged':not any(issubclass(w.category,ConvergenceWarning) for w in caught)},
        'publication_allowed':False,'actual_device':'cpu','account_input_columns':[]}
    manifest['manifest_sha256']=stable_hash(manifest);_save(path/'manifest.json',manifest);return manifest


def load_reference_quality_head(path,*,expected_contract=None):
    path=Path(path).resolve();m=json.loads((path/'manifest.json').read_text());body={k:v for k,v in m.items() if k!='manifest_sha256'}
    if (m.get('manifest_sha256')!=stable_hash(body) or m.get('head')!=HEAD or m.get('population')!=POPULATION
            or m.get('family')!=STRATEGY_ID or m.get('columns')!=list(COLUMNS) or m.get('feature_schema_hash')!=SCHEMA_HASH
            or m.get('target')!=TARGET or m.get('account_input_columns')!=[]):
        raise ValueError('reference quality manifest identity mismatch')
    _validate_contract(m['target_contract'])
    fold=m['fold']
    if not (fold['train_end']<fold['calibration_start']<=fold['calibration_end']<fold['policy_start']<=fold['policy_end']<=fold['validation_end']):
        raise ValueError('reference quality head chronology overlaps')
    if (m['available_at']!=fold['policy_end'] or any(m[key]>fold[end] for key,end in
            (('train_label_end','train_end'),('calibration_label_end','calibration_end'),('policy_label_end','policy_end')))
            or any(pd.isna(pd.to_datetime(m[key],errors='coerce')) for key in ('train_label_end','calibration_label_end','policy_label_end'))):
        raise ValueError('reference quality head label cutoffs or availability changed')
    started=pd.to_datetime(m['training_started_at'],errors='coerce',utc=True)
    completed=pd.to_datetime(m['actual_training_completed_at'],errors='coerce',utc=True)
    if pd.isna(started) or pd.isna(completed) or completed<started:
        raise ValueError('reference quality actual training clock invalid')
    if m.get('target_contract_hash')!=stable_hash(m['target_contract']) or expected_contract is not None and m['target_contract']!=expected_contract:
        raise ValueError('reference quality target source binding changed')
    checkpoint=(path/m['checkpoint']).resolve()
    if not checkpoint.is_relative_to(path) or _file_hash(checkpoint)!=m['checkpoint_sha256']:
        raise ValueError('reference quality checkpoint changed')
    if not math.isfinite(m['calibration']['slope']) or m['calibration']['slope']<=0 or not math.isfinite(m['calibration']['intercept']):
        raise ValueError('reference quality calibration changed')
    return m


def _predict_prior(values,day,bundles,root,contract,cache,usage_mode='historical'):
    choices=[b for b in bundles if b['available_at']<day]
    if not choices:return None,None
    bundle=max(choices,key=lambda b:(b['available_at'],b['name']))
    asset=bundle['heads'][HEAD];path=_asset(root,asset['path'])
    if str(path) not in cache:
        m=load_reference_quality_head(path,expected_contract=contract)
        if m['manifest_sha256']!=asset['manifest_sha256']:raise ValueError('reference quality catalog head changed')
        model=joblib.load(path/m['checkpoint'])
        if list(model.classes_)!=[0,1] or model.n_features_in_!=len(COLUMNS):
            raise ValueError('reference quality checkpoint class/input identity changed')
        cache[str(path)]=(m,model)
    m,model=cache[str(path)]
    if any(m[key]>=day for key in ('train_label_end','calibration_label_end','policy_label_end')) or m['available_at']>=day:
        raise ValueError('reference quality not strictly prior to reference date')
    if usage_mode in ('forward','production') and pd.Timestamp(m['actual_training_completed_at'])>pd.Timestamp(day+'T15:00:00+08:00'):
        return None,None
    xx=_transform(np.array([[values[k] for k in COLUMNS]],float),m['preprocess'])
    z=float(model.decision_function(xx)[0]);p=float(_sigmoid(np.array([z*m['calibration']['slope']+m['calibration']['intercept']]))[0])
    return p,m


def _asset(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('reference quality asset outside run')
    return path


def _probe_window_account(probe,account,window,calendar,bars,source_hash,selected):
    days=[d for d in calendar if window['test_start']<=d<=window['test_end']]
    ledger=[r for r in account['ledger'] if r['date']<=window['test_end']] if selected else []
    cash=PROBE_CASH;shares=0;index=0;last=0.;daily=[]
    for day in days:
        while index<len(ledger) and ledger[index]['date']<=day:
            fill=ledger[index];cash=float(fill['cash_after']);shares=int(fill['position_after']);index+=1
        bar=bars.get(day)
        if bar and bar.get('has_trade_price')==1 and bar.get('source') in ('tushare','tushare:daily','baostock','eastmoney_unadjusted_v1','fuyao_unadjusted_v1','sina_unadjusted_v1','tencent_star_unadjusted_v1'):
            close=float(bar['close']);last=close if math.isfinite(close) and close>0 else last
        daily.append({'date':day,'cash':cash,'shares':shares,'close':last or None,'equity':cash+shares*last})
    return {'symbol':probe['metadata']['symbol'],'ledger':ledger,'daily':daily,'terminal_state':{'cash':cash,'shares':shares},
        'cost_model':account['cost_model'],'series_context':{'strategy':STRATEGY_ID,'source_hash':source_hash,
            'source_revision_verified':True,'initial_cash':PROBE_CASH,'account_start':window['test_start'],
            'scope':POPULATION,'probe_id':probe['probe_id'],'calendar_hash':stable_hash(calendar)}}


def _qualification_worker(payload,*,persist=True):
    from my_strategy.services.czsc_account_artifacts import read_account_artifact,write_account_artifact
    from my_strategy.services.model_training_center import audit_account
    from my_strategy.services.czsc_trend_pipeline import read_snapshot
    record,calendar,cfg,root,population,bundles,contract=payload;root=Path(root)
    artifact=read_account_artifact(_asset(root,record['files']['probes.json']['path']),expected_sha256=record['files']['probes.json']['sha256'])
    frame,_=read_snapshot(record['source_record']);bars={r['date']:r for r in frame.to_dict('records')}
    observations=_validate_probe_rows(root,record,artifact,frame,cfg,calendar)
    cache={};result=[]
    for window in cfg['development_windows']:
        probes=[p for p in artifact['probes'] if window['test_start']<=p['signal_date']<=window['test_end']]
        arms={name:[] for name in ('candidate','rules','candidate_stress','rules_stress')};scores=[]
        for probe in probes:
            p,m=_predict_prior(probe['feature_row'],probe['signal_date'],bundles,root,contract,cache)
            base=probe['feature_row']['reference_buy_base_support']
            selected=p is not None and p>=cfg['consensus']['high_return_threshold'] and base+cfg['consensus']['quality_weight']*p+1e-12>=cfg['consensus']['buy_total_threshold']
            scores.append({'probe_id':probe['probe_id'],'probability':p,'selected':selected,'manifest_sha256':m['manifest_sha256'] if m else None})
            for multiplier,suffix in (('1.0',''),('2.0','_stress')):
                for kind,admit in (('candidate',selected),('rules',True)):
                    account=_probe_window_account(probe,probe['accounts'][multiplier],window,calendar,bars,record['source_record_sha256'],admit)
                    arms[kind+suffix].append(audit_account(account,calendar))
        dates=[d for d in calendar if window['test_start']<=d<=window['test_end']]
        measured={}
        for name,accounts in arms.items():
            count=max(1,len(accounts));trips=[r for a in accounts for r in a['trip_returns']]
            daily=[{'date':day,'equity':sum(a['daily'][i]['equity'] for a in accounts)/count if accounts else PROBE_CASH} for i,day in enumerate(dates)]
            peak=PROBE_CASH;dd=0.
            for item in daily:peak=max(peak,item['equity']);dd=max(dd,1-item['equity']/peak)
            measured[name]={'symbol':record['symbol'],'initial_cash':PROBE_CASH,'total_return':daily[-1]['equity']/PROBE_CASH-1,
                'drawdown':dd,'completed_round_trips':len(trips),'trip_returns':trips,'daily':daily,
                'fees':sum(a['fees'] for a in accounts)/count,'independent_probe_count':len(probes)}
        result.append({'window':window,'arms':measured,'scores':scores})
    if not persist:return {'symbol':record['symbol'],'windows':result,'rows_digest':_rows_digest(observations),
        'label_counts':_label_counts(observations)}
    directory=root/'validation/reference_quality/stocks'/record['symbol'].replace('.','_');directory.mkdir(parents=True)
    write_account_artifact(directory/'qualification.json',{'population':POPULATION,'symbol':record['symbol'],'windows':result})
    files={p.name:{'path':str(p.relative_to(root)),'sha256':_file_hash(p)} for p in directory.iterdir() if p.is_file()}
    print('DATASET reference_quality_fee '+record['symbol']+' entries='+str(len(artifact['probes']))+' exits=0',flush=True)
    return {'symbol':record['symbol'],'windows':[{'window':r['window'],'arms':r['arms']} for r in result],'files':files}


def run_reference_quality(records,calendar,config,context,trend_bundles,ma_bundles,cpu_workers,folds):
    from my_strategy.services.czsc_trend_pipeline import _bounded_results,_pool_parquets
    from my_strategy.services.model_training_center import aggregate_account_metrics,selection_gate
    root=context.run_dir();protocol=reference_protocol(config)
    _save(context.subdir('config')/'reference_quality_protocol.json',protocol)
    print('STAGE reference_quality_population',flush=True)
    summaries=[]
    for item in _bounded_results(_probe_worker,((r,calendar,config,str(root),trend_bundles,ma_bundles) for r in records),cpu_workers,config['stock_batch_size']):
        summaries.append(item)
        print('REFERENCE_QUALITY_PROGRESS '+json.dumps({'stage':'reference_quality_population',
            'scope':'current_reference_population_stocks','completed':len(summaries),'total':len(records),'fold':None}),flush=True)
    for item,source in zip(summaries,records,strict=True):
        if item['symbol']!=source['symbol']:raise ValueError('reference quality population order differs')
        item['source_record']=copy.deepcopy(source)
    directory=context.subdir('dataset','reference_quality');pooled=directory/'observations.parquet'
    _pool_parquets((_asset(root,r['files']['observations.parquet']['path']) for r in summaries),pooled,load=False)
    manifest={'population':POPULATION,'protocol':protocol,'protocol_hash':protocol['protocol_hash'],
        'source_snapshot_hash':config['source_snapshot_hash'],'pipeline_contract_hash':config['pipeline_contract_hash'],
        'continuation_policy_hash':config['E1_policy_hash'],'calendar_hash':stable_hash(calendar),
        'probe_initial_cash':PROBE_CASH,'records':summaries,'failures':[],
        'trend_catalog_hash':stable_hash(trend_bundles),'ma_exit_catalog_hash':stable_hash(ma_bundles),
        'pooled_path':str(pooled.relative_to(root)),'pooled_sha256':_file_hash(pooled)}
    manifest['population_hash']=stable_hash(manifest);path=directory/'manifest.json';_save(path,manifest)
    population={'population':POPULATION,'population_hash':manifest['population_hash'],'protocol_hash':protocol['protocol_hash'],
        'manifest_path':str(path.relative_to(root)),'manifest_sha256':_file_hash(path),'pooled_path':str(pooled.relative_to(root)),
        'pooled_sha256':_file_hash(pooled),'source_snapshot_hash':config['source_snapshot_hash'],
        'pipeline_contract_hash':config['pipeline_contract_hash'],'continuation_policy_hash':config['E1_policy_hash'],
        'calendar_hash':stable_hash(calendar),'symbols':[r['symbol'] for r in records]}
    population.update(trend_catalog_hash=manifest['trend_catalog_hash'],ma_exit_catalog_hash=manifest['ma_exit_catalog_hash'])
    data=pd.read_parquet(pooled);label_counts=_label_counts(data);contract=_target_contract(config,population);bundles=[];unavailable=[]
    print('STAGE train_reference_quality',flush=True)
    reference_folds=[*folds,split_validation(config['production'],calendar)]
    for fold in reference_folds:
        try:
            directory=context.subdir('models',HEAD,fold['name']);m=train_reference_head(data,directory,fold,config,contract)
            bundles.append({'name':fold['name'],'fold':fold,'available_at':m['available_at'],
                'heads':{HEAD:{'path':str(directory.relative_to(root)),'manifest_sha256':m['manifest_sha256']}}})
            print('REFERENCE_QUALITY_HEAD '+fold['name']+' '+str(m['sample_counts']),flush=True)
        except ValueError as exc:
            if 'insufficient mature' not in str(exc) and 'require both classes' not in str(exc):raise
            unavailable.append({'fold':fold['name'],'head':HEAD,'reason':str(exc)})
    del data
    print('STAGE reference_quality_fee_qualification',flush=True)
    results=[]
    for item in _bounded_results(_qualification_worker,((r,calendar,config,str(root),population,bundles,contract) for r in summaries),cpu_workers,config['stock_batch_size']):
        results.append(item)
        print('REFERENCE_QUALITY_PROGRESS '+json.dumps({'stage':'reference_quality_fee_qualification',
            'scope':'current_reference_population_stocks','completed':len(results),'total':len(records),'fold':None}),flush=True)
    windows=[{'window':window,'arms':{name:{'metrics':aggregate_account_metrics([r['windows'][i]['arms'][name] for r in results])}
        for name in ('candidate','rules','candidate_stress','rules_stress')}} for i,window in enumerate(config['development_windows'])]
    gate=selection_gate(windows,protocol['qualification'],len(records),False)
    qualification={'head':HEAD,'population':POPULATION,'population_hash':population['population_hash'],'protocol_hash':protocol['protocol_hash'],
        'contract_hash':contract['contract_hash'],'windows':windows,'assets':[{'symbol':r['symbol'],'files':r['files']} for r in results],
        **gate,'status':'passed' if gate['passed'] else 'pending' if gate['pending'] else 'rejected',
        'publication_allowed':False,'certification_status':'pending','core_selection_gate_changed':False,
        'aggregation':protocol['qualification']['aggregation']}
    qpath=context.subdir('validation','reference_quality')/'qualification.json';_save(qpath,qualification)
    block={'head':HEAD,'protocol':protocol,'protocol_hash':protocol['protocol_hash'],'population':population,'target_contract':contract,
        'bundles':bundles,'unavailable_heads':unavailable,'qualification':{'path':str(qpath.relative_to(root)),
            'sha256':_file_hash(qpath),'report_hash':stable_hash(qualification),'status':qualification['status'],'passed':qualification['passed']},
        'status':'qualified' if qualification['passed'] else 'unqualified','publication_allowed':False,
        'required_for_core_selection':False,'label_counts':label_counts}
    audit_reference_quality(root,block,config,[r['symbol'] for r in records],expected_records=records);return block


def audit_reference_quality(root,block,config,expected_symbols,*,expected_records=None):
    from my_strategy.services.czsc_account_artifacts import validate_account_artifact,read_account_artifact
    from my_strategy.services.model_training_center import selection_gate,aggregate_account_metrics,_event_asset_signature
    from my_strategy.services.czsc_trend_pipeline import GENERATING_FILES
    key=(str(Path(root).resolve()),stable_hash(block),stable_hash(config),tuple(expected_symbols),stable_hash(expected_records))
    def signature(paths):return _event_asset_signature(tuple(Path(p) for p in paths))
    cached=_AUDIT_CACHE.get(key)
    if cached and signature(cached['paths'])==cached['signature']:return copy.deepcopy(cached['report'])
    if (block.get('head')!=HEAD or block.get('required_for_core_selection') is not False
            or block.get('protocol')!=reference_protocol(config) or block.get('protocol_hash')!=block['protocol']['protocol_hash']):
        raise ValueError('reference quality protocol drift')
    p=block['population'];path=_asset(root,p['manifest_path'])
    if _file_hash(path)!=p['manifest_sha256'] or _file_hash(_asset(root,p['pooled_path']))!=p['pooled_sha256']:
        raise ValueError('reference quality frozen population changed')
    manifest=json.loads(path.read_text())
    paths=[path,_asset(root,p['pooled_path']),Path(__file__).parent/'model_training_center.py']
    paths.extend(Path(__file__).parents[1]/relative for relative in GENERATING_FILES)
    if (manifest.get('population_hash')!=stable_hash({k:v for k,v in manifest.items() if k!='population_hash'})
            or manifest.get('population_hash')!=p['population_hash'] or manifest.get('population')!=POPULATION
            or manifest.get('protocol')!=block['protocol'] or manifest.get('failures')
            or [r['symbol'] for r in manifest['records']]!=expected_symbols or p['symbols']!=expected_symbols
            or manifest['source_snapshot_hash']!=config['source_snapshot_hash']
            or manifest['continuation_policy_hash']!=config['E1_policy_hash']
            or manifest['pipeline_contract_hash']!=config['pipeline_contract_hash']
            or manifest['calendar_hash']!=p['calendar_hash'] or manifest['pooled_path']!=p['pooled_path']
            or manifest['pooled_sha256']!=p['pooled_sha256'] or manifest['protocol_hash']!=p['protocol_hash']
            or manifest['trend_catalog_hash']!=p['trend_catalog_hash'] or manifest['ma_exit_catalog_hash']!=p['ma_exit_catalog_hash']):
        raise ValueError('reference quality population source or cohort drift')
    contract=_target_contract(config,p)
    frozen_records=[record['source_record'] for record in manifest['records']]
    if expected_records is not None and frozen_records!=expected_records:
        raise ValueError('reference quality source records differ from complete native source catalogue')
    if block['target_contract']!=contract:raise ValueError('reference quality independent target contract drift')
    pooled=pd.read_parquet(_asset(root,p['pooled_path']))
    if block['label_counts']!=_label_counts(pooled):raise ValueError('reference population label counts differ from real observations')
    def files(records,name):
        for record in records:
            for asset in record['files'].values():
                asset_path=_asset(root,asset['path']);paths.append(asset_path)
                if _file_hash(asset_path)!=asset['sha256']:raise ValueError('reference probe evidence changed')
            path=_asset(root,record['files'][name]['path']);proof=validate_account_artifact(path,expected_sha256=record['files'][name]['sha256'])
            linked=record['files'].get(proof['asset'])
            if not linked or linked['sha256']!=proof['asset_sha256'] or _asset(root,linked['path'])!=path.parent/proof['asset']:
                raise ValueError('reference probe compact graph missing from frozen index')
    files(manifest['records'],'probes.json')
    for bundle in block['bundles']:
        asset=bundle['heads'][HEAD];m=load_reference_quality_head(_asset(root,asset['path']),expected_contract=contract)
        paths.extend((_asset(root,asset['path'])/'manifest.json',_asset(root,asset['path'])/m['checkpoint']))
        if (asset['manifest_sha256']!=m['manifest_sha256'] or bundle['available_at']!=m['available_at']
                or bundle['name']!=m['fold']['name'] or bundle['fold']!=m['fold']
                or bundle['available_at']!=m['fold']['policy_end']):
            raise ValueError('reference head catalog or availability drift')
        train,calibration,policy=time_masks(pooled,m['fold'],head='entry');labels=pooled.label.to_numpy(float)
        for name,mask,end_key in (('train',train,'train_end'),('calibration',calibration,'calibration_end'),('policy',policy,'policy_end')):
            expected={'rows':int(mask.sum()),'positive':int((labels[mask]==1).sum()),'negative':int((labels[mask]==0).sum())}
            cutoff=str(pooled.loc[mask,'label_end'].max()) if mask.any() else m['fold'][end_key]
            if (m['sample_counts'][name]!=expected or m[name+'_label_end']!=cutoff
                    or m['segment_data_digests'][name]!=_rows_digest(pooled.loc[mask])):
                raise ValueError('reference head samples or input digests differ from frozen real probe population')
            if name!='policy' and m['segment_weight_digests'][name]!=hashlib.sha256(_weights(pooled,mask,head='entry').tobytes()).hexdigest():
                raise ValueError('reference head economic weights differ from frozen original opportunity population')
    qa=block['qualification'];qpath=_asset(root,qa['path'])
    paths.append(qpath)
    if _file_hash(qpath)!=qa['sha256']:raise ValueError('reference quality fee qualification asset changed')
    q=json.loads(qpath.read_text())
    if (stable_hash(q)!=qa['report_hash'] or q['population_hash']!=p['population_hash'] or q['contract_hash']!=contract['contract_hash']
            or q['protocol_hash']!=block['protocol_hash'] or q['passed']!=qa['passed'] or q['status']!=qa['status']
            or [r['symbol'] for r in q['assets']]!=expected_symbols):raise ValueError('reference quality fee qualification source drift')
    files(q['assets'],'qualification.json')
    calendar_path=Path(root)/'data_snapshots/calendar.json'
    if calendar_path.exists():
        paths.append(calendar_path);calendar=json.loads(calendar_path.read_text())['dates']
    else:raise ValueError('reference quality calendar evidence missing')
    if stable_hash(calendar)!=p['calendar_hash']:raise ValueError('reference quality calendar identity drift')
    reconstructed=[];offset=0
    for record,asset in zip(manifest['records'],q['assets'],strict=True):
        if record['source_record_sha256']!=record['source_record']['sha256']:
            raise ValueError('reference probe source identity differs')
        for field in ('path','native_path','raw_path'):
            if record['source_record'].get(field):paths.append(resolve_frozen_asset(record['source_record'][field]))
        measured=_qualification_worker((record,calendar,config,str(root),p,block['bundles'],contract),persist=False)
        count=record['rows']
        if measured['rows_digest']!=_rows_digest(pooled.iloc[offset:offset+count]):
            raise ValueError('reference pooled rows differ from complete ordered stock probe observations')
        offset+=count
        persisted=read_account_artifact(_asset(root,asset['files']['qualification.json']['path']),expected_sha256=asset['files']['qualification.json']['sha256'])
        if measured['windows']!=persisted['windows']:
            raise ValueError('reference fee evidence differs from actual Broker ledger prefixes and prior model predictions')
        reconstructed.append(measured)
    if offset!=len(pooled):raise ValueError('reference pooled population has extra or missing probe rows')
    actual_windows=[{'window':window,'arms':{name:{'metrics':aggregate_account_metrics([r['windows'][i]['arms'][name] for r in reconstructed])}
        for name in ('candidate','rules','candidate_stress','rules_stress')}} for i,window in enumerate(config['development_windows'])]
    if q['windows']!=actual_windows:raise ValueError('reference fee summary differs from actual full-population probe accounts')
    gate=selection_gate(q['windows'],block['protocol']['qualification'],len(expected_symbols),False)
    if any(q[k]!=gate[k] for k in ('passed','pending','reasons','completed_round_trips','paired_statistics')):
        raise ValueError('reference quality fee gate differs from frozen measured metrics')
    paths=list(dict.fromkeys(paths));_AUDIT_CACHE[key]={'paths':paths,'signature':signature(paths),'report':copy.deepcopy(q)}
    return q


@dataclass(frozen=True)
class ReferenceQualityScore:
    status: str
    probability: float | None
    reason_codes: tuple
    date: str
    symbol: str
    reference_price: float | None
    risk_identity_hash: str | None
    opportunity_hash: str
    reference_population_hash: str | None
    qualification_hash: str | None
    config_hash: str
    run_id: str | None
    model_identity: tuple = ()
    fee_qualified: bool = False
    _issuer: object = field(default=None,repr=False,compare=False)
    _seal: str = field(default='',repr=False,compare=False)


def _score_seal(score):
    body={key:getattr(score,key) for key in score.__dataclass_fields__ if not key.startswith('_')}
    return hmac.new(_SEAL_KEY,stable_hash(body).encode(),hashlib.sha256).hexdigest()


def validate_reference_quality_score(score,row,risk,config,run_id=None):
    if not isinstance(score,ReferenceQualityScore) or score._issuer is not _ISSUER or not hmac.compare_digest(score._seal,_score_seal(score)):
        return ['reference_quality_untrusted_or_changed_result']
    reasons=[]
    if (score.date!=row['date'] or score.symbol!=row['symbol'] or score.reference_price!=row['reference_price']
            or score.opportunity_hash!=stable_hash(sorted(set(row.get('original_opportunity_ids') or [])))
            or score.config_hash!=stable_hash(config) or run_id is not None and score.run_id!=run_id):
        reasons.append('reference_quality_reference_context_mismatch')
    if score.status=='scored' and (not score.fee_qualified or score.probability is None or not 0<=score.probability<=1
            or not score.reference_population_hash or not score.qualification_hash
            or score.risk_identity_hash!=risk.get('original_buy_plan_hash') or score.risk_identity_hash!=stable_hash(row.get('entry_plan'))):
        reasons.append('reference_quality_scored_population_or_risk_mismatch')
    return reasons


class ReferenceQualitySession:
    def __init__(self,root,report,config=None,usage_mode='historical'):
        self.root=Path(root).resolve();self.report=report;self.config=copy.deepcopy(config or report['config'])
        self.usage_mode=usage_mode;self.cache={};self.block=report.get('reference_quality');self.qualification=None
        if self.usage_mode not in ('historical','forward','production'):raise ValueError('unknown reference quality usage mode')
        if self.block:
            self.qualification=audit_reference_quality(self.root,self.block,report['config'],[r['symbol'] for r in report['dataset_records']],expected_records=report['dataset_records'])
    def score(self,feature,prepared_row,event_prefix,risk_plan=None):
        from my_strategy.services.czsc_trend_runtime import _reference_risk_plan
        row=prepared_row;risk=risk_plan or {};plan=reference_probe_plan(row,risk,self.config)
        reasons=[];p=m=None
        if self.config!=self.report.get('config',self.config):reasons.append('reference_quality_policy_requires_new_population')
        if not self.block:reasons.append('reference_quality_population_unavailable')
        elif not self.qualification['passed']:reasons.append('reference_population_fee_qualification_not_passed')
        if not plan:reasons.append('reference_original_BUY_risk_population_ineligible')
        choices=[b for b in self.report.get('trend_bundles',[]) if b['available_at']<row['date']]
        expected=max(choices,key=lambda b:(b['available_at'],b['name'])) if choices else None
        identity=row['agents']['ma'].get('model_identity') or {}
        if self.block and (not expected or identity.get('checkpoint')!=expected['name']
                or identity.get('manifest_sha256')!=expected['heads']['trend_direction']['manifest_sha256']
                or self.block['population']['trend_catalog_hash']!=stable_hash(self.report.get('trend_bundles',[]))
                or self.block['population']['ma_exit_catalog_hash']!=stable_hash(self.report.get('ma_exit_bundles',[]))):
            reasons.append('reference_quality_trend_route_out_of_population')
        if not reasons:
            values=reference_feature_row(feature,row,event_prefix,risk)
            p,m=_predict_prior(values,row['date'],self.block['bundles'],self.root,self.block['target_contract'],self.cache,self.usage_mode)
            if p is None:reasons.append('no_strictly_prior_reference_quality_head')
        identity={} if m is None else {'family':STRATEGY_ID,'head':HEAD,'population':POPULATION,
            'manifest_sha256':m['manifest_sha256'],'feature_schema_hash':SCHEMA_HASH,'target_contract_hash':m['target_contract_hash'],
            'checkpoint':m['fold']['name'],'train_label_end':m['train_label_end'],'calibration_label_end':m['calibration_label_end'],
            'policy_label_end':m['policy_label_end'],'available_at':m['available_at'],
            'actual_training_completed_at':m['actual_training_completed_at'],'model_run_id':self.root.name}
        score=ReferenceQualityScore('scored' if p is not None and not reasons else 'pending',p,tuple(reasons),row['date'],row['symbol'],
            row['reference_price'],stable_hash(plan) if plan else None,stable_hash(sorted(set(row.get('original_opportunity_ids') or []))),
            self.block['population']['population_hash'] if self.block else None,
            self.block['qualification']['report_hash'] if self.block else None,stable_hash(self.config),self.root.name,
            tuple(sorted(identity.items())),bool(self.qualification and self.qualification['passed']),_ISSUER)
        return replace(score,_seal=_score_seal(score))
