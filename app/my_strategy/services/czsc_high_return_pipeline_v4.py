"""Versioned native population -> actual Broker labels -> exit-bound L1 -> L2.

Historical scores reconstruct chronological availability. No activity alias or
publication event is modified. Failed folds and unclosed labels are retained.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
import pandas as pd
from my_strategy.core.paths import MY_STRATEGY_ROOT, ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_research import _save, _file_hash as _hash_file, _json
from my_strategy.services.czsc_high_return_research import _source, _read_features, _ma_folds
from my_strategy.services.czsc_signal_points_v4 import build_signal_points, make_point_decisions, policy_identity, CANDIDATE_POLICY, RISK_POLICY
from my_strategy.services.czsc_high_return_features_v4 import (L1_MA_ENTRY_COLUMNS,L1_MA_EXIT_COLUMNS,ENTRY_FEATURE_COLUMNS,EXIT_FEATURE_COLUMNS,
    entry_feature_row,holding_feature_row,event_prefix_rows)
from my_strategy.services.czsc_high_return_models_v4 import HeadSession,train_head,load_head,schema_hash,split_validation
from my_strategy.services.czsc_trade_episodes_v4 import build_episode_dataset,training_frame,_settings,_max_exit_callback
from my_strategy.services.czsc_high_return_labels_v4 import label_config
from my_strategy.services.czsc_high_return_runtime import _head_path
from my_strategy.storage.czsc_results import ResultStore
VERSION='czsc_high_return_v4_native'
KIND='czsc-high-return-native-v4-train'
ENVELOPE='risk_qualified_fresh_native_union_v4'

def _file_hash(path):
    return _hash_file(Path(path))


def research_config(config=None):
    from my_strategy.services.czsc_analysis import strategy_config
    cfg=json.loads((MY_STRATEGY_ROOT/'configs/czsc_high_return_research.json').read_text())
    for key,value in (config or {}).items():
        cfg[key]={**cfg.get(key,{}),**value} if isinstance(value,dict) else value
    cfg.update(version=VERSION,ma_entry_alignment='signal_pinned_union_exit_v4',risk_policy=cfg.get('risk_policy',RISK_POLICY),
        candidate_policy=CANDIDATE_POLICY,candidate_envelope=ENVELOPE,publication_allowed=False)
    cfg['strategy']=strategy_config(cfg.get('strategy'))
    cfg['native_policy']={'version':'czsc_native_evidence_v4_20261012',
        'candidate_policy':CANDIDATE_POLICY,'activation':'first_risk_eligibility_once_while_original_still_valid',
        'risk_policy':cfg['risk_policy'],'signal_points':cfg['signal_points'],
        'continuous_support_does_not_create_candidate':True}
    return cfg




def _bundle_for(day,bundles,*,head=None,model_root=None,usage_mode='historical',validation_bundle=None):
    if validation_bundle is not None:
        fold=validation_bundle['fold']
        return validation_bundle if validation_bundle.get('heads',{}).get(head) and fold['calibration_end']<day and fold['policy_start']<=day<=fold['policy_end'] else None
    eligible=[b for b in bundles if b['available_at']<day and (head is None or b.get('heads',{}).get(head))]
    if usage_mode=='production':
        cutoff=pd.Timestamp(day+'T15:00:00+08:00')
        eligible=[b for b in eligible if pd.Timestamp(load_head(_head_path(model_root,b,head))['training_completed_at'])<=cutoff]
    return max(eligible,key=lambda b:(b['available_at'],b['name'])) if eligible else None


def _native_worker(payload):
    record,cfg,end,directory,calendar=payload
    from my_strategy.services.czsc_native_evidence_v4 import build_native_evidence_v4
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    frame=_read_features(record,cfg,end)
    pack=build_native_evidence_v4(frame,cfg,market_dates=calendar)
    target=Path(directory)/record['symbol'].replace('.','_'); target.mkdir(parents=True,exist_ok=True)
    numeric=pack['frame'].drop(columns=[c for c in pack['frame'] if pack['frame'][c].map(lambda v:isinstance(v,(dict,list,tuple))).any()])
    numeric.attrs={}; numeric.to_parquet(target/'features.parquet',index=False)
    _save(target/'native.json',{k:v for k,v in pack.items() if k!='frame'})
    return {**record,'original_path':record['path'],'original_sha256':record['sha256'],
        'path':str(target/'features.parquet'),'sha256':_file_hash(target/'features.parquet'),
        'native_path':str(target/'native.json'),'native_sha256':_file_hash(target/'native.json'),
        'native_config_hash':pack['config_hash'],'native_schema_hash':pack['schema_hash'], 'rows':len(numeric)}


def read_native(record):
    if _file_hash(record['path'])!=record['sha256'] or _file_hash(record['native_path'])!=record['native_sha256']:
        raise ValueError('native snapshot drift')
    frame=pd.read_parquet(record['path']);pack=json.loads(Path(record['native_path']).read_text());pack['frame']=frame
    return frame,pack


def _predict(session,values,root,bundle,head,*,validation=False,mode='historical',policy=None,source_snapshot_hash=None):
    path=_head_path(root,bundle,head)
    if path is None:return None,None
    key=str(Path(path).resolve())
    manifest=getattr(session,'cache',{}).get(key,(None,None))[0] or load_head(path)
    contract=manifest['target_contract']
    if contract.get('research_version')!=VERSION or contract.get('candidate_policy')!=CANDIDATE_POLICY:
        raise ValueError('native model target population mismatch')
    if policy is not None and any(contract.get(k)!=policy.get(k) for k in ('candidate_policy_version','candidate_policy_hash','risk_policy_hash')):
        raise ValueError('native model candidate/risk policy mismatch')
    if source_snapshot_hash is not None and contract.get('source_snapshot_hash')!=source_snapshot_hash:
        raise ValueError('native model training source snapshot mismatch')
    if manifest['manifest_sha256']!=bundle['heads'][head]['manifest_sha256'] or manifest['available_at']!=bundle['available_at'] or manifest['fold']['name']!=bundle['name']:
        raise ValueError('native model catalog binding drift')
    p=session.predict_one(values,path,expected_head=head,expected_contract=contract,validation=validation,
        usage_mode='production' if mode=='production' else 'historical_reconstruction')
    return float(p) if np.isfinite(p) else None,manifest


def pinned_exit_policy(frame,calendar,points,root,bundles,cfg,*,source='union',session=None):
    """Independent 20+10 MA exit frozen at entry, with fresh state per replay."""
    session=session or HeadSession();root=Path(root)
    bundles=[{**copy.deepcopy(b),'heads':{'ma_exit':copy.deepcopy(b['heads']['ma_exit'])}} for b in bundles if b.get('heads',{}).get('ma_exit')]
    plans=make_point_decisions(frame,points,source=source,config=cfg,include_unselected=True)
    bindings={}
    for row in plans:
        plan=row.get('entry_plan')
        if not plan:continue
        bundle=_bundle_for(row['date'],bundles,head='ma_exit',model_root=root)
        if bundle:
            manifest=load_head(_head_path(root,bundle,'ma_exit'))
            if manifest['train_label_end']>=row['date'] or manifest['calibration_label_end']>=row['date']:
                raise ValueError('exit head has immature labels')
        bindings[plan['plan_id']]={'candidate_signal_date':row['date'],'plan_id':plan['plan_id'],
            'checkpoint':bundle['name'] if bundle else None,'available_at':bundle['available_at'] if bundle else None,
            'bundle':bundle,'mode':'signal_time_pinned_ma_exit' if bundle else 'explicit_rule_v0_no_earlier_exit_head',
            'checkpoint_upgrade':False,'source':source}
    recipe={'version':'native_union_signal_pinned_ma_exit_v4','source':source,'catalog_hash':stable_hash(bundles),
        'points_policy_hash':points['config_hash'],'binding_rule':'strict_prior_head_at_candidate_signal_once','checkpoint_upgrade':False}
    recipe_hash=stable_hash(recipe);rows={r['date']:r for r in frame.to_dict('records')};midx={d:i for i,d in enumerate(calendar)}
    next_days=dict(zip(calendar[:-1],calendar[1:]))
    def fresh():
        prefixes={}; diagnostics=[]
        def callback(snapshot):
            key=snapshot.get('plan_id');binding=bindings.get(key)
            if binding is None or snapshot['entry_date']!=next_days.get(binding['candidate_signal_date']):
                raise ValueError('holding has no signal-time binding')
            prefix=prefixes.setdefault((key,snapshot['entry_date']),{})
            if prefix.get('through_date','')>=snapshot['date']:raise ValueError('reused holding prefix')
            prefix['through_date']=snapshot['date']
            values=holding_feature_row(rows[snapshot['date']],snapshot,entry_values={},prefix=prefix,market_index=midx)
            values['input_eligible']=bool(rows[snapshot['date']].get('model_input_eligible'))
            p,manifest=_predict(session,values,root,binding['bundle'],'ma_exit',policy=points['policy'],source_snapshot_hash=cfg.get('source_snapshot_hash')) if binding['bundle'] else (None,None)
            result={'head':'holding_exit','contract_hash':manifest['target_contract_hash'] if manifest else recipe_hash,
                'probability':p,'request_exit':p is not None and p>=cfg['signal_points']['ma_exit_threshold'],
                'apply':p is not None,'research_arm':True,'holding_contract_compatible':True,
                'status':binding['mode'],'candidate_exit_binding':{k:v for k,v in binding.items() if k!='bundle'},
                'prefix_through_date':snapshot['date'],'holding_prefix_source':'actual_broker_daily_close_snapshots'}
            diagnostics.append({**result,'date':snapshot['date']});return result
        callback.new_replay=fresh;callback.recipe=recipe;callback.recipe_hash=recipe_hash;callback.bindings=bindings;callback.diagnostics=diagnostics
        return callback
    return fresh()


def _head_data(data,base,head,columns):
    values=data.copy();contract={k:v for k,v in base.items() if k!='contract_hash'}
    contract.update(model_head=head,model_input_schema_hash=schema_hash(columns,head),
        output_type='continuous' if head=='exit_advantage' else 'probability')
    if head=='exit_advantage':values['label']=values.exit_advantage
    contract['contract_hash']=stable_hash(contract);values.attrs['label_contract']=contract
    return values,contract


def _fit(data,base,head,raw_fold,calendar,context,cfg,architecture='linear',device='cpu'):
    columns={'ma_entry':L1_MA_ENTRY_COLUMNS,'ma_exit':L1_MA_EXIT_COLUMNS,'entry':ENTRY_FEATURE_COLUMNS,'exit':EXIT_FEATURE_COLUMNS,
        'exit_advantage':EXIT_FEATURE_COLUMNS}[head]
    fold=split_validation(raw_fold,calendar);values,contract=_head_data(data,base,head,columns)
    target=context.subdir('models',head,fold['name'],architecture)
    manifest=train_head(values,columns,target,fold,cfg,head=head,architecture=architecture,device=device,
        contract=contract,regression=head=='exit_advantage')
    return {'name':fold['name'],'available_at':fold['validation_end'],'fold':fold,'architecture':architecture,
        'heads':{head:{'path':str(target.relative_to(context.run_dir())),'manifest_sha256':manifest['manifest_sha256']}},
        'policy':{'entry_threshold':cfg['default_entry_threshold'],'exit_threshold':.55},'publication_allowed':False},manifest


def forward_ma(frame,points,root,bundles,cfg,session=None,mode='historical'):
    session=session or HeadSession();cand={c['date']:c for c in points['candidates']};routes={}
    plans={r['date']:r.get('entry_plan') or {} for r in make_point_decisions(frame,points,config=cfg,include_unselected=True)}
    for row in frame.to_dict('records'):
        day=row['date'];candidate=cand.get(day);in_scope=bool(candidate and candidate['candidate_eligible'] and plans[day].get('entry_allowed'))
        bundle=_bundle_for(day,bundles,head='ma_entry',model_root=root,usage_mode=mode)
        values={**row,'input_eligible':bool(row.get('model_input_eligible'))}
        p,manifest=_predict(session,values,root,bundle,'ma_entry',mode=mode,policy=points['policy'],source_snapshot_hash=cfg.get('source_snapshot_hash')) if bundle else (None,None)
        identity=model_identity(manifest,bundle,root,day,points['policy'])
        routes[day]={'date':day,'available_at':day+'T15:00:00+08:00','probability':p,'in_scope':in_scope,
            'model_identity':identity,'lineage':identity,'target':'actual_complete_fee_net_roundtrip_positive',
            'mode':'forward_model' if bundle else 'causal_rule_v0','reason_codes':[] if p is not None else ['model_unavailable_at_signal'],
            'candidate_envelope':{'version':ENVELOPE,'in_scope':in_scope,'candidate_event_ids':candidate['source_event_ids'] if candidate else []}}
    return routes


def model_identity(manifest,bundle,root,day,policy):
    if manifest is None:return {'model_run_id':Path(root).name if root and Path(root).resolve()!=ARTIFACT_RUNS_ROOT.resolve() else None,'score_date':day,**policy}
    frozen_policy={k:manifest['target_contract'].get(k) for k in ('candidate_policy_version','candidate_policy_hash','risk_policy_hash')}
    if frozen_policy!=policy:raise ValueError('scored identity cannot impersonate current policy')
    return {**frozen_policy,'model_run_id':Path(root).name,'checkpoint':bundle['name'],'model_id':bundle['name'],
        'manifest_sha256':manifest['manifest_sha256'],'checkpoint_sha256':manifest['checkpoint_sha256'],
        'target_contract_hash':manifest['target_contract_hash'],'feature_schema_hash':manifest['feature_schema_hash'],
        'source_snapshot_hash':manifest['target_contract']['source_snapshot_hash'],'model_head':manifest['head'],
        'train_label_end':manifest['train_label_end'],'label_cutoff':manifest['train_label_end'],
        'calibration_label_end':manifest['calibration_label_end'],'available_at':manifest['available_at'],
        'actual_training_completed_at':manifest['training_completed_at'],'score_date':day,
        'verified_frozen_contract':True,'research_version':VERSION,'target':manifest['target_contract']['target'],
        'historical_reconstruction':True}


def _stage_worker(payload):
    record,calendar,cfg,context,stage,bundles,ma_scores=payload
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    frame,pack=read_native(record);points=build_signal_points(frame,cfg,native_pack=pack)
    if ma_scores:
        routes=forward_ma(frame,points,context.run_dir(),bundles,cfg)
        points=build_signal_points(frame,cfg,native_pack=pack,ma_routes=routes)
    policy=pinned_exit_policy(frame,calendar,points,context.run_dir(),bundles,cfg) if bundles else None
    target=context.subdir('dataset',stage,record['symbol'].replace('.','_'))
    def save_account(source,symbol,account):
        d=target/'reference'/source;d.mkdir(parents=True,exist_ok=True)
        for key in ('ledger','daily','rejections','holding_snapshots'):_save(d/(key+'.json'),account.get(key,[]))
        _save(d/'summary.json',account['metrics'])
    dataset=build_episode_dataset(frame,calendar,cfg,points=points,start=cfg['feature_start'],
        initial_cash=cfg['initial_cash'],source_snapshot_hash=cfg['source_snapshot_hash'],sources=('union',),
        exit_policies={'union':policy} if policy else None,exit_policy_hashes={'union':policy.recipe_hash} if policy else None,
        reference_callback=save_account)
    # The shared recipe binds the checkpoint catalog/policy, not a stock's
    # actual per-plan bindings. Store distinct stock recipe separately and
    # use a common frozen catalog identity for compatible cohort targets.
    _save(target/'points.json',points)
    if policy:_save(target/'exit_bindings.json',policy.bindings)
    for key in ('entries','exits','episodes'):
        dataset[key].to_parquet(target/(key+'.parquet'),index=False)
    _save(target/'contracts.json',dataset['contracts'])
    summary={'symbol':record['symbol'],'coverage':dataset['coverage'],'points':points['coverage'],
        'path':str(target),'contracts':dataset['contracts']}
    print(json.dumps({'stage':stage,'symbol':record['symbol'],'coverage':dataset['coverage']}),flush=True)
    return {k:dataset[k] for k in ('entries','exits','episodes','contracts')},summary

def _build_stage(records,calendar,cfg,context,stage,*,bundles=None,ma_scores=False):
    payloads=[(r,calendar,cfg,context,stage,bundles,ma_scores) for r in records]
    workers=int(cfg.get('cpu_workers',4))
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:done=list(pool.map(_stage_worker,payloads))
    else:done=[_stage_worker(p) for p in payloads]
    summaries=[item[1] for item in done]
    chunks={k:[item[0][k] for item in done] for k in ('entries','exits','episodes')}
    # All symbols share one frozen policy recipe; their actual holdings and
    # checkpoint choices are row-level bindings, never rewritten contracts.
    contracts=summaries[0]['contracts']
    if any(item['contracts']!=contracts for item in summaries):
        raise ValueError('mixed native cohort continuation contracts')
    combined={k:pd.concat(v,ignore_index=True) for k,v in chunks.items()}
    combined['contracts']=contracts
    _save(context.subdir('dataset',stage)/'manifest.json',{'records':summaries,'requested':len(records),'successful':len(records),'failures':[]})
    return combined,summaries


def train_native_research(*,symbols=None,end='2026-10-08',source_training_run_id=None,config=None,cpu_workers=4,device='cpu'):
    cfg=research_config(config);cfg['cpu_workers']=cpu_workers;original,calendar,old_snapshot,end=_source(source_training_run_id or cfg['source_training_run_id'],end,symbols)
    if any(not (r['symbol'].startswith('00') or r['symbol'].startswith('60')) for r in original):
        raise ValueError('freeze supported mainboard cohort; unsupported cash accounts belong to evaluation coverage')
    context=create_run_context(task='czsc-native-v4-snapshot-build',as_of_date=end,config=cfg,data_version=old_snapshot['snapshot_hash'],
        seed=cfg['seed'],stocks=[r['symbol'] for r in original],scope='train',start_date=cfg['feature_start'],end_date=end)
    print('RUN_ID '+context.run_id,flush=True)
    _save(context.subdir('config')/'research_initial.json',cfg)
    _save(context.subdir('config')/'native_source_hashes.json',{str(p):_file_hash(p) for p in (MY_STRATEGY_ROOT/'services').glob('*v4*.py')})
    _save(context.subdir('data_snapshots')/'original_source.json',old_snapshot)
    payloads=[(r,cfg,end,str(context.subdir('data_snapshots','native')),calendar) for r in original]
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:records=list(pool.map(_native_worker,payloads))
    else:records=[_native_worker(p) for p in payloads]
    if len({r['native_config_hash'] for r in records})!=1:raise ValueError('mixed native policies')
    snapshot={'source_run_id':old_snapshot['run_id'],'source_snapshot_hash':old_snapshot['snapshot_hash'],
        'source_data_end':old_snapshot['source_data_end'],'effective_data_end':end,'native_records_hash':stable_hash(records),
        'selected_count':len(records),'full_source_universe':False,'selected_symbols':[r['symbol'] for r in records],
        'native_config_hash':records[0]['native_config_hash'],'native_schema_hash':records[0]['native_schema_hash'],'snapshot_run_id':context.run_id}
    snapshot['snapshot_hash']=stable_hash(snapshot);cfg['source_snapshot_hash']=snapshot['snapshot_hash']
    cfg['native_policy']['native_config_hash']=stable_hash({'config':snapshot['native_config_hash'],'schema':snapshot['native_schema_hash']})
    snapshot_context=context
    _save(snapshot_context.subdir('data_snapshots')/'source.json',snapshot)
    _save(snapshot_context.subdir('data_snapshots')/'records.json',records)
    snapshot_context.write_metadata({'status':'complete','config':research_config(config),'source_snapshot':snapshot})
    context=create_run_context(task='czsc-high-return-native-v4-train',as_of_date=end,config=cfg,data_version=snapshot['snapshot_hash'],
        seed=cfg['seed'],stocks=[r['symbol'] for r in original],scope='train',start_date=cfg['feature_start'],end_date=end)
    print('TRAIN_RUN_ID '+context.run_id,flush=True)
    generator_names=['czsc_native_evidence_v4.py','czsc_signal_points_v4.py','czsc_high_return_features_v4.py',
        'czsc_high_return_labels_v4.py','czsc_high_return_models_v4.py','czsc_trade_episodes_v4.py','czsc_high_return_pipeline_v4.py']
    generating_code={str(MY_STRATEGY_ROOT/'services'/name):_file_hash(MY_STRATEGY_ROOT/'services'/name) for name in generator_names}
    _save(context.subdir('config')/'generating_code_hashes.json',generating_code)
    for name in generator_names:
        target=context.subdir('config','source_code')/name;target.write_bytes((MY_STRATEGY_ROOT/'services'/name).read_bytes())
    _save(context.subdir('config')/'research.json',cfg);_save(context.subdir('data_snapshots')/'source.json',snapshot)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'dates':calendar,'source':snapshot})
    _save(context.subdir('data_snapshots')/'records.json',records)
    bootstrap,_=_build_stage(records,calendar,cfg,context,'bootstrap')
    exit_data=training_frame(bootstrap,source='union',head='exit',ma_only=True)
    exit_base=exit_data.attrs['label_contract'];exit_bundles=[];failures=[];fitted=[]
    for fold in _ma_folds(end):
        try:
            bundle,manifest=_fit(exit_data,exit_base,'ma_exit',fold,calendar,context,cfg,device=device);exit_bundles.append(bundle);fitted.append(manifest)
        except ValueError as exc:failures.append({'stage':'ma_exit','fold':fold['name'],'error':str(exc)})
    _save(context.subdir('models')/'ma_exit_bundles.json',exit_bundles)
    derived,derived_records=_build_stage(records,calendar,cfg,context,'exit_bound_entry',bundles=exit_bundles)
    entry_data=training_frame(derived,source='union',head='entry',ma_only=True)
    ma_bundles=[]
    for fold in _ma_folds(end):
        try:
            bundle,manifest=_fit(entry_data,entry_data.attrs['label_contract'],'ma_entry',fold,calendar,context,cfg,device=device)
            matching=next((b for b in exit_bundles if b['name']==bundle['name']),None)
            if matching:bundle['heads'].update(matching['heads'])
            ma_bundles.append(bundle);fitted.append(manifest)
        except ValueError as exc:failures.append({'stage':'ma_entry','fold':fold['name'],'error':str(exc)})
    _save(context.subdir('models')/'ma_bundles.json',ma_bundles)
    final,final_records=_build_stage(records,calendar,cfg,context,'l2_final',bundles=ma_bundles,ma_scores=True)
    entry=training_frame(final,source='union',head='entry');exits=training_frame(final,source='union',head='exit')
    l2=[];architecture_comparisons=[]
    for fold in [*cfg['development_windows'],cfg['production']]:
        candidates=[]
        for architecture in cfg['architectures']:
            try:
                bundle,manifest=_fit(entry,entry.attrs['label_contract'],'entry',fold,calendar,context,cfg,architecture,device)
                candidates.append((bundle,manifest));fitted.append(manifest)
                architecture_comparisons.append({'fold':fold['name'],'architecture':architecture,'metrics':manifest.get('calibration_metrics'),'sample_counts':manifest['sample_counts']})
            except ValueError as exc:failures.append({'stage':'l2_entry','fold':fold['name'],'architecture':architecture,'error':str(exc)})
        if candidates:
            # Comparison reports remain inspectable; fixed linear is the first
            # frozen policy arm. No post-hoc classifier metric promotes a head.
            chosen=next((b for b,m in candidates if b['architecture']=='linear'),candidates[0][0]);l2.append(chosen)
            for head in ('exit','exit_advantage'):
                try:
                    b,m=_fit(exits,exits.attrs['label_contract'],head,fold,calendar,context,cfg,chosen['architecture'],device)
                    chosen['heads'].update(b['heads']);fitted.append(m)
                except ValueError as exc:failures.append({'stage':head,'fold':fold['name'],'error':str(exc)})
    _save(context.subdir('models')/'l2_bundles.json',l2)
    pipeline={'version':VERSION,'candidate_envelope':ENVELOPE,'bootstrap':'raw_union_rule_v0_no_entry_head',
        'independent_exit_heads':len(exit_bundles),'exit_bound_entry_heads':len(ma_bundles),
        'holding_checkpoint_upgrade':False,'entry_heads_used_in_bootstrap':0,'stages':['bootstrap','ma_exit','exit_bound_entry','ma_entry','l2_final','l2'],
        'final_mature_entries':int(final['entries'].label_available.sum()),'final_entry_rows':len(final['entries']),
        'final_mature_exits':int(final['exits'].label_available.sum()),'final_exit_rows':len(final['exits']),
        'final_mature_positives':int(final['entries'].label.eq(1).sum())}
    report={'run_id':context.run_id,'status':'completed' if not failures else 'completed_with_fold_failures','config':cfg,
        'strategy_version':VERSION,'model_family':'high_return_native_v4','source_snapshot':snapshot,'data_end':end,
        'dataset_records':records,'ma_bundles':ma_bundles,'model_bundles':l2,'pipeline':pipeline,'failures':failures,
        'architecture_comparisons':architecture_comparisons,'publication_allowed':False,
        'model_gate':{'passed':False,'reasons':['independent_windows_not_mature','partial_frozen_cohort_development_only']},
        'completed_at':local_now().isoformat(),'run_context':context.to_dict()}
    freeze={'research_version':VERSION,'actual_freeze_at':local_now().isoformat(),'config_hash':stable_hash(cfg),
        'source_snapshot_hash':snapshot['snapshot_hash'],'ma_bundles_hash':stable_hash(ma_bundles),'l2_bundles_hash':stable_hash(l2),
        'native_files':{r['symbol']:r['native_sha256'] for r in records},
        'code_hashes':generating_code,
        'independent_window_status':'pending','read_history_is_development_only':True}
    if any(_file_hash(path)!=sha for path,sha in generating_code.items()):raise ValueError('generating code drifted during labels/models fitting')
    freeze['freeze_hash']=stable_hash(freeze);report['protocol_freeze']=freeze
    _save(context.subdir('validation')/'protocol_freeze.json',freeze);_save(context.subdir('reports')/'native_v4.json',report)
    ResultStore().save(KIND,_json(report));context.write_metadata({'status':report['status'],'config':cfg,'publication_allowed':False})
    print(json.dumps({'run_id':context.run_id,'pipeline':pipeline,'failures':failures}),flush=True)
    return report
