"""Trend forecasts -> E0 actual accounts -> pinned MA exit E1 -> new L2.

Every trained head has a new target, population and continuation contract.
Independent forward acceptance remains pending; historical reconstruction is
explicitly separated from actual model completion time.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash, create_run_context
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_research import _save, _file_hash
from my_strategy.services.czsc_trend_contracts import strategy_config, STRATEGY_ID, TRAIN_KIND
from my_strategy.services.czsc_trend_model import (TrendSession, SCHEMA_HASH, bind_trend_inputs,
    build_trend_labels, chronological_folds, load_trend_head, train_trend_head, trend_config)
from my_strategy.services.czsc_trend_features import (ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS,
    L1_MA_EXIT_COLUMNS, entry_feature_row, holding_feature_row)
from my_strategy.services.czsc_trend_interval_model import HeadSession, load_head, train_head, schema_hash, bind_inputs


def read_run(model_run_id=None):
    if model_run_id is None:
        candidates=[]
        for directory in ARTIFACT_RUNS_ROOT.iterdir():
            if TRAIN_KIND not in directory.name:continue
            path=directory/'reports'/'trend_consensus.json'
            if not path.is_file():continue
            report=json.loads(path.read_text())
            if report.get('strategy_version')==STRATEGY_ID and report.get('status')=='completed' and not report.get('failures') and report.get('continuation_stage')=='E1' and report.get('automatic_selection_eligible') is True:
                candidates.append((directory.name,directory,report))
        if not candidates:return None,None
        _,directory,report=max(candidates,key=lambda x:x[0]);_verify_run(report,directory);return directory,report
    directory=(ARTIFACT_RUNS_ROOT/str(model_run_id)).resolve()
    if not directory.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):raise ValueError('model run outside artifact root')
    path=directory/'reports'/'trend_consensus.json'
    if not path.exists():raise ValueError('completed trend model report unavailable')
    report=json.loads(path.read_text())
    if report.get('strategy_version')!=STRATEGY_ID or report.get('status')!='completed' or report.get('failures') or report.get('continuation_stage')!='E1':
        raise ValueError('incomplete, failed or incompatible trend model run')
    _verify_run(report,directory);return directory,report


def _verify_run(report,directory):
    freeze=report.get('protocol_freeze') or {}
    if freeze.get('freeze_hash')!=stable_hash({k:v for k,v in freeze.items() if k!='freeze_hash'}):raise ValueError('trend run freeze identity invalid')
    if stable_hash(report['config'])!=freeze.get('config_hash'):raise ValueError('trend frozen config drift')
    if not freeze.get('code_hashes') or len(freeze['code_hashes'])<9:raise ValueError('trend generating-code freeze incomplete')
    catalogs={'trend':report['trend_bundles'],'ma_exit':report['ma_exit_bundles'],'architecture':report['architecture_bundles']}
    if stable_hash(catalogs)!=freeze.get('models_catalog_hash'):raise ValueError('trend model catalog/policy drift')
    if report['source_snapshot'].get('snapshot_hash')!=report['config'].get('source_snapshot_hash'):
        raise ValueError('trend source snapshot binding mismatch')
    if stable_hash(report['dataset_records'])!=report['source_snapshot']['records_hash']:raise ValueError('trend input catalog drift')
    for path,sha in freeze.get('code_hashes',{}).items():
        if _file_hash(Path(path))!=sha:raise ValueError('trend generating code drift requires new training')
    for bundle in [*report['trend_bundles'],*report['ma_exit_bundles'],*report['architecture_bundles']]:
        for head,record in bundle['heads'].items():
            path=(directory/record['path']).resolve()
            if not path.is_relative_to(directory.resolve()):raise ValueError('trend model path outside run')
            manifest=load_trend_head(path) if head=='trend_direction' else load_head(path)
            if manifest['manifest_sha256']!=record['manifest_sha256']:raise ValueError('trend catalog manifest drift')
            target=manifest['target_contract']
            if target['source_snapshot_hash']!=report['config']['source_snapshot_hash'] or target['pipeline_contract_hash']!=report['config']['pipeline_contract_hash']:
                raise ValueError('trend head belongs to another source/protocol')
            if bundle['available_at']!=manifest['fold']['policy_end']:raise ValueError('trend head availability backdated')


def read_snapshot(record):
    for name,sha in (('path','sha256'),('native_path','native_sha256')):
        if _file_hash(Path(record[name]))!=record[sha]:raise ValueError('frozen native input drift')
    frame=pd.read_parquet(record['path']);frame['date']=pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    pack=json.loads(Path(record['native_path']).read_text());pack['frame']=frame
    frame.attrs.update(volume_unit='shares',amount_unit='CNY',price_basis='unadjusted')
    return frame,pack


def _bundle(day,bundles,head,root,mode='historical'):
    valid=[]
    for bundle in bundles:
        if bundle.get('available_at','9999')>=day or not bundle.get('heads',{}).get(head):continue
        if mode=='production':
            path=Path(root)/bundle['heads'][head]['path']
            m=load_trend_head(path) if head=='trend_direction' else load_head(path)
            completed=m.get('actual_training_completed_at',m.get('training_completed_at'))
            if pd.Timestamp(completed)>pd.Timestamp(day+'T15:00:00+08:00'):continue
        valid.append(bundle)
    return max(valid,key=lambda b:(b['available_at'],b['name'])) if valid else None


def _identity(manifest,bundle,root,day):
    if not manifest:return {'model_run_id':Path(root).name if root else None,'model_head':None,
        'verified_frozen_contract':False,'score_date':day,'publication_allowed':False}
    contract=manifest['target_contract']
    return {'family':STRATEGY_ID,'model_run_id':Path(root).name,'model_head':manifest['head'],'head':manifest['head'],
        'target':contract['target'],'checkpoint':bundle['name'],'model_id':bundle['name'],
        'manifest_sha256':manifest['manifest_sha256'],'target_contract_hash':manifest['target_contract_hash'],
        'label_contract_hash':contract.get('contract_hash'),'feature_schema_hash':manifest['feature_schema_hash'],
        'pipeline_contract_hash':contract.get('pipeline_contract_hash'),
        'source_snapshot_hash':contract['source_snapshot_hash'],'train_label_end':manifest['train_label_end'],
        'calibration_label_end':manifest['calibration_label_end'],
        'policy_label_end':manifest.get('policy_label_end',manifest['fold']['policy_end']),
        'simulated_available_at':bundle['available_at'],'available_at':bundle['available_at'],
        'actual_training_completed_at':manifest.get('actual_training_completed_at',manifest.get('training_completed_at')),
        'verified_frozen_contract':True,'score_date':day,'publication_allowed':False,
        'continuation_policy_hash':contract.get('continuation_policy_hash'),'continuation_stage':contract.get('stage')}


def forward_trend(frame,root,trend_bundles,config,session=None,mode='historical',market_dates=None):
    session=session or TrendSession();root=Path(root)
    calendar=market_dates or frame.attrs.get('verified_market_dates') or json.loads((root/'data_snapshots'/'calendar.json').read_text())['dates']
    data=bind_trend_inputs(frame,calendar)
    data['date']=pd.to_datetime(data.date).dt.strftime('%Y-%m-%d');routes={};groups={}
    for i,row in enumerate(data.to_dict('records')):
        day=row['date'];bundle=_bundle(day,trend_bundles,'trend_direction',root,mode)
        routes[day]={'date':day,'available_at':day+'T15:00:00+08:00','input_eligible':bool(row.get('model_input_eligible')),
            'model_available':False,'p_up':None,'p_down':None,'p_neutral':None,
            'reason_codes':['no_prior_trend_head'] if bundle is None else ['trend_input_ineligible'],
            'model_identity':_identity(None,None,root,day)}
        if bundle:groups.setdefault(bundle['name'],{'bundle':bundle,'indices':[]})['indices'].append(i)
    for group in groups.values():
        bundle=group['bundle'];path=root/bundle['heads']['trend_direction']['path'];m=load_trend_head(path)
        if m['target_contract']['parameters']!=trend_config(config):raise ValueError('trend target parameters changed without retraining')
        if config.get('source_snapshot_hash') and m['target_contract']['source_snapshot_hash']!=config['source_snapshot_hash']:raise ValueError('trend source identity mismatch')
        values=data.iloc[group['indices']].copy();values.attrs['schema_hash']=SCHEMA_HASH
        probabilities,m=session.predict(values,path,expected_contract=m['target_contract'],mode=mode)
        for row,p in zip(values.to_dict('records'),probabilities):
            if np.isfinite(p).all():
                day=row['date'];routes[day].update(p_up=float(p[0]),p_down=float(p[1]),p_neutral=float(p[2]),
                    model_available=True,reason_codes=[],model_identity=_identity(m,bundle,root,day))
    return routes


def _predict_interval(values,root,bundle,head,session,mode='historical',validation=False,config=None):
    path=Path(root)/bundle['heads'][head]['path'];manifest=load_head(path)
    if manifest['family']!=STRATEGY_ID:raise ValueError('old head cannot serve new trend strategy')
    if config:
        contract=manifest['target_contract'];stage='E0' if head=='ma_exit' else 'E1'
        for name in ('source_snapshot_hash','pipeline_contract_hash','candidate_policy_hash'):
            if contract.get(name)!=config.get(name):raise ValueError('trend interval '+name+' mismatch')
        if contract.get('stage')!=stage or contract.get('continuation_policy_hash')!=config.get(stage+'_policy_hash'):
            raise ValueError('trend interval continuation/target policy changed without retraining')
    p=session.predict_one(values,path,expected_head=head,expected_contract=manifest['target_contract'],
        validation=validation,usage_mode='production' if mode=='production' else 'historical_reconstruction')
    return float(p) if np.isfinite(p) else None,manifest


def predict_entry(feature,state,candidate,plan,event_prefix,root,model_bundles,config,session=None,mode='historical'):
    session=session or HeadSession();day=str(state['date'])[:10]
    if int(state.get('shares',0))!=0:return None,_identity(None,None,root,day),'actual_existing_position'
    if not plan or plan.get('entry_allowed') is not True:return None,_identity(None,None,root,day),'candidate_risk_ineligible'
    bundle=_bundle(day,model_bundles,'entry',root,mode)
    if bundle is None:return None,_identity(None,None,root,day),'no_prior_high_return_head'
    values=entry_feature_row(feature,state,candidate,plan,initial_cash=config['initial_cash'],event_prefix=event_prefix)
    values['input_eligible']=bool(any(values.get(k+'_input_available',0) for k in candidate.get('sources',[])))
    p,m=_predict_interval(values,root,bundle,'entry',session,mode,config=config)
    if m['target_contract'].get('stage')!='E1':raise ValueError('E2 or bootstrap head cannot serve E1 account')
    return p,_identity(m,bundle,root,day),'scored' if p is not None else 'input_ineligible'


def pinned_exit_policy(frame,calendar,prepared,root,ma_exit_bundles,config,session=None,mode='historical'):
    frame=bind_trend_inputs(frame,calendar)
    session=session or HeadSession();root=Path(root);rows={str(r['date'])[:10]:r for r in frame.to_dict('records')}
    midx={d:i for i,d in enumerate(calendar)};bindings={}
    for row in prepared['rows']:
        plan=row.get('entry_plan') or row.get('raw_risk_plan')
        if not plan or not plan.get('plan_id'):continue
        signal=str(plan.get('signal_date',row['date']))[:10];bundle=_bundle(signal,ma_exit_bundles,'ma_exit',root,mode)
        if bundle:
            m=load_head(root/bundle['heads']['ma_exit']['path'])
            if m['train_label_end']>=signal or m['calibration_label_end']>=signal or m['target_contract']['stage']!='E0':raise ValueError('MA exit binding is not strictly prior E0 target')
        bindings[plan['plan_id']]={'plan_id':plan['plan_id'],'candidate_signal_date':signal,
            'checkpoint':bundle['name'] if bundle else None,'available_at':bundle['available_at'] if bundle else None,
            'bundle':bundle,'mode':'signal_time_pinned_holding_exit' if bundle else 'explicit_rule_E0_no_prior_exit_head',
            'checkpoint_upgrade':False,'source':'union'}
    recipe={'family':STRATEGY_ID,'stage':'E1','baseline':'E0','exit_catalog_hash':stable_hash(ma_exit_bundles),
        'candidate_policy_hash':prepared['config_hash'],'holding_checkpoint_upgrade':False,
        'fallback':'E0_for_entire_position','l2_exit_apply':False};recipe_hash=config.get('E1_policy_hash') or stable_hash(recipe)
    def fresh():
        prefixes={}
        def callback(snapshot):
            binding=bindings.get(snapshot.get('plan_id'))
            if binding is None:raise ValueError('actual holding has no original signal/plan binding')
            signal=binding['candidate_signal_date'];bundle=binding['bundle'];day=snapshot['date']
            if not signal<snapshot['entry_date']<=day:raise ValueError('holding clock violates original binding')
            saved=snapshot.get('exit_model_binding') or snapshot.get('entry_exit_binding') or snapshot.get('candidate_exit_binding')
            if saved and saved.get('checkpoint')!=binding['checkpoint']:raise ValueError('clone changed original exit checkpoint')
            prefix=prefixes.setdefault((snapshot['plan_id'],snapshot['entry_date']),copy.deepcopy(snapshot.get('holding_prefix') or {}))
            if prefix.get('through_date','')>=day:raise ValueError('reused actual holding prefix')
            prefix['through_date']=day
            values=holding_feature_row(rows[day],snapshot,entry_values={},prefix=prefix,market_index=midx)
            values['input_eligible']=bool(rows[day].get('model_input_eligible'))
            p,m=_predict_interval(values,root,bundle,'ma_exit',session,mode,config=config) if bundle else (None,None)
            threshold=float((bundle or {}).get('policy',{}).get('exit_threshold',config['account']['holding_exit_threshold']))
            identity=_identity(m,bundle,root,day)
            result={'date':day,'available_at':day+'T15:00:00+08:00','usage_mode':mode,'head':'holding_exit','model_identity':identity,'contract_hash':m['target_contract_hash'] if m else recipe_hash,
                'probability':p,'calibrated_threshold':threshold,'request_exit':p is not None and p>=threshold,
                'apply':p is not None,'research_arm':True,'holding_contract_compatible':bool(m),
                'status':binding['mode'],'entry_exit_binding':{k:v for k,v in binding.items() if k!='bundle'},
                'candidate_exit_binding':{k:v for k,v in binding.items() if k!='bundle'},
                'prefix_through_date':day,'holding_prefix_source':'actual_Broker_observed_prefix'}
            return result
        callback.new_replay=fresh;callback.recipe=recipe;callback.recipe_hash=recipe_hash;callback.bindings=bindings
        return callback
    return fresh()


def policy_trend(frame,root,bundle,config,session=None,market_dates=None):
    """Provisional train/cal-only head, confined to its explicit policy segment."""
    session=session or TrendSession()
    calendar=market_dates or frame.attrs.get('verified_market_dates') or json.loads((Path(root)/'data_snapshots'/'calendar.json').read_text())['dates']
    data=bind_trend_inputs(frame,calendar);data['date']=pd.to_datetime(data.date).dt.strftime('%Y-%m-%d')
    path=Path(root)/bundle['heads']['trend_direction']['path'];manifest=load_trend_head(path);fold=manifest['fold']
    scoped=data[(data.date>=fold['policy_start'])&(data.date<=fold['policy_end'])].copy();scoped.attrs['schema_hash']=SCHEMA_HASH
    if scoped.empty:return {}
    probabilities,m=session.predict(scoped,path,expected_contract=m_contract(manifest),policy=True)
    routes={}
    for row,p in zip(scoped.to_dict('records'),probabilities):
        day=row['date'];identity=_identity(m,bundle,root,day)
        identity.update(provisional_checkpoint=True,policy_start=fold['policy_start'],policy_end=fold['policy_end'],
            simulated_available_at=fold['calibration_end'],available_at=fold['calibration_end'])
        routes[day]={'date':day,'available_at':day+'T15:00:00+08:00','usage_mode':'policy_validation',
            'input_eligible':bool(row['model_input_eligible']),'model_available':bool(np.isfinite(p).all()),
            'p_up':float(p[0]) if np.isfinite(p).all() else None,'p_down':float(p[1]) if np.isfinite(p).all() else None,
            'p_neutral':float(p[2]) if np.isfinite(p).all() else None,'model_identity':identity,'reason_codes':[]}
    return routes


def m_contract(manifest):return manifest['target_contract']


def _policy_worker(payload):
    import itertools
    from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
    from my_strategy.services.czsc_trend_account import replay_trend_account,prepare_account_context
    record,calendar,cfg,root,bundle,variants=payload
    frame,pack=read_snapshot(record);fold=bundle['fold'];frame=frame[frame.date<=fold['policy_end']].copy()
    if not frame.date.ge(fold['policy_start']).any():return []
    routes=policy_trend(frame,root,bundle,cfg,market_dates=calendar);context=prepare_account_context(frame,calendar);done=[]
    # Every variant starts flat with the same cash. Its quote seasoning uses
    # only known prior observations, and all remaining inventory is disclosed.
    prior=frame[frame.date<fold['policy_start']]
    seed={'version':STRATEGY_ID,'symbol':record['symbol'],'initial_cash':cfg['initial_cash'],'cash':cfg['initial_cash'],
        'shares':0,'positions':{},'avg_cost':{},'ledger':[],'actual_position':None,'pending_intent':None,'last_buy_date':None,
        'holding_prefix':{},'last_close_date':prior.date.iloc[-1],'last_mark':float(prior.close.iloc[-1]),
        'seasoned_bars':int((prior.has_trade_price.eq(1)&prior.source.isin(cfg['account']['valid_sources'])).sum()),
        'consumed_plan_ids':[],'series_context_hash':'policy_validation_initial_flat',
        'source_context_hash':cfg['source_snapshot_hash'],'current_snapshot':{}}
    for variant in variants:
        candidate_cfg=copy.deepcopy(cfg);candidate_cfg['trend']['candidate_cross_threshold']=variant['cross']
        candidate_cfg['consensus'].update(buy_base_threshold=variant['buy_base'],sell_threshold=variant['sell'],margin=variant['margin'])
        prepared=prepare_trend_decisions(frame,calendar,evidence_pack=pack,trend_probabilities=routes,
            config=candidate_cfg,mode='E0',allow_policy_validation=True)
        account=replay_trend_account(frame,calendar,prepared,config=candidate_cfg,start=fold['policy_start'],end=fold['policy_end'],
            initial_state=seed,run_id='policy:'+record['symbol'],mode='E0',account_context=context)
        done.append({'variant':variant,'symbol':record['symbol'],'metrics':account['metrics'],'ledger':account['ledger'],
            'rejections':account['rejections'],'open_position':account['actual_position'],
            'ledger_hash':stable_hash(account['ledger']),'fold':fold['name']})
    return done


def select_source_policy(records,calendar,cfg,root,trend_bundles,context,cpu_workers):
    """Only earliest independent policy segment; locked development never selects."""
    import itertools
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    bundle=min(trend_bundles,key=lambda b:b['available_at']);grid=cfg['consensus']['policy_grid']
    variants=[{'cross':c,'buy_base':b,'sell':s,'margin':m} for c,b,s,m in itertools.product(
        cfg['trend']['candidate_cross_grid'],grid['buy_base_threshold'],grid['sell_threshold'],grid['margin'])]
    payloads=[(r,calendar,cfg,str(root),bundle,variants) for r in records]
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            chunks=list(pool.map(_policy_worker,payloads))
    else:chunks=[_policy_worker(p) for p in payloads]
    details=[x for chunk in chunks for x in chunk];aggregates=[]
    for variant in variants:
        group=[x for x in details if x['variant']==variant]
        count=sum(completed_round_trips(x['ledger']) for x in group)
        score=float(np.mean([x['metrics']['total_return'] for x in group])) if group else 0.
        aggregates.append({'variant':variant,'round_trips':count,'mean_fee_net_return':score,
            'mean_max_drawdown':float(np.mean([x['metrics']['max_drawdown'] for x in group])) if group else 0.,
            'scored_population':len(group),'eligible_for_selection':count>=cfg['min_validation_round_trips']})
    valid=[x for x in aggregates if x['eligible_for_selection']]
    choice=max(valid,key=lambda x:(x['mean_fee_net_return'],x['mean_max_drawdown'],-x['variant']['cross'])) if valid else None
    if choice:
        v=choice['variant'];cfg['trend']['candidate_cross_threshold']=v['cross']
        cfg['consensus'].update(buy_base_threshold=v['buy_base'],sell_threshold=v['sell'],margin=v['margin'])
    report={'phase':'candidate_and_ordinary_SELL_policy_only','fold':bundle['fold'],'variants':aggregates,
        'selected':choice,'status':'policy_selected' if choice else 'insufficient_policy_trades_use_preregistered_baseline',
        'initial_cash':cfg['initial_cash'],'four_development_windows_used':False,'high_return_filter_used':False}
    _save(context.subdir('validation')/'source_policy_accounts.json',details)
    _save(context.subdir('validation')/'source_policy_selection.json',report)
    return report


def _dataset_worker(payload):
    import torch
    torch.set_num_threads(1)
    from my_strategy.services.czsc_trend_dataset import build_account_dataset
    record,calendar,cfg,root,trend_bundles,ma_bundles,stage=payload
    frame,pack=read_snapshot(record)
    # Early bars remain warm-up context; no learned forecast exists before a
    # legitimately prior checkpoint. No source price is substituted.
    frame=frame[frame.date<=cfg['training_end']].copy();frame.attrs.update(source_snapshot_hash=cfg['source_snapshot_hash'])
    frame=bind_trend_inputs(frame,calendar)
    routes=forward_trend(frame,root,trend_bundles,cfg)
    callback=None
    if stage=='E1':
        from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
        prepared=prepare_trend_decisions(frame,calendar,evidence_pack=pack,trend_probabilities=routes,config=cfg,mode='E0')
        callback=pinned_exit_policy(frame,calendar,prepared,root,ma_bundles,cfg)
    directory=Path(root)/'dataset'/stage/record['symbol'].replace('.','_')
    result,summary=build_account_dataset(frame,calendar,pack,routes,cfg,stage=stage,exit_policy=callback,
        directory=directory,run_id=Path(root).name+':'+record['symbol'])
    print('DATASET '+stage+' '+record['symbol']+' entries='+str(summary['entry_rows'])+' exits='+str(summary['exit_rows']),flush=True)
    return summary


def build_stage(records,calendar,cfg,context,stage,trend_bundles,ma_bundles,cpu_workers):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    payloads=[(r,calendar,cfg,str(context.run_dir()),trend_bundles,ma_bundles,stage) for r in records];summaries=[]
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(_dataset_worker,p) for p in payloads]
            for future in as_completed(futures):summaries.append(future.result())
    else:summaries=[_dataset_worker(p) for p in payloads]
    pooled={}
    for key in ('entries','exits','episodes'):
        pooled[key]=pd.concat([pd.read_parquet(Path(context.run_dir())/'dataset'/stage/s['symbol'].replace('.','_')/(key+'.parquet'))
            for s in summaries],ignore_index=True)
        pooled[key].to_parquet(context.subdir('dataset',stage)/(key+'.parquet'),index=False)
    _save(context.subdir('dataset',stage)/'manifest.json',{'stage':stage,'records':summaries,'failures':[]})
    return pooled,summaries


def completed_round_trips(ledger):
    shares=0;count=0
    for row in ledger:
        shares+=int(row['shares']) if row['action']=='BUY' else -int(row['shares'])
        if row['action']!='BUY' and shares==0:count+=1
        if shares<0:raise ValueError('ledger inventory below zero')
    return count


def select_holding_exit_policy(dataset,bundle,root,config):
    """Policy-only paired net-proceeds utility; this is not portfolio acceptance."""
    from my_strategy.services.czsc_trend_interval_model import time_masks
    manifest=load_head(Path(root)/bundle['heads']['ma_exit']['path']);_,_,mask=time_masks(dataset,bundle['fold'])
    scoped=bind_inputs(dataset.loc[mask].copy(),L1_MA_EXIT_COLUMNS,'ma_exit')
    groups=scoped.get('economic_group_id',pd.Series(dtype=str));population=int(groups.nunique())
    result={'checkpoint':bundle['name'],'policy_start':bundle['fold']['policy_start'],'policy_end':bundle['fold']['policy_end'],
        'mature_rows':len(scoped),'independent_episodes':population,'criterion':'episode_weighted_remaining_net_proceeds_advantage',
        'thresholds':[],'selected_threshold':config['account']['holding_exit_threshold'],'development_windows_used':False,
        'status':'insufficient_policy_episodes_use_preregistered_threshold'}
    if scoped.empty:return result
    probabilities=HeadSession().predict(scoped,Path(root)/bundle['heads']['ma_exit']['path'],expected_head='ma_exit',
        expected_contract=manifest['target_contract'],validation=True)
    weights=1/groups.map(groups.value_counts()).to_numpy(float)
    advantage=scoped.exit_advantage.to_numpy(float)
    for threshold in config['exit_threshold_grid']:
        utility=float(np.sum(weights*np.where(probabilities>=threshold,advantage,0.))/weights.sum())
        result['thresholds'].append({'threshold':threshold,'mean_net_proceeds_advantage':utility,
            'requests':int((probabilities>=threshold).sum())})
    if population>=config['min_validation_round_trips']:
        best=max(result['thresholds'],key=lambda r:(r['mean_net_proceeds_advantage'],r['threshold']))
        bundle['policy']['exit_threshold']=best['threshold'];result.update(selected_threshold=best['threshold'],status='policy_selected')
    return result


def _fit_interval(dataset,head,fold,calendar,context,cfg,architecture='linear',device='cpu'):
    from my_strategy.services.czsc_trend_interval_model import split_validation
    if 'calibration_start' not in fold:fold=split_validation({**fold,'validation_start':fold.get('validation_start',str((pd.Timestamp(fold['validation_end']).to_period('Q')).start_time.date()))},calendar)
    columns=L1_MA_EXIT_COLUMNS if head=='ma_exit' else ENTRY_FEATURE_COLUMNS if head=='entry' else EXIT_FEATURE_COLUMNS
    path=context.subdir('models',architecture,fold['name'],head)
    manifest=train_head(dataset,columns,path,fold,cfg,head=head,architecture=architecture,regression=head=='exit_advantage',
        device=device,contract=dataset.attrs['label_contract'])
    relative=str(path.relative_to(context.run_dir()))
    return {'name':fold['name'],'available_at':fold['policy_end'],'fold':fold,'architecture':architecture,
        'heads':{head:{'path':relative,'manifest_sha256':manifest['manifest_sha256'],'target_contract_hash':manifest['target_contract_hash']}},
        'policy':{'entry_threshold':cfg['consensus']['high_return_threshold'],'exit_threshold':cfg['account']['holding_exit_threshold']}},manifest


def _train_one(*,source_native_run_id='20261012-120726-523936-20261008-czsc-high-return-native-v4-train-NO_GIT',
                          symbols=None,end='2026-10-08',config=None,cpu_workers=6,device='cpu',native_records=None):
    """A new immutable family; no old model checkpoint is imported."""
    import torch
    torch.set_num_threads(min(4,cpu_workers))
    from my_strategy.web_dashboard.high_return import _calendar
    from my_strategy.services.czsc_trend_dataset import training_frame
    cfg=strategy_config(config);cfg['cpu_workers']=cpu_workers;cfg['training_end']=end
    old=json.loads((ARTIFACT_RUNS_ROOT/source_native_run_id/'reports'/'native_v4.json').read_text())
    records=[r for r in (native_records or old['dataset_records']) if symbols is None or r['symbol'] in symbols]
    if not records or (symbols and {r['symbol'] for r in records}!=set(symbols)):raise ValueError('requested source cohort unavailable')
    source={'source_native_run_id':source_native_run_id,'effective_data_end':end,'selected_symbols':[r['symbol'] for r in records],
        'records_hash':stable_hash(records),'selected_count':len(records),'full_market_universe':False,
        'price_basis':'unadjusted','scope':'coverage_frozen_mainboard_cohort_not_return_selected','source_model_heads_imported':False}
    source['snapshot_hash']=stable_hash(source);cfg['source_snapshot_hash']=source['snapshot_hash']
    calendar,calendar_id=_calendar();calendar=[d for d in calendar if d<=end]
    cfg['pipeline_contract_hash']=stable_hash({'family':STRATEGY_ID,'source':source,'calendar_hash':stable_hash(calendar),'registered_config':cfg,
        'order':['independent_trend','policy_only_sources','E0_no_HR','MA_exit_E0','pinned_E1','L2_entry_E1','L2_exit_shadow'],
        'no_E2_entry_reuse':True})
    context=create_run_context(task=TRAIN_KIND,as_of_date=end,config=cfg,data_version=source['snapshot_hash'],
        seed=cfg['seed'],stocks=source['selected_symbols'],scope='train',start_date=cfg['feature_start'],end_date=end)
    print('TRAIN_RUN_ID '+context.run_id,flush=True)
    names=['czsc_trend_contracts.py','czsc_trend_model.py','czsc_trend_features.py','czsc_trend_labels.py',
        'czsc_trend_interval_model.py','czsc_trend_dataset.py','czsc_trend_pipeline.py','czsc_trend_consensus.py','czsc_trend_account.py']
    code={str(MY_STRATEGY_ROOT/'services'/n):_file_hash(MY_STRATEGY_ROOT/'services'/n) for n in names}
    _save(context.subdir('config')/'initial_config.json',cfg);_save(context.subdir('config')/'generating_code_hashes.json',code)
    for n in names:(context.subdir('config','source_code')/n).write_bytes((MY_STRATEGY_ROOT/'services'/n).read_bytes())
    _save(context.subdir('data_snapshots')/'source.json',source);_save(context.subdir('data_snapshots')/'records.json',records)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'source_run_id':calendar_id,'dates':calendar})
    failures=[];unavailable=[];manifests=[];trendframes=[]
    print('STAGE independent_three_class_trend',flush=True)
    for r in records:
        frame,_=read_snapshot(r);frame=frame[frame.date<=end]
        observations=build_trend_labels(frame,calendar,cfg,source_snapshot_hash=source['snapshot_hash'],pipeline_contract_hash=cfg['pipeline_contract_hash'])
        trendframes.append(observations[observations.date>=cfg['feature_start']].copy())
    contract=trendframes[0].attrs['label_contract'];trenddata=pd.concat(trendframes,ignore_index=True);trenddata.attrs['label_contract']=contract
    trenddata.to_parquet(context.subdir('dataset','trend')/'observations.parquet',index=False)
    folds=chronological_folds(end,calendar);trends=[]
    holding_policy=[]
    for fold in folds:
        try:
            path=context.subdir('models','trend',fold['name']);m=train_trend_head(trenddata,path,fold,cfg,contract=contract)
            bundle={'name':fold['name'],'fold':fold,'available_at':fold['policy_end'],
                'heads':{'trend_direction':{'path':str(path.relative_to(context.run_dir())),'manifest_sha256':m['manifest_sha256']}}}
            trends.append(bundle);manifests.append(m);print('TREND_HEAD '+fold['name']+' '+str(m['sample_counts']),flush=True)
        except ValueError as exc:
            if 'insufficient mature' not in str(exc):raise
            unavailable.append({'head':'trend_direction','fold':fold['name'],'reason':str(exc)})
    if not trends:raise ValueError('no qualified three-class head; cohort cannot support pipeline')
    _save(context.subdir('models')/'trend_bundles.json',trends)
    print('STAGE independent_policy_selection',flush=True)
    selection=select_source_policy(records,calendar,cfg,context.run_dir(),trends,context,cpu_workers)
    cfg['candidate_policy_hash']=stable_hash({'candidate':cfg['candidate_policy'],'risk_policy':cfg['risk_policy'],
        'trend_candidate':{k:cfg['trend'][k] for k in ('candidate_cross_threshold','candidate_margin')},
        'consensus':cfg['consensus'],'signal_points':cfg['signal_points'],'native_records':source['records_hash']})
    cfg['E0_policy_hash']=stable_hash({'stage':'E0','candidate_policy_hash':cfg['candidate_policy_hash'],'account':cfg['account'],
        'trend_catalog_hash':stable_hash(trends),'ordinary_sell':'new_consensus_direction','entry_HR':False})
    print('STAGE E0_actual_rule_bootstrap_no_HR',flush=True)
    baseline,baseline_records=build_stage(records,calendar,cfg,context,'E0',trends,[],cpu_workers)
    ma_data=training_frame(baseline,'ma_exit',cfg,calendar,'E0');ma_bundles=[]
    print('STAGE train_independent_actual_holding_MA_exit',flush=True)
    for fold in folds:
        try:
            b,m=_fit_interval(ma_data,'ma_exit',fold,calendar,context,cfg,device=device)
            holding_policy.append(select_holding_exit_policy(ma_data,b,context.run_dir(),cfg));ma_bundles.append(b);manifests.append(m)
            print('MA_EXIT_HEAD '+fold['name']+' '+str(m['sample_counts']),flush=True)
        except ValueError as exc:
            if not ('insufficient mature' in str(exc) or 'require both classes' in str(exc)):raise
            unavailable.append({'head':'ma_exit','fold':fold['name'],'reason':str(exc)})
    _save(context.subdir('models')/'ma_exit_bundles.json',ma_bundles)
    _save(context.subdir('validation')/'holding_exit_policy_selection.json',holding_policy)
    cfg['E1_policy_hash']=stable_hash({'stage':'E1','E0':cfg['E0_policy_hash'],'signal_time_pinned_MA_exit':stable_hash(ma_bundles),
        'holding_threshold':cfg['account']['holding_exit_threshold'],'no_head':'entire_position_E0','checkpoint_upgrade':False,'L2_exit_apply':False})
    print('STAGE E1_pinned_exit_rebuild_entry_and_shadow_exit_labels',flush=True)
    final,final_records=build_stage(records,calendar,cfg,context,'E1',trends,ma_bundles,cpu_workers)
    entrydata=training_frame(final,'entry',cfg,calendar,'E1');exits=training_frame(final,'exit',cfg,calendar,'E1')
    advantage=training_frame(final,'exit_advantage',cfg,calendar,'E1');models=[];architecture_bundles=[]
    print('STAGE L2_actual_interval_entry_and_shadow_only_exit',flush=True)
    for fold in [*cfg['development_windows'],cfg['production']]:
        for architecture in cfg['architectures']:
            bundle=None
            for head,data in (('entry',entrydata),('exit',exits),('exit_advantage',advantage)):
                try:
                    b,m=_fit_interval(data,head,fold,calendar,context,cfg,architecture,device);manifests.append(m)
                    if bundle is None:bundle=b
                    else:bundle['heads'].update(b['heads'])
                    print('L2_HEAD '+fold['name']+' '+architecture+' '+head+' '+str(m['sample_counts']),flush=True)
                except ValueError as exc:
                    if not ('insufficient mature' in str(exc) or 'require both classes' in str(exc)):raise
                    unavailable.append({'head':head,'fold':fold['name'],'architecture':architecture,'reason':str(exc)})
            if bundle:
                architecture_bundles.append(bundle)
                if architecture=='linear' and bundle['heads'].get('entry'):models.append(bundle)
    _save(context.subdir('models')/'l2_bundles.json',models)
    if any(_file_hash(p)!=sha for p,sha in code.items()):raise ValueError('generating code changed during frozen training')
    freeze={'family':STRATEGY_ID,'actual_freeze_at':local_now().isoformat(),'config_hash':stable_hash(cfg),
        'pipeline_contract_hash':cfg['pipeline_contract_hash'],'source_snapshot_hash':source['snapshot_hash'],'code_hashes':code,
        'models_hash':stable_hash([m['manifest_sha256'] for m in manifests]),
        'models_catalog_hash':stable_hash({'trend':trends,'ma_exit':ma_bundles,'architecture':architecture_bundles}),
        'independent_window_status':'pending',
        'historical_available_is_simulated':True,'actual_completion_not_backdated':True}
    freeze['freeze_hash']=stable_hash(freeze)
    pipeline={'stages':['independent_trend','source_policy','E0','MA_exit','E1','L2_entry_E1','L2_exit_shadow'],
        'trend_heads':len(trends),'ma_exit_heads':len(ma_bundles),'entry_heads':len(models),
        'holding_checkpoint_upgrade':False,'bootstrap_uses_HR':False,'applied_L2_exit':False,
        'final_entry_rows':len(final['entries']),'final_mature_entries':int(final['entries'].label_available.fillna(False).sum()),
        'final_positive_entries':int(final['entries'].label.eq(1).sum()),'final_exit_rows':len(final['exits']),
        'final_mature_exits':int(final['exits'].label_available.fillna(False).sum())}
    report={'run_id':context.run_id,'status':'completed','strategy_version':STRATEGY_ID,'config':cfg,'source_snapshot':source,
        'data_end':end,'dataset_records':records,'trend_bundles':trends,'ma_exit_bundles':ma_bundles,'model_bundles':models,
        'architecture_bundles':architecture_bundles,'source_policy_selection':selection,'holding_exit_policy_selection':holding_policy,'continuation_stage':'E1','pipeline':pipeline,
        'unavailable_heads':unavailable,'failures':failures,'protocol_freeze':freeze,'completed_at':local_now().isoformat(),
        'required_recent_heads_ready':any(b['name']==cfg['production']['name'] and b['heads'].get('entry') for b in models) and any(b['name']==max(trends,key=lambda b:b['available_at'])['name'] for b in ma_bundles),
        'automatic_selection_eligible':len(records)>=60 and cfg['trend']['horizon_sessions']==10 and cfg['risk_policy']=='confirmed_source_boundary_prior_atr' and any(b['name']==cfg['production']['name'] and b['heads'].get('entry') for b in models) and any(b['name']==max(trends,key=lambda b:b['available_at'])['name'] for b in ma_bundles),
        'architecture_selection':'preregistered_linear_main_tree_mlp_controlled_ablations',
        'validation_records':old['dataset_records'],
        'model_gate':{'passed':False,'reasons':['future_independent_validation_pending','partial_frozen_cohort_not_full_market',
            'development_fee_net_Broker_validation_pending','unadjusted_price_and_historical_ST_identity_limits']},'publication_allowed':False}
    _save(context.subdir('config')/'research.json',cfg);_save(context.subdir('validation')/'protocol_freeze.json',freeze)
    _save(context.subdir('reports')/'trend_consensus.json',report);context.write_metadata({'status':'completed','pipeline':pipeline,'failures':[]})
    print('COMPLETED '+context.run_id,flush=True);return report


def _expanded_native_worker(payload):
    from my_strategy.services.czsc_high_return_research import _read_features
    from my_strategy.services.czsc_native_evidence_v4 import build_native_evidence_v4
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    record,oldconfig,end,calendar,directory=payload
    frame=_read_features(record,oldconfig,end);pack=build_native_evidence_v4(frame,oldconfig,market_dates=calendar)
    target=Path(directory)/record['symbol'].replace('.','_');target.mkdir(parents=True,exist_ok=True)
    numeric=pack['frame'].drop(columns=[c for c in pack['frame'] if pack['frame'][c].map(lambda v:isinstance(v,(dict,list,tuple))).any()])
    numeric.attrs={};numeric.to_parquet(target/'features.parquet',index=False)
    _save(target/'native.json',{k:v for k,v in pack.items() if k!='frame'})
    return {**record,'original_path':record['path'],'original_sha256':record['sha256'],
        'path':str(target/'features.parquet'),'sha256':_file_hash(target/'features.parquet'),
        'native_path':str(target/'native.json'),'native_sha256':_file_hash(target/'native.json'),
        'native_config_hash':pack['config_hash'],'native_schema_hash':pack['schema_hash'],'rows':len(numeric)}


def expand_training_snapshot(base_report,size,end,cpu_workers):
    """Preregistered sample-capacity tiers, never selection by realized return."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from my_strategy.services.czsc_high_return_research import _source
    oldcfg=base_report['config'];original,calendar,snapshot,_=_source(oldcfg['source_training_run_id'],end)
    existing=base_report['dataset_records'];known={r['symbol'] for r in existing}
    ranked=sorted([r for r in original if r['symbol'] not in known and r['symbol'].startswith(('00','60'))],
        key=lambda r:stable_hash({'seed':42,'symbol':r['symbol']}))
    selected=[];rejected=[]
    for r in ranked:
        path=Path(r['path'])
        if _file_hash(path)!=r['sha256']:raise ValueError('expansion source revision drift')
        dates=pd.read_parquet(path,columns=['date','source','has_trade_price']);dates['date']=pd.to_datetime(dates.date).dt.strftime('%Y-%m-%d')
        prior=dates[dates.date<'2023-01-01'];qualified=prior.source.isin(oldcfg['strategy'].get('valid_sources',strategy_config()['account']['valid_sources']))&prior.has_trade_price.eq(1)
        if qualified.sum()<65:rejected.append({'symbol':r['symbol'],'reason':'insufficient_pre_feature_start_verified_history'});continue
        selected.append(r)
        if len(selected)+len(existing)>=size:break
    if len(selected)+len(existing)<size:raise ValueError('frozen source cannot fill preregistered training tier')
    context=create_run_context(task='czsc-trend-native-capacity-snapshot',as_of_date=end,config={'capacity_tier':size,'seed':42,
        'selection':'symbol_hash_plus_pre2023_input_coverage_only','development_validation_pool':60},
        data_version=snapshot['snapshot_hash'],scope='train',stocks=[r['symbol'] for r in selected])
    print('CAPACITY_SNAPSHOT '+context.run_id+' tier='+str(size),flush=True)
    payloads=[(r,oldcfg,end,calendar,str(context.subdir('data_snapshots','native'))) for r in selected]
    if cpu_workers>1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:extra=list(pool.map(_expanded_native_worker,payloads))
    else:extra=[_expanded_native_worker(p) for p in payloads]
    if any(r['native_config_hash']!=existing[0]['native_config_hash'] or r['native_schema_hash']!=existing[0]['native_schema_hash'] for r in extra):
        raise ValueError('expanded native input contract differs from frozen control cohort')
    records=existing+extra
    _save(context.subdir('data_snapshots')/'records.json',records)
    _save(context.subdir('reports')/'capacity_snapshot.json',{'size':size,'records':records,'rejected':rejected,
        'source_snapshot':snapshot,'selection_uses_return':False,'validation_cohort_changed':False,'publication_allowed':False})
    return records


def train_trend_consensus(*,source_native_run_id='20261012-120726-523936-20261008-czsc-high-return-native-v4-train-NO_GIT',
                          symbols=None,end='2026-10-08',config=None,cpu_workers=6,device='cpu'):
    cfg=strategy_config(config);cfg['training_capacity_tiers']=[60,180,360]
    cfg['capacity_expansion_criterion']='missing_recent_qualified_three_class_MA_exit_or_production_L2_entry_head_only'
    reports=[];native_records=None
    for tier in ([len(symbols)] if symbols else cfg['training_capacity_tiers']):
        if native_records is None and not symbols and tier>60:
            base=json.loads((ARTIFACT_RUNS_ROOT/source_native_run_id/'reports'/'native_v4.json').read_text())
            native_records=expand_training_snapshot(base,tier,end,cpu_workers)
        report=_train_one(source_native_run_id=source_native_run_id,symbols=symbols,end=end,config=cfg,
            cpu_workers=cpu_workers,device=device,native_records=native_records)
        reports.append({'run_id':report['run_id'],'tier':tier,'required_recent_heads_ready':report['required_recent_heads_ready']})
        if symbols or report['required_recent_heads_ready']:return report
        print('CAPACITY_EXPANSION '+str(tier)+' missing_recent_heads '+str(report['unavailable_heads']),flush=True)
        native_records=None
    return report
