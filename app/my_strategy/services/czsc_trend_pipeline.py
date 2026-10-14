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
from my_strategy.services.czsc_trend_support import _save, _file_hash, _json, resolve_frozen_asset
from my_strategy.services.czsc_trend_contracts import (strategy_config, validate_training_config,
    STRATEGY_ID, TRAIN_KIND, SPLIT_POLICY, ECONOMIC_WEIGHTING, ENTRY_HEADS, L2_HOLDING_HEADS,
    HELD_TRAINING_POPULATION, HELD_CONDITION_DOMAIN, HELD_SPLIT_POLICY,
    HELD_ECONOMIC_WEIGHTING, validate_holding_population)
from my_strategy.services.czsc_trend_model import (TrendSession, SCHEMA_HASH, bind_trend_inputs,
    build_trend_labels, chronological_folds, load_trend_head, train_trend_head, trend_config)
from my_strategy.services.czsc_trend_features import (ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS,
    L1_MA_EXIT_COLUMNS, entry_feature_row, holding_feature_row)
from my_strategy.services.czsc_trend_interval_model import HeadSession, load_head, train_head, schema_hash, bind_inputs

GENERATING_FILES = tuple('services/'+name for name in ('czsc_trend_contracts.py','czsc_trend_model.py',
    'czsc_trend_features.py','czsc_trend_labels.py','czsc_trend_interval_model.py',
    'czsc_trend_dataset.py','czsc_trend_pipeline.py','czsc_trend_consensus.py',
    'czsc_trend_account.py','czsc_trend_runtime.py','czsc_trend_finalization.py',
    'czsc_trend_source.py','czsc_native_cache.py','czsc_native_evidence.py','czsc_trend_support.py',
    'czsc_trend_math.py','czsc_liquidity.py','czsc_backtest.py','czsc_account_artifacts.py','czsc_reference_quality.py'))+(
    'execution/broker.py','execution/cost_model.py','execution/ledger.py','execution/order.py',
    'configs/czsc_trend_strategy.json',)
POPULATION_VERSION = 'strict_prior_l2_actual_flat_prefix_pre_gate_attempts'
FULL_POPULATION_MODE = 'full_liquidity_union'
FULL_CAPACITY_CRITERION = 'all_frozen_historical_liquidity_qualified_symbols_no_readiness_early_stop'


def verify_full_population(source,config,records):
    """A processing budget cannot turn a complete frozen population into a slice."""
    liquidity=source.get('liquidity') or source.get('source_snapshot',{}).get('liquidity') or {}
    symbols=[record['symbol'] for record in records]
    original=[record['symbol'] for record in source.get('dataset_records',records)]
    if (source.get('population_mode')!=FULL_POPULATION_MODE
            or len(symbols)!=len(set(symbols)) or symbols!=original
            or set(symbols)!=set(liquidity.get('selected_symbols',[]))
            or not liquidity.get('daily_membership_hash') or not liquidity.get('population_hash')):
        raise ValueError('training must cover the complete frozen historical liquidity union')
    if (liquidity.get('policy')!=config['liquidity_universe']
            or liquidity.get('policy_hash')!=stable_hash(config['liquidity_universe'])):
        raise ValueError('frozen source liquidity policy differs from the training contract')
    return liquidity


def verify_source_generators(source):
    from my_strategy.services.czsc_trend_source import NATIVE_GENERATING_FILES
    hashes=source.get('code_hashes') or {}
    if set(hashes)!=set(NATIVE_GENERATING_FILES):
        raise ValueError('fresh raw-to-native generating source identity incomplete')
    for relative,expected in hashes.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('raw-to-native generator identity must be package-relative')
        path=MY_STRATEGY_ROOT/path if len(path.parts)>1 else MY_STRATEGY_ROOT/'services'/path
        if _file_hash(path)!=expected:
            raise ValueError('fresh raw-to-native generating source changed')


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
    validate_training_config(report['config'])
    freeze=report.get('protocol_freeze') or {}
    if freeze.get('freeze_hash')!=stable_hash({k:v for k,v in freeze.items() if k!='freeze_hash'}):raise ValueError('trend run freeze identity invalid')
    if stable_hash(report['config'])!=freeze.get('config_hash'):raise ValueError('trend frozen config drift')
    if report.get('strategy_version')!=STRATEGY_ID or report.get('config',{}).get('version')!=STRATEGY_ID:
        raise ValueError('legacy or foreign family cannot serve reassessment strategy')
    if not freeze.get('code_hashes') or set(freeze['code_hashes'])!=set(GENERATING_FILES):
        raise ValueError('trend generating-code freeze incomplete')
    catalogs={'trend':report['trend_bundles'],'ma_exit':report['ma_exit_bundles'],'architecture':report['architecture_bundles']}
    if stable_hash(catalogs)!=freeze.get('models_catalog_hash'):raise ValueError('trend model catalog/policy drift')
    served=[bundle for bundle in report['architecture_bundles']
        if bundle.get('architecture')=='linear' and bundle.get('heads',{}).get('entry')]
    if report.get('model_bundles')!=served:
        raise ValueError('served L2 entry catalog differs from its frozen architecture catalog')
    source=report['source_snapshot']
    verify_full_population(source,report['config'],report['dataset_records'])
    if report.get('validation_records')!=report['dataset_records']:
        raise ValueError('training and validation must own the same complete frozen population')
    if source.get('snapshot_hash')!=stable_hash({k:v for k,v in source.items() if k!='snapshot_hash'}):
        raise ValueError('trend source snapshot identity invalid')
    if report['source_snapshot'].get('snapshot_hash')!=report['config'].get('source_snapshot_hash'):
        raise ValueError('trend source snapshot binding mismatch')
    if stable_hash(report['dataset_records'])!=report['source_snapshot']['records_hash']:raise ValueError('trend input catalog drift')
    verify_source_generators(report['source_snapshot']['source_snapshot'])
    if stable_hash(report.get('population_folds',[]))!=freeze.get('population_folds_hash'):
        raise ValueError('frozen actual-account population catalogue drift')
    if stable_hash(report.get('dataset_manifests',[]))!=freeze.get('dataset_manifests_hash'):
        raise ValueError('frozen rebuilt dataset manifest catalogue drift')
    held=report.get('holding_population')
    if stable_hash(held)!=freeze.get('holding_population_hash'):
        raise ValueError('frozen independent E1 holding population catalogue drift')
    audit_holding_population(held,directory,report['config'],report['dataset_records'])
    from my_strategy.services.czsc_reference_quality import audit_reference_quality
    reference=report.get('reference_quality')
    if stable_hash(reference)!=freeze.get('reference_quality_hash'):
        raise ValueError('frozen independent reference quality population/catalogue drift')
    audit_reference_quality(directory,reference,report['config'],[r['symbol'] for r in report['dataset_records']],expected_records=report['dataset_records'])
    if stable_hash(report.get('holding_population_diagnostics'))!=freeze.get('holding_population_diagnostics_hash'):
        raise ValueError('frozen E1 holding split diagnostics drift')
    selection=report.get('source_policy_selection') or {}
    asset=selection.get('accounts_asset')
    if asset:
        path=(directory/asset['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or _file_hash(path)!=asset['sha256']:
            raise ValueError('frozen source policy Broker account evidence changed')
    for asset in report.get('dataset_manifests',[]):
        path=(directory/asset['path']).resolve()
        if not path.is_relative_to(directory.resolve()) or _file_hash(path)!=asset['sha256']:
            raise ValueError('frozen E0/E1/actual-account dataset manifest changed')
    for path,sha in freeze.get('code_hashes',{}).items():
        if _file_hash(MY_STRATEGY_ROOT/path)!=sha:raise ValueError('trend generating code drift requires new training')
    for population in report.get('population_folds',[]):
        path=(directory/population['manifest_path']).resolve()
        if not path.is_relative_to(directory.resolve()) or _file_hash(path)!=population['manifest_sha256']:
            raise ValueError('frozen actual-account training population drift')
        value=json.loads(path.read_text())
        if value.get('reference_population')!=population['reference_population'] or value.get('failures'):
            raise ValueError('actual-account training population protocol mismatch')
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
            if head=='entry':
                population=next((item for item in report['population_folds'] if item['fold']==bundle['name']),None)
                if population is None or target.get('reference_population')!=population['reference_population']:
                    raise ValueError('L2 entry head belongs to another actual-account population')
            if head in L2_HOLDING_HEADS and target.get('holding_population')!=held:
                raise ValueError('L2 exit head belongs to another independent actual E1 holding population')


def read_snapshot(record):
    for name,sha in (('path','sha256'),('native_path','native_sha256')):
        if _file_hash(resolve_frozen_asset(record[name], runs_root=ARTIFACT_RUNS_ROOT))!=record[sha]:raise ValueError('frozen native input drift')
    frame=pd.read_parquet(resolve_frozen_asset(record['path'], runs_root=ARTIFACT_RUNS_ROOT));frame['date']=pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    native_path=resolve_frozen_asset(record['native_path'], runs_root=ARTIFACT_RUNS_ROOT)
    if native_path.suffix=='.gz':
        import gzip
        with gzip.open(native_path,'rt',encoding='utf-8') as stream:pack=json.load(stream)
    else:pack=json.loads(native_path.read_text())
    pack['frame']=frame
    attrs=record.get('raw_attrs')
    required={'price_basis','legacy_fuyao_bars','unverified_trade_price_bars','volume_unit','amount_unit','data_version'}
    if not isinstance(attrs,dict) or not required.issubset(attrs) or attrs['data_version']!=record.get('raw_series_hash'):
        raise ValueError('native snapshot lacks its original price-basis/observation identity')
    frame.attrs.update(copy.deepcopy(attrs))
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
        'continuation_policy_hash':contract.get('continuation_policy_hash'),'continuation_stage':contract.get('stage'),
        'training_population':contract.get('training_population'),'reference_population_hash':contract.get('reference_population_hash'),
        'holding_population_hash':contract.get('holding_population_hash'),
        'conditional_holding_domain':contract.get('conditional_holding_domain'),
        'economic_group_identity':contract.get('economic_group_identity'),'attempt_identity':contract.get('attempt_identity'),
        'split_policy':contract.get('split_policy'),'economic_weighting':contract.get('economic_weighting')}


def forward_trend(frame,root,trend_bundles,config,session=None,mode='historical',market_dates=None):
    session=session or TrendSession();root=Path(root)
    calendar=market_dates or frame.attrs.get('verified_market_dates') or json.loads((root/'data_snapshots'/'calendar.json').read_text())['dates']
    data=bind_trend_inputs(frame,calendar,config)
    data['date']=pd.to_datetime(data.date).dt.strftime('%Y-%m-%d');routes={};groups={}
    for i,row in enumerate(data.to_dict('records')):
        day=row['date'];bundle=_bundle(day,trend_bundles,'trend_direction',root,mode)
        routes[day]={'date':day,'available_at':day+'T15:00:00+08:00','input_eligible':bool(row.get('trend_input_eligible')),
            'model_available':False,'p_up':None,'p_down':None,'p_neutral':None,
            'reason_codes':['no_prior_trend_head'] if bundle is None else ['trend_input_ineligible'],
            'model_identity':_identity(None,None,root,day)}
        if bundle:groups.setdefault(bundle['name'],{'bundle':bundle,'indices':[]})['indices'].append(i)
    for group in groups.values():
        bundle=group['bundle'];path=root/bundle['heads']['trend_direction']['path'];m=load_trend_head(path)
        if m['target_contract']['parameters']!=trend_config(config):raise ValueError('trend target parameters changed without retraining')
        if config.get('source_snapshot_hash') and m['target_contract']['source_snapshot_hash']!=config['source_snapshot_hash']:raise ValueError('trend source identity mismatch')
        if config.get('pipeline_contract_hash') and m['target_contract']['pipeline_contract_hash']!=config['pipeline_contract_hash']:
            raise ValueError('trend pipeline contract changed without retraining')
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
        from my_strategy.services.czsc_trend_labels import label_config
        if contract.get('high_return') != label_config(config):
            raise ValueError('trend interval high_return target changed without retraining')
        if contract.get('exit_advantage_margin') != float(config.get('high_return', {}).get('exit_advantage_margin', .002)):
            raise ValueError('trend interval exit_advantage target changed without retraining')
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
    values['input_eligible']=bool(candidate.get('liquidity_qualified',feature.get('liquidity_pool_eligible',False))
        and any(values.get(k+'_input_available',0) for k in candidate.get('sources',[])))
    p,m=_predict_interval(values,root,bundle,'entry',session,mode,config=config)
    if m['target_contract'].get('stage')!='E1':raise ValueError('E2 or bootstrap head cannot serve E1 account')
    return p,_identity(m,bundle,root,day),'scored' if p is not None else 'input_ineligible'


def pinned_exit_policy(frame,calendar,prepared,root,ma_exit_bundles,config,session=None,mode='historical'):
    from my_strategy.services.czsc_trend_consensus import observable
    frame=bind_trend_inputs(frame,calendar,config)
    session=session or HeadSession();root=Path(root);rows={str(r['date'])[:10]:r for r in frame.to_dict('records')}
    midx={d:i for i,d in enumerate(calendar)};bindings={};catalog=copy.deepcopy(ma_exit_bundles)
    def original_binding(snapshot):
        # Reservation filtering can rebuild the accepted plan after the static
        # rule preparation. The Broker position retains that exact frozen plan.
        plan=snapshot.get('entry_plan') or {};identifier=snapshot.get('plan_id')
        signal=snapshot.get('entry_signal_date');day=snapshot['date']
        if (not identifier or not signal or plan.get('plan_id')!=identifier or
                plan.get('symbol')!=snapshot.get('symbol') or plan.get('signal_date')!=signal or
                not observable(plan.get('available_at'),signal+'T15:00:00+08:00')):
            raise ValueError('actual holding original signal/plan identity mismatch')
        if not signal<snapshot['entry_date']<=day:
            raise ValueError('holding clock violates original binding')
        plan_hash=stable_hash(plan)
        if identifier in bindings:
            binding=bindings[identifier]
            if binding['candidate_signal_date']!=signal or binding['entry_plan_hash']!=plan_hash:
                raise ValueError('actual holding changed its original accepted plan')
            return binding
        bundle=_bundle(signal,catalog,'ma_exit',root,mode)
        if bundle:
            m=load_head(root/bundle['heads']['ma_exit']['path'])
            if (m['train_label_end']>=signal or m['calibration_label_end']>=signal or
                    m.get('policy_label_end',m['fold']['policy_end'])>=signal or
                    m['fold']['policy_end']!=bundle['available_at'] or m['target_contract']['stage']!='E0'):
                raise ValueError('MA exit binding is not strictly prior E0 target')
        bindings[identifier]={'plan_id':identifier,'candidate_signal_date':signal,'entry_plan_hash':plan_hash,
            'binding_source':'actual_Broker_accepted_entry_plan',
            'checkpoint':bundle['name'] if bundle else None,'available_at':bundle['available_at'] if bundle else None,
            'bundle':bundle,'mode':'signal_time_pinned_holding_exit' if bundle else 'explicit_rule_E0_no_prior_exit_head',
            'checkpoint_upgrade':False,'source':'union'}
        return bindings[identifier]
    recipe={'family':STRATEGY_ID,'stage':'E1','baseline':'E0','exit_catalog_hash':stable_hash(ma_exit_bundles),
        'candidate_policy_hash':prepared['config_hash'],'holding_checkpoint_upgrade':False,
        'fallback':'E0_for_entire_position','l2_exit_apply':False};recipe_hash=config.get('E1_policy_hash') or stable_hash(recipe)
    def fresh():
        prefixes={}
        def callback(snapshot):
            binding=original_binding(snapshot)
            signal=binding['candidate_signal_date'];bundle=binding['bundle'];day=snapshot['date']
            saved=snapshot.get('exit_model_binding') or snapshot.get('entry_exit_binding') or snapshot.get('candidate_exit_binding')
            if saved and any(saved.get(key)!=binding.get(key) for key in
                    ('plan_id','candidate_signal_date','checkpoint','available_at','mode','source')):
                raise ValueError('clone changed original exit binding')
            if saved and saved.get('entry_plan_hash') and saved['entry_plan_hash']!=binding['entry_plan_hash']:
                raise ValueError('clone changed original accepted plan')
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
    data=bind_trend_inputs(frame,calendar,config);data['date']=pd.to_datetime(data.date).dt.strftime('%Y-%m-%d')
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
            'input_eligible':bool(row['trend_input_eligible']),'model_available':bool(np.isfinite(p).all()),
            'p_up':float(p[0]) if np.isfinite(p).all() else None,'p_down':float(p[1]) if np.isfinite(p).all() else None,
            'p_neutral':float(p[2]) if np.isfinite(p).all() else None,'model_identity':identity,'reason_codes':[]}
    return routes


def m_contract(manifest):return manifest['target_contract']


def _policy_worker(payload):
    import itertools
    from my_strategy.services.czsc_trend_consensus import prepare_policy_base,apply_policy_thresholds
    from my_strategy.services.czsc_trend_account import replay_trend_account,prepare_account_context
    record,calendar,cfg,root,bundle,variants=payload
    frame,pack=read_snapshot(record);fold=bundle['fold'];frame=frame[frame.date<=fold['policy_end']].copy()
    if not frame.date.ge(fold['policy_start']).any():
        from my_strategy.execution.broker import BrokerSimulator
        from my_strategy.execution.cost_model import CostModel
        # A later-listed or unobserved stock retains its original equal cash
        # account in every policy denominator. No missing quote is backfilled.
        account=BrokerSimulator(cfg['initial_cash'],CostModel(),run_id='policy:'+record['symbol'])
        ledger=account.ledger.to_frame().to_dict('records')
        return [{'variant':variant,'symbol':record['symbol'],'metrics':{
            'total_return':0.,'max_drawdown':0.,'final_equity':account.cash,'trade_count':0,'open_shares':0},
            'ledger':ledger,'rejections':[],'open_position':None,'ledger_hash':stable_hash(ledger),
            'fold':fold['name'],'status':'cash_no_stock_observation_in_policy_window',
            'reason_codes':['no_local_observation_in_policy_window']} for variant in variants]
    routes=policy_trend(frame,root,bundle,cfg,market_dates=calendar);context=prepare_account_context(frame,calendar,cfg);done=[]
    # Every variant starts flat with the same cash. Its quote seasoning uses
    # only known prior observations, and all remaining inventory is disclosed.
    prior=frame[frame.date<fold['policy_start']]
    seed={'version':STRATEGY_ID,'symbol':record['symbol'],'initial_cash':cfg['initial_cash'],'cash':cfg['initial_cash'],
        'shares':0,'positions':{},'avg_cost':{},'ledger':[],'actual_position':None,'pending_intent':None,'last_buy_date':None,
        'holding_prefix':{},'last_close_date':prior.date.iloc[-1] if not prior.empty else calendar[calendar.index(fold['policy_start'])-1],
        'last_mark':float(prior.close.iloc[-1]) if not prior.empty else None,
        'seasoned_bars':int((prior.has_trade_price.eq(1)&prior.source.isin(cfg['account']['valid_sources'])).sum()),
        'consumed_plan_ids':[],'series_context_hash':'policy_validation_initial_flat',
        'source_context_hash':cfg['source_snapshot_hash'],'current_snapshot':{}}
    prepared_base=None;base_cross=None
    for variant in variants:
        candidate_cfg=copy.deepcopy(cfg);candidate_cfg['trend']['candidate_cross_threshold']=variant['cross']
        candidate_cfg['consensus'].update(buy_base_threshold=variant['buy_base'],sell_threshold=variant['sell'],margin=variant['margin'])
        if prepared_base is None or base_cross != variant['cross']:
            prepared_base=prepare_policy_base(frame,calendar,pack,routes,candidate_cfg)
            base_cross=variant['cross']
        prepared=apply_policy_thresholds(prepared_base,candidate_cfg)
        account=replay_trend_account(frame,calendar,prepared,config=candidate_cfg,start=fold['policy_start'],end=fold['policy_end'],
            initial_state=seed,run_id='policy:'+record['symbol'],mode='E0',account_context=context)
        done.append({'variant':variant,'symbol':record['symbol'],'metrics':account['metrics'],'ledger':account['ledger'],
            'rejections':account['rejections'],'open_position':account['actual_position'],
            'ledger_hash':stable_hash(account['ledger']),'fold':fold['name']})
    return done


def select_source_policy(records,calendar,cfg,root,trend_bundles,context,cpu_workers):
    """Only earliest independent policy segment; locked development never selects."""
    import itertools
    if not trend_bundles:
        raise ValueError('source policy selection requires a qualified independent trend head')
    bundle=min(trend_bundles,key=lambda b:b['available_at']);grid=cfg['consensus']['policy_grid']
    variants=[{'cross':c,'buy_base':b,'sell':s,'margin':m} for c,b,s,m in itertools.product(
        cfg['trend']['candidate_cross_grid'],grid['buy_base_threshold'],grid['sell_threshold'],grid['margin'])]
    payloads=((r,calendar,cfg,str(root),bundle,variants) for r in records)
    stats={stable_hash(v):{'round_trips':0,'return_sum':0.,'drawdown_sum':0.,'count':0} for v in variants}
    accounts_path=context.subdir('validation')/'source_policy_accounts.jsonl'
    accounts_count=0;completed_stocks=0;total_stocks=len(records)
    with accounts_path.open('w',encoding='utf-8') as stream:
        for chunk in _bounded_results(_policy_worker,payloads,cpu_workers,cfg['stock_batch_size']):
            for account in chunk:
                stream.write(json.dumps(_json(account),ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                stats_row=stats[stable_hash(account['variant'])]
                stats_row['round_trips']+=completed_round_trips(account['ledger'])
                stats_row['return_sum']+=account['metrics']['total_return']
                stats_row['drawdown_sum']+=account['metrics']['max_drawdown']
                stats_row['count']+=1;accounts_count+=1
            completed_stocks+=1;stream.flush()
            print('POLICY_STOCK_DONE '+json.dumps({'symbol':chunk[0]['symbol'],'current':completed_stocks,
                'total':total_stocks,'failed':0,'device':'cpu','variants':len(chunk),'fold':bundle['fold']['name']},
                ensure_ascii=False,separators=(',',':')),flush=True)
    aggregates=[]
    for variant in variants:
        aggregate=stats[stable_hash(variant)];count=aggregate['round_trips'];n=aggregate['count']
        score=aggregate['return_sum']/n if n else 0.
        aggregates.append({'variant':variant,'round_trips':count,'mean_fee_net_return':score,
            'mean_max_drawdown':aggregate['drawdown_sum']/n if n else 0.,
            'scored_population':n,'eligible_for_selection':count>=cfg['min_validation_round_trips']})
    valid=[x for x in aggregates if x['eligible_for_selection']]
    choice=max(valid,key=lambda x:(x['mean_fee_net_return'],x['mean_max_drawdown'],-x['variant']['cross'])) if valid else None
    if choice:
        v=choice['variant'];cfg['trend']['candidate_cross_threshold']=v['cross']
        cfg['consensus'].update(buy_base_threshold=v['buy_base'],sell_threshold=v['sell'],margin=v['margin'])
    report={'phase':'candidate_and_ordinary_SELL_policy_only','fold':bundle['fold'],'variants':aggregates,
        'selected':choice,'status':'policy_selected' if choice else 'insufficient_policy_trades_use_preregistered_baseline',
        'initial_cash':cfg['initial_cash'],'four_development_windows_used':False,'high_return_filter_used':False,
        'accounts_asset':{'path':str(accounts_path.relative_to(context.run_dir())),'sha256':_file_hash(accounts_path),
            'format':'jsonl','records':accounts_count}}
    _save(context.subdir('validation')/'source_policy_selection.json',report)
    return report


def make_reference_population(fold, prior_entry_bundles, reassessment=None):
    """The head being trained can never choose its own account population."""
    catalog=copy.deepcopy(prior_entry_bundles)
    if any(bundle['name']==fold['name'] or bundle['available_at']>=fold['calibration_start'] or
           not bundle.get('heads',{}).get('entry') for bundle in catalog):
        raise ValueError('reference population requires completed earlier L2 entry heads only')
    return {'version':POPULATION_VERSION,'fold':copy.deepcopy(fold),'population_end':fold['policy_end'],
        'prior_entry_catalog':catalog,'prior_entry_catalog_hash':stable_hash(catalog),
        'reference_mode':'final','no_prior_entry':'blocked_actual_account_with_forced_entry_label_clone',
        'admission_rule':'strictly_prior_available_at_before_each_signal_day',
        'label_continuation':'forced_once_then_frozen_E1_no_L2_exit',
        'self_head_used':False,'reference_and_clone_ledgers_separate':True,
        'split_policy':(reassessment or {}).get('split_policy',SPLIT_POLICY),
        'economic_weighting':(reassessment or {}).get('economic_weighting',ECONOMIC_WEIGHTING),
        'atomic_split_policy':(reassessment or {}).get('split_policy',SPLIT_POLICY),
        'atomic_economic_weighting':(reassessment or {}).get('economic_weighting',ECONOMIC_WEIGHTING)}


def reference_entry_evaluator(frame, calendar, prepared, root, prior_bundles, cfg, population):
    from my_strategy.services.czsc_trend_features import event_prefix_rows, original_evidence_events
    catalog=population['prior_entry_catalog']
    if catalog!=prior_bundles or stable_hash(catalog)!=population['prior_entry_catalog_hash']:
        raise ValueError('reference L2 catalogue differs from its frozen population protocol')
    values={str(row['date'])[:10]:row for row in frame.to_dict('records')}
    prefixes=dict(zip(frame.date,event_prefix_rows(frame,original_evidence_events(prepared),calendar),strict=True))
    session=HeadSession()
    def evaluate(snapshot,candidate):
        day=snapshot['date'];plan=candidate.get('entry_plan') or {}
        probability,identity,status=predict_entry(values[day],snapshot,candidate,plan,prefixes[day],
            root,catalog,cfg,session,'historical')
        passed=status=='scored' and probability is not None and probability>=cfg['consensus']['high_return_threshold']
        return {'date':day,'available_at':day+'T15:00:00+08:00','status':status,
            'evaluation_status':('scored_passed' if passed else 'scored_rejected') if status=='scored' else status,
            'probability':probability,'model_identity':identity,'in_scope':status=='scored',
            'threshold':cfg['consensus']['high_return_threshold'],'usage_mode':'historical',
            'reason_codes':[] if passed else ['high_return_below_threshold'] if status=='scored' else [status],
            'reference_population_hash':stable_hash(population)}
    evaluate.recipe_hash=stable_hash({'family':STRATEGY_ID,'population':population,'continuation':cfg['E1_policy_hash']})
    return evaluate


def _dataset_worker(payload):
    import torch
    torch.set_num_threads(1)
    from my_strategy.services.czsc_trend_dataset import build_account_dataset
    record,calendar,cfg,root,trend_bundles,ma_bundles,stage,options=payload
    frame,pack=read_snapshot(record)
    # Early bars remain warm-up context; no learned forecast exists before a
    # legitimately prior checkpoint. No source price is substituted.
    frame=frame[frame.date<=cfg['training_end']].copy();frame.attrs.update(source_snapshot_hash=cfg['source_snapshot_hash'])
    frame=bind_trend_inputs(frame,calendar,cfg)
    routes=forward_trend(frame,root,trend_bundles,cfg,market_dates=calendar)
    callback=None
    if stage=='E1':
        from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
        prepared=prepare_trend_decisions(frame,calendar,evidence_pack=pack,trend_probabilities=routes,config=cfg,mode='E0')
        callback=pinned_exit_policy(frame,calendar,prepared,root,ma_bundles,cfg)
    population=options.get('reference_population')
    entry_callback=None
    if population:
        if stage!='E1':raise ValueError('actual final population requires frozen E1 continuation')
        entry_callback=reference_entry_evaluator(frame,calendar,prepared,root,
            population['prior_entry_catalog'],cfg,population)
    directory=Path(root)/'dataset'/options.get('subdirectory',stage)/record['symbol'].replace('.','_')
    result,summary=build_account_dataset(frame,calendar,pack,routes,cfg,stage=stage,exit_policy=callback,
        directory=directory,run_id=Path(root).name+':'+record['symbol'],
        reference_entry_evaluator=entry_callback,reference_population=population,
        population_end=options.get('population_end'))
    print('DATASET '+stage+' '+record['symbol']+' entries='+str(summary['entry_rows'])+' exits='+str(summary['exit_rows']),flush=True)
    return summary


def _bounded_results(worker,payloads,cpu_workers,batch_size):
    """Keep work/results bounded; batches never select or truncate stocks."""
    from collections import deque
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    if cpu_workers <= 1:
        for payload in payloads:
            yield worker(payload)
        return
    maximum=max(1,min(2*cpu_workers,batch_size))
    iterator=iter(payloads)
    with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending=deque()
        for _ in range(maximum):
            payload=next(iterator,None)
            if payload is None:break
            pending.append(pool.submit(worker,payload))
        while pending:
            result=pending.popleft().result()
            payload=next(iterator,None)
            if payload is not None:pending.append(pool.submit(worker,payload))
            yield result


def _pool_parquets(paths,destination,*,load=True):
    """Union schemas first, then persist each stock without a concat copy peak."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    paths=list(paths)
    if not paths:raise ValueError('full population has no stock dataset assets')
    schema=pa.unify_schemas([pq.read_schema(path) for path in paths],promote_options='permissive')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+'.tmp')
    with pq.ParquetWriter(temporary,schema,compression='snappy') as writer:
        for path in paths:
            table=pq.read_table(path)
            columns=[table[column.name].cast(column.type) if column.name in table.column_names
                else pa.nulls(table.num_rows,type=column.type) for column in schema]
            writer.write_table(pa.Table.from_arrays(columns,schema=schema))
    temporary.replace(destination)
    return pd.read_parquet(destination) if load else None


def build_stage(records,calendar,cfg,context,stage,trend_bundles,ma_bundles,cpu_workers,*,
                reference_population=None,population_end=None,subdirectory=None):
    subdirectory=subdirectory or stage
    options={'subdirectory':subdirectory,'reference_population':reference_population,'population_end':population_end}
    payloads=((r,calendar,cfg,str(context.run_dir()),trend_bundles,ma_bundles,stage,options) for r in records)
    def progress(completed):
        print('PROGRESS '+json.dumps({'stage':'dataset','population':subdirectory,
            'fold':(reference_population or {}).get('fold',{}).get('name'),
            'completed':completed,'total':len(records),'scope':'current_population_only'},
            ensure_ascii=False,separators=(',',':')),flush=True)
    summaries=[];progress(0)
    for summary in _bounded_results(_dataset_worker,payloads,cpu_workers,cfg['stock_batch_size']):
        summaries.append(summary);progress(len(summaries))
    order={record['symbol']:i for i,record in enumerate(records)}
    summaries.sort(key=lambda summary:order[summary['symbol']])
    if [summary['symbol'] for summary in summaries]!=[record['symbol'] for record in records]:
        raise ValueError('dataset preparation did not complete the entire frozen stock population')
    pooled={}
    pooled_files={}
    for key in ('entries','exits','episodes'):
        path=context.subdir('dataset',subdirectory)/(key+'.parquet')
        paths=(Path(context.run_dir())/'dataset'/subdirectory/s['symbol'].replace('.','_')/(key+'.parquet') for s in summaries)
        pooled[key]=_pool_parquets(paths,path)
        pooled_files[key]={'path':str(path),'sha256':_file_hash(path)}
    pooled['reference_population']=copy.deepcopy(reference_population or {})
    pooled['reference_mode']='final' if reference_population else stage
    _save(context.subdir('dataset',subdirectory)/'manifest.json',{'stage':stage,'records':summaries,'failures':[],
        'reference_population':reference_population or {},'reference_population_hash':stable_hash(reference_population or {}),
        'population_end':population_end,'source_snapshot_hash':cfg['source_snapshot_hash'],'pooled_files':pooled_files})
    return pooled,summaries


def freeze_holding_population(dataset,root,cfg,records):
    """Freeze actual E1 held pairs before any L2 head or gate chooses accounts."""
    root=Path(root).resolve();manifest_path=root/'dataset/E1/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    exits=manifest.get('pooled_files',{}).get('exits') or {}
    value={'version':HELD_TRAINING_POPULATION,'stage':'E1','reference_mode':'E1',
        'conditional_holding_domain':HELD_CONDITION_DOMAIN,
        'pending_L2_entry_or_exit_used':False,'candidate_clones_are_holding_population':False,
        'holding_checkpoint_upgrade':False,'split_policy':HELD_SPLIT_POLICY,
        'economic_weighting':HELD_ECONOMIC_WEIGHTING,'source_snapshot_hash':cfg['source_snapshot_hash'],
        'continuation_policy_hash':cfg['E1_policy_hash'],'candidate_policy_hash':cfg['candidate_policy_hash'],
        'manifest_path':'dataset/E1/manifest.json','manifest_sha256':_file_hash(manifest_path),
        'pooled_exits_path':'dataset/E1/exits.parquet','pooled_exits_sha256':exits.get('sha256'),
        'records_hash':stable_hash(manifest.get('records',[])),
        'frozen_before_L2_training_at':local_now().isoformat()}
    audit_holding_population(value,root,cfg,records)
    if dataset.get('reference_population') or dataset.get('reference_mode')!='E1':
        raise ValueError('independent exit population cannot be selected by an L2 entry gate')
    dataset['holding_population']=copy.deepcopy(value)
    return value


def audit_holding_population(population,root,cfg,records):
    validate_holding_population(population)
    if any(population[field]!=cfg[config_name] for field,config_name in
            (('source_snapshot_hash','source_snapshot_hash'),('continuation_policy_hash','E1_policy_hash'),
             ('candidate_policy_hash','candidate_policy_hash'))):
        raise ValueError('frozen actual E1 holding population belongs to another source/continuation')
    root=Path(root).resolve()
    path=(root/population['manifest_path']).resolve();exits=(root/population['pooled_exits_path']).resolve()
    if not path.is_relative_to(root) or not exits.is_relative_to(root):
        raise ValueError('frozen actual E1 holding population escaped its model run')
    if _file_hash(path)!=population['manifest_sha256'] or _file_hash(exits)!=population['pooled_exits_sha256']:
        raise ValueError('frozen actual E1 holding population manifest or paired rows changed')
    manifest=json.loads(path.read_text());stored=manifest.get('records') or []
    if (manifest.get('stage')!='E1' or manifest.get('reference_population') or manifest.get('failures')
            or manifest.get('source_snapshot_hash')!=cfg['source_snapshot_hash']
            or stable_hash(stored)!=population['records_hash']
            or [r['symbol'] for r in stored]!=[r['symbol'] for r in records]
            or any(r.get('reference_mode')!='E1' or r.get('reference_population') for r in stored)
            or manifest.get('pooled_files',{}).get('exits',{}).get('sha256')!=population['pooled_exits_sha256']
            or Path(manifest['pooled_files']['exits']['path']).resolve()!=exits):
        raise ValueError('frozen actual E1 holding population contains final-gated or foreign account evidence')
    return manifest


def holding_population_diagnostics(dataset,folds):
    """Exact unchanged split masks; counts belong to this one E1 population."""
    from my_strategy.services.czsc_trend_interval_model import time_masks
    frame=dataset['exits'];result={'population':HELD_TRAINING_POPULATION,
        'holding_population_hash':stable_hash(dataset['holding_population']),
        'rows':len(frame),'fold_counts_are_additive':False,
        'conditional_holding_domain':HELD_CONDITION_DOMAIN,'folds':[]}
    observed=frame.label_available.fillna(False) & frame.label_end.notna()
    result['mature_rows']=int(observed.sum())
    result['mature_classes']={str(label):int((observed & frame.label.eq(label)).sum()) for label in (0,1)}
    for fold in folds:
        item={'fold':fold,'heads':{}}
        for head in ('exit','exit_advantage'):
            values=frame.copy()
            if head=='exit_advantage':values['label']=values.exit_advantage
            masks=time_masks(values,fold,head=head,regression=head=='exit_advantage')
            item['heads'][head]={name:{'rows':int(mask.sum()),
                'actual_held_episodes':int(values.loc[mask,'episode_id'].nunique()),
                'classes':{str(label):int((mask & frame.label.eq(label).to_numpy()).sum()) for label in (0,1)},
                'label_end_max':str(pd.to_datetime(values.loc[mask,'label_end']).max().date()) if mask.any() else None}
                for name,mask in zip(('train','calibration','policy'),masks)}
        result['folds'].append(item)
    return result


def holding_head_status(bundles,unavailable,production_name):
    """Training completion is separate from final-account fee qualification."""
    return {head:{'completed_heads':sum(bool(bundle.get('heads',{}).get(head)) for bundle in bundles),
        'production_linear_completed':any(bundle.get('name')==production_name
            and bundle.get('architecture')=='linear' and bundle.get('heads',{}).get(head) for bundle in bundles),
        'unavailable_heads':[copy.deepcopy(item) for item in unavailable if item.get('head')==head],
        'conditional_holding_domain':HELD_CONDITION_DOMAIN,
        'final_account_performance_qualified':False,
        'final_account_performance_status':'requires_actual_fee_net_account_evaluation_not_E1_sample_counts'}
        for head in ('exit','exit_advantage')}


def _trend_observation_worker(payload):
    record,calendar,cfg,directory=payload
    frame,_=read_snapshot(record);frame=frame[frame.date<=cfg['training_end']]
    observations=build_trend_labels(frame,calendar,cfg,source_snapshot_hash=cfg['source_snapshot_hash'],
        pipeline_contract_hash=cfg['pipeline_contract_hash'])
    path=Path(directory)/(record['symbol'].replace('.','_')+'.parquet')
    observations[observations.date>=cfg['feature_start']].to_parquet(path,index=False)
    return {'symbol':record['symbol'],'path':str(path),'sha256':_file_hash(path),'label_contract':observations.attrs['label_contract']}


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



def _emit_head_resource(manifest, *, fold=None, head=None, architecture=None):
    resources=manifest.get('training_resources') or {}
    device=resources.get('actual_device',resources.get('device',manifest.get('actual_device','unknown')))
    print('HEAD '+json.dumps({'fold':(manifest.get('fold') or fold or {}).get('name'),'head':manifest.get('head',head),
        'architecture':manifest.get('architecture',architecture),'device':device,
        'training_resources':resources},ensure_ascii=False,separators=(',',':')),flush=True)


def _fit_interval(dataset,head,fold,calendar,context,cfg,architecture='linear',device='cpu'):
    from my_strategy.services.czsc_trend_interval_model import split_validation
    if 'calibration_start' not in fold:fold=split_validation({**fold,'validation_start':fold.get('validation_start',str((pd.Timestamp(fold['validation_end']).to_period('Q')).start_time.date()))},calendar)
    columns=L1_MA_EXIT_COLUMNS if head=='ma_exit' else ENTRY_FEATURE_COLUMNS if head in ENTRY_HEADS else EXIT_FEATURE_COLUMNS
    path=context.subdir('models',architecture,fold['name'],head)
    manifest=train_head(dataset,columns,path,fold,cfg,head=head,architecture=architecture,regression=head in {'return','exit_advantage'},
        device=device,contract=dataset.attrs['label_contract'])
    _emit_head_resource(manifest,fold=fold,head=head,architecture=architecture)
    relative=str(path.relative_to(context.run_dir()))
    return {'name':fold['name'],'available_at':fold['policy_end'],'fold':fold,'architecture':architecture,
        'heads':{head:{'path':relative,'manifest_sha256':manifest['manifest_sha256'],'target_contract_hash':manifest['target_contract_hash']}},
        'policy':{'entry_threshold':cfg['consensus']['high_return_threshold'],'exit_threshold':cfg['account']['holding_exit_threshold']}},manifest


def required_head_status(folds,trend_bundles,ma_exit_bundles,entry_bundles,production_name):
    expected=max(folds,key=lambda fold:fold['policy_end']) if folds else None
    def qualified(bundles,head):
        return bool(expected and any(bundle['name']==expected['name'] and
            bundle['available_at']==expected['policy_end'] and bundle.get('heads',{}).get(head) for bundle in bundles))
    return {'recent_checkpoint':expected['name'] if expected else None,
        'recent_trend_ready':qualified(trend_bundles,'trend_direction'),
        'same_checkpoint_ma_exit_ready':qualified(ma_exit_bundles,'ma_exit'),
        'production_entry_ready':any(bundle['name']==production_name and bundle.get('heads',{}).get('entry') for bundle in entry_bundles)}


def _capacity_probe(context,cfg,source,records,end,unavailable,code,folds):
    """A cohort without a trend dependency cannot claim a completed model DAG."""
    if any(_file_hash(MY_STRATEGY_ROOT/path)!=sha for path,sha in code.items()):
        raise ValueError('generating code changed during frozen capacity probe')
    verify_source_generators(source['source_snapshot'])
    assets=[]
    for relative in ('dataset/trend/observations.parquet','data_snapshots/source.json',
            'data_snapshots/records.json','data_snapshots/calendar.json','config/initial_config.json',
            'config/generating_code_hashes.json'):
        path=context.run_dir()/relative;assets.append({'path':relative,'sha256':_file_hash(path)})
    freeze={'family':STRATEGY_ID,'actual_freeze_at':local_now().isoformat(),'config_hash':stable_hash(cfg),
        'pipeline_contract_hash':cfg['pipeline_contract_hash'],'source_snapshot_hash':source['snapshot_hash'],
        'code_hashes':code,'probe_assets_hash':stable_hash(assets),'complete_model_population_DAG':False}
    freeze['freeze_hash']=stable_hash(freeze)
    report={'run_id':context.run_id,'status':'capacity_insufficient','strategy_version':STRATEGY_ID,
        'config':cfg,'source_snapshot':source,'data_end':end,'dataset_records':records,
        'trend_bundles':[],'ma_exit_bundles':[],'model_bundles':[],'architecture_bundles':[],
        'source_policy_selection':None,'holding_exit_policy_selection':[],'continuation_stage':None,
        'population_folds':[],'dataset_manifests':[],'unavailable_heads':unavailable,'failures':[],
        'pipeline':{'stages':['independent_trend'],'complete_model_population_DAG':False,
            'stopped_dependency':'qualified_three_class_trend_head','trend_heads':0,'ma_exit_heads':0,'entry_heads':0},
        'required_recent_heads_ready':False,'automatic_selection_eligible':False,
        'required_head_status':required_head_status(folds,[],[],[],cfg['production']['name']),
        'probe_assets':assets,'protocol_freeze':freeze,'completed_at':local_now().isoformat(),
        'model_gate':{'passed':False,'reasons':['no_qualified_trend_head','capacity_dependency_insufficient']},
        'publication_allowed':False}
    _save(context.subdir('validation')/'protocol_freeze.json',freeze)
    _save(context.subdir('reports')/'trend_consensus.json',report)
    context.write_metadata({'status':'capacity_insufficient','pipeline':report['pipeline'],'unavailable_heads':unavailable})
    print('CAPACITY_PROBE '+context.run_id+' no_qualified_trend_head',flush=True)
    return report


def _train_one(*,source_native_run_id,
                          symbols=None,end='2026-10-08',config=None,cpu_workers=6,device='cpu',native_records=None):
    """A new immutable family; no old model checkpoint is imported."""
    import torch
    torch.set_num_threads(min(4,cpu_workers))
    from my_strategy.services.czsc_trend_dataset import training_frame, label_maturity_counts
    cfg=validate_training_config(config);cfg['cpu_workers']=cpu_workers;cfg['training_end']=end
    old=json.loads((ARTIFACT_RUNS_ROOT/source_native_run_id/'reports'/'native_source.json').read_text())
    if (old.get('strategy_version')!=STRATEGY_ID or old.get('status')!='completed' or old.get('failures')
            or old.get('raw_only_rebuild') is not True or old.get('reused_model_or_label') is not False):
        raise ValueError('reassessment training requires a fresh current source rebuilt from raw inputs only')
    if end>str(old.get('data_end') or '')[:10]:
        raise ValueError('training end exceeds the new frozen raw input source')
    verify_source_generators(old.get('source_snapshot') or {})
    records=[r for r in (native_records or old['dataset_records']) if symbols is None or r['symbol'] in symbols]
    if not records or (symbols and {r['symbol'] for r in records}!=set(symbols)):raise ValueError('requested source cohort unavailable')
    liquidity=verify_full_population(old,cfg,records)
    cfg['training_capacity_tiers']=[len(records)]
    cfg['capacity_expansion_criterion']=FULL_CAPACITY_CRITERION
    for record in records:
        if record.get('raw_only_rebuild') is not True or record.get('reused_model_or_label') is not False or old['source_snapshot'].get('raw_input_hashes',{}).get(record['symbol'])!=record.get('raw_sha256'):
            raise ValueError('training record is not bound to this fresh raw-only input superset')
        if _file_hash(resolve_frozen_asset(record['raw_path'], runs_root=ARTIFACT_RUNS_ROOT))!=record['raw_sha256']:
            raise ValueError('fresh raw observation snapshot changed')
        if not isinstance(record.get('raw_attrs'),dict) or not record['raw_attrs'].get('price_basis'):
            raise ValueError('fresh raw input price-basis identity missing')
    price_bases=sorted({record['raw_attrs']['price_basis'] for record in records})
    source={'source_native_run_id':source_native_run_id,'effective_data_end':end,'selected_symbols':[r['symbol'] for r in records],
        'records_hash':stable_hash(records),'selected_count':len(records),'full_market_universe':False,
        'price_basis':price_bases[0] if len(price_bases)==1 else 'mixed_record_price_bases',
        'price_basis_counts':{basis:sum(record['raw_attrs']['price_basis']==basis for record in records) for basis in price_bases},
        'legacy_fuyao_bars':sum(record['raw_attrs'].get('legacy_fuyao_bars',0) for record in records),
        'unverified_trade_price_bars':sum(record['raw_attrs'].get('unverified_trade_price_bars',0) for record in records),
        'scope':'coverage_frozen_mainboard_cohort_not_return_selected','source_model_heads_imported':False,
        'population_mode':FULL_POPULATION_MODE,'liquidity':copy.deepcopy(liquidity),
        'source_report_sha256':_file_hash(ARTIFACT_RUNS_ROOT/source_native_run_id/'reports'/'native_source.json'),
        'raw_only_rebuild':True,'reused_model_or_label':False,'source_snapshot':old.get('source_snapshot',{})}
    source['snapshot_hash']=stable_hash(source);cfg['source_snapshot_hash']=source['snapshot_hash']
    calendar=old.get('calendar_dates')
    if not isinstance(calendar,list) or not calendar or calendar!=sorted(set(calendar)):
        raise ValueError('new raw input source requires its frozen verified ordered calendar')
    calendar=[d for d in calendar if d<=end]
    calendar_id=old.get('source_snapshot',{}).get('calendar_run_id')
    if not calendar or not calendar_id:
        raise ValueError('new raw input source lacks verified calendar provenance')
    cfg['pipeline_contract_hash']=stable_hash({'family':STRATEGY_ID,'source':source,'calendar_hash':stable_hash(calendar),'registered_config':cfg,
        'order':['independent_trend','policy_only_sources','E0_no_HR','MA_exit_E0','pinned_E1',
            'preregistered_actual_E1_holding_pairs','account_free_reference_quality_probes',
            'independent_reference_quality_heads_and_fee_qualification','strict_prior_L2_actual_flat_population_by_fold',
            'L2_entry_forced_E1_clone','L2_exit_preregistered_actual_E1_pairs'],
        'reference_population_protocol':POPULATION_VERSION,'self_training_population_feedback':False,
        'reassessment':cfg['reassessment'],'no_E2_entry_reuse':True})
    context=create_run_context(task=TRAIN_KIND,as_of_date=end,config=cfg,data_version=source['snapshot_hash'],
        seed=cfg['seed'],stocks=source['selected_symbols'],scope='train',start_date=cfg['feature_start'],end_date=end)
    print('TRAIN_RUN_ID '+context.run_id,flush=True)
    names=GENERATING_FILES
    code={n:_file_hash(MY_STRATEGY_ROOT/n) for n in names}
    _save(context.subdir('config')/'initial_config.json',cfg);_save(context.subdir('config')/'generating_code_hashes.json',code)
    for n in names:
        archived=context.subdir('config','source_code')/n;archived.parent.mkdir(parents=True,exist_ok=True)
        archived.write_bytes((MY_STRATEGY_ROOT/n).read_bytes())
    _save(context.subdir('data_snapshots')/'source.json',source);_save(context.subdir('data_snapshots')/'records.json',records)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'source_run_id':calendar_id,'dates':calendar})
    failures=[];unavailable=[];manifests=[]
    print('STAGE independent_three_class_trend',flush=True)
    observation_dir=context.subdir('dataset','trend','stocks')
    payloads=((r,calendar,cfg,str(observation_dir)) for r in records)
    observation_assets=list(_bounded_results(_trend_observation_worker,payloads,cpu_workers,cfg['stock_batch_size']))
    contract=observation_assets[0]['label_contract']
    if any(asset['label_contract']!=contract for asset in observation_assets):
        raise ValueError('per-stock trend population target contract differs')
    observations_path=context.subdir('dataset','trend')/'observations.parquet'
    _pool_parquets((asset['path'] for asset in observation_assets),observations_path,load=False)
    # The complete diagnostic fields remain persisted. Fits need only their
    # ordered numerical inputs, causal masks and economic observation identity.
    from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS
    fit_columns=[*MA_TREND_COLUMNS,'symbol','date','label_end','label','input_eligible','label_available','observation_id']
    trenddata=pd.read_parquet(observations_path,columns=fit_columns)
    trenddata.attrs['label_contract']=contract
    _save(context.subdir('dataset','trend')/'manifest.json',{'source_snapshot_hash':source['snapshot_hash'],
        'population_mode':FULL_POPULATION_MODE,'records':observation_assets,'failures':[]})
    folds=chronological_folds(end,calendar);trends=[]
    holding_policy=[]
    for fold in folds:
        try:
            path=context.subdir('models','trend',fold['name']);m=train_trend_head(trenddata,path,fold,cfg,contract=contract)
            bundle={'name':fold['name'],'fold':fold,'available_at':fold['policy_end'],
                'heads':{'trend_direction':{'path':str(path.relative_to(context.run_dir())),'manifest_sha256':m['manifest_sha256']}}}
            _emit_head_resource(m,fold=fold,head='trend_direction',architecture='logistic_three_class')
            trends.append(bundle);manifests.append(m);print('TREND_HEAD '+fold['name']+' '+str(m['sample_counts']),flush=True)
        except ValueError as exc:
            if 'insufficient mature' not in str(exc):raise
            unavailable.append({'head':'trend_direction','fold':fold['name'],'reason':str(exc)})
    _save(context.subdir('models')/'trend_bundles.json',trends)
    del trenddata
    print('STAGE independent_policy_selection',flush=True)
    selection=select_source_policy(records,calendar,cfg,context.run_dir(),trends,context,cpu_workers) if trends else {
        'phase':'candidate_and_ordinary_SELL_policy_only','status':'no_qualified_trend_head_use_preregistered_baseline',
        'selected':None,'variants':[],'four_development_windows_used':False,'high_return_filter_used':False}
    if not trends:_save(context.subdir('validation')/'source_policy_selection.json',selection)
    cfg['candidate_policy_hash']=stable_hash({'candidate':cfg['candidate_policy'],'risk_policy':cfg['risk_policy'],
        'trend_candidate':{k:cfg['trend'][k] for k in ('candidate_cross_threshold','candidate_margin')},
        'consensus':cfg['consensus'],'signal_points':cfg['signal_points'],'native_records':source['records_hash'],
        'reassessment':cfg['reassessment'],'training_population':POPULATION_VERSION})
    cfg['E0_policy_hash']=stable_hash({'stage':'E0','candidate_policy_hash':cfg['candidate_policy_hash'],'account':cfg['account'],
        'trend_catalog_hash':stable_hash(trends),'ordinary_sell':'continuous_sell_pass_for_actual_holding',
        'reassessment':cfg['reassessment'],'entry_HR':False})
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
    del baseline,ma_data
    cfg['E1_policy_hash']=stable_hash({'stage':'E1','E0':cfg['E0_policy_hash'],'signal_time_pinned_MA_exit':stable_hash(ma_bundles),
        'holding_threshold':cfg['account']['holding_exit_threshold'],'no_head':'entire_position_E0','checkpoint_upgrade':False,'L2_exit_apply':False})
    print('STAGE E1_pinned_exit_rebuild_entry_and_shadow_exit_labels',flush=True)
    final,final_records=build_stage(records,calendar,cfg,context,'E1',trends,ma_bundles,cpu_workers)
    holding_population=freeze_holding_population(final,context.run_dir(),cfg,records)
    models=[];architecture_bundles=[];population_folds=[];population_totals=dict.fromkeys([
        'entry_rows','exit_rows','entries_mature_at_frozen_source','entries_mature_by_population_end',
        'entries_positive_at_frozen_source','entries_positive_by_population_end',
        'exits_mature_at_frozen_source','exits_mature_by_population_end'],0)
    from my_strategy.services.czsc_trend_interval_model import split_validation
    l2_folds=[split_validation(fold,calendar) for fold in [*cfg['development_windows'],cfg['production']]]
    l2_folds=sorted(l2_folds,key=lambda fold:(fold['policy_end'],fold['name']))
    held_diagnostics=holding_population_diagnostics(final,l2_folds)
    _save(context.subdir('validation')/'holding_population.json',held_diagnostics)
    print('HELD_POPULATION '+json.dumps(held_diagnostics,ensure_ascii=False,separators=(',',':')),flush=True)
    from my_strategy.services.czsc_reference_quality import run_reference_quality
    reference_quality=run_reference_quality(records,calendar,cfg,context,trends,ma_bundles,cpu_workers,folds)
    print('STAGE sequential_L2_actual_flat_pre_gate_population_and_forced_E1_labels',flush=True)
    for fold in l2_folds:
        population=make_reference_population(fold,models,cfg['reassessment'])
        subdirectory='final_population/'+fold['name']
        print('STAGE L2_population '+fold['name']+' prior_entry_heads='+str(len(models)),flush=True)
        fold_data,fold_records=build_stage(records,calendar,cfg,context,'E1',trends,ma_bundles,cpu_workers,
            reference_population=population,population_end=fold['policy_end'],subdirectory=subdirectory)
        manifest_path=context.subdir('dataset',subdirectory)/'manifest.json'
        population_record={'fold':fold['name'],'reference_population':population,
            'manifest_path':str(manifest_path.relative_to(context.run_dir())),'manifest_sha256':_file_hash(manifest_path),
            'entry_rows':len(fold_data['entries']),'exit_rows':len(fold_data['exits']),
            **label_maturity_counts(fold_data,source_end=end,population_end=fold['policy_end']),
            'frozen_source_end':end,'effective_population_end':fold['policy_end'],
            'label_counts_scope':'actual_liquidation_cutoff_no_input_or_split_eligibility_inference',
            'reference_account_ledgers':len(fold_records),'own_head_used_for_population':False,
            'holding_population_hash':stable_hash(holding_population),
            'holding_exit_sample_counts':next(item['heads'] for item in held_diagnostics['folds'] if item['fold']==fold)}
        population_folds.append(population_record)
        for key in population_totals:population_totals[key]+=population_record[key]
        fitted={}
        for head in ('entry','exit','exit_advantage'):
            data=training_frame(fold_data if head in ENTRY_HEADS else final,head,cfg,calendar,'E1')
            for architecture in cfg['architectures']:
                try:
                    b,m=_fit_interval(data,head,fold,calendar,context,cfg,architecture,device);manifests.append(m)
                    if architecture not in fitted:fitted[architecture]=b
                    else:fitted[architecture]['heads'].update(b['heads'])
                    print('L2_HEAD '+fold['name']+' '+architecture+' '+head+' '+str(m['sample_counts']),flush=True)
                except ValueError as exc:
                    if not ('insufficient mature' in str(exc) or 'require both classes' in str(exc)):raise
                    unavailable.append({'head':head,'fold':fold['name'],'architecture':architecture,'reason':str(exc)})
            del data
        for architecture in cfg['architectures']:
            bundle=fitted.get(architecture)
            if bundle:
                architecture_bundles.append(bundle)
                if architecture=='linear' and bundle['heads'].get('entry'):models.append(bundle)
        _save(context.subdir('validation')/'population_folds.json',population_folds)
        _save(context.subdir('models')/'l2_bundles.json',models)
        del fold_data
    del final
    dataset_manifests=[]
    for relative in ['E0','E1',*['final_population/'+item['fold'] for item in population_folds]]:
        path=context.subdir('dataset',relative)/'manifest.json'
        dataset_manifests.append({'stage':relative,'path':str(path.relative_to(context.run_dir())),'sha256':_file_hash(path)})
    if any(_file_hash(MY_STRATEGY_ROOT/p)!=sha for p,sha in code.items()):raise ValueError('generating code changed during frozen training')
    verify_source_generators(source['source_snapshot'])
    freeze={'family':STRATEGY_ID,'actual_freeze_at':local_now().isoformat(),'config_hash':stable_hash(cfg),
        'pipeline_contract_hash':cfg['pipeline_contract_hash'],'source_snapshot_hash':source['snapshot_hash'],'code_hashes':code,
        'models_hash':stable_hash([m['manifest_sha256'] for m in manifests]),
        'models_catalog_hash':stable_hash({'trend':trends,'ma_exit':ma_bundles,'architecture':architecture_bundles}),
        'population_folds_hash':stable_hash(population_folds),
        'holding_population_hash':stable_hash(holding_population),
        'holding_population_diagnostics_hash':stable_hash(held_diagnostics),
        'reference_quality_hash':stable_hash(reference_quality),
        'dataset_manifests_hash':stable_hash(dataset_manifests),
        'independent_window_status':'pending',
        'historical_available_is_simulated':True,'actual_completion_not_backdated':True}
    freeze['freeze_hash']=stable_hash(freeze)
    pipeline={'stages':['independent_trend','source_policy','E0','MA_exit','E1','preregistered_actual_E1_holding_pairs',
        'account_free_reference_quality_probes','independent_reference_quality_heads_and_fee_qualification',
        'strict_prior_L2_population_by_fold','L2_entry_E1','L2_exit_preregistered_actual_E1_pairs'],
        'complete_model_population_DAG':True,
        'trend_heads':len(trends),'ma_exit_heads':len(ma_bundles),'entry_heads':len(models),
        'holding_checkpoint_upgrade':False,'bootstrap_uses_HR':False,'applied_L2_exit':False,
        **{'final_'+key:value for key,value in population_totals.items()},
        'population_counts_scope':'sum_of_fold_specific_populations_not_unique_examples_actual_liquidation_cutoffs_not_training_masks',
        'reference_population_protocol':POPULATION_VERSION,'entry_attempts_recorded_before_gate':True,
        'holding_training_population':HELD_TRAINING_POPULATION,'holding_population_hash':stable_hash(holding_population),
        'holding_exit_rows':held_diagnostics['rows'],'holding_exit_mature_rows':held_diagnostics['mature_rows'],
        'holding_exit_mature_classes':held_diagnostics['mature_classes'],
        'holding_exit_fold_counts_are_additive':False,
        'reference_quality_status':reference_quality['status'],
        'reference_quality_required_for_core_selection':False,
        'L2_exit_final_gated_account_validation':'requires_actual_fee_net_final_account_trades_not_E1_sample_count',
        'self_head_controls_its_training_population':False,'label_clones_and_reference_actual_trades_separate':True}
    pipeline['population_mode']=FULL_POPULATION_MODE
    held_head_status=holding_head_status(architecture_bundles,unavailable,cfg['production']['name'])
    pipeline['holding_head_status']=held_head_status
    pipeline['full_qualified_pool_completed']=True
    pipeline['actual_training_symbols']=len(records)
    pipeline['resources']={'stock_batch_size':cfg['stock_batch_size'],
        'max_preparation_inflight':max(1,min(2*cpu_workers,cfg['stock_batch_size'])),
        'pooling':'streamed_per_stock_parquet_no_list_concat','cpu_workers':cpu_workers,
        'neural_epochs':cfg['epochs'],'neural_batch_size':cfg['batch_size'],'neural_device':device,
        'trend_fit_device':'cpu','trend_fit_max_iter':500,'tree_max_iter':100,
        'warmup':'complete_frozen_raw_prefix_not_truncated'}
    required=required_head_status(folds,trends,ma_bundles,models,cfg['production']['name'])
    ready=all(required[key] for key in ('recent_trend_ready','same_checkpoint_ma_exit_ready','production_entry_ready'))
    report={'run_id':context.run_id,'status':'completed','strategy_version':STRATEGY_ID,'config':cfg,'source_snapshot':source,
        'data_end':end,'dataset_records':records,'trend_bundles':trends,'ma_exit_bundles':ma_bundles,'model_bundles':models,
        'architecture_bundles':architecture_bundles,'source_policy_selection':selection,'holding_exit_policy_selection':holding_policy,'continuation_stage':'E1','pipeline':pipeline,
        'population_folds':population_folds,'holding_population':holding_population,
        'holding_population_diagnostics':held_diagnostics,'holding_head_status':held_head_status,
        'reference_quality':reference_quality,
        'dataset_manifests':dataset_manifests,
        'unavailable_heads':unavailable,'failures':failures,'protocol_freeze':freeze,'completed_at':local_now().isoformat(),
        'required_recent_heads_ready':ready,'required_head_status':required,
        'automatic_selection_eligible':len(records)>=60 and cfg['full_qualified_pool'] is True and cfg['trend']['horizon_sessions']==10 and cfg['risk_policy']=='confirmed_source_boundary_prior_atr' and ready,
        'architecture_selection':'preregistered_linear_main_tree_mlp_controlled_ablations',
        'validation_records':old['dataset_records'],
        'model_gate':{'passed':False,'reasons':['future_independent_validation_pending','historical_listing_ST_delisting_identity_limits',
            'development_fee_net_Broker_validation_pending','unadjusted_price_and_historical_ST_identity_limits']},'publication_allowed':False}
    _save(context.subdir('config')/'research.json',cfg);_save(context.subdir('validation')/'protocol_freeze.json',freeze)
    _save(context.subdir('reports')/'trend_consensus.json',report);context.write_metadata({'status':'completed','pipeline':pipeline,'failures':[]})
    print('COMPLETED '+context.run_id,flush=True);return report


def train_trend_consensus(*,source_native_run_id,
                          symbols=None,end='2026-10-08',config=None,cpu_workers=6,device='cpu'):
    """Train the full frozen historical eligible union exactly once.

    Processing batches bound resources only. A small ready checkpoint cannot
    certify a larger untrained source catalogue, and explicit subsets fail.
    """
    cfg=validate_training_config(config)
    if type(cpu_workers) is not int or cpu_workers < 1:
        raise ValueError('CPU preparation workers must be a positive integer')
    source=json.loads((ARTIFACT_RUNS_ROOT/source_native_run_id/'reports/native_source.json').read_text())
    if (source.get('strategy_version')!=STRATEGY_ID or source.get('status')!='completed' or source.get('failures')
            or source.get('raw_only_rebuild') is not True):
        raise ValueError('full training requires the new raw-rebuilt current input population')
    records=source['dataset_records']
    verify_full_population(source,cfg,records)
    actual_symbols=[record['symbol'] for record in records]
    if symbols is not None and (len(symbols)!=len(actual_symbols) or set(symbols)!=set(actual_symbols)):
        raise ValueError('explicit cohort must equal the complete frozen historical liquidity union')
    cfg['training_capacity_tiers']=[len(records)]
    cfg['capacity_expansion_criterion']=FULL_CAPACITY_CRITERION
    report=_train_one(source_native_run_id=source_native_run_id,symbols=actual_symbols,end=end,config=cfg,
        cpu_workers=cpu_workers,device=device,native_records=records)
    report['capacity_attempts']=[{'run_id':report['run_id'],'tier':len(records),
        'status':report.get('status','completed'),'required_recent_heads_ready':report['required_recent_heads_ready'],
        'required_head_status':report.get('required_head_status',{}),'unavailable_heads':report['unavailable_heads'],
        'protocol_freeze_hash':(report.get('protocol_freeze') or {}).get('freeze_hash')}]
    report['capacity_status']='full_population_completed' if report.get('status')=='completed' else 'full_population_incomplete'
    _save(ARTIFACT_RUNS_ROOT/report['run_id']/'validation/capacity_attempts.json',{
        'source_native_run_id':source_native_run_id,'preregistered_tiers':[len(records)],
        'population_mode':FULL_POPULATION_MODE,'processing_batch_size':cfg['stock_batch_size'],
        'readiness_early_stop':False,'status':report['capacity_status'],'attempts':report['capacity_attempts']})
    _save(ARTIFACT_RUNS_ROOT/report['run_id']/'reports/trend_consensus.json',report)
    return report
