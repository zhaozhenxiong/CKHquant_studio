"""Immutable independent signal -> actual episode -> forward interval ML runs.

Historical reconstruction is developmental. No alias or legacy fixed-horizon
model is replaced, and a completed training run never certifies production.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_research import _file_hash, _save, _json
from my_strategy.services.czsc_high_return_models import VERSION, schema_hash, split_validation, train_head
from my_strategy.services.czsc_high_return_features import (L1_MA_ENTRY_COLUMNS, L1_MA_EXIT_COLUMNS,
    ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS)
from my_strategy.storage.czsc_results import ResultStore


def research_config(config=None):
    value = json.loads((MY_STRATEGY_ROOT / 'configs/czsc_high_return_research.json').read_text())
    incoming = json.loads(Path(config).read_text()) if isinstance(config,(str,Path)) else config or {}
    for key,item in incoming.items():
        if isinstance(item,dict) and isinstance(value.get(key),dict): value[key].update(item)
        else: value[key] = item
    if incoming.get('version') == 'czsc_high_return_v1' and 'ma_entry_alignment' not in incoming:
        value.pop('ma_entry_alignment', None)
    from my_strategy.services.czsc_high_return_ma_training import validate_alignment_version
    validate_alignment_version(value)
    value['strategy'] = strategy_config(value.get('strategy'))
    value['publication_allowed'] = False
    value['observed_windows_diagnostic_only'] = True
    if value['architectures'] != ['linear','tree','mlp']:
        raise ValueError('frozen architecture comparison requires linear/tree/mlp')
    return value


def _mainboard(symbol):
    return symbol.startswith('60') and symbol.endswith('.SH') or symbol.startswith('00') and symbol.endswith('.SZ')


def _behavior_signature(config):
    return stable_hash({key:config[key] for key in ('strategy','signal_points','high_return','feature_start','initial_cash')})


def _source(source_training_run_id, end=None, symbols=None):
    root = (ARTIFACT_RUNS_ROOT / source_training_run_id).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()): raise ValueError('invalid source run')
    report_path = next((root / p for p in ('reports/czsc.json','reports/research.json') if (root/p).exists()),None)
    if report_path is None: raise ValueError('frozen source report missing')
    report = json.loads(report_path.read_text())
    original = report.get('dataset_records',[])
    if not original or len({r['symbol'] for r in original}) != len(original): raise ValueError('unique frozen stock records required')
    source_end = report.get('data_end') or max(r['data_end'] for r in original)
    effective = str(end or source_end)[:10]
    if effective > source_end: raise ValueError(f'frozen source ends {source_end}; create a new feature snapshot for later training dates')
    calendar_path = Path(report['calendar']['path']).resolve()
    if not calendar_path.is_relative_to(root): raise ValueError('calendar outside frozen source')
    expected = report['calendar'].get('file_sha256')
    if expected and _file_hash(calendar_path) != expected: raise ValueError('source calendar hash mismatch')
    calendar_data = json.loads(calendar_path.read_text())
    calendar = calendar_data.get('dates',[])
    if not calendar_data.get('verified') or calendar != sorted(set(calendar)) or effective not in calendar:
        raise ValueError('training end must be a verified source calendar session')
    requested = set(symbols or [r['symbol'] for r in original])
    if requested - {r['symbol'] for r in original}: raise ValueError('requested symbols not in frozen source universe')
    records = []
    for item in original:
        if item['symbol'] not in requested: continue
        path = Path(item['path']).resolve()
        if not path.is_relative_to(root): raise ValueError('source dataset outside frozen run')
        records.append({'symbol':item['symbol'],'path':str(path),'sha256':item['sha256'],
            'data_end':item['data_end'],'data_version':item.get('data_version')})
    snapshot = {'run_id':source_training_run_id,'report_path':str(report_path),'report_sha256':_file_hash(report_path),
        'calendar_run_id':report['calendar']['run_id'],
        'calendar_path':str(calendar_path),'calendar_sha256':_file_hash(calendar_path),'calendar_hash':stable_hash(calendar),
        'source_data_end':source_end,'effective_data_end':effective,'source_universe_count':len(original),
        'selected_count':len(records),'full_source_universe':len(records)==len(original),
        'records_hash':stable_hash(records),'immutable_no_source_rewrite':True}
    # calendar_run_id is a locator already bound inside report_sha256; the
    # canonical physical snapshot identity remains stable when adding metadata.
    snapshot['snapshot_hash'] = stable_hash({k:v for k,v in snapshot.items() if k!='calendar_run_id'})
    return records,calendar,snapshot,effective


def _read_features(record, config, end):
    path = Path(record['path'])
    if _file_hash(path) != record['sha256']: raise ValueError('frozen source feature hash mismatch')
    frame = pd.read_parquet(path)
    frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    if frame.empty or frame.symbol.astype(str).nunique()!=1 or str(frame.symbol.iloc[0])!=record['symbol'] or frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError('frozen source symbol/date identity invalid')
    frame = frame.loc[frame.date <= end].copy()
    # Event ages require the whole observed prefix. Account execution still
    # starts at feature_start, and candidate clones bound their own raw replay.
    frame = frame.reset_index(drop=True)
    if 'w_test' not in frame:
        from my_strategy.services.czsc_wyckoff_features import prepare_wyckoff_inputs
        frame = prepare_wyckoff_inputs(frame)
    # Old fixed-ten-session outcomes must not enter the new label generator.
    frame = frame.drop(columns=[c for c in ('label','label_end','label_reason','label_available',
        'net_return','entry_date','exit_date','entry_shares','fees') if c in frame])
    return frame


def _artifact_worker(payload):
    record,calendar,config,end,directory,stage,sources,bundles,model_root,snapshot_hash = payload
    from my_strategy.services.czsc_signal_points import build_signal_points
    from my_strategy.services.czsc_trade_episodes import build_episode_dataset
    frame = _read_features(record,config,end)
    if not set(frame.date).issubset(calendar): raise ValueError('source row is outside verified calendar')
    root = Path(directory)/record['symbol'].replace('.','_'); root.mkdir(parents=True,exist_ok=True)
    alignment_stage = stage == 'ma_entry_alignment'
    aligned = config.get('ma_entry_alignment') == 'signal_pinned_learned_exit_v2'
    lineage = None
    if alignment_stage and not (aligned and bundles and sources == ('ma',)):
        raise ValueError('MA entry target stage requires only its frozen MA exit catalog')
    if aligned and bundles:
        from my_strategy.services.czsc_high_return_ma_training import read_exit_bootstrap_lineage
        lineage = read_exit_bootstrap_lineage(model_root,calendar,config,bundles=bundles,
            expected_source_snapshot_hash=snapshot_hash,allow_training=True)
    if bundles and not alignment_stage:
        from my_strategy.services.czsc_high_return_runtime import prepare_points, make_ma_exit_policy
        points = prepare_points(frame,calendar,model_root=Path(model_root),bundles=bundles,config=config)
        exit_policy = make_ma_exit_policy(frame,calendar,Path(model_root),bundles,config=config,points=points,
            allow_training=True)
        policies,policy_hashes = {'ma':exit_policy},{'ma':exit_policy.recipe_hash if aligned else stable_hash(bundles)}
    else:
        points = build_signal_points(frame,config); policies,policy_hashes = None,None
    _save(root/'points.json',points)
    result = {'symbol':record['symbol'],'source_path':record['path'],'source_sha256':record['sha256'],
        'path':str(root),'points_path':str(root/'points.json'),'points_sha256':_file_hash(root/'points.json'),
        'events':len(points['events']),'candidates':len(points['candidates']),
        'source_rows':len(frame),'source_hash_verified':True,'tradable_board':_mainboard(record['symbol'])}
    if stage == 'points' or not result['tradable_board']:
        result['episode_status'] = 'not_requested' if stage=='points' else 'cash_only_unsupported_board'
        return result
    def save_account(source,symbol,account):
        account_root = root/'reference'/source; account_root.mkdir(parents=True,exist_ok=True)
        for key in ('daily','ledger','rejections','holding_snapshots'):
            _save(account_root/(key+'.json'),account.get(key,[]))
        _save(account_root/'summary.json',account['metrics'])
    if alignment_stage:
        from my_strategy.services.czsc_high_return_ma_alignment import build_exit_bound_ma_entry_dataset
        dataset = build_exit_bound_ma_entry_dataset(frame,calendar,model_root=model_root,
            bundles=lineage['exit_only_bundles'],config=config,source_snapshot_hash=snapshot_hash,
            start=config['feature_start'],points=points,reference_callback=save_account,
            expected_exit_contract_hash=lineage['expected_exit_target_contract_hash'])
    else:
        dataset = build_episode_dataset(frame,calendar,config,points=points,start=config['feature_start'],
            initial_cash=config['initial_cash'],source_snapshot_hash=snapshot_hash,sources=sources,
            reference_callback=save_account,exit_policies=policies,exit_policy_hashes=policy_hashes)
        if aligned and policies:
            from my_strategy.services.czsc_high_return_ma_alignment import bind_ma_exit_lineage
            bind_ma_exit_lineage(dataset,exit_policy)
    if aligned and (alignment_stage or policies):
        _save(root/'ma_exit_alignment_recipe.json',dataset['alignment_recipe'])
        _save(root/'ma_candidate_exit_bindings.json',dataset['candidate_exit_bindings'])
        result.update(ma_exit_alignment_recipe_path=str(root/'ma_exit_alignment_recipe.json'),
            ma_exit_alignment_recipe_sha256=_file_hash(root/'ma_exit_alignment_recipe.json'),
            ma_candidate_exit_bindings_path=str(root/'ma_candidate_exit_bindings.json'),
            ma_candidate_exit_bindings_sha256=_file_hash(root/'ma_candidate_exit_bindings.json'))
    for key in ('entries','exits','episodes'):
        path = root/(key+'.parquet')
        dataset[key].to_parquet(path,index=False)
        result[key+'_path'] = str(path); result[key+'_sha256'] = _file_hash(path)
    _save(root/'contracts.json',dataset['contracts'])
    result.update(coverage=dataset['coverage'],contracts_path=str(root/'contracts.json'),
        contracts_sha256=_file_hash(root/'contracts.json'),episode_status='completed')
    return result


def _build(records, calendar, config, end, context, stage, *, sources=('ma','structure','wyckoff','union'),
           bundles=None, subdirectory='final', cpu_workers=4, snapshot_hash='', progress=None, check_cancel=None):
    check = check_cancel or (lambda:None)
    target = context.subdir('dataset',subdirectory)
    payloads = [(r,calendar,config,end,str(target),stage,sources,bundles or [],str(context.run_dir()),snapshot_hash) for r in records]
    done,failures = [],[]; started = time.monotonic()
    def consume(record,item=None,error=None):
        if error: failures.append({'symbol':record['symbol'],'error':str(error)})
        else: done.append(item)
        if progress: progress(subdirectory+'_'+stage,len(done)+len(failures),len(records),len(failures))
        if (len(done)+len(failures))%25==0:
            _save(target/'progress.json',{'completed':len(done),'failed':len(failures),'requested':len(records),'seconds':time.monotonic()-started})
    if cpu_workers > 1:
        with ProcessPoolExecutor(max_workers=cpu_workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(_artifact_worker,p):p[0] for p in payloads}
            for future in as_completed(futures):
                check()
                try: consume(futures[future],future.result())
                except Exception as exc: consume(futures[future],error=exc)
    else:
        for payload in payloads:
            check()
            try: consume(payload[0],_artifact_worker(payload))
            except Exception as exc: consume(payload[0],error=exc)
    done.sort(key=lambda r:r['symbol'])
    result = {'records':done,'failures':failures,'requested':len(records),'successful':len(done),
        'supported_mainboard':sum(_mainboard(r['symbol']) for r in records),
        'hash_verified_count':sum(r.get('source_hash_verified',False) for r in done),
        'seconds':time.monotonic()-started,'stage':stage,'sources':list(sources)}
    _save(target/'manifest.json',result)
    return result


def _finish(context,report,kind):
    report = _json(report)
    report.update(run_id=context.run_id,model_family='high_return',strategy_version=report.get('config',{}).get('version',VERSION),
        publication_allowed=False,completed_at=local_now().isoformat(),run_context=context.to_dict())
    _save(context.subdir('reports')/'high_return.json',report)
    ResultStore().save(kind,report)
    context.write_metadata({'model_family':'high_return','status':'complete' if report.get('status','').startswith('completed') else report.get('status'),
        'config':report['config'],'model_gate':report.get('model_gate'),'publication_allowed':False})
    return report


def build_high_return_artifacts(*,source_training_run_id=None,end=None,symbols=None,cpu_workers=4,
        config=None,stage='episodes',progress=None,check_cancel=None,run_callback=None):
    if stage not in {'points','episodes'}: raise ValueError('unknown high-return data stage')
    cfg = research_config(config)
    records,calendar,snapshot,end = _source(source_training_run_id or cfg['source_training_run_id'],end,symbols)
    context = create_run_context(task='czsc-high-return-'+stage+'-build',as_of_date=end,config=cfg,
        data_version=snapshot['snapshot_hash'],stocks=[r['symbol'] for r in records],scope='full',start_date=cfg['feature_start'],end_date=end)
    if run_callback: run_callback(context.run_id)
    _save(context.subdir('data_snapshots')/'source.json',snapshot)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'dates':calendar,'source':snapshot})
    result = _build(records,calendar,cfg,end,context,stage,cpu_workers=cpu_workers,snapshot_hash=snapshot['snapshot_hash'],progress=progress,check_cancel=check_cancel)
    return _finish(context,{'status':'completed' if not result['failures'] else 'completed_with_failures',
        'config':cfg,'source_snapshot':snapshot,'data_end':end,'dataset_records':result['records'],
        'coverage':{k:v for k,v in result.items() if k not in {'records','failures'}},'failures':result['failures'],
        'label_phase':'causal_rule_v0_bootstrap','promotion_status':'research_only'},'czsc-high-return-'+stage+'-build')


def _load_training(artifacts,source,head,ma_only=False,source_snapshot_hash=None):
    from my_strategy.services.czsc_trade_episodes import training_frame
    chunks,contract = [],None
    key = 'exits' if head=='exit' else 'entries'
    for record in artifacts['records']:
        if record.get('episode_status') != 'completed': continue
        path = Path(record[key+'_path'])
        if _file_hash(path) != record[key+'_sha256']: raise ValueError('new episode artifact hash mismatch')
        frame = pd.read_parquet(path)
        contract_path=Path(record['contracts_path'])
        if not record.get('contracts_sha256') or _file_hash(contract_path)!=record['contracts_sha256']:
            raise ValueError('new target contract artifact hash mismatch')
        contracts = json.loads(contract_path.read_text())
        dataset = {key:frame,'contracts':contracts}
        selected = training_frame(dataset,source=source,head=head,ma_only=ma_only)
        current = selected.attrs['label_contract']
        if current['contract_hash']!=stable_hash({k:v for k,v in current.items() if k!='contract_hash'}):
            raise ValueError('dataset target contract self identity mismatch')
        expected=source_snapshot_hash or current['source_snapshot_hash']
        if current['source_snapshot_hash']!=expected:
            raise ValueError('dataset contract source snapshot mismatch')
        if len(selected) and (not selected.source_snapshot_hash.eq(expected).all() or
                not selected.continuation_policy_hash.eq(current['continuation_policy_hash']).all() or not selected.source.eq(source).all()):
            raise ValueError('mixed row source/continuation target contracts')
        if contract is not None and contract != current: raise ValueError('mixed training continuation contracts')
        contract = current
        if len(selected): chunks.append(selected)
    if not chunks: raise ValueError('no actual mature episode training inputs')
    data = pd.concat(chunks,ignore_index=True)
    data.attrs['label_contract'] = contract
    return data,contract


def _head_data(data,base_contract,head,columns):
    frame = data.copy()
    target = base_contract['target']
    if head=='return': frame['label']=frame.net_return; target='actual_complete_fee_net_roundtrip_return'
    if head=='loss': frame['label']=(frame.net_return<0).astype(float).where(frame.net_return.notna()); target='actual_complete_fee_net_roundtrip_loss'
    if head=='exit_advantage': frame['label']=frame.exit_advantage
    contract = {k:v for k,v in base_contract.items() if k!='contract_hash'}
    contract.update(model_head=head,model_input_schema_hash=schema_hash(columns,head),target=target,
        output_type='continuous' if head in {'return','exit_advantage'} else 'probability')
    contract['contract_hash'] = stable_hash(contract)
    frame.attrs['label_contract'] = contract
    return frame,contract


def _ma_folds(end):
    folds=[]
    for period in pd.period_range('2024Q1',pd.Timestamp(end).to_period('Q'),freq='Q'):
        validation=period-1; training=period-2
        folds.append({'name':'ma_'+str(period),'train_end':str(training.end_time.date()),
            'validation_start':str(validation.start_time.date()),'validation_end':str(validation.end_time.date())})
    return folds


def _train_pair(entry,exit,contracts,fold,context,cfg,*,architecture='linear',prefix='l2',device='mps',check_cancel=None,progress=None):
    heads={}; failures=[]
    jobs=[(('ma_entry' if prefix=='l1' else 'entry'),entry,L1_MA_ENTRY_COLUMNS if prefix=='l1' else ENTRY_FEATURE_COLUMNS),
          (('ma_exit' if prefix=='l1' else 'exit'),exit,L1_MA_EXIT_COLUMNS if prefix=='l1' else EXIT_FEATURE_COLUMNS)]
    if prefix=='l2': jobs.append(('exit_advantage',exit,EXIT_FEATURE_COLUMNS))
    for head,data,columns in jobs:
        frame,contract=_head_data(data,contracts['entry' if head.endswith('entry') else 'exit'],head,columns)
        path=context.subdir('models',prefix,fold['name'],architecture,head)
        try:
            manifest=train_head(frame,columns,path,fold,cfg,head=head,architecture=architecture,device=device,
                contract=contract,check_cancel=check_cancel,regression=head=='exit_advantage')
            heads[head]={'path':str(path.relative_to(context.run_dir())),'manifest_sha256':manifest['manifest_sha256'],
                'target_contract_hash':manifest['target_contract_hash'],'actual_device':manifest['actual_device'],
                'training_batches':manifest['training_batches'],'sample_counts':manifest['sample_counts']}
        except ValueError as exc:
            failures.append({'head':head,'fold':fold['name'],'architecture':architecture,'reason':str(exc)})
        if progress: progress('train_'+prefix+'_'+fold['name']+'_'+architecture+'_'+head,len(heads)+len(failures),len(jobs),len(failures))
    return {'name':fold['name'],'available_at':fold['validation_end'],'fold':fold,'architecture':architecture,
        'heads':heads,'policy':{'entry_threshold':cfg.get('default_entry_threshold',.1),'exit_threshold':.55,'max_holding_sessions':cfg['high_return']['max_holding_sessions']},
        'publication_allowed':False,'historical_reconstruction':True,'failures':failures}


def _patterns(entry, exits=None, artifacts=None, train_end='2026-06-30'):
    from my_strategy.services.czsc_high_return_models import time_masks
    # Freeze pattern discovery at the latest training cutoff, excluding calibration/test outcomes.
    subset=entry.loc[entry.label_available & (entry.label_end<=train_end) & (entry.date<=train_end)].copy()
    classes=subset.outcome_class.value_counts().to_dict()
    differences=[]
    high=subset.label.eq(1); other=subset.label.eq(0)
    if high.any() and other.any():
        for column in ENTRY_FEATURE_COLUMNS:
            values=pd.to_numeric(subset[column],errors='coerce')
            deviation=values.std()
            if pd.notna(deviation) and deviation>0:
                differences.append({'feature':column,'high_mean':values[high].mean(),'other_mean':values[other].mean(),
                    'standardized_difference':(values[high].mean()-values[other].mean())/deviation})
    holding_patterns=[]; held_count=0
    if exits is not None and artifacts is not None:
        episodes=[]
        for record in artifacts['records']:
            if record.get('episode_status')=='completed':
                eps=pd.read_parquet(record['episodes_path'])
                if eps.empty: continue
                episodes.append(eps.loc[(eps.source=='union') & eps.label_available & (eps.label_end<=train_end),
                    ['episode_id','outcome_class','label_end']])
        if episodes:
            identities=pd.concat(episodes,ignore_index=True).drop_duplicates('episode_id')
            held=exits.loc[exits.date<=train_end].merge(identities[['episode_id','outcome_class']],on='episode_id',how='inner')
            held_count=len(held); high=held.outcome_class.eq('high'); other=~high
            from my_strategy.services.czsc_high_return_features import HOLDING_COLUMNS
            if high.any() and other.any():
                for column in HOLDING_COLUMNS:
                    values=pd.to_numeric(held[column],errors='coerce'); deviation=values.std()
                    if pd.notna(deviation) and deviation>0:
                        holding_patterns.append({'feature':column,'high_mean':values[high].mean(),'other_mean':values[other].mean(),
                            'standardized_difference':(values[high].mean()-values[other].mean())/deviation})
    return {'cutoff':train_end,'rows':len(subset),'outcome_class_counts':classes,
        'features':sorted(differences,key=lambda x:abs(x['standardized_difference']),reverse=True),
        'actual_holding_pattern_rows':held_count,'actual_holding_prefix_patterns':holding_patterns,
        'interpretation':'training_only_association_not_causal_effect','future_outcomes_excluded_from_X':True}


def _selection_score(report, cfg):
    # Actual fee-inclusive validation Broker accounts, never classification accuracy.
    comparisons=report.get('windows',report.get('evaluation',[]))
    if isinstance(comparisons,dict): comparisons=list(comparisons.values())
    if not comparisons: return float('-inf')
    window=comparisons[0]; arms=window.get('arms',window.get('variants',window.get('accounts',{})))
    actual=arms.get('full',{}); reference=arms.get('reference',arms.get('union_reference',{}))
    metrics=actual.get('metrics',actual.get('aggregate',{}).get('metrics',{}))
    trips=actual.get('round_trips',actual.get('round_trip_statistics',metrics))
    reference_metrics=reference.get('metrics',reference.get('aggregate',{}).get('metrics',{}))
    if actual.get('cash_retained_failures') or reference.get('cash_retained_failures'): return float('-inf')
    coverage=metrics.get('candidate_coverage')
    if coverage is None or coverage<cfg['min_candidate_coverage']: return float('-inf')
    if trips.get('completed_round_trips',0)<cfg['min_validation_round_trips']: return float('-inf')
    if metrics.get('max_drawdown',1)>reference_metrics.get('max_drawdown',1)+cfg['drawdown_tolerance']: return float('-inf')
    if metrics.get('net_return',float('-inf')) <= reference_metrics.get('net_return',float('inf')):
        return float('-inf')
    return float(metrics.get('net_return',float('-inf')))


def _compact_evaluation(report):
    # Every original binding remains in the immutable standalone evaluation.json.
    # Avoid replicating hundreds of thousands of file references in one UI report.
    if isinstance(report,dict):
        return {k:_compact_evaluation(v) for k,v in report.items() if k!='ledger_bindings'}
    if isinstance(report,list): return [_compact_evaluation(v) for v in report]
    return report


def _code_files():
    paths=list((MY_STRATEGY_ROOT/'services').glob('czsc_high_return*.py'))
    paths += [MY_STRATEGY_ROOT/part for part in ('services/czsc_trade_episodes.py',
        'services/czsc_signal_points.py','services/czsc_backtest.py','configs/czsc_high_return_research.json',
        'adapters/czsc_adapter.py','services/czsc_analysis.py','services/czsc_research_profiles.py',
        'services/czsc_research_ml.py','services/czsc_wyckoff_features.py','services/czsc_research_calendar.py',
        'services/czsc_research_features.py','services/czsc_research_data.py','services/czsc_research.py',
        'execution/broker.py','execution/cost_model.py','execution/ledger.py','execution/order.py')]
    return sorted(set(paths))


def _verified_final_artifacts(run_id, *, records, calendar, snapshot, config, bundles, context=None,
        progress=None, check_cancel=None):
    """Reuse a complete immutable data stage, never a partial policy/training run."""
    import ast
    if not isinstance(run_id,str) or Path(run_id).name!=run_id or run_id in {'.','..'}:
        raise ValueError('invalid complete final artifact root')
    root=(ARTIFACT_RUNS_ROOT/run_id).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()) or root==ARTIFACT_RUNS_ROOT.resolve():
        raise ValueError('invalid complete final artifact root')
    saved_snapshot=json.loads((root/'data_snapshots/source.json').read_text())
    saved_records=json.loads((root/'data_snapshots/source_records.json').read_text())
    saved_calendar=json.loads((root/'data_snapshots/calendar.json').read_text())
    saved_config=json.loads((root/'config/research.json').read_text())
    if (saved_snapshot['snapshot_hash']!=snapshot['snapshot_hash'] or saved_records!=records
            or saved_calendar.get('verified') is not True or saved_calendar.get('dates')!=calendar
            or _behavior_signature(saved_config)!=_behavior_signature(config)):
        raise ValueError('complete final artifact source/calendar/behavior differs')
    old_bundles=json.loads((root/'models/ma_bundles.json').read_text())
    if not bundles or stable_hash(old_bundles)!=stable_hash(bundles):
        raise ValueError('complete final artifact MA provenance differs')
    old_code=json.loads((root/'config/initial_code_hashes.json').read_text())
    current={str(p):_file_hash(p) for p in _code_files()}
    if set(old_code)!=set(current): raise ValueError('complete final artifact dependency set differs')
    # Evaluation/audit do not generate points, account episodes or their targets.
    # Research orchestration may add this verifier; generating functions and all
    # module-level bindings must still exactly match its archived initial code.
    exempt={'czsc_high_return_research.py','czsc_high_return_evaluation.py','czsc_high_return_audit.py'}
    for name,sha in old_code.items():
        original=Path(name)
        archived=root/'config/source_code'/original.relative_to(MY_STRATEGY_ROOT)
        if _file_hash(archived)!=sha: raise ValueError('complete final artifact archived code hash mismatch')
        if original.name not in exempt and current[name]!=sha:
            raise ValueError('complete final artifact generating dependency changed')
    archived=root/'config/source_code/services/czsc_high_return_research.py'
    def generating_identity(path):
        tree=ast.parse(path.read_text())
        required={'_mainboard','_behavior_signature','_source','_read_features','_artifact_worker','_build'}
        found={node.name:ast.dump(node,include_attributes=False) for node in tree.body
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in required}
        if set(found)!=required: raise ValueError('complete final artifact generating functions missing')
        # Permit only the new orchestration/verifier itself to differ. Include
        # every other function/class binding so imported helpers cannot be
        # silently overridden by a newly added top-level declaration.
        globals_=[ast.dump(node,include_attributes=False) for node in tree.body
            if not (isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
                and node.name in {'train_high_return_research','_verified_final_artifacts'})]
        return stable_hash({'functions':found,'module_bindings':globals_})
    generation_hash=generating_identity(Path(__file__))
    if generating_identity(archived)!=generation_hash:
        raise ValueError('complete final artifact generating functions changed')
    manifest_path=root/'dataset/final/manifest.json'
    final=json.loads(manifest_path.read_text())
    requested={item['symbol']:item for item in records}
    cached=final.get('records',[])
    if (final.get('stage')!='episodes' or final.get('sources')!=['ma','structure','wyckoff','union']
            or final.get('failures') or final.get('requested')!=len(records)
            or final.get('successful')!=len(records) or final.get('hash_verified_count')!=len(records)
            or final.get('supported_mainboard')!=sum(_mainboard(s) for s in requested)
            or len(cached)!=len(records) or {r['symbol'] for r in cached}!=set(requested)):
        raise ValueError('complete final artifact coverage/policies incomplete')
    check=check_cancel or (lambda:None); verified=0
    for offset,record in enumerate(cached,1):
        check(); source=requested[record['symbol']]
        if (record.get('source_path')!=source['path'] or record.get('source_sha256')!=source['sha256']
                or record.get('source_hash_verified') is not True
                or record.get('tradable_board')!=_mainboard(record['symbol'])
                or _file_hash(Path(source['path']))!=source['sha256']):
            raise ValueError('complete final artifact source hash/identity mismatch')
        folder=(root/'dataset/final'/record['symbol'].replace('.','_')).resolve()
        if Path(record['path']).resolve()!=folder:
            raise ValueError('complete final artifact record outside original data stage')
        keys=['points']
        if _mainboard(record['symbol']):
            if record.get('episode_status')!='completed': raise ValueError('complete final artifact mainboard episodes incomplete')
            keys+=['entries','exits','episodes','contracts']
        elif record.get('episode_status')!='cash_only_unsupported_board':
            raise ValueError('complete final artifact unsupported board identity differs')
        for key in keys:
            path=Path(record[key+'_path']).resolve()
            suffix='.parquet' if key in {'entries','exits','episodes'} else '.json'
            if path!=folder/(key+suffix) or not path.is_relative_to(root) or _file_hash(path)!=record[key+'_sha256']:
                raise ValueError('complete final artifact file hash/path mismatch')
            verified+=1
        if progress: progress('verified_complete_final_artifact_reuse',offset,len(cached),0)
    verification={'source_run_id':run_id,
        'original_manifest_sha256':_file_hash(manifest_path),'complete_stage_only':True,
        'records_referenced_without_rewrite':True,'source_snapshot_hash':snapshot['snapshot_hash'],
        'ma_bundles_hash':stable_hash(bundles),'generating_research_identity':generation_hash,
        'bound_generating_code_hashes':{p:sha for p,sha in old_code.items() if Path(p).name not in exempt},
        'verified_source_files':len(records),'verified_artifact_files':verified}
    if context is None:
        return final,verification
    _save(context.subdir('dataset','final')/'manifest.json',final)
    _save(context.subdir('dataset','final')/'artifact_reuse.json',verification)
    return final


def train_high_return_research(*,source_training_run_id=None,end=None,symbols=None,device='mps',cpu_workers=4,
        config=None,progress=None,check_cancel=None,run_callback=None,bootstrap_artifact_run_id=None,
        ma_checkpoint_run_id=None,final_artifact_run_id=None):
    cfg=research_config(config); check=check_cancel or (lambda:None)
    if cfg.get('ma_entry_alignment') == 'signal_pinned_learned_exit_v2' and any(
            (bootstrap_artifact_run_id,ma_checkpoint_run_id,final_artifact_run_id)):
        raise ValueError('exit-bound L1 target version requires fresh bootstrap, exit heads, derived entry labels and final P2b; legacy caches cannot satisfy this dependency')
    records,calendar,snapshot,end=_source(source_training_run_id or cfg['source_training_run_id'],end,symbols)
    context=create_run_context(task='czsc-high-return-research-train',as_of_date=end,config=cfg,data_version=snapshot['snapshot_hash'],
        seed=cfg['seed'],stocks=[r['symbol'] for r in records],scope='train',start_date=cfg['feature_start'],end_date=end)
    if run_callback: run_callback(context.run_id)
    initial_code_hashes={str(p):_file_hash(p) for p in _code_files()}
    _save(context.subdir('config')/'initial_code_hashes.json',initial_code_hashes)
    for path in _code_files():
        destination=context.subdir('config','source_code')/path.relative_to(MY_STRATEGY_ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(path.read_bytes())
    _save(context.subdir('config')/'research.json',cfg)
    _save(context.subdir('data_snapshots')/'source.json',snapshot)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'dates':calendar,'source':snapshot})
    _save(context.subdir('data_snapshots')/'source_records.json',records)
    if bootstrap_artifact_run_id:
        cached=ARTIFACT_RUNS_ROOT/bootstrap_artifact_run_id
        saved=json.loads((cached/'data_snapshots/source.json').read_text())
        meta=json.loads((cached/'metadata.json').read_text())
        if saved['snapshot_hash']!=snapshot['snapshot_hash'] or _behavior_signature(meta['config'])!=_behavior_signature(cfg):
            raise ValueError('bootstrap cache source/config differs from frozen training protocol')
        bootstrap=json.loads((cached/'dataset/bootstrap/manifest.json').read_text())
        if bootstrap['failures'] or bootstrap['requested']!=len(records) or bootstrap['sources']!=['ma']:
            raise ValueError('bootstrap cache incomplete or wrong candidate policy')
    else:
        bootstrap=_build(records,calendar,cfg,end,context,'episodes',sources=('ma',),subdirectory='bootstrap',cpu_workers=cpu_workers,
            snapshot_hash=snapshot['snapshot_hash'],progress=progress,check_cancel=check)
    if bootstrap['failures']: raise RuntimeError('unresolved bootstrap data failures; research training stopped')
    ma_bundles=[]; l1_pipeline=None
    if cfg.get('ma_entry_alignment') == 'signal_pinned_learned_exit_v2':
        from my_strategy.services.czsc_high_return_ma_training import train_exit_bound_ma
        ma_bundles,l1_pipeline=train_exit_bound_ma(records,calendar,snapshot,end,bootstrap,context,cfg,
            device=device,cpu_workers=cpu_workers,progress=progress,check_cancel=check)
    elif ma_checkpoint_run_id:
        import shutil
        from my_strategy.services.czsc_high_return_models import load_head
        checkpoint_root=(ARTIFACT_RUNS_ROOT/ma_checkpoint_run_id).resolve()
        if not checkpoint_root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()): raise ValueError('invalid MA checkpoint root')
        checkpoint_source=json.loads((checkpoint_root/'data_snapshots/source.json').read_text())
        checkpoint_cfg=json.loads((checkpoint_root/'config/research.json').read_text())
        old_code=json.loads((checkpoint_root/'config/initial_code_hashes.json').read_text())
        reused_code_names={'czsc_high_return_models.py','czsc_high_return_features.py','czsc_high_return_labels.py',
            'czsc_trade_episodes.py','czsc_signal_points.py','czsc_backtest.py','czsc_high_return_runtime.py',
            'czsc_adapter.py','czsc_analysis.py','czsc_research_ml.py','czsc_research_profiles.py',
            'czsc_wyckoff_features.py','czsc_research_calendar.py','broker.py','cost_model.py','ledger.py','order.py'}
        bound_files=[(path,sha) for path,sha in old_code.items() if Path(path).name in reused_code_names]
        if {Path(p).name for p,s in bound_files}!=reused_code_names or any(_file_hash(Path(p))!=sha for p,sha in bound_files):
            raise ValueError('MA checkpoint generating features/targets/execution code changed')
        parameters=('seed','epochs','batch_size','min_train_rows','min_calibration_rows')
        if checkpoint_source['snapshot_hash']!=snapshot['snapshot_hash'] or _behavior_signature(checkpoint_cfg)!=_behavior_signature(cfg) or any(checkpoint_cfg[k]!=cfg[k] for k in parameters):
            raise ValueError('MA checkpoint source/behavior/training protocol differs')
        expected_folds=[split_validation(f,calendar) for f in _ma_folds(end)]
        ma_bundles=json.loads((checkpoint_root/'models/ma_bundles.json').read_text())
        if [b['fold'] for b in ma_bundles]!=expected_folds or any(set(b['heads'])!={'ma_entry','ma_exit'} or b.get('failures') for b in ma_bundles):
            raise ValueError('only a complete chronological MA checkpoint stage can be reused')
        reused=[]
        for bundle in ma_bundles:
            for head,binding in bundle['heads'].items():
                source=(checkpoint_root/binding['path']).resolve()
                if not source.is_relative_to(checkpoint_root): raise ValueError('MA head outside checkpoint root')
                manifest=load_head(source)
                if manifest['head']!=head or manifest['manifest_sha256']!=binding['manifest_sha256'] or manifest['target_contract']['source_snapshot_hash']!=snapshot['snapshot_hash'] or manifest['actual_device']!=device:
                    raise ValueError('MA checkpoint identity/device mismatch')
                target=context.run_dir()/binding['path'];target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(source,target)
                if load_head(target)!=manifest: raise ValueError('MA checkpoint copy changed immutable manifest')
                reused.append({'head':head,'path':binding['path'],'manifest_sha256':manifest['manifest_sha256'],'training_completed_at':manifest['training_completed_at']})
        _save(context.subdir('models')/'ma_checkpoint_reuse.json',{'source_run_id':ma_checkpoint_run_id,'byte_identical':True,
            'bound_code_hashes':dict(bound_files),'bootstrap_manifest_sha256':_file_hash(cached/'dataset/bootstrap/manifest.json') if bootstrap_artifact_run_id else None,
            'heads':reused})
        if progress: progress('verified_MA_checkpoint_reuse',len(reused),len(reused),0)
    else:
        ma_entry,ma_entry_contract=_load_training(bootstrap,'ma','entry',True,source_snapshot_hash=snapshot['snapshot_hash'])
        ma_exit,ma_exit_contract=_load_training(bootstrap,'ma','exit',True,source_snapshot_hash=snapshot['snapshot_hash'])
        for raw_fold in _ma_folds(end):
            check(); fold=split_validation(raw_fold,calendar)
            bundle=_train_pair(ma_entry,ma_exit,{'entry':ma_entry_contract,'exit':ma_exit_contract},fold,context,cfg,
                prefix='l1',device=device,check_cancel=check,progress=progress)
            if bundle['heads']: ma_bundles.append(bundle)
        del ma_entry,ma_exit
    _save(context.subdir('models')/'ma_bundles.json',ma_bundles)
    if final_artifact_run_id:
        final=_verified_final_artifacts(final_artifact_run_id,records=records,calendar=calendar,
            snapshot=snapshot,config=cfg,bundles=ma_bundles,context=context,progress=progress,check_cancel=check)
    else:
        final=_build(records,calendar,cfg,end,context,'episodes',sources=('ma','structure','wyckoff','union'),bundles=ma_bundles,
            subdirectory='final',cpu_workers=cpu_workers,snapshot_hash=snapshot['snapshot_hash'],progress=progress,check_cancel=check)
    if final['failures']: raise RuntimeError('unresolved final episode data failures; research training stopped')
    entry,entry_contract=_load_training(final,'union','entry',source_snapshot_hash=snapshot['snapshot_hash'])
    exits,exit_contract=_load_training(final,'union','exit',source_snapshot_hash=snapshot['snapshot_hash'])
    patterns=_patterns(entry,exits,final,train_end=cfg['production']['train_end']); _save(context.subdir('reports')/'operation_patterns.json',patterns)
    from my_strategy.services.czsc_high_return_evaluation import evaluate_bundles, evaluate_policy_options
    evaluation_cfg={**cfg,'source_snapshot_hash':snapshot['snapshot_hash']}
    tradable=[r for r in records if _mainboard(r['symbol'])]
    selected=[]; diagnostic_bundles=[]; selection_reports=[]; skipped=[]
    for raw_fold in [*cfg['development_windows'],cfg['production']]:
        check(); fold=split_validation(raw_fold,calendar)
        candidates=[]; fitted=[]
        for architecture in cfg['architectures']:
            bundle=_train_pair(entry,exits,{'entry':entry_contract,'exit':exit_contract},fold,context,cfg,
                architecture=architecture,device=device,check_cancel=check,progress=progress)
            if not {'entry','exit','exit_advantage'} <= set(bundle['heads']):
                skipped.extend(bundle['failures']); continue
            matching=[b for b in ma_bundles if b['available_at']<=bundle['available_at']]
            bundle['heads'].update(matching[-1]['heads'] if matching else {})
            fitted.append(bundle)
        default_reports=evaluate_policy_options(tradable,calendar,model_root=context.run_dir(),ma_bundles=ma_bundles,
            options=fitted,config=evaluation_cfg,fold=fold,cpu_workers=cpu_workers,progress=progress,check_cancel=check,
            output_root=context.subdir('validation',fold['name'],'architecture_defaults'))
        for bundle,evaluated in zip(fitted,default_reports,strict=True):
            architecture=bundle['architecture']
            score=_selection_score(evaluated,cfg); candidates.append((score,bundle))
            selection_reports.append({'fold':fold['name'],'architecture':architecture,'policy':bundle['policy'],
                'score':score,'evidence':_compact_evaluation(evaluated),'stage':'architecture_at_frozen_default_threshold'})
        if not candidates:
            skipped.append({'fold':fold['name'],'reason':'no compatible mature two-class entry/exit heads'}); continue
        _,best=max(candidates,key=lambda item:item[0]); architecture=best['architecture']
        options=[]; policy_options=[]
        for entry_threshold in cfg['entry_threshold_grid']:
            for exit_threshold in cfg['exit_threshold_grid']:
                option=copy.deepcopy(best); option['policy'].update(entry_threshold=entry_threshold,exit_threshold=exit_threshold)
                policy_options.append(option)
        policy_reports=evaluate_policy_options(tradable,calendar,model_root=context.run_dir(),ma_bundles=ma_bundles,
            options=policy_options,config=evaluation_cfg,fold=fold,cpu_workers=cpu_workers,progress=progress,check_cancel=check,
            output_root=context.subdir('validation',fold['name'],architecture,'threshold_grid'))
        for option,evidence in zip(policy_options,policy_reports,strict=True):
            score=_selection_score(evidence,cfg); options.append((score,option))
            selection_reports.append({'fold':fold['name'],'architecture':architecture,'policy':option['policy'],
                'score':score,'evidence':_compact_evaluation(evidence),'stage':'threshold_grid_on_selected_architecture'})
        score,best=max(options,key=lambda item:item[0]); best['policy_selection_passed']=bool(np.isfinite(score))
        best['policy_selection_score']=score
        # Auxiliary heads explain/calibrate risk but cannot create independent buy opportunities.
        for head,data,base,columns in [('return',entry,entry_contract,ENTRY_FEATURE_COLUMNS),
                ('loss',entry,entry_contract,ENTRY_FEATURE_COLUMNS)]:
            frame,contract=_head_data(data,base,head,columns); path=context.subdir('models','l2',fold['name'],architecture,head)
            try:
                manifest=train_head(frame,columns,path,fold,cfg,head=head,architecture=architecture,regression=head!='loss',
                    device=device,contract=contract,check_cancel=check)
                best['heads'][head]={'path':str(path.relative_to(context.run_dir())), 'manifest_sha256':manifest['manifest_sha256'],
                    'actual_device':manifest['actual_device'],'training_batches':manifest['training_batches']}
            except ValueError as exc: skipped.append({'head':head,'fold':fold['name'],'reason':str(exc)})
        if best['policy_selection_passed']: selected.append(best)
        else: diagnostic_bundles.append(best)
        _save(context.subdir('models')/'selected_bundles.json',selected)
    del entry,exits
    bundles=sorted([*ma_bundles,*selected],key=lambda b:(b['available_at'],len(b['heads'])))
    evaluation=evaluate_bundles(records,calendar,model_root=context.run_dir(),bundles=bundles,config={**evaluation_cfg,'effective_end':end},
        windows=None,cpu_workers=cpu_workers,progress=progress,check_cancel=check,
        output_root=context.subdir('backtests','development'),allow_training=True)
    stress=[]
    for multiplier in (1.5,2.):
        stress.append(_compact_evaluation(evaluate_bundles(records,calendar,model_root=context.run_dir(),bundles=bundles,config=evaluation_cfg,
            windows=cfg['development_windows'],cpu_workers=cpu_workers,progress=progress,check_cancel=check,
            output_root=context.subdir('backtests',f'cost_x{multiplier}'),arms=['reference','full'],cost_multiplier=multiplier,
            allow_training=True)))
    freeze_time=local_now().isoformat(); freeze_date=freeze_time[:10]
    next_day=next((d for d in calendar if d>freeze_date),None)
    # Latest independently verified calendar may supply the next session; no invented future end.
    independent_calendar=None
    if next_day is None:
        from my_strategy.services.czsc_research_calendar import resolve_execution_calendar
        future,calendar_meta=resolve_execution_calendar(snapshot['calendar_run_id'],runs_root=ARTIFACT_RUNS_ROOT)
        next_day=next((d for d in future if d>freeze_date),None)
        independent_path=context.subdir('data_snapshots')/'independent-calendar.json'
        _save(independent_path,{'verified':True,'dates':future,'source':calendar_meta})
        independent_calendar={'path':str(independent_path),'sha256':_file_hash(independent_path),
            'dates_hash':stable_hash(future),'source':calendar_meta}
    final_code_hashes={str(p):_file_hash(p) for p in _code_files()}
    if final_code_hashes!=initial_code_hashes:
        raise RuntimeError('execution/feature/model source changed during training; final protocol cannot be frozen')
    from my_strategy.adapters.czsc_adapter import SOURCE_SHA256
    from my_strategy.services.czsc_high_return_forward_protocol import forward_protocol
    freeze={'actual_completed_at':freeze_time,'code_hashes':final_code_hashes,'native_source_sha256':SOURCE_SHA256,
        'config_hash':stable_hash(cfg),'source_snapshot_hash':snapshot['snapshot_hash'],'model_bundles_hash':stable_hash(bundles),
        'calendar_hash':stable_hash(calendar),'independent_calendar':independent_calendar,'forward_protocol':forward_protocol(cfg),
        'independent_start':next_day,'minimum_technical_sessions':480,'minimum_windows':3,'end':None,
        'qualification':'pending_future_verified_data','observed_windows_diagnostic_only':True}
    if l1_pipeline is not None:
        freeze.update(l1_ma_target_pipeline_hash=stable_hash(l1_pipeline),
            l1_exit_lineage_sha256=l1_pipeline['lineage_sha256'])
    freeze['freeze_hash']=stable_hash(freeze); _save(context.subdir('validation')/'protocol_freeze.json',freeze)
    return _finish(context,{'status':'completed' if not final['failures'] else 'completed_with_failures','config':cfg,
        'data_end':end,'source_snapshot':snapshot,'calendar':{'dates':calendar,'hash':stable_hash(calendar)},
        'dataset_records':records,'episode_records':final['records'],'bootstrap_coverage':{k:v for k,v in bootstrap.items() if k!='records'},
        'coverage':{k:v for k,v in final.items() if k!='records'},'training_rows':patterns['rows'],
        'model_bundles':bundles,'diagnostic_bundles':diagnostic_bundles,'l1_ma_target_pipeline':l1_pipeline,
        'selection':selection_reports,'selection_rule':'architecture_default_then_winner_threshold_grid_actual_broker',
        'skipped_heads':skipped,'evaluation':_compact_evaluation(evaluation),'cost_stress':stress,'operation_patterns_path':'reports/operation_patterns.json',
        'protocol_freeze':freeze,'independent_window':{**cfg['independent_window'],'start':next_day,'status':'pending_future_data'},
        'model_gate':{'pass':False,'reasons':['historical_windows_diagnostic_only','independent_future_window_not_complete',
            'unadjusted_price_basis_no_total_return_certification'],'research_ready':bool(selected)},
        'promotion_status':'shadow_only_no_alias','actual_execution':'BrokerSimulator_Ledger_next_verified_open'},'czsc-high-return-research-train')


def backtest_high_return(*,model_run_id,symbols=None,start=None,end=None,device='cpu',cpu_workers=4,
        config=None,progress=None,check_cancel=None,run_callback=None):
    model_root=(ARTIFACT_RUNS_ROOT/model_run_id).resolve()
    report=json.loads((model_root/'reports/high_return.json').read_text())
    if report.get('model_family')!='high_return': raise ValueError('explicit interval-family model run required')
    cfg=research_config(config or report['config'])
    selected_symbols=symbols or [r['symbol'] for r in report['dataset_records']]
    records,calendar,snapshot,effective_end=_source(report['source_snapshot']['run_id'],end or report['data_end'],selected_symbols)
    frozen={r['symbol']:r for r in report['dataset_records']}
    if snapshot['report_sha256']!=report['source_snapshot']['report_sha256'] or snapshot['calendar_sha256']!=report['source_snapshot']['calendar_sha256']:
        raise ValueError('model frozen source report/calendar changed')
    if any(r['symbol'] not in frozen or r['sha256']!=frozen[r['symbol']]['sha256'] or r['path']!=frozen[r['symbol']]['path'] for r in records):
        raise ValueError('backtest source differs from model frozen universe')
    context=create_run_context(task='czsc-high-return-backtest',as_of_date=effective_end,config=cfg,data_version=snapshot['snapshot_hash'],
        stocks=[r['symbol'] for r in records],scope='full',start_date=start or cfg['feature_start'],end_date=effective_end)
    if run_callback:run_callback(context.run_id)
    from my_strategy.services.czsc_high_return_evaluation import evaluate_bundles
    evaluated=evaluate_bundles(records,calendar,model_root=model_root,bundles=report['model_bundles'],
        config={**cfg,'source_snapshot_hash':report['source_snapshot']['snapshot_hash']},
        windows=[{'name':'explicit_offline','test_start':start or cfg['feature_start'],'test_end':effective_end}],cpu_workers=cpu_workers,
        progress=progress,check_cancel=check_cancel,output_root=context.subdir('backtests'))
    full=evaluated['variants']['full']; bindings=evaluated['windows']['explicit_offline']['ledger_bindings']['full']
    trades=[]
    for binding in bindings:
        ledger=pd.read_csv(Path(binding['path'])/'ledger.csv')
        trades.extend(ledger.to_dict('records'))
    trades.sort(key=lambda r:(r['date'],r['symbol']))
    return _finish(context,{'status':'completed','config':cfg,'model_run_id':model_run_id,'source_snapshot':snapshot,
        'data_end':effective_end,'evaluation':_compact_evaluation(evaluated),'metrics':full['metrics'],'daily':full['daily'],
        'initial_cash':full['metrics']['initial_cash'],'trades':trades[:10000],'trade_display_limit':10000,'trade_display_truncated':len(trades)>10000,
        'complete_ledger_evaluation_path':str(Path(evaluated['output_root'])/'evaluation.json'),
        'coverage':{'requested':len(bindings),'successful':len(bindings)-len(full['cash_retained_failures']),'failed':len(full['cash_retained_failures'])},
        'requested_device':device,'actual_inference_device':'cpu',
        'model_gate':{'pass':False,'reasons':['explicit_development_replay_not_production_certification']}},'czsc-high-return-backtest')


def status_high_return():
    from my_strategy.services.czsc_high_return_runtime import high_return_research_status
    runtime_status=high_return_research_status()
    runs=ResultStore().list('czsc-high-return-research-train',limit=30)
    result=[];forward_summary=None
    for item in runs:
        try:
            report=ResultStore().get(item['run_id'])
            if forward_summary is None and report.get('protocol_freeze',{}).get('forward_protocol'):
                from my_strategy.services.czsc_high_return_forward import forward_status
                try: forward_summary=forward_status(item['run_id'])
                except (ValueError,KeyError,OSError) as exc:
                    forward_summary={'model_run_id':item['run_id'],'status':'pending_artifact_validation',
                        'append_registry_count':0,'mature_technical_windows':0,'statistical_status':'pending_or_failed',
                        'reason':str(exc),'publication_allowed':False}
            result.append({**item,'model_family':'high_return','strategy_version':report.get('config',{}).get('version',VERSION),'data_end':report.get('data_end'),
                'model_bundles':[{'name':b['name'],'available_at':b['available_at'],'heads':list(b['heads'])} for b in report.get('model_bundles',[])],
                'model_gate':report.get('model_gate'),'independent_window':report.get('independent_window'),
                'promotion_status':report.get('promotion_status'),'publication_allowed':False})
        except (ValueError,KeyError,OSError): continue
    return {**runtime_status,'model_family':'high_return','strategy_version':research_config()['version'],'active_release':None,'publication_allowed':False,'runs':result,
        'target':'actual_complete_fee_net_roundtrip_high_return_and_initial_risk','exit_target':'actual_remaining_value_exit_advantage',
        'source_policy':'independent_ma_structure_wyckoff_points_union','production_status':'shadow_pending_future_certification',
        'forward_summary':forward_summary}
