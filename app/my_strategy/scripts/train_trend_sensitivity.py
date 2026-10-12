#!/usr/bin/env python3
"""Preregistered H20 direction sensitivity on a frozen H10 trend source pool.

This control trains new labels and heads. It cannot select an architecture,
change E1 continuation, supply production forecasts, or replace the H10 run.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_research import _save, _file_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID
from my_strategy.services.czsc_trend_model import (
    build_trend_labels, chronological_folds, load_trend_head, train_trend_head,
    trend_time_masks, trend_config,
)
from my_strategy.services.czsc_trend_pipeline import read_run, read_snapshot


def population_counts(observations):
    available=observations.input_eligible.fillna(False) & observations.label_available.fillna(False)
    return {
        'rows':len(observations),
        'input_eligible':int(observations.input_eligible.fillna(False).sum()),
        'mature':int(available.sum()),
        'classes':{k:int(v) for k,v in observations.loc[available,'label_class'].value_counts().items()},
        'unavailable_reasons':{k:int(v) for k,v in observations.loc[~available,'label_reason'].value_counts().items()},
    }


def fold_counts(observations, fold):
    masks=trend_time_masks(observations,fold)
    return {name:{'rows':int(mask.sum()),'classes':{label:int((observations.loc[mask,'label_class']==label).sum())
        for label in ('UP','DOWN','NEUTRAL')}} for name,mask in zip(('train','calibration','policy'),masks)}


def temporal_coverage(observations, fold):
    result={}
    for name,mask in zip(('train','calibration','policy'),trend_time_masks(observations,fold)):
        rows=observations.loc[mask]
        intervals=rows[['date','label_end']].drop_duplicates().sort_values(['label_end','date'])
        previous_end='';disjoint=0
        for row in intervals.itertuples(index=False):
            if row.date>previous_end:disjoint+=1;previous_end=row.label_end
        result[name]={'symbols':int(rows.symbol.nunique()),'signal_dates':int(rows.date.nunique()),
            'label_end_dates':int(rows.label_end.nunique()),'greedy_disjoint_calendar_intervals':disjoint,
            'interpretation':'temporal_coverage_not_independent_statistical_effective_sample_size'}
    return result


def read_source(model_run_id, *, allow_trend_stage_source=False):
    """An explicit stage control never pretends its parent pipeline is complete."""
    if not allow_trend_stage_source:
        root, report = read_run(model_run_id)
        return root, report, {'main_status': 'completed', 'stage_source': False}
    root = (ARTIFACT_RUNS_ROOT / str(model_run_id)).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):
        raise ValueError('sensitivity source outside artifact root')
    if (root / 'reports/trend_consensus.json').exists():
        directory, report = read_run(model_run_id)
        return directory, report, {'main_status': 'completed', 'stage_source': False}
    files = ['config/initial_config.json', 'config/generating_code_hashes.json',
             'data_snapshots/source.json', 'data_snapshots/records.json',
             'data_snapshots/calendar.json', 'models/trend_bundles.json',
             'dataset/trend/observations.parquet']
    assets = {name: _file_hash(root / name) for name in files}
    def read(name):
        return json.loads((root / name).read_text())
    cfg = read('config/initial_config.json')
    source = read('data_snapshots/source.json')
    records = read('data_snapshots/records.json')
    calendar = read('data_snapshots/calendar.json')
    code = read('config/generating_code_hashes.json')
    bundles = read('models/trend_bundles.json')
    if cfg.get('version') != STRATEGY_ID or trend_config(cfg)['horizon_sessions'] != 10:
        raise ValueError('stage sensitivity requires the new H10 family')
    if len(code) != 9 or any(_file_hash(Path(p)) != sha for p, sha in code.items()):
        raise ValueError('stage source nine-generator freeze changed')
    if source['snapshot_hash'] != stable_hash({k:v for k,v in source.items() if k != 'snapshot_hash'}):
        raise ValueError('stage source snapshot identity invalid')
    if stable_hash(records) != source['records_hash'] or source['snapshot_hash'] != cfg['source_snapshot_hash']:
        raise ValueError('stage source record catalog binding invalid')
    dates = calendar['dates']
    if calendar.get('verified') is not True or dates != sorted(set(dates)) or dates[-1] != source['effective_data_end']:
        raise ValueError('stage source requires the full verified historical calendar')
    folds = chronological_folds(source['effective_data_end'], dates)
    if [b['fold'] for b in bundles] != folds:
        raise ValueError('stage source requires every completed chronological trend head')
    for bundle in bundles:
        item = bundle['heads']['trend_direction']
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('stage source trend head outside parent run')
        manifest = load_trend_head(path)
        target = manifest['target_contract']
        if manifest['manifest_sha256'] != item['manifest_sha256'] or manifest['fold'] != bundle['fold']:
            raise ValueError('stage source trend catalog/checkpoint changed')
        if target['parameters'] != trend_config(cfg) or target['source_snapshot_hash'] != source['snapshot_hash'] or target['pipeline_contract_hash'] != cfg['pipeline_contract_hash']:
            raise ValueError('stage source trend target belongs to another input/protocol')
        if bundle['available_at'] != manifest['fold']['policy_end'] or not manifest.get('actual_training_completed_at'):
            raise ValueError('stage source trend head is incomplete or availability backdated')
    labels = pd.read_parquet(root / 'dataset/trend/observations.parquet',
        columns=['symbol', 'horizon_sessions'])
    label_contract = labels.attrs.get('label_contract', {})
    if label_contract.get('contract_hash') != stable_hash({k:v for k,v in label_contract.items() if k != 'contract_hash'}):
        raise ValueError('stage H10 observation contract identity invalid')
    if set(labels.symbol) != {r['symbol'] for r in records} or set(labels.horizon_sessions) != {10} or label_contract.get('source_snapshot_hash') != source['snapshot_hash'] or label_contract.get('pipeline_contract_hash') != cfg['pipeline_contract_hash']:
        raise ValueError('stage H10 observations do not match the frozen source')
    report = {'config': cfg, 'dataset_records': records, 'source_snapshot': source,
              'data_end': source['effective_data_end'], 'trend_bundles': bundles}
    audit = {'main_status': 'in_progress_trend_stage_frozen', 'stage_source': True,
        'main_pipeline_completed': False, 'main_completion_verification': 'pending',
        'source_assets': assets, 'nine_generator_code_hashes': code,
        'completed_trend_catalog_hash': stable_hash(bundles),
        'parent_policy_or_E0_E1_L2_parameters_used': False}
    return root, report, audit


def train_sensitivity(model_run_id, *, allow_trend_stage_source=False):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    threadpool_limits(limits=2)
    source_root,main,source_audit=read_source(model_run_id,
        allow_trend_stage_source=allow_trend_stage_source)
    if main['config']['trend']['horizon_sessions']!=10:
        raise ValueError('sensitivity requires an immutable H10 trend source')
    # Cross/ordinary-S/account/HR parameters have no role in this target.
    cfg={'feature_start':main['config']['feature_start'], 'seed':main['config']['seed'],
        'trend':trend_config(main['config']),
        'source_snapshot_hash':main['source_snapshot']['snapshot_hash']}
    cfg['trend']['horizon_sessions']=20
    calendar=json.loads((source_root/'data_snapshots/calendar.json').read_text())['dates']
    records=main['dataset_records']
    protocol={
        'family':STRATEGY_ID,'experiment':'preregistered_H20_direction_sensitivity',
        'main_H10_run_id':source_root.name,'source_snapshot_hash':main['source_snapshot']['snapshot_hash'],
        'records_hash':stable_hash(records),'calendar_hash':stable_hash(calendar),
        'horizon_sessions':20,'config':copy.deepcopy(cfg),'no_E0_E1_or_L2_reuse':True,
        'main_H10_pipeline_hash_provenance_only':main['config']['pipeline_contract_hash'],
        'parent_policy_or_E0_E1_L2_parameters_used':False,
        'selection_uses_development_returns':False,'automatic_selection_eligible':False,
    }
    protocol_hash=stable_hash(protocol)
    cfg['pipeline_contract_hash']=protocol_hash
    context=create_run_context(task='czsc-trend-horizon-sensitivity',as_of_date=main['data_end'],
        config=protocol,data_version=main['source_snapshot']['snapshot_hash'],scope='train',
        stocks=[r['symbol'] for r in records],start_date=cfg['feature_start'],end_date=main['data_end'])
    root=context.run_dir();print('SENSITIVITY_RUN_ID '+context.run_id,flush=True)
    _save(context.subdir('config')/'sensitivity.json',cfg)
    _save(context.subdir('config')/'protocol.json',protocol)
    _save(context.subdir('validation')/'source_trend_stage_audit.json',source_audit)
    _save(context.subdir('data_snapshots')/'records.json',records)
    _save(context.subdir('data_snapshots')/'calendar.json',{'verified':True,'dates':calendar,
        'source_main_run_id':source_root.name})
    code={str(Path(__file__).resolve()):_file_hash(Path(__file__).resolve())}
    from my_strategy.services import czsc_trend_model,czsc_trend_contracts,czsc_research_ml
    for module in (czsc_trend_model,czsc_trend_contracts,czsc_research_ml):
        path=Path(module.__file__);code[str(path)]=_file_hash(path)
    frames=[]
    for record in records:
        frame,_=read_snapshot(record);frame=frame[frame.date<=main['data_end']].copy()
        labels=build_trend_labels(frame,calendar,cfg,
            source_snapshot_hash=main['source_snapshot']['snapshot_hash'],
            pipeline_contract_hash=cfg['pipeline_contract_hash'])
        frames.append(labels[labels.date>=cfg['feature_start']].copy())
    contract=frames[0].attrs['label_contract'];observations=pd.concat(frames,ignore_index=True)
    observations.attrs['label_contract']=contract
    observations.to_parquet(context.subdir('dataset')/'observations.parquet',index=False)
    h10=pd.read_parquet(source_root/'dataset/trend/observations.parquet')
    folds=chronological_folds(main['data_end'],calendar)
    if folds!=[b['fold'] for b in main['trend_bundles']]:
        # A main may legitimately lack a qualified early head. Preserve the
        # same chronological protocol; availability is reported separately.
        main_fold_names={b['name'] for b in main['trend_bundles']}
    else:main_fold_names={f['name'] for f in folds}
    heads=[];comparisons=[];unavailable=[]
    h10_bundles={b['name']:b for b in main['trend_bundles']}
    for fold in folds:
        counts=fold_counts(observations,fold);h10_metrics=None
        if fold['name'] in h10_bundles:
            source=load_trend_head(source_root/h10_bundles[fold['name']]['heads']['trend_direction']['path'])
            h10_metrics={k:source[k] for k in ('sample_counts','calibration_metrics','policy_metrics')}
        comparison={'fold':fold,'H10':h10_metrics,'H20_counts':counts,
            'temporal_coverage':{'H10':temporal_coverage(h10,fold),'H20':temporal_coverage(observations,fold)},
            'same_chronological_split':True,'different_target_horizons_not_winner_selection':True}
        try:
            path=context.subdir('models',fold['name'])
            manifest=train_trend_head(observations,path,fold,cfg,contract=contract)
            heads.append({'name':fold['name'],'path':str(path.relative_to(root)),
                'manifest_sha256':manifest['manifest_sha256'],'target_contract_hash':manifest['target_contract_hash']})
            comparison['H20']={k:manifest[k] for k in ('sample_counts','calibration_metrics','policy_metrics')}
            print('H20_HEAD '+fold['name']+' '+str(manifest['sample_counts']),flush=True)
        except ValueError as exc:
            if 'insufficient mature' not in str(exc):raise
            row={'fold':fold['name'],'reason':str(exc),'sample_counts':counts};unavailable.append(row)
            comparison['H20_status']='unavailable_mature_class_minimum_not_met'
            print('H20_UNAVAILABLE '+fold['name']+' '+str(counts),flush=True)
        comparisons.append(comparison)
    if any(_file_hash(Path(p))!=sha for p,sha in code.items()):
        raise ValueError('sensitivity generating source changed during training')
    completed=local_now().isoformat()
    report={'run_id':context.run_id,'strategy_version':STRATEGY_ID,'status':'completed',
        'experiment':'preregistered_H20_direction_sensitivity','main_H10_run_id':source_root.name,
        'source_snapshot':main['source_snapshot'],'data_end':main['data_end'],'selected_count':len(records),
        'main_source_audit':source_audit,'actual_cpu_thread_cap':2,
        'parent_policy_or_E0_E1_L2_parameters_used':False,
        'target_contract':contract,'config':cfg,'fold_count':len(folds),'trained_head_count':len(heads),
        'heads':heads,'unavailable_heads':unavailable,'failures':[],
        'coverage':{'H10':population_counts(h10),'H20':population_counts(observations)},
        'calibration_comparison':comparisons,'protocol_hash':protocol_hash,
        'code_hashes':code,'models_catalog_hash':stable_hash(heads),'actual_training_completed_at':completed,
        'automatic_selection_eligible':False,'publication_allowed':False,
        'E0_E1_L2_retrained':False,'H10_models_reused_for_H20':False,
        'selection_uses_development_returns':False,'independent_validation':'pending',
        'result_scope':'direction_target_sensitivity_only_not_Broker_profit_or_strategy_acceptance',
        'qualified_main_fold_names':sorted(main_fold_names)}
    _save(context.subdir('reports')/'trend_sensitivity.json',report)
    context.write_metadata({'status':'completed','trained_head_count':len(heads),'automatic_selection_eligible':False})
    print('SENSITIVITY_COMPLETED '+context.run_id,flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run-id',required=True)
    parser.add_argument('--allow-trend-stage-source',action='store_true',
        help='explicitly audit completed H10 heads while the parent E0/E1/L2 pipeline is still running')
    args=parser.parse_args();report=train_sensitivity(args.model_run_id,
        allow_trend_stage_source=args.allow_trend_stage_source)
    print(json.dumps({k:report[k] for k in ('run_id','trained_head_count','automatic_selection_eligible')},ensure_ascii=False))


if __name__=='__main__':main()
