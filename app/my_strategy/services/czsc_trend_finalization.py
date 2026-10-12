"""Recover audited completion metadata after the Path-only hash finalizer fault.

This module does not create labels, change targets, or refit any checkpoint.
All nine generating source hashes and every target/cfg/catalog are verified
against the original run before a completion report can be materialized.
"""
from pathlib import Path
import json
import pandas as pd
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_research import _save,_file_hash
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID
from my_strategy.services.czsc_trend_model import chronological_folds,load_trend_head
from my_strategy.services.czsc_trend_interval_model import load_head,time_masks,split_validation
from my_strategy.services.czsc_trend_dataset import training_frame


def finalize_existing_run(run_id):
    root=(ARTIFACT_RUNS_ROOT/run_id).resolve()
    if not root.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):raise ValueError('invalid completion root')
    target=root/'reports/trend_consensus.json'
    if target.exists():raise ValueError('completion/rejection report is immutable; do not overwrite')
    code=json.loads((root/'config/generating_code_hashes.json').read_text())
    if len(code)!=9 or any(_file_hash(Path(path))!=sha for path,sha in code.items()):raise ValueError('generating source changed; cannot recover this run')
    cfg=json.loads((root/'config/initial_config.json').read_text());source=json.loads((root/'data_snapshots/source.json').read_text())
    records=json.loads((root/'data_snapshots/records.json').read_text());calendar=json.loads((root/'data_snapshots/calendar.json').read_text())['dates']
    selection=json.loads((root/'validation/source_policy_selection.json').read_text())
    if selection['selected']:
        v=selection['selected']['variant'];cfg['trend']['candidate_cross_threshold']=v['cross']
        cfg['consensus'].update(buy_base_threshold=v['buy_base'],sell_threshold=v['sell'],margin=v['margin'])
    cfg['candidate_policy_hash']=stable_hash({'candidate':cfg['candidate_policy'],'risk_policy':cfg['risk_policy'],
        'trend_candidate':{k:cfg['trend'][k] for k in ('candidate_cross_threshold','candidate_margin')},
        'consensus':cfg['consensus'],'signal_points':cfg['signal_points'],'native_records':source['records_hash']})
    trends=json.loads((root/'models/trend_bundles.json').read_text());ma=json.loads((root/'models/ma_exit_bundles.json').read_text())
    models=json.loads((root/'models/l2_bundles.json').read_text());architecture=[];manifests=[];unavailable=[]
    cfg['E0_policy_hash']=stable_hash({'stage':'E0','candidate_policy_hash':cfg['candidate_policy_hash'],'account':cfg['account'],
        'trend_catalog_hash':stable_hash(trends),'ordinary_sell':'new_consensus_direction','entry_HR':False})
    for b in trends:
        m=load_trend_head(root/b['heads']['trend_direction']['path']);manifests.append(m)
        assert b['heads']['trend_direction']['manifest_sha256']==m['manifest_sha256']
        assert m['target_contract']['pipeline_contract_hash']==cfg['pipeline_contract_hash']
    E0hash=stable_hash(cfg)
    for b in ma:
        m=load_head(root/b['heads']['ma_exit']['path']);manifests.append(m)
        if m['target_contract']['strategy_hash']!=E0hash or m['target_contract']['continuation_policy_hash']!=cfg['E0_policy_hash']:
            raise ValueError('E0 reconstructed policy differs from trained MA head')
        assert b['heads']['ma_exit']['manifest_sha256']==m['manifest_sha256']
    cfg['E1_policy_hash']=stable_hash({'stage':'E1','E0':cfg['E0_policy_hash'],'signal_time_pinned_MA_exit':stable_hash(ma),
        'holding_threshold':cfg['account']['holding_exit_threshold'],'no_head':'entire_position_E0','checkpoint_upgrade':False,'L2_exit_apply':False})
    final={'entries':pd.read_parquet(root/'dataset/E1/entries.parquet'),'exits':pd.read_parquet(root/'dataset/E1/exits.parquet')}
    end=source['effective_data_end'];E1hash=stable_hash(cfg)
    for fold in [*cfg['development_windows'],cfg['production']]:
        f=split_validation(fold,calendar)
        for arch in cfg['architectures']:
            bundle=None
            for head in ('entry','exit','exit_advantage'):
                path=root/'models'/arch/fold['name']/head
                if not (path/'manifest.json').exists():
                    data=training_frame(final,head,cfg,calendar,'E1');train,cal,policy=time_masks(data,f,head=='exit_advantage')
                    unavailable.append({'head':head,'fold':fold['name'],'architecture':arch,
                        'reason':'mature training/calibration population or both-class minimum not met',
                        'sample_counts':{'train':int(train.sum()),'calibration':int(cal.sum()),'policy':int(policy.sum())},
                        'calibration_class_counts':data.loc[cal,'label'].value_counts().to_dict()})
                    continue
                m=load_head(path);manifests.append(m)
                if m['target_contract']['strategy_hash']!=E1hash or m['target_contract']['continuation_policy_hash']!=cfg['E1_policy_hash']:
                    raise ValueError('E1 reconstructed target differs from trained L2 head')
                b={'name':f['name'],'available_at':f['policy_end'],'fold':f,'architecture':arch,'heads':{},
                    'policy':{'entry_threshold':cfg['consensus']['high_return_threshold'],'exit_threshold':cfg['account']['holding_exit_threshold']}}
                if bundle is None:bundle=b
                bundle['heads'][head]={'path':str(path.relative_to(root)),'manifest_sha256':m['manifest_sha256'],'target_contract_hash':m['target_contract_hash']}
            if bundle:architecture.append(bundle)
    expected=[b for b in architecture if b['architecture']=='linear' and b['heads'].get('entry')]
    if models!=expected:raise ValueError('saved main L2 catalog differs from physical trained manifests')
    for stage in ('E0','E1'):
        manifest=json.loads((root/'dataset'/stage/'manifest.json').read_text())
        if len(manifest['records'])!=len(records) or manifest['failures']:raise ValueError('incomplete actual-account cohort')
        for record in manifest['records']:
            for file in record['files'].values():
                if _file_hash(Path(file['path']))!=file['sha256']:raise ValueError('stored ledger/label asset drift')
    recent=any(b['name']==cfg['production']['name'] and b['heads'].get('entry') for b in models) and any(b['name']==max(trends,key=lambda b:b['available_at'])['name'] for b in ma)
    adapter={'reason':'Path normalization for final SHA check; no labels/models regenerated','source_run_id':run_id,
        'recovery_module_sha256':_file_hash(Path(__file__)),'original_generating_hashes_verified':True,'completed_at':local_now().isoformat()}
    freeze={'family':STRATEGY_ID,'actual_freeze_at':local_now().isoformat(),'config_hash':stable_hash(cfg),
        'pipeline_contract_hash':cfg['pipeline_contract_hash'],'source_snapshot_hash':source['snapshot_hash'],'code_hashes':code,
        'models_hash':stable_hash([m['manifest_sha256'] for m in manifests]),
        'models_catalog_hash':stable_hash({'trend':trends,'ma_exit':ma,'architecture':architecture}),
        'independent_window_status':'pending','historical_available_is_simulated':True,'actual_completion_not_backdated':True,
        'finalization_adapter':adapter}
    freeze['freeze_hash']=stable_hash(freeze)
    old=json.loads((ARTIFACT_RUNS_ROOT/source['source_native_run_id']/'reports/native_v4.json').read_text())
    report={'run_id':run_id,'status':'completed','strategy_version':STRATEGY_ID,'config':cfg,'source_snapshot':source,'data_end':end,
        'dataset_records':records,'validation_records':old['dataset_records'],'trend_bundles':trends,'ma_exit_bundles':ma,'model_bundles':models,
        'architecture_bundles':architecture,'source_policy_selection':selection,'holding_exit_policy_selection':json.loads((root/'validation/holding_exit_policy_selection.json').read_text()),
        'continuation_stage':'E1','unavailable_heads':unavailable,'failures':[],'protocol_freeze':freeze,'completed_at':local_now().isoformat(),
        'required_recent_heads_ready':recent,'automatic_selection_eligible':len(records)>=60 and cfg['trend']['horizon_sessions']==10 and recent,
        'architecture_selection':'preregistered_linear_main_tree_mlp_controlled_ablations','publication_allowed':False,
        'pipeline':{'stages':['independent_trend','source_policy','E0','MA_exit','E1','L2_entry_E1','L2_exit_shadow'],
            'trend_heads':len(trends),'ma_exit_heads':len(ma),'entry_heads':len(models),'holding_checkpoint_upgrade':False,'bootstrap_uses_HR':False,
            'applied_L2_exit':False,'final_entry_rows':len(final['entries']),'final_mature_entries':int(final['entries'].label_available.sum()),
            'final_positive_entries':int(final['entries'].label.eq(1).sum()),'final_exit_rows':len(final['exits']),'final_mature_exits':int(final['exits'].label_available.sum())},
        'model_gate':{'passed':False,'reasons':['future_independent_validation_pending','partial_frozen_cohort_not_full_market',
            'development_fee_net_Broker_validation_pending','unadjusted_price_and_historical_ST_identity_limits']}}
    _save(root/'config/research.json',cfg);_save(root/'validation/protocol_freeze.json',freeze)
    _save(root/'validation/finalization_adapter.json',adapter);_save(target,report)
    metadata=json.loads((root/'metadata.json').read_text());metadata.update(status='completed',pipeline=report['pipeline'],failures=[]);_save(root/'metadata.json',metadata)
    return report
