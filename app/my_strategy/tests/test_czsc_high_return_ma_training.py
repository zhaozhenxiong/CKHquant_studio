"""Dependency order, version compatibility and immutable context guards."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_high_return_research as original_research

from my_strategy.services import czsc_high_return_ma_training as training


class Context:
    def __init__(self, path):
        self.path = path
    def run_dir(self):
        return self.path
    def subdir(self, *parts):
        path = self.path.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


def config():
    return original_research.research_config({'version': training.RESEARCH_VERSION,
        'ma_entry_alignment': training.ALIGNMENT_MODE})


@pytest.mark.parametrize('version,mode,valid',[
    ('czsc_high_return_v1',None,True),
    (training.RESEARCH_VERSION,training.ALIGNMENT_MODE,True),
    ('czsc_high_return_v1',training.ALIGNMENT_MODE,False),
    (training.RESEARCH_VERSION,None,False),
    (training.RESEARCH_VERSION,'',False),
    ('invented_version',training.ALIGNMENT_MODE,False),
])
def test_version_cannot_relabel_an_old_dependency(version,mode,valid):
    cfg={'version':version,'ma_entry_alignment':mode}
    if valid:
        training.validate_alignment_version(cfg)
    else:
        with pytest.raises(ValueError,match='version and MA target dependency'):
            training.validate_alignment_version(cfg)


@pytest.fixture
def dependency(tmp_path,monkeypatch):
    cfg=config(); context=Context(tmp_path)
    calendar=pd.bdate_range('2023-01-02','2024-03-29').strftime('%Y-%m-%d').tolist()
    snap={'snapshot_hash':'the-frozen-input-snapshot'}
    records=[{'symbol':'000001.SZ'}]
    base=training._expected_v0_exit_contract(cfg,calendar,snap['snapshot_hash'])
    new_base=copy.deepcopy(base)
    new_base.update(target='actual_complete_fee_net_roundtrip_positive',
        label_version='czsc_ma_entry_positive_signal_pinned_learned_exit_v2',
        exit_alignment_recipe_hash='the-prior-exit-recipe',
        candidate_policy='broad_causal_ma_rule_no_entry_head')
    new_base['contract_hash']=stable_hash({k:v for k,v in new_base.items() if k!='contract_hash'})
    bootstrap={'name':'bootstrap','failures':[],'records':records}
    alignment={'name':'alignment','failures':[],'records':records,'successful':1}
    context.subdir('dataset','bootstrap').joinpath('manifest.json').write_text(json.dumps(bootstrap))
    events=[]
    exit_data=pd.DataFrame({'label':[0.,1.]})
    entry_data=pd.DataFrame({'label':[1.,0.]})
    def load(stage,source,head,ma_only=False,source_snapshot_hash=None):
        assert source=='ma' and ma_only and source_snapshot_hash==snap['snapshot_hash']
        events.append(('load',stage['name'],head))
        if stage is bootstrap:
            assert head=='exit', 'an entry head must not fit the original bootstrap labels'
            return exit_data,base
        assert stage is alignment and head=='entry'
        return entry_data,new_base
    def fit(data,contract,fold,ctx,c,*,head,calendar,device,check_cancel,progress):
        events.append(('fit',head))
        assert (head=='ma_exit' and data is exit_data and contract is base
            or head=='ma_entry' and data is entry_data and contract is new_base)
        split=original_research.split_validation(fold,calendar)
        return {'name':split['name'],'available_at':split['validation_end'],'fold':split,
            'architecture':'linear','heads':{head:{'target_contract_hash':head+'-fit-target'}},
            'policy':{'ma_exit_threshold':.55},'failures':[]}
    def build(recs,cal,c,end,ctx,stage,**kwargs):
        events.append(('build',stage))
        assert stage=='ma_entry_alignment'
        assert kwargs['sources']==('ma',)
        assert all(set(b['heads'])=={'ma_exit'} for b in kwargs['bundles'])
        stored=json.loads(ctx.subdir('models').joinpath('ma_exit_bootstrap_lineage.json').read_text())
        assert stored['entry_heads_used_in_label_generation']==0
        assert stored['bootstrap_exit_base_contract']==base
        assert stable_hash(stored['bootstrap_exit_model_contract'])==stored['expected_exit_target_contract_hash']
        ctx.subdir('dataset','ma_entry_alignment').joinpath('manifest.json').write_text(json.dumps(alignment))
        return alignment
    monkeypatch.setattr(original_research,'_load_training',load)
    monkeypatch.setattr(original_research,'_build',build)
    monkeypatch.setattr(training,'_train_ma_head',fit)
    return cfg,context,calendar,snap,records,base,bootstrap,alignment,events


def test_exit_only_labels_precede_entry_fit_and_no_bootstrap_entry_is_loaded(dependency):
    cfg,ctx,calendar,snap,records,base,bootstrap,alignment,events=dependency
    bundles,pipeline=training.train_exit_bound_ma(records,calendar,snap,'2024-03-29',bootstrap,ctx,cfg,
        device='cpu',cpu_workers=1)
    assert events==[('load','bootstrap','exit'),('fit','ma_exit'),
        ('build','ma_entry_alignment'),('load','alignment','entry'),('fit','ma_entry')]
    assert set(bundles[0]['heads'])=={'ma_exit','ma_entry'}
    assert pipeline['exit_heads']==pipeline['entry_heads']==1
    assert pipeline['entry_heads_used_in_label_generation']==0
    assert pipeline['entry_target_contract']['exit_alignment_recipe_hash']=='the-prior-exit-recipe'
    assert bundles[0]['l1_target_dependency']['entry_target_contract_hash']=='ma_entry-fit-target'


def test_canonical_contract_guard_rejects_another_continuation_before_any_fit(dependency):
    cfg,ctx,calendar,snap,records,base,bootstrap,alignment,events=dependency
    base['continuation_policy_hash']='foreign-learned-or-fixed-horizon-policy'
    base['contract_hash']=stable_hash({k:v for k,v in base.items() if k!='contract_hash'})
    with pytest.raises(ValueError,match='canonical raw-MA rule-v0'):
        training.train_exit_bound_ma(records,calendar,snap,'2024-03-29',bootstrap,ctx,cfg,
            device='cpu',cpu_workers=1)
    assert events==[('load','bootstrap','exit')]


def test_incomplete_derived_label_stage_stops_before_entry_fit(dependency):
    cfg,ctx,calendar,snap,records,base,bootstrap,alignment,events=dependency
    alignment['failures']=[{'symbol':'000001.SZ','error':'actual exit branch unmatured input corruption'}]
    with pytest.raises(RuntimeError,match='label stage is incomplete'):
        training.train_exit_bound_ma(records,calendar,snap,'2024-03-29',bootstrap,ctx,cfg,
            device='cpu',cpu_workers=1)
    assert events==[('load','bootstrap','exit'),('fit','ma_exit'),('build','ma_entry_alignment')]


@pytest.fixture
def lineage_context(tmp_path):
    cfg=config(); ctx=Context(tmp_path)
    dates=pd.bdate_range('2023-01-02','2024-03-29').strftime('%Y-%m-%d').tolist()
    records=[{'symbol':'000001.SZ','path':'frozen-feature.parquet','sha256':'bound-feature-sha'}]
    snapshot={'records_hash':stable_hash(records),'effective_data_end':dates[-1],
        'calendar_hash':stable_hash(dates),'selected_count':1}
    snapshot['snapshot_hash']=stable_hash(snapshot)
    base=training._expected_v0_exit_contract(cfg,dates,snapshot['snapshot_hash'])
    _,model=original_research._head_data(pd.DataFrame(),base,'ma_exit',training.L1_MA_EXIT_COLUMNS)
    fold=original_research.split_validation(original_research._ma_folds(dates[-1])[0],dates)
    catalog=[{'name':fold['name'],'fold':fold,'available_at':fold['validation_end'],
        'heads':{'ma_exit':{'path':'models/l1/one/ma_exit','manifest_sha256':'test-placeholder-head'}},
        'policy':{'ma_exit_threshold':.55}}]
    def save(relative,value):
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(value))
    bootstrap={'requested':1,'successful':1,'failures':[],'sources':['ma'],'records':records}
    save('dataset/bootstrap/manifest.json',bootstrap)
    lineage={'version':training.ALIGNMENT_MODE,'source_snapshot_hash':snapshot['snapshot_hash'],
        'calendar_hash':stable_hash(dates),'bootstrap_exit_base_contract':base,
        'bootstrap_exit_model_contract':model,'expected_exit_target_contract_hash':stable_hash(model),
        'bootstrap_manifest_path':'dataset/bootstrap/manifest.json',
        'bootstrap_manifest_sha256':training._file_hash(tmp_path/'dataset/bootstrap/manifest.json'),
        'exit_only_bundles':catalog,'exit_only_bundles_hash':stable_hash(catalog),
        'entry_heads_used_in_label_generation':0,'holding_checkpoint_upgrade':False}
    lineage['lineage_hash']=stable_hash(lineage)
    save('models/ma_exit_bootstrap_lineage.json',lineage)
    save('config/research.json',cfg)
    save('data_snapshots/calendar.json',{'verified':True,'dates':dates,'source':snapshot})
    save('data_snapshots/source.json',snapshot)
    save('data_snapshots/source_records.json',records)
    save('metadata.json',{'task':'czsc-high-return-research-train','scope':'train',
        'config_hash':stable_hash(cfg),'data_version':snapshot['snapshot_hash']})
    save('config/initial_code_hashes.json',{str(p):training._file_hash(p) for p in original_research._code_files()})
    return tmp_path,cfg,dates,lineage,save


def test_incomplete_context_is_not_a_completed_runtime(lineage_context):
    root,cfg,dates,lineage,save=lineage_context
    with pytest.raises(ValueError,match='requires a completed actual protocol freeze'):
        training.read_exit_bootstrap_lineage(root,dates,cfg)
    actual=training.read_exit_bootstrap_lineage(root,dates,cfg,allow_training=True)
    assert actual['_frozen_calendar_dates']==dates
    # This asserts context authorization only; placeholder heads cannot infer.


def test_internal_context_cannot_ignore_changed_initial_code(lineage_context):
    root,cfg,dates,lineage,save=lineage_context
    save('config/initial_code_hashes.json',{})
    with pytest.raises(ValueError,match='immutable training context'):
        training.read_exit_bootstrap_lineage(root,dates,cfg,allow_training=True)


def test_verified_future_calendar_extends_original_prefix(lineage_context):
    root,cfg,dates,lineage,save=lineage_context
    extended=dates+['2024-04-01','2024-04-02']
    actual=training.read_exit_bootstrap_lineage(root,extended,cfg,allow_training=True)
    assert actual['_frozen_calendar_dates']==dates
    with pytest.raises(ValueError,match='verified execution prefix changed'):
        training.read_exit_bootstrap_lineage(root,extended[1:],cfg,allow_training=True)


def test_self_rehashed_foreign_bootstrap_policy_is_still_rejected(lineage_context):
    root,cfg,dates,lineage,save=lineage_context
    lineage['bootstrap_exit_base_contract']['continuation_policy_hash']='foreign-fixed-horizon'
    lineage['bootstrap_exit_base_contract']['contract_hash']=stable_hash({k:v for k,v in
        lineage['bootstrap_exit_base_contract'].items() if k!='contract_hash'})
    lineage['lineage_hash']=stable_hash({k:v for k,v in lineage.items() if k!='lineage_hash'})
    save('models/ma_exit_bootstrap_lineage.json',lineage)
    with pytest.raises(ValueError,match='not the canonical raw-MA rule-v0'):
        training.read_exit_bootstrap_lineage(root,dates,cfg,allow_training=True)


def test_completed_inline_report_requires_matching_separate_freeze(lineage_context):
    root,cfg,dates,lineage,save=lineage_context
    pipeline={'lineage_sha256':training._file_hash(root/'models/ma_exit_bootstrap_lineage.json')}
    freeze={'config_hash':stable_hash(cfg),'source_snapshot_hash':lineage['source_snapshot_hash'],
        'model_bundles_hash':stable_hash(lineage['exit_only_bundles']),
        'l1_ma_target_pipeline_hash':stable_hash(pipeline),
        'l1_exit_lineage_sha256':pipeline['lineage_sha256']}
    freeze['freeze_hash']=stable_hash(freeze)
    report={'status':'completed','config':cfg,'source_snapshot':{'snapshot_hash':lineage['source_snapshot_hash']},
        'model_bundles':lineage['exit_only_bundles'],'l1_ma_target_pipeline':pipeline,'protocol_freeze':freeze}
    metadata=json.loads((root/'metadata.json').read_text())
    metadata.update(status='complete',model_family='high_return')
    save('metadata.json',metadata)
    save('reports/high_return.json',report);save('validation/protocol_freeze.json',freeze)
    assert training.read_exit_bootstrap_lineage(root,dates,cfg)['version']==training.ALIGNMENT_MODE
    altered=copy.deepcopy(freeze);altered['l1_exit_lineage_sha256']='different'
    altered['freeze_hash']=stable_hash({k:v for k,v in altered.items() if k!='freeze_hash'})
    save('validation/protocol_freeze.json',altered)
    with pytest.raises(ValueError,match='actual protocol freeze'):
        training.read_exit_bootstrap_lineage(root,dates,cfg,allow_training=True)


def test_default_requires_exit_bound_targets_and_legacy_cache_cannot_create_run(monkeypatch):
    cfg = original_research.research_config()
    assert cfg['version'] == training.RESEARCH_VERSION
    assert cfg['ma_entry_alignment'] == training.ALIGNMENT_MODE
    def forbidden_source(*args, **kwargs):
        raise AssertionError('cache rejection must happen before any run/source operation')
    monkeypatch.setattr(original_research, '_source', forbidden_source)
    for key in ('bootstrap_artifact_run_id', 'ma_checkpoint_run_id', 'final_artifact_run_id'):
        with pytest.raises(ValueError, match='requires fresh bootstrap'):
            original_research.train_high_return_research(**{key:'old-version-cache'})


def test_explicit_legacy_report_retains_its_original_routing_version():
    cfg = original_research.research_config({'version':'czsc_high_return_v1'})
    assert cfg['version'] == 'czsc_high_return_v1'
    assert cfg.get('ma_entry_alignment') is None
    from my_strategy.services.czsc_high_return_runtime import _config
    assert _config(cfg).get('ma_entry_alignment') is None


def test_status_separates_current_research_version_from_saved_run_versions(monkeypatch):
    from my_strategy.services import czsc_high_return_runtime as runtime
    reports = {
        'legacy': {'config': {'version': 'czsc_high_return_v1'}},
        'aligned': {'config': {'version': training.RESEARCH_VERSION}},
    }
    class IsolatedResultStore:
        def list(self, *args, **kwargs):
            return [{'run_id': name} for name in reports]
        def get(self, run_id):
            return reports[run_id]
    monkeypatch.setattr(original_research, 'ResultStore', IsolatedResultStore)
    monkeypatch.setattr(runtime, 'high_return_research_status', lambda: {'models': [], 'failures': []})
    status = original_research.status_high_return()
    assert status['strategy_version'] == training.RESEARCH_VERSION
    assert {item['run_id']: item['strategy_version'] for item in status['runs']} == {
        'legacy': 'czsc_high_return_v1', 'aligned': training.RESEARCH_VERSION}
    assert status['publication_allowed'] is False
