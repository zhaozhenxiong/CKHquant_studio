"""Daily attempts originate before admission in real strict-prior accounts."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_v2_contracts import strategy_config, STRATEGY_ID
from my_strategy.services.czsc_trend_v2_dataset import build_account_dataset, pre_gate_attempt, training_frame
from my_strategy.services.czsc_trend_v2_pipeline import make_reference_population, train_trend_consensus, verify_source_generators
from my_strategy.services import czsc_trend_v2_pipeline as pipeline
from my_strategy.services.czsc_research_profiles import MA_TREND_COLUMNS
from my_strategy.tests.test_czsc_trend_account import fixture


def inputs(monkeypatch):
    frame,calendar,prepared=fixture(90)
    for column in MA_TREND_COLUMNS:frame[column]=.1
    frame['v4_structure_input_eligible']=False;frame['v4_w_input_eligible']=False
    template=copy.deepcopy(prepared['rows'][60]['entry_plan'])
    for row in prepared['rows']:
        row.update(action='WAIT',entry_eligible=False,entry_plan=None,fresh_candidate_ids=[])
    for index in (70,71,72):
        plan={**template,'plan_id':'plan-'+str(index),'signal_date':calendar[index],
            'available_at':calendar[index]+'T15:00:00+08:00'}
        prepared['rows'][index].update(action='BUY',entry_eligible=True,entry_plan=plan,
            original_opportunity_ids=['same-original'],fresh_candidate_ids=['same-original'],buy_base_score=.65)
    prepared['events']=[]
    monkeypatch.setattr('my_strategy.services.czsc_trend_v2_dataset.prepare_trend_decisions',lambda *args,**kwargs:copy.deepcopy(prepared))
    config=strategy_config({'feature_start':calendar[70],'account':{'max_holding_sessions':3}})
    config.update(source_snapshot_hash='s',pipeline_contract_hash='p',candidate_policy_hash='c',E0_policy_hash='e0',E1_policy_hash='e1')
    return frame,calendar,prepared,config


def test_missing_prior_l2_preserves_all_daily_flat_attempts_without_actual_buys(monkeypatch,tmp_path):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    def blocked(snapshot,candidate):
        return {'status':'no_prior_high_return_head','probability':None}
    result,summary=build_account_dataset(frame,calendar,{}, {},cfg,stage='E1',
        reference_entry_evaluator=blocked,reference_population={'version':pipeline.POPULATION_VERSION},
        population_end=calendar[72],directory=tmp_path)
    entries=result['entries']
    assert len(entries)==entries.attempt_id.nunique()==3
    assert entries.economic_group_id.nunique()==1
    assert entries.date.tolist()==calendar[70:73]
    assert entries.reference_allowed.eq(False).all()
    assert entries.reference_gate_status.eq('no_prior_high_return_head').all()
    assert entries.actual_flat_reference.all() and entries.candidate_cash_ratio.eq(1).all()
    assert entries.label_available.all() and entries.label.eq(0).all()
    assert entries.entry_date.tolist()==calendar[71:74]
    reference=json.loads((tmp_path/'reference_account.json').read_text())
    assert reference['ledger']==[]
    assert all(row['pre_gate_state']['pending_intent'] is None for row in reference['entry_diagnostics'])
    clones=json.loads((tmp_path/'candidate_clones.json').read_text())
    assert len(clones)==3 and all(len([fill for fill in row['ledger'] if fill['action']=='BUY'])==1 for row in clones)
    assert summary['reference_mode']=='final' and summary['ordinary_sell_policy']=='continuous_sell_pass_for_actual_holding'
    assert result['exits'].empty


def test_training_never_uses_gate_after_state_with_queued_buy(monkeypatch,tmp_path):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    result,summary=build_account_dataset(frame,calendar,{}, {},cfg,stage='E0',directory=tmp_path)
    entries=result['entries']
    assert len(entries)==1 and entries.reference_allowed.iloc[0]
    reference=json.loads((tmp_path/'reference_account.json').read_text())
    diagnostic=reference['entry_diagnostics'][0]
    after=next(state for state in reference['account_snapshots'] if state['last_close_date']==calendar[70])
    assert diagnostic['pre_gate_state']['pending_intent'] is None
    assert after['pending_intent']['side']=='BUY'
    assert entries.reference_state_hash.iloc[0]==stable_hash(diagnostic['pre_gate_state'])
    poisoned=copy.deepcopy(diagnostic);poisoned['pre_gate_state']=after
    with pytest.raises(ValueError,match='pre-gate'):
        pre_gate_attempt(poisoned,frame.symbol.iloc[0],cfg)


def test_future_label_clones_do_not_create_future_reference_population(monkeypatch):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    blocked=lambda *args:{'status':'model_unavailable','probability':None}
    result,summary=build_account_dataset(frame,calendar,{}, {},cfg,stage='E1',reference_entry_evaluator=blocked,
        population_end=calendar[70])
    assert result['entries'].date.tolist()==[calendar[70]]
    assert result['entries'].label_end.iloc[0]>calendar[70]
    assert not result['entries'].reference_allowed.iloc[0]
    assert summary['entries_mature_at_frozen_source']==1
    assert summary['entries_mature_by_population_end']==0
    assert summary['entries_positive_by_population_end']==0
    assert 'mature_entries' not in summary and 'held_candidates' not in summary['counts']


def fold():
    return {'name':'later','train_end':'2025-06-30','calibration_start':'2025-07-01',
        'calibration_end':'2025-08-15','policy_start':'2025-08-18','policy_end':'2025-09-30'}


def bundle(name='prior',available='2025-03-31'):
    return {'name':name,'available_at':available,'heads':{'entry':{'path':'models/linear/prior/entry','manifest_sha256':'manifest'}}}


def test_population_catalog_is_frozen_and_rejects_own_or_nonprior_head():
    original=[bundle()]
    population=make_reference_population(fold(),original)
    original[0]['name']='mutated'
    assert population['prior_entry_catalog'][0]['name']=='prior'
    assert population['prior_entry_catalog_hash']==stable_hash(population['prior_entry_catalog'])
    assert population['fold']==fold() and population['population_end']=='2025-09-30'
    assert not population['self_head_used']
    for candidate in (bundle(name='later'),bundle(available='2025-07-01')):
        with pytest.raises(ValueError,match='earlier'):
            make_reference_population(fold(),[candidate])


def test_entry_contract_binds_specific_real_reference_population(monkeypatch):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    population=make_reference_population(fold(),[])
    dataset={'entries':pd.DataFrame(), 'reference_population':population}
    contract=training_frame(dataset,'entry',cfg,calendar,'E1').attrs['label_contract']
    assert contract['reference_population']==population
    assert contract['reference_population_hash']==stable_hash(population)
    assert contract['training_population']==pipeline.POPULATION_VERSION


def test_capacity_only_slices_fresh_superset_and_rebuilds_every_tier(tmp_path,monkeypatch):
    source=tmp_path/'fresh'/'reports';source.mkdir(parents=True)
    records=[{'symbol':str(i)} for i in range(4)]
    (source/'native_v4.json').write_text(json.dumps({'strategy_version':STRATEGY_ID,'raw_only_rebuild':True,'dataset_records':records}))
    monkeypatch.setattr(pipeline,'ARTIFACT_RUNS_ROOT',tmp_path)
    observed=[]
    def train(**kwargs):
        observed.append(copy.deepcopy(kwargs['native_records']))
        return {'run_id':str(len(observed)),'status':'capacity_insufficient','required_recent_heads_ready':False,
            'unavailable_heads':[{'head':'trend_direction','reason':'insufficient mature trend classes'}],
            'pipeline':{'complete_model_population_DAG':False}}
    monkeypatch.setattr(pipeline,'_train_one',train)
    report=train_trend_consensus(source_native_run_id='fresh',config={'training_capacity_tiers':[2,4]})
    assert observed==[records[:2],records]
    assert report['run_id']=='2'
    assert report['status']=='capacity_insufficient' and not report['pipeline']['complete_model_population_DAG']
    assert report['capacity_status']=='all_preregistered_tiers_exhausted_with_missing_dependencies'
    assert [row['tier'] for row in report['capacity_attempts']]==[2,4]
    assert all(row['unavailable_heads'] for row in report['capacity_attempts'])
    saved=json.loads((tmp_path/'2/validation/capacity_attempts.json').read_text())
    assert saved['attempts']==report['capacity_attempts']
    assert (tmp_path/'1/reports/trend_consensus.json').is_file()
    (source/'native_v4.json').write_text(json.dumps({'strategy_version':'czsc_trend_consensus','raw_only_rebuild':True,'dataset_records':records}))
    with pytest.raises(ValueError,match='raw-rebuilt v2'):
        train_trend_consensus(source_native_run_id='fresh',config={'training_capacity_tiers':[2,4]})


def test_source_generator_hashes_resolve_current_loaded_package(tmp_path,monkeypatch):
    (tmp_path/'services').mkdir()
    hashes={}
    for index in range(6):
        path=tmp_path/'services'/('source'+str(index)+'.py');path.write_text('frozen')
        hashes[path.name]=pipeline._file_hash(path)
    monkeypatch.setattr(pipeline,'MY_STRATEGY_ROOT',tmp_path)
    verify_source_generators({'code_hashes':hashes})
    (tmp_path/'services/source0.py').write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        verify_source_generators({'code_hashes':hashes})


@pytest.mark.parametrize('empty_population,missing_trend',[(False,False),(True,False),(True,True)])
def test_pipeline_builds_each_actual_population_before_its_own_entry_head(tmp_path,monkeypatch,empty_population,missing_trend):
    """Exercise the real stage orchestration with cheap fitting/data workers."""
    class Context:
        run_id='new-v2'
        def run_dir(self):return tmp_path/self.run_id
        def subdir(self,*parts):
            path=self.run_dir().joinpath(*parts);path.mkdir(parents=True,exist_ok=True);return path
        def write_metadata(self,*args):pass
    context=Context()
    source_dir=tmp_path/'fresh/reports';source_dir.mkdir(parents=True)
    raw=tmp_path/'raw.csv';raw.write_text('new raw observations')
    record={'symbol':'000001.SZ','raw_path':str(raw),'raw_sha256':pipeline._file_hash(raw),
        'raw_only_rebuild':True,'reused_model_or_label':False,'raw_series_hash':'raw-only',
        'raw_attrs':{'price_basis':'unadjusted','legacy_fuyao_bars':0,'unverified_trade_price_bars':0,
            'volume_unit':'shares','amount_unit':'CNY','data_version':'raw-only'}}
    calendar=pd.bdate_range('2023-01-02','2026-10-08').strftime('%Y-%m-%d').tolist()
    source={'strategy_version':STRATEGY_ID,'raw_only_rebuild':True,'reused_model_or_label':False,
        'data_end':calendar[-1],'calendar_dates':calendar,'dataset_records':[record],
        'source_snapshot':{'calendar_run_id':'new-raw','raw_input_hashes':{record['symbol']:record['raw_sha256']}}}
    (source_dir/'native_v4.json').write_text(json.dumps(source))
    monkeypatch.setattr(pipeline,'ARTIFACT_RUNS_ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'GENERATING_FILES',('services/czsc_trend_v2_dataset.py',))
    monkeypatch.setattr(pipeline,'verify_source_generators',lambda value:None)
    monkeypatch.setattr(pipeline,'create_run_context',lambda **kwargs:context)
    real_frame,real_calendar,prepared,cfg=inputs(monkeypatch)
    for row in prepared['rows']:
        row.update(action='WAIT',entry_eligible=False,entry_plan=None,fresh_candidate_ids=[])
    monkeypatch.setattr('my_strategy.services.czsc_trend_v2_dataset.prepare_trend_decisions',
        lambda *args,**kwargs:copy.deepcopy(prepared))
    monkeypatch.setattr(pipeline,'read_snapshot',lambda record:(real_frame.copy() if empty_population else pd.DataFrame({'date':[calendar[0]]}),{}))
    def observations(*args,**kwargs):
        value=pd.DataFrame({'date':[calendar[0]],'label':[1]});value.attrs['label_contract']={};return value
    monkeypatch.setattr(pipeline,'build_trend_labels',observations)
    earlier={'name':'trend-prior','train_end':'2024-06-28','validation_start':'2024-07-01',
        'validation_end':'2024-09-30','calibration_start':'2024-07-01',
        'calibration_end':'2024-08-15','policy_start':'2024-08-16','policy_end':'2024-09-30'}
    monkeypatch.setattr(pipeline,'chronological_folds',lambda *args:[earlier])
    def trend_fit(*args,**kwargs):
        if missing_trend:raise ValueError('insufficient mature trend classes train [0, 0, 0]')
        return {'manifest_sha256':'trend','sample_counts':{}}
    monkeypatch.setattr(pipeline,'train_trend_head',trend_fit)
    if not missing_trend:monkeypatch.setattr(pipeline,'select_source_policy',lambda *args:{})
    monkeypatch.setattr(pipeline,'select_holding_exit_policy',lambda *args:{})
    order=[]
    def stage(records,calendar,cfg,context,stage,*args,**options):
        population=options.get('reference_population') or {}
        name=population.get('fold',{}).get('name')
        order.append(('population' if population else 'bootstrap',name or stage,
            [bundle['name'] for bundle in population.get('prior_entry_catalog',[])]))
        directory=context.subdir('dataset',options.get('subdirectory',stage))
        if empty_population:
            # Real zero-attempt Broker replay and real empty typed dataset.
            data,summary=build_account_dataset(real_frame,calendar,{}, {},cfg,stage=stage,
                directory=directory/'000001_SZ',population_end=options.get('population_end'),
                reference_population=population,reference_entry_evaluator=(lambda *args:{'status':'model_unavailable'}) if population else None)
        else:
            data={'entries':pd.DataFrame({'label_available':[True,True],'label':[0,1],'label_end':['2025-02-01','2025-04-01']}),
                'exits':pd.DataFrame({'label_available':[True],'label':[0],'exit_advantage':[0.],'label_end':['2025-04-01']}),
                'reference_population':population}
            summary={'symbol':record['symbol']}
        (directory/'manifest.json').write_text(json.dumps({'reference_population':population}))
        return data,[summary]
    monkeypatch.setattr(pipeline,'build_stage',stage)
    def fit(data,head,fold,calendar,context,cfg,architecture='linear',device='cpu'):
        order.append(('fit',fold['name'],head))
        if head=='entry':
            population=data.attrs['label_contract']['reference_population']
            assert population['fold']==fold and fold['name'] not in [b['name'] for b in population['prior_entry_catalog']]
        return {'name':fold['name'],'available_at':fold['policy_end'],'architecture':architecture,
            'fold':fold,'heads':{head:{'path':'unused','manifest_sha256':fold['name']+head}}}, {
                'manifest_sha256':fold['name']+head,'sample_counts':{}}
    if not empty_population:monkeypatch.setattr(pipeline,'_fit_interval',fit)
    first={'name':'first','train_end':'2024-12-31','validation_start':'2025-01-01','validation_end':'2025-03-31'}
    second={'name':'production','train_end':'2025-06-30','validation_start':'2025-07-01','validation_end':'2025-09-30'}
    report=pipeline._train_one(source_native_run_id='fresh',cpu_workers=1,config={
        'development_windows':[first],'production':second,'architectures':['linear']})
    if empty_population:
        assert report['model_bundles']==[] and report['required_recent_heads_ready'] is False
        assert all('insufficient mature' in row['reason'] for row in report['unavailable_heads'])
        if missing_trend:
            assert order==[] and report['status']=='capacity_insufficient'
            assert report['trend_bundles']==[] and report['automatic_selection_eligible'] is False
            assert report['source_policy_selection'] is None
            assert report['pipeline']['complete_model_population_DAG'] is False
            assert report['population_folds']==[] and report['dataset_manifests']==[]
            assert report['required_head_status']['recent_trend_ready'] is False
            assert report['required_head_status']['same_checkpoint_ma_exit_ready'] is False
            assert report['probe_assets'] and all((context.run_dir()/asset['path']).is_file() for asset in report['probe_assets'])
            with pytest.raises(ValueError,match='incomplete'):
                pipeline.read_run(report['run_id'])
            return
        assert [item for item in order if item[0]=='population']==[
            ('population','first',[]),('population','production',[])]
        assert len([row for row in report['unavailable_heads'] if row['head']=='entry'])==2
        assert report['pipeline']['final_entry_rows']==0
    else:
        assert [item for item in order if item[0]=='population']==[
            ('population','first',[]),('population','production',['first'])]
        assert order.index(('population','first',[]))<order.index(('fit','first','entry'))
        assert order.index(('fit','first','entry'))<order.index(('population','production',['first']))
        first_population=report['population_folds'][0]
        assert first_population['entries_mature_at_frozen_source']==2
        assert first_population['entries_mature_by_population_end']==1
        assert first_population['entries_positive_at_frozen_source']==1
        assert first_population['entries_positive_by_population_end']==0
    assert len(report['dataset_manifests'])==4
    assert report['pipeline']['self_head_controls_its_training_population'] is False


def test_native_snapshot_restores_original_price_basis_and_units(tmp_path):
    features=tmp_path/'features.parquet';native=tmp_path/'native.json'
    pd.DataFrame({'date':['2024-01-01'],'symbol':['000001.SZ'],'source':['Fuyao']}).to_parquet(features,index=False)
    native.write_text('{}')
    attrs={'price_basis':'mixed_legacy_adjustment_unverified','legacy_fuyao_bars':1,
        'unverified_trade_price_bars':1,'volume_unit':'shares','amount_unit':'CNY','data_version':'raw-prefix'}
    record={'path':str(features),'sha256':pipeline._file_hash(features),
        'native_path':str(native),'native_sha256':pipeline._file_hash(native),
        'raw_attrs':attrs,'raw_series_hash':'raw-prefix'}
    frame,pack=pipeline.read_snapshot(record)
    assert frame.attrs==attrs and pack['frame'] is frame
    del record['raw_attrs']
    with pytest.raises(ValueError,match='price-basis'):
        pipeline.read_snapshot(record)


def test_auxiliary_frames_use_same_complete_entry_clone_population(monkeypatch):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    population=make_reference_population(fold(),[])
    dataset={'entries':pd.DataFrame({'net_return':[.12,-.03,.08],
        'label_available':[True,True,False],'label_end':['2024-06-01','2024-06-02',None]}),
        'reference_population':population}
    returns=training_frame(dataset,'return',cfg,calendar,'E1')
    losses=training_frame(dataset,'loss',cfg,calendar,'E1')
    assert returns.label.iloc[:2].tolist()==[.12,-.03] and pd.isna(returns.label.iloc[2])
    assert losses.label.iloc[:2].tolist()==[0.,1.] and pd.isna(losses.label.iloc[2])
    assert returns.attrs['label_contract']['output_type']=='continuous'
    assert losses.attrs['label_contract']['reference_population']==population


def test_readiness_requires_latest_expected_heads_not_latest_surviving_old_head():
    earlier={'name':'older','policy_end':'2025-03-31'}
    latest={'name':'latest','policy_end':'2025-09-30'}
    def ready(fold,head):
        return {'name':fold['name'],'available_at':fold['policy_end'],'heads':{head:{'path':'qualified'}}}
    production={'name':'production','heads':{'entry':{'path':'qualified'}}}
    status=pipeline.required_head_status([earlier,latest],[ready(earlier,'trend_direction')],
        [ready(earlier,'ma_exit')],[production],'production')
    assert status=={'recent_checkpoint':'latest','recent_trend_ready':False,
        'same_checkpoint_ma_exit_ready':False,'production_entry_ready':True}
    status=pipeline.required_head_status([earlier,latest],[ready(latest,'trend_direction')],
        [ready(earlier,'ma_exit')],[production],'production')
    assert status['recent_trend_ready'] and not status['same_checkpoint_ma_exit_ready']
    status=pipeline.required_head_status([earlier,latest],[ready(latest,'trend_direction')],
        [ready(latest,'ma_exit')],[production],'production')
    assert all(status[key] for key in ('recent_trend_ready','same_checkpoint_ma_exit_ready','production_entry_ready'))


def test_model_freeze_resolves_installed_code_and_binds_served_catalog(tmp_path,monkeypatch):
    loaded=tmp_path/'installed';(loaded/'services').mkdir(parents=True)
    originals=tmp_path/'old-checkout';(originals/'services').mkdir(parents=True)
    codes={}
    for index in range(6):
        relative='services/generator'+str(index)+'.py'
        (loaded/relative).write_text('same code');(originals/relative).write_text('same code')
        codes[relative]=pipeline._file_hash(loaded/relative)
    monkeypatch.setattr(pipeline,'MY_STRATEGY_ROOT',loaded)
    monkeypatch.setattr(pipeline,'GENERATING_FILES',tuple(codes))
    source={'records_hash':stable_hash([]),'source_snapshot':{'code_hashes':codes}}
    source['snapshot_hash']=stable_hash(source)
    config={'version':STRATEGY_ID,'source_snapshot_hash':source['snapshot_hash']}
    freeze={'config_hash':stable_hash(config),'code_hashes':codes,
        'models_catalog_hash':stable_hash({'trend':[],'ma_exit':[],'architecture':[]}),
        'population_folds_hash':stable_hash([]),'dataset_manifests_hash':stable_hash([])}
    freeze['freeze_hash']=stable_hash(freeze)
    report={'strategy_version':STRATEGY_ID,'config':config,'source_snapshot':source,
        'dataset_records':[],'population_folds':[],'dataset_manifests':[],
        'trend_bundles':[],'ma_exit_bundles':[],'architecture_bundles':[],'model_bundles':[],
        'protocol_freeze':freeze}
    pipeline._verify_run(report,tmp_path)
    changed=copy.deepcopy(report);changed['model_bundles']=[bundle()]
    with pytest.raises(ValueError,match='served L2'):
        pipeline._verify_run(changed,tmp_path)
    (loaded/'services/generator0.py').write_text('installed code changed')
    assert (originals/'services/generator0.py').read_text()=='same code'
    with pytest.raises(ValueError,match='changed|drift'):
        pipeline._verify_run(report,tmp_path)


def test_completion_audit_checks_all_pooled_and_account_assets(tmp_path):
    from my_strategy.services.czsc_trend_v2_finalization import audit_stage_assets
    directory=tmp_path/'dataset/E0';stock=directory/'000001_SZ';stock.mkdir(parents=True)
    files={}
    for name in ('entries.parquet','exits.parquet','episodes.parquet','reference_account.json',
                 'candidate_clones.json','holding_pairs.json'):
        path=stock/name;path.write_text('frozen')
        files[name]={'path':str(path),'sha256':pipeline._file_hash(path)}
    pooled={}
    for name in ('entries','exits','episodes'):
        path=directory/(name+'.parquet');path.write_text('frozen')
        pooled[name]={'path':str(path),'sha256':pipeline._file_hash(path)}
    manifest={'records':[{'symbol':'000001.SZ','files':files}],'failures':[],'pooled_files':pooled}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    assert audit_stage_assets(tmp_path,'E0',expected_symbols=['000001.SZ'])['stocks']==1
    (directory/'entries.parquet').write_text('tampered')
    with pytest.raises(ValueError,match='pooled'):
        audit_stage_assets(tmp_path,'E0',expected_symbols=['000001.SZ'])
    del manifest['records'][0]['files']['candidate_clones.json']
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='incomplete'):
        audit_stage_assets(tmp_path,'E0',expected_symbols=['000001.SZ'])
