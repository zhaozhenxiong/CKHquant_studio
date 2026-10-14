"""Complete source reuse with real Parquet, gzip, calendars and isolated databases."""
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import threading

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_source import NATIVE_GENERATING_FILES,_raw_content_hash,_population_body
from my_strategy.services.czsc_trend_pipeline import GENERATING_FILES
from my_strategy.services.czsc_liquidity import daily_eligibility,annotate_liquidity
from my_strategy.services.czsc_trend_support import _save,_file_hash
from my_strategy.storage.czsc_results import ResultStore

from my_strategy.services import model_training_center as center
from my_strategy.web_dashboard import api
from my_strategy.services import czsc_trend_source as current_source

LIVE=Path(current_source.__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path,monkeypatch):
    code=tmp_path/'code/my_strategy'
    for relative in set(GENERATING_FILES)|set(NATIVE_GENERATING_FILES):
        target=code/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(LIVE/relative,target)
    monkeypatch.setattr(center,'MY_STRATEGY_ROOT',code)
    runs=tmp_path/'runs'; raw_root=runs/'raw-input'; native_root=runs/'native-input'; cal_root=runs/'calendar-input'
    dates=pd.bdate_range('2026-10-01','2027-04-01').strftime('%Y-%m-%d').tolist()
    values=center.validate_parameters({'feature_start':'2027-01-01','end':dates[-1],'device':'cpu'})
    config=center._configuration(values,dates[-1]); policy=config['liquidity_universe']
    _save(cal_root/'data_snapshots/source.json',{'sessions':dates,'fixture_only':True})
    cal_asset=cal_root/'data_snapshots/source.json'
    _save(cal_root/'reports/calendar.json',dict(verified=True,source='independent_test_schedule',dates=dates,
        independent_from_market_bars=True,source_artifacts=[dict(path=str(cal_asset),sha256=_file_hash(cal_asset))]))
    members=[]; frames={}; daily_by_symbol={}; raw_records=[]; records=[]; inputs=[]
    input_contract=center._native_input_contract(config)
    candidates=['000001.SZ','000002.SZ','600000.SH','600001.SH']
    for index,symbol in enumerate(candidates):
        stock_dates=dates[:-5] if symbol=='600000.SH' else dates
        frame=pd.DataFrame(dict(symbol=symbol,date=stock_dates,open=10.,high=11.,low=9.,close=10.,
            volume=4000000.,amount=10000000. if symbol=='600001.SH' else 40000000.,source='sina_unadjusted_v1',has_trade_price=1))
        frame.attrs=dict(volume_unit='shares',amount_unit='CNY',data_version=stable_hash(symbol),price_basis='unadjusted',
            legacy_fuyao_bars=0,unverified_trade_price_bars=0)
        daily=daily_eligibility(frame,dates,policy,start=values['feature_start'],end=dates[-1]);daily.insert(0,'symbol',symbol)
        target=raw_root/'data_snapshots/liquidity'/f'{symbol.replace(".","_")}.parquet';target.parent.mkdir(parents=True,exist_ok=True)
        daily.attrs={};daily.to_parquet(target,index=False)
        members.append(dict(symbol=symbol,path=str(target),sha256=_file_hash(target),rows=len(daily)))
        daily_by_symbol[symbol]=daily;frames[symbol]=frame
        inputs.append(dict(symbol=symbol,raw_content_sha256=_raw_content_hash(frame)))
        if not daily.liquidity_pool_eligible.any():continue
        raw=annotate_liquidity(frame,dates,policy,end=dates[-1]);raw.attrs={}
        stock_dir=raw_root/'data_snapshots/raw'/symbol.replace('.','_');stock_dir.mkdir(parents=True)
        raw_path=stock_dir/'raw.parquet';raw.to_parquet(raw_path,index=False)
        member_path=stock_dir/'liquidity.parquet';shutil.copy2(target,member_path)
        attrs=dict(price_basis='unadjusted',legacy_fuyao_bars=0,unverified_trade_price_bars=0,
            volume_unit='shares',amount_unit='CNY',data_version=stable_hash(symbol))
        record=dict(symbol=symbol,raw_path=str(raw_path),raw_sha256=_file_hash(raw_path),raw_rows=len(raw),
            raw_content_sha256=_raw_content_hash(raw),raw_series_hash=stable_hash(symbol),raw_attrs=attrs,
            data_end=stock_dates[-1],liquidity_path=str(member_path),liquidity_sha256=_file_hash(member_path),
            liquidity_rows=len(daily),liquidity_policy_hash=stable_hash(policy))
        raw_records.append(record)
        native_dir=native_root/'data_snapshots/native'/symbol.replace('.','_');native_dir.mkdir(parents=True)
        features=native_dir/'features.parquet';raw.to_parquet(features,index=False)
        native_file=native_dir/'native.json.gz'
        with gzip.open(native_file,'wt') as handle:json.dump(dict(config_hash=input_contract['native_config_hash'],
            schema_hash=input_contract['native_schema_hash'],events=[],evaluations=[],wyckoff_events=[],auxiliary_evaluations=[]),handle)
        records.append(dict(**record,path=str(features),sha256=_file_hash(features),rows=len(raw),
            native_path=str(native_file),native_sha256=_file_hash(native_file),native_encoding='gzip_json',
            native_schema_hash=input_contract['native_schema_hash'],native_config_hash=input_contract['native_config_hash'],
            raw_only_rebuild=True,reused_model_or_label=False))
    symbols=[r['symbol'] for r in records]; task_dates=daily_by_symbol[candidates[0]].date.tolist()
    daily=[dict(date=day,symbols=[s for s in candidates if daily_by_symbol[s].set_index('date').loc[day,'liquidity_pool_eligible']]) for day in task_dates]
    population=dict(population_mode='full_liquidity_union',feature_start=values['feature_start'],end=dates[-1],policy=policy,
        policy_hash=stable_hash(policy),calendar_run_id='calendar-input',calendar_hash=stable_hash(dates),
        basis_mode='warehouse_observed_mainboard',candidate_symbols=candidates,candidate_count=len(candidates),
        selected_symbols=symbols,selected_count=len(symbols),latest_symbols=daily[-1]['symbols'],latest_count=len(daily[-1]['symbols']),
        daily_counts=[dict(date=row['date'],eligible_count=len(row['symbols']),excluded_count=len(candidates)-len(row['symbols'])) for row in daily],
        input_catalog=inputs,input_catalog_hash=stable_hash(inputs),daily_membership_hash=stable_hash(daily),
        stock_summaries=[dict(symbol=s) for s in candidates],membership_assets=members)
    population['population_hash']=stable_hash(_population_body(population))
    summaries=[]
    for name,value in [('liquidity_stocks.csv',population['stock_summaries']),('liquidity_daily_counts.csv',population['daily_counts'])]:
        path=raw_root/'data_snapshots'/name;pd.DataFrame(value).to_csv(path,index=False);summaries.append(dict(path=str(path),sha256=_file_hash(path)))
    population['summary_assets']=summaries
    _save(raw_root/'reports/liquidity_universe.json',population)
    raw_report=dict(run_id='raw-input',strategy_version='trend_consensus',status='completed',failures=[],end=dates[-1],data_end=dates[-1],
        raw_records=raw_records,selected_symbols=symbols,calendar_run_id='calendar-input',calendar_hash=stable_hash(dates),dates=dates,
        liquidity=population,liquidity_universe=population,population_mode='full_liquidity_union',raw_only_rebuild=True,reused_model_or_label=False)
    raw_report['snapshot_hash']=stable_hash(raw_report);_save(raw_root/'reports/raw_source.json',raw_report)
    snapshot=dict(raw_source_run_id='raw-input',raw_snapshot_hash=raw_report['snapshot_hash'],liquidity=population,
        code_hashes={r:_file_hash(code/r) for r in NATIVE_GENERATING_FILES},calendar_run_id='calendar-input',calendar_hash=stable_hash(dates),
        raw_input_hashes={r['symbol']:r['raw_sha256'] for r in records})
    native_report=dict(run_id='native-input',strategy_version='trend_consensus',status='completed',failures=[],data_end=dates[-1],config=config,
        dataset_records=records,selected_symbols=symbols,liquidity=population,liquidity_universe=population,source_snapshot=snapshot,
        population_mode='full_liquidity_union',raw_only_rebuild=True,reused_model_or_label=False,calendar_dates=dates)
    _save(native_root/'reports/native_source.json',native_report)
    for root in [raw_root,native_root]:_save(root/'data_snapshots/calendar.json',dict(verified=True,source_run_id='calendar-input',dates=dates))
    store=ResultStore(tmp_path/'results.db',runs); service=center.TrainingCenter(store=store,metadata_root=tmp_path/'metadata')
    return dict(service=service,values=values,config=config,native=native_report,raw=raw_report,population=population,
        native_root=native_root,raw_root=raw_root,code=code,runs=runs,dates=dates)


def preflight(source):
    result=source['service'].preflight(source['values'],source_native_run_id='native-input')
    assert result['valid'],result['errors']
    return result['plan']


def test_real_source_preflight_complete_hashes_and_historical_exit(source):
    events=[];result=source['service'].preflight(source['values'],source_native_run_id='native-input',progress=events.append)
    assert result['valid'],result['errors'];plan=result['plan'];proof=plan['source_reuse']
    assert proof['selected_count']==3 and proof['native_report_sha256']==_file_hash(source['native_root']/'reports/native_source.json')
    assert len(proof['asset_catalog'])==len({asset['path'] for asset in proof['asset_catalog']})
    assert all(_file_hash(source['runs']/asset['path'])==asset['sha256'] for asset in proof['asset_catalog'])
    assert proof['asset_catalog_hash']==stable_hash(proof['asset_catalog'])
    assert '600000.SH' in plan['cohort'] and '600000.SH' not in plan['liquidity_universe']['latest_symbols']
    assert plan['date_status']['code']=='verified_frozen_source_reuse' and events[-1]['completed']==events[-1]['total']
    assert source['service'].load_plan(plan['plan_id'],claim=True)==plan
    with pytest.raises(center.RevisionConflict):source['service'].load_plan(plan['plan_id'],claim=True)


@pytest.mark.parametrize('which',['raw','native','feature','record_member','member','summary','calendar'])
def test_actual_asset_tamper_after_preflight_is_rejected(source,which):
    plan=preflight(source);record=source['native']['dataset_records'][0]
    paths={'raw':record['raw_path'],'native':record['native_path'],'feature':record['path'],
        'record_member':record['liquidity_path'],'member':source['population']['membership_assets'][0]['path'],
        'summary':source['population']['summary_assets'][0]['path'],
        'calendar':source['runs']/'calendar-input/data_snapshots/source.json'}
    path=Path(paths[which]);path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises((ValueError,OSError)):
        source['service'].load_plan(plan['plan_id'],claim=True)
    with sqlite3.connect(source['service'].db_path) as conn:
        assert conn.execute('SELECT claimed FROM training_plans WHERE plan_id=?',(plan['plan_id'],)).fetchone()[0]==0


@pytest.mark.parametrize('which',['incomplete','model_import','slice','source_schema','population'])
def test_bad_source_metadata_does_not_get_a_valid_plan(source,which):
    path=source['native_root']/'reports/native_source.json';native=copy.deepcopy(source['native'])
    if which=='incomplete':native['status']='running'
    elif which=='model_import':native['reused_model_or_label']=True
    elif which=='slice':native['dataset_records']=native['dataset_records'][:-1]
    elif which=='source_schema':native['dataset_records'][0]['native_config_hash']='0'*64
    else:native['liquidity']={**native['liquidity'],'selected_count':5000}
    _save(path,native)
    result=source['service'].preflight(source['values'],source_native_run_id='native-input')
    assert not result['valid'] and result['errors']


def test_new_plan_rejects_later_model_code_change_and_native_generator_change(source):
    plan=preflight(source)
    path=source['code']/'services/czsc_trend_consensus.py';path.write_bytes(path.read_bytes()+b'\n# changed\n')
    with pytest.raises(ValueError,match='生成代码'):source['service'].validate_plan(plan['plan_id'])
    path=source['code']/'services/czsc_market_features.py';path.write_bytes(path.read_bytes()+b'\n# changed\n')
    result=source['service'].preflight(source['values'],source_native_run_id='native-input')
    assert not result['valid'] and any('生成代码' in reason for reason in result['errors'])


def test_source_reuse_rejects_new_end_changed_window_and_explicit_slice(source):
    for changed in [{'end':'2027-04-02'},{'feature_start':'2027-01-04'},{'symbols':['000001.SZ']}]:
        result=source['service'].preflight({**source['values'],**changed},source_native_run_id='native-input')
        assert not result['valid'],changed


def test_claim_race_only_one_wins_real_sqlite(source):
    plan=preflight(source);results=[]
    def claim():
        try:source['service'].load_plan(plan['plan_id'],claim=True);results.append('claimed')
        except center.RevisionConflict:results.append('duplicate')
    threads=[threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    assert sorted(results)==['claimed','duplicate']


def test_public_default_and_submit_do_not_accept_source_override(source):
    assert center.DEFAULTS['device']=='cpu'
    assert 'source_native_run_id' not in source['service'].profiles()['default_profile']['parameters']
    assert api.TrainingPreflightRequest(parameters={},source_native_run_id='native-input').source_native_run_id=='native-input'
    with pytest.raises(ValidationError):api.DefaultProfileRequest(parameters={},expected_revision=0,source_native_run_id='native-input')
    with pytest.raises(ValidationError):api.TrainingJobRequest(plan_id='new-plan',source_native_run_id='native-input')


def test_http_preflight_uses_actual_assets(source):
    app=FastAPI();app.include_router(api.router);app.state.training=source['service']
    with TestClient(app) as client:
        result=client.post('/api/training/preflight',json={'parameters':source['values'],'source_native_run_id':'native-input'})
        assert result.status_code==200 and result.json()['valid']
        assert result.json()['plan']['source_reuse']['selected_count']==3
        assert client.post('/api/training/jobs',json={'plan_id':'other','source_native_run_id':'native-input'}).status_code==422


def test_process_events_use_frozen_total_real_devices_and_new_model_id(source,tmp_path):
    plan=preflight(source);events=[];proof=plan['source_reuse']
    lines=['TRAIN_RUN_ID new-model', 'POLICY_STOCK_DONE '+json.dumps(dict(current=1,total=3,device='cpu',symbol='000001.SZ',fold='policy',failed=0)),
        'HEAD '+json.dumps(dict(fold='production',head='entry',architecture='mlp',device='mps',training_resources={'actual_device':'mps'})),
        'DATASET E0 000001.SZ entries=1 exits=0',
        'PROGRESS '+json.dumps(dict(stage='dataset',population='E0',fold=None,completed=1,total=3,scope='current_population_only'))]
    command=[sys.executable,'-u','-c','\n'.join('print('+repr(line)+',flush=True)' for line in lines)]
    ids=center.TrainingCenter._run_process(command,tmp_path/'training.log',events.append,None,
        initial_ids={'raw_source_run_id':'raw-input','source_native_run_id':'native-input'},source_total=3,source_reuse=proof)
    assert ids==dict(raw_source_run_id='raw-input',source_native_run_id='native-input',model_run_id='new-model')
    assert next(e for e in events if e['stage']=='independent_policy_selection')['actual_device']=='cpu'
    assert next(e for e in events if e['stage']=='independent_policy_selection')['device']=='cpu'
    head=next(e for e in events if e['stage']=='model_head');assert head['actual_device']==head['device']=='mps'
    assert next(e for e in events if e['stage']=='E0')['total']==3
    assert (tmp_path/'training.log').read_text().startswith('SOURCE_REUSE_VERIFIED ')


def test_source_prepare_reuse_creates_new_parent_without_build_or_old_heads(source,monkeypatch):
    from my_strategy.core import paths
    monkeypatch.setattr(paths,'ARTIFACT_RUNS_ROOT',source['runs'])
    plan=preflight(source);commands=[]
    def process(args,path,progress,check_cancel,**kwargs):
        commands.append((args,kwargs));return kwargs['initial_ids']
    monkeypatch.setattr(source['service'],'_run_process',process)
    result=source['service'].run_training_job(dict(plan_id=plan['plan_id'],stage='source_prepare'))
    assert result['run_id'] not in {'native-input','raw-input','old-partial-model'}
    assert result['status']=='source_prepared' and result['source_reused'] is True and result['reused_model_or_label'] is False
    args,kwargs=commands[0]
    assert args[args.index('--source-native-run-id')+1]=='native-input' and '--prepare-source-only' in args
    assert kwargs['source_total']==3 and not any('old-partial' in str(arg) for arg in args)
    assert source['native_root'].exists() and source['raw_root'].exists()


def test_existing_true_constraints_are_retained_for_source_reuse(source):
    for weakened in [{'min_net_return':.079},{'min_r_multiple':1.99},{'liquidity_min_median_amount':29999999},
                     {'liquidity_lookback_sessions':19},{'min_calibration_rows':29},{'commission':0}]:
        with pytest.raises(ValueError):source['service'].preflight({**source['values'],**weakened},source_native_run_id='native-input')
    windows=[]
    result=center.selection_gate(windows,dict(min_stocks=60,min_round_trips_per_window=50,min_round_trips_total=200,drawdown_tolerance=.02),3,False)
    assert result['passed'] is False


def trained_contract_fixture(source,plan):
    from my_strategy.services.czsc_trend_pipeline import POPULATION_VERSION,FULL_CAPACITY_CRITERION
    actual=copy.deepcopy(plan['config']);proof=plan['source_reuse']
    snapshot=dict(source_native_run_id=proof['source_native_run_id'],source_report_sha256=proof['native_report_sha256'],
        records_hash=proof['records_hash'],selected_symbols=plan['cohort'],liquidity=plan['liquidity_universe'],
        source_snapshot=source['native']['source_snapshot'])
    snapshot['snapshot_hash']=stable_hash(snapshot)
    actual.update(cpu_workers=plan['parameters']['cpu_workers'],training_end=plan['data_end'],
        capacity_expansion_criterion=FULL_CAPACITY_CRITERION,source_snapshot_hash=snapshot['snapshot_hash'])
    registered=copy.deepcopy(actual)
    actual['pipeline_contract_hash']=stable_hash(dict(family='trend_consensus',source=snapshot,
        calendar_hash=proof['calendar_hash'],registered_config=registered,
        order=['independent_trend','policy_only_sources','E0_no_HR','MA_exit_E0','pinned_E1',
            'preregistered_actual_E1_holding_pairs','account_free_reference_quality_probes',
            'independent_reference_quality_heads_and_fee_qualification','strict_prior_L2_actual_flat_population_by_fold',
            'L2_entry_forced_E1_clone','L2_exit_preregistered_actual_E1_pairs'],
        reference_population_protocol=POPULATION_VERSION,self_training_population_feedback=False,
        reassessment=registered['reassessment'],no_E2_entry_reuse=True))
    actual['candidate_policy_hash']=stable_hash(dict(candidate=actual['candidate_policy'],risk_policy=actual['risk_policy'],
        trend_candidate={k:actual['trend'][k] for k in ('candidate_cross_threshold','candidate_margin')},
        consensus=actual['consensus'],signal_points=actual['signal_points'],native_records=proof['records_hash'],
        reassessment=actual['reassessment'],training_population=POPULATION_VERSION))
    actual['E0_policy_hash']=stable_hash(dict(stage='E0',candidate_policy_hash=actual['candidate_policy_hash'],
        account=actual['account'],trend_catalog_hash=stable_hash([]),ordinary_sell='continuous_sell_pass_for_actual_holding',
        reassessment=actual['reassessment'],entry_HR=False))
    actual['E1_policy_hash']=stable_hash(dict(stage='E1',E0=actual['E0_policy_hash'],signal_time_pinned_MA_exit=stable_hash([]),
        holding_threshold=actual['account']['holding_exit_threshold'],no_head='entire_position_E0',checkpoint_upgrade=False,L2_exit_apply=False))
    from my_strategy.services.czsc_trend_contracts import (
        HELD_TRAINING_POPULATION,HELD_CONDITION_DOMAIN,HELD_SPLIT_POLICY,HELD_ECONOMIC_WEIGHTING)
    held=dict(version=HELD_TRAINING_POPULATION,stage='E1',reference_mode='E1',
        conditional_holding_domain=HELD_CONDITION_DOMAIN,pending_L2_entry_or_exit_used=False,
        candidate_clones_are_holding_population=False,holding_checkpoint_upgrade=False,
        split_policy=HELD_SPLIT_POLICY,economic_weighting=HELD_ECONOMIC_WEIGHTING,
        source_snapshot_hash=actual['source_snapshot_hash'],candidate_policy_hash=actual['candidate_policy_hash'],
        continuation_policy_hash=actual['E1_policy_hash'],manifest_path='dataset/E1/manifest.json',
        manifest_sha256='test-held-manifest',pooled_exits_path='dataset/E1/exits.parquet',
        pooled_exits_sha256='test-held-exits',records_hash='test-held-records')
    return dict(strategy_version='trend_consensus',status='completed',data_end=plan['data_end'],config=actual,
        dataset_records=source['native']['dataset_records'],source_snapshot=snapshot,trend_bundles=[],ma_exit_bundles=[],
        holding_population=held,
        protocol_freeze={**plan['training_code_freeze'],'holding_population_hash':stable_hash(held)},
        training_resources=dict(actual_device='mps'))


@pytest.mark.parametrize('change',['native_id','report_sha','records_hash','raw_id','population_hash',
    'generator_code','reference_generator_code','coverage'])
def test_post_training_source_binding_cannot_use_another_source_or_old_identity(source,change):
    plan=preflight(source);report=trained_contract_fixture(source,plan)
    center.TrainingCenter._validate_trained_plan(report,plan)
    report=copy.deepcopy(report)
    if change=='native_id':report['source_snapshot']['source_native_run_id']='wrong-native'
    elif change=='report_sha':report['source_snapshot']['source_report_sha256']='0'*64
    elif change=='records_hash':report['source_snapshot']['records_hash']='0'*64
    elif change=='raw_id':report['source_snapshot']['source_snapshot']['raw_source_run_id']='wrong-raw'
    elif change=='population_hash':report['source_snapshot']['liquidity']['population_hash']='0'*64
    elif change=='generator_code':report['protocol_freeze']['code_hashes']['services/czsc_trend_account.py']='0'*64
    elif change=='reference_generator_code':report['protocol_freeze']['code_hashes']['services/czsc_reference_quality.py']='0'*64
    else:report['dataset_records']=report['dataset_records'][:-1]
    with pytest.raises(ValueError):center.TrainingCenter._validate_trained_plan(report,plan)


@pytest.mark.parametrize('change',['missing','hash','gated-account','source','continuation'])
def test_completed_training_binds_actual_E1_holding_population_without_new_head_gate(source,change):
    plan=preflight(source);report=trained_contract_fixture(source,plan)
    # Existing head readiness gates still belong to _model. A frozen report
    # can be validated here without claiming an optional exit head passed.
    assert not report.get('model_bundles')
    center.TrainingCenter._validate_trained_plan(report,plan)
    if change=='missing':report.pop('holding_population')
    elif change=='hash':report['protocol_freeze']['holding_population_hash']='wrong'
    elif change=='gated-account':report['holding_population']['reference_mode']='final'
    elif change=='source':report['holding_population']['source_snapshot_hash']='foreign'
    else:report['holding_population']['continuation_policy_hash']='another-E1'
    with pytest.raises(ValueError,match='holding|E1|持仓'):
        center.TrainingCenter._validate_trained_plan(report,plan)


@pytest.mark.parametrize('change',['native_none','native_list','snapshot_none','records_none','record_none','config_none',
    'raw_none','raw_records_none','raw_record_none','population_none','population_policy_list','members_none'])
def test_malformed_external_source_shapes_return_invalid_preflight_http200(source,change):
    native=copy.deepcopy(source['native']);raw=copy.deepcopy(source['raw'])
    if change=='native_none':native=None
    elif change=='native_list':native=[]
    elif change=='snapshot_none':native['source_snapshot']=None
    elif change=='records_none':native['dataset_records']=None
    elif change=='record_none':native['dataset_records']=[None]
    elif change=='config_none':native['config']=None
    elif change=='raw_none':raw=None
    elif change=='raw_records_none':raw['raw_records']=None
    elif change=='raw_record_none':raw['raw_records']=[None]
    elif change=='population_none':native['liquidity']=None
    elif change=='population_policy_list':native['liquidity']['policy']=[]
    else:native['liquidity']['membership_assets']=None
    _save(source['native_root']/'reports/native_source.json',native)
    _save(source['raw_root']/'reports/raw_source.json',raw)
    app=FastAPI();app.include_router(api.router);app.state.training=source['service']
    with TestClient(app) as client:
        response=client.post('/api/training/preflight',json={'parameters':source['values'],'source_native_run_id':'native-input'})
        assert response.status_code==200
        assert response.json()['valid'] is False and response.json()['errors']
