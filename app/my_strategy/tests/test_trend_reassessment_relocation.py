"""A target-only receipt must catch missing, changed and escaping frozen data."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.scripts import verify_trend_reassessment_relocation as tool
from my_strategy.scripts.migration_manifest import file_sha256
from my_strategy.services import czsc_trend_v2_pipeline as pipeline


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else json.dumps(value,sort_keys=True,indent=2).encode())
    return file_sha256(path)


def complete_fixture(tmp_path):
    old=tmp_path/'origin/runs';target=tmp_path/'new/runs'
    run_id='model-v2';raw_id='raw-source';native_id='native-source';cal_id='verified-calendar'
    symbol='000001.SZ';dates=['2026-01-02','2026-01-05','2026-01-06']
    raw_path=old/raw_id/'data_snapshots/raw/000001_SZ/raw.parquet'
    raw_sha=write(raw_path,b'frozen raw observation bytes')
    raw_record={'symbol':symbol,'raw_path':str(raw_path),'raw_sha256':raw_sha,'raw_series_hash':'original-series',
        'raw_attrs':{'price_basis':'unadjusted','legacy_fuyao_bars':0,'unverified_trade_price_bars':0,
            'volume_unit':'shares','amount_unit':'CNY','data_version':'original-series'}}
    raw={'run_id':raw_id,'strategy_version':tool.FAMILY,'status':'completed','failures':[],
        'raw_only_rebuild':True,'reused_model_or_label':False,'selected_symbols':[symbol],
        'raw_records':[raw_record],'dates':dates,'calendar_run_id':cal_id,'calendar_hash':stable_hash(dates)}
    raw['snapshot_hash']=stable_hash(raw)
    write(old/raw_id/'reports/raw_source.json',raw)
    record={**raw_record,'raw_only_rebuild':True,'reused_model_or_label':False}
    for key,name,digest in [('path','features.parquet','sha256'),('native_path','native.json','native_sha256')]:
        path=old/native_id/'data_snapshots/native/000001_SZ'/name
        record.update({key:str(path),digest:write(path,b'frozen native '+name.encode())})
    source_snapshot={'raw_source_run_id':raw_id,'raw_snapshot_hash':raw['snapshot_hash'],
        'raw_input_hashes':{symbol:raw_sha},'calendar_run_id':cal_id,'calendar_hash':stable_hash(dates),
        'code_hashes':{name:file_sha256(pipeline.MY_STRATEGY_ROOT/name) for name in tool.SOURCE_GENERATORS}}
    native={'run_id':native_id,'strategy_version':tool.FAMILY,'status':'completed','data_end':dates[-1],
        'raw_only_rebuild':True,'reused_model_or_label':False,'dataset_records':[record],
        'source_snapshot':source_snapshot,'calendar_dates':dates}
    native_sha=write(old/native_id/'reports/native_v4.json',native)
    write(old/cal_id/'reports/calendar.json',{'run_id':cal_id,'verified':True,'source':'verified synthetic test','dates':dates})
    write(old/run_id/'data_snapshots/calendar.json',{'verified':True,'source_run_id':cal_id,'dates':dates})
    population=pipeline.make_reference_population({'name':'production','policy_end':dates[-1],'calibration_start':dates[0]},[])
    manifests=[];audits=[]
    for stage in ['E0','E1','final_population/production']:
        stock={'symbol':symbol,'files':{}}
        for name in tool.STOCK_ASSETS:
            path=old/run_id/'dataset'/stage/'000001_SZ'/name
            stock['files'][name]={'path':str(path),'sha256':write(path,b'actual frozen label/account '+name.encode())}
        write(old/run_id/'dataset'/stage/'000001_SZ/manifest.json',stock)
        pooled={}
        for key in ['entries','exits','episodes']:
            path=old/run_id/'dataset'/stage/(key+'.parquet')
            pooled[key]={'path':str(path),'sha256':write(path,b'pooled frozen '+key.encode())}
        manifest={'records':[stock],'failures':[],'pooled_files':pooled,
            'reference_population':population if stage.startswith('final_population') else {}}
        relative='dataset/'+stage+'/manifest.json'
        sha=write(old/run_id/relative,manifest)
        manifests.append({'stage':stage,'path':relative,'sha256':sha})
        audits.append({'stage':stage,'stocks':1,'files':6,'manifest_sha256':sha})
    populations=[{'fold':'production','manifest_path':manifests[-1]['path'],
        'manifest_sha256':manifests[-1]['sha256'],'reference_population':population}]
    source={'source_native_run_id':native_id,'source_report_sha256':native_sha,
        'source_snapshot':source_snapshot,'records_hash':stable_hash([record])}
    source['snapshot_hash']=stable_hash(source)
    config={'version':tool.FAMILY,'source_snapshot_hash':source['snapshot_hash']}
    freeze={'code_hashes':{name:file_sha256(pipeline.MY_STRATEGY_ROOT/name) for name in pipeline.GENERATING_FILES},
        'config_hash':stable_hash(config),'models_catalog_hash':stable_hash({'trend':[],'ma_exit':[],'architecture':[]}),
        'population_folds_hash':stable_hash(populations),'dataset_manifests_hash':stable_hash(manifests)}
    freeze['freeze_hash']=stable_hash(freeze)
    report={'run_id':run_id,'strategy_version':tool.FAMILY,'status':'completed','config':config,'data_end':dates[-1],
        'source_snapshot':source,'dataset_records':[record],'trend_bundles':[],'ma_exit_bundles':[],
        'model_bundles':[],'architecture_bundles':[],'population_folds':populations,'dataset_manifests':manifests,
        'protocol_freeze':freeze,'required_recent_heads_ready':False,'publication_allowed':False,'model_gate':{'passed':False}}
    report_sha=write(old/run_id/'reports/trend_consensus.json',report)
    audit_path=tmp_path/'completion-audit.json'
    write(audit_path,{'run_id':run_id,'strategy_version':tool.FAMILY,'status':'completed',
        'completion_audit':{'read_only':True,'family':tool.FAMILY,'model_population_DAG_verified':True,
            'report_sha256':report_sha,'dataset_assets':audits}})
    origin=tmp_path/'origin-receipt.json'
    tool.make_origin_receipt(original_runs_root=old,model_run_id=run_id,completion_audit=audit_path,output=origin)
    shutil.copytree(old,target)
    return {'old':old,'target':target,'run_id':run_id,'origin':origin,'audit':audit_path,
        'raw_id':raw_id,'native_id':native_id,'calendar_id':cal_id}


def verify(fixture,output,**changes):
    args=dict(original_runs_root=fixture['old'],target_runs_root=fixture['target'],
        model_run_id=fixture['run_id'],origin_receipt=fixture['origin'],output=output)
    args.update(changes)
    return tool.verify_relocation(**args)


def test_complete_target_verifies_after_original_root_is_unavailable(tmp_path):
    fixture=complete_fixture(tmp_path)
    fixture['old'].rename(tmp_path/'original-offline')
    certificate=verify(fixture,tmp_path/'certificate.json')
    assert certificate['complete_state_verified'] and certificate['old_root_access_denied']
    assert certificate['old_root_open_attempts']==[] and certificate['verified_asset_count']==41
    assert certificate['source_dependencies']['source_symbols']==1
    assert not certificate['required_recent_heads_ready'] and not certificate['publication_allowed']
    assert certificate['legacy_finalize_relocation_supported'] is False
    with pytest.raises(FileExistsError,match='overwrite'):
        verify(fixture,tmp_path/'certificate.json')


@pytest.mark.parametrize('relative',[
    'model-v2/dataset/E1/000001_SZ/candidate_clones.json',
    'model-v2/dataset/final_population/production/entries.parquet',
    'native-source/data_snapshots/native/000001_SZ/native.json',
    'raw-source/data_snapshots/raw/000001_SZ/raw.parquet'])
def test_target_byte_drift_cannot_receive_certificate(tmp_path,relative):
    fixture=complete_fixture(tmp_path);(fixture['target']/relative).write_bytes(b'changed')
    output=tmp_path/'must-not-exist.json'
    with pytest.raises(ValueError,match='drift'):
        verify(fixture,output)
    assert not output.exists()


def test_missing_source_dependency_is_not_partial_success(tmp_path):
    fixture=complete_fixture(tmp_path);shutil.rmtree(fixture['target']/fixture['native_id'])
    with pytest.raises(ValueError,match='missing'):
        verify(fixture,tmp_path/'missing.json')


def test_symlink_escape_is_rejected_even_for_identical_bytes(tmp_path):
    fixture=complete_fixture(tmp_path)
    path=fixture['target']/'model-v2/dataset/E0/000001_SZ/entries.parquet'
    outside=tmp_path/'outside.parquet';outside.write_bytes(path.read_bytes());path.unlink();path.symlink_to(outside)
    with pytest.raises(ValueError,match='linked'):
        verify(fixture,tmp_path/'escape.json')


def test_receipt_mapping_and_inventory_omissions_are_rejected(tmp_path):
    fixture=complete_fixture(tmp_path)
    value=json.loads(fixture['origin'].read_text());value['inventory']=[row for row in value['inventory'] if not row['path'].endswith('candidate_clones.json')]
    value['evidence_hash']=stable_hash({k:v for k,v in value.items() if k!='evidence_hash'})
    fixture['origin'].write_text(json.dumps(value))
    with pytest.raises(ValueError,match='omits mandatory'):
        verify(fixture,tmp_path/'omitted.json')
    with pytest.raises(ValueError,match='root mapping'):
        verify(fixture,tmp_path/'bad-root.json',original_runs_root=tmp_path/'other-old-root')


def test_unexpected_member_and_wrong_original_completion_are_rejected(tmp_path):
    fixture=complete_fixture(tmp_path);write(fixture['target']/'raw-source/untracked.json',{})
    with pytest.raises(ValueError,match='unexpected'):
        verify(fixture,tmp_path/'extra.json')
    value=json.loads(fixture['origin'].read_text());value['original_completion_receipt']['completion_audit']['report_sha256']='0'*64
    value['evidence_hash']=stable_hash({k:v for k,v in value.items() if k!='evidence_hash'})
    fixture['origin'].write_text(json.dumps(value))
    with pytest.raises(ValueError,match='completion receipt'):
        verify(fixture,tmp_path/'wrong-audit.json')


def test_runtime_only_does_not_certify_missing_input_and_label_archive(tmp_path):
    fixture=complete_fixture(tmp_path);shutil.rmtree(fixture['target']/fixture['native_id'])
    (fixture['target']/'model-v2/dataset/E0/000001_SZ/entries.parquet').unlink()
    certificate=verify(fixture,tmp_path/'runtime-only.json',scope='runtime-only')
    assert not certificate['complete_state_verified'] and certificate['scope']=='runtime-only'
    assert certificate['source_dependencies']=={}


def test_evidence_cannot_be_written_inside_frozen_artifacts(tmp_path):
    fixture=complete_fixture(tmp_path)
    with pytest.raises(ValueError,match='outside frozen'):
        verify(fixture,fixture['target']/'model-v2/new-certificate.json')


def test_raw_source_manifest_byte_change_is_detected_even_if_semantics_match(tmp_path):
    fixture=complete_fixture(tmp_path)
    path=fixture['target']/'raw-source/reports/raw_source.json';path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='byte drift'):
        verify(fixture,tmp_path/'whitespace-drift.json')


def test_mapping_rejects_an_audited_original_asset_outside_approved_root(tmp_path):
    fixture=complete_fixture(tmp_path)
    parent=fixture['old']/'model-v2/dataset/E0/manifest.json';manifest=json.loads(parent.read_text())
    manifest['records'][0]['files']['entries.parquet']['path']=str(tmp_path/'outside/entries.parquet')
    write(fixture['old']/'model-v2/dataset/E0/000001_SZ/manifest.json',manifest['records'][0])
    changed_sha=write(parent,manifest)
    report_path=fixture['old']/'model-v2/reports/trend_consensus.json';report=json.loads(report_path.read_text())
    report['dataset_manifests'][0]['sha256']=changed_sha
    report['protocol_freeze']['dataset_manifests_hash']=stable_hash(report['dataset_manifests'])
    report['protocol_freeze']['freeze_hash']=stable_hash({k:v for k,v in report['protocol_freeze'].items() if k!='freeze_hash'})
    report_sha=write(report_path,report)
    audit=json.loads(fixture['audit'].read_text());audit['completion_audit']['report_sha256']=report_sha
    audit['completion_audit']['dataset_assets'][0]['manifest_sha256']=changed_sha;write(fixture['audit'],audit)
    with pytest.raises(ValueError,match='outside approved old root'):
        tool.make_origin_receipt(original_runs_root=fixture['old'],model_run_id=fixture['run_id'],
            completion_audit=fixture['audit'],output=tmp_path/'unsafe-origin.json')


def test_calendar_source_artifacts_and_parent_provenance_are_mandatory(tmp_path):
    fixture=complete_fixture(tmp_path);old=fixture['old'];calendar=old/fixture['calendar_id']
    provider=calendar/'data_snapshots/provider.txt';provider_sha=write(provider,b'verified provider bytes')
    parent_id='earlier-calendar';parent_path=old/parent_id/'reports/calendar.json'
    parent={'run_id':parent_id,'verified':True,'source':'independent original calendar','dates':['2026-01-02','2026-01-05']}
    parent_sha=write(parent_path,parent)
    current=json.loads((calendar/'reports/calendar.json').read_text())
    current.update(source_artifacts=[{'path':str(provider),'sha256':provider_sha}],
        parent_calendar={'run_id':parent_id,'path':str(parent_path),'file_sha256':parent_sha,'hash':stable_hash(parent)})
    write(calendar/'reports/calendar.json',current)
    fixture['origin']=tmp_path/'origin-with-parent.json'
    tool.make_origin_receipt(original_runs_root=old,model_run_id=fixture['run_id'],completion_audit=fixture['audit'],output=fixture['origin'])
    shutil.copytree(calendar,fixture['target']/fixture['calendar_id'],dirs_exist_ok=True)
    shutil.copytree(old/parent_id,fixture['target']/parent_id)
    certificate=verify(fixture,tmp_path/'calendar-complete.json')
    assert certificate['source_dependencies']['calendar_provenance_run_ids']==['earlier-calendar','verified-calendar']
    (fixture['target']/fixture['calendar_id']/'data_snapshots/provider.txt').write_bytes(b'changed provider')
    with pytest.raises(ValueError,match='byte drift'):
        verify(fixture,tmp_path/'provider-drift.json')
    (fixture['target']/fixture['calendar_id']/'data_snapshots/provider.txt').write_bytes(b'verified provider bytes')
    shutil.rmtree(fixture['target']/parent_id)
    with pytest.raises(ValueError,match='missing'):
        verify(fixture,tmp_path/'missing-parent.json')


def test_runtime_receipt_cannot_sneak_in_an_old_root_read(tmp_path):
    fixture=complete_fixture(tmp_path);inside=fixture['old']/'old-completion.json'
    shutil.copy2(fixture['audit'],inside)
    with pytest.raises(ValueError,match='external'):
        verify(fixture,tmp_path/'inside.json',origin_receipt=None,completion_audit=inside,scope='runtime-only')
