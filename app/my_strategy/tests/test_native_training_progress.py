"""Raw-only reuse and truthful native preparation progress."""
import json
from pathlib import Path
import pytest
import pyarrow.parquet as pq
from fastapi.testclient import TestClient
from fastapi import FastAPI
from my_strategy.tests.test_current_source_reuse import source
from my_strategy.tests.test_current_source_reuse import preflight as native_preflight, trained_contract_fixture
from my_strategy.services.czsc_trend_support import _save
from my_strategy.core.run_context import stable_hash
from my_strategy.web_dashboard import api


def complete_raw(source):
    report = source['raw']
    for row in report['raw_records']:
        row['raw_columns'] = pq.read_schema(row['raw_path']).names
    report['snapshot_hash'] = stable_hash({k:v for k,v in report.items() if k != 'snapshot_hash'})
    _save(source['raw_root'] / 'reports/raw_source.json', report)
    return report


def test_raw_only_reuse_does_not_read_native_or_models(source):
    complete_raw(source)
    (source['native_root'] / 'reports/native_source.json').unlink()
    response = source['service'].preflight(source['values'], raw_source_run_id='raw-input')
    assert response['valid'], response['errors']
    plan = response['plan']
    assert plan['cohort'] == source['raw']['selected_symbols']
    assert plan['source_reuse'] is None
    assert plan['raw_source_reuse']['reused_model_or_label'] is False
    assert plan['raw_source_reuse']['selected_count'] == len(plan['cohort'])
    source['service']._verify_execution_plan(plan)


def test_raw_reuse_cold_verification_rejects_changed_bytes(source):
    complete_raw(source)
    result = source['service'].preflight(source['values'], raw_source_run_id='raw-input')
    assert result['valid']
    asset = Path(source['raw']['raw_records'][0]['raw_path'])
    asset.write_bytes(asset.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='SHA'):
        source['service']._verify_execution_plan(result['plan'])


def test_raw_reuse_rejects_policy_and_combined_native_request(source):
    complete_raw(source)
    wrong = dict(source['values'], liquidity_min_median_amount=40000000)
    result = source['service'].preflight(wrong, raw_source_run_id='raw-input')
    assert not result['valid']
    mixed = source['service'].preflight(source['values'], raw_source_run_id='raw-input',source_native_run_id='native-input')
    assert not mixed['valid']


def test_preflight_api_routes_raw_and_checkpoint_identities():
    class Service:
        def preflight(self, values, **sources):
            return {'valid':True,'received':sources}
    app = FastAPI(); app.include_router(api.router); app.state.training = Service()
    client = TestClient(app)
    response = client.post('/api/training/preflight',json={'parameters':{},'raw_source_run_id':'raw-input','resume_native_run_id':'new-native'})
    assert response.status_code == 200
    assert response.json()['received'] == {'raw_source_run_id':'raw-input','resume_native_run_id':'new-native'}
    assert client.post('/api/training/preflight',json={'parameters':{},'raw_source_run_id':'../escape'}).status_code == 422


def test_malformed_raw_or_checkpoint_header_is_a_preflight_error(source):
    complete_raw(source)
    _save(source['raw_root'] / 'reports/raw_source.json', [])
    response = source['service'].preflight(source['values'],raw_source_run_id='raw-input')
    assert not response['valid']
    assert any('格式无效' in error for error in response['errors'])
    _save(source['runs'] / 'new-native/reports/native_checkpoint.json', {'identity':[]})
    response = source['service'].preflight(source['values'],resume_native_run_id='new-native')
    assert not response['valid']
    assert any('格式无效' in error for error in response['errors'])


def test_raw_reuse_initializes_children_before_training_subprocess(source,monkeypatch):
    complete_raw(source)
    plan=source['service'].preflight(source['values'],raw_source_run_id='raw-input')['plan']
    assert plan['valid']
    observed={}
    def inspect(args,*positional,**kwargs):
        observed.update(args=args,kwargs=kwargs)
        raise RuntimeError('subprocess inspected')
    monkeypatch.setattr(source['service'],'_run_process',inspect)
    with pytest.raises(RuntimeError,match='subprocess inspected'):
        source['service'].run_training_job({'plan_id':plan['plan_id'],'stage':'source_prepare'})
    assert '--raw-source-run-id' in observed['args']
    assert observed['kwargs']['initial_ids']['raw_source_run_id']=='raw-input'


def test_completed_training_must_bind_the_frozen_raw_reuse_proof(source):
    plan=native_preflight(source)
    report=trained_contract_fixture(source,plan)
    proof=plan['source_reuse']
    plan['raw_source_reuse']={key:proof[key] for key in ('raw_source_run_id','raw_snapshot_hash','population_hash')}
    plan['source_reuse']=None
    source['service']._validate_trained_plan(report,plan)
    plan['raw_source_reuse']['raw_source_run_id']='different-raw'
    with pytest.raises(ValueError,match='原始来源'):
        source['service']._validate_trained_plan(report,plan)
