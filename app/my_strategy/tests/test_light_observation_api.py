"""Saved observation reads preserve all outcomes and never enqueue computation."""
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from my_strategy.core.run_context import stable_hash
from my_strategy.web_dashboard.light_api import router


def fixture():
    population={'source':'frozen','model_path':'/Users/private-user/models/model.safetensors'}
    manifest_id=stable_hash(population)
    value={'schema':'score_observation_manifest_v1','model_independent':True,'source_policy_id':'baseline-v1',
        'frozen_at':'2026-10-15T15:00:00+08:00','symbols':['002436.SZ'],'start':'2025-01-01',
        'end':'2025-12-31','status':'partial','total_calls':3,'batch_count':1,'requested':4,
        'coverage':{'succeeded':1,'unusable':1,'failed':1,'deferred':1},'population_identity':population,
        'observations':[{'symbol':'002436.SZ','as_of':f'2025-01-0{i+2}','status':status,
            'forecast':{'identity':{'input':{'symbol':'002436.SZ','data_sha256':'raw-sha'},
                'model_path':'/Users/private-user/models/model.safetensors'},
                'reason':'fixture failure /Users/private-user/private/data.db',
                'quality':{'prediction_usable':status=='succeeded'}}}
            for i,status in enumerate(('succeeded','unusable','failed','deferred'))]}
    value['observations'][-1]['training_exclusion_reason']='immature_label'
    catalog=[{'pointer':'old','reference':{'cache_key':'expired'}},
        {'pointer':'new','reference':{'cache_key':'current','object_path':'/private/cache/object.json'}}]
    light=SimpleNamespace(jobs={},submit=Mock(),store=SimpleNamespace(
        get_setting=Mock(return_value=catalog),get_by_key=Mock(side_effect=lambda key:value if key=='current' else None)),
        _read_checkpoint=Mock(side_effect=lambda identity,namespace:value if identity['manifest_id']==manifest_id else None))
    app=FastAPI();app.state.light=light;app.include_router(router)
    return TestClient(app),light,manifest_id,value


def test_observation_list_is_small_skips_expired_and_never_resumes():
    client,light,identifier,value=fixture()
    with client:
        response=client.get('/api/light/score/observations')
    assert response.status_code==200
    data=response.json();assert data['automatic_resume'] is False and len(data['manifests'])==1
    row=data['manifests'][0]
    assert row['manifest_id']==identifier and row['coverage']==value['coverage']
    assert 'observations' not in row and 'population_identity' not in row
    assert '/Users/' not in response.text and '/private/' not in response.text
    light.submit.assert_not_called();assert not light.jobs


def test_observation_detail_preserves_population_states_and_hides_host_locations():
    client,light,identifier,value=fixture()
    with client:response=client.get('/api/light/score/observations/'+identifier)
    assert response.status_code==200
    data=response.json()
    assert [o['status'] for o in data['observations']]==['succeeded','unusable','failed','deferred']
    assert data['observations'][-1]['training_exclusion_reason']=='immature_label'
    assert data['observations'][0]['forecast']['identity']['input']['data_sha256']=='raw-sha'
    assert '/Users/' not in response.text and '/private/' not in response.text
    assert value['population_identity']['model_path'].startswith('/Users/')
    light.submit.assert_not_called()


def test_missing_manifest_is_reviewable_and_cannot_read_arbitrary_namespace():
    client,light,identifier,value=fixture()
    with client:
        for key in ('wrong','f'*64):
            assert client.get('/api/light/score/observations/'+key).status_code==404
    assert all(call.args[1]=='score_observation_manifests' for call in light._read_checkpoint.call_args_list)
    light.submit.assert_not_called()
