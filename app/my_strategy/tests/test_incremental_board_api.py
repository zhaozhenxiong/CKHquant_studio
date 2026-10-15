from pathlib import Path
from unittest.mock import Mock

import pytest

from my_strategy.services import incremental_sync
from my_strategy.tests.test_light_api import api


@pytest.fixture
def configured(api,monkeypatch,tmp_path):
    for name in ('board_db_path','board_fluent_db_path','board_raw_db_path'):
        setattr(api.app.state,name,tmp_path/(name+'.db'))
    create=Mock(return_value={'plan_id':'a'*32,'status':'planned','requested':3,'estimated_deferred':2})
    read=Mock(return_value={'plan_id':'a'*32,'status':'planned'})
    monkeypatch.setattr(incremental_sync,'create_plan',create)
    monkeypatch.setattr(incremental_sync,'get_plan',read)
    return api,create,read


def test_plan_is_preview_only_and_binds_server_paths(configured):
    api,create,read=configured
    response=api.client.post('/api/boards/incremental/plan',json={'scope':'current_board','board_id':'THS_I_123456'})
    assert response.status_code==200 and response.json()['estimated_deferred']==2
    assert create.call_args.kwargs['scope']=='current_board'
    assert create.call_args.kwargs['raw_db_path']==api.app.state.board_raw_db_path
    assert create.call_args.kwargs['max_symbols']==20 and create.call_args.kwargs['budget_seconds']==300
    api.light.submit.assert_not_called();api.app.state.tasks.submit.assert_not_called()


@pytest.mark.parametrize('extra',[{'scope':'other'},{'max_symbols':101},{'budget_seconds':1801},
    {'raw_db_path':'/unexpected'},{'resume_sync_id':'old'}])
def test_plan_rejects_implicit_resume_and_injected_paths(configured,extra):
    api,create,read=configured
    assert api.client.post('/api/boards/incremental/plan',json=extra).status_code==422
    create.assert_not_called()


def test_execute_only_accepts_frozen_plan_and_uses_light_lane(configured):
    api,create,read=configured
    response=api.client.post('/api/boards/incremental/execute',json={'plan_id':'a'*32})
    assert response.status_code==202
    kind,spec=api.light.submit.call_args.args
    assert kind=='incremental' and spec=={'plan_id':'a'*32,'kb_db_path':str(api.app.state.board_db_path),
        'fluent_db_path':str(api.app.state.board_fluent_db_path)}
    api.app.state.tasks.submit.assert_not_called()
    assert api.client.post('/api/boards/incremental/execute',json={'plan_id':'a'*32,'scope':'full'}).status_code==422


def test_cancelled_plan_cannot_queue_and_status_is_readable(configured):
    api,create,read=configured;read.return_value={'plan_id':'a'*32,'status':'cancelled'}
    assert api.client.get('/api/boards/incremental/plans/'+'a'*32).json()['status']=='cancelled'
    assert api.client.post('/api/boards/incremental/execute',json={'plan_id':'a'*32}).status_code==409
    api.light.submit.assert_not_called()


def test_active_compute_rejects_execution_without_native_start(configured):
    api,create,read=configured;api.light.jobs['running']={'status':'running'}
    assert api.client.post('/api/boards/incremental/execute',json={'plan_id':'a'*32}).status_code==409
    api.light.submit.assert_not_called()


def test_legacy_global_prepare_requires_explicit_scope_and_small_default(configured):
    api,create,read=configured
    assert api.client.post('/api/boards/sync',json={}).status_code==409
    result=api.client.post('/api/boards/sync',json={'scope':'full'})
    assert result.status_code==202
    kind,spec=api.app.state.tasks.submit.call_args.args
    assert kind=='board_sync' and spec['cpu_workers']==1
    assert api.client.post('/api/boards/sync',json={'scope':'full','cpu_workers':6}).status_code==422
