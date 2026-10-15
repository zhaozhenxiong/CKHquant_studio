"""Standalone native ownership blocks duplicate sync without granting reuse."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import daily_feature_sync as sync
from my_strategy.tests.test_daily_feature_sync import environment, raw_db


@pytest.fixture
def active(tmp_path, monkeypatch):
    producer = tmp_path / 'producer'
    (producer / 'my_strategy').mkdir(parents=True)
    warehouse = tmp_path / 'raw.db'
    warehouse.write_bytes(b'guard must not open or mutate quotes')
    artifacts = tmp_path / 'artifacts'
    native = artifacts / 'runs/native-active'
    reports = native / 'reports'
    reports.mkdir(parents=True)
    status = dict(status='running',run_id=native.name,owner_pid=1234,attempt_id='actual-attempt',
        started_at=datetime.fromtimestamp(1200,timezone.utc).isoformat())
    status_path = reports / 'native_prepare_status.json'
    status_path.write_text(json.dumps(status))
    header = dict(format='native_source_checkpoint',run_id=native.name,
        identity={'raw_source_run_id':'raw-active'},config={'input':'frozen'})
    header['checkpoint_hash'] = stable_hash(header)
    (reports / 'native_checkpoint.json').write_text(json.dumps(header))
    lock = reports / 'native_preparation.lock'
    lock.touch()
    command = ['python', str(producer / 'my_strategy/scripts/refresh_native_cache.py'),
        '--end','2026-10-14','--_worker']
    env = {'KHQUANT_PROJECT_ROOT':str(producer),'KHQUANT_RAW_DB':str(warehouse),
        'KHQUANT_ARTIFACT_ROOT':str(artifacts)}
    owner = SimpleNamespace(create_time=lambda:1000.0,is_running=lambda:True,
        status=lambda:'sleeping',cmdline=lambda:command,cwd=lambda:str(producer),
        environ=lambda:env,open_files=lambda:[SimpleNamespace(path=str(lock))])
    monkeypatch.setattr(sync.psutil,'Process',lambda pid:owner)
    return SimpleNamespace(producer=producer,warehouse=warehouse,artifacts=artifacts,
        status=status,status_path=status_path,header=header,lock=lock,owner=owner,env=env,command=command,
        options=dict(raw_db_path=warehouse,source_root=producer,artifacts_root=artifacts))


def test_live_worker_and_open_run_lock_return_resource_busy_without_input_qualification(active):
    before = active.status_path.read_bytes(),active.warehouse.read_bytes(),active.lock.stat().st_mtime_ns
    result = sync._native_resource_busy(**active.options)
    assert result['resource_busy'] is True and result['compatibility_verified'] is False
    assert result['blocking_source_native_run_id']=='native-active'
    assert result['blocking_raw_source_run_id']=='raw-active'
    assert result['blocking_native_owner_pid']==1234
    assert 'source_native_run_id' not in result and 'reused' not in result
    assert before==(active.status_path.read_bytes(),active.warehouse.read_bytes(),active.lock.stat().st_mtime_ns)


@pytest.mark.parametrize('changed',['producer','raw','artifacts','worker','lock','future_birth','unknown_birth','dead','future_status','completed','unknown_owner'])
def test_stale_unknown_or_other_resource_owner_does_not_block(active,monkeypatch,changed):
    if changed=='producer':active.env['KHQUANT_PROJECT_ROOT']=str(active.producer/'other')
    elif changed=='raw':active.env['KHQUANT_RAW_DB']=str(active.warehouse.with_name('other.db'))
    elif changed=='artifacts':active.env['KHQUANT_ARTIFACT_ROOT']=str(active.artifacts/'other')
    elif changed=='worker':active.command.remove('--_worker')
    elif changed=='lock':active.owner.open_files=lambda:[]
    elif changed=='future_birth':active.owner.create_time=lambda:1300.0
    elif changed=='unknown_birth':active.owner.create_time=lambda:float('nan')
    elif changed=='dead':active.owner.is_running=lambda:False
    elif changed=='future_status':
        active.status['started_at']='2999-01-01T00:00:00+00:00';active.status_path.write_text(json.dumps(active.status))
    elif changed=='completed':
        active.status['status']='completed';active.status_path.write_text(json.dumps(active.status))
    else:
        def missing(pid):raise psutil.NoSuchProcess(pid)
        monkeypatch.setattr(sync.psutil,'Process',missing)
    assert sync._native_resource_busy(**active.options) is None


def test_access_denied_owner_is_not_a_permanent_lock(active):
    def unknown():raise psutil.AccessDenied(1234)
    active.owner.open_files=unknown
    assert sync._native_resource_busy(**active.options) is None


def test_command_line_resource_options_bind_actual_worker_resources(active):
    active.command.extend(['--raw-db='+str(active.warehouse),'--artifacts-root',str(active.artifacts),
        '--source-root',str(active.producer)])
    active.env.clear()
    assert sync._native_resource_busy(**active.options)['resource_busy'] is True


def test_owner_exit_during_observation_cannot_leave_busy(active):
    alive=iter([True,False])
    active.owner.is_running=lambda:next(alive)
    assert sync._native_resource_busy(**active.options) is None


def test_resource_busy_precedes_raw_identity_and_never_takes_fluent_lease(active,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('busy sync touched raw identity, lease, or native launcher')
    monkeypatch.setattr(sync,'qualified_raw_identity',forbidden)
    monkeypatch.setattr(sync,'_store',forbidden)
    monkeypatch.setattr(sync,'_run_native',forbidden)
    events=[]
    result=sync.sync_daily_features(**active.options,required_price_date='2026-10-15',
        config={'requested_config':'different'},progress=events.append)
    assert result['status']=='busy' and result['training_started'] is False
    assert result['compatibility_verified'] is False and result['resource_busy'] is True
    assert not result['market_published'] and not result['full_published']
    assert events[-1]['reason_code']=='native_preparation_resource_busy'


def test_late_standalone_owner_preserves_market_and_releases_own_sync(environment,monkeypatch):
    responses=iter([None,{'resource_busy':True,'compatibility_verified':False,
        'reason_code':'native_preparation_resource_busy','blocking_source_native_run_id':'other-source',
        'blocking_raw_source_run_id':'other-raw','error':'native resource busy'}])
    monkeypatch.setattr(sync,'_native_resource_busy',lambda **kwargs:next(responses))
    result=sync.sync_daily_features(**environment.options)
    assert result['status']=='partial' and result['market_published'] and not result['full_published']
    assert result['blocking_source_native_run_id']=='other-source' and result['training_started'] is False
    assert len(environment.calls['market'])==1 and not environment.calls['native']
    assert environment.store.get_sync(result['sync_id'])['status']=='partial'
    assert 'native_result' not in environment.store.get_sync(result['sync_id'])['checkpoint']


def test_no_active_owner_preserves_completed_reuse(environment):
    result=sync.sync_daily_features(**environment.options)
    again=sync.sync_daily_features(**environment.options)
    assert result['status']=='completed' and again['status']=='completed' and again['reused']
    assert again['sync_id']==result['sync_id'] and len(environment.calls['native'])==1


def test_cancellation_is_checked_before_resource_scan(active):
    with pytest.raises(sync.SyncCancelled):
        sync._native_resource_busy(**active.options,check_cancel=lambda:True)


def test_task_wrapper_keeps_specific_resource_busy_reason_and_source_ids(tmp_path,monkeypatch):
    from my_strategy.web_dashboard import tasks
    context=SimpleNamespace(run_id='board-job',to_dict=lambda:{'run_id':'board-job'})
    monkeypatch.setattr(tasks,'create_run_context',lambda **kwargs:context)
    monkeypatch.setattr(sync,'sync_daily_features',lambda **kwargs:{
        'status':'busy','resource_busy':True,'compatibility_verified':False,
        'blocking_source_native_run_id':'active-source','blocking_raw_source_run_id':'active-raw',
        'error':'已有原生结构准备正在使用本地来源，请待该批次完成后重试。','training_started':False})
    manager=tasks.TaskManager.__new__(tasks.TaskManager)
    manager._patch=lambda *args,**kwargs:None
    manager._check=lambda *args:None
    result=manager._board('job','board_sync',{})
    assert result['status']=='partial' and '原生结构准备' in result['error']
    assert '持有数据库' not in result['error']
    assert result['blocking_source_native_run_id']=='active-source'
    assert result['resource_busy'] is True and result['compatibility_verified'] is False
    assert result['run_id']=='board-job' and result['training_started'] is False


def test_cli_busy_exits_two_without_raw_hash_lease_or_native_launch(active,monkeypatch,capsys):
    from my_strategy.scripts.sync_daily_features import main
    def forbidden(*args,**kwargs):pytest.fail('busy CLI entered a data or publication path')
    monkeypatch.setattr(sync,'qualified_raw_identity',forbidden)
    monkeypatch.setattr(sync,'_store',forbidden)
    monkeypatch.setattr(sync,'_run_native',forbidden)
    assert main(['--raw-db',str(active.warehouse),'--source-root',str(active.producer),
        '--artifacts-root',str(active.artifacts),'--price-date','2026-10-14'])==2
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='busy' and result['compatibility_verified'] is False
    assert result['blocking_source_native_run_id']=='native-active' and result['training_started'] is False
