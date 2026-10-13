"""Current web boundary, frozen plans and qualification-only default routing."""
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from my_strategy.storage.czsc_results import ResultStore


class CenterDouble:
    def __init__(self, **kwargs):
        self.saved = None
        self.release = threading.Event()

    def capabilities(self):
        return {'model_family': 'current', 'strategy': 'trend_consensus'}

    def profiles(self):
        return {'revision': 0, 'parameters': {'epochs': 20}}

    def save_default(self, parameters, expected_revision):
        if expected_revision != 0:
            raise ValueError('修订冲突')
        self.saved = dict(parameters)
        return {'revision': 1, 'parameters': self.saved}

    def preflight(self, parameters):
        return {'plan': {'plan_id': 'frozen-plan', 'parameters': parameters, 'valid': True}}

    def validate_plan(self, plan_id, parameters=None):
        if plan_id != 'frozen-plan' or (parameters is not None and parameters != {'epochs': 2}):
            raise ValueError('预检计划与提交参数不一致')
        return {'valid': True}

    def status(self):
        return {'active_model_run_id': None, 'publication_allowed': False}

    def selection_events(self):
        return {'events': []}

    def run_training_job(self, spec, progress, check_cancel, run_callback):
        run_callback('training-parent')
        while not self.release.wait(.01):
            check_cancel()
        progress('完成', 1, 1)
        return {'run_id': 'training-parent', 'parameters': {'epochs': 2}, 'automatic_selection': {'status': 'blocked'}}


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    import my_strategy.services.model_training_center as training
    monkeypatch.setattr(training, 'TrainingCenter', CenterDouble)
    from my_strategy.web_dashboard.tasks import TaskManager
    from my_strategy.web_dashboard.app import create_app
    from my_strategy.storage.personal_portfolio import PersonalStore
    results = ResultStore(tmp_path/'results.db', tmp_path/'runs')
    manager = TaskManager(tmp_path/'metadata/tasks.db', results=results, workers=1)
    personal = PersonalStore(tmp_path/'metadata/personal.db', raw_db=tmp_path/'raw.db')
    app = create_app(manager, personal)
    with TestClient(app) as client:
        yield client, manager
        manager.training.release.set()


@pytest.mark.parametrize('path', ['/training', '/parameters'])
def test_training_shell_is_original_app(dashboard, path):
    client, _ = dashboard
    response = client.get(path)
    assert response.status_code == 200
    assert 'training.js' in response.text


@pytest.mark.parametrize('payload', [
    {'symbol':'600519.SH','model_family':'high_return'},
    {'symbol':'600519.SH','consensus_version':'czsc_trend_consensus_v2'},
    {'symbol':'600519.SH','usage_mode':'retrospective'},
    {'symbol':'600519.SH','usage_mode':'production'},
])
def test_retired_analysis_requests_do_not_run(dashboard, payload):
    client, _ = dashboard
    assert client.post('/api/analysis', json=payload).status_code == 422


@pytest.mark.parametrize('kind', ['research_train','dual_research_train','high_return_train','high_return_backtest'])
def test_retired_tasks_do_not_queue(dashboard, kind):
    client, manager = dashboard
    assert client.post('/api/tasks', json={'kind': kind}).status_code == 422
    assert manager.list() == []


@pytest.mark.parametrize('path', ['status','dual-status','promote','high-return-forward-status'])
def test_retired_paths_are_explicit(dashboard, path):
    client, _ = dashboard
    assert client.get('/api/research/'+path).status_code == 410


def test_default_profile_revision_is_required(dashboard):
    client, manager = dashboard
    assert client.put('/api/training/default-profile', json={'parameters':{}}).status_code == 422
    assert client.put('/api/training/default-profile', json={'parameters':{'epochs':2},'expected_revision':1}).status_code == 400
    result = client.put('/api/training/default-profile', json={'parameters':{'epochs':2},'expected_revision':0})
    assert result.status_code == 200
    assert manager.training.saved == {'epochs': 2}


def test_job_requires_frozen_plan_and_deduplicates_active_submission(dashboard):
    client, manager = dashboard
    assert client.post('/api/training/jobs', json={'parameters':{'epochs':2}}).status_code == 422
    assert client.post('/api/training/jobs', json={'plan_id':'frozen-plan','parameters':{'epochs':3}}).status_code == 400
    spec = {'plan_id':'frozen-plan','parameters':{'epochs':2}}
    first = client.post('/api/training/jobs', json=spec)
    second = client.post('/api/training/jobs', json=spec)
    assert first.status_code == second.status_code == 202
    assert first.json()['job_id'] == second.json()['job_id']
    manager.training.release.set()
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        job = manager.get(first.json()['job_id'])
        if job['status'] in {'succeeded','failed'}:
            break
        time.sleep(.02)
    assert job['status'] == 'succeeded', job
    assert job['result']['automatic_selection']['status'] == 'blocked'
    assert job['result']['parameters'] == {'epochs': 2}


def test_default_runtime_never_scans_candidates_or_uses_old_pointer(monkeypatch, tmp_path):
    import my_strategy.services.model_training_center as training
    from my_strategy.services import czsc_trend_runtime as runtime, czsc_trend_pipeline as pipeline
    class NoSelection:
        def active_model_run_id(self, **kwargs):
            return None
    monkeypatch.setattr(training, 'TrainingCenter', NoSelection)
    pointer = tmp_path/'current.json'
    pointer.write_text('{"strategy":"trend_consensus","model_run_id":"unqualified-candidate"}')
    monkeypatch.setattr(runtime, 'CURRENT_MODEL_PATH', pointer)
    monkeypatch.setattr(pipeline, 'read_run', lambda *args: pytest.fail('no qualified selection must not load any candidate'))
    assert runtime.load_model_run(as_of='2026-10-12') == (None, None)


def test_default_runtime_uses_selected_asof_package(monkeypatch):
    import my_strategy.services.model_training_center as training
    from my_strategy.services import czsc_trend_runtime as runtime, czsc_trend_pipeline as pipeline
    calls=[]
    class Selection:
        def active_model_run_id(self, **kwargs):
            calls.append(kwargs)
            return 'qualified-current'
    monkeypatch.setattr(training, 'TrainingCenter', Selection)
    monkeypatch.setattr(pipeline, 'read_run', lambda run_id:(Path(run_id), {'run_id':run_id}))
    assert runtime.load_model_run(as_of='2026-10-12')[0].name == 'qualified-current'
    assert calls == [{'as_of':'2026-10-12','scope':'research'}]
