"""Resume routing preserves source identity and cannot report a fresh run as resumed."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from my_strategy.services import model_training_center as center
from my_strategy.web_dashboard.api import TrainingJobRequest


@pytest.fixture
def routed_training(tmp_path, monkeypatch):
    plan = {'plan_id': 'verified-plan', 'data_end': '2026-10-12',
        'calendar_run_id': 'verified-calendar', 'calendar_hash': center.stable_hash(['2026-10-12']),
        'cohort': ['600000.SH'], 'plan_hash': 'frozen-plan-hash', 'config': {'seed': 42},
        'parameters': {'cpu_workers': 10, 'device': 'mps', 'stock_batch_size': 360},
        'source_reuse': {'source_native_run_id': 'completed-native', 'raw_source_run_id': 'completed-raw'}}
    instance = object.__new__(center.TrainingCenter)
    instance.runs_root = tmp_path / 'runs'
    instance.load_plan = lambda *args, **kwargs: plan
    def subdir(name):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path
    context = SimpleNamespace(run_id='resume-parent', subdir=subdir)
    monkeypatch.setattr(center, 'create_run_context', lambda **kwargs: context)
    from my_strategy.services import czsc_research_calendar
    monkeypatch.setattr(czsc_research_calendar, 'load_verified_calendar',
        lambda *args, **kwargs: (['2026-10-12'], {}))
    return instance, plan


def test_resume_routes_original_resources_and_complete_native_without_prepare(routed_training):
    instance, plan = routed_training
    captured = {}
    def run(args, *unused, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        raise RuntimeError('routing verified')
    instance._run_process = run
    with pytest.raises(RuntimeError, match='routing verified'):
        instance._execute_training_job({'plan_id': plan['plan_id'], 'resume_training_run_id': 'original-model'})
    args = captured['args']
    assert 'my_strategy.scripts.resume_training_legacy_cached' in args
    assert args[args.index('--resume-training-run-id') + 1] == 'original-model'
    assert args[args.index('--source-native-run-id') + 1] == 'completed-native'
    assert args[args.index('--cpu-workers') + 1] == '10'
    assert args[args.index('--device') + 1] == 'mps'
    assert '--prepare-source-only' not in args and '--resume-native-run-id' not in args
    assert captured['kwargs']['initial_ids'] == {'raw_source_run_id': 'completed-raw', 'source_native_run_id': 'completed-native'}


def test_accelerated_resume_routes_same_run_to_checkpoint_adapter(routed_training):
    instance, plan = routed_training
    sidecar = instance.runs_root / 'original-model/reports/training_acceleration.json'
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text('{}')  # The subprocess performs strict content verification.
    captured = {}
    def run(args, *unused, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        raise RuntimeError('routing verified')
    instance._run_process = run
    with pytest.raises(RuntimeError, match='routing verified'):
        instance._execute_training_job({'plan_id': plan['plan_id'], 'resume_training_run_id': 'original-model'})
    args = captured['args']
    assert 'my_strategy.scripts.resume_training_fast' in args
    assert args[args.index('--resume-training-run-id') + 1] == 'original-model'
    assert args[args.index('--source-native-run-id') + 1] == 'completed-native'
    assert '--prepare-source-only' not in args
    assert captured['kwargs']['source_reuse']['resume_training_run_id'] == 'original-model'


@pytest.mark.parametrize('invalid', ['no_source', 'source_prepare'])
def test_resume_rejects_incomplete_source_or_preparation_stage(routed_training, invalid):
    instance, plan = routed_training
    spec = {'plan_id': plan['plan_id'], 'resume_training_run_id': 'original-model'}
    if invalid == 'no_source': plan['source_reuse'] = None
    else: spec['stage'] = 'source_prepare'
    instance._run_process = lambda *args, **kwargs: pytest.fail('invalid resume launched a child')
    with pytest.raises(ValueError, match='完整native'):
        instance._execute_training_job(spec)


def test_resume_refuses_a_new_model_run_identity(routed_training):
    instance, plan = routed_training
    instance._run_process = lambda *args, **kwargs: {'model_run_id': 'different-new-model'}
    with pytest.raises(ValueError, match='不同模型run'):
        instance._execute_training_job({'plan_id': plan['plan_id'], 'resume_training_run_id': 'original-model'})


def test_resume_request_rejects_path_escape_and_preserves_lineage():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TrainingJobRequest(plan_id='plan', resume_training_run_id='../outside')
    request = TrainingJobRequest(plan_id='plan', resume_training_run_id='original-model', resume_from_job_id='original-job')
    assert request.model_dump(exclude_none=True)['resume_from_job_id'] == 'original-job'


def test_resume_asset_progress_and_cached_e1_population_are_separate(tmp_path):
    import sys
    events = []
    lines = [
        'RESUME_PROGRESS ' + json.dumps({'stage': 'training_resume_verification', 'completed': 9, 'total': 9}),
        'PROGRESS ' + json.dumps({'stage': 'dataset', 'population': 'E1', 'fold': None,
            'completed': 2, 'total': 3, 'scope': 'current_population_only'}),
        'PROGRESS ' + json.dumps({'stage': 'dataset', 'population': 'E1', 'fold': None,
            'completed': 3, 'total': 3, 'scope': 'current_population_only'}),
    ]
    code = '\n'.join('print(' + repr(line) + ',flush=True)' for line in lines)
    center.TrainingCenter._run_process([sys.executable, '-u', '-c', code],
        tmp_path / 'resume.log', events.append, None, source_total=3)
    assert [(e['stage'], e['completed'], e['total']) for e in events] == [
        ('training_resume_verification', 9, 9), ('E1', 2, 3), ('E1', 3, 3)]
