"""CLI/API resume keeps saved identities and never falls through to old scans."""
import json
from types import SimpleNamespace

import pytest

from my_strategy import cli
from my_strategy.services import analysis_methods as methods
from my_strategy.services import light_research
from my_strategy.storage import czsc_results
from my_strategy.web_dashboard import api, trend_strategy
from my_strategy.tests.test_analysis_methods import environment


@pytest.fixture
def previous_scan(environment, monkeypatch):
    light, store, calls = environment
    # This fixture supplies in-process numerical fakes. The production spawn
    # executor has separate real-process tests; keep CLI request checks isolated.
    from my_strategy.services import method_scan_acceleration
    monkeypatch.setattr(method_scan_acceleration, 'scan_methods', methods.scan_methods)
    original = methods.scan_methods(light, store, {
        'analysis_method': 'structure_kronos_ma_mlp', 'start': '2025-12-01', 'end': '2026-10-14',
        'usage_mode': 'forward', 'lookback': 256, 'horizon': 10, 'samples': 32,
        'seed': 7, 'device': 'cpu', 'max_predictions': 1,
    })
    assert original['coverage']['deferred'] == 1
    monkeypatch.setattr(czsc_results, 'ResultStore', lambda: store)
    light.close = lambda: None
    class FixtureResearch(light_research.LightResearch):
        # Preserve the actual method source used by generation identities while
        # substituting only construction of this already isolated fake service.
        def __new__(cls): return light
    monkeypatch.setattr(light_research, 'LightResearch', FixtureResearch)
    monkeypatch.setattr(trend_strategy, 'scan_trend_consensus', lambda **kwargs: pytest.fail('resume launched legacy scan'))
    return light, store, calls, original


def test_cli_resume_infers_method_and_preserves_all_nondefault_frozen_parameters(previous_scan, capsys):
    _, store, calls, original = previous_scan
    assert cli.main(['scan', '--resume-scan-id', original['run_id'], '--max-predictions', '1', '--json']) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed['status'] == 'succeeded' and resumed['coverage']['deferred'] == 0
    assert resumed['parent_scan_id'] == original['run_id'] and resumed['run_id'] != original['run_id']
    assert resumed['method_context'] == original['method_context']
    assert resumed['frozen_identity'] == original['frozen_identity']
    assert len(calls) == 2
    assert store.get(original['run_id'])['coverage']['deferred'] == 1


@pytest.mark.parametrize('arguments', [
    ['--analysis-method', 'legacy_reference'], ['--analysis-method', 'structure_kronos'],
    ['--start', '2026-01-01'], ['--usage-mode', 'historical'], ['--lookback', '128'],
    ['--horizon', '5'], ['--samples', '8'], ['--seed', '42'], ['--device', 'auto'],
    ['--score-model-id', 'b' * 64],
])
def test_cli_resume_explicit_differences_are_rejected(previous_scan, arguments, capsys):
    _, _, calls, original = previous_scan
    assert cli.main(['scan', '--resume-scan-id', original['run_id'], *arguments]) == 1
    assert '改变' in capsys.readouterr().err
    assert len(calls) == 1


@pytest.mark.parametrize('saved', [
    {'strategy_version': 'reference_position_v1'},
    {'strategy_version': methods.VERSION, 'method_context': {'analysis_method': 'structure_kronos'}},
])
def test_cli_rejects_noncombined_or_single_analysis_run_before_computation(monkeypatch, saved, capsys):
    monkeypatch.setattr(czsc_results, 'ResultStore', lambda: SimpleNamespace(get=lambda _: saved))
    monkeypatch.setattr(light_research, 'LightResearch', lambda: pytest.fail('invalid resume opened compute service'))
    monkeypatch.setattr(trend_strategy, 'scan_trend_consensus', lambda **kw: pytest.fail('invalid resume launched legacy scan'))
    assert cli.main(['scan', '--resume-scan-id', 'legacy-run']) == 1
    assert '只有保存的组合方法扫描支持续算' in capsys.readouterr().err


def test_cli_missing_resume_is_clear_error_and_analyze_cannot_resume(monkeypatch, capsys):
    def missing(_): raise KeyError('missing')
    monkeypatch.setattr(czsc_results, 'ResultStore', lambda: SimpleNamespace(get=missing))
    assert cli.main(['scan', '--resume-scan-id', 'missing']) == 1
    assert '续算标识不存在' in capsys.readouterr().err
    assert cli.main(['analyze', '--symbol', '000001.SZ', '--resume-scan-id', 'missing']) == 1
    assert '分析不接受' in capsys.readouterr().err


@pytest.mark.parametrize('arguments', [
    ['--model-run-id', 'legacy'], ['--model-fold', 'production'], ['--calendar-run-id', 'old-calendar'],
    ['--config', '/does-not-exist.json'], ['--current-position', '0'],
])
def test_combined_cli_does_not_silently_ignore_legacy_arguments(monkeypatch, arguments, capsys):
    monkeypatch.setattr(light_research, 'LightResearch', lambda: pytest.fail('invalid request opened compute service'))
    assert cli.main(['analyze', '--symbol', '000001.SZ', '--analysis-method', 'structure_kronos', *arguments]) == 1
    assert '组合方法不接受' in capsys.readouterr().err


def test_api_resume_submission_preserves_nondefault_context_and_remaining_work(previous_scan, monkeypatch):
    light, store, calls, original = previous_scan
    captured = []
    monkeypatch.setattr(api, 'submit_compute', lambda request, lane, kind, spec: captured.append(spec) or spec)
    request = api.TaskRequest(kind='scan', spec={
        'analysis_method': original['analysis_method'], 'resume_scan_id': original['run_id'], 'max_predictions': 1})
    submitted = api.submit(request, SimpleNamespace())
    assert 'start' not in submitted and 'usage_mode' not in submitted
    assert 'lookback' not in submitted and 'device' not in submitted
    resumed = methods.scan_methods(light, store, submitted)
    assert resumed['frozen_identity'] == original['frozen_identity']
    assert resumed['status'] == 'succeeded' and len(calls) == 2


@pytest.mark.parametrize('field,value', [('start', '2026-01-01'), ('usage_mode', 'historical')])
def test_api_resume_submission_does_not_discard_explicit_defaults(previous_scan, monkeypatch, field, value):
    light, store, calls, original = previous_scan
    monkeypatch.setattr(api, 'submit_compute', lambda request, lane, kind, spec: spec)
    request = api.TaskRequest(kind='scan', spec={
        'analysis_method': original['analysis_method'], 'resume_scan_id': original['run_id'], field: value})
    submitted = api.submit(request, SimpleNamespace())
    assert submitted[field] == value
    with pytest.raises(ValueError, match='改变'):
        methods.scan_methods(light, store, submitted)
    assert len(calls) == 1


def test_api_new_scan_keeps_existing_defaults(monkeypatch):
    monkeypatch.setattr(api, 'submit_compute', lambda request, lane, kind, spec: spec)
    submitted = api.submit(api.TaskRequest(kind='scan', spec={'analysis_method': 'structure_kronos'}), SimpleNamespace())
    assert submitted['start'] == '2026-01-01' and submitted['usage_mode'] == 'historical'
