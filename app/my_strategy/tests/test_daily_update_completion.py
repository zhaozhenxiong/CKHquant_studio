"""Target-day retries and public completion states without business data writes."""
from types import SimpleNamespace
from unittest.mock import Mock
import json
import sys

import pandas as pd
import pytest

from my_strategy.scripts import update_daily_data as update
from my_strategy.services import daily_feature_sync
from my_strategy.web_dashboard import tasks


@pytest.fixture
def hook(tmp_path, monkeypatch):
    call = Mock(return_value={'status': 'completed', 'training_started': False})
    monkeypatch.setattr(daily_feature_sync, 'sync_daily_features', call)
    monkeypatch.setattr(update, 'get_data_access', lambda: SimpleNamespace(settings=SimpleNamespace(raw_db=tmp_path / 'raw.db')))
    args = SimpleNamespace(skip_feature_sync=False, feature_mode="full", feature_kb_db=None, feature_fluent_db=None,
        feature_native_cache_db=None, feature_calendar_run_id=None, feature_sync_id=None,
        feature_cpu_workers=6, feature_stock_batch_size=360, feature_kb_budget=2000,
        feature_kb_min_interval=1.3, feature_source_root=None, feature_artifacts_root=tmp_path)
    return args, call


def test_already_covered_target_day_retries_structure_after_no_new_rows(hook):
    args, call = hook
    summary = {'update_id': 'retry-update', 'success': 0, 'failed': 0, 'total': 2, 'target_date_covered': 1}
    result = update._sync_after_update(args, summary, {})
    assert result['status'] == 'completed'
    assert call.call_args.kwargs['parent_update_id'] == 'retry-update'


def test_reported_target_coverage_still_blocks_when_only_older_qualified_prices_exist(hook):
    args, call = hook
    call.side_effect = daily_feature_sync.SyncTargetDateUnavailable(
        'qualified local price date 2026-10-12 differs from required target date 2026-10-14')
    summary = {'update_id': 'unverified-target', 'success': 1, 'failed': 0,
        'total': 1, 'target_date_covered': 1, 'end_date': '2026-10-14'}
    result = update._sync_after_update(args, summary, {})
    assert call.call_args.kwargs['required_price_date'] == '2026-10-14'
    assert result['status'] == 'blocked' and result['reason_code'] == 'qualified_target_date_not_available'
    assert '2026-10-12' in result['error'] and '2026-10-14' in result['error']


def test_no_new_rows_still_quality_checks_actual_target_day_coverage(monkeypatch):
    checker = Mock()
    checker.check.return_value = SimpleNamespace(run_id='retry-quality', checked_at='now',
        start_date='2026-10-09', end_date='2026-10-14', stocks=['600000.SH'],
        issues=[], has_errors=lambda: False)
    monkeypatch.setattr(update, 'DataQualityChecker', Mock(return_value=checker))
    report = pd.DataFrame([
        {'code': '600000.SH', 'last_date': '2026-10-14', 'status': 'skipped'},
        {'code': '000016.SZ', 'last_date': '2026-09-03', 'status': 'skipped'}])
    result = update._run_quality_check({}, report, '2026-10-14', 'retry-fixture')
    assert checker.check.call_args.kwargs['stocks'] == ['600000.SH']
    assert checker.check.call_args.kwargs['end_date'] == '2026-10-14' and result['errors'] == 0


@pytest.mark.parametrize('total,covered,failed,quality', [
    (0, 0, 0, {}), (0, 1, 0, {}), (2, 0, 0, {}),
    (2, 1, 1, {}), (2, 1, 0, {'has_errors': True}), (2, 1, 0, {'error': 'invalid source'})])
def test_retry_requires_nonempty_target_coverage_and_no_current_errors(hook, total, covered, failed, quality):
    args, call = hook
    result = update._sync_after_update(args, {'success': 0, 'total': total,
        'target_date_covered': covered, 'failed': failed}, quality)
    assert result['status'] in {'blocked', 'skipped'}
    call.assert_not_called()


@pytest.mark.parametrize('total,covered', [(1, 0), (0, 0), (0, 1)])
def test_successful_older_download_or_empty_request_never_starts_old_day_sync(hook, total, covered):
    args, call = hook
    result = update._sync_after_update(args, {'update_id': 'older-download', 'success': 1,
        'failed': 0, 'total': total, 'target_date_covered': covered}, {})
    assert result['status'] == 'skipped' and 'no confirmed target-date' in result['reason']
    call.assert_not_called()


def test_older_raw_success_retains_partial_overall_status_while_maintenance_skip_does_not():
    summary = {'failed': 0, 'total': 1, 'not_updated': 0,
        'feature_sync': {'status': 'skipped', 'reason_code': 'target_date_not_covered'}}
    update._set_completion_status(summary, {})
    assert summary['raw_status'] == 'completed' and summary['status'] == 'partial'
    summary['feature_sync'] = {'status': 'skipped', 'reason': 'explicit maintenance skip'}
    update._set_completion_status(summary, {})
    assert summary['raw_status'] == summary['status'] == 'completed'


@pytest.mark.parametrize('derived,overall', [('busy', 'partial'), ('running', 'partial'),
    ('partial', 'partial'), ('failed', 'partial'), ('completed', 'completed')])
def test_cli_reports_committed_raw_separately_from_derived_completion(tmp_path, monkeypatch, capsys, derived, overall):
    report = pd.DataFrame([{'code': '600000.SH', 'last_date': '2026-10-14', 'status': 'success'}])
    monkeypatch.setattr(sys, 'argv', ['update_daily_data.py', '--end-date', '2026-10-14',
        '--json', '--skip-quality-check', '--skip-metadata-refresh'])
    monkeypatch.setattr(update, 'load_data_config', lambda _: {})
    monkeypatch.setattr(update, 'DailyUpdater', lambda *a, **kw: SimpleNamespace(update_pool=lambda **kw: report))
    monkeypatch.setattr(update, 'get_data_access', lambda: SimpleNamespace(settings=SimpleNamespace(write_db=False)))
    monkeypatch.setattr(update, '_sync_after_update', lambda *a: {'status': derived, 'sync_id': 'retry-fixture'})
    with pytest.raises(SystemExit) as stopped:
        update.main()
    assert stopped.value.code == 0
    result = json.loads(capsys.readouterr().out)['summary']
    assert result['raw_status'] == 'completed' and result['status'] == overall
    assert result['feature_sync']['sync_id'] == 'retry-fixture'


@pytest.mark.parametrize('derived', ['busy', 'queued', 'running', 'partial', 'failed', 'interrupted', 'blocked', 'cancelled'])
def test_dashboard_never_reports_complete_while_derived_work_is_pending(tmp_path, monkeypatch, derived):
    from my_strategy.adapters import czsc_adapter
    monkeypatch.setattr(czsc_adapter, 'data_status', lambda: {'fixture': True})
    context = SimpleNamespace(run_id='isolated-update', to_dict=lambda: {'run_id': 'isolated-update'},
        subdir=lambda _: tmp_path)
    monkeypatch.setattr(tasks, 'create_run_context', lambda **_: context)
    payload = {'summary': {'status': 'completed', 'raw_status': 'completed', 'total': 1,
        'failed': 0, 'target_date_covered': 1,
        'feature_sync': {'status': derived, 'sync_id': 'recoverable-sync', 'reason': 'native not published'}}}
    def spawn(_command, *, stdout, **kwargs):
        stdout.write(json.dumps(payload, indent=2)); stdout.flush()
        return SimpleNamespace(returncode=0, poll=lambda: 0)
    monkeypatch.setattr(tasks.subprocess, 'Popen', spawn)
    manager = tasks.TaskManager.__new__(tasks.TaskManager)
    changes = []
    manager._patch = lambda job_id, **value: changes.append(value)
    result = manager._update('isolated-job', {})
    assert result['status'] == 'partial' and result['feature_sync']['sync_id'] == 'recoverable-sync'
    assert result['error'] == 'native not published'
    detail = changes[-1]['progress_detail']
    assert detail['feature_sync_status'] == derived and detail['sync_id'] == 'recoverable-sync'
    assert '结构与特征同步未完成' in detail['stage']
