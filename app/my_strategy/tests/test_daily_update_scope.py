"""Download scope and unavailable-data provenance, isolated from business stores."""
from types import SimpleNamespace
from unittest.mock import Mock
import json
import subprocess

import pandas as pd
import pytest

from my_strategy.data_manager import updater as module
from my_strategy.data_manager.update_universe import filter_update_pool, normalize_update_code
from my_strategy.data_manager.updater import DailyUpdater
from my_strategy.data_manager.downloader import DataDownloader, DownloadResult
from my_strategy.data_manager.market_data_provider import ProviderResult, SinaUnadjustedProvider, UnifiedDataProvider
from my_strategy.scripts import update_daily_data as script
from my_strategy.web_dashboard.tasks import _daily_update_result, _daily_update_failure


@pytest.mark.parametrize('code', ['920000.BJ', 'bj920000', 'BJ.920000', '920000.BSE', '920000', '430047', '830799'])
def test_beijing_forms_excluded(code):
    pool, info = filter_update_pool(pd.DataFrame({'stock': [code, '600000.SH', '000001.SZ']}))
    assert pool.stock.tolist() == ['600000.SH', '000001.SZ']
    assert info['excluded'] == 1 and info['eligible'] == 2


@pytest.mark.parametrize('code,expected', [('900901.SH', '900901.SH'), ('900901', '900901.SH'), ('200002.SZ', '200002.SZ'), ('688001.SH', '688001.SH'), ('SZ.000001', '000001.SZ')])
def test_explicit_other_exchanges_preserved(code, expected):
    assert normalize_update_code(code) == expected
    pool, info = filter_update_pool(pd.DataFrame({'stock': [code]}))
    assert pool.stock.tolist() == [expected] and info['excluded'] == 0


def _updater():
    item = DailyUpdater.__new__(DailyUpdater)
    item.config = {'update': {'sleep_seconds': 0, 'use_batch': False}}
    item.cache = Mock()
    item.cache.read_stock_daily.return_value = pd.DataFrame({'date': ['2026-09-03'], 'trade_close': [10]})
    item.downloader = Mock()
    item.workers, item.use_progress_bar, item.progress_callback = 1, False, None
    return item


@pytest.mark.parametrize('mode', ['local', 'db', 'all_a', 'custom'])
def test_scope_filter_precedes_limit_and_progress(monkeypatch, mode):
    item = _updater()
    manager = Mock()
    manager.build_pool.return_value = pd.DataFrame({'stock': ['920000.BJ', '000001.SZ', '600000.SH']})
    monkeypatch.setattr(module, 'StockPoolManager', Mock(return_value=manager))
    item.update_one = Mock(side_effect=lambda code, **kw: {'code': code, 'status': 'success'})
    progress = []; item.progress_callback = progress.append
    report = item.update_pool(mode=mode, max_stocks=1)
    assert report.code.tolist() == ['000001.SZ']
    assert progress[0]['total'] == progress[0]['current'] == 1
    assert report.attrs['update_universe']['excluded'] == 1


def test_single_beijing_code_never_reads_or_downloads():
    item = _updater(); result = item.update_one('bj920000')
    assert result['status'] == 'skipped' and result['error_msg'] == 'excluded_beijing_exchange'
    item.cache.read_stock_daily.assert_not_called(); item.downloader.download_stock_daily.assert_not_called()


@pytest.mark.parametrize('empty_range,status', [(True, 'skipped'), (False, 'failed')])
def test_single_empty_response_requires_successful_source_proof(empty_range, status):
    item = _updater()
    item.downloader.download_stock_daily.return_value = DownloadResult(empty_range, pd.DataFrame() if empty_range else None, 'test', '' if empty_range else 'request timeout', empty_range=empty_range)
    row = item.update_one('000016.SZ', end_date='2026-10-14')
    assert row['status'] == status and row['last_date'] == '2026-09-03'
    assert row['error_msg']
    item.cache.save_stock_daily.assert_not_called()


def test_batch_preserves_partial_failure_and_empty_stock_evidence():
    item = _updater()
    codes = ['000016.SZ', '600000.SH']
    item.downloader.download_stocks_daily.return_value = DownloadResult(True, pd.DataFrame(), 'test', '', stock_results={
        codes[0]: {'status': 'no_data'}, codes[1]: {'status': 'failed', 'error': 'request timeout'}})
    report = item.update_batch(codes, end_date='2026-10-14')
    rows = report.set_index('code')
    assert rows.loc[codes[0], 'status'] == 'skipped'
    assert rows.loc[codes[1], 'status'] == 'failed'
    assert rows.loc[codes[1], 'error_msg'] == 'request timeout'
    assert set(rows.last_date) == {'2026-09-03'}
    item.cache.save_stocks_daily.assert_not_called()


def test_provider_batch_keeps_each_outcome_and_downloader_retains_it():
    provider = UnifiedDataProvider.__new__(UnifiedDataProvider)
    provider._call_with_retry = Mock(side_effect=[ProviderResult(True, pd.DataFrame(), 'test', empty_range=True), ProviderResult(False, None, 'test', 'timeout')])
    result = provider._fetch_daily_batch_per_stock(SimpleNamespace(name='test'), ['000016.SZ', '600000.SH'], '2026-10-09', '2026-10-14', primary='test')
    assert result.ok and result.data.empty
    assert result.stock_results['000016.SZ']['status'] == 'no_data'
    assert result.stock_results['600000.SH']['error'] == 'timeout'
    downloader = DataDownloader({}); downloader._provider = Mock()
    downloader._provider.fetch_daily_batch.return_value = result
    assert downloader.download_stocks_daily(['000016.SZ', '600000.SH'], '2026-10-09', '2026-10-14').stock_results == result.stock_results


def test_sina_valid_history_empty_window_is_distinct_from_malformed_source(monkeypatch):
    from my_strategy.data_manager import market_data_provider as source
    monkeypatch.setattr(source, '_curl_executable', lambda: 'curl')
    monkeypatch.setattr(source.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, 'valid encoded', ''))
    rows = [{'date': '2026-09-03T00:00:00Z', 'open': 10, 'close': 10, 'high': 10, 'low': 10, 'volume': 10, 'amount': 100}]
    monkeypatch.setattr(SinaUnadjustedProvider, '_decode_history', staticmethod(lambda *a: rows))
    result = SinaUnadjustedProvider().fetch_daily('000016.SZ', '2026-10-09', '2026-10-14')
    assert result.ok and result.empty_range and result.data.empty
    rows[0]['date'] = 'malformed'
    result = SinaUnadjustedProvider().fetch_daily('000016.SZ', '2026-10-09', '2026-10-14')
    assert not result.ok and not result.empty_range


def test_quality_checks_only_successful_rows_and_never_future_days(monkeypatch):
    checker = Mock()
    checker.check.return_value = SimpleNamespace(run_id='test', checked_at='now', start_date='2026-10-09', end_date='2026-10-14', stocks=['600000.SH'], issues=[], has_errors=lambda: False)
    monkeypatch.setattr(script, 'DataQualityChecker', Mock(return_value=checker))
    report = pd.DataFrame([{'code': '600000.SH', 'status': 'success'}, {'code': '000016.SZ', 'status': 'skipped'}, {'code': '000001.SZ', 'status': 'failed'}])
    result = script._run_quality_check({}, report, '2026-10-14', 'test')
    assert checker.check.call_args.kwargs['stocks'] == ['600000.SH']
    assert checker.check.call_args.kwargs['end_date'] == '2026-10-14'
    assert result['errors'] == 0


def test_no_success_does_not_launch_derived_update():
    result = script._sync_after_update(SimpleNamespace(skip_feature_sync=False), {'success': 0}, {})
    assert result['status'] == 'skipped' and 'no confirmed target-date' in result['reason']


def test_failure_summary_survives_trailing_log_and_names_true_quality_error():
    value = {'summary': {'failed': 0, 'quality_check': {'has_errors': True, 'errors': 1, 'error_details': [{'entity_id': '000016.SZ', 'category': 'missing_data'}]}}, 'rows': [{'code': '920000.BJ', 'status': 'success'}]}
    log = 'progress\n' + json.dumps(value, indent=2) + '\ntrailing text\n'
    result = _daily_update_result(log)
    message = _daily_update_failure(result['summary'], 2)
    assert '000016.SZ' in message and '数据质量' in message and '920000' not in message


@pytest.mark.parametrize('stocks', [[], ['920000.BJ']])
def test_empty_realtime_selection_never_falls_back_to_full_market(monkeypatch, stocks):
    monkeypatch.setattr(script, 'LocalCache', Mock())
    manager = Mock()
    manager.build_pool.return_value = pd.DataFrame({'stock': stocks})
    monkeypatch.setattr(script, 'StockPoolManager', Mock(return_value=manager))
    storage = Mock(); monkeypatch.setattr(script, 'get_data_access', Mock(return_value=storage))
    request = Mock(); monkeypatch.setattr(script.subprocess, 'run', request)
    result = script.update_today_realtime({}, stocks, '2026-10-14', use_progress_bar=False)
    assert result.empty
    storage.list_stocks.assert_not_called(); request.assert_not_called()
    assert result.attrs['update_universe']['eligible'] == 0


def test_update_summary_keeps_stale_dates_separate_from_target_coverage(monkeypatch):
    storage = Mock(); storage.settings = SimpleNamespace(write_db=False)
    monkeypatch.setattr(script, 'get_data_access', lambda: storage)
    report = pd.DataFrame([{'code': '600000.SH', 'status': 'success', 'last_date': '2026-10-14'}, {'code': '000016.SZ', 'status': 'skipped', 'last_date': '2026-09-03'}])
    report.attrs['update_universe'] = {'excluded': 348, 'selected': 2}
    summary = script.summarize_update(report=report, update_id='test', started_at='now', mode='local', stocks_arg='', end_date='2026-10-14')
    assert summary['total'] == 2 and summary['failed'] == 0
    assert summary['target_date_covered'] == 1 and summary['not_updated'] == 1
    assert summary['update_universe']['excluded'] == 348
