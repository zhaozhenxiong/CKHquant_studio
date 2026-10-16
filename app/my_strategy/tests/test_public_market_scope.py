"""Permanent public-market scope without changing frozen numerical populations."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from my_strategy import cli
from my_strategy.adapters import czsc_adapter
from my_strategy.core import market_scope
from my_strategy.data_manager.downloader import DataDownloader
from my_strategy.data_manager import local_cache
from my_strategy.storage.access_layer import DataAccessLayer
from my_strategy.web_dashboard import api as public_api, trend_strategy
from my_strategy.web_dashboard.tasks import TaskManager
from my_strategy.tests.test_light_api import api


@pytest.mark.parametrize('symbol', [
    '920000.BJ', 'bj920000', 'BJ.920000', 'BJ:920000', 'BSE-920000',
    '920000.bse', '920000:BJ', '920000', '430047', '830799',
])
def test_beijing_aliases_rejected_before_public_dispatch(symbol):
    with pytest.raises(ValueError, match='北交所'):
        public_api.clean_symbol(symbol)


@pytest.mark.parametrize('symbol,expected', [
    ('900901.SH', '900901.SH'), ('900901', '900901.SH'), ('SH.900901', '900901.SH'),
    ('200002.SZ', '200002.SZ'), ('200002', '200002.SZ'),
    ('688001.SH', '688001.SH'), ('SZ:000001', '000001.SZ'),
])
def test_shenzhen_shanghai_and_b_shares_are_preserved(symbol, expected):
    assert public_api.clean_symbol(symbol) == expected


@pytest.mark.parametrize('path,body', [
    ('/api/analysis', {'symbol': '920000.BJ', 'analysis_method': 'structure'}),
    ('/api/analysis', {'symbol': '920000.BJ', 'analysis_method': 'structure_kronos_ma_mlp'}),
    ('/api/tasks', {'kind': 'scan', 'spec': {'symbols': ['920000.BSE'], 'analysis_method': 'legacy_reference'}}),
    ('/api/tasks', {'kind': 'scan', 'spec': {'symbols': ['BJ.920000'], 'analysis_method': 'structure_kronos_ma_l2'}}),
    ('/api/tasks', {'kind': 'update', 'spec': {'symbols': ['bj920000']}}),
    ('/api/watchlist', {'symbols': ['920000']}),
    ('/api/light/predict', {'symbol': '920000.BJ'}),
    ('/api/light/simulate', {'symbols': ['920000.BJ'], 'end': '2026-10-15'}),
    ('/api/light/score/train', {'symbols': ['920000.BJ']}),
    ('/api/light/score/observations', {'symbols': ['920000.BJ'], 'start': '2025-01-01', 'end': '2025-12-31'}),
    ('/api/light/tune', {'symbols': ['920000.BJ']}),
])
def test_http_rejects_excluded_symbols_before_starting_work(api, path, body):
    response = api.client.post(path, json=body)
    assert response.status_code == 422, response.text
    assert '北交所' in response.text
    api.light.submit.assert_not_called()
    api.app.state.tasks.submit.assert_not_called()


def test_search_filters_before_limit_and_keeps_archive_removal(api, monkeypatch):
    rows = [{'symbol': '430047.BJ', 'name': '北京'}, {'symbol': '600000.SH', 'name': '上海'},
            {'symbol': '900901.SH', 'name': '沪市B股'}]
    calls = []
    monkeypatch.setattr(czsc_adapter, 'list_symbols', lambda **kwargs: calls.append(kwargs) or rows)
    assert api.client.get('/api/symbols?q=股&limit=1').json()['symbols'] == [rows[1]]
    assert calls == [{'query': '股', 'limit': 10000}]
    assert rows[0]['symbol'] == '430047.BJ'
    api.app.state.personal = SimpleNamespace(remove_watchlist=Mock(return_value={'removed': True}))
    assert api.client.delete('/api/watchlist/430047.BJ').status_code == 200
    api.app.state.personal.remove_watchlist.assert_called_once_with('430047.BJ')


@pytest.mark.parametrize('method', ['structure', 'legacy_reference', 'structure_kronos', 'structure_kronos_ma_mlp', 'structure_kronos_ma_l2'])
@pytest.mark.parametrize('command', ['analyze', 'scan'])
def test_cli_rejects_beijing_before_any_compute_service(monkeypatch, capsys, command, method):
    from my_strategy.services import light_research
    monkeypatch.setattr(light_research, 'LightResearch', lambda: pytest.fail('opened light service'))
    monkeypatch.setattr(trend_strategy, 'scan_trend_consensus', lambda **kw: pytest.fail('started scan'))
    monkeypatch.setattr(trend_strategy, 'analyze_trend_consensus', lambda **kw: pytest.fail('started analysis'))
    assert cli.main([command, '--symbol', 'BSE.920000', '--analysis-method', method]) == 1
    assert '北交所' in capsys.readouterr().err


@pytest.mark.parametrize('method', ['structure', 'legacy_reference'])
def test_legacy_cli_empty_pool_is_fixed_before_numerical_entry(monkeypatch, capsys, method):
    monkeypatch.setattr(czsc_adapter, 'list_symbols', lambda **kw: [
        {'symbol': '430047.BJ'}, {'symbol': '600000.SH'}, {'symbol': '900901.SH'}])
    seen = []
    monkeypatch.setattr(trend_strategy, 'scan_trend_consensus', lambda **kw: seen.append(kw) or {'run_id': 'fixture'})
    assert cli.main(['scan', '--analysis-method', method]) == 0
    assert seen[0]['symbols'] == ['600000.SH', '900901.SH']


@pytest.mark.parametrize('research', [True, False])
def test_legacy_task_empty_pool_is_fixed_before_numerical_entry(monkeypatch, research):
    monkeypatch.setattr(czsc_adapter, 'list_symbols', lambda **kw: [
        {'symbol': '430047.BJ'}, {'symbol': '000001.SZ'}])
    manager = TaskManager.__new__(TaskManager)
    seen = []
    monkeypatch.setattr(trend_strategy, 'scan_trend_consensus', lambda **kw: seen.append(kw) or {'run_id': 'fixture'})
    manager._run('fixture', 'scan', {'research': research})
    assert seen[0]['symbols'] == ['000001.SZ']
    assert seen[0]['structure_only'] is (not research)


def test_only_excluded_catalog_does_not_fall_back_to_unfiltered_scan(monkeypatch):
    monkeypatch.setattr(czsc_adapter, 'list_symbols', lambda **kw: [{'symbol': '430047.BJ'}])
    with pytest.raises(ValueError, match='没有可扫描'):
        market_scope.new_scan_symbols()


@pytest.mark.parametrize('action', ['predict', 'score', 'simulate', 'tune', 'observations'])
def test_light_cli_rejects_alias_before_opening_service(monkeypatch, capsys, action):
    from my_strategy.scripts import light_research
    monkeypatch.setattr(light_research, 'LightResearch', lambda: pytest.fail('opened light service'))
    monkeypatch.setattr(sys, 'argv', ['light-research', action, '--symbols', 'bj920000'])
    with pytest.raises(SystemExit) as error:
        light_research.main()
    assert error.value.code == 2 and '北交所' in capsys.readouterr().err


def basic_frame():
    return pd.DataFrame({'code': ['920000.BJ', '600000.SH', '900901.SH', '000001.SZ'],
                         'market': ['BJ', 'SH', 'SH', 'BSE'], 'name': ['北京', '上海', 'B股', '元数据北交所']})


def test_basic_download_filters_before_coverage_minimum():
    downloader = DataDownloader({})
    downloader._provider = Mock()
    source = basic_frame()
    downloader._provider.fetch_basic.return_value = SimpleNamespace(ok=True, data=source, source='fixture')
    result = downloader.download_stock_basic(min_count=2)
    assert result.ok and result.data.code.tolist() == ['600000.SH', '900901.SH']
    assert len(source) == 4
    assert not downloader.download_stock_basic(min_count=3).ok


def test_basic_database_entry_does_not_reinsert_beijing():
    access = DataAccessLayer.__new__(DataAccessLayer)
    access.raw = Mock()
    access.save_stock_basic(basic_frame())
    assert access.raw.save_stock_basic.call_args.args[0].code.tolist() == ['600000.SH', '900901.SH']
    access.raw.reset_mock()
    assert access.save_stock_basic(basic_frame().iloc[[0, 3]]) == 0
    access.raw.save_stock_basic.assert_not_called()


def test_basic_parquet_and_database_receive_the_same_filtered_frame(monkeypatch):
    cache = local_cache.LocalCache.__new__(local_cache.LocalCache)
    cache.storage = Mock()
    cache._atomic_write_parquet = Mock()
    cache.stock_basic_path = Mock(return_value='isolated.parquet')
    monkeypatch.setattr(local_cache, 'write_db', lambda: True)
    monkeypatch.setattr(local_cache, 'write_files', lambda: True)
    cache.save_stock_basic(basic_frame())
    stored = cache.storage.save_stock_basic.call_args.args[0]
    written = cache._atomic_write_parquet.call_args.args[0]
    assert stored.code.tolist() == ['600000.SH', '900901.SH']
    pd.testing.assert_frame_equal(stored, written)
    cache.storage.reset_mock(); cache._atomic_write_parquet.reset_mock()
    cache.save_stock_basic(basic_frame().iloc[[0, 3]])
    cache.storage.save_stock_basic.assert_not_called()
    cache._atomic_write_parquet.assert_not_called()
