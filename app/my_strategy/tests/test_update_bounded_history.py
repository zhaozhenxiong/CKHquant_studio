"""Daily planning must not materialize an already-current universe."""
from unittest.mock import Mock

import pandas as pd

from my_strategy.data_manager.updater import DailyUpdater
from my_strategy.storage.raw_repository import RawMarketRepository


def test_current_universe_reads_summaries_without_price_frames():
    codes = [f'{i:06d}.SZ' for i in range(1, 2001)]
    updater = DailyUpdater.__new__(DailyUpdater)
    updater.config = {'update': {'start_date': '2015-01-01'}}
    updater.workers = 4
    updater.cache = Mock()
    updater.cache.stock_daily_update_summaries.return_value = {
        s: {'last_date': '2026-10-15', 'rows': 2800, 'has_trade_prices': True} for s in codes}
    updater.cache.read_stock_daily.side_effect = AssertionError('full history read during current-universe planning')
    updater.downloader = Mock()
    result = updater.update_batch(codes, end_date='2026-10-15')
    assert len(result) == 2000
    assert result['rows'].eq(2800).all()
    assert result.status.eq('success').all()
    updater.downloader.download_stocks_daily.assert_not_called()


def test_database_summary_counts_normalized_not_audit_rows(tmp_path):
    repository = RawMarketRepository(tmp_path/'raw.db')
    frame = pd.DataFrame({'date': ['2026-10-14', '2026-10-15'],
        'open': [10., 11.], 'high': [10.5, 11.5], 'low': [9.5, 10.5], 'close': [10., 11.],
        'volume': [1000, 1100], 'amount': [10000, 12100], 'trade_open': [10., 11.],
        'trade_high': [10.5, 11.5], 'trade_low': [9.5, 10.5], 'trade_close': [10., 11.]})
    repository.upsert_stock_daily('000001.SZ', frame, source='sina_unadjusted_v1')
    repository.upsert_stock_daily('000001.SZ', frame.iloc[-1:], source='sina_unadjusted_v1')
    assert repository.stock_daily_update_summaries(['000001.SZ', '000002.SZ']) == {
        '000001.SZ': {'last_date': '2026-10-15', 'rows': 2, 'has_trade_prices': True}}


def test_empty_stock_before_configured_start_is_a_zero_row_skip():
    updater = DailyUpdater.__new__(DailyUpdater)
    updater.config = {'update': {'start_date': '2020-01-01'}}
    updater.workers = 1
    updater.cache = Mock()
    updater.cache.stock_daily_update_summaries.return_value = {}
    updater.downloader = Mock()
    result = updater.update_batch(['000001.SZ'], end_date='2019-12-31')
    assert result.iloc[0]['rows'] == 0 and result.iloc[0]['last_date'] == ''
    updater.downloader.download_stocks_daily.assert_not_called()
