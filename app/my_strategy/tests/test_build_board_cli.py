"""The current-only entry point never creates a historical feature cache."""
import sqlite3

import pytest

from my_strategy.scripts.build_board_features import main


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    raw=tmp_path/'raw.db'
    with sqlite3.connect(raw) as connection:
        connection.execute('CREATE TABLE stock_daily_normalized(date TEXT,has_trade_price INTEGER,source TEXT)')
        connection.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?)',
            [('2026-10-09',1,'tushare'),('2026-10-12',1,'tushare'),('2026-10-13',1,'legacy')])
    from my_strategy.services import czsc_trend_support, board_features
    monkeypatch.setattr(czsc_trend_support,'_calendar',lambda _: (['2026-10-09','2026-10-12'],'verified-calendar'))
    calls=[]
    def build(store, **kwargs):
        calls.append(kwargs)
        return dict(mode=kwargs['mode'],price_date=kwargs['price_date'],membership_date='2026-10-14',
            training_eligible=False,qualification='pending',schema_hash='schema',cache_key='cache',diagnostics={})
    monkeypatch.setattr(board_features,'build_board_features',build)
    return ['--raw-db',str(raw),'--kb-db',str(tmp_path/'basic_kb.db')],calls,tmp_path/'basic_kb.db'


def test_default_cli_builds_current_members_against_latest_qualified_price(inputs):
    args,calls,_=inputs
    assert main(args)==0
    assert len(calls)==1
    assert calls[0]['mode']=='current_reference'
    assert calls[0]['price_date']=='2026-10-12'


def test_old_price_date_is_rejected_before_cache_creation(inputs):
    args,calls,kb=inputs
    with pytest.raises(ValueError,match='最新本地行情日'):
        main(args+['--price-date','2026-10-09'])
    assert calls==[] and not kb.exists()


def test_historical_build_mode_is_not_a_cli_option(inputs):
    args,calls,kb=inputs
    with pytest.raises(SystemExit):
        main(args+['--mode','causal'])
    assert calls==[] and not kb.exists()
