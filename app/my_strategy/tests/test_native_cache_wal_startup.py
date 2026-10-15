import sqlite3

import pytest

from my_strategy.services import czsc_native_cache as cache


def test_wal_startup_retries_only_transient_busy(monkeypatch):
    calls=[]
    class Connection:
        def execute(self,statement):
            calls.append(statement)
            if len(calls)<3:raise sqlite3.OperationalError('database is locked')
    monkeypatch.setattr(cache.time,'sleep',lambda _:None)
    cache._enable_wal(Connection())
    assert calls==['PRAGMA journal_mode=WAL']*3


@pytest.mark.parametrize('message',['disk I/O error','database disk image is malformed','readonly database'])
def test_wal_startup_does_not_hide_other_operational_errors(message,monkeypatch):
    class Connection:
        def execute(self,statement):raise sqlite3.OperationalError(message)
    monkeypatch.setattr(cache.time,'sleep',lambda _:pytest.fail('nonbusy errors cannot retry'))
    with pytest.raises(sqlite3.OperationalError,match=message):cache._enable_wal(Connection())


def test_wal_startup_retry_has_a_bound(monkeypatch):
    class Connection:
        def execute(self,statement):raise sqlite3.OperationalError('database is locked')
    with pytest.raises(sqlite3.OperationalError,match='locked'):cache._enable_wal(Connection(),timeout=0)
