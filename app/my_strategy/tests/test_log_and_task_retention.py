import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from datetime import timedelta

from my_strategy.core.log_retention import RetainedRotatingFileHandler, dashboard_logging_config
from my_strategy.core.tz import local_now
from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.tasks import TaskManager


def test_dashboard_diagnostics_bound_utf8_bytes_and_leave_audit_untouched(tmp_path):
    audit = tmp_path / 'audit.log'
    audit.write_text('permanent evidence')
    path = tmp_path / 'web.log'
    handler = RetainedRotatingFileHandler(path, maxBytes=100, backupCount=3)
    try:
        for _ in range(20):
            handler.emit(logging.LogRecord('test', logging.INFO, '', 0, '详细信息' * 100, (), None))
    finally:
        handler.close()
    files = list(tmp_path.glob('web.log*'))
    assert len(files) == 4
    assert all(file.stat().st_size <= 100 for file in files)
    assert 'truncated' in path.read_text()
    assert audit.read_text() == 'permanent evidence'


def test_seven_day_expiry_only_removes_owned_numbered_backups(tmp_path):
    path = tmp_path / 'web.log'
    for name in ('web.log', 'web.log.1', 'web.log.2', 'web.log.audit', 'another.log.1'):
        file = tmp_path / name
        file.write_text('old')
        os.utime(file, (time.time() - 8*86400,) * 2)
    handler = RetainedRotatingFileHandler(path)
    try:
        handler.emit(logging.LogRecord('test', logging.INFO, '', 0, 'current', (), None))
    finally:
        handler.close()
    assert path.read_text().strip() == 'current'
    assert not (tmp_path / 'web.log.1').exists()
    assert (tmp_path / 'web.log.audit').read_text() == 'old'
    assert (tmp_path / 'another.log.1').read_text() == 'old'
    assert dashboard_logging_config(tmp_path)['handlers']['retained']['backupCount'] == 3


def test_dashboard_log_bounds_match_bytes_with_windows_newline_default(tmp_path):
    path = tmp_path / 'web.log'
    handler = RetainedRotatingFileHandler(path, maxBytes=100, backupCount=3)
    newlines = []

    def windows_open(filename, mode, **kwargs):
        # Emulate Windows on every platform; explicit LF must survive rollover.
        newline = kwargs.pop('newline', '\r\n')
        newlines.append(newline)
        return open(filename, mode, newline=newline, **kwargs)

    handler._builtin_open = windows_open
    try:
        for _ in range(5):
            handler.emit(logging.LogRecord('test', logging.INFO, '', 0, '详细信息\n' * 100, (), None))
    finally:
        handler.close()
    files = list(tmp_path.glob('web.log*'))
    assert len(files) == 4 and len(newlines) == 5
    assert all(newline == '\n' for newline in newlines)
    for file in files:
        payload = file.read_bytes()
        assert len(payload) <= 100 and payload.endswith(b'\n') and b'\r\n' not in payload
        assert 'truncated' in payload.decode('utf-8')


def test_low_volume_logs_roll_daily_even_while_file_mtime_is_fresh(tmp_path):
    path = tmp_path / 'web.log'
    handler = RetainedRotatingFileHandler(path)
    try:
        handler.emit(logging.LogRecord('test', logging.INFO, '', 0, 'previous day', (), None))
        handler._rollover_at = time.time() - 1
        handler.emit(logging.LogRecord('test', logging.INFO, '', 0, 'new day', (), None))
    finally:
        handler.close()
    assert path.read_text().strip() == 'new day'
    assert (tmp_path / 'web.log.1').read_text().strip() == 'previous day'


def test_task_listing_limits_age_without_deleting_history_and_keeps_active(tmp_path):
    manager = TaskManager(tmp_path / 'tasks.db', ResultStore(tmp_path / 'results.db', tmp_path / 'runs'))
    now = local_now()
    entries = [('recent', 'succeeded', 29), ('old', 'succeeded', 31),
               ('failed_recent', 'failed', 89), ('failed_old', 'failed', 91),
               ('active_old', 'running', 500), ('unknown_legacy', 'succeeded', None)]
    try:
        for name, status, days in entries:
            if status == 'succeeded':
                manager.results.save('update', {'run_id': name, 'status': 'completed'})
            manager._write(dict(job_id=name, kind='update', spec={}, status=status, run_id=name,
                                created_at=(now - timedelta(days=days)).isoformat() if days is not None else 'unknown'))
        jobs = manager.list(1001)
        assert [job['job_id'] for job in jobs] == ['active_old', 'unknown_legacy', 'failed_recent', 'recent']
        assert manager.get('old')['status'] == 'succeeded'
        assert manager.get('failed_old')['status'] == 'failed'
        with sqlite3.connect(manager.db_path) as connection:
            assert connection.execute('SELECT count(*) FROM czsc_tasks').fetchone()[0] == len(entries)
            for index in range(1100):
                value = dict(job_id=str(index), kind='update', spec={}, status='succeeded', created_at=now.isoformat())
                connection.execute('INSERT INTO czsc_tasks VALUES(?,?,?,?)', (str(index), str(index), 'succeeded', json.dumps(value)))
        assert len(manager.list(5000)) == 1000
        assert manager.list(1)[0]['job_id'] == 'active_old'
    finally:
        manager.close()
