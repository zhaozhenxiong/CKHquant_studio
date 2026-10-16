"""External raw snapshot summaries. No metadata writes to the protected raw DB.

The file/header observation is an invalidation token, NOT a price/model content
identity. Model and source admission must continue to verify their actual SHA.
"""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from filelock import FileLock
from my_strategy.core.paths import METADATA_ROOT


def raw_revision(path):
    result = {'path': str(Path(path).resolve()), 'contract': 'sqlite_file_observation_v1'}
    for suffix in ('', '-wal'):
        target = Path(str(path) + suffix)
        try:
            stat = target.stat()
            with target.open('rb') as stream: header = stream.read(100 if not suffix else 32)
            result[suffix or 'db'] = [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,header.hex()]
        except FileNotFoundError:
            result[suffix or 'db'] = None
    return result


def read_status(path, compute, *, refresh=False, cache_path=None):
    path = Path(path).resolve()
    target = Path(cache_path or METADATA_ROOT / 'raw_status_summary.db')
    target.parent.mkdir(parents=True, exist_ok=True)
    key = str(path)
    revision = raw_revision(path)
    with closing(sqlite3.connect(target, timeout=30)) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS raw_status(path TEXT PRIMARY KEY,revision TEXT NOT NULL,payload TEXT NOT NULL,verified_at REAL NOT NULL)')
        conn.commit()
        cached = conn.execute('SELECT revision,payload,verified_at FROM raw_status WHERE path=?', (key,)).fetchone()
    def response(row, *, stale=False, error=None):
        return {**json.loads(row[1]), 'summary_verified_at':row[2], 'summary_stale':stale,
            'summary_revision':hashlib.sha256(row[0].encode()).hexdigest(),
            'summary_error':error, 'summary_contract':'observed_committed_raw_snapshot'}
    encoded = json.dumps(revision,sort_keys=True)
    if cached and cached[0] == encoded and not refresh:
        return response(cached)
    with FileLock(str(target)+'.'+hashlib.sha256(key.encode()).hexdigest()[:16]+'.lock'):
        with closing(sqlite3.connect(target,timeout=30)) as conn:
            cached = conn.execute('SELECT revision,payload,verified_at FROM raw_status WHERE path=?', (key,)).fetchone()
        revision = raw_revision(path); encoded = json.dumps(revision,sort_keys=True)
        if cached and cached[0] == encoded and not refresh: return response(cached)
        try:
            payload = compute()
        except Exception as exc:
            if cached: return response(cached,stale=True,error=str(exc))
            raise
        now = time.time()
        row = (encoded,json.dumps(payload,ensure_ascii=False,allow_nan=False),now)
        # A concurrent/partial commit can be observed, but must never claim that
        # this older snapshot describes the new file revision.
        if raw_revision(path) != revision:
            return response(row,stale=True,error='行情在核验期间变化；摘要未发布，请刷新')
        with closing(sqlite3.connect(target,timeout=30)) as conn, conn:
            conn.execute('INSERT OR REPLACE INTO raw_status VALUES(?,?,?,?)', (key,*row))
        return response(row)
