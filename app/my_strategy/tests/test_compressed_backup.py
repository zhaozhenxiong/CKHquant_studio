import gzip
import hashlib
import sqlite3
from pathlib import Path

import pytest

from my_strategy.storage.sqlite_backup import create_verified_backup


@pytest.mark.parametrize('compression', [None, 'gzip'])
def test_backup_restores_live_wal_without_mutating_source(tmp_path, compression):
    source = tmp_path / 'raw.db'
    with sqlite3.connect(source) as connection:
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE facts (id INTEGER PRIMARY KEY, value TEXT)')
        connection.executemany('INSERT INTO facts VALUES (?, ?)', [(i, 'verified'*100) for i in range(200)])
        connection.commit()
        before = source.read_bytes(), source.with_name('raw.db-wal').read_bytes()
        result = create_verified_backup(source, tmp_path/'backup', compression=compression)
        assert result['restore_verified'] and result['integrity_check'] == 'ok'
        assert result['table_rows'] == {'facts': 200}
        assert before == (source.read_bytes(), source.with_name('raw.db-wal').read_bytes())
        archive = Path(result['backup'])
        content = gzip.decompress(archive.read_bytes()) if compression else archive.read_bytes()
        assert hashlib.sha256(content).hexdigest() == result['restored_sha256']
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == result['sha256']
        if compression: assert result['bytes'] < result['restored_bytes']


def test_unknown_compression_never_creates_backup(tmp_path):
    with pytest.raises(ValueError, match='Unsupported'):
        create_verified_backup(tmp_path/'missing.db', tmp_path/'backup', compression='invalid')
    assert not (tmp_path/'backup').exists()
