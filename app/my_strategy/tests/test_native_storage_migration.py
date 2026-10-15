import json
from pathlib import Path
import sqlite3

import pytest

from my_strategy.services.czsc_native_cache import NativeStructureCache, file_sha256
from my_strategy.scripts.migrate_native_storage import stage, publish
from my_strategy.tests.test_native_structure_cache import candidate, legacy_entry


def old_database(candidate, tmp_path):
    record, _, _, identity = candidate
    database = tmp_path / 'native_structure.db'
    with NativeStructureCache(database) as store:
        store.store(identity, record)
        entry = legacy_entry(store, identity)
        with store.connection:
            store.connection.execute('DROP TABLE native_structure_files')
            store.connection.execute('DROP TABLE native_structure_settings')
            store.connection.execute('PRAGMA user_version=1')
    return database, record, identity, entry


def test_stage_and_publish_preserve_source_bytes_and_exact_payload(candidate, tmp_path):
    database, record, identity, before = old_database(candidate, tmp_path)
    audit = tmp_path / 'audit'
    original_sha = file_sha256(database)
    result = stage(database, tmp_path, audit)
    assert result['verified_rows'] == 1
    assert result['exclusive_blob_bytes_preserved'] == 0
    assert file_sha256(database) == original_sha
    done = publish(audit)
    assert done['old_database_retired']
    assert not database.with_name(database.name + '.verified-old-retirement').exists()
    with NativeStructureCache(database) as store:
        after = store.lookup(identity)
        assert after.features_blob == before.features_blob
        assert after.native_blob == before.native_blob
        assert after.producer == before.producer
        assert store.connection.execute('SELECT sum(length(features_blob)+length(native_blob)) FROM native_structure_entries').fetchone()[0] == 0
    assert file_sha256(record['native_path']) == record['native_sha256']


def test_exclusive_legacy_blobs_are_preserved_before_old_database_retirement(candidate, tmp_path):
    database, record, identity, before = old_database(candidate, tmp_path)
    Path(record['path']).unlink(); Path(record['native_path']).unlink()
    audit = tmp_path / 'audit'
    result = stage(database, tmp_path, audit)
    assert result['exclusive_blob_bytes_preserved'] == len(before.features_blob) + len(before.native_blob)
    publish(audit)
    with NativeStructureCache(database) as store:
        assert store.lookup(identity).native_blob == before.native_blob


def test_corrupt_blob_prevents_staging_and_keeps_original_database(candidate, tmp_path):
    database, _, _, _ = old_database(candidate, tmp_path)
    with sqlite3.connect(database) as conn:
        conn.execute('UPDATE native_structure_entries SET native_blob=?', (b'corrupted',))
    before = file_sha256(database)
    with pytest.raises(ValueError, match='BLOB SHA drift'): stage(database, tmp_path, tmp_path/'audit')
    assert file_sha256(database) == before
    assert not (tmp_path/'audit/native_storage_migration_stage.json').exists()


def test_database_change_after_staging_prevents_publish(candidate, tmp_path):
    database, _, _, _ = old_database(candidate, tmp_path)
    audit = tmp_path/'audit'; stage(database, tmp_path, audit)
    conn = sqlite3.connect(database)
    conn.execute("UPDATE native_structure_entries SET producer_json='{}'"); conn.commit(); conn.close()
    with pytest.raises(ValueError, match='source changed'): publish(audit)
    assert database.exists()


def test_changed_object_after_seal_blocks_retirement(candidate,tmp_path):
    database,record,identity,_=old_database(candidate,tmp_path)
    audit=tmp_path/'audit';receipt=stage(database,tmp_path,audit)
    original=file_sha256(database)
    object_path=Path(json.loads((audit/'native_object_migration.jsonl').read_text().splitlines()[0])['assets'][1]['object_path'])
    object_path.chmod(0o644);object_path.write_bytes(b'changed after verification')
    with pytest.raises(ValueError,match='object changed'):publish(audit)
    assert file_sha256(database)==original
