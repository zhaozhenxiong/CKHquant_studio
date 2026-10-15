from pathlib import Path
import copy
import shutil
import sqlite3

import pytest

from my_strategy.services import czsc_native_cache as cache
from my_strategy.tests.test_native_structure_cache import candidate, legacy_entry


def test_new_storage_has_no_blob_duplicate_and_reuse_links_same_inode(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        assert store.connection.execute('SELECT length(features_blob),length(native_blob) FROM native_structure_entries').fetchone() == (0, 0)
        assert Path(entry.native_path).stat().st_ino == Path(record['native_path']).stat().st_ino
        assert Path(entry.native_path).stat().st_mode & 0o222 == 0
        result = entry.materialize(tmp_path / 'another-run', record)
        assert Path(result['native_path']).stat().st_ino == Path(entry.native_path).stat().st_ino
        assert Path(result['path']).stat().st_ino == Path(entry.features_path).stat().st_ino


def test_native_file_change_is_detected_not_silently_recomputed(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        Path(entry.native_path).chmod(0o644)
        Path(entry.native_path).write_bytes(b'changed')
        with pytest.raises(ValueError, match='SHA drift'): store.lookup(identity)


def test_same_database_reads_legacy_and_file_entries(candidate, tmp_path):
    record, _, _, identity = candidate
    path = tmp_path / 'cache.db'
    with cache.NativeStructureCache(path) as store:
        entry = store.store(identity, record)
        legacy_entry(store, identity)
    # Authentic v1 has no extra tables and user_version 1.
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TABLE native_structure_files')
        conn.execute('DROP TABLE native_structure_settings')
        conn.execute('PRAGMA user_version=1')
    with cache.NativeStructureCache(path) as store:
        restored = store.lookup(identity)
        assert restored.native_blob == entry.native_blob
        assert restored.native_path == ''


def test_deleted_run_file_does_not_break_owned_content_link(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        store.store(identity, record)
        Path(record['path']).unlink()
        Path(record['native_path']).unlink()
        assert store.lookup(identity) is not None


def test_file_index_tampering_and_symlink_rejected(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        store.store(identity, record)
        with store.connection:
            store.connection.execute('UPDATE native_structure_files SET native_path=?', (str(tmp_path / 'outside'),))
        with pytest.raises(ValueError, match='paths SHA drift'): store.lookup(identity)


def test_equal_content_under_distinct_identities_deduplicates_producer_file(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        first=store.store(identity,record)
        second_dir=tmp_path/'second-producer';second_dir.mkdir()
        second={**record,'path':str(second_dir/'features.parquet'),'native_path':str(second_dir/'native.json.gz')}
        shutil.copyfile(record['path'],second['path']);shutil.copyfile(record['native_path'],second['native_path'])
        changed=copy.deepcopy(identity);changed['code_hashes']['services/generator.py']='f'*64
        other=store.store(changed,second)
        assert first.cache_key != other.cache_key
        assert Path(first.native_path).stat().st_ino == Path(second['native_path']).stat().st_ino
        assert len(list(store.object_root.glob('*/*'))) == 2
