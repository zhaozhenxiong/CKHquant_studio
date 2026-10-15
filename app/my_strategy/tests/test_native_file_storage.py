from pathlib import Path
import copy
import errno
import os
import shutil
import sqlite3
import stat

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
        entry = store.store(identity, record)
        cache.remove_native_alias(record['path'], entry.features_path, record['sha256'])
        cache.remove_native_alias(record['native_path'], entry.native_path, record['native_sha256'])
        assert store.lookup(identity) is not None
        assert Path(entry.features_path).stat().st_mode & 0o222 == 0
        assert Path(entry.native_path).stat().st_mode & 0o222 == 0


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


@pytest.mark.parametrize('windows_error', [False, True])
def test_cross_volume_store_and_materialize_copy_only_verified_bytes(candidate, tmp_path, monkeypatch, windows_error):
    record, _, _, identity = candidate
    link = os.link
    calls = []
    def cross_volume(source, destination, *args, **kwargs):
        calls.append((str(source), str(destination)))
        if not Path(source).name.startswith('.native-copy-'):
            error = OSError(errno.EXDEV if not windows_error else errno.EACCES, 'different volume')
            if windows_error: error.winerror = 17
            raise error
        return link(source, destination, *args, **kwargs)
    monkeypatch.setattr(cache.os, 'link', cross_volume)
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        result = entry.materialize(tmp_path / 'another-volume', record)
        for source, retained, materialized, digest in [
            (record['path'], entry.features_path, result['path'], record['sha256']),
            (record['native_path'], entry.native_path, result['native_path'], record['native_sha256']),
        ]:
            assert {cache.file_sha256(path) for path in (source, retained, materialized)} == {digest}
            assert not os.path.samefile(source, retained)
            assert not os.path.samefile(retained, materialized)
            assert all(Path(path).stat().st_mode & 0o222 == 0 for path in (source, retained, materialized))
        assert store.lookup(identity).metadata == entry.metadata
        assert len(list(store.object_root.glob('*/*'))) == 2
        assert len([source for source, _ in calls if Path(source).name.startswith('.native-copy-')]) == 4
        assert not list(tmp_path.rglob('.native-copy-*'))
        assert not list(tmp_path.rglob('.native-link-*'))


def test_cross_volume_copy_hash_mismatch_never_publishes(candidate, tmp_path, monkeypatch):
    record, _, _, _ = candidate
    source = Path(record['path']); target = tmp_path / 'object'
    def cross_volume(*args, **kwargs): raise OSError(errno.EXDEV, 'different volume')
    monkeypatch.setattr(cache.os, 'link', cross_volume)
    monkeypatch.setattr(cache.shutil, 'copyfileobj', lambda stream, output, **kwargs: output.write(b'partial'))
    with pytest.raises(ValueError, match='copy SHA mismatch'):
        cache._link_verified_file(source, target, record['sha256'])
    assert not target.exists()
    assert not list(tmp_path.glob('.native-copy-*'))
    assert cache.file_sha256(source) == record['sha256']


def test_cross_volume_copy_never_overwrites_a_concurrent_different_object(candidate, tmp_path, monkeypatch):
    record, _, _, _ = candidate
    target = tmp_path / 'raced-object'
    def concurrent_publication(source, destination, *args, **kwargs):
        if not Path(source).name.startswith('.native-copy-'):
            raise OSError(errno.EXDEV, 'different volume')
        Path(destination).write_bytes(b'other publisher')
        raise FileExistsError(errno.EEXIST, 'already published')
    monkeypatch.setattr(cache.os, 'link', concurrent_publication)
    with pytest.raises(ValueError, match='publication race drift'):
        cache._link_verified_file(record['path'], target, record['sha256'])
    assert target.read_bytes() == b'other publisher'
    assert not list(tmp_path.glob('.native-copy-*'))


def test_hardlink_io_error_is_not_hidden_by_copy_fallback(candidate, tmp_path, monkeypatch):
    record, _, _, _ = candidate
    error = PermissionError(errno.EACCES, 'ACL denies publication'); error.winerror = 5
    def denied(*args, **kwargs): raise error
    monkeypatch.setattr(cache.os, 'link', denied)
    monkeypatch.setattr(cache.shutil, 'copyfileobj', lambda *args, **kwargs: pytest.fail('unexpected copy'))
    with pytest.raises(PermissionError) as raised:
        cache._link_verified_file(record['path'], tmp_path / 'object', record['sha256'])
    assert raised.value is error
    assert not list(tmp_path.glob('.native-copy-*'))


@pytest.mark.parametrize('delete_fails', [False, True])
def test_windows_readonly_alias_cleanup_restores_retained_mode(candidate, tmp_path, monkeypatch, delete_fails):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        source, retained = Path(record['path']), Path(entry.features_path)
        unlink = Path.unlink
        def windows_unlink(path, *args, **kwargs):
            if path == source:
                if not path.stat().st_mode & stat.S_IWUSR:
                    error = PermissionError(errno.EACCES, 'readonly'); error.winerror = 5
                    raise error
                if delete_fails:
                    error = PermissionError(errno.EACCES, 'in use'); error.winerror = 32
                    raise error
            return unlink(path, *args, **kwargs)
        monkeypatch.setattr(Path, 'unlink', windows_unlink)
        if delete_fails:
            with pytest.raises(PermissionError, match='in use'):
                cache.remove_native_alias(source, retained, record['sha256'])
            assert source.exists() and source.stat().st_mode & 0o222 == 0
        else:
            cache.remove_native_alias(source, retained, record['sha256'])
            assert not source.exists()
        assert retained.stat().st_mode & 0o222 == 0
        assert cache.file_sha256(retained) == record['sha256']
        assert store.lookup(identity) is not None


def test_alias_removal_rejects_object_itself_and_drift(candidate, tmp_path):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        with pytest.raises(ValueError, match='SHA/path mismatch'):
            cache.remove_native_alias(entry.features_path, entry.features_path, record['sha256'])
        with pytest.raises(ValueError, match='SHA/path mismatch'):
            cache.remove_native_alias(record['path'], entry.features_path, '0' * 64)
        assert Path(record['path']).exists()
        assert store.lookup(identity) is not None


def test_materialize_failure_cleans_readonly_aliases_and_preserves_objects(candidate, tmp_path, monkeypatch):
    record, _, _, identity = candidate
    with cache.NativeStructureCache(tmp_path / 'cache.db') as store:
        entry = store.store(identity, record)
        link = os.link; unlink = Path.unlink
        def fail_second(source, destination, *args, **kwargs):
            if Path(destination).name == 'native.json.gz': raise OSError(errno.EIO, 'second file failed')
            return link(source, destination, *args, **kwargs)
        def windows_unlink(path, *args, **kwargs):
            if path.name == 'features.parquet' and not path.stat().st_mode & stat.S_IWUSR:
                error = PermissionError(errno.EACCES, 'readonly'); error.winerror = 5
                raise error
            return unlink(path, *args, **kwargs)
        monkeypatch.setattr(cache.os, 'link', fail_second)
        monkeypatch.setattr(Path, 'unlink', windows_unlink)
        with pytest.raises(OSError, match='second file failed'):
            entry.materialize(tmp_path / 'failed-materialization', record)
        assert list((tmp_path / 'failed-materialization').iterdir()) == []
        assert all(Path(path).stat().st_mode & 0o222 == 0 for path in (entry.features_path, entry.native_path))
        assert store.lookup(identity) is not None


def test_alias_replace_failure_restores_readonly_and_removes_temporary_link(tmp_path, monkeypatch):
    source, retained = tmp_path / 'features.parquet', tmp_path / 'object'
    source.write_bytes(b'verified'); retained.write_bytes(b'verified')
    source.chmod(0o444); retained.chmod(0o444)
    digest = cache.file_sha256(source); unlink = Path.unlink
    def fail_replace(*args, **kwargs): raise OSError(errno.EIO, 'replace failed')
    def windows_unlink(path, *args, **kwargs):
        if path.name.startswith('.native-link-') and not path.stat().st_mode & stat.S_IWUSR:
            error = PermissionError(errno.EACCES, 'readonly'); error.winerror = 5
            raise error
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(cache.os, 'replace', fail_replace)
    monkeypatch.setattr(Path, 'unlink', windows_unlink)
    with pytest.raises(OSError, match='replace failed'):
        cache.share_native_file(source, retained, digest)
    assert source.read_bytes() == retained.read_bytes() == b'verified'
    assert source.stat().st_mode & 0o222 == retained.stat().st_mode & 0o222 == 0
    assert not list(tmp_path.glob('.native-link-*'))
