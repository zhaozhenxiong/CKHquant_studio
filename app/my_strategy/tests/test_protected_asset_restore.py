import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from my_strategy.scripts.restore_protected_assets import restore_protected_assets
from my_strategy.scripts.snapshot_sqlite_databases import snapshot_database
from my_strategy.storage.research_store import ResearchStore
from my_strategy.tests.test_research_store_accounting import assert_accounting


def physical(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def make_backup(tmp_path, *, omit=None, legacy_unpinned=False):
    backup = tmp_path/'backup'
    live = tmp_path/'source'
    backup.mkdir()
    with ResearchStore(live, clock=lambda: 100) as store:
        account = store.put({'account': 1}, {'ledger': [{'qty': 100, 'cash': 42}]}, 'simulations', pin='account:1')
        model = store.put({'model': 1}, {'weights': 'immutable'}, 'score_models', pin='selected-model')
        shared = store.put({'alias': 1}, {'ledger': [{'qty': 100, 'cash': 42}]}, 'preview')
        omitted = store.put({'temporary': 1}, {'rebuildable': True}, 'features')
        store.acquire_lease(omitted['cache_key'], 'dead-worker', 1000)
        store.set_setting('enabled_accounts', {'1': {'result': account['cache_key']}})
        store.set_setting('current_score', model)
        if legacy_unpinned:
            store.unpin(account['cache_key'], 'account:1')
            store.unpin(model['cache_key'], 'selected-model')
    manifest = {'scope': 'protected_accounts_models_and_pointers', 'files': {}, 'databases': {}}
    for namespace, receipt in [('simulations', account), ('score_models', model)]:
        if omit == namespace:
            continue
        relative = f"metadata/light_research/objects/{receipt['sha256'][:2]}/{receipt['sha256']}.json.gz"
        target = backup/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((live/'objects'/receipt['sha256'][:2]/(receipt['sha256']+'.json.gz')).read_bytes())
        manifest['files'][relative] = physical(target)
    relative = 'runs/model-run/models/weights.pt'
    target = backup/relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b'exact-saved-model-bytes')
    manifest['files'][relative] = physical(target)
    relative = 'metadata/light_research/research.db'
    snapshot = backup/relative
    metadata = snapshot_database(live/'research.db', snapshot)
    archive = snapshot.with_suffix('.db.gz')
    archive.write_bytes(gzip.compress(snapshot.read_bytes(), mtime=0))
    metadata.update(compression='gzip', archive=archive.relative_to(backup).as_posix(),
                    archive_sha256=physical(archive)['sha256'], archive_bytes=archive.stat().st_size)
    snapshot.unlink()
    manifest['databases'][relative] = metadata
    (backup/'manifest.json').write_text(json.dumps(manifest))
    return backup, (account, model, shared, omitted)


def test_restore_prunes_only_absent_unprotected_index_and_keeps_account_model_and_shared_sha(tmp_path):
    backup, (account, model, shared, omitted) = make_backup(tmp_path)
    before = {p.relative_to(backup).as_posix(): physical(p) for p in backup.rglob('*') if p.is_file()}
    target = tmp_path/'restored'
    report = restore_protected_assets(backup, target)
    assert report['research']['removed_unprotected_results'] == 1
    assert report['research']['remaining_results'] == 3
    assert report['research']['pins_preserved'] == 2
    assert report['research']['all_result_payloads_verified']
    assert (target/'runs/model-run/models/weights.pt').read_bytes() == b'exact-saved-model-bytes'
    with ResearchStore(target/'metadata/light_research', clock=lambda: 100) as store:
        assert store.get_by_key(account['cache_key']) == {'ledger': [{'qty': 100, 'cash': 42}]}
        assert store.get_by_key(shared['cache_key']) == store.get_by_key(account['cache_key'])
        assert store.get_by_key(model['cache_key']) == {'weights': 'immutable'}
        assert store.get_by_key(omitted['cache_key']) is None
        assert store.get_setting('enabled_accounts') == {'1': {'result': account['cache_key']}}
        assert store.connection.execute('SELECT COUNT(*) FROM leases').fetchone()[0] == 0
        assert_accounting(store)
        receipt = store.put({'new': 1}, {'value': 3}, ttl_seconds=1)
        store.pin(receipt['cache_key'], 'probe')
        assert_accounting(store)
        store.unpin(receipt['cache_key'], 'probe')
        store.clock = lambda: 102
        store.gc(False)
        assert store.get_by_key(receipt['cache_key']) is None
        assert store.get_by_key(account['cache_key']) is not None
        assert_accounting(store)
    assert before == {p.relative_to(backup).as_posix(): physical(p) for p in backup.rglob('*') if p.is_file()}


@pytest.mark.parametrize('omitted_namespace', ['simulations', 'score_models'])
def test_restore_refuses_missing_permanent_assets_without_publishing(tmp_path, omitted_namespace):
    backup, _ = make_backup(tmp_path, omit=omitted_namespace)
    with pytest.raises(ValueError, match='Protected research object'):
        restore_protected_assets(backup, tmp_path/'restored')
    assert not (tmp_path/'restored').exists()
    assert not list(tmp_path.glob('.protected-restore-*'))


def test_model_corruption_fails_without_touching_destination_or_original_manifest(tmp_path):
    backup, _ = make_backup(tmp_path)
    expected_manifest = (backup/'manifest.json').read_bytes()
    (backup/'runs/model-run/models/weights.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='asset SHA'):
        restore_protected_assets(backup, tmp_path/'restored')
    assert not (tmp_path/'restored').exists()
    assert (backup/'manifest.json').read_bytes() == expected_manifest


def test_archive_corruption_and_existing_destination_refused(tmp_path):
    backup, _ = make_backup(tmp_path)
    destination = tmp_path/'existing'
    destination.mkdir()
    (destination/'keep').write_text('protected')
    with pytest.raises(FileExistsError):
        restore_protected_assets(backup, destination)
    assert (destination/'keep').read_text() == 'protected'
    (backup/'metadata/light_research/research.db.gz').write_bytes(b'bad archive')
    with pytest.raises(ValueError, match='database SHA'):
        restore_protected_assets(backup, tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


def test_manifest_path_escape_refused_and_external_file_untouched(tmp_path):
    backup, _ = make_backup(tmp_path)
    outside = tmp_path/'outside'
    outside.write_bytes(b'private')
    path = backup/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files']['../outside'] = physical(outside)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsafe backup path'):
        restore_protected_assets(backup, tmp_path/'restored')
    assert outside.read_bytes() == b'private'
    assert not (tmp_path/'restored').exists()


def test_recovery_pins_keep_unpinned_legacy_account_and_model_after_expiry_and_gc(tmp_path):
    backup, (account, model, _, _) = make_backup(tmp_path, legacy_unpinned=True)
    report = restore_protected_assets(backup, tmp_path/'restored')
    assert report['research']['pins_preserved'] == 0
    assert report['research']['added_recovery_pins'] == 2
    with ResearchStore(tmp_path/'restored/metadata/light_research', clock=lambda: 10**10) as store:
        store.gc(False)
        assert store.get_by_key(account['cache_key']) == {'ledger': [{'qty': 100, 'cash': 42}]}
        assert store.get_by_key(model['cache_key']) == {'weights': 'immutable'}
        assert store.is_pinned(account['cache_key'])
        assert_accounting(store)


def test_restore_fsync_uses_writable_temporary_copies_and_keeps_backup_read_only(tmp_path, monkeypatch):
    """Model Windows FlushFileBuffers' write-access requirement on every OS."""
    backup, _ = make_backup(tmp_path)
    before = {p.relative_to(backup).as_posix(): physical(p) for p in backup.rglob('*') if p.is_file()}
    original_open, original_fsync = Path.open, os.fsync
    streams, opened, flushed = {}, [], []
    def tracked_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get('mode', 'r')
        streams[stream.fileno()] = (path, stream, mode)
        opened.append((path, mode))
        return stream
    def windows_fsync(fd):
        path, stream, mode = streams[fd]
        if not stream.writable():
            raise PermissionError('FlushFileBuffers requires GENERIC_WRITE')
        assert not path.is_relative_to(backup)
        flushed.append((path, mode))
        return original_fsync(fd)
    monkeypatch.setattr(Path, 'open', tracked_open)
    monkeypatch.setattr(os, 'fsync', windows_fsync)
    target = tmp_path / 'restored'
    result = restore_protected_assets(backup, target)
    assert result['research']['all_result_payloads_verified']
    final_flushes = [(path.name, mode) for path, mode in flushed if mode == 'r+b']
    assert final_flushes == [('research.db', 'r+b'), ('restore-receipt.json', 'r+b')]
    assert all(not any(flag in mode for flag in ('w', 'a', '+')) for path, mode in opened
               if path.is_relative_to(backup))
    assert before == {p.relative_to(backup).as_posix(): physical(p) for p in backup.rglob('*') if p.is_file()}
    assert (target / 'restore-receipt.json').is_file()
