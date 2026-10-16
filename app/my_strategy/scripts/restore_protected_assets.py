"""Restore a selective accounts/models backup into a new isolated directory.

Never point this at a live application. The original backup remains unchanged.
This does not restore raw data, task history or all scan artifacts. Research
index references to intentionally omitted, unprotected cache objects are removed
only in the new copy, so they become genuine misses instead of corrupt hits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from my_strategy.storage.research_store import ResearchStore, canonical

PROTECTED_NAMESPACES = {'simulations', 'result_lookup', 'score_models', 'mlp_score_models',
                        'simulation_checkpoints', 'score_observation_manifests'}
RESEARCH_DB = 'metadata/light_research/research.db'


def _physical(path):
    with Path(path).open('rb') as reader:
        return {'sha256': hashlib.file_digest(reader, 'sha256').hexdigest(),
                'size_bytes': os.fstat(reader.fileno()).st_size}


def _path(root, relative):
    part = Path(relative)
    if part.is_absolute() or not part.parts or '..' in part.parts or '.' in part.parts:
        raise ValueError('Unsafe backup path')
    current = root
    for piece in part.parts:
        current = current / piece
        if current.is_symlink():
            raise ValueError('Backup symlinks are not supported')
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError('Backup path escapes root')
    return current


def _copy(source, target, *, compressed=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compressed else open
    with opener(source, 'rb') as reader, target.open('xb') as writer:
        shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _check_database(path):
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as connection:
        if [row[0] for row in connection.execute('PRAGMA integrity_check')] != ['ok']:
            raise ValueError('Restored database integrity failed')
        if connection.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Restored database has broken foreign keys')


def _check_accounting(connection):
    expected = connection.execute('''SELECT COALESCE(SUM(size_bytes),0),
        COALESCE(SUM(CASE WHEN NOT EXISTS(SELECT 1 FROM results r JOIN pins p USING(cache_key)
        WHERE r.sha256=o.sha256) THEN size_bytes ELSE 0 END),0) FROM objects o''').fetchone()
    actual = connection.execute('SELECT total_bytes,cache_bytes FROM research_totals WHERE id=1').fetchone()
    if tuple(actual) != tuple(expected):
        raise ValueError('Restored capacity counters differ')
    for obj in connection.execute('SELECT * FROM objects').fetchall():
        sha = obj['sha256']
        result = connection.execute('''SELECT COUNT(*),COALESCE(MAX(expires_at),0),COALESCE(MAX(accessed_at),0)
            FROM results WHERE sha256=?''', (sha,)).fetchone()
        pins = connection.execute('SELECT COUNT(*) FROM results JOIN pins USING(cache_key) WHERE sha256=?', (sha,)).fetchone()[0]
        lease = connection.execute('''SELECT COALESCE(MAX(l.expires_at),0) FROM results r JOIN leases l USING(cache_key)
            WHERE r.sha256=?''', (sha,)).fetchone()[0]
        usage = connection.execute('SELECT * FROM object_usage WHERE sha256=?', (sha,)).fetchone()
        expected_usage = (obj['size_bytes'], obj['state'], *tuple(result), pins, lease)
        actual_usage = (usage['size_bytes'], usage['state'], usage['ref_count'], usage['expires_at'],
                        usage['accessed_at'], usage['pin_count'], usage['lease_until']) if usage else None
        if expected_usage != actual_usage:
            raise ValueError('Restored reference/lease counters differ')
    if connection.execute('SELECT COUNT(*) FROM object_usage').fetchone()[0] != connection.execute('SELECT COUNT(*) FROM objects').fetchone()[0]:
        raise ValueError('Restored accounting contains stale objects')
    return {'total_bytes': actual[0], 'cache_bytes': actual[1]}


def repair_research_copy(root, *, recovery_owner):
    """Normalize only a private, verified restore copy; never runs GC."""
    with ResearchStore(root) as store:
        connection = store.connection
        pins_before = [tuple(row) for row in connection.execute('SELECT * FROM pins ORDER BY cache_key,owner')]
        settings_before = [tuple(row) for row in connection.execute('SELECT * FROM settings ORDER BY name')]
        rows = connection.execute('SELECT r.*,o.state,o.size_bytes FROM results r JOIN objects o USING(sha256)').fetchall()
        pinned_keys = {row[0] for row in pins_before}
        available, missing = set(), []
        for obj in connection.execute('SELECT * FROM objects').fetchall():
            path = store._path(obj['sha256'])
            if not path.exists():
                missing.append(obj['sha256'])
                continue
            if _physical(path) != {'sha256': obj['sha256'], 'size_bytes': obj['size_bytes']}:
                raise ValueError('Restored research object SHA/size mismatch')
            with gzip.open(path, 'rt', encoding='utf-8') as reader:
                json.load(reader)
            available.add(obj['sha256'])
        removed = [row for row in rows if row['sha256'] not in available]
        if any(row['namespace'] in PROTECTED_NAMESPACES or row['cache_key'] in pinned_keys for row in removed):
            raise ValueError('Protected research object was omitted; refuse partial recovery')
        for row in rows:
            if row['sha256'] in available:
                if row['state'] != 'active' or store.cache_key(json.loads(row['identity_json']), row['namespace']) != row['cache_key']:
                    raise ValueError('Restored result identity or state differs')
        # A pending deletion of a retained payload could silently destroy it on
        # first startup. Such an inconsistent snapshot must fail closed.
        if any(row[0] in available for row in connection.execute("SELECT sha256 FROM gc_journal WHERE state='pending'")):
            raise ValueError('Retained research object has a pending deletion')
        recovery_keys = [row['cache_key'] for row in rows if row['sha256'] in available
                         and row['namespace'] in PROTECTED_NAMESPACES and row['cache_key'] not in pinned_keys]
        with store._tx():
            connection.executemany('''INSERT INTO result_retirements VALUES(?,?,?,?,?)''',
                [(row['cache_key'], row['namespace'], row['sha256'], 'omitted_from_selective_backup', time.time()) for row in removed])
            connection.executemany('DELETE FROM results WHERE cache_key=?', [(row['cache_key'],) for row in removed])
            connection.executemany('DELETE FROM objects WHERE sha256=?', [(sha,) for sha in missing])
            connection.execute("UPDATE gc_journal SET state='deleted',completed_at=?,error=NULL WHERE state='pending'", (time.time(),))
            # Older saved results can lack modern pins. The recovery copy gives
            # these explicit permanent ownership before any future GC runs.
            connection.executemany('INSERT INTO pins(cache_key,owner,created_at) VALUES(?,?,?)',
                [(key, recovery_owner, time.time()) for key in recovery_keys])
        pins_after = {tuple(row) for row in connection.execute('SELECT * FROM pins')}
        if not set(pins_before).issubset(pins_after):
            raise ValueError('Restored pins changed')
        if settings_before != [tuple(row) for row in connection.execute('SELECT * FROM settings ORDER BY name')]:
            raise ValueError('Restored account/model pointers changed')
        counts = _check_accounting(connection)
        remaining = connection.execute('SELECT cache_key,namespace,sha256 FROM results').fetchall()
        if any(row['sha256'] not in available for row in remaining):
            raise ValueError('Restored results still have absent payloads')
        # Independently exercise the public reader for every permanent pin.
        for key in pinned_keys | set(recovery_keys):
            if store.get_by_key(key, touch=False) is None:
                raise ValueError('Pinned restored result is unreadable')
        result = {'remaining_results': len(remaining), 'verified_payloads': len(available),
                  'remaining_namespaces': dict(Counter(row['namespace'] for row in remaining)),
                  'removed_unprotected_results': len(removed),
                  'removed_namespaces': dict(Counter(row['namespace'] for row in removed)),
                  'removed_keys_sha256': hashlib.sha256(canonical(sorted(row['cache_key'] for row in removed)).encode()).hexdigest(),
                  'pins_preserved': len(pins_before), 'added_recovery_pins': len(recovery_keys),
                  'recovery_pin_owner': recovery_owner, 'settings_preserved': len(settings_before),
                  'accounting': counts, 'all_result_payloads_verified': True}
        connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        if connection.execute('PRAGMA journal_mode=DELETE').fetchone()[0].lower() != 'delete':
            raise ValueError('Restored index is not a self-contained database')
    _check_database(root/'research.db')
    return result


def restore_protected_assets(backup, destination):
    backup, destination = Path(backup).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Restore destination must be new')
    if destination.resolve().is_relative_to(backup) or backup.is_relative_to(destination.resolve()):
        raise ValueError('Restore destination overlaps backup')
    manifest_path = _path(backup, 'manifest.json')
    manifest_before = _physical(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('scope') != 'protected_accounts_models_and_pointers' or RESEARCH_DB not in manifest['databases']:
        raise ValueError('Unsupported protected backup scope')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.protected-restore-', dir=destination.parent))
    try:
        for relative, expected in manifest['files'].items():
            source, target = _path(backup, relative), _path(temporary, relative)
            if _physical(source) != expected:
                raise ValueError('Backup asset SHA/size mismatch')
            _copy(source, target)
            if _physical(target) != expected or _physical(source) != expected:
                raise ValueError('Backup asset changed during restoration')
        for relative, expected in manifest['databases'].items():
            compressed = expected.get('compression') == 'gzip'
            if expected.get('compression') not in (None, 'gzip'):
                raise ValueError('Unsupported backup compression')
            source = _path(backup, expected['archive'] if compressed else relative)
            source_sha = expected['archive_sha256'] if compressed else expected['snapshot_sha256']
            if _physical(source)['sha256'] != source_sha:
                raise ValueError('Backup database SHA mismatch')
            target = _path(temporary, relative)
            _copy(source, target, compressed=compressed)
            if _physical(target)['sha256'] != expected['snapshot_sha256'] or _physical(source)['sha256'] != source_sha:
                raise ValueError('Database restore SHA mismatch')
            _check_database(target)
        research = repair_research_copy(temporary/'metadata/light_research',
            recovery_owner='protected-recovery:' + manifest_before['sha256'])
        if _physical(manifest_path) != manifest_before:
            raise ValueError('Backup manifest changed during restoration')
        report = {'schema': 'protected_asset_restore_v2', 'backup': str(backup),
                  'backup_manifest_sha256': manifest_before['sha256'], 'destination': str(destination),
                  'scope': manifest['scope'], 'copied_asset_files': len(manifest['files']),
                  'databases': {name: _physical(_path(temporary, name)) for name in manifest['databases']},
                  'research': research, 'source_backup_unchanged': True,
                  'limitations': ['Selective accounts/models/pointers recovery; raw database, task history and all scan artifacts require their own recovery points.',
                                  'Does not bypass frozen account code identities or authorize next-day continuation.',
                                  'Install only while all writers are stopped; this tool never installs into the live workspace.']}
        receipt = temporary/'restore-receipt.json'
        receipt.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
        # Flush the normalized index and receipt before publishing the directory.
        # Windows FlushFileBuffers needs write access; these are new temporary
        # copies only. The original backup continues to be opened read-only.
        for relative in [RESEARCH_DB, 'restore-receipt.json']:
            with _path(temporary, relative).open('r+b') as output:
                os.fsync(output.fileno())
        if destination.exists():
            raise FileExistsError('Restore destination appeared during restoration')
        temporary.rename(destination)
        return report
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(restore_protected_assets(args.backup, args.destination), ensure_ascii=False))


if __name__ == '__main__':
    main()
