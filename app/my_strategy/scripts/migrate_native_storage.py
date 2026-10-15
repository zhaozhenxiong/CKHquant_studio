"""Audited, restartable BLOB-to-file migration; stage before maintenance publish.

No raw data, model, source record or producer identity is rewritten. The old
database remains intact until an independently readable staged index is ready.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

from my_strategy.services.czsc_native_cache import (
    NativeStructureCache, _canonical, _object_hash, _hash, _strict_json,
    _validate_identity, file_sha256, retain_file_object, share_native_file,
)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def logical_identity(connection):
    digest = hashlib.sha256(); count = total = 0
    for row in connection.execute('''SELECT cache_key,identity_json,metadata_json,metadata_sha256,
        producer_json,producer_sha256,length(features_blob),length(native_blob)
        FROM native_structure_entries ORDER BY cache_key'''):
        digest.update(_canonical(list(row)).encode()); digest.update(b'\n')
        count += 1; total += row[-2] + row[-1]
    return {'rows': count, 'blob_bytes': total, 'metadata_digest': digest.hexdigest()}


def _safe_existing(path, runs_root):
    path = Path(path)
    return (path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(runs_root))


def _find_asset(producer_path, symbol, filename, digest, size, runs_root):
    candidates = [Path(producer_path)]
    candidates.extend(sorted(runs_root.glob('*/data_snapshots/native/' + symbol.replace('.', '_') + '/' + filename)))
    for candidate in dict.fromkeys(candidates):
        if _safe_existing(candidate, runs_root) and candidate.stat().st_size == size and file_sha256(candidate) == digest:
            return candidate
    return None


def _retain_exclusive(blob, object_root, digest):
    target = object_root / digest[:2] / digest
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != digest: raise ValueError('exclusive object SHA drift')
        target.chmod(target.stat().st_mode & ~0o222)
        return target
    fd, temporary = tempfile.mkstemp(prefix='.migrate-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(blob); output.flush(); os.fsync(output.fileno())
        if file_sha256(temporary) != digest: raise ValueError('exclusive object write failed')
        os.replace(temporary, target)
        target.chmod(target.stat().st_mode & ~0o222)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return target


def stage(database, runs_root, audit_dir, staged_database=None):
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as source:
        return _stage(database, runs_root, audit_dir, staged_database, source)


def _stage(database, runs_root, audit_dir, staged_database, source):
    database, runs_root, audit_dir = Path(database).resolve(), Path(runs_root).resolve(), Path(audit_dir).resolve()
    staged_database = Path(staged_database or database.with_name(database.stem + '.file-index-stage.db')).resolve()
    if staged_database == database: raise ValueError('staging must not replace live database')
    audit_dir.mkdir(parents=True, exist_ok=True)
    object_root = database.with_suffix('.objects')
    source.execute('BEGIN')
    before = logical_identity(source)
    if source.execute('PRAGMA user_version').fetchone()[0] != 1:
        raise ValueError('migration requires original version 1 database')
    started, checked, reused_bytes, exclusive_bytes = time.time(), 0, 0, 0
    source_digest = hashlib.sha256()
    with NativeStructureCache(staged_database, object_root=object_root) as target, (audit_dir / 'native_object_migration.jsonl').open('w') as journal:
        for row in source.execute('''SELECT cache_key,identity_json,metadata_json,metadata_sha256,
                producer_json,producer_sha256,features_blob,native_blob,created_at
                FROM native_structure_entries ORDER BY cache_key'''):
            key, identity_json, metadata_json, metadata_sha, producer_json, producer_sha, features, native, created_at = row
            identity, metadata, producer = map(_strict_json, (identity_json, metadata_json, producer_json))
            _validate_identity(identity)
            if _object_hash(identity) != key or _hash(metadata_json.encode()) != metadata_sha or _hash(producer_json.encode()) != producer_sha:
                raise ValueError('legacy entry identity/metadata/producer SHA drift: ' + key)
            if set(producer) != {'raw_path', 'features_path', 'native_path'}:
                raise ValueError('legacy producer schema drift')
            retained, assets = {}, []
            for kind, blob, filename, digest in [('features', features, 'features.parquet', metadata['sha256']),
                    ('native', native, 'native.json.gz', metadata['native_sha256'])]:
                if _hash(blob) != digest: raise ValueError('legacy BLOB SHA drift: ' + key + '/' + kind)
                current = _find_asset(producer[kind + '_path'], identity['raw']['symbol'], filename, digest, len(blob), runs_root)
                if current is None:
                    destination = _retain_exclusive(blob, object_root, digest)
                    exclusive_bytes += len(blob)
                    mode = 'exclusive_blob_preserved'
                else:
                    destination = retain_file_object(current, object_root, digest, expected_bytes=len(blob))
                    reused_bytes += len(blob)
                    mode = 'verified_existing_file_hardlink'
                retained[kind + '_path'] = str(destination)
                stat = destination.stat()
                assets.append({'kind': kind, 'sha256': digest, 'bytes': len(blob), 'mode': mode,
                               'existing_path': str(current) if current else None, 'object_path': str(destination),
                               'verified_stat': {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}})
            with target.connection:
                old = target.connection.execute('SELECT identity_json,metadata_json,producer_json FROM native_structure_entries WHERE cache_key=?', (key,)).fetchone()
                if old and old != (identity_json, metadata_json, producer_json): raise ValueError('staged identity changed')
                target.connection.execute('''INSERT OR IGNORE INTO native_structure_entries VALUES(?,?,?,?,?,?,?,?,?)''',
                    (key, identity_json, metadata_json, metadata_sha, producer_json, producer_sha, b'', b'', created_at))
                target.connection.execute('INSERT OR REPLACE INTO native_structure_files VALUES(?,?,?,?)',
                    (key, retained['features_path'], retained['native_path'], _object_hash(retained)))
            proof = {'cache_key': key, 'symbol': identity['raw']['symbol'], 'data_end': identity['raw']['data_end'],
                     'metadata_sha256': metadata_sha, 'producer_sha256': producer_sha, 'assets': assets}
            line = _canonical(proof)
            journal.write(line + '\n'); source_digest.update(line.encode()); source_digest.update(b'\n')
            checked += 1
            if checked % 200 == 0:
                journal.flush()
                print(json.dumps({'stage': 'native_storage_migration', 'checked': checked, 'total': before['rows'], 'elapsed_s': round(time.time()-started, 1)}), flush=True)
        journal.flush(); os.fsync(journal.fileno())
        if target.connection.execute('SELECT count(*) FROM native_structure_entries').fetchone()[0] != checked:
            raise ValueError('staged database contains unexpected extra entries')
        if target.connection.execute('SELECT count(*) FROM native_structure_files').fetchone()[0] != checked:
            raise ValueError('staged file index coverage incomplete')
        if target.connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise ValueError('staged SQLite integrity failed')
        target.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    source.close()
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as source:
        if logical_identity(source) != before: raise ValueError('source database changed during migration')
    result = {'status': 'staged_verified', 'database': str(database), 'staged_database': str(staged_database),
              'object_root': str(object_root), 'runs_root': str(runs_root), 'source_identity': before,
              'verified_rows': checked, 'verified_existing_file_bytes': reused_bytes,
              'exclusive_blob_bytes_preserved': exclusive_bytes, 'object_journal_sha256': source_digest.hexdigest(),
              'old_database_bytes': database.stat().st_size, 'staged_database_bytes': staged_database.stat().st_size,
              'staged_database_sha256': file_sha256(staged_database),
              'free_bytes_before': shutil.disk_usage(database.parent).free, 'elapsed_seconds': time.time()-started,
              'old_source_metadata_modified': False, 'ready_for_maintenance_publish': True}
    atomic_json(audit_dir / 'native_storage_migration_stage.json', result)
    return result


def open_database_owners(database):
    import psutil
    paths = {str(database), str(database)+'-wal', str(database)+'-shm'}
    result = []
    for process in psutil.process_iter(['pid', 'name']):
        try:
            if any(item.path in paths for item in process.open_files()): result.append(process.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied): continue
    return result


def seal(audit_dir):
    """Independently recheck every staged file/index before the maintenance swap."""
    audit_dir = Path(audit_dir).resolve()
    receipt = json.loads((audit_dir/'native_storage_migration_stage.json').read_text())
    database, staged = Path(receipt['database']), Path(receipt['staged_database'])
    journal_path = audit_dir/'native_object_migration.jsonl'
    original_journal_sha = file_sha256(journal_path)
    if original_journal_sha != receipt['object_journal_sha256']: raise ValueError('original stage journal changed')
    rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
    shared_aliases = 0
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)) as old, closing(sqlite3.connect(staged.as_uri()+'?mode=ro',uri=True)) as new:
        if logical_identity(old) != receipt['source_identity']: raise ValueError('source changed before sealing')
        if len(rows) != receipt['verified_rows'] or new.execute('SELECT COUNT(*) FROM native_structure_files').fetchone()[0] != len(rows):
            raise ValueError('staged coverage differs from proof')
        for record in rows:
            columns = 'identity_json,metadata_json,metadata_sha256,producer_json,producer_sha256,created_at'
            before = old.execute('SELECT '+columns+' FROM native_structure_entries WHERE cache_key=?',(record['cache_key'],)).fetchone()
            after = new.execute('SELECT '+columns+' FROM native_structure_entries WHERE cache_key=?',(record['cache_key'],)).fetchone()
            if before != after or before is None: raise ValueError('staged source identity/metadata changed')
            paths = new.execute('SELECT features_path,native_path,paths_sha256 FROM native_structure_files WHERE cache_key=?',(record['cache_key'],)).fetchone()
            if not paths or _object_hash({'features_path':paths[0],'native_path':paths[1]}) != paths[2]: raise ValueError('staged paths changed')
            for i, asset in enumerate(record['assets']):
                path = Path(asset['object_path'])
                if str(path) != paths[i] or path.is_symlink() or not path.resolve().is_relative_to(Path(receipt['object_root']).resolve()):
                    raise ValueError('staged object path escaped or changed')
                if file_sha256(path) != asset['sha256'] or path.stat().st_size != asset['bytes']:
                    raise ValueError('staged object SHA changed before sealing')
                if asset.get('existing_path'):
                    alias = Path(asset['existing_path'])
                    if not _safe_existing(alias,Path(receipt['runs_root'])): raise ValueError('source alias changed before sealing')
                    if not os.path.samefile(alias,path):
                        shared_aliases += int(share_native_file(alias,path,asset['sha256']))
                path.chmod(path.stat().st_mode & ~0o222)
                stat = path.stat()
                asset['verified_stat']={'device':stat.st_dev,'inode':stat.st_ino,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if new.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise ValueError('staged SQLite integrity failed')
    temporary=journal_path.with_suffix('.seal-tmp')
    with temporary.open('w') as out:
        for record in rows:out.write(_canonical(record)+'\n')
        out.flush();os.fsync(out.fileno())
    temporary.replace(journal_path)
    receipt.update(object_journal_sha256=file_sha256(journal_path),staged_database_sha256=file_sha256(staged),
                   independent_seal={'objects_verified':sum(len(row['assets']) for row in rows),'source_aliases_deduplicated':shared_aliases,'original_journal_sha256':original_journal_sha,'script_sha256':file_sha256(__file__)})
    atomic_json(audit_dir/'native_storage_migration_stage.json',receipt)
    return {'status':'independently_sealed','rows':len(rows),'objects_verified':sum(len(row['assets']) for row in rows)}


def publish(audit_dir):
    audit_dir = Path(audit_dir).resolve()
    receipt = json.loads((audit_dir / 'native_storage_migration_stage.json').read_text())
    if receipt['status'] != 'staged_verified': raise ValueError('stage not verified')
    database, staged = Path(receipt['database']), Path(receipt['staged_database'])
    backup = database.with_name(database.name + '.verified-old-retirement')
    if backup.exists(): raise ValueError('unresolved previous retirement file exists')
    if file_sha256(staged) != receipt['staged_database_sha256']: raise ValueError('staged SQLite changed after verification')
    verified_journal = hashlib.sha256()
    with (audit_dir/'native_object_migration.jsonl').open() as journal:
        for line in journal:
            verified_journal.update(line.encode())
            record = json.loads(line)
            for asset in record['assets']:
                path = Path(asset['object_path'])
                if path.is_symlink() or not path.resolve().is_relative_to(Path(receipt['object_root']).resolve()):
                    raise ValueError('verified native object path changed')
                stat = path.stat()
                actual = {'device': stat.st_dev, 'inode': stat.st_ino, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
                if actual != asset['verified_stat']: raise ValueError('verified native object changed before publish')
    if verified_journal.hexdigest() != receipt['object_journal_sha256']: raise ValueError('object verification journal changed')
    owners = open_database_owners(database)
    if owners: raise RuntimeError('database maintenance requires all readers closed: ' + json.dumps(owners))
    with closing(sqlite3.connect(database)) as old:
        if logical_identity(old) != receipt['source_identity']: raise ValueError('source changed after staging')
        old.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    if open_database_owners(database): raise RuntimeError('database reopened during maintenance')
    for suffix in ['-wal', '-shm']:
        p = Path(str(database) + suffix)
        if p.exists(): raise RuntimeError('SQLite sidecar still present; keep database closed: ' + str(p))
    before_free, before_size = shutil.disk_usage(database.parent).free, database.stat().st_size
    os.replace(database, backup)
    try:
        os.replace(staged, database)
        with NativeStructureCache(database) as store:
            rows = store.connection.execute('SELECT cache_key,identity_json FROM native_structure_entries ORDER BY cache_key').fetchall()
            if len(rows) != receipt['verified_rows']: raise ValueError('published entry count mismatch')
            if store.connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise ValueError('published integrity failed')
            # All bytes were individually SHA-checked during staging. Exercise
            # full production parsers at boundary/interior keys before retirement.
            sample = sorted({0, len(rows)//4, len(rows)//2, len(rows)*3//4, len(rows)-1}) if rows else []
            for i in sample:
                if store.lookup(_strict_json(rows[i][1])) is None: raise ValueError('published reader lookup missing')
            store.connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        atomic_json(audit_dir / 'native_storage_before_retirement.json',
            {'status': 'new_index_readable_old_backup_present', 'rows': len(rows), 'sampled_full_reader': len(sample),
             'backup': str(backup), 'backup_bytes': backup.stat().st_size, 'published_bytes': database.stat().st_size})
    except BaseException:
        if database.exists(): os.replace(database, staged)
        os.replace(backup, database)
        raise
    backup.unlink()
    result = {**receipt, 'status': 'published_retired_duplicate_database', 'old_database_retired': True,
              'published_database_bytes': database.stat().st_size, 'file_bytes_reclaimed': before_size-database.stat().st_size,
              'free_bytes_before_publish': before_free, 'free_bytes_after_publish': shutil.disk_usage(database.parent).free,
              'sampled_full_reader': len(sample), 'source_run_files_deleted': 0, 'model_files_modified': 0}
    atomic_json(audit_dir / 'native_storage_migration_completed.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['stage', 'seal', 'publish'])
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--database')
    parser.add_argument('--runs-root')
    parser.add_argument('--staged-database')
    args = parser.parse_args()
    if args.action == 'stage':
        if not args.database or not args.runs_root: parser.error('stage needs --database and --runs-root')
        result = stage(args.database, args.runs_root, args.audit_dir, args.staged_database)
    elif args.action == 'seal': result = seal(args.audit_dir)
    else: result = publish(args.audit_dir)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
