"""Consistent SQLite backup with an isolated restore drill before publication."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from uuid import uuid4


def _check(connection: sqlite3.Connection) -> dict[str, int]:
    checks = [row[0] for row in connection.execute("PRAGMA integrity_check")]
    if checks != ["ok"]:
        raise ValueError(f"SQLite integrity check failed: {checks}")
    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {name: connection.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0] for name in tables}


def create_verified_backup(source: Path, backup_root: Path, *, compression: str | None = None) -> dict:
    if compression not in (None, 'gzip'):
        raise ValueError('Unsupported backup compression')
    source, backup_root = source.resolve(), backup_root.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.parent == backup_root:
        raise ValueError("Backup destination must differ from the source directory")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
        estimated_bytes = src.execute("PRAGMA page_count").fetchone()[0] * src.execute("PRAGMA page_size").fetchone()[0]
    existing_parent = backup_root
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    # Snapshot and restored copy coexist during the restore drill; retain 1 GiB.
    required_bytes = estimated_bytes * (3 if compression else 2) + 1024**3
    free_bytes = shutil.disk_usage(existing_parent).free
    if free_bytes < required_bytes:
        raise OSError(f"Insufficient backup space: free={free_bytes}, required={required_bytes}")
    backup_root.mkdir(parents=True, exist_ok=True)
    destination = backup_root / f"{source.stem}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex}.db"
    if compression: destination = destination.with_suffix('.db.gz')
    with tempfile.TemporaryDirectory(prefix=".backup-verify-", dir=backup_root) as temp:
        snapshot = Path(temp) / "snapshot.db"
        restored = Path(temp) / "restored.db"
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src, closing(sqlite3.connect(snapshot)) as dst:
            src.backup(dst)
            counts = _check(dst)
        # Restore the actual published representation, never merely test the
        # uncompressed source when publishing compressed bytes.
        archive = snapshot
        if compression:
            archive = Path(temp) / 'snapshot.db.gz'
            with snapshot.open('rb') as reader, archive.open('wb') as output:
                with gzip.GzipFile(filename='', mode='wb', fileobj=output, compresslevel=1, mtime=0) as writer:
                    shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
            with gzip.open(archive, 'rb') as reader, restored.open('wb') as writer:
                shutil.copyfileobj(reader, writer, 8 * 1024 * 1024)
        else:
            shutil.copyfile(snapshot, restored)
        with snapshot.open('rb') as reader:
            restored_sha = hashlib.file_digest(reader, 'sha256').hexdigest()
        with restored.open('rb') as reader:
            if hashlib.file_digest(reader, 'sha256').hexdigest() != restored_sha:
                raise ValueError('Restore physical SHA differs from snapshot')
        with closing(sqlite3.connect(restored.as_uri() + "?mode=ro", uri=True)) as recovery:
            if _check(recovery) != counts:
                raise ValueError("Restore table counts differ from backup")
        with archive.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        payload = {
            "status": "success", "source": str(source), "backup": str(destination),
            "created_at": datetime.now(timezone.utc).isoformat(), "bytes": archive.stat().st_size,
            "compression": compression, "restored_bytes": snapshot.stat().st_size,
            "restored_sha256": restored_sha,
            "sha256": digest, "table_rows": counts, "integrity_check": "ok", "restore_verified": True,
            "scope": "single_sqlite_database", "consistency": "SQLite online backup snapshot; separate databases are not one atomic snapshot",
            "space_preflight": {"estimated_database_bytes": estimated_bytes, "required_bytes": required_bytes, "free_bytes": free_bytes},
        }
        manifest = destination.with_suffix(".manifest.json")
        pending_manifest = Path(temp) / "manifest.json"
        pending_manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        archive.replace(destination)
        pending_manifest.replace(manifest)
    return {**payload, "manifest": str(manifest)}
