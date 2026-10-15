"""Bounded immutable research results; no run directories or business assets.

All readers, pins, leases and garbage collection use the same SQLite writer
transaction. GC records a durable tombstone before unlinking an unreferenced
object; interrupted deletions are retried at the next GC. Raw data, native
sources, model weights and account ledgers are deliberately outside this store.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time

from my_strategy.core.paths import DATA_ROOT


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CacheCapacityError(RuntimeError):
    pass


def _enable_wal(connection, *, timeout=30.):
    """Serialize SQLite's first journal-mode transition through bounded retries.

    PRAGMA journal_mode may return SQLITE_BUSY immediately even with a busy
    timeout when concurrent openers race from DELETE to WAL. Existing WAL needs
    no mode transition; other I/O/permission/corruption errors must still fail.
    """
    deadline=time.monotonic()+timeout
    while True:
        try:
            mode=connection.execute("PRAGMA journal_mode").fetchone()[0]
            if str(mode).lower()!='wal':
                mode=connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower()!='wal':raise RuntimeError('Research store requires WAL journal mode')
            return
        except sqlite3.OperationalError as exc:
            code=getattr(exc,'sqlite_errorcode',None)
            locked=(code is not None and (code & 0xff) in {sqlite3.SQLITE_BUSY,sqlite3.SQLITE_LOCKED})
            if code is None:locked=str(exc).lower() in {'database is locked','database table is locked'}
            remaining=deadline-time.monotonic()
            if not locked or remaining<=0:raise
            time.sleep(min(.01,remaining))


class ResearchStore:
    def __init__(self, root=None, *, ttl_seconds=7 * 86400, quota_bytes=8 * 1024**3, clock=time.time):
        self.root = Path(root or DATA_ROOT / "processed" / "research").expanduser().resolve()
        if ttl_seconds <= 0 or quota_bytes <= 0:
            raise ValueError("cache TTL and capacity must be positive")
        self.ttl_seconds, self.quota_bytes, self.clock = float(ttl_seconds), int(quota_bytes), clock
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / "objects"
        self.objects.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.root / "research.db", timeout=30, check_same_thread=False, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA busy_timeout=30000")
            _enable_wal(self.connection)
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            with self._tx():
                self.connection.execute("CREATE TABLE IF NOT EXISTS objects(sha256 TEXT PRIMARY KEY,size_bytes INTEGER NOT NULL,state TEXT NOT NULL CHECK(state IN ('active','deleting')))")
                self.connection.execute("CREATE TABLE IF NOT EXISTS results(cache_key TEXT PRIMARY KEY,namespace TEXT NOT NULL,identity_json TEXT NOT NULL,sha256 TEXT NOT NULL REFERENCES objects(sha256),created_at REAL NOT NULL,accessed_at REAL NOT NULL,expires_at REAL NOT NULL)")
                # Existing stores acquire the lookup index without rewriting
                # objects, expiry, pins or leases. GC checks every shared SHA.
                self.connection.execute("CREATE INDEX IF NOT EXISTS research_results_sha256 ON results(sha256)")
                self.connection.execute("CREATE TABLE IF NOT EXISTS pins(cache_key TEXT NOT NULL REFERENCES results(cache_key) ON DELETE CASCADE,owner TEXT NOT NULL,created_at REAL NOT NULL,PRIMARY KEY(cache_key,owner))")
                self.connection.execute("CREATE TABLE IF NOT EXISTS leases(cache_key TEXT NOT NULL REFERENCES results(cache_key) ON DELETE CASCADE,owner TEXT NOT NULL,expires_at REAL NOT NULL,PRIMARY KEY(cache_key,owner))")
                self.connection.execute("CREATE TABLE IF NOT EXISTS gc_journal(id INTEGER PRIMARY KEY,sha256 TEXT NOT NULL,reason TEXT NOT NULL,state TEXT NOT NULL,created_at REAL NOT NULL,completed_at REAL,error TEXT)")
                self.connection.execute("CREATE TABLE IF NOT EXISTS settings(name TEXT PRIMARY KEY,value_json TEXT NOT NULL,updated_at REAL NOT NULL)")
                self.connection.execute("CREATE TABLE IF NOT EXISTS result_retirements(cache_key TEXT NOT NULL,namespace TEXT NOT NULL,sha256 TEXT NOT NULL,reason TEXT NOT NULL,retired_at REAL NOT NULL)")
        except BaseException:
            self.connection.close()
            raise

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def close(self): self.connection.close()

    @contextmanager
    def _tx(self):
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    @staticmethod
    def cache_key(identity, namespace="default"):
        if not isinstance(identity, dict) or not identity or not isinstance(namespace, str) or not namespace:
            raise ValueError("nonempty complete identity and namespace required")
        return hashlib.sha256(canonical({"namespace": namespace, "identity": identity}).encode()).hexdigest()

    def _path(self, sha):
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid content identity")
        path = self.objects / sha[:2] / (sha + ".json.gz")
        if path.is_symlink() or not path.resolve().is_relative_to(self.objects.resolve()):
            raise ValueError("research object path escaped store")
        return path

    def _live_row(self, key, now):
        return self.connection.execute("""SELECT r.*,o.size_bytes FROM results r JOIN objects o USING(sha256)
            WHERE cache_key=? AND o.state='active' AND (r.expires_at>? OR
            EXISTS(SELECT 1 FROM pins p WHERE p.cache_key=r.cache_key) OR
            EXISTS(SELECT 1 FROM leases l WHERE l.cache_key=r.cache_key AND l.expires_at>?))""", (key, now, now)).fetchone()

    def get(self, identity, namespace="default"):
        key = self.cache_key(identity, namespace)
        with self._tx():
            row = self._live_row(key, self.clock())
            if row is None: return None
            if row["identity_json"] != canonical(identity): raise ValueError("cache identity drift")
            content = self._path(row["sha256"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != row["sha256"] or len(content) != row["size_bytes"]:
                raise ValueError("research object SHA/size mismatch")
            result = json.loads(gzip.decompress(content))
            self.connection.execute("UPDATE results SET accessed_at=? WHERE cache_key=?", (self.clock(), key))
            return result

    def get_by_key(self, cache_key):
        with self._tx():
            row = self._live_row(cache_key, self.clock())
            if row is None: return None
            if self.cache_key(json.loads(row['identity_json']), row['namespace']) != cache_key:
                raise ValueError('cache identity drift')
            content = self._path(row['sha256']).read_bytes()
            if hashlib.sha256(content).hexdigest() != row['sha256'] or len(content) != row['size_bytes']:
                raise ValueError('research object SHA/size mismatch')
            result = json.loads(gzip.decompress(content))
            self.connection.execute('UPDATE results SET accessed_at=? WHERE cache_key=?', (self.clock(), cache_key))
            return result

    def get_setting(self, name, default=None):
        with self._tx():
            row = self.connection.execute('SELECT value_json FROM settings WHERE name=?', (name,)).fetchone()
            return json.loads(row[0]) if row else default

    def set_setting(self, name, value):
        text = canonical(value)
        if not isinstance(name, str) or not name or len(name) > 200 or len(text.encode()) > 65536:
            raise ValueError('setting requires a short name and at most 64 KiB JSON')
        with self._tx():
            self.connection.execute('INSERT INTO settings VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at', (name, text, self.clock()))

    def _cache_bytes(self):
        return self.connection.execute('''SELECT COALESCE(SUM(o.size_bytes),0) FROM objects o WHERE NOT EXISTS(
            SELECT 1 FROM results r JOIN pins p USING(cache_key) WHERE r.sha256=o.sha256)''').fetchone()[0]

    def put(self, identity, result, namespace="default", *, ttl_seconds=None, pin=None):
        key, identity_json = self.cache_key(identity, namespace), canonical(identity)
        payload = gzip.compress(canonical(result).encode(), mtime=0)
        sha, size, now = hashlib.sha256(payload).hexdigest(), len(payload), self.clock()
        ttl = self.ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0: raise ValueError("cache TTL must be positive")
        if size > self.quota_bytes and not pin: raise CacheCapacityError("result exceeds cache capacity")
        with self._tx():
            existing = self.connection.execute("SELECT identity_json,sha256 FROM results WHERE cache_key=?", (key,)).fetchone()
            if existing and (existing["identity_json"] != identity_json or existing["sha256"] != sha):
                raise ValueError("same complete identity produced different immutable result")
            exists = self.connection.execute("SELECT 1 FROM objects WHERE sha256=? AND state='active'", (sha,)).fetchone()
        self.gc(dry_run=False, required_bytes=0 if exists or pin else size)
        with self._tx():
            old = self.connection.execute("SELECT * FROM results WHERE cache_key=?", (key,)).fetchone()
            if old and (old["identity_json"] != identity_json or old["sha256"] != sha):
                raise ValueError("same complete identity produced different immutable result")
            obj = self.connection.execute("SELECT * FROM objects WHERE sha256=?", (sha,)).fetchone()
            if obj and obj["state"] != "active": raise CacheCapacityError("object deletion in progress; retry after GC")
            additional = 0 if obj or pin else size
            total = self._cache_bytes()
            if not pin and total + additional > self.quota_bytes:
                # GC excludes a result being refreshed, and makes its journal durable
                # before any physical unlink. Retrying put avoids hidden overcommit.
                raise CacheCapacityError("cache capacity reached; run gc with required_bytes, or unpin unused results")
            path = self._path(sha)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != payload: raise ValueError("content-addressed object corruption")
            else:
                fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
                try:
                    with os.fdopen(fd, "wb") as output:
                        output.write(payload); output.flush(); os.fsync(output.fileno())
                    os.replace(temporary, path)
                finally:
                    if os.path.exists(temporary): os.unlink(temporary)
            path.chmod(path.stat().st_mode & ~0o222)
            self.connection.execute("INSERT OR IGNORE INTO objects VALUES(?,?,'active')", (sha, size))
            self.connection.execute("INSERT INTO results VALUES(?,?,?,?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET accessed_at=excluded.accessed_at,expires_at=excluded.expires_at", (key, namespace, identity_json, sha, now, now, now + ttl))
            if pin:
                self.connection.execute("INSERT OR IGNORE INTO pins VALUES(?,?,?)", (key, str(pin), now))
        return {"cache_key": key, "sha256": sha, "size_bytes": size, "expires_at": now + ttl}

    def pin(self, cache_key, reason):
        if not reason: raise ValueError("pin owner/reason required")
        with self._tx():
            if self._live_row(cache_key, self.clock()) is None: raise KeyError("result missing or expired")
            self.connection.execute("INSERT OR IGNORE INTO pins VALUES(?,?,?)", (cache_key, str(reason), self.clock()))

    def unpin(self, cache_key, reason):
        if not reason: raise ValueError("specific pin owner/reason required")
        with self._tx():
            return self.connection.execute("DELETE FROM pins WHERE cache_key=? AND owner=?", (cache_key, str(reason))).rowcount

    def is_pinned(self, cache_key):
        with self._tx():
            return self.connection.execute("SELECT 1 FROM pins WHERE cache_key=? LIMIT 1", (cache_key,)).fetchone() is not None

    def acquire_lease(self, cache_key, owner, ttl_seconds=300):
        if not owner or ttl_seconds <= 0: raise ValueError("lease owner and positive TTL required")
        with self._tx():
            if self._live_row(cache_key, self.clock()) is None: raise KeyError("result missing or expired")
            self.connection.execute("INSERT INTO leases VALUES(?,?,?) ON CONFLICT(cache_key,owner) DO UPDATE SET expires_at=excluded.expires_at", (cache_key, str(owner), self.clock() + ttl_seconds))

    def release_lease(self, cache_key, owner):
        with self._tx():
            return self.connection.execute("DELETE FROM leases WHERE cache_key=? AND owner=?", (cache_key, str(owner))).rowcount

    def list(self, namespace=None, limit=100):
        if not 1 <= int(limit) <= 1000: raise ValueError("limit must be 1..1000")
        with self._tx():
            rows = self.connection.execute("""SELECT r.cache_key,r.namespace,r.identity_json,r.sha256,r.created_at,r.accessed_at,r.expires_at,o.size_bytes,
                (SELECT count(*) FROM pins p WHERE p.cache_key=r.cache_key) AS pin_count
                FROM results r JOIN objects o USING(sha256) WHERE o.state='active' AND (? IS NULL OR namespace=?) ORDER BY accessed_at DESC LIMIT ?""", (namespace, namespace, int(limit))).fetchall()
            return [{**{k: r[k] for k in r.keys() if k != 'identity_json'}, 'identity': json.loads(r['identity_json'])} for r in rows]

    def limit_unpinned(self, namespace, keep=2, dry_run=True):
        """Retain newest unreferenced candidates; preserve all owners and leases.

        Small settings may contain current/rollback/account pointers, so those
        references are protected even if a caller has not added its pin yet.
        Shared content survives whenever another identity still references it.
        """
        if not isinstance(namespace, str) or not namespace or type(keep) is not int or keep < 0:
            raise ValueError("namespace and nonnegative integer keep required")
        now = self.clock()
        with self._tx():
            setting_keys = set()

            def references(value):
                if isinstance(value, dict):
                    for item in value.values(): references(item)
                elif isinstance(value, list):
                    for item in value: references(item)
                elif isinstance(value, str) and len(value) == 64:
                    setting_keys.add(value)

            for row in self.connection.execute('SELECT value_json FROM settings'):
                references(json.loads(row[0]))
            rows = self.connection.execute("""SELECT r.cache_key,r.sha256 FROM results r
                WHERE namespace=? AND NOT EXISTS(SELECT 1 FROM pins p WHERE p.cache_key=r.cache_key)
                AND NOT EXISTS(SELECT 1 FROM leases l WHERE l.cache_key=r.cache_key AND l.expires_at>?)
                ORDER BY created_at DESC,rowid DESC""", (namespace, now)).fetchall()
            eligible = [row for row in rows if row['cache_key'] not in setting_keys]
            retired = [dict(row) for row in eligible[keep:]]
            if not dry_run:
                for row in retired:
                    self.connection.execute('INSERT INTO result_retirements VALUES(?,?,?,?,?)',
                        (row['cache_key'], namespace, row['sha256'], 'candidate_limit', now))
                    self.connection.execute('DELETE FROM results WHERE cache_key=?', (row['cache_key'],))
        report = {'namespace': namespace, 'keep_unreferenced': keep, 'dry_run': bool(dry_run),
                  'retired': retired, 'retired_count': len(retired)}
        if not dry_run:
            report['gc'] = self.gc(dry_run=False)
        return report

    def gc(self, dry_run=True, *, required_bytes=0):
        now = self.clock()
        if required_bytes < 0: raise ValueError("required bytes cannot be negative")
        with self._tx():
            # A crash after file publication but before SQLite commit can leave
            # an unindexed CAS file. The writer lock excludes in-progress puts;
            # only final SHA-named objects inside this owned cache are adopted.
            orphan_rows = []
            for path in self.objects.glob('*/*.json.gz'):
                sha = path.name.removesuffix('.json.gz')
                if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha): continue
                if path != self._path(sha): continue
                if self.connection.execute('SELECT 1 FROM objects WHERE sha256=?', (sha,)).fetchone(): continue
                orphan_rows.append({'sha256':sha,'size_bytes':path.stat().st_size,'expires_at':None,'accessed_at':0})
                if not dry_run: self.connection.execute("INSERT INTO objects VALUES(?,?,'active')", (sha,path.stat().st_size))
            orphan_ids = {row['sha256'] for row in orphan_rows}
            total = self._cache_bytes()
            all_bytes = self.connection.execute('SELECT COALESCE(SUM(size_bytes),0) FROM objects').fetchone()[0]
            rows = self.connection.execute("""SELECT o.sha256,o.size_bytes,MAX(r.expires_at) AS expires_at,MAX(r.accessed_at) AS accessed_at
                FROM objects o LEFT JOIN results r USING(sha256) WHERE o.state='active' AND NOT EXISTS(
                  SELECT 1 FROM results x WHERE x.sha256=o.sha256 AND (
                    EXISTS(SELECT 1 FROM pins p WHERE p.cache_key=x.cache_key) OR
                    EXISTS(SELECT 1 FROM leases l WHERE l.cache_key=x.cache_key AND l.expires_at>?)))
                GROUP BY o.sha256 ORDER BY COALESCE(MAX(r.expires_at),0)<=? DESC,COALESCE(MAX(r.accessed_at),0),o.sha256""", (now, now)).fetchall()
            if dry_run:
                rows = orphan_rows + rows
                orphan_bytes = sum(row['size_bytes'] for row in orphan_rows)
                total += orphan_bytes; all_bytes += orphan_bytes
            candidates, remaining = [], total
            for row in rows:
                expired = row["expires_at"] is None or row["expires_at"] <= now
                if not expired and remaining + required_bytes <= self.quota_bytes: continue
                candidates.append({"sha256": row["sha256"], "size_bytes": row["size_bytes"], "reason": 'uncommitted_orphan' if row['sha256'] in orphan_ids else "expired" if expired else "quota"})
                remaining -= row["size_bytes"]
            if not dry_run:
                for row in candidates:
                    self.connection.execute("INSERT INTO gc_journal(sha256,reason,state,created_at) VALUES(?,?,'pending',?)", (row["sha256"], row["reason"], now))
                    self.connection.execute("DELETE FROM results WHERE sha256=?", (row["sha256"],))
                    self.connection.execute("UPDATE objects SET state='deleting' WHERE sha256=?", (row["sha256"],))
        report = {"dry_run": bool(dry_run), "candidates": candidates, "candidate_bytes": sum(r["size_bytes"] for r in candidates), "total_bytes": all_bytes, 'cache_bytes': total, 'pinned_bytes': all_bytes-total, "capacity_available": remaining + required_bytes <= self.quota_bytes, "deleted_bytes": 0}
        if not dry_run:
            # Recheck under the same writer protocol: new readers cannot lease a
            # deleting object, and a crash leaves an inspectable pending journal.
            with self._tx():
                pending = self.connection.execute("SELECT id,sha256 FROM gc_journal WHERE state='pending'").fetchall()
                for row in pending:
                    if self.connection.execute("SELECT 1 FROM results WHERE sha256=?", (row["sha256"],)).fetchone():
                        raise RuntimeError("GC object acquired a new reference")
                    obj = self.connection.execute("SELECT size_bytes FROM objects WHERE sha256=? AND state='deleting'", (row["sha256"],)).fetchone()
                    try:
                        path = self._path(row['sha256'])
                        if path.exists(): path.chmod(path.stat().st_mode | 0o200)
                        path.unlink(missing_ok=True)
                    except OSError as exc:
                        self.connection.execute("UPDATE gc_journal SET error=? WHERE id=?", (str(exc), row["id"]))
                        continue
                    self.connection.execute("DELETE FROM objects WHERE sha256=?", (row["sha256"],))
                    self.connection.execute("UPDATE gc_journal SET state='deleted',completed_at=?,error=NULL WHERE id=?", (self.clock(), row["id"]))
                    report["deleted_bytes"] += obj[0] if obj else 0
        return report
