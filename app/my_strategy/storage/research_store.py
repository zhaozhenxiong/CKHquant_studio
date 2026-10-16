"""Bounded immutable research results; no run directories or business assets.

Pins, leases, publication and GC share a SQLite writer protocol. Readers open
immutable objects without a writer transaction; a racing deletion yields a miss
or an already-open file descriptor whose complete bytes remain readable. GC
records a durable tombstone before unlinking; interrupted deletions are retried. Raw data, native
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
from my_strategy.storage.research_store_schema import install_accounting


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
    def __init__(self, root=None, *, ttl_seconds=7 * 86400, quota_bytes=8 * 1024**3, clock=time.time,
                 orphan_scan_interval_seconds=60.):
        self.root = Path(root or DATA_ROOT / "processed" / "research").expanduser().resolve()
        if ttl_seconds <= 0 or quota_bytes <= 0:
            raise ValueError("cache TTL and capacity must be positive")
        if not isinstance(orphan_scan_interval_seconds, (int, float)) or not 0 < orphan_scan_interval_seconds < float('inf'):
            raise ValueError("orphan scan interval must be positive and finite")
        self.ttl_seconds, self.quota_bytes, self.clock = float(ttl_seconds), int(quota_bytes), clock
        self.orphan_scan_interval_seconds = float(orphan_scan_interval_seconds)
        # Per-instance state only: every new process/store checks crash leftovers.
        # Explicit GC and dry runs always inspect the complete owned directory.
        self._last_orphan_scan_at = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / "objects"
        self.objects.mkdir(exist_ok=True)
        if self.objects.is_symlink():
            raise ValueError('research object path escaped store')
        self._owned_objects_root = self.objects.resolve()
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
                install_accounting(self.connection)
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
        if self.objects.is_symlink() or self.objects.resolve() != self._owned_objects_root:
            raise ValueError("research object path escaped store")
        path = self.objects / sha[:2] / (sha + ".json.gz")
        if path.is_symlink() or not path.resolve().is_relative_to(self._owned_objects_root):
            raise ValueError("research object path escaped store")
        return path

    def _live_row(self, key, now):
        return self.connection.execute("""SELECT r.*,o.size_bytes FROM results r JOIN objects o USING(sha256)
            WHERE cache_key=? AND o.state='active' AND (r.expires_at>? OR
            EXISTS(SELECT 1 FROM pins p WHERE p.cache_key=r.cache_key) OR
            EXISTS(SELECT 1 FROM leases l WHERE l.cache_key=r.cache_key AND l.expires_at>?))""", (key, now, now)).fetchone()

    def get(self, identity, namespace="default"):
        return self._read_result(self.cache_key(identity, namespace), identity_json=canonical(identity))

    def get_by_key(self, cache_key, *, touch=True):
        """Read verified immutable bytes; ``touch=False`` is strictly read-only.

        Open descriptors survive POSIX unlink. Platforms denying unlink of open
        files leave the GC journal pending and retry after the reader closes.
        If GC won before open, return a miss; an indexed missing file still
        raises instead of hiding corruption. No finite read lease can expire
        midway through a slow decode because the descriptor owns the bytes.
        """
        return self._read_result(cache_key, touch=touch)

    def _read_result(self, key, *, identity_json=None, touch=True):
        with self._lock:
            now = self.clock()
            row = self._live_row(key, now)
            if row is None:
                return None
            if identity_json is not None and row['identity_json'] != identity_json:
                raise ValueError('cache identity drift')
            if self.cache_key(json.loads(row['identity_json']), row['namespace']) != key:
                raise ValueError('cache identity drift')
            try:
                stream = self._path(row['sha256']).open('rb')
            except FileNotFoundError:
                current = self._live_row(key, self.clock())
                if current is None or current['sha256'] != row['sha256']:
                    return None
                raise
        with stream:
            content = stream.read()
        if hashlib.sha256(content).hexdigest() != row['sha256'] or len(content) != row['size_bytes']:
            raise ValueError('research object SHA/size mismatch')
        result = json.loads(gzip.decompress(content))
        # Access order has 30-second resolution. Hot reads perform no writes;
        # sampled touches are short and cannot resurrect a retired identity.
        if touch and (now - row['accessed_at'] >= 30 or now < row['accessed_at']):
            with self._tx():
                self.connection.execute("""UPDATE results SET accessed_at=?
                    WHERE cache_key=? AND sha256=? AND (accessed_at<=? OR accessed_at>?)""",
                    (now, key, row['sha256'], now - 30, now))
        return result

    def get_setting(self, name, default=None):
        with self._lock:
            row = self.connection.execute('SELECT value_json FROM settings WHERE name=?', (name,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, name, value):
        text = canonical(value)
        if not isinstance(name, str) or not name or len(name) > 200 or len(text.encode()) > 65536:
            raise ValueError('setting requires a short name and at most 64 KiB JSON')
        with self._tx():
            self.connection.execute('INSERT INTO settings VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET value_json=excluded.value_json,updated_at=excluded.updated_at', (name, text, self.clock()))

    def _cache_bytes(self):
        return self.connection.execute('SELECT cache_bytes FROM research_totals WHERE id=1').fetchone()[0]

    def _maintenance_needed(self, required_bytes, now):
        # Indexed probes replace the previous per-put full GROUP BY and SUM.
        with self._lock:
            last = self._last_orphan_scan_at
            if (last is None or now < last or now - last >= self.orphan_scan_interval_seconds
                    or self._cache_bytes() + required_bytes > self.quota_bytes):
                return True
            if self.connection.execute("SELECT 1 FROM gc_journal WHERE state='pending' LIMIT 1").fetchone():
                return True
            return self.connection.execute("""SELECT 1 FROM object_usage
                WHERE state='active' AND pin_count=0 AND MAX(expires_at,lease_until)<=? LIMIT 1""",
                (now,)).fetchone() is not None

    def put(self, identity, result, namespace="default", *, ttl_seconds=None, pin=None, refresh_keys=()):
        return self.put_many([{'identity': identity, 'result': result, 'namespace': namespace,
                               'ttl_seconds': ttl_seconds, 'pin': pin}], refresh_keys=refresh_keys)[0]

    def put_many(self, items, *, lease_owner=None, lease_ttl_seconds=300, refresh_keys=()):
        """Atomically publish a bounded batch, optionally acquiring its leases.

        ``items`` contains identity/result/namespace and optional TTL/pin fields.
        ``refresh_keys`` are live dependencies whose TTL is extended to at least
        the latest receipt expiry in the same transaction as publication.
        Receipts preserve input order. Immutable identity conflicts and capacity
        failures publish no result rows. A crash can only leave SHA-named orphan
        files, handled by normal durable GC. Callers should bound batch memory.
        """
        if lease_owner is not None and (not lease_owner or not 0 < lease_ttl_seconds < float('inf')):
            raise ValueError('lease owner and positive finite TTL required')
        prepared, identities, objects = [], {}, {}
        now = self.clock()
        for item in items:
            identity, namespace = item['identity'], item.get('namespace', 'default')
            key, identity_json = self.cache_key(identity, namespace), canonical(identity)
            payload = gzip.compress(canonical(item['result']).encode(), mtime=0)
            sha, size, pin = hashlib.sha256(payload).hexdigest(), len(payload), item.get('pin')
            requested_ttl = item.get('ttl_seconds')
            ttl = self.ttl_seconds if requested_ttl is None else float(requested_ttl)
            if not 0 < ttl < float('inf'):
                raise ValueError('cache TTL must be positive and finite')
            signature = (identity_json, sha)
            if key in identities and identities[key] != signature:
                raise ValueError('same complete identity produced different immutable result')
            identities[key] = signature
            prepared.append((key, namespace, identity_json, sha, now + ttl, pin))
            previous = objects.get(sha)
            objects[sha] = (payload, size, bool(pin) or bool(previous and previous[2]))
        if not prepared:
            return []
        # All owners of a shared object participate in quota calculation.
        for payload, size, pinned in objects.values():
            if size > self.quota_bytes and not pinned:
                raise CacheCapacityError('result exceeds cache capacity')
        refresh_keys = tuple(dict.fromkeys(refresh_keys))
        protected_shas = set(objects)
        with self._lock:
            self._check_immutable(identities)
            for key in refresh_keys:
                row = self._live_row(key, now)
                if row is None:
                    raise KeyError('result dependency missing or expired')
                protected_shas.add(row['sha256'])
            required = sum(size for sha, (_, size, pinned) in objects.items()
                if not pinned and self.connection.execute('SELECT 1 FROM objects WHERE sha256=?', (sha,)).fetchone() is None)
        if self._maintenance_needed(required, now):
            self.gc(dry_run=False, required_bytes=required, scan_orphans='auto', _protected_shas=protected_shas)
        with self._tx():
            # Concurrent publishers may have consumed space since preflight.
            # Recheck within the same transaction as all counters and leases.
            self._check_immutable(identities)
            projected = self._cache_bytes()
            for sha, (_, size, pinned) in objects.items():
                existing = self.connection.execute('SELECT * FROM object_usage WHERE sha256=?', (sha,)).fetchone()
                if existing and existing['state'] != 'active':
                    raise CacheCapacityError('object deletion in progress; retry after GC')
                if existing is None and not pinned:
                    projected += size
                elif existing and pinned and existing['pin_count'] == 0:
                    projected -= existing['size_bytes']
            if any(not entry[2] for entry in objects.values()) and projected > self.quota_bytes:
                raise CacheCapacityError('cache capacity reached; run gc with required_bytes, or unpin unused results')
            for sha, (payload, size, _) in objects.items():
                path = self._path(sha)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if path.read_bytes() != payload:
                        raise ValueError('content-addressed object corruption')
                else:
                    fd, temporary = tempfile.mkstemp(prefix='.write-', dir=path.parent)
                    try:
                        with os.fdopen(fd, 'wb') as output:
                            output.write(payload); output.flush(); os.fsync(output.fileno())
                        os.replace(temporary, path)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                path.chmod(path.stat().st_mode & ~0o222)
                self.connection.execute("INSERT OR IGNORE INTO objects VALUES(?,?,'active')", (sha, size))
            # TTL/leases start at publication, after compression and batch file
            # I/O; a slow batch must not publish already-expired protections.
            published_at = self.clock()
            prepared = [(key, namespace, identity_json, sha, expires + published_at - now, pin)
                        for key, namespace, identity_json, sha, expires, pin in prepared]
            now = published_at
            self._refresh_expiry_in_transaction(refresh_keys, max(row[4] for row in prepared), now)
            for key, namespace, identity_json, sha, expires, pin in prepared:
                self.connection.execute("""INSERT INTO results VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(cache_key) DO UPDATE SET accessed_at=excluded.accessed_at,expires_at=excluded.expires_at""",
                    (key, namespace, identity_json, sha, now, now, expires))
                if pin:
                    self.connection.execute('INSERT OR IGNORE INTO pins VALUES(?,?,?)', (key, str(pin), now))
                if lease_owner is not None:
                    self.connection.execute("""INSERT INTO leases VALUES(?,?,?) ON CONFLICT(cache_key,owner)
                        DO UPDATE SET expires_at=excluded.expires_at""", (key, str(lease_owner), now + lease_ttl_seconds))
        return [{'cache_key': key, 'sha256': sha, 'size_bytes': objects[sha][1], 'expires_at': expires}
                for key, _, _, sha, expires, _ in prepared]

    def _check_immutable(self, identities):
        for key, (identity_json, sha) in identities.items():
            old = self.connection.execute('SELECT identity_json,sha256 FROM results WHERE cache_key=?', (key,)).fetchone()
            if old and (old['identity_json'] != identity_json or old['sha256'] != sha):
                raise ValueError('same complete identity produced different immutable result')

    def _refresh_expiry_in_transaction(self, cache_keys, expires_at, now):
        for key in cache_keys:
            if self._live_row(key, now) is None:
                raise KeyError('result dependency missing or expired')
            self.connection.execute("""UPDATE results SET expires_at=MAX(expires_at,?),accessed_at=?
                WHERE cache_key=?""", (expires_at, now, key))

    def refresh_expiry(self, cache_keys, ttl_seconds=None):
        """Extend live dependencies atomically; never revive expired results."""
        ttl = self.ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        if not 0 < ttl < float('inf'):
            raise ValueError('cache TTL must be positive and finite')
        keys = tuple(dict.fromkeys(cache_keys))
        with self._tx():
            now = self.clock()
            self._refresh_expiry_in_transaction(keys, now + ttl, now)
        return len(keys)

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
        with self._lock:
            return self.connection.execute('SELECT 1 FROM pins WHERE cache_key=? LIMIT 1', (cache_key,)).fetchone() is not None

    def acquire_lease(self, cache_key, owner, ttl_seconds=300):
        self.acquire_leases([cache_key], owner, ttl_seconds)

    def acquire_leases(self, cache_keys, owner, ttl_seconds=300):
        """Acquire every requested live result in one transaction, or none."""
        if not owner or not 0 < ttl_seconds < float('inf'):
            raise ValueError('lease owner and positive finite TTL required')
        with self._tx():
            now = self.clock()
            for key in dict.fromkeys(cache_keys):
                if self._live_row(key, now) is None:
                    raise KeyError('result missing or expired')
                self.connection.execute("""INSERT INTO leases VALUES(?,?,?) ON CONFLICT(cache_key,owner)
                    DO UPDATE SET expires_at=excluded.expires_at""", (key, str(owner), now + ttl_seconds))

    def release_lease(self, cache_key, owner):
        return self.release_leases([cache_key], owner)

    def release_leases(self, cache_keys, owner):
        with self._tx():
            return sum(self.connection.execute('DELETE FROM leases WHERE cache_key=? AND owner=?',
                (key, str(owner))).rowcount for key in dict.fromkeys(cache_keys))

    def list(self, namespace=None, limit=100):
        if not 1 <= int(limit) <= 1000:
            raise ValueError('limit must be 1..1000')
        where, parameters = ('', []) if namespace is None else ('AND r.namespace=?', [namespace])
        with self._lock:
            rows = self.connection.execute(f"""SELECT r.cache_key,r.namespace,r.identity_json,r.sha256,r.created_at,r.accessed_at,r.expires_at,o.size_bytes,
                (SELECT count(*) FROM pins p WHERE p.cache_key=r.cache_key) AS pin_count
                FROM results r JOIN objects o USING(sha256) WHERE o.state='active' {where}
                ORDER BY accessed_at DESC LIMIT ?""", [*parameters, int(limit)]).fetchall()
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

    def _unindexed_objects(self):
        """Inspect owned shards without resolving/querying every indexed object.

        The caller holds the SQLite writer lock, so no other store can publish a
        file between the index snapshot and the directory walk. Symlink shards
        are never followed, and new objects still receive full path validation.
        Indexed objects are verified by their normal SHA-checked read path.
        """
        known = {row[0] for row in self.connection.execute('SELECT sha256 FROM objects')}
        if self.objects.is_symlink() or self.objects.resolve() != self._owned_objects_root:
            raise ValueError('research object path escaped store')
        with os.scandir(self.objects) as shards:
            for shard in shards:
                if len(shard.name) != 2 or any(c not in '0123456789abcdef' for c in shard.name):
                    continue
                if shard.is_symlink():
                    raise ValueError('research object path escaped store')
                if not shard.is_dir(follow_symlinks=False):
                    continue
                with os.scandir(shard.path) as entries:
                    for entry in entries:
                        if not entry.name.endswith('.json.gz'):
                            continue
                        sha = entry.name.removesuffix('.json.gz')
                        if len(sha) != 64 or sha[:2] != shard.name or any(c not in '0123456789abcdef' for c in sha):
                            continue
                        if entry.is_symlink():
                            raise ValueError('research object path escaped store')
                        if sha in known or not entry.is_file(follow_symlinks=False):
                            continue
                        path = Path(entry.path)
                        if path != self._path(sha):
                            continue
                        yield {'sha256':sha, 'size_bytes':path.stat().st_size,
                               'expires_at':None, 'accessed_at':0}

    def gc(self, dry_run=True, *, required_bytes=0, scan_orphans='full', _protected_shas=()):
        """Collect indexed expiry then LRU pressure; explicit GC walks files.

        Ordinary puts use indexed probes before calling GC. Accounting and
        candidates are derived transactionally; full orphan discovery is the
        only routine here that needs a complete owned-directory/index scan.
        """
        now = self.clock()
        if required_bytes < 0: raise ValueError("required bytes cannot be negative")
        if scan_orphans not in {'full', 'auto'}:
            raise ValueError("scan_orphans must be full or auto")
        with self._tx():
            # A crash after file publication but before SQLite commit can leave
            # an unindexed CAS file. The writer lock excludes in-progress puts;
            # only final SHA-named objects inside this owned cache are adopted.
            total = self._cache_bytes()
            last = self._last_orphan_scan_at
            scan_due = (dry_run or scan_orphans == 'full' or last is None or now < last
                        or now - last >= self.orphan_scan_interval_seconds
                        or total + required_bytes > self.quota_bytes)
            orphan_rows = list(self._unindexed_objects()) if scan_due else []
            if not dry_run:
                for row in orphan_rows:
                    self.connection.execute("INSERT INTO objects VALUES(?,?,'active')", (row['sha256'],row['size_bytes']))
            orphan_ids = {row['sha256'] for row in orphan_rows}
            if orphan_rows and not dry_run:
                total = self._cache_bytes()
            all_bytes = self.connection.execute('SELECT total_bytes FROM research_totals WHERE id=1').fetchone()[0]
            expired_rows = self.connection.execute("""SELECT sha256,size_bytes,expires_at,accessed_at
                FROM object_usage WHERE state='active' AND pin_count=0
                AND MAX(expires_at,lease_until)<=? ORDER BY accessed_at,sha256""", (now,)).fetchall()
            if dry_run:
                expired_rows = orphan_rows + expired_rows
                orphan_bytes = sum(row['size_bytes'] for row in orphan_rows)
                total += orphan_bytes; all_bytes += orphan_bytes
            candidates, remaining, selected = [], total, set(_protected_shas)
            for row in expired_rows:
                if row['sha256'] in selected:
                    continue
                candidates.append({'sha256': row['sha256'], 'size_bytes': row['size_bytes'],
                    'reason': 'uncommitted_orphan' if row['sha256'] in orphan_ids else 'expired'})
                selected.add(row['sha256']); remaining -= row['size_bytes']
            if remaining + required_bytes > self.quota_bytes:
                # The covering order index lets SQLite stop once enough bytes
                # are found; no GROUP BY/sort over the entire object population.
                rows = self.connection.execute("""SELECT sha256,size_bytes FROM object_usage
                    INDEXED BY research_usage_lru WHERE state='active' AND pin_count=0
                    AND lease_until<=? ORDER BY accessed_at,sha256""", (now,))
                for row in rows:
                    if remaining + required_bytes <= self.quota_bytes:
                        break
                    if row['sha256'] in selected:
                        continue
                    candidates.append({'sha256': row['sha256'], 'size_bytes': row['size_bytes'], 'reason': 'quota'})
                    selected.add(row['sha256']); remaining -= row['size_bytes']
                rows.close()
            if not dry_run:
                for row in candidates:
                    self.connection.execute("INSERT INTO gc_journal(sha256,reason,state,created_at) VALUES(?,?,'pending',?)", (row["sha256"], row["reason"], now))
                    self.connection.execute("DELETE FROM results WHERE sha256=?", (row["sha256"],))
                    self.connection.execute("UPDATE objects SET state='deleting' WHERE sha256=?", (row["sha256"],))
        if scan_due and not dry_run:
            with self._lock:
                self._last_orphan_scan_at = now
        report = {"dry_run": bool(dry_run), "candidates": candidates, "candidate_bytes": sum(r["size_bytes"] for r in candidates), "total_bytes": all_bytes, 'cache_bytes': total, 'pinned_bytes': all_bytes-total, "capacity_available": remaining + required_bytes <= self.quota_bytes, "deleted_bytes": 0, 'orphan_scan_performed':bool(scan_due)}
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
