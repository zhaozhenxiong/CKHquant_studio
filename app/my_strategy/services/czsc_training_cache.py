"""Disposable, verified training intermediates and completed-stock checkpoints.

The caller supplies a key binding every generating input, model, configuration,
calendar and code identity. This store never decides that two computations are
equivalent. JSON objects live in atomic content-addressed gzip files; SQLite is
only the index. Checkpoints reference existing artifacts without copying or
deleting them. A caller publishes a checkpoint only after its stage completion
manifest exists, and must include that manifest and every result in ``files``.

No pickle is read or written. Every hit rehashes all declared dependencies and
checkpoint files. Missing, partial or corrupt entries are cache misses.
"""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence
import zlib

from my_strategy.core.paths import PROCESSED_DATA_ROOT


CACHE_VERSION = 1
_FORMAT = "czsc_training_cache"
_MAX_JSON_BYTES = 256 * 1024 * 1024
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_MISS_ERRORS = (OSError, ValueError, TypeError, KeyError, sqlite3.DatabaseError,
                EOFError, zlib.error, UnicodeError)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _strict_json(data: bytes) -> Any:
    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, parse_constant=invalid, object_pairs_hook=pairs)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    """Hash an ordinary stable file, including same-size/same-mtime edits."""
    with path.open("rb") as source:
        before = os.fstat(source.fileno())
        result = hashlib.sha256()
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
        after = os.fstat(source.fileno())
    observed = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) or
           getattr(after, field) != getattr(observed, field) for field in fields):
        raise ValueError("cache dependency changed while hashing")
    return result.hexdigest()


def _name(value: str, what: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 1024 or "\x00" in value:
        raise ValueError(f"invalid cache {what}")
    return value


def _relative_file(root: Path, value: str | Path) -> tuple[str, Path]:
    """Only canonical relative paths inside the caller's explicit root."""
    text = str(value)
    relative = PurePosixPath(text)
    if (not text or "\\" in text or relative.is_absolute() or
            any(part in {".", ".."} for part in text.split("/")) or
            ":" in text or str(relative) != text):
        raise ValueError("checkpoint file must be a canonical relative path")
    path = root / text
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("checkpoint file escapes root or is not a file")
    # Reject symlinks even when their current target is in root. A later link
    # change must never permit the cache to follow an undeclared external path.
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("checkpoint paths must not contain symlinks")
    return text, resolved


def _dependencies(deps: Mapping[str | Path, str] | None) -> dict[str, str]:
    if deps is None:
        return {}
    if not isinstance(deps, Mapping):
        raise ValueError("dependencies must map absolute paths to SHA-256")
    result = {}
    for raw_path, digest in deps.items():
        path = Path(raw_path)
        if not path.is_absolute() or not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ValueError("dependency requires an absolute path and SHA-256")
        # Preserve the exact declared absolute path so a symlink retarget cannot
        # silently change the dependency identity. Its contents are rehashed.
        name = str(path)
        if name in result:
            raise ValueError("duplicate dependency path")
        if not path.is_file() or _sha_file(path) != digest:
            raise ValueError("cache dependency content mismatch")
        result[name] = digest
    return dict(sorted(result.items()))


class TrainingCache:
    """Thread/process-safe cache with short-lived connections and transactions.

    ``deps`` maps explicitly authorized absolute input paths to expected SHA-256.
    ``files`` is a nonempty list of root-relative output paths, or a mapping from
    root-relative paths to expected SHA-256. A changed root is a different scope.
    get methods return newly decoded objects, never shared mutable instances.
    Cache index initialization errors remain explicit; corrupt entries are misses.
    """

    def __init__(self, db_path: str | Path | None = None):
        default = PROCESSED_DATA_ROOT / "stock_training" / "training_cache.db"
        self.db_path = Path(db_path if db_path is not None else default).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.objects_root = self.db_path.with_name(self.db_path.stem + "_objects")
        if self.objects_root.is_symlink():
            raise ValueError("cache objects root must not be a symlink")
        self.objects_root.mkdir(exist_ok=True)
        self._stats = {"hits": 0, "misses": 0}
        self._stats_lock = threading.Lock()
        self._initialize()

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA synchronous=FULL")
            connection.row_factory = sqlite3.Row
            yield connection
        finally:
            connection.close()

    def _initialize(self):
        # Simultaneous first-use workers can contend on the initial WAL switch.
        for attempt in range(5):
            try:
                with self._connection() as connection:
                    mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                    if mode != "wal":
                        connection.execute("PRAGMA journal_mode=WAL")
                    with connection:
                        connection.execute("""CREATE TABLE IF NOT EXISTS training_cache_entries (
                            kind TEXT NOT NULL, namespace TEXT NOT NULL, cache_key TEXT NOT NULL,
                            scope TEXT NOT NULL, version INTEGER NOT NULL,
                            object_path TEXT NOT NULL, object_sha256 TEXT NOT NULL,
                            json_size INTEGER NOT NULL, status TEXT NOT NULL,
                            updated_at REAL NOT NULL,
                            PRIMARY KEY(kind, namespace, cache_key, scope)
                        )""")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 4:
                    raise
                time.sleep(0.02 * (2 ** attempt))

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def _count(self, hit: bool):
        with self._stats_lock:
            self._stats["hits" if hit else "misses"] += 1

    def close(self):
        """Compatibility helper; operations never keep a connection open."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _object_path(self, relative: str) -> Path:
        # DB corruption cannot turn an object reference into an arbitrary read.
        if not re.fullmatch(r"[a-f0-9]{2}/[a-f0-9]{64}\.json\.gz", relative):
            raise ValueError("invalid cache object reference")
        if self.objects_root.is_symlink():
            raise ValueError("cache objects root changed")
        path = self.objects_root / relative
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("cache object must not be a symlink")
        if not path.resolve().is_relative_to(self.objects_root):
            raise ValueError("cache object escapes root")
        return path

    def _publish(self, kind: str, namespace: str, key: str, scope: str, body: dict):
        envelope = {"format": _FORMAT, "version": CACHE_VERSION, "kind": kind,
                    "namespace": namespace, "key": key, "scope": scope, **body}
        data = _json_bytes(envelope)
        if len(data) > _MAX_JSON_BYTES:
            raise ValueError("cache JSON exceeds maximum size")
        digest = _sha_bytes(data)
        relative = f"{digest[:2]}/{digest}.json.gz"
        target = self._object_path(relative)
        target.parent.mkdir(exist_ok=True)
        target = self._object_path(relative)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=".pending-",
                                             suffix=".gz", delete=False) as output:
                temporary = Path(output.name)
                output.write(gzip.compress(data, compresslevel=3, mtime=0))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            temporary = None
            # File publication precedes the short index transaction. An abrupt
            # interruption can leave an unindexed cache blob, never a partial hit.
            with self._connection() as connection:
                with connection:
                    connection.execute("""INSERT INTO training_cache_entries
                        (kind,namespace,cache_key,scope,version,object_path,object_sha256,json_size,status,updated_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(kind,namespace,cache_key,scope)
                        DO UPDATE SET version=excluded.version, object_path=excluded.object_path,
                        object_sha256=excluded.object_sha256, json_size=excluded.json_size,
                        status=excluded.status, updated_at=excluded.updated_at""",
                        (kind, namespace, key, scope, CACHE_VERSION, relative, digest, len(data),
                         "complete", time.time()))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _read(self, kind: str, namespace: str, key: str, scope: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute("""SELECT * FROM training_cache_entries
                WHERE kind=? AND namespace=? AND cache_key=? AND scope=?""",
                (kind, namespace, key, scope)).fetchone()
        if row is None or row["version"] != CACHE_VERSION or row["status"] != "complete":
            return None
        size = row["json_size"]
        digest = row["object_sha256"]
        if (type(size) is not int or not 0 < size <= _MAX_JSON_BYTES or
                not isinstance(digest, str) or not _SHA.fullmatch(digest)):
            return None
        if row["object_path"] != f"{digest[:2]}/{digest}.json.gz":
            return None
        with gzip.open(self._object_path(row["object_path"]), "rb") as source:
            data = source.read(size + 1)
        if len(data) != size or _sha_bytes(data) != digest:
            return None
        value = _strict_json(data)
        identity = {"format": _FORMAT, "version": CACHE_VERSION, "kind": kind,
                    "namespace": namespace, "key": key, "scope": scope}
        if not isinstance(value, dict) or any(value.get(name) != expected for name, expected in identity.items()):
            return None
        return value

    def put_json(self, namespace: str, key: str, payload: Any,
                 deps: Mapping[str | Path, str] | None = None):
        namespace, key = _name(namespace, "namespace"), _name(key, "key")
        dependencies = _dependencies(deps)
        self._publish("json", namespace, key, "", {"deps": dependencies, "payload": payload})

    def get_json(self, namespace: str, key: str,
                 deps: Mapping[str | Path, str] | None = None) -> Any | None:
        hit = False
        try:
            namespace, key = _name(namespace, "namespace"), _name(key, "key")
            value = self._read("json", namespace, key, "")
            if value is None:
                return None
            dependencies = _dependencies(deps)
            if value.get("deps") != dependencies or "payload" not in value:
                return None
            hit = True
            return value["payload"]
        except _MISS_ERRORS:
            return None
        finally:
            self._count(hit)

    def put_checkpoint(self, key: str, root: str | Path, summary: Any,
                       files: Sequence[str | Path] | Mapping[str | Path, str]):
        key = _name(key, "key")
        root_path = Path(root).expanduser().resolve(strict=True)
        if not root_path.is_dir() or not files or isinstance(files, (str, bytes, Path)):
            raise ValueError("checkpoint requires a root directory and all completed files")
        expected = files if isinstance(files, Mapping) else None
        manifest = {}
        for filename in files:
            relative, path = _relative_file(root_path, filename)
            if relative in manifest:
                raise ValueError("duplicate checkpoint file")
            digest = _sha_file(path)
            if expected is not None and expected[filename] != digest:
                raise ValueError("checkpoint file content mismatch")
            manifest[relative] = digest
        self._publish("checkpoint", "completed-stock", key, str(root_path),
                      {"files": manifest, "summary": summary})

    def get_checkpoint(self, key: str, root: str | Path) -> Any | None:
        hit = False
        try:
            key = _name(key, "key")
            root_path = Path(root).expanduser().resolve(strict=True)
            value = self._read("checkpoint", "completed-stock", key, str(root_path))
            if value is None or "summary" not in value:
                return None
            manifest = value.get("files")
            if not isinstance(manifest, dict) or not manifest:
                return None
            for filename, expected in manifest.items():
                _, path = _relative_file(root_path, filename)
                if not isinstance(expected, str) or not _SHA.fullmatch(expected) or _sha_file(path) != expected:
                    return None
            hit = True
            return value["summary"]
        except _MISS_ERRORS:
            return None
        finally:
            self._count(hit)
