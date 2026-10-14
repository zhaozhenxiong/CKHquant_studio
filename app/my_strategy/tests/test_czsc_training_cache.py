"""Persistent cache proof, corruption misses, publication and concurrency."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3

import pytest

from my_strategy.services.czsc_training_cache import TrainingCache


def sha(data):
    return hashlib.sha256(data).hexdigest()


def row(store):
    with sqlite3.connect(store.db_path) as connection:
        connection.row_factory = sqlite3.Row
        return dict(connection.execute("SELECT * FROM training_cache_entries").fetchone())


def change_row(store, column, value):
    assert column in {"object_path", "object_sha256", "json_size", "status", "version"}
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(f"UPDATE training_cache_entries SET {column}=?", [value])


def process_write(args):
    db_path, worker = args
    store = TrainingCache(db_path)
    for index in range(8):
        key = f"{worker}:{index}"
        store.put_json("parallel", key, {"worker": worker, "index": index})
        assert store.get_json("parallel", key) == {"worker": worker, "index": index}
    return worker


def test_json_persists_by_namespace_and_complete_identity_with_defensive_values(tmp_path):
    store = TrainingCache(tmp_path / "cache.db")
    original = {"nested": [{"signal": "买入", "probabilities": [.1, .2, .7]}]}
    assert store.get_json("predictions", "full-input-hash") is None
    store.put_json("predictions", "full-input-hash", original)
    original["nested"][0]["signal"] = "mutated"
    read = store.get_json("predictions", "full-input-hash")
    assert read["nested"][0]["signal"] == "买入"
    read["nested"].clear()
    with TrainingCache(store.db_path) as reopened:
        assert len(reopened.get_json("predictions", "full-input-hash")["nested"]) == 1
        assert reopened.get_json("other-stage", "full-input-hash") is None
        assert reopened.get_json("predictions", "changed-model-hash") is None
    assert store.stats == {"hits": 1, "misses": 1}
    stats = store.stats
    stats["hits"] = 100
    assert store.stats["hits"] == 1


def test_dependencies_rehashed_even_if_size_and_mtime_unchanged(tmp_path):
    store = TrainingCache(tmp_path / "cache.db")
    source = tmp_path / "model.json"
    source.write_bytes(b"original")
    deps = {source: sha(source.read_bytes())}
    store.put_json("probabilities", "input", [1, 2, 3], deps)
    assert store.get_json("probabilities", "input", deps) == [1, 2, 3]
    assert store.get_json("probabilities", "input") is None
    stat = source.stat()
    source.write_bytes(b"modified")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert store.get_json("probabilities", "input", deps) is None
    assert store.get_json("probabilities", "input", {source: sha(b"modified")}) is None
    with pytest.raises(ValueError, match="mismatch"):
        store.put_json("probabilities", "new", [1], deps)
    source.unlink()
    assert store.get_json("probabilities", "input", deps) is None


@pytest.mark.parametrize("deps", [{"relative.json": "0" * 64}, {"/missing": "bad"}, []])
def test_invalid_dependencies_are_rejected_before_publication(tmp_path, deps):
    store = TrainingCache(tmp_path / "cache.db")
    with pytest.raises(ValueError):
        store.put_json("n", "k", {"complete": True}, deps)
    assert store.get_json("n", "k", deps) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"v": float("-inf")}, object()])
def test_non_json_payload_never_published(tmp_path, value):
    store = TrainingCache(tmp_path / "cache.db")
    with pytest.raises((ValueError, TypeError)):
        store.put_json("n", "k", value)
    assert store.get_json("n", "k") is None


@pytest.mark.parametrize("corruption", ["missing", "not-gzip", "alter-json", "truncated-gzip"])
def test_corrupt_external_json_is_miss(tmp_path, corruption):
    store = TrainingCache(tmp_path / "cache.db")
    store.put_json("n", "k", {"complete": True})
    path = store.objects_root / row(store)["object_path"]
    if corruption == "missing":
        path.unlink()
    elif corruption == "not-gzip":
        path.write_bytes(b"corrupt")
    elif corruption == "truncated-gzip":
        path.write_bytes(path.read_bytes()[:10])
    else:
        content = json.loads(gzip.decompress(path.read_bytes()))
        content["payload"]["complete"] = False
        path.write_bytes(gzip.compress(json.dumps(content).encode()))
    assert store.get_json("n", "k") is None


@pytest.mark.parametrize("column,value", [
    ("object_path", "../../outside.json.gz"), ("object_sha256", "0" * 64),
    ("json_size", -1), ("json_size", 10**12), ("status", "writing"), ("version", 999),
])
def test_corrupt_index_never_bypasses_verification(tmp_path, column, value):
    store = TrainingCache(tmp_path / "cache.db")
    store.put_json("n", "k", {"complete": True})
    change_row(store, column, value)
    assert store.get_json("n", "k") is None


def test_envelope_binds_key_even_if_database_reference_is_replaced(tmp_path):
    store = TrainingCache(tmp_path / "cache.db")
    store.put_json("n", "original-model", {"score": .9})
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("UPDATE training_cache_entries SET cache_key='different-model'")
    assert store.get_json("n", "different-model") is None


def test_object_symlink_cannot_escape_cache_root(tmp_path):
    store = TrainingCache(tmp_path / "cache.db")
    store.put_json("n", "k", {"score": .9})
    path = store.objects_root / row(store)["object_path"]
    outside = tmp_path / "outside.gz"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    assert store.get_json("n", "k") is None
    with pytest.raises(ValueError, match="symlink"):
        store.put_json("n", "k", {"score": .9})
    assert outside.exists()


@pytest.fixture
def completed_stock(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    stock = root / "features" / "000001_SZ"
    stock.mkdir(parents=True)
    (stock / "observations.parquet").write_bytes(b"existing parquet fixture")
    (stock / "account.json").write_text('{"completed": false}')
    (stock / "manifest.json").write_text('{"committed": true}')
    return root, [str(path.relative_to(root)) for path in sorted(stock.iterdir())]


def test_complete_checkpoint_reuses_summary_and_all_files_read_only(tmp_path, completed_stock):
    root, files = completed_stock
    store = TrainingCache(tmp_path / "cache.db")
    originals = {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns) for name in files}
    summary = {"symbol": "000001.SZ", "stage": "reference_quality", "rows": 42,
               "completed": False, "files": files}
    store.put_checkpoint("all-identity-inputs", root, summary, files)
    assert store.get_checkpoint("all-identity-inputs", root) == summary
    result = store.get_checkpoint("all-identity-inputs", root)
    result["files"].clear()
    assert store.get_checkpoint("all-identity-inputs", root) == summary
    assert store.get_checkpoint("changed-config", root) is None
    different = tmp_path / "different-run"
    different.mkdir()
    assert store.get_checkpoint("all-identity-inputs", different) is None
    assert {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns) for name in files} == originals


@pytest.mark.parametrize("index", range(3))
def test_every_checkpoint_file_is_hashed_not_only_completion_manifest(tmp_path, completed_stock, index):
    root, files = completed_stock
    store = TrainingCache(tmp_path / "cache.db")
    expected = {name: sha((root / name).read_bytes()) for name in files}
    store.put_checkpoint("k", root, {"rows": 1}, expected)
    path = root / files[index]
    before = path.stat()
    path.write_bytes(b"x" * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert store.get_checkpoint("k", root) is None
    with pytest.raises(ValueError, match="mismatch"):
        store.put_checkpoint("k", root, {"rows": 2}, expected)


def test_missing_file_or_uncommitted_publication_never_hits(tmp_path, completed_stock):
    root, files = completed_stock
    store = TrainingCache(tmp_path / "cache.db")
    with pytest.raises(FileNotFoundError):
        store.put_checkpoint("k", root, {"rows": 1}, files + ["not-committed-manifest.json"])
    assert store.get_checkpoint("k", root) is None
    assert not list(store.objects_root.rglob("*.gz"))
    with pytest.raises(ValueError):
        store.put_checkpoint("empty", root, {"rows": 1}, [])
    store.put_checkpoint("k", root, {"rows": 1}, files)
    (root / files[-1]).unlink()
    assert store.get_checkpoint("k", root) is None


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../b", "./a", "a//b", "C:\\outside"])
def test_checkpoint_rejects_path_escape_and_noncanonical_paths(tmp_path, completed_stock, path):
    root, _ = completed_stock
    store = TrainingCache(tmp_path / "cache.db")
    with pytest.raises(ValueError):
        store.put_checkpoint("k", root, {}, [path])
    assert store.get_checkpoint("k", root) is None


def test_checkpoint_rejects_symlinks_even_with_matching_content(tmp_path, completed_stock):
    root, files = completed_stock
    store = TrainingCache(tmp_path / "cache.db")
    store.put_checkpoint("k", root, {}, files)
    source = root / files[0]
    outside = tmp_path / "same-content"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    assert store.get_checkpoint("k", root) is None
    with pytest.raises(ValueError):
        store.put_checkpoint("k", root, {}, files)


def test_atomic_publication_failure_preserves_prior_complete_entry(tmp_path, monkeypatch):
    import my_strategy.services.czsc_training_cache as module
    store = TrainingCache(tmp_path / "cache.db")
    store.put_json("n", "k", {"version": 1})
    def fail(*args):
        raise OSError("simulated interruption before atomic publication")
    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        store.put_json("n", "k", {"version": 2})
    assert store.get_json("n", "k") == {"version": 1}
    assert not list(store.objects_root.rglob(".pending-*"))


def test_simultaneous_thread_initialization_and_writers(tmp_path):
    path = tmp_path / "threads.db"
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sorted(pool.map(process_write, [(str(path), i) for i in range(8)])) == list(range(8))
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM training_cache_entries WHERE status='complete'").fetchone()[0] == 64


def test_multiprocess_writers_publish_only_complete_objects(tmp_path):
    path = tmp_path / "processes.db"
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        assert sorted(pool.map(process_write, [(str(path), i) for i in range(3)])) == [0, 1, 2]
    store = TrainingCache(path)
    for worker in range(3):
        for index in range(8):
            assert store.get_json("parallel", f"{worker}:{index}") == {"worker": worker, "index": index}


def test_shared_instance_same_key_concurrency_keeps_valid_final_value(tmp_path):
    store = TrainingCache(tmp_path / "cache.db")
    def write(index):
        store.put_json("n", "k", {"version": index, "padding": "x" * 10000})
        result = store.get_json("n", "k")
        assert result is not None
        assert result["version"] in range(32)
        assert result["padding"] == "x" * 10000
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(32)))
    assert store.stats == {"hits": 32, "misses": 0}
