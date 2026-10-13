"""Owned process cancellation and immutable evidence on all supported OSes."""
from pathlib import Path, PureWindowsPath
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pandas as pd
import psutil
import pytest

from my_strategy.services import model_training_center as center
from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.storage.czsc_results import ResultStore


class FrozenMetadataFile:
    def __init__(self, path):
        self.path = path
        self.metadata = path.stat()
        self.calls = 0

    def __str__(self):
        return str(self.path)

    def stat(self):
        self.calls += 1
        return self.metadata

    def open(self, *args, **kwargs):
        return self.path.open(*args, **kwargs)


@pytest.mark.parametrize("signature", [center._asset_signature, center._event_asset_signature])
def test_windows_content_proof_rejects_equal_size_restored_mtime_and_birthtime(tmp_path, monkeypatch, signature):
    monkeypatch.setattr(center, "_is_windows", lambda: True)
    path = tmp_path / "immutable.json"
    path.write_bytes(b'{"action":"BUY"}')
    frozen = FrozenMetadataFile(path)
    before = signature(iter([frozen]))
    path.write_bytes(b'{"action":"BOY"}')
    os.utime(path, ns=(frozen.metadata.st_atime_ns, frozen.metadata.st_mtime_ns))
    after = signature(iter([frozen]))
    assert before[0][:-1] == after[0][:-1]
    assert before != after


def test_posix_signatures_keep_existing_fields_and_stat_call_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(center, "_is_windows", lambda: False)
    monkeypatch.setattr(center, "_file_hash", lambda path: pytest.fail("POSIX stat cache must not start hashing content"))
    path = tmp_path / "immutable"
    path.write_bytes(b"unchanged")
    frozen = FrozenMetadataFile(path)
    value = frozen.metadata
    assert center._asset_signature([frozen]) == ((str(path), value.st_dev, value.st_ino, value.st_size,
                                                 value.st_mtime_ns, value.st_ctime_ns),)
    assert frozen.calls == 1
    frozen.calls = 0
    assert center._event_asset_signature([frozen]) == ((str(path), value.st_size, value.st_mtime_ns,
                                                       value.st_ctime_ns, value.st_ino),)
    assert frozen.calls == 4


def _is_alive(pid):
    try:
        return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _owned_tree_cancellation(tmp_path):
    # This test runs unconditionally on native Windows and POSIX. The unrelated
    # canary belongs to pytest, not to the launched training process tree.
    canary = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    grandchild = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('READY',flush=True); time.sleep(60)"
    child = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-u','-c'," + repr(grandchild)
             + "],stdout=subprocess.PIPE,text=True); p.stdout.readline(); print('GRAND '+str(p.pid),flush=True); time.sleep(60)")
    code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-u','-c'," + repr(child)
            + "],stdout=subprocess.PIPE,text=True); grand=p.stdout.readline().strip().split()[1]; "
              "print('TREE '+str(p.pid)+' '+grand,flush=True); time.sleep(60)")
    stopped = threading.Event()
    owned = []

    def progress(event):
        if event.get("message", "").startswith("TREE "):
            owned.extend(int(value) for value in event["message"].split()[1:])
            stopped.set()

    def check_cancel():
        if stopped.is_set():
            raise RuntimeError("cancel owned nested tree")

    try:
        with pytest.raises(RuntimeError, match="cancel owned nested tree"):
            center.TrainingCenter._run_process([sys.executable, "-u", "-c", code],
                                               tmp_path / "owned-tree.log", progress, check_cancel)
        assert len(owned) == 2
        assert all(not _is_alive(pid) for pid in owned)
        assert canary.poll() is None
    finally:
        for pid in owned:
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass
        if canary.poll() is None:
            canary.kill()
        canary.wait(timeout=5)


def test_native_cancellation_stops_child_and_grandchild_but_keeps_unrelated_process(tmp_path):
    _owned_tree_cancellation(tmp_path)


def test_windows_cancel_branch_uses_group_and_verified_tree_without_killpg(tmp_path, monkeypatch):
    # On Windows this follows the real system implementation. POSIX additionally
    # exercises the Windows branch using real children and psutil hard kills.
    monkeypatch.setattr(center, "_is_windows", lambda: True)
    popen, run = subprocess.Popen, subprocess.run
    options = []

    if os.name != "nt":
        def portable_popen(*args, **kwargs):
            if "creationflags" in kwargs:
                options.append(dict(kwargs))
                assert kwargs.pop("creationflags") == 0x00000200
                assert "start_new_session" not in kwargs
                kwargs["start_new_session"] = True
            return popen(*args, **kwargs)

        def native_taskkill(*args, **kwargs):
            if args[0][0] == "taskkill":
                assert args[0][2].isdigit() and args[0][3:] == ["/T", "/F"]
                return SimpleNamespace(returncode=0)
            return run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "Popen", portable_popen)
        monkeypatch.setattr(subprocess, "run", native_taskkill)
        monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("Windows cancellation cannot call killpg"))
    _owned_tree_cancellation(tmp_path)
    if os.name != "nt":
        assert options


def test_comparison_registry_uses_portable_relative_asset_path(tmp_path):
    store = ResultStore(tmp_path / "results.db", tmp_path / "runs")
    service = center.TrainingCenter(store=store, metadata_root=tmp_path / "metadata",
                                   personal_db=tmp_path / "personal.db")
    result = dict(run_id="candidate", plan_id="plan", status="pending", passed=False)
    service._save_comparison("candidate", result)
    with sqlite3.connect(service.registry_path) as connection:
        relative = connection.execute("SELECT report_path FROM current_model_comparisons").fetchone()[0]
    assert relative == "candidate/validation/model_comparison/plan/comparison.json"
    assert "\\" not in relative
    assert service.comparison("candidate") == result


def test_broker_comparison_serializes_json_and_ledger_paths_as_posix(tmp_path, monkeypatch):
    from my_strategy.services import czsc_trend_pipeline
    monkeypatch.setattr(czsc_trend_pipeline, "read_snapshot", lambda record: (pd.DataFrame(columns=["date"]), {}))
    root = tmp_path / "candidate"
    dates = ["2027-01-04", "2027-01-05", "2027-01-06"]
    specification = dict(root=str(root), assets_root=str(root / "validation/model_comparison/plan/accounts"),
                         windows=[dict(name="first", test_start=dates[0], test_end=dates[-1])],
                         calendar=dates, models=[("rules", None)], config=strategy_config())
    result = center._compare_stock(dict(symbol="600000.SH", sha256="frozen-dataset"), specification)
    for name in ("rules", "rules_stress"):
        asset = result["windows"][0][name]["asset"]
        assert asset["path"] == "validation/model_comparison/plan/accounts/first/" + name + "/600000_SH.json"
        assert asset["ledger_path"] == "validation/model_comparison/plan/accounts/first/" + name + "/600000_SH.ledger.csv"
        assert all("\\" not in asset[key] and (root / asset[key]).is_file() for key in ("path", "ledger_path"))


def test_windows_relative_path_serialization_contract():
    relative = PureWindowsPath("validation") / "model_comparison" / "plan" / "comparison.json"
    assert relative.as_posix() == "validation/model_comparison/plan/comparison.json"
