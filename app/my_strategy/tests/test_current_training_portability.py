"""Owned process cancellation and immutable evidence on all supported OSes."""
from pathlib import Path, PureWindowsPath
import ctypes
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


def test_windows_cancel_branch_uses_owned_job_without_pid_scan_or_killpg(tmp_path, monkeypatch):
    # Native Windows exercises the real Job Object. POSIX checks orchestration
    # with a fake job owner, without pretending its process group is a Win job.
    monkeypatch.setattr(center, "_is_windows", lambda: True)
    popen, killpg = subprocess.Popen, getattr(os, "killpg", None)
    calls = []

    if os.name != "nt":
        def portable_start(args, **kwargs):
            assert kwargs.pop("creationflags") == 0x00000200
            assert "start_new_session" not in kwargs
            process = popen(args, start_new_session=True, **kwargs)

            def stop(owned):
                assert owned is process
                calls.append(process.pid)
                killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)

            return process, SimpleNamespace(stop=stop)

        monkeypatch.setattr(center, "_start_windows_process", portable_start)
        monkeypatch.setattr(os, "killpg", lambda *args: pytest.fail("Windows cancellation cannot call killpg"))
        monkeypatch.setattr(psutil.Process, "children", lambda *args, **kwargs: pytest.fail("Windows job cannot enumerate a PID tree"))
    _owned_tree_cancellation(tmp_path)
    if os.name != "nt":
        assert len(calls) == 1


def test_native_parent_exits_before_cancel_while_owned_grandchild_holds_stdout(tmp_path):
    # No skip: this exact orphan/stdout regression runs under native Windows CI.
    canary = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    grandchild = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
    child = ("import os,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-u','-c'," + repr(grandchild)
             + "]); print('ORPHAN '+str(os.getpid())+' '+str(p.pid),flush=True); time.sleep(60)")
    code = ("import os,subprocess,sys; print('PARENT '+str(os.getpid()),flush=True); "
            "subprocess.Popen([sys.executable,'-u','-c'," + repr(child) + "])")
    parent, owned = [], []
    deadline = time.monotonic() + 15

    def progress(event):
        message = event.get("message", "")
        if message.startswith("PARENT "):
            parent.append(int(message.split()[1]))
        if message.startswith("ORPHAN "):
            owned.extend(int(value) for value in message.split()[1:])

    def check_cancel():
        if parent and owned and not _is_alive(parent[0]):
            assert all(_is_alive(pid) for pid in owned)
            raise RuntimeError("cancel after owned parent exited")
        if time.monotonic() > deadline:
            raise AssertionError("owned parent did not exit before descendants")

    try:
        with pytest.raises(RuntimeError, match="cancel after owned parent exited"):
            center.TrainingCenter._run_process([sys.executable, "-u", "-c", code],
                                               tmp_path / "parent-exited.log", progress, check_cancel)
        assert len(parent) == 1 and len(owned) == 2
        # POSIX SIGKILL delivery is asynchronous after an already-exited parent.
        deadline = time.monotonic() + 5
        while any(_is_alive(pid) for pid in owned) and time.monotonic() < deadline:
            threading.Event().wait(.01)
        assert all(not _is_alive(pid) for pid in parent + owned)
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


def test_windows_job_structure_layout_uses_fixed_width_dword():
    assert ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(center._JobBasicLimit) == 64
    assert ctypes.sizeof(center._JobIoCounters) == 48
    assert ctypes.sizeof(center._JobExtendedLimit) == 144
    assert ctypes.sizeof(center._JobAccounting) == 48
    assert center._JobBasicLimit.flags.offset == 16
    assert center._JobAccounting.active_processes.offset == 40


def test_native_normal_return_does_not_leave_a_windows_owned_background_descendant(tmp_path):
    # Native Windows must clean even descendants that closed stdout themselves.
    # The existing POSIX normal-return lifecycle is deliberately retained.
    owned = []
    code = ("import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
            "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); print('BACKGROUND '+str(p.pid),flush=True)")

    def progress(event):
        message = event.get("message", "")
        if message.startswith("BACKGROUND "):
            owned.append(int(message.split()[1]))

    try:
        assert center.TrainingCenter._run_process([sys.executable, "-u", "-c", code],
                                                  tmp_path / "normal-background.log", progress, None) == {}
        assert len(owned) == 1
        assert _is_alive(owned[0]) == (os.name != "nt")
    finally:
        for pid in owned:
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass


def test_windows_reader_start_failure_still_stops_owned_job(tmp_path, monkeypatch):
    monkeypatch.setattr(center, "_is_windows", lambda: True)
    popen = subprocess.Popen
    acquired, stopped = [], []
    if os.name == "nt":
        original = center._start_windows_process

        def start(args, **kwargs):
            process, job = original(args, **kwargs)
            acquired.append(process)
            return process, job
    else:
        def start(args, **kwargs):
            kwargs.pop("creationflags")
            process = popen(args, **kwargs)
            acquired.append(process)

            def stop(owned):
                stopped.append(owned.pid)
                owned.kill()
                owned.wait(timeout=5)

            return process, SimpleNamespace(stop=stop)

    monkeypatch.setattr(center, "_start_windows_process", start)
    monkeypatch.setattr(threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("reader resource exhausted")))
    with pytest.raises(RuntimeError, match="reader resource exhausted"):
        center.TrainingCenter._run_process([sys.executable, "-c", "import time; time.sleep(60)"],
                                           tmp_path / "thread-start-failed.log", None, None)
    assert len(acquired) == 1 and acquired[0].poll() is not None
    assert acquired[0].stdout.closed
    if os.name != "nt":
        assert stopped == [acquired[0].pid]


class FakeJobApi:
    def __init__(self, fail=None):
        self.fail = fail
        self.events = []

    def _call(self, name, *args):
        self.events.append((name, *args))
        if self.fail == name:
            raise OSError("native failure " + name)

    def create(self):
        self._call("create")
        return 99

    def configure(self, job):
        self._call("configure", job)

    def assign(self, job, process):
        self._call("assign", job, process)

    def resume(self, thread):
        self._call("resume", thread)

    def terminate_process(self, process):
        self._call("terminate_process", process)

    def wait_process(self, process):
        self._call("wait_process", process)

    def terminate(self, job):
        self._call("terminate", job)

    def active_count(self, job):
        self._call("active_count", job)
        return 0

    def close(self, handle):
        self._call("close", handle)


@pytest.mark.parametrize("failure", ["create", "configure"])
def test_windows_job_creation_failure_releases_only_acquired_handle(failure):
    api = FakeJobApi(failure)
    with pytest.raises(OSError, match="native failure"):
        center._WindowsJob(api)
    assert ("close", 99) in api.events if failure == "configure" else not any(e[0] == "close" for e in api.events)


@pytest.mark.parametrize("failure", ["assign", "resume"])
def test_windows_suspended_start_failure_terminates_waits_and_closes_raw_handles(failure):
    api = FakeJobApi(failure)
    job = center._WindowsJob(api)
    events = []

    def create(*args):
        events.append(("create", args[5]))
        return 101, 102, 103, 104

    native = SimpleNamespace(CreateProcess=create, CloseHandle=lambda handle: events.append(("close", handle)))
    proxy = center._WindowsCreateProcessProxy(native, job)
    with pytest.raises(OSError, match="native failure " + failure):
        proxy.CreateProcess(None, "command", None, None, False, 0x200, None, None, None)
    assert events == [("create", 0x204), ("close", 102), ("close", 101)]
    assert ("terminate_process", 101) in api.events
    assert any(e[0] == "resume" for e in api.events) == (failure == "resume")
    job.stop()
    assert api.events[-1] == ("close", 99)


def test_windows_suspended_start_success_transfers_handles_to_cpython_only():
    api = FakeJobApi()
    job = center._WindowsJob(api)
    native = SimpleNamespace(CreateProcess=lambda *args: (101, 102, 103, 104),
                             CloseHandle=lambda *args: pytest.fail("successful proxy cannot close CPython-owned handles"))
    assert center._WindowsCreateProcessProxy(native, job).CreateProcess(None, "command", None, None, False, 0, None, None, None) == (101, 102, 103, 104)
    assert api.events[-2:] == [("assign", 99, 101), ("resume", 102)]
    assert job.raw_process == 101 and job.raw_thread == 102
    job.stop()
    assert ("wait_process", 101) in api.events
    assert ("close", 101) in api.events and ("close", 102) in api.events


def test_windows_create_proxy_tracks_cpython_thread_close_without_double_closing():
    api = FakeJobApi()
    job = center._WindowsJob(api)
    closed = []
    native = SimpleNamespace(CreateProcess=lambda *args: (101, 102, 103, 104), CloseHandle=closed.append)
    proxy = center._WindowsCreateProcessProxy(native, job)
    proxy.CreateProcess(None, "command", None, None, False, 0, None, None, None)
    proxy.CloseHandle(102)
    assert job.raw_thread is None and closed == [102]
    job.stop()
    assert ("close", 102) not in api.events


class FakeParentHandle(int):
    def __new__(cls, value, events):
        self = int.__new__(cls, value)
        self.events, self.closed = events, False
        return self

    def Close(self):
        if not self.closed:
            self.closed = True
            self.events.append(("close_parent",))


def test_windows_partial_popen_constructor_failure_releases_adopted_parent_and_pending_thread(monkeypatch):
    api = FakeJobApi()
    job = center._WindowsJob(api)
    handle = FakeParentHandle(101, api.events)

    def partial_constructor(self, *args, **kwargs):
        self._child_created = False  # Mock has no real subprocess to finalize.
        self._handle = handle
        self.wait = lambda **kwargs: api.events.append(("wait",))
        job.raw_process, job.raw_thread = 101, 102
        raise RuntimeError("partial constructor after hp adoption")

    monkeypatch.setattr(subprocess.Popen, "__init__", partial_constructor)
    with pytest.raises(RuntimeError, match="partial constructor after hp adoption"):
        center._OwnedWindowsPopen("command", windows_job=job)
    assert handle.closed and job.handle is None
    assert api.events[-5:] == [("wait",), ("close_parent",), ("close", 102), ("active_count", 99), ("close", 99)]
    assert ("close", 101) not in api.events


def test_windows_per_call_subprocess_namespace_never_assigns_unrelated_creation(monkeypatch):
    api = FakeJobApi()
    job = center._WindowsJob(api)
    calls = []
    native = SimpleNamespace(CreateProcess=lambda *args: calls.append(args[5]) or (101, 102, 103, 104))
    monkeypatch.setitem(globals(), "_winapi", native)

    def execute(self, flags):
        return _winapi.CreateProcess(None, "command", None, None, False, flags, None, None, None)

    monkeypatch.setattr(subprocess.Popen, "_execute_child", execute)
    process = object.__new__(center._OwnedWindowsPopen)
    process._child_created = False
    process._windows_job = job
    assert process._execute_child(0x200) == (101, 102, 103, 104)
    assert execute(None, 0x200) == (101, 102, 103, 104)
    assert execute.__globals__["_winapi"] is native
    assert calls == [0x204, 0x200]
    assert sum(event[0] == "assign" for event in api.events) == 1
    job.stop()


def test_windows_job_native_policy_is_noninheritable_kill_on_close_without_breakaway():
    api = object.__new__(center._WindowsJobApi)
    captured = []

    def set_limits(handle, kind, value, size):
        captured.append(("limits", handle, kind, value._obj.basic.flags, size))
        return 1

    api.kernel = SimpleNamespace(
        SetHandleInformation=lambda *args: captured.append(("inherit", *args)) or 1,
        SetInformationJobObject=set_limits)
    api.configure(99)
    assert captured == [("inherit", 99, 1, 0), ("limits", 99, 9, 0x2000, 144)]


@pytest.mark.parametrize("previous", [0, 2, 0xFFFFFFFF])
def test_windows_resume_rejects_any_count_except_one(previous, monkeypatch):
    api = object.__new__(center._WindowsJobApi)
    api.kernel = SimpleNamespace(ResumeThread=lambda thread: previous)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 123, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda error: OSError(error, "native resume error"), raising=False)
    with pytest.raises((RuntimeError, OSError)):
        api.resume(102)
    api.kernel.ResumeThread = lambda thread: 1
    api.resume(102)


@pytest.mark.parametrize("failure", [None, "terminate", "active_count"])
def test_windows_job_stop_closes_parent_before_accounting_and_never_hides_native_failure(failure):
    api = FakeJobApi(failure)
    job = center._WindowsJob(api)
    process = SimpleNamespace(wait=lambda **kwargs: api.events.append(("wait",)),
                              _handle=FakeParentHandle(101, api.events))
    if failure:
        with pytest.raises(OSError, match="native failure " + failure):
            job.stop(process)
    else:
        job.stop(process)
        assert api.events[-4:] == [("wait",), ("close_parent",), ("active_count", 99), ("close", 99)]
    assert job.handle is None and api.events[-1] == ("close", 99)
    job.stop(process)


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
