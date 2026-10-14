"""Current training plans, verified Broker comparisons and atomic model selection.

Training defaults never alter an existing run. A selection is an append-only
research event and does not confer formal publication certification.
"""
from __future__ import annotations

import copy
import ctypes
from contextlib import closing, contextmanager
from datetime import date
import json
import hashlib
import math
import os
from pathlib import Path
import queue
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from types import FunctionType
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, METADATA_ROOT, PROJECT_ROOT, MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash, create_run_context
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, strategy_config, validate_training_config
from my_strategy.services.czsc_trend_support import _save, _file_hash, _json, resolve_frozen_asset
from my_strategy.storage.czsc_results import ResultStore


class RevisionConflict(ValueError):
    """The default or incumbent changed after the caller read it."""


class ModelQualificationError(ValueError):
    """Intact research artifacts lack heads or statistical coverage for selection."""


def _field(key, label, group, kind, default, minimum=None, maximum=None, unit="", choices=None, impact="retrain"):
    return dict(key=key, name=key, label=label, group=group, type=kind, default=default,
                min=minimum, max=maximum, unit=unit, choices=choices, impact=impact)


FIELDS = [
    _field("feature_start", "历史起点", "数据和资源", "date", "2023-01-01", impact="rebuild"),
    _field("end", "截止日（空值取最新核验行情）", "数据和资源", "date", None, impact="rebuild"),
    _field("liquidity_min_median_amount", "最低日成交额中位数", "股票资格", "number", 30000000., 30000000., 100000000000., "元", impact="rebuild"),
    _field("liquidity_lookback_sessions", "流动性完整窗口", "股票资格", "integer", 20, 20, 120, "核验交易日", impact="rebuild"),
    _field("stock_batch_size", "股票准备批大小（完整训练全部合格股票）", "数据和资源", "integer", 360, 1, 10000, "股", impact="resource"),
    _field("device", "训练设备", "数据和资源", "enum", "cpu", choices=["cpu", "mps", "cuda"]),
    _field("cpu_workers", "准备进程数", "数据和资源", "integer", 6, 1, 16, "进程"),
    _field("seed", "随机种子", "训练", "integer", 42, 0, 2147483647),
    _field("epochs", "线性/MLP训练轮数（方向/树固定拟合）", "训练", "integer", 20, 1, 500, "轮"),
    _field("training_batch_size", "线性/MLP训练批大小", "训练", "integer", 2048, 1, 65536, "行"),
    _field("initial_cash", "每股独立研究本金", "费用和风险", "number", 100000, 1000, 100000000, "元", impact="rebuild"),
    _field("stop_loss", "保护止损", "费用和风险", "number", .08, .001, .08, "比例", impact="rebuild"),
    _field("commission", "佣金", "费用和风险", "number", .0003, .0003, .02, "比例", impact="rebuild"),
    _field("min_commission", "最低佣金", "费用和风险", "number", 5., 5., 1000., "元", impact="rebuild"),
    _field("stamp_tax", "卖出印花税", "费用和风险", "number", .0005, .0005, .02, "比例", impact="rebuild"),
    _field("slippage", "单边滑点", "费用和风险", "number", .0005, .0005, .05, "比例", impact="rebuild"),
    _field("ma_exit_threshold", "持仓 MA 退出初始概率（默认在政策段选定）", "目标和共识", "number", .55, 0, 1, "概率", impact="rebuild"),
    _field("trend_horizon_sessions", "趋势预测周期", "目标和共识", "integer", 10, 1, 60, "核验交易日", impact="rebuild"),
    _field("trend_return_boundary", "趋势收益边界", "目标和共识", "number", .01, .001, .2, "比例", impact="rebuild"),
    _field("ma_weight", "趋势权重", "目标和共识", "number", .4, 0, 1, impact="rebuild"),
    _field("structure_weight", "结构权重", "目标和共识", "number", .2, 0, 1, impact="rebuild"),
    _field("volume_price_weight", "量价权重", "目标和共识", "number", .3, 0, 1, impact="rebuild"),
    _field("quality_weight", "质量权重", "目标和共识", "number", .1, 0, 1, impact="rebuild"),
    _field("buy_base_threshold", "基础买方初始门槛（默认在政策段选定）", "目标和共识", "number", .4, 0, 1, impact="rebuild"),
    _field("buy_total_threshold", "总买方门槛", "目标和共识", "number", .5, 0, 1, impact="rebuild"),
    _field("sell_threshold", "卖方初始门槛（默认在政策段选定）", "目标和共识", "number", .4, 0, 1, impact="rebuild"),
    _field("quality_threshold", "高收益质量概率", "目标和共识", "number", .1, 0, 1),
    _field("min_train_rows", "入场训练最少行", "资格", "integer", 100, 100, 100000),
    _field("min_calibration_rows", "入场校准最少行", "资格", "integer", 30, 30, 100000),
    _field("min_net_return", "高收益费用后净收益", "资格", "number", .08, .08, 1., "比例", impact="rebuild"),
    _field("min_r_multiple", "高收益风险倍数", "资格", "number", 2., 2., 20., "R", impact="rebuild"),
    _field("max_holding_sessions", "退出意图上限", "资格", "integer", 60, 1, 60, "核验交易日", impact="rebuild"),
    _field("auto_replace", "通过研究验收自动替换", "选择", "boolean", True, impact="selection"),
]
DEFAULTS = {field["key"]: field["default"] for field in FIELDS}
EXTRA_KEYS = {"symbols", "calendar_run_id"}
EVALUATION_FILES = ("services/model_training_center.py", "execution/broker.py", "execution/cost_model.py",
                    "execution/ledger.py", "execution/order.py", "services/czsc_backtest.py",
                    "services/czsc_account_artifacts.py")


def evaluation_freeze():
    hashes = {relative:_file_hash(MY_STRATEGY_ROOT/relative) for relative in EVALUATION_FILES}
    return dict(code_hashes=hashes, freeze_hash=stable_hash(hashes))


def verify_evaluation_freeze(result):
    freeze = result.get("evaluation_freeze")
    if not isinstance(freeze, dict) or freeze != evaluation_freeze():
        raise ValueError("费用后评估或Broker执行代码已改变；须按当前代码重新核验，不能沿用旧通过结果")


def training_code_freeze():
    from my_strategy.services.czsc_trend_pipeline import GENERATING_FILES
    hashes = {relative: _file_hash(MY_STRATEGY_ROOT / relative) for relative in GENERATING_FILES}
    return dict(code_hashes=hashes, freeze_hash=stable_hash(hashes))


def _native_input_contract(config):
    from my_strategy.services.czsc_native_evidence import build_native_evidence
    empty = build_native_evidence(pd.DataFrame(), config, market_dates=[])
    body = dict(feature_start=config["feature_start"], liquidity_universe=config["liquidity_universe"],
                native_schema_hash=empty["schema_hash"], native_config_hash=empty["config_hash"])
    return {**body, "contract_hash": stable_hash(body)}


def _is_windows():
    return os.name == "nt"


def _asset_signature(paths):
    paths = tuple(paths)
    signature = tuple((str(path), value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
                      for path in paths for value in [path.stat()])
    if _is_windows():
        # Windows ctime is the birth time, so equal metadata is not a content
        # proof after a same-length write followed by restoring mtime.
        return tuple((*item, _file_hash(path)) for item, path in zip(signature, paths))
    return signature


def _event_asset_signature(paths):
    paths = tuple(paths)
    signature = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns,
                       path.stat().st_ctime_ns, path.stat().st_ino) for path in paths)
    if _is_windows():
        return tuple((*item, _file_hash(path)) for item, path in zip(signature, paths))
    return signature


class _JobBasicLimit(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", ctypes.c_uint32), ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t), ("active_limit", ctypes.c_uint32),
                ("affinity", ctypes.c_size_t), ("priority", ctypes.c_uint32),
                ("scheduling", ctypes.c_uint32)]


class _JobIoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class _JobExtendedLimit(ctypes.Structure):
    _fields_ = [("basic", _JobBasicLimit), ("io", _JobIoCounters),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class _JobAccounting(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("user_time", "kernel_time", "period_user", "period_kernel")] + [
        (name, ctypes.c_uint32) for name in ("page_faults", "total_processes", "active_processes", "terminated_processes")]


class _WindowsJobApi:
    """Native handles only; no PID enumeration or process-global API patching."""
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle, dword, pointer, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int32
        signatures = {
            "CreateJobObjectW": ([pointer, ctypes.c_wchar_p], handle),
            "SetHandleInformation": ([handle, dword, dword], boolean),
            "SetInformationJobObject": ([handle, ctypes.c_int32, pointer, dword], boolean),
            "AssignProcessToJobObject": ([handle, handle], boolean),
            "ResumeThread": ([handle], dword),
            "TerminateJobObject": ([handle, dword], boolean),
            "QueryInformationJobObject": ([handle, ctypes.c_int32, pointer, dword, pointer], boolean),
            "CloseHandle": ([handle], boolean),
            "TerminateProcess": ([handle, dword], boolean),
            "WaitForSingleObject": ([handle, dword], dword),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result

    @staticmethod
    def _check(result):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def create(self):
        return self._check(self.kernel.CreateJobObjectW(None, None))

    def configure(self, handle):
        self._check(self.kernel.SetHandleInformation(handle, 1, 0))
        limits = _JobExtendedLimit()
        # Descendants inherit this unnamed, non-inheritable job membership.
        # No BREAKAWAY flag permits a descendant to escape its owned family.
        limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        self._check(self.kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))

    def assign(self, job, process):
        self._check(self.kernel.AssignProcessToJobObject(job, process))

    def resume(self, thread):
        previous = self.kernel.ResumeThread(thread)
        if previous != 1:
            if previous == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            raise RuntimeError("训练主线程暂停计数异常，拒绝启动未受控进程")

    def terminate(self, job):
        self._check(self.kernel.TerminateJobObject(job, 1))

    def active_count(self, job):
        accounting = _JobAccounting()
        self._check(self.kernel.QueryInformationJobObject(job, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None))
        return accounting.active_processes

    def terminate_process(self, process):
        self._check(self.kernel.TerminateProcess(process, 1))
        self.wait_process(process)

    def wait_process(self, process):
        if self.kernel.WaitForSingleObject(process, 10000) != 0:
            raise RuntimeError("训练自有启动进程未完成终止")

    def close(self, handle):
        self._check(self.kernel.CloseHandle(handle))


class _WindowsJob:
    def __init__(self, api=None):
        self.api = api if api is not None else _WindowsJobApi()
        self.raw_process = self.raw_thread = None
        self.handle = self.api.create()
        try:
            self.api.configure(self.handle)
        except BaseException:
            self.api.close(self.handle)
            self.handle = None
            raise

    def _close_raw_handles(self):
        thread, self.raw_thread = self.raw_thread, None
        try:
            if thread is not None:
                self.api.close(thread)
        finally:
            process, self.raw_process = self.raw_process, None
            if process is not None:
                self.api.close(process)

    def _close_parent_handle(self, process):
        if process is not None:
            if self.raw_process == int(process._handle):
                self.raw_process = None
            process._handle.Close()

    def stop(self, process=None):
        if self.handle is None:
            return
        try:
            self.api.terminate(self.handle)
            if process is not None:
                process.wait(timeout=10)
                # Windows accounting retains an exited process until its last
                # reference is released. Popen already knows the return code.
                self._close_parent_handle(process)
            elif self.raw_process is not None:
                # CPython can fail while closing inherited pipe handles before
                # it adopts hp. Keep the raw handles owned until that handoff.
                self.api.wait_process(self.raw_process)
            self._close_raw_handles()
            deadline = time.monotonic() + 10
            while self.api.active_count(self.handle):
                if time.monotonic() >= deadline:
                    raise RuntimeError("训练结束后自有 Windows Job 仍包含未释放进程")
                threading.Event().wait(.05)
        finally:
            handle, self.handle = self.handle, None
            try:
                self._close_parent_handle(process)
            finally:
                try:
                    self._close_raw_handles()
                finally:
                    self.api.close(handle)


class _WindowsCreateProcessProxy:
    def __init__(self, native, job):
        self.native, self.job = native, job

    def __getattr__(self, name):
        return getattr(self.native, name)

    def CloseHandle(self, handle):
        self.native.CloseHandle(handle)
        if self.job.raw_thread == handle:
            self.job.raw_thread = None

    def CreateProcess(self, *arguments):
        arguments = list(arguments)
        arguments[5] |= 0x00000004  # CREATE_SUSPENDED before any descendant can spawn.
        process, thread, pid, tid = self.native.CreateProcess(*arguments)
        try:
            self.job.api.assign(self.job.handle, process)
            self.job.api.resume(thread)
        except BaseException:
            try:
                self.job.api.terminate_process(process)
            finally:
                try:
                    self.native.CloseHandle(thread)
                finally:
                    self.native.CloseHandle(process)
            raise
        # CPython takes hp ownership and closes ht after this successful return.
        self.job.raw_process, self.job.raw_thread = process, thread
        return process, thread, pid, tid


class _OwnedWindowsPopen(subprocess.Popen):
    def __init__(self, *arguments, windows_job, **options):
        self._windows_job = windows_job
        try:
            super().__init__(*arguments, **options)
        except BaseException:
            windows_job.stop(self if hasattr(self, "_handle") else None)
            raise

    def _execute_child(self, *arguments, **options):
        original = subprocess.Popen._execute_child
        if sys.version_info[:2] != (3, 12) or "_winapi" not in original.__globals__:
            raise RuntimeError("受控 Windows 训练启动仅支持已核验的 Python 3.12 subprocess")
        namespace = dict(original.__globals__)
        namespace["_winapi"] = _WindowsCreateProcessProxy(namespace["_winapi"], self._windows_job)
        local = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
        local.__kwdefaults__ = original.__kwdefaults__
        result = local(self, *arguments, **options)
        self._windows_job.raw_process = None  # Successful CPython hp ownership.
        return result


def _start_windows_process(args, **options):
    job = _WindowsJob()
    try:
        process = _OwnedWindowsPopen(args, windows_job=job, **options)
    except BaseException:
        job.stop()
        raise
    return process, job


def validate_parameters(parameters=None):
    if not isinstance(parameters or {}, dict):
        raise ValueError("parameters 必须是对象")
    unknown = set(parameters or {}) - set(DEFAULTS) - EXTRA_KEYS
    if unknown:
        raise ValueError("未实现的参数：" + ", ".join(sorted(unknown)))
    values = copy.deepcopy(DEFAULTS); values.update(copy.deepcopy(parameters or {}))
    for field in FIELDS:
        key, value, kind = field["key"], values[field["key"]], field["type"]
        if kind == "date":
            if value is None and key == "end":
                continue
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError(key + " 必须为 YYYY-MM-DD")
        elif kind == "boolean":
            if type(value) is not bool:
                raise ValueError(key + " 必须为布尔值")
        elif kind == "enum":
            if value not in field["choices"]:
                raise ValueError(key + " 选项不受支持")
        else:
            members = value if kind == "integer_list" and isinstance(value, list) else [value]
            if kind == "integer_list" and (not isinstance(value, list) or not value or value != sorted(set(value))):
                raise ValueError(key + " 必须为递增正整数列表")
            for number in members:
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                    raise ValueError(key + " 必须为有限数")
                if kind in {"integer", "integer_list"} and not isinstance(number, int):
                    raise ValueError(key + " 必须为整数")
                if field["min"] is not None and number < field["min"] or field["max"] is not None and number > field["max"]:
                    raise ValueError(key + " 超出当前允许范围")
    if not math.isclose(sum(values[k] for k in ("ma_weight", "structure_weight", "volume_price_weight", "quality_weight")), 1., abs_tol=1e-10):
        raise ValueError("趋势、结构、量价和质量权重合计须为 1")
    if values["end"] and values["feature_start"] >= values["end"]:
        raise ValueError("历史起点须早于截止日")
    if "symbols" in values:
        from my_strategy.adapters.czsc_adapter import normalize_symbol
        if not isinstance(values["symbols"], list) or not values["symbols"]:
            raise ValueError("symbols 必须为非空固定股票列表")
        values["symbols"] = list(dict.fromkeys(normalize_symbol(s) for s in values["symbols"]))
    if values.get("calendar_run_id") is not None:
        _identifier(values["calendar_run_id"])
    return values


def _identifier(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError("无效运行身份")
    return value


def _configuration(values, end):
    config = strategy_config()
    config.update(feature_start=values["feature_start"], seed=values["seed"], epochs=values["epochs"],
                  batch_size=values["training_batch_size"], min_train_rows=values["min_train_rows"],
                  min_calibration_rows=values["min_calibration_rows"])
    config.update(full_qualified_pool=True, stock_batch_size=values["stock_batch_size"],
        liquidity_universe=dict(lookback_sessions=values["liquidity_lookback_sessions"],
            min_median_amount=values["liquidity_min_median_amount"], complete_verified_window=True,
            selection="historical_daily_eligible_union"))
    config.pop("training_capacity_tiers", None)
    config["account"].update(initial_cash=values["initial_cash"], start=values["feature_start"], stop_loss=values["stop_loss"],
        commission=values["commission"], min_fee=values["min_commission"], stamp=values["stamp_tax"],
        slippage=values["slippage"], holding_exit_threshold=values["ma_exit_threshold"], max_holding_sessions=values["max_holding_sessions"])
    config["signal_points"]["ma_exit_threshold"] = values["ma_exit_threshold"]
    config["high_return"].update(min_net_return=values["min_net_return"], min_r_multiple=values["min_r_multiple"], max_holding_sessions=values["max_holding_sessions"])
    config["trend"].update(horizon_sessions=values["trend_horizon_sessions"], minimum_return_boundary=values["trend_return_boundary"])
    config["consensus"].update(source_weights=dict(ma=values["ma_weight"], structure=values["structure_weight"], wyckoff=values["volume_price_weight"]),
        quality_weight=values["quality_weight"], buy_base_threshold=values["buy_base_threshold"], buy_total_threshold=values["buy_total_threshold"],
        sell_threshold=values["sell_threshold"], high_return_threshold=values["quality_threshold"])
    for key in ("buy_base_threshold","sell_threshold"):
        if values[key]!=DEFAULTS[key]: config["consensus"]["policy_grid"][key]=[values[key]]
    if values["ma_exit_threshold"]!=DEFAULTS["ma_exit_threshold"]:
        config["exit_threshold_grid"]=[values["ma_exit_threshold"]]
    # Calendar-relative quarters advance with future data; development windows
    # never include the partially observed quarter containing the end date.
    last = pd.Timestamp(end).to_period("Q") - 1
    windows = []
    for period in pd.period_range(last - 3, last, freq="Q"):
        validation, training = period - 1, period - 2
        windows.append(dict(name=str(period), train_end=str(training.end_time.date()),
            validation_start=str(validation.start_time.date()), validation_end=str(validation.end_time.date()),
            test_start=str(period.start_time.date()), test_end=str(period.end_time.date())))
    config["development_windows"] = windows
    validation = pd.Timestamp(end).to_period("Q") - 1
    config["production"] = dict(name="production", train_end=str((validation-1).end_time.date()),
                                validation_start=str(validation.start_time.date()), validation_end=str(validation.end_time.date()))
    return validate_training_config(config)


def _connect(path):
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _reference_quality_summary(report):
    reference=report.get("reference_quality") or {}
    qualification=reference.get("qualification") or {}
    return dict(status=reference.get("status", "unavailable"),
        heads=len(reference.get("bundles") or []), unavailable_heads=reference.get("unavailable_heads") or [],
        population=(reference.get("population") or {}).get("population"),
        population_hash=(reference.get("population") or {}).get("population_hash"),
        fee_status=qualification.get("status", "unavailable"),
        fee_qualified=qualification.get("passed") is True,
        qualification_path=qualification.get("path"), qualification_sha256=qualification.get("sha256"),
        required_for_core_selection=False, publication_allowed=False)


class TrainingCenter:
    _proof_cache = {}
    _proof_lock = threading.RLock()
    def __init__(self, root=None, store=None, *, db_path=None, runs_root=None, metadata_root=None):
        self.store = store or ResultStore()
        self.runs_root = Path(runs_root or getattr(self.store, "runs_root", None) or ARTIFACT_RUNS_ROOT)
        self.db_path = Path(db_path or Path(metadata_root or METADATA_ROOT) / "model_training.db")
        self.registry_path = Path(self.store.db_path)
        self.selection_guard = None
        self.integrity_errors = []
        self._source_reuse_cache = {}
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS training_profiles(profile_id TEXT,revision INTEGER,name TEXT,parameters TEXT,hash TEXT,created_at TEXT,PRIMARY KEY(profile_id,revision));
                CREATE TABLE IF NOT EXISTS training_default(id INTEGER PRIMARY KEY CHECK(id=1),profile_id TEXT,revision INTEGER);
                CREATE TABLE IF NOT EXISTS training_plans(plan_id TEXT PRIMARY KEY,payload TEXT NOT NULL,created_at TEXT NOT NULL,claimed INTEGER NOT NULL DEFAULT 0);
            """)
            conn.execute("INSERT OR IGNORE INTO training_default VALUES (1,'builtin',0)")
            conn.commit()
        self._migrate_saved_default()
        with closing(_connect(self.registry_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS current_model_selection(id INTEGER PRIMARY KEY CHECK(id=1),run_id TEXT,event_id INTEGER,effective_from TEXT);
                CREATE TABLE IF NOT EXISTS current_model_events(event_id INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS current_model_comparisons(run_id TEXT PRIMARY KEY,report_path TEXT NOT NULL,sha256 TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS current_events_no_update BEFORE UPDATE ON current_model_events BEGIN SELECT RAISE(ABORT,'selection events are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS current_events_no_delete BEFORE DELETE ON current_model_events BEGIN SELECT RAISE(ABORT,'selection events are immutable'); END;
            """)
            conn.commit()

    def _migrate_saved_default(self):
        """Append a current policy revision; never rewrite archived defaults."""
        with closing(_connect(self.db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT * FROM training_default WHERE id=1").fetchone()
            row = conn.execute("SELECT * FROM training_profiles WHERE profile_id=? AND revision=?",
                               (current["profile_id"], current["revision"])).fetchone()
            if not row:
                return
            archived = json.loads(row["parameters"])
            if not ({"cohort_size", "capacity_tiers"} & set(archived)):
                return
            parameters = {k:v for k,v in archived.items() if k in DEFAULTS}
            parameters.update(liquidity_min_median_amount=30000000., liquidity_lookback_sessions=20,
                              stock_batch_size=360, end=None)
            parameters = validate_parameters(parameters)
            revision = current["revision"] + 1
            conn.execute("INSERT INTO training_profiles VALUES (?,?,?,?,?,?)", ("personal", revision,
                "个人默认 · 历史流动性完整池", json.dumps(parameters, ensure_ascii=False),
                stable_hash(parameters), local_now().isoformat()))
            conn.execute("UPDATE training_default SET profile_id='personal',revision=? WHERE id=1", (revision,))
            conn.commit()

    def capabilities(self):
        import torch
        devices = ["cpu"]
        if torch.backends.mps.is_available(): devices.append("mps")
        if torch.cuda.is_available(): devices.append("cuda")
        return dict(model_family="current", strategy_version=STRATEGY_ID, devices=devices,
            schema={"fields": copy.deepcopy(FIELDS)}, defaults=copy.deepcopy(DEFAULTS),
            stages=[dict(key=key, label=label, trainable=trainable, status=status) for key, label, trainable, status in (
                ("data", "行情与核验日历", False, "available"), ("trend", "三分类趋势", True, "available"),
                ("structure", "确认结构证据", False, "available"), ("volume_price", "因果量价证据", False, "available"),
                ("ma_exit", "信号固定持仓退出", True, "available"), ("entry", "高收益入场质量", True, "available"),
                ("reference_quality", "无账户参考质量", True, "available"),
                ("exit", "退出观察", True, "observation_only"), ("return_loss", "辅助收益与亏损", False, "not_connected"))],
            qualification={"min_stocks": 60, "round_trips_per_window": 50, "round_trips_total": 200,
                           "min_net_return": .08, "min_r_multiple": 2., "formal_windows": 3, "formal_sessions_per_window": 480})

    def profiles(self):
        builtin = dict(profile_id="builtin", name="内置基准", revision=0, parameters=copy.deepcopy(DEFAULTS), hash=stable_hash(DEFAULTS), builtin=True)
        with closing(_connect(self.db_path)) as conn:
            rows = conn.execute("SELECT p.* FROM training_profiles p WHERE revision=(SELECT MAX(revision) FROM training_profiles q WHERE p.profile_id=q.profile_id) ORDER BY created_at DESC").fetchall()
            default = conn.execute("SELECT * FROM training_default WHERE id=1").fetchone()
            selected = conn.execute("SELECT * FROM training_profiles WHERE profile_id=? AND revision=?", (default["profile_id"], default["revision"])).fetchone()
        def profile(row):
            values = json.loads(row["parameters"])
            return {**dict(row), "parameters": values, "builtin": False,
                    "archived": bool({"cohort_size", "capacity_tiers"} & set(values))}
        return dict(items=[builtin, *[profile(row) for row in rows]], default_profile=profile(selected) if selected else builtin,
                    default_profile_id=default["profile_id"], default_revision=default["revision"])

    def save_default(self, parameters, expected_revision):
        values = validate_parameters(parameters)
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValueError("expected_revision 必须为整数")
        with closing(_connect(self.db_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT revision FROM training_default WHERE id=1").fetchone()[0]
            if current != expected_revision:
                raise RevisionConflict("默认参数已被其他页面修改，请刷新后保存")
            revision = current + 1
            conn.execute("INSERT INTO training_profiles VALUES (?,?,?,?,?,?)", ("personal", revision, "个人默认", json.dumps(values, ensure_ascii=False), stable_hash(values), local_now().isoformat()))
            conn.execute("UPDATE training_default SET profile_id='personal',revision=? WHERE id=1", (revision,))
            conn.commit()
        return self.profiles()

    def current_selection(self):
        with closing(_connect(self.registry_path)) as conn:
            row = conn.execute("SELECT * FROM current_model_selection WHERE id=1").fetchone()
        return dict(row) if row else None

    def active_run_id(self, as_of=None):
        selected = self.current_selection()
        self.integrity_errors=[]
        if selected and selected["run_id"]:
            try:
                with closing(_connect(self.registry_path)) as conn:
                    latest=conn.execute("SELECT MAX(event_id) FROM current_model_events").fetchone()[0]
                proof=self._event_proof(selected["event_id"])
                if latest!=selected["event_id"] or proof["run_id"]!=selected["run_id"] or proof["effective_from"]!=selected["effective_from"]:
                    raise ValueError("活动引用与最新不可变选择事件不一致")
            except (OSError,ValueError,KeyError) as exc:
                self.integrity_errors=[str(exc)]; return None
        if selected and selected["run_id"] and not selected["effective_from"]:
            self._resolve_pending_session(selected)
            selected = self.current_selection()
        day = str(as_of or local_now().date())[:10]
        if not selected or not selected["run_id"]: return None
        if selected["effective_from"] and selected["effective_from"] <= day:
            return selected["run_id"]
        # A queued replacement must not make a still-effective incumbent vanish.
        events = self.selection_events(200)["items"]
        previous = next((item["payload"].get("previous_run_id") for item in events if item["event_id"] == selected["event_id"]), None)
        for item in events:
            payload = item["payload"]
            if payload.get("run_id") != previous: continue
            if payload.get("effective_from") and payload["effective_from"] <= day:
                try: self._event_proof(item["event_id"])
                except (OSError,ValueError,KeyError) as exc:
                    self.integrity_errors=[str(exc)]; return None
                return previous
            previous = payload.get("previous_run_id")
        return None

    def _event_proof(self,event_id):
        with closing(_connect(self.registry_path)) as conn:
            row=conn.execute("SELECT * FROM current_model_events WHERE event_id=?",(event_id,)).fetchone()
        if not row: raise ValueError("当前活动引用缺少选择事件")
        payload=json.loads(row["payload"])
        if row["kind"] not in {"select","activation","rollback"} or payload.get("scope")!="research_model_selection" or payload.get("publication_allowed") is not False or not payload.get("run_id"):
            raise ValueError("当前选择事件用途或模型身份无效")
        plan=self.load_plan(payload["plan_id"])
        if plan["plan_hash"]!=payload["plan_hash"]: raise ValueError("选择事件冻结计划不一致")
        root=self.runs_root/_identifier(payload["run_id"])
        path=root/"validation/model_comparison"/_identifier(payload["plan_id"])/"comparison.json"
        result=json.loads(path.read_text())
        if stable_hash(result)!=payload["comparison_sha256"] or _file_hash(path)!=payload["comparison_file_sha256"] or result.get("passed") is not True:
            raise ValueError("不可变选择比较证据哈希或验收结果无效")
        verify_evaluation_freeze(result)
        if result.get("run_id")!=payload["run_id"] or result.get("plan_hash")!=plan["plan_hash"]:
            raise ValueError("选择比较与实际当前包不同")
        report_path=root/"reports/trend_consensus.json"
        if _file_hash(report_path)!=result["candidate_report_sha256"]: raise ValueError("当前活动模型报告改变")
        key=(str(self.registry_path.resolve()),int(event_id))
        with self._proof_lock:
            cached=self._proof_cache.get(key)
            if cached and cached["payload_hash"]==stable_hash(payload):
                signature=_event_asset_signature(cached["files"])
                if signature==cached["signature"]: return payload
        _,report=self._model(payload["run_id"])
        records=self._evaluation_records(report,plan)
        files=[path,report_path]
        for window in result["windows"]:
            for arm in window["arms"].values():
                for asset in arm["assets"]:
                    for file_key,sha_key in (("path","sha256"),("ledger_path","ledger_sha256")):
                        target=(root/asset[file_key]).resolve()
                        if not target.is_relative_to(root.resolve()) or _file_hash(target)!=asset[sha_key]:
                            raise ValueError("当前活动选择账本资产改变")
                        if file_key == "path":
                            from my_strategy.services.czsc_account_artifacts import validate_account_artifact
                            validate_account_artifact(target, expected_sha256=asset[sha_key])
                        files.append(target)
        # Immutable current weights, targets, code and external frozen sources
        # share the cache signature; any change forces a complete cold check.
        files.extend(p for p in root.rglob("*") if p.is_file())
        files.extend(MY_STRATEGY_ROOT/p for p in report.get("protocol_freeze",{}).get("code_hashes",{}))
        files.extend(MY_STRATEGY_ROOT/p for p in EVALUATION_FILES)
        for record in records:
            files.extend(resolve_frozen_asset(record[k],runs_root=self.runs_root) for k in ("path","raw_path","native_path") if record.get(k))
        files.extend(self._source_provenance_files(report,plan))
        files=sorted(set(files),key=str)
        signature=_event_asset_signature(files)
        with self._proof_lock:
            self._proof_cache[key]=dict(files=files,signature=signature,payload_hash=stable_hash(payload))
        return payload

    def _source_provenance_files(self,report,plan):
        from my_strategy.services.czsc_research_calendar import load_verified_calendar
        native_root=self.runs_root/_identifier(report["source_snapshot"]["source_native_run_id"])
        native_path=native_root/"reports/native_source.json"
        native=json.loads(native_path.read_text())
        raw_root=self.runs_root/_identifier(native["source_snapshot"]["raw_source_run_id"])
        raw_path=raw_root/"reports/raw_source.json"
        raw=json.loads(raw_path.read_text())
        raw_hash=stable_hash({k:v for k,v in raw.items() if k!="snapshot_hash"})
        if raw_hash!=raw.get("snapshot_hash") or raw_hash!=native["source_snapshot"]["raw_snapshot_hash"]:
            raise ValueError("当前完整包原始来源谱系改变")
        if raw.get("strategy_version")!=STRATEGY_ID or raw.get("status")!="completed" or raw.get("failures"):
            raise ValueError("当前完整包原始来源不完整")
        dates,calendar=load_verified_calendar(plan["calendar_run_id"],runs_root=self.runs_root)
        if stable_hash([d for d in dates if d<=plan["data_end"]])!=plan["calendar_hash"]:
            raise ValueError("当前选择核验日历历史改变")
        if raw.get("calendar_hash")!=plan["calendar_hash"] or set(raw["selected_symbols"])!=set(plan["cohort"]):
            raise ValueError("当前完整来源覆盖或日历不同")
        if report.get("config", {}).get("full_qualified_pool"):
            population = raw.get("liquidity") or {}
            planned = plan.get("liquidity_universe") or {}
            if not planned.get("population_hash") or population.get("population_hash") != planned["population_hash"]:
                raise ValueError("完整来源历史流动性人口与冻结预检不同")
            if population.get("policy") != report["config"]["liquidity_universe"]:
                raise ValueError("实际流动性资格政策与模型配置不同")
        files=[native_path,raw_path,native_root/"data_snapshots/calendar.json",raw_root/"data_snapshots/calendar.json",Path(calendar["path"])]
        files.extend(resolve_frozen_asset(asset["path"],runs_root=self.runs_root) for asset in calendar.get("source_artifacts",[]))
        files.extend(MY_STRATEGY_ROOT/relative for relative in native["source_snapshot"].get("code_hashes",{}))
        for asset in (raw.get("liquidity") or {}).get("membership_assets", []) + (raw.get("liquidity") or {}).get("summary_assets", []):
            path = resolve_frozen_asset(asset["path"], runs_root=self.runs_root)
            if _file_hash(path) != asset["sha256"]:
                raise ValueError("冻结逐日股票池证据改变")
            files.append(path)
        if (raw_root/"reports/liquidity_universe.json").exists():
            files.append(raw_root/"reports/liquidity_universe.json")
        return files

    def _resolve_pending_session(self, selected):
        events = self.selection_events(200)["items"]
        event = next((item for item in events if item["event_id"] == selected["event_id"]), None)
        if not event or not event["payload"].get("calendar_run_id") or self._selection_reasons(): return
        from my_strategy.services.czsc_trend_support import _calendar
        try:
            calendar, _ = _calendar()
            self._model(selected["run_id"])
        except (OSError, ValueError, KeyError): return
        # When a new schedule becomes verifiable, append a real activation
        # decision for a future session; never backdate a previous deployment.
        effective = next((d for d in calendar if d > str(local_now().date())), None)
        if not effective: return
        payload = {**event["payload"], "effective_from":effective,"recorded_at":local_now().isoformat(),"pending_event_id":selected["event_id"]}
        with closing(_connect(self.registry_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT event_id,effective_from FROM current_model_selection WHERE id=1").fetchone()
            if not current or current["event_id"] != selected["event_id"] or current["effective_from"]: return
            if self._selection_reasons(): return
            identity = conn.execute("INSERT INTO current_model_events(kind,payload,created_at) VALUES ('activation',?,?)",(json.dumps(payload,ensure_ascii=False),payload["recorded_at"])).lastrowid
            conn.execute("UPDATE current_model_selection SET event_id=?,effective_from=? WHERE id=1",(identity,effective)); conn.commit()

    def active_model_run_id(self, as_of=None, scope="research"):
        if scope not in {"research", "production"}: raise ValueError("未知模型选择用途")
        # Formal deployment cannot borrow a development research selection.
        if scope == "production": return None
        return self.active_run_id(as_of)

    def selection_events(self, limit=100):
        with closing(_connect(self.registry_path)) as conn:
            rows = conn.execute("SELECT * FROM current_model_events ORDER BY event_id DESC LIMIT ?", (max(1, min(int(limit), 200)),)).fetchall()
        return dict(items=[{**dict(row), "payload": json.loads(row["payload"])} for row in rows])

    def status(self):
        from my_strategy.services.czsc_trend_pipeline import GENERATING_FILES
        current_code={name:_file_hash(MY_STRATEGY_ROOT/name) for name in GENERATING_FILES}
        active = self.active_run_id(); selected = self.current_selection()
        active_effective=None
        if active:
            active_effective=next((e["payload"].get("effective_from") for e in self.selection_events()["items"] if e["payload"].get("run_id")==active and e["payload"].get("effective_from") and e["payload"]["effective_from"]<=str(local_now().date())),None)
        models = []
        for path in sorted(self.runs_root.glob("*/reports/trend_consensus.json"), reverse=True)[:50]:
            try:
                report = json.loads(path.read_text())
                if report.get("strategy_version") != STRATEGY_ID: continue
                archived=(report.get("protocol_freeze", {}).get("code_hashes") != current_code or
                          report.get("config", {}).get("full_qualified_pool") is not True)
                models.append(dict(run_id=report["run_id"], status=report.get("status"), data_end=report.get("data_end"),
                    model_family="current", strategy_version=STRATEGY_ID, initial_cash=report.get("config",{}).get("initial_cash"),
                    checkpoints=[dict(name=b["name"],checkpoint=b["name"],available_at=b["available_at"],heads=sorted(b.get("heads",{}))) for b in report.get("model_bundles",[])],
                    reference_quality=_reference_quality_summary(report),
                    completed_at=report.get("completed_at"), required_heads=report.get("required_head_status", {}),
                    required_recent_heads_ready=report.get("required_recent_heads_ready", False), unavailable_heads=report.get("unavailable_heads", []),
                    trained_stock_count=len(report.get("dataset_records", [])),
                    historical_union_count=len(report.get("validation_records", report.get("dataset_records", []))),
                    latest_eligible_count=(report.get("source_snapshot", {}).get("liquidity") or {}).get("latest_count"),
                    archived=archived, validation_available=not archived,
                    archive_reason="原人口或生成/执行代码与当前规则不同；只读保留原诊断，完整训练须使用新来源" if archived else None,
                    automatic_selection_eligible=report.get("automatic_selection_eligible", False), publication_allowed=False))
            except (OSError, ValueError, KeyError): continue
        return dict(mode="integrity_blocked" if self.integrity_errors else "current_model_research" if active else "rule_observation_no_model", current_selection=selected,
            integrity_errors=self.integrity_errors,
            active_model=dict(run_id=active, effective_from=active_effective, publication_allowed=False) if active else None,
            models=models, certification=dict(status="pending", publication_allowed=False,
                reason="正式认证须在真实完整冻结后累计三个至少 480 个核验交易日的独立窗口"))

    def _verify_reusable_source(self, native_id, config, end, *, expected=None, progress=None, check_cancel=None):
        """Cold-verify complete input assets; cached proofs require unchanged inodes/ctime.

        Only raw/native inputs are reusable. No model, policy account or label
        checkpoint is read or imported here.
        """
        from my_strategy.services.czsc_trend_source import NATIVE_GENERATING_FILES, _verify_population, _raw_content_hash
        from my_strategy.services.czsc_research_calendar import load_verified_calendar
        from my_strategy.adapters.czsc_adapter import legacy_fuyao_source
        import pyarrow.parquet as pq
        native_id = _identifier(native_id)
        runs = self.runs_root.resolve()
        native_root = (runs / native_id).resolve()
        if not native_root.is_relative_to(runs): raise ValueError("复用原生来源越过运行目录")
        native_path = native_root / "reports/native_source.json"
        native_bytes = native_path.read_bytes(); native = json.loads(native_bytes)
        if (not isinstance(native,dict) or not isinstance(native.get("source_snapshot"),dict)
                or not isinstance(native.get("config"),dict)
                or not isinstance(native.get("dataset_records"),list)
                or any(not isinstance(row,dict) for row in native["dataset_records"])):
            raise ValueError("复用原生来源报告、snapshot、配置或记录目录格式无效")
        native_sha = hashlib.sha256(native_bytes).hexdigest()
        if (native.get("run_id") != native_id or native.get("strategy_version") != STRATEGY_ID
                or native.get("status") != "completed" or native.get("failures")
                or native.get("raw_only_rebuild") is not True or native.get("reused_model_or_label") is not False
                or native.get("population_mode") != "full_liquidity_union" or native.get("data_end") != end):
            raise ValueError("复用需要已完成、未复用模型标签且截止日相同的完整原生来源")
        snapshot = native["source_snapshot"]
        raw_id = _identifier(snapshot["raw_source_run_id"]); raw_root = (runs / raw_id).resolve()
        if not raw_root.is_relative_to(runs): raise ValueError("复用原始来源越过运行目录")
        raw_path = raw_root / "reports/raw_source.json"
        raw_bytes = raw_path.read_bytes(); raw = json.loads(raw_bytes)
        if (not isinstance(raw,dict) or not isinstance(raw.get("raw_records"),list)
                or any(not isinstance(row,dict) for row in raw["raw_records"])):
            raise ValueError("复用原始来源报告或记录目录格式无效")
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        if (raw.get("run_id") != raw_id or raw.get("strategy_version") != STRATEGY_ID
                or raw.get("status") != "completed" or raw.get("failures")
                or raw.get("raw_only_rebuild") is not True or raw.get("reused_model_or_label") is not False
                or raw.get("data_end") != end or raw.get("end") != end
                or raw.get("snapshot_hash") != stable_hash({k:v for k,v in raw.items() if k != "snapshot_hash"})
                or snapshot.get("raw_snapshot_hash") != raw["snapshot_hash"]):
            raise ValueError("复用原始来源未完成或来源谱系哈希改变")
        population = native.get("liquidity")
        if (not isinstance(population,dict) or not isinstance(population.get("policy"),dict)
                or not isinstance(population.get("selected_symbols"),list)
                or any(not isinstance(symbol,str) for symbol in population["selected_symbols"])
                or not isinstance(population.get("candidate_symbols"),list)
                or any(not isinstance(symbol,str) for symbol in population["candidate_symbols"])
                or not isinstance(population.get("membership_assets"),list)
                or any(not isinstance(row,dict) for row in population["membership_assets"])
                or not isinstance(population.get("summary_assets"),list)
                or any(not isinstance(row,dict) for row in population["summary_assets"])
                or not isinstance(population.get("daily_counts"),list)
                or any(not isinstance(row,dict) for row in population["daily_counts"])
                or not isinstance(population.get("input_catalog"),list)
                or any(not isinstance(row,dict) for row in population["input_catalog"])):
            raise ValueError("复用流动性人口或资产目录格式无效")
        _verify_population(population)
        if (population != raw.get("liquidity") or population != snapshot.get("liquidity")
                or native.get("liquidity_universe") != population or raw.get("liquidity_universe") != population
                or raw.get("population_mode") != "full_liquidity_union"
                or population.get("feature_start") != config["feature_start"] or population.get("end") != end
                or population.get("policy") != config["liquidity_universe"]
                or population.get("policy_hash") != stable_hash(config["liquidity_universe"])):
            raise ValueError("复用来源的完整人口、窗口或流动性政策与新计划不同")
        symbols = population["selected_symbols"]; records = native["dataset_records"]
        raw_records = raw["raw_records"]
        if (not symbols or symbols != sorted(set(symbols)) or population.get("selected_count") != len(symbols)
                or native.get("selected_symbols") != symbols or raw.get("selected_symbols") != symbols
                or [r["symbol"] for r in records] != symbols or [r["symbol"] for r in raw_records] != symbols):
            raise ValueError("复用来源记录必须覆盖完整历史合格股票并集，不能切片")
        hashes = snapshot.get("code_hashes") or {}
        if not isinstance(hashes,dict) or not isinstance(snapshot.get("raw_input_hashes"),dict):
            raise ValueError("复用生成代码或原始输入哈希目录格式无效")
        if set(hashes) != set(NATIVE_GENERATING_FILES) or any(
                _file_hash(MY_STRATEGY_ROOT / name) != hashes[name] for name in NATIVE_GENERATING_FILES):
            raise ValueError("复用原生来源的实际生成代码已改变或清单不完整")
        input_contract = _native_input_contract(config)
        if input_contract != _native_input_contract(native["config"]):
            raise ValueError("复用原生来源的输入配置或schema与新计划不同")
        calendar_id = _identifier(snapshot["calendar_run_id"])
        calendar, metadata = load_verified_calendar(calendar_id, runs_root=runs)
        dates = [day for day in calendar if day <= end]
        if (not dates or dates[-1] != end or native.get("calendar_dates") != dates or raw.get("dates") != dates
                or raw.get("calendar_run_id") != calendar_id or population.get("calendar_run_id") != calendar_id
                or snapshot.get("calendar_hash") != stable_hash(dates)
                or raw.get("calendar_hash") != stable_hash(dates) or population.get("calendar_hash") != stable_hash(dates)):
            raise ValueError("复用来源与核验日历历史不同")
        cache_key = (native_id, input_contract["contract_hash"])
        if check_cancel: check_cancel()
        cached = self._source_reuse_cache.get(cache_key)
        if cached and _asset_signature(cached["paths"]) == cached["signature"]:
            proof = copy.deepcopy(cached["proof"])
            if expected is not None and proof != expected: raise ValueError("提交来源与冻结复用证明不同")
            if progress: progress(dict(stage="frozen_source_verification", completed=len(proof["asset_catalog"]),
                total=len(proof["asset_catalog"]), message="全部来源文件身份未改变，复用已完成实际SHA冷核验"))
            return proof, copy.deepcopy(population), dates
        catalog = {}
        def asset(value, expected_sha, owner, role, rows=None):
            path = resolve_frozen_asset(value, runs_root=runs)
            if not path.is_relative_to(owner.resolve()) or path.is_symlink():
                raise ValueError("复用资产不属于指定来源运行：" + role)
            if (not isinstance(expected_sha,str) or len(expected_sha) != 64
                    or any(c not in "0123456789abcdef" for c in expected_sha)):
                raise ValueError("复用资产缺少实际SHA：" + role)
            relative = path.relative_to(runs).as_posix()
            prior = catalog.get(relative)
            if prior and (prior["sha256"] != expected_sha or rows is not None and prior.get("rows", rows) != rows):
                raise ValueError("复用资产目录中的身份冲突")
            item = prior or dict(path=relative, sha256=expected_sha, roles=[])
            item["roles"] = sorted(set([*item["roles"],role]))
            if rows is not None: item["rows"] = rows
            catalog[relative] = item
            return path
        asset(native_path,native_sha,native_root,"native_report")
        asset(raw_path,raw_sha,raw_root,"raw_report")
        universe_path = raw_root / "reports/liquidity_universe.json"
        if json.loads(universe_path.read_text()) != population: raise ValueError("冻结人口报告不同")
        asset(universe_path,_file_hash(universe_path),raw_root,"population_report")
        for owner in (raw_root,native_root):
            path = owner / "data_snapshots/calendar.json"; value = json.loads(path.read_text())
            if value.get("verified") is not True or value.get("source_run_id") != calendar_id or value.get("dates") != dates:
                raise ValueError("来源目录的冻结日历不同")
            asset(path,_file_hash(path),owner,"source_calendar")
        calendars = [metadata,*[load_verified_calendar(p["run_id"],runs_root=runs)[1] for p in metadata.get("ancestors",[])]]
        for cal in calendars:
            owner = runs / cal["run_id"]
            asset(cal["path"],cal["file_sha256"],owner,"verified_calendar")
            value = json.loads(Path(cal["path"]).read_text())
            for entry in cal.get("source_artifacts",[]): asset(entry["path"],entry["sha256"],owner,"calendar_source")
            if value.get("artifact"): asset(value["artifact"],value["sha256"],owner,"calendar_dates")
        members = population.get("membership_assets") or []
        candidates = population.get("candidate_symbols") or []
        if ([entry["symbol"] for entry in members] != candidates or candidates != sorted(set(candidates))
                or population.get("candidate_count") != len(candidates)):
            raise ValueError("逐日成员资产未覆盖完整候选股票人口")
        members_by_symbol = {entry["symbol"]:entry for entry in members}
        for entry in members: asset(entry["path"],entry["sha256"],raw_root,"daily_membership",entry["rows"])
        if len(population.get("summary_assets") or []) != 2: raise ValueError("完整来源缺少人口汇总资产")
        for entry in population["summary_assets"]: asset(entry["path"],entry["sha256"],raw_root,"population_summary")
        for record, original in zip(records,raw_records,strict=True):
            if not isinstance(record.get("raw_attrs"),dict): raise ValueError("复用原始观察单位身份格式无效")
            if any(record.get(key) != value for key,value in original.items()): raise ValueError("原生记录原始输入身份不同")
            if record.get("raw_only_rebuild") is not True or record.get("reused_model_or_label") is not False:
                raise ValueError("复用原生记录不能导入旧模型或标签")
            member = members_by_symbol[record["symbol"]]
            if (record.get("liquidity_sha256") != member["sha256"]
                    or record.get("liquidity_policy_hash") != population["policy_hash"]
                    or snapshot.get("raw_input_hashes",{}).get(record["symbol"]) != record.get("raw_sha256")
                    or record.get("native_schema_hash") != input_contract["native_schema_hash"]
                    or record.get("native_config_hash") != input_contract["native_config_hash"]):
                raise ValueError("复用记录资格或输入schema身份不同")
            asset(record["raw_path"],record["raw_sha256"],raw_root,"raw_observations",record["raw_rows"])
            asset(record["liquidity_path"],record["liquidity_sha256"],raw_root,"record_membership",record["liquidity_rows"])
            asset(record["path"],record["sha256"],native_root,"native_features",record["rows"])
            asset(record["native_path"],record["native_sha256"],native_root,"native_evidence")
        paths = [runs / key for key in sorted(catalog)]
        before = _asset_signature(paths)
        total = len(paths)
        if progress: progress(dict(stage="frozen_source_verification",completed=0,total=total,message="逐文件实际读取SHA，核验完整raw/native与人口资产"))
        for index,path in enumerate(paths,1):
            if check_cancel: check_cancel()
            item = catalog[path.relative_to(runs).as_posix()]
            if _file_hash(path) != item["sha256"]: raise ValueError("冻结复用来源资产SHA改变：" + item["path"])
            if "rows" in item and pq.ParquetFile(path).metadata.num_rows != item["rows"]:
                raise ValueError("冻结复用来源Parquet行数不同")
            item["logical_bytes"] = path.stat().st_size
            if progress and (index % 100 == 0 or index == total):
                progress(dict(stage="frozen_source_verification",completed=index,total=total,message="已核验真实来源文件SHA"))
        task_dates = [day for day in dates if day >= config["feature_start"]]
        daily_symbols = {day:[] for day in task_dates}
        for entry in members:
            frame = pd.read_parquet(resolve_frozen_asset(entry["path"],runs_root=runs))
            if (frame.date.tolist() != task_dates or not frame.symbol.eq(entry["symbol"]).all()
                    or not frame.liquidity_policy_hash.eq(population["policy_hash"]).all()
                    or frame.liquidity_pool_eligible.dtype.kind != 'b'):
                raise ValueError("逐日资格成员schema、日期或政策不同")
            for day in frame.loc[frame.liquidity_pool_eligible,"date"]: daily_symbols[day].append(entry["symbol"])
        daily = [dict(date=day,symbols=sorted(stocks)) for day,stocks in daily_symbols.items()]
        if (stable_hash(daily) != population["daily_membership_hash"]
                or sorted({symbol for stocks in daily_symbols.values() for symbol in stocks}) != symbols
                or daily[-1]["symbols"] != population["latest_symbols"]
                or population["latest_count"] != len(daily[-1]["symbols"])
                or population["daily_counts"] != [dict(date=day,eligible_count=len(stocks),excluded_count=len(candidates)-len(stocks))
                    for day,stocks in daily_symbols.items()]):
            raise ValueError("实际逐日资格与冻结人口并集或计数不同")
        for record in records:
            if check_cancel: check_cancel()
            raw_frame = pd.read_parquet(resolve_frozen_asset(record["raw_path"],runs_root=runs))
            feature_frame = pd.read_parquet(resolve_frozen_asset(record["path"],runs_root=runs))
            raw_dates = pd.to_datetime(raw_frame.date).dt.strftime('%Y-%m-%d').tolist()
            if (not raw_dates or raw_dates != sorted(set(raw_dates)) or raw_dates[-1] != record["data_end"]
                    or raw_dates[-1] > end or not raw_frame.symbol.eq(record["symbol"]).all()
                    or _raw_content_hash(raw_frame) != record["raw_content_sha256"]
                    or _raw_content_hash(feature_frame) != record["raw_content_sha256"]):
                raise ValueError("原始价格、原生输入行或实际观察身份改变")
            attrs = record.get("raw_attrs") or {}
            legacy = int(raw_frame.source.map(legacy_fuyao_source).sum()); unverified = int(raw_frame.has_trade_price.ne(1).sum())
            if (attrs.get("price_basis") != ("mixed_legacy_adjustment_unverified" if legacy else "unadjusted")
                    or attrs.get("legacy_fuyao_bars") != legacy or attrs.get("unverified_trade_price_bars") != unverified
                    or attrs.get("volume_unit") != "shares" or attrs.get("amount_unit") != "CNY"
                    or attrs.get("data_version") != record["raw_series_hash"]):
                raise ValueError("复用真实价格来源、数量单位或观察身份不同")
        if _asset_signature(paths) != before: raise ValueError("复用来源在冷核验过程中改变")
        proof = dict(mode="verified_complete_raw_native_inputs",source_native_run_id=native_id,
            native_report_sha256=native_sha,raw_source_run_id=raw_id,raw_report_sha256=raw_sha,
            raw_snapshot_hash=raw["snapshot_hash"],records_hash=stable_hash(records),
            calendar_run_id=calendar_id,calendar_hash=stable_hash(dates),data_end=end,feature_start=config["feature_start"],
            selected_count=len(symbols),selected_symbols_hash=stable_hash(symbols),population_hash=population["population_hash"],
            daily_membership_hash=population["daily_membership_hash"],policy_hash=population["policy_hash"],
            native_generator_code_hashes=hashes,source_input_contract=input_contract,
            asset_catalog=[catalog[key] for key in sorted(catalog)],reused_model_or_label=False)
        proof["asset_catalog_hash"] = stable_hash(proof["asset_catalog"])
        if expected is not None and proof != expected: raise ValueError("实际来源与新计划冻结的复用证明不同")
        self._source_reuse_cache[cache_key] = dict(paths=paths,signature=before,proof=copy.deepcopy(proof))
        return proof,copy.deepcopy(population),dates

    def preflight(self, parameters=None, profile_id=None, *, source_native_run_id=None, progress=None):
        values = validate_parameters(parameters if parameters is not None else self.profiles()["default_profile"]["parameters"])
        errors, warnings, cohort = [], [], []
        calendar_id, calendar, end = None, [], values["end"]
        requested_end = end
        config, universe, source_reuse = None, None, None
        date_status = dict(code="not_checked", requested_end=requested_end, effective_end=None)
        try:
            from my_strategy.services.czsc_trend_math import _device
            _device(values["device"])
            from my_strategy.services.czsc_trend_support import _calendar, VERIFIED_SOURCES
            if source_native_run_id is not None:
                identity = _identifier(source_native_run_id)
                header = json.loads((self.runs_root / identity / "reports/native_source.json").read_text())
                if (not isinstance(header,dict) or not isinstance(header.get("data_end"),str)
                        or not isinstance(header.get("source_snapshot"),dict)):
                    raise ValueError("复用原生来源预检头或snapshot格式无效")
                latest = header["data_end"]
                end = end or latest
                calendar_id = header["source_snapshot"]["calendar_run_id"]
                if values.get("calendar_run_id") not in {None,calendar_id}: raise ValueError("复用来源日历与请求不同")
                config = _configuration(values,end)
                source_reuse,universe,calendar = self._verify_reusable_source(identity,config,end,progress=progress)
            else:
                calendar, calendar_id = _calendar(values.get("calendar_run_id"))
                from my_strategy.adapters.czsc_adapter import _connection
                with _connection(None) as conn:
                    latest = conn.execute("SELECT MAX(date) FROM stock_daily_normalized WHERE has_trade_price=1 "
                        f"AND source IN ({','.join('?' for _ in VERIFIED_SOURCES)})", sorted(VERIFIED_SOURCES)).fetchone()[0]
            date_status["raw_latest"] = latest
            if not latest:
                date_status["code"] = "empty_verified_raw"
                raise ValueError("本地没有合格来源的真实原始行情")
            end = end or max((day for day in calendar if day <= latest), default=None)
            date_status.update(nearest_verified_before=max((d for d in calendar if end and d < end), default=None),
                               nearest_verified_after=next((d for d in calendar if end and d > end), None))
            if end is None:
                date_status["code"] = "no_common_verified_session"
                raise ValueError("原始行情与核验日历没有共同交易日")
            if end not in calendar:
                date_status["code"] = "non_verified_session"
                raise ValueError(f"截止日 {end} 不是核验交易日；前一核验日 {date_status['nearest_verified_before']}，"
                                 f"后一核验日 {date_status['nearest_verified_after']}。请修改日期或留空重新检查")
            if end > latest:
                date_status["code"] = "beyond_raw_latest"
                raise ValueError(f"截止日 {end} 超出合格原始行情截止 {latest}")
            date_status.update(code="latest_verified_frozen" if requested_end is None else "explicit_verified_session",
                               effective_end=end)
            config = config or _configuration(values, end)
            if universe is None:
                from my_strategy.services.czsc_trend_source import build_liquidity_universe
                universe = build_liquidity_universe(feature_start=values["feature_start"], end=end,
                    calendar_run_id=calendar_id, policy=config["liquidity_universe"], preferred_symbols=values.get("symbols"))
            elif values.get("symbols") is not None and values["symbols"] != universe["selected_symbols"]:
                raise ValueError("复用来源不能用显式股票列表截断完整人口")
            cohort = universe["selected_symbols"]
            if not cohort:
                raise ValueError("历史时点流动性和完整窗口核验后没有合格股票；保留筛选诊断，不降低门槛")
            config["training_capacity_tiers"] = [len(cohort)]
            if source_reuse: date_status["code"] = "verified_frozen_source_reuse"
            if cohort and len(cohort) < 60:
                warnings.append("历史合格股票并集不足 60 股，不能自动替换；不会降低门槛扩充")
            warnings.append("可观察原始行情人口尚不等于完整历史上市/ST/退市名录；正式独立认证继续 pending")
            if values["trend_horizon_sessions"] != 10:
                warnings.append("此目标为实验配置，不赋予当前默认 10 日目标研究选择资格")
        except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error) as exc:
            errors.append(str(exc))
        values["end"] = end
        current = self.current_selection()
        plan = dict(plan_id=uuid.uuid4().hex, parameters=values, config=config,
            config_hash=stable_hash(config), cohort=cohort, calendar_run_id=calendar_id,
            calendar_hash=stable_hash([day for day in calendar if end and day <= end]), data_end=end,
            requested_end=requested_end, effective_end=date_status["effective_end"], date_status=date_status,
            liquidity_universe=universe, historical_union_count=len(cohort),
            latest_eligible_count=(universe or {}).get("latest_count", 0), stock_batch_size=values["stock_batch_size"],
            windows=(config or {}).get("development_windows", []), incumbent_run_id=(current or {}).get("run_id"),
            source_policy="historical_daily_liquidity_eligible_union", valid=not errors, errors=errors, warnings=warnings,
            profile_revision=self.profiles()["default_revision"], created_at=local_now().isoformat(),
            source_reuse=source_reuse,training_code_freeze=training_code_freeze(),
            selection_policy=dict(min_stocks=60, min_round_trips_per_window=50, min_round_trips_total=200,
                drawdown_tolerance=.02, stress_multiplier=2., publication_allowed=False))
        plan["plan_hash"] = stable_hash(plan)
        with closing(_connect(self.db_path)) as conn:
            conn.execute("INSERT INTO training_plans(plan_id,payload,created_at) VALUES (?,?,?)", (plan["plan_id"], json.dumps(plan, ensure_ascii=False), plan["created_at"]))
            conn.commit()
        return dict(valid=not errors, errors=errors, warnings=warnings, plan=plan)

    def load_plan(self, plan_id, parameters=None, *, claim=False, progress=None, check_cancel=None):
        _identifier(plan_id)
        with closing(_connect(self.db_path)) as conn:
            if claim: conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM training_plans WHERE plan_id=?", (plan_id,)).fetchone()
            if not row: raise ValueError("预检查计划不存在，请先执行预检查")
            plan = json.loads(row["payload"])
            if not plan["valid"]: raise ValueError("预检查未通过：" + "; ".join(plan["errors"]))
            if plan["plan_hash"] != stable_hash({k:v for k,v in plan.items() if k != "plan_hash"}):
                raise ValueError("冻结训练计划已改变")
            if {"cohort_size", "capacity_tiers"} & set(plan.get("parameters", {})):
                raise ValueError("旧固定数量预检已退役；请按历史流动性完整池重新检查训练计划")
            if parameters is not None:
                supplied = validate_parameters(parameters)
                if supplied["end"] is None: supplied["end"] = plan["data_end"]
                if supplied != plan["parameters"]: raise ValueError("提交参数与已冻结预检计划不同")
            if claim:
                if row["claimed"]: raise RevisionConflict("同一冻结计划已提交，不能重复训练")
                self._verify_execution_plan(plan,progress=progress,check_cancel=check_cancel)
                conn.execute("UPDATE training_plans SET claimed=1 WHERE plan_id=?", (plan_id,)); conn.commit()
        return plan

    def validate_plan(self, plan_id, parameters=None):
        plan = self.load_plan(plan_id, parameters)
        self._verify_execution_plan(plan)
        return plan

    def _verify_execution_plan(self, plan, *, progress=None, check_cancel=None):
        if plan.get("training_code_freeze") != training_code_freeze():
            raise ValueError("预检查后训练生成代码已改变；必须重新预检查，不能沿用旧身份")
        if plan.get("config_hash") != stable_hash(plan["config"]): raise ValueError("冻结计划配置哈希不同")
        reuse = plan.get("source_reuse")
        if reuse:
            proof,population,dates = self._verify_reusable_source(reuse["source_native_run_id"],plan["config"],
                plan["data_end"],expected=reuse,progress=progress,check_cancel=check_cancel)
            if (population != plan["liquidity_universe"] or population["selected_symbols"] != plan["cohort"]
                    or stable_hash(dates) != plan["calendar_hash"] or proof["calendar_run_id"] != plan["calendar_run_id"]):
                raise ValueError("执行来源人口或日历与冻结计划不同")

    def run_training_job(self, spec, progress=None, check_cancel=None, run_callback=None):
        with self._training_resource(progress, check_cancel):
            return self._execute_training_job(spec, progress, check_cancel, run_callback)

    @contextmanager
    def _training_resource(self, progress, check_cancel):
        # A separate SQLite writer lease survives multiple app processes while
        # leaving profile and plan transactions available throughout training.
        conn = sqlite3.connect(self.db_path.parent / "training_resource.db", timeout=.25)
        waiting = False
        try:
            while True:
                if check_cancel: check_cancel()
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower(): raise
                    if not waiting and progress:
                        progress("resource_wait", 0, 1, message="等待上一训练释放 CPU/GPU 与内存资源")
                    waiting = True
            yield
        finally:
            conn.rollback(); conn.close()

    def _execute_training_job(self, spec, progress=None, check_cancel=None, run_callback=None):
        callback = progress
        if callback:
            def progress(payload):
                details = {k:v for k,v in payload.items() if k not in {"stage", "completed", "total"}}
                details.update(historical_union_count=len(plan["cohort"]), latest_eligible_count=plan.get("latest_eligible_count"))
                callback(payload.get("stage", "training"), payload.get("completed", 0), payload.get("total", 0), **details)
        def verification_progress(payload):
            if callback:
                callback(payload["stage"],payload.get("completed",0),payload.get("total",0),
                    **{k:v for k,v in payload.items() if k not in {"stage","completed","total"}})
        plan = self.load_plan(spec["plan_id"], spec.get("parameters"), claim=True,
            progress=verification_progress,check_cancel=check_cancel)
        from my_strategy.services.czsc_research_calendar import load_verified_calendar
        calendar, _ = load_verified_calendar(plan["calendar_run_id"],runs_root=self.runs_root)
        if stable_hash([d for d in calendar if d <= plan["data_end"]]) != plan["calendar_hash"]:
            raise ValueError("核验日历历史发生修订，请重新预检查")
        context = create_run_context(task="web-model-training", as_of_date=plan["data_end"], config=plan,
                                     stocks=plan["cohort"], scope="train", source="web")
        if run_callback: run_callback(context.run_id)
        _save(context.subdir("config") / "training_plan.json", plan)
        config_path = context.subdir("config") / "strategy.json"; _save(config_path, plan["config"])
        cohort_path = context.subdir("config") / "cohort.json"; _save(cohort_path, {"symbols": plan["cohort"],
            "plan_hash": plan["plan_hash"], "liquidity_universe": plan.get("liquidity_universe")})
        args = [sys.executable, "-u", "-m", "my_strategy.scripts.train_trend", "--end", plan["data_end"],
            "--cohort-file", str(cohort_path), "--calendar-run-id", plan["calendar_run_id"],
            "--cpu-workers", str(plan["parameters"]["cpu_workers"]), "--device", plan["parameters"]["device"], "--config", str(config_path),
            "--stock-batch-size", str(plan["parameters"]["stock_batch_size"])]
        stage = spec.get("stage", "complete")
        reuse = plan.get("source_reuse")
        if reuse:
            args.extend(["--source-native-run-id",reuse["source_native_run_id"]])
        if stage == "source_prepare": args.append("--prepare-source-only")
        elif stage != "complete": raise ValueError("当前阶段只支持 complete 或 source_prepare")
        if reuse:
            initial_ids = dict(raw_source_run_id=reuse["raw_source_run_id"],source_native_run_id=reuse["source_native_run_id"])
            result_ids = self._run_process(args,context.subdir("logs")/"training.log",progress,check_cancel,
                initial_ids=initial_ids,source_total=len(plan["cohort"]),source_reuse=reuse)
        else:
            result_ids = self._run_process(args, context.subdir("logs") / "training.log", progress, check_cancel,
                source_total=len(plan["cohort"]))
        model_id = result_ids.get("model_run_id")
        result = dict(run_id=context.run_id, training_run_id=model_id, model_run_id=model_id,
                      children=result_ids, plan_id=plan["plan_id"], publication_allowed=False,
                      historical_union_count=len(plan["cohort"]), latest_eligible_count=plan.get("latest_eligible_count"),
                      stock_batch_size=plan["parameters"]["stock_batch_size"])
        if reuse: result.update(source_reused=True,source_reuse=copy.deepcopy(reuse),reused_model_or_label=False)
        if not model_id:
            if stage != "source_prepare": raise ValueError("训练子进程未返回可核验模型运行身份")
            result.update(status="source_prepared", selection=dict(status="not_requested"))
        else:
            path = self.runs_root / _identifier(model_id) / "reports" / "trend_consensus.json"
            report = json.loads(path.read_text())
            result["training_status"] = report.get("status")
            result["reference_quality"] = _reference_quality_summary(report)
            result["trained_stock_count"] = len(report.get("dataset_records", []))
            if check_cancel: check_cancel()
            if report.get("status") == "completed":
                try:
                    result["diagnostics"] = self._run_diagnostics(model_id,plan,context.subdir("logs")/"diagnostics.log",progress,check_cancel)
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    result["diagnostics"] = dict(status="failed", error=str(exc))
                    result.setdefault("warnings", []).append("样本流失诊断失败，请核查诊断错误和对应模型资产")
            else:
                result["diagnostics"] = dict(status="not_available", reason="容量不足，完整入场标签依赖尚未生成")
            if check_cancel: check_cancel()
            if progress: progress(dict(stage="comparison", message="核验必需头与实际费用后 Broker 比较", completed=0, total=1))
            comparison = self.evaluate_candidate(model_id, plan, progress=progress, check_cancel=check_cancel)
            if result["diagnostics"]["status"] == "failed":
                comparison.update(status="rejected", passed=False,
                    reasons=["样本与资产诊断未完成，阻断当前完整包自动替换", *comparison.get("reasons", [])])
                self._save_comparison(model_id, comparison)
            result["comparison_status"] = comparison["status"]
            selection = dict(status="disabled")
            if plan["parameters"]["auto_replace"] and comparison["passed"]:
                if check_cancel: check_cancel()
                try:
                    selection = self.select_candidate(model_id, expected_run_id=plan["incumbent_run_id"])
                except RevisionConflict as exc:
                    selection = dict(status="blocked",reasons=[str(exc)],next_action="重新核验已完成模型，按新现任重新比较；无需重训")
            elif plan["parameters"]["auto_replace"]:
                selection = dict(status="blocked", reasons=comparison["reasons"])
            result.update(status="completed" if comparison["passed"] else "completed_unqualified", selection=selection, required_heads=report.get("required_head_status", {}))
        _save(context.subdir("reports") / "training_job.json", result)
        context.write_metadata({"status": result["status"], "training_run_id": model_id})
        return self.store.save("model-training", result)

    def run_validation_job(self, spec, progress=None, check_cancel=None, run_callback=None):
        """Revalidate a completed model without changing weights or old jobs."""
        with self._training_resource(progress, check_cancel):
            original = self.load_plan(spec["plan_id"])
            model_id = _identifier(spec["model_run_id"])
            from my_strategy.services.czsc_trend_pipeline import read_run
            root, report = read_run(model_id)
            self._validate_trained_plan(report, original)
            self._evaluation_records(report, original)
            # A new validation run freezes its own incumbent before evaluating.
            # The original submitted plan and failed task remain immutable.
            plan = copy.deepcopy(original)
            plan.update(plan_id=uuid.uuid4().hex, parent_plan_id=original["plan_id"], validation_model_run_id=model_id,
                incumbent_run_id=(self.current_selection() or {}).get("run_id"), created_at=local_now().isoformat())
            plan.pop("plan_hash",None); plan["plan_hash"] = stable_hash(plan)
            with closing(_connect(self.db_path)) as conn:
                conn.execute("INSERT INTO training_plans(plan_id,payload,created_at,claimed) VALUES (?,?,?,1)",
                             (plan["plan_id"],json.dumps(plan,ensure_ascii=False),plan["created_at"])); conn.commit()
            context = create_run_context(task="web-model-validation",as_of_date=plan["data_end"],config=plan,
                                         stocks=plan["cohort"],scope="train",source="web")
            if run_callback: run_callback(context.run_id)
            _save(context.subdir("config")/"validation_plan.json",plan)
            result = dict(run_id=context.run_id,model_run_id=model_id,training_run_id=model_id,
                plan_id=plan["plan_id"],original_plan_id=original["plan_id"],weights_unchanged=True,
                training_started=False,publication_allowed=False,reference_quality=_reference_quality_summary(report))
            def emit(payload):
                if check_cancel: check_cancel()
                if progress:
                    details={k:v for k,v in payload.items() if k not in {"stage","completed","total"}}
                    progress(payload.get("stage","validation"),payload.get("completed",0),payload.get("total",0),**details)
            try:
                result["diagnostics"] = self._run_diagnostics(model_id,plan,context.subdir("logs")/"diagnostics.log",emit,check_cancel)
            except (OSError,ValueError,KeyError,RuntimeError) as exc:
                if check_cancel: check_cancel()
                result["diagnostics"] = dict(status="failed",error=str(exc))
            if check_cancel: check_cancel()
            comparison=self.evaluate_candidate(model_id,plan,progress=emit,check_cancel=check_cancel)
            if result["diagnostics"]["status"]=="failed":
                comparison.update(status="rejected", passed=False,
                    reasons=["样本与资产诊断未完成，阻断当前完整包自动替换", *comparison.get("reasons", [])])
                self._save_comparison(model_id, comparison)
            result["comparison_status"]=comparison["status"]
            selection=dict(status="disabled")
            if plan["parameters"]["auto_replace"] and comparison["passed"]:
                try: selection=self.select_candidate(model_id,expected_run_id=plan["incumbent_run_id"])
                except RevisionConflict as exc: selection=dict(status="blocked",reasons=[str(exc)])
            elif plan["parameters"]["auto_replace"]:
                selection=dict(status="blocked",reasons=comparison["reasons"])
            result.update(status="completed" if comparison["passed"] else "completed_unqualified",selection=selection,
                          required_heads=report.get("required_head_status",{}))
            _save(context.subdir("reports")/"validation_job.json",result)
            context.write_metadata(dict(status=result["status"],model_run_id=model_id,training_started=False))
            return self.store.save("model-validation",result)

    def _run_diagnostics(self,model_id,plan,log_path,progress,check_cancel):
        identities=self._run_process([sys.executable,"-u","-m","my_strategy.scripts.model_diagnostics",
            "--model-run-id",model_id,"--cpu-workers",str(min(4,plan["parameters"]["cpu_workers"]))],
            log_path,progress,check_cancel,source_total=len(plan["cohort"]))
        identity=identities.get("diagnostic_run_id")
        if not identity: raise ValueError("诊断子进程未返回实际运行身份")
        path=self.runs_root/_identifier(identity)/"reports/high_return_sample_diagnostics.json"
        report=json.loads(path.read_text())
        if report.get("model_run_id")!=model_id or report.get("artifact_integrity_passed") is not True or report.get("source_report_unchanged") is not True:
            raise ValueError("诊断资产完整性或模型来源不合格")
        return dict(status="completed",run_id=identity,report_paths=report["report_paths"],report_sha256=_file_hash(path))

    @staticmethod
    def _validate_trained_plan(report,plan):
        expected,actual=plan["config"],report["config"]
        if report.get("strategy_version")!=STRATEGY_ID or report.get("status")!="completed" or report.get("data_end")!=plan["data_end"]:
            raise ValueError("重新核验须为已完成的当前训练，且截止日与原冻结计划相同")
        allowed_derived={"cpu_workers","training_end","source_snapshot_hash","pipeline_contract_hash","candidate_policy_hash","E0_policy_hash","E1_policy_hash","capacity_expansion_criterion","training_capacity_tiers"}
        if set(actual)-set(expected)-allowed_derived:
            raise ValueError("模型配置出现未登记派生字段")
        for key,value in expected.items():
            measured=copy.deepcopy(actual.get(key))
            if key=="trend" and measured:
                selected=measured.get("candidate_cross_threshold")
                if selected not in value["candidate_cross_grid"]: raise ValueError("趋势候选阈值不在冻结搜索范围")
                measured["candidate_cross_threshold"]=value["candidate_cross_threshold"]
            if key=="consensus" and measured:
                for policy_key in ("buy_base_threshold","sell_threshold","margin"):
                    if measured.get(policy_key)!=value.get(policy_key) and measured.get(policy_key) not in value["policy_grid"][policy_key]:
                        raise ValueError("共识政策不在冻结搜索范围")
                    measured[policy_key]=value[policy_key]
            if measured!=value: raise ValueError("模型配置与冻结计划不同："+key)
        if actual.get("cpu_workers")!=plan["parameters"]["cpu_workers"] or actual.get("training_end")!=plan["data_end"]:
            raise ValueError("实际训练进程数或截止日与原计划不同")
        criterion = ("all_frozen_historical_liquidity_qualified_symbols_no_readiness_early_stop"
                     if expected.get("full_qualified_pool") else
                     "missing_recent_qualified_three_class_MA_exit_or_production_L2_entry_head_only")
        if actual.get("capacity_expansion_criterion") != criterion:
            raise ValueError("容量扩展准则与当前登记规则不同")
        if expected.get("full_qualified_pool") and (
                actual.get("training_capacity_tiers") != [len(plan["cohort"])] or
                len(report.get("dataset_records", [])) != len(plan["cohort"]) or
                {r["symbol"] for r in report.get("dataset_records", [])} != set(plan["cohort"])):
            raise ValueError("完整训练未覆盖全部冻结历史合格股票")
        source=report["source_snapshot"]
        reuse=plan.get("source_reuse")
        if reuse:
            raw=source.get("source_snapshot") or {}
            population=source.get("liquidity") or {}
            if (source.get("source_native_run_id")!=reuse["source_native_run_id"]
                    or source.get("source_report_sha256")!=reuse["native_report_sha256"]
                    or source.get("records_hash")!=reuse["records_hash"]
                    or raw.get("raw_source_run_id")!=reuse["raw_source_run_id"]
                    or raw.get("raw_snapshot_hash")!=reuse["raw_snapshot_hash"]
                    or raw.get("calendar_run_id")!=reuse["calendar_run_id"]
                    or raw.get("calendar_hash")!=reuse["calendar_hash"]
                    or population.get("population_hash")!=reuse["population_hash"]
                    or population.get("daily_membership_hash")!=reuse["daily_membership_hash"]
                    or population.get("policy_hash")!=reuse["policy_hash"]
                    or stable_hash(source.get("selected_symbols"))!=reuse["selected_symbols_hash"]):
                raise ValueError("新训练实际来源与预登记复用身份或报告哈希不同")
        if (plan.get("training_code_freeze") and report.get("protocol_freeze",{}).get("code_hashes")
                !=plan["training_code_freeze"]["code_hashes"]):
            raise ValueError("新训练实际生成代码与新冻结计划不同")
        if actual.get("source_snapshot_hash")!=source["snapshot_hash"]:
            raise ValueError("实际来源哈希未绑定冻结来源")
        registered=copy.deepcopy(expected)
        registered.update(cpu_workers=actual["cpu_workers"],training_end=actual["training_end"],
            capacity_expansion_criterion=actual["capacity_expansion_criterion"],source_snapshot_hash=source["snapshot_hash"])
        if "training_capacity_tiers" not in registered and "training_capacity_tiers" in actual:
            registered["training_capacity_tiers"] = actual["training_capacity_tiers"]
        from my_strategy.services.czsc_trend_pipeline import POPULATION_VERSION
        pipeline=stable_hash(dict(family=STRATEGY_ID,source=source,calendar_hash=source["source_snapshot"]["calendar_hash"],registered_config=registered,
            order=["independent_trend","policy_only_sources","E0_no_HR","MA_exit_E0","pinned_E1",
                "preregistered_actual_E1_holding_pairs","account_free_reference_quality_probes",
                "independent_reference_quality_heads_and_fee_qualification","strict_prior_L2_actual_flat_population_by_fold",
                "L2_entry_forced_E1_clone","L2_exit_preregistered_actual_E1_pairs"],
            reference_population_protocol=POPULATION_VERSION,self_training_population_feedback=False,reassessment=registered["reassessment"],no_E2_entry_reuse=True))
        candidate=stable_hash(dict(candidate=actual["candidate_policy"],risk_policy=actual["risk_policy"],
            trend_candidate={k:actual["trend"][k] for k in ("candidate_cross_threshold","candidate_margin")},
            consensus=actual["consensus"],signal_points=actual["signal_points"],native_records=source["records_hash"],
            reassessment=actual["reassessment"],training_population=POPULATION_VERSION))
        bootstrap=stable_hash(dict(stage="E0",candidate_policy_hash=candidate,account=actual["account"],
            trend_catalog_hash=stable_hash(report["trend_bundles"]),ordinary_sell="continuous_sell_pass_for_actual_holding",reassessment=actual["reassessment"],entry_HR=False))
        continuation=stable_hash(dict(stage="E1",E0=bootstrap,signal_time_pinned_MA_exit=stable_hash(report["ma_exit_bundles"]),
            holding_threshold=actual["account"]["holding_exit_threshold"],no_head="entire_position_E0",checkpoint_upgrade=False,L2_exit_apply=False))
        if any(actual.get(key)!=value for key,value in (("pipeline_contract_hash",pipeline),("candidate_policy_hash",candidate),("E0_policy_hash",bootstrap),("E1_policy_hash",continuation))):
            raise ValueError("派生训练政策哈希与真实当前来源、配置和模型目录不同")
        from my_strategy.services.czsc_trend_contracts import validate_holding_population
        held=validate_holding_population(report.get("holding_population"))
        if (held["source_snapshot_hash"]!=source["snapshot_hash"]
                or held["candidate_policy_hash"]!=candidate or held["continuation_policy_hash"]!=continuation
                or report.get("protocol_freeze",{}).get("holding_population_hash")!=stable_hash(held)):
            raise ValueError("独立E1实际持仓退出人口未绑定当前完整训练来源和冻结续持")

    @staticmethod
    def _run_process(args, log_path, progress, check_cancel, *, initial_ids=None, source_total=0, source_reuse=None):
        windows_job = None
        process_options = (dict(creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
                           if _is_windows() else dict(start_new_session=True))
        options = dict(cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace", **process_options)
        if _is_windows():
            process, windows_job = _start_windows_process(args, **options)
        else:
            process = subprocess.Popen(args, **options)
        reader = None
        try:
            messages = queue.Queue()
            def collect():
                for line in process.stdout: messages.put(line)
                messages.put(None)
            reader = threading.Thread(target=collect, daemon=True); reader.start()
            result_ids = dict(initial_ids or {}); done = False; population_progress = {}; diagnostic_progress = {}; reference_progress = {}
            last_progress={}
            def publish_progress(payload):
                last_progress.clear();last_progress.update(copy.deepcopy(payload))
                if progress:progress(payload)
            with log_path.open("w", encoding="utf-8") as handle:
                if source_reuse:
                    event=dict(source_native_run_id=source_reuse["source_native_run_id"],
                        raw_source_run_id=source_reuse["raw_source_run_id"],
                        native_report_sha256=source_reuse["native_report_sha256"],
                        population_hash=source_reuse["population_hash"],selected_count=source_total,
                        source_reused=True,reused_model_or_label=False)
                    handle.write("SOURCE_REUSE_VERIFIED "+json.dumps(event,ensure_ascii=False)+"\n"); handle.flush()
                    if progress: publish_progress(dict(stage="frozen_source_verification",completed=source_total,total=source_total,
                        message="复用已实际核验的完整冻结输入，所有模型头在新运行重新训练",children=copy.deepcopy(result_ids),source_reused=True))
                while not done:
                    if check_cancel: check_cancel()
                    try: line = messages.get(timeout=.25)
                    except queue.Empty: continue
                    if line is None: done=True; continue
                    handle.write(line); handle.flush(); text = line.strip()
                    identity_changed=False
                    for prefix, key in (("RAW_SOURCE_RUN_ID ", "raw_source_run_id"), ("NATIVE_SOURCE_RUN_ID ", "source_native_run_id"), ("TRAIN_RUN_ID ", "model_run_id"), ("COMPLETED ", "model_run_id"), ("CAPACITY_PROBE ", "model_run_id"), ("DIAGNOSTIC_RUN_ID ","diagnostic_run_id")):
                        if text.startswith(prefix):
                            result_ids[key] = text[len(prefix):].split()[0];identity_changed=True
                    dataset_event=None
                    if text.startswith("PROGRESS "):
                        event=json.loads(text[len("PROGRESS "):])
                        population=event.get("population");fold=event.get("fold")
                        if (event.get("stage")!="dataset" or event.get("scope")!="current_population_only"
                                or type(event.get("completed")) is not int or type(event.get("total")) is not int
                                or event["total"]!=source_total or not 0<=event["completed"]<=source_total
                                or not isinstance(population,str)
                                or not (population in {"E0","E1"} and fold is None or
                                    isinstance(fold,str) and fold and population=="final_population/"+fold)
                                or event["completed"]<population_progress.get(population,0)):
                            raise ValueError("人口进度与冻结股票总数、单折身份或已完成数量不同")
                        population_progress[population]=event["completed"]
                        dataset_event=dict(stage=population,population=population,fold=fold,
                            completed=event["completed"],total=event["total"],scope=event["scope"],
                            message=text[-500:],children=copy.deepcopy(result_ids))
                    diagnostic_event=None
                    if text.startswith("DIAGNOSTIC_PROGRESS "):
                        event=json.loads(text[len("DIAGNOSTIC_PROGRESS "):]);fold=event.get("fold")
                        if (event.get("stage")!="diagnostics" or event.get("scope")!="current_diagnostic_fold_stocks"
                                or not isinstance(fold,str) or not fold
                                or type(event.get("completed")) is not int or type(event.get("total")) is not int
                                or event["total"]!=source_total or not 0<=event["completed"]<=source_total
                                or event["completed"]<diagnostic_progress.get(fold,0)):
                            raise ValueError("诊断进度与冻结单折股票人口、scope或完成数量不同")
                        diagnostic_progress[fold]=event["completed"]
                        diagnostic_event={**event,"children":copy.deepcopy(result_ids)}
                    reference_event=None
                    if text.startswith("REFERENCE_QUALITY_PROGRESS "):
                        event=json.loads(text[len("REFERENCE_QUALITY_PROGRESS "):]);stage=event.get("stage")
                        if (stage not in {"reference_quality_population", "reference_quality_fee_qualification"}
                                or event.get("scope")!="current_reference_population_stocks" or event.get("fold") is not None
                                or type(event.get("completed")) is not int or type(event.get("total")) is not int
                                or event["total"]!=source_total or not 0<=event["completed"]<=source_total
                                or event["completed"]<reference_progress.get(stage,0)):
                            raise ValueError("参考质量进度与冻结股票人口、scope或完成数量不同")
                        reference_progress[stage]=event["completed"]
                        reference_event={**event,"children":copy.deepcopy(result_ids),"actual_device":"cpu","device":"cpu"}
                    if progress:
                        if reference_event is not None:
                            publish_progress(reference_event)
                        elif dataset_event is not None:
                            publish_progress(dataset_event)
                        elif diagnostic_event is not None:
                            publish_progress(diagnostic_event)
                        elif text.startswith("POLICY_STOCK_DONE "):
                            event=json.loads(text[len("POLICY_STOCK_DONE "):])
                            if (event.get("total")!=source_total or type(event.get("current")) is not int
                                    or not 0<=event["current"]<=source_total or event.get("device")!="cpu"):
                                raise ValueError("政策完成事件与冻结股票人口或实际CPU设备不同")
                            publish_progress(dict(stage="independent_policy_selection",completed=event["current"],total=event["total"],
                                failed=event.get("failed",0),actual_device=event["device"],device=event["device"],message=text[-500:],
                                children=copy.deepcopy(result_ids),symbol=event.get("symbol"),fold=event.get("fold")))
                        elif text.startswith("HEAD "):
                            event=json.loads(text[len("HEAD "):]); actual=event.get("device") or "unknown"
                            publish_progress(dict(stage="model_head",completed=0,total=0,actual_device=actual,
                                device=actual,
                                head=event.get("head"),fold=event.get("fold"),architecture=event.get("architecture"),
                                training_resources=event.get("training_resources"),message=text[-500:],children=copy.deepcopy(result_ids)))
                        elif text.startswith("SOURCE_BATCH "):
                            batch = json.loads(text[len("SOURCE_BATCH "):]); source_total = batch["total"]
                            publish_progress(dict(stage="raw_source_freeze" if batch["stage"] == "raw" else "native_source_prepare",
                                completed=batch["completed"], total=batch["total"], message=text[-500:], children=copy.deepcopy(result_ids)))
                        elif text.startswith("DATASET "):
                            # Per-stock detail is retained in the log. Only the
                            # parent emits complete single-population progress.
                            pass
                        elif text.startswith("STAGE "):
                            publish_progress(dict(stage=text.split()[1],completed=0,total=0,
                                message=text[-500:],children=copy.deepcopy(result_ids)))
                        elif identity_changed:
                            payload=copy.deepcopy(last_progress) or dict(stage="training",completed=0,total=0)
                            payload.update(message=text[-500:],children=copy.deepcopy(result_ids))
                            publish_progress(payload)
            code = process.wait()
            if code: raise ValueError(f"训练阶段失败，退出码 {code}；日志：{log_path}")
        except BaseException:
            if _is_windows():
                raise
            import psutil
            try: descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.Error: descendants = []
            try: os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                for child in descendants:
                    try: child.terminate()
                    except psutil.Error: pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                process.poll()
                living = []
                for child in descendants:
                    try:
                        if child.is_running() and child.status() != psutil.STATUS_ZOMBIE: living.append(child)
                    except psutil.Error: pass
                if process.returncode is not None and not living: break
                threading.Event().wait(.05)
            # The parent may exit before a child that ignores TERM. Always
            # clean the entire isolated process group, not just the parent.
            try: os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError): pass
            for child in descendants:
                try: child.kill()
                except psutil.Error: pass
            if process.poll() is None: process.kill()
            process.wait(timeout=5)
            # SIGKILL delivery is asynchronous. Do not report cancellation
            # complete until owned descendants have actually stopped running.
            _, alive = psutil.wait_procs(descendants, timeout=5)
            live = []
            for child in alive:
                try:
                    if child.status() != psutil.STATUS_ZOMBIE: live.append(child)
                except psutil.Error: pass
            if live:
                for child in live:
                    try: child.kill()
                    except psutil.Error: pass
                psutil.wait_procs(live, timeout=5)
            raise
        finally:
            if windows_job is not None:
                windows_job.stop(process)
                if reader is not None and reader.ident is not None:
                    reader.join(timeout=5)
                    if reader.is_alive():
                        raise RuntimeError("训练自有进程已停止但日志管道仍未释放")
                process.stdout.close()
        return result_ids

    def _model(self, run_id):
        from my_strategy.services.czsc_trend_pipeline import read_run, required_head_status
        from my_strategy.services.czsc_trend_model import chronological_folds
        root, report = read_run(run_id)
        if root is None or report is None: raise ValueError("当前模型包不可用")
        records=report["dataset_records"]
        if len({r["symbol"] for r in records})!=len(records):
            raise ValueError("当前训练股票账户重复")
        if len(records)<60:
            raise ModelQualificationError("当前完整包训练覆盖不足 60 个股票账户")
        basis = (report.get("source_snapshot", {}).get("liquidity") or {}).get("basis_mode")
        if basis is not None and basis != "warehouse_observed_mainboard":
            raise ModelQualificationError("显式股票子集只能用于研究验证，不能代替完整可观察主板池自动选模")
        calendar=json.loads((root/"data_snapshots/calendar.json").read_text())["dates"]
        required=required_head_status(chronological_folds(report["data_end"],calendar),report["trend_bundles"],
            report["ma_exit_bundles"],report["model_bundles"],report["config"]["production"]["name"])
        if report.get("required_head_status") is not None and report["required_head_status"]!=required:
            raise ValueError("自报必需头状态与已冻结实际模型目录不同")
        if not all(required[key] for key in ("recent_trend_ready","same_checkpoint_ma_exit_ready","production_entry_ready")):
            raise ModelQualificationError("当前完整包实际必需头、类别或覆盖资格不足")
        if report["config"]["trend"]["horizon_sessions"] != 10:
            raise ModelQualificationError("实验目标不能替换当前 10 日趋势体系")
        if report["config"]["risk_policy"]!="confirmed_source_boundary_prior_atr":
            raise ValueError("当前模型实际风险契约不匹配")
        self._audit_required_heads(root,report,required,calendar)
        return root, report

    @staticmethod
    def _audit_required_heads(root,report,required,calendar):
        from my_strategy.services.czsc_trend_pipeline import read_snapshot
        from my_strategy.services.czsc_trend_model import load_trend_head,build_trend_labels,trend_time_masks,CLASSES
        from my_strategy.services.czsc_trend_interval_model import load_head,bind_inputs,time_masks
        from my_strategy.services.czsc_trend_finalization import audit_stage_assets
        config=report["config"]
        checkpoint=required["recent_checkpoint"]
        trend=next(b for b in report["trend_bundles"] if b["name"]==checkpoint)
        manifest=load_trend_head(root/trend["heads"]["trend_direction"]["path"])
        actual_counts={segment:{name:0 for name in CLASSES} for segment in ("train","calibration","policy")}
        for record in report["dataset_records"]:
            frame,_=read_snapshot(record); frame=frame.loc[frame.date<=report["data_end"]]
            labels=build_trend_labels(frame,calendar,config,source_snapshot_hash=config["source_snapshot_hash"],pipeline_contract_hash=config["pipeline_contract_hash"])
            data=labels.loc[labels.date>=config["feature_start"]].drop_duplicates("observation_id").reset_index(drop=True)
            # Direction masks depend only on the observation and its label end;
            # symbol/day identities cannot collide across unique stock records.
            for segment,mask in zip(("train","calibration","policy"),trend_time_masks(data,manifest["fold"]),strict=True):
                for index,name in enumerate(CLASSES):
                    actual_counts[segment][name]+=int(data.loc[mask,"label"].eq(index).sum())
        for segment,counts in actual_counts.items():
            if counts!=manifest["sample_counts"][segment] or min(counts.values())<config["trend"]["min_"+segment+"_per_class"]:
                raise ValueError("当前三分类趋势实际成熟类别不足或与模型资产不一致")
        expected_symbols=[r["symbol"] for r in report["dataset_records"]]
        ma=next(b for b in report["ma_exit_bundles"] if b["name"]==checkpoint)
        entry=next(b for b in report["model_bundles"] if b["name"]==config["production"]["name"])
        for bundle,head,stage,key in ((ma,"ma_exit","E0","exits"),(entry,"entry","final_population/"+entry["name"],"entries")):
            audit_stage_assets(root,stage,expected_symbols=expected_symbols)
            m=load_head(root/bundle["heads"][head]["path"])
            stage_manifest=json.loads((root/"dataset"/stage/"manifest.json").read_text())
            data=pd.read_parquet(stage_manifest["pooled_files"][key]["path"])
            data.attrs["label_contract"]=m["target_contract"]
            data=bind_inputs(data,m["columns"],head)
            train,calibration,_=time_masks(data,m["fold"],head=head)
            for segment,mask,minimum in (("train",train,config["min_train_rows"]),("calibration",calibration,config["min_calibration_rows"])):
                if int(mask.sum())!=m["sample_counts"][segment] or int(mask.sum())<minimum or set(pd.to_numeric(data.loc[mask,"label"]).unique())!={0.,1.}:
                    raise ValueError("当前 "+head+" 实际成熟行数或两类资格不足")

    def _evaluation_records(self, report, plan):
        source = report["source_snapshot"]
        source_root = self.runs_root / _identifier(source["source_native_run_id"])
        path = source_root / "reports" / "native_source.json"
        if _file_hash(path) != source["source_report_sha256"]:
            raise ValueError("完整评估股票池原生来源报告改变")
        native = json.loads(path.read_text())
        records = native["dataset_records"]
        if native.get("strategy_version") != STRATEGY_ID or native.get("raw_only_rebuild") is not True or native.get("reused_model_or_label") is not False:
            raise ValueError("比较需要当前从原始行情重建的实际完整来源")
        if records != report.get("validation_records") or len(records) != len(plan["cohort"]) or {r["symbol"] for r in records} != set(plan["cohort"]):
            raise ValueError("比较完整股票池与预登记覆盖池不同")
        if native.get("data_end") != plan["data_end"]:
            raise ValueError("完整比较来源截止日与冻结计划不同")
        if report.get("config", {}).get("full_qualified_pool"):
            population = native.get("liquidity") or {}
            planned = plan.get("liquidity_universe") or {}
            if (native.get("population_mode") != "full_liquidity_union" or not planned.get("population_hash") or
                    population.get("population_hash") != planned["population_hash"] or
                    population.get("selected_symbols") != sorted(plan["cohort"]) or
                    population.get("policy") != report["config"]["liquidity_universe"]):
                raise ValueError("比较历史流动性全池或冻结政策不同")
        from my_strategy.services.czsc_trend_pipeline import read_snapshot
        for record in records:
            if native["source_snapshot"]["raw_input_hashes"].get(record["symbol"]) != record.get("raw_sha256") or _file_hash(resolve_frozen_asset(record["raw_path"],runs_root=self.runs_root)) != record["raw_sha256"]:
                raise ValueError("比较原始行情资产改变")
            read_snapshot(record)
        return records

    def evaluate_candidate(self, run_id, plan, *, progress=None, check_cancel=None):
        result = dict(run_id=run_id, plan_id=plan["plan_id"], plan_hash=plan["plan_hash"], incumbent_run_id=plan["incumbent_run_id"],
                      status="pending", passed=False, reasons=[], windows=[], publication_allowed=False,
                      evaluation_freeze=evaluation_freeze(),
                      certification=dict(status="pending", publication_allowed=False))
        try:
            from my_strategy.services.czsc_trend_pipeline import read_run
            root, report = read_run(run_id)
            if root is None or report is None:
                raise ValueError("当前模型资产不可用，不能生成模型账本")
            self._validate_trained_plan(report, plan)
            qualification_reasons = []
            try:
                self._model(run_id)
            except ModelQualificationError as exc:
                qualification_reasons.append(str(exc))
            result["model_qualification"] = dict(passed=not qualification_reasons,
                reasons=qualification_reasons, required_heads=report.get("required_head_status", {}))
            records = self._evaluation_records(report, plan)
            if report["data_end"] != plan["data_end"]: raise ValueError("候选实际截止日与冻结计划不同")
            incumbent = self._model(plan["incumbent_run_id"]) if plan["incumbent_run_id"] else None
            if incumbent and any(incumbent[1]["config"]["account"][key] != report["config"]["account"][key] for key in ("initial_cash", "commission", "min_fee", "stamp", "slippage", "stop_loss", "max_holding_sessions")):
                raise ValueError("现任与候选费用、风险或账户语义不同，不能直接比较并替换")
            if incumbent and incumbent[1]["config"].get("liquidity_universe") != report["config"].get("liquidity_universe"):
                raise ValueError("现任与候选历史流动性政策不同；不能修改冻结现任的资格后自动比较并替换")
            from my_strategy.services.czsc_trend_pipeline import read_snapshot
            calendar = json.loads((root / "data_snapshots" / "calendar.json").read_text())["dates"]
            result["candidate_report_sha256"] = _file_hash(root / "reports" / "trend_consensus.json")
            result["incumbent_report_sha256"] = _file_hash(incumbent[0] / "reports" / "trend_consensus.json") if incumbent else None
            total = len(plan["windows"]) * len(plan["cohort"]) * (3 if incumbent else 2) * 2
            completed = 0
            assets_root = root / "validation" / "model_comparison" / plan["plan_id"]
            models = [("candidate", (str(root), report)), ("rules", None)]
            if incumbent: models.append(("incumbent", (str(incumbent[0]), incumbent[1])))
            specification = dict(root=str(root), assets_root=str(assets_root), calendar=calendar,
                                 config=report["config"], windows=plan["windows"], models=models)
            staged = [{"window": window, "arms": {}} for window in plan["windows"]]
            workers = min(plan["parameters"]["cpu_workers"], len(records))
            pool = None
            all_outputs_completed = False
            try:
                if workers > 1:
                    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                               initializer=_initialize_comparison_worker, initargs=(specification,))
                    # Bound outstanding work and keep one stock's native prefix
                    # shared across all windows/arms in each worker.
                    iterator = iter(records)
                    pending = []
                    for _ in range(min(workers * 2, len(records))):
                        pending.append(pool.submit(_comparison_worker, next(iterator)))
                    def outputs():
                        while pending:
                            if check_cancel: check_cancel()
                            future = pending.pop(0)
                            while not future.done():
                                if check_cancel: check_cancel()
                                try: future.result(timeout=.25)
                                except TimeoutError: continue
                            value = future.result()
                            record = next(iterator, None)
                            if record is not None: pending.append(pool.submit(_comparison_worker, record))
                            yield value
                    measured = outputs()
                else:
                    measured = (_compare_stock(record, specification) for record in records)
                for stock in measured:
                    if check_cancel: check_cancel()
                    for index, measured_window in enumerate(stock["windows"]):
                        for key, item in measured_window.items():
                            arm = staged[index]["arms"].setdefault(key, {"accounts": [], "assets": []})
                            arm["accounts"].append(item["metrics"]); arm["assets"].append(item["asset"])
                            completed += 1
                    if progress: progress(dict(stage="comparison", completed=completed, total=total,
                        message=f"真实Broker全部窗口与费用压力：{stock['symbol']}"))
                for window in staged:
                    for arm in window["arms"].values():
                        arm["metrics"] = aggregate_account_metrics(arm.pop("accounts"))
                result["windows"] = staged
                all_outputs_completed = True
            finally:
                if pool:
                    # Cancellation must not leave native/account subprocesses
                    # consuming resources after their owning task has stopped.
                    if not all_outputs_completed:
                        for process in list((getattr(pool, "_processes", None) or {}).values()): process.terminate()
                    pool.shutdown(wait=True, cancel_futures=True)
            result.update(selection_gate(result["windows"], plan["selection_policy"], len(plan["cohort"]), bool(incumbent)))
            if qualification_reasons:
                result["passed"] = False
                result["pending"] = True
                result["reasons"] = qualification_reasons + result["reasons"]
                result["diagnostic_replay_only"] = True
            result["status"] = "passed" if result["passed"] else "pending" if result["pending"] else "rejected"
            verify_evaluation_freeze(result)
        except (OSError, ValueError, KeyError) as exc:
            result["reasons"].append(str(exc))
        return self._save_comparison(run_id, result)

    def _save_comparison(self, run_id, result):
        result.setdefault("evaluation_freeze", evaluation_freeze())
        root = self.runs_root / _identifier(run_id)
        identity = result.get("plan_id") or uuid.uuid4().hex
        path = root / "validation" / "model_comparison" / _identifier(identity) / "comparison.json"
        if path.exists():
            if json.loads(path.read_text()) != result: raise ValueError("已有冻结比较不可覆盖，请重新预检与比较")
        else: _save(path, result)
        with closing(_connect(self.registry_path)) as conn:
            conn.execute("INSERT INTO current_model_comparisons VALUES (?,?,?) ON CONFLICT(run_id) DO UPDATE SET report_path=excluded.report_path,sha256=excluded.sha256", (run_id, path.relative_to(self.runs_root).as_posix(), _file_hash(path)))
            conn.commit()
        return result

    def comparison(self, run_id):
        with closing(_connect(self.registry_path)) as conn:
            row = conn.execute("SELECT * FROM current_model_comparisons WHERE run_id=?", (_identifier(run_id),)).fetchone()
        if not row: raise KeyError("此模型尚无当前比较证据")
        path = (self.runs_root / row["report_path"]).resolve()
        if not path.is_relative_to(self.runs_root.resolve()) or _file_hash(path) != row["sha256"]:
            raise ValueError("比较证据哈希不匹配")
        return json.loads(path.read_text())

    def _verify_comparison(self, run_id):
        result = self.comparison(run_id); plan = self.load_plan(result["plan_id"])
        verify_evaluation_freeze(result)
        root, report = self._model(run_id)
        if result["candidate_report_sha256"] != _file_hash(root / "reports" / "trend_consensus.json"):
            raise ValueError("比较后候选模型报告改变")
        if result["plan_hash"] != plan["plan_hash"]: raise ValueError("比较未绑定冻结计划")
        if plan["incumbent_run_id"]:
            incumbent_root, _ = self._model(plan["incumbent_run_id"])
            if result["incumbent_report_sha256"] != _file_hash(incumbent_root / "reports" / "trend_consensus.json"):
                raise ValueError("比较后现任模型报告改变，须重新比较")
        from my_strategy.services.czsc_trend_support import _calendar
        calendar, _ = _calendar(plan["calendar_run_id"])
        calendar = [d for d in calendar if d <= plan["data_end"]]
        windows = copy.deepcopy(result["windows"])
        if [w["window"] for w in windows] != plan["windows"]:
            raise ValueError("比较窗口与提交前冻结计划不同")
        expected_arms = {"candidate", "rules", "candidate_stress", "rules_stress"}
        if plan["incumbent_run_id"]: expected_arms |= {"incumbent", "incumbent_stress"}
        records = self._evaluation_records(report, plan)
        source_hashes = {record["symbol"]:record["sha256"] for record in records}
        for window in windows:
            if set(window["arms"]) != expected_arms:
                raise ValueError("比较臂不完整或含未登记对照")
            expected_dates = [d for d in calendar if window["window"]["test_start"] <= d <= window["window"]["test_end"]]
            for name, arm in window["arms"].items():
                metrics = []
                if len(arm["assets"]) != len(plan["cohort"]): raise ValueError("比较股票池不完整")
                symbols = []
                for asset in arm["assets"]:
                    path = (root / asset["path"]).resolve(); ledger = (root / asset["ledger_path"]).resolve()
                    if not path.is_relative_to(root.resolve()) or not ledger.is_relative_to(root.resolve()): raise ValueError("比较路径越界")
                    if _file_hash(path) != asset["sha256"] or _file_hash(ledger) != asset["ledger_sha256"]: raise ValueError("Broker 账本证据改变")
                    from my_strategy.services.czsc_account_artifacts import read_account_artifact
                    account = read_account_artifact(path, expected_sha256=asset["sha256"]); context = account["series_context"]
                    symbols.append(account["symbol"])
                    if context.get("source_hash") != source_hashes.get(account["symbol"]) or context.get("calendar_hash") != stable_hash(calendar):
                        raise ValueError("Broker 账本来源或日历与冻结输入不同")
                    if context.get("account_start") != window["window"]["test_start"] or [r["date"] for r in account["daily"]] != expected_dates:
                        raise ValueError("Broker 账本窗口覆盖不同")
                    arm_run = run_id if name.startswith("candidate") else plan["incumbent_run_id"] if name.startswith("incumbent") else None
                    if (context.get("model_identity") or {}).get("model_run_id") != arm_run or context.get("mode") != ("E0" if name.startswith("rules") else "final"):
                        raise ValueError("Broker 账本模型身份或实际执行用途不同")
                    multiplier = 2. if name.endswith("_stress") else 1.
                    cfg = report["config"]["account"]
                    if context.get("initial_cash") != cfg["initial_cash"]:
                        raise ValueError("比较账户本金与冻结条件不同")
                    expected_costs = {"commission":cfg["commission"]*multiplier,"min_commission":cfg["min_fee"]*multiplier,
                                      "stamp_tax":cfg["stamp"]*multiplier,"slippage":cfg["slippage"]*multiplier}
                    if any(account["cost_model"].get(key) != value for key,value in expected_costs.items()):
                        raise ValueError("比较费用或压力倍率与冻结条件不同")
                    metrics.append(audit_account(account, calendar))
                if len(set(symbols)) != len(symbols) or set(symbols) != set(plan["cohort"]):
                    raise ValueError("Broker 账本股票重复或与冻结股票池不同")
                measured = aggregate_account_metrics(metrics)
                if measured != arm["metrics"]: raise ValueError("比较指标与费用后账本不一致")
        gate = selection_gate(windows, plan["selection_policy"], len(plan["cohort"]), bool(plan["incumbent_run_id"]))
        if not gate["passed"] or not result["passed"]: raise ValueError("当前实际 Broker 比较未通过：" + "; ".join(gate["reasons"]))
        return result, plan, report

    def _selection_reasons(self):
        return list(self.selection_guard() or []) if self.selection_guard else []

    def select_candidate(self, run_id, expected_run_id=None):
        result, plan, report = self._verify_comparison(run_id)
        if expected_run_id != result["incumbent_run_id"]: raise RevisionConflict("调用现任与比较现任不同")
        reasons = self._selection_reasons()
        if reasons: return dict(status="blocked", reasons=reasons)
        from my_strategy.services.czsc_trend_support import _calendar
        calendar, _ = _calendar(plan["calendar_run_id"])
        earliest = max(str(local_now().date()), str(pd.Timestamp(report["completed_at"]).date()))
        effective = next((day for day in calendar if day > earliest), None)
        payload = dict(run_id=run_id, previous_run_id=expected_run_id, comparison_sha256=stable_hash(result),
            comparison_file_sha256=_file_hash(self.runs_root/run_id/"validation/model_comparison"/plan["plan_id"]/"comparison.json"),
            plan_id=plan["plan_id"],plan_hash=plan["plan_hash"], calendar_run_id=plan["calendar_run_id"], effective_from=effective, recorded_at=local_now().isoformat(),
            publication_allowed=False, certification_status="pending", scope="research_model_selection")
        with closing(_connect(self.registry_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT run_id FROM current_model_selection WHERE id=1").fetchone()
            if (current[0] if current else None) != expected_run_id:
                raise RevisionConflict("现任已变化，候选须针对新现任重新比较")
            reasons=self._selection_reasons()
            if reasons: return dict(status="blocked",reasons=reasons)
            event_id = conn.execute("INSERT INTO current_model_events(kind,payload,created_at) VALUES ('select',?,?)", (json.dumps(payload, ensure_ascii=False), payload["recorded_at"])).lastrowid
            conn.execute("INSERT INTO current_model_selection VALUES (1,?,?,?) ON CONFLICT(id) DO UPDATE SET run_id=excluded.run_id,event_id=excluded.event_id,effective_from=excluded.effective_from", (run_id, event_id, effective))
            conn.commit()
        return dict(status="selected" if effective else "awaiting_next_verified_session", event_id=event_id, **payload)

    def rollback(self, expected_run_id, target_run_id=None):
        proof=None
        if target_run_id:
            if target_run_id == expected_run_id:
                raise ValueError("回退目标与当前引用相同；不能用回退提前激活待生效模型")
            self._model(target_run_id)
            today=str(local_now().date())
            eligible=[item for item in self.selection_events(200)["items"] if item["kind"] in {"select","activation"}
                and item["payload"].get("run_id")==target_run_id and item["payload"].get("effective_from")
                and item["payload"]["effective_from"] <= today]
            if not eligible:
                raise ValueError("回退包必须曾通过当前选择事务并已到核验生效日；待生效模型不能提前激活")
            event=eligible[0]
            proof=self._event_proof(event["event_id"])
        reasons = self._selection_reasons()
        if reasons: return dict(status="blocked", reasons=reasons)
        effective = str(local_now().date()) if target_run_id else None
        payload = dict(**(proof or {}))
        payload.update(run_id=target_run_id, previous_run_id=expected_run_id, effective_from=effective,
            recorded_at=local_now().isoformat(), publication_allowed=False, scope="research_model_selection")
        with closing(_connect(self.registry_path)) as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT run_id FROM current_model_selection WHERE id=1").fetchone()
            if (current[0] if current else None) != expected_run_id: raise RevisionConflict("当前选择改变，请刷新后回滚")
            reasons=self._selection_reasons()
            if reasons: return dict(status="blocked",reasons=reasons)
            event_id = conn.execute("INSERT INTO current_model_events(kind,payload,created_at) VALUES ('rollback',?,?)", (json.dumps(payload, ensure_ascii=False), payload["recorded_at"])).lastrowid
            conn.execute("INSERT INTO current_model_selection VALUES (1,?,?,?) ON CONFLICT(id) DO UPDATE SET run_id=excluded.run_id,event_id=excluded.event_id,effective_from=excluded.effective_from", (target_run_id, event_id, effective))
            conn.commit()
        return dict(status="rolled_back", event_id=event_id, **payload)


_COMPARISON_SPEC = None


def _initialize_comparison_worker(specification):
    global _COMPARISON_SPEC
    _COMPARISON_SPEC = specification
    import torch
    torch.set_num_threads(1)


def _comparison_worker(record):
    return _compare_stock(record, _COMPARISON_SPEC)


def _compare_stock(record, specification):
    from my_strategy.services.czsc_trend_pipeline import read_snapshot
    from my_strategy.execution.ledger import Ledger, LedgerEntry
    root, assets_root = Path(specification["root"]), Path(specification["assets_root"])
    full_frame, pack = read_snapshot(record)
    windows = []
    for window in specification["windows"]:
        frame = full_frame.loc[full_frame.date <= window["test_end"]].copy().reset_index(drop=True)
        frame.attrs.update(full_frame.attrs)
        arms = {}
        for name, stored_model in specification["models"]:
            model = (Path(stored_model[0]), stored_model[1]) if stored_model else None
            for multiplier in (1., 2.):
                key = name + ("_stress" if multiplier == 2. else "")
                if frame.empty:
                    account = _cash_evaluation_account(record, specification["calendar"], model,
                        specification["config"], window, multiplier)
                else:
                    account = replay_evaluation_account(frame, specification["calendar"], model,
                        specification["config"], window, multiplier, record["sha256"], native_pack=pack)
                metrics = audit_account(account, specification["calendar"])
                target = assets_root / window["name"] / key / (record["symbol"].replace(".", "_") + ".json")
                evidence = {k:account[k] for k in ("symbol", "series_context", "ledger", "daily", "cost_model", "terminal_state")}
                from my_strategy.services.czsc_account_artifacts import write_account_artifact
                write_account_artifact(target, evidence)
                ledger = Ledger()
                ledger.entries = [LedgerEntry(**{k:v for k,v in row.items() if k in LedgerEntry.__dataclass_fields__}) for row in account["ledger"]]
                csv_path = target.with_suffix(".ledger.csv"); ledger.to_frame().to_csv(csv_path, index=False)
                arms[key] = dict(metrics=metrics, asset=dict(path=target.relative_to(root).as_posix(), sha256=_file_hash(target),
                    ledger_path=csv_path.relative_to(root).as_posix(), ledger_sha256=_file_hash(csv_path)))
        windows.append(arms)
    return dict(symbol=record["symbol"], windows=windows)


def _cash_evaluation_account(record, calendar, model, candidate_config, window, multiplier):
    """A not-yet-observed stock retains a real, uninvested Broker account."""
    from my_strategy.execution.broker import BrokerSimulator
    from my_strategy.execution.cost_model import CostModel
    cfg = candidate_config["account"]
    costs = CostModel(version="flat_v1", impact_tiers=(), commission=cfg["commission"]*multiplier,
        min_commission=cfg["min_fee"]*multiplier, stamp_tax=cfg["stamp"]*multiplier,
        slippage=cfg["slippage"]*multiplier)
    broker = BrokerSimulator(initial_cash=cfg["initial_cash"], cost_model=costs)
    daily = [dict(date=day, cash=broker.cash, shares=0, close=None, equity=broker.cash)
             for day in calendar if window["test_start"] <= day <= window["test_end"]]
    return dict(symbol=record["symbol"], daily=daily, ledger=broker.ledger.to_frame().to_dict("records"),
        cost_model=costs.as_config_dict(), terminal_state=dict(shares=0, cash=broker.cash),
        series_context=dict(strategy=STRATEGY_ID, initial_cash=cfg["initial_cash"], source_hash=record["sha256"],
            source_revision_verified=True, calendar_hash=stable_hash(calendar), account_start=window["test_start"],
            mode="final" if model else "E0", model_identity=dict(model_run_id=model[1]["run_id"] if model else None),
            liquidity_policy=copy.deepcopy(candidate_config.get("liquidity_universe")),
            cash_reason="outside_stock_observed_history_no_prices_or_trades_invented"))


def replay_evaluation_account(frame, calendar, model, candidate_config, window, multiplier, source_hash, native_pack=None):
    """Run every arm through the same causal account and execution adapter."""
    from my_strategy.services.czsc_native_evidence import build_native_evidence
    from my_strategy.services.czsc_trend_model import TrendSession, bind_trend_inputs
    from my_strategy.services.czsc_trend_interval_model import HeadSession
    from my_strategy.services.czsc_trend_pipeline import forward_trend, predict_entry, pinned_exit_policy
    from my_strategy.services.czsc_trend_features import event_prefix_rows, original_evidence_events
    from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
    from my_strategy.services.czsc_trend_account import replay_trend_account
    cfg = copy.deepcopy(model[1]["config"] if model else candidate_config)
    root, report = model if model else (ARTIFACT_RUNS_ROOT, {})
    for key in ("commission", "min_fee", "stamp", "slippage"):
        cfg["account"][key] = candidate_config["account"][key] * multiplier
    cfg["account"]["initial_cash"] = candidate_config["account"]["initial_cash"]
    cfg["initial_cash"] = cfg["account"]["initial_cash"]
    if candidate_config.get("liquidity_universe"):
        cfg["liquidity_universe"] = copy.deepcopy(candidate_config["liquidity_universe"])
    if model is None: cfg["consensus"]["require_high_return"] = False
    # Native evidence is cost-independent and already frozen and hash-audited.
    # Validate its settings with an empty schema probe rather than rebuilding
    # full historical CZSC structures for every cost/model/window arm.
    expected = build_native_evidence(frame.iloc[:0], cfg, market_dates=calendar)
    if native_pack is not None and native_pack.get("config_hash") == expected["config_hash"]:
        end = window["test_end"]
        pack = {k:([item for item in value if str(item.get("date", ""))[:10] <= end]
                   if isinstance(value, list) else value) for k,value in native_pack.items() if k != "frame"}
        pack["frame"] = frame.copy()
    else:
        pack = build_native_evidence(frame, cfg, market_dates=calendar)
    frame = bind_trend_inputs(pack["frame"], calendar, cfg)
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    trends = forward_trend(frame, root, report.get("trend_bundles", []), cfg, TrendSession(), "historical", market_dates=calendar)
    prepared = prepare_trend_decisions(frame, calendar, native_pack=pack, trend_probabilities=trends, config=cfg, mode="E0")
    session = HeadSession()
    prefixes = dict(zip(frame.date, event_prefix_rows(frame, original_evidence_events(prepared), calendar), strict=True))
    rows = {row["date"]: row for row in frame.to_dict("records")}
    def evaluate_entry(snapshot, candidate):
        day = snapshot["date"]
        probability, identity, status = predict_entry(rows[day], snapshot, candidate, candidate.get("entry_plan") or {}, prefixes[day], root, report.get("model_bundles", []), cfg, session, "historical")
        return dict(probability=probability, model_identity=identity, status=status,
                    in_scope=status == "scored", reason_codes=[] if status == "scored" else [status],
                    training_population=cfg["training_population"], date=day,
                    available_at=day + "T15:00:00+08:00", usage_mode="historical",
                    target="actual_complete_fee_net_roundtrip_high_return_and_initial_risk",
                    probability_scope="actual_flat_account_at_signal_close")
    evaluate_entry.recipe_hash = stable_hash(dict(model=report.get("run_id"), models=report.get("model_bundles", []), purpose="locked_development_comparison"))
    exits = pinned_exit_policy(frame, calendar, prepared, root, report.get("ma_exit_bundles", []), cfg, session, "historical")
    return replay_trend_account(frame, calendar, prepared, config=cfg, exit_policy=exits,
        entry_evaluator=evaluate_entry if model else None, start=window["test_start"], end=window["test_end"],
        mode="final" if model else "E0", source_context_hash=source_hash,
        model_identity=dict(model_run_id=report.get("run_id"), usage_mode="historical"),
        run_id="comparison:" + stable_hash(dict(model=report.get("run_id"), window=window, stress=multiplier, source=source_hash))[:20])


def audit_account(account, calendar):
    """Recompute returns, costs, T+1, cash and inventory from actual records."""
    context = account["series_context"]
    if context["strategy"] != STRATEGY_ID or not context.get("source_hash") or context.get("source_revision_verified") is not True:
        raise ValueError("账本缺少当前策略与实际来源身份")
    cash = initial = float(context["initial_cash"]); shares = 0; cashflows = 0.; basis = 0.; trips = []
    daily = account["daily"]; ledger = account["ledger"]; last_date = ""; last_buy = ""
    if not daily or [r["date"] for r in daily] != sorted(set(r["date"] for r in daily)):
        raise ValueError("每日账本为空、重复或乱序")
    next_day = dict(zip(calendar[:-1], calendar[1:])); fees = account["cost_model"]
    for row in ledger:
        day, side = row["date"], row["action"]
        qty, price, fee = row["shares"], float(row["price"]), float(row["fee"])
        if day < last_date or day not in calendar or row["signal_date"] >= day:
            raise ValueError("账本执行时点违反下一核验开盘")
        if side == "BUY" and next_day.get(row["signal_date"]) != day:
            raise ValueError("买入不是信号后的下一核验交易日")
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0 or qty % 100 or not math.isfinite(price) or price <= 0:
            raise ValueError("账本整手或成交价格无效")
        if row["position_before"] != shares: raise ValueError("成交前库存不一致")
        notional = qty * price
        expected_fee = max(notional * fees["commission"], fees["min_commission"]) + (notional * fees["stamp_tax"] if side in {"SELL", "REDUCE"} else 0.)
        if not math.isclose(fee, expected_fee, abs_tol=1e-6): raise ValueError("实际账本费用不一致")
        change = -notional-fee if side == "BUY" else notional-fee
        if not math.isclose(row["cash_flow"], change, abs_tol=1e-6): raise ValueError("成交现金流含费复算不一致")
        cash += change; cashflows += change
        if side == "BUY":
            basis += notional + fee; shares += qty; last_buy = day
        elif side in {"SELL", "REDUCE"}:
            if day <= last_buy: raise ValueError("卖出违反 T+1")
            shares -= qty
            if shares == 0:
                trips.append(cashflows / basis); cashflows = basis = 0.
        else: raise ValueError("未知成交方向")
        if shares < 0 or cash < -1e-6 or row["position_after"] != shares or not math.isclose(cash, row["cash_after"], abs_tol=1e-6):
            raise ValueError("库存或现金账本不一致")
        last_date = day
    positions = 0; mark_cash = initial; index = 0; values = [initial]
    for row in daily:
        while index < len(ledger) and ledger[index]["date"] <= row["date"]:
            fill = ledger[index]; positions = fill["position_after"]; mark_cash = fill["cash_after"]; index += 1
        close = float(row["close"] or 0.)
        equity = mark_cash + positions * close
        if row["shares"] != positions or not math.isclose(row["cash"], mark_cash, abs_tol=1e-6) or not math.isclose(row["equity"], equity, abs_tol=1e-6):
            raise ValueError("每日净值与实际现金库存不一致")
        if not math.isfinite(equity) or equity <= 0: raise ValueError("净值不是正有限数")
        values.append(equity)
    if index != len(ledger): raise ValueError("成交不在每日账本覆盖范围")
    terminal = account["terminal_state"]
    if terminal.get("shares") != shares or not math.isclose(float(terminal.get("cash", float("nan"))), cash, abs_tol=1e-6):
        raise ValueError("期末实际库存或现金与账本不同")
    peak = values[0]; drawdown = 0.
    for value in values:
        peak = max(peak, value); drawdown = max(drawdown, 1-value/peak)
    return dict(symbol=account["symbol"], initial_cash=initial, total_return=values[-1]/initial-1,
        drawdown=drawdown, completed_round_trips=len(trips), trip_returns=trips,
        daily=[dict(date=r["date"], equity=r["equity"]) for r in daily], fees=sum(r["fee"] for r in ledger))


def aggregate_account_metrics(accounts):
    if not accounts: raise ValueError("评估股票池为空")
    dates = [row["date"] for row in accounts[0]["daily"]]
    if any([row["date"] for row in account["daily"]] != dates for account in accounts):
        raise ValueError("评估账户日历覆盖不一致")
    cash = sum(a["initial_cash"] for a in accounts); peak = cash; dd = 0.
    for index in range(len(dates)):
        equity = sum(a["daily"][index]["equity"] for a in accounts)
        peak = max(peak, equity); dd = max(dd, 1-equity/peak)
    trips = [value for account in accounts for value in account["trip_returns"]]
    return dict(stocks=len(accounts), total_return=sum(a["daily"][-1]["equity"] for a in accounts)/cash-1,
        expectancy=sum(trips)/len(trips) if trips else None, drawdown=dd, completed_round_trips=len(trips),
        fees=sum(a["fees"] for a in accounts), stock_returns={a["symbol"]:a["total_return"] for a in accounts})


def selection_gate(windows, policy, stocks, has_incumbent):
    reasons, pending, total = [], False, 0
    if stocks < policy["min_stocks"]:
        reasons.append("历史合格股票并集不足 60"); pending = True
    if len(windows) < 4:
        reasons.append("开发比较窗口不完整"); pending = True
    paired = []
    for window in windows:
        name = window["window"]["name"]; arms = window["arms"]
        candidate = arms["candidate"]["metrics"]; baseline = arms["rules"]["metrics"]
        incumbent = arms["incumbent"]["metrics"] if has_incumbent else None
        valid_metrics = True
        for key, arm in arms.items():
            m = arm["metrics"]
            finite = lambda value: not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            trips = m.get("completed_round_trips")
            returns = m.get("stock_returns", {})
            if (type(trips) is not int or trips < 0 or type(m.get("stocks")) is not int or m["stocks"] != stocks or
                    any(not finite(m.get(k)) for k in ("total_return", "drawdown", "fees")) or
                    not 0 <= m["drawdown"] <= 1 or m["fees"] < 0 or
                    (trips > 0 and not finite(m.get("expectancy"))) or
                    (trips == 0 and m.get("expectancy") is not None) or
                    len(returns) != stocks or set(returns) != set(candidate["stock_returns"]) or
                    any(not finite(value) for value in returns.values())):
                reasons.append(name + " " + key + " 覆盖、交易数或有限指标不合格")
                valid_metrics = False
        if not valid_metrics:
            continue
        total += candidate["completed_round_trips"]
        if candidate["completed_round_trips"] < policy["min_round_trips_per_window"]:
            reasons.append(name + " 完整交易少于每窗要求"); pending = True
        benchmarks = [baseline] + ([incumbent] if incumbent else [])
        if any(candidate["total_return"] <= benchmark["total_return"] + 1e-10 for benchmark in benchmarks):
            reasons.append(name + " 费用后收益未严格优于对照")
        expected = candidate["expectancy"]
        if expected is None or baseline["expectancy"] is None or expected < baseline["expectancy"] or (incumbent and (incumbent["expectancy"] is None or expected <= incumbent["expectancy"] + 1e-10)) or (not incumbent and expected <= baseline["expectancy"] + 1e-10):
            reasons.append(name + " 完整交易费用后期望未通过")
        if candidate["drawdown"] > min(b["drawdown"] for b in benchmarks) + policy["drawdown_tolerance"] + 1e-10:
            reasons.append(name + " 回撤超过对照加两百分点")
        stressed = arms["candidate_stress"]["metrics"]
        stress_names = ["rules_stress"] + (["incumbent_stress"] if incumbent else [])
        if any(stressed["total_return"] <= arms[key]["metrics"]["total_return"] + 1e-10 for key in stress_names):
            reasons.append(name + " 费用与滑点加倍后未优于对照")
        deltas = [candidate["stock_returns"][s] - baseline["stock_returns"][s] for s in sorted(candidate["stock_returns"])]
        paired.append(dict(window=name, independent_unit="frozen_historical_stock_account", count=len(deltas), mean_delta=sum(deltas)/len(deltas) if deltas else None,
                           positive_pairs=sum(delta > 0 for delta in deltas), formal_significance="pending_dependency_block_forward_evidence"))
    if total < policy["min_round_trips_total"]:
        reasons.append("完整交易总数少于 200"); pending = True
    return dict(passed=not reasons, pending=pending, reasons=reasons, completed_round_trips=total, paired_statistics=paired)
