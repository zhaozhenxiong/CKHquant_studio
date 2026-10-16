"""Small public API for CZSC analysis, runs and jobs."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import date
import hashlib
from pathlib import Path
import re
import threading
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

router = APIRouter(prefix="/api")
_SUBMISSION_FALLBACK = threading.Lock()
_ACTIVE_COMPUTE = frozenset({'queued', 'running', 'cancelling'})


@contextmanager
def compute_submission(request, duplicate_request=None):
    """Atomically check both task queues before submitting any new computation.

    The in-process lock lasts through submission (or synchronous analysis), not
    a background update's lifetime: its child daily hook must acquire its own
    lightweight compute lock. An existing external lightweight worker is also
    detected before accepting a request.
    """
    state = request.app.state
    lock = getattr(state, 'compute_submit_lock', _SUBMISSION_FALLBACK)
    if not lock.acquire(blocking=False):
        raise HTTPException(409, '已有计算请求正在提交或执行，请稍后再试')
    try:
        manager = getattr(state, 'tasks', None)
        legacy = manager.list() if manager is not None else []
        light = getattr(state, 'light', None)
        with getattr(light, 'lock', None) or nullcontext():
            lightweight = list(getattr(light, 'jobs', {}).values())
        boards = list(getattr(state, 'board_jobs', {}).values())
        active = [(lane, row) for lane, rows in (('legacy', legacy), ('light', lightweight), ('board', boards))
                  for row in rows if row.get('status') in _ACTIVE_COMPUTE]
        if active:
            if duplicate_request and len(active) == 1:
                lane, kind, spec = duplicate_request
                owner, row = active[0]
                if owner == lane and row.get('kind') == kind and row.get('spec') == spec:
                    yield deepcopy(row)
                    return
            raise HTTPException(409, '已有后台任务占用资源，待其结束后再开始新的计算')
        compute_lock = getattr(light, 'compute_lock', None)
        if compute_lock is not None:
            from filelock import FileLock, Timeout
            try:
                # Release before submission so the newly started worker can
                # acquire the lock. Do not reuse the potentially reentrant lock.
                with FileLock(compute_lock.lock_file).acquire(timeout=0):
                    pass
            except Timeout as exc:
                raise HTTPException(409, '外部轻量任务正在使用计算资源') from exc
        yield None
    finally:
        lock.release()


def submit_compute(request, lane, kind, spec):
    with compute_submission(request, (lane, kind, spec)) as duplicate:
        if duplicate is not None:
            return duplicate
        manager = request.app.state.light if lane == 'light' else request.app.state.tasks
        return manager.submit(kind, spec)


def _structure_preview_identity(spec, settings):
    """Verify actual causal inputs before deciding whether analysis is cached."""
    from my_strategy.adapters.czsc_adapter import load_bars
    from my_strategy.core.run_context import stable_hash
    from my_strategy.services.czsc_trend_support import _calendar
    from my_strategy.services.light_research import structure_identity
    raw = load_bars(spec.symbol, end=spec.end, as_of=spec.as_of)
    dates, calendar_id = _calendar(None)
    base = Path(__file__).parent
    sources = [base/'api.py', base/'trend_strategy.py', base/'chart_evidence_trend.py', base/'structure_projection.py',
               base.parent/'services'/'czsc_trend_runtime.py',
               base.parent/'services'/'czsc_trend_support.py',
               base.parent/'services'/'czsc_analysis.py']
    return {'kind':'structure_analysis_preview_v2',
        'request':spec.model_dump(mode='json',exclude_none=True),
        'raw_prefix_sha256':raw.attrs['data_version'],
        'calendar':{'run_id':calendar_id,'sha256':stable_hash(dates)},
        'strategy':settings,'structure':structure_identity(),
        'adapter_code':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}


def clean_symbol(value: str) -> str:
    from my_strategy.core.market_scope import require_supported_symbol
    return require_supported_symbol(value)


class MethodFields(BaseModel):
    analysis_method: Literal['structure', 'structure_kronos', 'structure_kronos_ma_mlp', 'structure_kronos_ma_l2', 'legacy_reference'] | None = None
    score_model_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    lookback: Literal[128, 256] | None = None
    horizon: Literal[1, 5, 10] | None = None
    samples: Literal[8, 32, 64] | None = None
    seed: int | None = Field(default=None, ge=0, le=2147483647, strict=True)
    budget_seconds: int | None = Field(default=None, ge=10, le=1800, strict=True)
    max_predictions: int | None = Field(default=None, ge=1, le=10000, strict=True)


class AnalysisRequest(MethodFields):
    model_config = ConfigDict(extra='forbid')
    symbol: str
    save: bool = False
    analysis_mode: Literal['reference'] = 'reference'
    current_position_fraction: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    start: date | None = date(2026, 1, 1)
    end: date | None = None
    as_of: str | None = None
    research: bool = Field(default=True, strict=True)
    model_family: Literal['current'] = 'current'
    usage_mode: Literal['forward', 'historical'] = 'historical'
    model_policy: Literal['auto', 'pinned'] = 'auto'
    model_run_id: str | None = None
    model_fold: str | None = None
    calendar_run_id: str | None = None
    device: str | None = None
    consensus_config: dict[str, Any] | None = None
    consensus_version: Literal['reference_position_v1'] | None = None
    _symbol = field_validator('symbol')(clean_symbol)

    @field_validator('model_run_id', 'model_fold', 'calendar_run_id')
    @classmethod
    def identity(cls, value):
        if value is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', value):
            raise ValueError('运行标识不合法')
        return value

    @field_validator('current_position_fraction', mode='before')
    @classmethod
    def position_fraction(cls, value):
        if isinstance(value, bool):
            raise ValueError('当前仓位应为0到1的比例')
        return value

    @field_validator('device')
    @classmethod
    def compute_device(cls, value):
        if value is not None:
            from my_strategy.core.device import requested_torch_device
            return requested_torch_device(value)
        return value

    @model_validator(mode='after')
    def dates(self):
        if self.start and self.end and self.start > self.end:
            raise ValueError('开始日期不得晚于结束日期')
        if self.analysis_method is not None:
            self.research = self.analysis_method == 'legacy_reference'
        if self.analysis_method in {'structure_kronos', 'structure_kronos_ma_mlp', 'structure_kronos_ma_l2'}:
            if (self.model_run_id or self.model_fold or self.consensus_config is not None or self.consensus_version
                    or self.calendar_run_id or self.model_policy != 'auto' or self.current_position_fraction is not None):
                raise ValueError('组合方法不能使用历史参考模型检查点或参数')
            return self
        if self.model_fold:
            self.model_policy = 'pinned'
        if self.model_policy == 'pinned' and not self.model_run_id:
            raise ValueError('固定模型需要完整当前模型运行')
        if self.model_policy == 'pinned' and not self.model_fold:
            self.model_fold = 'production'
        from my_strategy.web_dashboard.trend_routes import resolve_analysis_consensus
        inherit_selected_consensus = self.consensus_config is None and self.model_run_id is None
        self.consensus_version, self.consensus_config = resolve_analysis_consensus(
            research=self.research, model_family=self.model_family, model_run_id=self.model_run_id,
            config=self.consensus_config, version=self.consensus_version)
        if inherit_selected_consensus:
            self.consensus_config = None
        return self


class TaskSpec(MethodFields):
    model_config = ConfigDict(extra='forbid')
    symbols: list[str] = Field(default_factory=list, max_length=10000)
    analysis_mode: Literal['reference'] | None = None
    start: date | None = None
    end: date | None = None
    device: str | None = None
    cpu_workers: int | None = Field(default=None, ge=1, le=16, strict=True)
    batch_size: int | None = Field(default=None, ge=1, le=128, strict=True)
    research: bool = Field(default=True, strict=True)
    model_family: Literal['current'] = 'current'
    consensus_config: dict[str, Any] | None = None
    consensus_version: Literal['reference_position_v1'] | None = None
    model_run_id: str | None = None
    model_fold: str | None = None
    calendar_run_id: str | None = None
    usage_mode: Literal['forward', 'historical'] = 'historical'
    model_policy: Literal['auto', 'pinned'] = 'auto'
    resume_scan_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')
    _identity = field_validator('model_run_id', 'model_fold', 'calendar_run_id')(AnalysisRequest.identity.__func__)
    _device = field_validator('device')(AnalysisRequest.compute_device.__func__)

    @field_validator('symbols')
    @classmethod
    def stocks(cls, values):
        return list(dict.fromkeys(clean_symbol(value) for value in values))

    @model_validator(mode='after')
    def dates(self):
        if self.start and self.end and self.start > self.end:
            raise ValueError('开始日期不得晚于结束日期')
        if self.analysis_method is not None:
            self.research = self.analysis_method == 'legacy_reference'
        if self.analysis_method in {'structure_kronos', 'structure_kronos_ma_mlp', 'structure_kronos_ma_l2'}:
            if (self.model_run_id or self.model_fold or self.consensus_config is not None or self.consensus_version
                    or self.calendar_run_id or self.model_policy != 'auto'):
                raise ValueError('组合方法不能使用历史参考模型检查点或参数')
            return self
        if self.resume_scan_id:
            raise ValueError('只有组合方法扫描支持按冻结结果续算')
        if self.model_policy == 'pinned' and not self.model_run_id:
            raise ValueError('固定模型需要完整当前模型运行')
        if self.model_fold and not self.model_run_id:
            raise ValueError('固定检查点需要模型运行')
        from my_strategy.web_dashboard.trend_routes import validate_consensus_config
        inherit_selected_consensus = self.consensus_config is None and self.model_run_id is None
        self.consensus_config = validate_consensus_config(self.consensus_config,
            version=self.consensus_version, model_run_id=self.model_run_id, model_family=self.model_family)
        self.consensus_version = 'reference_position_v1'
        if inherit_selected_consensus:
            self.consensus_config = None
        return self


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['scan', 'update']
    spec: TaskSpec = Field(default_factory=TaskSpec)

    @model_validator(mode='after')
    def start_date(self):
        expected_mode = 'reference' if self.kind == 'scan' else None
        if self.spec.analysis_mode is not None and self.spec.analysis_mode != expected_mode:
            raise ValueError('任务用途与分析模式不一致')
        self.spec.analysis_mode = expected_mode
        if self.kind == 'update' and (self.spec.analysis_method or self.spec.resume_scan_id):
            raise ValueError('更新行情不接受扫描方法或续算参数')
        if self.kind == 'scan' and self.spec.start is None and not self.spec.resume_scan_id:
            self.spec.start = date(2026, 1, 1)
            self.spec.dates()
        return self


class TrainingParametersRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    parameters: dict[str, Any] = Field(default_factory=dict)


class TrainingPreflightRequest(TrainingParametersRequest):
    source_native_run_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')
    raw_source_run_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')
    resume_native_run_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')


class DefaultProfileRequest(TrainingParametersRequest):
    expected_revision: int = Field(ge=0, strict=True)


class TrainingJobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    plan_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')
    parameters: dict[str, Any] | None = None
    resume_reference_run_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')
    resume_training_run_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')
    resume_from_job_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')


class ModelValidationJobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    plan_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')
    model_run_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')


class SelectionRollbackRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expected_run_id: str | None = None
    target_run_id: str | None = None


class WatchlistRequest(BaseModel):
    symbols: list[str] = Field(min_length=1, max_length=10000)

    @field_validator("symbols")
    @classmethod
    def stocks(cls, values):
        return list(dict.fromkeys(clean_symbol(value) for value in values))




@router.get("/watchlist")
def watchlist(request: Request):
    return request.app.state.personal.watchlist()


@router.post("/watchlist")
def add_watchlist(spec: WatchlistRequest, request: Request):
    return request.app.state.personal.add_watchlist(spec.symbols)


@router.delete("/watchlist/{symbol}")
def remove_watchlist(symbol: str, request: Request):
    # Removal of an older excluded entry must remain possible.
    from my_strategy.core.market_scope import normalize_public_symbol
    return request.app.state.personal.remove_watchlist(normalize_public_symbol(symbol))








@router.get("/health")
def health():
    return {"status": "ok", "method": "CZSC", "strategy": "reference_position_v1"}


@router.get("/data/status")
async def status(refresh: bool = False):
    from my_strategy.adapters.czsc_adapter import data_status
    return await asyncio.to_thread(data_status, refresh=refresh)


@router.get("/data/latest")
async def latest_dates(symbol: str | None = Query(default=None, max_length=20)):
    from my_strategy.web_dashboard.display_dates import latest_display_dates
    try:
        return await asyncio.to_thread(latest_display_dates, clean_symbol(symbol) if symbol else None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/compute/status")
async def compute(device: str | None = None):
    from my_strategy.services.czsc_compute import compute_status
    return await asyncio.to_thread(compute_status, device)


@router.api_route('/research/{retired_path:path}', methods=['GET', 'POST', 'PUT', 'DELETE'])
async def retired_research(retired_path: str):
    raise HTTPException(status_code=410, detail='旧研究和发布接口已退役，请使用模型训练中心')


@router.get('/training/capabilities')
async def training_capabilities(request: Request):
    return await asyncio.to_thread(request.app.state.training.capabilities)


@router.get('/training/profiles')
async def training_profiles(request: Request):
    return await asyncio.to_thread(request.app.state.training.profiles)


@router.put('/training/default-profile')
async def save_default(spec: DefaultProfileRequest, request: Request):
    return await asyncio.to_thread(request.app.state.training.save_default, spec.parameters, spec.expected_revision)


@router.post('/training/preflight')
async def training_preflight(spec: TrainingPreflightRequest, request: Request):
    sources = {key:getattr(spec,key) for key in ("source_native_run_id","raw_source_run_id","resume_native_run_id")
               if getattr(spec,key) is not None}
    return await asyncio.to_thread(request.app.state.training.preflight, spec.parameters, **sources)


@router.post('/training/jobs', status_code=202)
async def training_job(spec: TrainingJobRequest, request: Request):
    await asyncio.to_thread(request.app.state.training.validate_plan, spec.plan_id, spec.parameters)
    return submit_compute(request, 'legacy', 'model_training', spec.model_dump(exclude_none=True))


@router.post('/training/validation-jobs', status_code=202)
async def model_validation_job(spec: ModelValidationJobRequest, request: Request):
    await asyncio.to_thread(request.app.state.training.validate_plan, spec.plan_id, check_code=False)
    return submit_compute(request, 'legacy', 'model_validation', spec.model_dump())


@router.get('/training/status')
async def training_status(request: Request):
    status = await asyncio.to_thread(request.app.state.training.status)
    status['jobs'] = [job for job in request.app.state.tasks.list() if job['kind'] in {'model_training', 'model_validation'}]
    return status


@router.get('/training/selection-events')
async def selection_events(request: Request):
    return await asyncio.to_thread(request.app.state.training.selection_events)


@router.get('/training/runs/{run_id}/comparison')
async def model_comparison(run_id: str, request: Request):
    return await asyncio.to_thread(request.app.state.training.comparison, run_id)


@router.get('/training/diagnostics/{run_id}')
async def sample_diagnostics(run_id: str, request: Request,
    expected_sha256: str | None = Query(default=None, pattern=r'^[a-f0-9]{64}$')):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', run_id):
        raise HTTPException(status_code=422, detail='诊断运行标识不合法')
    def read():
        import json
        from my_strategy.services.czsc_trend_support import _file_hash
        base = request.app.state.tasks.results.runs_root.resolve()
        path = (base / run_id / 'reports/high_return_sample_diagnostics.json').resolve()
        if not path.is_relative_to(base):
            raise ValueError('诊断路径越界')
        if expected_sha256 and _file_hash(path) != expected_sha256:
            raise ValueError('诊断文件与任务核验哈希不同')
        report = json.loads(path.read_text())
        if report.get('family') != 'trend_consensus' or report.get('artifact_integrity_passed') is not True:
            raise ValueError('诊断不是已核验的当前模型证据')
        model = (base / report['model_run_id'] / 'reports/trend_consensus.json').resolve()
        if not model.is_relative_to(base) or _file_hash(model) != report['source_report_sha256']:
            raise ValueError('诊断来源模型报告已改变')
        return report
    return await asyncio.to_thread(read)


@router.post('/training/rollback')
async def selection_rollback(spec: SelectionRollbackRequest, request: Request):
    return await asyncio.to_thread(request.app.state.training.rollback, spec.expected_run_id, spec.target_run_id)


@router.get("/symbols")
async def symbols(q: str = "", limit: int = Query(default=50, ge=1, le=10000)):
    from my_strategy.core.market_scope import supported_catalog
    return {"symbols": await asyncio.to_thread(supported_catalog, query=q or None, limit=limit)}


@router.post("/analysis")
async def analysis(spec: AnalysisRequest, request: Request):
    from my_strategy.services.czsc_analysis import analyze_stock, strategy_config
    from my_strategy.core.run_context import create_run_context
    def calculate_guarded():
        query = spec.model_dump(mode="json", exclude_none=True)
        calculation_query = {key: value for key, value in query.items()
                             if key not in {"consensus_config", "consensus_version", "save"}}
        settings = strategy_config()
        preview_identity = None
        from my_strategy.services.analysis_methods import COMBINED, request_context
        combined = spec.analysis_method in COMBINED
        if not spec.save and not spec.research and not combined:
            preview_identity = _structure_preview_identity(spec, settings)
            cached = request.app.state.light.store.get(preview_identity, namespace='analysis_previews')
            if cached:
                return {**cached, 'reused':True}
        if combined:
            from my_strategy.services.analysis_methods import analyze_method
            result = analyze_method(request.app.state.light, query)
            query.update(result['method_context'])
        elif spec.research:
            from my_strategy.web_dashboard.trend_strategy import analyze_trend_consensus
            result = analyze_trend_consensus(**calculation_query, config=settings, consensus_config=spec.consensus_config,
                                         consensus_version=spec.consensus_version)
        else:
            baseline_query = {key: calculation_query[key] for key in ("symbol", "start", "end", "as_of", "current_position_fraction") if key in calculation_query}
            from my_strategy.web_dashboard.trend_strategy import analyze_trend_consensus
            result = analyze_trend_consensus(**baseline_query, config=settings, structure_only=True)
        if spec.analysis_method is not None:
            result.setdefault('analysis_method', spec.analysis_method)
            result.setdefault('method_context', request_context(query))
        consensus = result.get("consensus_decisions") or {}
        if consensus.get("config"):
            query["consensus_config"] = consensus["config"]
            query["consensus_version"] = consensus["version"]
        if result.get("strategy_version") == "reference_position_v1":
            research_context = result.get("research", {})
            selected_model = research_context.get("model_run_id")
            if selected_model:
                query["model_run_id"] = selected_model
            if research_context.get("calendar_run_id"):
                query["calendar_run_id"] = research_context["calendar_run_id"]
        run_config = {"request": query, "strategy": settings}
        if consensus.get("config_hash"):
            run_config["consensus_config_hash"] = consensus["config_hash"]
            run_config["consensus_version"] = consensus["version"]
        if result.get("research", {}).get("entry_plan_config_hash"):
            run_config["entry_plan_config_hash"] = result["research"]["entry_plan_config_hash"]
        if not spec.save:
            from my_strategy.core.run_context import stable_hash
            if preview_identity is not None and preview_identity != _structure_preview_identity(spec, settings):
                raise ValueError('分析期间行情、日历或代码已改变，请重试')
            identity = preview_identity or {"kind":"analysis_preview_v1", "config":run_config,
                                           "result_sha256":stable_hash(result)}
            store = request.app.state.light.store
            cached = store.get(identity, namespace="analysis_previews")
            if cached:
                return {**cached, "reused":True}
            result.update(request=query, saved=False, preview=True,
                          preview_id=stable_hash(identity), reused=False)
            store.gc(dry_run=False)
            store.put(identity, result, namespace="analysis_previews")
            return result
        context = create_run_context(task="czsc-analysis", as_of_date=result.get("decision_date", result["data_end"]),
                                     config=run_config,
                                     data_version=result["data_version"], scope="single", source="web",
                                     stocks=[spec.symbol], start_date=spec.start, end_date=result["data_end"])
        result.update(run_id=context.run_id, run_context=context.to_dict(), request=query)
        return request.app.state.tasks.results.save("analysis", result)
    def calculate():
        with compute_submission(request):
            return calculate_guarded()
    return await asyncio.to_thread(calculate)


@router.get("/tasks")
def tasks(request: Request):
    return {"tasks": [job for job in request.app.state.tasks.list() if job["kind"] in {"scan", "update", "model_training", "model_validation", "board_features", "board_sync", "board_knowledge"}]}


@router.post("/tasks", status_code=202)
def submit(spec: TaskRequest, request: Request):
    return submit_compute(request, 'legacy', spec.kind, spec.spec.model_dump(mode="json", exclude_none=True,
        exclude_unset=bool(spec.spec.resume_scan_id)))


@router.get("/tasks/{job_id}")
def task(job_id: str, request: Request, view: Literal["scan"] | None = None):
    if view not in (None, "scan"):
        raise HTTPException(status_code=422, detail="任务视图无效")
    saved = request.app.state.tasks.get(job_id)
    if saved.get("kind") not in {"scan", "update", "model_training", "model_validation", "board_features", "board_sync", "board_knowledge"}:
        raise HTTPException(status_code=404, detail="任务不存在")
    if view == "scan" and saved.get("kind") == "scan" and saved.get("status") in {"succeeded", "partial", "cancelled"} and isinstance(saved.get("result"), dict):
        return {**saved, "result": compact_scan_view(saved["result"])}
    return saved


@router.post("/tasks/{job_id}/cancel")
def cancel(job_id: str, request: Request):
    return request.app.state.tasks.cancel(job_id)


@router.get("/runs")
def runs(request: Request, kind: str | None = None, limit: int = Query(default=30, ge=1, le=200)):
    allowed = {"analysis", "scan", "update", "model_training", "model_validation", "current_train", "current_validation", "board_features", "board_sync", "board_knowledge"}
    if kind is not None and kind not in allowed:
        raise HTTPException(status_code=422, detail="运行类型不受支持")
    return {"runs": [run for run in request.app.state.tasks.results.list(kind, limit) if run.get("kind") in allowed]}


from my_strategy.storage.scan_view import compact_scan_view


@router.get("/runs/{run_id}")
def run(run_id: str, request: Request, view: Literal["scan"] | None = None,
        page: int | None = Query(default=None,ge=1), page_size: int = Query(default=40,ge=1,le=100),
        q: str = Query(default='',max_length=200), category: str = Query(default='all',max_length=40),
        symbols: str | None = Query(default=None,max_length=10000)):
    store = request.app.state.tasks.results
    if view not in (None, "scan") or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_id):
        raise HTTPException(status_code=422, detail="运行视图无效")
    from contextlib import closing
    from pathlib import Path
    import sqlite3
    path = Path(store.db_path).resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="扫描运行不存在")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
        conn.execute("PRAGMA query_only=ON")
        indexed = conn.execute("SELECT kind FROM czsc_runs WHERE run_id=?", (run_id,)).fetchone()
    if indexed is None:
        raise HTTPException(status_code=404, detail="扫描运行不存在")
    if indexed[0] not in {"analysis", "scan", "update", "model_training", "model_validation", "current_train", "current_validation", "board_features", "board_sync", "board_knowledge"}:
        raise HTTPException(status_code=404, detail="运行不存在")
    if view is None:
        return store.get(run_id)
    if indexed[0] != "scan":
        raise HTTPException(status_code=422, detail="紧凑扫描视图仅适用于扫描运行")
    return store.get_scan_page(run_id,page=page,page_size=page_size,query=q,category=category,
        symbols=symbols.split(',') if symbols is not None else None)


@router.get("/runs/{run_id}/market-history")
def market_history(run_id: str, request: Request,
                   date: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
                   sector: str | None = Query(default=None, max_length=120),
                   event_type: str | None = Query(default=None, max_length=120),
                   state: str | None = Query(default=None, max_length=40),
                   page: int = Query(default=1, ge=1),
                   page_size: int = Query(default=25, ge=1, le=100)):
    from contextlib import closing
    from pathlib import Path
    import sqlite3
    from my_strategy.storage.czsc_market_history import market_history_facts, STATES
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_id) or state is not None and state not in STATES:
        raise HTTPException(status_code=422, detail="统计运行或状态筛选无效")
    try:
        # A full-market report contains thousands of detailed decisions. Only
        # verify the small indexed run identity before reading bounded facts.
        path = Path(request.app.state.tasks.results.db_path).resolve()
        if not path.is_file():
            raise KeyError(run_id)
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
            conn.execute("PRAGMA query_only=ON")
            saved = conn.execute("SELECT kind FROM czsc_runs WHERE run_id=?", (run_id,)).fetchone()
        if saved is None or saved[0] != "scan":
            raise KeyError(run_id)
        return market_history_facts(run_id, date=date, sector=sector, event_type=event_type,
                                    state=state, page=page, page_size=page_size,
                                    db_path=request.app.state.tasks.results.db_path)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="统计运行不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
