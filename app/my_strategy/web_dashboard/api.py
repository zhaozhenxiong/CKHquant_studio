"""Small public API for CZSC analysis, runs and jobs."""
from __future__ import annotations

import asyncio
from datetime import date
import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

router = APIRouter(prefix="/api")
SYMBOL = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")


def clean_symbol(value: str) -> str:
    value = value.strip().upper()
    if not SYMBOL.fullmatch(value):
        raise ValueError("股票代码应为六位数字及 .SH/.SZ/.BJ 后缀")
    return value


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    symbol: str
    analysis_mode: Literal['reference'] = 'reference'
    start: date | None = date(2026, 1, 1)
    end: date | None = None
    as_of: str | None = None
    research: bool = Field(default=True, strict=True)
    model_family: Literal['current'] = 'current'
    usage_mode: Literal['forward', 'historical'] = 'historical'
    model_policy: Literal['auto', 'pinned'] = 'auto'
    entry_policy: Literal['risk', 'legacy'] = 'risk'
    model_run_id: str | None = None
    model_fold: str | None = None
    calendar_run_id: str | None = None
    device: str | None = None
    consensus_config: dict[str, Any] | None = None
    consensus_version: Literal['trend_consensus'] | None = None
    account_start: date | None = None
    account_initial_cash: float | None = Field(default=None, gt=0, le=1e12, allow_inf_nan=False)
    _symbol = field_validator('symbol')(clean_symbol)

    @field_validator('model_run_id', 'model_fold', 'calendar_run_id')
    @classmethod
    def identity(cls, value):
        if value is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', value):
            raise ValueError('运行标识不合法')
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
        if self.model_fold:
            self.model_policy = 'pinned'
        if self.model_policy == 'pinned' and not self.model_run_id:
            raise ValueError('固定模型需要完整当前模型运行')
        if self.model_policy == 'pinned' and not self.model_fold:
            self.model_fold = 'production'
        from my_strategy.web_dashboard.trend_routes import resolve_analysis_consensus
        self.consensus_version, self.consensus_config = resolve_analysis_consensus(
            research=self.research, model_family=self.model_family, model_run_id=self.model_run_id,
            config=self.consensus_config, version=self.consensus_version)
        self.entry_policy = 'risk' if self.research else 'legacy'
        return self


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    symbols: list[str] = Field(default_factory=list, max_length=10000)
    analysis_mode: Literal['reference', 'backtest'] | None = None
    start: date | None = None
    end: date | None = None
    initial_cash: float = Field(default=100000, gt=0, le=1e12, allow_inf_nan=False)
    device: str | None = None
    cpu_workers: int | None = Field(default=None, ge=1, le=16, strict=True)
    batch_size: int | None = Field(default=None, ge=1, le=128, strict=True)
    research: bool = Field(default=True, strict=True)
    model_family: Literal['current'] = 'current'
    consensus_config: dict[str, Any] | None = None
    consensus_version: Literal['trend_consensus'] | None = None
    model_run_id: str | None = None
    model_fold: str | None = None
    calendar_run_id: str | None = None
    usage_mode: Literal['forward', 'historical'] = 'historical'
    model_policy: Literal['auto', 'pinned'] = 'auto'
    entry_policy: Literal['risk', 'legacy'] = 'risk'
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
        if self.model_policy == 'pinned' and not self.model_run_id:
            raise ValueError('固定模型需要完整当前模型运行')
        if self.model_fold and not self.model_run_id:
            raise ValueError('固定检查点需要模型运行')
        from my_strategy.web_dashboard.trend_routes import validate_consensus_config
        self.consensus_config = validate_consensus_config(self.consensus_config,
            version=self.consensus_version, model_run_id=self.model_run_id, model_family=self.model_family)
        self.consensus_version = 'trend_consensus'
        self.entry_policy = 'risk' if self.research else 'legacy'
        return self


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['scan', 'backtest', 'update']
    spec: TaskSpec = Field(default_factory=TaskSpec)

    @model_validator(mode='after')
    def start_date(self):
        expected_mode = 'reference' if self.kind == 'scan' else 'backtest' if self.kind == 'backtest' else None
        if self.spec.analysis_mode is not None and self.spec.analysis_mode != expected_mode:
            raise ValueError('任务用途与分析模式不一致')
        self.spec.analysis_mode = expected_mode
        if self.kind == 'backtest' and (self.spec.usage_mode != 'historical' or self.spec.model_policy == 'pinned' or self.spec.model_fold is not None):
            raise ValueError('回测只支持历史按日期匹配，固定检查点与当前时点研究请使用分析入口')
        if self.kind in {'scan', 'backtest'} and self.spec.start is None:
            self.spec.start = date(2026, 1, 1)
            self.spec.dates()
        return self


class TrainingParametersRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    parameters: dict[str, Any] = Field(default_factory=dict)


class TrainingPreflightRequest(TrainingParametersRequest):
    source_native_run_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')


class DefaultProfileRequest(TrainingParametersRequest):
    expected_revision: int = Field(ge=0, strict=True)


class TrainingJobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    plan_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')
    parameters: dict[str, Any] | None = None


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


class HoldingRequest(BaseModel):
    shares: int = Field(gt=0, le=1_000_000_000, strict=True)
    average_cost: float = Field(gt=0, le=1_000_000_000, allow_inf_nan=False)

    @field_validator("average_cost", mode="before")
    @classmethod
    def no_boolean_cost(cls, value):
        if isinstance(value, bool):
            raise ValueError("平均成本应为正数")
        return value


@router.get("/watchlist")
def watchlist(request: Request):
    return request.app.state.personal.watchlist()


@router.post("/watchlist")
def add_watchlist(spec: WatchlistRequest, request: Request):
    return request.app.state.personal.add_watchlist(spec.symbols)


@router.delete("/watchlist/{symbol}")
def remove_watchlist(symbol: str, request: Request):
    return request.app.state.personal.remove_watchlist(clean_symbol(symbol))


@router.get("/holdings")
def holdings(request: Request):
    return request.app.state.personal.holdings()


@router.put("/holdings/{symbol}")
def save_holding(symbol: str, spec: HoldingRequest, request: Request):
    return request.app.state.personal.save_holding(clean_symbol(symbol), spec.shares, spec.average_cost)


@router.delete("/holdings/{symbol}")
def remove_holding(symbol: str, request: Request):
    return request.app.state.personal.remove_holding(clean_symbol(symbol))


@router.get("/health")
def health():
    return {"status": "ok", "method": "CZSC", "strategy": "trend_consensus"}


@router.get("/data/status")
async def status():
    from my_strategy.adapters.czsc_adapter import data_status
    return await asyncio.to_thread(data_status)


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
    if spec.source_native_run_id is None:
        return await asyncio.to_thread(request.app.state.training.preflight, spec.parameters)
    return await asyncio.to_thread(request.app.state.training.preflight, spec.parameters,
        source_native_run_id=spec.source_native_run_id)


@router.post('/training/jobs', status_code=202)
async def training_job(spec: TrainingJobRequest, request: Request):
    await asyncio.to_thread(request.app.state.training.validate_plan, spec.plan_id, spec.parameters)
    return request.app.state.tasks.submit('model_training', spec.model_dump(exclude_none=True))


@router.post('/training/validation-jobs', status_code=202)
async def model_validation_job(spec: ModelValidationJobRequest, request: Request):
    await asyncio.to_thread(request.app.state.training.validate_plan, spec.plan_id)
    return request.app.state.tasks.submit('model_validation', spec.model_dump())


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
    from my_strategy.adapters.czsc_adapter import list_symbols
    return {"symbols": await asyncio.to_thread(list_symbols, query=q or None, limit=limit)}


@router.post("/analysis")
async def analysis(spec: AnalysisRequest, request: Request):
    from my_strategy.services.czsc_analysis import analyze_stock, strategy_config
    from my_strategy.core.run_context import create_run_context
    def calculate():
        query = spec.model_dump(mode="json", exclude_none=True)
        calculation_query = {key: value for key, value in query.items()
                             if key not in {"consensus_config", "consensus_version"}}
        settings = strategy_config()
        if spec.research:
            from my_strategy.web_dashboard.trend_strategy import analyze_trend_consensus
            result = analyze_trend_consensus(**calculation_query, config=settings, consensus_config=spec.consensus_config,
                                         consensus_version=spec.consensus_version)
        else:
            baseline_query = {key: calculation_query[key] for key in ("symbol", "start", "end", "as_of") if key in calculation_query}
            from my_strategy.web_dashboard.trend_strategy import analyze_trend_consensus
            result = analyze_trend_consensus(**baseline_query, config=settings, structure_only=True)
        consensus = result.get("consensus_decisions") or {}
        if consensus.get("config"):
            query["consensus_config"] = consensus["config"]
            query["consensus_version"] = consensus["version"]
        if result.get("strategy_version") == "trend_consensus":
            research_context = result.get("research", {})
            selected_model = research_context.get("model_run_id")
            if selected_model:
                query["model_run_id"] = selected_model
            if research_context.get("calendar_run_id"):
                query["calendar_run_id"] = research_context["calendar_run_id"]
            account_context = (result.get("account_actions") or {}).get("context", {})
            if account_context.get("account_start"):
                query["account_start"] = account_context["account_start"]
            if account_context.get("initial_cash") is not None:
                query["account_initial_cash"] = account_context["initial_cash"]
        run_config = {"request": query, "strategy": settings}
        if consensus.get("config_hash"):
            run_config["consensus_config_hash"] = consensus["config_hash"]
            run_config["consensus_version"] = consensus["version"]
        if spec.entry_policy != "legacy" and result.get("research", {}).get("entry_plan_config_hash"):
            run_config["entry_plan_config_hash"] = result["research"]["entry_plan_config_hash"]
        context = create_run_context(task="czsc-analysis", as_of_date=result.get("decision_date", result["data_end"]),
                                     config=run_config,
                                     data_version=result["data_version"], scope="single", source="web",
                                     stocks=[spec.symbol], start_date=spec.start, end_date=result["data_end"])
        result.update(run_id=context.run_id, run_context=context.to_dict(), request=query)
        return request.app.state.tasks.results.save("analysis", result)
    return await asyncio.to_thread(calculate)


@router.get("/tasks")
def tasks(request: Request):
    return {"tasks": request.app.state.tasks.list()}


@router.post("/tasks", status_code=202)
def submit(spec: TaskRequest, request: Request):
    return request.app.state.tasks.submit(spec.kind, spec.spec.model_dump(mode="json", exclude_none=True))


@router.get("/tasks/{job_id}")
def task(job_id: str, request: Request, view: Literal["scan"] | None = None):
    if view not in (None, "scan"):
        raise HTTPException(status_code=422, detail="任务视图无效")
    saved = request.app.state.tasks.get(job_id)
    if view == "scan" and saved.get("kind") == "scan" and saved.get("status") == "succeeded" and isinstance(saved.get("result"), dict):
        return {**saved, "result": compact_scan_view(saved["result"])}
    return saved


@router.post("/tasks/{job_id}/cancel")
def cancel(job_id: str, request: Request):
    return request.app.state.tasks.cancel(job_id)


@router.get("/runs")
def runs(request: Request, kind: str | None = None, limit: int = Query(default=30, ge=1, le=200)):
    return {"runs": request.app.state.tasks.results.list(kind, limit)}


def _scan_display_metadata(value):
    """Keep saved display identities and scores, omitting raw model inputs/contracts."""
    if not isinstance(value, dict):
        return value
    retained = {key: item for key, item in value.items()
                if not isinstance(item, (dict, list))}
    for key in ("reason_codes", "reasons", "model_input_reason_codes", "original_opportunity_ids",
                "original_event_ids", "origin_event_ids", "sources", "point_types", "label_target", "expert_probabilities",
                "native_labels"):
        if key in value:
            retained[key] = value[key]
    for key in ("model_identity", "fallback_model", "rule_identity", "current_state", "current_valid_zone"):
        if key in value:
            retained[key] = _scan_display_metadata(value[key])
    if "model_availability_diagnostics" in value:
        def availability(record):
            if not isinstance(record, dict):
                return record
            result = _scan_display_metadata({key: item for key, item in record.items()
                                            if key != "model_availability_diagnostics"})
            if "candidates" in record:
                result["candidates"] = [_scan_display_metadata(item) for item in record["candidates"]]
            if "dual_fallback" in record:
                result["dual_fallback"] = availability(record["dual_fallback"])
            return result
        retained["model_availability_diagnostics"] = availability(value["model_availability_diagnostics"])
    # Targets are rendered directly, including older descriptive target objects.
    if "target" in value:
        retained["target"] = value["target"]
    return retained


_SCAN_EVENT_FIELDS = frozenset(("event_id", "opportunity_id", "original_event_id", "original_pattern_id",
    "symbol", "source", "source_frequency", "event_type", "signal", "name", "side", "action", "time",
    "anchor_at", "available_at", "first_available_at", "first_confirmed_at", "origin_observed_at",
    "signal_at", "observed_at", "confirmed_at", "reference_price_date", "reference_price_source",
    "point_type", "confirmed", "input_eligible", "candidate_eligible", "direction_eligible",
    "risk_candidate_eligible", "risk_reason_codes", "reason_codes", "reason", "reference_price",
    "stop_price", "risk_stop_price", "stop_distance", "price_floor", "price_ceiling", "target_weight", "sources",
    "candidate_direction_valid", "evidence_visible", "pattern_id", "version", "family", "rule_name",
    "is_confirmed", "invalidates_event_id"))


def _scan_events(values):
    if not isinstance(values, list):
        return values
    return [{key: item for key, item in value.items() if key in _SCAN_EVENT_FIELDS}
            if isinstance(value, dict) else value for value in values]


def _scan_account_context(value):
    if not isinstance(value, dict):
        return value
    fields = ("strategy", "symbol", "account_id", "account_kind", "account_start", "initial_cash",
              "initial_state_hash", "source_hash", "calendar_hash", "config_hash", "mode",
              "series_context_hash", "entry_recipe_hash", "exit_recipe_hash", "source_revision_verified")
    result = {key: value[key] for key in fields if key in value}
    if "model_identity" in value:
        result["model_identity"] = _scan_display_metadata(value["model_identity"])
    return result


def _scan_row_view(row):
    if not isinstance(row, dict):
        return row
    # Retain unknown small top-level display fields for older scan families.
    result = {key: value for key, value in row.items()
              if key not in {"active_candidates", "raw_risk_plan"}}
    if "entry_attempt" in row:
        result["entry_attempt"] = _scan_display_metadata(row["entry_attempt"])
        if isinstance(row["entry_attempt"], dict) and "evaluation" in row["entry_attempt"]:
            result["entry_attempt"]["evaluation"] = _scan_display_metadata(row["entry_attempt"]["evaluation"])
    for key in ("model_resolution", "exit_model", "policy_entry_model"):
        if key in row:
            result[key] = _scan_display_metadata(row[key])
    if isinstance(row.get("agents"), dict):
        result["agents"] = {key: _scan_display_metadata(value) for key, value in row["agents"].items()}
    if "account_context" in row:
        result["account_context"] = _scan_account_context(row["account_context"])
    if isinstance(row.get("account_action"), dict):
        action = row["account_action"]
        result["account_action"] = {key: value for key, value in action.items()
                                    if key not in {"direction_decision", "entry_evaluation", "exit_evaluation"}}
        for key in ("entry_evaluation", "exit_evaluation"):
            if key in action:
                result["account_action"][key] = _scan_display_metadata(action[key])
    candidate = row.get("candidate_evidence")
    if isinstance(candidate, list):
        result["candidate_evidence"] = _scan_events(candidate)
    elif isinstance(candidate, dict):
        result["candidate_evidence"] = {key: value for key, value in candidate.items()
                                        if not isinstance(value, (dict, list))}
        for key in ("facts", "risk_details"):
            if key in candidate:
                result["candidate_evidence"][key] = _scan_events(candidate[key])
    for key in ("events", "signal_points"):
        if key in row:
            result[key] = _scan_events(row[key])
    if isinstance(row.get("high_return_evidence"), dict):
        evidence = row["high_return_evidence"]
        result["high_return_evidence"] = {key: _scan_events(evidence[key]) if key != "sources" else evidence[key]
                                          for key in ("signal_points", "events", "sources") if key in evidence}
    return result


def compact_scan_view(result):
    """Opt-in transport projection only; the saved report and default GET stay full."""
    view = {**result, "rows": [_scan_row_view(row) for row in result.get("rows", [])]}
    contexts = [row.get("account_context") for row in result.get("rows", []) if isinstance(row, dict)]
    status = "unavailable"
    if contexts and all(isinstance(context, dict) for context in contexts):
        shared = {
            "account_start": [context.get("account_start") for context in contexts],
            "account_initial_cash": [context.get("initial_cash") for context in contexts],
            "model_policy": [(context.get("model_identity") or {}).get("model_policy") for context in contexts],
            "model_fold": [(context.get("model_identity") or {}).get("model_fold") for context in contexts],
            "model_run_id": [(context.get("model_identity") or {}).get("model_run_id") for context in contexts],
        }
        status = "consistent" if all(all(value == values[0] for value in values) for values in shared.values()) else "mixed"
        status = status if shared["account_start"][0] is not None and shared["account_initial_cash"][0] is not None else "unavailable"
        sources = [result, result.get("request") or {},
                   ((result.get("run_context") or {}).get("config") or {}).get("parameters") or {}]
        if status == "consistent":
            for key, values in shared.items():
                for source in sources:
                    name = "initial_cash" if key == "account_initial_cash" and "initial_cash" in source else key
                    if source.get(name) is not None and source[name] != values[0]:
                        status = "mixed"
            if status == "consistent":
                for key, values in shared.items():
                    if key != "model_run_id" and values[0] is not None:
                        if view.get(key) is None:
                            view[key] = values[0]
                        if isinstance(result.get("request"), dict):
                            view.setdefault("request", dict(result["request"]))
                            # Do not mutate the saved request while supplying
                            # the common explicit account for client drilling.
                            if view["request"] is result["request"]:
                                view["request"] = dict(result["request"])
                            if view["request"].get(key) is None:
                                view["request"][key] = values[0]
    view["scan_view"] = {"schema": "current_scan_view", "compact": True,
                         "account_context_status": status, "raw_evidence": "per_stock_analysis"}
    return view


@router.get("/runs/{run_id}")
def run(run_id: str, request: Request, view: Literal["scan"] | None = None):
    store = request.app.state.tasks.results
    if view is None:
        return store.get(run_id)
    if view != "scan" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", run_id):
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
    if indexed[0] != "scan":
        raise HTTPException(status_code=422, detail="紧凑扫描视图仅适用于扫描运行")
    return compact_scan_view(store.get(run_id))


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
