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
    symbol: str
    start: date | None = date(2026, 1, 1)
    end: date | None = None
    as_of: str | None = None
    research: bool = Field(default=True, strict=True)
    model_family: Literal["ma_trend", "dual", "wyckoff", "high_return"] = "ma_trend"
    usage_mode: Literal["production", "historical", "retrospective"] = "historical"
    model_policy: Literal["auto", "pinned"] = "auto"
    entry_policy: Literal["legacy", "fresh", "risk"] = "legacy"
    model_run_id: str | None = None
    model_fold: str | None = None
    calendar_run_id: str | None = None
    device: str | None = None
    consensus_config: dict[str, Any] | None = None
    consensus_version: Literal["czsc_agent_consensus_v1", "czsc_agent_consensus_v2", "czsc_agent_consensus_v3", "czsc_agent_consensus_v4", "czsc_trend_consensus"] | None = None
    account_start: date | None = None
    account_initial_cash: float | None = Field(default=None, gt=0, le=1e12, allow_inf_nan=False)
    _symbol = field_validator("symbol")(clean_symbol)

    @model_validator(mode="after")
    def arbitration_parameters(self):
        from my_strategy.web_dashboard.trend_routes import resolve_analysis_consensus
        self.consensus_version, self.consensus_config = resolve_analysis_consensus(
            research=self.research, model_family=self.model_family, model_run_id=self.model_run_id,
            config=self.consensus_config, version=self.consensus_version)
        if self.consensus_version == "czsc_trend_consensus":
            self.entry_policy = "risk"
        return self

    @field_validator("model_run_id", "model_fold", "calendar_run_id")
    @classmethod
    def identity(cls, value):
        if value is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", value):
            raise ValueError("研究运行标识不合法")
        return value

    @field_validator("device")
    @classmethod
    def compute_device(cls, value):
        if value is not None:
            from my_strategy.core.device import requested_torch_device
            return requested_torch_device(value)
        return value

    @model_validator(mode="after")
    def dates(self):
        if (self.account_start is not None or self.account_initial_cash is not None) and self.consensus_version != "czsc_trend_consensus":
            raise ValueError("账户研究参数需要趋势共识策略入口")
        if self.start and self.end and self.start > self.end:
            raise ValueError("开始日期不得晚于结束日期")
        if self.research and self.model_family == "high_return" and self.usage_mode == "retrospective":
            raise ValueError("高收益区间研究只支持前向历史检查点，不能使用事后模型研究历史图")
        if self.model_fold:
            self.model_policy = "pinned"
        if self.model_policy == "pinned" and not self.model_run_id:
            raise ValueError("固定模型需要研究运行标识")
        if self.model_policy == "pinned" and not self.model_fold:
            self.model_fold = "production"
        if self.usage_mode == "retrospective" and self.model_policy != "pinned":
            raise ValueError("事后研究需要明确固定模型检查点")
        if self.entry_policy == "risk" and self.usage_mode == "production" and self.model_family != "high_return":
            raise ValueError("价格与风险入场尚为实验，请使用历史研究模式")
        if not self.research and self.entry_policy != "legacy":
            raise ValueError("新入场计划需要组合研究模式")
        return self


class ReplayHoldingRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    replay_run_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$')
    window: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')
    arm: Literal['reference','full','entry_only','exit_only','ma','structure','wyckoff'] = 'full'
    symbol: str
    as_of: date
    usage_mode: Literal['production','historical'] = 'production'
    _symbol = field_validator('symbol')(clean_symbol)


class TaskSpec(BaseModel):
    symbols: list[str] = Field(default_factory=list, max_length=10000)
    start: date | None = None
    end: date | None = None
    initial_cash: float = Field(default=100000, gt=0, le=1e12, allow_inf_nan=False)
    device: str | None = None
    cpu_workers: int | None = Field(default=None, ge=1, le=16, strict=True)
    batch_size: int | None = Field(default=None, ge=1, le=128, strict=True)
    research: bool = Field(default=False, strict=True)
    model_family: Literal["ma_trend", "dual", "wyckoff", "high_return"] = "ma_trend"
    consensus_config: dict[str, Any] | None = None
    consensus_version: Literal["czsc_agent_consensus_v1", "czsc_agent_consensus_v2", "czsc_agent_consensus_v3", "czsc_agent_consensus_v4", "czsc_trend_consensus"] | None = None

    @model_validator(mode="after")
    def arbitration_parameters(self):
        if self.consensus_config is not None or self.consensus_version is not None:
            from my_strategy.web_dashboard.trend_routes import validate_consensus_config as validate_config
            self.consensus_config = validate_config(self.consensus_config, version=self.consensus_version,
                model_run_id=self.model_run_id, model_family=self.model_family)
        return self
    source_training_run_id: str | None = None
    research_config: dict[str, Any] | None = None
    model_run_id: str | None = None
    model_fold: str | None = None
    calendar_run_id: str | None = None
    usage_mode: Literal["production", "historical", "retrospective"] = "historical"
    model_policy: Literal["auto", "pinned"] = "auto"
    entry_policy: Literal["legacy", "fresh", "risk"] = "legacy"

    @field_validator("model_run_id", "model_fold", "calendar_run_id")
    @classmethod
    def research_identity(cls, value):
        if value is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", value):
            raise ValueError("研究运行标识不合法")
        return value

    @field_validator("source_training_run_id")
    @classmethod
    def source_identity(cls, value):
        if value is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", value):
            raise ValueError("冻结来源运行标识不合法")
        return value

    @field_validator("device")
    @classmethod
    def compute_device(cls, value):
        if value is not None:
            from my_strategy.core.device import requested_torch_device
            return requested_torch_device(value)
        return value

    @field_validator("symbols")
    @classmethod
    def stocks(cls, values):
        return list(dict.fromkeys(clean_symbol(value) for value in values))

    @model_validator(mode="after")
    def dates(self):
        if self.start and self.end and self.start > self.end:
            raise ValueError("开始日期不得晚于结束日期")
        if self.model_fold:
            self.model_policy = "pinned"
        if self.model_policy == "pinned" and self.model_run_id and not self.model_fold:
            self.model_fold = "production"
        if self.entry_policy == "risk" and self.usage_mode == "production" and self.model_family != "high_return":
            raise ValueError("价格与风险入场尚为实验，请使用历史研究模式")
        if not self.research and self.entry_policy != "legacy":
            raise ValueError("新入场计划需要组合研究模式")
        return self


class TaskRequest(BaseModel):
    kind: Literal["scan", "backtest", "update", "research_train", "dual_research_train", "wyckoff_research_train", "high_return_points_build", "high_return_episodes_build", "high_return_train", "high_return_backtest"]
    spec: TaskSpec = Field(default_factory=TaskSpec)

    @model_validator(mode="after")
    def research_start(self):
        if self.spec.model_fold is not None and (self.kind not in {"scan", "backtest", "high_return_backtest"} or not self.spec.research or self.spec.model_run_id is None):
            raise ValueError("模型检查点需要研究模式及模型运行")
        if self.spec.model_policy == "pinned" and not self.spec.model_run_id:
            raise ValueError("固定模型需要研究运行标识")
        if self.spec.usage_mode == "retrospective" and self.kind in {"scan", "backtest", "high_return_backtest"}:
            raise ValueError("事后模型研究仅用于结构分析，不用于策略扫描或回测认证")
        if self.kind == "research_train" and (self.spec.end is None or self.spec.calendar_run_id is None):
            raise ValueError("研究训练需要明确截止日及已核验交易日历运行")
        if self.kind == "dual_research_train" and (self.spec.end is None or self.spec.source_training_run_id is None):
            raise ValueError("双专家研究需要明确截止日及冻结来源运行")
        if self.kind == "dual_research_train":
            self.spec.model_family = "dual"
        if self.kind == "wyckoff_research_train":
            if self.spec.end is None or self.spec.source_training_run_id is None:
                raise ValueError("威科夫研究需要明确截止日及冻结来源运行")
            self.spec.model_family = "wyckoff"
        if self.kind in {"high_return_points_build", "high_return_episodes_build", "high_return_train"}:
            if self.spec.source_training_run_id is None:
                raise ValueError("高收益区间研究需要明确的冻结来源运行")
            self.spec.model_family = "high_return"
            self.spec.research = True
        if self.kind == "high_return_backtest":
            if self.spec.model_run_id is None:
                raise ValueError("高收益区间离线回测需要明确的高收益模型运行")
            if self.spec.model_policy != "auto" or self.spec.model_fold is not None:
                raise ValueError("高收益区间离线回测按日期自动匹配检查点，不能固定检查点")
            if self.spec.usage_mode != "historical":
                raise ValueError("高收益区间离线回测只支持历史滚动研究，尚未取得当前生产资格")
            self.spec.model_family = "high_return"
            self.spec.research = True
        if self.kind in {"scan", "backtest", "high_return_backtest"} and self.spec.start is None:
            self.spec.start = date(2026, 1, 1)
            self.spec.dates()
        if self.kind in {"high_return_points_build", "high_return_episodes_build", "high_return_train", "high_return_backtest"}:
            from my_strategy.web_dashboard.trend_routes import reject_legacy_operation as reject_native_legacy_operation
            reject_native_legacy_operation(self.kind, run_id=self.spec.model_run_id or self.spec.source_training_run_id,
                consensus_version=self.spec.consensus_version, research_config=self.spec.research_config)
        return self


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
    return {"status": "ok", "method": "CZSC", "version": "1.0.1"}


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


@router.get("/research/status")
async def research_status():
    from my_strategy.services.czsc_research import research_status as inspect_research
    return await asyncio.to_thread(inspect_research)


@router.get("/research/dual-status")
async def dual_status():
    from my_strategy.services.czsc_dual_runtime import dual_research_status
    return await asyncio.to_thread(dual_research_status)


@router.get("/research/wyckoff-status")
async def wyckoff_status():
    from my_strategy.services.czsc_wyckoff_runtime import wyckoff_research_status
    return await asyncio.to_thread(wyckoff_research_status)


@router.get("/research/high-return-status")
async def high_return_status():
    from my_strategy.web_dashboard.trend_routes import status_high_return
    return await asyncio.to_thread(status_high_return)


@router.get("/research/high-return-forward-status")
async def high_return_forward_status(model_run_id: str = Query(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,179}$")):
    from my_strategy.web_dashboard.trend_routes import reject_legacy_operation as reject_native_legacy_operation
    reject_native_legacy_operation('forward', run_id=model_run_id)
    from my_strategy.services.czsc_high_return_forward import forward_status
    return await asyncio.to_thread(forward_status, model_run_id)


@router.post('/research/high-return-holding')
async def high_return_holding(spec: ReplayHoldingRequest):
    from my_strategy.web_dashboard.holding_evidence import holding_exit_research
    return await asyncio.to_thread(holding_exit_research, **spec.model_dump(mode='json'))


class ModelPromotionRequest(BaseModel):
    model_run_id: str
    checkpoint: str = "production"
    reason: str = Field(min_length=1, max_length=1000)
    _identity = field_validator("model_run_id", "checkpoint")(AnalysisRequest.identity.__func__)


class ModelRollbackRequest(BaseModel):
    release_id: str | None = None
    reason: str = Field(min_length=1, max_length=1000)


@router.post("/research/promote")
async def promote_model(spec: ModelPromotionRequest, request: Request):
    from my_strategy.storage.czsc_model_releases import ModelReleaseStore
    store = ModelReleaseStore(request.app.state.tasks.results.db_path, request.app.state.tasks.results.runs_root)
    return await asyncio.to_thread(store.promote, spec.model_run_id, spec.checkpoint, reason=spec.reason)


@router.post("/research/rollback")
async def rollback_model(spec: ModelRollbackRequest, request: Request):
    from my_strategy.storage.czsc_model_releases import ModelReleaseStore
    store = ModelReleaseStore(request.app.state.tasks.results.db_path, request.app.state.tasks.results.runs_root)
    return await asyncio.to_thread(store.rollback, spec.release_id, reason=spec.reason)


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
        if spec.research and spec.model_family == "high_return":
            from my_strategy.web_dashboard.high_return import analyze_high_return
            result = analyze_high_return(**calculation_query, config=settings, consensus_config=spec.consensus_config,
                                         consensus_version=spec.consensus_version)
        else:
            result = analyze_stock(**calculation_query, config=settings)
        if "chart_evidence" not in result:
            from my_strategy.web_dashboard.chart_evidence import decorate_analysis
            result = decorate_analysis(result, consensus_config=spec.consensus_config,
                                       consensus_version=spec.consensus_version)
        consensus = result.get("consensus_decisions") or {}
        if consensus.get("config"):
            query["consensus_config"] = consensus["config"]
            query["consensus_version"] = consensus["version"]
        if result.get("strategy_version") == "czsc_trend_consensus":
            research_context = result.get("research", {})
            selected_model = research_context.get("model_run_id")
            if selected_model:
                query["model_run_id"] = selected_model
            if research_context.get("calendar_run_id"):
                query["calendar_run_id"] = research_context["calendar_run_id"]
            account_context = result.get("account_actions", {}).get("context", {})
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
def task(job_id: str, request: Request):
    return request.app.state.tasks.get(job_id)


@router.post("/tasks/{job_id}/cancel")
def cancel(job_id: str, request: Request):
    return request.app.state.tasks.cancel(job_id)


@router.get("/runs")
def runs(request: Request, kind: str | None = None, limit: int = Query(default=30, ge=1, le=200)):
    return {"runs": request.app.state.tasks.results.list(kind, limit)}


@router.get("/runs/{run_id}")
def run(run_id: str, request: Request):
    return request.app.state.tasks.results.get(run_id)


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
