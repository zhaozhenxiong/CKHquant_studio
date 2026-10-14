"""Board knowledge routes; GET reads never fetch data or rebuild features."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
import re
import threading
from typing import Literal
import uuid
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from my_strategy.storage.board_knowledge import BoardKnowledgeStore, _instant

router = APIRouter(prefix='/api/boards', tags=['board-knowledge'])


@contextmanager
def _store(request):
    yield BoardKnowledgeStore(getattr(request.app.state, 'board_db_path', None), readonly=True)


def _refresh_summary(result):
    """Keep counts and outcome; source responses remain in the verified database."""
    fields = ('run_id', 'source', 'status', 'requested_boards', 'completed_boards', 'request_count',
        'current_only', 'historical_membership_verified', 'resume', 'resumed_from', 'error')
    summary = {key: result[key] for key in fields if key in result}
    summary['failed_board_count'] = len(result.get('failed_board_ids', []))
    summary['missing_board_count'] = len(result.get('missing_board_ids', []))
    failures = result.get('failures') or []
    summary['source_failure_count'] = sum(item.get('board_id') is None for item in failures)
    if result.get('status') in {'partial', 'failed', 'interrupted'} and not summary.get('error') and failures:
        summary['error'] = failures[-1].get('error')
    if isinstance(result.get('stock_basic'), dict):
        summary['stock_basic'] = {key: result['stock_basic'][key] for key in ('records', 'inserted', 'stock_basic_hash')
            if key in result['stock_basic']}
    return summary


def _feature_summary(result):
    fields = ('mode', 'price_date', 'membership_date', 'training_eligible', 'qualification', 'schema_hash',
        'cache_key', 'available_at', 'historical_membership_status', 'price_basis', 'calendar_run_id')
    summary = {key: result[key] for key in fields if key in result}
    summary.update(board_count=len(result.get('boards', [])), stock_count=len(result.get('stocks', [])))
    return summary


def _feature_cutoff(value):
    cutoff = _instant(value, cutoff=True)
    return cutoff, cutoff.astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()


def _compatible_feature(row, store, cutoff=None):
    """Validate the stored artifact contract without reading prices or rebuilding."""
    from my_strategy.services.board_features import (BOARD_FEATURE_PROFILE, BOARD_SCHEMA_HASH,
        DEFAULT_BOARD_POLICY, _code_hashes, stable_hash)
    if not isinstance(row, dict): return False
    try:
        available_at = _instant(row.get('available_at'))
        _, day = _feature_cutoff(row.get('price_date'))
    except ValueError: return False
    dependencies = row.get('membership_dependency_hashes')
    return (row.get('profile') == BOARD_FEATURE_PROFILE['name'] and
        row.get('schema_hash') == BOARD_SCHEMA_HASH and
        row.get('policy_hash') == stable_hash(DEFAULT_BOARD_POLICY) and
        row.get('code_hashes') == _code_hashes(store) and
        all(isinstance(row.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', row[key])
            for key in ('cache_key', 'price_content_sha256', 'calendar_hash')) and
        isinstance(dependencies, list) and bool(dependencies) and
        all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) for value in dependencies) and
        row.get('price_date') == day and (cutoff is None or available_at <= cutoff))


def _start_job(request, kind, work, summarize):
    lock = request.app.state.board_job_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(409, '板块更新或特征计算正在运行')
    jobs = request.app.state.board_jobs
    job_id = uuid.uuid4().hex
    jobs[job_id] = {'job_id': job_id, 'kind': kind, 'status': 'running', 'progress': None}
    def progress(value):
        if isinstance(value, dict):
            jobs[job_id]['progress'] = {key: value[key] for key in ('run_id', 'status', 'completed', 'total', 'request_count', 'failed')
                if key in value}
    async def run():
        try:
            result = await asyncio.to_thread(work, progress)
            outcome = result.get('status', 'completed')
            if outcome not in {'completed', 'succeeded', 'partial', 'failed', 'interrupted'}:
                outcome = 'completed'
            summary = summarize(result)
            jobs[job_id].update(status=outcome, result=summary)
            if outcome in {'partial', 'failed', 'interrupted'} and summary.get('error'):
                jobs[job_id]['error'] = summary['error']
        except Exception as exc:
            jobs[job_id].update(status='failed', error=str(exc))
        finally:
            lock.release()
    task = asyncio.create_task(run())
    request.app.state.board_background_tasks.add(task)
    task.add_done_callback(request.app.state.board_background_tasks.discard)
    return dict(jobs[job_id])


@router.get('/status')
def status(request: Request):
    with _store(request) as store: return store.status()


@router.get('/catalog')
def catalog(request: Request, kind: str | None = None, current: bool = True, as_of: str | None = None):
    if kind not in (None, 'industry', 'concept'):
        raise HTTPException(422, '板块类型须为 industry 或 concept')
    with _store(request) as store:
        try:
            rows = store.catalog(kind=kind, as_of=as_of, current=current)
            fields = ('board_id', 'name', 'kind', 'source', 'taxonomy', 'observed_at', 'available_at',
                'snapshot_hash', 'expected_count', 'membership_scope', 'source_member_count', 'excluded_source_members')
            return {'boards': [{**{key: row.get(key) for key in fields}, 'member_count': len(row['members'])} for row in rows],
                'current': current, 'as_of': as_of}
        except ValueError as exc: raise HTTPException(422, str(exc)) from exc


@router.get('/stocks/{symbol}')
def stock_boards(symbol: str, request: Request, current: bool = True, as_of: str | None = None):
    with _store(request) as store:
        try:
            result = store.stock_boards(symbol.upper(), as_of=as_of, current=current)
            cutoff, day = _feature_cutoff(as_of) if not current and as_of else (None, None)
            features = store.latest_feature('stock', symbol=symbol.upper(),
                mode='current_reference' if current else 'causal', date=day)
            hashes = sorted(board['snapshot_hash'] for board in result['boards'])
            matches = (_compatible_feature(features, store, cutoff) and isinstance(features.get('snapshot_hashes'), list)
                and sorted(features['snapshot_hashes']) == hashes)
            result['features'] = features if matches else None
            result['needs_rebuild'] = bool(hashes) and not matches or features is not None and not matches
            result['feature_reference'] = 'stored_artifact'
            result['price_date_is_frozen'] = True
            return result
        except ValueError as exc: raise HTTPException(422, str(exc)) from exc


@router.get('/features/{board_id}')
def board_features(board_id: str, request: Request, date: str | None = None, current: bool = True):
    with _store(request) as store:
        try:
            cutoff, day = _feature_cutoff(date) if date else (None, None)
            features = store.latest_feature('daily', board_id=board_id, date=day,
                mode='current_reference' if current else 'causal')
            if not current and cutoff is None and _compatible_feature(features, store):
                cutoff, _ = _feature_cutoff(features['price_date'])
            snapshots = store.catalog(current=current, as_of=None if current else cutoff.isoformat()) if current or cutoff else []
            snapshot = next((item for item in snapshots if item['board_id'] == board_id), None)
            matches = (_compatible_feature(features, store, None if current else cutoff) and snapshot is not None and
                features.get('snapshot_hash') == snapshot['snapshot_hash'])
            return {'board_id': board_id, 'features': features if matches else None, 'available': matches,
                'needs_rebuild': snapshot is not None and not matches or features is not None and not matches,
                'feature_reference': 'stored_artifact', 'price_date_is_frozen': True}
        except ValueError as exc: raise HTTPException(422, str(exc)) from exc


class RefreshSpec(BaseModel):
    kinds: list[str] = Field(default_factory=lambda: ['industry', 'concept'])
    boards: list[str] = Field(default_factory=list)
    resume: str | None = None
    limit_budget: int = Field(default=2000, ge=1)


@router.post('/refresh', status_code=202)
async def refresh(spec: RefreshSpec, request: Request):
    if not spec.kinds or len(set(spec.kinds)) != len(spec.kinds) or any(kind not in ('industry', 'concept') for kind in spec.kinds):
        raise HTTPException(422, '请选择唯一的行业或概念类型')
    def work(progress):
        from my_strategy.scripts.refresh_board_knowledge import refresh_knowledge
        return refresh_knowledge(db_path=getattr(request.app.state, 'board_db_path', None),
            raw_db_path=getattr(request.app.state, 'board_raw_db_path', None), kinds=spec.kinds,
            boards=spec.boards or None, resume=spec.resume, limit_budget=spec.limit_budget, on_progress=progress)
    return _start_job(request, 'refresh', work, _refresh_summary)


class FeatureBuildSpec(BaseModel):
    mode: Literal['current_reference'] = 'current_reference'
    price_date: str | None = None
    symbols: list[str] | None = None


@router.post('/features/build', status_code=202)
async def build_features(spec: FeatureBuildSpec, request: Request):
    def work(progress):
        from my_strategy.adapters.czsc_adapter import _connection
        from my_strategy.services.czsc_trend_support import _calendar, VERIFIED_SOURCES
        from my_strategy.services.board_features import build_board_features
        calendar, calendar_id = _calendar(None)
        raw_path = getattr(request.app.state, 'board_raw_db_path', None)
        day = spec.price_date
        if day is None:
            with _connection(raw_path) as connection:
                day = connection.execute('SELECT MAX(date) FROM stock_daily_normalized WHERE has_trade_price=1 AND source IN (' +
                    ','.join('?' for _ in VERIFIED_SOURCES) + ')', sorted(VERIFIED_SOURCES)).fetchone()[0]
        if not day: raise ValueError('没有已核验的本地行情日期')
        if day not in calendar: raise ValueError('行情日期不在已核验交易日历中')
        progress({'status': 'running', 'completed': 0, 'total': 1})
        store = BoardKnowledgeStore(getattr(request.app.state, 'board_db_path', None), readonly=False)
        result = build_board_features(store, raw_db_path=raw_path, calendar_dates=calendar, price_date=day,
            mode=spec.mode, symbols=spec.symbols)
        progress({'status': 'completed', 'completed': 1, 'total': 1})
        return {**result, 'calendar_run_id': calendar_id}
    return _start_job(request, 'build_features', work, _feature_summary)


@router.get('/jobs/{job_id}')
def job(job_id: str, request: Request):
    result = request.app.state.board_jobs.get(job_id)
    if result is None: raise HTTPException(404, '找不到该板块任务，请查询知识库更新记录')
    return dict(result)


def configure_board_routes(app, *, db_path=None, raw_db_path=None):
    app.state.board_db_path = Path(db_path) if db_path else None
    app.state.board_raw_db_path = Path(raw_db_path) if raw_db_path else None
    app.state.board_job_lock = threading.Lock()
    app.state.board_jobs = {}
    app.state.board_background_tasks = set()
    app.include_router(router)
