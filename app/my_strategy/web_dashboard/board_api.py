"""Board knowledge routes; GET reads never fetch data or rebuild features."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
import math
import re
import sqlite3
import threading
from typing import Literal
import uuid
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel, Field, ConfigDict

from my_strategy.storage.board_knowledge import BoardKnowledgeStore, _instant
from my_strategy.core.paths import PROJECT_ROOT
from my_strategy.web_dashboard.api import compute_submission, submit_compute

router = APIRouter(prefix='/api/boards', tags=['board-knowledge'])


def _scope_metadata(generation=None, *, explicit=False):
    from my_strategy.services.board_calculation_scope import CALCULATION_SCOPE
    scope = (generation or {}).get('calculation_scope')
    return {'calculation_scope': scope, 'required_calculation_scope': CALCULATION_SCOPE,
        'scope_label': '沪深 A 股' if scope == CALCULATION_SCOPE else '历史原范围' if generation else '沪深 A 股 · 待补齐',
        'historical_view': bool(explicit), 'availability': 'available' if generation else 'pending',
        'needs_rebuild': generation is None}


def _selected_generation(store, generation_id=None):
    """A bad current publication is unavailable; explicit history never falls back."""
    try:
        generation = store.get_generation(generation_id) if generation_id else store.current_generation()
        if generation_id and (not generation or not generation.get('market_ready')):
            raise HTTPException(404, '该板块历史发布版本不存在或尚未完成')
        if generation_id: generation = store.verify_calculation_scope(generation_id)
        return generation
    except ValueError as exc:
        if generation_id: raise HTTPException(409, '该板块版本核验失败，保持不可用') from exc
        return None


def _generation_catalog(store):
    fields = ('generation_id', 'price_date', 'membership_date', 'created_at', 'board_count',
        'market_ready', 'full_ready', 'market_published', 'full_published')
    try:
        rows = store.list_generations(limit=50)
    except ValueError:
        # A damaged history index must not make its entries selectable or trigger a fallback.
        return []
    return [{**{key: row.get(key) for key in fields}, **_scope_metadata(row, explicit=True)} for row in rows]


def _compact_generation_status(generation):
    """Project an already verified publication for the main panel's status line.

    Memberships and original proofs remain available through the default full
    status response and explicit detail routes; this changes only serialization.
    """
    if generation is None:
        return None
    fields = ('generation_id', 'price_date', 'membership_date', 'created_at',
        'board_count', 'stock_count', 'market_ready', 'full_ready',
        'market_published', 'full_published', 'calculation_scope', 'mode',
        'training_eligible', 'training_started', 'capital_basis', 'qualification')
    return {key: generation[key] for key in fields if key in generation}


def _compact_ranking_row(row):
    """Table fields from a fully authenticated ranking and its market row."""
    fields = ('rank', 'board_id', 'name', 'kind', 'calculation_scope', 'activity_score',
        'board_return_1', 'board_amount', 'board_market_amount_share', 'board_relative_amount20',
        'effective_members', 'expected_members', 'coverage', 'generation_id', 'price_date')
    return {key: row[key] for key in fields if key in row}


@contextmanager
def _store(request):
    yield BoardKnowledgeStore(getattr(request.app.state, 'board_db_path', None), readonly=True)


def _fluent_stale(request, row):
    from my_strategy.services.board_fluent import freshness_status
    try:
        state = freshness_status(fluent_db_path=request.app.state.board_fluent_db_path,
            raw_db_path=request.app.state.board_raw_db_path, kb_db_path=request.app.state.board_db_path,
            source_root=PROJECT_ROOT)
    except ValueError:
        return True
    return bool(state['stale'] or state.get('native_stale') or
        row.get('generation_id') != (state.get('market') or {}).get('generation_id'))


def _refresh_summary(result):
    """Keep counts and outcome; source responses remain in the verified database."""
    fields = ('run_id', 'source', 'status', 'requested_boards', 'completed_boards', 'request_count',
        'current_only', 'historical_membership_verified', 'resume', 'resumed_from', 'error',
        'workers', 'requested_workers', 'min_interval', 'reuse_today', 'fresh_boards', 'reused_boards', 'elapsed_seconds')
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
    with compute_submission(request):
        return _start_job_guarded(request, kind, work, summarize)


def _start_job_guarded(request, kind, work, summarize):
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
def status(request: Request, generation_id: str | None = None, compact: bool = False):
    if not getattr(request.app.state, 'board_integrated', False):
        with _store(request) as store: return store.status()
    # Main-panel status is an indexed hint, not a membership eligibility proof.
    with _store(request) as store:
        with store._read() as connection:
            knowledge = {'available': connection is not None, 'current_only': True, 'history_eligible': False}
            if connection:
                knowledge['snapshot_count'] = connection.execute('SELECT COUNT(*) FROM board_snapshots').fetchone()[0]
                knowledge['stock_count'] = connection.execute('SELECT COUNT(DISTINCT symbol) FROM stock_basic').fetchone()[0]
                latest = connection.execute('SELECT run_id,status,finished_at FROM board_refresh_runs ORDER BY started_at DESC,run_id DESC LIMIT 1').fetchone()
                knowledge['latest_run'] = dict(latest) if latest else None
    from my_strategy.services.board_fluent import freshness_status
    from my_strategy.storage.board_fluent import BoardFluentStore
    fluent = BoardFluentStore(request.app.state.board_fluent_db_path, readonly=True)
    generation = _selected_generation(fluent, generation_id)
    if generation_id:
        state = {'available': True, 'market': generation,
            'full': generation if generation.get('full_ready') else None, 'stale': False,
            'stale_reasons': [], 'native_stale': False}
    elif generation is None:
        state = {'available': False, 'market': None, 'full': None, 'stale': True,
            'stale_reasons': ['no_valid_scope_publication'], 'native_stale': False}
    else:
        try:
            state = freshness_status(fluent_db_path=request.app.state.board_fluent_db_path,
                raw_db_path=request.app.state.board_raw_db_path, kb_db_path=request.app.state.board_db_path,
                source_root=PROJECT_ROOT)
            generation = state.get('market')
        except ValueError:
            generation = None
            state = {'available': False, 'market': None, 'full': None, 'stale': True,
                'stale_reasons': ['invalid_scope_publication'], 'native_stale': False}
    scope = _scope_metadata(generation, explicit=bool(generation_id))
    result = {**knowledge, 'fluent': {**state, **scope,
        'needs_rebuild': bool(state.get('stale') or scope['needs_rebuild'])},
        'generations': _generation_catalog(fluent)}
    if compact:
        # Keep selection, scope/SHA authentication, freshness and history checks
        # above identical to the default route. Only the final response shrinks.
        result['fluent'] = {**result['fluent'], **{
            key: _compact_generation_status(state.get(key)) for key in ('market', 'full')}}
    return result


@router.get('/generations')
def generations(request: Request):
    from my_strategy.storage.board_fluent import BoardFluentStore
    store = BoardFluentStore(request.app.state.board_fluent_db_path, readonly=True)
    current = _selected_generation(store)
    return {'generations': _generation_catalog(store), 'current_generation_id': (current or {}).get('generation_id'),
        **_scope_metadata(current)}


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
                'current': current, 'as_of': as_of, 'source_evidence_only': True,
                'membership_basis': 'immutable_source_snapshot'}
        except ValueError as exc: raise HTTPException(422, str(exc)) from exc


@router.get('/stocks/{symbol}')
def stock_boards(symbol: str, request: Request, current: bool = True, as_of: str | None = None,
                 generation_id: str | None = None):
    if generation_id and not current: raise HTTPException(422, '历史发布版本不能与因果日期查询混用')
    if getattr(request.app.state, 'board_integrated', False) and current:
        if as_of is not None: raise HTTPException(422, '当前成员与历史 as_of 不能混用')
        from my_strategy.storage.board_knowledge import normalize_stock
        from my_strategy.storage.board_fluent import BoardFluentStore
        try: symbol = normalize_stock(symbol.upper())
        except ValueError as exc: raise HTTPException(422, str(exc)) from exc
        fluent = BoardFluentStore(request.app.state.board_fluent_db_path, readonly=True)
        generation = _selected_generation(fluent, generation_id)
        row = fluent.stock(symbol, generation_id=generation['generation_id']) if generation else None
        identifiers = set((row or {}).get('board_ids', (row or {}).get('concept_ids', [])))
        if row and row.get('industry_id'): identifiers.add(row['industry_id'])
        boards = [fluent.board(identifier, generation_id=row['generation_id'], include_structure=False) for identifier in sorted(identifiers)]
        return {'symbol': symbol, 'boards': [b for b in boards if b], 'features': row,
            **_scope_metadata(generation, explicit=bool(generation_id)),
            'needs_rebuild': row is None or (not generation_id and _fluent_stale(request, row)),
            'current': not bool(generation_id), 'feature_reference': 'basic_fluent', 'price_date_is_frozen': True}
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
def board_features(board_id: str, request: Request, date: str | None = None, current: bool = True,
                   generation_id: str | None = None):
    if generation_id and not current: raise HTTPException(422, '历史发布版本不能与因果日期查询混用')
    if getattr(request.app.state, 'board_integrated', False) and current:
        from my_strategy.storage.board_fluent import BoardFluentStore
        store = BoardFluentStore(request.app.state.board_fluent_db_path, readonly=True)
        generation = _selected_generation(store, generation_id)
        row = store.board(board_id, generation_id=generation['generation_id']) if generation else None
        if date and row and row['price_date'] != _feature_cutoff(date)[1]: row = None
        return {'board_id': board_id, 'features': row, 'available': row is not None,
            **_scope_metadata(generation, explicit=bool(generation_id)),
            'needs_rebuild': row is None or (not generation_id and _fluent_stale(request, row)),
            'feature_reference': 'basic_fluent', 'price_date_is_frozen': True}
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
    workers: int = Field(default=4, ge=1, le=8, strict=True)
    min_interval: float = Field(default=1.3, ge=0, allow_inf_nan=False)
    reuse_today: bool = Field(default=True, strict=True)


@router.post('/refresh', status_code=202)
async def refresh(spec: RefreshSpec, request: Request):
    if not spec.kinds or len(set(spec.kinds)) != len(spec.kinds) or any(kind not in ('industry', 'concept') for kind in spec.kinds):
        raise HTTPException(422, '请选择唯一的行业或概念类型')
    if getattr(request.app.state, 'board_integrated', False):
        return submit_compute(request, 'legacy', 'board_knowledge', spec.model_dump())
    def work(progress):
        from my_strategy.scripts.refresh_board_knowledge import refresh_knowledge
        return refresh_knowledge(db_path=getattr(request.app.state, 'board_db_path', None),
            raw_db_path=getattr(request.app.state, 'board_raw_db_path', None), kinds=spec.kinds,
            boards=spec.boards or None, resume=spec.resume, limit_budget=spec.limit_budget,
            workers=spec.workers, min_interval=spec.min_interval, reuse_today=spec.reuse_today, on_progress=progress)
    return _start_job(request, 'refresh', work, _refresh_summary)


class FeatureBuildSpec(BaseModel):
    mode: Literal['current_reference'] = 'current_reference'
    price_date: str | None = None
    symbols: list[str] | None = None
    with_native: bool = True


@router.post('/features/build', status_code=202)
async def build_features(spec: FeatureBuildSpec, request: Request):
    if getattr(request.app.state, 'board_integrated', False):
        if spec.symbols: raise HTTPException(422, '板块发布须完整股票池，不能限定个股')
        return submit_compute(request, 'legacy', 'board_features', spec.model_dump(exclude={'symbols'}))
    def work(progress):
        from my_strategy.services.czsc_trend_support import _calendar
        from my_strategy.services.board_fluent import latest_qualified_price_date
        from my_strategy.services.board_features import build_board_features
        calendar, calendar_id = _calendar(None)
        raw_path = getattr(request.app.state, 'board_raw_db_path', None)
        day = spec.price_date
        if day is None:
            day = latest_qualified_price_date(raw_path)
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
    if getattr(request.app.state, 'board_integrated', False):
        result = request.app.state.tasks.get(job_id)
        if not result['kind'].startswith('board_'): raise HTTPException(404, '找不到板块任务')
        return result
    result = request.app.state.board_jobs.get(job_id)
    if result is None: raise HTTPException(404, '找不到该板块任务，请查询知识库更新记录')
    return dict(result)


def configure_board_routes(app, *, db_path=None, raw_db_path=None, fluent_db_path=None, integrated=False):
    from my_strategy.core.paths import BASIC_FLUENT_DB_PATH
    app.state.board_integrated = integrated
    if not hasattr(app.state, 'compute_submit_lock'):
        app.state.compute_submit_lock = threading.Lock()
    app.state.board_fluent_db_path = Path(fluent_db_path) if fluent_db_path else BASIC_FLUENT_DB_PATH
    app.state.board_db_path = Path(db_path) if db_path else None
    app.state.board_raw_db_path = Path(raw_db_path) if raw_db_path else None
    app.state.board_job_lock = threading.Lock()
    app.state.board_jobs = {}
    app.state.board_background_tasks = set()
    app.include_router(router)


@router.get('/rankings')
def rankings(request: Request, kind: Literal['industry', 'concept'] = 'industry',
             ranking_type: Literal['activity', 'capital'] = 'activity', limit: int = 30,
             generation_id: str | None = None, compact: bool = False):
    from my_strategy.storage.board_fluent import BoardFluentStore
    store = BoardFluentStore(request.app.state.board_fluent_db_path, readonly=True)
    generation = _selected_generation(store, generation_id)
    rows = store.rankings(kind, ranking_type, min(30, max(1, limit)), generation_id=generation['generation_id']) if generation else []
    if compact:
        rows = [_compact_ranking_row(row) for row in rows]
    return {'rankings': rows, 'generation_id': (generation or {}).get('generation_id'),
        'price_date': (generation or {}).get('price_date'), 'training_eligible': False,
        'capital_basis': 'turnover_amount_proxy', **_scope_metadata(generation, explicit=bool(generation_id))}


@router.get('/capital-visualization')
def capital_visualization(request: Request, kind: Literal['industry', 'concept'] = 'industry',
                          sort_by: Literal['amount', 'change', 'relative'] = 'amount',
                          limit: int = Query(default=30, ge=5, le=60), generation_id: str | None = None):
    """Visualize stored rows only; no price reads, rebuilds or network fetches."""
    from my_strategy.storage.board_fluent import BoardFluentStore
    store=BoardFluentStore(request.app.state.board_fluent_db_path,readonly=True)
    generation=_selected_generation(store,generation_id)
    population=store.boards(kind,generation['generation_id']) if generation else []
    def finite(value):
        return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    fields=('board_id','name','kind','board_amount','board_return_1','board_market_amount_share',
        'board_relative_amount20','coverage','expected_members','effective_members','price_date',
        'membership_date','snapshot_hash','generation_id','capital_visualization','calculation_scope',
        'calculation_membership')
    available=[row for row in population if finite(row.get('board_amount')) and row['board_amount']>=0]
    def change(row):return (row.get('capital_visualization') or {}).get('amount_change_1')
    def rank_value(row):
        if sort_by=='change':return abs(change(row)) if finite(change(row)) else None
        return row.get('board_relative_amount20') if sort_by=='relative' else row.get('board_amount')
    ranked=[row for row in available if finite(rank_value(row))]
    ranked=sorted(ranked,key=lambda row:(-rank_value(row),row['board_id']))[:limit]
    increases=sorted([row for row in available if finite(change(row)) and change(row)>0],
        key=lambda row:(-change(row),row['board_id']))[:8]
    decreases=sorted([row for row in available if finite(change(row)) and change(row)<0],
        key=lambda row:(change(row),row['board_id']))[:8]
    def public(row):return {key:row.get(key) for key in fields}
    prepared=sum(bool((row.get('capital_visualization') or {}).get('history')) for row in available)
    market_amount=next((row['capital_visualization']['history'][-1].get('market_amount')
        for row in available if (row.get('capital_visualization') or {}).get('history')),None)
    return {'rows':[public(row) for row in ranked],
        'increases':[public(row) for row in increases],'decreases':[public(row) for row in decreases],
        'generation_id':(generation or {}).get('generation_id'),'price_date':(generation or {}).get('price_date'),
        'membership_date':(generation or {}).get('membership_date'),'kind':kind,'sort_by':sort_by,
        'board_count':len(population),'available_board_count':len(available),'prepared_board_count':prepared,
        **_scope_metadata(generation, explicit=bool(generation_id)),
        'market_amount':market_amount,'needs_rebuild':generation is None or bool(available and prepared<len(available)),
        'capital_basis':'turnover_amount_proxy','training_eligible':False,
        'membership_basis':'fixed_current_snapshot_historical_price_comparison',
        'comparison_basis':'same_verified_members_on_both_days',
        'overlapping_boards':True}


@router.get('/capital-history/{board_id}')
def capital_history(board_id: str, request: Request, window: Literal['90', '50', '20'] = '90',
                    generation_id: str | None = None):
    """Display-only extension, pinned to an unchanged publication and raw witness."""
    from my_strategy.storage.board_fluent import BoardFluentStore
    from my_strategy.services.board_capital_history import capital_history as read_history
    store = BoardFluentStore(request.app.state.board_fluent_db_path, readonly=True)
    generation = _selected_generation(store, generation_id)
    if generation is None:
        raise HTTPException(404, '尚无可读取的板块发布版本')
    row = store.board(board_id, generation['generation_id'])
    if row is None:
        raise HTTPException(404, '该发布版本没有此板块')
    try:
        return read_history(generation=generation, board=row,
            raw_db_path=request.app.state.board_raw_db_path, window=int(window))
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get('/detail/{board_id}')
def detail(board_id: str, request: Request, generation_id: str | None = None):
    from my_strategy.storage.board_fluent import BoardFluentStore
    from my_strategy.services.board_calculation_scope import CALCULATION_SCOPE, verify_calculation_membership
    fluent = BoardFluentStore(request.app.state.board_fluent_db_path, readonly=True)
    generation = _selected_generation(fluent, generation_id)
    if generation is None:
        return {'features': None, 'members': [], 'membership_matches_generation': False, **_scope_metadata()}
    row = fluent.board(board_id, generation['generation_id'])
    if row is None: raise HTTPException(404, '该发布版本没有此板块')
    with _store(request) as store:
        # Exact stored membership SHA ties details to the selected immutable generation.
        with store._read() as connection:
            snapshot_row = connection.execute('SELECT * FROM board_snapshots WHERE snapshot_hash=?',
                (row['snapshot_hash'],)).fetchone() if connection else None
            snapshot = store._verified_snapshot(connection, snapshot_row) if snapshot_row else None
    projection = None
    if row.get('calculation_scope') == CALCULATION_SCOPE:
        try:
            projection = verify_calculation_membership(row.get('calculation_membership'), snapshot=snapshot)
            if projection['source_snapshot_hash'] != row['snapshot_hash'] or projection['board_id'] != board_id:
                raise ValueError('projection does not match stored row')
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(409, '该版本的计算成员投影核验失败，保持不可用') from exc
    return {'features': row, 'members': projection['eligible_members'] if projection else snapshot['members'] if snapshot else [],
        'membership_matches_generation': projection is not None or snapshot is not None,
        'source_members': snapshot['members'] if snapshot else [], 'source_membership_available': snapshot is not None,
        'calculation_membership': projection, **_scope_metadata(generation, explicit=bool(generation_id))}


class SyncSpec(BaseModel):
    scope: Literal['full'] | None = None
    resume_sync_id: str | None = None
    cpu_workers: int = Field(default=1, ge=1, le=2)
    stock_batch_size: int = Field(default=360, ge=1, le=2000)


@router.post('/sync', status_code=202)
def sync(spec: SyncSpec, request: Request):
    if not getattr(request.app.state, 'board_integrated', False): raise HTTPException(409, '请从主站发起同步')
    if spec.scope!='full':raise HTTPException(409,'请先预览“补齐结构与特征”差异计划；完整范围必须显式选择 full')
    return submit_compute(request, 'legacy', 'board_sync', spec.model_dump(exclude={'scope'}))


class IncrementalPlanSpec(BaseModel):
    model_config=ConfigDict(extra='forbid')
    scope: Literal['current_board','selected','watch','held','full']='watch'
    board_id: str | None = None
    symbols: list[str] | None = Field(default=None,max_length=50)
    price_date: str | None = None
    max_symbols: int = Field(default=20,ge=1,le=100)
    budget_seconds: int = Field(default=300,ge=1,le=1800)


class IncrementalExecutionSpec(BaseModel):
    model_config=ConfigDict(extra='forbid')
    plan_id: str = Field(pattern=r'^[a-f0-9]{32}$')


@router.post('/incremental/plan')
def incremental_plan(spec:IncrementalPlanSpec,request:Request):
    from my_strategy.services.incremental_sync import create_plan
    if not hasattr(request.app.state,'light'):raise HTTPException(409,'请从主站预览补齐计划')
    try:
        return create_plan(store=request.app.state.light.store,raw_db_path=request.app.state.board_raw_db_path,
            kb_db_path=request.app.state.board_db_path,fluent_db_path=request.app.state.board_fluent_db_path,
            **spec.model_dump())
    except (ValueError,KeyError,OSError) as exc:raise HTTPException(409,str(exc)) from exc


@router.get('/incremental/plans/{plan_id}')
def incremental_plan_status(plan_id:str,request:Request):
    from my_strategy.services.incremental_sync import get_plan
    if not hasattr(request.app.state,'light'):raise HTTPException(409,'请从主站查看补齐计划')
    try:return get_plan(store=request.app.state.light.store,plan_id=plan_id)
    except KeyError as exc:raise HTTPException(404,str(exc)) from exc


@router.post('/incremental/execute',status_code=202)
def incremental_execute(spec:IncrementalExecutionSpec,request:Request):
    from my_strategy.services.incremental_sync import get_plan
    if not hasattr(request.app.state,'light'):raise HTTPException(409,'请从主站执行补齐计划')
    try:
        plan=get_plan(store=request.app.state.light.store,plan_id=spec.plan_id)
        if plan['status']=='cancelled':raise ValueError('取消的计划不可恢复，请重新预览缺口')
    except (KeyError,ValueError) as exc:raise HTTPException(409,str(exc)) from exc
    return submit_compute(request,'light','incremental',{'plan_id':spec.plan_id,
        'kb_db_path':str(request.app.state.board_db_path) if request.app.state.board_db_path else None,
        'fluent_db_path':str(request.app.state.board_fluent_db_path)})


@router.get('/jobs')
def jobs(request: Request):
    if getattr(request.app.state, 'board_integrated', False):
        return {'jobs': [j for j in request.app.state.tasks.list() if j['kind'].startswith('board_')]}
    return {'jobs': list(request.app.state.board_jobs.values())}


@router.post('/jobs/{job_id}/cancel')
def cancel(job_id: str, request: Request):
    if not getattr(request.app.state, 'board_integrated', False): raise HTTPException(409, '请从主站管理同步')
    result = request.app.state.tasks.get(job_id)
    if not result['kind'].startswith('board_'): raise HTTPException(404, '找不到板块任务')
    return request.app.state.tasks.cancel(job_id)


# These display endpoints read complete saved method reports. They do not invoke
# board feature/native producers, initialize an index or start a compute task.
_Method = Literal['structure_kronos', 'structure_kronos_ma_mlp', 'structure_kronos_ma_l2']
_MethodCategory = Literal['all', 'buy', 'sell', 'wait', 'unavailable', 'failed', 'deferred', 'not_in_scan']
_method_readers_lock = threading.Lock()


def _method_reader(request):
    from my_strategy.services.board_method_summary import BoardMethodSummary
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, PROCESSED_DATA_ROOT
    manager = getattr(request.app.state, 'tasks', None)
    results = getattr(manager, 'results', None)
    db = getattr(results, 'db_path', PROCESSED_DATA_ROOT / 'czsc' / 'results.db')
    runs = getattr(results, 'runs_root', ARTIFACT_RUNS_ROOT)
    kb = getattr(request.app.state, 'board_db_path', None)
    key = (str(db), str(runs), str(kb))
    with _method_readers_lock:
        existing = getattr(request.app.state, 'board_method_reader', None)
        if existing is None or existing[0] != key:
            existing = (key, BoardMethodSummary(db, runs, kb))
            request.app.state.board_method_reader = existing
        return existing[1]


def _method_read(call):
    try:
        return call()
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, {'code': 'saved_method_result_missing', 'message': '所选保存结果或板块成员不存在'}) from exc
    except (ValueError, TypeError, sqlite3.DatabaseError, OSError) as exc:
        raise HTTPException(409, {'code': 'saved_method_identity_invalid', 'message': '保存结果或成员身份核验失败，未执行计算'}) from exc


def _method_symbols(value):
    if value is None:
        return None
    parts = value.split(',') if value else []
    if any(re.fullmatch(r'(?:6\d{5}\.SH|[03]\d{5}\.SZ)', item) is None for item in parts):
        raise HTTPException(422, '股票筛选必须使用规范沪深 A 股代码')
    return set(parts)


@router.get('/method-scans')
def method_scans(request: Request, method: _Method, as_of: str = Query(pattern=r'^\d{4}-\d{2}-\d{2}$'),
                 page: int = Query(default=1, ge=1), page_size: int = Query(default=20, ge=1, le=100)):
    return _method_read(lambda: _method_reader(request).list_runs(method=method, as_of=as_of, page=page, page_size=page_size))


@router.get('/method-scans/{run_id}/stocks')
def method_scan_stocks(request: Request, run_id: str, method: _Method,
                       as_of: str = Query(pattern=r'^\d{4}-\d{2}-\d{2}$'), page: int = Query(default=1, ge=1),
                       page_size: int = Query(default=40, ge=1, le=100), q: str = Query(default='', max_length=200),
                       category: _MethodCategory = 'all', symbols: str | None = Query(default=None, max_length=120000)):
    selected = _method_symbols(symbols)
    return _method_read(lambda: _method_reader(request).stocks(run_id, method=method, as_of=as_of,
        page=page, page_size=page_size, query=q, category=category, symbols=selected))


@router.get('/method-scans/{run_id}/members')
def method_scan_members(request: Request, run_id: str, method: _Method, board_id: str = Query(min_length=1, max_length=100),
                        as_of: str = Query(pattern=r'^\d{4}-\d{2}-\d{2}$'), page: int = Query(default=1, ge=1),
                        page_size: int = Query(default=40, ge=1, le=100), q: str = Query(default='', max_length=200),
                        category: _MethodCategory = 'all', symbols: str | None = Query(default=None, max_length=120000)):
    selected = _method_symbols(symbols)
    return _method_read(lambda: _method_reader(request).summary(run_id, method=method, as_of=as_of,
        board_id=board_id, page=page, page_size=page_size, query=q, category=category, symbols=selected))


@router.get('/method-scans/{run_id}')
def method_scan_summary(request: Request, run_id: str, method: _Method,
                        as_of: str = Query(pattern=r'^\d{4}-\d{2}-\d{2}$')):
    return _method_read(lambda: _method_reader(request).summary(run_id, method=method, as_of=as_of))
