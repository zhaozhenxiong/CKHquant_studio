"""Refresh current THS board knowledge through Fuyao or verified JSON export.

A frozen refresh can resume after an explicit request budget or interruption.
Finished manifests remain immutable; their remaining boards form a new run.
No price table, historical membership, model, or training resource is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import json
import math
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo

from my_strategy.core.tz import local_now
from my_strategy.data_manager.board_provider import (FuyaoBoardProvider, BoardAccessError,
    BoardBudgetExhausted, BoardSourceError, SOURCE, KINDS)
from my_strategy.storage.board_knowledge import BoardKnowledgeStore


def _ordered_snapshots(pending, provider, workers, check_cancel):
    """Bound both outstanding work and completed buffers; write on caller only."""
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='board-source')
    jobs = {}; next_index = 0
    def submit():
        nonlocal next_index
        if next_index < len(pending):
            jobs[next_index] = pool.submit(provider.snapshot, pending[next_index]); next_index += 1
    try:
        for _ in range(min(workers, len(pending))): submit()
        for index, board in enumerate(pending):
            future = jobs.pop(index)
            while True:
                if check_cancel: check_cancel()
                try: value = future.result(timeout=.05); error = None; break
                except FutureTimeout as exc:
                    if not future.done(): continue
                    value = None; error = exc; break
                except Exception as exc: value = None; error = exc; break
            # Deterministic frozen catalog order, irrespective of completion order.
            yield board, value, error
            submit()
    finally:
        if hasattr(provider, 'stop'): provider.stop()
        for future in jobs.values(): future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def _same_day_candidates(store, requested):
    today = local_now().date()
    by_id = {item['board_id']: item for item in store.catalog(current=True)}
    result = {}
    for board in requested:
        value = by_id.get(board['board_id'])
        if not value: continue
        from datetime import datetime
        observed = datetime.fromisoformat(value['observed_at']).astimezone(ZoneInfo('Asia/Shanghai')).date()
        old_proof = value.get('provenance', {}).get('catalog_provenance', {})
        new_proof = board.get('catalog_provenance', {})
        if (observed == today and value['source'] == SOURCE and
                all(value.get(key) == board.get(key) for key in ('source_code', 'name', 'kind', 'source', 'taxonomy')) and
                old_proof.get('catalog_set_hash') and old_proof['catalog_set_hash'] == new_proof.get('catalog_set_hash')):
            result[board['board_id']] = value
    return result


def _read_stock_basic(raw_db_path):
    from datetime import datetime
    from my_strategy.adapters.czsc_adapter import raw_path
    path = Path(raw_db_path or raw_path()).resolve()
    if not path.is_file(): raise FileNotFoundError('stock basic raw database is missing')
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'stock_basic' in tables:
            table = 'stock_basic'
            columns = {row[1] for row in connection.execute('PRAGMA table_info(stock_basic)')}
            if 'code' not in columns or not {'name', 'stock_name'} & columns:
                raise ValueError('raw stock basic code/name schema is missing')
            names = [name for name in ('name', 'stock_name') if name in columns]
            name_sql = "COALESCE(" + ','.join("NULLIF(" + name + ",'')" for name in names) + ",'')"
            extras = [name for name in ('updated_at', 'source', 'payload_json') if name in columns]
            suffix = '' if not extras else ',' + ','.join(extras)
            values = connection.execute('SELECT code,' + name_sql + ' AS name' + suffix + ' FROM stock_basic ORDER BY code').fetchall()
        elif 'securities' in tables:
            table = 'securities'
            columns = {row[1] for row in connection.execute('PRAGMA table_info(securities)')}
            if not {'stock', 'name'} <= columns: raise ValueError('raw securities code/name schema is missing')
            extras = [name for name in ('updated_at', 'source', 'source_file') if name in columns]
            suffix = '' if not extras else ',' + ','.join(extras)
            values = connection.execute('SELECT stock AS code,name' + suffix + ' FROM securities ORDER BY stock').fetchall()
        else: raise ValueError('raw stock basic or securities catalog is missing')
    if not values: raise ValueError('raw stock basic catalog is empty')
    records = []
    for row in values:
        original = dict(row)
        record = {'code': original.pop('code'), 'name': original.pop('name'),
            'source': original.get('source') or 'raw_sqlite:' + table, 'raw_metadata': original}
        updated = original.get('updated_at')
        if updated:
            record['source_updated_at'] = updated
            try:
                stamp = datetime.fromisoformat(str(updated).replace('Z', '+00:00'))
                if stamp.tzinfo is not None: record['updated_at'] = stamp.isoformat()
            except ValueError: pass
        records.append(record)
    return records


def _load_import(path):
    def reject(_): raise ValueError('board import contains nonfinite JSON')
    value = json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)
    snapshots = value if isinstance(value, list) else value.get('snapshots') if isinstance(value, dict) and 'snapshots' in value else [value]
    if not isinstance(snapshots, list) or not snapshots or any(not isinstance(item, dict) for item in snapshots):
        raise ValueError('board import must contain identified snapshot objects')
    sources = {item.get('source') for item in snapshots}
    if len(sources) != 1 or sources.pop() not in {'fuyao:ths', 'tonghuashun_public'}:
        raise ValueError('board import must use one identified THS source')
    ids = [item.get('board_id') for item in snapshots]
    if any(not isinstance(identity, str) for identity in ids) or len(set(ids)) != len(ids):
        raise ValueError('board import contains missing or duplicate identities')
    return sorted(snapshots, key=lambda item: item['board_id'])


def refresh_knowledge(*, db_path=None, raw_db_path=None, kinds=None, boards=None, resume=None,
                      import_path=None, limit_budget=2000, provider=None, on_progress=None,
                      source=SOURCE, min_interval=1.0, workers=4, reuse_today=True, check_cancel=None):
    """Refresh or resume one frozen current catalog; return an inspectable summary."""
    if source != SOURCE and import_path is None: raise ValueError('only the identified Fuyao THS API collector is supported')
    if type(limit_budget) is not int or limit_budget < 1: raise ValueError('request budget must be a positive integer')
    if type(workers) is not int or not 1 <= workers <= 8: raise ValueError('board workers must be an integer from 1 to 8')
    if type(reuse_today) is not bool: raise ValueError('reuse_today must be a boolean')
    if isinstance(min_interval, bool) or not isinstance(min_interval, (int, float)) or not math.isfinite(min_interval) or min_interval < 0:
        raise ValueError('request interval must be a finite non-negative number')
    selected_kinds = list(kinds or ('industry', 'concept'))
    if len(set(selected_kinds)) != len(selected_kinds) or any(kind not in KINDS for kind in selected_kinds):
        raise ValueError('unsupported or repeated board kind')
    if boards is not None and (not isinstance(boards, (list, tuple)) or len(set(boards)) != len(boards)):
        raise ValueError('explicit board IDs must be unique')
    if check_cancel: check_cancel()
    started = time.monotonic()
    store = BoardKnowledgeStore(db_path)
    stock_proof = store.seed_stock_basic(_read_stock_basic(raw_db_path))
    own_provider = provider is None
    snapshots = _load_import(import_path) if import_path else None
    imports = {} if snapshots is None else {item['board_id']: item for item in snapshots}
    if snapshots is not None: source = snapshots[0]['source']
    old_run = store.get_refresh(resume) if resume else None
    if resume and old_run is None: raise ValueError('board refresh to resume was not found')
    run_id, requested, completed = None, None, set()
    if old_run:
        if old_run['source'] != source: raise ValueError('resume source differs from the frozen refresh')
        requested = old_run['requested_boards']
        if any(not isinstance(item, dict) for item in requested):
            raise ValueError('resume requires the full frozen catalog metadata')
        if old_run['board_ids'] and boards and set(boards) != set(old_run['board_ids']): raise ValueError('resume cannot change the frozen board IDs')
        if requested and kinds and set(kinds) != {item['kind'] for item in requested}: raise ValueError('resume cannot change the frozen board kinds')
        completed = set(old_run['completed_board_ids'])
        if old_run['status'] == 'running': run_id = resume
        else:
            requested = [item for item in requested if item['board_id'] not in completed]
            completed = set()
            if not requested and old_run['board_ids']:
                return {**old_run.get('summary', {}), 'status': old_run['status'], 'run_id': resume,
                    'request_count': 0, 'stock_basic': stock_proof, 'current_only': True}
            if not old_run['board_ids']: requested = None
    if requested is None and snapshots is not None:
        requested = [{key: snapshot[key] for key in ('board_id', 'source_code', 'name', 'kind', 'source', 'taxonomy')}
                     for snapshot in snapshots if snapshot['kind'] in selected_kinds]
    try:
        if snapshots is None and provider is None:
            provider = FuyaoBoardProvider(request_budget=limit_budget, min_interval=min_interval)
        if requested is None: requested = provider.catalog(selected_kinds)
    except (BoardSourceError, ValueError) as exc:
        failed_run = store.begin_refresh(source=source, requested_boards=[], observed_at=local_now().isoformat())
        store.record_source_failure(failed_run, str(exc), scope='catalog')
        result = store.finish_refresh(failed_run)
        if own_provider and provider is not None: provider.close()
        return {**result, 'error': str(exc), 'stock_basic': stock_proof,
            'request_count': getattr(provider, 'request_count', 0), 'current_only': True}
    if boards and (not old_run or not old_run['board_ids']):
        by_id = {item['board_id']: item for item in requested}
        missing = sorted(set(boards) - set(by_id))
        if missing:
            if own_provider and provider is not None: provider.close()
            raise ValueError('board IDs are absent from the verified source catalog: ' + ','.join(missing))
        requested = [by_id[identity] for identity in sorted(boards)]
    if not requested: raise ValueError('requested board catalog is empty')
    if run_id is None: run_id = store.begin_refresh(source=source, requested_boards=requested, observed_at=local_now().isoformat())
    failures, consecutive_failures = [], 0
    fresh, reused = 0, 0
    # Explicit single-session adapters and JSON imports retain serial behavior.
    effective_workers = workers if snapshots is None and getattr(provider, 'concurrent_safe', False) else 1
    def metrics():
        return {'workers': effective_workers, 'requested_workers': workers, 'min_interval': min_interval,
            'reuse_today': reuse_today, 'fresh_boards': fresh, 'reused_boards': reused,
            'elapsed_seconds': round(time.monotonic() - started, 3)}
    pending = [item for item in requested if item['board_id'] not in completed]
    def progress(state='running'):
        value = {'run_id': run_id, 'status': state, 'completed': len(completed), 'total': len(requested),
            'request_count': getattr(provider, 'request_count', 0), 'failed': len(failures), **metrics()}
        if on_progress: on_progress(value)
    collection = None
    try:
        progress()
        candidates = _same_day_candidates(store, pending) if snapshots is None and reuse_today else {}
        for board in pending:
            if check_cancel: check_cancel()
            cached = candidates.get(board['board_id'])
            if cached:
                store.reuse_current_snapshot(run_id, board['board_id'], cached['snapshot_hash'])
                completed.add(board['board_id']); reused += 1; progress()
        pending = [board for board in pending if board['board_id'] not in completed]
        if snapshots is None and effective_workers > 1:
            collection = _ordered_snapshots(pending, provider, effective_workers, check_cancel)
        else:
            def serial():
                for board in pending:
                    if check_cancel: check_cancel()
                    try:
                        value = imports.get(board['board_id']) if snapshots is not None else provider.snapshot(board)
                        yield board, value, None
                    except (BoardSourceError, ValueError) as exc: yield board, None, exc
            collection = serial()
        for board, value, error in collection:
            if check_cancel: check_cancel()
            try:
                if error: raise error
                if value is None: raise ValueError('import is missing a board from the frozen requested catalog')
                store.import_snapshot(value, run_id=run_id)
                completed.add(board['board_id']); fresh += 1; consecutive_failures = 0
            except BoardBudgetExhausted as exc:
                store.record_failure(run_id, board['board_id'], str(exc)); progress('interrupted')
                return {'run_id': run_id, 'status': 'interrupted', 'completed_boards': len(completed),
                    'requested_boards': len(requested), 'completed_board_ids': sorted(completed),
                    'missing_board_ids': [item['board_id'] for item in pending if item['board_id'] not in completed],
                    'request_count': provider.request_count, 'error': str(exc), 'stock_basic': stock_proof,
                    'current_only': True, 'resume': run_id, **metrics()}
            except (BoardSourceError, ValueError) as exc:
                error = str(exc); failures.append({'board_id': board['board_id'], 'error': error})
                store.record_failure(run_id, board['board_id'], error); consecutive_failures += 1
                if isinstance(exc, BoardAccessError) or consecutive_failures >= 3:
                    store.record_source_failure(run_id, error, scope='source'); progress('stopped'); break
            progress()
        result = store.finish_refresh(run_id); progress(result['status'])
        return {**result, 'request_count': getattr(provider, 'request_count', 0), 'stock_basic': stock_proof,
            'current_only': True, 'historical_membership_verified': False,
            'resumed_from': resume, 'catalog_provenance': getattr(provider, 'catalog_provenance', {}), **metrics()}
    except BaseException:
        if pending:
            current = locals().get('board', pending[0])
            store.record_failure(run_id, current['board_id'], 'explicit interruption; frozen successes remain resumable')
        raise
    finally:
        if collection is not None: collection.close()
        if own_provider and provider is not None: provider.close()


def build_parser():
    parser = argparse.ArgumentParser(description='同花顺指数口径的当前板块知识库更新（扶摇API来源）')
    parser.add_argument('--db', type=Path, help='独立basic_kb.db；不写行情表')
    parser.add_argument('--raw-db', type=Path, help='只读股票基本信息原始数据库')
    parser.add_argument('--board', nargs='+', help='完整板块ID，如THS_I_881155.TI')
    parser.add_argument('--kinds', nargs='+', choices=tuple(KINDS), default=None)
    parser.add_argument('--import', dest='import_path', type=Path, help='导入具有来源与完整证明的JSON快照')
    parser.add_argument('--resume', help='恢复未完成刷新；完成的manifest保持不可变')
    parser.add_argument('--limit-budget', type=int, default=2000, help='本次最多HTTP请求次数；默认2000')
    parser.add_argument('--min-interval', type=float, default=1.0, help='所有线程共享的请求起始最小间隔秒数；默认1.0，限流后可调大')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4, help='成员采集线程数1..8；默认4，线程共享总限流及请求预算')
    parser.add_argument('--no-reuse-today', dest='reuse_today', action='store_false', help='强制重新采集；默认复用身份与目录证明一致的今日已核验成员')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = refresh_knowledge(db_path=args.db, raw_db_path=args.raw_db, boards=args.board,
        kinds=args.kinds, import_path=args.import_path, resume=args.resume, limit_budget=args.limit_budget,
        min_interval=args.min_interval, workers=args.workers, reuse_today=args.reuse_today,
        on_progress=lambda value: print('BOARD_KNOWLEDGE_PROGRESS ' + json.dumps(value, ensure_ascii=False), flush=True))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'succeeded' else 2


if __name__ == '__main__': raise SystemExit(main())
