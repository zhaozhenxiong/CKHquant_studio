"""Owned, resumable daily derived-data sync; this module never fits a model."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time

import psutil

from my_strategy.core.paths import ARTIFACT_ROOT, PROJECT_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now


class SyncCancelled(RuntimeError): pass


def _cancel(check_cancel):
    try:
        requested = check_cancel and check_cancel()
    except BaseException as exc:
        if 'cancel' in type(exc).__name__.lower(): raise SyncCancelled(str(exc)) from exc
        raise
    if requested: raise SyncCancelled('daily feature sync cancelled')


def _calendar(identity):
    from my_strategy.services.czsc_trend_support import _calendar as load
    return load(identity)


def _store(path):
    from my_strategy.storage.board_fluent import BoardFluentStore
    return BoardFluentStore(path)


def _build_fluent(**kwargs):
    from my_strategy.services.board_fluent import build_board_fluent
    return build_board_fluent(**kwargs)


def _refresh_kb(**kwargs):
    from my_strategy.scripts.refresh_board_knowledge import refresh_knowledge
    return refresh_knowledge(**kwargs)


def _validate_cached_native(generation, *, raw_db_path, source_root, artifacts_root, progress, check_cancel):
    from my_strategy.services.board_native_summary import validate_native_reuse
    if not generation.get('native_identity'): raise ValueError('completed generation native identity is missing')
    result = validate_native_reuse(generation['native_identity'], raw_db_path=raw_db_path,
        source_root=source_root, artifacts_root=artifacts_root,
        check_cancel=lambda: _cancel(check_cancel),
        progress=(lambda **value: progress({'event': 'feature_sync', **value})) if progress else None)
    if not result['valid']: raise ValueError('completed native source identity unavailable: ' + str(result['reason']))


def _catalog(path):
    from my_strategy.storage.board_knowledge import BoardKnowledgeStore
    store = BoardKnowledgeStore(path, readonly=True)
    rows = store.catalog(current=True)
    state = store.status()
    latest = state.get('latest_run') or {}
    today = str(local_now().date())
    # A small successful retry manifest is sufficient only when the complete
    # original catalog is covered by verified snapshots observed today.
    expected = set()
    for row in rows:
        proof = row.get('provenance', {}).get('catalog_provenance') or {}
        response = proof.get('source_response') or {}
        prefix = 'THS_I_' if row['kind'] == 'industry' else 'THS_C_'
        expected.update(prefix + item['thscode'] for item in response.get('data', {}).get('item', []))
    current = bool(rows) and latest.get('status') == 'succeeded' and bool(expected)
    current &= expected == {row['board_id'] for row in rows}
    current &= all(_shanghai_day(row['observed_at']) == today for row in rows)
    return dict(current=bool(current), catalog_hash=stable_hash(sorted(
        (row['board_id'], row['snapshot_hash']) for row in rows)), board_count=len(rows),
        latest_run_id=latest.get('run_id'), latest_status=latest.get('status'))


def _shanghai_day(value):
    from my_strategy.storage.board_knowledge import _instant
    from zoneinfo import ZoneInfo
    return _instant(value).astimezone(ZoneInfo('Asia/Shanghai')).date().isoformat()


def qualified_raw_identity(raw_db_path=None, *, progress=None, check_cancel=None):
    """Stream real qualified content, including older revisions, in one RO view."""
    from my_strategy.adapters.czsc_adapter import raw_path
    from my_strategy.services.czsc_trend_support import VERIFIED_SOURCES
    path = raw_path(raw_db_path)
    if not path.is_file(): raise FileNotFoundError('daily sync raw database is missing')
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30) as con:
        con.execute('PRAGMA query_only=ON'); con.execute('BEGIN')
        columns = {row[1] for row in con.execute('PRAGMA table_info(stock_daily_normalized)')}
        fields = ['stock', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'source', 'has_trade_price']
        if not set(fields) <= columns: raise ValueError('qualified raw price contract is missing')
        if 'available_at' in columns: fields.append('available_at')
        clause = 'has_trade_price=1 AND source IN (' + ','.join('?' for _ in VERIFIED_SOURCES) + ')'
        sources = sorted(VERIFIED_SOURCES)
        day = con.execute('SELECT MAX(date) FROM stock_daily_normalized WHERE ' + clause, sources).fetchone()[0]
        if not day: raise ValueError('no actual qualified local price date')
        cursor = con.execute('SELECT ' + ','.join(fields) + ' FROM stock_daily_normalized WHERE ' +
            clause + ' AND date<=? ORDER BY stock,date', [*sources, day])
        digest = hashlib.sha256(); count = 0; last_progress = 0.0
        digest.update(json.dumps(fields, separators=(',', ':')).encode())
        while batch := cursor.fetchmany(4096):
            _cancel(check_cancel)
            for row in batch:
                digest.update(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())
                digest.update(b'\n')
            count += len(batch)
            now = time.monotonic()
            if progress and now - last_progress >= 1:
                progress({'stage': 'raw_identity', 'current': count, 'total': None}); last_progress = now
        if progress: progress({'stage': 'raw_identity', 'current': count, 'total': count})
    return dict(price_date=day, qualified_rows=count, raw_content_sha256=digest.hexdigest(), fields=fields)


def _code_identity(config, source_root):
    from my_strategy.services.board_native_summary import native_generation_contract
    paths = [Path(__file__), PROJECT_ROOT / 'my_strategy/services/board_features.py',
        PROJECT_ROOT / 'my_strategy/services/board_fluent.py', PROJECT_ROOT / 'my_strategy/storage/board_fluent.py',
        PROJECT_ROOT / 'my_strategy/services/board_native_summary.py',
        PROJECT_ROOT / 'my_strategy/storage/board_knowledge.py',
        PROJECT_ROOT / 'my_strategy/data_manager/board_provider.py',
        PROJECT_ROOT / 'my_strategy/scripts/refresh_board_knowledge.py',
        PROJECT_ROOT / 'my_strategy/scripts/refresh_native_cache.py']
    hashes = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()}
    actual = native_generation_contract(source_root=source_root, config=config)
    return dict(code_hashes=hashes, native_input_contract=_native_contract_body(actual))


def _native_contract_body(actual):
    contract = {key: actual[key] for key in ('code_hashes', 'engine_contract', 'native_schema_hash',
        'native_config_hash', 'feature_schema_hashes', 'native_version')}
    contract.update(feature_start=actual['config']['feature_start'], liquidity_universe=actual['config']['liquidity_universe'])
    return contract


def _published_native_source(store, *, identity, raw_db_path, source_root, artifacts_root, progress, check_cancel):
    """Reuse a verified same-date producer; membership changes do not replay bars."""
    try:
        candidate = store.current_generation('full')
        if not candidate or not candidate.get('full_ready') or not candidate.get('full_published'): return None
        native = candidate['native_identity']; contract = native['generation_contract']
        day = identity['raw']['price_date']
        if (native['price_date'] != day or native.get('mode') != 'current_reference' or
                native.get('training_eligible') is not False or
                _native_contract_body(contract) != identity['generating']['native_input_contract']): return None
        raw_id, native_id = native['raw_source_run_id'], native['source_native_run_id']
        if any(not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', value)
                for value in (raw_id, native_id)): return None
        report_path = Path(artifacts_root) / 'runs' / raw_id / 'reports/raw_source.json'
        content = report_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != native['raw_report_sha256']: return None
        report = json.loads(content)
        if report.get('data_end') != day or report.get('calendar_hash') != identity['calendar_hash']: return None
        store.verify_generation(candidate['generation_id'], level='full')
        _validate_cached_native(candidate, raw_db_path=raw_db_path, source_root=source_root,
            artifacts_root=artifacts_root, progress=progress, check_cancel=check_cancel)
        return dict(status='ready', source_native_run_id=native_id, raw_source_run_id=raw_id,
            training_started=False, reused_published_source=True, source_generation_id=candidate['generation_id'])
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error):
        return None


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _stop_tree(process):
    """Terminate the owned child tree, escalate and wait; never touch another job."""
    try: parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        process.wait(); return
    descendants = parent.children(recursive=True)
    targets = [*reversed(descendants), parent]
    if os.name != 'nt':
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    for child in targets:
        try: child.terminate()
        except psutil.NoSuchProcess: pass
    _, alive = psutil.wait_procs(targets, timeout=5)
    running = []
    for child in alive:
        try:
            if child.status() != psutil.STATUS_ZOMBIE: running.append(child)
        except psutil.NoSuchProcess: pass
    alive = running
    for child in alive:
        try: child.kill()
        except psutil.NoSuchProcess: pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    process.wait(timeout=5)
    if alive: raise RuntimeError('owned native child tree did not exit after cancellation')


def _run_native(*, run_dir, price_date, raw_db_path, native_cache_db_path, calendar_run_id,
                cpu_workers, stock_batch_size, source_root, artifacts_root, config,
                checkpoint, progress, check_cancel):
    args = [sys.executable, '-u', '-m', 'my_strategy.scripts.refresh_native_cache', '--end', price_date,
        '--raw-db', str(raw_db_path), '--cpu-workers', str(cpu_workers), '--stock-batch-size', str(stock_batch_size),
        '--calendar-run-id', calendar_run_id, '--artifacts-root', str(artifacts_root), '--sync-id', run_dir.name]
    if native_cache_db_path: args += ['--cache-db', str(native_cache_db_path)]
    if source_root: args += ['--source-root', str(source_root)]
    if config:
        path = run_dir / 'config/native_input.json'; _save(path, config)
        args += ['--config', str(path)]
    if checkpoint.get('source_native_run_id'):
        args += ['--source-native-run-id', checkpoint['source_native_run_id']]
    for key in ('raw_source_run_id', 'resume_native_run_id'):
        if key == 'resume_native_run_id' and checkpoint.get('source_native_run_id'): continue
        if checkpoint.get(key): args += ['--' + key.replace('_', '-'), checkpoint[key]]
    log = run_dir / 'logs/native_prepare.log'; log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
        KHQUANT_RAW_DB=str(raw_db_path), KHQUANT_ARTIFACT_ROOT=str(artifacts_root))
    result = None; fragments = ''
    def parse(line):
        nonlocal result
        for prefix, key in (('RAW_SOURCE_RUN_ID ', 'raw_source_run_id'), ('NATIVE_SOURCE_RUN_ID ', 'resume_native_run_id')):
            if line.startswith(prefix): progress({'stage': 'native_prepare', key: line[len(prefix):].strip()})
        for prefix in ('SOURCE_BATCH ', 'PROGRESS ', 'NATIVE_CACHE_RESULT '):
            if line.startswith(prefix): line = line[len(prefix):]; break
        if line.startswith('{'):
            try: value, _ = json.JSONDecoder().raw_decode(line)
            except ValueError: return
            if not isinstance(value, dict): return
            if value.get('source_native_run_id') and value.get('training_started') is False:
                result = value
            else:
                progress({**value, 'stage': 'native_prepare'})
    with log.open('a', encoding='utf-8') as output, log.open('r', encoding='utf-8') as reader:
        reader.seek(0, os.SEEK_END)
        options = {'start_new_session': True} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
        process = subprocess.Popen(args, cwd=PROJECT_ROOT, stdout=output, stderr=subprocess.STDOUT, env=environment, **options)
        try:
            while True:
                _cancel(check_cancel)
                fragments += reader.read()
                lines = fragments.split('\n'); fragments = lines.pop()
                for line in lines: parse(line.strip())
                progress({'stage': 'native_prepare', 'heartbeat': True})
                if process.poll() is not None:
                    fragments += reader.read()
                    for line in fragments.splitlines(): parse(line.strip())
                    break
                time.sleep(.2)
            if process.returncode != 0: raise RuntimeError('native prepare failed; see ' + str(log))
            if not result or result.get('status') not in {'ready', 'completed', 'succeeded'}:
                raise ValueError('native prepare did not produce a verified ready source')
            return result
        finally:
            if process.poll() is None: _stop_tree(process)


def sync_daily_features(*, raw_db_path=None, kb_db_path=None, fluent_db_path=None,
                       calendar_run_id=None, sync_id=None, parent_update_id=None,
                       native_cache_db_path=None, cpu_workers=6, stock_batch_size=360,
                       kb_request_budget=2000, kb_min_interval=1.3, progress=None,
                       check_cancel=None, source_root=None, artifacts_root=None, config=None,
                       source_native_run_id=None, raw_source_run_id=None):
    """Publish market features first, then prepare-only native and a full generation."""
    from my_strategy.adapters.czsc_adapter import raw_path
    if type(cpu_workers) is not int or not 1 <= cpu_workers <= 16: raise ValueError('invalid native worker count')
    if type(stock_batch_size) is not int or stock_batch_size < 1: raise ValueError('invalid preparation batch size')
    if type(kb_request_budget) is not int or kb_request_budget < 1: raise ValueError('invalid board request budget')
    if isinstance(kb_min_interval, bool) or not isinstance(kb_min_interval, (int, float)) or not math.isfinite(kb_min_interval) or kb_min_interval < 0:
        raise ValueError('invalid board request interval')
    path = raw_path(raw_db_path)
    output_root = Path(artifacts_root or ARTIFACT_ROOT).resolve()
    raw = qualified_raw_identity(path, progress=progress, check_cancel=check_cancel)
    dates, calendar_id = _calendar(calendar_run_id)
    if raw['price_date'] not in dates: raise ValueError('actual qualified price date is absent from verified calendar')
    dates = [day for day in dates if day <= raw['price_date']]
    identity = dict(format='daily_feature_sync_v1', raw=raw, calendar_hash=stable_hash(dates),
        kb_refresh_day=str(local_now().date()), generating=_code_identity(config, source_root))
    if source_native_run_id or raw_source_run_id:
        for value in (source_native_run_id, raw_source_run_id):
            if value is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', value):
                raise ValueError('invalid prepared source identifier')
        identity['prepared_source'] = dict(source_native_run_id=source_native_run_id, raw_source_run_id=raw_source_run_id)
    store = _store(fluent_db_path)
    kb = _catalog(kb_db_path)
    if sync_id:
        previous = store.get_sync(sync_id)
        if previous is None: raise ValueError('daily sync checkpoint was not found')
        frozen = dict(previous['input_identity'])
        for key in ('kb_snapshot_hash', 'republish_generation'): frozen.pop(key, None)
        if not (source_native_run_id or raw_source_run_id) and frozen.get('prepared_source'):
            identity['prepared_source'] = frozen['prepared_source']
        if frozen != identity: raise ValueError('daily sync input changed; start a new sync instead of resuming')
        identity = previous['input_identity']
    lease = store.acquire_sync(stable_hash(identity), identity, sync_id=sync_id, parent_update_id=parent_update_id)
    if lease.get('reused') and lease.get('status') == 'completed':
        saved = lease.get('result') or (store.get_sync(lease['sync_id']) or {}).get('result') or {}
        if kb['current'] and saved.get('kb_catalog_hash') == kb['catalog_hash']:
            generation = store.verify_generation(saved['generation_id'], level='full')
            if generation['market_published'] and generation['full_published']:
                _validate_cached_native(generation, raw_db_path=path, source_root=source_root,
                    artifacts_root=output_root, progress=progress, check_cancel=check_cancel)
                return {**saved, 'sync_id': lease['sync_id'], 'reused': True, 'training_started': False}
        if sync_id: raise ValueError('completed sync membership changed; start a new sync')
        identity = {**identity, 'kb_snapshot_hash': kb['catalog_hash'], 'republish_generation': saved.get('generation_id')}
        lease = store.acquire_sync(stable_hash(identity), identity, parent_update_id=parent_update_id)
        if lease.get('reused') and lease.get('status') == 'completed':
            saved = lease.get('result') or {}
            generation = store.verify_generation(saved['generation_id'], level='full')
            if kb['current'] and saved.get('kb_catalog_hash') == kb['catalog_hash'] and generation['market_published'] and generation['full_published']:
                _validate_cached_native(generation, raw_db_path=path, source_root=source_root,
                    artifacts_root=output_root, progress=progress, check_cancel=check_cancel)
                return {**saved, 'sync_id': lease['sync_id'], 'reused': True, 'training_started': False}
            raise ValueError('completed derived sync publication changed; start a new sync')
    if not lease.get('owned'):
        return dict(sync_id=lease['sync_id'], status='busy', stage=lease.get('stage'),
            market_published=False, full_published=False, training_started=False)
    sync_id = lease['sync_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', sync_id): raise ValueError('invalid sync identifier')
    run_dir = output_root / 'runs' / sync_id
    record = store.get_sync(sync_id) or lease
    checkpoint = dict(record.get('checkpoint') or {})
    prepared = identity.get('prepared_source') or {}
    for key, value in prepared.items():
        if value and key not in checkpoint: checkpoint[key] = value
    if prepared: store.heartbeat_sync(sync_id, checkpoint=prepared)
    summary = dict(sync_id=sync_id, status='running', price_date=raw['price_date'],
        market_published=False, full_published=False, training_started=False, raw_content_sha256=raw['raw_content_sha256'])
    last_heartbeat = [0.0]
    def emit(value):
        saved = {key: value[key] for key in ('raw_source_run_id', 'resume_native_run_id') if value.get(key)}
        if value.get('stage') == 'board_knowledge' and value.get('run_id'): saved['kb_resume_id'] = value['run_id']
        if saved: checkpoint.update(saved)
        now = time.monotonic()
        if not value.get('heartbeat') or saved or now - last_heartbeat[0] >= 5:
            store.heartbeat_sync(sync_id, stage=value.get('stage'), checkpoint=saved or None)
            last_heartbeat[0] = now
        _cancel(check_cancel)
        if progress and not value.get('heartbeat'):
            progress({'event': 'feature_sync', 'sync_id': sync_id, **value})
    def stage(name): emit({'stage': name, 'status': 'running', 'current': 0, 'total': 1})
    def save_checkpoint(values):
        checkpoint.update(values); store.heartbeat_sync(sync_id, checkpoint=values)
    try:
        if not (run_dir / 'config/sync_input.json').exists(): _save(run_dir / 'config/sync_input.json', identity)
        stage('board_knowledge')
        if not kb['current']:
            refreshed = _refresh_kb(db_path=kb_db_path, raw_db_path=path,
                resume=checkpoint.get('kb_resume_id'), kinds=None if checkpoint.get('kb_resume_id') else ['industry', 'concept'],
                limit_budget=kb_request_budget, min_interval=kb_min_interval,
                on_progress=lambda value: emit({**value, 'stage': 'board_knowledge'}))
            save_checkpoint({'kb_result': {key: refreshed.get(key) for key in ('run_id', 'status', 'error')},
                'kb_resume_id': refreshed.get('run_id') if refreshed.get('status') != 'succeeded' else None})
            kb = _catalog(kb_db_path)
            summary['kb_status'] = refreshed.get('status')
            if refreshed.get('error'): summary['kb_error'] = refreshed['error']
        else: summary['kb_status'] = 'reused'
        summary['kb_catalog_hash'] = kb['catalog_hash']
        native = checkpoint.get('native_result')
        if kb['current'] and not native and not any(checkpoint.get(key) for key in
                ('source_native_run_id', 'raw_source_run_id', 'resume_native_run_id')):
            stage('native_source_reuse')
            native = _published_native_source(store, identity=identity, raw_db_path=path,
                source_root=source_root, artifacts_root=output_root,
                progress=lambda value: emit({**value, 'stage': 'native_source_reuse'}), check_cancel=check_cancel)
            if native: save_checkpoint({'native_result': native})
        stage('market_features')
        common = dict(kb_db_path=kb_db_path, raw_db_path=path, fluent_db_path=fluent_db_path,
            calendar_dates=dates, calendar_run_id=calendar_id, price_date=raw['price_date'],
            sync_id=sync_id, progress=emit)
        market = _build_fluent(**common, with_native=False)
        summary.update(market_published=market.get('market_published', False), market_generation_id=market.get('generation_id'))
        save_checkpoint({'market_result': market})
        if not summary['market_published']: raise ValueError('market generation was not published')
        store.verify_generation(market['generation_id'], level='market')
        if not kb['current']:
            summary.update(status='partial', error=summary.get('kb_error') or 'current board catalog refresh incomplete')
        else:
            native = checkpoint.get('native_result')
            stage('native_prepare')
            if not native:
                native = _run_native(run_dir=run_dir, price_date=raw['price_date'], raw_db_path=path,
                    native_cache_db_path=native_cache_db_path, calendar_run_id=calendar_id,
                    cpu_workers=cpu_workers, stock_batch_size=stock_batch_size,
                    source_root=source_root, artifacts_root=output_root, config=config,
                    checkpoint=checkpoint, progress=emit, check_cancel=check_cancel)
                save_checkpoint({'native_result': native})
            if native.get('training_started') is not False or not native.get('source_native_run_id'):
                raise ValueError('native preparation source is not ready')
            summary['source_native_run_id'] = native['source_native_run_id']
            summary['raw_source_run_id'] = native.get('raw_source_run_id')
            stage('native_summaries')
            latest = qualified_raw_identity(path, check_cancel=check_cancel,
                progress=lambda value: emit({**value, 'stage': 'source_recheck'}))
            if latest != raw or _code_identity(config, source_root) != identity['generating']:
                raise ValueError('daily sync raw content or generating identity changed during preparation')
            full = _build_fluent(**common, with_native=True, source_native_run_id=native['source_native_run_id'],
                raw_source_run_id=native.get('raw_source_run_id'), native_cache_db_path=native_cache_db_path,
                source_root=source_root, artifacts_root=output_root, native_config=config)
            summary.update(full_published=full.get('full_published', False), generation_id=full.get('generation_id'))
            if not summary['full_published']: raise ValueError('full native generation was not published')
            store.verify_generation(full['generation_id'], level='full')
            summary['status'] = 'completed'; save_checkpoint({'full_result': full})
        store.finish_sync(sync_id, status=summary['status'], result=summary, error=summary.get('error'))
    except (SyncCancelled, KeyboardInterrupt) as exc:
        summary.update(status='cancelled', error=str(exc) or 'daily sync interrupted')
        store.finish_sync(sync_id, status='cancelled', result=summary, error=summary['error'])
    except Exception as exc:
        summary.update(status='partial' if summary['market_published'] else 'failed', error=str(exc))
        store.finish_sync(sync_id, status=summary['status'], result=summary, error=str(exc))
    _save(run_dir / 'reports/daily_feature_sync.json', summary)
    if progress: progress({'event': 'feature_sync', 'sync_id': sync_id, 'stage': 'finished', **summary})
    return summary
