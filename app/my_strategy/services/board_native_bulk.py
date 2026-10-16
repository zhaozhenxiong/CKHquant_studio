"""Explicit full-board native preparation with bounded workers and one publication.

Every numerical result comes from IncrementalSource.summary. This coordinator
owns only a new plan, independently hashed checkpoints and final aggregation.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import uuid

from filelock import FileLock

from my_strategy.core.run_context import stable_hash
from my_strategy.services.incremental_source import IncrementalSource, code_identity
from my_strategy.services.incremental_sync import _catalog, _contract, _unavailable, create_plan
from my_strategy.services.board_fluent import generation_contract, source_witness
from my_strategy.storage.research_store import ResearchStore, canonical

FORMAT = 'board_native_bulk_v1'
MAX_CACHE_QUOTA_BYTES = 64 * 1024**3
_worker_source = None
_worker_stop = None


def _executor_identity():
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), root / 'scripts/fill_board_native.py')}


def _atomic_json(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as output:
            output.write(canonical(value));output.write('\n');output.flush();os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save(path, state):
    body = {key: value for key, value in state.items() if key != 'checkpoint_sha256'}
    state['checkpoint_sha256'] = stable_hash(body)
    _atomic_json(path, state)


def _load(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    body = {key: item for key, item in value.items() if key != 'checkpoint_sha256'}
    if value.get('format') != FORMAT or stable_hash(body) != value.get('checkpoint_sha256'):
        raise ValueError('bulk checkpoint identity or SHA mismatch')
    if (stable_hash(value['contract']) != value.get('contract_sha256') or
            [row['symbol'] for row in value['rows']] != value['contract']['symbols']):
        raise ValueError('bulk checkpoint frozen population or contract mismatch')
    return value


def _paths(raw_db_path, kb_db_path, fluent_db_path):
    from my_strategy.adapters.czsc_adapter import raw_path
    from my_strategy.core.paths import BASIC_KB_DB_PATH, BASIC_FLUENT_DB_PATH
    return {'raw_db_path': str(raw_path(raw_db_path)),
        'kb_db_path': str(Path(kb_db_path or BASIC_KB_DB_PATH).resolve()),
        'fluent_db_path': str(Path(fluent_db_path or BASIC_FLUENT_DB_PATH).resolve())}


def _guard(state):
    paths = state['paths']
    witness = source_witness(raw_db_path=paths['raw_db_path'], kb_db_path=paths['kb_db_path'])
    if witness['raw'] != state['witness']['raw']:
        raise ValueError('bulk input database changed; preview a new plan')
    if witness['knowledge'] != state['witness']['knowledge']:
        # A committed KB WAL may checkpoint when its last old reader closes.
        # Only an exact, newly verified source/projection population permits
        # rebinding that physical hint; membership changes still fail closed.
        catalog = _catalog(paths['kb_db_path'])
        if ({row['board_id']: row['snapshot_hash'] for row in catalog} != state['contract']['catalog'] or
                {row['board_id']: row['calculation_membership']['projection_hash'] for row in catalog} != state['contract']['projection_hashes']):
            raise ValueError('bulk knowledge membership changed; preview a new plan')
        witness = source_witness(raw_db_path=paths['raw_db_path'], kb_db_path=paths['kb_db_path'])
        if witness['raw'] != state['witness']['raw']:
            raise ValueError('bulk input database changed; preview a new plan')
        state['witness']['knowledge'] = witness['knowledge']
        state['knowledge_witness_reverified'] = state.get('knowledge_witness_reverified', 0) + 1
    if code_identity() != state['contract']['producer'] or generation_contract() != state['contract']['fluent']:
        raise ValueError('bulk producer or Fluent code changed; preview a new plan')
    if _executor_identity() != state['executor']:
        raise ValueError('bulk executor changed; preview a new plan')
    from my_strategy.services.czsc_trend_support import _calendar
    dates, _ = _calendar(None)
    if [day for day in dates if day <= state['price_date']] != state['dates']:
        raise ValueError('bulk verified calendar changed; preview a new plan')


def preview_bulk(*, store, checkpoint, raw_db_path=None, kb_db_path=None, fluent_db_path=None,
                 price_date=None, batch_size=100, cpu_workers=4, stop=None):
    """Create one complete gap plan, without preparing native observations."""
    if type(batch_size) is not int or not 1 <= batch_size <= 100:
        raise ValueError('batch_size must be 1..100')
    if type(cpu_workers) is not int or not 1 <= cpu_workers <= 12:
        raise ValueError('cpu_workers must be 1..12')
    if not 1 <= store.quota_bytes <= MAX_CACHE_QUOTA_BYTES:
        raise ValueError('bulk cache capacity must be positive and at most 64 GiB')
    path = Path(checkpoint).resolve()
    with FileLock(str(path) + '.lock').acquire(timeout=0):
        if _stopped(stop):raise InterruptedError('bulk preview cancelled before plan creation')
        if path.exists():
            raise ValueError('checkpoint exists; use explicit execute or a new checkpoint path')
        paths = _paths(raw_db_path, kb_db_path, fluent_db_path)
        plan = create_plan(store=store, **paths, scope='full', price_date=price_date,
            max_symbols=batch_size, budget_seconds=1800)
        contract, _, dates = _contract(store, paths['raw_db_path'], paths['kb_db_path'],
            'full', None, None, plan['price_date'])
        if contract != plan['contract']:
            raise ValueError('bulk inputs changed during preview; create a new plan')
        # Store only source identity digests per stock. Their repeated producer
        # contract remains once at the top of this small checkpoint index.
        rows = []
        for row in plan['rows']:
            compact = {key: value for key, value in row.items() if key != 'identity'}
            if row.get('identity'):
                compact['source_identity_sha256'] = stable_hash(row['identity'])
            rows.append(compact)
        state = {'format': FORMAT, 'bulk_id': uuid.uuid4().hex, 'plan_id': plan['plan_id'],
            'status': 'planned', 'price_date': plan['price_date'], 'contract': contract,
            'contract_sha256': stable_hash(contract), 'executor': _executor_identity(),
            'witness': source_witness(raw_db_path=paths['raw_db_path'], kb_db_path=paths['kb_db_path']),
            'paths': paths, 'cache_root': str(store.root), 'dates': dates, 'rows': rows,
            'counts': plan['counts'], 'batch_size': batch_size, 'cpu_workers': cpu_workers,
            'cache_quota_bytes': store.quota_bytes,
            'records': {}, 'attempts': [], 'created_at': time.time(), 'training_started': False,
            'legacy_tasks_resumed': False, 'full_published': False}
        _guard(state)
        if _stopped(stop):
            state.update(status='cancelled', error='bulk preview cancelled; create a new task')
            store.set_setting('incremental_plan:' + state['plan_id'], {'status': 'cancelled', 'bulk_id': state['bulk_id']})
        _save(path, state)
        return state


def _verify_summary(summary, frozen, day):
    if (summary.get('symbol') != frozen['symbol'] or summary.get('price_date') != day or
            summary.get('mode') != 'current_reference' or summary.get('training_eligible') is not False or
            summary.get('row_hash') != stable_hash({key: value for key, value in summary.items() if key != 'row_hash'})):
        raise ValueError('bulk per-stock summary identity or SHA mismatch')
    if frozen['action'] == 'missing_price':
        if (summary.get('status') != 'unavailable' or summary.get('structure') is not None or summary.get('wyckoff') is not None or
                summary.get('reason_codes') != [frozen.get('reason') or 'target_day_price_missing']):
            raise ValueError('bulk missing target price cannot become an available summary')
    else:
        identity = dict(summary.get('identity') or {});source_key = identity.pop('source_cache_key', None)
        if (summary.get('status') != 'available' or
                stable_hash(identity) != frozen.get('source_identity_sha256') or
                (frozen.get('cache_key') and source_key != frozen['cache_key'])):
            raise ValueError('bulk numerical summary differs from frozen source identity')


def _stock_path(checkpoint, symbol):
    return Path(str(checkpoint) + '.stocks') / (symbol.replace('.', '_') + '.json')


def _commit_stock(checkpoint, state, frozen, summary, seconds=0.):
    _verify_summary(summary, frozen, state['price_date'])
    payload = {'format': 'board_native_bulk_stock_v1', 'bulk_id': state['bulk_id'],
        'contract_sha256': state['contract_sha256'], 'symbol': frozen['symbol'], 'summary': summary}
    path = _stock_path(checkpoint, frozen['symbol']);_atomic_json(path, payload)
    return {'status': 'completed', 'summary_file': str(path),
        'summary_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'requested_action': frozen['action'], 'elapsed_seconds': seconds}


def _read_stock(checkpoint, state, frozen, record):
    path = _stock_path(checkpoint, frozen['symbol'])
    if record.get('summary_file') != str(path):
        raise ValueError('bulk summary checkpoint path mismatch')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record['summary_sha256']:
        raise ValueError('bulk stock checkpoint physical SHA mismatch')
    payload = json.loads(data)
    if (payload.get('format') != 'board_native_bulk_stock_v1' or payload.get('bulk_id') != state['bulk_id'] or
            payload.get('symbol') != frozen['symbol'] or payload.get('contract_sha256') != state['contract_sha256']):
        raise ValueError('bulk stock checkpoint frozen contract mismatch')
    _verify_summary(payload['summary'], frozen, state['price_date'])
    return payload['summary']


def _worker_initialize(root, raw_db, stop, quota_bytes):
    global _worker_source, _worker_stop
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    # Each spawned process owns its connection; the shared CAS and leases use
    # ResearchStore's existing transactional writer protocol.
    _worker_source = IncrementalSource(store=ResearchStore(root, quota_bytes=quota_bytes), raw_db=raw_db)
    _worker_stop = stop


def _worker_summary(symbol, day, dates):
    started = time.monotonic()
    try:
        return {'symbol': symbol, 'summary': _worker_source.summary(symbol, day,
            calendar_dates=dates, stop=_worker_stop), 'elapsed_seconds': time.monotonic() - started,
            'worker_pid': os.getpid(), 'cache_quota_bytes': _worker_source.store.quota_bytes}
    except InterruptedError:
        return {'symbol': symbol, 'cancelled': True}
    except Exception as exc:
        return {'symbol': symbol, 'error': type(exc).__name__ + ': ' + str(exc)}


def _source_verification(source, symbol, day, dates, identity_sha256, stop):
    observed = source.inspect(symbol, day, calendar_dates=dates, stop=stop)
    if observed['action'] != 'reuse' or stable_hash(observed['identity']) != identity_sha256:
        raise ValueError('bulk final raw or immutable source proof changed: ' + symbol)
    actual = source.summary(symbol, day, calendar_dates=dates, stop=stop)
    return {'symbol': symbol, 'observed': observed, 'summary': actual,
        'worker_pid': os.getpid(), 'cache_quota_bytes': source.store.quota_bytes}


def _worker_verify(symbol, day, dates, identity_sha256):
    try:
        return _source_verification(_worker_source, symbol, day, dates, identity_sha256, _worker_stop)
    except InterruptedError:
        return {'symbol': symbol, 'cancelled': True}
    except Exception as exc:
        return {'symbol': symbol, 'error': type(exc).__name__ + ': ' + str(exc)}


def _verify_missing_price(row, day, raw_db):
    from my_strategy.adapters.czsc_adapter import MissingHistoricalBars, load_bars
    reason = row.get('reason')
    try:raw = load_bars(row['symbol'], end=day, db_path=raw_db)
    except MissingHistoricalBars:
        if reason == 'verified_history_unavailable' and 'observed_end' in row and row['observed_end'] is None:return
    else:
        observed = str(raw.date.iloc[-1].date())
        if observed < day and observed == row.get('observed_end') and reason == 'target_day_price_missing':return
    raise ValueError('bulk missing price checkpoint differs from actual raw history: ' + row['symbol'])


def _pool(workers, root, raw_db, stop, quota_bytes):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
        initializer=_worker_initialize, initargs=(root, raw_db, stop, quota_bytes))


def _stopped(stop):
    return bool(stop and (stop.is_set() if hasattr(stop, 'is_set') else stop()))


def _receipt(state, checkpoint):
    records = state['records'];completed = sum(row['status'] == 'completed' for row in records.values())
    failed = sum(row['status'] == 'failed' for row in records.values())
    return {'bulk_id': state['bulk_id'], 'plan_id': state['plan_id'], 'status': state['status'],
        'price_date': state['price_date'], 'requested': len(state['rows']), 'completed': completed,
        'failed': failed, 'deferred': len(state['rows']) - completed - failed,
        'generation_id': state.get('generation_id'), 'full_published': state['full_published'],
        'checkpoint': str(checkpoint), 'error': state.get('error'), 'cpu_workers': state['cpu_workers'],
        'batch_size': state['batch_size'], 'training_started': False, 'legacy_tasks_resumed': False,
        'cache_metrics': (state['attempts'][-1].get('cache_metrics') if state['attempts'] else None),
        'final_verification': (state['attempts'][-1].get('final_verification') if state['attempts'] else None),
        'budget_policy': 'stop dispatch at the deadline; finish already running stocks; validation/publication are outside dispatch budget'}


def execute_bulk(*, store, checkpoint, budget_seconds=1800, max_stocks=None, cpu_workers=None,
                 cache_quota_bytes=None, stop=None, progress=None):
    """Run a bounded attempt; partial attempts require an explicit execute call."""
    if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float)) or not 0 < budget_seconds <= 86400:
        raise ValueError('budget_seconds must be in (0,86400]')
    if max_stocks is not None and (type(max_stocks) is not int or max_stocks < 1):
        raise ValueError('max_stocks must be a positive integer')
    if cpu_workers is not None and (type(cpu_workers) is not int or not 1 <= cpu_workers <= 12):
        raise ValueError('cpu_workers must be 1..12')
    if cache_quota_bytes is not None and (type(cache_quota_bytes) is not int or not 1 <= cache_quota_bytes <= MAX_CACHE_QUOTA_BYTES):
        raise ValueError('cache_quota_bytes must be a positive integer at most 64 GiB')
    path = Path(checkpoint).resolve()
    with FileLock(str(path) + '.lock').acquire(timeout=0), FileLock(str(store.root / 'compute.lock')).acquire(timeout=0):
        state = _load(path)
        if state['status'] == 'cancelled':
            raise ValueError('cancelled bulk task is terminal; preview a new task')
        if str(store.root) != state['cache_root']:
            raise ValueError('bulk cache root differs from the frozen plan')
        _guard(state)
        if state['status'] == 'completed':
            return _receipt(state, path)
        if cpu_workers is not None:state['cpu_workers'] = cpu_workers
        original_quota = store.quota_bytes
        state['cache_quota_bytes'] = cache_quota_bytes if cache_quota_bytes is not None else state['cache_quota_bytes']
        if not 1 <= state['cache_quota_bytes'] <= MAX_CACHE_QUOTA_BYTES:
            raise ValueError('bulk checkpoint cache capacity is outside 1..64 GiB')
        store.quota_bytes = state['cache_quota_bytes']
        state['status'] = 'running';state.pop('error', None)
        state['attempts'].append({'attempt_id': uuid.uuid4().hex, 'started_at': time.time(),
            'owner_pid': os.getpid(), 'budget_seconds': budget_seconds,
            'max_stocks': max_stocks, 'cpu_workers': state['cpu_workers'], 'cache_quota_bytes': store.quota_bytes})
        with store._lock:
            initial_objects = {row[0] for row in store.connection.execute("SELECT sha256 FROM objects WHERE state='active'")}
            initial_cache_bytes = store._cache_bytes()
        state['attempts'][-1]['cache_metrics'] = {'quota_bytes': store.quota_bytes,
            'cache_bytes_before': initial_cache_bytes,
            'measurement': 'new active CAS objects during this locked attempt; net cache change is reported separately'}
        store.set_setting('incremental_plan:' + state['plan_id'], {'status': 'running', 'bulk_id': state['bulk_id']})
        _save(path, state)
        deadline = time.monotonic() + budget_seconds
        context = multiprocessing.get_context('spawn');worker_stop = context.Event()
        pool = None;source = IncrementalSource(store=store, raw_db=state['paths']['raw_db_path'])
        frozen_by_symbol = {row['symbol']: row for row in state['rows']}
        lease_owner = 'board_native_bulk:' + state['bulk_id']
        lease_ttl = budget_seconds + 3600

        def protect_source(summary):
            key = (summary.get('identity') or {}).get('source_cache_key')
            if not key:return
            manifest = store.get_by_key(key, touch=False)
            if manifest is None:raise ValueError('bulk completed numerical source is missing from CAS')
            keys = [key, *[part['cache_key'] for part in manifest['chunks']]]
            store.acquire_leases(keys, lease_owner, ttl_seconds=lease_ttl)

        def emit(stage, **details):
            if progress:progress({'stage': stage, **_receipt(state, path), **details})

        def accept(frozen, result):
            if result.get('symbol') != frozen['symbol']:
                raise ValueError('bulk worker returned a different symbol')
            if result.get('cancelled'):
                raise InterruptedError('bulk native preparation cancelled')
            if result.get('cache_quota_bytes') is not None and result['cache_quota_bytes'] != store.quota_bytes:
                raise ValueError('bulk worker cache capacity differs from this attempt')
            if result.get('error'):
                state['records'][frozen['symbol']] = {'status': 'failed', 'error': result['error']}
            else:
                _verify_summary(result['summary'], frozen, state['price_date'])
                protect_source(result['summary'])
                state['records'][frozen['symbol']] = _commit_stock(path, state, frozen,
                    result['summary'], result.get('elapsed_seconds', 0.))
                for field in ('worker_pid','cache_quota_bytes'):
                    if field in result:state['records'][frozen['symbol']][field] = result[field]
            emit('native_stock', symbol=frozen['symbol'])

        try:
            # Revalidate committed numerical checkpoint proofs before trusting
            # their completion. The final pass independently rechecks raw/CAS.
            for symbol, record in state['records'].items():
                if symbol not in frozen_by_symbol:raise ValueError('bulk checkpoint contains an outside symbol')
                if record['status'] == 'completed':protect_source(_read_stock(path, state, frozen_by_symbol[symbol], record))
            # A completed atomic stock file may precede a crash before the next
            # batch-index commit. It is recoverable only under this exact bulk
            # and frozen source identity; partial temporary files are ignored.
            for frozen in state['rows']:
                if state['records'].get(frozen['symbol'], {}).get('status') == 'completed':continue
                stock_path = _stock_path(path, frozen['symbol'])
                if stock_path.is_file():
                    recovered = {'status': 'completed', 'summary_file': str(stock_path),
                        'summary_sha256': hashlib.sha256(stock_path.read_bytes()).hexdigest(),
                        'requested_action': frozen['action'], 'elapsed_seconds': 0., 'recovered_atomic_stock': True}
                    protect_source(_read_stock(path, state, frozen, recovered))
                    state['records'][frozen['symbol']] = recovered
            pending = [row for row in state['rows'] if state['records'].get(row['symbol'], {}).get('status') != 'completed']
            pending.sort(key=lambda row: (-row.get('raw_rows', 0), row['symbol']))
            if max_stocks is not None:pending = pending[:max_stocks]
            if state['cpu_workers'] > 1:
                pool = _pool(state['cpu_workers'], str(store.root), state['paths']['raw_db_path'], worker_stop, store.quota_bytes)
            for offset in range(0, len(pending), state['batch_size']):
                if _stopped(stop):raise InterruptedError('bulk native preparation cancelled')
                if time.monotonic() >= deadline:break
                _guard(state);batch = pending[offset:offset + state['batch_size']]
                work = []
                for frozen in batch:
                    if frozen['action'] == 'missing_price':
                        accept(frozen, {'symbol': frozen['symbol'], 'summary': _unavailable(
                            frozen['symbol'], state['price_date'], frozen.get('reason') or 'target_day_price_missing')})
                    elif frozen['action'] == 'failed':
                        accept(frozen, {'symbol': frozen['symbol'], 'error': frozen.get('reason', 'source inspection failed')})
                    else:work.append(frozen)
                if pool is None:
                    for frozen in work:
                        if _stopped(stop):raise InterruptedError('bulk native preparation cancelled')
                        if time.monotonic() >= deadline:break
                        started = time.monotonic()
                        try:
                            result = {'symbol': frozen['symbol'], 'summary': source.summary(frozen['symbol'],
                                state['price_date'], calendar_dates=state['dates'], stop=stop),
                                'elapsed_seconds': time.monotonic() - started, 'worker_pid': os.getpid(),
                                'cache_quota_bytes': store.quota_bytes}
                        except InterruptedError:raise
                        except Exception as exc:result = {'symbol': frozen['symbol'], 'error': type(exc).__name__ + ': ' + str(exc)}
                        accept(frozen, result)
                else:
                    waiting = iter(work);active = {};exhausted = False
                    while active or not exhausted:
                        if _stopped(stop):raise InterruptedError('bulk native preparation cancelled')
                        while not exhausted and len(active) < state['cpu_workers'] and time.monotonic() < deadline:
                            frozen = next(waiting, None)
                            if frozen is None:exhausted = True;break
                            future = pool.submit(_worker_summary, frozen['symbol'], state['price_date'], state['dates'])
                            active[future] = frozen
                        if time.monotonic() >= deadline:exhausted = True
                        if not active:break
                        completed, _ = wait(active, timeout=.2, return_when=FIRST_COMPLETED)
                        for future in completed:
                            frozen = active.pop(future)
                            try:result = future.result()
                            except Exception as exc:result = {'symbol': frozen['symbol'], 'error': type(exc).__name__ + ': ' + str(exc)}
                            accept(frozen, result)
                _guard(state);_save(path, state);emit('native_batch')
            if _stopped(stop):raise InterruptedError('bulk native preparation cancelled')
            if any(state['records'].get(row['symbol'], {}).get('status') != 'completed' for row in state['rows']):
                state['status'] = 'partial'
            else:
                _guard(state);emit('final_source_verification')
                contract, catalog, dates = _contract(store, state['paths']['raw_db_path'], state['paths']['kb_db_path'],
                    'full', None, None, state['price_date'])
                if contract != state['contract']:raise ValueError('bulk final frozen contract changed')
                verified = {};verification_pids = set();verification_started = time.monotonic()
                verification = state['attempts'][-1]['final_verification'] = {
                    'requested': len(state['rows']), 'verified': 0, 'cpu_workers': state['cpu_workers'], 'worker_pids': []}

                def accept_verified(row, saved, result=None):
                    if result is None:
                        _verify_missing_price(row, state['price_date'], state['paths']['raw_db_path'])
                    else:
                        if result.get('symbol') != row['symbol']:
                            raise ValueError('bulk verification worker returned a different symbol')
                        if result.get('cancelled'):raise InterruptedError('bulk native verification cancelled')
                        if result.get('error'):raise ValueError(result['error'])
                        observed, actual = result['observed'], result['summary']
                        if (observed['action'] != 'reuse' or
                                stable_hash(observed['identity']) != row['source_identity_sha256']):
                            raise ValueError('bulk final raw or immutable source proof changed: ' + row['symbol'])
                        if (actual != saved or actual.get('row_hash') != saved.get('row_hash') or
                                (observed.get('cache_key') and actual.get('identity', {}).get('source_cache_key') != observed['cache_key'])):
                            raise ValueError('bulk stock checkpoint differs from verified numerical CAS summary: ' + row['symbol'])
                        if result.get('cache_quota_bytes') is not None and result['cache_quota_bytes'] != store.quota_bytes:
                            raise ValueError('bulk verification worker cache capacity differs from this attempt')
                        if result.get('worker_pid') is not None:verification_pids.add(result['worker_pid'])
                    verified[row['symbol']] = saved
                    verification.update(verified=len(verified), worker_pids=sorted(verification_pids),
                        elapsed_seconds=time.monotonic()-verification_started)
                    if len(verified)%50 == 0 or len(verified) == len(state['rows']):emit('final_source_verified')

                if pool is None:
                    for row in state['rows']:
                        if _stopped(stop):raise InterruptedError('bulk native preparation cancelled')
                        saved = _read_stock(path, state, row, state['records'][row['symbol']])
                        result = None if row['action'] == 'missing_price' else _source_verification(
                            source, row['symbol'], state['price_date'], dates, row['source_identity_sha256'], stop)
                        accept_verified(row, saved, result)
                else:
                    waiting = iter(state['rows']);active = {};exhausted = False
                    while active or not exhausted:
                        if _stopped(stop):raise InterruptedError('bulk native verification cancelled')
                        while not exhausted and len(active) < state['cpu_workers']:
                            row = next(waiting, None)
                            if row is None:exhausted = True;break
                            saved = _read_stock(path, state, row, state['records'][row['symbol']])
                            if row['action'] == 'missing_price':accept_verified(row, saved);continue
                            future = pool.submit(_worker_verify, row['symbol'], state['price_date'], dates, row['source_identity_sha256'])
                            active[future] = (row, saved)
                        if not active:break
                        completed, _ = wait(active, timeout=.2, return_when=FIRST_COMPLETED)
                        for future in completed:
                            row, saved = active.pop(future)
                            accept_verified(row, saved, future.result())
                if _stopped(stop):raise InterruptedError('bulk native verification cancelled')
                summaries = {row['symbol']: verified[row['symbol']] for row in state['rows']}
                _guard(state)
                refs = [{'entity_id': symbol, 'source_identity': value['identity'], 'row_hash': value['row_hash']}
                    for symbol, value in sorted(summaries.items()) if value.get('identity')]
                identity = {'format': 'incremental_native_summary_collection_v1', 'price_date': state['price_date'],
                    'calculation_scope': contract['calculation_scope'], 'projection_hashes': contract['projection_hashes'],
                    'mode': 'current_reference', 'training_eligible': False, 'required_symbols': sorted(summaries),
                    'input_refs_hash': stable_hash(refs), 'calendar_hash': contract['calendar_hash'],
                    'producer': contract['producer'], 'scope': 'full', 'completed_symbols': sorted(summaries),
                    'catalog_hash': stable_hash(contract['catalog']), 'bulk_id': state['bulk_id'], 'executor': state['executor']}
                native = {'identity': identity, 'summaries': summaries, 'input_refs': refs, 'complete': True,
                    'processed_symbols': sorted(summaries), 'diagnostics': {'scope': 'full', 'requested': len(summaries),
                        'processed': len(summaries), 'verified': sum(row['status'] == 'available' for row in summaries.values()),
                        'unavailable': sum(row['status'] != 'available' for row in summaries.values()),
                        'bulk_id': state['bulk_id'], 'native_workers': state['cpu_workers']}}
                from my_strategy.services.board_fluent import build_board_fluent
                def check_cancel():
                    if _stopped(stop):raise InterruptedError('bulk native preparation cancelled')
                    _guard(state)
                emit('final_publication')
                result = build_board_fluent(**state['paths'], calendar_dates=dates, price_date=state['price_date'],
                    with_native=True, native_result=native, migrate=False, component_store=store,
                    check_cancel=check_cancel)
                if not result['full_published']:raise ValueError('bulk final full publication was not committed')
                state.update(status='completed', full_published=True, generation_id=result['generation_id'], publication=result)
        except InterruptedError as exc:
            worker_stop.set();state.update(status='cancelled', error=str(exc))
        except Exception as exc:
            worker_stop.set();state.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        finally:
            if pool is not None:pool.shutdown(wait=True, cancel_futures=True)
            # Partial attempts retain expiring leases for explicit continuation.
            # Terminal tasks release only their own CAS protection, never pins
            # or other account/model/task owners' leases.
            if state['status'] in {'completed', 'cancelled', 'failed'}:
                with store._lock:
                    keys = [row[0] for row in store.connection.execute('SELECT cache_key FROM leases WHERE owner=?', (lease_owner,))]
                store.release_leases(keys, lease_owner)
            with store._lock:
                new_objects = [tuple(row) for row in store.connection.execute("SELECT sha256,size_bytes FROM objects WHERE state='active'")
                    if row[0] not in initial_objects]
                final_cache_bytes = store._cache_bytes()
            state['attempts'][-1]['cache_metrics'].update(cache_bytes_after=final_cache_bytes,
                net_cache_bytes=final_cache_bytes-initial_cache_bytes,
                new_cas_objects=len(new_objects), new_cas_object_bytes=sum(row[1] for row in new_objects),
                remaining_quota_bytes=max(0,store.quota_bytes-final_cache_bytes))
            store.quota_bytes = original_quota
            state['attempts'][-1]['finished_at'] = time.time()
            store.set_setting('incremental_plan:' + state['plan_id'], {'status': state['status'], 'bulk_id': state['bulk_id']})
            _save(path, state);emit('finished')
        return _receipt(state, path)
