"""Actual raw identity and fake producers keep sync tests off business data."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from my_strategy.services import daily_feature_sync as sync
from my_strategy.tests.test_board_features import market
from my_strategy.tests.test_board_fluent import actual_market


class Store:
    def __init__(self): self.jobs = {}; self.generations = set(); self.blocked = False; self.published_full = None
    def current_generation(self, level): return copy.deepcopy(self.published_full) if level == 'full' else None
    def get_sync(self, identity): return copy.deepcopy(self.jobs.get(identity))
    def acquire_sync(self, fingerprint, input_identity, *, sync_id=None, **kwargs):
        existing = self.jobs.get(sync_id) if sync_id else next((v for v in self.jobs.values() if v['fingerprint'] == fingerprint), None)
        if existing:
            if existing['input_identity'] != input_identity: raise ValueError('identity mismatch')
            if self.blocked: return {**copy.deepcopy(existing), 'owned': False, 'reused': False}
            if existing['status'] == 'completed': return {**copy.deepcopy(existing), 'owned': False, 'reused': True}
            existing['status'] = 'running'
        else:
            identity = sync_id or 'sync-' + str(len(self.jobs) + 1)
            existing = dict(sync_id=identity, fingerprint=fingerprint, input_identity=copy.deepcopy(input_identity),
                checkpoint={}, status='running', stage=None, result=None)
            self.jobs[identity] = existing
        return {**copy.deepcopy(existing), 'owned': not self.blocked, 'reused': False}
    def heartbeat_sync(self, identity, stage=None, checkpoint=None):
        job = self.jobs[identity]
        if stage: job['stage'] = stage
        job['checkpoint'].update(copy.deepcopy(checkpoint or {}))
    def finish_sync(self, identity, status, result=None, error=None):
        self.jobs[identity].update(status=status, result=copy.deepcopy(result), error=error)
    def verify_generation(self, identity, level):
        if identity not in self.generations: raise ValueError('published generation SHA drift')
        return {'market_published': True, 'full_published': level == 'full'}


@pytest.fixture
def raw_db(tmp_path):
    path = tmp_path / 'raw.db'
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER)')
        con.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)', [
            ('600000.SH', '2010-01-04', 10, 12, 9, 11, 100, 1100, 'sina_unadjusted_v1', 1),
            ('600000.SH', '2026-10-12', 20, 22, 19, 21, 100, 2100, 'sina_unadjusted_v1', 1),
            ('600000.SH', '2026-10-13', 22, 24, 21, 23, 100, 2300, 'unverified_fixture', 1),
            ('000001.SZ', '2026-10-14', 2, 4, 1, 3, 100, 300, 'sina_unadjusted_v1', 0)])
    return path


@pytest.fixture
def environment(tmp_path, raw_db, monkeypatch):
    store = Store(); state = {'current': True, 'catalog_hash': 'catalog-a', 'board_count': 2}
    calls = {'market': [], 'native': [], 'refresh': []}
    monkeypatch.setattr(sync, '_store', lambda _: store)
    monkeypatch.setattr(sync, '_calendar', lambda _: (['2010-01-04', '2026-10-12'], 'calendar-fixture'))
    body = {'code_hashes': {'producer.py': 'a' * 64}, 'engine_contract': {'engine': 'fixture'},
        'native_schema_hash': 'b' * 64, 'native_config_hash': 'c' * 64, 'native_version': 'v1',
        'feature_schema_hashes': {'market': 'd' * 64, 'profile': 'e' * 64},
        'feature_start': '2023-01-01', 'liquidity_universe': {'policy': 'fixture'}}
    monkeypatch.setattr(sync, '_code_identity', lambda config, root: {'code': 'fixture', 'config': config,
        'native_input_contract': copy.deepcopy(body)})
    monkeypatch.setattr(sync, '_catalog', lambda _: copy.deepcopy(state))
    monkeypatch.setattr(sync, '_validate_cached_native', lambda *args, **kwargs: None)
    def build(**kwargs):
        calls['market'].append(kwargs)
        identity = 'generation-full' if kwargs['with_native'] else 'generation-market'
        store.generations.add(identity)
        return {'generation_id': identity, 'market_published': True, 'full_published': kwargs['with_native'], 'training_started': False}
    def native(**kwargs):
        calls['native'].append(kwargs)
        return {'status': 'ready', 'source_native_run_id': 'native-fixture', 'raw_source_run_id': 'raw-fixture', 'training_started': False}
    def refresh(**kwargs):
        calls['refresh'].append(kwargs); state['current'] = True
        return {'run_id': 'kb-fixture', 'status': 'succeeded'}
    monkeypatch.setattr(sync, '_build_fluent', build); monkeypatch.setattr(sync, '_run_native', native)
    monkeypatch.setattr(sync, '_refresh_kb', refresh)
    return SimpleNamespace(store=store, state=state, calls=calls,
        options={'raw_db_path': raw_db, 'artifacts_root': tmp_path / 'artifacts'})


def test_actual_qualified_date_and_complete_older_content_are_readonly(raw_db):
    original = raw_db.read_bytes(); identity = sync.qualified_raw_identity(raw_db)
    assert identity['price_date'] == '2026-10-12' and identity['qualified_rows'] == 2
    assert raw_db.read_bytes() == original
    with sqlite3.connect(raw_db) as con:
        con.execute("UPDATE stock_daily_normalized SET close=10.5 WHERE date='2010-01-04'")
    changed = sync.qualified_raw_identity(raw_db)
    assert changed['price_date'] == identity['price_date'] and changed['raw_content_sha256'] != identity['raw_content_sha256']


@pytest.mark.parametrize('required_date', ['2026-10-13', '2026-10-14'])
def test_unverified_or_unpriced_new_day_cannot_publish_older_qualified_date(environment, required_date):
    e = environment
    with pytest.raises(sync.SyncTargetDateUnavailable, match='differs from required target date'):
        sync.sync_daily_features(**e.options, required_price_date=required_date)
    assert not e.store.jobs and not e.calls['market'] and not e.calls['native'] and not e.calls['refresh']


def test_exact_required_qualified_date_is_enforced_on_reuse_too(environment):
    e = environment
    result = sync.sync_daily_features(**e.options, required_price_date='2026-10-12')
    assert result['status'] == 'completed'
    with pytest.raises(sync.SyncTargetDateUnavailable):
        sync.sync_daily_features(**e.options, sync_id=result['sync_id'], required_price_date='2026-10-13')
    assert len(e.calls['market']) == 2 and len(e.calls['native']) == 1


@pytest.mark.parametrize('required_date', ['20261012', '2026-10-12T15:00:00+08:00', 'not-a-date'])
def test_required_price_date_must_use_canonical_iso_date_before_lease(environment, required_date):
    e = environment
    with pytest.raises(ValueError, match='ISO YYYY-MM-DD'):
        sync.sync_daily_features(**e.options, required_price_date=required_date)
    assert not e.store.jobs and not e.calls['market']


def test_market_is_published_before_native_and_complete_reuse_ignores_worker_choices(environment):
    e = environment; progress = []
    result = sync.sync_daily_features(**e.options, progress=progress.append)
    assert result['status'] == 'completed' and result['training_started'] is False
    assert [c['with_native'] for c in e.calls['market']] == [False, True]
    assert e.calls['market'][0]['price_date'] == e.calls['native'][0]['price_date'] == '2026-10-12'
    native_stage = [item for item in progress if item.get('stage') == 'native_prepare']
    assert native_stage and result['market_published'] and result['full_published']
    reused = sync.sync_daily_features(**e.options, cpu_workers=2, stock_batch_size=17)
    assert reused['reused'] and reused['sync_id'] == result['sync_id'] and len(e.calls['native']) == 1
    assert len(e.calls['market']) == 2 and not e.calls['refresh']


def test_old_raw_revision_rejects_explicit_resume_and_creates_new_sync(environment, raw_db):
    e = environment; result = sync.sync_daily_features(**e.options)
    with sqlite3.connect(raw_db) as con:
        con.execute("UPDATE stock_daily_normalized SET amount=1200 WHERE date='2010-01-04'")
    with pytest.raises(ValueError, match='input changed'):
        sync.sync_daily_features(**e.options, sync_id=result['sync_id'])
    changed = sync.sync_daily_features(**e.options)
    assert changed['sync_id'] != result['sync_id'] and len(e.calls['native']) == 2


def test_source_429_preserves_partial_market_and_retry_resumes_frozen_catalog(environment, monkeypatch):
    e = environment; e.state['current'] = False
    def refused(**kwargs):
        e.calls['refresh'].append(kwargs)
        kwargs['on_progress']({'run_id': 'kb-partial', 'completed': 1, 'total': 2})
        return {'status': 'partial', 'run_id': 'kb-partial', 'error': 'HTTP 429'}
    monkeypatch.setattr(sync, '_refresh_kb', refused)
    partial = sync.sync_daily_features(**e.options, kb_request_budget=7)
    assert partial['status'] == 'partial' and partial['market_published'] and not partial['full_published']
    assert not e.calls['native'] and partial['error'] == 'HTTP 429'
    assert e.calls['refresh'][0]['limit_budget'] == 7 and e.calls['refresh'][0]['min_interval'] == 1.3
    def refreshed(**kwargs):
        assert kwargs['resume'] == 'kb-partial'
        e.state['current'] = True; return {'status': 'succeeded', 'run_id': 'kb-partial'}
    monkeypatch.setattr(sync, '_refresh_kb', refreshed)
    result = sync.sync_daily_features(**e.options, sync_id=partial['sync_id'])
    assert result['status'] == 'completed' and result['sync_id'] == partial['sync_id']


def test_native_failure_keeps_market_and_full_failure_reuses_completed_native(environment, monkeypatch):
    e = environment; real_build = sync._build_fluent
    def build(**kwargs):
        if kwargs['with_native']: raise ValueError('full generation failed')
        return real_build(**kwargs)
    monkeypatch.setattr(sync, '_build_fluent', build)
    first = sync.sync_daily_features(**e.options)
    assert first['status'] == 'partial' and first['market_published'] and not first['full_published']
    assert e.store.get_sync(first['sync_id'])['checkpoint']['native_result']['source_native_run_id'] == 'native-fixture'
    monkeypatch.setattr(sync, '_build_fluent', real_build)
    second = sync.sync_daily_features(**e.options, sync_id=first['sync_id'])
    assert second['status'] == 'completed' and len(e.calls['native']) == 1


def test_cancel_preserves_native_ids_but_same_sync_cannot_resume(environment, monkeypatch):
    e = environment; flag = [False]
    def native(**kwargs):
        flag[0] = True
        kwargs['progress']({'stage': 'native_prepare', 'raw_source_run_id': 'raw-started', 'resume_native_run_id': 'native-started'})
    monkeypatch.setattr(sync, '_run_native', native)
    first = sync.sync_daily_features(**e.options, check_cancel=lambda: flag[0])
    assert first['status'] == 'cancelled' and first['market_published']
    checkpoint = e.store.get_sync(first['sync_id'])['checkpoint']
    assert checkpoint['resume_native_run_id'] == 'native-started'
    flag[0] = False
    monkeypatch.setattr(sync, 'qualified_raw_identity', lambda *a,**k:pytest.fail('cancelled sync performed full identity read'))
    monkeypatch.setattr(sync, '_run_native', lambda *a,**k:pytest.fail('cancelled native resumed'))
    with pytest.raises(ValueError,match='cancelled.*terminal'):
        sync.sync_daily_features(**e.options, sync_id=first['sync_id'])
    assert e.store.get_sync(first['sync_id'])['status']=='cancelled'


def test_corrupted_completed_generation_fails_closed(environment):
    e = environment; sync.sync_daily_features(**e.options); e.store.generations.clear()
    with pytest.raises(ValueError, match='SHA drift'): sync.sync_daily_features(**e.options)


def test_current_kb_revision_creates_new_sync(environment):
    e = environment; first = sync.sync_daily_features(**e.options)
    e.state['catalog_hash'] = 'catalog-b'
    next_result = sync.sync_daily_features(**e.options)
    assert next_result['sync_id'] != first['sync_id'] and next_result['kb_catalog_hash'] == 'catalog-b'
    assert sync.sync_daily_features(**e.options)['reused'] is True


def test_source_changes_during_native_preparation_do_not_publish_full_generation(environment, raw_db, monkeypatch):
    e = environment
    def native(**kwargs):
        with sqlite3.connect(raw_db) as con:
            con.execute("UPDATE stock_daily_normalized SET close=10.5 WHERE date='2010-01-04'")
        return {'status': 'ready', 'source_native_run_id': 'native-fixture', 'training_started': False}
    monkeypatch.setattr(sync, '_run_native', native)
    result = sync.sync_daily_features(**e.options)
    assert result['status'] == 'partial' and result['market_published'] and not result['full_published']
    assert len(e.calls['market']) == 1 and 'changed during preparation' in result['error']


def test_other_live_owner_is_not_replaced(environment):
    e = environment; e.store.blocked = True
    result = sync.sync_daily_features(**e.options)
    assert result['status'] == 'busy' and not e.calls['market'] and not e.calls['native']


def test_unverified_calendar_date_never_starts_a_source(environment, monkeypatch):
    e = environment; monkeypatch.setattr(sync, '_calendar', lambda _: (['2010-01-04'], 'fixture'))
    with pytest.raises(ValueError, match='verified calendar'): sync.sync_daily_features(**e.options)
    assert not e.store.jobs and not e.calls['refresh']


def test_task_manager_cancel_exception_is_normalized():
    class Cancelled(RuntimeError): pass
    def cancel(): raise Cancelled('task cancelled')
    with pytest.raises(sync.SyncCancelled): sync._cancel(cancel)


def _native_options(tmp_path, raw_db):
    return dict(run_dir=tmp_path / 'run', price_date='2026-10-12', raw_db_path=raw_db,
        native_cache_db_path=None, calendar_run_id='fixture-calendar', cpu_workers=1,
        stock_batch_size=2, source_root=None, artifacts_root=tmp_path / 'artifacts',
        config=None, checkpoint={}, progress=lambda _: None, check_cancel=None)


def test_supervised_prepare_handles_concatenated_progress_and_never_uses_stdout_pipe(tmp_path, raw_db, monkeypatch):
    real_popen = subprocess.Popen; launches = []; progress = []
    program = 'print(\'SOURCE_BATCH {"completed":1,"total":1}OTHER_TEXT\',flush=True);print(\'NATIVE_CACHE_RESULT {"status":"ready","source_native_run_id":"fixture","training_started":false}\',flush=True)'
    def spawn(args, **kwargs):
        launches.append((args, kwargs))
        return real_popen([sys.executable, '-c', program], **kwargs)
    monkeypatch.setattr(sync.subprocess, 'Popen', spawn)
    options = _native_options(tmp_path, raw_db); options['progress'] = progress.append
    options['checkpoint'] = {'source_native_run_id': 'verified-native', 'raw_source_run_id': 'verified-raw'}
    result = sync._run_native(**options)
    assert result['source_native_run_id'] == 'fixture' and result['training_started'] is False
    assert any(item.get('completed') == 1 for item in progress)
    assert launches[0][1]['stdout'] != subprocess.PIPE
    assert 'my_strategy.scripts.refresh_native_cache' in launches[0][0]
    assert 'my_strategy.scripts.train_trend' not in launches[0][0]
    assert '--source-native-run-id' in launches[0][0] and 'verified-native' in launches[0][0]


def test_supervised_cancel_waits_for_owned_native_child_tree_exit(tmp_path, raw_db, monkeypatch):
    real_popen = subprocess.Popen; marker = tmp_path / 'child.pid'; parent = []
    program = 'import subprocess,sys,time,pathlib; c=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);pathlib.Path(' + repr(str(marker)) + ').write_text(str(c.pid));time.sleep(60)'
    def spawn(_args, **kwargs):
        child = real_popen([sys.executable, '-c', program], **kwargs); parent.append(child); return child
    monkeypatch.setattr(sync.subprocess, 'Popen', spawn)
    options = _native_options(tmp_path, raw_db); options['check_cancel'] = marker.exists
    with pytest.raises(sync.SyncCancelled): sync._run_native(**options)
    assert parent[0].poll() is not None
    identity = int(marker.read_text())
    try: assert psutil.Process(identity).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess: pass


def test_update_hook_defaults_to_lightweight_and_skip_is_explicit(tmp_path, monkeypatch):
    from my_strategy.scripts import update_daily_data as update
    from my_strategy.services import light_daily
    calls = []
    monkeypatch.setattr(sync, 'sync_daily_features', lambda **kwargs:pytest.fail('default update started full market native'))
    monkeypatch.setattr(light_daily, 'sync_light_daily', lambda **kwargs: calls.append(kwargs) or {'status': 'partial', 'mode':'lightweight'})
    monkeypatch.setattr(update, 'get_data_access', lambda: SimpleNamespace(settings=SimpleNamespace(raw_db=tmp_path / 'raw.db')))
    args = SimpleNamespace(skip_feature_sync=False, feature_kb_db=None, feature_fluent_db=None,
        feature_native_cache_db=None, feature_calendar_run_id=None, feature_sync_id=None,
        feature_cpu_workers=6, feature_stock_batch_size=360, feature_kb_budget=2000,
        feature_kb_min_interval=1.3, feature_source_root=None, feature_artifacts_root=tmp_path)
    result = update._sync_after_update(args, {'update_id': 'actual-update', 'success': 1, 'end_date': '2026-10-12',
        'total': 1, 'target_date_covered': 1}, {})
    assert result['status'] == 'partial' and result['mode']=='lightweight'
    assert calls[0]['raw_db_path'] == tmp_path / 'raw.db' and 'end_date' not in calls[0]
    assert calls[0]['required_price_date'] == '2026-10-12'
    args.skip_feature_sync = True
    assert update._sync_after_update(args, {}, {})['status'] == 'skipped' and len(calls) == 1
    args.skip_feature_sync = False
    assert update._sync_after_update(args, {}, {'has_errors': True})['status'] == 'blocked'


def test_explicit_full_update_hook_sigint_reaches_supervisor_check_cancel(tmp_path, monkeypatch):
    from my_strategy.scripts import update_daily_data as update
    def work(**kwargs):
        signal.raise_signal(signal.SIGINT)
        assert kwargs['check_cancel']() is True
        raise sync.SyncCancelled('interrupt fixture')
    monkeypatch.setattr(sync, 'sync_daily_features', work)
    monkeypatch.setattr(update, 'get_data_access', lambda: SimpleNamespace(settings=SimpleNamespace(raw_db=tmp_path / 'raw.db')))
    args = SimpleNamespace(skip_feature_sync=False, feature_mode='full',feature_kb_db=None, feature_fluent_db=None,
        feature_native_cache_db=None, feature_calendar_run_id=None, feature_sync_id=None,
        feature_cpu_workers=6, feature_stock_batch_size=360, feature_kb_budget=2000,
        feature_kb_min_interval=1.3, feature_source_root=None, feature_artifacts_root=tmp_path)
    before = signal.getsignal(signal.SIGINT)
    assert update._sync_after_update(args, {'update_id': 'fixture', 'success': 1,
        'total': 1, 'target_date_covered': 1}, {})['status'] == 'cancelled'
    assert signal.getsignal(signal.SIGINT) == before


def test_standalone_cli_partial_returns_success_and_restores_signal_handlers(monkeypatch):
    from my_strategy.scripts import sync_daily_features as cli
    calls = []
    monkeypatch.setattr(sync, 'sync_daily_features', lambda **kwargs: calls.append(kwargs) or {'status': 'partial', 'training_started': False})
    before = signal.getsignal(signal.SIGTERM)
    assert cli.main(['--resume-sync-id', 'sync-fixture', '--kb-min-interval', '1.3', '--price-date', '2026-10-12']) == 0
    assert calls[0]['sync_id'] == 'sync-fixture' and signal.getsignal(signal.SIGTERM) == before
    assert calls[0]['required_price_date'] == '2026-10-12'


def test_actual_fluent_lease_market_and_full_publication_with_fake_native_producer(actual_market, tmp_path, monkeypatch):
    from my_strategy.core.run_context import stable_hash
    from my_strategy.services import board_native_summary
    from my_strategy.storage.board_fluent import BoardFluentStore
    value = actual_market
    monkeypatch.setattr(sync, '_calendar', lambda _: (value['dates'], 'calendar-fixture'))
    monkeypatch.setattr(sync, '_code_identity', lambda *args: {'code': 'fixture'})
    monkeypatch.setattr(sync, '_catalog', lambda _: {'current': True, 'catalog_hash': 'complete-fixture', 'board_count': 3})
    monkeypatch.setattr(sync, '_validate_cached_native', lambda *args, **kwargs: None)
    monkeypatch.setattr(sync, '_run_native', lambda **_: {'status': 'ready', 'source_native_run_id': 'fake-source', 'training_started': False})
    day = value['dates'][-1]
    rows = {}
    for symbol in value['symbols']:
        row = {'symbol': symbol, 'price_date': day, 'mode': 'current_reference', 'training_eligible': False,
            'status': 'unavailable', 'reason_codes': ['fixture_native_not_acquired']}
        row['row_hash'] = stable_hash(row); rows[symbol] = row
    native = {'complete': True, 'summaries': rows, 'input_refs': [],
        'identity': {'price_date': day, 'mode': 'current_reference', 'training_eligible': False,
            'calendar_hash': stable_hash(value['dates']), 'source': 'fixture'}}
    monkeypatch.setattr(board_native_summary, 'collect_native_summaries', lambda **_: native)
    # Commit the fixture's existing WAL into the main file before taking a
    # physical baseline; later GC of fixture writers must not change it.
    connection = sqlite3.connect(value['kb'])
    try:
        checkpoint = connection.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()
        assert checkpoint[0] == 0
    finally:
        connection.close()
    raw_before = value['path'].read_bytes(); kb_before = value['kb'].read_bytes()
    options = {'raw_db_path': value['path'], 'kb_db_path': value['kb'], 'fluent_db_path': value['fluent'],
        'artifacts_root': tmp_path / 'artifacts'}
    result = sync.sync_daily_features(**options)
    assert result['status'] == 'completed' and result['full_published'], result
    store = BoardFluentStore(value['fluent'], readonly=True)
    assert store.get_sync(result['sync_id'])['status'] == 'completed'
    assert store.verify_generation(result['generation_id'], 'full')['full_published'] is True
    assert store.current_generation('market')['generation_id'] == store.current_generation('full')['generation_id']
    assert sync.sync_daily_features(**options)['reused'] is True
    assert value['path'].read_bytes() == raw_before and value['kb'].read_bytes() == kb_before


def test_prepared_source_hints_are_persisted_and_bound_to_sync_identity(environment):
    e = environment
    result = sync.sync_daily_features(**e.options, source_native_run_id='verified-native', raw_source_run_id='verified-raw')
    job = e.store.get_sync(result['sync_id'])
    assert job['input_identity']['prepared_source']['source_native_run_id'] == 'verified-native'
    assert e.calls['native'][0]['checkpoint']['source_native_run_id'] == 'verified-native'
    assert job['checkpoint']['raw_source_run_id'] == 'verified-raw'
    assert sync.sync_daily_features(**e.options, sync_id=result['sync_id'])['reused'] is True


def _published_source(environment):
    e = environment; root = e.options['artifacts_root']
    report_path = root / 'runs/published-raw/reports/raw_source.json'; report_path.parent.mkdir(parents=True)
    from my_strategy.core.run_context import stable_hash
    report = {'data_end': '2026-10-12', 'calendar_hash': stable_hash(['2010-01-04', '2026-10-12'])}
    report_path.write_text(json.dumps(report))
    body = sync._code_identity(None, None)['native_input_contract']
    contract = {key: value for key, value in body.items() if key not in {'feature_start', 'liquidity_universe'}}
    contract['config'] = {key: body[key] for key in ('feature_start', 'liquidity_universe')}
    e.store.published_full = {'generation_id': 'published-full', 'full_ready': True, 'full_published': True,
        'native_identity': {'price_date': '2026-10-12', 'mode': 'current_reference', 'training_eligible': False,
            'source_native_run_id': 'published-native', 'raw_source_run_id': 'published-raw',
            'raw_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(), 'generation_contract': contract}}
    e.store.generations.add('published-full')
    return e.store.published_full, report_path


def test_default_same_day_sync_reuses_verified_published_source_even_if_membership_changed(environment, monkeypatch):
    e = environment; _published_source(e); e.state['catalog_hash'] = 'new-membership'
    def forbidden(**kwargs): pytest.fail('same-day native reuse rematerialized source assets')
    monkeypatch.setattr(sync, '_run_native', forbidden)
    result = sync.sync_daily_features(**e.options)
    assert result['status'] == 'completed' and result['source_native_run_id'] == 'published-native'
    assert e.calls['market'][1]['raw_source_run_id'] == 'published-raw'
    assert e.store.get_sync(result['sync_id'])['checkpoint']['native_result']['reused_published_source'] is True


@pytest.mark.parametrize('reason', ['config', 'calendar', 'invalid_raw', 'unpublished'])
def test_invalid_published_native_identity_uses_safe_prepare(environment, monkeypatch, reason):
    e = environment; candidate, report_path = _published_source(e)
    if reason == 'config': candidate['native_identity']['generation_contract']['native_config_hash'] = 'changed'
    elif reason == 'calendar':
        report = json.loads(report_path.read_text()); report['calendar_hash'] = 'changed'
        report_path.write_text(json.dumps(report))
        candidate['native_identity']['raw_report_sha256'] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    elif reason == 'unpublished': candidate['full_published'] = False
    else:
        def invalid(*args, **kwargs): raise ValueError('current old raw revision differs from producer')
        monkeypatch.setattr(sync, '_validate_cached_native', invalid)
    result = sync.sync_daily_features(**e.options)
    assert result['status'] == 'completed' and len(e.calls['native']) == 1
    assert result['source_native_run_id'] == 'native-fixture'
