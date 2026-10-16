"""Exercise the installed v1 agent stack using bounded synthetic inputs only.

This probe never loads a workstation model, downloads weights, registers a model,
or resumes an application task. Its scratch database and compact fitted model are
removed before returning; only the caller's requested verification report persists.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def mlp_probe(device):
    import numpy as np
    import pandas as pd
    from my_strategy.services.light_score_training import BASE_FEATURES, EXTRA_FEATURES, KRONOS_FEATURES, chronological_split_ranges
    from my_strategy.services.light_score_v2 import fit_light_score, score_light_samples

    rng = np.random.default_rng(81)
    dates = pd.bdate_range('2010-01-01', periods=1200)
    frame = pd.DataFrame(rng.normal(size=(1200, len(BASE_FEATURES + EXTRA_FEATURES))), columns=BASE_FEATURES + EXTRA_FEATURES)
    frame['symbol'] = '000001.SZ'
    frame['as_of'] = dates.strftime('%Y-%m-%d')
    frame['label_mature_at'] = (dates + pd.offsets.BDay(5)).strftime('%Y-%m-%d')
    frame['label'] = np.arange(len(frame)) % 3
    frame['feature_available_at'] = frame.as_of + 'T15:00:00+08:00'
    frame.loc[:50, 'support_distance'] = np.nan
    options = dict(source_identity={'fixture': 'synthetic_installation_only'},
                   label_contract={'horizon': 5, 'entry': 'next_verified_open', 'threshold': .003},
                   trials=[{'widths': [8, 4], 'max_epochs': 3, 'patience': 2}], device=device, budget_seconds=120, cpu_threads=1)
    ranges = chronological_split_ranges(frame)
    fitted = fit_light_score(frame, ranges, **options)
    restored = json.loads(json.dumps(fitted, allow_nan=False))
    row = frame.iloc[[-1]].copy()
    row['support_distance'] = np.nan
    day = row.iloc[0].as_of
    first = score_light_samples(fitted, row, as_of=day, historical_reconstruction=True)
    repeat = score_light_samples(restored, row, as_of=day, historical_reconstruction=True)
    linear = score_light_samples(restored, row, as_of=day, historical_reconstruction=True, model_kind='l2')
    paper = score_light_samples(restored, row, as_of=day, historical_reconstruction=True, mode='paper')
    require(first == repeat and first['status'] == 'scored', 'MLP JSON reload or optional missing-feature handling failed')
    require(math.isclose(sum(first['probabilities'][0].values()), 1., abs_tol=1e-6), 'MLP probabilities are not normalized')
    require(linear['scorer_id'] != first['scorer_id'] and linear['model_type'] == 'multinomial_logistic_L2', 'L2 comparator identity was lost')
    require(paper['qualification']['allow_increase'] is False, 'Synthetic model acquired paper allocation qualification')
    try:
        score_light_samples(restored, row.drop(columns=[BASE_FEATURES[0]]), as_of=day, historical_reconstruction=True)
    except ValueError as exc:
        require('Missing core score feature' in str(exc), 'Unexpected missing-feature failure')
    else:
        raise RuntimeError('Missing core input was scored')
    b_frame = frame.assign(kronos_state='missing', **{column: np.nan for column in KRONOS_FEATURES})
    absent_b = fit_light_score(b_frame, ranges, variant='B', **options)
    require(absent_b['status'] == 'pending' and absent_b['model_created'] is False, 'Absent Kronos history fabricated a B model')
    actual = fitted['runtime']['device']
    require(device == 'auto' or actual == device, 'Requested MLP training device silently fell back')
    return dict(status='passed', model_type='torch_mlp', variant='A', training_device=actual,
                calibration_device='cpu', inference_device='cpu', numeric_stage_contract='MLP fit on requested device; NumPy temperature and score inference on CPU',
                input_dimensions=fitted['architecture']['input_dimensions'], parameter_count=fitted['runtime']['parameter_count'],
                save_reload_identical=True, missing_optional_masked=True, missing_core_rejected=True,
                l2_same_input_comparator=True, b_without_observations_pending=True,
                paper_allocation_eligible=False, market_effectiveness_certified=False)


def incremental_probe(root):
    import numpy as np
    import pandas as pd
    from my_strategy.services.incremental_source import IncrementalSource
    from my_strategy.services.incremental_sync import execute_plan
    from my_strategy.storage.research_store import ResearchStore
    from my_strategy.storage.board_fluent import BoardFluentStore

    rng = np.random.default_rng(17)
    count = 340
    dates = pd.bdate_range('2020-01-02', periods=count).strftime('%Y-%m-%d').tolist()
    close = 20 * np.exp(np.cumsum(rng.normal(.0006, .025, count)))
    volume = rng.uniform(600_000, 2_000_000, count)
    raw = root / 'synthetic-prices.db'
    # The SQLite transaction context commits but does not release its file handle.
    with closing(sqlite3.connect(raw)) as con, con:
        con.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,has_trade_price INTEGER,source TEXT)')
        con.execute('CREATE TABLE securities(stock TEXT,name TEXT,first_date TEXT,last_date TEXT,raw_rows INTEGER)')
        con.executemany('INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?,?,?)', [
            ('000001.SZ', day, float(c*.998), float(c*1.009), float(c*.99), float(c), float(v), float(c*v), 1, 'sina_unadjusted_v1')
            for day, c, v in zip(dates, close, volume)])
        con.execute('INSERT INTO securities VALUES (?,?,?,?,?)', ('000001.SZ', 'synthetic fixture', dates[0], dates[-1], count))
    with ResearchStore(root / 'source-cache') as store:
        service = IncrementalSource(store=store, raw_db=raw)
        prefix = service.source('000001.SZ', dates[309], calendar_dates=dates)
        old = service._peek(prefix[3]['source_cache_key'])
        # New instance proves persistent JSON state restores across service lifetime.
        restarted = IncrementalSource(store=store, raw_db=raw)
        appended = restarted.source('000001.SZ', dates[-1], calendar_dates=dates)
        actual = appended[1].attrs['incremental']
        require(actual['mode'] == 'native_state_append' and actual['native_rows_appended'] == 30, 'Compiled native state did not append')
        current = restarted._peek(appended[3]['source_cache_key'])
        require(old['chunks'][0] == current['chunks'][0], 'Unchanged feature chunk was duplicated')
        entries = len(store.list(limit=1000))
        reused = restarted.source('000001.SZ', dates[-1], calendar_dates=dates)
        require(reused[1].attrs['incremental']['reused'] and len(store.list(limit=1000)) == entries, 'Identical source repeated computation/storage')
        with ResearchStore(root / 'independent-full') as full_store:
            full = IncrementalSource(store=full_store, raw_db=raw).source('000001.SZ', dates[-1], calendar_dates=dates)
        pd.testing.assert_frame_equal(appended[1], full[1], check_dtype=False, check_exact=True)
        require(appended[2] == full[2], 'Append changed causal event history')
        try:
            restarted.source('000002.SZ', dates[-1], calendar_dates=dates, stop=lambda: True)
        except InterruptedError:
            pass
        else:
            raise RuntimeError('Cancelled source computed')
        require(len(store.list(limit=1000)) == entries, 'Cancelled source wrote objects')
        store.put({'plan_id': 'cancelled-fixture'}, {'plan_id': 'cancelled-fixture', 'status': 'planned', 'expires_at': store.clock()+60}, namespace='incremental_plans')
        store.set_setting('incremental_plan:cancelled-fixture', {'status': 'cancelled'})
        try:
            execute_plan(store=store, plan_id='cancelled-fixture')
        except ValueError as exc:
            require('取消' in str(exc), 'Unexpected cancelled-plan failure')
        else:
            raise RuntimeError('Cancelled incremental plan restarted')
    boards = BoardFluentStore(root / 'synthetic-boards.db')
    cancelled = boards.acquire_sync('fixture', {'input': 'synthetic'})
    boards.finish_sync(cancelled['sync_id'], status='cancelled')
    try:
        boards.acquire_sync('fixture', {'input': 'synthetic'}, sync_id=cancelled['sync_id'])
    except ValueError as exc:
        require('terminal' in str(exc), 'Unexpected cancelled legacy task failure')
    else:
        raise RuntimeError('Cancelled legacy task reacquired')
    return dict(status='passed', fixture_rows=count, same_input_reused=True, append_matches_full=True,
                native_rows_appended=30, persistent_json_state_restored=True, immutable_chunk_reused=True,
                cancelled_source_writes_nothing=True, cancelled_terminal=True, legacy_cancelled_terminal=True,
                full_market_started=False, market_data_read=False)


def ledger_probe():
    import numpy as np
    import pandas as pd
    from my_strategy.services.position_simulation import run_position_simulation

    dates = pd.bdate_range('2025-01-01', periods=25).strftime('%Y-%m-%d').tolist()
    prices = 10 + np.arange(len(dates))*.01
    frame = pd.DataFrame(dict(date=dates, open=prices, high=prices+.2, low=prices-.2, close=prices,
                              volume=10_000_000, amount=100_000_000, has_trade_price=1))
    frame.loc[9:, ['open', 'close']] = 9.4
    frame.loc[9:, 'high'] = 9.6
    frame.loc[9:, 'low'] = 9.2
    symbol = '000001.SZ'
    event = dict(date=dates[5], event_id='installation-buy', confirmed=True, side='B', risk_boundary=float(frame.close.iloc[5])*.95,
                 forecast=dict(quality={'prediction_usable': True}, summary={'expected_return': .02, 'up_fraction': .8}))
    config = dict(initial_cash=1_000_000, commission=0, min_commission=0, slippage=0, transfer_fee=0,
                  ma_period=3, atr_period=3, max_holding_sessions=200, time_exit_sessions=200, policy='structure_kronos')
    kwargs = dict(config=config, calendar=dates)
    full = run_position_simulation({symbol: frame}, {symbol: [event]}, **kwargs)
    prefix = run_position_simulation({symbol: frame}, {symbol: [event]}, end=dates[7], **kwargs)
    resumed = run_position_simulation({symbol: frame}, {symbol: [event]}, state=json.loads(json.dumps(prefix['state'])), **kwargs)
    require([t['action'] for t in full['trades']] == ['BUY', 'SELL'], 'Synthetic ledger did not actually buy and sell')
    require(full['trades'][0]['date'] == dates[6] and full['trades'][-1]['date'] == dates[9], 'Next-open/hard-stop execution clock changed')
    require(full['trades'][-1]['reason'] == 'frozen_stop_gap', 'Missing later forecast blocked hard exit')
    for key in ('daily', 'trades', 'orders', 'decisions', 'metrics', 'lineage'):
        require(full[key] == resumed[key], 'Ledger continuation differs from one-pass execution: '+key)
    for row in full['daily']:
        shares = row['positions'][symbol]['shares']
        close = float(frame.loc[frame.date == row['date'], 'close'].iloc[0])
        require(math.isclose(row['equity'], row['cash'] + shares*close, abs_tol=1e-7), 'Cash/share/equity conservation failed')
    require(full['daily'][-1]['positions'][symbol]['shares'] == 0 and full['profit_certification'] is False, 'Fixture claims profitability qualification')
    return dict(status='passed', actual_fills=len(full['trades']), buy_fills=1, sell_fills=1, next_verified_open=True,
                share_cash_conserved=True, exit_without_score=True, continuation_matches_full=True, profit_certification=False)


def ma_profile_probe(device):
    """Fit the MA profile on generated bars; never create a model registry entry."""
    import numpy as np
    import pandas as pd
    from my_strategy.services.light_ma_features import MA_TREND_COLUMNS, prepare_ma_features
    from my_strategy.services.light_score_training import make_direction_training_frame, chronological_split_ranges
    from my_strategy.services.light_score_v2 import fit_light_score, score_light_samples

    dates = pd.bdate_range('2012-01-02', periods=1300)
    days = dates.strftime('%Y-%m-%d').tolist()
    close = 10*np.exp(np.cumsum(np.random.default_rng(54).normal(0, .012, len(days))))
    bars = pd.DataFrame(dict(symbol='000001.SZ', date=dates, open=close*.999,
        high=close*1.02, low=close*.98, close=close, volume=10000., amount=100000.,
        source='tushare', has_trade_price=1))
    features = prepare_ma_features(bars, calendar_dates=days)
    prefix = prepare_ma_features(bars.iloc[:210], calendar_dates=days)
    pd.testing.assert_frame_equal(features.loc[:209, list(MA_TREND_COLUMNS)], prefix[list(MA_TREND_COLUMNS)], check_exact=True)
    require(not features.score_input_eligible.iloc[:119].any(), 'MA warmup became eligible')
    missing_calendar = prepare_ma_features(bars)
    require(not missing_calendar.score_input_eligible.any(), 'MA scored without a verified calendar')
    training = make_direction_training_frame(features, days)
    training = training.loc[training.score_input_eligible].copy()
    fitted = fit_light_score(training, chronological_split_ranges(training), feature_profile='ma_trend_v1',
        source_identity={'fixture': 'synthetic_installation_only'}, label_contract={'horizon': 5},
        trials=[{'widths': [8, 4], 'max_epochs': 3, 'patience': 2}], device=device,
        budget_seconds=120, cpu_threads=1)
    restored = json.loads(json.dumps(fitted, allow_nan=False))
    row = training.iloc[[-1]].copy()
    options = dict(as_of=row.iloc[0].as_of, historical_reconstruction=True)
    mlp = score_light_samples(fitted, row, **options)
    require(mlp == score_light_samples(restored, row, **options), 'MA JSON reload changed predictions')
    l2 = score_light_samples(restored, row, model_kind='l2', **options)
    require(fitted['feature_profile'] == 'ma_trend_v1' and fitted['architecture']['input_dimensions'] == 20
        and mlp['features'] == l2['features'] == list(MA_TREND_COLUMNS), 'MA/L2 input population or columns changed')
    require(mlp['status'] == l2['status'] == 'scored' and mlp['scorer_id'] != l2['scorer_id']
        and mlp['model_version'] == l2['model_version'], 'MA/L2 paired identity was lost')
    require(not fitted['identity']['feature_contract']['kronos_input'], 'MA claims to learn Kronos input')
    for value in (mlp, l2):
        require(math.isclose(sum(value['probabilities'][0].values()), 1., abs_tol=1e-6)
            and value['qualification']['paper_allocation_eligible'] is False, 'MA/L2 probability or qualification failed')
    row['score_input_eligible'] = False
    for kind in ('mlp', 'l2'):
        unavailable = score_light_samples(restored, row, model_kind=kind, **options)
        require(unavailable['status'] == 'unavailable' and unavailable['scores'] == [None]
            and not unavailable['qualification']['allow_increase'], 'MA/L2 bypassed source quality')
    actual = fitted['runtime']['device']
    require(device == 'auto' or actual == device, 'Requested MA device silently fell back')
    return dict(status='passed', feature_profile='ma_trend_v1', input_dimensions=20,
        training_device=actual, calibration_device='cpu', inference_device='cpu',
        same_input_l2=True, distinct_scorer_same_model=True, save_reload_identical=True,
        causal_prefix_identical=True, missing_calendar_rejected=True, ineligible_stays_unscored=True,
        consumes_kronos=False, active_model_registered=False, paper_allocation_eligible=False)


def board_scope_probe(root):
    from copy import deepcopy
    from my_strategy.core.market_scope import require_supported_symbol
    from my_strategy.core.run_context import stable_hash
    from my_strategy.services.board_calculation_scope import calculation_membership, verify_calculation_membership, calculation_contract
    from my_strategy.storage.board_fluent import BoardFluentStore

    source = dict(board_id='synthetic-concept', snapshot_hash=stable_hash({'fixture': 'source-proof'}),
        members=['000001.SZ', '600000.SH', '430001.BJ', '920001.BJ', '430001.NQ', '900901.SH'],
        expected_count=6, complete=True)
    original = deepcopy(source)
    projection = calculation_membership(source)
    require(source == original and projection['source_snapshot_hash'] == source['snapshot_hash'], 'Board source evidence was rewritten')
    require(projection['eligible_members'] == ['000001.SZ', '600000.SH'] and projection['eligible_count'] == 2
        and len(projection['excluded_members']) == 4 and projection['usable_membership'], 'Board calculation denominator includes out-of-scope members')
    require(verify_calculation_membership(projection, source) == projection, 'Board projection failed its own identity')
    contract = calculation_contract({source['board_id']: projection})
    require(contract['calculation_scope'] == 'sh_sz_a_v1' and contract['required_symbols'] == projection['eligible_members']
        and contract['projection_hashes'] == {source['board_id']: projection['projection_hash']}, 'Board publication identity lost its calculation population')
    changed = deepcopy(projection); changed['eligible_members'].append('430001.BJ')
    try:
        verify_calculation_membership(changed, source)
    except ValueError:
        pass
    else:
        raise RuntimeError('Altered board projection was accepted')
    empty = calculation_membership({**source, 'members': ['430001.BJ'], 'expected_count': 1})
    require(not empty['usable_membership'] and empty['eligible_count'] == 0, 'Empty board scope became complete')
    aliases = ['430001', '430001.BJ', 'BJ430001', 'BSE:920001', '920001.BSE']
    for symbol in aliases:
        try:
            require_supported_symbol(symbol)
        except ValueError as exc:
            require('北交所' in str(exc), 'Unexpected BJ boundary failure')
        else:
            raise RuntimeError('Public request accepted Beijing symbol: '+symbol)
    require(require_supported_symbol('SH600000') == '600000.SH'
        and require_supported_symbol('900901.SH') == '900901.SH', 'Scope check reclassified explicit Shanghai symbols')
    board = dict(board_id=source['board_id'], kind='concept', mode='current_reference', training_eligible=False,
        price_date='2024-01-02', snapshot_hash=source['snapshot_hash'], expected_members=6)
    legacy = dict(cache_key=stable_hash({'fixture': 'legacy-mixed-board'}), cache_identity={'fixture': True},
        mode='current_reference', training_eligible=False, qualification='pending', price_date='2024-01-02',
        boards=[board], stocks=[])
    fluent = BoardFluentStore(root / 'scope-publications.db')
    old_id = fluent.save_market(legacy)
    require(fluent.current_generation() is None and fluent.load_market(legacy['cache_key']) == legacy
        and fluent.board(board['board_id'], generation_id=old_id), 'Historical mixed board became the default or disappeared')
    scoped = deepcopy(legacy)
    scoped.update(cache_key=stable_hash({'fixture': 'scoped-board', 'projection': projection['projection_hash']}),
        calculation_scope='sh_sz_a_v1', calculation_membership={source['board_id']: projection})
    scoped['boards'][0].update(calculation_scope='sh_sz_a_v1', calculation_membership=projection,
        source_snapshot_hash=source['snapshot_hash'], projection_hash=projection['projection_hash'], expected_members=2)
    new_id = fluent.save_market(scoped, publication_inputs=contract)
    require(fluent.current_generation()['generation_id'] == new_id
        and fluent.verify_generation(new_id)['calculation_scope'] == 'sh_sz_a_v1'
        and fluent.load_market(legacy['cache_key']) == legacy, 'Scoped board publication changed historical source facts')
    malformed = deepcopy(scoped)
    malformed['cache_key'] = stable_hash({'fixture': 'tampered-projected-board'})
    malformed['boards'][0]['expected_members'] = 6
    try:
        fluent.save_market(malformed, publication_inputs=contract)
    except ValueError:
        pass
    else:
        raise RuntimeError('Board publication accepted the old mixed denominator')
    return dict(status='passed', calculation_scope=contract['calculation_scope'],
        source_member_count=6, eligible_count=2, excluded_count=4, source_preserved=True,
        projection_tamper_rejected=True, empty_scope_unusable=True, bj_aliases_rejected=len(aliases),
        historical_requires_explicit=True, scoped_publication_verified=True, mixed_denominator_rejected=True,
        source_authentication='projection consumes source identity; source proof authentication remains in KB reader')


def task_maintenance_probe(root):
    import threading
    from types import SimpleNamespace
    from my_strategy.services.task_maintenance import TaskMaintenance, MaintenanceConflict
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.storage.research_store import ResearchStore

    directory = root / 'maintenance'; directory.mkdir()
    tasks_db = directory / 'tasks.db'
    with closing(sqlite3.connect(tasks_db)) as con, con:
        con.execute('CREATE TABLE czsc_tasks(job_id TEXT PRIMARY KEY,fingerprint TEXT,status TEXT,payload TEXT)')
        for job, state in [('done', 'succeeded'), ('active', 'running')]:
            payload = dict(job_id=job, kind='scan', status=state, spec={}, created_at='2026-01-01T00:00:00+08:00')
            con.execute('INSERT INTO czsc_tasks VALUES(?,?,?,?)', (job, job, state, json.dumps(payload)))
    with ResearchStore(directory / 'research') as store:
        light = SimpleNamespace(store=store, jobs={}, stops={}, lock=threading.RLock())
        results = ResultStore(directory / 'results.db', directory / 'runs')
        tasks = SimpleNamespace(db_path=tasks_db, results=results, _lock=threading.RLock())
        service = TaskMaintenance(tasks, light)
        page = service.list(page_size=1)
        require(page['total'] == 2 and page['items'][0]['job_id'] == 'active'
            and page['items'][0]['can_delete'] is False, 'Task history pagination/active protection failed')
        refs = [dict(backend='legacy', job_id=job) for job in ('done', 'active')]
        deleted = service.delete_records(refs)
        require(deleted['deleted'] == refs[:1] and len(deleted['skipped']) == 1, 'Task deletion did not protect active work')
        # The active receipt is synthetic. End it before exercising cleanup on
        # the same isolated fixture; no application task is submitted/resumed.
        with closing(sqlite3.connect(tasks_db)) as con, con:
            con.execute('DELETE FROM czsc_tasks WHERE job_id=?', ('active',))
        results.save('analysis', {'run_id': 'synthetic-cleanup', 'symbol': '000001.SZ', 'fixture': True})
        path = results.runs_root / 'synthetic-cleanup'
        plan = service.preview()
        require(path.exists() and any(row['type'] == 'analysis_run' for row in plan['candidates']), 'Cleanup preview modified or omitted fixture')
        try:
            service.execute(plan['plan_id'], '0'*64)
        except MaintenanceConflict:
            pass
        else:
            raise RuntimeError('Cleanup accepted a changed plan hash')
        result = service.execute(plan['plan_id'], plan['plan_sha256'])
        require(not result['failed'] and not result['skipped'] and not path.exists()
            and result['deleted_bytes'] == plan['candidate_bytes'], 'Exact synthetic cleanup failed')
    return dict(status='passed', pagination=True, active_delete_rejected=True,
        terminal_record_deleted=True, preview_non_destructive=True, wrong_plan_hash_rejected=True,
        exact_fixture_cleanup=True, production_state_touched=False)


def spawn_probe(root):
    """Real native work in real spawn children; prediction is a synthetic stub."""
    import hashlib
    import multiprocessing
    import os
    import threading
    from types import MethodType, SimpleNamespace
    from unittest.mock import patch
    from filelock import FileLock
    from my_strategy.services import analysis_methods as methods, method_scan_acceleration as accelerated
    from my_strategy.services import kronos_prediction, light_research
    from my_strategy.services.light_research import LightResearch
    from my_strategy.storage.czsc_results import ResultStore
    from my_strategy.storage.research_store import ResearchStore

    raw = root / 'synthetic-prices.db'
    with closing(sqlite3.connect(raw)) as con, con:
        con.execute("INSERT INTO stock_daily_normalized SELECT '600000.SH',date,open,high,low,close,volume,amount,has_trade_price,source FROM stock_daily_normalized WHERE stock='000001.SZ'")
        con.execute("INSERT INTO securities SELECT '600000.SH',name,first_date,last_date,raw_rows FROM securities WHERE stock='000001.SZ'")
        days = [row[0] for row in con.execute("SELECT date FROM stock_daily_normalized WHERE stock='000001.SZ' ORDER BY date LIMIT 160")]
    lights = []
    def make(name):
        directory = root / name; directory.mkdir()
        store = ResearchStore(directory / 'research')
        light = SimpleNamespace(raw_db=raw, store=store, compute_lock=FileLock(str(directory / 'compute.lock')),
            calendar=lambda: (days, {'sha256': 'synthetic-installation-calendar'}),
            prediction=lambda symbol, end, **kwargs: dict(status='completed', as_of=end,
                summary={'terminal_return_median': .02}, quality={'prediction_usable': True},
                result_ref={'cache_key': 'synthetic-forecast-'+symbol}))
        light.source = MethodType(LightResearch.source, light)
        lights.append(light)
        return light, ResultStore(directory / 'results.db', directory / 'runs')
    spec = dict(analysis_method='structure_kronos', start=days[0], end=days[-1], cpu_workers=2,
        max_predictions=2, budget_seconds=120, lookback=128, horizon=5, samples=32, seed=42,
        device='cpu', usage_mode='historical')
    before = {child.pid for child in multiprocessing.active_children()}
    try:
        with patch.object(kronos_prediction, 'asset_status', lambda: {'ready': True}), \
                patch.object(light_research, 'prediction_identity', lambda *args: {'fixture': 'synthetic-no-model'}):
            base, base_results = make('thread-baseline')
            worker, worker_results = make('spawn-complete')
            baseline = methods.scan_methods(base, base_results, spec)
            actual = accelerated.scan_methods(worker, worker_results, spec)
            for key in ('rows', 'frozen_identity', 'population', 'coverage', 'method_context', 'status'):
                require(actual[key] == baseline[key], 'Spawn differs from original scan: '+key)
            require(actual['status'] == 'succeeded' and actual['coverage']['evaluated'] == 2, 'Spawn fixture did not finish both stocks')
            evidence = actual['compute_info']['execution_adapter']
            require(evidence['start_method'] == 'spawn' and evidence['worker_pids']
                and os.getpid() not in evidence['worker_pids'] and evidence['source_tasks_received'] == 2
                and evidence['algorithm_bytecode_preserved'] and evidence['prediction_owner'] == 'parent_process', 'Spawn execution proof is incomplete')
            require(evidence['sha256'] == hashlib.sha256(Path(accelerated.__file__).read_bytes()).hexdigest(), 'Spawn adapter source hash differs')
            cancelled, cancelled_results = make('spawn-cancel')
            requested = threading.Event(); predict = cancelled.prediction
            def prediction(*args, **kwargs):
                value = predict(*args, **kwargs); requested.set(); return value
            def check_cancel():
                if requested.is_set():
                    raise InterruptedError('synthetic cancellation')
            cancelled.prediction = prediction
            result = accelerated.scan_methods(cancelled, cancelled_results, {**spec, 'cpu_workers': 1}, check_cancel=check_cancel)
            require(result['status'] == 'cancelled' and result['coverage']['evaluated'] == 1
                and result['coverage']['deferred'] == 1 and result['coverage']['failed'] == 0, 'Spawn cancellation lost pending population')
            # The clone retains the original budget guard before worker launch.
            deadline_light, deadline_results = make('spawn-deadline')
            try:
                accelerated.scan_methods(deadline_light, deadline_results, {**spec, 'budget_seconds': .000001})
            except TimeoutError:
                pass
            else:
                raise RuntimeError('Expired spawn budget started work')
            require(deadline_results.list() == [], 'Expired spawn budget published a result')
    finally:
        for light in lights:
            light.store.close()
    require({child.pid for child in multiprocessing.active_children()} <= before, 'Spawn probe leaked child processes')
    return dict(status='passed', start_method='spawn', stocks=2, real_native_preparation=True,
        rows_and_frozen_identity_identical=True, cancelled_preserves_pending=True, expired_budget_rejected=True,
        children_joined=True, algorithm_bytecode_preserved=True, prediction='synthetic_stub_no_weights',
        adapter_sha256=evidence['sha256'], worker_pids=evidence['worker_pids'], parent_pid=os.getpid())


def run_probe(device='cpu'):
    import os
    from unittest.mock import patch
    from my_strategy.services.kronos_assets import asset_status
    with tempfile.TemporaryDirectory(prefix='khquant-v1-installed-') as temporary:
        root = Path(temporary)
        # Also isolate legacy RunContext writers and inherited paths in children.
        isolated = {'KHQUANT_HOME': root, 'KHQUANT_DATA_ROOT': root/'data',
            'KHQUANT_ARTIFACT_ROOT': root/'artifacts', 'KHQUANT_METADATA_ROOT': root/'data/metadata',
            'KHQUANT_RAW_DB': root/'synthetic-prices.db', 'KHQUANT_PROCESSED_DB': root/'processed.db',
            'KHQUANT_OUTPUT_ROOT': root/'output', 'KHQUANT_LOG_ROOT': root/'logs',
            'KHQUANT_KRONOS_ROOT': root/'absent-kronos', 'KHQUANT_CURRENT_STRATEGY_CONFIG': root/'current.json'}
        with patch.dict(os.environ, {key: str(value) for key, value in isolated.items()}):
            absent = asset_status(root / 'absent-kronos')
            require(absent['ready'] is False and absent['offline_only'] is True and len(absent['missing']) == 4, 'Fresh installation fabricated Kronos availability')
            result = dict(status='passed', family='agent_mlp_sim_v1', fixture='synthetic_installation_only', device=device,
                          mlp=mlp_probe(device), incremental_source=incremental_probe(root), simulation_ledger=ledger_probe(),
                          v101=dict(ma_profile=ma_profile_probe(device), board_scope=board_scope_probe(root),
                              task_maintenance=task_maintenance_probe(root), spawn=spawn_probe(root)),
                          absent_kronos=dict(status='passed', ready=False, downloads_started=False),
                          publication_allowed=False, market_effectiveness_certified=False, business_models_loaded=False)
    result['scratch_removed'] = not root.exists()
    require(result['scratch_removed'], 'Probe left synthetic state behind')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--internal-probe', action='store_true', required=True)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.device), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
