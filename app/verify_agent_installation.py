"""Exercise the installed v1 agent stack using bounded synthetic inputs only.

This probe never loads a workstation model, downloads weights, registers a model,
or resumes an application task. Its scratch database and compact fitted model are
removed before returning; only the caller's requested verification report persists.
"""
from __future__ import annotations

import argparse
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
    with sqlite3.connect(raw) as con:
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


def run_probe(device='cpu'):
    from my_strategy.services.kronos_assets import asset_status
    with tempfile.TemporaryDirectory(prefix='khquant-v1-installed-') as temporary:
        root = Path(temporary)
        absent = asset_status(root / 'absent-kronos')
        require(absent['ready'] is False and absent['offline_only'] is True and len(absent['missing']) == 4, 'Fresh installation fabricated Kronos availability')
        result = dict(status='passed', family='agent_mlp_sim_v1', fixture='synthetic_installation_only', device=device,
                      mlp=mlp_probe(device), incremental_source=incremental_probe(root), simulation_ledger=ledger_probe(),
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
