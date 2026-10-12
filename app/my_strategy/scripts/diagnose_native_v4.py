"""Independent development architecture and L2 threshold Broker diagnostics.

Frozen candidates, risk plans, MA opinions and raw-union/signal-pinned MA exits
are shared by all arms. Only the real entry head and final entry probability
threshold differ. Full-prefix arbitration is evaluated once per unique arm;
windows use causal slices and fresh actual Broker cash. No models, catalogs or
publication aliases are modified. The linear .1 architecture/control arm is
also the identical .1 threshold arm, so its actual ledgers are reused explicitly.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_agent_consensus import build_consensus
from my_strategy.services.czsc_analysis import strategy_config
from my_strategy.services.czsc_backtest import execute_decisions
from my_strategy.services.czsc_high_return_features_v4 import entry_feature_row, event_prefix_rows
from my_strategy.services.czsc_high_return_models_v4 import HeadSession, bind_inputs, load_head
from my_strategy.services.czsc_high_return_pipeline_v4 import (
    _bundle_for, _head_path, _predict, model_identity, pinned_exit_policy,
)
from my_strategy.services.czsc_high_return_runtime_v4 import read_run, runtime_overlay
from my_strategy.services.czsc_research_data import VERIFIED_SOURCES
from my_strategy.services.czsc_research_ml import _metrics
from my_strategy.services.czsc_trade_episodes_v4 import _max_exit_callback
from my_strategy.scripts.validate_native_v4 import (
    CONTROL_STOCKS, DEVELOPMENT_WINDOWS, _actual_episodes, _cash_failure, _cohort,
    _persist_account, _save, _turnover, _worker_init, aggregate_accounts,
    map_final_points_to_decisions, verify_actual_account,
)
from my_strategy.web_dashboard.high_return import _calendar, _prepare

VERSION = 'native_v4_architecture_threshold_development_diagnostic_20261012'
ARCHITECTURES = ('linear', 'tree', 'mlp')
THRESHOLDS = (.05, .1, .2, .35, .5)
POLICY_KEYS = ('candidate_policy_version', 'candidate_policy_hash', 'risk_policy_hash')


def diagnostic_arms():
    arms = [{'id': f'{architecture}_entry_0p1', 'architecture': architecture,
             'entry_threshold': .1, 'comparisons': ['architecture']}
            for architecture in ARCHITECTURES]
    arms[0]['comparisons'].append('threshold_grid')
    for threshold in THRESHOLDS:
        if threshold != .1:
            arms.append({'id': 'linear_entry_' + str(threshold).replace('.', 'p'),
                         'architecture': 'linear', 'entry_threshold': threshold,
                         'comparisons': ['threshold_grid']})
    return arms


def architecture_catalog(root: Path, report: dict, architecture: str, *, loader=load_head):
    """Bind actual architecture manifests; never rename a linear checkpoint."""
    if architecture not in ARCHITECTURES:
        raise ValueError('unsupported diagnostic architecture')
    catalogs, coverage = [], []
    for original in report['model_bundles']:
        path = root / 'models' / 'entry' / original['name'] / architecture
        if not (path / 'manifest.json').is_file():
            coverage.append({'fold': original['name'], 'architecture': architecture,
                             'status': 'missing_architecture_head_no_linear_substitution'})
            continue
        manifest = loader(path)
        if manifest['head'] != 'entry' or manifest['architecture'] != architecture:
            raise ValueError('diagnostic architecture/head identity mismatch')
        linear = loader(_head_path(root, original, 'entry'))
        if manifest['target_contract_hash'] != linear['target_contract_hash']:
            raise ValueError('architecture candidate/risk/continuation targets differ')
        if manifest['available_at'] != original['available_at'] or manifest['fold']['name'] != original['name']:
            raise ValueError('architecture chronological fold binding differs')
        if manifest['target_contract']['source_snapshot_hash'] != report['source_snapshot']['snapshot_hash']:
            raise ValueError('architecture training source snapshot differs')
        if float(original['policy']['entry_threshold']) != .1:
            raise ValueError('original fold threshold is not the predefined .1 control')
        catalogs.append({**copy.deepcopy(original), 'architecture': architecture,
                         'heads': {'entry': {'path': str(path.relative_to(root)),
                                            'manifest_sha256': manifest['manifest_sha256']}}})
        coverage.append({'fold': original['name'], 'architecture': architecture, 'status': 'available',
                         'path': str(path), 'manifest_sha256': manifest['manifest_sha256'],
                         'target_contract_hash': manifest['target_contract_hash'],
                         'calibration_metrics': manifest.get('calibration_metrics'),
                         'sample_counts': manifest.get('sample_counts'),
                         'available_at': manifest['available_at'],
                         'actual_training_completed_at': manifest['training_completed_at']})
    return catalogs, coverage


def replace_entry_evaluation(original: dict, evaluation: dict):
    """Separate wrapper only: native facts, MA and risk remain identical objects."""
    row = dict(original)
    row['agents'] = {**original['agents'], 'high_return': evaluation}
    evidence = dict(original['candidate_evidence'])
    evidence['evaluations'] = {**evidence['evaluations'], 'high_return': evaluation}
    row['candidate_evidence'] = evidence
    row['entry_probability'] = row['high_return_probability'] = evaluation['probability']
    row['model_resolution'] = {**evaluation.get('model_identity', {}),
                               'probability': evaluation['probability'],
                               'reason_codes': evaluation.get('reason_codes', []),
                               'publication_allowed': False, 'historical_reconstruction': True}
    return row


def score_architecture(overlay, calendar, root, report, catalog, architecture, session=None):
    """Use exactly runtime's entry inputs and frozen initial-flat chart scope."""
    if architecture == 'linear':
        return {row['date']: copy.deepcopy(row['agents']['high_return']) for row in overlay['rows']}
    session = session or HeadSession()
    cfg = report['config']; initial_cash = float(cfg['initial_cash'])
    candidates = {row['date']: row for row in overlay['candidates']}
    features = overlay['features'].to_dict('records')
    prefix = event_prefix_rows(overlay['features'], overlay['signal_points'], calendar)
    result = {}
    for index, (feature, original) in enumerate(zip(features, overlay['rows'], strict=True)):
        day = original['date']; candidate = candidates.get(day)
        evaluation = copy.deepcopy(original['agents']['high_return'])
        probability = manifest = bundle = None
        if not candidate or not original['risk_eligible']:
            # Preserve no-candidate/input/risk reasons without attempting inference.
            evaluation['probability'] = None
            evaluation['model_identity'] = model_identity(None, None, root, day, overlay['candidate_evidence_policy'])
            result[day] = evaluation
            continue
        bundle = _bundle_for(day, catalog, head='entry', model_root=root, usage_mode='historical')
        status = 'model_unavailable' if bundle is None else 'scored'
        if bundle is not None:
            state = {'date': day, 'close': float(feature['close']), 'cash': initial_cash, 'equity': initial_cash}
            values = entry_feature_row(feature, state, candidate, original['raw_risk_plan'],
                                       initial_cash=initial_cash, event_prefix=prefix[index])
            values['input_eligible'] = True
            probability, manifest = _predict(session, values, root, bundle, 'entry', mode='historical',
                policy=overlay['candidate_evidence_policy'], source_snapshot_hash=cfg['source_snapshot_hash'])
            if probability is None:
                status = 'input_ineligible'
        evaluation.update(status=status, probability=probability,
            threshold=bundle['policy']['entry_threshold'] if bundle else cfg['default_entry_threshold'],
            model_identity=model_identity(manifest, bundle, root, day, overlay['candidate_evidence_policy']),
            reason_codes=[] if status == 'scored' else [status])
        result[day] = evaluation
    return result


def causal_window(consensus, rows, start, end, symbol, max_weight):
    """Future rows/events cannot enter a fresh account, even from full replay."""
    events = [event for event in consensus.get('events', []) if start <= event['date'] <= end]
    prefix = [row for row in rows if row['date'] <= end]
    clipped = {**consensus, 'events': events}
    decisions, blocked = map_final_points_to_decisions(prefix, clipped, symbol=symbol,
        start=start, end=end, max_weight=max_weight)
    return decisions, blocked, events


def coverage_for_window(overlay, evaluations, start, end):
    rows = [row for row in overlay['rows'] if start <= row['date'] <= end]
    facts = [event for event in overlay['signal_points'] if start <= event['date'] <= end]
    sources = {}
    for source in ('ma', 'structure', 'wyckoff'):
        own = [event for event in facts if event['source'] == source]
        sources[source] = {'raw_facts': len(own),
            'direction_buy': sum(e['side'] == 'BUY' and e.get('direction_eligible') is True for e in own),
            'direction_sell': sum(e['side'] == 'SELL' and e.get('direction_eligible') is True for e in own),
            'risk_qualified_buy': sum(e['side'] == 'BUY' and e.get('candidate_eligible') is True for e in own),
            'input_qualified_days': sum(row['agents'][source].get('input_eligible') is True
                for row in rows) if source != 'ma' else sum(row['agents']['ma']['direction_state']['input_eligible'] is True for row in rows)}
    states = Counter(evaluations[row['date']]['status'] for row in rows)
    probabilities = [evaluations[row['date']]['probability'] for row in rows
                     if evaluations[row['date']]['probability'] is not None]
    return {'bars': len(rows), 'sources': sources, 'l2_raw_states': dict(states),
            'scored_rows': len(probabilities), 'probability_quantiles':
            {str(q): float(np.quantile(probabilities, q)) for q in (0., .1, .25, .5, .75, .9, 1.)} if probabilities else {}}


def forward_label_scores(root, report, symbol, catalog, session):
    """Read matured real labels and their original actual-account input vectors.

    These classification diagnostics have actual raw-union candidate cash state;
    they are separate from the initial-flat chart forecasts driving Broker arms.
    """
    frame = pd.read_parquet(root / 'dataset' / 'l2_final' / symbol.replace('.', '_') / 'entries.parquet')
    if not frame.empty and not frame.source_snapshot_hash.eq(report['source_snapshot']['snapshot_hash']).all():
        raise ValueError('mature label source snapshot mismatch')
    days = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    ends = pd.to_datetime(frame.label_end, errors='coerce').dt.strftime('%Y-%m-%d')
    available = frame.label_available.fillna(False) & frame.input_eligible.fillna(False)
    available &= pd.to_numeric(frame.label, errors='coerce').isin([0., 1.]) & ends.notna()
    if (available & (ends <= days)).any():
        raise ValueError('entry label maturity not strictly after decision')
    frame = frame.loc[available].copy()
    predictions = []
    for bundle in catalog:
        manifest = load_head(_head_path(root, bundle, 'entry'))
        routed = frame.loc[[(_bundle_for(str(day)[:10], catalog, head='entry', model_root=root,
                    usage_mode='historical') or {}).get('name') == bundle['name'] for day in frame.date]].copy()
        if routed.empty:
            continue
        bound = bind_inputs(routed, manifest['columns'], 'entry')
        bound.attrs['label_contract'] = manifest['target_contract']
        probabilities = session.predict(bound, _head_path(root, bundle, 'entry'),
            expected_head='entry', expected_contract=manifest['target_contract'], usage_mode='historical_reconstruction')
        for record, probability in zip(routed.to_dict('records'), probabilities, strict=True):
            predictions.append({'symbol': symbol, 'date': str(record['date'])[:10],
                'label_end': str(record['label_end'])[:10], 'label': float(record['label']),
                'probability': float(probability), 'fold': bundle['name'],
                'source_ma': bool(record['source_ma']), 'source_structure': bool(record['source_structure']),
                'source_wyckoff': bool(record['source_wyckoff']),
                'manifest_sha256': manifest['manifest_sha256'], 'target_contract_hash': manifest['target_contract_hash'],
                'net_return': float(record['net_return']), 'r_multiple': float(record['r_multiple']),
                'economic_group_id': record.get('economic_group_id')})
    return predictions


def classification_metrics(predictions, windows, label_cutoff):
    result = []
    for name, (start, finish) in windows.items():
        end = min(finish, label_cutoff)
        selected = [row for row in predictions if start <= row['date'] <= end and row['label_end'] <= end
                    and np.isfinite(row['probability'])]
        groups = {'union': selected, **{source: [r for r in selected if r['source_' + source]]
                                        for source in ('ma', 'structure', 'wyckoff')}}
        for source, rows in groups.items():
            metrics = _metrics(np.array([r['label'] for r in rows]), np.array([r['probability'] for r in rows])) if rows else {'rows': 0}
            thresholds = []
            for threshold in THRESHOLDS:
                predicted = [r for r in rows if r['probability'] >= threshold]
                positives = sum(r['label'] == 1 for r in rows)
                true_positives = sum(r['label'] == 1 for r in predicted)
                thresholds.append({'threshold': threshold, 'accepted': len(predicted),
                    'precision': true_positives / len(predicted) if predicted else None,
                    'recall': true_positives / positives if positives else None})
            result.append({'window': name, 'source': source, **metrics, 'thresholds': thresholds,
                'label_cutoff': end, 'independent_window_status': 'development_only',
                'population': 'mature_actual_raw_union_reference_candidate_account_features',
                'chart_population_differs': 'chart_forecasts_use_frozen_initial_flat_candidate_cash_state'})
    return result


def compact_consensus(consensus):
    return {**{key: value for key, value in consensus.items() if key != 'rows'}, 'rows': [
        {key: row[key] for key in ('date', 'action', 'emitted', 'reason_codes', 'buy_base_score',
                                 'sell_base_score', 'buy_total_score', 'buy_quality_bonus', 'funnel')}
        for row in consensus['rows']]}


def evaluate_stock(payload):
    _worker_init()
    symbol, calendar = payload['symbol'], payload['calendar']
    root, report = read_run(payload['model_run_id']); cfg = report['config']
    initial_cash = float(cfg['initial_cash']); settings = strategy_config(cfg['strategy'])
    output = Path(payload['output']) / 'accounts' / symbol.replace('.', '_')
    accounts, failures, coverage, label_predictions = [], [], {}, {}
    try:
        item = _prepare(symbol, payload['end'], calendar)
        overlay = runtime_overlay(item['features'], calendar, model_run_id=payload['model_run_id'],
                                  as_of=payload['end'], device='cpu')
        from my_strategy.services.czsc_signal_points_v4 import build_signal_points
        points = build_signal_points(overlay['features'], cfg, native_pack=overlay['native_pack'],
                                      ma_routes={row['date']: row for row in overlay['ma_routes']})
        exit_template = pinned_exit_policy(overlay['features'], calendar, points, root, report['ma_bundles'], cfg)
        native_ok = True
        raw_windows = {name: item['raw'].loc[pd.to_datetime(item['raw'].date) <= min(finish, payload['end'])]
                          .reset_index(drop=True) for name, (_, finish) in payload['windows'].items()}
    except Exception as exc:
        native_ok = False
        failures.append({'symbol': symbol, 'stage': 'native_shared_overlay', 'error': repr(exc)})
    score_cache = {}
    for arm in payload['arms']:
        architecture = arm['architecture']; consensus = None
        try:
            if not native_ok:
                raise ValueError('shared native overlay unavailable')
            if architecture not in score_cache:
                session = HeadSession(); catalog, head_coverage = architecture_catalog(root, report, architecture)
                evaluations = score_architecture(overlay, calendar, root, report, catalog, architecture, session)
                score_cache[architecture] = evaluations
                coverage[architecture] = {'heads': head_coverage,
                    'windows': {name: coverage_for_window(overlay, evaluations, start, min(finish, payload['end']))
                                for name, (start, finish) in payload['windows'].items()},
                    'inference': dict(session.diagnostics)}
                try:
                    label_predictions[architecture] = forward_label_scores(root, report, symbol, catalog, session)
                except Exception as exc:
                    failures.append({'symbol': symbol, 'stage': 'mature_forward_classification',
                                     'architecture': architecture, 'error': repr(exc)})
                    label_predictions[architecture] = []
            evaluations = score_cache[architecture]
            branch_rows = [replace_entry_evaluation(row, evaluations[row['date']]) for row in overlay['rows']]
            consensus = build_consensus(branch_rows, symbol=symbol, model_family='high_return',
                version='czsc_agent_consensus_v4', config={'evidence_valid_bars': 3,
                    'high_return_threshold': arm['entry_threshold']})
            _save(output / arm['id'] / 'full_prefix_consensus.json', compact_consensus(consensus))
            _save(output / arm['id'] / 'score_lineage.json', evaluations)
        except Exception as exc:
            failures.append({'symbol': symbol, 'stage': 'branch_arbitration', 'arm': arm['id'], 'error': repr(exc)})
        for name, (start, finish) in payload['windows'].items():
            end = min(finish, payload['end']); decisions = None; blocked = []; events = []
            if consensus is not None:
                try:
                    decisions, blocked, events = causal_window(consensus, branch_rows, start, end,
                        symbol, float(settings['execution']['max_weight']))
                    if len(decisions) != len(raw_windows[name]):
                        raise ValueError('causal window raw/decisions misaligned')
                    _save(output / arm['id'] / name / 'broker_decisions.json', decisions)
                    _save(output / arm['id'] / name / 'blocked_final_points.json', blocked)
                except Exception as exc:
                    failures.append({'symbol': symbol, 'stage': 'causal_window', 'arm': arm['id'],
                                     'window': name, 'error': repr(exc)})
            for cost in payload['costs']:
                summary = {'symbol': symbol, 'window': name, 'start': start, 'end': end,
                    'variant': arm['id'], 'architecture': architecture, 'entry_threshold': arm['entry_threshold'],
                    'comparisons': arm['comparisons'], 'evidence_bars': 3, 'slippage_multiplier': cost,
                    'initial_cash': initial_cash, 'status': 'cash_failure', 'publication_allowed': False,
                    'final_point_count': len(events), 'blocked_final_buys':
                    sum(r['reason'] != 'chart_sell_outside_frozen_source_exit_policy' for r in blocked),
                    'chart_sell_outside_frozen_source_exit_policy':
                    sum(r['reason'] == 'chart_sell_outside_frozen_source_exit_policy' for r in blocked),
                    'holding_policy': 'unchanged_raw_union_source_exits_signal_pinned_native_MA_exit_hard_stop_max60',
                    'exit_threshold_unchanged': cfg['signal_points']['ma_exit_threshold']}
                try:
                    if decisions is None:
                        raise ValueError('branch decisions unavailable')
                    execution = copy.deepcopy(settings); execution['execution']['slippage'] *= cost
                    extra = exit_template.new_replay()
                    callback = _max_exit_callback({day: i for i, day in enumerate(calendar)},
                        int(cfg['high_return']['max_holding_sessions']), extra.recipe_hash, extra_policy=extra)
                    account_id = f"{payload['run_id']}:{symbol}:{name}:{arm['id']}:slip{cost}"
                    account = execute_decisions(symbol, raw_windows[name], decisions, start, initial_cash,
                        execution, account_id, market_dates=calendar, verified_sources=VERIFIED_SOURCES,
                        entry_policy='risk', exit_policy=callback)
                    checks = verify_actual_account(account, calendar=calendar, decisions=decisions,
                                                   initial_cash=initial_cash, symbol=symbol)
                    account['episodes'] = _actual_episodes(account)
                    complete = [e for e in account['episodes'] if e['complete']]
                    summary.update(status='completed', checks=checks, metrics=account['metrics'],
                        turnover=_turnover(account, initial_cash), buy_fills=sum(f['action'] == 'BUY' for f in account['ledger']),
                        sell_fills=sum(f['action'] == 'SELL' for f in account['ledger']),
                        completed_round_trips=len(complete),
                        win_rate=sum(e['net_return'] > 0 for e in complete) / len(complete) if complete else None,
                        rejection_reasons=dict(Counter(r['reason'] for r in account['rejections'])),
                        execution_config_hash=stable_hash(execution), consensus_config_hash=consensus['config_hash'])
                except Exception as exc:
                    failures.append({'symbol': symbol, 'stage': 'actual_broker', 'arm': arm['id'],
                                     'window': name, 'slippage_multiplier': cost, 'error': repr(exc)})
                    account = _cash_failure(symbol, start, end, calendar, initial_cash, repr(exc))
                    summary.update(metrics=account['metrics'], turnover=0., buy_fills=0, sell_fills=0,
                                   rejection_reasons={}, failure=account['failure'])
                accounts.append(_persist_account(output / arm['id'] / name / f'slippage_{cost:g}', account, summary))
    result = {'symbol': symbol, 'accounts': accounts, 'failures': failures, 'coverage': coverage,
              'label_predictions': label_predictions}
    _save(output / 'stock_summary.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-run-id', required=True)
    parser.add_argument('--cohort', default='training')
    parser.add_argument('--cpu-workers', type=int, default=2)
    parser.add_argument('--calendar-run-id')
    parser.add_argument('--end', default='2026-10-09')
    parser.add_argument('--windows', default=','.join(DEVELOPMENT_WINDOWS))
    parser.add_argument('--slippage-multipliers', default='1,2')
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_workers <= 6:
        raise ValueError('diagnostic cpu-workers must be between 1 and 6')
    root, report = read_run(args.model_run_id)
    calendar, calendar_id = _calendar(args.calendar_run_id)
    names = args.windows.split(',')
    if len(set(names)) != len(names) or set(names) - set(DEVELOPMENT_WINDOWS):
        raise ValueError('unknown or repeated development windows')
    costs = [float(x) for x in args.slippage_multipliers.split(',')]
    if not costs or len(set(costs)) != len(costs) or set(costs) - {1., 2.}:
        raise ValueError('only unique normal/double slippage arms are allowed')
    windows = {name: DEVELOPMENT_WINDOWS[name] for name in names}
    symbols = _cohort(args.cohort, report); end = pd.Timestamp(args.end).date().isoformat()
    arms = diagnostic_arms()
    catalogs = {a: architecture_catalog(root, report, a)[1] for a in ARCHITECTURES}
    spec = {'version': VERSION, 'model_run_id': args.model_run_id, 'symbols': symbols,
        'cohort_hash': stable_hash(symbols), 'windows': windows, 'arms': arms,
        'slippage_multipliers': costs, 'evidence_valid_bars': 3, 'calendar_hash': stable_hash(calendar),
        'calendar_run_id': calendar_id, 'raw_end': end, 'frozen_feature_end': report['source_snapshot']['source_data_end'],
        'initial_cash_per_account': float(report['config']['initial_cash']), 'source_snapshot_hash': report['source_snapshot']['snapshot_hash'],
        'default_fold_threshold_and_unified_control': .1, 'default_catalog_sha256': stable_hash(report['model_bundles']),
        'architecture_head_identity': catalogs, 'independent_window_status': 'pending', 'publication_allowed': False,
        'no_automatic_parameter_or_model_selection': True,
        'reused_actual_arm': 'linear_entry_0p1 is both architecture-linear and threshold-.1; identical actual ledgers',
        'continuation': 'frozen_risk_raw_union_signal_pinned_MA_exit_hard_stop_max60_no_L2_exit_replacement',
        'cost_stress': 'double_slippage_only_commission_minimum_and_stamp_tax_unchanged',
        'classification_population': 'original_l2_final_actual_reference_candidate_inputs; matured_by_window_end',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    context = create_run_context(task='czsc-native-v4-development-diagnostic', as_of_date=end,
        config=spec, data_version=report['source_snapshot']['snapshot_hash'], stocks=symbols,
        start_date=min(start for start, _ in windows.values()), end_date=end)
    _save(context.subdir('config') / 'diagnostic.json', spec)
    print('RUN_ID ' + context.run_id, flush=True)
    payload = {'model_run_id': args.model_run_id, 'calendar': calendar, 'windows': windows,
               'arms': arms, 'costs': costs, 'end': end, 'output': str(context.run_dir()), 'run_id': context.run_id}
    results = []
    def accept(stock):
        results.append(stock)
        print(json.dumps({'completed': len(results), 'requested': len(symbols), 'symbol': stock['symbol'],
                          'actual_accounts': len(stock['accounts']), 'failures': len(stock['failures'])}), flush=True)
    if args.cpu_workers == 1:
        for symbol in symbols:
            accept(evaluate_stock({**payload, 'symbol': symbol}))
    else:
        with ProcessPoolExecutor(max_workers=args.cpu_workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_init) as pool:
            pending = {pool.submit(evaluate_stock, {**payload, 'symbol': symbol}): symbol for symbol in symbols}
            for future in as_completed(pending):
                symbol = pending[future]
                try:
                    accept(future.result())
                except Exception as exc:
                    stock = {'symbol': symbol, 'accounts': [], 'coverage': {}, 'label_predictions': {},
                             'failures': [{'stage': 'worker', 'error': repr(exc)}]}
                    for arm in arms:
                        for name, (start, finish) in windows.items():
                            finish = min(finish, end)
                            for cost in costs:
                                account = _cash_failure(symbol, start, finish, calendar, spec['initial_cash_per_account'], repr(exc))
                                summary = {'symbol': symbol, 'window': name, 'variant': arm['id'], 'architecture': arm['architecture'],
                                    'entry_threshold': arm['entry_threshold'], 'comparisons': arm['comparisons'], 'evidence_bars': 3,
                                    'slippage_multiplier': cost, 'status': 'cash_failure', 'metrics': account['metrics'],
                                    'turnover': 0., 'failure': account['failure']}
                                stock['accounts'].append(_persist_account(context.run_dir() / 'accounts' /
                                    symbol.replace('.', '_') / arm['id'] / name / f'slippage_{cost:g}', account, summary))
                    accept(stock)
    results.sort(key=lambda r: symbols.index(r['symbol']))
    aggregate = aggregate_accounts(results, symbols, spec['initial_cash_per_account'])
    arm_by_id = {arm['id']: arm for arm in arms}
    for item in aggregate:
        item.update(architecture=arm_by_id[item['variant']]['architecture'],
                    entry_threshold=arm_by_id[item['variant']]['entry_threshold'],
                    comparisons=arm_by_id[item['variant']]['comparisons'])
    classification = {architecture: classification_metrics(
        [row for stock in results for row in stock['label_predictions'].get(architecture, [])], windows,
        report['source_snapshot']['source_data_end']) for architecture in ARCHITECTURES}
    out = {'run_id': context.run_id, 'config': spec, 'aggregate': aggregate,
        'account_summaries': [a for stock in results for a in stock['accounts']],
        'stock_records': [{k: v for k, v in stock.items() if k not in {'accounts', 'label_predictions'}} for stock in results],
        'classification': classification, 'actual_account_count': sum(len(s['accounts']) for s in results),
        'failed_cash_accounts': sum(a['status'] != 'completed' for s in results for a in s['accounts']),
        'status': 'completed' if all(not s['failures'] for s in results) else 'completed_with_failures_retained',
        'independent_window_status': 'pending', 'publication_allowed': False}
    _save(context.subdir('reports') / 'native_diagnostic.json', out)
    context.write_metadata({'status': out['status'], 'config': spec, 'publication_allowed': False})
    print(json.dumps({'run_id': context.run_id, 'status': out['status'], 'actual_accounts': out['actual_account_count'],
        'failed_cash_accounts': out['failed_cash_accounts'], 'report': str(context.subdir('reports') / 'native_diagnostic.json')}), flush=True)
    return out


if __name__ == '__main__':
    main()
