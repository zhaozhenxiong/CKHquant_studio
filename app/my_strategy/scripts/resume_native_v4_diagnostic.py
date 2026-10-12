"""Resume only complete, hash-inventoried stocks in a new diagnostic context.

This orchestrator does not alter diagnostic evaluation, models or policies.
Original account identities and files stay in the parent context; reused file
references are explicit and hash verified again before final aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.scripts import diagnose_native_v4 as diag
from my_strategy.scripts.validate_native_v4 import _save, _worker_init


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_accounts(spec):
    return {(arm['id'], window, float(cost)) for arm in spec['arms']
            for window in spec['windows'] for cost in spec['slippage_multipliers']}


def verify_stock(parent, symbol, spec, expected_hashes=None):
    directory = Path(parent).resolve() / 'accounts' / symbol.replace('.', '_')
    path = directory / 'stock_summary.json'
    if not path.is_file():
        return None, {}
    stock = json.loads(path.read_text())
    if stock['symbol'] != symbol:
        raise ValueError('stock identity differs from requested cohort')
    accounts = stock['accounts']
    keys = [(a['variant'], a['window'], float(a['slippage_multiplier'])) for a in accounts]
    if len(keys) != len(set(keys)) or set(keys) != expected_accounts(spec):
        raise ValueError('stock summary lacks complete unique account denominator')
    hashes = {str(path): file_sha(path)}
    for account in accounts:
        if account['symbol'] != symbol or account['evidence_bars'] != 3 or account['initial_cash'] != spec['initial_cash_per_account']:
            raise ValueError('reused account scope/cash differs')
        if account['status'] not in {'completed', 'cash_failure'}:
            raise ValueError('unfinished account cannot be reused')
        if set(account['artifacts']) != {'ledger', 'daily', 'rejections', 'holding_snapshots',
                'exit_policy_diagnostics', 'entry_diagnostics', 'episodes'}:
            raise ValueError('reused account lacks required artifacts')
        for filename in account['artifacts'].values():
            artifact = Path(filename).resolve()
            if not artifact.is_relative_to(directory) or not artifact.is_file():
                raise ValueError('reused artifact missing or outside stock directory')
            hashes[str(artifact)] = file_sha(artifact)
        for filename in ['summary.json', 'open_position.json']:
            artifact = Path(account['artifacts']['daily']).parent / filename
            if not artifact.is_file():
                raise ValueError('reused account summary/holding artifact missing')
            hashes[str(artifact)] = file_sha(artifact)
        if account['status'] == 'cash_failure':
            daily = json.loads(Path(account['artifacts']['daily']).read_text())
            if any(row['shares'] != 0 or row['equity'] != spec['initial_cash_per_account'] for row in daily):
                raise ValueError('declared failed account did not retain frozen cash')
    # Include score identities, consensus, causal decisions and refusal reasons.
    for artifact in directory.rglob('*.json'):
        hashes[str(artifact.resolve())] = file_sha(artifact)
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError('reused artifact SHA changed after inventory')
    return stock, hashes


def without_account_id(value):
    if isinstance(value, dict):
        return {key: without_account_id(item) for key, item in value.items() if key not in {'run_id', 'account_id'}}
    if isinstance(value, list):
        return [without_account_id(item) for item in value]
    return value


def event_semantics(event):
    value = json.loads(json.dumps(event))
    value.pop('reference_price_basis', None)  # chart-only price-basis annotation
    value.get('consensus', {}).get('agents', {}).get('wyckoff', {}).pop('target', None)
    return value


def retained_worker_failure(symbol, spec, context, calendar, error):
    stock = {'symbol': symbol, 'accounts': [], 'coverage': {}, 'label_predictions': {},
             'failures': [{'stage': 'worker', 'error': repr(error)}]}
    for arm in spec['arms']:
        for name, (start, finish) in spec['windows'].items():
            end = min(finish, spec['raw_end'])
            for cost in spec['slippage_multipliers']:
                account = diag._cash_failure(symbol, start, end, calendar, spec['initial_cash_per_account'], repr(error))
                summary = {'symbol': symbol, 'window': name, 'variant': arm['id'], 'architecture': arm['architecture'],
                    'entry_threshold': arm['entry_threshold'], 'comparisons': arm['comparisons'], 'evidence_bars': 3,
                    'start': start, 'end': end, 'initial_cash': spec['initial_cash_per_account'],
                    'slippage_multiplier': cost, 'status': 'cash_failure', 'metrics': account['metrics'],
                    'turnover': 0., 'failure': account['failure'], 'publication_allowed': False}
                stock['accounts'].append(diag._persist_account(context.run_dir() / 'accounts' /
                    symbol.replace('.', '_') / arm['id'] / name / f'slippage_{cost:g}', account, summary))
    _save(context.run_dir() / 'accounts' / symbol.replace('.', '_') / 'stock_summary.json', stock)
    return stock


def compare_baseline(stock, baseline):
    """Strict actual economic and vote checks; only documented UI fields differ."""
    symbol = stock['symbol']; checks = []
    for account in stock['accounts']:
        if account['variant'] != 'linear_entry_0p1':
            continue
        base = Path(baseline) / 'accounts' / symbol.replace('.', '_') / account['window'] / 'v4_age3'
        real = Path(account['artifacts']['daily']).parent
        source = real.parent.parent / 'full_prefix_consensus.json'
        one = json.loads((base / 'consensus.json').read_text()); two = json.loads(source.read_text())
        events = [event for event in two['events'] if account['start'] <= event['date'] <= account['end']]
        if one['config'] != two['config'] or [event_semantics(e) for e in one['events']] != [event_semantics(e) for e in events]:
            raise ValueError(f'baseline consensus differs: {symbol}/{account["window"]}')
        row_map = {row['date']: row for row in one['rows']}
        for row in two['rows']:
            if account['start'] <= row['date'] <= account['end']:
                if row != {key: row_map[row['date']][key] for key in row}:
                    raise ValueError(f'baseline daily arbitration differs: {symbol}/{row["date"]}')
        original = base / f'slippage_{account["slippage_multiplier"]:g}'
        for key, filename in account['artifacts'].items():
            left = json.loads((original / (key + '.json')).read_text())
            right = json.loads(Path(filename).read_text())
            if without_account_id(left) != without_account_id(right):
                raise ValueError(f'baseline {key} differs: {symbol}/{account["window"]}/{account["slippage_multiplier"]}')
        left = json.loads((original / 'summary.json').read_text())
        if left['metrics'] != account['metrics']:
            raise ValueError('baseline economic summary metrics differ')
        checks.append({'symbol': symbol, 'window': account['window'], 'slippage_multiplier': account['slippage_multiplier'],
            'consensus_and_economic_artifacts_equal': True, 'display_annotations_only': one['events'] != events,
            'display_fields_excluded': ['reference_price_basis', 'consensus.agents.wyckoff.target'],
            'account_identity_excluded': ['run_id', 'account_id']})
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run-id', required=True)
    parser.add_argument('--baseline-run-id', required=True)
    parser.add_argument('--cpu-workers', type=int, default=6)
    parser.add_argument('--inventory-only', action='store_true', help='Read-only capacity plan; allows a running parent without resuming it')
    args = parser.parse_args(argv)
    if not 1 <= args.cpu_workers <= 6:
        raise ValueError('resume capacity must be 1..6')
    parent = (ARTIFACT_RUNS_ROOT / args.parent_run_id).resolve()
    baseline = (ARTIFACT_RUNS_ROOT / args.baseline_run_id).resolve()
    if not parent.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()) or not baseline.is_relative_to(ARTIFACT_RUNS_ROOT.resolve()):
        raise ValueError('resume context outside artifact roots')
    metadata = json.loads((parent / 'metadata.json').read_text())
    if not args.inventory_only and metadata.get('status') != 'stopped_capacity_resume':
        raise ValueError('parent must be explicitly stopped before capacity resume')
    original = json.loads((parent / 'config/diagnostic.json').read_text())
    root, report = diag.read_run(original['model_run_id'])
    calendar, calendar_id = diag._calendar(original['calendar_run_id'])
    if original['script_sha256'] != file_sha(diag.__file__):
        raise ValueError('diagnostic evaluator changed since parent execution')
    if original['source_snapshot_hash'] != report['source_snapshot']['snapshot_hash'] or original['calendar_hash'] != stable_hash(calendar):
        raise ValueError('resume data/calendar identity changed')
    if original['default_catalog_sha256'] != stable_hash(report['model_bundles']):
        raise ValueError('resume model catalog changed')
    for architecture in diag.ARCHITECTURES:
        if original['architecture_head_identity'][architecture] != diag.architecture_catalog(root, report, architecture)[1]:
            raise ValueError('resume actual architecture identity changed')
    baseline_spec = json.loads((baseline / 'config/validation.json').read_text())
    if baseline_spec['model_run_id'] != original['model_run_id'] or baseline_spec['calendar_hash'] != original['calendar_hash'] or baseline_spec['cohort_hash'] != original['cohort_hash']:
        raise ValueError('baseline model/source differs')
    symbols = original['symbols']; reused, inventory, remaining = [], {}, []
    for symbol in symbols:
        stock, hashes = verify_stock(parent, symbol, original)
        if stock is None:
            remaining.append(symbol)
        else:
            reused.append(stock); inventory[symbol] = hashes
    if args.inventory_only:
        plan = {'parent_run_id': args.parent_run_id, 'model_run_id': original['model_run_id'],
            'evaluator_sha256': original['script_sha256'], 'reused_complete_symbols': len(reused),
            'remaining_symbols': len(remaining), 'expected_actual_accounts': len(symbols) * len(expected_accounts(original)),
            'inventory_sha256': stable_hash(inventory), 'read_only': True}
        print(json.dumps(plan), flush=True)
        return plan
    spec = {**original, 'parent_run_id': args.parent_run_id, 'baseline_run_id': args.baseline_run_id,
        'resume_cpu_workers': args.cpu_workers, 'reused_symbols': [s['symbol'] for s in reused], 'remaining_symbols': remaining,
        'parent_config_sha256': file_sha(parent / 'config/diagnostic.json'),
        'input_model_report_sha256': file_sha(root / 'reports/native_v4.json'),
        'resume_orchestrator_sha256': file_sha(__file__), 'reused_artifact_inventory_sha256': stable_hash(inventory),
        'reused_account_identity': 'original parent run IDs retained; absolute immutable artifact references',
        'expected_actual_accounts': len(symbols) * len(expected_accounts(original)),
        'no_automatic_parameter_or_model_selection': True, 'publication_allowed': False}
    context = create_run_context(task='czsc-native-v4-diagnostic-capacity-resume', as_of_date=original['raw_end'],
        config=spec, data_version=original['source_snapshot_hash'], stocks=symbols,
        start_date=min(start for start, _ in original['windows'].values()), end_date=original['raw_end'])
    _save(context.subdir('config') / 'diagnostic.json', spec)
    _save(context.subdir('config') / 'reused_artifact_inventory.json', inventory)
    print('RUN_ID ' + context.run_id, flush=True)
    results = list(reused)
    payload = {'model_run_id': original['model_run_id'], 'calendar': calendar, 'windows': original['windows'],
        'arms': original['arms'], 'costs': original['slippage_multipliers'], 'end': original['raw_end'],
        'output': str(context.run_dir()), 'run_id': context.run_id}
    with ProcessPoolExecutor(max_workers=args.cpu_workers, mp_context=multiprocessing.get_context('spawn'), initializer=_worker_init) as pool:
        pending = {pool.submit(diag.evaluate_stock, {**payload, 'symbol': symbol}): symbol for symbol in remaining}
        for future in as_completed(pending):
            try:
                stock = future.result()
            except Exception as exc:
                stock = retained_worker_failure(pending[future], original, context, calendar, exc)
            results.append(stock)
            print(json.dumps({'completed': len(results), 'requested': len(symbols), 'symbol': stock['symbol'],
                'actual_accounts': len(stock['accounts']), 'failures': len(stock['failures'])}), flush=True)
    for symbol, hashes in inventory.items():
        verify_stock(parent, symbol, original, expected_hashes=hashes)
    results.sort(key=lambda stock: symbols.index(stock['symbol']))
    count = sum(len(stock['accounts']) for stock in results)
    if {stock['symbol'] for stock in results} != set(symbols) or count != spec['expected_actual_accounts']:
        raise ValueError('resume fixed cohort/account denominator changed')
    aggregate = diag.aggregate_accounts(results, symbols, original['initial_cash_per_account'])
    arms = {arm['id']: arm for arm in original['arms']}
    for item in aggregate:
        item.update(architecture=arms[item['variant']]['architecture'], entry_threshold=arms[item['variant']]['entry_threshold'], comparisons=arms[item['variant']]['comparisons'])
    equivalence, equivalence_errors = [], []
    for stock in results:
        try:
            equivalence.extend(compare_baseline(stock, baseline))
        except Exception as exc:
            equivalence_errors.append({'symbol': stock['symbol'], 'error': repr(exc)})
    classification = {architecture: diag.classification_metrics(
        [row for stock in results for row in stock['label_predictions'].get(architecture, [])], original['windows'], original['frozen_feature_end'])
        for architecture in diag.ARCHITECTURES}
    out = {'run_id': context.run_id, 'config': spec, 'aggregate': aggregate,
        'account_summaries': [account for stock in results for account in stock['accounts']],
        'stock_records': [{key: value for key, value in stock.items() if key not in {'accounts', 'label_predictions'}} for stock in results],
        'classification': classification, 'actual_account_count': count,
        'failed_cash_accounts': sum(a['status'] != 'completed' for stock in results for a in stock['accounts']),
        'status': 'completed' if all(not stock['failures'] for stock in results) and not equivalence_errors else 'completed_with_failures_retained',
        'baseline_equivalence': equivalence, 'baseline_equivalence_errors': equivalence_errors,
        'baseline_equivalence_passed': not equivalence_errors and len(equivalence) == len(symbols) * len(original['windows']) * len(original['slippage_multipliers']),
        'independent_window_status': 'pending', 'publication_allowed': False}
    _save(context.subdir('reports') / 'native_diagnostic.json', out)
    context.write_metadata({'status': out['status'], 'config': spec, 'publication_allowed': False})
    print(json.dumps({'run_id': context.run_id, 'actual_accounts': count, 'failed_cash_accounts': out['failed_cash_accounts'], 'status': out['status']}), flush=True)
    return out


if __name__ == '__main__':
    main()
