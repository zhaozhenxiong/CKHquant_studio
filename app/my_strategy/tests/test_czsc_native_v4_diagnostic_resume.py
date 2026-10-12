"""Capacity changes reuse only complete immutable accounts without selecting winners."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from my_strategy.scripts import resume_native_v4_diagnostic as resume
from my_strategy.scripts.validate_native_v4 import _cash_failure, _persist_account, _save, aggregate_accounts


def spec():
    return {'arms': [{'id': 'linear_entry_0p1'}], 'windows': {'window': ['2026-01-05', '2026-01-06']},
            'slippage_multipliers': [1.], 'initial_cash_per_account': 100000.}


def stock_fixture(parent, symbol='002436.SZ', status='completed'):
    directory = parent / 'accounts' / symbol.replace('.', '_')
    account = _cash_failure(symbol, '2026-01-05', '2026-01-06', ['2026-01-05', '2026-01-06'], 100000., 'explicit failure')
    summary = {'symbol': symbol, 'window': 'window', 'variant': 'linear_entry_0p1', 'evidence_bars': 3,
               'slippage_multiplier': 1., 'initial_cash': 100000., 'status': status,
               'metrics': account['metrics'], 'turnover': 0., 'start': '2026-01-05', 'end': '2026-01-06'}
    summary = _persist_account(directory / 'linear_entry_0p1/window/slippage_1', account, summary)
    stock = {'symbol': symbol, 'accounts': [summary], 'coverage': {}, 'label_predictions': {},
             'failures': [] if status == 'completed' else [{'stage': 'worker', 'error': 'explicit failure'}]}
    _save(directory / 'stock_summary.json', stock)
    return stock


def test_complete_inventory_preserves_original_files_and_identity(tmp_path):
    original = stock_fixture(tmp_path)
    stock, hashes = resume.verify_stock(tmp_path, '002436.SZ', spec())
    assert stock == original
    assert len(hashes) == 10  # seven economic files, holding, summary, stock
    assert resume.verify_stock(tmp_path, '002436.SZ', spec(), hashes) == (stock, hashes)
    assert resume.verify_stock(tmp_path, '000333.SZ', spec()) == (None, {})


def test_completed_summary_with_missing_artifact_is_rejected(tmp_path):
    stock = stock_fixture(tmp_path)
    Path(stock['accounts'][0]['artifacts']['ledger']).unlink()
    with pytest.raises(ValueError, match='artifact missing'):
        resume.verify_stock(tmp_path, '002436.SZ', spec())


def test_artifact_tampering_after_sha_inventory_is_rejected(tmp_path):
    stock = stock_fixture(tmp_path)
    _, hashes = resume.verify_stock(tmp_path, '002436.SZ', spec())
    path = Path(stock['accounts'][0]['artifacts']['daily'])
    rows = json.loads(path.read_text()); rows[0]['equity'] = 99999.
    _save(path, rows)
    with pytest.raises(ValueError, match='SHA changed'):
        resume.verify_stock(tmp_path, '002436.SZ', spec(), hashes)


def test_incomplete_or_duplicate_account_sets_cannot_be_reused(tmp_path):
    stock = stock_fixture(tmp_path)
    path = tmp_path / 'accounts/002436_SZ/stock_summary.json'
    stock['accounts'].append(stock['accounts'][0]); _save(path, stock)
    with pytest.raises(ValueError, match='denominator'):
        resume.verify_stock(tmp_path, '002436.SZ', spec())


def test_declared_failed_account_stays_in_equal_cash_denominator(tmp_path):
    successful = stock_fixture(tmp_path, '002436.SZ')
    failed = stock_fixture(tmp_path, '000333.SZ', 'cash_failure')
    checked, _ = resume.verify_stock(tmp_path, '000333.SZ', spec())
    assert checked['failures']
    aggregate = aggregate_accounts([successful, failed], ['002436.SZ', '000333.SZ'], 100000.)[0]
    assert aggregate['requested_accounts'] == aggregate['retained_accounts'] == 2
    assert aggregate['failed_cash_accounts'] == 1
    assert aggregate['daily'][-1]['equity'] == 200000.
    assert aggregate['total_return'] == 0


def test_failure_claim_cannot_hide_lost_cash(tmp_path):
    stock = stock_fixture(tmp_path, status='cash_failure')
    path = Path(stock['accounts'][0]['artifacts']['daily']); rows = json.loads(path.read_text())
    rows[-1]['equity'] -= 1; _save(path, rows)
    with pytest.raises(ValueError, match='retain frozen cash'):
        resume.verify_stock(tmp_path, '002436.SZ', spec())


def test_economic_identity_normalization_preserves_fee_and_decision_differences():
    left = {'run_id': 'parent', 'fees': .5, 'nested': [{'account_id': 'original', 'action': 'BUY'}]}
    right = {'run_id': 'resumed', 'fees': .5, 'nested': [{'account_id': 'new', 'action': 'BUY'}]}
    assert resume.without_account_id(left) == resume.without_account_id(right)
    right['fees'] = .6
    assert resume.without_account_id(left) != resume.without_account_id(right)
    one = {'date': '2026-01-05', 'action': 'BUY', 'reference_price_basis': 'unadjusted',
           'consensus': {'agents': {'wyckoff': {'target': 'rule strength', 'buy_contribution': .3}}}}
    two = {'date': '2026-01-05', 'action': 'BUY', 'reference_price_basis': 'unavailable',
           'consensus': {'agents': {'wyckoff': {'target': None, 'buy_contribution': .3}}}}
    assert resume.event_semantics(one) == resume.event_semantics(two)
    two['consensus']['agents']['wyckoff']['buy_contribution'] = .2
    assert resume.event_semantics(one) != resume.event_semantics(two)


def test_unexpected_worker_failure_creates_all_predefined_cash_accounts(tmp_path):
    config = spec()
    config.update(raw_end='2026-01-06', slippage_multipliers=[1., 2.])
    config['arms'][0].update(architecture='linear', entry_threshold=.1, comparisons=['architecture', 'threshold_grid'])
    context = SimpleNamespace(run_dir=lambda: tmp_path)
    stock = resume.retained_worker_failure('002436.SZ', config, context, ['2026-01-05', '2026-01-06'], RuntimeError('worker failed'))
    assert len(stock['accounts']) == len(resume.expected_accounts(config)) == 2
    assert all(a['status'] == 'cash_failure' and a['metrics']['final_equity'] == 100000. for a in stock['accounts'])
    assert stock['failures'][0]['stage'] == 'worker'
    assert resume.verify_stock(tmp_path, '002436.SZ', config)[0] == stock
