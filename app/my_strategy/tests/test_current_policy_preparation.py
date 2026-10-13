"""Full causal preparation and shared E0 base must remain economically equal.

Tests use only current code and current economic fixtures. Frozen original-code
benchmark proofs are audit artifacts rather than dependencies of this suite.
"""
from __future__ import annotations

import copy
import itertools
import json

import pandas as pd
import pytest

from my_strategy.services import czsc_trend_consensus as consensus
from my_strategy.services import czsc_trend_pipeline as pipeline
from my_strategy.services.czsc_trend_account import prepare_account_context, replay_trend_account
from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.tests.test_current_trend_engine import fixture, SYMBOL


def _variants(config):
    grid = config['consensus']['policy_grid']
    return [dict(cross=cross, buy_base=buy, sell=sell, margin=margin)
            for cross, buy, sell, margin in itertools.product(
                config['trend']['candidate_cross_grid'], grid['buy_base_threshold'],
                grid['sell_threshold'], grid['margin'])]


def _candidate_config(config, variant):
    cfg = copy.deepcopy(config)
    cfg['trend']['candidate_cross_threshold'] = variant['cross']
    cfg['consensus'].update(buy_base_threshold=variant['buy_base'],
                            sell_threshold=variant['sell'], margin=variant['margin'])
    return cfg


def _full(frame, dates, pack, trends, cfg):
    return consensus.prepare_trend_decisions(
        frame, dates, evidence_pack=pack, trend_probabilities=trends,
        config=cfg, mode='E0', allow_policy_validation=True)


def test_all_frozen_policy_combinations_match_full_preparation_and_real_Broker():
    frame, dates, pack, trends = fixture()
    config = strategy_config()
    account_context = prepare_account_context(frame, dates, config)
    base, cross = None, None
    for variant in _variants(config):
        cfg = _candidate_config(config, variant)
        if base is None or cross != variant['cross']:
            base = consensus.prepare_policy_base(frame, dates, pack, trends, cfg)
            cross = variant['cross']
        full = _full(frame, dates, pack, trends, cfg)
        shared = consensus.apply_policy_thresholds(base, cfg)
        assert full == shared
        arguments = dict(config=cfg, start=dates[60], end=dates[-1],
                         run_id='current-policy-equivalence', mode='E0', account_context=account_context)
        full_account = replay_trend_account(frame, dates, full, **arguments)
        shared_account = replay_trend_account(frame, dates, shared, **arguments)
        assert full_account == shared_account
        assert [fill['action'] for fill in full_account['ledger']] == ['BUY', 'SELL']
        assert full_account['ledger'][0]['date'] > full_account['ledger'][0]['signal_date']
        assert all(fill['fee'] > 0 for fill in full_account['ledger'])


@pytest.mark.parametrize('failure', ['low-liquidity', 'risk', 'partial-sell', 'missing-next-open', 'future-fact'])
def test_shared_base_keeps_actual_refusals_and_partial_inventory(failure):
    frame, dates, pack, trends = fixture()
    if failure == 'low-liquidity':
        frame['amount'] = 1000.
    elif failure == 'risk':
        frame['raw_risk_plan'] = [{'status': 'rejected', 'entry_allowed': False,
                                  'reason_codes': ['explicit_bad_stop']}] * len(frame)
    elif failure == 'partial-sell':
        frame.loc[[65, 66], 'volume'] = 15000.
        frame.loc[[65, 66], 'amount'] = 150000.
    elif failure == 'missing-next-open':
        frame.loc[61, ['open', 'high', 'low', 'close', 'has_trade_price']] = [None, None, None, None, 0]
    else:
        pack['events'][0]['first_confirmed_at'] = dates[-1] + 'T15:00:00+08:00'
    cfg = strategy_config()
    base = consensus.prepare_policy_base(frame, dates, pack, trends, cfg)
    shared = consensus.apply_policy_thresholds(base, cfg)
    full = _full(frame, dates, pack, trends, cfg)
    assert full == shared
    args = dict(config=cfg, start=dates[60], end=dates[-1], run_id='current-policy-refusal', mode='E0')
    first = replay_trend_account(frame, dates, full, **args)
    second = replay_trend_account(frame, dates, shared, **args)
    assert first == second
    if failure in {'low-liquidity', 'risk', 'missing-next-open', 'future-fact'}:
        assert not any(fill['action'] == 'BUY' for fill in first['ledger'])
    if failure == 'partial-sell':
        sells = [fill for fill in first['ledger'] if fill['action'] in {'SELL', 'REDUCE'}]
        assert len(sells) == 3 and first['actual_position'] is None
        assert all(fill['signal_date'] == dates[65] for fill in sells)
        assert len({fill['intent_id'] for fill in sells}) == 1


@pytest.mark.parametrize('path,value', [
    (('trend', 'candidate_cross_threshold'), .91),
    (('consensus', 'source_weights'), {'ma': .4, 'structure': .2, 'wyckoff': .2}),
    (('consensus', 'min_independent'), 3),
    (('consensus', 'signal_valid_bars'), 30),
    (('consensus', 'high_return_threshold'), .99),
    (('consensus', 'buy_total_threshold'), .99),
    (('signal_points', 'risk_fraction'), .017),
    (('liquidity_universe', 'min_median_amount'), 40000000.),
    (('source_snapshot_hash',), 'another-frozen-source'),
    (('training_end',), '2026-01-01'),
])
def test_base_rejects_every_configuration_change_outside_three_arbitration_thresholds(path, value):
    frame, dates, pack, trends = fixture()
    cfg = strategy_config()
    base = consensus.prepare_policy_base(frame, dates, pack, trends, cfg)
    changed = copy.deepcopy(cfg)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        consensus.apply_policy_thresholds(base, changed)


def test_worker_never_reuses_causal_base_across_fold_source_or_symbol(monkeypatch):
    frame, dates, pack, trends = fixture()
    cfg = strategy_config()
    cfg['source_snapshot_hash'] = frame.attrs['source_context_hash']
    variants = _variants(cfg)[::27]
    created = []
    original = consensus.prepare_policy_base
    def count_base(*args, **kwargs):
        created.append((args[0].iloc[0]['symbol'], args[4]['source_snapshot_hash']))
        return original(*args, **kwargs)
    monkeypatch.setattr(consensus, 'prepare_policy_base', count_base)
    monkeypatch.setattr(pipeline, 'read_snapshot', lambda record: (frame, pack))
    monkeypatch.setattr(pipeline, 'policy_trend', lambda *args, **kwargs: trends)
    original_frame, original_pack = frame, pack
    cases = [(60, 69, 'source-A', SYMBOL), (70, 84, 'source-A', SYMBOL),
             (60, 69, 'source-B', SYMBOL), (60, 69, 'source-A', 'SECOND.SH')]
    for start, end, source, symbol in cases:
        frame, pack = original_frame.copy(), copy.deepcopy(original_pack)
        frame['symbol'] = symbol
        for event in pack['events']:
            event['symbol'] = symbol
        for row in pack['evaluations']:
            for source_name in ['structure', 'wyckoff']:
                for event in row[source_name].get('support_facts', []):
                    event['symbol'] = symbol
        actual = copy.deepcopy(cfg)
        actual['source_snapshot_hash'] = source
        bundle = {'fold': {'name': 'fold-' + str(start), 'policy_start': dates[start], 'policy_end': dates[end]}}
        output = pipeline._policy_worker(({'symbol': symbol}, dates, actual, '.', bundle, variants))
        assert len(output) == 3
        assert all(item['fold'] == bundle['fold']['name'] for item in output)
    assert len(created) == 12
    assert created == [(symbol, source) for _, _, source, symbol in cases for _ in range(3)]


def test_pure_timestamp_cache_keeps_timezone_and_full_cutoff_boundary():
    timestamp = '2026-01-02T15:00:01+08:00'
    assert not consensus.observable(timestamp, '2026-01-02T15:00:00+08:00')
    assert consensus.observable(timestamp, '2026-01-02T15:00:01+08:00')
    assert consensus.observable('2026-01-02T07:00:00Z', '2026-01-02T15:00:00+08:00')
    assert consensus.observable('2026-01-02T15:00:00', '2026-01-02T15:00:00+08:00')
    assert not consensus.observable(timestamp, '2026-01-02')
    for invalid in [None, pd.NaT, 'invalid', '2026-99-99']:
        assert not consensus.observable(invalid, '2026-01-02T15:00:00+08:00')
    assert consensus.day('2026-01-02T23:00:00-08:00') == '2026-01-02'
    assert consensus._iso_observable.cache_info().maxsize == 131072
    assert consensus._iso_timestamp.cache_info().maxsize == 65536
    assert consensus._iso_day.cache_info().maxsize == 65536


def test_policy_progress_reports_only_persisted_whole_stock_chunks(tmp_path, monkeypatch, capsys):
    cfg = strategy_config()
    records = [{'symbol': 'FIRST.SZ'}, {'symbol': 'SECOND.SH'}]
    bundle = {'name': 'independent-fold', 'available_at': '2026-01-01', 'fold': {'name': 'independent-fold'}}
    def complete_chunks(worker, payloads, *args):
        for record, calendar, config, root, bundle, variants in payloads:
            yield [{'symbol': record['symbol'], 'variant': variant, 'ledger': [],
                    'metrics': {'total_return': 0., 'max_drawdown': 0.}} for variant in variants]
    monkeypatch.setattr(pipeline, '_bounded_results', complete_chunks)
    class Context:
        def subdir(self, *args):
            path = tmp_path.joinpath(*args)
            path.mkdir(parents=True, exist_ok=True)
            return path
        def run_dir(self):
            return tmp_path
    report = pipeline.select_source_policy(records, ['2026-01-01'], cfg, '.', [bundle], Context(), 1)
    events = [json.loads(line.removeprefix('POLICY_STOCK_DONE ')) for line in capsys.readouterr().out.splitlines()
              if line.startswith('POLICY_STOCK_DONE ')]
    assert [event['current'] for event in events] == [1, 2]
    assert [event['symbol'] for event in events] == ['FIRST.SZ', 'SECOND.SH']
    assert all(event['total'] == 2 and event['variants'] == 81 and event['failed'] == 0 and event['device'] == 'cpu'
               for event in events)
    assert len((tmp_path / 'validation/source_policy_accounts.jsonl').read_text().splitlines()) == 162
    assert report['accounts_asset']['records'] == 162
    assert all(item['scored_population'] == 2 for item in report['variants'])


def test_head_progress_reports_actual_device_even_when_request_differs(capsys):
    pipeline._emit_head_resource({'fold': {'name': 'fold'}, 'head': 'entry', 'architecture': 'linear',
                                 'requested_device': 'mps', 'actual_device': 'cpu'})
    event = json.loads(capsys.readouterr().out.removeprefix('HEAD '))
    assert event['device'] == 'cpu'
    pipeline._emit_head_resource({'fold': {'name': 'fold'}, 'head': 'entry', 'architecture': 'mlp',
                                 'requested_device': 'cpu', 'actual_device': 'mps',
                                 'training_resources': {'actual_device': 'mps', 'input_staging': 'device_resident'}})
    event = json.loads(capsys.readouterr().out.removeprefix('HEAD '))
    assert event['device'] == 'mps' and event['training_resources']['input_staging'] == 'device_resident'
