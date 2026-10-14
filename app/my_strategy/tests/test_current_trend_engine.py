"""Canonical current engine acceptance: causal targets, actual Broker populations and frozen contracts.

Synthetic fitting thresholds are confined to fixtures. Production defaults and
all trading, maturity, class, risk and dependency requirements stay unchanged.
No retired implementation or retired test helper is imported.
"""
from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.execution.cost_model import CostModel
from my_strategy.services import czsc_trend_pipeline as pipeline
from my_strategy.services import czsc_trend_source as source
from my_strategy.services import czsc_trend_runtime as runtime
from my_strategy.services.czsc_trend_contracts import (
    STRATEGY_ID, TRAINING_POPULATION, SPLIT_POLICY, ECONOMIC_WEIGHTING,
    HELD_SPLIT_POLICY, HELD_ECONOMIC_WEIGHTING, head_dependency_policy,
    strategy_config, HELD_TRAINING_POPULATION, HELD_CONDITION_DOMAIN,
)
from my_strategy.services.czsc_trend_features import (
    ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS, L1_MA_ENTRY_COLUMNS,
    L1_MA_EXIT_COLUMNS, HOLDING_COLUMNS, causal_market_row,
)
from my_strategy.services.czsc_trend_labels import target_contract, episode_outcome, exit_outcome
from my_strategy.services.czsc_trend_interval_model import (
    HeadSession, _group_ids, _validate_contract, _weights, bind_inputs, load_head,
    schema_hash, time_masks, train_head,
)
from my_strategy.services.czsc_trend_model import (
    build_trend_labels, trend_time_masks, CLASSES, trend_contract,
)
from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions
from my_strategy.services.czsc_trend_account import (
    replay_trend_account, clone_candidate_account, _flat_state_hash,
)
from my_strategy.services.czsc_trend_dataset import (
    build_account_dataset, pre_gate_attempt, training_frame,
)
from my_strategy.services.czsc_trend_pipeline import (
    make_reference_population, train_trend_consensus, verify_source_generators,
)
from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS
from my_strategy.services.czsc_native_evidence import STRUCTURE_RULE_VERSION, W_RULE_VERSION

SYMBOL = '002436.SZ'


@pytest.fixture(autouse=True)
def small_cpu_fits():
    """Keep meaningful real fits small without changing their label standards."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        yield


def test_current_identity_retains_production_defaults_and_causal_feature_shapes():
    config = strategy_config()
    assert STRATEGY_ID == 'trend_consensus'
    assert config['version'] == config['consensus']['version'] == STRATEGY_ID
    assert config['high_return']['min_net_return'] == .08
    assert config['high_return']['min_r_multiple'] == 2.
    assert config['min_train_rows'] == 100
    assert config['min_calibration_rows'] == 30
    assert len(L1_MA_ENTRY_COLUMNS) == 20
    assert len(L1_MA_EXIT_COLUMNS) == 30
    assert L1_MA_EXIT_COLUMNS == (*L1_MA_ENTRY_COLUMNS, *HOLDING_COLUMNS)
    assert CLASSES == ('UP', 'DOWN', 'NEUTRAL')
    assert not any(column.startswith('v4_') for column in ENTRY_FEATURE_COLUMNS)
    assert config['reassessment']['quality_rejection_consumes'] is False
    assert config['reassessment']['broker_rejection_retry'] is False
    assert {'execution/broker.py','execution/cost_model.py','execution/ledger.py',
        'execution/order.py','services/czsc_backtest.py'} <= set(pipeline.GENERATING_FILES)


def ledger_fixture(net_return=.10, *, fee=5.):
    paid = 10000. + fee
    return [
        {'action': 'BUY', 'date': '2025-01-03', 'shares': 1000, 'price': 10.,
         'cash_flow': -paid, 'fee': fee},
        {'action': 'SELL', 'date': '2025-01-06', 'shares': 1000,
         'price': paid * (1 + net_return) / 1000,
         'cash_flow': paid * (1 + net_return), 'fee': fee},
    ]


@pytest.mark.parametrize('net_return,stop,expected', [
    (.10, 9.7, 1.), (.079, 9.7, 0.), (.10, 9., 0.), (-.01, 9.7, 0.),
])
def test_high_return_requires_both_fee_net_eight_percent_and_two_R(net_return, stop, expected):
    outcome = episode_outcome(ledger_fixture(net_return), stop_price=stop,
                              cost_model=CostModel(), config=strategy_config())
    assert outcome['label_available'] is True
    assert outcome['label'] == expected
    assert outcome['net_return'] == pytest.approx(net_return)
    assert outcome['fees'] == 10.
    assert outcome['label_end'] == '2025-01-06'
    assert outcome['label_available_at'] == '2025-01-06T09:30:00+08:00'


@pytest.mark.parametrize('kind', ['rejected', 'unclosed', 'invalid-risk'])
def test_unfilled_unmatured_and_invalid_risk_are_unknown_never_negative(kind):
    ledger = ledger_fixture(-.10)
    stop = 9.7
    if kind == 'rejected':
        ledger = []
    elif kind == 'unclosed':
        ledger = ledger[:1]
    else:
        stop = 10.1
    outcome = episode_outcome(ledger, stop_price=stop, cost_model=CostModel())
    assert outcome['label_available'] is False
    assert np.isnan(outcome['label'])
    assert outcome['label_reason'] == {
        'rejected': 'entry_unfilled', 'unclosed': 'episode_unclosed',
        'invalid-risk': 'risk_identity_unavailable',
    }[kind]


def test_exit_label_matures_only_after_both_real_liquidations():
    inputs = {'forced': {'date': '2025-01-06', 'proceeds': 10100., 'rejections': []},
              'continued_proceeds': 10000., 'continued_end': '2025-01-08',
              'shares': 1000, 'close': 10.}
    mature = exit_outcome(**inputs)
    assert mature['label'] == 1. and mature['label_available']
    assert mature['label_end'] == '2025-01-08'
    for field, value in [('continued_end', None), ('forced', {'date': None, 'rejections': []})]:
        pending = exit_outcome(**{**inputs, field: value})
        assert not pending['label_available'] and np.isnan(pending['label'])


@pytest.mark.parametrize('segment', ['train', 'calibration'])
def test_zero_positive_class_is_rejected_even_with_enough_fixture_rows(tmp_path, segment):
    frame, _, fold, config, contract = entry_fixture()
    train, calibration, _ = time_masks(frame, fold, head='entry')
    frame.loc[train if segment == 'train' else calibration, 'label'] = 0
    with pytest.raises(ValueError, match='both classes'):
        train_head(frame, ENTRY_FEATURE_COLUMNS, tmp_path / segment, fold, config,
                   head='entry', device='cpu', contract=contract)


@pytest.mark.parametrize('old_identity', [
    'czsc_trend_consensus', 'czsc_trend_consensus_v2', 'czsc_high_return',
    'czsc_agent_consensus_v4',
])
def test_explicit_archived_run_cannot_be_loaded_as_current(tmp_path, monkeypatch, old_identity):
    directory = tmp_path / 'explicit-archived' / 'reports'
    directory.mkdir(parents=True)
    (directory / 'trend_consensus.json').write_text(json.dumps({
        'strategy_version': old_identity, 'status': 'completed', 'failures': [],
        'continuation_stage': 'E1', 'automatic_selection_eligible': True,
    }))
    monkeypatch.setattr(pipeline, 'ARTIFACT_RUNS_ROOT', tmp_path)
    monkeypatch.setattr(pipeline, '_verify_run', lambda *a: pytest.fail('archive must fail before verification'))
    with pytest.raises(ValueError, match='incompatible'):
        pipeline.read_run('explicit-archived')


def test_same_source_and_head_require_the_current_pipeline_contract(tmp_path, monkeypatch):
    from my_strategy.services.czsc_trend_model import trend_config
    config = strategy_config()
    config.update(source_snapshot_hash='same-source', pipeline_contract_hash='new-protocol')
    manifest = {'target_contract': trend_contract(config, source_snapshot_hash='same-source',
                                                pipeline_contract_hash='old-protocol')}
    assert manifest['target_contract']['parameters'] == trend_config(config)
    bundle = {'name': 'prior', 'heads': {'trend_direction': {'path': 'prior'}}}
    monkeypatch.setattr(pipeline, '_bundle', lambda *args: bundle)
    monkeypatch.setattr(pipeline, 'bind_trend_inputs', lambda frame, _, config=None: frame.copy())
    monkeypatch.setattr(pipeline, 'load_trend_head', lambda _: manifest)
    class Session:
        def predict(self, *args, **kwargs):
            pytest.fail('different pipeline must fail before inference')
    frame = pd.DataFrame({'date': ['2025-01-02'], 'symbol': ['TEST.SZ'], 'model_input_eligible': [True]})
    with pytest.raises(ValueError, match='pipeline|protocol'):
        pipeline.forward_trend(frame, tmp_path, [bundle], config, session=Session(), market_dates=frame.date.tolist())


def test_changed_high_return_target_cannot_reuse_same_head_and_pipeline(tmp_path, monkeypatch):
    _, _, _, config, contract = entry_fixture()
    config.update(source_snapshot_hash=contract['source_snapshot_hash'],
                  pipeline_contract_hash=contract['pipeline_contract_hash'],
                  candidate_policy_hash=contract['candidate_policy_hash'],
                  E1_policy_hash=contract['continuation_policy_hash'])
    config['high_return']['min_net_return'] = .12
    manifest = {'family': STRATEGY_ID, 'target_contract': contract}
    monkeypatch.setattr(pipeline, 'load_head', lambda _: manifest)
    class Session:
        def predict_one(self, *args, **kwargs):
            pytest.fail('changed target must fail before inference')
    bundle = {'heads': {'entry': {'path': 'same-head'}}}
    with pytest.raises(ValueError, match='target|high.return|retrain'):
        pipeline._predict_interval({}, tmp_path, bundle, 'entry', Session(), config=config)


def test_real_MA_holding_head_keeps_thirty_inputs_and_own_episode_contract(tmp_path):
    frame, dates, fold, config, _ = entry_fixture()
    for number, column in enumerate(HOLDING_COLUMNS):
        frame[column] = np.sin(np.arange(len(frame)) / 11 + number)
    frame['episode_id'] = [f'actual-held:{i // 2}' for i in range(len(frame))]
    contract = target_contract(head='ma_exit', config=config, market_dates=dates,
        source_snapshot_hash='frozen-source', feature_schema_hash=schema_hash(L1_MA_EXIT_COLUMNS, 'ma_exit'),
        continuation_policy_hash='frozen-E0', initial_cash=config['initial_cash'], stage='E0',
        candidate_policy_hash='frozen-candidate')
    contract = freeze_contract({**contract, 'model_input_schema_hash': schema_hash(L1_MA_EXIT_COLUMNS, 'ma_exit')})
    frame.attrs['label_contract'] = contract
    manifest = train_head(frame, L1_MA_EXIT_COLUMNS, tmp_path / 'ma-held', fold, config,
                          head='ma_exit', device='cpu', contract=contract)
    assert manifest['columns'] == list(L1_MA_EXIT_COLUMNS)
    assert manifest['group_split_policy'] == HELD_SPLIT_POLICY
    assert manifest['episode_weighting'] == HELD_ECONOMIC_WEIGHTING
    future = bind_inputs(frame.iloc[61:64], L1_MA_EXIT_COLUMNS, 'ma_exit')
    probability = HeadSession().predict(future, tmp_path / 'ma-held', expected_head='ma_exit', expected_contract=contract)
    assert probability.shape == (3,) and np.isfinite(probability).all()


def test_source_worker_freezes_raw_only_and_original_basis_metadata(tmp_path, monkeypatch):
    frame, calendar, _, _ = fixture()
    attributes = {'price_basis': 'unadjusted', 'legacy_fuyao_bars': 0,
                  'unverified_trade_price_bars': 0, 'volume_unit': 'shares',
                  'amount_unit': 'CNY', 'data_version': 'real-raw-observation-prefix'}
    frame.attrs = attributes
    frame['future_label'] = 999
    monkeypatch.setattr(source, 'load_bars', lambda *args, **kwargs: frame.copy())
    record, error = source._raw_worker((SYMBOL, calendar[-1], str(tmp_path / 'raw'), None))
    assert error is None and record['raw_attrs'] == attributes
    frozen = pd.read_parquet(record['raw_path'])
    assert 'future_label' not in frozen and 'close_above_ma20' not in frozen
    assert not frozen.attrs
    assert pipeline._file_hash(Path(record['raw_path'])) == record['raw_sha256']
    Path(record['raw_path']).write_bytes(b'tampered raw observations')
    with pytest.raises(ValueError, match='immutable raw'):
        source._native_worker((record, strategy_config(), calendar, str(tmp_path / 'native')))


def test_optional_source_missing_values_do_not_create_a_new_complete_feature_gate(tmp_path):
    from my_strategy.services.czsc_market_profiles import STRUCTURE_COLUMNS
    from my_strategy.services.czsc_price_volume_features import W_INPUT_COLUMNS
    frame, _, fold, config, contract = entry_fixture()
    columns = [column for column in (*STRUCTURE_COLUMNS, *W_INPUT_COLUMNS) if column in frame]
    frame.loc[:, columns] = np.nan
    assert not np.isfinite(frame[list(ENTRY_FEATURE_COLUMNS)].to_numpy()).all(axis=1).any()
    manifest = train_head(frame, ENTRY_FEATURE_COLUMNS, tmp_path / 'partial-source', fold, config,
                          head='entry', device='cpu', contract=contract)
    train, calibration, _ = time_masks(frame, fold, head='entry')
    assert manifest['sample_counts']['train'] == train.sum()
    assert manifest['sample_counts']['calibration'] == calibration.sum()
    assert np.isfinite(manifest['preprocess']['median']).all()


def test_new_buy_cannot_sell_same_day_when_close_triggers_protection():
    frame, calendar, pack, trends = fixture(sell_from=None)
    frame.loc[61, ['open', 'close', 'high', 'low']] = [9.6, 9., 9.7, 8.9]
    frame.loc[62, ['open', 'close', 'high', 'low']] = [9., 9., 9.1, 8.9]
    prepared = prepare(frame, calendar, pack, trends)
    result = replay_trend_account(frame, calendar, prepared, start=calendar[60], end=calendar[62],
                                  mode='E0', run_id='real-T-plus-one')
    assert [fill['action'] for fill in result['ledger']] == ['BUY', 'SELL']
    assert [fill['date'] for fill in result['ledger']] == calendar[61:63]
    assert result['ledger'][0]['signal_date'] == calendar[60]
    assert result['ledger'][1]['reason'] == 'actual_account_stop'



def freeze_contract(value):
    value = copy.deepcopy(value)
    value['contract_hash'] = stable_hash({k: v for k, v in value.items() if k != 'contract_hash'})
    return value



def entry_fixture():
    dates = pd.bdate_range('2023-01-02', periods=80).strftime('%Y-%m-%d').tolist()
    frame = pd.DataFrame({c: np.sin(np.arange(80) / 7 + i) for i, c in enumerate(ENTRY_FEATURE_COLUMNS)})
    frame = frame.assign(date=dates, symbol='TEST.SZ', label=np.arange(80) % 2,
        label_end=(pd.to_datetime(dates) + pd.offsets.BDay(1)).strftime('%Y-%m-%d'),
        label_available=True, input_eligible=True,
        attempt_id=[f'attempt:{i}' for i in range(80)],
        economic_group_id=[f'original:{i}' for i in range(80)])
    frame['original_opportunity_ids'] = [[f'structure:original:{i}:BUY'] for i in range(80)]
    frame['original_event_ids'] = frame.original_opportunity_ids
    frame['original_event_first_pre_gate_dates'] = [json.dumps({f'structure:original:{i}:BUY': day})
        for i, day in enumerate(dates)]
    fold = {'name': 'unit', 'train_end': dates[29], 'calibration_start': dates[30],
        'calibration_end': dates[44], 'policy_start': dates[45],
        'policy_end': dates[59], 'validation_end': dates[59]}
    cfg = strategy_config({'epochs': 1, 'batch_size': 16, 'min_train_rows': 2, 'min_calibration_rows': 2})
    population = {'version': TRAINING_POPULATION, 'fold': fold, 'population_end': fold['policy_end'],
        'prior_entry_catalog': [], 'prior_entry_catalog_hash': stable_hash([]),
        'split_policy': SPLIT_POLICY, 'economic_weighting': ECONOMIC_WEIGHTING}
    cfg['reference_population'] = population
    contract = target_contract(head='entry', config=cfg, market_dates=dates,
        source_snapshot_hash='frozen-source', feature_schema_hash=schema_hash(ENTRY_FEATURE_COLUMNS, 'entry'),
        continuation_policy_hash='frozen-E1', initial_cash=cfg['initial_cash'], stage='E1',
        candidate_policy_hash='frozen-candidate')
    contract = freeze_contract({**contract, 'model_input_schema_hash': schema_hash(ENTRY_FEATURE_COLUMNS, 'entry')})
    frame.attrs['label_contract'] = contract
    return frame, dates, fold, cfg, contract


def holding_protocol_fixture():
    return {'version':HELD_TRAINING_POPULATION,'stage':'E1','reference_mode':'E1',
        'conditional_holding_domain':HELD_CONDITION_DOMAIN,
        'pending_L2_entry_or_exit_used':False,'candidate_clones_are_holding_population':False,
        'holding_checkpoint_upgrade':False,'split_policy':HELD_SPLIT_POLICY,
        'economic_weighting':HELD_ECONOMIC_WEIGHTING,
        'source_snapshot_hash':'frozen-source','continuation_policy_hash':'frozen-E1',
        'candidate_policy_hash':'frozen-candidate','manifest_path':'dataset/E1/manifest.json',
        'manifest_sha256':'fixture-manifest','pooled_exits_path':'dataset/E1/exits.parquet',
        'pooled_exits_sha256':'fixture-exits','records_hash':'fixture-records'}



@pytest.mark.parametrize('change', [
    {'version': 'czsc_trend_consensus'},
    {'consensus': {'version': 'czsc_trend_consensus'}},
    {'training_population': 'actual_flat_candidate_clone'},
    {'reassessment': {'quality_rejection_consumes': True}},
    {'reassessment': {'holding_sell': 'fresh_sell_event_only'}},
])
def test_current_configuration_rejects_old_or_partially_relabelled_policy(change):
    with pytest.raises(ValueError):
        strategy_config(change)



def test_future_unclosed_attempt_keeps_mature_history_without_cross_segment_group_leak():
    fold = {'train_end': '2025-01-10', 'calibration_start': '2025-01-13', 'calibration_end': '2025-01-24',
        'policy_start': '2025-01-27', 'policy_end': '2025-02-07', 'validation_end': '2025-02-07'}
    early = pd.DataFrame([
        {'symbol': 'A', 'date': '2025-01-02', 'label_end': '2025-01-06', 'label': 1., 'label_available': True, 'economic_group_id': 'same-original'},
        {'symbol': 'A', 'date': '2025-01-07', 'label_end': '2025-01-14', 'label': 0., 'label_available': True, 'economic_group_id': 'late-end'},
        {'symbol': 'A', 'date': '2025-01-13', 'label_end': '2025-01-15', 'label': 0., 'label_available': True, 'economic_group_id': 'new-cal'},
        {'symbol': 'A', 'date': '2025-01-27', 'label_end': '2025-01-29', 'label': 1., 'label_available': True, 'economic_group_id': 'new-policy'},
    ])
    future = pd.DataFrame([
        {'symbol': 'A', 'date': '2025-01-14', 'label_end': '2025-01-16', 'label': 0., 'label_available': True, 'economic_group_id': 'same-original'},
        {'symbol': 'A', 'date': '2025-02-10', 'label_end': None, 'label': np.nan, 'label_available': False, 'economic_group_id': 'same-original'},
    ])
    before = time_masks(early, fold)
    frame = pd.concat([early, future], ignore_index=True)
    after = time_masks(frame, fold)
    for previous, current in zip(before, after):
        assert np.array_equal(previous, current[:len(early)])
    assert after[0].tolist() == [True, False, False, False, False, False]
    assert after[1].tolist() == [False, False, True, False, False, False]
    assert after[2].tolist() == [False, False, False, True, False, False]
    groups = _group_ids(frame)
    assert not (set(groups[after[0]]) & set(groups[after[1]]))
    assert not (set(groups[after[1]]) & set(groups[after[2]]))



def test_daily_attempts_share_original_family_weight_while_distinct_stocks_do_not():
    frame = pd.DataFrame({'symbol': ['A', 'A', 'A', 'B'], 'economic_group_id': ['one', 'one', 'two', 'one'],
        'attempt_id': ['day1', 'day2', 'day1', 'day1']})
    weights = _weights(frame, np.ones(4, dtype=bool))
    assert weights[:2].sum() == pytest.approx(weights[2])
    assert weights[2] == pytest.approx(weights[3])
    assert weights.mean() == pytest.approx(1.)



def test_shared_atomic_opportunity_purge_and_weights_are_prefix_stable():
    fold = {'train_end': '2025-01-10', 'calibration_start': '2025-01-13', 'calibration_end': '2025-01-24',
        'policy_start': '2025-01-27', 'policy_end': '2025-02-07', 'validation_end': '2025-02-07'}
    def row(date, end, group, ids, first):
        return {'symbol': 'S', 'date': date, 'label_end': end, 'label': 1., 'label_available': True,
            'economic_group_id': group, 'original_event_ids': ids,
            'original_event_first_pre_gate_dates': json.dumps(first)}
    frame = pd.DataFrame([
        row('2025-01-02', '2025-01-06', 'A', ['A'], {'A': '2025-01-02'}),
        row('2025-01-13', '2025-01-15', 'AB', ['A', 'B'], {'A': '2025-01-02', 'B': '2025-01-13'}),
        row('2025-01-14', '2025-01-16', 'B', ['B'], {'B': '2025-01-13'}),
        row('2025-01-14', '2025-01-16', 'C', ['C'], {'C': '2025-01-14'}),
        row('2025-01-27', '2025-01-29', 'BD', ['B', 'D'], {'B': '2025-01-13', 'D': '2025-01-27'}),
        row('2025-01-27', '2025-01-29', 'E', ['E'], {'E': '2025-01-27'}),
    ])
    masks = time_masks(frame, fold)
    assert masks[0].tolist() == [True, False, False, False, False, False]
    # A+B cannot reuse A's training dependency. B's own first pre-gate date
    # lies in calibration; a later B+D cannot reuse it for policy selection.
    assert masks[1].tolist() == [False, False, True, True, False, False]
    assert masks[2].tolist() == [False, False, False, False, False, True]
    all_first_three = np.asarray([True, True, True, False, False, False])
    assert _weights(frame, all_first_three).tolist() == pytest.approx([1., 1., 1.])
    # A / A+B / B are one independent component, with total weight equal C.
    four = np.asarray([True, True, True, True, False, False])
    weights = _weights(frame, four)
    assert weights[:3].sum() == pytest.approx(weights[3])
    other = frame.iloc[0:1].copy()
    other['symbol'] = 'OTHER'
    separate_stock = pd.concat([frame.iloc[:4], other], ignore_index=True)
    separate_weights = _weights(separate_stock, np.ones(5, dtype=bool))
    assert separate_weights[:3].sum() == pytest.approx(separate_weights[3])
    assert separate_weights[3] == pytest.approx(separate_weights[4])
    pending = row('2025-02-10', None, 'ABC', ['A', 'B', 'C'],
        {'A': '2025-01-02', 'B': '2025-01-13', 'C': '2025-01-14'})
    pending.update(label=np.nan, label_available=False)
    extended = pd.concat([frame, pd.DataFrame([pending])], ignore_index=True)
    extended_masks = time_masks(extended, fold)
    for before, after in zip(masks, extended_masks):
        assert np.array_equal(before, after[:len(frame)])
        assert not after[-1]
        assert np.array_equal(_weights(frame, before), _weights(extended, after))
    assert extended.economic_group_id.iloc[0] == 'A' and extended.economic_group_id.iloc[1] == 'AB'



def test_future_atom_metadata_cannot_backfill_an_earlier_first_attempt():
    frame, _, fold, _, _ = entry_fixture()
    atom = frame.original_event_ids.iloc[0][0]
    future = frame.iloc[-1:].copy()
    future['original_event_ids'] = [[atom]]
    future['original_event_first_pre_gate_dates'] = json.dumps({atom: '2022-12-30'})
    with pytest.raises(ValueError, match='changed after observation'):
        time_masks(pd.concat([frame, future], ignore_index=True), fold)



@pytest.mark.parametrize('mutation', ['old-family', 'old-population', 'bad-population-hash', 'self-head', 'future-head'])
def test_entry_contract_enforces_population_and_strict_prior_dependency(mutation):
    _, _, fold, _, contract = entry_fixture()
    altered = copy.deepcopy(contract)
    if mutation == 'old-family':
        altered['family'] = 'czsc_trend_consensus'
    elif mutation == 'old-population':
        altered['training_population'] = 'actual_flat_candidate_clone'
    elif mutation == 'bad-population-hash':
        altered['reference_population_hash'] = 'not-the-population'
    else:
        population = altered['reference_population']
        population['prior_entry_catalog'] = [{'name': fold['name'] if mutation == 'self-head' else 'previous',
            'available_at': '2022-12-31' if mutation == 'self-head' else fold['calibration_start']}]
        population['prior_entry_catalog_hash'] = stable_hash(population['prior_entry_catalog'])
        altered['reference_population_hash'] = stable_hash(population)
    with pytest.raises(ValueError):
        _validate_contract(freeze_contract(altered), 'entry', ENTRY_FEATURE_COLUMNS)



def test_old_signed_model_manifests_are_rejected_before_checkpoint_load(tmp_path):
    from my_strategy.services.czsc_trend_model import load_trend_head
    directory = tmp_path / 'v1'
    directory.mkdir()
    old = {'family': 'czsc_trend_consensus', 'head': 'entry'}
    old['manifest_sha256'] = stable_hash(old)
    (directory / 'manifest.json').write_text(json.dumps(old))
    with pytest.raises(ValueError, match='identity'):
        load_head(directory)
    with pytest.raises(ValueError, match='identity'):
        load_trend_head(directory)



def test_real_new_head_serialization_cutoffs_and_caller_schema(tmp_path):
    frame, _, fold, cfg, contract = entry_fixture()
    manifest = train_head(frame, ENTRY_FEATURE_COLUMNS, tmp_path / 'new-head', fold, cfg,
        head='entry', device='cpu', contract=contract)
    loaded = load_head(tmp_path / 'new-head')
    assert loaded['family'] == STRATEGY_ID
    assert loaded['target_contract']['training_population'] == TRAINING_POPULATION
    assert manifest['train_label_end'] <= fold['train_end']
    assert manifest['calibration_label_end'] <= fold['calibration_end']
    selected = frame.iloc[61:64].copy()
    selected = bind_inputs(selected, ENTRY_FEATURE_COLUMNS, 'entry')
    p = HeadSession().predict(selected, tmp_path / 'new-head', expected_head='entry', expected_contract=contract)
    assert p.shape == (3,) and np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    selected.attrs['schema_hash'] = 'frozen-v1-schema'
    with pytest.raises(ValueError, match='schema'):
        HeadSession().predict(selected, tmp_path / 'new-head', expected_contract=contract)
    early = bind_inputs(frame.iloc[59:60], ENTRY_FEATURE_COLUMNS, 'entry')
    with pytest.raises(ValueError, match='unavailable'):
        HeadSession().predict(early, tmp_path / 'new-head', expected_contract=contract)



def test_entry_training_rejects_duplicate_daily_attempt_and_wrong_fold(tmp_path):
    frame, _, fold, cfg, contract = entry_fixture()
    duplicate = frame.copy()
    duplicate.loc[1, 'attempt_id'] = duplicate.loc[0, 'attempt_id']
    with pytest.raises(ValueError, match='unique'):
        train_head(duplicate, ENTRY_FEATURE_COLUMNS, tmp_path / 'duplicates', fold, cfg,
            head='entry', device='cpu', contract=contract)
    with pytest.raises(ValueError, match='another chronological fold'):
        train_head(frame, ENTRY_FEATURE_COLUMNS, tmp_path / 'wrong-fold', {**fold, 'name': 'another'}, cfg,
            head='entry', device='cpu', contract=contract)



def test_real_current_three_class_head_keeps_own_family_and_actual_completion_clock(tmp_path):
    from my_strategy.services.czsc_trend_model import (SCHEMA_HASH, TrendSession,
        train_trend_head, trend_contract, load_trend_head)
    frame, _, fold, cfg, _ = entry_fixture()
    frame['label'] = np.arange(len(frame)) % 3
    frame['observation_id'] = [stable_hash({'symbol': 'TEST.SZ', 'date': day}) for day in frame.date]
    frame['model_input_eligible'] = True
    cfg['trend'].update(min_train_per_class=2, min_calibration_per_class=2, min_policy_per_class=2)
    contract = trend_contract(cfg, source_snapshot_hash='frozen-source', pipeline_contract_hash='current-pipeline')
    frame.attrs['label_contract'] = contract
    train_trend_head(frame, tmp_path / 'trend', fold, cfg, contract=contract)
    manifest = load_trend_head(tmp_path / 'trend')
    assert manifest['family'] == STRATEGY_ID and manifest['classes'] == ['UP', 'DOWN', 'NEUTRAL']
    future = frame.iloc[61:64].copy()
    future.attrs['schema_hash'] = SCHEMA_HASH
    p, _ = TrendSession().predict(future, tmp_path / 'trend', expected_contract=contract)
    assert p.shape == (3, 3) and np.allclose(p.sum(axis=1), 1.)
    with pytest.raises(ValueError, match='not actually trained'):
        TrendSession().predict(future, tmp_path / 'trend', expected_contract=contract, mode='production')



@pytest.mark.parametrize('head', ['ma_exit', 'exit', 'exit_advantage', 'return', 'loss'])
def test_real_head_manifest_binds_actual_dependency_policy(head, tmp_path):
    frame, dates, fold, cfg, _ = entry_fixture()
    held = head in {'ma_exit', 'exit', 'exit_advantage'}
    columns = L1_MA_EXIT_COLUMNS if head == 'ma_exit' else EXIT_FEATURE_COLUMNS if held else ENTRY_FEATURE_COLUMNS
    for column in columns:
        if column not in frame: frame[column] = .1
    if held:
        frame['episode_id'] = [f'actual-episode:{i}' for i in range(len(frame))]
        if head!='ma_exit':
            cfg['holding_population']=holding_protocol_fixture()
            frame['stage']='E1';frame['holding_reference_mode']='E1'
        # Holding heads do not claim atomic preprocessing, even if an audit
        # table also happens to include original entry provenance columns.
        frame['original_event_ids'] = [['same-original']] * len(frame)
        frame['original_event_first_pre_gate_dates'] = json.dumps({'same-original': dates[0]})
    continuous = head in {'return', 'exit_advantage'}
    if continuous: frame['label'] = np.sin(np.arange(len(frame))) / 10
    contract = target_contract(head=head, config=cfg, market_dates=dates,
        source_snapshot_hash='frozen-source', feature_schema_hash=schema_hash(columns, head),
        continuation_policy_hash='frozen-E0' if head == 'ma_exit' else 'frozen-E1',
        initial_cash=cfg['initial_cash'], stage='E0' if head == 'ma_exit' else 'E1',
        candidate_policy_hash='frozen-candidate')
    contract = freeze_contract({**contract, 'model_input_schema_hash': schema_hash(columns, head)})
    frame.attrs['label_contract'] = contract
    manifest = train_head(frame, columns, tmp_path / head, fold, cfg, head=head,
        regression=continuous, device='cpu', contract=contract)
    expected = head_dependency_policy(head)
    assert manifest['group_split_policy'] == expected['split_policy']
    assert manifest['episode_weighting'] == expected['economic_weighting']
    assert manifest['economic_group_identity'] == expected['economic_group_identity']
    assert load_head(tmp_path / head)['target_contract']['split_policy'] == expected['split_policy']
    assert manifest['sample_counts']['calibration'] > 0
    if held:
        assert contract['split_policy'] == HELD_SPLIT_POLICY
        assert contract['economic_weighting'] == HELD_ECONOMIC_WEIGHTING
        assert contract['economic_group_identity'] == 'symbol_actual_held_episode_id'
        assert 'reference_population' not in contract
    else:
        assert contract['training_population'] == TRAINING_POPULATION
        broken = frame.drop(columns=['original_event_first_pre_gate_dates'])
        with pytest.raises(ValueError, match='identity fields missing'):
            train_head(broken, columns, tmp_path / (head + '-broken'), fold, cfg, head=head,
                regression=continuous, device='cpu', contract=contract)



def test_holding_episode_weights_do_not_merge_shared_entry_atoms():
    frame = pd.DataFrame({'symbol': ['S'] * 4, 'episode_id': ['actual1', 'actual1', 'actual2', 'actual3'],
        'economic_group_id': ['A', 'AB', 'B', 'A'],
        'original_event_ids': [['A'], ['A', 'B'], ['B'], ['A']]})
    mask = np.ones(len(frame), dtype=bool)
    held = _weights(frame, mask, head='exit')
    assert held[:2].sum() == pytest.approx(held[2])
    assert held[2] == pytest.approx(held[3])
    assert not np.allclose(held, _weights(frame, mask, head='entry'))
    frame.attrs['label_contract'] = {'model_head': 'exit'}
    assert np.array_equal(_weights(frame, mask), held)



def test_holding_contract_rejects_inaccurate_atomic_policy_claim():
    _, dates, _, cfg, _ = entry_fixture()
    contract = target_contract(head='ma_exit', config=cfg, market_dates=dates,
        source_snapshot_hash='frozen-source', feature_schema_hash=schema_hash(L1_MA_EXIT_COLUMNS, 'ma_exit'),
        continuation_policy_hash='E0', initial_cash=cfg['initial_cash'], stage='E0', candidate_policy_hash='candidate')
    contract['model_input_schema_hash'] = schema_hash(L1_MA_EXIT_COLUMNS, 'ma_exit')
    contract = freeze_contract({**contract, 'split_policy': SPLIT_POLICY})
    with pytest.raises(ValueError, match='dependency policy mismatch'):
        _validate_contract(contract, 'ma_exit', L1_MA_EXIT_COLUMNS)



def identity(head="trend_direction", family=STRATEGY_ID):
    return {"family": family, "head": head, "verified_frozen_contract": True,
        "checkpoint": "strict-prior", "manifest_sha256": "frozen-manifest", "feature_schema_hash": "frozen-schema",
        "label_contract_hash": "frozen-target", "train_label_end": "2024-12-31", "calibration_label_end": "2025-01-31",
        "policy_label_end": "2025-02-28", "simulated_available_at": "2025-03-01T15:00:00+08:00"}



def fixture(length=85, *, stop=9., boundary=9.3, prices=None, sell_from=65):
    dates = pd.bdate_range("2026-01-01", periods=length).strftime("%Y-%m-%d").tolist()
    frame = pd.DataFrame([{"date": date, "symbol": SYMBOL, "open": (prices or {}).get(i, 10.),
        "high": (prices or {}).get(i, 10.) + .2, "low": (prices or {}).get(i, 10.) - .2,
        "close": (prices or {}).get(i, 10.), "volume": 4000000., "amount": 40000000.,
        "source": "baostock", "has_trade_price": 1,
        "close_above_ma20": 1., "ma20_slope5": .1, "ma_alignment": 1., "ret_5": .1} for i, date in enumerate(dates)])
    frame.attrs["source_context_hash"] = "frozen-quotes-for-Broker-contract-test"
    frame.attrs.update(volume_unit='shares', amount_unit='CNY')
    at = dates[60] + "T15:00:00+08:00"
    fact = {"symbol": SYMBOL, "date": dates[60], "available_at": at, "first_available_at": at,
        "first_confirmed_at": at, "event_id": "original-buy", "original_event_id": "original-buy",
        "pattern_id": "original-pattern", "source": "structure", "side": "BUY", "confirmed": True,
        "input_eligible": True, "invalidation_price": boundary, "risk_stop_price": stop}
    evaluations, trends = [], []
    for i, date in enumerate(dates):
        at, close = date + "T15:00:00+08:00", float(frame.iloc[i].close)
        bearish = sell_from is not None and i >= sell_from
        structure = {"date": date, "available_at": at, "source_id": "structure:" + date,
            "version": STRUCTURE_RULE_VERSION, "input_eligible": True,
            "support_facts": [copy.deepcopy(fact)] if i >= 60 else [],
            "current_state": {"close": close, "buy_score": .1 if bearish else .9 if i >= 60 else 0.,
                "sell_score": .85 if bearish else 0., "support_valid": True, "support_floor": boundary,
                "resistance_valid": True, "resistance_ceiling": 11.}}
        wyckoff = {"date": date, "available_at": at, "source_id": "wyckoff:" + date,
            "version": W_RULE_VERSION, "input_eligible": True, "support_facts": [],
            "current_state": {"close": close, "buy_score": .05 if bearish else .65, "sell_score": .8 if bearish else .05}}
        evaluations.append({"date": date, "available_at": at, "structure": structure, "wyckoff": wyckoff})
        p = (.1, .5, .4) if bearish else (.65, .1, .25)
        trends.append({"date": date, "available_at": at, "input_eligible": True, "model_available": True,
            "p_up": p[0], "p_down": p[1], "p_neutral": p[2], "model_identity": identity()})
    return frame, dates, {"events": [fact], "evaluations": evaluations}, trends



def prepare(frame, dates, pack, trends, cfg=None):
    return prepare_trend_decisions(frame, dates, native_pack=pack, trend_probabilities=trends, config=cfg, mode="E0")



def score(snapshot, probability=.2, *, family=STRATEGY_ID, in_scope=True):
    return {"date": snapshot["date"], "available_at": snapshot["available_at"], "status": "scored",
        "probability": probability, "in_scope": in_scope, "model_identity": identity("entry", family)}



def test_prepare_reassesses_every_valid_original_day_without_consumption_or_entry_cooldown():
    frame, dates, pack, trends = fixture(sell_from=None)
    result = prepare(frame, dates, pack, trends)
    assert all(result["rows"][i]["entry_eligible"] for i in range(60, 80))
    assert all(result["rows"][i]["action"] == "BUY" for i in range(60, 80))
    assert not result["rows"][80]["entry_eligible"]
    assert result["rows"][60]["original_opportunity_ids"] == result["rows"][79]["original_opportunity_ids"]
    assert result["rows"][60]["entry_plan"]["plan_id"] != result["rows"][61]["entry_plan"]["plan_id"]
    assert prepare(frame.iloc[:70], dates, pack, trends)["rows"] == result["rows"][:70]



def test_l2_reject_missing_then_pass_same_opportunity_real_fill_and_no_event_continuous_sell():
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    observed = []

    def evaluator(snapshot, candidate):
        observed.append((copy.deepcopy(snapshot), copy.deepcopy(candidate)))
        if snapshot["date"] == dates[61]:
            return {"status": "model_unavailable", "probability": None, "reason_codes": ["no_prior_high_return_head"]}
        return score(snapshot, .02 if snapshot["date"] == dates[60] else .2)

    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", entry_evaluator=evaluator,
        run_id="same-opportunity", account_id="same-account")
    diagnostics = result["entry_diagnostics"]
    assert [row["date"] for row in diagnostics] == dates[60:63]
    assert [row["allowed"] for row in diagnostics] == [False, False, True]
    assert len({row["attempt_id"] for row in diagnostics}) == 3
    assert len({row["economic_group_id"] for row in diagnostics}) == 1
    for row in diagnostics:
        state = row["pre_gate_state"]
        assert state["shares"] == 0 and state["pending_intent"] is None
        assert state["cash"] == state["initial_cash"] == 100000
        assert state["reserved_opportunity_ids"] == []
        assert row["reference_state_hash"] == stable_hash(state)
        assert row["entry_plan_hash"] == stable_hash(row["candidate"]["entry_plan"])
        assert row["snapshot"]["cash"] == state["cash"]
    buy, sell = result["ledger"]
    assert (buy["action"], buy["signal_date"], buy["date"]) == ("BUY", dates[62], dates[63])
    assert (sell["action"], sell["signal_date"], sell["date"]) == ("SELL", dates[65], dates[66])
    assert buy["shares"] > 0 and buy["fee"] > 0 and sell["position_after"] == 0 and sell["fee"] > 0
    assert sell["reason"] == "ordinary_consensus_sell"
    assert prepared["rows"][65]["sell_pass"] is True
    assert prepared["rows"][65]["candidate_evidence"] == []
    assert not any(event["side"] == "SELL" for event in prepared["rows"][65]["active_candidates"])
    assert result["terminal_state"]["reserved_opportunity_ids"] == diagnostics[2]["original_opportunity_ids"]



def test_pre_gate_clone_exists_after_reference_E1_already_admitted_and_owns_real_cash():
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    reference = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E1", run_id="E1-reference")
    assert reference["holding_snapshots"][0]["date"] == dates[61]
    final = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="gated-reference",
        entry_evaluator=lambda snapshot, candidate: score(snapshot, .02 if snapshot["date"] == dates[60] else .2))
    diagnostic = final["entry_diagnostics"][1]
    assert diagnostic["date"] == dates[61] and diagnostic["pre_gate_state"]["shares"] == 0
    assert not any(row["date"] == dates[61] for row in reference["entry_diagnostics"])
    clone = clone_candidate_account(frame, dates, prepared, diagnostic["pre_gate_state"], diagnostic["candidate"], run_id="label-clone")
    assert [row["action"] for row in clone["ledger"]] == ["BUY", "SELL"]
    assert clone["ledger"][0]["date"] == dates[62]
    assert clone["ledger"][0]["signal_date"] == dates[61]
    assert clone["metrics"]["open_shares"] == 0 and clone["metrics"]["fees"] > 0



def test_quality_probability_alone_cannot_bypass_total_gate_or_consume_opportunity():
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    prepared["rows"][60]["buy_base_score"] = .4
    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="total-gate",
        entry_evaluator=lambda snapshot, candidate: score(snapshot))
    rejected, accepted = result["entry_diagnostics"][:2]
    assert rejected["reason"] == "buy_total_support_below_threshold" and not rejected["allowed"]
    assert accepted["date"] == dates[61] and accepted["allowed"]
    assert rejected["original_opportunity_ids"] == accepted["original_opportunity_ids"]
    assert result["ledger"][0]["date"] == dates[62]



@pytest.mark.parametrize("head,expected_sell", [("holding_exit", 64), ("exit", 66)])
def test_exit_binding_uses_accepted_signal_after_reassessment_and_L2_exit_stays_shadow(head, expected_sell):
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    snapshots = []

    def exit_policy(snapshot):
        snapshots.append(copy.deepcopy(snapshot))
        return {"date": snapshot["date"], "available_at": snapshot["available_at"], "usage_mode": "historical",
            "head": head, "probability": .9, "apply": True, "request_exit": True, "research_arm": True,
            "holding_contract_compatible": True, "contract_hash": "frozen-exit-target",
            "model_identity": identity("ma_exit" if head == "holding_exit" else "exit"),
            "entry_exit_binding": {"plan_id": snapshot["plan_id"], "candidate_signal_date": snapshot["entry_signal_date"],
                "checkpoint": "strict-prior"}}

    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="pinned-reassessment",
        entry_evaluator=lambda snapshot, candidate: score(snapshot, .02 if snapshot["date"] < dates[62] else .2),
        exit_policy=exit_policy)
    assert result["ledger"][0]["signal_date"] == dates[62]
    assert snapshots[0]["entry_signal_date"] == dates[62] != pack["events"][0]["date"]
    assert result["ledger"][-1]["date"] == dates[expected_sell]
    assert result["ledger"][-1]["reason"] == ("signal_pinned_ma_exit" if head == "holding_exit" else "ordinary_consensus_sell")
    if head == "exit":
        assert all(not row["applied"] for row in result["exit_policy_diagnostics"])
        assert all(row["probability"] is None for row in result["exit_policy_diagnostics"])



def test_continuous_sell_partial_fills_keep_one_first_intent_until_inventory_is_flat():
    frame, dates, pack, trends = fixture()
    frame.loc[[65,66], "volume"] = 15000.
    frame.loc[[65,66], "amount"] = 150000.
    result = replay_trend_account(frame, dates, prepare(frame, dates, pack, trends), start=dates[60], mode="E0",
        run_id="partial-continuous-sell")
    buy, *sells = result["ledger"]
    assert len(sells) == 3 and [row["date"] for row in sells] == dates[66:69]
    assert all(row["signal_date"] == dates[65] for row in sells)
    assert len({row["intent_id"] for row in sells}) == 1
    assert sum(row["shares"] for row in sells) == buy["shares"]
    assert len([row for row in result["final_action_intents"] if row["side"] == "SELL"]) == 1
    assert result["metrics"]["open_shares"] == 0



def test_risk_block_does_not_call_l2_or_reserve_and_later_valid_day_can_enter():
    prices = {i: 9.4 for i in range(62, 85)}
    prices[61] = 9.6
    frame, dates, pack, trends = fixture(stop=8., boundary=7.7, prices=prices, sell_from=65)
    prepared = prepare(frame, dates, pack, trends)
    assert not prepared["rows"][60]["entry_eligible"] and not prepared["rows"][61]["entry_eligible"]
    assert prepared["rows"][62]["entry_eligible"]
    scored = []
    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final",
        entry_evaluator=lambda snapshot, candidate: (scored.append(snapshot["date"]) or score(snapshot)), run_id="risk-recovers")
    assert scored == [dates[62]]
    assert result["ledger"][0]["signal_date"] == dates[62]
    assert result["ledger"][0]["date"] == dates[63]
    assert all(not state["reserved_opportunity_ids"] for state in result["account_snapshots"][:2])



def test_broker_rejection_keeps_one_original_reservation_and_never_retries_buy():
    frame, dates, pack, trends = fixture(sell_from=None)
    frame.loc[61, ["open", "high"]] = [12., 12.1]
    prepared = prepare(frame, dates, pack, trends)
    evaluations = []
    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="one-shot",
        entry_evaluator=lambda snapshot, candidate: (evaluations.append(snapshot["date"]) or score(snapshot)))
    assert evaluations == [dates[60]]
    assert len(result["final_action_intents"]) == len(result["rejections"]) == 1
    assert result["ledger"] == [] and result["metrics"]["open_shares"] == 0
    assert result["terminal_state"]["reserved_opportunity_ids"] == result["entry_diagnostics"][0]["original_opportunity_ids"]
    resumed = replay_trend_account(frame, dates, prepared, initial_state=result["account_snapshots"][4], mode="final",
        entry_evaluator=lambda snapshot, candidate: pytest.fail("reserved original must not be evaluated again"), run_id="resumed")
    assert resumed["ledger"] == [] and resumed["entry_diagnostics"] == []



def test_future_calendar_or_run_lineage_does_not_change_old_attempt_but_known_cash_does():
    frame, dates, pack, trends = fixture()
    prepared = prepare(frame, dates, pack, trends)
    original = replay_trend_account(frame, dates, prepared, start=dates[60], end=dates[62], mode="final", run_id="original")
    calendar = [*dates, "2027-01-04"]
    changed = replay_trend_account(frame, calendar, prepared, start=dates[60], end=dates[62], mode="final", run_id="new-run")
    for old, new in zip(original["entry_diagnostics"], changed["entry_diagnostics"], strict=True):
        assert old["reference_state_hash"] != new["reference_state_hash"]
        assert old["flat_state_hash"] == new["flat_state_hash"] and old["attempt_id"] == new["attempt_id"]
    state = copy.deepcopy(original["entry_diagnostics"][0]["pre_gate_state"])
    state["cash"] = 90000.
    assert _flat_state_hash(state) != original["entry_diagnostics"][0]["flat_state_hash"]



def test_new_unreserved_source_does_not_borrow_old_reserved_opportunity_stop():
    frame, dates, pack, trends = fixture(stop=9.5)
    new = copy.deepcopy(pack["events"][0])
    new.update(event_id="new-buy", original_event_id="new-buy", pattern_id="new-pattern", date=dates[70],
        available_at=dates[70]+"T15:00:00+08:00", first_available_at=dates[70]+"T15:00:00+08:00",
        first_confirmed_at=dates[70]+"T15:00:00+08:00", risk_stop_price=9.)
    pack["events"].append(new)
    for i in range(70, len(dates)):
        pack["evaluations"][i]["structure"]["support_facts"].append(copy.deepcopy(new))
        if i < 75:
            pack["evaluations"][i]["structure"]["current_state"].update(buy_score=.9, sell_score=0.)
            pack["evaluations"][i]["wyckoff"]["current_state"].update(buy_score=.65, sell_score=.05)
            trends[i].update(p_up=.55, p_down=.1, p_neutral=.35)  # Continuous support without a new MA cross.
    prepared = prepare(frame, dates, pack, trends)
    assert prepared["rows"][70]["entry_plan"]["stop_price"] == 9.5
    result = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E0", run_id="new-source-only")
    second = next(row for row in result["entry_diagnostics"] if row["date"] == dates[70])
    assert second["candidate"]["entry_plan"]["stop_price"] == 9.
    assert second["candidate"]["entry_plan"]["source_event_ids"] == ["new-buy"]
    assert second["original_opportunity_ids"] == ["structure:new-pattern:BUY"]
    assert [row["action"] for row in result["ledger"]] == ["BUY", "SELL", "BUY", "SELL"]



def test_original_expiry_clock_and_permanent_boundary_break_do_not_resurrect():
    frame, dates, pack, trends = fixture(sell_from=None)
    delayed = copy.deepcopy(pack["events"][0]); delayed.update(date=dates[70], available_at=dates[70]+"T15:00:00+08:00")
    pack["events"] = [delayed]
    result = prepare(frame, dates, pack, trends)
    assert result["rows"][70]["entry_eligible"] and not result["rows"][80]["entry_eligible"]
    frame, dates, pack, trends = fixture(prices={61:9.2}, sell_from=None)
    result = prepare(frame, dates, pack, trends)
    assert result["rows"][60]["entry_eligible"]
    assert not result["rows"][61]["active_candidates"]
    assert not result["rows"][62]["active_candidates"]  # Rebound must not revive the broken original boundary.



@pytest.mark.parametrize("family,in_scope,reason", [("czsc_trend_consensus", True, "identity_error"),
    (STRATEGY_ID, False, "high_return_population_out_of_scope")])
def test_old_family_or_out_of_scope_probability_never_bypasses_entry_gate(family, in_scope, reason):
    frame, dates, pack, trends = fixture()
    result = replay_trend_account(frame, dates, prepare(frame, dates, pack, trends), start=dates[60], mode="final",
        entry_evaluator=lambda snapshot, candidate: score(snapshot, .9, family=family, in_scope=in_scope), run_id="reject-contract")
    assert result["ledger"] == []
    assert all(row["reason"] == reason and not row["allowed"] for row in result["entry_diagnostics"])
    assert result["terminal_state"]["reserved_opportunity_ids"] == []



def test_actual_stop_overrides_continuous_sell_and_flat_sell_creates_no_broker_trade():
    frame, dates, pack, trends = fixture(prices={65:9.1, **{i:9.1 for i in range(66,85)}})
    frame.loc[65, ["open", "high"]] = [9.35, 9.4]
    prepared = prepare(frame, dates, pack, trends)
    actual = replay_trend_account(frame, dates, prepared, start=dates[60], mode="E0", run_id="stop-priority")
    sell = actual["ledger"][-1]
    assert sell["reason"] == "actual_account_stop" and "ordinary_consensus_sell" in sell["all_triggered_reasons"]
    flat = replay_trend_account(frame, dates, prepared, start=dates[60], mode="final", run_id="flat-no-model")
    assert flat["ledger"] == [] and flat["final_action_intents"] == []
    assert len(flat["entry_diagnostics"]) > 1



def inputs(monkeypatch):
    frame,calendar,native_pack,trends=fixture(90)
    prepared=prepare(frame,calendar,native_pack,trends)
    frame["model_input_eligible"]=True
    for column in MA_TREND_COLUMNS:frame[column]=.1
    frame['native_structure_input_eligible']=False;frame['native_w_input_eligible']=False
    template=copy.deepcopy(prepared['rows'][60]['entry_plan'])
    for row in prepared['rows']:
        row.update(action='WAIT',entry_eligible=False,entry_plan=None,fresh_candidate_ids=[])
    for index in (70,71,72):
        plan={**template,'plan_id':'plan-'+str(index),'signal_date':calendar[index],
            'available_at':calendar[index]+'T15:00:00+08:00'}
        prepared['rows'][index].update(action='BUY',entry_eligible=True,entry_plan=plan,
            original_opportunity_ids=['same-original'],fresh_candidate_ids=['same-original'],buy_base_score=.65)
    prepared['events']=[]
    monkeypatch.setattr('my_strategy.services.czsc_trend_dataset.prepare_trend_decisions',lambda *args,**kwargs:copy.deepcopy(prepared))
    config=strategy_config({'feature_start':calendar[70],'account':{'max_holding_sessions':3}})
    config.update(source_snapshot_hash='s',pipeline_contract_hash='p',candidate_policy_hash='c',E0_policy_hash='e0',E1_policy_hash='e1')
    return frame,calendar,prepared,config



def test_missing_prior_l2_preserves_all_daily_flat_attempts_without_actual_buys(monkeypatch,tmp_path):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    def blocked(snapshot,candidate):
        return {'status':'no_prior_high_return_head','probability':None}
    result,summary=build_account_dataset(frame,calendar,{}, {},cfg,stage='E1',
        reference_entry_evaluator=blocked,reference_population={'version':pipeline.POPULATION_VERSION},
        population_end=calendar[72],directory=tmp_path)
    entries=result['entries']
    assert len(entries)==entries.attempt_id.nunique()==3
    assert entries.economic_group_id.nunique()==1
    assert entries.date.tolist()==calendar[70:73]
    assert entries.reference_allowed.eq(False).all()
    assert entries.reference_gate_status.eq('no_prior_high_return_head').all()
    assert entries.actual_flat_reference.all() and entries.candidate_cash_ratio.eq(1).all()
    assert entries.label_available.all() and entries.label.eq(0).all()
    assert entries.entry_date.tolist()==calendar[71:74]
    reference=__import__('my_strategy.services.czsc_account_artifacts',fromlist=['read_account_artifact']).read_account_artifact(tmp_path/'reference_account.json')
    assert reference['ledger']==[]
    assert all(row['pre_gate_state']['pending_intent'] is None for row in reference['entry_diagnostics'])
    clones=__import__('my_strategy.services.czsc_account_artifacts',fromlist=['read_account_artifact']).read_account_artifact(tmp_path/'candidate_clones.json')
    assert len(clones)==3 and all(len([fill for fill in row['ledger'] if fill['action']=='BUY'])==1 for row in clones)
    assert summary['reference_mode']=='final' and summary['ordinary_sell_policy']=='continuous_sell_pass_for_actual_holding'
    assert result['exits'].empty



def test_training_never_uses_gate_after_state_with_queued_buy(monkeypatch,tmp_path):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    result,summary=build_account_dataset(frame,calendar,{}, {},cfg,stage='E0',directory=tmp_path)
    entries=result['entries']
    assert len(entries)==1 and entries.reference_allowed.iloc[0]
    reference=__import__('my_strategy.services.czsc_account_artifacts',fromlist=['read_account_artifact']).read_account_artifact(tmp_path/'reference_account.json')
    diagnostic=reference['entry_diagnostics'][0]
    after=next(state for state in reference['account_snapshots'] if state['last_close_date']==calendar[70])
    assert diagnostic['pre_gate_state']['pending_intent'] is None
    assert after['pending_intent']['side']=='BUY'
    assert entries.reference_state_hash.iloc[0]==stable_hash(diagnostic['pre_gate_state'])
    poisoned=copy.deepcopy(diagnostic);poisoned['pre_gate_state']=after
    with pytest.raises(ValueError,match='pre-gate'):
        pre_gate_attempt(poisoned,frame.symbol.iloc[0],cfg)



def test_future_label_clones_do_not_create_future_reference_population(monkeypatch):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    blocked=lambda *args:{'status':'model_unavailable','probability':None}
    result,summary=build_account_dataset(frame,calendar,{}, {},cfg,stage='E1',reference_entry_evaluator=blocked,
        population_end=calendar[70])
    assert result['entries'].date.tolist()==[calendar[70]]
    assert result['entries'].label_end.iloc[0]>calendar[70]
    assert not result['entries'].reference_allowed.iloc[0]
    assert summary['entries_mature_at_frozen_source']==1
    assert summary['entries_mature_by_population_end']==0
    assert summary['entries_positive_by_population_end']==0
    assert 'mature_entries' not in summary and 'held_candidates' not in summary['counts']



def fold():
    return {'name':'later','train_end':'2025-06-30','calibration_start':'2025-07-01',
        'calibration_end':'2025-08-15','policy_start':'2025-08-18','policy_end':'2025-09-30'}



def bundle(name='prior',available='2025-03-31'):
    return {'name':name,'available_at':available,'heads':{'entry':{'path':'models/linear/prior/entry','manifest_sha256':'manifest'}}}



def test_population_catalog_is_frozen_and_rejects_own_or_nonprior_head():
    original=[bundle()]
    population=make_reference_population(fold(),original)
    original[0]['name']='mutated'
    assert population['prior_entry_catalog'][0]['name']=='prior'
    assert population['prior_entry_catalog_hash']==stable_hash(population['prior_entry_catalog'])
    assert population['fold']==fold() and population['population_end']=='2025-09-30'
    assert not population['self_head_used']
    for candidate in (bundle(name='later'),bundle(available='2025-07-01')):
        with pytest.raises(ValueError,match='earlier'):
            make_reference_population(fold(),[candidate])



def test_entry_contract_binds_specific_real_reference_population(monkeypatch):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    population=make_reference_population(fold(),[])
    dataset={'entries':pd.DataFrame(), 'reference_population':population}
    contract=training_frame(dataset,'entry',cfg,calendar,'E1').attrs['label_contract']
    assert contract['reference_population']==population
    assert contract['reference_population_hash']==stable_hash(population)
    assert contract['training_population']==pipeline.POPULATION_VERSION



def test_source_generator_hashes_resolve_current_loaded_package(tmp_path,monkeypatch):
    hashes={}
    for relative in source.NATIVE_GENERATING_FILES:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('frozen')
        hashes[relative]=pipeline._file_hash(path)
    monkeypatch.setattr(pipeline,'MY_STRATEGY_ROOT',tmp_path)
    verify_source_generators({'code_hashes':hashes})
    (tmp_path/source.NATIVE_GENERATING_FILES[0]).write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        verify_source_generators({'code_hashes':hashes})



@pytest.mark.parametrize('empty_population,missing_trend',[(False,False),(True,False),(True,True)])
def test_pipeline_builds_each_actual_population_before_its_own_entry_head(tmp_path,monkeypatch,empty_population,missing_trend):
    """Exercise the real stage orchestration with cheap fitting/data workers."""
    class Context:
        run_id='new-current'
        def run_dir(self):return tmp_path/self.run_id
        def subdir(self,*parts):
            path=self.run_dir().joinpath(*parts);path.mkdir(parents=True,exist_ok=True);return path
        def write_metadata(self,*args):pass
    context=Context()
    source_dir=tmp_path/'fresh/reports';source_dir.mkdir(parents=True)
    raw=tmp_path/'raw.csv';raw.write_text('new raw observations')
    record={'symbol':'000001.SZ','raw_path':str(raw),'raw_sha256':pipeline._file_hash(raw),
        'raw_only_rebuild':True,'reused_model_or_label':False,'raw_series_hash':'raw-only',
        'raw_attrs':{'price_basis':'unadjusted','legacy_fuyao_bars':0,'unverified_trade_price_bars':0,
            'volume_unit':'shares','amount_unit':'CNY','data_version':'raw-only'}}
    calendar=pd.bdate_range('2023-01-02','2026-10-08').strftime('%Y-%m-%d').tolist()
    from my_strategy.services.czsc_liquidity import liquidity_policy
    pool_policy=liquidity_policy()
    source={'strategy_version':STRATEGY_ID,'status':'completed','raw_only_rebuild':True,'reused_model_or_label':False,
        'population_mode':'full_liquidity_union','liquidity':{'policy':pool_policy,'policy_hash':stable_hash(pool_policy),
            'selected_symbols':[record['symbol']],'daily_membership_hash':'daily','population_hash':'population'},
        'data_end':calendar[-1],'calendar_dates':calendar,'dataset_records':[record],
        'source_snapshot':{'calendar_run_id':'new-raw','raw_input_hashes':{record['symbol']:record['raw_sha256']}}}
    (source_dir/'native_source.json').write_text(json.dumps(source))
    monkeypatch.setattr(pipeline,'ARTIFACT_RUNS_ROOT',tmp_path)
    monkeypatch.setattr(pipeline,'GENERATING_FILES',('services/czsc_trend_dataset.py',))
    monkeypatch.setattr(pipeline,'verify_source_generators',lambda value:None)
    monkeypatch.setattr(pipeline,'create_run_context',lambda **kwargs:context)
    real_frame,real_calendar,prepared,cfg=inputs(monkeypatch)
    real_frame['symbol']=record['symbol']
    for row in prepared['rows']:
        row.update(action='WAIT',entry_eligible=False,entry_plan=None,fresh_candidate_ids=[])
    monkeypatch.setattr('my_strategy.services.czsc_trend_dataset.prepare_trend_decisions',
        lambda *args,**kwargs:copy.deepcopy(prepared))
    monkeypatch.setattr(pipeline,'read_snapshot',lambda record:(real_frame.copy() if empty_population else pd.DataFrame({'date':[calendar[0]]}),{}))
    def observations(*args,**kwargs):
        value=pd.DataFrame({'date':[calendar[0]],'label':[1],'symbol':[record['symbol']],
            'label_end':[calendar[1]],'input_eligible':[True],'label_available':[True],'observation_id':['one']})
        for column in MA_TREND_COLUMNS:value[column]=.1
        value.attrs['label_contract']={};return value
    monkeypatch.setattr(pipeline,'build_trend_labels',observations)
    earlier={'name':'trend-prior','train_end':'2024-06-28','validation_start':'2024-07-01',
        'validation_end':'2024-09-30','calibration_start':'2024-07-01',
        'calibration_end':'2024-08-15','policy_start':'2024-08-16','policy_end':'2024-09-30'}
    monkeypatch.setattr(pipeline,'chronological_folds',lambda *args:[earlier])
    def trend_fit(*args,**kwargs):
        if missing_trend:raise ValueError('insufficient mature trend classes train [0, 0, 0]')
        return {'manifest_sha256':'trend','sample_counts':{}}
    monkeypatch.setattr(pipeline,'train_trend_head',trend_fit)
    if not missing_trend:monkeypatch.setattr(pipeline,'select_source_policy',lambda *args:{})
    monkeypatch.setattr(pipeline,'select_holding_exit_policy',lambda *args:{})
    order=[]
    # This test owns orchestration, while the reference population's real
    # Broker/labels/assets are verified by test_reference_quality. Keep this
    # adapter explicitly unqualified so it cannot supply core head readiness.
    reference_fixture={'head':'reference_quality','status':'unqualified','bundles':[],
        'qualification':{'status':'pending','passed':False},'required_for_core_selection':False}
    def reference_stage(records,calendar,config,ctx,trends,ma_heads,workers,folds):
        assert records==[record] and ctx is context and workers==1
        assert config['E1_policy_hash'] and folds==[earlier]
        assert ('bootstrap','E1',[]) in order
        assert not any(item[0]=='population' for item in order)
        order.append(('reference_quality',None,[]))
        return copy.deepcopy(reference_fixture)
    monkeypatch.setattr('my_strategy.services.czsc_reference_quality.run_reference_quality',reference_stage)
    def stage(records,calendar,cfg,context,stage,*args,**options):
        population=options.get('reference_population') or {}
        name=population.get('fold',{}).get('name')
        order.append(('population' if population else 'bootstrap',name or stage,
            [bundle['name'] for bundle in population.get('prior_entry_catalog',[])]))
        directory=context.subdir('dataset',options.get('subdirectory',stage))
        if empty_population:
            # Real zero-attempt Broker replay and real empty typed dataset.
            data,summary=build_account_dataset(real_frame,calendar,{}, {},cfg,stage=stage,
                directory=directory/'000001_SZ',population_end=options.get('population_end'),
                reference_population=population,reference_entry_evaluator=(lambda *args:{'status':'model_unavailable'}) if population else None)
        else:
            data={'entries':pd.DataFrame({'label_available':[True,True],'label':[0,1],'label_end':['2025-02-01','2025-04-01']}),
                'exits':pd.DataFrame({'label_available':[True],'label':[0],'exit_advantage':[0.],
                    'label_end':['2025-04-01'],'date':['2025-03-31'],'episode_id':['held'],
                    'stage':[stage],'holding_reference_mode':['final' if population else stage]}),
                'episodes':pd.DataFrame(),'reference_population':population,
                'reference_mode':'final' if population else stage}
            summary={'symbol':record['symbol'],'reference_mode':data['reference_mode'],'reference_population':population}
        pooled={}
        for key in ('entries','exits','episodes'):
            path=directory/(key+'.parquet');data[key].to_parquet(path,index=False)
            pooled[key]={'path':str(path),'sha256':pipeline._file_hash(path)}
        (directory/'manifest.json').write_text(json.dumps({'reference_population':population,
            'stage':stage,'source_snapshot_hash':cfg['source_snapshot_hash'],'records':[summary],
            'failures':[],'pooled_files':pooled}))
        return data,[summary]
    monkeypatch.setattr(pipeline,'build_stage',stage)
    def fit(data,head,fold,calendar,context,cfg,architecture='linear',device='cpu'):
        order.append(('fit',fold['name'],head))
        if head=='entry':
            population=data.attrs['label_contract']['reference_population']
            assert population['fold']==fold and fold['name'] not in [b['name'] for b in population['prior_entry_catalog']]
        return {'name':fold['name'],'available_at':fold['policy_end'],'architecture':architecture,
            'fold':fold,'heads':{head:{'path':'unused','manifest_sha256':fold['name']+head}}}, {
                'manifest_sha256':fold['name']+head,'sample_counts':{}}
    if not empty_population:monkeypatch.setattr(pipeline,'_fit_interval',fit)
    first={'name':'first','train_end':'2024-12-31','validation_start':'2025-01-01','validation_end':'2025-03-31'}
    second={'name':'production','train_end':'2025-06-30','validation_start':'2025-07-01','validation_end':'2025-09-30'}
    report=pipeline._train_one(source_native_run_id='fresh',cpu_workers=1,config={
        'development_windows':[first],'production':second,'architectures':['linear']})
    if empty_population:
        assert report['model_bundles']==[] and report['required_recent_heads_ready'] is False
        assert all('insufficient mature' in row['reason'] for row in report['unavailable_heads'])
        if missing_trend:
            assert report['status']=='completed'
            assert report['trend_bundles']==[] and report['automatic_selection_eligible'] is False
            assert report['source_policy_selection']['status']=='no_qualified_trend_head_use_preregistered_baseline'
            assert report['pipeline']['complete_model_population_DAG'] is True
            assert report['population_folds'] and report['dataset_manifests']
            assert report['required_head_status']['recent_trend_ready'] is False
            assert report['required_head_status']['same_checkpoint_ma_exit_ready'] is False
        assert [item for item in order if item[0]=='population']==[
            ('population','first',[]),('population','production',[])]
        assert len([row for row in report['unavailable_heads'] if row['head']=='entry'])==2
        assert report['pipeline']['final_entry_rows']==0
    else:
        assert [item for item in order if item[0]=='population']==[
            ('population','first',[]),('population','production',['first'])]
        assert order.index(('population','first',[]))<order.index(('fit','first','entry'))
        assert order.index(('fit','first','entry'))<order.index(('population','production',['first']))
        first_population=report['population_folds'][0]
        assert first_population['entries_mature_at_frozen_source']==2
        assert first_population['entries_mature_by_population_end']==1
        assert first_population['entries_positive_at_frozen_source']==1
        assert first_population['entries_positive_by_population_end']==0
    assert len(report['dataset_manifests'])==4
    assert report['pipeline']['self_head_controls_its_training_population'] is False
    assert report['reference_quality']==reference_fixture
    assert report['protocol_freeze']['reference_quality_hash']==stable_hash(reference_fixture)
    assert order.index(('bootstrap','E1',[]))<order.index(('reference_quality',None,[]))<order.index(('population','first',[]))
    assert report['pipeline']['reference_quality_required_for_core_selection'] is False



@pytest.mark.parametrize('compressed',[False,True])
def test_native_snapshot_restores_original_price_basis_and_units(tmp_path,compressed):
    features=tmp_path/'features.parquet';native=tmp_path/('native.json.gz' if compressed else 'native.json')
    pd.DataFrame({'date':['2024-01-01'],'symbol':['000001.SZ'],'source':['Fuyao']}).to_parquet(features,index=False)
    document={'semantics':[True,None,1.5], 'evidence':{'date':'2024-01-01'}}
    content=json.dumps(document).encode('utf-8')
    if compressed:
        import gzip
        native.write_bytes(gzip.compress(content,compresslevel=1,mtime=0))
    else:native.write_bytes(content)
    attrs={'price_basis':'mixed_legacy_adjustment_unverified','legacy_fuyao_bars':1,
        'unverified_trade_price_bars':1,'volume_unit':'shares','amount_unit':'CNY','data_version':'raw-prefix'}
    record={'path':str(features),'sha256':pipeline._file_hash(features),
        'native_path':str(native),'native_sha256':pipeline._file_hash(native),
        'raw_attrs':attrs,'raw_series_hash':'raw-prefix'}
    frame,pack=pipeline.read_snapshot(record)
    assert frame.attrs==attrs and pack['frame'] is frame
    assert {k:v for k,v in pack.items() if k!='frame'}==document
    native.write_bytes(native.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='input drift'):
        pipeline.read_snapshot(record)
    native.write_bytes(gzip.compress(content,compresslevel=1,mtime=0) if compressed else content)
    del record['raw_attrs']
    with pytest.raises(ValueError,match='price-basis'):
        pipeline.read_snapshot(record)



def test_auxiliary_frames_use_same_complete_entry_clone_population(monkeypatch):
    frame,calendar,prepared,cfg=inputs(monkeypatch)
    population=make_reference_population(fold(),[])
    dataset={'entries':pd.DataFrame({'net_return':[.12,-.03,.08],
        'label_available':[True,True,False],'label_end':['2024-06-01','2024-06-02',None]}),
        'reference_population':population}
    returns=training_frame(dataset,'return',cfg,calendar,'E1')
    losses=training_frame(dataset,'loss',cfg,calendar,'E1')
    assert returns.label.iloc[:2].tolist()==[.12,-.03] and pd.isna(returns.label.iloc[2])
    assert losses.label.iloc[:2].tolist()==[0.,1.] and pd.isna(losses.label.iloc[2])
    assert returns.attrs['label_contract']['output_type']=='continuous'
    assert losses.attrs['label_contract']['reference_population']==population



def test_readiness_requires_latest_expected_heads_not_latest_surviving_old_head():
    earlier={'name':'older','policy_end':'2025-03-31'}
    latest={'name':'latest','policy_end':'2025-09-30'}
    def ready(fold,head):
        return {'name':fold['name'],'available_at':fold['policy_end'],'heads':{head:{'path':'qualified'}}}
    production={'name':'production','heads':{'entry':{'path':'qualified'}}}
    status=pipeline.required_head_status([earlier,latest],[ready(earlier,'trend_direction')],
        [ready(earlier,'ma_exit')],[production],'production')
    assert status=={'recent_checkpoint':'latest','recent_trend_ready':False,
        'same_checkpoint_ma_exit_ready':False,'production_entry_ready':True}
    status=pipeline.required_head_status([earlier,latest],[ready(latest,'trend_direction')],
        [ready(earlier,'ma_exit')],[production],'production')
    assert status['recent_trend_ready'] and not status['same_checkpoint_ma_exit_ready']
    status=pipeline.required_head_status([earlier,latest],[ready(latest,'trend_direction')],
        [ready(latest,'ma_exit')],[production],'production')
    assert all(status[key] for key in ('recent_trend_ready','same_checkpoint_ma_exit_ready','production_entry_ready'))



@pytest.mark.parametrize('changed_relative',[
    'services/generator0.py','execution/broker.py','execution/cost_model.py',
    'execution/ledger.py','execution/order.py','services/czsc_backtest.py'])
def test_model_freeze_resolves_installed_code_and_binds_served_catalog(tmp_path,monkeypatch,changed_relative):
    loaded=tmp_path/'installed';(loaded/'services').mkdir(parents=True)
    originals=tmp_path/'old-checkout';(originals/'services').mkdir(parents=True)
    codes={}
    dependencies=[*['services/generator'+str(index)+'.py' for index in range(6)],
        'execution/broker.py','execution/cost_model.py','execution/ledger.py',
        'execution/order.py','services/czsc_backtest.py']
    for relative in dependencies:
        for root in (loaded,originals):
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('same code')
        codes[relative]=pipeline._file_hash(loaded/relative)
    native_codes={}
    for relative in source.NATIVE_GENERATING_FILES:
        for root in (loaded,originals):
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('same native code')
        native_codes[relative]=pipeline._file_hash(loaded/relative)
    monkeypatch.setattr(pipeline,'MY_STRATEGY_ROOT',loaded)
    monkeypatch.setattr(pipeline,'GENERATING_FILES',tuple(codes))
    policy=strategy_config()['liquidity_universe']
    native_source={'records_hash':stable_hash([]),'source_snapshot':{'code_hashes':native_codes},
        'population_mode':'full_liquidity_union','liquidity':{'policy':policy,'policy_hash':stable_hash(policy),
            'selected_symbols':[],'daily_membership_hash':'daily','population_hash':'population'}}
    native_source['snapshot_hash']=stable_hash(native_source)
    config={**strategy_config(),'source_snapshot_hash':native_source['snapshot_hash']}
    reference_fixture={'head':'reference_quality','status':'unqualified',
        'required_for_core_selection':False,'test_fixture':'source_identity_boundary_only'}
    freeze={'config_hash':stable_hash(config),'code_hashes':codes,
        'models_catalog_hash':stable_hash({'trend':[],'ma_exit':[],'architecture':[]}),
        'population_folds_hash':stable_hash([]),'dataset_manifests_hash':stable_hash([]),
        'holding_population_hash':stable_hash(None),'holding_population_diagnostics_hash':stable_hash(None),
        'reference_quality_hash':stable_hash(reference_fixture)}
    freeze['freeze_hash']=stable_hash(freeze)
    report={'strategy_version':STRATEGY_ID,'config':config,'source_snapshot':native_source,
        'dataset_records':[],'validation_records':[],'population_folds':[],'dataset_manifests':[],
        'trend_bundles':[],'ma_exit_bundles':[],'architecture_bundles':[],'model_bundles':[],
        'protocol_freeze':freeze,'reference_quality':copy.deepcopy(reference_fixture)}
    monkeypatch.setattr(pipeline,'audit_holding_population',lambda *args:None)
    audits=[]
    def reference_audit(root,reference,cfg,symbols,*,expected_records):
        assert root==tmp_path and reference==reference_fixture and cfg==config
        assert symbols==expected_records==[]
        audits.append(reference)
    monkeypatch.setattr('my_strategy.services.czsc_reference_quality.audit_reference_quality',reference_audit)
    pipeline._verify_run(report,tmp_path)
    assert len(audits)==1
    changed=copy.deepcopy(report);changed['reference_quality']['status']='qualified'
    with pytest.raises(ValueError,match='reference quality population/catalogue drift'):
        pipeline._verify_run(changed,tmp_path)
    changed=copy.deepcopy(report);changed['model_bundles']=[bundle()]
    with pytest.raises(ValueError,match='served L2'):
        pipeline._verify_run(changed,tmp_path)
    (loaded/changed_relative).write_text('installed code changed')
    assert (originals/changed_relative).read_text()=='same code'
    with pytest.raises(ValueError,match='changed|drift'):
        pipeline._verify_run(report,tmp_path)



def pinned_fixture(monkeypatch,tmp_path,signal='2025-04-01'):
    cfg=strategy_config();cfg.update(E0_policy_hash='e0',E1_policy_hash='e1',candidate_policy_hash='c')
    dates=['2025-04-02','2025-04-04','2025-04-07']
    frame=pd.DataFrame({'date':dates,'symbol':['TEST.SZ']*3,'close':[10.]*3,'model_input_eligible':[True]*3})
    monkeypatch.setattr(pipeline,'bind_trend_inputs',lambda frame,calendar,config=None:frame.copy())
    monkeypatch.setattr(pipeline,'holding_feature_row',lambda feature,state,**kwargs:{'actual_shares':state['shares']})
    bundles=[{'name':name,'available_at':available,'heads':{'ma_exit':{'path':name}}}
        for name,available in [('prior','2025-03-31'),('newer','2025-04-03')]]
    manifests={bundle['name']:{'head':'ma_exit','manifest_sha256':bundle['name'],
        'target_contract_hash':'target','feature_schema_hash':'schema','train_label_end':'2025-02-28',
        'calibration_label_end':'2025-03-15','policy_label_end':bundle['available_at'],
        'fold':{'policy_end':bundle['available_at']},'target_contract':{'stage':'E0','target':'actual_held',
            'source_snapshot_hash':'source','split_policy':cfg['holding_split_policy'],
            'economic_weighting':cfg['holding_economic_weighting']}}
        for bundle in bundles}
    monkeypatch.setattr(pipeline,'load_head',lambda path:manifests[path.name])
    calls=[]
    def predict(values,root,bundle,*args,**kwargs):
        calls.append((bundle['name'],values['actual_shares']))
        return .7,manifests[bundle['name']]
    monkeypatch.setattr(pipeline,'_predict_interval',predict)
    # The actual B-only plan is absent from the unfiltered A+B preparation.
    prepared={'config_hash':'prepared','rows':[{'date':signal,'entry_plan':{'plan_id':'unfiltered-A+B'}}]}
    plan={'plan_id':'accepted-B-only','symbol':'TEST.SZ','signal_date':signal,
        'available_at':signal+'T15:00:00+08:00','stop_price':9.,'original_opportunity_ids':['B'],
        'holding_boundaries':[{'first_available_at':'2025-03-01T15:00:00+08:00'}]}
    snapshot={'plan_id':plan['plan_id'],'symbol':'TEST.SZ','date':dates[0],
        'entry_date':dates[0],'entry_signal_date':signal,'entry_plan':plan,'shares':1000,
        'holding_prefix':{'peak':10.,'trough':10.},'exit_model_binding':None}
    callback=pipeline.pinned_exit_policy(frame,dates,prepared,tmp_path,bundles,cfg)
    return callback,snapshot,calls,bundles



def test_filtered_actual_plan_is_pinned_at_accepted_signal_and_survives_partial_clone(monkeypatch,tmp_path):
    callback,snapshot,calls,_=pinned_fixture(monkeypatch,tmp_path)
    first=callback(snapshot)
    binding=first['entry_exit_binding']
    assert binding['plan_id']=='accepted-B-only' and binding['candidate_signal_date']=='2025-04-01'
    assert binding['checkpoint']=='prior' and binding['available_at']=='2025-03-31'
    assert binding['binding_source']=='actual_Broker_accepted_entry_plan'
    later={**copy.deepcopy(snapshot),'date':'2025-04-04','exit_model_binding':binding,'shares':800}
    assert callback(later)['entry_exit_binding']==binding
    cloned={**later,'date':'2025-04-07','shares':300}
    assert callback.new_replay()(cloned)['entry_exit_binding']==binding
    assert calls==[('prior',1000),('prior',800),('prior',300)]



def test_no_prior_exit_head_remains_rule_fallback_for_entire_holding_and_clone(monkeypatch,tmp_path):
    callback,snapshot,calls,_=pinned_fixture(monkeypatch,tmp_path,signal='2025-03-28')
    first=callback(snapshot);binding=first['entry_exit_binding']
    assert binding['checkpoint'] is None and first['probability'] is None and not first['apply']
    assert binding['mode']=='explicit_rule_E0_no_prior_exit_head'
    later={**copy.deepcopy(snapshot),'date':'2025-04-07','exit_model_binding':binding,'shares':500}
    assert callback.new_replay()(later)['entry_exit_binding']==binding
    assert calls==[]



@pytest.mark.parametrize('field,value',[('plan_id','other-plan'),('candidate_signal_date','2025-04-04'),
    ('checkpoint','newer'),('available_at','2025-04-03'),('mode','upgrade'),('source','other')])
def test_saved_actual_exit_binding_cannot_change_any_frozen_anchor(monkeypatch,tmp_path,field,value):
    callback,snapshot,calls,_=pinned_fixture(monkeypatch,tmp_path)
    binding=callback(snapshot)['entry_exit_binding'];binding[field]=value
    later={**copy.deepcopy(snapshot),'date':'2025-04-04','exit_model_binding':binding}
    with pytest.raises(ValueError,match='original exit binding'):
        callback.new_replay()(later)



def test_snapshot_must_retain_exact_actual_entry_plan_and_signal(monkeypatch,tmp_path):
    callback,snapshot,calls,_=pinned_fixture(monkeypatch,tmp_path)
    callback(snapshot)
    later={**copy.deepcopy(snapshot),'date':'2025-04-04'}
    later['entry_plan']['stop_price']=8.
    with pytest.raises(ValueError,match='original accepted plan'):
        callback.new_replay()(later)
    later=copy.deepcopy(snapshot);later['entry_plan']['signal_date']='2025-03-28'
    with pytest.raises(ValueError,match='original signal/plan identity'):
        callback.new_replay()(later)



def _frame(n=90):
    dates=pd.bdate_range('2025-01-01',periods=n).strftime('%Y-%m-%d').tolist()
    f=pd.DataFrame({'date':dates,'symbol':'TEST.SZ','open':100.,'high':101.,'low':99.,'close':100.,
                    'volume':4000000.,'amount':400000000.,
                    'source':'sina_unadjusted_v1','has_trade_price':1,'model_input_eligible':True,
                    'wyckoff_corporate_action_suspected':False})
    for c in MA_TREND_COLUMNS:f[c]=.1
    f.attrs.update(volume_unit='shares',amount_unit='CNY')
    return f,dates



def _labels(f,calendar):return build_trend_labels(f,calendar,source_snapshot_hash='snapshot',pipeline_contract_hash='pipeline')



def test_horizon_includes_reference_entry_and_missing_quotes_are_not_neutral():
    f,cal=_frame();f.loc[80,'close']=110.;data=_labels(f,cal)
    row=data.iloc[70];assert row.label_start==cal[71] and row.label_end==cal[80]
    assert row.label_class=='UP' and row.label==CLASSES.index('UP')
    missing=_labels(f.drop(index=71).reset_index(drop=True),cal)
    assert not missing.loc[missing.date==cal[70],'label_available'].iloc[0]
    assert pd.isna(missing.loc[missing.date==cal[70],'label_class'].iloc[0])



def test_future_labels_do_not_change_causal_input_or_start_band():
    f,cal=_frame();a=_labels(f,cal);f.loc[71:,'high']=999.;f.loc[80,'close']=80.;b=_labels(f,cal)
    assert a.iloc[70].trend_band_log==b.iloc[70].trend_band_log
    assert np.array_equal(a.loc[70,list(MA_TREND_COLUMNS)],b.loc[70,list(MA_TREND_COLUMNS)])
    assert b.iloc[70].label_class=='DOWN'



def test_horizon_tail_and_corporate_action_stay_unavailable():
    f,cal=_frame();f.loc[77,'wyckoff_corporate_action_suspected']=True;data=_labels(f,cal)
    assert data.iloc[70].label_reason=='future_price_basis_or_corporate_action_ambiguous'
    clean,calendar=_frame();assert _labels(clean,calendar).iloc[-1].label_reason=='horizon_not_mature'



def test_information_intervals_are_purged_without_transitive_stock_group():
    f,cal=_frame(200);data=_labels(f,cal)
    fold={'train_end':cal[100],'calibration_start':cal[101],'calibration_end':cal[139],
          'policy_start':cal[140],'policy_end':cal[199]}
    train,calibration,policy=trend_time_masks(data,fold)
    assert train[80] and not train[95]
    assert calibration[120] and not calibration[135]
    assert policy[170]
    assert not (train&calibration).any() and not (calibration&policy).any()



def test_old_profit_target_is_not_new_three_class_identity():
    contract=trend_contract(source_snapshot_hash='s',pipeline_contract_hash='p')
    assert contract['target']=='next_verified_open_horizon_direction'
    assert contract['classes']==['UP','DOWN','NEUTRAL']
    assert 'profit' not in contract['target']



def test_three_class_fit_prior_identity_and_production_clock(tmp_path):
    from my_strategy.services.czsc_trend_model import train_trend_head,trend_contract,TrendSession,SCHEMA_HASH
    from my_strategy.services.czsc_market_profiles import MA_TREND_COLUMNS
    from my_strategy.core.run_context import stable_hash
    import numpy as np
    dates=pd.bdate_range('2023-01-02',periods=190)
    frame=pd.DataFrame({k:np.sin(np.arange(190)/7+i) for i,k in enumerate(MA_TREND_COLUMNS)})
    frame['date']=dates.strftime('%Y-%m-%d');frame['symbol']='TEST';frame['label']=np.arange(190)%3
    frame['label_end']=(dates+pd.offsets.BDay(1)).strftime('%Y-%m-%d');frame['label_available']=True
    frame['input_eligible']=True;frame['model_input_eligible']=True
    frame['observation_id']=[stable_hash({'day':d}) for d in frame.date]
    config={'trend':{'min_train_per_class':2,'min_calibration_per_class':2,'min_policy_per_class':2}}
    contract=trend_contract(config,source_snapshot_hash='synthetic-unit',pipeline_contract_hash='unit-protocol')
    frame.attrs['label_contract']=contract
    fold={'name':'unit','train_end':frame.date.iloc[89],'calibration_start':frame.date.iloc[91],
        'calibration_end':frame.date.iloc[119],'policy_start':frame.date.iloc[121],'policy_end':frame.date.iloc[149]}
    manifest=train_trend_head(frame,tmp_path/'model',fold,config,contract=contract)
    assert manifest['classes']==['UP','DOWN','NEUTRAL']
    assert manifest['train_label_end']<=fold['train_end']
    assert manifest['calibration_label_end']<=fold['calibration_end']
    assert manifest['preprocess']['median']==pytest.approx(frame.loc[:88,list(MA_TREND_COLUMNS)].median().tolist())
    later=frame.iloc[151:154].copy();later.attrs['schema_hash']=SCHEMA_HASH
    p,_=TrendSession().predict(later,tmp_path/'model',expected_contract=contract)
    assert p.shape==(3,3) and np.allclose(p.sum(axis=1),1)
    assert np.all(p[:,2]>0) and not np.allclose(p[:,1],1-p[:,0])
    with pytest.raises(ValueError,match='not actually trained'):
        TrendSession().predict(later,tmp_path/'model',expected_contract=contract,mode='production')
    early=frame.iloc[120:121].copy();early.attrs['schema_hash']=SCHEMA_HASH
    with pytest.raises(ValueError,match='not available'):
        TrendSession().predict(early,tmp_path/'model',expected_contract=contract)



def test_independent_MA_dependency_requires_complete_verified_65_raw_sessions():
    from my_strategy.services.czsc_trend_model import bind_trend_inputs
    frame,calendar=_frame(200)
    full=bind_trend_inputs(frame,calendar)
    assert not full.iloc[63].model_input_eligible and full.iloc[64].model_input_eligible
    missing=bind_trend_inputs(frame.drop(index=100).reset_index(drop=True),calendar)
    assert not missing.loc[missing.date==calendar[150],'model_input_eligible'].iloc[0]
    assert missing.loc[missing.date==calendar[166],'model_input_eligible'].iloc[0]
    prefix=bind_trend_inputs(frame.iloc[:130],calendar)
    assert prefix.model_input_eligible.tolist()==full.iloc[:130].model_input_eligible.tolist()



def test_historical_pool_keeps_former_and_later_members_without_pre2023_or_latest_list(tmp_path,monkeypatch):
    db=tmp_path/'raw.db'
    dates=pd.bdate_range('2025-01-02',periods=70).strftime('%Y-%m-%d').tolist()
    monkeypatch.setattr('my_strategy.services.czsc_trend_support._calendar',lambda _: (dates,'verified-fixture'))
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,amount REAL,source TEXT,has_trade_price INTEGER)')
        rows=[]
        for symbol in ('000001.SZ','600519.SH','002436.SZ','300001.SZ'):
            for index,date in enumerate(dates):
                if symbol=='600519.SH' and index<40:continue
                amount=40_000_000. if symbol!='000001.SZ' or index<35 else 10_000_000.
                origin='legacy_unverified' if symbol=='002436.SZ' else 'sina_unadjusted_v1'
                rows.append((symbol,date,10.,10.2,9.8,10.,amount/10.,amount,origin,1))
        conn.executemany('INSERT INTO stock_daily_normalized VALUES(?,?,?,?,?,?,?,?,?,?)',rows)
    population=source.build_liquidity_universe(feature_start=dates[0],end=dates[-1],db_path=db)
    assert population['population_mode']=='full_liquidity_union'
    assert population['selected_symbols']==['000001.SZ','600519.SH']
    assert population['latest_symbols']==['600519.SH']
    assert population['candidate_symbols']==['000001.SZ','002436.SZ','600519.SH']
    assert population['unsupported_stock_count']==1
    assert not hasattr(source,'select_cohort')



@pytest.mark.parametrize('changes', [
    {'full_qualified_pool': False},
    {'liquidity_universe': {'lookback_sessions': 19}},
    {'liquidity_universe': {'min_median_amount': 29999999.}},
    {'liquidity_universe': {'complete_verified_window': False}},
])
def test_public_full_training_cannot_weaken_liquidity_or_population(changes):
    from my_strategy.services.czsc_trend_contracts import validate_training_config
    with pytest.raises(ValueError):
        validate_training_config(changes)


def _full_population_source(tmp_path, count=721):
    records = [{'symbol': f'{index:06d}.SZ'} for index in range(count)]
    policy = strategy_config()['liquidity_universe']
    report = {'strategy_version': STRATEGY_ID, 'status':'completed', 'raw_only_rebuild': True,
        'dataset_records': records, 'population_mode': 'full_liquidity_union',
        'liquidity': {'policy': policy, 'policy_hash': stable_hash(policy),
            'selected_symbols': [row['symbol'] for row in records],
            'daily_membership_hash': 'independently-frozen-days', 'population_hash': 'complete-union'}}
    directory = tmp_path / 'full-source' / 'reports'; directory.mkdir(parents=True)
    (directory / 'native_source.json').write_text(json.dumps(report))
    return records


def test_full_training_does_not_stop_at_small_ready_heads_or_processing_batch(tmp_path, monkeypatch):
    records = _full_population_source(tmp_path)
    monkeypatch.setattr(pipeline, 'ARTIFACT_RUNS_ROOT', tmp_path)
    calls = []
    def fit(**kwargs):
        calls.append(kwargs)
        return {'run_id': 'all-stocks', 'status': 'completed',
            'required_recent_heads_ready': True, 'unavailable_heads': [], 'required_head_status': {}}
    monkeypatch.setattr(pipeline, '_train_one', fit)
    report = train_trend_consensus(source_native_run_id='full-source', config={'stock_batch_size': 360,
        'training_capacity_tiers': [60, 180, 360]}, cpu_workers=1)
    assert len(calls) == 1
    assert calls[0]['native_records'] == records
    assert calls[0]['symbols'] == [row['symbol'] for row in records]
    assert calls[0]['config']['training_capacity_tiers'] == [721]
    assert calls[0]['config']['capacity_expansion_criterion'] == pipeline.FULL_CAPACITY_CRITERION
    assert calls[0]['config']['epochs'] == 20
    assert report['capacity_status'] == 'full_population_completed'
    manifest = json.loads((tmp_path / 'all-stocks/validation/capacity_attempts.json').read_text())
    assert manifest['readiness_early_stop'] is False
    assert manifest['processing_batch_size'] == 360 and manifest['preregistered_tiers'] == [721]


def test_explicit_subset_cannot_claim_the_complete_frozen_population(tmp_path, monkeypatch):
    records = _full_population_source(tmp_path)
    monkeypatch.setattr(pipeline, 'ARTIFACT_RUNS_ROOT', tmp_path)
    monkeypatch.setattr(pipeline, '_train_one', lambda **kwargs: pytest.fail('subset cannot start'))
    with pytest.raises(ValueError, match='complete frozen'):
        train_trend_consensus(source_native_run_id='full-source', symbols=[r['symbol'] for r in records[:60]])


def test_resource_queue_is_bounded_without_losing_the_stock_union(monkeypatch):
    import concurrent.futures
    counts = {'pending': 0, 'maximum': 0}
    class Future:
        def __init__(self, worker, payload):
            self.worker, self.payload = worker, payload
        def result(self):
            counts['pending'] -= 1
            return self.worker(self.payload)
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, worker, payload):
            counts['pending'] += 1
            counts['maximum'] = max(counts['maximum'], counts['pending'])
            return Future(worker, payload)
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', Pool)
    result = list(pipeline._bounded_results(abs, (-x for x in range(1024)), 6, 3))
    assert result == list(range(1024))
    assert counts == {'pending': 0, 'maximum': 3}


def test_streamed_stock_pool_preserves_rows_columns_and_unknown_values(tmp_path):
    inputs = [pd.DataFrame(), pd.DataFrame({'id': ['first'], 'label': [None], 'value': [1]}),
        pd.DataFrame({'id': ['second'], 'label': [1.], 'value': [2.], 'extra': ['observed']})]
    paths = []
    for index, frame in enumerate(inputs):
        path = tmp_path / f'{index}.parquet'; frame.to_parquet(path, index=False); paths.append(path)
    actual = pipeline._pool_parquets(paths, tmp_path / 'all.parquet')
    assert actual.id.tolist() == ['first', 'second']
    assert pd.isna(actual.loc[0, 'label']) and actual.loc[1, 'label'] == 1.
    assert pd.isna(actual.loc[0, 'extra']) and actual.loc[1, 'extra'] == 'observed'
    assert actual.value.tolist() == [1., 2.]


def test_liquidity_block_preserves_raw_risk_attempts_and_unknown_labels(monkeypatch):
    frame, calendar, prepared, config = inputs(monkeypatch)
    frame['amount'] = 10_000_000.
    frame['volume'] = 1_000_000.
    dataset, summary = build_account_dataset(frame, calendar, {}, {}, config, stage='E0')
    attempts = dataset['entries']
    assert len(attempts) == 3 and summary['counts']['raw_risk_opportunities'] == 3
    assert summary['counts']['liquidity_rejected_attempts'] == 3
    assert not attempts.input_eligible.any() and not attempts.label_available.any()
    assert attempts.label.isna().all() and attempts.label_reason.eq('entry_unfilled').all()
    assert attempts.reference_gate_reason.eq('historical_liquidity_ineligible').all()
    assert summary['reference_metrics']['trade_count'] == 0
    account = replay_trend_account(frame, calendar, prepared, config=config, start=calendar[60], mode='E0')
    assert account['ledger'] == [] and account['final_action_intents'] == []
    assert all(row['risk_qualified'] and not row['liquidity_qualified'] for row in account['entry_diagnostics'])
    assert account['terminal_state']['reserved_opportunity_ids'] == []


def test_liquidity_pool_exit_does_not_cancel_a_pinned_holding_exit():
    frame, calendar, pack, trends = fixture(100, sell_from=None)
    frame.loc[61:, ['amount', 'volume']] = [10_000_000., 1_000_000.]
    prepared = prepare(frame, calendar, pack, trends)
    seen = []
    def exit_policy(snapshot):
        seen.append(snapshot['date'])
        return {'date': snapshot['date'], 'available_at': snapshot['available_at'], 'status': 'scored',
            'probability': .9 if snapshot['date'] == calendar[80] else .1, 'threshold': .55,
            'head':'holding_exit','apply':True,'request_exit':snapshot['date']==calendar[80],
            'holding_contract_compatible':True,'contract_hash':'frozen-exit-target','usage_mode':'historical',
            'model_identity': identity('ma_exit'),'entry_exit_binding':{'checkpoint':'strict-prior',
                'plan_id':snapshot['plan_id'],'candidate_signal_date':snapshot['entry_signal_date']}}
    account = replay_trend_account(frame, calendar, prepared, start=calendar[60], mode='E1', exit_policy=exit_policy)
    assert [fill['action'] for fill in account['ledger']] == ['BUY', 'SELL']
    assert account['ledger'][0]['date'] == calendar[61]
    assert account['ledger'][1]['signal_date'] == calendar[80]
    assert account['ledger'][1]['date'] == calendar[81]
    assert account['ledger'][1]['reason'] == 'signal_pinned_ma_exit'
    assert calendar[80] in seen and account['terminal_state']['shares'] == 0
    from my_strategy.services.czsc_liquidity import daily_eligibility
    membership = daily_eligibility(frame, calendar)
    assert not membership.loc[membership.date == calendar[80], 'liquidity_pool_eligible'].iloc[0]


def test_pool_mask_keeps_holding_inputs_but_blocks_trend_labels_and_forecasts():
    from my_strategy.services.czsc_trend_model import bind_trend_inputs
    frame, calendar = _frame(100)
    frame.loc[65:, 'amount'] = 1_000_000.
    bound = bind_trend_inputs(frame, calendar)
    assert bound.iloc[90].model_input_eligible
    assert not bound.iloc[90].trend_input_eligible
    labels = _labels(frame, calendar)
    assert labels.iloc[85].label_reason == 'historical_liquidity_ineligible'
    assert not labels.iloc[85].label_available and pd.isna(labels.iloc[85].label)
    prefix = bind_trend_inputs(frame.iloc[:90], calendar)
    pd.testing.assert_series_equal(prefix.trend_input_eligible, bound.iloc[:90].trend_input_eligible)


def test_forced_clone_uses_signal_liquidity_not_next_open_volume():
    frame, calendar, pack, trends = fixture(100, sell_from=None)
    frame.loc[:60, ['amount', 'volume']] = [10_000_000., 1_000_000.]
    frame.loc[61:, ['amount', 'volume']] = [1_000_000_000., 100_000_000.]
    prepared = prepare(frame, calendar, pack, trends)
    reference = replay_trend_account(frame, calendar, prepared, start=calendar[60], end=calendar[60], mode='E0')
    diagnostic = reference['entry_diagnostics'][0]
    assert diagnostic['risk_qualified'] and not diagnostic['liquidity_qualified']
    clone = clone_candidate_account(frame, calendar, prepared, diagnostic['pre_gate_state'], diagnostic['candidate'])
    assert clone['ledger'] == [] and clone['metrics']['open_shares'] == 0
    assert clone['rejections'][0]['reason'] == 'historical_liquidity_ineligible'


def test_fast_direction_economic_weights_equal_causal_interval_reference():
    from my_strategy.services.czsc_trend_model import _concurrency_weights
    frame = pd.DataFrame({'symbol': ['one'] * 4 + ['two'] * 2,
        'date': ['2025-01-02', '2025-01-03', '2025-01-06', '2025-01-07', '2025-01-03', '2025-01-07'],
        'label_end': ['2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-07', '2025-01-09']})
    mask = np.array([True, False, True, True, True, True])
    reference = {}
    for _, group in frame.loc[mask].groupby('symbol', sort=False):
        days = sorted(set(group.date.tolist() + group.label_end.tolist())); counts = np.zeros(len(days))
        positions = {day: index for index, day in enumerate(days)}
        for row in group.itertuples(): counts[positions[row.date]:positions[row.label_end]+1] += 1
        for row in group.itertuples(): reference[row.Index] = np.mean(1 / counts[positions[row.date]:positions[row.label_end]+1])
    expected = pd.Series(reference).reindex(frame.loc[mask].index).to_numpy(float)
    expected = expected / expected.mean()
    assert _concurrency_weights(frame, mask) == pytest.approx(expected, abs=1e-12)


def test_source_generator_closure_cannot_omit_liquidity_implementation(tmp_path,monkeypatch):
    hashes={}
    for relative in source.NATIVE_GENERATING_FILES:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('exact frozen source')
        hashes[relative]=pipeline._file_hash(path)
    monkeypatch.setattr(pipeline,'MY_STRATEGY_ROOT',tmp_path)
    pipeline.verify_source_generators({'code_hashes':hashes})
    del hashes['services/czsc_liquidity.py']
    with pytest.raises(ValueError,match='source identity incomplete'):
        pipeline.verify_source_generators({'code_hashes':hashes})


def test_stock_absent_from_policy_window_remains_real_Broker_cash_in_each_variant(monkeypatch):
    frame,calendar,_,_=fixture()
    variants=[{'cross':.55,'buy_base':.35,'sell':.4,'margin':.05},
        {'cross':.65,'buy_base':.45,'sell':.5,'margin':.15}]
    bundle={'fold':{'name':'strict-prior-policy','policy_start':'2025-01-02','policy_end':'2025-02-28'}}
    monkeypatch.setattr(pipeline,'read_snapshot',lambda record:(frame,{}))
    accounts=pipeline._policy_worker(({'symbol':SYMBOL},calendar,strategy_config(),'.',bundle,variants))
    assert len(accounts)==2 and [row['variant'] for row in accounts]==variants
    for account in accounts:
        assert account['ledger']==[] and account['ledger_hash']==stable_hash([])
        assert account['metrics']['final_equity']==100000.
        assert account['metrics']['total_return']==0. and account['metrics']['open_shares']==0
        assert account['status']=='cash_no_stock_observation_in_policy_window'


def test_streamed_completely_empty_stock_pool_is_a_valid_zero_population(tmp_path):
    paths=[]
    for index in range(3):
        path=tmp_path/f'empty-{index}.parquet';pd.DataFrame().to_parquet(path,index=False);paths.append(path)
    actual=pipeline._pool_parquets(paths,tmp_path/'zero.parquet')
    assert actual.empty and (tmp_path/'zero.parquet').is_file()
