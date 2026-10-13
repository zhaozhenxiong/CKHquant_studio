"""New daily-attempt heads cannot reuse frozen v1 population or schemas."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_trend_v2_contracts import (STRATEGY_ID, TRAINING_POPULATION,
    SPLIT_POLICY, ECONOMIC_WEIGHTING, HELD_SPLIT_POLICY, HELD_ECONOMIC_WEIGHTING,
    head_dependency_policy, strategy_config)
from my_strategy.services.czsc_trend_v2_features import ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS, L1_MA_EXIT_COLUMNS
from my_strategy.services.czsc_trend_v2_labels import target_contract
from my_strategy.services.czsc_trend_v2_interval_model import (
    HeadSession, _group_ids, _validate_contract, _weights, bind_inputs, load_head,
    schema_hash, time_masks, train_head,
)


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


@pytest.mark.parametrize('change', [
    {'version': 'czsc_trend_consensus'},
    {'consensus': {'version': 'czsc_trend_consensus'}},
    {'training_population': 'actual_flat_candidate_clone'},
    {'reassessment': {'quality_rejection_consumes': True}},
    {'reassessment': {'holding_sell': 'fresh_sell_event_only'}},
])
def test_v2_configuration_rejects_old_or_partially_relabelled_policy(change):
    with pytest.raises(ValueError):
        strategy_config(change)


def test_identical_columns_have_distinct_feature_and_model_family_schemas():
    from my_strategy.services import czsc_trend_features as old_features
    from my_strategy.services import czsc_trend_interval_model as old_interval
    from my_strategy.services import czsc_trend_model as old_trend
    from my_strategy.services import czsc_trend_v2_features as new_features
    from my_strategy.services import czsc_trend_v2_model as new_trend
    assert ENTRY_FEATURE_COLUMNS == old_features.ENTRY_FEATURE_COLUMNS
    for name in ('ENTRY_SCHEMA_HASH', 'EXIT_SCHEMA_HASH', 'MA_ENTRY_SCHEMA_HASH', 'MA_EXIT_SCHEMA_HASH'):
        assert getattr(new_features, name) != getattr(old_features, name)
    for head, columns in [('entry', ENTRY_FEATURE_COLUMNS), ('exit', EXIT_FEATURE_COLUMNS), ('ma_exit', L1_MA_EXIT_COLUMNS)]:
        assert schema_hash(columns, head) != old_interval.schema_hash(columns, head)
    assert new_trend.SCHEMA_HASH != old_trend.SCHEMA_HASH
    assert new_trend.trend_contract(source_snapshot_hash='s', pipeline_contract_hash='p')['family'] == STRATEGY_ID
    assert 'source_training_run_id' not in strategy_config()


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
    from my_strategy.services.czsc_trend_v2_model import load_trend_head
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


def test_real_v2_three_class_head_keeps_own_family_and_actual_completion_clock(tmp_path):
    from my_strategy.services.czsc_trend_v2_model import (SCHEMA_HASH, TrendSession,
        train_trend_head, trend_contract, load_trend_head)
    frame, _, fold, cfg, _ = entry_fixture()
    frame['label'] = np.arange(len(frame)) % 3
    frame['observation_id'] = [stable_hash({'symbol': 'TEST.SZ', 'date': day}) for day in frame.date]
    frame['model_input_eligible'] = True
    cfg['trend'].update(min_train_per_class=2, min_calibration_per_class=2, min_policy_per_class=2)
    contract = trend_contract(cfg, source_snapshot_hash='frozen-source', pipeline_contract_hash='v2-pipeline')
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
