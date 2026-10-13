"""UI composition exports actual L2 rejection and retry identity unchanged."""
import copy

import pandas as pd
import pytest

from my_strategy.services.czsc_trend_v2_contracts import STRATEGY_ID, TRAINING_POPULATION, strategy_config
from my_strategy.services import czsc_trend_v2_runtime as runtime


def test_daily_attempts_keep_original_family_and_actual_rejection_in_runtime(monkeypatch, tmp_path):
    from my_strategy.services import czsc_trend_v2_pipeline as pipeline
    from my_strategy.services import czsc_trend_v2_features as features
    from my_strategy.services import czsc_trend_v2_consensus as consensus
    from my_strategy.services import czsc_trend_v2_account as account
    from my_strategy.services import czsc_trend_v2_model as trend_model
    from my_strategy.services import czsc_native_evidence_v4 as native

    cfg = strategy_config()
    report = {'config': cfg, 'trend_bundles': [], 'model_bundles': [], 'ma_exit_bundles': []}
    monkeypatch.setattr(runtime, 'load_model_run', lambda _: (tmp_path, report))
    seen = {}

    def native_pack(frame, *args, **kwargs):
        seen['native_dates'] = frame.date.tolist()
        return {'frame': frame.copy(), 'wyckoff_events': []}

    def decisions(frame, *args, **kwargs):
        return {'events': [], 'rows': [{'date': bar['date'], 'symbol': bar['symbol'],
            'reference_price': bar['close'], 'buy_base_score': .4, 'sell_base_score': .1,
            'action': 'BUY', 'candidate_evidence': [], 'agents': {
                'ma': {'status': 'unavailable'},
                'structure': {'status': 'scored', 'buy_strength': .8, 'sell_strength': 0.},
                'wyckoff': {'status': 'scored', 'buy_strength': .8, 'sell_strength': .1}}}
            for bar in frame.to_dict('records')]}

    def actual_account(frame, calendar, prepared, **kwargs):
        rows, attempts = [], []
        for day in frame.date:
            candidate = {'entry_plan': {}, 'attempt_id': 'attempt:' + day,
                'economic_group_id': 'original-family', 'original_opportunity_ids': ['structure:original:BUY'],
                'reference_state_hash': 'actual-audit:' + day, 'flat_state_hash': 'actual-flat:' + day,
                'entry_plan_hash': 'frozen-plan:' + day}
            proposed = kwargs['entry_evaluator']({'date': day, 'shares': 0}, candidate)
            assert proposed['evaluation_status'] == 'scored_passed'
            actual = {**proposed, 'evaluation_status': 'scored_rejected', 'qualified': False,
                'reason_codes': ['buy_total_support_below_threshold']}
            rows.append({'date': day, 'action': 'HOLD', 'emitted': False, 'shares': 0,
                'entry_evaluation': actual, 'final_action_intents': []})
            attempts.append({'date': day, **candidate, 'evaluation': copy.deepcopy(actual),
                'allowed': False, 'reason': 'buy_total_support_below_threshold'})
        return {'rows': rows, 'entry_diagnostics': attempts, 'holding_snapshots': [],
            'final_action_intents': [], 'series_context': {'account_id': kwargs['account_id']}}

    monkeypatch.setattr(native, 'build_native_evidence_v4', native_pack)
    monkeypatch.setattr(trend_model, 'bind_trend_inputs', lambda frame, calendar: frame.copy())
    monkeypatch.setattr(pipeline, 'forward_trend', lambda frame, *a, **kw: {day: {
        'model_available': False, 'input_eligible': True, 'p_up': None, 'p_down': None,
        'p_neutral': None, 'reason_codes': ['no_prior_trend_head']} for day in frame.date})
    monkeypatch.setattr(pipeline, 'predict_entry', lambda *a, **kw: (.9, {'family': STRATEGY_ID, 'head': 'entry'}, 'scored'))
    monkeypatch.setattr(pipeline, 'pinned_exit_policy', lambda *a, **kw: None)
    monkeypatch.setattr(consensus, 'prepare_trend_decisions', decisions)
    monkeypatch.setattr(features, 'event_prefix_rows', lambda frame, *a: [{} for _ in frame.date])
    monkeypatch.setattr(account, 'replay_trend_account', actual_account)

    frame = pd.DataFrame({'date': ['2026-01-02', '2026-01-05', '2026-01-06'],
        'symbol': ['TEST.SZ'] * 3, 'close': [20., 21., 999.]})
    result = runtime.runtime_overlay(frame, frame.date.tolist(), as_of='2026-01-05')
    assert seen['native_dates'] == ['2026-01-02', '2026-01-05']
    assert result['strategy_version'] == STRATEGY_ID
    assert result['latest_decision']['action'] == 'HOLD'
    assert result['trend_consensus']['direction_points'] == []
    assert result['account_actions']['intents'] == []
    one, two = result['rows']
    assert one['entry_attempt']['attempt_id'] != two['entry_attempt']['attempt_id']
    assert one['entry_attempt']['economic_group_id'] == two['entry_attempt']['economic_group_id']
    for row in result['rows']:
        quality = row['agents']['high_return']
        assert quality['probability'] == .9
        assert quality['evaluation_status'] == 'scored_rejected'
        assert quality['training_population'] == TRAINING_POPULATION
        assert row['entry_attempt']['flat_state_hash'] == 'actual-flat:' + row['date']
        assert row['model_resolution']['reason_codes'] == ['buy_total_support_below_threshold']
        assert row['model_resolution']['original_opportunity_ids'] == ['structure:original:BUY']
        assert row['account_action']['reason_codes'] == ['buy_total_support_below_threshold']


def test_old_run_config_cannot_be_relabelled_by_v2_runtime(monkeypatch, tmp_path):
    cfg = strategy_config()
    cfg['version'] = 'czsc_trend_consensus'
    monkeypatch.setattr(runtime, 'load_model_run', lambda _: (tmp_path, {'config': cfg}))
    frame = pd.DataFrame({'date': ['2026-01-05'], 'symbol': ['TEST.SZ'], 'close': [20.]})
    with pytest.raises(ValueError, match='身份不匹配'):
        runtime.runtime_overlay(frame, frame.date.tolist())
