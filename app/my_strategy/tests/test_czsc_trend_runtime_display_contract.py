"""Runtime exports the causal prefix, frozen policy and actual account fields.

Model/native/account producers are isolated here; full Broker acceptance is
separate. These checks target the production runtime's composition boundaries.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.services import czsc_trend_runtime as runtime


@pytest.fixture
def composed_runtime(monkeypatch, tmp_path):
    from my_strategy.services import czsc_trend_pipeline as pipeline
    from my_strategy.services import czsc_trend_features as features
    from my_strategy.services import czsc_trend_consensus as consensus
    from my_strategy.services import czsc_native_evidence_v4 as native
    from my_strategy.services import czsc_trend_account as account
    from my_strategy.services import czsc_trend_model as trend_model

    cfg = strategy_config()
    cfg['consensus']['sell_threshold'] = .45
    report = {'config': cfg, 'trend_bundles': [], 'model_bundles': [],
        'ma_exit_bundles': [], 'protocol_freeze': {'freeze_hash': 'frozen'}}
    captures = {'actual_broker_replay': account.replay_trend_account}
    monkeypatch.setattr(runtime, 'load_model_run', lambda run_id: (tmp_path, report))
    monkeypatch.setattr(pipeline, 'pinned_exit_policy', lambda *args: None)

    def native_pack(frame, cfg, market_dates):
        captures['native_dates'] = frame.date.tolist()
        captures['native_cfg'] = copy.deepcopy(cfg)
        return {'frame': frame.copy(), 'wyckoff_events': []}

    def bind_inputs(frame, calendar):
        captures['bind_dates'] = frame.date.tolist()
        captures['bind_calendar'] = list(calendar)
        return frame.copy()

    def trend_values(frame, *args, **kwargs):
        captures['trend_dates'] = frame.date.tolist()
        captures['trend_calendar'] = kwargs['market_dates']
        return {d: {'p_up': None, 'p_down': None, 'p_neutral': None,
            'model_available': False, 'input_eligible': True,
            'reason_codes': ['no_prior_trend_head']} for d in frame.date}

    def directions(frame, calendar, **kwargs):
        rows = []
        for bar in frame.to_dict('records'):
            rows.append({'date': bar['date'], 'symbol': bar['symbol'],
                'reference_price': bar['close'], 'buy_base_score': .25,
                'candidate_evidence': [], 'agents': {
                    'ma': {'status': 'unavailable', 'p_up': None, 'p_down': None, 'p_neutral': None},
                    'structure': {'status': 'scored', 'buy_strength': .5, 'sell_strength': 0.},
                    'wyckoff': {'status': 'scored', 'buy_strength': .5, 'sell_strength': .1}}})
        return {'rows': rows, 'events': [], 'config': kwargs['config']['consensus']}

    def actual_account(frame, calendar, decisions, **kwargs):
        captures['account_dates'] = frame.date.tolist()
        captures['account_cfg'] = copy.deepcopy(kwargs['config'])
        rows, held = [], []
        for day in frame.date:
            tag = {'symbol': '002436.SZ', 'date': day,
                'account_id': kwargs['account_id'], 'series_context_hash': 'bound-series'}
            snapshot = {**tag, 'shares': 100, 'entry_date': '2026-01-01'}
            held.append(snapshot)
            rows.append({**tag, 'action': 'HOLD', 'emitted': False, 'shares': 100,
                'holding_snapshot': snapshot, 'exit_evaluation': {
                    'probability': .42, 'holding_contract_compatible': True,
                    'model_identity': {'head': 'ma_exit', 'checkpoint': 'pinned-at-entry'}}})
        return {'rows': rows, 'holding_snapshots': held, 'final_action_intents': [],
            'series_context_hash': 'bound-series', 'series_context': {
                'account_id': kwargs['account_id'], 'account_kind': 'research_broker',
                'account_start': kwargs['start'], 'initial_cash': cfg['initial_cash']}}

    monkeypatch.setattr(native, 'build_native_evidence_v4', native_pack)
    monkeypatch.setattr(trend_model, 'bind_trend_inputs', bind_inputs)
    monkeypatch.setattr(pipeline, 'forward_trend', trend_values)
    monkeypatch.setattr(consensus, 'prepare_trend_decisions', directions)
    monkeypatch.setattr(features, 'event_prefix_rows', lambda frame, *args: [{} for _ in frame.date])
    monkeypatch.setattr(account, 'replay_trend_account', actual_account)
    frame = pd.DataFrame({'date': ['2026-01-01', '2026-01-02', '2026-01-05'],
        'symbol': ['002436.SZ'] * 3, 'close': [20., 21., 999.]})
    captures.update(frame=frame, actual_account=actual_account)
    result = runtime.runtime_overlay(frame, frame.date.tolist(), model_run_id='frozen-model',
        as_of='2026-01-02', source_context_hash='raw-vintage')
    return result, captures


def test_all_producers_receive_strict_prefix_and_frozen_selected_exit_policy(composed_runtime):
    result, seen = composed_runtime
    for stage in ('native_dates', 'bind_dates', 'trend_dates', 'account_dates'):
        assert seen[stage] == ['2026-01-01', '2026-01-02']
    assert seen['trend_calendar'] == ['2026-01-01', '2026-01-02', '2026-01-05']
    assert seen['bind_calendar'] == seen['trend_calendar']
    assert seen['native_cfg']['consensus']['sell_threshold'] == .45
    assert seen['account_cfg']['consensus']['sell_threshold'] == .45
    assert all(row['agents']['ma']['p_up'] is None for row in result['rows'])
    assert all(row['agents']['high_return']['probability'] is None for row in result['rows'])


def test_display_target_comes_from_selected_models_frozen_label_contract(composed_runtime, monkeypatch, tmp_path):
    _, seen = composed_runtime
    cfg = copy.deepcopy(seen['account_cfg'])
    cfg['high_return'].update(min_net_return=.12, min_r_multiple=3.)
    report = {'config': cfg, 'trend_bundles': [], 'model_bundles': [], 'ma_exit_bundles': []}
    monkeypatch.setattr(runtime, 'load_model_run', lambda run_id: (tmp_path, report))
    result = runtime.runtime_overlay(seen['frame'], seen['frame'].date.tolist(), as_of='2026-01-02')
    target = {'min_net_return': .12, 'min_r_multiple': 3.}
    assert result['high_return_target'] == target
    assert all(row['agents']['high_return']['label_target'] == target for row in result['rows'])


def test_partial_quality_override_keeps_the_selected_models_frozen_policy(composed_runtime):
    _, seen = composed_runtime
    result = runtime.runtime_overlay(seen['frame'], seen['frame'].date.tolist(), as_of='2026-01-02',
        consensus_config={'high_return_threshold': .07})
    assert seen['account_cfg']['consensus']['sell_threshold'] == .45
    assert seen['account_cfg']['consensus']['high_return_threshold'] == .07
    assert result['rows'][-1]['agents']['high_return']['threshold'] == .07


def test_actual_held_exit_and_context_survive_export_without_becoming_direction_s(composed_runtime):
    result, _ = composed_runtime
    account = result['account_actions']
    assert account['intents'] == []
    assert account['context']['account_start'] == '2023-01-01'
    for row in result['rows']:
        assert row['holding']['shares'] == 100
        assert row['agents']['high_return']['evaluation_status'] == 'actual_existing_position'
        assert row['account_action']['exit_evaluation']['probability'] == .42
        assert row['account_action']['exit_evaluation']['model_identity']['checkpoint'] == 'pinned-at-entry'
        assert row['account_action']['emitted'] is False
        assert row['exit_probability'] is None  # Legacy outer scalar cannot replace actual-held detail.
        identity = row['account_action']['exit_evaluation']['model_identity']
        assert identity['usage_mode'] == 'historical'
        assert identity['historical_reconstruction'] is True


@pytest.mark.parametrize('status,probability,expected', [
    ('no_prior_high_return_head', None, 'model_unavailable'),
    ('scored', .03, 'scored_rejected'), ('scored', .2, 'scored_passed')])
def test_runtime_quality_export_preserves_null_reason_target_and_historical_identity(
        composed_runtime, monkeypatch, status, probability, expected):
    from my_strategy.services import czsc_trend_account as account
    from my_strategy.services import czsc_trend_pipeline as pipeline
    _, captures = composed_runtime

    def flat_first_day(frame, calendar, directions, **kwargs):
        first = frame.iloc[0].date
        kwargs['entry_evaluator']({'date': first, 'shares': 0}, {'entry_plan': {}})
        result = captures['actual_account'](frame, calendar, directions, **kwargs)
        result['holding_snapshots'] = result['holding_snapshots'][1:]
        result['rows'][0].update(shares=0, holding_snapshot=None)
        return result

    monkeypatch.setattr(account, 'replay_trend_account', flat_first_day)
    monkeypatch.setattr(pipeline, 'predict_entry', lambda *args: (probability,
        {'head': 'entry', 'actual_training_completed_at': '2026-10-12T13:35:00+08:00'}, status))
    result = runtime.runtime_overlay(captures['frame'], captures['frame'].date.tolist(),
        model_run_id='frozen-model', as_of='2026-01-02', source_context_hash='raw-vintage')
    quality = result['rows'][0]['agents']['high_return']
    assert quality['evaluation_status'] == expected
    assert quality['probability'] == probability
    assert quality['probability_scope'] == 'actual_flat_account_at_signal_close'
    if probability is None:
        assert quality['status'] == 'model_unavailable'
        assert quality['reason_codes'] == ['no_prior_high_return_head']
        assert quality['quality_contribution'] is None
    else:
        assert quality['quality_contribution'] == pytest.approx(probability * .1)
    assert quality['model_identity']['historical_reconstruction'] is True
    assert quality['model_identity']['usage_mode'] == 'historical'
    assert all(row['agents']['ma']['model_identity']['historical_reconstruction'] is True for row in result['rows'])


@pytest.mark.parametrize('account_status,reasons,expected,exported_probability', [
    ('scored_rejected', ['buy_total_support_below_threshold'], 'scored_rejected', .8),
    ('identity_error', ['model_contract_unverified'], 'identity_error', None),
    ('scored_rejected', ['high_return_population_out_of_scope'], 'input_ineligible', None)])
def test_actual_account_validation_overrides_adapter_pass_without_bypassing_ui(
        composed_runtime, monkeypatch, account_status, reasons, expected, exported_probability):
    from my_strategy.services import czsc_trend_account as account
    from my_strategy.services import czsc_trend_pipeline as pipeline
    _, captures = composed_runtime

    def rejected_account(frame, calendar, directions, **kwargs):
        first = frame.iloc[0].date
        adapter_score = kwargs['entry_evaluator']({'date': first, 'shares': 0}, {'entry_plan': {}})
        assert adapter_score['evaluation_status'] == 'scored_passed'
        checked_score = {**adapter_score, 'evaluation_status': account_status,
            'reason_codes': reasons, 'qualified': False}
        result = captures['actual_account'](frame, calendar, directions, **kwargs)
        result['holding_snapshots'] = result['holding_snapshots'][1:]
        result['rows'][0].update(shares=0, holding_snapshot=None, entry_evaluation=checked_score)
        return result

    monkeypatch.setattr(account, 'replay_trend_account', rejected_account)
    monkeypatch.setattr(pipeline, 'predict_entry', lambda *args: (.8,
        {'head': 'entry', 'actual_training_completed_at': '2026-10-12T13:35:00+08:00'}, 'scored'))
    result = runtime.runtime_overlay(captures['frame'], captures['frame'].date.tolist(),
        model_run_id='frozen-model', as_of='2026-01-02', source_context_hash='raw-vintage')
    row = result['rows'][0]
    quality = row['agents']['high_return']
    assert quality['evaluation_status'] == expected
    assert quality['probability'] == exported_probability
    assert quality['reason_codes'] == reasons
    assert row['model_resolution']['probability'] == exported_probability
    assert row['account_action']['emitted'] is False
    assert result['account_actions']['intents'] == []
    if exported_probability is None:
        assert quality['raw_probability'] == .8
        assert quality['quality_contribution'] is None
        assert quality['in_scope'] is False
        assert row['high_return_probability'] is None
    else:
        assert quality['probability'] >= quality['threshold']
        assert quality['quality_contribution'] == pytest.approx(.08)
        assert row['buy_total_score'] < result['trend_consensus']['config']['buy_total_threshold']
        assert 'raw_probability' not in quality
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required to verify the exported account check in shipped UI')
    ui_path = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js/chart_evidence.js'
    script = r'''
const assert=require('node:assert/strict'),ui=require(process.argv[1]);
const {row,expected,p}=JSON.parse(process.argv[2]);
const view=ui.currentAgentView(row,'high_return',row.agents.high_return);
assert.equal(view.evaluationStatus,expected);
const card=ui.opinionHtml(row.agents.high_return,'high_return',{decision:{version:'czsc_trend_consensus'}});
assert(!card.includes('通过门槛'));
assert(p===null?!view.summary.includes('80%'):view.summary.includes('80%'));
assert(ui.historyHtml([row]).includes('<td>—</td>'));
'''
    display = subprocess.run([node, '-e', script, str(ui_path),
        json.dumps({'row': row, 'expected': expected, 'p': exported_probability})],
        capture_output=True, text=True)
    assert display.returncode == 0, display.stderr


@pytest.mark.parametrize('reason,label,signal_index,gap_open', [
    ('holding_confirmed_boundary', '入场计划冻结的确认边界失效', 62, False),
    ('actual_account_stop', '研究账户冻结止损', 64, False),
    ('actual_account_stop', '研究账户冻结止损', 63, True)])
def test_real_broker_sell_row_reaches_history_with_owned_exit_reason(
        composed_runtime, monkeypatch, reason, label, signal_index, gap_open):
    """Controlled quotes produce real Broker rows, then runtime + shipped JS.

    Native/model producers stay isolated; the selected account and its exit are
    replayed by the production Broker rather than a literal S fixture.
    """
    from my_strategy.services import czsc_trend_account as account
    from my_strategy.services import czsc_trend_consensus as consensus
    from my_strategy.services import czsc_trend_pipeline as pipeline
    from my_strategy.tests.test_czsc_trend_account import fixture
    _, captures = composed_runtime
    frame, dates, prepared = fixture()
    if reason == 'holding_confirmed_boundary':
        prepared['rows'][60]['entry_plan']['holding_boundaries'] = [{
            'symbol': '002436.SZ', 'source': 'structure', 'event_id': 'candidate-origin',
            'original_event_id': 'candidate-origin', 'pattern_id': 'owned-pattern', 'price': 9.7,
            'first_available_at': dates[60]+'T15:00:00+08:00'}]
        frame.loc[62:, ['open', 'close', 'high', 'low']] = [9.6, 9.6, 9.7, 9.5]
    elif gap_open:
        frame.loc[62, ['open', 'close', 'high', 'low']] = [9.6, 9.6, 9.7, 9.5]
        frame.loc[63, ['open', 'close', 'high', 'low']] = [9.2, 9.6, 9.7, 9.1]
    else:
        for index, price in [(62, 9.7), (63, 9.4), (64, 9.2), (65, 9.2)]:
            frame.loc[index, ['open', 'close', 'high', 'low']] = [price, price, price+.05, price-.05]
        frame.loc[64, ['open', 'high']] = [9.3, 9.35]
    for row, bar in zip(prepared['rows'], frame.to_dict('records'), strict=True):
        row.update(reference_price=bar['close'], candidate_evidence=[])
        for source in ('structure', 'wyckoff'):
            row['agents'][source] = {'status': 'scored', 'buy_strength': .5, 'sell_strength': .1}
    cfg = strategy_config({'consensus': {'require_high_return': False}})
    prepared.update(events=[], config=cfg['consensus'])
    monkeypatch.setattr(runtime, 'load_model_run', lambda run_id: (Path('controlled-model'), {
        'config': cfg, 'trend_bundles': [], 'model_bundles': [], 'ma_exit_bundles': []}))
    monkeypatch.setattr(consensus, 'prepare_trend_decisions', lambda *args, **kwargs: copy.deepcopy(prepared))
    def actual_broker(frame, calendar, decisions, **kwargs):
        assert kwargs['config']['consensus']['require_high_return'] is False
        return captures['actual_broker_replay'](frame, calendar, decisions, **kwargs)
    monkeypatch.setattr(account, 'replay_trend_account', actual_broker)
    result = runtime.runtime_overlay(frame, dates, as_of=dates[-1],
        source_context_hash='controlled-frozen-quote-calendar')
    assert result['account_actions']['context']['consensus']['require_high_return'] is False
    row = result['rows'][signal_index]
    assert row['account_action']['emitted'] is True, {
        'ledger': result['account_actions']['ledger'],
        'rejections': result['account_actions']['rejections'],
        'entry': result['account_actions'].get('entry_diagnostics')}
    assert row['account_action']['action'] == 'SELL'
    assert reason in row['account_action']['reason_codes']
    assert result['account_actions']['ledger'][-1]['reason'] == reason
    assert result['account_actions']['ledger'][-1]['signal_date'] == dates[signal_index]
    assert result['account_actions']['ledger'][-1]['date'] == dates[signal_index+(0 if gap_open else 1)]
    if gap_open:
        intent = row['account_action']['final_action_intents'][0]
        assert intent['original_signal_at'].endswith('T09:30:00+08:00')
        assert intent['reference_price'] == 9.2
        assert row['reference_close'] == 9.6
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for shipped S history display')
    ui_path = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js/chart_evidence.js'
    rendered = subprocess.run([node, '-e', r'''
const assert=require('node:assert/strict'),ui=require(process.argv[1]);
const {row,label,gap}=JSON.parse(process.argv[2]);const html=ui.historyHtml([row]);
assert(html.includes('<td>S</td>'));assert(html.includes(label));
assert(!html.includes('本日无新账户操作'));
if(gap){
 const detail=ui.accountActionHtml(row);assert(detail.includes('开盘风控触发；以下 Agent 分数为当日收盘复核，不是该卖出的触发依据'));
 assert(detail.includes('T09:30:00+08:00'));
 const intent=row.account_action.final_action_intents[0],context={evidence:{rows:[row]},groups:[{kind:'intention',sources:['account_action'],entries:[{event_date:row.date,row:intent}]}],barDate:row.date};
 assert.equal(ui.referenceSelection(context).price,9.2);assert(ui.referenceSelection(context).available_at.endsWith('T09:30:00+08:00'));
}
''', str(ui_path), json.dumps({'row': row, 'label': label, 'gap': gap_open})], capture_output=True, text=True)
    assert rendered.returncode == 0, rendered.stderr
