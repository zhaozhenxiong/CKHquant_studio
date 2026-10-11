"""Final chart directions run through real guarded, fee-charging Broker paths."""
import pandas as pd
import pytest

from my_strategy.scripts.validate_candidate_consensus import broker_decisions, replay


def fixture():
    days = pd.bdate_range('2026-01-01', periods=75).strftime('%Y-%m-%d').tolist()
    frame = pd.DataFrame([{'date': day, 'symbol': '002436.SZ', 'open': 10., 'high': 10.2,
        'low': 9.8, 'close': 10., 'volume': 1000000., 'amount': 10000000.,
        'source': 'tushare', 'has_trade_price': 1} for day in days])
    plan = {'symbol': '002436.SZ', 'signal_date': days[61], 'plan_id': 'frozen-example',
        'available_at': days[61]+'T15:00:00+08:00', 'status': 'active', 'policy': 'risk',
        'entry_allowed': True, 'valid_for_sessions': 1, 'target_weight': .2,
        'stop_price': 9., 'price_floor': 9.1, 'price_ceiling': 10.5, 'risk_fraction': .01}
    evidence = [{'date': days[61], 'raw_risk_plan': plan}]
    decisions = [{'date': days[61], 'action': 'BUY'}, {'date': days[65], 'action': 'SELL'}]
    return frame, days, evidence, decisions


def test_buy_and_sell_fill_only_next_open_and_wait_preserves_actual_position():
    frame, days, evidence, decisions = fixture()
    result = replay(frame, evidence, decisions, days, start=days[60], end=days[-1], run_id='test')
    assert [(r['action'], r['signal_date'], r['date']) for r in result['ledger']] == [
        ('BUY', days[61], days[62]), ('SELL', days[65], days[66])]
    assert result['completed_round_trips'] == 1
    assert result['metrics']['fees'] > 0 and result['metrics']['total_return'] < 0
    assert result['policy']['l2_continuation_contract_matches'] is False


def test_last_close_signal_has_no_invented_execution_after_window():
    frame, days, evidence, decisions = fixture()
    result = replay(frame, evidence, decisions[:1], days, start=days[60], end=days[61], run_id='test')
    assert result['ledger'] == []
    assert result['metrics']['open_shares'] == 0


def test_missing_next_market_bar_rejects_entry_instead_of_using_later_open():
    frame, days, evidence, decisions = fixture()
    frame = frame[frame.date != days[62]]
    result = replay(frame, evidence, decisions, days, start=days[60], end=days[-1], run_id='test')
    assert result['ledger'] == []
    assert any(r['reason'] == 'entry_plan_expired' for r in result['rejections'])


def test_final_buy_cannot_execute_without_matching_current_risk_plan():
    frame, days, evidence, decisions = fixture()
    evidence[0]['raw_risk_plan']['signal_date'] = days[60]
    with pytest.raises(ValueError, match='exact-day'):
        broker_decisions(frame, evidence, decisions, start=days[60], end=days[-1])
