"""The current chart describes target changes without borrowing legacy gates."""
import copy

import pandas as pd
import pytest

from my_strategy.web_dashboard.chart_evidence_trend import decorate_analysis


@pytest.mark.parametrize('action,expected', [('WAIT', '同日参考目标未变化；来源点独立保留'),
    ('BUY', '同日参考目标增加；来源点独立保留'), ('SELL', '同日参考目标减少；来源点独立保留')])
def test_reference_source_labels_follow_target_change_without_rejecting_source(action, expected):
    day = '2026-10-12'
    event = {'symbol': '000001.SZ', 'date': day, 'available_at': day + 'T15:00:00+08:00',
        'first_available_at': day + 'T15:00:00+08:00', 'side': 'BUY', 'source': 'structure',
        'event_id': 'source', 'reference_layer': 'source', 'candidate_direction_valid': True,
        'confirmed': True, 'input_eligible': True}
    row = {'symbol': event['symbol'], 'date': day, 'action': action, 'events': [event],
        'agents': {}, 'source_reference_points': [event],
        'position_reference': {'schema': 'reference_position_v1'}}
    final = [{**event, 'side': action, 'action': action, 'reference_layer': 'consensus', 'event_id': 'target'}] if action != 'WAIT' else []
    overlay = {'strategy_version': 'reference_position_v1', 'display_name': '参考买卖点与逐股仓位',
        'analysis_mode': 'reference', 'rows': [row], 'latest_decision': row,
        'trend_consensus': {'rows': [row], 'events': final}, 'reference_points': final,
        'source_reference_points': [event], 'reference_recommendation': {},
        'position_reference': row['position_reference']}
    result = {'symbol': event['symbol'], 'data_end': day, 'frequencies': {'日线': {'bars': [{'time': day}]}}}
    raw = pd.DataFrame([{'date': day, 'open': 10., 'high': 10.2, 'low': 9.8, 'close': 10.}])
    out = decorate_analysis(result, raw=raw, overlay=overlay)
    assert out['source_reference_points'][0]['source_relationship_text'] == expected
    assert out['chart_evidence']['rows'][0]['events'][0]['source_relationship_text'] == expected
    assert out['trend_consensus']['rows'][0]['source_reference_points'][0]['source_relationship_text'] == expected
    assert '未通过' not in out['source_reference_points'][0]['source_relationship_text']
    if final:
        assert out['reference_points'][0]['reference_display_label'] == {'BUY': '参考增仓', 'SELL': '参考减仓'}[action]
    # The original native event must remain immutable and independently valid.
    assert 'source_relationship_text' not in event and event['input_eligible'] is True
    archived = copy.deepcopy(overlay)
    archived['strategy_version'] = 'trend_consensus'
    legacy = decorate_analysis(result, raw=raw, overlay=archived)
    assert 'reference_position_schema' not in legacy['source_reference_points'][0]


def test_current_native_chart_warning_has_reference_scope_without_retired_fee_target(monkeypatch):
    from my_strategy.services import czsc_analysis
    from my_strategy.web_dashboard.trend_strategy import _reference_structure
    class Session:
        def __init__(self, *args, **kwargs): pass
        def advance_structure(self, row): pass
        def payload(self, lower): return {'日线': {'bars': []}}
    monkeypatch.setattr(czsc_analysis, 'CzscSession', Session)
    raw = pd.DataFrame([{'symbol': '000001.SZ', 'date': pd.Timestamp('2026-10-12'),
        'open': 10., 'high': 10.2, 'low': 9.8, 'close': 10., 'volume': 100.,
        'amount': 1000., 'source': 'tushare', 'has_trade_price': 1}])
    current = _reference_structure(raw, start=None, decision_date='2026-10-12',
        reference_schema='reference_position_v1')
    assert '高收益门槛' not in '\n'.join(current['warnings'])
    assert '逐股参考目标增加或减少' in '\n'.join(current['warnings'])
    assert '不模拟成交或记录账户持仓' in '\n'.join(current['warnings'])
