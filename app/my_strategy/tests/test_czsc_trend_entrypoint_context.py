"""Cross-entrypoint source binding and requested observation end remain distinct."""
import pandas as pd
import pytest

from my_strategy.web_dashboard import trend_strategy as strategy


@pytest.mark.parametrize('requested_end,decision_date', [
    ('2026-01-02', '2026-01-02'), ('2026-01-05', '2026-01-05'), ('2026-01-04', '2026-01-02')])
def test_analysis_and_worker_bind_same_vintage_but_observe_requested_prefix(monkeypatch, requested_end, decision_date):
    from my_strategy.web_dashboard import high_return
    from my_strategy.services import czsc_analysis, czsc_trend_runtime
    from my_strategy.adapters import czsc_adapter
    from my_strategy.web_dashboard import chart_evidence_trend

    dates = ['2026-01-02', '2026-01-05']
    item = {'features': pd.DataFrame({'date': dates, 'symbol': ['002436.SZ']*2}),
        'raw': pd.DataFrame({'date': dates}), 'data_version': 'frozen-vintage', 'data_end': dates[-1]}
    captured = []
    monkeypatch.setattr(czsc_adapter, 'latest_market_date', lambda *args: dates[-1])
    monkeypatch.setattr(high_return, '_calendar', lambda *args: (dates, 'verified-calendar'))
    monkeypatch.setattr(high_return, '_prepare', lambda *args: item)
    monkeypatch.setattr(high_return, '_worker_init', lambda: None)
    monkeypatch.setattr(czsc_analysis, 'analyze_stock', lambda *args, **kwargs: {'data_end': dates[0]})

    def runtime(frame, calendar, **kwargs):
        captured.append(kwargs)
        return {'model_run_id': 'bound-model'}

    monkeypatch.setattr(czsc_trend_runtime, 'runtime_overlay', runtime)
    monkeypatch.setattr(chart_evidence_trend, 'decorate_analysis', lambda *args, **kwargs: {'research': {}})
    result = strategy.analyze_trend_consensus(symbol='002436.SZ', end=requested_end)
    strategy._stock_worker(('002436.SZ', decision_date, dates, {}, dates[-1]))
    assert captured[0]['source_context_hash'] == captured[1]['source_context_hash']
    assert captured[0]['as_of'] == captured[1]['as_of'] == decision_date
    assert result['research']['observed_end'] == dates[0]
    assert result['research']['raw_vintage_end'] == dates[-1]
    assert result['research']['decision_date'] == decision_date
    assert result['research']['requested_end'] == requested_end


def test_saved_auto_analysis_keeps_actual_run_and_account_for_replay(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from my_strategy.web_dashboard import high_return, trend_routes
    from my_strategy.web_dashboard.app import create_app
    from my_strategy.web_dashboard.tasks import TaskManager
    from my_strategy.services.czsc_trend_contracts import strategy_config
    from my_strategy.storage.czsc_results import ResultStore
    cfg = strategy_config()['consensus']
    monkeypatch.setattr(trend_routes, '_saved_consensus', lambda *args: cfg)
    monkeypatch.setattr(high_return, 'analyze_high_return', lambda **kwargs: {
        'symbol': '002436.SZ', 'data_end': '2026-09-30', 'decision_date': '2026-10-09',
        'data_version': 'actual-vintage', 'strategy_version': 'czsc_trend_consensus',
        'chart_evidence': {}, 'consensus_decisions': {'version': 'czsc_trend_consensus', 'config': cfg},
        'research': {'model_run_id': 'chosen-trained-run'},
        'account_actions': {'context': {'account_start': '2023-01-01', 'initial_cash': 100000.}}})
    manager = TaskManager(tmp_path/'tasks.db', ResultStore(tmp_path/'results.db', tmp_path/'runs'))
    with TestClient(create_app(task_manager=manager)) as client:
        response = client.post('/api/analysis', json={'symbol': '002436.SZ',
            'model_family': 'high_return', 'consensus_version': 'czsc_trend_consensus'})
        assert response.status_code == 200
        result = response.json()
        saved = client.get('/api/runs/'+result['run_id']).json()
        assert saved == result
        assert saved['request']['model_policy'] == 'auto'
        assert saved['request']['model_run_id'] == 'chosen-trained-run'
        assert saved['request']['account_start'] == '2023-01-01'
        assert saved['request']['account_initial_cash'] == 100000.
        assert saved['run_context']['as_of_date'] == '2026-10-09'


def test_new_analysis_request_always_declares_actual_risk_plan(monkeypatch, tmp_path):
    from my_strategy.web_dashboard import trend_routes
    from my_strategy.web_dashboard.api import AnalysisRequest
    monkeypatch.setattr(trend_routes, 'CURRENT_PATH', tmp_path/'missing-current.json')
    monkeypatch.setattr(trend_routes, '_saved_consensus', lambda *args: None)
    request = AnalysisRequest(symbol='002436.SZ', model_family='high_return',
        consensus_version='czsc_trend_consensus', entry_policy='legacy')
    assert request.entry_policy == 'risk'


def test_pinned_entry_does_not_change_trend_continuation_catalogue():
    from my_strategy.services.czsc_trend_runtime import _pinned_catalogues
    report = {'model_bundles': [{'name': 'production', 'available_at': '2026-09-30'}],
        'trend_bundles': [{'name': 'trend_2026Q3', 'available_at': '2026-06-30'},
            {'name': 'trend_2026Q4', 'available_at': '2026-09-30'},
            {'name': 'trend_future', 'available_at': '2026-12-31'}]}
    trends, entries = _pinned_catalogues(report, 'production', '2026-10-09')
    assert entries[0]['name'] == 'production'
    assert trends == report['trend_bundles']
    with pytest.raises(ValueError, match='尚不可用'):
        _pinned_catalogues(report, 'production', '2026-09-30')


def test_cancelled_market_scan_does_not_queue_remaining_universe(monkeypatch):
    from concurrent.futures import Future
    from my_strategy.adapters import czsc_adapter
    monkeypatch.setattr(czsc_adapter, 'latest_market_date', lambda: '2026-10-09')
    submitted, shutdown = [], []

    class Pool:
        def __init__(self, **kwargs): pass
        def submit(self, function, payload):
            submitted.append(payload[0])
            future = Future();future.set_result(payload[0]);return future
        def shutdown(self, **kwargs): shutdown.append(kwargs)

    monkeypatch.setattr(strategy, 'ProcessPoolExecutor', Pool)
    cancelled = False
    def cancel():
        if cancelled: raise RuntimeError('cancelled')
    iterator = strategy._batch(list(range(50)), '2026-10-09', ['2026-10-09'], {}, 2, check_cancel=cancel)
    next(iterator)
    cancelled = True
    with pytest.raises(RuntimeError, match='cancelled'):
        next(iterator)
    assert len(submitted) == 4
    assert shutdown == [{'wait': True, 'cancel_futures': True}]


def test_first_use_without_model_directory_is_explicitly_unavailable(monkeypatch, tmp_path):
    from my_strategy.services import czsc_trend_runtime
    monkeypatch.setattr(czsc_trend_runtime, 'ARTIFACT_RUNS_ROOT', tmp_path/'absent-runs')
    assert czsc_trend_runtime.load_model_run() == (None, None)
    with pytest.raises(ValueError, match='unavailable'):
        czsc_trend_runtime.load_model_run('explicit-missing-model')


def test_current_run_selection_keeps_auto_heads_and_explicit_requests_win(monkeypatch, tmp_path):
    from my_strategy.services import czsc_trend_runtime, czsc_trend_pipeline
    current = tmp_path/'current.json'
    current.write_text('{"strategy":"czsc_trend_consensus","model_run_id":"delivered-run"}')
    monkeypatch.setattr(czsc_trend_runtime, 'CURRENT_MODEL_PATH', current)
    monkeypatch.setattr(czsc_trend_runtime, 'ARTIFACT_RUNS_ROOT', tmp_path)
    seen = []
    def read_run(run_id):
        seen.append(run_id)
        return run_id, {'automatic_selection_eligible': False}
    monkeypatch.setattr(czsc_trend_pipeline, 'read_run', read_run)
    assert czsc_trend_runtime.load_model_run()[0] == 'delivered-run'
    assert czsc_trend_runtime.load_model_run('explicit-research-run')[0] == 'explicit-research-run'
    assert seen == ['delivered-run', 'explicit-research-run']


def test_suspended_stock_cannot_present_prior_buy_or_score_as_current():
    from my_strategy.services.czsc_trend_runtime import _latest_observed_decision
    old = {'date': '2026-09-30', 'action': 'BUY', 'reference_close': 999.,
        'agents': {'ma': {'p_up': .9}}, 'high_return_probability': .8}
    action = {'date': '2026-10-09', 'action': 'SELL', 'shares': 100,
        'emitted': True, 'reason_codes': ['max_holding_sessions']}
    actual = _latest_observed_decision([old], {'2026-10-09': action}, '2026-10-09', '002436.SZ')
    assert actual['date'] == '2026-10-09' and actual['reference_close'] is None
    assert actual['agents']['ma']['p_up'] is None and actual['high_return_probability'] is None
    assert actual['account_action'] == action  # A clock-based real holding exit still survives.


def test_backtest_aggregation_preserves_failed_cash_and_ui_coverage(monkeypatch, tmp_path):
    from my_strategy.services import czsc_trend_runtime
    from my_strategy.web_dashboard import high_return
    from my_strategy.adapters import czsc_adapter
    monkeypatch.setattr(czsc_adapter, 'latest_market_date', lambda *args: '2026-01-02')
    monkeypatch.setattr(high_return, '_calendar', lambda *args: (['2026-01-02'], 'verified'))
    monkeypatch.setattr(czsc_trend_runtime, 'load_model_run', lambda *args: (tmp_path, {'model_gate': {'passed': False}}))
    account = {'metrics': {'total_return': .1}, 'series_context_hash': 'one-account',
        'ledger': [], 'rejections': [{'date': '2026-01-02', 'reason': 'risk_open_outside_frozen_price_band'}],
        'daily': [{'date': '2026-01-02', 'equity': 110000.}]}
    monkeypatch.setattr(strategy, '_batch', lambda *args, **kwargs: iter([
        (('002436.SZ', {}, {'account_actions': account}), None),
        (None, {'symbol': '600519.SH', 'error': 'source unavailable'})]))
    result = strategy.backtest_trend_consensus(model_run_id='known-model',
        symbols=['002436.SZ', '600519.SH'], end='2026-01-02')
    assert result['coverage']['success'] == result['coverage']['successful'] == 1
    assert result['coverage']['failed'] == 1 and result['failures'] == result['failed']
    assert result['initial_cash'] == 200000. and result['per_stock_initial_cash'] == 100000.
    assert result['daily'][-1]['equity'] == 210000.
    assert result['metrics']['total_return'] == .05
    assert len(result['rejections']) == result['metrics']['rejection_count'] == 1
