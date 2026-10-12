"""Dashboard capabilities prevent crossing old account and new native schemas."""
import asyncio
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from my_strategy.web_dashboard import high_return_routes_v4 as routes
from my_strategy.web_dashboard.api import TaskRequest
from my_strategy.tests.test_czsc_agent_consensus_v4 import _row


@pytest.fixture
def native_root(monkeypatch, tmp_path):
    root = tmp_path / 'native' / 'reports'
    root.mkdir(parents=True)
    (root / 'native_v4.json').write_text('{}')
    monkeypatch.setattr(routes, 'ARTIFACT_RUNS_ROOT', tmp_path)
    return tmp_path


def test_explicit_native_scan_and_old_arbiter_cannot_mix(native_root):
    assert routes.native_scan_requested(model_run_id='native') is True
    for version in ('czsc_agent_consensus_v1', 'czsc_agent_consensus_v2', 'czsc_agent_consensus_v3'):
        with pytest.raises(ValueError, match='不能用于v1/v2/v3扫描'):
            routes.native_scan_requested(model_run_id='native', consensus_version=version)
    assert routes.native_scan_requested(consensus_version=routes.V4) is True


@pytest.mark.parametrize('operation', ['backtest', 'holding', 'forward', 'high_return_train'])
def test_native_run_rejects_legacy_contract_explicitly(native_root, operation):
    with pytest.raises(routes.NativeVersionUnsupported, match='native v4不支持旧'):
        routes.reject_native_legacy_operation(operation, run_id='native')
    assert routes.reject_native_legacy_operation(operation, run_id='legacy') is None


def test_task_schema_preserves_explicit_native_scan_version_and_config():
    task = TaskRequest(kind='scan', spec={'research': True, 'model_family': 'high_return', 'consensus_version': routes.V4})
    assert task.spec.consensus_version == routes.V4
    assert task.spec.consensus_config['version'] == routes.V4
    assert task.spec.consensus_config['source_weights'] == {'ma': .4, 'structure': .2, 'wyckoff': .3}


def test_old_backtest_request_does_not_queue_native_model(native_root):
    with pytest.raises(ValidationError, match='native v4不支持旧'):
        TaskRequest(kind='high_return_backtest', spec={'model_run_id': 'native'})
    assert TaskRequest(kind='high_return_backtest', spec={'model_run_id': 'legacy'}).spec.model_run_id == 'legacy'


def test_native_holding_cannot_reach_old_ledger_reader(native_root, monkeypatch):
    from my_strategy.web_dashboard import holding_evidence
    monkeypatch.setattr(holding_evidence, '_load_reference', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('old reader called')))
    result = holding_evidence.holding_exit_research(replay_run_id='native', window='2026YTD', arm='full',
        symbol='000001.SZ', as_of='2026-01-05', runs_root=native_root)
    assert result['status'] == 'unsupported_version'
    assert result['exit_probability'] is None and result['holding_contract_compatible'] is False


def test_native_forward_api_rejects_before_old_protocol(native_root, monkeypatch):
    from my_strategy.web_dashboard import api
    from my_strategy.services import czsc_high_return_forward as old
    monkeypatch.setattr(old, 'forward_status', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('old protocol called')))
    with pytest.raises(routes.NativeVersionUnsupported):
        asyncio.run(api.high_return_forward_status('native'))


def test_status_includes_native_models_and_preserves_old_models(monkeypatch, native_root):
    from my_strategy.services import czsc_high_return_research as old
    from my_strategy.services import czsc_high_return_runtime_v4 as native
    from my_strategy import storage
    import my_strategy.storage.czsc_results as results
    monkeypatch.setattr(old, 'status_high_return', lambda: {'runs': [{'run_id': '2025-old'}], 'models': [{'run_id': '2025-old'}], 'failures': []})
    class Store:
        def list(self, *args, **kwargs): return [{'run_id': '2026-native'}]
    monkeypatch.setattr(results, 'ResultStore', Store)
    report = {'data_end': '2026-10-08', 'model_bundles': [{'name': 'production', 'available_at': '2026-09-30', 'heads': {'entry': {}}}],
        'model_gate': {'passed': False}, 'config': {'initial_cash': 100000, 'risk_policy': 'source_confirmed_boundary_prior_atr_v4'}, 'source_snapshot': {'selected_count': 60, 'snapshot_hash': 'frozen'}}
    monkeypatch.setattr(native, 'read_run', lambda _: (native_root, report))
    status = routes.status_high_return()
    assert [entry['run_id'] for entry in status['runs']] == ['2026-native', '2025-old']
    new = status['native_models'][0]
    assert new['capabilities']['chart'] and new['capabilities']['scan'] and new['capabilities']['offline_validation']
    assert not new['capabilities']['legacy_backtest'] and not new['capabilities']['legacy_holding']
    assert status['models'][-1]['capabilities']['legacy_backtest'] is True


def test_native_scan_uses_full_prefix_final_points_and_keeps_failures(monkeypatch):
    from my_strategy.adapters import czsc_adapter
    from my_strategy.web_dashboard import high_return
    from my_strategy.services import czsc_high_return_runtime_v4 as native
    day = '2026-01-05'
    monkeypatch.setattr(czsc_adapter, 'latest_market_date', lambda _: day)
    monkeypatch.setattr(high_return, '_calendar', lambda _: ([day], 'calendar'))
    raw = pd.DataFrame([{'date': day, 'symbol': '000001.SZ', 'close': 100.}])
    def prepare(symbol, *args, **kwargs):
        if symbol == '000002.SZ': raise ValueError('missing verified day')
        return {'symbol': symbol, 'data_end': day, 'data_version': 'raw-hash', 'features': raw, 'raw': raw}
    monkeypatch.setattr(high_return, '_prepare', prepare)
    row = _row()
    monkeypatch.setattr(native, 'runtime_overlay', lambda *a, **kw: {'rows': [row], 'latest_decision': row})
    written = []
    class Context:
        run_id = 'native-scan'
        def to_dict(self): return {'run_id': self.run_id}
        def write_metadata(self, value): written.append(value)
    monkeypatch.setattr(routes, 'create_run_context', lambda **kwargs: Context())
    result = routes.scan_native_high_return(symbols=['000001.SZ', '000002.SZ'], cpu_workers=4)
    assert result['coverage'] == {'requested': 2, 'success': 1, 'failed': 1}
    assert result['rows'][0]['action'] == 'BUY' and result['rows'][0]['category'] == 'buy'
    assert result['rows'][0]['consensus_version'] == routes.V4 and result['rows'][0]['exit_probability'] is None
    assert result['failures'][0]['symbol'] == '000002.SZ'
    assert result['publication_allowed'] is False and written[0]['failures']


def test_task_native_backtest_rejects_and_legacy_backtest_still_routes(native_root, monkeypatch):
    from my_strategy.web_dashboard.tasks import TaskManager
    from my_strategy.services import czsc_high_return_research as old
    manager = object.__new__(TaskManager)
    manager._check = lambda _: None
    manager._patch = lambda *a, **kw: None
    calls = []
    monkeypatch.setattr(old, 'backtest_high_return', lambda **kw: calls.append(kw) or {'run_id': 'old-result'})
    with pytest.raises(routes.NativeVersionUnsupported):
        manager._run('job', 'high_return_backtest', {'model_run_id': 'native'})
    assert calls == []
    result = manager._run('job', 'high_return_backtest', {'model_run_id': 'legacy'})
    assert result['run_id'] == 'old-result' and calls[0]['model_run_id'] == 'legacy'


def test_risk_control_model_is_explicit_only_in_status(monkeypatch, native_root):
    from my_strategy.services import czsc_high_return_research as old
    from my_strategy.services import czsc_high_return_runtime_v4 as native
    import my_strategy.storage.czsc_results as results
    monkeypatch.setattr(old, 'status_high_return', lambda: {'runs': [], 'models': [], 'failures': []})
    class Store:
        def list(self, *a, **kw): return [{'run_id': 'main'}, {'run_id': 'control'}]
    monkeypatch.setattr(results, 'ResultStore', Store)
    def report(run):
        return native_root, {'data_end': '2026-10-08', 'model_bundles': [], 'model_gate': {'passed': False},
            'config': {'initial_cash': 100000, 'risk_policy': 'source_confirmed_boundary_prior_atr_v4' if run=='main' else 'legacy_trailing20_v4_control'},
            'source_snapshot': {'selected_count': 60, 'snapshot_hash': 'frozen'}}
    monkeypatch.setattr(native, 'read_run', report)
    models = {m['run_id']: m for m in routes.status_high_return()['native_models']}
    assert models['main']['automatic_selection_eligible'] is True
    assert models['control']['automatic_selection_eligible'] is False
    assert models['control']['risk_policy'] == 'legacy_trailing20_v4_control'
    assert models['control']['capabilities']['chart'] is True


def test_explicit_legacy_scan_model_retains_legacy_route_after_default_v4(monkeypatch):
    monkeypatch.setattr(routes, 'resolve_consensus', lambda *a, **kw: {'version': routes.V4})
    monkeypatch.setattr(routes, 'is_native_run', lambda run: run == 'native')
    assert routes.native_scan_requested(model_run_id='legacy') is False
    assert routes.native_scan_requested(model_run_id='native') is True
    assert routes.native_scan_requested() is True


def test_known_legacy_high_return_model_defaults_to_v3_without_name_guess(monkeypatch, native_root):
    from my_strategy.web_dashboard.api import AnalysisRequest
    old = native_root / 'legacy' / 'reports'
    old.mkdir(parents=True)
    (old / 'high_return.json').write_text('{}')
    assert AnalysisRequest(symbol='000001.SZ', model_family='high_return', model_run_id='legacy').consensus_version == 'czsc_agent_consensus_v3'
    assert AnalysisRequest(symbol='000001.SZ', model_family='high_return', model_run_id='native').consensus_version == routes.V4
    # No naming-based inference: an unknown ID follows the family default, and
    # the selected model loader remains responsible for rejecting its absence.
    assert AnalysisRequest(symbol='000001.SZ', model_family='high_return', model_run_id='unknown-v3-name').consensus_version == routes.V4
    with pytest.raises(ValidationError, match='v4仲裁只支持'):
        AnalysisRequest(symbol='000001.SZ', model_family='high_return', model_run_id='legacy', consensus_version=routes.V4)
