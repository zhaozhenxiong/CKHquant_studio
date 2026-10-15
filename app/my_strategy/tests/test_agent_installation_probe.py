"""The release probe executes real code while all fixtures remain isolated."""
from pathlib import Path
import importlib.util
import sqlite3

import pytest


@pytest.fixture
def fixture_connections(monkeypatch):
    """Keep handles alive so resource leaks fail on POSIX as well as Windows."""
    connections = []
    original_connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        closed = False

        def close(self):
            super().close()
            self.closed = True

    def connect(database, *args, **kwargs):
        if str(database).split('?')[0].endswith('synthetic-prices.db'):
            kwargs['factory'] = TrackedConnection
            connection = original_connect(database, *args, **kwargs)
            connections.append(connection)
            return connection
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, 'connect', connect)
    try:
        yield connections
    finally:
        for connection in connections:
            if not connection.closed:
                connection.close()


def test_v1_installation_probe_uses_real_bounded_isolated_components(fixture_connections):
    path = Path(__file__).resolve().parents[2] / 'verify_agent_installation.py'
    spec = importlib.util.spec_from_file_location('agent_installation_probe_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run_probe('cpu')
    assert result['status'] == 'passed' and result['family'] == 'agent_mlp_sim_v1'
    assert result['mlp']['parameter_count'] > 0
    assert result['mlp']['missing_core_rejected'] and result['mlp']['b_without_observations_pending']
    assert result['incremental_source']['native_rows_appended'] == 30
    assert result['incremental_source']['append_matches_full'] and result['incremental_source']['cancelled_terminal']
    assert result['simulation_ledger']['actual_fills'] == 2
    assert result['scratch_removed'] and not result['business_models_loaded']
    assert fixture_connections and all(connection.closed for connection in fixture_connections)


def test_installation_fixture_commits_and_closes_database(tmp_path, fixture_connections):
    from verify_installation import fixture_database

    path = tmp_path / 'synthetic-prices.db'
    dates = fixture_database(path)
    assert fixture_connections and all(connection.closed for connection in fixture_connections)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute('SELECT count(*) FROM stock_daily_normalized').fetchone()[0] == len(dates) == 260
    finally:
        connection.close()
    path.unlink()
