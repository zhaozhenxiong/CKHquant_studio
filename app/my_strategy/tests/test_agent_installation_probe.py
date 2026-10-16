"""The release probe executes real code while all fixtures remain isolated."""
from pathlib import Path
from contextlib import closing
import importlib.util
import json
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
    from verify_installation import verify_lightweight_probe
    assert verify_lightweight_probe(result, 'cpu') is result
    evidence = result['v101']
    assert evidence['ma_profile']['input_dimensions'] == 20
    assert evidence['ma_profile']['same_input_l2'] and not evidence['ma_profile']['active_model_registered']
    assert evidence['board_scope']['eligible_count'] == 2 and evidence['board_scope']['excluded_count'] == 4
    assert evidence['task_maintenance']['active_delete_rejected']
    assert evidence['task_lifecycle']['restart_terminal_preserved']
    assert evidence['task_lifecycle']['history_projection_consistent'] and evidence['task_lifecycle']['workers_joined']
    assert evidence['spawn']['start_method'] == 'spawn' and evidence['spawn']['stocks'] == 2
    assert evidence['spawn']['children_joined'] and evidence['spawn']['worker_pids']
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


def test_installed_task_lifecycle_uses_real_manager_and_maintenance(tmp_path, monkeypatch):
    import verify_agent_installation as probe
    from my_strategy.web_dashboard.tasks import TaskManager

    original = TaskManager._write
    writes = []
    def write(self, job):
        writes.append(job['status'])
        return original(self, job)
    monkeypatch.setattr(TaskManager, '_write', write)
    result = probe.task_lifecycle_probe(tmp_path)
    assert result['status'] == 'passed' and result['workers_joined']
    assert result['restart_terminal_preserved'] and result['history_projection_consistent']
    assert {'queued', 'running', 'succeeded', 'cancelling', 'cancelled'} <= set(writes)


def test_installed_task_lifecycle_rejects_stale_trigger_projection(tmp_path, monkeypatch):
    import verify_agent_installation as probe
    from my_strategy.services.task_maintenance import TaskMaintenance

    original = TaskMaintenance._prepare_history_index
    def stale_index(self):
        original(self)
        with closing(sqlite3.connect(self.tasks.db_path)) as conn:
            conn.executescript('''DROP TRIGGER task_display_update;
                CREATE TRIGGER task_display_update AFTER UPDATE ON czsc_tasks BEGIN SELECT 1; END;''')
    monkeypatch.setattr(TaskMaintenance, '_prepare_history_index', stale_index)
    with pytest.raises(RuntimeError, match='task/history status mismatch'):
        probe.task_lifecycle_probe(tmp_path)


def test_installed_task_lifecycle_rejects_old_upsert_with_real_triggers(tmp_path, monkeypatch):
    import verify_agent_installation as probe
    from my_strategy.core.run_context import stable_hash
    from my_strategy.web_dashboard.tasks import TaskManager

    errors = []
    def old_write(self, job):
        payload = {key: value for key, value in job.items() if key != 'result'}
        with closing(sqlite3.connect(self.db_path)) as conn:
            try:
                conn.execute('INSERT INTO czsc_tasks VALUES (?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,payload=excluded.payload',
                    (job['job_id'], stable_hash({'kind': job['kind'], 'spec': job['spec']}), job['status'], json.dumps(payload)))
                conn.commit()
            except sqlite3.IntegrityError as exc:
                errors.append(str(exc))
                raise
    monkeypatch.setattr(TaskManager, '_write', old_write)
    with pytest.raises(RuntimeError, match='did not enter running'):
        probe.task_lifecycle_probe(tmp_path)
    assert errors and all('UNIQUE constraint failed: task_display.job_id' in error for error in errors)
