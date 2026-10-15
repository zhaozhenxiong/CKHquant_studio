"""The release probe executes real code while all fixtures remain isolated."""
from pathlib import Path
import importlib.util


def test_v1_installation_probe_uses_real_bounded_isolated_components():
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
