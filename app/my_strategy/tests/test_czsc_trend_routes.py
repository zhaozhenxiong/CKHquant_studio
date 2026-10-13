"""Saved model families cannot be silently reinterpreted by current routing."""
import pytest

from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, strategy_config
from my_strategy.web_dashboard import trend_routes as routes


@pytest.fixture
def routing_root(monkeypatch, tmp_path):
    for name, report in (("trend", "trend_consensus.json"), ("native", "native_v4.json")):
        path = tmp_path / name / "reports"
        path.mkdir(parents=True)
        (path / report).write_text("{}")
    monkeypatch.setattr(routes, "ARTIFACT_RUNS_ROOT", tmp_path)
    monkeypatch.setattr(routes, "CURRENT_PATH", tmp_path / "current.json")
    monkeypatch.setattr(routes, "_saved_consensus", lambda model_run_id=None, family=STRATEGY_ID: None)
    import my_strategy.web_dashboard.high_return_routes_v4 as old
    monkeypatch.setattr(old, "ARTIFACT_RUNS_ROOT", tmp_path)
    return tmp_path


def test_default_activation_does_not_rewrite_explicit_saved_versions(routing_root):
    routes.CURRENT_PATH.write_text('{"strategy":"czsc_trend_consensus"}')
    version, cfg = routes.resolve_analysis_consensus(research=True, model_family="high_return")
    assert version == STRATEGY_ID
    assert cfg["source_weights"] == {"ma": .4, "structure": .2, "wyckoff": .3}
    for old in ("czsc_agent_consensus_v1", "czsc_agent_consensus_v2", "czsc_agent_consensus_v3", "czsc_agent_consensus_v4"):
        actual, _ = routes.resolve_analysis_consensus(research=True, model_family="high_return", version=old)
        assert actual == old


def test_saved_new_model_can_never_enter_older_consensus(routing_root):
    assert routes.trend_requested(model_run_id="trend")
    with pytest.raises(ValueError, match="不能混入旧"):
        routes.resolve_analysis_consensus(research=True, model_family="high_return", model_run_id="trend", version="czsc_agent_consensus_v4")
    with pytest.raises(ValueError, match="对应的新模型"):
        routes.resolve_analysis_consensus(research=True, model_family="high_return", model_run_id="native", version=STRATEGY_ID)


def test_current_strategy_does_not_capture_structure_or_ma_entrypoints(routing_root):
    routes.CURRENT_PATH.write_text('{"strategy":"czsc_trend_consensus"}')
    for research, family in ((False, "high_return"), (True, "ma_trend"), (True, "dual")):
        version, _ = routes.resolve_analysis_consensus(research=research, model_family=family)
        assert version == "czsc_agent_consensus_v3"
    with pytest.raises(ValueError, match="组合研究"):
        routes.resolve_analysis_consensus(research=False, model_family="high_return", version=STRATEGY_ID)


def test_default_resolves_frozen_selected_policy_instead_of_provisional_config(routing_root, monkeypatch):
    routes.CURRENT_PATH.write_text('{"strategy":"czsc_trend_consensus"}')
    saved = strategy_config()["consensus"]
    saved["sell_threshold"] = .45
    monkeypatch.setattr(routes, "_saved_consensus", lambda model_run_id=None, family=STRATEGY_ID: saved)
    version, actual = routes.resolve_analysis_consensus(research=True, model_family="high_return")
    assert version == STRATEGY_ID
    assert actual["sell_threshold"] == .45


@pytest.mark.parametrize('model_run_id,version', [(None, STRATEGY_ID), ('trend', None)])
def test_partial_live_quality_request_inherits_model_policy_before_runtime(
        routing_root, monkeypatch, model_run_id, version):
    saved = strategy_config()["consensus"]
    saved.update(buy_base_threshold=.4, sell_threshold=.35, margin=.1)
    monkeypatch.setattr(routes, '_saved_consensus', lambda model_run_id=None, family=STRATEGY_ID: saved)
    actual_version, cfg = routes.resolve_analysis_consensus(research=True, model_family='high_return',
        model_run_id=model_run_id, version=version, config={'high_return_threshold': .07})
    assert actual_version == STRATEGY_ID
    validated = routes.validate_consensus_config({'high_return_threshold': .07},
        version=version, model_run_id=model_run_id)
    for values in (cfg, validated):
        assert values['high_return_threshold'] == .07
        assert values['sell_threshold'] == .35
        assert values['buy_base_threshold'] == .4
        assert values['margin'] == .1
    assert saved['high_return_threshold'] == .1


def test_current_strategy_does_not_capture_other_families_partial_task_config(routing_root):
    routes.CURRENT_PATH.write_text('{"strategy":"czsc_trend_consensus"}')
    for family in ('ma_trend', 'dual', 'wyckoff'):
        cfg = routes.validate_consensus_config({}, model_family=family)
        assert cfg['version'] == 'czsc_agent_consensus_v4'


def test_delivered_backtest_is_allowed_but_old_holding_forward_are_rejected(routing_root):
    assert routes.reject_legacy_operation("high_return_backtest", run_id="trend") is None
    for operation in ("holding", "forward", "high_return_train"):
        with pytest.raises(ValueError, match="旧入口"):
            routes.reject_legacy_operation(operation, run_id="trend")


@pytest.mark.parametrize("bad", [{"trend": {"horizon_sessions": True}}, {"trend": {"horizon_sessions": 1.5}},
    {"consensus": {"margin": -1}}, {"consensus": {"source_weights": {"ma": True}}},
    {"consensus": {"source_weights": {"ma": .5}}}])
def test_invalid_contract_parameters_are_not_coerced(bad):
    with pytest.raises(ValueError):
        strategy_config(bad)
