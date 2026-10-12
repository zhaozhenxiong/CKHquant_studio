"""Delivered strategy routing with explicit isolation of saved older contracts."""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, DISPLAY_NAME, TRAIN_KIND, strategy_config

CURRENT_PATH = Path(os.environ.get("KHQUANT_CURRENT_STRATEGY_CONFIG",
    str(MY_STRATEGY_ROOT / "configs/current_strategy.json")))
CAPABILITIES = {"chart": True, "scan": True, "offline_validation": True,
                "legacy_backtest": True, "legacy_holding": False, "legacy_forward": False}


def current_strategy():
    if not CURRENT_PATH.is_file():
        return None
    return json.loads(CURRENT_PATH.read_text(encoding="utf-8")).get("strategy")


def _saved_consensus(model_run_id=None):
    from my_strategy.services.czsc_trend_runtime import load_model_run
    _, report = load_model_run(model_run_id)
    return copy.deepcopy(report["config"]["consensus"]) if report else None


def is_trend_run(run_id, *, runs_root=None):
    if run_id is None:
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}", str(run_id)):
        raise ValueError("研究运行标识不合法")
    base = Path(runs_root or ARTIFACT_RUNS_ROOT).resolve()
    root = (base / str(run_id)).resolve()
    if not root.is_relative_to(base):
        raise ValueError("模型运行不属于当前结果目录")
    return (root / "reports/trend_consensus.json").is_file()


def trend_requested(*, model_run_id=None, consensus_version=None, consensus_config=None):
    explicit = consensus_version or (consensus_config or {}).get("version")
    actual = is_trend_run(model_run_id)
    if actual and explicit and explicit != STRATEGY_ID:
        raise ValueError("趋势共识模型须使用对应策略，不能混入旧模型契约")
    if explicit == STRATEGY_ID:
        if model_run_id and not actual:
            raise ValueError("趋势共识策略需要对应的新模型运行")
        return True
    if explicit or model_run_id:
        return actual
    return current_strategy() == STRATEGY_ID


def resolve_analysis_consensus(*, research, model_family, model_run_id=None, config=None, version=None):
    capable = research is True and model_family == "high_return"
    requested = trend_requested(model_run_id=model_run_id, consensus_version=version, consensus_config=config)
    if requested and capable:
        from my_strategy.services.czsc_trend_consensus import consensus_config
        values = {**(_saved_consensus(model_run_id) or {}), **(config or {})}
        return STRATEGY_ID, consensus_config(values)
    if requested and (version == STRATEGY_ID or is_trend_run(model_run_id)):
        raise ValueError("趋势共识策略需要组合研究入口")
    from my_strategy.web_dashboard.high_return_routes_v4 import resolve_analysis_consensus as legacy
    return legacy(research=research, model_family=model_family, model_run_id=model_run_id,
                  config=config, version=version)


def validate_consensus_config(config=None, *, version=None, model_run_id=None, model_family="high_return"):
    explicit = version == STRATEGY_ID or (config or {}).get("version") == STRATEGY_ID or is_trend_run(model_run_id)
    if explicit or (model_family == "high_return" and
            trend_requested(model_run_id=model_run_id, consensus_version=version, consensus_config=config)):
        from my_strategy.services.czsc_trend_consensus import consensus_config
        return consensus_config({**(_saved_consensus(model_run_id) or {}), **(config or {})})
    from my_strategy.services.czsc_agent_consensus import consensus_config as legacy
    return legacy(config, version=version)


def reject_legacy_operation(operation, *, run_id=None, runs_root=None, consensus_version=None, research_config=None):
    if is_trend_run(run_id, runs_root=runs_root) or consensus_version == STRATEGY_ID:
        if operation == "high_return_backtest":
            return
        raise ValueError("趋势共识策略使用对应账户与模型协议；该旧入口不接收新模型")
    from my_strategy.web_dashboard.high_return_routes_v4 import reject_native_legacy_operation
    return reject_native_legacy_operation(operation, run_id=run_id, runs_root=runs_root,
        consensus_version=consensus_version, research_config=research_config)


def status_high_return():
    from my_strategy.web_dashboard.high_return_routes_v4 import status_high_return as legacy
    from my_strategy.storage.czsc_results import ResultStore
    result = copy.deepcopy(legacy())
    models, failures = [], []
    for item in ResultStore().list(TRAIN_KIND, limit=30):
        try:
            from my_strategy.services.czsc_trend_pipeline import read_run
            _, report = read_run(item["run_id"])
            checkpoints = [{"name": b["name"], "available_at": b["available_at"],
                "heads": list(b.get("heads", {}))} for b in report.get("model_bundles", [])]
            models.append({**item, "display_name": DISPLAY_NAME, "model_family": "high_return",
                "schema_family": STRATEGY_ID, "strategy_version": STRATEGY_ID,
                "data_end": report["data_end"], "checkpoints": checkpoints, "model_bundles": checkpoints,
                "model_gate": report.get("model_gate", {"passed": False}), "publication_allowed": False,
                "capabilities": dict(CAPABILITIES), "initial_cash": report["config"]["initial_cash"],
                "selected_count": report.get("source_snapshot", {}).get("selected_count"),
                "full_source_universe": False, "automatic_selection_eligible": report.get("automatic_selection_eligible") is True,
                "automatic_selection_scope": "current_research_account_policy",
                "independent_window": report.get("independent_window", {"status": "pending"}),
                "pipeline": report.get("pipeline", {}), "failures": report.get("failures", [])})
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            failures.append({"run_id": item["run_id"], "schema_family": STRATEGY_ID, "error": str(exc)})
    result["models"] = [*models, *result.get("models", [])]
    result["runs"] = [*models, *result.get("runs", [])]
    result["failures"] = [*failures, *result.get("failures", [])]
    if current_strategy() == STRATEGY_ID:
        selected = json.loads(CURRENT_PATH.read_text(encoding="utf-8"))
        result.update(display_name=DISPLAY_NAME, strategy_version=STRATEGY_ID,
            default_consensus_version=STRATEGY_ID, trend_capabilities=dict(CAPABILITIES),
            current_model_run_id=selected.get("model_run_id"),
            model_selection="configured_run_auto_prior_heads" if selected.get("model_run_id") else "automatic_qualified_runs")
    return result
