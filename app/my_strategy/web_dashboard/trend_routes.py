"""Delivered strategy routing with explicit isolation of saved older contracts."""
from __future__ import annotations

import copy
import importlib
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
TREND_FAMILIES = {STRATEGY_ID: "czsc_trend", "czsc_trend_consensus_v2": "czsc_trend_v2"}


def family_module(family, component):
    if family not in TREND_FAMILIES:
        raise ValueError("未知趋势策略版本")
    return importlib.import_module(f"my_strategy.services.{TREND_FAMILIES[family]}_{component}")


def current_strategy():
    if not CURRENT_PATH.is_file():
        return None
    return json.loads(CURRENT_PATH.read_text(encoding="utf-8")).get("strategy")


def _saved_consensus(model_run_id=None, family=STRATEGY_ID):
    _, report = family_module(family, "runtime").load_model_run(model_run_id)
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


def trend_family(*, model_run_id=None, consensus_version=None, consensus_config=None):
    explicit = consensus_version or (consensus_config or {}).get("version")
    actual = is_trend_run(model_run_id)
    identity = None
    if actual:
        report = json.loads((ARTIFACT_RUNS_ROOT / model_run_id / "reports/trend_consensus.json").read_text())
        identity = report.get("strategy_version", STRATEGY_ID)
        if identity not in TREND_FAMILIES:
            raise ValueError("模型运行不是已登记的趋势策略版本")
    if actual and explicit and explicit != identity:
        raise ValueError("趋势共识模型须使用对应策略，不能混入旧模型契约")
    if explicit in TREND_FAMILIES:
        if model_run_id and not actual:
            raise ValueError("趋势共识策略需要对应的新模型运行")
        return explicit
    if explicit or model_run_id:
        return identity
    current = current_strategy()
    return current if current in TREND_FAMILIES else None


def trend_requested(**kwargs):
    return trend_family(**kwargs) is not None


def strategy_adapter(*, model_run_id=None, consensus_version=None, consensus_config=None):
    family = trend_family(model_run_id=model_run_id, consensus_version=consensus_version,
                          consensus_config=consensus_config)
    name = "trend_v2_strategy" if family == "czsc_trend_consensus_v2" else "trend_strategy"
    if family is None:
        raise ValueError("请求未选择趋势策略")
    return importlib.import_module("my_strategy.web_dashboard." + name)


def resolve_analysis_consensus(*, research, model_family, model_run_id=None, config=None, version=None):
    capable = research is True and model_family == "high_return"
    family = trend_family(model_run_id=model_run_id, consensus_version=version, consensus_config=config)
    if family and capable:
        values = {**(_saved_consensus(model_run_id, family) or {}), **(config or {})}
        return family, family_module(family, "consensus").consensus_config(values)
    if family and (version in TREND_FAMILIES or is_trend_run(model_run_id)):
        raise ValueError("趋势共识策略需要组合研究入口")
    from my_strategy.web_dashboard.high_return_routes_v4 import resolve_analysis_consensus as legacy
    return legacy(research=research, model_family=model_family, model_run_id=model_run_id,
                  config=config, version=version)


def validate_consensus_config(config=None, *, version=None, model_run_id=None, model_family="high_return"):
    explicit = version in TREND_FAMILIES or (config or {}).get("version") in TREND_FAMILIES or is_trend_run(model_run_id)
    if explicit or (model_family == "high_return" and
            trend_requested(model_run_id=model_run_id, consensus_version=version, consensus_config=config)):
        family = trend_family(model_run_id=model_run_id, consensus_version=version, consensus_config=config)
        return family_module(family, "consensus").consensus_config({**(_saved_consensus(model_run_id, family) or {}), **(config or {})})
    from my_strategy.services.czsc_agent_consensus import consensus_config as legacy
    return legacy(config, version=version)


def reject_legacy_operation(operation, *, run_id=None, runs_root=None, consensus_version=None, research_config=None):
    if is_trend_run(run_id, runs_root=runs_root) or consensus_version in TREND_FAMILIES:
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
    items = [(family, item) for family in TREND_FAMILIES for item in ResultStore().list(family_module(family, "contracts").TRAIN_KIND, limit=30)]
    for family, item in items:
        try:
            _, report = family_module(family, "pipeline").read_run(item["run_id"])
            name = family_module(family, "contracts").DISPLAY_NAME
            checkpoints = [{"name": b["name"], "available_at": b["available_at"],
                "heads": list(b.get("heads", {}))} for b in report.get("model_bundles", [])]
            models.append({**item, "display_name": name, "model_family": "high_return",
                "schema_family": family, "strategy_version": family,
                "data_end": report["data_end"], "checkpoints": checkpoints, "model_bundles": checkpoints,
                "model_gate": report.get("model_gate", {"passed": False}), "publication_allowed": False,
                "capabilities": dict(CAPABILITIES), "initial_cash": report["config"]["initial_cash"],
                "selected_count": report.get("source_snapshot", {}).get("selected_count"),
                "full_source_universe": False, "automatic_selection_eligible": report.get("automatic_selection_eligible") is True,
                "automatic_selection_scope": "current_research_account_policy",
                "independent_window": report.get("independent_window", {"status": "pending"}),
                "pipeline": report.get("pipeline", {}), "failures": report.get("failures", [])})
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            failures.append({"run_id": item["run_id"], "schema_family": family, "error": str(exc)})
    result["models"] = [*models, *result.get("models", [])]
    result["runs"] = [*models, *result.get("runs", [])]
    result["failures"] = [*failures, *result.get("failures", [])]
    if current_strategy() in TREND_FAMILIES:
        selected = json.loads(CURRENT_PATH.read_text(encoding="utf-8"))
        family = selected["strategy"]
        result.update(display_name=family_module(family, "contracts").DISPLAY_NAME, strategy_version=family,
            default_consensus_version=family, trend_capabilities=dict(CAPABILITIES),
            current_model_run_id=selected.get("model_run_id"),
            model_selection="configured_run_auto_prior_heads" if selected.get("model_run_id") else "automatic_qualified_runs")
    return result
