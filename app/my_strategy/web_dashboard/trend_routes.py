"""Route the public current family to account-free reference positions."""
from __future__ import annotations
import importlib
import json
import os
from pathlib import Path
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT

STRATEGY_ID = "reference_position_v1"
DISPLAY_NAME = "参考买卖点与每股仓位"
CURRENT_PATH = Path(os.environ.get('KHQUANT_CURRENT_STRATEGY_CONFIG', str(MY_STRATEGY_ROOT / 'configs/current_strategy.json')))
TREND_FAMILIES = {STRATEGY_ID: 'reference', 'trend_consensus': 'czsc_trend'}
CAPABILITIES = {'chart': True, 'scan': True, 'offline_validation': True, 'holding': False,
                'reference_analysis': True, 'reference_position_path': True,
                'default_analysis_mode': 'reference', 'analysis_account_simulated': False}


def family_module(family, component):
    if family == STRATEGY_ID and component == "contracts":
        return importlib.import_module('my_strategy.services.reference_runtime')
    if family == 'trend_consensus':
        return importlib.import_module(f'my_strategy.services.czsc_trend_{component}')
    raise ValueError('该模型身份不支持当前参考分析')


def current_strategy():
    return STRATEGY_ID


def is_trend_run(run_id, *, runs_root=None):
    if run_id is None:
        return False
    try:
        from my_strategy.services.reference_training import load_reference_run
        load_reference_run(run_id, runs_root=runs_root or ARTIFACT_RUNS_ROOT)
        return True
    except (ValueError, OSError, KeyError):
        return False


def trend_family(*, model_run_id=None, consensus_version=None, consensus_config=None):
    explicit = consensus_version or (consensus_config or {}).get('version')
    if explicit is not None and explicit != STRATEGY_ID:
        raise ValueError('旧成交研究策略仅保留历史诊断；请使用参考仓位体系')
    if model_run_id:
        from my_strategy.services.reference_training import load_reference_run
        load_reference_run(model_run_id)
    return STRATEGY_ID


def trend_requested(**kwargs):
    return trend_family(**kwargs) == STRATEGY_ID


def strategy_adapter(**kwargs):
    trend_family(**kwargs)
    return importlib.import_module('my_strategy.web_dashboard.trend_strategy')


def validate_consensus_config(config=None, *, version=None, model_run_id=None, model_family='current'):
    if model_family != 'current':
        raise ValueError('旧模型族已退役，请使用当前训练入口')
    trend_family(model_run_id=model_run_id, consensus_version=version, consensus_config=config)
    from my_strategy.services.reference_runtime import validate_position_policy
    saved = {}
    if model_run_id:
        from my_strategy.services.reference_training import load_reference_run
        _, report = load_reference_run(model_run_id)
        saved = report.get('position_policy') or report.get('config', {}).get('position_policy') or report.get('config', {}).get('position') or {}
    return validate_position_policy({**saved, **(config or {})})


def resolve_analysis_consensus(*, research, model_family, model_run_id=None, config=None, version=None):
    values = validate_consensus_config(config, version=version, model_run_id=model_run_id, model_family=model_family)
    return STRATEGY_ID, values


def reject_legacy_operation(operation, **kwargs):
    raise ValueError('旧研究操作已退役，请使用模型训练中心')


def status_high_return():
    from my_strategy.services.model_training_center import TrainingCenter
    return TrainingCenter().status()
