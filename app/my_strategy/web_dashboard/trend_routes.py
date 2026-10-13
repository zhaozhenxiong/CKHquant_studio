"""Route every current entry point to the same model and account contract."""
from __future__ import annotations
import copy
import importlib
import json
import os
from pathlib import Path
from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.services.czsc_trend_contracts import STRATEGY_ID, DISPLAY_NAME, strategy_config

CURRENT_PATH = Path(os.environ.get('KHQUANT_CURRENT_STRATEGY_CONFIG', str(MY_STRATEGY_ROOT / 'configs/current_strategy.json')))
TREND_FAMILIES = {STRATEGY_ID: 'czsc_trend'}
CAPABILITIES = {'chart': True, 'scan': True, 'offline_validation': True, 'holding': True}


def family_module(family, component):
    if family != STRATEGY_ID:
        raise ValueError('该模型身份已退役，请使用当前趋势共识体系')
    return importlib.import_module(f'my_strategy.services.czsc_trend_{component}')


def current_strategy():
    return STRATEGY_ID


def is_trend_run(run_id, *, runs_root=None):
    if run_id is None:
        return False
    from my_strategy.services.czsc_trend_pipeline import read_run
    if runs_root is not None and Path(runs_root).resolve() != ARTIFACT_RUNS_ROOT.resolve():
        path = (Path(runs_root) / str(run_id) / 'reports/trend_consensus.json').resolve()
        if not path.is_relative_to(Path(runs_root).resolve()) or not path.is_file():
            return False
        return json.loads(path.read_text()).get('strategy_version') == STRATEGY_ID
    try:
        read_run(run_id)
        return True
    except (ValueError, OSError, KeyError):
        return False


def trend_family(*, model_run_id=None, consensus_version=None, consensus_config=None):
    explicit = consensus_version or (consensus_config or {}).get('version')
    if explicit is not None and explicit != STRATEGY_ID:
        raise ValueError('该策略身份已退役，请使用当前趋势共识体系')
    if model_run_id:
        from my_strategy.services.czsc_trend_pipeline import read_run
        read_run(model_run_id)
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
    from my_strategy.services.czsc_trend_consensus import consensus_config
    saved = None
    if model_run_id:
        from my_strategy.services.czsc_trend_pipeline import read_run
        _, report = read_run(model_run_id)
        saved = report['config']['consensus']
    return consensus_config({**(saved or {}), **(config or {})})


def resolve_analysis_consensus(*, research, model_family, model_run_id=None, config=None, version=None):
    values = validate_consensus_config(config, version=version, model_run_id=model_run_id, model_family=model_family)
    return STRATEGY_ID, values


def reject_legacy_operation(operation, **kwargs):
    raise ValueError('旧研究操作已退役，请使用模型训练中心')


def status_high_return():
    from my_strategy.services.model_training_center import TrainingCenter
    return TrainingCenter().status()
