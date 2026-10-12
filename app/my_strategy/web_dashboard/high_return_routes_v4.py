"""Version-aware dashboard capabilities; legacy account contracts stay isolated."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_agent_consensus import consensus_config as resolve_consensus

V4 = 'czsc_agent_consensus_v4'
NATIVE = 'czsc_high_return_v4_native'
class NativeVersionUnsupported(ValueError):
    pass


CAPABILITIES = {'chart': True, 'scan': True, 'offline_validation': True,
                'legacy_backtest': False, 'legacy_holding': False, 'legacy_forward': False}


def is_native_run(run_id, *, runs_root=None):
    if run_id is None:
        return False
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', str(run_id)):
        raise ValueError('研究运行标识不合法')
    base = Path(runs_root or ARTIFACT_RUNS_ROOT).resolve()
    root = (base / str(run_id)).resolve()
    if not root.is_relative_to(base):
        raise ValueError('研究运行不属于当前结果目录')
    if (root / 'reports/native_v4.json').is_file():
        return True
    metadata = root / 'metadata.json'
    if metadata.is_file() and json.loads(metadata.read_text()).get('config', {}).get('version') == NATIVE:
        return True
    report = root / 'reports/broker_validation.json'
    if report.is_file():
        version = json.loads(report.read_text()).get('config', {}).get('version', '')
        return 'native' in str(version) and 'v4' in str(version)
    return False


def resolve_analysis_consensus(*, research, model_family, model_run_id=None, config=None, version=None):
    """New native defaults never reinterpret a legacy feature envelope."""
    from my_strategy.services import czsc_agent_consensus_v1 as v1, czsc_agent_consensus_v2 as v2, czsc_agent_consensus_v3 as v3
    explicit = version or (config or {}).get('version')
    native_capable = research is True and model_family == 'high_return'
    native = is_native_run(model_run_id)
    legacy_model = False
    if model_run_id is not None:
        # Only a real saved report identifies a schema; names and unknown IDs do
        # not establish compatibility. The selected loader rejects unknown IDs.
        root = (ARTIFACT_RUNS_ROOT / str(model_run_id)).resolve()
        legacy_model = (root / 'reports/high_return.json').is_file() and not native
    if explicit == V4 and (not native_capable or legacy_model):
        raise ValueError('v4仲裁只支持high_return原生研究输入；结构基线与旧模型请选择v1/v2/v3')
    if native and not native_capable:
        raise ValueError('native v4模型只能用于high_return原生研究输入')
    frozen_shapes = [set(module._DEFAULTS)-{'version'} for module in (v1, v2, v3)]
    if explicit is not None or config is not None and set(config) in frozen_shapes:
        values = resolve_consensus(config, version=version)
    elif not native_capable or legacy_model:
        values = resolve_consensus(config, version=v3.CONSENSUS_VERSION)
    elif native:
        values = resolve_consensus(config, version=V4)
    else:
        values = resolve_consensus(config)
    selected = values.get('version', v1.CONSENSUS_VERSION)
    if selected == V4 and not native_capable:
        raise ValueError('v4仲裁只支持high_return原生研究输入')
    if native and selected != V4:
        raise ValueError('native v4模型不能使用旧版本仲裁输入')
    return selected, values


def native_scan_requested(*, model_run_id=None, consensus_version=None, consensus_config=None):
    native = is_native_run(model_run_id)
    explicit = consensus_version or (consensus_config or {}).get('version')
    if native and explicit and explicit != V4:
        raise ValueError('native v4模型不能用于v1/v2/v3扫描，请选择v4仲裁')
    if model_run_id is not None and explicit is None and not native:
        return False
    return native or resolve_consensus(consensus_config, version=consensus_version)['version'] == V4


def reject_native_legacy_operation(operation, *, run_id=None, runs_root=None, consensus_version=None,
                                   research_config=None):
    native_requested = is_native_run(run_id, runs_root=runs_root) or consensus_version == V4
    native_requested |= (research_config or {}).get('version') == NATIVE
    if native_requested:
        raise NativeVersionUnsupported(f'native v4不支持旧{operation}账户/产物协议；图表与扫描可用，实际Broker验证请使用validate_native_v4.py独立入口')


def status_high_return():
    from my_strategy.services.czsc_high_return_research import status_high_return as legacy_status
    from my_strategy.services.czsc_high_return_runtime_v4 import read_run
    from my_strategy.services.czsc_high_return_pipeline_v4 import KIND, RISK_POLICY
    from my_strategy.storage.czsc_results import ResultStore
    result = copy.deepcopy(legacy_status())
    native_runs, native_models, failures = [], [], []
    for item in ResultStore().list(KIND, limit=30):
        try:
            _, report = read_run(item['run_id'])
            checkpoints = [{'name': b['name'], 'available_at': b['available_at'], 'heads': list(b['heads'])}
                           for b in report.get('model_bundles', [])]
            entry = {**item, 'model_family': 'high_return', 'schema_family': 'high_return_native_v4',
                'strategy_version': NATIVE, 'data_end': report['data_end'], 'model_bundles': checkpoints,
                'model_gate': report['model_gate'], 'publication_allowed': False, 'capabilities': dict(CAPABILITIES),
                'independent_window': {'status': 'pending'}, 'pipeline': report.get('pipeline', {}),
                'failures': report.get('failures', [])}
            native_runs.append(entry)
            native_models.append({**entry, 'checkpoints': checkpoints,
                'initial_cash': report['config']['initial_cash'], 'selected_count': report['source_snapshot']['selected_count'],
                'source_snapshot_hash': report['source_snapshot']['snapshot_hash'], 'full_source_universe': False,
                'risk_policy': report['config'].get('risk_policy'),
                'automatic_selection_eligible': report['config'].get('risk_policy') == RISK_POLICY, 'automatic_selection_scope': 'shadow_current_native_policy_only'})
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            failures.append({'run_id': item['run_id'], 'error': str(exc), 'schema_family': 'high_return_native_v4'})
    for entry in result.get('runs', []):
        entry.setdefault('capabilities', {'chart': True, 'scan': True, 'offline_validation': True,
            'legacy_backtest': True, 'legacy_holding': True, 'legacy_forward': True})
    for entry in result.get('models', []):
        entry.setdefault('capabilities', {'chart': True, 'scan': True, 'offline_validation': True,
            'legacy_backtest': True, 'legacy_holding': True, 'legacy_forward': True})
    result['runs'] = sorted([*native_runs, *result.get('runs', [])], key=lambda r: r['run_id'], reverse=True)
    result['models'] = [*native_models, *result.get('models', [])]
    result['failures'] = [*result.get('failures', []), *failures]
    default_version = resolve_consensus()['version']
    result.update(native_models=native_models, default_consensus_version=default_version,
                  native_capabilities=dict(CAPABILITIES))
    if default_version == V4:
        result.update(strategy_version=NATIVE, source_policy='independent_points_union_v4_native')
    return result


def scan_native_high_return(*, symbols=None, start='2026-01-01', end=None, model_run_id=None,
                            model_fold=None, usage_mode='historical', calendar_run_id=None,
                            device='cpu', cpu_workers=4, consensus_version=V4, consensus_config=None,
                            progress=None, check_cancel=None, run_callback=None):
    """Scan final directional opinions with the same full-prefix chart arbiter.

    Every failure stays in coverage. Scan opinions neither establish a holding
    nor apply an entry/exit model to real orders.
    """
    from my_strategy.adapters.czsc_adapter import latest_market_date, list_symbols
    from my_strategy.services.czsc_high_return_runtime_v4 import runtime_overlay
    from my_strategy.services.czsc_agent_consensus import build_consensus
    from my_strategy.services.czsc_research import _json
    from my_strategy.web_dashboard.high_return import _calendar, _prepare, _latest
    if usage_mode not in {'historical', 'production'}:
        raise ValueError('native v4扫描不支持事后模型')
    if model_run_id is not None and not is_native_run(model_run_id):
        raise ValueError('v4扫描需要native v4模型运行，不能混用旧模型')
    values = resolve_consensus(consensus_config, version=V4)
    target = latest_market_date(end)
    if not target:
        raise ValueError('截止日之前没有本地行情')
    dates, calendar_id = _calendar(calendar_run_id)
    symbols = list(dict.fromkeys(symbols or [r['symbol'] for r in list_symbols()]))
    if not symbols:
        raise ValueError('本地行情没有有效股票')
    spec = dict(symbols=symbols, start=start, end=target, model_run_id=model_run_id, model_fold=model_fold,
        usage_mode=usage_mode, model_family='high_return', research=True, calendar_run_id=calendar_id,
        device=device, cpu_workers=cpu_workers, consensus_version=V4, consensus_config=values)
    context = create_run_context(task='czsc-native-v4-shadow-scan', as_of_date=target, config=spec,
        data_version=target, stocks=symbols, start_date=start, end_date=target)
    if run_callback:
        run_callback(context.run_id)
    rows, failures, versions = [], [], {}
    # The CPU-only causal native engine stays in the job's cancellable process.
    # No unbounded nested executor owns model or account state.
    for symbol in symbols:
        if check_cancel:
            check_cancel()
        try:
            item = _prepare(symbol, target, dates)
            if item['data_end'] != target:
                raise ValueError(f'缺少扫描日 {target} 行情，最后日期 {item["data_end"]}')
            overlay = runtime_overlay(item['features'], dates, model_run_id=model_run_id, model_fold=model_fold,
                model_mode=usage_mode, as_of=target, device=device or 'cpu')
            consensus = build_consensus(overlay['rows'], symbol=symbol, config=values, version=V4)
            opinion = consensus['rows'][-1]
            row = _latest(overlay, item)
            row.update(action=opinion['action'], candidate_action=opinion['action'],
                category={'BUY': 'buy', 'SELL': 'exit', 'WAIT': 'watch'}[opinion['action']],
                consensus=opinion, consensus_version=V4, schema_family='high_return_native_v4',
                publication_allowed=False, exit_probability=None)
            rows.append(row)
            versions[symbol] = item['data_version']
        except (ValueError, OSError, RuntimeError) as exc:
            failures.append({'symbol': symbol, 'error': str(exc)})
        if check_cancel:
            check_cancel()
        if progress:
            progress('独立结构量价v4最终方向影子扫描', len(rows)+len(failures), len(symbols), len(failures))
    result = dict(run_id=context.run_id, run_context=context.to_dict(), request=spec,
        rows=rows, failures=failures, input_versions=versions, data_version=stable_hash(versions),
        model_family='high_return', research_version=NATIVE, consensus_version=V4, publication_allowed=False,
        data_range={'start': start, 'end': target}, coverage={'requested': len(symbols), 'success': len(rows), 'failed': len(failures)},
        category_counts={kind: sum(row['category']==kind for row in rows) for kind in ('buy', 'watch', 'exit', 'excluded')},
        model_gate={'passed': False, 'reason': '新独立验证尚未成熟，仅作影子方向'},
        compute_info={'research': True, 'model_family': 'high_return_native_v4', 'cpu_workers': 1,
                      'requested_cpu_workers': cpu_workers, 'requested_device': device or 'cpu',
                      'actual_inference_device': 'cpu', 'actual_structure_device': 'cpu', 'selected_device': 'cpu'})
    context.write_metadata({'status': 'complete', 'coverage': result['coverage'], 'failures': failures})
    return _json(result)
