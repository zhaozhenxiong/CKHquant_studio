"""Read-only exit research from a verified, saved real Broker replay.

The caller supplies identities, never shares, cost, inventory or feature values.
Saved hashes and a causal deterministic replay both bind the observed state.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re

import pandas as pd

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now

ARMS = ('reference', 'full', 'entry_only', 'exit_only', 'ma', 'structure', 'wyckoff')


def _root(identity, runs_root):
    if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', identity):
        raise ValueError('回放运行标识不合法')
    root = (runs_root / identity).resolve()
    if not root.is_relative_to(runs_root.resolve()) or root == runs_root.resolve():
        raise ValueError('回放运行越过结果目录')
    return root


def _within(path, root):
    value = Path(path).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError('账户引用越过已保存回放目录')
    return value


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _equal(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-8)
    return left == right


def _validate_prefix(saved, replay, features, calendar, *, symbol, as_of, initial_cash):
    """No future row enters inventory, extrema, fees, or the scoring prefix."""
    daily = saved['daily'].loc[saved['daily'].date <= as_of].reset_index(drop=True)
    ledger = saved['ledger'].loc[saved['ledger'].date <= as_of].reset_index(drop=True)
    for key, expected in (('daily', daily), ('ledger', ledger)):
        actual = pd.DataFrame(replay.get(key, []), columns=expected.columns)
        # CSV represents optional all-empty fields as NaN, whereas the Broker
        # keeps None. Normalize that serialization difference, never values.
        actual = actual.where(actual.notna(), float('nan')).infer_objects()
        expected = expected.where(expected.notna(), float('nan')).infer_objects()
        try:
            pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False,
                rtol=1e-12, atol=1e-8)
        except AssertionError as exc:
            raise ValueError('保存的' + key + '与冻结策略的因果真实回放不一致') from exc
    history = [s for s in saved['holding_snapshots'] if s['date'] <= as_of]
    if not _equal(history, replay.get('holding_snapshots', [])):
        raise ValueError('保存的持仓快照与冻结真实成交前缀不一致')
    if daily.empty or str(daily.iloc[-1].date) != as_of:
        raise ValueError('指定日期没有已观察的账户收盘状态')
    # Independent cash/inventory/cost verification also catches coordinated
    # edits to all saved artifacts, even if their binding hashes were updated.
    cash, shares, cost = float(initial_cash), 0, 0.
    for row in ledger.to_dict('records'):
        count = int(row['shares'])
        if row['symbol'] != symbol or count <= 0 or count != row['shares'] or int(row['position_before']) != shares:
            raise ValueError('真实成交股票或份额链不一致')
        before = shares
        if row['action'] == 'BUY':
            shares += count
            debit = count * float(row['price']) + float(row['fee'])
            cost = (cost * before + debit) / shares
            expected_flow = -debit
        elif row['action'] in {'SELL', 'REDUCE'} and count <= shares:
            shares -= count
            expected_flow = count * float(row['price']) - float(row['fee'])
            if shares == 0: cost = 0.
        else:
            raise ValueError('真实成交方向或剩余份额不一致')
        cash += expected_flow
        for actual, expected in ((row['cash_flow'], expected_flow), (row['cash_after'], cash),
                (row['holding_cost_after'], cost), (row['position_after'], shares)):
            if not _equal(actual, expected): raise ValueError('真实成交含费成本或现金链不一致')
        signal = str(row['signal_date'])
        if signal not in calendar or calendar.index(signal) + 1 >= len(calendar) or calendar[calendar.index(signal)+1] != row['date']:
            raise ValueError('真实成交不是信号后的下一核验开盘')
    last = daily.iloc[-1]
    if not _equal(last.cash, cash) or not _equal(last.shares, shares) or not _equal(last.equity, cash + shares * last.close):
        raise ValueError('账户每日净值与真实成交不一致')
    if shares == 0: return None
    if not history or history[-1]['date'] != as_of:
        raise ValueError('真实有仓账户缺少当日持仓快照')
    current = copy.deepcopy(history[-1])
    history = [s for s in history if s['entry_date'] == current['entry_date'] and s.get('plan_id') == current.get('plan_id')]
    expected_days = [d for d in calendar if current['entry_date'] <= d <= as_of]
    if [s['date'] for s in history] != expected_days:
        raise ValueError('持仓历史缺少核验交易日，不能伪造停牌日状态')
    bars = features.set_index('date')
    for snapshot in history:
        day = snapshot['date']
        if (snapshot['symbol'] != symbol or snapshot['shares'] <= 0 or snapshot['entry_shares'] <= 0
                or snapshot['available_at'] != day + 'T15:00:00+08:00'):
            raise ValueError('真实持仓日期或身份不兼容')
        if day not in bars.index or not _equal(snapshot['bar'], {k: bars.loc[day, k] for k in snapshot['bar']}):
            raise ValueError('持仓快照价格不是对应日期的冻结因果行情')
    if not _equal(current['actual_cost_per_share'], cost):
        raise ValueError('持仓成本不等于真实成交含费成本')
    current['history'] = history
    return current


def _load_reference(replay_run_id, window, arm, symbol, as_of, runs_root):
    from my_strategy.services.czsc_high_return_research import _file_hash, _compact_evaluation, _read_features, _source
    from my_strategy.services.czsc_high_return_forward import _model
    root = _root(replay_run_id, runs_root)
    report_path = root / 'reports/high_return.json'
    report = json.loads(report_path.read_text())
    metadata = json.loads((root / 'metadata.json').read_text())
    if (report.get('run_id') != replay_run_id or report.get('model_family') != 'high_return'
            or report.get('status') != 'completed' or metadata.get('status') != 'complete'):
        raise ValueError('需要已完整完成的高收益实际账户回放')
    model_id = report.get('model_run_id') or replay_run_id
    model_root, model, freeze = _model(model_id)
    if not model_root.is_relative_to(runs_root.resolve()): raise ValueError('模型不属于当前结果目录')
    evaluation_path = _within(report.get('complete_ledger_evaluation_path') or
        str(Path(report['evaluation']['output_root']) / 'evaluation.json'), root)
    evaluation = json.loads(evaluation_path.read_text())
    if _compact_evaluation(evaluation) != report['evaluation']:
        raise ValueError('完整账本报告与运行报告不一致')
    cfg = model['config']
    if (evaluation['model_bundles_hash'] != stable_hash(model['model_bundles'])
            or evaluation['source_snapshot_hash'] != model['source_snapshot']['snapshot_hash']
            or report['config'] != cfg or evaluation.get('cost_multiplier') != 1
            or evaluation.get('validation_bundle_mode')):
        raise ValueError('账户的模型、来源、成本或策略契约不兼容')
    if as_of > report['data_end'] or as_of > local_now().date().isoformat():
        raise ValueError('指定日期尚无实际观察数据')
    selected_window = evaluation['windows'].get(window)
    if not selected_window: raise ValueError('回放窗口不存在')
    bindings = selected_window['ledger_bindings'].get(arm, [])
    selected = [b for b in bindings if b['symbol'] == symbol]
    if len(selected) != 1: raise ValueError('回放臂内没有唯一对应股票账户')
    binding = selected[0]; directory = _within(binding['path'], root)
    if binding['window'] != window or binding['arm'] != arm:
        raise ValueError('账户窗口或策略臂身份不一致')
    files = {'daily':'csv','ledger':'csv','rejections':'csv','episodes':'csv','account':'json',
        'holding_snapshots':'jsonl','entry_diagnostics':'jsonl','entry_gate_diagnostics':'jsonl','exit_policy_diagnostics':'jsonl'}
    for key, extension in files.items():
        path = _within(directory / (key + '.' + extension), root)
        if _file_hash(path) != binding.get(key + '_sha256'):
            raise ValueError('真实账户文件哈希不一致：' + key)
    account = json.loads((directory / 'account.json').read_text())
    if binding['status'] != 'complete' or account['status'] != 'complete':
        raise ValueError('该账户为未观察或失败保留现金，不能提供真实有仓模型状态')
    records, calendar, source, _ = _source(model['source_snapshot']['run_id'], model['data_end'], [symbol])
    frozen = next((r for r in model['dataset_records'] if r['symbol'] == symbol), None)
    if (not frozen or frozen != records[0] or source['report_sha256'] != model['source_snapshot']['report_sha256']
            or source['calendar_sha256'] != model['source_snapshot']['calendar_sha256']
            or evaluation['calendar_hash'] != stable_hash(calendar)):
        raise ValueError('账户冻结特征来源或核验日历改变')
    features = _read_features(frozen, cfg, as_of)
    daily, ledger = pd.read_csv(directory/'daily.csv'), pd.read_csv(directory/'ledger.csv')
    for frame in (daily, ledger):
        if (frame is daily and frame.date.duplicated().any()) or not frame.date.is_monotonic_increasing:
            raise ValueError('账户日期重复或乱序')
    saved = {'daily': daily, 'ledger': ledger, 'holding_snapshots': _jsonl(directory/'holding_snapshots.jsonl')}
    fold = selected_window['fold']
    start = fold.get('start') or fold['test_start']
    from my_strategy.services.czsc_high_return_runtime import replay_high_return
    source_name = arm if arm in {'ma','structure','wyckoff'} else 'union'
    replay = replay_high_return(features, calendar, model_root=model_root, bundles=model['model_bundles'],
        config=cfg, source=source_name, mode='reference' if source_name != 'union' else arm,
        start=start, end=as_of, initial_cash=float(cfg['initial_cash']))
    current = _validate_prefix(saved, replay, features, calendar, symbol=symbol, as_of=as_of, initial_cash=cfg['initial_cash'])
    identity = {'replay_run_id': replay_run_id, 'model_run_id': model_id, 'window': window, 'arm': arm,
        'symbol': symbol, 'source': source_name, 'as_of': as_of, 'account_sha256': binding['account_sha256'],
        'actual_replay_completed_at': report.get('completed_at'), 'replay_data_end': report['data_end'],
        'account_binding_hash': stable_hash(binding), 'evaluation_sha256': _file_hash(evaluation_path),
        'run_report_sha256': _file_hash(report_path), 'model_freeze_hash': freeze['freeze_hash'],
        'source_snapshot_hash': model['source_snapshot']['snapshot_hash'], 'source_feature_sha256': frozen['sha256'],
        'calendar_hash': stable_hash(calendar), 'observed_prefix_replayed_and_verified': True}
    return features, calendar, model_root, model, replay, current, identity


def holding_exit_research(*, replay_run_id, window, arm, symbol, as_of, usage_mode='production', runs_root=None):
    """Compute shadow evidence without writing a run, alias, order or holding."""
    request = {'replay_run_id':replay_run_id,'window':window,'arm':arm,'symbol':symbol,'as_of':str(as_of), 'usage_mode':usage_mode}
    result = {'request': request, 'model_family':'high_return', 'action':'WAIT', 'publication_allowed':False,
        'holding_state_scope':'saved_actual_broker_replay', 'qualification':'observed_historical_development_diagnostic',
        'exit_probability':None, 'exit_advantage':None, 'holding_contract_compatible':False,
        'account_reference':None, 'holding_state':None}
    try:
        if usage_mode not in {'production','historical'} or arm not in ARMS:
            raise ValueError('只支持明确的生产影子或历史回放持仓研究')
        if not re.fullmatch(r'\d{6}\.(SH|SZ|BJ)', symbol) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(as_of)):
            raise ValueError('股票或观察日期不合法')
        pd.Timestamp(as_of)
        features, calendar, root, model, replay, snapshot, identity = _load_reference(
            replay_run_id, window, arm, symbol, str(as_of), Path(runs_root or ARTIFACT_RUNS_ROOT))
        result['account_reference'] = identity
        result['available_holding_dates'] = [s['date'] for s in replay['holding_snapshots']][-60:]
        if snapshot is None:
            return {**result,'status':'flat','reason':'指定已观察日的真实账户为空仓，未生成持仓退出概率'}
        if identity['source'] != 'union':
            return {**result,'status':'incompatible','reason':'该独立来源账户与并集续持退出模型的目标契约不兼容'}
        from my_strategy.services.czsc_high_return_features import entry_feature_row, event_prefix_rows, holding_feature_row
        from my_strategy.services.czsc_high_return_runtime import runtime_overlay, _bundle_for, _predict, EXIT_TARGET
        from my_strategy.services.czsc_high_return_models import HeadSession, load_head
        gates = [g for g in replay['entry_gate_snapshots'] if g['plan_id'] == snapshot['plan_id']]
        if len(gates) != 1: raise ValueError('真实持仓没有唯一的实际空仓入场状态')
        gate = gates[0]; rows = dict(zip(features.date, features.to_dict('records')))
        prefixes = dict(zip(features.date, event_prefix_rows(features, replay['signal_points'], calendar)))
        candidate = next(c for c in replay['candidate_points'] if c['date'] == gate['date'])
        snapshot['entry_values'] = entry_feature_row(rows[gate['date']], gate, candidate, gate['entry_plan'],
            initial_cash=model['config']['initial_cash'], event_prefix=prefixes[gate['date']])
        overlay = runtime_overlay(features, calendar, model_run_id=identity['model_run_id'],
            model_mode=usage_mode, as_of=str(as_of), holding_snapshot=snapshot)
        exit_model = overlay['latest_decision']['exit_model']; probability = exit_model['probability']
        # Rebuild only the already observed extrema for display and the optional
        # independently contracted advantage head. No final episode result is read.
        prefix = {}; values = None; index = {d:i for i,d in enumerate(calendar)}
        for state in snapshot['history']:
            prefix.update(prefixes[state['date']])
            values = holding_feature_row(rows[state['date']], state, entry_values=snapshot['entry_values'], prefix=prefix, market_index=index)
        result.update(holding_contract_compatible=True, holding_state={k:snapshot[k] for k in
            ('symbol','date','entry_date','shares','entry_shares','actual_cost_per_share','cash','equity','close','stop_price','plan_id')},
            observed_holding_features={k:values[k] for k in ('holding_sessions','holding_return','running_peak_return',
                'running_trough_return','peak_giveback','running_close_drawdown','remaining_fraction')},
            prefix_through_date=str(as_of), holding_prefix_source='verified_saved_actual_broker_replay',
            probability_target=EXIT_TARGET, exit_probability=probability)
        bundle = _bundle_for(str(as_of), model['model_bundles'], usage_mode=usage_mode, head='exit', model_root=root)
        if bundle is None:
            historical = _bundle_for(str(as_of), model['model_bundles'], head='exit', model_root=root)
            if historical:
                manifest = load_head(root/historical['heads']['exit']['path'])
                result['exit_model'] = {'checkpoint':historical['name'], 'available_at':historical['available_at'],
                    'actual_training_completed_at':manifest['training_completed_at'], 'status':'unavailable_at_observed_time',
                    'target_contract_hash':manifest['target_contract_hash']}
            return {**result,'status':'model_unavailable','reason':'观察日没有已可用且符合实际训练完成时间的持仓退出头'}
        manifest = load_head(root/bundle['heads']['exit']['path'])
        result['exit_model'] = {'checkpoint':bundle['name'], 'available_at':bundle['available_at'],
            'actual_training_completed_at':manifest['training_completed_at'], 'target_contract_hash':manifest['target_contract_hash'],
            'status':'production_shadow' if usage_mode=='production' else 'historical_shadow'}
        if probability is None:
            return {**result,'status':'input_unavailable','reason':'真实持仓已核验，但冻结持仓输入不合资格，概率保持为空'}
        values['input_eligible'] = True
        advantage, _ = _predict(HeadSession(), values, root, bundle, 'exit_advantage', initial_cash=snapshot['initial_cash'],
            usage_mode=usage_mode, contract_config=model['config'], source='union')
        hard = (snapshot['native_target_weight']==0 or snapshot['stop_price'] is not None and snapshot['close']<=snapshot['stop_price']
            or snapshot['sell_pending'] not in {None,'research_exit_model'}
            or values['holding_sessions']>=model['config']['high_return']['max_holding_sessions'])
        result.update(status='scored', reason='真实回放持仓的退出优势影子评分；未改变账户或下单', exit_advantage=advantage,
            hard_exit_priority=bool(hard), exit_threshold=bundle['policy']['exit_threshold'],
            no_action_margin=model['config']['high_return']['exit_advantage_margin'])
        return result
    except (ValueError, KeyError, OSError, StopIteration, TypeError) as exc:
        return {**result,'status':'incompatible','reason':str(exc) or '缺少完整真实回放持仓资料',
            'exit_probability':None,'exit_advantage':None,'holding_contract_compatible':False}
