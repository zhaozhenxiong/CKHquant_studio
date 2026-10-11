"""Signal-time-pinned MA exit policies and exit-bound entry target generation.

Each holding keeps its candidate-time checkpoint and causal Broker prefix.
The recipe and candidate bindings are archived with the training artifacts.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import pandas as pd
from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_high_return_features import holding_feature_row, L1_MA_EXIT_COLUMNS
from my_strategy.services.czsc_high_return_labels import label_config
from my_strategy.services.czsc_high_return_models import HeadSession, load_head, schema_hash
from my_strategy.services.czsc_high_return_runtime import _bundle_for, _head_path, _predict, _validate_compatibility
from my_strategy.services.czsc_signal_points import build_signal_points, make_point_decisions
from my_strategy.services.czsc_trade_episodes import EPISODE_VERSION, _settings, build_episode_dataset

ALIGNMENT_VERSION = 'czsc_ma_entry_positive_signal_pinned_learned_exit_v2'
EXIT_TARGET = 'remaining_net_sell_proceeds_exit_next_open_minus_frozen_continue_over_current_market_value'


def make_signal_pinned_ma_exit_policy(features, calendar, model_root, bundles, *, config,
        points, source_snapshot_hash=None, session=None, usage_mode='historical', expected_exit_contract_hash=None,
        exit_training_calendar_hash=None, exit_training_calendar_dates=None,
        expected_source_snapshot_hash=None, expected_calendar_hash=None):
    """Freeze exit checkpoint at plan signal, never at fill/current holding date.

    Candidates retain rejected plans. No earlier head is explicit v0 fallback;
    corrupt/incompatible heads raise instead of masquerading as unavailable.
    Every callback.new_replay() creates fresh prefix state for the same immutable
    recipe/bindings; no new model is trained or moved by this function.
    """
    if usage_mode not in {'historical', 'production'}:
        raise ValueError('unknown alignment usage mode')
    frame = features.copy()
    frame['date'] = pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    if frame.symbol.nunique() != 1 or frame.date.duplicated().any() or not frame.date.is_monotonic_increasing:
        raise ValueError('pinned policy needs one unique ascending source prefix')
    dates = [str(day) for day in calendar]
    if dates != sorted(set(dates)) or not set(frame.date).issubset(dates):
        raise ValueError('invalid verified calendar or source dates')
    if source_snapshot_hash and expected_source_snapshot_hash and source_snapshot_hash != expected_source_snapshot_hash:
        raise ValueError('conflicting declared bootstrap source snapshots')
    source_snapshot_hash = expected_source_snapshot_hash or source_snapshot_hash
    if not source_snapshot_hash:
        raise ValueError('a declared original exit-training source snapshot is required')
    if exit_training_calendar_hash and expected_calendar_hash and exit_training_calendar_hash != expected_calendar_hash:
        raise ValueError('conflicting declared bootstrap calendars')
    expected_calendar_hash = expected_calendar_hash or exit_training_calendar_hash or stable_hash(dates)
    if expected_calendar_hash != stable_hash(dates) and exit_training_calendar_dates is None:
        frozen_path = Path(model_root)/'data_snapshots/calendar.json'
        if not frozen_path.is_file():
            raise ValueError('extended execution calendar requires frozen bootstrap calendar dates')
        frozen = json.loads(frozen_path.read_text())
        if (frozen.get('verified') is not True
                or frozen.get('source', {}).get('snapshot_hash') != source_snapshot_hash):
            raise ValueError('invalid original bootstrap calendar/source evidence')
        exit_training_calendar_dates = frozen['dates']
    if exit_training_calendar_dates is not None:
        original_dates = [str(d) for d in exit_training_calendar_dates]
        if stable_hash(original_dates) != expected_calendar_hash or dates[:len(original_dates)] != original_dates:
            raise ValueError('execution calendar changed frozen bootstrap prefix')
    session = session or HeadSession()
    root = Path(model_root).resolve()
    catalog = []
    # Ignore entry heads when constructing the exit-only policy identity.
    for item in bundles:
        if not item.get('heads', {}).get('ma_exit'):
            continue
        bundle = copy.deepcopy(item)
        bundle['heads'] = {'ma_exit': bundle['heads']['ma_exit']}
        path = _head_path(root, bundle, 'ma_exit')
        manifest = load_head(path)
        if (manifest['head'] != 'ma_exit' or manifest['columns'] != list(L1_MA_EXIT_COLUMNS)
                or manifest['feature_schema_hash'] != schema_hash(L1_MA_EXIT_COLUMNS, 'ma_exit')
                or manifest['target_contract']['target'] != EXIT_TARGET
                or manifest['available_at'] != bundle['available_at']
                or manifest['manifest_sha256'] != bundle['heads']['ma_exit']['manifest_sha256']
                or manifest['fold'].get('name') != bundle['name']):
            raise ValueError('exit-only checkpoint identity/schema/availability mismatch')
        if (manifest['target_contract']['source_snapshot_hash'] != source_snapshot_hash
                or manifest['target_contract']['calendar_hash'] != expected_calendar_hash):
            raise ValueError('exit catalog bootstrap source/calendar mismatch')
        _validate_compatibility(manifest['target_contract'], contract_config=config, source='ma',
            initial_cash=config.get('initial_cash', 100000.))
        if not expected_exit_contract_hash or manifest['target_contract_hash'] != expected_exit_contract_hash:
            raise ValueError('bootstrap exit target missing or changed')
        fold = manifest['fold']
        if not (manifest['train_label_end'] <= fold['train_end'] < fold['calibration_start']
                <= manifest['calibration_label_end'] <= fold['calibration_end'] < fold['policy_start']
                <= fold['policy_end'] <= fold['validation_end'] == manifest['available_at']):
            raise ValueError('exit checkpoint immature train/calibration labels')
        threshold = float(bundle.get('policy', {}).get('ma_exit_threshold',
            config.get('signal_points', {}).get('ma_exit_threshold', .55)))
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('invalid frozen exit threshold')
        catalog.append((bundle, manifest, str(path), threshold))
    names = [item[0]['name'] for item in catalog]
    if len(names) != len(set(names)):
        raise ValueError('duplicate exit checkpoint identities')
    plan_rows = make_point_decisions(frame, points, source='ma', config=config, include_unselected=True)
    bindings = {}
    bundles_only = [b for b, _, _, _ in catalog]
    for decision in plan_rows:
        plan = decision.get('entry_plan')
        if not plan:
            continue
        day = plan['signal_date']
        eligible = bundles_only
        if usage_mode == 'production':
            cutoff = pd.Timestamp(day+'T15:00:00+08:00')
            eligible = [b for b,m,p,t in catalog if pd.Timestamp(m['training_completed_at']) <= cutoff]
        selected = _bundle_for(day, eligible, head='ma_exit', model_root=root, usage_mode='historical')
        binding = {'version': ALIGNMENT_VERSION, 'plan_id': plan['plan_id'], 'plan_sha256':stable_hash(plan),
            'opportunity_id': plan['opportunity_id'],
            'symbol': plan['symbol'], 'candidate_signal_date': day, 'binding_clock': day+'T15:00:00+08:00',
            'mode': 'explicit_rule_v0_no_earlier_exit_head', 'checkpoint': None,
            'exit_contract_hash': None, 'available_at': None, 'actual_training_completed_at': None,
            'threshold': None, 'usage_mode': usage_mode, 'checkpoint_upgrade': False,
            'fallback_policy': 'source_rule_risk_max60', 'plan_status': plan['status']}
        if selected:
            bundle, manifest, path, threshold = next(row for row in catalog if row[0]['name'] == selected['name'])
            if not (manifest['available_at'] < day and manifest['train_label_end'] < day
                    and manifest['calibration_label_end'] < day):
                raise ValueError('pinned exit checkpoint is not strictly earlier than candidate')
            binding.update(mode='signal_time_pinned_ma_exit', checkpoint=bundle['name'],
                exit_contract_hash=manifest['target_contract_hash'], available_at=manifest['available_at'],
                actual_training_completed_at=manifest['training_completed_at'], threshold=threshold,
                manifest_sha256=manifest['manifest_sha256'], checkpoint_sha256=manifest['checkpoint_sha256'],
                export_sha256=manifest.get('export_sha256'), model_path=path,
                exit_target_source_snapshot_hash=manifest['target_contract']['source_snapshot_hash'],
                exit_target_bootstrap_continuation_policy_hash=manifest['target_contract']['continuation_policy_hash'],
                train_label_end=manifest['train_label_end'], calibration_label_end=manifest['calibration_label_end'])
        binding['binding_hash'] = stable_hash(binding)
        if plan['plan_id'] in bindings:
            raise ValueError('duplicate plan identity')
        bindings[plan['plan_id']] = (binding, selected)
    recipe = {'version': ALIGNMENT_VERSION, 'source': 'ma', 'candidate_policy': 'provided_causal_ma_candidate_plans',
        'clock': 'candidate_signal_close', 'holding_checkpoint_upgrade': False,
        'branch_prefix': 'fresh_per_execution_replay', 'fallback': 'explicit_rule_v0_no_earlier_exit_head',
        'holding_input_unavailable': 'abstain_retaining_frozen_source_risk_max60',
        'source_snapshot_hash':source_snapshot_hash, 'expected_exit_target_contract_hash':expected_exit_contract_hash,
        'exit_training_calendar_hash':expected_calendar_hash, 'calendar_hash': stable_hash(dates),
        'calendar_compatibility':'identical' if expected_calendar_hash==stable_hash(dates) else 'verified_original_prefix_then_appended',
        'signal_points_config_hash':points['config_hash'], 'signal_points_config':copy.deepcopy(points['config']),
        'strategy_hash': stable_hash(_settings(config)),
        'label_config': label_config(config), 'usage_mode': usage_mode,
        'exit_catalog': [{'name':b['name'], 'available_at':m['available_at'], 'manifest_sha256':m['manifest_sha256'],
            'checkpoint_sha256':m['checkpoint_sha256'], 'export_sha256':m.get('export_sha256'),
            'target_contract_hash':m['target_contract_hash'], 'threshold':t,
            'actual_training_completed_at':m['training_completed_at']} for b,m,p,t in catalog]}
    recipe_hash = stable_hash(recipe)
    rows = {row['date']:row for row in frame.to_dict('records')}
    market_index = {day:i for i,day in enumerate(dates)}
    next_market = dict(zip(dates[:-1],dates[1:]))
    def fresh():
        prefixes, diagnostics = {}, []
        def callback(snapshot):
            plan_id = snapshot.get('plan_id')
            if plan_id not in bindings:
                raise ValueError('actual holding has no frozen candidate-time exit binding')
            binding, bundle = bindings[plan_id]
            if (snapshot['symbol'] != binding['symbol']
                    or snapshot['entry_date'] != next_market.get(binding['candidate_signal_date'])
                    or snapshot['date'] < snapshot['entry_date']
                    or pd.Timestamp(snapshot['available_at']) != pd.Timestamp(snapshot['date']+'T15:00:00+08:00')):
                raise ValueError('actual holding and candidate-time binding mismatch')
            key = (snapshot['symbol'], snapshot['entry_date'], plan_id)
            prefix = prefixes.setdefault(key, {})
            day = snapshot['date']
            if day not in rows or prefix.get('through_date', '') >= day:
                raise ValueError('holding prefix reused or source date missing')
            prefix['through_date'] = day
            values = holding_feature_row(rows[day], snapshot, entry_values={}, prefix=prefix, market_index=market_index)
            values['input_eligible'] = bool(rows[day].get('model_input_eligible', False))
            response = {'head':'holding_exit', 'contract_hash':recipe_hash, 'probability':None, 'request_exit':False,
                'apply':False, 'research_arm':True, 'holding_contract_compatible':True,
                'status':binding['mode'], 'alignment_recipe_hash':recipe_hash,
                'candidate_exit_binding':copy.deepcopy(binding), 'prefix_through_date':day,
                'holding_prefix_source':'actual_broker_daily_close_snapshots'}
            if bundle is not None:
                probability, manifest = _predict(session, values, root, bundle, 'ma_exit',
                    initial_cash=snapshot['initial_cash'], contract_config=config, source='ma', usage_mode=usage_mode)
                identity_pairs = {'manifest_sha256':'manifest_sha256', 'checkpoint_sha256':'checkpoint_sha256',
                    'export_sha256':'export_sha256', 'target_contract_hash':'exit_contract_hash', 'available_at':'available_at',
                    'training_completed_at':'actual_training_completed_at'}
                if any(manifest.get(mkey) != binding.get(bkey) for mkey,bkey in identity_pairs.items()):
                    raise ValueError('inference head drifted from frozen candidate exit binding')
                response.update(probability=probability, apply=probability is not None,
                    request_exit=probability is not None and probability >= binding['threshold'],
                    contract_hash=manifest['target_contract_hash'], threshold=binding['threshold'],
                    checkpoint=binding['checkpoint'], status='pinned_ma_actual_holding_exit' if probability is not None
                    else 'pinned_ma_exit_input_unavailable_abstain')
            diagnostics.append({**copy.deepcopy(response),'date':day,'entry_date':snapshot['entry_date']})
            return response
        callback.diagnostics = diagnostics
        callback.session = session
        callback.new_replay = fresh
        callback.recipe = copy.deepcopy(recipe)
        callback.recipe_hash = recipe_hash
        callback.continuation_policy_hash = stable_hash({'version':EPISODE_VERSION,'source':'ma',
            'points_config':points['config_hash'],'strategy':_settings(config),
            'protocol':label_config(config),'exit_policy':recipe_hash})
        callback.bindings = {key:copy.deepcopy(value[0]) for key,value in bindings.items()}
        return callback
    return fresh()


def build_exit_bound_ma_entry_dataset(features, calendar, *, model_root, bundles, config,
        source_snapshot_hash, start=None, session=None, points=None, expected_exit_contract_hash=None,
        reference_callback=None, expected_calendar_hash=None):
    """Generate broad MA entry labels under chronologically available exit heads.

    The entry model cannot condition its own training candidate population.
    Final four-source episodes and MA runtime replay use the same pinned factory.
    """
    if expected_calendar_hash is not None and expected_calendar_hash != stable_hash([str(d) for d in calendar]):
        raise ValueError('new entry label stage calendar differs from frozen bootstrap calendar')
    broad = build_signal_points(features, config)
    # Recompute the full canonical raw population; supplied model scores, veto
    # flags, deleted opportunities or extra lineage are never accepted here.
    if points is not None and stable_hash(points) != stable_hash(broad):
        raise ValueError('MA-entry target points cannot depend on an entry model or altered candidate population')
    policy = make_signal_pinned_ma_exit_policy(features, calendar, model_root, bundles,
        config=config, points=broad, source_snapshot_hash=source_snapshot_hash, session=session,
        expected_exit_contract_hash=expected_exit_contract_hash,expected_calendar_hash=expected_calendar_hash)
    for binding in policy.bindings.values():
        if binding.get('exit_target_source_snapshot_hash') not in {None, source_snapshot_hash}:
            raise ValueError('exit bootstrap target source differs from new candidate source')
    dataset = build_episode_dataset(features, calendar, config, points=broad, start=start,
        initial_cash=config.get('initial_cash',100000.), source_snapshot_hash=source_snapshot_hash,
        sources=('ma',), reference_callback=reference_callback,
        exit_policies={'ma':policy}, exit_policy_hashes={'ma':policy.recipe_hash})
    bind_ma_exit_lineage(dataset,policy)
    dataset['contracts']['ma']['ma_entry']['candidate_policy'] = 'broad_causal_ma_rule_no_entry_head'
    for contract in dataset['contracts']['ma'].values():
        body = {k:v for k,v in contract.items() if k!='contract_hash'}
        contract['contract_hash'] = stable_hash(body)
    _sync_contract_attrs(dataset)
    return dataset


def _sync_contract_attrs(dataset):
    for kind,head in [('entries','entry'),('exits','exit')]:
        dataset[kind].attrs['source_contracts'] = {s:c[head] for s,c in dataset['contracts'].items()}


def bind_ma_exit_lineage(dataset, policy):
    """Attach only MA-row lineage for both aligned stage and final four paths."""
    if any(contract['continuation_policy_hash'] != policy.continuation_policy_hash
            for contract in dataset['contracts']['ma'].values()):
        raise ValueError('MA dataset was not generated under this pinned exit policy')
    by_signal = {binding['candidate_signal_date']:binding for binding in policy.bindings.values()}
    by_episode = {row['episode_id']:policy.bindings[row['plan_id']] for row in dataset['episodes'].to_dict('records') if row['source']=='ma'}
    for kind in ('entries','exits','episodes'):
        frame = dataset[kind]
        bindings = [(by_signal[row['date']] if kind=='entries' else by_episode[row['episode_id']])
            if row['source']=='ma' else None for row in frame.to_dict('records')]
        ma_rows = [row for row in frame.to_dict('records') if row['source']=='ma']
        if any(row['continuation_policy_hash'] != policy.continuation_policy_hash
                or row['source_snapshot_hash'] != policy.recipe['source_snapshot_hash'] for row in ma_rows):
            raise ValueError('MA row target/source mismatch for pinned exit binding')
        if kind=='entries' and any(row['candidate_plan_hash'] != by_signal[row['date']]['plan_sha256'] for row in ma_rows):
            raise ValueError('MA candidate plan changed after exit binding')
        frame['ma_exit_binding_json'] = [json.dumps(value,sort_keys=True) if value else None for value in bindings]
        frame['ma_exit_binding_hash'] = [value['binding_hash'] if value else None for value in bindings]
        frame['ma_exit_binding_mode'] = [value['mode'] if value else None for value in bindings]
        frame['alignment_recipe_hash'] = [policy.recipe_hash if value else None for value in bindings]
    for key, contract in dataset['contracts']['ma'].items():
        body = {k:v for k,v in contract.items() if k != 'contract_hash'}
        body.update(alignment_version=ALIGNMENT_VERSION,
            exit_binding_clock='candidate_signal_close', holding_checkpoint_upgrade=False,
            exit_alignment_recipe_hash=policy.recipe_hash)
        if key == 'ma_entry':body['label_version'] = ALIGNMENT_VERSION
        dataset['contracts']['ma'][key] = {**body,'contract_hash':stable_hash(body)}
    _sync_contract_attrs(dataset)
    dataset['alignment_recipe'] = copy.deepcopy(policy.recipe)
    dataset['candidate_exit_bindings'] = copy.deepcopy(policy.bindings)
    return dataset
