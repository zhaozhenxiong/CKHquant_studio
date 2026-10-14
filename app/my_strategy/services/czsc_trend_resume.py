"""Continue one interrupted current training run without changing its generators.

The frozen pipeline still executes its original algorithm.  This adapter only
substitutes verified, completed assets at its preparation and fit boundaries.
It never prepares raw/native inputs, imports another model family, or relaxes a
population, label, qualification or final completion check.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid

import pandas as pd
import pyarrow.parquet as pq

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT
from my_strategy.core.run_context import stable_hash
from my_strategy.core.tz import local_now
from my_strategy.services.czsc_trend_support import _file_hash, _save, resolve_frozen_asset


_THREAD_LOCK = threading.Lock()
_REQUIRED_STOCK = {'entries.parquet', 'exits.parquet', 'episodes.parquet',
                   'reference_account.json', 'candidate_clones.json', 'holding_pairs.json'}
_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}\Z')


class TrainingResumeCancelled(KeyboardInterrupt):
    pass


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _inside(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('training resume asset escapes its frozen run')
    return resolved


def _asset(root, value, sha, proof):
    path = _inside(root, value)
    if not path.is_file() or _file_hash(path) != sha:
        raise ValueError(f'training resume asset SHA mismatch: {path.name}')
    proof[str(path)] = sha
    return path


def _tracked(root, relative, proof):
    path = _inside(root, relative)
    value = _read(path)
    proof[str(path)] = _file_hash(path)
    return value


def _emit(progress, completed, total, **details):
    payload = dict(stage='training_resume_verification', completed=completed, total=total, **details)
    if progress:
        progress(payload)
    else:
        print('RESUME_PROGRESS ' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')), flush=True)


def _derive_config(initial, selection, trends, ma, records_hash):
    cfg = copy.deepcopy(initial)
    choice = selection.get('selected')
    if choice:
        variant = choice['variant']
        if choice not in selection.get('variants', []) or choice.get('eligible_for_selection') is not True:
            raise ValueError('frozen source policy selection is not a registered qualified choice')
        cfg['trend']['candidate_cross_threshold'] = variant['cross']
        cfg['consensus'].update(buy_base_threshold=variant['buy_base'],
                                sell_threshold=variant['sell'], margin=variant['margin'])
    cfg['candidate_policy_hash'] = stable_hash({
        'candidate': cfg['candidate_policy'], 'risk_policy': cfg['risk_policy'],
        'trend_candidate': {k: cfg['trend'][k] for k in ('candidate_cross_threshold', 'candidate_margin')},
        'consensus': cfg['consensus'], 'signal_points': cfg['signal_points'], 'native_records': records_hash,
        'reassessment': cfg['reassessment'],
        'training_population': 'strict_prior_l2_actual_flat_prefix_pre_gate_attempts'})
    cfg['E0_policy_hash'] = stable_hash({
        'stage': 'E0', 'candidate_policy_hash': cfg['candidate_policy_hash'], 'account': cfg['account'],
        'trend_catalog_hash': stable_hash(trends), 'ordinary_sell': 'continuous_sell_pass_for_actual_holding',
        'reassessment': cfg['reassessment'], 'entry_HR': False})
    cfg['E1_policy_hash'] = stable_hash({
        'stage': 'E1', 'E0': cfg['E0_policy_hash'], 'signal_time_pinned_MA_exit': stable_hash(ma),
        'holding_threshold': cfg['account']['holding_exit_threshold'], 'no_head': 'entire_position_E0',
        'checkpoint_upgrade': False, 'L2_exit_apply': False})
    return cfg


def _graph_value(graph, token, *, limit=10000):
    """Expand only the small series-context subtree, never a ledger graph."""
    count = [0]
    def expand(value, before=None):
        if not isinstance(value, dict):
            return value
        if set(value) != {'ref'} or type(value['ref']) is not int:
            raise ValueError('invalid account graph reference')
        index = value['ref']
        if not 0 <= index < len(graph['objects']) or (before is not None and index >= before):
            raise ValueError('invalid account graph reference order')
        count[0] += 1
        if count[0] > limit:
            raise ValueError('resume account context is unexpectedly large')
        node = graph['objects'][index]
        if set(node) == {'d'} and isinstance(node['d'], dict):
            return {k: expand(v, index) for k, v in node['d'].items()}
        if set(node) == {'l'} and isinstance(node['l'], list):
            return [expand(v, index) for v in node['l']]
        raise ValueError('invalid account graph context node')
    return expand(token)


def _verify_account(directory, summary, cfg, calendar_hash, run_id):
    descriptors = {name: _read(directory / name) for name in
                   ('reference_account.json', 'candidate_clones.json', 'holding_pairs.json')}
    descriptor = descriptors['reference_account.json']
    graph_path = _inside(directory, descriptor['asset'])
    compressed = graph_path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != descriptor['asset_sha256']:
        raise ValueError('resume account graph compressed SHA mismatch')
    with gzip.GzipFile(fileobj=__import__('io').BytesIO(compressed)) as stream:
        graph_bytes = stream.read(descriptor['graph_bytes'] + 1)
    if len(graph_bytes) != descriptor['graph_bytes'] or hashlib.sha256(graph_bytes).hexdigest() != descriptor['graph_sha256']:
        raise ValueError('resume account graph SHA mismatch')
    graph = json.loads(graph_bytes)
    if (graph.get('schema') != 'czsc_account_graph_v1' or not isinstance(graph.get('objects'), list)
            or not isinstance(graph.get('documents'), dict)):
        raise ValueError('resume account graph schema mismatch')
    for name, item in descriptors.items():
        if (item.get('schema') != 'czsc_account_reference_v1' or item.get('document') != name
                or item.get('asset') != graph_path.name or item.get('asset_sha256') != descriptor['asset_sha256']
                or item.get('graph_sha256') != descriptor['graph_sha256'] or item.get('graph_bytes') != descriptor['graph_bytes']
                or name not in graph['documents']
                or summary['files'].get(graph_path.name, {}).get('sha256') != item['asset_sha256']):
            raise ValueError('resume compact account reference is not indexed in its frozen summary')
    token = graph['documents']['reference_account.json']
    if not isinstance(token, dict) or set(token) != {'ref'}:
        raise ValueError('resume reference account root missing')
    node = graph['objects'][token['ref']]
    account = node.get('d') or {}
    context = _graph_value(graph, account['series_context'])
    from my_strategy.services.czsc_trend_contracts import strategy_config
    stage = summary['stage']
    if (account.get('symbol') != summary['symbol'] or account.get('mode') != stage
            or account.get('account_id') != f"research:{run_id}:{summary['symbol']}:reference"
            or account.get('version') != 'trend_consensus' or type(account.get('completed')) is not bool
            or account.get('series_context_hash') != stable_hash(context)
            or context.get('source_hash') != cfg['source_snapshot_hash']
            or context.get('calendar_hash') != calendar_hash
            or context.get('symbol') != summary['symbol'] or context.get('mode') != stage
            or context.get('strategy') != 'trend_consensus'
            or context.get('source_revision_verified') is not True
            or context.get('entry_recipe_hash') is not None
            or context.get('initial_cash') != cfg['initial_cash']
            or context.get('exit_recipe_hash') != (cfg['E1_policy_hash'] if stage == 'E1' else None)
            or context.get('config_hash') != stable_hash(strategy_config(cfg if stage == 'E1' else
                {k: v for k, v in cfg.items() if k != 'E1_policy_hash'}))):
        raise ValueError('resume actual account belongs to another source, configuration, calendar or continuation')


def _verify_stock(root, summary, cfg, calendar_hash, run_id, proof, *, expected_source_end=None):
    from my_strategy.services.czsc_trend_features import ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS
    stage, symbol = summary['stage'], summary['symbol']
    directory = root / 'dataset' / stage / symbol.replace('.', '_')
    if (stage not in {'E0', 'E1'} or summary.get('reference_mode') != stage
            or summary.get('reference_population') or summary.get('population_end') is not None
            or summary.get('reference_population_hash') != stable_hash({})
            or summary.get('frozen_source_end') != (expected_source_end or cfg['training_end'])
            or summary.get('effective_population_end') != summary.get('frozen_source_end')
            or summary.get('holding_checkpoint_upgrade') is not False
            or not _REQUIRED_STOCK.issubset(summary.get('files', {}))):
        raise ValueError('resume per-stock stage is not the frozen independent account population')
    for name, asset in summary['files'].items():
        path = _asset(directory, asset['path'], asset['sha256'], proof)
        if path.name != name or path.parent != directory.resolve() or name == 'manifest.json':
            raise ValueError('resume per-stock asset path/name mismatch')
    for key, columns in (('entries', ENTRY_FEATURE_COLUMNS), ('exits', EXIT_FEATURE_COLUMNS), ('episodes', ())):
        path = directory / (key + '.parquet')
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        if len(names) != len(set(names)) or not set(columns).issubset(names):
            raise ValueError('resume training feature columns/order mismatch')
        expected_rows = summary['exit_rows'] if key == 'exits' else summary['entry_rows']
        if parquet.metadata.num_rows != expected_rows:
            raise ValueError('resume stock parquet row count mismatch')
        if not expected_rows:
            continue
        expected = {'stage': stage, 'continuation_policy_hash': cfg[stage + '_policy_hash']}
        if key != 'exits':
            expected.update(candidate_policy_hash=cfg['candidate_policy_hash'], reference_population_hash=stable_hash({}))
        if not set(expected).issubset(names):
            raise ValueError('resume parquet continuation identity columns missing')
        table = parquet.read(columns=list(expected))
        for column, value in expected.items():
            if table.column(column).unique().to_pylist() != [value]:
                raise ValueError('resume parquet continuation/source policy changed')
    _verify_account(directory, summary, cfg, calendar_hash, run_id)
    proof[str(directory / 'manifest.json')] = _file_hash(directory / 'manifest.json')
    return summary


def inspect_training_resume(run_id, *, progress=None, check_cancel=None):
    """Read-only, fail-closed verification of the actual interrupted run."""
    from my_strategy.services import czsc_trend_pipeline as pipeline
    from my_strategy.services.czsc_trend_contracts import validate_training_config
    if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
        raise ValueError('invalid training resume run identifier')
    root = _inside(ARTIFACT_RUNS_ROOT, run_id)
    if (root / 'reports/trend_consensus.json').exists():
        raise ValueError('completed training runs cannot be resumed or overwritten')
    proof = {}
    metadata = _tracked(root, 'metadata.json', proof)
    initial = _tracked(root, 'config/initial_config.json', proof)
    code = _tracked(root, 'config/generating_code_hashes.json', proof)
    source = _tracked(root, 'data_snapshots/source.json', proof)
    records = _tracked(root, 'data_snapshots/records.json', proof)
    calendar = _tracked(root, 'data_snapshots/calendar.json', proof)
    validate_training_config(initial)
    if (metadata.get('run_id') != run_id or metadata.get('task') != 'trend-consensus-train'
            or metadata.get('config_hash') != stable_hash(initial)
            or source.get('snapshot_hash') != stable_hash({k: v for k, v in source.items() if k != 'snapshot_hash'})
            or initial.get('source_snapshot_hash') != source.get('snapshot_hash')
            or metadata.get('data_version') != source.get('snapshot_hash')
            or stable_hash(records) != source.get('records_hash')
            or metadata.get('stocks') != [r['symbol'] for r in records]
            or source.get('selected_symbols') != metadata['stocks']
            or initial.get('training_end') != metadata.get('as_of_date')
            or calendar.get('verified') is not True
            or calendar.get('dates') != sorted(set(calendar.get('dates', [])))
            or not calendar.get('dates')):
        raise ValueError('training resume frozen input/configuration identity mismatch')
    if set(code) != set(pipeline.GENERATING_FILES):
        raise ValueError('training resume generating-code catalogue incomplete')
    for relative, sha in code.items():
        _asset(MY_STRATEGY_ROOT, relative, sha, proof)
        _asset(root, 'config/source_code/' + relative, sha, proof)
    native_root = _inside(ARTIFACT_RUNS_ROOT, source['source_native_run_id'])
    native_path = _asset(native_root, 'reports/native_source.json', source['source_report_sha256'], proof)
    native = _read(native_path)
    if (native.get('status') != 'completed' or native.get('failures') or native.get('raw_only_rebuild') is not True
            or native.get('dataset_records') != records or native.get('source_snapshot') != source['source_snapshot']
            or [d for d in native.get('calendar_dates', []) if d <= initial['training_end']] != calendar['dates']
            or calendar.get('source_run_id') != source['source_snapshot'].get('calendar_run_id')):
        raise ValueError('training resume does not own a complete matching native source')
    pipeline.verify_full_population(native, initial, records)
    pipeline.verify_source_generators(source['source_snapshot'])
    total = len(records) * 3
    _emit(progress, 0, total, message='核验冻结原生输入、已拟合模型和已完成训练阶段，不重建原生结构')
    for index, record in enumerate(records):
        if check_cancel:
            check_cancel()
        for path_name, hash_name in (('raw_path', 'raw_sha256'), ('path', 'sha256'), ('native_path', 'native_sha256')):
            path = resolve_frozen_asset(record[path_name], runs_root=ARTIFACT_RUNS_ROOT)
            _asset(ARTIFACT_RUNS_ROOT, path, record[hash_name], proof)
        if index % 40 == 0 or index + 1 == len(records):
            _emit(progress, index + 1, total, verified_native_inputs=index + 1, source_count=len(records))
    trends = _tracked(root, 'models/trend_bundles.json', proof)
    ma = _tracked(root, 'models/ma_exit_bundles.json', proof)
    folds = pipeline.chronological_folds(initial['training_end'], calendar['dates'])
    if ([b['fold'] for b in trends] != folds or [b['fold'] for b in ma] != folds
            or len(trends) != len(folds) or len(ma) != len(folds)):
        raise ValueError('training resume requires completed trend and MA exit head stages')
    manifests = {}
    for head, bundles, loader in (('trend_direction', trends, pipeline.load_trend_head), ('ma_exit', ma, pipeline.load_head)):
        for bundle in bundles:
            directory = _inside(root, bundle['heads'][head]['path'])
            manifest = loader(directory)
            if (manifest['manifest_sha256'] != bundle['heads'][head]['manifest_sha256']
                    or manifest['fold'] != bundle['fold'] or manifest['head'] != head
                    or bundle['available_at'] != manifest['fold']['policy_end']
                    or manifest['target_contract']['source_snapshot_hash'] != source['snapshot_hash']
                    or manifest['target_contract']['pipeline_contract_hash'] != initial['pipeline_contract_hash']):
                raise ValueError('resume model head source/fold contract mismatch')
            for file in directory.iterdir():
                if file.is_file():
                    proof[str(file)] = _file_hash(file)
            manifests[(head, bundle['name'])] = manifest
    selection = _tracked(root, 'validation/source_policy_selection.json', proof)
    asset = selection.get('accounts_asset')
    if not asset or selection.get('four_development_windows_used') is not False:
        raise ValueError('training resume source-policy evidence incomplete')
    _asset(root, asset['path'], asset['sha256'], proof)
    holding_policy = _tracked(root, 'validation/holding_exit_policy_selection.json', proof)
    if [p.get('checkpoint') for p in holding_policy] != [b['name'] for b in ma]:
        raise ValueError('training resume pinned holding policy catalogue incomplete')
    cfg = _derive_config(initial, selection, trends, ma, source['records_hash'])
    for bundle in ma:
        contract = manifests[('ma_exit', bundle['name'])]['target_contract']
        if contract.get('stage') != 'E0' or contract.get('continuation_policy_hash') != cfg['E0_policy_hash']:
            raise ValueError('training resume MA exit model continuation differs from frozen E0')
    observations = _tracked(root, 'dataset/trend/manifest.json', proof)
    if (observations.get('failures') or observations.get('source_snapshot_hash') != source['snapshot_hash']
            or [r['symbol'] for r in observations.get('records', [])] != source['selected_symbols']):
        raise ValueError('training resume trend feature population incomplete')
    observed_rows = 0
    for record in observations['records']:
        path = _asset(root, record['path'], record['sha256'], proof)
        observed_rows += pq.ParquetFile(path).metadata.num_rows
        if record['label_contract'] != manifests[('trend_direction', trends[0]['name'])]['target_contract']:
            raise ValueError('training resume trend label contract drift')
    pooled_observation = root / 'dataset/trend/observations.parquet'
    if not pooled_observation.is_file() or pq.ParquetFile(pooled_observation).metadata.num_rows != observed_rows:
        raise ValueError('training resume pooled trend feature population incomplete')
    proof[str(pooled_observation)] = _file_hash(pooled_observation)
    baseline = _tracked(root, 'dataset/E0/manifest.json', proof)
    if (baseline.get('stage') != 'E0' or baseline.get('failures') or baseline.get('reference_population')
            or baseline.get('source_snapshot_hash') != source['snapshot_hash']
            or [r['symbol'] for r in baseline.get('records', [])] != source['selected_symbols']
            or set(baseline.get('pooled_files', {})) != {'entries', 'exits', 'episodes'}):
        raise ValueError('training resume E0 full population incomplete')
    for key, asset in baseline['pooled_files'].items():
        path = _asset(root, asset['path'], asset['sha256'], proof)
        expected = sum(s['exit_rows' if key == 'exits' else 'entry_rows'] for s in baseline['records'])
        if pq.ParquetFile(path).metadata.num_rows != expected:
            raise ValueError('training resume pooled E0 row count mismatch')
    calendar_hash = stable_hash(calendar['dates'])
    cached = {}
    source_records = {record['symbol']: record for record in records}
    for index, record in enumerate(baseline['records']):
        if check_cancel:
            check_cancel()
        stored = _tracked(root, 'dataset/E0/' + record['symbol'].replace('.', '_') + '/manifest.json', proof)
        if stored != record:
            raise ValueError('training resume E0 aggregate/per-stock summary mismatch')
        actual_end = min(str(source_records[record['symbol']]['data_end'])[:10], cfg['training_end'])
        _verify_stock(root, stored, cfg, calendar_hash, run_id, proof, expected_source_end=actual_end)
        if index % 40 == 0 or index + 1 == len(records):
            _emit(progress, len(records) + index + 1, total, verified_E0=index + 1)
    for index, record in enumerate(records):
        if check_cancel:
            check_cancel()
        path = root / 'dataset/E1' / record['symbol'].replace('.', '_') / 'manifest.json'
        if path.is_file():
            summary = _tracked(root, path, proof)
            if summary.get('symbol') != record['symbol'] or summary.get('stage') != 'E1':
                raise ValueError('training resume E1 stock identity mismatch')
            actual_end = min(str(record['data_end'])[:10], cfg['training_end'])
            cached[record['symbol']] = _verify_stock(root, summary, cfg, calendar_hash, run_id, proof,
                                                   expected_source_end=actual_end)
        if index % 40 == 0 or index + 1 == len(records):
            _emit(progress, 2 * len(records) + index + 1, total, verified_E1=len(cached))
    # This adapter deliberately resumes the interruption boundary it can prove.
    # Later population/quality checkpoints require their own exact DAG verifier.
    if (root / 'dataset/final_population').exists() or (root / 'models/l2_bundles.json').exists():
        raise ValueError('training resume boundary is beyond the supported E1 interruption')
    return dict(run_id=run_id, root=root, metadata=metadata, initial_config=initial, config=cfg,
                source=source, records=records, calendar=calendar, code=code, proof=proof,
                trends=trends, ma=ma, manifests=manifests, observations=observations, baseline=baseline,
                selection=selection, holding_policy=holding_policy, cached_E1=cached,
                reused=dict(raw=len(records), native=len(records), trend_features=len(records),
                            trend_heads=len(trends), source_policy=True, E0=len(records), ma_exit_heads=len(ma),
                            E1=len(cached)), missing_E1=len(records) - len(cached), training_parameters_unchanged=True)


class _ImmutableResumePath(type(Path())):
    def write_bytes(self, data):
        if self.is_file():
            if self.read_bytes() != data:
                raise ValueError('resume would replace a frozen source archive')
            return len(data)
        return super().write_bytes(data)


class _ResumeContext:
    def __init__(self, plan):
        self.root, self.metadata = plan['root'], copy.deepcopy(plan['metadata'])
        self.run_id = plan['run_id']

    def run_dir(self):
        return self.root

    def subdir(self, *parts):
        path = _inside(self.root, Path(*parts))
        path.mkdir(parents=True, exist_ok=True)
        return _ImmutableResumePath(path)

    def write_metadata(self, extra=None):
        payload = copy.deepcopy(self.metadata)
        payload.update(extra or {})
        _save(self.root / 'metadata.json', payload)
        return self.root / 'metadata.json'


@contextmanager
def _run_lease(root):
    with (root / '.training_resume.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if not handle.seek(0, os.SEEK_END):
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise ValueError('the same training run already has an active resume owner') from exc
        try:
            yield
        finally:
            release()


@contextmanager
def _reuse_boundaries(plan, check_cancel=None):
    """Substitute verified assets locally; spawned workers execute frozen code."""
    from my_strategy.services import czsc_trend_pipeline as p
    originals = {name: getattr(p, name) for name in (
        'create_run_context', '_save', '_bounded_results', '_pool_parquets', 'train_trend_head',
        'select_source_policy', '_fit_interval', 'select_holding_exit_policy', 'build_stage')}
    root = plan['root']
    immutable = {path: sha for path, sha in plan['proof'].items() if Path(path).is_relative_to(root)}
    stats = {'E1_new': 0, 'E1_reused': 0}
    def cancelled():
        if check_cancel:
            check_cancel()
    def context(**kwargs):
        cancelled()
        if (stable_hash(kwargs['config']) != plan['metadata']['config_hash']
                or kwargs['stocks'] != plan['source']['selected_symbols']
                or kwargs['data_version'] != plan['source']['snapshot_hash']
                or str(kwargs['as_of_date']) != plan['initial_config']['training_end']):
            raise ValueError('original frozen pipeline derived another resume run identity')
        return _ResumeContext(plan)
    def save(path, value):
        path = Path(path).resolve()
        if str(path) in immutable:
            if _file_hash(path) != immutable[str(path)] or _read(path) != value:
                raise ValueError('original pipeline would overwrite a completed frozen training asset')
            return None
        return originals['_save'](path, value)
    def results(worker, payloads, cpu_workers, batch_size):
        cancelled()
        if worker is p._trend_observation_worker:
            for asset in plan['observations']['records']:
                cancelled()
                yield copy.deepcopy(asset)
            return
        if worker is p._dataset_worker:
            payloads = list(payloads)
            pending = []
            for payload in payloads:
                record, _, cfg, _, _, _, stage, options = payload
                if stage == 'E1' and options['subdirectory'] == 'E1' and record['symbol'] in plan['cached_E1']:
                    if cfg['E1_policy_hash'] != plan['config']['E1_policy_hash']:
                        raise ValueError('resume E1 continuation changed before cached dispatch')
                    stats['E1_reused'] += 1
                    cancelled()
                    yield copy.deepcopy(plan['cached_E1'][record['symbol']])
                else:
                    pending.append(payload)
            for result in originals['_bounded_results'](worker, pending, cpu_workers, batch_size):
                cancelled()
                if result['stage'] == 'E1' and not result.get('reference_population'):
                    stats['E1_new'] += 1
                yield result
            return
        for result in originals['_bounded_results'](worker, payloads, cpu_workers, batch_size):
            cancelled()
            yield result
    def pool(paths, destination, *, load=True):
        path = Path(destination).resolve()
        if str(path) in immutable:
            if _file_hash(path) != immutable[str(path)]:
                raise ValueError('resume pooled feature asset changed')
            return pd.read_parquet(path) if load else None
        return originals['_pool_parquets'](paths, destination, load=load)
    def trend(dataset, directory, fold, cfg, *, contract):
        manifest = plan['manifests'][('trend_direction', fold['name'])]
        if manifest['target_contract'] != contract or manifest['fold'] != fold:
            raise ValueError('resume trend fit call differs from the completed head')
        return copy.deepcopy(manifest)
    def policy(records, calendar, cfg, *args):
        derived = _derive_config(cfg, plan['selection'], plan['trends'], plan['ma'], plan['source']['records_hash'])
        cfg['trend'] = derived['trend']
        cfg['consensus'] = derived['consensus']
        return copy.deepcopy(plan['selection'])
    def interval(dataset, head, fold, calendar, context, cfg, architecture='linear', device='cpu'):
        if head == 'ma_exit':
            manifest = plan['manifests'][('ma_exit', fold['name'])]
            bundle = next(b for b in plan['ma'] if b['name'] == fold['name'])
            if manifest['target_contract'] != dataset.attrs.get('label_contract') or architecture != 'linear':
                raise ValueError('resume MA exit fit call differs from the completed frozen head')
            return copy.deepcopy(bundle), copy.deepcopy(manifest)
        cancelled()
        return originals['_fit_interval'](dataset, head, fold, calendar, context, cfg, architecture, device)
    def holding(dataset, bundle, *args):
        return copy.deepcopy(next(v for v in plan['holding_policy'] if v['checkpoint'] == bundle['name']))
    def stage(records, calendar, cfg, context, stage, *args, **kwargs):
        cancelled()
        if stage == 'E0' and not kwargs.get('subdirectory'):
            if cfg['E0_policy_hash'] != plan['config']['E0_policy_hash']:
                raise ValueError('resume E0 candidate/continuation changed')
            data = {key: pd.read_parquet(asset['path']) for key, asset in plan['baseline']['pooled_files'].items()}
            data.update(reference_population={}, reference_mode='E0')
            print('STAGE reused_E0_verified_full_population', flush=True)
            return data, copy.deepcopy(plan['baseline']['records'])
        if stage == 'E1' and not kwargs.get('subdirectory') and not kwargs.get('reference_population'):
            if cfg['E1_policy_hash'] != plan['config']['E1_policy_hash']:
                raise ValueError('resume E1 continuation changed')
            trends, ma, workers = args
            options = dict(subdirectory='E1', reference_population=None, population_end=None)
            summaries = [copy.deepcopy(plan['cached_E1'][r['symbol']]) for r in records
                         if r['symbol'] in plan['cached_E1']]
            stats['E1_reused'] = len(summaries)
            def emit():
                print('PROGRESS ' + json.dumps(dict(stage='dataset', population='E1', fold=None,
                    completed=len(summaries), total=len(records), scope='current_population_only'),
                    ensure_ascii=False, separators=(',', ':')), flush=True)
            emit()
            payloads = ((r, calendar, cfg, str(root), trends, ma, 'E1', options) for r in records
                        if r['symbol'] not in plan['cached_E1'])
            for summary in results(p._dataset_worker, payloads, workers, cfg['stock_batch_size']):
                summaries.append(summary)
                emit()
            order = {record['symbol']: index for index, record in enumerate(records)}
            summaries.sort(key=lambda summary: order[summary['symbol']])
            if [s['symbol'] for s in summaries] != [r['symbol'] for r in records]:
                raise ValueError('resumed E1 did not complete the full frozen population')
            pooled, files = {}, {}
            for key in ('entries', 'exits', 'episodes'):
                path = context.subdir('dataset', 'E1') / (key + '.parquet')
                paths = (root / 'dataset/E1' / s['symbol'].replace('.', '_') / (key + '.parquet') for s in summaries)
                pooled[key] = pool(paths, path)
                files[key] = dict(path=str(path), sha256=_file_hash(path))
            pooled.update(reference_population={}, reference_mode='E1')
            save(context.subdir('dataset', 'E1') / 'manifest.json', dict(stage='E1', records=summaries, failures=[],
                reference_population={}, reference_population_hash=stable_hash({}), population_end=None,
                source_snapshot_hash=cfg['source_snapshot_hash'], pooled_files=files))
            return pooled, summaries
        return originals['build_stage'](records, calendar, cfg, context, stage, *args, **kwargs)
    replacement = dict(create_run_context=context, _save=save, _bounded_results=results, _pool_parquets=pool,
                       train_trend_head=trend, select_source_policy=policy, _fit_interval=interval,
                       select_holding_exit_policy=holding, build_stage=stage)
    try:
        for name, function in replacement.items():
            setattr(p, name, function)
        yield stats
    finally:
        for name, function in originals.items():
            setattr(p, name, function)


def resume_training_run(run_id, *, device=None, cpu_workers=None, progress=None, check_cancel=None, request=None):
    """Resume in place under an OS process lease and persist the normal result."""
    from my_strategy.services import czsc_trend_pipeline as pipeline
    from my_strategy.services.czsc_trend_contracts import TRAIN_KIND
    from my_strategy.storage.czsc_results import ResultStore
    if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
        raise ValueError('invalid training resume run identifier')
    root = _inside(ARTIFACT_RUNS_ROOT, run_id)
    if not _THREAD_LOCK.acquire(blocking=False):
        raise ValueError('a training resume adapter is already running in this process')
    try:
        with _run_lease(root):
            attempt_id = local_now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:12]
            attempt_path = root / 'reports' / ('training_resume.' + attempt_id + '.json')
            audit = dict(schema='czsc_training_resume_v1', run_id=run_id, attempt_id=attempt_id,
                         status='verifying', owner_pid=os.getpid(), started_at=local_now().isoformat(),
                         orchestrator_sha256=_file_hash(Path(__file__)), raw_rebuilt=False,
                         native_rebuilt=False, model_qualification_changed=False)
            script = MY_STRATEGY_ROOT / 'scripts/resume_trend_training.py'
            audit['orchestration_code_hashes'] = {'services/czsc_trend_resume.py': audit['orchestrator_sha256'],
                                                'scripts/resume_trend_training.py': _file_hash(script)}
            _save(attempt_path, audit)
            try:
                plan = inspect_training_resume(run_id, progress=progress, check_cancel=check_cancel)
                if request:
                    _verify_request(plan, **request)
                if cpu_workers is not None and cpu_workers != plan['initial_config']['cpu_workers']:
                    raise ValueError('resume CPU workers differ from the original frozen training parameters')
                actual_device = plan['manifests'][('ma_exit', plan['ma'][0]['name'])].get('actual_device')
                if actual_device is None:
                    actual_device = plan['manifests'][('ma_exit', plan['ma'][0]['name'])].get('device')
                if actual_device not in {'cpu', 'mps', 'cuda'}:
                    raise ValueError('original neural training device cannot be verified')
                if device is not None and device != actual_device:
                    raise ValueError('resume device differs from the original fitted head device')
                audit.update(status='running', reused=plan['reused'], missing_E1=plan['missing_E1'],
                             frozen_input_identity_hash=stable_hash(plan['proof']),
                             initial_config_hash=stable_hash(plan['initial_config']), device=actual_device,
                             cpu_workers=plan['initial_config']['cpu_workers'])
                _save(attempt_path, audit)
                print('TRAIN_RUN_ID ' + run_id, flush=True)
                print('RESUME_REUSED ' + json.dumps(plan['reused'], ensure_ascii=False, separators=(',', ':')), flush=True)
                input_config = copy.deepcopy(plan['initial_config'])
                # The original pipeline adds this derived field after hashing
                # registered_config. Feeding the result back would self-hash.
                input_config.pop('pipeline_contract_hash', None)
                with _reuse_boundaries(plan, check_cancel) as stats:
                    report = pipeline.train_trend_consensus(
                        source_native_run_id=plan['source']['source_native_run_id'], end=plan['initial_config']['training_end'],
                        config=input_config, cpu_workers=plan['initial_config']['cpu_workers'], device=actual_device)
                # Metadata is the only verified pre-existing file the original
                # completion deliberately updates. Everything else stays frozen.
                for path, sha in plan['proof'].items():
                    if Path(path) != root / 'metadata.json' and _file_hash(Path(path)) != sha:
                        raise ValueError('completed reused training asset changed during continuation')
                if any(_file_hash(MY_STRATEGY_ROOT / relative) != sha
                       for relative, sha in audit['orchestration_code_hashes'].items()):
                    raise ValueError('training resume orchestration changed during continuation')
                audit.update(status='completed', completed_at=local_now().isoformat(), newly_calculated=stats,
                             publication_allowed=False, training_parameters_unchanged=True)
                _save(attempt_path, audit)
                report['resume_provenance'] = dict(audit_path=str(attempt_path.relative_to(root)),
                    audit_sha256=_file_hash(attempt_path), attempt_id=attempt_id,
                    orchestrator_sha256=audit['orchestrator_sha256'],
                    orchestration_code_hashes=audit['orchestration_code_hashes'],
                    reused=plan['reused'], newly_calculated=stats,
                    original_run_id=run_id, actual_resumed_completion_at=audit['completed_at'])
                _save(root / 'reports/trend_consensus.json', report)
                pipeline._verify_run(report, root)
                from my_strategy.services.czsc_trend_support import _json
                ResultStore().save(TRAIN_KIND, _json(report))
                return report
            except BaseException as error:
                audit.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                             finished_at=local_now().isoformat(), error=type(error).__name__ + ': ' + str(error))
                _save(attempt_path, audit)
                raise
    finally:
        _THREAD_LOCK.release()


def resume_summary(plan):
    return dict(run_id=plan['run_id'], data_end=plan['initial_config']['training_end'],
                source_native_run_id=plan['source']['source_native_run_id'], reused=plan['reused'],
                missing_E1=plan['missing_E1'], source_identity_hash=stable_hash(plan['proof']),
                training_parameters_unchanged=True, native_rebuilt=False, raw_rebuilt=False)


def _verify_request(plan, *, source_native_run_id=None, config=None, end=None,
                    cpu_workers=None, device=None, stock_batch_size=None, cohort=None, calendar_run_id=None):
    from my_strategy.services.czsc_trend_contracts import validate_training_config
    initial = plan['initial_config']
    if source_native_run_id is not None and source_native_run_id != plan['source']['source_native_run_id']:
        raise ValueError('resume requested another native input source')
    if end is not None and end != initial['training_end']:
        raise ValueError('resume requested another training end date')
    if cpu_workers is not None and cpu_workers != initial['cpu_workers']:
        raise ValueError('resume requested another frozen worker count')
    if stock_batch_size is not None and stock_batch_size != initial['stock_batch_size']:
        raise ValueError('resume requested another stock preparation batch size')
    if cohort is not None:
        actual = cohort.get('symbols') if isinstance(cohort, dict) else cohort
        if actual != plan['source']['selected_symbols']:
            raise ValueError('resume requested another frozen stock cohort/order')
    if calendar_run_id is not None and calendar_run_id != plan['calendar']['source_run_id']:
        raise ValueError('resume requested another verified calendar source')
    if config is not None:
        registered = validate_training_config(config)
        derived = {'pipeline_contract_hash', 'source_snapshot_hash', 'training_end', 'cpu_workers',
                   'training_capacity_tiers', 'capacity_expansion_criterion'}
        if {k: v for k, v in registered.items() if k not in derived} != {k: v for k, v in initial.items() if k not in derived}:
            raise ValueError('resume requested different frozen model/label/qualification parameters')
    actual_device = plan['manifests'][('ma_exit', plan['ma'][0]['name'])].get('actual_device')
    if actual_device not in {'cpu', 'mps', 'cuda'} or (device is not None and device != actual_device):
        raise ValueError('resume requested another or unverified original training device')


def inspect_resume(training_run_id, *, progress=None, check_cancel=None, **request):
    plan = inspect_training_resume(training_run_id, progress=progress, check_cancel=check_cancel)
    _verify_request(plan, **request)
    return resume_summary(plan)


def run_resume(training_run_id, *, progress=None, check_cancel=None, **request):
    # Request validation runs inside the same lease as asset verification and
    # continuation, so no caller can change the source/parameters after checking.
    return resume_training_run(training_run_id, progress=progress, check_cancel=check_cancel, request=request)
