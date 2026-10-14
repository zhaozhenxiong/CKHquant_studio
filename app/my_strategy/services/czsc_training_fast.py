"""Scoped execution accelerators for the unchanged, frozen training generators.

This module owns scheduling and reuse, never financial calculations. A cached
answer is usable only for identical inputs, generators and physical assets.
Spawned processes enter the same scope explicitly; importing this module alone
does not patch anything. Completed results are spooled to private temporary
files so a slow first stock cannot starve workers or grow the RAM result queue.
"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import hashlib
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import uuid

import numpy as np
import pandas as pd

from my_strategy.services.czsc_training_cache import TrainingCache
from my_strategy.services.czsc_trend_support import _file_hash, resolve_frozen_asset

VERSION = 'training-fast-v1'
_STATE = None
_SCOPE_LOCK = threading.RLock()


def _encode(value):
    """Lossless, non-executable JSON codec (including missing numeric values)."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else ['float', repr(value)]
    if isinstance(value, np.generic):
        return ['numpy', str(value.dtype), _encode(value.item())]
    if value is pd.NA:
        return ['missing']
    if isinstance(value, pd.Timestamp):
        return ['timestamp', value.isoformat()]
    if isinstance(value, Path):
        return ['path', str(value)]
    if isinstance(value, dict):
        return ['dict', [[_encode(k), _encode(v)] for k, v in value.items()]]
    if isinstance(value, (list, tuple)):
        return ['tuple' if isinstance(value, tuple) else 'list', [_encode(v) for v in value]]
    if isinstance(value, np.ndarray):
        return ['array', str(value.dtype), list(value.shape), _encode(value.tolist())]
    if isinstance(value, pd.DataFrame):
        # All cells, index, column order, dtypes and attrs belong to the input.
        # A data_version attr alone cannot prove an unchanged sliced frame.
        return ['frame', _encode(value.to_dict('split')), list(map(str, value.dtypes)), _encode(value.attrs)]
    raise TypeError('unsupported training cache value: ' + type(value).__name__)


def _decode(value):
    if not isinstance(value, list):
        return value
    kind = value[0]
    if kind == 'float':
        return float(value[1])
    if kind == 'numpy':
        return np.asarray(_decode(value[2]), dtype=value[1])[()]
    if kind == 'missing':
        return pd.NA
    if kind == 'timestamp':
        return pd.Timestamp(value[1])
    if kind == 'path':
        return Path(value[1])
    if kind == 'dict':
        return {_decode(k): _decode(v) for k, v in value[1]}
    if kind in ('tuple', 'list'):
        values = [_decode(v) for v in value[1]]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'array':
        return np.asarray(_decode(value[3]), dtype=value[1]).reshape(value[2])
    raise ValueError('unsupported training cache payload')


def _dtype_identity(dtype):
    identity = {'name': str(dtype), 'representation': repr(dtype)}
    if isinstance(dtype, pd.CategoricalDtype):
        identity.update(categories=dtype.categories.tolist(), ordered=dtype.ordered,
                        categories_dtype=_dtype_identity(dtype.categories.dtype))
    return identity


def _frame_identity(frame):
    """Exact typed columns with native numeric hashing, avoiding row expansion.

    Object columns use the typed JSON fallback; pandas' object hash merges such
    inputs as integer 1 and string "1", which is unsuitable for reuse identity.
    """
    columns = []
    for index, name in enumerate(frame.columns):
        column = frame.iloc[:, index]
        values = column.to_numpy(copy=False)
        if values.dtype.kind in 'biufcMm':
            digest = hashlib.sha256(values.tobytes(order='C')).hexdigest()
        else:
            digest = _json_content_digest(column.tolist())
        columns.append((name, _dtype_identity(column.dtype), str(values.dtype), digest))
    return {'columns': columns, 'column_index_type': type(frame.columns).__name__,
            'column_names': list(frame.columns.names), 'column_index_dtype': _dtype_identity(frame.columns.dtype),
            'index_type': type(frame.index).__name__,
            'index_names': list(frame.index.names), 'index_dtype': _dtype_identity(frame.index.dtype),
            'index': _json_content_digest(frame.index.tolist()), 'attrs': frame.attrs}


def _key_encode(value):
    if isinstance(value, pd.DataFrame):
        return ['frame_identity', _key_encode(_frame_identity(value))]
    if isinstance(value, dict):
        return ['dict', [[_key_encode(k), _key_encode(v)] for k, v in value.items()]]
    if isinstance(value, (list, tuple)):
        return ['tuple' if isinstance(value, tuple) else 'list', [_key_encode(v) for v in value]]
    return _encode(value)


def _digest(value):
    return hashlib.sha256(json.dumps(_key_encode(value), ensure_ascii=False, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def _json_content_digest(value):
    """Fast exact hashing for JSON-native evidence, with a typed fallback.

    Native packs are decoded JSON and contain long repeated event-id lists.
    Validate their types without rebuilding those lists before using the C JSON
    encoder. This still hashes every value on every call: no object-id trust.
    """
    pending = [value]
    while pending:
        item = pending.pop()
        kind = type(item)
        if kind is dict:
            if any(type(key) is not str for key in item):
                return _digest(value)
            pending.extend(item.values())
        elif kind is list:
            pending.extend(item)
        elif kind is float:
            if not math.isfinite(item):
                return _digest(value)
        elif item is not None and kind not in (str, int, bool):
            return _digest(value)
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
        separators=(',', ':')).encode()).hexdigest()


def _stat(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _inside(root, value):
    root = Path(root).resolve()
    path = Path(value)
    unresolved = path if path.is_absolute() else root / path
    path = unresolved.resolve()
    if not path.is_relative_to(root) or any(p.is_symlink() for p in (unresolved, *unresolved.parents)
                                         if p != root.parent and p.is_relative_to(root)):
        raise ValueError('training cache asset escapes run')
    return path


def _generator_hashes():
    from my_strategy.services import czsc_trend_pipeline as pipeline
    files = {str((pipeline.MY_STRATEGY_ROOT / name).resolve()):
             _file_hash(pipeline.MY_STRATEGY_ROOT / name) for name in pipeline.GENERATING_FILES}
    files[str(Path(__file__).resolve())] = _file_hash(Path(__file__))
    cache_module = Path(__file__).with_name('czsc_training_cache.py').resolve()
    files[str(cache_module)] = _file_hash(cache_module)
    return files


class _Runtime:
    def __init__(self, cpu_workers, cache_path):
        self.cpu_workers = int(cpu_workers)
        self.cache_path = str(Path(cache_path).resolve()) if cache_path is not None else None
        self.cache = TrainingCache(self.cache_path)
        self.generators = _generator_hashes()
        self.models = {}
        self.originals = {}
        self.patches = []
        self.audit_results = None
        self.worker_deps = {}
        self.stats = {'computed': 0, 'reused': 0, 'model_loads': 0, 'model_hits': 0,
                      'value_hits': 0, 'value_misses': 0}

    def patch(self, module, name, replacement):
        original = getattr(module, name)
        self.patches.append((module, name, original))
        setattr(module, name, replacement)
        return original

    def verify(self):
        for path, expected in self.generators.items():
            if _file_hash(Path(path)) != expected:
                raise ValueError('training accelerator or frozen generator changed during execution')
        for item in self.models.values():
            for path, expected in item['files'].items():
                if _file_hash(Path(path)) != expected:
                    raise ValueError('training model asset changed during execution')

    def load_model(self, kind, path, expected_contract=None):
        path = Path(path).resolve()
        key = kind, str(path)
        cached = self.models.get(key)
        if cached is not None:
            if any(_stat(p) != signature for p, signature in cached['stats'].items()):
                raise ValueError('immutable training model asset changed during execution')
            manifest = cached['manifest']
            self.stats['model_hits'] += 1
        else:
            manifest = self.originals[kind](path)
            files = {str(path / 'manifest.json'): _file_hash(path / 'manifest.json')}
            checkpoint = _inside(path, manifest['checkpoint'])
            files[str(checkpoint)] = manifest['checkpoint_sha256']
            if manifest.get('export_sha256'):
                files[str(path / 'linear-layers.json')] = manifest['export_sha256']
            # The real loader already checked schemas, contract and checkpoints.
            # Record physical hashes too, including the JSON manifest itself.
            for asset, expected in files.items():
                if _file_hash(Path(asset)) != expected:
                    raise ValueError('model changed while validating training cache')
            cached = {'manifest': copy.deepcopy(manifest), 'files': files,
                      'stats': {p: _stat(p) for p in files}}
            self.models[key] = cached
            self.stats['model_loads'] += 1
        if expected_contract is not None and manifest['target_contract'] != expected_contract:
            raise ValueError('reference quality target source binding changed')
        return copy.deepcopy(manifest)

    def dependencies(self, payload, root):
        deps = dict(self.generators)
        def add(path, expected):
            path = Path(path).resolve()
            if _file_hash(path) != expected:
                raise ValueError('training checkpoint input SHA mismatch: ' + str(path))
            previous = deps.setdefault(str(path), expected)
            if previous != expected:
                raise ValueError('training checkpoint has conflicting input hashes')
        def visit(value):
            if isinstance(value, (tuple, list)):
                for child in value:
                    visit(child)
            elif isinstance(value, dict):
                for field, sha in (('path', 'sha256'), ('native_path', 'native_sha256'), ('raw_path', 'raw_sha256')):
                    if value.get(field) and value.get(sha):
                        candidate = Path(value[field])
                        if not candidate.is_absolute() and (Path(root) / candidate).is_file():
                            candidate = _inside(root, candidate)
                        else:
                            candidate = resolve_frozen_asset(value[field])
                        add(candidate, value[sha])
                for head, asset in (value.get('heads') or {}).items():
                    if not isinstance(asset, dict) or not asset.get('path'):
                        continue
                    path = _inside(root, asset['path'])
                    kind = 'trend' if head == 'trend_direction' else 'reference' if head == 'reference_quality' else 'interval'
                    manifest = self.load_model(kind, path)
                    if asset.get('manifest_sha256') != manifest['manifest_sha256']:
                        raise ValueError('training checkpoint model catalog drift')
                    deps.update(self.models[(kind, str(path))]['files'])
                for child in value.values():
                    visit(child)
        visit(payload)
        return deps

    def value(self, namespace, arguments, call, extra_deps=None):
        try:
            key = _digest({'version': VERSION, 'generators': self.generators, 'arguments': arguments})
        except TypeError:
            # Calls with callbacks or policy-base side effects stay unmodified.
            return call()
        # Bind only the files this computation reads. The complete worker source
        # and catalogue are independently verified by _checkpoint_worker. Adding
        # an unrelated prior L2 entry head must not invalidate trend/event values.
        deps = {**self.generators, **(extra_deps or {})}
        cached = self.cache.get_json(namespace, key, deps=deps)
        if cached is not None:
            self.stats['value_hits'] += 1
            return _decode(cached)
        result = call()
        try:
            encoded = _encode(result)
        except TypeError:
            return result
        self.cache.put_json(namespace, key, encoded, deps=deps)
        self.stats['value_misses'] += 1
        return result


def _load_head(path):
    return _STATE.load_model('interval', path)


def _load_trend_head(path):
    return _STATE.load_model('trend', path)


def _load_reference_head(path, *, expected_contract=None):
    return _STATE.load_model('reference', path, expected_contract)


def _forward_trend(*args, **kwargs):
    state = _STATE
    original = state.originals['forward_trend']
    bound = inspect.signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    # Sessions own mutable inference counters/caches; preserve explicit sessions.
    if values.get('session') is not None:
        return original(*args, **kwargs)
    deps = state.dependencies(values['trend_bundles'], values['root'])
    return state.value('forward_trend', values, lambda: original(*args, **kwargs), deps)


def _event_prefix_rows(*args, **kwargs):
    original = _STATE.originals['event_prefix_rows']
    bound = inspect.signature(original).bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    # The original function reads only these two feature columns. Physical
    # source/model dependencies remain checked by the surrounding worker scope.
    values['features'] = values['features'][['date', 'symbol']]
    return _STATE.value('event_prefix_rows', values, lambda: original(*args, **kwargs))


def checkpoint_key(kind, payload):
    """Identity helper for checked checkpoint importers; no implicit adoption."""
    generators = _STATE.generators if _STATE is not None else _generator_hashes()
    return _digest({'version': VERSION, 'generators': generators, 'worker': kind, 'payload': payload})


def _worker_root(kind, payload):
    if kind == 'observation':
        directory = Path(payload[3]).resolve()
        if directory.parts[-3:] != ('dataset', 'trend', 'stocks'):
            raise ValueError('unexpected trend observation directory')
        return directory.parents[2]
    return Path(payload[3]).resolve()


def _output_path(kind, payload, root):
    symbol = str(payload[0]['symbol']).replace('.', '_')
    if '/' in symbol or '\\' in symbol or symbol in ('.', '..'):
        raise ValueError('unsafe training checkpoint symbol')
    if kind == 'observation':
        path = Path(payload[3]) / (symbol + '.parquet')
    elif kind == 'dataset':
        path = root / 'dataset' / payload[7].get('subdirectory', payload[6]) / symbol
    elif kind == 'probe':
        path = root / 'dataset/reference_quality/stocks' / symbol
    else:
        path = root / 'validation/reference_quality/stocks' / symbol
    return _inside(root, path)


def _checkpoint_files(summary, root, output):
    files = {}
    assets = summary.get('files', {}).values() if 'files' in summary else [summary]
    for asset in assets:
        path = _inside(root, asset['path'])
        if _file_hash(path) != asset['sha256']:
            raise ValueError('completed training checkpoint output SHA mismatch')
        files[path.relative_to(root).as_posix()] = asset['sha256']
    if output.is_dir():
        # Include manifests and compact account graphs, even when an original
        # summary lists the files before its final manifest is written.
        for path in output.iterdir():
            if path.is_file():
                files[_inside(root, path).relative_to(root).as_posix()] = _file_hash(path)
    if not files:
        raise ValueError('empty completed training checkpoint')
    return files


def _checkpoint_worker(kind, payload):
    state = _STATE
    root = _worker_root(kind, payload)
    dependencies = state.dependencies(payload, root)
    key = checkpoint_key(kind, payload)
    cached = state.cache.get_checkpoint(key, root)
    if cached is not None:
        state.stats['reused'] += 1
        return _decode(cached['typed_summary'])
    output = _output_path(kind, payload, root)
    if output.exists():
        # The caller owns the same-run lease. Preserve incomplete/unindexed
        # output as evidence before a worker that requires mkdir(exist_ok=False).
        quarantine = root / 'quarantine/training_fast' / kind
        quarantine.mkdir(parents=True, exist_ok=True)
        output.rename(quarantine / (output.name + '-' + uuid.uuid4().hex))
    previous = state.worker_deps
    state.worker_deps = dependencies
    try:
        result = state.originals[kind](payload)
    finally:
        state.worker_deps = previous
    # Recheck all physical inputs before the transaction makes reuse visible.
    for path, expected in dependencies.items():
        if _file_hash(Path(path)) != expected:
            raise ValueError('training checkpoint input changed while computing')
    files = _checkpoint_files(result, root, output)
    state.cache.put_checkpoint(key, root, {'typed_summary': _encode(result)}, files)
    state.stats['computed'] += 1
    return result


def _dataset_worker(payload):
    return _checkpoint_worker('dataset', payload)


def _observation_worker(payload):
    return _checkpoint_worker('observation', payload)


def _policy_worker(payload):
    """Persist the full original account chunk, never an aggregate proxy."""
    state = _STATE
    dependencies = state.dependencies(payload, Path(payload[3]).resolve())
    previous = state.worker_deps
    state.worker_deps = dependencies
    try:
        result = state.value('source_policy_stock', payload,
                             lambda: state.originals['policy'](payload), dependencies)
        for path, expected in dependencies.items():
            if _file_hash(Path(path)) != expected:
                raise ValueError('source policy checkpoint input changed')
        return result
    finally:
        state.worker_deps = previous


def _probe_worker(payload):
    return _checkpoint_worker('probe', payload)


def _qualification_worker(payload, *, persist=True):
    if persist:
        return _checkpoint_worker('qualification', payload)
    if _STATE.audit_results is not None:
        path = _STATE.audit_results.pop(_digest(payload), None)
        if path is None:
            raise ValueError('independent reference fee reconstruction identity differs')
        return _decode(json.loads(path.read_text()))
    return _STATE.originals['qualification'](payload, persist=False)


def _audit_measure_worker(payload):
    # Deliberately does not read the persisted qualification checkpoint: this is
    # an independent reconstruction for the original complete ledger comparison.
    root = Path(payload[3]).resolve()
    dependencies = _STATE.dependencies(payload, root)
    result = _STATE.originals['qualification'](payload, persist=False)
    for path, expected in dependencies.items():
        if _file_hash(Path(path)) != expected:
            raise ValueError('reference fee reconstruction input changed')
    return result


def _worker_entry(worker, payload, cache_path):
    with accelerated_training(1, cache_path):
        return worker(payload)


def _bounded_results(worker, payloads, cpu_workers, batch_size):
    """Refill on completion; retain original summation and population order."""
    if cpu_workers <= 1:
        for payload in payloads:
            yield worker(payload)
        return
    maximum = max(1, min(2 * cpu_workers, batch_size))
    iterator = iter(payloads)
    cache_path = _STATE.cache_path if _STATE else None
    with tempfile.TemporaryDirectory(prefix='khquant-training-results-') as directory:
        with ProcessPoolExecutor(max_workers=cpu_workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {}
            ready = set()
            submitted = 0
            next_result = 0
            def submit():
                nonlocal submitted
                try:
                    payload = next(iterator)
                except StopIteration:
                    return False
                pending[pool.submit(_worker_entry, worker, payload, cache_path)] = submitted
                submitted += 1
                return True
            for _ in range(maximum):
                if not submit():
                    break
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        result = future.result()
                        (Path(directory) / str(index)).write_text(json.dumps(_encode(result), allow_nan=False))
                        ready.add(index)
                        del result
                        submit()
                    while next_result in ready:
                        path = Path(directory) / str(next_result)
                        result = _decode(json.loads(path.read_text()))
                        path.unlink()
                        ready.remove(next_result)
                        next_result += 1
                        yield result
                if ready:
                    raise RuntimeError('ordered training result spool has a gap')
            finally:
                for future in pending:
                    future.cancel()


def _audit_reference_quality(root, block, config, expected_symbols, *, expected_records=None):
    state = _STATE
    original = state.originals['audit']
    if state.cpu_workers <= 1:
        return original(root, block, config, expected_symbols, expected_records=expected_records)
    from my_strategy.services import czsc_reference_quality as reference
    from my_strategy.services.model_training_center import _event_asset_signature
    from my_strategy.core.run_context import stable_hash
    cache_key = (str(Path(root).resolve()), stable_hash(block), stable_hash(config),
                 tuple(expected_symbols), stable_hash(expected_records))
    audit_cache = reference._AUDIT_CACHE.get(cache_key)
    if audit_cache and _event_asset_signature(tuple(Path(p) for p in audit_cache['paths'])) == audit_cache['signature']:
        return original(root, block, config, expected_symbols, expected_records=expected_records)
    root = Path(root).resolve()
    population = block['population']
    manifest_path = _inside(root, population['manifest_path'])
    if _file_hash(manifest_path) != population['manifest_sha256']:
        raise ValueError('reference quality frozen population changed')
    manifest = json.loads(manifest_path.read_text())
    records = manifest['records']
    if [record['symbol'] for record in records] != list(expected_symbols):
        raise ValueError('reference reconstruction population differs')
    if expected_records is not None and [r['source_record'] for r in records] != expected_records:
        raise ValueError('reference reconstruction sources differ')
    calendar = json.loads((root / 'data_snapshots/calendar.json').read_text())['dates']
    contract = reference._target_contract(config, population)
    payloads = [(r, calendar, config, str(root), population, block['bundles'], contract) for r in records]
    with tempfile.TemporaryDirectory(prefix='khquant-reference-audit-') as directory:
        measured = {}
        for index, result in enumerate(_bounded_results(_audit_measure_worker, iter(payloads),
                state.cpu_workers, config['stock_batch_size'])):
            path = Path(directory) / str(index)
            path.write_text(json.dumps(_encode(result), allow_nan=False))
            measured[_digest(payloads[index])] = path
        previous = state.audit_results
        state.audit_results = measured
        try:
            report = original(root, block, config, expected_symbols, expected_records=expected_records)
            if state.audit_results:
                raise ValueError('reference audit did not consume the complete reconstructed population')
            return report
        finally:
            state.audit_results = previous


@contextmanager
def accelerated_training(cpu_workers, cache_path=None):
    """Use scoped accelerators; the caller must own any reused run's lease.

    No generated file, model contract, input config or qualification gate changes.
    The yielded object exposes counters for tests and audit evidence.
    """
    global _STATE
    with _SCOPE_LOCK:
        if _STATE is not None:
            yield _STATE
            return
        from my_strategy.services import czsc_trend_pipeline as pipeline
        from my_strategy.services import czsc_trend_model as trend
        from my_strategy.services import czsc_trend_interval_model as interval
        from my_strategy.services import czsc_trend_features as features
        from my_strategy.services import czsc_trend_dataset as dataset
        from my_strategy.services import czsc_reference_quality as reference
        state = _Runtime(cpu_workers, cache_path)
        _STATE = state
        try:
            state.originals.update(interval=interval.load_head, trend=trend.load_trend_head,
                reference=reference.load_reference_quality_head,
                forward_trend=pipeline.forward_trend,
                event_prefix_rows=features.event_prefix_rows, dataset=pipeline._dataset_worker,
                observation=pipeline._trend_observation_worker, policy=pipeline._policy_worker,
                probe=reference._probe_worker,
                qualification=reference._qualification_worker, audit=reference.audit_reference_quality)
            for module, name, replacement in (
                    (pipeline, '_bounded_results', _bounded_results),
                    (pipeline, '_dataset_worker', _dataset_worker),
                    (pipeline, '_trend_observation_worker', _observation_worker),
                    (pipeline, '_policy_worker', _policy_worker),
                    (reference, '_probe_worker', _probe_worker),
                    (reference, '_qualification_worker', _qualification_worker),
                    (reference, 'audit_reference_quality', _audit_reference_quality),
                    (pipeline, 'load_head', _load_head), (interval, 'load_head', _load_head),
                    (pipeline, 'load_trend_head', _load_trend_head), (trend, 'load_trend_head', _load_trend_head),
                    (reference, 'load_reference_quality_head', _load_reference_head),
                    (pipeline, 'forward_trend', _forward_trend),
                    (features, 'event_prefix_rows', _event_prefix_rows),
                    (dataset, 'event_prefix_rows', _event_prefix_rows),
                    (reference, 'event_prefix_rows', _event_prefix_rows)):
                state.patch(module, name, replacement)
            yield state
            state.verify()
        finally:
            for module, name, original in reversed(state.patches):
                setattr(module, name, original)
            _STATE = None
