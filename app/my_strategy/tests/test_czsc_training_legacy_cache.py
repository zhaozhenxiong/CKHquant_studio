"""Legacy recovery proves completed populations and never repeats their workers."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import types

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_training_legacy_cache as adapter
from my_strategy.services import czsc_reference_quality as quality
from my_strategy.services import czsc_trend_resume as legacy
from my_strategy.services import czsc_trend_pipeline as pipeline
from my_strategy.services.czsc_account_artifacts import write_account_artifact
from my_strategy.services.czsc_trend_contracts import strategy_config


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _resign(root, manifest):
    manifest['population_hash'] = stable_hash({k: v for k, v in manifest.items() if k != 'population_hash'})
    _write(root / 'dataset/reference_quality/manifest.json', manifest)


def population(tmp_path, symbols=('000001.SZ', '600000.SH')):
    root = (tmp_path / 'run').resolve()
    root.mkdir()
    cfg = strategy_config()
    cfg.update(source_snapshot_hash='source', pipeline_contract_hash='pipeline', E1_policy_hash='e1', candidate_policy_hash='candidate',
               cpu_workers=2, stock_batch_size=10, training_end='2026-10-12')
    records = [dict(symbol=s, sha256='input-' + s) for s in symbols]
    calendar = ['2026-10-09', '2026-10-12']
    trends, ma = [dict(name='trend')], [dict(name='ma')]
    e1 = dict(stage='E1', failures=[], reference_population={}, population_end=None,
              reference_population_hash=stable_hash({}), source_snapshot_hash='source',
              records=[dict(symbol=s, stage='E1', reference_mode='E1', reference_population={}) for s in symbols], pooled_files={})
    for name in ('entries', 'exits', 'episodes'):
        path = root / 'dataset/E1' / (name + '.parquet')
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'value': [1]}).to_parquet(path, index=False)
        e1['pooled_files'][name] = dict(path=str(path), sha256=adapter._file_hash(path))
    _write(root / 'dataset/E1/manifest.json', e1)
    summaries, frames = [], []
    for index, record in enumerate(records):
        directory = root / 'dataset/reference_quality/stocks' / record['symbol'].replace('.', '_')
        directory.mkdir(parents=True)
        frame = pd.DataFrame({'value': [float(index), None], 'symbol': [record['symbol']] * 2})
        frame.to_parquet(directory / 'observations.parquet', index=False)
        frames.append(frame)
        write_account_artifact(directory / 'probes.json', dict(population=quality.POPULATION,
                               symbol=record['symbol'], probes=[]))
        files = {p.name: dict(path=p.relative_to(root).as_posix(), sha256=adapter._file_hash(p))
                 for p in directory.iterdir()}
        summaries.append(dict(symbol=record['symbol'], rows=2, files=files,
                              source_record_sha256=record['sha256'], source_record=record))
    pooled = root / 'dataset/reference_quality/observations.parquet'
    pd.concat(frames, ignore_index=True).to_parquet(pooled, index=False)
    protocol = quality.reference_protocol(cfg)
    _write(root / 'config/reference_quality_protocol.json', protocol)
    manifest = dict(population=quality.POPULATION, protocol=protocol, protocol_hash=protocol['protocol_hash'],
                    source_snapshot_hash='source', pipeline_contract_hash='pipeline', continuation_policy_hash='e1',
                    calendar_hash=stable_hash(calendar), probe_initial_cash=quality.PROBE_CASH,
                    records=summaries, failures=[], trend_catalog_hash=stable_hash(trends), ma_exit_catalog_hash=stable_hash(ma),
                    pooled_path=pooled.relative_to(root).as_posix(), pooled_sha256=adapter._file_hash(pooled))
    _resign(root, manifest)
    return dict(root=root, proof={}, config=cfg, records=records, calendar=dict(dates=calendar),
                trends=trends, ma=ma, cached_E1={r['symbol']: r for r in e1['records']}, reused={},
                initial_config=dict(cpu_workers=2)), manifest


def test_verify_complete_population_and_pooled_without_recalculation(tmp_path, monkeypatch):
    plan, _ = population(tmp_path)
    monkeypatch.setattr(quality, '_probe_worker', lambda payload: pytest.fail('probe replayed'))
    events = []
    result = adapter.verify_completed_reference(plan, progress=events.append)
    assert result is plan
    assert result['reused'] == dict(reference_population=2, reference_pooled=True, E1_pooled=True)
    assert len(plan['proof']) == 13
    assert events[-1]['verified_reference_stocks'] == 2
    assert str(plan['root'] / 'dataset/E1/entries.parquet') in plan['proof']


def test_reference_population_writer_produces_resumable_canonical_manifest(tmp_path, monkeypatch):
    plan, manifest = population(tmp_path)
    root = plan['root']
    summaries = [{k: v for k, v in item.items() if k != 'source_record'} for item in manifest['records']]
    monkeypatch.setattr(pipeline, '_bounded_results', lambda *args: iter(copy.deepcopy(summaries)))

    def stop_before_training(data):
        raise RuntimeError('population persisted; stop before model fitting')

    def subdir(*parts):
        path = root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(quality, '_label_counts', stop_before_training)
    with pytest.raises(RuntimeError, match='population persisted'):
        quality.run_reference_quality(plan['records'], plan['calendar']['dates'], plan['config'],
            types.SimpleNamespace(run_dir=lambda: root, subdir=subdir), plan['trends'], plan['ma'], 2, [])
    produced = json.loads((root / 'dataset/reference_quality/manifest.json').read_text())
    assert produced['pooled_path'] == 'dataset/reference_quality/observations.parquet'
    assert all('\\' not in asset['path'] for record in produced['records'] for asset in record['files'].values())
    assert adapter.verify_completed_reference(plan)['reused']['reference_pooled'] is True


@pytest.mark.parametrize('path', [
    r'dataset\reference_quality\observations.parquet',
    'dataset/reference_quality/../reference_quality/observations.parquet',
    './dataset/reference_quality/observations.parquet',
])
def test_reference_pool_rejects_resigned_noncanonical_path(tmp_path, path):
    plan, manifest = population(tmp_path)
    manifest['pooled_path'] = path
    _resign(plan['root'], manifest)
    with pytest.raises(ValueError, match='identity mismatch'):
        adapter.verify_completed_reference(plan)


@pytest.mark.parametrize('field', ['source_snapshot_hash', 'pipeline_contract_hash', 'continuation_policy_hash',
                                  'calendar_hash', 'trend_catalog_hash', 'ma_exit_catalog_hash', 'protocol_hash'])
def test_reference_cannot_reuse_foreign_identity(tmp_path, field):
    plan, manifest = population(tmp_path)
    manifest[field] = 'other'
    _resign(plan['root'], manifest)
    with pytest.raises(ValueError, match='identity mismatch'):
        adapter.verify_completed_reference(plan)


def test_reference_requires_all_E1_and_exact_aggregate(tmp_path):
    plan, _ = population(tmp_path)
    saved = plan['cached_E1'].pop('000001.SZ')
    with pytest.raises(ValueError, match='complete E1'):
        adapter.verify_completed_reference(plan)
    plan['cached_E1']['000001.SZ'] = {**saved, 'changed': True}
    with pytest.raises(ValueError, match='E1 manifest differs'):
        adapter.verify_completed_reference(plan)


@pytest.mark.parametrize('change', ['bytes', 'missing', 'record', 'inventory', 'order', 'graph_link'])
def test_stock_file_and_record_proof_cannot_be_faked(tmp_path, change):
    plan, manifest = population(tmp_path)
    first = manifest['records'][0]
    path = plan['root'] / first['files']['observations.parquet']['path']
    if change == 'bytes':
        path.write_bytes(path.read_bytes() + b'changed')
    elif change == 'missing':
        path.unlink()
    elif change == 'record':
        first['source_record'] = dict(symbol=first['symbol'], sha256='other')
    elif change == 'inventory':
        first['files'].pop(next(k for k in first['files'] if k.endswith('.gz')))
    elif change == 'order':
        manifest['records'].reverse()
    else:
        descriptor = plan['root'] / first['files']['probes.json']['path']
        body = json.loads(descriptor.read_text())
        body['asset_sha256'] = '0' * 64
        _write(descriptor, body)
        first['files']['probes.json']['sha256'] = adapter._file_hash(descriptor)
    _resign(plan['root'], manifest)
    with pytest.raises(ValueError):
        adapter.verify_completed_reference(plan)


@pytest.mark.parametrize('change', ['value', 'order', 'rows', 'schema'])
def test_even_resigned_pooled_file_must_equal_exact_stock_tables(tmp_path, change):
    plan, manifest = population(tmp_path)
    path = plan['root'] / manifest['pooled_path']
    frame = pd.read_parquet(path)
    if change == 'value':
        frame.loc[0, 'value'] = 99.
    elif change == 'order':
        frame = frame.iloc[::-1]
    elif change == 'rows':
        frame = frame.iloc[:-1]
    else:
        frame['value'] = frame['value'].astype('str')
    frame.to_parquet(path, index=False)
    manifest['pooled_sha256'] = adapter._file_hash(path)
    _resign(plan['root'], manifest)
    with pytest.raises(ValueError, match='pooled|schema'):
        adapter.verify_completed_reference(plan)


@pytest.mark.parametrize('symlink', [False, True])
def test_stock_assets_cannot_escape_run(tmp_path, symlink):
    plan, manifest = population(tmp_path)
    asset = manifest['records'][0]['files']['observations.parquet']
    path = plan['root'] / asset['path']
    outside = tmp_path / 'outside.parquet'
    outside.write_bytes(path.read_bytes())
    if symlink:
        path.unlink()
        path.symlink_to(outside)
    else:
        asset['path'] = str(outside)
        _resign(plan['root'], manifest)
    with pytest.raises(ValueError, match='escapes'):
        adapter.verify_completed_reference(plan)


def test_reference_replay_skips_only_exact_probe_payload_and_restores(tmp_path, monkeypatch):
    plan, _ = population(tmp_path)
    adapter.verify_completed_reference(plan)
    root = plan['root']
    before = {p: Path(p).stat().st_mtime_ns for p in plan['proof']}
    delegated = []
    def bounded(worker, payloads, workers, batch):
        delegated.append(worker)
        yield {'other': True}
    def run(records, calendar, cfg, context, trends, ma, workers, folds):
        output = list(pipeline._bounded_results(quality._probe_worker,
            ((r, calendar, cfg, str(root), trends, ma) for r in records), workers, 10))
        assert output == [{k: v for k, v in r.items() if k != 'source_record'} for r in plan['reference_population']['records']]
        assert list(pipeline._bounded_results(quality._qualification_worker, [], workers, 10)) == [{'other': True}]
        quality._save(root / 'dataset/reference_quality/manifest.json', plan['reference_population'])
        return output
    monkeypatch.setattr(pipeline, '_bounded_results', bounded)
    monkeypatch.setattr(quality, 'run_reference_quality', run)
    with adapter._reference_replay({'plan': plan}, tmp_path / 'cache.db', {}) as cache:
        quality.run_reference_quality(plan['records'], plan['calendar']['dates'], plan['config'],
            types.SimpleNamespace(run_dir=lambda: root), plan['trends'], plan['ma'], 2, [])
        with pytest.raises(ValueError, match='another computation'):
            quality.run_reference_quality(plan['records'], ['other'], plan['config'],
                types.SimpleNamespace(run_dir=lambda: root), plan['trends'], plan['ma'], 2, [])
    assert delegated == [quality._qualification_worker]
    assert pipeline._bounded_results is bounded
    assert quality.run_reference_quality is run
    assert {p: Path(p).stat().st_mtime_ns for p in before} == before


def test_reference_head_removes_only_empty_directory_and_reuses_checkpoint(tmp_path, monkeypatch):
    root = tmp_path / 'run'; root.mkdir()
    directory = root / 'models/reference_quality/fold'; directory.mkdir(parents=True)
    calls = []
    def fit(data, path, fold, cfg, contract):
        calls.append(True)
        assert not path.exists()
        path.mkdir(parents=True)
        (path / 'model.joblib').write_bytes(b'model')
        manifest = dict(fold=fold, contract=contract)
        _write(path / 'manifest.json', manifest)
        return manifest
    monkeypatch.setattr(quality, 'train_reference_head', fit)
    monkeypatch.setattr(quality, 'load_reference_quality_head', lambda path, **kw: json.loads((path / 'manifest.json').read_text()))
    data = pd.DataFrame({'value': [1.]})
    with adapter._reference_replay({'plan': dict(root=root, proof={})}, tmp_path / 'cache.db', {}) as cache:
        result = quality.train_reference_head(data, directory, {'name': 'fold'}, {}, {'source': 'a'})
        stamp = (directory / 'model.joblib').stat().st_mtime_ns
        assert quality.train_reference_head(data, directory, {'name': 'fold'}, {}, {'source': 'a'}) == result
        assert (directory / 'model.joblib').stat().st_mtime_ns == stamp
        with pytest.raises(ValueError, match='nonempty without a verified'):
            quality.train_reference_head(data, directory, {'name': 'fold'}, {}, {'source': 'changed'})
        (directory / 'model.joblib').write_bytes(b'corrupt')
        with pytest.raises(ValueError, match='nonempty without a verified'):
            quality.train_reference_head(data, directory, {'name': 'fold'}, {}, {'source': 'a'})
    assert calls == [True]
    assert quality.train_reference_head is fit


def test_reference_fit_insufficient_samples_has_no_checkpoint_or_partial_directory(tmp_path, monkeypatch):
    root = tmp_path / 'run'; root.mkdir()
    directory = root / 'models/fold'; directory.mkdir(parents=True)
    def fit(*args):
        raise ValueError('insufficient mature reference_quality rows')
    monkeypatch.setattr(quality, 'train_reference_head', fit)
    with adapter._reference_replay({'plan': dict(root=root, proof={})}, tmp_path / 'cache.db', {}):
        with pytest.raises(ValueError, match='insufficient mature'):
            quality.train_reference_head(pd.DataFrame(), directory, {}, {}, {})
    assert not directory.exists()


def orchestration(tmp_path, monkeypatch):
    from my_strategy.services import czsc_training_fast as fast
    root = tmp_path / 'runs/run'; root.mkdir(parents=True)
    code = tmp_path / 'code.py'; code.write_text('original')
    asset = root / 'frozen.json'; _write(asset, {'data': 1})
    _write(root / 'config/initial_config.json', {'cpu_workers': 2})
    plan = dict(root=root, proof={str(asset): adapter._file_hash(asset)}, initial_config={'cpu_workers': 2},
                reused={'reference_population': 2}, reference_population={'population_hash': 'population'})
    monkeypatch.setattr(adapter, 'ARTIFACT_RUNS_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(adapter, 'MY_STRATEGY_ROOT', tmp_path)
    monkeypatch.setattr(adapter, '_code_identity', lambda: {'code.py': adapter._file_hash(code)})
    monkeypatch.setattr(legacy, 'inspect_training_resume', lambda run_id, **kw: copy.deepcopy(plan))
    monkeypatch.setattr(adapter, 'verify_completed_reference', lambda plan, **kw: plan)
    requests = []
    monkeypatch.setattr(legacy, '_verify_request', lambda plan, **kw: requests.append(kw))
    events = []
    @contextmanager
    def lease(path):
        events.append('locked')
        try:
            yield
        finally:
            for p in (root / 'reports').glob('training_legacy_cache.*.json'):
                events.append(json.loads(p.read_text())['status'])
            events.append('released')
    @contextmanager
    def boundaries(plan, check_cancel=None):
        yield {}
    @contextmanager
    def accelerated(workers, **kw):
        events.append(('workers', workers))
        yield
    @contextmanager
    def replay(*args):
        yield types.SimpleNamespace(db_path=tmp_path / 'cache.db')
    monkeypatch.setattr(legacy, '_run_lease', lease)
    monkeypatch.setattr(legacy, '_reuse_boundaries', boundaries)
    monkeypatch.setattr(fast, 'accelerated_training', accelerated)
    monkeypatch.setattr(adapter, '_reference_replay', replay)
    monkeypatch.setattr(pipeline, '_save', _write)
    def resume(run_id, **kw):
        with legacy._run_lease(root):
            found = legacy.inspect_training_resume(run_id)
            assert str(code) in found['proof']
            assert kw['request'] == requests[0]
            with legacy._reuse_boundaries(found):
                if mutation:
                    mutation()
                pipeline._save(root / 'reports/trend_consensus.json', {'run_id': run_id})
            events.append('published')
            return {'run_id': run_id}
    mutation = None
    monkeypatch.setattr(legacy, 'resume_training_run', resume)
    def mutate(callback):
        nonlocal mutation
        mutation = callback
    return root, code, asset, requests, events, mutate, (lease, boundaries)


def test_resume_forwards_frozen_request_and_finishes_audit_under_lease(tmp_path, monkeypatch):
    root, code, asset, requests, events, mutate, originals = orchestration(tmp_path, monkeypatch)
    result = adapter.resume_legacy_cached('run', cpu_workers=2, device='mps', end='2026-10-12')
    assert result['run_id'] == 'run'
    assert requests == [dict(cpu_workers=2, device='mps', end='2026-10-12')]
    assert events[-3:] == ['published', 'completed', 'released']
    assert legacy._run_lease is originals[0] and legacy._reuse_boundaries is originals[1]
    assert not adapter._LOCK.locked()


@pytest.mark.parametrize('target', ['code', 'asset'])
def test_mutation_fails_before_result_save_and_failed_audit_stays_under_lease(tmp_path, monkeypatch, target):
    root, code, asset, requests, events, mutate, originals = orchestration(tmp_path, monkeypatch)
    mutate(lambda: (code if target == 'code' else asset).write_text('changed'))
    with pytest.raises(ValueError, match='changed'):
        adapter.resume_legacy_cached('run')
    assert not (root / 'reports/trend_consensus.json').exists()
    assert 'published' not in events and events[-2:] == ['failed', 'released']
    assert not adapter._LOCK.locked()


def test_verify_only_never_initializes_cache_or_writes_attempt(tmp_path, monkeypatch):
    from my_strategy.services.czsc_training_cache import TrainingCache
    root, code, asset, requests, events, mutate, originals = orchestration(tmp_path, monkeypatch)
    monkeypatch.setattr(TrainingCache, '__init__', lambda *args, **kw: pytest.fail('verify created cache'))
    result = adapter.resume_legacy_cached('run', verify_only=True, cache_path=tmp_path / 'absent.db')
    assert result['status'] == 'source_verified' and result['protected_files'] == 2
    assert not (tmp_path / 'absent.db').exists() and not (root / 'reports').exists()
    assert events == ['locked', 'released']


@pytest.mark.parametrize('within', [False, True])
def test_symlink_run_rejected_before_lease_or_cache(tmp_path, monkeypatch, within):
    runs = tmp_path / 'runs'; runs.mkdir()
    target = (runs if within else tmp_path) / 'outside'; target.mkdir()
    (runs / 'run').symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(adapter, 'ARTIFACT_RUNS_ROOT', runs)
    with pytest.raises(ValueError, match='escapes'):
        adapter.resume_legacy_cached('run', verify_only=True)
    assert not list(target.iterdir())


def test_code_inventory_failure_does_not_leak_thread_lock(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'; (runs / 'run').mkdir(parents=True)
    monkeypatch.setattr(adapter, 'ARTIFACT_RUNS_ROOT', runs)
    monkeypatch.setattr(adapter, '_code_identity', lambda: (_ for _ in ()).throw(FileNotFoundError('code')))
    with pytest.raises(FileNotFoundError):
        adapter.resume_legacy_cached('run', verify_only=True)
    assert not adapter._LOCK.locked()


def test_earlier_E1_boundary_keeps_original_reference_generation(tmp_path, monkeypatch):
    plan, _ = population(tmp_path)
    calls = []
    def run(*args):
        calls.append(args)
        return {'generated': True}
    monkeypatch.setattr(quality, 'run_reference_quality', run)
    with adapter._reference_replay({'plan': plan}, tmp_path / 'cache.db', {}):
        result = quality.run_reference_quality(plan['records'], plan['calendar']['dates'], plan['config'],
            types.SimpleNamespace(run_dir=lambda: plan['root']), plan['trends'], plan['ma'], 2, [])
    assert result == {'generated': True} and len(calls) == 1


def test_E1_only_resume_does_not_require_reference_manifest(tmp_path, monkeypatch):
    root, code, asset, requests, events, mutate, originals = orchestration(tmp_path, monkeypatch)
    original_inspect = legacy.inspect_training_resume
    def inspect(*args, **kw):
        plan = original_inspect(*args, **kw)
        plan.pop('reference_population')
        return plan
    monkeypatch.setattr(legacy, 'inspect_training_resume', inspect)
    monkeypatch.setattr(adapter, 'verify_completed_reference', lambda *args, **kw: pytest.fail('reference not present'))
    adapter.resume_legacy_cached('run')
    audit = json.loads(next((root / 'reports').glob('training_legacy_cache.*.json')).read_text())
    assert audit['reference_population_hash'] is None and audit['status'] == 'completed'


def test_existing_invalid_reference_manifest_cannot_fall_back_to_regeneration(tmp_path, monkeypatch):
    root, code, asset, requests, events, mutate, originals = orchestration(tmp_path, monkeypatch)
    _write(root / 'dataset/reference_quality/manifest.json', {'incomplete': True})
    def fail(*args, **kw):
        raise ValueError('invalid existing reference population')
    monkeypatch.setattr(adapter, 'verify_completed_reference', fail)
    with pytest.raises(ValueError, match='invalid existing reference'):
        adapter.resume_legacy_cached('run', verify_only=True)
    assert not (root / 'reports').exists()


def test_probe_payload_identity_change_fails_before_any_worker(tmp_path, monkeypatch):
    plan, _ = population(tmp_path)
    adapter.verify_completed_reference(plan)
    def run(records, calendar, cfg, context, trends, ma, workers, folds):
        changed = copy.deepcopy(cfg); changed['seed'] = 'other'
        return list(pipeline._bounded_results(quality._probe_worker,
                    ((r, calendar, changed, str(plan['root']), trends, ma) for r in records), workers, 10))
    monkeypatch.setattr(quality, 'run_reference_quality', run)
    monkeypatch.setattr(pipeline, '_bounded_results', lambda *args: pytest.fail('worker scheduled'))
    with adapter._reference_replay({'plan': plan}, tmp_path / 'cache.db', {}):
        with pytest.raises(ValueError, match='worker payload changed'):
            quality.run_reference_quality(plan['records'], plan['calendar']['dates'], plan['config'],
                types.SimpleNamespace(run_dir=lambda: plan['root']), plan['trends'], plan['ma'], 2, [])


def test_runtime_exit_failure_is_failed_before_lease_release(tmp_path, monkeypatch):
    from my_strategy.services import czsc_training_fast as fast
    root, code, asset, requests, events, mutate, originals = orchestration(tmp_path, monkeypatch)
    @contextmanager
    def failing_runtime(*args, **kw):
        yield
        raise ValueError('runtime model changed')
    monkeypatch.setattr(fast, 'accelerated_training', failing_runtime)
    with pytest.raises(ValueError, match='runtime model changed'):
        adapter.resume_legacy_cached('run')
    assert events[-2:] == ['failed', 'released']
    audit = json.loads(next((root / 'reports').glob('training_legacy_cache.*.json')).read_text())
    assert audit['status'] == 'failed' and 'runtime model changed' in audit['error']


def test_runtime_model_verification_precedes_report_write(tmp_path, monkeypatch):
    from my_strategy.services import czsc_training_fast as fast
    root, code, asset, requests, events, mutate, originals = orchestration(tmp_path, monkeypatch)
    def fail():
        raise ValueError('runtime model changed')
    monkeypatch.setattr(fast, '_STATE', types.SimpleNamespace(verify=fail))
    with pytest.raises(ValueError, match='runtime model changed'):
        adapter.resume_legacy_cached('run')
    assert not (root / 'reports/trend_consensus.json').exists()
    assert 'published' not in events and events[-2:] == ['failed', 'released']


def holding_refreeze_fixture(tmp_path):
    plan, _ = population(tmp_path)
    adapter.verify_completed_reference(plan)
    value = pipeline.freeze_holding_population({'reference_mode': 'E1', 'reference_population': {}},
                                               plan['root'], plan['config'], plan['records'])
    original = {'rows': 10, 'folds': [{'name': 'same', 'rows': 5}], 'holding_population_hash': 'old-clock-hash'}
    path = plan['root'] / 'validation/holding_population.json'
    _write(path, original)
    before = path.read_bytes()
    attempt = plan['root'] / 'reports/attempt.json'
    info = adapter._archive_holding_diagnostics(plan, attempt)
    return plan, value, original, path, before, info


def test_holding_refreeze_archives_original_and_changes_only_clock_derived_hash(tmp_path):
    plan, value, old, path, before, info = holding_refreeze_fixture(tmp_path)
    assert str(path) not in plan['proof']
    archive = plan['root'] / info['archived_path']
    assert archive.read_bytes() == before
    new = {**old, 'holding_population_hash': stable_hash(value)}
    evidence = adapter._save_refrozen_holding(plan, new, value, _write)
    assert json.loads(path.read_text()) == new
    assert archive.read_bytes() == before
    assert plan['proof'][str(archive)] == info['original_sha256']
    assert plan['proof'][str(path)] == evidence['new_sha256']
    assert evidence['frozen_before_L2_training_at'] == value['frozen_before_L2_training_at']


@pytest.mark.parametrize('change', ['rows', 'masks', 'binding', 'source', 'tamper', 'started_L2'])
def test_holding_refreeze_rejects_any_nonclock_change(tmp_path, change):
    plan, value, old, path, before, info = holding_refreeze_fixture(tmp_path)
    new = {**old, 'holding_population_hash': stable_hash(value)}
    if change == 'rows':
        new['rows'] += 1
    elif change == 'masks':
        new['folds'] = []
    elif change == 'binding':
        new['holding_population_hash'] = 'unbound'
    elif change == 'source':
        value['source_snapshot_hash'] = 'other'
        new['holding_population_hash'] = stable_hash(value)
    elif change == 'tamper':
        path.write_text('changed')
    else:
        (plan['root'] / 'dataset/final_population').mkdir()
    current = path.read_bytes()
    with pytest.raises(ValueError):
        adapter._save_refrozen_holding(plan, new, value, _write)
    assert path.read_bytes() == current
    assert (plan['root'] / info['archived_path']).read_bytes() == before
