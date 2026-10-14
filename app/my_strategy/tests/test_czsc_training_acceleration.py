from contextlib import contextmanager
from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd
import pytest

from my_strategy.services import czsc_training_acceleration as a
from my_strategy.services import czsc_training_fast as fast
from my_strategy.services import czsc_trend_pipeline as p
from my_strategy.services import czsc_reference_quality as q


class Context:
    def __init__(self, root):
        self.root = root
        self.run_id = root.name
    def run_dir(self):
        return self.root
    def subdir(self, *parts):
        path = self.root.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


@pytest.fixture
def runner(tmp_path, monkeypatch):
    root = tmp_path / 'run'; root.mkdir()
    ctx = Context(root)
    monkeypatch.setattr(a, 'code_identity', lambda: {'frozen': 'same'})
    monkeypatch.setattr(p, 'create_run_context', lambda **kw: ctx)
    @contextmanager
    def no_acceleration(*args, **kwargs):
        yield
    monkeypatch.setattr(fast, 'accelerated_training', no_acceleration)
    return ctx, tmp_path / 'cache.db'


def create(ctx):
    return p.create_run_context(config={'initial': 1}, stocks=['600000.SH'], data_version='source')


def test_frame_identity_tracks_values_dtype_index_and_contract():
    frame = pd.DataFrame({'x': [1., np.nan]}); frame.attrs['label_contract'] = {'a': 1}
    digest = a.frame_identity(frame)
    same = frame.copy(); assert a.frame_identity(same) == digest
    for changed in [frame.assign(x=[2., np.nan]), frame.astype({'x': 'float32'}), frame.set_axis([2, 3])]:
        assert a.frame_identity(changed) != digest
    same.attrs['label_contract'] = {'a': 2}
    assert a.frame_identity(same) != digest


def test_fitted_model_reused_only_after_complete_assets(runner, monkeypatch):
    ctx, db = runner; calls = []
    def fit(data, directory, fold, cfg, contract):
        calls.append(1); directory.mkdir(parents=True, exist_ok=True)
        (directory / 'checkpoint').write_bytes(b'complete')
        (directory / 'manifest.json').write_text('{}')
        return {'value': np.nan, 'tuple': (1, 2)}
    monkeypatch.setattr(p, 'train_trend_head', fit)
    data = pd.DataFrame({'x': [1., 2.]}); directory = ctx.root / 'models/trend/fold'
    for _ in range(2):
        with a.accelerated_pipeline(1, cache_path=db) as stats:
            create(ctx)
            result = p.train_trend_head(data, directory, {'name': 'fold'}, {'seed': 42}, contract={'target': 'x'})
    assert len(calls) == 1 and stats['stage_hits'] == 1
    assert np.isnan(result['value']) and result['tuple'] == (1, 2)
    (directory / 'checkpoint').write_bytes(b'tampered')
    with a.accelerated_pipeline(1, cache_path=db):
        create(ctx); p.train_trend_head(data, directory, {'name': 'fold'}, {'seed': 42}, contract={'target': 'x'})
    assert len(calls) == 2


def test_source_policy_replays_config_side_effect(runner, monkeypatch):
    ctx, db = runner; calls = []
    def policy(records, calendar, cfg, root, bundles, context, workers):
        calls.append(1); cfg['trend']['threshold'] = .7; cfg['consensus']['buy'] = .6
        (context.subdir('validation') / 'source_policy_accounts.jsonl').write_text('[]\n')
        (context.subdir('validation') / 'source_policy_selection.json').write_text('{}')
        return {'accounts_asset': {'path': 'validation/source_policy_accounts.jsonl'}}
    monkeypatch.setattr(p, 'select_source_policy', policy)
    for _ in range(2):
        cfg = {'trend': {'threshold': .5}, 'consensus': {'buy': .4}}
        with a.accelerated_pipeline(1, cache_path=db):
            create(ctx); p.select_source_policy([], [], cfg, ctx.root, [], ctx, 1)
        assert cfg == {'trend': {'threshold': .7}, 'consensus': {'buy': .6}}
    assert calls == [1]


def test_holding_policy_restores_bundle_mutation(runner, monkeypatch):
    ctx, db = runner; directory = ctx.subdir('models/ma'); (directory / 'weights').write_text('weights')
    calls = []
    def policy(data, bundle, root, config):
        calls.append(1); bundle['policy']['exit_threshold'] = .8
        return {'selected_threshold': .8}
    monkeypatch.setattr(p, 'select_holding_exit_policy', policy)
    for _ in range(2):
        bundle = {'heads': {'ma_exit': {'path': 'models/ma'}}, 'policy': {'exit_threshold': .5}}
        with a.accelerated_pipeline(1, cache_path=db):
            create(ctx); p.select_holding_exit_policy(pd.DataFrame({'x': [1]}), bundle, ctx.root, {})
        assert bundle['policy']['exit_threshold'] == .8
    assert calls == [1]


def test_failure_restores_bindings_and_records_state(runner, monkeypatch):
    ctx, db = runner; original = p.create_run_context
    with pytest.raises(RuntimeError, match='interruption'):
        with a.accelerated_pipeline(1, cache_path=db):
            create(ctx)
            raise RuntimeError('interruption')
    assert p.create_run_context is original
    state = json.loads((ctx.root / 'reports/training_acceleration.json').read_text())
    assert state['status'] == 'failed' and 'interruption' in state['error']
    with a.accelerated_pipeline(1, cache_path=db):
        create(ctx)


def test_code_drift_blocks_completion(runner, monkeypatch):
    ctx, db = runner
    with pytest.raises(ValueError, match='code changed'):
        with a.accelerated_pipeline(1, cache_path=db):
            create(ctx); monkeypatch.setattr(a, 'code_identity', lambda: {'frozen': 'changed'})
    state = json.loads((ctx.root / 'reports/training_acceleration.json').read_text())
    assert state['status'] == 'failed'


def test_resume_preserves_later_catalogue_prefix(runner):
    ctx, db = runner
    plan = {'root': ctx.root, 'run_id': ctx.run_id, 'metadata': {'config_hash': a.stable_hash({'initial': 1})},
            'source': {'selected_symbols': ['600000.SH'], 'snapshot_hash': 'source'}}
    path = ctx.subdir('models') / 'l2_bundles.json'; path.write_text('[{"name":"a"},{"name":"b"}]')
    before = path.stat().st_mtime_ns
    with a.accelerated_pipeline(1, cache_path=db, resume_plan=plan):
        create(ctx); p._save(path, [{'name': 'a'}])
    assert json.loads(path.read_text()) == [{'name': 'a'}, {'name': 'b'}]
    assert path.stat().st_mtime_ns == before


def test_failed_fit_never_publishes_checkpoint(runner, monkeypatch):
    ctx, db = runner; calls = []
    def fit(data, directory, fold, cfg, contract):
        calls.append(1); directory.mkdir(parents=True, exist_ok=True); (directory/'weights').write_text('partial')
        raise ValueError('insufficient mature')
    monkeypatch.setattr(p, 'train_trend_head', fit)
    for _ in range(2):
        with a.accelerated_pipeline(1, cache_path=db):
            create(ctx)
            with pytest.raises(ValueError, match='insufficient mature'):
                p.train_trend_head(pd.DataFrame({'x':[1]}), ctx.root/'models/trend', {}, {}, contract={})
    assert len(calls) == 2


@pytest.mark.parametrize('left,right', [(1, '1'), (True, 'True'), (None, 'None')])
def test_object_values_keep_type_in_cache_identity(left, right):
    assert a.frame_identity(pd.DataFrame({'x': pd.Series([left], dtype=object)})) != a.frame_identity(pd.DataFrame({'x': pd.Series([right], dtype=object)}))


def test_runtime_exit_failure_is_not_marked_complete(runner, monkeypatch):
    ctx, db = runner
    @contextmanager
    def failing_verification(*args, **kwargs):
        yield
        raise ValueError('final asset SHA drift')
    monkeypatch.setattr(fast, 'accelerated_training', failing_verification)
    with pytest.raises(ValueError, match='SHA drift'):
        with a.accelerated_pipeline(1, cache_path=db):
            create(ctx)
    state = json.loads((ctx.root / 'reports/training_acceleration.json').read_text())
    assert state['status'] == 'failed'


@pytest.fixture
def frozen_resume(tmp_path, monkeypatch):
    from my_strategy.services import czsc_trend_contracts as contracts
    root = tmp_path / 'runs/run'; root.mkdir(parents=True)
    monkeypatch.setattr(a, 'ARTIFACT_RUNS_ROOT', root.parent)
    monkeypatch.setattr(contracts, 'validate_training_config', lambda x: x)
    monkeypatch.setattr(p, 'GENERATING_FILES', ('services/producer.py',))
    monkeypatch.setattr(p, 'verify_full_population', lambda *args: None)
    monkeypatch.setattr(p, 'verify_source_generators', lambda *args: None)
    monkeypatch.setattr(a, 'resolve_frozen_asset', lambda x: Path(x))
    archive = root / 'config/source_code/services/producer.py'; archive.parent.mkdir(parents=True); archive.write_text('unchanged')
    code = {'services/producer.py': a._file_hash(archive)}
    monkeypatch.setattr(a, 'code_identity', lambda: code)
    record = {'symbol': '600000.SH'}
    for field, sha in [('raw_path','raw_sha256'), ('path','sha256'), ('native_path','native_sha256')]:
        file = tmp_path / field; file.write_text(field); record[field] = str(file); record[sha] = a._file_hash(file)
    native = {'status': 'completed', 'failures': [], 'dataset_records': [record],
              'source_snapshot': {'calendar_run_id': 'calendar'}, 'calendar_dates': ['2026-10-12']}
    native_path = root.parent / 'native/reports/native_source.json'; a._save(native_path, native)
    source = {'source_native_run_id': 'native', 'source_snapshot': native['source_snapshot'],
              'source_report_sha256': a._file_hash(native_path), 'selected_symbols': ['600000.SH'],
              'records_hash': a.stable_hash([record])}
    source['snapshot_hash'] = a.stable_hash(source)
    cfg = {'training_end': '2026-10-12', 'source_snapshot_hash': source['snapshot_hash'],
           'cpu_workers': 1, 'stock_batch_size': 360}
    meta = {'run_id': 'run', 'task': 'trend-consensus-train', 'config_hash': a.stable_hash(cfg),
            'data_version': source['snapshot_hash'], 'stocks': ['600000.SH'], 'as_of_date': cfg['training_end']}
    from my_strategy.services.czsc_training_cache import TrainingCache
    cache = TrainingCache(tmp_path / 'custom/cache.db')
    state = {'version': a.VERSION, 'run_id': 'run', 'code_identity': code, 'config_hash': a.stable_hash(cfg),
             'data_version': source['snapshot_hash'], 'device': 'cpu', 'cache_path': str(cache.db_path)}
    for name, value in [('reports/training_acceleration.json', state), ('config/initial_config.json', cfg),
            ('metadata.json', meta), ('data_snapshots/source.json', source), ('data_snapshots/records.json', [record]),
            ('data_snapshots/calendar.json', {'verified': True, 'dates': ['2026-10-12'], 'source_run_id': 'calendar'}),
            ('config/generating_code_hashes.json', code)]:
        a._save(root / name, value)
    return root, cache


def test_resume_verifies_source_and_follows_original_cache(frozen_resume):
    root, cache = frozen_resume
    assert a.inspect_fast_resume('run')['state']['cache_path'] == str(cache.db_path)
    result = a.resume_fast_training('run', verify_only=True)
    assert result['status'] == 'source_verified'
    assert result['checkpoint_validation'] == 'on_replay'
    with pytest.raises(ValueError, match='another checkpoint database'):
        a.resume_fast_training('run', verify_only=True, cache_path=root / 'empty.db')
    cache.db_path.unlink()
    with pytest.raises(ValueError, match='database is missing'):
        a.resume_fast_training('run', verify_only=True)


@pytest.mark.parametrize('part', ['source', 'archive', 'calendar', 'device', 'config'])
def test_resume_rejects_frozen_input_drift(frozen_resume, part):
    root, _ = frozen_resume
    request = {}
    if part == 'source':
        record = json.loads((root / 'data_snapshots/records.json').read_text())[0]
        Path(record['path']).write_text('tamper')
    elif part == 'archive':
        (root / 'config/source_code/services/producer.py').write_text('changed')
    elif part == 'calendar':
        a._save(root / 'data_snapshots/calendar.json', {'verified': True, 'dates': ['2026-10-13'], 'source_run_id': 'calendar'})
    elif part == 'device':
        request['device'] = 'mps'
    else:
        request['config'] = {'different': True}
    with pytest.raises(ValueError):
        a.resume_fast_training('run', verify_only=True, **request)


@pytest.mark.parametrize('external', [False, True])
def test_resume_rejects_symlink_before_creating_lease(tmp_path, monkeypatch, external):
    runs = tmp_path / 'runs'; runs.mkdir()
    target = tmp_path / 'outside' if external else runs / 'other'
    target.mkdir()
    (runs / 'run').symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(a, 'ARTIFACT_RUNS_ROOT', runs)
    with pytest.raises(ValueError, match='outside artifact root'):
        a.resume_fast_training('run', verify_only=True)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize('partial', [False, True])
def test_reference_fit_handles_precreated_directory_and_reuses_complete_head(runner, monkeypatch, partial):
    ctx, db = runner; directory = ctx.subdir('models/reference/fold'); calls = []
    if partial:
        (directory / 'interrupted-weights').write_text('retain evidence')
    def fit(data, path, fold, cfg, contract):
        # The original trainer makes this exact mkdir call, without exist_ok.
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text('{}')
        (path / 'model.joblib').write_text('complete')
        calls.append(1)
        return {'fitted': True}
    monkeypatch.setattr(q, 'train_reference_head', fit)
    for _ in range(2):
        with a.accelerated_pipeline(1, cache_path=db):
            create(ctx)
            assert q.train_reference_head(pd.DataFrame({'x': [1]}), directory, {}, {}, {}) == {'fitted': True}
    assert calls == [1]
    if partial:
        preserved = list((ctx.root / 'quarantine/training_fast/reference_head').glob('*/interrupted-weights'))
        assert len(preserved) == 1 and preserved[0].read_text() == 'retain evidence'


def test_holding_population_preserves_freeze_time_and_dataset_binding(runner, monkeypatch):
    ctx, db = runner; directory = ctx.subdir('dataset/E1'); calls = []
    (directory / 'manifest.json').write_text('{}')
    (directory / 'exits.parquet').write_text('verified paired rows')
    def freeze(dataset, root, cfg, records):
        calls.append(1)
        return {'frozen_before_L2_training_at': 'attempt-' + str(len(calls))}
    monkeypatch.setattr(p, 'freeze_holding_population', freeze)
    audits = []
    monkeypatch.setattr(p, 'audit_holding_population', lambda value, *args: audits.append(copy.deepcopy(value)))
    for _ in range(2):
        data = {'reference_population': {}, 'reference_mode': 'E1'}
        with a.accelerated_pipeline(1, cache_path=db):
            create(ctx)
            result = p.freeze_holding_population(data, ctx.root, {}, [])
        assert data['holding_population'] == result == {'frozen_before_L2_training_at': 'attempt-1'}
    assert calls == [1] and len(audits) == 2
    assert json.loads((ctx.root / 'data_snapshots/holding_population.json').read_text()) == result
