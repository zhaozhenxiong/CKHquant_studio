"""Accelerators must preserve results, population order and asset boundaries."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest

from my_strategy.services import czsc_training_fast as fast
from my_strategy.services import czsc_trend_pipeline as pipeline
from my_strategy.services import czsc_trend_model as trend
from my_strategy.services import czsc_reference_quality as reference
from my_strategy.services.czsc_trend_support import _file_hash


def _spawn_probe(value):
    """Importable under a real spawn process, unlike a monkeypatched closure."""
    from my_strategy.services import czsc_training_fast as child
    from my_strategy.services import czsc_trend_pipeline as child_pipeline
    return {'value': value, 'active': child._STATE is not None,
            'patched': child_pipeline._bounded_results is child._bounded_results}


def test_lossless_codec_preserves_missing_and_scalar_types():
    value = {'missing': float('nan'), 'infinity': float('inf'), 'negative': -float('inf'),
             'tuple': (1, 2), 'list': ['dict', [1]], 'numpy': np.float32(.1),
             'array': np.array([np.nan, 1.], dtype=np.float32), 'timestamp': pd.Timestamp('2026-01-02')}
    result = fast._decode(json.loads(json.dumps(fast._encode(value), allow_nan=False)))
    assert math.isnan(result['missing']) and result['infinity'] == float('inf')
    assert result['negative'] == -float('inf') and result['tuple'] == (1, 2)
    assert result['list'] == ['dict', [1]] and type(result['numpy']) is np.float32
    np.testing.assert_array_equal(result['array'], value['array'])
    assert result['timestamp'] == value['timestamp']


def test_scheduler_refills_past_slow_head_and_preserves_order(tmp_path, monkeypatch):
    started_fifth = threading.Event()
    started = []
    lock = threading.Lock()
    def worker(index):
        with lock:
            started.append(index)
        if index == 4:
            started_fifth.set()
        if index == 0:
            assert started_fifth.wait(3), 'refill blocked behind slow first stock'
        return {'value': index}
    class Pool(ThreadPoolExecutor):
        def __init__(self, max_workers, mp_context):
            super().__init__(max_workers=max_workers)
        def submit(self, entry, worker, payload, cache_path):
            return super().submit(worker, payload)
    monkeypatch.setattr(fast, 'ProcessPoolExecutor', Pool)
    with fast.accelerated_training(2, tmp_path / 'cache.db'):
        results = list(pipeline._bounded_results(worker, iter(range(12)), 2, 4))
    assert [r['value'] for r in results] == list(range(12))
    assert sorted(started) == list(range(12))


def test_real_spawn_enables_child_runtime(tmp_path):
    with fast.accelerated_training(2, tmp_path / 'cache.db'):
        results = list(pipeline._bounded_results(_spawn_probe, iter(range(3)), 2, 4))
    assert results == [{'value': i, 'active': True, 'patched': True} for i in range(3)]


def _payload(tmp_path, stage='E1', *, kind='dataset'):
    root = tmp_path / 'run'
    root.mkdir(exist_ok=True)
    source = tmp_path / 'source.parquet'
    if not source.exists():
        source.write_bytes(b'frozen_source')
    record = {'symbol': '000001.SZ', 'path': str(source), 'sha256': _file_hash(source),
              'raw_attrs': {'price_basis': 'actual'}}
    cfg = {'source_snapshot_hash': 'source', 'training_end': '2026-10-12', 'fees': .0003}
    payload = (record, ['2026-10-09', '2026-10-12'], cfg, str(root), [], [])
    if kind == 'dataset':
        payload += (stage, {'subdirectory': stage, 'reference_population': None})
    return payload


def _fake_dataset(payload):
    root = Path(payload[3])
    output = root / 'dataset' / payload[7]['subdirectory'] / payload[0]['symbol'].replace('.', '_')
    output.mkdir(parents=True)
    path = output / 'entries.parquet'
    path.write_text(json.dumps({'stage': payload[6], 'fees': payload[2]['fees']}))
    result = {'symbol': payload[0]['symbol'], 'stage': payload[6], 'label': float('nan'),
              'files': {'entries.parquet': {'path': str(path), 'sha256': _file_hash(path)}}}
    (output / 'manifest.json').write_text('committed')
    return result


def test_stock_checkpoint_reuses_only_committed_complete_files(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    calls = []
    def worker(payload):
        calls.append(payload)
        return _fake_dataset(payload)
    monkeypatch.setattr(pipeline, '_dataset_worker', worker)
    with fast.accelerated_training(1, tmp_path / 'cache.db') as state:
        first = pipeline._dataset_worker(payload)
        before = {str(p): p.stat().st_mtime_ns for p in Path(payload[3]).rglob('*') if p.is_file()}
        second = pipeline._dataset_worker(payload)
        after = {str(p): p.stat().st_mtime_ns for p in Path(payload[3]).rglob('*') if p.is_file()}
    assert len(calls) == 1 and state.stats['reused'] == 1 and before == after
    assert first['files'] == second['files'] and math.isnan(second['label'])


@pytest.mark.parametrize('mutation', ['fees', 'calendar', 'stage', 'population', 'source_attrs'])
def test_checkpoint_does_not_mix_contracts(tmp_path, monkeypatch, mutation):
    original = _payload(tmp_path)
    changed = list(copy.deepcopy(original))
    if mutation == 'fees':
        changed[2]['fees'] *= 2
    elif mutation == 'calendar':
        changed[1].append('2026-10-13')
    elif mutation == 'stage':
        changed[6] = 'E0'
        changed[7]['subdirectory'] = 'E0'
    elif mutation == 'population':
        changed[7]['reference_population'] = {'population': 'strict_prior_L2', 'fold': 'future'}
    else:
        changed[0]['raw_attrs']['price_basis'] = 'foreign'
    calls = []
    def worker(payload):
        calls.append(payload)
        return _fake_dataset(payload)
    monkeypatch.setattr(pipeline, '_dataset_worker', worker)
    with fast.accelerated_training(1, tmp_path / 'cache.db'):
        pipeline._dataset_worker(original)
        pipeline._dataset_worker(tuple(changed))
    assert len(calls) == 2


def test_checkpoint_checks_source_sha_before_reuse(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    monkeypatch.setattr(pipeline, '_dataset_worker', _fake_dataset)
    with fast.accelerated_training(1, tmp_path / 'cache.db'):
        pipeline._dataset_worker(payload)
        Path(payload[0]['path']).write_bytes(b'tampered_source')
        with pytest.raises(ValueError, match='input SHA mismatch'):
            pipeline._dataset_worker(payload)


@pytest.mark.parametrize('asset', ['entries.parquet', 'manifest.json'])
def test_checkpoint_tamper_quarantines_exact_output_and_recomputes(tmp_path, monkeypatch, asset):
    payload = _payload(tmp_path)
    monkeypatch.setattr(pipeline, '_dataset_worker', _fake_dataset)
    with fast.accelerated_training(1, tmp_path / 'cache.db') as state:
        pipeline._dataset_worker(payload)
        output = Path(payload[3]) / 'dataset/E1/000001_SZ'
        (output / asset).write_text('incomplete')
        pipeline._dataset_worker(payload)
    assert state.stats['computed'] == 2 and state.stats['reused'] == 0
    quarantined = list((Path(payload[3]) / 'quarantine/training_fast/dataset').iterdir())
    assert len(quarantined) == 1 and (quarantined[0] / asset).read_text() == 'incomplete'


def test_failed_worker_has_no_checkpoint(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    def failed(payload):
        _fake_dataset(payload)
        raise RuntimeError('stopped before result commit')
    monkeypatch.setattr(pipeline, '_dataset_worker', failed)
    with fast.accelerated_training(1, tmp_path / 'cache.db') as state:
        with pytest.raises(RuntimeError, match='stopped'):
            pipeline._dataset_worker(payload)
        assert state.cache.get_checkpoint(fast.checkpoint_key('dataset', payload), Path(payload[3])) is None


def test_trend_observation_checkpoint_is_per_stock_file(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    directory = Path(payload[3]) / 'dataset/trend/stocks'
    directory.mkdir(parents=True)
    payload = (*payload[:3], str(directory))
    calls = []
    def worker(payload):
        calls.append(payload)
        path = Path(payload[3]) / (payload[0]['symbol'].replace('.', '_') + '.parquet')
        path.write_text('observation')
        return {'symbol': payload[0]['symbol'], 'path': str(path), 'sha256': _file_hash(path), 'label_contract': {'causal': True}}
    monkeypatch.setattr(pipeline, '_trend_observation_worker', worker)
    with fast.accelerated_training(1, tmp_path / 'cache.db'):
        first = pipeline._trend_observation_worker(payload)
        assert pipeline._trend_observation_worker(payload) == first
    assert len(calls) == 1


def _fake_model(path):
    return {'checkpoint': 'model.joblib', 'checkpoint_sha256': _file_hash(Path(path) / 'model.joblib'),
            'manifest_sha256': 'logical-manifest', 'target_contract': {'source': 'frozen'}}


def test_model_loader_validates_once_and_returns_defensive_copy(tmp_path, monkeypatch):
    (tmp_path / 'manifest.json').write_text('frozen_manifest')
    (tmp_path / 'model.joblib').write_bytes(b'model')
    calls = []
    def loader(path):
        calls.append(path)
        return _fake_model(path)
    monkeypatch.setattr(trend, 'load_trend_head', loader)
    with fast.accelerated_training(1, tmp_path / 'cache.db') as state:
        first = pipeline.load_trend_head(tmp_path)
        first['target_contract']['source'] = 'caller_mutation'
        assert pipeline.load_trend_head(tmp_path)['target_contract']['source'] == 'frozen'
    assert len(calls) == 1 and state.stats['model_hits'] == 1


def test_model_stat_change_fails_before_stale_reuse(tmp_path, monkeypatch):
    (tmp_path / 'manifest.json').write_text('manifest')
    model = tmp_path / 'model.joblib'
    model.write_bytes(b'model')
    monkeypatch.setattr(trend, 'load_trend_head', _fake_model)
    with pytest.raises(ValueError, match='immutable training model'):
        with fast.accelerated_training(1, tmp_path / 'cache.db'):
            pipeline.load_trend_head(tmp_path)
            model.write_bytes(b'other')
            pipeline.load_trend_head(tmp_path)
    assert fast._STATE is None


def test_final_model_sha_verification_detects_change_without_more_predictions(tmp_path, monkeypatch):
    (tmp_path / 'manifest.json').write_text('manifest')
    model = tmp_path / 'model.joblib'
    model.write_bytes(b'model')
    monkeypatch.setattr(trend, 'load_trend_head', _fake_model)
    with pytest.raises(ValueError, match='model asset changed'):
        with fast.accelerated_training(1, tmp_path / 'cache.db'):
            pipeline.load_trend_head(tmp_path)
            model.write_bytes(b'other')
    assert fast._STATE is None


def test_value_cache_binds_frame_cells_attrs_config_and_calendar(tmp_path):
    calls = []
    frame = pd.DataFrame({'date': ['2026-10-12'], 'close': [10.]})
    frame.attrs['price_basis'] = 'actual'
    arguments = {'frame': frame, 'calendar': ['2026-10-12'], 'config': {'fee': .001}}
    def compute():
        calls.append(1)
        return {'p': .25, 'missing': np.nan, 'nested': {'values': [1]}}
    with fast.accelerated_training(1, tmp_path / 'cache.db') as state:
        first = state.value('test', arguments, compute)
        first['nested']['values'].append(2)
        assert state.value('test', arguments, compute)['nested']['values'] == [1]
        frame.loc[0, 'close'] += 1
        state.value('test', arguments, compute)
        frame.attrs['price_basis'] = 'unverified'
        state.value('test', arguments, compute)
        arguments['config']['fee'] *= 2
        state.value('test', arguments, compute)
        arguments['calendar'].append('2026-10-13')
        state.value('test', arguments, compute)
    assert len(calls) == 5


def test_public_computation_reuses_across_unrelated_l2_catalogues(tmp_path):
    calls = []
    arguments = {'events': [{'event_id': 'a', 'side': 'BUY'}], 'calendar': ['2026-10-12']}
    model = tmp_path / 'trend.model'
    model.write_bytes(b'trend-v1')
    def compute():
        calls.append(1)
        return {'value': len(calls)}
    with fast.accelerated_training(1, tmp_path / 'cache.db') as state:
        state.worker_deps = {'unrelated/prior-entry-fold-1': 'unused-file'}
        assert state.value('trend-test', arguments, compute, {str(model): _file_hash(model)}) == {'value': 1}
        state.worker_deps = {'unrelated/prior-entry-fold-2': 'different-unused-file'}
        assert state.value('trend-test', arguments, compute, {str(model): _file_hash(model)}) == {'value': 1}
        model.write_bytes(b'trend-v2')
        assert state.value('trend-test', arguments, compute, {str(model): _file_hash(model)}) == {'value': 2}
        arguments['events'][0]['side'] = 'SELL'
        assert state.value('trend-test', arguments, compute, {str(model): _file_hash(model)}) == {'value': 3}
    assert len(calls) == 3


def test_policy_base_mutation_is_not_cached(tmp_path, monkeypatch):
    from my_strategy.services import czsc_trend_consensus as consensus
    calls = []
    def prepare(frame, market_dates, *, _policy_base_rows=None):
        calls.append(1)
        _policy_base_rows.append('original-side-effect')
        return {'rows': []}
    monkeypatch.setattr(consensus, 'prepare_trend_decisions', prepare)
    with fast.accelerated_training(1, tmp_path / 'cache.db'):
        rows = []
        consensus.prepare_trend_decisions([], [], _policy_base_rows=rows)
        consensus.prepare_trend_decisions([], [], _policy_base_rows=rows)
    assert len(calls) == 2 and rows == ['original-side-effect', 'original-side-effect']


def test_decision_preparation_remains_original_and_observes_native_mutations(tmp_path, monkeypatch):
    from my_strategy.services import czsc_trend_consensus as consensus
    frame = pd.DataFrame({'date': ['2026-10-12'], 'symbol': ['000001.SZ']})
    pack = {'evaluations': [{'date': '2026-10-12', 'invalidated_event_ids': ['event-a']}], 'events': []}
    calls = []
    def prepare(frame, market_dates, evidence_pack=None, native_pack=None, _policy_base_rows=None):
        calls.append(1)
        return copy.deepcopy({'rows': evidence_pack['evaluations']})
    monkeypatch.setattr(consensus, 'prepare_trend_decisions', prepare)
    with fast.accelerated_training(1, tmp_path / 'cache.db') as state:
        assert consensus.prepare_trend_decisions is prepare
        first = consensus.prepare_trend_decisions(frame, ['2026-10-12'], evidence_pack=pack)
        first['rows'][0]['invalidated_event_ids'].append('caller-only')
        second = consensus.prepare_trend_decisions(frame, ['2026-10-12'], evidence_pack=pack)
        assert second['rows'][0]['invalidated_event_ids'] == ['event-a']
        pack['evaluations'][0]['invalidated_event_ids'].append('native-mutation')
        third = consensus.prepare_trend_decisions(frame, ['2026-10-12'], evidence_pack=pack)
        assert third['rows'][0]['invalidated_event_ids'] == ['event-a', 'native-mutation']
    assert len(calls) == 3


@pytest.mark.parametrize('left,right', [(1, '1'), (True, 'True'), (None, 'None'), ([1], (1,)),
                                      (np.float32(1), 1.), ({1: 'a'}, {'1': 'a'})])
def test_native_content_hash_keeps_non_json_types_distinct(left, right):
    assert fast._json_content_digest({'value': left}) != fast._json_content_digest({'value': right})


def test_frame_identity_binds_category_order_even_when_observed_values_match():
    first = pd.DataFrame({'value': pd.Categorical(['a'], categories=['a', 'b'], ordered=False)})
    reordered = pd.DataFrame({'value': pd.Categorical(['a'], categories=['b', 'a'], ordered=False)})
    ordered = pd.DataFrame({'value': pd.Categorical(['a'], categories=['a', 'b'], ordered=True)})
    assert len({fast._digest(frame) for frame in (first, reordered, ordered)}) == 3
    assert fast._digest(first) == fast._digest(first.copy(deep=True))


def test_worker_generating_identity_includes_cache_implementation():
    hashes = fast._generator_hashes()
    assert str(Path(fast.__file__).with_name('czsc_training_cache.py').resolve()) in hashes


def test_policy_stock_cache_preserves_full_ledger_and_variant_identity(tmp_path, monkeypatch):
    base = _payload(tmp_path, kind='probe')
    payload = (*base[:4], {}, [{'cross': .55, 'buy_base': .45}])
    calls = []
    def policy(payload):
        calls.append(payload)
        return [{'variant': payload[5][0], 'symbol': payload[0]['symbol'],
                 'ledger': [{'date': '2026-10-12', 'shares': 100, 'commission': 5.}],
                 'rejections': [], 'open_position': None, 'metrics': {'drawdown': np.float64(.1)}}]
    monkeypatch.setattr(pipeline, '_policy_worker', policy)
    with fast.accelerated_training(1, tmp_path / 'cache.db'):
        first = pipeline._policy_worker(payload)
        assert pipeline._policy_worker(payload) == first
        changed = copy.deepcopy(payload)
        changed[5][0]['cross'] = .6
        assert pipeline._policy_worker(changed)[0]['variant']['cross'] == .6
    assert len(calls) == 2


def test_reference_audit_consumes_only_exact_independent_measurement(tmp_path, monkeypatch):
    payload = _payload(tmp_path, kind='probe')
    calls = []
    def qualify(payload, *, persist=True):
        calls.append(persist)
        return {'symbol': payload[0]['symbol'], 'windows': ['independent-ledger'], 'rows_digest': 'rows'}
    monkeypatch.setattr(reference, '_qualification_worker', qualify)
    with fast.accelerated_training(1, tmp_path / 'cache.db') as state:
        measurement = fast._audit_measure_worker(payload)
        path = tmp_path / 'audit.json'
        path.write_text(json.dumps(fast._encode(measurement)))
        state.audit_results = {fast._digest(payload): path}
        assert reference._qualification_worker(payload, persist=False) == measurement
        with pytest.raises(ValueError, match='reconstruction identity differs'):
            reference._qualification_worker(payload, persist=False)
    assert calls == [False]


def test_complete_reference_fee_audit_matches_original_real_ledgers(tmp_path, monkeypatch):
    from my_strategy.tests.test_reference_quality import qualified_reference_fixture
    root, report, config, *_ = qualified_reference_fixture(tmp_path, monkeypatch)
    records = report['dataset_records']
    symbols = [r['symbol'] for r in records]
    expected = reference.audit_reference_quality(root, report['reference_quality'], config, symbols,
                                                expected_records=records)
    reference._AUDIT_CACHE.clear()
    rebuilt = []
    # Keep this financial fixture's synthetic read_snapshot monkeypatch local;
    # actual spawn and completion scheduling have separate integration tests.
    def measure_all(worker, payloads, workers, batch_size):
        assert worker is fast._audit_measure_worker and workers == 2
        for payload in payloads:
            rebuilt.append(payload[0]['symbol'])
            yield worker(payload)
    monkeypatch.setattr(fast, '_bounded_results', measure_all)
    with fast.accelerated_training(2, tmp_path / 'cache.db'):
        actual = reference.audit_reference_quality(root, report['reference_quality'], config, symbols,
                                                  expected_records=records)
    assert rebuilt == symbols and actual == expected
    assert actual['passed'] is True


def test_scope_restores_every_original_on_error(tmp_path):
    originals = (pipeline._bounded_results, pipeline._dataset_worker, pipeline.forward_trend,
                 reference._qualification_worker, reference.audit_reference_quality, trend.load_trend_head)
    with pytest.raises(RuntimeError, match='original training failure'):
        with fast.accelerated_training(1, tmp_path / 'cache.db'):
            raise RuntimeError('original training failure')
    assert originals == (pipeline._bounded_results, pipeline._dataset_worker, pipeline.forward_trend,
                         reference._qualification_worker, reference.audit_reference_quality, trend.load_trend_head)
    assert fast._STATE is None
