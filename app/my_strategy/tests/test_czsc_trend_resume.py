"""Verified-stage reuse must never schedule completed native/account work."""
import copy
import json
from pathlib import Path
import types

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_trend_resume as resume
from my_strategy.services.czsc_account_artifacts import write_account_artifacts
from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.services.czsc_trend_features import ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS


def stock_fixture(tmp_path, *, stage='E1', rows=1, completed=False, source_hash='source'):
    root = tmp_path / 'run'
    directory = root / 'dataset' / stage / '000001_SZ'
    directory.mkdir(parents=True)
    cfg = strategy_config()
    cfg.update(training_end='2026-10-12', source_snapshot_hash='source', pipeline_contract_hash='pipeline',
               candidate_policy_hash='candidate', E0_policy_hash='e0', E1_policy_hash='e1', cpu_workers=10)
    context_cfg = cfg if stage == 'E1' else {k: v for k, v in cfg.items() if k != 'E1_policy_hash'}
    context = dict(source_hash=source_hash, calendar_hash='calendar', symbol='000001.SZ', mode=stage,
                   strategy='trend_consensus', source_revision_verified=True, entry_recipe_hash=None,
                   initial_cash=cfg['initial_cash'], exit_recipe_hash='e1' if stage == 'E1' else None,
                   config_hash=stable_hash(strategy_config(context_cfg)))
    reference = dict(symbol='000001.SZ', mode=stage, account_id='research:run:000001.SZ:reference',
                     version='trend_consensus', completed=completed, series_context=context,
                     series_context_hash=stable_hash(context), ledger=[])
    for key, columns in (('entries', ENTRY_FEATURE_COLUMNS), ('exits', EXIT_FEATURE_COLUMNS), ('episodes', ())):
        values = {column: [0.1] * rows for column in columns}
        values.update(stage=[stage] * rows, continuation_policy_hash=[cfg[stage + '_policy_hash']] * rows)
        if key != 'exits':
            values.update(candidate_policy_hash=['candidate'] * rows, reference_population_hash=[stable_hash({})] * rows)
        pd.DataFrame(values).to_parquet(directory / (key + '.parquet'), index=False)
    documents = {'reference_account.json': reference, 'candidate_clones.json': [], 'holding_pairs.json': []}
    write_account_artifacts(directory, documents)
    summary = dict(symbol='000001.SZ', stage=stage, reference_mode=stage, reference_population={}, population_end=None,
                   reference_population_hash=stable_hash({}), frozen_source_end='2026-10-12', effective_population_end='2026-10-12',
                   holding_checkpoint_upgrade=False, entry_rows=rows, exit_rows=rows)
    summary['files'] = {p.name: dict(path=str(p), sha256=resume._file_hash(p)) for p in directory.iterdir()}
    (directory / 'manifest.json').write_text(json.dumps(summary))
    return root, directory, summary, cfg


@pytest.mark.parametrize('stage,rows', [('E0', 1), ('E1', 1), ('E0', 0), ('E1', 0)])
def test_verified_stock_preserves_unclosed_or_empty_account(tmp_path, stage, rows):
    root, directory, summary, cfg = stock_fixture(tmp_path, stage=stage, rows=rows, completed=False)
    assert resume._verify_stock(root, summary, cfg, 'calendar', 'run', {}) == summary
    assert json.loads((directory / 'reference_account.json').read_text())['document'] == 'reference_account.json'


def test_stock_sha_failure_cannot_be_scheduled_as_reused(tmp_path):
    root, directory, summary, cfg = stock_fixture(tmp_path)
    with (directory / 'entries.parquet').open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='SHA mismatch'):
        resume._verify_stock(root, summary, cfg, 'calendar', 'run', {})


def test_valid_file_sha_cannot_hide_foreign_continuation(tmp_path):
    root, directory, summary, cfg = stock_fixture(tmp_path)
    path = directory / 'exits.parquet'
    frame = pd.read_parquet(path)
    frame['continuation_policy_hash'] = 'foreign'
    frame.to_parquet(path, index=False)
    summary['files'][path.name]['sha256'] = resume._file_hash(path)
    with pytest.raises(ValueError, match='continuation/source policy changed'):
        resume._verify_stock(root, summary, cfg, 'calendar', 'run', {})


def test_empty_stock_still_requires_real_source_bound_account(tmp_path):
    root, _, summary, cfg = stock_fixture(tmp_path, rows=0, source_hash='foreign')
    with pytest.raises(ValueError, match='another source'):
        resume._verify_stock(root, summary, cfg, 'calendar', 'run', {})


def test_asset_path_escape_rejected(tmp_path):
    root, _, summary, cfg = stock_fixture(tmp_path)
    outside = tmp_path / 'unrelated.parquet'
    outside.write_bytes(b'validbytes')
    summary['files']['entries.parquet'] = dict(path=str(outside), sha256=resume._file_hash(outside))
    with pytest.raises(ValueError, match='escapes'):
        resume._verify_stock(root, summary, cfg, 'calendar', 'run', {})


def test_same_run_lease_cannot_be_stolen_and_releases(tmp_path):
    with resume._run_lease(tmp_path):
        with pytest.raises(ValueError, match='active resume owner'):
            with resume._run_lease(tmp_path):
                pytest.fail('same run accepted another owner')
    with resume._run_lease(tmp_path):
        pass


def test_frozen_source_archive_is_not_rewritten(tmp_path):
    path = resume._ImmutableResumePath(tmp_path / 'source.py')
    path.write_bytes(b'original')
    stamp = path.stat().st_mtime_ns
    assert path.write_bytes(b'original') == 8
    assert path.stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match='replace a frozen'):
        path.write_bytes(b'changed')
    assert path.read_bytes() == b'original'


def test_dispatch_only_missing_E1_and_restore_original_functions(tmp_path, monkeypatch):
    from my_strategy.services import czsc_trend_pipeline as pipeline
    scheduled = []
    def bounded(worker, payloads, workers, batch):
        for payload in payloads:
            scheduled.append(payload[0]['symbol'])
            yield dict(symbol=payload[0]['symbol'], stage=payload[6], reference_population={})
    monkeypatch.setattr(pipeline, '_bounded_results', bounded)
    root = tmp_path / 'run'
    root.mkdir()
    original = pipeline._bounded_results
    cfg = dict(E1_policy_hash='same')
    plan = dict(root=root, proof={}, cached_E1={'A': dict(symbol='A', stage='E1', reference_population={})},
                config=cfg, observations={'records': []})
    options = dict(subdirectory='E1')
    payloads = [(dict(symbol=s), [], cfg, str(root), [], [], 'E1', options) for s in ('A', 'B', 'C')]
    with resume._reuse_boundaries(plan) as stats:
        result = list(pipeline._bounded_results(pipeline._dataset_worker, payloads, 10, 360))
        assert [r['symbol'] for r in result] == ['A', 'B', 'C']
        assert scheduled == ['B', 'C']
        assert stats == dict(E1_new=2, E1_reused=1)
    assert pipeline._bounded_results is original


def test_context_uses_original_identity_and_rejects_drift(tmp_path):
    from my_strategy.services import czsc_trend_pipeline as pipeline
    root = tmp_path / 'run'
    root.mkdir()
    cfg = {'original': True}
    plan = dict(root=root, run_id='run', proof={}, metadata=dict(config_hash=stable_hash(cfg)),
                source=dict(selected_symbols=['A'], snapshot_hash='source'), initial_config=dict(training_end='2026-10-12'))
    with resume._reuse_boundaries(plan):
        context = pipeline.create_run_context(config=cfg, stocks=['A'], data_version='source', as_of_date='2026-10-12')
        assert context.run_dir() == root
        assert context.run_id == 'run'
        with pytest.raises(ValueError, match='another resume run identity'):
            pipeline.create_run_context(config={'original': False}, stocks=['A'], data_version='source', as_of_date='2026-10-12')


@pytest.mark.parametrize('field,value', [('source_native_run_id', 'other'), ('end', '2026-10-14'),
                                      ('cpu_workers', 6), ('stock_batch_size', 60), ('cohort', ['B']),
                                      ('calendar_run_id', 'other'), ('device', 'cpu')])
def test_request_cannot_change_any_frozen_execution_identity(field, value):
    plan = dict(initial_config=dict(training_end='2026-10-12', cpu_workers=10, stock_batch_size=360),
                source=dict(source_native_run_id='native', selected_symbols=['A']),
                calendar=dict(source_run_id='calendar'), ma=[dict(name='fold')],
                manifests={('ma_exit', 'fold'): dict(actual_device='mps')})
    with pytest.raises(ValueError, match='resume requested'):
        resume._verify_request(plan, **{field: value})


def test_graph_context_ref_must_be_backward_and_bounded():
    with pytest.raises(ValueError, match='reference order'):
        resume._graph_value(dict(objects=[dict(d={'cycle': dict(ref=0)})]), dict(ref=0))


def test_completed_report_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, 'ARTIFACT_RUNS_ROOT', tmp_path)
    directory = tmp_path / 'run' / 'reports'
    directory.mkdir(parents=True)
    (directory / 'trend_consensus.json').write_text('{}')
    with pytest.raises(ValueError, match='completed training runs'):
        resume.inspect_training_resume('run')


def test_suspended_stock_end_is_bound_to_its_own_frozen_input(tmp_path):
    root, directory, summary, cfg = stock_fixture(tmp_path)
    summary['frozen_source_end'] = summary['effective_population_end'] = '2026-09-03'
    (directory / 'manifest.json').write_text(json.dumps(summary))
    assert resume._verify_stock(root, summary, cfg, 'calendar', 'run', {}, expected_source_end='2026-09-03') == summary
    with pytest.raises(ValueError, match='frozen independent account'):
        resume._verify_stock(root, summary, cfg, 'calendar', 'run', {}, expected_source_end='2026-10-12')
