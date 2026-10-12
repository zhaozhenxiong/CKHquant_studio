import hashlib
from pathlib import Path

import pytest

from my_strategy.scripts.train_trend_consensus import install_finalization_path_adapter
from my_strategy.services import czsc_trend_finalization, czsc_trend_pipeline


def test_cli_path_adapter_preserves_hash_and_frozen_pipeline_source(tmp_path, monkeypatch):
    source = Path(czsc_trend_pipeline.__file__)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    asset = tmp_path / 'immutable-asset'
    asset.write_bytes(b'actual checkpoint bytes\x00')
    expected = hashlib.sha256(asset.read_bytes()).hexdigest()
    original = czsc_trend_pipeline._file_hash
    monkeypatch.setattr(czsc_trend_pipeline, '_file_hash', original)
    install_finalization_path_adapter()
    assert czsc_trend_pipeline._file_hash(asset) == expected
    assert czsc_trend_pipeline._file_hash(str(asset)) == expected
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_completion_recovery_rejects_existing_immutable_report(tmp_path, monkeypatch):
    monkeypatch.setattr(czsc_trend_finalization, 'ARTIFACT_RUNS_ROOT', tmp_path)
    report = tmp_path / 'already-complete' / 'reports' / 'trend_consensus.json'
    report.parent.mkdir(parents=True)
    report.write_text('{"status":"completed"}')
    before = report.read_bytes()
    with pytest.raises(ValueError, match='immutable'):
        czsc_trend_finalization.finalize_existing_run('already-complete')
    assert report.read_bytes() == before


def test_completion_recovery_rejects_escape_and_source_drift(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(czsc_trend_finalization, 'ARTIFACT_RUNS_ROOT', tmp_path)
    with pytest.raises(ValueError, match='invalid completion root'):
        czsc_trend_finalization.finalize_existing_run('../outside')
    run = tmp_path / 'incomplete-run'
    (run / 'config').mkdir(parents=True)
    assets = []
    for number in range(9):
        source = tmp_path / ('generator' + str(number) + '.py')
        source.write_text('pass\n')
        assets.append(source)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
    (run / 'config' / 'generating_code_hashes.json').write_text(json.dumps(hashes))
    assets[0].write_text('changed = True\n')
    with pytest.raises(ValueError, match='generating source changed'):
        czsc_trend_finalization.finalize_existing_run('incomplete-run')
    assert not (run / 'reports' / 'trend_consensus.json').exists()
