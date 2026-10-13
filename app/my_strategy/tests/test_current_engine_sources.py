"""Current build evidence replaces historical package identity admission."""
import hashlib
import json

import pytest

from my_strategy.adapters import czsc_adapter
from my_strategy.services import czsc_analysis
from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.services.czsc_trend_contracts import validate_training_config


def engine_source(tmp_path, monkeypatch):
    source = tmp_path / "vendor" / "czsc"
    source.mkdir(parents=True)
    code = source / "engine.txt"
    code.write_bytes(b"current-engine-source")
    manifest = {"package": "czsc", "version": "9.2", "archive_sha256": "retired-archive-is-not-a-gate",
                "files": [{"path": "engine.txt", "sha256": hashlib.sha256(code.read_bytes()).hexdigest()}]}
    (tmp_path / "czsc-source-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(czsc_adapter, "runtime_resource_root", lambda: tmp_path)
    return code, manifest


def test_engine_contract_uses_actual_source_without_fixed_historical_version(tmp_path, monkeypatch):
    _, _ = engine_source(tmp_path, monkeypatch)
    identity = czsc_adapter.engine_contract()
    assert identity["build_version"] == "9.2"
    assert len(identity["source_tree_sha256"]) == 64
    assert len(identity["manifest_sha256"]) == 64
    assert "native_signal_calls" in identity["capabilities"]


def test_engine_contract_rejects_changed_source_bytes(tmp_path, monkeypatch):
    code, _ = engine_source(tmp_path, monkeypatch)
    code.write_bytes(b"changed-engine-source")
    with pytest.raises(RuntimeError, match="源码校验失败"):
        czsc_adapter.engine_contract()


def test_engine_contract_rejects_manifest_paths_outside_current_build(tmp_path, monkeypatch):
    _, manifest = engine_source(tmp_path, monkeypatch)
    manifest["files"][0]["path"] = "../outside.txt"
    (tmp_path / "czsc-source-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="越过构建目录"):
        czsc_adapter.engine_contract()


def test_current_structure_config_needs_no_retired_attachment_fields(monkeypatch):
    current = strategy_config()["strategy"]
    assert "source_sha256" not in current and "czsc_version" not in current
    identity = {"build_version": "9.2", "source_tree_sha256": "current-source", "manifest_sha256": "current-manifest"}
    monkeypatch.setattr(czsc_analysis, "engine_contract", lambda: identity)
    result = czsc_analysis.strategy_config(current)
    assert result["czsc_version"] == "9.2" and result["source_sha256"] == "current-source"
    assert result["engine_contract"] == identity
    assert current["higher_period_closure"] == "next_period_observed"
    assert current["position"]["t0"] is False
    with pytest.raises(ValueError, match="实际观测后闭合"):
        czsc_analysis.strategy_config({**current, "higher_period_closure": "same_period"})
    with pytest.raises(ValueError, match="禁止 T\\+0"):
        czsc_analysis.strategy_config({**current, "position": {**current["position"], "t0": True}})


@pytest.mark.parametrize("overrides", [
    {"min_train_rows": 99}, {"min_calibration_rows": 29},
    {"trend": {"min_train_per_class": 99}},
    {"trend": {"min_calibration_per_class": 29}},
    {"trend": {"min_policy_per_class": 29}},
    {"high_return": {"min_net_return": .079}},
    {"high_return": {"min_r_multiple": 1.99}},
    {"high_return": {"max_risk_fraction": .16}},
    {"high_return": {"max_holding_sessions": 61}},
])
def test_public_current_training_cannot_relax_qualification_floors(overrides):
    with pytest.raises(ValueError):
        validate_training_config(overrides)


def test_saved_absolute_asset_uses_current_run_storage_even_if_old_file_exists(tmp_path):
    from my_strategy.services.czsc_trend_support import resolve_frozen_asset
    old = tmp_path / "old" / "runs" / "source-run" / "data_snapshots" / "raw.parquet"
    current = tmp_path / "current" / "runs" / "source-run" / "data_snapshots" / "raw.parquet"
    old.parent.mkdir(parents=True); current.parent.mkdir(parents=True)
    old.write_bytes(b"old-location"); current.write_bytes(b"current-frozen-bytes")
    assert resolve_frozen_asset(str(old), runs_root=current.parents[2]) == current.resolve()
    assert resolve_frozen_asset(str(old), runs_root=current.parents[2]).read_bytes() == b"current-frozen-bytes"
    current.unlink()
    with pytest.raises(FileNotFoundError, match="current storage"):
        resolve_frozen_asset(str(old), runs_root=current.parents[2])
    assert old.read_bytes() == b"old-location"


@pytest.mark.parametrize("record", ["../outside/file", "/old/runs/run/../outside/file", "C:\\old\\runs\\run\\..\\outside\\file"])
def test_frozen_asset_relocation_rejects_escaping_path(tmp_path, record):
    from my_strategy.services.czsc_trend_support import resolve_frozen_asset
    with pytest.raises(ValueError, match="escapes"):
        resolve_frozen_asset(record, runs_root=tmp_path)
