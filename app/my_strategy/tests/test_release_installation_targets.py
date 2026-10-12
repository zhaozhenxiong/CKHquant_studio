"""Public wheels retain immutable native identity and new target serialization."""
import hashlib
import json
from pathlib import Path
import runpy


def test_application_wheel_resource_copies_redact_host_paths_without_changing_source(monkeypatch, tmp_path):
    import setuptools
    from setuptools.command.build_py import build_py
    from setuptools.dist import Distribution

    app = Path(__file__).resolve().parents[2]
    originals = {path: path.read_bytes() for path in (app / "czsc-source-manifest.json", app / "vendor/wheels/wheel-index.json")}
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(build_py, "run", lambda self: None)
    runpy.run_path(str(app.parent / "setup.py"))
    command = captured["cmdclass"]["build_py"](Distribution())
    command.build_lib = str(tmp_path / "wheel-build")
    command.run()
    runtime = Path(command.build_lib) / "my_strategy/_runtime"
    source = json.loads(originals[app / "czsc-source-manifest.json"])
    public = json.loads((runtime / "czsc-source-manifest.json").read_text())
    assert public["files"] == source["files"]
    assert public["archive_sha256"] == source["archive_sha256"]
    assert public["built_wheel_sha256"] == source["built_wheel_sha256"]
    for entry in source["files"]:
        assert hashlib.sha256((runtime / "vendor/czsc" / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    index = json.loads(originals[app / "vendor/wheels/wheel-index.json"])
    public_index = json.loads((runtime / "vendor/wheels/wheel-index.json").read_text())
    assert public_index["source_tree_sha256"] == index["source_tree_sha256"]
    assert [(r["filename"], r["sha256"], r["tags"]) for r in public_index["wheels"]] == [
        (r["filename"], r["sha256"], r["tags"]) for r in index["wheels"]]
    for path, original in originals.items():
        assert path.read_bytes() == original
    assert "Users" not in (runtime / "czsc-source-manifest.json").read_text()
    assert "Users" not in (runtime / "vendor/wheels/wheel-index.json").read_text()


def test_installation_probe_reloads_new_direction_MA_exit_and_entry_targets(monkeypatch, tmp_path):
    from verify_installation import trend_model_probe
    monkeypatch.setenv("KHQUANT_ARTIFACT_ROOT", str(tmp_path / "isolated-artifacts"))
    result = trend_model_probe("cpu")
    assert result["fixture"] == "synthetic_installation_labels_only"
    assert result["market_effectiveness_certified"] is False and result["publication_allowed"] is False
    heads = result["heads"]
    assert heads["trend_direction"]["classes"] == ["UP", "DOWN", "NEUTRAL"]
    assert heads["trend_direction"]["input_columns"] == 20
    assert heads["ma_exit"]["input_columns"] == 30 and heads["ma_exit"]["stage"] == "E0"
    assert "exit_next_open_minus_frozen_continue" in heads["ma_exit"]["target"]
    assert heads["entry"]["stage"] == "E1" and "high_return_and_initial_risk" in heads["entry"]["target"]
    assert all(head["prediction_rows"] == 4 and head["save_reload_identical"] and head["actual_device"] == "cpu" for head in heads.values())
