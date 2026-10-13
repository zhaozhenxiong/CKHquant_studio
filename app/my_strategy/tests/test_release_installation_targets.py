"""Public wheels retain immutable native identity and new target serialization."""
import hashlib
import json
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def no_model_installation_probe(monkeypatch):
    import verify_installation

    status = {"default_consensus_version": "czsc_trend_consensus_v2", "current_model_run_id": None,
              "models": [], "model_selection": "automatic_qualified_runs"}
    result = {"strategy_version": "czsc_trend_consensus_v2", "research": {"model_run_id": None},
              "run_id": "installation-analysis", "publication_allowed": False,
              "chart_evidence": {"rows": [{"agents": {
                  "ma": {"model_available": False, "p_up": None, "p_down": None, "p_neutral": None},
                  "high_return": {"probability": None}}}]}}
    calls = []

    def request(base, path, method="GET", body=None):
        calls.append((base, path, method, body))
        if path == "/api/research/high-return-status":
            return status
        assert path == "/api/analysis" and method == "POST"
        return result

    monkeypatch.setattr(verify_installation, "request", request)
    return verify_installation.verify_no_model_trend, status, result, calls


def test_installation_verifier_accepts_only_public_v2_untrained_default(no_model_installation_probe):
    verify, _, _, calls = no_model_installation_probe
    result = verify("http://isolated", "fixture-calendar", "2024-12-30")
    assert result["status"] == "passed" and result["strategy"] == "czsc_trend_consensus_v2"
    assert result["model_run_id"] is None and result["probabilities"] is None
    assert result["publication_allowed"] is False and result["analysis_rows"] == 1
    body = calls[1][3]
    assert body == {"symbol": "000001.SZ", "start": "2024-01-01", "end": "2024-12-30",
                    "research": True, "model_family": "high_return", "calendar_run_id": "fixture-calendar",
                    "device": "cpu"}
    assert "consensus_version" not in body  # Exercise the actual public default.


@pytest.mark.parametrize("key,value", [
    ("default_consensus_version", "czsc_trend_consensus"),
    ("current_model_run_id", "workstation-model"),
    ("models", [{"run_id": "workstation-model"}]),
    ("model_selection", "configured_run_auto_prior_heads"),
])
def test_installation_verifier_rejects_wrong_default_or_workstation_model(no_model_installation_probe, key, value):
    verify, status, _, calls = no_model_installation_probe
    status[key] = value
    with pytest.raises(RuntimeError, match="Public defaults must select trend consensus without a workstation model"):
        verify("http://isolated", "fixture-calendar", "2024-12-30")
    assert len(calls) == 1


@pytest.mark.parametrize("case", ["legacy_analysis", "bound_model", "empty_rows"])
def test_installation_verifier_rejects_invalid_untrained_analysis(no_model_installation_probe, case):
    verify, _, result, _ = no_model_installation_probe
    if case == "legacy_analysis":
        result["strategy_version"] = "czsc_trend_consensus"
    elif case == "bound_model":
        result["research"]["model_run_id"] = "workstation-model"
    else:
        result["chart_evidence"]["rows"] = []
    with pytest.raises(RuntimeError, match="Untrained installation failed its actual default trend analysis"):
        verify("http://isolated", "fixture-calendar", "2024-12-30")


@pytest.mark.parametrize("key,value", [("model_available", True), ("p_up", 0.0), ("p_down", 0.5), ("p_neutral", 1.0)])
def test_installation_verifier_rejects_fabricated_ma_probabilities(no_model_installation_probe, key, value):
    verify, _, result, _ = no_model_installation_probe
    result["chart_evidence"]["rows"][0]["agents"]["ma"][key] = value
    with pytest.raises(RuntimeError, match="Untrained installation fabricated MA probabilities"):
        verify("http://isolated", "fixture-calendar", "2024-12-30")


@pytest.mark.parametrize("probability", [0.0, 0.5])
def test_installation_verifier_rejects_fabricated_high_return_probability(no_model_installation_probe, probability):
    verify, _, result, _ = no_model_installation_probe
    result["chart_evidence"]["rows"][0]["agents"]["high_return"]["probability"] = probability
    with pytest.raises(RuntimeError, match="Untrained installation fabricated high-return probability"):
        verify("http://isolated", "fixture-calendar", "2024-12-30")


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
