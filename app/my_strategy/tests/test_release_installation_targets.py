"""Public wheels retain immutable native identity and new target serialization."""
import hashlib
import json
from pathlib import Path
import runpy
from urllib.error import HTTPError

import pytest


@pytest.fixture
def no_model_installation_probe(monkeypatch):
    import verify_installation

    status = {"model_family": "current", "strategy_version": "reference_position_v1",
              "current_selection": None, "active_model_run_id": None, "active_model": None,
              "models": [], "selection_events": [], "certification": {"publication_allowed": False}}
    position = {"schema": "reference_position_v1", "account_scope": "none", "reference_only": True,
        "target_position_fraction": 0., "previous_reference_target": 0., "change_fraction": 0.,
        "reference_delta": 0., "change_basis": "previous_reference_target_for_same_symbol",
        "risk_budget_fraction": .01, "stale": True, "predictive_gate_passed": False, "action": "WAIT"}
    result = {"strategy_version": "reference_position_v1", "research": {"model_run_id": None, "account_simulated": False},
              "run_id": "installation-analysis", "publication_allowed": False,
              "analysis_mode": "reference", "account_scope": "none", "account_actions": None,
              "source_reference_points": [], "reference_points": [],
              "chart_evidence": {"statistics": {"visible_window": {"broker_fills": 0}}, "rows": [{
                  "direction": "UNAVAILABLE", "action": "WAIT", "emitted": False,
                  "position_reference": position, "agents": {
                  "ma": {"status": "model_unavailable", "in_scope": False,
                         "p_up": None, "p_down": None, "p_neutral": None}}}]}}
    calls = []

    def request(base, path, method="GET", body=None):
        calls.append((base, path, method, body))
        if path == "/api/training/status":
            return status
        assert path == "/api/analysis" and method == "POST"
        return result

    monkeypatch.setattr(verify_installation, "request", request)
    return verify_installation.verify_no_model_reference, status, result, calls


def test_installation_verifier_accepts_only_current_untrained_default(no_model_installation_probe):
    verify, _, _, calls = no_model_installation_probe
    result = verify("http://isolated", "fixture-calendar", "2024-12-30")
    assert result["status"] == "passed" and result["strategy"] == "reference_position_v1"
    assert result["model_run_id"] is None and result["probabilities"] is None
    assert result["publication_allowed"] is False and result["analysis_rows"] == 1
    assert result["account_simulated"] is False and result["unqualified_allocation_blocked"] is True
    assert result["target_position_fraction"] == result["change_fraction"] == 0. and result["stale"]
    assert calls[1][3] == {"symbol": "000001.SZ", "start": "2024-01-01", "end": "2024-12-30",
                    "model_family": "current", "calendar_run_id": "fixture-calendar", "device": "cpu"}


@pytest.mark.parametrize("key,value", [
    ("strategy_version", "trend_consensus"), ("model_family", "legacy"),
    ("current_selection", {"run_id": "workstation-model"}),
    ("models", [{"run_id": "workstation-model"}]), ("active_model_run_id", "workstation-model"),
    ("active_model", {"run_id": "workstation-model"}), ("selection_events", [{"run_id": "workstation-model"}]),
    ("certification", {"publication_allowed": True}),
])
def test_installation_verifier_rejects_wrong_default_or_workstation_model(no_model_installation_probe, key, value):
    verify, status, _, calls = no_model_installation_probe
    status[key] = value
    with pytest.raises(RuntimeError, match="Public defaults must select reference_position_v1 without a workstation model"):
        verify("http://isolated", "fixture-calendar", "2024-12-30")
    assert len(calls) == 1


@pytest.mark.parametrize("case", ["legacy_analysis", "bound_model", "empty_rows", "publication"])
def test_installation_verifier_rejects_invalid_untrained_analysis(no_model_installation_probe, case):
    verify, _, result, _ = no_model_installation_probe
    if case == "legacy_analysis":
        result["strategy_version"] = "trend_consensus"
    elif case == "bound_model":
        result["research"]["model_run_id"] = "workstation-model"
    elif case == "publication":
        result["publication_allowed"] = True
    else:
        result["chart_evidence"]["rows"] = []
    with pytest.raises(RuntimeError, match="Untrained installation failed its actual default reference analysis"):
        verify("http://isolated", "fixture-calendar", "2024-12-30")


@pytest.mark.parametrize("key,value", [("status", "scored"), ("in_scope", True),
    ("p_up", 0.0), ("p_down", 0.5), ("p_neutral", 1.0)])
def test_installation_verifier_rejects_fabricated_probabilities(no_model_installation_probe, key, value):
    verify, _, result, _ = no_model_installation_probe
    result["chart_evidence"]["rows"][0]["agents"]["ma"][key] = value
    with pytest.raises(RuntimeError, match="Untrained installation fabricated reference direction probabilities"):
        verify("http://isolated", "fixture-calendar", "2024-12-30")


@pytest.mark.parametrize("case", ["quality_probability", "high_return_probability", "account_simulation", "broker_fill",
    "missing_source_points", "wrong_reference_clock", "unqualified_target", "missing_stale", "wrong_delta",
    "wrong_previous", "manual_change_basis", "higher_risk_budget", "actual_exit_probability", "fake_final_point"])
def test_installation_verifier_preserves_account_free_unscored_reference(no_model_installation_probe, case):
    verify, _, result, _ = no_model_installation_probe
    row = result["chart_evidence"]["rows"][0]
    if case == "quality_probability":
        row["agents"]["reference_quality"] = {"probability": .9}
    elif case == "high_return_probability":
        row["high_return_probability"] = .5
    elif case == "account_simulation":
        result["research"]["account_simulated"] = True
    elif case == "broker_fill":
        result["chart_evidence"]["statistics"]["visible_window"]["broker_fills"] = 1
    elif case == "missing_source_points":
        result.pop("source_reference_points")
    elif case == "wrong_reference_clock":
        result["source_reference_points"] = [{"side": "SELL", "date": "2024-01-05",
            "reference_price_source": "signal_day_close", "reference_price_date": "2024-01-08"}]
    elif case == "unqualified_target":
        row["position_reference"]["target_position_fraction"] = .2
    elif case == "missing_stale":
        row["position_reference"]["stale"] = False
    elif case == "wrong_delta":
        row["position_reference"]["change_fraction"] = -.2
    elif case == "wrong_previous":
        row["position_reference"]["previous_reference_target"] = .2
    elif case == "manual_change_basis":
        row["position_reference"]["change_basis"] = "manual_inventory"
    elif case == "higher_risk_budget":
        row["position_reference"]["risk_budget_fraction"] = .02
    elif case == "fake_final_point":
        result["reference_points"] = [{"side": "SELL", "change_fraction": 0.}]
    else:
        row["exit_probability"] = .6
    with pytest.raises(RuntimeError):
        verify("http://isolated", "fixture-calendar", "2024-12-30")


def test_installation_verifier_preserves_independent_source_sell_without_forced_target_sale(no_model_installation_probe):
    verify, _, result, _ = no_model_installation_probe
    result["source_reference_points"] = [{"side": "SELL", "date": "2024-01-05",
        "reference_price_source": "signal_day_close", "reference_price_date": "2024-01-05"}]
    actual = verify("http://isolated", "fixture-calendar", "2024-12-30")
    assert actual["source_reference_sides"] == ["SELL"] and actual["final_reference_points"] == 0
    assert actual["target_position_fraction"] == actual["change_fraction"] == 0.


@pytest.fixture
def reference_training_capabilities(monkeypatch, tmp_path):
    import verify_installation
    from my_strategy.services import reference_training as training
    payloads = {"/api/training/capabilities": training.capabilities(),
                "/api/training/profiles": training.profiles(metadata_root=tmp_path / "metadata"),
                "/api/training/status": training.status(metadata_root=tmp_path / "metadata", runs_root=tmp_path / "runs")}
    monkeypatch.setattr(verify_installation, "request", lambda base, path: payloads[path])
    return verify_installation.verify_training_center, payloads


def test_installation_training_center_has_reference_device_and_defaults(reference_training_capabilities):
    verify, _ = reference_training_capabilities
    result = verify("http://isolated")
    assert result["strategy_version"] == "reference_position_v1" and result["field_count"] == 13
    assert result["defaults"]["device"] == "auto" and result["defaults"]["batch_size"] == 8192
    assert result["training_job_submitted"] is False and result["publication_allowed"] is False


@pytest.mark.parametrize("case", ["legacy_family", "legacy_defaults", "cpu_default", "weaker_risk", "old_trainable_head", "publication"])
def test_installation_training_center_rejects_stale_or_weakened_contract(reference_training_capabilities, case):
    verify, payloads = reference_training_capabilities
    capabilities = payloads["/api/training/capabilities"]
    if case == "legacy_family":
        capabilities["strategy_version"] = "trend_consensus"
    elif case == "legacy_defaults":
        capabilities["defaults"]["min_net_return"] = .08
    elif case == "cpu_default":
        capabilities["defaults"]["device"] = "cpu"
    elif case == "weaker_risk":
        payloads["/api/training/profiles"]["default_profile"]["parameters"]["risk_budget"] = .05
    elif case == "old_trainable_head":
        capabilities["stages"].append({"key": "entry", "trainable": True})
    else:
        capabilities["qualification"]["publication_allowed"] = True
    with pytest.raises(RuntimeError):
        verify("http://isolated")


@pytest.mark.parametrize("code", [200, 400, 405])
def test_installation_verifier_rejects_reintroduced_route_or_method_alias(monkeypatch, code):
    import verify_installation
    def request(*args):
        if code != 200:
            raise HTTPError("http://isolated/api/holdings", code, "unexpected", {}, None)
        return {}
    monkeypatch.setattr(verify_installation, "request", request)
    with pytest.raises(RuntimeError, match="Retired route"):
        verify_installation.expect_removed_route("http://isolated", "/api/holdings")


def test_installation_verifier_accepts_exact_retired_404(monkeypatch):
    import verify_installation
    def request(*args):
        raise HTTPError("http://isolated/api/holdings", 404, "removed", {}, None)
    monkeypatch.setattr(verify_installation, "request", request)
    assert verify_installation.expect_removed_route("http://isolated", "/api/holdings") is None


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


def test_installation_probe_fits_calibrates_reloads_current_reference_model(monkeypatch, tmp_path):
    from verify_installation import model_probe
    monkeypatch.setenv("KHQUANT_ARTIFACT_ROOT", str(tmp_path / "isolated-artifacts"))
    result = model_probe("cpu")
    assert result["training_device"] == result["inference"]["device"] == "cpu"
    assert "trend_consensus" not in result
    model = result["reference_position_v1"]
    assert model["fixture"] == "synthetic_installation_labels_only"
    assert model["market_effectiveness_certified"] is False and model["publication_allowed"] is False
    assert model["active_model_registered"] is False and model["account_simulated"] is False
    assert model["family"] == "reference_position_v1" and model["classes"] == ["UP", "DOWN", "NEUTRAL"]
    assert model["input_columns"] == 58 and model["architecture"] in {"linear", "mlp32x16"}
    assert model["training_device"] == model["calibration_device"] == model["inference_device"] == "cpu"
    assert model["numeric_dtype"] == "float32" and model["price_label_dtype"] == "float64"
    assert model["prediction_rows"] == 4 and model["unscored_rows"] == 2
    assert model["save_reload_identical"] and model["missing_input_stays_unscored"]
    assert model["calibration_method"] == "chronological_positive_temperature_no_intercepts"
    assert model["selection_source"] == "purged_train_internal_only" and model["temperature"] > 0
    assert len(model["weights_sha256"]) == 64
