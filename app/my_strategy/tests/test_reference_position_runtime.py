"""Reference paths are causal recommendations, not inferred actual inventory."""
import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from my_strategy.services.reference_runtime import ReferencePositionPath, risk_geometry, validate_position_policy


def advance(path, day="2026-10-12", **kwargs):
    return path.advance(day=day, p_up=kwargs.pop("p_up", .7), p_down=kwargs.pop("p_down", .1),
        risk_cap=kwargs.pop("risk_cap", .2), stop_price=kwargs.pop("stop_price", 9.), **kwargs)


def test_delta_is_previous_reference_not_manual_position_and_sells_can_be_partial():
    path = ReferencePositionPath()
    first = advance(path, current_position_fraction=.9)
    assert first["initialization"] == "zero_reference_target"
    assert first["action"] == "BUY"
    assert first["target_position_fraction"] == .2
    assert first["change_fraction"] == .2
    assert first["manual_comparison_change_fraction"] == pytest.approx(-.7)
    partial = advance(path, p_up=.45, p_down=.25, current_position_fraction=.0)
    assert partial["action"] == "SELL"
    assert partial["target_position_fraction"] == .1
    assert partial["change_fraction"] == -.1
    assert partial["manual_comparison_change_fraction"] == .1
    assert "reference_probability_target_increased" in first["reason_codes"]
    assert "reference_probability_target_decreased" in partial["reason_codes"]
    assert partial["risk_cap_level_fraction"] == .2
    assert partial["position_step_fraction"] == .025
    assert partial["position_change_threshold_fraction"] == pytest.approx(.0225)


def test_wait_and_unavailable_do_not_clear_exposure_or_emit_sell():
    path = ReferencePositionPath()
    advance(path)
    wait = advance(path, p_up=.35, p_down=.34)
    assert wait["action"] == "WAIT" and wait["target_position_fraction"] == .2
    assert wait["stale"] is False
    missing = advance(path, day="2026-10-13", p_up=None, scored=False)
    assert missing["target_position_fraction"] == .2
    assert missing["change_fraction"] == 0 and missing["action"] == "WAIT"
    assert missing["stale"] and missing["last_scored_date"] == "2026-10-12"
    assert missing["continuous_target_fraction"] is None


def test_unqualified_predictions_never_generate_positive_targets():
    path = ReferencePositionPath()
    out = advance(path, predictive_passed=False)
    assert out["target_position_fraction"] == 0 and out["action"] == "WAIT"
    assert out["stale"]
    assert "reference_predictive_gate_pending" in out["reason_codes"]
    advance(path)
    out = advance(path, predictive_passed=False, p_up=.0, p_down=.95)
    assert out["target_position_fraction"] == .2 and out["action"] == "WAIT"


def test_risk_cap_is_a_ceiling_not_an_active_target_and_hard_stop_clears():
    path = ReferencePositionPath()
    out = advance(path, p_up=.37, p_down=.25)
    assert 0 < out["target_position_fraction"] < out["risk_position_cap_fraction"]
    stopped = advance(path, hard_invalidation=True)
    assert stopped["action"] == "SELL" and stopped["target_position_fraction"] == 0
    assert "reference_frozen_stop_invalidated" in stopped["reason_codes"]
    assert path.stop_price is None


def test_small_prediction_jitter_cannot_emit_daily_repeat_points():
    path = ReferencePositionPath()
    out = advance(path, p_up=.45, p_down=.25)
    assert out["target_position_fraction"] == .1
    for edge in (.195, .205, .19, .21, .20):
        row = advance(path, p_up=.25 + edge, p_down=.25)
        assert row["action"] == "WAIT"
        assert row["change_fraction"] == 0
    # A tighter cap moves to a whole level below the cap, never a tiny S.
    reduced = advance(path, risk_cap=.09)
    assert reduced["action"] == "SELL" and reduced["target_position_fraction"] == .075


def test_feature_tail_mutations_do_not_change_historical_risk_caps():
    frame = pd.DataFrame({"close": 10 + np.arange(60) / 100,
        "high": 10.2 + np.arange(60) / 100, "low": 9.8 + np.arange(60) / 100})
    original = risk_geometry(frame)
    altered = frame.copy()
    altered.loc[40:, ["close", "high", "low"]] *= 100
    pd.testing.assert_frame_equal(original.iloc[:40], risk_geometry(altered).iloc[:40])
    assert original.eligible.iloc[20:].all()
    assert (original.cap.dropna() <= .2).all()


@pytest.mark.parametrize("config", [{"hysteresis": .02}, {"risk_budget": -1},
    {"max_stock_weight": 2}, {"edge_entry": .4}, {"position_step": 0}, {"unexpected": True}])
def test_position_policy_rejects_ambiguous_or_unsafe_contract(config):
    with pytest.raises(ValueError):
        validate_position_policy(config)


def test_known_stop_breach_is_separate_from_missing_model_evidence():
    path = ReferencePositionPath()
    advance(path)
    missing = advance(path, p_up=None, scored=False)
    assert missing["action"] == "WAIT" and missing["target_position_fraction"] == .2
    breached = advance(path, p_up=None, scored=False, hard_invalidation=True)
    assert breached["action"] == "SELL" and breached["target_position_fraction"] == 0
    assert breached["previous_frozen_stop_price"] == 9.0
    assert breached["invalidation_price"] == 9.0
    assert path.stop_price is None


@pytest.fixture
def overlay_inputs(monkeypatch, tmp_path):
    from my_strategy.services import reference_runtime, reference_model, reference_training, czsc_native_evidence
    dates = pd.bdate_range("2025-01-01", periods=80).strftime("%Y-%m-%d").tolist()
    frame = pd.DataFrame({"symbol": "000001.SZ", "date": dates, "close": 10., "open": 10.,
        "high": 10.2, "low": 9.8, "source": "tushare", "has_trade_price": 1,
        "volume": 1e7, "amount": 1e8, "model_input_eligible": True,
        "liquidity_pool_eligible": True, "available_at": [day + "T15:00:00+08:00" for day in dates],
        "liquidity_observed_at": [day + "T15:00:00+08:00" for day in dates]})
    for column in reference_model.MA_TREND_COLUMNS:
        frame[column] = .5
    manifest = tmp_path / "manifest.json"
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"isolated test model weights")
    model_manifest = {"schema_hash": reference_model.SCHEMA_HASH,
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "weights_file": weights.name}
    manifest.write_text(json.dumps(model_manifest))
    report = {"config": {"position": {}}, "folds": [{"name": "fold", "split": {
        "test_start": dates[64], "test_end": dates[-1]}, "model_path": str(manifest),
        "research_reconstruction": True, "predictive_gate": {"passed": True}}]}
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "reference_training.json").write_text(json.dumps(report))
    monkeypatch.setattr(reference_training, "active_reference_run_id", lambda: "reference-run")
    monkeypatch.setattr(reference_training, "load_reference_run", lambda run_id=None: (tmp_path, report))
    monkeypatch.setattr(czsc_native_evidence, "build_native_evidence", lambda data, **kw:
        {"frame": data.copy(), "evaluations": [], "events": [], "wyckoff_events": []})
    class Model:
        manifest = model_manifest
        def predict(self, values):
            assert values.input_eligible.all()
            return np.tile([.7, .1, .2], (len(values), 1))
    monkeypatch.setattr(reference_runtime, "_model_session", lambda *args: Model())
    return frame, dates, report


def test_overlay_batches_model_and_does_not_publish_daily_duplicate_markers(overlay_inputs):
    from my_strategy.services.reference_runtime import reference_overlay
    frame, dates, report = overlay_inputs
    out = reference_overlay(frame, dates, device="cpu")
    assert len(out["reference_points"]) == 1
    assert out["reference_points"][0]["side"] == "BUY"
    assert out["reference_points"][0]["date"] == dates[64]
    assert out["position_reference"]["target_position_fraction"] == .2
    assert out["position_reference"]["change_fraction"] == 0
    assert out["compute_info"]["model_inference_rows"] == 16
    assert out["compute_info"]["model_inference_batches"] == 1
    assert out["account_actions"] is None
    assert all(row["historical_reconstruction"] for row in out["rows"][64:])


def test_overlay_cutoff_and_missing_session_preserve_prior_target(overlay_inputs):
    from my_strategy.services.reference_runtime import reference_overlay
    frame, dates, report = overlay_inputs
    missing = reference_overlay(frame.iloc[:-1], dates, as_of=dates[-1], device="cpu")
    assert missing["latest_decision"]["direction"] == "UNAVAILABLE"
    assert missing["latest_decision"]["reference_price"] is None
    assert missing["position_reference"]["target_position_fraction"] == .2
    assert missing["position_reference"]["change_fraction"] == 0
    assert missing["position_reference"]["stale"]
    assert missing["latest_decision"]["agents"]["ma"]["status"] == "input_ineligible"
    assert missing["latest_decision"]["agents"]["ma"]["p_up"] is None
    assert missing["latest_decision"]["agents"]["ma"]["model_identity"] == {}
    assert missing["latest_decision"]["buy_base_score"] is None
    assert missing["latest_decision"]["historical_reconstruction"] is False
    assert missing["reference_risk_plan"]["date"] == dates[-1]
    assert missing["reference_risk_plan"]["diagnostic_stop_price"] is None
    assert missing["model_gate"] == {"passed": False}
    before = reference_overlay(frame, dates, as_of=dates[70], device="cpu")
    frame.loc[71:, ["open", "high", "low", "close"]] *= 100
    after = reference_overlay(frame, dates, as_of=dates[70], device="cpu")
    assert before["rows"] == after["rows"]
    assert before["reference_points"] == after["reference_points"]


def test_overlay_no_active_model_and_failed_gate_are_unscored(overlay_inputs, monkeypatch):
    from my_strategy.services import reference_runtime, reference_training
    frame, dates, report = overlay_inputs
    report["folds"][0]["predictive_gate"] = {"passed": False}
    pending = reference_runtime.reference_overlay(frame, dates, device="cpu")
    assert pending["position_reference"]["target_position_fraction"] == 0
    assert pending["reference_points"] == []
    monkeypatch.setattr(reference_training, "active_reference_run_id", lambda: None)
    missing = reference_runtime.reference_overlay(frame, dates, device="cpu")
    assert missing["reference_points"] == [] and missing["model_run_id"] is None
    assert missing["compute_info"]["model_inference_rows"] == 0


def test_forward_uses_actual_effective_date_and_never_backfills_production(overlay_inputs):
    from my_strategy.services.reference_runtime import reference_overlay
    frame, dates, report = overlay_inputs
    report["production"] = {**report["folds"][0], "name": "production", "effective_from": dates[75],
        "research_reconstruction": False, "split": {"policy_end": dates[74]}}
    before = reference_overlay(frame, dates, as_of=dates[74], model_mode="production", device="cpu")
    assert before["reference_points"] == []
    assert before["compute_info"]["model_inference_rows"] == 0
    after = reference_overlay(frame, dates, model_mode="production", device="cpu")
    assert after["reference_points"][0]["date"] == dates[75]
    assert after["compute_info"]["model_inference_rows"] == 5
    assert not any(row["historical_reconstruction"] for row in after["rows"])


def test_risk_cap_drift_never_creates_fractional_level_markers():
    path = ReferencePositionPath()
    first = advance(path, risk_cap=.201)
    assert first['target_position_fraction'] == .2
    drift = advance(path, risk_cap=.199)
    assert drift['target_position_fraction'] == .175
    assert drift['change_fraction'] == -.025
    for cap in (.198, .190, .183, .176):
        row = advance(path, risk_cap=cap)
        assert row['action'] == 'WAIT' and row['change_fraction'] == 0
        assert row['target_position_fraction'] <= cap
    tiny = advance(ReferencePositionPath(), risk_cap=.0249)
    assert tiny['target_position_fraction'] == 0 and tiny['action'] == 'WAIT'


@pytest.mark.parametrize('updates', [{}, {'symbol': 'other'}, {'source': 'unknown'},
    {'original_event_id': None}, {'pattern_id': None}, {'confirmed': False},
    {'input_eligible': False}, {'evidence_visible': False}, {'candidate_direction_valid': False},
    {'first_confirmed_at': '2026-10-13T15:00:00+08:00'},
    {'observed_at': '2026-10-12T15:01:00+08:00'}, {'available_at': None}])
def test_source_points_require_causal_clocks_and_original_identity(updates):
    from my_strategy.services.reference_runtime import _source_reference_events
    event = {'symbol': '000001.SZ', 'source': 'structure', 'side': 'BUY',
        'event_id': 'e', 'original_event_id': 'e', 'pattern_id': 'pattern', 'confirmed': True,
        'input_eligible': True, 'candidate_direction_valid': True, 'date': '2026-10-12',
        'available_at': '2026-10-12T15:00:00+08:00', 'first_available_at': '2026-10-12T15:00:00+08:00',
        **updates}
    results = _source_reference_events({'events': [event]}, '000001.SZ', '2026-10-12')
    if updates:
        assert results == []
    else:
        assert results == [event] and results[0] is not event


def test_risk_failure_and_pending_gate_do_not_hide_scored_probabilities(overlay_inputs):
    from my_strategy.services.reference_runtime import reference_overlay
    frame, dates, report = overlay_inputs
    report["folds"][0]["predictive_gate"] = {"passed": False}
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [9.7, 9.9, 9.5, 9.7]
    out = reference_overlay(frame, dates, device="cpu")
    latest = out["latest_decision"]
    ma, risk, position = latest["agents"]["ma"], out["reference_risk_plan"], out["position_reference"]
    assert ma["status"] == "scored" and ma["probability_scored"]
    assert [ma["p_up"], ma["p_down"], ma["p_neutral"]] == pytest.approx([.7, .1, .2])
    assert ma["reason_codes"] == []
    assert not ma["risk_geometry_available"] and not ma["predictive_gate_passed"]
    assert risk["status"] == "unavailable"
    assert risk["diagnostic_stop_price"] == risk["prior_low20"] == 9.8
    assert risk["diagnostic_stop_distance_fraction"] == pytest.approx((9.7 - 9.8) / 9.7)
    assert risk["stop_price"] is risk["stop_distance_fraction"] is risk["risk_position_cap_fraction"] is None
    assert "reference_risk_boundary_breached" in risk["reason_codes"]
    assert position["action"] == "WAIT" and position["target_position_fraction"] == 0
    assert position["probability_scored"] and not position["position_usable"]
    assert "reference_predictive_gate_pending" in position["reason_codes"]
    assert "no_effective_reference_model" not in position["reason_codes"]


def test_invalid_new_risk_keeps_old_target_without_freezing_invalid_boundary(overlay_inputs):
    from my_strategy.services.reference_runtime import reference_overlay
    frame, dates, report = overlay_inputs
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [9.85, 10., 9.8, 9.85]
    out = reference_overlay(frame, dates, device="cpu")
    position = out["position_reference"]
    assert out["latest_decision"]["agents"]["ma"]["status"] == "scored"
    assert out["reference_risk_plan"]["reason_codes"] == [
        "reference_risk_geometry_unavailable", "reference_risk_distance_below_minimum"]
    assert position["action"] == "WAIT" and position["target_position_fraction"] == .2
    assert position["invalidation_price"] == 9.8 and position["stop_price"] is None
    assert position["stale"]


def test_cap_reliability_requires_the_verified_prior_price_window(overlay_inputs):
    from my_strategy.services.reference_runtime import reference_overlay
    frame, dates, _ = overlay_inputs
    clean = reference_overlay(frame, dates, device="cpu")
    assert clean["reference_risk_plan"]["risk_cap_evidence_reliable"]
    frame.loc[frame.index[-10], "source"] = "unverified-source"
    damaged = reference_overlay(frame, dates, device="cpu")
    # The v1 finite numerical geometry stays unchanged. It cannot qualify an
    # independent cap-only policy when prior observations are unverified.
    assert damaged["reference_risk_plan"]["status"] == "available"
    assert not damaged["reference_risk_plan"]["risk_cap_evidence_reliable"]
    assert damaged["position_reference"]["target_position_fraction"] == .2
    assert damaged["position_reference"]["action"] == "WAIT"


@pytest.mark.parametrize("probabilities", [[np.nan, .1, .9], [1.1, -.1, 0.], [.6, .2, .3], [0., 0., 0.]])
def test_invalid_three_class_output_is_unscored_and_never_fills_zero(overlay_inputs, monkeypatch, probabilities):
    from my_strategy.services import reference_runtime
    frame, dates, _ = overlay_inputs
    class InvalidModel:
        def predict(self, values):
            return np.tile(probabilities, (len(values), 1))
    monkeypatch.setattr(reference_runtime, "_model_session", lambda *args: InvalidModel())
    out = reference_runtime.reference_overlay(frame, dates, device="cpu")
    ma = out["latest_decision"]["agents"]["ma"]
    assert ma["status"] == "probability_invalid" and not ma["probability_scored"]
    assert ma["p_up"] is ma["p_down"] is ma["p_neutral"] is None
    assert ma["reason_codes"] == ["reference_probability_output_invalid"]
    assert out["reference_points"] == [] and out["position_reference"]["target_position_fraction"] == 0
    assert out["compute_info"]["model_inference_rows"] == 16


def test_disabled_missing_fold_and_ineligible_inputs_have_separate_states(overlay_inputs, monkeypatch):
    from my_strategy.services import reference_runtime, reference_training
    frame, dates, _ = overlay_inputs
    disabled = reference_runtime.reference_overlay(frame, dates, use_models=False, model_fold="ignored-fold", device="cpu")
    assert disabled["latest_decision"]["agents"]["ma"]["status"] == "models_disabled"
    assert disabled["latest_decision"]["model_policy"] == "disabled"
    assert disabled["latest_decision"]["model_fold"] is None
    mismatched = reference_runtime.reference_overlay(frame, dates, model_fold="other-fold", device="cpu")
    assert mismatched["latest_decision"]["agents"]["ma"]["status"] == "fixed_fold_unavailable"
    frame.loc[frame.index[-1], "model_input_eligible"] = False
    ineligible = reference_runtime.reference_overlay(frame, dates, device="cpu")
    assert ineligible["latest_decision"]["agents"]["ma"]["status"] == "input_ineligible"
    assert ineligible["position_reference"]["target_position_fraction"] == .2
    monkeypatch.setattr(reference_training, "active_reference_run_id", lambda: None)
    missing = reference_runtime.reference_overlay(frame, dates, device="cpu")
    assert missing["latest_decision"]["agents"]["ma"]["status"] == "model_unavailable"


def test_model_asset_identity_failure_is_not_demoted_to_normal_missing_score(overlay_inputs, monkeypatch):
    from my_strategy.services import reference_runtime, reference_training
    frame, dates, _ = overlay_inputs
    def invalid_asset(*args):
        raise ValueError("model asset integrity mismatch")
    monkeypatch.setattr(reference_training, "load_reference_run", invalid_asset)
    with pytest.raises(ValueError, match="integrity"):
        reference_runtime.reference_overlay(frame, dates, device="cpu")


def test_risk_diagnostics_keep_missing_history_and_extreme_distances_separate():
    frame = pd.DataFrame({"close": np.full(30, 100.), "high": np.full(30, 102.), "low": np.full(30, 98.)})
    values = risk_geometry(frame)
    assert "reference_risk_atr_unavailable" in values.iloc[0].reason_codes
    assert "reference_risk_prior_low_unavailable" in values.iloc[0].reason_codes
    frame.loc[29, ["close", "high", "low"]] = [200., 202., 198.]
    wide = risk_geometry(frame, {"atr_stop_multiple": 40}).iloc[-1]
    assert not wide.eligible and wide.diagnostic_stop_distance_fraction > .25
    assert wide.reason_codes == ["reference_risk_distance_above_maximum"]


def test_v1_retains_stale_target_above_known_cap_as_diagnostic_not_new_policy():
    path = ReferencePositionPath()
    advance(path, risk_cap=.075)
    row = advance(path, day="2026-10-13", p_up=None, p_down=None, scored=False, risk_cap=.05)
    assert row["action"] == "WAIT" and row["target_position_fraction"] == .075
    assert row["retained_target_exceeds_risk_cap"]
    assert "reference_stale_target_exceeds_risk_cap" in row["reason_codes"]
    # An independent cap-only candidate would reduce here, whereas unavailable
    # risk cannot support that candidate. It is evaluated, never activated in v1.
    independent_candidate = min(row["previous_reference_target"], row["risk_cap_level_fraction"])
    assert independent_candidate == .05 and independent_candidate < row["target_position_fraction"]
    missing_risk = advance(path, day="2026-10-14", p_up=None, scored=False, risk_cap=None)
    assert missing_risk["risk_cap_level_fraction"] is None
    assert not missing_risk["retained_target_exceeds_risk_cap"]
    assert missing_risk["action"] == "WAIT" and missing_risk["target_position_fraction"] == .075
    stopped = advance(path, day="2026-10-15", p_up=None, scored=False, risk_cap=None, hard_invalidation=True)
    assert stopped["action"] == "SELL" and stopped["target_position_fraction"] == 0


def test_source_sell_at_zero_is_preserved_with_no_duplicate_target_sell(overlay_inputs, monkeypatch):
    from my_strategy.services import reference_runtime, czsc_native_evidence
    frame, dates, report = overlay_inputs
    report["folds"][0]["predictive_gate"] = {"passed": False}
    event = {"symbol": "000001.SZ", "source": "structure", "side": "SELL", "event_id": "source-sell",
        "original_event_id": "source-sell", "pattern_id": "sell-pattern", "confirmed": True,
        "input_eligible": True, "candidate_direction_valid": True, "date": dates[-1],
        "available_at": dates[-1] + "T15:00:00+08:00"}
    monkeypatch.setattr(czsc_native_evidence, "build_native_evidence", lambda data, **kw:
        {"frame": data.copy(), "evaluations": [], "events": [event], "wyckoff_events": []})
    out = reference_runtime.reference_overlay(frame, dates, device="cpu")
    assert out["source_reference_points"][0]["side"] == "SELL" and out["reference_points"] == []
    assert "reference_source_sell_without_target_to_reduce" in out["position_reference"]["reason_codes"]


def test_loaded_model_identity_binds_actual_report_manifest_and_verified_weights(overlay_inputs):
    from pathlib import Path
    from my_strategy.services import reference_runtime, reference_model
    frame, dates, report = overlay_inputs
    out = reference_runtime.reference_overlay(frame, dates, device="cpu", source_context_hash="source-revision")
    manifest = Path(report["folds"][0]["model_path"])
    expected = {"run_id": manifest.parent.name,
        "report_sha256": hashlib.sha256((manifest.parent / "reports" / "reference_training.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256((manifest.parent / "weights.pt").read_bytes()).hexdigest(),
        "feature_schema_hash": reference_model.SCHEMA_HASH}
    assert out["latest_decision"]["agents"]["ma"]["model_asset_identity"] == expected
    assert out["latest_decision"]["agents"]["ma"]["model_identity"]["model_asset_identity"] == expected
    assert out["latest_decision"]["model_resolution"]["model_asset_identity"] == expected
    assert out["model_asset_identity"] == expected and out["model_asset_identities"] == [expected]
    assert out["runtime_code_hash"] == out["runtime_identity"]["code_sha256"]
    assert "services/reference_runtime.py" in out["runtime_identity"]["files"]
    assert len(out["reference_context_hash"]) == 64
    assert out["rows"][0]["agents"]["ma"]["model_asset_identity"] == {}
    missing = reference_runtime.reference_overlay(frame.iloc[:-1], dates, as_of=dates[-1], device="cpu")
    assert missing["latest_decision"]["agents"]["ma"]["model_asset_identity"] == {}
    assert missing["model_asset_identity"] == {}


def test_test_double_without_manifest_never_claims_verified_weight_sha(overlay_inputs, monkeypatch):
    from my_strategy.services import reference_runtime
    frame, dates, _ = overlay_inputs
    class MinimalModel:
        def predict(self, values):
            return np.tile([.7, .1, .2], (len(values), 1))
    monkeypatch.setattr(reference_runtime, "_model_session", lambda *args: MinimalModel())
    out = reference_runtime.reference_overlay(frame, dates, device="cpu")
    identity = out["model_asset_identity"]
    assert identity["manifest_sha256"] and identity["weights_sha256"] is None


def test_reference_context_changes_with_assets_mode_calendar_source_policy_and_runtime(overlay_inputs, monkeypatch):
    from pathlib import Path
    from my_strategy.services import reference_runtime
    frame, dates, report = overlay_inputs
    def infer(**kwargs):
        return reference_runtime.reference_overlay(frame, dates, device="cpu", **kwargs)
    base = infer(source_context_hash="source-a")
    assert base["reference_context_hash"] == infer(source_context_hash="source-a")["reference_context_hash"]
    assert base["reference_context_hash"] != infer(source_context_hash="source-b")["reference_context_hash"]
    assert base["reference_context_hash"] != infer(source_context_hash="source-a", model_mode="production")["reference_context_hash"]
    assert base["reference_context_hash"] != infer(source_context_hash="source-a", consensus_config={"risk_budget": .005})["reference_context_hash"]
    extra_calendar = dates + [(pd.Timestamp(dates[-1]) + pd.offsets.BDay()).strftime("%Y-%m-%d")]
    extended = reference_runtime.reference_overlay(frame, extra_calendar, as_of=dates[-1], device="cpu", source_context_hash="source-a")
    assert base["reference_context_hash"] != extended["reference_context_hash"]
    manifest = Path(report["folds"][0]["model_path"])
    report_file = manifest.parent / "reports" / "reference_training.json"
    report_file.write_text(report_file.read_text() + "\n")
    revised_report = infer(source_context_hash="source-a")
    assert base["model_asset_identity"]["report_sha256"] != revised_report["model_asset_identity"]["report_sha256"]
    assert base["reference_context_hash"] != revised_report["reference_context_hash"]
    revised_manifest = json.loads(manifest.read_text())
    weights = manifest.parent / "weights.pt"
    weights.write_bytes(b"another isolated model weight payload")
    revised_manifest["weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(revised_manifest))
    class RevisedModel:
        manifest = revised_manifest
        def predict(self, values):
            return np.tile([.7, .1, .2], (len(values), 1))
    monkeypatch.setattr(reference_runtime, "_model_session", lambda *args: RevisedModel())
    revised_model = infer(source_context_hash="source-a")
    assert revised_report["model_asset_identity"]["manifest_sha256"] != revised_model["model_asset_identity"]["manifest_sha256"]
    assert revised_report["model_asset_identity"]["weights_sha256"] != revised_model["model_asset_identity"]["weights_sha256"]
    assert revised_report["reference_context_hash"] != revised_model["reference_context_hash"]
    monkeypatch.setattr(reference_runtime, "_runtime_identity", lambda: {"version": "test", "code_sha256": "revised-code", "files": {}})
    assert revised_model["reference_context_hash"] != infer(source_context_hash="source-a")["reference_context_hash"]


def test_cached_loaded_manifest_mismatch_is_an_integrity_error(overlay_inputs):
    from pathlib import Path
    from my_strategy.services import reference_runtime
    frame, dates, report = overlay_inputs
    manifest = Path(report["folds"][0]["model_path"])
    data = json.loads(manifest.read_text())
    data["weights_sha256"] = "changed-manifest-during-cached-session"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="loaded session and manifest identity"):
        reference_runtime.reference_overlay(frame, dates, device="cpu")


@pytest.mark.parametrize("quote", [[0., .2, 0., 0.], [-.2, -.1, -.3, -.2],
    [9.7, 9.65, 9.5, 9.7], [9.7, 9.9, 9.8, 9.7], [9.7, np.nan, 9.5, 9.7]])
def test_invalid_current_ohlc_cannot_clear_frozen_reference_target(overlay_inputs, quote):
    from my_strategy.services import reference_runtime
    frame, dates, _ = overlay_inputs
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = quote
    out = reference_runtime.reference_overlay(frame, dates, device="cpu")
    position = out["position_reference"]
    assert position["action"] == "WAIT" and position["target_position_fraction"] == .2
    assert position["invalidation_price"] == position["previous_frozen_stop_price"] == 9.8
    assert position["stale"]
    assert "reference_frozen_stop_invalidated" not in position["reason_codes"]
    assert "reference_quote_ohlc_invalid" in position["reason_codes"]
    assert out["latest_decision"]["agents"]["ma"]["status"] == "input_ineligible"
    assert out["reference_risk_plan"]["status"] == "unavailable"
    assert out["reference_risk_plan"]["risk_position_cap_fraction"] is None
    assert not out["reference_risk_plan"]["risk_cap_evidence_reliable"]
    assert [p["side"] for p in out["reference_points"]] == ["BUY"]


def test_valid_current_quote_can_clear_old_stop_even_without_model_input(overlay_inputs):
    from my_strategy.services import reference_runtime
    frame, dates, _ = overlay_inputs
    frame.loc[frame.index[-1], ["open", "high", "low", "close"]] = [9.7, 9.9, 9.5, 9.7]
    frame.loc[frame.index[-1], "model_input_eligible"] = False
    out = reference_runtime.reference_overlay(frame, dates, device="cpu")
    assert out["position_reference"]["action"] == "SELL" and out["position_reference"]["target_position_fraction"] == 0
    assert "reference_frozen_stop_invalidated" in out["position_reference"]["reason_codes"]
    assert out["latest_decision"]["agents"]["ma"]["status"] == "input_ineligible"
