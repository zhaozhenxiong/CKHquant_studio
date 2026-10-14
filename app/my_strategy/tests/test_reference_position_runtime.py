"""Reference paths are causal recommendations, not inferred actual inventory."""
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
    manifest.write_text("{}")
    report = {"config": {"position": {}}, "folds": [{"name": "fold", "split": {
        "test_start": dates[64], "test_end": dates[-1]}, "model_path": str(manifest),
        "research_reconstruction": True, "predictive_gate": {"passed": True}}]}
    monkeypatch.setattr(reference_training, "active_reference_run_id", lambda: "reference-run")
    monkeypatch.setattr(reference_training, "load_reference_run", lambda run_id=None: (tmp_path, report))
    monkeypatch.setattr(czsc_native_evidence, "build_native_evidence", lambda data, **kw:
        {"frame": data.copy(), "evaluations": [], "events": [], "wyckoff_events": []})
    class Model:
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
