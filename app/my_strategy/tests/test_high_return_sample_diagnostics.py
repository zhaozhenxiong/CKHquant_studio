"""Attrition must explain the frozen gates without changing sample eligibility."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.cli import build_parser
from my_strategy.services.czsc_high_return_sample_diagnostics import (
    _ledger_values, _read_verified, add_candidate_details, diagnose_entry_frame, render_markdown,
    verify_pooled_outcomes,
)
from my_strategy.services.czsc_trend_features import ENTRY_FEATURE_COLUMNS
from my_strategy.services.czsc_trend_interval_model import _weights, time_masks


FOLD = {
    "name": "diagnostic-unit", "train_end": "2025-01-10",
    "calibration_start": "2025-01-13", "calibration_end": "2025-01-24",
    "policy_start": "2025-01-27", "policy_end": "2025-02-07",
    "validation_end": "2025-02-07",
}
CONFIG = {"min_train_rows": 2, "min_calibration_rows": 2,
          "high_return": {"min_net_return": .08, "min_r_multiple": 2.}}


def sample(identifier, date, end, label=0., *, group=None, atoms=None, first=None, **changes):
    """One real-clone row with the full frozen operation input schema."""
    atom_ids = atoms or [identifier]
    net = .12 if label == 1 else -.02
    row = dict.fromkeys(ENTRY_FEATURE_COLUMNS, 1.)
    row.update(symbol="TEST.SZ", date=date, label_end=end, label=label,
        label_available=True, input_eligible=True, attempt_id=identifier,
        economic_group_id=group or identifier, opportunity_id=group or identifier,
        original_event_ids=atom_ids,
        original_event_first_pre_gate_dates=json.dumps(first or {atom: date for atom in atom_ids}),
        sources="structure", reference_allowed=False, reference_gate_status="scored_rejected",
        reference_gate_reason="high_return_below_threshold", label_reason="available",
        net_return=net, net_profit=net * 1000., entry_cash=1000., exit_cash=(1 + net) * 1000.,
        risk_cash=40., risk_fraction=.04, r_multiple=net / .04, fees=2.)
    row.update(changes)
    return row


def qualified_frame():
    return pd.DataFrame([
        sample("train-positive", "2025-01-02", "2025-01-06", 1.),
        sample("train-negative", "2025-01-03", "2025-01-07"),
        sample("cal-positive", "2025-01-13", "2025-01-15", 1.),
        sample("cal-negative", "2025-01-14", "2025-01-16"),
        sample("policy-positive", "2025-01-27", "2025-01-29", 1.),
        sample("policy-negative", "2025-01-28", "2025-01-30"),
    ])


def assert_original_masks_and_weights(frame, audit):
    for name, mask in zip(("train", "calibration", "policy"), time_masks(frame, FOLD, head="entry")):
        assert np.array_equal(audit[name + "_selected"].to_numpy(), mask)
        assert audit.loc[~mask, name + "_weight"].eq(0).all()
        if mask.any():
            assert audit.loc[mask, name + "_weight"].to_numpy() == pytest.approx(
                _weights(frame, mask, head="entry"))


def test_first_failure_losses_reconcile_and_match_unchanged_masks():
    frame = qualified_frame()
    rows = [
        sample("unfilled", "2025-01-15", None, np.nan, label_available=False,
               label_reason="entry_unfilled", input_eligible=False),
        sample("unclosed", "2025-01-16", None, np.nan, label_available=False,
               label_reason="episode_unclosed"),
        sample("bad-risk", "2025-01-17", "2025-01-20", np.nan,
               label_available=False, label_reason="risk_identity_unavailable"),
        sample("bad-input", "2025-01-15", "2025-01-17", 1., input_eligible=False),
        sample("bad-label", "2025-01-16", "2025-01-20", .5),
        sample("missing-end", "2025-01-17", None),
        sample("old-family", "2025-01-15", "2025-01-17", 1., group="train-positive",
               atoms=["train-positive"], first={"train-positive": "2025-01-02"}),
        sample("late-liquidation", "2025-01-20", "2025-01-27", 1.),
        sample("cross-atom", "2025-01-15", "2025-01-17", 1.,
               atoms=["train-negative", "new-atom"],
               first={"train-negative": "2025-01-03", "new-atom": "2025-01-15"}),
        sample("outside", "2025-02-10", None, np.nan, label_available=False,
               label_reason="episode_unclosed"),
    ]
    frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    original = frame.copy(deep=True)
    diagnostic, audit = diagnose_entry_frame(frame, FOLD, CONFIG)
    calibration = diagnostic["segments"]["calibration"]
    expected = {"label_unavailable": 3, "input_ineligible": 1, "invalid_binary_label": 1,
        "label_end_missing": 1, "family_first_outside_segment": 1,
        "label_end_after_cutoff": 1, "atomic_dependency_outside_segment": 1}
    assert {key: item["rows"] for key, item in calibration["exclusive_losses"].items()} == expected
    assert calibration["selected"] == {"rows": 2, "positive": 1, "negative": 1, "unknown": 0}
    assert sum(expected.values()) + calibration["selected"]["rows"] == calibration["target"]["candidates"]
    assert calibration["funnel"][0]["unknown"] == 4
    assert audit.set_index("attempt_id").loc["unfilled", "calibration_reason"] == "label_unavailable"
    assert audit.set_index("attempt_id").loc["outside", "calibration_reason"] == "outside_decision_segment"
    assert calibration["reference_quality"]["high_labels_retained_despite_reference_rejection"] == 5
    assert calibration["reference_quality"]["quality_rejection_removes_training_candidate"] is False
    assert_original_masks_and_weights(frame, audit)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("reason", ["entry_unfilled", "episode_unclosed", "risk_identity_unavailable",
                                  "price_basis_or_corporate_action_ambiguous"])
def test_unavailable_clones_remain_unknown_and_never_become_negative(reason):
    pending = sample("unknown", "2025-01-15", None, np.nan, label_available=False,
                     label_reason=reason, net_return=np.nan, net_profit=np.nan, r_multiple=np.nan)
    frame = pd.concat([qualified_frame(), pd.DataFrame([pending])], ignore_index=True)
    diagnostic, audit = diagnose_entry_frame(frame, FOLD, CONFIG)
    calibration = diagnostic["segments"]["calibration"]
    assert calibration["exclusive_losses"]["label_unavailable"] == {
        "rows": 1, "positive": 0, "negative": 0, "unknown": 1}
    assert calibration["target"]["exclusive_target_reasons"][reason] == 1
    assert calibration["target"]["available_labels"] == 2
    assert calibration["selected"]["negative"] == 1
    assert not audit.loc[audit.attempt_id.eq("unknown"), "calibration_selected"].any()


def test_future_unclosed_attempt_does_not_erase_mature_family_history():
    frame = qualified_frame()
    before, audit_before = diagnose_entry_frame(frame, FOLD, CONFIG)
    later = sample("later-open", "2025-01-15", None, np.nan,
        group="train-positive", atoms=["train-positive"],
        first={"train-positive": "2025-01-02"}, label_available=False,
        label_reason="episode_unclosed", net_return=np.nan, r_multiple=np.nan)
    extended = pd.concat([frame, pd.DataFrame([later])], ignore_index=True)
    after, audit_after = diagnose_entry_frame(extended, FOLD, CONFIG)
    for name in ("train", "calibration", "policy"):
        assert before["segments"][name]["selected"] == after["segments"][name]["selected"]
        assert np.array_equal(audit_before[name + "_selected"], audit_after[name + "_selected"].iloc[:-1])
        assert np.array_equal(audit_before[name + "_weight"], audit_after[name + "_weight"].iloc[:-1])
    assert audit_after.loc[0, "train_selected"]
    assert not audit_after.iloc[-1].train_selected


def test_invalid_maturity_and_changed_original_first_date_fail_closed():
    frame = qualified_frame()
    invalid_end = frame.copy()
    invalid_end.loc[0, "label_end"] = invalid_end.loc[0, "date"]
    with pytest.raises(ValueError, match="label maturity must follow decision date"):
        diagnose_entry_frame(invalid_end, FOLD, CONFIG)
    changed_first = sample("changed-prefix", "2025-01-15", "2025-01-17", 1.,
        atoms=["train-positive"], first={"train-positive": "2025-01-01"})
    with pytest.raises(ValueError, match="first pre-gate date changed after observation"):
        diagnose_entry_frame(pd.concat([frame, pd.DataFrame([changed_first])], ignore_index=True), FOLD, CONFIG)


def test_diagnostic_refuses_to_explain_a_mask_different_from_training(monkeypatch):
    from my_strategy.services import czsc_high_return_sample_diagnostics as diagnostics
    frame = qualified_frame()
    masks = tuple(mask.copy() for mask in time_masks(frame, FOLD, head="entry"))
    masks[1][2] = False
    monkeypatch.setattr(diagnostics, "time_masks", lambda *args, **kwargs: masks)
    with pytest.raises(ValueError, match="selection differs from frozen training time_masks"):
        diagnose_entry_frame(frame, FOLD, CONFIG)


def test_shared_original_atoms_have_original_economic_component_weights():
    frame = pd.DataFrame([
        sample("A", "2025-01-02", "2025-01-06", 1., atoms=["A"]),
        sample("AB", "2025-01-03", "2025-01-07", 1., atoms=["A", "B"],
               first={"A": "2025-01-02", "B": "2025-01-03"}),
        sample("B", "2025-01-06", "2025-01-08", atoms=["B"], first={"B": "2025-01-03"}),
        sample("C", "2025-01-07", "2025-01-09"),
    ])
    diagnostic, audit = diagnose_entry_frame(frame, FOLD, CONFIG)
    train = diagnostic["segments"]["train"]
    assert train["economic_components"] == 2
    assert audit.train_weight.iloc[:3].sum() == pytest.approx(audit.train_weight.iloc[3])
    assert train["positive_normalized_weight"] == pytest.approx(4 / 3)
    assert train["positive_inverse_component_mass"] == pytest.approx(2 / 3)
    assert_original_masks_and_weights(frame, audit)
    other = frame.iloc[:1].copy()
    other["symbol"] = "OTHER.SZ"
    diagnostic_other, _ = diagnose_entry_frame(pd.concat([frame, other], ignore_index=True), FOLD, CONFIG)
    assert diagnostic_other["segments"]["train"]["economic_components"] == 3


def test_missing_observational_features_are_not_a_new_sample_gate():
    frame = qualified_frame()
    for column in ("zone_low_distance", "zone_high_distance", "zone_width_ratio",
                   "w_event_age", "ma_buy_event_age", "ma_sell_event_age"):
        frame[column] = np.nan
    diagnostic, audit = diagnose_entry_frame(frame, FOLD, CONFIG)
    calibration = diagnostic["segments"]["calibration"]
    assert calibration["selected"]["rows"] == 2
    assert calibration["exclusive_losses"]["input_ineligible"]["rows"] == 0
    assert calibration["feature_observations"]["rows_with_missing_features"] == 2
    assert calibration["feature_observations"]["nonfinite_by_column"]["w_event_age"] == 2
    assert calibration["source_input_availability"]["structure"]["available"] == 2
    assert_original_masks_and_weights(frame, audit)


def test_row_counts_and_train_calibration_both_classes_rules_are_preserved():
    frame = qualified_frame()
    diagnostic, _ = diagnose_entry_frame(frame, FOLD, CONFIG)
    assert diagnostic["qualification"]["passed"] is True
    one_class = frame.copy()
    one_class.loc[one_class.attempt_id.eq("cal-positive"), "label"] = 0.
    diagnostic, _ = diagnose_entry_frame(one_class, FOLD, CONFIG)
    qualification = diagnostic["qualification"]
    assert qualification["row_gate_passed"] is True
    assert qualification["both_classes_passed"] is False
    assert qualification["both_classes_required"] is True
    assert qualification["reason"] == "entry train/calibration require both classes"
    diagnostic, _ = diagnose_entry_frame(frame, FOLD, {})
    assert diagnostic["qualification"]["min_train_rows"] == 100
    assert diagnostic["qualification"]["min_calibration_rows"] == 30
    assert diagnostic["qualification"]["reason"] == "insufficient mature entry rows train=2 calibration=2"


def test_target_quadrants_and_fee_observation_keep_frozen_denominators():
    frame = pd.DataFrame([
        sample("loss", "2025-01-13", "2025-01-15"),
        sample("ordinary", "2025-01-14", "2025-01-16", net_return=.05,
               net_profit=50., exit_cash=1050., risk_cash=20., risk_fraction=.02, r_multiple=2.5),
        sample("wide-risk", "2025-01-15", "2025-01-17", net_return=.10,
               net_profit=100., exit_cash=1100., risk_cash=100., risk_fraction=.10, r_multiple=1.),
        sample("fee-edge", "2025-01-16", "2025-01-20", net_return=.079,
               net_profit=79., exit_cash=1079., r_multiple=1.975, fees=3.),
        sample("high", "2025-01-17", "2025-01-20", 1.),
    ])
    diagnostic, audit = diagnose_entry_frame(frame, FOLD, CONFIG)
    target = diagnostic["segments"]["calibration"]["target"]
    assert target["exclusive_target_reasons"] == {
        "net_nonpositive": 1, "net_below_target": 2, "r_below_target": 1, "high": 1}
    assert target["target_quadrants"] == {"both": 1, "return_only": 1, "r_only": 1, "neither": 2}
    assert target["fee_excluded_same_denominator_high"] == 2
    assert target["fee_only_high_to_negative"] == 1
    assert "slippage_retained_same_frozen_entry_cash_and_risk_cash" in target["fee_comparison_scope"]
    assert audit.set_index("attempt_id").loc["wide-risk", "required_net_return"] == pytest.approx(.20)
    assert frame.set_index("attempt_id").loc["fee-edge", "label"] == 0.


def ledger():
    return [
        {"action": "BUY", "shares": 100, "price": 10.10, "fee": 2.,
         "cash_flow": -1012., "applied_slippage": .01},
        {"action": "REDUCE", "shares": 40, "price": 10.89, "fee": 1.5,
         "cash_flow": 434.1, "applied_slippage": .01, "reason": "partial_exit"},
        {"action": "SELL", "shares": 60, "price": 11.88, "fee": 2.,
         "cash_flow": 710.8, "applied_slippage": .01, "reason": "complete_exit"},
    ]


def test_ledger_fees_and_slippage_reconcile_without_double_deduction():
    rows = ledger()
    result = _ledger_values(rows)
    assert result["ledger_complete"] is True
    assert result["ledger_gross_profit_after_slippage"] == pytest.approx(138.4)
    assert result["ledger_fees"] == pytest.approx(5.5)
    assert result["ledger_net_profit"] == pytest.approx(sum(row["cash_flow"] for row in rows))
    assert result["ledger_net_return"] == pytest.approx(132.9 / 1012.)
    assert result["observed_slippage_cash"] == pytest.approx(21.6)
    assert result["last_exit_reason"] == "complete_exit"
    open_result = _ledger_values(rows[:2])
    assert open_result["ledger_complete"] is False
    assert open_result["ledger_open_shares"] == 60
    assert open_result["ledger_net_return"] is None
    assert _ledger_values([]) == {"ledger_filled": False}


@pytest.mark.parametrize("mutation, message", [
    ("cash_flow", "cash flow does not reconcile"),
    ("multiple_buys", "multiple BUY fills"),
    ("oversold", "inventory below zero"),
])
def test_ledger_errors_fail_closed(mutation, message):
    rows = copy.deepcopy(ledger())
    if mutation == "cash_flow":
        rows[-1]["cash_flow"] += 10.
    elif mutation == "multiple_buys":
        rows.append(rows[0].copy())
    else:
        rows[-1]["shares"] = 61
        rows[-1]["cash_flow"] = 61 * rows[-1]["price"] - rows[-1]["fee"]
    with pytest.raises(ValueError, match=message):
        _ledger_values(rows)


def test_verified_json_rejects_tampering(tmp_path):
    path = tmp_path / "frozen.json"
    original = b'{"frozen": true, "rows": 2}'
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    assert _read_verified(path, digest) == {"frozen": True, "rows": 2}
    path.write_bytes(b'{"frozen": true, "rows": 3}')
    with pytest.raises(ValueError, match="input SHA256 mismatch"):
        _read_verified(path, digest)


def pooled_outcome_audit():
    return pd.DataFrame([
        {"attempt_id": "mature", "label_available": True, "label_end": "2025-01-20",
         "recomputed_label_end": "2025-01-20", "risk_cash": 40., "recomputed_risk_cash": 40.,
         "risk_fraction": .04, "recomputed_risk_fraction": .04,
         "r_multiple": 3., "recomputed_r_multiple": 3., "saved_outcome_recomputed": True},
        {"attempt_id": "unfilled", "label_available": False, "label_end": None,
         "recomputed_label_end": None, "risk_cash": np.nan, "recomputed_risk_cash": np.nan,
         "risk_fraction": np.nan, "recomputed_risk_fraction": np.nan,
         "r_multiple": np.nan, "recomputed_r_multiple": np.nan, "saved_outcome_recomputed": True},
        {"attempt_id": "unclosed", "label_available": False, "label_end": None,
         "recomputed_label_end": None, "risk_cash": 20., "recomputed_risk_cash": 20.,
         "risk_fraction": .02, "recomputed_risk_fraction": .02,
         "r_multiple": np.nan, "recomputed_r_multiple": np.nan, "saved_outcome_recomputed": True},
    ])


def test_pooled_risk_and_maturity_match_with_unknown_unfilled_and_unclosed_clones():
    audit = pooled_outcome_audit()
    original = audit.copy(deep=True)
    verify_pooled_outcomes(audit)
    pd.testing.assert_frame_equal(audit, original)
    assert audit.loc[1, ["risk_cash", "risk_fraction", "r_multiple"]].isna().all()
    assert pd.isna(audit.loc[1, "label_end"])


@pytest.mark.parametrize("key, altered", [
    ("risk_cash", 400.), ("risk_fraction", .4), ("r_multiple", 30.),
    ("label_end", "2025-01-21"),
])
def test_pooled_risk_or_label_end_tampering_fails_closed(key, altered):
    audit = pooled_outcome_audit()
    audit.loc[0, key] = altered
    with pytest.raises(ValueError, match="pooled outcome differs from recomputed clone: " + key):
        verify_pooled_outcomes(audit)


@pytest.mark.parametrize("key", ["risk_cash", "risk_fraction", "r_multiple"])
@pytest.mark.parametrize("finite_side", ["pooled", "recomputed"])
def test_unknown_risk_cannot_match_a_finite_value(key, finite_side):
    audit = pooled_outcome_audit()
    column = key if finite_side == "pooled" else "recomputed_" + key
    audit.loc[1, column] = 1.
    with pytest.raises(ValueError, match="pooled outcome differs from recomputed clone: " + key):
        verify_pooled_outcomes(audit)


def saved_detail_report(tmp_path):
    frame = qualified_frame()
    # Three weekday dates are absent from this frozen verified calendar.
    calendar = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07",
                "2025-01-13", "2025-01-14", "2025-01-20", "2025-01-27",
                "2025-01-28", "2025-01-29", "2025-01-30"]
    entry_dates = ["2025-01-03", "2025-01-06", "2025-01-14", "2025-01-14",
                   "2025-01-28", "2025-01-29"]
    exit_dates = ["2025-01-06", "2025-01-07", "2025-01-20", "2025-01-20",
                  "2025-01-29", "2025-01-30"]
    frame["entry_date"] = entry_dates
    frame["exit_date"] = exit_dates
    frame["label_end"] = exit_dates
    frame.loc[frame.attempt_id.eq("cal-positive"),
              ["label", "net_return", "net_profit", "r_multiple", "exit_cash"]] = [0., .02, 20., .5, 1020.]
    item, audit = diagnose_entry_frame(frame, FOLD, CONFIG)
    audit["last_exit_reason"] = "continuous_seller_consensus"
    daily = pd.DataFrame([
        {"date": "2025-01-13", "reason": "buy_base_not_qualified", "buy_base_score": .3,
         "sell_base_score": .3, "buy_independent_families": ["structure"]},
        {"date": "2025-01-14", "reason": "buy_base_not_qualified", "buy_base_score": .7,
         "sell_base_score": .6, "buy_independent_families": ["structure"]},
        {"date": "2025-01-20", "reason": "buy_base_not_qualified", "buy_base_score": .7,
         "sell_base_score": .6, "buy_independent_families": ["structure", "wyckoff"]},
        {"date": "2025-01-20", "reason": "buy_base_not_qualified", "buy_base_score": .5,
         "sell_base_score": 0., "buy_independent_families": ["structure", "wyckoff"]},
        {"date": "2025-01-14", "reason": "entry_envelope_not_eligible", "buy_base_score": .7,
         "sell_base_score": 0., "buy_independent_families": ["structure", "wyckoff"]},
        {"date": "2025-01-27", "reason": "buy_base_not_qualified", "buy_base_score": .3,
         "sell_base_score": .3, "buy_independent_families": ["structure"]},
    ])
    daily["risk_plan_reason_codes"] = json.dumps(["risk_stop_unavailable"])
    item["daily_candidate_path"] = {}
    for name, segment in item["segments"].items():
        segment["ledger_audit"] = {"fee_and_slippage_excluded_same_denominator_high": 0}
        scope = daily.date.le(segment["end"]) if segment["start"] is None else daily.date.between(segment["start"], segment["end"])
        item["daily_candidate_path"][name] = {"exclusive_first_block": daily.loc[scope, "reason"].value_counts().to_dict()}
    directory = tmp_path / "diagnostics" / FOLD["name"]
    directory.mkdir(parents=True)
    audit.to_parquet(directory / "samples.parquet", index=False)
    daily.to_parquet(directory / "daily_candidate_path.parquet", index=False)
    report = {"model_run_id": "frozen-model", "run_id": "fresh-diagnostic", "data_end": "2025-02-07",
        "cohort_size": 1, "folds": [item], "limitations": ["历史窗口只用于开发诊断。"],
        "rules": {"consensus": {"buy_base_threshold": .6, "min_independent": 2, "margin": .2},
                  "high_return": CONFIG["high_return"]},
        "report_paths": {"json": str(tmp_path / "reports" / "diagnostic.json")}}
    return report, calendar


def test_base_gate_first_failure_counts_are_distinct_from_overlapping_failures(tmp_path):
    report, calendar = saved_detail_report(tmp_path)
    result = add_candidate_details(report, calendar)
    path = result["folds"][0]["daily_candidate_path"]["calibration"]
    assert path["base_gate_exclusive_details"] == {
        "buy_base_below_threshold": 2, "independent_sources_insufficient": 1,
        "buy_sell_margin_insufficient": 1}
    assert sum(path["base_gate_exclusive_details"].values()) == 4
    assert path["base_gate_overlapping_failures"] == {
        "buy_base_below_threshold": 2, "independent_sources_insufficient": 2,
        "buy_sell_margin_insufficient": 3}
    assert sum(path["base_gate_overlapping_failures"].values()) > 4
    assert result["folds"][0]["diagnosis"]["quality_gate_directly_removes_label_clones"] is False


def test_holding_distance_uses_verified_calendar_and_chinese_report_renders(tmp_path):
    report, calendar = saved_detail_report(tmp_path)
    result = add_candidate_details(report, calendar)
    cal = result["folds"][0]["segments"]["calibration"]
    duration = cal["ledger_audit"]
    assert duration["holding_session_distance_mean"] == pytest.approx(1.)
    assert duration["holding_session_distance_quantiles"]["0.5"] == pytest.approx(1.)
    assert duration["holding_duration_scope"] == "verified_calendar_exit_index_minus_entry_index"
    assert len(pd.bdate_range("2025-01-14", "2025-01-20")) - 1 == 4
    assert result["folds"][0]["diagnosis"]["primary_observed_cause"] == "frozen_outcomes_do_not_meet_joint_return_and_R_target"
    markdown = render_markdown(result)
    assert "# 高收益校准正类流失诊断" in markdown
    assert "基础门槛重叠失败（不能相加）" in markdown
    assert "核验持仓交易日距离中位 1.0，平均 1.00" in markdown


def test_diagnostic_cli_requires_model_identity_and_has_bounded_worker_default():
    args = build_parser().parse_args(["model-diagnostics", "--model-run-id", "frozen-run"])
    assert args.command == "model-diagnostics"
    assert args.model_run_id == "frozen-run" and args.cpu_workers == 2
    args = build_parser().parse_args(["model-diagnostics", "--model-run-id", "frozen-run",
                                    "--cpu-workers", "4", "--json"])
    assert args.cpu_workers == 4 and args.json
    with pytest.raises(SystemExit):
        build_parser().parse_args(["model-diagnostics"])
