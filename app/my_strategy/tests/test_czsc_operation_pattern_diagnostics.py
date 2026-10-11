"""Independent diagnostics preserve time boundaries and economic sample mass."""
from __future__ import annotations

import math
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_high_return_features import ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS
from my_strategy.services.czsc_operation_pattern_diagnostics import (
    TEMPLATE_ENTRY_COLUMNS, TEMPLATE_HOLDING_COLUMNS, _mature, _write,
    discover_causal_templates, episode_excursion_diagnostics,
    summarize_episode_excursions, weighted_probability_diagnostics,
    run_operation_diagnostics, _sha,
)


FOLD = {"name": "synthetic", "train_end": "2024-12-31",
    "calibration_start": "2025-01-01", "calibration_end": "2025-01-31",
    "policy_start": "2025-02-01", "policy_end": "2025-03-31", "validation_end": "2025-03-31"}


def row(group, *, date="2025-02-05", end="2025-02-10", label=1., value=1., source="union"):
    return {"symbol": "000001.SZ", "economic_group_id": group, "episode_id": group,
        "opportunity_id": group, "date": date, "label_end": end, "label_available": True,
        "input_eligible": True, "label": label, "outcome_class": "high" if label == 1 else "loss",
        "ma20_bias": value, "source": source, "source_ma": 1., "source_structure": 1.,
        "source_wyckoff": 0.}


def templates(frame, **kwargs):
    return discover_causal_templates(frame, columns=("ma20_bias",), **kwargs)


def probability(frame, scores, **kwargs):
    return weighted_probability_diagnostics(frame, scores, head=kwargs.pop("head", "entry"),
        segment=kwargs.pop("segment", "policy"), fold=FOLD, **kwargs)


def test_future_and_unavailable_labels_cannot_fit_template_thresholds():
    frame = pd.DataFrame([row(str(i), value=i, label=i % 2) for i in range(12)])
    future = row("future", date="2026-07-01", end="2026-07-02", value=1e9)
    immature = row("immature", value=-1e9); immature["label_available"] = False
    missing_end = row("no_end", value=-1e8); missing_end["label_end"] = None
    ineligible = row("bad_input", value=1e8); ineligible["input_eligible"] = False
    changed = pd.concat([frame, pd.DataFrame([future, immature, missing_end, ineligible])], ignore_index=True)
    a, b = templates(frame), templates(changed)
    assert a["templates"] == b["templates"]
    assert b["eligible_rows"] == 12 and b["excluded_rows"] == 4


@pytest.mark.parametrize("columns", [("net_return",), ("max_favorable_excursion",),
    ("future_profit",), ("ma20_bias", "ma20_bias"), ()])
def test_outcomes_and_unknown_columns_cannot_enter_templates(columns):
    with pytest.raises(ValueError, match="approved causal"):
        discover_causal_templates(pd.DataFrame([row("a")]), columns=columns)


def test_defined_template_columns_are_causal_and_retrospective_duration_is_separate():
    assert set(TEMPLATE_ENTRY_COLUMNS) <= set(ENTRY_FEATURE_COLUMNS)
    assert "final_peak_giveback" not in TEMPLATE_HOLDING_COLUMNS
    assert "holding_sessions" not in TEMPLATE_ENTRY_COLUMNS


def test_full_interval_duration_cannot_enter_entry_template_schema():
    frame = pd.DataFrame([row("a")]); frame["holding_sessions"] = 60
    with pytest.raises(ValueError, match="approved causal"):
        discover_causal_templates(frame, columns=("holding_sessions",))
    held = discover_causal_templates(frame, columns=("holding_sessions",), population="holding")
    assert held["decision_population"] == "holding"


def test_unknown_template_population_rejected():
    with pytest.raises(ValueError, match="population"):
        templates(pd.DataFrame([row("a")]), population="future_whole_interval")


def test_repeated_holding_states_have_one_episode_mass_and_quantile_invariance():
    unique = pd.DataFrame([row("a", value=0., label=0), row("b", value=1., label=1)])
    repeated = pd.concat([unique.iloc[[0]], pd.concat([unique.iloc[[1]]] * 8)], ignore_index=True)
    a, b = templates(unique), templates(repeated)
    assert b["baseline"]["weighted_class_share"]["high"] == .5
    assert b["baseline"]["economic_groups"] == 2
    assert [x["conditions"] for x in a["templates"]] == [x["conditions"] for x in b["templates"]]
    for current in b["templates"]:
        assert not current["statistical_pass"] and not current["tradable_rule"]


def test_source_fractions_are_nonexclusive_and_not_an_invented_partition():
    result = templates(pd.DataFrame([row("a"), row("b", label=0)]))
    shares = result["baseline"]["source_proportions_nonexclusive"]
    assert shares == {"ma": 1., "structure": 1., "wyckoff": 0.}
    assert sum(shares.values()) == 2.


def test_one_immature_state_censors_whole_economic_group_but_not_another_symbol():
    a, b = row("same"), row("same"); b["label_available"] = False
    c = row("same"); c["symbol"] = "000002.SZ"
    result = templates(pd.DataFrame([a, b, c]))
    assert result["eligible_rows"] == 1 and result["baseline"]["economic_groups"] == 1


def test_group_spanning_quarters_belongs_to_first_observed_quarter():
    frame = pd.DataFrame([row("span", date="2025-03-25", end="2025-04-04"),
        row("span", date="2025-04-01", end="2025-04-04")])
    result = templates(frame)
    assert list(result["time_block_baselines"]) == ["2025Q1"]
    assert result["time_block_baselines"]["2025Q1"]["economic_group_weight_mass"] == 1.


def stable_data(reverse=False):
    rows = []
    for q, date in enumerate(("2024-01-05", "2024-04-05", "2024-07-05", "2024-10-05")):
        end = (pd.Timestamp(date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        for j in range(10):
            value = float(j >= 5)
            label = int(value) if not reverse or q < 3 else 1-int(value)
            rows.append(row(f"{q}-{j}", date=date, end=end, label=label, value=value))
    return pd.DataFrame(rows)


def test_training_block_consistency_is_descriptive_and_reversal_cannot_pass():
    good = templates(stable_data(), min_template_groups=10, min_block_groups=2)
    bad = templates(stable_data(reverse=True), min_template_groups=10, min_block_groups=2)
    key = lambda x: x["conditions"] == [{"column": "ma20_bias", "operator": "gt", "value": 0.}]
    good_template = next(x for x in good["templates"] if key(x))
    bad_template = next(x for x in bad["templates"] if key(x))
    assert good_template["stability_status"] == "consistent_positive_descriptive_only"
    assert good_template["positive_blocks"] == 4
    assert bad_template["stability_status"] == "not_consistently_positive"
    assert bad_template["positive_blocks"] == 3
    assert not good_template["statistical_pass"] and not good["publication_allowed"]


def test_insufficient_history_and_no_mature_rows_do_not_fake_success():
    frame = pd.DataFrame([row("a")])
    result = templates(frame)
    assert all(x["stability_status"] == "insufficient_temporal_support" for x in result["templates"])
    frame["label_available"] = False
    empty = templates(frame)
    assert empty["eligible_rows"] == 0 and empty["templates"] == []
    assert empty["baseline"]["weighted_class_share"]["high"] is None


def test_multifeature_operations_keep_only_training_fit_thresholds_and_causal_columns():
    frame = stable_data()
    frame["ma_alignment"] = frame.ma20_bias
    frame["ma20_slope5"] = frame.ma20_bias
    frame["w_demand_score"] = frame.ma20_bias
    columns = ("ma20_bias", "ma_alignment", "ma20_slope5", "w_demand_score")
    result = discover_causal_templates(frame, columns=columns, min_template_groups=10, min_block_groups=2)
    operation = next(x for x in result["templates"] if x["family"] == "trend_with_demand")
    assert len(operation["conditions"]) == 3 and operation["weighted_class_share"]["high"] == 1.
    assert operation["stability_status"] == "consistent_positive_descriptive_only"
    assert operation["threshold_fit_scope"] == "eligible_mature_production_training_only"
    assert not operation["tradable_rule"] and not operation["statistical_pass"]


def test_weighted_ap_handles_ties_exactly_like_external_metric():
    frame = pd.DataFrame([row("a", label=0), *[row("b", label=1)] * 4,
        row("c", label=1), row("d", label=0)])
    p = np.array([.2, .5, .5, .5, .5, .5, .9])
    weights = np.array([1., .25, .25, .25, .25, 1., 1.])
    result = probability(frame, p)
    assert result["average_precision"] == pytest.approx(average_precision_score(frame.label, p, sample_weight=weights))
    assert result["weighted_brier"] == pytest.approx(np.average((p-frame.label.to_numpy())**2, weights=weights))
    assert result["weighted_positive_share"] == .5
    assert result["eligible_economic_groups"] == 4
    assert not result["threshold_selection_performed"]


def test_missing_scores_preserve_coverage_without_probability_imputation():
    frame = pd.DataFrame([row("a", label=1), *[row("b", label=0)] * 2])
    result = probability(frame, [.9, np.nan, .2])
    assert result["score_coverage"] == .75 and result["scored_group_weight_mass"] == 1.5
    assert result["unavailable_score_rows"] == 1
    missing = probability(frame, [np.nan]*3)
    assert missing["status"] == "unavailable_no_mature_scored_rows"
    assert missing["average_precision"] is None


def test_exact_fixed_bins_include_boundaries_zero_and_one():
    frame = pd.DataFrame([row(str(i), label=i % 2) for i in range(5)])
    result = probability(frame, [0., .1, .3, .9, 1.])
    assert sum(x["rows"] for x in result["calibration_bins"]) == 5
    assert [(x["lower"], x["rows"]) for x in result["calibration_bins"]] == [(0., 1), (.1, 1), (.3, 1), (.9, 2)]
    assert math.isfinite(result["weighted_log_loss"])


def test_probability_segment_purges_whole_cross_boundary_episode():
    frame = pd.DataFrame([row("cross", date="2025-01-29", end="2025-02-05"),
        row("cross", date="2025-02-02", end="2025-02-05"), row("valid", label=0)])
    policy = probability(frame, [.9, .9, .1])
    calibration = probability(frame, [.9, .9, .1], segment="calibration")
    assert policy["eligible_rows"] == 1 and calibration["eligible_rows"] == 0


@pytest.mark.parametrize("scores,segment", [([1.1], "policy"), ([-.1], "policy"), ([.5], "test")])
def test_invalid_probability_or_test_segment_rejected(scores, segment):
    with pytest.raises(ValueError):
        probability(pd.DataFrame([row("a")]), scores, segment=segment)


def test_one_class_or_no_accepted_scores_does_not_produce_precision_certificate():
    frame = pd.DataFrame([row("a", label=0), row("b", label=0)])
    result = probability(frame, [0., 0.], thresholds=(.5,))
    assert result["average_precision"] is None and result["status"] == "insufficient_two_class_support"
    assert result["fixed_threshold_precision_recall"][0]["precision"] is None
    assert result["fixed_threshold_precision_recall"][0]["recall"] is None
    assert not result["statistical_pass"]


def test_continuous_advantage_diagnostics_use_real_values_not_binary_clipping():
    frame = pd.DataFrame([row("a", label=-.2), *[row("b", label=.3)] * 2])
    result = probability(frame, [-.1, .2, .4], head="exit_advantage")
    assert result["status"] == "continuous_head_diagnostic"
    assert result["weighted_MSE"] == pytest.approx(.01)
    assert result["weighted_MAE"] == pytest.approx(.1)
    assert result["average_precision"] is None


def actual_episode():
    ledger = [{"action": "BUY", "date": "2025-01-02", "price": 10., "shares": 100,
        "fee": 5., "cash_flow": -1005.},
        {"action": "SELL", "date": "2025-01-06", "price": 12., "shares": 40, "fee": 5., "cash_flow": 475.},
        {"action": "SELL", "date": "2025-01-08", "price": 11., "shares": 60, "fee": 5., "cash_flow": 655.}]
    episode = {**row("actual", date="2025-01-01", end="2025-01-08"), "entry_date": "2025-01-02",
        "exit_date": "2025-01-08", "open_shares": 0, "ledger_sha256": stable_hash(ledger)}
    snapshots = [{"entry_date": "2025-01-02", "date": "2025-01-02", "shares": 100,
        "bar": {"high": 10.5, "low": 9.}},
        {"entry_date": "2025-01-02", "date": "2025-01-06", "shares": 60,
        "bar": {"high": 13., "low": 11.}}]
    return episode, snapshots, ledger


def test_excursion_recomputation_uses_actual_fee_basis_partial_fills_and_visible_interval():
    episode, snapshots, ledger = actual_episode()
    snapshots.extend([{"entry_date": "2025-01-02", "date": "2025-01-20", "shares": 100,
        "bar": {"high": 1e9, "low": 0.}},
        {"entry_date": "2024-01-01", "date": "2025-01-06", "shares": 100,
        "bar": {"high": 1e9, "low": 0.}}])
    result = episode_excursion_diagnostics(episode, snapshots, ledger=ledger)
    assert result["max_favorable_excursion"] == pytest.approx(13/10.05-1)
    assert result["max_adverse_excursion"] == pytest.approx(9/10.05-1)
    assert result["final_peak_giveback"] == pytest.approx(2/10.05)
    assert result["entry_fee_ratio"] == pytest.approx(5/1005)
    assert result["scope"] == "whole_episode_retrospective_never_prediction_input"


def test_fill_price_is_observed_even_when_exit_day_has_no_holding_snapshot():
    episode, _, ledger = actual_episode()
    result = episode_excursion_diagnostics(episode, [], ledger=ledger)
    assert result["max_favorable_excursion"] == pytest.approx(12/10.05-1)
    assert result["max_adverse_excursion"] == pytest.approx(10/10.05-1)


@pytest.mark.parametrize("change", ["wrong_hash", "oversell", "second_buy"])
def test_fabricated_interval_or_inventory_rejected(change):
    episode, snapshots, ledger = actual_episode()
    if change == "wrong_hash": episode["ledger_sha256"] = "bad"
    elif change == "oversell": ledger[-1]["shares"] = 61; episode["ledger_sha256"] = stable_hash(ledger)
    else: ledger[-1]["action"] = "BUY"; episode["ledger_sha256"] = stable_hash(ledger)
    with pytest.raises(ValueError):
        episode_excursion_diagnostics(episode, snapshots, ledger=ledger)


def test_unfinished_actual_episode_remains_unavailable():
    episode, snapshots, ledger = actual_episode(); episode["open_shares"] = 10
    assert episode_excursion_diagnostics(episode, snapshots, ledger=ledger)["available"] is False


def test_retrospective_sources_are_separate_policies_not_joint_maturity():
    a = row("same", source="ma")
    b = row("same", source="structure", end="2026-07-02")
    frame = pd.DataFrame([a, b])
    assert _mature(frame, "2026-06-30").sum() == 0
    result = summarize_episode_excursions(frame)
    assert result["eligible_rows"] == 1
    assert next(x for x in result["groups"] if x["source"] == "ma" and x["outcome_class"] == "high")["rows"] == 1
    assert not result["prediction_X_contains_retrospective_fields"]


def test_diagnostic_outputs_cannot_overwrite_evidence(tmp_path):
    path = tmp_path / "report.json"
    _write(path, {"status": "pending"})
    with pytest.raises(FileExistsError):
        _write(path, {"status": "pass"})
    assert path.read_text() == '{\n  "status": "pending"\n}'


def test_end_to_end_keeps_mature_candidate_when_reference_account_never_traded(tmp_path, monkeypatch):
    """Counterfactual candidates do not require an actual reference episode."""
    from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
    from my_strategy.services import czsc_high_return_research as research
    run = "synthetic-independent-diagnostic"
    root = ARTIFACT_RUNS_ROOT / run
    root.mkdir(parents=True)
    code = tmp_path / "frozen.py"; code.write_text("frozen = True\n")
    monkeypatch.setattr(research, "_code_files", lambda: [code])
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    save(root / "config/initial_code_hashes.json", {str(code): _sha(code)})
    save(root / "config/research.json", {"production": {"name": "production", "train_end": "2026-06-30"},
        "development_windows": [], "architectures": [], "feature_start": "2023-01-01"})
    save(root / "data_snapshots/source.json", {"snapshot_hash": "synthetic-frozen-source"})
    records = []
    for symbol, no_reference in (("000001.SZ", True), ("000002.SZ", False)):
        folder = root / "dataset/final" / symbol.replace(".", "_")
        folder.mkdir(parents=True)
        values = row(symbol)
        values.update({column: .01 for column in ENTRY_FEATURE_COLUMNS})
        values.update(symbol=symbol, date="2025-01-01", label_end="2025-01-08")
        pd.DataFrame([values]).to_parquet(folder / "entries.parquet", index=False)
        if no_reference:
            pd.DataFrame().to_parquet(folder / "episodes.parquet", index=False)
            pd.DataFrame().to_parquet(folder / "exits.parquet", index=False)
        else:
            episode, snapshots, ledger = actual_episode()
            episode.update(symbol=symbol, signal_date=episode["date"])
            excursion = episode_excursion_diagnostics(episode, snapshots, ledger=ledger)
            episode.update({field: excursion[field] for field in (
                "max_favorable_excursion", "max_adverse_excursion", "final_peak_giveback", "entry_fee_ratio")})
            pd.DataFrame([episode]).to_parquet(folder / "episodes.parquet", index=False)
            held = {column: .01 for column in EXIT_FEATURE_COLUMNS}
            held.update(row("actual", date="2025-01-05", end="2025-01-08"))
            held["symbol"] = symbol
            pd.DataFrame([held]).to_parquet(folder / "exits.parquet", index=False)
            save(folder / "reference/union/holding_snapshots.json", snapshots)
            save(folder / "reference/union/ledger.json", ledger)
        save(folder / "contracts.json", {})
        record = {"symbol": symbol, "episode_status": "completed", "path": str(folder)}
        for kind in ("entries", "exits", "episodes", "contracts"):
            path = folder / (kind + (".json" if kind == "contracts" else ".parquet"))
            record.update({kind + "_path": str(path), kind + "_sha256": _sha(path)})
        records.append(record)
    save(root / "dataset/final/manifest.json", {"requested": 2, "successful": 2, "failures": [],
        "sources": ["ma", "structure", "wyckoff", "union"], "records": records})
    before = {str(p): _sha(p) for p in root.rglob("*") if p.is_file()}
    result = run_operation_diagnostics(run, include_probabilities=False)
    assert result["coverage"]["entry_rows"] == 2
    assert result["coverage"]["holding_rows"] == 1
    assert result["retrospective_actual_interval_diagnostics"]["independently_recomputed_union_intervals"] == 1
    assert {str(p): _sha(p) for p in root.rglob("*") if p.is_file()} == before
    saved = ARTIFACT_RUNS_ROOT / result["run_id"] / "reports/operation-diagnostics.json"
    assert json.loads(saved.read_text())["coverage"]["entry_rows"] == 2
    assert not result["publication_allowed"]
