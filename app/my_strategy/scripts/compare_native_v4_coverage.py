"""Compare saved native source/risk populations without changing any model.

This report counts raw observations, independent source eligibility, risk
qualification, reference selection and matured actual Broker targets. Source
membership can overlap and is never added to claim a union population count.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd

from my_strategy.core.run_context import create_run_context, stable_hash
from my_strategy.services.czsc_high_return_runtime_v4 import read_run

SOURCES = ("ma", "structure", "wyckoff")


def _count(series: pd.Series) -> int:
    return int(series.fillna(False).astype(bool).sum())


def candidate_coverage(candidates: list[dict]) -> dict:
    by_source = {}
    for source in SOURCES:
        members = [row for row in candidates if source in row.get("sources", [])]
        reasons = Counter(reason for row in members for reason in row.get("source_reasons", {}).get(source, []))
        by_source[source] = {
            "member_opportunities": len(members),
            "source_risk_qualified": sum(bool(row.get("source_eligibility", {}).get(source)) for row in members),
            "source_selected_by_reference": sum(bool(row.get("source_selected_by_reference", {}).get(source)) for row in members),
            "source_reasons": dict(reasons),
        }
    return {
        "union_opportunities": len(candidates),
        "union_risk_qualified": sum(bool(row.get("candidate_eligible")) for row in candidates),
        "union_selected_by_reference": sum(bool(row.get("selected_by_reference")) for row in candidates),
        "by_source_overlapping_membership": by_source,
    }


def summarize_entries(frame: pd.DataFrame) -> dict:
    mature = frame.label_available.fillna(False).astype(bool)
    outcome = frame.loc[mature]
    by_source = {}
    for source in SOURCES:
        flag = frame.get("source_" + source, pd.Series(False, index=frame.index)).fillna(0).astype(bool)
        values = frame.loc[flag]
        live = values.label_available.fillna(False).astype(bool)
        by_source[source] = {
            "rows": int(flag.sum()), "mature": int(live.sum()),
            "positive": int((values.loc[live, "label"] == 1).sum()),
            "unavailable_reasons": dict(Counter(values.loc[~live, "label_reason"].fillna("unknown"))),
        }
    return {
        "rows": len(frame), "input_eligible": _count(frame.input_eligible),
        "selected_by_reference": _count(frame.selected_by_reference),
        "risk_available": _count(frame.risk_available), "mature": int(mature.sum()),
        "positive": int((outcome.label == 1).sum()),
        "mature_positive_fraction": float((outcome.label == 1).mean()) if len(outcome) else None,
        "unavailable_rows": int((~mature).sum()),
        "unavailable_reasons": dict(Counter(frame.loc[~mature, "label_reason"].fillna("unknown"))),
        "outcome_classes": dict(Counter(outcome.outcome_class.fillna("unknown"))),
        "by_source_overlapping_membership": by_source,
    }


def inspect_run(run_id: str, start: str) -> tuple[dict, dict]:
    root, report = read_run(run_id)
    if report is None or report.get("failures"):
        raise ValueError("coverage requires a completed failure-free native training run")
    events, candidates, routes, stock_records, stage_frames = [], [], [], [], {}
    keyed_candidates = {}
    for record in report["dataset_records"]:
        symbol = record["symbol"]
        stock = root / "dataset" / "l2_final" / symbol.replace(".", "_")
        points = json.loads((stock / "points.json").read_text())
        local_events = [row for row in points["events"] if str(row["date"])[:10] >= start]
        local_candidates = [row for row in points["candidates"] if str(row["date"])[:10] >= start]
        local_routes = [row for row in points["ma_routes"] if str(row["date"])[:10] >= start]
        events.extend(local_events); candidates.extend(local_candidates); routes.extend(local_routes)
        stock_records.append({"symbol": symbol, "points_sha256": hashlib.sha256((stock / "points.json").read_bytes()).hexdigest(),
                              "candidate_coverage": candidate_coverage(local_candidates)})
        for candidate in local_candidates:
            key = symbol + ":" + str(candidate["date"])[:10]
            if key in keyed_candidates:
                raise ValueError("more than one union opportunity per stock/date")
            keyed_candidates[key] = candidate
        for stage in ("bootstrap", "exit_bound_entry", "l2_final"):
            dataset = root / "dataset" / stage / symbol.replace(".", "_") / "entries.parquet"
            frame = pd.read_parquet(dataset)
            frame = frame.loc[pd.to_datetime(frame.date) >= start].copy()
            stage_frames.setdefault(stage, []).append(frame)
    source_counts = {}
    for source in SOURCES:
        selected = [row for row in events if row.get("source") == source]
        source_counts[source] = {
            "events": len(selected), "by_side": dict(Counter(row["side"] for row in selected)),
            "input_eligible": sum(bool(row.get("input_eligible")) for row in selected),
            "evidence_visible": sum(bool(row.get("evidence_visible")) for row in selected),
            "candidate_eligible": sum(bool(row.get("candidate_eligible")) for row in selected),
            "direction_eligible": sum(bool(row.get("direction_eligible")) for row in selected),
            "activation_kind": dict(Counter(row.get("activation_kind", "unspecified") for row in selected)),
            "reason_codes": dict(Counter(reason for row in selected for reason in row.get("reason_codes", []))),
        }
    in_scope = [row for row in routes if row.get("in_scope")]
    result = {
        "run_id": run_id, "risk_policy": report["config"].get("risk_policy"),
        "source_snapshot_hash": report["source_snapshot"]["snapshot_hash"],
        "cohort_hash": stable_hash(report["source_snapshot"]["selected_symbols"]),
        "source_data_end": report["source_snapshot"]["source_data_end"],
        "symbols": len(stock_records), "start": start,
        "pipeline": report["pipeline"], "events_by_source": source_counts,
        "candidate_coverage": candidate_coverage(candidates),
        "ma_forward_routes": {"all_daily_rows": len(routes), "fresh_risk_scope_rows": len(in_scope),
                              "in_scope_probability_available": sum(row.get("probability") is not None for row in in_scope),
                              "in_scope_modes": dict(Counter(row.get("mode", "unknown") for row in in_scope)),
                              "in_scope_reason_codes": dict(Counter(reason for row in in_scope for reason in row.get("reason_codes", [])))},
        "entry_target_stages": {stage: summarize_entries(pd.concat(frames, ignore_index=True)) for stage, frames in stage_frames.items()},
        "stock_records": stock_records,
    }
    return result, keyed_candidates


def compare_candidates(main: dict, control: dict) -> dict:
    shared = set(main) & set(control)
    transitions = Counter()
    source_transitions = {source: Counter() for source in SOURCES}
    mismatched_membership, changed_stops = [], []
    for key in sorted(shared):
        a, b = main[key], control[key]
        transitions[f"main_{bool(a['candidate_eligible'])}_control_{bool(b['candidate_eligible'])}"] += 1
        if set(a["sources"]) != set(b["sources"]):
            mismatched_membership.append(key)
        for source in SOURCES:
            if source not in set(a["sources"]) | set(b["sources"]):
                continue
            source_transitions[source][f"main_{bool(a['source_eligibility'].get(source))}_control_{bool(b['source_eligibility'].get(source))}"] += 1
        if a.get("stop_price") != b.get("stop_price"):
            changed_stops.append({"stock_date": key, "main_stop": a.get("stop_price"), "control_stop": b.get("stop_price"),
                                  "main_qualified": a["candidate_eligible"], "control_qualified": b["candidate_eligible"]})
    return {"shared_stock_dates": len(shared), "main_only_stock_dates": sorted(set(main) - set(control)),
            "control_only_stock_dates": sorted(set(control) - set(main)),
            "risk_qualification_transitions": dict(transitions),
            "source_risk_transitions": {source: dict(counts) for source, counts in source_transitions.items()},
            "raw_source_membership_differences": mismatched_membership,
            "changed_union_stops": changed_stops,
            "identity_basis": "stock_date_matching_raw_source_membership; opportunity IDs bind different risk contracts"}


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run-id", required=True)
    parser.add_argument("--risk-control-run-id", required=True)
    parser.add_argument("--start", default="2023-01-01")
    args = parser.parse_args(argv)
    main_report, main_candidates = inspect_run(args.main_run_id, args.start)
    control_report, control_candidates = inspect_run(args.risk_control_run_id, args.start)
    if main_report["cohort_hash"] != control_report["cohort_hash"]:
        raise ValueError("coverage comparisons require the same frozen cohort")
    config = {"version": "native_v4_saved_population_coverage_20261012", "main_run_id": args.main_run_id,
              "risk_control_run_id": args.risk_control_run_id, "start": args.start,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "publication_allowed": False, "development_diagnostic_only": True}
    context = create_run_context(task="czsc-native-v4-risk-coverage", as_of_date=main_report["source_data_end"],
                                 config=config, stocks=[], data_version=stable_hash([main_report["source_snapshot_hash"], control_report["source_snapshot_hash"]]))
    report = {"run_id": context.run_id, "config": config, "main": main_report, "risk_control": control_report,
              "paired_candidate_coverage": compare_candidates(main_candidates, control_candidates),
              "caveats": ["source membership overlaps and cannot be added as a union count",
                          "unavailable, unexecuted, unclosed and existing-inventory cases retain null targets",
                          "mature positive rates are actual frozen-target diagnostics, not return or model acceptance",
                          "each risk arm uses its own regenerated labels and models; no incompatible-head reuse"]}
    path = context.subdir("reports") / "risk_coverage.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    context.write_metadata({"status": "completed", "config": config, "publication_allowed": False})
    print(json.dumps({"run_id": context.run_id, "report": str(path)}), flush=True)
    return report


if __name__ == "__main__":
    main()
