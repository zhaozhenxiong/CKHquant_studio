"""Independent L1 dependency audit; staging only until its owner integrates it.

Recomputes canonical rules, target contracts, candidate-time policy bindings and
train/calibration/policy masks from original immutable stage files. It does not
fit models, refit preprocessing/Platt, replay a Broker or approve publication.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_high_return_features import L1_MA_ENTRY_COLUMNS, L1_MA_EXIT_COLUMNS
from my_strategy.services.czsc_high_return_models import bind_inputs, load_head, split_validation, time_masks
from my_strategy.services.czsc_research import _file_hash
from my_strategy.services.czsc_signal_points import build_signal_points, make_point_decisions
from my_strategy.services.czsc_trade_episodes import training_frame

AUDIT_VERSION = "czsc_ma_entry_signal_pinned_dependency_audit_v1"
STAGE_ORDER = ["rule_v0_bootstrap", "train_only_ma_exit", "prior_exit_only_entry_targets", "train_only_ma_entry"]
REQUIRED_ALIGNMENT_CODE = {'czsc_high_return_ma_training.py', 'czsc_high_return_ma_alignment.py',
    'czsc_high_return_ma_alignment_audit.py'}


def _inside(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("alignment artifact is outside this immutable run")
    return path


def _read(path):
    return json.loads(Path(path).read_text())


def _hashed(path, expected, root):
    path = _inside(path, root)
    if not expected or _file_hash(path) != expected:
        raise ValueError("alignment original artifact SHA256 differs")
    return path


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _stage_manifest(root, stage, expected_sources, records):
    path = root / "dataset" / stage / "manifest.json"
    manifest = _read(path)
    expected = {r["symbol"]: r for r in records}
    actual = manifest.get("records", [])
    _require(bool(expected) and len(expected) == len(records)
        and [r["symbol"] for r in actual] == sorted(expected)
        and manifest.get("sources") == expected_sources
        and manifest.get("stage") == ("ma_entry_alignment" if stage == "ma_entry_alignment" else "episodes")
        and not manifest.get("failures")
        and manifest.get("requested") == len(records)
        and manifest.get("successful") == len(records)
        and manifest.get("hash_verified_count") == len(records),
        "alignment stage lacks zero-failure full ordered original-source coverage")
    from my_strategy.services.czsc_high_return_research import _mainboard
    _require(manifest.get("supported_mainboard") == sum(_mainboard(s) for s in expected),
        "alignment supported-board count changed")
    for item in actual:
        source = expected[item["symbol"]]
        board = _mainboard(item["symbol"])
        _require(item.get("source_path") == source["path"]
            and item.get("source_sha256") == source["sha256"]
            and item.get("source_hash_verified") is True and item.get("tradable_board") == board,
            "alignment record differs from frozen original source")
        _require(item.get("episode_status") == ("completed" if board else "cash_only_unsupported_board"),
            "unsupported source was relabeled or a supported account disappeared")
    return manifest


def _frame_files(root, record):
    frames = {}
    for kind in ("entries", "exits", "episodes"):
        path = _hashed(record[kind + "_path"], record[kind + "_sha256"], root)
        frames[kind] = pd.read_parquet(path)
    contract_path = _hashed(record["contracts_path"], record["contracts_sha256"], root)
    frames["contracts"] = _read(contract_path)
    for source, contracts in frames["contracts"].items():
        for contract in contracts.values():
            _require(contract.get("source") == source
                and contract.get("contract_hash") == stable_hash({k:v for k,v in contract.items() if k != "contract_hash"}),
                "alignment target contract self/source identity differs")
    for kind, head in (("entries", "entry"), ("exits", "exit")):
        _require(frames[kind].attrs.get("source_contracts") ==
            {s:c[head] for s,c in frames["contracts"].items()},
            "serialized alignment frame target attrs differ from contracts.json")
    return frames


def _raw_population_identity(points, raw_points):
    _require(stable_hash(points) == stable_hash(raw_points),
        'bootstrap/alignment target population is not exact independent raw-wide MA points')


def _bootstrap_original_contract(data, expected_v0, source_hash):
    _require(data['contracts']['ma']['ma_exit'] == expected_v0,
        'counterfeit bootstrap did not use the canonical MA rule-v0 contract')
    for kind in ('entries', 'exits', 'episodes'):
        rows = data[kind]
        if len(rows):
            _require(rows.source.eq('ma').all() and rows.source_snapshot_hash.eq(source_hash).all()
                and rows.continuation_policy_hash.eq(expected_v0['continuation_policy_hash']).all(),
                'bootstrap rows differ from the canonical original v0 source/policy')


def _unavailable_and_maturity(data):
    for kind in ("entries", "exits"):
        frame = data[kind]
        if not len(frame):
            continue
        available = frame.label_available.fillna(False).astype(bool)
        labels = pd.to_numeric(frame.label, errors="coerce")
        _require(not (np.isfinite(labels) & ~available).any(),
            "an unavailable/censored target became a classifier negative")
        _require(not (available & ~labels.isin([0., 1.])).any(), "mature classifier target is not binary")
        ends = pd.to_datetime(frame.label_end, errors="coerce")
        days = pd.to_datetime(frame.date, errors="coerce")
        _require(not (available & (ends.isna() | (ends <= days))).any(),
            "mature target does not follow its original decision date")
        if kind == "entries":
            positive = pd.to_numeric(frame.positive_return_label, errors="coerce")
            net = pd.to_numeric(frame.net_return, errors="coerce")
            _require(not (available & (positive != (net > 0).astype(float))).any(),
                "MA-entry label differs from complete fee-net positive return")
            physical_end = pd.to_datetime(frame.exit_date, errors="coerce")
            _require(not (available & (physical_end.isna() | (physical_end != ends))).any(),
                "entry target matured before final actual liquidation")
            if "open_shares" in frame:
                censored = pd.to_numeric(frame.open_shares, errors="coerce").fillna(0) > 0
                _require(not (censored & (available | np.isfinite(positive))).any(),
                    "an open actual holding became a MA-entry negative")
            _require(not (available & ((pd.to_numeric(frame.entry_shares, errors="coerce") <= 0)
                | pd.to_datetime(frame.entry_date, errors="coerce").isna())).any(),
                "an unfilled candidate became a mature MA-entry target")
        else:
            forced = pd.to_datetime(frame.forced_exit_date, errors="coerce")
            continued = pd.to_datetime(frame.continuation_exit_date, errors="coerce")
            paired_end = pd.concat([forced, continued], axis=1).max(axis=1)
            _require(not (available & (forced.isna() | continued.isna() | (paired_end != ends))).any(),
                "holding target matured before both complete liquidation branches")


def _annotated_rows(data, policy, calendar, symbol):
    """Verify stored annotations against a fresh shared factory, never decorate."""
    from my_strategy.services.czsc_high_return_ma_alignment import ALIGNMENT_VERSION
    next_day = dict(zip(calendar[:-1], calendar[1:]))
    bindings = policy.bindings
    by_signal = {b["candidate_signal_date"]: b for b in bindings.values()}
    episodes = {r["episode_id"]:r for r in data["episodes"].to_dict("records") if r["source"] == "ma"}
    _require(len(episodes) == int(data["episodes"].get("source", pd.Series(dtype=str)).eq("ma").sum()),
        "duplicate MA episode identity")
    counts = Counter()
    for binding in bindings.values():
        signal = binding['candidate_signal_date']
        _require(binding['binding_clock'] == signal + 'T15:00:00+08:00'
            and binding.get('checkpoint_upgrade') is False
            and binding['binding_hash'] == stable_hash({k:v for k,v in binding.items() if k != 'binding_hash'}),
            'candidate binding clock/hash permits a holding-day policy upgrade')
        if binding['mode'] == 'signal_time_pinned_ma_exit':
            _require(binding['available_at'] < signal and binding['train_label_end'] < signal
                and binding['calibration_label_end'] < signal and bool(binding['manifest_sha256'])
                and bool(binding['checkpoint_sha256']) and bool(binding['exit_contract_hash']),
                'future or immature exit checkpoint was used to label its own historical candidates')
        else:
            _require(binding['mode'] == 'explicit_rule_v0_no_earlier_exit_head'
                and binding['checkpoint'] is None and binding['exit_contract_hash'] is None,
                'no-checkpoint candidate lacks explicit frozen rule-v0 fallback')
    for kind in ("entries", "exits", "episodes"):
        for row in data[kind].to_dict("records"):
            if row["source"] != "ma":
                _require(pd.isna(row.get("ma_exit_binding_hash")), "non-MA target received learned-MA exit lineage")
                continue
            if kind == "entries":
                binding = by_signal[row["date"]]
                _require(row["candidate_plan_hash"] == binding["plan_sha256"]
                    and row["opportunity_id"] == binding["opportunity_id"],
                    "wide candidate plan/opportunity differs from frozen signal-time binding")
                filled = row.get("entry_date")
            else:
                episode = row if kind == "episodes" else episodes[row["episode_id"]]
                binding = bindings[episode["plan_id"]]
                _require(episode["signal_date"] == binding["candidate_signal_date"]
                    and episode["opportunity_id"] == binding["opportunity_id"],
                    "episode plan was rebound to another original candidate")
                filled = episode["entry_date"]
                if kind == "exits":
                    _require(row["actual_entry_date"] == filled and row["date"] >= filled,
                        "daily holding state differs from its original episode")
            _require(row["symbol"] == symbol == binding["symbol"]
                and row["source_snapshot_hash"] == policy.recipe["source_snapshot_hash"]
                and row["continuation_policy_hash"] == policy.continuation_policy_hash,
                "row source/actual policy differs from candidate-time exit recipe")
            if pd.notna(filled):
                _require(filled == next_day.get(binding["candidate_signal_date"]),
                    "actual entry fill is not the next verified session of the bound signal")
            saved = json.loads(row["ma_exit_binding_json"])
            _require(saved == binding and saved.get("binding_hash") == stable_hash({k:v for k,v in saved.items() if k != "binding_hash"})
                and row["ma_exit_binding_hash"] == binding["binding_hash"]
                and row["ma_exit_binding_mode"] == binding["mode"]
                and row["alignment_recipe_hash"] == policy.recipe_hash,
                "stored candidate exit binding differs from the original pinned factory")
            counts[kind + ":" + binding["mode"]] += 1
    for contract in data["contracts"]["ma"].values():
        _require(contract.get("alignment_version") == ALIGNMENT_VERSION
            and contract.get("exit_alignment_recipe_hash") == policy.recipe_hash
            and contract.get("exit_binding_clock") == "candidate_signal_close"
            and contract.get("holding_checkpoint_upgrade") is False
            and contract["continuation_policy_hash"] == policy.continuation_policy_hash,
            "old target was relabeled as v2 without generation under the pinned policy")
    _require(data["contracts"]["ma"]["ma_entry"].get("label_version") == ALIGNMENT_VERSION,
        "MA-entry target did not adopt the distinct dependency version")
    return counts


def _fold_counts(frame, contract, head, folds):
    from my_strategy.services.czsc_high_return_research import _head_data
    columns = L1_MA_ENTRY_COLUMNS if head == "ma_entry" else L1_MA_EXIT_COLUMNS
    derived, model_contract = _head_data(frame, contract, head, columns)
    if not len(derived):
        return {fold['name']:{'all':0,'train':0,'calibration':0,'policy':0} for fold in folds}, model_contract
    derived = bind_inputs(derived, columns, head)
    result = {}
    for fold in folds:
        masks = time_masks(derived, fold)
        stats = {"all":len(derived)}
        for name, mask in zip(("train", "calibration", "policy"), masks, strict=True):
            stats[name] = int(mask.sum())
            dates = pd.to_datetime(derived.loc[mask, "label_end"], errors="coerce")
            stats[name + "_label_end"] = dates.max().date().isoformat() if len(dates) else None
            if head == "ma_entry":
                stats[name + "_binding_modes"] = Counter(derived.loc[mask, "ma_exit_binding_mode"])
        # time_masks enforces original whole economic groups, maturity and gaps;
        # independently assert there is no partial selection of any such group.
        from my_strategy.services.czsc_high_return_models import _group_ids
        groups = _group_ids(derived)
        for mask in masks:
            selected = pd.Series(mask, index=derived.index).groupby(groups)
            _require(not ((selected.sum() > 0) & (selected.sum() < selected.size())).any(),
                "a whole economic group was split across fitting stages")
        result[fold["name"]] = stats
    return result, model_contract


def audit_ma_alignment(model_root, report, calendar, *, check_cancel=None):
    """Audit a completed new run. Every failure is explicit; nothing is written.

The caller may integrate each ``errors`` item with its existing state.block,
and persist this returned evidence under a new immutable audit name. Run only
after training completes; per-record processing bounds retained data memory.
"""
    from my_strategy.services.czsc_high_return_ma_training import (
        ALIGNMENT_MODE, _expected_v0_exit_contract, read_exit_bootstrap_lineage)
    from my_strategy.services.czsc_high_return_ma_alignment import make_signal_pinned_ma_exit_policy
    from my_strategy.services.czsc_high_return_research import _ma_folds, _read_features
    root = Path(model_root).resolve()
    check = check_cancel or (lambda: None)
    errors, verified, counts = [], Counter(), Counter()
    fold_totals, model_contracts = {"ma_exit":{}, "ma_entry":{}}, {}
    config = report.get("config", {})
    if config.get("ma_entry_alignment") != ALIGNMENT_MODE:
        return {"version":AUDIT_VERSION,"status":"not_applicable","errors":[],"publication_allowed":False}
    def attempt(name, fn, detail=None):
        check()
        try:
            value = fn()
            verified[name] += 1
            return value
        except Exception as exc:
            errors.append({"check":name,"detail":detail,"error":str(exc)})
            return None
    lineage = attempt("completed_l1_bootstrap_lineage", lambda: read_exit_bootstrap_lineage(
        root, calendar, config, bundles=report["model_bundles"],
        expected_source_snapshot_hash=report["source_snapshot"]["snapshot_hash"]))
    if lineage is None:
        return {"version":AUDIT_VERSION,"status":"fail","checks":dict(verified),"errors":errors,"publication_allowed":False}
    pipeline = report["l1_ma_target_pipeline"]
    folds = [split_validation(f, calendar) for f in _ma_folds(report["data_end"])]
    ma_bundles = _read(root / "models/ma_bundles.json")
    def pipeline_identity():
        _require(REQUIRED_ALIGNMENT_CODE <= {Path(p).name for p in report.get('protocol_freeze', {}).get('code_hashes', {})},
            'new target policy, training and audit dependencies were not frozen')
        _require(_read(root / 'reports/high_return.json') == report
            and _read(root / 'data_snapshots/calendar.json')['dates'] == calendar
            and ma_bundles == [b for b in report['model_bundles'] if set(b.get('heads', {})) == {'ma_entry','ma_exit'}]
            and pipeline.get("stage_order") == STAGE_ORDER and pipeline.get("status") == "completed"
            and pipeline.get("entry_heads_used_in_label_generation") == 0
            and pipeline.get("holding_checkpoint_upgrade") is False
            and pipeline.get("exit_heads") == pipeline.get("entry_heads") == len(folds)
            and [b["fold"] for b in ma_bundles] == folds
            and [b["fold"] for b in lineage["exit_only_bundles"]] == folds
            and all(set(b["heads"]) == {"ma_entry", "ma_exit"} and not b.get("failures") for b in ma_bundles),
            "L1 pipeline is not a complete dependency-ordered chronological head pair catalog")
        _hashed(root / pipeline["lineage_path"], pipeline["lineage_sha256"], root)
        _hashed(root / pipeline["artifact_manifest_path"], pipeline["artifact_manifest_sha256"], root)
    attempt("l1_pipeline_dependency_order_coverage", pipeline_identity)
    source_hash = report["source_snapshot"]["snapshot_hash"]
    expected_v0 = _expected_v0_exit_contract(config, calendar, source_hash)
    for stage, sources in (("bootstrap", ["ma"]), ("ma_entry_alignment", ["ma"]),
                           ("final", ["ma", "structure", "wyckoff", "union"])):
        manifest = attempt(stage + "_ordered_manifest", lambda stage=stage, sources=sources:
            _stage_manifest(root, stage, sources, report["dataset_records"]))
        if manifest is None:
            continue
        if stage == "ma_entry_alignment":
            attempt("alignment_report_coverage_identity", lambda: _require(
                pipeline["artifact_coverage"] == {k:v for k,v in manifest.items() if k != "records"},
                "alignment report coverage differs from the original stage manifest"))
        if stage == "final":
            attempt("final_report_record_identity", lambda: _require(report["episode_records"] == manifest["records"],
                "final report is not bound to its original four-path dataset records"))
        for record in manifest["records"]:
            symbol = record["symbol"]
            def audit_record():
                original = next(r for r in report["dataset_records"] if r["symbol"] == symbol)
                frame = _read_features(original, config, report["data_end"])
                points = _read(_hashed(record["points_path"], record["points_sha256"], root))
                raw_points = build_signal_points(frame, config)
                _require(record["source_rows"] == len(frame) and record["events"] == len(points["events"])
                    and record["candidates"] == len(points["candidates"]), "original points/feature row counts changed")
                if stage != "final":
                    _raw_population_identity(points, raw_points)
                if not record["tradable_board"]:
                    counts[stage + ":unsupported_cash_only"] += 1
                    return
                data = _frame_files(root, record)
                _require(list(data["contracts"]) == sources,
                    "data stage source-contract order differs from its four-path declaration")
                _unavailable_and_maturity(data)
                if stage == "bootstrap":
                    _bootstrap_original_contract(data, expected_v0, source_hash)
                else:
                    policy = make_signal_pinned_ma_exit_policy(frame, calendar, root,
                        lineage["exit_only_bundles"], config=config, points=points,
                        source_snapshot_hash=source_hash,
                        expected_exit_contract_hash=lineage["expected_exit_target_contract_hash"],
                        exit_training_calendar_hash=lineage["calendar_hash"],
                        exit_training_calendar_dates=lineage['_frozen_calendar_dates'])
                    recipe = _read(_hashed(record["ma_exit_alignment_recipe_path"],
                        record["ma_exit_alignment_recipe_sha256"], root))
                    saved = _read(_hashed(record["ma_candidate_exit_bindings_path"],
                        record["ma_candidate_exit_bindings_sha256"], root))
                    _require(recipe == policy.recipe and saved == policy.bindings,
                        "saved recipe/catalog/candidate bindings differ from a fresh shared pinned factory")
                    counts.update({stage + ":" + k:v for k,v in _annotated_rows(data, policy, calendar, symbol).items()})
                    if stage == "ma_entry_alignment":
                        _require(data["contracts"]["ma"]["ma_entry"].get("candidate_policy") == "broad_causal_ma_rule_no_entry_head"
                            and data["contracts"]["ma"]["ma_entry"] == pipeline["entry_target_contract"],
                            "new MA-entry model borrowed an old or model-conditioned target")
                    else:
                        _require(all(not contract.get("alignment_version") and not contract.get("exit_alignment_recipe_hash")
                            for source, heads in data["contracts"].items() if source != "ma" for contract in heads.values()),
                            "non-MA continuation targets were silently changed by MA alignment")
                if stage in {"bootstrap", "ma_entry_alignment"}:
                    head, kind = ("ma_exit", "exit") if stage == "bootstrap" else ("ma_entry", "entry")
                    training = training_frame(data, source="ma", head=kind, ma_only=True)
                    base = data["contracts"]["ma"][head]
                    totals, derived = _fold_counts(training, base, head, folds)
                    if head in model_contracts:
                        _require(model_contracts[head] == derived, "L1 fitting inputs mix distinct target contracts")
                    model_contracts[head] = derived
                    for fold_name, values in totals.items():
                        accumulated = fold_totals[head].setdefault(fold_name, {"all":0,"train":0,"calibration":0,"policy":0})
                        for key, value in values.items():
                            if key.endswith("_label_end"):
                                if value is not None:
                                    accumulated[key] = max(accumulated.get(key, ""), value)
                            elif key.endswith("_binding_modes"):
                                accumulated.setdefault(key, Counter()).update(value)
                            else:
                                accumulated[key] += value
                counts[stage + ":supported_records"] += 1
            attempt(stage + "_original_record", audit_record, symbol)
    for bundle in ma_bundles:
        for head, binding in bundle["heads"].items():
            def audit_fit():
                path = _inside(root / binding["path"], root)
                manifest = load_head(path)
                counts_for_fold = fold_totals[head][bundle["name"]]
                _require(manifest["head"] == head and manifest["fold"] == bundle["fold"]
                    and manifest["available_at"] == bundle["available_at"]
                    and manifest["manifest_sha256"] == binding["manifest_sha256"]
                    and manifest["target_contract"] == model_contracts[head]
                    and manifest["target_contract_hash"] == stable_hash(model_contracts[head]) == binding["target_contract_hash"]
                    and manifest["sample_counts"] == {k:counts_for_fold[k] for k in ("train", "calibration", "policy", "all")}
                    and manifest["train_label_end"] == counts_for_fold.get("train_label_end")
                    and manifest["calibration_label_end"] == counts_for_fold.get("calibration_label_end"),
                    "L1 head source/target/fold/mature whole-group sample counts differ from original stage inputs")
                dep = bundle["l1_target_dependency"]
                _require(dep["entry_target_contract_hash"] == stable_hash(model_contracts["ma_entry"])
                    and dep["exit_bootstrap_target_contract_hash"] == lineage["expected_exit_target_contract_hash"]
                    and dep["label_recipe_hash"] == pipeline["entry_target_contract"]["exit_alignment_recipe_hash"]
                    and dep["label_exit_routing"] == "strictly_prior_inner_oof_at_candidate_close"
                    and dep["holding_checkpoint_upgrade"] is False,
                    "L1 pair dependency metadata differs from its actual distinct fitting contracts")
            attempt("l1_head_original_masks_and_dependency", audit_fit, {"fold":bundle["name"],"head":head})
    return {"version":AUDIT_VERSION,"status":"fail" if errors else "pass", "checks":dict(verified),
        "errors":errors,"stage_counts":dict(counts),"fold_mature_inputs":fold_totals,
        "numeric_preprocess_or_platt_refit":False,"publication_allowed":False,
        "qualification":"historical_dependency_and_original_masks_audit_not_independent_return_qualification"}
