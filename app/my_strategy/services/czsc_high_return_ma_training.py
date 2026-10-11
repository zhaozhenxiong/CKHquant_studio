"""Dependency-ordered L1 exit and exit-bound entry fitting.

The checkpoint serialization/schema remains the existing high-return family.
The new research/target version is separately frozen in configuration/contracts.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from my_strategy.core.run_context import stable_hash
from my_strategy.services.czsc_high_return_features import L1_MA_ENTRY_COLUMNS, L1_MA_EXIT_COLUMNS
from my_strategy.services.czsc_high_return_models import split_validation, train_head
from my_strategy.services.czsc_research import _file_hash, _save

ALIGNMENT_MODE = "signal_pinned_learned_exit_v2"
RESEARCH_VERSION = "czsc_high_return_v2_ma_exit_bound"


def validate_alignment_version(config):
    version, alignment = config.get("version", "czsc_high_return_v1"), config.get("ma_entry_alignment")
    if (version == "czsc_high_return_v1" and alignment is None
            or version == RESEARCH_VERSION and alignment == ALIGNMENT_MODE):
        return
    raise ValueError("high-return research version and MA target dependency are inconsistent")


def _expected_v0_exit_contract(config, calendar, source_snapshot_hash):
    from my_strategy.services.czsc_high_return_features import MA_EXIT_SCHEMA_HASH
    from my_strategy.services.czsc_high_return_labels import label_config, target_contract
    from my_strategy.services.czsc_signal_points import signal_point_parameters
    from my_strategy.services.czsc_trade_episodes import EPISODE_VERSION, _settings
    settings = _settings(config)
    policy_hash = stable_hash({"version": EPISODE_VERSION, "source": "ma",
        "points_config": stable_hash(signal_point_parameters(config)), "strategy": settings,
        "protocol": label_config(config), "exit_policy": "source_rule_exit_v0"})
    return target_contract(head="exit", source="ma", config=config, market_dates=calendar,
        source_snapshot_hash=source_snapshot_hash, feature_schema_hash=MA_EXIT_SCHEMA_HASH,
        continuation_policy_hash=policy_hash, initial_cash=config["initial_cash"],
        strategy_settings=settings)


def read_exit_bootstrap_lineage(model_root, calendar, config, *, bundles=None,
                               expected_source_snapshot_hash=None, allow_training=False):
    """Read the original exit-only catalog, including during future replay.

Later verified calendars may extend the unchanged training calendar. A later
feature snapshot is an execution input and cannot replace the exit training
snapshot recorded here. Training stages supply an explicit expected snapshot.
    """
    from my_strategy.services.czsc_high_return_runtime import _validate_compatibility
    root = Path(model_root).resolve()
    path = root / "models/ma_exit_bootstrap_lineage.json"
    lineage = json.loads(path.read_text())
    if lineage.get("version") != ALIGNMENT_MODE or lineage.get("lineage_hash") != stable_hash(
            {k:v for k,v in lineage.items() if k != "lineage_hash"}):
        raise ValueError("frozen MA exit bootstrap lineage identity changed")
    contract = lineage["bootstrap_exit_model_contract"]
    if (contract.get("contract_hash") != stable_hash({k:v for k,v in contract.items() if k != "contract_hash"})
            or stable_hash(contract) != lineage["expected_exit_target_contract_hash"]
            or contract.get("source_snapshot_hash") != lineage["source_snapshot_hash"]
            or contract.get("calendar_hash") != lineage["calendar_hash"]):
        raise ValueError("MA exit lineage differs from its bootstrap target")
    _validate_compatibility(contract, contract_config=config, source="ma", initial_cash=config["initial_cash"])
    if expected_source_snapshot_hash and lineage["source_snapshot_hash"] != expected_source_snapshot_hash:
        raise ValueError("MA alignment cannot borrow an exit head trained on another source snapshot")
    frozen_calendar = json.loads((root / "data_snapshots/calendar.json").read_text())
    frozen_dates = frozen_calendar["dates"]
    original_config = json.loads((root / "config/research.json").read_text())
    expected_base = _expected_v0_exit_contract(original_config, frozen_dates, lineage["source_snapshot_hash"])
    from my_strategy.services.czsc_high_return_research import _head_data
    _, expected_model = _head_data(pd.DataFrame(), expected_base,
        "ma_exit", L1_MA_EXIT_COLUMNS)
    if (lineage.get("bootstrap_exit_base_contract") != expected_base
            or lineage.get("bootstrap_exit_model_contract") != expected_model
            or lineage["expected_exit_target_contract_hash"] != stable_hash(expected_model)):
        raise ValueError("exit lineage is not the canonical raw-MA rule-v0 bootstrap target")
    dates = [str(day) for day in calendar]
    if (frozen_calendar.get("verified") is not True or stable_hash(frozen_dates) != lineage["calendar_hash"]
            or dates != sorted(set(dates)) or dates[:len(frozen_dates)] != frozen_dates
            or frozen_calendar["source"]["snapshot_hash"] != lineage["source_snapshot_hash"]):
        raise ValueError("MA exit training calendar/source or verified execution prefix changed")
    bootstrap_path = (root / lineage["bootstrap_manifest_path"]).resolve()
    if not bootstrap_path.is_relative_to(root) or _file_hash(bootstrap_path) != lineage["bootstrap_manifest_sha256"]:
        raise ValueError("MA exit bootstrap manifest changed")
    bootstrap = json.loads(bootstrap_path.read_text())
    original_snapshot = json.loads((root / 'data_snapshots/source.json').read_text())
    original_records = json.loads((root / 'data_snapshots/source_records.json').read_text())
    if (bootstrap.get('failures') or bootstrap.get('sources') != ['ma']
            or bootstrap.get('requested') != len(original_records)
            or bootstrap.get('successful') != len(original_records)
            or sorted(r['symbol'] for r in bootstrap.get('records', [])) != sorted(r['symbol'] for r in original_records)
            or original_snapshot.get('snapshot_hash') != lineage['source_snapshot_hash']
            or original_snapshot.get('records_hash') != stable_hash(original_records)
            or original_snapshot.get('snapshot_hash') != stable_hash({k:v for k,v in original_snapshot.items()
                if k not in {'snapshot_hash','calendar_run_id'}})):
        raise ValueError('MA exit bootstrap coverage is not the frozen original source universe')
    canonical = lineage["exit_only_bundles"]
    from my_strategy.services.czsc_high_return_research import _ma_folds
    expected_folds = [split_validation(f, frozen_dates) for f in _ma_folds(original_snapshot['effective_data_end'])]
    if (stable_hash(canonical) != lineage["exit_only_bundles_hash"]
            or any(set(b.get("heads", {})) != {"ma_exit"} for b in canonical)
            or [b.get('fold') for b in canonical] != expected_folds
            or any(b.get('name') != f['name'] or b.get('available_at') != f['validation_end']
                for b,f in zip(canonical, expected_folds, strict=True))
            or lineage.get("entry_heads_used_in_label_generation") != 0
            or lineage.get("holding_checkpoint_upgrade") is not False):
        raise ValueError("MA exit lineage is not an exit-only immutable catalog")
    if bundles is not None:
        supplied = {b["name"]: b for b in bundles if b.get("heads", {}).get("ma_exit")}
        for item in canonical:
            actual = supplied.get(item["name"])
            if (actual is None or actual.get("available_at") != item["available_at"]
                    or actual.get("fold") != item["fold"]
                    or actual["heads"]["ma_exit"] != item["heads"]["ma_exit"]
                    or actual.get("policy", {}).get("ma_exit_threshold") != item["policy"]["ma_exit_threshold"]):
                raise ValueError("runtime MA exit catalog differs from the frozen bootstrap checkpoints")
    report_path = root / "reports/high_return.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        metadata = json.loads((root / 'metadata.json').read_text())
        pipeline = report.get("l1_ma_target_pipeline", {})
        freeze = report.get("protocol_freeze", {})
        freeze_path = root / "validation/protocol_freeze.json"
        frozen_file = json.loads(freeze_path.read_text())
        if (report.get("status") != "completed" or metadata.get('status') != 'complete'
                or metadata.get('model_family') != 'high_return' or frozen_file != freeze
                or freeze.get("freeze_hash") != stable_hash({k:v for k,v in freeze.items() if k != "freeze_hash"})
                or freeze.get("config_hash") != stable_hash(original_config)
                or report.get("config") != original_config
                or freeze.get("source_snapshot_hash") != lineage["source_snapshot_hash"]
                or report.get("source_snapshot", {}).get("snapshot_hash") != lineage["source_snapshot_hash"]
                or freeze.get("model_bundles_hash") != stable_hash(report.get("model_bundles", []))
                or pipeline.get("lineage_sha256") != _file_hash(path)
                or freeze.get("l1_ma_target_pipeline_hash") != stable_hash(pipeline)
                or freeze.get("l1_exit_lineage_sha256") != _file_hash(path)):
            raise ValueError("completed MA exit lineage differs from the actual protocol freeze")
    elif not allow_training:
        raise ValueError("runtime MA exit lineage requires a completed actual protocol freeze")
    else:
        from my_strategy.services.czsc_high_return_research import _code_files
        initial = json.loads((root / "config/initial_code_hashes.json").read_text())
        metadata = json.loads((root / "metadata.json").read_text())
        if (initial != {str(p):_file_hash(p) for p in _code_files()}
                or metadata.get("task") != "czsc-high-return-research-train"
                or metadata.get("scope") != "train"
                or metadata.get("config_hash") != stable_hash(original_config)
                or metadata.get("data_version") != lineage["source_snapshot_hash"]):
            raise ValueError("in-progress MA policy lacks the current immutable training context")
    lineage['_frozen_calendar_dates'] = frozen_dates
    return lineage


def _train_ma_head(data, base_contract, raw_fold, context, config, *, head, calendar,
                   device, check_cancel, progress):
    from my_strategy.services.czsc_high_return_research import _head_data
    if head not in {"ma_entry", "ma_exit"}:
        raise ValueError("dependency-ordered L1 requires an explicit single MA head")
    columns = L1_MA_ENTRY_COLUMNS if head == "ma_entry" else L1_MA_EXIT_COLUMNS
    fold = split_validation(raw_fold, calendar)
    frame, contract = _head_data(data, base_contract, head, columns)
    path = context.subdir("models", "l1", fold["name"], "linear", head)
    manifest = train_head(frame, columns, path, fold, config, head=head,
        architecture="linear", device=device, contract=contract, check_cancel=check_cancel)
    binding = {"path": str(path.relative_to(context.run_dir())),
        "manifest_sha256": manifest["manifest_sha256"],
        "target_contract_hash": manifest["target_contract_hash"],
        "actual_device": manifest["actual_device"],
        "training_batches": manifest["training_batches"],
        "sample_counts": manifest["sample_counts"]}
    if progress:
        progress("train_l1_" + fold["name"] + "_linear_" + head, 1, 1, 0)
    return {"name": fold["name"], "available_at": fold["validation_end"], "fold": fold,
        "architecture": "linear", "heads": {head: binding},
        "policy": {"entry_threshold": config.get("default_entry_threshold", .1),
            "exit_threshold": .55,
            "ma_exit_threshold": config["signal_points"]["ma_exit_threshold"],
            "max_holding_sessions": config["high_return"]["max_holding_sessions"]},
        "publication_allowed": False, "historical_reconstruction": True, "failures": []}


def train_exit_bound_ma(records, calendar, snapshot, end, bootstrap, context, config,
                        *, device, cpu_workers, progress=None, check_cancel=None):
    """Bootstrap exit -> prior-exit targets -> entry; do not fit bootstrap entry.

Each derived target uses a checkpoint already historically available at its
candidate date, independently of the outer entry training fold. Consequently
the outer entry fold cannot score earlier labels with its own later exit head.
Candidates before the first available exit head explicitly retain rule-v0
continuation. Their counts must remain visible in the new label-stage audit.
    """
    from my_strategy.services.czsc_high_return_research import _load_training, _head_data, _ma_folds, _build
    check = check_cancel or (lambda: None)
    if config.get("ma_entry_alignment") != ALIGNMENT_MODE:
        raise ValueError("new L1 stages require the frozen alignment protocol")
    exit_data, exit_contract = _load_training(bootstrap, "ma", "exit", True,
        source_snapshot_hash=snapshot["snapshot_hash"])
    if exit_contract != _expected_v0_exit_contract(config, calendar, snapshot["snapshot_hash"]):
        raise ValueError("MA exit training inputs do not carry the canonical raw-MA rule-v0 contract")
    _, derived_exit_contract = _head_data(exit_data, exit_contract, "ma_exit", L1_MA_EXIT_COLUMNS)
    exit_bundles = []
    folds = _ma_folds(end)
    for raw_fold in folds:
        check()
        exit_bundles.append(_train_ma_head(exit_data, exit_contract, raw_fold, context, config,
            head="ma_exit", calendar=calendar, device=device, check_cancel=check, progress=progress))
    del exit_data
    lineage = {"version": ALIGNMENT_MODE, "bootstrap_candidate_policy": "causal_rule_v0",
        "bootstrap_manifest_path": "dataset/bootstrap/manifest.json",
        "bootstrap_manifest_sha256": _file_hash(context.run_dir() / "dataset/bootstrap/manifest.json"),
        "source_snapshot_hash": snapshot["snapshot_hash"], "calendar_hash": stable_hash(calendar),
        "bootstrap_exit_base_contract": exit_contract,
        "bootstrap_exit_model_contract": derived_exit_contract,
        "expected_exit_target_contract_hash": stable_hash(derived_exit_contract),
        "exit_only_bundles": exit_bundles,
        "exit_only_bundles_hash": stable_hash(exit_bundles),
        "entry_heads_used_in_label_generation": 0,
        "holding_checkpoint_upgrade": False, "publication_allowed": False}
    lineage["lineage_hash"] = stable_hash(lineage)
    lineage_path = context.subdir("models") / "ma_exit_bootstrap_lineage.json"
    _save(lineage_path, lineage)
    _save(context.subdir("models") / "ma_exit_bundles.json", exit_bundles)
    artifacts = _build(records, calendar, config, end, context, "ma_entry_alignment",
        sources=("ma",), bundles=exit_bundles, subdirectory="ma_entry_alignment",
        cpu_workers=cpu_workers, snapshot_hash=snapshot["snapshot_hash"],
        progress=progress, check_cancel=check)
    if artifacts["failures"] or artifacts["successful"] != len(records):
        raise RuntimeError("exit-bound MA entry label stage is incomplete")
    entry_data, entry_contract = _load_training(artifacts, "ma", "entry", True,
        source_snapshot_hash=snapshot["snapshot_hash"])
    if not entry_contract.get("exit_alignment_recipe_hash"):
        raise ValueError("MA entry target lacks its frozen learned-exit dependency")
    bundles = []
    for raw_fold, exit_bundle in zip(folds, exit_bundles, strict=True):
        check()
        entry_bundle = _train_ma_head(entry_data, entry_contract, raw_fold, context, config,
            head="ma_entry", calendar=calendar, device=device, check_cancel=check, progress=progress)
        if entry_bundle["fold"] != exit_bundle["fold"]:
            raise ValueError("MA head fold identity changed between dependency stages")
        bundle = {**exit_bundle, "heads": {**exit_bundle["heads"], **entry_bundle["heads"]}}
        bundle["l1_target_dependency"] = {
            "version": ALIGNMENT_MODE,
            "entry_target_contract_hash": entry_bundle["heads"]["ma_entry"]["target_contract_hash"],
            "exit_bootstrap_target_contract_hash": lineage["expected_exit_target_contract_hash"],
            "label_recipe_hash": entry_contract["exit_alignment_recipe_hash"],
            "label_exit_routing": "strictly_prior_inner_oof_at_candidate_close",
            "holding_checkpoint_upgrade": False}
        bundles.append(bundle)
    del entry_data
    manifest_path = context.run_dir() / "dataset/ma_entry_alignment/manifest.json"
    return bundles, {"version": ALIGNMENT_MODE, "status": "completed",
        "stage_order": ["rule_v0_bootstrap", "train_only_ma_exit", "prior_exit_only_entry_targets", "train_only_ma_entry"],
        "exit_heads": len(exit_bundles), "entry_heads": len(bundles),
        "entry_heads_used_in_label_generation": 0,
        "lineage_path": str(lineage_path.relative_to(context.run_dir())),
        "lineage_sha256": _file_hash(lineage_path),
        "artifact_manifest_path": str(manifest_path.relative_to(context.run_dir())),
        "artifact_manifest_sha256": _file_hash(manifest_path),
        "artifact_coverage": {k:v for k,v in artifacts.items() if k != "records"},
        "entry_target_contract": entry_contract,
        "holding_checkpoint_upgrade": False,
        "qualification": "historical_development_reconstruction_not_production_availability"}
