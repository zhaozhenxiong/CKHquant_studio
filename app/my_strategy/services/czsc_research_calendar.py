"""Verified execution schedules can append sessions without rewriting model calendars."""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.core.run_context import stable_hash


def _valid_sessions(dates):
    if not isinstance(dates, list) or not dates:
        return False
    try:
        return (all(isinstance(day, str) and date.fromisoformat(day).isoformat() == day for day in dates)
                and dates == sorted(set(dates)))
    except (TypeError, ValueError):
        return False


def calendar_prefix_compatible(frozen_dates, runtime_dates):
    """Allow equality or appending; reject historical deletion, revision or insertion."""
    return (_valid_sessions(frozen_dates) and _valid_sessions(runtime_dates)
            and len(frozen_dates) <= len(runtime_dates)
            and runtime_dates[:len(frozen_dates)] == frozen_dates)


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_verified_calendar(calendar_run_id, *, runs_root=None, require_sources=False, _seen=None):
    root = Path(runs_root or ARTIFACT_RUNS_ROOT).resolve()
    if not isinstance(calendar_run_id, str) or Path(calendar_run_id).name != calendar_run_id or calendar_run_id in {"", ".", ".."}:
        raise ValueError("invalid verified calendar run identity")
    seen = set(_seen or ())
    if calendar_run_id in seen:
        raise ValueError("verified calendar parent cycle")
    seen.add(calendar_run_id)
    path = root / calendar_run_id / "reports" / "calendar.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    dates = value.get("dates")
    if (value.get("verified") is not True or not isinstance(value.get("source"), str)
            or not value["source"].strip() or not _valid_sessions(dates)):
        raise ValueError("研究需要有序、唯一且已核验的独立交易日历")
    from my_strategy.services.czsc_trend_support import resolve_frozen_asset
    sources = value.get("source_artifacts", [])
    if require_sources and (value.get("independent_from_market_bars") is not True or not sources):
        raise ValueError("执行日历扩展需要独立来源及冻结来源哈希")
    for artifact in sources:
        if _file_hash(resolve_frozen_asset(artifact["path"], runs_root=root)) != artifact["sha256"]:
            raise ValueError("verified calendar source artifact hash mismatch")
    if value.get("artifact") and value.get("sha256") and _file_hash(resolve_frozen_asset(value["artifact"], runs_root=root)) != value["sha256"]:
        raise ValueError("verified calendar date artifact hash mismatch")
    ancestors = []
    parent = value.get("parent_calendar")
    if parent:
        parent_dates, parent_meta = load_verified_calendar(parent["run_id"], runs_root=root, _seen=seen)
        if (resolve_frozen_asset(parent["path"], runs_root=root) != Path(parent_meta["path"]).resolve()
                or parent["file_sha256"] != parent_meta["file_sha256"]
                or parent["hash"] != parent_meta["hash"]
                or not calendar_prefix_compatible(parent_dates, dates)):
            raise ValueError("verified calendar parent identity or historical prefix mismatch")
        ancestors = [parent, *parent_meta.get("ancestors", [])]
    metadata = {"run_id": calendar_run_id, "source": value["source"], "hash": stable_hash(value),
                "path": str(path), "file_sha256": _file_hash(path), "dates_hash": stable_hash(dates),
                "first_session": dates[0], "last_session": dates[-1], "session_count": len(dates),
                "independent_from_market_bars": value.get("independent_from_market_bars", False),
                "source_artifacts": sources, "ancestors": ancestors,
                "observed_market_data_end": value.get("observed_market_data_end"),
                "executable_verified_range_end": value.get("executable_verified_range_end"),
                "schedule_verified_range_end": value.get("schedule_verified_range_end", dates[-1])}
    return list(dates), metadata


def resolve_execution_calendar(calendar_run_id, *, runs_root=None, requested_calendar=None):
    """Upgrade a selected frozen schedule only along a verified append-only lineage."""
    root = Path(runs_root or ARTIFACT_RUNS_ROOT)
    dates, metadata = requested_calendar or load_verified_calendar(calendar_run_id, runs_root=root)
    requested = {key: metadata[key] for key in ("run_id", "source", "hash", "path", "file_sha256") if key in metadata}
    selected_dates, selected = dates, metadata
    rejected = []
    for path in sorted(root.glob("*/reports/calendar.json")):
        run_id = path.parent.parent.name
        if run_id == calendar_run_id:
            continue
        try:
            candidate_dates, candidate = load_verified_calendar(run_id, runs_root=root, require_sources=True)
            belongs = any(parent["run_id"] == calendar_run_id and parent["hash"] == metadata.get("hash")
                          for parent in candidate["ancestors"])
            if (belongs and candidate["source"] == metadata.get("source")
                    and calendar_prefix_compatible(dates, candidate_dates)
                    and len(candidate_dates) > len(selected_dates)):
                selected_dates, selected = candidate_dates, candidate
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append({"run_id": run_id, "reason": str(exc)})
    return selected_dates, {**selected, "requested_calendar": requested,
        "extension_policy": "verified_parent_lineage_strict_date_prefix",
        "extended_from_requested": selected.get("run_id") != calendar_run_id,
        "extension_rejections": rejected}
