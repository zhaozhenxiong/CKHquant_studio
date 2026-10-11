"""Execution calendars append verified sessions while training identities stay frozen."""
import hashlib
import json

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_research_calendar as calendar
from my_strategy.services.czsc_research import _calendar_quality


def _write_calendar(root, run_id, dates, *, parent=None):
    path = root / run_id / "reports" / "calendar.json"
    path.parent.mkdir(parents=True)
    source = root / run_id / "calendar-source.txt"
    source.write_text("independent published exchange schedule", encoding="utf-8")
    value = {"verified": True, "source": "exchange:published-calendar", "dates": dates,
             "independent_from_market_bars": True,
             "source_artifacts": [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}
    if parent:
        _, metadata = calendar.load_verified_calendar(parent, runs_root=root)
        value["parent_calendar"] = {key: metadata[key] for key in ("run_id", "path", "file_sha256", "hash")}
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("candidate", [
    ["2026-10-08", "2026-10-09"],
    ["2026-09-29", "2026-09-30", "2026-10-08", "2026-10-09"],
    ["2026-09-30", "2026-10-07", "2026-10-08", "2026-10-09"],
    ["2026-09-30", "2026-10-08", "2026-10-08"],
    ["2026-10-08", "2026-09-30", "2026-10-09"],
    ["20260930", "20261008"],
])
def test_calendar_compatibility_rejects_historical_revision(candidate):
    assert not calendar.calendar_prefix_compatible(["2026-09-30", "2026-10-08"], candidate)


def test_selected_old_calendar_uses_verified_extension_without_changing_original(tmp_path):
    original_dates = ["2026-09-30", "2026-10-08"]
    old_path = _write_calendar(tmp_path, "old", original_dates)
    original_bytes = old_path.read_bytes()
    _write_calendar(tmp_path, "extension", [*original_dates, "2026-10-09", "2026-10-12"], parent="old")
    dates, metadata = calendar.resolve_execution_calendar("old", runs_root=tmp_path)
    assert dates == [*original_dates, "2026-10-09", "2026-10-12"]
    assert metadata["run_id"] == "extension" and metadata["extended_from_requested"]
    assert metadata["requested_calendar"]["run_id"] == "old"
    assert metadata["requested_calendar"]["hash"] == stable_hash(json.loads(original_bytes))
    assert old_path.read_bytes() == original_bytes
    assert calendar.load_verified_calendar("old", runs_root=tmp_path)[0] == original_dates


def test_unrelated_or_corrupt_extensions_are_not_selected(tmp_path):
    original_dates = ["2026-09-30", "2026-10-08"]
    _write_calendar(tmp_path, "old", original_dates)
    _write_calendar(tmp_path, "unrelated", [*original_dates, "2026-10-09"])
    corrupt = _write_calendar(tmp_path, "corrupt", [*original_dates, "2026-10-09"], parent="old")
    payload = json.loads(corrupt.read_text())
    payload["dates"].insert(1, "2026-10-07")
    corrupt.write_text(json.dumps(payload))
    dates, metadata = calendar.resolve_execution_calendar("old", runs_root=tmp_path)
    assert dates == original_dates and metadata["run_id"] == "old"
    assert any(item["run_id"] == "corrupt" for item in metadata["extension_rejections"])
    (tmp_path / "old" / "calendar-source.txt").write_text("changed")
    with pytest.raises(ValueError, match="source artifact hash mismatch"):
        calendar.load_verified_calendar("old", runs_root=tmp_path)


def test_calendar_extension_restores_new_session_but_retains_actual_missing_bar_veto():
    dates = pd.bdate_range(end="2026-10-09", periods=125).strftime("%Y-%m-%d").tolist()
    features = pd.DataFrame({"input_eligible": True, "model_input_eligible": True,
        "reason_codes": [[] for _ in dates], "model_reason_codes": [[] for _ in dates]})
    for key in ("rule_czsc_buy", "rule_czsc_sell", "rule_price_volume_buy", "rule_price_volume_sell", "rule_buy", "rule_sell"):
        features[key] = True
    features.attrs.update(quality_window_bars=120, model_quality_window_bars=120)
    raw = pd.DataFrame({"date": dates})
    old = _calendar_quality(features, raw, dates[:-1])
    restored = _calendar_quality(features, raw, dates)
    assert not old.iloc[-1]["model_input_eligible"]
    assert restored.iloc[-1]["input_eligible"] and restored.iloc[-1]["model_input_eligible"]
    missing = _calendar_quality(features.drop(index=61).reset_index(drop=True), raw.drop(index=61).reset_index(drop=True), dates)
    assert not missing.iloc[-1]["input_eligible"] and not missing.iloc[-1]["model_input_eligible"]
    assert "calendar_missing_market_bar_window" in missing.iloc[-1]["reason_codes"]


def test_frozen_cache_with_old_calendar_veto_is_rebuilt_without_overwriting_cache(tmp_path, monkeypatch):
    from my_strategy.adapters.czsc_adapter import SOURCE_SHA256
    from my_strategy.services import czsc_research_features, czsc_research_profiles, czsc_research_runtime
    raw = pd.DataFrame({"symbol": ["000001.SZ"], "date": ["2026-10-09"], "open": [10.],
        "high": [11.], "low": [9.], "close": [10.], "volume": [100.], "amount": [1000.],
        "source": ["tushare"], "has_trade_price": [True]})
    raw.attrs["data_version"] = "immutable-raw"
    base = raw.copy().assign(x=1., input_eligible=True)
    base["reason_codes"] = [[]]
    for key in ("rule_czsc_buy", "rule_czsc_sell", "rule_price_volume_buy", "rule_price_volume_sell", "rule_buy", "rule_sell"):
        base[key] = True
    base.attrs.update(feature_version=czsc_research_features.FEATURE_VERSION,
                      schema_hash=czsc_research_features.FEATURE_SCHEMA_HASH,
                      source_sha256=SOURCE_SHA256, config_hash=stable_hash({}))
    stale = _calendar_quality(base, raw, ["2026-10-08"])
    path = tmp_path / "frozen.parquet"
    stale.to_parquet(path, index=False)
    before = path.read_bytes()
    monkeypatch.setattr(czsc_research_runtime, "load_bars", lambda *args, **kwargs: raw)
    monkeypatch.setattr(czsc_research_runtime, "strategy_config", lambda: {})
    monkeypatch.setattr(czsc_research_profiles, "get_feature_profile", lambda _: {"name": "legacy", "columns": ["x"], "schema_hash": czsc_research_features.FEATURE_SCHEMA_HASH})
    monkeypatch.setattr(czsc_research_profiles, "bind_feature_profile", lambda value, _: value)
    monkeypatch.setattr(czsc_research_features, "build_features", lambda _: base.copy())
    result = czsc_research_runtime._prepare_stock("000001.SZ", "2026-10-09", ["2026-10-08", "2026-10-09"],
        {"path": str(path), "sha256": hashlib.sha256(before).hexdigest()})
    assert not result["cache_hit"] and result["features"].iloc[-1]["input_eligible"]
    assert result["features"].iloc[-1]["reason_codes"] == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("family", ["ma_trend", "dual", "wyckoff"])
def test_runtime_keeps_extended_execution_dates_when_resolver_returns_frozen_training_calendar(tmp_path, monkeypatch, family):
    from my_strategy.services import czsc_compute, czsc_research, czsc_research_models
    from my_strategy.services import czsc_dual_models, czsc_wyckoff_models, czsc_research_ml
    from my_strategy.services.czsc_research_runtime import ResearchRuntime
    old_dates = ["2026-09-30", "2026-10-08"]
    new_dates = [*old_dates, "2026-10-09", "2026-10-12"]
    _write_calendar(tmp_path, "old", old_dates)
    _write_calendar(tmp_path, "new", new_dates, parent="old")
    monkeypatch.setattr(calendar, "ARTIFACT_RUNS_ROOT", tmp_path)
    monkeypatch.setattr(czsc_research, "ARTIFACT_RUNS_ROOT", tmp_path)
    monkeypatch.setattr(czsc_research, "research_status", lambda: {"calendar_run_id": "old", "models": []})
    monkeypatch.setattr(czsc_compute, "compute_status", lambda _: {"available": True, "selected_device": "cpu"})
    captured = {}
    class Resolver:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.market_dates, self.calendar = old_dates, {"run_id": "old", "hash": "frozen"}
            self.cache = {}
            self.catalog = [{"calendar_run_id": "old", "calendar_hash": "frozen"}]
    class Predictor:
        def __init__(self, **kwargs):
            self.diagnostics = {"device": "cpu", "rows": 0, "batches": 0}
    monkeypatch.setattr(czsc_research_models, "ModelResolver", Resolver)
    monkeypatch.setattr(czsc_dual_models, "DualResolver", Resolver)
    monkeypatch.setattr(czsc_wyckoff_models, "WyckoffResolver", Resolver)
    monkeypatch.setattr(czsc_research_ml, "PredictorSession", Predictor)
    monkeypatch.setattr(czsc_dual_models, "DualPredictorSession", Predictor)
    monkeypatch.setattr(czsc_wyckoff_models, "WyckoffPredictorSession", Predictor)
    worker = ResearchRuntime(end="2026-10-09", calendar_run_id="old", model_family=family, device="cpu")
    assert worker.market_dates == new_dates and worker.calendar["run_id"] == "new"
    assert captured["calendar_run_id"] == "new" and captured["runtime_calendar_dates"] == new_dates
    assert worker.resolver.market_dates == old_dates
    info = worker.compute_info()
    assert info["runtime_execution_calendar"]["run_id"] == "new"
    assert info["model_training_calendars"] == [{"run_id": "old", "hash": "frozen"}]
