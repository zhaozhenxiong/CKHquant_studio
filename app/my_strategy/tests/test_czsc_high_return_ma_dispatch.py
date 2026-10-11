"""Cross-module dispatch contract regressions. All fit/evaluation calls are fakes.

These tests run canonical dispatch functions, not a real research training job.
They write only pytest tmp_path files; pytest restores every function mock.
"""
from __future__ import annotations

import copy

import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash

@pytest.fixture
def integrated(monkeypatch, tmp_path):
    # Canonical imports run after public integration. No staging overlay is used.
    from my_strategy.services import (
        czsc_high_return_research as research,
        czsc_high_return_ma_training as training,
        czsc_high_return_runtime as runtime,
        czsc_high_return_evaluation as evaluation,
    )
    from my_strategy.core import config_loader
    modules = {"czsc_high_return_research": research,
               "czsc_high_return_ma_training": training,
               "czsc_high_return_runtime": runtime,
               "czsc_high_return_evaluation": evaluation}
    cfg = research.research_config()
    monkeypatch.setattr(config_loader, "load_config", lambda name: copy.deepcopy(cfg))
    return modules, cfg, tmp_path

def test_actual_default_dispatch_uses_new_version_and_explicit_v1_stays_old(integrated):
    modules, cfg, _ = integrated
    research, runtime = (modules[name] for name in ("czsc_high_return_research", "czsc_high_return_runtime"))
    assert research.research_config()["ma_entry_alignment"] == runtime._config()["ma_entry_alignment"]
    assert research.research_config()["version"] == "czsc_high_return_v2_ma_exit_bound"
    for value in (research.research_config({"version": "czsc_high_return_v1"}),
                  runtime._config({"version": "czsc_high_return_v1"})):
        assert value["version"] == "czsc_high_return_v1"
        assert "ma_entry_alignment" not in value
    for value in ({"version": "czsc_high_return_v1", "ma_entry_alignment": cfg["ma_entry_alignment"]},
                  {"version": cfg["version"], "ma_entry_alignment": None}):
        with pytest.raises(ValueError, match="version and MA target dependency"):
            research.research_config(value)
        with pytest.raises(ValueError, match="version and MA target dependency"):
            runtime._config(value)


@pytest.mark.parametrize("cache", ["bootstrap_artifact_run_id", "ma_checkpoint_run_id", "final_artifact_run_id"])
def test_top_level_rejects_each_old_cache_before_source_context_or_fit(integrated, monkeypatch, cache):
    modules, _, _ = integrated
    research = modules["czsc_high_return_research"]
    def forbidden(*args, **kwargs):
        pytest.fail("old cache rejection must precede source/context/fit")
    monkeypatch.setattr(research, "_source", forbidden)
    monkeypatch.setattr(research, "create_run_context", forbidden)
    monkeypatch.setattr(modules["czsc_high_return_ma_training"], "_train_ma_head", forbidden)
    with pytest.raises(ValueError, match="requires fresh bootstrap"):
        research.train_high_return_research(**{cache: "an-old-v1-run"})


def test_real_top_level_calls_exit_alignment_entry_in_order_without_model_fit(integrated, monkeypatch):
    modules, original_cfg, tmp_path = integrated
    research, training, evaluation = (modules[name] for name in
        ("czsc_high_return_research", "czsc_high_return_ma_training", "czsc_high_return_evaluation"))
    cfg = research.research_config({"development_windows": []})
    dates = pd.bdate_range("2023-01-02", "2026-10-08").strftime("%Y-%m-%d").tolist()
    records = [{"symbol": "000001.SZ"}]
    snapshot = {"snapshot_hash": "synthetic-frozen-input", "calendar_run_id": "synthetic-calendar"}
    events = []
    class Context:
        run_id = "synthetic-dispatch-only-no-fit"
        def run_dir(self): return tmp_path / "run"
        def subdir(self, *parts):
            path = self.run_dir().joinpath(*parts); path.mkdir(parents=True, exist_ok=True)
            return path
    context = Context()
    monkeypatch.setattr(research, "_source", lambda *args: (records, dates, snapshot, dates[-1]))
    monkeypatch.setattr(research, "create_run_context", lambda **kwargs: context)
    monkeypatch.setattr(research, "_code_files", lambda: [])
    def build(recs, calendar, config, end, ctx, stage, **kwargs):
        subdir = kwargs["subdirectory"]
        events.append(("build", subdir))
        if subdir == "bootstrap":
            assert stage == "episodes" and kwargs["sources"] == ("ma",)
            assert not kwargs.get("bundles")
        elif subdir == "ma_entry_alignment":
            assert stage == "ma_entry_alignment" and kwargs["sources"] == ("ma",)
            assert len(kwargs["bundles"]) == 12
            assert all(set(b["heads"]) == {"ma_exit"} for b in kwargs["bundles"])
        else:
            assert subdir == "final" and stage == "episodes"
            assert all(set(b["heads"]) == {"ma_exit", "ma_entry"} for b in kwargs["bundles"])
        result = {"records": records, "requested": 1, "successful": 1, "failures": [],
                  "stage": stage, "sources": list(kwargs.get("sources", ("ma", "structure", "wyckoff", "union"))),
                  "synthetic_stage": subdir}
        research._save(ctx.subdir("dataset", subdir) / "manifest.json", result)
        return result
    def load(artifacts, source, head, ma_only=False, source_snapshot_hash=None):
        events.append(("load", artifacts["synthetic_stage"], head))
        base = training._expected_v0_exit_contract(cfg, dates, snapshot["snapshot_hash"])
        if artifacts["synthetic_stage"] == "bootstrap":
            assert source == "ma" and head == "exit" and ma_only
        elif artifacts["synthetic_stage"] == "ma_entry_alignment":
            assert source == "ma" and head == "entry" and ma_only
            base.update(target="actual_complete_fee_net_roundtrip_positive",
                        exit_alignment_recipe_hash="synthetic-prior-exit-recipe")
            base["contract_hash"] = stable_hash({k:v for k,v in base.items() if k != "contract_hash"})
        else:
            assert source == "union" and not ma_only
        return pd.DataFrame(), base
    def fit(data, contract, raw_fold, ctx, config, *, head, calendar, **kwargs):
        events.append(("fake_fit", head))
        fold = research.split_validation(raw_fold, calendar)
        _, derived = research._head_data(data, contract, head,
            training.L1_MA_EXIT_COLUMNS if head == "ma_exit" else training.L1_MA_ENTRY_COLUMNS)
        return {"name": fold["name"], "available_at": fold["validation_end"], "fold": fold,
            "heads": {head: {"target_contract_hash": stable_hash(derived)}},
            "policy": {"ma_exit_threshold": .55}, "failures": []}
    monkeypatch.setattr(research, "_build", build)
    monkeypatch.setattr(research, "_load_training", load)
    monkeypatch.setattr(training, "_train_ma_head", fit)
    monkeypatch.setattr(research, "_patterns", lambda *args, **kwargs: {"rows": 0})
    # No L2 model is fitted. The empty bundle exercises the skipped-head path.
    monkeypatch.setattr(research, "_train_pair", lambda *args, **kwargs: {"heads": {}, "failures": []})
    monkeypatch.setattr(evaluation, "evaluate_policy_options", lambda *args, **kwargs: [])
    def evaluate(*args, **kwargs):
        assert kwargs["allow_training"] is True
        events.append(("fake_evaluate", kwargs.get("cost_multiplier", 1)))
        return {}
    monkeypatch.setattr(evaluation, "evaluate_bundles", evaluate)
    monkeypatch.setattr(research, "_finish", lambda ctx, report, kind: report)
    from my_strategy.services import czsc_research_calendar
    monkeypatch.setattr(czsc_research_calendar, "resolve_execution_calendar", lambda *args, **kwargs: (dates, {}))
    report = research.train_high_return_research(config={"development_windows": []}, device="cpu", cpu_workers=1)
    assert events[:2] == [("build", "bootstrap"), ("load", "bootstrap", "exit")]
    assert events[2:14] == [("fake_fit", "ma_exit")] * 12
    assert events[14:16] == [("build", "ma_entry_alignment"), ("load", "ma_entry_alignment", "entry")]
    assert events[16:28] == [("fake_fit", "ma_entry")] * 12
    assert events[28:31] == [("build", "final"), ("load", "final", "entry"), ("load", "final", "exit")]
    assert events[31:] == [("fake_evaluate", 1), ("fake_evaluate", 1.5), ("fake_evaluate", 2.)]
    pipeline = report["l1_ma_target_pipeline"]
    assert pipeline["exit_heads"] == pipeline["entry_heads"] == 12
    assert report["protocol_freeze"]["l1_ma_target_pipeline_hash"] == stable_hash(pipeline)
    assert report["protocol_freeze"]["l1_exit_lineage_sha256"] == pipeline["lineage_sha256"]
    assert report["model_gate"]["pass"] is False


@pytest.mark.parametrize("allow", [False, True])
def test_evaluation_to_runtime_and_wrapper_preserve_explicit_training_context(integrated, monkeypatch, allow):
    modules, cfg, _ = integrated
    evaluation, runtime = (modules[name] for name in ("czsc_high_return_evaluation", "czsc_high_return_runtime"))
    calls = []
    def inner(*args, **kwargs):
        calls.append(kwargs)
        return {"synthetic": "no-broker-no-fit"}
    monkeypatch.setattr(runtime, "_replay_high_return", inner)
    frame = pd.DataFrame({"date": ["2026-10-08"], "symbol": ["000001.SZ"]})
    result = evaluation._replay(frame, ["2026-10-08"], model_root="synthetic", bundles=[],
        config=cfg, start="2026-10-08", end="2026-10-08", arm="ma", validation_bundle=None,
        allow_training=allow)
    assert result == {"synthetic": "no-broker-no-fit"}
    assert len(calls) == 1 and calls[0]["allow_training"] is allow
    assert calls[0]["source"] == "ma" and calls[0]["mode"] == "reference"


def test_runtime_uses_canonical_exit_catalog_and_forwards_original_calendar(integrated, monkeypatch):
    modules, cfg, _ = integrated
    training, runtime = (modules[name] for name in ("czsc_high_return_ma_training", "czsc_high_return_runtime"))
    canonical = [{"name": f"ma_{i}", "heads": {"ma_exit": {"path": f"exit-{i}"}}} for i in range(12)]
    duplicates = canonical + [{"name": "selected_L2", "heads": {"ma_exit": {"path": "exit-11"}, "entry": {}}}]
    original_dates = ["2026-10-08"]
    lineage = {"exit_only_bundles": canonical, "expected_exit_target_contract_hash": "complete-target-hash",
               "source_snapshot_hash": "original-source", "calendar_hash": stable_hash(original_dates),
               "_frozen_calendar_dates": original_dates}
    reader_calls, factory_calls = [], []
    def reader(root, calendar, config, **kwargs):
        reader_calls.append(kwargs)
        return lineage
    def factory(features, calendar, root, bundles, **kwargs):
        factory_calls.append((bundles, kwargs)); return "synthetic-callback"
    monkeypatch.setattr(training, "read_exit_bootstrap_lineage", reader)
    from my_strategy.services import czsc_high_return_ma_alignment as alignment
    monkeypatch.setattr(alignment, "make_signal_pinned_ma_exit_policy", factory)
    points = {"synthetic": "provided-scored-points"}
    extended = original_dates + ["2026-10-09"]
    assert runtime.make_ma_exit_policy(pd.DataFrame(), extended, "synthetic", duplicates, config=cfg,
        points=points) == "synthetic-callback"
    assert reader_calls == [{"bundles": duplicates, "allow_training": False}]
    routed, kwargs = factory_calls[0]
    assert routed is canonical and len(routed) == 12
    assert kwargs["exit_training_calendar_dates"] == original_dates
    assert kwargs["exit_training_calendar_hash"] == stable_hash(original_dates)
    assert kwargs["source_snapshot_hash"] == "original-source"
    assert kwargs["expected_exit_contract_hash"] == "complete-target-hash"
    assert kwargs["points"] is points
