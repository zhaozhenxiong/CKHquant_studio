"""Independent checks of policy binding, causal scope and real Broker labels."""
import copy

import numpy as np
import pandas as pd
import pytest

from my_strategy.core.run_context import stable_hash
from my_strategy.services import czsc_high_return_pipeline_v4 as pipeline
from my_strategy.services.czsc_high_return_features_v4 import ENTRY_FEATURE_COLUMNS, EXIT_FEATURE_COLUMNS
from my_strategy.services.czsc_high_return_labels_v4 import target_contract
from my_strategy.services.czsc_signal_points_v4 import (SIGNAL_POINTS_VERSION, CANDIDATE_POLICY,
    build_signal_points, policy_identity, signal_point_parameters)
from my_strategy.services.czsc_trade_episodes_v4 import build_episode_dataset, training_frame
from my_strategy.tests.test_czsc_trade_episodes import market as legacy_market_fixture


@pytest.fixture
def market():
    raw, days, cfg, points = legacy_market_fixture.__wrapped__()
    for column in ENTRY_FEATURE_COLUMNS:
        if column not in raw:
            raw[column] = .1
    cfg["version"] = pipeline.VERSION
    cfg["native_policy"] = {"native_config_hash": stable_hash({"config": "native-config", "schema": "native-schema"})}
    cfg["signal_points"] = signal_point_parameters(cfg)
    cfg["candidate_policy"] = CANDIDATE_POLICY
    cfg["source_snapshot_hash"] = "frozen-source"
    points["version"] = SIGNAL_POINTS_VERSION
    points["policy"] = policy_identity(cfg, cfg["native_policy"]["native_config_hash"])
    points["config"] = signal_point_parameters(cfg)
    points["config_hash"] = stable_hash({"parameters": points["config"], **points["policy"]})
    for event in points["events"]:
        event.update(original_event_id=event["event_id"], pattern_id=event["event_id"],
            direction_eligible=True, evidence_visible=True, first_confirmed_at=event["available_at"],
            origin_observed_at=event["available_at"], activation_kind="new_confirmation")
    return raw, days, cfg, points


def test_target_contract_uses_identical_population_and_risk_hash_expression(market):
    _, days, cfg, points = market
    contract = target_contract(head="entry", source="union", config=cfg, market_dates=days,
        source_snapshot_hash="frozen-source", feature_schema_hash="schema", continuation_policy_hash="exit",
        initial_cash=100000.)
    assert contract["candidate_policy"] == CANDIDATE_POLICY
    for key, value in points["policy"].items():
        assert contract[key] == value
    assert contract["candidate_envelope"] == pipeline.ENVELOPE


def test_scored_identity_cannot_assign_current_policy_to_incompatible_manifest(market):
    _, days, _, points = market
    manifest = {"target_contract": {**points["policy"], "risk_policy_hash": "other-risk"}}
    with pytest.raises(ValueError, match="impersonate"):
        pipeline.model_identity(manifest, {"name": "fold"}, "run", days[70], points["policy"])


def test_predict_requires_matching_frozen_candidate_risk_and_source_before_inference(monkeypatch, market):
    _, days, cfg, points = market
    manifest = {"target_contract": {"research_version": pipeline.VERSION, "candidate_policy": CANDIDATE_POLICY,
        "source_snapshot_hash": "frozen-source", **points["policy"]}, "manifest_sha256": "verified-manifest",
        "available_at": days[60], "fold": {"name": "fold"}}
    monkeypatch.setattr(pipeline, "load_head", lambda _: manifest)
    calls = []
    class Session:
        def predict_one(self, *args, **kwargs):
            calls.append(args)
            return .8
    bundle = {"name": "fold", "available_at": days[60],
        "heads": {"ma_entry": {"path": "head", "manifest_sha256": "verified-manifest"}}}
    changed = {**points["policy"], "risk_policy_hash": "different"}
    with pytest.raises(ValueError, match="candidate/risk"):
        pipeline._predict(Session(), {"date": days[70]}, "/tmp/native-v4-test", bundle, "ma_entry",
            policy=changed, source_snapshot_hash=cfg["source_snapshot_hash"])
    with pytest.raises(ValueError, match="source snapshot"):
        pipeline._predict(Session(), {"date": days[70]}, "/tmp/native-v4-test", bundle, "ma_entry",
            policy=points["policy"], source_snapshot_hash="other-source")
    assert calls == []
    assert pipeline._predict(Session(), {"date": days[70]}, "/tmp/native-v4-test", bundle, "ma_entry",
        policy=points["policy"], source_snapshot_hash=cfg["source_snapshot_hash"])[0] == .8


def test_actual_broker_complete_fee_net_label_and_maturity(market):
    raw, days, cfg, points = market
    raw.loc[72:81, ["open", "close"]] = 11.
    raw.loc[72:81, "high"] = 11.2
    raw.loc[72:81, "low"] = 10.8
    result = build_episode_dataset(raw, days, cfg, points=points, sources=("union",), source_snapshot_hash="frozen-source")
    account = result["reference_accounts"]["union:000001.SZ"]
    buy, sell = account["ledger"]
    row = result["entries"].iloc[0]
    assert buy["date"] == days[71] and sell["date"] == days[81]
    assert row.net_return == pytest.approx((buy["cash_flow"] + sell["cash_flow"]) / -buy["cash_flow"])
    assert row.label == 1 and row.r_multiple >= 2 and row.label_end == days[81]
    assert row.fees == pytest.approx(buy["fee"] + sell["fee"])
    assert row.label_available_at == days[81] + "T09:30:00+08:00"
    assert training_frame(result, head="entry", ma_only=True).attrs["label_contract"]["candidate_policy"] == CANDIDATE_POLICY


def test_rejection_and_tail_are_censored_never_false_negative(market):
    raw, days, cfg, points = market
    raw.loc[71, "open"] = 10.6
    refused = build_episode_dataset(raw, days, cfg, points=points, sources=("union",), source_snapshot_hash="frozen-source")
    assert refused["episodes"].empty and refused["entries"].label.isna().all()
    assert refused["entries"].label_reason.iloc[0].startswith("entry_rejected:")
    raw.loc[71, "open"] = 10.
    short = build_episode_dataset(raw.iloc[:77].copy(), days, cfg, points=points, sources=("union",), source_snapshot_hash="frozen-source")
    assert short["entries"].label.isna().all() and not short["entries"].label_available.any()
    assert short["exits"].label.isna().all() and not short["exits"].label_available.any()


def test_union_exit_recipe_is_stock_invariant_and_binding_is_signal_time_pinned(monkeypatch, market):
    raw, days, cfg, points = market
    result = build_episode_dataset(raw, days, cfg, points=points, sources=("union",), source_snapshot_hash="frozen-source")
    snapshot = next(item for item in result["reference_accounts"]["union:000001.SZ"]["holding_snapshots"] if item["date"] == days[75])
    bundles = [{"name": "prior", "available_at": days[60], "heads": {"ma_exit": "prior-head"}},
               {"name": "later", "available_at": days[72], "heads": {"ma_exit": "later-head"}}]
    manifest = {"train_label_end": days[30], "calibration_label_end": days[50], "target_contract_hash": "verified-target"}
    monkeypatch.setattr(pipeline, "load_head", lambda _: manifest)
    calls = []
    def predict(session, values, root, bundle, head, **kwargs):
        calls.append(bundle["name"])
        return .8, manifest
    monkeypatch.setattr(pipeline, "_predict", predict)
    policy = pipeline.pinned_exit_policy(raw, days, points, "/tmp/native-v4-test", bundles, cfg)
    answer = policy(snapshot)
    assert answer["candidate_exit_binding"]["checkpoint"] == "prior"
    assert answer["candidate_exit_binding"]["checkpoint_upgrade"] is False and answer["request_exit"] is True
    assert calls == ["prior"]
    with pytest.raises(ValueError, match="reused holding prefix"):
        policy(snapshot)
    assert policy.new_replay()(snapshot)["candidate_exit_binding"]["checkpoint"] == "prior"
    other_points = copy.deepcopy(points)
    other_raw = raw.assign(symbol="600519.SH")
    for event in other_points["events"]:
        event["symbol"] = "600519.SH"
    for candidate in other_points["candidates"]:
        candidate["symbol"] = "600519.SH"
    other = pipeline.pinned_exit_policy(other_raw, days, other_points, "/tmp/native-v4-test", bundles, cfg)
    assert policy.recipe_hash == other.recipe_hash


def test_earliest_position_without_prior_exit_head_stays_rule_v0(monkeypatch, market):
    raw, days, cfg, points = market
    bundles = [{"name": "future", "available_at": days[72], "heads": {"ma_exit": "future-head"}}]
    policy = pipeline.pinned_exit_policy(raw, days, points, "/tmp/native-v4-test", bundles, cfg)
    binding = next(iter(policy.bindings.values()))
    assert binding["mode"] == "explicit_rule_v0_no_earlier_exit_head"
    assert binding["checkpoint"] is None and binding["checkpoint_upgrade"] is False


def test_daily_ma_research_score_is_explicitly_out_of_candidate_population(monkeypatch, market):
    raw, days, cfg, points = market
    manifest = {"target_contract": {**points["policy"], "source_snapshot_hash": "frozen-source", "target": "actual_complete_fee_net_roundtrip_positive"},
        "manifest_sha256": "manifest", "checkpoint_sha256": "checkpoint", "target_contract_hash": "target", "feature_schema_hash": "schema",
        "head": "ma_entry", "train_label_end": days[30], "calibration_label_end": days[50],
        "available_at": days[60], "training_completed_at": "2026-10-12T15:00:00+08:00"}
    monkeypatch.setattr(pipeline, "_predict", lambda *args, **kwargs: (.8, manifest))
    bundle = {"name": "fold", "available_at": days[60], "heads": {"ma_entry": "head"}}
    routes = pipeline.forward_ma(raw, points, "/tmp/native-v4-test", [bundle], cfg)
    assert routes[days[70]]["in_scope"] is True
    assert routes[days[70]]["candidate_envelope"]["candidate_event_ids"] == ["buy70"]
    assert routes[days[75]]["probability"] == .8 and routes[days[75]]["in_scope"] is False
    assert routes[days[75]]["candidate_envelope"]["candidate_event_ids"] == []


def _native_pack_event(raw, day, *, confirmed_at=None, risk_stop=9.5):
    stamp = day + "T15:00:00+08:00"
    event = {"event_id": "pure-pattern", "original_event_id": "pure-pattern", "pattern_id": "pure-pattern",
        "symbol": "000001.SZ", "source": "structure", "side": "BUY", "event_type": "pure_second_bs_buy",
        "date": day, "available_at": stamp, "observed_at": stamp, "first_confirmed_at": confirmed_at or stamp,
        "origin_observed_at": stamp, "input_eligible": True, "evidence_visible": True,
        "confirmed": True, "candidate_eligible": True, "risk_stop_price": risk_stop, "support_low": 9., "invalidation_price": 9.}
    return {"frame": raw, "config_hash": "native-config", "schema_hash": "native-schema", "events": [event], "evaluations": []}


def test_new_source_future_confirmation_cannot_become_training_candidate(market):
    raw, days, cfg, _ = market
    pack = _native_pack_event(raw, days[70], confirmed_at=days[71] + "T15:00:00+08:00")
    points = build_signal_points(raw, cfg, native_pack=pack)
    event = next(event for event in points["events"] if event["source"] == "structure")
    assert event["direction_eligible"] is False and event["candidate_eligible"] is False
    assert event["evidence_visible"] is False


def test_bullish_ma_short_tactical_weakening_does_not_create_long_failure(market):
    raw, days, cfg, _ = market
    raw["close_above_ma20"] = 1.
    raw["ma20_slope5"] = .1
    raw["ma_alignment"] = 1.
    raw["ma5_ma10_gap"] = .1
    raw["ret_5"] = .1
    raw.loc[70:, "ma5_ma10_gap"] = -.004546
    raw.loc[70:, "ret_5"] = -.07438
    pack = {"frame": raw, "config_hash": "native-config", "schema_hash": "native-schema", "events": [], "evaluations": []}
    points = build_signal_points(raw, cfg, native_pack=pack)
    assert not any(event["source"] == "ma" and event["side"] == "SELL" for event in points["events"])
    raw.loc[75:, "close_above_ma20"] = -1.
    raw.loc[75:, "ma20_slope5"] = -.1
    points = build_signal_points(raw, cfg, native_pack=pack)
    sells = [event for event in points["events"] if event["source"] == "ma" and event["side"] == "SELL"]
    assert len(sells) == 1 and sells[0]["date"] == days[75] and sells[0]["event_type"] == "ma_trend_failure"


def test_production_route_loads_new_family_and_checks_actual_completion(monkeypatch, market):
    _, days, _, _ = market
    from my_strategy.services import czsc_high_return_runtime as legacy_runtime
    monkeypatch.setattr(legacy_runtime, "load_head", lambda _: (_ for _ in ()).throw(AssertionError("new route must not read old family")))
    completed = {"prior-head": days[60] + "T15:00:00+08:00", "future-head": days[75] + "T16:00:00+08:00"}
    monkeypatch.setattr(pipeline, "load_head", lambda path: {"training_completed_at": completed[path.name]})
    bundles = [{"name": "prior", "available_at": days[60], "heads": {"ma_entry": "prior-head"}},
               {"name": "future", "available_at": days[72], "heads": {"ma_entry": "future-head"}}]
    assert pipeline._bundle_for(days[75], bundles, model_root="/tmp/native-v4-test", head="ma_entry", usage_mode="production")["name"] == "prior"
    assert pipeline._bundle_for(days[76], bundles, model_root="/tmp/native-v4-test", head="ma_entry", usage_mode="production")["name"] == "future"
    assert pipeline._bundle_for(days[75], bundles, model_root="/tmp/native-v4-test", head="ma_entry", usage_mode="historical")["name"] == "future"


def test_first_risk_eligibility_activation_is_once_and_keeps_original_confirmation(market):
    raw, days, cfg, _ = market
    raw.loc[71:, ["open", "close"]] = 9.3
    raw.loc[71:, "high"] = 9.5
    raw.loc[71:, "low"] = 9.1
    pack = _native_pack_event(raw, days[70], risk_stop=8.)
    original = pack["events"][0]
    original.update(invalidation_price=8.1, support_low=8.1,
        reference_price=10., reference_price_date=days[70], reference_price_source="signal_day_close")
    pack["evaluations"] = [{"date": day, "structure": {"support_facts": [original]},
                            "wyckoff": {"support_facts": []}} for day in days[70:]]
    points = build_signal_points(raw, cfg, native_pack=pack)
    structure = [event for event in points["events"] if event["source"] == "structure"]
    assert len(structure) == 2
    first, activated = structure
    assert first["candidate_eligible"] is False and activated["candidate_eligible"] is True
    assert activated["activation_kind"] == "first_risk_eligibility" and activated["date"] == days[71]
    assert activated["original_event_id"] == first["event_id"] and activated["first_confirmed_at"] == first["first_confirmed_at"]
    assert activated["available_at"] == days[71] + "T15:00:00+08:00"
    assert activated["reference_price"] == 9.3 and activated["reference_price_date"] == days[71]
    assert activated["origin_reference_price"] == 10. and activated["origin_reference_price_date"] == days[70]


@pytest.mark.parametrize("field,new_value", [("manifest_sha256", "drift"), ("available_at", "2024-02-01"), ("fold_name", "other-fold")])
def test_model_catalog_drift_rejects_score_before_inference(monkeypatch, market, field, new_value):
    _, days, cfg, points = market
    manifest = {"target_contract": {"research_version": pipeline.VERSION, "candidate_policy": CANDIDATE_POLICY,
        "source_snapshot_hash": "frozen-source", **points["policy"]}, "manifest_sha256": "verified-manifest",
        "available_at": days[60], "fold": {"name": "fold"}}
    if field == "fold_name":
        manifest["fold"]["name"] = new_value
    else:
        manifest[field] = new_value
    monkeypatch.setattr(pipeline, "load_head", lambda _: manifest)
    class Session:
        def predict_one(self, *args, **kwargs):
            raise AssertionError("catalog drift must reject before inference")
    bundle = {"name": "fold", "available_at": days[60],
        "heads": {"ma_entry": {"path": "head", "manifest_sha256": "verified-manifest"}}}
    with pytest.raises(ValueError, match="catalog binding drift"):
        pipeline._predict(Session(), {"date": days[70]}, "/tmp/native-v4-test", bundle, "ma_entry",
            policy=points["policy"], source_snapshot_hash=cfg["source_snapshot_hash"])
