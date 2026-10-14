"""Source references and risk ceilings cannot silently become trading targets."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pandas as pd
import pytest

from my_strategy.services import czsc_trend_runtime as runtime
from my_strategy.services.czsc_native_evidence import _pure_third_sell_signal
from my_strategy.services.czsc_trend_contracts import strategy_config
from my_strategy.tests.test_reference_analysis import reference_inputs, DATES


def _source(day, side="SELL", **updates):
    return {"symbol": "000001.SZ", "date": day, "source": "structure", "side": side,
        "event_id": "source:" + day, "original_event_id": "source:" + day, "pattern_id": "pattern:" + day,
        "event_type": "pure_first_sell", "confirmed": True, "input_eligible": True,
        "candidate_direction_valid": True, "available_at": day + "T15:00:00+08:00",
        "first_available_at": day + "T15:00:00+08:00", "first_confirmed_at": day + "T15:00:00+08:00",
        "stop_price": 13., "invalidation_price": 13., **updates}


def test_source_sell_survives_without_final_sell_and_preserves_original_identity(reference_inputs, monkeypatch):
    from my_strategy.services import czsc_native_evidence
    original = czsc_native_evidence.build_native_evidence
    event = _source(DATES[0])
    def native(*args, **kwargs):
        pack = original(*args, **kwargs)
        pack["events"] = [copy.deepcopy(event)]
        return pack
    monkeypatch.setattr(czsc_native_evidence, "build_native_evidence", native)
    result = runtime.reference_overlay(reference_inputs, DATES)
    source = result["source_reference_points"]
    assert len(source) == 1 and source[0]["side"] == "SELL"
    for key in ("event_id", "original_event_id", "pattern_id", "available_at", "first_available_at"):
        assert source[0][key] == event[key]
    assert source[0]["reference_layer"] == "source"
    assert source[0]["consensus_qualified"] is False and source[0]["consensus_action"] == "BUY"
    assert result["reference_points"][0]["side"] == "BUY"
    assert source[0]["reference_price"] == 10.
    from my_strategy.web_dashboard.chart_evidence_trend import decorate_analysis
    chart = {"symbol": "000001.SZ", "data_end": DATES[-1], "frequencies": {
        "日线": {"bars": [{"time": day} for day in DATES[1:]]}}}
    visible = decorate_analysis(chart, raw=reference_inputs, overlay=result, start=DATES[1])
    assert visible["source_reference_points"] == []
    stats = visible["chart_evidence"]["statistics"]
    assert stats["full_prefix"]["source_reference_sell"] == 1
    assert stats["visible_window"]["source_reference_sell"] == 0


@pytest.mark.parametrize("updates", [
    {"confirmed": False}, {"input_eligible": False}, {"candidate_direction_valid": False},
    {"source": "foreign"}, {"symbol": "other"}, {"first_confirmed_at": "2026-10-01T15:00:00+08:00"},
    {"available_at": "2026-10-01T15:00:00+08:00"}, {"original_event_id": None},
])
def test_invalid_or_unconfirmed_source_never_becomes_bs(updates):
    row = {"symbol": "000001.SZ", "date": DATES[0], "available_at": DATES[0] + "T15:00:00+08:00",
        "reference_price": 10., "direction": "SELL", "agents": {"ma": {"status": "unavailable"}}}
    assert runtime._source_reference_point(_source(DATES[0], **updates), row) is None


def _risk_inputs(close=10., stop=9.5):
    days = pd.bdate_range("2026-01-02", periods=25).strftime("%Y-%m-%d").tolist()
    bars = [{"date": day, "high": 10.2, "low": 9.8, "close": close} for day in days]
    row = {"symbol": "000001.SZ", "date": days[-1], "reference_price": close,
        "available_at": days[-1] + "T15:00:00+08:00", "direction": "BUY",
        "agents": {"ma": {"status": "unavailable"}, "structure": {"status": "scored"}, "wyckoff": {"status": "scored"}}}
    event = _source(days[-2], side="BUY", invalidation_price=min(stop + .001, 9.9999), risk_stop_price=stop)
    native = {"structure": {"support_facts": [event]}, "wyckoff": {"support_facts": []}}
    return row, bars, native


def _qualified_buy_inputs():
    from my_strategy.services.czsc_trend_consensus import _risk_plan
    row, bars, native = _risk_inputs()
    event = native["structure"]["support_facts"][0]
    event["opportunity_id"] = "structure:" + event["pattern_id"] + ":BUY"
    row.update(buy_base_score=.55, sell_base_score=.05, buy_base_qualified=True,
        entry_eligible=True, buy_independent_families=["ma", "structure"],
        active_candidates=[copy.deepcopy(event)], original_opportunity_ids=[event["opportunity_id"]])
    row["agents"]["ma"]["status"] = "scored"
    row["entry_plan"] = _risk_plan(row, row["active_candidates"], len(bars) - 1, bars, strategy_config())
    return row, bars, native


def test_live_buy_risk_uses_original_quality_geometry_over_a_higher_auxiliary_zone():
    row, bars, native = _qualified_buy_inputs()
    native["structure"]["current_state"] = {"current_valid_zone": {"support_low": 9.9,
        "confirmed": True, "dependency_calendar_complete": True, "zone_id": "auxiliary-zone",
        "first_confirmed_at": row["available_at"]}}
    risk = runtime._reference_risk_plan(row, bars, len(bars) - 1, native, strategy_config())
    plan = row["entry_plan"]
    assert risk["status"] == "available"
    assert risk["stop_price"] == plan["stop_price"] == 9.5
    assert risk["risk_position_cap_fraction"] == plan["target_weight"] == .2
    assert risk["reference_price_floor"] == plan["price_floor"]
    assert risk["reference_price_ceiling"] == plan["price_ceiling"]
    assert risk["original_buy_plan_id"] == plan["plan_id"]
    assert risk["source_event_ids"] == plan["source_event_ids"]


@pytest.mark.parametrize("changes", [
    {"entry_eligible": False}, {"buy_base_score": .399999}, {"sell_base_score": .451},
    {"buy_independent_families": ["ma"]}, {"original_opportunity_ids": []},
])
def test_original_buy_geometry_cannot_be_claimed_without_original_admission(changes):
    row, _, _ = _qualified_buy_inputs()
    row.update(changes)
    plan, reasons = runtime._reference_buy_plan(row, strategy_config())
    assert plan is None and reasons


@pytest.mark.parametrize("changes", [
    {"symbol": "other"}, {"signal_date": "2099-01-01"}, {"available_at": "2099-01-01T15:00:00+08:00"},
    {"reference_price": 10.01}, {"stop_price": 9.51}, {"target_weight": .21},
    {"price_floor": 1.}, {"price_ceiling": 100.}, {"source_event_ids": []},
])
def test_original_buy_geometry_must_match_exact_price_clock_risk_and_opportunities(changes):
    row, _, _ = _qualified_buy_inputs()
    row["entry_plan"].update(changes)
    plan, reasons = runtime._reference_buy_plan(row, strategy_config())
    assert plan is None and reasons


@pytest.mark.parametrize("changes", [
    {"confirmed": False}, {"input_eligible": False}, {"source": "foreign"},
    {"first_confirmed_at": "2099-01-01T15:00:00+08:00"},
    {"first_available_at": "2099-01-01T15:00:00+08:00"},
])
def test_original_buy_plan_never_borrows_unconfirmed_or_future_opportunities(changes):
    row, _, _ = _qualified_buy_inputs()
    row["active_candidates"][0].update(changes)
    plan, reasons = runtime._reference_buy_plan(row, strategy_config())
    assert plan is None and "original_buy_opportunity_identity_unavailable" in reasons


def test_real_risk_ceiling_kept_pending_without_qualified_ml_or_quality():
    row, bars, native = _risk_inputs()
    plan = runtime._reference_risk_plan(row, bars, len(bars) - 1, native, strategy_config())
    assert plan["status"] == "available"
    assert plan["atr14"] == pytest.approx(.4)
    assert plan["stop_distance_fraction"] == pytest.approx(.05)
    assert plan["risk_position_cap_fraction"] == pytest.approx(.2)
    assert plan["risk_budget_fraction"] == .01 and plan["max_position_fraction"] == .95
    position = runtime._reference_position(row, plan, current_position_fraction=.3)
    assert position["status"] == "pending"
    assert position["target_position_fraction"] is None
    assert position["change_fraction"] is None
    assert position["current_position_fraction"] == .3
    assert position["risk_position_cap_fraction"] == pytest.approx(.2)
    assert position["entry_probability"] is position["exit_probability"] is None
    assert "qualified_trend_model_unavailable" in position["reason_codes"]
    assert "entry_quality_not_evaluated_reference_population" in position["reason_codes"]


def test_authentic_reference_session_without_assets_is_pending_with_its_real_reason(tmp_path):
    from my_strategy.services.czsc_reference_quality import ReferenceQualitySession
    row, bars, native = _qualified_buy_inputs()
    cfg = strategy_config()
    risk = runtime._reference_risk_plan(row, bars, len(bars) - 1, native, cfg)
    score = ReferenceQualitySession(tmp_path, {"config": cfg}).score({}, row, {}, risk_plan=risk)
    position = runtime._reference_position(row, risk, reference_quality=score, config=cfg,
        model_run_id=tmp_path.name)
    assert position["quality_status"] == "pending"
    assert position["target_position_fraction"] is position["reference_quality_probability"] is None
    assert "reference_quality_population_unavailable" in position["reason_codes"]
    assert "identity_error" not in position["reason_codes"]


def test_plain_verified_dictionary_cannot_self_certify_a_reference_target():
    row, bars, native = _qualified_buy_inputs()
    cfg = strategy_config()
    risk = runtime._reference_risk_plan(row, bars, len(bars) - 1, native, cfg)
    position = runtime._reference_position(row, risk, reference_quality={"status": "scored",
        "probability": 1., "fee_qualified": True, "verified": True}, config=cfg)
    assert position["quality_status"] == "identity_error"
    assert position["target_position_fraction"] is position["reference_quality_probability"] is None
    assert "reference_quality_untrusted_or_changed_result" in position["reason_codes"]


def test_replacing_probability_on_a_session_score_cannot_reuse_its_issuer(tmp_path):
    from dataclasses import replace
    from my_strategy.services.czsc_reference_quality import ReferenceQualitySession
    row, bars, native = _qualified_buy_inputs()
    cfg = strategy_config()
    risk = runtime._reference_risk_plan(row, bars, len(bars) - 1, native, cfg)
    score = ReferenceQualitySession(tmp_path, {"config": cfg}).score({}, row, {}, risk_plan=risk)
    forged = replace(score, status="scored", probability=.99, fee_qualified=True,
        reference_population_hash="fake-population", qualification_hash="fake-qualified")
    position = runtime._reference_position(row, risk, reference_quality=forged, config=cfg,
        model_run_id=tmp_path.name)
    assert position["target_position_fraction"] is None
    assert position["quality_status"] == "identity_error"


@pytest.fixture(scope="module")
def qualified_reference_bundle(tmp_path_factory):
    from my_strategy.tests.test_reference_quality import qualified_reference_fixture
    patch = pytest.MonkeyPatch()
    try:
        bundle = qualified_reference_fixture(tmp_path_factory.mktemp("reference-consumer"), patch)
        bundle[4]["symbol"] = bundle[3]["symbol"]
        yield bundle
    finally:
        patch.undo()


def _real_reference_score(bundle, *, probability=None, buy_base=None):
    from my_strategy.services.czsc_reference_quality import ReferenceQualitySession, COLUMNS
    root, report, cfg, row, risk, feature, prefix = bundle
    row, risk, feature = copy.deepcopy(row), copy.deepcopy(risk), copy.deepcopy(feature)
    if buy_base is not None:
        row["buy_base_score"] = buy_base
    session = ReferenceQualitySession(root, report, config=cfg)
    if probability is not None:
        # Vary a real causal feature through the fitted classifier; never
        # replace its probability, private issuer or fee qualification.
        low, high = -4., 4.
        for _ in range(45):
            feature[COLUMNS[0]] = (low + high) / 2
            score = session.score(feature, row, prefix, risk_plan=risk)
            if score.probability < probability:
                low = feature[COLUMNS[0]]
            else:
                high = feature[COLUMNS[0]]
    score = session.score(feature, row, prefix, risk_plan=risk)
    return root, cfg, row, risk, score


def test_fee_qualified_reference_has_positive_target_under_frozen_nonbaseline_policy(qualified_reference_bundle, monkeypatch):
    from my_strategy.execution.broker import BrokerSimulator
    root, cfg, row, risk, score = _real_reference_score(qualified_reference_bundle)
    monkeypatch.setattr(BrokerSimulator, "__init__", lambda *a, **k: pytest.fail("reference allocation must never replay Broker"))
    position = runtime._reference_position(row, risk, reference_quality=score, config=cfg,
        model_run_id=root.name, current_position_fraction=.1)
    assert cfg["consensus"]["buy_base_threshold"] == .45
    assert cfg["consensus"]["margin"] == .15
    assert position["status"] == "increase_exposure"
    assert position["target_position_fraction"] == position["risk_position_cap_fraction"] == .2
    assert position["change_fraction"] == pytest.approx(.1)
    assert position["reference_quality_probability"] == score.probability
    assert position["reference_quality_identity"]["head"] == "reference_quality"
    assert position["entry_probability"] is position["exit_probability"] is None
    assert position["reason_codes"] == []
    assert qualified_reference_bundle[1]["reference_quality"]["qualification"]["passed"] is True
    # The independent formal certificate remains pending in this synthetic
    # Ledger fixture; it does not negate its local research fee qualification.
    assert qualified_reference_bundle[1]["reference_quality"]["required_for_core_selection"] is False


@pytest.mark.parametrize("probability,base,expected,reason", [
    (.09999, .65, "pending", "reference_quality_below_threshold"),
    (.10001, .65, "increase_exposure", None),
    (.4999, .45, "pending", "reference_total_support_below_threshold"),
    (.5001, .45, "increase_exposure", None),
])
def test_real_reference_probability_and_total_support_keep_original_gates(qualified_reference_bundle, probability, base, expected, reason):
    root, cfg, row, risk, score = _real_reference_score(qualified_reference_bundle, probability=probability, buy_base=base)
    assert score.probability == pytest.approx(probability, abs=2e-7)
    position = runtime._reference_position(row, risk, reference_quality=score, config=cfg, model_run_id=root.name)
    assert position["status"] == expected
    assert position["reference_quality_threshold"] == .1
    assert position["reference_buy_total_threshold"] == .5
    if reason:
        assert reason in position["reason_codes"] and position["target_position_fraction"] is None
    else:
        assert position["target_position_fraction"] == .2


@pytest.mark.parametrize("field,value", [("reference_price", 10.01), ("date", "2026-10-13"), ("symbol", "foreign")])
def test_verified_quality_score_cannot_transfer_to_another_price_clock_or_symbol(qualified_reference_bundle, field, value):
    root, cfg, row, risk, score = _real_reference_score(qualified_reference_bundle)
    row[field] = value
    position = runtime._reference_position(row, risk, reference_quality=score, config=cfg, model_run_id=root.name)
    assert position["target_position_fraction"] is None and position["quality_status"] == "identity_error"


def test_old_pinned_trend_remains_reference_only_and_cannot_borrow_positive_quality(qualified_reference_bundle):
    from my_strategy.services.czsc_reference_quality import ReferenceQualitySession
    root, report, cfg, row, risk, feature, prefix = qualified_reference_bundle
    row = copy.deepcopy(row)
    row["agents"]["ma"]["model_identity"]["checkpoint"] = "older-pinned"
    score = ReferenceQualitySession(root, report, config=cfg).score(feature, row, prefix, risk_plan=risk)
    position = runtime._reference_position(row, risk, reference_quality=score, config=cfg, model_run_id=root.name)
    assert position["status"] == "pending" and position["target_position_fraction"] is None
    assert "reference_quality_trend_route_out_of_population" in position["reason_codes"]


def test_default_reference_overlay_keeps_the_selected_nonbaseline_policy(qualified_reference_bundle, monkeypatch):
    from my_strategy.services import czsc_native_evidence, czsc_trend_model, czsc_trend_pipeline, czsc_trend_consensus
    from my_strategy.services.czsc_trend_consensus import _risk_plan
    from my_strategy.core.run_context import stable_hash
    root, report, cfg, template, _, feature, _ = qualified_reference_bundle
    days = pd.bdate_range(end=template["date"], periods=25).strftime("%Y-%m-%d").tolist()
    frame = pd.DataFrame([{**feature, "date": day, "open": 10., "close": 10.,
        "high": 10.2, "low": 9.6} for day in days])
    final = copy.deepcopy(template)
    for event in final["active_candidates"]:
        event["risk_stop_price"] = 9.5
    final["entry_plan"] = _risk_plan(final, final["active_candidates"], len(days) - 1,
        frame.to_dict("records"), cfg)
    rows = []
    for day in days:
        row = copy.deepcopy(final)
        row.update(date=day, available_at=day + "T15:00:00+08:00", candidate_evidence=[])
        if day != days[-1]:
            row.update(direction="WAIT", entry_eligible=False, buy_base_qualified=False,
                entry_plan=None, active_candidates=[], original_opportunity_ids=[])
        rows.append(row)
    monkeypatch.setattr(runtime, "load_reference_model_run", lambda *a, **k: (root, report))
    monkeypatch.setattr(czsc_trend_model, "bind_trend_inputs", lambda data, *a, **k: data.copy())
    monkeypatch.setattr(czsc_native_evidence, "build_native_evidence", lambda data, *a, **k: {
        "frame": data.copy(), "events": [], "evaluations": [], "wyckoff_events": []})
    monkeypatch.setattr(czsc_trend_pipeline, "forward_trend", lambda data, *a, **k: {
        day: {"model_available": True, "input_eligible": True} for day in data.date})
    monkeypatch.setattr(czsc_trend_consensus, "prepare_trend_decisions", lambda *a, **k: {
        "rows": copy.deepcopy(rows), "events": [], "config_hash": stable_hash(cfg)})
    output = runtime.reference_overlay(frame, days, consensus_config=None)
    position = output["position_reference"]
    assert position["status"] == "increase_exposure"
    assert position["target_position_fraction"] == .2
    assert output["reference_quality_probability"] > .1
    assert output["latest_decision"]["agents"]["reference_quality"]["applied_to_reference_position"] is True
    assert output["latest_decision"]["high_return_probability"] is None


def test_reference_session_rejects_changed_checkpoint_sha_before_any_target(qualified_reference_bundle, tmp_path):
    import shutil
    from my_strategy.services.czsc_reference_quality import ReferenceQualitySession
    root, report, cfg, _, _, _, _ = qualified_reference_bundle
    copied = tmp_path / "changed-model"
    shutil.copytree(root, copied)
    checkpoint = copied / report["reference_quality"]["bundles"][0]["heads"]["reference_quality"]["path"] / "model.joblib"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        ReferenceQualitySession(copied, report, config=cfg)


def test_production_reference_cannot_use_quality_trained_after_signal(qualified_reference_bundle):
    from my_strategy.services.czsc_reference_quality import ReferenceQualitySession
    root, report, cfg, row, risk, feature, prefix = qualified_reference_bundle
    score = ReferenceQualitySession(root, report, config=cfg, usage_mode="production").score(
        feature, row, prefix, risk_plan=risk)
    position = runtime._reference_position(row, risk, reference_quality=score, config=cfg, model_run_id=root.name)
    assert position["status"] == "pending" and position["target_position_fraction"] is None
    assert score.probability is None
    assert "no_strictly_prior_reference_quality_head" in position["reason_codes"]


@pytest.mark.parametrize("stop,expected", [(9.99, .95), (8.5, 1 / 15), (8.49, None), (9.999, None)])
def test_risk_exact_limits_not_rounded_or_lowered(stop, expected):
    row, bars, native = _risk_inputs(stop=stop)
    plan = runtime._reference_risk_plan(row, bars, len(bars) - 1, native, strategy_config())
    if expected is None:
        assert plan["risk_position_cap_fraction"] is None
        assert plan["status"] == "unavailable"
    else:
        assert plan["risk_position_cap_fraction"] == pytest.approx(expected)


def test_no_confirmed_stop_keeps_missingness_and_s_reduces_exposure_without_held_features():
    row, bars, native = _risk_inputs()
    native["structure"]["support_facts"] = []
    risk = runtime._reference_risk_plan(row, bars, len(bars) - 1, native, strategy_config())
    assert risk["stop_price"] is risk["risk_position_cap_fraction"] is None
    assert "confirmed_reference_stop_unavailable" in risk["reason_codes"]
    row["direction"] = "SELL"
    position = runtime._reference_position(row, risk, current_position_fraction=.45)
    assert position["status"] == "reduce_exposure"
    assert position["target_position_fraction"] == position["exposure_cap_fraction"] == 0
    assert position["change_fraction"] == -.45 and position["exit_probability"] is None
    row["direction"] = "WAIT"
    wait = runtime._reference_position(row, risk)
    assert wait["current_position_fraction"] is wait["target_position_fraction"] is None


@pytest.mark.parametrize("current", [float("nan"), float("inf"), -1, 1.01, True])
def test_reference_percentage_cannot_forge_account_or_invalid_inventory(current):
    row, bars, native = _risk_inputs()
    plan = runtime._reference_risk_plan(row, bars, len(bars) - 1, native, strategy_config())
    with pytest.raises(ValueError):
        runtime._reference_position(row, plan, current_position_fraction=current)


def test_third_sell_mirrors_pure_native_third_buy_on_actual_compiled_pens():
    import numpy as np
    from my_strategy.services.czsc_analysis import CzscSession, strategy_config as native_config
    rng = np.random.default_rng(17)
    close = 20 * np.exp(np.cumsum(rng.normal(.0006, .025, 1000)))
    bars = pd.DataFrame({"symbol": "000001.SZ", "date": pd.bdate_range("2020-01-02", periods=len(close)),
        "open": close * .998, "high": close * 1.009, "low": close * .99, "close": close,
        "volume": 1000000., "amount": close * 1000000.})
    bars["id"] = np.arange(len(bars))
    bars["dt"] = bars.date + pd.Timedelta(hours=15)
    reflected = bars.copy()
    pivot = float(bars.high.max()) * 2
    reflected["open"], reflected["close"] = pivot - bars.open, pivot - bars.close
    reflected["high"], reflected["low"] = pivot - bars.low, pivot - bars.high
    reflected["amount"] = reflected.close * reflected.volume
    first = CzscSession("000001.SZ", native_config(), len(bars), create_position=False)
    second = CzscSession("000001.SZ", native_config(), len(bars), create_position=False)
    matches = 0
    for original, mirror in zip(bars.itertuples(index=False), reflected.itertuples(index=False)):
        first.advance_structure(original); second.advance_structure(mirror)
        native = second.kas["日线"]
        di = len(native.bi_list) - len(native.finished_bis) + 1
        signal = second.czsc._native.call_signal("cxt_third_buy_V230228", native, {"di": di})[0]
        actual, size = _pure_third_sell_signal(first.kas["日线"])
        assert actual == ("三卖" if str(signal.v1) == "三买" else "其他")
        if actual == "三卖":
            assert str(signal.v2) == str(size) + "笔"
            matches += 1
    assert matches > 0


def test_third_sell_never_reads_forming_pens():
    class Forbidden:
        @property
        def bi_list(self):
            pytest.fail("forming pens must not be observed")
        finished_bis = []
    assert _pure_third_sell_signal(Forbidden()) == ("其他", None)
