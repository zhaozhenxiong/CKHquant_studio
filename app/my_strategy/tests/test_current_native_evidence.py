"""Real compiled CZSC and causal event/state contracts for the current producer."""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_analysis import CzscSession, _pen_key, strategy_config
from my_strategy.services.czsc_native_evidence import (
    NATIVE_EVIDENCE_VERSION, NATIVE_SCHEMA_HASH, _VerifiedCalendar, _zone_snapshot, build_native_evidence,
)


def _raw(n=220, *, varied=False):
    if varied:
        rng = np.random.default_rng(17)
        close = 20 * np.exp(np.cumsum(rng.normal(.0006, .025, n)))
        volume = rng.uniform(600_000, 2_000_000, n)
        high, low, op = close * 1.009, close * .99, close * .998
    else:
        close = 10 + np.choose(np.arange(n) % 4, [-.4, .4, .2, -.2])
        volume = 1_000_000. + (np.arange(n) % 7) * 10_000.
        high, low, op = np.full(n, 11.), np.full(n, 9.), close - .05
    result = pd.DataFrame({"symbol": "000001.SZ", "date": pd.bdate_range("2020-01-02", periods=n),
        "open": op, "high": high, "low": low, "close": close, "volume": volume,
        "amount": close * volume, "source": "sina_unadjusted_v1", "has_trade_price": 1},
        index=np.arange(n) * 3 + 5)
    result.attrs.update(price_basis="unadjusted", volume_unit="shares", amount_unit="CNY")
    result.attrs["verified_market_dates"] = result.date.dt.strftime("%Y-%m-%d").tolist()
    return result


def _set_bar(raw, i, *, op, hi, lo, close, volume):
    for key, value in {"open": op, "high": hi, "low": lo, "close": close,
                       "volume": volume, "amount": close * volume}.items():
        raw.loc[raw.index[i], key] = value


@pytest.fixture(scope="module")
def bars():
    return _raw(650, varied=True)


@pytest.fixture(scope="module")
def pack(bars):
    return build_native_evidence(bars)


def test_old_columns_and_input_unchanged_and_separate_schema(bars, pack):
    original = bars.copy(deep=True)
    pd.testing.assert_frame_equal(pack["frame"][bars.columns], original)
    assert pack["version"] == NATIVE_EVIDENCE_VERSION
    assert pack["schema_hash"] == NATIVE_SCHEMA_HASH
    assert len(pack["evaluations"]) == len(bars)
    assert pack["frame"].index.equals(bars.index)
    assert pack["frame"].attrs["price_basis"] == "unadjusted"
    assert not pack["frame"]["native_w_input_eligible"].iloc[:119].any()
    assert pack["frame"]["native_w_demand_score"].iloc[:119].isna().all()
    assert pack["frame"]["native_structure_buy_support"].iloc[:119].isna().all()
    # Re-evaluating must not mutate the caller or any previous result object.
    before = copy.deepcopy(pack["evaluations"])
    build_native_evidence(bars.iloc[:150])
    assert before == pack["evaluations"]
    pd.testing.assert_frame_equal(bars, original)


@pytest.mark.parametrize("end", [1, 119, 120, 311, 521])
def test_exact_prefix_equivalence_for_frames_facts_and_snapshots(bars, pack, end):
    prefix = build_native_evidence(bars.iloc[:end].copy())
    columns = [key for key in prefix["frame"] if key.startswith("native_")]
    pd.testing.assert_frame_equal(prefix["frame"][columns], pack["frame"][columns].iloc[:end], check_exact=True)
    end_day = bars.iloc[end - 1].date.strftime("%Y-%m-%d")
    assert prefix["events"] == [item for item in pack["events"] if item["date"] <= end_day]
    assert prefix["evaluations"] == pack["evaluations"][:end]


def test_future_mutation_has_no_historical_effect(bars, pack):
    mutated = bars.copy(deep=True)
    cut = 310
    for key in ("open", "high", "low", "close"):
        mutated.loc[mutated.index[cut]:, key] *= 9
    mutated.loc[mutated.index[cut]:, ["volume", "amount"]] *= 8
    mutated.loc[mutated.index[cut]:, "source"] = "unsafe_later_source"
    changed = build_native_evidence(mutated)
    columns = [key for key in pack["frame"] if key.startswith("native_")]
    pd.testing.assert_frame_equal(changed["frame"][columns].iloc[:cut], pack["frame"][columns].iloc[:cut], check_exact=True)
    assert changed["evaluations"][:cut] == pack["evaluations"][:cut]


def test_all_structural_patterns_have_unique_immutable_confirmed_identity(pack):
    patterns = [item for item in pack["events"] if item["source"] == "structure"]
    assert patterns
    assert len({item["pattern_id"] for item in patterns}) == len(patterns)
    assert len({item["event_id"] for item in patterns}) == len(patterns)
    allowed = {"cxt_first_buy_V221126", "cxt_first_sell_V221126", "cxt_second_bs_V240524",
               "cxt_third_buy_V230228", "confirmed_third_sell_price_mirror", "confirmed_support_close_break"}
    for item in patterns:
        assert item["rule_name"] in allowed
        assert item["first_confirmed_at"] == item["available_at"] == item["observed_at"]
        assert item["event_id"] == item["original_event_id"]
        assert "rejected" not in item["event_id"]
        assert not (item["rule_name"] == "cxt_third_buy_V230228" and item["side"] == "SELL")
        if item["dependencies"]:
            assert all(pen[1] <= item["date"] for pen in item["dependencies"])
    # Current structure persists; risk or repeating native state never births
    # an event for every supporting day.
    supported = [item["structure"] for item in pack["evaluations"] if item["structure"]["support_facts"]]
    ids = Counter(fact["event_id"] for item in supported for fact in item["support_facts"])
    assert max(ids.values()) > 1
    invalidated = set()
    for evaluation in pack["evaluations"]:
        state = evaluation["structure"]
        invalidated.update(state["invalidated_event_ids"])
        assert not invalidated.intersection(fact["event_id"] for fact in state["support_facts"])


def test_pure_confirmed_native_values_match_compiled_runtime(bars, pack):
    engine = bars.copy()
    engine["id"] = np.arange(len(engine)); engine["dt"] = engine.date + pd.Timedelta(hours=15)
    session = CzscSession("000001.SZ", strategy_config(), len(engine), collect_confirmations=False, create_position=False)
    rules = {"first_buy": "cxt_first_buy_V221126", "first_sell": "cxt_first_sell_V221126",
             "second_bs": "cxt_second_bs_V240524", "third_buy": "cxt_third_buy_V230228"}
    for i, bar in enumerate(engine.itertuples(index=False)):
        session.advance_structure(bar)
        native = session.kas["日线"]
        di = len(native.bi_list) - len(native.finished_bis) + 1
        for key, rule in rules.items():
            parameters = {"di": di, **({"w": 9, "t": 2} if key == "second_bs" else {})}
            label = str(session.czsc._native.call_signal(rule, native, parameters)[0].v1) if native.finished_bis else "其他"
            assert pack["evaluations"][i]["structure"]["current_state"]["native_labels"][key] == label


def test_latest_forming_zone_keeps_previous_confirmed_geometry(bars, pack):
    engine = bars.copy(); engine["id"] = np.arange(len(engine)); engine["dt"] = engine.date + pd.Timedelta(hours=15)
    session = CzscSession("000001.SZ", strategy_config(), len(engine), collect_confirmations=False, create_position=False)
    found = 0
    for i, bar in enumerate(engine.itertuples(index=False)):
        session.advance_structure(bar); native = session.kas["日线"]
        zones = native.zs_list
        if len(zones) < 2 or len(zones[-1].bis) >= 3:
            continue
        confirmed = [z for z in zones if len(z.bis) >= 3 and z.is_valid()
                     and all(_pen_key(bi) in {_pen_key(pen) for pen in native.finished_bis} for bi in z.bis[:3])]
        if not confirmed:
            continue
        actual = pack["evaluations"][i]["structure"]["current_state"]["current_valid_zone"]
        assert actual is not None
        assert actual["support_low"] == float(confirmed[-1].zd)
        assert actual["resistance_high"] == float(confirmed[-1].zg)
        found += 1
    assert found > 0


def test_verified_calendar_is_explicit_and_absent_schedule_never_inferred():
    raw=_raw(180);raw.attrs.pop('verified_market_dates')
    pack=build_native_evidence(raw)
    assert not pack['frame'].native_structure_input_eligible.any()
    assert not pack['frame'].native_w_input_eligible.any()
    assert pack['frame'].native_w_demand_score.isna().all()
    assert pack['calendar_provenance']['provided'] is False
    assert all(not event['input_eligible'] for event in pack['events'])
    supplied=build_native_evidence(raw,market_dates=raw.date.dt.strftime('%Y-%m-%d').tolist())
    assert supplied['frame'].native_w_input_eligible.iloc[-1]


def test_calendar_dependency_gap_after_120_bars_stays_incomplete_but_gap_before_span_does_not():
    raw=_raw(400);calendar=raw.attrs['verified_market_dates']
    sparse=raw.drop(index=raw.index[50]).copy()
    checker=_VerifiedCalendar(sparse,calendar)
    spanning=checker.span(20,300)
    assert spanning['dependency_expected_sessions']==spanning['dependency_observed_sessions']+1
    assert spanning['dependency_calendar_complete'] is False
    # Starting with the first bar AFTER the missing session does not count a
    # missing edge before the dependency. No invented suspension row is needed.
    following=checker.span(50,300)
    assert following['dependency_calendar_complete']
    assert following['dependency_expected_sessions']==following['dependency_observed_sessions']


def test_future_calendar_extension_does_not_change_policy_or_historical_evidence(bars,pack):
    future=pd.bdate_range(bars.date.iloc[-1]+pd.Timedelta(days=1),periods=40).strftime('%Y-%m-%d').tolist()
    extended=build_native_evidence(bars,market_dates=[*bars.attrs['verified_market_dates'],*future])
    assert extended['config_hash']==pack['config_hash'] and extended['schema_hash']==pack['schema_hash']
    assert extended['events']==pack['events'] and extended['evaluations']==pack['evaluations']
    assert extended['calendar_provenance']==pack['calendar_provenance']


def test_actual_native_pattern_crossing_old_gap_never_recovers_as_a_new_fact():
    raw=_raw(650,varied=True);calendar=raw.attrs['verified_market_dates']
    sparse=raw.drop(index=raw.index[220]).copy()
    pack=build_native_evidence(sparse,market_dates=calendar)
    late=[e for e in pack['events'] if e['source']=='structure' and e.get('dependency_calendar_complete') is False
          and e['date']>calendar[341]]
    assert late  # compiled first-buy pattern outlives the 120-row quality veto
    dates=pack['frame'].date.dt.strftime('%Y-%m-%d').tolist()
    for event in late:
        i=dates.index(event['date'])
        assert pack['frame'].iloc[i].native_w_input_eligible  # current trailing data recovered
        assert not event['input_eligible'] and not event['candidate_eligible']
        assert event['dependency_expected_sessions']==event['dependency_observed_sessions']+1
        assert all(event['event_id'] not in e['structure']['active_buy_event_ids']+e['structure']['active_sell_event_ids']
                   for e in pack['evaluations'][i:])


def test_auxiliary_ma_dependency_cannot_use_the_same_old_gap_as_an_eligible_trigger():
    raw=_raw(650);t=np.arange(len(raw));price=20+3*np.sin(t*2*np.pi/70)+.02*np.sin(t*2*np.pi/8)
    raw['close']=price;raw['open']=price-.05;raw['high']=price+.2;raw['low']=price-.2
    raw['amount']=raw['volume']*price
    calendar=raw.attrs['verified_market_dates'];sparse=raw.drop(index=raw.index[220]).copy()
    pack=build_native_evidence(sparse,market_dates=calendar)
    observed=pack['frame'].iloc[342:]
    bad=observed[observed.native_w_input_eligible & ~observed.native_auxiliary_calendar_complete]
    assert len(bad)>0
    assert not bad.native_auxiliary_input_eligible.any()
    assert (bad.native_auxiliary_expected_sessions==bad.native_auxiliary_observed_sessions+1).all()


@pytest.mark.parametrize('change',['duplicate','order','empty'])
def test_invalid_verified_calendar_fails_closed(change):
    raw=_raw(180);calendar=raw.attrs['verified_market_dates']
    invalid=calendar+[calendar[-1]] if change=='duplicate' else list(reversed(calendar)) if change=='order' else []
    with pytest.raises(ValueError,match='calendar'):
        build_native_evidence(raw,market_dates=invalid)


def test_w_full_pending_export_original_clock_and_permanent_specific_invalidation():
    raw = _raw(180)
    _set_bar(raw, 140, op=10.5, hi=11.8, lo=10.4, close=11.7, volume=2_000_000.)
    _set_bar(raw, 141, op=11.1, hi=11.5, lo=11.0, close=11.4, volume=700_000.)
    _set_bar(raw, 142, op=11.3, hi=11.8, lo=11.2, close=11.6, volume=700_000.)
    _set_bar(raw, 143, op=11.2, hi=11.3, lo=10.6, close=10.8, volume=800_000.)
    _set_bar(raw, 144, op=11.0, hi=11.7, lo=10.9, close=11.5, volume=800_000.)
    p = build_native_evidence(raw)
    sos = next(item for item in p["wyckoff_events"] if item["event"] == "SOS" and item["date"] == _day(raw, 140))
    lps = next(item for item in p["wyckoff_events"] if item["event"] == "LPS" and item["date"] == _day(raw, 141))
    assert not sos["candidate_eligible"] and not sos["confirmed"]
    assert lps["candidate_eligible"] and lps["confirmed"]
    assert lps["support_low"] == lps["invalidation_price"] == 11.
    assert lps["origin_event_id"] == sos["event_id"]
    assert lps["stop_price"] == 11.
    assert lps["risk_stop_price"] < lps["stop_price"]
    carry = p["evaluations"][142]["wyckoff"]
    support = next(item for item in carry["support_facts"] if item["event_id"] == lps["event_id"])
    assert support["observed_at"] == _day(raw, 141) + "T15:00:00+08:00"
    assert support["snapshot_at"] == _day(raw, 142) + "T15:00:00+08:00"
    assert support["event_strength"] == .95
    assert carry["current_state"]["continuous_direction_valid"]
    for i in range(143, len(raw)):
        current = p["evaluations"][i]["wyckoff"]
        assert lps["event_id"] in current["invalidated_event_ids"]
        assert lps["event_id"] not in current["active_buy_event_ids"]


def _day(raw, i):
    return raw.iloc[i].date.strftime("%Y-%m-%d")


def test_daily_volume_state_has_two_channels_even_without_a_range_event():
    raw = _raw(180)
    p = build_native_evidence(raw)
    row = p["evaluations"][150]["wyckoff"]
    assert row["input_eligible"]
    assert row["current_state"]["buy_score"] > 0
    assert row["current_state"]["sell_score"] >= 0
    assert row["current_state"]["event_demand_support"] == row["current_state"]["event_supply_support"] == 0
    assert p["frame"].iloc[150]["native_events_at_day"] == []


def test_rules_are_independent_of_old_ma_structure_and_final_decisions(bars, pack):
    changed = bars.copy()
    changed["ma20"] = 1e99; changed["rule_buy"] = True; changed["rule_sell"] = False
    changed["input_eligible"] = False; changed["structure_low"] = -1e99
    changed["native_bs2_side"] = -1; changed["final_side"] = "SELL"
    result = build_native_evidence(changed)
    columns = [key for key in pack["frame"] if key.startswith("native_")]
    pd.testing.assert_frame_equal(result["frame"][columns], pack["frame"][columns], check_exact=True)
    assert result["events"] == pack["events"]


def test_input_quality_failure_is_unavailable_and_cannot_resurrect_events(bars, pack):
    bad = bars.copy(); cut = 410
    bad.loc[bad.index[cut], "source"] = "not_verified"
    result = build_native_evidence(bad)
    assert not result["frame"].iloc[cut].native_structure_input_eligible
    assert not result["frame"].iloc[cut].native_w_input_eligible
    assert pd.isna(result["frame"].iloc[cut].native_w_demand_score)
    assert result["evaluations"][cut]["structure"]["reason_codes"] == ["native_structure_input_ineligible"]
    assert "wyckoff_unverified_source_window" in result["evaluations"][cut]["wyckoff"]["reason_codes"]
    for source in ("structure", "wyckoff"):
        original_ids = {item["event_id"] for item in result["evaluations"][cut - 1][source]["support_facts"]}
        for item in result["evaluations"][cut:]:
            assert not original_ids.intersection(fact["event_id"] for fact in item[source]["support_facts"])


def test_changed_config_has_changed_schema_and_does_not_mutate_config():
    config = {"native_evidence": {"continuous_direction_threshold": .4}}
    before = copy.deepcopy(config)
    changed = build_native_evidence(_raw(125), config)
    assert changed["schema_hash"] != NATIVE_SCHEMA_HASH
    assert before == config
    wyckoff = build_native_evidence(_raw(125), {"wyckoff": {"events": {"test_volume_ratio": .7}}})
    assert wyckoff["schema_hash"] != NATIVE_SCHEMA_HASH
    strategy = strategy_config(); strategy["min_bi_len"] = 7
    structural = build_native_evidence(_raw(125), {"czsc_strategy": strategy})
    assert structural["schema_hash"] != NATIVE_SCHEMA_HASH


@pytest.mark.parametrize("value", [-1, float("nan"), 2])
def test_invalid_strength_setting_fails(value):
    with pytest.raises(ValueError):
        build_native_evidence(_raw(), {"native_evidence": {"continuous_direction_threshold": value}})


def test_empty_pack_has_exact_explicit_column_types():
    empty = build_native_evidence(pd.DataFrame())
    assert empty["events"] == empty["evaluations"] == empty["wyckoff_events"] == []
    assert empty["schema_hash"] == NATIVE_SCHEMA_HASH
    assert str(empty["frame"].native_w_demand_score.dtype) == "float64"
    assert str(empty["frame"].native_w_snapshot.dtype) == "object"
    assert str(empty["frame"].native_w_input_eligible.dtype) == "bool"
