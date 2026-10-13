"""Independent review using real native evidence and trend candidate producers."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from my_strategy.services.czsc_market_history import MarketHistoryBuilder, stock_facts
from my_strategy.services.czsc_native_evidence_v4 import build_native_evidence_v4
from my_strategy.services.czsc_trend_consensus import prepare_trend_decisions


def causal_bars():
    n = 175
    close = 10 + np.choose(np.arange(n) % 4, [-.4, .4, .2, -.2])
    volume = 1_000_000. + (np.arange(n) % 7) * 10_000.
    frame = pd.DataFrame({"symbol": "000001.SZ", "date": pd.bdate_range("2020-01-02", periods=n),
        "open": close - .05, "high": np.full(n, 11.), "low": np.full(n, 9.),
        "close": close, "volume": volume, "amount": close * volume,
        "source": "sina_unadjusted_v1", "has_trade_price": 1})
    for index, values in {
        140: (10.5, 11.8, 10.4, 11.7, 2_000_000.),
        141: (11.1, 11.5, 11., 11.4, 700_000.),
        142: (11.3, 11.8, 11.2, 11.6, 700_000.),
        143: (11.2, 11.3, 10.6, 10.8, 800_000.),
        144: (11., 11.7, 10.9, 11.5, 800_000.),
    }.items():
        op, high, low, cl, vol = values
        frame.loc[index, ["open", "high", "low", "close", "volume", "amount"]] = [op, high, low, cl, vol, cl * vol]
    frame.attrs.update(price_basis="unadjusted", volume_unit="shares", amount_unit="CNY")
    frame.attrs["verified_market_dates"] = frame.date.dt.strftime("%Y-%m-%d").tolist()
    return frame


def native_overlay(frame):
    calendar = frame.attrs["verified_market_dates"]
    pack = build_native_evidence_v4(frame, market_dates=calendar)
    prepared = prepare_trend_decisions(pack["frame"], calendar, native_pack=pack, mode="E0")
    return {**prepared, "native_pack": pack}, pack


@pytest.fixture(scope="module")
def produced():
    raw = causal_bars()
    overlay, pack = native_overlay(raw)
    return raw, overlay, pack


def test_real_native_pending_confirmed_and_permanent_invalidation_are_not_backfilled(produced):
    raw, overlay, pack = produced
    dates = raw.date.dt.strftime("%Y-%m-%d").tolist()
    facts = stock_facts("000001.SZ", dates[140:146], overlay=overlay)
    sos = next(event for event in pack["wyckoff_events"] if event["event_type"] == "SOS" and event["date"] == dates[140])
    lps = next(event for event in pack["wyckoff_events"] if event["event_type"] == "LPS" and event["date"] == dates[141])
    assert lps["origin_event_id"] == sos["event_id"]
    pending = next(event for event in facts[0]["new_events"] if event["id"] == sos["event_id"])
    assert pending["confirmed"] is False
    assert sos["event_id"] not in {event["id"] for event in facts[0]["active_events"]}
    assert lps["event_id"] not in {event["id"] for event in facts[0]["new_events"]}
    confirmed = next(event for event in facts[1]["new_events"] if event["id"] == lps["event_id"])
    assert confirmed["confirmed"] is True
    assert confirmed["available_at"] == dates[141] + "T15:00:00+08:00"
    assert lps["event_id"] in {event["id"] for event in facts[1]["active_events"]}
    assert lps["event_id"] in {event["id"] for event in facts[2]["active_events"]}
    assert facts[3]["invalidated_events"] >= 1
    assert all(lps["event_id"] not in {event["id"] for event in fact["active_events"]} for fact in facts[3:])


def test_real_native_future_append_and_future_price_change_preserve_historical_facts(produced):
    raw, full, _ = produced
    dates = raw.date.dt.strftime("%Y-%m-%d").tolist()[138:143]
    prefix, _ = native_overlay(raw.iloc[:143].copy())
    original = stock_facts("000001.SZ", dates, overlay=prefix)
    assert stock_facts("000001.SZ", dates, overlay=full) == original
    changed = raw.copy(deep=True)
    changed.loc[143:, ["open", "high", "low", "close"]] *= 3
    changed.loc[143:, ["volume", "amount"]] *= 5
    changed.loc[143:, "source"] = "unsafe_future_source"
    mutated, _ = native_overlay(changed)
    assert stock_facts("000001.SZ", dates, overlay=mutated) == original


def test_candidate_first_seen_uses_warmup_prefix_and_duplicate_events_keep_event_units(produced):
    raw, original, pack = produced
    dates = raw.date.dt.strftime("%Y-%m-%d").tolist()
    lps = next(event for event in pack["wyckoff_events"] if event["event_type"] == "LPS" and event["date"] == dates[141])
    overlay = {**original, "rows": copy.deepcopy(original["rows"]), "native_pack": copy.deepcopy({
        "evaluations": pack["evaluations"], "wyckoff_events": pack["wyckoff_events"]})}
    # Isolate one authentic opportunity so unrelated structure events do not
    # obscure whether this existing opportunity is new at the display boundary.
    for row in overlay["rows"]:
        row["active_candidates"] = [event for event in row["active_candidates"] if event["event_id"] == lps["event_id"]]
    facts = stock_facts("000001.SZ", dates[141:143], overlay=overlay)
    assert facts[0]["counts"]["new_candidates"]
    assert not facts[1]["counts"]["new_candidates"]
    later_window = stock_facts("000001.SZ", dates[142:143], overlay=overlay)
    assert later_window[0]["counts"]["candidates"]
    assert not later_window[0]["counts"]["new_candidates"], "display-window entry cannot create a new opportunity"
    overlay["native_pack"]["wyckoff_events"].append(copy.deepcopy(lps))
    overlay["native_pack"]["evaluations"][141]["wyckoff"]["support_facts"].append(copy.deepcopy(
        next(event for event in overlay["native_pack"]["evaluations"][141]["wyckoff"]["support_facts"] if event["event_id"] == lps["event_id"])))
    assert stock_facts("000001.SZ", dates[141:143], overlay=overlay) == facts
    builder = MarketHistoryBuilder(["000001.SZ"], dates, dates[142], run_id="review", window_sessions=2)
    builder.add_stock("000001.SZ", overlay=overlay)
    summary = builder.finish()
    event_type = next(event for event in summary["event_types"] if event["id"] == "LPS")
    assert event_type["new_total"] == 1, "new_total counts original event records"
    assert event_type["active_stock_days"] == 2, "active_stock_days counts involved stock/day pairs"
    assert summary["daily"][0]["counts"]["new_candidates"] == 1, "candidate counts use distinct stocks"


def test_failed_and_missing_stocks_keep_fixed_denominator_and_have_no_copied_direction(produced):
    raw, overlay, _ = produced
    dates = raw.date.dt.strftime("%Y-%m-%d").tolist()[141:144]
    builder = MarketHistoryBuilder(["000001.SZ", "missing", "failed"], dates, dates[-1], run_id="denominator")
    builder.add_stock("000001.SZ", overlay=overlay)
    missing = builder.add_stock("missing", overlay={"rows": []})
    failed = builder.add_stock("failed", failure={"error": "invalid source"})
    summary = builder.finish()
    assert summary["universe_size"] == summary["sectors"][0]["member_count"] == 3
    for day in summary["daily"]:
        assert day["coverage"]["price_available"] == day["coverage"]["missing_price"] == day["coverage"]["failed"] == 1
        assert day["buy_rate"] == day["counts"]["direction_buy"] / 3
        assert day["sell_rate"] == day["counts"]["direction_sell"] / 3
        assert day["counts"]["hr_unscored"] >= 2
    assert all(fact["direction"] == "UNAVAILABLE" and fact["quality_status"] == "missing_price"
        and fact["high_return_probability"] is None and not fact["new_events"] and not fact["active_events"] for fact in missing)
    assert all(fact["counts"]["failed"] and fact["quality_status"] == "failed"
        and fact["holding_shares"] is None and fact["high_return_probability"] is None for fact in failed)


def test_failure_record_without_error_text_still_has_consistent_failure_state():
    fact = stock_facts("failed", ["2026-10-12"], failure={})[0]
    assert fact["counts"]["failed"] and not fact["counts"]["missing_price"]
    assert fact["quality_status"] == "failed"
    assert fact["reason_codes"] == ["scan_failed"]
    assert not fact["price_available"] and fact["high_return_probability"] is None
