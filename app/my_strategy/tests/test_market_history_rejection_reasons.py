"""Saved market facts expose validated L2 veto reasons in the stock detail UI."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from my_strategy.services.czsc_market_history import stock_facts
from my_strategy.storage.czsc_market_history import MarketHistoryStore, market_history_facts


DAY = "2026-09-03"
STATIC = Path(__file__).resolve().parents[1] / "web_dashboard/static/js"


def decision(*, reasons=None, model_reasons=("high_return_below_threshold", "high_return_below_threshold")):
    return {"date": DAY, "symbol": "002436.SZ", "direction": "BUY", "reason_codes": reasons or [],
        "active_candidates": [{"side": "BUY", "event_id": "candidate"}],
        "agents": {"high_return": {"status": "scored", "evaluation_status": "scored_rejected",
            "probability": .025664368644356728, "threshold": .1}},
        "model_resolution": {"checkpoint": "2025Q4", "probability": .025664368644356728,
            "reason_codes": list(model_reasons) if model_reasons is not None else None}}


def test_l2_veto_reason_survives_fact_storage_and_is_visible_in_stock_details(tmp_path):
    row = decision()
    unchanged = copy.deepcopy(row)
    fact = stock_facts("002436.SZ", [DAY], overlay={"rows": [row]})[0]
    assert fact["reason_codes"] == ["high_return_below_threshold"]
    assert fact["quality_status"] == "scored_rejected"
    assert fact["high_return_probability"] == row["agents"]["high_return"]["probability"]
    assert fact["counts"]["hr_scored_rejected"] and fact["counts"]["direction_buy"]
    assert row == unchanged, "display aggregation cannot mutate the saved evaluation"
    database = tmp_path / "results.db"
    store = MarketHistoryStore(database)
    store.begin("rejection-review", {"dates": [DAY]})
    store.save_stock("rejection-review", [fact])
    store.finish("rejection-review", {"dates": [DAY], "universe_size": 1})
    saved = market_history_facts("rejection-review", date=DAY, state="hr_scored_rejected", db_path=database)
    assert saved["rows"][0]["reason_codes"] == ["high_return_below_threshold"]
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for saved rejection detail rendering")
    script = r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'market_history.js'));
const data=JSON.parse(process.argv[2]),before=JSON.stringify(data);
const html=ui.factsHtml(data,{event_types:[]},{page:1});
assert(html.includes('已评分 · 未通过'));
assert(html.includes('高收益分数低于门槛'));
assert.equal((html.match(/高收益分数低于门槛/g)||[]).length,1);
assert.equal(JSON.stringify(data),before);
// Earlier saved runs have a blank top-level reason channel while retaining
// their validated quality reason. Rendering must work without rewriting them.
const legacy={...data,rows:data.rows.map(row=>({...row,reason_codes:[]}))};
const legacyBefore=JSON.stringify(legacy),legacyHtml=ui.factsHtml(legacy,{event_types:[]},{page:1});
assert(legacyHtml.includes('高收益分数低于门槛'));
assert.equal((legacyHtml.match(/高收益分数低于门槛/g)||[]).length,1);
assert.equal(JSON.stringify(legacy),legacyBefore);
'''
    result = subprocess.run([node, "-e", script, str(STATIC), json.dumps(saved, ensure_ascii=False)],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_direction_and_quality_reasons_merge_in_order_without_duplicates():
    row = decision(reasons=["opportunity_cooldown", "risk_plan_unavailable", "opportunity_cooldown"],
        model_reasons=["high_return_below_threshold", "opportunity_cooldown", "buy_total_support_below_threshold"])
    fact = stock_facts("002436.SZ", [DAY], overlay={"rows": [row]})[0]
    assert fact["reason_codes"] == ["opportunity_cooldown", "risk_plan_unavailable",
        "high_return_below_threshold", "buy_total_support_below_threshold"]
    assert fact["model_resolution"] == row["model_resolution"]


@pytest.mark.parametrize("reasons,model_reasons,expected", [
    (["no_live_unconsumed_opportunity"], [], ["no_live_unconsumed_opportunity"]),
    ([], [], []),
    ([], None, []),
])
def test_empty_reason_channels_do_not_fabricate_vetoes(reasons, model_reasons, expected):
    fact = stock_facts("002436.SZ", [DAY], overlay={"rows": [decision(
        reasons=reasons, model_reasons=model_reasons)]})[0]
    assert fact["reason_codes"] == expected


def test_failed_or_missing_quote_facts_keep_their_actual_unavailable_reason():
    failed = stock_facts("002436.SZ", [DAY], overlay={"rows": [decision()]}, failure={})[0]
    missing = stock_facts("002436.SZ", [DAY], overlay={"rows": []})[0]
    assert failed["reason_codes"] == ["scan_failed"]
    assert missing["reason_codes"] == ["missing_market_session_price"]
    assert failed["high_return_probability"] is missing["high_return_probability"] is None
