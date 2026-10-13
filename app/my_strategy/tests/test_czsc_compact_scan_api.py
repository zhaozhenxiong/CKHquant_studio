"""An opt-in transport view cannot change saved decisions or drill identities."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from my_strategy.storage.czsc_results import ResultStore
from my_strategy.web_dashboard.api import compact_scan_view, router


def scan_fixture(version="czsc_trend_consensus_v2"):
    model = {"model_run_id": "final-model", "model_policy": "auto", "model_fold": None,
             "strategy": version, "manifest_sha256": "actual-manifest", "target_contract_hash": "actual-target",
             "target": "actual_complete_fee_net_roundtrip_high_return_and_initial_risk",
             "checkpoint": "prior-head", "historical_reconstruction": True,
             "model_input": {"raw_features": [1.] * 1000}}
    row = {"symbol": "002436.SZ", "name": "兴森科技", "date": "2026-10-09", "data_end": "2026-10-09",
           "version": version, "schema_family": version, "model_family": "high_return",
           "category": "buy", "action": "BUY", "direction": "BUY", "position_intent": "BUY",
           "reason": "original searchable reason", "signal": "original searchable event",
           "reason_codes": ["continuous_consensus_buy"], "entry_plan": {"plan_id": "actual-plan", "entry_allowed": True,
               "signal_date": "2026-10-09", "price_floor": 20., "price_ceiling": 21., "stop_price": 19., "target_weight": .2},
           "original_opportunity_ids": ["original-1"], "high_return_probability": .25,
           "agents": {"ma": {"p_up": .6, "p_down": .2, "p_neutral": .2, "kind": "model_probability",
               "status": "scored", "target": "next_verified_open_horizon_direction", "model_identity": model},
               "high_return": {"probability": .25, "kind": "model_probability", "status": "scored",
               "evaluation_status": "scored_passed", "threshold": .1, "target": model["target"],
               "label_target": {"min_net_return": .08, "min_r_multiple": 2.}, "model_identity": model}},
           "model_resolution": {"model_family": "high_return", "schema_family": version, "probability": .25,
               "model_run_id": "final-model", "reason_codes": [], "target": model["target"], "model_identity": model,
               "target_contract": {"full_raw_training_data": [2.] * 1000}},
           "account_context": {"account_id": "research:stock-1", "symbol": "002436.SZ", "strategy": version,
               "account_start": "2023-01-01", "initial_cash": 100000., "account_kind": "research_broker",
               "calendar_hash": "verified-calendar-hash", "source_hash": "actual-source", "config_hash": "frozen-config",
               "model_identity": model, "consensus": {"large_raw_contract": [0.] * 1000}},
           "account_action": {"action": "BUY", "emitted": True, "shares": 0, "cash": 100000.,
               "account_id": "research:stock-1", "series_context_hash": "actual-series", "entry_evaluation": {"probability": .25,
                   "evaluation_status": "scored_passed", "model_identity": model, "reference_state": {"raw": [0.] * 1000}},
               "exit_evaluation": {"probability": None, "reason": "actual_holding_required"},
               "direction_decision": {"duplicated_complete_row": [0.] * 1000}},
           "candidate_evidence": [{"event_id": "original-1", "source": "structure", "side": "BUY", "event_type": "pure_first_buy_buy",
               "anchor_at": "2026-10-08", "available_at": "2026-10-09T15:00:00+08:00", "direction_eligible": True,
               "candidate_eligible": True, "raw_model_inputs": [0.] * 1000}],
           "active_candidates": [{"complete_event_prefix": [0.] * 1000}],
           "entry_attempt": {"attempt_id": "actual-attempt", "economic_group_id": "economic-original", "original_event_ids": ["original-1"],
               "pre_gate_state": {"large_actual_state": [0.] * 1000}}}
    return {"run_id": "saved-scan", "strategy_version": version, "model_family": "high_return", "model_run_id": "final-model",
            "run_context": {"config_hash": "saved-config", "stocks": [row["symbol"]]}, "calendar_run_id": "verified-calendar",
            "data_range": {"start": "2026-01-01", "end": "2026-10-09"}, "coverage": {"requested": 1, "successful": 1},
            "category_counts": {"buy": 1}, "consensus_config": {"version": version}, "publication_allowed": False,
            "market_history": {"run_id": "saved-scan", "dates": ["2026-10-09"], "universe_size": 1}, "rows": [row], "failures": []}


@pytest.fixture
def client(tmp_path):
    store = ResultStore(tmp_path / "results.db", tmp_path / "runs")
    app = FastAPI(); app.include_router(router); app.state.tasks = SimpleNamespace(results=store)
    with TestClient(app) as value:
        yield value, store


@pytest.mark.parametrize("version", ["czsc_trend_consensus", "czsc_trend_consensus_v2", "czsc_agent_consensus_v4"])
def test_opt_in_preserves_full_default_report_and_saved_bytes(client, version):
    client, store = client; source = scan_fixture(version); store.save("scan", source)
    path = store.runs_root / "saved-scan/reports/czsc.json"; original = path.read_bytes()
    assert client.get("/api/runs/saved-scan").json() == source
    response = client.get("/api/runs/saved-scan?view=scan"); assert response.status_code == 200
    view = response.json(); row = view["rows"][0]
    for key in ("run_id", "strategy_version", "model_run_id", "calendar_run_id", "run_context", "coverage", "market_history", "consensus_config"):
        assert view[key] == source[key]
    for key in ("symbol", "name", "date", "data_end", "category", "reason", "signal", "entry_plan", "original_opportunity_ids"):
        assert row[key] == source["rows"][0][key]
    assert row["account_context"]["account_id"] == "research:stock-1"
    assert row["account_action"]["emitted"] is True and row["account_action"]["shares"] == 0
    assert row["entry_attempt"]["attempt_id"] == "actual-attempt"
    assert row["agents"]["ma"]["model_identity"]["manifest_sha256"] == "actual-manifest"
    assert row["agents"]["high_return"]["target"] == source["rows"][0]["agents"]["high_return"]["target"]
    assert "target_contract" not in row["model_resolution"] and "model_input" not in row["agents"]["ma"]["model_identity"]
    assert "direction_decision" not in row["account_action"] and "active_candidates" not in row
    assert "pre_gate_state" not in row["entry_attempt"]
    assert view["account_start"] == "2023-01-01" and view["account_initial_cash"] == 100000.
    assert view["model_policy"] == "auto" and view["scan_view"]["account_context_status"] == "consistent"
    assert path.read_bytes() == original
    assert client.get("/api/runs/saved-scan").json() == source


@pytest.mark.parametrize("kind", ["analysis", "backtest", "czsc-trend-consensus-v2-train"])
def test_compact_view_rejects_other_kinds_without_altering_default(client, kind):
    client, store = client; source = {"run_id": "other-run", "rows": [{"raw": [1., 2.]}], "kind": kind}
    store.save(kind, source)
    assert client.get("/api/runs/other-run").json() == source
    assert client.get("/api/runs/other-run?view=scan").status_code == 422


def test_illegal_view_and_missing_run_are_rejected(client):
    client, store = client; store.save("scan", scan_fixture())
    assert client.get("/api/runs/saved-scan?view=unknown").status_code == 422
    assert client.get("/api/runs/absent?view=scan").status_code == 404


@pytest.mark.parametrize("change", ["mixed_cash", "missing_context", "top_conflict", "request_conflict", "parameters_conflict"])
def test_unbound_context_is_explicit_and_never_fabricates_shared_account(change):
    source = scan_fixture(); source["rows"].append(copy.deepcopy(source["rows"][0]))
    source["rows"][1]["symbol"] = "000333.SZ"
    source["rows"][1]["account_context"].update(symbol="000333.SZ", account_id="research:stock-2")
    if change == "mixed_cash": source["rows"][1]["account_context"]["initial_cash"] = 200000.
    elif change == "missing_context": source["rows"][1].pop("account_context")
    elif change == "top_conflict": source["account_start"] = "2026-01-01"
    elif change == "request_conflict": source["request"] = {"account_initial_cash": 200000.}
    else: source["run_context"]["config"] = {"parameters": {"initial_cash": 200000.}}
    before = copy.deepcopy(source); view = compact_scan_view(source)
    assert view["scan_view"]["account_context_status"] in {"mixed", "unavailable"}
    assert "account_id" not in view and "account_context" not in view
    assert source == before


def test_common_account_is_available_to_saved_request_without_mutation():
    source=scan_fixture();source['request']={'model_run_id':'final-model','model_policy':None,'account_start':None}
    source['account_initial_cash']=None;before=copy.deepcopy(source);view=compact_scan_view(source)
    assert view['request']['account_start']=='2023-01-01' and view['request']['account_initial_cash']==100000.
    assert view['request']['model_policy']=='auto' and view['account_initial_cash']==100000.
    assert source==before


def test_current_trend_and_high_return_display_are_identical():
    node = shutil.which("node")
    if not node: pytest.skip("Node is required for actual display contract comparison")
    source = scan_fixture()
    row = source["rows"][0]
    row["agents"]["structure"] = {"kind": "rule_evidence", "status": "available",
        "current_state": {"support_floor": 19., "resistance_ceiling": 23.,
            "current_valid_zone": {"support_low": 18., "resistance_high": 20.},
            "native_labels": {"third_buy": "三买"}, "raw_inputs": [1.] * 1000}}
    row["agents"]["wyckoff"] = {"kind": "rule_evidence", "status": "available",
        "input_eligible": True, "current_demand": .24, "current_supply": .31,
        "event_demand": 0., "event_supply": 0., "effective_demand": .24, "effective_supply": .31}
    # Older saved evidence uses this qualification field without direction_eligible.
    row["candidate_evidence"].append({"event_id": "legacy-sell", "source": "structure",
        "side": "SELL", "candidate_direction_valid": True, "raw_features": [0.] * 1000})
    view = compact_scan_view(source)
    script = """
const fs=require('fs'); const ui=require(process.argv[1]);
const rows=JSON.parse(fs.readFileSync(0,'utf8'));
for (const key of ['ma','high_return','structure','wyckoff']) {
  if (JSON.stringify(ui.currentAgentView(rows[0],key))!==JSON.stringify(ui.currentAgentView(rows[1],key)))
    throw new Error('display changed: '+key);
}
if (!ui.currentAgentView(rows[1],'structure').summary.includes('卖 1'))
  throw new Error('legacy directional sell qualification was lost');
if (!ui.currentAgentView(rows[1],'structure').secondary.includes('纯结构 三买'))
  throw new Error('saved structure state was lost');
"""
    path = Path(__file__).parents[1] / "web_dashboard/static/js/chart_evidence.js"
    result = subprocess.run([node, "-e", script, str(path)], input=json.dumps([source["rows"][0], view["rows"][0]]),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('kind,status', [('scan','succeeded'),('scan','running'),('scan','failed'),('backtest','succeeded')])
def test_task_view_is_opt_in_and_only_projects_successful_scans(client, kind, status):
    client, store = client
    source = scan_fixture(); store.save('scan', source)
    saved = {'job_id':'job-1','kind':kind,'status':status,'run_id':source['run_id'],
             'spec':{'model_run_id':'final-model','model_policy':'auto'},
             'result':source if status=='succeeded' else None}
    before=copy.deepcopy(saved); client.app.state.tasks.get=lambda job_id:saved
    assert client.get('/api/tasks/job-1').json()==before
    response=client.get('/api/tasks/job-1?view=scan');assert response.status_code==200
    if kind=='scan' and status=='succeeded':
        body=response.json(); assert body['spec']==before['spec'] and body['run_id']==before['run_id']
        assert body['result']['scan_view']['compact'] and body['result']['account_start']=='2023-01-01'
        assert body['result']['rows'][0]['account_context']['account_id']=='research:stock-1'
    else:assert response.json()==before
    assert saved==before and client.get('/api/tasks/job-1').json()==before
    assert client.get('/api/tasks/job-1?view=unknown').status_code==422
