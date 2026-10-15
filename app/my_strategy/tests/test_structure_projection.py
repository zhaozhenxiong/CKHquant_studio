from copy import deepcopy

from my_strategy.web_dashboard.structure_projection import structure_projection
from my_strategy.tests.test_reference_analysis_ui import run_js


def test_source_projection_preserves_evidence_and_removes_uninitialized_targets():
    event = {"action": "SELL", "source": "structure", "reference_layer": "source",
             "date": "2026-07-31", "reference_price": 31.2, "event_id": "source-s"}
    row = {"date": "2026-10-14", "position_reference": {"target_position_fraction": 0},
           "agents": {"ma": {"status": "model_unavailable"}, "structure": {"sell_strength": .5}},
           "source_reference_points": [event], "reference_points": [{"action": "SELL"}]}
    original = {"research": {"calendar_run_id": "calendar"}, "position_reference": {"target": 0},
                "next_session_decision": row, "chart_evidence": {"rows": [row]},
                "source_reference_points": [event], "trend_consensus": {"rows": [row]}}
    before = deepcopy(original)
    result = structure_projection(original)
    assert original == before
    assert result["source_reference_points"][0]["event_id"] == "source-s"
    assert "position_reference" not in result
    assert "position_reference" not in result["next_session_decision"]
    assert "ma" not in result["chart_evidence"]["rows"][0]["agents"]
    assert not result["capabilities"]["reference_position"]
    assert result["structure_context"]["calendar_run_id"] == "calendar"


def test_source_popup_shows_evidence_without_model_or_zero_position_claim():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-10-14',structure_only:true,reference_close:37.74,events:[]};
const evidence={structure_only:true,rows:[row]};
const html=ui.popupHtml({evidence,consensus:{rows:[row]},groups:[],barDate:row.date,frequency:'日线'});
assert(html.includes('结构与量价来源'));
assert(html.includes('37.74'));
for(const text of ['缺兼容模型','趋势概率未评分','参考目标','0%'])assert(!html.includes(text),text);
assert.equal(ui.positionHtml(row),'');
const unavailable=ui.referencePositionSummary({stale:true,target_position_fraction:0,previous_reference_target:0});
assert(unavailable.includes('尚未建立有效参考目标'));
assert(!unavailable.includes('0%'));
''')
