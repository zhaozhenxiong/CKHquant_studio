"""Display aggregation preserves causal events, score state and saved identity."""
import shutil
import subprocess
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[1] / "web_dashboard/static/js"


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped chart evidence checks")
    result = subprocess.run([node, "-e", script, str(STATIC)], capture_output=True,
                            text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_compact_daily_events_keep_direction_confirmation_and_original_identity():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const markers=require(path.join(process.argv[1],'wyckoff_markers.js'));
const base={symbol:'002436.SZ',observed_at:'2026-07-01T15:00:00+08:00',available_at:'2026-07-01T15:00:00+08:00'};
const events=[{...base,event_id:'spring-1',event:'Spring',confirmed:false},{...base,event_id:'test-1',event:'Test',confirmed:true},
  {...base,event_id:'supply-1',event:'Upthrust',confirmed:true},
  {...base,event_id:'spring-1',event:'Spring',confirmed:false},
  {...base,event_id:'future',event:'SOS',available_at:'2026-07-02T15:00:00+08:00'},
  {...base,event_id:'foreign',event:'LPS',symbol:'600000.SH'}];
const unchanged=JSON.stringify(events);
const result=markers.buildDetailed({events,bars:[{time:'2026-07-01'},{time:'2026-07-02'}],symbol:base.symbol,asOf:'2026-07-01'});
assert.equal(JSON.stringify(events),unchanged);assert.equal(result.eventCount,3);assert.equal(result.markers.length,2);
const buy=result.groups.find(group=>group.side==='BUY'),sell=result.groups.find(group=>group.side==='SELL');
assert.equal(buy.event_count,2);assert.equal(buy.confirmed_count,1);assert.equal(buy.pending_count,1);assert.equal(buy.status,'mixed');
assert.deepEqual(buy.event_types,['Spring','Test']);assert.deepEqual(new Set(buy.entries.map(entry=>entry.event_id)),new Set(['spring-1','test-1']));
assert.equal(sell.confirmed_count,1);assert.equal(sell.pending_count,0);
const buyMarker=result.markers.find(marker=>marker.id===buy.id),sellMarker=result.markers.find(marker=>marker.id===sell.id);
assert.equal(buyMarker.text,'2');assert.equal(buyMarker.shape,'circle');assert.equal(sellMarker.shape,'square');
assert.equal(buyMarker.position,'belowBar');assert.equal(sellMarker.position,'aboveBar');assert.notEqual(buyMarker.color,sellMarker.color);
assert.equal(sellMarker.text,'');assert(!result.markers.some(marker=>/Spring|Test|Upthrust|确认|待测试/.test(marker.text)));
''')


def test_compact_higher_period_popup_lists_every_event_without_future_projection():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
global.KHQuantMarkers=require(path.join(process.argv[1],'chart_markers.js'));
const markers=require(path.join(process.argv[1],'wyckoff_markers.js')),ui=require(path.join(process.argv[1],'chart_evidence.js'));
const event=(name,id,date,confirmed)=>({symbol:'002436.SZ',event:name,event_id:id,confirmed,
  observed_at:date+'T15:00:00+08:00',first_available_at:date+'T15:00:00+08:00'});
const events=[event('Spring','original-spring','2026-07-01',false),event('Test','later-test','2026-07-02',true),event('SOS','future','2026-07-06',false)];
for(const frequency of ['周线','月线']) {
  const result=markers.buildDetailed({events,bars:[{time:'2026-07-03',is_closed:false}],frequency,symbol:'002436.SZ',asOf:'2026-07-03'});
  assert.equal(result.eventCount,2);assert.equal(result.markers.length,1);assert.equal(result.markers[0].text,'2');
  assert.equal(result.groups[0].period_status,'forming');assert.equal(result.formingProjectedCount,2);
  const html=ui.popupHtml({evidence:{rows:[{date:'2026-07-02',agents:{}}]},groups:result.groups,barDate:'2026-07-03',frequency});
  assert(html.includes('本根K线原始量价事件 2 条'));assert(html.includes('W Spring · 需求 · 待测试'));
  assert(html.includes('W Test · 需求 · 已确认'));assert(html.includes('已确认 1／待测试 1'));
  assert(html.includes('original-spring'));assert(html.includes('later-test'));assert(!html.includes('>事件标识 future'));
  assert(!html.includes('2026-07-06'));assert(html.includes('日线事件已可知／周期形成中'));
  assert.deepEqual(ui.groupDates(result.groups),['2026-07-01','2026-07-02']);
}
''')


def test_agent_details_are_open_use_correct_identity_and_explain_unscored_and_zero():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path'),ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-09-15',agents:{
  ma:{kind:'model_probability',p_up:.31,p_down:.25,p_neutral:.44,target:'next_verified_open_horizon_direction',
    model_identity:{model_run_id:'saved-run',checkpoint:'prior-head',model_head:'trend_direction',available_at:'2026-06-30',
      train_label_end:'2026-03-31',actual_training_completed_at:'2026-10-12T15:00:00+08:00',target_contract_hash:'target-hash',
      manifest_sha256:'manifest-hash',feature_schema_hash:'schema-hash',source_snapshot_hash:'source-hash',historical_reconstruction:true}},
  structure:{kind:'rule_evidence',input_eligible:true,version:'native-structure-v4',snapshot_id:'structure<saved>',
    available_at:'2026-09-15T15:00:00+08:00',current_state:{dependency_start:'2026-06-01',dependency_end:'2026-09-15'}},
  wyckoff:{kind:'rule_evidence',input_eligible:true,event_demand:0,event_supply:0,current_demand:.1187,current_supply:.2122,
    effective_demand:.1187,effective_supply:.2122},
  high_return:{kind:'model_probability',status:'risk_blocked',evaluation_status:'risk_blocked',probability:null,threshold:.10,
    reason_codes:['risk_stop_distance_outside_contract']}}};
const before=JSON.stringify(row),html=ui.agentsHtml(row);
assert.equal(JSON.stringify(row),before);assert.equal((html.match(/<details open data-agent-details=/g)||[]).length,4);
assert.equal((html.match(/<summary>评分目标与模型身份/g)||[]).length,2);
assert.equal((html.match(/<summary>规则口径与数据身份/g)||[]).length,2);
assert(!/<summary>[^<]*<\/summary><\/details>/.test(html));
assert(html.includes('schema-hash'));assert(html.includes('source-hash'));assert(html.includes('训练完成：2026-10-12'));
assert(html.includes('历史重建评分'));assert(html.includes('冻结交易日窗口内的上涨／下跌／中性方向'));
assert(html.includes('structure&lt;saved&gt;'));assert(!html.includes('structure<saved>'));
assert(html.includes('native-structure-v4'));assert(html.includes('行情依赖区间：2026-06-01～2026-09-15'));
assert(html.includes('规则版本未保存'));assert(html.includes('本日未提供模型身份'));
assert(html.includes('当时有效的事件强度两侧均为0'));assert(html.includes('由价量单独计算'));
assert(html.includes('最终有效需求 11.87 / 100'));assert(html.includes('最终有效供给 21.22 / 100'));
assert(html.includes('未评分不代表0%概率')===false);assert(html.includes('因此未进行高收益入场评分'));
assert(html.includes('止损距离不符合冻结风险契约'));assert(!html.includes('高收益区间概率 0%'));
const empty=ui.agentsHtml({date:'2026-09-15',agents:{high_return:{status:'no_candidate',probability:null}}});
assert(empty.includes('未评分不代表0%概率'));assert(empty.includes('评分目标未保存'));
assert(!ui.currentAgentView({agents:{wyckoff:{event_demand:null,event_supply:null}}},'wyckoff').explanation.includes('两侧均为0'));
assert.equal(ui.ruleScore(null),'不可用');assert.equal(ui.ruleScore(0),'0 / 100');
assert.equal(ui.highReturnState({status:'actual_existing_position'}),'actual_existing_position');
assert.equal(ui.highReturnState({reason_codes:['no_qualified_direction_candidate']}),'no_candidate');
const objective=ui.agentsHtml({agents:{ma:{kind:'model_probability',probability:.4,target:{horizon_sessions:10,classes:['up','down','neutral']}}}});
assert(objective.includes('horizon_sessions'));assert(!objective.includes('[object Object]'));
''')
