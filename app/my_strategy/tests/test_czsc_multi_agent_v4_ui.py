"""Exercise shipped evidence contracts across native source and display periods."""
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "web_dashboard/static/js"


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped display contract checks")
    result = subprocess.run([node, "-e", script, str(STATIC)], capture_output=True,
                            text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_full_wyckoff_events_preserve_pending_and_confirmed_first_visibility():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const w=require(path.join(process.argv[1],'wyckoff_markers.js'));
const events=[
 {event_id:'pending',event:'SOS',direction:'BUY',observed_at:'2026-06-02T15:00:00+08:00',available_at:'2026-06-02T15:00:00+08:00',confirmed:false,status:'pending_test'},
 {event_id:'confirmed',event:'LPS',direction:'BUY',origin_observed_at:'2026-06-04T15:00:00+08:00',first_available_at:'2026-06-04T15:00:00+08:00',available_at:'2026-06-25T15:00:00+08:00',confirmed:true},
 {event_id:'future',event:'LPS',observed_at:'2026-06-05T15:00:00+08:00',available_at:'2026-06-05T15:00:00+08:00',confirmed:true},
 {event_id:'badclock',event:'Test',observed_at:'2026-06-04T16:00:00+08:00',available_at:'2026-06-04T15:00:00+08:00',confirmed:true},
];
const args={events,bars:['2026-06-02','2026-06-04','2026-06-05'].map(time=>({time})),asOf:'2026-06-04'};
const out=w.buildDetailed(args);
assert.equal(out.eventCount,2);assert.equal(out.markers.length,2);
assert(out.markers.find(item=>item.time==='2026-06-02').text.includes('SOS待测试'));
assert(out.markers.find(item=>item.time==='2026-06-04').text.includes('LPS确认'));
assert.equal(out.groups.find(item=>item.event_type==='LPS').entries[0].row.first_available_at,'2026-06-04T15:00:00+08:00');
assert.equal(w.buildDetailed({...args,events:[]}).markers.length,0);
''')


def test_higher_period_wyckoff_aggregates_same_type_but_keeps_original_dates_and_ids():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const w=require(path.join(process.argv[1],'wyckoff_markers.js'));
const event=(event_id,event,day,confirmed=true)=>({event_id,event,observed_at:day+'T15:00:00+08:00',available_at:day+'T15:00:00+08:00',confirmed});
const original=[event('one','Upthrust','2026-06-23'),event('two','Upthrust','2026-06-24'),event('three','SOS','2026-06-24',false)];
const events=[...original,original[0],{...original[1],event_id:'other',symbol:'600519.SH'}];
for(const frequency of ['周线','月线']) {
 const out=w.buildDetailed({events,bars:[{time:'2026-06-25',is_closed:false}],frequency,asOf:'2026-06-25',symbol:'002436.SZ'});
 assert.equal(out.eventCount,3);assert.equal(out.groups.length,2);assert.equal(out.formingProjectedCount,3);
 const supply=out.groups.find(item=>item.event_type==='Upthrust');
 assert.equal(supply.period_status,'forming');assert.equal(supply.event_count,2);
 assert.deepEqual(supply.entries.map(item=>item.event_date),['2026-06-23','2026-06-24']);
 assert(out.markers.find(item=>item.id===supply.id).text.includes('×2'));
 const ui=require(path.join(process.argv[1],'chart_evidence.js'));
 const popup=ui.popupHtml({groups:[supply],barDate:'2026-06-25',frequency,evidence:{rows:[]}});
 assert(popup.includes('W Upthrust确认'));assert(popup.includes('日线事件已可知／周期形成中'));
 assert(popup.includes('data-evidence-date="2026-06-23"'));assert(popup.includes('原观察 2026-06-24T15:00:00+08:00'));
 assert(!popup.includes('>S</span>'));assert.equal(ui.referenceSelection({groups:[supply],barDate:'2026-06-25',evidence:{rows:[]}}),null);
}
''')


def test_v4_daily_final_points_project_to_forming_period_while_native_signals_wait():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const m=require(path.join(process.argv[1],'chart_markers.js'));
const args={bars:[{time:'2026-10-09',is_closed:false}],dailyBars:[{time:'2026-10-09'}],frequency:'周线',asOf:'2026-10-09',symbol:'002436.SZ',up:'red',down:'green',consensusVersion:'czsc_agent_consensus_v4'};
const event={event_id:'point',source:'consensus',symbol:'002436.SZ',date:'2026-10-08',evidence_date:'2026-10-08',first_available_at:'2026-10-08T15:00:00+08:00',action:'BUY',source_frequency:'日线'};
const out=m.build({...args,events:[event]});
assert.equal(out.intentCount,1);assert.equal(out.formingProjectedCount,1);assert.equal(out.markers[0].text,'B');
assert.equal(out.groups[0].entries[0].first_available_at,event.first_available_at);
const native=m.build({...args,events:[{...event,source_frequency:'周线',source_period_date:'2026-10-09'}]});
assert.equal(native.intentCount,0);assert.equal(native.deferredCount,1);
const older=m.build({...args,consensusVersion:'czsc_agent_consensus_v2',events:[event]});
assert.equal(older.intentCount,0);assert.equal(older.deferredCount,1);
''')


def test_l2_seven_states_are_shared_and_null_does_not_become_zero_probability():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const states={no_candidate:'无候选',risk_blocked:'风险阻止',input_ineligible:'输入不合格',model_unavailable:'缺兼容模型',scored_rejected:'已评分 · 未过门槛',scored_passed:'已评分 · 通过门槛',identity_error:'评分身份核验失败'};
for(const [evaluation_status,label] of Object.entries(states)) {
 const p=evaluation_status==='scored_rejected'?0:evaluation_status==='scored_passed'?.4:null;
 const row={date:'2026-06-04',agents:{high_return:{kind:'model_probability',evaluation_status,probability:p,reason:'source reason',threshold:.1}}};
 const view=ui.currentAgentView(row,'high_return');
 assert.equal(view.evaluationStatus,evaluation_status);assert(view.summary.includes(label));
 const history=ui.historyHtml([row]),popup=ui.agentsHtml(row);
 assert(history.includes(label));assert(popup.includes(label));
 if(p===null){assert(!history.includes('0%'));assert(!popup.includes('高收益区间概率 0%'));}
 if(p===0){assert(history.includes('0%'));assert(popup.includes('高收益区间概率 0%'));}
}
// A risk veto keeps its real reason before any generic arbiter identity result.
assert.equal(ui.highReturnState({probability:null,status:'score_unavailable',reason:'candidate_risk_ineligible'},{status:'unavailable',reason_codes:['high_return_frozen_model_contract_unverified']}),'risk_blocked');
assert.equal(ui.highReturnState({probability:null,evaluation_status:'scored_passed'}),'identity_error');
assert.equal(ui.highReturnState({probability:1.5,evaluation_status:'scored_passed'}),'identity_error');
''')


def test_native_structure_and_current_volume_views_match_history_and_popup():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-06-04',agents:{ma:{probability:.8,in_scope:false},structure:{buy_count:0,sell_count:0},wyckoff:{current_demand:.7,current_supply:0,event_demand:.95,event_supply:0,buy_score:.95,sell_score:0},high_return:{probability:null,evaluation_status:'risk_blocked'}},
 candidate_evidence:{facts:[{event_id:'a',source:'structure',side:'BUY',direction_eligible:true,candidate_eligible:false},{event_id:'b',source:'structure',side:'BUY',direction_eligible:true,candidate_eligible:false}],risk_details:[{event_id:'a',risk_reason_codes:['stop_distance_exceeds_frozen_maximum']},{event_id:'b',risk_reason_codes:['stop_distance_exceeds_frozen_maximum']}]}};
const decision={version:'czsc_agent_consensus_v4',agents:{structure:{judgment:'BUY',origin_event_ids:['still-valid'],configured_weight:.2,buy_contribution:.2},high_return:{status:'risk_blocked',configured_weight:.1,threshold:.1}}};
const popup=ui.agentsHtml(row,false,decision),history=ui.historyHtml([{...row,consensus:decision}]);
assert(popup.includes('当日方向事实：买 2 · 卖 0'));assert(popup.includes('原始入场资格通过 0 条'));
assert(history.includes('原始 2 / 0'));assert(history.includes('风险通过买入 0'));assert(history.includes('有效历史支持 1'));
assert(popup.includes('当日需求 70 / 100 · 当日供给 0 / 100'));assert(history.includes('当日需求 70 / 100 · 当日供给 0 / 100'));
assert(popup.includes('事件衰减强度：需求 95 / 100'));assert(history.includes('事件衰减强度：需求 95 / 100'));
assert(popup.includes('质量权重 10 分'));assert(popup.includes('当前仲裁质量门槛 10%'));
assert(popup.includes('范围外研究分 80%'));assert(history.includes('范围外研究分'));
const counts={visible_window:{bars:183,source_events:108,wyckoff_events:16,final_points:1},full_prefix:{bars:2719,source_events:821,wyckoff_events:122,final_points:3}};
assert(ui.historyHtml([],0,'',counts).includes('显示窗口：日线 183 · 来源事实 108 · 完整量价事件 16 · 最终 B/S 1'));
assert(ui.historyHtml([],0,'',counts).includes('完整预热前缀：日线 2,719'));
''')


def test_zero_neutral_rule_scores_are_available_and_quality_weight_is_not_a_probability_threshold():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const neutral={date:'2026-10-09',agents:{wyckoff:{kind:'rule_evidence',input_eligible:true,current_demand:0,current_supply:0,event_demand:0,event_supply:0}}};
const neutralView=ui.currentAgentView(neutral,'wyckoff',{judgment:'ABSTAIN',status:'not_supported'});
assert.equal(neutralView.status,'available');assert(neutralView.summary.includes('当日需求 0 / 100 · 当日供给 0 / 100'));
const unavailable={date:neutral.date,agents:{wyckoff:{kind:'rule_evidence',input_eligible:false,current_demand:null,current_supply:null,buy_score:null,sell_score:null,reason_codes:['wyckoff_input_unavailable']}}};
const missing=ui.currentAgentView(unavailable,'wyckoff',{judgment:'UNAVAILABLE',status:'input_ineligible'});
assert.equal(missing.status,'input_unavailable');assert(missing.summary.includes('不可用'));assert(!missing.summary.includes('0 / 100'));
const row={date:neutral.date,agents:{high_return:{kind:'model_probability',probability:.12,evaluation_status:'scored_rejected',threshold:.07}}};
const final={version:'czsc_agent_consensus_v4',agents:{high_return:{status:'rejected',configured_weight:.1,threshold:.2,model_threshold:.07,quality_contribution:0}}};
const html=ui.agentsHtml(row,false,final);
assert(html.includes('高收益区间概率 12% · 已评分 · 未过门槛'));
assert(html.includes('质量权重 10 分'));assert(html.includes('买入质量加分 0 分'));
assert(html.includes('当前仲裁质量门槛 20%'));assert(html.includes('原模型冻结门槛 7%'));
assert(!html.includes('当前仲裁质量门槛 10%'));
''')


def test_model_picker_respects_explicit_entry_capabilities_and_keeps_undeclared_legacy_models():
    run_js(r'''
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('node:assert/strict');
const elements=new Map();function element(id){if(!elements.has(id))elements.set(id,{value:'',innerHTML:'',dataset:{}});return elements.get(id);}
global.document={getElementById:element};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);',
 'globalThis.picker={state,researchModels,populateResearchModels};'));
const native={run_id:'native-v4',capabilities:{chart:true,scan:true,legacy_backtest:false}};
const old={run_id:'legacy-no-capabilities'};
const scanOnly={run_id:'scan-only',capabilities:{chart:false,scan:true,legacy_backtest:false}};
picker.state.research={high_return:{models:[native,old,scanOnly]}};
for(const prefix of ['analysis','scan','backtest'])element(prefix+'-model-family').value='high_return';
assert.deepEqual(picker.researchModels('analysis').map(m=>m.run_id),['native-v4','legacy-no-capabilities']);
assert.deepEqual(picker.researchModels('scan').map(m=>m.run_id),['native-v4','legacy-no-capabilities','scan-only']);
assert.deepEqual(picker.researchModels('backtest').map(m=>m.run_id),['legacy-no-capabilities']);
element('backtest-model').value='native-v4';picker.populateResearchModels('backtest');
assert.equal(element('backtest-model').value,'');assert(!element('backtest-model').innerHTML.includes('native-v4'));
assert(element('backtest-model').innerHTML.includes('legacy-no-capabilities'));
element('analysis-model').value='native-v4';picker.populateResearchModels('analysis');
assert.equal(element('analysis-model').value,'native-v4');
picker.state.research.high_return.default_consensus_version='czsc_agent_consensus_v4';
native.schema_family='high_return_native_v4';
const nativeByVersion={run_id:'native-control',strategy_version:'czsc_high_return_v4_native',capabilities:{chart:true,scan:true,legacy_backtest:false}};
picker.state.research.high_return.models.push(nativeByVersion);
assert.deepEqual(picker.researchModels('analysis').map(m=>m.run_id),['native-v4','native-control']);
assert.deepEqual(picker.researchModels('scan').map(m=>m.run_id),['native-v4','native-control']);
assert.deepEqual(picker.researchModels('backtest').map(m=>m.run_id),['legacy-no-capabilities']);
element('analysis-model').value='legacy-no-capabilities';picker.populateResearchModels('analysis');
assert.equal(element('analysis-model').value,'');
delete picker.state.research.high_return.default_consensus_version;
assert(picker.researchModels('analysis').some(m=>m.run_id==='legacy-no-capabilities'));
// Other families without new declarations retain their existing options.
picker.state.research.dual={models:[{run_id:'old-dual'}]};element('backtest-model-family').value='dual';
assert.deepEqual(picker.researchModels('backtest').map(m=>m.run_id),['old-dual']);
''')


def test_native_guidance_separates_trade_eligibility_from_input_status_and_translates_pure_events():
    run_js(r'''
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('node:assert/strict');
global.document={getElementById:()=>({value:''})};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);',
 'globalThis.guidance={ruleQualificationText,readableEvent,highReturnEvidence};'));
const base={eligible:false,candidate_evidence:{version:'czsc_candidate_evidence_v4',facts:[]},agents:{high_return:{status:'no_candidate'}},signal_points:[]};
assert.equal(guidance.ruleQualificationText(base),'候选资格：本日无新候选');
assert(!guidance.ruleQualificationText(base).includes('输入不足'));
assert(guidance.ruleQualificationText({...base,agents:{high_return:{status:'input_ineligible'}}}).includes('输入不足'));
assert(guidance.ruleQualificationText({...base,agents:{high_return:{status:'candidate_risk_ineligible'}}}).includes('冻结风险规则阻止'));
assert(guidance.ruleQualificationText({...base,source_candidate_eligible:true,risk_eligible:true,agents:{high_return:{status:'scored'}}}).includes('冻结风险资格通过'));
assert(guidance.ruleQualificationText({...base,signal_points:[{side:'SELL',direction_eligible:true}]}).includes('独立卖出证据触发'));
for (const [event_type,label] of Object.entries({pure_first_buy_buy:'纯结构一买',pure_first_sell_sell:'纯结构一卖',pure_second_bs_buy:'纯结构二买',pure_second_bs_sell:'纯结构二卖',pure_third_buy_buy:'纯结构三买',structure_invalidation:'已确认结构支撑失效'})) {
 assert.equal(guidance.readableEvent({source:'structure',event_type}),`CZSC结构 · ${label}`);
}
// Legacy eligible continues to describe its old input contract.
assert.equal(guidance.ruleQualificationText({model_family:'high_return',eligible:false}),'候选资格：输入不足');
const nativeNote=guidance.highReturnEvidence(base);
assert(nativeNote.includes('新版通过独立 Broker 验收入口'));assert(nativeNote.includes('旧版账户回测仍选择旧版模型'));
assert(!guidance.highReturnEvidence({model_family:'high_return'}).includes('新版通过独立 Broker 验收入口'));
''')


def test_scored_models_and_valid_rules_do_not_claim_missing_scores_when_reasons_are_empty():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-06-04',agents:{
 ma:{kind:'model_probability',probability:.5952436327934265,reason_codes:[]},
 wyckoff:{kind:'rule_evidence',input_eligible:true,current_demand:.1531,current_supply:0,event_demand:1,event_supply:0,reason_codes:[]}
}};
for(const key of ['ma','wyckoff']) {
 assert.equal(ui.agentView(row.agents[key],key).reason,'');
 const view=ui.currentAgentView(row,key);
 assert(!view.reason.includes('该时点没有可用评分'));
}
assert.equal(ui.agentView({kind:'model_probability',probability:0,reason_codes:[]},'ma').reason,'');
assert.equal(ui.agentView({kind:'rule_evidence',input_eligible:true,buy_score:0,sell_score:0,reason_codes:[]},'wyckoff').reason,'');
assert(ui.agentView({kind:'model_probability',probability:null,reason_codes:[]},'ma').reason.includes('没有可用评分'));
assert(ui.agentView({kind:'rule_evidence',input_eligible:false,reason_codes:[]},'wyckoff').reason.includes('没有可用评分'));
assert(ui.agentView({kind:'model_probability',probability:.6,reason_codes:['ma_tactical_weakening_is_not_confirmed_long_trend_failure']},'ma').reason.includes('短期转弱'));
''')


def test_popup_overview_separates_real_scores_missing_scores_weights_and_actual_contributions():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-01-21',agents:{ma:{probability:.4658,status:'scored',target:'actual_complete_fee_net_roundtrip_positive'},
 structure:{input_eligible:true,current_state:{buy_score:1,sell_score:0}},wyckoff:{input_eligible:true,current_demand:.7,current_supply:.1},
 high_return:{probability:.12,evaluation_status:'scored_passed'}}};
const decision={date:row.date,version:'czsc_agent_consensus_v4',action:'BUY',emitted:true,agents:{
 ma:{judgment:'ABSTAIN',configured_weight:.4,threshold:.55,buy_contribution:0,sell_contribution:0},
 structure:{judgment:'BUY',configured_weight:.2,buy_contribution:.2,sell_contribution:0},
 wyckoff:{judgment:'BUY',configured_weight:.3,buy_contribution:.3,sell_contribution:0},
 high_return:{judgment:'BUY',configured_weight:.1,quality_contribution:.1,buy_contribution:.1,sell_contribution:0}
}};
const html=ui.agentOverviewHtml(row,decision);
const section=key=>html.match(new RegExp(`<tr data-agent-summary-row="${key}">([\\s\\S]*?)</tr>`))[1];
assert.equal((html.match(/data-agent-summary-row=/g)||[]).length,4);
assert(section('ma').includes('已评分 46.58%'));assert(section('ma').includes('40%'));assert(section('ma').includes('买 0 / 卖 0 分'));
assert(section('ma').includes('门槛 55%'));assert(!section('ma').includes('未评分'));
assert(section('structure').includes('支持买入'));assert(section('structure').includes('规则买 100 / 100 · 卖 0 / 100'));
assert(section('wyckoff').includes('当日需求 70 / 100 · 当日供给 10 / 100'));
assert(section('high_return').includes('已评分 · 通过门槛'));assert(section('high_return').includes('买 10 / 卖 0 分'));assert(section('high_return').includes('买入质量加分'));
assert(html.includes('MA 目标：完整交易费用后获利概率'));assert(!html.includes('趋势入场概率'));
const missing=ui.agentOverviewHtml({...row,agents:{...row.agents,ma:{probability:null,status:'model_unavailable'},high_return:{probability:null,evaluation_status:'no_candidate'}}},null);
assert(missing.includes('未评分 · 缺兼容模型'));assert(missing.includes('无候选 · 未评估'));assert(missing.includes('买 — / 卖 — 分'));
assert(!missing.includes('已评分 0%'));assert(!missing.includes('买 0 / 卖 0 分'));assert(!missing.includes('完整交易费用后获利概率'));
const zero=ui.agentOverviewHtml({...row,agents:{...row.agents,ma:{probability:0,status:'scored'}}},decision);
assert(zero.includes('已评分 0%'));
const out=ui.agentOverviewHtml({...row,agents:{...row.agents,ma:{probability:.6627,status:'scored',in_scope:false}}},decision);
assert(out.includes('已评分 66.27%'));assert(out.includes('范围外研究分 · 不参与当前投票'));
const popup=ui.popupHtml({evidence:{rows:[row]},consensus:{rows:[decision]},barDate:row.date,frequency:'月线',selectedDate:row.date});
assert(popup.indexOf('信号可知日')<popup.indexOf('四路评分总览'));
assert(popup.indexOf('四路评分总览')<popup.indexOf('仲裁意见'));
assert(popup.includes('class="agent-evidence-grid"')); // The detailed cards remain.
''')
