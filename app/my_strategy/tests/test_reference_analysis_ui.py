"""Reference display preserves causal B/S evidence without borrowing account state."""
from pathlib import Path
import shutil
import subprocess

import pytest


STATIC = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js'


def run_js(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for shipped reference display checks')
    result = subprocess.run([node, '-e', script, str(STATIC)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_reference_chart_keeps_causal_points_and_rejects_foreign_or_future_records():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const markers=require(path.join(process.argv[1],'chart_markers.js'));
const point={event_id:'reference-buy',symbol:'000001.SZ',date:'2026-10-09',
  first_available_at:'2026-10-09T15:00:00+08:00',available_at:'2026-10-09T15:00:00+08:00',
  action:'BUY',side:'BUY',reference_only:true,reference_price:23.45,reference_price_date:'2026-10-09'};
const result={symbol:'000001.SZ',analysis_mode:'reference',as_of:'2026-10-09',reference_points:[point,
  {...point,event_id:'foreign',symbol:'600000.SH'},
  {...point,event_id:'future',first_available_at:'2026-10-12T15:00:00+08:00'},
  {...point,event_id:'account-intent',reference_only:false}],
  account_actions:{context:{account_id:'unrelated-account'},trades:[{symbol:'000001.SZ',action:'BUY',date:'2026-10-09'}]}};
const selected=ui.referenceMarkerInputs(result);
assert.equal(selected.events.length,1);assert.equal(selected.events[0].source,'reference_consensus');
assert.equal('trades' in selected,false);
const grouped=markers.build({bars:[{time:'2026-10-09',is_closed:false}],dailyBars:[{time:'2026-10-09'}],
  frequency:'月线',events:selected.events,symbol:result.symbol,consensusVersion:'trend_consensus',asOf:result.as_of,up:'red',down:'green'});
assert.equal(grouped.intentCount,1);assert.equal(grouped.formingProjectedCount,1);
assert.equal(grouped.markers[0].shape,'arrowUp');assert.equal(grouped.markers[0].color,'red');
assert(markers.describe(grouped.mappings,'2026-10-09','月线').includes('最终共识参考 2026-10-09'));
const row={symbol:'000001.SZ',date:'2026-10-09',analysis_mode:'reference',reference_only:true,reference_action:'BUY',action:'BUY',emitted:true,
  buy_base_score:.64,sell_base_score:.08,reference_close:23.45,reference_price_date:'2026-10-09',
  available_at:point.available_at,reason_codes:['candidate_not_confirmed_or_observable'],
  agents:{ma:{kind:'trend_probability',status:'model_unavailable',p_up:.9,p_down:.05,p_neutral:.05},
    structure:{kind:'rule_evidence',status:'scored',buy_strength:.6,sell_strength:0},
    wyckoff:{kind:'rule_evidence',status:'scored',buy_score:.7,sell_score:.1},
    high_return:{kind:'model_probability',probability:.99}},
  account_action:{action:'BUY',shares:100,emitted:true},holding:{shares:100},broker_fills:[{price:99}]};
result.chart_evidence={rows:[row]};result.trend_consensus={version:'trend_consensus',rows:[row]};
const evidence=ui.referenceEvidence(result),clean=evidence.rows[0];
assert.equal('accountActionHtml' in ui,false);assert.equal('accountMarkerInputs' in ui,false);
assert.equal(ui.currentAgentView(clean,'ma').exit,'');
assert.equal(ui.trendProbabilities(clean.agents.ma).available,false);
const popup=ui.popupHtml({evidence,consensus:result.trend_consensus,groups:grouped.groups,barDate:'2026-10-09',frequency:'月线'});
for(const unexpected of ['账户观察','账户买入','持仓退出概率','高收益二阶段','99%','上涨 90%'])assert(!popup.includes(unexpected),unexpected);
assert(popup.includes('三路参考评分'));assert(popup.includes('收盘参考价 23.45 元'));assert(popup.includes('非成交价'));
assert(popup.includes('买方参考'));assert(popup.includes('买方支持 64 分'));
assert(popup.includes('当日参考依据已可见'));assert(!popup.includes('候选未确认或尚不可知'));
assert(ui.opinionReasons({source:'structure',confirmed:false,date:row.date,available_at:point.available_at,
  reason_codes:['candidate_not_confirmed_or_observable']}).includes('候选未确认或尚不可知'),
  'an observable reference must never confirm an unfinished native candidate');
assert(ui.opinionReasons({...row,first_available_at:'2026-10-12T15:00:00+08:00',as_of:'2026-10-09'})
  .includes('候选未确认或尚不可知'),'future reference status stays blocked');
assert(ui.opinionReasons({...row,as_of:'2026-10-09T14:59:59+08:00'})
  .includes('候选未确认或尚不可知'),'same-day future publication stays blocked');
const selection=ui.referenceSelection({evidence,consensus:result.trend_consensus,groups:grouped.groups,barDate:'2026-10-09'});
assert.equal(selection.price,23.45);assert.equal(selection.signal_date,'2026-10-09');
const history=ui.historyHtml(ui.mergeEvidenceRows(evidence.rows,result.trend_consensus.rows));
assert(history.includes('来源 B / S'));assert(history.includes('最终共识 / 原因'));
for(const unexpected of ['高收益概率','账户最终','成交执行记录','拒单记录','风险通过买入','上涨 90%'])assert(!history.includes(unexpected),unexpected);
// Explicit simulations retain their account marker and display contract.
''')


def test_reference_market_history_omits_account_and_high_return_metrics_and_filters():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const market=require(path.join(process.argv[1],'market_history.js'));
const day={date:'2026-10-09',universe_size:2,counts:{direction_buy:1,direction_sell:1,candidates:1,new_candidates:1,
  account_buy:1,account_sell:1,holding:2,filled_buy:2,filled_sell:1,hr_scored_passed:2,failed:0,missing_price:0},
  coverage:{price_available:2,input_available:2,model_scored:0,failed:0,missing_price:0},events:{},mean_support:{}};
const summary=market.normalize({analysis_mode:'reference',dates:[day.date],daily:[day],universe_size:2,
  sectors:[],event_types:[],notes:[]});
const state={date:day.date,mode:'count',metric:'direction_buy',sector:'',event:'',filterState:'',page:1};
const html=market.viewHtml(summary,state);
for(const unexpected of ['账户 B','账户 S','实际研究持仓','高收益评分','研究账户按每股','filled_buy','hr_scored_passed'])assert(!html.includes(unexpected),unexpected);
assert(html.includes('B / S 参考方向'));assert(html.includes('趋势模型评分'));
const facts=market.factsHtml({rows:[{symbol:'000001.SZ',direction:'SELL',price_available:true,holding_shares:200,
  quality_status:'scored_passed',high_return_probability:.99,model_scored:false,account_action:'SELL'}],total:1,page:1,page_size:25},summary,state);
assert(facts.includes('趋势模型状态'));assert(facts.includes('缺少合格模型 · 未评分'));
for(const unexpected of ['持仓 200','高收益状态','99.00%','B 意图','S 意图'])assert(!facts.includes(unexpected),unexpected);
const simulated=market.viewHtml({...summary,analysis_mode:'simulation'},state);
assert(!simulated.includes('账户 B'));assert(!simulated.includes('高收益评分'));
''')


def test_reference_model_picker_excludes_failed_or_archived_candidates():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),vm=require('vm');
global.window={addEventListener(){}};
const source=fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);',
 'globalThis.referencePicker={state,researchModels};');
vm.runInThisContext(source);
referencePicker.state.research={active_model:{run_id:'qualified'},models:[
 {run_id:'qualified',status:'completed',model_family:'current',automatic_selection_eligible:true,archived:false},
 {run_id:'not-passed',status:'completed',model_family:'current',automatic_selection_eligible:false,archived:false},
 {run_id:'archived',status:'completed',model_family:'current',automatic_selection_eligible:true,archived:true}]};
for(const prefix of ['analysis','scan']) assert.deepEqual(referencePicker.researchModels(prefix).map(item=>item.run_id),['qualified']);

''')


def test_reference_probability_risk_and_predictive_gate_are_independent():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const day='2026-09-24';
const row={symbol:'600519.SH',date:day,analysis_mode:'reference',reference_only:true,
  reference_close:1237,available_at:day+'T15:00:00+08:00',action:'WAIT',direction:'UNAVAILABLE',
  position_reference:{schema:'reference_position_v1',status:'stale',stale:true,date:day,
    target_position_fraction:0,previous_reference_target:0,change_fraction:0,
    probability_scored:true,risk_geometry_available:false,predictive_gate_passed:false,
    reason_codes:['reference_risk_geometry_unavailable','reference_risk_boundary_breached','reference_predictive_gate_pending']},
  reference_risk_plan:{status:'unavailable',reference_price:1237,prior_low20:1248.1,
    atr14:18.442142857142876,atr_stop_price:1190.8946428571428,
    diagnostic_stop_price:1248.1,diagnostic_stop_distance_fraction:-.00897332255,
    stop_price:null,risk_position_cap_fraction:null,
    reason_codes:['reference_risk_boundary_breached']},
  agents:{ma:{status:'scored',probability_scored:true,risk_geometry_available:false,
    predictive_gate_passed:false,p_up:.2854868173599243,p_down:.41692468523979187,p_neutral:.2975884974002838,
    model_identity:{name:'2026Q3',research_reconstruction:true,
      split:{test_start:'2026-07-01',test_end:'2026-09-30'},
      predictive_gate:{passed:false,log_loss:1.1500904560089111,baseline_log_loss:1.1112407445907593,
        brier:.6971314549446106,baseline_brier:.6769112944602966}},reason_codes:[]},
    structure:{status:'scored'},wyckoff:{status:'scored'}},source_reference_points:[]};
assert.equal(ui.trendProbabilities(row.agents.ma).available,true);
const summary=ui.agentOverviewHtml(row),position=ui.positionHtml(row),details=ui.agentsHtml(row);
for(const expected of ['上涨 28.55%','下跌 41.69%','中性 29.76%','方向概率已评分','风险计划不可用','预测基线未通过']) assert(summary.includes(expected),expected);
for(const expected of ['1,248.1','1,190.89','边界距离 -0.9%','候选边界的诊断值','不增加参考目标']) assert(position.includes(expected),expected);
assert(!summary.includes('缺兼容模型'));assert(!position.includes('缺失评分不视作下跌'));
assert(details.includes('检查点：2026Q3'));assert(details.includes('历史留出覆盖：2026-07-01 至 2026-09-30'));
assert(!details.includes('本日未提供模型身份'));
assert(ui.historyHtml([row]).includes('上涨 28.55%'));
assert.equal(row.action,'WAIT');assert.equal(row.position_reference.change_fraction,0);
const archived={...row,position_reference:{schema:'trend_consensus'},agents:{...row.agents,
  ma:{...row.agents.ma,status:'model_unavailable'}}};
assert.equal(ui.trendProbabilities(archived.agents.ma).available,false);
''')


def test_unavailable_probability_states_never_expose_residual_numbers():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const base={kind:'trend_probability',p_up:.6,p_down:.1,p_neutral:.3};
for(const status of ['model_unavailable','models_disabled','fixed_fold_unavailable','input_ineligible','probability_invalid','score_unavailable']) {
  assert.equal(ui.trendProbabilities({...base,status,probability_scored:true}).available,false,status);
}
assert.equal(ui.trendProbabilities({...base,status:'scored',probability_scored:false}).available,false);
for(const bad of [{p_up:NaN},{p_up:1.1},{p_neutral:.2}]) assert.equal(ui.trendProbabilities({...base,status:'scored',...bad}).available,false);
const labels={models_disabled:'未启用模型',fixed_fold_unavailable:'固定检查点不覆盖该日',input_ineligible:'输入不合格',probability_invalid:'概率核验失败',model_unavailable:'缺当日可用模型'};
for(const [status,label] of Object.entries(labels)) {
  const row={position_reference:{schema:'reference_position_v1'},agents:{ma:{...base,status}}};
  const html=ui.agentOverviewHtml(row);assert(html.includes(label),label);assert(!html.includes('上涨 60%'));
}
''')


def test_missing_target_session_has_no_current_probability_or_risk_claim():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-10-14',analysis_mode:'reference',reference_close:null,
  agents:{ma:{status:'input_ineligible',probability_scored:false,p_up:null,p_down:null,p_neutral:null,
    model_identity:{},reason_codes:['missing_target_market_session']}},
  reference_risk_plan:{status:'unavailable',reason_codes:['missing_target_market_session']},
  position_reference:{schema:'reference_position_v1',date:'2026-10-14',status:'stale',stale:true,
    target_position_fraction:.1,previous_reference_target:.1,change_fraction:0,
    probability_scored:false,predictive_gate_passed:false,risk_geometry_available:false,
    last_scored_date:'2026-10-13',reason_codes:['missing_target_market_session']}};
const html=ui.positionHtml(row);assert(html.includes('缺当日行情'));assert(html.includes('不沿用前日评分冒充本日判断'));
assert(html.includes('最近有效参考 2026-10-13'));assert(!html.includes('预测基线未通过'));
assert(!ui.agentOverviewHtml(row).includes('方向概率已评分'));assert(html.includes('本日无新调整'));
''')


def test_current_reference_consensus_does_not_label_probabilities_as_rule_support():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={analysis_mode:'reference',reference_only:true,action:'WAIT',buy_base_score:.2854868173599243,
  sell_base_score:.41692468523979187,agents:{ma:{status:'scored',probability_scored:true,
    p_up:.2854868173599243,p_down:.41692468523979187,p_neutral:.2975884974002838}},
  position_reference:{schema:'reference_position_v1',status:'stale',stale:true,
    target_position_fraction:0,previous_reference_target:0,change_fraction:0,risk_geometry_available:false},
  reason_codes:['reference_quote_ohlc_invalid']};
const current=ui.consensusHtml(row);
assert(current.includes('上涨 28.55%'));assert(current.includes('参考买卖点与每股仓位'));
assert(current.includes('风险计划不可用'));assert(!current.includes('买方支持 28.55 分'));
assert(!current.includes('规则支持分'));assert(!current.includes('方向支持不足'));
const archived=ui.consensusHtml({...row,position_reference:{schema:'trend_consensus'}},null,'trend_consensus');
assert(archived.includes('买方支持 28.55 分'));assert(archived.includes('规则支持分不代表盈利概率'));
assert.equal(ui.opinionReasons(row),'当日 OHLC 价格不合法，无法核验风险失效');
''')
