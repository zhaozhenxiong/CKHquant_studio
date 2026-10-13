"""Trend display keeps score targets, account identity and signal/fill clocks."""
import shutil
import subprocess
import os
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js'


def run_js(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for shipped trend strategy display')
    result = subprocess.run([node, '-e', script, str(STATIC)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_three_class_trend_scores_and_continuous_contributions_do_not_become_roi_or_binary_votes():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-06-04',agents:{ma:{kind:'trend_probability',p_up:.65,p_down:.15,p_neutral:.2,probability:.99,status:'scored'},
structure:{input_eligible:true,current_state:{buy_score:.7,sell_score:.1}},
wyckoff:{input_eligible:true,current_demand:.2,current_supply:.1,event_demand:.8,event_supply:0,effective_demand:.8,effective_supply:.1,support_basis:'confirmed_event'},
high_return:{probability:.11,evaluation_status:'scored_passed',target:'actual_complete_fee_net_roundtrip_high_return_and_initial_risk'}}};
const decision={version:'czsc_trend_consensus',date:row.date,action:'BUY',emitted:true,agents:{
 ma:{judgment:'BUY',configured_weight:.4,buy_contribution:.26,sell_contribution:.06},
 structure:{judgment:'BUY',configured_weight:.2,buy_contribution:.14,sell_contribution:.02},
 wyckoff:{judgment:'BUY',configured_weight:.3,buy_contribution:.24,sell_contribution:.03},
 high_return:{status:'passed',configured_weight:.1,quality_contribution:.011,sell_contribution:0}}};
const view=ui.currentAgentView(row,'ma',decision.agents.ma);
assert(view.summary.includes('上涨 65%'));assert(view.summary.includes('下跌 15%'));assert(view.summary.includes('中性 20%'));
assert(!view.summary.includes('99%'));
const overview=ui.agentOverviewHtml(row,decision);
assert(overview.includes('买 26 / 卖 6 分'));assert(overview.includes('买 1.1 / 卖 0 分'));
assert(overview.includes('最终有效需求 80 / 100'));assert(overview.includes('事件需求 80 / 100'));
assert(!overview.includes('完整交易费用后获利概率'));
const continuous={...decision,agents:{...decision.agents,
 ma:{status:'scored',buy_strength:.65,sell_strength:.15,configured_weight:.4,buy_contribution:.26,sell_contribution:.06},
 wyckoff:{status:'scored',buy_strength:.8,sell_strength:.1,configured_weight:.3,buy_contribution:.24,sell_contribution:.03}}};
const continuousOverview=ui.agentOverviewHtml(row,continuous);
assert(continuousOverview.includes('连续支持 买 65 / 100 · 卖 15 / 100'));
assert(continuousOverview.includes('连续支持 买 80 / 100 · 卖 10 / 100'));
assert(!continuousOverview.match(/data-agent-summary-row="(?:ma|wyckoff)"[^]*?<span class="subtle">不可用<\/span>/));
const quality=ui.currentAgentView(row,'high_return');assert.equal(quality.target,'完整交易费用后收益至少8%，且≥2倍冻结止损参考风险');
const detail=ui.agentsHtml(row,false,decision);assert(detail.includes('完整交易费用后收益至少8%，且≥2倍冻结止损参考风险'));assert(detail.includes('成交采用滑点模拟均价；R按冻结止损参考假设估计'));
assert.equal(row.agents.high_return.probability,.11);assert.equal(row.agents.high_return.target,'actual_complete_fee_net_roundtrip_high_return_and_initial_risk');
row.agents.high_return.label_target={min_net_return:.12,min_r_multiple:3};
assert.equal(ui.currentAgentView(row,'high_return').target,'完整交易费用后收益至少12%，且≥3倍冻结止损参考风险');
assert.equal(ui.highReturnTargetText(row.agents.high_return.label_target),'完整交易费用后收益至少12%，且≥3倍冻结止损参考风险');
const legacy=ui.currentAgentView({agents:{high_return:row.agents.high_return}},'high_return');assert.equal(legacy.target,row.agents.high_return.target);assert(!legacy.pricingNote);
assert(ui.opinionHtml({status:'scored',evaluation_status:'scored_passed',probability:.11},'high_return',{decision:{version:'czsc_trend_consensus'}}).includes('质量闸门：已评分 · 通过门槛'));
assert(ui.opinionHtml({status:'scored',evaluation_status:'scored_rejected',probability:.05},'high_return',{decision:{version:'czsc_trend_consensus'}}).includes('质量闸门：已评分 · 未过门槛'));
const zero=ui.currentAgentView({...row,agents:{...row.agents,ma:{kind:'trend_probability',p_up:0,p_down:.25,p_neutral:.75}}},'ma');
assert(zero.summary.includes('上涨 0%'));assert(!zero.summary.includes('未评分'));
const missing=ui.currentAgentView({...row,agents:{...row.agents,ma:{kind:'trend_probability',p_up:null,p_down:null,p_neutral:null,status:'model_unavailable'}}},'ma');
assert(missing.summary.includes('未评分'));assert(!missing.summary.includes('上涨 0%'));
assert(!ui.trendProbabilities({p_up:.65,p_down:.65,p_neutral:.2}).available);
const disabled={...row,agents:{...row.agents,high_return:{probability:null,status:'entry_quality_gate_disabled',evaluation_status:'entry_quality_gate_disabled',applied_to_entry:false,configured_weight:.1,reason_codes:['high_return_filter_disabled']}}};
const disabledView=ui.currentAgentView(disabled,'high_return');assert.equal(disabledView.evaluationStatus,'entry_quality_gate_disabled');assert.equal(disabledView.summary,'质量过滤已关闭，本次未评分');
assert(!disabledView.summary.includes('无候选'));assert(!disabledView.summary.includes('缺兼容模型'));assert(!disabledView.summary.includes('概率 0%'));
assert.equal(disabled.agents.high_return.probability,null);assert.equal(disabled.agents.high_return.configured_weight,.1);
assert(ui.historyHtml([disabled]).includes('data-evaluation-status="entry_quality_gate_disabled"'));assert(ui.historyHtml([disabled]).includes('质量过滤已关闭，本次未评分'));
assert(ui.opinionHtml(disabled.agents.high_return,'high_return',{decision:{version:'czsc_trend_consensus'}}).includes('质量闸门：质量过滤已关闭，本次未评分'));
assert.equal(ui.highReturnState({status:'entry_quality_gate_disabled',reason_codes:['high_return_filter_disabled'],probability:null}),'entry_quality_gate_disabled');
''')


def test_only_selected_account_intents_and_fills_enter_markers_and_foreign_same_symbol_records_are_rejected():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const bound={symbol:'002436.SZ',account_id:'research-a',series_context_hash:'series-a'};
const result={symbol:bound.symbol,trend_consensus:{version:'czsc_trend_consensus',events:[{...bound,date:'2026-01-27',action:'SELL'}]},account_actions:{
 context:{account_id:bound.account_id,account_kind:'research_broker_replay'},series_context_hash:bound.series_context_hash,
 intents:[{...bound,date:'2026-01-27',signal_at:'2026-01-27T15:00:00+08:00',intent_id:'exit',action:'SELL',reason_codes:['hard_stop']},
 {...bound,account_id:'other',intent_id:'foreign',date:'2026-01-27',action:'SELL'}],
 ledger:[{...bound,date:'2026-01-28',signal_date:'2026-01-27',action:'SELL',price:24.83},
 {...bound,series_context_hash:'stale',date:'2026-01-28',action:'SELL'}]}};
const input=ui.accountMarkerInputs(result,bound.symbol);
assert.equal(input.events.length,1);assert.equal(input.trades.length,1);assert.equal(input.rejectedRecords,2);
assert.equal(input.events[0].source,'account_action');assert.equal(input.events[0].event_id,'exit');
assert.equal(input.events[0].evidence_date,'2026-01-27');assert.equal(input.trades[0].date,'2026-01-28');
assert.equal(ui.accountMarkerInputs({...result,account_actions:null},bound.symbol).trades.length,0);
assert.equal(ui.accountMarkerInputs({...result,account_actions:{...result.account_actions,series_context_hash:'different'}},bound.symbol).events.length,0);
assert.equal(ui.accountMarkerInputs({...result,as_of:'2026-01-27'},bound.symbol).trades.length,0);
''')


def test_native_fact_list_keeps_direction_and_risk_counts_separate_without_assuming_missing_eligibility():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-01-15',candidate_evidence:[
 {event_id:'valid-sell',source:'structure',side:'SELL',candidate_direction_valid:true,candidate_eligible:true},
 {event_id:'risk-blocked-buy',source:'structure',side:'BUY',candidate_direction_valid:true,candidate_eligible:true,risk_candidate_eligible:false,risk_reason_codes:['invalid_causal_stop']},
 {event_id:'missing-eligibility',source:'structure',side:'BUY',candidate_eligible:true},
 {event_id:'explicit-invalid',source:'structure',side:'SELL',direction_eligible:false,candidate_direction_valid:true}],
 agents:{structure:{status:'scored',kind:'rule_evidence',current_state:{buy_score:.5,sell_score:1},input_eligible:true}},
 consensus:{version:'czsc_trend_consensus',agents:{structure:{status:'scored',buy_strength:1/3,sell_strength:2/3,origin_event_ids:['valid-sell']}}}};
assert.deepEqual(ui.structureFacts(row),{available:true,buy:1,sell:1,qualifiedBuy:0,riskRejected:1});
const history=ui.historyHtml([row]);assert(history.includes('原始 1 / 1'));assert(history.includes('风险通过买入 0'));
assert(history.includes('当前 连续支持 买 33.33 / 100 · 卖 66.67 / 100'));assert(!history.includes('当前 不可用'));
const cards=ui.agentsHtml(row,false,row.consensus,null,'czsc_trend_consensus');
assert(cards.includes('当前连续')||cards.includes('连续支持 买 33.33 / 100'));assert(!cards.includes('当日结构意见 不可用'));
''')


def test_strategy_picker_and_history_cache_use_complete_series_identity():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),vm=require('vm');
const elements=new Map();global.document={getElementById:id=>{if(!elements.has(id))elements.set(id,{value:'high_return',innerHTML:''});return elements.get(id);}};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);','globalThis.pick={state,researchModels,modelName,ruleQualificationText,resultResearchContext,storeHistoryEvidence,history:()=>historyRows};'));
pick.state.research={high_return:{default_consensus_version:'czsc_trend_consensus',models:[
 {run_id:'new',schema_family:'czsc_trend_consensus',capabilities:{chart:true,scan:true,legacy_backtest:true}},
 {run_id:'old',schema_family:'high_return_native_v4',capabilities:{chart:true,scan:true,legacy_backtest:false}}]}};
for(const prefix of ['analysis','scan','backtest'])assert.deepEqual(pick.researchModels(prefix).map(x=>x.run_id),['new']);
assert.equal(pick.modelName({schema_family:'czsc_trend_consensus'}),'趋势共识策略');
assert.equal(pick.ruleQualificationText({model_resolution:{schema_family:'czsc_trend_consensus'},agents:{ma:{kind:'model_probability'},high_return:{status:'no_candidate'}},eligible:false}),'候选资格：本日无新候选');
const automatic=pick.resultResearchContext({strategy_version:'czsc_trend_consensus',model_family:'high_return',model_run_id:'new',model_gate:{passed:false}});
assert.equal(automatic.model_policy,'auto');assert.equal(automatic.consensus_version,'czsc_trend_consensus');
const backtest=pick.resultResearchContext({strategy_version:'czsc_trend_consensus',model_family:'high_return',model_run_id:'new',model_gate:{passed:false},initial_cash:200000,per_stock_initial_cash:100000,data_range:{start:'2026-01-01'}});
assert.equal(backtest.account_start,'2026-01-01');assert.equal(backtest.account_initial_cash,100000);assert.equal(backtest.model_policy,'auto');
assert.equal(pick.resultResearchContext({model_family:'high_return',model_run_id:'old',model_gate:{}}).model_policy,'pinned');
pick.storeHistoryEvidence([{date:'2026-01-21',account_action:{account_id:'a'}}],false,'cash100-cost1');
pick.storeHistoryEvidence([{date:'2026-02-24',account_action:{account_id:'a'}}],true,'cash100-cost1');assert.equal(pick.history().length,2);
pick.storeHistoryEvidence([{date:'2026-02-24',account_action:{account_id:'b'}}],true,'cash200-cost2');
assert.equal(pick.history().length,1);assert.equal(pick.history()[0].account_action.account_id,'b');
''')


def test_signal_plan_risk_fraction_uses_only_signal_reference_and_frozen_stop_target():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const plan={status:'active',entry_allowed:true,reference_price:10,stop_price:9,target_weight:.8};
assert(ui.riskHtml(plan).includes('信号计划风险占比 8%'));
assert(ui.riskHtml({...plan,execution_price:500,fill_price:600}).includes('信号计划风险占比 8%'));
assert(ui.riskHtml({...plan,target_weight:0}).includes('信号计划风险占比 0%'));
assert(ui.riskHtml({...plan,reference_price:null,execution_price:10}).includes('信号计划风险占比 不可用'));
assert(ui.riskHtml({...plan,stop_price:11}).includes('信号计划风险占比 不可用'));
assert(ui.riskHtml({...plan,risk_fraction:.02}).includes('风险预算 2%'));
''')


def test_new_backtest_preserves_allocation_and_explains_pinned_MA_exit_not_L2_exit_application():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),vm=require('vm');
const elements=new Map();function element(id){if(!elements.has(id))elements.set(id,{value:'',innerHTML:'',textContent:'',hidden:false,dataset:{},querySelector:()=>element(id+'-child')});return elements.get(id);}
global.document={getElementById:element,querySelector:()=>element('query'),querySelectorAll:()=>[]};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);','globalThis.ui={state,highReturnBacktestView,renderBacktest,renderCheckpointHelp};'));
const result={strategy_version:'czsc_trend_consensus',model_family:'high_return',run_id:'account-run',model_run_id:'trained-new',initial_cash:200000,per_stock_initial_cash:100000,
allocation:'每股固定独立初始资金；失败账户保留现金；合计资金曲线不代表共享现金账户',metrics:{total_return:0,max_drawdown:0,final_equity:200000,trade_count:0},trades:[],daily:[],rejections:[],failures:[]};
const view=ui.highReturnBacktestView(result);assert.equal(view.allocation,result.allocation);assert.equal(view.initial_cash,200000);assert.equal(view.per_stock_initial_cash,100000);
ui.state.backtest=view;ui.renderBacktest();
const rendered=element('backtest-research-result').innerHTML,help=element('backtest-observation-help').textContent;
assert(rendered.includes(result.allocation));assert(rendered.includes('合计初始资金 200,000 元'));assert(rendered.includes('每股初始资金 100,000 元'));
assert(help.includes('缺兼容模型不自动放行'));assert(help.includes('入场时固定的均线持仓退出模型'));assert(help.includes('二阶段退出模型仅作观察'));
assert(help.includes('完整交易费用后收益至少8%，且≥2倍冻结止损参考风险'));assert(help.includes('成交采用滑点模拟均价；R按冻结止损参考假设估计'));
assert(!rendered.includes('缺模型时执行冻结参考政策'));
ui.state.backtest={...view,accounts:[{symbol:'002436.SZ',exit_policy_diagnostics:[{date:'2026-01-27',head:'exit',probability:.8,applied:true}]}]};ui.renderBacktest();
assert(element('backtest-model-observation-rows').innerHTML.includes('二阶段退出观察，未用于成交'));assert(!element('backtest-model-observation-rows').innerHTML.includes('已应用'));
ui.state.research={high_return:{default_consensus_version:'czsc_trend_consensus',models:[]}};
for(const id of ['backtest-model-family'])element(id).value='high_return';element('backtest-mode').value='research';element('backtest-usage-mode').value='historical';element('backtest-model-policy').value='auto';
ui.renderCheckpointHelp('backtest');const checkpoint=element('backtest-checkpoint-help').textContent;
assert(checkpoint.includes('缺少兼容高收益入场模型不自动放行'));assert(checkpoint.includes('持仓退出模型在入场时固定'));assert(checkpoint.includes('二阶段退出模型仅作观察'));assert(!checkpoint.includes('无模型日期继续规则研究'));
ui.state.research={high_return:{models:[]}};ui.state.backtest=ui.highReturnBacktestView({...result,strategy_version:'legacy'});ui.renderBacktest();
assert(element('backtest-research-result').innerHTML.includes('缺模型时执行冻结参考政策'));
assert(!element('backtest-observation-help').textContent.includes('冻结止损参考风险'));
ui.renderCheckpointHelp('backtest');assert(element('backtest-checkpoint-help').textContent.includes('显式离线应用入场和退出头'));
''')


def test_current_trend_scan_cannot_submit_legacy_training_and_keeps_MA_scores_visible():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),vm=require('vm');
global.KHQuantChartEvidence=require(path.join(process.argv[1],'chart_evidence.js'));
const elements=new Map();function element(id){if(!elements.has(id))elements.set(id,{value:'',innerHTML:'',textContent:'',hidden:false,disabled:false,dataset:{},options:[],listeners:{},addEventListener(type,fn){this.listeners[type]=fn;},querySelector:()=>element(id+'-child'),reportValidity:()=>true});return elements.get(id);}
global.document={getElementById:element,querySelectorAll:()=>[],addEventListener(){}};global.window={addEventListener(){}};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);','globalThis.ui={state,updateResearchControls,renderTasks,bind,scanActionCell};'));
ui.state.research={calendar_run_id:'calendar',high_return:{default_consensus_version:'czsc_trend_consensus',models:[{run_id:'new',schema_family:'czsc_trend_consensus'},{run_id:'old',schema_family:'high_return_legacy'}]}};
element('scan-mode').value='research';element('scan-model-family').value='high_return';element('scan-model-policy').value='auto';element('scan-usage-mode').value='historical';element('scan-end').value='2026-10-09';element('high-return-source').value='legacy-source';
ui.updateResearchControls('scan');ui.renderTasks();
for(const id of ['high-return-source-label','build-research-points','build-research-episodes','start-research_train'])assert(element(id).hidden);
assert(element('start-research_train').disabled);assert(element('scan-research-target-help').textContent.includes('对应版本的独立重训入口'));
assert(element('scan-research-target-help').textContent.includes('完整交易费用后收益至少8%，且≥2倍冻结止损参考风险'));assert(element('scan-research-target-help').textContent.includes('成交采用滑点模拟均价；R按冻结止损参考假设估计'));
let network=0;global.sessionStorage={getItem:()=>null};global.fetch=()=>{network++;throw new Error('Legacy submission is forbidden');};
ui.bind();for(const id of ['start-research_train','build-research-points','build-research-episodes'])element(id).listeners.click();
assert.equal(network,0);assert(element('page-error-child').textContent.includes('train_trend_consensus'));
const scored=ui.scanActionCell({strategy_version:'czsc_trend_consensus',category:'watch',account_action:{emitted:false,action:'HOLD'},agents:{ma:{p_up:.65,p_down:.15,p_neutral:.2},high_return:{probability:null,evaluation_status:'model_unavailable'}}});
assert(scored.includes('上涨 65%'));assert(scored.includes('下跌 15%'));assert(scored.includes('中性 20%'));assert(scored.includes('缺兼容模型 · 未评估'));assert(!scored.includes('模型概率不可用'));assert(scored.includes('账户观察'));
element('scan-model').value='old';ui.updateResearchControls('scan');assert(element('start-research_train').hidden); // A stale disabled selection cannot override the current automatic strategy.
element('scan-model-policy').value='pinned';ui.updateResearchControls('scan');assert(!element('start-research_train').hidden);assert(!element('build-research-points').hidden);
assert(element('scan-research-target-help').textContent.includes('至少2倍初始风险'));assert(!element('scan-research-target-help').textContent.includes('冻结止损参考风险'));
element('scan-model-family').value='ma_trend';element('scan-model').value='';ui.updateResearchControls('scan');assert(!element('start-research_train').hidden);assert(!element('start-research_train').disabled);
''')


def test_current_new_default_restores_explicit_saved_v4_model_checkpoint_and_label_without_changing_auto():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),vm=require('vm');
const elements=new Map();function element(id){if(!elements.has(id)){
 const strict=/-(model|checkpoint)$/.test(id),node={_value:'',_html:'',textContent:'',dataset:{},hidden:false,disabled:false,options:strict?[{value:''}]:[],querySelector:()=>element('child')};
 Object.defineProperty(node,'value',{get(){return this._value;},set(v){this._value=strict&&!this.options.some(o=>o.value===v)?'':v;}});
 Object.defineProperty(node,'innerHTML',{get(){return this._html;},set(v){this._html=v;if(strict){this.options=[...v.matchAll(/<option value="([^"]*)"[^>]*>([^<]*)/g)].map(m=>({value:m[1],textContent:m[2]}));if(!this.options.some(o=>o.value===this._value))this._value='';}}});elements.set(id,node);
}return elements.get(id);}
global.document={getElementById:element};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);','globalThis.ui={state,researchModels,applyResearchContext,researchSpec};'));
element('analysis-model-family').options=[{value:'high_return',textContent:'趋势共识策略'}];element('analysis-mode').value='research';element('analysis-model-family').value='high_return';element('analysis-end').value='2026-10-09';
ui.state.research={high_return:{default_consensus_version:'czsc_trend_consensus',models:[
 {run_id:'new-main',schema_family:'czsc_trend_consensus',capabilities:{chart:true},checkpoints:[{name:'2025Q2',available_at:'2025-03-31'},{name:'2025Q4',available_at:'2025-09-30'}]},
 {run_id:'saved-v4',model_family:'high_return',schema_family:'high_return_native_v4',capabilities:{chart:true},checkpoints:[{name:'2026Q3',available_at:'2026-06-30'}]},
 {run_id:'other-old',schema_family:'high_return_legacy',capabilities:{chart:true},checkpoints:[]}]}};
assert.deepEqual(ui.researchModels('analysis').map(x=>x.run_id),['new-main']);
const saved={research:true,model_family:'high_return',model_run_id:'saved-v4',model_policy:'pinned',model_fold:'2026Q3',usage_mode:'historical',entry_policy:'risk',consensus_version:'czsc_agent_consensus_v4',consensus_config:{version:'czsc_agent_consensus_v4'}};
ui.applyResearchContext('analysis',saved);
assert.equal(element('analysis-model').value,'saved-v4');assert.equal(element('analysis-checkpoint').value,'2026Q3');
assert.deepEqual(ui.researchModels('analysis').map(x=>x.run_id),['new-main','saved-v4']);
assert(!element('analysis-model').options.find(x=>x.value==='saved-v4').textContent.includes('趋势共识策略'));
assert.equal(element('analysis-model-family').options[0].textContent,'三路买卖点＋高收益区间 ML');
const request=ui.researchSpec('analysis');assert.equal(request.model_run_id,'saved-v4');assert.equal(request.model_fold,'2026Q3');assert.equal(request.model_policy,'pinned');
assert.equal(request.consensus_version,'czsc_agent_consensus_v4');assert.deepEqual(request.consensus_config,saved.consensus_config);
assert.equal(ui.state.pendingResearchContexts.analysis.consensus_version,'czsc_agent_consensus_v4');
ui.applyResearchContext('analysis',{research:true,model_family:'high_return',model_policy:'auto',usage_mode:'historical',consensus_version:'czsc_trend_consensus'});
assert.equal(element('analysis-model').value,'');assert.deepEqual(ui.researchModels('analysis').map(x=>x.run_id),['new-main']);
assert.equal(element('analysis-model-family').options[0].textContent,'趋势共识策略');assert(!ui.researchSpec('analysis').model_run_id);
assert.equal(ui.researchSpec('analysis').consensus_version,'czsc_trend_consensus');assert(!ui.researchSpec('analysis').consensus_config);
ui.applyResearchContext('analysis',{research:true,model_family:'high_return',model_run_id:'new-main',model_policy:'auto',usage_mode:'historical',consensus_version:'czsc_trend_consensus'});
assert.equal(element('analysis-checkpoint').value,'');assert(element('analysis-checkpoint').disabled);
assert(element('analysis-checkpoint').innerHTML.includes('自动按日期匹配'));assert(!element('analysis-checkpoint').innerHTML.includes('2025Q2'));
ui.applyResearchContext('analysis',{research:true,model_family:'high_return',model_run_id:'new-main',model_policy:'pinned',model_fold:'2025Q4',usage_mode:'historical',consensus_version:'czsc_trend_consensus'});
assert.equal(element('analysis-checkpoint').value,'2025Q4');assert(!element('analysis-checkpoint').disabled);
assert.equal(ui.researchSpec('analysis').model_fold,'2025Q4');
''')


def test_current_configured_model_is_default_without_replacing_explicit_selection_or_legacy_family():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),vm=require('vm');
const elements=new Map();function element(id){if(!elements.has(id)){
 const node={_value:'',_html:'',options:[{value:''}],dataset:{}};
 Object.defineProperty(node,'value',{get(){return this._value;},set(v){this._value=id.endsWith('-model')&&!this.options.some(o=>o.value===v)?'':v;}});
 Object.defineProperty(node,'innerHTML',{get(){return this._html;},set(v){this._html=v;this.options=[...v.matchAll(/<option value="([^"]*)"/g)].map(m=>({value:m[1]}));if(id.endsWith('-model')&&!this.options.some(o=>o.value===this._value))this._value='';}});elements.set(id,node);
}return elements.get(id);}
global.document={getElementById:element};global.location={hash:''};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);','globalThis.ui={state,populateResearchModels};'));
ui.state.research={models:[{run_id:'legacy-ma'}],high_return:{default_consensus_version:'czsc_trend_consensus',current_model_run_id:'configured-main',model_selection:'configured_run_auto_prior_heads',models:[
 {run_id:'newer-experiment',schema_family:'czsc_trend_consensus',capabilities:{chart:true,scan:true,legacy_backtest:true}},
 {run_id:'configured-main',schema_family:'czsc_trend_consensus',capabilities:{chart:true,scan:true,legacy_backtest:true}}]}};
for(const prefix of ['analysis','scan','backtest']){
 element(prefix+'-model-family').value='high_return';ui.populateResearchModels(prefix);
 assert.equal(element(prefix+'-model').value,'configured-main',prefix);
}
ui.state.research.high_return.current_model_run_id='newer-experiment';ui.populateResearchModels('backtest');assert.equal(element('backtest-model').value,'newer-experiment'); // A configuration change updates an untouched default.
ui.state.research.high_return.current_model_run_id='configured-main';ui.populateResearchModels('backtest');assert.equal(element('backtest-model').value,'configured-main');
for(const prefix of ['analysis','scan','backtest']){
 element(prefix+'-model').value='newer-experiment';ui.populateResearchModels(prefix);
 assert.equal(element(prefix+'-model').value,'newer-experiment',prefix);
}
ui.state.pendingResearchContexts.analysis={model_run_id:'configured-main'};
ui.populateResearchModels('analysis');assert.equal(element('analysis-model').value,'newer-experiment'); // User choice after opening a saved result survives a status refresh.
element('analysis-model').value='';ui.populateResearchModels('analysis');assert.equal(element('analysis-model').value,'configured-main');
element('backtest-model-family').value='ma_trend';ui.populateResearchModels('backtest');assert.equal(element('backtest-model').value,'');
ui.state.research.high_return.current_model_run_id='missing-or-incompatible';element('scan-model').value='';ui.populateResearchModels('scan');assert.equal(element('scan-model').value,''); // No implicit newest-model fallback.
''')


def test_saved_real_trend_api_uses_bound_intents_on_all_periods_and_one_history_popup_view():
    fixture = os.environ.get('KHQUANT_TREND_UI_RESPONSE_FIXTURE')
    if not fixture:
        pytest.skip('Set KHQUANT_TREND_UI_RESPONSE_FIXTURE to a saved actual API response')
    import json
    run_js('const actualFixture=' + json.dumps(fixture) + ';\n' + r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs');
const ui=require(path.join(process.argv[1],'chart_evidence.js')),markers=require(path.join(process.argv[1],'chart_markers.js'));
global.KHQuantMarkers=markers;const wyckoff=require(path.join(process.argv[1],'wyckoff_markers.js'));
const response=JSON.parse(fs.readFileSync(actualFixture,'utf8'));
assert.equal(response.trend_consensus.version,'czsc_trend_consensus');
const input=ui.accountMarkerInputs(response,response.symbol);assert(input.valid);assert.equal(input.rejectedRecords,0);
const evidence=ui.accountEvidence(response),rows=ui.mergeEvidenceRows(evidence.rows,response.trend_consensus.rows);
const start=rows[0].date,cutoff=response.as_of;
for(const frequency of ['日线','周线','月线']) {
 const out=markers.build({bars:response.frequencies[frequency].bars,dailyBars:response.frequencies['日线'].bars,frequency,
 events:input.events,trades:input.trades,consensusVersion:'czsc_trend_consensus',symbol:response.symbol,start,asOf:cutoff});
 const expected=input.events.filter(e=>e.evidence_date>=start && e.evidence_date<=cutoff);
 assert.equal(out.intentCount,expected.length,frequency);
 assert(out.groups.filter(g=>g.kind==='intention').every(g=>g.sources.includes('account_action')));
 const projected=out.groups.filter(g=>g.kind==='intention').flatMap(g=>g.entries.map(e=>e.row.action));
 for(const action of ['BUY','SELL'])assert.equal(projected.filter(value=>value===action).length,expected.filter(e=>e.action===action).length,frequency+' '+action); // Never net opposing account actions inside a week/month.
 assert(out.mappings.filter(m=>m.kind==='execution').every(m=>m.event_date===m.row.date.slice(0,10)));
 const w=wyckoff.buildDetailed({events:response.wyckoff_events,bars:response.frequencies[frequency].bars,frequency,symbol:response.symbol,start,asOf:cutoff});
 const expectedW=response.wyckoff_events.filter(e=>e.evidence_visible!==false&&String(e.first_available_at||e.available_at).slice(0,10)>=start&&String(e.first_available_at||e.available_at).slice(0,10)<=cutoff);
 assert.equal(w.eventCount,expectedW.length,frequency);assert.equal(w.groups.flatMap(g=>g.entries).length,expectedW.length);
 assert(w.markers.every(m=>m.text.startsWith('W ')));assert.equal(wyckoff.buildDetailed({events:[],bars:response.frequencies[frequency].bars,frequency}).markers.length,0);
}
for(const row of rows) {
 const trend=ui.trendProbabilities(row.agents.ma),view=ui.currentAgentView(row,'ma',row.consensus?.agents?.ma);
 const popup=ui.popupHtml({evidence,consensus:response.trend_consensus,groups:[],barDate:row.date,frequency:'日线',selectedDate:row.date});
 assert(popup.includes(view.summary));assert(!popup.includes('v5'));
 if(trend.available){
  assert(view.summary.includes('上涨'));assert(view.summary.includes('下跌'));assert(view.summary.includes('中性'));
  const ma=row.consensus.agents.ma,weight=response.trend_consensus.config.source_weights.ma;
  assert(Math.abs(ma.buy_contribution-row.agents.ma.p_up*weight)<1e-10);assert(Math.abs(ma.sell_contribution-row.agents.ma.p_down*weight)<1e-10);
  assert(ui.agentOverviewHtml(row,row.consensus,response.trend_consensus.config).includes(`买 ${ui.supportScore(ma.buy_contribution)} / 卖 ${ui.supportScore(ma.sell_contribution)} 分`));
  if(row.agents.ma.model_identity.historical_reconstruction===true)assert(popup.includes('历史重建评分 · 非当时实际部署预测'));
 }
 else assert(!view.summary.includes('上涨 0%'));
 if(row.account_action?.emitted && row.account_action.action==='SELL')assert(popup.includes('S · 账户退出意图'));
 const history=ui.historyHtml([row]);assert(history.includes('账户最终 B / S'));assert(history.includes(view.summary));
 const highReturn=row.agents.high_return,quality=ui.currentAgentView(row,'high_return',row.consensus?.agents?.high_return);
 assert(popup.includes(quality.summary));assert(history.includes(`data-evaluation-status="${quality.evaluationStatus}"`));
 if(highReturn.probability!==null && highReturn.probability!==undefined){
  assert(history.includes(ui.probability(highReturn.probability)));assert(!quality.summary.includes('未评估'));
  if(highReturn.target==='actual_complete_fee_net_roundtrip_high_return_and_initial_risk'){
   assert(popup.includes('完整交易费用后收益至少8%，且≥2倍冻结止损参考风险'));assert(popup.includes('成交采用滑点模拟均价；R按冻结止损参考假设估计'));
  }
  if(highReturn.evaluation_status==='scored_rejected'&&highReturn.reason_codes.includes('high_return_below_threshold')){
   assert(popup.includes('高收益评分未达到冻结门槛'));assert(history.includes('高收益评分未达到冻结门槛'));
   assert(!row.account_action.emitted);assert(history.includes('<td>—</td>'));
  }
 }else assert(!quality.summary.includes('概率 0%'));
 const facts=Array.isArray(row.candidate_evidence)?row.candidate_evidence:row.candidate_evidence?.facts;
 if(Array.isArray(facts)) {
  const actualFacts=ui.structureFacts(row);
  const visibleStructure=facts.filter(e=>e.source==='structure'&&(e.direction_eligible===true||e.direction_eligible==null&&e.candidate_direction_valid===true));
  assert(actualFacts.available);assert.equal(actualFacts.buy,visibleStructure.filter(e=>e.side==='BUY').length);
  assert.equal(actualFacts.sell,visibleStructure.filter(e=>e.side==='SELL').length);
  assert.equal(actualFacts.qualifiedBuy,visibleStructure.filter(e=>e.side==='BUY'&&(Object.hasOwn(e,'risk_candidate_eligible')?e.risk_candidate_eligible:e.candidate_eligible)===true).length);
  if(!row.consensus?.agents?.structure?.judgment&&row.consensus?.agents?.structure?.status==='scored')assert(history.includes('当前 连续支持'));
 }
}
''')


def test_saved_real_trend_workbench_dynamic_counts_and_W_toggle_do_not_invent_account_BS():
    fixture = os.environ.get('KHQUANT_TREND_UI_RESPONSE_FIXTURE')
    if not fixture:
        pytest.skip('Set KHQUANT_TREND_UI_RESPONSE_FIXTURE to a saved actual API response')
    import json
    run_js('const actualFixture=' + json.dumps(fixture) + ';\n' + r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),vm=require('vm');
global.KHQuantMarkers=require(path.join(process.argv[1],'chart_markers.js'));global.KHQuantWyckoffMarkers=require(path.join(process.argv[1],'wyckoff_markers.js'));global.KHQuantChartEvidence=require(path.join(process.argv[1],'chart_evidence.js'));
const elements=new Map(),windowListeners=new Map();function element(id){if(!elements.has(id))elements.set(id,{value:'',innerHTML:'',textContent:'',dataset:{},style:{},checked:false,hidden:false,clientWidth:640,clientHeight:450,offsetWidth:460,offsetHeight:500,attributes:{},listeners:{},addEventListener(type,fn){this.listeners[type]=fn;},querySelector:()=>element('query'),replaceChildren(){this.innerHTML='';},append(){},setAttribute(){},getBoundingClientRect:()=>({left:0,top:0})});return elements.get(id);}
global.document={getElementById:element,documentElement:{},querySelectorAll:()=>[],addEventListener(){}};global.window={innerWidth:1000,innerHeight:900,addEventListener(type,fn){windowListeners.set(type,fn);}};global.location={hash:''};global.getComputedStyle=()=>({getPropertyValue:()=> '#fff'});global.requestAnimationFrame=()=>1;global.ResizeObserver=class{observe(){}};
let click,hover,candles;const series=()=>({setData(){},setMarkers(value){this.markers=value;},applyOptions(){},priceScale:()=>({applyOptions(){}}),priceToCoordinate:()=>50,createPriceLine:()=>({}),removePriceLine(){}});
const chart={addCandlestickSeries:()=>candles=series(),addHistogramSeries:series,addLineSeries:series,removeSeries(){},resize(){},subscribeClick:fn=>click=fn,subscribeCrosshairMove:fn=>hover=fn,timeScale:()=>({subscribeVisibleLogicalRangeChange(){},getVisibleLogicalRange:()=>({from:0,to:2}),fitContent(){},height:()=>30,timeToCoordinate:()=>50}),priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};window.LightweightCharts=LightweightCharts;
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);','globalThis.app={state,renderAnalysis,bind,resetChartSelection,openRun,nextSessionGuidance,highReturnResolutionText,history:()=>historyRows};'));
const response=JSON.parse(fs.readFileSync(actualFixture,'utf8')),ui=KHQuantChartEvidence,input=ui.accountMarkerInputs(response);
const latest=response.next_session_decision,guidance=app.nextSessionGuidance(latest,response.as_of);
if(response.trend_consensus.config.require_high_return===false){
 assert(response.chart_evidence.rows.every(row=>row.agents.high_return.probability===null&&row.agents.high_return.evaluation_status==='entry_quality_gate_disabled'&&row.agents.high_return.applied_to_entry===false));
 assert(guidance.includes('质量过滤已关闭，本次未评分'));assert(!guidance.includes('无兼容检查点'));
}
if(latest.account_action?.shares===0&&['WAIT','HOLD'].includes(latest.action))assert(guidance.includes('空仓观望'));
if(latest.agents.high_return.evaluation_status==='no_candidate'){
 assert(guidance.includes('本日无合格买入候选，高收益本日未评分'));assert(!guidance.includes('无兼容检查点'));
}
const actualReason=ui.opinionReasons(latest.account_action)||ui.opinionReasons(latest);
if(actualReason){assert(guidance.includes(actualReason));assert(!guidance.includes('条件依据暂未返回'));}
const held={...latest,action:'HOLD',holding:null,account_action:{...latest.account_action,holding_snapshot:null,shares:100}};
assert(app.nextSessionGuidance(held,response.as_of).includes('持有观察'));
const unknown={...latest,action:'WAIT',holding:null,account_action:{...latest.account_action,holding_snapshot:null,shares:undefined}};
assert(app.nextSessionGuidance(unknown,response.as_of).includes('仓位未提供 · 观察'));
const existing={...held,agents:{...held.agents,high_return:{probability:null,status:'actual_existing_position',evaluation_status:'actual_existing_position'}}};
const heldStatus=app.highReturnResolutionText(latest.model_resolution,existing);assert(heldStatus.includes('持仓中 · 入场不适用'));assert(!heldStatus.includes('无兼容检查点'));
assert(app.highReturnResolutionText({model_family:'high_return'},{}).includes('无兼容检查点')); // Old saved results keep their prior unavailable-model wording.
for(const row of response.chart_evidence.rows.filter(row=>row.agents.high_return.probability!=null)){
 const status=app.highReturnResolutionText({model_family:'high_return',schema_family:'czsc_trend_consensus'},row);
 assert(status.includes(ui.probability(row.agents.high_return.probability)));assert(!status.includes('无兼容检查点'));
 if(row.agents.high_return.evaluation_status==='scored_rejected')assert(status.includes('已评分 · 未过门槛'));
}
app.state.analysis=response;app.state.query={symbol:response.symbol,start:response.chart_evidence.rows[0].date,as_of:response.as_of};
app.state.backtest={trades:[{symbol:response.symbol,date:response.as_of,action:'BUY',price:999}]};
element('layer-signals').checked=true;element('layer-trades').checked=true;element('layer-wyckoff').checked=true;element('view-analysis').hidden=false;
app.bind();
for(const frequency of ['日线','周线','月线']) {
 app.resetChartSelection();app.state.frequency=frequency;app.renderAnalysis(false);
 const header=element('marker-summary').textContent;
 assert(header.includes(`选定账户 B/S ${input.events.length} 个`));assert(header.includes(`账户成交 ${input.trades.length} 条`));
 assert(header.includes(`起点 ${response.account_actions.context.account_start}`));assert(header.includes(`原始量价事件 ${response.wyckoff_events.length} 条`));
 const w=candles.markers.filter(m=>m.text.startsWith('W '));assert(w.length>0);
 if(!input.events.length)assert(!candles.markers.some(m=>['B','S'].includes(m.text)));
 if(!input.trades.length)assert(!candles.markers.some(m=>['实B','实S'].includes(m.text)));
 hover({time:w[0].time,point:{x:20,y:20},seriesData:new Map([[candles,{close:25}]])});assert(element('chart-evidence-popup').hidden);
 click({time:w[0].time,hoveredObjectId:w[0].id,point:{x:20,y:20}});assert(!element('chart-evidence-popup').hidden);
 assert.equal((element('chart-evidence-popup').innerHTML.match(/data-agent-summary-row=/g)||[]).length,4);
 assert(!element('chart-evidence-popup').innerHTML.includes('999'));
 windowListeners.get('keydown')({key:'Escape'});assert(element('chart-evidence-popup').hidden);
 const sell=candles.markers.find(m=>m.text==='S');
 if(sell){
  hover({time:sell.time,point:{x:20,y:20},seriesData:new Map([[candles,{close:25}]])});assert(element('chart-evidence-popup').hidden);
  click({time:sell.time,hoveredObjectId:sell.id,point:{x:20,y:20}});const sellDetail=element('chart-evidence-popup').innerHTML;
  assert(sellDetail.includes('S · 账户退出意图'));assert(sellDetail.includes(response.account_actions.context.account_id));assert(!sellDetail.includes('999'));
  if(response.trend_consensus.config.require_high_return===false)assert(sellDetail.includes('质量过滤已关闭，本次未评分'));
  windowListeners.get('keydown')({key:'Escape'});assert(element('chart-evidence-popup').hidden);
 }
 element('layer-wyckoff').checked=false;app.renderAnalysis(false);assert(!candles.markers.some(m=>m.text.startsWith('W ')));
 assert(!element('marker-summary').textContent.includes('原始量价事件'));element('layer-wyckoff').checked=true;
}
if(response.trend_consensus.events.length) {
 element('layer-directions').checked=true;element('layer-directions').listeners.change();assert(candles.markers.some(m=>m.text.startsWith('方向')));
 assert(element('marker-summary').textContent.includes(`选定账户 B/S ${input.events.length} 个`));
 element('layer-directions').checked=false;element('layer-directions').listeners.change();assert(!candles.markers.some(m=>m.text.startsWith('方向')));
}
global.sessionStorage={getItem:()=>null};global.fetch=async()=>({ok:true,status:200,text:async()=>JSON.stringify(response)});
const modelRun=response.account_actions.context.model_identity.model_run_id;
app.state.research={models:[],high_return:{default_consensus_version:'czsc_trend_consensus',models:[{run_id:modelRun,schema_family:'czsc_trend_consensus',checkpoints:[]} ]}};
app.state.analysis={symbol:'old',chart_evidence:{rows:[{date:'1900-01-01',reference_close:999}]}};
(async()=>{
 await app.openRun('actual-saved');assert.equal(app.state.analysis.strategy_version,'czsc_trend_consensus');
 assert.equal(app.history().length,response.chart_evidence.rows.length);assert(!app.history().some(row=>row.date==='1900-01-01'));
 assert.equal(app.state.query.model_policy,'auto');assert.equal(app.state.query.model_run_id,modelRun);
 assert.equal(app.state.query.account_start,response.account_actions.context.account_start);assert.equal(app.state.query.account_initial_cash,response.account_actions.context.initial_cash);
 assert(app.history().every(row=>row.account_context.account_id===response.account_actions.context.account_id));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_account_b_then_s_in_same_week_month_keep_dates_and_fill_after_signal_on_forming_period():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const marker=require(path.join(process.argv[1],'chart_markers.js'));
const bound={symbol:'002436.SZ',account_id:'account',series_context_hash:'series',source:'account_action',source_frequency:'日线'};
const dates=['2026-02-24','2026-02-25','2026-02-26'];
const events=[{...bound,date:dates[0],first_available_at:dates[0]+'T15:00:00+08:00',event_id:'buy',action:'BUY'},
 {...bound,date:dates[1],first_available_at:dates[1]+'T15:00:00+08:00',event_id:'sell',action:'SELL'}];
for(const frequency of ['日线','周线','月线']) {
 const bars=frequency==='日线'?dates.map(time=>({time,is_closed:true})):[{time:dates[2],is_closed:false}];
 const out=marker.build({bars,dailyBars:dates.map(time=>({time})),frequency,events,asOf:dates[2],consensusVersion:'czsc_trend_consensus',symbol:bound.symbol});
 assert.equal(out.intentCount,2);assert.deepEqual(out.markers.map(m=>m.text),['B','S']);
 assert.deepEqual(out.groups.flatMap(g=>g.entries.map(e=>e.event_date)).sort(),dates.slice(0,2));
 if(frequency!=='日线')assert.equal(out.formingProjectedCount,2);
}
const clipped=marker.build({bars:dates.map(time=>({time})),frequency:'日线',events,asOf:dates[0],symbol:bound.symbol});
assert.equal(clipped.intentCount,1);assert.equal(clipped.markers[0].text,'B');
const riskExit=marker.build({bars:dates.map(time=>({time})),frequency:'日线',events:[{...events[1],candidate_eligible:false}],asOf:dates[2],symbol:bound.symbol});
assert.equal(riskExit.intentCount,1);assert.equal(riskExit.markers[0].text,'S'); // Account exit is not a buy candidate.
''')


def test_account_action_popup_and_history_show_holding_exit_and_original_execution_date():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const action={date:'2026-01-27',action:'SELL',emitted:true,intent_id:'exit-1',reason_codes:['hard_stop'],account_id:'research',series_context_hash:'series'};
action.exit_evaluation={probability:.7377,holding_contract_compatible:true};
const row={date:action.date,reference_close:25,agents:{ma:{kind:'trend_probability',p_up:.2,p_down:.6,p_neutral:.2}},account_action:action,
account_context:{account_kind:'research_broker',account_start:'2023-01-01',initial_cash:100000},
holding:{shares:300},broker_fills:[{date:'2026-01-28',signal_date:'2026-01-27',action:'SELL',price:24.83}],rejections:[]};
const html=ui.popupHtml({evidence:{rows:[row]},consensus:{version:'czsc_trend_consensus',rows:[{date:row.date,action:'WAIT',emitted:false}]},barDate:row.date,frequency:'日线'});
assert(html.includes('S · 账户退出意图'));assert(html.includes('持仓 300 股'));assert(html.includes('2026-01-28'));assert(html.includes('24.83'));
assert(html.includes('持仓退出概率 73.77%'));assert(!html.includes('缺少兼容真实持仓'));
assert(html.includes('起点 2023-01-01'));assert(html.includes('初始资金 100,000 元'));assert(html.includes('显示区间不会重置账户'));
const history=ui.historyHtml([row]);assert(history.includes('账户最终 B / S'));assert(history.includes('<td>S</td>'));
assert(history.includes('300 股'));assert(history.includes('2026-01-28'));assert(history.includes('上涨 20%'));
const missing=ui.currentAgentView({...row,holding:null,account_action:null},'ma');assert(missing.exit.includes('持仓状态未提供'));assert(!missing.exit.includes('空仓'));
const flat=ui.currentAgentView({...row,holding:null,account_action:{shares:0}},'ma');assert(flat.exit.includes('空仓'));
const quality=ui.currentAgentView({agents:{high_return:{probability:null,evaluation_status:'actual_existing_position'}}},'high_return');
assert(quality.summary.includes('持仓中 · 入场不适用'));assert(!quality.summary.includes('概率 0%'));
const groups=[{kind:'intention',sources:['consensus'],diagnostic:true,entries:[{event_date:action.date,row:{source:'consensus',date:action.date,action:'BUY',reference_price:999,reference_price_date:action.date}}]},
 {kind:'intention',sources:['account_action'],entries:[{event_date:action.date,row:{source:'account_action',date:action.date,action:'SELL',first_available_at:action.date+'T09:30:00+08:00',reference_price:24.9,reference_price_date:action.date}}]}];
assert.equal(ui.referenceSelection({evidence:{rows:[row]},groups,barDate:action.date}).price,24.9);
delete groups[1].entries[0].row.reference_price;
assert.equal(ui.referenceSelection({evidence:{rows:[row]},groups,barDate:action.date}),null); // A diagnostic close cannot replace a missing open trigger reference.
''')


def test_live_workbench_uses_bound_account_not_global_trades_and_keeps_click_only_details():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),vm=require('vm');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));global.KHQuantMarkers=require(path.join(process.argv[1],'chart_markers.js'));
global.KHQuantWyckoffMarkers=require(path.join(process.argv[1],'wyckoff_markers.js'));global.KHQuantChartEvidence=ui;
const elements=new Map(),listeners={};function element(id){if(!elements.has(id))elements.set(id,{value:'',innerHTML:'',textContent:'',dataset:{},style:{},checked:false,hidden:false,
 clientWidth:640,clientHeight:450,offsetWidth:460,offsetHeight:500,children:[],attributes:{},addEventListener(){},querySelector:()=>element('query'),replaceChildren(){this.innerHTML='';},append(){},setAttribute(){},getBoundingClientRect:()=>({left:0,top:0})});return elements.get(id);}
global.document={getElementById:element,documentElement:{},querySelectorAll:()=>[],addEventListener:(key,fn)=>listeners[key]=fn};
global.window={innerWidth:1000,innerHeight:900,addEventListener:(key,fn)=>listeners[key]=fn};global.location={hash:''};global.getComputedStyle=()=>({getPropertyValue:()=> '#fff'});
global.requestAnimationFrame=()=>1;global.ResizeObserver=class{observe(){}};
let click,hover,candles;const series=()=>({setData(){},setMarkers(value){this.markers=value;},applyOptions(){},priceScale:()=>({applyOptions(){}}),priceToCoordinate:()=>50,createPriceLine:()=>({}),removePriceLine(){}});
const chart={addCandlestickSeries:()=>candles=series(),addHistogramSeries:series,addLineSeries:series,removeSeries(){},resize(){},subscribeClick:fn=>click=fn,subscribeCrosshairMove:fn=>hover=fn,
 timeScale:()=>({subscribeVisibleLogicalRangeChange(){},getVisibleLogicalRange:()=>({from:0,to:2}),fitContent(){},height:()=>30,timeToCoordinate:()=>50}),priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};window.LightweightCharts=LightweightCharts;
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);','globalThis.app={state,renderAnalysis,bind,selection:()=>chartSelection};'));
const dates=['2026-02-24','2026-02-25','2026-02-26'],bound={account_id:'selected',series_context_hash:'series',symbol:'002436.SZ'};
const intent=(date,action)=>({...bound,date,action,side:action,event_id:date,intent_id:date,signal_at:date+'T15:00:00+08:00',emitted:true});
const rows=dates.map(date=>({...bound,date,account_action:intent(date,date===dates[0]?'BUY':date===dates[1]?'SELL':'WAIT'),agents:{ma:{kind:'trend_probability',p_up:.65,p_down:.15,p_neutral:.2}}}));
app.state.analysis={symbol:bound.symbol,as_of:dates[2],trend_consensus:{version:'czsc_trend_consensus',rows:dates.map(date=>({date,action:'WAIT',emitted:false})),events:[]},
 account_actions:{context:{account_id:'selected',account_kind:'research_broker_replay'},series_context_hash:'series',rows:rows.map(r=>r.account_action),intents:[intent(dates[0],'BUY'),intent(dates[1],'SELL')],
 ledger:[{...bound,date:dates[1],signal_date:dates[0],action:'BUY',price:25},{...bound,date:dates[2],signal_date:dates[1],action:'SELL',price:26}]},
 chart_evidence:{rows},frequencies:{'日线':{bars:dates.map(time=>({time,open:25,high:26,low:24,close:25,volume:100,is_closed:true}))}}};
app.state.query={symbol:bound.symbol,start:dates[0],as_of:dates[2]};app.state.backtest={trades:[{symbol:bound.symbol,date:dates[2],action:'BUY',price:999}]};
element('layer-signals').checked=true;element('layer-trades').checked=true;element('view-analysis').hidden=false;
app.bind();app.renderAnalysis(false);assert.deepEqual(candles.markers.map(m=>m.text),['B','实B','S','实S']);
assert(element('marker-summary').textContent.includes('选定账户 B/S 2'));assert(!element('marker-summary').textContent.includes('v5'));
hover({time:dates[0],point:{x:20,y:20},seriesData:new Map([[candles,{close:25}]])});assert(element('chart-evidence-popup').hidden);
click({time:dates[1],hoveredObjectId:candles.markers.find(m=>m.text==='S').id,point:{x:20,y:20}});
assert(!element('chart-evidence-popup').hidden);assert(element('chart-evidence-popup').innerHTML.includes('S · 账户退出意图'));
assert(!element('chart-evidence-popup').innerHTML.includes('999'));
app.state.analysis.account_actions.intents=[];app.state.analysis.account_actions.ledger=[];
app.state.analysis.account_actions.rows=dates.map(date=>({...bound,date,action:'WAIT',emitted:false,shares:0}));
listeners.keydown({key:'Escape'});app.renderAnalysis(false);assert.equal(candles.markers.length,0);
hover({time:dates[2],point:{x:20,y:20},seriesData:new Map([[candles,{close:25}]])});assert(element('chart-evidence-popup').hidden);
click({time:dates[2],point:{x:20,y:20}});assert(!element('chart-evidence-popup').hidden);
assert(element('chart-evidence-popup').innerHTML.includes('K线评分查看'));
assert(element('chart-evidence-popup').innerHTML.includes('评分日期'));assert(!element('chart-evidence-popup').innerHTML.includes('信号可知日'));
assert(element('chart-evidence-popup').innerHTML.includes('上涨 65%'));
assert(!element('chart-evidence-popup').innerHTML.includes('S · 账户退出意图'));
assert(!element('chart-evidence-popup').innerHTML.includes('B · 账户买入意图'));
assert.equal(candles.markers.length,0);
for(const frequency of ['周线','月线']){
 listeners.keydown({key:'Escape'});app.state.frequency=frequency;
 app.state.analysis.frequencies[frequency]={bars:[{time:dates[2],open:25,high:26,low:24,close:25,volume:300,is_closed:true}]};
 app.renderAnalysis(false);hover({time:dates[2],point:{x:20,y:20},seriesData:new Map([[candles,{close:25}]])});
 assert(element('chart-evidence-popup').hidden);click({time:dates[2],point:{x:20,y:20}});
 const popup=element('chart-evidence-popup').innerHTML;
 assert(popup.includes('K线评分查看'));assert(popup.includes('上涨 65%'));
 for(const date of dates)assert(popup.includes(`data-evidence-date="${date}"`));
 assert.equal(candles.markers.length,0);
}
''')
