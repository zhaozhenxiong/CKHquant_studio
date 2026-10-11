"""Exercise v2 scores and saved-version replay through shipped JavaScript."""
import shutil
import subprocess
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[1] / "web_dashboard/static/js"


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped chart interaction checks")
    result = subprocess.run([node, "-e", script, str(STATIC)], capture_output=True,
                            text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_weighted_popup_displays_saved_contributions_and_distinct_probability():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const version='czsc_agent_consensus_v2';
const config={source_weights:{ma:.4,structure:.2,wyckoff:.3},high_return_quality_weight:.1,
  min_buy_base_support:.5,min_buy_support:.6,min_sell_support:.5,min_direction_margin:.2,min_independent_families:2};
const row={date:'2026-07-01',reference_close:11,agents:{ma:{probability:.8},structure:{buy_count:2,sell_count:0},
  wyckoff:{kind:'rule_evidence',buy_score:.9,sell_score:.1},high_return:{probability:.6}}};
const decision={date:row.date,version,action:'BUY',emitted:true,buy_support:.7,sell_support:0,
  buy_base_score:.7,sell_base_score:0,buy_quality_bonus:.1,buy_total_score:.8,
  buy_independent_count:2,sell_independent_count:0,buy_independent_families:['ma','wyckoff'],
  reason_codes:['multi_agent_direction_confirmed'],agents:{
    ma:{judgment:'BUY',configured_weight:.4,buy_contribution:.4,sell_contribution:0},
    structure:{judgment:'BUY',configured_weight:.2,buy_contribution:0,sell_contribution:0,reason_codes:['ma_assisted_structure_dependency_merged']},
    wyckoff:{judgment:'BUY',configured_weight:.3,buy_contribution:.3,sell_contribution:0},
    high_return:{status:'passed',configured_weight:.1,buy_contribution:.1,quality_contribution:.1,required:true},
    arbiter:{judgment:'BUY'}},family_details:{
      ma:{side:'BUY',weight:.4,sources:['ma','structure'],conflict:false},
      wyckoff:{side:'BUY',weight:.3,sources:['wyckoff'],conflict:false}}};
const consensus={version,config,rows:[decision]},group={kind:'intention',action:'BUY',entries:[{row:{evidence_date:row.date}}]};
for(const frequency of ['日线','周线','月线']) {
  const html=ui.popupHtml({evidence:{rows:[row]},consensus,groups:[group],barDate:row.date,frequency});
  assert(html.includes('B · 最终买点'));
  assert(html.includes('买入基础 70 / 90 分 ＋ 高收益质量 10 / 10 分 ＝ 买入总分 80 / 100 分'));
  assert(html.includes('卖出基础 0 / 90 分'));assert(html.includes('基础方向差额（买−卖）+70 分'));
  assert(html.includes('基础权重 40 分 · 买入贡献 40 分'));
  assert(html.includes('基础权重 20 分 · 买入贡献 0 分'));
  assert(html.includes('基础权重 30 分 · 买入贡献 30 分'));
  assert(html.includes('质量权重 10 分 · 买入质量加分 10 分 · 不计独立证据数量'));
  assert(html.includes('均线／辅助结构：支持买入 · 实际贡献 40 分 · 来源 均线 ML、CZSC 结构'));
  assert(html.includes('买入基础 50 分 · 买入总分 60 分 · 卖出基础 50 分 · 基础方向优势 20 分 · 至少 2 组独立证据'));
  assert(html.includes('支持分，非获利概率'));assert(html.includes('入场模型分 80%'));
  assert(html.includes('高收益区间概率 60%'));assert(!html.includes('买入总分 80%'));
}
// A configured weight never substitutes for a missing saved contribution.
const missing=structuredClone(decision);delete missing.agents.structure.buy_contribution;
assert(ui.agentsHtml(row,false,missing,config,version).includes('基础权重 20 分 · 买入贡献 — 分'));
// Result-level version is authoritative even when a row includes v2 fields.
const old=ui.consensusHtml({...decision,buy_support:2}, {min_buy_support:2,min_sell_support:2,min_direction_margin:1},'czsc_agent_consensus_v1');
assert(old.includes('加权支持：买 2 · 卖 0'));assert(old.includes('仲裁 v1'));
assert(!old.includes('买入总分'));assert(!old.includes('支持分，非获利概率'));
const oldAgents=ui.agentsHtml(row,false,decision,null,'czsc_agent_consensus_v1');
assert(oldAgents.includes('不计入支持票数'));assert(!oldAgents.includes('基础权重'));
''')


def test_wait_and_family_conflict_remain_final_decision_not_raw_score_inference():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const config={version:'czsc_agent_consensus_v2',source_weights:{ma:.4,structure:.2,wyckoff:.3},high_return_quality_weight:.1,
  min_buy_base_support:.5,min_buy_support:.6,min_sell_support:.5,min_direction_margin:.2,min_independent_families:2};
const decision={date:'2026-07-01',version:config.version,action:'WAIT',emitted:false,
  buy_base_score:.5,sell_base_score:.4,buy_quality_bonus:.1,buy_total_score:.6,
  buy_independent_count:2,sell_independent_count:1,reason_codes:['direction_conflict_or_insufficient_margin','source_conflict_propagated_to_dependency_family'],
  family_details:{ma:{side:'CONFLICT',weight:0,sources:['ma','structure'],conflict:true}},agents:{arbiter:{judgment:'WAIT'}}};
const html=ui.consensusHtml(decision,config);
assert(html.includes('本日未生成最终买卖点'));assert(!html.includes('B · 最终买点'));
assert(html.includes('买入总分 60 / 100 分'));assert(html.includes('基础方向差额（买−卖）+10 分'));
assert(html.includes('买卖支持相互冲突或方向优势不足'));
assert(html.includes('同一 Agent 的买卖冲突已传递到依赖家族，相关家族贡献归零'));
assert(html.includes('同家族方向冲突，贡献归零'));assert(html.includes('实际贡献 0 分'));
assert.equal(ui.finalSide(decision),'');
const history=ui.historyHtml(ui.mergeEvidenceRows([{date:decision.date,agents:{},events:[{action:'BUY'}]}],[decision]));
assert(!history.includes('<td>B</td>'));assert(history.includes('买卖支持相互冲突或方向优势不足'));
''')


def test_saved_run_and_replay_bind_version_but_new_form_uses_default():
    run_js(r'''
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('node:assert/strict');
const root=process.argv[1];require(path.join(root,'chart_evidence.js'));
global.KHQuantMarkers=require(path.join(root,'chart_markers.js'));
const elements=new Map();
function newElement(){return {dataset:{},style:{},checked:false,hidden:false,disabled:false,value:'',innerHTML:'',textContent:'',listeners:{},options:[],
  addEventListener(key,callback){this.listeners[key]=callback;},querySelector:()=>newElement(),
  replaceChildren(){this.innerHTML='';},setAttribute(){},removeAttribute(){}};}
function element(id){if(!elements.has(id))elements.set(id,newElement());return elements.get(id);}
global.document={getElementById:element,querySelectorAll:()=>[],addEventListener(){},documentElement:{dataset:{}}};
global.window={addEventListener(){}};global.location={hash:'analysis'};global.sessionStorage={getItem:()=>null};
global.setTimeout=()=>1;global.clearTimeout=()=>{};global.requestAnimationFrame=()=>1;
const code=fs.readFileSync(path.join(root,'workbench.js'),'utf8').replace('init().catch(showError);',
 'globalThis.interactions={state,openRun,replay,analyze,bind,consensusRequestContext};');
vm.runInThisContext(code);
const {state,openRun,replay,analyze,bind,consensusRequestContext}=interactions;
const date='2026-07-01',base={symbol:'600027.SH',as_of:date,data_end:date,events:[],chart_evidence:{rows:[]},frequencies:{'日线':{bars:[]}}};
const cfg1={source_weights:{ma:1,structure:1,wyckoff:1},min_buy_support:2,min_sell_support:2,min_direction_margin:1};
const cfg2={source_weights:{ma:.4,structure:.2,wyckoff:.3},high_return_quality_weight:.1,min_buy_base_support:.5,min_buy_support:.6,min_sell_support:.5,min_direction_margin:.2,min_independent_families:2};
const makeResult=(version,config)=>({...base,request:{symbol:base.symbol,start:date,end:date,research:false,consensus_version:'czsc_agent_consensus_v2'},
  consensus_decisions:{version,config,rows:[],events:[]}});
const old=makeResult('czsc_agent_consensus_v1',cfg1),current=makeResult('czsc_agent_consensus_v2',cfg2);
const captured=[];let nextResult=old;
global.fetch=async(url,options={})=>{if(options.body)captured.push(JSON.parse(options.body));
  return {ok:true,status:200,text:async()=>JSON.stringify(url.startsWith('/api/runs/')?{kind:'analysis',result:nextResult}:nextResult)};};
(async()=>{
  await openRun('old-v1');assert.equal(state.query.consensus_version,'czsc_agent_consensus_v1');assert.deepEqual(state.query.consensus_config,cfg1);
  state.dates=[date];await replay(0);assert.equal(captured.at(-1).consensus_version,'czsc_agent_consensus_v1');assert.deepEqual(captured.at(-1).consensus_config,cfg1);
  nextResult=current;await openRun('new-v2');assert.equal(state.query.consensus_version,'czsc_agent_consensus_v2');
  state.dates=[date];await replay(0);assert.equal(captured.at(-1).consensus_version,'czsc_agent_consensus_v2');assert.deepEqual(captured.at(-1).consensus_config,cfg2);
  // Normal analysis also pins the effective response version for its next replay.
  await analyze({symbol:base.symbol,research:false});assert.equal(state.query.consensus_version,'czsc_agent_consensus_v2');
  assert.equal(captured.at(-1).consensus_version,undefined);assert.equal(captured.at(-1).consensus_config,undefined);
  // A new form submission clears any prior saved-run config/version.
  bind();element('analysis-symbol').value=base.symbol;element('analysis-mode').value='structure';
  element('analysis-form').listeners.submit({preventDefault(){}});await new Promise(setImmediate);
  assert.equal(captured.at(-1).consensus_version,undefined);assert.equal(captured.at(-1).consensus_config,undefined);
  assert.equal(state.query.consensus_version,'czsc_agent_consensus_v2');
  assert.deepEqual(consensusRequestContext({consensus_decisions:{config:{...cfg1,version:'czsc_agent_consensus_v1'}}}).consensus_version,'czsc_agent_consensus_v1');
})().catch(error=>{console.error(error);process.exitCode=1;});
''')
