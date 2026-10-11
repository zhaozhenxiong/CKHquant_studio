"""Exercise v3 final-point timing, saved diagnostics, and click-only prices."""
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


def test_known_daily_points_project_into_forming_period_with_stable_identity():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const markers=require(path.join(process.argv[1],'chart_markers.js'));
const symbol='002436.SZ',version='czsc_agent_consensus_v3';
const bar=(time,closed=true)=>({time,open:10,high:12,low:9,close:11,is_closed:closed});
const event=(id,date,action='BUY')=>({symbol,source:'consensus',source_frequency:'日线',version,
  event_id:id,date,time:date,evidence_date:date,available_at:date+'T15:00:00+08:00',
  first_available_at:date+'T15:00:00+08:00',action,reference_price:11,reference_price_date:date});
const events=[event('final-one','2026-07-01'),event('final-two','2026-07-02')];
const args={symbol,consensusVersion:version,events,up:'red',down:'green'};
for(const frequency of ['周线','月线']) {
  const first=markers.build({...args,frequency,bars:[bar('2026-07-01',false)],dailyBars:[bar('2026-07-01')],asOf:'2026-07-01'});
  assert.equal(first.intentCount,1);assert.equal(first.intentMarkerCount,1);
  assert.equal(first.formingProjectedCount,1);assert.equal(first.deferredCount,0);
  assert.equal(first.groups[0].period_status,'forming');assert.equal(first.groups[0].closed_at,null);
  assert.equal(first.mappings[0].original_date,'2026-07-01');
  assert.equal(first.mappings[0].first_available_at,events[0].first_available_at);
  assert(markers.describe(first.mappings,'2026-07-01',frequency).includes('日线已确认／'+frequency+'周期形成中'));
  const next=markers.build({...args,frequency,bars:[bar('2026-07-02',false)],dailyBars:[bar('2026-07-01'),bar('2026-07-02')],asOf:'2026-07-02'});
  assert.equal(next.intentCount,2);assert.equal(next.intentMarkerCount,1);
  assert.equal(next.groups[0].id,first.groups[0].id);
  assert.equal(next.groups[0].period_id,first.groups[0].period_id);
  assert.equal(next.mappings[0].entry_id,first.mappings[0].entry_id);
  assert.deepEqual(next.groups[0].event_ids,['final-one','final-two']);
  const end=frequency==='周线'?'2026-07-03':'2026-07-31';
  const closure=frequency==='周线'?'2026-07-06':'2026-08-03';
  const closed=markers.build({...args,frequency,bars:[bar(end)],dailyBars:[bar('2026-07-01'),bar('2026-07-02'),bar(end),bar(closure)],asOf:closure});
  assert.equal(closed.groups[0].id,first.groups[0].id);
  assert.equal(closed.groups[0].period_status,'closed');assert.equal(closed.groups[0].closed_at,closure);
  assert.equal(closed.formingProjectedCount,0);assert.equal(closed.intentCount,2);
  for(const oldVersion of ['czsc_agent_consensus_v1','czsc_agent_consensus_v2']) {
    const legacy=markers.build({...args,consensusVersion:oldVersion,frequency,bars:[bar('2026-07-02',false)],dailyBars:[bar('2026-07-01'),bar('2026-07-02')],asOf:'2026-07-02'});
    assert.equal(legacy.intentCount,0);assert.equal(legacy.deferredCount,2);
  }
}
const duplicate=markers.build({...args,events:[events[0],events[0]],frequency:'日线',bars:[bar('2026-07-01')],asOf:'2026-07-01'});
assert.equal(duplicate.intentCount,1);assert.equal(duplicate.intentMarkerCount,1);
''')


def test_native_higher_period_signal_requires_observed_closure_and_causal_availability():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const markers=require(path.join(process.argv[1],'chart_markers.js'));
const bar=(time,closed=true)=>({time,is_closed:closed});
const symbol='002436.SZ',version='czsc_agent_consensus_v3';
for(const frequency of ['周线','月线']) {
  const end=frequency==='周线'?'2026-07-03':'2026-07-31';
  const closure=frequency==='周线'?'2026-07-06':'2026-08-03';
  const base={frequency,symbol,consensusVersion:version,up:'red',down:'green'};
  const event={symbol,version,source:'consensus',source_frequency:frequency,event_id:'native',
    time:end,date:end,source_period_date:end,evidence_date:closure,first_available_at:closure+'T15:00:00+08:00',action:'SELL'};
  const early=markers.build({...base,events:[event],bars:[bar(end,false)],dailyBars:[bar(end)],asOf:end});
  assert.equal(early.intentCount,0);assert.equal(early.markers.length,0);
  const closed=markers.build({...base,events:[event],bars:[bar(end)],dailyBars:[bar(end),bar(closure)],asOf:closure});
  assert.equal(closed.intentCount,1);assert.equal(closed.groups[0].daily_confirmed,false);
  assert.equal(closed.groups[0].closed_at,closure);assert.equal(closed.mappings[0].original_date,closure);
  assert.equal(closed.markers[0].time,end);
  const falselyEarly={...event,evidence_date:end,first_available_at:end+'T15:00:00+08:00'};
  const rejected=markers.build({...base,events:[falselyEarly],bars:[bar(end)],dailyBars:[bar(end),bar(closure)],asOf:closure});
  assert.equal(rejected.intentCount,0);assert.equal(rejected.deferredCount,1);
}
''')


def test_v3_popup_keeps_saved_scores_funnel_active_ages_original_dates_and_reference():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const version='czsc_agent_consensus_v3',config={version,source_weights:{ma:.4,structure:.2,wyckoff:.3},high_return_quality_weight:.1,
 min_buy_base_support:.5,min_buy_support:.6,min_sell_support:.5,min_direction_margin:.2,min_independent_families:2};
const dates=['2026-07-01','2026-07-02'];
const rows=dates.map(date=>({date,reference_close:date===dates[0]?40.87:42,reference_price_date:date,agents:{ma:{probability:.634},high_return:{probability:.114}},events:[]}));
rows[0].events=[{event_id:'risk-rejected',source:'structure',event_type:'raw_buy',candidate_eligible:false}];
rows[0].candidate_evidence={facts:[{event_id:'risk-rejected',direction_eligible:true}],risk_details:[{event_id:'risk-rejected',candidate_eligible:false,reference_price:40.87,stop_price:30,stop_distance:.266,risk_reason_codes:['stop_distance_exceeds_frozen_maximum']}]};
const decision={date:dates[0],version,action:'BUY',emitted:true,buy_base_score:.7,sell_base_score:0,buy_quality_bonus:.1,buy_total_score:.8,
 buy_independent_count:2,sell_independent_count:0,reason_codes:['multi_agent_direction_confirmed'],
 funnel:{raw_events:1,observable_events:1,fresh_candidates:1,active_evidence:2,directional_sources:2,independent_buy_families:2,independent_sell_families:0,l1_qualified:1,risk_eligible:1,quality_passed:1,final_emitted:1},
 active_evidence:[{event_id:'ma-old<script>',source:'ma',side:'BUY',origin_date:'2026-06-30',age_bars:1,remaining_bars:2,opportunity_id:'opportunity-1',revalidated:true}],
 agents:{ma:{judgment:'BUY',configured_weight:.4,buy_contribution:.4,sell_contribution:0,evidence_mode:'current_snapshot',snapshot_id:'ma-day-1'},
 structure:{judgment:'ABSTAIN',configured_weight:.2,buy_contribution:0,sell_contribution:0,reason_codes:['ma_full_identity_unavailable']},
 wyckoff:{judgment:'BUY',configured_weight:.3,buy_contribution:.3,sell_contribution:0},
 high_return:{status:'passed',configured_weight:.1,quality_contribution:.1,required:true}}};
const entries=dates.map((date,i)=>({original_date:date,displayed:true,first_available_at:date+'T15:00:00+08:00',
 row:{symbol:'002436.SZ',source:'consensus',event_id:'final-'+i,evidence_date:date,first_available_at:date+'T15:00:00+08:00',
 action:'BUY',reference_price:i?42:40.87,reference_price_date:date,anchor_dates:['2026-06-29']}}));
const group={action:'BUY',kind:'intention',visible:true,daily_confirmed:true,period_status:'forming',closed_at:null,entries};
const context={evidence:{rows},consensus:{version,config,rows:[decision]},groups:[group],barDate:dates[1],frequency:'月线',selectedDate:dates[0]};
const html=ui.popupHtml(context);
assert(html.includes('仲裁 v3 · 各 Agent 共同评估新候选'));
assert(html.includes('买入总分 80 / 100 分'));assert(html.includes('入场模型分 63.4%'));
assert(html.includes('候选逐层判断'));assert(html.includes('原始事件 1 · 当日可见 1 · 新候选 1 · 活跃证据 2'));
assert(html.includes('方向与新机会：通过 · 风险资格：通过 · 高收益质量：通过 · 最终点：通过'));
assert(html.includes('已经过 1 根日线 · 剩余 2 根'));assert(html.includes('已按当日状态重新核验'));
assert(html.includes('ma-old&lt;script&gt;'));assert(!html.includes('ma-old<script>'));
assert(html.includes('原始方向证据：有效；实际贡献按当日状态复核'));
assert(html.includes('新入场风险拒绝：止损距离超过冻结上限'));assert(html.includes('距离 26.6%'));
assert(html.includes('均线评分缺少完整模型身份'));assert(html.includes('评估当日状态'));
assert(html.includes('日线已确认／周期形成中'));assert(!html.includes('周期闭合'));
assert(html.includes('信号首次可用 2026-07-01T15:00:00+08:00'));
assert(html.includes('原始结构／事件锚点 2026-06-29'));
for(const date of dates) assert(html.includes('data-evidence-date="'+date+'"'));
const selected=ui.referenceSelection(context);assert.equal(selected.price,40.87);assert.equal(selected.signal_date,dates[0]);
assert.equal(selected.bar_date,dates[1]);assert.deepEqual(selected.anchor_dates,['2026-06-29']);
assert.equal(ui.referenceSelection({...context,selectedDate:dates[1]}).price,42);
const bad=structuredClone(context);bad.groups[0].entries[0].row.reference_price_date='2026-06-30';
assert.equal(ui.referenceSelection(bad),null);
const sell=ui.consensusHtml({...decision,action:'SELL',proposed_action:'SELL',invalidated_evidence:[{source:'ma',origin_date:'2026-06-30',invalidated_at:'2026-07-01T15:00:00+08:00',event_id:'bad-old'}]},config);
assert(sell.includes('风险资格：卖出不适用 · 高收益质量：卖出不适用'));assert(sell.includes('当日永久失效证据 1 条'));
assert(ui.consensusHtml({...decision,emitted:false,action:'WAIT',reason_codes:['no_fresh_candidate_trigger']},config).includes('本日没有新的候选触发，活跃证据不会自行生成买卖点'));
''')


def test_reference_price_line_and_point_only_exist_during_click_window_and_clean_on_all_resets():
    run_js(r'''
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('node:assert/strict');
const root=process.argv[1];require(path.join(root,'chart_evidence.js'));
global.KHQuantMarkers=require(path.join(root,'chart_markers.js'));require(path.join(root,'wyckoff_markers.js'));
const elements=new Map(),windowListeners={},documentListeners={};
function newElement(tag='div') {return {tag,dataset:{},style:{},checked:false,hidden:false,disabled:false,
 clientWidth:640,clientHeight:450,offsetWidth:460,offsetHeight:500,value:'',textContent:'',innerHTML:'',children:[],attributes:{},listeners:{},
 getBoundingClientRect:()=>({left:20,top:180}),addEventListener(key,callback){(this.listeners[key] ||= []).push(callback);},querySelector:()=>newElement(),
 replaceChildren(){this.children=[];this.innerHTML='';},append(child){this.children.push(child);},hasAttribute(key){return key in this.attributes;},setAttribute(key,value){this.attributes[key]=value;}};}
function element(id){if(!elements.has(id))elements.set(id,newElement());return elements.get(id);}
const weeklyButton=newElement('button');weeklyButton.dataset.frequency='周线';
global.document={getElementById:element,documentElement:{},createElementNS:(_,tag)=>newElement(tag),
 querySelectorAll:selector=>selector==='[data-frequency]'?[weeklyButton]:[],addEventListener(key,fn){documentListeners[key]=fn;}};
global.getComputedStyle=()=>({getPropertyValue:()=> '#123456'});global.ResizeObserver=class{observe(){}};
global.requestAnimationFrame=()=>1;global.window={innerWidth:907,innerHeight:924,addEventListener(key,fn){windowListeners[key]=fn;}};
global.location={hash:''};global.setTimeout=()=>1;global.clearTimeout=()=>{};
let range={from:0,to:2},crosshair,click,firstSeries;const lines=new Set(),points=new Set();
const scale={subscribeVisibleLogicalRangeChange(){},getVisibleLogicalRange:()=>range,setVisibleLogicalRange(value){range=value;},fitContent(){},height:()=>30,timeToCoordinate:()=>50};
function series(){return {setData(value){this.data=value;},setMarkers(value){this.markers=value;},applyOptions(){},priceScale:()=>({applyOptions(){}}),priceToCoordinate:price=>350-(price-8)*60};}
const chart={addCandlestickSeries(){firstSeries=series();firstSeries.createPriceLine=options=>{lines.add(options);return options;};firstSeries.removePriceLine=line=>lines.delete(line);return firstSeries;},
 addHistogramSeries:series,addLineSeries(options){const result=series();if(options.pointMarkersVisible)points.add(result);return result;},removeSeries(item){points.delete(item);},resize(){},
 subscribeCrosshairMove(fn){crosshair=fn;},subscribeClick(fn){click=fn;},timeScale:()=>scale,priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};window.LightweightCharts=LightweightCharts;
const code=fs.readFileSync(path.join(root,'workbench.js'),'utf8').replace('init().catch(showError);',
 'globalThis.interactions={state,renderAnalysis,bind,resetChartSelection,selection:()=>chartSelection};');vm.runInThisContext(code);
const {state,renderAnalysis,bind,resetChartSelection}=interactions;
const dates=['2026-07-01','2026-07-02','2026-07-03'],symbol='002436.SZ',version='czsc_agent_consensus_v3';
const bars=dates.map(time=>({time,open:10,high:12,low:9,close:11,volume:100,is_closed:true}));
const events=[dates[0],dates[2]].map((date,i)=>({date,time:date,evidence_date:date,first_available_at:date+'T15:00:00+08:00',source_frequency:'日线',source:'consensus',action:i?'SELL':'BUY',symbol,event_id:'final-'+i,reference_price:i?13:10.75,reference_price_date:date}));
state.analysis={symbol,as_of:dates[2],data_end:dates[2],consensus_decisions:{version,config:{version},rows:events.map(e=>({...e,emitted:true})),events},
 chart_evidence:{rows:dates.map((date,i)=>({date,reference_close:10.75+i,reference_price_date:date,agents:{}}))},
 frequencies:{'日线':{bars},'周线':{bars:[{...bars[2],is_closed:false}]}}};
state.query={symbol,start:dates[0],end:dates[2]};state.dates=dates;element('layer-signals').checked=true;
bind();renderAnalysis(false);
assert(element('marker-summary').textContent.includes('最终 B/S 2 个 · 周期标记 2 个'));
const hover=time=>crosshair({time,point:{x:420,y:430},seriesData:new Map([[firstSeries,{close:11}]])});
const clicked=time=>click({time,point:{x:750,y:700}});
hover(dates[0]);assert.equal(lines.size,0);assert.equal(points.size,0);assert(element('chart-evidence-popup').hidden);
clicked(dates[0]);assert.equal(lines.size,1);assert.equal(points.size,1);assert.equal([...lines][0].price,10.75);
assert([...lines][0].title.includes('参考 · 非成交'));assert.deepEqual([...points][0].data,[{time:dates[0],value:10.75}]);
hover(dates[2]);assert.equal([...lines][0].price,10.75);
clicked(dates[2]);assert.equal(lines.size,1);assert.equal(points.size,1);assert.equal([...lines][0].price,13);
const close=newElement('button');close.setAttribute('data-chart-evidence-close','');
documentListeners.click({target:{closest:()=>close}});assert.equal(lines.size,0);assert.equal(points.size,0);
clicked(dates[0]);clicked(dates[1]);assert.equal(lines.size,0);assert.equal(points.size,0);
clicked(dates[0]);windowListeners.keydown({key:'Escape'});assert.equal(lines.size,0);assert.equal(points.size,0);
clicked(dates[0]);weeklyButton.listeners.click[0]();assert.equal(lines.size,0);assert.equal(points.size,0);
assert.equal(firstSeries.markers.length,2);assert(element('marker-summary').textContent.includes('日线最终点投影于形成中周期'));
const buy=firstSeries.markers.find(marker=>marker.text==='B');click({time:dates[2],hoveredObjectId:buy.id,point:{x:750,y:700}});
assert.equal([...lines][0].price,10.75);assert.deepEqual([...points][0].data,[{time:dates[2],value:10.75}]);
assert(element('chart-evidence-popup').innerHTML.includes('日线已确认／周期形成中'));
const dateButton=newElement('button');dateButton.dataset.evidenceDate=dates[2];
documentListeners.click({target:{closest:()=>dateButton}});assert.equal(lines.size,1);assert.equal([...lines][0].price,13);
resetChartSelection();assert.equal(lines.size,0);assert.equal(points.size,0);
''')


def test_v3_structure_panel_counts_direction_facts_before_risk_and_reports_current_opinion():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-06-04',agents:{structure:{kind:'rule_evidence',status:'no_signal',buy_count:0,sell_count:0,reason:'当日未触发独立结构买卖点'}},
 candidate_evidence:{facts:[{event_id:'one',source:'structure',side:'BUY',direction_eligible:true,candidate_eligible:false},
  {event_id:'two',source:'structure',side:'BUY',direction_eligible:true,candidate_eligible:false}],
 risk_details:[{event_id:'one',risk_reason_codes:['stop_distance_exceeds_frozen_maximum']},{event_id:'two',risk_reason_codes:['invalid_causal_stop']}]}};
const decision={version:'czsc_agent_consensus_v3',agents:{structure:{judgment:'BUY',status:'supported',evidence_mode:'active_event',origin_event_ids:['1','2','3','4','5','6'],configured_weight:.2,buy_contribution:.2,sell_contribution:0}}};
const html=ui.agentsHtml(row,false,decision);
assert(html.includes('当日方向事实：买 2 · 卖 0 · 原始入场资格通过 0 条买入候选'));
assert(html.includes('规则证据 · 当日复核支持'));assert(html.includes('当日方向事实含 2 条风险未通过的买入候选'));
assert(html.includes('已按当日状态复核 6 条原始来源（含有效历史证据）'));
assert(html.includes('仍有效的历史事件 · 原始事件 6 条</p>'));
assert(!html.includes('买入触发 0'));assert(!html.includes('当日未触发独立结构买卖点'));assert(!html.includes('条}'));
const old=ui.agentsHtml(row,false,decision,null,'czsc_agent_consensus_v2');
assert(old.includes('买入触发 0 · 卖出触发 0'));assert(old.includes('当日未触发独立结构买卖点'));assert(!old.includes('当日方向事实：'));
''')
