"""Current chart visibility, source closure and click reference regressions."""
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
const symbol='002436.SZ',version='trend_consensus';
const bar=(time,closed=true)=>({time,open:10,high:12,low:9,close:11,is_closed:closed});
const event=(id,date,action='BUY')=>({symbol,source:'reference_consensus',reference_only:true,source_frequency:'日线',version,
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
const symbol='002436.SZ',version='trend_consensus';
for(const frequency of ['周线','月线']) {
  const end=frequency==='周线'?'2026-07-03':'2026-07-31';
  const closure=frequency==='周线'?'2026-07-06':'2026-08-03';
  const base={frequency,symbol,consensusVersion:version,up:'red',down:'green'};
  const event={symbol,version,source:'reference_consensus',reference_only:true,source_frequency:frequency,event_id:'native',
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


@pytest.mark.parametrize('synchronous_remove_callback', ['none', 'price_line', 'series', 'creation', 'both'])
def test_reference_price_line_and_point_only_exist_during_click_window_and_clean_on_all_resets(synchronous_remove_callback):
    script = r'''
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
const monthlyButton=newElement('button');monthlyButton.dataset.frequency='月线';
global.document={getElementById:element,documentElement:{},createElementNS:(_,tag)=>newElement(tag),
 querySelectorAll:selector=>selector==='[data-frequency]'?[weeklyButton,monthlyButton]:[],addEventListener(key,fn){documentListeners[key]=fn;}};
global.getComputedStyle=()=>({getPropertyValue:()=> '#123456'});global.ResizeObserver=class{observe(){}};
global.requestAnimationFrame=()=>1;global.window={innerWidth:907,innerHeight:924,addEventListener(key,fn){windowListeners[key]=fn;}};
global.location={hash:''};global.setTimeout=()=>1;global.clearTimeout=()=>{};
let range={from:0,to:2},crosshair,click,firstSeries;const lines=new Set(),points=new Set(),liveSeries=new Set();
const scale={subscribeVisibleLogicalRangeChange(){},getVisibleLogicalRange:()=>range,setVisibleLogicalRange(value){range=value;},fitContent(){},height:()=>30,timeToCoordinate:()=>50};
function series(){return {setData(value){this.data=value;},setMarkers(value){this.markers=value;},applyOptions(){},priceScale:()=>({applyOptions(){}}),priceToCoordinate:price=>350-(price-8)*60};}
const chart={addCandlestickSeries(){firstSeries=series();firstSeries.createPriceLine=options=>{lines.add(options);if(['creation','both'].includes(synchronousRemoveCallback))crosshair({seriesData:new Map()});return options;};firstSeries.removePriceLine=line=>{const removed=lines.delete(line);if(['price_line','both'].includes(synchronousRemoveCallback)){assert(removed,'Price line not found');crosshair({seriesData:new Map()});}};return firstSeries;},
 addHistogramSeries:series,addLineSeries(options){const result=series();liveSeries.add(result);if(options.pointMarkersVisible){points.add(result);result.setData=value=>{result.data=value;if(['creation','both'].includes(synchronousRemoveCallback))crosshair({time:value[0].time,point:{x:50,y:50},seriesData:new Map([[firstSeries,{close:11}]])});};}return result;},removeSeries(item){assert(liveSeries.delete(item),'Series not found');points.delete(item);if(['series','both'].includes(synchronousRemoveCallback))crosshair({seriesData:new Map()});},resize(){},
 subscribeCrosshairMove(fn){crosshair=fn;},subscribeClick(fn){click=fn;},timeScale:()=>scale,priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};window.LightweightCharts=LightweightCharts;
const code=fs.readFileSync(path.join(root,'workbench.js'),'utf8').replace('init().catch(showError);',
 'globalThis.interactions={state,renderAnalysis,bind,resetChartSelection,selection:()=>chartSelection};');vm.runInThisContext(code);
const {state,renderAnalysis,bind,resetChartSelection}=interactions;
const dates=['2026-07-01','2026-07-02','2026-07-03'],symbol='002436.SZ',version='trend_consensus';
const bars=dates.map(time=>({time,open:10,high:12,low:9,close:11,volume:100,is_closed:true}));
const events=[dates[0],dates[2]].map((date,i)=>({date,time:date,evidence_date:date,first_available_at:date+'T15:00:00+08:00',source_frequency:'日线',source:'reference_consensus',reference_only:true,action:i?'SELL':'BUY',symbol,event_id:'final-'+i,reference_price:i?13:10.75,reference_price_date:date}));
state.analysis={symbol,analysis_mode:'reference',reference_points:events.map(e=>({...e,reference_only:true})),as_of:dates[2],data_end:dates[2],consensus_decisions:{version,config:{version},rows:events.map(e=>({...e,emitted:true})),events},
 chart_evidence:{rows:dates.map((date,i)=>({date,symbol,analysis_mode:'reference',reference_close:10.75+i,reference_price_date:date,agents:{}}))},
 frequencies:{'日线':{bars},'周线':{bars:[{...bars[2],is_closed:false}]},'月线':{bars:[{...bars[2],is_closed:false}]}}};
state.query={symbol,start:dates[0],end:dates[2]};state.dates=dates;element('layer-signals').checked=true;
bind();renderAnalysis(false);
assert(element('marker-summary').textContent.includes('最终共识 B 1 / S 1 · 周期标记 2 个'));
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
assert.equal(firstSeries.markers.length,2);assert(element('marker-summary').textContent.includes('日线参考点投影于形成中周期'));
const buy=firstSeries.markers.find(marker=>marker.text==='共B');click({time:dates[2],hoveredObjectId:buy.id,point:{x:750,y:700}});
assert.equal([...lines][0].price,10.75);assert.deepEqual([...points][0].data,[{time:dates[2],value:10.75}]);
assert(element('chart-evidence-popup').innerHTML.includes('日线已确认／周期形成中'));
const dateButton=newElement('button');dateButton.dataset.evidenceDate=dates[2];
documentListeners.click({target:{closest:()=>dateButton}});assert.equal(lines.size,1);assert.equal([...lines][0].price,13);
resetChartSelection();assert.equal(lines.size,0);assert.equal(points.size,0);
// Real regression: an open weekly popup closes, then monthly click still works.
weeklyButton.listeners.click[0]();clicked(dates[2]);assert.equal(lines.size,1);
windowListeners.keydown({key:'Escape'});assert.equal(lines.size,0);assert.equal(points.size,0);
monthlyButton.listeners.click[0]();clicked(dates[2]);assert.equal(state.frequency,'月线');assert.equal(lines.size,1);
assert(!element('chart-evidence-popup').hidden);assert.equal([...lines][0].price,13);
// Switching directly while open must clear safely before the next popup.
weeklyButton.listeners.click[0]();assert.equal(lines.size,0);assert.equal(points.size,0);clicked(dates[2]);assert.equal(lines.size,1);
resetChartSelection();assert.equal(lines.size,0);assert.equal(points.size,0);assert.equal(interactions.selection().pinned,null);
'''
    run_js("const synchronousRemoveCallback = '"+synchronous_remove_callback+"';\n"+script)


def test_current_structure_panel_counts_direction_facts_before_risk_and_reports_current_opinion():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-06-04',agents:{structure:{kind:'rule_evidence',status:'no_signal',buy_count:0,sell_count:0,reason:'当日未触发独立结构买卖点'}},
 candidate_evidence:{facts:[{event_id:'one',source:'structure',side:'BUY',direction_eligible:true,candidate_eligible:false},
  {event_id:'two',source:'structure',side:'BUY',direction_eligible:true,candidate_eligible:false}],
 risk_details:[{event_id:'one',risk_reason_codes:['stop_distance_exceeds_frozen_maximum']},{event_id:'two',risk_reason_codes:['invalid_causal_stop']}]}};
const decision={version:'trend_consensus',agents:{structure:{judgment:'BUY',status:'supported',evidence_mode:'active_event',origin_event_ids:['1','2','3','4','5','6'],configured_weight:.2,buy_contribution:.2,sell_contribution:0}}};
const html=ui.agentsHtml(row,false,decision);
assert(html.includes('当日方向事实：买 2 · 卖 0 · 原始入场资格通过 0 条买入候选'));
assert(html.includes('规则证据 · 当日复核支持'));assert(html.includes('当日方向事实含 2 条风险未通过的买入候选'));
assert(html.includes('已按当日状态复核 6 条原始来源（含有效历史证据）'));
assert(html.includes('仍有效的历史事件 · 原始事件 6 条</p>'));
assert(!html.includes('买入触发 0'));assert(!html.includes('当日未触发独立结构买卖点'));assert(!html.includes('条}'));
''')


