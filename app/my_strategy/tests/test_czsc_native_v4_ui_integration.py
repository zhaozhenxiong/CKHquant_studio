"""Shipped JS against a compact projection of a real native v4 API response.

The default fixture preserves every visible day and event, with public stored
prices and null model scores. Set KHQUANT_V4_UI_RESPONSE_FIXTURE explicitly to
validate a complete saved live response; no business database is read by tests.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "web_dashboard/static/js"
FIXTURE = Path(__file__).with_name("fixtures") / "native_v4_display_contract_20261012.json.gz"
SIDEBAR_FIXTURE = Path(__file__).with_name("fixtures") / "native_v4_sidebar_contract_20261012.json.gz"


def run_js(script, default_fixture=FIXTURE):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for native API display integration")
    fixture = Path(os.environ.get("KHQUANT_V4_UI_RESPONSE_FIXTURE", default_fixture))
    prefix = r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),zlib=require('zlib');
const fixture=process.argv[2],bytes=fs.readFileSync(fixture);
const response=JSON.parse((fixture.endsWith('.gz')?zlib.gunzipSync(bytes):bytes).toString('utf8'));
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const markers=require(path.join(process.argv[1],'chart_markers.js'));
const wyckoff=require(path.join(process.argv[1],'wyckoff_markers.js'));
'''
    result = subprocess.run([node, "-e", prefix + script, str(STATIC), str(fixture)],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_real_response_projects_every_w_event_on_all_periods_without_importing_prefix_points():
    run_js(r'''
const stats=response.chart_evidence.statistics;
assert.equal(stats.visible_window.bars,response.chart_evidence.rows.length);
assert.equal(stats.visible_window.wyckoff_events,response.wyckoff_events.length);
assert.equal(stats.visible_window.source_events,response.events.length);
assert.equal(stats.visible_window.final_points,response.consensus_decisions.events.length);
assert(stats.full_prefix.bars>stats.visible_window.bars);
const expected=new Set(response.wyckoff_events.map(e=>e.event_id));
for (const frequency of ['日线','周线','月线']) {
 const bars=response.frequencies[frequency].bars;
 const w=wyckoff.buildDetailed({events:response.wyckoff_events,bars,frequency,asOf:response.as_of,symbol:response.symbol});
 assert.equal(w.eventCount,expected.size,frequency);
 assert.deepEqual(new Set(w.groups.flatMap(g=>g.entries.map(e=>e.event_id))),expected);
 assert(w.groups.every(g=>g.entries.every(e=>e.event_date===e.first_available_at.slice(0,10))));
 const final=markers.build({bars,dailyBars:response.frequencies['日线'].bars,frequency,events:response.consensus_decisions.events,
  consensusVersion:response.consensus_decisions.version,symbol:response.symbol,asOf:response.as_of,up:'red',down:'green'});
 assert.equal(final.intentCount,stats.visible_window.final_points);
 assert(final.markers.every(m=>['B','S'].includes(m.text)));
 assert.equal(wyckoff.buildDetailed({bars,frequency,events:[]}).eventCount,0);
}
// The untrained fixture deliberately has no visible B/S; full-prefix S points
// stay a separate count and are never imported into its displayed chart.
if (response._fixture_provenance) {
 assert.equal(stats.visible_window.wyckoff_events,16);assert.equal(stats.visible_window.source_events,62);
 assert.equal(stats.visible_window.final_points,0);assert.equal(stats.full_prefix.final_points,19);
}
''')


def test_real_daily_rows_use_one_fact_risk_and_null_status_view_in_history_and_popup():
    run_js(r'''
const rows=ui.mergeEvidenceRows(response.chart_evidence.rows,response.consensus_decisions.rows);
const allowed=new Set(['no_candidate','risk_blocked','input_ineligible','model_unavailable','scored_rejected','scored_passed','identity_error']);
const statusCounts={};
for (const row of rows) {
 const status=row.agents.high_return.evaluation_status;
 assert(allowed.has(status),`${row.date}: unknown L2 state ${status}`);
 const view=ui.currentAgentView(row,'high_return',row.consensus.agents.high_return);
 assert.equal(view.evaluationStatus,status);
 statusCounts[status]=(statusCounts[status]||0)+1;
 const history=ui.historyHtml([row]);
 const popup=ui.agentsHtml(row,false,row.consensus,response.consensus_decisions.config,response.consensus_decisions.version);
 assert(history.includes(view.evaluationLabel));assert(popup.includes(view.evaluationLabel));
 if(row.agents.high_return.probability===null) {
  assert(!view.summary.includes('概率 0%'));assert(!popup.includes('高收益区间概率 0%'));
 }
 const facts=ui.structureFacts(row);
 assert.equal(facts.buy,row.agents.structure.buy_count);assert.equal(facts.sell,row.agents.structure.sell_count);
 assert.equal(facts.qualifiedBuy,row.agents.structure.risk_buy_count ?? facts.qualifiedBuy);
 // Native evaluations expose input_eligible but deliberately do not invent a
 // second status field. The UI must not label finite eligible W data missing.
 if(row.agents.wyckoff.input_eligible===true) {
  const wview=ui.currentAgentView(row,'wyckoff',row.consensus.agents.wyckoff);
  assert.notEqual(wview.status,'unavailable',`${row.date}: qualified W data mislabeled missing`);
 }
}
if(response._fixture_provenance)assert.deepEqual(statusCounts,{model_unavailable:23,no_candidate:154,risk_blocked:6});
const risk=rows.find(r=>r.date==='2026-08-06');
if(risk && response._fixture_provenance) {
 assert.deepEqual([ui.structureFacts(risk).buy,ui.structureFacts(risk).qualifiedBuy,ui.structureFacts(risk).riskRejected],[2,0,2]);
 const details=ui.rawEventsHtml(risk,[],response.consensus_decisions.version);
 assert(details.includes('新入场风险拒绝：止损距离超过冻结上限'));
 assert(details.includes('参考 31.53 元'));assert(details.includes('止损 25.19 元'));
 assert(ui.currentAgentView(risk,'high_return').summary.includes('风险阻止'));
}
const html=ui.historyHtml(rows,0,'',response.chart_evidence.statistics);
assert(html.includes('显示窗口：'));assert(html.includes('完整预热前缀：'));
assert(html.includes(`完整量价事件 ${response.chart_evidence.statistics.visible_window.wyckoff_events}`));
assert(html.includes(`最终 B/S ${response.chart_evidence.statistics.visible_window.final_points}`));
for(const row of rows.filter(r=>r.agents.high_return.evaluation_status==='scored_passed' && !r.consensus.emitted)) {
 const html=ui.consensusHtml(row.consensus,response.consensus_decisions.config,response.consensus_decisions.version);
 if(row.consensus.reason_codes.includes('no_fresh_candidate_trigger')) {
  assert(html.includes('没有取得当前方向资格的新候选'));
  assert(html.includes('新候选（当前资格通过）'));
  assert(row.events.some(e=>e.side==='BUY'));
 }
}
''')


def test_real_w_popup_preserves_confirmation_dates_and_daily_continuous_scores():
    run_js(r'''
const groups=wyckoff.buildDetailed({events:response.wyckoff_events,bars:response.frequencies['月线'].bars,
 frequency:'月线',asOf:response.as_of,symbol:response.symbol}).groups;
const lps=groups.find(g=>g.event_type==='LPS' && g.entries.some(e=>e.event_date==='2026-06-04'));
if(lps) {
 const popup=ui.popupHtml({evidence:response.chart_evidence,consensus:response.consensus_decisions,groups:[lps],
  barDate:lps.time,frequency:'月线',selectedDate:'2026-06-04'});
 assert(popup.includes('原观察 2026-06-04T15:00:00+08:00'));
 assert(popup.includes('首次可用 2026-06-04T15:00:00+08:00'));
 assert(popup.includes('收盘参考价 40.87 元'));assert(popup.includes('当日需求'));
 assert(popup.includes('事件衰减强度'));assert(!popup.includes('>B</span>'));
 assert(popup.includes('data-evidence-date="2026-06-24"'));
}
for(const cutoff of ['2026-06-04','2026-06-25']) {
 const out=wyckoff.buildDetailed({events:response.wyckoff_events,bars:response.frequencies['日线'].bars,
  frequency:'日线',asOf:cutoff,symbol:response.symbol});
 assert.equal(out.eventCount,response.wyckoff_events.filter(e=>e.first_available_at.slice(0,10)<=cutoff).length);
 assert(out.groups.every(g=>g.entries.every(e=>e.first_available_at.slice(0,10)<=cutoff)));
}
''')


def test_real_sidebar_has_original_direction_price_date_and_consistent_scored_agent_copy():
    run_js(r'''
const vm=require('vm'),elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{value:'',textContent:'',innerHTML:'',hidden:false});return elements.get(id);}
global.document={getElementById:element};global.location={hash:'analysis'};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);',
 'globalThis.sidebar={state,renderEvents,referencePriceText,readableEvent};'));
sidebar.state.analysis=response;sidebar.state.query={as_of:response.as_of};sidebar.renderEvents();
const html=element('event-list').innerHTML;
assert.equal(element('event-count').textContent,String(response.events.length));
assert.equal((html.match(/class="event-entry"/g)||[]).length,response.events.length);
for(const event of response.events) {
 assert.equal(event.action,String(event.side).toLowerCase());
 assert.equal(event.time,event.available_at.slice(0,10));
 const label=event.side==='BUY'?'买入':'卖出';
 assert(html.includes(`${label}意图`));assert(html.includes(sidebar.readableEvent(event)));
 assert(html.includes(sidebar.referencePriceText(event.reference_price,event.reference_price_date,`${label}参考价`)));
 assert(html.includes(`发生 ${event.time}<br>可用 ${event.available_at}`));
}
assert(!html.includes('无操作意图'));assert(!html.includes('未触发交易规则'));assert(!html.includes('发生 —'));
assert(!html.includes('参考价暂不可用'));
assert(element('event-context').innerHTML.includes('候选资格：本日无新候选'));
assert(!element('event-context').innerHTML.includes('候选资格：输入不足'));
const row=response.display_test_row || response.chart_evidence.rows.find(r=>r.date==='2026-06-04');
assert(row.agents.ma.probability>0);assert(row.agents.wyckoff.input_eligible===true);
for(const key of ['ma','wyckoff'])assert.equal(ui.agentView(row.agents[key],key).reason,'');
''', SIDEBAR_FIXTURE)


@pytest.mark.parametrize("frequency", ["日线", "周线", "月线"])
def test_real_response_workbench_hover_click_and_w_layer_toggle(frequency):
    """Only the browser's DOM/chart surfaces are mocked, not the response."""
    import json

    run_js("const frequency=" + json.dumps(frequency, ensure_ascii=False) + ";\n" + r'''
const vm=require('vm'),elements=new Map(),documentListeners={},windowListeners={};
function newElement(tag='div') {return {tag,dataset:{},style:{},checked:false,hidden:false,disabled:false,
 clientWidth:145,clientHeight:450,offsetWidth:460,offsetHeight:500,value:'',textContent:'',innerHTML:'',children:[],attributes:{},listeners:{},
 getBoundingClientRect:()=>({left:20,top:180}),addEventListener(key,fn){(this.listeners[key] ||= []).push(fn);},
 querySelector:()=>newElement(),replaceChildren(){this.children=[];this.innerHTML='';},append(child){this.children.push(child);},
 hasAttribute(key){return key in this.attributes;},setAttribute(key,value){this.attributes[key]=value;}};}
function element(id){if(!elements.has(id))elements.set(id,newElement());return elements.get(id);}
global.document={getElementById:element,documentElement:{},createElementNS:(_,tag)=>newElement(tag),
 querySelectorAll:()=>[],addEventListener(key,fn){documentListeners[key]=fn;}};
global.window={innerWidth:907,innerHeight:924,addEventListener(key,fn){windowListeners[key]=fn;}};
global.getComputedStyle=()=>({getPropertyValue:()=> '#123456'});global.ResizeObserver=class{observe(){}};
global.requestAnimationFrame=()=>1;global.location={hash:'analysis'};global.setTimeout=()=>1;global.clearTimeout=()=>{};
global.KHQuantMarkers=markers;global.KHQuantWyckoffMarkers=wyckoff;global.KHQuantChartEvidence=ui;
let crosshair,click,candles,referenceLinesCreated=0;
const scale={subscribeVisibleLogicalRangeChange(){},getVisibleLogicalRange:()=>({from:0,to:10}),
 setVisibleLogicalRange(){},fitContent(){},height:()=>30,timeToCoordinate:()=>50};
function series(){return {setData(){},setMarkers(value){this.markers=value;},applyOptions(){},
 priceScale:()=>({applyOptions(){}}),priceToCoordinate:price=>350-price,
 createPriceLine(){referenceLinesCreated++;return {};},removePriceLine(){}};}
const chart={addCandlestickSeries(){candles=series();return candles;},addHistogramSeries:series,addLineSeries:series,
 removeSeries(){},resize(){},subscribeCrosshairMove(fn){crosshair=fn;},subscribeClick(fn){click=fn;},
 timeScale:()=>scale,priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};window.LightweightCharts=LightweightCharts;
const code=fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);',
 'globalThis.liveInteraction={state,bind,renderAnalysis,selection:()=>chartSelection};');
vm.runInThisContext(code);
const {state,bind,renderAnalysis}=liveInteraction;
state.analysis=response;state.frequency=frequency;state.query={symbol:response.symbol,start:response.chart_evidence.rows[0].date,as_of:response.as_of};
state.dates=response.chart_evidence.rows.map(r=>r.date);state.replayIndex=state.dates.length-1;
element('layer-signals').checked=true;element('view-analysis').hidden=false;element('analysis-chart').clientWidth=640;
bind();renderAnalysis(false);
assert(!candles.markers.some(m=>m.text.startsWith('W ')));
const hover=time=>crosshair({time,point:{x:350,y:180},seriesData:new Map([[candles,{close:11}]])});
const available=wyckoff.buildDetailed({events:response.wyckoff_events,bars:response.frequencies[frequency].bars,
 frequency,asOf:response.as_of,symbol:response.symbol,start:state.query.start});
assert(available.groups.length>0);
const group=available.groups[0],selectedDate=ui.groupDates([group]).at(-1);
hover(group.time);assert(element('chart-evidence-popup').hidden);assert.equal(liveInteraction.selection().popupOpen,false);
element('layer-wyckoff').checked=true;element('layer-wyckoff').listeners.change[0]();
const drawn=candles.markers.filter(m=>m.text.startsWith('W '));
assert.equal(drawn.length,available.markers.length);
assert(element('marker-summary').textContent.includes(`量价事件 ${available.eventCount} 条／${drawn.length} 个标记`));
hover(group.time);assert(element('chart-evidence-popup').hidden);
element('chart-evidence-popup').scrollTop=777;
click({time:group.time,hoveredObjectId:group.id,point:{x:350,y:180}});
assert.equal(element('chart-evidence-popup').scrollTop,0,'new click starts at the decision header');
assert(!element('chart-evidence-popup').hidden);assert.equal(liveInteraction.selection().selectedDate,selectedDate);
const popup=element('chart-evidence-popup').innerHTML,selectedRow=ui.rowAt(response.chart_evidence,selectedDate);
assert(popup.includes(`信号可知日 ${selectedDate}`));assert(popup.includes(ui.referenceHtml(selectedRow)));
assert(popup.includes('原始量价事件'));assert(popup.includes('均线 ML'));assert(popup.includes('高收益二阶段'));
assert(popup.includes('当日需求'));assert(popup.includes('事件衰减强度'));
for(const entry of group.entries.filter(e=>e.event_date===selectedDate)) {
 assert(popup.includes(`首次可用 ${entry.first_available_at}`));
}
if(!response.consensus_decisions.events.some(e=>e.date===selectedDate))assert.equal(referenceLinesCreated,0);
// A pinned click persists while the pointer moves; closing it does not allow
// a subsequent hover to reopen any agent details.
hover(response.frequencies[frequency].bars.at(-1).time);assert.equal(liveInteraction.selection().selectedDate,selectedDate);
const close=newElement('button');close.setAttribute('data-chart-evidence-close','');
documentListeners.click({target:{closest:()=>close}});assert(element('chart-evidence-popup').hidden);
hover(group.time);assert(element('chart-evidence-popup').hidden);
click({time:group.time,hoveredObjectId:group.id,point:{x:350,y:180}});assert(!element('chart-evidence-popup').hidden);
element('layer-wyckoff').checked=false;element('layer-wyckoff').listeners.change[0]();
assert(element('chart-evidence-popup').hidden);assert.equal(liveInteraction.selection().pinned,null);
assert(!candles.markers.some(m=>m.text.startsWith('W ')));
hover(group.time);assert(element('chart-evidence-popup').hidden);
''')
