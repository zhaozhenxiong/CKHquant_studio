"""Exercise shipped workbench ranges, archive boundaries and causal history caching."""
import shutil
import subprocess
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js'


HARNESS = r'''
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('node:assert/strict');
const root=process.argv[1];require(path.join(root,'chart_evidence.js'));
global.KHQuantMarkers=require(path.join(root,'chart_markers.js'));require(path.join(root,'wyckoff_markers.js'));
const elements=new Map(),listeners={};
function newElement(){return {dataset:{},style:{},checked:false,hidden:false,disabled:false,value:'',textContent:'',innerHTML:'',children:[],attributes:{},clientWidth:700,clientHeight:400,offsetWidth:400,offsetHeight:400,
addEventListener(key,fn){this[key]=fn;},getBoundingClientRect:()=>({left:0,top:0}),querySelector:()=>newElement(),replaceChildren(){this.children=[];this.innerHTML='';},append(child){this.children.push(child);},setAttribute(key,value){this.attributes[key]=value;},hasAttribute(key){return key in this.attributes;}};}
function el(id){if(!elements.has(id))elements.set(id,newElement());return elements.get(id);}
global.document={getElementById:el,documentElement:{},createElementNS:()=>newElement(),querySelectorAll:()=>[],addEventListener(key,fn){listeners[key]=fn;}};
global.window={innerWidth:1000,innerHeight:900,addEventListener(){}};global.location={hash:''};
global.getComputedStyle=()=>({getPropertyValue:()=> '#123456'});global.ResizeObserver=class{observe(){}};
global.requestAnimationFrame=()=>1;global.setTimeout=()=>1;global.clearTimeout=()=>{};
global.sessionStorage={getItem:()=>null};global.localStorage={getItem:()=>null,setItem(){}};
let range={from:0,to:4},candles;let rangeCallback;
const scale={subscribeVisibleLogicalRangeChange(fn){rangeCallback=fn;},getVisibleLogicalRange:()=>range,
setVisibleLogicalRange(value){range=value;rangeCallback?.();},fitContent(){range={from:0,to:4};rangeCallback?.();},height:()=>25,timeToCoordinate:()=>50};
const series=()=>({data:[],markers:[],setData(x){this.data=x;},setMarkers(x){this.markers=x;},applyOptions(){},priceScale:()=>({applyOptions(){}}),priceToCoordinate:()=>100,createPriceLine:x=>x,removePriceLine(){}});
const chart={addCandlestickSeries:()=>candles=series(),addHistogramSeries:series,addLineSeries:series,removeSeries(){},resize(){},subscribeClick(){},subscribeCrosshairMove(){},timeScale:()=>scale,priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};window.LightweightCharts=LightweightCharts;
const source=fs.readFileSync(path.join(root,'workbench.js'),'utf8').replace('init().catch(showError);',
'globalThis.ui={state,bind,applyChartRange,syncChartRange,renderAnalysis,renderEvents,renderMarkerScope,locateLatestSell,resetChartSelection,updateChartSelection,referenceEventCounts,filteredReferenceEvents,latestSellEvent,prepareAnalysisView,currentReanalysisQuery,openRun,replay,analysisHistoryIdentity,storeHistoryEvidence,scanActionCell,history:()=>historyRows};');
vm.runInThisContext(source);
const {state}=ui;
const symbol='002436.SZ',dates=['2026-01-05','2026-01-09','2026-03-09','2026-03-10','2026-07-31'];
const bars=dates.map(time=>({time,is_closed:true,open:10,high:12,low:9,close:11,volume:100}));
const event=(id,date,action,source='reference_consensus')=>({event_id:id,symbol,date,time:date,reference_price:11,reference_price_date:date,first_available_at:date+'T15:00:00+08:00',available_at:date+'T15:00:00+08:00',action,side:action,source,reference_only:true,confirmed:true,candidate_eligible:true,input_eligible:true,evidence_visible:true});
const target=[event('b1',dates[0],'BUY'),event('s1',dates[1],'SELL'),event('b2',dates[2],'BUY'),event('s2',dates[3],'SELL')];
const sources=[event('source-b',dates[2],'BUY','structure'),event('source-s1',dates[1],'SELL','structure'),event('source-s2',dates[4],'SELL','wyckoff')];
function result(version='reference_position_v1'){return {symbol,run_id:'current',strategy_version:version,analysis_mode:'reference',as_of:dates.at(-1),data_end:dates.at(-1),request:{symbol,start:dates[0],end:dates.at(-1),research:true,model_family:'current',usage_mode:'historical',model_policy:'auto',consensus_version:version},
reference_points:target,source_reference_points:sources,trend_consensus:{version,config:{version},rows:dates.map(date=>({date,available_at:date+'T15:00:00+08:00',agents:{}})),events:target},
chart_evidence:{rows:dates.map(date=>({date,available_at:date+'T15:00:00+08:00',symbol,analysis_mode:'reference',agents:{}}))},position_reference:{schema:version,status:'pending',target_position_fraction:0},
frequencies:{'日线':{bars},'周线':{bars},'月线':{bars}}};}
state.analysis=result();state.query={symbol,start:dates[0],as_of:dates.at(-1)};state.dates=dates;
el('layer-sources').checked=true;el('layer-signals').checked=true;el('event-layer-filter').value='all';el('event-side-filter').value='all';
'''


def run_js(body):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for workbench behavior checks')
    result = subprocess.run([node, '-e', HARNESS + body, str(STATIC)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_full_loaded_visible_counts_filters_and_sell_navigation_are_independent():
    run_js(r'''
ui.renderAnalysis(false);ui.renderEvents();
assert(el('marker-total-summary').textContent.includes('来源 B 1 / S 2 · 目标变化 B 2 / S 2'));
assert.equal(candles.markers.filter(x=>x.text.endsWith('S')).length,4);
scale.setVisibleLogicalRange({from:4,to:4});
assert(el('marker-view-summary').textContent.includes('来源 B 0 / S 1 · 目标变化 B 0 / S 0'));
assert(el('marker-empty-explanation').textContent.includes('目标 S 在当前视窗外'));
assert(el('marker-empty-explanation').textContent.includes('参考目标为 0'));
el('layer-sources').checked=false;ui.renderAnalysis(false);
assert(el('marker-total-summary').textContent.includes('来源 B 1 / S 2'));
assert(el('marker-summary').textContent.includes('来源 B 0 / S 0'));
assert(el('marker-empty-explanation').textContent.includes('来源图层未启用'));
el('event-layer-filter').value='target';el('event-side-filter').value='SELL';ui.renderEvents();
assert.equal(el('event-count').textContent,'2 / 7');
assert(el('event-list').innerHTML.includes('2026-03-10'));
assert(!el('event-list').innerHTML.includes('2026-07-31'));
assert.equal(ui.latestSellEvent([...sources.map(x=>({...x,reference_layer:'source'})),...target],'target').date,'2026-03-10');
el('layer-signals').checked=false;ui.locateLatestSell();
assert.equal(el('layer-signals').checked,true);assert.equal(el('event-side-filter').value,'SELL');
assert(range.from<=3 && range.to>=3);
assert(el('marker-total-summary').textContent.includes('目标变化 B 2 / S 2'));
''')


def test_explicit_target_sell_navigation_ignores_later_source_sell_and_does_not_cover_chart():
    run_js(r'''
global.fetch=()=>{throw new Error('navigation must not recompute analysis');};
ui.renderAnalysis(false);scale.setVisibleLogicalRange({from:4,to:4});
el('layer-signals').checked=false;
ui.locateLatestSell('target');
assert.equal(el('event-layer-filter').value,'target');assert.equal(el('event-side-filter').value,'SELL');
assert.equal(el('layer-signals').checked,true);
assert(range.from<=3 && range.to>=3);
const selected=candles.markers.filter(x=>x.text.includes('定位'));
assert.equal(selected.length,1);assert.equal(selected[0].time,'2026-03-10');
assert.equal(selected[0].text,'参S · 定位');assert.equal(selected[0].size,1.7);
assert.equal(el('chart-evidence-popup').hidden,true);
assert(el('marker-location-summary').textContent.includes('2026-03-10 目标 S · 参考减仓'));
assert(el('marker-location-summary').textContent.includes('箭头绘于对应 K 线上方'));
assert(el('chart-locate-source-s').textContent.includes('2026-07-31'));
assert(el('chart-locate-target-s').textContent.includes('2026-03-10'));
assert.equal(state.analysis.reference_points.length,4);
assert(el('marker-total-summary').textContent.includes('目标变化 B 2 / S 2'));
''')


def test_source_sell_navigation_replaces_target_highlight_and_selection_reset_clears_it():
    run_js(r'''
ui.renderAnalysis(false);ui.locateLatestSell('target');
el('layer-sources').checked=false;ui.locateLatestSell('source');
assert.equal(el('layer-sources').checked,true);assert.equal(el('event-layer-filter').value,'source');
const selected=candles.markers.filter(x=>x.text.includes('定位'));
assert.equal(selected.length,1);assert.equal(selected[0].time,'2026-07-31');
assert.equal(selected[0].text,'量S · 定位');
assert.equal(el('chart-evidence-popup').hidden,true);
assert(el('marker-location-summary').textContent.includes('2026-07-31 来源 S'));
ui.resetChartSelection();
assert.equal(candles.markers.some(x=>x.text.includes('定位')),false);
assert.equal(el('marker-location-summary').hidden,true);
assert.equal(candles.markers.filter(x=>x.text.endsWith('S')).length,4);
''')


def test_sell_navigation_disables_missing_target_without_manufacturing_events():
    run_js(r'''
state.analysis.reference_points=[];state.analysis.trend_consensus.events=[];
ui.renderAnalysis(false);ui.renderEvents();
assert.equal(el('chart-locate-target-s').disabled,true);
assert(el('chart-locate-target-s').textContent.includes('无记录'));
assert.equal(el('chart-locate-source-s').disabled,false);
assert(el('marker-total-summary').textContent.includes('目标变化 B 0 / S 0'));
assert.equal(candles.markers.some(x=>x.text==='参S'),false);
''')


def test_user_clear_and_manual_candle_selection_remove_prior_sell_focus():
    run_js(r'''
ui.renderAnalysis(false);
for(const action of ['clear','pin']) {
  ui.locateLatestSell('target');
  assert(candles.markers.some(x=>x.text==='参S · 定位'));
  // These are the actual transition calls used by Escape/reset and chart clicks.
  ui.updateChartSelection(action,'2026-07-31',null,'2026-07-31');
  assert.equal(candles.markers.some(x=>x.text.includes('定位')),false);
  assert.equal(el('marker-location-summary').hidden,true);
  assert.equal(el('marker-location-summary').textContent,'');
}
''')


def test_weekly_sell_navigation_highlights_projection_but_preserves_original_daily_date():
    run_js(r'''
state.frequency='周线';
state.analysis.frequencies['周线'].bars=['2026-01-09','2026-03-13','2026-07-31'].map(time=>({...bars[0],time}));
ui.renderAnalysis(false);ui.locateLatestSell('target');
const selected=candles.markers.find(x=>x.text==='参S · 定位');
assert(selected);assert.equal(selected.time,'2026-03-13');
assert(el('marker-location-summary').textContent.includes('2026-03-10'));
assert(el('event-list').innerHTML.includes('2026-03-10'));
assert.equal(el('chart-evidence-popup').hidden,true);
''')


def test_archive_load_is_readonly_and_invalid_identity_preserves_previous_chart():
    run_js(r'''
(async()=>{
ui.renderAnalysis(false);
const archive=result('trend_consensus');archive.run_id='archive';archive.request.model_run_id='retired-model';archive.request.consensus_config={version:'trend_consensus',old:true};
el('analysis-model').value='current-form-model';
let payload={kind:'analysis',result:archive,created_at:'2026-10-14T22:18:26+08:00'},calls=[];
global.fetch=async(url,options={})=>{calls.push({url,options});return {ok:true,status:200,text:async()=>JSON.stringify(payload)};};
await ui.openRun('archive','analysis');
assert.equal(state.analysis.run_id,'archive');
assert.equal(state.archive,true);assert.equal(el('analysis-model').value,'current-form-model');
assert(!Object.hasOwn(state.query,'model_run_id'));assert(!Object.hasOwn(state.query,'consensus_config'));
assert(el('analysis-result-caption').textContent.includes('只读归档 · trend_consensus'));
assert.equal(el('replay-slider').disabled,true);
await ui.replay(0);assert.equal(calls.length,1);
const query=ui.currentReanalysisQuery(archive);
assert.equal(query.consensus_version,'reference_position_v1');assert.equal(query.model_policy,'auto');
for(const key of ['model_run_id','model_fold','calendar_run_id','consensus_config','current_position_fraction']) assert(!Object.hasOwn(query,key));
const previous=state.analysis,previousMarkers=JSON.stringify(candles.markers),previousQuery=state.query;
payload={kind:'analysis',result:{...result(),run_id:'bad',request:{...result().request,symbol:'600519.SH'}}};
await assert.rejects(ui.openRun('bad','analysis'),/股票身份不一致/);
assert.equal(state.analysis,previous);assert.equal(state.query,previousQuery);assert.equal(JSON.stringify(candles.markers),previousMarkers);
payload={kind:'analysis',result:{...result(),run_id:'bad-context',request:{...result().request,usage_mode:'retired'}}};
await assert.rejects(ui.openRun('bad-context','analysis'),/旧用途已退役/);
assert.equal(state.analysis,previous);assert.equal(JSON.stringify(candles.markers),previousMarkers);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_history_identity_switches_on_schema_usage_model_calendar_source_and_policy_and_rejects_future_rows():
    run_js(r'''
const base=result();base.research={source_context_hash:'source-a'};base.request.calendar_run_id='calendar-a';base.trend_consensus.rows[0].agents.ma={model_identity:{manifest_sha256:'report-a',checkpoint:'Q1',score_date:dates[0]}};
const key=ui.analysisHistoryIdentity(base);
for(const change of [r=>r.strategy_version='trend_consensus',r=>r.request.usage_mode='forward',r=>r.request.calendar_run_id='calendar-b',r=>r.request.consensus_config={position_step:.1},r=>r.research.source_context_hash='source-b',r=>r.trend_consensus.rows[0].agents.ma.model_identity.manifest_sha256='report-b']) {
const changed=JSON.parse(JSON.stringify(base));change(changed);assert.notEqual(ui.analysisHistoryIdentity(changed),key);
}
const earlier={date:dates[0],available_at:dates[0]+'T15:00:00+08:00'},later={date:dates[1],available_at:dates[1]+'T15:00:00+08:00'};
ui.storeHistoryEvidence([earlier,later],false,key,dates[1]);assert.equal(ui.history().length,2);
ui.storeHistoryEvidence([earlier],true,key,dates[0]);assert.deepEqual(ui.history(),[earlier]);
ui.storeHistoryEvidence([{...earlier,available_at:dates[1]+'T15:00:00+08:00'}],false,key,dates[0]);assert.deepEqual(ui.history(),[]);
ui.storeHistoryEvidence([{...earlier,available_at:dates[0]+'T15:00:00+08:00'}],false,key,dates[0]+'T14:59:00+08:00');assert.deepEqual(ui.history(),[]);
ui.storeHistoryEvidence([earlier],false,key,dates[1]);ui.storeHistoryEvidence([later],true,key+'changed',dates[1]);assert.deepEqual(ui.history(),[later]);
// Date-only cutoffs mean Shanghai close, including all nested observation clocks.
for(const clock of ['2026-01-05T15:00:01+08:00','2026-01-05T16:00:00+08:00','2026-01-05T07:00:01Z','2026-01-05T16:00:00']) {
  ui.storeHistoryEvidence([{...earlier,first_available_at:clock}],false,key,dates[0]);assert.deepEqual(ui.history(),[]);
  ui.storeHistoryEvidence([{...earlier,agents:{ma:{first_available_at:clock}}}],false,key,dates[0]);assert.deepEqual(ui.history(),[]);
  ui.storeHistoryEvidence([{...earlier,events:[{confirmed_at:clock}]}],false,key,dates[0]);assert.deepEqual(ui.history(),[]);
}
ui.storeHistoryEvidence([{...earlier,first_available_at:'2026-01-05T07:00:00Z'}],false,key,dates[0]);assert.equal(ui.history().length,1);
ui.storeHistoryEvidence([{...earlier,agents:{ma:{model_identity:{actual_training_completed_at:'2026-10-15T00:00:00+08:00',available_at:'2026-10-15'}}}}],false,key,dates[0]);assert.equal(ui.history().length,1);
const assets={run_id:'run',report_sha256:'report-a',manifest_sha256:'manifest-a',weights_sha256:'weights-a',feature_schema_hash:'schema-a'};
base.model_asset_identities=[assets];base.runtime_code_hash='runtime-a';base.reference_context_hash='context-a';
const assetKey=ui.analysisHistoryIdentity(base);
for(const name of ['report_sha256','manifest_sha256','weights_sha256']) {const changed=JSON.parse(JSON.stringify(base));changed.model_asset_identities[0][name]+='changed';assert.notEqual(ui.analysisHistoryIdentity(changed),assetKey);}
for(const name of ['runtime_code_hash','reference_context_hash']) {const changed=JSON.parse(JSON.stringify(base));changed[name]+='changed';assert.notEqual(ui.analysisHistoryIdentity(changed),assetKey);}
for(const location of ['agent','nested','resolution']) {
 const changed=JSON.parse(JSON.stringify(base));
 if(location==='agent')changed.trend_consensus.rows[0].agents.ma.model_asset_identity={...assets,weights_sha256:'different'};
 if(location==='nested')changed.trend_consensus.rows[0].agents.ma.model_identity.model_asset_identity={...assets,report_sha256:'different'};
 if(location==='resolution')changed.trend_consensus.rows[0].model_resolution={model_asset_identity:{...assets,manifest_sha256:'different'}};
 assert.notEqual(ui.analysisHistoryIdentity(changed),assetKey);
}
''')


def test_current_scan_uses_probability_edge_and_archived_scan_keeps_rule_support():
    run_js(r'''
const ma={kind:'trend_probability',status:'scored',p_up:.6,p_down:.1,p_neutral:.3};
const base={action:'WAIT',agents:{ma},buy_base_score:.6,sell_base_score:.1};
for(const row of [{...base,position_reference:{schema:'reference_position_v1'}},{...base,version:'reference_position_v1'},{...base,strategy_version:'reference_position_v1'}]) {
 const html=ui.scanActionCell(row);assert(html.includes('上涨减下跌 50 个百分点'));assert(!html.includes('买方支持'));assert(!html.includes('卖方支持'));
}
state.scan={strategy_version:'reference_position_v1'};
assert(ui.scanActionCell(base).includes('上涨减下跌 50 个百分点'));
const unscored=ui.scanActionCell({...base,agents:{ma:{...ma,status:'model_unavailable'}}});assert(!unscored.includes('上涨减下跌'));assert(!unscored.includes('买方支持'));
const legacy=ui.scanActionCell({...base,version:'trend_consensus'});assert(legacy.includes('买方支持 60 分 · 卖方支持 10 分'));assert(!legacy.includes('上涨减下跌'));
''')


def test_bound_range_input_draft_survives_change_hover_clear_and_applies_captured_dates():
    run_js(r'''
ui.bind();ui.renderAnalysis(false);ui.locateLatestSell('source');
const before=JSON.stringify(range);
el('chart-range-start').value='2026-03-09';el('chart-range-start').input();
ui.updateChartSelection('hover','2026-01-09');ui.syncChartRange();
assert.equal(el('chart-range-start').value,'2026-03-09');
el('chart-range-start').change();
assert.equal(el('chart-range-start').value,'2026-03-09');
assert.equal(candles.markers.some(x=>x.text.includes('定位')),false);
el('chart-range-end').value='2026-03-10';el('chart-range-end').input();el('chart-range-end').change();
ui.updateChartSelection('pin','2026-07-31');ui.updateChartSelection('clear');
assert.equal(el('chart-range-start').value,'2026-03-09');assert.equal(el('chart-range-end').value,'2026-03-10');
el('chart-range-apply').click();
assert.deepEqual(range,{from:1.5,to:3.5});
assert.equal(el('chart-range-start').value,'2026-03-09');assert.equal(el('chart-range-end').value,'2026-03-10');
assert(el('chart-range-summary').textContent.includes('显示 2 根'));
// Clicking Apply also captures values when the browser has not dispatched change yet.
el('chart-range-start').value='2026-01-05';el('chart-range-end').value='2026-01-09';el('chart-range-apply').click();
assert.deepEqual(range,{from:-.5,to:1.5});
''')


def test_bound_invalid_range_preserves_draft_and_selection_until_explicit_navigation():
    run_js(r'''
ui.bind();ui.renderAnalysis(false);ui.locateLatestSell('target');
const before=JSON.stringify(range);
el('chart-range-start').value='2026-07-31';el('chart-range-start').input();
el('chart-range-end').value='2026-01-09';el('chart-range-end').input();
el('chart-range-apply').click();
assert.equal(JSON.stringify(range),before);assert.equal(el('chart-range-start').value,'2026-07-31');assert.equal(el('chart-range-end').value,'2026-01-09');
assert(candles.markers.some(x=>x.text==='参S · 定位'));
ui.syncChartRange();assert.equal(el('chart-range-start').value,'2026-07-31');
el('chart-range-start').value='2026-02-01';el('chart-range-end').value='2026-02-02';el('chart-range-apply').click();
assert.equal(JSON.stringify(range),before);assert.equal(el('chart-range-start').value,'2026-02-01');
assert(el('notice').textContent.includes('没有可显示'));
el('chart-range-reset').click();assert.equal(el('chart-range-start').value,dates[0]);assert.equal(el('chart-range-end').value,dates.at(-1));
el('chart-range-start').value='2026-03-09';el('chart-range-start').input();el('chart-zoom-in').click();
assert.notEqual(el('chart-range-start').value,'2026-03-09');
el('chart-range-start').value='2026-03-09';el('chart-range-start').input();ui.locateLatestSell('source');
assert.equal(el('chart-range-start').value,dates[0]);
el('chart-range-start').value='2026-03-09';el('chart-range-start').input();scale.setVisibleLogicalRange({from:1,to:3});
assert.equal(el('chart-range-start').value,'2026-01-09');assert.equal(el('chart-range-end').value,'2026-03-10');
// A new result or period render uses its actual view rather than the preceding draft.
el('chart-range-start').value='2026-07-31';el('chart-range-start').input();state.frequency='周线';ui.renderAnalysis();
assert.equal(el('chart-range-start').value,dates[0]);assert.equal(el('chart-range-end').value,dates.at(-1));
''')


def test_unidentified_structural_markers_are_not_focused_without_a_sell_selection():
    run_js(r'''
state.analysis.frequencies['日线'].fractals=[{time:dates[0],kind:'top',is_confirmed:true}];
state.analysis.frequencies['日线'].divergences=[{time:dates[1],kind:'top',label:'顶背驰',confirmed_at:dates[1]+'T15:00:00+08:00',available_at:dates[1]+'T15:00:00+08:00'}];
el('layer-fractals').checked=true;el('layer-divergences').checked=true;
ui.renderAnalysis(false);
assert.equal(candles.markers.some(marker=>marker.text.includes('定位')),false);
ui.locateLatestSell('target');
assert.equal(candles.markers.filter(marker=>marker.text.includes('定位')).length,1);
assert.equal(candles.markers.find(marker=>marker.text==='参S · 定位').time,'2026-03-10');
ui.updateChartSelection('clear');assert.equal(candles.markers.some(marker=>marker.text.includes('定位')),false);
''')
