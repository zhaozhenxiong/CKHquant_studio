"""Exercise date controls and popup state through the shipped workbench events."""
import shutil
import subprocess
from pathlib import Path

import pytest


STATIC = Path(__file__).resolve().parents[1] / "web_dashboard/static/js"

HARNESS = r'''
const assert=require('node:assert/strict'),path=require('path'),fs=require('fs'),vm=require('vm');
global.KHQuantMarkers=require(path.join(process.argv[1],'chart_markers.js'));
global.KHQuantWyckoffMarkers=require(path.join(process.argv[1],'wyckoff_markers.js'));
global.KHQuantChartEvidence=require(path.join(process.argv[1],'chart_evidence.js'));
const elements=new Map();
function newElement(tag='div') {
  const el={tag,dataset:{},style:{},checked:false,hidden:false,disabled:false,value:'',textContent:'',className:'',
    clientWidth:640,clientHeight:450,offsetWidth:460,offsetHeight:500,scrollTop:0,options:[],listeners:{},attributes:{},children:[],details:[],
    getBoundingClientRect:()=>({left:20,top:180}),addEventListener(type,fn){(this.listeners[type] ||= []).push(fn);},
    setAttribute(key,value){this.attributes[key]=value;},hasAttribute(key){return key in this.attributes;},
    removeAttribute(key){delete this.attributes[key];},append(node){this.children.push(node);},
    replaceChildren(){this.children=[];this.innerHTML='';},
    querySelectorAll(selector){return selector==='details[data-agent-details]' ? this.details : [];},
    querySelector(selector){const key=selector.match(/^details\[data-agent-details="([^"]+)"\]$/)?.[1];
      return key ? this.details.find(node=>node.dataset.agentDetails===key) : newElement();}};
  let html='';
  Object.defineProperty(el,'innerHTML',{configurable:true,get(){return html;},set(value){html=String(value);this.scrollTop=0;
    this.details=[...html.matchAll(/<details\b[^>]*data-agent-details="([^"]+)"[^>]*>/g)].map(match=>({
      dataset:{agentDetails:match[1]},open:/\bopen\b/.test(match[0])}));}});
  return el;
}
function element(id){if(!elements.has(id))elements.set(id,newElement());return elements.get(id);}
global.document={getElementById:element,documentElement:{},querySelectorAll:()=>[],
  createElementNS:(_,tag)=>newElement(tag),addEventListener(){}};
global.window={innerWidth:977,innerHeight:904,addEventListener(){}};
global.location={hash:'analysis'};global.sessionStorage={getItem:()=>null};
global.getComputedStyle=()=>({getPropertyValue:()=> '#123456'});
global.ResizeObserver=class{observe(){}};global.requestAnimationFrame=()=>1;
global.setTimeout=()=>1;global.clearTimeout=()=>{};
let click;
const scale={subscribeVisibleLogicalRangeChange(){},getVisibleLogicalRange:()=>({from:0,to:1}),setVisibleLogicalRange(){},
  fitContent(){},height:()=>30,timeToCoordinate:()=>50};
function series(){return {setData(){},setMarkers(){},applyOptions(){},priceScale:()=>({applyOptions(){}}),priceToCoordinate:price=>350-price};}
const chart={addCandlestickSeries:series,addHistogramSeries:series,addLineSeries:series,removeSeries(){},resize(){},
  subscribeCrosshairMove(){},subscribeClick(fn){click=fn;},timeScale:()=>scale,priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};window.LightweightCharts=LightweightCharts;
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'workbench.js'),'utf8')
  .replace('async function analyze(query, preserveDates = false, locate = null) {',
    'async function analyze(query, preserveDates = false, locate = null) { globalThis.lastAnalysisQuery=query; return;')
  .replace('init().catch(showError);',
    'globalThis.workbench={state,bind,refreshLatestDates,refreshData,renderAnalysis,resizeCharts,renderChartPopup,analyzeSymbol,researchSpec,resultResearchContext,applyResearchContext,populateResearchModels};'));
workbench.bind();
async function fire(id,type,event={}){for(const fn of element(id).listeners[type] || [])await fn(event);}
async function settle(){for(let i=0;i<12;i++)await Promise.resolve();}
const response=value=>({ok:true,status:200,text:async()=>JSON.stringify(value)});
function pendingNetwork(){const requests=[];global.fetch=(url,options)=>new Promise(resolve=>requests.push({url,options,resolve}));return requests;}
const dates=(symbol,latest,market='2026-10-12')=>({symbol,latest_date:latest,symbol_latest_date:latest,market_latest_date:market,
  latest_symbols:1,total_symbols:2,partial_coverage:true,coverage_fraction:.5,scope:'latest_stored_closed_daily_bars'});
'''


def run_js(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped workbench date controls")
    result = subprocess.run([node, "-e", HARNESS + "\n(async()=>{\n" + script +
                             "\n})().catch(error=>{console.error(error);process.exitCode=1;});", str(STATIC)],
                            capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr


def test_latest_stock_and_market_dates_preserve_manual_history_and_restore_latest():
    run_js(r'''
element('analysis-symbol').value='002436.SZ';
let latest=dates('002436.SZ','2026-10-09');
global.fetch=async()=>response(latest);
await workbench.refreshLatestDates();
assert.equal(element('analysis-end').value,'2026-10-09','single-stock latest must use its actual last bar');
assert.equal(element('scan-end').value,'2026-10-12');assert.equal(element('backtest-end').value,'2026-10-12');
assert.equal(element('connection-state').textContent,'最新日部分覆盖');
assert(element('coverage-label').textContent.includes('1 / 2'));
element('analysis-end').value='2026-08-21';await fire('analysis-end','change');
element('scan-end').value='2026-08-31';await fire('scan-end','change');
assert.equal(element('analysis-date-mode').textContent,'历史截止日');
latest=dates('002436.SZ','2026-10-12','2026-10-13');
await workbench.refreshLatestDates();
assert.equal(element('analysis-end').value,'2026-08-21','manual history survives availability refresh');
assert.equal(element('scan-end').value,'2026-08-31');assert.equal(element('backtest-end').value,'2026-10-13');
await fire('analysis-latest','click');
assert.equal(element('analysis-end').value,'2026-10-12');
assert.equal(element('analysis-date-mode').textContent,'自动最新可用行情');
latest=dates('002436.SZ',null,'2026-10-13');await workbench.refreshLatestDates();
assert.equal(element('analysis-end').value,'','a stock with no bars cannot inherit another stock date');
''')


def test_out_of_order_availability_responses_cannot_overwrite_new_stock_date():
    run_js(r'''
const requests=pendingNetwork();element('analysis-end').value='2026-10-09';
element('analysis-symbol').value='002436.SZ';const first=workbench.refreshLatestDates();
element('analysis-symbol').value='000001.SZ';await fire('analysis-symbol','change');
assert.equal(requests.length,2);assert(requests[0].url.includes('002436.SZ'));assert(requests[1].url.includes('000001.SZ'));
requests[1].resolve(response(dates('000001.SZ','2026-10-12')));await settle();
assert.equal(element('analysis-end').value,'2026-10-12');
requests[0].resolve(response(dates('002436.SZ','2026-10-08')));await first;await settle();
assert.equal(element('analysis-end').value,'2026-10-12','older response must not replace the new symbol date');
''')


def test_stock_changed_by_typing_rejects_previous_stock_availability_before_blur():
    run_js(r'''
const requests=pendingNetwork();element('analysis-end').value='2026-10-09';
element('analysis-symbol').value='002436.SZ';const first=workbench.refreshLatestDates();
element('analysis-symbol').value='000001.SZ';await fire('analysis-symbol','input');
requests[0].resolve(response(dates('002436.SZ','2026-10-08')));await first;await settle();
assert.equal(element('analysis-end').value,'2026-10-09','a reply for the stock before the edit must not change the cutoff');
''')


def test_late_general_data_status_does_not_erase_partial_coverage_from_latest_api():
    run_js(r'''
const requests=pendingNetwork();element('analysis-symbol').value='002436.SZ';
const general=workbench.refreshData(),latest=workbench.refreshLatestDates();
requests[1].resolve(response(dates('002436.SZ','2026-10-09')));await latest;
requests[0].resolve(response({latest_date:'2026-10-12',stocks:2,rows:200,source:'SQLite'}));await general;
assert.equal(element('analysis-end').value,'2026-10-09');
assert.equal(element('connection-state').textContent,'最新日部分覆盖','startup request order must not erase coverage state');
assert(element('coverage-label').textContent.includes('1 / 2'));
''')


def test_popup_resize_keeps_user_expansion_and_scroll_even_when_content_refreshes():
    run_js(r'''
const dates=['2026-07-01','2026-07-02'];
const bars=dates.map(time=>({time,open:10,high:12,low:9,close:11,volume:100,is_closed:true}));
const events=dates.map((day,index)=>({symbol:'002436.SZ',event:'SOS',event_id:'pending-'+index,confirmed:false,
  observed_at:day+'T15:00:00+08:00',available_at:day+'T15:00:00+08:00'}));
workbench.state.analysis={symbol:'002436.SZ',as_of:dates[1],data_end:dates[1],wyckoff_events:events,
  frequencies:{'日线':{bars,indicators:{rows:bars}}},chart_evidence:{rows:dates.map(date=>({date,reference_close:11,agents:{}}))}};
workbench.state.query={symbol:'002436.SZ',start:dates[0],end:dates[1]};
element('view-analysis').hidden=false;element('layer-wyckoff').checked=true;
workbench.renderAnalysis(false);click({time:dates[0],point:{x:350,y:180}});
const popup=element('chart-evidence-popup');assert(!popup.hidden);assert.equal(popup.details.length,4);
assert(popup.details.every(node=>node.open));
popup.querySelector('details[data-agent-details="ma"]').open=false;
popup.querySelector('details[data-agent-details="wyckoff"]').open=false;popup.scrollTop=321;
window.innerWidth=700;workbench.resizeCharts();workbench.renderChartPopup();
assert.equal(popup.scrollTop,321);assert.equal(popup.querySelector('details[data-agent-details="ma"]').open,false);
assert.equal(popup.querySelector('details[data-agent-details="structure"]').open,true);
assert.equal(popup.querySelector('details[data-agent-details="wyckoff"]').open,false);
assert(parseFloat(popup.style.left)+popup.offsetWidth<=window.innerWidth-12);
workbench.state.analysis.chart_evidence.rows[0].reference_close=12;workbench.renderChartPopup();
assert(popup.innerHTML.includes('收盘参考价 12 元'));
assert.equal(popup.scrollTop,321,'refreshing a pinned daily card preserves the user scroll');
assert.equal(popup.querySelector('details[data-agent-details="ma"]').open,false);
assert.equal(popup.querySelector('details[data-agent-details="wyckoff"]').open,false);
click({time:dates[1],point:{x:350,y:180}});
assert.equal(popup.scrollTop,0,'a different daily card starts from its header');
assert(popup.details.every(node=>node.open),'a new date starts with useful agent details expanded');
''')


def test_scan_history_before_display_start_drills_without_resetting_account_context():
    run_js(r'''
const source={strategy_version:'trend_consensus',request:{start:'2026-10-01',research:true,model_family:'current',
  model_run_id:'saved-model',usage_mode:'historical',model_policy:'auto',calendar_run_id:'verified-calendar',
  account_start:'2023-01-01',account_initial_cash:100000,consensus_version:'trend_consensus',
  consensus_config:{version:'trend_consensus',quality_weight:.1}},
  data_range:{start:'2026-10-01',end:'2026-10-12'},market_history:{window_start:'2026-09-01'}};
await workbench.analyzeSymbol('002436.SZ','2026-09-15',source);
assert.equal(lastAnalysisQuery.start,'2026-09-01');assert.equal(lastAnalysisQuery.end,'2026-09-15');
assert.equal(lastAnalysisQuery.as_of,'2026-09-15');assert.equal(lastAnalysisQuery.account_start,'2023-01-01');
assert.equal(lastAnalysisQuery.model_run_id,'saved-model');assert.equal(lastAnalysisQuery.calendar_run_id,'verified-calendar');
assert.deepEqual(lastAnalysisQuery.consensus_config,source.request.consensus_config);
await workbench.analyzeSymbol('002436.SZ','2026-10-09',source);
assert.equal(lastAnalysisQuery.start,'2026-10-01','valid requested display start is preserved');
delete source.market_history;
await workbench.analyzeSymbol('002436.SZ','2026-09-15',source);
assert.equal(lastAnalysisQuery.start,'2026-09-15','an older run without a window still has a valid display interval');
''')


def test_reference_checkpoint_controls_keep_model_context_without_simulated_account():
    run_js(r'''
workbench.state.research={calendar_run_id:'calendar',default_consensus_version:'trend_consensus',
 models:[{run_id:'new-run',schema_family:'trend_consensus'}]};
workbench.state.pendingResearchContexts.analysis={model_run_id:'new-run',consensus_version:'trend_consensus',
 account_start:'2023-01-01',account_initial_cash:150000,consensus_config:{version:'trend_consensus'}};
element('analysis-mode').value='research';element('analysis-model-family').value='current';
element('analysis-model').value='new-run';element('analysis-model-policy').value='auto';element('analysis-usage-mode').value='historical';
const actual=workbench.researchSpec('analysis');
assert.equal(actual.consensus_version,'trend_consensus');assert.equal(actual.model_run_id,'new-run');
assert.equal(actual.analysis_mode,'reference');assert.equal(actual.account_start,undefined);assert.equal(actual.account_initial_cash,undefined);
assert.equal(actual.consensus_config.version,'trend_consensus');
const context=workbench.resultResearchContext({strategy_version:'trend_consensus',research:{model_run_id:'new-run'},
 request:{consensus_version:'trend_consensus',account_start:'2023-01-01',account_initial_cash:150000}});
assert.equal(context.consensus_version,'trend_consensus');assert.equal(context.account_initial_cash,150000);
''')


def test_saved_model_missing_from_recent_list_remains_bound_in_real_select_semantics():
    run_js(r'''
// A native select returns an empty value when an absent option is selected.
const select=element('analysis-model');let selected='',html='';
Object.defineProperty(select,'innerHTML',{configurable:true,get(){return html;},set(value){
 html=String(value);this.options=[...html.matchAll(/<option value="([^"]*)"/g)].map(match=>({value:match[1]}));selected='';}});
Object.defineProperty(select,'value',{configurable:true,get(){return selected;},set(value){
 selected=this.options.some(option=>option.value===value) ? value : '';}});
workbench.state.research={calendar_run_id:'calendar',default_consensus_version:'trend_consensus',
 models:[{run_id:'recent-run',schema_family:'trend_consensus'}]};
const context={research:true,model_family:'current',model_run_id:'old-explicit-run',model_policy:'auto',
 usage_mode:'historical',consensus_version:'trend_consensus',account_start:'2023-01-01',
 account_initial_cash:150000,consensus_config:{version:'trend_consensus',high_return_threshold:.1}};
workbench.applyResearchContext('analysis',context);
assert.equal(select.value,'old-explicit-run');
assert(select.options.some(option=>option.value==='old-explicit-run'));
workbench.populateResearchModels('analysis');
const actual=workbench.researchSpec('analysis');
assert.equal(actual.model_run_id,'old-explicit-run');assert.equal(actual.consensus_version,context.consensus_version);
assert.equal(actual.analysis_mode,'reference');assert.equal(actual.account_start,undefined);assert.equal(actual.account_initial_cash,undefined);
assert.deepEqual(actual.consensus_config,context.consensus_config);
assert.throws(()=>workbench.applyResearchContext('analysis',{...context,model_family:'retired_family'}),/已退役/);
workbench.applyResearchContext('analysis',context);
select.value='recent-run';await fire('analysis-model','change');
assert.equal(workbench.state.pendingResearchContexts.analysis,undefined,'the new manual model does not inherit the previous account');
assert.equal(workbench.researchSpec('analysis').model_run_id,'recent-run');
assert.equal(workbench.researchSpec('analysis').account_initial_cash,undefined);
''')


def test_absent_saved_pinned_checkpoint_preserves_native_and_new_family_identity():
    run_js(r'''
for(const id of ['analysis-model','analysis-checkpoint']){
 const select=element(id);let selected='',html='';
 Object.defineProperty(select,'innerHTML',{configurable:true,get(){return html;},set(value){
  html=String(value);this.options=[...html.matchAll(/<option value="([^"]*)"/g)].map(match=>({value:match[1]}));
  if(!this.options.some(option=>option.value===selected))selected='';}});
 Object.defineProperty(select,'value',{configurable:true,get(){return selected;},set(value){
  selected=this.options.some(option=>option.value===value) ? value : '';}});
}
workbench.state.research={default_consensus_version:'trend_consensus',models:[]};
element('analysis-end').value='2026-10-12';
for(const version of ['trend_consensus']){
 const saved={research:true,model_family:'current',model_run_id:'older-'+version,model_policy:'pinned',
  model_fold:'2026Q3',usage_mode:'historical',consensus_version:version,consensus_config:{version}};
 workbench.applyResearchContext('analysis',saved);
 assert.equal(element('analysis-model').value,saved.model_run_id);
 assert.equal(element('analysis-checkpoint').value,'2026Q3');assert(!element('analysis-checkpoint').disabled);
 const actual=workbench.researchSpec('analysis');
 assert.equal(actual.model_run_id,saved.model_run_id);assert.equal(actual.model_fold,'2026Q3');
 assert.equal(actual.consensus_version,version);assert.deepEqual(actual.consensus_config,saved.consensus_config);
}
''')


def test_retired_saved_context_is_rejected_without_changing_current_model():
    run_js(r'''
workbench.state.research={models:[{run_id:'active-current',model_family:'current',strategy_version:'trend_consensus'}]};
element('analysis-model-family').value='current';element('analysis-mode').value='research';
workbench.populateResearchModels('analysis');
assert.throws(()=>workbench.applyResearchContext('analysis',{research:true,model_family:'high_return',model_run_id:'retired',consensus_version:'czsc_trend_consensus'}),/已退役/);
assert.equal(workbench.state.pendingResearchContexts.analysis,undefined);
assert(!element('analysis-model').innerHTML.includes('retired'));
assert(workbench.researchSpec('analysis').model_family==='current');
const currentContext={research:true,model_family:'current',consensus_version:'trend_consensus',model_run_id:'active-current',usage_mode:'forward'};
workbench.applyResearchContext('analysis',currentContext);
assert.equal(workbench.researchSpec('analysis').usage_mode,'forward');
const savedPending=workbench.state.pendingResearchContexts.analysis;
assert.throws(()=>workbench.applyResearchContext('analysis',{...currentContext,usage_mode:'production'}),/旧用途已退役，不恢复资格/);
assert.equal(workbench.state.pendingResearchContexts.analysis,savedPending);
assert.equal(element('analysis-usage-mode').value,'forward');
element('analysis-usage-mode').value='production';
assert.throws(()=>workbench.researchSpec('analysis'),/旧用途已退役/);
''')
