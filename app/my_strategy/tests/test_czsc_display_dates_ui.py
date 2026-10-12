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
  Object.defineProperty(el,'innerHTML',{get(){return html;},set(value){html=String(value);this.scrollTop=0;
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
    'globalThis.workbench={state,bind,refreshLatestDates,refreshData,renderAnalysis,resizeCharts,renderChartPopup,analyzeSymbol};'));
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
const source={strategy_version:'czsc_trend_consensus',request:{start:'2026-10-01',research:true,model_family:'high_return',
  model_run_id:'saved-model',usage_mode:'historical',model_policy:'auto',calendar_run_id:'verified-calendar',
  account_start:'2023-01-01',account_initial_cash:100000,consensus_version:'czsc_trend_consensus',
  consensus_config:{version:'czsc_trend_consensus',quality_weight:.1}},
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
