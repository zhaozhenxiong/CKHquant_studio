"""Exercise shipped display code, including historical interaction state."""
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


def test_evidence_keeps_probability_rule_target_and_original_dates_separate():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
assert.equal(ui.probability(null),'不可用');assert.equal(ui.probability(false),'不可用');
assert.equal(ui.probability(true),'不可用');assert.equal(ui.probability(''),'不可用');
assert.equal(ui.probability(0),'0%');assert.equal(ui.probability(.72),'72%');
assert.equal(ui.probability(1.5),'不可用');assert.equal(ui.ruleScore(.4),'40 / 100');
const row={date:'2026-07-03',reference_close:42.4,reference_price_date:'2026-07-03',reference_price_basis:'unadjusted',
  ohlc:{open:41,high:44,low:40,close:42.4},next_market_session:'2026-07-06',
  agents:{ma:{kind:'model_probability',status:'shadow',probability:.72,exit_probability:null,target:'forward actual policy',threshold:.55,reason:'threshold_met'},
  structure:{kind:'rule_evidence',buy_count:2,sell_count:1,point_types:['三买'],status:'available'},
  wyckoff:{kind:'rule_evidence',buy_score:.4,sell_score:.8,status:'available'},
  high_return:{kind:'model_probability',probability:null,status:'no_candidate',reason:'no_candidate_signal',target:'full round trip'}}};
const evidence={rows:[row]};
assert.equal(ui.rowAt(evidence,'2026-07-02'),null);
const html=ui.agentsHtml(row);
assert(html.includes('入场模型分 72%'));assert(html.includes('需求 40 / 100'));
assert(html.includes('供给 80 / 100'));assert(!html.includes('供给 80%'));
assert(html.includes('持仓退出概率 不可用'));assert(!html.includes('28%'));
assert(html.includes('影子观察'));assert(html.includes('本日没有新的买入候选'));
assert(html.includes('forward actual policy'));
assert(ui.riskHtml({price_floor:40,price_ceiling:43,entry_allowed:false,status:'blocked',reason_codes:['calendar_gap']}).includes('未通过入场资格'));
assert(ui.riskHtml({entry_allowed:true,status:'active',reason_codes:['high_return_independent_window_missing']}).includes('已生成'));
assert(ui.riskHtml({entry_allowed:true,status:'active',reason_codes:['high_return_independent_window_missing']}).includes('独立前向验证尚未完成'));
const group={kind:'intention',action:'BUY',visible:true,source_count:3,event_count:3,time:'2026-07-03',closed_at:'2026-07-03',entries:[
  {event_date:'2026-07-02',displayed:true,row:{source:'ma',event_type:'x<script>'}},
  {event_date:'2026-07-03',displayed:true,row:{source:'structure',event_type:'三买'}},
  {event_date:'2026-07-06',displayed:false,excluded:true,row:{source:'wyckoff'}}]};
const popup=ui.popupHtml({evidence,groups:[group],barDate:'2026-07-03',frequency:'周线',pinned:true,profileAsOf:'2026-07-03'});
assert(popup.includes('信号可知日 2026-07-03'));assert(!popup.includes('信号可知日 2026-07-06'));
assert(popup.includes('>B</span>'));assert(!popup.includes('B · 3 路'));assert(popup.includes('data-evidence-date="2026-07-02"'));
assert(popup.includes('周期闭合 2026-07-03'));assert(popup.includes('42.4 元'));
assert(popup.includes('筹码按所指周期截至 2026-07-03'));
const delayed={...group,entries:[{event_date:'2026-07-02',displayed:true,row:{evidence_date:'2026-07-03',original_event_date:'2026-07-02',available_at:'2026-07-03T15:00:00',source:'structure'}}]};
const causal=ui.popupHtml({evidence,groups:[delayed],barDate:'2026-07-03',frequency:'日线',pinned:true});
assert(causal.includes('信号可知日 2026-07-03'));assert(causal.includes('42.4 元'));assert(causal.includes('发生 2026-07-02'));
const escaped=ui.popupHtml({evidence,groups:[group],barDate:'2026-07-03',frequency:'周线',pinned:true,selectedDate:'2026-07-02'});
assert(escaped.includes('x&lt;script&gt;'));assert(!escaped.includes('x<script>'));
const sell={...group,action:'SELL',event_count:1,source_count:1};
const both=ui.popupHtml({evidence,groups:[group,sell],barDate:'2026-07-03',frequency:'日线',pinned:true});
assert(both.includes('>B</span>'));assert(both.includes('>S</span>'));
const rows=Array.from({length:45},(_,i)=>({...row,date:`2026-07-${String(i+1).padStart(2,'0')}`}));
assert.equal(ui.historyPage(rows,0).rows.length,20);assert.equal(ui.historyPage(rows,2).rows.length,5);
assert.equal(ui.historyPage(rows,50).page,2);assert.equal(ui.historyPage(rows,-1).page,0);
const hidden=ui.historyHtml([{...row,events:[{action:'BUY',candidate_eligible:false},{action:'SELL',evidence_visible:false}]}]);
assert(!hidden.includes('<td>B</td>'));assert(!hidden.includes('<td>S</td>'));
let selection=ui.selectionTransition(null,{type:'hover',time:'2026-07-02'});
selection=ui.selectionTransition(selection,{type:'pin',time:'2026-07-02'});
assert.equal(ui.selectionTransition(selection,{type:'hover',time:'2026-07-03'}).pinned,'2026-07-02');
assert.equal(ui.selectionTransition(selection,{type:'leave'}).hover,'2026-07-02');
assert.deepEqual(ui.selectionTransition(selection,{type:'loading'}),{hover:null,pinned:null,selectedDate:null,popupOpen:false});
''')


def test_workbench_crosshair_pin_profile_and_history_loading_use_current_evidence():
    run_js(r'''
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('node:assert/strict');
const root=process.argv[1];const ui=require(path.join(root,'chart_evidence.js'));
global.KHQuantMarkers=require(path.join(root,'chart_markers.js'));require(path.join(root,'wyckoff_markers.js'));
const elements=new Map();function newElement(tag='div') {return {tag,dataset:{},style:{},checked:true,hidden:false,disabled:false,
  clientWidth:145,clientHeight:450,offsetWidth:460,offsetHeight:500,value:'',textContent:'',innerHTML:'',children:[],attributes:{},listeners:{},
  getBoundingClientRect:()=>({left:20,top:180}),addEventListener(key,callback){this.listeners[key]=callback;},
  querySelector:()=>newElement(),replaceChildren(){this.children=[];},append(child){this.children.push(child);},setAttribute(key,value){this.attributes[key]=value;}};}
function element(id){if(!elements.has(id))elements.set(id,newElement());return elements.get(id);}
global.document={getElementById:element,documentElement:{},createElementNS:(_,tag)=>newElement(tag),querySelectorAll:()=>[]};
global.getComputedStyle=()=>({getPropertyValue:()=> '#123456'});global.ResizeObserver=class{observe(){}};
global.requestAnimationFrame=()=>1;global.window={innerWidth:907,innerHeight:924};global.location={hash:''};
let range={from:0,to:2},crosshair,click,firstSeries;const resizeWidths=[],fitWidths=[];
const scale={subscribeVisibleLogicalRangeChange(){},getVisibleLogicalRange:()=>range,setVisibleLogicalRange(value){range=value;},
  fitContent(){fitWidths.push(element('analysis-chart').clientWidth);},height:()=>30,timeToCoordinate:()=>50};
function series(){const result={setData(){},setMarkers(value){this.markers=value;},applyOptions(){},priceScale:()=>({applyOptions(){}}),priceToCoordinate:price=>350-(price-8)*60};return result;}
const chart={addCandlestickSeries(){firstSeries=series();return firstSeries;},addHistogramSeries:series,addLineSeries:series,removeSeries(){},resize(width){resizeWidths.push(width);},
  subscribeCrosshairMove(fn){crosshair=fn;},subscribeClick(fn){click=fn;},timeScale:()=>scale,priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};window.LightweightCharts=LightweightCharts;
const code=fs.readFileSync(path.join(root,'workbench.js'),'utf8').replace('init().catch(showError);',
  'globalThis.interactions={state,renderAnalysis,updateChartSelection,selectedChipProfile,renderChartPopup,renderReplayEvidence,updateReplay,analyze,locateTime,openRun,loading:()=>analysisLoading,setRunDisplay:fn=>{showResult=fn;},selection:()=>chartSelection,seedHistory:rows=>{historyRows=rows;}};');
vm.runInThisContext(code);
const {state,renderAnalysis,updateChartSelection,selectedChipProfile,renderChartPopup,updateReplay,analyze}=interactions;
const dates=['2026-07-01','2026-07-02','2026-07-03'];
const bars=dates.map(time=>({time,open:10,high:12,low:9,close:11,volume:100,is_closed:true}));
const profile=as_of=>({status:'available',as_of,window_bars:120,chip70_low:9.5,chip70_high:11.5,chip50:10.5,bins:[{price_low:9,price_high:10,mass_fraction:.2}]});
const old=profile(dates[0]),latest=profile(dates[2]);latest.bins=[{price_low:10,price_high:11,mass_fraction:.4}];
const rows=dates.map((date,i)=>({date,symbol:'600027.SH',reference_close:11+i,reference_price_date:date,
  agents:{ma:{kind:'model_probability',probability:.2+i*.1,status:'shadow'},structure:{kind:'rule_evidence',buy_count:1,sell_count:0},wyckoff:{kind:'rule_evidence',buy_score:.3,sell_score:.7},high_return:{kind:'model_probability',probability:null,reason:'no_candidate_signal'}}}));
const events=['ma','structure','wyckoff'].map((source,i)=>({time:dates[0],available_at:dates[0]+'T15:00:00+08:00',source,action:'BUY',candidate_eligible:true,event_type:'test'+i,symbol:'600027.SH'}));
events.push({time:dates[0],available_at:dates[0]+'T15:00:00+08:00',source:'ma',action:'SELL',event_type:'exit',symbol:'600027.SH'});
const consensusRows=dates.map((date,i)=>({date,action:i===0?'BUY':i===2?'SELL':'WAIT',emitted:i!==1,buy_support:i===0?2:0,sell_support:i===2?2:0,reason_codes:[i===1?'insufficient_consensus':'consensus_confirmed'],agents:{ma:{judgment:i===0?'BUY':'WAIT',reason_codes:['saved_model_score']},structure:{judgment:i===2?'SELL':'BUY',reason_codes:['confirmed_structure']},wyckoff:{judgment:'WAIT',reason_codes:['balanced_supply_demand']},high_return:{judgment:'UNAVAILABLE',reason_codes:['no_candidate_signal']},arbiter:{judgment:i===0?'BUY':i===2?'SELL':'WAIT',reason_codes:['independent_evidence_families']}}}));
const consensus={config:{version:'consensus_v1',min_support:2},rows:consensusRows,events:consensusRows.filter(row=>row.emitted).map(row=>({source:'consensus',date:row.date,evidence_date:row.date,action:row.action,symbol:'600027.SH',event_id:row.date}))};
state.analysis={symbol:'600027.SH',consensus_decisions:consensus,as_of:dates[2],data_end:dates[2],frequencies:{'日线':{bars,indicators:{rows:bars.map(bar=>({...bar,ma5:11})),chip_profile:latest,chip_profiles:{[dates[0]]:old,[dates[2]]:latest}}}},events,chart_evidence:{rows}};
state.query={symbol:'600027.SH',start:dates[0],end:dates[2]};state.dates=dates;state.replayIndex=2;
element('analysis-chart').clientWidth=640;element('view-analysis').hidden=false;
renderAnalysis(false);interactions.seedHistory(rows);updateReplay();
assert(element('history-evidence-table').innerHTML.includes('2026-07-01'));
assert(element('replay-evidence-card').innerHTML.includes('13 元'));
const hover=time=>crosshair({time,point:{x:420,y:430},seriesData:new Map([[firstSeries,{close:11}]])});
hover(dates[0]);assert.equal(selectedChipProfile().as_of,dates[0]);
assert(element('chart-evidence-popup').hidden);assert.equal(interactions.selection().popupOpen,false);
assert(element('chip-profile-caption').textContent.includes(dates[0]));
click({time:dates[0],point:{x:750,y:700}});
assert.equal(interactions.selection().pinned,dates[0]);assert.equal(element('chart-evidence-popup').dataset.pinned,'true');assert(!element('chart-evidence-popup').hidden);assert(element('chart-evidence-popup').innerHTML.includes('B · 最终买点'));assert(element('chart-evidence-popup').innerHTML.includes('仲裁意见：支持买入'));
assert(parseFloat(element('chart-evidence-popup').style.left)+460<=907-12);
assert(parseFloat(element('chart-evidence-popup').style.top)+500<=924-12);
hover(dates[2]);assert.equal(selectedChipProfile().as_of,dates[0]);
element('analysis-chart').listeners.pointerleave();assert.equal(selectedChipProfile().as_of,dates[0]);
updateChartSelection('clear');assert.equal(selectedChipProfile().as_of,dates[2]);assert(element('chart-evidence-popup').hidden);hover(dates[0]);assert(element('chart-evidence-popup').hidden);
hover(dates[1]);assert.equal(selectedChipProfile().status,'unavailable');
assert(!element('chip-profile-caption').textContent.includes(dates[2]));
assert(!element('indicator-chips').textContent.includes('截至 '+dates[2]));
// A higher-period selection can explicitly select its original daily snapshot.
state.analysis.frequencies['日线'].bars=[...bars,{...bars[2],time:'2026-07-06'}];state.query.as_of='2026-07-06';
state.frequency='周线';state.analysis.frequencies['周线']={bars:[bars[2]],indicators:{rows:[bars[2]],chip_profile:latest,chip_profiles:{[dates[2]]:latest}}};
renderAnalysis(false);updateChartSelection('pin',dates[2]);interactions.locateTime(dates[0]);
assert.equal(selectedChipProfile().as_of,dates[0]);
assert(element('chart-evidence-popup').innerHTML.includes('信号可知日 '+dates[0]));
assert(element('chip-profile-caption').textContent.includes('截至 '+dates[0]));
assert(!element('chip-profile-caption').textContent.includes('截至 '+dates[2]));
assert(element('indicator-chips').textContent.includes('日线截至 '+dates[0]));
assert.equal(Number(element('chip-profile-overlay').children.find(item=>item.tag==='rect').attributes.y),230);
// Monthly locating must likewise repaint the actual SVG and lower readout.
const monthDate='2026-07-31',monthProfile={...latest,as_of:monthDate};
state.analysis.frequencies['日线'].bars=[...bars,{...bars[2],time:'2026-08-03'}];state.query.as_of='2026-08-03';
state.frequency='月线';state.analysis.frequencies['月线']={bars:[{...bars[2],time:monthDate}],indicators:{rows:[{...bars[2],time:monthDate}],chip_profile:monthProfile,chip_profiles:{[monthDate]:monthProfile}}};
renderAnalysis(false);updateChartSelection('pin',monthDate);interactions.locateTime(dates[0]);
assert(element('chart-evidence-popup').innerHTML.includes('信号可知日 '+dates[0]));
assert(element('chip-profile-caption').textContent.includes('截至 '+dates[0]));
assert(element('indicator-chips').textContent.includes('日线截至 '+dates[0]));
assert.equal(Number(element('chip-profile-overlay').children.find(item=>item.tag==='rect').attributes.y),230);
state.frequency='日线';updateChartSelection('clear');
global.sessionStorage={getItem:()=>null};
let resolveRequest;global.fetch=()=>new Promise(resolve=>{resolveRequest=resolve;});
const pending=analyze({...state.query,as_of:dates[0]},true);
assert(element('replay-evidence-card').innerHTML.includes('加载中'));
assert(!element('replay-evidence-card').innerHTML.includes('13 元'));
assert(element('history-evidence-table').innerHTML.includes('加载中'));assert(element('chart-evidence-popup').hidden);
assert(element('replay-prev').disabled);
const response={...state.analysis,as_of:dates[0],data_end:dates[0],chart_evidence:{rows:[rows[0]]},
 frequencies:{'日线':{bars:[bars[0]],indicators:{rows:[bars[0]],chip_profile:old,chip_profiles:{[dates[0]]:old}}}}};
resolveRequest({ok:true,status:200,text:async()=>JSON.stringify(response)});
pending.then(async()=>{assert(element('replay-evidence-card').innerHTML.includes('11 元'));assert.deepEqual(state.query.consensus_config,consensus.config);
  assert(!element('history-evidence-table').innerHTML.includes('2026-07-03'));
  assert(element('history-evidence-table').innerHTML.includes('2026-07-01'));
  global.fetch=async()=>({ok:false,status:500,text:async()=>JSON.stringify({detail:'analysis failed'})});
  await analyze({...state.query,as_of:dates[1]},true);
  assert(element('replay-evidence-card').innerHTML.includes('不可用'));
  assert(!element('replay-evidence-card').innerHTML.includes('11 元'));
  assert(element('history-evidence-table').innerHTML.includes('不可用'));
  const requests=[];global.fetch=()=>new Promise(resolve=>requests.push(resolve));
  const earlier=analyze({...state.query,as_of:dates[0]},true);
  const newer=analyze({...state.query,as_of:dates[2]},true);
  requests[1]({ok:true,status:200,text:async()=>JSON.stringify({...response,as_of:dates[2],data_end:dates[2],chart_evidence:{rows}})});
  await newer;assert(element('replay-evidence-card').innerHTML.includes('13 元'));
  requests[0]({ok:true,status:200,text:async()=>JSON.stringify(response)});await earlier;
  assert(element('replay-evidence-card').innerHTML.includes('13 元'));assert.equal(state.query.as_of,dates[2]);
  // Saved B wins over an analysis A that was already running when B opened.
  const savedB={...response,symbol:'600028.SH',chart_evidence:{rows:[{...rows[0],symbol:'600028.SH',reference_close:88}]}};
  const queued=[];global.fetch=()=>new Promise(resolve=>queued.push(resolve));
  const oldAnalysis=analyze({symbol:'600027.SH',as_of:dates[0]},true);
  const saved=interactions.openRun('saved-B');
  queued[1]({ok:true,status:200,text:async()=>JSON.stringify({kind:'analysis',result:savedB})});await saved;
  assert.equal(state.analysis.symbol,'600028.SH');assert(element('replay-evidence-card').innerHTML.includes('88 元'));
  assert.equal(interactions.loading(),false);
  queued[0]({ok:true,status:200,text:async()=>JSON.stringify(response)});await oldAnalysis;
  assert.equal(state.analysis.symbol,'600028.SH');assert(element('replay-evidence-card').innerHTML.includes('88 元'));
  // Two saved-run requests also obey the latest request, independent of arrival.
  const runs=[];global.fetch=()=>new Promise(resolve=>runs.push(resolve));
  const runFirst=interactions.openRun('older-saved'),runLast=interactions.openRun('newer-saved');
  const savedC={...savedB,symbol:'600029.SH',chart_evidence:{rows:[{...rows[0],symbol:'600029.SH',reference_close:99}]}};
  runs[1]({ok:true,status:200,text:async()=>JSON.stringify({kind:'analysis',result:savedC})});await runLast;
  runs[0]({ok:true,status:200,text:async()=>JSON.stringify({kind:'analysis',result:savedB})});await runFirst;
  assert.equal(state.analysis.symbol,'600029.SH');assert(element('replay-evidence-card').innerHTML.includes('99 元'));
  // Opening another result type does not cancel a pending analysis.
  let displayedKind;interactions.setRunDisplay(kind=>{displayedKind=kind;});
  const mixed=[];global.fetch=()=>new Promise(resolve=>mixed.push(resolve));
  const activeAnalysis=analyze({symbol:'600027.SH',as_of:dates[0]},true);
  const savedScan=interactions.openRun('scan-run');
  mixed[1]({ok:true,status:200,text:async()=>JSON.stringify({kind:'scan',result:{rows:[]}})});await savedScan;
  assert.equal(displayedKind,'scan');assert.equal(interactions.loading(),true);
  mixed[0]({ok:true,status:200,text:async()=>JSON.stringify(response)});await activeAnalysis;
  assert.equal(state.analysis.symbol,'600027.SH');assert(element('replay-evidence-card').innerHTML.includes('11 元'));
  // A new analysis started after a saved request stays newer than that request.
  const last=[];global.fetch=()=>new Promise(resolve=>last.push(resolve));
  const staleSaved=interactions.openRun('stale-saved'),freshAnalysis=analyze({symbol:'600027.SH',as_of:dates[0]},true);
  last[1]({ok:true,status:200,text:async()=>JSON.stringify(response)});await freshAnalysis;
  last[0]({ok:true,status:200,text:async()=>JSON.stringify({kind:'analysis',result:savedC})});await staleSaved;
  assert.equal(state.analysis.symbol,'600027.SH');assert(element('replay-evidence-card').innerHTML.includes('11 元'));
  // Opening a saved analysis from the hidden Data view must reveal Analysis
  // synchronously before chip layout resize and fitContent, without waiting
  // for the browser's later hashchange callback.
  element('view-analysis').id='view-analysis';element('view-data').id='view-data';
  element('view-analysis').hidden=true;element('view-data').hidden=false;
  element('chip-profile').hidden=true;location.hash='data';
  Object.defineProperty(element('analysis-chart'),'clientWidth',{get:()=>element('view-analysis').hidden?0:640});
  document.querySelectorAll=selector=>selector==='.view'?[element('view-analysis'),element('view-data')]:[];
  resizeWidths.length=0;fitWidths.length=0;
  global.fetch=async()=>({ok:true,status:200,text:async()=>JSON.stringify({kind:'analysis',result:response})});
  await interactions.openRun('saved-from-hidden-data');
  assert.equal(element('view-analysis').hidden,false);assert.equal(element('view-data').hidden,true);
  assert.equal(location.hash,'analysis');assert.deepEqual(resizeWidths,[640]);assert.deepEqual(fitWidths,[640]);
}).catch(error=>{console.error(error);process.exitCode=1;});
''')
