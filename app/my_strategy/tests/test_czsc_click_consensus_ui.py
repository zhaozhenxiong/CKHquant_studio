"""Run the shipped JavaScript against click-only and final-decision contracts."""
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


def test_final_decision_details_preserve_scores_reasons_and_raw_candidates():
    run_js(r'''
const assert=require('node:assert/strict'),path=require('path');
const ui=require(path.join(process.argv[1],'chart_evidence.js'));
const row={date:'2026-07-01',reference_close:11,events:[
  {source:'ma',event_type:'recovery<script>',date:'2026-07-01',available_at:'2026-07-01T15:00:00',candidate_eligible:true},
  {source:'structure',event_type:'three_buy',date:'2026-07-01',candidate_eligible:false}],
  agents:{ma:{kind:'model_probability',probability:.72},structure:{kind:'rule_evidence',buy_count:1,sell_count:0},
  wyckoff:{kind:'rule_evidence',buy_score:.8,sell_score:.1},high_return:{kind:'model_probability',probability:null}}};
const decision={date:row.date,action:'BUY',emitted:true,buy_support:2,sell_support:0,state_before:'NEUTRAL',state_after:'BULLISH',
  buy_independent_count:2,sell_independent_count:0,buy_independent_families:['ma','wyckoff'],sell_independent_families:[],reason_codes:['multi_agent_direction_confirmed'],agents:{ma:{judgment:'BUY',threshold:.55,model_threshold:.6,historical_reconstruction:true,reason_codes:['eligible_causal_source_support']},
  structure:{judgment:'UNAVAILABLE',reason_codes:['input_ineligible']},wyckoff:{judgment:'BUY',reason_codes:['demand_advantage']},
  high_return:{judgment:'ABSTAIN',status:'unavailable',threshold:.1,required:true,counts_as_vote:false,reason_codes:['high_return_score_unavailable']},arbiter:{judgment:'BUY',reason_codes:['multi_agent_direction_confirmed']}}};
const consensus={rows:[decision]},evidence={rows:[row]};
const group={kind:'intention',source_count:1,action:'BUY',visible:true,entries:[{row:{source:'consensus',evidence_date:row.date}}]};
const html=ui.popupHtml({evidence,consensus,groups:[group],barDate:row.date,frequency:'日线'});
assert(html.indexOf('B · 最终买点')<html.indexOf('收盘参考价'));
assert(html.includes('入场模型分 72%'));assert(html.includes('需求 80 / 100'));assert(!html.includes('需求 80%'));
assert(html.includes('仲裁意见：支持买入'));assert(html.includes('仲裁意见：不可用'));
assert(html.includes('持仓退出概率 不可用'));assert(!html.includes('28%'));
assert(html.includes('原始候选证据 2 条（含未通过资格的证据）'));
assert(html.includes('recovery&lt;script&gt;'));assert(!html.includes('recovery<script>'));
assert(html.includes('多 Agent 已确认最终方向'));assert(html.includes('独立证据家族：买 2（均线／辅助结构、威科夫量价）'));assert(html.includes('均线 ML'));assert(html.includes('当前仲裁门槛 55%'));assert(html.includes('原模型冻结门槛 60%'));assert(html.includes('不计入支持票数'));assert(html.includes('历史重建意见'));assert(html.includes('方向 中性 → 看多'));
const wait={...decision,action:'WAIT',proposed_action:'BUY',emitted:false,reason_codes:['insufficient_independent_support']};
const merged=ui.mergeEvidenceRows([row],[wait]);
assert.equal(ui.finalSide(merged[0].consensus),'');
assert(ui.historyHtml(merged).includes('独立证据家族支持未达到门槛'));
assert(!ui.historyHtml(merged).includes('<td>B</td>')); // Raw candidates never become final history points.
assert(!ui.historyHtml([row]).includes('<td>B</td>'));
assert(ui.consensusHtml(wait).includes('本日未生成最终买卖点'));
assert(ui.consensusHtml(null).includes('最终判断不可用'));
assert(ui.historyHtml(ui.mergeEvidenceRows([row],[decision])).includes('<td>B</td>'));
assert.equal(ui.probability(null),'不可用');assert.equal(ui.probability(false),'不可用');assert.equal(ui.probability(0),'0%');
let state=ui.selectionTransition(null,{type:'hover',time:row.date});assert.equal(state.popupOpen,false);
state=ui.selectionTransition(state,{type:'pin',time:row.date,popupOpen:true});assert.equal(state.popupOpen,true);
state=ui.selectionTransition(state,{type:'clear'});
state=ui.selectionTransition(state,{type:'hover',time:row.date});assert.equal(state.popupOpen,false);
''')


def test_click_only_window_switch_close_blank_frequency_dates_and_no_raw_fallback():
    run_js(r'''
const fs=require('fs'),vm=require('vm'),path=require('path'),assert=require('node:assert/strict');
const root=process.argv[1];require(path.join(root,'chart_evidence.js'));
global.KHQuantMarkers=require(path.join(root,'chart_markers.js'));require(path.join(root,'wyckoff_markers.js'));
const elements=new Map(),windowListeners={},documentListeners={};
function newElement(tag='div') {return {tag,dataset:{},style:{},checked:true,hidden:false,disabled:false,
  clientWidth:145,clientHeight:450,offsetWidth:460,offsetHeight:500,value:'',textContent:'',innerHTML:'',children:[],attributes:{},listeners:{},
  getBoundingClientRect:()=>({left:20,top:180}),addEventListener(key,callback){(this.listeners[key] ||= []).push(callback);},
  querySelector:()=>newElement(),replaceChildren(){this.children=[];this.innerHTML='';},append(child){this.children.push(child);},
  hasAttribute(key){return key in this.attributes;},setAttribute(key,value){this.attributes[key]=value;}};}
function element(id){if(!elements.has(id))elements.set(id,newElement());return elements.get(id);}
const weeklyButton=newElement('button');weeklyButton.dataset.frequency='周线';
global.document={getElementById:element,documentElement:{},createElementNS:(_,tag)=>newElement(tag),
  querySelectorAll:selector=>selector==='[data-frequency]'?[weeklyButton]:[],addEventListener(key,fn){documentListeners[key]=fn;}};
global.getComputedStyle=()=>({getPropertyValue:()=> '#123456'});global.ResizeObserver=class{observe(){}};
global.requestAnimationFrame=()=>1;global.window={innerWidth:907,innerHeight:924,addEventListener(key,fn){windowListeners[key]=fn;}};
global.location={hash:''};global.setTimeout=()=>1;global.clearTimeout=()=>{};
let range={from:0,to:2},crosshair,click,firstSeries;
const scale={subscribeVisibleLogicalRangeChange(){},getVisibleLogicalRange:()=>range,setVisibleLogicalRange(value){range=value;},
  fitContent(){},height:()=>30,timeToCoordinate:()=>50};
function series(){return {setData(){},setMarkers(value){this.markers=value;},applyOptions(){},priceScale:()=>({applyOptions(){}}),priceToCoordinate:price=>350-(price-8)*60};}
const chart={addCandlestickSeries(){firstSeries=series();return firstSeries;},addHistogramSeries:series,addLineSeries:series,removeSeries(){},resize(){},
  subscribeCrosshairMove(fn){crosshair=fn;},subscribeClick(fn){click=fn;},timeScale:()=>scale,priceScale:()=>({width:()=>50})};
global.LightweightCharts={createChart:()=>chart,LineStyle:{Solid:0,Dashed:2}};window.LightweightCharts=LightweightCharts;
const code=fs.readFileSync(path.join(root,'workbench.js'),'utf8').replace('init().catch(showError);',
  'globalThis.interactions={state,renderAnalysis,renderReplayEvidence,bind,selectedChipProfile,selection:()=>chartSelection};');
vm.runInThisContext(code);
const {state,renderAnalysis,bind,selectedChipProfile}=interactions;
const dates=['2026-07-01','2026-07-02','2026-07-03','2026-07-06'];
const bars=dates.map(time=>({time,open:10,high:12,low:9,close:11,volume:100,is_closed:true}));
const profile=as_of=>({status:'available',as_of,window_bars:120,chip70_low:9.5,chip70_high:11.5,chip50:10.5,bins:[{price_low:9,price_high:10,mass_fraction:.2}]});
const profiles=Object.fromEntries(dates.map(day=>[day,profile(day)]));
const rows=dates.map((date,i)=>({date,reference_close:11+i,agents:{},events:[{source:'structure',date,action:'BUY',event_type:'raw_candidate'}]}));
const consensusRows=dates.map((date,i)=>({date,action:i===0?'BUY':i===2?'SELL':'WAIT',emitted:i===0||i===2,reason_codes:[i===0||i===2?'consensus_confirmed':'insufficient_support'],agents:{arbiter:{judgment:i===0?'BUY':i===2?'SELL':'WAIT'}}}));
const consensus={config:{min_support:2},rows:consensusRows,events:consensusRows.filter(row=>row.emitted).map(row=>({date:row.date,time:row.date,evidence_date:row.date,source:'consensus',action:row.action,symbol:'600027.SH',event_id:row.date}))};
const raw=dates.map(date=>({source:'structure',date,time:date,action:'BUY',candidate_eligible:true,symbol:'600027.SH'}));
state.analysis={symbol:'600027.SH',as_of:dates[3],data_end:dates[3],consensus_decisions:consensus,events:raw,chart_evidence:{rows},
  frequencies:{'日线':{bars,indicators:{rows:bars,chip_profile:profiles[dates[3]],chip_profiles:profiles}},
  '周线':{bars:[bars[2]],indicators:{rows:[bars[2]],chip_profile:profiles[dates[2]],chip_profiles:{[dates[2]]:profiles[dates[2]]}}}}};
state.query={symbol:'600027.SH',start:dates[0],end:dates[3]};state.dates=dates;state.replayIndex=3;
element('analysis-chart').clientWidth=640;element('view-analysis').hidden=false;
for(const name of ['fractals','divergences','wyckoff','trades']) element('layer-'+name).checked=false;
bind();renderAnalysis(false);
assert.deepEqual(firstSeries.markers.map(({time,text})=>({time,text})),[{time:dates[0],text:'B'},{time:dates[2],text:'S'}]);
const hover=time=>crosshair({time,point:{x:420,y:430},seriesData:new Map([[firstSeries,{close:11}]])});
const clicked=time=>click({time,point:{x:750,y:700}});
hover(dates[0]);assert(element('chart-evidence-popup').hidden);assert.equal(selectedChipProfile().as_of,dates[0]);
clicked(dates[0]);assert(!element('chart-evidence-popup').hidden);assert(element('chart-evidence-popup').innerHTML.includes('B · 最终买点'));
clicked(dates[2]);assert(element('chart-evidence-popup').innerHTML.includes('S · 最终卖点'));assert(element('chart-evidence-popup').innerHTML.includes('13 元'));
const closeButton=newElement('button');closeButton.setAttribute('data-chart-evidence-close','');
documentListeners.click({target:{closest:()=>closeButton}});assert(element('chart-evidence-popup').hidden);
hover(dates[0]);assert(element('chart-evidence-popup').hidden);
clicked(dates[0]);clicked(dates[1]);assert(element('chart-evidence-popup').hidden);assert.equal(interactions.selection().pinned,null);
clicked(dates[0]);click({point:{x:1,y:1}});assert(element('chart-evidence-popup').hidden);
clicked(dates[0]);windowListeners.keydown({key:'Escape'});hover(dates[0]);assert(element('chart-evidence-popup').hidden);
clicked(dates[0]);element('chart-evidence-reset').listeners.click[0]();hover(dates[2]);assert(element('chart-evidence-popup').hidden);
clicked(dates[0]);element('analysis-start').listeners.change[0]();hover(dates[0]);assert(element('chart-evidence-popup').hidden);
clicked(dates[0]);element('analysis-symbol').listeners.input[0]();hover(dates[0]);assert(element('chart-evidence-popup').hidden);
clicked(dates[0]);element('chart-range-start').listeners.change[0]();hover(dates[0]);assert(element('chart-evidence-popup').hidden);
clicked(dates[0]);weeklyButton.listeners.click[0]();hover(dates[2]);assert(element('chart-evidence-popup').hidden);
assert.deepEqual(firstSeries.markers.map(marker=>marker.text),['B','S']);
state.backtest={trades:[{symbol:'600027.SH',date:dates[1],action:'BUY',price:11}]};element('layer-trades').checked=true;renderAnalysis(false);
clicked(dates[2]);assert(!element('chart-evidence-popup').hidden);assert(!element('chart-evidence-popup').innerHTML.includes('data-evidence-date="2026-07-02"'));
const buyMarker=firstSeries.markers.find(marker=>marker.text==='B');click({time:dates[2],hoveredObjectId:buyMarker.id,point:{x:750,y:700}});
assert(element('chart-evidence-popup').innerHTML.includes('B · 最终买点'));assert.equal(selectedChipProfile().as_of,dates[0]);
assert(element('chart-evidence-popup').innerHTML.includes('data-evidence-date="2026-07-01"'));
assert(element('chart-evidence-popup').innerHTML.includes('data-evidence-date="2026-07-03"'));
const dateButton=newElement('button');dateButton.dataset.evidenceDate=dates[0];
documentListeners.click({target:{closest:()=>dateButton}});assert.equal(selectedChipProfile().as_of,dates[0]);
assert(element('chart-evidence-popup').innerHTML.includes('B · 最终买点'));
state.frequency='日线';element('layer-trades').checked=false;delete state.analysis.consensus_decisions;renderAnalysis(false);
assert.equal(firstSeries.markers.length,0);clicked(dates[0]);hover(dates[0]);assert(element('chart-evidence-popup').hidden);
assert(element('marker-summary').textContent.includes('没有多 Agent 仲裁'));
''')


def test_default_shell_point_labels_are_final_only():
    from html.parser import HTMLParser

    class Layers(HTMLParser):
        def __init__(self):
            super().__init__()
            self.layers = {}

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if tag == "input" and attributes.get("id", "").startswith("layer-"):
                self.layers[attributes["id"]] = "checked" in attributes

    parser = Layers()
    parser.feed((STATIC.parent / "index.html").read_text(encoding="utf-8"))
    assert parser.layers["layer-signals"]
    for name in ("fractals", "divergences", "wyckoff", "trades"):
        assert not parser.layers[f"layer-{name}"]
