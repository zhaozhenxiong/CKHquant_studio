"""Verify the actual shipped shell and point-in-time marker transformation."""
from pathlib import Path
import shutil
import subprocess

import pytest

from my_strategy.web_dashboard.scripts.validate_czsc_workbench import validate
from my_strategy.web_dashboard.api import AnalysisRequest


def test_new_analysis_default_and_explicit_history():
    assert str(AnalysisRequest(symbol="000001.SZ").start) == "2026-01-01"
    assert str(AnalysisRequest(symbol="000001.SZ", start="2024-01-01", end="2024-09-30").start) == "2024-01-01"
    assert AnalysisRequest(symbol="000001.SZ", start=None, as_of="2025-09-30").start is None


def test_shipped_shell_defaults_and_markers():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for the shipped JavaScript validation")
    assert validate(node)["status"] == "passed"
    script = Path(__file__).resolve().parents[1] / "web_dashboard/scripts/test_chart_markers.js"
    result = subprocess.run([node, str(script)], check=True, capture_output=True, text=True)
    assert '"status":"passed"' in result.stdout


def test_shipped_overlay_rendering_and_direction_colors():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped chart rendering checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const elements = new Map(), liveLines = new Set();
function element(id) {
  if (id === 'crosshair-value') return null;
  if (!elements.has(id)) elements.set(id, {checked:true,hidden:false,clientWidth:800,clientHeight:450,
    _text:'',_html:'',children:[],attributes:{},
    set textContent(value) { this._text=value; this._html=''; }, get textContent() { return this._text; },
    set innerHTML(value) { this._html=value; }, get innerHTML() { return this._html; },
    addEventListener() {},querySelector:() => element(id+'-child'), replaceChildren() { this.children=[]; this._html=''; this._text=''; },
    setAttribute(key,value) { this.attributes[key]=value; }, append(child) { this.children.push(child); }});
  return elements.get(id);
}
global.document = {getElementById:element,documentElement:{},createElementNS:() => element('svg-'+elements.size)};
global.getComputedStyle = () => ({getPropertyValue:(key) => ({'--up':'red','--down':'green','--muted':'gray'}[key] || '#123456')});
global.ResizeObserver = class { observe() {} };
global.requestAnimationFrame = () => 1;
global.KHQuantMarkers = require(process.argv[2]);
require(require('path').join(require('path').dirname(process.argv[1]),'wyckoff_markers.js'));
let candles, crosshair;
function series() { return {data:[],markers:[],setData(rows) { this.data=rows; },setMarkers(rows) { this.markers=rows; },
  applyOptions() {},priceScale:() => ({applyOptions() {}}),priceToCoordinate:(value) => 150-value}; }
const chart = {addCandlestickSeries:() => (candles=series()),addHistogramSeries:series,
  addLineSeries(options) { const line=series(); line.options=options; liveLines.add(line); return line; },
  removeSeries:(line) => liveLines.delete(line),subscribeCrosshairMove:(fn) => { crosshair=fn; },subscribeClick() {},
  resize() {},timeScale:() => ({subscribeVisibleLogicalRangeChange() {},fitContent() {},getVisibleLogicalRange:() => ({from:0,to:1}),height:() => 20,timeToCoordinate:(time) => time.endsWith('02') ? 30 : 80}),
  priceScale:() => ({width:() => 50})};
global.LightweightCharts = {createChart:() => chart,LineStyle:{Solid:0,Dashed:2}};
global.window = {LightweightCharts};
const source = fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.overlayTest = {state,renderAnalysis,renderIndicatorReadout,emptyChart};');
vm.runInThisContext(source);
const {state,renderAnalysis,renderIndicatorReadout,emptyChart} = overlayTest;
const first = {time:'2026-01-02',is_closed:true,ma5:null,ma10:null,ma20:null,ma60:null,boll_mid:null,boll_upper:null,boll_lower:null,
  chip70_low:null,chip70_high:null,chip50:null,chip_as_of:null,ma_alignment:'insufficient',ma20_direction:'insufficient',ma60_direction:'insufficient'};
const last = {...first,time:'2026-01-05',is_closed:false,ma5:10.25,ma10:10,ma20:9,ma60:8,
  boll_mid:9,boll_upper:11,boll_lower:7,chip70_low:8,chip70_high:12,chip50:10,chip_as_of:'2026-01-05',
  ma_alignment:'bullish',ma20_direction:'rising',ma60_direction:'falling'};
const bars = [first,last].map(({time,is_closed}) => ({time,is_closed,open:9,high:12,low:8,close:10,volume:100}));
const daily = {bars,indicators:{rows:[first,last]},pens:[],
  fractals:[{time:first.time,kind:'top',is_confirmed:true},{time:last.time,kind:'bottom',is_confirmed:false}],
  divergences:[{time:first.time,price:12,kind:'top',label:'类趋势顶背驰',confirmed_at:'2026-01-02T15:00:00+08:00',available_at:'2026-01-05T15:00:00+08:00',pen_anchors:[[first.time,last.time,8,12]]},
               {time:last.time,price:8,kind:'bottom',label:'aAb式底背驰',confirmed_at:'2026-01-05T15:00:00+08:00',available_at:'2026-01-05T15:00:00+08:00'}],
  zones:['up','down','range'].map((direction) => ({direction,start_time:first.time,end_time:last.time,low:8,high:10,is_confirmed:true}))};
state.query = {symbol:'000001.SZ',as_of:last.time};
state.analysis = {symbol:'000001.SZ',frequencies:{'日线':daily,'月线':{bars:[bars[0]],indicators:{rows:[first]}}},events:[]};
renderAnalysis(false);
assert.equal(liveLines.size,8);
for (const line of liveLines) { assert.equal(line.data[0].time,first.time); assert(!('value' in line.data[0])); }
assert(element('indicator-ma').innerHTML.includes('MA5 10.25'));
assert(element('indicator-ma').innerHTML.includes('多头排列 · MA20上行 · MA60下行'));
assert(element('indicator-context').textContent.includes('形成中，仅供展示'));
assert(element('indicator-chips').textContent.includes('中位价 10 · 日线截至 2026-01-05'));
assert.deepEqual(candles.markers.filter((marker) => !marker.text.includes('背驰')).map(({color,shape}) => [color,shape]),[['green','arrowDown'],['red','circle']]);
assert.deepEqual(candles.markers.filter((marker) => marker.text.includes('背驰')).map(({color,position,shape,text}) => [color,position,shape,text]),
  [['green','aboveBar','arrowDown','顶背驰参考'],['red','belowBar','arrowUp','底背驰参考']]);
assert(element('divergence-list').innerHTML.includes('类趋势顶背驰'));
assert(element('divergence-list').innerHTML.includes('末笔确认 2026-01-02T15:00:00+08:00'));
assert(element('divergence-list').innerHTML.includes('首次可见 2026-01-05T15:00:00+08:00'));
assert(element('divergence-list').innerHTML.includes('2026-01-02 → 2026-01-05'));
assert.deepEqual(element('zone-overlay').children[0].children.map((rect) => rect.attributes.stroke),['red','green','gray']);
daily.pens=[{start_time:first.time,end_time:last.time,start_price:12,end_price:8,is_confirmed:false,confirmed_at:null}];
renderAnalysis(false);
let pens=[...liveLines].filter(line=>line.options.lineWidth===2);
assert.equal(pens.length,1);assert.equal(pens[0].options.lineStyle,LightweightCharts.LineStyle.Dashed);
assert.deepEqual(pens[0].data,[{time:first.time,value:12},{time:last.time,value:8}]);
assert(element('structure-list').innerHTML.includes('形成中'));
daily.pens[0].is_confirmed=true;renderAnalysis(false);
pens=[...liveLines].filter(line=>line.options.lineWidth===2);
assert.equal(pens.length,1);assert.equal(pens[0].options.lineStyle,LightweightCharts.LineStyle.Solid);
element('layer-pens').checked=false;renderAnalysis(false);
assert(![...liveLines].some(line=>line.options.lineWidth===2));
element('layer-pens').checked=true;daily.pens=[];renderAnalysis(false);
element('layer-divergences').checked=false; renderAnalysis(false);
assert(!candles.markers.some((marker) => marker.text.includes('背驰')));
assert(element('divergence-details').hidden); assert.equal(element('divergence-list').innerHTML,'');
element('layer-divergences').checked=true;
crosshair({time:first.time,seriesData:new Map([[candles,bars[0]]])});
assert(element('indicator-ma').innerHTML.includes('MA5 —'));
assert(!element('indicator-ma').innerHTML.includes('10.25'));
assert(element('indicator-context').textContent.includes('所指K线 2026-01-02'));
crosshair({seriesData:new Map()});
assert(element('indicator-context').textContent.includes('最新K线 2026-01-05'));
element('layer-ma').checked=false; renderAnalysis(false);
assert.equal(liveLines.size,5); assert(element('indicator-ma').hidden);
assert([...liveLines].some((line) => line.data.at(-1).value === last.boll_mid));
for (const name of ['boll','chips']) element('layer-'+name).checked=false;
renderAnalysis(false); assert.equal(liveLines.size,0); assert(element('indicator-readout').hidden);
for (const name of ['ma','boll','chips']) element('layer-'+name).checked=true;
state.frequency='月线'; renderAnalysis(false);
assert.equal(liveLines.size,0); assert(element('indicator-ma').innerHTML.includes('MA60 —'));
assert(!element('indicator-chips').textContent.includes('2026-01-05'));
state.frequency='日线'; renderAnalysis(false); assert.equal(liveLines.size,8);
emptyChart('空结果','没有行情'); assert.equal(liveLines.size,0); assert(element('indicator-readout').hidden);
assert(element('divergence-details').hidden); assert.equal(element('divergence-list').innerHTML,'');
for (const name of ['context','ma','boll','chips']) assert.equal(element('indicator-'+name).textContent,'');
state.analysis=null; renderAnalysis(false); assert.equal(liveLines.size,0);
"""
    markers = path.with_name("chart_markers.js")
    result = subprocess.run([node, "-e", script, str(path), str(markers)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_current_checkpoint_time_check_and_failed_backtest_empty_state():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped backtest UI checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('node:assert/strict');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id,{value:'',hidden:false,disabled:false,innerHTML:'',textContent:'',querySelector:(selector) => element(id+selector)});
  return elements.get(id);
}
global.document = {getElementById:element};
const source = fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.backtestTest={state,updateResearchControls,researchSpec,renderBacktest};');
vm.runInThisContext(source);
const {state,updateResearchControls,researchSpec,renderBacktest}=backtestTest;
state.research={calendar_run_id:'verified',models:[{run_id:'train',checkpoints:[
  {name:'production',available_at:'2026-09-30',train_label_end:'2026-06-30'},
  {name:'2025Q4',available_at:'2025-09-30',train_label_end:'2025-06-30'}]}]};
element('analysis-mode').value='research'; element('analysis-model').value='train';
element('analysis-model-policy').value='pinned'; element('analysis-usage-mode').value='historical';
element('analysis-end').value='2026-01-01'; element('analysis-checkpoint').value='production';
updateResearchControls('analysis');
assert(element('analysis-checkpoint').innerHTML.includes('2025Q4 · 最早可用 2025-09-30'));
assert(element('analysis-checkpoint-help').textContent.includes('固定入场检查点 production'));
assert.throws(() => researchSpec('analysis'),/2026-09-30 才可用/);
element('analysis-checkpoint').value='2025Q4';
assert.deepEqual(researchSpec('analysis'),{research:true,model_family:'current',consensus_version:'trend_consensus',usage_mode:'historical',model_policy:'pinned',entry_policy:'risk',model_run_id:'train',model_fold:'2025Q4',calendar_run_id:'verified',device:'auto'});
element('scan-mode').value='research'; element('scan-model').value='train';
assert.equal(researchSpec('scan').model_fold,undefined);
element('backtest-mode').value='research'; element('backtest-model').value='';
element('backtest-model-policy').value='pinned'; element('backtest-usage-mode').value='forward';
updateResearchControls('backtest');
assert(element('backtest-checkpoint').disabled);
assert.equal(element('backtest-model-policy').value,'auto');
assert(element('backtest-model-policy').disabled);
assert.equal(element('backtest-usage-mode').value,'historical');
assert.deepEqual(researchSpec('backtest'),{research:true,model_family:'current',consensus_version:'trend_consensus',usage_mode:'historical',model_policy:'auto',entry_policy:'risk',calendar_run_id:'verified'});
state.tracked.backtest='job'; state.tasks=[{job_id:'job',status:'failed',error:'模型 <too late>'}];
renderBacktest();
assert(element('backtest-metrics').innerHTML.includes('本次回测失败：模型 &lt;too late&gt;'));
assert(!element('backtest-metrics').innerHTML.includes('等待'));
assert.equal(element('equity-emptystrong').textContent,'回测失败');
assert(element('trade-rows').innerHTML.includes('本次未生成可展示成交账本'));
state.tasks[0].status='cancelled'; renderBacktest();
assert(element('backtest-metrics').innerHTML.includes('已取消'));
state.tasks[0].status='running'; renderBacktest();
assert(element('backtest-metrics').innerHTML.includes('回测正在运行'));
state.tasks=[]; renderBacktest();
assert(element('backtest-metrics').innerHTML.includes('尚未运行回测'));
"""
    result = subprocess.run([node, "-e", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_current_model_unavailability_and_actual_holding_contract_are_explicit():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for shipped model qualification checks")
    path = Path(__file__).resolve().parents[1] / "web_dashboard/static/js/workbench.js"
    script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('node:assert/strict');
global.KHQuantEvidence=require(require('path').join(require('path').dirname(process.argv[1]),'chart_evidence.js'));
const source=fs.readFileSync(process.argv[1],'utf8').replace('init().catch(showError);',
  'globalThis.modelTest={modelResolutionText,modelInputReasonText,ruleQualificationText,nextSessionGuidance};');
vm.runInThisContext(source);
const {modelResolutionText,modelInputReasonText}=modelTest;
assert(modelResolutionText({model_family:'retired_family',probability:.9}).includes('历史模型协议已退役'));
assert(!modelResolutionText({model_family:'retired_family',probability:.9}).includes('90%'));
const resolution={model_family:'current',strategy_version:'trend_consensus',reason_codes:['no_prior_high_return_head']};
const row={model_family:'current',strategy_version:'trend_consensus',agents:{high_return:{evaluation_status:'model_unavailable',probability:.9}},
  account_context:{model_identity:{model_run_id:'saved-current'}},holding:{shares:0}};
let text=modelResolutionText(resolution,row);
assert(text.includes('所选研究运行 saved-current'));
assert(text.includes('缺兼容模型 · 未评估'));
assert(text.includes('该研究账户空仓，退出不适用'));
assert(!text.includes('90%'));
assert(text.includes('尚未取得生产资格'));
text=modelResolutionText(resolution,{...row,holding:{shares:100},account_action:{exit_evaluation:{probability:.8,holding_contract_compatible:false}}});
assert(text.includes('此日未生成兼容退出评分'));
assert(!text.includes('80%'));
text=modelResolutionText(resolution,{...row,holding:{shares:100},account_action:{exit_evaluation:{probability:.8,holding_contract_compatible:true}}});
assert(text.includes('持仓退出概率 80%'));
assert.equal(modelInputReasonText({model_input_reason_codes:['quality_warmup','wyckoff_quality_warmup']},
  {model_input_reason_codes:['quality_warmup']}),'可靠历史不足');
"""
    result = subprocess.run([node, "-e", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
