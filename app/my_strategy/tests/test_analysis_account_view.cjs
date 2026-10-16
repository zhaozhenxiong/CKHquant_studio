const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '../web_dashboard/static/js/analysis_account_view.js'), 'utf8');
const id = 'a'.repeat(64);
const expected = {result_id: id, symbol: '002156.SZ', strategy: 'structure_kronos', as_of: '2026-10-16'};
function projection() {
  const trades = [['2026-10-14','BUY',10.005],['2026-10-15','REDUCE',10.995],['2026-10-16','SELL',10.995]].map(([date,action,price],i)=>({
    date,action,price,symbol:expected.symbol,shares:100,fee:5.01,order_id:String(i),ledger_version:3}));
  return {schema:'saved_account_chart_v1',semantics:'broker_ledger_fills',...expected,
    prediction_started:false,account_mutated:false,current_method_qualification_changed:false,price_basis:'unadjusted',
    label:'结构与量价 + Kronos',display_start:'2026-10-13',display_end:'2026-10-16',saved:false,
    bars:['2026-10-13','2026-10-14','2026-10-15','2026-10-16'].map(time=>({time,open:10,high:12,low:9,close:11})),
    trades,counts:{BUY:1,SELL:2,ADD:0,REDUCE:1}};
}
function boot() {
  const ids=['analysis-chart-content','analysis-forecast-view','analysis-account-view','analysis-account-result-label',
    'analysis-account-result','analysis-account-strategy-label','analysis-account-strategy','analysis-account-state',
    'analysis-account-chart','analysis-account-readout','analysis-account-trades','analysis-prediction-title',
    'analysis-prediction-summary','view-analysis','analysis-prediction'];
  const nodes = Object.fromEntries(ids.map(id=>[id,{value:'',hidden:false,textContent:'',listeners:{},
    addEventListener(kind,handler){this.listeners[kind]=handler;},
    set innerHTML(value){this.html=value;this.value=/<option value="([^"]+)"/.exec(value)?.[1]||'';},
    get innerHTML(){return this.html||'';}}]));
  nodes['analysis-chart-content'].value='forecast';
  const rendered=[];
  const context={document:{getElementById:id=>nodes[id],documentElement:{}},AbortController,URLSearchParams,
    getComputedStyle:()=>({getPropertyValue:()=>''}),console,
    LightweightCharts:{createChart(){const entry={removed:false,data:null,markers:null};rendered.push(entry);return {
      remove(){entry.removed=true;},addCandlestickSeries(){return {setData(data){entry.data=data;},setMarkers(markers){entry.markers=markers;}};},
      timeScale(){return {fitContent(){}};},subscribeCrosshairMove(){},subscribeClick(){}};}}};
  vm.createContext(context);vm.runInContext(source,context);
  return {view:context.KHQuantAnalysisAccountView,nodes,rendered};
}
const settle = async()=>{await new Promise(resolve=>setImmediate(resolve));await new Promise(resolve=>setImmediate(resolve));};
test('ledger projection preserves REDUCE and actual fill prices without changing input',()=>{
  const {view}=boot(),value=projection(),before=JSON.stringify(value);
  assert.equal(view.validateProjection(value,expected),value);
  assert.equal(JSON.stringify(value),before);
  const markers=view.tradeMarkers(value.trades);
  assert.deepEqual(Array.from(markers,x=>[x.time,x.text,x.position]),[
    ['2026-10-14','B','belowBar'],['2026-10-15','S','aboveBar'],['2026-10-16','S','aboveBar']]);
  assert.equal(value.trades[0].price,10.005);
});
test('multiple actual reductions on one day are grouped only in the marker, never in the ledger',()=>{
  const {view}=boot(),value=projection();value.trades.push({...value.trades[1],order_id:'extra'});
  const markers=view.tradeMarkers(value.trades);
  assert.equal(markers.find(x=>x.time==='2026-10-15').text,'S×2');
  assert.equal(value.trades.length,4);
});
for(const [name,change] of [
  ['foreign result',v=>v.result_id='b'.repeat(64)],['foreign stock',v=>v.symbol='002436.SZ'],
  ['foreign strategy',v=>v.strategy='structure_rules'],['future cutoff',v=>v.as_of='2026-10-19'],
  ['non-ledger semantics',v=>v.semantics='method_direction'],['implicit inference',v=>v.prediction_started=true],
  ['future candle',v=>v.bars[3].time='2026-10-19'],['duplicate candle',v=>v.bars[1].time=v.bars[0].time],
  ['invalid OHLC',v=>v.bars[0].close=100],['wrong marker count',v=>v.counts.SELL=1],
  ['invalid fill',v=>v.trades[0].price=NaN],['rejected order',v=>v.trades[0].action='WAIT'],
  ['missing Ledger identity',v=>delete v.trades[0].ledger_version],
])test(`fails closed for ${name}`,()=>{const {view}=boot(),value=projection();change(value);assert.throws(()=>view.validateProjection(value,expected));});
test('account chart is lazy, uses two GET readers, and displays all actual fills including reductions',async()=>{
  const {view,nodes,rendered}=boot(),calls=[];
  const api=async(url,options)=>{calls.push({url,options});return url.includes('/account-charts?')?
    {symbol:expected.symbol,as_of:expected.as_of,results:[{result_id:id,start:'2026-10-13',end:'2026-10-16',saved:false,strategies:['structure_kronos']}]}:projection();};
  view.setContext({symbol:expected.symbol,analysis_method:'structure_kronos',data_end:expected.as_of},api,{});
  assert.equal(calls.length,0);nodes['analysis-chart-content'].value='account';nodes['analysis-chart-content'].listeners.change();await settle();
  assert.equal(calls.length,2);assert.ok(calls.every(x=>!x.options?.body&&!x.options?.method));
  assert.equal(rendered.length,1);assert.equal(rendered[0].markers.length,3);
  assert.match(nodes['analysis-account-trades'].innerHTML,/S 减仓/);
  assert.match(nodes['analysis-account-trades'].innerHTML,/10\.005/);
  assert.match(nodes['analysis-account-state'].textContent,/已有回测预览/);
  assert.match(nodes['analysis-prediction-summary'].textContent,/B 1 \/ S 2/);
});
test('old asynchronous account lookup cannot populate a different stock context',async()=>{
  const {view,nodes,rendered}=boot();let resolve;
  const delayed=new Promise(done=>resolve=done),calls=[];
  view.setContext({symbol:expected.symbol,analysis_method:'structure_kronos',data_end:expected.as_of},url=>{calls.push(url);return delayed;},{});
  nodes['analysis-chart-content'].value='account';nodes['analysis-chart-content'].listeners.change();
  view.setContext({symbol:'002436.SZ',analysis_method:'structure_kronos',data_end:expected.as_of},()=>Promise.reject(Error('not requested')),{});
  resolve({symbol:expected.symbol,as_of:expected.as_of,results:[{result_id:id,strategies:['structure_kronos']}]});await settle();
  assert.equal(rendered.length,0);assert.equal(calls.length,1);
  assert.equal(nodes['analysis-chart-content'].value,'forecast');
  assert.equal(nodes['analysis-account-result'].textContent,'');
});
test('switching back to forecast disposes the account chart without changing method qualification',async()=>{
  const {view,nodes,rendered}=boot(),result={symbol:expected.symbol,analysis_method:'structure_kronos',data_end:expected.as_of,method_view:{valid:false,status:'unavailable'}};
  const before=JSON.stringify(result),api=async url=>url.includes('/account-charts?')?{symbol:expected.symbol,as_of:expected.as_of,results:[{result_id:id,start:'2026-10-13',end:'2026-10-16',strategies:['structure_kronos']}]}:projection();
  view.setContext(result,api,{});nodes['analysis-chart-content'].value='account';nodes['analysis-chart-content'].listeners.change();await settle();
  nodes['analysis-chart-content'].value='forecast';nodes['analysis-chart-content'].listeners.change();
  assert.ok(rendered[0].removed);assert.equal(JSON.stringify(result),before);
});
