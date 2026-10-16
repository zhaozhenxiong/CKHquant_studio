/* Existing Broker fills, on their own saved candles; GET requests only. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=value=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('zh-CN',{maximumFractionDigits:6}):'—';
  const labels={structure_rules:'结构与量价',structure_kronos:'结构与量价 + Kronos',structure_kronos_score:'结构与量价 + Kronos + 评分模型',structure_kronos_l2:'结构与量价 + 同输入 L2 + Kronos'};
  const strategyFor={structure:'structure_rules',structure_kronos:'structure_kronos',structure_kronos_ma_mlp:'structure_kronos_score',structure_kronos_ma_l2:'structure_kronos_l2'};
  const finite=value=>typeof value==='number'&&Number.isFinite(value);
  const buySide=row=>['BUY','ADD'].includes(row.action);
  const actionLabel=row=>({BUY:'B 买入',ADD:'B 加仓',SELL:'S 卖出',REDUCE:'S 减仓'})[row.action];
  const day=value=>typeof value==='object'&&value?`${value.year}-${String(value.month).padStart(2,'0')}-${String(value.day).padStart(2,'0')}`:String(value||'').slice(0,10);
  function validateProjection(value, expected) {
    if(value?.schema!=='saved_account_chart_v1'||value.semantics!=='broker_ledger_fills'||
       ['result_id','symbol','strategy','as_of'].some(key=>value[key]!==expected[key])||
       value.prediction_started!==false||value.account_mutated!==false||
       value.current_method_qualification_changed!==false||value.price_basis!=='unadjusted')throw Error('回测图与所选股票、策略或截止日不一致');
    const bars=value.bars,trades=value.trades;
    if(!Array.isArray(bars)||!bars.length||!Array.isArray(trades))throw Error('回测行情或成交未记录');
    const dates=new Set();
    bars.forEach((row,index)=>{
      if(!/^\d{4}-\d{2}-\d{2}$/.test(row.time)||row.time>value.as_of||index&&row.time<=bars[index-1].time||
         ['open','high','low','close'].some(key=>!finite(row[key])||row[key]<=0)||
         row.high<Math.max(row.open,row.close,row.low)||row.low>Math.min(row.open,row.close,row.high))throw Error('回测行情日期或价格校验失败');
      dates.add(row.time);
    });
    trades.forEach(row=>{
      if(row.symbol!==value.symbol||!dates.has(row.date)||!['BUY','ADD','SELL','REDUCE'].includes(row.action)||
         !finite(row.price)||row.price<=0||!finite(row.shares)||row.shares<=0||
         !finite(row.fee)||row.fee<0||!Number.isInteger(row.ledger_version)||row.ledger_version<1||!row.order_id)throw Error('回测成交缺少可核验 Ledger 依据');
    });
    if(value.counts?.BUY!==trades.filter(buySide).length||
       value.counts?.SELL!==trades.filter(row=>!buySide(row)).length||
       value.counts?.ADD!==trades.filter(row=>row.action==='ADD').length||
       value.counts?.REDUCE!==trades.filter(row=>row.action==='REDUCE').length)throw Error('回测成交计数不一致');
    return value;
  }
  function tradeMarkers(trades) {
    const grouped=new Map();
    for(const trade of trades){const action=buySide(trade)?'BUY':'SELL',key=trade.date+'|'+action,prior=grouped.get(key);if(prior)prior.count++;else grouped.set(key,{date:trade.date,action,count:1});}
    return [...grouped.values()].sort((a,b)=>a.date.localeCompare(b.date)||a.action.localeCompare(b.action)).map(row=>({time:row.date,
      position:row.action==='BUY'?'belowBar':'aboveBar',color:row.action==='BUY'?'#d44954':'#278979',
      shape:row.action==='BUY'?'arrowUp':'arrowDown',text:(row.action==='BUY'?'B':'S')+(row.count>1?'×'+row.count:'')}));
  }
  let context=null,requestApi=null,onChange=null,key=null,sequence=0,controller=null;
  let options=null,data=null,chart=null,loadPromise=null;
  const isActive=()=>$('analysis-chart-content')?.value==='account';
  const visible=()=>isActive()&&!$('view-analysis')?.hidden&&!$('analysis-prediction')?.hidden;
  function dispose(){chart?.remove();chart=null;}
  function abort(){++sequence;controller?.abort();controller=null;loadPromise=null;}
  function clear(){abort();dispose();context=null;options=null;data=null;key=null;if($('analysis-chart-content'))$('analysis-chart-content').value='forecast';for(const id of ['analysis-account-result','analysis-account-strategy','analysis-account-trades','analysis-account-readout'])if($(id))$(id).textContent='';if($('analysis-account-chart'))$('analysis-account-chart').hidden=true;state('');toggle();}
  function toggle(){
    const active=isActive();if($('analysis-forecast-view'))$('analysis-forecast-view').hidden=active;
    for(const id of ['analysis-account-view','analysis-account-result-label','analysis-account-strategy-label'])if($(id))$(id).hidden=!active;
    if(!active)dispose();
  }
  function state(message){if($('analysis-account-state'))$('analysis-account-state').textContent=message;}
  function selected(){return {result_id:$('analysis-account-result').value,symbol:context.symbol,strategy:$('analysis-account-strategy').value,as_of:context.as_of};}
  function setStrategies(){
    const chosen=options?.find(row=>row.result_id===$('analysis-account-result').value);
    const allowed=chosen?.strategies||[];
    $('analysis-account-strategy').innerHTML=allowed.map(value=>`<option value="${esc(value)}">${esc(chosen?.strategy_labels?.[value]||labels[value]||value)}</option>`).join('');
    const preferred=strategyFor[context?.method];if(allowed.includes(preferred))$('analysis-account-strategy').value=preferred;
  }
  function readout(time){
    if(!data)return;
    const row=data.bars.find(row=>row.time===time),fills=data.trades.filter(row=>row.date===time);
    $('analysis-account-readout').textContent=(row?`${time} · 回测行情 · 开 ${num(row.open)} / 高 ${num(row.high)} / 低 ${num(row.low)} / 收 ${num(row.close)}`:'')+
      fills.map(fill=>` · ${actionLabel(fill)} ${num(fill.shares)} 股，实际成交 ${num(fill.price)} 元，费用 ${num(fill.fee)} 元`).join('');
  }
  function draw(){
    if(!data||!visible())return;dispose();
    $('analysis-prediction-title').textContent=`${data.symbol} · ${data.label} · 回测成交 B/S`;
    $('analysis-prediction-summary').textContent=`${data.display_start} → ${data.display_end} · B ${data.counts.BUY} / S ${data.counts.SELL}${data.counts.REDUCE?'（含减仓 '+data.counts.REDUCE+'）':''}`;
    state(`${data.saved?'已保存账户':'已有回测预览'} ${data.result_id.slice(0,12)} · 只读取原 Ledger 成交；截止日方法资格独立显示。`);
    $('analysis-account-trades').innerHTML=data.trades.length?`<details><summary>成交明细 · ${data.trades.length>100?'最近 100 / ':''}${data.trades.length} 笔</summary><div class="light-table-scroll"><table><thead><tr><th>成交日</th><th>B / S</th><th>实际成交价</th><th>股数</th><th>费用</th></tr></thead><tbody>${data.trades.slice(-100).map(row=>`<tr><td>${esc(row.date)}</td><td>${actionLabel(row)}</td><td>${num(row.price)}</td><td>${num(row.shares)}</td><td>${num(row.fee)}</td></tr>`).join('')}</tbody></table></div></details>`:'<p>该股票在此策略和截止日前没有实际成交。</p>';
    if(!globalThis.LightweightCharts){state('图表暂不可用，可查看下方原成交明细。');return;}
    $('analysis-account-chart').hidden=false;
    const css=getComputedStyle(document.documentElement),color=name=>css.getPropertyValue('--'+name).trim();
    chart=LightweightCharts.createChart($('analysis-account-chart'),{autoSize:true,height:320,layout:{background:{color:color('panel')||'#fff'},textColor:color('ink')||'#24364a'},grid:{vertLines:{color:color('border')||'#e6ebf2'},horzLines:{color:color('border')||'#e6ebf2'}},timeScale:{timeVisible:false,borderVisible:false},rightPriceScale:{borderVisible:false}});
    const candles=chart.addCandlestickSeries({upColor:'#d44954',downColor:'#278979',borderVisible:false,wickUpColor:'#d44954',wickDownColor:'#278979'});
    candles.setData(data.bars);candles.setMarkers(tradeMarkers(data.trades));chart.timeScale().fitContent();
    chart.subscribeCrosshairMove(point=>{if(point.time)readout(day(point.time));});
    chart.subscribeClick(point=>{if(point.time)readout(day(point.time));});
    readout(data.bars.at(-1).time);
  }
  async function loadSelected(){
    abort();dispose();data=null;$('analysis-account-chart').hidden=true;$('analysis-account-trades').textContent='';$('analysis-account-readout').textContent='';
    if(!context||!isActive()||!$('analysis-account-result').value||!$('analysis-account-strategy').value)return;
    const seq=sequence,expected=selected();controller=typeof AbortController!=='undefined'?new AbortController():null;
    state('正在读取已有回测成交…');
    try{
      const value=await requestApi('/api/light/account-charts/'+encodeURIComponent(expected.result_id)+'?'+new URLSearchParams({symbol:expected.symbol,strategy:expected.strategy,as_of:expected.as_of}),controller?{signal:controller.signal}:{});
      if(seq!==sequence||!isActive())return;data=validateProjection(value,expected);draw();
    }catch(error){if(seq===sequence&&isActive()){dispose();data=null;state(error.message||String(error));}}
  }
  async function loadOptions(){
    if(options){setStrategies();return loadSelected();}
    if(loadPromise)return loadPromise;
    abort();const seq=sequence;controller=typeof AbortController!=='undefined'?new AbortController():null;state('正在查找该股票已有回测…');
    const expected={...context};
    loadPromise=(async()=>{try{
      const value=await requestApi('/api/light/account-charts?'+new URLSearchParams({symbol:expected.symbol,as_of:expected.as_of}),controller?{signal:controller.signal}:{});
      if(seq!==sequence||!isActive())return;
      if(value.symbol!==expected.symbol||value.as_of!==expected.as_of||!Array.isArray(value.results))throw Error('已有回测列表身份不一致');
      options=value.results;
      $('analysis-account-result').innerHTML=options.map(row=>`<option value="${esc(row.result_id)}">${esc(row.start)} → ${esc(row.end)} · ${row.saved?'已保存':'预览'} ${esc(row.result_id.slice(0,8))}</option>`).join('');
      if(!options.length){$('analysis-account-strategy').innerHTML='';state('当前股票没有可读取的已有回测；可使用上方“用此股票回测”创建。');return;}
      setStrategies();await loadSelected();
    }catch(error){if(seq===sequence&&isActive())state(error.message||String(error));}finally{loadPromise=null;}})();
    return loadPromise;
  }
  function setContext(result,request,{dirty=false,onChange:callback}={}){
    if(dirty||!result?.symbol){clear();return;}
    const next={symbol:result.symbol,as_of:result.method_view?.as_of||result.as_of||result.data_end,method:result.analysis_method||result.method_view?.method};
    const nextKey=JSON.stringify(next);onChange=callback;requestApi=request;
    if(nextKey===key){toggle();return;}
    clear();context=next;key=nextKey;
  }
  function onRoute(){toggle();if(visible())draw();else dispose();}
  function activate(){if(!context)return;$('analysis-chart-content').value='account';toggle();if(data)draw();else loadOptions();}
  $('analysis-chart-content')?.addEventListener('change',()=>{abort();toggle();onChange?.();if(isActive()&&context){if(data)draw();else loadOptions();}});
  $('analysis-account-result')?.addEventListener('change',()=>{setStrategies();loadSelected();});
  $('analysis-account-strategy')?.addEventListener('change',loadSelected);
  if(typeof MutationObserver!=='undefined')new MutationObserver(()=>{if(visible())draw();}).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  globalThis.KHQuantAnalysisAccountView={setContext,clear,onRoute,isActive,activate,validateProjection,tradeMarkers};
})();
