/* Read-only rendering of a verified prediction snapshot; view controls never run inference. */
(() => {
  'use strict';
  const day = value => typeof value === 'object' && value ? `${value.year}-${String(value.month).padStart(2,'0')}-${String(value.day).padStart(2,'0')}` : String(value || '').slice(0,10);
  const numeric = value => value != null && Number.isFinite(Number(value));
  const format = value => numeric(value) ? Number(value).toLocaleString('zh-CN',{maximumFractionDigits:3}) : '—';
  function prepareSeries(value) {
    const verified = value.history_verification?.status === 'verified' && value.history_verification.input_sha256_match === true;
    const history = verified ? (value.history || []).map(row => ({...row,time:day(row.date)})) : [];
    if (history.some((row,index) => !['open','high','low','close'].every(key => numeric(row[key])) || row.time > value.as_of || index && row.time <= history[index-1].time) || history.length && history.at(-1).time !== value.as_of) throw Error('历史图表输入日期或价格校验失败');
    const forecast = (value.forecast || []).map(row => ({time:day(row.date || row.time),middle:row.close ?? row.close_median ?? row.median?.close ?? row.p50,low:row.close_p10 ?? row.p10,high:row.close_p90 ?? row.p90}));
    if (forecast.some((row,index) => row.time <= value.as_of || index && row.time <= forecast[index-1].time)) throw Error('预测交易日顺序校验失败');
    return {history,forecast};
  }
  function boundaryLabelLeft(coordinate,width,labelWidth) {
    const edge=Math.min(8,Math.max(0,(width-labelWidth)/2));
    return Math.max(edge,Math.min(coordinate-labelWidth-8,width-labelWidth-edge));
  }
  function create({prefix='light',isVisible}={}) {
  const $=id=>document.getElementById(id.replace(/^light-/,prefix+'-'));
  let chart=null,data=null,sequence=0,resizeObserver=null,themeObserver=null,controller=null;
  function visible() { return isVisible ? isVisible() : !document.getElementById('view-simulation')?.hidden && !$('light-prediction')?.hidden; }
  function dispose() { if (chart) chart.remove(); chart = null; resizeObserver?.disconnect(); resizeObserver = null; if($('light-forecast-boundary'))$('light-forecast-boundary').hidden = true; }
  function clear(message='') { ++sequence;controller?.abort();controller=null;dispose();data=null;if($('light-prediction-history-state'))$('light-prediction-history-state').textContent=message;if($('light-chart-readout'))$('light-chart-readout').textContent=''; }
  function boundary() {
    const host=$('light-forecast-boundary');
    if (!chart || !data || data.history_verification?.status !== 'verified') {host.hidden=true;return;}
    const coordinate=chart.timeScale().timeToCoordinate(data.as_of);
    host.hidden=coordinate==null || coordinate < 0 || coordinate > $('light-prediction-chart').clientWidth;
    if (!host.hidden) {
      host.style.left=`${coordinate}px`;host.style.right='0';const label=host.querySelector('span'),width=$('light-prediction-chart').clientWidth;
      label.textContent=`已知行情截止 ${data.as_of}`;label.style.maxWidth=`${Math.max(0,Math.min(200,width-16))}px`;label.style.whiteSpace='normal';label.style.width='max-content';
      label.style.left=`${boundaryLabelLeft(coordinate,width,label.offsetWidth)-coordinate}px`;
    }
  }
  function range() {
    if (!chart || !data) return;
    const series=prepareSeries(data), history=series.history, future=series.forecast;
    const count=$('light-history-window').value, requested=$('light-history-start').value;
    const defaultStart=history.length ? history[Math.max(0,history.length-(count==='all'?history.length:Number(count)||60))].time : future[0]?.time;
    const start=requested && history.some(row=>row.time>=requested) ? history.find(row=>row.time>=requested)?.time : defaultStart;
    const end=future.at(-1)?.time || history.at(-1)?.time;
    if (start && end && start<end) chart.timeScale().setVisibleRange({from:start,to:end}); else chart.timeScale().fitContent();
    boundary();
  }
  function draw() {
    if (!data || !visible()) return;
    dispose();
    if (!globalThis.LightweightCharts) {$('light-chart-readout').textContent='图表暂不可用，请阅读下方预测数据表。';return;}
    const {history,forecast}=prepareSeries(data), css=getComputedStyle(document.documentElement), color=name=>css.getPropertyValue('--'+name).trim();
    chart=LightweightCharts.createChart($('light-prediction-chart'),{autoSize:true,height:320,layout:{background:{color:color('panel')||'#fff'},textColor:color('ink')||'#24364a'},grid:{vertLines:{color:color('border')||'#e6ebf2'},horzLines:{color:color('border')||'#e6ebf2'}},timeScale:{timeVisible:false,borderVisible:false},rightPriceScale:{borderVisible:false}});
    if (history.length) {const candles=chart.addCandlestickSeries({upColor:'#d84b55',downColor:'#248274',borderUpColor:'#d84b55',borderDownColor:'#248274',wickUpColor:'#d84b55',wickDownColor:'#248274',priceLineVisible:false,lastValueVisible:false});candles.setData(history.map(({time,open,high,low,close})=>({time,open:Number(open),high:Number(high),low:Number(low),close:Number(close)})));}
    for (const [key,color] of [['middle','#397bd7'],['low','#7895b8'],['high','#7895b8']]) {const points=forecast.filter(row=>numeric(row[key])).map(row=>({time:row.time,value:Number(row[key])}));if(key==='middle'&&history.length)points.unshift({time:data.as_of,value:Number(history.at(-1).close)});const line=chart.addLineSeries({color,lineWidth:key==='middle'?2:1,lineStyle:LightweightCharts.LineStyle.Dashed,pointMarkersVisible:true,priceLineVisible:false,lastValueVisible:false});line.setData(points);}
    const historyMap=new Map(history.map(row=>[row.time,row])), futureMap=new Map(forecast.map(row=>[row.time,row]));
    chart.subscribeCrosshairMove(event=>{if(!event.time)return;const date=day(event.time),h=historyMap.get(date),f=futureMap.get(date);$('light-chart-readout').textContent=h?`${date} · 真实行情 · 开 ${format(h.open)} / 高 ${format(h.high)} / 低 ${format(h.low)} / 收 ${format(h.close)} · 量 ${format(h.volume)} ${data.input?.volume_unit||'单位未记录'} · 额 ${format(h.amount)} ${data.input?.amount_unit||'单位未记录'}`:f?`${date} · 预测 · 中位 ${format(f.middle)} · P10 ${format(f.low)} / P90 ${format(f.high)}`:'该日无可展示值';});
    chart.timeScale().subscribeVisibleLogicalRangeChange(boundary);
    resizeObserver=new ResizeObserver(boundary);resizeObserver.observe($('light-prediction-chart'));
    $('light-history-start').min=history[0]?.time||'';$('light-history-start').max=history.at(-1)?.time||'';$('light-history-start').disabled=!history.length;$('light-history-window').disabled=!history.length;
    range();
  }
  function apply(value) {
    prepareSeries(value);data=value;
    const verification=value.history_verification||{},basis=value.input?.price_basis;
    if ($('light-prediction-history-state')) $('light-prediction-history-state').textContent=verification.status==='verified'?`已核验 ${value.history.length} 根历史 · 截止 ${value.as_of} · ${basis==='unadjusted'?'实际未复权价格':basis||'价格口径未记录'}。显示范围不改变预测参数。`:`历史输入无法核验：${verification.reason||'该结果没有可核验的历史引用'}。保留原预测；未补算。`;
    draw();
  }
  async function render(result,api) {
    clear();const request=sequence;controller=typeof AbortController!=='undefined'?new AbortController():null;$('light-history-start').value='';$('light-history-window').value='60';
    const reference=result.result_ref?.cache_key,asOf=result.as_of||result.input?.as_of;
    apply({...result,as_of:asOf,history:[],history_verification:{status:'unavailable',reason:reference?'正在读取原预测的历史输入':'旧结果未保存外层预测引用'}});
    if (!reference) return;
    try {const value=await api('/predictions/'+encodeURIComponent(reference)+'/chart',controller?{signal:controller.signal}:{});if(request!==sequence)return;
      const actual=typeof value.prediction_ref==='object'?value.prediction_ref.cache_key:value.prediction_ref;
      if(actual!==reference || value.as_of!==asOf || value.symbol!==(result.symbol||result.input?.symbol))throw Error('历史图表与当前预测身份不一致');
      apply(value);
    } catch(error) {if(request!==sequence)return;if($('light-prediction-history-state'))$('light-prediction-history-state').textContent=`历史图表读取失败：${error.message||error}。保留原预测，不自动重算。`;}
  }
  function onRoute() {if(visible())draw();else dispose();}
  function boot() {
    $('light-history-window')?.addEventListener('change',()=>{$('light-history-start').value='';range();});
    $('light-history-start')?.addEventListener('change',range);
    $('light-history-reset')?.addEventListener('click',()=>{$('light-history-window').value='60';$('light-history-start').value='';range();});
    if(typeof MutationObserver!=='undefined'){themeObserver=new MutationObserver(()=>{if(visible())draw();});themeObserver.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});}
  }
  boot();return {render,onRoute,clear,destroy(){clear();themeObserver?.disconnect();}};
  }
  const defaultChart=create();
  globalThis.KHQuantPredictionChart={...defaultChart,create,prepareSeries,boundaryLabelLeft};
})();
