/* CZSC workbench. Only service responses become charts, signals or ledger rows. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = (value, digits = 0) => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toLocaleString('zh-CN', {maximumFractionDigits: digits});
  const percent = (value) => value == null ? '—' : `${number(Number(value) * 100, 2)}%`;
  const basisLabel = (value) => ({unadjusted:'未复权价格',mixed_legacy_adjustment_unverified:'含旧复权未核验记录'})[value] || '价格口径尚未核验';
  const display = (value) => typeof value === 'object' && value !== null ? JSON.stringify(value) : String(value ?? '—');
  const day = (value) => value && typeof value === 'object' && 'year' in value ? `${value.year}-${String(value.month).padStart(2,'0')}-${String(value.day).padStart(2,'0')}` : typeof value === 'number' ? new Intl.DateTimeFormat('sv-SE', {timeZone:'Asia/Shanghai'}).format(new Date(value < 1e12 ? value * 1000 : value)) : String(value ?? '').slice(0, 10);
  const activeStatuses = new Set(['queued', 'running', 'cancelling']);
  const statusNames = {queued:'等待开始', running:'运行中', succeeded:'已完成', failed:'失败', cancelling:'正在取消', cancelled:'已取消'};
  const kindNames = {analysis:'结构分析',scan:'全市场扫描',backtest:'策略回测',update:'行情更新',model_training:'完整模型训练',model_validation:'已完成模型复核',current_train:'完整模型训练'};
  const PAGE_SIZE = 100;
  const state = {analysis:null, query:null, frequency:'日线', dates:[], replayIndex:0, scan:null, backtest:null, scanPage:0, tradePage:0, tasks:[], delivered:new Set(), tracked:{}, favorites:new Set(), symbolCatalog:new Map(), analysisSequence:0, taskSequence:0, equityMode:'equity', pollTimer:null, watchlist:null, holdings:null, personalLoaded:false, watchlistBusy:false, holdingsBusy:false, watchlistSequence:0, holdingsSequence:0,research:null,pendingResearchContexts:{}};
  let chart, candles, volume, equityChart, equitySeries;
  let penSeries = [];
  let indicatorSeries = [];
  let resizeObserver;
  let searchTimer;
  let noticeTimer;
  let personalReady;
  let computeLoaded = false;
  let latestDefaultEnd;
  const dateModes = {analysis:'latest',scan:'latest',backtest:'latest'};
  let latestDateSequence = 0;
  let latestCoverage = null;
  let popupRenderedHtml = null, popupRenderedContext = null;
  let markerMappings = [];
  let markerGroups = [];
  let chartSelection = {hover:null,pinned:null,selectedDate:null,popupOpen:false};
  let popupAnchor = null;
  let selectedReferenceLine = null, selectedReferenceSeries = null, selectedReferenceKey = null;
  let referenceMutationDepth = 0;
  let historyRows = [];
  let historyPage = 0;
  let historySeriesContext = null;
  let analysisLoading = false;
  let openRunSequence = 0;
  const evidenceUI = () => globalThis.KHQuantChartEvidence;
  const displayedConsensus = () => state.analysis?.trend_consensus || state.analysis?.consensus_decisions;
  const displayedEvidence = () => state.analysis?.trend_consensus ? evidenceUI()?.accountEvidence(state.analysis) : state.analysis?.chart_evidence;
  function storeHistoryEvidence(incoming, preserveDates, seriesContext) {
    const preserve = preserveDates && (!seriesContext || seriesContext === historySeriesContext);
    historyRows = preserve ? [...new Map([...historyRows,...incoming].map((row) => [row.date,row])).values()] : incoming;
    historySeriesContext = seriesContext || null;
  }
  let priceOverlayFrame = null;
  const computeSequences = {analysis:0,scan:0,backtest:0,data:0};
  const usageNames = {historical:'历史滚动研究',forward:'当前时点研究'};
  const entryPolicyNames = {legacy:'原目标仓位对照',fresh:'有效买点研究',risk:'价格与风险实验'};
  function entryPolicyCaption(policy) {
    return policy === 'risk' ? '价格与风险实验：区间、结构失效价及风险仓位未经独立认证；参考价不保证成交。' : policy === 'fresh' ? '有效买点研究：区分新入场与理论持仓；规则尚未独立认证，ML资格单独判断。' : '原目标仓位对照：保留原生 Position 的目标仓位与风险语义，供同一输入比较。';
  }

  async function request(path, options = {}) {
    const headers = {...options.headers};
    if (options.body) headers['Content-Type'] = 'application/json';
    const token = sessionStorage.getItem('khquant_api_token');
    if (token) headers.Authorization = `Bearer ${token}`;
    const response = await fetch(path, {...options, headers, credentials:'same-origin'});
    const text = await response.text();
    let payload;
    try { payload = text ? JSON.parse(text) : {}; } catch (_) { throw new Error(`服务返回了无法解析的响应（${response.status}）。`); }
    if (!response.ok) throw new Error(display(payload.detail || payload.error || `请求失败（${response.status}）`));
    return payload;
  }
  function showError(error) { $('page-error').hidden = false; $('page-error').querySelector('span').textContent = error.message || String(error); }
  function notice(message) { clearTimeout(noticeTimer); $('notice').textContent = message; $('notice').hidden = false; noticeTimer = setTimeout(() => { $('notice').hidden = true; }, 4000); }
  function showView(selected) {
    document.querySelectorAll('.view').forEach((view) => { view.hidden = view.id !== `view-${selected}`; });
    document.querySelectorAll('[data-route]').forEach((link) => { if (link.dataset.route === selected) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current'); });
  }
  function route() {
    const name = location.hash.slice(1) || location.pathname.replace(/^\//,'');
    const selected = ['analysis', 'scan', 'backtest', 'watchlist', 'holdings', 'training', 'parameters', 'data'].includes(name) ? name : 'analysis';
    if (selected !== 'analysis') resetChartSelection();
    showView(selected);
    requestAnimationFrame(resizeCharts);
    if (['training','parameters'].includes(selected)) globalThis.KHQuantTraining?.refresh().catch(showError);
    if (selected === 'data') { refreshTasks().catch(showError); refreshRuns().catch(showError); }
    if (['analysis','scan','backtest'].includes(selected)) refreshResearch().catch(showError);
    if (selected === 'watchlist' || selected === 'holdings') (state.personalLoaded ? selected === 'watchlist' ? refreshWatchlist() : refreshHoldings() : ensurePersonal()).catch(showError);
  }
  function colors() { const style = getComputedStyle(document.documentElement); return Object.fromEntries(['panel','ink','muted','border','accent','up','down','ma5','ma10','ma20','ma60','boll','chips'].map((key) => [key, style.getPropertyValue(`--${key}`).trim()])); }
  function chartOptions(host) {
    const c = colors();
    return {width:Math.max(1, host.clientWidth), height:host.clientHeight || 350, layout:{background:{type:'solid', color:c.panel}, textColor:c.muted, fontFamily:'Microsoft YaHei UI, sans-serif', fontSize:11}, grid:{vertLines:{color:c.border, visible:false}, horzLines:{color:c.border}}, rightPriceScale:{borderVisible:false}, timeScale:{borderVisible:false, timeVisible:false, rightOffset:5}, crosshair:{mode:0}, localization:{locale:'zh-CN'}};
  }
  function ensureChart() {
    if (chart) return;
    if (!window.LightweightCharts) throw new Error('本地图表资源未加载，请检查静态资源。');
    const c = colors();
    chart = LightweightCharts.createChart($('analysis-chart'), chartOptions($('analysis-chart')));
    candles = chart.addCandlestickSeries({upColor:c.up, downColor:c.down, borderUpColor:c.up, borderDownColor:c.down, wickUpColor:c.up, wickDownColor:c.down, priceLineVisible:false});
    volume = chart.addHistogramSeries({priceFormat:{type:'volume'}, priceScaleId:'volume', visible:false, lastValueVisible:false, priceLineVisible:false});
    volume.priceScale().applyOptions({scaleMargins:{top:.82,bottom:0}, visible:false});
    chart.timeScale().subscribeVisibleLogicalRangeChange(refreshChartOverlays);
    chart.subscribeCrosshairMove((param) => {
      const bar = param.seriesData.get(candles);
      updateChartSelection(bar ? 'hover' : 'leave',bar ? day(param.time) : null,param.point);
    });
    chart.subscribeClick((param) => { const time = param.time ? day(param.time) : null, groups = popupGroupsAt(time); if (time && groups.length) { const clickedGroup = groups.find((group) => group.id === param.hoveredObjectId); const selectedDate = clickedGroup ? evidenceUI()?.groupDates([clickedGroup]).at(-1) : null; updateChartSelection('pin',time,param.point,selectedDate); highlightEvent(evidenceUI()?.groupDates(groups) || [time]); } else updateChartSelection('clear'); });
    $('analysis-chart').addEventListener('pointerleave',() => updateChartSelection('leave'));
    ['pointermove','pointerup','wheel'].forEach((event) => $('analysis-chart').addEventListener(event,schedulePriceOverlays));
    resizeObserver = new ResizeObserver(resizeCharts);
    resizeObserver.observe($('analysis-chart'));
    resizeObserver.observe($('equity-chart'));
  }
  function resizeCharts() {
    if (chart && !$('view-analysis').hidden) chart.resize(Math.max(1, $('analysis-chart').clientWidth), $('analysis-chart').clientHeight);
    if (equityChart && !$('view-backtest').hidden) equityChart.resize(Math.max(1, $('equity-chart').clientWidth), $('equity-chart').clientHeight);
    refreshChartOverlays();
    if (chartSelection.pinned) renderChartPopup();
  }
  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    $('theme-toggle').textContent = theme === 'dark' ? '明色' : '暗色';
    localStorage.setItem('khquant_czsc_theme', theme);
    const c = colors();
    if (chart) { chart.applyOptions(chartOptions($('analysis-chart'))); candles.applyOptions({upColor:c.up,downColor:c.down,borderUpColor:c.up,borderDownColor:c.down,wickUpColor:c.up,wickDownColor:c.down}); renderAnalysis(false); }
    if (equityChart) { equityChart.applyOptions(chartOptions($('equity-chart'))); equitySeries.applyOptions({color:c.accent}); }
  }
  function emptyChart(message, detail) { resetChartSelection(); clearIndicators(); $('chip-profile').hidden = true; $('chip-profile-caption').hidden = true; $('chip-profile-overlay').replaceChildren(); syncChartRange(); $('divergence-details').hidden = true; $('divergence-list').replaceChildren(); $('analysis-empty').hidden = false; $('analysis-empty').querySelector('strong').textContent = message; $('analysis-empty').querySelector('span').textContent = detail; $('marker-summary').textContent = '尚无可展示的买卖标记'; markerMappings = []; markerGroups = []; $('marker-details').textContent = KHQuantMarkers.describe(markerMappings,null,state.frequency); $('zone-overlay').replaceChildren(); }
  function normalizeBars(rows) {
    return (rows || []).map((bar) => ({time:day(bar.time), open:Number(bar.open), high:Number(bar.high), low:Number(bar.low), close:Number(bar.close), volume:Number(bar.volume ?? bar.vol ?? 0), is_closed:bar.is_closed !== false}));
  }
  function clearIndicators() {
    indicatorSeries.forEach((series) => chart.removeSeries(series));
    indicatorSeries = [];
    $('indicator-readout').hidden = true;
    ['context','ma','boll','chips'].forEach((name) => { $(`indicator-${name}`).textContent = ''; });
  }
  function renderIndicators(data) {
    clearIndicators();
    const rows = data?.indicators?.rows || [], c = colors();
    const lines = [];
    if ($('layer-ma').checked) [5,10,20,60].forEach((period) => lines.push({key:`ma${period}`,color:c[`ma${period}`],dashed:false}));
    if ($('layer-boll').checked) ['upper','lower',...($('layer-ma').checked ? [] : ['mid'])].forEach((part) => lines.push({key:`boll_${part}`,color:c.boll,dashed:true}));
    if ($('layer-chips').checked) ['low','high'].forEach((part) => lines.push({key:`chip70_${part}`,color:c.chips,dashed:true}));
    lines.forEach(({key,color,dashed}) => {
      if (!rows.some((row) => Number.isFinite(row[key]))) return;
      const series = chart.addLineSeries({color,lineWidth:1,lineStyle:dashed ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
      series.setData(rows.map((row) => ({time:day(row.time),...(Number.isFinite(row[key]) ? {value:row[key]} : {})})));
      indicatorSeries.push(series);
    });
    renderIndicatorReadout();
  }
  function renderIndicatorReadout(time = null) {
    const rows = state.analysis?.frequencies?.[state.frequency]?.indicators?.rows || [];
    const row = time ? rows.find((item) => day(item.time) === time) : rows.at(-1);
    $('indicator-readout').hidden = !row || !['ma','boll','chips'].some((name) => $(`layer-${name}`).checked);
    if (!row) return;
    $('indicator-context').textContent = `${time ? '所指K线' : '最新K线'} ${day(row.time)} · ${state.frequency}${row.is_closed === false ? ' · 形成中，仅供展示' : ''}`;
    $('indicator-ma').hidden = !$('layer-ma').checked;
    const alignment = {bullish:'多头排列',bearish:'空头排列',mixed:'交错排列',insufficient:'数据不足'};
    const direction = {rising:'上行',falling:'下行',flat:'走平',insufficient:'数据不足'};
    $('indicator-ma').innerHTML = [5,10,20,60].map((period) => `<span class="indicator-ma${period}"><i></i>MA${period} ${number(row[`ma${period}`],2)}</span>`).join('') + `<span class="indicator-structure">${esc(alignment[row.ma_alignment] || '数据不足')} · MA20${esc(direction[row.ma20_direction] || '数据不足')} · MA60${esc(direction[row.ma60_direction] || '数据不足')}</span>`;
    $('indicator-boll').hidden = !$('layer-boll').checked;
    $('indicator-boll').textContent = `BOLL20（2σ）上 ${number(row.boll_upper,2)} · 中 ${number(row.boll_mid,2)} · 下 ${number(row.boll_lower,2)}`;
    $('indicator-chips').hidden = !$('layer-chips').checked;
    const profile = selectedChipProfile();
    $('indicator-chips').textContent = profile ? `70%估算下 ${number(profile.chip70_low,2)} · 上 ${number(profile.chip70_high,2)} · 中位价 ${number(profile.chip50,2)} · 日线截至 ${profile.as_of ? day(profile.as_of) : '—'}` : `70%估算下 ${number(row.chip70_low,2)} · 上 ${number(row.chip70_high,2)} · 中位价 ${number(row.chip50,2)} · 日线截至 ${row.chip_as_of ? day(row.chip_as_of) : '—'}`;
  }
  function renderAnalysis(fit = true) {
    if (!state.analysis) { clearIndicators(); $('chip-profile').hidden = true; $('chip-profile-caption').hidden = true; $('chip-profile-overlay').replaceChildren(); syncChartRange(); return; }
    const data = state.analysis.frequencies?.[state.frequency];
    const bars = normalizeBars(data?.bars);
    $('chart-symbol').textContent = `${state.analysis.symbol || state.query?.symbol || ''} ${state.analysis.name || state.symbolCatalog.get(state.analysis.symbol)?.name || ''}`.trim();
    renderDiagnostics('analysis-caveats',state.analysis.warnings);
    $('analysis-caveats').open = false;
    if (!bars.length) { if (candles) candles.setData([]); $('chart-information').textContent = `${state.frequency} · 0 根 · ${basisLabel(state.analysis.price_basis)}`; $('structure-list').innerHTML = '<div class="empty-copy">该周期暂无可展示结构。</div>'; $('analysis-state').textContent = '该周期暂无K线'; emptyChart('该周期没有可用行情', '请检查日期范围和数据覆盖，或切换周期。'); return; }
    ensureChart();
    $('analysis-empty').hidden = true;
    const profileVisible = $('layer-chips').checked;
    $('chip-profile-caption').hidden = !profileVisible;
    if ($('chip-profile').hidden === profileVisible) { $('chip-profile').hidden = !profileVisible; chart.resize(Math.max(1,$('analysis-chart').clientWidth),$('analysis-chart').clientHeight); }
    const c = colors();
    candles.setData(bars.map(({volume:_, is_closed, ...bar}) => ({...bar,...(is_closed ? {} : {color:c.border,borderColor:c.muted,wickColor:c.muted})})));
    const first = bars[0].time, last = bars[bars.length - 1].time;
    volume.setData(bars.map((bar) => ({time:bar.time,value:bar.volume,color:!bar.is_closed ? c.border : bar.close >= bar.open ? c.up : c.down})));
    volume.applyOptions({visible:$('layer-volume').checked});
    candles.priceScale().applyOptions({scaleMargins:{top:.08,bottom:$('layer-volume').checked ? .22 : .08}});
    renderIndicators(data);
    penSeries.forEach((series) => chart.removeSeries(series));
    penSeries = [];
    if ($('layer-pens').checked) (data.pens || []).forEach((pen) => {
      let start = day(pen.start_time);
      const end = day(pen.end_time);
      if (start >= end) return;
      if (end < first || start > last) return;
      let startPrice = Number(pen.start_price);
      if (start < first) { const fraction = (Date.parse(first) - Date.parse(start)) / (Date.parse(end) - Date.parse(start)); startPrice += (Number(pen.end_price) - startPrice) * fraction; start = first; }
      if (start >= end) return;
      const series = chart.addLineSeries({color:c.accent,lineWidth:2,lineStyle:pen.is_confirmed ? LightweightCharts.LineStyle.Solid : LightweightCharts.LineStyle.Dashed,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
      series.setData([{time:start,value:startPrice},{time:end,value:Number(pen.end_price)}]);
      penSeries.push(series);
    });
    const markers = [];
    if ($('layer-fractals').checked) (data.fractals || []).forEach((fractal) => {
      const top = /top|ding|顶|^g$/i.test(fractal.kind || '');
      markers.push({time:day(fractal.time), position:top ? 'aboveBar' : 'belowBar', color:top ? c.down : c.up, shape:fractal.is_confirmed ? (top ? 'arrowDown' : 'arrowUp') : 'circle', text:fractal.is_confirmed ? (top ? '顶' : '底') : (top ? '形成顶' : '形成底'), size:fractal.is_confirmed ? .6 : .4});
    });
    const divergences = $('layer-divergences').checked ? data.divergences || [] : [];
    divergences.forEach((item) => {
      const top = item.kind === 'top';
      markers.push({time:day(item.time),position:top ? 'aboveBar' : 'belowBar',color:top ? c.down : c.up,shape:top ? 'arrowDown' : 'arrowUp',text:top ? '顶背驰参考' : '底背驰参考',size:.8});
    });
    $('divergence-details').hidden = !divergences.length;
    $('divergence-list').innerHTML = [...divergences].sort((a,b) => String(b.available_at).localeCompare(String(a.available_at))).slice(0,3).map((item) => `<article class="event-entry"><strong class="${item.kind === 'top' ? 'valuation-loss' : 'valuation-gain'}">${esc(item.label)} · 锚点 ${esc(day(item.time))} · ${number(item.price,2)}</strong><p>末笔确认 ${esc(item.confirmed_at || '—')}<br>首次可见 ${esc(item.available_at || '—')}</p><details class="rule-original"><summary>五笔锚点</summary><p>${(item.pen_anchors || []).map((pen) => `${esc(day(pen[0]))} → ${esc(day(pen[1]))}`).join('<br>')}</p></details><button type="button" data-locate-event="${esc(day(item.time))}">定位参考</button></article>`).join('');
    const symbol = state.query?.symbol;
    const trendStrategy = Boolean(state.analysis.trend_consensus);
    const accountMarkers = trendStrategy ? evidenceUI()?.accountMarkerInputs(state.analysis,symbol) : null;
    const selectedEvents = trendStrategy ? accountMarkers?.events || [] : displayedConsensus()?.events || [];
    const selectedTrades = trendStrategy ? accountMarkers?.trades || [] : state.backtest?.trades || [];
    const tradeMarkers = KHQuantMarkers.build({bars, dailyBars:normalizeBars(state.analysis.frequencies?.['日线']?.bars), frequency:state.frequency,
      events:$('layer-signals').checked ? selectedEvents : [], trades:$('layer-trades').checked ? selectedTrades : [],
      symbol,consensusVersion:displayedConsensus()?.version || displayedConsensus()?.config?.version || '',start:state.query?.start || '',asOf:day(state.query?.as_of || state.analysis.as_of || state.analysis.data_end),up:c.up,down:c.down});
    markers.push(...tradeMarkers.markers);
    const directionMarkers = trendStrategy && $('layer-directions')?.checked ? KHQuantMarkers.build({bars,dailyBars:normalizeBars(state.analysis.frequencies?.['日线']?.bars),frequency:state.frequency,
      events:state.analysis.trend_consensus.events || [],symbol,consensusVersion:displayedConsensus()?.version || 'trend_consensus',start:state.query?.start || '',asOf:day(state.query?.as_of || state.analysis.as_of),up:c.up,down:c.down}) : null;
    if (directionMarkers) markers.push(...directionMarkers.markers.map((item) => ({...item,id:`${item.id}:direction`,text:`方向${item.text}`,size:.7})));
    const wyckoff = KHQuantWyckoffMarkers.buildDetailed({events:$('layer-wyckoff').checked ? state.analysis.wyckoff_events || [] : [],bars,frequency:state.frequency,symbol,start:state.query?.start || '',asOf:day(state.query?.as_of || state.analysis.as_of || state.analysis.data_end)});
    const wyckoffMarkers = wyckoff.markers;
    markers.push(...wyckoffMarkers);

    markerMappings = tradeMarkers.mappings;
    markerGroups = [...(tradeMarkers.groups || []),...(directionMarkers?.groups || []).map((group) => ({...group,id:`${group.id}:direction`,diagnostic:true})),...wyckoff.groups];
    $('marker-details').textContent = KHQuantMarkers.describe(markerMappings,null,state.frequency);
    $('marker-summary').textContent = state.analysis.consensus_decisions ? `最终 B/S ${tradeMarkers.intentCount} 个 · 周期标记 ${tradeMarkers.intentMarkerCount} 个${!tradeMarkers.intentCount && !tradeMarkers.deferredCount ? ' · 本区间暂无满足仲裁条件的最终点，可在历史判断查看原因' : ''} · 实际成交 ${tradeMarkers.tradeCount} 条${wyckoff.eventCount ? ` · 量价事件 ${wyckoff.eventCount} 条／${wyckoffMarkers.length} 个标记` : ''} · 点击查看多 Agent 判断${state.frequency === '日线' ? '' : '与原始日期'}${tradeMarkers.formingProjectedCount ? ` · ${tradeMarkers.formingProjectedCount} 个日线最终点投影于形成中周期` : ''}${wyckoff.formingProjectedCount ? ` · ${wyckoff.formingProjectedCount} 条已知量价事件位于形成中周期` : ''}${tradeMarkers.deferredCount ? ` · ${tradeMarkers.deferredCount} 个标记等待所属周期闭合` : ''}` : '该结果没有多 Agent 仲裁，不显示最终买卖点；请重新分析。';
    markers.sort((a,b) => a.time.localeCompare(b.time));
    candles.setMarkers(markers);
    if (trendStrategy && !accountMarkers?.valid) $('marker-summary').textContent = '趋势共识策略 · 账户绑定不可用；账户买卖点与成交未显示。';
    if (trendStrategy && accountMarkers?.valid) $('marker-summary').textContent = `趋势共识策略 · 选定账户 B/S ${tradeMarkers.intentCount} 个 · 周期标记 ${tradeMarkers.intentMarkerCount} 个 · 账户成交 ${tradeMarkers.tradeCount} 条 · ${evidenceUI().accountContextText(accountMarkers.context)}${accountMarkers?.rejectedRecords ? ` · ${accountMarkers.rejectedRecords} 条账户或时点不符记录未显示` : ''}${wyckoff.eventCount ? ` · 原始量价事件 ${wyckoff.eventCount} 条` : ''} · 点击查看评分和退出原因`;
    if (fit) chart.timeScale().fitContent();
    $('chart-symbol').textContent = `${state.analysis.symbol || symbol || ''} ${state.analysis.name || state.symbolCatalog.get(state.analysis.symbol)?.name || ''}`.trim();
    const formingCount = bars.filter((bar) => !bar.is_closed).length;
    $('chart-information').textContent = `${state.frequency} · ${bars.length} 根${formingCount ? `（${formingCount} 根淡色形成中，未用于规则）` : ''} · 可知截至 ${state.query?.as_of || state.analysis.as_of || state.analysis.data_end || last} · ${basisLabel(state.analysis.price_basis)}`;
    $('analysis-state').textContent = state.analysis.warmup?.eligible === false ? '预热不足，不能产生有效交易' : state.query?.as_of ? `回放至 ${state.query.as_of}` : '最新可用结构';
    const structures = [...(data.fractals || []).map((item) => ({...item,label:item.kind === 'top' ? '顶分型' : '底分型'})),...(data.pens || []).map((item) => ({...item,label:'笔',time:item.end_time})),...(data.zones || []).map((item) => ({...item,label:'中枢',time:item.end_time}))].sort((a,b) => day(b.time).localeCompare(day(a.time))).slice(0,6);
    $('structure-list').innerHTML = structures.length ? structures.map((item) => `<article class="event-entry"><strong>${esc(item.label)} · ${esc(day(item.time))} · ${item.is_confirmed ? '已确认' : '形成中'}</strong><p>${item.is_confirmed ? `确认于 ${esc(item.confirmed_at || '服务未返回确认时间')}` : '尚未确认，后续可能变化'}</p><button type="button" data-locate-event="${esc(day(item.time))}">定位结构</button></article>`).join('') : '<div class="empty-copy">该周期尚无结构。</div>';
    refreshChartOverlays();
  }
  function schedulePriceOverlays() {
    if (priceOverlayFrame != null) return;
    priceOverlayFrame = requestAnimationFrame(() => { priceOverlayFrame = null; drawZones(); drawChipProfile(); });
  }
  function refreshChartOverlays() { drawZones(); drawChipProfile(); syncChartRange(); schedulePriceOverlays(); }
  function selectedChipProfile() {
    if (chartSelection.pinned && chartSelection.selectedDate) return state.analysis?.frequencies?.['日线']?.indicators?.chip_profiles?.[chartSelection.selectedDate] || {status:'unavailable',as_of:chartSelection.selectedDate};
    const indicators = state.analysis?.frequencies?.[state.frequency]?.indicators;
    const selected = chartSelection.pinned || chartSelection.hover;
    return selected ? indicators?.chip_profiles?.[selected] || {status:'unavailable',as_of:selected} : indicators?.chip_profile;
  }
  function groupsAt(time) { return markerGroups.filter((group) => day(group.time || group.bar_date) === time); }
  function finalGroupsAt(time) { return groupsAt(time).filter((group) => group.visible !== false && group.kind === 'intention' && group.sources?.includes('consensus')); }
  function popupGroupsAt(time) {
    const groups = groupsAt(time).filter((group) => group.visible !== false && (group.kind === 'wyckoff' || group.kind === 'intention' && group.sources?.some((source) => ['consensus','account_action'].includes(source)) || Boolean(state.analysis?.trend_consensus) && group.kind === 'execution'));
    if (groups.length || !state.analysis?.trend_consensus || !time) return groups;
    if (!state.analysis.frequencies?.[state.frequency]?.bars?.some((bar) => day(bar.time) === time)) return [];
    const key = KHQuantMarkers.periodKey(time,state.frequency);
    const rows = (displayedEvidence()?.rows || []).filter((row) => KHQuantMarkers.periodKey(row.date,state.frequency) === key);
    // Inspection groups are never supplied to the marker builder or counted
    // as signals. They only select actual daily evidence within this candle.
    return rows.length ? [{id:`inspection:${state.frequency}:${time}`,kind:'inspection',time,visible:false,
      entries:rows.map((row) => ({kind:'inspection',original_date:row.date,event_date:row.date,displayed:true,row}))}] : [];
  }
  function updateChartSelection(type, time = null, point = null, selectedDate = null) {
    const ui = evidenceUI();
    if (analysisLoading || referenceMutationDepth > 0) return;
    if (!ui) { $('marker-details').textContent = KHQuantMarkers.describe(markerMappings,time,state.frequency); renderIndicatorReadout(time); return; }
    const previous = chartSelection;
    const canOpen = type === 'pin' && popupGroupsAt(time).length > 0;
    chartSelection = ui.selectionTransition(chartSelection,{type:type === 'locate' ? 'pin' : type,time,date:selectedDate,popupOpen:canOpen});
    if (chartSelection.pinned && (type === 'hover' || type === 'leave')) return;
    if (point) { const rect = $('analysis-chart').getBoundingClientRect(); popupAnchor = {x:rect.left + point.x,y:rect.top + point.y}; }
    const active = chartSelection.pinned || chartSelection.hover;
    $('marker-details').textContent = KHQuantMarkers.describe(markerMappings,active,state.frequency);
    $('chart-evidence-reset').hidden = !chartSelection.pinned;
    $('chart-selection-status').textContent = active ? `${chartSelection.pinned ? '已固定' : '所指 K 线'} ${active} · ${state.frequency} · 筹码截至 ${selectedChipProfile()?.as_of || active}` : '悬浮 K 线切换筹码分布；点击固定日期。';
    renderIndicatorReadout(active);
    if (previous.hover !== chartSelection.hover || previous.pinned !== chartSelection.pinned || previous.selectedDate !== chartSelection.selectedDate || type === 'clear') { drawChipProfile(); syncChartRange(); }
    renderChartPopup();
    if (type === 'pin') $('chart-evidence-popup').scrollTop = 0;
  }
  function clearSelectedReference() {
    const line = selectedReferenceLine, series = selectedReferenceSeries;
    // Chart removal may synchronously emit crosshair callbacks. Detach first
    // so reentrant cleanup cannot remove the same object a second time.
    selectedReferenceLine = null; selectedReferenceSeries = null; selectedReferenceKey = null;
    if (!line && !series) return;
    referenceMutationDepth++;
    try {
      if (line && candles?.removePriceLine) candles.removePriceLine(line);
      if (series && chart) chart.removeSeries(series);
    } finally {
      referenceMutationDepth--;
    }
  }
  function renderSelectedReference(groups, active) {
    if (referenceMutationDepth > 0) return;
    const selection = evidenceUI()?.referenceSelection({evidence:displayedEvidence(),consensus:displayedConsensus(),groups,barDate:active,selectedDate:chartSelection.selectedDate});
    if (!selection || !candles?.createPriceLine || !chart) { clearSelectedReference(); return; }
    const key = [state.frequency,selection.bar_date,selection.signal_date,selection.event_id,selection.price,selection.action].join(':');
    if (key === selectedReferenceKey) return;
    clearSelectedReference();
    const c = colors(), buying = selection.action === 'BUY', color = buying ? c.up : c.down;
    // Creation and setData may emit the same synchronous chart callbacks.
    // Keep interaction paused until both references and their key are complete.
    referenceMutationDepth++;
    try {
      selectedReferenceLine = candles.createPriceLine({price:selection.price,color,lineWidth:1,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:`${buying ? 'B' : 'S'} 参考 · 非成交`});
      selectedReferenceSeries = chart.addLineSeries({color,lineVisible:false,pointMarkersVisible:true,pointMarkersRadius:4,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false});
      selectedReferenceSeries.setData([{time:selection.bar_date,value:selection.price}]);
      selectedReferenceKey = key;
    } finally {
      referenceMutationDepth--;
    }
  }
  function resetChartSelection() {
    popupRenderedHtml = null; popupRenderedContext = null;
    chartSelection = {hover:null,pinned:null,selectedDate:null,popupOpen:false}; popupAnchor = null;
    clearSelectedReference();
    $('chart-evidence-popup').hidden = true; $('chart-evidence-popup').replaceChildren(); $('chart-evidence-reset').hidden = true;
    $('chart-selection-status').textContent = '悬浮 K 线切换筹码分布；点击固定日期。';
  }
  function renderChartPopup() {
    const ui = evidenceUI(), host = $('chart-evidence-popup'), active = chartSelection.pinned;
    const groups = popupGroupsAt(active);
    if (!ui || !chartSelection.popupOpen || !active || !state.analysis || analysisLoading || !groups.length) { host.hidden = true; clearSelectedReference(); return; }
    renderSelectedReference(groups,active);
    host.dataset.pinned = String(Boolean(chartSelection.pinned));
    const context = `${state.analysisSequence}|${active}|${state.frequency}|${chartSelection.selectedDate || ''}`;
    const html = ui.popupHtml({evidence:displayedEvidence(),consensus:displayedConsensus(),groups,barDate:active,frequency:state.frequency,selectedDate:chartSelection.selectedDate,pinned:Boolean(chartSelection.pinned),profileAsOf:selectedChipProfile()?.as_of});
    if (html !== popupRenderedHtml) {
      const keep = context === popupRenderedContext;
      const expanded = keep ? [...(host.querySelectorAll?.('details[data-agent-details]') || [])].map((node) => [node.dataset.agentDetails,node.open]) : [];
      const scroll = keep ? host.scrollTop : 0;
      host.innerHTML = html;
      for (const [key,open] of expanded) {
        const node = host.querySelector?.(`details[data-agent-details="${key}"]`);
        if (node) node.open = open;
      }
      host.scrollTop = scroll;
      popupRenderedHtml = html; popupRenderedContext = context;
    }
    host.hidden = false;
    const anchor = popupAnchor || {x:Math.min(window.innerWidth - 24,160),y:Math.min(window.innerHeight - 24,180)};
    const width = host.offsetWidth, height = host.offsetHeight;
    const left = Math.max(12,Math.min(window.innerWidth - width - 12,anchor.x + width + 24 > window.innerWidth ? anchor.x - width - 12 : anchor.x + 12));
    const top = Math.max(12,Math.min(window.innerHeight - height - 12,anchor.y - 35));
    host.style.left = `${left}px`; host.style.top = `${top}px`;
  }
  function drawChipProfile() {
    const overlay = $('chip-profile-overlay');
    overlay.replaceChildren();
    if (!chart || !state.analysis || !$('layer-chips').checked || $('view-analysis').hidden) return;
    const profile = selectedChipProfile();
    const width = $('chip-profile').clientWidth, height = $('analysis-chart').clientHeight;
    if (!width || !height) return;
    const c = colors(), plotHeight = height - chart.timeScale().height();
    overlay.setAttribute('viewBox',`0 0 ${width} ${height}`);
    const node = (tag, attributes, text = null) => { const item = document.createElementNS('http://www.w3.org/2000/svg',tag); Object.entries(attributes).forEach(([key,value]) => item.setAttribute(key,String(value))); if (text != null) item.textContent = text; overlay.append(item); return item; };
    const caption = `右侧筹码量价估算 · ${profile?.window_bars || 120}日 · 截至 ${profile?.as_of || '—'}${chartSelection.pinned ? ' · 已固定' : ''} · 非真实持仓筹码`;
    $('chip-profile-caption').textContent = caption;
    if (profile?.status !== 'available') { node('text',{x:8,y:25,fill:c.muted,'font-size':10},profile?.status === 'no_volume' ? '窗口内无有效成交量' : profile?.status === 'insufficient_history' ? `历史不足 ${profile.window_bars} 根` : '暂无分布数据'); return; }
    const bins = profile.bins || [], maximum = Math.max(...bins.map((bin) => bin.mass_fraction),0);
    bins.forEach((bin) => {
      if (!Number.isFinite(bin.mass_fraction) || bin.mass_fraction <= 0 || !maximum) return;
      const high = candles.priceToCoordinate(bin.price_high), low = candles.priceToCoordinate(bin.price_low);
      if (![high,low].every((value) => value != null && Number.isFinite(value))) return;
      const y = Math.max(0,Math.min(high,low)), bottom = Math.min(plotHeight,Math.max(high,low));
      if (bottom <= y) return;
      const in70 = bin.price_high >= profile.chip70_low && bin.price_low <= profile.chip70_high;
      const length = (width - 16) * bin.mass_fraction / maximum;
      const rect = node('rect',{x:width - 8 - length,y,width:length,height:Math.max(1,bottom - y - .5),fill:in70 ? c.chips : c.muted,'fill-opacity':in70 ? .8 : .35});
      const title = document.createElementNS('http://www.w3.org/2000/svg','title'); title.textContent = `${number(bin.price_low,2)}–${number(bin.price_high,2)} 元 · 窗口成交量占比 ${percent(bin.mass_fraction)}`; rect.append(title);
      rect.setAttribute('tabindex','0');
      rect.setAttribute('aria-label',title.textContent);
      const showBin = () => { rect.setAttribute('fill-opacity','1'); rect.setAttribute('stroke',c.accent); $('chip-profile-caption').textContent = `${caption} · ${title.textContent}`; };
      const hideBin = () => { rect.setAttribute('fill-opacity',in70 ? '.8' : '.35'); rect.setAttribute('stroke','none'); $('chip-profile-caption').textContent = caption; };
      rect.addEventListener('pointerenter',showBin); rect.addEventListener('focus',showBin); rect.addEventListener('pointerleave',hideBin); rect.addEventListener('blur',hideBin);
    });
    const middle = candles.priceToCoordinate(profile.chip50);
    if (middle != null && middle >= 0 && middle <= plotHeight) node('line',{x1:6,x2:width - 6,y1:middle,y2:middle,stroke:c.chips,'stroke-dasharray':'3 3'});
    node('text',{x:8,y:plotHeight + 16,fill:c.chips,'font-size':9},`70% ${number(profile.chip70_low,2)}–${number(profile.chip70_high,2)}`);
  }
  function syncChartRange() {
    const bars = state.analysis?.frequencies?.[state.frequency]?.bars || [];
    const available = !!chart && bars.length > 0;
    ['chart-range-start','chart-range-end','chart-range-apply','chart-zoom-in','chart-zoom-out','chart-range-reset'].forEach((id) => { $(id).disabled = !available; });
    if (!available) { $('chart-range-start').value = ''; $('chart-range-end').value = ''; $('chart-range-summary').textContent = ''; return; }
    const range = chart.timeScale().getVisibleLogicalRange();
    if (!range) return;
    const from = Math.max(0,Math.min(bars.length - 1,Math.ceil(range.from))), to = Math.max(from,Math.min(bars.length - 1,Math.floor(range.to)));
    ['chart-range-start','chart-range-end'].forEach((id) => { $(id).min = day(bars[0].time); $(id).max = day(bars.at(-1).time); });
    $('chart-range-start').value = day(bars[from].time); $('chart-range-end').value = day(bars[to].time);
    $('chart-range-summary').textContent = `${state.frequency} · 显示 ${to - from + 1} 根 · 判断截至 ${state.analysis.data_end || state.query?.as_of || '—'} · 筹码截至 ${selectedChipProfile()?.as_of || '—'} · 滚轮缩放 / 拖动平移`;
  }
  function applyChartRange() {
    const bars = state.analysis?.frequencies?.[state.frequency]?.bars || [];
    if (!chart || !bars.length) return;
    const start = $('chart-range-start').value, end = $('chart-range-end').value;
    if (!start || !end || start > end) { notice('显示区间开始日期不能晚于结束日期。'); return; }
    const from = bars.findIndex((bar) => day(bar.time) >= start), to = bars.findLastIndex((bar) => day(bar.time) <= end);
    if (from < 0 || to < from) { notice('所选区间没有可显示的K线。'); return; }
    chart.timeScale().setVisibleLogicalRange({from:from - .5,to:to + .5});
  }
  function zoomChart(factor) {
    if (!chart) return;
    const bars = state.analysis?.frequencies?.[state.frequency]?.bars || [], range = chart.timeScale().getVisibleLogicalRange();
    if (!bars.length || !range) return;
    const left = Math.max(-.5,range.from), right = Math.min(bars.length - .5,range.to), center = (left + right) / 2;
    const span = Math.min(bars.length,Math.max(Math.min(5,bars.length),(right - left) * factor));
    const from = Math.max(-.5,Math.min(bars.length - span - .5,center - span / 2));
    chart.timeScale().setVisibleLogicalRange({from,to:from + span});
  }
  function drawZones() {
    const overlay = $('zone-overlay');
    overlay.replaceChildren();
    if (!chart || !state.analysis || !$('layer-zones').checked || $('view-analysis').hidden) return;
    const c = colors();
    const width = $('analysis-chart').clientWidth - chart.priceScale('right').width();
    const height = $('analysis-chart').clientHeight - chart.timeScale().height();
    overlay.setAttribute('viewBox', `0 0 ${$('analysis-chart').clientWidth} ${$('analysis-chart').clientHeight}`);
    const group = document.createElementNS('http://www.w3.org/2000/svg','g');
    (state.analysis.frequencies?.[state.frequency]?.zones || []).forEach((zone) => {
      const bars = state.analysis.frequencies?.[state.frequency]?.bars || [];
      if (!bars.length) return;
      const first = day(bars[0].time), last = day(bars[bars.length - 1].time);
      const start = day(zone.start_time), end = day(zone.end_time);
      if (end < first || start > last) return;
      const x1 = chart.timeScale().timeToCoordinate(start < first ? first : start);
      const x2 = chart.timeScale().timeToCoordinate(end > last ? last : end);
      const y1 = candles.priceToCoordinate(Number(zone.high));
      const y2 = candles.priceToCoordinate(Number(zone.low));
      if ([x1,x2,y1,y2].some((v) => v == null || !Number.isFinite(v))) return;
      const x = Math.max(0, Math.min(x1,x2)), y = Math.max(0, Math.min(y1,y2));
      const w = Math.min(width,Math.max(x1,x2)) - x, h = Math.min(height,Math.max(y1,y2)) - y;
      if (w <= 0 || h <= 0) return;
      const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
      const color = zone.direction === 'up' ? c.up : zone.direction === 'down' ? c.down : c.muted;
      Object.entries({x,y,width:w,height:h,fill:color,'fill-opacity':.09,stroke:color,'stroke-opacity':.65,'stroke-width':1,'stroke-dasharray':zone.is_confirmed ? 'none' : '5 4'}).forEach(([key,value]) => rect.setAttribute(key,String(value)));
      group.append(rect);
    });
    overlay.append(group);
  }
  function actionLabel(value) { return ({BUY:'买入',SELL:'卖出',HOLD:'持有',WAIT:'等待',REDUCE:'减仓',buy:'买入',sell:'卖出',hold:'持有',watch:'观察',abstain:'无操作',open:'开仓',close:'平仓'})[value] || String(value || '无操作'); }
  function actionClass(value) { return /buy|open|买|开/i.test(value || '') ? 'buy' : /sell|reduce|close|卖|平|减/i.test(value || '') ? 'sell' : ''; }
  function actionShape(value, execution = false) { return actionClass(value) === 'buy' ? execution ? '● ' : '▲ ' : actionClass(value) === 'sell' ? execution ? '■ ' : '▼ ' : ''; }
  function readableReason(event) {
    if (event.category && event.reason_codes?.length) {
      const labels = {missing_target_date:'目标日行情缺失',unsupported_board:'当前成交模型不支持此板块',model_shadow_gate_not_passed:'模型尚未证实改善，保持观察',unconfirmed_point:'尚无已确认辅助买卖点',combined_rule_not_met:'结构、量价或周线条件未同时满足',model_probability_unavailable:'模型概率不可用',model_probability_below_threshold:'模型概率未达冻结门槛',calendar_missing_market_bar_window:'近期交易日记录不完整',quality_warmup:'可靠历史不足',unverified_source_current_bar:'当日来源未经核验',unverified_source_window:'近期来源未经核验',unverified_source_structural:'结构依赖来源未经核验',structural_input_unverified:'结构输入尚未通过核验',suspended_current_bar:'当日无成交量与金额',source_missing_window:'近期来源缺失',trade_price_unverified_window:'近期成交价格未核验'};
      return [...new Set(event.reason_codes.map((code) => labels[code] || '数据或成交条件未通过核验'))].join('；');
    }
    const reason = display(event.reason);
    if (/止损/.test(reason)) return '止损条件触发';
    if (/超时|超期|timeout/i.test(reason)) return '持仓期限条件触发';
    if (reason !== '—' && !/_|[LS][OE]#/.test(reason)) return reason;
    return actionClass(event.action) === 'buy' ? '已确认结构满足开仓规则' : actionClass(event.action) === 'sell' ? '退出条件触发' : '未触发交易规则';
  }
  function readableEvent(event) {
    if (event.source && event.event_type) {
      const source = {ma:'均线ML',structure:'CZSC结构',wyckoff:'威科夫量价'}[event.source] || event.source;
      const kind = {ma_trend_recovery:'趋势恢复',ma_trend_onset:'趋势启动',ma_trend_failure:'趋势失效',ma_independent_exit_evidence:'独立退出证据',pure_first_buy_buy:'纯结构一买',pure_first_sell_sell:'纯结构一卖',pure_second_bs_buy:'纯结构二买',pure_second_bs_sell:'纯结构二卖',pure_third_buy_buy:'纯结构三买',structure_invalidation:'已确认结构支撑失效'}[event.event_type] || event.event_type.replace(/^native_bs[123]_/, '');
      return `${source} · ${kind}`;
    }
    if (event.position_intent) return ({BUY:'组合策略开仓意图',SELL:'组合策略退出意图',HOLD:'理论持仓继续持有',WAIT:'理论空仓等待'})[event.position_intent] || actionLabel(event.position_intent);
    if (actionClass(event.action) === 'sell') return /止损|超时|超期|timeout/i.test(display(event.reason)) ? readableReason(event) : '退出条件触发';
    const signal = String(event.signal || '');
    const daily = signal.match(/日线_确认笔_方向V1=(向上|向下|其他)/);
    const weekly = signal.match(/周线_闭合价格_方向V1=(向上|向下|其他)/);
    const labels = [];
    if (daily) labels.push(`确认笔${daily[1] === '其他' ? '暂无方向' : daily[1]}`);
    if (weekly) labels.push(`闭合周价格${weekly[1] === '其他' ? '暂无方向' : weekly[1]}`);
    if (labels.length) return labels.join(' · ');
    if (signal && !/_|[LS][OE]#/.test(signal)) return signal;
    return actionClass(event.action) === 'buy' ? '开仓条件触发' : '无确认交易事件';
  }
  function ruleOriginal(event) { return `<details class="rule-original"><summary>规则原文</summary><p>信号<code>${esc(display(event.signal))}</code></p><p>依据<code>${esc(display(event.reason))}</code></p></details>`; }
  function referencePriceText(value, date, label) {
    if (value == null || value === '' || !Number.isFinite(Number(value)) || Number(value) <= 0) return `${label}暂不可用`;
    return `${label} ${number(value,2)} 元（${date ? `${day(date)} 日` : '信号日'}收盘，非成交价）`;
  }
  function isHighReturn(...records) { return records.some((record) => record?.model_family === 'current' || record?.research?.model_family === 'current' || record?.model_resolution?.model_family === 'current' || record?.strategy_version === 'trend_consensus'); }
  function isTrendVersion(value) { return value === 'trend_consensus'; }
  function trendVersion(...records) { return records.flatMap(record => [record?.strategy_version,record?.schema_family,record?.version,record?.consensus_version,record?.model_resolution?.schema_family,record?.trend_consensus?.version]).find(isTrendVersion); }
  function isTrendStrategy(...records) { return Boolean(trendVersion(...records)) || isHighReturn(...records); }
  function selectedTrendVersion() { return 'trend_consensus'; }
  function trendInputSelected(prefix) { return $(`${prefix}-mode`).value === 'research' && $(`${prefix}-model-family`).value === 'current'; }
  function modelName() { return '趋势共识完整模型包'; }
  function modelInputUnavailable(resolution, row = {}) {
    if (resolution?.status === 'input_veto' || resolution?.reason_codes?.includes('model_input_ineligible')) return true;
    const explicit = resolution?.model_input_eligible ?? row.model_input_eligible;
    if (explicit != null) return explicit === false;
    return resolution?.input_eligible === false || row.input_eligible === false;
  }
  function modelProbability(resolution, row = {}) {
    if (modelInputUnavailable(resolution,row) || ['rules_no_model','no_model','unavailable'].includes(resolution?.status)
      || resolution?.reason_codes?.some((code) => ['model_unavailable','pinned_model_unavailable'].includes(code))) return null;
    const value = resolution && Object.prototype.hasOwnProperty.call(resolution,'probability') ? resolution.probability
      : Object.prototype.hasOwnProperty.call(row,'model_probability') ? row.model_probability : row.probability;
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;
  }
  function modelEntryApplied(resolution, row = {}) {
    const policy = row.entry_policy || row.entry_plan?.policy || resolution?.entry_policy;
    return !['fresh','risk'].includes(policy) && row.eligible !== false && row.ml_filter_applied !== false && !modelInputUnavailable(resolution,row)
      && !['entry_contract_shadow','historical_shadow','production_shadow','retrospective_shadow','rule_input_veto_shadow'].includes(resolution?.status)
      && resolution?.applied_to_entry === true && modelProbability(resolution,row) != null;
  }
  function ruleQualificationText(row) {
    if (isTrendStrategy(row) || row.agents?.ma?.kind === 'trend_probability') {
      const status = row.agents?.high_return?.status || row.agents?.high_return?.evaluation_status;
      const facts = row.signal_points || (Array.isArray(row.candidate_evidence) ? row.candidate_evidence : row.candidate_evidence?.facts) || [];
      const validSell = facts.some((event) => String(event.side || event.action).toUpperCase() === 'SELL' && event.direction_eligible === true);
      if (['input_ineligible','feature_ineligible'].includes(status)) return '候选资格：输入不足';
      if (['candidate_risk_ineligible','risk_blocked','risk_blocked_unscored'].includes(status)) return '候选资格：买点触发，冻结风险规则阻止入场；高收益未评估';
      if (status === 'no_candidate') return validSell ? '候选资格：独立卖出证据触发，本日无新买入候选' : '候选资格：本日无新候选';
      if (row.source_candidate_eligible === true && row.risk_eligible === true) return '候选资格：三路独立买点并集触发，冻结风险资格通过';
      return validSell ? '候选资格：独立卖出证据触发，最终方向另行仲裁' : '候选资格：尚无合格的新买入候选';
    }
    if (isHighReturn(row)) return row.input_eligible === false || row.eligible === false ? '候选资格：输入不足' : row.candidate_action === 'BUY' ? row.risk_eligible === false ? '候选资格：买点触发，冻结风险规则未通过' : row.source_selected_by_reference === false ? '候选资格：原始买点仅作影子评分，第一阶段未选中' : '候选资格：三路独立买点并集触发' : row.candidate_action === 'SELL' ? '候选资格：独立卖出证据触发' : '候选资格：本日未触发新买点';
    if (row.eligible === false || row.input_eligible === false || row.model_resolution?.status === 'rule_input_veto_shadow' || row.entry_plan?.reason_codes?.includes('input_ineligible')) return '规则资格：输入不可用';
    if (row.rule_buy === true || row.raw_rule_buy === true || row.candidate_action === 'BUY') return '规则资格：满足买入组合条件';
    if (row.rule_sell === true || row.candidate_action === 'SELL') return '规则资格：退出组合条件触发';
    if (row.rule_buy === false || row.candidate_action === 'WAIT') return '规则资格：买入组合条件未满足';
    return '规则资格：本次记录未保存';
  }
  function modelInputReasonText(resolution, row = {}) {
    const labels = {calendar_missing_market_bar_window:'近期行情与核验交易日历不一致或存在缺口',wyckoff_calendar_gap_window:'近期行情与核验交易日历不一致或存在缺口',structural_input_ineligible:'结构输入未通过核验',structural_input_unverified:'结构依赖行情未通过核验',insufficient_confirmed_structure:'已确认结构不足',structure_not_observable:'结构确认时间尚不可见',future_structure_metadata:'结构记录含信号时点之后的信息',quality_warmup:'可靠历史不足',wyckoff_quality_warmup:'可靠历史不足',insufficient_active_history:'有效成交历史不足',wyckoff_insufficient_active_history:'有效成交历史不足',ma_history_unavailable:'均线输入历史不足',volume_baseline_unavailable:'成交量比较基准不足',source_missing_window:'近期行情来源缺失',unverified_source_window:'近期行情来源未经核验',wyckoff_unverified_source_window:'近期行情来源未经核验',unverified_source_current_bar:'当日行情来源未经核验',unverified_source_structural:'结构依赖行情来源未经核验',legacy_fuyao_window:'近期窗口含已退役行情来源',trade_price_unverified_window:'近期成交价格未经核验',wyckoff_trade_price_unverified_window:'近期成交价格未经核验',missing_values_window:'近期成交量或成交金额缺失',negative_volume_amount_window:'近期成交量或成交金额为负',volume_amount_mismatch_window:'近期成交量与成交金额不一致',wyckoff_invalid_quantity_window:'近期成交量或成交金额无效',wyckoff_invalid_price_window:'近期行情价格无效',wyckoff_volume_amount_unit_disagreement_window:'成交量与金额单位核验不一致',wyckoff_suspected_corporate_action_window:'近期价格跳变疑似公司行动，尚未核验',wyckoff_units_not_verified:'成交量与金额单位未经核验',wyckoff_price_basis_not_unadjusted:'量价输入不是已核验未复权价格',suspended_current_bar:'当日无有效成交',wyckoff_suspended_current_bar:'当日无有效成交',unsupported_board:'当前模型仅支持沪深主板'};
    const codes = [...(resolution?.model_input_reason_codes || []),...(row.model_input_reason_codes || [])];
    return [...new Set(codes.map((code) => labels[code] || code))].join('；');
  }
  function modelAvailabilityText(resolution) {
    const diagnostics = resolution.model_availability_diagnostics;
    if (!diagnostics) return '';
    const time = (value) => {
      const parsed = new Date(value);
      return Number.isNaN(parsed.getTime()) ? String(value) : `${new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(parsed)}（北京时间）`;
    };
    const candidateText = (record, label) => {
      const candidate = record?.candidates?.at(-1);
      if (!candidate) return '';
      const evidence = [];
      if (candidate.actual_training_completed_at) evidence.push(`实际训练完成 ${time(candidate.actual_training_completed_at)}`);
      if (candidate.reason_codes?.includes('model_validation_after_signal') && candidate.available_at) evidence.push(`验证数据可用 ${time(candidate.available_at)}`);
      return evidence.length ? `${label}${candidate.checkpoint ? ` ${candidate.checkpoint}` : ''}：${evidence.join('；')}。` : '';
    };
    return `${diagnostics.signal_at ? `信号时点 ${time(diagnostics.signal_at)}。` : ''}${candidateText(diagnostics,'最近模型候选')}${candidateText(diagnostics.dual_fallback,'双专家回退候选')}`;
  }
  function modelResolutionText(resolution, row = {}) {
    if (!resolution) return '该结果未保存当前模型作用记录。';
    if (!isTrendStrategy(resolution,row)) return '历史模型协议已退役，不具备当前判断资格。';
    return highReturnResolutionText(resolution,row);
  }
  function highReturnResolutionText(resolution, row = {}) {
    const trendAccount = isTrendStrategy(resolution,row) || evidenceUI()?.trendProbabilities(row.agents?.ma).present;
    const probability = modelProbability(resolution,row);
    const selectedRun = row.account_context?.model_identity?.model_run_id;
    const identity = resolution.model_run_id ? `${resolution.model_run_id} / ${resolution.checkpoint || '—'}` : trendAccount ? selectedRun ? `所选研究运行 ${selectedRun}` : '当日未使用高收益评分头' : '无兼容检查点';
    const codes = [...new Set([...(resolution.reason_codes || []),resolution.reason].filter(Boolean))];
    const reasons = trendAccount ? evidenceUI()?.opinionReasons({reason_codes:codes}) || codes.join('；') : codes.join('；');
    const target = row.probability_target || resolution.probability_target || '按冻结退出政策形成费用后高收益区间';
    const exit = trendAccount ? row.account_action?.exit_evaluation || {} : row.exit_model || {};
    const holding = row.holding || row.holding_snapshot || row.account_action?.holding_snapshot;
    const holdingShares = holding?.shares ?? row.account_action?.shares;
    const compatible = (holdingShares > 0) && exit.holding_contract_compatible === true;
    const exitProbability = compatible && typeof exit.probability === 'number' && Number.isFinite(exit.probability) ? exit.probability : null;
    if (trendAccount) {
      const view = evidenceUI()?.currentAgentView(row,'high_return'), status = view?.evaluationStatus;
      const statusText = status === 'no_candidate' || !status && codes.includes('no_qualified_direction_candidate') ? '本日无合格买入候选，高收益本日未评分' : view?.summary || '当日未提供高收益评估状态';
      const exitText = exitProbability != null ? `持仓退出概率 ${percent(exitProbability)}` : exit.reason ? evidenceUI()?.opinionReasons({reason:exit.reason}) || exit.reason : holdingShares > 0 ? '该研究账户此日未生成兼容退出评分' : holdingShares === 0 ? '该研究账户空仓，退出不适用' : '该研究账户持仓状态未提供';
      return `${modelName(resolution,row)} · ${identity}。按所选研究Broker账户状态评估，尚未取得生产资格。${statusText}。持仓退出评分：${exitText}。${reasons ? `原因：${reasons}。` : ''}`;
    }
    return `${modelName(resolution,row)} · ${identity}。${resolution.partial_scope ? `部分股票诊断模型（${number(resolution.selected_count)}只），未作为全市场自动模型。` : ''}${trendAccount ? '按所选研究Broker账户状态评估，尚未取得生产资格。' : resolution.applied_to_entry ? '已在显式离线实验中参与入场筛选，尚未取得生产资格。' : '影子观察，未改变实际交易策略。'}${probability == null ? '高收益区间概率不可用。' : `高收益区间研究概率 ${percent(probability)}；目标：${target}。`}${trendAccount ? '持仓退出评分' : '退出优势头'}：${exitProbability == null ? exit.reason || (trendAccount ? holdingShares > 0 ? '该研究账户此日未生成兼容退出评分' : holdingShares === 0 ? '该研究账户空仓，退出不适用' : '该研究账户持仓状态未提供'  : '缺少兼容真实持仓状态，未生成退出概率') : `下一核验开盘退出优于冻结续持政策的概率 ${percent(exitProbability)}`}。${resolution.available_at ? `数据可用 ${resolution.available_at}。` : ''}${resolution.actual_training_completed_at ? `实际训练完成 ${resolution.actual_training_completed_at}。` : ''}${reasons ? `原因：${reasons}。` : ''}`;
  }
  function highReturnEvidence(row) {
    const evidence = row.high_return_evidence || {}, points = row.signal_points || evidence.signal_points || evidence.events || [], sources = row.candidate_sources || row.sources || evidence.sources || [];
    const names = {ma:'均线ML',ma_ml:'均线ML',structure:'CZSC结构',wyckoff:'威科夫量价'};
    const sourceText = Array.isArray(sources) ? sources.map((item) => names[item] || item).join('、') : String(sources || '');
    const events = Array.isArray(points) ? points.map((point) => `<p>${esc(names[point.source] || point.source || '')} · ${esc(point.event_type || point.signal || '')} · ${esc(point.side || point.action || '')}<br>锚点 ${esc(point.anchor_at || '—')} · 确认可用 ${esc(point.available_at || point.signal_at || '—')}</p>`).join('') : '';
    const trendAccount = isTrendStrategy(row) || evidenceUI()?.trendProbabilities(row.agents?.ma).present;
    const exitNote = '退出模型只使用截至当日的兼容真实持仓状态；MA退出头在入场时固定。';
    const pricingNote = trendAccount ? `<p>高收益目标：${esc(evidenceUI()?.highReturnTargetText(row.agents?.high_return?.label_target))}。成交采用滑点模拟均价；R按冻结止损参考假设估计。</p>` : '';
    return `<section class="research-evidence high-return-evidence" aria-label="高收益区间研究"><strong>三路独立买卖点与第二阶段判断</strong><p>候选来源：${esc(sourceText || '本日无新候选')}。同股同日候选保留各路证据。</p><p>${esc(highReturnResolutionText(row.model_resolution || {model_family:'current'},row))}</p>${events}${typeof row.expected_net_return === 'number' ? `<p>辅助预期费用后收益 ${percent(row.expected_net_return)} · 亏损研究概率 ${percent(row.loss_probability)}。</p>` : ''}<p>${trendAccount ? '入场质量使用所选研究Broker账户当日真实空仓资金状态；持仓退出使用该账户已发生的成交和持仓前缀。' : '新入场评分使用冻结初始资金的假设空仓账户；具体成交与资金状态在Broker回放中核验。'}</p>${pricingNote}<p>入场模型只判断已产生的候选。完整区间标签由实际成交、费用与冻结退出政策形成；高收益标签不代表未来最高涨幅。${esc(exitNote)}</p></section>`;
  }
  function entryPlanHtml(row) {
    if (isHighReturn(row)) return '';
    const plan = row?.entry_plan, policy = plan?.policy || row?.entry_policy || 'legacy';
    if (!plan) return `<div class="entry-plan"><strong>${esc(entryPolicyNames[policy] || policy)}</strong><p>${policy === 'legacy' ? '该运行未保存独立入场计划，按原目标仓位对照恢复。' : '服务未返回入场计划，请重新运行后核查。'}</p></div>`;
    const codes = plan.reason_codes || [], reasons = entryPlanReasons(codes);
    const label = plan.status === 'holding' ? '理论持仓 · 不新增买入' : plan.status === 'exit' ? '退出计划' : codes.includes('point_age_exceeds_experimental_limit') ? '已过期（实验期限）' : codes.includes('buy_point_already_consumed') ? '已使用 · 等待新买点' : plan.status === 'active' && plan.entry_allowed ? '待下一开盘核验' : codes.includes('rule_buy_not_met') ? '待触发' : plan.status === 'rejected' ? '入场未通过' : '无可执行计划';
    const validPrice = (value) => value != null && value !== '' && Number.isFinite(Number(value)) && Number(value) > 0;
    const range = validPrice(plan.price_floor) && validPrice(plan.price_ceiling) ? `${number(plan.price_floor,2)}～${number(plan.price_ceiling,2)} 元` : policy === 'risk' ? '未形成可执行区间' : '该规则未设价格区间';
    const stop = validPrice(plan.stop_price) ? `${number(plan.stop_price,2)} 元` : policy === 'risk' ? '未形成有效失效价' : '该规则未设结构失效价';
    const risk = plan.risk_fraction == null ? '未设置' : percent(plan.risk_fraction), weight = plan.target_weight == null ? '未设置' : percent(plan.target_weight);
    const diagnostics = plan.diagnostics || {}, point = plan.identity || {};
    const frozen = ['holding','exit'].includes(plan.status), ageLabel = frozen ? '入场时买点龄' : '当前买点确认龄';
    const direction = (value) => Number(value) === 1 ? '向上' : Number(value) === -1 ? '向下' : '暂无方向';
    const weekly = diagnostics.weekly_price_direction != null || diagnostics.weekly_bi_direction != null ? `<p>${frozen ? '入场时' : ''}闭合周价格：${esc(direction(diagnostics.weekly_price_direction))} · 已确认周笔：${esc(direction(diagnostics.weekly_bi_direction))}</p>` : '';
    return `<section class="entry-plan" aria-label="入场研究计划"><div class="entry-plan-heading"><strong>${esc(entryPolicyNames[policy] || policy)}</strong><span class="badge">${esc(label)}</span></div><p class="entry-plan-note">${esc(entryPolicyCaption(policy))}</p><dl><div><dt>参考入场区间</dt><dd>${esc(range)}</dd></div><div><dt>结构失效价</dt><dd>${esc(stop)}</dd></div><div><dt>账户风险预算</dt><dd>${esc(risk)}</dd></div><div><dt>建议目标仓位</dt><dd>${esc(weight)}</dd></div></dl>${point.point_type ? `<p>${esc(point.point_type)} · 确认 ${esc(point.point_confirmed_at || '尚未确认')}${diagnostics.point_age_bars != null ? ` · ${ageLabel} ${number(diagnostics.point_age_bars)} 根股票K线` : ''}</p>` : ''}${weekly}${reasons ? `<p class="entry-plan-reasons">${esc(reasons)}</p>` : ''}<p>${esc(referencePriceText(plan.reference_price,plan.signal_date,'信号收盘参考'))}${plan.status === 'active' && plan.entry_allowed && plan.valid_for_sessions != null ? ` · 入场计划仅对下一 ${number(plan.valid_for_sessions)} 个核验交易日有效` : ''}</p><p class="entry-plan-note">理论研究计划独立于个人持仓；实际价格、成交股数与拒单以账本为准。</p></section>`;
  }
  function entryPlanReasons(codes = []) { return codes.map((code) => entryReasonNames[code] || code).join('；'); }
  const entryReasonNames = {pre_start_warmup:'账户开始日前仅预热，不执行入场',input_ineligible:'行情输入不满足研究要求',opposing_exit_signal:'退出信号优先，当前不入场',rule_buy_not_met:'入场规则尚未满足',ml_entry_veto:'具备资格的模型过滤未通过',buy_point_identity_unavailable:'缺少已确认辅助买点身份',buy_point_already_consumed:'同一买点计划已使用，等待新确认结构',already_positioned:'理论已经持仓，不新增买入',risk_inputs_unavailable:'价格与风险实验所需输入不足',point_age_exceeds_experimental_limit:'确认买点年龄超过实验期限；该期限尚未认证',ma20_distance_exceeds_experimental_limit:'价格距离MA20超过实验范围；该范围尚未认证',invalid_structure_stop:'无法形成有效结构失效价',structural_stop_triggered:'已跌破冻结结构失效价',position_exit:'原生 Position 触发退出'};
  function nextSessionGuidance(decision, requestedAsOf) {
    const labels = {BUY:'买入',SELL:'卖出',HOLD:'持有',WAIT:'等待'};
    if (!decision || !labels[decision.action]) return '下一交易日判断暂不可用，请重新分析。';
    const cutoff = day(requestedAsOf), referenceDate = day(decision.reference_price_date), nextSession = day(decision.next_market_session);
    const historical = cutoff && referenceDate && nextSession && referenceDate < cutoff && nextSession <= cutoff;
    const warning = historical ? `当前判断暂不可用：行情仅截至 ${referenceDate}，未覆盖请求截止 ${cutoff}；下列为历史规则判断。` : '';
    const session = historical ? `历史规则判断（信号日 ${referenceDate}，对应交易日 ${nextSession}）` : nextSession ? `下一交易日 ${nextSession}` : '下一交易日（日期待核验）';
    const priceLabel = decision.action === 'BUY' ? '买入参考价' : decision.action === 'SELL' ? '卖出参考价' : '收盘参考价';
    const trendAccount = isTrendStrategy(decision);
    const shares = (decision.holding || decision.holding_snapshot || decision.account_action?.holding_snapshot)?.shares ?? decision.account_action?.shares;
    const label = trendAccount && ['HOLD','WAIT'].includes(decision.action) ? shares === 0 ? '空仓观望' : shares > 0 ? '持有观察' : '仓位未提供 · 观察' : labels[decision.action];
    const basis = decision.basis || (trendAccount ? evidenceUI()?.opinionReasons(decision.account_action) || evidenceUI()?.opinionReasons(decision) : '') || '条件依据暂未返回。';
    const mode = decision.usage_mode || state.query?.usage_mode;
    const model = decision.model_resolution;
    const scope = trendAccount ? evidenceUI()?.accountContextText(decision.account_context) : decision.position_scope === 'actual_broker_state_required' ? '退出判断需要兼容的真实Broker持仓历史' : decision.position_scope === 'theoretical_prefix_position' ? '组合策略理论持仓，与手动持仓独立' : decision.position_scope === 'flat_at_start' ? `从研究开始日 ${day(decision.position_start) || '所选开始日'} 初始化为空仓，之后连续模拟；与手动持仓独立` : decision.position_scope || decision.scope;
    return `${warning ? `${esc(warning)}<br>` : ''}${mode ? `<span class="decision-mode">${esc(usageNames[mode] || '旧用途已退役，不恢复资格')}${mode === 'retrospective' ? ' · 不能作为历史可执行或认证成绩' : ''}</span><br>` : ''}<strong>${esc(session)} · <span class="badge ${actionClass(decision.action)}">${esc(label)}</span></strong><br>${esc(referencePriceText(decision.reference_close,decision.reference_price_date,priceLabel))}<br>条件：${esc(basis)}${scope ? `<br>仓位语境：${esc(display(scope))}` : ''}${model ? `<br>${esc(ruleQualificationText(decision))}<br><span class="model-resolution">${esc(modelResolutionText(model,decision))}</span>` : ''}`;
  }
  function renderEvents() {
    const events = state.analysis?.events || [];
    $('event-count').textContent = String(events.length);
    $('event-context').innerHTML = nextSessionGuidance(state.analysis?.next_session_decision,state.query?.as_of || state.analysis?.as_of);
    const research = state.analysis?.research;
    $('analysis-research-evidence').hidden = !research;
    $('analysis-research-evidence').innerHTML = research ? researchEvidence({...research,...state.analysis.next_session_decision}) : '';
    if (state.analysis?.compute_info) $('analysis-compute-status').textContent = research ? researchComputeCaption(state.analysis.compute_info,state.analysis) : computeCaption(state.analysis.compute_info);
    $('event-list').innerHTML = events.length ? [...events].reverse().map((event) => `<article class="event-entry" data-event-time="${esc(day(event.time))}" data-evidence-date="${esc(day(event.evidence_date || event.time))}"><strong>${esc(readableEvent(event))}</strong><p><span class="badge ${actionClass(event.action)}">${actionShape(event.action)}${esc(actionLabel(event.action))}意图</span></p><p class="event-reference">${esc(referencePriceText(event.reference_price,event.reference_price_date,`${actionLabel(event.action)}参考价`))}</p>${event.entry_plan ? entryPlanHtml(event) : ''}<p>${esc(readableReason(event))}</p><p class="event-time">发生 ${esc(display(event.time))}<br>可用 ${esc(display(event.available_at))}</p><button type="button" data-locate-event="${esc(day(event.evidence_date || event.time))}">定位主图</button>${ruleOriginal(event)}</article>`).join('') : '<div class="empty-copy">该时点没有规则事件。形成中的结构不代表确认信号。</div>';
  }
  function highlightEvent(time) { const dates = new Set(Array.isArray(time) ? time : [time]); document.querySelectorAll('[data-event-time]').forEach((entry) => { entry.style.background = (dates.has(entry.dataset.eventTime) || dates.has(entry.dataset.evidenceDate)) ? 'var(--accent-soft)' : ''; }); }
  function locateTime(time, openPopup = true) {
    if (!chart) return;
    const bars = state.analysis?.frequencies?.[state.frequency]?.bars || [];
    const index = bars.findIndex((bar) => KHQuantMarkers.periodKey(bar.time,state.frequency) === KHQuantMarkers.periodKey(time,state.frequency));
    if (index >= 0) { chart.timeScale().setVisibleLogicalRange({from:Math.max(0,index - 35),to:index + 5}); updateChartSelection(openPopup ? 'pin' : 'locate',day(bars[index].time),null,time); }
    highlightEvent(time);
  }
  function getSymbol(value) { const text = value.trim(); const known = [...state.symbolCatalog.values()].find((item) => item.symbol === text.toUpperCase() || item.symbol.slice(0,6) === text || item.name === text); if (known) return known.symbol; const match = text.match(/\d{6}(?:\.(?:SH|SZ|BJ))?/i); return match ? match[0].toUpperCase() : text; }
  function symbols(value) { return [...new Set(value.split(/[\s,，;；]+/).map(getSymbol).filter(Boolean))]; }
  function dateSpec(prefix) { const spec = {}; ['start','end'].forEach((key) => { if ($(`${prefix}-${key}`).value) spec[key] = $(`${prefix}-${key}`).value; }); if (spec.start && spec.end && spec.start > spec.end) throw new Error('开始日期不能晚于结束日期。'); return spec; }
  function consensusRequestContext(result) {
    const consensus = result?.trend_consensus || result?.consensus_decisions;
    const version = consensus?.version || consensus?.config?.version || result?.request?.consensus_version;
    return {...(consensus?.config ? {consensus_config:consensus.config} : {}),...(version ? {consensus_version:version} : {})};
  }
  async function analyze(query, preserveDates = false, locate = null) {
    const sequence = ++state.analysisSequence;
    analysisLoading = true; resetChartSelection();
    emptyChart('正在按所选时点分析','参考价与评分加载中…');
    $('event-context').textContent = '正在按所选时点加载判断…'; $('event-list').innerHTML = '<div class="empty-copy">事件加载中…</div>';
    $('analysis-research-evidence').hidden = true; $('analysis-research-evidence').replaceChildren();
    if (!preserveDates) { historyRows = []; historyPage = 0; }
    renderReplayEvidence(); updateReplay();
    $('analysis-state').textContent = '正在分析…';
    $('analysis-form').querySelector('[type=submit]').disabled = true;
    try {
      const data = await request('/api/analysis', {method:'POST',body:JSON.stringify(query)});
      if (sequence !== state.analysisSequence) return;
      state.analysis = data; state.query = {...query,...data.request,...consensusRequestContext(data)}; analysisLoading = false;
      const incoming = evidenceUI()?.mergeEvidenceRows(displayedEvidence()?.rows,displayedConsensus()?.rows) || displayedEvidence()?.rows || [];
      storeHistoryEvidence(incoming,preserveDates,data.account_actions?.series_context_hash);
      if (!preserveDates) { state.dates = (data.frequencies?.['日线']?.bars || []).map((bar) => day(bar.time)); state.replayIndex = Math.max(0,state.dates.length - 1); }
      renderAnalysis(); renderEvents(); updateReplay();
      if (locate) locateTime(locate,false);
    } catch (error) {
      if (sequence !== state.analysisSequence) return;
      state.analysis = null; analysisLoading = false;
      $('event-context').innerHTML = nextSessionGuidance(null);
      $('analysis-research-evidence').hidden = true; $('analysis-research-evidence').replaceChildren();
      if (candles) { candles.setData([]); candles.setMarkers([]); penSeries.forEach((series) => chart.removeSeries(series)); penSeries = []; volume.setData([]); }
      $('event-list').innerHTML = '<div class="empty-copy">分析未完成，无法展示该请求的事件。</div>';
      $('event-count').textContent = '0'; $('analysis-state').textContent = '分析失败';
      $('chart-symbol').textContent = query.symbol; $('chart-information').textContent = '未完成'; $('structure-list').replaceChildren(); $('analysis-caveats').hidden = true; updateReplay();
      emptyChart('分析未完成', error.message); showError(error);
    } finally { if (sequence === state.analysisSequence) { analysisLoading = false; $('analysis-form').querySelector('[type=submit]').disabled = false; updateReplay(); } }
  }
  function renderReplayEvidence() {
    const ui = evidenceUI(); if (!ui) return;
    if (analysisLoading) { $('replay-evidence-card').innerHTML = '<p class="subtle">正在按所选日期重新判断，参考价与评分加载中…</p>'; $('history-evidence-table').innerHTML = '<p class="subtle">历史参考记录加载中…</p>'; return; }
    const evidence = displayedEvidence();
    const selected = state.query?.as_of || state.analysis?.as_of || state.analysis?.data_end || state.dates[state.replayIndex];
    const row = ui.rowAt(evidence,selected), decision = ui.consensusAt(displayedConsensus(),selected);
    $('replay-evidence-card').innerHTML = state.analysis ? `<strong>信息截止 ${esc(selected || '—')}</strong>${ui.accountActionHtml(row)}${ui.consensusHtml(decision,displayedConsensus()?.config,displayedConsensus()?.version)}${ui.referenceHtml(row)}${ui.agentsHtml(row,false,decision,displayedConsensus()?.config,displayedConsensus()?.version)}${ui.riskHtml(row?.raw_risk_plan)}${ui.rawEventsHtml(row,[],state.analysis?.consensus_decisions?.version)}<p class="subtle">参考价使用该日行情；评分按当时可见的模型与事件展示。</p>` : '<p class="subtle">尚无成功分析结果，参考价与评分不可用。</p>';
    if (!state.analysis) { $('history-evidence-table').innerHTML = '<p class="subtle">历史参考记录不可用，请重新分析。</p>'; return; }
    const cutoff = day(state.query?.as_of || state.analysis?.as_of || state.analysis?.data_end);
    const rows = cutoff ? historyRows.filter((item) => item.date <= cutoff) : historyRows;
    historyPage = ui.historyPage(rows,historyPage).page;
    $('history-evidence-table').innerHTML = ui.historyHtml(rows,historyPage,day(selected),evidence?.statistics);
  }
  function updateReplay() {
    const available = Boolean(state.analysis) && state.dates.length > 0 && !analysisLoading;
    ['replay-slider','replay-date','replay-reset'].forEach((id) => { $(id).disabled = !available; });
    $('replay-slider').max = String(Math.max(0,state.dates.length - 1)); $('replay-slider').value = String(state.replayIndex);
    $('replay-date').min = state.dates[0] || ''; $('replay-date').max = state.dates[state.dates.length - 1] || ''; $('replay-date').value = state.dates[state.replayIndex] || '';
    $('replay-prev').disabled = !available || state.replayIndex === 0; $('replay-next').disabled = !available || state.replayIndex >= state.dates.length - 1;
    renderReplayEvidence();
  }
  async function replay(index) {
    if (!state.query || !state.dates.length) return;
    state.replayIndex = Math.max(0,Math.min(state.dates.length - 1,index)); updateReplay();
    await analyze({...state.query,as_of:state.dates[state.replayIndex]},true);
  }
  async function searchSymbols(query = '', updateOptions = true) {
    const data = await request(`/api/symbols?q=${encodeURIComponent(query)}&limit=200`);
    const list = data.symbols || [];
    list.forEach((item) => state.symbolCatalog.set(item.symbol,item));
    if (updateOptions) $('symbol-options').innerHTML = list.map((item) => `<option value="${esc(item.symbol)}" label="${esc(item.name || item.symbol)}"></option>`).join('');
    return list;
  }
  async function refreshData() {
    const data = await request('/api/data/status');
    const latest = data.latest_date || data.data_end || data.end;
    $('latest-date').textContent = latest || '暂无行情'; $('data-latest').textContent = latest || '暂无';
    $('coverage-label').textContent = `${number(data.stocks ?? data.symbols)} 只股票 · ${number(data.rows)} 条行情`;
    $('data-stocks').textContent = number(data.stocks ?? data.symbols); $('data-rows').textContent = number(data.rows); $('data-source').textContent = display(data.source || (data.available ? '本地 SQLite 行情' : '尚未连接'));
    $('data-detail').textContent = latest ? `基础行情与合成周期：${(data.frequencies || []).join('、') || '以分析结果为准'}。${basisLabel(data.price_basis)}。更新后重新核对覆盖。` : data.reason || '没有可用行情。请先更新数据，再运行分析或回测。';
    $('connection-state').textContent = latest ? '数据已连接' : '尚无数据'; $('connection-state').className = `status-dot ${latest ? 'ready' : ''}`;
    if (latestCoverage?.market_latest_date) {
      $('latest-date').textContent = latestCoverage.market_latest_date;
      $('connection-state').textContent = latestCoverage.partial_coverage ? '最新日部分覆盖' : '数据已连接';
      $('coverage-label').textContent = `最新日 ${number(latestCoverage.latest_symbols)} / ${number(latestCoverage.total_symbols)} 只股票有行情`;
    }
    if (latest && !latestCoverage) { const next = day(latest); ['analysis','scan','backtest'].forEach((prefix) => { if (dateModes[prefix] === 'latest') $(`${prefix}-end`).value = next; }); latestDefaultEnd = next; }
    if (data.latest_symbols != null) $('data-detail').textContent += ` 最新日覆盖 ${number(data.latest_symbols)} / ${number(data.stocks ?? data.symbols)} 只；缺行情股票不沿用前日评分。`;
    if (latest && (!latestCoverage || latestCoverage.market_latest_date !== day(latest) || latestCoverage.latest_symbols !== data.latest_symbols)) await refreshLatestDates();
    return data;
  }
  function setDateMode(prefix, mode) {
    dateModes[prefix] = mode;
    const caption = $(`${prefix}-date-mode`);
    if (caption) caption.textContent = mode === 'latest' ? '自动最新可用行情' : '历史截止日';
  }
  async function refreshLatestDates() {
    const sequence = ++latestDateSequence;
    let symbol;
    try { symbol = getSymbol($('analysis-symbol').value); } catch (_) { symbol = null; }
    const data = await request(`/api/data/latest${symbol ? `?symbol=${encodeURIComponent(symbol)}` : ''}`);
    let currentSymbol;
    try { currentSymbol = getSymbol($('analysis-symbol').value); } catch (_) { currentSymbol = null; }
    if (sequence !== latestDateSequence || currentSymbol !== symbol) return data;
    latestCoverage = data; latestDefaultEnd = data.market_latest_date || data.latest_date;
    ['scan','backtest'].forEach((prefix) => { if (dateModes[prefix] === 'latest' && latestDefaultEnd) $(`${prefix}-end`).value = latestDefaultEnd; });
    if (dateModes.analysis === 'latest') $('analysis-end').value = data.latest_date || '';
    if (data.market_latest_date) {
      $('latest-date').textContent = data.market_latest_date;
      $('connection-state').textContent = data.partial_coverage ? '最新日部分覆盖' : '数据已连接';
      $('connection-state').className = 'status-dot ready';
      $('coverage-label').textContent = `最新日 ${number(data.latest_symbols)} / ${number(data.total_symbols)} 只股票有行情`;
    }
    return data;
  }
  function computeCaption(info, capability = false) {
    if (!info) return '该记录未保存计算设备信息。';
    const device = info.selected_device === 'mixed' ? (info.selected_devices || []).join('、') : info.selected_device || info.requested_device || '未选择';
    if (capability) return `${info.available ? '设备可用' : '设备不可用'} · ${device}${info.torch_version ? ` · PyTorch ${info.torch_version}` : ''}${info.cuda_version ? ` · CUDA ${info.cuda_version}` : ''}${info.fallback_reason ? ` · ${info.fallback_reason}` : ''}`;
    const work = info.cuda_work ? `本次 CUDA 批量信号计算 · ${device}` : info.bars === 0 ? '本批没有可计算行情' : `本次 ${info.backend === 'pytorch' ? 'PyTorch CPU' : 'CPU'} 信号计算`;
    return `${work}${info.batches != null ? ` · ${number(info.batches)} 个微批` : ''}${info.bars != null ? ` · ${number(info.bars)} 根预热及研究行情` : ''}；结构、Position 与实际成交在 CPU。${info.selected_device === 'mps' ? ' MPS 结构价格比较保留 CPU float64 精度。' : ''}${info.fallback_reason ? ` ${info.fallback_reason}` : ''}`;
  }
  async function refreshCompute(kind = 'data', device) {
    const sequence = ++computeSequences[kind];
    const data = await request(`/api/compute/status${device ? `?device=${encodeURIComponent(device)}` : ''}`);
    if (sequence !== computeSequences[kind]) return;
    if (!computeLoaded) {
      let options = '<option value="auto">自动选择</option><option value="cpu">CPU</option>' + (data.devices || []).map((gpu) => `<option value="${esc(gpu.device || `cuda:${Number(gpu.index)}`)}">${gpu.type === 'mps' ? 'MPS' : `GPU ${Number(gpu.index)}`} · ${esc(gpu.name)}</option>`).join('');
      const selected = data.requested_device === 'cuda' ? data.selected_device : data.requested_device;
      if (/^(mps|cuda:\d+)$/.test(selected || '') && !(data.devices || []).some((gpu) => (gpu.device || `cuda:${gpu.index}`) === selected)) options += `<option value="${esc(selected)}">${esc(selected)}（不可用）</option>`;
      ['analysis','scan','backtest','data'].forEach((prefix) => { $(`${prefix}-device`).innerHTML = options; if ([...$(`${prefix}-device`).options].some((option) => option.value === selected)) $(`${prefix}-device`).value = selected; $(`${prefix}-compute-status`).textContent = computeCaption(data,true); });
      computeLoaded = true;
    } else $(`${kind}-compute-status`).textContent = computeCaption(data,true);
    $('compute-devices').textContent = (data.devices || []).map((gpu) => gpu.type === 'mps' ? 'Apple MPS · float32 模型训练与推理' : `GPU ${gpu.index} · ${gpu.name} · ${number(gpu.total_memory_bytes / 1024 ** 3,1)} GB`).join('；') || (data.torch_available === false ? 'PyTorch 不可用；完整研究需要安装 Torch。' : '当前使用 CPU；未检测到可用 CUDA 或 MPS 设备。');
    renderScanComputePlan();
    return data;
  }
  function inferenceReasonText(info, result = {}) {
    const labels = {model_unavailable:'无兼容检查点',model_input_ineligible:'模型输入不可用',unsupported_board:'板块暂不支持 ML 推理'};
    if (info.inference_reason_counts) {
      const reasons = Object.entries(info.inference_reason_counts).filter(([,count]) => Number(count) > 0)
        .map(([code,count]) => `${labels[code] || code} ${number(count)} 行`);
      if (reasons.length) return reasons.join('；');
    }
    const latest = result.next_session_decision || {};
    if (Number(info.model_inference_rows) === 0 && latest.agents?.ma?.reason_codes?.includes('no_prior_trend_head')) return '截至判断日没有更早可用的趋势检查点';
    const routes = result.research?.model_routes || result.model_routes || [];
    const identities = routes.map((route) => route.identity || route);
    const reasons = [];
    if (identities.length && identities.every((route) => ['rules_no_model','no_model'].includes(route.status) || route.reason_codes?.includes('model_unavailable'))) reasons.push('无兼容检查点');
    if (modelInputUnavailable(latest.model_resolution,latest)) reasons.push('当前模型输入不可用');
    return reasons.length ? reasons.join('；') : '本次记录未保存未推理原因';
  }
  function researchComputeCaption(info, result = {}) {
    if (info.model_inference_rows == null && isHighReturn(result) && result.actual_inference_device) return `${modelName(info,result)} · 本次模型推理设备 ${result.actual_inference_device}；逐账户推理记录保存在完整离线证据中。实际Broker成交在 CPU。`;
    if (info.trained_models != null) return `${modelName(info,result)}研究 · ${info.actual_mps_training ? '本次 MPS' : info.actual_cuda_training ? '本次 CUDA' : '本次 CPU'} 训练 ${number(info.trained_models)} 个模型、${number(info.training_batches)} 批；入场评估 ${number(info.inference_rows)} 行，真实持仓退出 CPU NumPy ${number(info.exit_cpu_numpy_rows)} 行。训练设备与退出推理设备分别记录；模型保持影子。`;
    const name = modelName(info,result), trainingLabel = '模型训练', inferenceLabel = '模型推理';
    const actual = info.actual_device || info.device || (info.mps_work || info.actual_mps_inference ? 'mps' : info.cuda_work || info.actual_cuda_inference ? 'cuda' : info.selected_device === 'cpu' ? 'cpu' : null);
    if (info.training) return `${name} · 本次 ${info.mps_work ? 'MPS' : info.cuda_work ? 'CUDA' : info.actual_devices?.length ? info.actual_devices.join('、') : actual || '实际设备未记录'} ${trainingLabel} · 完成 ${number(info.model_training_models)} 个模型 · ${number(info.model_training_batches)} 个优化批次；结构准备与实际账本在 CPU。`;
    if (info.model_inference_rows == null) return `${name} · 所选设备 ${info.selected_device || '—'}；该记录未保存本次 ML 推理统计。结构、量价与实际成交在 CPU。`;
    const rows = Number(info.model_inference_rows);
    const work = rows > 0 ? `${info.cuda_work || info.actual_cuda_inference ? `本次 CUDA ${inferenceLabel}` : info.mps_work || info.actual_mps_inference ? `本次 MPS ${inferenceLabel}` : actual === 'cpu' ? `本次 CPU ${inferenceLabel}` : `本次 ${inferenceLabel}（实际设备未记录）`} · ${actual || '—'} · ${number(rows)} 行` : `本次未执行 ML 推理（0 行）：${inferenceReasonText(info,result)}；结构与量价在 CPU`;
    const experts = info.expert_inference;
    const expertWork = experts ? ' · 专家实际行数 ' + Object.entries(experts).map(([name,value]) => `${({ma:'均线',structure:'结构',wyckoff:'量价',fusion:'三专家融合',dual_fusion:'双专家融合'})[name] || name} ${number(value.rows)}`).join(' / ') : '';
    const timings = [['结构准备',info.preparation_wall_seconds],['推理阶段',info.inference_wall_seconds],['决策',info.decision_wall_seconds]].filter(([,value]) => value != null).map(([label,value]) => `${label} ${number(value,2)}秒`).join(' · ');
    return `${name} · ${work}${info.model_inference_batches != null ? ` · ${number(info.model_inference_batches)} 个推理批次` : ''}${info.model_loads != null ? ` · 模型加载 ${number(info.model_loads)} 次` : ''}${info.cpu_workers != null ? ` · CPU 准备并发上限 ${number(info.cpu_workers)}` : ''}${info.cache_hits != null ? ` · 特征缓存命中 ${number(info.cache_hits)}` : ''}${timings ? ` · ${timings}` : ''}${expertWork}。GPU 工作仅依据本次执行记录；结构识别与实际成交在 CPU。`;
  }
  function renderComputeResult(kind, info) { $(`${kind}-compute-result`).hidden = !state[kind]; $(`${kind}-compute-result`).textContent = info?.research ? researchComputeCaption(info,state[kind]) : computeCaption(info); }
  function renderScanComputePlan() {
    const research = $('scan-mode').value === 'research', device = $('scan-device').value;
    $('scan-compute-plan').textContent = !research ? '结构识别在 CPU；CUDA 可执行批量信号运算，MPS 的结构价格比较保持 CPU float64 精度。' : `CPU 准备结构与量价特征，按所选日期匹配模型并批量推理（设备 ${device === 'auto' ? '自动选择' : device}）；缺少合格入场头时明确阻断。真实推理行数、批次和 GPU 工作以本次结果为准。`;
  }
  function gateCaption(gate) { return gate?.passed ? '样本外门槛通过' : '影子观察：尚未证实改善'; }
  function researchModels(prefix) {
    const capability = {analysis:'chart',scan:'scan',backtest:'backtest'}[prefix];
    return (state.research?.models || []).filter(model => (!model.model_family || model.model_family === 'current') && model.capabilities?.[capability] !== false);
  }
  function populateResearchModels(prefix) {
    const select = $(`${prefix}-model`), previous = select.value, models = researchModels(prefix), context = state.pendingResearchContexts[prefix];
    const configured = state.research?.active_model?.run_id || state.research?.current_model_run_id;
    const preferred = previous && previous !== select.dataset?.configuredDefaultModel ? previous : context?.model_run_id;
    const explicit = Boolean(preferred), absent = explicit && !models.some(model => model.run_id === preferred) ? [preferred] : [];
    select.innerHTML = '<option value="">自动使用当前模型</option>' + models.map(model => `<option value="${esc(model.run_id)}">${esc(model.display_name || modelName())} · ${esc(model.run_id)} · ${esc(gateCaption(model.model_gate))}</option>`).join('') + absent.map(id => `<option value="${esc(id)}">已绑定当前运行（加载时核验） · ${esc(id)}</option>`).join('');
    select.value = explicit ? preferred : '';
    if (select.dataset) select.dataset.configuredDefaultModel = explicit ? '' : configured || '';
  }
  function updateResearchControls(prefix) {
    const enabled = $(`${prefix}-mode`).value === 'research';
    $(`${prefix}-model-family`).value = 'current'; $(`${prefix}-model-family`).disabled = true;
    $(`${prefix}-entry-policy`).value = 'risk'; $(`${prefix}-entry-policy`).disabled = true;
    if (prefix === 'backtest') { $(`${prefix}-usage-mode`).value = 'historical'; $(`${prefix}-model-policy`).value = 'auto'; }
    $(`${prefix}-model-policy`).disabled = !enabled || prefix === 'backtest';
    $(`${prefix}-usage-mode`).disabled = !enabled || prefix === 'backtest';
    const pinned = $(`${prefix}-model-policy`).value === 'pinned', model = researchModels(prefix).find(item => item.run_id === $(`${prefix}-model`).value);
    const checkpoints = researchCheckpoints(prefix,model), select = $(`${prefix}-checkpoint`), previous = select.value;
    select.innerHTML = '<option value="">自动按日期匹配</option>' + checkpoints.map(item => `<option value="${esc(item.name)}">${esc(item.name)} · 最早可用 ${esc(item.available_at || '加载时核验')}</option>`).join('');
    if (checkpoints.some(item => item.name === previous)) select.value = previous;
    select.disabled = !enabled || !pinned || !checkpoints.length;
    $(`${prefix}-model`).disabled = !enabled;
    const help = $(`${prefix}-entry-policy-help`); help.hidden = !enabled;
    help.textContent = '有效原机会在实际空仓且风险合格期间逐日重评；质量拒绝、缺模型及暂时风险阻断不消费机会。实际成交使用下一核验开盘与Broker规则；持仓退出头在入场时固定，保护出口优先。';
    if (prefix === 'backtest') { const cash = $('initial-cash'); cash.disabled = enabled; if (enabled) cash.value = String(model?.initial_cash || state.research?.initial_cash || 100000); $('initial-cash-label').textContent = '每只股票初始资金（元）'; }
    renderCheckpointHelp(prefix);
  }
  function researchCheckpoints(prefix, model) {
    if (model) return model.checkpoints || [];
    const restored = state.pendingResearchContexts[prefix];
    return restored?.model_fold && restored.model_run_id === $(`${prefix}-model`).value
      ? [{name:restored.model_fold,available_at:null,restored:true}] : [];
  }
  function selectedCheckpoint(prefix) {
    const model = researchModels(prefix).find((item) => item.run_id === $(`${prefix}-model`).value);
    return researchCheckpoints(prefix,model).find((item) => item.name === $(`${prefix}-checkpoint`).value);
  }
  function renderCheckpointHelp(prefix = 'backtest') {
    const help = $(`${prefix}-checkpoint-help`); help.hidden = $(`${prefix}-mode`).value !== 'research';
    if (help.hidden) { help.textContent = ''; return; }
    const mode = $(`${prefix}-usage-mode`).value || 'historical', pinned = $(`${prefix}-model-policy`).value === 'pinned', checkpoint = selectedCheckpoint(prefix);
    help.textContent = `${usageNames[mode]}。${pinned ? checkpoint ? `固定入场检查点 ${checkpoint.name}，最早可用 ${checkpoint.available_at || '加载时核验'}。` : '请选择合格模型与检查点。' : '按信号日期匹配合格先验检查点，账户与账本连续。'}缺少合格入场头明确阻断；MA退出头在入场时固定，硬止损与冻结边界优先。退出观察头不自动改变交易。${mode === 'historical' ? '历史评分为重建研究。' : '依据实际训练完成时间核验。'}正式资格以独立认证为准。`;
  }
  function researchSpec(prefix) {
    if ($(`${prefix}-mode`).value !== 'research') return {research:false};
    const usage_mode = prefix === 'backtest' ? 'historical' : $(`${prefix}-usage-mode`).value || 'historical';
    if (!['historical','forward'].includes(usage_mode)) throw new Error('旧用途已退役，不恢复资格；请选择历史滚动研究或当前时点研究。');
    const model_policy = prefix === 'backtest' ? 'auto' : $(`${prefix}-model-policy`).value || 'auto', model = $(`${prefix}-model`).value;
    let checkpoint;
    if (model_policy === 'pinned') {
      if (!model) throw new Error('请选择当前模型包与检查点。');
      checkpoint = selectedCheckpoint(prefix); if (!checkpoint) throw new Error('所选模型没有合格检查点，请刷新状态。');
      const boundary = $(`${prefix}-end`).value;
      if (boundary && checkpoint.available_at > boundary) throw new Error(`检查点 ${checkpoint.name} 在 ${checkpoint.available_at} 才可用。`);
    }
    const restored = state.pendingResearchContexts[prefix];
    return {research:true,model_family:'current',consensus_version:'trend_consensus',usage_mode,model_policy,entry_policy:'risk',
      ...(model ? {model_run_id:model} : {}),...(checkpoint ? {model_fold:checkpoint.name} : {}),
      ...(restored?.model_run_id === model ? Object.fromEntries(['account_start','account_initial_cash','consensus_config'].filter(key => restored[key] != null).map(key => [key,restored[key]])) : {}),
      ...(state.research?.calendar_run_id ? {calendar_run_id:state.research.calendar_run_id} : {}),...(prefix === 'analysis' ? {device:$('analysis-device').value || 'auto'} : {})};
  }
  async function refreshResearch() {
    const data = await request('/api/training/status'); state.research = data;
    ['analysis','scan','backtest'].forEach(prefix => { populateResearchModels(prefix); updateResearchControls(prefix); if (state.pendingResearchContexts[prefix]) { const context = state.pendingResearchContexts[prefix]; delete state.pendingResearchContexts[prefix]; applyResearchContext(prefix,context); } });
    renderScanComputePlan();
    $('research-status').textContent = `趋势共识完整模型包 · ${(data.models || []).length} 个当前运行 · ${data.active_model?.run_id ? `当前选择 ${data.active_model.run_id}` : '尚无合格当前选择'}。${data.calendar_run_id ? '核验交易日历可用。' : '核验日历状态以训练预检为准。'}缺少合格入场头明确阻断；正式资格 ${data.certification?.status === 'qualified' ? '已认证，仍按信号时点核验' : '尚未取得'}。`;
    renderTasks(); return data;
  }
  function operationFeatureEvidence(row) {
    const featureGroups = row.expert_features || row.dual_features;
    if (!featureGroups) return '';
    const three = Boolean(featureGroups.wyckoff);
    const labels={ma5_bias:'距5日均线',ma10_bias:'距10日均线',ma20_bias:'距20日均线',ma60_bias:'距60日均线',ma5_slope5:'5日均线近5日斜率',ma10_slope5:'10日均线近5日斜率',ma20_slope5:'20日均线近5日斜率',ma60_slope5:'60日均线近5日斜率',ma_alignment:'均线排列',close_above_ma20:'收盘相对20日均线方向',ret_1:'近1日收益',ret_5:'近5日收益',ret_20:'近20日收益',atr14_ratio:'14日真实波幅比',volatility20:'20日波动率',drawdown20:'20日回撤',drawdown60:'60日回撤',ma5_ma10_gap:'5与10日均线间距',ma10_ma20_gap:'10与20日均线间距',ma20_ma60_gap:'20与60日均线间距',bi1_direction:'最近笔方向',bi1_return:'最近笔涨跌',bi1_length:'最近笔长度',bi1_volume_ratio:'最近笔量能比',bi2_direction:'次近笔方向',bi2_return:'次近笔涨跌',bi2_length:'次近笔长度',bi3_direction:'第三近笔方向',bi3_return:'第三近笔涨跌',bi3_length:'第三近笔长度',bi1_power_ratio:'相邻笔力度比',confirmed_age:'结构确认距今',zone_low_distance:'距中枢下沿',zone_high_distance:'距中枢上沿',zone_width_ratio:'中枢宽度',weekly_return:'闭合周收益',weekly_direction:'闭合周方向',monthly_return:'闭合月收益',monthly_direction:'闭合月方向',weekly_bi_direction:'周线笔方向',monthly_bi_direction:'月线笔方向',finished_bi_count:'完成笔数',w_spread_atr:'振幅相对前期ATR',w_body_atr:'实体相对前期ATR',w_close_location:'收盘在当日振幅位置',w_upper_shadow:'上影线占比',w_lower_shadow:'下影线占比',w_gap_atr:'开盘缺口相对ATR',w_relative_volume20:'相对前20日成交量',w_volume_robust60:'前60日稳健量能偏离',w_volume5_to20:'5日与20日量能比',w_relative_amount20:'相对前20日成交额',w_up_down_volume20:'上涨与下跌量能比',w_price_volume_divergence:'价量变化分歧',w_pullback_push_volume20:'回撤与推进量能比',w_volume_contraction5:'近5日量能收缩',w_range_width_atr:'历史区间宽度相对ATR',w_range_duration:'区间样本长度',w_range_position:'收盘相对历史区间位置',w_support_distance_atr:'距历史支撑相对ATR',w_resistance_distance_atr:'距历史阻力相对ATR',w_compression20_60:'20日与60日区间压缩',w_down_probe_atr:'下探支撑幅度',w_up_probe_atr:'上探阻力幅度',w_spring:'Spring下探收回',w_test:'Test后续缩量测试',w_sos:'SOS放量突破',w_lps:'LPS后续回测确认',w_upthrust:'Upthrust上冲回落',w_sow:'SOW放量跌破',w_lpsy:'LPSY弱反弹确认',w_event_age:'最近事件距今',w_demand_score:'需求规则评分',w_supply_score:'供应规则评分'};
    const groups=Object.entries(featureGroups).map(([name,group]) => `<details><summary>${name === 'ma' ? '均线专家 · 20项输入' : name === 'structure' ? '结构专家 · 22项输入' : '威科夫量价专家 · 32项输入'}</summary><p>当前可见输入；此处不代表模型重要性或因果贡献。</p><table><tbody>${Object.entries(group.values || {}).map(([key,value]) => `<tr><th>${esc(labels[key] || key)}</th><td>${value == null ? '缺失' : number(value,5)}</td></tr>`).join('')}</tbody></table></details>`).join('');
    return `<details class="research-evidence"><summary>当前因果特征与决策路径</summary><p>已确认结构、均线趋势${three ? '与逐日量价证据' : ''} → 趋势与高收益质量分别评分 → 当日资格与入场门控 → 下一核验开盘。独立退出头另读真实Broker持仓；当前未认证，均为影子。需求与供应规则评分不等于盈利概率。</p>${wyckoffEvidence(row)}${groups}</details>`;
  }
  function wyckoffEvidence(row) {
    const evidence = row.wyckoff_evidence;
    if (!evidence) return '';
    const states = {unknown:'输入或历史区间不足',range:'历史区间观察',spring_pending:'Spring等待后续测试',sos_pending:'SOS等待后续回测',demand_test_confirmed:'Test需求测试已确认',demand_retest_confirmed:'LPS需求回测已确认',upthrust_pending:'上冲回落供应观察',sow_pending:'跌破供应观察',supply_retest_confirmed:'LPSY供应回测已确认',failed_demand:'需求候选已失效',failed_supply:'供应候选已失效',conflict:'量价证据冲突'};
    return `<section class="entry-plan" aria-label="威科夫量价证据"><strong>威科夫量价证据 · ${esc(states[evidence.state] || evidence.state || '未知')}</strong><p>事件 ${esc(evidence.event || 'none')} · 原始锚点 ${esc(evidence.anchor_at || '—')} · 首次观测 ${esc(evidence.observed_at || '—')} · 可用 ${esc(evidence.available_at || '—')}</p><p>历史区间 ${number(evidence.range_low,2)}～${number(evidence.range_high,2)} · 区间形成截至 ${esc(evidence.range_formed_at || '—')} · 失效参考 ${number(evidence.stop,2)}</p><p>${evidence.buy_candidate ? '后续测试已形成规则买点候选' : '本日未形成量价规则买点'} · ${evidence.sell_evidence ? '本日存在供应方向证据' : '本日无新供应事件'} · ${evidence.input_eligible ? '量价数值输入可用' : '量价输入不可用'}。${esc((evidence.reason_codes || []).join('；'))}</p><p>日线标记显示首次确认日；Spring/SOS本身仍需后续测试。事件表示量价解释，不表示真实机构行为。新模型和规则买点仍处于研究阶段。</p></section>`;
  }
  function researchEvidence(row) {
    if (isHighReturn(row)) return highReturnEvidence(row);
    if (!row.category && !row.model_resolution && !row.agent_evidence?.length) return '';
    const roles = {czsc:'缠论辅助结构',price_volume:'均线量价',ml:'机器学习',data_execution:'数据与成交风险'};
    const judgments = {support:'支持',sell:'退出方向',observe:'观察',shadow:'影子观察',allow:'允许',eligible_signal:'信号输入合格',unavailable:'不可用',veto:'否决'};
    const evidence = (row.agent_evidence || []).map((item) => {
      let detail = '', role = roles[item.role] || item.role, judgment = judgments[item.judgment] || item.judgment;
      if (item.role === 'czsc') detail = `${item.point_type || '无确认辅助点'} · 确认 ${item.confirmed_at || '尚未确认'}`;
      if (item.role === 'price_volume') detail = `距20日均线 ${percent(item.values?.ma20_bias)} · 距60日均线 ${percent(item.values?.ma60_bias)} · 量比 ${number(item.values?.vol_ratio20,2)} 倍`;
      if (item.role === 'ml') {
        const resolution = {...(item.model || {}),...(row.model_resolution || {})};
        if (!Object.prototype.hasOwnProperty.call(row.model_resolution || {},'probability')) {
          if (Object.prototype.hasOwnProperty.call(row,'model_probability')) resolution.probability = row.model_probability;
          else if (Object.prototype.hasOwnProperty.call(row,'probability')) resolution.probability = row.probability;
          else if (!Object.prototype.hasOwnProperty.call(resolution,'probability') && Object.prototype.hasOwnProperty.call(item,'probability')) resolution.probability = item.probability;
        }
        const probability = modelProbability(resolution,row), applied = modelEntryApplied({...resolution,applied_to_entry:row.model_resolution?.applied_to_entry ?? item.applied ?? resolution.applied_to_entry},row);
        if (modelName(resolution,row) !== 'ML') role = modelName(resolution,row);
        if (probability == null) judgment = modelInputUnavailable(resolution,row) ? '输入不可用' : '不可用';
        else if (!applied) judgment = '影子观察';
        detail = probability == null ? `概率不可用 · ${modelInputUnavailable(resolution,row) ? '模型输入不可用' : ['rules_no_model','no_model'].includes(resolution.status) ? '无兼容检查点' : '未生成模型概率'}` : `假设新入场的10交易日费用后正收益概率 ${percent(probability)} · 门槛 ${percent(item.threshold ?? resolution.probability_threshold)} · ${applied ? '已参与入场过滤' : '未参与入场，仅作影子观察'}`;
      }
      if (item.role === 'data_execution') detail = `实际行情 ${item.actual_data_end || row.data_end || row.reference_price_date || '缺失'} · ${['allow','eligible_signal'].includes(item.judgment) ? '输入质量通过，下一开盘仍需成交检查' : readableReason({category:'excluded',reason_codes:item.reasons || []})}`;
      return `<div><strong>${esc(role)}：${esc(judgment)}</strong><span class="stock-name">${esc(detail)}</span></div>`;
    }).join('');
    const levels = row.reference_levels;
    const references = levels ? `<div>参考：MA20 ${number(levels.ma20,2)} · MA60 ${number(levels.ma60,2)} · 结构低/高 ${number(levels.structure_low,2)} / ${number(levels.structure_high,2)}。参考值不保证成交。</div>` : '';
    return `${entryPlanHtml(row)}${operationFeatureEvidence(row)}<details class="research-evidence"><summary>触发条件与四角色证据</summary>${row.candidate_action ? `<div>组合条件：${esc(actionLabel(row.candidate_action))}候选；仓位意图：${esc(actionLabel(row.position_intent || row.action))}。条件满足可能仍无新的转仓。</div>` : ''}<div>${esc(ruleQualificationText(row))}</div><div>买入：${esc(row.entry_trigger || '以组合规则和仓位变化为准')}</div><div>失效：${esc(row.invalidation || '以结构退出及原生持仓约束为准')}</div><div>退出：${esc(row.exit_trigger || '规则退出与原生风控；ML 仅过滤入场')}</div>${references}<div>计划最早执行：${esc(row.next_market_session || '需刷新交易日历')}</div><div>${esc(row.probability_target || 'ML 估计假设新入场后固定10交易日费用后正收益，非下一日涨幅或成交价。')}</div>${row.model_resolution ? `<div>${esc(modelResolutionText(row.model_resolution,row))}</div>` : `<div>模型训练标签截至：${esc(row.model_train_end || '无可用模型')} · 模型可用 ${esc(row.model_available_at || '缺失')}</div>`}${evidence}</details>`;
  }
  function scanActionCell(row) {
    const categories = {buy:'买入候选',watch:'继续观察',exit:'持仓退出研究',excluded:'排除 / 不足'};
    const intent = row.position_intent || row.action;
    const trendStrategy = isTrendStrategy(row);
    const primary = row.category === 'excluded' ? categories.excluded : trendStrategy ? row.account_action?.emitted === true ? `${row.account_action.action === 'SELL' ? 'S · 账户退出意图' : 'B · 账户买入意图'}` : '账户观察' : row.position_intent ? `理论仓位意图：${actionLabel(intent)}` : categories[row.category] || actionLabel(intent);
    const candidate = row.candidate_action ? `<span class="stock-name">组合条件：${esc(actionLabel(row.candidate_action))}候选</span>` : '';
    const personal = row.personal_exit_review ? `<span class="stock-name valuation-loss">手动持仓退出复核：${esc(actionLabel(row.personal_exit_review.action || row.personal_exit_review))}（独立于策略意图）</span>` : '';
    const resolution = row.model_resolution;
    const probability = modelProbability(resolution,row);
    const model = trendStrategy ? `<span class="stock-name">趋势共识策略 · ${esc(evidenceUI()?.trendProbabilities(row.agents?.ma).summary || '趋势概率未评分')}</span><span class="stock-name">${esc(evidenceUI()?.currentAgentView(row,'high_return').summary || '高收益评分未提供')}</span>` : row.category ? `<span class="stock-name">${esc(ruleQualificationText(row))}</span><span class="stock-name">${esc(modelName(resolution,row))} · ${probability == null ? modelInputUnavailable(resolution,row) ? '模型输入不可用' : '模型概率不可用' : `${isHighReturn(row) ? '高收益区间研究概率' : '10日研究概率'} ${percent(probability)}`} · ${modelEntryApplied(resolution,row) ? 'ML参与入场' : probability == null ? '未生成模型概率' : '影子观察，ML未参与入场'}</span>` : '';
    return `<span class="badge ${row.category === 'excluded' ? '' : actionClass(intent)}">${esc(primary)}</span>${candidate}${personal}${model}`;
  }
  function diagnostic(row) {
    if (!row || typeof row !== 'object') return display(row);
    const reasons = {unsupported_board:'该板块尚不支持模拟成交',insufficient_seasoned_bars:'有效历史成交记录不足',legacy_fuyao_price_basis_unverified:'当日或前日含旧复权未核验记录，拒绝成交',unverified_trade_price:'成交价格尚未核验',suspended_or_zero_volume:'停牌或成交量为零',conservative_upper_price_guard:'触发保守上限价格保护',conservative_lower_price_guard:'触发保守下限价格保护',t_plus_one:'当日买入库存尚不可卖出',insufficient_cash_or_prior_liquidity:'现金或前日流动性不足',broker_rejected:'资金或库存检查未通过',entry_plan_not_active:'当前没有可执行的新入场计划',entry_plan_consumed:'该入场计划已使用，禁止重复追补',entry_plan_identity_mismatch:'入场计划股票或身份不匹配',entry_plan_invalid_time_contract:'入场计划时间口径不完整',entry_plan_not_available:'信号计划在该开盘时尚不可用',entry_plan_expired:'已错过计划对应开盘，入场计划过期',entry_plan_invalid_risk_contract:'入场区间或风险预算不完整',entry_plan_price_above_ceiling:'实际执行价高于参考入场上限，取消追价',entry_plan_price_below_floor:'实际执行价低于参考入场下限，取消入场',native_target_exit:'原生目标仓位触发退出',actual_account_stop:'实际持仓成本或冻结结构失效价触发退出'};
    return [row.date ? day(row.date) : '',row.symbol || '',row.action ? actionLabel(row.action) : '',reasons[row.reason] || row.error || row.reason || display(row)].filter(Boolean).join(' · ');
  }
  function renderDiagnostics(id, rows) { const list = Array.isArray(rows) ? rows : rows ? [rows] : []; $(id).hidden = !list.length; $(id).querySelector('.diagnostics').innerHTML = list.map((row) => `<div>${esc(diagnostic(row))}</div>`).join(''); }
  function renderScan() {
    globalThis.KHMarketHistory?.render($('scan-market-history'),state.scan?.market_history,{runId:state.scan?.run_id,request,onAnalyze:({symbol,date}) => analyzeSymbol(symbol,date,state.scan).catch(showError)});
    const query = $('scan-search').value.trim().toLowerCase(), filter = $('scan-filter').value;
    const all = state.scan?.rows || [];
    renderComputeResult('scan',state.scan?.compute_info);
    const filtered = all.filter((row) => (!query || `${row.symbol} ${row.name || state.symbolCatalog.get(row.symbol)?.name || ''} ${row.signal || ''} ${display(row.reason)} ${readableEvent(row)} ${readableReason(row)}`.toLowerCase().includes(query)) && (filter === 'all' || filter === 'favorites' && state.favorites.has(row.symbol) || filter === (row.category || actionClass(row.action)) || filter === 'sell' && row.category === 'exit'));
    state.scanPage = Math.max(0,Math.min(state.scanPage,Math.ceil(filtered.length / PAGE_SIZE) - 1));
    const page = filtered.slice(state.scanPage * PAGE_SIZE,(state.scanPage + 1) * PAGE_SIZE);
    $('scan-count').textContent = number(all.length);
    const categories = {buy:'买入候选',watch:'继续观察',exit:'持仓退出研究',excluded:'排除 / 不足'};
    $('scan-rows').innerHTML = page.length ? page.map((row) => `<tr><td><button type="button" class="favorite" data-favorite="${esc(row.symbol)}" aria-label="关注 ${esc(row.symbol)}" aria-pressed="${state.favorites.has(row.symbol)}" ${!state.personalLoaded || state.watchlistBusy ? 'disabled' : ''}>${state.favorites.has(row.symbol) ? '★' : '☆'}</button></td><td><strong>${esc(row.symbol)}</strong><span class="stock-name">${esc(row.name || state.symbolCatalog.get(row.symbol)?.name || '')}</span></td><td>${esc(readableEvent(row))}</td><td>${scanActionCell(row)}</td><td>${row.category ? `行情 ${esc(row.data_end || '缺失')}<span class="stock-name">结构确认 ${esc(row.point_confirmed_at || '未确认')}</span>` : ''}${esc(display(row.available_at))}</td><td class="reason">${esc(readableReason(row))}${ruleOriginal(row)}${researchEvidence(row)}</td><td><button type="button" class="link-button" data-analyze-symbol="${esc(row.symbol)}" data-analysis-source="scan" ${row.data_end ? `data-analyze-end="${esc(row.data_end)}"` : ''}>分析</button></td></tr>`).join('') : `<tr><td colspan="7" class="empty-cell">${state.scan ? '没有符合筛选条件的股票。' : '开始扫描后，结果显示在这里。'}</td></tr>`;
    pagination('scan',filtered.length,state.scanPage);
    $('scan-research-result').hidden = !state.scan?.model_gate;
    if (state.scan) { $('scan-coverage').textContent = `覆盖 ${number(state.scan.coverage?.success)} / ${number(state.scan.coverage?.requested)} 只成功 · 失败 ${number((state.scan.failures || []).length)}${state.scan.data_range?.requested_end ? ` · 请求截止 ${state.scan.data_range.requested_end} · 实际行情 ${state.scan.data_end || state.scan.data_range.end || '缺失'}` : ''}`; renderDiagnostics('scan-failures',state.scan.failures);
      if (state.scan.model_gate) $('scan-research-result').textContent = `${gateCaption(state.scan.model_gate)} · ${Object.entries(categories).map(([key,label]) => `${label} ${number(state.scan.category_counts?.[key] || 0)}`).join(' · ')}。${state.scan.model_gate.reason || ''}`;
    }
  }
  function pagination(prefix,total,page) { $(prefix + '-page-label').textContent = total ? `${number(total)} 条 · 第 ${page + 1} / ${Math.ceil(total / PAGE_SIZE)} 页` : '0 条'; $(prefix + '-prev').disabled = page === 0; $(prefix + '-next').disabled = (page + 1) * PAGE_SIZE >= total; }
  function personalSymbol(value) {
    const text = String(value).trim();
    const known = [...state.symbolCatalog.values()].find((item) => item.symbol === text.toUpperCase() || item.symbol.slice(0,6) === text || item.name === text);
    if (known) return known.symbol;
    if (!/^\d{6}(?:\.(?:SH|SZ|BJ))?$/i.test(text)) throw new Error('请选择股票名称，或填写六位股票代码，例如 000001.SZ。');
    if (text.includes('.')) return text.toUpperCase();
    return `${text}.${/^[489]/.test(text) ? 'BJ' : text.startsWith('6') ? 'SH' : 'SZ'}`;
  }
  const returnPct = (value) => value == null ? '—' : `${number(value,2)}%`;
  const valuationClass = (value) => value == null || Number(value) === 0 ? '' : Number(value) > 0 ? 'valuation-gain' : 'valuation-loss';
  function quoteNotes(items) {
    return items.flatMap((item) => {
      const warnings = Array.isArray(item.warnings) ? item.warnings : item.warnings ? [item.warnings] : [];
      return [...(!item.quote_available && !warnings.length ? ['没有可用本地收盘价，估值显示为缺失。'] : []),...warnings].map((warning) => `${item.symbol} · ${display(warning)}`);
    });
  }
  function applyWatchlist(data) { state.watchlist = data; state.favorites = new Set((data.items || []).map((item) => item.symbol)); renderWatchlist(); renderScan(); }
  function applyHoldings(data) { state.holdings = data; renderHoldings(); }
  function personalBusy() {
    ['add-watchlist','refresh-watchlist','scan-watchlist','backtest-watchlist'].forEach((id) => { $(id).disabled = state.watchlistBusy; });
    document.querySelectorAll('[data-favorite],[data-remove-watchlist]').forEach((button) => { button.disabled = !state.personalLoaded || state.watchlistBusy; });
    $('watchlist-symbol').disabled = state.watchlistBusy;
    ['save-holding','refresh-holdings','cancel-holding-edit'].forEach((id) => { $(id).disabled = state.holdingsBusy; });
    document.querySelectorAll('[data-edit-holding],[data-remove-holding]').forEach((button) => { button.disabled = state.holdingsBusy; });
    ['holdings-symbol','holdings-shares','holdings-cost'].forEach((id) => { $(id).disabled = state.holdingsBusy; });
    $('watchlist-form').setAttribute('aria-busy',String(state.watchlistBusy));
    $('holdings-form').setAttribute('aria-busy',String(state.holdingsBusy));
  }
  function renderWatchlist() {
    const data = state.watchlist; if (!data) return;
    const items = data.items || [], coverage = data.coverage || {};
    $('watchlist-count').textContent = number(items.length);
    $('watchlist-coverage').textContent = `本地收盘价覆盖 ${number(coverage.priced)} / ${number(coverage.total)} 只 · 缺失 ${number(coverage.missing)}`;
    $('watchlist-rows').innerHTML = items.length ? items.map((item) => `<tr><td><strong>${esc(item.symbol)}</strong><span class="stock-name">${esc(item.name || state.symbolCatalog.get(item.symbol)?.name || '')}</span></td><td class="numeric">${item.quote_available ? number(item.close,3) : '无可用行情'}<span class="stock-name">${esc(basisLabel(item.price_basis))}</span></td><td>${esc(item.price_date || '—')}</td><td><button type="button" class="link-button" data-analyze-symbol="${esc(item.symbol)}">分析结构</button></td><td><button type="button" class="link-button" data-remove-watchlist="${esc(item.symbol)}">取消关注</button></td></tr>`).join('') : '<tr><td colspan="5" class="empty-cell">还没有关注股票。输入代码添加，或在扫描结果中点击星标。</td></tr>';
    renderDiagnostics('watchlist-caveats',quoteNotes(items));
    personalBusy();
  }
  function renderHoldings() {
    const data = state.holdings; if (!data) return;
    const items = data.items || [], totals = data.totals || {}, coverage = data.coverage || {}, partial = items.length > 0 && !totals.complete;
    $('holdings-count').textContent = number(items.length);
    $('holdings-coverage').textContent = `本地收盘价覆盖 ${number(coverage.priced)} / ${number(coverage.total)} 只${partial ? ' · 市值与盈亏仅汇总有价格的持仓' : ''}`;
    const metrics = [['总持仓成本（元）',totals.cost_basis], [partial ? '已覆盖市值（元）' : '持仓市值（元）',totals.market_value], [partial ? '已覆盖浮动盈亏（元）' : '浮动盈亏（元）',totals.unrealized_profit], [partial ? '已覆盖持仓收益率' : '持仓收益率',totals.unrealized_return_pct]];
    $('holdings-metrics').innerHTML = metrics.map(([label,value],index) => `<div class="metric"><span>${label}</span><strong class="${index > 1 ? valuationClass(value) : ''}">${index === 3 ? returnPct(value) : number(value,2)}</strong></div>`).join('');
    $('holdings-rows').innerHTML = items.length ? items.map((item) => `<tr><td><strong>${esc(item.symbol)}</strong><span class="stock-name">${esc(item.name || state.symbolCatalog.get(item.symbol)?.name || '')}</span></td><td class="numeric">${number(item.shares)}</td><td class="numeric">${number(item.average_cost,4)}</td><td class="numeric">${item.quote_available ? number(item.close,3) : '无可用行情'}<span class="stock-name">${esc(item.price_date || '—')} · ${esc(basisLabel(item.price_basis))}</span></td><td class="numeric">${number(item.market_value,2)}</td><td class="numeric ${valuationClass(item.unrealized_profit)}">${number(item.unrealized_profit,2)}</td><td class="numeric ${valuationClass(item.unrealized_return_pct)}">${returnPct(item.unrealized_return_pct)}</td><td><div class="personal-actions"><button type="button" class="link-button" data-analyze-symbol="${esc(item.symbol)}">分析</button><button type="button" data-edit-holding="${esc(item.symbol)}">修改</button><button type="button" data-remove-holding="${esc(item.symbol)}">删除</button></div></td></tr>`).join('') : '<tr><td colspan="8" class="empty-cell">还没有持仓记录。填写股票、股数和平均成本后保存。</td></tr>';
    renderDiagnostics('holdings-caveats',quoteNotes(items));
    personalBusy();
  }
  async function loadPersonal() {
    let legacy = [];
    try { const saved = JSON.parse(localStorage.getItem('khquant_czsc_favorites') || '[]'); if (Array.isArray(saved)) legacy = saved; } catch (_) { /* Keep an unreadable legacy value; never overwrite persistent records. */ }
    if (legacy.length) {
      const imported = [], skipped = [];
      legacy.forEach((value) => { try { imported.push(personalSymbol(value)); } catch (_) { skipped.push(value); } });
      if (imported.length) await request('/api/watchlist',{method:'POST',body:JSON.stringify({symbols:[...new Set(imported)]})});
      if (skipped.length) localStorage.setItem('khquant_czsc_favorites',JSON.stringify(skipped)); else localStorage.removeItem('khquant_czsc_favorites');
      notice(skipped.length ? `${skipped.length} 条原关注无法识别，已保留原本地记录；其余已迁移` : `已将 ${new Set(imported).size} 只原关注股票保存到本机服务`);
    }
    const [watchlist,holdings] = await Promise.all([request('/api/watchlist'),request('/api/holdings')]);
    state.personalLoaded = true; applyWatchlist(watchlist); applyHoldings(holdings);
    $('watchlist-state').textContent = '已从本机服务读取'; $('holdings-state').textContent = '已从本机服务读取';
  }
  function ensurePersonal() {
    if (!personalReady) personalReady = loadPersonal().catch((error) => { personalReady = null; $('watchlist-state').textContent = '读取失败，请刷新重试'; $('holdings-state').textContent = '读取失败，请刷新重试'; throw error; });
    return personalReady;
  }
  async function refreshWatchlist() { await ensurePersonal(); if (state.watchlistBusy) return; const sequence = ++state.watchlistSequence; const data = await request('/api/watchlist'); if (sequence === state.watchlistSequence) { applyWatchlist(data); $('watchlist-state').textContent = '关注与行情已刷新'; } }
  async function refreshHoldings() { await ensurePersonal(); if (state.holdingsBusy) return; const sequence = ++state.holdingsSequence; const data = await request('/api/holdings'); if (sequence === state.holdingsSequence) { applyHoldings(data); $('holdings-state').textContent = '持仓与行情已刷新'; } }
  async function changeWatchlist(symbol,remove) {
    await ensurePersonal(); if (state.watchlistBusy) return;
    const previous = new Set(state.favorites); state.watchlistBusy = true; ++state.watchlistSequence;
    remove ? state.favorites.delete(symbol) : state.favorites.add(symbol); renderScan(); personalBusy();
    $('watchlist-state').textContent = '正在保存…';
    try {
      const data = await request(remove ? `/api/watchlist/${encodeURIComponent(symbol)}` : '/api/watchlist',remove ? {method:'DELETE'} : {method:'POST',body:JSON.stringify({symbols:[symbol]})});
      applyWatchlist(data); $('watchlist-state').textContent = '已保存到本机服务'; notice(remove ? '已取消关注' : '已添加关注');
    } catch (error) { state.favorites = previous; renderScan(); $('watchlist-state').textContent = '保存失败，关注状态已恢复'; throw error; }
    finally { state.watchlistBusy = false; personalBusy(); }
  }
  function resetHoldingForm() { $('holdings-form').reset(); $('holdings-symbol').readOnly = false; $('save-holding').textContent = '保存持仓'; $('cancel-holding-edit').hidden = true; }
  function editHolding(symbol) {
    const item = state.holdings?.items?.find((row) => row.symbol === symbol); if (!item) return;
    $('holdings-symbol').value = symbol; $('holdings-symbol').readOnly = true;
    $('holdings-shares').value = item.shares; $('holdings-cost').value = item.average_cost;
    $('save-holding').textContent = '保存修改'; $('cancel-holding-edit').hidden = false; $('holdings-shares').focus();
  }
  async function changeHolding(symbol,values) {
    await ensurePersonal(); if (state.holdingsBusy) return;
    state.holdingsBusy = true; ++state.holdingsSequence; personalBusy(); $('holdings-state').textContent = '正在保存…';
    try {
      const data = await request(`/api/holdings/${encodeURIComponent(symbol)}`,values ? {method:'PUT',body:JSON.stringify(values)} : {method:'DELETE'});
      applyHoldings(data); if (values || $('holdings-symbol').value === symbol) resetHoldingForm();
      $('holdings-state').textContent = '已保存到本机服务'; notice(values ? '持仓已保存' : '持仓记录已删除');
    } catch (error) { $('holdings-state').textContent = '保存失败，原记录仍保留'; throw error; }
    finally { state.holdingsBusy = false; personalBusy(); }
  }
  async function watchlistTask(kind) {
    await refreshWatchlist(); const list = (state.watchlist?.items || []).map((item) => item.symbol);
    if (!list.length) { notice('请先添加关注股票'); return; }
    $(`${kind}-symbols`).value = list.join(', '); location.hash = kind; route();
    if (kind === 'backtest' && !$('backtest-start').value) { notice('已选择关注股票，请设置回测开始日期'); $('backtest-start').focus(); return; }
    if (kind === 'backtest' && !$('backtest-form').reportValidity()) return;
    const spec = {symbols:list,...dateSpec(kind),...researchSpec(kind),...(kind === 'backtest' ? {initial_cash:Number($('initial-cash').value)} : {})};
    await submitTask(kind,spec);
  }
  function modelObservationUnavailableReason(item) {
    const status = item.status || '';
    if (status === 'wyckoff_policy_initial_capital_contract_unavailable') return '模型按每股10万元初始资金训练，当前资金不适用';
    if (status === 'wyckoff_policy_dual_fallback_no_policy_head') return '当日回退到双专家，没有对应策略入场头';
    if (status.includes('checkpoint_unavailable_at_signal')) return '信号当天尚无可用检查点';
    if (status.includes('source_contract_unavailable')) return '数据来源与模型目标不匹配';
    if (status.includes('contract_unavailable')) return '当前入场或退出规则与模型目标不匹配';
    if (status.includes('market_input_missing') || status.includes('input_ineligible')) return '当日输入尚不具备评分资格';
    if (status.includes('model_unavailable')) return item.head === 'policy_entry' ? '当日策略入场头不可用' : '当日独立退出头不可用';
    return '当日模型或输入无法提供此目标的概率';
  }
  function evaluationWindows(result) {
    const evaluation = result?.evaluation;
    return Array.isArray(evaluation) ? evaluation : Object.values(evaluation?.windows || {});
  }
  function highReturnBacktestView(result) {
    if (!isHighReturn(result)) return result;
    if (isTrendStrategy(result)) return {...result,allocation:result.allocation || '每股固定独立初始资金；失败账户保留现金；合计资金曲线不代表共享现金账户',
      limitations:[...(result.limitations || []),'显式离线开发回放；未取得新独立窗口认证。']};
    const window = evaluationWindows(result)[0], full = window?.variants?.full || window?.variants?.reference;
    return {...result,metrics:result.metrics || full?.metrics || {},daily:result.daily || full?.daily || [],
      initial_cash:result.initial_cash || window?.initial_cash || result.evaluation?.initial_cash,
      allocation:'每股冻结10万元独立账户；完整股票池包含失败账户保留的现金',
      limitations:[...(result.limitations || []),'显式离线开发回放；未取得新独立窗口认证。']};
  }
  function backtestObservationApplication(item, result) {
    if (isTrendStrategy(result)) {
      const head = item.model_identity?.head || item.model_identity?.model_head || item.head;
      if (head === 'exit') return '二阶段退出观察，未用于成交';
      return item.applied ? '账户已应用' : '未用于成交或未触发';
    }
    return isHighReturn(result) ? item.applied ? '显式离线策略已应用' : '参考规则或未触发' : '影子，未用于成交';
  }
  function researchComparisonHtml(result) {
    const names = {ma:'均线独立路径',structure:'结构独立路径',wyckoff:'威科夫独立路径',reference:'三路并集参考',entry_only:'仅高收益入场',exit_only:'仅退出优势',full:'高收益入场＋退出'};
    const windows = evaluationWindows(result);
    if (!windows.length) return '';
    return `<table><caption>冻结股票池、每股固定独立资金的费用后账本对照；失败账户保留现金。</caption><thead><tr><th>窗口</th><th>策略</th><th>收益率</th><th>最大回撤</th><th>完整交易</th><th>高收益占比</th><th>入场评分覆盖</th><th>持仓评分覆盖</th></tr></thead><tbody>${windows.flatMap((window) => Object.entries(window.variants || {}).map(([arm,item]) => `<tr><td>${esc(window.fold?.name === 'explicit_offline' ? '显式离线回放' : window.fold?.name || '')}</td><td>${esc(names[arm] || arm)}</td><td>${percent(item.metrics?.total_return)}</td><td>${percent(item.metrics?.max_drawdown)}</td><td>${number(item.metrics?.completed_round_trips)}</td><td>${percent(item.metrics?.high_return_share)}</td><td>${percent(item.metrics?.entry_model_score_coverage)}</td><td>${percent(item.metrics?.exit_model_score_coverage)}</td></tr>`)).join('')}</tbody></table>`;
  }
  function renderBacktest() {
    const result = state.backtest;
    const trendStrategy = isTrendStrategy(result);
    $('backtest-observation-title').textContent = trendStrategy ? '趋势共识策略 · 账户入场与退出判断' : isHighReturn(result) ? '实际账户的高收益入场与退出政策判断' : '实际账户的独立入场与退出影子概率';
    $('backtest-observation-help').textContent = trendStrategy ? '高收益入场评分读取账户实际空仓资金与冻结风险计划，缺兼容模型不自动放行。高收益目标为完整交易费用后收益至少8%，且≥2倍冻结止损参考风险。成交采用滑点模拟均价；R按冻结止损参考假设估计。退出使用入场时固定的均线持仓退出模型、硬止损、冻结确认边界及卖方共识；二阶段退出模型仅作观察。最长持仓期限是冻结政策触发。' : isHighReturn(result) ? '显式离线策略：入场头读取真实空仓与冻结风险计划，退出头读取真实持仓前缀；符合冻结门槛时可改变离线账本。最大持仓期限属于固定政策触发，不是模型概率。' : '退出头读取实际持仓；策略入场头读取实际空仓现金及候选，目标是假设采用冻结模型退出政策的完整往返。普通回测继续执行规则，两类概率均不改变成交。';
    const observations = (result?.accounts || []).flatMap((account) => [
      ...(account.exit_policy_diagnostics || []).map((item) => ({...item,symbol:account.symbol,target:item.probability_target || '实际持仓：下一开盘退出优于冻结续持'})),
      ...(account.entry_gate_diagnostics || []).map((item) => ({...item,symbol:account.symbol,target:item.probability_target || '假设冻结退出政策：完整往返费用后正收益'}))
    ]).sort((a,b) => String(b.date).localeCompare(String(a.date)));
    $('backtest-model-observations').hidden = !observations.length;
    $('backtest-model-observation-rows').innerHTML = observations.slice(0,30).map((item) => `<tr><td>${esc(item.symbol)}</td><td>${esc(day(item.date))}</td><td>${esc(item.target)}</td><td>${item.forced_by_policy ? '冻结规则触发' : percent(item.probability)}</td><td>${backtestObservationApplication(item,result)}${item.probability == null ? ' · ' + esc(modelObservationUnavailableReason(item)) : ''}</td></tr>`).join('');
    renderComputeResult('backtest',result?.compute_info || (isHighReturn(result) && result.actual_inference_device ? {research:true,model_family:'current'} : undefined));
    $('backtest-research-result').hidden = !result?.model_gate && !result?.model_summary;
    if (trendStrategy) { $('backtest-research-result').hidden = false; $('backtest-research-result').innerHTML = `<p>趋势共识策略显式离线回放 · ${esc(result.model_run_id || '模型身份保存在运行报告')}。高收益入场缺兼容模型不自动放行；退出使用入场时固定的均线持仓退出模型与冻结风险、确认边界及卖方共识，二阶段退出模型仅作观察。费用后账本用于开发验证，尚未取得生产资格。</p><p>${esc(result.allocation || '')}${result.per_stock_initial_cash != null ? ` · 每股初始资金 ${number(result.per_stock_initial_cash,2)} 元` : ''}${result.initial_cash != null ? ` · 合计初始资金 ${number(result.initial_cash,2)} 元` : ''}</p>${researchComparisonHtml(result)}`; } else if (isHighReturn(result)) { $('backtest-research-result').hidden = false; $('backtest-research-result').innerHTML = `<p>高收益策略显式离线回放 · ${result.model_run_id || result.source_training_run_id || '模型身份保存在运行报告'} · 按当日可用检查点筛选三路独立候选并评估真实持仓退出；缺模型时执行冻结参考政策。费用后账本用于开发验证，尚未取得生产资格。</p>${researchComparisonHtml(result)}`; } else if (result?.model_summary) {
      const summary = result.model_summary, routes = result.model_routes || [];
      $('backtest-research-result').innerHTML = `<p>${esc(entryPolicyNames[result.entry_policy || result.request?.entry_policy || 'legacy'])}。${esc(entryPolicyCaption(result.entry_policy || result.request?.entry_policy || 'legacy'))}</p><p>${esc(usageNames[summary.usage_mode] || summary.usage_mode || '研究回测')} · ${summary.model_policy === 'pinned' ? '固定检查点' : '自动按日期'}。ML参与入场 ${number(summary.ml_applied_days)} 股票日 · 影子规则执行 ${number(summary.shadow_days)} 股票日 · 无模型规则执行 ${number(summary.no_model_days)} 股票日。${summary.ml_applied_days ? '只有逐日生效发布参与过滤。' : '本次组合规则执行，ML未参与入场。'}</p>${routes.length ? `<details class="model-routes"><summary>实际模型路由 ${number(routes.length)} 段</summary>${routes.slice(0,20).map((route) => `<p>${esc(route.symbol || '')} ${esc(route.start)}～${esc(route.end)} · ${number(route.days)} 日<br>${esc(modelResolutionText(route.identity || route))}</p>`).join('')}${routes.length > 20 ? '<p>此处展示前20段；完整路由保存在本次运行报告中。</p>' : ''}</details>` : ''}`;
    } else if (result?.model_gate) $('backtest-research-result').textContent = `${gateCaption(result.model_gate)} · ${result.model_run_id || '未使用学习模型'}${result.model_fold ? ` · 检查点 ${result.model_fold}（可用 ${result.model_available_at || '—'}）` : ''}。该旧记录未保存逐日发布与路由，请重跑获取模型实际作用。${result.model_gate.reason || ''}`;
    if (!result) {
      const task = state.tasks.find((item) => item.job_id === state.tracked.backtest);
      const failed = task?.status === 'failed', cancelled = task?.status === 'cancelled', active = activeStatuses.has(task?.status);
      const message = failed ? `本次回测失败：${display(task.error)}` : cancelled ? '本次回测已取消，没有生成可展示结果。' : active ? '回测正在运行，等待本次账本。' : '尚未运行回测，选择股票与时间范围后开始。';
      $('backtest-metrics').innerHTML = `<span class="subtle">${esc(message)}</span>`;
      $('trade-rows').innerHTML = `<tr><td colspan="8" class="empty-cell">${failed || cancelled ? '本次未生成可展示成交账本。' : '尚无本次运行的成交记录。'}</td></tr>`;
      $('trade-count').textContent = '0'; $('backtest-run').textContent = '';
      $('equity-empty').hidden = false; if (equitySeries) equitySeries.setData([]);
      $('equity-empty').querySelector('strong').textContent = failed ? '回测失败' : cancelled ? '回测已取消' : active ? '正在生成资金曲线' : '还没有回测结果';
      $('equity-empty').querySelector('span').textContent = failed ? '请根据上方原因调整参数后重试。' : cancelled ? '可以重新选择范围并提交回测。' : '选择股票池与时间范围，运行一条已冻结的规则。';
      $('backtest-rejections').hidden = true; $('backtest-caveats').hidden = true;
      pagination('trade',0,0);
      return;
    }
    const metrics = result.metrics || {};
    const fields = [['总收益',metrics.total_return ?? metrics.net_return,percent],['最大回撤',metrics.max_drawdown == null ? null : Math.abs(metrics.max_drawdown),percent],['期末资金',metrics.final_equity,(v) => number(v,2)],['成交笔数',metrics.trade_count ?? (result.trades || []).length,number]];
    $('backtest-metrics').innerHTML = fields.map(([label,value,format]) => `<div class="metric"><span>${label}</span><strong>${esc(format(value))}</strong></div>`).join('');
    $('backtest-run').textContent = result.run_id || '';
    const trades = result.trades || [];
    $('trade-count').textContent = number(trades.length);
    state.tradePage = Math.max(0,Math.min(state.tradePage,Math.ceil(trades.length / PAGE_SIZE) - 1));
    const page = trades.slice(state.tradePage * PAGE_SIZE,(state.tradePage + 1) * PAGE_SIZE);
    $('trade-rows').innerHTML = page.length ? page.map((trade,index) => `<tr><td>${esc(day(trade.date))}</td><td>${esc(trade.symbol)}</td><td><span class="badge ${actionClass(trade.action)}">${actionShape(trade.action,true)}${esc(actionLabel(trade.action))}成交</span></td><td class="numeric">${number(trade.price,3)}</td><td class="numeric">${number(trade.shares)}</td><td class="numeric">${number(trade.fee,2)}</td><td>${esc(day(trade.signal_date))}</td><td><button type="button" class="link-button" data-trade-index="${state.tradePage * PAGE_SIZE + index}">定位成交</button></td></tr>`).join('') : metrics.trade_count > 0 ? '<tr><td colspan="8" class="empty-cell">完整成交账本已保存于本次运行产物，当前未加载交易明细。</td></tr>' : '<tr><td colspan="8" class="empty-cell">该运行没有实际成交；请查看事件与未成交明细。</td></tr>';
    pagination('trade',trades.length,state.tradePage);
    renderDiagnostics('backtest-rejections',[...(result.rejections || []),...(result.failures || [])]);
    renderDiagnostics('backtest-caveats',result.limitations);
    renderEquity();
  }
  function renderEquity() {
    const rows = state.backtest?.daily || [];
    $('equity-empty').hidden = rows.length > 0;
    if (!rows.length) { if (equitySeries) equitySeries.setData([]); return; }
    if (!equityChart) {
      ensureChart();
      equityChart = LightweightCharts.createChart($('equity-chart'),chartOptions($('equity-chart')));
      equitySeries = equityChart.addLineSeries({color:colors().accent,lineWidth:2,priceLineVisible:false,lastValueVisible:true});
    }
    const initial = Number(state.backtest.initial_cash);
    document.querySelector('[data-equity-mode=equity]').textContent = initial > 0 ? '净值' : '资金';
    equitySeries.applyOptions({priceFormat:state.equityMode === 'drawdown' ? {type:'percent',precision:2,minMove:.01} : {type:'price',precision:3,minMove:.001}});
    equitySeries.setData(rows.filter((row) => row[state.equityMode] != null).map((row) => ({time:day(row.date),value:Number(row[state.equityMode]) * (state.equityMode === 'drawdown' ? 100 : initial > 0 ? 1 / initial : 1)})));
    equityChart.timeScale().fitContent(); resizeCharts();
  }
  function taskDetail(task) {
    const d = task.progress_detail;
    if (!d || typeof d !== 'object') return display(d || '');
    const info = task.compute_info, research = Boolean(task.spec?.plan_id) || task.spec?.research;
    const compute = info ? research ? ` · ${researchComputeCaption(info)}` : ` · ${info.cuda_work ? '已执行 CUDA 信号运算' : 'CPU 信号运算'}` : task.spec?.device ? ` · 所选设备 ${task.spec.device}（尚非 GPU 执行证明）` : '';
    return `${d.stage || ''}${d.current != null ? ` · ${number(d.current)} / ${number(d.total)}` : ''}${d.device && activeStatuses.has(task.status) ? ` · 当前阶段 ${d.device === 'cpu' ? 'CPU' : d.device}` : ''}${d.failed ? ` · 失败 ${number(d.failed)}` : ''}${compute}`;
  }
  function taskActions(task) {
    const cancel = activeStatuses.has(task.status) ? `<button type="button" data-cancel="${esc(task.job_id)}" ${task.status === 'cancelling' ? 'disabled' : ''}>取消</button>` : '';
    const result = task.status !== 'succeeded' ? '' : task.run_id ? `<button type="button" data-run-id="${esc(task.run_id)}" data-run-kind="${esc(task.kind || '')}">查看结果</button>` : task.result ? `<button type="button" data-task-result="${esc(task.job_id)}">查看结果</button>` : '';
    return cancel + result;
  }
  function renderTasks() {
    $('task-list').innerHTML = state.tasks.length ? state.tasks.map(task => `<article class="task-row"><div class="task-heading"><div><strong>${esc(kindNames[task.kind] || '当前研究任务')}</strong><small>${esc(task.created_at || '')} · ${esc(task.run_id || task.job_id)}</small></div><div class="task-actions"><span class="badge">${esc(statusNames[task.status] || task.status)}</span>${taskActions(task)}</div></div><div class="progress-track"><i style="width:${Math.max(0,Math.min(100,Number(task.progress) || 0))}%"></i></div><p class="task-detail ${task.error ? 'task-error' : ''}">${esc(task.error ? display(task.error) : taskDetail(task))}</p></article>`).join('') : '<div class="empty-copy">尚无任务，可发起训练、更新行情、扫描或回测。</div>';
    ['scan','backtest'].forEach(kind => { const task = state.tasks.find(item => item.kind === kind && item.job_id === state.tracked[kind]), host = $(kind + '-progress'); host.hidden = !task; if (task) host.innerHTML = `<div class="task-actions">${taskActions(task)}</div>${esc(statusNames[task.status] || task.status)} · ${number(task.progress)}%<br>${esc(task.error ? display(task.error) : taskDetail(task))}`; });
    ['scan','backtest','update'].forEach(kind => { const button = $(kind === 'update' ? 'update-data' : `start-${kind}`); button.disabled = state.tasks.some(task => task.kind === kind && activeStatuses.has(task.status)); });
  }
  function showResult(kind,result,navigate = false) {
    if (kind === 'scan') { state.scan = result; state.scanPage = 0; renderScan(); }
    else if (kind === 'backtest') { state.backtest = highReturnBacktestView(result); state.tradePage = 0; renderBacktest(); if (state.analysis) renderAnalysis(false); }
    else if (kind !== 'update') { globalThis.KHQuantTraining?.showResult(result); if (navigate) location.hash = 'training'; return; }
    if (navigate) location.hash = kind === 'update' ? 'data' : kind;
  }
  async function refreshTasks() {
    const sequence = ++state.taskSequence;
    clearTimeout(state.pollTimer);
    const data = await request('/api/tasks');
    if (sequence !== state.taskSequence) return;
    state.tasks = Array.isArray(data) ? data : data.tasks || [];
    renderTasks();
    if (state.tracked.backtest && !state.backtest) renderBacktest();
    for (const task of state.tasks) {
      if (task.status !== 'succeeded' || state.delivered.has(task.job_id) || task.job_id !== state.tracked[task.kind]) continue;
      state.delivered.add(task.job_id);
      try {
        if (task.kind === 'update') await refreshData();
        else {
          let data;
          if (task.kind === 'scan') {
            const job = await request(`/api/tasks/${encodeURIComponent(task.job_id)}?view=scan`);
            if (job.kind !== 'scan') throw new Error('扫描任务结果与当前任务身份不匹配');
            data = job.result || (task.run_id ? await request(runResultUrl(task.run_id,'scan')) : null);
          } else data = task.result || (task.run_id ? await request(runResultUrl(task.run_id,task.kind)) : null);
          if (!data) { state.delivered.delete(task.job_id); continue; }
          if (task.job_id !== state.tracked[task.kind]) continue;
          showResult(task.kind,data.result || data);
        }
        if (task.job_id !== state.tracked[task.kind]) continue;
        notice(`${kindNames[task.kind] || task.kind}已完成`);
        refreshRuns().catch(showError);
      } catch (error) { state.delivered.delete(task.job_id); if (task.job_id === state.tracked[task.kind]) showError(error); }
    }
    if (sequence === state.taskSequence && state.tasks.some((task) => activeStatuses.has(task.status))) state.pollTimer = setTimeout(() => refreshTasks().catch((error) => { showError(error); state.pollTimer = setTimeout(() => refreshTasks().catch(showError),5000); }),2500);
  }
  async function submitTask(kind,spec) {
    if (!['scan','backtest','update'].includes(kind)) throw new Error('该任务入口已退役，请使用模型训练页。');
    const button = $(kind === 'update' ? 'update-data' : `start-${kind}`); button.disabled = true;
    try { if (kind !== 'update' && computeLoaded) spec = {...spec,device:$(`${kind}-device`).value}; const task = await request('/api/tasks',{method:'POST',body:JSON.stringify({kind,spec})}); state.tracked[kind] = task.job_id;
      if (kind === 'scan') { state.scan = null; state.scanPage = 0; renderScan(); $('scan-coverage').textContent = '等待本次扫描结果'; $('scan-failures').hidden = true; }
      if (kind === 'backtest') { state.backtest = null; state.tradePage = 0; renderBacktest(); if (state.analysis) renderAnalysis(false); }
      notice(`${kindNames[kind]}任务已提交`); await refreshTasks();
    } catch (error) { showError(error); button.disabled = false; }
  }
  async function cancelTask(jobId) { await request(`/api/tasks/${encodeURIComponent(jobId)}/cancel`,{method:'POST',body:'{}'}); await refreshTasks(); notice('已请求取消任务'); }
  async function refreshRuns() {
    const data = await request('/api/runs?limit=30');
    const rows = Array.isArray(data) ? data : data.runs || [];
    $('run-list').innerHTML = rows.length ? rows.map((run) => `<article class="run-row"><div><strong>${esc(kindNames[run.kind] || run.kind || '分析运行')}</strong><small>${esc(run.run_id)} · ${esc(run.created_at || '')}</small></div><button type="button" data-run-id="${esc(run.run_id)}" data-run-kind="${esc(run.kind || '')}">打开结果</button></article>`).join('') : '<div class="empty-copy">尚无新系统运行记录。</div>';
  }
  function resultResearchContext(result) {
    const source = result?.request || result || {}, context = {};
    ['research','model_family','usage_mode','model_policy','model_run_id','model_fold','calendar_run_id','device','entry_policy'].forEach((key) => { if (source[key] != null) context[key] = source[key]; });
    if (isTrendStrategy(result)) {
      ['model_policy','model_fold','calendar_run_id'].forEach((key) => {
        if (context[key] == null && result[key] != null) context[key] = result[key];
      });
      const account = result.account_actions?.context || result.account_context || result.research?.account_context || {}, params = result.run_context?.config?.parameters || {};
      context.consensus_version = 'trend_consensus';
      context.model_family = 'current';
      context.model_run_id ||= result.research?.model_run_id || account.model_identity?.model_run_id || result.model_run_id;
      context.model_policy ||= context.model_fold ? 'pinned' : account.model_identity?.model_policy || 'auto';
      const config = source.consensus_config || result.consensus_config || result.trend_consensus?.config || account.consensus || params.consensus_config;
      if (config) context.consensus_config = config;
      const accountStart = source.account_start || result.account_start || account.account_start || params.account_start || (result.per_stock_initial_cash != null ? result.data_range?.start : null);
      const accountCash = source.account_initial_cash ?? result.account_initial_cash ?? account.initial_cash ?? params.initial_cash ?? result.per_stock_initial_cash;
      if (accountStart) context.account_start = accountStart;
      if (accountCash != null) context.account_initial_cash = accountCash;
    }
    if (!('research' in context)) context.research = Boolean(result?.research || result?.model_gate);
    if (context.research) { context.usage_mode ||= 'historical'; context.model_policy ||= context.model_run_id ? 'pinned' : 'auto'; context.entry_policy ||= result?.entry_policy || 'legacy'; }
    return context;
  }
  function applyResearchContext(prefix, context) {
    if (context.research !== false && ((context.model_family && context.model_family !== 'current') || (context.consensus_version && context.consensus_version !== 'trend_consensus'))) throw new Error('该历史模型运行已退役，只能查看归档证据，不能恢复当前操作上下文。');
    if (context.research !== false && context.usage_mode && !['historical','forward'].includes(context.usage_mode)) throw new Error('旧用途已退役，不恢复资格；请重新选择当前研究用途。');
    if (context.model_run_id) state.pendingResearchContexts[prefix] = context;
    else delete state.pendingResearchContexts[prefix];
    $(`${prefix}-model`).value = '';
    $(`${prefix}-mode`).value = context.research === false ? 'structure' : 'research';
    $(`${prefix}-model-family`).value = 'current';
    populateResearchModels(prefix);
    $(`${prefix}-entry-policy`).value = context.entry_policy || 'legacy';
    $(`${prefix}-usage-mode`).value = context.usage_mode || 'historical'; $(`${prefix}-model-policy`).value = context.model_policy || 'auto';
    if (context.model_run_id) $(`${prefix}-model`).value = context.model_run_id;
    if (context.device) $(`${prefix}-device`).value = context.device;
    updateResearchControls(prefix);
    // Restored checkpoints must be selected after the model's options exist.
    if (trendInputSelected(prefix) && $(`${prefix}-model-policy`).value !== 'pinned') $(`${prefix}-checkpoint`).value = '';
    else if (context.model_fold || $(`${prefix}-model-policy`).value === 'pinned') $(`${prefix}-checkpoint`).value = context.model_fold || '';
    renderCheckpointHelp(prefix);
  }
  function runResultUrl(id, kind = '') {
    return `/api/runs/${encodeURIComponent(id)}${kind === 'scan' ? '?view=scan' : ''}`;
  }
  async function openRun(id, kindHint = '') {
    resetChartSelection();
    const sequence = ++openRunSequence, analysisAtStart = state.analysisSequence;
    let data;
    try { data = await request(runResultUrl(id,kindHint)); }
    catch (error) { if (sequence !== openRunSequence) return; throw error; }
    if (sequence !== openRunSequence) return;
    const result = data.result || data;
    const kind = data.kind || (result.model_manifest && result.evaluation ? 'model_training' : result.trades ? 'backtest' : result.rows ? 'scan' : result.data_status ? 'update' : 'analysis');
    if (kind !== 'analysis') { showResult(kind,result,true); return; }
    // A saved analysis replaces only the generation that existed when opened.
    // Other run types keep a pending analysis intact; a newer analysis wins.
    if (analysisAtStart !== state.analysisSequence) return;
    ++state.analysisSequence;
    resetChartSelection(); analysisLoading = false;
    $('analysis-form').querySelector('[type=submit]').disabled = false;
    state.analysis = result;
    storeHistoryEvidence(evidenceUI()?.mergeEvidenceRows(displayedEvidence()?.rows,displayedConsensus()?.rows) || displayedEvidence()?.rows || [],false,result.account_actions?.series_context_hash); historyPage = 0;
    state.query = {...(result.request || {symbol:result.symbol,start:null,end:result.data_end,as_of:result.as_of}),...resultResearchContext(result),...consensusRequestContext(result)};
    applyResearchContext('analysis',state.query);
    $('analysis-symbol').value = state.query.symbol || result.symbol; $('analysis-start').value = state.query.start || ''; $('analysis-end').value = state.query.end || result.data_end;
    setDateMode('analysis','historical');
    state.dates = (result.frequencies?.['日线']?.bars || []).map((bar) => day(bar.time)); state.replayIndex = Math.max(0,state.dates.length - 1);
    location.hash = 'analysis'; showView('analysis'); renderAnalysis(); renderEvents(); updateReplay();
  }
  async function analyzeSymbol(symbol,end,source = null) {
    if (source?.scan_view?.compact && isTrendStrategy(source) && source.scan_view.account_context_status !== 'consistent') {
      throw new Error('此扫描无法核验一致的研究账户来源，请检查原扫描报告后再钻取。');
    }
    $('analysis-symbol').value = symbol;
    location.hash = 'analysis';
    const context = source ? resultResearchContext(source) : researchSpec('analysis');
    if (source) { applyResearchContext('analysis',context); if (source.request?.start) $('analysis-start').value = source.request.start; }
    if (end) {
      setDateMode('analysis','historical'); $('analysis-end').value = end;
      if ($('analysis-start').value > end) {
        const starts = [source?.data_range?.start,source?.market_history?.window_start,end].filter((value) => value && value <= end).sort();
        $('analysis-start').value = starts[0] || end;
      }
    }
    else if (dateModes.analysis === 'latest') await refreshLatestDates();
    await analyze({symbol,...dateSpec('analysis'),...context,...(end ? {as_of:end} : {})},false,end || null);
  }
  function bind() {
    ['analysis','scan','backtest'].forEach((prefix) => $(`${prefix}-end`).addEventListener('change',() => setDateMode(prefix,$(`${prefix}-end`).value ? 'historical' : 'latest')));
    $('analysis-latest')?.addEventListener('click',async () => { setDateMode('analysis','latest'); updateChartSelection('clear'); try { await refreshLatestDates(); } catch (error) { showError(error); } });
    $('analysis-symbol').addEventListener('change',() => { if (dateModes.analysis === 'latest') refreshLatestDates().catch(showError); });
    window.addEventListener('hashchange',route);
    window.addEventListener('khquant-selection-updated',() => refreshResearch().catch(showError));
    window.addEventListener('keydown',(event) => { if (event.key === 'Escape') updateChartSelection('clear'); });
    $('chart-evidence-reset').addEventListener('click',() => updateChartSelection('clear'));
    $('theme-toggle').addEventListener('click',() => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
    $('page-error').querySelector('button').addEventListener('click',() => { $('page-error').hidden = true; });
    $('analysis-symbol').addEventListener('input',() => { updateChartSelection('clear'); clearTimeout(searchTimer); searchTimer = setTimeout(() => searchSymbols($('analysis-symbol').value).catch(showError),250); });
    $('analysis-form').addEventListener('submit',async (event) => { event.preventDefault(); try { if (dateModes.analysis === 'latest') await refreshLatestDates(); await analyze({symbol:getSymbol($('analysis-symbol').value),...dateSpec('analysis'),...researchSpec('analysis')}); } catch (error) { showError(error); } });
    $('send-backtest').addEventListener('click',() => { try { const symbol = getSymbol($('analysis-symbol').value); if (!symbol) { notice('请先选择股票'); return; } const context = researchSpec('analysis'); $('backtest-symbols').value = symbol; ['start','end'].forEach((key) => { $(`backtest-${key}`).value = $(`analysis-${key}`).value; }); setDateMode('backtest','historical'); applyResearchContext('backtest',context); location.hash = 'backtest'; } catch (error) { showError(error); } });
    document.querySelectorAll('[data-frequency]').forEach((button) => button.addEventListener('click',() => { resetChartSelection(); state.frequency = button.dataset.frequency; document.querySelectorAll('[data-frequency]').forEach((item) => item.setAttribute('aria-pressed',String(item === button))); renderAnalysis(); }));
    ['ma','boll','chips','fractals','pens','zones','divergences','signals','directions','wyckoff','trades','volume'].forEach((name) => $(`layer-${name}`).addEventListener('change',() => { resetChartSelection(); renderAnalysis(false); }));
    ['start','end','mode','model-family','entry-policy','usage-mode','model-policy','model','checkpoint','device'].forEach((name) => $(`analysis-${name}`).addEventListener('change',() => updateChartSelection('clear')));
    ['chart-range-start','chart-range-end'].forEach((id) => $(id).addEventListener('change',() => updateChartSelection('clear')));
    $('chart-range-apply').addEventListener('click',() => { updateChartSelection('clear'); applyChartRange(); });
    $('chart-zoom-in').addEventListener('click',() => zoomChart(.7)); $('chart-zoom-out').addEventListener('click',() => zoomChart(1.4));
    $('chart-range-reset').addEventListener('click',() => { if (chart) chart.timeScale().fitContent(); });
    $('replay-slider').addEventListener('change',() => replay(Number($('replay-slider').value)));
    $('replay-prev').addEventListener('click',() => replay(state.replayIndex - 1)); $('replay-next').addEventListener('click',() => replay(state.replayIndex + 1));
    $('replay-date').addEventListener('change',() => { const date = $('replay-date').value; const index = state.dates.findLastIndex((value) => value <= date); if (index < 0) notice('请选择可用行情范围内的日期'); else replay(index); });
    $('replay-reset').addEventListener('click',() => { if (!state.query) return; const {as_of:_,...query} = state.query; state.replayIndex = Math.max(0,state.dates.length - 1); analyze(query,true); });
    $('scan-form').addEventListener('submit',(event) => { event.preventDefault(); try { submitTask('scan',{symbols:symbols($('scan-symbols').value),...dateSpec('scan'),...researchSpec('scan')}); } catch(error) { showError(error); } });
    $('backtest-form').addEventListener('submit',(event) => { event.preventDefault(); try { const spec = {symbols:symbols($('backtest-symbols').value),...dateSpec('backtest'),...researchSpec('backtest'),initial_cash:Number($('initial-cash').value)}; submitTask('backtest',spec); } catch(error) { showError(error); } });
    ['analysis','scan','backtest'].forEach((prefix) => {
      ['mode','model-family','usage-mode','model-policy','model','entry-policy'].forEach((name) => $(`${prefix}-${name}`).addEventListener('change',() => {
        if (name === 'model-family' || name === 'model' && $(`${prefix}-model`).value !== state.pendingResearchContexts[prefix]?.model_run_id) delete state.pendingResearchContexts[prefix];
        if (name === 'model-family') populateResearchModels(prefix);
        updateResearchControls(prefix);
        if (prefix === 'scan') renderScanComputePlan();
      }));
      $(`${prefix}-checkpoint`).addEventListener('change',() => renderCheckpointHelp(prefix));
      $(`${prefix}-${prefix === 'backtest' ? 'start' : 'end'}`).addEventListener('change',() => renderCheckpointHelp(prefix));
    });
    $('refresh-research').addEventListener('click',() => refreshResearch().catch(showError));
    ['scan-search','scan-filter'].forEach((id) => $(id).addEventListener(id === 'scan-search' ? 'input' : 'change',async () => {
      state.scanPage = 0; renderScan();
      const query = $('scan-search').value.trim();
      if (id === 'scan-search' && query && (state.scan?.rows || []).some((row) => !row.name)) {
        try { await searchSymbols(query,false); if ($('scan-search').value.trim() === query) renderScan(); }
        catch (error) { showError(error); }
      }
    }));
    ['scan','trade'].forEach((prefix) => ['prev','next'].forEach((direction) => $(`${prefix}-${direction}`).addEventListener('click',() => { state[prefix === 'scan' ? 'scanPage' : 'tradePage'] += direction === 'prev' ? -1 : 1; prefix === 'scan' ? renderScan() : renderBacktest(); })));
    document.querySelectorAll('[data-equity-mode]').forEach((button) => button.addEventListener('click',() => { state.equityMode = button.dataset.equityMode; document.querySelectorAll('[data-equity-mode]').forEach((item) => item.setAttribute('aria-pressed',String(item === button))); renderEquity(); }));
    $('refresh-data').addEventListener('click',() => refreshData().catch(showError)); $('refresh-tasks').addEventListener('click',() => refreshTasks().catch(showError)); $('refresh-runs').addEventListener('click',() => refreshRuns().catch(showError));
    ['analysis','scan','backtest','data'].forEach((kind) => $(`${kind}-device`).addEventListener('change',() => { if (kind === 'scan') renderScanComputePlan(); refreshCompute(kind,$(`${kind}-device`).value).catch(showError); }));
    $('refresh-compute').addEventListener('click',() => refreshCompute('data',$('data-device').value).catch(showError));
    $('update-form').addEventListener('submit',(event) => { event.preventDefault(); const end = $('update-end').value; submitTask('update',{symbols:symbols($('update-symbols').value),...(end ? {end} : {})}); });
    ['watchlist-symbol','holdings-symbol'].forEach((id) => $(id).addEventListener('input',() => { clearTimeout(searchTimer); searchTimer = setTimeout(() => searchSymbols($(id).value).catch(showError),250); }));
    $('watchlist-form').addEventListener('submit',(event) => { event.preventDefault(); try { changeWatchlist(personalSymbol($('watchlist-symbol').value),false).then(() => { $('watchlist-symbol').value = ''; }).catch(showError); } catch (error) { showError(error); } });
    $('refresh-watchlist').addEventListener('click',() => refreshWatchlist().catch(showError));
    $('refresh-holdings').addEventListener('click',() => refreshHoldings().catch(showError));
    $('scan-watchlist').addEventListener('click',() => watchlistTask('scan').catch(showError));
    $('backtest-watchlist').addEventListener('click',() => watchlistTask('backtest').catch(showError));
    $('cancel-holding-edit').addEventListener('click',resetHoldingForm);
    $('holdings-form').addEventListener('submit',(event) => {
      event.preventDefault();
      try {
        const symbol = personalSymbol($('holdings-symbol').value), shares = Number($('holdings-shares').value), average_cost = Number($('holdings-cost').value);
        if (!Number.isSafeInteger(shares) || shares <= 0 || shares > 1e9) throw new Error('持仓股数须为 1 至 10 亿之间的整数。');
        if (!Number.isFinite(average_cost) || average_cost <= 0 || average_cost > 1e9) throw new Error('平均成本须为大于零、不超过 10 亿元的有效数值。');
        changeHolding(symbol,{shares,average_cost}).catch(showError);
      } catch (error) { showError(error); }
    });
    document.addEventListener('click',(event) => {
      const button = event.target.closest('button'); if (!button) return;
      if (button.hasAttribute('data-chart-evidence-close')) updateChartSelection('clear');
      if (button.dataset.evidenceDate) { chartSelection = evidenceUI().selectionTransition(chartSelection,{type:'date',date:button.dataset.evidenceDate}); renderChartPopup(); drawChipProfile(); renderIndicatorReadout(chartSelection.pinned); syncChartRange(); $('chart-selection-status').textContent = `已固定原始日期 ${button.dataset.evidenceDate} · 筹码截至 ${selectedChipProfile()?.as_of || '—'}`; }
      if (button.dataset.evidenceLocate) { highlightEvent(button.dataset.evidenceLocate); const entry = [...document.querySelectorAll('[data-event-time]')].find((item) => item.dataset.eventTime === button.dataset.evidenceLocate || item.dataset.evidenceDate === button.dataset.evidenceLocate); entry?.scrollIntoView({behavior:'smooth',block:'nearest'}); }
      if (button.dataset.evidenceReplay || button.dataset.historyDate) { const time = button.dataset.evidenceReplay || button.dataset.historyDate; const index = state.dates.findIndex((item) => item === time); if (index >= 0 && !analysisLoading) replay(index).catch(showError); }
      if (button.dataset.historyPage != null && !analysisLoading) { historyPage = Number(button.dataset.historyPage); renderReplayEvidence(); }
      if (button.dataset.favorite) changeWatchlist(button.dataset.favorite,state.favorites.has(button.dataset.favorite)).catch(showError);
      if (button.dataset.removeWatchlist) changeWatchlist(button.dataset.removeWatchlist,true).catch(showError);
      if (button.dataset.editHolding) editHolding(button.dataset.editHolding);
      if (button.dataset.removeHolding) changeHolding(button.dataset.removeHolding,null).catch(showError);
      if (button.dataset.analyzeSymbol) analyzeSymbol(button.dataset.analyzeSymbol,button.dataset.analyzeEnd,button.dataset.analysisSource === 'scan' ? state.scan : null).catch(showError);
      if (button.dataset.tradeIndex != null) { const trade = state.backtest?.trades?.[Number(button.dataset.tradeIndex)]; if (trade) analyzeSymbol(trade.symbol,day(trade.date),state.backtest).catch(showError); }
      if (button.dataset.locateEvent) locateTime(button.dataset.locateEvent);
      if (button.dataset.cancel) cancelTask(button.dataset.cancel).catch(showError);
      if (button.dataset.taskResult) { const task = state.tasks.find((item) => item.job_id === button.dataset.taskResult); if (task?.result) showResult(task.kind,task.result,true); }
      if (button.dataset.runId) openRun(button.dataset.runId,button.dataset.runKind || '').catch(showError);
    });
  }
  async function init() {
    const theme = localStorage.getItem('khquant_czsc_theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); setTheme(theme);
    bind(); route();
    refreshLatestDates().catch(showError);
    const results = await Promise.allSettled([refreshData(),searchSymbols(),refreshTasks(),refreshRuns(),ensurePersonal(),refreshCompute(),refreshResearch()]);
    results.forEach((result) => { if (result.status === 'rejected') showError(result.reason); });
    if (results[0].status === 'rejected') { $('connection-state').textContent = '数据检查失败'; $('connection-state').className = 'status-dot failed'; $('data-detail').textContent = '数据状态请求失败，请检查服务后刷新。'; }
    if (results[1].status === 'fulfilled' && results[1].value.length && !$('analysis-symbol').value) { $('analysis-symbol').value = results[1].value[0].symbol; refreshLatestDates().catch(showError); }
  }
  init().catch(showError);
})();
