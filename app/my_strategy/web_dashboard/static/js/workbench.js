/* CZSC workbench. Only causal service responses become charts and reference signals. */
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
  const statusNames = {queued:'等待开始', running:'运行中', succeeded:'已完成', failed:'失败', cancelling:'正在取消', cancelled:'已取消',partial:'部分完成'};
  const kindNames = {analysis:'结构分析',scan:'全市场扫描',update:'行情与特征更新',board_features:'板块特征计算',board_sync:'板块与结构同步',board_knowledge:'当前板块成员更新',model_training:'完整模型训练',model_validation:'已完成模型复核',current_train:'完整模型训练'};
  const PAGE_SIZE = 100;
  const state = {analysis:null, query:null, frequency:'日线', dates:[], replayIndex:0, scan:null, scanPage:0, tasks:[], delivered:new Set(), tracked:{}, favorites:new Set(), symbolCatalog:new Map(), analysisSequence:0, taskSequence:0, pollTimer:null, watchlist:null, personalLoaded:false, watchlistBusy:false, watchlistSequence:0,research:null,pendingResearchContexts:{}};
  let chart, candles, volume;
  let penSeries = [];
  let indicatorSeries = [];
  let resizeObserver;
  let searchTimer;
  let noticeTimer;
  let personalReady;
  let computeLoaded = false;
  let latestDefaultEnd;
  const dateModes = {analysis:'latest',scan:'latest'};
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
  const referenceMode = (...records) => evidenceUI()?.referenceMode(...records) === true;
  const displayedConsensus = () => state.analysis?.trend_consensus || state.analysis?.consensus_decisions;
  const displayedEvidence = () => state.analysis?.trend_consensus || referenceMode(state.analysis) ? evidenceUI()?.referenceEvidence(state.analysis) : state.analysis?.chart_evidence;
  function storeHistoryEvidence(incoming, preserveDates, seriesContext) {
    const preserve = preserveDates && (!seriesContext || seriesContext === historySeriesContext);
    historyRows = preserve ? [...new Map([...historyRows,...incoming].map((row) => [row.date,row])).values()] : incoming;
    historySeriesContext = seriesContext || null;
  }
  let priceOverlayFrame = null;
  const computeSequences = {analysis:0,scan:0,data:0};
  const usageNames = {historical:'历史滚动研究',forward:'当前时点研究'};


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
    const selected = ['analysis', 'scan', 'watchlist', 'training', 'parameters', 'data'].includes(name) ? name : 'analysis';
    if (selected !== 'analysis') resetChartSelection();
    showView(selected);
    requestAnimationFrame(resizeCharts);
    if (['training','parameters'].includes(selected)) globalThis.KHQuantTraining?.refresh().catch(showError);
    if (selected === 'data') { refreshTasks().catch(showError); refreshRuns().catch(showError); }
    if (['analysis','scan'].includes(selected)) refreshResearch().catch(showError);
    if (selected === 'scan') globalThis.KHQuantBoardPanel?.onRoute();
    if (selected === 'watchlist') (state.personalLoaded ? refreshWatchlist() : ensurePersonal()).catch(showError);
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
  }
  function resizeCharts() {
    if (chart && !$('view-analysis').hidden) chart.resize(Math.max(1, $('analysis-chart').clientWidth), $('analysis-chart').clientHeight);
    refreshChartOverlays();
    if (chartSelection.pinned) renderChartPopup();
  }
  function setTheme(theme) {
    document.documentElement.dataset.theme = theme;
    $('theme-toggle').textContent = theme === 'dark' ? '明色' : '暗色';
    localStorage.setItem('khquant_czsc_theme', theme);
    const c = colors();
    if (chart) { chart.applyOptions(chartOptions($('analysis-chart'))); candles.applyOptions({upColor:c.up,downColor:c.down,borderUpColor:c.up,borderDownColor:c.down,wickUpColor:c.up,wickDownColor:c.down}); renderAnalysis(false); }
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
    const reference = referenceMode(state.analysis);
    const referenceMarkers = evidenceUI()?.referenceMarkerInputs(state.analysis,symbol) || {events:[]};
    const selectedEvents = referenceMarkers.events.filter((event) => event.reference_layer === 'source' ? $('layer-sources').checked : $('layer-signals').checked);
    const referenceMarkersBuilt = KHQuantMarkers.build({bars,dailyBars:normalizeBars(state.analysis.frequencies?.['日线']?.bars),frequency:state.frequency,
      events:selectedEvents,symbol,consensusVersion:displayedConsensus()?.version || 'reference_position_v1',start:state.query?.start || '',asOf:day(state.query?.as_of || state.analysis.as_of || state.analysis.data_end),up:c.up,down:c.down});
    markers.push(...referenceMarkersBuilt.markers);
    const wyckoff = KHQuantWyckoffMarkers.buildDetailed({events:$('layer-wyckoff').checked ? state.analysis.wyckoff_events || [] : [],bars,frequency:state.frequency,symbol,start:state.query?.start || '',asOf:day(state.query?.as_of || state.analysis.as_of || state.analysis.data_end)});
    const wyckoffMarkers = wyckoff.markers;
    markers.push(...wyckoffMarkers);

    markerMappings = referenceMarkersBuilt.mappings;
    markerGroups = [...(referenceMarkersBuilt.groups || []),...wyckoff.groups];
    $('marker-details').textContent = KHQuantMarkers.describe(markerMappings,null,state.frequency);
    const sourceEvents = referenceMarkersBuilt.mappings.filter((event) => event.reference_layer === 'source' && event.displayed), finalEvents = referenceMarkersBuilt.mappings.filter((event) => event.reference_layer !== 'source' && event.displayed);
    const count = (items,side) => items.filter((item) => item.action === side).length;
    const finalSignalName = state.analysis.strategy_version === 'reference_position_v1' ? '目标变化' : '最终共识';
    $('marker-summary').textContent = `来源 B ${count(sourceEvents,'BUY')} / S ${count(sourceEvents,'SELL')} · ${finalSignalName} B ${count(finalEvents,'BUY')} / S ${count(finalEvents,'SELL')} · 周期标记 ${referenceMarkersBuilt.intentMarkerCount} 个${referenceMarkersBuilt.formingProjectedCount ? ` · ${referenceMarkersBuilt.formingProjectedCount} 个日线参考点投影于形成中周期` : ''} · 点击查看来源及当日依据`;
    markers.sort((a,b) => a.time.localeCompare(b.time)); candles.setMarkers(markers);
    $('consensus-semantics').textContent = state.analysis.strategy_version === 'reference_position_v1' ? '结B/S、量B/S为已确认来源参考；参B/S为每股参考目标的增加或减少。来源点与目标变化独立显示，参考价取信号日收盘。' : '结B/S、量B/S、均B/S为已确认来源参考；共B/S为达原门槛的最终共识。参考价取信号日收盘，支持分不代表盈利概率。';
    if (fit) chart.timeScale().fitContent();
    $('chart-symbol').textContent = `${state.analysis.symbol || symbol || ''} ${state.analysis.name || state.symbolCatalog.get(state.analysis.symbol)?.name || ''}`.trim();
    const formingCount = bars.filter((bar) => !bar.is_closed).length;
    $('chart-information').textContent = `${state.frequency} · ${bars.length} 根${formingCount ? `（${formingCount} 根淡色形成中，未用于规则）` : ''} · 可知截至 ${state.query?.as_of || state.analysis.as_of || state.analysis.data_end || last} · ${basisLabel(state.analysis.price_basis)}`;
    $('analysis-state').textContent = state.analysis.warmup?.eligible === false ? reference ? '历史不足，参考判断不可用' : '预热不足，不能产生有效交易' : state.query?.as_of ? `回放至 ${state.query.as_of}` : '最新可用结构';
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

  function popupGroupsAt(time) {
    const groups = groupsAt(time).filter((group) => group.visible !== false && (['wyckoff','intention','source_reference'].includes(group.kind)));
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
    if (referenceMode(event)) return evidenceUI()?.opinionReasons(event) || event.reference_recommendation?.basis || '按当日可见证据提供参考方向';
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
    if (referenceMode(event)) return ({BUY:'B · 买方参考',SELL:'S · 卖方参考',WAIT:'趋势／结构／量价观察'})[event.reference_action || event.action || event.side] || '趋势／结构／量价参考';
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
  function isHighReturn(...records) { return records.some((record) => record?.model_family === 'current' || record?.research?.model_family === 'current' || record?.model_resolution?.model_family === 'current' || ['trend_consensus','reference_position_v1'].includes(record?.strategy_version)); }




  function trendInputSelected(prefix) { return $(`${prefix}-mode`).value === 'research' && $(`${prefix}-model-family`).value === 'current'; }
  function modelName(...records) { return !records.length || records.some(record => record?.strategy_version === 'reference_position_v1' || record?.position_reference?.schema === 'reference_position_v1') ? '参考买卖点与每股仓位' : '趋势共识完整模型包'; }
  function modelInputUnavailable(resolution, row = {}) {
    if (resolution?.status === 'input_veto' || resolution?.reason_codes?.includes('model_input_ineligible')) return true;
    const explicit = resolution?.model_input_eligible ?? row.model_input_eligible;
    if (explicit != null) return explicit === false;
    return resolution?.input_eligible === false || row.input_eligible === false;
  }










  function nextSessionGuidance(decision, requestedAsOf) {
    if (!decision) return '参考判断不可用，请重新分析。';
    const action = decision.reference_action || decision.action || 'WAIT';
    const label = decision.position_reference?.schema === 'reference_position_v1' ? {BUY:'参B · 参考增仓',SELL:'参S · 参考减仓',WAIT:'参考目标未变化',HOLD:'参考目标未变化'}[action] || '参考未评分' : {BUY:'共B · 最终买方参考',SELL:'共S · 最终卖方参考',WAIT:'继续观察',HOLD:'继续观察'}[action] || '继续观察';
    const referenceDate = day(decision.reference_price_date || decision.date || state.analysis.data_end);
    const requested = day(requestedAsOf);
    const stale = requested && referenceDate && referenceDate < requested ? `<p>行情截至 ${esc(referenceDate)}，未覆盖请求日期 ${esc(requested)}。</p>` : '';
    return `${stale}<strong>${esc(referenceDate)} · <span class="badge ${actionClass(action)}">${esc(label)}</span></strong><p>${esc(referencePriceText(decision.reference_close ?? decision.reference_price,referenceDate,'收盘参考价'))}</p><p>依据：${esc(decision.basis || decision.reference_recommendation?.basis || evidenceUI()?.opinionReasons(decision) || '方向支持不足，继续观察')}</p>`;
  }
  function renderEvents() {
    const reference = referenceMode(state.analysis);
    const events = reference ? evidenceUI()?.referenceMarkerInputs(state.analysis).events || [] : state.analysis?.events || [];
    $('event-count').textContent = String(events.length);
    $('position-reference-card').innerHTML = evidenceUI()?.positionHtml(state.analysis) || '';
    $('event-context').innerHTML = nextSessionGuidance(state.analysis?.next_session_decision,state.query?.as_of || state.analysis?.as_of);
    const research = state.analysis?.research;
    $('analysis-research-evidence').hidden = !research;
    $('analysis-research-evidence').innerHTML = research ? researchEvidence({...research,...state.analysis.next_session_decision}) : '';
    if (state.analysis?.compute_info) $('analysis-compute-status').textContent = research ? researchComputeCaption(state.analysis.compute_info,state.analysis) : computeCaption(state.analysis.compute_info);
    $('event-list').innerHTML = events.length ? [...events].reverse().map((event) => `<article class="event-entry" data-event-time="${esc(day(event.time || event.date))}" data-evidence-date="${esc(day(event.evidence_date || event.time || event.date))}"><strong>${esc(event.reference_layer === 'source' ? `${event.source_label || event.source}来源 ${event.action === 'BUY' ? 'B' : 'S'}` : event.reference_display_label || `最终共识 ${event.action === 'BUY' ? 'B' : 'S'}`)}</strong><p><span class="badge ${actionClass(event.action)}">${actionShape(event.action)}${esc(actionLabel(event.action))}${reference ? '参考' : '意图'}</span></p><p class="event-reference">${esc(referencePriceText(event.reference_price,event.reference_price_date,`${actionLabel(event.action)}参考价`))}</p><p>${event.reference_layer === 'source' ? esc(event.source_relationship_text || `已确认来源证据 · ${event.consensus_qualified ? '同日最终共识通过' : '同日最终共识未通过；来源点独立保留'}`) : esc(readableReason(event))}</p><p class="event-time">发生 ${esc(display(event.time || event.date))}<br>可用 ${esc(display(event.available_at))}</p><button type="button" data-locate-event="${esc(day(event.evidence_date || event.time || event.date))}">定位主图</button></article>`).join('') : '<div class="empty-copy">该时点没有规则事件。形成中的结构不代表确认信号。</div>';
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
    query = {...query,analysis_mode:'reference'};

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
      storeHistoryEvidence(incoming,preserveDates,JSON.stringify([data.symbol,query.start,query.model_policy,query.model_run_id,query.model_fold,query.consensus_config,query.current_position_fraction]));
      if (!preserveDates) { state.dates = (data.frequencies?.['日线']?.bars || []).map((bar) => day(bar.time)); state.replayIndex = Math.max(0,state.dates.length - 1); }
      renderAnalysis(); renderEvents(); updateReplay();
      if (locate) locateTime(locate,false);
    } catch (error) {
      if (sequence !== state.analysisSequence) return;
      state.analysis = null; analysisLoading = false;
      $('position-reference-card').innerHTML = evidenceUI()?.positionHtml(state.analysis) || '';
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
    $('replay-evidence-card').innerHTML = state.analysis ? `<strong>信息截止 ${esc(selected || '—')}</strong>${ui.consensusHtml(decision,displayedConsensus()?.config,displayedConsensus()?.version)}${ui.referenceHtml(row)}${ui.agentsHtml(row,false,decision,displayedConsensus()?.config,displayedConsensus()?.version)}${ui.positionHtml(row)}${ui.rawEventsHtml(row,[],state.analysis?.consensus_decisions?.version)}<p class="subtle">参考价使用该日行情；评分按当时可见的模型与事件展示。</p>` : '<p class="subtle">尚无成功分析结果，参考价与评分不可用。</p>';
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
    if (latest && !latestCoverage) { const next = day(latest); ['analysis','scan'].forEach((prefix) => { if (dateModes[prefix] === 'latest') $(`${prefix}-end`).value = next; }); latestDefaultEnd = next; }
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
    ['scan'].forEach((prefix) => { if (dateModes[prefix] === 'latest' && latestDefaultEnd) $(`${prefix}-end`).value = latestDefaultEnd; });
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
      ['analysis','scan','data'].forEach((prefix) => { $(`${prefix}-device`).innerHTML = options; if ([...$(`${prefix}-device`).options].some((option) => option.value === selected)) $(`${prefix}-device`).value = selected; $(`${prefix}-compute-status`).textContent = computeCaption(data,true); });
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
    return `${name} · ${work}${info.model_inference_batches != null ? ` · ${number(info.model_inference_batches)} 个推理批次` : ''}${info.model_loads != null ? ` · 模型加载 ${number(info.model_loads)} 次` : ''}${info.cpu_workers != null ? ` · CPU 准备并发上限 ${number(info.cpu_workers)}` : ''}${info.cache_hits != null ? ` · 特征缓存命中 ${number(info.cache_hits)}` : ''}${timings ? ` · ${timings}` : ''}${expertWork}。GPU 工作仅依据本次执行记录；${referenceMode(result) ? '结构与量价参考在 CPU。' : '结构识别与实际成交在 CPU。'}`;
  }
  function renderComputeResult(kind, info) { $(`${kind}-compute-result`).hidden = !state[kind]; $(`${kind}-compute-result`).textContent = info?.research ? researchComputeCaption(info,state[kind]) : computeCaption(info); }
  function renderScanComputePlan() {
    const research = $('scan-mode').value === 'research', device = $('scan-device').value;
    $('scan-compute-plan').textContent = !research ? '结构识别在 CPU；CUDA 可执行批量信号运算，MPS 的结构价格比较保持 CPU float64 精度。' : `CPU 准备结构与量价特征，按所选日期匹配模型并批量推理（设备 ${device === 'auto' ? '自动选择' : device}）；缺少合格趋势模型时保持未评分。真实推理行数、批次和 GPU 工作以本次结果为准。`;
  }
  function gateCaption(gate) { return gate?.passed ? '样本外门槛通过' : '影子观察：尚未证实改善'; }
  function researchModels(prefix) {
    const capability = {analysis:'chart',scan:'scan'}[prefix];
    return (state.research?.models || []).filter(model => (!model.model_family || ['current','reference_position_v1'].includes(model.model_family)) && model.capabilities?.[capability] !== false && (model.archived !== true && model.status === 'completed' && (model.run_id === state.research?.active_model?.run_id || model.automatic_selection_eligible !== false)));
  }
  function populateResearchModels(prefix) {
    const select = $(`${prefix}-model`), previous = select.value, models = researchModels(prefix), context = state.pendingResearchContexts[prefix];
    const configured = state.research?.active_model?.run_id || state.research?.current_model_run_id;
    const preferred = previous && previous !== select.dataset?.configuredDefaultModel ? previous : context?.model_run_id;
    const explicit = Boolean(preferred), absent = explicit && !models.some(model => model.run_id === preferred) ? [preferred] : [];
    select.innerHTML = '<option value="">自动使用当前模型</option>' + models.map(model => `<option value="${esc(model.run_id)}">${esc(model.display_name || modelName())} · ${esc(model.run_id)} · ${esc(model.predictive_gate?.passed === true ? '政策段预测基线通过' : model.run_id === state.research?.active_model?.run_id ? '当前参考版本 · 仓位资格逐折核验' : '技术完成 · 预测资格逐折核验')}</option>`).join('') + absent.map(id => `<option value="${esc(id)}">已绑定当前运行（加载时核验） · ${esc(id)}</option>`).join('');
    select.value = explicit ? preferred : '';
    if (select.dataset) select.dataset.configuredDefaultModel = explicit ? '' : configured || '';
  }
  function updateResearchControls(prefix) {
    const enabled = $(`${prefix}-mode`).value === 'research';
    $(`${prefix}-model-family`).value = 'current'; $(`${prefix}-model-family`).disabled = true;
    $(`${prefix}-model-policy`).disabled = !enabled;
    $(`${prefix}-usage-mode`).disabled = !enabled;
    const pinned = $(`${prefix}-model-policy`).value === 'pinned', model = researchModels(prefix).find(item => item.run_id === $(`${prefix}-model`).value);
    const checkpoints = researchCheckpoints(prefix,model), select = $(`${prefix}-checkpoint`), previous = select.value;
    select.innerHTML = '<option value="">自动按日期匹配</option>' + checkpoints.map(item => `<option value="${esc(item.name)}">${esc(item.name)} · 最早可用 ${esc(item.available_at || '加载时核验')}</option>`).join('');
    if (checkpoints.some(item => item.name === previous)) select.value = previous;
    select.disabled = !enabled || !pinned || !checkpoints.length;
    $(`${prefix}-model`).disabled = !enabled;
    const help = $(`${prefix}-reference-help`); help.hidden = !enabled;
    help.textContent = '联合方向模型使用趋势、结构与量价特征；来源规则另行展示。概率可用于诊断，预测门槛未通过时不增加参考仓位。';
    renderCheckpointHelp(prefix);
  }
  function researchCheckpoints(prefix, model) {
    if (model) {
      const forward = $(`${prefix}-usage-mode`).value === 'forward';
      return (model.checkpoints || []).filter(item => !forward || Boolean(item.forward_available_at)).map(item => ({...item,available_at:forward ? item.forward_available_at : item.historical_available_at || item.available_at}));
    }
    const restored = state.pendingResearchContexts[prefix];
    return restored?.model_fold && restored.model_run_id === $(`${prefix}-model`).value
      ? [{name:restored.model_fold,available_at:null,restored:true}] : [];
  }
  function selectedCheckpoint(prefix) {
    const model = researchModels(prefix).find((item) => item.run_id === $(`${prefix}-model`).value);
    return researchCheckpoints(prefix,model).find((item) => item.name === $(`${prefix}-checkpoint`).value);
  }
  function renderCheckpointHelp(prefix = 'analysis') {
    const help = $(`${prefix}-checkpoint-help`); help.hidden = $(`${prefix}-mode`).value !== 'research';
    if (help.hidden) { help.textContent = ''; return; }
    const mode = $(`${prefix}-usage-mode`).value || 'historical', pinned = $(`${prefix}-model-policy`).value === 'pinned', checkpoint = selectedCheckpoint(prefix);
    help.textContent = `${usageNames[mode]}。${mode === 'historical' ? '历史滚动重建，按各窗口拟合边界匹配；不代表当时已发布模型。' : '按实际冻结完成后的核验交易日匹配可用模型。'}${pinned ? checkpoint ? `固定检查点 ${checkpoint.name}，最早可用 ${checkpoint.available_at || '加载时核验'}。` : '请选择模型与检查点。' : ''}概率可用于诊断；预测门槛未过不增加参考仓位，缺少模型时保持未评分。`;
  }
  function researchSpec(prefix) {
    if ($(`${prefix}-mode`).value !== 'research') return {research:false,analysis_mode:'reference',...(prefix === 'analysis' && $('analysis-position').value !== '' ? {current_position_fraction:Number($('analysis-position').value) / 100} : {})};
    const usage_mode = $(`${prefix}-usage-mode`).value || 'historical';
    if (!['historical','forward'].includes(usage_mode)) throw new Error('旧用途已退役，不恢复资格；请选择历史滚动研究或当前时点研究。');
    const model_policy = $(`${prefix}-model-policy`).value || 'auto', model = $(`${prefix}-model`).value;
    let checkpoint;
    if (model_policy === 'pinned') {
      if (!model) throw new Error('请选择当前模型包与检查点。');
      checkpoint = selectedCheckpoint(prefix); if (!checkpoint) throw new Error('所选模型没有合格检查点，请刷新状态。');
      const boundary = $(`${prefix}-end`).value;
      if (boundary && checkpoint.available_at > boundary) throw new Error(`检查点 ${checkpoint.name} 在 ${checkpoint.available_at} 才可用。`);
    }
    if (prefix === 'analysis' && $('analysis-position').value !== '' && (!Number.isFinite(Number($('analysis-position').value)) || Number($('analysis-position').value) < 0 || Number($('analysis-position').value) > 100)) throw new Error('当前参考仓位须为0至100%。');
    const restored = state.pendingResearchContexts[prefix];
    return {research:true,analysis_mode:'reference',model_family:'current',consensus_version:'reference_position_v1',usage_mode,model_policy,...(prefix === 'analysis' && $('analysis-position').value !== '' ? {current_position_fraction:Number($('analysis-position').value) / 100} : {}),
      ...(model ? {model_run_id:model} : {}),...(checkpoint ? {model_fold:checkpoint.name} : {}),
      ...(restored?.model_run_id === model ? Object.fromEntries(['consensus_config'].filter(key => restored[key] != null).map(key => [key,restored[key]])) : {}),
      ...(state.research?.calendar_run_id ? {calendar_run_id:state.research.calendar_run_id} : {}),...(prefix === 'analysis' ? {device:$('analysis-device').value || 'auto'} : {})};
  }
  async function refreshResearch() {
    const data = await request('/api/training/status'); state.research = data;
    ['analysis','scan'].forEach(prefix => { populateResearchModels(prefix); updateResearchControls(prefix); if (state.pendingResearchContexts[prefix]) { const context = state.pendingResearchContexts[prefix]; delete state.pendingResearchContexts[prefix]; applyResearchContext(prefix,context); } });
    renderScanComputePlan();
    $('research-status').textContent = `参考买卖点与每股仓位 · ${(data.models || []).filter(item => item.archived !== true).length} 个参考运行 · ${data.active_model?.run_id ? `当前选择 ${data.active_model.run_id}` : '尚无合格当前选择'}。${data.calendar_run_id ? '核验交易日历可用。' : '核验日历状态以训练预检为准。'}概率评分与参考仓位资格分别核验；独立前瞻资格 ${data.certification?.status === 'qualified' ? '已认证，仍按信号时点核验' : '尚未取得'}。`;
    renderTasks(); return data;
  }


  function researchEvidence(row) {
    return `<details class="research-evidence"><summary>趋势、结构与量价参考依据</summary>${evidenceUI()?.agentOverviewHtml(row,row.consensus || row,row.consensus_config || state.scan?.consensus_config) || ''}<p class="subtle">规则支持分为0–100分；趋势概率来自当时可见的合格模型，均不代表盈利概率。</p></details>`;
  }
  function scanActionCell(row) {
    const action = row.reference_action || row.reference_recommendation?.action || row.recommendation?.action || row.action || 'WAIT';
    const isReference = row.position_reference?.schema === 'reference_position_v1';
    const label = row.category === 'excluded' ? '数据不足' : isReference ? {BUY:'B · 参考增仓',SELL:'S · 参考减仓',WAIT:'保持参考目标',HOLD:'保持参考目标'}[action] || '参考未评分' : {BUY:'共B · 最终买方参考',SELL:'共S · 最终卖方参考',WAIT:'继续观察',HOLD:'继续观察'}[action] || '继续观察';
    const trend = evidenceUI()?.trendProbabilities(row.agents?.ma);
    if (isReference) return `<span class="badge ${row.category === 'excluded' ? '' : actionClass(action)}">${esc(label)}</span><span class="stock-name">${esc(trend?.summary || '方向概率未评分')}</span>`;
    return `<span class="badge ${row.category === 'excluded' ? '' : actionClass(action)}">${esc(label)}</span><span class="stock-name">${esc(trend?.summary || '趋势概率未评分')}</span><span class="stock-name">买方支持 ${number(row.buy_base_score == null ? null : row.buy_base_score * 100,2)} 分 · 卖方支持 ${number(row.sell_base_score == null ? null : row.sell_base_score * 100,2)} 分</span>`;
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
    const reference = referenceMode(state.scan);
    const all = (state.scan?.rows || []).map((row) => reference ? {...row,analysis_mode:'reference'} : row);
    renderComputeResult('scan',state.scan?.compute_info);
    const filtered = all.filter((row) => (!query || `${row.symbol} ${row.name || state.symbolCatalog.get(row.symbol)?.name || ''} ${row.signal || ''} ${display(row.reason)} ${readableEvent(row)} ${readableReason(row)}`.toLowerCase().includes(query)) && (filter === 'all' || filter === 'favorites' && state.favorites.has(row.symbol) || filter === (row.category || actionClass(row.action)) || filter === 'sell' && (row.category === 'exit' || row.reference_action === 'SELL')));
    state.scanPage = Math.max(0,Math.min(state.scanPage,Math.ceil(filtered.length / PAGE_SIZE) - 1));
    const page = filtered.slice(state.scanPage * PAGE_SIZE,(state.scanPage + 1) * PAGE_SIZE);
    $('scan-count').textContent = number(all.length);
    const categories = {buy:'买入候选',watch:'继续观察',exit:'持仓退出研究',excluded:'排除 / 不足'};
    $('scan-rows').innerHTML = page.length ? page.map((row) => `<tr><td><button type="button" class="favorite" data-favorite="${esc(row.symbol)}" aria-label="关注 ${esc(row.symbol)}" aria-pressed="${state.favorites.has(row.symbol)}" ${!state.personalLoaded || state.watchlistBusy ? 'disabled' : ''}>${state.favorites.has(row.symbol) ? '★' : '☆'}</button></td><td><strong>${esc(row.symbol)}</strong><span class="stock-name">${esc(row.name || state.symbolCatalog.get(row.symbol)?.name || '')}</span></td><td>${esc(readableEvent(row))}</td><td>${scanActionCell(row)}</td><td>${row.category ? `行情 ${esc(row.data_end || '缺失')}<span class="stock-name">结构确认 ${esc(row.point_confirmed_at || '未确认')}</span>` : ''}${esc(display(row.available_at))}</td><td>${evidenceUI()?.positionSummaryHtml(row) || '不可用'}</td><td class="reason">${esc(readableReason(row))}${ruleOriginal(row)}${researchEvidence(row)}</td><td><button type="button" class="link-button" data-analyze-symbol="${esc(row.symbol)}" data-analysis-source="scan" ${row.data_end ? `data-analyze-end="${esc(row.data_end)}"` : ''}>分析</button></td></tr>`).join('') : `<tr><td colspan="8" class="empty-cell">${state.scan ? '没有符合筛选条件的股票。' : '开始扫描后，结果显示在这里。'}</td></tr>`;
    pagination('scan',filtered.length,state.scanPage);
    $('scan-research-result').hidden = !reference && !state.scan?.model_gate;
    if (reference) $('scan-research-result').textContent = 'B/S 为参考方向；买方与卖方支持分分别展示。缺少合格模型时趋势概率不可用。';
    if (state.scan) { $('scan-coverage').textContent = `覆盖 ${number(state.scan.coverage?.success)} / ${number(state.scan.coverage?.requested)} 只成功 · 失败 ${number((state.scan.failures || []).length)}${state.scan.data_range?.requested_end ? ` · 请求截止 ${state.scan.data_range.requested_end} · 实际行情 ${state.scan.data_end || state.scan.data_range.end || '缺失'}` : ''}`; renderDiagnostics('scan-failures',state.scan.failures);
      if (!reference && state.scan.model_gate) $('scan-research-result').textContent = `${gateCaption(state.scan.model_gate)} · ${Object.entries(categories).map(([key,label]) => `${label} ${number(state.scan.category_counts?.[key] || 0)}`).join(' · ')}。${state.scan.model_gate.reason || ''}`;
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
  function quoteNotes(items) {
    return items.flatMap((item) => {
      const warnings = Array.isArray(item.warnings) ? item.warnings : item.warnings ? [item.warnings] : [];
      return [...(!item.quote_available && !warnings.length ? ['没有可用本地收盘价，估值显示为缺失。'] : []),...warnings].map((warning) => `${item.symbol} · ${display(warning)}`);
    });
  }
  function applyWatchlist(data) { state.watchlist = data; state.favorites = new Set((data.items || []).map((item) => item.symbol)); renderWatchlist(); renderScan(); }

  function personalBusy() {
    ['add-watchlist','refresh-watchlist','scan-watchlist'].forEach((id) => { $(id).disabled = state.watchlistBusy; });
    document.querySelectorAll('[data-favorite],[data-remove-watchlist]').forEach((button) => { button.disabled = !state.personalLoaded || state.watchlistBusy; });
    $('watchlist-symbol').disabled = state.watchlistBusy;
    $('watchlist-form').setAttribute('aria-busy',String(state.watchlistBusy));
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

  async function loadPersonal() {
    const watchlist = await request('/api/watchlist'); state.personalLoaded = true; applyWatchlist(watchlist);
    $('watchlist-state').textContent = '已从本机服务读取';
  }
  function ensurePersonal() {
    if (!personalReady) personalReady = loadPersonal().catch((error) => { personalReady = null; $('watchlist-state').textContent = '读取失败，请刷新重试'; throw error; });
    return personalReady;
  }
  async function refreshWatchlist() { await ensurePersonal(); if (state.watchlistBusy) return; const sequence = ++state.watchlistSequence; const data = await request('/api/watchlist'); if (sequence === state.watchlistSequence) { applyWatchlist(data); $('watchlist-state').textContent = '关注与行情已刷新'; } }

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



  async function watchlistTask() {
    await refreshWatchlist(); const list = (state.watchlist?.items || []).map((item) => item.symbol);
    if (!list.length) { notice('请先添加关注股票'); return; }
    $('scan-symbols').value = list.join(', '); location.hash = 'scan'; route();
    await submitTask('scan',{symbols:list,...dateSpec('scan'),...researchSpec('scan')});
  }







  function taskDetail(task) {
    const d = task.progress_detail;
    if (!d || typeof d !== 'object') return display(d || '');
    const info = task.compute_info, research = Boolean(task.spec?.plan_id) || task.spec?.research;
    const compute = info ? research ? ` · ${researchComputeCaption(info)}` : ` · ${info.cuda_work ? '已执行 CUDA 信号运算' : 'CPU 信号运算'}` : task.spec?.device ? ` · 所选设备 ${task.spec.device}（尚非 GPU 执行证明）` : '';
    return `${d.stage || ''}${d.current != null ? ` · ${number(d.current)} / ${number(d.total)}` : ''}${d.device && activeStatuses.has(task.status) ? ` · 当前阶段 ${d.device === 'cpu' ? 'CPU' : d.device}` : ''}${d.failed ? ` · 失败 ${number(d.failed)}` : ''}${Number.isFinite(d.not_updated) ? ` · 未更新 ${number(d.not_updated)}` : ''}${Number.isFinite(d.excluded) ? ` · 北交排除 ${number(d.excluded)}` : ''}${Number.isFinite(d.target_date_covered) ? ` · 目标日覆盖 ${number(d.target_date_covered)}` : ''}${Number.isFinite(d.in_flight) ? ` · 计算中 ${number(d.in_flight)}` : ''}${Number.isFinite(d.cache_hits) ? ` · 数据库复用 ${number(d.cache_hits)}` : ''}${Number.isFinite(d.stocks_per_minute) && d.stocks_per_minute > 0 ? ` · ${number(d.stocks_per_minute)} 股/分钟` : ''}${compute}`;
  }
  function taskActions(task) {
    const cancel = activeStatuses.has(task.status) ? `<button type="button" data-cancel="${esc(task.job_id)}" ${task.status === 'cancelling' ? 'disabled' : ''}>取消</button>` : '';
    const result = !task.result_saved && !['succeeded','partial'].includes(task.status) ? '' : task.run_id ? `<button type="button" data-run-id="${esc(task.run_id)}" data-run-kind="${esc(task.kind || '')}">查看结果</button>` : task.result ? `<button type="button" data-task-result="${esc(task.job_id)}">查看结果</button>` : '';
    const retry = task.kind?.startsWith('board_') && ['partial','failed','cancelled'].includes(task.status) ? `<button type="button" data-board-retry="${esc(task.job_id)}">重试</button>` : '';
    return cancel + result + retry;
  }
  function renderTasks() {
    $('task-list').innerHTML = state.tasks.length ? state.tasks.map(task => `<article class="task-row"><div class="task-heading"><div><strong>${esc(kindNames[task.kind] || '当前研究任务')}</strong><small>${esc(task.created_at || '')} · ${esc(task.run_id || task.job_id)}</small></div><div class="task-actions"><span class="badge">${esc(statusNames[task.status] || task.status)}</span>${taskActions(task)}</div></div><div class="progress-track"><i style="width:${Math.max(0,Math.min(100,Number(task.progress) || 0))}%"></i></div><p class="task-detail ${task.error ? 'task-error' : ''}">${esc(task.error ? display(task.error) : taskDetail(task))}</p></article>`).join('') : '<div class="empty-copy">尚无任务，可发起训练、更新行情、扫描。</div>';
    ['scan'].forEach(kind => { const task = state.tasks.find(item => item.kind === kind && item.job_id === state.tracked[kind]), host = $(kind + '-progress'); host.hidden = !task; if (task) host.innerHTML = `<div class="task-actions">${taskActions(task)}</div>${esc(statusNames[task.status] || task.status)} · ${number(task.progress)}%<br>${esc(task.error ? display(task.error) : taskDetail(task))}`; });
    ['scan','update'].forEach(kind => { const button = $(kind === 'update' ? 'update-data' : `start-${kind}`); button.disabled = state.tasks.some(task => task.kind === kind && activeStatuses.has(task.status)); });
  }
  function showResult(kind,result,navigate = false) {
    if (kind.startsWith('board_')) { globalThis.KHQuantBoardPanel?.showResult(result); if (navigate) location.hash = 'scan'; return; }
    if (kind === 'scan') { state.scan = result; state.scanPage = 0; renderScan(); }
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
    if (!['scan','update'].includes(kind)) throw new Error('该任务入口已退役，请使用模型训练页。');
    const button = $(kind === 'update' ? 'update-data' : `start-${kind}`); button.disabled = true;
    try { if (kind !== 'update' && computeLoaded) spec = {...spec,device:$(`${kind}-device`).value}; const task = await request('/api/tasks',{method:'POST',body:JSON.stringify({kind,spec})}); state.tracked[kind] = task.job_id;
      if (kind === 'scan') { state.scan = null; state.scanPage = 0; renderScan(); $('scan-coverage').textContent = '等待本次扫描结果'; $('scan-failures').hidden = true; }
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
    const source = result?.request || result || {}, context = {analysis_mode:'reference'};
    ['research','model_family','usage_mode','model_policy','model_run_id','model_fold','calendar_run_id','device','current_position_fraction'].forEach((key) => { if (source[key] != null) context[key] = source[key]; });
    context.consensus_version = 'reference_position_v1'; context.model_family = 'current';
    ['model_policy','model_fold','calendar_run_id','current_position_fraction'].forEach((key) => { if (context[key] == null && result?.[key] != null) context[key] = result[key]; });
    context.model_run_id ||= result?.research?.model_run_id || result?.model_run_id;
    context.model_policy ||= context.model_fold ? 'pinned' : 'auto';
    const config = source.consensus_config || result?.consensus_config || result?.trend_consensus?.config;
    if (config) context.consensus_config = config;
    if (!('research' in context)) context.research = Boolean(result?.research || result?.model_gate);
    if (context.research) { context.usage_mode ||= 'historical';  }
    return context;
  }
  function applyResearchContext(prefix, context) {
    if (context.research !== false && ((context.model_family && context.model_family !== 'current') || (context.consensus_version && context.consensus_version !== 'reference_position_v1'))) throw new Error('该历史模型运行已退役，只能查看归档证据，不能恢复当前操作上下文。');
    if (context.research !== false && context.usage_mode && !['historical','forward'].includes(context.usage_mode)) throw new Error('旧用途已退役，不恢复资格；请重新选择当前研究用途。');
    if (prefix === 'analysis') $('analysis-position').value = context.current_position_fraction == null ? '' : String(context.current_position_fraction * 100);
    if (context.model_run_id) state.pendingResearchContexts[prefix] = context;
    else delete state.pendingResearchContexts[prefix];
    $(`${prefix}-model`).value = '';
    $(`${prefix}-mode`).value = context.research === false ? 'structure' : 'research';
    $(`${prefix}-model-family`).value = 'current';
    populateResearchModels(prefix);
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
    const kind = data.kind || kindHint || (result.model_manifest && result.evaluation ? 'model_training' : result.rows ? 'scan' : result.data_status ? 'update' : 'analysis');
    if (kind !== 'analysis') { showResult(kind,result,true); return; }
    // A saved analysis replaces only the generation that existed when opened.
    // Other run types keep a pending analysis intact; a newer analysis wins.
    if (analysisAtStart !== state.analysisSequence) return;
    ++state.analysisSequence;
    resetChartSelection(); analysisLoading = false;
    $('analysis-form').querySelector('[type=submit]').disabled = false;
    state.analysis = result;
    storeHistoryEvidence(evidenceUI()?.mergeEvidenceRows(displayedEvidence()?.rows,displayedConsensus()?.rows) || displayedEvidence()?.rows || [],false,result.reference_context_hash); historyPage = 0;
    state.query = {...(result.request || {symbol:result.symbol,start:null,end:result.data_end,as_of:result.as_of}),...resultResearchContext(result),...consensusRequestContext(result)};
    applyResearchContext('analysis',state.query);
    $('analysis-symbol').value = state.query.symbol || result.symbol; $('analysis-start').value = state.query.start || ''; $('analysis-end').value = state.query.end || result.data_end;
    setDateMode('analysis','historical');
    state.dates = (result.frequencies?.['日线']?.bars || []).map((bar) => day(bar.time)); state.replayIndex = Math.max(0,state.dates.length - 1);
    location.hash = 'analysis'; showView('analysis'); renderAnalysis(); renderEvents(); updateReplay();
  }
  async function analyzeSymbol(symbol,end,source = null) {
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
  window.addEventListener('khquant:board-stock', (event) => { analyzeSymbol(event.detail.symbol,event.detail.price_date).catch(showError); });
  function bind() {
    ['analysis','scan'].forEach((prefix) => $(`${prefix}-end`).addEventListener('change',() => setDateMode(prefix,$(`${prefix}-end`).value ? 'historical' : 'latest')));
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
    document.querySelectorAll('[data-frequency]').forEach((button) => button.addEventListener('click',() => { resetChartSelection(); state.frequency = button.dataset.frequency; document.querySelectorAll('[data-frequency]').forEach((item) => item.setAttribute('aria-pressed',String(item === button))); renderAnalysis(); }));
    ['ma','boll','chips','fractals','pens','zones','divergences','signals','sources','wyckoff','volume'].forEach((name) => $(`layer-${name}`).addEventListener('change',() => { resetChartSelection(); renderAnalysis(false); }));
    ['start','end','mode','model-family','usage-mode','model-policy','model','checkpoint','device','position'].forEach((name) => $(`analysis-${name}`).addEventListener('change',() => updateChartSelection('clear')));
    ['chart-range-start','chart-range-end'].forEach((id) => $(id).addEventListener('change',() => updateChartSelection('clear')));
    $('chart-range-apply').addEventListener('click',() => { updateChartSelection('clear'); applyChartRange(); });
    $('chart-zoom-in').addEventListener('click',() => zoomChart(.7)); $('chart-zoom-out').addEventListener('click',() => zoomChart(1.4));
    $('chart-range-reset').addEventListener('click',() => { if (chart) chart.timeScale().fitContent(); });
    $('replay-slider').addEventListener('change',() => replay(Number($('replay-slider').value)));
    $('replay-prev').addEventListener('click',() => replay(state.replayIndex - 1)); $('replay-next').addEventListener('click',() => replay(state.replayIndex + 1));
    $('replay-date').addEventListener('change',() => { const date = $('replay-date').value; const index = state.dates.findLastIndex((value) => value <= date); if (index < 0) notice('请选择可用行情范围内的日期'); else replay(index); });
    $('replay-reset').addEventListener('click',() => { if (!state.query) return; const {as_of:_,...query} = state.query; state.replayIndex = Math.max(0,state.dates.length - 1); analyze(query,true); });
    $('scan-form').addEventListener('submit',(event) => { event.preventDefault(); try { submitTask('scan',{symbols:symbols($('scan-symbols').value),...dateSpec('scan'),...researchSpec('scan')}); } catch(error) { showError(error); } });
    ['analysis','scan'].forEach((prefix) => {
      ['mode','model-family','usage-mode','model-policy','model'].forEach((name) => $(`${prefix}-${name}`).addEventListener('change',() => {
        if (name === 'model-family' || name === 'model' && $(`${prefix}-model`).value !== state.pendingResearchContexts[prefix]?.model_run_id) delete state.pendingResearchContexts[prefix];
        if (name === 'model-family') populateResearchModels(prefix);
        updateResearchControls(prefix);
        if (prefix === 'scan') renderScanComputePlan();
      }));
      $(`${prefix}-checkpoint`).addEventListener('change',() => renderCheckpointHelp(prefix));
      $(`${prefix}-end`).addEventListener('change',() => renderCheckpointHelp(prefix));
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
    ['prev','next'].forEach((direction) => $(`scan-${direction}`).addEventListener('click',() => { state.scanPage += direction === 'prev' ? -1 : 1; renderScan(); }));
    $('refresh-data').addEventListener('click',() => refreshData().catch(showError)); $('refresh-tasks').addEventListener('click',() => refreshTasks().catch(showError)); $('refresh-runs').addEventListener('click',() => refreshRuns().catch(showError));
    ['analysis','scan','data'].forEach((kind) => $(`${kind}-device`).addEventListener('change',() => { if (kind === 'scan') renderScanComputePlan(); refreshCompute(kind,$(`${kind}-device`).value).catch(showError); }));
    $('refresh-compute').addEventListener('click',() => refreshCompute('data',$('data-device').value).catch(showError));
    $('update-form').addEventListener('submit',(event) => { event.preventDefault(); const end = $('update-end').value; submitTask('update',{symbols:symbols($('update-symbols').value),...(end ? {end} : {})}); });
    ['watchlist-symbol'].forEach((id) => $(id).addEventListener('input',() => { clearTimeout(searchTimer); searchTimer = setTimeout(() => searchSymbols($(id).value).catch(showError),250); }));
    $('watchlist-form').addEventListener('submit',(event) => { event.preventDefault(); try { changeWatchlist(personalSymbol($('watchlist-symbol').value),false).then(() => { $('watchlist-symbol').value = ''; }).catch(showError); } catch (error) { showError(error); } });
    $('refresh-watchlist').addEventListener('click',() => refreshWatchlist().catch(showError));
    $('scan-watchlist').addEventListener('click',() => watchlistTask('scan').catch(showError));
    document.addEventListener('click',(event) => {
      const button = event.target.closest('button'); if (!button) return;
      if (button.hasAttribute('data-chart-evidence-close')) updateChartSelection('clear');
      if (button.dataset.evidenceDate) { chartSelection = evidenceUI().selectionTransition(chartSelection,{type:'date',date:button.dataset.evidenceDate}); renderChartPopup(); drawChipProfile(); renderIndicatorReadout(chartSelection.pinned); syncChartRange(); $('chart-selection-status').textContent = `已固定原始日期 ${button.dataset.evidenceDate} · 筹码截至 ${selectedChipProfile()?.as_of || '—'}`; }
      if (button.dataset.evidenceLocate) { highlightEvent(button.dataset.evidenceLocate); const entry = [...document.querySelectorAll('[data-event-time]')].find((item) => item.dataset.eventTime === button.dataset.evidenceLocate || item.dataset.evidenceDate === button.dataset.evidenceLocate); entry?.scrollIntoView({behavior:'smooth',block:'nearest'}); }
      if (button.dataset.evidenceReplay || button.dataset.historyDate) { const time = button.dataset.evidenceReplay || button.dataset.historyDate; const index = state.dates.findIndex((item) => item === time); if (index >= 0 && !analysisLoading) replay(index).catch(showError); }
      if (button.dataset.historyPage != null && !analysisLoading) { historyPage = Number(button.dataset.historyPage); renderReplayEvidence(); }
      if (button.dataset.favorite) changeWatchlist(button.dataset.favorite,state.favorites.has(button.dataset.favorite)).catch(showError);
      if (button.dataset.removeWatchlist) changeWatchlist(button.dataset.removeWatchlist,true).catch(showError);
      if (button.dataset.analyzeSymbol) analyzeSymbol(button.dataset.analyzeSymbol,button.dataset.analyzeEnd,button.dataset.analysisSource === 'scan' ? state.scan : null).catch(showError);
      if (button.dataset.locateEvent) locateTime(button.dataset.locateEvent);
      if (button.dataset.boardRetry) globalThis.KHQuantBoardPanel?.retryTask(button.dataset.boardRetry).catch(showError);
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
