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
  const methodDefinitions = Object.freeze({
    structure:{label:'结构与量价'},
    structure_kronos:{label:'结构与量价 + Kronos',prediction:true},
    structure_kronos_ma_mlp:{label:'结构与量价 + Kronos + 均线模型 MLP',prediction:true,score:'mlp'},
    structure_kronos_ma_l2:{label:'结构与量价 + Kronos + 同输入 L2',prediction:true,score:'l2'},
    legacy_reference:{label:'历史参考模型（对照）',legacy:true}
  });
  const methodFields = ['analysis_method','usage_mode','score_model_id','lookback','horizon','samples','seed','budget_seconds','max_predictions','device'];
  function normalizeMethod(value) { return value === 'research' ? 'legacy_reference' : value || 'structure'; }
  function selectedMethod(prefix) { const method=normalizeMethod($(`${prefix}-mode`).value); if(!methodDefinitions[method]) throw new Error('未知分析方法，请重新选择。'); return method; }
  function resultMethod(result) { return normalizeMethod(result?.analysis_method || result?.method_context?.analysis_method || result?.request?.analysis_method || (result?.research ? 'legacy_reference' : 'structure')); }
  function methodLabel(value) { return methodDefinitions[normalizeMethod(value)]?.label || '未知方法'; }
  const methodSettings = {analysis:{},scan:{}};
  let methodStatus=null, methodStatusSequence=0;
  let analysisTransfer=null;
  function analysisFormContext() { return {symbol:getSymbol($('analysis-symbol').value),...dateSpec('analysis'),...researchSpec('analysis')}; }
  function analysisContext() {
    const context=analysisFormContext();
    if(state.analysis && !analysisLoading && analysisTransfer?.form===JSON.stringify(stableIdentity(context)) && analysisTransfer.modelId)context.score_model_id=analysisTransfer.modelId;
    return context;
  }
  function methodReason(value) { return ({batch_budget_deferred:'本批预算已用尽，等待续批',kronos_assets_missing:'本地 Kronos 资源未就绪',kronos_prediction_unusable:'Kronos 预测未通过可用性检查',ma_score_model_missing:'缺少兼容的均线评分候选',ma_score_unavailable:'均线评分不可用',same_input_l2_missing:'该候选缺少同输入 L2 对照',scored:'已评分'})[value]||display(value); }
  function methodEvaluationHtml(value, compact=false) {
    if(!value) return '';
    const score=value.score || {}, kronos=value.kronos || {};
    const probabilities=Array.isArray(score.probabilities)?score.probabilities:score.probabilities?['down','flat','up'].map(key=>score.probabilities[key]):null;
    const reasons=[...(value.missing_reasons || []), ...(score.reason ? [score.reason] : []), ...(kronos.reason ? [kronos.reason] : [])]
      .filter(item=>item!=='scored').map(methodReason);
    const direction={BUY:'买方参考',SELL:'卖方参考',WAIT:'观察'}[value.direction] || '方向待核验';
    const scoreValue=score.score ?? score.scores?.[0];
    const scoreName=({torch_mlp:'MLP',mlp:'MLP',l2:'L2'})[score.model_type] || (value.method?.endsWith('l2')?'L2':'MLP');
    const probabilityText=Array.isArray(probabilities) && probabilities.length===3 ? `下跌 ${percent(probabilities[0])} · 平稳 ${percent(probabilities[1])} · 上涨 ${percent(probabilities[2])}` : '';
    const summary=kronos.summary || {}, quality=kronos.quality || {};
    const forecast=[['terminal_return_median','中位涨跌'],['terminal_return_p10','P10'],['terminal_return_p90','P90'],['up_path_fraction','上涨采样比例']]
      .filter(([key])=>typeof summary[key]==='number'&&Number.isFinite(summary[key]))
      .map(([key,label])=>`${label} ${percent(summary[key])}`);
    if(typeof quality.valid_samples==='number'&&Number.isFinite(quality.valid_samples))forecast.push(`有效路径 ${number(quality.valid_samples)}${typeof quality.requested_samples==='number'&&Number.isFinite(quality.requested_samples)?` / ${number(quality.requested_samples)}`:''}`);
    const qualification=score.qualification || {};
    const purpose=qualification.mode==='historical_candidate'?'历史研究候选':qualification.mode==='paper'?'自动模拟用途':qualification.research_only===true?'仅研究用途':'用途未记录';
    const forward=({pending:'独立前瞻资格待验证',qualified:'独立前瞻资格已通过',failed:'独立前瞻资格未通过'})[qualification.independent_forward_status] || '独立前瞻资格未记录';
    const qualificationText=score.qualification ? `${purpose}；${forward}` : '';
    return `<div class="method-summary"><strong>${esc(value.label || methodLabel(value.method))}</strong><span class="badge">${({complete:'已完成',partial:'部分可用',deferred:'预算延后',failed:'评估失败'})[value.status]||'待核验'}</span><span>${esc(direction)}</span></div>${compact?'':`<p class="subtle">信息截止 ${esc(value.as_of || '—')} · ${value.model_id ? `模型 ${esc(String(value.model_id).slice(0,12))}${String(value.model_id).length>12?'…':''}` : '未使用评分候选'}${value.feature_profile ? ` · ${esc(value.feature_profile==='ma_trend_v1'?'均线趋势输入':value.feature_profile)}` : ''}</p>`}<p>${scoreValue!=null ? `${esc(scoreName)} ${score.status==='scored'?'已评分':'评分'} ${number(scoreValue,2)} / 100` : methodDefinitions[value.method]?.score ? '评分不可用' : '结构与量价来源独立保留'}${probabilityText?` · ${esc(probabilityText)}`:''}</p>${qualificationText?`<p class="subtle">${esc(qualificationText)}</p>`:''}${forecast.length?`<p>Kronos：${esc(forecast.join(' · '))}</p><p class="subtle">采样分位数与上涨比例未经校准；预测涨跌不等于实际收益。</p>`:''}${reasons.length?`<p class="method-missing">${[...new Set(reasons.map(display))].map(esc).join('；')}</p>`:''}${compact?'':`<details><summary>查看预测、评分与资格依据</summary><pre>${esc(JSON.stringify({model_id:value.model_id,feature_profile:value.feature_profile,kronos,score,missing_reasons:value.missing_reasons},null,2))}</pre></details><p class="subtle">预测与评分为模型参考；不代表模拟成交或目标仓位。</p>`}`;
  }
  function renderMethodResult(prefix,result) {
    const host=$(`${prefix}-method-result`); if(!host)return;
    const evaluation=result?.method_evaluation;
    host.hidden=!evaluation && !(prefix==='scan' && methodDefinitions[resultMethod(result)]?.prediction);
    host.innerHTML=evaluation ? methodEvaluationHtml(evaluation) : result ? `<strong>${esc(methodLabel(resultMethod(result)))}</strong><p>预测与均线评分按扫描截止日分别核验；展开每行依据查看缺失与延后原因。</p>` : '';
    if(prefix==='scan') { const resume=$('scan-resume'); if(resume) resume.hidden=!(result?.run_id && (result.status==='partial' || Number(result.coverage?.deferred)>0)); }
  }
  function applyMethodSettings(prefix,context={}) {
    methodSettings[prefix]=Object.fromEntries(methodFields.filter(key=>context[key]!=null).map(key=>[key,context[key]]));
    const fields={lookback:'lookback',horizon:'horizon',samples:'samples',seed:'seed',budget_seconds:'budget',max_predictions:'max-predictions'};
    Object.entries(fields).forEach(([key,suffix])=>{if(context[key]!=null&&$(`${prefix}-${suffix}`))$(`${prefix}-${suffix}`).value=String(context[key]);});
    const select=$(`${prefix}-score-model`); if(select && context.score_model_id) { if(![...(select.options||[])].some(option=>option.value===context.score_model_id)) select.innerHTML+=`<option value="${esc(context.score_model_id)}">已绑定均线候选 · ${esc(context.score_model_id)}</option>`; select.value=context.score_model_id; }
  }
  async function refreshMethodModels() {
    const sequence=++methodStatusSequence, data=await request('/api/light/status'); if(sequence!==methodStatusSequence)return;
    methodStatus=data;
    const models=(data.score_candidates||[]).filter(model=>(model.feature_profile || model.artifact?.feature_profile || model.identity?.feature_profile)==='ma_trend_v1');
    ['analysis','scan'].forEach(prefix=>{
      const select=$(`${prefix}-score-model`); if(!select)return;
      const previous=select.value || methodSettings[prefix].score_model_id || '';
      select.innerHTML='<option value="">自动选择兼容均线候选</option>'+models.map(model=>{const id=model.model_id||model.model_ref?.cache_key||model.cache_key;return `<option value="${esc(id)}">均线 ${(model.model_type||'MLP').toUpperCase()} · ${esc(model.end||model.training_end||model.identity?.end||'')} · ${esc(String(id).slice(0,12))}</option>`;}).join('');
      if(previous && !models.some(model=>(model.model_id||model.model_ref?.cache_key||model.cache_key)===previous))select.innerHTML+=`<option value="${esc(previous)}">已绑定候选（执行时核验） · ${esc(previous)}</option>`;
      select.value=previous;
      const text=$(`${prefix}-method-state`); if(text)text.textContent=`Kronos ${data.assets?.ready?'就绪':data.assets?.status||'待核验'} · ${models.length} 个均线候选。${models.length?'按实际冻结时间与输入核验资格。':'尚无均线模型；可到预测与回测页训练候选。'}缺失项保留明确状态。`;
    });
  }
  globalThis.KHQuantMethods={definitions:methodDefinitions,normalize:normalizeMethod,label:methodLabel,evaluationHtml:methodEvaluationHtml,analysisContext};
  const PAGE_SIZE = 100;
  const state = {analysis:null, query:null, frequency:'日线', dates:[], replayIndex:0, scan:null, scanPage:0, tasks:[], delivered:new Set(), tracked:{}, favorites:new Set(), symbolCatalog:new Map(), analysisSequence:0, taskSequence:0, pollTimer:null, watchlist:null, personalLoaded:false, watchlistBusy:false, watchlistSequence:0,research:null,pendingResearchContexts:{},archive:false,resultCreatedAt:null};
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
  let chartMarkers = [], sellFocus = null;
  let chartRangeDraft = null, chartRangeObserved = null;
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
  function stableIdentity(value) {
    if (Array.isArray(value)) return value.map(stableIdentity);
    if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key,stableIdentity(value[key])]));
    return value ?? null;
  }
  function analysisHistoryIdentity(result, query = {}) {
    const request = {...query,...result?.request}, research = result?.research || {};
    const identities = [...new Map((result?.trend_consensus?.rows || result?.chart_evidence?.rows || []).map(row => {
      const identity = row.agents?.ma?.model_identity || row.model_identity || {};
      const {score_date:_,...frozen} = identity;
      const assets = row.agents?.ma?.model_asset_identity || identity.model_asset_identity || row.model_resolution?.model_asset_identity || {};
      frozen.model_asset_identity = assets;
      return [JSON.stringify(stableIdentity(frozen)),frozen];
    })).values()];
    return JSON.stringify(stableIdentity({symbol:result?.symbol,start:request.start,
      schema:result?.strategy_version || result?.trend_consensus?.version,
      model_mode:request.research,method_context:Object.fromEntries(methodFields.map(key=>[key,request[key]??result?.method_context?.[key]])),usage_mode:request.usage_mode || research.usage_mode,
      model_policy:request.model_policy,model_run_id:request.model_run_id || result?.model_run_id || research.model_run_id,
      model_fold:request.model_fold,identities,model_asset_identities:result?.model_asset_identities || research.model_asset_identities || [],
      calendar_run_id:request.calendar_run_id || result?.calendar_run_id,
      source_context:research.source_context_hash || result?.source_context_hash,
      runtime:result?.runtime_code_hash || research.runtime_code_hash || result?.runtime_identity || research.runtime_identity,
      policy:request.consensus_config || result?.trend_consensus?.config,current_position_fraction:request.current_position_fraction,
      series_identity:result?.reference_context_hash || research.reference_context_hash || research.series_context_hash}));
  }
  function shanghaiClock(value) {
    const text = String(value || '');
    if (/^\d{4}-\d{2}-\d{2}$/.test(text)) return Date.parse(`${text}T15:00:00+08:00`);
    // Legacy offset-free ISO clocks are interpreted in the application's Shanghai zone.
    return Date.parse(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?$/.test(text) ? `${text}+08:00` : text);
  }
  function rowObservable(row, cutoff) {
    if (!row?.date || cutoff && day(row.date) > day(cutoff)) return false;
    if (!cutoff) return true;
    const boundary = shanghaiClock(cutoff);
    if (!Number.isFinite(boundary)) return false;
    const observableKeys = new Set(['available_at','first_available_at','first_confirmed_at','confirmed_at',
      'observed_at','origin_observed_at','first_observed_at','score_available_at','score_first_available_at','evidence_available_at']);
    const check = value => {
      if (!value || typeof value !== 'object') return true;
      return Object.entries(value).every(([key,item]) => {
        // Actual model-training/publication clocks belong to reconstruction identity,
        // rather than the first-available clock of a displayed observation.
        if (['model_identity','model_asset_identity','model_asset_identities'].includes(key)) return true;
        if (observableKeys.has(key) && item != null && item !== '') {
          const clock = shanghaiClock(item);
          if (!Number.isFinite(clock) || clock > boundary) return false;
        }
        return item && typeof item === 'object' ? check(item) : true;
      });
    };
    return check(row);
  }
  function storeHistoryEvidence(incoming, preserveDates, seriesContext, cutoff = state.query?.as_of || state.analysis?.as_of || state.analysis?.data_end) {
    const preserve = preserveDates && Boolean(seriesContext) && seriesContext === historySeriesContext;
    const observable = (incoming || []).filter(row => rowObservable(row,cutoff));
    // A shorter replay never retains future rows, even when its frozen series identity matches.
    historyRows = (preserve ? [...new Map([...historyRows,...observable].map(row => [row.date,row])).values()] : observable)
      .filter(row => rowObservable(row,cutoff));
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
    const selected = ['analysis', 'scan', 'watchlist', 'training', 'parameters', 'data', 'simulation'].includes(name) ? name : 'analysis';
    if (selected !== 'analysis') resetChartSelection();
    showView(selected);
    requestAnimationFrame(resizeCharts);
    if (['training','parameters'].includes(selected)) globalThis.KHQuantTraining?.refresh().catch(showError);
    if (selected === 'data') { refreshTasks().catch(showError); refreshRuns().catch(showError); }
    if (['analysis','scan'].includes(selected)) { if(selectedMethod(selected)==='legacy_reference') refreshResearch().catch(showError); else if(methodDefinitions[selectedMethod(selected)].prediction) refreshMethodModels().catch(showError); }
    if (selected === 'scan') globalThis.KHQuantBoardPanel?.onRoute();
    if (selected === 'simulation') globalThis.KHQuantLightResearch?.refresh();
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
  function emptyChart(message, detail) { resetChartSelection(); clearIndicators(); $('chip-profile').hidden = true; $('chip-profile-caption').hidden = true; $('chip-profile-overlay').replaceChildren(); syncChartRange(); $('divergence-details').hidden = true; $('divergence-list').replaceChildren(); $('analysis-empty').hidden = false; $('analysis-empty').querySelector('strong').textContent = message; $('analysis-empty').querySelector('span').textContent = detail; $('marker-summary').textContent = '尚无可展示的买卖标记'; markerMappings = []; markerGroups = []; renderMarkerScope(); renderResultIdentity(); $('marker-details').textContent = state.analysis?.structure_only ? '点击来源 B/S 或对应 K 线查看结构与量价依据；悬浮切换指标与筹码。' : KHQuantMarkers.describe(markerMappings,null,state.frequency); $('zone-overlay').replaceChildren(); }
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
    if (fit) chartRangeDraft = null;
    if (!state.analysis) { clearIndicators(); $('chip-profile').hidden = true; $('chip-profile-caption').hidden = true; $('chip-profile-overlay').replaceChildren(); syncChartRange(); return; }
    renderMethodResult('analysis',state.analysis);
    const sourceOnly = state.analysis.structure_only === true;
    $('position-reference-card').hidden = sourceOnly;
    $('analysis-compute-status').hidden = sourceOnly;
    if($('layer-signals').closest?.('label')) $('layer-signals').closest('label').hidden = sourceOnly;
    $('chart-locate-target-s').hidden = sourceOnly;
    $('event-target-option').hidden = sourceOnly;
    if (sourceOnly && $('event-layer-filter').value === 'target') $('event-layer-filter').value = 'source';
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
    $('marker-details').textContent = state.analysis?.structure_only ? '点击来源 B/S 或对应 K 线查看结构与量价依据；悬浮切换指标与筹码。' : KHQuantMarkers.describe(markerMappings,null,state.frequency);
    const sourceEvents = referenceMarkersBuilt.mappings.filter((event) => event.reference_layer === 'source' && event.displayed), finalEvents = referenceMarkersBuilt.mappings.filter((event) => event.reference_layer !== 'source' && event.displayed);
    const count = (items,side) => items.filter((item) => item.action === side).length;
    const finalSignalName = state.analysis.strategy_version === 'reference_position_v1' ? '目标变化' : '最终共识';
    $('marker-summary').textContent = `已启用图层 · 来源 B ${count(sourceEvents,'BUY')} / S ${count(sourceEvents,'SELL')}${sourceOnly ? '' : ` · ${finalSignalName} B ${count(finalEvents,'BUY')} / S ${count(finalEvents,'SELL')}`} · 周期标记 ${referenceMarkersBuilt.intentMarkerCount} 个${referenceMarkersBuilt.formingProjectedCount ? ` · ${referenceMarkersBuilt.formingProjectedCount} 个日线参考点投影于形成中周期` : ''} · 点击查看来源及当日依据`;
    chartMarkers = markers.sort((a,b) => a.time.localeCompare(b.time)); applySellFocus();
    $('consensus-semantics').textContent = state.analysis.structure_only ? '结B/S、量B/S为已确认结构与量价来源，参考价取信号日收盘。实际买卖、持仓和退出在模拟账户中核验。' : state.analysis.strategy_version === 'reference_position_v1' ? '结B/S、量B/S为已确认来源参考；参B/S为每股参考目标的增加或减少。来源点与目标变化独立显示，参考价取信号日收盘。' : '结B/S、量B/S、均B/S为已确认来源参考；共B/S为达原门槛的最终共识。参考价取信号日收盘，支持分不代表盈利概率。';
    if (fit) chart.timeScale().fitContent();
    renderMarkerScope(); renderResultIdentity();
    $('chart-symbol').textContent = `${state.analysis.symbol || symbol || ''} ${state.analysis.name || state.symbolCatalog.get(state.analysis.symbol)?.name || ''}`.trim();
    const formingCount = bars.filter((bar) => !bar.is_closed).length;
    $('chart-information').textContent = `${state.frequency} · ${bars.length} 根${formingCount ? `（${formingCount} 根淡色形成中，未用于规则）` : ''} · 可知截至 ${state.query?.as_of || state.analysis.as_of || state.analysis.data_end || last} · ${basisLabel(state.analysis.price_basis)}`;
    $('analysis-state').textContent = state.analysis.warmup?.eligible === false ? reference ? '历史不足，参考判断不可用' : '预热不足，不能产生有效交易' : state.query?.as_of ? `回放至 ${state.query.as_of}` : `结构截至 ${state.analysis.as_of || state.analysis.data_end || state.query?.end || last}`;
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
    if (type === 'clear' || type === 'pin') clearSellFocus();
    if (!ui) { $('marker-details').textContent = KHQuantMarkers.describe(markerMappings,time,state.frequency); renderIndicatorReadout(time); return; }
    const previous = chartSelection;
    const canOpen = type === 'pin' && popupGroupsAt(time).length > 0;
    chartSelection = ui.selectionTransition(chartSelection,{type:type === 'locate' ? 'pin' : type,time,date:selectedDate,popupOpen:canOpen});
    if (chartSelection.pinned && (type === 'hover' || type === 'leave')) return;
    if (point) { const rect = $('analysis-chart').getBoundingClientRect(); popupAnchor = {x:rect.left + point.x,y:rect.top + point.y}; }
    const active = chartSelection.pinned || chartSelection.hover;
    $('marker-details').textContent = state.analysis?.structure_only && !active ? '点击来源 B/S 或对应 K 线查看结构与量价依据；悬浮切换指标与筹码。' : KHQuantMarkers.describe(markerMappings,active,state.frequency);
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
    clearSellFocus();
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
  function referenceEventCounts(items) {
    const count = (layer,side) => items.filter(item => (item.reference_layer === 'source' ? 'source' : 'target') === layer && (item.action || item.side) === side).length;
    return {sourceB:count('source','BUY'),sourceS:count('source','SELL'),targetB:count('target','BUY'),targetS:count('target','SELL')};
  }
  function countText(items) {
    const c = referenceEventCounts(items), target = state.analysis?.strategy_version === 'reference_position_v1' ? '目标变化' : '最终共识';
    if(state.analysis?.structure_only) return `来源 B ${c.sourceB} / S ${c.sourceS}`;
    return `来源 B ${c.sourceB} / S ${c.sourceS} · ${target} B ${c.targetB} / S ${c.targetS}`;
  }
  function visibleMarkerMappings() {
    const bars = state.analysis?.frequencies?.[state.frequency]?.bars || [], range = chart?.timeScale().getVisibleLogicalRange();
    if (!range || !bars.length) return [];
    const dates = new Set(bars.filter((_,index) => index >= Math.ceil(range.from) && index <= Math.floor(range.to)).map(bar => day(bar.time)));
    return markerMappings.filter(item => item.displayed !== false && dates.has(item.bar_date));
  }
  function renderMarkerScope() {
    const total = $('marker-total-summary'), view = $('marker-view-summary'), note = $('marker-empty-explanation');
    if (!total || !view || !note) return;
    if (!state.analysis || analysisLoading) { total.textContent = ''; view.textContent = ''; note.textContent = ''; ['event-locate-s','chart-locate-source-s','chart-locate-target-s'].forEach(id => { if ($(id)) $(id).disabled = true; }); return; }
    const events = evidenceUI()?.referenceMarkerInputs(state.analysis).events || [], visible = visibleMarkerMappings();
    const bars = state.analysis.frequencies?.[state.frequency]?.bars || [];
    const range = chart?.timeScale().getVisibleLogicalRange();
    const windowBars = range ? bars.filter((_,index) => index >= Math.ceil(range.from) && index <= Math.floor(range.to)) : [];
    const begin = day(windowBars[0]?.time), end = day(windowBars.at(-1)?.time);
    const groups = new Set(visible.map(item => item.group_id)).size;
    total.textContent = `整份结果 · ${countText(events)} · ${events.length} 条原始日线事件`;
    view.textContent = `当前可见 ${begin || '—'} 至 ${end || '—'} · ${countText(visible)} · ${visible.length} 条原始日线事件 / ${groups} 个${state.frequency}标记`;
    const all = referenceEventCounts(events), current = referenceEventCounts(visible);
    const explanations = [];
    if (!current.sourceS && all.sourceS) explanations.push(!$('layer-sources').checked ? '来源图层未启用，来源 S 暂不绘制。' : '来源 S 在当前视窗外，可定位最近 S。');
    if (!current.targetS && all.targetS) explanations.push(!$('layer-signals').checked ? '目标图层未启用，目标 S 暂不绘制。' : '目标 S 在当前视窗外，可点击“定位目标 S”直接查看。');
    const plan = state.analysis.position_reference || {}, target = plan.target_position_fraction ?? plan.target_fraction;
    if (!state.analysis.structure_only && !current.targetS && state.analysis.strategy_version === 'reference_position_v1') {
      if (Number.isFinite(target) && target === 0) explanations.push('结果截止日参考目标为 0；来源 S 独立保留，目标未进一步下降时不重复生成参 S。');
      else if (plan.status === 'pending' || plan.trajectory_status === 'pending') explanations.push('结果截止日参考资格待通过；来源点与目标变化分别记录。');
      else if (!all.targetS) explanations.push('本份结果没有参考目标下降，未生成参 S。');
    }
    note.textContent = explanations.join(' ');
    const button = $('event-locate-s');
    if (button) button.disabled = !latestSellEvent(events,$('event-layer-filter')?.value || 'all');
    renderSellNavigation(events);
  }
  function filteredReferenceEvents(events, layer = 'all', side = 'all') {
    return events.filter(event => (layer === 'all' || (event.reference_layer === 'source' ? 'source' : 'target') === layer) && (side === 'all' || (event.action || event.side) === side));
  }
  function latestSellEvent(events, layer = 'all') {
    return filteredReferenceEvents(events,layer,'SELL').sort((a,b) => String(a.first_available_at || a.available_at || a.date).localeCompare(String(b.first_available_at || b.available_at || b.date))).at(-1);
  }
  function renderSellNavigation(events) {
    const targetLabel = state.analysis?.strategy_version === 'reference_position_v1' ? '目标' : '共识';
    for (const [layer,id,label] of [['source','chart-locate-source-s','来源'],['target','chart-locate-target-s',targetLabel]]) {
      const button = $(id), latest = latestSellEvent(events,layer);
      if (button) { button.disabled = !latest; button.textContent = `定位${label} S${latest ? ` · ${day(latest.first_available_at || latest.available_at || latest.date)}` : ' · 无记录'}`; }
    }
  }
  function applySellFocus() {
    if (!candles) return;
    referenceMutationDepth++;
    try { candles.setMarkers(chartMarkers.map(marker => sellFocus && marker.id === sellFocus.groupId ? {...marker,size:1.7,text:`${marker.text} · 定位`} : marker)); }
    finally { referenceMutationDepth--; }
  }
  function clearSellFocus() {
    const previous = sellFocus;
    sellFocus = null;
    if (previous) applySellFocus();
    if ($('marker-location-summary')) { $('marker-location-summary').textContent = ''; $('marker-location-summary').hidden = true; }
  }
  function locateLatestSell(layer = null) {
    if (analysisLoading || !state.analysis) return;
    const selected = typeof layer === 'string' ? layer : $('event-layer-filter').value || 'all';
    const events = evidenceUI()?.referenceMarkerInputs(state.analysis).events || [], event = latestSellEvent(events,selected);
    if (!event) { notice('所选图层在本份结果中没有 S。'); return; }
    resetChartSelection();
    if (selected !== 'all') $('event-layer-filter').value = selected;
    $(event.reference_layer === 'source' ? 'layer-sources' : 'layer-signals').checked = true;
    $('event-side-filter').value = 'SELL'; renderAnalysis(false); renderEvents();
    const date = day(event.first_available_at || event.available_at || event.date);
    const mapping = markerMappings.find(item => item.displayed !== false &&
      (item.reference_layer === 'source') === (event.reference_layer === 'source') &&
      (event.event_id ? item.event_id === event.event_id : item.original_date === date && item.source === event.source) && item.action === 'SELL');
    sellFocus = mapping ? {groupId:mapping.group_id,date,layer:event.reference_layer === 'source' ? 'source' : 'target'} : null;
    applySellFocus(); locateTime(date,false,true);
    const label = event.reference_layer === 'source' ? `来源 S · ${event.source_label || event.source || '来源'}` : state.analysis.strategy_version === 'reference_position_v1' ? '目标 S · 参考减仓' : '共识 S · 卖方参考';
    const summary = $('marker-location-summary');
    if (summary) { summary.hidden = false; summary.textContent = mapping ? `已定位 ${date} ${label}；图中放大的向下箭头标注“${chartMarkers.find(marker => marker.id === mapping.group_id)?.text || 'S'} · 定位”。参考价 ${number(event.reference_price,2)} 元，取信号日收盘；箭头绘于对应 K 线上方。` : `${date} ${label} 未映射到当前周期，请切换日线查看；未新增标记。`; }
  }
  function resultSchema(result) { return result?.strategy_version || result?.trend_consensus?.version || result?.consensus_decisions?.version || result?.request?.consensus_version || '未记录'; }
  function archivedResult(result) { return !(result?.analysis_method && methodDefinitions[result.analysis_method] && result.analysis_method!=='legacy_reference') && resultSchema(result) !== 'reference_position_v1'; }
  function renderResultIdentity() {
    const box = $('analysis-result-identity'); if (!box) return;
    box.hidden = !state.analysis || analysisLoading;
    $('analysis-current-recompute').disabled = analysisLoading;
    if (!state.analysis || analysisLoading) return;
    const result = state.analysis, source = result.request || {}, research = result.research || {}, identity = result.next_session_decision?.agents?.ma?.model_identity || {};
    const targetLabel = resultSchema(result) === 'reference_position_v1' ? '目标变化' : '最终共识';
    if ($('layer-signals-caption')) $('layer-signals-caption').textContent = `${targetLabel}B/S`;
    if ($('event-target-option')) $('event-target-option').textContent = `${targetLabel} B/S`;
    $('analysis-result-caption').textContent = `${state.archive ? '只读归档' : result.preview ? '限时预览 · 未创建永久运行' : '当前参考结果'} · ${resultSchema(result)} · 生成 ${state.resultCreatedAt || result.run_context?.created_at || '未记录'} · 行情 ${result.data_end || '—'} · ${usageNames[source.usage_mode || research.usage_mode] || source.usage_mode || '结构规则'} · 检查点 ${source.model_fold || identity.checkpoint || '自动按日期'}`;
    if(result.structure_only) $('analysis-result-caption').textContent = `${result.preview ? '限时预览 · ' : ''}${methodLabel(resultMethod(result))} · 行情 ${result.data_end || '—'} · 仅使用截止日可知输入`;
    if ($('replay-mode-caption')) $('replay-mode-caption').textContent = result.structure_only ? '按历史时点查看结构与量价来源' : state.archive ? '只读历史记录 · 原保存参考价与评分' : '按历史时点重新判断 · 历史参考价与 agent 评分';
    if ($('history-evidence-hint')) $('history-evidence-hint').textContent = state.archive ? '点击日期查看原保存的当日证据' : '点击日期按当时可见信息重新判断';
    $('analysis-archive-note').hidden = !state.archive;
    $('analysis-current-recompute').hidden = !state.archive;
  }
  function syncChartRange() {
    const bars = state.analysis?.frequencies?.[state.frequency]?.bars || [];
    const available = !!chart && bars.length > 0;
    ['chart-range-start','chart-range-end','chart-range-apply','chart-zoom-in','chart-zoom-out','chart-range-reset'].forEach((id) => { $(id).disabled = !available; });
    renderMarkerScope();
    if (!available) { chartRangeDraft = null; chartRangeObserved = null; $('chart-range-start').value = ''; $('chart-range-end').value = ''; $('chart-range-summary').textContent = ''; return; }
    const range = chart.timeScale().getVisibleLogicalRange();
    if (!range) return;
    const observed = [state.analysisSequence,state.frequency,range.from,range.to].join('|');
    if (chartRangeObserved != null && chartRangeObserved !== observed) chartRangeDraft = null;
    chartRangeObserved = observed;
    const from = Math.max(0,Math.min(bars.length - 1,Math.ceil(range.from))), to = Math.max(from,Math.min(bars.length - 1,Math.floor(range.to)));
    ['chart-range-start','chart-range-end'].forEach((id) => { $(id).min = day(bars[0].time); $(id).max = day(bars.at(-1).time); });
    if (!chartRangeDraft) { $('chart-range-start').value = day(bars[from].time); $('chart-range-end').value = day(bars[to].time); }
    $('chart-range-summary').textContent = `${state.frequency} · 显示 ${to - from + 1} 根 · 判断截至 ${state.analysis.data_end || state.query?.as_of || '—'} · 筹码截至 ${selectedChipProfile()?.as_of || '—'} · 滚轮缩放 / 拖动平移`;
  }
  function captureChartRangeDraft() {
    chartRangeDraft = {start:$('chart-range-start').value,end:$('chart-range-end').value};
    return {...chartRangeDraft};
  }
  function applyChartRange(requested = captureChartRangeDraft()) {
    const bars = state.analysis?.frequencies?.[state.frequency]?.bars || [];
    if (!chart || !bars.length) return false;
    // Validate the captured inputs before clearing selection, which can resync the chart.
    chartRangeDraft = {...requested};
    const {start,end} = requested;
    if (!start || !end || start > end) { notice('显示区间开始日期不能晚于结束日期。'); return false; }
    const from = bars.findIndex((bar) => day(bar.time) >= start), to = bars.findLastIndex((bar) => day(bar.time) <= end);
    if (from < 0 || to < from) { notice('所选区间没有可显示的K线。'); return false; }
    updateChartSelection('clear');
    chartRangeDraft = null;
    chart.timeScale().setVisibleLogicalRange({from:from - .5,to:to + .5});
    syncChartRange();
    return true;
  }
  function zoomChart(factor) {
    if (!chart) return;
    chartRangeDraft = null;
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
    if (state.analysis?.structure_only) return `结构与量价 · 截至 ${esc(state.analysis.data_end)}。点击来源 B/S 查看首次可用时间与依据；模拟账户中查看实际买卖。`;
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
    const allEvents = reference ? evidenceUI()?.referenceMarkerInputs(state.analysis).events || [] : state.analysis?.events || [];
    const events = filteredReferenceEvents(allEvents,$('event-layer-filter')?.value || 'all',$('event-side-filter')?.value || 'all');
    $('event-count').textContent = `${events.length} / ${allEvents.length}`;
    const latest = latestSellEvent(allEvents,$('event-layer-filter')?.value || 'all');
    if ($('event-locate-s')) $('event-locate-s').disabled = !latest;
    if ($('event-filter-summary')) $('event-filter-summary').textContent = latest ? `所选图层最近 S：${day(latest.first_available_at || latest.available_at || latest.date)} · ${latest.reference_layer === 'source' ? '来源 S' : '目标变化 S'}` : '所选图层在本份结果中没有 S。';
    renderSellNavigation(allEvents);
    $('position-reference-card').innerHTML = evidenceUI()?.positionHtml(state.analysis) || '';
    $('event-context').innerHTML = nextSessionGuidance(state.analysis?.next_session_decision,state.query?.as_of || state.analysis?.as_of);
    const research = state.analysis?.research;
    $('analysis-research-evidence').hidden = !research;
    $('analysis-research-evidence').innerHTML = research ? researchEvidence({...research,...state.analysis.next_session_decision}) : '';
    if (state.analysis?.compute_info) $('analysis-compute-status').textContent = research ? researchComputeCaption(state.analysis.compute_info,state.analysis) : computeCaption(state.analysis.compute_info);
    $('event-list').innerHTML = events.length ? [...events].reverse().map((event) => `<article class="event-entry" data-event-time="${esc(day(event.time || event.date))}" data-evidence-date="${esc(day(event.evidence_date || event.time || event.date))}"><strong>${esc(event.reference_layer === 'source' ? `${event.source_label || event.source}来源 ${event.action === 'BUY' ? 'B' : 'S'}` : event.reference_display_label || `最终共识 ${event.action === 'BUY' ? 'B' : 'S'}`)}</strong><p><span class="badge ${actionClass(event.action)}">${actionShape(event.action)}${esc(actionLabel(event.action))}${reference ? '参考' : '意图'}</span></p><p class="event-reference">${esc(referencePriceText(event.reference_price,event.reference_price_date,`${actionLabel(event.action)}参考价`))}</p><p>${event.reference_layer === 'source' ? esc(event.source_relationship_text || `已确认来源证据 · ${event.consensus_qualified ? '同日最终共识通过' : '同日最终共识未通过；来源点独立保留'}`) : esc(readableReason(event))}</p><p class="event-time">发生 ${esc(display(event.time || event.date))}<br>可用 ${esc(display(event.available_at))}</p><button type="button" data-locate-event="${esc(day(event.evidence_date || event.time || event.date))}">定位主图</button></article>`).join('') : '<div class="empty-copy">所选筛选没有事件。可切换图层或 B/S；形成中的结构不代表确认信号。</div>';
  }
  function highlightEvent(time) { const dates = new Set(Array.isArray(time) ? time : [time]); document.querySelectorAll('[data-event-time]').forEach((entry) => { entry.style.background = (dates.has(entry.dataset.eventTime) || dates.has(entry.dataset.evidenceDate)) ? 'var(--accent-soft)' : ''; }); }
  function locateTime(time, openPopup = true, center = false) {
    if (!chart) return;
    chartRangeDraft = null;
    const bars = state.analysis?.frequencies?.[state.frequency]?.bars || [];
    const index = bars.findIndex((bar) => KHQuantMarkers.periodKey(bar.time,state.frequency) === KHQuantMarkers.periodKey(time,state.frequency));
    if (index >= 0) { chart.timeScale().setVisibleLogicalRange(center ? {from:Math.max(0,index - 20),to:Math.min(bars.length - 1,index + 20)} : {from:Math.max(0,index - 35),to:index + 5}); updateChartSelection(openPopup ? 'pin' : 'locate',day(bars[index].time),null,time); }
    highlightEvent(time);
  }
  function getSymbol(value) { const text = value.trim(); const known = [...state.symbolCatalog.values()].find((item) => item.symbol === text.toUpperCase() || item.symbol.slice(0,6) === text || item.name === text); if (known) return known.symbol; const match = text.match(/\d{6}(?:\.(?:SH|SZ|BJ))?/i); return match ? match[0].toUpperCase() : text; }
  function symbols(value) { return [...new Set(value.split(/[\s,，;；]+/).map(getSymbol).filter(Boolean))]; }
  function dateSpec(prefix) { const spec = {}; ['start','end'].forEach((key) => { if ($(`${prefix}-${key}`).value) spec[key] = $(`${prefix}-${key}`).value; }); if (spec.start && spec.end && spec.start > spec.end) throw new Error('开始日期不能晚于结束日期。'); return spec; }
  function consensusRequestContext(result) {
    if(result?.analysis_method && resultMethod(result)!=='legacy_reference')return {};
    const consensus = result?.trend_consensus || result?.consensus_decisions;
    const version = consensus?.version || consensus?.config?.version || result?.request?.consensus_version;
    return {...(consensus?.config ? {consensus_config:consensus.config} : {}),...(version ? {consensus_version:version} : {})};
  }
  async function analyze(query, preserveDates = false, locate = null) {
    query = {...query,analysis_mode:'reference'};
    let submittedForm=null;
    try { const form=analysisFormContext(); if(Object.entries(form).every(([key,value])=>query[key]===value))submittedForm=JSON.stringify(stableIdentity(form)); } catch(_) {}

    const sequence = ++state.analysisSequence;
    analysisLoading = true; renderMethodResult('analysis',null); resetChartSelection();
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
      state.analysis = data; state.query = {...query,...data.request,...Object.fromEntries(methodFields.filter(key=>data.method_context?.[key]!=null).map(key=>[key,data.method_context[key]])),...consensusRequestContext(data)}; state.archive = false; state.resultCreatedAt = data.run_context?.created_at; analysisLoading = false;
      analysisTransfer=submittedForm?{form:submittedForm,modelId:state.query.score_model_id||data.method_evaluation?.model_id||null}:null;
      const incoming = evidenceUI()?.mergeEvidenceRows(displayedEvidence()?.rows,displayedConsensus()?.rows) || displayedEvidence()?.rows || [];
      storeHistoryEvidence(incoming,preserveDates,analysisHistoryIdentity(data,state.query));
      if (!preserveDates) { state.dates = (data.frequencies?.['日线']?.bars || []).map((bar) => day(bar.time)); state.replayIndex = Math.max(0,state.dates.length - 1); }
      renderAnalysis(); renderEvents(); updateReplay();
      if (locate) locateTime(locate,false);
    } catch (error) {
      if (sequence !== state.analysisSequence) return;
      state.analysis = null; analysisLoading = false; renderMarkerScope(); renderResultIdentity();
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
    if(state.analysis?.structure_only) $('replay-evidence-card').innerHTML = `<strong>结构与量价 · 信息截止 ${esc(selected||'—')}</strong><p>已知收盘 ${number(row?.reference_close,2)} 元</p>${ui.rawEventsHtml(row,[],state.analysis?.consensus_decisions?.version)}`;
    if (!state.analysis) { $('history-evidence-table').innerHTML = '<p class="subtle">历史参考记录不可用，请重新分析。</p>'; return; }
    const cutoff = day(state.query?.as_of || state.analysis?.as_of || state.analysis?.data_end);
    const rows = historyRows.filter(item => rowObservable(item,state.query?.as_of || cutoff));
    historyPage = ui.historyPage(rows,historyPage).page;
    $('history-evidence-table').innerHTML = ui.historyHtml(rows,historyPage,day(selected),evidence?.statistics);
  }
  function updateReplay() {
    const available = Boolean(state.analysis) && state.dates.length > 0 && !analysisLoading && !state.archive;
    ['replay-slider','replay-date','replay-reset'].forEach((id) => { $(id).disabled = !available; });
    $('replay-slider').max = String(Math.max(0,state.dates.length - 1)); $('replay-slider').value = String(state.replayIndex);
    $('replay-date').min = state.dates[0] || ''; $('replay-date').max = state.dates[state.dates.length - 1] || ''; $('replay-date').value = state.dates[state.replayIndex] || '';
    $('replay-prev').disabled = !available || state.replayIndex === 0; $('replay-next').disabled = !available || state.replayIndex >= state.dates.length - 1;
    renderReplayEvidence();
  }
  async function replay(index) {
    if (!state.query || !state.dates.length || state.archive) return;
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
  function renderComputeResult(kind, info) { $(`${kind}-compute-result`).hidden = !state[kind]; $(`${kind}-compute-result`).textContent = methodDefinitions[resultMethod(state[kind])]?.prediction ? `本批预测 ${number(info?.prediction_calls_this_batch)} 次 · 累计 ${number(info?.prediction_calls_total)} 次${info?.score_model_id?` · 均线候选 ${info.score_model_id}`:''}。实际预测设备与质量见逐股证据。` : info?.research ? researchComputeCaption(info,state[kind]) : computeCaption(info); }
  function renderScanComputePlan() {
    const method=selectedMethod('scan'), research = method==='legacy_reference', device = $('scan-device').value;
    if(methodDefinitions[method].prediction) { $('scan-compute-plan').textContent=`结构量价在 CPU；Kronos 与均线评分按所选设备运行。每批最多 ${$('scan-max-predictions')?.value||200} 次预测、${$('scan-budget')?.value||300} 秒；未完成股票保留延后状态，可手动续批。`; return; }
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
    const method=selectedMethod(prefix), definition=methodDefinitions[method], enabled=method==='legacy_reference';
    const options=$(`${prefix}-method-options`); if(options)options.hidden=!definition.prediction;
    const scoreLabel=$(`${prefix}-score-model-label`); if(scoreLabel)scoreLabel.hidden=!definition.score;
    if(prefix==='scan') { if($('research-toolbar'))$('research-toolbar').hidden=!enabled; $('scan-research-target-help').hidden=!enabled; }
    ['model-family','usage-mode','model-policy','model','checkpoint','device'].forEach(key => { const node=$(`${prefix}-${key}`); if(node?.closest?.('label')) node.closest('label').hidden=key==='device' ? !(enabled||definition.prediction) : !enabled; });
    if(prefix === 'analysis') {
      if($('analysis-position').closest?.('label')) $('analysis-position').closest('label').hidden=!enabled;
      $('position-reference-card').hidden=!enabled || state.analysis?.structure_only === true;
      $('analysis-compute-status').hidden=!enabled;
      if($('layer-signals').closest?.('label')) $('layer-signals').closest('label').hidden=!enabled;
      $('chart-locate-target-s').hidden=!enabled;
      $('event-target-option').hidden=!enabled;
      if(!enabled && $('event-layer-filter').value === 'target') $('event-layer-filter').value='source';
    }
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
    const method=selectedMethod(prefix), definition=methodDefinitions[method];
    if (!definition.legacy) {
      const context={research:false,analysis_mode:'reference',analysis_method:method,...(methodSettings[prefix].usage_mode?{usage_mode:methodSettings[prefix].usage_mode}:{})};
      if(definition.prediction) {
        const fields={lookback:'lookback',horizon:'horizon',samples:'samples',seed:'seed',budget_seconds:'budget',max_predictions:'max-predictions'};
        Object.entries(fields).forEach(([key,suffix])=>{const value=Number($(`${prefix}-${suffix}`)?.value || methodSettings[prefix][key] || ({lookback:128,horizon:5,samples:8,seed:42,budget_seconds:300,max_predictions:prefix==='analysis'?1:200})[key]);if(!Number.isFinite(value)||!Number.isInteger(value)||value<(key==='seed'?0:key==='budget_seconds'?10:1)||value>({seed:2147483647,budget_seconds:1800,max_predictions:10000}[key]||10000))throw new Error('预测参数须为有效整数。');context[key]=value;});
        context.device=$(`${prefix}-device`).value||'auto';
        const model=$(`${prefix}-score-model`)?.value; if(definition.score && model)context.score_model_id=model;
      }
      return context;
    }
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
    return {research:true,analysis_mode:'reference',analysis_method:'legacy_reference',model_family:'current',consensus_version:'reference_position_v1',usage_mode,model_policy,...(prefix === 'analysis' && $('analysis-position').value !== '' ? {current_position_fraction:Number($('analysis-position').value) / 100} : {}),
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
    if(row.method_evaluation) {const value=row.method_evaluation,label=({deferred:'待处理',failed:'评估失败',partial:'证据不足'})[value.status]||({BUY:'B · 模型买方参考',SELL:'S · 模型卖方参考',WAIT:'观察'})[value.direction]||'未评分';return `<span class="badge">${esc(label)}</span><span class="stock-name">${esc(methodLabel(value.method))}</span><span class="stock-name">${value.score?.score!=null?`评分 ${number(value.score.score,2)} / 100`:value.status==='complete'?'预测已核验':'未完成评分'}</span>`;}

    const action = row.reference_action || row.reference_recommendation?.action || row.recommendation?.action || row.action || 'WAIT';
    const rowSchema = row.position_reference?.schema || row.strategy_version || row.version || state.scan?.strategy_version || state.scan?.trend_consensus?.version;
    const isReference = rowSchema === 'reference_position_v1';
    const label = row.category === 'excluded' ? '数据不足' : isReference ? {BUY:'B · 参考增仓',SELL:'S · 参考减仓',WAIT:'保持参考目标',HOLD:'保持参考目标'}[action] || '参考未评分' : {BUY:'共B · 最终买方参考',SELL:'共S · 最终卖方参考',WAIT:'继续观察',HOLD:'继续观察'}[action] || '继续观察';
    const trend = evidenceUI()?.trendProbabilities(row.agents?.ma);
    if (isReference) {
      const edge = trend?.available ? (Number(row.agents.ma.p_up) - Number(row.agents.ma.p_down)) * 100 : null;
      return `<span class="badge ${row.category === 'excluded' ? '' : actionClass(action)}">${esc(label)}</span><span class="stock-name">${esc(trend?.summary || '方向概率未评分')}</span>${edge == null ? '' : `<span class="stock-name">上涨减下跌 ${number(edge,2)} 个百分点</span>`}`;
    }
    return `<span class="badge ${row.category === 'excluded' ? '' : actionClass(action)}">${esc(label)}</span><span class="stock-name">${esc(trend?.summary || '趋势概率未评分')}</span><span class="stock-name">买方支持 ${number(row.buy_base_score == null ? null : row.buy_base_score * 100,2)} 分 · 卖方支持 ${number(row.sell_base_score == null ? null : row.sell_base_score * 100,2)} 分</span>`;
  }
  function diagnostic(row) {
    if (!row || typeof row !== 'object') return display(row);
    const reasons = {unsupported_board:'该板块尚不支持模拟成交',insufficient_seasoned_bars:'有效历史成交记录不足',legacy_fuyao_price_basis_unverified:'当日或前日含旧复权未核验记录，拒绝成交',unverified_trade_price:'成交价格尚未核验',suspended_or_zero_volume:'停牌或成交量为零',conservative_upper_price_guard:'触发保守上限价格保护',conservative_lower_price_guard:'触发保守下限价格保护',t_plus_one:'当日买入库存尚不可卖出',insufficient_cash_or_prior_liquidity:'现金或前日流动性不足',broker_rejected:'资金或库存检查未通过',entry_plan_not_active:'当前没有可执行的新入场计划',entry_plan_consumed:'该入场计划已使用，禁止重复追补',entry_plan_identity_mismatch:'入场计划股票或身份不匹配',entry_plan_invalid_time_contract:'入场计划时间口径不完整',entry_plan_not_available:'信号计划在该开盘时尚不可用',entry_plan_expired:'已错过计划对应开盘，入场计划过期',entry_plan_invalid_risk_contract:'入场区间或风险预算不完整',entry_plan_price_above_ceiling:'实际执行价高于参考入场上限，取消追价',entry_plan_price_below_floor:'实际执行价低于参考入场下限，取消入场',native_target_exit:'原生目标仓位触发退出',actual_account_stop:'实际持仓成本或冻结结构失效价触发退出'};
    return [row.date ? day(row.date) : '',row.symbol || '',row.action ? actionLabel(row.action) : '',reasons[row.reason] || row.error || row.reason || display(row)].filter(Boolean).join(' · ');
  }
  function renderDiagnostics(id, rows) { const list = Array.isArray(rows) ? rows : rows ? [rows] : []; $(id).hidden = !list.length; $(id).querySelector('.diagnostics').innerHTML = list.map((row) => `<div>${esc(diagnostic(row))}</div>`).join(''); }
  function scanTimingCell(row) {
    const evaluation=row.method_evaluation;
    if(evaluation?.status==='deferred')return '尚未计算<span class="stock-name">本批预算延后</span>';
    if(evaluation?.status==='failed')return `${row.data_end?`行情 ${esc(row.data_end)}`:'行情状态未核验'}<span class="stock-name">评估失败，查看原因</span>`;
    if(evaluation)return `行情 ${esc(row.data_end||'未提供')}<span class="stock-name">${row.point_confirmed_at?`结构确认 ${esc(row.point_confirmed_at)}`:'结构时间见依据'}</span>${row.available_at?`参考可用 ${esc(display(row.available_at))}`:''}`;
    return `${row.category?`行情 ${esc(row.data_end||'缺失')}<span class="stock-name">结构确认 ${esc(row.point_confirmed_at||'未确认')}</span>`:''}${esc(display(row.available_at))}`;
  }
  function scanEventText(row) {
    if(row.method_evaluation?.status==='deferred')return '尚未计算';
    if(row.method_evaluation?.status==='failed')return '未产生结果';
    if(row.method_evaluation && !(row.events||row.source_reference_points||[]).length)return '无当日来源事件';
    return readableEvent(row);
  }
  function scanReasonText(row) { return row.method_evaluation ? (row.method_evaluation.missing_reasons||[]).map(methodReason).join('；')||'结构、预测与评分见下方依据' : readableReason(row); }
  function renderScan() {
    renderMethodResult('scan',state.scan);
    globalThis.KHMarketHistory?.render($('scan-market-history'),state.scan?.market_history,{runId:state.scan?.run_id,request,onAnalyze:({symbol,date}) => analyzeSymbol(symbol,date,state.scan).catch(showError)});
    const query = $('scan-search').value.trim().toLowerCase(), filter = $('scan-filter').value;
    const reference = referenceMode(state.scan);
    const all = (state.scan?.rows || []).map((row) => reference ? {...row,analysis_mode:'reference'} : row);
    renderComputeResult('scan',state.scan?.compute_info);
    const filtered = state.scan?.scan_view?.paged ? all : all.filter((row) => (!query || `${row.symbol} ${row.name || state.symbolCatalog.get(row.symbol)?.name || ''} ${row.signal || ''} ${display(row.reason)} ${readableEvent(row)} ${readableReason(row)}`.toLowerCase().includes(query)) && (filter === 'all' || filter === 'favorites' && state.favorites.has(row.symbol) || filter === (row.category || actionClass(row.action)) || ['exit','sell'].includes(filter) && (row.category === 'exit' || row.reference_action === 'SELL' || row.method_evaluation?.direction==='SELL')));
    const total=state.scan?.scan_view?.paged?state.scan.scan_view.total:filtered.length;
    state.scanPage = Math.max(0,Math.min(state.scanPage,Math.ceil(total / PAGE_SIZE) - 1));
    const page = state.scan?.scan_view?.paged ? filtered : filtered.slice(state.scanPage * PAGE_SIZE,(state.scanPage + 1) * PAGE_SIZE);
    $('scan-count').textContent = number(state.scan?.scan_view?.total_rows ?? all.length);
    const categories = {buy:'买入候选',watch:'继续观察',exit:'持仓退出研究',excluded:'排除 / 不足'};
    $('scan-rows').innerHTML = page.length ? page.map((row) => `<tr><td><button type="button" class="favorite" data-favorite="${esc(row.symbol)}" aria-label="关注 ${esc(row.symbol)}" aria-pressed="${state.favorites.has(row.symbol)}" ${!state.personalLoaded || state.watchlistBusy ? 'disabled' : ''}>${state.favorites.has(row.symbol) ? '★' : '☆'}</button></td><td><strong>${esc(row.symbol)}</strong><span class="stock-name">${esc(row.name || state.symbolCatalog.get(row.symbol)?.name || '')}</span></td><td>${esc(scanEventText(row))}</td><td>${scanActionCell(row)}</td><td>${scanTimingCell(row)}</td><td>${row.method_evaluation?'模型参考无目标仓位':evidenceUI()?.positionSummaryHtml(row) || '不可用'}</td><td class="reason">${esc(scanReasonText(row))}${ruleOriginal(row)}${row.method_evaluation?`<details><summary>预测与评分依据</summary>${methodEvaluationHtml(row.method_evaluation,true)}</details>`:researchEvidence(row)}</td><td><button type="button" class="link-button" data-analyze-symbol="${esc(row.symbol)}" data-analysis-source="scan" ${row.data_end ? `data-analyze-end="${esc(row.data_end)}"` : ''}>分析</button></td></tr>`).join('') : `<tr><td colspan="8" class="empty-cell">${state.scan ? '没有符合筛选条件的股票。' : '开始扫描后，结果显示在这里。'}</td></tr>`;
    pagination('scan',total,state.scanPage);
    $('scan-research-result').hidden = !reference && !state.scan?.model_gate;
    if (reference) $('scan-research-result').textContent = methodDefinitions[resultMethod(state.scan)]?.prediction ? 'B/S 为所选方法的模型方向参考；结构来源、预测与评分分别保留，缺输入或预算延后不当作零分。' : 'B/S 为参考方向；买方与卖方支持分分别展示。缺少合格模型时趋势概率不可用。';
    if (state.scan) { $('scan-coverage').textContent = `覆盖 ${number(state.scan.coverage?.success)} / ${number(state.scan.coverage?.requested)} 只成功 · 失败 ${number((state.scan.failures || []).length)}${state.scan.coverage?.deferred ? ` · 延后 ${number(state.scan.coverage.deferred)}` : ''}${state.scan.coverage?.unavailable ? ` · 输入不可用 ${number(state.scan.coverage.unavailable)}` : ''}${state.scan.data_range?.requested_end ? ` · 请求截止 ${state.scan.data_range.requested_end} · 实际行情 ${state.scan.data_end || state.scan.data_range.end || '缺失'}` : ''}`; renderDiagnostics('scan-failures',state.scan.failures);
      if (!reference && state.scan.model_gate) $('scan-research-result').textContent = `${gateCaption(state.scan.model_gate)} · ${Object.entries(categories).map(([key,label]) => `${label} ${number(state.scan.category_counts?.[key] || 0)}`).join(' · ')}。${state.scan.model_gate.reason || ''}`;
    }
  }
  let scanPageSequence=0;
  async function loadScanPage(){
    if(!state.scan?.scan_view?.paged){renderScan();return;}
    const sequence=++scanPageSequence, runId=state.scan.run_id, filter=$('scan-filter').value;
    const params=new URLSearchParams({view:'scan',page:state.scanPage+1,page_size:PAGE_SIZE,q:$('scan-search').value.trim(),category:filter});
    if(filter==='favorites')params.set('symbols',[...state.favorites].join(','));
    const value=await request(`/api/runs/${encodeURIComponent(runId)}?${params}`);
    if(sequence!==scanPageSequence||state.scan?.run_id!==runId)return;
    state.scan=value;renderScan();
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
  function applyWatchlist(data) { state.watchlist = data; state.favorites = new Set((data.items || []).map((item) => item.symbol)); renderWatchlist(); if(state.scan?.scan_view?.paged && $('scan-filter').value==='favorites')loadScanPage().catch(showError);else renderScan(); }

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
    if(!globalThis.KHQuantTaskHistory) $('task-list').innerHTML = state.tasks.length ? state.tasks.map(task => `<article class="task-row"><div class="task-heading"><div><strong>${esc(kindNames[task.kind] || '当前研究任务')}</strong><small>${esc(task.created_at || '')} · ${esc(task.run_id || task.job_id)}</small></div><div class="task-actions"><span class="badge">${esc(statusNames[task.status] || task.status)}</span>${taskActions(task)}</div></div><div class="progress-track"><i style="width:${Math.max(0,Math.min(100,Number(task.progress) || 0))}%"></i></div><p class="task-detail ${task.error ? 'task-error' : ''}">${esc(task.error ? display(task.error) : taskDetail(task))}</p></article>`).join('') : '<div class="empty-copy">尚无任务，可发起训练、更新行情、扫描。</div>';
    ['scan'].forEach(kind => { const task = state.tasks.find(item => item.kind === kind && item.job_id === state.tracked[kind]), host = $(kind + '-progress'); host.hidden = !task; if (task) host.innerHTML = `<div class="task-actions">${taskActions(task)}</div>${esc(statusNames[task.status] || task.status)} · ${number(task.progress)}%<br>${esc(task.error ? display(task.error) : taskDetail(task))}`; });
    if($('scan-resume'))$('scan-resume').disabled=state.tasks.some(task=>task.kind==='scan'&&activeStatuses.has(task.status));
    ['scan','update'].forEach(kind => { const button = $(kind === 'update' ? 'update-data' : `start-${kind}`); button.disabled = state.tasks.some(task => task.kind === kind && activeStatuses.has(task.status)); });
  }
  function showResult(kind,result,navigate = false) {
    if (kind.startsWith('board_')) { globalThis.KHQuantBoardPanel?.showResult(result); if (navigate) location.hash = 'scan'; return; }
    if (kind === 'scan') { if(navigate)globalThis.KHQuantBoardPanel?.showStock?.(); state.scan = result; state.scanPage = 0; ++scanPageSequence; if(result?.analysis_method)applyResearchContext('scan',resultResearchContext(result)); if(result?.scan_view?.paged && ($('scan-search').value.trim() || $('scan-filter').value!=='all'))loadScanPage().catch(showError);else renderScan(); }
    else if (kind !== 'update') { globalThis.KHQuantTraining?.showResult(result); if (navigate) location.hash = 'training'; return; }
    if (navigate) location.hash = kind === 'update' ? 'data' : kind;
  }
  async function refreshTasks() {
    if(document.hidden){clearTimeout(state.pollTimer);return;}
    const sequence = ++state.taskSequence;
    clearTimeout(state.pollTimer);
    const data = await request('/api/tasks');
    if (sequence !== state.taskSequence) return;
    state.tasks = Array.isArray(data) ? data : data.tasks || [];
    renderTasks();

    for (const task of state.tasks) {
      if (!['succeeded','partial'].includes(task.status) || state.delivered.has(task.job_id) || task.job_id !== state.tracked[task.kind]) continue;
      state.delivered.add(task.job_id);
      try {
        if (task.kind === 'update') await refreshData();
        else {
          let data;
          if (task.kind === 'scan') {
            data = task.run_id ? await request(runResultUrl(task.run_id,'scan')) : null;
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
    if (!document.hidden && sequence === state.taskSequence && state.tasks.some((task) => activeStatuses.has(task.status))) state.pollTimer = setTimeout(() => refreshTasks().catch((error) => { showError(error); state.pollTimer = setTimeout(() => refreshTasks().catch(showError),5000); }),2500);
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
  function runMethodCaption(run) {
    const summary=run.summary||{},method=summary.analysis_method||summary.method_context?.analysis_method||run.analysis_method;
    const knownMethod=method&&methodDefinitions[normalizeMethod(method)];
    const schema=summary.strategy_version;
    const current=knownMethod&&normalizeMethod(method)!=='legacy_reference'||schema==='reference_position_v1'||schema==='account_free_methods_v1';
    return `${knownMethod?methodLabel(method):schema==='account_free_methods_v1'?'结构、预测与均线评分':schema||'版本未记录'}${schema&&!current?' · 只读归档':''}`;
  }
  async function refreshRuns() {
    const data = await request('/api/runs?limit=30');
    const rows = Array.isArray(data) ? data : data.runs || [];
    $('run-list').innerHTML = rows.length ? rows.map((run) => `<article class="run-row"><div><strong>${esc(kindNames[run.kind] || run.kind || '分析运行')}</strong><small>${esc(run.run_id)} · ${esc(run.created_at || '')}<br>${esc(runMethodCaption(run))} · 行情 ${esc(run.summary?.data_range?.end || '—')}</small></div><button type="button" data-run-id="${esc(run.run_id)}" data-run-kind="${esc(run.kind || '')}">打开结果</button></article>`).join('') : '<div class="empty-copy">尚无新系统运行记录。</div>';
  }
  function resultResearchContext(result) {
    const explicit=result?.analysis_method || result?.method_context?.analysis_method || result?.request?.analysis_method;
    if(explicit && normalizeMethod(explicit)!=='legacy_reference') { const source={...result?.request,...result?.method_context,analysis_method:explicit}; return {research:false,analysis_mode:'reference',...Object.fromEntries(methodFields.filter(key=>source[key]!=null).map(key=>[key,source[key]]))}; }
    const source = result?.request || result || {}, context = {analysis_mode:'reference'};
    ['research','model_family','usage_mode','model_policy','model_run_id','model_fold','calendar_run_id','device','current_position_fraction'].forEach((key) => { if (source[key] != null) context[key] = source[key]; });
    context.consensus_version = source.consensus_version || resultSchema(result); context.model_family ||= result?.model_family || 'current';
    ['model_policy','model_fold','calendar_run_id','current_position_fraction'].forEach((key) => { if (context[key] == null && result?.[key] != null) context[key] = result[key]; });
    context.model_run_id ||= result?.research?.model_run_id || result?.model_run_id;
    context.model_policy ||= context.model_fold ? 'pinned' : 'auto';
    const config = source.consensus_config || result?.consensus_config || result?.trend_consensus?.config;
    if (config) context.consensus_config = config;
    if (!('research' in context)) context.research = Boolean(result?.research || result?.model_gate);
    if (context.research) { context.usage_mode ||= 'historical';  }
    return context;
  }
  function validateResearchContext(context) {
    if(context.analysis_method && !methodDefinitions[normalizeMethod(context.analysis_method)])throw new Error('未知分析方法，无法恢复该结果。');
    if (context.research !== false && ((context.model_family && context.model_family !== 'current') || (context.consensus_version && context.consensus_version !== 'reference_position_v1'))) throw new Error('该历史模型运行已退役，只能查看归档证据，不能恢复当前操作上下文。');
    if (context.research !== false && context.usage_mode && !['historical','forward'].includes(context.usage_mode)) throw new Error('旧用途已退役，不恢复资格；请重新选择当前研究用途。');
  }
  function applyResearchContext(prefix, context) {
    validateResearchContext(context);
    if(context.analysis_method && normalizeMethod(context.analysis_method)!=='legacy_reference') { $(`${prefix}-mode`).value=normalizeMethod(context.analysis_method); applyMethodSettings(prefix,context); if(context.device)$(`${prefix}-device`).value=context.device; delete state.pendingResearchContexts[prefix]; updateResearchControls(prefix); return; }
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
    return `/api/runs/${encodeURIComponent(id)}${kind === 'scan' ? `?view=scan&page=1&page_size=${PAGE_SIZE}` : ''}`;
  }
  function prepareAnalysisView(result, id = null) {
    if (!result || typeof result !== 'object' || !result.symbol || !Array.isArray(result.frequencies?.['日线']?.bars)) throw new Error('分析结果缺少股票或日线身份，保留上一份结果。');
    if (id && result.run_id && result.run_id !== id) throw new Error('分析结果运行身份不一致，保留上一份结果。');
    if (result.request?.symbol && result.request.symbol !== result.symbol) throw new Error('分析结果股票身份不一致，保留上一份结果。');
    const archive = archivedResult(result);
    const displayQuery = {symbol:result.symbol,start:result.request?.start ?? null,end:result.request?.end || result.data_end,as_of:result.as_of || result.request?.as_of};
    // Viewing an archived result never restores retired ids, policies or configs into an execution request.
    const query = archive ? displayQuery : {...result.request,...resultResearchContext(result),...consensusRequestContext(result),...displayQuery};
    if (!archive) validateResearchContext(query);
    for (const key of ['source_reference_points','reference_points','events']) if (result[key] != null && !Array.isArray(result[key])) throw new Error('分析结果事件契约无效，保留上一份结果。');
    const dates = result.frequencies['日线'].bars.map(bar => day(bar.time));
    if (dates.some(date => !/^\d{4}-\d{2}-\d{2}$/.test(date)) || dates.some((date,index) => index > 0 && date <= dates[index - 1])) throw new Error('分析结果日线时序无效，保留上一份结果。');
    return {query,archive,dates,historyIdentity:analysisHistoryIdentity(result,query)};
  }
  function currentReanalysisQuery(result) {
    if (!result?.symbol) throw new Error('缺少可重新分析的股票。');
    return {symbol:result.symbol,analysis_mode:'reference',start:result.request?.start ?? null,
      end:result.data_end || result.request?.end,as_of:result.data_end || day(result.as_of),research:true,
      model_family:'current',consensus_version:'reference_position_v1',usage_mode:'historical',model_policy:'auto',device:'auto'};
  }
  async function reanalyzeCurrentResult() {
    const query = currentReanalysisQuery(state.analysis);
    applyResearchContext('analysis',query);
    $('analysis-symbol').value = query.symbol; $('analysis-start').value = query.start || ''; $('analysis-end').value = query.end || '';
    setDateMode('analysis','historical');
    await analyze(query);
  }
  async function openRun(id, kindHint = '') {
    const sequence = ++openRunSequence, analysisAtStart = state.analysisSequence;
    let data;
    try { data = await request(runResultUrl(id,kindHint)); }
    catch (error) { if (sequence !== openRunSequence) return; throw error; }
    if (sequence !== openRunSequence) return;
    const result = data.result || data;
    const kind = data.kind || kindHint || (result.model_manifest && result.evaluation ? 'model_training' : result.rows ? 'scan' : result.data_status ? 'update' : 'analysis');
    if (kind !== 'analysis') { showResult(kind,result,true); return; }
    if (analysisAtStart !== state.analysisSequence) return;
    // All identity and execution-context checks finish before the visible result is replaced.
    const prepared = prepareAnalysisView(result,id);
    if (!prepared.archive) applyResearchContext('analysis',prepared.query);
    ++state.analysisSequence;
    resetChartSelection(); analysisLoading = false;
    $('analysis-form').querySelector('[type=submit]').disabled = false;
    state.analysis = result; state.query = prepared.query; state.archive = prepared.archive;
    state.resultCreatedAt = data.created_at || result.run_context?.created_at;
    storeHistoryEvidence(evidenceUI()?.mergeEvidenceRows(displayedEvidence()?.rows,displayedConsensus()?.rows) || displayedEvidence()?.rows || [],false,prepared.historyIdentity); historyPage = 0;
    if (!prepared.archive) {
      $('analysis-symbol').value = prepared.query.symbol; $('analysis-start').value = prepared.query.start || ''; $('analysis-end').value = prepared.query.end || result.data_end;
      setDateMode('analysis','historical');
    }
    state.dates = prepared.dates; state.replayIndex = Math.max(0,state.dates.length - 1);
    location.hash = 'analysis'; showView('analysis'); renderAnalysis(); renderEvents(); updateReplay();
  }
  async function analyzeSymbol(symbol,end,source = null) {
    $('analysis-symbol').value = symbol;
    location.hash = 'analysis';
    const context = source ? archivedResult(source) ? currentReanalysisQuery({...source,symbol,data_end:end || source.data_end}) : resultResearchContext(source) : researchSpec('analysis');
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
    document.addEventListener('visibilitychange',()=>{clearTimeout(state.pollTimer);if(!document.hidden)refreshTasks().catch(showError);});
    window.addEventListener('khquant:task-history-changed',()=>{refreshTasks().catch(showError);refreshRuns().catch(showError);});
    window.addEventListener('keydown',(event) => { if (event.key === 'Escape') updateChartSelection('clear'); });
    $('chart-evidence-reset').addEventListener('click',() => updateChartSelection('clear'));
    $('theme-toggle').addEventListener('click',() => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
    $('page-error').querySelector('button').addEventListener('click',() => { $('page-error').hidden = true; });
    $('analysis-symbol').addEventListener('input',() => { updateChartSelection('clear'); clearTimeout(searchTimer); searchTimer = setTimeout(() => searchSymbols($('analysis-symbol').value).catch(showError),250); });
    $('analysis-form').addEventListener('submit',async (event) => { event.preventDefault(); try { if (dateModes.analysis === 'latest') await refreshLatestDates(); await analyze({symbol:getSymbol($('analysis-symbol').value),...dateSpec('analysis'),...researchSpec('analysis')}); } catch (error) { showError(error); } });
    document.querySelectorAll('[data-frequency]').forEach((button) => button.addEventListener('click',() => { resetChartSelection(); state.frequency = button.dataset.frequency; document.querySelectorAll('[data-frequency]').forEach((item) => item.setAttribute('aria-pressed',String(item === button))); renderAnalysis(); }));
    ['ma','boll','chips','fractals','pens','zones','divergences','signals','sources','wyckoff','volume'].forEach((name) => $(`layer-${name}`).addEventListener('change',() => { resetChartSelection(); renderAnalysis(false); }));
    ['event-layer-filter','event-side-filter'].forEach(id => $(id)?.addEventListener('change',() => { renderEvents(); renderMarkerScope(); }));
    $('event-locate-s')?.addEventListener('click',() => locateLatestSell());
    $('chart-locate-source-s')?.addEventListener('click',() => locateLatestSell('source'));
    $('chart-locate-target-s')?.addEventListener('click',() => locateLatestSell('target'));
    $('analysis-current-recompute')?.addEventListener('click',() => reanalyzeCurrentResult().catch(showError));
    ['start','end','mode','model-family','usage-mode','model-policy','model','checkpoint','device','position'].forEach((name) => $(`analysis-${name}`).addEventListener('change',() => updateChartSelection('clear')));
    ['chart-range-start','chart-range-end'].forEach((id) => {
      $(id).addEventListener('input',captureChartRangeDraft);
      $(id).addEventListener('change',() => { captureChartRangeDraft(); updateChartSelection('clear'); });
    });
    $('chart-range-apply').addEventListener('click',() => { const requested = captureChartRangeDraft(); applyChartRange(requested); });
    $('chart-zoom-in').addEventListener('click',() => zoomChart(.7)); $('chart-zoom-out').addEventListener('click',() => zoomChart(1.4));
    $('chart-range-reset').addEventListener('click',() => { chartRangeDraft = null; if (chart) { chart.timeScale().fitContent(); syncChartRange(); } });
    $('replay-slider').addEventListener('change',() => replay(Number($('replay-slider').value)));
    $('replay-prev').addEventListener('click',() => replay(state.replayIndex - 1)); $('replay-next').addEventListener('click',() => replay(state.replayIndex + 1));
    $('replay-date').addEventListener('change',() => { const date = $('replay-date').value; const index = state.dates.findLastIndex((value) => value <= date); if (index < 0) notice('请选择可用行情范围内的日期'); else replay(index); });
    $('replay-reset').addEventListener('click',() => { if (!state.query) return; const {as_of:_,...query} = state.query; state.replayIndex = Math.max(0,state.dates.length - 1); analyze(query,true); });
    $('scan-form').addEventListener('submit',(event) => { event.preventDefault(); try { submitTask('scan',{symbols:symbols($('scan-symbols').value),...dateSpec('scan'),...researchSpec('scan')}); } catch(error) { showError(error); } });
    ['analysis','scan'].forEach((prefix) => {
      ['mode','model-family','usage-mode','model-policy','model'].forEach((name) => $(`${prefix}-${name}`).addEventListener('change',() => {
        if (name === 'model-family' || name === 'model' && $(`${prefix}-model`).value !== state.pendingResearchContexts[prefix]?.model_run_id) delete state.pendingResearchContexts[prefix];
        if (name === 'model-family') populateResearchModels(prefix);
        if (name === 'mode') { if(selectedMethod(prefix)==='legacy_reference')refreshResearch().catch(showError); else if(methodDefinitions[selectedMethod(prefix)].prediction)refreshMethodModels().catch(showError); }
        updateResearchControls(prefix);
        if (prefix === 'scan') renderScanComputePlan();
      }));
      $(`${prefix}-checkpoint`).addEventListener('change',() => renderCheckpointHelp(prefix));
      $(`${prefix}-end`).addEventListener('change',() => renderCheckpointHelp(prefix));
    });
    $('scan-resume')?.addEventListener('click',()=>{if(!state.scan?.run_id)return; const source=state.scan; submitTask('scan',{...(source.request||{}),...resultResearchContext(source),symbols:source.request?.symbols||source.symbols||[],start:source.request?.start||source.data_range?.start,end:source.request?.end||source.data_range?.requested_end||source.data_end,resume_scan_id:source.run_id});});
    $('refresh-research').addEventListener('click',() => refreshResearch().catch(showError));
    ['scan-search','scan-filter'].forEach((id) => $(id).addEventListener(id === 'scan-search' ? 'input' : 'change',async () => {
      state.scanPage = 0; if(state.scan?.scan_view?.paged){await loadScanPage().catch(showError);return;} renderScan();
      const query = $('scan-search').value.trim();
      if (id === 'scan-search' && query && (state.scan?.rows || []).some((row) => !row.name)) {
        try { await searchSymbols(query,false); if ($('scan-search').value.trim() === query) renderScan(); }
        catch (error) { showError(error); }
      }
    }));
    ['prev','next'].forEach((direction) => $(`scan-${direction}`).addEventListener('click',() => { state.scanPage += direction === 'prev' ? -1 : 1; loadScanPage().catch(showError); }));
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
      if (button.dataset.evidenceReplay || button.dataset.historyDate) { const time = button.dataset.evidenceReplay || button.dataset.historyDate; const index = state.dates.findIndex((item) => item === time); if (index >= 0 && !analysisLoading) { if (state.archive) locateTime(time); else replay(index).catch(showError); } }
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
    bind(); ['analysis','scan'].forEach(prefix=>{const select=$(`${prefix}-mode`);select.innerHTML=Object.entries(methodDefinitions).map(([key,value])=>`<option value="${key==='legacy_reference'?'research':key}" ${key==='structure'?'selected':''}>${esc(value.label)}</option>`).join('');updateResearchControls(prefix);}); route();
    refreshLatestDates().catch(showError);
    const results = await Promise.allSettled([refreshData(),searchSymbols(),refreshTasks(),refreshRuns(),ensurePersonal(),refreshCompute(),...(['#training','#parameters','#scan'].includes(location.hash)?[refreshResearch()]:[])]);
    results.forEach((result) => { if (result.status === 'rejected') showError(result.reason); });
    if (results[0].status === 'rejected') { $('connection-state').textContent = '数据检查失败'; $('connection-state').className = 'status-dot failed'; $('data-detail').textContent = '数据状态请求失败，请检查服务后刷新。'; }
    if (results[1].status === 'fulfilled' && results[1].value.length && !$('analysis-symbol').value) { $('analysis-symbol').value = results[1].value[0].symbol; refreshLatestDates().catch(showError); }
  }
  init().catch(showError);
})();
