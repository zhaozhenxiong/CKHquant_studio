/* Saved stock-method evidence grouped by verified board membership; every request is read-only. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id), esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=value=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('zh-CN',{maximumFractionDigits:2}):'—';
  const pct=value=>typeof value==='number'&&Number.isFinite(value)?num(value*100)+'%':'—';
  const state={visible:false,selection:null,scope:null,runs:[],result:null,board:null,members:null,page:1,sequence:0,memberSequence:0,listPage:1,listTotal:0};
  let chart=null, booted=false;
  const api=async path=>{const token=sessionStorage.getItem('khquant_api_token'),response=await fetch(path,{credentials:'same-origin',headers:token?{Authorization:'Bearer '+token}:{}});let value;try{value=await response.json();}catch(_){throw Error('方法结果响应无法解析');}if(!response.ok)throw Error(value.detail?.message||value.detail||`方法结果读取失败（${response.status}）`);return value;};
  const combination=method=>Boolean(KHQuantMethods.definitions[method]?.prediction);
  function params(selection=state.selection) {return new URLSearchParams({method:selection.method,as_of:selection.as_of});}
  function caption(context={}) {return `输入 ${context.lookback??'—'} 根 · 未来 ${context.horizon??'—'} 日 · ${context.samples??'—'} 条路径 · 种子 ${context.seed??'—'} · 模型 ${context.score_model_id?String(context.score_model_id).slice(0,12)+'…':'无评分模型'}`;}
  function identity(value,run=state.selection?.run_id) {const selection=value.selection||value;if((selection.run_id||value.run_id)!==run||(selection.method||value.analysis_method)!==state.selection.method||(selection.as_of||value.data_end)!==state.selection.as_of)throw Error('保存结果的方法、日期或运行身份不一致；未展示该结果。');}
  function message(text) {$('board-method-state').textContent=text;}
  function clearResult() {++state.memberSequence;state.result=null;state.members=null;state.board=null;$('board-method-summary').innerHTML='';$('board-method-rows').innerHTML='';$('board-method-detail').hidden=true;$('board-prediction').hidden=true;chart?.clear();}
  function renderRuns() {$('board-method-run').innerHTML='<option value="">请选择匹配的已保存扫描</option>'+state.runs.map(row=>`<option value="${esc(row.run_id)}">${esc(row.created_at||row.run_id)} · ${esc(row.status||'')} · ${esc(caption(row.context))}</option>`).join('');$('board-method-run').value=state.selection?.run_id||'';$('board-method-more').hidden=state.runs.length>=state.listTotal;}
  function renderBoards() {
    if(!state.result)return;
    $('board-method-summary').innerHTML=KHQuantMethodPresentation.summaryHtml(state.result.population_summary);
    const kind=$('board-method-kind').value;
    const rows=(state.result.boards||[]).filter(row=>kind==='all'||row.kind===kind);
    $('board-method-rows').innerHTML=rows.map(row=>{const c=row.counts||row,f=row.forecast||{},s=row.score||{},coverage=row.coverage||{};return `<tr><td><button type="button" class="link-button" data-method-board="${esc(row.board_id)}">${esc(row.name||row.board_id)}</button><small>${esc(row.kind)}</small></td><td>B ${num(c.buy)} / S ${num(c.sell)} / WAIT ${num(c.wait)}<br><small>不可用 ${num(c.unavailable)} · 失败 ${num(c.failed)} · 待处理 ${num(c.deferred)} · 未纳入 ${num(c.not_in_scan)}</small></td><td>${num(c.complete)} / ${num(c.expected)} · ${pct(coverage.method_complete_ratio)}<br><small>${row.ranking_eligible?'覆盖资格通过':'覆盖不足，不参与合格排名'}</small></td><td>成员中位涨跌均值 ${pct(f.mean_member_return_median)}<br><small>上涨采样比例均值 ${pct(f.mean_member_up_path_fraction)} · ${num(f.count)} 只有效预测</small></td><td>${num(s.mean)}<br><small>${num(s.count)} 只有效评分</small></td></tr>`;}).join('')||'<tr><td colspan="5">此范围没有可展示的板块成员快照。</td></tr>';
    $('board-method-rows').querySelectorAll('[data-method-board]').forEach(button=>button.addEventListener('click',()=>openBoard(button.dataset.methodBoard).catch(fail)));
  }
  function fail(error) {message(error.message||String(error));}
  async function chooseRun(run) {
    const sequence=++state.sequence;clearResult();state.selection={...state.selection,run_id:run};renderRuns();if(!run){message('请选择已保存结果；切换方法或日期不会发起扫描。');return;}
    message('正在读取保存扫描及板块成员证据…');
    try {const value=await api(`/api/boards/method-scans/${encodeURIComponent(run)}?${params()}`);if(sequence!==state.sequence)return;identity(value,run);if(!value.identity?.summary_sha256)throw Error('保存扫描缺少成员快照核验身份');
      state.result=value;const context={...value.method_context,analysis_method:state.selection.method};
      KHQuantMethods.adoptScanContext(context,state.selection.as_of);state.selection.method_context=context;state.scope=selectionKey(KHQuantMethods.scanSelection());
      $('board-method-context').textContent=`已保存结果 ${run} · ${KHQuantMethods.label(state.selection.method)} · 截止 ${state.selection.as_of} · ${caption(context)}。已将保存参数载入扫描草稿。`;
      message('仅汇总所选保存扫描中的个股方法证据；原生结构与资金统计独立列于下方。');renderBoards();
    } catch(error) {if(sequence!==state.sequence)return;clearResult();message(error.message||String(error));}
  }
  function selectionKey(selection) {return JSON.stringify([selection.method,selection.as_of,selection.method_context,selection.run_id||null]);}
  async function load(selection) {
    const sequence=++state.sequence;state.selection={...selection};state.scope=selectionKey(selection);state.runs=[];state.listPage=1;state.listTotal=0;clearResult();renderRuns();
    $('board-method-mode').value=selection.method;$('board-method-date').value=selection.as_of||'';$('board-method-context').textContent='';
    if(!combination(selection.method)){message('结构与量价及历史对照请查看下方原生来源；组合方法板块汇总需选择对应已保存扫描。');return;}
    if(!selection.as_of){message('请先选择截止日期。');return;}
    message('正在查找方法与日期匹配的保存扫描…');
    try {const value=await api('/api/boards/method-scans?'+params(selection));if(sequence!==state.sequence)return;
      if(value.selection?.method!==selection.method||value.selection?.as_of!==selection.as_of)throw Error('候选扫描的方法或日期不匹配');
      state.runs=value.runs||[];state.listTotal=value.pagination?.total||state.runs.length;renderRuns();
      if(selection.run_id){if(!state.runs.some(row=>row.run_id===selection.run_id))state.runs.unshift({run_id:selection.run_id,context:selection.method_context,status:'当前个股结果'});await chooseRun(selection.run_id);}
      else {state.selection.run_id=null;renderRuns();message(state.runs.length?'请选择匹配的保存扫描；选择后显示其冻结参数。':'尚未生成匹配的保存扫描；可回到个股页显式开始扫描。');}
    } catch(error) {if(sequence===state.sequence)fail(error);}
  }
  async function moreRuns(){const sequence=state.sequence,query=params();query.set('page',state.listPage+1);$('board-method-more').disabled=true;try{const value=await api('/api/boards/method-scans?'+query);if(sequence!==state.sequence)return;if(value.selection?.method!==state.selection.method||value.selection?.as_of!==state.selection.as_of)throw Error('候选身份已变化');++state.listPage;state.listTotal=value.pagination?.total||0;state.runs=[...new Map([...state.runs,...(value.runs||[])].map(row=>[row.run_id,row])).values()];renderRuns();}finally{$('board-method-more').disabled=false;}}
  async function openBoard(boardId) {state.board=(state.result?.boards||[]).find(row=>row.board_id===boardId);if(!state.board)return;state.page=1;$('board-method-member-filter').value='all';$('board-method-member-search').value='';$('board-method-detail').hidden=false;await loadMembers();}
  async function loadMembers() {
    if(!state.board||!state.result)return;const sequence=++state.memberSequence,boardId=state.board.board_id,run=state.selection.run_id;
    const query=params();query.set('board_id',boardId);query.set('page',state.page);query.set('page_size',40);query.set('category',$('board-method-member-filter').value||'all');query.set('q',$('board-method-member-search').value.trim());
    state.members=null;$('board-method-member-state').textContent='正在读取成员…';$('board-method-members').innerHTML='';
    try {const value=await api(`/api/boards/method-scans/${encodeURIComponent(run)}/members?${query}`);if(sequence!==state.memberSequence)return;identity(value,run);if(!state.result?.identity?.summary_sha256||value.identity?.summary_sha256!==state.result.identity.summary_sha256)throw Error('板块成员快照已变化，请刷新保存结果并重新选择板块。');if(value.board?.board_id!==boardId)throw Error('板块成员身份不一致');state.members=value;
      const c=value.board.counts||value.board;$('board-method-detail-title').textContent=`${value.board.name||boardId} · ${KHQuantMethods.label(state.selection.method)}`;
      $('board-method-member-state').textContent=`截止 ${state.selection.as_of} · 计算成员 ${num(c.expected)} · 本次匹配 ${num(c.matched)} · 有效 ${num(c.complete)}。截至当时已知的当前成员参考；非经核验历史成分，不用于训练。重叠板块不能相加当市场总数。`;
      $('board-method-members').innerHTML=(value.rows||[]).map((row,index)=>{const view=row.method_view||{},reasons=view.reason_codes||[];return `<tr><td>${esc(row.symbol)}<br><small>${esc(row.name||'')}</small></td><td>${esc(KHQuantMethodPresentation.label(view))}<br><small>${esc(reasons.map(KHQuantMethodPresentation.reason).join('；'))}</small></td><td>${view.score_valid?num(view.score_value):'—'}</td><td>${view.forecast_valid?pct(view.forecast_summary?.terminal_return_median):'—'}</td><td><button type="button" data-method-draft="${index}">带入分析</button>${view.prediction_ref?` <button type="button" data-method-prediction="${index}">预测图</button>`:''}</td></tr>`;}).join('')||'<tr><td colspan="5">没有符合条件的成员。</td></tr>';
      const p=value.pagination||{},pages=Math.max(1,Math.ceil((p.total||0)/(p.page_size||40)));$('board-method-member-page').textContent=`${state.page} / ${pages} · ${num(p.total)} 条`;$('board-method-member-prev').disabled=state.page<=1;$('board-method-member-next').disabled=state.page>=pages;
      $('board-method-members').querySelectorAll('[data-method-draft]').forEach(button=>button.addEventListener('click',()=>draft(value.rows[Number(button.dataset.methodDraft)])));
      $('board-method-members').querySelectorAll('[data-method-prediction]').forEach(button=>button.addEventListener('click',()=>prediction(value.rows[Number(button.dataset.methodPrediction)])));
    } catch(error) {if(sequence===state.memberSequence)$('board-method-member-state').textContent=error.message||String(error);}
  }
  function draft(row) {if(!row||!state.result)return;KHQuantMethods.draftAnalysis(row.symbol,state.selection.as_of,{...state.result.method_context,analysis_method:state.selection.method},state.selection.run_id);}
  function prediction(row) {
    const view=row?.method_view;if(!view?.prediction_ref)return;$('board-prediction').hidden=false;$('board-prediction-title').textContent=`${row.symbol} · ${KHQuantMethods.label(state.selection.method)} · 原预测图`;
    $('board-prediction-summary').textContent=`保存扫描 ${state.selection.run_id} · 截止 ${state.selection.as_of} · ${caption(state.result.method_context)}。${view.forecast_valid?'':'预测质量未通过，不产生有效方法 B/S。'}采样比例非校准概率；显示范围不改变输入。`;
    chart??=KHQuantPredictionChart.create({prefix:'board',isVisible:()=>state.visible&&!$('board-prediction').hidden&&!$('view-scan').hidden});
    chart.render({symbol:row.symbol,as_of:state.selection.as_of,result_ref:{cache_key:view.prediction_ref},forecast:[]},path=>api('/api/light'+path));
  }
  function boot() {
    if(booted)return;booted=true;
    const options=Object.entries(KHQuantMethods.definitions).map(([key,value])=>`<option value="${key}">${esc(value.label)}</option>`).join('');
    $('board-method-panel').innerHTML=`<h2>所选方法 · 板块成员分布</h2><div class="method-board-controls"><label>个股方法<select id="board-method-mode">${options}</select></label><label>截止日期<input id="board-method-date" type="date"></label><button id="board-method-refresh" type="button">刷新保存结果</button></div><label>已保存扫描<select id="board-method-run"><option value="">请选择匹配的已保存扫描</option></select></label><button id="board-method-more" type="button" hidden>更多已保存结果</button><p id="board-method-context" class="subtle"></p><p id="board-method-state" aria-live="polite"></p><div id="board-method-summary" class="method-counts"></div><p class="subtle">预测统计为成员横截面，使用各自有效样本数；不是板块联合预测、胜率或收益区间。不可用、失败、待处理不计为 WAIT。</p><label>板块类型<select id="board-method-kind"><option value="all">全部</option><option value="industry">行业</option><option value="concept">概念</option></select></label><div class="table-scroll"><table><thead><tr><th>板块</th><th>截止日方向</th><th>方法覆盖</th><th>预测横截面</th><th>评分均值</th></tr></thead><tbody id="board-method-rows"></tbody></table></div><section id="board-method-detail" hidden><h3 id="board-method-detail-title">成员</h3><p id="board-method-member-state" aria-live="polite"></p><div class="method-board-controls"><label>方向与状态<select id="board-method-member-filter"><option value="all">全部</option><option value="buy">BUY 买方</option><option value="sell">SELL 卖方</option><option value="wait">WAIT 有效观望</option><option value="unavailable">不可用</option><option value="failed">失败</option><option value="deferred">待处理</option><option value="not_in_scan">未纳入扫描</option></select></label><label>筛选成员<input id="board-method-member-search" type="search" placeholder="代码或名称"></label></div><div class="table-scroll"><table><thead><tr><th>股票</th><th>方法方向 / 状态</th><th>评分</th><th>预测中位涨跌</th><th>查看</th></tr></thead><tbody id="board-method-members"></tbody></table></div><div class="pager"><button id="board-method-member-prev" type="button">上一页</button><span id="board-method-member-page"></span><button id="board-method-member-next" type="button">下一页</button></div></section><section id="board-prediction" hidden><h3 id="board-prediction-title"></h3><p id="board-prediction-summary" class="subtle"></p><p id="board-prediction-history-state" aria-live="polite"></p><div class="light-result-controls"><label>历史显示范围<select id="board-history-window"><option value="30">最近30根</option><option value="60" selected>最近60根</option><option value="all">完整输入窗口</option></select></label><label>显示起点<input id="board-history-start" type="date"></label><button id="board-history-reset" type="button">恢复默认范围</button></div><div class="light-prediction-host"><div id="board-prediction-chart" class="light-chart"></div><div id="board-forecast-boundary" class="light-forecast-boundary" hidden><span></span></div></div><p id="board-chart-readout" class="subtle"></p></section>`;
    ['board-method-mode','board-method-date'].forEach(id=>$(id).addEventListener('change',()=>{KHQuantMethods.chooseScanMethod($('board-method-mode').value,$('board-method-date').value);load(KHQuantMethods.scanSelection()).catch(fail);}));
    $('board-method-more').addEventListener('click',()=>moreRuns().catch(fail));
    $('board-method-refresh').addEventListener('click',()=>load(KHQuantMethods.scanSelection()).catch(fail));$('board-method-run').addEventListener('change',()=>chooseRun($('board-method-run').value).catch(fail));$('board-method-kind').addEventListener('change',renderBoards);
    ['board-method-member-filter','board-method-member-search'].forEach(id=>$(id).addEventListener('change',()=>{state.page=1;loadMembers().catch(fail);}));
    $('board-method-member-prev').addEventListener('click',()=>{--state.page;loadMembers().catch(fail);});$('board-method-member-next').addEventListener('click',()=>{++state.page;loadMembers().catch(fail);});
  }
  function onRoute(visible) {boot();state.visible=visible;chart?.onRoute();if(!visible){++state.sequence;++state.memberSequence;if(!state.result)state.scope=null;if(!state.members){state.board=null;$('board-method-detail').hidden=true;}return;}try{const selection=KHQuantMethods.scanSelection();if(selectionKey(selection)!==state.scope)load(selection).catch(fail);}catch(error){fail(error);}}
  globalThis.KHQuantBoardMethods={onRoute,refresh:()=>load(KHQuantMethods.scanSelection())};
})();
