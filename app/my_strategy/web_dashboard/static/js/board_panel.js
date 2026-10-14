/* Current board snapshots: browsing only reads published rows. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const active = new Set(['queued','running','cancelling']);
  const names = {queued:'等待',running:'运行中',cancelling:'取消中',cancelled:'已取消',succeeded:'完成',completed:'完成',partial:'部分完成',failed:'失败',interrupted:'中断'};
  let loaded = false, sequence = 0, job = null, timer = null, detailSequence = 0;
  const numeric = (v, digits=2) => typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString('zh-CN',{maximumFractionDigits:digits}) : '—';
  const percent = v => typeof v === 'number' && Number.isFinite(v) ? `${numeric(v*100)}%` : '—';
  const amount = v => typeof v === 'number' && Number.isFinite(v) ? `${numeric(v/1e8)} 亿` : '—';
  function node(tag, text, className) { const n=document.createElement(tag); if(text != null)n.textContent=text; if(className)n.className=className; return n; }
  async function request(path, options={}) {
    const headers={...options.headers}; if(options.body)headers['Content-Type']='application/json';
    const token=sessionStorage.getItem('khquant_api_token'); if(token)headers.Authorization=`Bearer ${token}`;
    const response=await fetch(path,{...options,headers,credentials:'same-origin'});
    let result; try { result=await response.json(); } catch (_) { throw new Error(`板块服务响应无法解析（${response.status}）`); }
    if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:`板块请求失败（${response.status}）`);
    return result;
  }
  function error(exc) { $('board-status').textContent=exc.message || String(exc); $('board-status').classList.add('task-error'); }
  function selectTab(board) {
    $('scan-stock-panel').hidden=board; $('scan-board-panel').hidden=!board; $('start-scan').hidden=board;
    $('scan-stock-tab').setAttribute('aria-selected',String(!board)); $('scan-board-tab').setAttribute('aria-selected',String(board));
    sessionStorage.setItem('khquant_scan_tab',board?'board':'stock');
    if(board && !loaded)refresh().catch(error);
  }
  function renderStatus(status) {
    $('board-status').classList.remove('task-error');
    const data=status.fluent || status, market=data.market, full=data.full;
    const pieces=[market ? `行情 ${market.price_date} · 成员 ${market.membership_date || '—'} · ${market.board_count} 个板块` : '尚无已发布特征'];
    if(market)pieces.push(market.full_ready && full?.generation_id===market.generation_id ? '结构与量价摘要已发布' : '行情特征已发布，结构摘要待同步');
    const reasonNames={raw_date_or_revision_changed:'行情日期或内容变化',membership_source_changed:'成员来源变化',schema_policy_or_generating_code_changed:'特征计算版本变化',no_published_market:'尚未计算'};
    if(data.native_stale && !data.stale)pieces.push('原生结构来源需要重新核验，点击同步');
    if(data.stale)pieces.push(`需要更新：${(data.stale_reasons || []).map(x=>reasonNames[x] || '来源待核验').join('、') || '来源已变化'}`);
    $('board-status').textContent=pieces.join(' · ');
  }
  function renderRows(result) {
    const rows=result.rankings || result.rows || [];
    $('board-count').textContent=`${rows.length} 个合格板块 · Top30 · 行情 ${result.price_date || rows[0]?.price_date || '—'}`;
    const host=$('board-rows'); host.replaceChildren();
    if(!rows.length) {const tr=node('tr'), td=node('td','暂无合格结果。计算后，成员和行情覆盖不足的板块保持缺失。','empty-cell');td.colSpan=9;tr.append(td);host.append(tr);return;}
    rows.forEach(row => {
      const tr=node('tr'), name=node('td'); name.append(node('strong',row.name || row.board_id),node('small',row.board_id));
      tr.append(node('td',row.rank),name,node('td',numeric(row.activity_score)),node('td',percent(row.board_return_1)),node('td',amount(row.board_amount)),node('td',percent(row.board_market_amount_share)),node('td',numeric(row.board_relative_amount20)),node('td',`${row.effective_members ?? '—'}/${row.expected_members ?? '—'} · ${percent(row.coverage)}`));
      const cell=node('td'), button=node('button','查看');button.type='button';button.addEventListener('click',()=>detail(row.board_id,row.generation_id).catch(error));cell.append(button);tr.append(cell);host.append(tr);
    });
  }
  async function refresh() {
    const current=++sequence;
    const [status,ranking]=await Promise.all([request('/api/boards/status'),request(`/api/boards/rankings?kind=${encodeURIComponent($('board-kind').value)}&ranking_type=${encodeURIComponent($('board-ranking').value)}`)]);
    if(current!==sequence)return;loaded=true;renderStatus(status);renderRows(ranking);
    if(!job) {const jobs=await request('/api/boards/jobs');const pending=(jobs.jobs || []).find(x=>active.has(x.status));if(pending){job=pending;poll().catch(error);}}
  }
  function metrics(host, pairs) {const dl=node('dl','', 'board-metrics');pairs.forEach(([label,value])=>{dl.append(node('dt',label),node('dd',value));});host.append(dl);}
  async function detail(id,generation) {
    const current=++detailSequence;
    const result=await request(`/api/boards/detail/${encodeURIComponent(id)}?generation_id=${encodeURIComponent(generation || '')}`);
    if(current!==detailSequence)return;
    const row=result.features || {}, host=$('board-detail-content');host.replaceChildren();$('board-detail').hidden=false;$('board-detail-title').textContent=row.name || id;
    host.append(node('p',`行情 ${row.price_date || '—'} · 当前成员 ${row.membership_date || '—'}`,'subtle'));
    metrics(host,[['涨跌 1 / 5 / 20 日',`${percent(row.board_return_1)} / ${percent(row.board_return_5)} / ${percent(row.board_return_20)}`],['上涨成员',percent(row.board_advance_ratio)],['站上 MA20',percent(row.board_above_ma20_ratio)],['20 日突破',percent(row.board_breakout20_ratio)],['成交额',amount(row.board_amount)],['相对成交额 20 日',numeric(row.board_relative_amount20)],['20 日波动',percent(row.board_volatility20)]]);
    const structures=row.native_structure || [];
    host.append(node('h4','结构与威科夫量价'));
    if(!structures.length)host.append(node('p','结构摘要尚未发布，点击“同步结构与特征”。'));
    structures.forEach(s=>{
      if(s.structure_basis==='member_native') {
        [['structure','CZSC 成员结构'],['wyckoff','威科夫成员量价']].forEach(([source,label])=>{
          host.append(node('h5',label));
          metrics(host,[['有效成员覆盖',`${s[source+'_eligible_members'] ?? '—'}/${s.expected_members ?? '—'} · ${percent(s[source+'_coverage'])}`],['买方 / 卖方占比',`${percent(s[source+'_buy_ratio'])} / ${percent(s[source+'_sell_ratio'])}`],['买方 / 卖方均分',`${numeric(s[source+'_buy_score_mean'])} / ${numeric(s[source+'_sell_score_mean'])}`],['活跃买方 / 卖方事件',`${s[source+'_active_buy_event_ids']?.length ?? '—'} / ${s[source+'_active_sell_event_ids']?.length ?? '—'}`]]);
          if(!s[source+'_available'])host.append(node('p','有效成员不足 5 个或覆盖不足 80%，评分和占比保持缺失。','subtle'));
        });
      } else {
        host.append(node('h5',s.structure_basis));
        metrics(host,[['有效成员覆盖',`${s.eligible_members ?? '—'}/${s.expected_members ?? '—'} · ${percent(s.coverage)}`],['买方 / 卖方占比',`${percent(s.buy_ratio)} / ${percent(s.sell_ratio)}`],['平均评分',numeric(s.mean_score)]]);
      }
    });
    if(row.reason_codes?.length)host.append(node('p',`可用性：${row.reason_codes.join('、')}`,'subtle'));
    host.append(node('h4',result.membership_matches_generation?`该版本成员 · ${result.members?.length || 0}`:'成员证明暂不可用'));
    if(!result.membership_matches_generation)host.append(node('p','该特征版本的成员快照缺失，请更新并重新计算。','task-error'));
    const list=node('div','', 'board-members');(result.members || []).forEach(member=>{const symbol=typeof member==='string'?member:member.symbol;const button=node('button',typeof member==='string'?member:`${symbol} ${member.name || ''}`);button.type='button';button.addEventListener('click',()=>window.dispatchEvent(new CustomEvent('khquant:board-stock',{detail:{symbol,price_date:row.price_date}})));list.append(button);});host.append(list);
  }
  function renderJob(value) {
    $('board-job').hidden=false;const host=$('board-job');host.replaceChildren();const d=value.progress_detail || value.progress || {};
    host.append(node('p',`${names[value.status] || value.status} · ${d.stage || ''} ${d.current ?? d.completed ?? ''}${d.total ? ` / ${d.total}` : ''}${value.error ? ` · ${value.error}` : ''}`));
    if(active.has(value.status)) {const cancel=node('button','取消');cancel.type='button';cancel.disabled=value.status==='cancelling';cancel.addEventListener('click',async()=>{try{job=await request(`/api/boards/jobs/${encodeURIComponent(value.job_id)}/cancel`,{method:'POST',body:'{}'});renderJob(job);}catch(exc){error(exc);}});host.append(cancel);}
    else if(['partial','failed','interrupted','cancelled'].includes(value.status)) {const retry=node('button','重试未完成阶段');retry.type='button';retry.addEventListener('click',()=>{const path=value.kind==='board_knowledge'?'/refresh':value.kind==='board_features'?'/features/build':'/sync';const body=value.kind==='board_knowledge'?{...value.spec,resume:value.result?.knowledge_run_id || value.knowledge_run_id || value.progress_detail?.knowledge_run_id || value.spec?.resume || null}:value.kind==='board_features'?value.spec:{resume_sync_id:value.result?.sync_id || value.progress_detail?.sync_id || value.spec?.resume_sync_id || null};start(path,body).catch(error);});host.append(retry);}
    ['board-calculate','board-sync','board-refresh-members'].forEach(id=>$(id).disabled=active.has(value.status));
  }
  async function poll() {
    clearTimeout(timer);if(!job)return;const value=await request(`/api/boards/jobs/${encodeURIComponent(job.job_id)}`);job=value;renderJob(value);
    if(active.has(value.status))timer=setTimeout(()=>poll().catch(error),2000);else {job=null;await refresh();}
  }
  async function start(path,body={}) {job=await request(`/api/boards${path}`,{method:'POST',body:JSON.stringify(body)});renderJob(job);await poll();}
  const run=(path,body)=>start(path,body).catch(error);
  if(!$('scan-board-panel'))return;
  $('scan-stock-tab').addEventListener('click',()=>selectTab(false));$('scan-board-tab').addEventListener('click',()=>{const alreadyLoaded=loaded;selectTab(true);if(alreadyLoaded)refresh().catch(error);});
  ['board-kind','board-ranking'].forEach(id=>$(id).addEventListener('change',()=>refresh().catch(error)));
  $('board-refresh-view').addEventListener('click',()=>refresh().catch(error));$('board-calculate').addEventListener('click',()=>run('/features/build',{with_native:true}));$('board-sync').addEventListener('click',()=>run('/sync'));
  $('board-refresh-members').addEventListener('click',()=>run('/refresh'));$('board-detail-close').addEventListener('click',()=>{++detailSequence;$('board-detail').hidden=true;});
  const panel={onRoute(){const wasLoaded=loaded;const desired=new URLSearchParams(location.search).get('board')==='1' || sessionStorage.getItem('khquant_scan_tab')==='board';selectTab(desired);if(desired && wasLoaded)refresh().catch(error);},refresh,async retryTask(id){const value=await request(`/api/boards/jobs/${encodeURIComponent(id)}`);selectTab(true);location.hash='scan';renderJob(value);},showResult(result){selectTab(true);refresh().catch(error);request('/api/boards/jobs').then(async data=>{const saved=(data.jobs || []).find(x=>x.run_id===result.run_id);if(saved)renderJob(await request(`/api/boards/jobs/${encodeURIComponent(saved.job_id)}`));}).catch(error);}};
  globalThis.KHQuantBoardPanel=panel;panel.onRoute();
})();
