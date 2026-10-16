(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  async function api(path, options={}) {
    const headers=new Headers(options.headers || {}), token=sessionStorage.getItem('khquant_api_token');
    if(token)headers.set('Authorization','Bearer '+token);
    const response = await fetch('/api/boards'+path, {...options,headers});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '请求失败');
    return result;
  }
  function text(id, value) { $(id).textContent = value ?? ''; }
  function showError(error) { text('error',error.message); }
  function table(target, headers, rows) {
    target.replaceChildren(); const wrapper=document.createElement('div');wrapper.className='scroll';
    const element=document.createElement('table'), head=document.createElement('thead'), body=document.createElement('tbody'), top=document.createElement('tr');
    headers.forEach(value=>{const cell=document.createElement('th');cell.textContent=value;top.append(cell);});head.append(top);
    rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(value=>{const cell=document.createElement('td');cell.textContent=value ?? '不可用';tr.append(cell);});body.append(tr);});
    element.append(head,body);wrapper.append(element);target.append(wrapper);
  }
  const baseNames={return_1:'1日收益',return_5:'5日收益',return_10:'10日收益',return_20:'20日收益',advance_ratio:'上涨成员比例',above_ma20_ratio:'站上MA20比例',breakout20_ratio:'创新高比例',amount:'总成交额（元）',market_amount_share:'市场成交额占比',relative_amount20:'20日成交额强度',volatility20:'20日年化波动',drawdown20:'20日最大回撤'};
  const names={};
  for(const [key,value] of Object.entries(baseNames)){names['board_'+key]=value;names['industry_'+key]='同行'+value;}
  Object.assign(names,{relative_industry_1:'相对同行1日收益',relative_industry_5:'相对同行5日收益',relative_industry_20:'相对同行20日收益',industry_peer_rank20:'同行20日收益分位',concept_return_1_mean:'所属概念1日收益均值',concept_return_5_mean:'所属概念5日收益均值',concept_return_20_mean:'所属概念20日收益均值',concept_advance_ratio_mean:'所属概念上涨比例均值',concept_positive_ratio20:'所属概念20日上涨占比',concept_known_count:'已知概念数量'});
  function format(key,value) {
    if(typeof value!=='number')return null;
    if(key==='concept_known_count')return String(value);
    if(key.endsWith('_amount'))return value.toLocaleString('zh-CN',{maximumFractionDigits:0});
    if(/return|ratio|volatility|drawdown|relative_industry|peer_rank/.test(key)&&!key.endsWith('relative_amount20'))return (value*100).toFixed(2)+'%';
    return value.toFixed(4);
  }
  function renderFeatures(target, payload) {
    if (!payload) {target.textContent='尚无已计算特征。';return;}
    const values=payload.features || payload.values || payload;
    const rows=Object.keys(names).filter(key=>key in values).map(key=>[names[key],format(key,values[key])]);
    if (!rows.length) rows.push(['状态',payload.reason || payload.status || '暂无合格数值']);
    table(target,['指标','值'],rows);
    const note=document.createElement('p');note.className='muted';
    note.textContent=`已存特征 · 行情 ${payload.price_date || payload.date || '未知'} · 成员 ${payload.membership_date || '未知'} · 当前成员参考。${(payload.reason_messages || []).join('')}`;
    target.append(note);
  }
  async function status() {
    const s=await api('/status');
    const count=s.board_count ?? s.boards ?? 0, stocks=s.stock_count ?? s.stocks ?? 0;
    const run=s.latest_run || {}, summary=run.summary || {}, kinds=s.counts_by_kind || {}, completed=new Set(run.completed_board_ids || []);
    const failures=new Set((run.failures || []).filter(item=>item.board_id&&!completed.has(item.board_id)).map(item=>item.board_id)).size;
    const remaining=summary.missing_board_ids?.length ?? Math.max(0,(run.board_ids || []).length-completed.size);
    text('status',`板块 ${count} 个（行业 ${kinds.industry || 0}，概念 ${kinds.concept || 0}） · 基本资料 ${stocks} 只 · 覆盖 ${s.covered_stock_count || 0} 只 · 最新更新 ${run.status || '尚未更新'} · 失败 ${failures} 个 · 未完成 ${remaining} 个`);
    text('source',`同花顺分类口径 · 扶摇 API 来源 · 更新 ${run.finished_at || run.observed_at || '尚未完成'}。只补当前板块特征，不补历史成员变化。`);
  }
  async function catalog() {
    const data=await api('/catalog?kind='+encodeURIComponent($('kind').value));
    text('catalog-count',`${data.boards.length} 个已取得完整成员的板块`);
    const box=document.createElement('div');box.className='chips';
    data.boards.forEach(board=>{const button=document.createElement('button');button.type='button';button.textContent=board.name || board.board_id;button.addEventListener('click',()=>detail(board).catch(showError));box.append(button);});
    $('catalog').replaceChildren(box);
  }
  async function detail(board) {
    const result=await api('/features/'+encodeURIComponent(board.board_id));
    const p=result.features;text('detail-title',board.name || board.board_id);
    text('feature-note',p?`当前成员参考 · 行情日期 ${p.date || p.price_date || '未提供'} · 成员资料 ${p.observed_at || p.membership_date || board.observed_at || '未提供'}`:'成员资料已入库，特征尚未计算或数据不足。');
    const excluded=p?.excluded_source_members || board.excluded_source_members || [];
    if(excluded.length)text('feature-note',$('feature-note').textContent+` · 来源全体 ${p?.source_member_count || board.source_member_count} 个，特征采用 ${p?.expected_members || board.expected_count} 个 A 股；非 A 股保留在来源证明中：${excluded.join('、')}。`);
    renderFeatures($('features'),p);
    if(result.needs_rebuild)text('feature-note','成员快照已更新，请计算当前板块特征。');
  }
  async function lookup(event) {
    event?.preventDefault();text('error','');
    const data=await api('/stocks/'+encodeURIComponent($('symbol').value.toUpperCase())+'?current=true');
    const industries=data.boards.filter(b=>b.kind==='industry').length;
    text('membership-note',`当前成员口径 · ${data.availability || '已读取'} · 资料 ${data.observed_at || data.available_at || '未知'}。${industries>1?'该股票属于多个行业指数，层级尚未核验，个股相对行业指标等待明确口径。':''}`);
    table($('memberships'),['类型','名称','板块代码'],data.boards.map(b=>[b.kind==='industry'?'行业':'概念',b.name,b.board_id]));
    renderFeatures($('stock-features'),data.features);
    if(data.needs_rebuild)$('stock-features').textContent='成员快照已更新，请计算当前板块特征。';
  }
  async function startJob(path,body) {
    $('refresh-data').disabled=true;$('build-features').disabled=true;text('error','');
    const unlock=()=>{$('refresh-data').disabled=false;$('build-features').disabled=false;};
    try {
      const j=await api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      async function poll() {
    if(document.hidden){setTimeout(()=>poll().catch(()=>{}),15000);return;}
        const state=await api('/jobs/'+j.job_id), labels={running:path==='/refresh'?'正在更新板块资料…':'正在计算板块特征…',succeeded:'任务完成。',completed:'任务完成。',partial:'部分板块更新成功，失败记录已保存，可继续补采。',interrupted:'本次采集已暂停，已完成快照已保存，可续采。',failed:'任务失败：'+(state.error || '请查看知识库更新记录')};
        text('job',labels[state.status] || state.status);
        if(state.status==='running')setTimeout(()=>poll().catch(e=>{showError(e);unlock();}),3000);
        else{unlock();await status();await catalog();await lookup();}
      }
      await poll();
    } catch(e){unlock();throw e;}
  }
  $('refresh-status').addEventListener('click',()=>status().catch(showError));
  $('refresh-data').addEventListener('click',()=>startJob('/refresh',{kinds:['industry','concept']}).catch(showError));
  $('build-features').addEventListener('click',()=>startJob('/features/build',{mode:'current_reference'}).catch(showError));
  $('lookup').addEventListener('submit',e=>{e.preventDefault();lookup().catch(showError);});
  $('kind').addEventListener('change',()=>catalog().catch(showError));
  Promise.all([status(),catalog()]).then(()=>lookup()).catch(showError);
})();
