/* Current board snapshots: browsing only reads published rows. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const active = new Set(['queued','running','cancelling']);
  const names = {queued:'等待',running:'运行中',cancelling:'取消中',cancelled:'已取消',succeeded:'完成',completed:'完成',partial:'部分完成',failed:'失败',interrupted:'中断'};
  let loaded = false, sequence = 0, job = null, timer = null, detailSequence = 0;
  let fillPlan=null,fillJob=null,fillTimer=null,fillResult=null;
  let capitalData = null, selectedCapital = null, colorMetric = 'change', displayedGeneration = null;
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
  const valid = v => typeof v === 'number' && Number.isFinite(v);
  const signedAmount = v => valid(v) ? `${v > 0 ? '+' : ''}${numeric(v/1e8)} 亿` : '—';
  const signedPercent = v => valid(v) ? `${v > 0 ? '+' : ''}${numeric(v*100)}%` : '—';
  function control(label, id, options) {
    const host=node('label',label), select=node('select');select.id=id;
    options.forEach(([value,text])=>{const option=node('option',text);option.value=value;select.append(option);});host.append(select);return host;
  }
  function setupCapital() {
    if($('board-capital-dashboard'))return;
    const host=node('section',null,'board-capital-dashboard');host.id='board-capital-dashboard';host.setAttribute('aria-label','板块成交资金变化图');
    const header=node('div',null,'capital-heading'), copy=node('div');copy.append(node('h2','板块成交资金变化'),node('p','看成交活跃度如何变化，点击板块查看近期走势。','subtle'));header.append(copy);
    const controls=node('div',null,'capital-controls');controls.append(control('展示排序','capital-sort',[['amount','成交额 Top30'],['change','成交额变化 Top30'],['relative','相对成交额 Top30']]),control('方块颜色','capital-color',[['change','成交额较前日变化'],['return','当日涨跌幅']]));header.append(controls);host.append(header);
    const stats=node('div',null,'capital-stats');stats.id='capital-stats';host.append(stats);
    const grid=node('div',null,'capital-grid');
    const mapCard=node('div',null,'capital-card'), mapHead=node('div',null,'capital-card-heading');mapHead.append(node('h3','成交额热力图'));const legend=node('p','面积＝当日成交额 · 红色增加 / 绿色减少','subtle');legend.id='capital-map-legend';mapHead.append(legend);mapCard.append(mapHead);
    const map=node('div',null,'capital-map');map.id='capital-map';map.setAttribute('aria-label','板块成交额热力图，可用键盘选择板块');mapCard.append(map);grid.append(mapCard);
    const changeCard=node('div',null,'capital-card');changeCard.append(node('h3','成交额增减 · 各 Top5'),node('p','按两日都有核验行情的同一批成员比较，单位：亿元。','subtle'));const change=node('div',null,'capital-changes');change.id='capital-changes';changeCard.append(change);grid.append(changeCard);host.append(grid);
    const trend=node('div',null,'capital-card capital-trend-card'), trendHead=node('div',null,'capital-heading');
    const trendCopy=node('div');const title=node('h3','选择一个板块');title.id='capital-trend-title';const meta=node('p','','subtle');meta.id='capital-trend-meta';trendCopy.append(title,meta);trendHead.append(trendCopy);
    const trendControls=node('div',null,'capital-controls');trendControls.append(control('走势指标','capital-trend-metric',[['amount','成交额（亿元）'],['share','占本地市场成交额（%）']]),control('区间','capital-window',[['20','近20个交易日'],['10','近10个交易日'],['5','近5个交易日']]));const detailButton=node('button','查看成员');detailButton.id='capital-detail-button';detailButton.type='button';trendControls.append(detailButton);trendHead.append(trendControls);trend.append(trendHead);
    const figure=node('div',null,'capital-trend');figure.id='capital-trend';trend.append(figure);host.append(trend);
    host.append(node('p','成交额是交易活跃度代理，不表示净流入。历史曲线按该发布版本的当前成员重建，仅供展示；板块可重叠，不能相加当作全市场资金。','capital-method'));
    const tooltip=node('div',null,'capital-tooltip');tooltip.id='capital-tooltip';tooltip.hidden=true;tooltip.setAttribute('role','status');host.append(tooltip);
    $('scan-board-panel').querySelector('.board-layout').before(host);
    $('capital-sort').addEventListener('change',()=>refresh().catch(error));
    $('capital-color').addEventListener('change',()=>{colorMetric=$('capital-color').value;renderCapital(capitalData);});
    ['capital-trend-metric','capital-window'].forEach(id=>$(id).addEventListener('change',renderCapitalTrend));
    detailButton.addEventListener('click',()=>{if(selectedCapital)detail(selectedCapital.board_id,selectedCapital.generation_id).catch(error);});
  }
  function tip(element, lines) {
    const show=()=>{const host=$('capital-tooltip');host.replaceChildren();lines().filter(Boolean).forEach((text,i)=>host.append(node(i?'div':'strong',text)));host.hidden=false;
      const box=element.getBoundingClientRect(), bounds=$('board-capital-dashboard').getBoundingClientRect();
      host.style.left=`${Math.max(8,Math.min(bounds.width-280,box.left-bounds.left+8))}px`;host.style.top=`${Math.max(0,box.bottom-bounds.top+5)}px`;};
    const hide=()=>{$('capital-tooltip').hidden=true;};element.addEventListener('mouseenter',show);element.addEventListener('focus',show);element.addEventListener('mouseleave',hide);element.addEventListener('blur',hide);element.addEventListener('click',hide);
  }
  function tiles(rows) {
    const width=1000,height=420,total=rows.reduce((n,row)=>n+row.board_amount,0);if(!(total>0))return [];
    const queue=rows.filter(row=>row.board_amount>0).map(row=>({row,area:row.board_amount/total*width*height}));
    const output=[];let x=0,y=0,w=width,h=height,group=[];
    function worst(items,side) {if(!items.length)return Infinity;const values=items.map(item=>item.area),sum=values.reduce((a,b)=>a+b,0);return Math.max(side*side*Math.max(...values)/(sum*sum),sum*sum/(side*side*Math.min(...values)));}
    function place(items) {const sum=items.reduce((n,item)=>n+item.area,0);if(w>=h){const rw=sum/h;let top=y;items.forEach(item=>{const rh=item.area/rw;output.push({...item,x,y:top,w:rw,h:rh});top+=rh;});x+=rw;w-=rw;}else{const rh=sum/w;let left=x;items.forEach(item=>{const rw=item.area/rh;output.push({...item,x:left,y,w:rw,h:rh});left+=rw;});y+=rh;h-=rh;}}
    while(queue.length){const side=Math.min(w,h),next=queue[0];if(!group.length || worst([...group,next],side)<=worst(group,side)){group.push(queue.shift());}else{place(group);group=[];}}
    if(group.length)place(group);return output;
  }
  function rowTip(row) {const v=row.capital_visualization || {};return [row.name,`成交额 ${amount(row.board_amount)} · 涨跌 ${signedPercent(row.board_return_1)}`,`成交额变化 ${signedAmount(v.amount_change_1)} / ${signedPercent(v.amount_change_ratio_1)}`,`同成员比较 ${v.comparable_members ?? '—'}/${row.expected_members ?? '—'} · 覆盖 ${percent(v.comparable_coverage)}`,`行情 ${row.price_date} · 当前成员 ${row.membership_date}`];}
  function clearFillPlan(){fillPlan=null;if($('board-fill-execute'))$('board-fill-execute').hidden=true;if(!fillJob&&$('board-fill-plan')){if(fillResult)renderFillResult(fillResult);else $('board-fill-plan').replaceChildren();}}
  function chooseCapital(row) {if(selectedCapital?.board_id!==row.board_id)clearFillPlan();selectedCapital=row;document.querySelectorAll('[data-capital-board]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.capitalBoard===row.board_id)));renderCapitalTrend();}
  function renderCapital(data) {
    if(!data)return;capitalData=data;
    const rows=data.rows || [], stats=$('capital-stats');stats.replaceChildren();
    [['行情截止',data.price_date || '—'],['本地市场成交额',amount(data.market_amount)],['当前类型可用板块',`${data.available_board_count ?? 0} / ${data.board_count ?? 0}`],['图中板块',`${rows.length} 个`]].forEach(([label,value])=>{const item=node('div');item.append(node('span',label),node('strong',value));stats.append(item);});
    const map=$('capital-map');map.replaceChildren();
    $('capital-map-legend').textContent=`面积＝当日成交额 · ${colorMetric==='change'?'颜色＝成交额较前日变化':'颜色＝当日涨跌幅'} · 红增绿减`;
    if(!rows.length)map.append(node('p',data.needs_rebuild?'历史变化尚未准备，点击“计算当前板块”。':'暂无达到覆盖要求的板块。','capital-empty'));
    const metric=row=>colorMetric==='return'?row.board_return_1:row.capital_visualization?.amount_change_ratio_1;
    const scale=Math.max(colorMetric==='return'?.02:.1,...rows.map(row=>valid(metric(row))?Math.abs(metric(row)):0));
    tiles([...rows].sort((a,b)=>b.board_amount-a.board_amount)).forEach(item=>{
      const row=item.row,value=metric(row),button=node('button',null,'capital-tile');button.type='button';button.dataset.capitalBoard=row.board_id;
      button.style.cssText=`left:${item.x/10}%;top:${item.y/4.2}%;width:${item.w/10}%;height:${item.h/4.2}%;`;
      const alpha=valid(value) ? .18+.64*Math.min(1,Math.abs(value)/scale) : .08;
      button.style.background=valid(value)?`rgba(${value>=0?'189,55,69':'31,126,104'},${alpha})`:'var(--border)';if(alpha>.45)button.style.color='#fff';
      button.append(node('strong',row.name),node('span',amount(row.board_amount)),node('small',signedPercent(value)));if(item.w*(map.clientWidth || 1000)/1000<95 || item.h<66)button.classList.add('capital-tile-small');
      button.setAttribute('aria-label',`${row.name}，成交额${amount(row.board_amount)}，${colorMetric==='change'?'成交额变化':'涨跌幅'}${signedPercent(value)}`);button.setAttribute('aria-pressed',String(selectedCapital?.board_id===row.board_id));button.addEventListener('click',()=>chooseCapital(row));tip(button,()=>rowTip(row));map.append(button);
    });
    const changes=$('capital-changes');changes.replaceChildren();const ups=(data.increases || []).slice(0,5),downs=(data.decreases || []).slice(0,5),all=[...ups,...downs],max=Math.max(1,...all.map(row=>Math.abs(row.capital_visualization.amount_change_1)));
    [['成交额增加',ups],['成交额减少',downs]].forEach(([label,items])=>{changes.append(node('h4',label));if(!items.length)changes.append(node('p','暂无同成员可比数据。','subtle'));items.forEach(row=>{const value=row.capital_visualization.amount_change_1,button=node('button',null,'capital-change-row');button.type='button';button.append(node('span',row.name,'capital-change-name'));const rail=node('span',null,'capital-change-rail'),bar=node('span',null,value>=0?'capital-positive':'capital-negative'),size=Math.abs(value)/max*49;bar.style.width=`${size}%`;bar.style.left=`${value>=0?50:50-size}%`;rail.append(bar);button.append(rail,node('span',signedAmount(value),'capital-change-value'));button.dataset.capitalBoard=row.board_id;button.setAttribute('aria-label',`${row.name}成交额变化${signedAmount(value)}`);button.addEventListener('click',()=>chooseCapital(row));tip(button,()=>rowTip(row));changes.append(button);});});
    selectedCapital=rows.find(row=>row.board_id===selectedCapital?.board_id) || rows[0] || null;if(selectedCapital)chooseCapital(selectedCapital);else renderCapitalTrend();
  }
  function svgNode(tag,attributes={},text) {const element=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attributes).forEach(([name,value])=>element.setAttribute(name,String(value)));if(text!=null)element.textContent=text;return element;}
  function renderCapitalTrend() {
    const host=$('capital-trend');if(!host)return;host.replaceChildren();const row=selectedCapital;
    $('capital-detail-button').disabled=!row;$('capital-trend-title').textContent=row?`${row.name} · 近期成交资金变化`:'选择一个板块';
    const v=row?.capital_visualization,history=(v?.history || []).slice(-Number($('capital-window').value));
    $('capital-trend-meta').textContent=row?`当前成员 ${row.membership_date} · ${history[0]?.date || '—'} 至 ${row.price_date} · 当前成员历史对照`:'点击热力图或增减排行查看';
    if(!history.length){host.append(node('p','该发布版本没有成交额历史，点击“计算当前板块”生成。','capital-empty'));return;}
    const metric=$('capital-trend-metric').value,amountMode=metric==='amount',values=history.map(point=>amountMode?valid(point.amount)?point.amount/1e8:null:valid(point.market_share)?point.market_share*100:null);
    const mean=amountMode && valid(v.prior20_mean_amount)?v.prior20_mean_amount/1e8:null,max=Math.max(1,...values.filter(valid),mean || 0)*1.12;
    const width=Math.max(360,host.clientWidth || 1000),height=245,left=62,right=24,top=24,bottom=38,plotW=width-left-right,plotH=height-top-bottom;
    const svg=svgNode('svg',{viewBox:`0 0 ${width} ${height}`,role:'img','aria-label':`${row.name}近${history.length}个交易日${amountMode?'成交额亿元':'市场成交额占比百分比'}`});host.append(svg);
    const y=value=>top+plotH*(1-value/max),x=i=>left+(i+.5)*plotW/history.length;
    for(let i=0;i<=4;i++){const value=max*i/4,at=y(value);svg.append(svgNode('line',{x1:left,x2:width-right,y1:at,y2:at,class:'capital-gridline'}),svgNode('text',{x:left-9,y:at+4,'text-anchor':'end',class:'capital-axis'},numeric(value,amountMode?0:1)));}
    svg.append(svgNode('text',{x:8,y:14,class:'capital-axis'},amountMode?'亿元':'%'));
    if(mean!=null){const at=y(mean);svg.append(svgNode('line',{x1:left,x2:width-right,y1:at,y2:at,class:'capital-mean'}),svgNode('text',{x:width-right,y:at-6,'text-anchor':'end',class:'capital-axis'},`截止日前20日均额 ${numeric(mean)} 亿`));}
    let segment=[];function finishLine(){if(segment.length)svg.append(svgNode('polyline',{points:segment.join(' '),fill:'none',class:'capital-share-line'}));segment=[];}
    history.forEach((point,i)=>{const value=values[i],at=x(i);if(!valid(value)){finishLine();const missing=svgNode('text',{x:at,y:y(0)-5,'text-anchor':'middle',class:'capital-missing',tabindex:0},'×');tip(missing,()=>[point.date,'行情覆盖不足，保持缺失，不按零成交额绘制']);svg.append(missing);return;}
      let mark;if(amountMode){const barW=plotW/history.length*.56;mark=svgNode('rect',{x:at-barW/2,y:y(value),width:barW,height:Math.max(1,y(0)-y(value)),rx:3,class:point.return_1>=0?'capital-bar-up':'capital-bar-down',tabindex:0,'aria-label':`${point.date}成交额${amount(point.amount)}`});}else{segment.push(`${at},${y(value)}`);mark=svgNode('circle',{cx:at,cy:y(value),r:4,class:'capital-share-dot',tabindex:0,'aria-label':`${point.date}市场占比${percent(point.market_share)}`});}
      tip(mark,()=>[`${row.name} · ${point.date}`,`成交额 ${amount(point.amount)} · 市场占比 ${percent(point.market_share)}`,`成员覆盖 ${point.effective_members}/${point.expected_members} · ${percent(point.coverage)}`]);svg.append(mark);
    });finishLine();
    const step=Math.max(1,Math.ceil(history.length/6));history.forEach((point,i)=>{if(i%step===0 || i===history.length-1)svg.append(svgNode('text',{x:x(i),y:height-12,'text-anchor':'middle',class:'capital-axis'},point.date.slice(5)));});
    host.append(node('p',`最新成交额 ${amount(row.board_amount)} · 同成员较前日 ${signedAmount(v.amount_change_1)}（${signedPercent(v.amount_change_ratio_1)}） · ${v.comparable_members ?? '—'} 个可比成员`,'subtle'));
  }
  async function refresh() {
    const current=++sequence;
    const kind=$('board-kind').value,rankingType=$('board-ranking').value,sortBy=$('capital-sort').value;
    // Pin all reads to the publication observed by status. A newer publication
    // may arrive between requests; its immutable predecessor remains readable.
    const status=await request('/api/boards/status');
    if(current!==sequence)return;
    const market=(status.fluent || status).market,generation=market?.generation_id || null;
    let ranking={rankings:[],generation_id:null,price_date:null},visual={rows:[],increases:[],decreases:[],generation_id:null,
      price_date:null,board_count:0,available_board_count:0,market_amount:null};
    if(generation) {
      const pinned=`&generation_id=${encodeURIComponent(generation)}`;
      [ranking,visual]=await Promise.all([
        request(`/api/boards/rankings?kind=${encodeURIComponent(kind)}&ranking_type=${encodeURIComponent(rankingType)}${pinned}`),
        request(`/api/boards/capital-visualization?kind=${encodeURIComponent(kind)}&sort_by=${encodeURIComponent(sortBy)}${pinned}`)
      ]);
      if(current!==sequence)return;
      const rows=[...(ranking.rankings || []),...(visual.rows || []),...(visual.increases || []),...(visual.decreases || [])];
      if(ranking.generation_id!==generation || visual.generation_id!==generation ||
          ranking.price_date!==market.price_date || visual.price_date!==market.price_date ||
          rows.some(row=>row.generation_id!==generation || row.price_date!==market.price_date))
        throw new Error('板块发布版本核验失败，已保留原图，请刷新重试。');
    }
    if(current!==sequence)return;
    if(displayedGeneration!==generation) {clearFillPlan();++detailSequence;$('board-detail').hidden=true;}
    $('capital-tooltip').hidden=true;
    // Commit status, table and charts together only after every pinned read is ready.
    loaded=true;displayedGeneration=generation;renderStatus(status);renderRows(ranking);renderCapital(visual);
    if(!fillJob){
      const light=await request('/api/light/status');if(current!==sequence)return;
      const fills=(light.jobs||[]).filter(x=>x.kind==='incremental').sort((a,b)=>fillTime(b).localeCompare(fillTime(a)));
      const pending=fills.find(x=>active.has(x.status));
      if(pending){fillJob=pending.job_id;$('board-fill-execute').hidden=true;$('board-fill-cancel').hidden=false;$('board-sync').disabled=true;pollFill().catch(error);}
      else {
        const latest=fills[0];
        if(latest && (!fillResult || fillTime(latest)>=fillTime(fillResult)) && latest.job_id!==fillResult?.job_id){
          const terminal=await request('/api/light/jobs/'+encodeURIComponent(latest.job_id));
          if(current!==sequence)return;
          // A read may discover a newly running job, but never queues or resumes one.
          if(active.has(terminal.status)){fillJob=terminal.job_id;$('board-fill-execute').hidden=true;$('board-fill-cancel').hidden=false;$('board-sync').disabled=true;pollFill().catch(error);}
          else fillResult=terminal;
        }
        if(fillResult&&!fillPlan&&!fillJob)renderFillResult(fillResult);
      }
    }
    if(!job) {
      const jobs=await request('/api/boards/jobs');
      if(current!==sequence || job)return;
      const saved=jobs.jobs || [],pending=saved.find(x=>active.has(x.status));
      if(pending){job=pending;poll().catch(error);}
      else if(saved.length)renderJob(saved[0]);
      else {
        if(fillResult)renderJob(fillResult);else $('board-job').hidden=true;
        ['board-calculate','board-sync','board-refresh-members'].forEach(id=>$(id).disabled=!!fillJob);
      }
    }
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
    if(!structures.length)host.append(node('p','本板块结构摘要待补齐，可先预览缺口再执行。'));
    if(structures.length && row.native_component_complete===false)host.append(node('p','仅显示已核验成员，完整覆盖见下方；全市场结构仍有缺口，未发布全量完成标记。','subtle'));
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
    if(fillJob&&!active.has(value.status)){
      host.append(node('p','当前增量补齐正在运行，逐股进度见下方。'));
      if(value.status==='cancelled')host.append(node('p','此前板块任务已取消并保持终止，本次是单独提交的增量任务。','subtle'));
      ['board-calculate','board-sync','board-refresh-members'].forEach(id=>$(id).disabled=true);return;
    }
    if(fillResult&&!active.has(value.status)&&fillTime(fillResult)>=fillTime(value)){
      const result=fillResult.result || {},counts=result.counts || {};
      host.append(node('p',`最近增量补齐 · ${names[result.status||fillResult.status]||result.status||fillResult.status} · ${result.price_date||'—'} · 复用 ${counts.reused??'未记录'} · 追加 ${counts.appended??'未记录'} · 重建 ${counts.rebuilt??'未记录'} · 失败 ${counts.failed??'未记录'} · 延后 ${counts.deferred??'未记录'}`));
      if(value.job_id!==fillResult.job_id){const history=node('details');history.append(node('summary',`历史板块任务：${names[value.status]||value.status}${value.status==='cancelled'?'（保持终止）':''}`),node('p',`${d.stage||''} ${d.current??d.completed??''}${d.total?` / ${d.total}`:''}${value.error?' · '+value.error:''}`));host.append(history);}
      ['board-calculate','board-sync','board-refresh-members'].forEach(id=>$(id).disabled=!!fillJob);return;
    }
    host.append(node('p',`${names[value.status] || value.status} · ${d.stage || ''} ${d.current ?? d.completed ?? ''}${d.total ? ` / ${d.total}` : ''}${value.error ? ` · ${value.error}` : ''}`));
    if(active.has(value.status)) {const cancel=node('button','取消');cancel.type='button';cancel.disabled=value.status==='cancelling';cancel.addEventListener('click',async()=>{try{job=await request(`/api/boards/jobs/${encodeURIComponent(value.job_id)}/cancel`,{method:'POST',body:'{}'});renderJob(job);}catch(exc){error(exc);}});host.append(cancel);}
    else if(['partial','failed','interrupted'].includes(value.status) && ['board_knowledge','board_features'].includes(value.kind)) {const retry=node('button','重试未完成阶段');retry.type='button';retry.addEventListener('click',()=>{const path=value.kind==='board_knowledge'?'/refresh':value.kind==='board_features'?'/features/build':'/sync';const body=value.kind==='board_knowledge'?{...value.spec,resume:value.result?.knowledge_run_id || value.knowledge_run_id || value.progress_detail?.knowledge_run_id || value.spec?.resume || null}:value.kind==='board_features'?value.spec:{resume_sync_id:value.result?.sync_id || value.progress_detail?.sync_id || value.spec?.resume_sync_id || null};start(path,body).catch(error);});host.append(retry);}
    ['board-calculate','board-sync','board-refresh-members'].forEach(id=>$(id).disabled=active.has(value.status)||!!fillJob);
  }
  async function poll() {
    clearTimeout(timer);if(!job)return;const value=await request(`/api/boards/jobs/${encodeURIComponent(job.job_id)}`);job=value;renderJob(value);
    if(active.has(value.status))timer=setTimeout(()=>poll().catch(error),2000);else {job=null;await refresh();}
  }
  async function start(path,body={}) {job=await request(`/api/boards${path}`,{method:'POST',body:JSON.stringify(body)});renderJob(job);await poll();}
  function fillTime(value){return value?.finished_at||value?.updated_at||value?.created_at||'';}
  function renderFillResult(value){
    $('board-fill-execute').hidden=true;$('board-fill-cancel').hidden=true;
    if(value.result)describeFill(value.result,true);
    else {const host=$('board-fill-plan');host.replaceChildren(node('p',`本批结果 · ${names[value.status]||value.status}${value.error?' · '+value.error:''}`),node('p','任务已结束，未返回完整逐股统计。刷新只读取状态，不会恢复任务。','subtle'));}
  }
  function describeFill(plan,done=false) {
    const host=$('board-fill-plan');host.replaceChildren();
    const scopes={watch:'我的关注',held:'模拟账户持仓',current_board:'当前板块',full:'全市场',selected:'指定股票'};
    host.append(node('p',`${done?'本批结果':'补齐计划'} · ${plan.price_date||fillPlan?.price_date||'—'} · ${scopes[plan.scope||fillPlan?.scope]||plan.scope||fillPlan?.scope||''} ${plan.board_id||fillPlan?.board_id||''} · 请求 ${plan.requested??fillPlan?.requested??'—'} 只`));
    const names={reuse:'可直接复用',append:'可追加',rebuild:'需重建该股',missing_price:'缺目标日行情',failed:'核验失败',reused:'已复用',appended:'已追加',rebuilt:'已重建',deferred:'下批待处理'};
    const counts=done?['reused','appended','rebuilt','missing_price','failed','deferred'].map(key=>[key,plan.counts?.[key]??'未记录']):Object.entries(plan.counts||{});
    metrics(host,counts.map(([key,value])=>[names[key]||key,String(value)]));
    host.append(node('p',done?`状态 ${({completed:'完成',partial:'部分完成',cancelled:'已取消',failed:'失败'})[plan.status]||plan.status||'待核验'} · 全量结构 ${plan.full_published?'已发布':'仍未完整发布'}。刷新只读取结果，已取消任务保持终止。`:`本批最多计算 ${plan.max_symbols} 只 · 预计延后 ${plan.estimated_deferred??0} 只 · 预算 ${plan.budget_seconds} 秒。已取消的任务保持终止。`,'subtle'));
    const details=node('details'),summary=node('summary','逐股差异');details.append(summary);
    (plan.rows||[]).forEach(row=>details.append(node('p',`${row.symbol} · ${names[row.status||row.action]||row.status||row.action} · ${row.reason||''}${row.first_changed_date?' · 最早变化 '+row.first_changed_date:''}`)));host.append(details);
  }
  async function previewFill(){
    if(fillJob)return;
    const scope=$('board-fill-scope').value;
    if(scope==='current_board'&&!selectedCapital)throw Error('先从资金图选择一个板块。');
    $('board-fill-execute').hidden=true;
    fillPlan=await request('/api/boards/incremental/plan',{method:'POST',body:JSON.stringify({scope,...(scope==='current_board'?{board_id:selectedCapital.board_id}:{}),price_date:capitalData?.price_date,max_symbols:20,budget_seconds:300})});
    describeFill(fillPlan);$('board-fill-execute').hidden=false;
  }
  async function pollFill(){
    clearTimeout(fillTimer);if(!fillJob)return;
    const j=await request('/api/light/jobs/'+encodeURIComponent(fillJob));
    $('board-fill-plan').replaceChildren(node('p',`${names[j.status]||j.status} · ${j.progress?.stage||''} ${j.progress?.completed??''} / ${j.progress?.total??''}`));
    if(active.has(j.status)){fillTimer=setTimeout(()=>pollFill().catch(error),1500);return;}
    fillJob=null;fillPlan=null;fillResult=j;$('board-fill-cancel').hidden=true;$('board-sync').disabled=false;
    renderFillResult(j);
    await refresh();
  }
  async function executeFill(){if(!fillPlan||fillJob)return;
    if(fillPlan.scope==='current_board'&&fillPlan.board_id!==selectedCapital?.board_id){clearFillPlan();throw Error('选定板块已变化，请重新预览缺口。');}
    const j=await request('/api/boards/incremental/execute',{method:'POST',body:JSON.stringify({plan_id:fillPlan.plan_id})});
    fillJob=j.job_id;$('board-fill-execute').hidden=true;$('board-fill-cancel').hidden=false;$('board-sync').disabled=true;await pollFill();
  }
  const run=(path,body)=>start(path,body).catch(error);
  if(!$('scan-board-panel'))return;
  setupCapital();
  window.addEventListener('resize',()=>{if(capitalData && !$('scan-board-panel').hidden){$('capital-tooltip').hidden=true;renderCapital(capitalData);}});
  window.addEventListener('keydown',event=>{if(event.key==='Escape')$('capital-tooltip').hidden=true;});
  $('scan-stock-tab').addEventListener('click',()=>selectTab(false));$('scan-board-tab').addEventListener('click',()=>{const alreadyLoaded=loaded;selectTab(true);if(alreadyLoaded)refresh().catch(error);});
  ['board-kind','board-ranking'].forEach(id=>$(id).addEventListener('change',()=>refresh().catch(error)));
  $('board-refresh-view').addEventListener('click',()=>refresh().catch(error));$('board-calculate').addEventListener('click',()=>run('/features/build',{with_native:true}));$('board-sync').addEventListener('click',()=>previewFill().catch(error));
  $('board-fill-execute').addEventListener('click',()=>executeFill().catch(error));
  $('board-fill-cancel').addEventListener('click',()=>{if(fillJob)request('/api/light/jobs/'+encodeURIComponent(fillJob)+'/cancel',{method:'POST',body:'{}'}).catch(error);});
  $('board-fill-scope').addEventListener('change',clearFillPlan);
  $('board-refresh-members').addEventListener('click',()=>run('/refresh'));$('board-detail-close').addEventListener('click',()=>{++detailSequence;$('board-detail').hidden=true;});
  const panel={onRoute(){const wasLoaded=loaded;const desired=new URLSearchParams(location.search).get('board')==='1' || sessionStorage.getItem('khquant_scan_tab')==='board';selectTab(desired);if(desired && wasLoaded)refresh().catch(error);},refresh,async retryTask(id){const value=await request(`/api/boards/jobs/${encodeURIComponent(id)}`);selectTab(true);location.hash='scan';renderJob(value);},showResult(result){selectTab(true);refresh().catch(error);request('/api/boards/jobs').then(async data=>{const saved=(data.jobs || []).find(x=>x.run_id===result.run_id);if(saved)renderJob(await request(`/api/boards/jobs/${encodeURIComponent(saved.job_id)}`));}).catch(error);}};
  globalThis.KHQuantBoardPanel=panel;panel.onRoute();
})();
