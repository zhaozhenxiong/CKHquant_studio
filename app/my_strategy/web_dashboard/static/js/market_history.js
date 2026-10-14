/* Saved scan aggregates and paged facts; read-only causal reference facts. */
(function (root) {
  'use strict';
  const mounts = new WeakMap();
  const referenceMode = summary => summary?.analysis_mode === 'reference' || summary?.account_scope === 'none';
  const referenceStateKeys = ['direction_buy','direction_sell','candidates','new_candidates','risk_blocked','failed','missing_price'];
  const countLabels = {direction_buy:'买方方向',direction_sell:'卖方方向',candidates:'有效买方候选',new_candidates:'新增买方候选',failed:'分析失败',missing_price:'缺当日行情'};
  const qualityLabels = {failed:'分析失败 · 未评估',missing_price:'缺当日行情 · 未评估',no_candidate:'无合格方向候选',risk_blocked:'风险阻止 · 未评估',input_ineligible:'输入不合格 · 未评估',model_unavailable:'缺兼容模型 · 未评估',no_prior_high_return_head:'缺先验模型 · 未评估',identity_error:'评分身份核验失败',entry_quality_gate_disabled:'质量过滤关闭 · 未评分',scored:'已评分',scored_passed:'已评分 · 通过',scored_rejected:'已评分 · 未通过'};
  const reasonLabels = {scan_failed:'股票分析失败',missing_market_session_price:'缺核验当日行情',risk_stop_distance_outside_contract:'止损距离超出冻结范围',risk_blocked_unscored:'风险阻止，高收益未评估',no_qualified_direction_candidate:'未进入合格方向候选评分流程',no_live_unconsumed_opportunity:'没有有效且未消费的新机会',high_return_below_threshold:'高收益分数低于门槛',buy_total_support_below_threshold:'买方总分低于门槛',no_prior_high_return_head:'该日没有合法先验高收益模型',missing_target_market_session:'缺核验当日行情',input_ineligible:'输入不合格',failed:'分析失败'};
  const esc = value => String(value == null ? '' : value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const finite = value => value !== null && value !== undefined && typeof value !== 'boolean' && value !== '' && Number.isFinite(Number(value));
  const count = value => finite(value) && Number(value) >= 0 ? Number(value) : null;
  const number = value => count(value) === null ? '—' : Number(value).toLocaleString('zh-CN', {maximumFractionDigits:0});
  const percent = (value, denominator) => count(value) === null || count(denominator) === null || Number(denominator) <= 0 ? null : Number(value) / Number(denominator) * 100;
  const metricValue = (day, key, mode) => {
    const raw = count((day?.counts || {})[key] ?? (day?.coverage || {})[key]);
    if (!referenceStateKeys.includes(key) || count(day?.coverage?.price_available) === 0 && !['failed','missing_price'].includes(key)) return null;
    return mode === 'rate' ? percent(raw, day?.universe_size) : raw;
  };
  const metricText = (day, key, mode) => {
    const value = metricValue(day, key, mode);
    return value === null ? '—' : mode === 'rate' ? `${value.toFixed(1)}%` : number(value);
  };
  const supportText = value => finite(value) && Number(value) >= 0 && Number(value) <= 1 ? (Number(value) * 100).toFixed(1) : '未评估';
  const qualityText = status => qualityLabels[status] || String(status || '状态未提供');
  const options = (items, selected, allLabel) => `${allLabel ? `<option value="">${esc(allLabel)}</option>` : ''}${items.map(([value,label]) => `<option value="${esc(value)}"${String(selected) === String(value) ? ' selected' : ''}>${esc(label)}</option>`).join('')}`;
  const metricKeys = ['direction_buy','direction_sell','candidates'];
  const stateKeys = Object.keys(countLabels);

  // Redrawing a selected date must retain the user's place in keyboard navigation.
  const controls = container => [...(container.querySelectorAll?.('button, select, rect[role="button"]') || [])];
  function matchesFocus(node, key) {
    if (String(node.tagName).toLowerCase() !== key.tag) return false;
    if (key.page) return Object.hasOwn(node.dataset || {},'mhPage') && (node.textContent || '').trim() === key.page;
    return key.fields.every(([name,value]) => node.dataset?.[name] === value);
  }
  function captureFocus(container) {
    const active=container.ownerDocument?.activeElement;
    if (!active || !container.contains(active)) return null;
    const fields=Object.entries(active.dataset || {}).filter(([name]) => name.startsWith('mh') && !['mhAnalyze','mhPage'].includes(name));
    const page=Object.hasOwn(active.dataset || {},'mhPage') ? (active.textContent || '').trim() : null;
    if (!fields.length && !page) return null;
    const key={tag:String(active.tagName).toLowerCase(),fields,page};
    key.ordinal=controls(container).filter(node => matchesFocus(node,key)).indexOf(active);
    return key;
  }
  function restoreFocus(container, key) {
    if (!key) return;
    const matches=controls(container).filter(node => matchesFocus(node,key) && !node.disabled);
    const node=matches[key.ordinal] || matches[0] || (key.page ? controls(container).find(item => Object.hasOwn(item.dataset || {},'mhPage') && !item.disabled) : null);
    node?.focus?.({preventScroll:true});
  }

  function normalize(summary) {
    if (!summary || !Array.isArray(summary.dates) || !Array.isArray(summary.daily)) return null;
    const dates = summary.dates;
    if (!dates.length || dates.length > 30 || dates.some((date,i) => !/^\d{4}-\d{2}-\d{2}$/.test(date) || (i > 0 && date <= dates[i-1]))) return null;
    const byDate = new Map(summary.daily.map(day => [day.date,day]));
    if (byDate.size !== summary.daily.length || dates.some(date => !byDate.has(date))) return null;
    return {...summary, dates:[...dates], daily:dates.map(date => byDate.get(date)), sectors:Array.isArray(summary.sectors) ? summary.sectors : [], event_types:Array.isArray(summary.event_types) ? summary.event_types : [], notes:Array.isArray(summary.notes) ? summary.notes : []};
  }

  function coverageHtml(summary, selectedDate) {
    const strip = summary.daily.map(day => {
      const coverage = day.coverage || {}, rate = percent(coverage.price_available,day.universe_size);
      const title = `${day.date} · 行情 ${number(coverage.price_available)}/${number(day.universe_size)} · 输入 ${number(coverage.input_available)} · 趋势模型评分 ${number(coverage.model_scored)} · 失败 ${number(coverage.failed)} · 缺行情 ${number(coverage.missing_price)}`;
      return `<button type="button" data-mh-date="${esc(day.date)}" data-selected="${day.date === selectedDate}" aria-pressed="${day.date === selectedDate}" title="${esc(title)}" aria-label="${esc(title)}">${rate === null ? '' : `<i style="height:${Math.min(100,Math.max(0,rate))}%"></i>`}</button>`;
    }).join('');
    return `<h3>每日覆盖</h3><p class="mh-note">蓝色为取得当日行情结果的股票占比；缺行情和计算失败分别统计，失败不代表停牌。点击日期查看明细。</p><div class="mh-coverage" style="grid-template-columns:repeat(${summary.dates.length},minmax(0,1fr))">${strip}</div><div class="mh-date-labels"><span>${esc(summary.dates[0])}</span><span>${esc(summary.dates.at(-1))}</span></div>`;
  }

  function trendHtml(summary, selectedDate, mode) {
    const width=900, height=210, left=42, right=18, top=15, bottom=34;

    const series = [['direction_buy','buy'],['direction_sell','sell']];
    const values = series.flatMap(([key]) => summary.daily.map(day => metricValue(day,key,mode))).filter(value => value !== null);
    const maximum = mode === 'rate' ? Math.max(1,...values) : Math.max(1,...values);
    const ymax = mode === 'rate' ? Math.min(100,Math.ceil(maximum / 5) * 5) : Math.max(2,Math.ceil(maximum / 2) * 2);
    const x = i => left + i * (width-left-right) / Math.max(1,summary.dates.length-1), y = value => height-bottom-value/ymax*(height-top-bottom);
    const ticks = [0,.5,1].map(fraction => `<line class="mh-gridline" x1="${left}" y1="${y(ymax*fraction)}" x2="${width-right}" y2="${y(ymax*fraction)}"/><text x="${left-7}" y="${y(ymax*fraction)+3}" text-anchor="end">${mode==='rate' ? `${(ymax*fraction).toFixed(0)}%` : number(ymax*fraction)}</text>`).join('');
    const paths = series.map(([key,cls]) => {
      let segment=false;
      const path=summary.daily.map((day,i) => {const value=metricValue(day,key,mode); if(value===null){segment=false;return '';} const command=segment?'L':'M';segment=true;return `${command}${x(i).toFixed(2)},${y(value).toFixed(2)}`;}).join(' ');
      return `<path class="mh-line ${cls}" d="${path}"/>`;
    }).join('');
    const selectedIndex=summary.dates.indexOf(selectedDate), selectedLine=selectedIndex<0?'':`<line class="mh-selected-date" x1="${x(selectedIndex)}" y1="${top}" x2="${x(selectedIndex)}" y2="${height-bottom}"/>`;
    const hits=summary.daily.map((day,i)=> {const dx=(width-left-right)/Math.max(1,summary.dates.length-1), title=`${day.date} · ${series.map(([key])=>`${countLabels[key]} ${metricText(day,key,mode)}`).join(' · ')}`;return `<rect class="mh-hit" tabindex="0" role="button" data-mh-date="${esc(day.date)}" aria-label="${esc(title)}" x="${Math.max(left,x(i)-dx/2)}" y="${top}" width="${Math.min(dx,width-right-Math.max(left,x(i)-dx/2))}" height="${height-top-bottom}"><title>${esc(title)}</title></rect>`;}).join('');
    const labels=[...new Set([0,Math.floor((summary.dates.length-1)/2),summary.dates.length-1])].map(i=>`<text x="${x(i)}" y="${height-10}" text-anchor="${i===0?'start':i===summary.dates.length-1?'end':'middle'}">${esc(summary.dates[i].slice(5))}</text>`).join('');
    return `<h3>${'买方／卖方参考方向'}</h3><div class="mh-legend"><span><i class="mh-key buy"></i>买方方向</span><span><i class="mh-key sell"></i>卖方方向</span>${''}</div><svg class="mh-chart" viewBox="0 0 ${width} ${height}" role="group" aria-label="${mode==='rate'?'股票比例':'股票数量'}随交易日的变化">${ticks}${selectedLine}${paths}${hits}${labels}</svg><p class="mh-note">${'B/S 参考方向反映当日市场证据，按股票统计。'}</p>`;
  }

  function barsHtml(day, keys, mode) {
    const maximum = Math.max(1,...keys.map(key => count((day.counts || {})[key]) || 0));
    return keys.map(key => {const value=count((day.counts || {})[key]);return `<button class="mh-bar-row" type="button" data-mh-state="${key}" title="点击查看${countLabels[key]}股票"><span>${countLabels[key]}</span><span class="mh-bar-track">${value===null?'':`<i style="width:${value/maximum*100}%"></i>`}</span><strong>${metricText(day,key,mode)}</strong></button>`;}).join('');
  }

  function sectorsHtml(summary, state) {
    const classificationNote=summary.classification?.mode==='current_snapshot'?'当前行业分类映射历史信号，不代表当时的行业归属。':'行业资料未核验，未知分类保留在分母中。';
    const metric=state.metric, ranked=[...summary.sectors].sort((a,b) => (b.daily||[]).reduce((sum,day)=>sum+(count(day.counts?.[metric])||0),0)-(a.daily||[]).reduce((sum,day)=>sum+(count(day.counts?.[metric])||0),0) || String(a.name||a.id).localeCompare(String(b.name||b.id),'zh-CN'));
    const sectors=state.sector?ranked.filter(sector=>sector.id===state.sector):ranked.slice(0,20);
    const unknown=summary.sectors.find(sector=>sector.id===summary.unknown_sector_id);
    if(!state.sector&&unknown&&!sectors.some(sector=>sector.id===unknown.id)){if(sectors.length===20)sectors.pop();sectors.push(unknown);}
    const allValues=summary.sectors.flatMap(sector=>(sector.daily||[]).map(day=>metricValue(day,metric,state.mode))).filter(value=>value!==null), maximum=Math.max(1,...allValues);
    const rows=sectors.map(sector=>{const byDate=new Map((sector.daily||[]).map(day=>[day.date,day]));return `<tr><td><button type="button" data-mh-sector="${esc(sector.id)}">${esc(sector.name || (sector.id === summary.unknown_sector_id ? '未知板块' : sector.id))}</button><small>${number(sector.member_count)} 只</small></td>${summary.dates.map(date=>{const day=byDate.get(date),value=day?metricValue(day,metric,state.mode):null,available=day && count(day.coverage?.price_available)>0, text=value===null?'—':state.mode==='rate'?`${value.toFixed(0)}%`:number(value), label=`${sector.name || sector.id} · ${date} · ${countLabels[metric]} ${text} · 行情覆盖 ${number(day?.coverage?.price_available)}/${number(day?.universe_size)}`,alpha=value===null?0:.06+.62*Math.min(1,value/maximum);return `<td><button class="mh-heat-cell" type="button" data-mh-sector="${esc(sector.id)}" data-mh-date="${date}" data-mh-state="${metric}" data-available="${Boolean(available)}" data-selected="${state.date===date&&state.sector===sector.id}" title="${esc(label)}" aria-label="${esc(label)}"${available&&value!==null?` style="background:color-mix(in srgb,var(--accent) ${(alpha*100).toFixed(1)}%,var(--panel))"`:''}>${text}</button></td>`;}).join('')}</tr>`;}).join('');
    return `<div class="mh-card-head"><h3>行业 × 交易日</h3><div class="mh-controls"><label>颜色与数值<select data-mh-metric>${options(metricKeys.map(key=>[key,countLabels[key]]),metric)}</select></label><label>查看行业<select data-mh-heat-sector>${options(summary.sectors.map(sector=>[sector.id,sector.name||sector.id]),state.sector,'前20个行业')}</select></label></div></div><p class="mh-note">${classificationNote} 默认显示前20个行业，可选择其余行业。比例分母为行业成员数，灰色纹理表示无当日行情覆盖。点击格子可查看股票。</p>${sectors.length?`<div class="mh-heat-scroll"><table class="mh-heat"><thead><tr><th scope="col">行业 / 股票数</th>${summary.dates.map(date=>`<th scope="col"><button type="button" data-mh-date="${date}">${date.slice(5)}</button></th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`:'<p class="mh-note">未提供行业聚合；股票仍可按日期和状态查看。</p>'}`;
  }

  function eventsHtml(summary, day) {
    const events=day.events||{},newTypes=events.new_by_type||{},activeTypes=events.active_by_type||{},ids=[...new Set([...Object.keys(newTypes),...Object.keys(activeTypes)])],labels=new Map(summary.event_types.map(event=>[event.id,event.label]));
    ids.sort((a,b)=>(count(newTypes[b])||0)+(count(activeTypes[b])||0)-(count(newTypes[a])||0)-(count(activeTypes[a])||0));
    const maximum=Math.max(1,...ids.map(id=>count(activeTypes[id])||count(newTypes[id])||0)),support=day.mean_support||{};
    return `<h3>量价事件与连续支持</h3><p class="mh-note">${day.date} · 新增 ${number(events.new_total)} 条 · 有效 ${number(events.active_total)} 条 · 失效 ${number(events.invalidated)} 条。事件统计为条数，独立于股票比例。</p><div class="mh-events">${ids.length?ids.map(id=>`<button type="button" class="mh-bar-row" data-mh-event="${esc(id)}" title="点击查看该日新增或有效的${esc(labels.get(id)||id)}"><span>${esc(labels.get(id)||id)}</span><span class="mh-bar-track"><i style="width:${(count(activeTypes[id])||count(newTypes[id])||0)/maximum*100}%"></i></span><strong>${number(newTypes[id]??0)} / ${number(activeTypes[id]??0)}</strong></button>`).join(''):'<p class="mh-note">本日没有已记录的新增或有效量价事件。</p>'}</div><p class="mh-note">上方每行：新增 / 有效事件条数；点击后按涉及股票去重。</p><div class="mh-support">${[['continuous','连续支持'],['event','事件支持'],['effective','最终有效支持']].map(([key,label])=>`<div><small>${label} · 均值/100</small><strong>需 ${supportText(support[`${key}_demand`])} · 供 ${supportText(support[`${key}_supply`])}</strong></div>`).join('')}</div>`;
  }

  function viewHtml(summary, state) {
    const day=summary.daily.find(item=>item.date===state.date)||summary.daily.at(-1), coverage=day.coverage||{},classification=summary.classification||{};

    const cards=[['取得当日行情结果',`${number(coverage.price_available)} / ${number(day.universe_size)}`,`缺行情 ${number(coverage.missing_price)} · 失败 ${number(coverage.failed)}`,''],['趋势模型评分',number(coverage.model_scored),`输入合格 ${number(coverage.input_available)}；缺模型不是零分`,''],['B / S 参考方向',`${number(day.counts?.direction_buy)} / ${number(day.counts?.direction_sell)}`,'按股票去重','direction_buy'],['新增 / 有效买方证据',`${number(day.counts?.new_candidates)} / ${number(day.counts?.candidates)}`,'独立来源的方向事实','candidates']];
    return `<div class="mh-heading"><div><h2>最近30个交易日市场状态</h2><p>${esc(summary.dates[0])} — ${esc(summary.dates.at(-1))} · 已保存 ${summary.dates.length} 个核验交易日 · 本次股票池 ${number(summary.universe_size)} 只</p></div><div class="mh-controls"><label>观察日期<select data-mh-date-select>${options(summary.dates.map(date=>[date,date]),state.date)}</select></label><div class="segmented" aria-label="统计显示方式"><button type="button" data-mh-mode="count" aria-pressed="${state.mode==='count'}">数量</button><button type="button" data-mh-mode="rate" aria-pressed="${state.mode==='rate'}">比例</button></div></div></div><p class="mh-context">数量按股票去重；比例分母为本次股票池，行业格子使用该行业成员数。${'B/S 与量价评分用于分析参考。'}</p><div class="mh-metrics">${cards.map(([label,value,note,key])=>`<div class="mh-metric">${key?`<button type="button" data-mh-state="${key}">`:''}<small>${label}</small><strong>${value}</strong><small>${note}</small>${key?'</button>':''}</div>`).join('')}</div><div class="mh-grid"><section class="mh-card">${coverageHtml(summary,state.date)}${trendHtml(summary,state.date,state.mode)}</section><section class="mh-card"><h3>${esc(state.date)} · 候选与评分状态</h3><p class="mh-note">显示已保存的阶段状态；候选、排除和动作可交叉，不按相减推算。</p><div class="mh-funnel">${barsHtml(day,['new_candidates','candidates','direction_buy','direction_sell','failed','missing_price'],state.mode)}</div></section></div><section class="mh-card mh-sector-card">${sectorsHtml(summary,state)}</section><section class="mh-card">${eventsHtml(summary,day)}</section><section class="mh-drill"><h3>股票明细</h3><div class="mh-controls"><label>行业<select data-mh-sector-select>${options(summary.sectors.map(sector=>[sector.id,sector.name||sector.id]),state.sector,'全部行业')}</select></label><label>状态<select data-mh-state-select>${options((referenceStateKeys).map(key=>[key,countLabels[key]]),state.filterState,'全部状态')}</select></label><label>量价事件<select data-mh-event-select>${options(summary.event_types.map(event=>[event.id,event.label||event.id]),state.event,'全部事件')}</select></label><button type="button" data-mh-reset>清除筛选</button></div><p class="mh-filter-status" data-mh-filter-status></p><div data-mh-facts aria-live="polite"></div></section><div class="mh-note-block"><p>${classification.mode==='current_snapshot'?'行业归属使用本次保存的当前分类快照映射到历史，可能与当时归属不同。':'没有已核验行业分类，未分类股票归入未知板块。'} 已分类 ${number(classification.classified_symbols)} · 未知 ${number(classification.unknown_symbols)}。</p>${classification.note?`<p>${esc(classification.note)}</p>`:''}<details><summary>统计口径与来源说明</summary>${summary.classification?.source?`<p>分类来源 ${esc(summary.classification.source)} · 获取 ${esc(summary.classification.observed_at||'未记录')}</p>`:''}${summary.notes.map(note=>`<p>${esc(typeof note==='string'&&note.includes('强度单位为0至1')?'需求与供给均值按0–100显示，仅统计该日输入可用的股票。':typeof note==='string'?note:JSON.stringify(note))}</p>`).join('')}</details></div>`;
  }

  function filterUrl(runId, state) {
    if (!runId) throw new Error('缺少已保存扫描运行，无法读取股票明细');
    const query=new URLSearchParams({date:state.date,page:String(state.page),page_size:'25'});
    if(state.sector)query.set('sector',state.sector);
    if(state.event)query.set('event_type',state.event);
    if(state.filterState)query.set('state',state.filterState);
    return `/api/runs/${encodeURIComponent(runId)}/market-history?${query}`;
  }
  function filtersLabel(summary,state) {
    const sector=summary.sectors.find(item=>item.id===state.sector),event=summary.event_types.find(item=>item.id===state.event);
    return [state.date,sector?.name,state.filterState&&countLabels[state.filterState],event?.label].filter(Boolean).join(' · ');
  }
  function factsHtml(data, summary, state) {

    const rows=data.rows, total=count(data.total),page=count(data.page)||state.page,size=count(data.page_size)||25;
    const table=rows.length?`<div class="table-wrap"><table><thead><tr><th>股票 / 行业</th><th>${'B / S 参考方向'}</th><th>${'趋势模型状态'}</th><th>事件 / 原因</th><th>查看</th></tr></thead><tbody>${rows.map((row,index)=>{
      const direction=row.direction==='BUY'?'▲ 买方':row.direction==='SELL'?'▼ 卖方':row.direction==='WAIT'?'观望':'未提供';
      const eventLabels=new Map(summary.event_types.map(event=>[event.id,event.label])),events=[...new Set([...(row.new_events||[]),...(row.active_events||[])].map(event=>typeof event==='string'?event:event.event_type||event.type).filter(Boolean))].map(id=>eventLabels.get(id)||id).join('、');
      const reasons=[...new Set([...(row.reason_codes||[]),...(row.model_resolution?.reason_codes||[])])].map(code=>reasonLabels[code]||code).join('；');
      return `<tr><td><strong>${esc(row.symbol)}</strong><span class="mh-row-name">${esc(row.name||'')} · ${esc(row.sector||'未知板块')}</span>${row.price_available===false?`<span class="mh-row-name">${row.counts?.failed?'计算失败，行情状态未核验':'缺当日行情'}</span>`:''}</td><td><span class="badge ${row.direction==='BUY'?'buy':row.direction==='SELL'?'sell':''}">${direction}</span>${''}</td><td>${row.price_available === false ? '行情缺失 · 未评分' : row.input_available === false || row.model_input_eligible === false || row.input_eligible === false ? '输入不足 · 未评分' : row.model_scored === true || row.trend_model_scored === true || [row.p_up,row.p_down,row.p_neutral].every(value => finite(value) && Number(value) >= 0 && Number(value) <= 1) ? '已评分' : '缺少合格模型 · 未评分'}</td><td class="mh-reason">${esc(events||'无已记录事件')}<span class="mh-row-name">${esc(reasons||'—')}</span>${['scored','scored_passed','scored_rejected'].includes(row.quality_status) ? `<span class="mh-row-name">已保存质量诊断：${esc(qualityText(row.quality_status))}</span>` : ''}</td><td><button type="button" class="link-button mh-analysis-button" data-mh-analyze="${index}">按此日分析</button></td></tr>`;
    }).join('')}</tbody></table></div>`:'<p class="mh-empty">该日没有符合当前筛选条件的股票。</p>';
    return `${table}<div class="pagination"><span>${number(total)} 只 · 第 ${page} / ${Math.max(1,Math.ceil((total||0)/size))} 页</span><button type="button" data-mh-page="${page-1}"${page<=1?' disabled':''}>上一页</button><button type="button" data-mh-page="${page+1}"${total===null||page*size>=total?' disabled':''}>下一页</button></div>`;
  }

  function clear(container,message) {
    if(!container)return;
    const previous=mounts.get(container);
    if(previous){previous.sequence++;container.removeEventListener('click',previous.click);container.removeEventListener('change',previous.change);container.removeEventListener('keydown',previous.keydown);mounts.delete(container);}
    container.innerHTML=`<div class="mh-empty"><strong>最近30个交易日市场状态</strong><p>${esc(message||'此扫描结果未保存30交易日状态。重新扫描后可查看历史覆盖、板块和量价事件。')}</p></div>`;
  }

  function render(container, source, settings={}) {
    if(!container)return null;
    const summary=normalize(source);
    if(!summary){clear(container,source?'此结果没有可用的已保存交易日状态，无法生成历史图表。':'此扫描结果未保存30交易日状态。完成新扫描后可查看；旧结果保留原扫描列表。');return null;}
    const runId=settings.runId||summary.run_id,previous=mounts.get(container);
    if(previous?.source===source&&previous.runId===runId)return previous.controller;
    const prior=previous?.runId===runId?previous.state:null;
    clear(container);
    const state={date:prior&&summary.dates.includes(prior.date)?prior.date:summary.dates.at(-1),mode:prior?.mode||'count',metric:prior?.metric||'direction_buy',allSectors:prior?.allSectors||false,sector:prior?.sector||'',event:prior?.event||'',filterState:prior?.filterState||'',page:prior?.page||1};
    const mounted={source,runId,summary,state,sequence:0,rows:[],data:null,controller:null};
    const request=settings.request||async function(url){const response=await fetch(url,{headers:{Accept:'application/json'}});if(!response.ok)throw new Error(`读取失败 (${response.status})`);return response.json();};
    async function loadFacts(){
      const sequence=++mounted.sequence,target=container.querySelector('[data-mh-facts]'),label=container.querySelector('[data-mh-filter-status]');
      const focus=captureFocus(container), document=container.ownerDocument;
      label.textContent=filtersLabel(summary,state);target.innerHTML='<p class="mh-empty mh-loading">正在读取已保存的股票明细…</p>';mounted.rows=[];mounted.data=null;
      try{const data=await request(filterUrl(runId,state));if(mounts.get(container)!==mounted||sequence!==mounted.sequence)return;
        if(!data||!Array.isArray(data.rows)||data.run_id!==runId)throw new Error('股票明细与当前扫描运行不匹配');
        for(const [key,value] of [['date',state.date],['sector',state.sector],['event_type',state.event],['state',state.filterState]])if(data.filters&&Object.hasOwn(data.filters,key)&&(data.filters[key]||'')!==value)throw new Error('股票明细筛选与当前选择不匹配');
        mounted.rows=data.rows;mounted.data=data;target.innerHTML=factsHtml(data,summary,state);
        if (!document?.activeElement || document.activeElement === document.body) restoreFocus(container,focus);
      }catch(error){if(mounts.get(container)===mounted&&sequence===mounted.sequence)target.innerHTML=`<p class="mh-empty mh-error" role="alert">${esc(error.message||error)}</p><button type="button" data-mh-retry>重新读取明细</button>`;}
    }
    function paint(load=true){const focus=captureFocus(container);container.innerHTML=viewHtml(summary,state);if(!load&&mounted.data){container.querySelector('[data-mh-filter-status]').textContent=filtersLabel(summary,state);container.querySelector('[data-mh-facts]').innerHTML=factsHtml(mounted.data,summary,state);}else mounted.controller.ready=loadFacts();restoreFocus(container,focus);}
    function filter(values){Object.assign(state,values,{page:1});paint();}
    mounted.click=function(event){const target=event.target.closest?.('[data-mh-date],[data-mh-mode],[data-mh-sector],[data-mh-state],[data-mh-event],[data-mh-page],[data-mh-analyze],[data-mh-reset],[data-mh-sectors-toggle],[data-mh-retry]');if(!target||!container.contains(target)||target.disabled)return;
      const data=target.dataset;
      if(data.mhAnalyze!==undefined){const row=mounted.rows[Number(data.mhAnalyze)],context={symbol:row?.symbol,date:state.date,runId,row};if(row&&settings.onAnalyze)Promise.resolve().then(()=>settings.onAnalyze(context)).catch(error=>{const label=container.querySelector('[data-mh-filter-status]');if(label){label.textContent=error.message||String(error);label.classList.add('mh-error');}});return;}
      if(data.mhPage!==undefined){state.page=Math.max(1,Number(data.mhPage)||1);mounted.controller.ready=loadFacts();return;}
      if(data.mhRetry!==undefined){mounted.controller.ready=loadFacts();return;}
      if(data.mhReset!==undefined){filter({sector:'',event:'',filterState:''});return;}
      if(data.mhMode){state.mode=data.mhMode==='rate'?'rate':'count';paint(false);return;}
      if(data.mhSectorsToggle!==undefined){state.allSectors=!state.allSectors;paint();return;}
      const values={};if(data.mhDate)values.date=data.mhDate;if(data.mhSector)values.sector=data.mhSector;if(data.mhEvent)values.event=data.mhEvent;if(data.mhState)values.filterState=data.mhState;filter(values);
    };
    mounted.change=function(event){const target=event.target,data=target.dataset||{};if(Object.hasOwn(data,'mhDateSelect'))filter({date:target.value});else if(Object.hasOwn(data,'mhSectorSelect'))filter({sector:target.value});else if(Object.hasOwn(data,'mhStateSelect'))filter({filterState:target.value});else if(Object.hasOwn(data,'mhEventSelect'))filter({event:target.value});else if(Object.hasOwn(data,'mhHeatSector'))filter({sector:target.value});else if(Object.hasOwn(data,'mhMetric')){state.metric=target.value;paint(false);}};
    mounted.keydown=function(event){if((event.key==='Enter'||event.key===' ')&&event.target.matches?.('rect[data-mh-date]')){event.preventDefault();mounted.click(event);}};
    mounted.controller={ready:null,getState:()=>({...state}),refresh:()=>{mounted.controller.ready=loadFacts();return mounted.controller.ready;}};
    mounts.set(container,mounted);container.addEventListener('click',mounted.click);container.addEventListener('change',mounted.change);container.addEventListener('keydown',mounted.keydown);paint();return mounted.controller;
  }
  const api={referenceMode,render,clear,normalize,metricValue,metricText,supportText,coverageHtml,trendHtml,sectorsHtml,eventsHtml,viewHtml,filterUrl,factsHtml};
  if(typeof module==='object'&&module.exports)module.exports=api;
  if(root)root.KHMarketHistory=api;
})(typeof window!=='undefined'?window:globalThis);
