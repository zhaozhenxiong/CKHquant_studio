/* Display-only evidence: every score keeps its saved target and original date. */
(function (root) {
  'use strict';
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const date = (value) => String(value ?? '').slice(0,10);
  const finite = (value) => (typeof value === 'number' || typeof value === 'string' && value.trim() !== '') && Number.isFinite(Number(value));
  const num = (value, digits = 2) => finite(value) ? Number(value).toLocaleString('zh-CN',{maximumFractionDigits:digits}) : '—';
  const probability = (value) => finite(value) && Number(value) >= 0 && Number(value) <= 1 ? `${num(Number(value) * 100)}%` : '不可用';
  const ruleScore = (value) => finite(value) && Number(value) >= 0 && Number(value) <= 1 ? `${num(Number(value) * 100)} / 100` : '不可用';
  const labels = {ma:'均线 ML',structure:'CZSC 结构',wyckoff:'威科夫量价',high_return:'高收益二阶段'};
  const reasonLabels = {no_candidate_signal:'本日没有新的买入候选',actual_holding_required:'需要兼容的真实持仓记录',high_return_independent_window_missing:'独立前向验证尚未完成',model_unavailable:'该时点没有可用模型',model_not_available_at_signal:'该模型在信号时点尚不可用',ma_model_not_available_at_signal:'均线模型在信号时点尚不可用',ma_model_veto:'均线分数未达到冻结门槛',model_probability_below_threshold:'模型分数未达到冻结门槛',threshold_met:'达到冻结门槛',calendar_gap:'交易日历存在行情缺口',no_saved_ma_probability:'该时点未保存均线入场评分',no_ma_entry_score:'该时点没有已保存的均线入场评分',no_candidate:'本日没有买入候选',missing_saved_score:'该时点没有已保存的评分'};
  const reason = (value) => String(value || '该时点没有可用评分').split(/[；;]/).map((part) => reasonLabels[part] || part).join('；');
  const statusLabel = (value) => ({available:'可用',unavailable:'不可用',shadow:'影子观察',no_candidate:'没有候选',candidate:'候选',qualified:'资格通过',rules_only:'规则判断',rule_evidence:'规则证据',input_veto:'输入未通过',actual_holding_required:'需要真实持仓',scored:'已评分',score_unavailable:'评分不可用',rule_triggered:'规则触发',rule_fallback:'规则回退',no_signal:'未触发',input_unavailable:'输入未通过',blocked:'已阻止',active:'已生成'})[value] || String(value || '不可用');
  function rowAt(evidence, day) { return (evidence?.rows || []).find((row) => date(row.date) === date(day)) || null; }
  function agentView(agent, key) {
    const item = agent || {}, model = item.kind === 'model_probability' || item.score_kind === 'model_probability' || !item.kind && ['ma','high_return'].includes(key);
    let summary;
    if (key === 'structure') summary = `买入触发 ${num(item.buy_count,0)} · 卖出触发 ${num(item.sell_count,0)}`;
    else if (key === 'wyckoff' && !model) summary = `需求 ${ruleScore(item.buy_score)} · 供给 ${ruleScore(item.sell_score)}`;
    else summary = `${key === 'high_return' ? '高收益区间概率' : '入场模型分'} ${probability(item.probability)}`;
    const identity = item.model_identity || {};
    const identityText = ['model_run_id','checkpoint','available_at','train_label_end','actual_training_completed_at','target_contract_hash','manifest_sha256'].filter((field) => identity[field] != null).map((field) => `${({model_run_id:'运行',checkpoint:'检查点',available_at:'可用',train_label_end:'训练标签截止',actual_training_completed_at:'训练完成',target_contract_hash:'目标契约 SHA256',manifest_sha256:'模型清单 SHA256'})[field]} ${identity[field]}`).join(' · ');
    return {label:item.label || labels[key] || key,summary,kind:model ? '模型概率' : '规则证据',status:String(item.status || 'unavailable'),reason:reason(item.reason),target:item.target || identity.target || '',threshold:item.threshold,identityText,
      pointTypes:Array.isArray(item.point_types) ? item.point_types.join('、') : '',exit:key === 'ma' || key === 'high_return' ? `持仓退出概率 ${probability(item.exit_probability)}${finite(item.exit_probability) ? '' : ` · ${reason(item.exit_reason || item.exit_status || 'actual_holding_required')}`}` : ''};
  }
  function agentsHtml(row, compact = false) {
    return `<div class="agent-evidence-grid">${Object.keys(labels).map((key) => {
      const view = agentView(row?.agents?.[key],key);
      return `<section class="agent-evidence"><div class="agent-heading"><strong>${esc(view.label)}</strong><span>${esc(view.kind)} · ${esc(statusLabel(view.status))}</span></div><p class="agent-score">${esc(view.summary)}</p><p class="subtle">${esc(view.reason)}</p>${!compact ? `${view.pointTypes ? `<p>${esc(view.pointTypes)}</p>` : ''}${view.exit ? `<p class="subtle">${esc(view.exit)}</p>` : ''}<details><summary>评分目标与模型身份</summary>${view.target ? `<p>目标：${esc(view.target)}</p>` : ''}${finite(view.threshold) ? `<p>冻结门槛 ${esc(probability(view.threshold))}</p>` : ''}${view.identityText ? `<p>${esc(view.identityText)}</p>` : ''}</details>` : ''}</section>`;
    }).join('')}</div>`;
  }
  function referenceHtml(row) {
    if (!row) return '<p class="evidence-unavailable">该日期未取得参考价与评分证据。</p>';
    const ohlc = row.ohlc || {}, priceBasis = {unadjusted:'未复权',mixed_legacy_adjustment_unverified:'旧复权口径未核验'}[row.reference_price_basis] || row.reference_price_basis || '价格口径待核验';
    return `<div class="evidence-price"><strong>收盘参考价 ${num(row.reference_close)} 元</strong><span>${esc(row.reference_price_date || row.date)} · ${esc(priceBasis)} · 非成交价</span></div><p class="evidence-ohlc">开 ${num(ohlc.open)} · 高 ${num(ohlc.high)} · 低 ${num(ohlc.low)} · 收 ${num(ohlc.close)}</p><p class="subtle">下一核验交易日 ${esc(row.next_market_session || '等待核验')}</p>`;
  }
  function riskHtml(plan) {
    if (!plan) return '<p class="subtle">该日没有可用的价格与风险计划。</p>';
    return `<p class="subtle">价格与风险计划：${plan.entry_allowed === true ? '入场资格通过' : plan.entry_allowed === false ? '未通过入场资格' : '入场资格待核验'} · ${esc(statusLabel(plan.status))}${(plan.reason_codes || []).length ? ` · ${esc(plan.reason_codes.map(reason).join('；'))}` : ''}</p><div class="evidence-risk"><span>参考区间 ${num(plan.price_floor)}–${num(plan.price_ceiling)} 元</span><span>失效价 ${num(plan.stop_price)} 元</span><span>风险预算 ${finite(plan.risk_fraction) ? probability(plan.risk_fraction) : '不可用'}</span><span>目标仓位 ${finite(plan.target_weight) ? probability(plan.target_weight) : '不可用'}</span></div>`;
  }
  function groupDates(groups) {
    return [...new Set((groups || []).flatMap((group) => (group.entries || []).map((entry) => date(entry.row?.evidence_date || entry.event_date || entry.row?.date || entry.row?.time || entry.original_date))).filter(Boolean))].sort();
  }
  function popupHtml({evidence,groups = [],barDate,frequency,selectedDate,pinned = false,profileAsOf}) {
    const dates = groupDates(groups), displayedDates = groupDates(groups.map((group) => ({...group,entries:(group.entries || []).filter((entry) => entry.displayed !== false)})));
    const selected = dates.includes(selectedDate) ? selectedDate : displayedDates.at(-1) || date(barDate);
    const row = rowAt(evidence,selected), visible = groups.filter((group) => group.visible !== false);
    const badges = visible.map((group) => `<span class="badge ${group.action === 'BUY' ? 'buy' : 'sell'}">${group.kind === 'execution' ? '实际成交 ' : ''}${group.action === 'BUY' ? 'B' : 'S'}${group.source_count > 1 ? ` · ${group.source_count} 路` : ''} · ${group.event_count || group.trade_count || 0} 条</span>`).join('');
    const entries = groups.flatMap((group) => group.entries || []).filter((entry) => date(entry.row?.evidence_date || entry.event_date || entry.row?.date || entry.row?.time || entry.original_date) === selected);
    return `<header class="chart-popup-heading"><div><strong>${esc(frequency)} ${esc(barDate)}</strong><p>${badges || 'K线判断'}${pinned ? ' · 已固定' : ' · 点击固定'}</p></div><button type="button" data-chart-evidence-close aria-label="关闭图表判断">×</button></header>${dates.length > 1 || dates.length && frequency !== '日线' ? `<div class="evidence-date-list" aria-label="周期内信号可知日期">${dates.map((item) => `<button type="button" data-evidence-date="${esc(item)}" aria-pressed="${item === selected}">${esc(item)}</button>`).join('')}</div><p class="subtle">按信号可知日期查看当日评分；点击日期同步对应日线筹码。</p>` : ''}${frequency !== '日线' && !selectedDate ? `<p class="subtle">筹码按所指周期截至 ${esc(profileAsOf || barDate)}；以下评分保留信号可知日。</p>` : ''}<p class="evidence-original-date">信号可知日 ${esc(selected)}${groups.some((group) => group.closed_at) ? ` · 周期闭合 ${esc(groups.find((group) => group.closed_at)?.closed_at)}` : ''}</p>${referenceHtml(row)}${agentsHtml(row,!pinned)}${pinned ? `${riskHtml(row?.raw_risk_plan)}<div class="evidence-actions"><button type="button" data-evidence-locate="${esc(selected)}">定位事件列表</button><button type="button" data-evidence-replay="${esc(selected)}">按此日重新判断</button></div><details class="evidence-original"><summary>原始事件 ${entries.length} 条${entries.some((entry) => entry.excluded) ? '（含未通过资格的证据）' : ''}</summary>${entries.map((entry) => `<p>${esc(entry.row?.source || entry.source || '规则')} · ${esc(entry.row?.event_type || entry.row?.event || entry.row?.reason || '')}<br>发生 ${esc(entry.row?.original_event_date || entry.event_date)} · 可用 ${esc(entry.row?.available_at || entry.original_date)}${entry.excluded ? `<br>未显示：${esc((entry.exclusion_reasons || []).join('、'))}` : ''}${entry.kind === 'execution' || entry.row?.price != null ? `<br>账本成交价 ${num(entry.row?.price)} 元` : ''}</p>`).join('')}</details>` : ''}`;
  }
  function historyPage(rows, requestedPage = 0, pageSize = 20) {
    const sorted = [...(rows || [])].sort((a,b) => date(b.date).localeCompare(date(a.date))), pages = Math.max(1,Math.ceil(sorted.length / pageSize));
    const page = Math.max(0,Math.min(pages - 1,Number(requestedPage) || 0));
    return {page,pages,total:sorted.length,rows:sorted.slice(page * pageSize,(page + 1) * pageSize)};
  }
  function historyHtml(rows, page = 0, selectedDate = '') {
    const result = historyPage(rows,page);
    const side = (event) => /buy|open|买/i.test(String(event.action || event.side)) ? 'B' : /sell|close|卖/i.test(String(event.action || event.side)) ? 'S' : '';
    const body = result.rows.map((row) => {
      const intents = [...new Set((row.events || []).filter((event) => event.candidate_eligible !== false && event.evidence_visible !== false).map(side).filter(Boolean))].join(' / ') || '—';
      return `<tr${row.date === selectedDate ? ' class="history-selected"' : ''}><td><button type="button" data-history-date="${esc(row.date)}">${esc(row.date)}</button></td><td>${num(row.reference_close)}</td><td>${esc(intents)}</td><td>${probability(row.agents?.ma?.probability)}</td><td>${num(row.agents?.structure?.buy_count,0)} / ${num(row.agents?.structure?.sell_count,0)}</td><td>需求 ${ruleScore(row.agents?.wyckoff?.buy_score)} · 供给 ${ruleScore(row.agents?.wyckoff?.sell_score)}</td><td>${probability(row.agents?.high_return?.probability)}</td></tr>`;
    }).join('');
    return `<div class="table-wrap"><table class="history-evidence-table"><thead><tr><th>日期 · 点击回放</th><th>收盘参考价</th><th>意图</th><th>均线入场概率</th><th>结构买 / 卖事件</th><th>量价需求 / 供给（规则分）</th><th>高收益概率</th></tr></thead><tbody>${body || '<tr><td colspan="7">尚无历史参考记录。</td></tr>'}</tbody></table></div><div class="history-pagination"><button type="button" data-history-page="${result.page - 1}"${result.page === 0 ? ' disabled' : ''}>上一页</button><span>${result.page + 1} / ${result.pages} · ${result.total} 个交易日 · 每页 20 行</span><button type="button" data-history-page="${result.page + 1}"${result.page + 1 >= result.pages ? ' disabled' : ''}>下一页</button></div>`;
  }
  function selectionTransition(current, action) {
    const state = {...{hover:null,pinned:null,selectedDate:null},...current};
    if (action.type === 'clear' || action.type === 'loading') return {hover:null,pinned:null,selectedDate:null};
    if (action.type === 'hover' && !state.pinned) return {...state,hover:action.time || null,selectedDate:null};
    if (action.type === 'leave' && !state.pinned) return {...state,hover:null,selectedDate:null};
    if (action.type === 'pin') return {...state,pinned:action.time,hover:action.time,selectedDate:action.date || null};
    if (action.type === 'date') return {...state,selectedDate:action.date};
    return state;
  }
  const api = {esc,date,finite,probability,ruleScore,rowAt,agentView,agentsHtml,referenceHtml,riskHtml,groupDates,popupHtml,historyPage,historyHtml,selectionTransition};
  root.KHQuantChartEvidence = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
