/* Display-only evidence: every score keeps its saved target and original date. */
(function (root) {
  'use strict';
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const date = (value) => String(value ?? '').slice(0,10);
  const finite = (value) => (typeof value === 'number' || typeof value === 'string' && value.trim() !== '') && Number.isFinite(Number(value));
  const num = (value, digits = 2) => finite(value) ? Number(value).toLocaleString('zh-CN',{maximumFractionDigits:digits}) : '—';
  const probability = (value) => finite(value) && Number(value) >= 0 && Number(value) <= 1 ? `${num(Number(value) * 100)}%` : '不可用';
  const ruleScore = (value) => finite(value) && Number(value) >= 0 && Number(value) <= 1 ? `${num(Number(value) * 100)} / 100` : '不可用';
  const supportScore = (value) => finite(value) ? num(Number(value) * 100) : '—';
  const consensusVersion = (row, config = null, version = '') => version || row?.version || config?.version || '';
  const weightedConsensus = (row, config = null, version = '') => consensusVersion(row,config,version) === 'czsc_agent_consensus_v2';
  const labels = {ma:'均线 ML',structure:'CZSC 结构',wyckoff:'威科夫量价',high_return:'高收益二阶段'};
  const reasonLabels = {no_candidate_signal:'本日没有新的买入候选',actual_holding_required:'需要兼容的真实持仓记录',high_return_independent_window_missing:'独立前向验证尚未完成',model_unavailable:'该时点没有可用模型',model_not_available_at_signal:'该模型在信号时点尚不可用',ma_model_not_available_at_signal:'均线模型在信号时点尚不可用',ma_model_veto:'均线分数未达到冻结门槛',model_probability_below_threshold:'模型分数未达到冻结门槛',threshold_met:'达到冻结门槛',calendar_gap:'交易日历存在行情缺口',no_saved_ma_probability:'该时点未保存均线入场评分',no_ma_entry_score:'该时点没有已保存的均线入场评分',no_candidate:'本日没有买入候选',missing_saved_score:'该时点没有已保存的评分',no_fresh_eligible_candidate:'本日没有新的合格候选',score_lineage_availability_unavailable:'缺少评分首次可用时间',score_not_observable:'该评分在此时点尚不可见',retrospective_score_rejected:'事后推理分数不参与历史仲裁',score_date_mismatch:'评分日期与所选日期不一致',score_label_overlap:'训练标签与判断日存在重叠',foreign_or_missing_event_symbol:'事件股票身份不匹配',event_not_eligible_or_visible:'候选未通过资格或尚不可见',event_input_unavailable:'候选输入不可用',event_identity_unavailable:'缺少原始事件身份',event_not_observed_today:'事件不属于当日可知证据',event_not_observable:'事件确认时间晚于判断时点',unknown_event_direction:'事件方向不可识别',actual_holding_required_for_independent_exit:'独立退出证据需要真实持仓',explicit_causal_rule_fallback:'该时点明确使用因果规则回退',ma_entry_score_unavailable:'该时点均线入场评分不可用',ma_entry_below_threshold:'均线入场分数未达到当前仲裁门槛',wyckoff_input_unavailable:'量价输入不可用',wyckoff_strength_unavailable:'量价规则强度不可用',wyckoff_strength_below_threshold:'量价规则强度未达到当前仲裁门槛',explicit_wyckoff_rule_support:'当日威科夫规则明确支持该方向',structure_input_unavailable:'结构输入不可用',opposing_events_within_source:'同一来源同时支持买入与卖出',eligible_causal_source_support:'当日合格证据支持该方向',no_buy_candidate:'本日没有新的买入候选',high_return_score_unavailable:'该时点高收益概率不可用',high_return_below_consensus_threshold:'高收益概率未达到当前质量闸门',high_return_quality_gate_passed:'通过本次高收益质量闸门',ma_assisted_structure_dependency_merged:'均线辅助定义的结构与均线合并为同一证据家族',entry_risk_plan_unavailable:'缺少当日入场风险计划',entry_risk_plan_invalid:'入场风险计划无效',entry_risk_plan_symbol_mismatch:'风险计划股票身份不匹配',entry_risk_plan_not_observable:'风险计划在此时点尚不可见',entry_risk_plan_date_mismatch:'风险计划日期不一致',entry_risk_gate_rejected:'未通过当日入场风险检查',exact_signal_day_reference_price_unavailable:'缺少信号当日有效收盘参考价',direction_evidence_expired:'原方向证据已到有效期，恢复中性',invalid_event:'原始事件格式无效',unknown_evidence_source:'证据来源未纳入仲裁',duplicate_event_identity:'重复事件只保留一次',opportunity_already_consumed:'该机会已产生最终点，不再重复使用',direction_conflict_or_insufficient_margin:'买卖支持相互冲突或方向优势不足',insufficient_independent_support:'独立证据家族支持未达到门槛',no_eligible_source_support:'本日没有合格来源支持方向',same_direction_already_active:'当前方向仍有效，不重复生成同向点',same_direction_cooldown:'同方向新机会仍处于冷却期',multi_agent_direction_confirmed:'多 Agent 已确认最终方向',buy_base_support_below_threshold:'买入基础支持分未达到门槛',buy_total_support_below_threshold:'买入总支持分未达到门槛',sell_base_support_below_threshold:'卖出基础支持分未达到门槛',insufficient_independent_families:'独立证据家族数量未达到门槛',opposing_sources_within_dependency_family:'同一依赖家族方向冲突，已弃权',source_conflict_propagated_to_dependency_family:'同一 Agent 的买卖冲突已传递到依赖家族，相关家族贡献归零',mixed_structure_budget_assigned_to_independent_evidence:'结构包含独立与均线辅助证据，结构权重只分配给独立证据'};
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
    const reconstructionText = identity.historical_reconstruction === true ? '历史重建评分 · 非当时实际部署预测' : '';
    return {reconstructionText,label:item.label || labels[key] || key,summary,kind:model ? '模型概率' : '规则证据',status:String(item.status || 'unavailable'),reason:reason(item.reason),target:item.target || identity.target || '',threshold:item.threshold,identityText,
      pointTypes:Array.isArray(item.point_types) ? item.point_types.join('、') : '',exit:key === 'ma' || key === 'high_return' ? `持仓退出概率 ${probability(item.exit_probability)}${finite(item.exit_probability) ? '' : ` · ${reason(item.exit_reason || item.exit_status || 'actual_holding_required')}`}` : ''};
  }
  function agentsHtml(row, compact = false, consensus = null, config = null, version = '') {
    return `<div class="agent-evidence-grid">${Object.keys(labels).map((key) => {
      const view = agentView(row?.agents?.[key],key), opinion = consensus?.agents?.[key];
      return `<section class="agent-evidence"><div class="agent-heading"><strong>${esc(view.label)}</strong><span>${esc(view.kind)} · ${esc(statusLabel(view.status))}</span></div><p class="agent-score">${esc(view.summary)}</p>${opinion ? opinionHtml(opinion,key,{decision:consensus,config,version}) : ''}<p class="subtle">${esc(view.reason)}</p>${!compact ? `${view.pointTypes ? `<p>${esc(view.pointTypes)}</p>` : ''}${view.exit ? `<p class="subtle">${esc(view.exit)}</p>` : ''}<details><summary>评分目标与模型身份</summary>${view.target ? `<p>目标：${esc(view.target)}</p>` : ''}${finite(view.threshold) ? `<p>冻结门槛 ${esc(probability(view.threshold))}</p>` : ''}${view.reconstructionText ? `<p>${esc(view.reconstructionText)}</p>` : ''}${view.identityText ? `<p>${esc(view.identityText)}</p>` : ''}</details>` : ''}</section>`;
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
  const judgmentLabel = (value) => ({BUY:'支持买入',SELL:'支持卖出',WAIT:'观察',HOLD:'观察',UNAVAILABLE:'不可用',ABSTAIN:'弃权 · 无有效意见',CONFLICT:'同源冲突',buy:'支持买入',sell:'支持卖出',wait:'观察',unavailable:'不可用'})[value] || String(value || '不可用');
  function opinionReasons(item) {
    const values = item?.reason_codes || item?.reasons || (item?.reason ? [item.reason] : []);
    return (Array.isArray(values) ? values : [values]).map(reason).join('；');
  }
  function opinionHtml(item, key, context = {}) {
    const gate = key === 'high_return', weighted = weightedConsensus(context.decision,context.config,context.version);
    const gateStatus = ({passed:'通过',rejected:'未通过',unavailable:'不可用',no_candidate:'没有买入候选'})[item.status] || statusLabel(item.status);
    const label = gate ? `质量闸门：${gateStatus}` : `仲裁意见：${judgmentLabel(item.judgment || item.judgement || item.opinion || item.action)}`;
    const contribution = context.decision?.contributions?.[key] || item;
    const weight = finite(item.configured_weight) ? item.configured_weight : gate ? context.config?.high_return_quality_weight : context.config?.source_weights?.[key];
    const contributionText = weighted ? gate
      ? `质量权重 ${supportScore(weight)} 分 · 买入质量加分 ${supportScore(contribution.quality_contribution)} 分 · 不计独立证据数量`
      : `基础权重 ${supportScore(weight)} 分 · 买入贡献 ${supportScore(contribution.buy_contribution)} 分 · 卖出贡献 ${supportScore(contribution.sell_contribution)} 分` : '';
    return `<p class="agent-opinion"><strong>${esc(label)}</strong>${opinionReasons(item) ? `<br>${esc(opinionReasons(item))}` : ''}</p>${contributionText ? `<p class="agent-contribution">${esc(contributionText)}</p>` : ''}${finite(item.threshold) ? `<p class="subtle">当前仲裁${gate ? '质量' : ''}门槛 ${esc(key === 'wyckoff' ? ruleScore(item.threshold) : probability(item.threshold))}</p>` : ''}${finite(item.model_threshold) ? `<p class="subtle">原模型冻结门槛 ${esc(probability(item.model_threshold))}</p>` : ''}${item.historical_reconstruction === true ? '<p class="subtle">历史重建意见 · 模型实际训练完成时间见身份详情</p>' : ''}${gate ? `<p class="subtle">候选质量闸门 · ${weighted ? '仅为已有买入候选加质量分' : '不计入支持票数'}${typeof item.required === 'boolean' ? ` · 本方案${item.required ? '必需' : '可选'}` : ''}</p>` : ''}`;
  }
  function consensusAt(consensus, selected) { return rowAt(consensus,selected); }
  function mergeEvidenceRows(rows, consensusRows) {
    const byDate = new Map((consensusRows || []).map((row) => [date(row.date),row]));
    return (rows || []).map((row) => ({...row,consensus:byDate.get(date(row.date)) || null}));
  }
  function finalSide(row) {
    if (!row || row.emitted !== true) return '';
    return row.action === 'BUY' ? 'B' : row.action === 'SELL' ? 'S' : '';
  }
  function consensusHtml(row, config = null, version = '') {
    if (!row) return '<section class="consensus-result"><strong>最终判断不可用</strong><p class="subtle">该结果没有已保存的多 Agent 仲裁，主图不显示最终买卖点。</p></section>';
    const side = finalSide(row), arbiter = row.agents?.arbiter, weighted = weightedConsensus(row,config,version);
    const label = side ? `${side} · 最终${side === 'B' ? '买点' : '卖点'}` : '观察 · 本日未生成最终买卖点';
    const familyLabels = {ma:'均线／辅助结构',structure:'独立结构',wyckoff:'威科夫量价'};
    const families = (names) => (Array.isArray(names) ? names : []).map((key) => familyLabels[key] || key).join('、');
    const familyText = finite(row.buy_independent_count) && finite(row.sell_independent_count) ? `独立证据家族：买 ${num(row.buy_independent_count,0)}${row.buy_independent_families?.length ? `（${families(row.buy_independent_families)}）` : ''} · 卖 ${num(row.sell_independent_count,0)}${row.sell_independent_families?.length ? `（${families(row.sell_independent_families)}）` : ''}` : '';
    const codes = opinionReasons(row), stateLabel = (value) => ({neutral:'中性',bullish:'看多',bearish:'看空',NEUTRAL:'中性',BULLISH:'看多',BEARISH:'看空'})[value] || String(value || '—');
    const direction = row.state_before || row.state_after ? `方向 ${stateLabel(row.state_before)} → ${stateLabel(row.state_after)}` : '';
    let scores, thresholds = '';
    if (weighted) {
      const weights = config?.source_weights || {}, baseMaximum = Object.keys(weights).length && Object.values(weights).every(finite) ? Object.values(weights).reduce((total,value) => total + Number(value),0) : null;
      const qualityMaximum = config?.high_return_quality_weight, totalMaximum = finite(baseMaximum) && finite(qualityMaximum) ? Number(baseMaximum) + Number(qualityMaximum) : null;
      const difference = finite(row.buy_base_score) && finite(row.sell_base_score) ? Number(row.buy_base_score) - Number(row.sell_base_score) : null;
      const margin = finite(difference) ? `${Number(difference) > 0 ? '+' : ''}${supportScore(difference)}` : '—';
      scores = `买入基础 ${supportScore(row.buy_base_score)} / ${supportScore(baseMaximum)} 分 ＋ 高收益质量 ${supportScore(row.buy_quality_bonus)} / ${supportScore(qualityMaximum)} 分 ＝ 买入总分 ${supportScore(row.buy_total_score)} / ${supportScore(totalMaximum)} 分<br>卖出基础 ${supportScore(row.sell_base_score)} / ${supportScore(baseMaximum)} 分 · 基础方向差额（买−卖）${esc(margin)} 分`;
      thresholds = config ? `<p class="subtle">本次门槛：买入基础 ${supportScore(config.min_buy_base_support)} 分 · 买入总分 ${supportScore(config.min_buy_support)} 分 · 卖出基础 ${supportScore(config.min_sell_support)} 分 · 基础方向优势 ${supportScore(config.min_direction_margin)} 分 · 至少 ${num(config.min_independent_families,0)} 组独立证据</p>` : '';
    } else {
      scores = `加权支持：买 ${num(row.buy_support)} · 卖 ${num(row.sell_support)}`;
      thresholds = config ? `<p class="subtle">本次门槛：买入支持 ${num(config.min_buy_support)} · 卖出支持 ${num(config.min_sell_support)} · 方向优势 ${num(config.min_direction_margin)}</p>` : '';
    }
    const familyDetails = weighted && row.family_details && typeof row.family_details === 'object' ? Object.entries(row.family_details).map(([key,item]) => `<li>${esc(familyLabels[key] || key)}：${esc(judgmentLabel(item.side))} · 实际贡献 ${supportScore(item.weight)} 分${Array.isArray(item.sources) && item.sources.length ? ` · 来源 ${esc(item.sources.map((source) => labels[source] || source).join('、'))}` : ''}${item.conflict === true ? ' · 同家族方向冲突，贡献归零' : ''}</li>`).join('') : '';
    return `<section class="consensus-result"><strong class="${side === 'B' ? 'valuation-gain' : side === 'S' ? 'valuation-loss' : ''}">${esc(label)}</strong>${codes ? `<p>${esc(codes)}</p>` : ''}${arbiter ? `<p class="subtle">仲裁 Agent：${esc(judgmentLabel(arbiter.judgment || arbiter.judgement || arbiter.opinion || arbiter.action))}${opinionReasons(arbiter) ? ` · ${esc(opinionReasons(arbiter))}` : ''}</p>` : ''}<p class="subtle">${esc(familyText)}${familyText ? '<br>' : ''}${scores}${direction ? `<br>${esc(direction)}` : ''}</p>${weighted ? '<p class="subtle">仲裁 v2 · 支持分，非获利概率；高收益只加买入质量分，不参与卖出或基础方向差额。</p>' : '<p class="subtle">仲裁 v1 · 按保存的加权支持票数判断。</p>'}${thresholds}${familyDetails ? `<details><summary>独立家族与合并贡献</summary><ul>${familyDetails}</ul></details>` : ''}</section>`;
  }
  function rawEventsHtml(row, entries = []) {
    const raw = Array.isArray(row?.events) ? row.events : entries.map((entry) => ({...entry.row,original_event_date:entry.row?.original_event_date || entry.event_date,excluded:entry.excluded,exclusion_reasons:entry.exclusion_reasons}));
    return `<details class="evidence-original"><summary>原始候选证据 ${raw.length} 条${raw.some((event) => event.excluded || event.candidate_eligible === false || event.evidence_visible === false) ? '（含未通过资格的证据）' : ''}</summary>${raw.map((event) => `<p>${esc(event.source || '规则')} · ${esc(event.event_type || event.event || event.reason || '')}<br>发生 ${esc(event.original_event_date || date(event.date || event.time))} · 可用 ${esc(event.available_at || event.evidence_date || event.signal_at || '—')}${event.excluded || event.candidate_eligible === false || event.evidence_visible === false ? `<br>未计入最终判断：${esc((event.exclusion_reasons || event.reason_codes || []).map(reason).join('、') || '未通过输入资格')}` : ''}${event.price != null ? `<br>账本成交价 ${num(event.price)} 元` : ''}</p>`).join('')}</details>`;
  }
  function popupHtml({evidence,consensus,groups = [],barDate,frequency,selectedDate,profileAsOf}) {
    const dates = groupDates(groups), displayedDates = groupDates(groups.map((group) => ({...group,entries:(group.entries || []).filter((entry) => entry.displayed !== false)})));
    const selected = dates.includes(selectedDate) ? selectedDate : displayedDates.at(-1) || date(barDate);
    const row = rowAt(evidence,selected), decision = consensusAt(consensus,selected), visible = groups.filter((group) => group.visible !== false);
    const badges = visible.map((group) => `<span class="badge ${group.action === 'BUY' ? 'buy' : 'sell'}">${group.kind === 'execution' ? '实际成交 ' : ''}${group.action === 'BUY' ? 'B' : 'S'}</span>`).join('');
    const entries = groups.flatMap((group) => group.entries || []).filter((entry) => date(entry.row?.evidence_date || entry.event_date || entry.row?.date || entry.row?.time || entry.original_date) === selected);
    return `<header class="chart-popup-heading"><div><strong>${esc(frequency)} ${esc(barDate)}</strong><p>${badges || '最终判断'} · 点击查看</p></div><button type="button" data-chart-evidence-close aria-label="关闭图表判断">×</button></header>${dates.length > 1 || dates.length && frequency !== '日线' ? `<div class="evidence-date-list" aria-label="周期内最终买卖点日期">${dates.map((item) => `<button type="button" data-evidence-date="${esc(item)}" aria-pressed="${item === selected}">${esc(item)}</button>`).join('')}</div><p class="subtle">按最终信号可知日期查看当日判断；点击日期同步对应日线筹码。</p>` : ''}${frequency !== '日线' && !selectedDate ? `<p class="subtle">筹码按所指周期截至 ${esc(profileAsOf || barDate)}；以下判断保留信号可知日。</p>` : ''}<p class="evidence-original-date">信号可知日 ${esc(selected)}${groups.some((group) => group.closed_at) ? ` · 周期闭合 ${esc(groups.find((group) => group.closed_at)?.closed_at)}` : ''}</p>${consensusHtml(decision,consensus?.config,consensus?.version)}${referenceHtml(row)}${agentsHtml(row,false,decision,consensus?.config,consensus?.version)}${riskHtml(row?.raw_risk_plan)}<div class="evidence-actions"><button type="button" data-evidence-locate="${esc(selected)}">定位原始证据</button><button type="button" data-evidence-replay="${esc(selected)}">按此日重新判断</button></div>${rawEventsHtml(row,entries)}`;
  }
  function historyPage(rows, requestedPage = 0, pageSize = 20) {
    const sorted = [...(rows || [])].sort((a,b) => date(b.date).localeCompare(date(a.date))), pages = Math.max(1,Math.ceil(sorted.length / pageSize));
    const page = Math.max(0,Math.min(pages - 1,Number(requestedPage) || 0));
    return {page,pages,total:sorted.length,rows:sorted.slice(page * pageSize,(page + 1) * pageSize)};
  }
  function historyHtml(rows, page = 0, selectedDate = '') {
    const result = historyPage(rows,page);
    const body = result.rows.map((row) => {
      const intents = finalSide(row.consensus) || '—', decisionReason = row.consensus ? opinionReasons(row.consensus) || '本日未生成最终买卖点' : '仲裁不可用';
      return `<tr${row.date === selectedDate ? ' class="history-selected"' : ''}><td><button type="button" data-history-date="${esc(row.date)}">${esc(row.date)}</button></td><td>${num(row.reference_close)}</td><td>${esc(intents)}</td><td>${esc(decisionReason)}</td><td>${probability(row.agents?.ma?.probability)}</td><td>${num(row.agents?.structure?.buy_count,0)} / ${num(row.agents?.structure?.sell_count,0)}</td><td>需求 ${ruleScore(row.agents?.wyckoff?.buy_score)} · 供给 ${ruleScore(row.agents?.wyckoff?.sell_score)}</td><td>${probability(row.agents?.high_return?.probability)}</td></tr>`;
    }).join('');
    return `<div class="table-wrap"><table class="history-evidence-table"><thead><tr><th>日期 · 点击回放</th><th>收盘参考价</th><th>最终 B / S</th><th>判断原因</th><th>均线入场概率</th><th>结构买 / 卖事件</th><th>量价需求 / 供给（规则分）</th><th>高收益概率</th></tr></thead><tbody>${body || '<tr><td colspan="8">尚无历史参考记录。</td></tr>'}</tbody></table></div><div class="history-pagination"><button type="button" data-history-page="${result.page - 1}"${result.page === 0 ? ' disabled' : ''}>上一页</button><span>${result.page + 1} / ${result.pages} · ${result.total} 个交易日 · 每页 20 行</span><button type="button" data-history-page="${result.page + 1}"${result.page + 1 >= result.pages ? ' disabled' : ''}>下一页</button></div>`;
  }
  function selectionTransition(current, action) {
    const state = {...{hover:null,pinned:null,selectedDate:null,popupOpen:false},...current};
    if (action.type === 'clear' || action.type === 'loading') return {hover:null,pinned:null,selectedDate:null,popupOpen:false};
    if (action.type === 'hover' && !state.pinned) return {...state,hover:action.time || null,selectedDate:null};
    if (action.type === 'leave' && !state.pinned) return {...state,hover:null,selectedDate:null};
    if (action.type === 'pin') return {...state,pinned:action.time,hover:action.time,selectedDate:action.date || null,popupOpen:action.popupOpen === true};
    if (action.type === 'date') return {...state,selectedDate:action.date};
    return state;
  }
  const api = {esc,date,finite,probability,ruleScore,supportScore,consensusVersion,weightedConsensus,rowAt,agentView,agentsHtml,judgmentLabel,opinionReasons,opinionHtml,consensusAt,mergeEvidenceRows,finalSide,consensusHtml,rawEventsHtml,referenceHtml,riskHtml,groupDates,popupHtml,historyPage,historyHtml,selectionTransition};
  root.KHQuantChartEvidence = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
