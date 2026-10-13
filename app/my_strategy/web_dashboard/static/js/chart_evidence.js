/* Display-only evidence: every score keeps its saved target and original date. */
(function (root) {
  'use strict';
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const date = (value) => String(value ?? '').slice(0,10);
  const finite = (value) => (typeof value === 'number' || typeof value === 'string' && value.trim() !== '') && Number.isFinite(Number(value));
  const validProbability = (value) => finite(value) && Number(value) >= 0 && Number(value) <= 1;
  const num = (value, digits = 2) => finite(value) ? Number(value).toLocaleString('zh-CN',{maximumFractionDigits:digits}) : '—';
  const probability = (value) => finite(value) && Number(value) >= 0 && Number(value) <= 1 ? `${num(Number(value) * 100)}%` : '不可用';
  const ruleScore = (value) => finite(value) && Number(value) >= 0 && Number(value) <= 1 ? `${num(Number(value) * 100)} / 100` : '不可用';
  const supportScore = (value) => finite(value) ? num(Number(value) * 100) : '—';
  const consensusVersion = (row, config = null, version = '') => version || row?.version || config?.version || '';
  const weightedConsensus = (row, config = null, version = '') => consensusVersion(row,config,version) === 'trend_consensus';
  const currentEvidenceConsensus = (row, config = null, version = '') => consensusVersion(row,config,version) === 'trend_consensus';
  const isTrendConsensus = (row, config = null, version = '') => consensusVersion(row,config,version) === 'trend_consensus';
  function trendProbabilities(item = {}) {
    const present = item.kind === 'trend_probability' || ['p_up','p_down','p_neutral'].some((key) => Object.hasOwn(item,key));
    const available = present && ['p_up','p_down','p_neutral'].every((key) => validProbability(item[key])) && Math.abs(Number(item.p_up)+Number(item.p_down)+Number(item.p_neutral)-1)<1e-5;
    return {present,available,summary:available ? `上涨 ${probability(item.p_up)} · 下跌 ${probability(item.p_down)} · 中性 ${probability(item.p_neutral)}` : '趋势概率未评分'};
  }
  function accountMarkerInputs(result, symbol = result?.symbol) {
    const account = result?.account_actions, context = account?.context;
    const series = account?.series_context_hash || context?.series_context_hash;
    if (!context?.account_id || !series) return {events:[],trades:[],valid:false,rejectedRecords:0,context};
    let rejectedRecords = 0;
    const bound = (items) => (items || []).filter((item) => {
      const cutoff = date(result.as_of || result.data_end);
      const knownDate = [item.execution_at,item.first_available_at,item.available_at,item.signal_at,item.date].map(date).filter(Boolean).sort().at(-1);
      const valid = item.symbol === symbol && item.account_id === context.account_id && item.series_context_hash === series && (!cutoff || knownDate <= cutoff);
      if (!valid) rejectedRecords++;
      return valid;
    });
    const events = bound(account.intents || account.final_action_intents).map((item) => ({...item,
      event_id:item.event_id || item.intent_id,source:'account_action',source_frequency:item.source_frequency || '日线',
      evidence_date:date(item.evidence_date || item.original_signal_at || item.signal_at || item.date),
      first_available_at:item.first_available_at || item.available_at || item.signal_at}));
    const trades = bound(account.trades || account.ledger);
    return {events,trades,valid:true,rejectedRecords,context,series_context_hash:series};
  }
  function accountEvidence(result) {
    const evidence = result?.chart_evidence;
    if (!result?.trend_consensus) return evidence;
    const selected = accountMarkerInputs(result), account = result.account_actions;
    const bound = (item) => selected.valid && item.symbol === result.symbol && item.account_id === selected.context.account_id && item.series_context_hash === selected.series_context_hash && (!result.as_of || date(item.available_at || item.date) <= date(result.as_of));
    const actions = new Map((account?.rows || []).filter(bound).map((item) => [date(item.date),item]));
    const holdings = new Map((account?.holding_snapshots || []).filter(bound).map((item) => [date(item.date),item]));
    const rejects = (account?.rejections || []).filter(bound);
    return {...evidence,rows:(evidence?.rows || []).map((row) => ({...row,
      account_action:actions.get(row.date) || (bound(row.account_action || {}) ? row.account_action : null),
      account_context:selected.context,holding:holdings.get(row.date) || (bound(row.holding || row.holding_snapshot || {}) ? row.holding || row.holding_snapshot : null),
      broker_fills:selected.trades.filter((item) => date(item.date) === row.date || date(item.signal_date) === row.date),
      rejections:rejects.filter((item) => date(item.date || item.last_attempt_at) === row.date)}))};
  }
  const labels = {ma:'均线 ML',structure:'CZSC 结构',wyckoff:'威科夫量价',high_return:'高收益二阶段'};
  function highReturnTargetText(target) {
    const net = finite(target?.min_net_return) ? target.min_net_return : .08;
    const risk = finite(target?.min_r_multiple) ? target.min_r_multiple : 2;
    return `完整交易费用后收益至少${num(net * 100)}%，且≥${num(risk)}倍冻结止损参考风险`;
  }
  const accountContextText = (context) => context ? `${context.account_kind === 'real_broker' ? '真实账户' : '模拟研究账户'}${context.account_start ? ` · 起点 ${date(context.account_start)}` : ''}${finite(context.initial_cash) ? ` · 初始资金 ${num(context.initial_cash)} 元` : ''}` : '';
  const reasonLabels = {no_candidate_signal:'本日没有新的买入候选',actual_holding_required:'需要兼容的真实持仓记录',high_return_independent_window_missing:'独立前向验证尚未完成',model_unavailable:'该时点没有可用模型',model_not_available_at_signal:'该模型在信号时点尚不可用',ma_model_not_available_at_signal:'均线模型在信号时点尚不可用',ma_model_veto:'均线分数未达到冻结门槛',model_probability_below_threshold:'模型分数未达到冻结门槛',threshold_met:'达到冻结门槛',calendar_gap:'交易日历存在行情缺口',no_saved_ma_probability:'该时点未保存均线入场评分',no_ma_entry_score:'该时点没有已保存的均线入场评分',no_candidate:'本日没有买入候选',missing_saved_score:'该时点没有已保存的评分',no_fresh_eligible_candidate:'本日没有新的合格候选',score_lineage_availability_unavailable:'缺少评分首次可用时间',score_not_observable:'该评分在此时点尚不可见',retrospective_score_rejected:'事后推理分数不参与历史仲裁',score_date_mismatch:'评分日期与所选日期不一致',score_label_overlap:'训练标签与判断日存在重叠',foreign_or_missing_event_symbol:'事件股票身份不匹配',event_not_eligible_or_visible:'候选未通过资格或尚不可见',event_input_unavailable:'候选输入不可用',event_identity_unavailable:'缺少原始事件身份',event_not_observed_today:'事件不属于当日可知证据',event_not_observable:'事件确认时间晚于判断时点',unknown_event_direction:'事件方向不可识别',actual_holding_required_for_independent_exit:'独立退出证据需要真实持仓',explicit_causal_rule_fallback:'该时点明确使用因果规则回退',ma_entry_score_unavailable:'该时点均线入场评分不可用',ma_entry_below_threshold:'均线入场分数未达到当前仲裁门槛',wyckoff_input_unavailable:'量价输入不可用',wyckoff_strength_unavailable:'量价规则强度不可用',wyckoff_strength_below_threshold:'量价规则强度未达到当前仲裁门槛',explicit_wyckoff_rule_support:'当日威科夫规则明确支持该方向',structure_input_unavailable:'结构输入不可用',opposing_events_within_source:'同一来源同时支持买入与卖出',eligible_causal_source_support:'当日合格证据支持该方向',no_buy_candidate:'本日没有新的买入候选',high_return_score_unavailable:'该时点高收益概率不可用',high_return_below_consensus_threshold:'高收益概率未达到当前质量闸门',high_return_quality_gate_passed:'通过本次高收益质量闸门',ma_assisted_structure_dependency_merged:'均线辅助定义的结构与均线合并为同一证据家族',entry_risk_plan_unavailable:'缺少当日入场风险计划',entry_risk_plan_invalid:'入场风险计划无效',entry_risk_plan_symbol_mismatch:'风险计划股票身份不匹配',entry_risk_plan_not_observable:'风险计划在此时点尚不可见',entry_risk_plan_date_mismatch:'风险计划日期不一致',entry_risk_gate_rejected:'未通过当日入场风险检查',exact_signal_day_reference_price_unavailable:'缺少信号当日有效收盘参考价',direction_evidence_expired:'原方向证据已到有效期，恢复中性',invalid_event:'原始事件格式无效',unknown_evidence_source:'证据来源未纳入仲裁',duplicate_event_identity:'重复事件只保留一次',opportunity_already_consumed:'该机会已产生最终点，不再重复使用',direction_conflict_or_insufficient_margin:'买卖支持相互冲突或方向优势不足',insufficient_independent_support:'独立证据家族支持未达到门槛',no_eligible_source_support:'本日没有合格来源支持方向',same_direction_already_active:'当前方向仍有效，不重复生成同向点',same_direction_cooldown:'同方向新机会仍处于冷却期',multi_agent_direction_confirmed:'多 Agent 已确认最终方向',buy_base_support_below_threshold:'买入基础支持分未达到门槛',buy_total_support_below_threshold:'买入总支持分未达到门槛',sell_base_support_below_threshold:'卖出基础支持分未达到门槛',insufficient_independent_families:'独立证据家族数量未达到门槛',opposing_sources_within_dependency_family:'同一依赖家族方向冲突，已弃权',source_conflict_propagated_to_dependency_family:'同一 Agent 的买卖冲突已传递到依赖家族，相关家族贡献归零',mixed_structure_budget_assigned_to_independent_evidence:'结构包含独立与均线辅助证据，结构权重只分配给独立证据'};
  Object.assign(reasonLabels,{no_fresh_candidate_trigger:'本日没有新的候选触发，活跃证据不会自行生成买卖点',evidence_state_unavailable:'当日状态证据不可用',evidence_state_date_mismatch:'状态证据日期与判断日不一致',evidence_state_not_observable:'当日状态证据尚不可知',active_evidence_expired:'历史方向证据已过有效期',current_state_invalidated:'当日状态已使历史方向证据失效',ma_full_identity_unavailable:'均线评分缺少完整模型身份',ma_causal_trend_not_supported:'当日因果均线趋势不支持该方向',wyckoff_current_direction_not_supported:'当日量价状态不支持该方向',independent_opportunity_cooldown:'独立新机会仍处于冷却期'});
  Object.assign(reasonLabels,{candidate_evidence_contract_unavailable:'缺少候选评估契约',evidence_state_input_unavailable:'当日状态输入未通过',evidence_snapshot_identity_unavailable:'缺少当日证据快照身份',current_state_unavailable:'当日状态未保存',current_state_price_mismatch:'状态参考价与判断日行情不一致',current_state_not_observable:'状态依赖在判断时点尚不可见',active_evidence_expired_or_invalid:'原始证据已过期或失效',ma_frozen_model_contract_unverified:'均线模型冻结契约未核验',ma_candidate_risk_contract_mismatch:'均线模型候选或风险契约不匹配',high_return_candidate_risk_contract_mismatch:'高收益模型候选或风险契约不匹配',high_return_frozen_model_contract_unverified:'高收益模型冻结契约未核验',high_return_score_date_mismatch:'高收益评分日期与判断日不一致',ma_causal_sell_origin_unavailable:'缺少仍有效的均线趋势失效证据',structure_current_support_invalid:'结构支撑或阻力当前无效',structure_current_price_invalidated:'当日价格已突破结构失效边界',wyckoff_strength_below_threshold_or_opposing:'量价强度不足或反向强度占优',currently_revalidated_candidate_support:'已按当日状态复核候选支持',no_current_direction_support:'当日状态不支持方向',no_fresh_qualified_opportunity:'本日没有新的合格机会',fresh_candidate_direction_mismatch:'新候选方向与仲裁支持方向不一致',active_evidence_permanently_invalidated:'原始证据已被当日状态永久失效',invalid_invalidation_reference:'失效记录未指向对应来源的原始事件',ma_current_input_unavailable:'当日均线输入不可用',ma_current_complete_lineage_unavailable:'当日均线完整评分身份不可用',ma_current_entry_probability_veto:'当日均线入场评分未过门槛',structure_current_input_or_confirmation_unavailable:'当日结构输入或确认时间不可用',structure_current_support_invalidated:'原结构支撑已失效',wyckoff_current_metadata_or_input_unavailable:'当日量价时点或输入不可用',wyckoff_current_strength_or_range_invalidated:'原量价方向或价格区间已失效',risk_inputs_unavailable:'风险输入不可用',invalid_causal_stop:'因果止损价无效',stop_distance_exceeds_frozen_maximum:'止损距离超过冻结上限'});
  Object.assign(reasonLabels,{candidate_risk_ineligible:'候选未通过风险检查，未评估高收益概率',risk_blocked:'候选被风险阻止，未评估',input_ineligible:'评分输入未通过资格检查',model_unavailable:'该时点缺少兼容模型',identity_error:'已有评分的模型身份核验失败',ma_candidate_out_of_scope:'该候选不在均线模型已验证的适用范围',candidate_envelope_mismatch:'候选人口与模型训练范围不一致'});
  Object.assign(reasonLabels,{risk_blocked_unscored:'风险阻止候选，高收益未评估',native_evidence_contract_unverified:'独立来源的原生证据契约未核验',original_pattern_identity_unavailable:'缺少原始确认模式身份',origin_event_not_observable:'原始确认模式在该时点尚不可知',foreign_support_fact:'持续支持包含其他股票或其他来源的事实',invalid_support_fact:'持续支持事实格式不合格',support_origin_not_observable:'持续支持的原始事件尚不可知',native_structure_input_ineligible:'纯结构依赖行情未通过输入资格',ma_candidate_population_out_of_scope:'均线模型尚未覆盖本次候选人口',ma_candidate_envelope_mismatch:'均线模型的候选人口版本不匹配',ma_candidate_envelope_event_mismatch:'均线评分对应的候选身份与当日新机会不匹配',ma_causal_long_trend_not_supported:'均线长期趋势当前不支持入场',high_return_candidate_population_out_of_scope:'高收益模型尚未覆盖本次候选人口',close_broke_frozen_structure_boundary:'收盘价破坏原结构冻结边界',new_confirmed_opposite_pattern:'新确认反向结构使原支持失效',close_broke_frozen_wyckoff_boundary:'收盘价破坏该量价事件的原失效边界',pending_confirmation_expired:'量价事件超过确认等待期',input_quality_gap:'输入质量缺口使原支持失效'});
  Object.assign(reasonLabels,{model_unavailable_at_signal:'信号时点缺少兼容模型',source_candidate_ineligible:'原始候选未通过入场资格'});
  Object.assign(reasonLabels,{risk_atr_unavailable:'入场风险需要的历史波动数据不可用',risk_stop_unavailable_or_broken:'入场风险止损缺失或已被当前价格破坏',risk_distance_below_frozen_minimum:'止损距离低于冻结最小风险距离'});
  Object.assign(reasonLabels,{ma_tactical_weakening_is_not_confirmed_long_trend_failure:'短期转弱，长期趋势失效尚未确认',wyckoff_buy_event_and_continuous_support_below_threshold:'已确认需求事件与当日需求支持均未达门槛',wyckoff_sell_event_and_continuous_support_below_threshold:'已确认供给事件与当日供给支持均未达门槛',wyckoff_current_continuous_support:'当日连续量价证据支持该方向',wyckoff_valid_confirmed_event_support:'原已确认量价事件按专属边界复核仍有效'});
  Object.assign(reasonLabels,{ma_current_long_trend_failure_rule_support:'当日均线长期趋势失效规则支持卖出方向'});
  Object.assign(reasonLabels,{trend_input_ineligible:'趋势输入未通过核验',trend_model_unavailable:'缺少兼容趋势模型',trend_probability_simplex_invalid:'三分类趋势概率核验失败',model_contract_unverified:'模型契约未通过核验',model_head_mismatch:'模型评分头不匹配',model_target_contract_missing:'缺少模型目标契约',model_actual_training_after_signal:'模型实际训练晚于信号时点',native_input_ineligible:'原生规则输入未通过核验',native_identity_missing:'缺少原生规则身份',native_snapshot_date_mismatch:'规则快照日期不匹配',native_snapshot_not_observable:'规则快照在此时点尚不可知',native_snapshot_price_mismatch:'规则快照价格不匹配',native_continuous_strength_missing:'连续规则支持未提供',same_source_explicit_conflict:'同一来源存在明确方向冲突',high_return_population_out_of_scope:'高收益评分不适用于当前候选',foreign_or_unknown_candidate:'候选股票或来源不匹配',candidate_not_confirmed_or_observable:'候选未确认或尚不可知',candidate_origin_identity_missing:'缺少候选原始身份',no_live_unconsumed_opportunity:'当前没有仍有效且未使用的新机会',opportunity_cooldown:'新机会尚在冻结间隔内',risk_plan_unavailable:'风险计划不可用',high_return_or_total_support_not_qualified:'高收益质量或总支持分未达门槛',hard_stop:'冻结持仓止损',max_holding_sessions:'达到冻结最长持仓期限',continuous_consensus_buy:'趋势共识买入资格通过',continuous_consensus_sell:'连续卖方共识退出',structure_invalidation:'持仓冻结结构边界失效',research_exit_model:'持仓退出模型触发'});
  Object.assign(reasonLabels,{actual_account_stop:'研究账户冻结止损',ordinary_consensus_sell:'连续卖方共识退出',signal_pinned_ma_exit:'入场时固定的持仓退出模型触发',protective_gap_stop:'开盘跳空触发冻结保护退出',forced_next_open_exit_branch:'冻结研究分支强制下一开盘退出',actual_existing_position:'已有持仓，入场质量不适用',no_qualified_direction_candidate:'本日没有合格方向候选',missing_target_market_session:'缺少应有核验交易日行情'});
  Object.assign(reasonLabels,{no_prior_trend_head:'该时点没有更早可用的趋势模型',no_prior_high_return_head:'该时点没有更早可用的高收益模型',high_return_below_threshold:'高收益评分未达到冻结门槛',E0_E1_do_not_read_high_return:'方向判断不读取二阶段高收益评分',high_return_filter_disabled:'质量过滤已关闭，本次未评分'});
  Object.assign(reasonLabels,{holding_confirmed_boundary:'入场计划冻结的确认边界失效',risk_plan_identity_or_time_mismatch:'入场风险计划身份或时点不匹配',risk_plan_unavailable_at_signal:'信号时点没有可用风险计划',risk_plan_frozen_fields_missing:'风险计划缺少冻结字段',risk_plan_outside_frozen_contract:'风险计划参数超出冻结契约'});
  Object.assign(reasonLabels,{risk_stop_distance_outside_contract:'止损距离不符合冻结风险契约'});
  const reason = (value) => String(value || '该时点没有可用评分').split(/[；;]/).map((part) => reasonLabels[part] || part).join('；');
  const statusLabel = (value) => ({available:'可用',unavailable:'不可用',shadow:'影子观察',no_candidate:'没有候选',candidate:'候选',qualified:'资格通过',rules_only:'规则判断',rule_evidence:'规则证据',input_veto:'输入未通过',actual_holding_required:'需要真实持仓',scored:'已评分',score_unavailable:'评分不可用',rule_triggered:'规则触发',rule_fallback:'规则回退',no_signal:'未触发',input_unavailable:'输入未通过',blocked:'已阻止',active:'已生成',current_supported:'当日复核支持',current_conflict:'当日方向冲突',not_supported:'当日未支持',supported:'当日支持',model_unavailable:'缺兼容模型',input_ineligible:'输入不合格',out_of_scope:'范围外研究',identity_error:'评分身份核验失败',risk_blocked_unscored:'风险阻止 · 未评估'})[value] || String(value || '不可用');
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
    const savedReason = item.reason || (Array.isArray(item.reason_codes) ? item.reason_codes.join('；') : item.reason_codes);
    const unavailable = model ? !validProbability(item.probability) : item.input_eligible === false || ['unavailable','input_unavailable','input_ineligible'].includes(item.status);
    return {reconstructionText,label:item.label || labels[key] || key,summary,kind:model ? '模型概率' : '规则证据',status:String(item.status || 'unavailable'),reason:savedReason ? reason(savedReason) : unavailable ? reason(null) : '',target:item.target || identity.target || '',threshold:item.threshold,identityText,
      pointTypes:Array.isArray(item.point_types) ? item.point_types.join('、') : '',exit:key === 'ma' || key === 'high_return' ? `持仓退出概率 ${probability(item.exit_probability)}${finite(item.exit_probability) ? '' : ` · ${reason(item.exit_reason || item.exit_status || 'actual_holding_required')}`}` : ''};
  }
  function riskDetails(row) {
    const saved = row?.candidate_evidence?.risk_details;
    const risk = new Map((Array.isArray(saved) ? saved : []).map((detail) => [detail.event_id,detail]));
    const riskCodes = new Set(['risk_inputs_unavailable','invalid_causal_stop','stop_distance_exceeds_frozen_maximum','risk_atr_unavailable','risk_stop_unavailable_or_broken','risk_distance_below_frozen_minimum']);
    for (const fact of candidateFacts(row) || []) {
      if (risk.has(fact.event_id)) continue;
      const codes = (fact.risk_reason_codes || fact.reason_codes || []).filter((code) => riskCodes.has(code));
      risk.set(fact.event_id,{event_id:fact.event_id,candidate_eligible:fact.candidate_eligible,
        ...(Object.hasOwn(fact,'risk_candidate_eligible') ? {candidate_eligible:fact.risk_candidate_eligible} : {}),
        risk_reason_codes:codes,reference_price:fact.reference_price,stop_price:fact.stop_price,
        stop_distance:fact.stop_distance});
    }
    return risk;
  }
  function candidateFacts(row) {
    return Array.isArray(row?.candidate_evidence) ? row.candidate_evidence : row?.candidate_evidence?.facts;
  }
  function sourceOpinionText(item) {
    if (item?.judgment || item?.judgement) return judgmentLabel(item.judgment || item.judgement);
    if (finite(item?.buy_strength) && finite(item?.sell_strength)) return `连续支持 买 ${ruleScore(item.buy_strength)} · 卖 ${ruleScore(item.sell_strength)}`;
    return statusLabel(item?.status || 'unavailable');
  }
  function structureFacts(row) {
    const saved = candidateFacts(row);
    const facts = Array.isArray(saved) ? saved : [];
    const own = facts.filter((event) => event.source === 'structure' && (event.direction_eligible === true || event.direction_eligible == null && event.candidate_direction_valid === true));
    const side = (event) => String(event.side || event.action || '').toUpperCase();
    const risk = riskDetails(row);
    if (!Array.isArray(saved)) return {available:false,buy:row?.agents?.structure?.buy_count,sell:row?.agents?.structure?.sell_count,qualifiedBuy:null,riskRejected:null};
    return {available:true,buy:own.filter((event) => side(event) === 'BUY').length,sell:own.filter((event) => side(event) === 'SELL').length,
      qualifiedBuy:own.filter((event) => side(event) === 'BUY' && (Object.hasOwn(event,'risk_candidate_eligible') ? event.risk_candidate_eligible : event.candidate_eligible) === true).length,
      riskRejected:own.filter((event) => side(event) === 'BUY' && risk.get(event.event_id)?.risk_reason_codes?.length).length};
  }
  const evaluationLabels = {no_candidate:'无候选 · 未评估',risk_blocked:'风险阻止 · 未评估',input_ineligible:'输入不合格 · 未评估',model_unavailable:'缺兼容模型 · 未评估',scored_rejected:'已评分 · 未过门槛',scored_passed:'已评分 · 通过门槛',identity_error:'评分身份核验失败',actual_existing_position:'持仓中 · 入场不适用',entry_quality_gate_disabled:'质量过滤已关闭，本次未评分'};
  const evaluationExplanations = {
    no_candidate:'高收益入场评分只评估该研究账户当日合格的候选；本日没有待评估候选，未评分不代表0%概率。',
    risk_blocked:'候选先被冻结风险检查阻止，因此未进行高收益入场评分；风险原因保留在当日入场计划中。',
    input_ineligible:'候选或评分输入未通过资格检查，本日没有可用于入场质量判断的概率。',
    model_unavailable:'该信号时点没有兼容且可用的高收益检查点；需要质量过滤时不会自动放行。',
    scored_rejected:'本日已有高收益评分，但未达到当日冻结门槛。',
    scored_passed:'高收益评分达到冻结门槛；最终账户操作仍由方向、风险与账户约束共同决定。',
    identity_error:'评分身份或时点核验未通过，已有原始分数不能作为有效入场概率。',
    actual_existing_position:'该研究账户已持仓，高收益入场评分不适用；持仓退出按该账户独立退出证据判断。',
    entry_quality_gate_disabled:'本次关闭了高收益质量过滤，因此未进行这项评分。'
  };
  function highReturnState(item = {}, opinion = null) {
    const explicit = item.evaluation_status || opinion?.evaluation_status;
    if (Object.hasOwn(evaluationLabels,explicit)) return ['scored_passed','scored_rejected'].includes(explicit) && !validProbability(item.probability) ? 'identity_error' : explicit;
    const codes = [item.status,item.reason,...(item.reason_codes || [])].filter(Boolean).join(';');
    if (/entry_quality_gate_disabled|high_return_filter_disabled/.test(codes)) return 'entry_quality_gate_disabled';
    if (/actual_existing_position/.test(codes)) return 'actual_existing_position';
    if (/candidate_risk_ineligible|risk_blocked|entry_risk_gate_rejected/.test(codes)) return 'risk_blocked';
    if (/no_candidate|no_qualified_direction_candidate|no_live_unconsumed_opportunity/.test(codes)) return 'no_candidate';
    if (/input_ineligible|input_veto|feature_ineligible/.test(codes)) return 'input_ineligible';
    if (/identity_error|contract_mismatch|contract_unverified|score_date_mismatch/.test(codes)) return 'identity_error';
    if (/model_unavailable|model_not_available|model_incompatible|no_prior_high_return_head/.test(codes)) return 'model_unavailable';
    if (validProbability(item.probability)) {
      const threshold = finite(opinion?.threshold) ? Number(opinion.threshold) : finite(item.threshold) ? Number(item.threshold) : null;
      if (threshold != null) return Number(item.probability) >= threshold ? 'scored_passed' : 'scored_rejected';
      if (opinion?.status === 'passed') return 'scored_passed';
      if (opinion?.status === 'rejected') return 'scored_rejected';
    }
    return '';
  }
  function currentAgentView(row, key, opinion = null) {
    const item = row?.agents?.[key] || {}, view = agentView(item,key);
    if (key === 'structure') {
      const facts = structureFacts(row);
      if (facts.available) {
        view.summary = `当日方向事实：买 ${facts.buy} · 卖 ${facts.sell} · 原始入场资格通过 ${facts.qualifiedBuy} 条买入候选`;
        view.reason = `${facts.riskRejected ? `其中 ${facts.riskRejected} 条买入候选未通过风险检查；` : '原始方向与入场风险资格分别核验；'}当日结构意见 ${sourceOpinionText(opinion)} · 当前有效历史支持 ${opinion?.origin_event_ids?.length || 0} 条。`;
      }
      const state = opinion?.current_state || item.current_state;
      if (state && typeof state === 'object') {
        const patternLabels = Object.values(state.native_labels || {}).filter((label) => label !== '其他');
        view.secondary = `当前支撑 ${num(state.support_floor)} 元 · 当前阻力 ${num(state.resistance_ceiling)} 元${state.current_valid_zone ? ` · 已确认中枢 ${num(state.current_valid_zone.support_low)}–${num(state.current_valid_zone.resistance_high)} 元` : ''}${patternLabels.length ? ` · 纯结构 ${[...new Set(patternLabels)].join('、')}` : ''}`;
      }
    } else if (key === 'wyckoff') {
      if (finite(item.current_demand) && finite(item.current_supply)) {
        view.summary = `当日需求 ${ruleScore(item.current_demand)} · 当日供给 ${ruleScore(item.current_supply)}`;
        view.secondary = `事件衰减强度：需求 ${ruleScore(item.event_demand ?? item.buy_score)} · 供给 ${ruleScore(item.event_supply ?? item.sell_score)}`;
        view.status = item.input_eligible === false || opinion?.status === 'input_ineligible' ? 'input_unavailable' :
          opinion?.status === 'unavailable' ? 'unavailable' : ['BUY','SELL'].includes(opinion?.judgment) ? 'current_supported' :
          opinion?.judgment === 'CONFLICT' ? 'current_conflict' : 'available';
      } else view.summary = `需求 ${ruleScore(item.buy_score)} · 供给 ${ruleScore(item.sell_score)}（事件衰减强度）`;
      if (item.input_eligible === false || opinion?.status === 'input_ineligible') view.status = 'input_unavailable';
      if (Object.hasOwn(item,'effective_demand') || Object.hasOwn(item,'effective_supply')) {
        view.secondary = `事件需求 ${ruleScore(item.event_demand)} · 事件供给 ${ruleScore(item.event_supply)}；最终有效需求 ${ruleScore(item.effective_demand)} · 最终有效供给 ${ruleScore(item.effective_supply)}`;
        const basis = {confirmed_event:'有效已确认事件',continuous:'当日连续支持',current_snapshot:'当日连续支持',mixed:'连续与事件支持',conflict:'同源冲突弃权'};
        view.reason = `${view.reason}${view.reason ? '；' : ''}贡献依据：${basis[item.support_basis] || item.support_basis || '已保存的最终有效支持'}`;
      }
      const eventDemand = Object.hasOwn(item,'event_demand') ? item.event_demand : opinion?.event_demand;
      const eventSupply = Object.hasOwn(item,'event_supply') ? item.event_supply : opinion?.event_supply;
      view.explanation = item.input_eligible === false || opinion?.status === 'input_ineligible'
        ? '量价输入未通过核验，缺失或无效的评分保持不可用。'
        : finite(eventDemand) && finite(eventSupply) && Number(eventDemand) === 0 && Number(eventSupply) === 0
          ? '当时有效的事件强度两侧均为0；当日连续需求与供给由价量单独计算，仍可为非零。待测试、强度到期或失效的事件不等于有效已确认支持，具体状态可在原始事件明细核对。'
          : '事件强度反映当时可知且仍有效的量价事件；连续支持由当日价量单独计算。最终有效支持与实际加权贡献以已保存判断为准，规则分不是盈利概率。';
    } else if (key === 'high_return') {
      const status = highReturnState(item,opinion);
      view.evaluationStatus = status;
      view.evaluationLabel = evaluationLabels[status] || statusLabel(item.status);
      view.status = status || view.status;
      view.summary = ['scored_passed','scored_rejected'].includes(status) && validProbability(item.probability) ? `高收益区间概率 ${probability(item.probability)} · ${view.evaluationLabel}` : view.evaluationLabel;
      view.explanation = evaluationExplanations[status] || '该日未提供完整高收益评估状态；概率不可用与0%概率分别展示。';
      if (trendProbabilities(row?.agents?.ma).present || isTrendConsensus(row?.consensus)) {
        if (view.target === 'actual_complete_fee_net_roundtrip_high_return_and_initial_risk') view.target = highReturnTargetText(item.label_target);
        view.pricingNote = '成交采用滑点模拟均价；R按冻结止损参考假设估计。';
      }
    } else if (key === 'ma') {
      const trend = trendProbabilities(item);
      if (trend.present) {
        view.summary = trend.summary;
        view.kind = '三分类趋势概率';
        view.reason = trend.available ? (item.reason_codes || []).map(reason).join('；') : reason(item.reason || (item.reason_codes || []).join('；') || 'model_unavailable');
        view.target = item.target || '未来核验交易日价格方向：上涨／下跌／中性';
        view.trend = trend;
        const exit = row?.account_action?.exit_evaluation;
        const holding = row?.holding || row?.account_action?.holding_snapshot;
        const shares = holding?.shares ?? row?.account_action?.shares;
        view.exit = shares > 0 ? exit?.holding_contract_compatible === true && validProbability(exit?.probability)
          ? `该研究账户持仓退出概率 ${probability(exit.probability)}` : `该研究账户退出未评估${exit?.reason_codes?.length ? ` · ${exit.reason_codes.map(reason).join('；')}` : ''}` : finite(shares) && Number(shares) === 0 ? '该研究账户空仓 · 持仓退出不适用' : '该研究账户持仓状态未提供 · 退出未评估';
        return view;
      }
      if (item.in_scope === false || opinion?.in_scope === false) view.summary = `范围外研究分 ${probability(item.probability)} · 不计入已验证入场质量`;
      if (opinion?.trend_state) view.secondary = `长期趋势 ${({bullish:'看多',bearish:'看空',neutral:'中性'})[opinion.trend_state] || opinion.trend_state} · 短期转弱 ${opinion.tactical_weakening === true ? '是' : opinion.tactical_weakening === false ? '否' : '未评估'}`;
    }
    return view;
  }
  function agentDetailsHtml(row, key, view, opinion = null) {
    const item = row?.agents?.[key] || {}, state = opinion?.current_state || item.current_state || {};
    const model = !['structure','wyckoff'].includes(key) || view.kind === '模型概率';
    const savedText = (value) => value && typeof value === 'object' ? JSON.stringify(value) : String(value ?? '');
    let paragraphs = [];
    if (model) {
      const targetLabels = {
        next_verified_open_horizon_direction:'从下一核验开盘起，冻结交易日窗口内的上涨／下跌／中性方向',
        actual_complete_fee_net_roundtrip_positive:'完整交易费用后获利概率',
        actual_complete_fee_net_roundtrip_high_return_and_initial_risk:highReturnTargetText(item.label_target)
      };
      paragraphs.push(view.target ? `目标：${Object.hasOwn(targetLabels,view.target) ? targetLabels[view.target] : savedText(view.target)}` : '评分目标未保存，无法核对该分数的预测对象。');
      if (finite(view.threshold)) paragraphs.push(`冻结门槛 ${probability(view.threshold)}`);
      if (view.reconstructionText) paragraphs.push(view.reconstructionText);
      const identity = item.model_identity || opinion?.model_identity || {};
      const noHead = identity.verified_frozen_contract === false && !identity.checkpoint && !identity.manifest_sha256 && !identity.model_head && !identity.head;
      const fields = [['model_run_id','运行'],['checkpoint','检查点'],['model_head','评分头'],['available_at','可用'],
        ['train_label_end','训练标签截止'],['actual_training_completed_at','训练完成'],
        ['target_contract_hash','目标契约 SHA256'],['manifest_sha256','模型清单 SHA256'],
        ['feature_schema_hash','输入 schema SHA256'],['source_snapshot_hash','来源快照 SHA256']];
      const values = fields.map(([field,label]) => [label,field === 'model_run_id' && noHead ? null : field === 'model_head' ? identity.model_head || identity.head : identity[field]]);
      const supplied = values.filter(([,value]) => value != null && value !== '');
      paragraphs.push(...supplied.map(([label,value]) => `${label}：${savedText(value)}`));
      const missing = values.filter(([,value]) => value == null || value === '').map(([label]) => label);
      if (missing.length) paragraphs.push(supplied.length ? `未提供：${missing.join('、')}；这些身份字段不能从概率反推。` : `本日未提供模型身份；${view.reason || view.evaluationLabel || '该日期未保存模型评分及谱系'}。`);
    } else {
      paragraphs.push(`规则口径：${key === 'structure' ? '当时已确认结构的支撑、阻力与买卖方向事实；入场风险资格另行核验。' : '当时因果量价事件与当日连续需求／供给强度；规则分不是盈利概率。'}`);
      const ruleIdentity = item.rule_identity || {};
      const ruleVersion = ruleIdentity.version || item.version;
      const snapshot = ruleIdentity.snapshot_id || item.snapshot_id || item.source_id;
      const observedAt = ruleIdentity.available_at || item.available_at || state.snapshot_at || item.snapshot_at;
      paragraphs.push(`评分日期：${date(item.date || row?.date) || '未提供'}`);
      paragraphs.push(ruleVersion ? `规则版本：${savedText(ruleVersion)}` : '规则版本未保存，无法核对该规则快照的完整版本。');
      paragraphs.push(snapshot ? `证据快照标识：${savedText(snapshot)}` : '证据快照标识未保存，不能补造模型身份。');
      paragraphs.push(observedAt ? `快照可用：${savedText(observedAt)}` : '快照可用时间未提供。');
      const start = ruleIdentity.dependency_start || item.dependency_start || state.dependency_start;
      const end = ruleIdentity.dependency_end || item.dependency_end || state.dependency_end;
      paragraphs.push(start && end ? `行情依赖区间：${date(start)}～${date(end)}` : '行情依赖区间未随本日规则证据提供。');
      if (item.input_eligible === true) paragraphs.push('本日规则输入核验：通过。');
      else if (item.input_eligible === false) paragraphs.push(`本日规则输入核验：未通过；${view.reason || '具体资格原因未提供'}。`);
      else paragraphs.push('本日规则输入核验状态未提供。');
    }
    return `<details open data-agent-details="${esc(key)}"><summary>${model ? '评分目标与模型身份' : '规则口径与数据身份'}</summary>${paragraphs.map((text) => `<p>${esc(text)}</p>`).join('')}</details>`;
  }
  function agentsHtml(row, compact = false, consensus = null, config = null, version = '') {
    return `<div class="agent-evidence-grid">${Object.keys(labels).map((key) => {
      const opinion = consensus?.agents?.[key];
      const legacyStructure = key === 'structure' && consensusVersion(consensus,config,version) && !currentEvidenceConsensus(consensus,config,version);
      const view = legacyStructure ? agentView(row?.agents?.[key],key) : currentAgentView(row,key,opinion);
      if (key === 'structure' && structureFacts(row).available && currentEvidenceConsensus(consensus,config,version)) {
        const facts = structureFacts(row), judgment = opinion?.judgment;
        view.summary = `当日方向事实：买 ${facts.buy} · 卖 ${facts.sell} · 原始入场资格通过 ${facts.qualifiedBuy} 条买入候选`;
        view.status = ['BUY','SELL'].includes(judgment) ? 'current_supported' : judgment === 'CONFLICT' ? 'current_conflict' : !judgment && opinion?.status === 'scored' ? 'scored' : opinion?.status === 'unavailable' ? 'unavailable' : 'not_supported';
        view.reason = `${facts.riskRejected ? `当日方向事实含 ${facts.riskRejected} 条风险未通过的买入候选；` : '方向事实与原始入场资格分别展示；'}当日结构意见 ${sourceOpinionText(opinion)} · 已按当日状态复核 ${opinion?.origin_event_ids?.length || 0} 条原始来源（含有效历史证据）。新入场仍需通过风险与高收益闸门。`;
      }
      return `<section class="agent-evidence"><div class="agent-heading"><strong>${esc(view.label)}</strong><span>${esc(view.kind)} · ${esc(view.evaluationLabel || statusLabel(view.status))}</span></div><p class="agent-score">${esc(view.summary)}</p>${view.secondary ? `<p class="subtle">${esc(view.secondary)}</p>` : ''}${opinion ? opinionHtml(opinion,key,{decision:consensus,config,version}) : ''}<p class="subtle">${esc(view.reason)}</p>${view.explanation ? `<p class="subtle">${esc(view.explanation)}</p>` : ''}${view.pricingNote ? `<p class="subtle">${esc(view.pricingNote)}</p>` : ''}${!compact ? `${view.pointTypes ? `<p>${esc(view.pointTypes)}</p>` : ''}${view.exit ? `<p class="subtle">${esc(view.exit)}</p>` : ''}${agentDetailsHtml(row,key,view,opinion)}` : ''}</section>`;
    }).join('')}</div>`;
  }
  function agentOverviewHtml(row, consensus = null, config = null) {
    const cellStyle = 'padding:6px;font-size:11px;line-height:1.4;overflow-wrap:anywhere';
    const body = Object.keys(labels).map((key) => {
      const item = row?.agents?.[key] || {}, opinion = consensus?.agents?.[key];
      const view = currentAgentView(row,key,opinion);
      const contribution = consensus?.contributions?.[key] || opinion || {};
      const weight = finite(opinion?.configured_weight) ? opinion.configured_weight : key === 'high_return' ? (config?.high_return_quality_weight ?? config?.quality_weight) : config?.source_weights?.[key];
      let score = view.summary, detail = '';
      if (key === 'ma') {
        const trend = trendProbabilities(item);
        score = trend.present ? trend.summary : validProbability(item.probability) ? `已评分 ${probability(item.probability)}` : `未评分 · ${statusLabel(view.status)}`;
        detail = item.in_scope === false || opinion?.in_scope === false ? '范围外研究分 · 不参与当前投票' : opinion ? `${sourceOpinionText(opinion)}${finite(opinion.threshold) ? ` · 门槛 ${probability(opinion.threshold)}` : ''}` : '';
      } else if (key === 'structure') {
        const state = opinion?.current_state || item.current_state || {};
        const effectiveBuy = finite(opinion?.buy_strength) ? opinion.buy_strength : state.buy_score, effectiveSell = finite(opinion?.sell_strength) ? opinion.sell_strength : state.sell_score;
        score = opinion?.judgment ? judgmentLabel(opinion.judgment) : statusLabel(opinion?.status);
        detail = item.input_eligible === false || opinion?.input_eligible === false ? '规则输入未通过' : `规则买 ${ruleScore(effectiveBuy)} · 卖 ${ruleScore(effectiveSell)}`;
      } else if (key === 'wyckoff') {
        detail = Object.hasOwn(item,'effective_demand') ? `${view.secondary} · ${opinion ? sourceOpinionText(opinion) : statusLabel(view.status)}` : opinion ? sourceOpinionText(opinion) : statusLabel(view.status);
      } else if (key === 'high_return') {
        score = validProbability(item.probability) ? probability(item.probability) : '未评分';
        detail = view.evaluationLabel;
      }
      const buy = key === 'high_return' ? contribution.quality_contribution : contribution.buy_contribution;
      const actual = `买 ${supportScore(buy)} / 卖 ${supportScore(contribution.sell_contribution)} 分`;
      return `<tr data-agent-summary-row="${key}"><td style="${cellStyle}">${esc(view.label)}</td><td style="${cellStyle}"><strong>${esc(score)}</strong>${detail ? `<br><span class="subtle">${esc(detail)}</span>` : ''}</td><td style="${cellStyle}">${finite(weight) ? `${supportScore(weight)}%` : '—'}</td><td style="${cellStyle}">${esc(actual)}${key === 'high_return' ? '<br><span class="subtle">买入质量加分</span>' : ''}</td></tr>`;
    }).join('');
    const target = row?.agents?.ma?.target || row?.agents?.ma?.model_identity?.target;
    const targetNote = target === 'actual_complete_fee_net_roundtrip_positive' ? '<p class="subtle" style="margin:6px 0">MA 目标：完整交易费用后获利概率。</p>' : '';
    return `<section class="agent-score-overview" aria-label="四路评分总览"><strong>四路评分总览</strong><div class="table-wrap" style="margin-top:5px"><table style="table-layout:fixed"><colgroup><col style="width:20%"><col style="width:40%"><col style="width:10%"><col style="width:30%"></colgroup><thead><tr>${['Agent','当前分值 / 状态','权重','买 / 卖实际贡献'].map((label) => `<th style="${cellStyle}">${esc(label)}</th>`).join('')}</tr></thead><tbody>${body}</tbody></table></div>${targetNote}</section>`;
  }
  function referenceHtml(row) {
    if (!row) return '<p class="evidence-unavailable">该日期未取得参考价与评分证据。</p>';
    const ohlc = row.ohlc || {}, priceBasis = {unadjusted:'未复权',mixed_legacy_adjustment_unverified:'旧复权口径未核验'}[row.reference_price_basis] || row.reference_price_basis || '价格口径待核验';
    return `<div class="evidence-price"><strong>收盘参考价 ${num(row.reference_close)} 元</strong><span>${esc(row.reference_price_date || row.date)} · ${esc(priceBasis)} · 非成交价</span></div><p class="evidence-ohlc">开 ${num(ohlc.open)} · 高 ${num(ohlc.high)} · 低 ${num(ohlc.low)} · 收 ${num(ohlc.close)}</p><p class="subtle">下一核验交易日 ${esc(row.next_market_session || '等待核验')}</p>`;
  }
  function accountActionHtml(row) {
    const action = row?.account_action;
    if (!action) return '';
    const side = finalSide(action), label = side === 'S' ? 'S · 账户退出意图' : side === 'B' ? 'B · 账户买入意图' : '账户观察';
    const holding = row.holding || row.holding_snapshot || action.holding_snapshot || action.holding_before;
    const shares = holding?.shares ?? action.shares;
    const fills = (row.broker_fills || row.trades || []).map((fill) => `<p>执行 ${esc(date(fill.date || fill.execution_at))} · ${esc(fill.action)} · 成交 ${num(fill.price)} 元 · 信号 ${esc(date(fill.signal_date || fill.signal_at))}</p>`).join('');
    const rejections = (row.rejections || []).map((item) => `<p>拒单 ${esc(date(item.date || item.last_attempt_at))} · ${esc(item.reason || (item.reason_codes || []).map(reason).join('；'))}</p>`).join('');
    const origin = action.final_action_intents?.[0] || action;
    const exit = action.exit_evaluation;
    const exitText = holding?.shares > 0 && exit?.holding_contract_compatible === true && validProbability(exit.probability) ? `<p>持仓退出概率 ${probability(exit.probability)}</p>` : '';
    const reasons = opinionReasons(action) || (action.final_action_intents || []).map((intent) => reason(intent.reason || (intent.reason_codes || []).join('；'))).join('；');
    const contextText = accountContextText(row.account_context);
    const openRisk = /T09:30:/.test(String(origin.original_signal_at || origin.first_available_at || origin.signal_at || '')) || origin.reference_price_source === 'trigger_open' || [origin.reason,...(origin.reason_codes || []),...(action.reason_codes || [])].includes('gap_open_stop');
    const openNote = openRisk ? '<p class="subtle">开盘风控触发；以下 Agent 分数为当日收盘复核，不是该卖出的触发依据。</p>' : '';
    return `<section class="account-action-evidence"><strong>${esc(label)}</strong>${openNote}${contextText ? `<p class="subtle">${esc(contextText)} · 显示区间不会重置账户</p>` : ''}<p>${finite(shares) ? `持仓 ${num(shares,0)} 股` : '持仓状态未提供'} · ${esc(reasons)}</p><p class="subtle">${esc(action.account_id || '')} · 信号 ${esc(origin.original_signal_at || origin.signal_at || origin.date)}${origin.intended_session ? ` · 下一次执行核验 ${esc(origin.intended_session)}` : ''}</p>${exitText}${fills}${rejections}</section>`;
  }
  function riskHtml(plan) {
    if (!plan) return '<p class="subtle">该日没有可用的价格与风险计划。</p>';
    const canDerive = finite(plan.reference_price) && finite(plan.stop_price) && validProbability(plan.target_weight) && Number(plan.reference_price) > Number(plan.stop_price) && Number(plan.stop_price) > 0;
    const risk = finite(plan.risk_fraction) ? Number(plan.risk_fraction) : canDerive ? Number(plan.target_weight) * (Number(plan.reference_price) - Number(plan.stop_price)) / Number(plan.reference_price) : null;
    const riskLabel = finite(plan.risk_fraction) ? '风险预算' : '信号计划风险占比';
    return `<p class="subtle">价格与风险计划：${plan.entry_allowed === true ? '入场资格通过' : plan.entry_allowed === false ? '未通过入场资格' : '入场资格待核验'} · ${esc(statusLabel(plan.status))}${(plan.reason_codes || []).length ? ` · ${esc(plan.reason_codes.map(reason).join('；'))}` : ''}</p><div class="evidence-risk"><span>参考区间 ${num(plan.price_floor)}–${num(plan.price_ceiling)} 元</span><span>失效价 ${num(plan.stop_price)} 元</span><span>${riskLabel} ${finite(risk) ? probability(risk) : '不可用'}</span><span>目标仓位 ${finite(plan.target_weight) ? probability(plan.target_weight) : '不可用'}</span></div>`;
  }
  function groupDates(groups) {
    return [...new Set((groups || []).flatMap((group) => (group.entries || []).map((entry) => date(entry.row?.evidence_date || entry.event_date || entry.row?.date || entry.row?.time || entry.original_date))).filter(Boolean))].sort();
  }
  const judgmentLabel = (value) => ({BUY:'支持买入',SELL:'支持卖出',WAIT:'观察',HOLD:'观察',UNAVAILABLE:'不可用',ABSTAIN:'弃权 · 无有效意见',CONFLICT:'同源冲突',buy:'支持买入',sell:'支持卖出',wait:'观察',unavailable:'不可用'})[value] || String(value || '不可用');
  function opinionReasons(item, version = '') {
    const values = item?.reason_codes || item?.reasons || (item?.reason ? [item.reason] : []);
    const nativeCandidates = (version || item?.version) === 'trend_consensus';
    return (Array.isArray(values) ? values : [values]).map((code) => nativeCandidates && code === 'no_fresh_candidate_trigger'
      ? '本日没有取得当前方向资格的新候选，活跃证据不会自行生成买卖点' : reason(code)).join('；');
  }
  function opinionHtml(item, key, context = {}) {
    const gate = key === 'high_return', weighted = weightedConsensus(context.decision,context.config,context.version);
    const gateStatus = gate && isTrendConsensus(context.decision,context.config,context.version) ? evaluationLabels[highReturnState(item,item)] || statusLabel(item.status) : ({passed:'通过',rejected:'未通过',unavailable:'不可用',no_candidate:'没有买入候选'})[item.status] || statusLabel(item.status);
    const label = gate ? `质量闸门：${gateStatus}` : isTrendConsensus(context.decision,context.config,context.version) ? `仲裁意见：${sourceOpinionText(item)}` : `仲裁意见：${judgmentLabel(item.judgment || item.judgement || item.opinion || item.action)}`;
    const contribution = context.decision?.contributions?.[key] || item;
    const weight = finite(item.configured_weight) ? item.configured_weight : gate ? (context.config?.high_return_quality_weight ?? context.config?.quality_weight) : context.config?.source_weights?.[key];
    const contributionText = weighted ? gate
      ? `质量权重 ${supportScore(weight)} 分 · 买入质量加分 ${supportScore(contribution.quality_contribution)} 分 · 不计独立证据数量`
      : `基础权重 ${supportScore(weight)} 分 · 买入贡献 ${supportScore(contribution.buy_contribution)} 分 · 卖出贡献 ${supportScore(contribution.sell_contribution)} 分` : '';
    const evidenceMode = ({current_snapshot:'评估当日状态',fresh_event:'当日新事件',active_event:'仍有效的历史事件',mixed:'当日状态与有效事件'})[item.evidence_mode];
    const evidenceText = evidenceMode ? `${evidenceMode}${item.origin_event_ids?.length ? ` · 原始事件 ${item.origin_event_ids.length} 条` : ''}` : '';
    return `${evidenceText ? `<p class="subtle">${esc(evidenceText)}</p>` : ''}<p class="agent-opinion"><strong>${esc(label)}</strong>${opinionReasons(item) ? `<br>${esc(opinionReasons(item))}` : ''}</p>${contributionText ? `<p class="agent-contribution">${esc(contributionText)}</p>` : ''}${finite(item.threshold) ? `<p class="subtle">当前仲裁${gate ? '质量' : ''}门槛 ${esc(key === 'wyckoff' ? ruleScore(item.threshold) : probability(item.threshold))}</p>` : ''}${finite(item.model_threshold) ? `<p class="subtle">原模型冻结门槛 ${esc(probability(item.model_threshold))}</p>` : ''}${item.historical_reconstruction === true ? '<p class="subtle">历史重建意见 · 模型实际训练完成时间见身份详情</p>' : ''}${gate ? `<p class="subtle">候选质量闸门 · ${weighted ? '仅为已有买入候选加质量分' : '不计入支持票数'}${typeof item.required === 'boolean' ? ` · 本方案${item.required ? '必需' : '可选'}` : ''}</p>` : ''}`;
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
  function diagnosticsHtml(row) {
    const funnel = row?.funnel;
    if (!funnel || typeof funnel !== 'object') return '';
    const countStages = [['raw_events','原始事件'],['observable_events','当日可见'],['fresh_candidates','有效机会（当前资格通过）'],['active_evidence','活跃证据'],['directional_sources','有方向来源'],['independent_buy_families','买入独立家族'],['independent_sell_families','卖出独立家族']];
    const stages = countStages.filter(([key]) => finite(funnel[key])).map(([key,label]) => `${label} ${num(funnel[key],0)}`).join(' · ');
    const selling = row.proposed_action === 'SELL' || row.action === 'SELL';
    const gates = [['l1_qualified','方向与新机会'],['risk_eligible','风险资格'],['quality_passed','高收益质量'],['final_emitted','最终点']]
      .filter(([key]) => typeof funnel[key] === 'boolean' || finite(funnel[key]) && [0,1].includes(Number(funnel[key]))).map(([key,label]) => `${label}：${selling && ['risk_eligible','quality_passed'].includes(key) ? '卖出不适用' : funnel[key] === true || Number(funnel[key]) === 1 ? '通过' : '未通过'}`).join(' · ');
    const active = Array.isArray(row.active_evidence) ? row.active_evidence : [];
    const details = active.map((item) => `<li>${esc(labels[item.source] || item.source || '证据')} · ${esc(judgmentLabel(item.side))} · 首次可知 ${esc(item.origin_date || '—')} · 已经过 ${num(item.age_bars,0)} 根日线 · 剩余 ${num(item.remaining_bars,0)} 根（含当日）${item.revalidated === true ? ' · 已按当日状态重新核验' : ''}${item.event_id ? `<br>原始事件 ${esc(item.event_id)}` : ''}${item.opportunity_id ? `<br>机会 ${esc(item.opportunity_id)}` : ''}</li>`).join('');
    const invalidated = Array.isArray(row.invalidated_evidence) ? row.invalidated_evidence : [];
    const invalidationDetails = invalidated.map((item) => `<li>${esc(labels[item.source] || item.source || '证据')} · 原始可知日 ${esc(item.origin_date || '—')} · 失效 ${esc(item.invalidated_at || '—')}${item.event_id ? `<br>原始事件 ${esc(item.event_id)}` : ''}</li>`).join('');
    return `<section class="consensus-diagnostics"><strong>候选逐层判断</strong>${stages ? `<p class="subtle">${esc(stages)}</p>` : ''}${gates ? `<p class="subtle">${esc(gates)}</p>` : ''}<p class="subtle">原始机会事实、方向支持与最终交易资格分别核验；缺失分数不重新分配权重。</p>${details ? `<details><summary>当前有效证据 ${active.length} 条及有效期</summary><ul>${details}</ul></details>` : ''}${invalidationDetails ? `<details><summary>当日永久失效证据 ${invalidated.length} 条</summary><ul>${invalidationDetails}</ul></details>` : ''}</section>`;
  }
  function consensusHtml(row, config = null, version = '') {
    if (!row) return '<section class="consensus-result"><strong>最终判断不可用</strong><p class="subtle">该结果没有已保存的多 Agent 仲裁，主图不显示最终买卖点。</p></section>';
    const side = finalSide(row), arbiter = row.agents?.arbiter, weighted = weightedConsensus(row,config,version);
    const currentVersion = consensusVersion(row,config,version);
    if (currentVersion !== 'trend_consensus') return '<section class="consensus-result"><strong>该仲裁记录已退役</strong><p>历史身份不具备当前判断资格，请查看归档证据。</p></section>';
    const label = isTrendConsensus(row,config,version) ? `趋势共识策略 · 方向诊断：${side || judgmentLabel(row.action)}` : side ? `${side} · 最终${side === 'B' ? '买点' : '卖点'}` : '观察 · 本日未生成最终买卖点';
    const familyLabels = {ma:'均线／辅助结构',structure:'独立结构',wyckoff:'威科夫量价'};
    const families = (names) => (Array.isArray(names) ? names : []).map((key) => familyLabels[key] || key).join('、');
    const familyText = finite(row.buy_independent_count) && finite(row.sell_independent_count) ? `独立证据家族：买 ${num(row.buy_independent_count,0)}${row.buy_independent_families?.length ? `（${families(row.buy_independent_families)}）` : ''} · 卖 ${num(row.sell_independent_count,0)}${row.sell_independent_families?.length ? `（${families(row.sell_independent_families)}）` : ''}` : '';
    const codes = opinionReasons(row), stateLabel = (value) => ({neutral:'中性',bullish:'看多',bearish:'看空',NEUTRAL:'中性',BULLISH:'看多',BEARISH:'看空'})[value] || String(value || '—');
    const direction = row.state_before || row.state_after ? `方向 ${stateLabel(row.state_before)} → ${stateLabel(row.state_after)}` : '';
    let scores, thresholds = '';
    if (weighted) {
      const weights = config?.source_weights || {}, baseMaximum = Object.keys(weights).length && Object.values(weights).every(finite) ? Object.values(weights).reduce((total,value) => total + Number(value),0) : null;
      const qualityMaximum = (config?.high_return_quality_weight ?? config?.quality_weight), totalMaximum = finite(baseMaximum) && finite(qualityMaximum) ? Number(baseMaximum) + Number(qualityMaximum) : null;
      const difference = finite(row.buy_base_score) && finite(row.sell_base_score) ? Number(row.buy_base_score) - Number(row.sell_base_score) : null;
      const margin = finite(difference) ? `${Number(difference) > 0 ? '+' : ''}${supportScore(difference)}` : '—';
      scores = `买入基础 ${supportScore(row.buy_base_score)} / ${supportScore(baseMaximum)} 分 ＋ 高收益质量 ${supportScore(row.buy_quality_bonus)} / ${supportScore(qualityMaximum)} 分 ＝ 买入总分 ${supportScore(row.buy_total_score)} / ${supportScore(totalMaximum)} 分<br>卖出基础 ${supportScore(row.sell_base_score)} / ${supportScore(baseMaximum)} 分 · 基础方向差额（买−卖）${esc(margin)} 分`;
      thresholds = config ? `<p class="subtle">本次门槛：买入基础 ${supportScore(config.min_buy_base_support ?? config.buy_base_threshold)} 分 · 买入总分 ${supportScore(config.min_buy_support ?? config.buy_total_threshold)} 分 · 卖出基础 ${supportScore(config.min_sell_support ?? config.sell_threshold)} 分 · 基础方向优势 ${supportScore(config.min_direction_margin ?? config.margin)} 分 · 至少 ${num(config.min_independent_families ?? config.min_independent,0)} 组独立证据</p>` : '';
    } else {
      scores = `加权支持：买 ${num(row.buy_support)} · 卖 ${num(row.sell_support)}`;
      thresholds = config ? `<p class="subtle">本次门槛：买入支持 ${num(config.min_buy_support)} · 卖出支持 ${num(config.min_sell_support)} · 方向优势 ${num(config.min_direction_margin)}</p>` : '';
    }
    const familyDetails = weighted && row.family_details && typeof row.family_details === 'object' ? Object.entries(row.family_details).map(([key,item]) => `<li>${esc(familyLabels[key] || key)}：${esc(judgmentLabel(item.side))} · 实际贡献 ${supportScore(item.weight)} 分${Array.isArray(item.sources) && item.sources.length ? ` · 来源 ${esc(item.sources.map((source) => labels[source] || source).join('、'))}` : ''}${item.conflict === true ? ' · 同家族方向冲突，贡献归零' : ''}</li>`).join('') : '';
    return `<section class="consensus-result"><strong class="${side === 'B' ? 'valuation-gain' : side === 'S' ? 'valuation-loss' : ''}">${esc(label)}</strong>${codes ? `<p>${esc(codes)}</p>` : ''}${arbiter ? `<p class="subtle">仲裁 Agent：${esc(judgmentLabel(arbiter.judgment || arbiter.judgement || arbiter.opinion || arbiter.action))}${opinionReasons(arbiter,currentVersion) ? ` · ${esc(opinionReasons(arbiter,currentVersion))}` : ''}</p>` : ''}<p class="subtle">${esc(familyText)}${familyText ? '<br>' : ''}${scores}${direction ? `<br>${esc(direction)}` : ''}</p><p class="subtle">趋势共识策略 · 连续加权支持；方向诊断与账户操作分别记录。</p>${thresholds}${currentEvidenceConsensus(row,config,version) ? diagnosticsHtml(row) : ''}${familyDetails ? `<details><summary>独立家族与合并贡献</summary><ul>${familyDetails}</ul></details>` : ''}</section>`;
  }
  function rawEventsHtml(row, entries = [], version = '') {
    const candidateStateAssessment = currentEvidenceConsensus(null,null,version);
    const raw = Array.isArray(row?.events) ? row.events : entries.filter((entry) => entry.kind !== 'inspection').map((entry) => ({...entry.row,original_event_date:entry.row?.original_event_date || entry.event_date,excluded:entry.excluded,exclusion_reasons:entry.exclusion_reasons}));
    const facts = new Map((candidateFacts(row) || []).map((event) => [event.event_id,event]));
    const risks = riskDetails(row);
    const rows = raw.map((event) => {
      const excluded = event.excluded || event.candidate_eligible === false || event.evidence_visible === false;
      const fact = facts.get(event.event_id), risk = risks.get(event.event_id);
      const directional = candidateStateAssessment && fact ? `<br>原始方向证据：${fact.direction_eligible === true || fact.direction_eligible == null && fact.candidate_direction_valid === true ? '有效；实际贡献按当日状态复核' : '未通过资格或不可见'}` : '';
      const riskText = candidateStateAssessment && risk ? `<br>${risk.risk_reason_codes?.length ? `新入场风险拒绝：${esc(risk.risk_reason_codes.map(reason).join('、'))}` : `原始候选交易资格：${risk.candidate_eligible ? '通过' : '未通过'}`} · 参考 ${num(risk.reference_price)} 元 · 止损 ${num(risk.stop_price)} 元 · 距离 ${probability(risk.stop_distance)}` : '';
      return `<p>${esc(event.source || '规则')} · ${esc(event.event_type || event.event || event.reason || '')}<br>发生 ${esc(event.original_event_date || date(event.date || event.time))} · 可用 ${esc(event.available_at || event.evidence_date || event.signal_at || '—')}${directional}${riskText}${excluded && !(candidateStateAssessment && fact) ? `<br>${candidateStateAssessment ? '原始候选资格或可见性未通过（方向与最终交易资格分别核验）' : '未计入最终判断'}：${esc((event.exclusion_reasons || event.reason_codes || []).map(reason).join('、') || '未通过输入资格')}` : ''}${event.price != null ? `<br>账本成交价 ${num(event.price)} 元` : ''}</p>`;
    }).join('');
    return `<details class="evidence-original"><summary>原始候选证据 ${raw.length} 条${raw.some((event) => event.excluded || event.candidate_eligible === false || event.evidence_visible === false) ? '（含未通过资格的证据）' : ''}</summary>${rows}</details>`;
  }
  function popupSelection({evidence,consensus,groups = [],barDate,selectedDate}) {
    const dates = groupDates(groups), displayedDates = groupDates(groups.map((group) => ({...group,entries:(group.entries || []).filter((entry) => entry.displayed !== false)})));
    const accountDates = groupDates(groups.filter((group) => group.kind === 'intention' && group.sources?.includes('account_action')));
    const finalDates = accountDates.length ? accountDates : groupDates(groups.filter((group) => group.kind === 'intention' && group.sources?.includes('consensus')));
    const selected = dates.includes(selectedDate) ? selectedDate : finalDates.at(-1) || displayedDates.at(-1) || date(barDate);
    const entries = groups.flatMap((group) => group.entries || []).filter((entry) => date(entry.row?.evidence_date || entry.event_date || entry.row?.date || entry.row?.time || entry.original_date) === selected);
    return {dates,selected,entries,row:rowAt(evidence,selected),decision:consensusAt(consensus,selected)};
  }
  function referenceSelection(context) {
    if (context.groups?.length && context.groups.every((group) => group.kind === 'inspection')) return null;
    const {selected,entries,row,decision} = popupSelection(context);
    if (entries.length && entries.every((entry) => entry.row?.source === 'wyckoff' && (entry.row?.event || entry.row?.event_type))) return null;
    if (entries.length && entries.every((entry) => entry.kind === 'execution')) return null;
    const finalEntry = entries.find((entry) => entry.displayed !== false && entry.row?.source === 'account_action') || entries.find((entry) => entry.displayed !== false && entry.row?.source === 'consensus') || entries.find((entry) => entry.displayed !== false);
    const event = finalEntry?.row || {};
    const signalStamp = event.first_available_at || event.available_at || event.original_signal_at || event.signal_at;
    if (event.source === 'account_action' && /T(?:09|10|11|13|14):/.test(String(signalStamp || '')) && !finite(event.reference_price)) return null;
    const price = finite(event.reference_price) ? Number(event.reference_price) : finite(row?.reference_close) ? Number(row.reference_close) : null;
    const priceDate = date(event.reference_price_date || row?.reference_price_date || selected);
    if (price == null || price <= 0 || priceDate !== selected) return null;
    const anchorDates = Array.isArray(event.anchor_dates) ? event.anchor_dates : (row?.events || []).map((item) => item.original_event_date || date(item.time || item.date)).filter(Boolean);
    return {price,signal_date:selected,bar_date:date(context.barDate),action:event.action || decision?.action,
      available_at:event.first_available_at || event.available_at || finalEntry?.first_available_at || null,
      anchor_dates:[...new Set(anchorDates.map(date).filter(Boolean))].sort(),event_id:event.event_id || null};
  }
  function popupHtml({evidence,consensus,groups = [],barDate,frequency,selectedDate,profileAsOf}) {
    const context = {evidence,consensus,groups,barDate,selectedDate};
    const {dates,selected,row,decision,entries} = popupSelection(context), reference = referenceSelection(context);
    const inspection = groups.some((group) => group.kind === 'inspection');
    const visible = groups.filter((group) => group.visible !== false);
    const selectedGroup = groups.find((group) => group.entries?.some((entry) => entries.includes(entry)));
    const formingProjection = frequency !== '日线' && selectedGroup?.daily_confirmed && selectedGroup?.period_status === 'forming';
    const dailyLabel = selectedGroup?.kind === 'wyckoff' ? '日线事件已可知' : '日线已确认';
    const periodText = formingProjection ? ` · ${dailyLabel}／周期形成中` : frequency !== '日线' && selectedGroup?.daily_confirmed ? ` · ${dailyLabel}／周期闭合 ${selectedGroup.closed_at || '时间待核验'}` : groups.some((group) => group.closed_at) ? ` · 周期闭合 ${groups.find((group) => group.closed_at)?.closed_at}` : '';
    const anchors = reference?.anchor_dates?.length ? `<p class="subtle">原始结构／事件锚点 ${esc(reference.anchor_dates.join('、'))} · 锚点不替代信号可知日</p>` : '';
    const badges = visible.map((group) => {
      const amount = group.event_count || group.entries?.length || 1;
      const wyckoff = amount === 1 ? `W ${esc(group.event_type)}${group.confirmed ? '确认' : '待测试'}`
        : `量价${group.action === 'BUY' ? '需求' : '供给'} ×${amount} · 已确认 ${group.confirmed_count ?? group.entries?.filter((entry) => entry.row?.confirmed === true).length ?? 0}／待测试 ${group.pending_count ?? group.entries?.filter((entry) => entry.row?.confirmed !== true).length ?? 0}`;
      return `<span class="badge ${group.action === 'BUY' ? 'buy' : 'sell'}">${group.kind === 'wyckoff' ? wyckoff : `${group.kind === 'execution' ? '实际成交 ' : group.diagnostic ? '方向诊断 ' : ''}${group.action === 'BUY' ? 'B' : 'S'}`}</span>`;
    }).join('');
    // Show every constituent event of the displayed candle. The date selector
    // above still controls the independent daily score/account snapshot.
    const wyckoffEntries = visible.flatMap((group) => group.kind === 'wyckoff' ? group.entries || [] : [])
      .filter((entry) => entry.row?.source === 'wyckoff' && (entry.row?.event || entry.row?.event_type))
      .sort((a,b) => String(a.first_available_at || a.event_date).localeCompare(String(b.first_available_at || b.event_date)) || String(a.event_id).localeCompare(String(b.event_id)));
    const wyckoffHtml = wyckoffEntries.length ? `<section class="wyckoff-event-details"><strong>本根K线原始量价事件 ${wyckoffEntries.length} 条</strong>${wyckoffEntries.map((entry) => `<p>W ${esc(entry.row.event || entry.row.event_type)} · ${entry.row.side === 'SELL' ? '供给' : '需求'} · ${entry.row.confirmed === true ? '已确认' : '待测试'}<br>原观察 ${esc(entry.row.origin_observed_at || entry.row.observed_at || entry.original_date)} · 首次可用 ${esc(entry.first_available_at || entry.row.available_at)}${entry.row.event_id ? `<br>事件标识 ${esc(entry.row.event_id)}` : ''}${finite(entry.row.invalidation_price) ? `<br>原冻结失效边界 ${esc(num(entry.row.invalidation_price))} 元` : ''}${entry.row.reason_codes?.length ? `<br>${esc(entry.row.reason_codes.map(reason).join('；'))}` : ''}</p>`).join('')}<p class="subtle">● 需求事件 · ■ 供给事件；主图数字为同方向事件数量，包含已确认和待测试。量价事件与最终 B/S 分别记录；周期内日线证据已可知，形成中周/月线仍按其闭合规则参与原生判断。上方日期选择只切换当日评分与账户状态。</p></section>` : '';
    return `<header class="chart-popup-heading"><div><strong>${esc(frequency)} ${esc(barDate)}</strong><p>${badges || (inspection ? 'K线评分查看' : '最终判断')} · 点击查看</p></div><button type="button" data-chart-evidence-close aria-label="关闭图表判断">×</button></header><p class="evidence-original-date">${inspection ? '评分日期' : '信号可知日'} ${esc(selected)}${esc(periodText)}</p>${agentOverviewHtml(row,decision,consensus?.config)}${accountActionHtml(row)}${dates.length > 1 || dates.length && frequency !== '日线' ? `<div class="evidence-date-list" aria-label="${inspection ? '周期内评分日期' : '周期内信号与事件可知日期'}">${dates.map((item) => `<button type="button" data-evidence-date="${esc(item)}" aria-pressed="${item === selected}">${esc(item)}</button>`).join('')}</div><p class="subtle">${inspection ? '按核验交易日查看当日评分' : '按首次可知日期查看当日判断'}；点击日期同步对应日线筹码。</p>` : ''}${frequency !== '日线' && !selectedDate ? `<p class="subtle">筹码按所指周期截至 ${esc(profileAsOf || barDate)}；以下判断保留信号可知日。</p>` : ''}${reference?.available_at ? `<p class="subtle">信号首次可用 ${esc(reference.available_at)}</p>` : ''}${anchors}${wyckoffHtml}${consensusHtml(decision,consensus?.config,consensus?.version)}${referenceHtml(row)}${agentsHtml(row,false,decision,consensus?.config,consensus?.version)}${riskHtml(row?.raw_risk_plan)}<div class="evidence-actions"><button type="button" data-evidence-locate="${esc(selected)}">定位原始证据</button><button type="button" data-evidence-replay="${esc(selected)}">按此日重新判断</button></div>${rawEventsHtml(row,entries,consensus?.version || consensus?.config?.version)}`;
  }
  function historyPage(rows, requestedPage = 0, pageSize = 20) {
    const sorted = [...(rows || [])].sort((a,b) => date(b.date).localeCompare(date(a.date))), pages = Math.max(1,Math.ceil(sorted.length / pageSize));
    const page = Math.max(0,Math.min(pages - 1,Number(requestedPage) || 0));
    return {page,pages,total:sorted.length,rows:sorted.slice(page * pageSize,(page + 1) * pageSize)};
  }
  function statisticsHtml(statistics) {
    if (!statistics || typeof statistics !== 'object') return '';
    const fields = [['bars','日线'],['source_events','来源事实'],['wyckoff_events','完整量价事件'],['final_points','最终 B/S']];
    const sections = [['visible_window','显示窗口'],['full_prefix','完整预热前缀']].map(([key,label]) => {
      const item = statistics[key];
      if (!item || typeof item !== 'object') return '';
      const counts = fields.filter(([field]) => finite(item[field])).map(([field,name]) => `${name} ${num(item[field],0)}`);
      return counts.length ? `${label}：${counts.join(' · ')}` : '';
    }).filter(Boolean);
    return sections.length ? `<p class="evidence-statistics subtle">${esc(sections.join('；'))}</p>` : '';
  }
  function historyHtml(rows, page = 0, selectedDate = '', statistics = null) {
    const result = historyPage(rows,page);
    const accountMode = (rows || []).some((row) => Object.hasOwn(row,'account_action'));
    const trendMode = (rows || []).some((row) => trendProbabilities(row.agents?.ma).present);
    const body = result.rows.map((row) => {
      const action = accountMode ? row.account_action : row.consensus;
      const intents = finalSide(action) || '—', decisionReason = action ? opinionReasons(action) || (accountMode ? '本日无新账户操作' : '本日未生成最终买卖点') : accountMode ? '账户操作不可用' : '仲裁不可用';
      const facts = structureFacts(row), structure = row.consensus?.agents?.structure;
      const structureCell = facts.available ? `原始 ${num(facts.buy,0)} / ${num(facts.sell,0)}<br>风险通过买入 ${num(facts.qualifiedBuy,0)}<br>当前 ${esc(sourceOpinionText(structure))}<br>有效历史支持 ${num(structure?.origin_event_ids?.length || 0,0)}` : `${num(facts.buy,0)} / ${num(facts.sell,0)}`;
      const w = currentAgentView(row,'wyckoff',row.consensus?.agents?.wyckoff), l2 = currentAgentView(row,'high_return',row.consensus?.agents?.high_return);
      const l2Cell = validProbability(row.agents?.high_return?.probability) ? `${probability(row.agents.high_return.probability)}${l2.evaluationLabel ? `<br>${esc(l2.evaluationLabel)}` : ''}` : esc(l2.evaluationLabel || '不可用');
      const ma = row.agents?.ma || {}, maScope = ma.in_scope === false || row.consensus?.agents?.ma?.in_scope === false ? '<br>范围外研究分' : '';
      const shares = (row.holding || row.holding_snapshot)?.shares ?? row.account_action?.shares;
      const accountCells = accountMode ? `<td>${finite(shares) ? `${num(shares,0)} 股` : '未提供'}</td><td>${esc((row.broker_fills || []).map((fill) => `${date(fill.date)} ${fill.action} ${num(fill.price)}元`).join('；') || '无成交')}</td><td>${esc((row.rejections || []).map((item) => `${date(item.date || item.last_attempt_at)} ${item.reason || (item.reason_codes || []).map(reason).join('；')}`).join('；') || '无拒单')}</td>` : '';
      const maCell = trendMode ? esc(trendProbabilities(ma).summary) : `${probability(ma.probability)}${maScope}`;
      return `<tr${row.date === selectedDate ? ' class="history-selected"' : ''}><td><button type="button" data-history-date="${esc(row.date)}">${esc(row.date)}</button></td><td>${num(row.reference_close)}</td><td>${esc(intents)}</td><td>${esc(decisionReason)}</td><td>${maCell}</td><td>${structureCell}</td><td>${esc(w.summary)}${w.secondary ? `<br>${esc(w.secondary)}` : ''}</td><td data-evaluation-status="${esc(l2.evaluationStatus || '')}">${l2Cell}</td>${accountCells}</tr>`;
    }).join('');
    return `${statisticsHtml(statistics)}<div class="table-wrap"><table class="history-evidence-table"><thead><tr><th>日期 · 点击回放</th><th>收盘参考价</th><th>${accountMode ? '账户最终 B / S' : '最终 B / S'}</th><th>${accountMode ? '操作／退出原因' : '判断原因'}</th><th>${trendMode ? '均线趋势概率（涨／跌／中性）' : '均线入场概率'}</th><th>结构原始买 / 卖与当前意见</th><th>量价需求 / 供给（规则分）</th><th>高收益概率与评估状态</th>${accountMode ? '<th>持仓</th><th>成交执行记录</th><th>拒单记录</th>' : ''}</tr></thead><tbody>${body || `<tr><td colspan="${accountMode ? 11 : 8}">尚无历史参考记录。</td></tr>`}</tbody></table></div><div class="history-pagination"><button type="button" data-history-page="${result.page - 1}"${result.page === 0 ? ' disabled' : ''}>上一页</button><span>${result.page + 1} / ${result.pages} · ${result.total} 个交易日 · 每页 20 行</span><button type="button" data-history-page="${result.page + 1}"${result.page + 1 >= result.pages ? ' disabled' : ''}>下一页</button></div>`;
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
  const api = {esc,date,finite,probability,ruleScore,supportScore,consensusVersion,weightedConsensus,currentEvidenceConsensus,isTrendConsensus,trendProbabilities,accountMarkerInputs,accountEvidence,accountContextText,accountActionHtml,rowAt,agentView,currentAgentView,riskDetails,structureFacts,highReturnState,agentsHtml,agentOverviewHtml,judgmentLabel,opinionReasons,opinionHtml,consensusAt,mergeEvidenceRows,finalSide,consensusHtml,diagnosticsHtml,popupSelection,referenceSelection,rawEventsHtml,referenceHtml,riskHtml,groupDates,popupHtml,historyPage,historyHtml,statisticsHtml,selectionTransition};
  api.highReturnTargetText = highReturnTargetText;
  root.KHQuantChartEvidence = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
