/* Current training plans and personal defaults. Eligibility comes from the service. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const state = {capabilities:null, profiles:null, status:null, fields:[], parameters:{}, revision:0, plan:null, planFingerprint:null, planState:'initial',planMessage:'',busyAction:null,jobs:new Map(), validationReceipts:new Map(), busy:false, loaded:false, dirty:false,runDirty:false,timer:null, diagnosticSequence:0,layoutObserver:null};
  const active = new Set(['queued','running','cancelling']);
  const statuses = {queued:'等待开始',running:'运行中',cancelling:'正在取消',cancelled:'已取消',succeeded:'已完成',failed:'失败',pending:'待补齐证据',blocked:'阻断',eligible:'具备研究选择资格',selected:'已更新研究选择',unchanged:'保留当前选择',qualified:'已通过',unavailable:'不可用',passed:'已通过',rejected:'未通过',disabled:'自动更新已关闭',not_requested:'未申请替换'};
  const runKeys = new Set(['end','source_native_run_id','device','cpu_workers']);
  const fingerprint = value => JSON.stringify(Object.fromEntries(Object.keys(value).sort().map(key => [key,value[key]])));
  const message = value => typeof value === 'string' ? value : value?.message || value?.reason || value?.detail || value?.code || JSON.stringify(value);
  const keyOf = field => field.key || field.name;
  const percentage = field => ['%','percent','percentage','比例','概率'].includes(field.unit) || keyOf(field).endsWith('_weight');
  const scale = field => keyOf(field) === 'liquidity_min_median_amount' ? .0001 : percentage(field) ? 100 : 1;
  const textValue = (field,value) => Array.isArray(value) ? value.join(', ') : typeof value === 'number' ? String(Number((value * scale(field)).toPrecision(12))) : String(value ?? '');

  async function api(path, options = {}) {
    const headers = {...options.headers}; if (options.body) headers['Content-Type'] = 'application/json';
    const token = sessionStorage.getItem('khquant_api_token'); if (token) headers.Authorization = `Bearer ${token}`;
    const response = await fetch(path,{...options,headers,credentials:'same-origin'});
    let payload; try { payload = await response.json(); } catch (_) { throw new Error(`服务响应无法读取（${response.status}）`); }
    if (!response.ok) throw new Error(message(payload.detail || payload.error || payload));
    return payload;
  }
  function fail(error, page = 'training') { const host = $(`${page}-error`); host.hidden = false; host.textContent = error.message || message(error); }
  function clearErrors() { ['training','parameters'].forEach(page => { $(`${page}-error`).hidden = true; }); }
  function currentPlanReady() {
    if (!state.loaded || !state.plan?.plan_id || state.plan.valid !== true || !state.planFingerprint) return false;
    try { return fingerprint(readParameters('training-form',state.parameters,false)) === state.planFingerprint; } catch (_) { return false; }
  }
  function renderStartState() {
    const labels = {initial:'尚未预检，请先检查训练计划。',parameters_changed:'本次参数已改变，请重新检查训练计划。',submitted:'任务已提交，请展开任务详情查看进度；再次训练需要新的预检。',preflight_failed:'计划检查失败，尚未取得有效计划；请查看错误后重新检查。',submission_failed:'未能确认任务提交，请先刷新任务状态；重新提交需要预检。',defaults_saved:'默认参数已保存，请重新检查训练计划。',ready:'预检通过，参数与冻结计划一致，可以开始完整训练。',blocked:'预检未通过，训练保持阻断。'};
    const busy = state.busyAction === 'preflight' ? '正在检查日期、数据和训练依赖…' : state.busyAction === 'submit' ? '正在提交冻结计划，请勿重复提交…' : state.busyAction === 'save' ? '正在保存默认参数…' : '正在读取训练能力…';
    $('training-start-state').textContent = state.busy ? busy : state.planState === 'ready' && !currentPlanReady() ? '当前参数与冻结计划不一致，请重新检查训练计划。' : state.planMessage || labels[state.planState] || labels.initial;
    $('training-start-state').dataset.state = state.busy ? 'busy' : state.planState;
    if (!state.busy && !state.plan && !$('training-preflight-result').hidden) $('training-preflight-result').textContent = $('training-start-state').textContent;
  }
  function setBusy(value) {
    state.busy = value;
    $('training-preflight').disabled = value || !state.loaded;
    $('training-start').disabled = value || !currentPlanReady();
    $('parameters-save').disabled = value || !state.loaded; $('parameters-reset').disabled = value || !state.loaded;
    if (!value) state.busyAction = null;
    renderStartState(); renderResourceControls(); scheduleTaskLayout();
  }
  function invalidatePlan(cause = 'parameters_changed', detail = '') {
    if (typeof cause !== 'string') cause = 'parameters_changed';
    state.plan = null; state.planFingerprint = null; state.planState = cause; state.planMessage = detail;
    $('training-start').disabled = true; renderStartState();
    if (!$('training-preflight-result').hidden) $('training-preflight-result').textContent = $('training-start-state').textContent;
    scheduleTaskLayout();
  }
  function control(field, value, prefix) {
    const key = keyOf(field), id = `${prefix}-${key}`, readonly = field.locked === true || field.readonly === true || field.editable === false;
    const resourceHints = {batch_size:'Torch每次训练批的样本行数；多股样本合批，全部历史合格股票参与',stock_batch_size:'股票准备分组上限，不是同时运行的股票数，也不是神经批量；CPU进程控制并行',cpu_workers:'CPU负责文件读取、已确认结构和精确价格；神经训练与批推理优先GPU',training_batch_size:'线性/MLP每次训练批的样本行数，与股票准备分组不同',device:'auto 优先CUDA、其次MPS、最后CPU；训练、批推理和校准使用Torch。明确指定的设备不可用时拒绝'};
    const hint = [key === 'liquidity_min_median_amount' ? '单位：万元人民币；保存和提交按元换算' : percentage(field) ? '单位：%' : field.unit ? `单位：${field.unit}` : '',resourceHints[key],field.impact_label || ({rebuild:'需重建来源、标签及受影响后代',retrain:'新任务重新训练',selection:'新任务的研究选择设置',resource:'仅计算分批，不限制全部合格股票人口'})[field.impact] || '新任务冻结此参数'].filter(Boolean).join(' · ');
    let input;
    if (field.type === 'boolean' || typeof value === 'boolean') input = `<input id="${esc(id)}" data-parameter="${esc(key)}" type="checkbox" ${value ? 'checked' : ''} ${readonly ? 'disabled' : ''}>`;
    else if (field.choices?.length) input = `<select id="${esc(id)}" data-parameter="${esc(key)}" ${readonly ? 'disabled' : ''}>${choiceOptions(field,value)}</select>`;
    else {
      const array = field.type?.includes('array') || field.type?.includes('list') || Array.isArray(value), numeric = ['integer','int','number','float'].includes(field.type), type = key === 'end' || key === 'feature_start' || field.type === 'date' ? 'date' : numeric ? 'number' : 'text';
      input = `<input id="${esc(id)}" data-parameter="${esc(key)}" type="${type}" value="${esc(textValue(field,value))}" ${numeric ? `step="${field.type === 'integer' || field.type === 'int' ? 1 : 'any'}"` : ''} ${field.min != null && numeric ? `min="${field.min * scale(field)}"` : ''} ${field.max != null && numeric ? `max="${field.max * scale(field)}"` : ''} ${readonly ? 'readonly' : ''} ${field.required === false || key === 'end' || array ? '' : 'required'} ${array ? 'placeholder="逗号分隔"' : ''}>`;
    }
    return `<label class="training-field ${readonly ? 'locked' : ''}" for="${esc(id)}"><span>${esc(field.label || field.name || key)}${readonly ? ' · 固定约束' : ''}</span>${input}<small>${esc(hint)}</small></label>`;
  }
  function renderParameters(parameters, updateRun = true) {
    const groups = new Map();
    for (const field of state.fields) { const group = field.group_label || field.group || '训练参数'; if (!groups.has(group)) groups.set(group,[]); groups.get(group).push(field); }
    $('parameters-fields').innerHTML = [...groups].map(([name,fields]) => `<fieldset class="training-parameter-group"><legend>${esc(name)}</legend><div class="training-fields">${fields.map(field => control(field,parameters[keyOf(field)],'default')).join('')}</div></fieldset>`).join('');
    if (updateRun) $('training-run-fields').innerHTML = state.fields.filter(field => runKeys.has(keyOf(field))).map(field => control(field,parameters[keyOf(field)],'run')).join('');
    const profile = state.profiles.default_profile || {};
    $('parameters-revision').textContent = `${profile.name || '我的默认配置'} · 保存修订 ${state.revision}`;
    $('training-profile-note').textContent = `使用已保存默认参数（修订 ${state.revision}）；本次可调整数据和资源。其他参数在参数设置中保存后生效。`;
    renderResourceControls();
  }
  const deviceName = value => ({auto:'自动（CUDA → MPS → CPU）',cpu:'CPU',mps:'MPS',cuda:'CUDA'})[value] || value || '未报告';
  function choiceOptions(field, value) {
    const devices = Array.isArray(state.capabilities?.devices) ? state.capabilities.devices : null;
    return field.choices.map(choice => {
      const option = typeof choice === 'object' ? choice.value : choice;
      let label = typeof choice === 'object' ? choice.label || option : option;
      const device = keyOf(field) === 'device', unavailable = device && option !== 'auto' && devices && !devices.includes(option);
      if (device) label = `${deviceName(option)} · ${devices ? unavailable ? '当前不可用，不自动回退' : '可用' : '可用性未报告'}`;
      return `<option value="${esc(option)}" ${String(option) === String(value) ? 'selected' : ''} ${unavailable ? 'disabled' : ''}>${esc(label)}</option>`;
    }).join('');
  }
  function suggestedDevice() {
    const devices = state.capabilities?.devices, field = state.fields.find(item => keyOf(item) === 'device');
    if (!Array.isArray(devices) || !field) return null;
    const choices = field.choices.map(choice => typeof choice === 'object' ? choice.value : choice);
    return choices.includes('auto') ? 'auto' : ['cuda','mps','cpu'].find(device => devices.includes(device) && choices.includes(device)) || null;
  }
  function renderResourceControls() {
    const devices = state.capabilities?.devices, field = state.fields.find(item => keyOf(item) === 'device');
    if (field) for (const prefix of ['run','default']) { const select = $(`${prefix}-device`); if (select) { const value = select.value; select.innerHTML = choiceOptions(field,value); select.value = value; } }
    const selected = $('run-device')?.value || state.parameters.device, suggested = suggestedDevice(), button = $('training-device-suggest');
    $('training-resource-availability').textContent = Array.isArray(devices) ? `接口报告可用训练设备：${devices.map(deviceName).join('、') || '无'}。可用性不代表任务已使用GPU。` : '设备可用性尚未报告；保留原选择，不自动换设备。';
    button.textContent = suggested ? `本次选择 ${deviceName(suggested)}` : '设备建议暂不可用';
    button.disabled = state.busy || !state.loaded || !suggested || selected === suggested;
    const unavailable = Array.isArray(devices) && selected && selected !== 'auto' && !devices.includes(selected);
    $('training-resource-status').textContent = `${selected ? `本次表单设备：${deviceName(selected)}。` : ''}${unavailable ? '当前不可用，预检将拒绝；不会自动回退CPU。' : '设备建议仅修改本次表单，需重新预检；不保存个人默认，也不改变已提交任务。'}`;
  }
  function applyDeviceSuggestion() {
    if (state.busy || !state.loaded) return;
    const device = suggestedDevice(), select = $('run-device');
    if (!device || !select || select.value === device) return;
    select.value = device; state.runDirty = true;
    invalidatePlan('parameters_changed',`本次训练设备已明确选择 ${deviceName(device)}，请重新预检；个人默认未保存。`);
    renderResourceControls();
  }
  function readParameters(formId, initial, validate = true) {
    const form = $(formId); if (validate && !form.reportValidity()) throw new Error('请补齐参数并检查取值范围。');
    const result = Object.fromEntries(state.fields.map(field => [keyOf(field),initial[keyOf(field)] ?? field.default ?? null]));
    for (const element of form.querySelectorAll('[data-parameter]')) {
      const field = state.fields.find(item => keyOf(item) === element.dataset.parameter), key = keyOf(field);
      if (field.locked === true || field.readonly === true || field.editable === false) continue;
      let value;
      if (element.type === 'checkbox') value = element.checked;
      else if (field.type?.includes('array') || field.type?.includes('list') || Array.isArray(initial[key])) {
        const parts = element.value.split(/[,，\s]+/).filter(Boolean);
        value = field.type?.includes('string') ? parts : parts.map(Number);
        if (value.some(item => typeof item === 'number' && !Number.isSafeInteger(item))) throw new Error(`${field.label || key}必须填写整数列表。`);
      } else if (['integer','int','number','float'].includes(field.type)) {
        value = Number(element.value) / scale(field);
        if (!element.value.trim() || !Number.isFinite(value) || (['integer','int'].includes(field.type) && !Number.isSafeInteger(value))) throw new Error(`${field.label || key}必须为有效数值。`);
      } else value = ['end','source_native_run_id'].includes(key) && !element.value ? null : element.value;
      result[key] = value;
    }
    return result;
  }
  function renderConstraints() {
    const supplied = state.capabilities.locked_constraints || state.capabilities.constraints || [];
    const fixed = ['预测下一核验交易日开盘至第10交易日收盘的价格方向；不模拟成交','训练内部选型、校准、仓位政策与留出测试按时间隔离；标签截止清洗','自然类别分布训练；缺失、未成熟与不合格样本不填作下跌','每股目标仓位的变化相对于上一参考目标；手工仓位仅单独比较','技术完整性、预测基线比较与独立前瞻资格分别报告'];
    const values = Array.isArray(supplied) && supplied.length ? supplied.map(message) : fixed;
    const html = values.map(item => `<li>${esc(item)}</li>`).join(''); $('training-constraints').innerHTML = html; $('parameters-constraints').innerHTML = `<ul>${html}</ul>`;
    const stages = state.capabilities.stages || ['核验行情、日历和可复用因果特征','新纯价格目标与时间清洗','Torch线性/MLP训练内部选型','独立校准与仓位政策选择','留出预测基线验证','每股参考目标状态机与B/S变化点'];
    $('training-stages').innerHTML = stages.map(item => `<li>${esc(typeof item === 'object' ? (item.label || item.name || item.key) + (item.status === 'not_connected' ? ' · 尚未接入' : item.status === 'observation_only' ? ' · 只作观察' : item.trainable === false ? ' · 规则与数据计算' : '') : item)}</li>`).join('');
  }
  function renderProfiles() {
    const items = state.profiles.items || [];
    $('parameters-history').innerHTML = items.length ? items.map(item => `<article><strong>${esc(item.name || item.profile_id)}</strong><span>${item.archived ? '历史归档 · 只读，不参与当前训练' : item.builtin ? '内置基准' : '个人配置'} · 修订 ${esc(item.revision)}</span>${item.hash ? `<details><summary>配置凭据</summary><code>${esc(item.hash)}</code></details>` : ''}</article>`).join('') : '<p class="subtle">当前仅有默认配置；保存后保留修订凭据。</p>';
  }
  function populationHtml(value = {}) {
    const union = value.historical_union_count, latest = value.latest_eligible_count, trained = value.trained_stock_count;
    return `<p class="training-population">历史合格并集 <strong>${count(union)}</strong> 股 · 当日合格 <strong>${count(latest)}</strong> 股 · 实际完成 <strong>${count(trained)}</strong> 股${trained == null ? '（尚未报告）' : ''}</p>`;
  }
  function renderStatus() {
    const status = state.status || {}, model = status.active_model, certificate = status.certification || {};
    $('training-overview').innerHTML = `<article><span>当前研究模型</span><strong>${model ? esc(model.display_name || '参考买卖点与每股仓位') : '尚无合格当前选择'}</strong><small>${esc(model?.run_id || '未评分时保留缺失状态')}${model?.effective_from ? ` · 生效 ${esc(model.effective_from)}` : ''}</small></article><article><span>数据实际截止</span><strong>${esc(status.data_end || state.capabilities?.data_end || '预检时核验')}</strong><small>${status.calendar_run_id ? '已登记核验交易日历' : '预检校验日历来源与覆盖'}</small></article><article><span>正式生产资格</span><strong>${certificate.status === 'qualified' && certificate.publication_allowed === true ? '已通过独立认证' : '尚未取得'}</strong><small>${esc(certificate.reason || '研究选择不代替独立认证')}</small></article>`;
    const selection = status.current_selection || status.selection || {}, blockers = selection.reasons || selection.errors || status.blockers || [];
    $('training-selection').innerHTML = `<p><strong>${esc(selection.status ? statuses[selection.status] || selection.status : '自动替换按完整证据判定')}</strong></p><p class="training-note">技术验证检查资产、因果时间和完整人口；预测验证比较独立留出段的LogLoss、Brier、F1及基线。独立前瞻资格另行记录。</p>${blockers.length ? `<ul class="training-blockers">${blockers.map(item => `<li>${esc(message(item))}</li>`).join('')}</ul>` : ''}${renderModelRows(status.models || [])}`;
  }
  function referenceQualityTrainingHtml(summary, diagnosticReasons = []) {
    if (!summary || typeof summary !== 'object') return '';
    const names = {trained:'已训练',completed:'已完成',qualified:'资格通过',pending:'待核验',unavailable:'不可用',failed:'失败',passed:'通过',rejected:'未通过'};
    const fee = summary.fee_qualified === true ? '已通过' : summary.fee_status === 'pending' ? '待完整费用验收' : '未通过或证据不足';
    const reasons = (Array.isArray(diagnosticReasons) ? diagnosticReasons : []).filter((item) => /reference_quality|reference_population|参考/.test(message(item)));
    return `<section class="training-reference-quality" data-job-detail="reference-quality"><h4>独立参考质量</h4><p>${esc(names[summary.status] || summary.status || '尚未完成')} · 完成头 ${esc(summary.heads ?? 0)} · 完整费用资格 ${fee}</p><p class="training-note">专属无账户参考买方人口；目标仍为完整费用后至少8%且2R。参考质量与核心自动选模分别核验；参考资格不足时，参考仓位目标保持待核验。</p>${reasons.length ? `<ul class="reference-quality-diagnostics">${reasons.map((item) => `<li>${esc(message(item))}</li>`).join('')}</ul>` : ''}<details><summary>参考头与费用证据</summary>${summary.population ? `<p>人口：独立参考买方机会</p>` : ''}${summary.population_hash ? `<p>人口凭据 <code>${esc(summary.population_hash)}</code></p>` : ''}${summary.qualification_path ? `<p>费用报告 <code>${esc(summary.qualification_path)}</code></p>` : ''}${summary.qualification_sha256 ? `<p>费用报告校验 <code>${esc(summary.qualification_sha256)}</code></p>` : ''}${summary.unavailable_heads?.length ? `<pre>${esc(JSON.stringify(summary.unavailable_heads,null,2))}</pre>` : '<p>没有已报告的参考头不足原因。</p>'}<p>此参考质量结果不授予正式生产资格。</p></details></section>`;
  }
  function referenceValidationHtml(value = {}) {
    const technical = value.technical_validation ?? value.validation?.technical ?? {}, predictive = value.predictive_validation || value.predictive_gate || value.validation?.predictive || {}, forward = value.forward_validation || value.certification || {};
    const label = part => typeof part === 'boolean' ? (part ? '通过' : '未通过') : typeof part === 'string' ? statuses[part] || part : statuses[part.status] || part.status || (part.passed === true ? '通过' : part.passed === false ? '未通过' : '待核验');
    const finiteMetric = metric => typeof metric === 'number' && Number.isFinite(metric);
    const metricText = metric => finiteMetric(metric) ? metric.toFixed(6) : '未报告';
    const predictiveLabel = predictive.passed === false && finiteMetric(predictive.log_loss) ? '未通过基线（不生成积极增仓目标）' : predictive.passed === true ? '已通过基线' : '待核验';
    const metrics = finiteMetric(predictive.log_loss) || finiteMetric(predictive.brier) ? `<p class="training-predictive-metrics">LogLoss：模型 ${metricText(predictive.log_loss)} / 基线 ${metricText(predictive.baseline_log_loss)} · Brier：模型 ${metricText(predictive.brier)} / 基线 ${metricText(predictive.baseline_brier)}（越低越好）</p>` : '';
    return `<section class="training-reference-quality" data-job-detail="reference-validation"><h4>参考模型验证</h4><p>技术验证：${esc(label(technical))} · 预测基线：${esc(predictiveLabel)} · 独立前瞻：${esc(label(forward))}</p>${metrics}<p class="training-note">10交易日价格方向概率与每股参考仓位；仓位变化相对上一参考目标。自然类别分布训练，时间段互相隔离。</p><details><summary>验证指标与资产凭据</summary><pre>${esc(JSON.stringify({technical,predictive,forward},null,2))}</pre></details></section>`;
  }
  function renderModelRows(models) {
    if (!models.length) return '<p class="subtle">尚无完成的参考仓位模型；请先提交训练计划。</p>';
    return `<div class="table-wrap"><table><thead><tr><th>模型运行与实际人口</th><th>验证</th><th>使用状态</th></tr></thead><tbody>${models.map(model => `<tr><td>${esc(model.run_id)}${model.archived === true ? '<span class="badge">历史归档 · 只读</span>' : ''}${populationHtml(model)}</td><td>${model.archived === true ? esc(model.archive_reason || '历史模型仅保留诊断') : referenceValidationHtml(model)}</td><td>${model.archived === true ? '新架构需重新训练' : esc(model.selection_status || model.status || '待核验')}</td></tr>`).join('')}</tbody></table></div>`;
  }
  function dateStatusHtml(result, plan) {
    const value = plan.date_status || result.date_status || {}, code = typeof value === 'object' ? value.code : value;
    const requested = plan.requested_end || result.requested_end;
    const effective = Object.hasOwn(plan,'effective_end') ? plan.effective_end : Object.hasOwn(result,'effective_end') ? result.effective_end : plan.data_end;
    const before = plan.nearest_verified_before || result.nearest_verified_before || value.nearest_verified_before;
    const after = plan.nearest_verified_after || result.nearest_verified_after || value.nearest_verified_after;
    return `<p class="training-date-status">请求截止 ${esc(requested || '自动最新')} · 有效数据截止 ${esc(effective || '未核验')}${typeof value === 'object' && value.message ? ` · ${esc(value.message)}` : /non_trading|non_verified|not.*session|unverified/.test(code || '') ? ' · 所选日期不是已核验交易日，需重新选择日期' : ''}${before ? `<br>最近前一核验交易日 ${esc(before)}` : ''}${after ? ` · 最近后一核验交易日 ${esc(after)}` : ''}</p>`;
  }
  function renderPreflight(result) {
    const plan = result.plan || result, valid = result.valid === true && Boolean(plan.plan_id), errors = result.errors || plan.errors || [], warnings = result.warnings || plan.warnings || [];
    state.plan = {...plan,valid};
    state.planState = valid ? 'ready' : 'blocked'; state.planMessage = valid ? '' : `预检未通过：${message(errors[0] || '未取得完整冻结计划凭据')}。训练保持阻断。`;
    const host = $('training-preflight-result'); host.hidden = false;
    const universe = plan.liquidity_universe || {latest_count:plan.latest_eligible_count}, union = universe.selected_symbols?.length ?? plan.historical_union_count;
    $('training-population-status').innerHTML = populationHtml({historical_union_count:union,latest_eligible_count:universe.latest_count});
    host.innerHTML = `<h3>${valid ? '预检通过，可以提交冻结计划' : '训练计划被阻断'}</h3>${dateStatusHtml(result,plan)}${errors.length ? `<ul class="training-blockers">${errors.map(item => `<li>${esc(message(item))}</li>`).join('')}</ul>` : ''}${warnings.length ? `<ul class="training-warnings">${warnings.map(item => `<li>${esc(message(item))}</li>`).join('')}</ul>` : ''}<dl><div><dt>历史合格并集</dt><dd>${count(union)} 股 · 全部参与</dd></div><div><dt>数据截止日合格</dt><dd>${count(universe.latest_count)} 股</dd></div><div><dt>比较现任</dt><dd>${esc(plan.incumbent_run_id || '无现任；初始化资格另行核验')}</dd></div><div><dt>日历</dt><dd>${esc(plan.calendar_run_id || '尚未核验')}</dd></div></dl><p>复用来源保留逐日流动性资格及完整历史人口；新标签与权重重新训练。GPU行批 ${count(plan.parameters?.batch_size)}，CPU准备线程 ${count(plan.parameters?.cpu_workers)}；计算分批不截断股票人口。</p>${plan.windows?.length ? `<details><summary>时间切分与成熟范围</summary><pre>${esc(JSON.stringify(plan.windows,null,2))}</pre></details>` : ''}${universe.policy_hash ? `<details><summary>逐日资格与冻结人口凭据</summary><pre>${esc(JSON.stringify({policy:universe.policy,policy_hash:universe.policy_hash,daily_counts:universe.daily_counts,exclusion_counts:universe.exclusion_counts},null,2))}</pre></details>` : ''}${plan.rebuild_scope ? `<p>重建范围：${esc(message(plan.rebuild_scope))}</p>` : ''}${plan.plan_id ? `<details><summary>冻结计划凭据</summary><code>${esc(plan.plan_id)}</code><p>配置SHA：${esc(plan.config_hash || '提交时核验')}</p></details>` : ''}`;
    renderStartState(); scheduleTaskLayout();
  }
  function stageName(value) {
    const labels = {source_verification:'核验因果来源',reference_labels:'生成纯价格方向目标',dataset_load:'读取已核验训练样本',reference_model:'训练Torch方向概率',model_fit:'训练Torch模型',model_select:'训练内部选择模型',calibration:'独立概率校准',validation:'验证预测基线与资产',completed:'完整训练完成',reference_source_verification:'核验可复用行情与因果特征',reference_dataset:'生成新纯价格目标',reference_training:'Torch参考模型训练',reference_calibration:'GPU概率校准',reference_policy:'核验逐股仓位政策',reference_validation:'留出预测与资产验证',reference_selection:'记录参考模型选择',reference_fit:'训练方向概率',reference_evaluation:'比较留出预测基线',calendar:'核验交易日历',raw_source:'冻结原始行情',raw_source_freeze:'冻结原始行情',native_source:'准备原生特征',native_source_prepare:'结构数据准备',data:'核验行情与日历',trend:'训练三分类趋势',independent_three_class_trend:'训练三分类趋势',source_policy:'核验来源政策',independent_policy_selection:'核验来源政策',missing_trend_dependency:'趋势模型依赖不足',ma_exit:'训练固定持仓退出',train_independent_actual_holding_MA_exit:'训练固定持仓退出',sequential_L2_actual_flat_pre_gate_population_and_forced_E1_labels:'逐折构建评分前尝试与真实克隆目标',strict_prior_L2_population_by_fold:'逐折构建严格先验样本',entry:'训练高收益入场质量',reference_quality_population:'生成独立参考机会人口',train_reference_quality:'训练独立参考质量头',reference_quality_fee_qualification:'核验参考人口完整费用资格',diagnostics:'复算样本与费用',comparison:'比较真实账户账本',selection:'检查合格模型替换',training:'完整训练',validation:'重新核验已完成模型',queued:'等待训练资源',resource_wait:'等待上一任务释放计算资源',frozen_source_verification:'核验冻结来源',model_head:'模型头训练完成'};
    if (labels[value]) return labels[value];
    if (String(value).startsWith('E0')) return '构建初始真实参考账户';
    if (String(value).startsWith('E1')) return '构建固定退出头续持账户';
    if (String(value).startsWith('trend_')) return '训练与核验趋势检查点';
    if (String(value).startsWith('ma_exit_')) return '训练与核验持仓退出检查点';
    if (String(value).startsWith('L2') || String(value).startsWith('l2')) return '逐折训练高收益与退出观察';
    return value || '等待阶段报告';
  }
  const jobTitle = job => job.kind === 'model_validation' || job.result?.training_started === false ? '已完成模型复核' : '完整模型训练';
  function jobResourceHtml(job) {
    const detail = job.progress_detail || {}, stage = String(detail.stage || ''), parameters = job.spec?.parameters || {};
    let rule = '本阶段执行方式尚未报告；仅据实际设备字段判断计算位置。';
    if (job.result?.strategy_version === 'reference_position_v1' || parameters.horizon_sessions != null || /^(reference_model|reference_labels|reference_validation|model_fit|model_select|calibration|dataset_load)$/.test(stage)) rule = /source|labels|dataset|data|calendar/.test(stage) ? '行情读写、已确认结构和精确价格比较在CPU；可复用因果特征会先核验哈希。' : '新参考模型的训练、批量推理与概率校准优先使用CUDA/MPS；阶段实际设备以下方报告为准。';
    else if (stage === 'independent_policy_selection') rule = '历史政策评估：CPU；此记录属于旧成交研究任务。';
    else if (/ma_exit|^entry$|^L2/i.test(stage)) rule = '历史线性/MLP训练按冻结请求设备执行，树头在CPU；此记录只读保留。';
    else if (/raw_source|native_source|calendar|reference_quality/.test(stage)) rule = '历史来源与费用准备在CPU；设备报告保留原始记录。';
    return `<details data-job-detail="resources" class="training-job-resources"><summary>阶段设备与资源</summary><p>${esc(rule)}</p><p>冻结请求设备：${esc(deviceName(parameters.device))} · 本阶段实际设备报告：${esc(deviceName(detail.actual_device ?? detail.device))}</p>${parameters.stock_batch_size != null || parameters.cpu_workers != null || parameters.training_batch_size != null || parameters.batch_size != null ? `<p>股票准备分组 ${esc(parameters.stock_batch_size ?? '未报告')} 股 · CPU准备进程 ${esc(parameters.cpu_workers ?? '未报告')} · 神经训练行批 ${esc(parameters.batch_size ?? parameters.training_batch_size ?? '未报告')} 行</p>` : ''}<p>实际设备只引用任务明确报告；完成模型以训练报告的实际设备记录核验。不推算GPU利用率或预计完成时间。</p></details>`;
  }
  function jobProgressText(job) {
    const detail = job.progress_detail || {}, children = detail.children || {};
    const waitingNativeBatch = active.has(job.status) && !children.model_run_id &&
      (['native_source','native_source_prepare'].includes(detail.stage) || detail.stage === 'training' &&
        (children.source_native_run_id || String(detail.message || '').startsWith('NATIVE_SOURCE_RUN_ID '))) &&
      Number(detail.current || 0) === 0 && Number(detail.total || 0) === 0;
    if (waitingNativeBatch) return detail.checkpoint_available === true ? '结构数据准备中 · 按逐股核验更新计数' : '结构数据准备中 · 首批完成后更新计数';
    const hasTotal = Number.isFinite(Number(detail.total)) && Number(detail.total) > 0;
    return stageName(detail.stage) + (hasTotal && detail.current != null ? ` · ${detail.current} / ${detail.total}` : '');
  }
  const validationModel = job => (state.status?.models || []).find(item => item.run_id === (job.result?.training_run_id || job.result?.model_run_id || job.progress_detail?.children?.model_run_id));
  function validationBlock(job) {
    const model = validationModel(job);
    if (model?.archived === true) return model.archive_reason || '此模型已归档，只读保留原诊断；需使用新完整来源训练，不能复核旧权重。';
    return model?.validation_available === false ? model.archive_reason || '完整模型资产尚不可核验，不能提交复核。' : '';
  }
  function validationSpec(job) {
    const result = job.result || {}, model = result.training_run_id || result.model_run_id || job.progress_detail?.children?.model_run_id;
    const plan = result.original_plan_id || result.plan_id || job.spec?.plan_id;
    const current = validationModel(job), completed = current?.status === 'completed';
    return model && plan && completed && current.validation_available === true && !validationBlock(job) && !active.has(job.status) ? {model_run_id:model,plan_id:plan} : null;
  }
  function validationButton(job) {
    const model = job.result?.training_run_id || job.result?.model_run_id || job.progress_detail?.children?.model_run_id;
    if (!model || active.has(job.status)) return '';
    const token = job.job_id || job.run_id || model, spec = validationSpec(job), blocked = validationBlock(job);
    state.validationReceipts.set(token,job);
    return `<button type="button" data-training-validate="${esc(token)}" ${spec ? '' : `disabled title="${esc(blocked || '完整训练资产和原冻结计划尚未核验可用')}"`}>重新核验已完成模型</button>${blocked ? `<small>${validationModel(job)?.archived === true ? '历史归档' : '复核不可用'} · ${esc(blocked)} 原任务与原诊断保留，可继续查看。</small>` : spec ? '<small>核验资产、时间切分与预测基线；模型权重保持不变。</small>' : '<small>完整模型资产或原计划尚不可核验，复核不可提交。</small>'}`;
  }
  async function revalidate(job) {
    const spec = validationSpec(job);
    if (!spec) throw new Error(validationBlock(job) || '完整模型资产和原冻结计划尚未核验可用，不能提交复核。');
    const task = await api('/api/training/validation-jobs',{method:'POST',body:JSON.stringify(spec)});
    if (!task.job_id) throw new Error('服务未返回复核任务标识。');
    state.jobs.set(task.job_id,task); localStorage.setItem('khquant_current_training_job',task.job_id); renderJobs(); await pollJobs();
  }
  function nativeProgressStatsHtml(job) {
    const d = job.progress_detail || {};
    if (d.stage !== 'native_source_prepare') return '';
    const valid = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
    const rate = d.stocks_per_minute ?? d.throughput_per_minute;
    const inflight = d.in_flight ?? d.inflight;
    const pieces = [];
    if (valid(inflight)) pieces.push(`计算中 ${inflight} 股`);
    if (valid(d.cache_hits)) pieces.push(`数据库复用 ${d.cache_hits} 股`);
    if (valid(d.reused)) pieces.push(`断点复用 ${d.reused} 股`);
    if (valid(rate) && rate > 0) pieces.push(`实测 ${rate.toFixed(1)} 股/分钟`);
    if (valid(d.eta_seconds) && d.current < d.total) pieces.push(`本阶段剩余约 ${Math.max(1,Math.ceil(d.eta_seconds / 60))} 分钟`);
    return pieces.length ? `<p class="training-note">${esc(pieces.join(' · '))}</p>` : '';
  }
  function nativeResumeSpec(job) {
    return null; // Old recovery is archive-only; new runs require a fresh plan.
  }
  async function resumeNative(job) {
    const spec = nativeResumeSpec(job);
    if (!spec) throw new Error('旧训练恢复入口已退役，请新建参考仓位预检计划。');
    const checked = await api('/api/training/preflight',{method:'POST',body:JSON.stringify(spec)});
    if (checked.valid !== true || !checked.plan?.plan_id) throw new Error((checked.errors || ['检查点核验未通过']).join('；'));
    const task = await api('/api/training/jobs',{method:'POST',body:JSON.stringify({plan_id:checked.plan.plan_id,parameters:spec.parameters})});
    if (!task.job_id) throw new Error('服务未返回恢复任务标识。');
    state.jobs.set(task.job_id,task);localStorage.setItem('khquant_current_training_job',task.job_id);
    renderJobs();await pollJobs();
  }
  function renderJob(job) {
    const detail = job.progress_detail || {}, result = job.result || {}, progress = Math.max(0,Math.min(100,Number(job.progress) || 0));
    const comparison = result.comparison || result.selection || {}, reasons = comparison.reasons || result.blockers || [];
    return `<article class="training-job" data-training-job="${esc(job.job_id || job.run_id)}"><div class="task-heading"><strong>${jobTitle(job)}</strong><span class="badge">${esc(statuses[job.status] || job.status || '读取中')}</span></div><small>${esc(job.run_id || job.job_id)}</small><div class="progress-track" aria-label="${job.status === 'succeeded' ? '任务完成' : '当前阶段进度'} ${progress}%"><i style="width:${progress}%"></i></div><p>${active.has(job.status) ? `当前阶段 ${progress}% · ` : ''}${esc(jobProgressText(job))}${detail.failed ? ` · 失败 ${esc(detail.failed)}` : ''}${detail.device ? ` · 实际设备 ${esc(detail.device)}` : ''}</p>${nativeProgressStatsHtml(job)}${jobResourceHtml(job)}${job.error ? `<p class="training-message error">${esc(message(job.error))}</p>` : ''}${job.status === 'succeeded' ? `<p>${job.kind === 'model_validation' || result.training_started === false ? '复核已完成' : '训练已完成'}；替换结果：${esc(statuses[result.selection?.status || result.comparison_status] || result.selection?.status || result.comparison_status || '以资格报告为准')}。</p>` : ''}${result.training_started === false && result.weights_unchanged === true ? '<p>本次没有启动训练，模型权重保持不变；原任务与历史诊断保留。</p>' : ''}${reasons.length ? `<ul class="training-blockers">${reasons.map(reason => `<li>${esc(message(reason))}</li>`).join('')}</ul>` : ''}${result.training_run_id ? `<p>模型运行：${esc(result.training_run_id)}</p>` : ''}${result.strategy_version === "reference_position_v1" ? referenceValidationHtml(result) : referenceQualityTrainingHtml(result.reference_quality,result.result_reasons)}${populationHtml({...detail,...result})}${childrenHtml(detail.children || result.children)}${diagnosticsHtml(result.diagnostics)}${result.required_heads ? `<details data-job-detail="heads"><summary>必需头资格</summary><pre>${esc(JSON.stringify(result.required_heads,null,2))}</pre></details>` : ''}<div class="button-group">${active.has(job.status) ? `<button type="button" data-training-cancel="${esc(job.job_id)}" ${job.status === 'cancelling' ? 'disabled' : ''}>${job.kind === 'model_validation' ? '取消复核' : '取消训练'}</button>` : ''}${job.status === 'succeeded' && job.run_id ? `<button type="button" data-training-result="${esc(job.job_id)}">查看结果与阻断原因</button>` : ''}${result.training_run_id ? `<button type="button" data-training-comparison="${esc(result.training_run_id)}">查看验证与基线</button>` : ''}${validationButton(job)}${nativeResumeSpec(job) ? `<button type="button" data-training-resume="${esc(job.job_id)}">继续结构准备与训练</button>` : ''}</div>${job.logs?.length ? `<details data-job-detail="logs"><summary>阶段日志</summary><pre>${esc(job.logs.map(message).join('\n'))}</pre></details>` : ''}</article>`;
  }
  function renderTaskSummary(jobs) {
    const running = jobs.filter(job => active.has(job.status)), failed = jobs.filter(job => job.status === 'failed'), cancelled = jobs.filter(job => job.status === 'cancelled');
    const blocked = jobs.filter(job => job.result?.selection?.status === 'blocked' || job.result?.diagnostics?.status === 'failed');
    const current = running[0], summary = $('training-task-summary');
    summary.innerHTML = `<span>共 ${jobs.length} 项 · 活动 ${running.length}</span>${failed.length ? `<span class="badge task-bad">失败 ${failed.length}</span>` : ''}${cancelled.length ? `<span class="badge">已取消 ${cancelled.length}</span>` : ''}${blocked.length ? `<span class="badge task-bad">资格／诊断阻断 ${blocked.length}</span>` : ''}${current ? `<span>${jobTitle(current)} · ${esc(jobProgressText(current))}</span>` : '<span class="subtle">当前没有运行中的任务；关闭页面后已提交任务继续。</span>'}`;
  }
  function renderJobs() {
    const host = $('training-jobs'), scroll = $('training-job-scroll'), previousScroll = scroll.scrollTop;
    const opened = new Map([...host.querySelectorAll('details[data-job-detail]')].map(item => [item.closest('[data-training-job]')?.dataset.trainingJob + ':' + item.dataset.jobDetail,item.open]));
    const jobs = [...state.jobs.values()].sort((a,b) => String(b.created_at || b.job_id).localeCompare(String(a.created_at || a.job_id)));
    host.innerHTML = jobs.length ? jobs.map(renderJob).join('') : '<p class="empty-copy">尚无当前训练或复核任务。</p>';
    for (const item of host.querySelectorAll('details[data-job-detail]')) { const key = item.closest('[data-training-job]')?.dataset.trainingJob + ':' + item.dataset.jobDetail; if (opened.has(key)) item.open = opened.get(key); }
    scroll.scrollTop = previousScroll; renderTaskSummary(jobs); scheduleTaskLayout();
  }
  function taskLayoutBudget(panelTop, boundaryBottom, headingHeight, padding, gap) {
    const panel = Math.max(headingHeight + padding, boundaryBottom - panelTop);
    return {panel,body:Math.max(0,panel - headingHeight - padding - gap)};
  }
  function syncTaskLayout() {
    const panel = $('training-task-panel'), boundary = $('training-qualification-panel'), heading = $('training-task-heading'), body = $('training-job-scroll');
    if (!panel?.getBoundingClientRect || !boundary?.getBoundingClientRect || !globalThis.getComputedStyle) return;
    const desktop = globalThis.matchMedia?.('(min-width: 801px)').matches;
    if (!desktop) { panel.style.removeProperty('--task-panel-size'); body.style.removeProperty('--task-body-size'); return; }
    const box = panel.getBoundingClientRect(), target = boundary.getBoundingClientRect(); if (!box.width || !target.height) return;
    const style = getComputedStyle(panel), bodyStyle = getComputedStyle(body), px = value => Number.parseFloat(value) || 0;
    const padding = ['paddingTop','paddingBottom','borderTopWidth','borderBottomWidth'].reduce((sum,key) => sum + px(style[key]),0);
    const budget = taskLayoutBudget(box.top,target.bottom,heading.getBoundingClientRect().height,padding,px(bodyStyle.marginTop));
    panel.style.setProperty('--task-panel-size',budget.panel + 'px'); body.style.setProperty('--task-body-size',budget.body + 'px');
  }
  let layoutFrame = null;
  function scheduleTaskLayout() {
    if (!globalThis.requestAnimationFrame || layoutFrame != null) return;
    layoutFrame = requestAnimationFrame(() => { layoutFrame = null; syncTaskLayout(); });
  }
  async function events() {
    const response = await api('/api/training/selection-events'), items = Array.isArray(response) ? response : response.events || response.items || [];
    $('training-selection-events').innerHTML = items.length ? items.map(item => { const payload = item.payload || item; return `<article><strong>${esc(({select:'更新当前研究模型',rollback:'回滚当前研究选择'})[item.kind] || item.action || item.event_type || '模型选择记录')}</strong><span>记录 ${esc(item.created_at || payload.recorded_at || payload.selected_at || '')}${payload.effective_from ? ` · 生效 ${esc(payload.effective_from)}` : ''}</span><p>${esc(payload.reason || payload.reason_code || '通过冻结证据检查后记录')}</p><small>模型 ${esc(payload.run_id || payload.model_run_id || '无模型状态')}${payload.previous_run_id ? ` · 前任 ${esc(payload.previous_run_id)}` : ''}</small></article>`; }).join('') : '<p class="subtle">尚无当前体系替换事件；训练完成不会自动伪造选择或发布记录。</p>';
  }
  async function pollJobs() {
    if(document.hidden){clearTimeout(state.timer);state.timer=setTimeout(()=>pollJobs().catch(()=>{}),15000);return;}
    clearTimeout(state.timer);
    const jobs = [...state.jobs.values()].filter(job => active.has(job.status));
    for (const job of jobs) { try { state.jobs.set(job.job_id,await api(`/api/tasks/${encodeURIComponent(job.job_id)}`)); } catch (error) { fail(error); } }
    renderJobs();
    if ([...state.jobs.values()].some(job => active.has(job.status))) state.timer = setTimeout(() => pollJobs().catch(fail),2500);
    else if (jobs.length) { state.status = await api('/api/training/status'); renderStatus(); renderJobs(); await events(); window.dispatchEvent(new Event('khquant-selection-updated')); }
  }
  async function refresh() {
    if (state.busy) return;
    clearErrors();
    const [capabilities,profiles,status] = await Promise.all([api('/api/training/capabilities'),api('/api/training/profiles'),api('/api/training/status')]);
    state.capabilities = capabilities; state.profiles = profiles; state.status = status;
    const fields = capabilities.schema?.fields || capabilities.parameter_schema?.fields || capabilities.fields;
    if (!Array.isArray(fields) || !fields.length) throw new Error('服务未提供可编辑参数契约，训练暂不可提交。');
    state.fields = fields.filter(field => !['cohort_size','capacity_tiers'].includes(keyOf(field))).map(field => {
      const key = keyOf(field), choices = field.choices || field.options, numericChoice = choices?.length && typeof choices[0] === 'number';
      return {...field,choices,type:numericChoice ? 'integer' : ['end','feature_start'].includes(key) ? 'date' : field.type,
        required:!['end','source_native_run_id'].includes(key),
        unit:field.unit || (['max_stock_weight','risk_budget','position_step','hysteresis'].includes(key) ? '比例' : '')};
    });
    if (!state.loaded || !state.dirty) {
      state.revision = profiles.default_profile?.revision ?? profiles.default_revision ?? 0;
      const saved = profiles.default_profile?.archived ? {} : profiles.default_profile?.parameters || {}, values = {...(capabilities.defaults || capabilities.builtin_profile?.parameters || {}),...saved};
      state.parameters = Object.fromEntries(state.fields.map(field => [keyOf(field),values[keyOf(field)] ?? field.default ?? null]));
      renderParameters(state.parameters,!state.runDirty);
    }
    state.loaded = true; renderConstraints(); renderProfiles(); renderStatus();
    for (const job of status.jobs || []) if (job.job_id) state.jobs.set(job.job_id,job);
    const remembered = localStorage.getItem('khquant_current_training_job');
    if (remembered && !state.jobs.has(remembered)) { try { state.jobs.set(remembered,await api(`/api/tasks/${encodeURIComponent(remembered)}`)); } catch (error) { fail(error); } }
    setBusy(false); renderJobs(); await events(); await pollJobs();
  }
  async function preflight(event) {
    event?.preventDefault(); clearErrors(); state.busyAction = 'preflight'; setBusy(true);
    try { const parameters = readParameters('training-form',state.parameters); const result = await api('/api/training/preflight',{method:'POST',body:JSON.stringify({parameters})}); renderPreflight(result); state.planFingerprint = fingerprint(parameters); }
    catch (error) { invalidatePlan('preflight_failed'); fail(error); }
    finally { setBusy(false); }
  }
  async function start() {
    clearErrors();
    try {
      const parameters = readParameters('training-form',state.parameters);
      if (!state.plan?.plan_id || state.plan.valid !== true || fingerprint(parameters) !== state.planFingerprint) { invalidatePlan(); throw new Error('训练参数或计划已变更，请重新预检。'); }
      state.busyAction = 'submit'; setBusy(true); const job = await api('/api/training/jobs',{method:'POST',body:JSON.stringify({plan_id:state.plan.plan_id,parameters})});
      if (!job.job_id) throw new Error('服务未返回任务标识，不能确认训练已提交。');
      state.jobs.set(job.job_id,job); localStorage.setItem('khquant_current_training_job',job.job_id); invalidatePlan('submitted',`任务已提交（任务号：${job.job_id}）。请展开训练与复核任务查看详情；新的训练需重新预检。`); renderJobs(); await pollJobs();
    } catch (error) { if (state.busyAction === 'submit' && state.planState !== 'submitted') invalidatePlan('submission_failed'); fail(error); }
    finally { setBusy(false); }
  }
  async function save(event) {
    event.preventDefault(); clearErrors(); state.busyAction = 'save'; setBusy(true);
    try { const parameters = readParameters('parameters-form',state.parameters); await api('/api/training/default-profile',{method:'PUT',body:JSON.stringify({expected_revision:state.revision,parameters})}); state.dirty = false; state.runDirty = false; invalidatePlan('defaults_saved'); $('parameters-save-status').textContent = '已保存。以后新训练采用此默认配置，已有任务和模型不变。'; }
    catch (error) { fail(error,'parameters'); }
    finally { setBusy(false); }
    if (!state.dirty) await refresh().catch(error => fail(error,'parameters'));
  }
  function childrenHtml(children) {
    if (!children || !Object.keys(children).length) return '';
    const labels = {raw_source_run_id:'原始来源',source_native_run_id:'原生来源',native_source_run_id:'原生来源',model_run_id:'模型训练'};
    return `<details data-job-detail="children"><summary>已完成阶段与来源运行</summary><ul>${Object.entries(children).map(([key,value]) => `<li>${esc(labels[key] || key)}：${esc(message(value))}</li>`).join('')}</ul></details>`;
  }
  function diagnosticsHtml(diagnostics) {
    if (!diagnostics) return '';
    return `<section class="training-diagnostic"><p><strong>样本诊断：${diagnostics.status === 'completed' ? '已完成核验' : '未完成，替换保持阻断'}</strong>${diagnostics.run_id ? `<br>诊断运行 ${esc(diagnostics.run_id)}` : ''}</p>${diagnostics.error || diagnostics.reason ? `<p class="training-blockers">${esc(message(diagnostics.error || diagnostics.reason))}</p>` : ''}${diagnostics.status === 'completed' && diagnostics.run_id ? `<button type="button" data-training-diagnostics="${esc(diagnostics.run_id)}" data-diagnostic-sha256="${esc(diagnostics.report_sha256 || '')}">查看样本流失诊断</button>` : ''}${diagnostics.report_paths ? `<details data-job-detail="diagnostics"><summary>查看诊断报告位置与凭据</summary><ul>${Object.entries(diagnostics.report_paths).map(([key,path]) => `<li>${esc(({json:'逐层统计JSON',markdown:'中文诊断报告'})[key] || key)}：<code>${esc(path)}</code></li>`).join('')}</ul></details>` : ''}</section>`;
  }
  const diagnosticReasons = {
    decision_segment:'本段信号日候选',selected:'实际入选',outside_decision_segment:'信号日不在本段',
    label_unavailable:'标签不可用',input_ineligible:'输入资格失败',invalid_binary_label:'非有效二分类标签',
    label_end_missing:'缺清仓成熟日',family_first_outside_segment:'原机会首次尝试不在本段',
    label_end_after_cutoff:'清仓晚于本段截止',atomic_dependency_outside_segment:'原机会原子跨段依赖',
    entry_unfilled:'Broker 买入未成交',episode_unclosed:'实际持仓未清仓',
    price_basis_or_corporate_action_ambiguous:'价基或公司行动有歧义',risk_identity_unavailable:'初始风险身份不合格',
    net_nonpositive:'费用后未盈利',net_below_target:'盈利但净收益未达标',r_below_target:'净收益达标但 R 不足',high:'净收益与 R 均达标',
    no_direction_observation:'缺有效当日方向输入',no_active_buy_opportunity:'无有效原买方机会',buy_base_not_qualified:'买方基础共识未通过',
    entry_envelope_not_eligible:'风险或机会入场资格未通过',account_state_or_reservation_block:'持仓、待执行或机会预留阻断',
    risk_plan_rejected:'真实空仓风险计划被拒',pre_gate_attempt:'质量评分前克隆人口',
    buy_base_below_threshold:'买方基础支持不足',independent_sources_insufficient:'独立来源不足',buy_sell_margin_insufficient:'买卖支持差不足',
    signal_pinned_ma_exit:'入场时固定 MA 退出头',actual_account_stop:'真实账户止损',frozen_stop:'冻结止损边界',
    signal_pinned_consensus_exit:'持续卖方共识退出',holding_session_limit:'冻结持仓时限'
  };
  const diagnosticReason = key => diagnosticReasons[key] || globalThis.KHQuantChartEvidence?.opinionReasons({reason_codes:[key]}) || key;
  const count = value => typeof value === 'number' && Number.isFinite(value) ? String(value) : '—';
  const ratio = value => typeof value === 'number' && Number.isFinite(value) ? (value * 100).toFixed(2) + '%' : '—';
  function countTable(values, title) {
    const rows = Object.entries(values || {}).sort((a,b) => Number(b[1]) - Number(a[1]));
    return `<details><summary>${esc(title)}</summary>${rows.length ? `<div class="table-wrap"><table><thead><tr><th>互斥原因</th><th>数量</th></tr></thead><tbody>${rows.map(([key,value]) => `<tr><td>${esc(diagnosticReason(key))}</td><td>${count(value)}</td></tr>`).join('')}</tbody></table></div>` : '<p class="subtle">本段未记录此项。</p>'}</details>`;
  }
  function segmentDiagnosticHtml(segment, path, name, rules) {
    const target = segment.target || {}, selected = segment.selected || {}, ledger = segment.ledger_audit || {}, quality = segment.reference_quality || {};
    const missing = segment.feature_observations?.rows_with_missing_features;
    const sourceCounts = Object.entries(segment.source_input_availability || {}).map(([key,value]) => `${({ma:'均线',structure:'结构',wyckoff:'量价'})[key] || key}可用 ${count(value.available)}／不可用 ${count(value.unavailable)}`).join('；');
    return `<details class="diagnostic-segment" ${name === 'calibration' ? 'open' : ''}><summary>${({train:'训练段',calibration:'校准段',policy:'政策段'})[name] || esc(name)} · ${esc(segment.start || '来源起点')}—${esc(segment.end)} · 入选 ${count(selected.rows)} 行／正类 ${count(selected.positive)}</summary><div class="diagnostic-metrics"><p>信号日候选 <strong>${count(target.candidates)}</strong> → 有效标签 <strong>${count(target.available_labels)}</strong> → 费用后盈利 <strong>${count(target.fee_net_profitable)}</strong>；净收益 ≥ ${ratio(rules.min_net_return ?? .08)}：<strong>${count(target.net_return_target_pass)}</strong>，R ≥ ${count(rules.min_r_multiple ?? 2)}：<strong>${count(target.r_multiple_target_pass)}</strong>，同时达标 <strong>${count(target.both_target_pass)}</strong>。</p><p>最终入选 ${count(selected.rows)} 行：正 ${count(selected.positive)}／负 ${count(selected.negative)}／未知 ${count(selected.unknown)}。有效标签统计截至冻结来源已清仓；段截止前成熟另见逐层清洗。</p></div><div class="table-wrap"><table><thead><tr><th>逐层输入与时间清洗（按顺序归因）</th><th>保留行</th><th>保留正类</th><th>本步排除</th><th>排除正类</th></tr></thead><tbody>${(segment.funnel || []).map(stage => `<tr><td>${esc(diagnosticReason(stage.stage))}</td><td>${count(stage.rows)}</td><td>${count(stage.positive)}</td><td>${count(stage.removed)}</td><td>${count(stage.positive_removed ?? 0)}</td></tr>`).join('')}</tbody></table></div>${countTable(path?.exclusive_first_block,'股票日候选首个阻断原因')}${countTable(target.exclusive_target_reasons,'标签及目标首个未达原因')}${countTable(path?.base_gate_exclusive_details,'基础共识首次失败细分')}${countTable(ledger.last_exit_reasons,'实际清仓原因（按数量排序）')}<p class="training-note">费用总额 ${count(target.fees_total)} 元。保持原成交滑点与冻结分母，仅加回费用后同时达标 ${count(target.fee_excluded_same_denominator_high)}，费用边缘损失 ${count(target.fee_only_high_to_negative)}；再加回观测滑点后同时达标 ${count(ledger.fee_and_slippage_excluded_same_denominator_high)}。这些是费用观察，不产生新标签。</p><p>初始风险中位 ${ratio(target.risk_fraction_quantiles?.['0.5'])}，联合目标所需净收益中位 ${ratio(target.required_net_return_quantiles?.['0.5'])}。持仓交易日距离中位 ${count(ledger.holding_session_distance_quantiles?.['0.5'])}，按退出日索引减入场日索引计算。</p><p>${quality.quality_rejection_removes_training_candidate === false ? '质量门槛拒绝没有直接删除强制入场克隆。' : '质量门槛与克隆保留关系未确认，需核验明细。'}参考质量拒绝后仍保留高收益标签 ${count(quality.high_labels_retained_despite_reference_rejection)}。缺失特征观测 ${count(missing)} 行，沿用训练段拟合的填补，不新增过滤。</p>${countTable(quality.reason,'参考账户质量状态原因')}<p class="subtle">${sourceCounts ? `来源输入：${esc(sourceCounts)}。候选仅要求所属来源至少一项可用。` : ''}经济连通分量 ${count(segment.economic_components)}；分量内候选不能视作独立机会。</p></details>`;
  }
  function sampleDiagnosticsHtml(report) {
    const folds = report.folds || [], rules = report.rules?.high_return || {};
    const rows = folds.map(item => { const segment = item.segments?.calibration || {}, target = segment.target || {}, selected = segment.selected || {}; return `<tr><td>${esc(item.fold?.name)}</td><td>${count(target.candidates)}</td><td>${count(target.available_labels)}</td><td>${count(target.fee_net_profitable)}</td><td>${count(target.net_return_target_pass)}</td><td>${count(target.r_multiple_target_pass)}</td><td>${count(target.both_target_pass)}</td><td>${count(selected.rows)}／${count(selected.positive)}</td></tr>`; }).join('');
    const details = folds.map(item => {
      const cal = item.segments?.calibration || {}, target = cal.target || {}, selected = cal.selected || {}, qualification = item.qualification || {};
      const cause = target.available_labels === 0 ? '标签结果不可用：先核查未成交、未清仓、价基及风险身份，当前不足以归因于收益目标。' : target.both_target_pass === 0 ? '目标是本折校准正类不足的直接原因：有效标签在输入与时间清洗前已无净收益和 R 同时达标正类。' : target.both_target_pass > selected.positive ? '门槛造成正类损失：清洗前已有正类，逐层保留正类与排除正类说明具体流失位置。' : '本折仍有正类，请继续核验最低行数、双类门槛与经济依赖。';
      const rowGate = qualification.row_gate_passed === true ? '最低行数通过' : `最低行数未通过（训练 ≥ ${count(qualification.min_train_rows)}、校准 ≥ ${count(qualification.min_calibration_rows)}）`;
      return `<details class="diagnostic-fold" ${item.fold?.name === 'production' ? 'open' : ''}><summary>${esc(item.fold?.name)} · ${qualification.passed === true ? '样本资格通过' : '样本资格阻断'}</summary><p class="diagnostic-cause"><strong>${esc(cause)}</strong></p><p>${esc(rowGate)}；${qualification.both_classes_passed === true ? '训练与校准均有两类' : '训练或校准缺少正／负类'}。目标与交易门槛保持冻结值。</p>${['calibration','train','policy'].filter(name => item.segments?.[name]).map(name => segmentDiagnosticHtml(item.segments[name],item.daily_candidate_path?.[name],name,rules)).join('')}</details>`;
    }).join('');
    return `<h2>高收益样本流失诊断</h2><p>诊断 ${esc(report.run_id)} · 模型 ${esc(report.model_run_id)} · 冻结数据截至 ${esc(report.data_end)} · ${count(report.cohort_size)} 股</p><p class="training-note">每折人口分别统计，折间有重叠，不能相加。统计读取冻结 Broker 克隆，费用复算与最终 mask 核验，不训练、不降低目标或资格。观察不能证明退出头或全部行情数据有错误。</p><p>${report.artifact_integrity_passed === true && report.source_report_unchanged === true ? '资产校验、费用与结果复算通过，源报告未变。' : '完整性证明未齐备，请查看 JSON 核验。'}</p><div class="table-wrap"><table><thead><tr><th>折（校准段）</th><th>候选</th><th>有效标签</th><th>费用后盈利</th><th>≥ ${ratio(rules.min_net_return ?? .08)}</th><th>≥ ${count(rules.min_r_multiple ?? 2)}R</th><th>同时达标</th><th>入选行／正类</th></tr></thead><tbody>${rows}</tbody></table></div>${details}<details><summary>完整诊断 JSON、输入输出哈希与逐行审计路径</summary><pre>${esc(JSON.stringify(report,null,2))}</pre></details>`;
  }
  async function showDiagnostics(runId, expectedSha = '') {
    const sequence = ++state.diagnosticSequence, host = $('training-sample-diagnostics'); host.hidden = false; host.textContent = '正在读取并核验样本诊断…';
    try {
      const response = await api(`/api/training/diagnostics/${encodeURIComponent(runId)}${expectedSha ? '?expected_sha256=' + encodeURIComponent(expectedSha) : ''}`);
      if (sequence !== state.diagnosticSequence) return;
      if (response.run_id !== runId) throw new Error('诊断返回身份与所选运行不一致。');
      host.innerHTML = sampleDiagnosticsHtml(response); location.hash = 'training'; host.scrollIntoView?.({block:'start',behavior:'smooth'});
    } catch (error) { if (sequence === state.diagnosticSequence) host.textContent = '诊断读取或身份核验失败：' + (error.message || message(error)); throw error; }
  }
  async function showComparison(runId) {
    const comparison = await api(`/api/training/runs/${encodeURIComponent(runId)}/comparison`), reasons = comparison.reasons || [];
    if (comparison.strategy_version === 'reference_position_v1' || comparison.family === 'reference_position_v1' || comparison.technical_validation || comparison.predictive_gate || comparison.predictive_validation || comparison.technical || comparison.predictive) {
      $('training-selection').innerHTML = `<h3>参考模型验证与预测基线</h3>${referenceValidationHtml(comparison)}<pre>${esc(JSON.stringify(comparison,null,2))}</pre>`;
      location.hash = 'training'; return;
    }
    const arms = {candidate:'候选',incumbent:'现任',rules:'规则基线',candidate_stress:'候选 · 双倍费用滑点',incumbent_stress:'现任 · 双倍费用滑点',rules_stress:'基线 · 双倍费用滑点'};
    const pct = value => typeof value === 'number' && Number.isFinite(value) ? (value * 100).toFixed(2) + '%' : '—';
    $('training-selection').innerHTML = `<h3>费用后账本对比 · ${esc(statuses[comparison.status] || comparison.status)}</h3><p class="training-note">固定等额独立账户，失败股票保留现金。开发比较不代表独立认证。</p>${reasons.length ? `<ul class="training-blockers">${reasons.map(reason => `<li>${esc(message(reason))}</li>`).join('')}</ul>` : ''}${comparison.windows?.length ? `<div class="table-wrap"><table><thead><tr><th>窗口</th><th>策略</th><th>费用后收益</th><th>最大回撤</th><th>完整交易</th><th>交易期望</th></tr></thead><tbody>${comparison.windows.flatMap(window => Object.entries(window.arms || {}).map(([key,arm]) => { const metrics = arm.metrics || {}; return `<tr><td>${esc(window.window?.name || window.name || '')}</td><td>${esc(arms[key] || key)}</td><td>${pct(metrics.total_return)}</td><td>${pct(metrics.drawdown)}</td><td>${esc(metrics.completed_round_trips ?? '—')}</td><td>${pct(metrics.expectancy)}</td></tr>`; })).join('')}</tbody></table></div>` : '<p class="subtle">关键头、类别或证据未齐备，尚不能进行账本对比。</p>'}`;
    location.hash = 'training';
  }
  async function showResult(result, task = null) {
    const reasons = result.selection?.reasons || result.blockers || [];
    const receipt = task || {run_id:result.run_id,status:'succeeded',result};
    $('training-jobs').innerHTML = `<article class="training-job"><h3>${jobTitle(receipt)}结果 · ${esc(statuses[result.selection?.status] || result.selection?.status || '资格待核验')}</h3><p>任务运行 ${esc(result.run_id || '未提供')}</p>${result.training_started === false && result.weights_unchanged === true ? '<p>本次没有启动训练，模型权重保持不变；原任务与历史诊断保留。</p>' : ''}${result.training_run_id ? `<p>模型运行 ${esc(result.training_run_id)}</p><button type="button" data-training-comparison="${esc(result.training_run_id)}">查看验证与基线</button>` : ''}${populationHtml(result)}${result.strategy_version === "reference_position_v1" ? referenceValidationHtml(result) : referenceQualityTrainingHtml(result.reference_quality,result.result_reasons)}${validationButton(receipt)}${reasons.length ? `<ul class="training-blockers">${reasons.map(reason => `<li>${esc(message(reason))}</li>`).join('')}</ul>` : ''}${childrenHtml(result.children)}${diagnosticsHtml(result.diagnostics)}${result.required_heads ? `<details data-job-detail="heads"><summary>必需头资格</summary><pre>${esc(JSON.stringify(result.required_heads,null,2))}</pre></details>` : ''}<details><summary>完整任务证据</summary><pre>${esc(JSON.stringify(result,null,2))}</pre></details></article>`;
    location.hash = 'training';
  }
  function bind() {
    if (!$('training-form')) return;
    $('training-task-panel').addEventListener('toggle',() => { const label = $('training-task-panel').querySelector('.training-task-toggle'); if (label) label.textContent = $('training-task-panel').open ? '收起任务详情' : '展开任务详情'; scheduleTaskLayout(); });
    if (globalThis.ResizeObserver) { state.layoutObserver = new ResizeObserver(scheduleTaskLayout); ['training-side','training-request-panel','training-task-heading'].forEach(id => state.layoutObserver.observe($(id))); }
    globalThis.addEventListener?.('resize',scheduleTaskLayout);
    $('training-form').addEventListener('submit',preflight); $('training-start').addEventListener('click',start);
    ['input','change'].forEach(type => $('training-form').addEventListener(type,() => { state.runDirty = true; invalidatePlan(); renderResourceControls(); }));
    $('training-device-suggest').addEventListener('click',applyDeviceSuggestion);
    $('training-resource-guide').addEventListener('toggle',scheduleTaskLayout);
    $('parameters-form').addEventListener('submit',save); $('parameters-form').addEventListener('input',() => { state.dirty = true; $('parameters-save-status').textContent = '表单有未保存修改。'; });
    $('parameters-form').addEventListener('change',() => { state.dirty = true; });
    $('parameters-reset').addEventListener('click',() => { renderParameters(state.capabilities.defaults || state.capabilities.builtin_profile?.parameters || {},false); state.dirty = true; $('parameters-save-status').textContent = '已载入内置值，点击保存后才成为以后训练默认。'; });
    $('training-refresh').addEventListener('click',() => refresh().catch(fail)); $('training-events-refresh').addEventListener('click',() => events().catch(fail));
    $('training-jobs').addEventListener('click',async event => {
      const cancel = event.target.closest('[data-training-cancel]'), result = event.target.closest('[data-training-result]'), comparison = event.target.closest('[data-training-comparison]'), diagnostics = event.target.closest('[data-training-diagnostics]'), validation = event.target.closest('[data-training-validate]'), resume = event.target.closest('[data-training-resume]');
      try { if (resume) { const job = state.jobs.get(resume.dataset.trainingResume); if (!job) throw new Error('恢复任务凭据不可用，请刷新。'); await resumeNative(job); } else if (cancel) { const job = await api(`/api/tasks/${encodeURIComponent(cancel.dataset.trainingCancel)}/cancel`,{method:'POST',body:'{}'}); state.jobs.set(job.job_id,job); await pollJobs(); } else if (diagnostics) { await showDiagnostics(diagnostics.dataset.trainingDiagnostics,diagnostics.dataset.diagnosticSha256); } else if (validation && !validation.disabled) { const job = state.jobs.get(validation.dataset.trainingValidate) || state.validationReceipts.get(validation.dataset.trainingValidate); if (!job) throw new Error('所选任务凭据已不可用，请刷新。'); await revalidate(job); } else if (comparison) { await showComparison(comparison.dataset.trainingComparison); } else if (result) { const job = await api(`/api/tasks/${encodeURIComponent(result.dataset.trainingResult)}`); await showResult(job.result || job,job); } } catch (error) { fail(error); }
    });
  }
  globalThis.KHQuantTraining = {refresh,showResult};
  bind();
})();
