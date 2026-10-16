/* Read-only rendering of saved method evidence. No inference, qualification or data writes. */
(function (root) {
  'use strict';
  const labels = {
    structure: '结构与量价', structure_kronos: '结构与量价 + Kronos',
    structure_kronos_ma_mlp: '结构与量价 + Kronos + 均线模型 MLP',
    structure_kronos_ma_l2: '结构与量价 + Kronos + 同输入 L2',
    legacy_reference: '历史参考模型（对照）'
  };
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const object = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const integer = value => Number.isSafeInteger(value) && value >= 0;
  const number = value => finite(value) ? value.toLocaleString('zh-CN', {maximumFractionDigits: 2}) : '—';
  const percent = value => finite(value) && finite(value * 100) ? `${number(value * 100)}%` : '—';
  const fraction = value => finite(value) && value >= 0 && value <= 1;
  const scalar = value => ['string', 'number'].includes(typeof value) ? String(value) : '—';
  const boolState = value => value === true ? 'pass' : value === false ? 'fail' : 'unknown';
  const stateText = {pass: '已通过', fail: '未通过', unknown: '待核验', na: '未使用'};
  const chip = (state, text = stateText[state]) => `<span class="amv-state amv-state-${state}">${esc(text)}</span>`;
  const row = (name, state, detail, label) => `<div class="amv-check"><dt>${esc(name)}</dt><dd>${chip(state, label)}<span>${esc(detail)}</span></dd></div>`;

  function sampleInfo(quality) {
    const valid = quality.valid_samples, requested = quality.requested_samples;
    const known = integer(valid) && integer(requested) && requested > 0 && valid <= requested;
    const ratio = known ? valid / requested : null;
    return {valid, requested, known, ratio, passed: known && requested >= 8 && valid >= 8 && ratio >= .8};
  }

  function meter(value, label, kind = '') {
    if (!fraction(value)) return `<div class="amv-meter-unavailable">${esc(label)}：未记录</div>`;
    const width = (value * 100).toFixed(4);
    return `<div class="amv-meter ${kind}" role="img" aria-label="${esc(label)} ${esc(percent(value))}"><span style="width:${width}%"></span></div>`;
  }

  function quantileHtml(summary) {
    const low = summary.terminal_return_p10, middle = summary.terminal_return_median, high = summary.terminal_return_p90;
    if (![low, middle, high].every(finite) || low > middle || middle > high) {
      return '<p class="amv-empty">分位数区间未完整记录，无法绘制。</p>';
    }
    // Normalize first so finite extreme inputs cannot overflow the geometry.
    const scale = Math.max(Math.abs(low), Math.abs(high), 1);
    const min = Math.min(low / scale, 0), max = Math.max(high / scale, 0), span = max - min;
    const x = value => span > 0 ? 20 + ((value / scale - min) / span) * 360 : 200;
    const description = `Kronos 采样涨跌分位数：P10 ${percent(low)}，中位涨跌 ${percent(middle)}，P90 ${percent(high)}。`;
    return `<div class="amv-quantile"><svg viewBox="0 0 400 44" role="img" aria-label="${esc(description)}"><line class="amv-axis" x1="20" x2="380" y1="22" y2="22"/><line class="amv-zero" x1="${x(0).toFixed(3)}" x2="${x(0).toFixed(3)}" y1="8" y2="36"/><line class="amv-interval" x1="${x(low).toFixed(3)}" x2="${x(high).toFixed(3)}" y1="22" y2="22"/><circle class="amv-bound" cx="${x(low).toFixed(3)}" cy="22" r="4"/><circle class="amv-bound" cx="${x(high).toFixed(3)}" cy="22" r="4"/><circle class="amv-median" cx="${x(middle).toFixed(3)}" cy="22" r="6"/></svg><div class="amv-quantile-labels"><span>P10 <strong>${percent(low)}</strong></span><span>中位涨跌 <strong>${percent(middle)}</strong></span><span>P90 <strong>${percent(high)}</strong></span></div></div>`;
  }

  function rejectionHtml(quality) {
    const rejected = object(quality.rejected_by_rule);
    const rules = [['nonfinite', '数值不是有限值'], ['price', '价格不是正数'],
      ['negative_activity', '成交量或成交额为负'], ['ohlc_geometry', 'OHLC 高低价关系不成立']];
    const items = rules.filter(([key]) => integer(rejected[key]) && rejected[key] > 0);
    if (!items.length) return '';
    return `<div class="amv-rejections"><h5>路径剔除原因</h5><ul>${items.map(([key, name]) => `<li><span>${name}</span><strong>${number(rejected[key])} 条</strong></li>`).join('')}</ul>${items.length > 1 ? '<p class="subtle">同一路径可能触发多项检查，各项数量不相加。</p>' : ''}</div>`;
  }

  function forecastHtml(kronos, view, compact) {
    const summary = object(kronos.summary), quality = object(kronos.quality), samples = sampleInfo(quality);
    const valid = view?.forecast_valid === true;
    const qualityState = view ? boolState(view.forecast_valid) : quality.prediction_usable === false ? 'fail' : 'unknown';
    const count = samples.known ? `${number(samples.valid)} / ${number(samples.requested)}` : '未记录';
    return `<section class="amv-card amv-forecast"><div class="amv-card-heading"><h4>Kronos 预测采样</h4>${chip(qualityState, valid ? '预测可用' : qualityState === 'fail' ? '质量未通过' : '核验待确认')}</div>${!valid ? '<p class="amv-raw-note">以下为原始有效采样的摘要；尚不能用于组合判断。</p>' : ''}${quantileHtml(summary)}<div class="amv-stat-grid"><div><span>上涨采样比例</span><strong>${fraction(summary.up_path_fraction) ? percent(summary.up_path_fraction) : '—'}</strong>${meter(summary.up_path_fraction, '上涨采样比例', 'amv-up')}</div><div><span>有效路径</span><strong>${count}</strong>${meter(samples.ratio, '有效路径比例', samples.passed ? '' : 'amv-low')}<small>${samples.known ? `${percent(samples.ratio)} · ` : ''}门槛：至少 8 条且 ≥ 80%</small></div></div>${!compact ? rejectionHtml(quality) : ''}<p class="subtle">采样分位数与上涨比例未经校准；预测涨跌不等于实际收益。</p></section>`;
  }

  function scoreHtml(value, view) {
    const score = object(value.score), scoredMethod = /_ma_(mlp|l2)$/.test(value.method || '');
    if (!scoredMethod) return '';
    const rawScore = finite(score.score) ? score.score : Array.isArray(score.scores) && finite(score.scores[0]) ? score.scores[0] : null;
    const scoreValue = rawScore !== null && rawScore >= 0 && rawScore <= 100 ? rawScore : null;
    const valid = view?.score_valid === true;
    const name = value.method === 'structure_kronos_ma_l2' || score.model_type === 'l2' ? 'L2' : 'MLP';
    const probabilities = Array.isArray(score.probabilities) ? score.probabilities : ['down', 'flat', 'up'].map(key => object(score.probabilities)[key]);
    const probabilitiesValid = probabilities.length === 3 && probabilities.every(fraction) && Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) <= 1e-6;
    const label = scoreValue === null ? '评分不可用' : `${name} ${score.status === 'scored' ? '已评分' : '原评分'} ${number(scoreValue)} / 100`;
    return `<section class="amv-card"><div class="amv-card-heading"><h4>均线${esc(name)}评分</h4>${chip(view ? boolState(view.score_valid) : 'unknown')}</div><p class="amv-score-value">${esc(label)}</p>${scoreValue !== null ? meter(scoreValue / 100, `${name}评分`) : ''}${scoreValue !== null && !valid ? '<p class="amv-raw-note">显示原评分值；是否可用于组合判断，以核验结果为准。</p>' : ''}${probabilitiesValid ? `<div class="amv-probabilities">${['下跌', '平稳', '上涨'].map((label, i) => `<div><span>${label} ${percent(probabilities[i])}</span>${meter(probabilities[i], label, ['amv-down', 'amv-flat', 'amv-up'][i])}</div>`).join('')}</div>` : '<p class="amv-empty">三分类概率未完整记录。</p>'}</section>`;
  }

  function qualificationHtml(value, view) {
    const quality = object(object(value.kronos).quality), samples = sampleInfo(quality), score = object(value.score);
    const scoredMethod = /_ma_(mlp|l2)$/.test(value.method || ''), qualification = object(score.qualification);
    const hasQualification = Object.keys(qualification).length > 0;
    const combined = view?.valid === true && view.status === 'complete';
    const purpose = qualification.mode === 'historical_candidate' ? '历史研究候选' : qualification.mode === 'paper' ? '自动模拟用途' : qualification.research_only === true ? '仅研究用途' : '用途未记录';
    const forward = ({pending: '独立前瞻资格待验证', qualified: '独立前瞻资格已通过', failed: '独立前瞻资格未通过'})[qualification.independent_forward_status] || '独立前瞻资格未记录';
    const sampleState = samples.known ? samples.passed ? 'pass' : 'fail' : 'unknown';
    let rows = row('结构与量价来源', 'na', value.source_identity ? '独立来源已记录；与组合方向分别展示' : '来源依据在主图中独立保留', '独立保留');
    rows += row('有效路径数量与占比', sampleState, samples.known ? `${number(samples.valid)} / ${number(samples.requested)}（${percent(samples.ratio)}）；至少 8 条且 ≥ 80%` : '路径计数未记录；至少 8 条且 ≥ 80%');
    rows += row('行情来源核验', boolState(quality.source_verified), quality.source_verified === true ? '使用已核验来源' : quality.source_verified === false ? '来源核验未通过' : '未记录核验结果');
    rows += row('交易日历核验', boolState(quality.calendar_verified), quality.calendar_verified === true ? '核验日历一致' : quality.calendar_verified === false ? '日历核验未通过' : '未记录核验结果');
    rows += row('模型输入契约', boolState(quality.input_contract_verified), quality.input_contract_verified === true ? '输入契约已核验' : quality.input_contract_verified === false ? '输入契约未通过' : '未记录核验结果');
    rows += row('预测整体可用性', view ? boolState(view.forecast_valid) : 'unknown', view?.forecast_valid === true ? '预测摘要、质量与参数核验通过' : view?.forecast_valid === false ? '预测质量或身份未通过，原摘要仅供查看' : '组合视图尚未核验');
    rows += row('均线评分', scoredMethod ? view ? boolState(view.score_valid) : 'unknown' : 'na', scoredMethod ? view?.score_valid === true ? '评分、输入与模型身份核验通过' : '评分或模型身份仍需核验' : '当前方法不需要评分模型');
    rows += row('组合判断', combined ? 'pass' : view ? ['failed', 'unavailable'].includes(view.status) ? 'fail' : 'unknown' : 'unknown', combined ? '全部必要证据通过，可展示截止日方向' : '必要证据未全部通过，暂不生成组合方向');
    if (hasQualification) {
      rows += row('评分用途', qualification.mode === 'historical_candidate' || qualification.mode === 'paper' ? 'pass' : 'unknown', `${purpose}；${forward}`);
      rows += row('自动模拟配置资格', boolState(qualification.paper_allocation_eligible), qualification.paper_allocation_eligible === true ? '已记录自动模拟配置资格' : qualification.paper_allocation_eligible === false ? '尚未取得自动模拟配置资格' : '未记录资格结果');
      rows += row('独立前瞻资格', ({qualified:'pass', failed:'fail', pending:'unknown'})[qualification.independent_forward_status] || 'unknown', forward);
    }
    const modelId = typeof value.model_id === 'string' ? value.model_id : null;
    return `<details class="amv-card amv-qualification"><summary>可用性与资格核验</summary><div class="amv-qualification-body"><dl class="amv-checks">${rows}</dl>${modelId ? `<details class="amv-identity"><summary>模型身份</summary><dl><dt>评分模型</dt><dd>${esc(modelId)}</dd>${value.feature_profile ? `<dt>输入</dt><dd>${esc(value.feature_profile === 'ma_trend_v1' ? '均线趋势输入' : scalar(value.feature_profile))}</dd>` : ''}</dl></details>` : ''}</div></details>`;
  }

  function evaluationHtml(value, compact = false, view = null) {
    if (!value || typeof value !== 'object') return '';
    const combined = view?.valid === true && view.status === 'complete';
    const score = object(value.score), kronos = object(value.kronos), samples = sampleInfo(object(kronos.quality));
    const direction = combined ? ({BUY:'B · 方法买方参考', SELL:'S · 方法卖方参考', WAIT:'WAIT · 有效观察'})[view.direction] || '方向待核验' : ({deferred:'预算延后', failed:'评估失败'})[view?.status || value.status] || (view ? '组合判断未通过' : '组合判断待核验');
    const evidenceStatus = ({complete:'证据已计算', partial:'部分证据可用', deferred:'预算延后', failed:'评估失败'})[value.status] || '证据待核验';
    let explanation = '';
    if (!combined && (value.status === 'partial' || view?.forecast_valid === false)) {
      explanation = samples.known && !samples.passed
        ? `结构与量价来源独立保留；Kronos 仅 ${number(samples.valid)} / ${number(samples.requested)} 条路径有效（${percent(samples.ratio)}），未达到至少 8 条且 80% 的门槛，因此暂不生成组合方向。`
        : '结构与量价来源独立保留；预测、评分或身份核验仍有缺失，因此暂不生成组合方向。';
    }
    const model = typeof value.model_id === 'string' ? ` · 模型 ${value.model_id.slice(0, 12)}${value.model_id.length > 12 ? '…' : ''}` : /_ma_(mlp|l2)$/.test(value.method || '') ? ' · 未绑定评分候选' : '';
    const directionState = combined ? 'pass' : view?.status === 'unavailable' || view?.status === 'failed' ? 'fail' : 'unknown';
    const head = `<div class="method-summary"><strong>${esc(scalar(value.label || labels[value.method] || '未知方法'))}</strong>${chip(combined ? 'pass' : 'unknown', evidenceStatus)}${chip(directionState, direction)}</div>${compact ? '' : `<p class="subtle amv-meta">信息截止 ${esc(scalar(value.as_of))}${esc(model)}</p>`}${explanation ? `<p class="amv-explanation">${esc(explanation)}</p>` : ''}`;
    return `<div class="amv${compact ? ' amv-compact' : ''}">${head}<div class="amv-grid">${forecastHtml(kronos, view, compact)}${scoreHtml(value, view)}</div>${compact ? '' : qualificationHtml(value, view)}</div>`;
  }

  const api = {evaluationHtml};
  root.KHQuantAnalysisMethodView = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
