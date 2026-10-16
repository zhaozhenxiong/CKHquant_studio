/* Saved market evidence only. Expanding a row reads its original prediction. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  const icon = name => `<img class="market-icon" src="/static/vendor/bootstrap-icons/${name}.svg" alt="" aria-hidden="true">`;
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const number = value => finite(value) ? value.toLocaleString('zh-CN', {maximumFractionDigits: 2}) : '—';
  const validDay = value => typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value);
  const list = value => Array.isArray(value) ? value : [];
  const state = {rows: [], options: {}, scope: null, content: null, expanded: null, chart: null, generation: 0};
  const bound = new WeakSet();

  function cutoff(row, result) {
    const value = row.method_view?.as_of || row.data_end || result?.as_of || result?.data_end;
    return validDay(value) ? value : null;
  }

  function referencePrice(row, result) {
    const date = cutoff(row, result);
    // Match the saved scan's verified_local_close contract. A forecast, earlier
    // close, numeric string or source-event price is not a cutoff closing price.
    if (!date || row.data_end !== date || row.reference_price_date !== date ||
        (result?.as_of && result.as_of !== date) || (result?.data_end && result.data_end !== date) ||
        row.reference_price_source !== 'verified_local_close' ||
        !finite(row.reference_price) || row.reference_price <= 0 ||
        !finite(row.close) || row.close !== row.reference_price) return null;
    return row.reference_price;
  }

  function direction(row) {
    const view = row.method_view;
    if (view?.valid === true && view.status === 'complete') {
      return ({BUY: 'B · 方法买方参考', SELL: 'S · 方法卖方参考', WAIT: 'WAIT · 方法观察'})[view.direction] || '方法方向未核验';
    }
    return ({failed: '方法评估失败', deferred: '方法待处理', unavailable: '方法不可用', not_applicable: '来源参考'})[view?.status] || '方法方向未核验';
  }

  function predictionReference(row) {
    const view = row.method_view;
    return typeof view?.prediction_ref === 'string' && /^[a-f0-9]{64}$/.test(view.prediction_ref) &&
      validDay(view.as_of) && typeof row.symbol === 'string' ? view.prediction_ref : null;
  }

  function sourceHtml(row) {
    const points = list(row.source_reference_points).filter(point => point &&
      point.reference_layer === 'source' && (!point.symbol || point.symbol === row.symbol));
    if (!points.length) return '<p class="market-opportunity-muted">此行未保存来源 B/S 明细。</p>';
    return `<ul class="market-opportunity-sources">${points.map(point => {
      const action = point.action || point.side;
      const label = ({BUY: 'B · 买方参考', SELL: 'S · 卖方参考'})[action] || '方向未记录';
      return `<li><strong>${esc(point.source_label || point.source || '来源未记录')} · ${label}</strong><span>${esc(point.reference_price_date || point.date || '日期未记录')} · 首次可用 ${esc(point.first_available_at || point.available_at || '未记录')}</span></li>`;
    }).join('')}</ul>`;
  }

  function reasonsHtml(row) {
    const reasons = list(row.method_view?.reason_codes);
    if (!reasons.length) return '';
    return `<p class="market-opportunity-muted">${reasons.map(reason => esc(globalThis.KHQuantMethodPresentation?.reason(reason) || reason)).join('；')}</p>`;
  }

  function methodSummaryHtml(row) {
    const view = row.method_view || {}, summary = view.forecast_valid === true ? view.forecast_summary || {} : {};
    const percent = value => finite(value) ? `${number(value * 100)}%` : '未记录';
    return `<div class="market-opportunity-method-brief"><h3>已保存的方法依据</h3><strong>${direction(row)}</strong>
      ${view.forecast_valid === true ? `<dl><div><dt>预测中位涨跌</dt><dd>${percent(summary.terminal_return_median)}</dd></div><div><dt>上涨采样比例</dt><dd>${percent(summary.up_path_fraction)}</dd></div>${view.score_valid === true && finite(view.score_value) ? `<div><dt>所选模型评分</dt><dd>${number(view.score_value)} / 100</dd></div>` : ''}</dl>` : '<p class="market-opportunity-muted">预测摘要未通过核验。</p>'}
      <p class="market-opportunity-muted">采样比例未经校准，不是历史胜率。多方法独立同向票数未记录。</p></div>`;
  }

  function pricePlanHtml(row, result) {
    const price = referencePrice(row, result), day = cutoff(row, result);
    const positive = value => finite(value) && value > 0;
    const atCutoff = value => typeof value === 'string' &&
      /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value) &&
      Number.isFinite(Date.parse(value)) && Date.parse(value) <= Date.parse(`${day}T15:00:00+08:00`);
    const view = row.method_view || {};
    const methodBuy = view.valid === true && view.status === 'complete' && view.direction === 'BUY';
    const entry = row.entry_plan || row.raw_risk_plan;
    const reference = row.reference_risk_plan;
    // A stored reference risk band explicitly is not an executable entry band.
    // Never infer a limit, stop or profit target from forecast quantiles/returns.
    const entryValid = price != null && entry && entry.symbol === row.symbol &&
      entry.signal_date === day && atCutoff(entry.available_at) && entry.reference_price === price &&
      (!entry.as_of || entry.as_of === day) && (!entry.date || entry.date === day) &&
      (!entry.reference_price_date || entry.reference_price_date === day) &&
      (!entry.reference_price_source || ['signal_day_close', 'verified_local_close'].includes(entry.reference_price_source)) &&
      entry.status === 'active' && entry.entry_allowed === true && entry.policy === 'risk' &&
      entry.valid_for_sessions === 1 && typeof entry.plan_id === 'string' && entry.plan_id.length > 0 &&
      list(entry.source_event_ids).length > 0 && Array.isArray(entry.reason_codes) && !entry.reason_codes.length &&
      positive(entry.stop_price) && entry.stop_price < price && positive(entry.price_floor) &&
      positive(entry.price_ceiling) && entry.stop_price < entry.price_floor && entry.price_floor <= entry.price_ceiling;
    const referenceValid = price != null && reference && reference.status === 'available' &&
      (!reference.symbol || reference.symbol === row.symbol) && reference.date === day &&
      (!reference.as_of || reference.as_of === day) && (!reference.basis_date || reference.basis_date === day) &&
      reference.reference_price_date === day && reference.reference_price_source === 'signal_day_close' &&
      reference.reference_price === price && atCutoff(reference.available_at || row.available_at) &&
      Array.isArray(reference.reason_codes) && !reference.reason_codes.length &&
      positive(reference.stop_price) && reference.stop_price < price;
    const risk = entryValid ? entry : referenceValid ? reference : null;
    const invalidation = risk && positive(risk.invalidation_price) && risk.stop_price <= risk.invalidation_price &&
      risk.invalidation_price < price ? risk.invalidation_price : null;
    const target = methodBuy && entryValid && positive(entry.target_price) &&
      entry.target_price > Math.max(price, entry.price_ceiling) &&
      typeof entry.target_price_basis === 'string' && entry.target_price_basis.trim() ? entry.target_price : null;
    let entryState, explanation;
    if (price == null) {
      entryState = '截止日收盘未核验';
      explanation = '当前收盘价与保存结果的日期或来源不一致，价位计划未展示。';
    } else if (!methodBuy) {
      entryState = view.valid === true && view.status === 'complete'
        ? (view.direction === 'SELL' ? '当前为卖方方向' : '当前为观察方向') : '买方条件未通过';
      explanation = '所选方法没有合格买方条件。';
    } else if (entryValid) {
      entryState = `${number(entry.price_floor)}–${number(entry.price_ceiling)} 元`;
      explanation = `原计划 ${day} 收盘后形成，仅供下一交易日条件核验。`;
    } else if (entry) {
      entryState = '保存计划未通过核验';
      explanation = '原计划的股票、日期、参考价、可知时间或风险边界不完整，未展示买入区间。';
    } else {
      entryState = '未保存买入条件';
      explanation = '此结果只保存方法方向，未保存条件买入计划。';
    }
    const stop = risk ? `${number(risk.stop_price)} 元` : '未保存有效风险边界';
    const boundary = invalidation == null ? '未保存失效依据' : `${number(invalidation)} 元`;
    return `<div class="market-opportunity-prices"><h3>价位计划</h3>
      <dl><div><dt>截止日收盘参考</dt><dd>${price == null ? '尚未核验' : `${number(price)} 元`}</dd></div>
      <div><dt>条件买入区间</dt><dd>${entryState}</dd></div>
      <div><dt>失效参考价</dt><dd>${boundary}</dd></div>
      <div><dt>风险止损参考</dt><dd>${stop}</dd></div>
      <div><dt>目标价</dt><dd>${target == null ? '此方法未定义目标价' : `${number(target)} 元`}</dd></div></dl>
      <p class="market-opportunity-plan-state">${explanation}</p>
      ${referenceValid && !entryValid ? '<p class="market-opportunity-muted">风险参考带不作为条件买入区间。</p>' : ''}
      </div>`;
  }

  function detailsHtml(row, index) {
    const reference = predictionReference(row);
    const evaluation = row.method_evaluation;
    const evidence = evaluation && globalThis.KHQuantMethods?.evaluationHtml
      ? globalThis.KHQuantMethods.evaluationHtml(evaluation, false, row.method_view || null)
      : '<p class="market-opportunity-muted">此行未保存可展示的方法评估明细。</p>';
    return `<section id="market-opportunity-detail-${index}" class="market-opportunity-details" aria-label="${esc(row.name || row.symbol)}的预测与依据">
      <div class="market-opportunity-detail-grid">
        <section class="market-opportunity-prediction" aria-label="原预测图">
          <div class="market-opportunity-section-heading"><h3>价格走势与原预测</h3><span>${esc(cutoff(row, state.options.result) || '截止日未记录')}</span></div>
          ${reference && row.method_view?.forecast_valid !== true ? '<p class="market-opportunity-muted">预测质量未通过可用性核验；曲线仅展示原保存结果。</p>' : ''}
          ${reference ? `<details class="market-opportunity-display-settings"><summary>显示范围</summary><div class="market-opportunity-history-controls">
            <label>历史显示范围<select id="market-history-window"><option value="60">最近60根</option><option value="120">最近120根</option><option value="all">全部已存历史</option></select></label>
            <label>显示起点<input id="market-history-start" type="date"></label>
            <button id="market-history-reset" type="button">恢复默认范围</button>
          </div>
          <p id="market-prediction-history-state" class="market-opportunity-muted" role="status">正在读取已保存的预测。</p></details>
          <div class="market-opportunity-chart-legend"><span>真实行情（日K）</span><span>预测中位（虚线）</span><span>样本 P10–P90 · 非买入区间</span></div>
          <div class="market-opportunity-chart-host"><div id="market-prediction-chart"></div><div id="market-forecast-boundary" class="market-opportunity-forecast-boundary" hidden><span></span></div></div>
          <p id="market-chart-readout" class="market-opportunity-readout" aria-live="polite"></p>`
        : '<div class="market-opportunity-prediction-empty"><strong>没有已保存的预测引用</strong><p>可查看下方方法依据；展开此行不会运行预测。</p></div>'}
        </section>
        <section class="market-opportunity-plan" aria-label="价位计划与验证状态">
          ${pricePlanHtml(row, state.options.result)}
          ${methodSummaryHtml(row)}
        </section>
      </div>
      <details class="market-opportunity-full-evidence"><summary>完整方法与来源依据</summary><div class="market-opportunity-evidence-grid"><section class="market-opportunity-method"><h3>方法依据</h3>${evidence}${reasonsHtml(row)}<p class="market-opportunity-muted">未记录多方法独立同向票数，也未判断首次买入触发。跨日稳定未记录，历史有效性待回测验证。</p></section><section class="market-opportunity-source"><h3>来源 B/S</h3><p class="market-opportunity-muted">结构与量价来源独立保留，不等于所选方法方向。</p>${sourceHtml(row)}</section></div></details>
      <div class="market-opportunity-actions"><button type="button" data-market-action="analyze" data-market-index="${index}"${typeof state.options.onAnalyze !== 'function' ? ' disabled' : ''}>查看结构</button><button type="button" data-market-action="backtest" data-market-index="${index}"${typeof state.options.onBacktest !== 'function' ? ' disabled' : ''}>带入回测</button><span class="market-opportunity-muted">带入参数后再执行，不自动启动计算。</span></div>
    </section>`;
  }

  function rowHtml(row, index) {
    const price = referencePrice(row, state.options.result);
    const expanded = state.expanded === row.symbol;
    const view = row.method_view;
    const method = view?.label || globalThis.KHQuantMethods?.label(row.analysis_method || state.options.result?.analysis_method) || '方法未记录';
    const category = view?.valid === true && view.status === 'complete' ? ({BUY: 'buy', SELL: 'sell', WAIT: 'wait'})[view.direction] || 'unavailable' : 'unavailable';
    return `<article class="market-opportunity-row" data-market-symbol="${esc(row.symbol)}" data-expanded="${expanded}">
      <div class="market-opportunity-summary">
        <div class="market-opportunity-stock"><button type="button" class="market-opportunity-favorite" data-market-action="favorite" data-market-index="${index}" aria-pressed="false" aria-label="关注 ${esc(row.symbol)}">${icon('star')}</button><div><strong>${esc(row.name || row.symbol || '股票未记录')}</strong><span>${esc(row.symbol)}</span></div></div>
        <div class="market-opportunity-direction"><span class="market-opportunity-label">所选方法</span><strong class="market-opportunity-direction-value" data-direction="${category}">${direction(row)}</strong><span class="market-opportunity-method-name">${esc(method)}</span></div>
        <div class="market-opportunity-checks"><div><span class="market-opportunity-label">跨日稳定</span><strong>未记录</strong></div><div><span class="market-opportunity-label">历史检验</span><strong>待回测验证</strong></div></div>
        <div class="market-opportunity-price"><span class="market-opportunity-label">截止日收盘参考</span><strong>${price == null ? '—' : `¥${number(price)}`}</strong><span>${esc(cutoff(row, state.options.result) || '日期未记录')}${price == null ? ' · 尚未核验' : ''}</span></div>
        <button class="market-opportunity-toggle" type="button" data-market-action="toggle" data-market-index="${index}" aria-expanded="${expanded}" aria-controls="market-opportunity-detail-${index}">${expanded ? '收起依据' : '展开依据'}${icon(expanded?'chevron-up':'chevron-down')}</button>
      </div>
      ${expanded ? detailsHtml(row, index) : `<section id="market-opportunity-detail-${index}" hidden></section>`}
    </article>`;
  }

  function visible() {
    const host = $('scan-opportunities');
    return Boolean(host && !host.hidden && !host.closest('[hidden]'));
  }

  function destroyChart() {
    ++state.generation;
    state.chart?.destroy();
    state.chart = null;
  }

  function updateFavorites() {
    const favorites = state.options.favorites || new Set();
    $('scan-opportunities')?.querySelectorAll('[data-market-action="favorite"]').forEach(button => {
      const row = state.rows[Number(button.dataset.marketIndex)];
      if (!row) return;
      const selected = favorites.has(row.symbol);
      button.innerHTML = icon(selected ? 'star-fill' : 'star');
      button.setAttribute('aria-pressed', String(selected));
      button.setAttribute('aria-label', `${selected ? '取消关注' : '关注'} ${row.name || row.symbol}`);
      button.disabled = !state.options.personalReady || typeof state.options.onFavorite !== 'function';
    });
  }

  function draw() {
    destroyChart();
    const host = $('scan-opportunities');
    if (!host) return;
    host.classList.add('market-opportunity-list');
    host.innerHTML = state.rows.length
      ? state.rows.map(rowHtml).join('') + '<p class="market-opportunity-feedback" role="status" aria-live="polite"></p>'
      : `<div class="market-opportunity-empty"><strong>${state.options.result ? '没有符合条件的股票' : '尚无保存结果'}</strong><p>${state.options.result ? '可调整方法方向或搜索条件，查看其他已保存记录。' : '读取或生成方法筛选结果后，在这里查看个股依据与原预测。'}</p></div>`;
    updateFavorites();
    const row = state.rows.find(item => item.symbol === state.expanded);
    const reference = row && predictionReference(row);
    if (!reference) return;
    if (!globalThis.KHQuantPredictionChart?.create || typeof state.options.request !== 'function') {
      $('market-prediction-history-state').textContent = '原预测图暂不可用；没有启动计算。';
      return;
    }
    const generation = state.generation;
    state.chart = globalThis.KHQuantPredictionChart.create({prefix: 'market', isVisible: visible});
    const request = state.options.request;
    Promise.resolve(state.chart.render({symbol: row.symbol, as_of: row.method_view.as_of,
      result_ref: {cache_key: reference}, forecast: []}, (path, options) => request('/api/light' + path, options)))
      .then(() => {
        if (generation !== state.generation) return;
        const message = $('market-prediction-history-state')?.textContent || '';
        // Verification failures stay visible even while range controls are closed.
        if (message.startsWith('历史图表读取失败') && $('market-chart-readout'))
          $('market-chart-readout').textContent = message;
        else if (message.startsWith('历史输入无法核验') && $('market-chart-readout'))
          $('market-chart-readout').textContent = '历史行情未核验；原预测保留，详见显示范围。';
      })
      .catch(error => {
        if (generation === state.generation && $('market-prediction-history-state')) {
          $('market-prediction-history-state').textContent = `原预测图读取失败：${error.message || error}。没有自动重算。`;
          if ($('market-chart-readout')) $('market-chart-readout').textContent = $('market-prediction-history-state').textContent;
        }
      });
  }

  function bind(host) {
    if (bound.has(host)) return;
    bound.add(host);
    host.addEventListener('click', async event => {
      const button = event.target.closest('[data-market-action]');
      if (!button || !host.contains(button) || button.disabled) return;
      const row = state.rows[Number(button.dataset.marketIndex)];
      if (!row) return;
      const action = button.dataset.marketAction;
      if (action === 'toggle') {
        state.expanded = state.expanded === row.symbol ? null : row.symbol;
        draw();
        host.querySelector(`[data-market-action="toggle"][data-market-index="${Number(button.dataset.marketIndex)}"]`)?.focus();
        return;
      }
      const generation = state.generation;
      try {
        const callback = ({favorite: state.options.onFavorite, analyze: state.options.onAnalyze, backtest: state.options.onBacktest})[action];
        if (typeof callback === 'function') await callback(action === 'favorite' ? row.symbol : row);
      } catch (error) {
        if (generation === state.generation) {
          const feedback = host.querySelector('.market-opportunity-feedback');
          if (feedback) feedback.textContent = `操作未完成：${error.message || error}`;
        }
      }
    });
  }

  function render(options = {}) {
    const host = $('scan-opportunities');
    if (!host) { clear(); return; }
    bind(host);
    const rows = list(options.rows).filter(row => row && typeof row === 'object');
    const result = options.result;
    const scope = JSON.stringify([result?.run_id, result?.as_of || result?.data_end,
      result?.analysis_method, result?.scan_view?.page, result?.scan_view?.offset,
      rows.map(row => row.symbol)]);
    if (scope !== state.scope) {
      state.expanded = rows.find(row => row.method_view?.forecast_valid === true && predictionReference(row))?.symbol ?? null;
    } else if (!rows.some(row => row.symbol === state.expanded)) state.expanded = null;
    const content = JSON.stringify([scope, rows, typeof options.onAnalyze === 'function',
      typeof options.onBacktest === 'function', typeof options.request === 'function']);
    const changed = content !== state.content;
    state.rows = rows;
    state.options = options;
    state.scope = scope;
    state.content = content;
    if (changed) draw();
    else { updateFavorites(); state.chart?.onRoute(); }
  }

  function clear() {
    destroyChart();
    state.rows = [];
    state.options = {};
    state.scope = state.content = state.expanded = null;
    if ($('scan-opportunities')) $('scan-opportunities').innerHTML = '';
  }

  globalThis.KHQuantMarketList = {render, clear, onRoute() { state.chart?.onRoute(); }};
})();
