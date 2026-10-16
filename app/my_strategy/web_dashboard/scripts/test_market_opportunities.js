/* Run with: node --test my_strategy/web_dashboard/scripts/test_market_opportunities.js */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const scripts = path.join(__dirname, '..', 'static', 'js');
const date = '2026-10-15';
const refA = 'a'.repeat(64), refB = 'b'.repeat(64);
const tick = () => new Promise(resolve => setImmediate(resolve));

function harness() {
  const elements = new Map(), calls = [], instances = [];
  let controls = [], dynamicIds = [];
  class Element {
    constructor(id) {
      this.id = id; this.listeners = {}; this.attributes = {}; this.dataset = {};
      this.hidden = false; this.disabled = false; this.value = ''; this.textContent = '';
      this.style = {}; this.classList = {add() {}}; this._html = '';
    }
    set innerHTML(value) {
      this._html = value;
      if (this.id !== 'scan-opportunities') return;
      dynamicIds.forEach(id => elements.delete(id)); dynamicIds = []; controls = [];
      for (const match of value.matchAll(/\bid="([^"]+)"/g)) {
        const element = new Element(match[1]);
        elements.set(match[1], element); dynamicIds.push(match[1]);
      }
      for (const match of value.matchAll(/<button\b([^>]*)>([\s\S]*?)<\/button>/g)) {
        const element = new Element('button'); element.host = this;
        for (const attribute of match[1].matchAll(/([\w-]+)="([^"]*)"/g)) {
          element.attributes[attribute[1]] = attribute[2];
          if (attribute[1].startsWith('data-')) element.dataset[attribute[1].slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())] = attribute[2];
        }
        element.disabled = /\sdisabled(?:\s|$)/.test(match[1]); element.textContent = match[2];
        controls.push(element);
      }
      const feedback = new Element('feedback'); elements.set('feedback', feedback);
    }
    get innerHTML() { return this._html; }
    addEventListener(type, callback) { this.listeners[type] = callback; }
    setAttribute(name, value) { this.attributes[name] = value; }
    contains(element) { return element.host === this && controls.includes(element); }
    closest(selector) { return selector === '[data-market-action]' && this.dataset.marketAction ? this : null; }
    querySelectorAll(selector) { return selector === '[data-market-action="favorite"]' ? controls.filter(item => item.dataset.marketAction === 'favorite') : []; }
    querySelector(selector) {
      if (selector === '.market-opportunity-feedback') return elements.get('feedback');
      const action = selector.match(/data-market-action="([^"]+)"/)?.[1];
      const index = selector.match(/data-market-index="([^"]+)"/)?.[1];
      return controls.find(item => item.dataset.marketAction === action && item.dataset.marketIndex === index) || null;
    }
    focus() { this.focused = true; }
  }
  const host = new Element('scan-opportunities'); elements.set(host.id, host);
  const context = vm.createContext({console, AbortController, Set, WeakSet,
    document: {getElementById: id => elements.get(id) || null, documentElement: new Element('html')},
    KHQuantMethods: {label: () => '结构与量价 + Kronos', evaluationHtml: () => '<p>已保存的方法依据</p>'}});
  vm.runInContext(fs.readFileSync(path.join(scripts, 'prediction_chart.js'), 'utf8'), context);
  const create = context.KHQuantPredictionChart.create;
  context.KHQuantPredictionChart.create = options => {
    const instance = create(options), record = {options, destroyed: 0};
    const destroy = instance.destroy;
    instance.destroy = () => { record.destroyed++; destroy(); };
    instances.push(record); return instance;
  };
  vm.runInContext(fs.readFileSync(path.join(scripts, 'market_opportunities.js'), 'utf8'), context);
  const request = (url, options = {}) => new Promise((resolve, reject) => calls.push({url, options, resolve, reject}));
  const click = (action, index) => {
    const target = controls.find(item => item.dataset.marketAction === action && item.dataset.marketIndex === String(index));
    assert(target, `Missing ${action} control ${index}`);
    return host.listeners.click({target});
  };
  return {ui: context.KHQuantMarketList, host, calls, instances, request, click,
    element: id => elements.get(id), buttons: () => controls};
}

function row(symbol = '000001.SZ', predictionRef = refA, overrides = {}) {
  return {symbol, name: '合成测试股票', data_end: date, reference_price_date: date,
    reference_price_source: 'verified_local_close', close: 23.45, reference_price: 23.45,
    method_view: {as_of: date, valid: true, status: 'complete', direction: 'BUY',
      label: '结构与量价 + Kronos', prediction_ref: predictionRef, forecast_valid: true},
    method_evaluation: {status: 'complete'}, ...overrides};
}

function options(h, rows, extra = {}) {
  return {rows, result: {run_id: 'saved-test-run', as_of: date}, request: h.request,
    favorites: new Set(), personalReady: true, onFavorite() {}, onAnalyze() {}, onBacktest() {}, ...extra};
}

function chartResponse(symbol, reference) {
  return {symbol, as_of: date, prediction_ref: reference, forecast: [], history: [],
    history_verification: {status: 'unavailable', reason: 'fixture without history'}};
}

test('uses only verified cutoff close; missing, stale, false and forecast prices are not prices', () => {
  for (const overrides of [
    {reference_price: null}, {reference_price: '23.45'}, {reference_price: false},
    {reference_price: NaN}, {reference_price: Infinity}, {reference_price: -1},
    {reference_price_date: '2026-10-14'}, {data_end: '2026-10-14'},
    {reference_price_source: 'forecast'}, {close: 21},
  ]) {
    const h = harness();
    h.ui.render(options(h, [row('000001.SZ', null, overrides)]));
    assert(h.host.innerHTML.includes('尚未核验'));
    assert(!h.host.innerHTML.includes('¥23.45'));
    assert(!h.host.innerHTML.includes('¥0'));
    assert.equal(h.calls.length, 0);
  }
  const h = harness(); h.ui.render(options(h, [row()]));
  assert(h.host.innerHTML.includes('¥23.45'));
  assert(h.host.innerHTML.includes('待回测验证'));
  assert(h.host.innerHTML.includes('跨日稳定'));
  assert(h.host.innerHTML.includes('未保存买入条件'));
  assert(!h.host.innerHTML.includes('尚未生成'));
  assert(h.host.innerHTML.includes('未记录多方法独立同向票数'));
  assert(!h.host.innerHTML.includes('历史检验通过'));
  assert(!h.host.innerHTML.includes('条件买入区间</dt><dd>23.45'));
  h.ui.render(options(h, [row()], {result: {run_id: 'saved-test-run', as_of: '2026-10-14'}}));
  assert(!h.host.innerHTML.includes('¥23.45'));
});

test('escapes saved stock, method, date and source evidence content', async () => {
  const h = harness(), unsafe = '<img src=x onerror="alert(1)">';
  const item = row('000001.SZ', null, {name: unsafe,
    method_view: {label: unsafe, status: 'unavailable', direction: 'BUY'},
    source_reference_points: [{reference_layer: 'source', source_label: unsafe, available_at: unsafe, action: 'BUY'}]});
  h.ui.render(options(h, [item])); await h.click('toggle', 0);
  assert(!h.host.innerHTML.includes(unsafe));
  assert(h.host.innerHTML.includes('&lt;img'));
  assert(h.host.innerHTML.includes('&quot;'));
  assert(h.host.innerHTML.includes('方法不可用'));
  assert(!h.host.innerHTML.includes('B · 方法买方参考'));
  assert(h.host.innerHTML.includes('B · 买方参考'));
  assert(h.host.innerHTML.includes('不等于所选方法方向'));
  assert.equal(h.calls.length, 0);
});

test('default expands one valid saved forecast; same page rerenders preserve expansion and avoid rereads', async () => {
  const h = harness(), rows = [row('000001.SZ', null), row('000002.SZ', refB)];
  const args = options(h, rows); h.ui.render(args);
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].url, `/api/light/predictions/${refB}/chart`);
  for (const id of ['market-history-window','market-history-start','market-history-reset',
    'market-prediction-chart','market-prediction-history-state','market-forecast-boundary','market-chart-readout']) {
    assert.equal((h.host.innerHTML.match(new RegExp(`id="${id}"`, 'g')) || []).length, 1, id);
  }
  h.ui.render({...args, favorites: new Set(['000002.SZ'])});
  assert.equal(h.calls.length, 1); assert.equal(h.instances.length, 1);
  assert.equal(h.buttons().find(button => button.dataset.marketAction === 'favorite' && button.dataset.marketIndex === '1').attributes['aria-pressed'], 'true');
  await h.click('toggle', 1);
  assert.equal(h.instances[0].destroyed, 1); assert(h.calls[0].options.signal.aborted);
  h.ui.render(args);
  assert.equal(h.calls.length, 1, 'same page must not reopen a manually collapsed row');
});

test('changing rows or run cancels old prediction and rejects late response writes', async () => {
  const h = harness(); h.ui.render(options(h, [row()]));
  const first = h.calls[0];
  h.ui.render(options(h, [row('000002.SZ', refB)]));
  assert(first.options.signal.aborted); assert.equal(h.instances[0].destroyed, 1);
  const before = h.element('market-prediction-history-state').textContent;
  first.resolve(chartResponse('000001.SZ', refA)); await tick();
  assert.equal(h.element('market-prediction-history-state').textContent, before);
  h.calls[1].resolve(chartResponse('000002.SZ', refB)); await tick();
  assert(h.element('market-prediction-history-state').textContent.includes('fixture without history'));
  h.ui.render(options(h, [row('000002.SZ', refB)], {result: {run_id: 'another-run', as_of: date}}));
  assert.equal(h.instances[1].destroyed, 1); assert.equal(h.calls.length, 3);
  h.ui.clear(); assert(h.calls[2].options.signal.aborted);
  assert.equal(h.host.innerHTML, '');
  h.calls[2].resolve(chartResponse('000002.SZ', refB)); await tick();
  assert.equal(h.host.innerHTML, '');
});

test('switching expanded rows keeps one chart and preserves chosen row when favorite state changes', async () => {
  const h = harness(), rows = [row(), row('000002.SZ', refB)], args = options(h, rows);
  h.ui.render(args); await h.click('toggle', 1);
  assert.equal(h.instances[0].destroyed, 1); assert(h.calls[0].options.signal.aborted);
  assert.equal(h.calls.length, 2);
  assert.equal((h.host.innerHTML.match(/id="market-prediction-chart"/g) || []).length, 1);
  h.ui.render({...args, favorites: new Set(['000001.SZ'])});
  assert.equal(h.calls.length, 2);
  assert(h.host.innerHTML.includes('data-market-symbol="000002.SZ" data-expanded="true"'));
  h.ui.onRoute(); assert.equal(h.calls.length, 2);
});

test('callbacks get original saved row and favorite symbol; absent permissions disable favorites', async () => {
  const h = harness(), item = row('000001.SZ', null), actions = [];
  const args = options(h, [item], {onAnalyze: value => actions.push(['analyze', value]),
    onBacktest: value => actions.push(['backtest', value]), onFavorite: value => actions.push(['favorite', value])});
  h.ui.render(args); await h.click('toggle', 0);
  await h.click('analyze', 0); await h.click('backtest', 0); await h.click('favorite', 0);
  assert.strictEqual(actions[0][1], item); assert.strictEqual(actions[1][1], item);
  assert.equal(actions[2][1], item.symbol); assert.equal(h.calls.length, 0);
  h.ui.render({...args, personalReady: false}); await h.click('favorite', 0);
  assert.equal(actions.length, 3);
  h.ui.render({...args, onBacktest: async () => { throw Error('<unsafe failure>'); }});
  await h.click('backtest', 0);
  assert.equal(h.element('feedback').textContent, '操作未完成：<unsafe failure>');
});

test('empty results and unavailable prediction infrastructure stay readable without work submission', () => {
  const h = harness(); h.ui.render({rows: []});
  assert(h.host.innerHTML.includes('尚无保存结果'));
  h.ui.render({rows: [], result: {run_id: 'saved'}});
  assert(h.host.innerHTML.includes('没有符合条件的股票'));
  h.ui.render(options(h, [row()], {request: undefined}));
  assert(h.element('market-prediction-history-state').textContent.includes('没有启动计算'));
  assert.equal(h.calls.length, 0);
});

test('unusable forecast is not opened automatically and remains labeled when explicitly expanded', async () => {
  const h = harness(), item = row(); item.method_view.forecast_valid = false;
  item.method_view.valid = false; item.method_view.status = 'unavailable';
  h.ui.render(options(h, [item])); assert.equal(h.calls.length, 0);
  await h.click('toggle', 0);
  assert.equal(h.calls.length, 1);
  assert(h.host.innerHTML.includes('预测质量未通过可用性核验'));
  assert(h.host.innerHTML.includes('方法不可用'));
});

test('compact layout keeps full evidence collapsed and shows only verified saved summary statistics', async () => {
  const h = harness(), item = row();
  item.method_view.forecast_summary = {terminal_return_median: .035, up_path_fraction: .625};
  item.method_view.score_valid = true; item.method_view.score_value = 68;
  h.ui.render(options(h, [item]));
  const html = h.host.innerHTML;
  assert(html.includes('<details class="market-opportunity-display-settings"><summary>显示范围</summary>'));
  assert(html.includes('<details class="market-opportunity-full-evidence"><summary>完整方法与来源依据</summary>'));
  const summary = html.split('class="market-opportunity-method-brief"')[1].split('<details class="market-opportunity-full-evidence"')[0];
  assert(summary.includes('3.5%')); assert(summary.includes('62.5%')); assert(summary.includes('68 / 100'));
  assert(summary.includes('不是历史胜率'));
  assert(html.indexOf('class="market-opportunity-actions"') > html.indexOf('完整方法与来源依据'));
  item.method_view.forecast_valid = false;
  h.ui.render(options(h, [item]));
  const invalidSummary = h.host.innerHTML.split('class="market-opportunity-method-brief"')[1].split('<details class="market-opportunity-full-evidence"')[0];
  assert(invalidSummary.includes('预测摘要未通过核验'));
  assert(!invalidSummary.includes('62.5%')); assert(!invalidSummary.includes('3.5%'));
});

test('chart failures stay visible outside collapsed range settings', async () => {
  const h = harness(); h.ui.render(options(h, [row()]));
  h.calls[0].reject(Error('保存引用不可读')); await tick();
  assert(h.element('market-chart-readout').textContent.includes('保存引用不可读'));
  assert(h.element('market-chart-readout').textContent.includes('不自动重算'));
});

function entryPlan(overrides = {}) {
  return {plan_id: 'frozen-plan', symbol: '000001.SZ', signal_date: date,
    available_at: date + 'T15:00:00+08:00', reference_price: 23.45,
    policy: 'risk', status: 'active', entry_allowed: true, valid_for_sessions: 1,
    source_event_ids: ['original-buy-source'], reason_codes: [],
    stop_price: 22.5, invalidation_price: 22.6, price_floor: 22.8, price_ceiling: 23.6,
    ...overrides};
}

test('displays genuine saved entry, separate invalidation and stop, with no invented target', () => {
  const h = harness(), plan = entryPlan(), item = row('000001.SZ', null, {entry_plan: plan});
  const before = JSON.stringify(item);
  h.ui.render(options(h, [item])); h.click('toggle', 0);
  const html = h.host.innerHTML;
  assert(html.includes('条件买入区间</dt><dd>22.8–23.6 元'));
  assert(html.includes('失效参考价</dt><dd>22.6 元'));
  assert(html.includes('风险止损参考</dt><dd>22.5 元'));
  assert(html.includes('目标价</dt><dd>此方法未定义目标价'));
  assert(html.includes('仅供下一交易日条件核验'));
  assert.equal(JSON.stringify(item), before, 'rendering must not mutate saved row or plan');
  assert.equal(h.calls.length, 0, 'displaying price plans must not submit work');
});

test('entry identity, as-of, source, future time, numeric geometry and rejected plans fail closed', () => {
  for (const invalid of [
    {symbol: '000002.SZ'}, {signal_date: '2026-10-14'}, {as_of: '2026-10-16'},
    {date: '2026-10-14'}, {reference_price_date: '2026-10-14'},
    {available_at: date + 'T15:00:01+08:00'}, {available_at: '2026-10-16T15:00:00+08:00'},
    {available_at: date}, {available_at: 'invalid'}, {reference_price: 20},
    {reference_price_source: 'forecast'}, {reference_price_source: 'next_open'},
    {status: 'rejected'}, {entry_allowed: false}, {valid_for_sessions: 5},
    {reason_codes: ['risk_stop_distance_outside_contract']}, {source_event_ids: []},
    {plan_id: ''}, {policy: 'unknown'}, {stop_price: 25}, {stop_price: -1},
    {price_floor: 21}, {price_floor: '22.8'}, {price_ceiling: NaN}, {price_ceiling: 22.7},
  ]) {
    const h = harness(); h.ui.render(options(h, [row('000001.SZ', null, {entry_plan: entryPlan(invalid)})]));
    h.click('toggle', 0);
    assert(h.host.innerHTML.includes('保存计划未通过核验'), JSON.stringify(invalid));
    assert(!h.host.innerHTML.includes('22.8–23.6 元'), JSON.stringify(invalid));
    assert(!h.host.innerHTML.includes('风险止损参考</dt><dd>22.5 元'), JSON.stringify(invalid));
    assert.equal(h.calls.length, 0);
  }
});

test('a reference-only risk band remains risk evidence and is never an entry range', () => {
  const h = harness(), item = row('000001.SZ', null, {available_at: date + 'T15:00:00+08:00',
    reference_risk_plan: {status: 'available', symbol: '000001.SZ', date,
      reference_price: 23.45, reference_price_date: date, reference_price_source: 'signal_day_close',
      stop_price: 22.5, invalidation_price: 22.6, reason_codes: [],
      reference_price_band: {low: 22.8, high: 23.6, not_fill_or_order_range: true, reference_only: true}}});
  h.ui.render(options(h, [item])); h.click('toggle', 0);
  assert(h.host.innerHTML.includes('未保存买入条件'));
  assert(h.host.innerHTML.includes('风险止损参考</dt><dd>22.5 元'));
  assert(h.host.innerHTML.includes('失效参考价</dt><dd>22.6 元'));
  assert(h.host.innerHTML.includes('风险参考带不作为条件买入区间'));
  assert(!h.host.innerHTML.includes('22.8–23.6 元'));
  item.reference_risk_plan.available_at = '2026-10-16T15:00:00+08:00';
  h.ui.render(options(h, [item]));
  assert(!h.host.innerHTML.includes('风险止损参考</dt><dd>22.5 元'));
});

test('a target requires an explicit saved target and its basis, never a forecast or limit price', () => {
  for (const invalid of [{}, {target_price: 30}, {planned_target_price: 30},
    {target_price: '30', target_price_basis: 'frozen-resistance'},
    {target_price: 22, target_price_basis: 'frozen-resistance'}]) {
    const h = harness(); h.ui.render(options(h, [row('000001.SZ', null, {entry_plan: entryPlan(invalid)})]));
    h.click('toggle', 0);
    assert(h.host.innerHTML.includes('目标价</dt><dd>此方法未定义目标价'));
  }
  const h = harness(); h.ui.render(options(h, [row('000001.SZ', null,
    {entry_plan: entryPlan({target_price: 30, target_price_basis: 'frozen-resistance'})})]));
  h.click('toggle', 0);
  assert(h.host.innerHTML.includes('目标价</dt><dd>30 元'));
});

test('an otherwise valid entry plan does not create a method BUY for WAIT, SELL or unavailable methods', () => {
  for (const method of [{valid: true, status: 'complete', direction: 'SELL'},
    {valid: true, status: 'complete', direction: 'WAIT'},
    {valid: false, status: 'unavailable', direction: 'BUY'}]) {
    const h = harness(); h.ui.render(options(h, [row('000001.SZ', null,
      {entry_plan: entryPlan(), method_view: {...method, as_of: date}})]));
    h.click('toggle', 0);
    assert(!h.host.innerHTML.includes('22.8–23.6 元'));
    assert(!h.host.innerHTML.includes('目标价</dt><dd>30 元'));
    assert.equal(h.calls.length, 0);
  }
});
