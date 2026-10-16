'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {evaluationHtml} = require('../web_dashboard/static/js/analysis_method_view.js');

function evidence(overrides = {}) {
  return {method: 'structure_kronos', as_of: '2026-10-16', status: 'partial', direction: 'WAIT',
    missing_reasons: ['kronos_prediction_unusable'], source_identity: {schema: 'saved_source'},
    kronos: {status: 'completed', quality: {prediction_usable: false, source_verified: true,
      calendar_verified: true, input_contract_verified: true, valid_samples: 2, requested_samples: 8,
      valid_fraction: .25, rejected_by_rule: {nonfinite: 0, price: 0, negative_activity: 0, ohlc_geometry: 6}},
    summary: {terminal_return_p10: .052093623427608, terminal_return_median: .08241729234703665,
      terminal_return_p90: .11274096126646531, up_path_fraction: 1}}, ...overrides};
}
function view(overrides = {}) {
  return {status: 'unavailable', valid: false, forecast_valid: false, score_valid: false,
    reason_codes: ['forecast_quality_unavailable', 'forecast_sample_quality_invalid'], ...overrides};
}

test('真实 2/8 质量失败区分原采样、来源与组合判断，并显示实际剔除原因', () => {
  const input = evidence(), before = JSON.stringify(input);
  const html = evaluationHtml(input, false, view());
  for (const text of ['部分证据可用', '组合判断未通过', '2 / 8', '25%', '至少 8 条且 80%',
    'OHLC 高低价关系不成立', '6 条', '以下为原始有效采样的摘要', '中位涨跌', '8.24%', '上涨采样比例', '100%']) assert(html.includes(text), text);
  assert(!html.includes('方法买方参考'));
  assert(!html.includes('组合判断已通过'));
  assert(!/<pre\b|JSON|"prediction_usable"|"rejected_by_rule"/.test(html));
  assert.equal(JSON.stringify(input), before, '渲染不能改写冻结对象');
  assert(html.includes('role="img" aria-label="Kronos 采样涨跌分位数'));
});

test('组合方向只接受服务端完整有效视图，原 direction 不能提升为通过', () => {
  const raw = evidence({status: 'complete', direction: 'BUY', missing_reasons: []});
  const unknown = evaluationHtml(raw);
  assert(unknown.includes('组合判断待核验'));
  assert(!unknown.includes('B · 方法买方参考'));
  for (const [status, valid] of [['unavailable', true], ['complete', false], ['failed', false], ['deferred', false]]) {
    const html = evaluationHtml(raw, false, view({status, valid, direction: 'BUY'}));
    assert(!html.includes('B · 方法买方参考'), `${status}/${valid}`);
  }
  const confirmed = evaluationHtml(raw, false, view({status: 'complete', valid: true, forecast_valid: true,
    direction: 'WAIT', reason_codes: []}));
  assert(confirmed.includes('WAIT · 有效观察'));
});

test('0 分和 0% 是实际值，null、布尔值和数字字符串不能冒充零', () => {
  const input = evidence({method: 'structure_kronos_ma_mlp', score: {status: 'scored', score: 0,
    probabilities: {down: 1, flat: 0, up: 0}, qualification: {mode: 'historical_candidate',
    paper_allocation_eligible: false, independent_forward_status: 'pending'}}});
  const zero = evaluationHtml(input, false, view({score_valid: true}));
  assert(zero.includes('MLP 已评分 0 / 100'));
  assert(zero.includes('上涨 0%'));
  assert(zero.includes('尚未取得自动模拟配置资格'));
  assert(zero.includes('独立前瞻资格待验证'));
  for (const score of [null, undefined, false, '0', -1, 101, Infinity, NaN]) {
    const html = evaluationHtml({...input, score: {...input.score, score}}, false, view());
    assert(html.includes('评分不可用'), String(score));
    assert(!html.includes('已评分 0 / 100'));
  }
});

test('未选择评分方法时，空score或误带评分对象均不新增MLP失败卡', () => {
  for (const score of [undefined, null, {}, {status: 'unavailable'}, {status: 'scored', score: 60}]) {
    const html = evaluationHtml(evidence({method: 'structure_kronos', score}), false, view());
    assert(!html.includes('<h4>均线MLP评分</h4>'));
    assert(!html.includes('评分不可用'));
    assert(html.includes('当前方法不需要评分模型'));
  }
  for (const method of ['structure_kronos_ma_mlp', 'structure_kronos_ma_l2']) {
    const html = evaluationHtml(evidence({method, score: {}}), false, view());
    assert(html.includes('评分不可用'), method);
  }
});

test('资格矩阵默认折叠，核心解释、预测区间、有效路径和评分卡保留在展开区之前', () => {
  const input = evidence({method:'structure_kronos_ma_mlp', model_id:'c'.repeat(64),
    score:{status:'scored',score:60,qualification:{mode:'historical_candidate',independent_forward_status:'pending'}}});
  const html = evaluationHtml(input, false, view());
  const start = html.indexOf('<details class="amv-card amv-qualification">');
  assert(start > 0);
  assert(!/<details\b[^>]*amv-qualification[^>]*\bopen\b/.test(html));
  const visible = html.slice(0,start), folded = html.slice(start);
  for(const text of ['暂不生成组合方向','<svg','2 / 8','MLP 已评分 60 / 100']) assert(visible.includes(text),text);
  for(const text of ['<summary>可用性与资格核验</summary>','amv-checks','模型身份','c'.repeat(64),'独立前瞻资格待验证']) assert(folded.includes(text),text);
});

test('0 条有效路径准确显示，不以100%上涨采样掩盖0/8质量失败', () => {
  const input = evidence();
  input.kronos.quality.valid_samples = 0;
  input.kronos.summary = null;
  const html = evaluationHtml(input, false, view());
  assert(html.includes('0 / 8'));
  assert(html.includes('0%'));
  assert(html.includes('分位数区间未完整记录，无法绘制'));
  assert(!html.includes('>100%</strong>'));
  assert(!html.includes('<svg'));
});

test('样本门槛同时要求至少8条和80%，缺失/矛盾计数保持待核验', () => {
  for (const [valid, requested, expected] of [[7,8,'fail'], [24,32,'fail'], [26,32,'pass'], [8,8,'pass'],
    [8,null,'unknown'], [9,8,'unknown'], [true,8,'unknown'], ['8',8,'unknown'], [0,0,'unknown']]) {
    const input = evidence();
    Object.assign(input.kronos.quality, {valid_samples: valid, requested_samples: requested});
    const html = evaluationHtml(input, false, view());
    const row = html.split('<dt>有效路径数量与占比</dt>')[1].split('</div>')[0];
    assert(row.includes(`amv-state-${expected}`), `${valid}/${requested}: ${row}`);
  }
});

test('资格字段三态显示，研究分数不会变成模拟或独立前瞻资格', () => {
  const input = evidence({method: 'structure_kronos_ma_l2', model_id: 'a'.repeat(64),
    feature_profile: 'ma_trend_v1', score: {status: 'scored', model_type: 'l2', score: 60,
    probabilities: [.3,.2,.5], qualification: {mode:'historical_candidate', research_only:true,
    paper_allocation_eligible:false, independent_forward_status:'pending'}}});
  const html = evaluationHtml(input, false, view());
  assert(html.includes('历史研究候选；独立前瞻资格待验证'));
  assert(html.includes('尚未取得自动模拟配置资格'));
  assert(html.includes('显示原评分值'));
  assert(html.includes(input.model_id));
  assert(!html.includes('<pre'));
  delete input.score.qualification.paper_allocation_eligible;
  const unknown = evaluationHtml(input, false, view()).split('<dt>自动模拟配置资格</dt>')[1].split('</div>')[0];
  assert(unknown.includes('amv-state-unknown'));
  assert(!unknown.includes('未通过'));
});

test('所有外来标签、身份和原因都转义且不暴露原始对象', () => {
  const attack = '<img src=x onerror="alert(1)">';
  const input = evidence({label: attack, as_of: attack, model_id: attack, feature_profile: attack,
    missing_reasons: [attack, {secret: attack}], score: {reason: attack}});
  const html = evaluationHtml(input, false, view({reason_codes: [attack]}));
  assert(!html.includes('<img'));
  assert(html.includes('&lt;img'));
  assert(!html.includes('[object Object]'));
  assert(!html.includes('"secret"'));
  assert(!html.includes('alert(1)\">'));
  assert(!html.includes('amv-reasons'), '删除重复原因区，但核心标签和资格仍转义');
  assert(!html.includes('预测与评分为模型参考；不代表模拟成交或目标仓位。'));
});

test('不完整/逆序分位数及非法上涨比例不生成误导图形', () => {
  for (const summary of [{terminal_return_p10: 1, terminal_return_median: 0, terminal_return_p90: 2},
    {terminal_return_p10: null, terminal_return_median: 0, terminal_return_p90: 2},
    {terminal_return_p10: 0, terminal_return_median: NaN, terminal_return_p90: Infinity}]) {
    const input = evidence(); input.kronos.summary = {...summary, up_path_fraction: 2};
    const html = evaluationHtml(input, false, view());
    assert(!html.includes('<svg'));
    assert(!html.includes('200%'));
    assert(!html.includes('NaN'));
    assert(!html.includes('Infinity'));
  }
  const input = evidence();
  input.kronos.summary = {terminal_return_p10: -1e308, terminal_return_median: 0, terminal_return_p90: 1e308};
  const huge = evaluationHtml(input, false, view());
  assert(huge.includes('<svg'));
  assert(!/NaN|Infinity/.test(huge));
});

test('无效分类概率不画条，合法列表和命名对象保持相同含义', () => {
  for (const probabilities of [[.1,.2,.6], {down:.1,flat:null,up:.9}, [false,0,1], [0,0,2]]) {
    const html = evaluationHtml(evidence({method:'structure_kronos_ma_mlp', score:{score:50, probabilities}}), false, view());
    assert(!html.includes('class="amv-probabilities"'));
    assert(html.includes('三分类概率未完整记录'));
  }
  for (const probabilities of [[.1,.2,.7], {down:.1,flat:.2,up:.7}]) {
    const html = evaluationHtml(evidence({method:'structure_kronos_ma_mlp', score:{score:80, probabilities}}), false, view());
    for (const text of ['下跌 10%', '平稳 20%', '上涨 70%']) assert(html.includes(text));
  }
});

test('渲染只读取输入，compact不展开大矩阵或完整模型身份', () => {
  global.fetch = () => { throw new Error('渲染不能请求数据'); };
  const input = evidence({model_id:'b'.repeat(64)});
  const html = evaluationHtml(input, true, view());
  assert(html.includes('amv-compact'));
  assert(html.includes('2 / 8'));
  assert(!html.includes('可用性与资格核验'));
  assert(!html.includes('b'.repeat(64)));
  assert.equal(evaluationHtml(null), '');
});
