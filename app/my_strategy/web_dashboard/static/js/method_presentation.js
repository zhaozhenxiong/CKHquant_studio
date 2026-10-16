/* Saved method evidence only. Presentation never evaluates a model or submits work. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id);
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=value=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('zh-CN',{maximumFractionDigits:2}):'—';
  const statusNames={unavailable:'方法不可用',failed:'评估失败',deferred:'待处理',not_applicable:'未使用组合方法',not_in_scan:'未入扫描'};
  const categories={BUY:'buy',SELL:'sell',WAIT:'watch'};
  function category(row) {
    const view=row?.method_view;
    return view?.valid===true&&view.status==='complete' ? categories[view.direction]||'unavailable' : ({failed:'failed',deferred:'deferred'})[view?.status]||'unavailable';
  }
  function label(view) {
    return view?.valid===true&&view.status==='complete' ? ({BUY:'B · 方法买方参考',SELL:'S · 方法卖方参考',WAIT:'WAIT · 有效观察'})[view.direction]||'方法不可用' : statusNames[view?.status]||'方法视图未核验';
  }
  function reason(code) {
    const names={batch_budget_deferred:'本批预算已用尽，等待续批',not_in_frozen_scan_population:'未纳入这份扫描的冻结股票池',forecast_quality_unavailable:'预测质量未通过',forecast_sample_quality_invalid:'预测路径质量不合格',forecast_summary_invalid:'预测摘要不完整',score_unavailable:'评分不可用',method_evidence_incomplete:'方法证据不足',complete_method_has_missing_evidence:'方法仍有缺失证据',cutoff_reference_price_unavailable:'截止日收盘参考价缺失',kronos_assets_missing:'本地预测资源未就绪',kronos_prediction_unusable:'预测未通过可用性检查',ma_score_model_missing:'缺少兼容均线模型',ma_score_unavailable:'均线评分不可用',same_input_l2_missing:'缺少同输入 L2'};
    return names[code] || (/identity|mismatch|context|schema|parameters/.test(code)?'方法身份或参数核验不一致':/direction/.test(code)?'方法方向未通过一致性核验':String(code||'证据缺失'));
  }
  function summaryHtml(summary) {
    if(!summary)return '<p class="subtle">完整人口统计尚未读取；不按当前分页估算。</p>';
    const counts=summary.counts||summary;
    return `<div class="method-count-grid">${[['expected','扫描人口'],['buy','BUY'],['sell','SELL'],['wait','WAIT'],['unavailable','不可用'],['failed','失败'],['deferred','待处理']].map(([key,title])=>`<div><span>${title}</span><strong>${number(counts[key])}</strong></div>`).join('')}</div><p class="subtle">完整扫描人口 · BUY/SELL/WAIT 仅计完整有效的方法评估；缺失与未处理单列。这里是截止日方向，未判断首次触发。</p>`;
  }
  let instance=null,renderedKey=null;
  function analysisChart(result,request,{dirty=false}={}) {
    const host=$('analysis-prediction'),view=result?.method_view;
    if(!host)return;
    globalThis.KHQuantAnalysisAccountView?.setContext(result,request,{dirty,onChange:()=>analysisChart(result,request,{dirty})});
    if(dirty||!view){
      renderedKey=null;instance?.clear();host.hidden=true;return;
    }
    host.hidden=false;
    const forecastOption=$('analysis-chart-content')?.querySelector('option[value="forecast"]');
    if(forecastOption)forecastOption.disabled=view.status==='not_applicable';
    if(view.status==='not_applicable'&&globalThis.KHQuantAnalysisAccountView){
      instance?.clear();$('analysis-prediction-title').textContent=`${result.symbol||''} · 已有回测成交 B/S`;
      $('analysis-prediction-summary').textContent='';globalThis.KHQuantAnalysisAccountView.activate();return;
    }
    if(globalThis.KHQuantAnalysisAccountView?.isActive()){instance?.onRoute();globalThis.KHQuantAnalysisAccountView.onRoute();return;}
    $('analysis-prediction-title').textContent=`${result.symbol||''} · ${view.label||'所选方法'} · ${view.as_of||'—'}`;
    $('analysis-prediction-summary').textContent=`截止日：${label(view)}${view.forecast_valid===false?' · 预测路径质量不足':''}`;
    $('analysis-prediction-details').textContent='';$('analysis-prediction-resource').textContent='';
    if(!instance&&globalThis.KHQuantPredictionChart)instance=KHQuantPredictionChart.create({prefix:'analysis',isVisible:()=>!$('view-analysis')?.hidden&&!host.hidden&&!globalThis.KHQuantAnalysisAccountView?.isActive()});
    const key=JSON.stringify([result.symbol,view.method,view.as_of,view.prediction_ref]);
    if(key===renderedKey){instance?.onRoute();return;}
    renderedKey=key;
    if(!view.prediction_ref){instance?.clear('此结果未保存可读取的预测引用；没有自动重算。');return;}
    instance?.render({symbol:result.symbol,as_of:view.as_of,result_ref:{cache_key:view.prediction_ref},forecast:[]},
      (path,options)=>request('/api/light'+path,options));
  }
  function clearAnalysis(){renderedKey=null;instance?.clear();globalThis.KHQuantAnalysisAccountView?.clear();if($('analysis-prediction'))$('analysis-prediction').hidden=true;}
  globalThis.KHQuantMethodPresentation={category,label,reason,summaryHtml,analysisChart,clearAnalysis,onRoute:()=>{instance?.onRoute();globalThis.KHQuantAnalysisAccountView?.onRoute();}};
})();
