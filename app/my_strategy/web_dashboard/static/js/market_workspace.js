/* Market layout: reading saved evidence and drafting backtests never starts a job. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  let savedSequence=0, accountSequence=0, opening=false, displayedRun=null;
  const api = path => globalThis.KHQuantMarketBridge.request(path);
  const label = method => globalThis.KHQuantMethods?.label(method) || method || '结构与量价';
  function status(message, failed=false) { $('market-load-state').textContent=message; $('market-load-state').classList.toggle('task-error',failed); }
  async function refreshSaved() {
    const sequence=++savedSequence, select=$('market-saved-select'), previous=select.value;
    try {
      const value=await api('/api/runs?kind=scan&limit=100');
      if(sequence!==savedSequence)return;
      const rows=(Array.isArray(value)?value:value.runs||[]).filter(row=>row.kind==='scan');
      select.innerHTML=rows.length?rows.map(row=>{const s=row.summary||{}, method=s.analysis_method||s.method_context?.analysis_method;return `<option value="${esc(row.run_id)}">${esc(s.data_range?.end||s.data_end||'日期未提供')} · ${esc(label(method))} · 保存于 ${esc(String(row.created_at||'').slice(0,16).replace('T',' '))}</option>`;}).join(''):'<option value="">尚无保存结果</option>';
      if(rows.some(row=>row.run_id===previous))select.value=previous;
      $('market-open-saved').disabled=!select.value||opening;
      if(!rows.length)status('尚无保存结果。选择方法和条件后，可更新方法结果。');
    } catch(error) {if(sequence===savedSequence)status(`保存结果读取失败：${error.message}`,true);}
  }
  async function openSelected() {
    const id=$('market-saved-select').value;if(!id||opening)return;
    opening=true;$('market-open-saved').disabled=true;status('正在读取保存结果与逐股依据…');
    try {await globalThis.KHQuantMarketBridge.openSaved(id);}
    catch(error){status(`结果读取失败：${error.message}`,true);}
    finally {opening=false;$('market-open-saved').disabled=!$('market-saved-select').value;}
  }
  function syncFilters() {document.querySelectorAll('[data-market-filter]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.marketFilter===$('scan-filter').value)));}
  function update(result) {
    syncFilters();
    if(!result)return;
    const date=result.data_end||result.as_of||result.data_range?.end||'日期未提供';
    const method=result.analysis_method||result.method_context?.analysis_method||'structure';
    $('market-result-date').textContent=`当前结果 · ${date} · ${label(method)}`;
    status(`展示 ${date} 的保存方法结果；连续看涨、历史资格与每日买入价位尚待建立。`);
    if(result.run_id!==displayedRun){$('market-saved-disclosure').open=false;displayedRun=result.run_id;}
    if([...$('market-saved-select').options].some(option=>option.value===result.run_id))$('market-saved-select').value=result.run_id;
  }
  function normalizeSymbols(text) {
    const symbols=[...new Set(String(text).trim().split(/[\s,，;；]+/).filter(Boolean).map(value=>{
      const s=value.toUpperCase();if(!/^\d{6}(?:\.(?:SH|SZ))?$/.test(s))throw Error('请填写沪深 A 股代码，例如 002436.SZ。');
      const code=s.slice(0,6);if(!/^(?:60\d|68\d|00\d|30\d)\d{3}$/.test(code))throw Error('当前回测仅支持沪深 A 股代码。');
      const suffix=code.startsWith('6')?'SH':'SZ';if(s.includes('.')&&!s.endsWith('.'+suffix))throw Error('股票代码与交易所不匹配。');
      return `${code}.${suffix}`;
    }))];
    if(!symbols.length||symbols.length>10)throw Error('请选择 1–10 只股票；不会自动截取。');
    return symbols;
  }
  function validateContext(context) {
    for(const [key,allowed] of Object.entries({lookback:[128,256],horizon:[1,5,10],samples:[8,32]}))if(context[key]!=null&&!allowed.includes(Number(context[key])))throw Error(`保存结果的 ${key}=${context[key]} 不在账户回测支持范围内，请到模拟账户显式配置。`);
    if(context.research===true||(context.analysis_method&&!['structure','structure_kronos','structure_kronos_ma_mlp','structure_kronos_ma_l2'].includes(context.analysis_method)))throw Error('历史参考模型不支持此账户回测，请选择当前方法。');
    return context;
  }
  function draftBacktest(row, context={}) {
    const symbols=normalizeSymbols(row.symbol), source=validateContext({...context});
    $('market-backtest-symbols').value=symbols.join(', ');
    $('market-backtest-start').value=source.start||'2026-01-01';
    $('market-backtest-end').value=row.data_end||source.end||'';
    $('market-backtest-method').value=source.analysis_method||'structure';
    // Preserve the exact saved method parameters until the user explicitly changes method.
    backtestContext=source;
    $('market-backtest-state').textContent='已带入该股票与保存方法参数，配置回测后可检查费用和交易规则。';
    globalThis.KHQuantBoardPanel?.showBacktest();
  }
  let backtestContext={};
  function submitBacktest(event) {
    event.preventDefault();
    try {
      const symbols=normalizeSymbols($('market-backtest-symbols').value), start=$('market-backtest-start').value,end=$('market-backtest-end').value;
      if(!start||!end||start>=end)throw Error('请填写有效的回测开始与结束日期。');
      const context=validateContext({...backtestContext,symbols,start,end,analysis_method:$('market-backtest-method').value});
      globalThis.KHQuantLightResearch.transferToAccounts(context);
    } catch(error){$('market-backtest-state').textContent=error.message;}
  }
  async function refreshBacktests() {
    const sequence=++accountSequence, host=$('market-backtest-results');
    host.innerHTML='<p class="empty-copy">正在读取已保存账户…</p>';
    try {
      const data=await api('/api/light/saved');if(sequence!==accountSequence)return;
      const rows=(data.results||[]).filter(row=>row.pin_count>0);
      host.innerHTML=rows.length?rows.map(row=>{const identity=row.identity||{}, symbols=identity.symbols||(identity.sources||[]).map(source=>source.symbol).filter(Boolean), savedAt=typeof row.created_at==='number'?new Date(row.created_at*1000).toLocaleString('zh-CN',{hour12:false}):String(row.created_at||'').slice(0,16).replace('T',' ');return `<article class="market-saved-account"><div><strong>${esc(symbols.join('、')||'已保存模拟账户')}</strong><p>${esc(identity.start||'')} — ${esc(identity.end||'')} · ${esc(savedAt)}</p><small>${esc(row.cache_key)}</small></div><a class="button-link" href="#watchlist?tab=accounts&amp;result=${encodeURIComponent(row.cache_key)}">查看回测</a></article>`;}).join(''):'<p class="empty-copy">尚无已保存账户。配置回测后可保存并追踪结果。</p>';
    } catch(error){if(sequence===accountSequence)host.innerHTML=`<p class="empty-copy">读取失败：${esc(error.message)}</p>`;}
  }
  function onTab(tab) {
    globalThis.KHQuantMarketList?.onRoute();
    if(tab==='backtest'){
      if(!$('market-backtest-end').value)$('market-backtest-end').value=$('scan-end').value;
      refreshBacktests();
    }
  }
  $('market-conditions-toggle').addEventListener('click',()=>{$('market-conditions').open=!$('market-conditions').open;});
  $('market-conditions').addEventListener('toggle',()=>$('market-conditions-toggle').setAttribute('aria-expanded',String($('market-conditions').open)));
  $('market-refresh-saved').addEventListener('click',refreshSaved);
  $('market-open-saved').addEventListener('click',openSelected);
  $('market-saved-select').addEventListener('change',()=>{$('market-open-saved').disabled=!$('market-saved-select').value||opening;});
  document.querySelectorAll('[data-market-filter]').forEach(button=>button.addEventListener('click',()=>{$('scan-filter').value=button.dataset.marketFilter;syncFilters();$('scan-filter').dispatchEvent(new Event('change'));}));
  $('scan-filter').addEventListener('change',syncFilters);
  $('market-backtest-form').addEventListener('submit',submitBacktest);
  $('market-backtest-method').addEventListener('change',()=>{backtestContext={};$('market-backtest-state').textContent='已更改方法；其余参数使用模拟账户中的草稿，运行前请检查。';});
  $('market-refresh-backtests').addEventListener('click',refreshBacktests);
  globalThis.KHQuantMarketWorkspace={update,onTab,draftBacktest,refreshSaved};
  refreshSaved();
  if(!$('scan-backtest-panel').hidden)onTab('backtest');
})();
