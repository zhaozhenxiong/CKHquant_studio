/* Local prediction and actual paper inventory, independent of reference markers. */
(() => {
 'use strict';
 const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const num=(x,d=2)=>x==null||!Number.isFinite(Number(x))?'—':Number(x).toLocaleString('zh-CN',{maximumFractionDigits:d});
 const pct=x=>x==null?'—':`${num(x*100)}%`;
 const labels={structure_rules:'结构与量价',structure_kronos:'结构与量价 + Kronos',structure_kronos_score:'结构与量价 + Kronos + MLP',structure_kronos_l2:'结构与量价 + Kronos + 同输入 L2'};
 const signed=(x,unit='')=>x==null||!Number.isFinite(Number(x))?'未记录':`${Number(x)>0?'+':''}${num(x)}${unit}`;
 const profileOf=value=>value?.feature_profile||value?.artifact?.feature_profile||value?.identity?.feature_profile||value?.lineage?.feature_profile||'mixed_v1';
 const accountLabel=(key,value)=>{const ma=profileOf(value)==='ma_trend_v1'||profileOf(result)==='ma_trend_v1';return key==='structure_kronos_score'?`结构与量价 + Kronos + ${ma?'均线模型 MLP':value?.lineage?.score_model_type?.toUpperCase()||result?.score_model_type?.toUpperCase()||(result?.version==='structure_kronos_sim_v1'?'L2 评分':'MLP（历史融合）')}`:key==='structure_kronos_l2'&&ma?'结构与量价 + Kronos + 同输入 L2（均线）':labels[key]||key;};
 let desiredAccount=null;
 let observationManifest=null,result=null,active=null,timer=null,chart=null,tradeChart=null;
 let starting=false,polling=false,refreshing=null,resultSequence=0,routeResult=null,enabledAccounts={};
 const accountKinds=new Set(['simulation','daily']);
 const owner=kind=>accountKinds.has(kind)?'accounts':'prediction';
 const activeStates=new Set(['queued','running','cancelling']);
 const currentOwner=()=>globalThis.KHQuantNavigation?.parse().tab==='accounts'?'accounts':'prediction';
 const accountVisible=()=>!$('view-watchlist').hidden&&!$('watchlist-accounts-panel').hidden;
 const ownerId=(section,suffix)=>section==='accounts'?'light-account-'+suffix:'light-'+suffix;
 function navigate(section,resultId){globalThis.KHQuantNavigation?.go(section==='accounts'?'watchlist':'simulation',section==='accounts'?{tab:'accounts',...(resultId?{result:resultId}:{})}:{});}
 async function api(path,body,options={}){const r=await fetch('/api/light'+path,{...options,method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json',...(sessionStorage.getItem('khquant_api_token')?{Authorization:'Bearer '+sessionStorage.getItem('khquant_api_token')}: {})},body:body===undefined?undefined:JSON.stringify(body)});const v=await r.json();if(!r.ok)throw Error(typeof v.detail==='string'?v.detail:v.detail?.message||v.detail?.code||JSON.stringify(v.detail));return v;}
 function error(e,section=currentOwner()){const node=$(ownerId(section,'error'));node.textContent=e.message||String(e);node.hidden=false;}
 function busy(value){$('light-enable-daily').disabled=value||!result;document.querySelectorAll('#view-simulation button[data-start],#watchlist-accounts-panel button[data-start]').forEach(b=>b.disabled=value||b.id==='light-collect'&&observationManifest?.status==='completed');for(const section of ['prediction','accounts'])$(ownerId(section,'cancel')).hidden=!value||!active||owner(active.kind)!==section;}
 function progress(job){const section=owner(job.kind),p=job.progress||{},label=({queued:'等待计算',running:'计算中',cancelling:'正在取消',succeeded:'已完成',failed:'失败',cancelled:'已取消',partial:'部分完成',blocked:'条件未满足',interrupted:'服务中断'})[job.status]||job.status;
  $(ownerId(section,'progress')).textContent=`${label} ${activeStates.has(job.status)?`${p.stage||''} ${p.symbol||''} ${p.date||''} ${p.completed!=null?`${p.completed}/${p.total}`:''}`:''}`;
  const other=section==='accounts'?'prediction':'accounts',href=section==='accounts'?'#watchlist?tab=accounts':'#simulation';$(ownerId(other,'progress')).innerHTML=`${section==='accounts'?'账户':'预测或学习'}任务：${esc(label)} · <a href="${href}">查看${activeStates.has(job.status)?'进度与取消':'任务结果'}</a>`;
  const open=$(ownerId(section,'open-task'));open.hidden=!job.result||activeStates.has(job.status);open.dataset.jobId=job.job_id;
 }
 function table(head,rows){return `<div class="light-table-scroll"><table><thead><tr>${head.map(h=>`<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
 function renderCandidates(status){
  const models=(status.score_candidates||[]).filter(m=>profileOf(m)===$('light-sim-feature-profile').value),selected=$('light-score-model').value;
  $('light-score-model').innerHTML='<option value="">自动选择兼容候选</option>'+models.map(m=>{const id=m.model_id||m.model_ref?.cache_key||m.cache_key;return `<option value="${esc(id)}">${esc((m.model_type||'MLP').toUpperCase())} · ${esc(m.end||m.training_end||m.identity?.end||'')} · ${esc(id.slice(0,12))}</option>`;}).join('');
  if(selected&&!models.some(m=>(m.model_id||m.model_ref?.cache_key||m.cache_key)===selected))$('light-score-model').innerHTML+=`<option value="${esc(selected)}">已绑定候选（执行时核验） · ${esc(selected.slice(0,12))}</option>`;
  if(selected)$('light-score-model').value=selected;
  const training=(status.score_candidates||[]).filter(m=>profileOf(m)===$('light-feature-profile').value),latest=training[0];
  $('light-score-state').textContent=latest?`评分模型 · ${latest.model_type?.toUpperCase()||'L2'} · ${latest.feature_dimensions??'—'} 项输入 · ${training.length} 个候选。历史回测与每日账户资格分别核验。`:'尚无兼容评分模型；可在下方学习区域训练候选。';
 }
 function renderEnabledAccounts(value){enabledAccounts=value.accounts||{};const entries=Object.values(enabledAccounts);$('light-accounts-state').innerHTML=entries.length?table(['账户','股票池','每日续算','最近日期'],entries.map(a=>[esc(String(a.account_id||'').slice(0,12)),esc((a.symbols||[]).join('、')),a.enabled?`已启用 · ${({shadow_research:'影子研究',paper:'纸上账户'})[a.continuation_mode||a.role]||'用途未记录'}`:'未启用',esc(a.last_date||'—')])):'尚未启用每日续算账户；打开保存结果不会自动启用。';if(value.last_daily)renderDaily(value.last_daily);}
 async function refresh(){
  if(refreshing)return refreshing;
  refreshing=(async()=>{try{const [status,saved,accounts]=await Promise.all([api('/status'),api('/saved'),api('/accounts')]);
   $('light-model-state').textContent=`Kronos-base · 本地单份模型 · ${status.assets.status||(status.assets.ready?'就绪':'待核验')}`;
   renderCandidates(status);renderEnabledAccounts(accounts);await refreshManifests();
   const running=(status.jobs||[]).find(j=>j.kind!=='incremental'&&activeStates.has(j.status));if(running&&!active&&!starting){active={job_id:running.job_id,kind:running.kind};progress(running);busy(true);poll();}
   const previous=$('light-saved').value,results=(saved.results||[]).filter(r=>r.pin_count>0);
   $('light-saved').innerHTML='<option value="">选择结果</option>'+results.map(r=>{const i=r.identity||{};return `<option value="${esc(r.cache_key)}">${esc(i.start)} → ${esc(i.end)} · ${esc((i.sources||[]).map(x=>x.symbol).join('、'))} · ${i.partial_stage?'部分完成':i.version==='agent_mlp_sim_v1'?'完整回放':'旧版结果'} · ${esc(r.cache_key.slice(0,12))}</option>`;}).join('');
   if(results.some(r=>r.cache_key===previous))$('light-saved').value=previous;
   if(!results.length&&!result)$('light-new-account').open=true;
   $('light-storage-state').textContent='预览缓存7天 · 保存账户长期保留 · 日更不自动训练';
   const response=await fetch('/api/data/status'),data=await response.json(),latest=data.latest_date||data.latest||data.data_end||$('latest-date')?.textContent;
   if(/^\d{4}-\d{2}-\d{2}$/.test(latest||''))for(const id of ['light-end','light-predict-date','light-daily-date'])if(!$(id).value)$(id).value=latest;
  }catch(e){error(e);}})();
  try{return await refreshing;}finally{refreshing=null;}
 }
 async function loadManifest(id){
  observationManifest=id?await api('/score/observations/'+encodeURIComponent(id)):null;
  if(id)sessionStorage.setItem('khquant_observation_manifest',id);else sessionStorage.removeItem('khquant_observation_manifest');
  $('light-collect').textContent=observationManifest?'采集下一批未完成观察':'采集一批历史观察';
  $('light-collect').disabled=!!active||starting||observationManifest?.status==='completed';
  $('light-observe-status').textContent=observationManifest?`${observationManifest.symbols?.join('、')} · ${observationManifest.start} → ${observationManifest.end} · 已尝试 ${observationManifest.total_calls} 次 / ${observationManifest.batch_count} 批 · ${JSON.stringify(observationManifest.coverage)}。点击按钮才启动新一批，取消任务不会恢复。`:'新建时冻结学习股票池与观察日期；刷新不会自动采集。';
 }
 async function refreshManifests(){
  const data=await api('/score/observations'),select=$('light-observe-manifest');
  const selected=observationManifest?.manifest_id||select.value||sessionStorage.getItem('khquant_observation_manifest')||'';
  select.innerHTML='<option value="">新建观察人口</option>'+(data.manifests||[]).map(m=>`<option value="${esc(m.manifest_id)}">${esc(m.start)} → ${esc(m.end)} · ${esc(m.symbols.join('、'))} · ${m.total_calls}次 / ${m.batch_count}批</option>`).join('');
  if([...select.options].some(o=>o.value===selected))select.value=selected;
  await loadManifest(select.value);
 }
 async function submit(path,spec){
  if(starting||active)return;const kind=path==='/simulate'?'simulation':path==='/daily'?'daily':path==='/predict'?'prediction':path==='/score/observations'?'score_observations':'score_training',section=owner(kind);
  $(ownerId(section,'error')).hidden=true;starting=true;busy(true);
  try{const job=await api(path,spec);active={job_id:job.job_id,kind:job.kind||kind};progress({...job,kind:active.kind});busy(true);poll();}catch(e){busy(false);error(e,section);}finally{starting=false;}
 }
 async function deliver(job){
  if(job.kind==='prediction')renderPrediction(job.result);
  else if(job.kind==='simulation'){const request=++resultSequence;const value=job.result.saved?await api('/results/'+job.result.result_id):job.result;if(request===resultSequence){result=value;renderSimulation();}}
  else if(job.kind==='daily'){renderDaily(job.result);await refresh();}
  else if(job.kind==='score_observations'){observationManifest=job.result;sessionStorage.setItem('khquant_observation_manifest',job.result.manifest_id);await refreshManifests();$('light-score-result').textContent=JSON.stringify(job.result,null,2);}
  else{await refresh();$('light-score-result').textContent=JSON.stringify(job.result,null,2);}
 }
 async function poll(){
  clearTimeout(timer);timer=null;if(document.hidden||!active||polling)return;polling=true;const current=active;
  try{const job=await api('/jobs/'+encodeURIComponent(current.job_id));if(active!==current)return;progress(job);
   if(activeStates.has(job.status)){timer=setTimeout(poll,1200);return;}active=null;busy(false);
   if(['failed','blocked','interrupted'].includes(job.status)){error(Error(job.error||job.result?.reason||'任务未完成'),owner(job.kind));return;}
   if(job.status==='cancelled'){if(job.kind==='score_observations')await refreshManifests();return;}
   if(job.result)await deliver(job);
  }catch(e){error(e,owner(current.kind));if(active===current)timer=setTimeout(poll,2500);}
  finally{polling=false;if(active&&!timer&&!document.hidden)timer=setTimeout(poll,1200);}
 }
 document.addEventListener('visibilitychange',()=>{clearTimeout(timer);if(!document.hidden&&active)poll();});
 function renderDaily(value){
  const accounts=value.accounts||[],pending=accounts.filter(a=>a.status!=='completed');
  const summary=`${value.price_date} · 账户完成 ${accounts.length-pending.length}，待处理 ${pending.length} · 关注 ${value.requested} 项：完成 ${value.succeeded}，失败 ${value.failed}，延后 ${value.deferred} · 未启动训练或全市场补结构`;
  const roles={shadow_research:'影子研究',paper:'纸上账户'},states={blocked:'条件未满足',partial:'部分完成',failed:'失败',cancelled:'已取消',pending:'待处理'};
  $('light-daily-state').innerHTML=`<p>${esc(summary)}</p>`+(pending.length?table(['账户','用途','状态','原因'],pending.map(a=>[
   esc(String(a.account_id||a.result_id||'未记录').slice(0,10)),esc(roles[a.role]||a.role||'用途未记录'),esc(states[a.status]||a.status||'待处理'),
   esc(a.status==='blocked' && /版本改变|版本.*变化|代码身份/.test(a.reason||'') ? '历史结果可查看；算法版本已变化，续算需新建账户分支。' : a.reason||a.error||(a.dependency_errors||[]).map(d=>d.error||d.reason||d.dependency).filter(Boolean).join('；')||'服务未返回具体原因，请查看任务详情')
  ])): '');
 }
 function createChart(id){const node=$(id),css=getComputedStyle(document.documentElement),color=name=>css.getPropertyValue('--'+name).trim();node.replaceChildren();return LightweightCharts.createChart(node,{autoSize:true,height:280,layout:{background:{color:color('panel')||'transparent'},textColor:color('ink')||'#55657a'},grid:{vertLines:{color:color('border')||'#e6ebf2'},horzLines:{color:color('border')||'#e6ebf2'}},timeScale:{timeVisible:false,borderVisible:false},rightPriceScale:{borderVisible:false}});}
 function line(ch,data,color){const s=ch.addLineSeries({color,lineWidth:2,priceLineVisible:false,lastValueVisible:false});s.setData(data.filter(d=>Number.isFinite(d.value)));return s;}
 function renderPrediction(r){$('light-prediction').hidden=false;const q=r.quality||{},s=r.summary||{};$('light-prediction-title').textContent=`${r.symbol||r.input?.symbol||$('light-symbol').value} · 未来${r.forecast?.length||r.input?.horizon||'—'}个交易日`;
  $('light-prediction-summary').textContent=`${r.reused?'复用缓存':'本次预测'} · 有效路径 ${q.valid_samples??q.valid_paths??'—'}/${q.requested_samples??q.total_samples??q.total_paths??$('light-samples').value} · 上涨路径占比 ${pct(s.up_path_fraction??s.up_fraction??s.up_probability)} · 预期变化 ${pct(s.terminal_return_median??s.expected_return??s.mean_return)} · ${q.prediction_usable?'可供研究':'路径质量不足，仅展示'}。采样频率未经校准，不是可靠胜率。`;
  const rows=r.forecast||[],date=row=>String(row.date||row.time).slice(0,10),middle=row=>row.close??row.close_median??row.median?.close??row.p50;
  globalThis.KHQuantPredictionChart?.render(r,(path,options)=>api(path,undefined,options));
  $('light-prediction-details').innerHTML=table(['预测日期','收盘中位','10%','90%'],rows.map(row=>[esc(date(row)),num(middle(row)),num(row.close_p10??row.p10),num(row.close_p90??row.p90)]));
  $('light-prediction-resource').textContent=`计算设备 ${(r.resource?.actual_device||'未知').toUpperCase()} · 原计算耗时 ${num(r.resource?.elapsed_seconds)} 秒 · ${r.reused?'本次直接读取缓存':'冻结权重推理，无模型训练'}`;
 }
 function renderSimulation(){if(!result)return;$('light-results').hidden=false;const c=result.prediction_coverage||{};$('light-result-meta').textContent=`结果 ${String(result.result_id||'预览').slice(0,12)} · ${result.symbols.join('、')} · ${result.start} → ${result.scheduler?result.scheduler.complete_through||'尚无完整评估日':result.end}${result.status==='partial'?'（请求至 '+result.end+'，尚未完整评估）':''} · ${result.reused?'复用已核验结果':'新回放'} · ${result.saved?'已保存账户':'预览缓存'}。Kronos请求 ${c.requested}，可用 ${c.succeeded}，无效 ${c.unusable}，失败 ${c.failed}，预算外 ${c.budget_skipped}。`;
  $('light-comparison').innerHTML=table(['独立账户','费用后收益','夏普','最大回撤','成交笔数'],Object.keys(labels).filter(k=>result.comparisons[k]).map(k=>{const v=result.comparisons[k];return [k,v];}).map(([k,v])=>{const m=v.metrics||{};return [esc(accountLabel(k,v)),pct(m.total_return??m.return),num(m.sharpe??m.sharpe_ratio),pct(m.max_drawdown),num(m.trade_count??v.trades?.length,0)];}));
  const priorAccount=$('light-account').value;$('light-account').innerHTML=Object.keys(labels).filter(k=>result.comparisons[k]).map(k=>`<option value="${k}">${esc(accountLabel(k,result.comparisons[k]))}</option>`).join('');if(desiredAccount&&result.comparisons[desiredAccount]){$('light-account').value=desiredAccount;desiredAccount=null;}else if(result.comparisons[priorAccount])$('light-account').value=priorAccount;
  $('light-limitations').textContent=(result.limitations||[]).join(' ');renderReport();$('light-trade-symbol').innerHTML=result.symbols.map(s=>`<option>${esc(s)}</option>`).join('');$('light-enable-daily').disabled=!!active||starting;renderAccount();
 }
 function renderReport(){
  const r=result.experiment_report,host=$('light-experiment-report');if(!host)return;
  if(!r){host.textContent='旧结果未保存同日配对研究报告。';return;}
  host.innerHTML=table(['账户','平均仓位','换手','完整持仓周期','费用'],Object.entries(r.metrics||{}).map(([k,m])=>[esc(accountLabel(k,result.comparisons[k])),pct(m.average_exposure),num(m.turnover),num(m.completed_cycles,0),num(m.total_fees)]));
  const note=document.createElement('p');note.textContent='费用压力 '+Object.entries(r.cost_stress||{}).map(([k,v])=>k+'倍：'+(v.status==='completed'?'已回放':'待回放')).join('；')+'。配对日期块抽样及样本充分性见下方；样本不足不会授予自动增仓资格。';host.append(note);
  const detail=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');summary.textContent='配对置信区间与核验明细';pre.textContent=JSON.stringify({readiness:r.readiness,paired_differences:r.paired_differences,fusion_paired_differences:r.fusion_paired_differences,research_sufficiency:r.research_sufficiency,issues:r.comparison_issues,sources:result.experiment_report_derivation},null,2);detail.append(summary,pre);host.append(detail);
 }
 function renderAccount(){if(!result||!accountVisible())return;const chosen=$('light-account').value,v=result.comparisons[chosen],daily=v.daily||[],trades=v.trades||[],orders=v.orders||[];
  $('light-account-summary').textContent=`${accountLabel(chosen,v)} · ${v.metrics?.sharpe_reasons?.map(r=>({no_fills:'无实际成交，夏普不适用',zero_or_insufficient_variance:'净值波动不足',fewer_than_60_sessions:'不足60个交易日',fewer_than_60_verified_sessions:'不足60个核验交易日',stale_held_valuation:'持仓存在缺失估值日'})[r]||r).join('；')||'夏普按含空仓日的每日净值及费用计算，年化252日，无风险年利率0。'} 期末仓位保留，不强制平仓。`;
  if(chart)chart.remove();chart=createChart('light-equity-chart');const mode=$('light-chart-mode').value;
  const baseline=result.identity?.cash||v.lineage?.config?.initial_cash||v.metrics?.initial_cash||v.config?.initial_cash||daily[0]?.initial_cash||Number($('light-cash').value);
  line(chart,daily.map(d=>({time:String(d.date).slice(0,10),value:Number(mode==='equity'?(d.equity??d.nav*baseline)/baseline:mode==='drawdown'?(d.drawdown??0)*100:(d.exposure??d.gross_exposure??d.position_weight??0)*100)})),'#2563eb');chart.timeScale().fitContent();
  renderDiagnostics(v);
  $('light-benchmarks').innerHTML=table(['基准','费用后收益','夏普','最大回撤'],Object.entries(v.benchmarks||{}).map(([k,b])=>[esc(({cash:'现金',buy_hold_initial_budget:'同预算买入持有',buy_hold_full_budget:'满预算买入持有',buy_hold:'同预算买入持有',buy_hold_full:'满预算买入持有',buy_and_hold:'同预算买入持有',full_buy_and_hold:'满预算买入持有'})[k]||k),pct(b.metrics?.total_return),num(b.metrics?.sharpe),pct(b.metrics?.max_drawdown)]));
  $('light-trades').innerHTML=trades.length?table(['成交日','股票','B / S','股数变化','成交价','费用','剩余股数','成交仓位变化','理由'],trades.map(t=>{const x=t.execution_change||{};return [esc(t.date),esc(t.symbol),`<strong class="${t.action==='BUY'?'valuation-gain':'valuation-loss'}">${t.action==='BUY'?'B 买入':'S 卖出'}</strong>`,signed(x.share_delta??(['SELL','REDUCE'].includes(t.action)?-t.shares:t.shares),' 股'),num(t.price),num(t.fee),num(x.shares_after??t.position_after??t.shares_after??t.holdings_after,0),x.weight_before!=null&&x.weight_after!=null?`${pct(x.weight_before)} → ${pct(x.weight_after)}<br>${signed((x.trade_weight_delta??x.weight_after-x.weight_before)*100,' 个百分点')}`:'旧结果未记录',esc(reason(t.reason))];})):'<p>该账户没有实际成交。下方列出阻挡条件与委托原因；没有成交时夏普不适用。</p>';
  $('light-orders').innerHTML=orders.length?table(['信号日','股票','委托状态','计划股数变化','信号仓位 → 目标','计划仓位变化','原因'],orders.slice(-150).map(o=>{const p=o.quantity_plan||{},target=p.target_weight??o.target_weight??o.target;return [esc(o.date||o.signal_date),esc(o.symbol),esc(({queued:'待执行',pending:'待执行',filled:'已成交',partially_filled:'部分成交',expired:'已过期',cancelled:'已取消',rejected:'已拒绝'})[o.status]||o.status),signed(p.share_delta,' 股'),p.weight_before!=null?`${pct(p.weight_before)} → ${pct(target)}`:target!=null?pct(target):'旧结果未记录',p.weight_delta!=null?signed(p.weight_delta*100,' 个百分点'):'未记录',esc(reason(o.terminal_reason||o.rejection_reason||o.reason))];}))+'<p class="subtle">计划按信号日收盘与权益计算，成交按实际开盘与当日权益计算；两者分别留痕。</p>':'<p>暂无委托。</p>';
  $('light-decisions').textContent=JSON.stringify((v.decisions||[]).slice(-30),null,2);
  chart.subscribeClick(p=>{if(p.time)inventory(v,String(p.time));});renderTrades(v);inventory(v,daily.at(-1)?.date);
 }
 function renderDiagnostics(v){
  const q=v.qualification_mode||result.qualification_mode||result.research_mode||'历史记录', scheduler=result.scheduler||{};
  $('light-qualification').textContent=`用途：${({historical_candidate:'历史候选研究',paper:'纸上账户'})[q]||q}${result.account_role==='shadow_research'?' · 影子研究':''}。风险检查截至 ${scheduler.risk_through||result.end}；完整决策截至 ${result.scheduler?scheduler.complete_through||'尚无完整评估日':result.end}${result.status==='partial'?' · 尚有待完成阶段':''}。历史候选可做对照，自动纸上增仓需独立政策资格；前瞻认证另行验证。`;
  const f=v.opportunity_funnel||v.diagnostics?.opportunity_funnel||result.diagnostics?.opportunity_funnel||{};
  const countKeys=['evaluated','eligible','blocked','unique_opportunities','orders','fills','completed_cycles'];
  const rows=Array.isArray(f)?f.map(x=>[esc(reason(x.reason||x.stage)),num(x.count,0)]):countKeys.filter(k=>f[k]!=null).map(k=>[esc(reason(k)),num(f[k],0)]);
  const causes=Object.entries(f.primary_reason_counts||{}).filter(([key])=>key!=='eligible').sort((a,b)=>b[1]-a[1]);
  const opportunities=v.opportunities||[], recent=opportunities.slice(-30);
  const outcome=x=>x.primary_reason==='eligible'?'进入委托评估':x.primary_reason?'未进入委托':reason(x.status||x.action)||'未记录';
  $('light-funnel').innerHTML=(rows.length?table(['机会流转','次数'],rows)+'<p class="subtle">评估次数按股票、候选与决策日计；同一机会可连续复核。进入委托评估后仍需满足资金、整手和成交条件。</p>':'<p>这份旧结果未保存完整机会统计；可用新版本重新回放。</p>')+(causes.length?table(['首要阻挡原因','次数'],causes.map(([key,n])=>[esc(reason(key)),num(n,0)])):'')+(recent.length?'<details><summary>最近30次机会评估</summary>'+table(['日期','股票','结果','原因'],recent.map(x=>[esc(x.date),esc(x.symbol),esc(outcome(x)),esc([...new Set([x.primary_reason||x.reason,...(x.all_reasons||x.reason_codes||[])].filter(Boolean))].map(reason).join('；'))]))+'</details>':'');
  const agents=v.agent_evidence||[], agentNames={structure:'结构',price_volume:'量价',kronos:'Kronos',fusion:'融合评分',risk:'独立风控'}, directions={BUY:'偏买',SELL:'偏卖',WAIT:'等待',HOLD:'持有',EXIT:'退出',REDUCE:'减仓',ADD:'加仓'};
  $('light-agents').innerHTML=Array.isArray(agents)&&agents.length?table(['日期','股票','各项判断','决策原因'],agents.slice(-20).map(x=>[esc(x.date),esc(x.symbol),Object.entries(x.agents||{}).map(([key,a])=>`${esc(agentNames[key]||key)}：${esc(a.missing_reason?reason(a.missing_reason):directions[a.direction]||'无方向')}${a.score!=null?' · '+num(a.score)+'分':''}`).join('<br>'),esc(reason(x.primary_reason||x.reason)||'按账户规则复核')])):'<p>展开下方决策依据查看逐日证据。独立风控可直接触发退出，不等待模型一致。</p>';

 }
 function inventory(v,day){const row=(v.daily||[]).find(d=>String(d.date).slice(0,10)===String(day).slice(0,10));if(!row)return;$('light-inventory').innerHTML=`<p><strong>${esc(day)}</strong> · 现金 ${num(row.cash)} · 总权益 ${num(row.equity)} · 实际仓位 ${pct(row.exposure)}</p>`+table(['股票','实际股数','可卖','成本','市值','目标仓位'],Object.entries(row.positions||{}).map(([s,p])=>[esc(s),num(p.shares,0),num(p.sellable_shares,0),num(p.average_cost),num(p.value),pct(p.target_weight)]));}
 function renderTrades(v){if(!accountVisible())return;if(tradeChart)tradeChart.remove();tradeChart=createChart('light-trade-chart');const symbol=$('light-trade-symbol').value,series=tradeChart.addCandlestickSeries({upColor:'#d44954',downColor:'#278979',borderVisible:false,wickUpColor:'#d44954',wickDownColor:'#278979'});series.setData((result.chart_bars?.[symbol]||[]).map(b=>({time:String(b.date).slice(0,10),open:b.open,high:b.high,low:b.low,close:b.close})));series.setMarkers((v.trades||[]).filter(t=>t.symbol===symbol).map(t=>({time:String(t.date).slice(0,10),position:t.action==='BUY'?'belowBar':'aboveBar',color:t.action==='BUY'?'#d44954':'#278979',shape:t.action==='BUY'?'arrowUp':'arrowDown',text:t.action==='BUY'?'B':'S'})));tradeChart.timeScale().fitContent();tradeChart.subscribeClick(p=>{if(p.time)inventory(v,String(p.time));});}
 function reason(r){return ({evaluated:'已评估',eligible:'进入委托评估',blocked:'暂不入场',unique_opportunities:'独立机会',orders:'委托',fills:'实际成交',completed_cycles:'完整持仓周期',structure_entry_trend_not_ready:'结构入场趋势尚未确认',add_requires_profit_and_interval:'加仓需满足浮盈与间隔',exit_pending:'退出委托待成交',reentry_cooldown:'离场后冷静期',missing_or_invalid_structure_risk_boundary:'缺少有效结构风险边界',simple_score_below_threshold:'融合评分低于入场阈值',source_candidates:'已确认来源候选',held_reviews:'持仓复核',candidate_reviews:'观察期复核',forecast_requested:'预测请求',forecast_usable:'预测可用',entry_allowed:'可进入交易评估',qualification_blocked:'模型资格待核验',score_blocked:'评分门槛未满足',risk_blocked:'风险条件阻挡',lot_budget_blocked:'资金不足一手',filled:'已成交',expired:'观察期结束',confirmed_structure_entry:'确认结构后建仓',confirmed_structure_sell:'结构卖出：减仓',confirmed_profitable_add:'浮盈且新证据确认后加仓',frozen_stop_gap:'跳空触发冻结保护线',frozen_price_stop:'触发价格保护线',ma_trend_failure:'均线趋势失效',holding_expiry:'到达持仓期限',time_failure_without_one_r:'持仓未达到预期进展',cash_risk_or_lot_budget_insufficient:'资金或风险预算不足一手',simple_score_policy_not_passed:'评分未通过政策资格',forecast_missing_or_unusable:'预测缺失或质量不足',forecast_direction_blocked:'预测方向未满足入场条件',portfolio_drawdown_exit:'组合回撤触发退出',kronos_weakness_reduction:'连续预测转弱：减仓',consecutive_forecast_weakness:'连续预测转弱：减仓',consecutive_fusion_weakness:'连续融合证据转弱：减仓'})[r]||r;}
 function symbols(id='light-symbols',limit=10){const values=[...new Set($(id).value.split(/[\s,，;；]+/).filter(Boolean).map(s=>s.toUpperCase()))];if(!values.length||values.length>limit)throw Error(`请选择1至${limit}只股票；不会自动截取股票池`);return values;}
 function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 function updateFeatureControls(){
  const ma=$('light-feature-profile').value==='ma_trend_v1',variant=$('light-score-variant'),previous=variant.value;
  variant.innerHTML=ma?'<option value="A">均线趋势 · 20 项数值特征</option>':'<option value="A">A · 结构与量价</option><option value="B">B · 加入 Kronos 摘要</option>';
  variant.value=ma?'A':previous==='B'?'B':'A';
  if(ma)$('light-score-type').value='mlp';
  $('light-score-type').disabled=ma;variant.disabled=ma;$('light-observe-panel').hidden=ma;
  $('light-feature-help').textContent=ma?'均线模型使用 20 项数值特征，涵盖均线位置、斜率与距离；配对 L2 使用相同训练人口和输入。Kronos 单独提供证据。单次最多10分钟，不训练 Kronos 权重。':'历史融合 A 使用结构量价数值与缺失标记；B 需要冻结的历史预测覆盖，证据不足显示待补齐。单次最多10分钟。';
 }

 function simulationSpec(){return {symbols:symbols(),start:$('light-start').value,end:$('light-end').value,initial_cash:Number($('light-cash').value),samples:Number($('light-sim-samples').value),lookback:Number($('light-sim-lookback').value),horizon:Number($('light-sim-horizon').value),device:$('light-sim-device').value,seed:Number($('light-sim-seed').value),save:$('light-save').checked,score_model_id:$('light-score-model').value||null,score_model_type:$('light-sim-score-type').value,feature_profile:$('light-sim-feature-profile').value,research_mode:'historical_candidate',candidate_sessions:Number($('light-candidate-days').value),fusion_mode:$('light-fusion').value,cost_stress:Number($('light-cost-stress').value)};}
 function updateSimulationControls(){const ma=$('light-sim-feature-profile').value==='ma_trend_v1';if(ma)$('light-sim-score-type').value='mlp';$('light-sim-score-type').disabled=ma;$('light-sim-parameters').textContent=`回测专用：历史 ${$('light-sim-lookback').value} 根 · 未来 ${$('light-sim-horizon').value} 日 · ${$('light-sim-samples').value} 条路径 · 种子 ${$('light-sim-seed').value} · ${$('light-sim-device').value.toUpperCase()} · ${ma?'均线 MLP / 同输入 L2':'历史融合 '+$('light-sim-score-type').value.toUpperCase()}。预测页和学习页的修改不影响此草稿。`;}
 function transferToPrediction(context){$('light-symbol').value=context.symbol||context.symbols?.[0]||'';if(context.end)$('light-predict-date').value=context.end;for(const key of ['lookback','horizon','samples','seed','device'])if(context[key]!=null)$('light-'+key).value=String(context[key]);$('light-transfer-context').hidden=false;$('light-transfer-context').textContent='已带入股票与参数；点击预测按钮才开始计算。';saveDraft('prediction');navigate('prediction');}
 function transferToAccounts(context){const values=context.symbols||[context.symbol];if(!values.length||values.length>10)throw Error('请选择1至10只股票带入回测；不会自动截取');$('light-symbols').value=values.join(', ');if(context.start)$('light-start').value=context.start;if(context.end)$('light-end').value=context.end;
  for(const key of ['lookback','horizon','samples','seed','device'])if(context[key]!=null)$('light-sim-'+key).value=String(context[key]);
  if(context.analysis_method){$('light-sim-feature-profile').value='ma_trend_v1';$('light-score-model').value='';if(context.score_model_id){const select=$('light-score-model');if(![...select.options].some(o=>o.value===context.score_model_id))select.innerHTML+=`<option value="${esc(context.score_model_id)}">已绑定均线候选 · ${esc(context.score_model_id.slice(0,12))}</option>`;select.value=context.score_model_id;}
   desiredAccount=({structure:'structure_rules',structure_kronos:'structure_kronos',structure_kronos_ma_mlp:'structure_kronos_score',structure_kronos_ma_l2:'structure_kronos_l2'})[context.analysis_method]||'structure_rules';}
  updateSimulationControls();$('light-new-account').open=true;$('light-account-transfer').hidden=false;$('light-account-transfer').textContent=`已带入 ${values.join('、')}；这是新回测草稿，不改变已保存账户。点击运行才开始计算。`;saveDraft('accounts');navigate('accounts');
 }
 async function openSaved(id,{navigateTo=true}={}){const request=++resultSequence,routeAtStart=location.hash;$(ownerId('accounts','error')).hidden=true;if(result?.result_id!==id){result=null;$('light-results').hidden=true;$('light-enable-daily').disabled=true;}try{const value=await api('/results/'+encodeURIComponent(id));if(request!==resultSequence)return;result=value;routeResult=id;if(navigateTo&&location.hash===routeAtStart)navigate('accounts',id);$('light-new-account').open=false;renderSimulation();requestAnimationFrame(()=>{if(accountVisible())$('light-results').scrollIntoView({block:'start',behavior:'smooth'});});}catch(e){if(request===resultSequence){routeResult=null;error(e,'accounts');}}}
 async function openTask(id){const job=await api('/jobs/'+encodeURIComponent(id));if(!job.result)throw Error('该任务没有可读取的结果，可能已过期。');navigate(owner(job.kind));await deliver(job);if(job.kind==='simulation'){$('light-new-account').open=false;requestAnimationFrame(()=>{if(accountVisible())$('light-results').scrollIntoView({block:'start'});});}else if(!accountKinds.has(job.kind)&&job.kind!=='prediction')$('light-training-panel').open=true;}
 function onRoute(route){globalThis.KHQuantPredictionChart?.onRoute();if(!accountVisible()){if(chart)chart.remove();if(tradeChart)tradeChart.remove();chart=tradeChart=null;}
  if(route.view==='watchlist'&&route.tab==='accounts'){if(route.result&&routeResult!==route.result){routeResult=route.result;openSaved(route.result,{navigateTo:false});}else if(result)renderAccount();refresh();}else if(route.view==='simulation')refresh();
 }
 const draftFields={prediction:['light-symbol','light-predict-date','light-lookback','light-horizon','light-samples','light-seed','light-device'],accounts:['light-symbols','light-start','light-end','light-cash','light-score-model','light-sim-lookback','light-sim-horizon','light-sim-samples','light-sim-seed','light-sim-device','light-sim-feature-profile','light-sim-score-type','light-candidate-days','light-fusion','light-cost-stress','light-save'],training:['light-training-symbols','light-training-device','light-feature-profile','light-score-type','light-score-variant','light-score-end','light-observe-start'],daily:['light-daily-date']};
 function saveDraft(section){try{sessionStorage.setItem('khquant_light_draft_'+section,JSON.stringify(Object.fromEntries(draftFields[section].map(id=>[id,$(id).type==='checkbox'?$(id).checked:$(id).value]))));}catch(_){}}
 function restoreDrafts(){for(const [section,ids] of Object.entries(draftFields)){try{const value=JSON.parse(sessionStorage.getItem('khquant_light_draft_'+section)||'null');if(value)for(const id of ids)if(Object.hasOwn(value,id)){const node=$(id);if(node.type==='checkbox')node.checked=!!value[id];else if(id==='light-score-model'&&value[id]){node.innerHTML+=`<option value="${esc(value[id])}">已绑定候选（执行时核验） · ${esc(value[id].slice(0,12))}</option>`;node.value=value[id];}else node.value=String(value[id]);}}catch(_){}for(const id of ids)for(const event of ['input','change'])$(id).addEventListener(event,()=>queueMicrotask(()=>saveDraft(section)));}}
 function bindAction(id,fn,section){$(id)?.addEventListener('click',()=>Promise.resolve().then(fn).catch(e=>error(e,section||currentOwner())));}
 function boot(){if(!$('view-simulation'))return;restoreDrafts();
  $('light-predict-form').addEventListener('submit',e=>{e.preventDefault();submit('/predict',{symbol:$('light-symbol').value.toUpperCase(),end:$('light-predict-date').value,lookback:Number($('light-lookback').value),horizon:Number($('light-horizon').value),samples:Number($('light-samples').value),device:$('light-device').value,seed:Number($('light-seed').value)});});
  $('light-sim-form').addEventListener('submit',e=>{e.preventDefault();try{submit('/simulate',simulationSpec());}catch(err){error(err,'accounts');}});
  bindAction('light-train',()=>submit('/score/train',{symbols:symbols('light-training-symbols',20),end:$('light-score-end').value,model_type:$('light-score-type').value,feature_profile:$('light-feature-profile').value,variant:$('light-score-variant').value,...($('light-score-variant').value==='B'&&observationManifest?{observation_manifest:observationManifest}:{}),device:'cpu',budget_seconds:600,select:false}),'prediction');
  $('light-observe-manifest').addEventListener('change',()=>loadManifest($('light-observe-manifest').value).catch(e=>error(e,'prediction')));
  bindAction('light-collect',()=>submit('/score/observations',observationManifest?{manifest_id:observationManifest.manifest_id}:{symbols:symbols('light-training-symbols',20),start:$('light-observe-start').value,end:$('light-score-end').value,device:$('light-training-device').value}),'prediction');
  bindAction('light-tune',()=>{const values=symbols('light-training-symbols',5);return submit('/tune',{symbols:values,device:$('light-training-device').value});},'prediction');
  bindAction('light-training-watchlist',()=>{const values=globalThis.KHQuantWatchlist.selected();if(!values.length||values.length>20)throw Error('请先在关注股票中选择1至20只股票');$('light-training-symbols').value=values.join(', ');saveDraft('training');},'prediction');
  bindAction('light-daily',()=>{if(!$('light-daily-date').value)throw Error('请选择续算目标行情日');return submit('/daily',{required_price_date:$('light-daily-date').value});},'accounts');
  bindAction('light-enable-daily',async()=>{if(!result)throw Error('请先打开账户结果');if(active||starting)throw Error('请等待当前计算结束');const a=await api('/accounts/'+result.result_id,{enabled:true,shadow_research:$('light-shadow').checked});$('light-daily-state').textContent=`已启用 ${a.symbols.join('、')} 的每日续算，最近日期 ${a.last_date}`;await refresh();},'accounts');
  bindAction('light-open-saved',()=>{const id=$('light-saved').value;if(id)return openSaved(id);},'accounts');
  $('light-trade-symbol').addEventListener('change',()=>{if(result)renderTrades(result.comparisons[$('light-account').value]);});
  for(const section of ['prediction','accounts']){bindAction(ownerId(section,'cancel'),async()=>{const id=active?.job_id;if(id&&owner(active.kind)===section)await api('/jobs/'+encodeURIComponent(id)+'/cancel',{});},section);bindAction(ownerId(section,'open-task'),()=>openTask($(ownerId(section,'open-task')).dataset.jobId),section);}
  bindAction('light-export',()=>{if(result)download('KHQuant-'+result.result_id.slice(0,12)+'.json',JSON.stringify(result,null,2),'application/json');},'accounts');
  bindAction('light-export-trades',()=>{if(!result)return;const trades=result.comparisons[$('light-account').value].trades||[],cols=['date','symbol','action','shares','price','fee','position_after','cash_after','reason','share_delta','weight_before','weight_after','trade_weight_delta'],q=x=>'"'+String(x??'').replaceAll('"','""')+'"';download('KHQuant-trades.csv','\ufeff'+[cols,...trades.map(t=>cols.map(c=>t.execution_change?.[c]??t[c]))].map(r=>r.map(q).join(',')).join('\r\n'),'text/csv;charset=utf-8');},'accounts');
  $('light-account').addEventListener('change',renderAccount);$('light-chart-mode').addEventListener('change',renderAccount);bindAction('light-refresh',refresh);
  bindAction('open-light-analysis',()=>transferToPrediction(globalThis.KHQuantMethods.analysisContext()),'prediction');
  bindAction('open-account-analysis',()=>transferToAccounts(globalThis.KHQuantMethods.analysisContext()),'accounts');
  $('light-feature-profile').addEventListener('change',()=>{updateFeatureControls();refresh();});
  for(const key of ['lookback','horizon','samples','seed','device','feature-profile','score-type'])$('light-sim-'+key).addEventListener('change',()=>{if(key==='feature-profile'){$('light-score-model').value='';refresh();}updateSimulationControls();});
  window.addEventListener('khquant:task-history-changed',()=>{const route=globalThis.KHQuantNavigation?.parse();if(route?.view==='simulation'||route?.view==='watchlist'&&route?.tab==='accounts')refresh();});
  updateFeatureControls();updateSimulationControls();
  if(typeof MutationObserver!=='undefined'){const observer=new MutationObserver(()=>{if(accountVisible()&&result)renderAccount();});observer.observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});}
 }
 globalThis.KHQuantLightResearch={refresh,openTask};boot();
 Object.assign(globalThis.KHQuantLightResearch,{openSaved,onRoute,transferToAccounts,transferToPrediction,simulationSpec});
 const initialRoute=globalThis.KHQuantNavigation?.parse();if(initialRoute)onRoute(initialRoute);
})();
