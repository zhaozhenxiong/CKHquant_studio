/* Unified task history. Deleting a record and reclaiming an artifact are separate operations. */
(() => {
  'use strict';
  const $=id=>document.getElementById(id), esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const active=new Set(['queued','running','cancelling']);
  const statusNames={queued:'等待开始',running:'运行中',cancelling:'正在取消',succeeded:'已完成',partial:'部分完成',failed:'失败',cancelled:'已取消',blocked:'条件未满足',interrupted:'服务中断'};
  const kindNames={analysis:'结构分析',scan:'全市场扫描',update:'行情更新',prediction:'Kronos 预测',simulation:'模拟回测',score_training:'评分模型训练',score_observations:'历史预测观察',tuning:'预测参数比较',kronos_tuning:'预测参数比较',daily:'每日续算',incremental:'结构增量补齐',board_features:'板块特征',board_sync:'板块结构同步',board_knowledge:'板块成员',model_training:'历史模型训练',model_validation:'模型复核'};
  const state={page:1,pageSize:20,total:0,items:[],active:[],open:new Set(),selected:new Set(),plan:null,sequence:0,busy:false,timer:null};
  const key=item=>`${item.backend}:${item.job_id}`;
  const label=item=>kindNames[item.kind]||item.kind||'任务';
  const size=bytes=>`${(Number(bytes||0)/1024/1024).toLocaleString('zh-CN',{maximumFractionDigits:2})} MiB`;
  async function api(path,body){const token=sessionStorage.getItem('khquant_api_token');const response=await fetch(path,{method:body===undefined?'GET':'POST',headers:{...(token?{Authorization:`Bearer ${token}`}:{ }),...(body===undefined?{}:{'Content-Type':'application/json'})},credentials:'same-origin',body:body===undefined?undefined:JSON.stringify(body)});const value=await response.json();if(!response.ok)throw Error(typeof value.detail==='string'?value.detail:JSON.stringify(value.detail||value));return value;}
  function notice(text){$('task-history-state').textContent=text;}
  function remember(){try{sessionStorage.setItem('khquant_task_details',JSON.stringify([...state.open]));}catch(_){}}
  function capture(){document.querySelectorAll('details[data-history-key]').forEach(node=>{if(node.open)state.open.add(node.dataset.historyKey);else state.open.delete(node.dataset.historyKey);});remember();}
  function row(item){const id=key(item), running=active.has(item.status), progress=typeof item.progress==='number'?item.progress:typeof item.progress==='object'?Number(item.progress.completed||0)/Math.max(1,Number(item.progress.total||1))*100:0;
    const disabled=!item.can_delete||state.busy;
    const result=item.backend==='legacy'&&item.run_id?`<button type="button" data-run-id="${esc(item.run_id)}" data-run-kind="${esc(item.kind)}">查看结果</button>`:item.backend==='light'&&['succeeded','partial'].includes(item.status)?`<button type="button" data-history-open="${esc(id)}">查看结果</button>`:'';
    const actions=running?`<button type="button" data-history-cancel="${esc(id)}" ${item.status==='cancelling'?'disabled':''}>取消</button>`:`<button type="button" data-history-delete="${esc(id)}" ${disabled?'disabled':''} title="${esc(item.delete_blocked_reason||'仅删除记录，保留结果')}">删除记录</button>`;
    return `<article class="task-row task-history-row"><div class="task-heading"><div>${running?'':`<input type="checkbox" class="task-select" data-history-select="${esc(id)}" aria-label="选择 ${esc(label(item))} ${esc(item.created_at)}" ${state.selected.has(id)?'checked':''} ${disabled?'disabled':''}>`}<strong>${esc(label(item))}</strong><small>${esc(item.created_at||'')} · ${item.backend==='light'?'轻量研究':'分析与数据'}</small></div><div class="task-actions"><span class="badge">${esc(statusNames[item.status]||item.status)}</span>${result}${actions}</div></div><details data-history-key="${esc(id)}" ${state.open.has(id)?'open':''}><summary>任务详情</summary><p class="subtle">${esc(item.run_id||item.job_id)}</p><div class="progress-track"><i style="width:${Math.max(0,Math.min(100,progress||0))}%"></i></div><p class="task-detail ${item.error?'task-error':''}">${esc(typeof item.error==='string'?item.error:item.error?JSON.stringify(item.error):typeof item.progress_detail==='string'?item.progress_detail:JSON.stringify(item.progress_detail||item.progress||{}))}</p>${item.delete_blocked_reason?`<p class="subtle">暂不能删除：${esc(item.delete_blocked_reason)}</p>`:''}</details></article>`;
  }
  function selection(){const enabled=state.items.filter(item=>item.can_delete);$('task-delete-selected').disabled=state.busy||!enabled.some(item=>state.selected.has(key(item)));}
  function render(){capture();$('task-active-list').innerHTML=state.active.length?state.active.map(row).join(''):'<p class="empty-copy">没有进行中的任务。</p>';$('task-list').innerHTML=state.items.length?state.items.map(row).join(''):'<p class="empty-copy">没有符合筛选条件的历史任务。</p>';$('task-page-label').textContent=`${state.total} 条 · 第 ${state.page} / ${Math.max(1,Math.ceil(state.total/state.pageSize))} 页`;$('task-page-prev').disabled=state.page<=1;$('task-page-next').disabled=state.page*state.pageSize>=state.total;selection();}
  async function refresh(){if(!$('task-history-panel')||document.hidden)return;clearTimeout(state.timer);const sequence=++state.sequence;const filters=new URLSearchParams({kind:$('task-kind-filter').value||'all',backend:$('task-backend-filter').value||'all'});
    const history=await api(`/api/task-history/overview?${filters}&status=${encodeURIComponent($('task-status-filter').value||'terminal')}&page=${state.page}&page_size=${state.pageSize}`);
    const current={items:history.active_items,total:history.active_total};
    if(sequence!==state.sequence)return;state.active=current.items||[];state.items=history.items||[];state.total=history.total||0;
    if(state.page>1&&!state.items.length&&state.total<(state.page-1)*state.pageSize+1){state.page=Math.max(1,Math.ceil(state.total/state.pageSize));return refresh();}
    $('task-active-count').textContent=String(current.total||0);$('task-terminal-count').textContent=String(history.counts?.terminal??history.total??0);render();
    if(!document.hidden&&state.active.length&&location.hash==='#data')state.timer=setTimeout(()=>refresh().catch(error=>notice(error.message)),3000);
  }
  function find(id){return [...state.items,...state.active].find(item=>key(item)===id);}
  function selectedTasks(){return state.items.filter(item=>item.can_delete&&state.selected.has(key(item))).map(({backend,job_id})=>({backend,job_id}));}
  async function deleteTasks(tasks){if(state.busy||!tasks.length)return;state.busy=true;selection();try{const result=await api('/api/task-history/delete',{tasks});state.selected.clear();notice(`记录已删除 ${result.deleted?.length||0} 条；跳过 ${result.skipped?.length||0}，失败 ${result.failed?.length||0}。结果产物保留。`);await refresh();window.dispatchEvent(new CustomEvent('khquant:task-history-changed'));}finally{state.busy=false;render();}}
  async function previewCleanup(){if(state.busy)return;state.busy=true;$('task-cleanup-preview').disabled=true;$('task-cleanup-preview').setAttribute('aria-busy','true');notice('正在核验引用与可回收空间…');try{state.plan=await api('/api/task-history/cleanup/preview',{tasks:selectedTasks(),include_artifacts:true});const plan=state.plan;$('task-cleanup-plan').hidden=false;$('task-cleanup-summary').textContent=`候选 ${plan.candidates?.length||0} 项，可释放 ${size(plan.candidate_bytes)}；保护 ${plan.protected?.length||0} 项。${selectedTasks().length?'范围：所选历史任务。':'范围：全部历史任务与已确认无引用的旧产物。'}`;$('task-cleanup-items').innerHTML=`<details><summary>查看候选与保护项</summary><ul>${(plan.candidates||[]).map(item=>`<li>${esc(item.type)} · ${esc(item.path||item.id)} · ${size(item.size_bytes)} · ${esc(item.reason)}</li>`).join('')}${(plan.protected||[]).map(item=>`<li>保留 · ${esc(item.path||item.id||item.job_id)} · ${esc(item.reason)}</li>`).join('')}</ul></details>`;$('task-cleanup-execute').disabled=!(plan.candidates||[]).length;notice('清理预览已完成，请查看候选与保护项。');}finally{state.busy=false;$('task-cleanup-preview').disabled=false;$('task-cleanup-preview').removeAttribute('aria-busy');selection();}}
  async function executeCleanup(){if(state.busy||!state.plan)return;state.busy=true;$('task-cleanup-execute').disabled=true;try{const plan=state.plan,result=await api('/api/task-history/cleanup/execute',{plan_id:plan.plan_id,plan_sha256:plan.plan_sha256});state.plan=null;$('task-cleanup-plan').hidden=true;state.selected.clear();notice(`清理完成：删除 ${result.deleted?.length||0} 项，释放 ${size(result.deleted_bytes)}；跳过 ${result.skipped?.length||0}，失败 ${result.failed?.length||0}。`);await refresh();window.dispatchEvent(new CustomEvent('khquant:task-history-changed'));}finally{state.busy=false;selection();}}
  function run(action){return Promise.resolve().then(action).catch(error=>notice(error.message||String(error)));}
  function boot(){if(!$('task-history-panel'))return;try{state.open=new Set(JSON.parse(sessionStorage.getItem('khquant_task_details')||'[]'));}catch(_){}
    ['task-kind-filter','task-backend-filter','task-status-filter'].forEach(id=>$(id).addEventListener('change',()=>{state.page=1;state.selected.clear();run(refresh);}));
    $('refresh-tasks').addEventListener('click',()=>run(refresh));
    $('task-page-prev').addEventListener('click',()=>{state.page=Math.max(1,state.page-1);run(refresh);});$('task-page-next').addEventListener('click',()=>{state.page++;run(refresh);});
    $('task-select-page').addEventListener('click',()=>{const rows=state.items.filter(item=>item.can_delete),all=rows.every(item=>state.selected.has(key(item)));rows.forEach(item=>all?state.selected.delete(key(item)):state.selected.add(key(item)));render();});
    $('task-delete-selected').addEventListener('click',()=>run(()=>deleteTasks(selectedTasks())));
    $('task-cleanup-preview').addEventListener('click',()=>run(previewCleanup));$('task-cleanup-execute').addEventListener('click',()=>run(executeCleanup));$('task-cleanup-dismiss').addEventListener('click',()=>{$('task-cleanup-plan').hidden=true;state.plan=null;});
    ['task-collapse-all','task-expand-all'].forEach(id=>$(id).addEventListener('click',()=>{const expand=id==='task-expand-all';document.querySelectorAll('details[data-history-key]').forEach(node=>{node.open=expand;});capture();}));
    document.addEventListener('toggle',event=>{if(event.target.matches?.('details[data-history-key]'))capture();},true);
    document.addEventListener('change',event=>{const id=event.target.dataset?.historySelect;if(id){event.target.checked?state.selected.add(id):state.selected.delete(id);selection();}});
    document.addEventListener('click',event=>{const button=event.target.closest('button');if(!button)return;const data=button.dataset;
      if(data.historyDelete){const item=find(data.historyDelete);if(item?.can_delete)run(()=>deleteTasks([{backend:item.backend,job_id:item.job_id}]));}
      if(data.historyCancel){const item=find(data.historyCancel);if(item)run(async()=>{await api(item.backend==='light'?`/api/light/jobs/${encodeURIComponent(item.job_id)}/cancel`:`/api/tasks/${encodeURIComponent(item.job_id)}/cancel`,{});await refresh();});}
      if(data.historyOpen){const item=find(data.historyOpen);if(item)run(()=>globalThis.KHQuantLightResearch?.openTask(item.job_id));}
    });
    window.addEventListener('hashchange',()=>{if(location.hash==='#data')run(refresh);else clearTimeout(state.timer);});
    document.addEventListener('visibilitychange',()=>{clearTimeout(state.timer);if(!document.hidden&&location.hash==='#data')run(refresh);});
    if(location.hash==='#data')run(refresh);
  }
  globalThis.KHQuantTaskHistory={refresh};boot();
})();
