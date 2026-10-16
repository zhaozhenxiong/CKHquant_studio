"""Terminal incremental receipts survive publication changes and read-only refresh."""
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'web_dashboard/static/js/board_panel.js'


def run_js(body):
    runtime=shutil.which('node')
    if not runtime:pytest.skip('Node required for board UI state checks')
    harness=r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
class Element {
 constructor(){this.children=[];this._text='';this.hidden=false;this.disabled=false;this.value='';this.classList={add(){},remove(){}};}
 set textContent(value){this._text=String(value);this.children=[];}
 get textContent(){return this._text+this.children.map(c=>c.textContent||'').join(' ');}
 append(...values){this.children.push(...values);}
 replaceChildren(...values){this._text='';this.children=values;}
 setAttribute(){} addEventListener(){}
}
const elements=new Map(),calls=[],timers=[];
function el(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);}
global.document={getElementById:el,createElement:()=>new Element(),querySelectorAll:()=>[]};
global.sessionStorage={getItem:()=>null,setItem(){}};
global.setTimeout=fn=>{timers.push(fn);return timers.length;};global.clearTimeout=()=>{};
el('board-kind').value='industry';el('board-ranking').value='capital';el('capital-sort').value='amount';
const script=fs.readFileSync(process.argv[1],'utf8').replace("  if(!$('scan-board-panel'))return;",`
  renderStatus=()=>{};renderRows=()=>{};renderCapital=()=>{};
  globalThis.ui={refresh,pollFill,clearFillPlan,renderJob,
    setJob:id=>fillJob=id,getResult:()=>fillResult,getJob:()=>fillJob};return;
`);vm.runInThisContext(script);
const completed={job_id:'new-fill',kind:'incremental',status:'succeeded',created_at:'2026-10-15T14:19:38+08:00',finished_at:'2026-10-15T14:20:08+08:00',result:{status:'completed',price_date:'2026-10-14',scope:'watch',requested:2,counts:{reused:1,appended:0,rebuilt:1,missing_price:0,failed:0,deferred:0},rows:[{symbol:'002156.SZ',status:'rebuilt'},{symbol:'002436.SZ',status:'reused'}],full_published:false}};
const cancelled={job_id:'old-sync',kind:'board_sync',status:'cancelled',updated_at:'2026-10-15T10:23:00+08:00',progress_detail:{stage:'raw_identity',current:2600960},error:'用户取消任务'};
const summary=job=>Object.fromEntries(Object.entries(job).filter(([k])=>k!=='result'));
const response=value=>({ok:true,status:200,json:async()=>value});
const status=g=>({fluent:{market:{generation_id:g,calculation_scope:'sh_sz_a_v1',price_date:'2026-10-14'}}});
let generation='g2',lightJobs=[summary(completed)],oldJobs=[cancelled];
async function defaults(path,options={}){
 calls.push({path,method:options.method||'GET'});
 if(path==='/api/boards/status')return response(status(generation));
 if(path==='/api/light/status')return response({jobs:lightJobs});
 if(path==='/api/boards/jobs')return response({jobs:oldJobs});
 if(path==='/api/light/jobs/new-fill')return response(completed);
 return response({generation_id:new URL(path,'http://local').searchParams.get('generation_id'),price_date:'2026-10-14',rows:[],rankings:[]});
}
global.fetch=defaults;
const flush=async()=>{for(let i=0;i<20;i++)await Promise.resolve();};
'''
    result=subprocess.run([runtime,'-e',harness+body,str(SCRIPT)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_terminal_poll_survives_its_publication_refresh_and_scope_change():
    run_js(r'''
(async()=>{
ui.setJob('new-fill');await ui.pollFill();
assert.match(el('board-fill-plan').textContent,/本批结果.*2026-10-14.*我的关注.*请求 2/);
assert.match(el('board-fill-plan').textContent,/已复用 1/);assert.match(el('board-fill-plan').textContent,/已重建 1/);
assert.match(el('board-fill-plan').textContent,/核验失败 0/);assert.match(el('board-fill-plan').textContent,/下批待处理 0/);
assert.match(el('board-fill-plan').textContent,/全量结构 仍未完整发布/);
generation='g3';await ui.refresh();ui.clearFillPlan();
assert.match(el('board-fill-plan').textContent,/本批结果/);assert.equal(el('board-fill-execute').hidden,true);
assert.equal(ui.getJob(),null);assert.equal(timers.length,0);assert(calls.every(c=>c.method==='GET'));
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_fresh_refresh_restores_latest_incremental_and_separates_old_cancelled_task():
    run_js(r'''
(async()=>{
lightJobs=[{...summary(completed),job_id:'older-fill',finished_at:'2026-10-15T13:00:00+08:00'},summary(completed)];
await ui.refresh();
assert.equal(ui.getResult().job_id,'new-fill');assert.match(el('board-job').textContent,/最近增量补齐 · 完成/);
assert.match(el('board-job').textContent,/复用 1.*追加 0.*重建 1.*失败 0.*延后 0/);
assert.match(el('board-job').textContent,/历史板块任务：已取消（保持终止）/);
assert.match(el('board-fill-plan').textContent,/002156.SZ.*已重建/);
await ui.refresh();assert.equal(calls.filter(c=>c.path==='/api/light/jobs/new-fill').length,1);
assert(calls.every(c=>c.method==='GET'));assert.equal(timers.length,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_cancelled_incremental_receipt_never_restores_execution_or_invents_counts():
    run_js(r'''
(async()=>{
completed.status='cancelled';delete completed.result;completed.error='用户取消增量补齐';lightJobs=[summary(completed)];
await ui.refresh();
assert.match(el('board-fill-plan').textContent,/本批结果 · 已取消/);
assert.match(el('board-fill-plan').textContent,/未返回完整逐股统计/);
assert.match(el('board-job').textContent,/复用 未记录/);
assert.equal(el('board-fill-execute').hidden,true);assert.equal(el('board-fill-cancel').hidden,true);
assert.equal(ui.getJob(),null);assert.equal(timers.length,0);assert(calls.every(c=>c.method==='GET'));
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_delayed_old_terminal_read_cannot_replace_new_receipt():
    run_js(r'''
(async()=>{
let release;const delayed=new Promise(resolve=>release=resolve);let read=0;
global.fetch=async(path,options)=>{
 if(path==='/api/light/status')return response({jobs:[++read===1?{...summary(completed),job_id:'slow-old',finished_at:'2026-10-15T13:00:00+08:00'}:summary(completed)]});
 if(path==='/api/light/jobs/slow-old')return delayed;
 return defaults(path,options);
};
const first=ui.refresh();await flush();await ui.refresh();release(response({...completed,job_id:'slow-old',status:'cancelled',result:null}));await first;
assert.equal(ui.getResult().job_id,'new-fill');assert.match(el('board-fill-plan').textContent,/本批结果.*请求 2/);
assert.match(el('board-job').textContent,/最近增量补齐 · 完成/);assert(calls.every(c=>c.method==='GET'));
})().catch(e=>{console.error(e);process.exitCode=1;});
''')
