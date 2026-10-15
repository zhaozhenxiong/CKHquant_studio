"""Exercise publication changes between shipped board UI network requests."""
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js/board_panel.js'


def run_js(body):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for board publication behavior checks')
    harness = r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map(),commits=[],calls=[],jobCommits=[];
function el(id){if(!elements.has(id))elements.set(id,{value:'',hidden:false,replaceChildren(){}});return elements.get(id);}
el('board-kind').value='industry';el('board-ranking').value='capital';el('capital-sort').value='amount';
global.document={getElementById:el};global.sessionStorage={getItem:()=>null};
const source=fs.readFileSync(process.argv[1],'utf8').replace("  if(!$('scan-board-panel'))return;",`
  renderStatus=value=>commits.push(['status',(value.fluent || value).market?.generation_id || null]);
  renderRows=value=>commits.push(['ranking',value.generation_id]);
  renderCapital=value=>commits.push(['visual',value.generation_id]);
  renderJob=value=>{el('board-job').hidden=false;jobCommits.push(value);};
  poll=async()=>jobCommits.push({polled:job.job_id});
  globalThis.ui={refresh};return;
`);
vm.runInThisContext(source);
const day=g=>g==='g1'?'2026-10-13':'2026-10-14';
const status=g=>({fluent:{market:g?{generation_id:g,price_date:day(g)}:null}});
const ranking=g=>({generation_id:g,price_date:day(g),rankings:[{generation_id:g,price_date:day(g)}]});
const visual=g=>({generation_id:g,price_date:day(g),rows:[{generation_id:g,price_date:day(g)}]});
const response=value=>({ok:true,status:200,json:async()=>value});
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
const flush=async()=>{for(let i=0;i<12;i++)await Promise.resolve();};
'''
    result = subprocess.run([node, '-e', harness + body, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_new_publication_between_status_and_rows_keeps_one_frozen_generation():
    run_js(r'''
(async()=>{
let latest='g1';
global.fetch=async path=>{calls.push(path);if(path==='/api/boards/status'){const old=latest;latest='g2';return response(status(old));}
if(path==='/api/light/status'||path==='/api/boards/jobs')return response({jobs:[]});
const selected=new URL(path,'http://local').searchParams.get('generation_id') || latest;
return response(path.includes('/rankings?')?ranking(selected):visual(selected));};
await ui.refresh();
assert.deepEqual(commits,[['status','g1'],['ranking','g1'],['visual','g1']]);
assert.equal(calls.filter(p=>p.includes('generation_id=g1')).length,2);
assert.equal(el('board-detail').hidden,true);
await ui.refresh();
assert.deepEqual(commits.slice(-3),[['status','g2'],['ranking','g2'],['visual','g2']]);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_delayed_chart_is_committed_with_table_and_generation_mismatch_is_rejected():
    run_js(r'''
(async()=>{
const pending=deferred();let bad=false;
global.fetch=async path=>{if(path==='/api/boards/status')return response(status('g2'));
if(path==='/api/light/status'||path==='/api/boards/jobs')return response({jobs:[]});
if(path.includes('/rankings?'))return response(ranking('g2'));
if(bad)return response(visual('g1'));return pending.promise;};
const loading=ui.refresh();await flush();
assert.deepEqual(commits,[],'status/table must wait for the chart');
pending.resolve(response(visual('g2')));await loading;
assert.deepEqual(commits,[['status','g2'],['ranking','g2'],['visual','g2']]);
commits.length=0;el('board-detail').hidden=false;bad=true;
await assert.rejects(ui.refresh(),/板块发布版本核验失败/);
assert.deepEqual(commits,[],'invalid reply must preserve the complete prior view');
assert.equal(el('board-detail').hidden,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_empty_publication_never_adopts_unpinned_rows_and_slow_old_refresh_cannot_overwrite():
    run_js(r'''
(async()=>{
global.fetch=async path=>{calls.push(path);return response(path==='/api/boards/status'?status(null):{jobs:[]});};
await ui.refresh();assert.deepEqual(commits,[['status',null],['ranking',null],['visual',null]]);
assert.equal(calls.some(path=>path.includes('/rankings?') || path.includes('/capital-visualization?')),false);
commits.length=0;const old=deferred();let n=0;
global.fetch=async path=>{if(path==='/api/boards/status')return response(status(++n===1?'g1':'g2'));
if(path==='/api/light/status'||path==='/api/boards/jobs')return response({jobs:[]});
const g=new URL(path,'http://local').searchParams.get('generation_id');
if(g==='g1' && path.includes('/capital-visualization?'))return old.promise;
return response(path.includes('/rankings?')?ranking(g):visual(g));};
const first=ui.refresh();await flush();await ui.refresh();old.resolve(response(visual('g1')));await first;
assert.deepEqual(commits,[['status','g2'],['ranking','g2'],['visual','g2']]);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_job_banner_adopts_latest_completed_retry_and_keeps_current_partial_actionable():
    run_js(r'''
(async()=>{
let saved=[{job_id:'failed-parent',status:'partial',kind:'board_knowledge'}];
global.fetch=async path=>{
if(path==='/api/boards/status')return response(status('g2'));
if(path==='/api/light/status'||path==='/api/boards/jobs')return response({jobs:saved});
return response(path.includes('/rankings?')?ranking('g2'):visual('g2'));};
await ui.refresh();assert.equal(jobCommits.at(-1).job_id,'failed-parent');
saved=[{job_id:'retry-success',status:'succeeded',kind:'board_knowledge'},...saved];
await ui.refresh();assert.equal(jobCommits.at(-1).job_id,'retry-success');
assert.equal(jobCommits.at(-1).status,'succeeded');
saved=[];await ui.refresh();assert.equal(el('board-job').hidden,true);
for(const id of ['board-calculate','board-sync','board-refresh-members'])assert.equal(el(id).disabled,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_active_board_job_takes_priority_over_finished_job():
    run_js(r'''
(async()=>{
global.fetch=async path=>{
if(path==='/api/boards/status')return response(status('g2'));
if(path==='/api/light/status'||path==='/api/boards/jobs')return response({jobs:[{job_id:'done',status:'succeeded'},{job_id:'active',status:'running'}]});
return response(path.includes('/rankings?')?ranking('g2'):visual('g2'));};
await ui.refresh();assert.deepEqual(jobCommits,[{polled:'active'}]);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_delayed_old_job_reply_cannot_restore_failed_banner_after_new_refresh():
    run_js(r'''
(async()=>{
const old=deferred();let calls=0;
global.fetch=async path=>{
if(path==='/api/boards/status')return response(status('g2'));
if(path==='/api/light/status'||path==='/api/boards/jobs')return ++calls===1?old.promise:response({jobs:[{job_id:'new',status:'succeeded'}]});
return response(path.includes('/rankings?')?ranking('g2'):visual('g2'));};
const first=ui.refresh();await flush();await ui.refresh();
old.resolve(response({jobs:[{job_id:'old',status:'partial'}]}));await first;
assert.deepEqual(jobCommits,[{job_id:'new',status:'succeeded'}]);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')
