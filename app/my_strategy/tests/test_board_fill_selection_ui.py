"""Pending board previews remain bound to the selection the user reviewed."""
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js/board_panel.js'


def run_js(body):
    runtime = shutil.which('node')
    if not runtime:
        pytest.skip('Node required for board preview selection checks')
    harness = r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
class Element {
 constructor(){this.children=[];this._text='';this.hidden=false;this.disabled=false;this.value='';
  this.style={};this.dataset={};this.clientWidth=900;this.classList={add(){},remove(){}};}
 set textContent(value){this._text=String(value);this.children=[];}
 get textContent(){return this._text+this.children.map(c=>c.textContent||'').join(' ');}
 append(...values){this.children.push(...values);}
 replaceChildren(...values){this._text='';this.children=values;}
 setAttribute(){} addEventListener(){}
}
const elements=new Map(),calls=[];
function el(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);}
global.document={getElementById:el,createElement:()=>new Element(),createElementNS:()=>new Element(),querySelectorAll:()=>[]};
global.sessionStorage={getItem:()=>null,setItem(){}};
el('board-kind').value='industry';el('board-ranking').value='capital';el('capital-sort').value='amount';
el('board-fill-scope').value='current_board';el('board-fill-execute').hidden=true;
el('capital-window').value='20';el('capital-trend-metric').value='amount';
const source=fs.readFileSync(process.argv[1],'utf8').replace("  if(!$('scan-board-panel'))return;",`
  renderGenerations=()=>{};renderStatus=()=>{};renderRows=()=>{};pollFill=async()=>{};
  globalThis.ui={renderCapital,renderCapitalTrend,chooseCapital,refresh,changeBoardKind,previewFill,executeFill,clearFillPlan,
    plan:()=>fillPlan,selected:()=>selectedCapital,setGeneration:g=>displayedGeneration=g};return;
`);vm.runInThisContext(source);
const row=(kind='industry',generation='g1',id)=>({
 board_id:id||(kind==='industry'?'THS_I_881121.TI':'THS_C_885338.TI'),
 name:kind==='industry'?'半导体':'融资融券',kind,generation_id:generation,
 price_date:'2026-10-15',membership_date:'2026-10-15',board_amount:10000,
 capital_visualization:{history:[],amount_change_1:100,amount_change_ratio_1:.01}
});
const visual=(kind='industry',generation='g1',rows)=>({generation_id:generation,price_date:'2026-10-15',kind,
 board_count:1,available_board_count:1,market_amount:100000,rows:rows||[row(kind,generation)],increases:[],decreases:[]});
const response=value=>({ok:true,status:200,json:async()=>value});
const deferred=()=>{let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no;});return {promise,resolve,reject};};
const flush=async()=>{for(let i=0;i<20;i++)await Promise.resolve();};
const plan=(spec,id='reviewed')=>({plan_id:id,...spec,requested:spec.scope==='current_board'&&spec.board_id==='THS_I_881121.TI'?184:22,
 max_symbols:20,budget_seconds:300,counts:{rebuild:20},rows:[]});
let liveGeneration='g1';
async function defaults(path,options={}){
 calls.push({path,method:options.method||'GET',body:options.body?JSON.parse(options.body):null});
 const url=new URL(path,'http://local');
 if(url.pathname==='/api/boards/incremental/plan')return response(plan(JSON.parse(options.body)));
 if(url.pathname==='/api/boards/incremental/execute')return response({job_id:'executed'});
 if(url.pathname==='/api/boards/status')return response({fluent:{market:{generation_id:liveGeneration,
  calculation_scope:'sh_sz_a_v1',price_date:'2026-10-15'}}});
 if(url.pathname==='/api/boards/rankings')return response({generation_id:liveGeneration,price_date:'2026-10-15',rankings:[]});
 if(url.pathname==='/api/boards/capital-visualization')return response(visual(url.searchParams.get('kind'),liveGeneration));
 if(path==='/api/light/status'||path==='/api/boards/jobs')return response({jobs:[]});
 throw Error('Unexpected request '+path);
}
global.fetch=defaults;
ui.setGeneration('g1');ui.renderCapital(visual());
const executions=()=>calls.filter(c=>c.path==='/api/boards/incremental/execute');
'''
    result = subprocess.run([runtime, '-e', harness + body, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_kind_change_clears_reviewed_old_board_before_slow_chart_and_new_plan_targets_new_board():
    run_js(r'''
(async()=>{
await ui.previewFill();assert.match(el('board-fill-plan').textContent,/半导体|THS_I_881121.TI/);
assert.equal(ui.plan().requested,184);assert.equal(el('board-fill-execute').hidden,false);
const delayed=deferred();global.fetch=(path,options)=>new URL(path,'http://local').pathname==='/api/boards/capital-visualization'?delayed.promise:defaults(path,options);
el('board-kind').value='concept';const changing=ui.changeBoardKind();
assert.equal(ui.plan(),null);assert.equal(el('board-fill-execute').hidden,true);assert.equal(el('board-fill-plan').textContent,'');
await assert.rejects(ui.previewFill(),/板块类型正在切换/);
assert.equal(calls.filter(c=>c.path==='/api/boards/incremental/plan').length,1);
await flush();delayed.resolve(response(visual('concept')));await changing;
assert.equal(ui.selected().name,'融资融券');assert.equal(ui.plan(),null);assert.equal(el('board-fill-execute').hidden,true);
await ui.previewFill();assert.equal(ui.plan().board_id,'THS_C_885338.TI');assert.equal(ui.plan().requested,22);
assert(!el('board-fill-plan').textContent.includes('THS_I_881121.TI'));assert.equal(el('board-fill-execute').hidden,false);
assert.equal(executions().length,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_rendering_different_or_empty_board_clears_plan_but_same_identity_refresh_keeps_it():
    run_js(r'''
(async()=>{
await ui.previewFill();const reviewed=ui.plan();
ui.renderCapital(visual());await ui.refresh();
assert.equal(ui.plan(),reviewed);assert.equal(el('board-fill-execute').hidden,false);
ui.renderCapital(visual('industry','g1',[row('industry','g1','THS_I_OTHER.TI')]));
assert.equal(ui.plan(),null);assert.equal(el('board-fill-execute').hidden,true);
await ui.previewFill();ui.renderCapital(visual('industry','g1',[]));
assert.equal(ui.selected(),null);assert.equal(ui.plan(),null);assert.equal(el('board-fill-execute').hidden,true);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


@pytest.mark.parametrize('change', ['scope', 'board', 'generation', 'kind', 'history'])
def test_inflight_preview_is_discarded_after_its_selection_changes(change):
    run_js(r'''
(async()=>{
const delayed=deferred();global.fetch=(path,options)=>path==='/api/boards/incremental/plan'?delayed.promise:defaults(path,options);
const preview=ui.previewFill();const oldSpec={scope:'current_board',board_id:'THS_I_881121.TI',price_date:'2026-10-15'};
''' + {
        'scope': "el('board-fill-scope').value='watch';ui.clearFillPlan();",
        'board': "ui.chooseCapital(row('industry','g1','THS_I_OTHER.TI'));",
        'generation': "ui.renderCapital(visual('industry','g2'));",
        'kind': "el('board-kind').value='concept';",
        'history': "el('board-generation').value='historical';ui.clearFillPlan();",
    }[change] + r'''
delayed.resolve(response(plan(oldSpec,'stale')));await preview;
assert.equal(ui.plan(),null);assert.equal(el('board-fill-execute').hidden,true);assert.equal(executions().length,0);
await ui.executeFill();assert.equal(executions().length,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_newer_preview_wins_even_when_old_response_finishes_last():
    run_js(r'''
(async()=>{
const old=deferred();let reads=0;
global.fetch=(path,options)=>path==='/api/boards/incremental/plan'&&++reads===1?old.promise:defaults(path,options);
const first=ui.previewFill();await ui.previewFill();const reviewed=ui.plan();
assert.equal(reviewed.plan_id,'reviewed');old.resolve(response(plan({scope:'current_board',board_id:'THS_I_881121.TI',price_date:'2026-10-15'},'stale')));
await first;assert.equal(ui.plan(),reviewed);assert.equal(el('board-fill-execute').hidden,false);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


@pytest.mark.parametrize('change', ['scope', 'kind', 'generation'])
def test_execution_rechecks_current_selection_even_without_a_change_event(change):
    run_js(r'''
(async()=>{
await ui.previewFill();
''' + {
        'scope': "el('board-fill-scope').value='watch';",
        'kind': "el('board-kind').value='concept';",
        # Simulate the selected publication changing before its next render.
        'generation': "ui.setGeneration('g2');el('board-generation').value='g2';",
    }[change] + r'''
await assert.rejects(ui.executeFill(),/请重新预览缺口|请先切回/);
assert.equal(executions().length,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_preview_reply_for_wrong_board_is_rejected_and_only_reviewed_current_plan_executes_once():
    run_js(r'''
(async()=>{
global.fetch=(path,options)=>path==='/api/boards/incremental/plan'?response(plan({scope:'current_board',board_id:'THS_C_OTHER.TI',price_date:'2026-10-15'})):defaults(path,options);
await assert.rejects(ui.previewFill(),/计划与当前选择不一致/);assert.equal(ui.plan(),null);assert.equal(el('board-fill-execute').hidden,true);
global.fetch=defaults;await ui.previewFill();const delayed=deferred();
global.fetch=(path,options)=>path==='/api/boards/incremental/execute'?(calls.push({path,body:JSON.parse(options.body)}),delayed.promise):defaults(path,options);
const first=ui.executeFill();await ui.executeFill();assert.equal(executions().length,1);
assert.deepEqual(executions()[0].body,{plan_id:'reviewed'});delayed.resolve(response({job_id:'executed'}));await first;
assert.equal(el('board-fill-execute').hidden,true);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_long_capital_window_is_lazy_pinned_cached_and_stale_board_response_cannot_replace_selection():
    run_js(r'''
(async()=>{
const days=Array.from({length:90},(_,i)=>new Date(Date.UTC(2026,6,18+i)).toISOString().slice(0,10));
const history=days.map(date=>({date,amount:10000,market_amount:100000,market_share:.1,effective_members:5,expected_members:5,coverage:1,return_1:0}));
const current={...row(),snapshot_hash:'source-a',capital_visualization:{history:history.slice(-21)}};
ui.renderCapital(visual('industry','g1',[current]));
assert.equal(calls.filter(c=>c.path.includes('/capital-history/')).length,0,'20日首屏复用已发布历史');
assert.equal(el('capital-trend').dataset.points,'20');
let reads=0;const pending=deferred();global.fetch=async(path,options)=>{
 if(path.includes('/capital-history/')){reads++;assert(path.includes('window=90'));assert(path.includes('generation_id=g1'));return pending.promise;}
 return defaults(path,options);
};
el('capital-window').value='90';ui.renderCapitalTrend();
el('capital-window').value='50';ui.renderCapitalTrend();assert.equal(reads,1,'同一冻结板块共用正在读取的90日请求');
pending.resolve(response({generation_id:'g1',board_id:current.board_id,price_date:current.price_date,snapshot_hash:'source-a',calculation_scope:'sh_sz_a_v1',history}));await flush();
assert.equal(el('capital-trend').dataset.points,'50');
el('capital-window').value='90';ui.renderCapitalTrend();assert.equal(reads,1);assert.equal(el('capital-trend').dataset.points,'90');
el('capital-window').value='20';ui.renderCapitalTrend();assert.equal(el('capital-trend').dataset.points,'20');
const old=deferred();global.fetch=()=>old.promise;
el('capital-window').value='90';ui.chooseCapital({...current,board_id:'slow-board',snapshot_hash:'slow'});
el('capital-window').value='20';ui.chooseCapital(current);
old.resolve(response({generation_id:'g1',board_id:'slow-board',price_date:current.price_date,snapshot_hash:'slow',calculation_scope:'sh_sz_a_v1',history}));await flush();
assert.equal(ui.selected().board_id,current.board_id);assert.equal(el('capital-trend').dataset.points,'20');
})().catch(e=>{console.error(e);process.exitCode=1;});
''')
