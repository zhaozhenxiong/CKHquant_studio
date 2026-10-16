"""Current scope and explicit historical views use the shipped browser script."""
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js/board_panel.js'


def run_js(body):
    runtime = shutil.which('node')
    if not runtime:
        pytest.skip('Node required for board scope browser logic')
    harness = r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
class Element {
 constructor(tag='div'){this.tagName=tag;this.children=[];this._text='';this.hidden=false;this.value='';this.listeners={};this.classList={add(){},remove(){}};}
 set textContent(value){this._text=String(value);this.children=[];}
 get textContent(){return this._text+this.children.map(c=>c.textContent||'').join(' ');}
 append(...values){this.children.push(...values);}
 replaceChildren(...values){this._text='';this.children=values;}
 addEventListener(type,fn){this.listeners[type]=fn;} setAttribute(){}
}
const elements=new Map(),calls=[],commits=[],events=[];
function el(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);}
global.document={getElementById:el,createElement:tag=>new Element(tag)};
global.sessionStorage={getItem:()=>null};global.window={dispatchEvent:e=>events.push(e)};
global.CustomEvent=class{constructor(type,spec){this.type=type;this.detail=spec.detail;}};
el('board-kind').value='industry';el('board-ranking').value='capital';el('capital-sort').value='amount';
const source=fs.readFileSync(process.argv[1],'utf8').replace("  if(!$('scan-board-panel'))return;",`
 renderRows=value=>commits.push(['rows',value.generation_id]);
 renderCapital=value=>commits.push(['visual',value.generation_id]);
 globalThis.ui={refresh,detail,start,previewFill,renderGenerations,renderStatus};return;
`);vm.runInThisContext(source);
const response=value=>({ok:true,status:200,json:async()=>value});
const current={generation_id:'new-scope',calculation_scope:'sh_sz_a_v1',price_date:'2026-10-15',board_count:1};
const legacy={generation_id:'old-mixed',price_date:'2026-10-14',board_count:1};
const catalog=[{...current,scope_label:'沪深 A 股'},{...legacy,scope_label:'历史原范围'}];
const state=(market,historical=false)=>({generations:catalog,fluent:{market,historical_view:historical,scope_label:market?.calculation_scope?'沪深 A 股':'历史原范围'}});
const descendants=element=>[element,...element.children.flatMap(descendants)];
'''
    result = subprocess.run([runtime, '-e', harness + body, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_current_ui_refuses_old_scope_even_if_stale_server_returns_it():
    run_js(r'''
(async()=>{
global.fetch=async(path,options={})=>{calls.push([path,options.method||'GET']);return response(new URL(path,'http://local').pathname==='/api/boards/status'?state(legacy):{jobs:[]});};
await ui.refresh();
assert.deepEqual(commits,[['rows',null],['visual',null]]);
assert.equal(calls.some(([path])=>path.includes('/rankings?')||path.includes('/capital-visualization?')),false);
assert.match(el('board-status').textContent,/沪深 A 股板块尚未生成，待补齐/);
assert.match(el('board-generation').textContent,/历史原范围/);
assert.equal(el('board-generation').value,'');assert(calls.every(([,method])=>method==='GET'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_explicit_history_pins_generation_and_cannot_submit_new_work():
    run_js(r'''
(async()=>{
el('board-generation').value='old-mixed';
global.fetch=async(path,options={})=>{
 calls.push([path,options.method||'GET']);
 if(path.startsWith('/api/boards/status'))return response(state(legacy,true));
 if(path.includes('generation_id=old-mixed'))return response({generation_id:'old-mixed',price_date:legacy.price_date,rows:[],rankings:[]});
 return response({jobs:[]});
};
await ui.refresh();
assert.equal(calls[0][0],'/api/boards/status?compact=true&generation_id=old-mixed');
assert.deepEqual(commits,[['rows','old-mixed'],['visual','old-mixed']]);
assert.match(el('board-status').textContent,/历史原范围.*显式历史版本/);
await assert.rejects(ui.start('/features/build',{}),/切回/);
await assert.rejects(ui.previewFill(),/切回/);
assert(calls.every(([,method])=>method==='GET'));
el('board-generation').value='';
global.fetch=async path=>response(new URL(path,'http://local').pathname==='/api/boards/status'?state(null):{jobs:[]});
await ui.refresh();assert.deepEqual(commits.slice(-2),[['rows',null],['visual',null]]);
assert.equal(el('board-detail').hidden,true);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_history_members_are_preserved_but_beijing_has_no_new_analysis_action():
    run_js(r'''
(async()=>{
global.fetch=async()=>response({features:{...legacy,board_id:'old',name:'历史板块'},scope_label:'历史原范围',membership_matches_generation:true,members:['000001.SZ','920001.BJ','900001.SH']});
await ui.detail('old','old-mixed');
const host=el('board-detail-content'),buttons=descendants(host).filter(x=>x.tagName==='button');
assert.match(host.textContent,/920001.BJ（历史成员，不提供新分析）/);
assert.match(host.textContent,/历史原范围/);assert.equal(buttons.length,2);
assert(buttons.some(button=>button.textContent==='900001.SH'),'explicit Shanghai B shares remain valid new-analysis symbols');
buttons[0].listeners.click();assert.equal(events[0].detail.symbol,'000001.SZ');
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_new_detail_explains_calculation_denominator_and_source_evidence():
    run_js(r'''
(async()=>{
global.fetch=async()=>response({features:{...current,board_id:'new',name:'沪深板块'},scope_label:'沪深 A 股',membership_matches_generation:true,members:['000001.SZ'],calculation_membership:{eligible_count:1,source_member_count:2,excluded_members:['920001.BJ'],source_snapshot_hash:'original-source',projection_hash:'saved-projection'}});
await ui.detail('new','new-scope');
assert.match(el('board-detail-content').textContent,/计算成员 1.*快照成员 2.*本次范围排除 1/);
assert.match(el('board-detail-content').textContent,/覆盖率按计算成员计/);
assert.match(el('board-detail-content').textContent,/原.*SHA original-source|来源 SHA original-source/);
assert.match(el('board-detail-content').textContent,/投影 SHA saved-projection/);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')
