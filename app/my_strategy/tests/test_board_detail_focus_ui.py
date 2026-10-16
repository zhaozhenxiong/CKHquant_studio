"""Detail navigation uses the shipped panel and preserves request/focus ownership."""
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js/board_panel.js'


def run_js(body):
    runtime = shutil.which('node')
    if not runtime:
        pytest.skip('Node required for board detail navigation checks')
    harness = r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map(),focuses=[],scrolls=[];
class Element {
 constructor(tag='div'){this.tagName=tag;this.children=[];this._text='';this.hidden=false;this.value='';this.listeners={};this.style={};this.dataset={};this.isConnected=true;this.classList={add(){},remove(){}};}
 set id(value){this._id=value;elements.set(value,this);} get id(){return this._id;}
 set textContent(value){this._text=String(value);this.children=[];}
 get textContent(){return this._text+this.children.map(c=>c.textContent||'').join(' ');}
 append(...values){this.children.push(...values);}
 replaceChildren(...values){this._text='';this.children=values;}
 addEventListener(type,fn){this.listeners[type]=fn;} setAttribute(){}
 querySelector(){return el('board-layout');} before(){}
 focus(options){document.activeElement=this;focuses.push({element:this,options,content:el('board-detail-content').textContent});}
 scrollIntoView(options){scrolls.push({element:this,options,focused:document.activeElement,content:el('board-detail-content').textContent});}
}
function el(id){if(!elements.has(id)){const element=new Element();element.id=id;}return elements.get(id);}
for(const id of ['scan-board-panel','board-kind','board-ranking','board-generation','board-count','board-rows','board-detail','board-detail-title','board-detail-content','board-status','board-fill-execute','board-fill-plan','board-job','board-calculate','board-sync','board-refresh-members'])el(id);
global.document={getElementById:id=>elements.get(id)||null,createElement:tag=>new Element(tag),querySelectorAll:()=>[],activeElement:el('initial-focus')};
global.sessionStorage={getItem:()=>null};global.window={dispatchEvent(){}};
global.matchMedia=query=>({matches:false});
el('board-kind').value='industry';el('board-ranking').value='capital';
const source=fs.readFileSync(process.argv[1],'utf8').replace("  if(!$('scan-board-panel'))return;",`
 globalThis.ui={detail,closeDetail,renderRows,setupCapital,refresh,setCapital:row=>{selectedCapital=row;}};return;
`);vm.runInThisContext(source);
const response=value=>({ok:true,status:200,json:async()=>value});
const value=id=>({features:{board_id:id,name:'板块 '+id,generation_id:'g',price_date:'2026-10-16'},scope_label:'沪深 A 股',membership_matches_generation:true,members:['000001.SZ'],calculation_membership:{eligible_count:1,source_member_count:1,excluded_members:[],source_snapshot_hash:'source-'+id,projection_hash:'projection-'+id}});
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
const flush=async()=>{for(let i=0;i<20;i++)await Promise.resolve();};
const descendants=element=>[element,...element.children.flatMap(descendants)];
'''
    result = subprocess.run([runtime, '-e', harness + body, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('reduced', [False, True])
def test_detail_focus_and_scroll_follow_complete_content_and_motion_preference(reduced):
    run_js(f'global.matchMedia=query=>({{matches:{str(reduced).lower()}}});\n' + r'''
(async()=>{
const trigger=el('trigger');document.activeElement=trigger;
global.fetch=async()=>response(value('complete'));
await ui.detail('complete','g',trigger);
assert.equal(el('board-detail').hidden,false);assert.equal(el('board-detail-title').tabIndex,-1);
assert.equal(focuses.length,1);assert.equal(focuses[0].element,el('board-detail-title'));
assert.deepEqual(focuses[0].options,{preventScroll:true});
assert.match(focuses[0].content,/projection-complete/,'focus waits for final evidence content');
assert.equal(scrolls.length,1);assert.equal(scrolls[0].element,el('board-detail'));
assert.equal(scrolls[0].focused,el('board-detail-title'));assert.equal(scrolls[0].options.block,'start');
assert.equal(scrolls[0].options.behavior,global.matchMedia().matches?'auto':'smooth');
ui.closeDetail();assert.equal(el('board-detail').hidden,true);assert.equal(document.activeElement,trigger);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_latest_detail_owns_scroll_focus_and_close_trigger():
    run_js(r'''
(async()=>{
const old=deferred(),first=el('first-trigger'),second=el('second-trigger');
global.fetch=async path=>path.includes('/old?')?old.promise:response(value('new'));
document.activeElement=first;const pending=ui.detail('old','g',first);
document.activeElement=second;await ui.detail('new','g',second);
old.resolve(response(value('old')));await pending;
assert.equal(scrolls.length,1);assert.equal(focuses.length,1);
assert.equal(el('board-detail-title').textContent,'板块 new');assert.match(scrolls[0].content,/projection-new/);
ui.closeDetail();assert.equal(document.activeElement,second);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_close_while_loading_and_failed_response_never_scroll_or_focus():
    run_js(r'''
(async()=>{
const pending=deferred();global.fetch=async()=>pending.promise;
const loading=ui.detail('closed','g',el('trigger'));ui.closeDetail();
pending.resolve(response(value('closed')));await loading;
assert.equal(el('board-detail').hidden,true);assert.equal(focuses.length,0);assert.equal(scrolls.length,0);
global.fetch=async()=>({ok:false,status:409,json:async()=>({detail:'核验失败'})});
await assert.rejects(ui.detail('failed','g',el('trigger')),/核验失败/);
assert.equal(focuses.length,0);assert.equal(scrolls.length,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_user_focus_change_while_loading_is_preserved():
    run_js(r'''
(async()=>{
const pending=deferred(),trigger=el('trigger'),other=el('other-control');
document.activeElement=trigger;global.fetch=async()=>pending.promise;
const loading=ui.detail('new','g',trigger);document.activeElement=other;
pending.resolve(response(value('new')));await loading;
assert.equal(el('board-detail').hidden,false);assert.equal(document.activeElement,other);
assert.equal(focuses.length,0);assert.equal(scrolls.length,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


@pytest.mark.parametrize('unavailable', ['detached', 'disabled'])
def test_close_does_not_focus_a_detached_or_disabled_trigger(unavailable):
    run_js('const unavailable=' + repr(unavailable) + ';\n' + r'''
(async()=>{
const trigger=el('trigger');document.activeElement=trigger;global.fetch=async()=>response(value('new'));
await ui.detail('new','g',trigger);
if(unavailable==='detached')trigger.isConnected=false;else trigger.disabled=true;
ui.closeDetail();assert.equal(focuses.length,1);assert.equal(document.activeElement,el('board-detail-title'));
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_both_actual_view_buttons_bind_their_own_close_return_target():
    run_js(r'''
(async()=>{
global.fetch=async()=>response(value('new'));
ui.renderRows({rankings:[{board_id:'new',generation_id:'g'}]});
const rankingButton=descendants(el('board-rows')).find(item=>item.tagName==='button');
document.activeElement=rankingButton;await rankingButton.listeners.click();ui.closeDetail();
assert.equal(document.activeElement,rankingButton);
ui.setupCapital();ui.setCapital({board_id:'new',generation_id:'g'});
const capitalButton=el('capital-detail-button');document.activeElement=capitalButton;
capitalButton.listeners.click();await flush();ui.closeDetail();
assert.equal(document.activeElement,capitalButton);assert.equal(scrolls.length,2);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')


def test_background_refresh_does_not_scroll_or_focus_detail():
    run_js(r'''
(async()=>{
ui.setupCapital();const other=el('other-control');document.activeElement=other;
global.fetch=async path=>response(path.startsWith('/api/boards/status')?{fluent:{market:null,full:null},generations:[]}:{jobs:[]});
await ui.refresh();assert.equal(document.activeElement,other);
assert.equal(focuses.length,0);assert.equal(scrolls.length,0);
})().catch(error=>{console.error(error);process.exitCode=1;});
''')
