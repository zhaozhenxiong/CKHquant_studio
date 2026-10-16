"""Saved board evidence is ranked without dropping unavailable rows from inspection."""
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js/board_method_view.js'


def run_js(body):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for board ranking UI checks')
    harness = r'''
const fs=require('fs'),vm=require('vm'),assert=require('node:assert/strict');
const elements=new Map();
function el(id){
 if(!elements.has(id))elements.set(id,{value:'',innerHTML:'',textContent:'',hidden:false,listeners:{},
  querySelectorAll:()=>[],addEventListener(type,fn){this.listeners[type]=fn;}});
 return elements.get(id);
}
global.document={getElementById:el};
global.KHQuantMethodPresentation={summaryHtml:()=>'<p>saved population</p>'};
global.KHQuantMethods={definitions:{structure_kronos:{label:'结构与量价 + Kronos',prediction:true}}};
global.fetch=()=>{throw Error('Display controls must not fetch or start computation');};
const source=fs.readFileSync(process.argv[1],'utf8').replace('globalThis.KHQuantBoardMethods=',
 'globalThis.boardUI={state,renderBoards,boot};globalThis.KHQuantBoardMethods=');
vm.runInThisContext(source);
el('board-method-kind').value='all';el('board-method-limit').value='30';
const board=(id,buy,complete,coverage=1,eligible=true,kind='industry')=>({
 board_id:id,name:id,kind,ranking_eligible:eligible,
 counts:{buy,complete,expected:Math.round(complete/coverage),sell:complete-buy,wait:0,
  unavailable:0,failed:0,deferred:0,not_in_scan:0},coverage:{method_complete_ratio:coverage}});
const shown=()=>[...el('board-method-rows').innerHTML.matchAll(/data-method-board="([^"]+)"/g)].map(m=>m[1]);
'''
    result = subprocess.run([node, '-e', harness + body, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_top30_uses_valid_buy_share_and_stable_ties_without_mutating_saved_evidence():
    run_js(r'''
const rows=[board('D',10,20),board('A',10,20,.9),board('B',5,10),board('C',10,20),
 board('leader',4,5,.8),board('low-coverage',5,5,.5,false),
 ...Array.from({length:35},(_,i)=>board('filler-'+String(i).padStart(2,'0'),0,10))];
const frozen=JSON.stringify(rows);boardUI.state.result={boards:rows};boardUI.renderBoards();
assert.equal(shown().length,30);assert.deepEqual(shown().slice(0,5),['leader','C','D','B','A']);
assert(!shown().includes('low-coverage'));assert.equal(JSON.stringify(rows),frozen);
assert(el('board-method-rows').innerHTML.includes('有效 BUY 占比 80%'));
assert(el('board-method-ranking-state').textContent.includes('显示 30 个'));
''')


def test_type_and_display_controls_are_independent_and_all_keeps_low_coverage():
    run_js(r'''
boardUI.boot();
assert(el('board-method-panel').innerHTML.includes('<option value="30" selected>Top30</option>'));
assert(el('board-method-panel').innerHTML.includes('BUY ÷ 完整有效成员'));
boardUI.state.result={boards:[board('industry',8,10),board('concept',9,10,1,true,'concept'),
 board('incomplete',5,5,.5,false),board('unknown-qualification',5,5,1,'true')]};
el('board-method-kind').value='industry';el('board-method-kind').listeners.change();
assert.deepEqual(shown(),['industry']);assert.equal(el('board-method-limit').value,'30');
assert(el('board-method-ranking-state').textContent.includes('显示 1 个'));
el('board-method-limit').value='all';el('board-method-limit').listeners.change();
assert.deepEqual(shown(),['industry','unknown-qualification','incomplete']);
assert(el('board-method-rows').innerHTML.includes('覆盖不足，不参与合格排名'));
el('board-method-kind').value='concept';el('board-method-kind').listeners.change();
assert.deepEqual(shown(),['concept']);assert.equal(el('board-method-limit').value,'all');
''')


def test_top30_empty_state_does_not_fill_with_unqualified_boards():
    run_js(r'''
boardUI.state.result={boards:[board('low-coverage',4,4,.4,false)]};boardUI.renderBoards();
assert.deepEqual(shown(),[]);assert(el('board-method-rows').innerHTML.includes('暂无覆盖资格通过'));
assert(el('board-method-ranking-state').textContent.includes('显示 0 个'));
el('board-method-limit').value='all';boardUI.renderBoards();assert.deepEqual(shown(),['low-coverage']);
''')
