"""Verify saved market evidence and backtest drafting through shipped UI code."""
import shutil
import subprocess

import pytest

from test_czsc_display_dates_ui import HARNESS, STATIC


SETUP = HARNESS + r'''
const memory=new Map();global.sessionStorage={getItem:key=>memory.get(key)||null,setItem:(key,value)=>memory.set(key,String(value)),removeItem:key=>memory.delete(key)};
for(const node of elements.values())node.classList={add(){},remove(){},toggle(){}};
const oldElement=element;
document.getElementById=id=>{const node=oldElement(id);node.classList||={add(){},remove(){},toggle(){}};return node;};
document.createElement=tag=>newElement(tag);
function select(id,initial=''){
 const node=element(id);let html='',value=initial;
 Object.defineProperty(node,'innerHTML',{configurable:true,get(){return html;},set(source){html=String(source);node.options=[...html.matchAll(/<option\b[^>]*value="([^"]*)"/g)].map(match=>({value:match[1]}));if(!node.options.some(option=>option.value===value))value=node.options[0]?.value||'';}});
 Object.defineProperty(node,'value',{configurable:true,get(){return value;},set(next){value=String(next);}});
 return node;
}
select('market-saved-select');element('scan-backtest-panel').hidden=true;
element('scan-end').value='2026-10-15';element('scan-filter').value='all';
const calls=[],opened=[],transfers=[],tabs=[];
let answer=async path=>path.startsWith('/api/runs?')?{runs:[]}:path==='/api/light/saved'?{results:[]}:Promise.reject(Error('Unexpected request '+path));
globalThis.KHQuantMarketBridge={request:async path=>{calls.push({path,method:'GET'});return answer(path);},openSaved:async id=>{opened.push(id);}};
globalThis.KHQuantLightResearch={transferToAccounts:context=>transfers.push(context)};
globalThis.KHQuantBoardPanel={showBacktest:()=>tabs.push('backtest')};
vm.runInThisContext(fs.readFileSync(path.join(process.argv[1],'market_workspace.js'),'utf8'));
const prevent={preventDefault(){}};
const frozen={analysis_method:'structure_kronos_ma_l2',score_model_id:'frozen-ma',start:'2026-01-01',end:'2026-10-15',lookback:256,horizon:10,samples:32,seed:7,device:'mps'};
'''


def run_js(body):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for market workspace checks')
    result = subprocess.run(
        [node, '-e', SETUP + '\n(async()=>{await settle();\n' + body +
         '\n})().catch(error=>{console.error(error);process.exitCode=1;});', str(STATIC)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_saved_scan_choices_preserve_selection_and_ignore_stale_catalog_response():
    run_js(r'''
const row=id=>({run_id:id,kind:'scan',created_at:'2026-10-16T09:00:00+08:00',summary:{analysis_method:'structure_kronos_ma_l2',data_range:{end:'2026-10-15'}}});
answer=async()=>({runs:[row('saved-one'),row('saved-two')]});
await KHQuantMarketWorkspace.refreshSaved();element('market-saved-select').value='saved-two';
await KHQuantMarketWorkspace.refreshSaved();assert.equal(element('market-saved-select').value,'saved-two');
let finishOld;answer=async()=>await new Promise(resolve=>finishOld=resolve);
const old=KHQuantMarketWorkspace.refreshSaved();await settle();answer=async()=>({runs:[row('newest')]});
await KHQuantMarketWorkspace.refreshSaved();finishOld({runs:[row('stale')]});await old;
assert.equal(element('market-saved-select').value,'newest');assert(!element('market-saved-select').innerHTML.includes('stale'));
assert(calls.every(call=>call.method==='GET'));
''')


def test_saved_result_identity_and_opening_do_not_follow_changed_method_draft():
    run_js(r'''
element('scan-mode').value='structure';element('scan-end').value='2026-10-16';
KHQuantMarketWorkspace.update({run_id:'old-frozen',data_end:'2026-10-15',analysis_method:'structure_kronos_ma_l2',method_context:frozen});
assert(element('market-result-date').textContent.includes('2026-10-15'));assert(element('market-result-date').textContent.includes('L2'));
assert.equal(element('scan-mode').value,'structure');assert.equal(element('scan-end').value,'2026-10-16');
element('market-saved-select').value='old-frozen';await fire('market-open-saved','click');
assert.deepEqual(opened,['old-frozen']);assert.equal(transfers.length,0);assert(calls.every(call=>call.method==='GET'));
''')


def test_backtest_draft_keeps_frozen_forecast_model_and_only_explicitly_transfers():
    run_js(r'''
KHQuantMarketWorkspace.draftBacktest({symbol:'002436',data_end:'2026-10-15'},frozen);
assert.equal(transfers.length,0);assert.deepEqual(tabs,['backtest']);
element('scan-mode').value='structure';element('scan-samples').value='8';
await fire('market-backtest-form','submit',prevent);
assert.equal(transfers.length,1);const actual=transfers[0];assert.deepEqual(actual.symbols,['002436.SZ']);
for(const [key,value] of Object.entries(frozen))assert.equal(actual[key],value,key);
assert.equal(frozen.analysis_method,'structure_kronos_ma_l2');assert(calls.every(call=>call.method==='GET'));
''')


def test_backtest_rejects_unsupported_or_misidentified_pool_without_truncating():
    run_js(r'''
for(const symbol of ['', '920001.BJ', '600519.SZ', '000001.SH', '900001.SH'])assert.throws(()=>KHQuantMarketWorkspace.draftBacktest({symbol},frozen));
assert.throws(()=>KHQuantMarketWorkspace.draftBacktest({symbol:'600519.SH'},{...frozen,samples:64}),/64|支持范围/);
KHQuantMarketWorkspace.draftBacktest({symbol:'600519.SH'},frozen);
element('market-backtest-symbols').value=Array.from({length:11},(_,i)=>String(600001+i)).join(',');
await fire('market-backtest-form','submit',prevent);assert.equal(transfers.length,0);assert.match(element('market-backtest-state').textContent,/1.?10|10/);
element('market-backtest-symbols').value='600519, 600519.SH, 000001.SZ';await fire('market-backtest-form','submit',prevent);
assert.deepEqual(transfers[0].symbols,['600519.SH','000001.SZ']);
assert(calls.every(call=>call.method==='GET'));
''')


def test_same_day_backtest_is_rejected_before_account_draft_is_changed():
    run_js(r'''
KHQuantMarketWorkspace.draftBacktest({symbol:'600519.SH'},frozen);
element('market-backtest-start').value='2026-10-15';element('market-backtest-end').value='2026-10-15';
await fire('market-backtest-form','submit',prevent);assert.equal(transfers.length,0);
assert.match(element('market-backtest-state').textContent,/开始|结束|日期/);
''')


def test_legacy_reference_context_cannot_silently_become_structure_backtest():
    run_js(r'''
const legacy={analysis_method:'legacy_reference',strategy_version:'reference_position_v1',research:true,request:{research:true,model_family:'current',model_run_id:'old-reference',usage_mode:'historical',end:'2026-10-15'}};
const context=workbench.resultResearchContext(legacy);
assert.throws(()=>KHQuantMarketWorkspace.draftBacktest({symbol:'600519.SH',data_end:'2026-10-15'},context),/历史|参考|方法/);
assert.equal(transfers.length,0);assert.equal(tabs.length,0);
''')


def test_saved_backtests_use_actual_sources_and_timestamp_and_only_read_metadata():
    run_js(r'''
answer=async path=>path==='/api/light/saved'?{results:[
 {cache_key:'saved+one',pin_count:1,created_at:1792126800,identity:{start:'2026-01-01',end:'2026-10-15',sources:[{symbol:'600519.SH'},{symbol:'000001.SZ'}]}},
 {cache_key:'preview',pin_count:0,identity:{start:'2026-01-01',end:'2026-10-15'}}
]}:{runs:[]};
KHQuantMarketWorkspace.onTab('backtest');await settle();const html=element('market-backtest-results').innerHTML;
assert(html.includes('600519.SH'));assert(html.includes('000001.SZ'));assert(!html.includes('preview'));
assert(html.includes('result=saved%2Bone'));assert(!html.includes('1792126800'));
assert.equal(element('market-backtest-end').value,'2026-10-15');assert.equal(transfers.length,0);assert(calls.every(call=>call.method==='GET'));
''')


def test_three_market_tabs_keep_one_visible_panel_and_clear_board_query():
    run_js(r'''
const savedLocation=location;global.location={href:'http://127.0.0.1:8124/?board=1#scan',search:'?board=1',hash:'#scan'};
global.history={state:null,replaceState(state,title,url){const next=new URL(url,'http://127.0.0.1:8124');location.href=next.href;location.search=next.search;location.hash=next.hash;}};
let boardReads=0;globalThis.KHQuantBoardMethods={onRoute:()=>{}};
const source=fs.readFileSync(path.join(process.argv[1],'board_panel.js'),'utf8').replace("  if(!$('scan-board-panel'))return;",`\n refresh=async()=>{boardReads++;};globalThis.boardTabs={selectTab,showStock,showBacktest};return;\n`);
vm.runInThisContext(source);
for(const tab of ['board','backtest','stock','backtest']){boardTabs.selectTab(tab);for(const name of ['stock','backtest','board']){assert.equal(element('scan-'+name+'-panel').hidden,name!==tab);assert.equal(element('scan-'+name+'-tab').attributes['aria-selected'],String(name===tab));assert.equal(element('scan-'+name+'-tab').tabIndex,name===tab?0:-1);}assert.equal(element('start-scan').hidden,tab!=='stock');assert.equal(memory.get('khquant_scan_tab'),tab);}
boardTabs.showBacktest();assert.equal(location.search,'');boardTabs.showStock();assert.equal(element('scan-stock-panel').hidden,false);assert.equal(element('scan-backtest-panel').hidden,true);
location.hash='#watchlist?tab=accounts&result=saved%2Bkey';assert.deepEqual(KHQuantNavigation.parse(),{view:'watchlist',tab:'accounts',result:'saved+key'});
assert.equal(transfers.length,0);assert(calls.every(call=>call.method==='GET'));
''')
