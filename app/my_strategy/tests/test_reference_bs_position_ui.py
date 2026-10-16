"""Run shipped transformations on causal source events and risk recommendations."""
from pathlib import Path
import shutil
import subprocess
import pytest

STATIC = Path(__file__).resolve().parents[1] / 'web_dashboard/static/js'

def run_js(script):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required for shipped reference UI')
    result = subprocess.run([node, '-e', script, str(STATIC)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_source_sell_is_visible_independently_of_wait_and_final_buy():
    run_js(r'''
const assert=require('node:assert/strict'),p=require('path'),ui=require(p.join(process.argv[1],'chart_evidence.js')),m=require(p.join(process.argv[1],'chart_markers.js'));
const symbol='002436.SZ',date='2026-07-02',stamp=date+'T15:00:00+08:00';
const point={symbol,date,reference_only:true,reference_price:30,reference_price_date:date,first_available_at:stamp,available_at:stamp,confirmed_at:stamp,first_confirmed_at:stamp,confirmed:true,candidate_eligible:true,input_eligible:true,evidence_visible:true,source:'structure',source_label:'结构',action:'SELL',side:'SELL',event_id:'s1',consensus_action:'WAIT',consensus_qualified:false};
const final={...point,event_id:'b1',reference_layer:'consensus',action:'BUY',side:'BUY'};
const result={symbol,analysis_mode:'reference',as_of:date,source_reference_points:[point,{...point,event_id:'w1',source:'wyckoff',source_label:'量价'}],reference_points:[final]};
const selected=ui.referenceMarkerInputs(result);
assert.equal(selected.events.length,3);assert.equal(selected.events.filter(e=>e.reference_layer==='source'&&e.action==='SELL').length,2);
const grouped=m.build({bars:[{time:date,is_closed:true}],dailyBars:[{time:date}],frequency:'日线',symbol,events:selected.events,consensusVersion:'trend_consensus',asOf:date,up:'red',down:'green'});
assert.deepEqual(new Set(grouped.markers.map(e=>e.text)),new Set(['结S','量S','共B']));
assert.equal(grouped.groups.filter(g=>g.kind==='source_reference').length,2);
assert.equal(grouped.groups.find(g=>g.kind==='intention').action,'BUY');
assert(grouped.markers.filter(e=>e.text.endsWith('S')).every(e=>e.color==='green'&&e.shape==='arrowDown'));
const sourceGroup=grouped.groups.find(g=>g.kind==='source_reference'&&g.source_label==='结构');
const evidence={rows:[{date,symbol,analysis_mode:'reference',reference_close:30,reference_price_date:date,agents:{}}]};
const html=ui.popupHtml({evidence,groups:[sourceGroup],consensus:{version:'trend_consensus',rows:[{date,action:'WAIT',analysis_mode:'reference',agents:{}}]},barDate:date,frequency:'日线'});
assert(html.includes('结构来源 S'));assert(html.includes('S · 卖方参考'));
assert(!html.includes('同日最终共识未通过'));assert(html.includes('首次可用 '+stamp));
assert.equal(point.consensus_qualified,false);assert.equal(point.consensus_action,'WAIT');
assert(!html.includes('账户退出'));assert.equal(ui.referenceSelection({evidence,groups:[sourceGroup],barDate:date}).price,30);
''')


def test_source_confirmation_and_full_timestamp_cutoff_prevent_future_sell_leakage():
    run_js(r'''
const assert=require('node:assert/strict'),p=require('path'),ui=require(p.join(process.argv[1],'chart_evidence.js'));
const good={symbol:'000001.SZ',date:'2026-10-09',source:'structure',action:'SELL',event_id:'good',confirmed:true,reference_only:true,first_available_at:'2026-10-09T15:00:00+08:00',confirmed_at:'2026-10-09T15:00:00+08:00'};
const result={symbol:good.symbol,analysis_mode:'reference',as_of:'2026-10-09',source_reference_points:[good,{...good,event_id:'pending',confirmed:false},{...good,event_id:'foreign',symbol:'600000.SH'},{...good,event_id:'hidden',evidence_visible:false},{...good,event_id:'future-confirm',confirmed_at:'2026-10-12T15:00:00+08:00'},{...good,event_id:'future',date:'2026-10-12',first_available_at:'2026-10-12T15:00:00+08:00'},{...good,event_id:'anonymous',symbol:null}]};
assert.deepEqual(ui.referenceMarkerInputs(result).events.map(e=>e.event_id),['good']);
assert.equal(ui.referenceMarkerInputs({...result,as_of:'2026-10-09T14:59:59+08:00'}).events.length,0);
''')


def test_source_sell_projects_to_forming_week_month_without_backdating_confirmation():
    run_js(r'''
const assert=require('node:assert/strict'),p=require('path'),ui=require(p.join(process.argv[1],'chart_evidence.js')),m=require(p.join(process.argv[1],'chart_markers.js'));
const point={symbol:'000001.SZ',date:'2026-07-02',source:'structure',source_label:'结构',action:'SELL',event_id:'s',confirmed:true,reference_only:true,first_available_at:'2026-07-02T15:00:00+08:00'};
const result={symbol:point.symbol,analysis_mode:'reference',as_of:point.date,source_reference_points:[point]};
for(const frequency of ['周线','月线']) {
 const grouped=m.build({bars:[{time:point.date,is_closed:false}],dailyBars:[{time:'2026-07-01'},{time:point.date}],events:ui.referenceMarkerInputs(result).events,frequency,symbol:point.symbol,consensusVersion:'trend_consensus',asOf:point.date,up:'r',down:'g'});
 assert.equal(grouped.intentCount,1);assert.equal(grouped.groups[0].period_status,'forming');assert.equal(grouped.groups[0].daily_confirmed,true);assert.equal(grouped.mappings[0].original_date,point.date);assert.equal(grouped.markers[0].text,'结S');
 const early=m.build({bars:[{time:'2026-07-01',is_closed:false}],dailyBars:[{time:'2026-07-01'}],events:ui.referenceMarkerInputs(result).events,frequency,symbol:point.symbol,consensusVersion:'trend_consensus',asOf:'2026-07-01',up:'r',down:'g'});
 assert.equal(early.intentCount,0);
}
''')


def test_pending_position_has_risk_cap_without_invented_target_or_entry_probability():
    run_js(r'''
const assert=require('node:assert/strict'),p=require('path'),ui=require(p.join(process.argv[1],'chart_evidence.js'));
const risk={status:'available',date:'2026-10-09',reference_price:35,reference_price_date:'2026-10-09',stop_price:31.5,invalidation_price:32,stop_distance_fraction:.1,risk_budget_fraction:.01,risk_position_cap_fraction:.1,reference_price_band:{low:33,high:36,basis_date:'2026-10-09',reference_only:true,not_fill_or_order_range:true}};
const pending={status:'pending',target_position_fraction:null,current_position_fraction:.2,change_fraction:null,basis:'质量资格不足',reason_codes:['qualified_trend_model_unavailable']};
const record={reference_risk_plan:risk,position_reference:pending};
let html=ui.positionHtml(record);assert(html.includes('推荐待资格核验'));assert(html.includes('仓位上限'));assert(html.includes('10%'));assert(html.includes('当前参考仓位 20%'));assert(html.includes('33–36'));assert(html.includes('非成交区间'));assert(html.includes('上限不等于建议仓位'));assert(!html.includes('建议变化'));assert(!html.includes('盈利概率'));assert(!html.includes('目标仓位 10%'));
html=ui.positionHtml({...record,position_reference:{...pending,status:'reduce_exposure',target_position_fraction:0,change_fraction:-.2}});
assert(html.includes('最终共识S · 参考降至 0%'));assert(html.includes('建议变化 -20 个百分点'));
const qualified={...pending,quality_status:'scored',reference_quality_probability:.8,reference_quality_threshold:.1,reference_buy_total_score:.58,reference_buy_total_threshold:.5,status:'increase_exposure',action:'BUY',target_position_fraction:.08,current_position_fraction:.02,change_fraction:.06};
html=ui.positionHtml({...record,position_reference:qualified});assert(html.includes('最终共识B · 参考目标 8%'));assert(html.includes('建议变化 6 个百分点'));
for(const bad of [{...qualified,target_position_fraction:.11},{...qualified,target_position_fraction:-.01},{...qualified,target_position_fraction:NaN},{...qualified,action:'WAIT'},{...qualified,status:'pending'}]) {
 const rejected=ui.positionHtml({...record,position_reference:bad});assert(!rejected.includes('参考目标'));assert(!rejected.includes('建议变化'));
}
html=ui.positionHtml({...record,reference_risk_plan:{...risk,status:'unavailable'},position_reference:{...pending,status:'unavailable'}});
assert(html.includes('风险参考不可用'));assert(!html.includes('风险仓位上限 10%'));assert(!html.includes('33–36'));
''')


def test_reference_shell_removed_account_and_backtest_handlers_and_keeps_binding_targets():
    from my_strategy.web_dashboard.scripts.validate_czsc_workbench import validate
    assert set(validate()['routes']) == {'analysis','scan','watchlist','training','parameters','data','simulation'}
    html=(STATIC.parent/'index.html').read_text()
    js=(STATIC/'workbench.js').read_text()
    for removed in ['view-backtest','view-holdings','send-backtest','id="trade-rows"','id="equity-chart"','id="backtest-form"','id="holdings-form"']:
        assert removed not in html and removed not in js
    # The current market backtest form drafts the independent simulation account.
    assert 'id="market-backtest-form"' in html
    run_js(r'''
const assert=require('node:assert/strict'),fs=require('fs'),vm=require('vm'),p=require('path');
const ids=new Set([...fs.readFileSync(p.join(process.argv[1],'../index.html'),'utf8').matchAll(/id="([^"]+)"/g)].map(m=>m[1]));
const nodes=new Map();const element=id=>{assert(ids.has(id),'dangling binding '+id);if(!nodes.has(id))nodes.set(id,{value:'',dataset:{},style:{},hidden:false,checked:false,addEventListener(){},querySelector:()=>({addEventListener(){}})});return nodes.get(id);};
global.document={getElementById:element,querySelectorAll:()=>[],addEventListener(){}};global.window={addEventListener(){}};
vm.runInThisContext(fs.readFileSync(p.join(process.argv[1],'workbench.js'),'utf8').replace('init().catch(showError);','globalThis.bindTest={bind,applyResearchContext,state};'));
bindTest.bind();
for(const prefix of ['analysis','scan']) {
 element(`${prefix}-mode`).value='research';element(`${prefix}-usage-mode`).value='historical';element(`${prefix}-model-policy`).value='auto';
 element(`${prefix}-model`).options=[{value:''}];element(`${prefix}-checkpoint`).options=[{value:''}];
 bindTest.state.research={models:[]};
 bindTest.applyResearchContext(prefix,{research:true,analysis_mode:'reference',model_family:'current',consensus_version:'reference_position_v1',usage_mode:'historical',model_policy:'auto',current_position_fraction:.5});
}
assert.equal(element('analysis-position').value,'50');
''')
